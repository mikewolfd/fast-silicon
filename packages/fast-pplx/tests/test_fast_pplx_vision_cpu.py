"""Offline CPU unit tests for `fast_pplx` visual encoding (`FastPplxGPU.encode_images`, `FastPplx.encode_images`, CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw
import pytest

from fast_pplx.cli import build_parser, main as pplx_cli_main
from fast_pplx.config import DEFAULT_PPLX_CONFIG
from fast_pplx.engine import FastPplx
from fast_pplx.gpu import FastPplxGPU
from fast_pplx.paths import resolve_paths
from fast_shared.qwen35_vision import FastQwen35VisionModel, Qwen35VisionConfig


def _make_tiny_vision_model(out_hidden_size: int = 1024) -> FastQwen35VisionModel:
    cfg = Qwen35VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=64,
    )
    vis = FastQwen35VisionModel(cfg)
    vis.set_dtype(mx.float16)
    mx.eval(vis.parameters())
    return vis


class _FakePplxModel:
    """Minimal backbone stub verifying MRoPE shapes + distinct `<|chunk_sep|>` retention."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def forward_hidden(
        self,
        hidden: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        seg_flags: Any = None,
        sdpa_mask: mx.array | None = None,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
    ) -> mx.array:
        B, S, D = hidden.shape
        assert rope_cos is not None and rope_sin is not None
        assert rope_cos.shape[:3] == (B, 1, S)
        assert rope_sin.shape[:3] == (B, 1, S)
        self.calls.append({"B": B, "S": S, "has_mask": sdpa_mask is not None})
        return hidden * mx.array(0.25, dtype=hidden.dtype)


class _SpyVisionWrapper:
    def __init__(self, inner: FastQwen35VisionModel) -> None:
        self.inner = inner
        self.wave_grids: list[list[tuple[int, int, int]]] = []

    def __call__(self, pixel_values: mx.array, grid_thw: list[tuple[int, int, int]]) -> mx.array:
        self.wave_grids.append(list(grid_thw))
        return self.inner(pixel_values, grid_thw)


def _make_stub_gpu(dim: int = 1024) -> tuple[FastPplxGPU, _SpyVisionWrapper, _FakePplxModel]:
    gpu = object.__new__(FastPplxGPU)
    gpu.config = DEFAULT_PPLX_CONFIG
    gpu.dtype_name = "float16"
    gpu.act_dtype = mx.float16
    gpu.max_batch_size = 8
    gpu.max_seq_len = 32768
    gpu.dim = dim
    gpu.max_dim = dim
    gpu.local_files_only = True
    gpu._mapped_embed = None
    gpu._prev_wired_limit = None
    gpu._vision_dir = None

    spy_vis = _SpyVisionWrapper(_make_tiny_vision_model(out_hidden_size=dim))
    fake_model = _FakePplxModel(dim=dim)
    gpu._vision_model = spy_vis
    gpu.model = fake_model

    def _embed_inputs(input_ids_np: np.ndarray) -> mx.array:
        B, S = input_ids_np.shape
        arr = np.zeros((B, S, dim), dtype=np.float16)
        arr[..., 0] = (input_ids_np.astype(np.float32) % 97.0) * 0.01
        return mx.array(arr, dtype=mx.float16)

    def _prepare_batch(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray | None]:
        max_l = max(len(r) for r in rows)
        pad = np.zeros((len(rows), max_l), dtype=np.int32)
        mask = np.ones((len(rows), max_l), dtype=np.uint8)
        uniform = all(len(r) == max_l for r in rows)
        for i, r in enumerate(rows):
            pad[i, : len(r)] = r
            mask[i, len(r) :] = 0
        return pad, (None if uniform else mask)

    def _span_pool(hidden: mx.array, chunk_token_spans: np.ndarray) -> np.ndarray:
        h_np = np.asarray(hidden.astype(mx.float32))
        B, max_sp, _ = chunk_token_spans.shape
        out = np.zeros((B, max_sp, dim), dtype=np.float32)
        for b in range(B):
            for c in range(max_sp):
                s0, s1 = int(chunk_token_spans[b, c, 0]), int(chunk_token_spans[b, c, 1])
                if s1 > s0:
                    out[b, c] = h_np[b, s0:s1].mean(axis=0)
        return out

    gpu._embed_inputs = _embed_inputs  # type: ignore[method-assign]
    gpu._prepare_batch_inputs_and_mask = _prepare_batch  # type: ignore[method-assign]
    gpu._project_span_pool = _span_pool  # type: ignore[method-assign]
    return gpu, spy_vis, fake_model


def _make_doc_img(w: int, h: int, *, margin: int = 10) -> Image.Image:
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for y in range(margin, h - margin, 18):
        dr.rectangle([margin, y, w - margin, min(h - margin, y + 6)], fill=(10, 10, 10))
    return im


def test_gpu_encode_images_wave_grouping_order_mrl_and_dtypes() -> None:
    gpu, spy_vis, fake_model = _make_stub_gpu()
    assert gpu.encode_images([]) == []

    # Unsorted sizes: img0 == img2 size, img1 has distinct aspect ratio -> should batch [0, 2] then [1]
    img0 = _make_doc_img(256, 320)
    img1 = _make_doc_img(448, 224)
    img2 = _make_doc_img(256, 320)

    vecs = gpu.encode_images(
        [img0, img1, img2],
        chunks=4,
        batch_size=4,
        min_pixels=16384,
        max_pixels=120_000,
    )
    assert len(vecs) == 3
    assert len(spy_vis.wave_grids) == 2
    assert sorted(len(g) for g in spy_vis.wave_grids) == [1, 2]
    assert len(fake_model.calls) == 2

    # Order matches inputs (`img0` & `img2` have matching chunk count, `img1` differs unless capped to same bands)
    for m in vecs:
        assert m.dtype == np.float32
        assert m.shape[1] == 1024
        assert np.all(np.isfinite(m))
        np.testing.assert_allclose(np.linalg.norm(m, axis=-1), np.ones(m.shape[0]), atol=1e-4)

    # Single-chunk, int8, MRL dim=256, and crop_margins
    vecs_i8 = gpu.encode_images(
        [img0],
        chunks=1,
        dimensions=256,
        output_dtype="int8",
        crop_margins=True,
        min_pixels=16384,
        max_pixels=120_000,
    )
    assert len(vecs_i8) == 1
    assert vecs_i8[0].shape == (1, 256)
    assert vecs_i8[0].dtype == np.int8


def test_engine_lazy_vision_isolation_and_ane_error() -> None:
    class _StubTokenizer:
        config = DEFAULT_PPLX_CONFIG

        def encode_contextual_document(
            self, chunks: list[str], *, return_spans: bool = True, max_length: int = 32768
        ) -> tuple[list[int], list[tuple[int, int]]]:
            return [1, 2, 3, 4], [(1, 3)]

    class _MinimalGPU:
        config = DEFAULT_PPLX_CONFIG
        max_seq_len = 32768
        _vision_model = None

        def __init__(self) -> None:
            self.tokenizer = _StubTokenizer()
            self.img_calls = 0

        def encode_contextual_tokens(self, encoded_docs: Any, **kwargs: Any) -> list[np.ndarray]:
            return [np.ones((len(sp), 1024), dtype=np.float32) for _toks, sp in encoded_docs]

        def encode_tokens(self, token_lists: Any, **kwargs: Any) -> np.ndarray:
            return np.ones((len(token_lists), 1024), dtype=np.float32)

        def encode_images(self, images: Any, **kwargs: Any) -> list[np.ndarray]:
            self.img_calls += 1
            return [np.ones((2, 1024), dtype=np.float32) for _ in images]

        def close(self) -> None:
            pass

    min_gpu = _MinimalGPU()
    eng = FastPplx(mode="gpu", _gpu_backend=min_gpu, enable_ane=False)
    try:
        doc_vecs = eng.encode_documents(["hello world"], batch_size=1)
        assert doc_vecs.shape == (1, 1024)
        assert min_gpu._vision_model is None
        assert min_gpu.img_calls == 0

        img_vecs = eng.encode_images([_make_doc_img(192, 192)], chunks=2)
        assert len(img_vecs) == 1 and min_gpu.img_calls == 1
    finally:
        eng.close()

    # Forced ANE-only engine raises descriptive RuntimeError
    class _DummyAne:
        def close(self) -> None:
            pass

    ane_eng = FastPplx(mode="ane", _ane_backend=_DummyAne(), _tokenizer=_StubTokenizer())
    try:
        with pytest.raises(RuntimeError, match="mode='ane'"):
            ane_eng.encode_images([_make_doc_img(128, 128)])
    finally:
        ane_eng.close()


def test_gpu_resolve_vision_weights_dir_skips_text_only_dir(tmp_path: Path) -> None:
    text_dir = tmp_path / "text_only"
    text_dir.mkdir()
    (text_dir / "config.json").write_text(json.dumps({"vision_config": {"depth": 12}}), encoding="utf-8")
    # Write a minimal safetensors without `visual.*` keys
    import struct

    hdr = json.dumps({"model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode("utf-8")
    with open(text_dir / "model.safetensors", "wb") as fp:
        fp.write(struct.pack("<Q", len(hdr)))
        fp.write(hdr)
        fp.write(b"\x00" * 8)

    assert FastPplxGPU._has_vision_checkpoint(text_dir) is False

    vis_dir = tmp_path / "with_vision"
    vis_dir.mkdir()
    (vis_dir / "config.json").write_text(json.dumps({"vision_config": {"depth": 12}}), encoding="utf-8")
    vhdr = json.dumps({"model.visual.patch_embed.proj.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode("utf-8")
    with open(vis_dir / "model.safetensors", "wb") as fp:
        fp.write(struct.pack("<Q", len(vhdr)))
        fp.write(vhdr)
        fp.write(b"\x00" * 8)

    assert FastPplxGPU._has_vision_checkpoint(vis_dir) is True

    gpu = object.__new__(FastPplxGPU)
    gpu.local_files_only = True
    gpu._vision_dir = vis_dir
    gpu.paths = resolve_paths(tmp_path)
    assert gpu._resolve_vision_weights_dir() == vis_dir


def test_cli_embed_images_parser_and_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = build_parser()
    ns = parser.parse_args(
        ["embed-images", "a.png", "--chunks", "4", "--crop-margins", "--dtype", "int8", "--output", "out.npz"]
    )
    assert ns.command == "embed-images"
    assert ns.chunks == 4
    assert ns.crop_margins is True
    assert ns.dtype == "int8"

    img_p = tmp_path / "sample.png"
    _make_doc_img(192, 192).save(img_p)
    out_npz = tmp_path / "vecs.npz"

    class _FakeEng:
        dim = 1024

        def encode_images(self, images: list[Any], **kwargs: Any) -> list[np.ndarray]:
            assert len(images) == 1
            assert kwargs["chunks"] == 4
            assert kwargs["crop_margins"] is True
            assert kwargs["output_dtype"] == "float32"
            return [np.ones((4, 1024), dtype=np.float32)]

        def close(self) -> None:
            pass

    monkeypatch.setattr("fast_pplx.cli._make_engine", lambda _args: _FakeEng())
    rc = pplx_cli_main(["embed-images", str(img_p), "--chunks", "4", "--crop-margins", "--output", str(out_npz)])
    assert rc == 0
    assert out_npz.is_file()
    with np.load(out_npz) as npz:
        assert npz["embeddings"].shape == (4, 1024)
        assert list(npz["offsets"]) == [0, 4]


def test_normalize_rotations_cw_validation() -> None:
    from fast_pplx.gpu import normalize_rotations_cw

    assert normalize_rotations_cw(None, 3) == [0, 0, 0]
    assert normalize_rotations_cw(90, 3) == [90, 90, 90]
    assert normalize_rotations_cw(-90, 2) == [270, 270]
    assert normalize_rotations_cw(360, 2) == [0, 0]
    assert normalize_rotations_cw([0, 90, 180, 270], 4) == [0, 90, 180, 270]
    assert normalize_rotations_cw(np.asarray([90, -180], dtype=np.int32), 2) == [90, 180]
    assert normalize_rotations_cw(None, 0) == []

    for bad_rot, count in [
        (45, 1),
        (45, 0),
        (True, 1),
        ("90", 1),
        (90.0, 1),
        ([0, 90], 3),
        ([0], 0),
        ([0, 180, 15], 3),
        ([False, 90], 2),
    ]:
        with pytest.raises(ValueError):
            normalize_rotations_cw(bad_rot, count)


def test_encode_images_forwards_rotations_cw_and_rejects_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    import mlx.core as mx
    from PIL import Image, ImageDraw
    import fast_shared.vision_preprocess as vp

    # Asymmetric synthetic page so upright vs 90-degree CCW rotated are distinguishable.
    upright = Image.new("RGB", (224, 160), (255, 255, 255))
    draw = ImageDraw.Draw(upright)
    draw.rectangle((18, 20, 140, 38), fill=(15, 15, 15))
    draw.rectangle((24, 90, 96, 128), fill=(40, 40, 40))
    rotated_ccw_90 = upright.transpose(Image.Transpose.ROTATE_90)

    captured_items: list[vp.PplxImageItem] = []
    real_prepare = vp.prepare_pplx_image_item

    def _spy_prepare(*args: Any, **kwargs: Any) -> vp.PplxImageItem:
        item = real_prepare(*args, **kwargs)
        captured_items.append(item)
        return item

    monkeypatch.setattr(vp, "prepare_pplx_image_item", _spy_prepare)

    def _stub_vis(_px: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
        n_merged = sum((int(g[1]) // 2) * (int(g[2]) // 2) for g in grids)
        return mx.ones((n_merged, 128), dtype=mx.float32)

    gpu = object.__new__(FastPplxGPU)
    gpu.dim = 1024
    gpu.max_dim = 1024
    gpu.max_batch_size = 4
    gpu.act_dtype = mx.float32
    gpu.dtype_name = "float32"
    gpu.supports_dimensions = True
    gpu.is_mrl_trained = False
    gpu.config = SimpleNamespace(
        rope_dim=64,
        rope_theta=1000000.0,
        mrope_section=(11, 11, 10),
        hidden_size=128,
    )
    gpu._ensure_vision_model = lambda: _stub_vis  # type: ignore[method-assign]
    gpu._embed_inputs = lambda input_ids: mx.zeros(  # type: ignore[method-assign]
        (int(input_ids.shape[0]), int(input_ids.shape[1]), 128), dtype=mx.float32
    )
    gpu.model = SimpleNamespace(forward_hidden=lambda hidden_mm, **_kw: hidden_mm)
    gpu._project_span_pool = lambda out_h, span_arr: mx.ones(  # type: ignore[method-assign]
        (int(out_h.shape[0]), int(span_arr.shape[1]), 1024), dtype=mx.float32
    )

    # Trigger FastPplxGPU.encode_images directly with upright (rot=0) vs CCW 90 (rot=90).
    out = gpu.encode_images(
        [upright, rotated_ccw_90],
        chunks=4,
        crop_margins=True,
        rotations_cw=[0, 90],
        output_dtype="float32",
    )
    assert len(out) == 2
    assert len(captured_items) == 2
    assert np.array_equal(captured_items[0].vision_inputs.pixel_values, captured_items[1].vision_inputs.pixel_values)
    assert captured_items[0].vision_inputs.grid_thw == captured_items[1].vision_inputs.grid_thw
    assert np.array_equal(captured_items[0].position_ids, captured_items[1].position_ids)
    assert captured_items[0].chunk_token_spans == captured_items[1].chunk_token_spans

    # Conflicting prebuilt PplxImageItem with non-zero rotation raises cleanly.
    with pytest.raises(ValueError, match="PplxImageItem"):
        gpu.encode_images([captured_items[0]], rotations_cw=90)

    # FastPplx wrapper forwards normalized rotations_cw and rejects invalid length/angles.
    seen_rots: list[Any] = []

    class _RecordingGPU:
        _vision_model = None

        def encode_images(self, imgs: list[Any], **kwargs: Any) -> list[np.ndarray]:
            seen_rots.append(kwargs.get("rotations_cw"))
            return [np.ones((2, 1024), dtype=np.float32) for _ in imgs]

        def close(self) -> None:
            pass

    eng = FastPplx(mode="gpu", _gpu_backend=_RecordingGPU(), _tokenizer=SimpleNamespace(), enable_ane=False)
    try:
        eng.encode_images([upright, rotated_ccw_90], rotations_cw=90)
        assert seen_rots[-1] == [90, 90]
        with pytest.raises(ValueError):
            eng.encode_images([upright, rotated_ccw_90], rotations_cw=[0])
        with pytest.raises(ValueError):
            eng.encode_images([upright], rotations_cw=33)
    finally:
        eng.close()


def test_build_image_index_offsets_maxsim_and_edge_cases() -> None:
    from types import SimpleNamespace
    from fast_shared.retrieval_index import CompactRetrievalIndex

    dim = 16
    calls: list[dict[str, Any]] = []

    class _IndexGPU:
        _vision_model = None

        def encode_images(self, imgs: list[Any], **kwargs: Any) -> list[np.ndarray]:
            calls.append(kwargs)
            if not imgs:
                return []
            if len(imgs) == 2 and imgs[0] == "zero_band_case":
                return [
                    np.empty((0, dim), dtype=np.float32),
                    np.eye(2, dim, dtype=np.float32),
                ]
            # Three pages with distinct band counts (2, 3, 1) along orthogonal basis axes.
            p0 = np.zeros((2, dim), dtype=np.float32)
            p0[0, 0] = 1.0
            p0[1, 1] = 1.0
            p1 = np.zeros((3, dim), dtype=np.float32)
            p1[0, 2] = 1.0
            p1[1, 3] = 1.0
            p1[2, 4] = 1.0
            p2 = np.zeros((1, dim), dtype=np.float32)
            p2[0, 5] = 1.0
            return [p0, p1, p2][: len(imgs)]

        def close(self) -> None:
            pass

    eng = FastPplx(mode="gpu", _gpu_backend=_IndexGPU(), _tokenizer=SimpleNamespace(), enable_ane=False)
    try:
        idx = eng.build_image_index(
            ["doc0", "doc1", "doc2"],
            chunks=4,
            rotations_cw=[0, 90, 270],
            agg="maxsim",
        )
        assert isinstance(idx, CompactRetrievalIndex)
        assert idx.num_docs == 3
        assert idx.num_windows == 6
        assert idx.doc_offsets is not None
        assert idx.doc_offsets.dtype == np.int32
        assert list(idx.doc_offsets) == [0, 2, 5, 6]
        assert calls[-1]["crop_margins"] is True
        assert calls[-1]["rotations_cw"] == [0, 90, 270]
        assert calls[-1]["output_dtype"] == "float32"
        assert calls[-1]["normalize"] is True

        # Query matching band 1 of doc1 (basis dim 3) must rank doc1 first via MaxSim.
        q = np.zeros((1, dim), dtype=np.float32)
        q[0, 3] = 1.0
        top_idx, top_scores = idx.search(q, k=3)
        assert int(top_idx[0, 0]) == 1
        assert float(top_scores[0, 0]) > float(top_scores[0, 1])

        # Empty 0-band page matrix inserts a zero row so doc_offsets stays strictly increasing.
        idx_zero = eng.build_image_index(["zero_band_case", "nonempty"], chunks=2)
        assert idx_zero.num_docs == 2
        assert idx_zero.num_windows == 3
        assert idx_zero.doc_offsets is not None
        assert list(idx_zero.doc_offsets) == [0, 1, 3]

        # Empty image list returns empty valid index.
        idx_empty = eng.build_image_index([], chunks=2)
        assert idx_empty.num_docs == 0
        assert idx_empty.num_windows == 0
        assert idx_empty.doc_offsets is not None
        assert list(idx_empty.doc_offsets) == [0]

        with pytest.raises(ValueError, match="dimensions"):
            eng.build_image_index(["doc0"], dimensions=0)
    finally:
        eng.close()


def test_cli_embed_images_rotate_build_index_and_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fast_shared.retrieval_index import CompactRetrievalIndex

    img0 = tmp_path / "p0.png"
    img1 = tmp_path / "p1.png"
    _make_doc_img(192, 192).save(img0)
    _make_doc_img(192, 192).save(img1)

    recorded: dict[str, Any] = {}

    class _FakeEng:
        dim = 1024

        def encode_images(self, images: list[Any], **kwargs: Any) -> list[np.ndarray]:
            recorded["encode_rotations_cw"] = kwargs.get("rotations_cw")
            return [np.ones((2, 16), dtype=np.float32) for _ in images]

        def build_image_index(self, images: list[Any], **kwargs: Any) -> CompactRetrievalIndex:
            recorded["index_kwargs"] = kwargs
            vecs = np.zeros((4, 16), dtype=np.float32)
            vecs[0:2, 0] = 1.0
            vecs[2:4, 1] = 1.0
            return CompactRetrievalIndex(
                vecs,
                doc_offsets=np.array([0, 2, 4], dtype=np.int32),
                mode="int8",
                is_mrl_trained=False,
                agg="maxsim",
            )

        def encode_queries(self, queries: list[str], **_kwargs: Any) -> np.ndarray:
            recorded["queries"] = list(queries)
            q = np.zeros((len(queries), 16), dtype=np.float32)
            q[:, 1] = 1.0
            return q

        def close(self) -> None:
            pass

    monkeypatch.setattr("fast_pplx.cli._make_engine", lambda _args: _FakeEng())

    assert pplx_cli_main(["embed-images", str(img0), "--rotate-cw", "90"]) == 0
    assert recorded["encode_rotations_cw"] == 90

    rc = pplx_cli_main(
        [
            "embed-images",
            str(img0),
            str(img1),
            "--crop-margins",
            "--rotate-cw",
            "180",
            "--build-index",
            "--query",
            "second page target",
            "--top-k",
            "2",
        ]
    )
    assert rc == 0
    assert recorded["index_kwargs"]["rotations_cw"] == 180
    assert recorded["index_kwargs"]["crop_margins"] is True
    assert recorded["queries"] == ["second page target"]
    out_txt = capsys.readouterr().out
    assert "Built image index: docs=2 windows=4" in out_txt
    assert "#1: [1] p1.png" in out_txt
