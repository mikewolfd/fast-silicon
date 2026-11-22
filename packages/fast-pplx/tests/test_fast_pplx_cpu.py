"""CPU-only unit tests for fast-pplx config, paths, tokenizer, weights, server, Parquet, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace
import urllib.request

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fast_pplx import (
    ContextualTokenizer,
    DEFAULT_MODEL_NAME,
    DEFAULT_PPLX_CONFIG,
    FastPplxServer,
    FastPplxTokenizer,
    PplxConfig,
    RepoPaths,
    resolve_paths,
)
from fast_pplx.cli import build_parser, main as cli_main
from fast_pplx.gpu import (
    _build_sdpa_mask,
    _canonicalize_weights,
    _find_local_hf_snapshot,
)
from fast_pplx.parquet_pipeline import embed_parquet
from fast_pplx.paths import require_runtime_paths
from fast_shared.embeddings import MappedEmbedding, tile_aligned_slices


def _write_synthetic_tokenizer(path: Path) -> Path:
    vocab = {
        "<unk>": 0,
        "D": 1,
        "[": 2,
        "]": 3,
        "hello": 4,
        "world": 5,
        "chunk": 6,
        "one": 7,
        "two": 8,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|endoftext|>", "[Q] ", "[D] "])
    tok.add_tokens(["<|chunk_sep|>"])
    out = path / "tokenizer.json"
    tok.save(str(out))
    return out


def _write_synthetic_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> Path:
    offset = 0
    header: dict[str, dict[str, object]] = {}
    payloads: list[bytes] = []
    for name, (arr, dtype_tag) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": dtype_tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad_len = (-len(hdr_bytes)) % 8
    hdr_bytes = hdr_bytes + b" " * pad_len
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for chunk in payloads:
            fh.write(chunk)
    return path


def test_paths_resolution_and_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_PPLX_HOME", str(tmp_path))
    p = resolve_paths()
    assert isinstance(p, RepoPaths)
    assert p.home == tmp_path.resolve()
    assert p.weights_dir == tmp_path.resolve() / "artifacts" / "weights"
    assert p.coreml_dylib == tmp_path.resolve() / "build" / "libpplx_coreml_runner.dylib"
    assert p.io_dylib == tmp_path.resolve() / "build" / "libpplx_fast_io.dylib"
    assert p.ane_cli == tmp_path.resolve() / "build" / "pplx-ane-cli"

    with pytest.raises(FileNotFoundError, match="Missing exported weights"):
        require_runtime_paths(tmp_path)

    p.weights_npz.parent.mkdir(parents=True, exist_ok=True)
    p.weights_npz.write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match="Missing tokenizer"):
        require_runtime_paths(tmp_path)

    p.tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
    p.tokenizer_json.write_text("{}", encoding="utf-8")
    assert require_runtime_paths(tmp_path, require_ane=False) == p

    with pytest.raises(FileNotFoundError, match="Missing native ANE runner"):
        require_runtime_paths(tmp_path, require_ane=True)


def test_config_overrides_interval_and_handles_null_pad(tmp_path: Path) -> None:
    assert DEFAULT_PPLX_CONFIG.full_attention_interval == 2
    assert DEFAULT_PPLX_CONFIG.rope_theta == 10_000_000.0
    assert DEFAULT_PPLX_CONFIG.chunk_sep_token_id == 248079
    assert DEFAULT_PPLX_CONFIG.query_prefix_token_id == 248077

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "embedding_dim": 1024,
                "query_prefix": "[Q] ",
                "document_prefix": "[D] ",
                "boundary_marker": "<|chunk_sep|>",
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 12,
                    "intermediate_size": 3584,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "full_attention_interval": 4,
                    "layer_types": ["linear_attention", "full_attention"] * 6,
                    "pad_token_id": None,
                    "eos_token_id": 248044,
                    "rope_parameters": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000,
                        "mrope_section": [11, 11, 10],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PplxConfig.from_json(cfg_path)
    assert cfg.full_attention_interval == 2
    assert cfg.pad_token_id == 248044
    assert cfg.rope_dim == 64
    assert cfg.mrope_section == (11, 11, 10)


def test_synthetic_tokenizer_special_tokens_and_spans(tmp_path: Path) -> None:
    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)
    assert FastPplxTokenizer is ContextualTokenizer
    doc_prefix_id = tok.reserved_id("[D] ")

    q_ids = tok.encode_query("hello world")
    assert q_ids == [tok.query_prefix_id, 4, 5]

    # Document prefix [D]  is special=True, so encode_special_tokens=True splits literal prefix
    # rather than emitting reserved ID; user control tokens are neutralized too.
    d_ids = tok.encode_document("hello <|endoftext|> world")
    assert doc_prefix_id not in d_ids
    assert tok.pad_id not in d_ids

    c_ids, c_spans = tok.encode_contextual_document(["hello", "", "chunk two"], return_spans=True)
    assert c_ids.count(tok.chunk_sep_id) == 2
    assert c_spans[1] == (0, 0)
    assert c_ids[c_spans[0][0] : c_spans[0][1]] == [4]
    assert c_ids[c_spans[2][0] : c_spans[2][1]] == [6, 8]

    chunked = tok.chunk_text(
        "hello world chunk one two hello world",
        max_chunk_tokens=8,
        overlap=1,
        add_document_prefix=True,
    )
    assert len(chunked) >= 2
    assert all(len(ids) <= 8 for ids, _, _ in chunked)


def test_tile_aligned_slices_and_mapped_embedding(tmp_path: Path) -> None:
    assert tile_aligned_slices(256, 512, 64) == [(0, 256)]
    assert tile_aligned_slices(520, 512, 64) == [(0, 520)]
    assert tile_aligned_slices(1087, 512, 64) == [(0, 512), (512, 1087)]
    assert tile_aligned_slices(1088, 512, 64) == [(0, 512), (512, 1024), (1024, 1088)]

    f32 = np.arange(24, dtype=np.float32).reshape(6, 4)
    bf16_u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    st_path = _write_synthetic_safetensors(
        tmp_path / "embed.safetensors",
        {"language_model.embed_tokens.weight": (bf16_u16, "BF16")},
    )
    with MappedEmbedding(st_path, tensor_name="language_model.embed_tokens.weight") as mm:
        gathered = mm.gather(np.array([[2, 5]], dtype=np.int32), out_dtype="float16")
        assert gathered.shape == (1, 2, 4)
        assert gathered.dtype == np.float16
        np.testing.assert_allclose(gathered[0, 0], f32[2], atol=0.1)
        np.testing.assert_allclose(gathered[0, 1], f32[5], atol=0.1)


def test_canonicalize_weights_and_additive_sdpa_mask() -> None:
    raw = {
        "visual.patch_embed.weight": mx.ones((4, 4), dtype=mx.bfloat16),
        "language_model.embed_tokens.weight": mx.ones((16, 8), dtype=mx.bfloat16),
        "language_model.norm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "contextual_projection.weight": mx.eye(8, dtype=mx.bfloat16),
        "language_model.layers.0.input_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.0.post_attention_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.A_log": mx.ones((2,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.dt_bias": mx.ones((2,), dtype=mx.bfloat16),
        "language_model.layers.0.linear_attn.conv1d.weight": mx.ones((12, 1, 4), dtype=mx.bfloat16),
        "language_model.layers.0.mlp.gate_proj.weight": mx.ones((4, 8), dtype=mx.bfloat16),
        "language_model.layers.0.mlp.up_proj.weight": mx.full((4, 8), 2.0, dtype=mx.bfloat16),
        "language_model.layers.1.input_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.1.post_attention_layernorm.weight": mx.zeros((8,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.q_norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.k_norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.q_proj.weight": mx.ones((8, 8), dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.k_proj.weight": mx.full((4, 8), 2.0, dtype=mx.bfloat16),
        "language_model.layers.1.self_attn.v_proj.weight": mx.full((4, 8), 3.0, dtype=mx.bfloat16),
    }
    items, head_w = _canonicalize_weights(
        raw,
        num_layers=2,
        act_dtype=mx.float16,
        include_embed=False,
        norms_already_shifted=False,
    )
    wmap = dict(items)
    assert not any(k.startswith("visual.") or k == "embed_tokens.weight" for k in wmap)
    assert head_w.dtype == mx.float32
    assert np.allclose(np.array(wmap["norm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.0.input_layernorm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.1.self_attn.q_norm.weight"]), 1.0)
    assert np.allclose(np.array(wmap["layers.0.linear_attn.norm.weight"]), 0.0)
    assert wmap["layers.0.linear_attn.norm.weight"].dtype == mx.float32
    assert wmap["layers.0.linear_attn.A_log"].dtype == mx.float32
    assert wmap["layers.0.linear_attn.conv1d.weight"].shape == (12, 4, 1)
    assert wmap["layers.0.mlp.gate_up_proj.weight"].shape == (8, 8)
    assert wmap["layers.1.self_attn.qkv_proj.weight"].shape == (16, 8)

    mask = np.array([[True, True, False]], dtype=np.bool_)
    sdpa_f16 = _build_sdpa_mask(mask, dtype_name="float16")
    assert sdpa_f16.dtype == mx.float16
    assert sdpa_f16.shape == (1, 1, 1, 3)
    assert float(sdpa_f16[0, 0, 0, 0]) == 0.0
    assert float(sdpa_f16[0, 0, 0, 2]) == -65504.0


def test_cli_hw_check_and_server_parquet_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    for sub in ("setup-weights", "setup-ane", "embed", "bench", "parquet", "serve"):
        assert parser.parse_args([sub] if sub in ("setup-weights", "setup-ane", "bench", "serve") else (
            ["embed", "hi"] if sub == "embed" else ["parquet", "--input", "a.parquet", "--output", "b.parquet"]
        )).command == sub

    called = {"hw": 0}

    def _fake_hw_warn(*args, **kwargs) -> bool:
        called["hw"] += 1
        return False

    monkeypatch.setattr("fast_pplx.cli.warn_if_untested_hardware", _fake_hw_warn)
    src_dir = tmp_path / "src_hf"
    src_dir.mkdir()
    _write_synthetic_tokenizer(src_dir)
    (src_dir / "config.json").write_text("{}", encoding="utf-8")
    (src_dir / "model.safetensors").write_bytes(b"stub")
    home_dir = tmp_path / "home"
    rc = cli_main(["setup-weights", "--home", str(home_dir), "--model-dir", str(src_dir), "--local-files-only"])
    assert rc == 0
    assert called["hw"] == 1
    assert (home_dir / "artifacts" / "tokenizer" / "tokenizer.json").exists()
    assert (home_dir / "artifacts" / "weights" / "model" / "model.safetensors").exists()

    # Verify FastPplxServer and parquet_pipeline contract using a deterministic fake engine
    tok = ContextualTokenizer(home_dir / "artifacts" / "tokenizer" / "tokenizer.json", DEFAULT_PPLX_CONFIG)

    class _FakeEngine:
        dim = 8
        max_dim = 8
        supports_dimensions = True
        model_name = DEFAULT_MODEL_NAME
        max_seq_len = 512
        ane_available = False
        default_mode = "gpu"
        tokenizer = tok

        def encode_query(self, token_ids, *, dimensions=None, **_kwargs):
            d = dimensions or self.dim
            v = np.ones((d,), dtype=np.float32)
            v /= np.linalg.norm(v)
            return v

        def encode_corpus(self, token_lists, *, dimensions=None, **_kwargs):
            d = dimensions or self.dim
            out = np.zeros((len(token_lists), d), dtype=np.float32)
            for idx, ids in enumerate(token_lists):
                out[idx, idx % d] = float(max(1, len(ids)))
                out[idx] /= np.linalg.norm(out[idx])
            return out

    eng = _FakeEngine()
    with FastPplxServer(eng, host="127.0.0.1", port=0) as srv:
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
        assert health["model"] == DEFAULT_MODEL_NAME

        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/v1/embeddings",
            data=json.dumps({"input": ["hello world"], "dimensions": 4}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert len(payload["data"][0]["embedding"]) == 4

    in_pq = tmp_path / "in.parquet"
    out_pq = tmp_path / "out.parquet"
    pq.write_table(pa.table({"id": [1, 2], "text": ["hello world", "chunk one"]}), in_pq)
    summary = embed_parquet(eng, in_pq, out_pq, text_col="text", dimensions=4)
    assert summary.rows_emitted == 2


def test_fixture_tokenizer_all_16_cases_match_reference_npz() -> None:
    snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if snap is None:
        pytest.skip("Local HuggingFace snapshot not cached; skipping HF tokenizer parity test")
    fix_dir = Path(__file__).resolve().parent / "fixtures"
    cases = json.loads((fix_dir / "cases.json").read_text(encoding="utf-8"))
    ref = np.load(fix_dir / "reference.npz")

    cfg = PplxConfig.from_json(snap / "config.json")
    tok = ContextualTokenizer(snap, cfg)
    for case in cases:
        cid = case["id"]
        if case["task"] == "query":
            out = tok.encode_queries(case["documents"])
            np.testing.assert_array_equal(out["input_ids"], ref[f"{cid}/input_ids"], err_msg=cid)
        else:
            out = tok.encode_documents(case["documents"])
            np.testing.assert_array_equal(out["input_ids"], ref[f"{cid}/input_ids"], err_msg=cid)
            np.testing.assert_array_equal(out["chunk_token_spans"], ref[f"{cid}/spans"], err_msg=cid)


def test_context_window_packing_and_hetero2_cli_cpu(tmp_path: Path) -> None:
    from fast_pplx.engine import FastPplx
    from fast_pplx.parquet_pipeline import embed_contextual_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    doc_text = " ".join(["hello world chunk one"] * 10)
    grouped = tok.chunk_and_group_document(doc_text, chunk_tokens=10, context_tokens=24, overlap=2)
    assert len(grouped) >= 2
    for win_chunks, spans in grouped:
        assert len(win_chunks) == len(spans) >= 1
        if len(win_chunks) > 1:
            assert len(tok.encode_contextual_document(win_chunks, return_spans=False)) <= 24

    class _MockGPU:
        max_seq_len = 32768

        def encode_tokens(self, token_lists, *, batch_size=16, dimensions=None, output_dtype="float32", normalize=True):
            dim = dimensions or 1024
            return np.ones((len(token_lists), dim), dtype=np.float32) * 0.125

        def encode_contextual_tokens(self, docs, *, batch_size=8, dimensions=None, output_dtype="float32", normalize=True):
            dim = dimensions or 1024
            return [np.ones((len(spans), dim), dtype=np.float32) * 0.5 for _, spans in docs]

        def close(self):
            pass

    with FastPplx(home=tmp_path, mode="auto", _gpu_backend=_MockGPU(), _tokenizer=tok) as eng:
        assert eng.ane_available is False
        assert eng.default_mode == "gpu"
        in_pq = tmp_path / "ctx_in.parquet"
        out_pq = tmp_path / "ctx_out.parquet"
        pq.write_table(pa.table({"doc_id": ["d0"], "text": [doc_text]}), in_pq)
        summary = embed_contextual_parquet(
            eng, in_pq, out_pq, chunk_tokens=10, context_tokens=24, overlap=2, dimensions=128
        )
        assert summary.rows_emitted >= 4
        out_tbl = pq.read_table(out_pq)
        assert {"window_index", "chunk_index", "char_start", "char_end", "vector"} <= set(out_tbl.schema.names)

    ns = build_parser().parse_args(
        ["parquet", "--input", "in.parquet", "--output", "out.parquet", "--mode", "hetero2", "--context-tokens", "2048", "--chunk-tokens", "256"]
    )
    assert ns.mode == "hetero2"
    assert ns.context_tokens == 2048
    assert ns.chunk_tokens == 256


@pytest.mark.parametrize("mode", ["gpu", "ane", "hetero", "hetero2"])
def test_public_contextual_limit_and_chunk_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from fast_pplx.engine import FastPplx
    from fast_pplx.gpu import FastPplxGPU

    tok = ContextualTokenizer(_write_synthetic_tokenizer(tmp_path), DEFAULT_PPLX_CONFIG)
    encode = tok.encode_contextual_document
    calls = []

    def count_tokenizations(chunks, **kwargs):
        calls.append(kwargs["max_length"])
        return encode(chunks, **kwargs)

    monkeypatch.setattr(tok, "encode_contextual_document", count_tokenizations)

    # Use the GPU's batching, padding, span pooling and result ordering on the MLX CPU stream.
    class IdentityGPU(FastPplxGPU):
        def __init__(self):
            self.tokenizer = tok
            self.max_seq_len = 32768
            self.max_batch_size = 2
            self.dim = 1
            self.model = SimpleNamespace(contextual_projection=lambda x: x)

        def forward_hidden(self, input_ids, attention_mask):
            assert input_ids.shape[1] <= 10
            return mx.array(input_ids[..., None], dtype=mx.float32)

        def _project_span_pool(self, hidden, spans):
            with mx.stream(mx.cpu):
                return super()._project_span_pool(hidden, spans)

    # The bucket stand-in returns the mean token ID of each live span.
    class MeanANE:
        caps = (10,) if mode == "ane" else (7,)

        def run_buckets(self, buckets, **kwargs):
            return [
                (bucket, np.array([[bucket.token_ids[t.start:t.end].mean()]
                                   for t in bucket.targets if t.end > t.start], dtype=np.float32))
                for bucket in buckets
            ]

        def close(self):
            pass

    docs = [
        ["hello world one", "two one hello chunk world one", "two"],
        ["two"], [], ["chunk", "", "world"],
    ]
    # Cut the second span after three tokens. Its successor and each empty chunk keep their zero row.
    expected = [[16 / 3, 19 / 3, 0], [8], [], [6, 0, 5]]
    with FastPplx(home=tmp_path, mode=mode, _gpu_backend=IdentityGPU(), _ane_backend=MeanANE(), _tokenizer=tok) as eng:
        actual = eng.encode_contextual_chunks(docs, max_length=10, batch_size=2, mode=mode, dimensions=1, normalize=False)
        assert calls == [10] * len(docs)
        for vectors, means in zip(actual, expected, strict=True):
            np.testing.assert_allclose(vectors[:, 0], means, rtol=1e-6)
        if mode.startswith("hetero"):
            assert eng.last_stats["gpu_items"] == 1 and eng.last_stats["ane_items"] == 3


def test_streamed_gdn_sdpa_whole_in_proj_parity_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_pplx.gpu as pplx_gpu_mod
    from fast_pplx.gpu import PplxModel

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(99)
        cfg = PplxConfig(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=("linear_attention", "full_attention"),
        )
        model = PplxModel(cfg, has_embed_module=False)
        model.layers[0].linear_attn.use_kernel = False
        mx.eval(model.parameters())

        seq_len, valid_b1 = 97, 72
        x = mx.random.normal((2, seq_len, cfg.hidden_size))
        mask_np = np.ones((2, seq_len), dtype=np.bool_)
        mask_np[1, valid_b1:] = False
        ssm_mask = mx.array(mask_np, dtype=mx.bool_)
        sdpa_mask = _build_sdpa_mask(mask_np, dtype_name="float32")

        monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 10_000)
        whole = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

        monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 64)
        monkeypatch.setattr(pplx_gpu_mod, "_GDN_SLICE_LEN", 32)
        monkeypatch.setattr(pplx_gpu_mod, "_SDPA_SLICE_LEN", 32)
        monkeypatch.setattr(pplx_gpu_mod, "_SDPA_MIN_TAIL", 32)
        streamed = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

        mx.eval(whole, streamed)
        assert np.allclose(np.array(whole, dtype=np.float32), np.array(streamed, dtype=np.float32), atol=2e-4, rtol=1e-4)
    finally:
        mx.set_default_device(prev_dev)


def test_resolve_home_checks_artifacts_and_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_pplx.paths import resolve_home

    monkeypatch.delenv("FAST_PPLX_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)

    # Fake monorepo layout: <tmp>/work/fast-silicon/packages/fast-pplx/src/fast_pplx/paths.py
    pkg_dir = tmp_path / "work" / "fast-silicon" / "packages" / "fast-pplx"
    anchor = pkg_dir / "src" / "fast_pplx" / "paths.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# stub\n", encoding="utf-8")
    # Put pyproject.toml + native/ + build/ inside pkg_dir, WITHOUT artifacts/
    (pkg_dir / "pyproject.toml").write_text("[project]\nname='fast-pplx'\n", encoding="utf-8")
    (pkg_dir / "native").mkdir()
    (pkg_dir / "build").mkdir()

    # 1. Must NOT stop on native/ or pyproject.toml; falls back to ~/.cache/fast-pplx when nothing has artifacts/
    fake_home = tmp_path / "user_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    assert resolve_home(_anchor_file=anchor) == (fake_home / ".cache" / "fast-pplx").resolve()

    # 2. Populated ~/.cache/fast-pplx/artifacts resolves to cache dir while build_dir falls back to pkg_dir/build
    (fake_home / ".cache" / "fast-pplx" / "artifacts").mkdir(parents=True)
    p_cache = resolve_paths(_anchor_file=anchor)
    assert p_cache.home == (fake_home / ".cache" / "fast-pplx").resolve()
    assert p_cache.build_dir == (pkg_dir / "build").resolve()

    # 3. Sibling checkout <tmp>/work/fast-pplx/artifacts beats ~/.cache
    sib_dir = tmp_path / "work" / "fast-pplx"
    (sib_dir / "artifacts").mkdir(parents=True)
    assert resolve_home(_anchor_file=anchor) == sib_dir.resolve()

    # 4. FAST_ARTIFACTS_ROOT/fast-pplx/artifacts beats sibling checkout
    art_root = tmp_path / "shared_artifacts"
    (art_root / "fast-pplx" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(art_root))
    assert resolve_home(_anchor_file=anchor) == (art_root / "fast-pplx").resolve()

    # 5. Direct artifacts/ inside pkg_dir beats FAST_ARTIFACTS_ROOT
    (pkg_dir / "artifacts").mkdir()
    assert resolve_home(_anchor_file=anchor) == pkg_dir.resolve()


def test_tf32_telemetry_chip_gated_attention_and_fp32_norms(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from fast_pplx.gpu import (
        FastPplxGPU,
        PplxAttention,
        PplxRMSNorm,
        _detect_gpu_family_info,
        get_mlx_runtime_telemetry,
        normalize_qk,
    )

    # Importing fast_pplx / fast_pplx.gpu / fast_pplx.cli must NEVER force MLX_ENABLE_TF32=0 globally.
    clean_env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, fast_pplx, fast_pplx.gpu, fast_pplx.cli; "
            "assert os.environ.get('MLX_ENABLE_TF32') is None, os.environ.get('MLX_ENABLE_TF32')",
        ],
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, f"fast_pplx polluted MLX_ENABLE_TF32 on import: {probe.stderr}"

    tel = get_mlx_runtime_telemetry()
    assert "mlx_enable_tf32" in tel
    assert "is_apple10" in tel and "sdpa_slice_len" in tel

    # Apple9 (M4 Pro) -> 512; Apple10 (M5 26.2+) -> 1024; M5 pre-26.2 -> 512
    m4 = _detect_gpu_family_info({"device_name": "Apple M4 Pro", "architecture": "applegpu_g16s"}, "26.6")
    assert m4["is_apple10"] is False and m4["sdpa_slice_len"] == 512
    m5 = _detect_gpu_family_info({"device_name": "Apple M5", "architecture": "applegpu_g17g"}, "26.2")
    assert m5["is_apple10"] is True and m5["nax_sdpa_eligible"] is True and m5["sdpa_slice_len"] == 1024
    m5_old = _detect_gpu_family_info({"device_name": "Apple M5", "architecture": "applegpu_g17g"}, "26.0")
    assert m5_old["is_apple10"] is True and m5_old["nax_sdpa_eligible"] is False and m5_old["sdpa_slice_len"] == 512

    cfg = PplxConfig(
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, rope_dim=8,
        layer_types=("linear_attention", "full_attention"),
    )
    attn_m5 = PplxAttention(cfg, gpu_family=m5)
    attn_m4 = PplxAttention(cfg, gpu_family=m4)
    assert attn_m5.sdpa_slice_len == 1024
    assert attn_m4.sdpa_slice_len == 512

    # Bool mask must be rejected on PplxAttention fast path
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x_small = mx.ones((1, 4, 64), dtype=mx.float16)
        with pytest.raises(TypeError, match="SDPA mask"):
            attn_m5(x_small, sdpa_mask=mx.ones((1, 1, 4, 4), dtype=mx.bool_))
        # Causal string fast path works with zero array mask allocation
        out_causal = attn_m5(x_small, sdpa_mask="causal")
        mx.eval(out_causal)
        assert out_causal.shape == (1, 4, 64)

        # normalize_qk protects tiny FP16 inputs (where eps/128 < 2^-24) & large >256 inputs against overflow
        q_large = mx.full((1, 2, 2, 128), 512.0, dtype=mx.float16)
        k_tiny = mx.full((1, 2, 2, 128), 1e-4, dtype=mx.float16)
        qn, kn = normalize_qk(q_large, k_tiny, inv_scale=128.0 ** -0.5, eps=1e-6)
        mx.eval(qn, kn)
        assert np.all(np.isfinite(np.array(qn, dtype=np.float32)))
        assert np.all(np.isfinite(np.array(kn, dtype=np.float32)))
        assert qn.dtype == mx.float16 and kn.dtype == mx.float16

        # PplxRMSNorm protects deep FP16 residual outliers (>256) from x*x FP16 overflow
        norm = PplxRMSNorm(16, eps=1e-6)
        big_x = mx.full((1, 2, 16), 2048.0, dtype=mx.float16)
        normed = norm(big_x)
        mx.eval(normed)
        assert np.all(np.isfinite(np.array(normed, dtype=np.float32)))
        assert np.allclose(np.array(normed, dtype=np.float32), 1.0, atol=1e-2)

        # _embed_inputs gathers MappedEmbedding as float16 even for bfloat16 (no CPU float32 upcast)
        seen_dtypes: list[str] = []

        class _SpyMapped:
            def gather(self, ids, *, out_dtype="float16"):
                seen_dtypes.append(str(out_dtype))
                return np.ones((*ids.shape, 8), dtype=np.float16)

        stub = object.__new__(FastPplxGPU)
        stub._mapped_embed = _SpyMapped()
        stub.dtype_name = "bfloat16"
        stub.act_dtype = mx.bfloat16
        emb = stub._embed_inputs(np.array([[1, 2]], dtype=np.int32))
        mx.eval(emb)
        assert seen_dtypes == ["float16"]
        assert emb.dtype == mx.bfloat16
    finally:
        mx.set_default_device(prev_dev)


def test_parquet_int8_streaming_sharded_resume_and_cli_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_pplx.parquet_pipeline import embed_contextual_parquet, embed_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    in_pq = tmp_path / "docs.parquet"
    doc_a = " ".join(["hello world chunk one two"] * 6)
    doc_b = " ".join(["chunk two hello world one"] * 6)
    pq.write_table(pa.table({"doc_id": ["a", "b", "empty"], "text": [doc_a, doc_b, "   "]}), in_pq)

    encode_calls = {"ctx": 0, "flat": 0}

    class _Int8Engine:
        dim = 16
        max_dim = 16
        max_seq_len = 512
        supports_dimensions = True
        tokenizer = tok

        def encode_contextual_chunks(self, docs, *, dimensions=None, output_dtype="float32", normalize=True, mode=None):
            encode_calls["ctx"] += 1
            d = dimensions or self.dim
            dt = {"int8": np.int8, "float16": np.float16, "float32": np.float32}[output_dtype]
            return [np.full((len(chunks), d), 42 if output_dtype == "int8" else 0.5, dtype=dt) for chunks in docs]

        def encode_corpus(self, tok_lists, *, dimensions=None, output_dtype="float32", normalize=True, mode=None):
            encode_calls["flat"] += 1
            d = dimensions or self.dim
            dt = {"int8": np.int8, "float16": np.float16, "float32": np.float32}[output_dtype]
            return np.full((len(tok_lists), d), -17 if output_dtype == "int8" else 0.25, dtype=dt)

        def close(self):
            pass

    eng = _Int8Engine()

    # 1. Contextual single-file int8 output
    out_ctx_file = tmp_path / "ctx_i8.parquet"
    s1 = embed_contextual_parquet(
        eng, in_pq, out_ctx_file, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", skip_empty=True,
    )
    assert s1.rows_read == 3
    assert s1.rows_emitted >= 4
    tbl = pq.read_table(out_ctx_file)
    assert tbl.schema.field("vector").type == pa.list_(pa.int8(), 8)
    assert all(v == 42 for v in tbl["vector"][0].as_py())

    # 2. Contextual sharded directory + --resume skips completed shards without re-encoding
    out_shards = tmp_path / "ctx_shards"
    encode_calls["ctx"] = 0
    s2 = embed_contextual_parquet(
        eng, in_pq, out_shards, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", shard_rows=2, resume=False,
    )
    shards = sorted(out_shards.glob("part-*.parquet"))
    assert len(shards) >= 2
    first_calls = encode_calls["ctx"]
    assert first_calls >= 2

    encode_calls["ctx"] = 0
    _ = embed_contextual_parquet(
        eng, in_pq, out_shards, chunk_tokens=10, context_tokens=24, overlap=2,
        dimensions=8, dtype="int8", shard_rows=2, resume=True,
    )
    assert encode_calls["ctx"] == 0  # Every existing shard matched row count and was skipped!

    # 3. Non-contextual embed_parquet supports int8 streaming
    out_flat_i8 = tmp_path / "flat_i8.parquet"
    s3 = embed_parquet(eng, in_pq, out_flat_i8, text_col="text", dimensions=8, dtype="int8", skip_empty=True)
    assert s3.rows_emitted == 2
    flat_tbl = pq.read_table(out_flat_i8)
    assert flat_tbl.schema.field("vector").type == pa.list_(pa.int8(), 8)
    assert all(v == -17 for v in flat_tbl["vector"][0].as_py())

    # 4. CLI forwards --resume, --shard-rows, --encode-batch-rows, --skip-empty, --query-prefix, --dtype int8
    captured: dict[str, object] = {}

    def _spy_ctx(*args, **kwargs):
        captured.update(kwargs)
        from fast_shared.parquet_pipeline import ParquetEmbedSummary

        return ParquetEmbedSummary(1, 1, 1, 10, 0.01, tmp_path / "out")

    monkeypatch.setattr("fast_pplx.cli.warn_if_untested_hardware", lambda *a, **k: False)
    monkeypatch.setattr("fast_pplx.cli._make_engine", lambda a, **kw: eng)
    monkeypatch.setattr("fast_pplx.cli.embed_contextual_parquet", _spy_ctx)

    rc = cli_main([
        "parquet", "--input", str(in_pq), "--output", str(tmp_path / "cli_out"),
        "--context-tokens", "2048", "--chunk-tokens", "256", "--dtype", "int8",
        "--resume", "--shard-rows", "128", "--encode-batch-rows", "64",
        "--skip-empty", "--query-prefix",
    ])
    assert rc == 0
    assert captured["resume"] is True
    assert captured["shard_rows"] == 128
    assert captured["encode_batch_rows"] == 64
    assert captured["skip_empty"] is True
    assert captured["add_query_prefix"] is True
    assert captured["dtype"] == "int8"

    # Verify new `verify` and upgraded `bench` CLI flags parse cleanly
    p = build_parser()
    assert p.parse_args(["verify", "--mode", "gpu", "--json"]).command == "verify"
    b_ns = p.parse_args(["bench", "--mode", "all", "--runs", "2", "--timing-scope", "e2e"])
    assert b_ns.mode == "all" and b_ns.runs == 2 and b_ns.timing_scope == "e2e"


def test_thread_safe_last_stats_and_hetero_vs_hetero2_deterministic(tmp_path: Path) -> None:
    import time as _time
    from fast_pplx.engine import FastPplx

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    class _MockGPU:
        max_seq_len = 32768

        def encode_tokens(self, token_lists, *, batch_size=16, dimensions=None, output_dtype="float32", normalize=True):
            return np.ones((len(token_lists), dimensions or 16), dtype=np.float32)

        def close(self):
            pass

    class _MockANE:
        caps = (64, 128)
        max_cap = 128

        def run_buckets(self, buckets, *, out_dim, apply_tanh_int8, apply_l2_norm):
            _time.sleep(0.005)
            out = []
            for bkt in buckets:
                live = [t for t in bkt.targets if t.end > t.start]
                out.append((bkt, np.ones((len(live), out_dim), dtype=np.float32) * 2.0))
            return out

        def close(self):
            pass

    # 1. `hetero` is deterministic static partition (zero stealing); `hetero2` steals
    with FastPplx(home=tmp_path, mode="hetero", _gpu_backend=_MockGPU(), _ane_backend=_MockANE(), _tokenizer=tok) as eng:
        items = [[4, 5, 6] for _ in range(20)] + [[7] * 200]
        _ = eng.encode_tokens(items, dimensions=16, mode="hetero")
        st_h = eng.last_stats
        assert st_h["deterministic"] is True
        assert st_h["gpu_stolen_items"] == 0
        assert st_h["ane_items"] == 20 and st_h["gpu_items"] == 1

        # 2. Thread-local last_stats isolation across concurrent threads
        results_by_thread: dict[str, dict[str, object]] = {}

        def _worker(tag: str, mode_name: str, n_items: int) -> None:
            _ = eng.encode_tokens([[4, 5]] * n_items, dimensions=16, mode=mode_name)
            _time.sleep(0.01)
            results_by_thread[tag] = eng.last_stats

        t1 = threading.Thread(target=_worker, args=("gpu_t", "gpu", 3))
        t2 = threading.Thread(target=_worker, args=("ane_t", "ane", 5))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert results_by_thread["gpu_t"]["mode"] == "gpu"
        assert results_by_thread["gpu_t"]["gpu_items"] == 3
        assert results_by_thread["ane_t"]["mode"] == "ane"
        assert results_by_thread["ane_t"]["ane_items"] == 5

    # 3. deterministic=True routes auto / hetero2 strictly to deterministic gpu
    with FastPplx(
        home=tmp_path, mode="auto", deterministic=True,
        _gpu_backend=_MockGPU(), _ane_backend=_MockANE(), _tokenizer=tok,
    ) as eng_det:
        assert eng_det.default_mode == "gpu"
        _ = eng_det.encode_tokens([[4, 5, 6]], dimensions=16, mode="hetero2")
        assert eng_det.last_stats["mode"] == "gpu"
        assert eng_det.last_stats["deterministic"] is True


def test_max_length_validation_and_hetero2_contextual_work_stealing_cpu(tmp_path: Path) -> None:
    """Verify `max_length <= 0` raises `ValueError` and `hetero2` contextual work-stealing preserves order & limits."""
    from fast_pplx.engine import FastPplx
    from fast_pplx.gpu import FastPplxGPU

    tok = ContextualTokenizer(_write_synthetic_tokenizer(tmp_path), DEFAULT_PPLX_CONFIG)
    gpu_stole_evt = threading.Event()
    ane_first_call = {"done": False}
    ane_lock = threading.Lock()
    gpu_calls = {"n": 0}

    class _IdentityGPU(FastPplxGPU):
        def __init__(self) -> None:
            self.tokenizer = tok
            self.max_seq_len = 32768
            self.max_batch_size = 4
            self.dim = 1
            self.model = SimpleNamespace(contextual_projection=lambda x: x)

        def forward_hidden(self, input_ids, attention_mask):
            assert input_ids.shape[1] <= 10
            return mx.array(input_ids[..., None], dtype=mx.float32)

        def _project_mean_pool(self, hidden, attention_mask):
            with mx.stream(mx.cpu):
                return super()._project_mean_pool(hidden, attention_mask)

        def _project_span_pool(self, hidden, spans):
            with mx.stream(mx.cpu):
                return super()._project_span_pool(hidden, spans)

        def encode_contextual_tokens(self, encoded_docs, **kwargs):
            out = super().encode_contextual_tokens(encoded_docs, **kwargs)
            gpu_calls["n"] += 1
            # Any call with a short (<=7 tok) doc means GPU has stolen work from short_dq
            if any(len(ids) <= 7 for ids, _ in encoded_docs):
                gpu_stole_evt.set()
            return out

    class _MeanANE:
        caps = (7,)

        def run_buckets(self, buckets, **kwargs):
            with ane_lock:
                is_first = not ane_first_call["done"]
                ane_first_call["done"] = True
            if is_first:
                # Hold ANE's first chunk until GPU steals at least one short batch (bounded wait, no sleep)
                gpu_stole_evt.wait(timeout=5.0)
            return [
                (
                    bucket,
                    np.array(
                        [[bucket.token_ids[t.start : t.end].mean()] for t in bucket.targets if t.end > t.start],
                        dtype=np.float32,
                    ),
                )
                for bucket in buckets
            ]

        def close(self) -> None:
            pass

    gpu = _IdentityGPU()
    # 1. Direct GPU and FastPplx validation on max_length <= 0
    for bad_ml in (0, -1, -32):
        with pytest.raises(ValueError, match="max_length must be > 0"):
            gpu.encode(["hello world"], max_length=bad_ml, dimensions=1, normalize=False)
        with pytest.raises(ValueError, match="max_length must be > 0"):
            gpu.encode_contextual_chunks([["hello", "world"]], max_length=bad_ml, dimensions=1, normalize=False)

    with FastPplx(home=tmp_path, mode="hetero2", _gpu_backend=gpu, _ane_backend=_MeanANE(), _tokenizer=tok) as eng:
        for bad_ml in (0, -1, -16):
            with pytest.raises(ValueError, match="max_length must be > 0"):
                eng.encode(["hello world"], max_length=bad_ml, dimensions=1, normalize=False, mode="gpu")
            with pytest.raises(ValueError, match="max_length must be > 0"):
                eng.encode_contextual_chunks(
                    [["hello", "world"]], max_length=bad_ml, dimensions=1, normalize=False, mode="gpu"
                )
            with pytest.raises(ValueError, match="max_length must be > 0"):
                eng.encode_texts(["hello world"], max_length=bad_ml, dimensions=1, normalize=False, mode="gpu")

        # max_length=None uses self.max_seq_len cleanly
        none_ok = eng.encode(["hello"], max_length=None, dimensions=1, normalize=False, mode="gpu")
        assert none_ok.shape == (1, 1)

        # 2. >= 6-document hetero2 contextual work-stealing (16 short + 1 long = 17 docs, steal_min_chunk=4)
        short_templates: list[tuple[list[str], list[float]]] = [
            (["hello"], [4.0]),
            (["world"], [5.0]),
            (["chunk", "", "world"], [6.0, 0.0, 5.0]),
            (["two"], [8.0]),
            ([], []),
            (["one", "two"], [7.0, 8.0]),
            (["hello", "one"], [4.0, 7.0]),
            (["chunk", "two"], [6.0, 8.0]),
        ]
        docs: list[list[str]] = [
            ["hello world one", "two one hello chunk world one", "two"],  # long (>7 toks, truncated to 10)
        ]
        expected: list[list[float]] = [[16 / 3, 19 / 3, 0.0]]
        for idx in range(16):
            d_chunks, exp_means = short_templates[idx % len(short_templates)]
            docs.append(d_chunks)
            expected.append(exp_means)

        actual = eng.encode_contextual_chunks(
            docs, max_length=10, batch_size=4, mode="hetero2", dimensions=1, normalize=False
        )
        assert len(actual) == len(docs) == 17
        for vectors, means in zip(actual, expected, strict=True):
            assert vectors.shape == (len(means), 1)
            np.testing.assert_allclose(vectors[:, 0], means, rtol=1e-6)

        st = eng.last_stats
        assert st["mode"] == "hetero2"
        assert st["deterministic"] is False
        assert st["gpu_stolen_items"] >= 4
        assert st["ane_items"] > 0
        assert st["ane_items"] + st["gpu_items"] == len(docs)


def test_horizontal_fusion_in_proj_all_and_qk_rope_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-P1.1: Fuse in_proj_{qkv,z,b,a} -> in_proj_all [8224, 1024] and [Q, K] norm + RoPE."""
    from fast_pplx.gpu import PplxAttention, PplxGatedDeltaNet

    gdn_default = PplxGatedDeltaNet(DEFAULT_PPLX_CONFIG)
    assert hasattr(gdn_default, "in_proj_all")
    assert gdn_default.in_proj_all.weight.shape == (8224, 1024)
    assert not hasattr(gdn_default, "in_proj_a")
    assert not hasattr(gdn_default, "in_proj_b")

    raw = {
        "contextual_projection.weight": mx.eye(8, dtype=mx.float32),
        "language_model.layers.0.linear_attn.in_proj_qkv.weight": mx.ones((12, 8), dtype=mx.float16),
        "language_model.layers.0.linear_attn.in_proj_z.weight": mx.full((4, 8), 2.0, dtype=mx.float16),
        "language_model.layers.0.linear_attn.in_proj_b.weight": mx.full((2, 8), 3.0, dtype=mx.float16),
        "language_model.layers.0.linear_attn.in_proj_a.weight": mx.full((2, 8), 4.0, dtype=mx.float16),
    }
    items, _ = _canonicalize_weights(raw, num_layers=1, act_dtype=mx.float16, include_embed=False)
    wmap = dict(items)
    assert "layers.0.linear_attn.in_proj_all.weight" in wmap
    assert wmap["layers.0.linear_attn.in_proj_all.weight"].shape == (20, 8)
    assert "layers.0.linear_attn.in_proj_a.weight" not in wmap

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        cfg = PplxConfig(
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            rope_dim=8,
            layer_types=("linear_attention", "full_attention"),
        )
        attn = PplxAttention(cfg)
        attn.q_norm.weight = mx.linspace(0.8, 1.2, 32, dtype=mx.float32)
        attn.k_norm.weight = mx.linspace(0.9, 1.1, 32, dtype=mx.float32)
        mx.eval(attn.parameters())

        rope_calls = {"n": 0}
        orig_rope = mx.fast.rope

        def _spy_rope(*args, **kwargs):
            rope_calls["n"] += 1
            return orig_rope(*args, **kwargs)

        monkeypatch.setattr(mx.fast, "rope", _spy_rope)
        x = mx.random.normal((2, 9, 64)).astype(mx.float16)
        y = attn(x)
        mx.eval(y)
        assert rope_calls["n"] == 1, f"Expected 1 fused [Q,K] RoPE call per layer, got {rope_calls['n']}"
        assert y.shape == (2, 9, 64)
    finally:
        mx.set_default_device(prev_dev)


def test_context_window_header_anchoring_and_single_tokenization_parquet_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-P1.1: Default max_context_tokens=2048 + header anchoring + zero double-tokenization in Parquet."""
    import inspect
    from fast_pplx.engine import FastPplx
    from fast_pplx.parquet_pipeline import embed_contextual_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    sig = inspect.signature(tok.group_chunks_by_context_windows)
    assert "max_context_tokens" in sig.parameters
    assert sig.parameters["max_context_tokens"].default == 2048

    chunks = ["hello world", "chunk one two", "hello chunk one", "world two one", "chunk two hello"]
    wins_hdr = tok.group_chunks_by_context_windows(chunks, max_context_tokens=12, header_text="hello")
    assert len(wins_hdr) >= 2
    for w in wins_hdr:
        assert w[0] == "hello"
        assert len(tok.encode_contextual_document(w, return_spans=False)) <= 12

    wins_rep = tok.group_chunks_by_context_windows(chunks, max_context_tokens=12, repeat_header_tokens=1)
    assert len(wins_rep) >= 2
    assert wins_rep[1][0] == "hello"
    for w in wins_rep:
        assert len(tok.encode_contextual_document(w, return_spans=False)) <= 12

    # Verify embed_contextual_parquet does NOT double-call encode_contextual_document per window
    orig_enc = tok.encode_contextual_document
    enc_calls = {"n": 0}

    def _count_enc(*args, **kwargs):
        enc_calls["n"] += 1
        return orig_enc(*args, **kwargs)

    monkeypatch.setattr(tok, "encode_contextual_document", _count_enc)

    class _MockGPU:
        max_seq_len = 32768

        def encode_contextual_tokens(self, docs, *, batch_size=8, dimensions=None, output_dtype="float32", normalize=True):
            d = dimensions or 1024
            return [np.ones((len(spans), d), dtype=np.float32) * 0.5 for _, spans in docs]

        def close(self):
            pass

    in_pq = tmp_path / "single_enc_in.parquet"
    out_pq = tmp_path / "single_enc_out.parquet"
    doc_text = " ".join(["hello world chunk one two"] * 8)
    expected_windows = len(tok.chunk_and_group_document(doc_text, chunk_tokens=10, context_tokens=20, overlap=2))
    enc_calls["n"] = 0
    pq.write_table(pa.table({"doc_id": ["d0"], "text": [doc_text]}), in_pq)

    with FastPplx(home=tmp_path, mode="gpu", _gpu_backend=_MockGPU(), _tokenizer=tok) as eng:
        summary = embed_contextual_parquet(
            eng, in_pq, out_pq, chunk_tokens=10, context_tokens=20, overlap=2, dimensions=16
        )
    assert summary.rows_emitted >= expected_windows
    assert enc_calls["n"] == expected_windows, (
        f"Expected 1x encode_contextual_document per window ({expected_windows}), got {enc_calls['n']}"
    )


def test_batched_varlen_waves_and_multifunction_plan_inspect_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify [B_bins, T_wave] varlen wave batching, GPU mask/RoPE helper, and per-function ANE plan inspection."""
    import fast_pplx.gpu as pplx_gpu_mod
    from fast_pplx.gpu import FastPplxGPU, PplxModel, _build_varlen_mask_and_rope

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(77)
        cfg = PplxConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=("linear_attention", "full_attention"),
        )
        model = PplxModel(cfg, has_embed_module=True)
        model.layers[0].linear_attn.use_kernel = False
        mx.eval(model.parameters())

        # Direct check on _build_varlen_mask_and_rope
        doc_mx = mx.array([[0, 0, -1, 1, 1, -1]], dtype=mx.int32)
        pos_mx = mx.array([[0, 1, 0, 0, 1, 0]], dtype=mx.int32)
        cos_mx = mx.ones((16, 8), dtype=mx.float16)
        sin_mx = mx.zeros((16, 8), dtype=mx.float16)
        msk, rc, rs = _build_varlen_mask_and_rope(
            doc_mx, pos_mx, cos_mx, sin_mx, mx.array(0.0, dtype=mx.float16), mx.array(-65504.0, dtype=mx.float16)
        )
        mx.eval(msk, rc, rs)
        msk_np = np.array(msk[0, 0], dtype=np.float32)
        assert msk_np[0, 1] == 0.0 and msk_np[0, 3] == -65504.0
        assert msk_np[2, 2] == 0.0 and msk_np[2, 0] == -65504.0
        assert rc.shape == (1, 1, 6, 8) and rs.shape == (1, 1, 6, 8)

        # Verify _run_varlen_bins groups multiple packed bins into ONE [B_bins, T_wave] forward_hidden call
        gpu = FastPplxGPU.__new__(FastPplxGPU)
        gpu.config = cfg
        gpu.model = model
        proj_dim = int(model.contextual_projection.weight.shape[0])
        gpu.dim = proj_dim
        gpu.max_seq_len = 2048
        gpu.max_batch_size = 8
        gpu.dtype_name = "float32"
        gpu.act_dtype = mx.float32
        gpu._mapped_embed = None
        gpu.tokenizer = SimpleNamespace(pad_id=0)

        fwd_shapes: list[tuple[int, int]] = []
        orig_fwd = model.forward_hidden

        def _spy_fwd(h, *a, **kw):
            fwd_shapes.append((int(h.shape[0]), int(h.shape[1])))
            return orig_fwd(h, *a, **kw)

        monkeypatch.setattr(model, "forward_hidden", _spy_fwd)

        tok_lists = [
            [5, 6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34],
            [35, 36, 37],
        ]
        bins = gpu._pack_varlen_token_bins(tok_lists, max_packed_tokens=20, align_to=8)
        assert len(bins) >= 2
        proj_bins = gpu._run_varlen_bins(bins, max_wave_tokens=256)
        assert len(fwd_shapes) == 1 and fwd_shapes[0][0] == len(bins), f"Expected 1 batched wave, got {fwd_shapes}"

        packed_out = np.zeros((len(tok_lists), proj_dim), dtype=np.float32)
        for b_spec, p_b in zip(bins, proj_bins, strict=True):
            for pi, oi in enumerate(b_spec["orig_indices"]):
                packed_out[oi] = p_b[pi]
        packed_norm = packed_out / np.linalg.norm(packed_out, axis=-1, keepdims=True)

        singles = []
        for row in tok_lists:
            h_s = orig_fwd(model.embed_tokens(mx.array([row], dtype=mx.int32)))
            p_s = gpu._project_span_pool(h_s, np.array([[[0, len(row)]]], dtype=np.int32))[0, 0]
            singles.append(p_s / np.linalg.norm(p_s))
        singles_arr = np.stack(singles, axis=0)
        assert float(np.min(np.sum(packed_norm * singles_arr, axis=-1))) >= 0.9999
    finally:
        mx.set_default_device(prev_dev)

    pkg_root = Path(__file__).resolve().parents[1]
    cli_src = (pkg_root / "src" / "fast_pplx" / "cli.py").read_text(encoding="utf-8")
    build_src = (pkg_root / "scripts" / "build_ane_models.py").read_text(encoding="utf-8")
    runner_src = (pkg_root / "native" / "pplx_coreml_runner.mm").read_text(encoding="utf-8")
    assert "_discover_ane_stage_specs" in cli_src
    assert "pplx_ane_inspect_plan_fn" in cli_src
    assert "if inspect_placement and not use_multifunction:" in build_src
    assert "@available(macOS 15.0, *)" in runner_src
    assert 'std::strcmp(function_name, "main") != 0' in runner_src


def test_state_dtype_normalization_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_pplx.gpu import normalize_state_dtype

    monkeypatch.delenv("FAST_PPLX_STATE_DTYPE", raising=False)
    assert normalize_state_dtype() == "float32"
    assert normalize_state_dtype(state_fp32=False) == "bfloat16"
    assert normalize_state_dtype("fp32") == "float32"
    assert normalize_state_dtype("BF16") == "bfloat16"
    assert normalize_state_dtype("i8") == "int8"

    monkeypatch.setenv("FAST_PPLX_STATE_DTYPE", "i8")
    assert normalize_state_dtype() == "int8"
    assert normalize_state_dtype("float32") == "float32"

    with pytest.raises(ValueError, match="Unsupported state_dtype"):
        normalize_state_dtype("fp8_e4m3")

    monkeypatch.setenv("FAST_PPLX_STATE_DTYPE", "invalid_dtype")
    with pytest.raises(ValueError, match="Unsupported state_dtype"):
        normalize_state_dtype()


def test_block32_int8_state_codec_cpu_roundtrip_and_nonfinite_guard() -> None:
    from fast_pplx.gpu import (
        dequantize_gdn_state_i8,
        gated_delta_update,
        quantize_gdn_state_i8,
    )

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(42)
        st_np = rng.normal(0.0, 0.35, (2, 4, 8, 128)).astype(np.float32)
        st_np[0, 1, 2, 17] = np.nan
        st_np[1, 3, 5, 99] = np.inf
        st_mx = mx.array(st_np)

        codes, scales = quantize_gdn_state_i8(st_mx)
        mx.eval(codes, scales)
        assert codes.dtype == mx.int8
        assert scales.dtype == mx.bfloat16
        assert codes.shape == (2, 4, 8, 128)
        assert scales.shape == (2, 4, 8, 4)

        fp32_bytes = st_np.nbytes
        i8_bytes = codes.nbytes + scales.nbytes
        assert i8_bytes * 3 < fp32_bytes

        restored = dequantize_gdn_state_i8(codes, scales, out_dtype=mx.float32)
        mx.eval(restored)
        restored_np = np.asarray(restored, dtype=np.float32)
        sc_u16 = np.asarray(scales.view(mx.uint16), dtype=np.uint16)
        # Non-finite blocks emit 0x7fc0 quiet NaN scales and restore entire 32-element block to NaN
        assert sc_u16[0, 1, 2, 0] == 0x7FC0
        assert sc_u16[1, 3, 5, 3] == 0x7FC0
        assert np.all(np.isnan(restored_np[0, 1, 2, 0:32]))
        assert np.all(np.isnan(restored_np[1, 3, 5, 96:128]))

        # All other blocks remain finite and closely match input
        clean_mask = np.ones_like(st_np, dtype=bool)
        clean_mask[0, 1, 2, 0:32] = False
        clean_mask[1, 3, 5, 96:128] = False
        c_got = restored_np[clean_mask]
        c_ref = st_np[clean_mask]
        assert np.all(np.isfinite(c_got))
        cos = float(np.dot(c_got, c_ref) / (np.linalg.norm(c_got) * np.linalg.norm(c_ref)))
        assert cos >= 0.9999

        empty_c, empty_s = quantize_gdn_state_i8(mx.zeros((0, 4, 8, 128), dtype=mx.float32))
        mx.eval(empty_c, empty_s)
        assert empty_c.shape == (0, 4, 8, 128) and empty_s.shape == (0, 4, 8, 4)
        empty_r = dequantize_gdn_state_i8(empty_c, empty_s)
        mx.eval(empty_r)
        assert empty_r.shape == (0, 4, 8, 128)

        # CPU gated_delta_update fallback supports float32, bfloat16, and int8 tuple state roundtrips
        q = mx.array(rng.normal(0.0, 0.15, (1, 12, 4, 64)).astype(np.float32))
        k = mx.array(rng.normal(0.0, 0.15, (1, 12, 4, 64)).astype(np.float32))
        v = mx.array(rng.normal(0.0, 0.25, (1, 12, 4, 64)).astype(np.float32))
        a = mx.array(rng.normal(-0.4, 0.3, (1, 12, 4)).astype(np.float32))
        b = mx.array(rng.normal(-0.2, 0.3, (1, 12, 4)).astype(np.float32))
        A_log = mx.full((4,), -2.0, dtype=mx.float32)
        dt_bias = mx.full((4,), -1.0, dtype=mx.float32)

        y_f32, s_f32 = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False, state_dtype="float32")
        y_bf16, s_bf16 = gated_delta_update(q, k, v, a, b, A_log, dt_bias, use_kernel=False, state_dtype="bfloat16")
        y_i8_1, s_i8_1 = gated_delta_update(
            q[:, :6], k[:, :6], v[:, :6], a[:, :6], b[:, :6], A_log, dt_bias, use_kernel=False, state_dtype="int8"
        )
        y_i8_2, s_i8_2 = gated_delta_update(
            q[:, 6:], k[:, 6:], v[:, 6:], a[:, 6:], b[:, 6:], A_log, dt_bias, state=s_i8_1, use_kernel=False, state_dtype="int8"
        )
        y_i8 = mx.concatenate([y_i8_1, y_i8_2], axis=1)
        mx.eval(y_f32, s_f32, y_bf16, s_bf16, y_i8, s_i8_2[0], s_i8_2[1])
        assert s_bf16.dtype == mx.bfloat16
        assert isinstance(s_i8_2, tuple) and s_i8_2[0].dtype == mx.int8 and s_i8_2[1].dtype == mx.bfloat16

        yf = np.asarray(y_f32, dtype=np.float32).ravel()
        yi = np.asarray(y_i8, dtype=np.float32).ravel()
        yb = np.asarray(y_bf16, dtype=np.float32).ravel()
        assert float(np.dot(yf, yi) / (np.linalg.norm(yf) * np.linalg.norm(yi))) >= 0.9999
        assert float(np.dot(yf, yb) / (np.linalg.norm(yf) * np.linalg.norm(yb))) >= 0.9999
    finally:
        mx.set_default_device(prev_dev)


def test_cpu_fallback_norms_mlp_and_gdn_non32_and_empty_shapes() -> None:
    from fast_pplx.gpu import PplxGatedDeltaNet, PplxGatedRMSNorm, PplxMLP, PplxRMSNorm

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        # Non-32 dims exercise pure MLX fallbacks cleanly on CPU
        rms = PplxRMSNorm(24, eps=1e-6)
        grms = PplxGatedRMSNorm(24, eps=1e-6)
        x = mx.ones((2, 3, 24), dtype=mx.float32) * 0.5
        gate = mx.ones((2, 3, 24), dtype=mx.float32) * 0.25
        y_rms = rms(x)
        y_grms = grms(x, gate)
        mx.eval(y_rms, y_grms)
        assert y_rms.shape == (2, 3, 24) and y_grms.shape == (2, 3, 24)

        cfg = PplxConfig(
            hidden_size=24,
            intermediate_size=14,
            num_hidden_layers=1,
            layer_types=("linear_attention",),
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            rope_dim=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            vocab_size=64,
        )
        mlp = PplxMLP(cfg)
        y_mlp = mlp(mx.ones((1, 4, 24), dtype=mx.float32))
        mx.eval(y_mlp)
        assert y_mlp.shape == (1, 4, 24)

        gdn = PplxGatedDeltaNet(cfg, state_dtype="float32")
        gdn.use_kernel = False
        y_short = gdn(mx.ones((1, 2, 24), dtype=mx.float32) * 0.1)
        mx.eval(y_short)
        assert y_short.shape == (1, 2, 24)
        assert np.all(np.isfinite(np.asarray(y_short, dtype=np.float32)))
    finally:
        mx.set_default_device(prev_dev)


def test_engine_forwards_state_dtype_and_benchmark_parser_accepts_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import fast_pplx.engine as eng_mod

    seen: dict[str, object] = {}
    tok_json = _write_synthetic_tokenizer(tmp_path)

    class _FakeGPU:
        def __init__(self, **kw: object) -> None:
            seen.update(kw)
            self.config = DEFAULT_PPLX_CONFIG
            self.tokenizer = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)
            self.max_seq_len = 32768
            self.dim = DEFAULT_PPLX_CONFIG.embedding_dim

        def close(self) -> None:
            pass

    monkeypatch.setattr(eng_mod, "FastPplxGPU", _FakeGPU)
    eng = eng_mod.FastPplx(mode="gpu", state_dtype="int8", enable_ane=False)
    try:
        assert seen["state_dtype"] == "int8"
    finally:
        eng.close()

    bench_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("fast_pplx_bench_cli", bench_path)
    assert spec is not None and spec.loader is not None
    bench_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench_mod)
    args = bench_mod.build_parser().parse_args(["speed", "--state-dtype", "int8"])
    assert args.state_dtype == "int8"


def test_embed_inputs_uses_float32_gather_for_non_f16_mapped_embedding(tmp_path: Path) -> None:
    from fast_pplx.gpu import FastPplxGPU

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        values = np.array(
            [[65536.0, 1e-6, 1.25, -0.5], [100000.0, -3e-6, -2.5, 0.75]],
            dtype=np.float32,
        )
        f32_path = _write_synthetic_safetensors(
            tmp_path / "embed_f32.safetensors",
            {"weight": (values, "F32")},
        )
        bf16_u16 = (values.view(np.uint32) >> 16).astype(np.uint16)
        bf16_path = _write_synthetic_safetensors(
            tmp_path / "embed_bf16.safetensors",
            {"weight": (bf16_u16, "BF16")},
        )

        class _Stub:
            def __init__(self, mapped: MappedEmbedding, act_dtype: mx.Dtype) -> None:
                self._mapped_embed = mapped
                self.act_dtype = act_dtype
                self.model = None

        ids = np.array([[0, 1]], dtype=np.int32)
        with MappedEmbedding(f32_path, tensor_name="weight") as mm_f32:
            out_f32 = FastPplxGPU._embed_inputs(_Stub(mm_f32, mx.bfloat16), ids)
            mx.eval(out_f32)
            assert out_f32.dtype == mx.bfloat16
            ref_f32 = mx.array(values[None, ...], dtype=mx.bfloat16)
            mx.eval(ref_f32)
            np.testing.assert_allclose(
                np.asarray(out_f32.astype(mx.float32)),
                np.asarray(ref_f32.astype(mx.float32)),
                atol=0.0,
                rtol=0.0,
            )

        with MappedEmbedding(bf16_path, tensor_name="weight") as mm_bf16:
            out_bf16 = FastPplxGPU._embed_inputs(_Stub(mm_bf16, mx.bfloat16), ids)
            mx.eval(out_bf16)
            assert out_bf16.dtype == mx.bfloat16
            expected_bf16 = ((bf16_u16.astype(np.uint32) << 16).view(np.float32))[None, ...]
            np.testing.assert_allclose(
                np.asarray(out_bf16.astype(mx.float32)),
                expected_bf16,
                atol=0.0,
                rtol=0.0,
            )

        seen_dtypes: list[str] = []

        class _Spy:
            def __init__(self, dtype_tag: str) -> None:
                self.dtype = dtype_tag

            def gather(self, token_ids: np.ndarray, *, out_dtype: str = "float16") -> np.ndarray:
                seen_dtypes.append(str(out_dtype))
                return np.ones((*token_ids.shape, 4), dtype=np.dtype(out_dtype))

        FastPplxGPU._embed_inputs(_Stub(_Spy("F32"), mx.bfloat16), ids)  # type: ignore[arg-type]
        FastPplxGPU._embed_inputs(_Stub(_Spy("F32"), mx.float16), ids)  # type: ignore[arg-type]
        FastPplxGPU._embed_inputs(_Stub(_Spy("F16"), mx.bfloat16), ids)  # type: ignore[arg-type]
        assert seen_dtypes == ["float32", "float16", "float16"]
    finally:
        mx.set_default_device(prev_dev)


def test_contextual_window_anchoring_overlap_and_parquet_1to1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify plan_contextual_windows + embed_contextual_parquet anchor/overlap emission without duplicating rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fast_pplx.cli import main as cli_main
    from fast_pplx.engine import FastPplx
    from fast_pplx.parquet_pipeline import embed_contextual_parquet

    tok_json = _write_synthetic_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, DEFAULT_PPLX_CONFIG)

    doc_text = " ".join(["hello world chunk one two"] * 6)
    base_plans = tok.plan_contextual_windows(doc_text, chunk_tokens=8, context_tokens=18, overlap=0)
    assert len(base_plans) >= 2
    base_spans = [sp for w in base_plans for sp in w.spans]

    # 1. Explicit header_text + window_overlap_chunks=1: context carries prefix, emitted spans stay strictly 1:1
    ov_plans = tok.plan_contextual_windows(
        doc_text, chunk_tokens=8, context_tokens=22, overlap=0, header_text="hello", window_overlap_chunks=1
    )
    assert len(ov_plans) >= 2
    assert ov_plans[0].has_header_anchor and ov_plans[0].emit_start == 1
    assert ov_plans[1].has_header_anchor and ov_plans[1].overlap_chunks == 1 and ov_plans[1].emit_start == 2
    for w in ov_plans:
        assert len(tok.encode_contextual_document(w.chunks, return_spans=False)) <= 22
    assert [sp for w in ov_plans for sp in w.spans] == base_spans

    # 2. Positional vector test proves embed_contextual_parquet slices vec[emit_start : emit_start + n_emit]
    seen_windows: list[list[str]] = []

    class _PositionalGPU:
        max_seq_len = 32768

        def encode_contextual_tokens(self, docs, *, batch_size=8, dimensions=None, output_dtype="float32", normalize=True):
            d = dimensions or 8
            out = []
            for win_idx, (_, spans) in enumerate(docs):
                # Store [win_pos_in_doc, pos_within_window, ...] so slicing out prefixes is directly observable.
                mat = np.zeros((len(spans), d), dtype=np.float32)
                for j in range(len(spans)):
                    mat[j, 0] = float(j)
                out.append(mat)
            return out

        def close(self):
            pass

    orig_enc = tok.encode_contextual_document

    def _spy_enc(chunks_w, *args, **kwargs):
        seen_windows.append(list(chunks_w))
        return orig_enc(chunks_w, *args, **kwargs)

    monkeypatch.setattr(tok, "encode_contextual_document", _spy_enc)

    in_pq = tmp_path / "hdr_in.parquet"
    pq.write_table(
        pa.table({"doc_id": ["d0", "d1"], "title": ["hello", ""], "text": [doc_text, doc_text]}),
        in_pq,
    )
    out_hdr_pq = tmp_path / "hdr_out.parquet"
    out_shard_dir = tmp_path / "hdr_shards"

    with FastPplx(home=tmp_path, mode="gpu", _gpu_backend=_PositionalGPU(), _tokenizer=tok) as eng:
        with pytest.raises(KeyError, match="missing_col"):
            embed_contextual_parquet(
                eng, in_pq, out_hdr_pq, header_col="missing_col", chunk_tokens=8, context_tokens=22, overlap=0
            )

        summary = embed_contextual_parquet(
            eng,
            in_pq,
            out_hdr_pq,
            header_col="title",
            chunk_tokens=8,
            context_tokens=22,
            overlap=0,
            repeat_header_tokens=2,
            window_overlap_chunks=1,
            dimensions=8,
        )
        assert summary.rows_emitted == 2 * len(base_spans)
        out_tbl = pq.read_table(out_hdr_pq)
        assert out_tbl.num_rows == 2 * len(base_spans)
        d0_vecs = [v for did, v in zip(out_tbl["doc_id"].to_pylist(), out_tbl["vector"].to_pylist(), strict=True) if did == "d0"]
        d1_vecs = [v for did, v in zip(out_tbl["doc_id"].to_pylist(), out_tbl["vector"].to_pylist(), strict=True) if did == "d1"]
        # d0 window 0 has header anchor ("hello") -> emit_start=1, window 1 has anchor+overlap -> emit_start=2
        assert d0_vecs[0][0] == 1.0
        assert any(v[0] == 2.0 for v in d0_vecs)
        # d1 had blank title -> falls back to repeat_header_tokens=2: window 0 starts at 0.0, window 1 at 2.0
        assert d1_vecs[0][0] == 0.0
        assert any(v[0] == 2.0 for v in d1_vecs)

        # Shard-dir mode matches exact non-duplicated emitted rows too
        shard_sum = embed_contextual_parquet(
            eng,
            in_pq,
            out_shard_dir,
            header_col="title",
            chunk_tokens=8,
            context_tokens=22,
            overlap=0,
            repeat_header_tokens=2,
            window_overlap_chunks=1,
            shard_rows=4,
            dimensions=8,
        )
        assert shard_sum.rows_emitted == 2 * len(base_spans)

        # 3. Auto repeat_header_tokens=128 activates for context_tokens >= 512, and repeat_header_tokens=0 disables
        long_doc = "hello world " + " ".join(["chunk one two"] * 200)
        big_in = tmp_path / "big_in.parquet"
        pq.write_table(pa.table({"doc_id": ["big"], "text": [long_doc]}), big_in)

        seen_windows.clear()
        s_auto = embed_contextual_parquet(
            eng, big_in, tmp_path / "big_auto.parquet", chunk_tokens=160, context_tokens=512, overlap=0, dimensions=8
        )
        assert len(seen_windows) >= 2
        assert seen_windows[1][0].startswith("hello world")
        assert sum(len(w) for w in seen_windows) == s_auto.rows_emitted + (len(seen_windows) - 1)

        seen_windows.clear()
        s_off = embed_contextual_parquet(
            eng,
            big_in,
            tmp_path / "big_off.parquet",
            chunk_tokens=160,
            context_tokens=512,
            overlap=0,
            repeat_header_tokens=0,
            dimensions=8,
        )
        assert s_auto.rows_emitted == s_off.rows_emitted
        assert sum(len(w) for w in seen_windows) == s_off.rows_emitted  # zero anchors when disabled

    # 4. CLI forwards all 4 contextual window options
    captured: dict[str, object] = {}

    def _spy_ctx(*args, **kwargs):
        captured.update(kwargs)
        from fast_shared.parquet_pipeline import ParquetEmbedSummary

        return ParquetEmbedSummary(1, 1, 1, 10, 0.01, tmp_path / "cli_out.parquet")

    monkeypatch.setattr("fast_pplx.cli.warn_if_untested_hardware", lambda *a, **k: False)
    monkeypatch.setattr("fast_pplx.cli._make_engine", lambda a, **kw: eng)
    monkeypatch.setattr("fast_pplx.cli.embed_contextual_parquet", _spy_ctx)

    rc = cli_main(
        [
            "parquet",
            "--input",
            str(in_pq),
            "--output",
            str(tmp_path / "cli_out.parquet"),
            "--context-tokens",
            "2048",
            "--header-text",
            "Contract Title",
            "--header-col",
            "title",
            "--repeat-header-tokens",
            "96",
            "--window-overlap-chunks",
            "1",
        ]
    )
    assert rc == 0
    assert captured["header_text"] == "Contract Title"
    assert captured["header_col"] == "title"
    assert captured["repeat_header_tokens"] == 96
    assert captured["window_overlap_chunks"] == 1
