"""CPU unit tests + lock-serialized ANE/hetero2 parity tests for fast-pplx."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fast_pplx.cli import build_parser
from fast_pplx.config import PplxConfig
from fast_pplx.engine import (
    DEFAULT_MODEL_NAME,
    EMBED_DIM,
    FastPplx,
    MAX_EMBED_DIM,
    MRL_DIMENSIONS,
    pack_ane_buckets,
)
from fast_pplx.gpu import PplxModel, _build_sdpa_mask
import fast_pplx.gpu as pplx_gpu_mod
from fast_pplx.parquet_pipeline import embed_contextual_parquet
from fast_pplx.paths import resolve_paths
from fast_pplx.tokenizer import ContextualTokenizer

_PKG_ROOT = Path(__file__).resolve().parent.parent
_BUILD_ANE_PATH = _PKG_ROOT / "scripts" / "build_ane_models.py"
_EXPORT_WEIGHTS_PATH = _PKG_ROOT / "scripts" / "export_weights.py"
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_test_tokenizer(directory: Path) -> Path:
    vocab = {
        "<unk>": 0,
        "D": 1,
        "[": 2,
        "]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "epsilon": 8,
        "zeta": 9,
        "one": 10,
        "two": 11,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|endoftext|>", "[Q] ", "[D] "])
    tok.add_tokens(["<|chunk_sep|>"])
    out = directory / "tokenizer.json"
    tok.save(str(out))
    return out


def test_build_ane_helpers_ftz_scaling_stages_and_outliers() -> None:
    bam = _load_script_module("fast_pplx_build_ane_models", _BUILD_ANE_PATH)

    assert bam.plan_stage_ranges(12, 3) == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert bam.plan_stage_ranges(12, 4) == [(0, 4), (4, 8), (8, 12)]
    assert bam._parse_seq_lens(["64,128", "256, 512"]) == (64, 128, 256, 512)

    eps_128 = bam.compute_gnorm_eps_scaled(128.0)
    assert np.isclose(eps_128, 0.016384, rtol=1e-7)
    assert eps_128 > 250.0 * 6.103515625e-5  # Well above FP16_MIN_NORMAL FTZ floor

    # Synthesize calibration stats matching the 8 linear absmax >= 50 outliers
    calib = {"layers": {}}
    for li in range(12):
        calib["layers"][f"layers.{li}.mlp.gate_proj"] = {"absmax": 12.0}
        calib["layers"][f"layers.{li}.mlp.down_proj"] = {"absmax": 120.0 if li in (5, 7, 8, 9, 10, 11) else 15.0}
        if li % 2 == 0:
            calib["layers"][f"layers.{li}.linear_attn.in_proj_qkv"] = {"absmax": 20.0}
            calib["layers"][f"layers.{li}.linear_attn.out_proj"] = {"absmax": 80.0 if li in (8, 10) else 18.0}
        else:
            calib["layers"][f"layers.{li}.self_attn.q_proj"] = {"absmax": 24.0}
            calib["layers"][f"layers.{li}.self_attn.o_proj"] = {"absmax": 19.0}

    s5 = bam.select_qdq_scales(5, "full_attention", calib, outlier_threshold=50.0)
    assert s5["l5_mlp_gu"] == 12.0
    assert s5["l5_mlp_dn"] is None
    assert s5["l5_qkvgate"] == 24.0
    assert s5["l5_o_proj"] == 19.0

    s8 = bam.select_qdq_scales(8, "linear_attention", calib, outlier_threshold=50.0)
    assert s8["l8_qkvzba"] == 20.0
    assert s8["l8_out_proj"] is None
    assert s8["l8_mlp_dn"] is None


def test_hierarchical_block_lower_tri_inverse_accuracy() -> None:
    def _np_block_inv(m_strict: np.ndarray) -> np.ndarray:
        c = m_strict.shape[-1]
        m2 = np.zeros((c, c), dtype=m_strict.dtype)
        for k in range(0, c, 2):
            if k + 1 < c:
                m2[k + 1, k] = 1.0
        t_cur = np.eye(c, dtype=m_strict.dtype) + m_strict * m2
        b = 2
        while b < c:
            mask = np.zeros((c, c), dtype=m_strict.dtype)
            for k in range(0, c, 2 * b):
                r1 = min(c, k + 2 * b)
                if k + b < r1:
                    mask[k + b : r1, k : k + b] = 1.0
            off = m_strict * mask
            t_cur = t_cur + (t_cur @ off) @ t_cur
            b *= 2
        return t_cur

    rng = np.random.default_rng(7)
    l_f32 = np.tril(rng.normal(0.0, 0.35, size=(6, 32, 32)), k=-1).astype(np.float32)
    ref = np.linalg.inv(np.eye(32, dtype=np.float32) - l_f32)
    got_f32 = _np_block_inv(l_f32)
    assert np.abs(got_f32 - ref).max() < 2e-5

    got_f16 = _np_block_inv(l_f32.astype(np.float16)).astype(np.float32)
    cos = float(np.sum(got_f16 * ref) / (np.linalg.norm(got_f16) * np.linalg.norm(ref)))
    assert cos >= 0.99999


def test_multi_doc_bucket_packing_c32_gap3_isolation() -> None:
    # Verify exact boundary transitions:
    # doc0 len=29 -> next start = ceil((29+3)/32)*32 = 32 (gap=3)
    # doc1 len=30 -> end=62 -> next start = ceil((62+3)/32)*32 = 96 (gap=34)
    # doc2 len=32 -> end=96+32=128 (smallest bucket cap >= 128 is 128)
    items = [
        (0, list(range(1, 30)), [(1, 29)]),         # len=29
        (1, list(range(100, 130)), [(2, 28)]),     # len=30
        (2, list(range(200, 232)), [(0, 0)]),      # len=32 empty chunk span
    ]
    bkts = pack_ane_buckets(items, supported_caps=(64, 128, 256, 512), chunk_size=32, conv_gap=3)
    assert len(bkts) == 1
    bkt = bkts[0]
    assert bkt.seq_len == 128
    assert bkt.n_docs == 3

    starts_by_item = {t.item_index: (t.start, t.end) for t in bkt.targets}
    assert starts_by_item[0] == (1, 29)
    assert starts_by_item[1] == (34, 60)  # 32 + (2, 28)
    assert starts_by_item[2] == (0, 0)

    # Verify doc_ids isolation & >= 3 PAD slots between docs
    assert np.all(bkt.doc_ids[0:29] == 0)
    assert np.all(bkt.doc_ids[29:32] == -1)  # 3 zero-pad tokens before doc1 at 32
    assert np.all(bkt.doc_ids[32:62] == 1)
    assert np.all(bkt.doc_ids[62:96] == -1)  # >= 3 zero-pad tokens before doc2 at 96
    assert np.all(bkt.doc_ids[96:128] == 2)
    assert bkt.pos_ids[0:29].tolist() == list(range(29))
    assert bkt.pos_ids[32:62].tolist() == list(range(30))
    assert bkt.pos_ids[96:128].tolist() == list(range(32))


def test_native_fast_io_pack_rope_and_project_parity() -> None:
    paths = resolve_paths()
    if not paths.io_dylib.is_file():
        pytest.skip(f"Native IO dylib not compiled at {paths.io_dylib}")

    from fast_pplx.engine import _load_io_lib

    lib = _load_io_lib(paths.io_dylib)
    rng = np.random.default_rng(42)

    # 1. RoPE table check
    max_pos, rope_half, theta = 128, 32, 10_000_000.0
    cos_u16 = np.empty((max_pos, rope_half), dtype=np.uint16)
    sin_u16 = np.empty((max_pos, rope_half), dtype=np.uint16)
    lib.pplx_build_rope_table_f16(cos_u16.ctypes.data, sin_u16.ctypes.data, max_pos, rope_half, theta)

    inv_freq = 1.0 / (theta ** (np.arange(0, rope_half * 2, 2, dtype=np.float64) / (rope_half * 2)))
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv_freq[None, :]
    ref_cos = np.cos(ang).astype(np.float16).view(np.uint16)
    ref_sin = np.sin(ang).astype(np.float16).view(np.uint16)
    assert np.all(np.abs(cos_u16.view(np.float16) - ref_cos.view(np.float16)) <= 2e-3)
    assert np.all(np.abs(sin_u16.view(np.float16) - ref_sin.view(np.float16)) <= 2e-3)

    # 2. Pack bucket with 2 isolated docs (29 tokens + 3 pad + 20 tokens + tail pad in S=64)
    vocab, hidden, seq_len = 64, 1024, 64
    embed_f16 = rng.normal(0.0, 0.5, size=(vocab, hidden)).astype(np.float16)
    tok_ids = np.zeros((seq_len,), dtype=np.int32)
    doc_ids = np.full((seq_len,), -1, dtype=np.int32)
    pos_ids = np.zeros((seq_len,), dtype=np.int32)

    tok_ids[:29] = rng.integers(1, vocab, size=(29,), dtype=np.int32)
    doc_ids[:29] = 0
    pos_ids[:29] = np.arange(29, dtype=np.int32)

    tok_ids[32:52] = rng.integers(1, vocab, size=(20,), dtype=np.int32)
    doc_ids[32:52] = 1
    pos_ids[32:52] = np.arange(20, dtype=np.int32)

    h_in_u16 = np.empty((hidden, seq_len), dtype=np.uint16)
    bias_u16 = np.empty((seq_len, seq_len), dtype=np.uint16)
    c32_u16 = np.empty((seq_len, rope_half), dtype=np.uint16)
    s32_u16 = np.empty((seq_len, rope_half), dtype=np.uint16)
    tvalid_u16 = np.empty((seq_len,), dtype=np.uint16)
    carry_u16 = np.empty((seq_len // 32,), dtype=np.uint16)

    lib.pplx_pack_bucket_f16(
        embed_f16.ctypes.data,
        vocab,
        hidden,
        seq_len,
        32,
        rope_half,
        tok_ids.ctypes.data,
        doc_ids.ctypes.data,
        pos_ids.ctypes.data,
        cos_u16.ctypes.data,
        sin_u16.ctypes.data,
        max_pos,
        h_in_u16.ctypes.data,
        bias_u16.ctypes.data,
        c32_u16.ctypes.data,
        s32_u16.ctypes.data,
        tvalid_u16.ctypes.data,
        carry_u16.ctypes.data,
    )

    h_f16 = h_in_u16.view(np.float16)
    assert np.array_equal(h_f16[:, :29], embed_f16[tok_ids[:29]].T)
    assert np.all(h_f16[:, 29:32] == 0.0)
    assert np.array_equal(h_f16[:, 32:52], embed_f16[tok_ids[32:52]].T)

    bias_f16 = bias_u16.view(np.float16)
    assert np.all(bias_f16[:29, :29] == 0.0)
    assert np.all(bias_f16[:29, 29:] == np.float16(-10000.0))
    assert np.all(bias_f16[32:52, 32:52] == 0.0)
    assert np.all(bias_f16[32:52, :32] == np.float16(-10000.0))
    # Pad rows attend to pad keys so softmax never produces NaN
    assert np.all(bias_f16[29:32, 29:32] == 0.0)

    tvalid = tvalid_u16.view(np.float16)
    assert np.all(tvalid[:29] == 1.0)
    assert np.all(tvalid[29:32] == 0.0)
    assert np.all(tvalid[32:52] == 1.0)
    assert np.all(tvalid[52:] == 0.0)
    # New document at chunk 1 (index 32) resets GatedDeltaNet carry
    assert carry_u16.view(np.float16).tolist() == [0.0, 0.0]

    # 3. Pool + FP32 cblas_sgemm contextual projection + MRL + int8 + L2 norm parity
    h_out_f16 = rng.normal(0.0, 0.4, size=(hidden, seq_len)).astype(np.float16)
    proj_w = rng.normal(0.0, 0.1, size=(hidden, hidden)).astype(np.float32)
    starts = np.array([2, 33, 0], dtype=np.int32)
    ends = np.array([15, 50, 0], dtype=np.int32)
    out_dim = 256

    out_l2 = np.empty((3, out_dim), dtype=np.float32)
    lib.pplx_pool_and_project_f32(
        h_out_f16.ctypes.data,
        hidden,
        seq_len,
        3,
        starts.ctypes.data,
        ends.ctypes.data,
        None,
        proj_w.ctypes.data,
        hidden,
        out_dim,
        0,
        1,
        out_l2.ctypes.data,
    )

    h_f32 = h_out_f16.astype(np.float32)
    p0 = h_f32[:, 2:15].mean(axis=1) @ proj_w[:out_dim].T
    p1 = h_f32[:, 33:50].mean(axis=1) @ proj_w[:out_dim].T
    ref0 = p0 / np.linalg.norm(p0)
    ref1 = p1 / np.linalg.norm(p1)
    assert np.allclose(out_l2[0], ref0, atol=1e-4)
    assert np.allclose(out_l2[1], ref1, atol=1e-4)
    assert np.all(out_l2[2] == 0.0)

    out_i8 = np.empty((3, out_dim), dtype=np.float32)
    lib.pplx_pool_and_project_f32(
        h_out_f16.ctypes.data,
        hidden,
        seq_len,
        3,
        starts.ctypes.data,
        ends.ctypes.data,
        None,
        proj_w.ctypes.data,
        hidden,
        out_dim,
        1,
        0,
        out_i8.ctypes.data,
    )
    assert np.array_equal(out_i8[0], np.clip(np.rint(np.tanh(p0) * 127.0), -128, 127))
    assert np.array_equal(out_i8[2], np.zeros((out_dim,), dtype=np.float32))


def test_hetero2_scheduler_routing_and_cli_contract(tmp_path: Path) -> None:
    tok_json = _write_test_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, PplxConfig())

    calls: dict[str, list[Any]] = {"ane": [], "gpu": []}
    call_lock = threading.Lock()

    class _MockGPU:
        max_seq_len = 32768

        def encode_tokens(self, token_lists, *, batch_size=16, dimensions=None, output_dtype="float32", normalize=True):
            with call_lock:
                calls["gpu"].append([len(t) for t in token_lists])
            dim = dimensions or 1024
            time.sleep(0.002)
            out = np.zeros((len(token_lists), dim), dtype=np.float32)
            for i, row in enumerate(token_lists):
                out[i, 0] = float(row[0] if row else 0)
                out[i, 1] = 2.0
            if normalize:
                out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
            return out

        def encode_contextual_chunks(self, docs, *, batch_size=8, max_length=32768, dimensions=None, output_dtype="float32", normalize=True):
            dim = dimensions or 1024
            return [np.ones((len(d), dim), dtype=np.float32) * 0.25 for d in docs]

        def close(self):
            pass

    class _MockANE:
        caps = (64, 128, 256, 512)
        max_cap = 512

        def run_buckets(self, buckets, *, out_dim, apply_tanh_int8, apply_l2_norm):
            done = []
            for bkt in buckets:
                with call_lock:
                    calls["ane"].append(bkt.seq_len)
                live = [t for t in bkt.targets if t.end > t.start]
                mat = np.zeros((len(live), out_dim), dtype=np.float32)
                for row_i, tgt in enumerate(live):
                    mat[row_i, 0] = float(bkt.token_ids[tgt.start])
                    mat[row_i, 1] = 2.0
                if apply_l2_norm:
                    mat /= np.maximum(np.linalg.norm(mat, axis=-1, keepdims=True), 1e-12)
                done.append((bkt, mat))
            return done

        def close(self):
            pass

    # Mixed workload: long > 512 goes strictly to GPU, short <= 512 runs through ANE/hetero2
    with FastPplx(home=tmp_path, mode="hetero2", _gpu_backend=_MockGPU(), _ane_backend=_MockANE(), _tokenizer=tok) as eng:
        assert eng.default_mode == "hetero2"
        assert eng.ane_available is True
        assert eng.dim == EMBED_DIM
        assert eng.max_dim == MAX_EMBED_DIM
        assert eng.supports_dimensions is True
        assert eng.mrl_dimensions == MRL_DIMENSIONS
        assert eng.model_name == DEFAULT_MODEL_NAME

        short_seq = [4, 5, 6]
        long_seq = [7] * 600
        vecs = eng.encode_tokens([short_seq, long_seq, [8, 9]], dimensions=256)
        assert vecs.shape == (3, 256)
        assert len(calls["gpu"]) >= 1 and 600 in [x for batch in calls["gpu"] for x in batch]
        assert len(calls["ane"]) >= 1
        assert np.allclose(vecs[0], vecs[0]) and vecs[0, 0] > 0.0

        # Query helper returns 1D vector; corpus helper returns 2D
        qv = eng.encode_query("alpha beta", dimensions=128)
        assert qv.shape == (128,)
        cv = eng.encode_corpus(["alpha beta", "gamma delta"], dimensions=128)
        assert cv.shape == (2, 128)
        ctx = eng.encode_contextual_chunks([["alpha", "beta"], ["one " * 320, "two " * 320]], dimensions=128)
        assert len(ctx) == 2 and ctx[0].shape == (2, 128) and ctx[1].shape == (2, 128)

    # Graceful fallback to gpu in auto mode when bundles/weights are absent
    with FastPplx(home=tmp_path, mode="auto", _gpu_backend=_MockGPU(), _tokenizer=tok) as eng_fb:
        assert eng_fb.ane_available is False
        assert eng_fb.default_mode == "gpu"

    parser = build_parser()
    ns = parser.parse_args(["parquet", "--input", "in.parquet", "--output", "out.parquet", "--mode", "hetero2", "--context-tokens", "2048", "--chunk-tokens", "128"])
    assert ns.mode == "hetero2"
    assert ns.context_tokens == 2048
    assert ns.chunk_tokens == 128


def test_two_knob_context_window_packing_and_parquet(tmp_path: Path) -> None:
    tok_json = _write_test_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, PplxConfig())

    doc_text = " ".join(["alpha beta gamma delta"] * 12)  # 48 words -> 48 tokens
    grouped = tok.chunk_and_group_document(doc_text, chunk_tokens=12, context_tokens=28, overlap=2)
    assert len(grouped) >= 2
    for win_chunks, spans in grouped:
        assert len(win_chunks) == len(spans) >= 1
        full_ids = tok.encode_contextual_document(win_chunks, return_spans=False)
        if len(win_chunks) > 1:
            assert len(full_ids) <= 28

    in_pq = tmp_path / "docs.parquet"
    out_pq = tmp_path / "out.parquet"
    pq.write_table(pa.table({"doc_id": ["d0", "d1"], "text": [doc_text, "one two alpha beta"]}), in_pq)

    class _FakeEngine:
        dim = 128
        max_dim = 1024
        tokenizer = tok

        def encode_contextual_chunks(self, documents, **kwargs):
            return [np.ones((len(chunks), 128), dtype=np.float32) * 0.5 for chunks in documents]

    summary = embed_contextual_parquet(
        _FakeEngine(),
        in_pq,
        out_pq,
        chunk_tokens=12,
        context_tokens=28,
        overlap=2,
        dimensions=128,
        dtype="float16",
    )
    assert summary.rows_read == 2
    assert summary.rows_emitted >= 5
    tbl = pq.read_table(out_pq)
    assert {"doc_id", "window_index", "chunk_index", "char_start", "char_end", "chunk_text", "vector"} <= set(
        tbl.schema.names
    )
    assert len(tbl["vector"][0].as_py()) == 128


def test_streamed_long_context_parity_on_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify S > threshold streamed GatedDeltaNet + SDPA matches whole execution purely on CPU."""
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(123)
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

        for seq_len, valid_b1 in [(65, 65), (96, 70), (129, 100), (160, 120)]:
            x = mx.random.normal((2, seq_len, cfg.hidden_size))
            mask_np = np.ones((2, seq_len), dtype=np.bool_)
            if valid_b1 < seq_len:
                mask_np[1, valid_b1:] = False
            ssm_mask = mx.array(mask_np, dtype=mx.bool_)
            sdpa_mask = _build_sdpa_mask(mask_np, dtype_name="float32") if valid_b1 < seq_len else None

            monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 10_000)
            whole = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

            monkeypatch.setattr(pplx_gpu_mod, "_STREAMED_SEQ_THRESHOLD", 64)
            monkeypatch.setattr(pplx_gpu_mod, "_GDN_SLICE_LEN", 32)
            monkeypatch.setattr(pplx_gpu_mod, "_SDPA_SLICE_LEN", 32)
            monkeypatch.setattr(pplx_gpu_mod, "_SDPA_MIN_TAIL", 32)
            streamed = model.forward_hidden(x, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)

            mx.eval(whole, streamed)
            whole_np = np.array(whole, dtype=np.float32)
            streamed_np = np.array(streamed, dtype=np.float32)
            assert np.allclose(whole_np, streamed_np, atol=2e-4, rtol=1e-4), (
                f"Streamed mismatch at seq_len={seq_len}, valid_b1={valid_b1}: "
                f"max_abs={np.abs(whole_np - streamed_np).max()}"
            )
    finally:
        mx.set_default_device(prev_dev)


@contextmanager
def _hw_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl_mod = __import__("fcntl")
        fcntl_mod.flock(fd, fcntl_mod.LOCK_EX)
        yield
    finally:
        try:
            __import__("fcntl").flock(fd, __import__("fcntl").LOCK_UN)
        finally:
            os.close(fd)


@pytest.mark.gpu_ane
def test_real_staged_ane_vs_gpu_parity_and_hetero2_speed() -> None:
    paths = resolve_paths()
    if not (paths.ane_models_dir.is_dir() and any(paths.ane_models_dir.glob("*.mlmodelc"))):
        pytest.skip("Compiled ANE .mlmodelc bundles not staged yet (run scripts/build_ane_models.py)")
    if not (paths.weights_npz.is_file() and paths.embed_tokens_safetensors.is_file()):
        pytest.skip("Exported F16 weights not staged yet (run scripts/export_weights.py)")

    from fast_pplx.engine import _discover_ane_stage_paths, _load_coreml_lib

    stage_map = _discover_ane_stage_paths(paths.ane_models_dir)
    assert stage_map, "No ANE stages discovered"

    coreml_lib = _load_coreml_lib(paths.coreml_dylib)
    buf = ctypes.create_string_buffer(512)
    with _hw_lock():
        for cap, stages in stage_map.items():
            for st_path in stages:
                rc = coreml_lib.pplx_ane_inspect_plan(str(st_path).encode("utf-8"), 0, buf, 512)
                assert rc > 0
                plan = json.loads(buf.value.decode("utf-8"))
                assert plan.get("ane_ops", 0) > 0, f"{st_path} has 0 ANE ops: {plan}"
                assert plan.get("gpu_ops", 0) == 0 and plan.get("cpu_ops", 0) == 0, f"Fallback ops in {st_path}: {plan}"

        docs = [
            "Apple Silicon unified memory enables high-bandwidth tensor streaming.",
            "Qwen3.5 alternates gated delta linear attention with grouped-query full attention.",
            "Contextual chunk embeddings capture surrounding document semantics accurately.",
            "Researchers measured stable retrieval scores across 2048-token windows.",
        ]
        with FastPplx(home=paths.home, mode="hetero2", dtype="float16", local_files_only=True) as eng:
            vecs_ane = eng.encode_documents(docs, mode="ane", normalize=True)
            vecs_gpu = eng.encode_documents(docs, mode="gpu", normalize=True)
            cosines = np.sum(vecs_ane * vecs_gpu, axis=-1)
            assert cosines.min() >= 0.985, f"ANE vs GPU min cosine {cosines.min():.6f} < 0.985"
            assert float(np.mean(cosines)) >= 0.990, f"ANE vs GPU mean cosine {float(np.mean(cosines)):.6f} < 0.990"

            mixed = [*docs * 4, "Long document token repeat. " * 250]
            vecs_h2 = eng.encode_documents(mixed, mode="hetero2", normalize=True)
            assert vecs_h2.shape == (len(mixed), 1024)
            assert np.all(np.isfinite(vecs_h2))


def test_build_ane_eps_clamp_prescaling_and_outlier_gdn_fp16_preservation() -> None:
    bam = _load_script_module("fast_pplx_build_ane_models_remediation", _BUILD_ANE_PATH)

    # 1. FP16 eps floor >= 2^-14 (never subnormal 1e-6 / 1.2e-7)
    fp16_min = float(np.finfo(np.float16).tiny)
    assert np.isclose(bam.FP16_MIN_NORMAL, fp16_min, rtol=1e-9)
    for raw_eps in (0.0, 1.2e-7, 1e-6, 1e-5, 6.0e-5):
        clamped = bam.clamp_fp16_eps(raw_eps)
        assert clamped >= fp16_min
        assert np.float16(clamped) >= np.float16(fp16_min)
        assert np.float16(clamped) > np.float16(0.0)
    assert bam.clamp_fp16_eps(0.016384) == pytest.approx(0.016384)

    # 2. Extended 1/64 prescaling across l5..l11 + final_norm (no 1/8 overflow risk on l6..l8)
    for name in ("l5_in_ln", "l6_post_ln", "l7_in_ln", "l8_post_ln", "l9_in_ln", "l11_post_ln", "final_norm"):
        assert bam.resolve_rmsnorm_prescale(name) == pytest.approx(1.0 / 64.0)
    for name in ("l0_in_ln", "l4_post_ln"):
        assert bam.resolve_rmsnorm_prescale(name) == pytest.approx(1.0)
    # Calibrated override respected
    assert bam.resolve_rmsnorm_prescale("l3_in_ln", calib_stats={"norms": {"l3_in_ln": {"absmax": 3000.0}}}) == pytest.approx(1.0 / 64.0)

    # 3. High-outlier + sensitive GDN projections excluded from global INT8 quantization
    cfg = PplxConfig()
    preserved = bam.select_fp16_preserved_ops(0, 12, cfg.layer_types, preserve_gdn_fp16=True)
    for expected in (
        "l5_mlp_dn", "l7_mlp_dn", "l8_mlp_dn", "l9_mlp_dn", "l10_mlp_dn", "l11_mlp_dn",
        "l0_qkvzba", "l0_conv1d", "l0_out_proj", "l8_out_proj", "l10_out_proj",
    ):
        assert expected in preserved
    assert "l1_qkvgate" not in preserved
    assert "l1_mlp_gu" not in preserved

    # Mocked coremltools.optimize.coreml config builder verifies op_name_configs disables INT8 on excluded ops
    class _FakeCTO:
        class OpLinearQuantizerConfig:
            def __init__(self, **kw):
                self.kw = kw

        class OptimizationConfig:
            def __init__(self, *, op_type_configs, op_name_configs):
                self.op_type_configs = op_type_configs
                self.op_name_configs = op_name_configs

    q_cfg = bam.build_weight_quant_config(preserved, cto_module=_FakeCTO)
    assert q_cfg.op_type_configs["conv"].kw["dtype"] == "int8"
    assert q_cfg.op_name_configs["l8_mlp_dn"] is None
    assert q_cfg.op_name_configs["l0_qkvzba"] is None
    assert "l1_mlp_gu" not in q_cfg.op_name_configs

    # 4. Core AI macOS 27.0+ status helper
    st_off = bam.check_coreai_backend_status(macos_version="26.6", use_os_coreai=False)
    assert st_off["eligible"] is False
    st_old = bam.check_coreai_backend_status(macos_version="26.6", use_os_coreai=True)
    assert st_old["eligible"] is False and "27.0+" in st_old["reason"]


def test_build_ane_coreml_mil_emits_normal_fp16_eps_when_coremltools_installed() -> None:
    pytest.importorskip("coremltools")
    bam = _load_script_module("fast_pplx_build_ane_models_ct", _BUILD_ANE_PATH)
    assert bam.FP16_MIN_NORMAL >= 6.1e-5


def test_native_ane_cli_zip_bit3_and_alignment_and_error_propagation(tmp_path: Path) -> None:
    import io
    import subprocess
    import zipfile
    import zlib
    from fast_pplx.engine import _coreml_last_error, _load_coreml_lib

    paths = resolve_paths()
    if not (paths.ane_cli.is_file() and paths.coreml_dylib.is_file()):
        pytest.skip("Native binaries not built yet")

    # 1. Misaligned seq_len % chunk_size != 0 must fail cleanly with clear stderr message
    proc_align = subprocess.run(
        [
            str(paths.ane_cli),
            "--stage", str(tmp_path / "missing.mlmodelc"),
            "--seq-len", "100",
            "--chunk-size", "32",
            "--weights-npz", str(tmp_path / "w.npz"),
            "--embed-safetensors", str(tmp_path / "e.safetensors"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_align.returncode != 0
    assert "divisible by chunk_size" in proc_align.stderr

    # 2. Construct a valid bit-3 (data descriptor, local comp_sz=0, uncomp_sz=0) .npz archive
    #    and a synthetic F16 safetensors file with `model.embed_tokens.weight` key.
    #    Running pplx-ane-cli with a nonexistent stage proves BOTH bit-3 ZIP parsing AND
    #    broadened safetensors key mmap succeeded (it reaches stage load error, not ZIP/embed error).
    hidden = 16
    vocab = 8
    embed_f16 = np.ones((vocab, hidden), dtype=np.float16)
    raw_emb = embed_f16.tobytes()
    hdr_dict = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [vocab, hidden],
            "data_offsets": [0, len(raw_emb)],
        }
    }
    hdr_b = json.dumps(hdr_dict, separators=(",", ":")).encode("utf-8")
    hdr_b += b" " * ((-len(hdr_b)) % 8)
    sf_path = tmp_path / "embed.safetensors"
    with sf_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_b)))
        fh.write(hdr_b)
        fh.write(raw_emb)

    # Create bit-3 data-descriptor ZIP containing contextual_projection.weight.npy (deflated)
    proj = np.eye(hidden, dtype=np.float32)
    npy_buf = io.BytesIO()
    np.save(npy_buf, proj)
    npy_bytes = npy_buf.getvalue()
    crc = zlib.crc32(npy_bytes) & 0xFFFFFFFF
    comp_obj = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    comp_bytes = comp_obj.compress(npy_bytes) + comp_obj.flush()
    fname = b"contextual_projection.weight.npy"

    # Local header with bit 3 set (flags=0x0008) and zeroed crc/comp_sz/uncomp_sz
    local_hdr = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50, 20, 0x0008, 8, 0, 0, 0, 0, 0, len(fname), 0,
    ) + fname
    data_desc = struct.pack("<IIII", 0x08074B50, crc, len(comp_bytes), len(npy_bytes))
    cd_offset = len(local_hdr) + len(comp_bytes) + len(data_desc)
    cd_hdr = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50, 20, 20, 0x0008, 8, 0, 0, crc, len(comp_bytes), len(npy_bytes),
        len(fname), 0, 0, 0, 0, 0, 0,
    ) + fname
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50, 0, 0, 1, 1, len(cd_hdr), cd_offset, 0,
    )
    npz_bit3 = tmp_path / "bit3_weights.npz"
    npz_bit3.write_bytes(local_hdr + comp_bytes + data_desc + cd_hdr + eocd)
    # Sanity check Python zipfile reads our synthetic archive cleanly
    with zipfile.ZipFile(npz_bit3, "r") as zf:
        assert "contextual_projection.weight.npy" in zf.namelist()

    proc_zip = subprocess.run(
        [
            str(paths.ane_cli),
            "--stage", str(tmp_path / "does_not_exist.mlmodelc"),
            "--seq-len", "64",
            "--chunk-size", "32",
            "--weights-npz", str(npz_bit3),
            "--embed-safetensors", str(sf_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_zip.returncode != 0
    assert "Failed to mmap" not in proc_zip.stderr
    assert "Failed to read contextual_projection" not in proc_zip.stderr
    assert "Failed to load CoreML ANE stage pipeline" in proc_zip.stderr

    # 3. Verify pplx_coreml_runner propagates NSError via pplx_ane_last_error and returns <0 on bad plan
    lib = _load_coreml_lib(paths.coreml_dylib)
    buf = ctypes.create_string_buffer(512)
    rc = lib.pplx_ane_inspect_plan(str(tmp_path / "missing.mlmodelc").encode("utf-8"), 0, buf, 512)
    assert rc < 0
    parsed = json.loads(buf.value.decode("utf-8"))
    assert "error" in parsed and parsed["error"]
    assert _coreml_last_error(lib) != ""

    bad_arr = (ctypes.c_char_p * 1)(str(tmp_path / "missing.mlmodelc").encode("utf-8"))
    handle = lib.pplx_ane_load_stages(bad_arr, 1, 64, 1024, 32, 32, 2, 0)
    assert not handle
    err_txt = _coreml_last_error(lib)
    assert "MLComputePlan" in err_txt or "failed" in err_txt.lower()
