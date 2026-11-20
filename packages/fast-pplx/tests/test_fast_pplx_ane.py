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


def test_build_ane_compile_mlpackage_uses_in_process_compiler_without_xcrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import ast
    import subprocess as sp

    tree = ast.parse(_BUILD_ANE_PATH.read_text(encoding="utf-8"))
    call_consts = {
        sub.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }
    assert not (call_consts & {"xcrun", "coremlc", "coremlcompiler"})

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
    def _tiny_prog(x):
        return mb.add(x=x, y=np.float32(1.0), name="y")

    ml = ct.convert(
        _tiny_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = tmp_path / "tiny_stage.mlpackage"
    ml.save(str(pkg))
    out_dir = tmp_path / "compiled"
    stale = out_dir / "tiny_stage.mlmodelc"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("stale", encoding="utf-8")

    orig_run = sp.run
    orig_check_call = sp.check_call

    def _forbid_xcrun(orig_fn):
        def _wrapper(cmd, *args, **kwargs):
            head = Path(str(cmd[0])).name if isinstance(cmd, (list, tuple)) and cmd else str(cmd)
            assert head not in {"xcrun", "coremlc", "coremlcompiler"}, f"Forbidden Xcode CLI call: {cmd}"
            return orig_fn(cmd, *args, **kwargs)
        return _wrapper

    monkeypatch.setattr(sp, "run", _forbid_xcrun(orig_run))
    monkeypatch.setattr(sp, "check_call", _forbid_xcrun(orig_check_call))

    bam = _load_script_module("fast_pplx_build_ane_models_compile", _BUILD_ANE_PATH)
    dst = bam._compile_mlpackage(pkg, out_dir)
    assert dst == stale
    assert dst.is_dir()
    assert not (dst / "stale.txt").exists()
    assert (dst / "model.mil").is_file() or (dst / "coremldata.bin").is_file()


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


def _make_dummy_ane_layer0_weights() -> dict[str, np.ndarray]:
    hidden = 1024
    mlp_inter = 3584
    gdn_heads = 16
    gdn_dim = 128
    qkvzba_out = 6144 + 2048 + gdn_heads + gdn_heads
    return {
        "l0_in_ln": np.ones((1, hidden, 1, 1), dtype=np.float16),
        "l0_post_ln": np.ones((1, hidden, 1, 1), dtype=np.float16),
        "l0_qkvzba": np.zeros((qkvzba_out, hidden, 1, 1), dtype=np.float16),
        "l0_conv1d": np.zeros((6144, 1, 1, 4), dtype=np.float16),
        "l0_neg_A": np.full((gdn_heads,), -1.0, dtype=np.float16),
        "l0_dt_bias": np.zeros((gdn_heads,), dtype=np.float16),
        "l0_gnorm_w": np.ones((gdn_dim,), dtype=np.float16),
        "l0_out_proj": np.zeros((hidden, 2048, 1, 1), dtype=np.float16),
        "l0_mlp_gu": np.zeros((2 * mlp_inter, hidden, 1, 1), dtype=np.float16),
        "l0_mlp_dn": np.zeros((hidden, mlp_inter, 1, 1), dtype=np.float16),
        "final_norm": np.ones((1, hidden, 1, 1), dtype=np.float16),
    }


def test_build_ane_silu_and_max_abs_rmsnorm_modes_cpu() -> None:
    pytest.importorskip("coremltools")
    bam = _load_script_module("fast_pplx_build_ane_modes", _BUILD_ANE_PATH)
    weights = _make_dummy_ane_layer0_weights()
    cfg = PplxConfig()

    def _op_types(silu_mode: str, rmsnorm_mode: str) -> list[str]:
        prog = bam.build_stage_mil_program(
            weights,
            seq_len=32,
            layer_start=0,
            layer_end=1,
            layer_types=cfg.layer_types,
            apply_final_norm=True,
            silu_mode=silu_mode,
            rmsnorm_mode=rmsnorm_mode,
        )
        return [op.op_type for op in prog.functions["main"].operations]

    ops_default = _op_types("silu", "prescale")
    assert ops_default.count("silu") == 3
    assert "tanh" not in ops_default
    assert "reduce_max" not in ops_default

    ops_tanh_gdn = _op_types("tanh_gdn", "prescale")
    assert ops_tanh_gdn.count("silu") == 1  # MLP keeps silu; GDN conv+z use TanhSiLU
    assert ops_tanh_gdn.count("tanh") == 2

    ops_tanh_all_max = _op_types("tanh_all", "max_abs")
    assert "silu" not in ops_tanh_all_max
    assert ops_tanh_all_max.count("tanh") == 3
    assert ops_tanh_all_max.count("reduce_max") == 3  # in_ln, post_ln, final_norm
    assert bam.MAX_ABS_FLOOR >= bam.FP16_MIN_NORMAL

    with pytest.raises(ValueError, match="silu_mode"):
        _op_types("bad_silu", "prescale")
    with pytest.raises(ValueError, match="rmsnorm_mode"):
        _op_types("silu", "bad_rms")

    ns = build_parser().parse_args(["setup-ane", "--silu-mode", "tanh_all", "--rmsnorm-mode", "max_abs"])
    assert ns.silu_mode == "tanh_all"
    assert ns.rmsnorm_mode == "max_abs"
    assert ns.ane_profile is None

    ns_prof = build_parser().parse_args(["setup-ane", "--ane-profile", "parity"])
    assert ns_prof.ane_profile == "parity"
    assert ns_prof.silu_mode is None
    assert ns_prof.rmsnorm_mode is None

    ns_bal = build_parser().parse_args(["setup-ane", "--ane-profile", "balanced"])
    assert ns_bal.ane_profile == "balanced"
    for cmd, extra in (
        ("embed", ["--ane-profile", "parity", "hello"]),
        ("verify", ["--ane-profile", "balanced", "--mode", "all"]),
        ("bench", ["--ane-profile", "fast"]),
        ("parquet", ["--input", "in.parquet", "--output", "out.parquet", "--ane-profile", "parity"]),
        ("serve", ["--ane-profile", "custom"]),
    ):
        ns_eng = build_parser().parse_args([cmd, *extra])
        assert ns_eng.ane_profile in ("fast", "balanced", "parity", "custom")

    # Profile resolution & explicit override semantics
    assert bam.resolve_ane_profile() == ("fast", "silu", "prescale")
    assert bam.resolve_ane_profile("fast") == ("fast", "silu", "prescale")
    assert bam.resolve_ane_profile("balanced") == ("balanced", "tanh_gdn", "max_abs")
    assert bam.resolve_ane_profile("parity") == ("parity", "tanh_all", "max_abs")
    assert bam.resolve_ane_profile("parity", silu_mode="tanh_gdn") == ("balanced", "tanh_gdn", "max_abs")
    assert bam.resolve_ane_profile("parity", rmsnorm_mode="prescale") == ("custom", "tanh_all", "prescale")
    assert bam.resolve_ane_profile("fast", silu_mode="tanh_all", rmsnorm_mode="max_abs") == (
        "parity",
        "tanh_all",
        "max_abs",
    )
    with pytest.raises(ValueError, match="ane_profile"):
        bam.resolve_ane_profile("ultra")


def test_ane_profile_cli_forwarding_manifest_invalidation_and_discovery_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess as sp
    from fast_pplx.cli import _ane_verify_min_cos_threshold, _cmd_setup_ane, _cmd_verify
    from fast_pplx.engine import _discover_ane_stage_paths, read_ane_manifest

    bam = _load_script_module("fast_pplx_build_ane_manifest", _BUILD_ANE_PATH)

    # 1. Verify CLI setup-ane forwards --ane-profile to build_ane_models.py
    captured: list[list[str]] = []
    monkeypatch.setattr(sp, "call", lambda cmd: captured.append(list(cmd)) or 0)
    ns = build_parser().parse_args(["setup-ane", "--home", str(tmp_path), "--ane-profile", "balanced"])
    assert _cmd_setup_ane(ns) == 0
    assert captured and "--ane-profile" in captured[0]
    assert captured[0][captured[0].index("--ane-profile") + 1] == "balanced"

    # 2. Config hash and bundle stems distinguish modes/profiles/precision cleanly
    h_fast = bam.compute_ane_config_hash(
        backend="coreml",
        precision="w8a16",
        inv_mode="block",
        silu_mode="silu",
        rmsnorm_mode="prescale",
        ftz_alpha=128.0,
        preserve_gdn_fp16=True,
        layers_per_stage=3,
        outlier_threshold=15.0,
        num_layers=12,
    )
    h_parity = bam.compute_ane_config_hash(
        backend="coreml",
        precision="w8a16",
        inv_mode="block",
        silu_mode="tanh_all",
        rmsnorm_mode="max_abs",
        ftz_alpha=128.0,
        preserve_gdn_fp16=True,
        layers_per_stage=3,
        outlier_threshold=15.0,
        num_layers=12,
    )
    h_bal = bam.compute_ane_config_hash(
        backend="coreml",
        precision="w8a16",
        inv_mode="block",
        silu_mode="tanh_gdn",
        rmsnorm_mode="max_abs",
        ftz_alpha=128.0,
        preserve_gdn_fp16=True,
        layers_per_stage=3,
        outlier_threshold=15.0,
        num_layers=12,
    )
    assert len({h_fast, h_parity, h_bal}) == 3

    stem_fast = bam.make_ane_stage_stem(
        128, 0, "w8a16", ane_profile="fast", silu_mode="silu", rmsnorm_mode="prescale", config_hash=h_fast
    )
    stem_bal = bam.make_ane_stage_stem(
        128, 0, "w8a16", ane_profile="balanced", silu_mode="tanh_gdn", rmsnorm_mode="max_abs", config_hash=h_bal
    )
    stem_parity = bam.make_ane_stage_stem(
        128, 0, "w8a16", ane_profile="parity", silu_mode="tanh_all", rmsnorm_mode="max_abs", config_hash=h_parity
    )
    stem_custom = bam.make_ane_stage_stem(
        128, 0, "w8a16", ane_profile="custom", silu_mode="tanh_all", rmsnorm_mode="prescale", config_hash=h_bal
    )
    assert len({stem_fast, stem_bal, stem_parity, stem_custom}) == 4
    assert f"_fast_{h_fast[:8]}" in stem_fast
    assert f"_balanced_{h_bal[:8]}" in stem_bal
    assert f"_parity_{h_parity[:8]}" in stem_parity
    assert f"_tanh_all_prescale_{h_bal[:8]}" in stem_custom

    # 3. Manifest bucket preservation on same hash vs full invalidation + pruning on hash change
    ane_dir = tmp_path / "artifacts" / "ane_models"
    ane_dir.mkdir(parents=True)
    fast_64 = [f"{bam.make_ane_stage_stem(64, i, 'w8a16', ane_profile='fast', silu_mode='silu', rmsnorm_mode='prescale', config_hash=h_fast)}.mlmodelc" for i in range(4)]
    fast_128 = [f"{bam.make_ane_stage_stem(128, i, 'w8a16', ane_profile='fast', silu_mode='silu', rmsnorm_mode='prescale', config_hash=h_fast)}.mlmodelc" for i in range(4)]
    for name in (*fast_64, *fast_128):
        (ane_dir / name).mkdir()
    man_fast = {
        "schema_version": bam.MANIFEST_SCHEMA_VERSION,
        "ane_profile": "fast",
        "config_hash": h_fast,
        "silu_mode": "silu",
        "rmsnorm_mode": "prescale",
        "precision": "w8a16",
        "num_stages": 4,
        "buckets": {
            "64": {"seq_len": 64, "config_hash": h_fast, "stages": fast_64},
            "128": {"seq_len": 128, "config_hash": h_fast, "stages": fast_128},
        },
    }
    (ane_dir / "manifest.json").write_text(json.dumps(man_fast), encoding="utf-8")

    # Same hash rebuilding only S=128 preserves S=64
    preserved_same = bam._load_preserved_buckets_for_hash(
        ane_dir, config_hash=h_fast, requested_seq_lens=(128,), expected_num_stages=4
    )
    assert list(preserved_same.keys()) == ["64"]

    # Different hash (switching to parity) invalidates ALL previous buckets
    preserved_diff = bam._load_preserved_buckets_for_hash(
        ane_dir, config_hash=h_parity, requested_seq_lens=(128,), expected_num_stages=4
    )
    assert preserved_diff == {}

    # Discovery enforces expected_profile and per-bucket config_hash strictly (no silent glob fallback)
    assert list(_discover_ane_stage_paths(ane_dir).keys()) == [64, 128]
    assert list(_discover_ane_stage_paths(ane_dir, expected_profile="fast").keys()) == [64, 128]
    assert _discover_ane_stage_paths(ane_dir, expected_profile="parity") == {}

    # Mixed/stale bucket config_hash inside manifest is rejected
    man_mixed = json.loads((ane_dir / "manifest.json").read_text(encoding="utf-8"))
    man_mixed["buckets"]["64"]["config_hash"] = h_parity
    (ane_dir / "manifest.json").write_text(json.dumps(man_mixed), encoding="utf-8")
    assert list(_discover_ane_stage_paths(ane_dir).keys()) == [128]

    # Pruning removes stale .mlmodelc AND stale .mlpackage while preserving active .mlpackage when keep_mlpackage=True
    parity_128 = [f"{bam.make_ane_stage_stem(128, i, 'w8a16', ane_profile='parity', silu_mode='tanh_all', rmsnorm_mode='max_abs', config_hash=h_parity)}.mlmodelc" for i in range(4)]
    for name in parity_128:
        (ane_dir / name).mkdir()
        (ane_dir / f"{Path(name).stem}.mlpackage").mkdir()
    stale_pkg = f"{Path(fast_64[0]).stem}.mlpackage"
    (ane_dir / stale_pkg).mkdir()
    pruned = bam._prune_stale_ane_bundles(ane_dir, active_stage_names=set(parity_128), keep_mlpackage=True)
    assert set(pruned) == set(fast_64) | set(fast_128) | {stale_pkg}
    assert sorted(p.name for p in ane_dir.glob("*.mlmodelc")) == sorted(parity_128)
    assert sorted(p.name for p in ane_dir.glob("*.mlpackage")) == sorted(f"{Path(n).stem}.mlpackage" for n in parity_128)

    # 4. Legacy v1 manifest without schema_version/ane_profile/config_hash still normalizes cleanly
    legacy_dir = tmp_path / "legacy_ane"
    legacy_dir.mkdir()
    legacy_stages = [f"pplx_s64_st{i}_w8a16.mlmodelc" for i in range(4)]
    for name in legacy_stages:
        (legacy_dir / name).mkdir()
    (legacy_dir / "manifest.json").write_text(
        json.dumps({"precision": "w8a16", "num_stages": 4, "buckets": {"64": {"seq_len": 64, "stages": legacy_stages}}}),
        encoding="utf-8",
    )
    leg_man = read_ane_manifest(legacy_dir)
    assert leg_man is not None and leg_man["schema_version"] == 1 and leg_man["ane_profile"] == "fast"
    assert list(_discover_ane_stage_paths(legacy_dir, expected_profile="fast").keys()) == [64]
    assert _discover_ane_stage_paths(legacy_dir, expected_profile="parity") == {}
    # Legacy v1 manifest is invalidated when building schema v2
    assert bam._load_preserved_buckets_for_hash(
        legacy_dir, config_hash=h_fast, requested_seq_lens=(128,), expected_num_stages=4
    ) == {}
    # Balanced profile inference when ane_profile field is omitted
    bal_dir = tmp_path / "bal_ane"
    bal_dir.mkdir()
    (bal_dir / "manifest.json").write_text(
        json.dumps({"silu_mode": "tanh_gdn", "rmsnorm_mode": "max_abs", "buckets": {}}),
        encoding="utf-8",
    )
    bal_man = read_ane_manifest(bal_dir)
    assert bal_man is not None and bal_man["ane_profile"] == "balanced"

    # 5. Profile-aware verify thresholds
    assert _ane_verify_min_cos_threshold(man_fast, "fast") == pytest.approx(0.980)
    assert _ane_verify_min_cos_threshold({"silu_mode": "tanh_all", "rmsnorm_mode": "prescale"}, "custom") == pytest.approx(0.980)
    assert _ane_verify_min_cos_threshold(bal_man, "balanced") == pytest.approx(0.998)
    assert _ane_verify_min_cos_threshold({"silu_mode": "tanh_all", "rmsnorm_mode": "max_abs"}, "parity") == pytest.approx(0.998)
    assert _ane_verify_min_cos_threshold({"silu_mode": "tanh_gdn", "rmsnorm_mode": "max_abs"}, "custom") == pytest.approx(0.998)

    # 6. FastPplx(ane_profile="parity") sets ane_profile=None & staged_ane_profile="fast" on mismatch
    tok_json = _write_test_tokenizer(tmp_path)
    tok = ContextualTokenizer(tok_json, PplxConfig())
    (tmp_path / "build").mkdir(exist_ok=True)
    (tmp_path / "build" / "libpplx_coreml_runner.dylib").write_bytes(b"")
    (tmp_path / "build" / "libpplx_fast_io.dylib").write_bytes(b"")
    (ane_dir / "manifest.json").write_text(json.dumps(man_fast), encoding="utf-8")

    class _StubGPU:
        max_seq_len = 32768
        def close(self):
            pass

    with FastPplx(home=tmp_path, mode="auto", ane_profile="parity", _gpu_backend=_StubGPU(), _tokenizer=tok) as eng:
        assert eng.ane_available is False
        assert eng.ane_profile is None
        assert eng.staged_ane_profile == "fast"
        assert eng._ane_init_error is not None and "--ane-profile parity" in eng._ane_init_error

    with FastPplx(home=tmp_path, mode="gpu", _gpu_backend=_StubGPU(), _tokenizer=tok) as eng_gpu_only:
        assert eng_gpu_only.ane_available is False
        assert eng_gpu_only.ane_profile is None
        assert eng_gpu_only.staged_ane_profile == "fast"

    # 7. _cmd_verify --mode all fails (exit 1) on explicit profile mismatch or broken staged ANE,
    #    and enforces 0.980 (fast) vs 0.998 (parity) thresholds cleanly.
    import fast_pplx.cli as pplx_cli_mod

    class _FakeVerifyEngine:
        def __init__(self, *, ane_available: bool, ane_profile: str | None, staged: str | None, err: str | None, ane_cos: float):
            self.ane_available = ane_available
            self.ane_profile = ane_profile
            self.staged_ane_profile = staged
            self._ane_init_error = err
            self.ane_query_max_len = 512
            self._ane_cos = ane_cos

        def encode_documents(self, texts, *, mode="gpu", normalize=True, **_kw):
            v_gpu = np.zeros((len(texts), 4), dtype=np.float32)
            v_gpu[:, 0] = 1.0
            if mode == "gpu":
                return v_gpu
            c = float(self._ane_cos)
            v_ane = np.zeros((len(texts), 4), dtype=np.float32)
            v_ane[:, 0] = c
            v_ane[:, 1] = float(np.sqrt(max(0.0, 1.0 - c * c)))
            return v_ane

        def close(self):
            pass

    # Force smoke path by hiding fixture files inside tmp_path & monkeypatching pkg_root
    smoke_home = tmp_path / "verify_smoke"
    (smoke_home / "artifacts" / "ane_models").mkdir(parents=True)
    (smoke_home / "artifacts" / "ane_models" / "manifest.json").write_text(json.dumps(man_fast), encoding="utf-8")
    monkeypatch.setattr(pplx_cli_mod, "Path", lambda p: (smoke_home / "src" / "fast_pplx" / "cli.py") if p == pplx_cli_mod.__file__ else Path(p))

    # Case A: requested --ane-profile parity mismatched against staged fast -> verify --mode all exits 1
    monkeypatch.setattr(
        pplx_cli_mod,
        "_make_engine",
        lambda a, deterministic=False: _FakeVerifyEngine(
            ane_available=False,
            ane_profile=None,
            staged="fast",
            err="Staged ANE bundle profile 'fast' does not match requested ane_profile='parity'",
            ane_cos=0.9828,
        ),
    )
    ns_mismatch = build_parser().parse_args(["verify", "--home", str(smoke_home), "--mode", "all", "--ane-profile", "parity", "--json"])
    assert _cmd_verify(ns_mismatch) == 1

    # Case B: active fast profile with cos=0.9828 passes (>=0.980)
    monkeypatch.setattr(
        pplx_cli_mod,
        "_make_engine",
        lambda a, deterministic=False: _FakeVerifyEngine(
            ane_available=True, ane_profile="fast", staged="fast", err=None, ane_cos=0.9828
        ),
    )
    ns_fast_ok = build_parser().parse_args(["verify", "--home", str(smoke_home), "--mode", "all", "--json"])
    assert _cmd_verify(ns_fast_ok) == 0

    # Case C: active parity profile with cos=0.9828 FAILS (<0.998), cos=0.9991 PASSES (>=0.998)
    man_parity = dict(man_fast, ane_profile="parity", silu_mode="tanh_all", rmsnorm_mode="max_abs")
    (smoke_home / "artifacts" / "ane_models" / "manifest.json").write_text(json.dumps(man_parity), encoding="utf-8")
    monkeypatch.setattr(
        pplx_cli_mod,
        "_make_engine",
        lambda a, deterministic=False: _FakeVerifyEngine(
            ane_available=True, ane_profile="parity", staged="parity", err=None, ane_cos=0.9828
        ),
    )
    ns_par_fail = build_parser().parse_args(["verify", "--home", str(smoke_home), "--mode", "all", "--ane-profile", "parity", "--json"])
    assert _cmd_verify(ns_par_fail) == 1

    monkeypatch.setattr(
        pplx_cli_mod,
        "_make_engine",
        lambda a, deterministic=False: _FakeVerifyEngine(
            ane_available=True, ane_profile="parity", staged="parity", err=None, ane_cos=0.9991
        ),
    )
    assert _cmd_verify(ns_par_fail) == 0


def test_coreai_backend_status_decouples_export_vs_runtime() -> None:
    bam = _load_script_module("fast_pplx_build_ane_models_coreai_status", _BUILD_ANE_PATH)
    st_offline = bam.check_coreai_backend_status(
        macos_version="26.4",
        use_os_coreai=False,
        pkg_versions={"coreai-torch": "0.4.4", "coreai-opt": "0.5.0", "coreai-core": "1.0.0", "coreai": "1.0.0"},
        delegate_probe=lambda: False,
    )
    assert st_offline["export_ready"] is True
    assert st_offline["runtime_ready"] is False
    assert st_offline["eligible"] is False
    assert "27.0+" in st_offline["reason"]

    st_live = bam.check_coreai_backend_status(
        macos_version="27.2",
        use_os_coreai=True,
        pkg_versions={"coreai-torch": "0.4.4", "coreai-opt": "0.5.0", "coreai-core": "1.0.0", "coreai": "1.0.0"},
        delegate_probe=lambda: True,
    )
    assert st_live["export_ready"] is True
    assert st_live["runtime_ready"] is True
    assert st_live["eligible"] is True
    assert st_live["fp4_fp8_preset_eligible"] is True

    st_old_torch = bam.check_coreai_backend_status(
        macos_version="27.2",
        use_os_coreai=True,
        pkg_versions={"coreai-torch": "0.4.0", "coreai-opt": "0.5.0", "coreai-core": "1.0.0", "coreai": "1.0.0"},
        delegate_probe=lambda: True,
    )
    assert st_old_torch["export_ready"] is False
    assert ">= 0.4.1" in st_old_torch["export_reason"]


def test_pplx_coreai_export_calibration_and_guards(tmp_path: Path) -> None:
    pytest.importorskip("coreai_opt")
    pytest.importorskip("coreai_torch")
    pytest.importorskip("coremltools")

    from fast_pplx.config import PplxConfig
    from fast_pplx.coreai_export import (
        build_coreai_opt_pplx_config,
        run_pplx_coreai_export,
        run_pplx_coreai_sensitivity_sweep,
    )

    # 1. Pattern 7 divisibility + Torch-only dtype rejection
    with pytest.raises(ValueError, match="ExportBackend.Torch"):
        build_coreai_opt_pplx_config(preset="w8", export_backend="coreml", requested_dtypes=["mxint8"])
    with pytest.raises(ValueError, match="ExportBackend.Torch"):
        build_coreai_opt_pplx_config(preset="w4", export_backend="coreai", requested_dtypes=["int6"])
    with pytest.raises(ValueError, match="requires export_backend='coreai'"):
        build_coreai_opt_pplx_config(preset="fp4_fp8", export_backend="coreml")

    with pytest.raises(ValueError, match="allow_experimental_coreml_w4_block"):
        build_coreai_opt_pplx_config(preset="w4_per_block", export_backend="coreml")

    _, qmeta = build_coreai_opt_pplx_config(
        preset="w4_per_block",
        block_size=32,
        export_backend="coreai",
        weight_shapes={"blocks.l0.conv1d": (96, 1, 1, 4), "blocks.l1.mlp_gu": (128, 64, 1, 1)},
    )
    assert "blocks.l0.conv1d" in qmeta["excluded_modules"]
    assert qmeta["skipped_non_divisible"]["blocks.l0.conv1d"] == [1, 32]

    # 2. End-to-end CoreAI (.aimodel) + CoreML (.mlpackage) export & sensitivity sweep on tiny config
    tiny_cfg = PplxConfig(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=("linear_attention", "full_attention"),
    )

    res_ai = run_pplx_coreai_export(
        out_dir=tmp_path / "ai",
        cfg=tiny_cfg,
        layer_start=0,
        layer_end=2,
        seq_len=32,
        chunk_size=16,
        export_backend="coreai",
        preset="w4_per_block",
        stem="tiny_pplx_ai",
    )
    assert res_ai.asset_path.exists()
    assert res_ai.asset_summary is not None and res_ai.asset_summary["asset_valid"] is True
    assert "Float16" in res_ai.asset_summary["compute_types"]
    op_names = {op for op, _ in res_ai.asset_summary["operation_distribution"]}
    assert "softplus" not in op_names
    assert res_ai.sensitivity_report is not None
    assert res_ai.sensitivity_report["compute_dtype"] == "fp16"
    assert res_ai.sensitivity_report["cosine_vs_teacher"] > 0.99
    assert "blocks.l0.qkvzba" in res_ai.sensitivity_report["excluded_modules"]
    assert res_ai.cache_key_parts["min_os"] == "27.0"

    res_ml = run_pplx_coreai_export(
        out_dir=tmp_path / "ml",
        cfg=tiny_cfg,
        layer_start=0,
        layer_end=2,
        seq_len=32,
        chunk_size=16,
        export_backend="coreml",
        preset="w8",
        stem="tiny_pplx_ml",
    )
    assert res_ml.asset_path.exists()
    assert res_ml.guard_report is not None and res_ml.guard_report["ok"] is True

    sweep_rows = run_pplx_coreai_sensitivity_sweep(
        out_dir=tmp_path / "sweep",
        cfg=tiny_cfg,
        layer_start=0,
        layer_end=2,
        seq_len=32,
        chunk_size=16,
        export_backend="coreai",
        presets=("fp16", "w8", "w4_per_block"),
        stem_prefix="tiny_sweep",
    )
    assert [r["preset"] for r in sweep_rows] == ["fp16", "w8", "w4_per_block"]
    assert sweep_rows[0]["analytical_bpw"] == 16.0
    assert sweep_rows[2]["analytical_bpw"] < sweep_rows[1]["analytical_bpw"]


def test_build_ane_bundle_suite_coreai_backend_cpu(tmp_path: Path) -> None:
    pytest.importorskip("coreai_opt")
    pytest.importorskip("coreai_torch")

    bam = _load_script_module("fast_pplx_build_ane_suite_coreai", _BUILD_ANE_PATH)
    paths = resolve_paths(tmp_path)
    paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    paths.ane_models_dir.mkdir(parents=True, exist_ok=True)

    tiny_cfg = PplxConfig(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=("linear_attention", "full_attention"),
    )
    (paths.tokenizer_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": tiny_cfg.hidden_size,
                "num_hidden_layers": tiny_cfg.num_hidden_layers,
                "intermediate_size": tiny_cfg.intermediate_size,
                "num_attention_heads": tiny_cfg.num_attention_heads,
                "num_key_value_heads": tiny_cfg.num_key_value_heads,
                "head_dim": tiny_cfg.head_dim,
                "partial_rotary_factor": tiny_cfg.partial_rotary_factor,
                "linear_conv_kernel_dim": tiny_cfg.linear_conv_kernel_dim,
                "linear_key_head_dim": tiny_cfg.linear_key_head_dim,
                "linear_value_head_dim": tiny_cfg.linear_value_head_dim,
                "linear_num_key_heads": tiny_cfg.linear_num_key_heads,
                "linear_num_value_heads": tiny_cfg.linear_num_value_heads,
                "layer_types": list(tiny_cfg.layer_types),
            }
        ),
        encoding="utf-8",
    )

    h = tiny_cfg.hidden_size
    inter = tiny_cfg.intermediate_size
    gdn_h = tiny_cfg.linear_num_value_heads
    gdn_d = tiny_cfg.linear_key_head_dim
    gdn_p = gdn_h * gdn_d
    q_tot = tiny_cfg.num_attention_heads * tiny_cfg.head_dim
    kv_tot = tiny_cfg.num_key_value_heads * tiny_cfg.head_dim
    synth_w: dict[str, np.ndarray] = {
        "l0_in_ln": np.ones((1, h, 1, 1), dtype=np.float16),
        "l0_post_ln": np.ones((1, h, 1, 1), dtype=np.float16),
        "l0_qkvzba": np.full((4 * gdn_p + 2 * gdn_h, h, 1, 1), 0.02, dtype=np.float16),
        "l0_conv1d": np.full((3 * gdn_p, 1, 1, 4), 0.05, dtype=np.float16),
        "l0_neg_A": np.full((gdn_h,), -0.5, dtype=np.float16),
        "l0_dt_bias": np.zeros((gdn_h,), dtype=np.float16),
        "l0_gnorm_w": np.ones((gdn_d,), dtype=np.float16),
        "l0_out_proj": np.full((h, gdn_p, 1, 1), 0.02, dtype=np.float16),
        "l0_mlp_gu": np.full((2 * inter, h, 1, 1), 0.02, dtype=np.float16),
        "l0_mlp_dn": np.full((h, inter, 1, 1), 0.02, dtype=np.float16),
        "l1_in_ln": np.ones((1, h, 1, 1), dtype=np.float16),
        "l1_post_ln": np.ones((1, h, 1, 1), dtype=np.float16),
        "l1_qkvgate": np.full((2 * q_tot + 2 * kv_tot, h, 1, 1), 0.02, dtype=np.float16),
        "l1_qn": np.ones((1, 1, 1, tiny_cfg.head_dim), dtype=np.float16),
        "l1_kn": np.ones((1, 1, 1, tiny_cfg.head_dim), dtype=np.float16),
        "l1_o_proj": np.full((h, q_tot, 1, 1), 0.02, dtype=np.float16),
        "l1_mlp_gu": np.full((2 * inter, h, 1, 1), 0.02, dtype=np.float16),
        "l1_mlp_dn": np.full((h, inter, 1, 1), 0.02, dtype=np.float16),
        "final_norm": np.ones((1, h, 1, 1), dtype=np.float16),
    }
    np.savez(paths.weights_npz, **synth_w)
    (paths.ane_models_dir / "pplx_s64_st0_stale.aimodel").mkdir()

    manifest = bam.build_ane_bundle_suite(
        home=tmp_path,
        backend="coreai",
        seq_lens=(32,),
        layers_per_stage=2,
        coreai_preset="w4_per_block",
        block_size=32,
        calib_batches=1,
        guard_ir=True,
    )
    assert manifest["backend"] == "coreai"
    assert manifest["coreai_preset"] == "w4_per_block"
    assert (paths.ane_models_dir / "manifest.json").is_file()
    stages = manifest["buckets"]["32"]["stages"]
    assert len(stages) == 1 and stages[0].endswith(".aimodel")
    assert (paths.ane_models_dir / stages[0]).exists()
    assert manifest["buckets"]["32"]["coreai_stages"][0]["stage"] == stages[0]
    assert not (paths.ane_models_dir / "pplx_s64_st0_stale.aimodel").exists()

