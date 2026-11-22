"""CPU unit tests for `retrieval_index`, `CorpusBackfillConfig.for_engine`, and `kv_quant`."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from fast_shared import (
    CompactRetrievalIndex,
    CorpusBackfillConfig,
    HadamardTransformBank,
    QuantizedKVCache,
    build_engine_index,
    dequantize_int8_symmetric,
    fused_quantized_sdpa,
    normalize_kv_mode,
    pack_binary_bits,
    quantize_int8_symmetric,
    recommended_backfill_mrl_dim,
    resolve_index_spec,
    score_document_windows,
)


@dataclass
class _DummyEngine:
    dim: int
    max_dim: int
    is_mrl_trained: bool

    def encode_documents(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        rng = np.random.default_rng(len(texts) + self.dim)
        raw = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_model_aware_backfill_and_index_spec() -> None:
    mrl_eng = _DummyEngine(dim=768, max_dim=768, is_mrl_trained=True)
    pplx_eng = _DummyEngine(dim=1024, max_dim=1024, is_mrl_trained=False)
    bge_eng = _DummyEngine(dim=384, max_dim=384, is_mrl_trained=False)

    # Preserve direct dataclass default backwards-compatibility
    assert CorpusBackfillConfig().mrl_dim == 512

    # Engine-aware defaults protect non-MRL models from premature 512d truncation
    assert recommended_backfill_mrl_dim(mrl_eng) == 512
    assert recommended_backfill_mrl_dim(pplx_eng) is None
    assert recommended_backfill_mrl_dim(bge_eng) is None
    assert CorpusBackfillConfig.for_engine(mrl_eng).mrl_dim == 512
    assert CorpusBackfillConfig.for_engine(pplx_eng).mrl_dim is None
    assert CorpusBackfillConfig.for_engine(bge_eng).mrl_dim is None
    assert CorpusBackfillConfig.for_engine(pplx_eng, mrl_dim=512).mrl_dim == 512

    assert resolve_index_spec(raw_dim=768, mode="auto", is_mrl_trained=True) == ("mrl_int8", 512)
    assert resolve_index_spec(raw_dim=1024, mode="auto", is_mrl_trained=False) == ("int8", 1024)
    assert resolve_index_spec(raw_dim=384, mode="auto", is_mrl_trained=False) == ("int8", 384)
    assert resolve_index_spec(raw_dim=1024, mode="binary_int8", is_mrl_trained=False) == ("binary_int8", 1024)


def test_compact_retrieval_index_single_and_multivector_parity() -> None:
    rng = np.random.default_rng(42)
    docs = rng.standard_normal((256, 768)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    # Plant clear needles for 8 queries
    queries = docs[:8].copy() + 0.05 * rng.standard_normal((8, 768)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    idx_fp32 = CompactRetrievalIndex(docs, mode="fp32")
    idx_i8_np = CompactRetrievalIndex(docs, mode="int8", use_neon=False)
    idx_i8_neon = CompactRetrievalIndex(docs, mode="int8", use_neon=True)
    idx_bin_np = CompactRetrievalIndex(docs, mode="binary_int8", is_mrl_trained=True, use_neon=False)
    idx_bin_neon = CompactRetrievalIndex(docs, mode="binary_int8", is_mrl_trained=True, use_neon=True)

    assert idx_i8_np.compression_ratio_vs_fp32 == pytest.approx(3072.0 / 770.0, rel=1e-2)
    assert idx_bin_neon.dim == 512
    assert idx_bin_neon.binary_bits == 512

    top_fp32, sc_fp32 = idx_fp32.search(queries, k=5)
    top_i8_np, sc_i8_np = idx_i8_np.search(queries, k=5)
    top_i8_neon, sc_i8_neon = idx_i8_neon.search(queries, k=5)
    assert np.array_equal(top_fp32[:, 0], np.arange(8, dtype=np.int32))
    assert np.array_equal(top_i8_np, top_i8_neon)
    np.testing.assert_allclose(sc_i8_np, sc_i8_neon, atol=1e-4)
    np.testing.assert_allclose(sc_i8_neon, sc_fp32, atol=1e-2)

    top_bin_np, sc_bin_np = idx_bin_np.search(queries, k=5, rerank_top_r=64)
    top_bin_neon, sc_bin_neon = idx_bin_neon.search(queries, k=5, rerank_top_r=64)
    assert np.array_equal(top_bin_np[:, 0], np.arange(8, dtype=np.int32))
    assert np.array_equal(top_bin_np, top_bin_neon)
    np.testing.assert_allclose(sc_bin_np, sc_bin_neon, atol=1e-4)

    # Multi-vector MaxSim rescue over diluted 16-window haystack documents
    haystack = rng.standard_normal((160, 1024)).astype(np.float32)
    haystack /= np.linalg.norm(haystack, axis=1, keepdims=True)
    doc_offsets = np.arange(0, 161, 16, dtype=np.int32)  # 10 docs x 16 windows
    q_needle = rng.standard_normal((1024,)).astype(np.float32)
    q_needle /= np.linalg.norm(q_needle)
    # Place exact needle inside window 13 of doc 7
    haystack[7 * 16 + 13] = q_needle

    mv_idx = CompactRetrievalIndex(haystack, doc_offsets=doc_offsets, mode="int8", agg="maxsim")
    ids_max, sc_max = mv_idx.search(q_needle, k=3, agg="maxsim")
    assert int(ids_max[0]) == 7
    assert float(sc_max[0]) > 0.98

    ids_top2, _ = mv_idx.search(q_needle, k=3, agg="top2_mean")
    assert int(ids_top2[0]) == 7

    mv_bin = CompactRetrievalIndex(haystack, doc_offsets=doc_offsets, mode="binary_int8", agg="maxsim")
    ids_mbin, _ = mv_bin.search(q_needle, k=3, rerank_top_r=16)
    assert int(ids_mbin[0]) == 7

    # Engine helper test
    eng = _DummyEngine(dim=768, max_dim=768, is_mrl_trained=True)
    built = build_engine_index(eng, ["a", "b", "c"], mode="auto")
    assert built.mode == "mrl_int8" and built.dim == 512


def test_quantize_int8_and_window_scoring_edge_cases() -> None:
    q, s = quantize_int8_symmetric(np.zeros((2, 64), dtype=np.float32))
    dq = dequantize_int8_symmetric(q, s)
    assert dq.shape == (2, 64) and np.all(dq == 0.0)
    bits = pack_binary_bits(np.ones((3, 128), dtype=np.float32))
    assert bits.shape == (3, 16) and np.all(bits == 255)

    ws = np.array([[0.1, 0.9, 0.2, 0.4, 0.8]], dtype=np.float32)
    offs = [0, 2, 2, 5]  # includes empty middle doc!
    ms = score_document_windows(ws, offs, agg="maxsim")
    assert ms.shape == (1, 3)
    assert ms[0, 0] == pytest.approx(0.9)
    assert ms[0, 1] < -1e20
    assert ms[0, 2] == pytest.approx(0.8)


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_hadamard_transform_bank_orthogonality_cpu(head_dim: int) -> None:
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        bank = HadamardTransformBank(head_dim=head_dim, use_metal=False)
        rng = np.random.default_rng(head_dim)
        x_np = rng.standard_normal((2, 4, 5, head_dim)).astype(np.float32)
        x_mx = mx.array(x_np, dtype=mx.float32)
        y = bank.transform(x_mx, inverse=False)
        rt = bank.transform(y, inverse=True)
        mx.eval(y, rt)
        # Norm preservation (isometry) + exact inverse
        np.testing.assert_allclose(np.asarray(rt), x_np, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(y), axis=-1),
            np.linalg.norm(x_np, axis=-1),
            atol=1e-5,
            rtol=1e-5,
        )
    finally:
        mx.set_default_device(prev_dev)


def test_quantized_kv_cache_and_quarot_sdpa_parity_cpu() -> None:
    assert normalize_kv_mode(False) == "bf16"
    assert normalize_kv_mode(True) == "q8"
    assert normalize_kv_mode("q4_quarot") == "q4_quarot"

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(123)
        b, n_q, n_kv, s_pre, d = 2, 8, 2, 19, 128
        scale = d**-0.5

        q_pre = mx.array(rng.standard_normal((b, n_q, s_pre, d)).astype(np.float32), dtype=mx.bfloat16)
        # Persistent channel outlier in K/V (models RoPE / RMSNorm outlier channels in Qwen3.5)
        k_np = rng.standard_normal((b, n_kv, s_pre, d)).astype(np.float32)
        k_np[:, :, :, 0] *= 20.0
        k_pre = mx.array(k_np, dtype=mx.bfloat16)
        v_np = rng.standard_normal((b, n_kv, s_pre, d)).astype(np.float32)
        v_np[:, :, :, 3] *= 20.0
        v_pre = mx.array(v_np, dtype=mx.bfloat16)

        q_dec = mx.array(rng.standard_normal((b, n_q, 1, d)).astype(np.float32), dtype=mx.bfloat16)
        k_dec = mx.array(rng.standard_normal((b, n_kv, 1, d)).astype(np.float32), dtype=mx.bfloat16)
        v_dec = mx.array(rng.standard_normal((b, n_kv, 1, d)).astype(np.float32), dtype=mx.bfloat16)

        # Reference BF16 SDPA
        k_all = mx.concatenate([k_pre, k_dec], axis=2)
        v_all = mx.concatenate([v_pre, v_dec], axis=2)
        ref_out = mx.fast.scaled_dot_product_attention(q_dec, k_all, v_all, scale=scale, mask=None)

        # 1. q8 QuantizedKVCache (step=16 forces strided growth at 19 + 1 = 20 < alloc=32!)
        c_q8 = QuantizedKVCache(step=16, bits=8, group_size=64)
        _ = c_q8.attend(q_pre, k_pre, v_pre, scale=scale, mask="causal")
        out_q8 = c_q8.attend(q_dec, k_dec, v_dec, scale=scale, mask="causal")
        mx.eval(ref_out, out_q8)
        ref_f = np.asarray(ref_out.astype(mx.float32)).reshape(-1)
        q8_f = np.asarray(out_q8.astype(mx.float32)).reshape(-1)
        cos_q8 = float(np.dot(ref_f, q8_f) / (np.linalg.norm(ref_f) * np.linalg.norm(q8_f)))
        assert cos_q8 > 0.9985

        # Raw 4-bit without QuaRot suffers quantization distortion on the 18-sigma channel outlier
        c_q4_raw = QuantizedKVCache(step=16, bits=4, group_size=64, use_quarot=False)
        _ = c_q4_raw.attend(q_pre, k_pre, v_pre, scale=scale, mask="causal")
        out_q4_raw = c_q4_raw.attend(q_dec, k_dec, v_dec, scale=scale, mask="causal")

        # 2. q4_quarot QuantizedKVCache with online (Q, K, V) FWHT-64 + inverse before gate
        bank = HadamardTransformBank(head_dim=d, use_metal=False)
        c_q4 = QuantizedKVCache(step=16, bits=4, group_size=64, use_quarot=True)
        _ = c_q4.attend(
            bank.transform(q_pre),
            bank.transform(k_pre),
            bank.transform(v_pre),
            scale=scale,
            mask="causal",
        )
        out_q4_rot = c_q4.attend(
            bank.transform(q_dec),
            bank.transform(k_dec),
            bank.transform(v_dec),
            scale=scale,
            mask="causal",
        )
        out_q4 = bank.transform(out_q4_rot, inverse=True)
        mx.eval(out_q4_raw, out_q4)
        raw_f = np.asarray(out_q4_raw.astype(mx.float32)).reshape(-1)
        q4_f = np.asarray(out_q4.astype(mx.float32)).reshape(-1)
        cos_raw = float(np.dot(ref_f, raw_f) / (np.linalg.norm(ref_f) * np.linalg.norm(raw_f)))
        cos_q4 = float(np.dot(ref_f, q4_f) / (np.linalg.norm(ref_f) * np.linalg.norm(q4_f)))
        err_raw = float(np.linalg.norm(ref_f - raw_f) / np.linalg.norm(ref_f))
        err_q4 = float(np.linalg.norm(ref_f - q4_f) / np.linalg.norm(ref_f))
        assert cos_q4 > 0.985
        assert err_q4 < err_raw
        assert c_q4.active_nbytes() < c_q8.active_nbytes()

        # Test slice_quantized + append_cached_quantized + rewind_to
        sl = c_q4.slice_quantized(0, 16)
        c_restored = QuantizedKVCache(step=32, bits=4, group_size=64, use_quarot=True)
        c_restored.append_cached_quantized(*sl)
        assert c_restored.offset == 16
        c_restored.rewind_to(8)
        assert c_restored.offset == 8
    finally:
        mx.set_default_device(prev_dev)
