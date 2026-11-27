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

    # Default binary search rerank width is 256 (higher recall with negligible cost) and executes cleanly
    import inspect
    assert inspect.signature(CompactRetrievalIndex.search).parameters["rerank_top_r"].default == 256
    top_bin_default, sc_bin_default = idx_bin_neon.search(queries, k=5)
    assert np.array_equal(top_bin_default[:, 0], np.arange(8, dtype=np.int32))
    assert sc_bin_default.shape == (8, 5)

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


def test_compact_retrieval_index_from_prequantized_zero_copy_and_bitwise_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(314)
    docs = rng.standard_normal((128, 96)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries = docs[:4].copy() + 0.03 * rng.standard_normal((4, 96)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # 1. binary_int8 zero-copy parity vs regular CompactRetrievalIndex (NEON + NumPy 2 bitwise_count + LUT fallback)
    ref_bin = CompactRetrievalIndex(docs, mode="binary_int8", use_neon=True)
    assert ref_bin.codes_i8 is not None and ref_bin.scales_f16 is not None and ref_bin.binary_u8 is not None

    pq_bin = CompactRetrievalIndex.from_prequantized(
        mode="binary_int8",
        dim=96,
        codes_i8=ref_bin.codes_i8,
        scales_f16=ref_bin.scales_f16,
        binary_u8=ref_bin.binary_u8,
        use_neon=True,
    )
    assert pq_bin.binary_bits == 96
    assert pq_bin.codes_i8 is ref_bin.codes_i8
    assert pq_bin.scales_f16 is ref_bin.scales_f16
    assert pq_bin.binary_u8 is ref_bin.binary_u8
    assert np.shares_memory(pq_bin.codes_i8, ref_bin.codes_i8)
    assert np.shares_memory(pq_bin.scales_f16, ref_bin.scales_f16)
    assert np.shares_memory(pq_bin.binary_u8, ref_bin.binary_u8)

    idx_ref, sc_ref = ref_bin.search(queries, k=5, rerank_top_r=64)
    idx_pq, sc_pq = pq_bin.search(queries, k=5, rerank_top_r=64)
    assert np.array_equal(idx_pq, idx_ref)
    np.testing.assert_allclose(sc_pq, sc_ref, atol=1e-6)

    # Verify non-NEON NumPy 2 bitwise_count and NumPy <2 _POPCOUNT_LUT fallback produce exact parity!
    pq_bin_np = CompactRetrievalIndex.from_prequantized(
        mode="binary_int8",
        dim=96,
        codes_i8=ref_bin.codes_i8,
        scales_f16=ref_bin.scales_f16,
        binary_u8=ref_bin.binary_u8,
        use_neon=False,
    )
    idx_bc, sc_bc = pq_bin_np.search(queries, k=5, rerank_top_r=64)
    assert np.array_equal(idx_bc, idx_ref)
    np.testing.assert_allclose(sc_bc, sc_ref, atol=1e-4)

    monkeypatch.delattr(np, "bitwise_count", raising=False)
    idx_lut, sc_lut = pq_bin_np.search(queries, k=5, rerank_top_r=64)
    assert np.array_equal(idx_lut, idx_bc)
    np.testing.assert_allclose(sc_lut, sc_bc, atol=1e-6)

    # 2. int8 zero-copy parity
    ref_i8 = CompactRetrievalIndex(docs, mode="int8", use_neon=True)
    assert ref_i8.codes_i8 is not None and ref_i8.scales_f16 is not None
    pq_i8 = CompactRetrievalIndex.from_prequantized(
        mode="int8",
        dim=96,
        codes_i8=ref_i8.codes_i8,
        scales_f16=ref_i8.scales_f16,
        use_neon=True,
    )
    assert pq_i8.codes_i8 is ref_i8.codes_i8
    assert pq_i8.scales_f16 is ref_i8.scales_f16
    i_r, s_r = ref_i8.search(queries, k=5)
    i_p, s_p = pq_i8.search(queries, k=5)
    assert np.array_equal(i_p, i_r)
    np.testing.assert_allclose(s_p, s_r, atol=1e-6)

    # 3. fp16 & fp32 zero-copy parity
    for dense_mode in ("fp16", "fp32"):
        ref_d = CompactRetrievalIndex(docs, mode=dense_mode)
        assert ref_d.dense_vecs is not None
        pq_d = CompactRetrievalIndex.from_prequantized(
            mode=dense_mode,
            dim=96,
            dense_vecs=ref_d.dense_vecs,
        )
        assert pq_d.dense_vecs is ref_d.dense_vecs
        assert np.shares_memory(pq_d.dense_vecs, ref_d.dense_vecs)
        di_r, ds_r = ref_d.search(queries, k=5)
        di_p, ds_p = pq_d.search(queries, k=5)
        assert np.array_equal(di_p, di_r)
        np.testing.assert_allclose(ds_p, ds_r, atol=1e-6)

    # 4. Empty N=0 prequantized index and validation errors
    empty_idx = CompactRetrievalIndex.from_prequantized(
        mode="binary_int8",
        dim=64,
        codes_i8=np.empty((0, 64), dtype=np.int8),
        scales_f16=np.empty((0,), dtype=np.float16),
        binary_u8=np.empty((0, 8), dtype=np.uint8),
    )
    assert empty_idx.num_docs == 0
    ei, es = empty_idx.search(np.ones(64, dtype=np.float32), k=3)
    assert np.all(ei == -1) and es.shape == (3,)

    with pytest.raises(ValueError, match="binary_bits"):
        CompactRetrievalIndex.from_prequantized(
            mode="binary_int8",
            dim=64,
            codes_i8=np.zeros((2, 64), dtype=np.int8),
            scales_f16=np.ones((2,), dtype=np.float16),
            binary_u8=np.zeros((2, 8), dtype=np.uint8),
            binary_bits=128,
        )
    with pytest.raises(ValueError, match="dtype"):
        CompactRetrievalIndex.from_prequantized(
            mode="int8",
            dim=64,
            codes_i8=np.zeros((2, 64), dtype=np.float32),
            scales_f16=np.ones((2,), dtype=np.float16),
        )
    with pytest.raises(ValueError, match="shape"):
        CompactRetrievalIndex.from_prequantized(
            mode="fp16",
            dim=64,
            dense_vecs=np.zeros((2, 32), dtype=np.float16),
        )


def test_binary_topr_rerank_dim70_chunked_non_neon_fallback_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_shared.retrieval_index as ri

    assert ri._BINARY_POPCOUNT_CHUNK_ROWS == 32_768

    dim = 70  # Non-multiple of 8 (n_bytes=9, trailing 2 bits zero-padded)
    n_docs = 128
    rng = np.random.default_rng(7070)
    docs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    docs[0, 0] = 0.0
    docs[0, -1] = 0.0
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    docs[-1] = 0.0  # Explicit exact-zero row

    queries = docs[:4].copy() + 0.03 * rng.standard_normal((4, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    idx_neon = CompactRetrievalIndex(docs, mode="binary_int8", use_neon=True)
    assert idx_neon.binary_u8 is not None and idx_neon.binary_u8.shape == (n_docs, 9)
    assert np.all(idx_neon.binary_u8[-1] == 0)

    top_neon, sc_neon = idx_neon.search(queries, k=5, rerank_top_r=48)
    assert np.array_equal(top_neon[:, 0], np.arange(4, dtype=np.int32))

    # Exercise multi-chunk non-NEON path (37-row chunks over 128 docs -> 4 chunks with partial tail).
    monkeypatch.setattr(ri, "_BINARY_POPCOUNT_CHUNK_ROWS", 37)
    idx_np = CompactRetrievalIndex.from_prequantized(
        mode="binary_int8",
        dim=dim,
        codes_i8=idx_neon.codes_i8,
        scales_f16=idx_neon.scales_f16,
        binary_u8=idx_neon.binary_u8,
        use_neon=False,
    )
    top_bc, sc_bc = idx_np.search(queries, k=5, rerank_top_r=48)
    assert np.array_equal(top_bc, top_neon)
    np.testing.assert_allclose(sc_bc, sc_neon, atol=1e-4)

    # Also verify _POPCOUNT_LUT chunked fallback produces identical results.
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    top_lut, sc_lut = idx_np.search(queries, k=5, rerank_top_r=48)
    assert np.array_equal(top_lut, top_bc)
    np.testing.assert_allclose(sc_lut, sc_bc, atol=1e-6)


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


def test_cpu_isa_detection_and_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_shared import (
        VALID_CPU_BACKEND_MODES,
        cpu_compile_flags,
        detect_cpu_isa,
        resolve_cpu_backend,
    )

    assert VALID_CPU_BACKEND_MODES == frozenset(
        {"auto", "neon_i8mm", "neon", "avx512", "avx2", "portable_cpp", "numpy"}
    )
    isa = detect_cpu_isa()
    assert int(isa["cpu_count"]) >= 1
    assert int(isa["p_cores"]) >= 1
    assert isa["recommended_backend"] in VALID_CPU_BACKEND_MODES

    monkeypatch.delenv("FAST_SILICON_CPU_BACKEND", raising=False)
    monkeypatch.delenv("FAST_SILICON_DISABLE_NEON_INDEX", raising=False)

    # Linux x86_64 AVX-512 VNNI + VPOPCNTDQ
    avx512_isa = {
        "system": "Linux",
        "machine": "x86_64",
        "has_cxx": True,
        "arm_dotprod": False,
        "arm_i8mm": False,
        "x86_avx2": True,
        "x86_popcnt": True,
        "x86_avx512_vnni": True,
        "x86_avx512_vpopcntdq": True,
        "recommended_backend": "avx512",
    }
    assert resolve_cpu_backend("auto", isa_info=avx512_isa) == "avx512"
    assert resolve_cpu_backend("avx2", isa_info=avx512_isa) == "avx2"
    assert resolve_cpu_backend("portable_cpp", isa_info=avx512_isa) == "portable_cpp"

    # Linux x86_64 AVX2-only falls back to avx2 even if avx512 requested
    avx2_isa = {**avx512_isa, "x86_avx512_vnni": False, "recommended_backend": "avx2"}
    assert resolve_cpu_backend("auto", isa_info=avx2_isa) == "avx2"
    assert resolve_cpu_backend("avx512", isa_info=avx2_isa) == "avx2"

    # ARM64 DotProd+I8MM keeps auto on neon (vdotq_s32) and exposes neon_i8mm opt-in
    arm_isa = {
        **avx512_isa,
        "system": "Linux",
        "machine": "aarch64",
        "arm_dotprod": True,
        "arm_i8mm": True,
        "x86_avx2": False,
        "x86_popcnt": False,
        "x86_avx512_vnni": False,
        "x86_avx512_vpopcntdq": False,
        "recommended_backend": "neon",
    }
    assert resolve_cpu_backend("auto", isa_info=arm_isa) == "neon"
    assert resolve_cpu_backend("neon_i8mm", isa_info=arm_isa) == "neon_i8mm"

    # Generic non-SIMD CPU falls back cleanly to portable_cpp; no compiler falls back to numpy
    generic_isa = {**arm_isa, "machine": "riscv64", "arm_dotprod": False, "arm_i8mm": False, "recommended_backend": "portable_cpp"}
    assert resolve_cpu_backend("auto", isa_info=generic_isa) == "portable_cpp"
    assert resolve_cpu_backend("auto", isa_info={**generic_isa, "has_cxx": False}) == "numpy"

    # Overrides & validation
    assert resolve_cpu_backend("auto", use_neon=False, isa_info=avx512_isa) == "numpy"
    monkeypatch.setenv("FAST_SILICON_CPU_BACKEND", "portable_cpp")
    assert resolve_cpu_backend("auto", isa_info=avx512_isa) == "portable_cpp"
    monkeypatch.setenv("FAST_SILICON_DISABLE_NEON_INDEX", "1")
    assert resolve_cpu_backend("auto", isa_info=avx512_isa) == "numpy"
    with pytest.raises(ValueError, match="cpu_backend"):
        resolve_cpu_backend("invalid_mode")  # type: ignore[arg-type]

    # Cross-ISA compiler flag checks
    f512 = cpu_compile_flags("avx512", system="Linux", arch="x86_64", shared=True)
    assert "-mavx512vnni" in f512 and "-mavx512vpopcntdq" in f512 and "-pthread" in f512
    f2 = cpu_compile_flags("avx2", system="Linux", arch="x86_64", shared=True)
    assert "-mavx2" in f2 and "-mpopcnt" in f2
    fneon = cpu_compile_flags("neon", system="Linux", arch="aarch64", shared=True)
    assert "-march=armv8.2-a+dotprod" in fneon and "-DFS_ENABLE_I8MM=0" in fneon
    fi8 = cpu_compile_flags("neon_i8mm", system="Darwin", arch="arm64", shared=False)
    assert "-arch" in fi8 and "arm64" in fi8 and "-DFS_ENABLE_I8MM=1" in fi8
    fport = cpu_compile_flags("portable_cpp", system="Linux", arch="riscv64", shared=False)
    assert "-O3" in fport and "-std=c++17" in fport


def test_compact_retrieval_cross_backend_parity_and_edges() -> None:
    from fast_shared import detect_cpu_isa

    isa = detect_cpu_isa()
    if not isa.get("has_cxx", False):
        pytest.skip("C++17 compiler required for C++ backend parity test")

    backends = ["portable_cpp"]
    if isa.get("arm_dotprod"):
        backends.append("neon")
    if isa.get("arm_i8mm"):
        backends.append("neon_i8mm")

    rng = np.random.default_rng(919)

    def _lex_topk(scores_2d: np.ndarray, k_val: int) -> tuple[np.ndarray, np.ndarray]:
        qn, nd = scores_2d.shape
        oi = np.full((qn, k_val), -1, dtype=np.int32)
        os_ = np.full((qn, k_val), -1e30, dtype=np.float32)
        if qn == 0 or nd == 0:
            return oi, os_
        take = min(k_val, nd)
        dids = np.arange(nd, dtype=np.int32)
        for qi in range(qn):
            row = scores_2d[qi]
            ord_i = np.lexsort((dids, -row))[:take]
            oi[qi, :take] = ord_i
            os_[qi, :take] = row[ord_i]
        return oi, os_

    # 1. Flat INT8 + Binary+INT8 parity across prime & unaligned dims, zero vectors, ties, and Q/N edges
    for dim in (1, 15, 17, 63, 64, 65, 384):
        for n_docs in (0, 1, 7, 257):
            for q_n in (0, 1, 5):
                docs = rng.standard_normal((n_docs, dim), dtype=np.float32) if n_docs else np.empty((0, dim), dtype=np.float32)
                if n_docs >= 4:
                    docs[2] = docs[0].copy()
                    docs[3] = 0.0
                qs = rng.standard_normal((q_n, dim), dtype=np.float32) if q_n else np.empty((0, dim), dtype=np.float32)
                if q_n >= 2:
                    qs[1] = 0.0

                idx_np_i8 = CompactRetrievalIndex(docs, mode="int8", cpu_backend="numpy")
                ref_i, ref_s = _lex_topk(idx_np_i8.score_all(qs), 10)
                idx_np_bin = CompactRetrievalIndex(docs, mode="binary_int8", cpu_backend="numpy")
                ref_bi, ref_bs = idx_np_bin.search(qs, k=10, rerank_top_r=64)

                for be in backends:
                    c_i8 = CompactRetrievalIndex.from_prequantized(
                        mode="int8",
                        dim=dim,
                        codes_i8=idx_np_i8.codes_i8,
                        scales_f16=idx_np_i8.scales_f16,
                        cpu_backend=be,  # type: ignore[arg-type]
                        max_threads=2,
                    )
                    assert c_i8.resolved_cpu_backend == be
                    got_i, got_s = c_i8.search(qs, k=10)
                    assert np.array_equal(got_i, ref_i)
                    valid = got_i >= 0
                    if np.any(valid):
                        np.testing.assert_allclose(got_s[valid], ref_s[valid], atol=2e-5)

                    c_bin = CompactRetrievalIndex.from_prequantized(
                        mode="binary_int8",
                        dim=dim,
                        codes_i8=idx_np_bin.codes_i8,
                        scales_f16=idx_np_bin.scales_f16,
                        binary_u8=idx_np_bin.binary_u8,
                        cpu_backend=be,  # type: ignore[arg-type]
                        max_threads=2,
                    )
                    got_bi, got_bs = c_bin.search(qs, k=10, rerank_top_r=64)
                    if dim > 1 or n_docs <= 10:
                        assert np.array_equal(got_bi, ref_bi)
                        valid_b = got_bi >= 0
                        if np.any(valid_b):
                            np.testing.assert_allclose(got_bs[valid_b], ref_bs[valid_b], atol=2e-5)
                    elif np.any(got_bi >= 0):
                        # dim=1 quantized int8 vectors take only {-1, 0, +1}; scores still match exactly
                        np.testing.assert_allclose(got_bs, ref_bs, atol=2e-5)

    # 2. Multi-vector skewed doc_offsets with empty middle docs across all 4 aggregations
    offs = [0, 3, 3, 4, 21, 21, 25]
    mv_vecs = rng.standard_normal((25, 97), dtype=np.float32)
    mv_qs = rng.standard_normal((6, 97), dtype=np.float32)
    for agg in ("maxsim", "hybrid", "top2_mean", "mean"):
        np_mv_i8 = CompactRetrievalIndex(mv_vecs, doc_offsets=offs, mode="int8", agg=agg, cpu_backend="numpy")
        ref_mi, ref_ms = _lex_topk(np_mv_i8.score_all(mv_qs, agg=agg), 4)
        np_mv_bin = CompactRetrievalIndex(mv_vecs, doc_offsets=offs, mode="binary_int8", agg=agg, cpu_backend="numpy")
        ref_mbi, ref_mbs = np_mv_bin.search(mv_qs, k=4, rerank_top_r=16, agg=agg)

        for be in backends:
            c_mv = CompactRetrievalIndex(
                mv_vecs,
                doc_offsets=offs,
                mode="int8",
                agg=agg,
                cpu_backend=be,  # type: ignore[arg-type]
                max_threads=2,
            )
            gi, gs = c_mv.search(mv_qs, k=4)
            assert np.array_equal(gi, ref_mi)
            np.testing.assert_allclose(gs, ref_ms, atol=2e-5)

            c_mvb = CompactRetrievalIndex(
                mv_vecs,
                doc_offsets=offs,
                mode="binary_int8",
                agg=agg,
                cpu_backend=be,  # type: ignore[arg-type]
                max_threads=2,
            )
            gbi, gbs = c_mvb.search(mv_qs, k=4, rerank_top_r=16)
            assert np.array_equal(gbi, ref_mbi)
            np.testing.assert_allclose(gbs, ref_mbs, atol=2e-5)

    # 3. All-empty multi-vector doc_offsets edge case
    for mode_s in ("int8", "binary_int8"):
        empty_mv = CompactRetrievalIndex(
            np.empty((0, 32), dtype=np.float32),
            doc_offsets=[0, 0, 0],
            mode=mode_s,  # type: ignore[arg-type]
            cpu_backend="portable_cpp",
        )
        ei, es = empty_mv.search(rng.standard_normal(32, dtype=np.float32), k=2)
        assert ei.shape == (2,) and es.shape == (2,)


def test_x86_64_avx2_rosetta_and_avx512_compile_parity(tmp_path: Path) -> None:
    import platform
    import shutil
    import subprocess
    from fast_shared.cpu_kernels import _RETRIEVAL_CPP_SOURCE, cpu_compile_flags

    if platform.system() != "Darwin":
        pytest.skip("Darwin clang++/otool x86_64 cross-ISA check")
    cxx = shutil.which("clang++")
    otool = shutil.which("otool")
    if not cxx or not otool:
        pytest.skip("clang++ or otool missing")

    harness = _RETRIEVAL_CPP_SOURCE + r"""
#include <cmath>
#include <cstdio>
int main() {
    const int Q = 5, N = 256, D = 99;
    std::vector<int8_t> q_i8(Q * D), d_i8(N * D);
    std::vector<uint16_t> q_sc(Q, 0x3C00), d_sc(N, 0x3C00);
    std::vector<uint8_t> q_bits(Q * 13), d_bits(N * 13);
    for (int i = 0; i < Q * D; ++i) q_i8[i] = static_cast<int8_t>(((i * 37) % 255) - 127);
    for (int i = 0; i < N * D; ++i) d_i8[i] = static_cast<int8_t>(((i * 53) % 255) - 127);
    for (int i = 0; i < Q * 13; ++i) q_bits[i] = static_cast<uint8_t>((i * 29) & 0xFF);
    for (int i = 0; i < N * 13; ++i) d_bits[i] = static_cast<uint8_t>((i * 41) & 0xFF);
    q_i8[0] = -128; q_i8[1] = -128; d_i8[0] = -128; d_i8[1] = 127;

    std::vector<float> scores(Q * N);
    fast_cpu_int8_scores(q_i8.data(), q_sc.data(), d_i8.data(), d_sc.data(), Q, N, D, 2, scores.data());
    for (int qi = 0; qi < Q; ++qi) {
        for (int di = 0; di < N; ++di) {
            int32_t ref = 0;
            for (int k = 0; k < D; ++k) ref += static_cast<int32_t>(q_i8[qi * D + k]) * static_cast<int32_t>(d_i8[di * D + k]);
            if (std::fabs(scores[qi * N + di] - static_cast<float>(ref)) > 1e-4f) return 2;
        }
    }
    std::vector<int32_t> bi(Q * 10);
    std::vector<float> bs(Q * 10);
    fast_cpu_binary_topr_rerank_int8(q_bits.data(), q_i8.data(), q_sc.data(), d_bits.data(), d_i8.data(), d_sc.data(), Q, N, 13, D, 64, 10, 2, bi.data(), bs.data());
    std::printf("X86_AVX2_PARITY_OK %d %.1f\n", bi[0], bs[0]);
    return 0;
}
"""
    src_p = tmp_path / "x86_cross.cpp"
    src_p.write_text(harness, encoding="utf-8")

    # 1. Verify AVX-512 VNNI + VPOPCNTDQ clean compile + instruction emission
    avx512_obj = tmp_path / "avx512.o"
    f512 = cpu_compile_flags("avx512", system="Darwin", arch="x86_64", shared=False)
    subprocess.run([cxx, "-c", "-Werror", *f512, str(src_p), "-o", str(avx512_obj)], check=True, capture_output=True, text=True)
    disasm = subprocess.check_output([otool, "-tV", str(avx512_obj)], text=True)
    assert "vpdpbusd" in disasm and "vpopcntq" in disasm

    # 2. Verify AVX2 clean compile & execute under Rosetta 2 if available
    avx2_bin = tmp_path / "x86_avx2"
    f2 = cpu_compile_flags("avx2", system="Darwin", arch="x86_64", shared=False)
    subprocess.run([cxx, "-Werror", *f2, str(src_p), "-o", str(avx2_bin)], check=True, capture_output=True, text=True)
    try:
        res = subprocess.run([str(avx2_bin)], check=True, capture_output=True, text=True, timeout=15)
    except OSError as exc:
        pytest.skip(f"Rosetta 2 unavailable to execute x86_64 AVX2 binary ({exc})")
    assert "X86_AVX2_PARITY_OK" in res.stdout
