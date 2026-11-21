"""CPU unit tests for Post-Norm Online Hadamard W8A8 ANE FFN, calibration, breaker, and fallback."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from fast_shared.ane_ffn import (
    ANE_FFN_INPUT_BLOCK,
    ANE_FFN_INTERMEDIATE_BLOCK,
    ANE_FFN_PEAK_FLOOR,
    BLOB_DTYPE_FP16,
    BLOB_DTYPE_INT8,
    BLOB_MAGIC,
    AneFfnBreaker,
    AneFfnCalibration,
    AneFfnStartup,
    AneFfnState,
    AneHandoffLibrary,
    AneWave,
    CalibrationSample,
    build_ffn_geglu_segment_mil,
    build_linear_conv_mil,
    compute_error_metrics,
    eval_ffn_fp32_ref,
    eval_ffn_w8a8,
    gelu_pytorch_tanh,
    hadamard_matrix,
    hadamard_rotate_blocks,
    make_hadamard_signs,
    pack_weight_blobs,
    prepare_ffn_w8a8_weights,
    quantize_symmetric_int8,
    segment_intermediate_sizes,
)


def test_hadamard_orthogonality_and_inner_product_preservation():
    for n in (128, 512):
        h = hadamard_matrix(n)
        gram = h @ h.T
        np.testing.assert_allclose(gram, np.eye(n, dtype=np.float32) * n, atol=1e-4, rtol=1e-4)

    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(8, 1536)).astype(np.float32)
    w = rng.normal(0.0, 1.0, size=(16, 1536)).astype(np.float32)
    signs = make_hadamard_signs(1536)

    x_rot = hadamard_rotate_blocks(x, ANE_FFN_INPUT_BLOCK, signs)
    w_rot = hadamard_rotate_blocks(w, ANE_FFN_INPUT_BLOCK, signs)
    np.testing.assert_allclose(x_rot @ w_rot.T, x @ w.T, atol=1e-3, rtol=1e-4)

    with pytest.raises(ValueError, match="divisible"):
        hadamard_rotate_blocks(np.zeros((2, 130), dtype=np.float32), 128)


def test_post_norm_hadamard_handles_extreme_norm_outliers_without_collapse():
    rng = np.random.default_rng(99)
    M, H, D = 32, 256, 512
    # Simulate extreme Gemma-4 pre_feedforward_layernorm weight scaling where certain
    # channels carry large weights (the exact regime that collapses SpinQuant norm folding).
    w_norm = rng.uniform(0.5, 2.0, size=(H,)).astype(np.float32)
    w_norm[0] = 60.0
    w_norm[129] = 45.0

    h_hidden = rng.normal(0.0, 1.0, size=(M, H)).astype(np.float32)
    h_hidden = h_hidden / np.sqrt(np.mean(h_hidden**2, axis=-1, keepdims=True) + 1e-6)
    x_post_norm = h_hidden * w_norm[None, :]

    w_gate = rng.normal(0.0, 0.05, size=(D, H)).astype(np.float32)
    w_up = rng.normal(0.0, 0.05, size=(D, H)).astype(np.float32)
    w_down = rng.normal(0.0, 0.05, size=(H, D)).astype(np.float32)

    ref = eval_ffn_fp32_ref(x_post_norm, w_gate, w_up, w_down, activation="gelu_tanh")
    w8_none = prepare_ffn_w8a8_weights(w_gate, w_up, w_down, hadamard_mode="none")
    w8_full = prepare_ffn_w8a8_weights(w_gate, w_up, w_down, hadamard_mode="full")

    y_none = eval_ffn_w8a8(x_post_norm, w8_none, activation="gelu_tanh")
    y_full = eval_ffn_w8a8(x_post_norm, w8_full, activation="gelu_tanh")

    m_none = compute_error_metrics(ref, y_none)
    m_full = compute_error_metrics(ref, y_full)
    assert m_full["cosine"] > 0.999
    assert m_full["cosine"] > m_none["cosine"]
    assert m_full["rel_rms_err"] < m_none["rel_rms_err"]


def test_gelu_fp16_safe_clamp_prevents_cubic_overflow():
    vals = np.array([-100.0, -50.0, -2.0, 0.0, 2.0, 50.0, 100.0], dtype=np.float32)
    exact = gelu_pytorch_tanh(vals, fp16_safe_clamp=False)
    clamped = gelu_pytorch_tanh(vals, fp16_safe_clamp=True)
    assert np.all(np.isfinite(clamped))
    np.testing.assert_allclose(clamped, exact, atol=1e-5, rtol=1e-5)

    # Verify intermediate cubic term under float16 with clamp remains finite
    xc = np.clip(vals.astype(np.float16), -10.0, 10.0)
    cubic = (xc * xc * xc).astype(np.float16)
    assert np.all(np.isfinite(cubic))


def test_quantize_zero_row_uses_peak_floor():
    x = np.zeros((3, ANE_FFN_INTERMEDIATE_BLOCK), dtype=np.float32)
    q, s = quantize_symmetric_int8(x)
    assert np.all(q == 0)
    np.testing.assert_allclose(s, ANE_FFN_PEAK_FLOOR / 127.0, rtol=1e-6)


def test_weight_blob_packing_and_segment_partition():
    assert segment_intermediate_sizes(6144) == [2560, 2560, 1024]
    assert segment_intermediate_sizes(12288) == [2560, 2560, 2560, 2560, 2048]

    a = np.ones((16, 8, 1, 1), dtype=np.float16)
    b = np.ones((16,), dtype=np.float16)
    c = np.full((16, 8, 1, 1), 3, dtype=np.int8)
    blob, offs = pack_weight_blobs([("a", a, BLOB_DTYPE_FP16), ("b", b, BLOB_DTYPE_FP16), ("c", c, BLOB_DTYPE_INT8)])
    assert len(blob) % 64 == 0
    for name in ("a", "b", "c"):
        off = offs[name]
        assert off % 64 == 0
        magic, dtype_code, raw_len, data_off = struct.unpack_from("<IIQQ", blob, off)
        assert magic == BLOB_MAGIC
        assert data_off == off + 64
        assert raw_len > 0

    mil_conv = build_linear_conv_mil(128, 64, 256, weight_offset=offs["c"], scale_offset=offs["b"], weight_dtype="int8_dequant")
    assert "constexpr_blockwise_shift_scale" in mil_conv
    mil_ffn = build_ffn_geglu_segment_mil(
        1536,
        2560,
        512,
        w_gate_off=64,
        w_up_off=128,
        w_down_off=192,
        weight_dtype="fp16",
        activation="gelu_tanh",
    )
    assert 'TANH_APPROXIMATION' in mil_ffn
    assert "conv(" in mil_ffn


def test_calibration_fit_choose_and_refit():
    # Suppose gpu_full_ms = 10.0, ANE(s) = 1.0 + 4.0*s, overlap u = 0.1
    # Optimal share sits near balance point where ANE(s) ~= (1-s)*10 + 0.1*ANE(s) -> s ~ 0.65
    samples = [
        CalibrationSample(share=s, ane_ms=1.0 + 4.0 * s, tail_gpu_ms=10.0 * (1.0 - s) + 0.1 * (1.0 + 4.0 * s), wall_ms=0.0)
        for s in (0.25, 0.50, 0.75, 1.0)
    ]
    cal = AneFfnCalibration.fit(samples, gpu_full_ms=10.0)
    assert cal.ane_fixed_ms == pytest.approx(1.0, abs=1e-2)
    assert cal.ane_slope_ms == pytest.approx(4.0, abs=1e-2)
    assert cal.overlap_u == pytest.approx(0.1, abs=1e-2)

    dec = cal.choose(6144, channel_unit=512, min_gain=0.05)
    assert dec.state == AneFfnState.SPLIT
    assert dec.ane_channels > 0
    assert dec.ane_channels % 512 == 0
    assert dec.predicted_gain > 0.45

    # Refit with high fixed/overlap ANE cost slower everywhere -> must transition to NoGain
    slow_samples = [
        CalibrationSample(share=s, ane_ms=10.5 + 8.0 * s, tail_gpu_ms=10.0 * (1.0 - s) + 0.5 * (10.5 + 8.0 * s), wall_ms=0.0)
        for s in (0.25, 0.5, 1.0)
    ]
    cal_slow = cal.refit(slow_samples)
    dec_slow = cal_slow.choose(6144, channel_unit=512, min_gain=0.05)
    assert dec_slow.state == AneFfnState.NO_GAIN
    assert dec_slow.ane_channels == 0
    assert dec_slow.gpu_channels == 6144


def test_breaker_trips_only_when_rolling_window_loses_gain():
    breaker = AneFfnBreaker(window=8, min_gain=0.05, chunk_noise=0.02)
    for _ in range(8):
        assert breaker.observe(gpu_full_ms=10.0, split_ms=9.0) is True
        assert not breaker.tripped

    # Fresh window: 7 regressing observations do not trip before full window; 8th trips cleanly.
    breaker_reg = AneFfnBreaker(window=8, min_gain=0.05, chunk_noise=0.02)
    for _ in range(7):
        assert breaker_reg.observe(gpu_full_ms=10.0, split_ms=10.5) is True
        assert not breaker_reg.tripped
    assert breaker_reg.observe(gpu_full_ms=10.0, split_ms=10.5) is False
    assert breaker_reg.tripped
    assert "realized_gain" in breaker_reg.reason


def test_startup_state_machine_all_outcomes():
    good_samples = [CalibrationSample(share=0.5, ane_ms=2.0, tail_gpu_ms=5.0, wall_ms=5.0)]
    bad_samples = [
        CalibrationSample(share=0.25, ane_ms=11.0, tail_gpu_ms=9.5, wall_ms=11.0),
        CalibrationSample(share=0.75, ane_ms=15.0, tail_gpu_ms=5.5, wall_ms=15.0),
    ]

    dec_off, _ = AneFfnStartup.decide(
        enabled=False, ane_available=True, tokens=512, hidden_size=1536, intermediate_size=6144,
        samples=good_samples, gpu_full_ms=10.0,
    )
    assert dec_off.state == AneFfnState.OFF

    dec_refused, _ = AneFfnStartup.decide(
        enabled=True, ane_available=True, tokens=512, hidden_size=1536, intermediate_size=6144,
        samples=good_samples, gpu_full_ms=10.0, refusal_count=2,
    )
    assert dec_refused.state == AneFfnState.REFUSED

    dec_unsup, _ = AneFfnStartup.decide(
        enabled=True, ane_available=True, tokens=64, hidden_size=1536, intermediate_size=6144,
        samples=good_samples, gpu_full_ms=10.0,
    )
    assert dec_unsup.state == AneFfnState.UNSUPPORTED

    dec_unavail, _ = AneFfnStartup.decide(
        enabled=True, ane_available=False, tokens=512, hidden_size=1536, intermediate_size=6144,
        samples=good_samples, gpu_full_ms=10.0,
    )
    assert dec_unavail.state == AneFfnState.UNAVAILABLE

    dec_nogain, _ = AneFfnStartup.decide(
        enabled=True, ane_available=True, tokens=512, hidden_size=1536, intermediate_size=6144,
        samples=bad_samples, gpu_full_ms=10.0,
    )
    assert dec_nogain.state == AneFfnState.NO_GAIN

    dec_split, _ = AneFfnStartup.decide(
        enabled=True, ane_available=True, tokens=512, hidden_size=1536, intermediate_size=6144,
        samples=good_samples, gpu_full_ms=10.0,
    )
    assert dec_split.state == AneFfnState.SPLIT
    assert dec_split.ane_channels % 512 == 0


def test_missing_native_dylib_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        AneHandoffLibrary("/nonexistent/path/libfast_ane_handoff.dylib")


def test_packed_token_scale_safe_gelu_normalize_mid_mil():
    mil = build_ffn_geglu_segment_mil(
        1536,
        2560,
        256,
        w_gate_off=64,
        w_up_off=128,
        w_down_off=192,
        w_gate_scale_off=256,
        w_up_scale_off=320,
        w_down_scale_off=384,
        weight_dtype="int8_dequant",
        input_dtype="fp16",
        activation="gelu_tanh",
        packed_token_scale=True,
        fp16_safe_clamp=True,
        normalize_mid=True,
    )
    assert "tensor<fp16, [1, 1537, 1, 256]> x_cat" in mil
    assert "slice_by_size(" in mil
    assert "clip(alpha = c_lo, beta = c_hi, x = g)" in mil
    assert "relu(x = g_diff)" in mil
    assert "real_div(x = mid, y = mscale)" in mil
    assert "constexpr_blockwise_shift_scale(" in mil

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_ffn_geglu_segment_mil(
            1536, 512, 128, w_gate_off=64, w_up_off=128, w_down_off=192, apply_token_scale=True, packed_token_scale=True
        )


def test_ane_wave_input_validation_without_hardware():
    class _StubCtx:
        _lib = None

    ctx = _StubCtx()
    with pytest.raises(ValueError, match="non-empty and equal length"):
        AneWave(
            ctx,  # type: ignore[arg-type]
            seg_progs=[],
            seg_out_surfs=[],
            surf_in=object(),  # type: ignore[arg-type]
            host_in_fp16=np.zeros((4, 128), dtype=np.float16),
            signs_fp16=np.ones((128,), dtype=np.float16),
        )
    with pytest.raises(ValueError, match="Expected 2D"):
        AneWave(
            ctx,  # type: ignore[arg-type]
            seg_progs=[object()],  # type: ignore[list-item]
            seg_out_surfs=[object()],  # type: ignore[list-item]
            surf_in=object(),  # type: ignore[arg-type]
            host_in_fp16=np.zeros((128,), dtype=np.float16),
            signs_fp16=np.ones((128,), dtype=np.float16),
        )
    with pytest.raises(ValueError, match="signs shape"):
        AneWave(
            ctx,  # type: ignore[arg-type]
            seg_progs=[object()],  # type: ignore[list-item]
            seg_out_surfs=[object()],  # type: ignore[list-item]
            surf_in=object(),  # type: ignore[arg-type]
            host_in_fp16=np.zeros((4, 128), dtype=np.float16),
            signs_fp16=np.ones((64,), dtype=np.float16),
        )
