"""CPU unit tests for Post-Norm Online Hadamard W8A8 ANE FFN, calibration, breaker, and fallback."""

from __future__ import annotations

import inspect
import struct

import numpy as np
import pytest

from fast_shared.ane_ffn import (
    ANE_FFN_DEFAULT_SEGMENT,
    ANE_FFN_INPUT_BLOCK,
    ANE_FFN_INTERMEDIATE_BLOCK,
    ANE_FFN_PEAK_FLOOR,
    BLOB_DTYPE_FP16,
    BLOB_DTYPE_INT8,
    BLOB_MAGIC,
    AneContext,
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


def test_ane_wave_default_quantize_a8_is_false_w8a16():
    assert inspect.signature(AneContext.begin_ffn_wave).parameters["quantize_a8"].default is False
    assert inspect.signature(AneWave.__init__).parameters["quantize_a8"].default is False

    seen: dict[str, int] = {}

    class _FakeCLib:
        def fast_ane_ffn_wave_begin(
            self,
            _ctx,
            _progs,
            _outs,
            n_segs,
            _surf_in,
            _host_in,
            _signs,
            tokens,
            hidden,
            quantize_a8,
            use_shared_event,
            timeout_ms,
            out_ptr,
            _err,
            _err_len,
        ):
            seen["n_segs"] = int(n_segs)
            seen["tokens"] = int(tokens)
            seen["hidden"] = int(hidden)
            seen["quantize_a8"] = int(quantize_a8)
            seen["use_shared_event"] = int(use_shared_event)
            seen["timeout_ms"] = int(timeout_ms)
            out_ptr._obj.value = 0
            return 0

        def fast_ane_ffn_wave_destroy(self, _ptr):
            return None

    class _FakeLib:
        _lib = _FakeCLib()

    class _FakeCtx:
        _lib = _FakeLib()
        _ptr = 1

    class _FakeHandle:
        _ptr = 1

    wave = AneWave(
        _FakeCtx(),  # type: ignore[arg-type]
        seg_progs=[_FakeHandle()],  # type: ignore[list-item]
        seg_out_surfs=[_FakeHandle()],  # type: ignore[list-item]
        surf_in=_FakeHandle(),  # type: ignore[arg-type]
        host_in_fp16=np.zeros((8, 2560), dtype=np.float16),
        signs_fp16=np.ones((2560,), dtype=np.float16),
    )
    wave.close()
    assert seen == {
        "n_segs": 1,
        "tokens": 8,
        "hidden": 2560,
        "quantize_a8": 0,
        "use_shared_event": 1,
        "timeout_ms": 5000,
    }


def test_e4b_h_gt_2048_segmentation_and_mil_generation():
    H, d_ff, M = 2560, 10240, 128
    segs = segment_intermediate_sizes(d_ff, ANE_FFN_DEFAULT_SEGMENT)
    assert segs == [2560, 2560, 2560, 2560]

    rng = np.random.default_rng(2026)
    w_norm = rng.uniform(0.5, 3.0, size=(H,)).astype(np.float32)
    w_norm[2150] = 110.0  # Outlier past 2048 channel threshold
    x_raw = rng.normal(0.0, 1.0, size=(M, H)).astype(np.float32)
    x = (x_raw / np.sqrt(np.mean(x_raw**2, axis=-1, keepdims=True) + 1e-6)) * w_norm[None, :]
    wg = rng.normal(0.0, 0.02, size=(d_ff, H)).astype(np.float32)
    wu = rng.normal(0.0, 0.02, size=(d_ff, H)).astype(np.float32)
    wd = rng.normal(0.0, 0.02, size=(H, d_ff)).astype(np.float32)

    ref = eval_ffn_fp32_ref(x, wg, wu, wd, activation="gelu_tanh")
    w8 = prepare_ffn_w8a8_weights(wg, wu, wd, hadamard_mode="full")
    y = eval_ffn_w8a8(x, w8, activation="gelu_tanh")
    m = compute_error_metrics(ref, y)
    assert m["cosine"] > 0.999

    mil = build_ffn_geglu_segment_mil(
        H,
        segs[0],
        M,
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
    assert f"tensor<fp16, [1, {H + 1}, 1, {M}]> x_cat" in mil
    assert f"val = tensor<int32, [4]>([1, {H}, 1, {M}])" in mil
    assert "constexpr_blockwise_shift_scale(" in mil


def test_build_ane_models_post_norm_hadamard_mil_and_options():
    import importlib.util
    import sys
    from pathlib import Path

    pytest.importorskip("coremltools")
    build_script = Path(__file__).resolve().parents[2] / "fast-gemma4" / "scripts" / "build_ane_models.py"
    if not build_script.is_file():
        pytest.skip(f"build_ane_models.py not present at {build_script}")

    spec = importlib.util.spec_from_file_location("build_ane_models_track_e_test", build_script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_ane_models_track_e_test"] = mod
    spec.loader.exec_module(mod)

    # 1. Option resolution and validation
    with pytest.raises(ValueError, match="hadamard_mode"):
        mod.resolve_variant_mil_options("w8", hadamard_mode="bad")
    with pytest.raises(ValueError, match="hadamard_op"):
        mod.resolve_variant_mil_options("w8", hadamard_op="matmul")
    auto_w8 = mod.resolve_variant_mil_options("w8")
    assert auto_w8["hadamard_mode"] == "post_norm"
    assert auto_w8["hadamard_op"] == "grouped_conv"
    assert auto_w8["safe_gelu_clamp"] is False
    assert auto_w8["normalize_mid"] is False
    auto_fp16 = mod.resolve_variant_mil_options("fp16")
    assert auto_fp16["hadamard_mode"] == "none"
    assert auto_fp16["safe_gelu_clamp"] is False
    assert auto_fp16["normalize_mid"] is False
    opts = mod.resolve_variant_mil_options(
        "w8", hadamard_mode="post_norm", hadamard_op="grouped_conv", safe_gelu_clamp="true", normalize_mid="true"
    )
    assert opts["hadamard_mode"] == "post_norm"
    assert opts["hadamard_op"] == "grouped_conv"
    assert opts["safe_gelu_clamp"] is True
    assert opts["normalize_mid"] is True

    # 2. Synthetic EarlyExit NPZ with E4B-width H=2560 (n_kv=2, k=4: 2 SWA + 2 Full)
    rng = np.random.default_rng(314)
    k, H, d_ff, ple_dim, n_q, n_kv = 4, 2560, 512, 64, 8, 2
    npz: dict[str, np.ndarray] = {
        "per_layer_model_proj_w": (rng.normal(0.0, 0.02, size=((k - 1) * ple_dim, H, 1, 1))).astype(np.float16),
        "per_layer_proj_scale": np.array([1.0], dtype=np.float16),
        "per_layer_proj_norm_w": np.ones((ple_dim,), dtype=np.float16),
        "per_layer_input_scale": np.array([1.0], dtype=np.float16),
        "n_q_heads": np.array(n_q, dtype=np.int32),
    }
    for i in range(k):
        hd = 512 if i in (1, k - 1) else 256
        qdim = n_q * hd
        kv_dim = n_kv * hd
        p = f"l{i}_"
        npz[p + "w_in"] = rng.uniform(0.5, 2.0, size=(1, H, 1, 1)).astype(np.float16)
        npz[p + "w_kn"] = np.ones((1, 1, 1, hd), dtype=np.float16)
        if i == k - 1:
            npz[p + "kv"] = rng.normal(0.0, 0.02, size=(2 * kv_dim, H, 1, 1)).astype(np.float16)
        else:
            npz[p + "qkv"] = rng.normal(0.0, 0.02, size=(qdim + 2 * kv_dim, H, 1, 1)).astype(np.float16)
            npz[p + "w_qn"] = np.ones((1, 1, 1, hd), dtype=np.float16)
            npz[p + "wo"] = rng.normal(0.0, 0.02, size=(H, qdim, 1, 1)).astype(np.float16)
            npz[p + "w_pa"] = np.ones((1, H, 1, 1), dtype=np.float16)
            npz[p + "w_pff"] = rng.uniform(0.5, 2.0, size=(1, H, 1, 1)).astype(np.float16)
            npz[p + "gu"] = rng.normal(0.0, 0.02, size=(2 * d_ff, H, 1, 1)).astype(np.float16)
            npz[p + "dn"] = rng.normal(0.0, 0.02, size=(H, d_ff, 1, 1)).astype(np.float16)
            npz[p + "w_poff"] = np.ones((1, H, 1, 1), dtype=np.float16)
            npz[p + "w_pg"] = rng.normal(0.0, 0.02, size=(ple_dim, H, 1, 1)).astype(np.float16)
            npz[p + "w_pp"] = rng.normal(0.0, 0.02, size=(H, ple_dim, 1, 1)).astype(np.float16)
            npz[p + "w_pple"] = np.ones((1, H, 1, 1), dtype=np.float16)
            npz[p + "scalar"] = np.array([1.0], dtype=np.float16)

    # Check exact WHT inner-product preservation on transformed weights
    arch = mod.infer_ane_arch_from_npz(npz)
    npz_rot, signs_map = mod.apply_post_norm_hadamard_transforms(npz, arch, hadamard_mode="post_norm")
    assert set(signs_map) >= {"l0_in", "l0_pff", "l0_mid", "l3_in"}
    x_norm = rng.normal(0.0, 1.0, size=(4, H)).astype(np.float32)
    y_orig = x_norm @ npz["l0_qkv"].astype(np.float32)[:, :, 0, 0].T
    x_had = hadamard_rotate_blocks(x_norm * signs_map["l0_in"][None, :], ANE_FFN_INPUT_BLOCK)
    y_rot = x_had @ npz_rot["l0_qkv"].astype(np.float32)[:, :, 0, 0].T
    np.testing.assert_allclose(y_rot, y_orig, atol=2e-2, rtol=2e-2)

    # Build both grouped_conv and shared_conv MIL programs cleanly
    for op_mode in ("grouped_conv", "shared_conv"):
        prog, got_arch = mod.build_early_prefix_mil_program(
            npz,
            S=32,
            P_SWA=64,
            P_FULL=64,
            hadamard_mode="post_norm",
            hadamard_op=op_mode,
            safe_gelu_clamp=True,
            normalize_mid=True,
        )
        assert got_arch["hidden_size"] == H
        mil_txt = str(prog)
        assert "had_in" in mil_txt
        assert "had_pff" in mil_txt
        assert "had_mid" in mil_txt
        assert "gclip" in mil_txt
        assert "mscale" in mil_txt
