"""Locked Apple Silicon hardware tests for private _ANEClient / MTLSharedEvent handoff and ANE FFN split."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pytest

from fast_shared.ane_ffn import (
    ANE_FFN_DEFAULT_SEGMENT,
    ANE_FFN_INPUT_BLOCK,
    AneFfnBreaker,
    AneFfnCalibration,
    AneFfnStartup,
    AneFfnState,
    AneHandoffLibrary,
    BLOB_DTYPE_FP16,
    BLOB_DTYPE_INT8,
    CalibrationSample,
    build_ffn_geglu_segment_mil,
    build_linear_conv_mil,
    compute_error_metrics,
    ensure_native_ane_built,
    eval_ffn_fp32_ref,
    hadamard_rotate_blocks,
    make_hadamard_signs,
    pack_weight_blobs,
    quantize_symmetric_int8,
    segment_intermediate_sizes,
)

pytestmark = pytest.mark.gpu_ane


@contextmanager
def _hw_lock():
    if os.environ.get("M4_GPU_LOCK_HELD") == "1" or os.environ.get("FAST_SILICON_GPU_LOCK_HELD") == "1":
        yield
        return
    lock_path = Path(os.environ.get("FAST_ANE_LOCK_PATH", "/tmp/m4_gpu_validation.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev_m4 = os.environ.get("M4_GPU_LOCK_HELD")
        prev_fs = os.environ.get("FAST_SILICON_GPU_LOCK_HELD")
        os.environ["M4_GPU_LOCK_HELD"] = "1"
        os.environ["FAST_SILICON_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if prev_m4 is None:
                os.environ.pop("M4_GPU_LOCK_HELD", None)
            else:
                os.environ["M4_GPU_LOCK_HELD"] = prev_m4
            if prev_fs is None:
                os.environ.pop("FAST_SILICON_GPU_LOCK_HELD", None)
            else:
                os.environ["FAST_SILICON_GPU_LOCK_HELD"] = prev_fs
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@pytest.fixture(scope="module")
def ane_lib() -> AneHandoffLibrary:
    dylib = ensure_native_ane_built()
    return AneHandoffLibrary(dylib)


def test_ane_compat_and_shared_event_gpu(ane_lib: AneHandoffLibrary):
    with _hw_lock():
        compat = ane_lib.probe_compat()
        assert compat["ane_available"] is True
        assert compat["framework_loaded"] is True
        assert compat["client_connected"] is True
        assert compat["shared_event_created"] is True
        assert compat["shared_event_has_event_port"] is True
        assert compat["iosurface_metal_spi"] is True
        assert compat["metal_runtime_compile_ok"] is True
        assert compat["all_selectors_present"] is True
        assert compat["all_encodings_exact"] is True
        sigs = compat["signatures"]
        assert len(sigs) >= 15
        assert all(s["present"] and s["exact"] for s in sigs)


def test_ane_handoff_roundtrip_shared_event_vs_cpu_sync_gpu(ane_lib: AneHandoffLibrary):
    rng = np.random.default_rng(17)
    C, S = 64, 128
    w = rng.normal(0.0, 0.2, size=(C, C)).astype(np.float32)
    wq, ws = quantize_symmetric_int8(w)
    blob, offs = pack_weight_blobs(
        [
            ("wq", wq.reshape(C, C, 1, 1), BLOB_DTYPE_INT8),
            ("ws", ws.astype(np.float16), BLOB_DTYPE_FP16),
        ]
    )
    mil = build_linear_conv_mil(
        C, C, S, weight_offset=offs["wq"], scale_offset=offs["ws"], weight_dtype="int8_dequant"
    )
    x_fp16 = rng.normal(0.0, 1.0, size=(C, S)).astype(np.float16)
    ref = (wq.astype(np.float32) * ws[:, None]) @ x_fp16.astype(np.float32)

    with _hw_lock(), ane_lib.create_context(allow_unavailable=True) as ctx:
        assert ctx.ane_available and ctx.shared_event_available
        with (
            ctx.compile_program(mil, weight_blob=blob, preformatted_blob=True) as prog,
            ctx.surface(S, C) as surf_in,
            ctx.surface(S, C) as surf_out,
        ):
            y_se, m_se = ctx.handoff_roundtrip(prog, surf_in, surf_out, x_fp16, use_shared_event=True)
            y_cs, m_cs = ctx.handoff_roundtrip(prog, surf_in, surf_out, x_fp16, use_shared_event=False)

    assert m_se["used_shared_event"] is True
    assert m_se["used_iosurface_metal_buffer"] is True
    assert m_cs["used_shared_event"] is False
    # `ane_smoke_pre` (x * 2 + 1) -> conv -> `ane_smoke_post` (+ 5)
    expected = (wq.astype(np.float32) * ws[:, None]) @ (x_fp16.astype(np.float32) * 2.0 + 1.0) + 5.0
    m1 = compute_error_metrics(expected, y_se.astype(np.float32))
    m2 = compute_error_metrics(expected, y_cs.astype(np.float32))
    assert m1["cosine"] > 0.9999
    assert m2["cosine"] > 0.9999
    np.testing.assert_allclose(y_se.astype(np.float32), y_cs.astype(np.float32), atol=1e-2, rtol=1e-2)
    assert ref.shape == expected.shape


def test_ane_ffn_wave_packed_unit_scale_accuracy_gpu(ane_lib: AneHandoffLibrary):
    rng = np.random.default_rng(42)
    M, H, D = 128, 1536, 3072
    w_norm = rng.uniform(0.5, 4.0, size=(H,)).astype(np.float32)
    w_norm[0] = 120.0
    w_norm[777] = 95.0
    h = rng.normal(0.0, 1.0, size=(M, H)).astype(np.float32)
    x_real = (h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-6)) * w_norm[None, :]
    x_fp16 = x_real.astype(np.float16)

    wg = rng.normal(0.0, 0.02, size=(D, H)).astype(np.float32)
    wu = rng.normal(0.0, 0.02, size=(D, H)).astype(np.float32)
    wd = rng.normal(0.0, 0.02, size=(H, D)).astype(np.float32)
    ref_fp32 = eval_ffn_fp32_ref(x_real, wg, wu, wd, activation="gelu_tanh")

    signs_in = make_hadamard_signs(H)
    signs_fp16 = signs_in.astype(np.float16)

    # Split: 2560 channels on ANE + 512 channels GPU tail
    d_ane = 2560
    qg, sg = quantize_symmetric_int8(hadamard_rotate_blocks(wg[:d_ane], ANE_FFN_INPUT_BLOCK, signs_in))
    qu, su = quantize_symmetric_int8(hadamard_rotate_blocks(wu[:d_ane], ANE_FFN_INPUT_BLOCK, signs_in))
    qd, sd = quantize_symmetric_int8(wd[:, :d_ane])
    tail_fp16 = eval_ffn_fp32_ref(
        x_fp16.astype(np.float32), wg[d_ane:], wu[d_ane:], wd[:, d_ane:], activation="gelu_tanh"
    ).astype(np.float16)

    seg_sizes = segment_intermediate_sizes(d_ane, ANE_FFN_DEFAULT_SEGMENT)
    assert seg_sizes == [2560]

    blob, offs = pack_weight_blobs(
        [
            ("wg_q", qg.reshape(d_ane, H, 1, 1), BLOB_DTYPE_INT8),
            ("wg_s", sg.astype(np.float16), BLOB_DTYPE_FP16),
            ("wu_q", qu.reshape(d_ane, H, 1, 1), BLOB_DTYPE_INT8),
            ("wu_s", su.astype(np.float16), BLOB_DTYPE_FP16),
            ("wd_q", qd.reshape(H, d_ane, 1, 1), BLOB_DTYPE_INT8),
            ("wd_s", sd.astype(np.float16), BLOB_DTYPE_FP16),
        ]
    )
    mil = build_ffn_geglu_segment_mil(
        H,
        d_ane,
        M,
        w_gate_off=offs["wg_q"],
        w_up_off=offs["wu_q"],
        w_down_off=offs["wd_q"],
        w_gate_scale_off=offs["wg_s"],
        w_up_scale_off=offs["wu_s"],
        w_down_scale_off=offs["wd_s"],
        weight_dtype="int8_dequant",
        input_dtype="fp16",
        activation="gelu_tanh",
        packed_token_scale=True,
        fp16_safe_clamp=True,
        normalize_mid=True,
    )

    with _hw_lock(), ane_lib.create_context(allow_unavailable=True) as ctx, ExitStack() as stack:
        assert ctx.ane_available and ctx.shared_event_available
        prog = stack.enter_context(ctx.compile_program(mil, weight_blob=blob, preformatted_blob=True))
        surf_in = stack.enter_context(ctx.surface(M, H + 1))
        surf_out = stack.enter_context(ctx.surface(M, H))

        for use_se in (True, False):
            with ctx.begin_ffn_wave(
                [prog], [surf_out], surf_in, x_fp16, signs_fp16, quantize_a8=True, use_shared_event=use_se
            ) as wave:
                y, metrics = wave.finish(tail_fp16)
            assert metrics["used_shared_event"] is use_se
            assert np.all(np.isfinite(y))
            err = compute_error_metrics(ref_fp32, y.astype(np.float32))
            assert err["cosine"] >= 0.999, f"cosine={err['cosine']} (use_se={use_se})"


def test_ane_ffn_startup_and_breaker_live_hw_gpu(ane_lib: AneHandoffLibrary):
    rng = np.random.default_rng(101)
    M, H, D_seg = 256, 1536, 2560
    x_fp16 = rng.normal(0.0, 1.0, size=(M, H)).astype(np.float16)
    signs_fp16 = make_hadamard_signs(H).astype(np.float16)

    wg = rng.normal(0.0, 0.02, size=(D_seg, H)).astype(np.float32)
    wu = rng.normal(0.0, 0.02, size=(D_seg, H)).astype(np.float32)
    wd = rng.normal(0.0, 0.02, size=(H, D_seg)).astype(np.float32)
    qg, sg = quantize_symmetric_int8(wg)
    qu, su = quantize_symmetric_int8(wu)
    qd, sd = quantize_symmetric_int8(wd)

    blob, offs = pack_weight_blobs(
        [
            ("wg_q", qg.reshape(D_seg, H, 1, 1), BLOB_DTYPE_INT8),
            ("wg_s", sg.astype(np.float16), BLOB_DTYPE_FP16),
            ("wu_q", qu.reshape(D_seg, H, 1, 1), BLOB_DTYPE_INT8),
            ("wu_s", su.astype(np.float16), BLOB_DTYPE_FP16),
            ("wd_q", qd.reshape(H, D_seg, 1, 1), BLOB_DTYPE_INT8),
            ("wd_s", sd.astype(np.float16), BLOB_DTYPE_FP16),
        ]
    )
    mil = build_ffn_geglu_segment_mil(
        H,
        D_seg,
        M,
        w_gate_off=offs["wg_q"],
        w_up_off=offs["wu_q"],
        w_down_off=offs["wd_q"],
        w_gate_scale_off=offs["wg_s"],
        w_up_scale_off=offs["wu_s"],
        w_down_scale_off=offs["wd_s"],
        weight_dtype="int8_dequant",
        input_dtype="fp16",
        activation="gelu_tanh",
        packed_token_scale=True,
        fp16_safe_clamp=True,
        normalize_mid=True,
    )

    measured_walls: list[float] = []
    with _hw_lock(), ane_lib.create_context(allow_unavailable=True) as ctx, ExitStack() as stack:
        prog = stack.enter_context(ctx.compile_program(mil, weight_blob=blob, preformatted_blob=True))
        surf_in = stack.enter_context(ctx.surface(M, H + 1))
        surf_out = stack.enter_context(ctx.surface(M, H))

        for _ in range(2):
            with ctx.begin_ffn_wave([prog], [surf_out], surf_in, x_fp16, signs_fp16) as w:
                w.finish(None)
        for _ in range(8):
            t0 = time.perf_counter()
            with ctx.begin_ffn_wave([prog], [surf_out], surf_in, x_fp16, signs_fp16) as w:
                w.finish(None)
            measured_walls.append((time.perf_counter() - t0) * 1000.0)

    med_ms = float(np.median(measured_walls))
    samples = [CalibrationSample(share=0.5, ane_ms=med_ms, tail_gpu_ms=med_ms, wall_ms=med_ms)]

    # 1) When baseline GPU is substantially slower than measured ANE wave, Startup enables SPLIT and Breaker holds
    dec_gain, cal = AneFfnStartup.decide(
        enabled=True,
        ane_available=True,
        tokens=M,
        hidden_size=H,
        intermediate_size=D_seg * 2,
        samples=samples,
        gpu_full_ms=med_ms * 3.0,
    )
    assert cal is not None
    assert dec_gain.state == AneFfnState.SPLIT
    breaker_ok = AneFfnBreaker(window=8)
    for w_ms in measured_walls:
        assert breaker_ok.observe(gpu_full_ms=med_ms * 3.0, split_ms=w_ms) is True
    assert not breaker_ok.tripped

    # 2) When baseline GPU is faster than measured ANE wave, Startup conservatively chooses NO_GAIN and Breaker trips
    dec_nogain, _ = AneFfnStartup.decide(
        enabled=True,
        ane_available=True,
        tokens=M,
        hidden_size=H,
        intermediate_size=D_seg * 2,
        samples=samples,
        gpu_full_ms=med_ms * 0.25,
    )
    assert dec_nogain.state == AneFfnState.NO_GAIN
    breaker_trip = AneFfnBreaker(window=8)
    for w_ms in measured_walls:
        breaker_trip.observe(gpu_full_ms=med_ms * 0.25, split_ms=w_ms)
    assert breaker_trip.tripped
