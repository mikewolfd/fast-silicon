"""CPU & locked-GPU tests for Apple10+ NAX `metal_mpp_int8` (`W8A8`/`W4A8`) gating and integration."""

from __future__ import annotations

import os

import pytest


def test_mpp_int8_apple9_strictly_disabled_cpu() -> None:
    from fast_shared.metal_mpp_int8 import (
        should_use_mpp_int8_linear,
        should_use_mpp_w4a8,
        should_use_mpp_w8a8,
    )

    shapes = (
        (1536, 12288),
        (6144, 1536),
        (1536, 2560),
        (2048, 1536),
        (1536, 256),
        (2880, 5120),
        (4096, 2880),
        (2880, 2880),
    )
    for bits in (4, 8):
        for m in (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            for k, n in shapes:
                assert not should_use_mpp_int8_linear(m, k, n, bits=bits, apple10_nax=False)
                if bits == 8:
                    assert not should_use_mpp_w8a8(m, k, n, apple10_nax=False)
                else:
                    assert not should_use_mpp_w4a8(m, k, n, apple10_nax=False)


def test_mpp_int8_apple10_regimes_cpu() -> None:
    from fast_shared.metal_mpp_int8 import (
        should_use_mpp_int8_linear,
        should_use_mpp_w4a8,
        should_use_mpp_w8a8,
    )

    # Unsupported formats / small dims / decode M < 32 must stay OFF even on Apple10
    assert not should_use_mpp_w8a8(16, 1536, 12288, apple10_nax=True)
    assert not should_use_mpp_w8a8(1024, 1536, 256, apple10_nax=True)
    assert not should_use_mpp_w8a8(1024, 256, 1536, apple10_nax=True)
    assert not should_use_mpp_w8a8(1024, 1536, 12288, group_size=32, apple10_nax=True)
    assert not should_use_mpp_w8a8(1024, 1536, 12288, mode="mxfp4", apple10_nax=True)
    assert not should_use_mpp_int8_linear(1024, 1536, 12288, bits=6, apple10_nax=True)

    # W8A8 winning regimes on Apple10 (large matrices >= 8M elems; medium < 8M stay OFF)
    assert should_use_mpp_w8a8(32, 1536, 12288, apple10_nax=True)
    assert should_use_mpp_w8a8(32, 6144, 1536, apple10_nax=True)
    assert should_use_mpp_w8a8(32, 2880, 5120, apple10_nax=True)
    assert should_use_mpp_w8a8(32, 4096, 2880, apple10_nax=True)
    assert should_use_mpp_w8a8(32, 2880, 2880, apple10_nax=True)
    assert not should_use_mpp_w8a8(256, 2048, 1536, apple10_nax=True)
    assert not should_use_mpp_w8a8(2048, 1536, 2560, apple10_nax=True)
    assert not should_use_mpp_w8a8(2048, 1536, 5120, apple10_nax=True)
    assert not should_use_mpp_w8a8(2048, 4096, 1536, apple10_nax=True)

    # W4A8 winning regimes on Apple10 (never medium shapes < 8M)
    assert not should_use_mpp_w4a8(2048, 1536, 2560, apple10_nax=True)
    assert not should_use_mpp_w4a8(2048, 2048, 1536, apple10_nax=True)
    assert should_use_mpp_w4a8(32, 1536, 12288, apple10_nax=True)
    assert should_use_mpp_w4a8(64, 1536, 12288, apple10_nax=True)
    assert should_use_mpp_w4a8(64, 6144, 1536, apple10_nax=True)
    assert not should_use_mpp_w4a8(256, 6144, 1536, apple10_nax=True)
    assert should_use_mpp_w4a8(512, 6144, 1536, apple10_nax=True)
    assert should_use_mpp_w4a8(32, 2880, 5120, apple10_nax=True)
    assert not should_use_mpp_w4a8(128, 2880, 2880, apple10_nax=True)
    assert should_use_mpp_w4a8(256, 2880, 2880, apple10_nax=True)


def test_mpp_int8_kernel_parity_and_nan_isolation_gpu() -> None:
    """Locked Metal GPU test: verifies Apple9 stays gated off and Apple10 NAX executes exact W8A8/W4A8 + NaN guard."""
    import contextlib
    import fcntl

    import mlx.core as mx
    import numpy as np

    from fast_shared.hw_check import supports_apple10_nax
    from fast_shared.metal_mpp_int8 import (
        mpp_quantized_linear,
        pack_affine_weights_for_mpp_i8,
        quantize_activations_i8_k64,
        should_use_mpp_int8_linear,
    )

    is_m5 = bool(supports_apple10_nax())
    lock_p = "/tmp/m5_gpu_validation.lock" if is_m5 else "/tmp/m4_gpu_validation.lock"

    @contextlib.contextmanager
    def _locked():
        fd = os.open(lock_p, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    with _locked():
        if not is_m5:
            assert not should_use_mpp_int8_linear(1024, 1536, 12288, bits=8)
            assert not should_use_mpp_int8_linear(1024, 1536, 12288, bits=4)
            return

        rng = np.random.default_rng(29)
        k, n = 1536, 2048
        w = mx.array(rng.normal(scale=0.04, size=(n, k)).astype(np.float32), dtype=mx.bfloat16)
        bias = mx.array(rng.normal(scale=0.1, size=(n,)).astype(np.float32), dtype=mx.bfloat16)
        for bits in (8, 4):
            wq, sq, bq = mx.quantize(w, group_size=64, bits=bits, mode="affine")
            mx.eval(wq, sq, bq)
            pw = pack_affine_weights_for_mpp_i8(wq, sq, bq, bits=bits)
            mx.eval(*pw)
            for m, dt in ((65, mx.bfloat16), (33, mx.float16)):
                x_np = rng.normal(scale=0.5, size=(m, k)).astype(np.float32)
                x_np[2, 70] = np.nan
                x_np[19, 400] = np.inf
                x = mx.array(x_np, dtype=dt)
                mx.eval(x)
                out = mpp_quantized_linear(
                    x, wq, sq, bq, bits=bits, bias=bias, out_dtype=dt, packed_weights=pw
                )
                ref = mx.quantized_matmul(
                    x, wq, scales=sq, biases=bq, transpose=True, group_size=64, bits=bits, mode="affine"
                ) + bias.astype(dt)
                mx.eval(out, ref)
                o_np = np.asarray(out.astype(mx.float32))
                r_np = np.asarray(ref.astype(mx.float32))
                assert np.all(np.isnan(o_np[2])) and np.all(np.isnan(o_np[19]))
                ok_rows = [r for r in range(m) if r not in (2, 19)]
                cos = float(
                    np.sum(r_np[ok_rows] * o_np[ok_rows])
                    / (np.linalg.norm(r_np[ok_rows]) * np.linalg.norm(o_np[ok_rows]) + 1e-12)
                )
                assert cos > 0.9997, f"bits={bits} m={m} dt={dt} cos={cos}"

        # Verify `x_meta` uses `mx.uint32` (compliant with strict IR dtype guard)
        x_q_u, x_m_u = quantize_activations_i8_k64(mx.ones((32, 128), dtype=mx.bfloat16))
        mx.eval(x_q_u, x_m_u)
        assert x_q_u.dtype == mx.int8
        assert x_m_u.dtype == mx.uint32


def test_fast_gemma4_new_without_mpp_attr_falls_back_cpu() -> None:
    """`FastGemma4Model.__new__` without `_mpp_apple10_int8_enabled` must safely fall back via `getattr`."""
    import mlx.core as mx
    import numpy as np
    from fast_gemma4.model import FastGemma4Model

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        m = FastGemma4Model.__new__(FastGemma4Model)
        rng = np.random.default_rng(11)
        x = mx.array(rng.normal(scale=0.1, size=(2, 64)).astype(np.float32), dtype=mx.bfloat16)
        lin = lambda arr: arr @ mx.ones((64, 32), dtype=mx.bfloat16)
        got = m._apply_qlinear({"proj": lin}, "proj", x)
        ref = lin(x)
        mx.eval(got, ref)
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32)), rtol=0, atol=0)
    finally:
        mx.set_default_device(prev_dev)

