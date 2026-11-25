"""CPU unit tests for shared 1-pass `gpu_argmax_conf` & `gpu_argmax_nan` (+Inf, -Inf, NaN, tie, tiny V)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from fast_shared.argmax import gpu_argmax_conf, gpu_argmax_nan, stats_to_logprob_margin


@pytest.mark.parametrize("vocab_n", [1, 2, 17, 513])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_argmax_nan_inf_and_tie_edge_cases_cpu(vocab_n: int, dtype: mx.Dtype) -> None:
    rng = np.random.default_rng(100 + vocab_n)
    rows = 5 if vocab_n >= 4 else 4
    arr = rng.normal(0.0, 1.5, size=(rows, vocab_n)).astype(np.float32)

    # Row 0: normal finite values
    # Row 1: +INFINITY must NOT poison LSE with NaN (fixes P1-2)
    arr[1, :] = -2.0
    arr[1, 0] = float("inf")
    if vocab_n >= 3:
        arr[1, 2] = float("inf")

    # Row 2: NaN must return idx=-1 and all-NaN stats
    arr[2, vocab_n - 1] = np.nan

    # Row 3: all -INFINITY must return finite/valid idx=0 without NaN LSE
    arr[3, :] = float("-inf")

    # Row 4 (when V >= 4): exact tie picks lowest index
    if rows == 5:
        arr[4, :] = -5.0
        arr[4, 1] = 3.25
        arr[4, 3] = 3.25

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x = mx.array(arr).astype(dtype)
        idx_nan = gpu_argmax_nan(x)
        idx_conf, stats = gpu_argmax_conf(x)
        mx.eval(idx_nan, idx_conf, stats)

        idx_n = idx_nan.tolist()
        idx_c = idx_conf.tolist()
        st_np = np.asarray(stats)

        assert idx_n == idx_c

        # Row 1 (+Inf): idx=0, top1=+inf, lse=+inf (never NaN)
        assert idx_c[1] == 0
        assert math.isinf(float(st_np[1, 0])) and float(st_np[1, 0]) > 0.0
        assert math.isinf(float(st_np[1, 2])) and float(st_np[1, 2]) > 0.0
        assert not math.isnan(float(st_np[1, 2]))
        lp_inf, margin_inf = stats_to_logprob_margin(
            float(st_np[1, 0]), float(st_np[1, 1]), float(st_np[1, 2])
        )
        assert lp_inf == -60.0 and margin_inf == 0.0

        # Row 2 (NaN): idx=-1 and NaN stats
        assert idx_c[2] == -1
        assert np.all(np.isnan(st_np[2]))

        # Row 3 (all -Inf): idx=0, lse is -inf (never NaN)
        assert idx_c[3] == 0
        assert float(st_np[3, 0]) == float("-inf")
        assert not math.isnan(float(st_np[3, 2]))

        # Row 4 (tie): picks lowest tied index
        if rows == 5:
            assert idx_c[4] == 1
            assert float(st_np[4, 1]) == pytest.approx(3.25, abs=1e-2)
    finally:
        mx.set_default_device(prev)
