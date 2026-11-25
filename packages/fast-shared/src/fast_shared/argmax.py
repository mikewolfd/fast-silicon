"""Single-pass NaN/Inf-safe Metal GPU argmax + online-softmax confidence stats with CPU fallbacks."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

_ARGMAX_CONF_KERNEL: Any | None = None
_ARGMAX_NAN_KERNEL: Any | None = None


def _select_tpg(vocab_n: int) -> int:
    """Scale threadgroup width to `vocab_n` (`32..512`, multiple of SIMD width 32)."""
    if vocab_n <= 32:
        return 32
    return min(512, ((vocab_n + 31) // 32) * 32)


def _get_argmax_conf_kernel() -> Any:
    global _ARGMAX_CONF_KERNEL
    if _ARGMAX_CONF_KERNEL is None:
        source = """
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_index_in_threadgroup;
            uint tpg = threads_per_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint n_simd = (tpg + 31) / 32;

            device const T* row_ptr = x + row * V;

            float local_max1 = -INFINITY;
            int local_idx1 = 0x7fffffff;
            float local_max2 = -INFINITY;
            float lse_max = -INFINITY;
            float lse_sum = 0.0f;
            bool local_nan = false;
            bool local_pos_inf = false;

            for (uint i = tid; i < V; i += tpg) {
                float v = float(row_ptr[i]);
                uint bits = as_type<uint>(v);
                if ((bits & 0x7fffffffu) > 0x7f800000u) {
                    local_nan = true;
                } else {
                    if (bits == 0x7f800000u) {
                        local_pos_inf = true;
                    } else if (isfinite(v)) {
                        if (v > lse_max) {
                            lse_sum = (lse_sum > 0.0f) ? (lse_sum * metal::exp(lse_max - v) + 1.0f) : 1.0f;
                            lse_max = v;
                        } else {
                            lse_sum += metal::exp(v - lse_max);
                        }
                    }
                    if (v > local_max1) {
                        local_max2 = local_max1;
                        local_max1 = v;
                        local_idx1 = int(i);
                    } else if (v > local_max2) {
                        local_max2 = v;
                    }
                }
            }

            threadgroup float sg_max1[32];
            threadgroup int sg_idx1[32];
            threadgroup float sg_max2[32];
            threadgroup float sg_lse_max[32];
            threadgroup float sg_lse_sum[32];
            threadgroup uint sg_flags[32];

            for (uint offset = 16; offset > 0; offset >>= 1) {
                float o_max1 = simd_shuffle_down(local_max1, offset);
                int o_idx1 = simd_shuffle_down(local_idx1, offset);
                float o_max2 = simd_shuffle_down(local_max2, offset);
                float o_lm = simd_shuffle_down(lse_max, offset);
                float o_ls = simd_shuffle_down(lse_sum, offset);
                uint o_nan = simd_shuffle_down(uint(local_nan), offset);
                uint o_pinf = simd_shuffle_down(uint(local_pos_inf), offset);

                local_nan = local_nan || (o_nan != 0);
                local_pos_inf = local_pos_inf || (o_pinf != 0);
                if (o_max1 > local_max1 || (o_max1 == local_max1 && o_idx1 < local_idx1)) {
                    local_max2 = max(local_max1, o_max2);
                    local_max1 = o_max1;
                    local_idx1 = o_idx1;
                } else {
                    local_max2 = max(local_max2, o_max1);
                }
                if (o_ls > 0.0f) {
                    if (lse_sum <= 0.0f) {
                        lse_max = o_lm;
                        lse_sum = o_ls;
                    } else {
                        float m_new = max(lse_max, o_lm);
                        lse_sum = lse_sum * metal::exp(lse_max - m_new) + o_ls * metal::exp(o_lm - m_new);
                        lse_max = m_new;
                    }
                }
            }

            if (simd_lane == 0) {
                sg_max1[simd_group] = local_max1;
                sg_idx1[simd_group] = local_idx1;
                sg_max2[simd_group] = local_max2;
                sg_lse_max[simd_group] = lse_max;
                sg_lse_sum[simd_group] = lse_sum;
                sg_flags[simd_group] = uint(local_nan) | (uint(local_pos_inf) << 1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                float m1 = (tid < n_simd) ? sg_max1[tid] : -INFINITY;
                int i1 = (tid < n_simd) ? sg_idx1[tid] : 0x7fffffff;
                float m2 = (tid < n_simd) ? sg_max2[tid] : -INFINITY;
                float lm = (tid < n_simd) ? sg_lse_max[tid] : -INFINITY;
                float ls = (tid < n_simd) ? sg_lse_sum[tid] : 0.0f;
                uint flg = (tid < n_simd) ? sg_flags[tid] : 0u;
                bool has_n = (flg & 1u) != 0;
                bool has_pi = (flg & 2u) != 0;

                for (uint offset = 16; offset > 0; offset >>= 1) {
                    float om1 = simd_shuffle_down(m1, offset);
                    int oi1 = simd_shuffle_down(i1, offset);
                    float om2 = simd_shuffle_down(m2, offset);
                    float olm = simd_shuffle_down(lm, offset);
                    float ols = simd_shuffle_down(ls, offset);
                    uint onan = simd_shuffle_down(uint(has_n), offset);
                    uint opi = simd_shuffle_down(uint(has_pi), offset);

                    has_n = has_n || (onan != 0);
                    has_pi = has_pi || (opi != 0);
                    if (om1 > m1 || (om1 == m1 && oi1 < i1)) {
                        m2 = max(m1, om2);
                        m1 = om1;
                        i1 = oi1;
                    } else {
                        m2 = max(m2, om1);
                    }
                    if (ols > 0.0f) {
                        if (ls <= 0.0f) {
                            lm = olm;
                            ls = ols;
                        } else {
                            float m_new = max(lm, olm);
                            ls = ls * metal::exp(lm - m_new) + ols * metal::exp(olm - m_new);
                            lm = m_new;
                        }
                    }
                }
                if (tid == 0) {
                    if (has_n || i1 == 0x7fffffff) {
                        out_idx[row] = -1;
                        stats[row * 3 + 0] = NAN;
                        stats[row * 3 + 1] = NAN;
                        stats[row * 3 + 2] = NAN;
                    } else if (has_pi) {
                        out_idx[row] = i1;
                        stats[row * 3 + 0] = INFINITY;
                        stats[row * 3 + 1] = m2;
                        stats[row * 3 + 2] = INFINITY;
                    } else {
                        out_idx[row] = i1;
                        stats[row * 3 + 0] = m1;
                        stats[row * 3 + 1] = m2;
                        stats[row * 3 + 2] = (ls > 0.0f) ? (lm + metal::log(ls)) : -INFINITY;
                    }
                }
            }
        """
        _ARGMAX_CONF_KERNEL = mx.fast.metal_kernel(
            name="fast_shared_argmax_conf_stats_1pass",
            input_names=["x"],
            output_names=["out_idx", "stats"],
            source=source,
        )
    return _ARGMAX_CONF_KERNEL


def _get_argmax_nan_kernel() -> Any:
    global _ARGMAX_NAN_KERNEL
    if _ARGMAX_NAN_KERNEL is None:
        source = """
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_index_in_threadgroup;
            uint tpg = threads_per_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint n_simd = (tpg + 31) / 32;

            device const T* row_ptr = x + row * V;

            float local_max1 = -INFINITY;
            int local_idx1 = 0x7fffffff;
            bool local_nan = false;

            for (uint i = tid; i < V; i += tpg) {
                float v = float(row_ptr[i]);
                if ((as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u) {
                    local_nan = true;
                } else if (v > local_max1) {
                    local_max1 = v;
                    local_idx1 = int(i);
                }
            }

            threadgroup float sg_max1[32];
            threadgroup int sg_idx1[32];
            threadgroup uint sg_nan[32];

            for (uint offset = 16; offset > 0; offset >>= 1) {
                float o_max1 = simd_shuffle_down(local_max1, offset);
                int o_idx1 = simd_shuffle_down(local_idx1, offset);
                uint o_nan = simd_shuffle_down(uint(local_nan), offset);
                local_nan = local_nan || (o_nan != 0);
                if (o_max1 > local_max1 || (o_max1 == local_max1 && o_idx1 < local_idx1)) {
                    local_max1 = o_max1;
                    local_idx1 = o_idx1;
                }
            }
            if (simd_lane == 0) {
                sg_max1[simd_group] = local_max1;
                sg_idx1[simd_group] = local_idx1;
                sg_nan[simd_group] = uint(local_nan);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                float m1 = (tid < n_simd) ? sg_max1[tid] : -INFINITY;
                int i1 = (tid < n_simd) ? sg_idx1[tid] : 0x7fffffff;
                bool has_n = (tid < n_simd) ? (sg_nan[tid] != 0) : false;
                for (uint offset = 16; offset > 0; offset >>= 1) {
                    float om1 = simd_shuffle_down(m1, offset);
                    int oi1 = simd_shuffle_down(i1, offset);
                    uint onan = simd_shuffle_down(uint(has_n), offset);
                    has_n = has_n || (onan != 0);
                    if (om1 > m1 || (om1 == m1 && oi1 < i1)) {
                        m1 = om1;
                        i1 = oi1;
                    }
                }
                if (tid == 0) {
                    out_idx[row] = (has_n || i1 == 0x7fffffff) ? -1 : i1;
                }
            }
        """
        _ARGMAX_NAN_KERNEL = mx.fast.metal_kernel(
            name="fast_shared_argmax_nan",
            input_names=["x"],
            output_names=["out_idx"],
            source=source,
        )
    return _ARGMAX_NAN_KERNEL


def _can_use_metal_argmax() -> bool:
    return bool(mx.metal.is_available() and mx.default_device().type == mx.DeviceType.gpu)


def gpu_argmax_conf(logits_2d: mx.array) -> tuple[mx.array, mx.array]:
    """Compute `(top1_idx[rows], stats[rows, 3])` in one pass where `stats=[top1_val, top2_val, lse]` (`-1` on NaN)."""
    if logits_2d.ndim != 2:
        raise ValueError(f"Expected 2D logits array, got shape {logits_2d.shape}")
    rows, vocab_n = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    if rows == 0:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0, 3), dtype=mx.float32)  # guard: allow-fp32
    if vocab_n == 0:
        return (
            mx.full((rows,), -1, dtype=mx.int32),
            mx.full((rows, 3), float("nan"), dtype=mx.float32),  # guard: allow-fp32
        )

    if not _can_use_metal_argmax():
        arr_f32 = logits_2d.astype(mx.float32)  # guard: allow-fp32
        has_nan = mx.any(mx.isnan(arr_f32), axis=-1)
        pos_inf = arr_f32 == float("inf")
        has_pos_inf = mx.any(pos_inf, axis=-1)
        top1_idx = mx.argmax(arr_f32, axis=-1).astype(mx.int32)
        top1_val = mx.max(arr_f32, axis=-1)
        if vocab_n == 1:
            top2_val = mx.full((rows,), float("-inf"), dtype=mx.float32)  # guard: allow-fp32
        else:
            col_ids = mx.arange(vocab_n, dtype=mx.int32)[None, :]
            masked = mx.where(
                col_ids == top1_idx[:, None],
                mx.array(float("-inf"), dtype=mx.float32),  # guard: allow-fp32
                arr_f32,
            )
            top2_val = mx.max(masked, axis=-1)
        # Exclude +Inf when computing finite logsumexp, then restore +inf when +Inf is present and no NaN.
        fin_vals = mx.where(pos_inf, mx.array(float("-inf"), dtype=mx.float32), arr_f32)  # guard: allow-fp32
        lse_fin = mx.logsumexp(fin_vals, axis=-1)
        lse = mx.where(has_pos_inf, mx.array(float("inf"), dtype=mx.float32), lse_fin)  # guard: allow-fp32
        idx = mx.where(has_nan, mx.array(-1, dtype=mx.int32), top1_idx)
        stats = mx.where(
            has_nan[:, None],
            mx.array(float("nan"), dtype=mx.float32),  # guard: allow-fp32
            mx.stack([top1_val, top2_val, lse], axis=-1),
        )
        return idx, stats

    x = mx.contiguous(logits_2d)
    dtype = x.dtype if x.dtype in (mx.bfloat16, mx.float16, mx.float32) else mx.float32
    if x.dtype != dtype:
        x = x.astype(dtype)
    kernel = _get_argmax_conf_kernel()
    tpg = _select_tpg(vocab_n)
    idx, stats = kernel(  # guard: allow-fp32
        inputs=[x],
        template=[("T", dtype), ("V", vocab_n)],
        grid=(rows * tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(rows,), (rows, 3)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return idx, stats


def gpu_argmax_nan(logits_2d: mx.array) -> mx.array:
    """Compute `top1_idx[rows]` (`-1` on NaN or empty vocab) without copying logits to CPU."""
    if logits_2d.ndim != 2:
        raise ValueError(f"Expected 2D logits array, got shape {logits_2d.shape}")
    rows, vocab_n = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    if rows == 0:
        return mx.zeros((0,), dtype=mx.int32)
    if vocab_n == 0:
        return mx.full((rows,), -1, dtype=mx.int32)

    if not _can_use_metal_argmax():
        arr_f32 = logits_2d.astype(mx.float32)  # guard: allow-fp32
        has_nan = mx.any(mx.isnan(arr_f32), axis=-1)
        top1_idx = mx.argmax(arr_f32, axis=-1).astype(mx.int32)
        return mx.where(has_nan, mx.array(-1, dtype=mx.int32), top1_idx)

    x = mx.contiguous(logits_2d)
    dtype = x.dtype if x.dtype in (mx.bfloat16, mx.float16, mx.float32) else mx.float32
    if x.dtype != dtype:
        x = x.astype(dtype)
    kernel = _get_argmax_nan_kernel()
    tpg = _select_tpg(vocab_n)
    (idx,) = kernel(
        inputs=[x],
        template=[("T", dtype), ("V", vocab_n)],
        grid=(rows * tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.int32],
    )
    return idx


def stats_to_logprob_margin(top1_val: float, top2_val: float, lse: float) -> tuple[float, float]:
    """Convert `(top1_val, top2_val, lse)` GPU scalars to `(logprob, margin)` matching `_StepConfidenceState`."""
    if not math.isfinite(top1_val):
        return -60.0, 0.0
    lp = min(0.0, top1_val - lse) if math.isfinite(lse) else -60.0
    margin = (top1_val - top2_val) if top2_val > -1e29 else 0.0
    return float(lp), float(margin)


__all__ = [
    "gpu_argmax_conf",
    "gpu_argmax_nan",
    "stats_to_logprob_margin",
]
