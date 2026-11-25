"""Fused projections, general per-layer spec-bucketing, `lora_accel`, and GDN/RoPE kernels for `fast-qwen3`."""

from __future__ import annotations

from fast_shared.grammar import compute_loop_breaker_penalties
from fast_shared.hw_check import detect_apple10_d256_force_fused
from fast_shared.kv_quant import (
    SpecBucketedLinearGroup,
    concat_compatible_linears,
    linear_quant_spec_key,
)
from fast_shared.prefix_tree import (
    dequantize_gdn_state_block32,
    quantize_gdn_state_block32,
)
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
import math
import os
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class QwenDoubleQuantError(ValueError):
    """Raised when attempting 2-pass `dequant -> merge LoRA -> 4-bit requant` (grid-snap erasure)."""


def detect_rmsnorm_add_one(
    raw_keys: Sequence[str],
    sample_norm_weight: mx.array | np.ndarray,
) -> bool:
    """Detect HF zero-centered `1 + w` RMSNorm (`mean ~ 0`) vs `mlx-lm` pre-shifted `w` (`mean ~ 1`).

    - `mlx-community/*` checkpoints store keys under `language_model.model.*` with `+1.0` already
      baked into `RMSNorm.weight` (`mean ~ 0.95..1.05`).
    - Official `Qwen/Qwen3.5-*` checkpoints store keys under `model.language_model.*` or `model.layers.*`
      with zero-centered weights (`mean ~ 0.0..0.15`), requiring `(1.0 + w)` in FP32 at load time.
    """
    arr = np.asarray(sample_norm_weight.astype(mx.float32)) if isinstance(sample_norm_weight, mx.array) else np.asarray(sample_norm_weight, dtype=np.float32)
    mean_val = float(np.mean(arr))
    if any(k.startswith("language_model.model.") for k in raw_keys) and mean_val > 0.5:
        return False
    if any(k.startswith("model.language_model.") for k in raw_keys) and mean_val < 0.65:
        return True
    return abs(mean_val - 1.0) > abs(mean_val)


@partial(mx.compile, shapeless=True)
def _rmsnorm_gated_compiled(
    y: mx.array,
    z: mx.array,
    weight: mx.array,
    eps: float,
) -> mx.array:
    yf = mx.fast.rms_norm(y.astype(mx.float32), weight.astype(mx.float32), eps)
    zf = nn.silu(z.astype(mx.float32))
    return (yf * zf).astype(y.dtype)


class RMSNormGated(nn.Module):
    """Unshifted Gated RMSNorm (`y_norm * silu(z)` in `float32`) used inside `GatedDeltaNet`."""

    def __init__(self, weight: mx.array, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight.astype(mx.bfloat16)
        self.eps = float(eps)

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        return _rmsnorm_gated_compiled(y, z, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def fused_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def fused_sigmoid_attn_gate(attn_out: mx.array, gate: mx.array) -> mx.array:
    return (attn_out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(attn_out.dtype)


@partial(mx.compile, shapeless=True)
def compute_gdn_gates(
    A_log: mx.array,
    a: mx.array,
    dt_bias: mx.array,
    b: mx.array,
) -> tuple[mx.array, mx.array]:
    g = mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
    )
    beta = mx.sigmoid(b.astype(mx.float32)).astype(a.dtype)
    return g, beta


@partial(mx.compile, shapeless=True)
def normalize_gdn_qk(
    q: mx.array,
    k: mx.array,
    inv_scale: float,
    rms_eps: float,
) -> tuple[mx.array, mx.array]:
    """L2-normalize GDN `(q, k)` in `float32` to avoid subnormal `rms_eps` underflow."""
    qf = mx.fast.rms_norm(q.astype(mx.float32), None, rms_eps) * (inv_scale * inv_scale)
    kf = mx.fast.rms_norm(k.astype(mx.float32), None, rms_eps) * inv_scale
    return qf.astype(q.dtype), kf.astype(k.dtype)


@mx.compile
def apply_partial_rope_trig(
    x: mx.array,
    cos_a: mx.array,
    sin_a: mx.array,
    rot_dims: int,
) -> mx.array:
    """Apply non-traditional (`GPT-NeoX` half-split) RoPE to the leading `rot_dims` of `x`."""
    half = rot_dims // 2
    x_rot, x_pass = mx.split(x, [rot_dims], axis=-1)
    x0, x1 = mx.split(x_rot, [half], axis=-1)
    c = cos_a.astype(x.dtype)
    s = sin_a.astype(x.dtype)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return mx.concatenate([y0, y1, x_pass], axis=-1)


def permute_q_proj_rows_to_q_gate(
    arr: mx.array,
    num_q_heads: int,
    head_dim: int,
) -> tuple[mx.array, mx.array]:
    """Split Qwen3.5 interleaved `[num_q_heads, 2, head_dim, ...]` rows into `(query_rows, gate_rows)` (`0` ULP)."""
    tail_shape = arr.shape[1:]
    reshaped = arr.reshape((num_q_heads, 2, head_dim, *tail_shape))
    q_rows = mx.contiguous(reshaped[:, 0].reshape((num_q_heads * head_dim, *tail_shape)))
    g_rows = mx.contiguous(reshaped[:, 1].reshape((num_q_heads * head_dim, *tail_shape)))
    return q_rows, g_rows


def split_linear_q_proj_to_q_gate(
    lin: nn.Module,
    num_q_heads: int,
    head_dim: int,
) -> tuple[nn.Module, nn.Module]:
    """Split a `Linear` or `QuantizedLinear` `q_proj` along axis 0 into `(q_query_lin, q_gate_lin)` with `0.0` ULP."""
    out_half = num_q_heads * head_dim
    if isinstance(lin, nn.QuantizedLinear):
        in_dims = (int(lin.weight.shape[1]) * 32) // int(lin.bits)
        wq, wg = permute_q_proj_rows_to_q_gate(lin.weight, num_q_heads, head_dim)
        sq, sg = permute_q_proj_rows_to_q_gate(lin.scales, num_q_heads, head_dim)
        has_b = getattr(lin, "biases", None) is not None
        lq = nn.QuantizedLinear(in_dims, out_half, bias=False, group_size=lin.group_size, bits=lin.bits)
        lg = nn.QuantizedLinear(in_dims, out_half, bias=False, group_size=lin.group_size, bits=lin.bits)
        lq.weight, lq.scales = wq, sq
        lg.weight, lg.scales = wg, sg
        if has_b:
            bq, bg = permute_q_proj_rows_to_q_gate(lin.biases, num_q_heads, head_dim)
            lq.biases, lg.biases = bq, bg
        if getattr(lin, "bias", None) is not None:
            bi_q, bi_g = permute_q_proj_rows_to_q_gate(lin.bias, num_q_heads, head_dim)
            lq.bias, lg.bias = bi_q, bi_g
        return lq, lg

    if isinstance(lin, nn.Linear):
        in_dims = int(lin.weight.shape[1])
        has_bias = "bias" in lin
        wq, wg = permute_q_proj_rows_to_q_gate(lin.weight, num_q_heads, head_dim)
        lq = nn.Linear(in_dims, out_half, bias=has_bias)
        lg = nn.Linear(in_dims, out_half, bias=has_bias)
        lq.weight, lg.weight = wq, wg
        if has_bias:
            bi_q, bi_g = permute_q_proj_rows_to_q_gate(lin.bias, num_q_heads, head_dim)
            lq.bias, lg.bias = bi_q, bi_g
        return lq, lg

    raise TypeError(f"Unsupported q_proj module type: {type(lin)}")


def merge_lora_weights_safe(
    base_weights: Mapping[str, mx.array],
    lora_tensors: Mapping[str, tuple[mx.array, mx.array, float]],
    *,
    target_bit_schedule: Mapping[str, int] | None = None,
    group_size: int = 64,
    allow_4bit_requant: bool = False,
) -> dict[str, mx.array]:
    """Offline 1-pass `BF16 -> OptiQ/quant` LoRA merge with hard block on 2-pass `4-bit` requantization.

    Verified empirically: `dequant(4-bit) + delta -> requant(4-bit)` suffers grid-snap erasure
    (`4-bit RMSE 0.00123` is `3.9x` `delta RMS 0.00032`, collapsing downstream EM `64.2% -> 10.0%`).
    """
    out: dict[str, mx.array] = dict(base_weights)
    for stem, (la, lb, scale) in lora_tensors.items():
        sc_key = stem + ".scales"
        w_key = stem + ".weight"
        if w_key not in out:
            raise KeyError(f"Missing base weight {w_key!r} for LoRA stem {stem!r}")

        is_prequant = sc_key in out
        if is_prequant:
            wq = out[w_key]
            sc = out[sc_key]
            in_dims = int(la.shape[1])
            src_bits = (int(wq.shape[-1]) * 32) // in_dims
            dst_bits = int(target_bit_schedule.get(stem, src_bits)) if target_bit_schedule else src_bits
            if (src_bits <= 4 or dst_bits <= 4) and not allow_4bit_requant:
                raise QwenDoubleQuantError(
                    f"Refusing 2-pass dequant->merge->requant on {stem!r} (src={src_bits}-bit, dst={dst_bits}-bit): "
                    f"4-bit grid-snap erases low-rank LoRA deltas. Use hot-swappable `load_lora_adapter` (`lora_accel`) "
                    f"on quantized checkpoints or merge into BF16 base weights first."
                )
            src_gs = in_dims // int(sc.shape[-1])
            bi = out.get(stem + ".biases")
            w_dense = mx.dequantize(wq, sc, bi, group_size=src_gs, bits=src_bits)
        else:
            w_dense = out[w_key]
            dst_bits = int(target_bit_schedule[stem]) if (target_bit_schedule and stem in target_bit_schedule) else 16

        delta = (lb.astype(mx.float32) @ la.astype(mx.float32)) * float(scale)
        w_merged = (w_dense.astype(mx.float32) + delta).astype(mx.bfloat16)

        if dst_bits < 16:
            wq_new, sc_new, bi_new = mx.quantize(w_merged, group_size=group_size, bits=dst_bits)
            out[w_key] = wq_new
            out[sc_key] = sc_new
            out[stem + ".biases"] = bi_new
        else:
            out[w_key] = w_merged
            out.pop(sc_key, None)
            out.pop(stem + ".biases", None)
    return out


def quantize_gdn_state_int8(state: mx.array) -> tuple[mx.array, mx.array]:
    """Block-32 symmetric `INT8` compression (`3.76x` memory reduction) via `fast_shared.prefix_tree`."""
    if int(state.shape[-1]) % 32 != 0:
        raise ValueError(f"GDN Dk must be divisible by 32 for block-32 INT8 codec, got {state.shape[-1]}")
    return quantize_gdn_state_block32(state)


def dequantize_gdn_state_int8(
    codes: mx.array,
    scales: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Restore `[B, Hv, Dv, Dk]` GDN recurrent state from block-32 `(int8 codes, bfloat16 scales)`."""
    return dequantize_gdn_state_block32(codes, scales, out_dtype=dtype)


def build_additive_causal_mask(
    pad_lens: Sequence[int],
    q_len: int,
    kv_len: int | None = None,
    *,
    act_dtype: mx.Dtype = mx.bfloat16,
) -> str | mx.array:
    """Build `fast-ir-guard`-clean additive `act_dtype` mask (`'causal'` when unpadded, never bool SDPA mask)."""
    eff_kv = q_len if kv_len is None else int(kv_len)
    if all(int(pl) == 0 for pl in pad_lens) and eff_kv == q_len:
        return "causal"
    pl = mx.array([int(p) for p in pad_lens], dtype=mx.int32)[:, None, None, None]
    if q_len == 1 and eff_kv > 1:
        k_pos = mx.arange(eff_kv, dtype=mx.int32)[None, None, None, :]
        allow = k_pos >= pl
    else:
        offset = eff_kv - q_len
        q_pos = mx.arange(q_len, dtype=mx.int32)[None, None, :, None] + offset
        k_pos = mx.arange(eff_kv, dtype=mx.int32)[None, None, None, :]
        allow = ((q_pos >= pl) & (k_pos >= pl) & (k_pos <= q_pos)) | ((q_pos < pl) & (k_pos == q_pos))
    neg = -65504.0 if act_dtype == mx.float16 else -1e9
    return mx.where(allow, mx.array(0.0, dtype=act_dtype), mx.array(neg, dtype=act_dtype))


def safe_training_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: str | mx.array = "causal",
) -> mx.array:
    """Exact GQA matmul + `precise=True` softmax bypassing MLX `0.32.4` Apple10 `D=256` SDPA VJP NaNs."""
    b, h_q, s_q, d = q.shape
    _, h_kv, s_k, _ = k.shape
    if h_q != h_kv:
        rep = h_q // h_kv
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
    scores = mx.matmul(q.astype(mx.float32), mx.transpose(k.astype(mx.float32), (0, 1, 3, 2))) * float(scale)
    if isinstance(mask, str):
        if mask == "causal" and s_q > 1:
            offset = s_k - s_q
            qi = mx.arange(s_q, dtype=mx.int32)[:, None] + offset
            ki = mx.arange(s_k, dtype=mx.int32)[None, :]
            scores = mx.where(ki <= qi, scores, mx.array(-1e9, dtype=mx.float32))
    elif mask is not None:
        scores = scores + mask.astype(mx.float32)
    probs = mx.softmax(scores, axis=-1, precise=True)
    return mx.matmul(probs, v.astype(mx.float32)).astype(q.dtype)


def find_repeated_4gram_penalties(
    token_history: Sequence[int],
    *,
    window: int = 96,
    min_repeats: int = 4,
    penalty: float = 8.0,
    stop_token_ids: frozenset[int] = frozenset(),
) -> dict[int, float]:
    """Delegate to `fast_shared.grammar.compute_loop_breaker_penalties` (`A4_v2` `p=1..8` cycle + 4-gram breaker)."""
    return compute_loop_breaker_penalties(
        token_history,
        window=window,
        min_prior_hits=min_repeats,
        base_penalty=penalty,
        stop_token_ids=stop_token_ids,
    )


_ARGMAX_CONF_KERNEL: Any | None = None
_ARGMAX_NAN_KERNEL: Any | None = None


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
            bool local_nan = false;

            for (uint i = tid; i < V; i += tpg) {
                float v = float(row_ptr[i]);
                if (isnan(v)) {
                    local_nan = true;
                } else if (v > local_max1) {
                    local_max2 = local_max1;
                    local_max1 = v;
                    local_idx1 = int(i);
                } else if (v > local_max2) {
                    local_max2 = v;
                }
            }

            threadgroup float sg_max1[32];
            threadgroup int sg_idx1[32];
            threadgroup float sg_max2[32];
            threadgroup uint sg_nan[32];

            for (uint offset = 16; offset > 0; offset >>= 1) {
                float o_max1 = simd_shuffle_down(local_max1, offset);
                int o_idx1 = simd_shuffle_down(local_idx1, offset);
                float o_max2 = simd_shuffle_down(local_max2, offset);
                uint o_nan = simd_shuffle_down(uint(local_nan), offset);
                local_nan = local_nan || (o_nan != 0);
                if (o_max1 > local_max1 || (o_max1 == local_max1 && o_idx1 < local_idx1)) {
                    local_max2 = max(local_max1, o_max2);
                    local_max1 = o_max1;
                    local_idx1 = o_idx1;
                } else {
                    local_max2 = max(local_max2, o_max1);
                }
            }
            if (simd_lane == 0) {
                sg_max1[simd_group] = local_max1;
                sg_idx1[simd_group] = local_idx1;
                sg_max2[simd_group] = local_max2;
                sg_nan[simd_group] = uint(local_nan);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group == 0) {
                float m1 = (tid < n_simd) ? sg_max1[tid] : -INFINITY;
                int i1 = (tid < n_simd) ? sg_idx1[tid] : 0x7fffffff;
                float m2 = (tid < n_simd) ? sg_max2[tid] : -INFINITY;
                bool has_n = (tid < n_simd) ? (sg_nan[tid] != 0) : false;
                for (uint offset = 16; offset > 0; offset >>= 1) {
                    float om1 = simd_shuffle_down(m1, offset);
                    int oi1 = simd_shuffle_down(i1, offset);
                    float om2 = simd_shuffle_down(m2, offset);
                    uint onan = simd_shuffle_down(uint(has_n), offset);
                    has_n = has_n || (onan != 0);
                    if (om1 > m1 || (om1 == m1 && oi1 < i1)) {
                        m2 = max(m1, om2);
                        m1 = om1;
                        i1 = oi1;
                    } else {
                        m2 = max(m2, om1);
                    }
                }
                if (tid == 0) {
                    sg_max1[0] = m1;
                    sg_idx1[0] = i1;
                    sg_max2[0] = m2;
                    sg_nan[0] = uint(has_n);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float row_max1 = sg_max1[0];
            int row_idx1 = sg_idx1[0];
            float row_max2 = sg_max2[0];
            bool row_nan = (sg_nan[0] != 0);

            float local_sum = 0.0f;
            if (!row_nan) {
                for (uint i = tid; i < V; i += tpg) {
                    local_sum += metal::exp(float(row_ptr[i]) - row_max1);
                }
            }
            local_sum = simd_sum(local_sum);
            if (simd_lane == 0) {
                sg_max1[simd_group] = local_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float s = (tid < n_simd) ? sg_max1[tid] : 0.0f;
                s = simd_sum(s);
                if (tid == 0) {
                    out_idx[row] = (row_nan || row_idx1 == 0x7fffffff) ? -1 : row_idx1;
                    out_stats[row * 3 + 0] = row_max1;
                    out_stats[row * 3 + 1] = row_max2;
                    out_stats[row * 3 + 2] = row_nan ? NAN : (row_max1 + metal::log(max(s, 1e-30f)));
                }
            }
        """
        _ARGMAX_CONF_KERNEL = mx.fast.metal_kernel(
            name="fast_qwen3_argmax_conf_stats",
            input_names=["x"],
            output_names=["out_idx", "out_stats"],
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
                if (isnan(v)) {
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
            name="fast_qwen3_argmax_nan",
            input_names=["x"],
            output_names=["out_idx"],
            source=source,
        )
    return _ARGMAX_NAN_KERNEL


def _can_use_metal_argmax() -> bool:
    return bool(mx.metal.is_available() and mx.default_device().type == mx.DeviceType.gpu)


def gpu_argmax_conf(logits_2d: mx.array) -> tuple[mx.array, mx.array]:
    """Compute `(top1_idx[rows], stats[rows, 3])` on GPU where `stats=[top1_val, top2_val, lse]` (`-1` on NaN)."""
    if logits_2d.ndim != 2:
        raise ValueError(f"Expected 2D logits array, got shape {logits_2d.shape}")
    rows, vocab_n = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    if rows == 0:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0, 3), dtype=mx.float32)
    if vocab_n == 0:
        return mx.full((rows,), -1, dtype=mx.int32), mx.full((rows, 3), float("nan"), dtype=mx.float32)

    if not _can_use_metal_argmax():
        arr_f32 = logits_2d.astype(mx.float32)
        has_nan = mx.any(mx.isnan(arr_f32), axis=-1)
        top1_idx = mx.argmax(arr_f32, axis=-1).astype(mx.int32)
        top1_val = mx.max(arr_f32, axis=-1)
        if vocab_n == 1:
            top2_val = mx.full((rows,), float("-inf"), dtype=mx.float32)
        else:
            col_ids = mx.arange(vocab_n, dtype=mx.int32)[None, :]
            masked = mx.where(col_ids == top1_idx[:, None], mx.array(float("-inf"), dtype=mx.float32), arr_f32)
            top2_val = mx.max(masked, axis=-1)
        lse = mx.logsumexp(arr_f32, axis=-1)
        idx = mx.where(has_nan, mx.array(-1, dtype=mx.int32), top1_idx)
        stats = mx.where(has_nan[:, None], mx.array(float("nan"), dtype=mx.float32), mx.stack([top1_val, top2_val, lse], axis=-1))
        return idx, stats

    x = mx.contiguous(logits_2d)
    dtype = x.dtype if x.dtype in (mx.bfloat16, mx.float16, mx.float32) else mx.float32
    if x.dtype != dtype:
        x = x.astype(dtype)
    kernel = _get_argmax_conf_kernel()
    tpg = 512
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
        arr_f32 = logits_2d.astype(mx.float32)
        has_nan = mx.any(mx.isnan(arr_f32), axis=-1)
        top1_idx = mx.argmax(arr_f32, axis=-1).astype(mx.int32)
        return mx.where(has_nan, mx.array(-1, dtype=mx.int32), top1_idx)

    x = mx.contiguous(logits_2d)
    dtype = x.dtype if x.dtype in (mx.bfloat16, mx.float16, mx.float32) else mx.float32
    if x.dtype != dtype:
        x = x.astype(dtype)
    kernel = _get_argmax_nan_kernel()
    tpg = 512
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
