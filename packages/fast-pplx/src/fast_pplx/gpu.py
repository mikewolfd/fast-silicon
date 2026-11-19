"""Metal/MLX GPU engine for pplx-embed-v2-context-0.6b.

Combines:
- Pure MLX/Metal 12-layer hybrid Qwen3.5 backbone (alternating GatedDeltaNet +
  bidirectional Full Attention, overriding config ``full_attention_interval=4`` to ``2``)
- Zero-centered ``+1.0`` weight shifts kept in ``float32`` on ``RMSNorm`` layers
  (leaving ``GatedRMSNorm`` unshifted and multiplied in ``float32``)
- FP32 ``normalize_qk`` and ``PplxRMSNorm`` protecting against FP16 subnormal flush and
  deep residual overflow while preserving ``bfloat16`` precision
- Fused ``qkv_proj`` (``q+gate, k, v``) and ``gate_up_proj`` projections
- Fast ``mx.float16`` / ``mx.bfloat16`` GatedDeltaNet path (``S <= 2048``) with zero per-layer ``mx.eval``
- Long-context streamed path (``S > 2048``, up to ``32K``) carrying FP32 recurrent GDN + Conv1d
  state across ``1024``-token slices with async pipelining (``mx.async_eval``) and chip-gated
  ``SDPA`` slices (``1024`` on M5 ``Apple10`` NAX hosts, ``512`` on M4 ``Apple9``)
- ``MappedEmbedding`` safetensors memmap gathered as ``float16`` keeping ``embed_tokens`` out of Metal GPU RAM
- MIT-adapted ``gated_delta_update`` kernels from ``mlx-lm`` 0.32.2 (Copyright © 2025 Apple Inc.)
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import inspect
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterator, Mapping, Sequence

# Default MLX_ENABLE_TF32=0 before importing mlx.core so FP32 projections and reference
# paths stay exact IEEE FP32 across M4 (Apple9) and M5 (Apple10 NAX) unless explicitly overridden.
_TF32_WAS_USER_SET = "MLX_ENABLE_TF32" in os.environ
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from fast_shared.bench import clear_mlx_arena
from fast_shared.embeddings import (
    MappedEmbedding,
    resolve_embedding_dim,
    set_mlx_cache_limit,
    tile_aligned_slices,
    truncate_and_normalize,
)
from fast_shared.hf import resolve_hf_model_dir

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .paths import RepoPaths, resolve_paths
from .tokenizer import ContextualTokenizer

_STREAMED_SEQ_THRESHOLD = 2048
_GDN_SLICE_LEN = 1024
_DEFAULT_SDPA_SLICE_LEN = 512
_M5_SDPA_SLICE_LEN = 1024
_SDPA_SLICE_LEN = _DEFAULT_SDPA_SLICE_LEN
_SDPA_MIN_TAIL = _SDPA_SLICE_LEN
_STREAMED_SYNC_STRIDE = 4

_ZERO_CENTERED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

_EMBED_TENSOR_CANDIDATES = (
    "language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)

_ENABLE_GDN_PACKED = os.environ.get("MLX_GDN_PACKED", "1") != "0"
_ARCH_GEN_RE = re.compile(r"g(\d+)", re.IGNORECASE)


def _sdpa_supports_force_fused() -> bool:
    try:
        sig = inspect.signature(mx.fast.scaled_dot_product_attention)
        return "force_fused" in sig.parameters
    except (TypeError, ValueError):
        return False


_SDPA_SUPPORTS_FORCE_FUSED = _sdpa_supports_force_fused()


def _parse_macos_version(ver: str | None = None) -> tuple[int, int]:
    raw = platform.mac_ver()[0] if ver is None else str(ver)
    parts = [p for p in raw.strip().split(".") if p.isdigit()]
    if not parts:
        return (0, 0)
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    return (major, minor)


def _get_mlx_device_info() -> dict[str, Any]:
    dev_info_fn = getattr(mx, "device_info", None)
    if callable(dev_info_fn):
        try:
            return dict(dev_info_fn())
        except Exception:
            pass
    metal_mod = getattr(mx, "metal", None)
    metal_info_fn = getattr(metal_mod, "device_info", None) if metal_mod is not None else None
    if callable(metal_info_fn):
        try:
            return dict(metal_info_fn())
        except Exception:
            pass
    return {}


def _detect_gpu_family_info(
    device_info: Mapping[str, Any] | None = None,
    macos_version: str | None = None,
) -> dict[str, Any]:
    """Detect Apple9 (M4) vs Apple10+ (M5+ NAX) GPU family for SDPA dispatch."""
    info = _get_mlx_device_info() if device_info is None else dict(device_info)
    name = str(info.get("device_name", ""))
    arch = str(info.get("architecture", ""))
    name_low = name.lower()

    gpu_gen = 0
    match = _ARCH_GEN_RE.search(arch)
    if match is not None:
        gpu_gen = int(match.group(1))

    is_apple10 = (
        gpu_gen >= 17
        or any(tok in name_low for tok in ("apple m5", "apple m6", "apple m7", "apple m8", "apple m9"))
    )
    os_ver_str = platform.mac_ver()[0] if macos_version is None else str(macos_version)
    os_tuple = _parse_macos_version(os_ver_str)
    nax_sdpa_eligible = bool(is_apple10 and os_tuple >= (26, 2))
    sdpa_slice_len = _M5_SDPA_SLICE_LEN if nax_sdpa_eligible else _DEFAULT_SDPA_SLICE_LEN
    return {
        "device_name": name,
        "architecture": arch,
        "gpu_gen": gpu_gen,
        "macos_version": os_ver_str,
        "is_apple10": is_apple10,
        "nax_sdpa_eligible": nax_sdpa_eligible,
        "sdpa_slice_len": sdpa_slice_len,
    }


def get_mlx_runtime_telemetry() -> dict[str, Any]:
    """Return active MLX precision, TF32, and Apple9/Apple10 dispatch telemetry."""
    fam = _detect_gpu_family_info()
    return {
        "mlx_version": str(getattr(mx, "__version__", "unknown")),
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "0"),
        "mlx_enable_tf32_source": "env" if _TF32_WAS_USER_SET else "default_0",
        "mlx_metal_no_nax": os.environ.get("MLX_METAL_NO_NAX", "0"),
        "device_name": fam["device_name"],
        "architecture": fam["architecture"],
        "macos_version": fam["macos_version"],
        "is_apple10": fam["is_apple10"],
        "nax_sdpa_eligible": fam["nax_sdpa_eligible"],
        "sdpa_slice_len": fam["sdpa_slice_len"],
        "sdpa_supports_force_fused": _SDPA_SUPPORTS_FORCE_FUSED,
    }


def _apply_recommended_wired_limit() -> int | None:
    """Safely raise MLX wired limit to `max_recommended_working_set_size`, returning prior limit."""
    set_wl = getattr(mx, "set_wired_limit", None)
    if not callable(set_wl) or not mx.metal.is_available():
        return None
    info = _get_mlx_device_info()
    max_ws = int(info.get("max_recommended_working_set_size", 0) or 0)
    if max_ws <= 0:
        return None
    try:
        prev = set_wl(max_ws)
        return int(prev) if prev is not None else 0
    except Exception:
        return None


def _restore_wired_limit(prev_limit: int | None) -> None:
    if prev_limit is None:
        return
    set_wl = getattr(mx, "set_wired_limit", None)
    if callable(set_wl) and mx.metal.is_available():
        try:
            set_wl(int(prev_limit))
        except Exception:
            pass


@contextmanager
def wired_limit_scope() -> Iterator[int | None]:
    """Context manager setting MLX wired limit safely and restoring the previous limit on exit."""
    prev = _apply_recommended_wired_limit()
    try:
        yield prev
    finally:
        _restore_wired_limit(prev)


def _open_mapped_embedding(path_or_dir: Path) -> MappedEmbedding | None:
    if not path_or_dir.exists():
        return None
    for cand in _EMBED_TENSOR_CANDIDATES:
        try:
            return MappedEmbedding.from_checkpoint(path_or_dir, tensor_name=cand)
        except (KeyError, FileNotFoundError):
            continue
    return None


# ---------------------------------------------------------------------------
# MIT-adapted GatedDeltaNet kernels (from mlx-lm 0.32.2 gated_delta.py)
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    a_f32 = a.astype(mx.float32)
    dt_f32 = dt_bias.astype(mx.float32)
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a_f32 + dt_f32))


def normalize_qk(
    q: mx.array,
    k: mx.array,
    *,
    inv_scale: float,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """L2-normalize q/k in ``float32`` to prevent FP16 subnormal ``rms_eps`` underflow and BF16 rounding."""
    rms_eps = eps * (inv_scale * inv_scale)
    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    q_scale = mx.array(inv_scale * inv_scale, dtype=mx.float32)
    k_scale = mx.array(inv_scale, dtype=mx.float32)
    q_norm = (q_scale * mx.fast.rms_norm(q_f32, None, rms_eps)).astype(q.dtype)
    k_norm = (k_scale * mx.fast.rms_norm(k_f32, None, rms_eps)).astype(k.dtype)
    return q_norm, k_norm


def _make_gated_delta_kernel(has_mask: bool = False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * static_cast<float>(g_[hv_idx]);
              kv_mem += state[i] * static_cast<float>(k_[s_idx]);
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
              out += state[i] * static_cast<float>(q_[s_idx]);
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else if (thread_index_in_simdgroup == 0) {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"pplx_gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_packed_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        static_assert(Dk >= 16 && (Dk % (lanes_per_row * 4)) == 0, "Dk must be a positive multiple of 16");
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;
        static_assert(partials_per_lane >= 1, "partials_per_lane must be >= 1");

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          float part[partials_per_lane];
          #pragma unroll
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          #pragma unroll
          for (int stride = 1; stride < partials_per_lane; stride <<= 1) {
            #pragma unroll
            for (int p = 0; p + stride < partials_per_lane; p += (stride << 1)) {
              part[p] += part[p + stride];
            }
          }
          float kv_mem = part[0];
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          #pragma unroll
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          #pragma unroll
          for (int stride = 1; stride < partials_per_lane; stride <<= 1) {
            #pragma unroll
            for (int p = 0; p + stride < partials_per_lane; p += (stride << 1)) {
              part[p] += part[p + stride];
            }
          }
          float out = part[0];
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_GATED_DELTA_KERNEL = _make_gated_delta_kernel(has_mask=False)
_GATED_DELTA_KERNEL_MASKED = _make_gated_delta_kernel(has_mask=True)
_GATED_DELTA_KERNEL_PACKED = _make_gated_delta_packed_kernel()


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    old_state = state
    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    v_f32 = v.astype(mx.float32)
    g_f32 = g.astype(mx.float32)
    beta_f32 = beta.astype(mx.float32)

    state = state * g_f32[..., None, None]
    kv_mem = (state * k_f32[..., None, :]).sum(axis=-1)
    delta = (v_f32 - kv_mem) * beta_f32[..., None]
    state = state + k_f32[..., None, :] * delta[..., None]
    y = (state * q_f32[..., None, :]).sum(axis=-1)

    if mask is not None:
        m = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(m, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    use_kernel: bool = True,
    state_fp32: bool = True,
) -> tuple[mx.array, mx.array]:
    """Run GatedDeltaNet recurrence keeping ``q, k, v`` in activation dtype and ``state`` in FP32."""
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32 if state_fp32 else q.dtype)
    elif state_fp32 and state.dtype != mx.float32:
        state = state.astype(mx.float32)

    if (
        not use_kernel
        or _GATED_DELTA_KERNEL is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state.astype(mx.float32), mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    packed_eligible = (
        mask is None
        and g.ndim == 3
        and Dk >= 32
        and Dk % 32 == 0
        and Dv % 8 == 0
        and _GATED_DELTA_KERNEL_PACKED is not None
        and _ENABLE_GDN_PACKED
    )
    if packed_eligible:
        kernel = _GATED_DELTA_KERNEL_PACKED
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv // 8, B * Hv)
        threadgroup = (32, 2, 1)
    elif mask is not None and _GATED_DELTA_KERNEL_MASKED is not None:
        kernel = _GATED_DELTA_KERNEL_MASKED
        inputs = [q, k, v, g, beta, state, T, mask]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    else:
        kernel = _GATED_DELTA_KERNEL
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)

    # Intentional FP32 recurrent state_out (StT) preserves accuracy across >2048 chunk handoffs while y stays InT.
    out_y, out_state = kernel(  # guard: allow-fp32
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )
    return out_y, out_state


# ---------------------------------------------------------------------------
# Hybrid Qwen3.5 text backbone + contextual head
# ---------------------------------------------------------------------------


class PplxRMSNorm(nn.Module):
    """RMSNorm evaluated in float32 to prevent FP16 residual overflow and BF16 weight rounding."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        w_f32 = self.weight if self.weight.dtype == mx.float32 else self.weight.astype(mx.float32)
        out_f32 = mx.fast.rms_norm(x.astype(mx.float32), w_f32, self.eps)
        return out_f32.astype(x.dtype)


class PplxGatedRMSNorm(nn.Module):
    """Unshifted RMSNormGated computed in float32 before weight multiply."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        g_f32 = gate.astype(mx.float32)
        w_f32 = self.weight.astype(mx.float32)
        normed_f32 = mx.fast.rms_norm(x_f32, w_f32, self.eps)
        out_f32 = normed_f32 * nn.silu(g_f32)
        return out_f32.astype(x.dtype)


class PplxAttention(nn.Module):
    """Bidirectional grouped-query attention with output gating, partial RoPE, and Apple9/Apple10 dispatch."""

    def __init__(
        self,
        cfg: PplxConfig,
        *,
        gpu_family: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_dim
        self.rope_theta = float(cfg.rope_theta)
        self.scale = float(cfg.head_dim) ** -0.5

        fam = _detect_gpu_family_info() if gpu_family is None else dict(gpu_family)
        self.is_apple10 = bool(fam.get("is_apple10", False))
        self.nax_sdpa_eligible = bool(fam.get("nax_sdpa_eligible", False))
        self.sdpa_slice_len = int(fam.get("sdpa_slice_len", _DEFAULT_SDPA_SLICE_LEN))
        self._supports_force_fused = _SDPA_SUPPORTS_FORCE_FUSED

        self.q_proj_dim = 2 * self.num_heads * self.head_dim
        self.kv_proj_dim = self.num_kv_heads * self.head_dim
        total_qkv_dim = self.q_proj_dim + 2 * self.kv_proj_dim

        self.qkv_proj = nn.Linear(cfg.hidden_size, total_qkv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = PplxRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = PplxRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def _run_sdpa(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | str | None,
    ) -> mx.array:
        if isinstance(mask, mx.array) and mask.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise TypeError(
                f"SDPA mask must be typed additive float (float16/bfloat16) or None/'causal', got {mask.dtype}"
            )
        # On M5 (Apple10) macOS 26.2+, MLX 0.32.3+ auto-selects fused D256 NAX at Q>=1024;
        # for 384 <= Q < 1024 FP16 with additive mask, force_fused=True wins on M5 (never M4 Pro).
        q_len = int(q.shape[2])
        use_force_fused = (
            self.nax_sdpa_eligible
            and self._supports_force_fused
            and self.head_dim == 256
            and q.dtype == mx.float16
            and isinstance(mask, mx.array)
            and 384 <= q_len < _M5_SDPA_SLICE_LEN
        )
        if use_force_fused:
            return mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                mask=mask,
                force_fused=True,
            )
        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )

    def __call__(
        self,
        x: mx.array,
        sdpa_mask: mx.array | str | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q_and_gate, keys, values = mx.split(
            qkv,
            [self.q_proj_dim, self.q_proj_dim + self.kv_proj_dim],
            axis=-1,
        )
        q_and_gate = q_and_gate.reshape(B, S, self.num_heads, 2 * self.head_dim)
        queries, gate = mx.split(q_and_gate, 2, axis=-1)
        gate = gate.reshape(B, S, self.num_heads * self.head_dim)

        queries = self.q_norm(queries).swapaxes(1, 2)
        keys = self.k_norm(keys.reshape(B, S, self.num_kv_heads, self.head_dim)).swapaxes(1, 2)
        values = values.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

        queries = mx.fast.rope(
            queries,
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )
        keys = mx.fast.rope(
            keys,
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=0,
        )

        if S <= _STREAMED_SEQ_THRESHOLD or sdpa_mask == "causal":
            attn_out = self._run_sdpa(queries, keys, values, sdpa_mask)
        else:
            # Honor module-level monkeypatches in unit tests while defaulting to 1024 on M5 Apple10 / 512 on M4.
            slice_len = _SDPA_SLICE_LEN if _SDPA_SLICE_LEN != _DEFAULT_SDPA_SLICE_LEN else self.sdpa_slice_len
            min_tail = _SDPA_MIN_TAIL if _SDPA_MIN_TAIL != _DEFAULT_SDPA_SLICE_LEN else slice_len
            slices = tile_aligned_slices(S, slice_len=slice_len, min_tail=min_tail)
            parts: list[mx.array] = []
            for s0, s1 in slices:
                if isinstance(sdpa_mask, mx.array) and sdpa_mask.ndim == 4 and sdpa_mask.shape[2] == S:
                    m_slice: mx.array | str | None = sdpa_mask[:, :, s0:s1, :]
                else:
                    m_slice = sdpa_mask
                part = self._run_sdpa(queries[:, :, s0:s1, :], keys, values, m_slice)
                mx.async_eval(part)
                parts.append(part)
            attn_out = mx.concatenate(parts, axis=2)

        attn_flat = attn_out.swapaxes(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        gated = attn_flat * mx.sigmoid(gate)
        return self.o_proj(gated)


class PplxGatedDeltaNet(nn.Module):
    """Linear-attention GatedDeltaNet layer with causal Conv1d and FP32 state-carried streaming."""

    def __init__(self, cfg: PplxConfig) -> None:
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.inv_scale = float(self.head_k_dim) ** -0.5

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,), dtype=mx.float32)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=mx.float32)
        self.norm = PplxGatedRMSNorm(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.use_kernel = True

    def _run_step(
        self,
        x_slice: mx.array,
        a_slice: mx.array,
        b_slice: mx.array,
        conv_state: mx.array,
        rec_state: mx.array,
        mask_slice: mx.array | None,
        *,
        keep_conv_state_fp32: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        B, S_sub, _ = x_slice.shape

        mixed_qkv = self.in_proj_qkv(x_slice)
        z = self.in_proj_z(x_slice).reshape(B, S_sub, self.num_v_heads, self.head_v_dim)

        conv_prefix = conv_state if conv_state.dtype == mixed_qkv.dtype else conv_state.astype(mixed_qkv.dtype)
        conv_in = mx.concatenate([conv_prefix, mixed_qkv], axis=1)
        tail = conv_in[:, -(self.conv_kernel_size - 1) :, :]
        new_conv_state = tail.astype(mx.float32) if keep_conv_state_fp32 else tail
        conv_out = nn.silu(self.conv1d(conv_in))

        q_flat, k_flat, v_flat = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q_flat.reshape(B, S_sub, self.num_k_heads, self.head_k_dim)
        k = k_flat.reshape(B, S_sub, self.num_k_heads, self.head_k_dim)
        v = v_flat.reshape(B, S_sub, self.num_v_heads, self.head_v_dim)

        q, k = normalize_qk(q, k, inv_scale=self.inv_scale, eps=1e-6)
        out, new_rec_state = gated_delta_update(
            q,
            k,
            v,
            a_slice,
            b_slice,
            self.A_log,
            self.dt_bias,
            state=rec_state,
            mask=mask_slice,
            use_kernel=self.use_kernel,
            state_fp32=True,
        )
        normed = self.norm(out, z).reshape(B, S_sub, self.value_dim)
        return self.out_proj(normed), new_conv_state, new_rec_state

    def __call__(
        self,
        x: mx.array,
        ssm_mask: mx.array | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        if ssm_mask is not None:
            zero_val = mx.array(0.0, dtype=x.dtype)
            x = mx.where(ssm_mask[..., None], x, zero_val)

        # Narrow projections (N=16) depend on M-tile rounding, so evaluate once whole before slicing.
        a_all = self.in_proj_a(x)
        b_all = self.in_proj_b(x)
        # Keep rec_state in FP32 so S > 2048 chunk handoffs never suffer FP16/BF16 state truncation.
        rec_state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
        if S <= _STREAMED_SEQ_THRESHOLD:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
            out, _, _ = self._run_step(
                x, a_all, b_all, conv_state, rec_state, ssm_mask, keep_conv_state_fp32=False
            )
            return out

        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=mx.float32)
        mx.async_eval(a_all, b_all)
        outs: list[mx.array] = []
        for s0, s1 in tile_aligned_slices(S, slice_len=_GDN_SLICE_LEN, min_tail=_GDN_SLICE_LEN):
            mask_slice = None if ssm_mask is None else ssm_mask[:, s0:s1]
            y_slice, conv_state, rec_state = self._run_step(
                x[:, s0:s1, :],
                a_all[:, s0:s1, :],
                b_all[:, s0:s1, :],
                conv_state,
                rec_state,
                mask_slice,
                keep_conv_state_fp32=True,
            )
            mx.async_eval(y_slice, conv_state, rec_state)
            outs.append(y_slice)
        return mx.concatenate(outs, axis=1)


class PplxMLP(nn.Module):
    """Fused Gate+Up SwiGLU MLP with async-pipelined sequence chunking above 2048 tokens."""

    def __init__(self, cfg: PplxConfig) -> None:
        super().__init__()
        self.intermediate_size = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def _run_slice(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, [self.intermediate_size], axis=-1)
        return self.down_proj(nn.silu(gate) * up)

    def __call__(self, x: mx.array) -> mx.array:
        S = x.shape[1]
        if S <= _STREAMED_SEQ_THRESHOLD:
            return self._run_slice(x)
        outs: list[mx.array] = []
        for s0, s1 in tile_aligned_slices(S, slice_len=_GDN_SLICE_LEN, min_tail=_GDN_SLICE_LEN):
            part = self._run_slice(x[:, s0:s1, :])
            mx.async_eval(part)
            outs.append(part)
        return mx.concatenate(outs, axis=1)


class PplxDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: PplxConfig,
        is_linear: bool,
        *,
        gpu_family: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.is_linear = is_linear
        self.input_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = PplxMLP(cfg)
        if is_linear:
            self.linear_attn = PplxGatedDeltaNet(cfg)
        else:
            self.self_attn = PplxAttention(cfg, gpu_family=gpu_family)

    def __call__(
        self,
        h: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        sdpa_mask: mx.array | str | None = None,
    ) -> mx.array:
        x_in = self.input_layernorm(h)
        if self.is_linear:
            r = self.linear_attn(x_in, ssm_mask=ssm_mask)
        else:
            r = self.self_attn(x_in, sdpa_mask=sdpa_mask)
        h = h + r
        return h + self.mlp(self.post_attention_layernorm(h))


class PplxModel(nn.Module):
    def __init__(
        self,
        cfg: PplxConfig,
        *,
        has_embed_module: bool = True,
        gpu_family: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = cfg
        self.gpu_family = _detect_gpu_family_info() if gpu_family is None else dict(gpu_family)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) if has_embed_module else None
        self.layers = [
            PplxDecoderLayer(
                cfg,
                is_linear=(cfg.layer_types[i] == "linear_attention"),
                gpu_family=self.gpu_family,
            )
            for i in range(cfg.num_hidden_layers)
        ]
        self.norm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.contextual_projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def forward_hidden(
        self,
        hidden: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        sdpa_mask: mx.array | str | None = None,
        return_trace: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        trace: list[mx.array] = []
        if return_trace:
            trace.append(hidden)
        h = hidden
        is_streamed = hidden.shape[1] > _STREAMED_SEQ_THRESHOLD
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)
            if is_streamed:
                # Pipeline asynchronously per layer and only bound peak memory every 4 layers.
                if (layer_idx + 1) % _STREAMED_SYNC_STRIDE == 0:
                    mx.eval(h)  # guard: allow-eval
                else:
                    mx.async_eval(h)
            if return_trace:
                trace.append(h)
        h = self.norm(h)
        if return_trace:
            trace.append(h)
            return h, trace
        return h


def _is_zero_centered_norm_key(key: str) -> bool:
    if key == "norm.weight":
        return True
    if key.endswith("linear_attn.norm.weight"):
        return False
    return any(key.endswith(sfx) for sfx in _ZERO_CENTERED_NORM_SUFFIXES)


def _strip_language_prefix(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    if key.startswith("visual.") or key.startswith("model.visual."):
        return None
    return key


def _canonicalize_weights(
    raw_weights: Mapping[str, mx.array],
    *,
    num_layers: int,
    act_dtype: Any,
    include_embed: bool,
    norms_already_shifted: bool = False,
) -> tuple[list[tuple[str, mx.array]], mx.array]:
    normalized: dict[str, mx.array] = {}
    head_w: mx.array | None = None

    for orig_key, val in raw_weights.items():
        if orig_key == "contextual_projection.weight" or orig_key.endswith(".contextual_projection.weight"):
            head_w = val.astype(mx.float32)
            continue
        if orig_key.startswith("__"):
            continue
        key = _strip_language_prefix(orig_key)
        if key is None:
            continue
        if key == "embed_tokens.weight" and not include_embed:
            continue

        if "conv1d.weight" in key and val.ndim == 3 and val.shape[-1] != 1:
            val = val.moveaxis(2, 1)

        is_zero_centered_norm = _is_zero_centered_norm_key(key)
        if not norms_already_shifted and is_zero_centered_norm:
            val = val.astype(mx.float32) + 1.0

        # Keep zero-centered norm weights, GatedRMSNorm weights, and GDN decay params in FP32.
        if (
            is_zero_centered_norm
            or key.endswith(".A_log")
            or key.endswith(".dt_bias")
            or key.endswith("linear_attn.norm.weight")
        ):
            val = val.astype(mx.float32)
        else:
            val = val.astype(act_dtype)

        normalized[key] = val

    # Fuse q_proj + k_proj + v_proj -> qkv_proj and gate_proj + up_proj -> gate_up_proj idempotently.
    for i in range(num_layers):
        attn_prefix = f"layers.{i}.self_attn"
        qkvs = (f"{attn_prefix}.q_proj.weight", f"{attn_prefix}.k_proj.weight", f"{attn_prefix}.v_proj.weight")
        if all(k in normalized for k in qkvs):
            normalized[f"{attn_prefix}.qkv_proj.weight"] = mx.concatenate([normalized.pop(k) for k in qkvs], axis=0)

        mlp_prefix = f"layers.{i}.mlp"
        gu = (f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight")
        if all(k in normalized for k in gu):
            normalized[f"{mlp_prefix}.gate_up_proj.weight"] = mx.concatenate([normalized.pop(k) for k in gu], axis=0)

    if head_w is None:
        raise FileNotFoundError("Missing contextual_projection.weight in loaded weights")

    return sorted(normalized.items()), head_w


def _load_safetensors_dir(model_dir: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    for st_file in st_files:
        weights.update(mx.load(str(st_file)))
    return weights


def _find_local_hf_snapshot(repo_id: str) -> Path | None:
    hf_home = os.environ.get("HF_HOME")
    default_cache = Path.home() / ".cache" / "huggingface"
    hub_dir = Path(os.environ.get("HF_HUB_CACHE") or (Path(hf_home) if hf_home else default_cache) / "hub")
    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        cand = snapshots_dir / ref_main.read_text(encoding="utf-8").strip()
        if (cand / "config.json").exists():
            return cand
    for cand in sorted(snapshots_dir.iterdir()):
        if cand.is_dir() and (cand / "config.json").exists() and (cand / "tokenizer.json").exists():
            return cand
    return None


def _find_checkpoint_source(
    paths: RepoPaths,
    *,
    model_dir: str | Path | None,
    local_files_only: bool,
) -> tuple[str, Path, Path | None]:
    """Return ``(kind, main_path, optional_embed_safetensors)``."""
    if model_dir is not None:
        md = Path(model_dir).expanduser().resolve()
        return "dir", md, None
    if paths.weights_npz.exists():
        embed_st = paths.embed_tokens_safetensors if paths.embed_tokens_safetensors.exists() else None
        return "npz", paths.weights_npz, embed_st
    if (paths.model_dir / "config.json").exists():
        return "dir", paths.model_dir, None
    local_snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if local_snap is not None:
        return "dir", local_snap, None
    hf_dir = resolve_hf_model_dir(None, DEFAULT_MODEL_NAME, local_files_only=local_files_only)
    return "dir", hf_dir, None


def _build_sdpa_mask(attention_mask: np.ndarray, *, dtype_name: str) -> mx.array:
    """Build a typed additive 4D SDPA mask (`0.0` valid, `-65504.0` for FP16, `-inf`/`-1e9` for BF16/FP32)."""
    mask_bool = np.asarray(attention_mask, dtype=np.bool_)
    valid_4d = mask_bool[:, None, None, :]
    if dtype_name == "float16":
        arr = np.where(valid_4d, np.float16(0.0), np.float16(-65504.0))
        return mx.array(arr, dtype=mx.float16)
    # Use -inf on rows that have at least one valid token, and finite -1e9 on all-pad rows to prevent NaN softmax.
    row_has_valid = np.any(mask_bool, axis=1)[:, None, None, None]
    neg_vals = np.where(row_has_valid, np.float32(-np.inf), np.float32(-1e9))
    arr = np.where(valid_4d, np.float32(0.0), neg_vals)
    if dtype_name == "float32":
        return mx.array(arr, dtype=mx.float32)
    return mx.array(arr, dtype=mx.bfloat16)


def _quantize_int8_np(projected: np.ndarray) -> np.ndarray:
    scaled = np.tanh(projected) * 127.0
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


class FastPplxGPU:
    """High-throughput MLX Metal GPU backend for `pplx-embed-v2-context-0.6b`."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        model_dir: str | Path | None = None,
        dtype: str = "float16",
        use_mapped_embedding: bool = True,
        cache_limit_mb: int = 512,
        local_files_only: bool = True,
        max_batch_size: int = 32,
        max_seq_len: int = 32768,
    ) -> None:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype {dtype!r}; expected 'float16' or 'bfloat16'")

        self.paths = resolve_paths(home)
        self.dtype_name = dtype
        self.act_dtype = mx.float16 if dtype == "float16" else mx.bfloat16
        self.use_mapped_embedding = bool(use_mapped_embedding)
        self.cache_limit_mb = int(cache_limit_mb)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.runtime_telemetry = get_mlx_runtime_telemetry()

        set_mlx_cache_limit(self.cache_limit_mb)
        self._prev_wired_limit: int | None = _apply_recommended_wired_limit()

        try:
            kind, source_path, embed_st = _find_checkpoint_source(
                self.paths,
                model_dir=model_dir,
                local_files_only=local_files_only,
            )

            # Resolve config & tokenizer
            cfg_candidates = [
                self.paths.model_dir / "config.json",
                self.paths.tokenizer_dir / "config.json",
            ]
            if kind == "dir":
                cfg_candidates.insert(0, source_path / "config.json")
            cfg_file = next((c for c in cfg_candidates if c.exists()), None)
            self.config = PplxConfig.from_json(cfg_file) if cfg_file is not None else DEFAULT_PPLX_CONFIG

            tok_candidates = [
                source_path / "tokenizer.json" if kind == "dir" else self.paths.tokenizer_json,
                self.paths.tokenizer_json,
                self.paths.model_dir / "tokenizer.json",
            ]
            tok_file = next((c for c in tok_candidates if c.exists()), None)
            if tok_file is None:
                raise FileNotFoundError(f"Could not find tokenizer.json in {tok_candidates}")
            self.tokenizer = ContextualTokenizer(tok_file, self.config, default_max_length=self.max_seq_len)

            self.dim = self.config.embedding_dim
            self.max_dim = self.config.embedding_dim
            self.supports_dimensions = True
            self.model_name = DEFAULT_MODEL_NAME

            self._mapped_embed: MappedEmbedding | None = None
            if self.use_mapped_embedding:
                embed_candidates = [
                    p
                    for p in (
                        embed_st,
                        source_path if kind == "dir" else None,
                        self.paths.embed_tokens_safetensors,
                        self.paths.model_dir,
                    )
                    if p is not None
                ]
                for cand_path in embed_candidates:
                    self._mapped_embed = _open_mapped_embedding(cand_path)
                    if self._mapped_embed is not None:
                        break

            using_mapped = self._mapped_embed is not None

            self.model = PplxModel(
                self.config,
                has_embed_module=(not using_mapped),
                gpu_family=_detect_gpu_family_info(),
            )

            if kind == "dir":
                raw_weights = _load_safetensors_dir(source_path)
                norms_shifted = False
            else:
                raw_weights = dict(mx.load(str(source_path)))
                flag = raw_weights.pop("__norms_shifted__", None)
                norms_shifted = bool(int(flag.item())) if flag is not None else False
                if not using_mapped and not any(
                    _strip_language_prefix(k) == "embed_tokens.weight" or k in _EMBED_TENSOR_CANDIDATES
                    for k in raw_weights
                ):
                    embed_cand = next(
                        (
                            p
                            for p in (
                                embed_st,
                                self.paths.embed_tokens_safetensors,
                                self.paths.model_dir / "embed_tokens.safetensors",
                            )
                            if p is not None and p.is_file()
                        ),
                        None,
                    )
                    if embed_cand is not None:
                        loaded_embed = mx.load(str(embed_cand))
                        for cand in (*_EMBED_TENSOR_CANDIDATES, *loaded_embed.keys()):
                            if cand in loaded_embed and (
                                cand in _EMBED_TENSOR_CANDIDATES
                                or _strip_language_prefix(cand) == "embed_tokens.weight"
                            ):
                                raw_weights["embed_tokens.weight"] = loaded_embed[cand]
                                break

            backbone_items, head_w = _canonicalize_weights(
                raw_weights,
                num_layers=self.config.num_hidden_layers,
                act_dtype=self.act_dtype,
                include_embed=(not using_mapped),
                norms_already_shifted=norms_shifted,
            )
            all_items = [*backbone_items, ("contextual_projection.weight", head_w)]
            self.model.load_weights(all_items, strict=True)
            mx.eval(self.model.parameters())
        except Exception:
            self.close()
            raise

    def _embed_inputs(self, input_ids: np.ndarray) -> mx.array:
        if self._mapped_embed is not None:
            # Always gather as float16 (2 B/el) and cast FP16 -> BF16 on GPU in MLX when needed.
            gathered = self._mapped_embed.gather(input_ids, out_dtype="float16")
            arr = mx.array(gathered, dtype=mx.float16)
            return arr if self.act_dtype == mx.float16 else arr.astype(self.act_dtype)
        assert self.model.embed_tokens is not None
        return self.model.embed_tokens(mx.array(input_ids, dtype=mx.int32))

    def forward_hidden(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        *,
        return_trace: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        ids_np = np.asarray(input_ids, dtype=np.int32)
        if ids_np.ndim != 2:
            raise ValueError(f"Expected 2D input_ids, got shape {ids_np.shape}")
        if attention_mask is None:
            has_padding = False
            ssm_mask = None
            sdpa_mask = None
        else:
            mask_np = np.asarray(attention_mask, dtype=np.bool_)
            has_padding = not bool(np.all(mask_np))
            ssm_mask = mx.array(mask_np, dtype=mx.bool_) if has_padding else None
            sdpa_mask = _build_sdpa_mask(mask_np, dtype_name=self.dtype_name) if has_padding else None

        hidden = self._embed_inputs(ids_np)
        return self.model.forward_hidden(
            hidden,
            ssm_mask=ssm_mask,
            sdpa_mask=sdpa_mask,
            return_trace=return_trace,
        )

    def _project_mean_pool(self, hidden: mx.array, attention_mask: np.ndarray) -> np.ndarray:
        hidden_f32 = hidden.astype(mx.float32)
        mask_f32 = mx.array(attention_mask.astype(np.float32))[..., None]
        denom = mx.maximum(mx.sum(mask_f32, axis=1), mx.array(1.0, dtype=mx.float32))
        pooled = mx.sum(hidden_f32 * mask_f32, axis=1) / denom
        projected = self.model.contextual_projection(pooled)
        mx.eval(projected)
        out = np.array(projected, dtype=np.float32, copy=False)
        if hidden.shape[1] > _STREAMED_SEQ_THRESHOLD:
            clear_mlx_arena(256)
        return out

    def _project_span_pool(self, hidden: mx.array, chunk_token_spans: np.ndarray) -> np.ndarray:
        hidden_f32 = hidden.astype(mx.float32)
        B, _, H = hidden_f32.shape
        prefix = mx.concatenate([mx.zeros((B, 1, H), dtype=mx.float32), mx.cumsum(hidden_f32, axis=1)], axis=1)
        spans_mx = mx.array(chunk_token_spans, dtype=mx.int32)
        starts = spans_mx[..., 0]
        ends = spans_mx[..., 1]
        rows = mx.arange(B, dtype=mx.int32)[:, None]
        diff = ends - starts
        valid = diff > 0
        counts = mx.maximum(diff, 1)[..., None].astype(mx.float32)
        pooled = (prefix[rows, ends] - prefix[rows, starts]) / counts
        pooled = mx.where(valid[..., None], pooled, mx.array(0.0, dtype=mx.float32))
        projected = self.model.contextual_projection(pooled)
        projected = mx.where(valid[..., None], projected, mx.array(0.0, dtype=mx.float32))
        mx.eval(projected)
        out = np.array(projected, dtype=np.float32, copy=False)
        if hidden.shape[1] > _STREAMED_SEQ_THRESHOLD:
            clear_mlx_arena(256)
        return out

    def _format_output(
        self,
        projected: np.ndarray,
        *,
        dimensions: int | None,
        output_dtype: str,
        normalize: bool,
    ) -> np.ndarray:
        if output_dtype not in ("float32", "float16", "int8"):
            raise ValueError(f"Unsupported output_dtype {output_dtype!r}")
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.dim)
        if projected.shape[0] == 0:
            np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
            return np.zeros((0, target_dim), dtype=np_dt)
        sliced = projected[:, :target_dim]
        if output_dtype == "int8":
            return _quantize_int8_np(sliced)
        if normalize:
            vecs = truncate_and_normalize(projected, target_dim)
            # truncate_and_normalize returns unnormalized vectors when target_dim == width.
            if target_dim == projected.shape[-1]:
                norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
                vecs = np.ascontiguousarray(vecs / np.maximum(norms, 1e-12), dtype=np.float32)
        else:
            vecs = np.asarray(sliced, dtype=np.float32)
        if output_dtype == "float16":
            return vecs.astype(np.float16)
        return vecs.astype(np.float32, copy=False)

    def encode_tokens(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        batch_size: int = 16,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        n = len(token_lists)
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.dim)
        if n == 0:
            np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
            return np.zeros((0, target_dim), dtype=np_dt)

        lengths = np.array([len(row) for row in token_lists], dtype=np.int64)
        order = np.argsort(lengths, kind="stable")
        bs = max(1, min(int(batch_size), self.max_batch_size))
        proj_all = np.zeros((n, self.dim), dtype=np.float32)

        for start in range(0, n, bs):
            idxs = order[start : start + bs]
            rows = [token_lists[int(i)] for i in idxs]
            input_ids, attention_mask = self.tokenizer.pad(rows)
            hidden = self.forward_hidden(input_ids, attention_mask)
            proj_batch = self._project_mean_pool(hidden, attention_mask)
            proj_all[idxs] = proj_batch

        clear_mlx_arena(256)
        return self._format_output(
            proj_all,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
        )

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        if is_query:
            token_lists = [self.tokenizer.encode_query(t, max_length=eff_max) for t in items]
        else:
            token_lists = [self.tokenizer.encode_document(t, max_length=eff_max) for t in items]
        out = self.encode_tokens(
            token_lists,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
        )
        return out

    def encode_queries(
        self,
        queries: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=True,
        )

    def encode_documents(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> np.ndarray:
        return self.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=False,
        )

    def encode_contextual_chunks(
        self,
        documents: Sequence[Sequence[str]],
        *,
        batch_size: int = 8,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> list[np.ndarray]:
        if not documents:
            return []
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        encoded_docs = [
            self.tokenizer.encode_contextual_document(chunks, return_spans=True, max_length=eff_max)
            for chunks in documents
        ]
        lengths = np.array([len(item[0]) for item in encoded_docs], dtype=np.int64)
        order = np.argsort(lengths, kind="stable")
        bs = max(1, min(int(batch_size), self.max_batch_size))
        results: list[np.ndarray | None] = [None] * len(documents)

        for start in range(0, len(documents), bs):
            idxs = order[start : start + bs]
            sub_docs = [documents[int(i)] for i in idxs]
            rows = [encoded_docs[int(i)][0] for i in idxs]
            spans_list = [encoded_docs[int(i)][1] for i in idxs]
            input_ids, attention_mask = self.tokenizer.pad(rows)
            max_c = max(1, *(len(s) for s in spans_list))
            span_arr = np.zeros((len(idxs), max_c, 2), dtype=np.int32)
            for row_i, sp in enumerate(spans_list):
                for col_j, (t0, t1) in enumerate(sp):
                    span_arr[row_i, col_j] = (t0, t1)

            hidden = self.forward_hidden(input_ids, attention_mask)
            proj = self._project_span_pool(hidden, span_arr)
            for row_i, orig_idx in enumerate(idxs):
                n_chunks = len(sub_docs[row_i])
                doc_proj = proj[row_i, :n_chunks]
                results[int(orig_idx)] = self._format_output(
                    doc_proj,
                    dimensions=dimensions,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )

        clear_mlx_arena(256)
        return [r for r in results if r is not None]

    def close(self) -> None:
        if getattr(self, "_mapped_embed", None) is not None:
            self._mapped_embed.close()
            self._mapped_embed = None
        prev_wl = getattr(self, "_prev_wired_limit", None)
        if prev_wl is not None:
            self._prev_wired_limit = None
            _restore_wired_limit(prev_wl)
        clear_mlx_arena(256)

    def __enter__(self) -> FastPplxGPU:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "FastPplxGPU",
    "MappedEmbedding",
    "PplxAttention",
    "PplxDecoderLayer",
    "PplxGatedDeltaNet",
    "PplxGatedRMSNorm",
    "PplxMLP",
    "PplxModel",
    "PplxRMSNorm",
    "compute_g",
    "gated_delta_ops",
    "gated_delta_update",
    "get_mlx_runtime_telemetry",
    "normalize_qk",
    "set_mlx_cache_limit",
    "tile_aligned_slices",
    "wired_limit_scope",
]
