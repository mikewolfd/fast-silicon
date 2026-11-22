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
from functools import lru_cache, partial
import inspect
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterator, Literal, Mapping, Sequence

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
_GDN_TILE_ROWS_DEFAULT = 16
_GDN_TILE_ROWS = max(
    8,
    min(24, int(os.environ.get("FAST_PPLX_GDN_TILE_ROWS", str(_GDN_TILE_ROWS_DEFAULT)))),
)
_USE_MX_FAST_GDN = (
    hasattr(mx.fast, "gated_delta_update")
    and os.environ.get("FAST_PPLX_DISABLE_MX_FAST_GDN", "0") != "1"
)
_CACHED_IS_APPLE10_HOST: bool | None = None


def _is_apple10_gdn_host() -> bool:
    global _CACHED_IS_APPLE10_HOST
    if _CACHED_IS_APPLE10_HOST is None:
        _CACHED_IS_APPLE10_HOST = bool(_detect_gpu_family_info().get("is_apple10", False))
    return _CACHED_IS_APPLE10_HOST
_ENABLE_VARLEN_PACK = os.environ.get("FAST_PPLX_VARLEN_PACK", "1") != "0"
_ARCH_GEN_RE = re.compile(r"g(\d+)", re.IGNORECASE)
_VALID_QUANTIZE_WEIGHTS: tuple[str, ...] = ("none", "w8")

_VALID_STATE_DTYPES = ("float32", "bfloat16", "int8")
_STATE_DTYPE_ALIASES = {
    "fp32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "i8": "int8",
    "int8": "int8",
}


def normalize_state_dtype(state_dtype: str | None = None, *, state_fp32: bool = True) -> str:
    """Resolve canonical GDN recurrent state format (`'float32'`, `'bfloat16'`, or `'int8'`)."""
    if state_dtype is None:
        env_sd = os.environ.get("FAST_PPLX_STATE_DTYPE", "").strip()
        if env_sd:
            state_dtype = env_sd
        else:
            return "float32" if state_fp32 else "bfloat16"
    key = str(state_dtype).strip().lower()
    if key not in _STATE_DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported state_dtype {state_dtype!r}; expected one of {_VALID_STATE_DTYPES}"
        )
    return _STATE_DTYPE_ALIASES[key]


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
    tf32_env = os.environ.get("MLX_ENABLE_TF32")
    return {
        "mlx_version": str(getattr(mx, "__version__", "unknown")),
        "mlx_enable_tf32": tf32_env,
        "mlx_enable_tf32_source": "env" if tf32_env is not None else "mlx_default",
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


def _make_gated_delta_staged_kernel(*, segmented: bool = False, state_i8: bool = False):
    """Threadgroup-staged GdnDecodeShared kernel (128 threads, TG_DV=16, configurable TILE_ROWS).

    Supports FP32/BF16/FP16 dense recurrent state or Block-32 INT8 ``state_code_quad``
    compressed recurrent state, optional sequence segment flags (``0=pad/skip``, ``1=valid``,
    ``2=doc_start_reset``), ``#pragma clang fp reassociate(off)``, and Splash bitwise finite checks.
    """
    if not mx.metal.is_available():
        return None

    seg_decl = "threadgroup uint8_t sh_f[TILE_ROWS];" if segmented else ""
    flg_ptr = "auto flg_base = flags + b_idx * T;" if segmented else ""
    seg_load = (
        "if (int(tid) < tile_rows) { sh_f[tid] = flg_base[t_base + int(tid)]; }"
        if segmented
        else ""
    )
    seg_open = (
        """
            uint8_t flg = sh_f[r];
            if (flg == 0) {
              if (lane_in_row == 0 && rdv < Dv) {
                y_row[rdv] = static_cast<InT>(0);
              }
            } else {
              if (flg == 2) {
                #pragma unroll
                for (int pb = 0; pb < kQuads; ++pb) {
                  st[pb] = float4(0.0f);
                }
              }
        """
        if segmented
        else "{"
    )

    if state_i8:
        state_load_block = """
        if (rdv < Dv) {
          auto rc = state_codes_in + (n * Dv + rdv) * Dk;
          auto rs = state_scales_in + (n * Dv + rdv) * kQuads;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            bfloat sc_bf = rs[pb];
            ushort sc_bits = as_type<ushort>(sc_bf);
            bool sc_finite = (sc_bits & 0x7f80u) < 0x7f80u;
            int d0 = pb * 32 + int(lane_in_row) * 4;
            if (sc_finite) {
              float sc = float(sc_bf);
              st[pb] = float4(
                  static_cast<float>(rc[d0 + 0]) * sc,
                  static_cast<float>(rc[d0 + 1]) * sc,
                  static_cast<float>(rc[d0 + 2]) * sc,
                  static_cast<float>(rc[d0 + 3]) * sc);
            } else {
              st[pb] = float4(as_type<float>(0x7fc00000u));
            }
          }
        } else {
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            st[pb] = float4(0.0f);
          }
        }
        """
        state_store_block = """
        if (rdv < Dv) {
          auto wc = state_codes_out + (n * Dv + rdv) * Dk;
          auto ws = state_scales_out + (n * Dv + rdv) * kQuads;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            float4 v4 = st[pb];
            uint m0 = as_type<uint>(v4.x) & 0x7fffffffu;
            uint m1 = as_type<uint>(v4.y) & 0x7fffffffu;
            uint m2 = as_type<uint>(v4.z) & 0x7fffffffu;
            uint m3 = as_type<uint>(v4.w) & 0x7fffffffu;
            uint mag = max(max(m0, m1), max(m2, m3));
            mag = max(mag, simd_shuffle_xor(mag, 1));
            mag = max(mag, simd_shuffle_xor(mag, 2));
            mag = max(mag, simd_shuffle_xor(mag, 4));
            bfloat sc_bf = (mag < 0x7f800000u)
                ? bfloat(as_type<float>(mag) * (1.0f / 127.0f))
                : as_type<bfloat>(ushort(0x7fc0));
            if (lane_in_row == 0) {
              ws[pb] = sc_bf;
            }
            float sc_f = float(sc_bf);
            bool ok = (mag < 0x7f800000u) && (sc_f > 0.0f);
            int d0 = pb * 32 + int(lane_in_row) * 4;
            if (ok) {
              wc[d0 + 0] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.x, sc_f)), -127.0f, 127.0f));
              wc[d0 + 1] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.y, sc_f)), -127.0f, 127.0f));
              wc[d0 + 2] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.z, sc_f)), -127.0f, 127.0f));
              wc[d0 + 3] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v4.w, sc_f)), -127.0f, 127.0f));
            } else {
              wc[d0 + 0] = static_cast<int8_t>(0);
              wc[d0 + 1] = static_cast<int8_t>(0);
              wc[d0 + 2] = static_cast<int8_t>(0);
              wc[d0 + 3] = static_cast<int8_t>(0);
            }
          }
        }
        """
        input_names = ["q", "k", "v", "g", "beta", "state_codes_in", "state_scales_in", "T_arr"]
        output_names = ["y", "state_codes_out", "state_scales_out"]
        kname = f"pplx_gated_delta_staged{'_seg' if segmented else ''}_i8"
    else:
        state_load_block = """
        if (rdv < Dv) {
          auto rs = state_in + (n * Dv + rdv) * Dk;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            int d0 = pb * 32 + int(lane_in_row) * 4;
            st[pb] = float4(
                static_cast<float>(rs[d0 + 0]),
                static_cast<float>(rs[d0 + 1]),
                static_cast<float>(rs[d0 + 2]),
                static_cast<float>(rs[d0 + 3]));
          }
        } else {
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            st[pb] = float4(0.0f);
          }
        }
        """
        state_store_block = """
        if (rdv < Dv) {
          auto ws = state_out + (n * Dv + rdv) * Dk;
          #pragma unroll
          for (int pb = 0; pb < kQuads; ++pb) {
            int d0 = pb * 32 + int(lane_in_row) * 4;
            float4 v4 = st[pb];
            ws[d0 + 0] = static_cast<StT>(v4.x);
            ws[d0 + 1] = static_cast<StT>(v4.y);
            ws[d0 + 2] = static_cast<StT>(v4.z);
            ws[d0 + 3] = static_cast<StT>(v4.w);
          }
        }
        """
        input_names = ["q", "k", "v", "g", "beta", "state_in", "T_arr"]
        output_names = ["y", "state_out"]
        kname = f"pplx_gated_delta_staged{'_seg' if segmented else ''}"

    if segmented:
        input_names.append("flags")

    source = f"""
        #pragma clang fp reassociate(off)
        constexpr int kThreads = 128;
        constexpr int kLanesPerRow = 8;
        constexpr int kSgs = kThreads / 32;
        constexpr int kRowsPerSg = 32 / kLanesPerRow;
        static_assert(Dk >= 32 && (Dk % 32) == 0, "Dk must be a positive multiple of 32");
        constexpr int kQuads = Dk / 32;
        static_assert(TG_DV == kSgs * kRowsPerSg, "TG_DV must equal 16");

        int T = int(T_arr[0]);
        uint tid = thread_position_in_grid.x;
        uint row_tile = thread_position_in_grid.y;
        uint n = thread_position_in_grid.z;
        uint b_idx = n / Hv;
        uint hv_idx = n % Hv;
        uint hk_idx = hv_idx / (Hv / Hk);

        uint sg = tid >> 5;
        uint lane = tid & 31u;
        uint row_in_sg = lane / kLanesPerRow;
        uint lane_in_row = lane & (kLanesPerRow - 1u);

        threadgroup InT sh_q[TILE_ROWS * Dk];
        threadgroup InT sh_k[TILE_ROWS * Dk];
        threadgroup InT sh_v[TILE_ROWS * TG_DV];
        threadgroup float sh_g[TILE_ROWS];
        threadgroup float sh_b[TILE_ROWS];
        {seg_decl}

        auto q_base = q + (b_idx * T * Hk + hk_idx) * Dk;
        auto k_base = k + (b_idx * T * Hk + hk_idx) * Dk;
        auto v_base = v + (b_idx * T * Hv + hv_idx) * Dv;
        auto y_base = y + (b_idx * T * Hv + hv_idx) * Dv;
        auto g_base = g + b_idx * T * Hv + hv_idx;
        auto b_base = beta + b_idx * T * Hv + hv_idx;
        {flg_ptr}

        int ldv = int(sg) * kRowsPerSg + int(row_in_sg);
        int rdv = int(row_tile) * TG_DV + ldv;
        float4 st[kQuads];
        {state_load_block}

        int total_qk_stride = Hk * Dk;
        int total_vy_stride = Hv * Dv;
        int dv_tile_base = int(row_tile) * TG_DV;
        constexpr int kQkVecs = Dk / 4;
        constexpr int kVVecs = TG_DV / 4;

        for (int t_base = 0; t_base < T; t_base += TILE_ROWS) {{
          int tile_rows = min(TILE_ROWS, T - t_base);
          int total_qk4 = tile_rows * kQkVecs;
          for (int idx4 = int(tid); idx4 < total_qk4; idx4 += kThreads) {{
            int r = idx4 / kQkVecs;
            int d = (idx4 - r * kQkVecs) * 4;
            int src_off = (t_base + r) * total_qk_stride + d;
            int dst_off = r * Dk + d;
            sh_q[dst_off + 0] = q_base[src_off + 0];
            sh_q[dst_off + 1] = q_base[src_off + 1];
            sh_q[dst_off + 2] = q_base[src_off + 2];
            sh_q[dst_off + 3] = q_base[src_off + 3];
            sh_k[dst_off + 0] = k_base[src_off + 0];
            sh_k[dst_off + 1] = k_base[src_off + 1];
            sh_k[dst_off + 2] = k_base[src_off + 2];
            sh_k[dst_off + 3] = k_base[src_off + 3];
          }}
          int total_v4 = tile_rows * kVVecs;
          for (int idx4 = int(tid); idx4 < total_v4; idx4 += kThreads) {{
            int r = idx4 / kVVecs;
            int dv_off = (idx4 - r * kVVecs) * 4;
            int grdv = dv_tile_base + dv_off;
            int dst_off = r * TG_DV + dv_off;
            if (grdv + 3 < Dv) {{
              int src_off = (t_base + r) * total_vy_stride + grdv;
              sh_v[dst_off + 0] = v_base[src_off + 0];
              sh_v[dst_off + 1] = v_base[src_off + 1];
              sh_v[dst_off + 2] = v_base[src_off + 2];
              sh_v[dst_off + 3] = v_base[src_off + 3];
            }} else {{
              #pragma unroll
              for (int j = 0; j < 4; ++j) {{
                sh_v[dst_off + j] = (grdv + j < Dv)
                    ? v_base[(t_base + r) * total_vy_stride + grdv + j]
                    : static_cast<InT>(0);
              }}
            }}
          }}
          if (int(tid) < tile_rows) {{
            sh_g[tid] = static_cast<float>(g_base[(t_base + int(tid)) * Hv]);
            sh_b[tid] = static_cast<float>(b_base[(t_base + int(tid)) * Hv]);
          }}
          {seg_load}
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (int r = 0; r < tile_rows; ++r) {{
            auto y_row = y_base + (t_base + r) * total_vy_stride;
            {seg_open}
            float gt = sh_g[r];
            float bt = sh_b[r];
            const threadgroup InT* kr_ptr = sh_k + r * Dk;
            const threadgroup InT* qr_ptr = sh_q + r * Dk;
            float v_val = static_cast<float>(sh_v[r * TG_DV + ldv]);

            float4 k_reg[kQuads];
            float4 q_reg[kQuads];
            float4 kv_sum4 = float4(0.0f);
            #pragma unroll
            for (int pb = 0; pb < kQuads; ++pb) {{
              int d0 = pb * 32 + int(lane_in_row) * 4;
              k_reg[pb] = float4(
                  static_cast<float>(kr_ptr[d0 + 0]),
                  static_cast<float>(kr_ptr[d0 + 1]),
                  static_cast<float>(kr_ptr[d0 + 2]),
                  static_cast<float>(kr_ptr[d0 + 3]));
              q_reg[pb] = float4(
                  static_cast<float>(qr_ptr[d0 + 0]),
                  static_cast<float>(qr_ptr[d0 + 1]),
                  static_cast<float>(qr_ptr[d0 + 2]),
                  static_cast<float>(qr_ptr[d0 + 3]));
              st[pb] *= gt;
              kv_sum4 += st[pb] * k_reg[pb];
            }}
            float kv_mem = (kv_sum4.x + kv_sum4.y) + (kv_sum4.z + kv_sum4.w);
            kv_mem += simd_shuffle_xor(kv_mem, 1);
            kv_mem += simd_shuffle_xor(kv_mem, 2);
            kv_mem += simd_shuffle_xor(kv_mem, 4);

            float delta = (v_val - kv_mem) * bt;

            float4 out_sum4 = float4(0.0f);
            #pragma unroll
            for (int pb = 0; pb < kQuads; ++pb) {{
              st[pb] += k_reg[pb] * delta;
              out_sum4 += st[pb] * q_reg[pb];
            }}
            float out_val = (out_sum4.x + out_sum4.y) + (out_sum4.z + out_sum4.w);
            out_val += simd_shuffle_xor(out_val, 1);
            out_val += simd_shuffle_xor(out_val, 2);
            out_val += simd_shuffle_xor(out_val, 4);

            if (lane_in_row == 0 && rdv < Dv) {{
              y_row[rdv] = static_cast<InT>(out_val);
            }}
            }}
          }}
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {state_store_block}
    """
    return mx.fast.metal_kernel(
        name=kname,
        input_names=input_names,
        output_names=output_names,
        source=source,
    )


def _make_gdn_state_quantize_i8_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        int rows = int(Rows_arr[0]);
        constexpr int kQuads = Dk / 32;
        uint tid = thread_position_in_grid.x;
        uint row_tile = thread_position_in_grid.y;
        uint sg = tid >> 5;
        uint lane = tid & 31u;
        int row = int(row_tile) * 16 + int(sg) * 4 + int(lane >> 3);
        int lane_in_row = int(lane & 7u);
        if (row >= rows) return;

        auto src = state + row * Dk;
        auto wc = codes + row * Dk;
        auto ws = scales + row * kQuads;
        #pragma unroll
        for (int pb = 0; pb < kQuads; ++pb) {
          int d0 = pb * 32 + lane_in_row * 4;
          float v0 = static_cast<float>(src[d0 + 0]);
          float v1 = static_cast<float>(src[d0 + 1]);
          float v2 = static_cast<float>(src[d0 + 2]);
          float v3 = static_cast<float>(src[d0 + 3]);
          uint mag = max(
              max(as_type<uint>(v0) & 0x7fffffffu, as_type<uint>(v1) & 0x7fffffffu),
              max(as_type<uint>(v2) & 0x7fffffffu, as_type<uint>(v3) & 0x7fffffffu));
          mag = max(mag, simd_shuffle_xor(mag, 1));
          mag = max(mag, simd_shuffle_xor(mag, 2));
          mag = max(mag, simd_shuffle_xor(mag, 4));
          bfloat sc_bf = (mag < 0x7f800000u)
              ? bfloat(as_type<float>(mag) * (1.0f / 127.0f))
              : as_type<bfloat>(ushort(0x7fc0));
          if (lane_in_row == 0) {
            ws[pb] = sc_bf;
          }
          float sc_f = float(sc_bf);
          bool ok = (mag < 0x7f800000u) && (sc_f > 0.0f);
          if (ok) {
            wc[d0 + 0] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v0, sc_f)), -127.0f, 127.0f));
            wc[d0 + 1] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v1, sc_f)), -127.0f, 127.0f));
            wc[d0 + 2] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v2, sc_f)), -127.0f, 127.0f));
            wc[d0 + 3] = static_cast<int8_t>(clamp(precise::rint(precise::divide(v3, sc_f)), -127.0f, 127.0f));
          } else {
            wc[d0 + 0] = static_cast<int8_t>(0);
            wc[d0 + 1] = static_cast<int8_t>(0);
            wc[d0 + 2] = static_cast<int8_t>(0);
            wc[d0 + 3] = static_cast<int8_t>(0);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_gdn_state_quantize_i8",
        input_names=["state", "Rows_arr"],
        output_names=["codes", "scales"],
        source=source,
    )


def _make_gdn_state_dequantize_i8_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        int rows = int(Rows_arr[0]);
        constexpr int kQuads = Dk / 32;
        uint tid = thread_position_in_grid.x;
        uint row_tile = thread_position_in_grid.y;
        uint sg = tid >> 5;
        uint lane = tid & 31u;
        int row = int(row_tile) * 16 + int(sg) * 4 + int(lane >> 3);
        int lane_in_row = int(lane & 7u);
        if (row >= rows) return;

        auto rc = codes + row * Dk;
        auto rs = scales + row * kQuads;
        auto dst = state + row * Dk;
        #pragma unroll
        for (int pb = 0; pb < kQuads; ++pb) {
          bfloat sc_bf = rs[pb];
          ushort sc_bits = as_type<ushort>(sc_bf);
          bool sc_finite = (sc_bits & 0x7f80u) < 0x7f80u;
          float sc_f = sc_finite ? float(sc_bf) : as_type<float>(0x7fc00000u);
          int d0 = pb * 32 + lane_in_row * 4;
          if (sc_finite) {
            dst[d0 + 0] = static_cast<OutT>(static_cast<float>(rc[d0 + 0]) * sc_f);
            dst[d0 + 1] = static_cast<OutT>(static_cast<float>(rc[d0 + 1]) * sc_f);
            dst[d0 + 2] = static_cast<OutT>(static_cast<float>(rc[d0 + 2]) * sc_f);
            dst[d0 + 3] = static_cast<OutT>(static_cast<float>(rc[d0 + 3]) * sc_f);
          } else {
            OutT qnan = static_cast<OutT>(sc_f);
            dst[d0 + 0] = qnan;
            dst[d0 + 1] = qnan;
            dst[d0 + 2] = qnan;
            dst[d0 + 3] = qnan;
          }
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_gdn_state_dequantize_i8",
        input_names=["codes", "scales", "Rows_arr"],
        output_names=["state"],
        source=source,
    )


def _make_conv_silu_qknorm_kernel():
    """Fused T-parallel causal Conv1d(K=4)+SiLU+L2 QK-norm with Apple9-safe carry ordering."""
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        constexpr int kHeadsQ = Hk;
        constexpr int kKeyDim = Hk * Dk;
        constexpr int kValDim = Hv * Dv;
        constexpr int kConvDim = 2 * kKeyDim + kValDim;
        constexpr int kQKTiles = 2 * kHeadsQ;
        constexpr int kSgs = Dk / 32;
        static_assert(Dk >= 32 && (Dk % 32) == 0 && kSgs <= 8, "Dk must be 32..256 and multiple of 32");

        int T = int(T_arr[0]);
        int n_tblks = int(NBlk_arr[0]);
        uint lane_c = thread_position_in_grid.x;
        uint tile = thread_position_in_grid.y;
        uint bz = thread_position_in_grid.z;
        uint b_idx = bz / uint(n_tblks);
        int t_start = int(bz - b_idx * uint(n_tblks)) * T_BLK;
        int t_end = min(T, t_start + T_BLK);
        uint sg = lane_c >> 5;
        uint lane_in_sg = lane_c & 31u;

        int ch = -1;
        bool is_qk = false;
        bool is_q = false;
        if (int(tile) < kHeadsQ) {
          ch = int(tile) * Dk + int(lane_c);
          is_qk = true;
          is_q = true;
        } else if (int(tile) < kQKTiles) {
          int kt = int(tile) - kHeadsQ;
          ch = kKeyDim + kt * Dk + int(lane_c);
          is_qk = true;
        } else {
          int vt = int(tile) - kQKTiles;
          int voff = vt * Dk + int(lane_c);
          if (voff < kValDim) {
            ch = 2 * kKeyDim + voff;
          }
        }

        threadgroup float sg_red[8];

        // Apple9 macOS 26.6 guard: load conv carry & weights BEFORE loop, store ONLY AFTER loop.
        float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
        if (ch >= 0) {
          auto cw = conv_w + ch * 4;
          w0 = static_cast<float>(cw[0]);
          w1 = static_cast<float>(cw[1]);
          w2 = static_cast<float>(cw[2]);
          w3 = static_cast<float>(cw[3]);

          auto st_in_b = conv_state_in + b_idx * 3 * kConvDim + ch;
          auto xb = x + b_idx * T * kConvDim + ch;
          int t0 = t_start - 3;
          int t1 = t_start - 2;
          int t2 = t_start - 1;
          c0 = (t0 < 0) ? static_cast<float>(st_in_b[(3 + t0) * kConvDim]) : static_cast<float>(xb[t0 * kConvDim]);
          c1 = (t1 < 0) ? static_cast<float>(st_in_b[(3 + t1) * kConvDim]) : static_cast<float>(xb[t1 * kConvDim]);
          c2 = (t2 < 0) ? static_cast<float>(st_in_b[(3 + t2) * kConvDim]) : static_cast<float>(xb[t2 * kConvDim]);
        }

        float q_sc = float(inv_scale[0]);
        float eps_v = 1e-6f;
        bool write_state = (t_end == T) && (ch >= 0);

        for (int t = t_start; t < t_end; ++t) {
          float val = 0.0f;
          if (ch >= 0) {
            float xt = static_cast<float>(x[(b_idx * T + t) * kConvDim + ch]);
            float acc = c0 * w0 + c1 * w1 + c2 * w2 + xt * w3;
            c0 = c1;
            c1 = c2;
            c2 = xt;
            float sig = 1.0f / (1.0f + exp(-acc));
            val = acc * sig;
          }
          if (is_qk) {
            float sq = val * val;
            float s_sq = simd_sum(sq);
            if (lane_in_sg == 0) {
              sg_red[sg] = s_sq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot_sq = 0.0f;
            #pragma unroll
            for (int s = 0; s < kSgs; ++s) {
              tot_sq += sg_red[s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint sq_bits = as_type<uint>(tot_sq);
            bool sq_finite = (sq_bits & 0x7f800000u) < 0x7f800000u;
            float inv_n = sq_finite ? rsqrt(tot_sq + eps_v) : as_type<float>(0x7fc00000u);
            float normed = val * inv_n * (is_q ? q_sc : 1.0f);
            if (is_q) {
              q_out[(b_idx * T + t) * kKeyDim + int(tile) * Dk + int(lane_c)] = static_cast<InT>(normed);
            } else {
              k_out[(b_idx * T + t) * kKeyDim + (int(tile) - kHeadsQ) * Dk + int(lane_c)] = static_cast<InT>(normed);
            }
          } else if (ch >= 0) {
            int voff = ch - 2 * kKeyDim;
            v_out[(b_idx * T + t) * kValDim + voff] = static_cast<InT>(val);
          }
        }

        if (write_state) {
          auto st_out_b = conv_state_out + b_idx * 3 * kConvDim + ch;
          st_out_b[0 * kConvDim] = static_cast<StOutT>(c0);
          st_out_b[1 * kConvDim] = static_cast<StOutT>(c1);
          st_out_b[2 * kConvDim] = static_cast<StOutT>(c2);
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_conv_silu_qknorm_tpar",
        input_names=["x", "conv_state_in", "conv_w", "inv_scale", "T_arr", "NBlk_arr"],
        output_names=["q_out", "k_out", "v_out", "conv_state_out"],
        source=source,
    )


def _make_rms_norm_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        uint row = thread_position_in_grid.y;
        uint tid = thread_position_in_grid.x;
        constexpr int kStep = 32;
        constexpr int kVec = D / kStep;
        uint base = row * D;
        float xr[kVec];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
          uint d = uint(i) * uint(kStep) + tid;
          float v = static_cast<float>(x[base + d]);
          xr[i] = v;
          sq += v * v;
        }
        float tot = simd_sum(sq);
        uint bits = as_type<uint>(tot);
        bool fin = (bits & 0x7f800000u) < 0x7f800000u;
        float inv = fin ? rsqrt(tot * (1.0f / float(D)) + float(eps_arr[0])) : as_type<float>(0x7fc00000u);
        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
          uint d = uint(i) * uint(kStep) + tid;
          out[base + d] = static_cast<T>((xr[i] * inv) * static_cast<float>(w[d]));
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_rms_norm_f32",
        input_names=["x", "w", "eps_arr"],
        output_names=["out"],
        source=source,
    )


def _make_gated_rms_norm_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        uint row = thread_position_in_grid.y;
        uint tid = thread_position_in_grid.x;
        constexpr int kVec = D / 32;
        uint base = row * D;
        float xreg[kVec];
        float zreg[kVec];
        float wreg[kVec];
        float sq_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
          uint d = uint(i) * 32u + tid;
          float xv = static_cast<float>(x[base + d]);
          float zv = static_cast<float>(z[base + d]);
          float wv = static_cast<float>(w[d]);
          xreg[i] = xv;
          zreg[i] = zv;
          wreg[i] = wv;
          sq_sum += xv * xv;
        }
        float tot_sq = simd_sum(sq_sum);
        uint sq_bits = as_type<uint>(tot_sq);
        bool finite_sq = (sq_bits & 0x7f800000u) < 0x7f800000u;
        float inv_rms = finite_sq ? rsqrt(tot_sq * (1.0f / float(D)) + float(eps_arr[0]))
                                  : as_type<float>(0x7fc00000u);
        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
          uint d = uint(i) * 32u + tid;
          float zv = zreg[i];
          float sig = 1.0f / (1.0f + exp(-zv));
          float silu_z = zv * sig;
          out[base + d] = static_cast<T>(xreg[i] * inv_rms * wreg[i] * silu_z);
        }
    """
    return mx.fast.metal_kernel(
        name="pplx_gated_rms_norm_f32",
        input_names=["x", "z", "w", "eps_arr"],
        output_names=["out"],
        source=source,
    )


def _make_swiglu_split_kernel():
    if not mx.metal.is_available():
        return None
    source = r"""
        #pragma clang fp reassociate(off)
        uint row = thread_position_in_grid.y;
        uint col4 = thread_position_in_grid.x;
        uint h4 = H / 4u;
        if (col4 >= h4) return;

        uint base = row * (2u * H) + col4 * 4u;
        float g0 = static_cast<float>(gate_up[base + 0u]);
        float g1 = static_cast<float>(gate_up[base + 1u]);
        float g2 = static_cast<float>(gate_up[base + 2u]);
        float g3 = static_cast<float>(gate_up[base + 3u]);

        float u0 = static_cast<float>(gate_up[base + H + 0u]);
        float u1 = static_cast<float>(gate_up[base + H + 1u]);
        float u2 = static_cast<float>(gate_up[base + H + 2u]);
        float u3 = static_cast<float>(gate_up[base + H + 3u]);

        uint obase = row * H + col4 * 4u;
        out[obase + 0u] = static_cast<T>((g0 / (1.0f + exp(-g0))) * u0);
        out[obase + 1u] = static_cast<T>((g1 / (1.0f + exp(-g1))) * u1);
        out[obase + 2u] = static_cast<T>((g2 / (1.0f + exp(-g2))) * u2);
        out[obase + 3u] = static_cast<T>((g3 / (1.0f + exp(-g3))) * u3);
    """
    return mx.fast.metal_kernel(
        name="pplx_swiglu_split_x4",
        input_names=["gate_up"],
        output_names=["out"],
        source=source,
    )


_GATED_DELTA_KERNEL = _make_gated_delta_kernel(has_mask=False)
_GATED_DELTA_KERNEL_MASKED = _make_gated_delta_kernel(has_mask=True)
_GATED_DELTA_KERNEL_STAGED = _make_gated_delta_staged_kernel(segmented=False, state_i8=False)
_GATED_DELTA_KERNEL_STAGED_SEG = _make_gated_delta_staged_kernel(segmented=True, state_i8=False)
_GATED_DELTA_KERNEL_STAGED_I8 = _make_gated_delta_staged_kernel(segmented=False, state_i8=True)
_GATED_DELTA_KERNEL_STAGED_SEG_I8 = _make_gated_delta_staged_kernel(segmented=True, state_i8=True)
_GDN_STATE_QUANTIZE_I8_KERNEL = _make_gdn_state_quantize_i8_kernel()
_GDN_STATE_DEQUANTIZE_I8_KERNEL = _make_gdn_state_dequantize_i8_kernel()
_CONV_SILU_QKNORM_KERNEL = _make_conv_silu_qknorm_kernel()
_RMS_NORM_KERNEL = _make_rms_norm_kernel()
_GATED_RMS_NORM_KERNEL = _make_gated_rms_norm_kernel()
_SWIGLU_SPLIT_KERNEL = _make_swiglu_split_kernel()


def quantize_gdn_state_i8(state: mx.array) -> tuple[mx.array, mx.array]:
    """Compress recurrent GDN state ``(..., Dk)`` into Block-32 INT8 codes + BF16 scales.

    Uses Splash's ``state_code_quad`` non-finite preservation: any Block-32 quad containing
    ``Inf`` or ``NaN`` encodes to a quiet-``NaN`` bfloat16 scale (``0x7fc0``) and zero codes
    so dequantization restores every element of that block as ``NaN``.
    """
    if state.ndim < 1 or state.shape[-1] < 32 or state.shape[-1] % 32 != 0:
        raise ValueError(f"Expected last dim >= 32 divisible by 32, got shape {state.shape}")
    Dk = int(state.shape[-1])
    k_quads = Dk // 32
    codes_shape = tuple(int(d) for d in state.shape)
    scales_shape = (*codes_shape[:-1], k_quads)
    if state.size == 0:
        return mx.zeros(codes_shape, dtype=mx.int8), mx.zeros(scales_shape, dtype=mx.bfloat16)

    rows = int(state.size // Dk)
    if (
        _GDN_STATE_QUANTIZE_I8_KERNEL is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        codes, scales = _GDN_STATE_QUANTIZE_I8_KERNEL(
            inputs=[state.reshape(rows, Dk), mx.array([rows], dtype=mx.int32)],
            template=[("InT", state.dtype), ("Dk", Dk)],
            grid=(128, (rows + 15) // 16, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[codes_shape, scales_shape],
            output_dtypes=[mx.int8, mx.bfloat16],
        )
        return codes, scales

    st_np = np.asarray(state.astype(mx.float32), dtype=np.float32).reshape(rows, k_quads, 32)
    finite_blk = np.all(np.isfinite(st_np), axis=-1)
    amax = np.where(finite_blk, np.max(np.abs(st_np), axis=-1), 0.0).astype(np.float32)
    scales_f32 = np.where(finite_blk, amax * (1.0 / 127.0), np.nan).astype(np.float32)
    scales_bf16 = mx.array(scales_f32, dtype=mx.bfloat16)
    scales_rounded = np.asarray(scales_bf16.astype(mx.float32), dtype=np.float32)[..., None]
    valid = finite_blk[..., None] & (scales_rounded > 0.0)
    denom = np.where(valid, scales_rounded, 1.0)
    scaled = np.where(valid, st_np / denom, 0.0)
    codes_np = np.clip(np.rint(scaled), -127.0, 127.0).astype(np.int8)
    return mx.array(codes_np.reshape(codes_shape), dtype=mx.int8), scales_bf16.reshape(scales_shape)


def dequantize_gdn_state_i8(
    codes: mx.array,
    scales: mx.array,
    *,
    out_dtype: Any = mx.float32,
) -> mx.array:
    """Restore Block-32 INT8 ``(codes, scales)`` GDN recurrent state to ``out_dtype``."""
    if codes.ndim < 1 or codes.shape[-1] < 32 or codes.shape[-1] % 32 != 0:
        raise ValueError(f"Expected codes last dim divisible by 32, got {codes.shape}")
    Dk = int(codes.shape[-1])
    k_quads = Dk // 32
    expected_scales_shape = (*tuple(int(d) for d in codes.shape[:-1]), k_quads)
    if tuple(int(d) for d in scales.shape) != expected_scales_shape:
        raise ValueError(f"Expected scales shape {expected_scales_shape}, got {scales.shape}")
    if codes.size == 0:
        return mx.zeros(codes.shape, dtype=out_dtype)

    rows = int(codes.size // Dk)
    scales_bf16 = scales if scales.dtype == mx.bfloat16 else scales.astype(mx.bfloat16)
    if (
        _GDN_STATE_DEQUANTIZE_I8_KERNEL is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return _GDN_STATE_DEQUANTIZE_I8_KERNEL(  # guard: allow-fp32
            inputs=[codes.reshape(rows, Dk), scales_bf16.reshape(rows, k_quads), mx.array([rows], dtype=mx.int32)],
            template=[("OutT", out_dtype), ("Dk", Dk)],
            grid=(128, (rows + 15) // 16, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[codes.shape],
            output_dtypes=[out_dtype],
        )[0]

    sc_u16 = np.asarray(scales_bf16.view(mx.uint16), dtype=np.uint16).reshape(rows, k_quads, 1)
    sc_fin = (sc_u16 & np.uint16(0x7F80)) < np.uint16(0x7F80)
    sc_f32 = np.asarray(scales_bf16.astype(mx.float32), dtype=np.float32).reshape(rows, k_quads, 1)
    c_f32 = np.asarray(codes, dtype=np.float32).reshape(rows, k_quads, 32)
    restored = np.where(sc_fin, c_f32 * np.where(sc_fin, sc_f32, 0.0), np.nan).reshape(codes.shape)
    return mx.array(restored.astype(np.float32), dtype=out_dtype)


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
        y = mx.where(mask[:, None, None], y, mx.zeros_like(y))
    return y.astype(q.dtype), state


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    seg_flags: mx.array | None = None,
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
        if seg_flags is not None:
            flg_t = seg_flags[:, t]
            state = mx.where((flg_t == 2)[:, None, None, None], mx.zeros_like(state), state)
            step_mask = flg_t != 0
        else:
            step_mask = None if mask is None else mask[:, t]
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            step_mask,
        )
        ys.append(y)
    return mx.stack(ys, axis=1), state


def _effective_gdn_tile_rows(dk: int, input_type: Any, tile_rows: int | None) -> int:
    req = _GDN_TILE_ROWS if tile_rows is None else int(tile_rows)
    if req < 4 or req > 24:
        raise ValueError(f"Unsupported GDN tile_rows {req!r}; expected 4..24")
    elem_bytes = 4 if input_type == mx.float32 else 2
    per_row_bytes = (2 * dk + 16) * elem_bytes + 9
    max_rows = max(4, (28672 // per_row_bytes // 4) * 4)
    return min(req, max_rows)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | tuple[mx.array, mx.array] | None = None,
    mask: mx.array | None = None,
    *,
    seg_flags: mx.array | None = None,
    use_kernel: bool = True,
    state_fp32: bool = True,
    state_dtype: str | None = None,
    tile_rows: int | None = None,
) -> tuple[mx.array, mx.array | tuple[mx.array, mx.array]]:
    """Run GatedDeltaNet recurrence keeping ``q, k, v`` in activation dtype and configurable ``state_dtype``."""
    # Promote FP16 sigmoid inputs to FP32 so GdnDecodeShared keeps high-precision beta without disturbing BF16 parity.
    beta = mx.sigmoid(b.astype(mx.float32)) if b.dtype == mx.float16 else mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    has_carried_state = state is not None

    if state_dtype is not None or os.environ.get("FAST_PPLX_STATE_DTYPE", "").strip():
        resolved_state_dtype = normalize_state_dtype(state_dtype, state_fp32=state_fp32)
    else:
        resolved_state_dtype = "float32" if state_fp32 else "legacy_non_fp32"

    is_i8 = resolved_state_dtype == "int8"
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype

    if is_i8:
        if state is None:
            codes_in = mx.zeros((B, Hv, Dv, Dk), dtype=mx.int8)
            scales_in = mx.zeros((B, Hv, Dv, Dk // 32), dtype=mx.bfloat16)
        elif isinstance(state, tuple):
            codes_in, scales_in = state
        else:
            codes_in, scales_in = quantize_gdn_state_i8(state)
        state_out_target_dtype = None
        dense_state = None
    else:
        if resolved_state_dtype == "float32":
            state_out_target_dtype = mx.float32
        elif resolved_state_dtype == "bfloat16":
            state_out_target_dtype = mx.bfloat16
        else:
            state_out_target_dtype = (
                state.dtype
                if (state is not None and not isinstance(state, tuple))
                else q.dtype
            )
        if state is None:
            dense_state = mx.zeros((B, Hv, Dv, Dk), dtype=state_out_target_dtype)
        elif isinstance(state, tuple):
            dense_state = dequantize_gdn_state_i8(
                state[0], state[1], out_dtype=state_out_target_dtype
            )
        elif state.dtype != state_out_target_dtype:
            dense_state = state.astype(state_out_target_dtype)
        else:
            dense_state = state
        codes_in = None
        scales_in = None

    if T == 0 or B == 0:
        out_empty = mx.zeros((B, T, Hv, Dv), dtype=input_type)
        if is_i8:
            assert codes_in is not None and scales_in is not None
            return out_empty, (codes_in, scales_in)
        assert dense_state is not None
        return out_empty, dense_state

    if (
        not use_kernel
        or _GATED_DELTA_KERNEL is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or Dk < 32
        or Dk % 32 != 0
    ):
        if is_i8:
            assert codes_in is not None and scales_in is not None
            st_f32 = dequantize_gdn_state_i8(codes_in, scales_in, out_dtype=mx.float32)
            out_y, out_st_f32 = gated_delta_ops(q, k, v, g, beta, st_f32, mask, seg_flags=seg_flags)
            return out_y, quantize_gdn_state_i8(out_st_f32)
        assert dense_state is not None
        out_y, out_st_f32 = gated_delta_ops(
            q, k, v, g, beta, dense_state.astype(mx.float32), mask, seg_flags=seg_flags
        )
        assert state_out_target_dtype is not None
        if out_st_f32.dtype != state_out_target_dtype:
            out_st_f32 = out_st_f32.astype(state_out_target_dtype)
        return out_y, out_st_f32

    # 1. Native C++ mlx mx.fast.gated_delta_update on Apple10 hosts for single-pass unmasked/unsegmented FP16 (T <= 2048).
    #    For carried state (has_carried_state=True), T > 2048, BF16, masked/segmented batches, INT8 state, or Apple9 (M4),
    #    GdnDecodeShared keeps recurrent state strictly in FP32 registers and beats mx.fast on Apple9 (1.69 ms vs 1.96 ms).
    if (
        not has_carried_state
        and not is_i8
        and state_out_target_dtype == mx.float32
        and tile_rows is None
        and T <= 2048
        and input_type == mx.float16
        and mask is None
        and seg_flags is None
        and _USE_MX_FAST_GDN
        and _is_apple10_gdn_host()
        and g.ndim == 3
        and Dk >= 32
        and Dk % 32 == 0
        and Dv >= 32
        and Dv % 32 == 0
    ):
        out_y, out_state = mx.fast.gated_delta_update(q, k, v, g, beta)
        if out_state.dtype != mx.float32:
            out_state = out_state.astype(mx.float32)
        return out_y, out_state

    # 2. Threadgroup-staged GdnDecodeShared Metal kernel (128 threads, TG_DV=16, TILE_ROWS=16 default).
    staged_shape_ok = g.ndim == 3 and Dk >= 32 and Dk % 32 == 0 and _ENABLE_GDN_PACKED
    eff_flags = (
        seg_flags.astype(mx.uint8)
        if seg_flags is not None
        else (mask.astype(mx.uint8) if mask is not None else None)
    )
    is_seg = eff_flags is not None

    if is_i8:
        kernel = _GATED_DELTA_KERNEL_STAGED_SEG_I8 if is_seg else _GATED_DELTA_KERNEL_STAGED_I8
        if kernel is None or not staged_shape_ok:
            assert codes_in is not None and scales_in is not None
            st_f32 = dequantize_gdn_state_i8(codes_in, scales_in, out_dtype=mx.float32)
            out_y, out_st_f32 = gated_delta_ops(q, k, v, g, beta, st_f32, mask, seg_flags=seg_flags)
            return out_y, quantize_gdn_state_i8(out_st_f32)
        eff_tile_rows = _effective_gdn_tile_rows(Dk, input_type, tile_rows)
        assert codes_in is not None and scales_in is not None
        T_arr = mx.array([T], dtype=mx.int32)
        inputs = [q, k, v, g, beta, codes_in, scales_in, T_arr]
        if is_seg:
            inputs.append(eff_flags)
        out_y, codes_out, scales_out = kernel(
            inputs=inputs,
            template=[
                ("InT", input_type),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
                ("TILE_ROWS", eff_tile_rows),
                ("TG_DV", 16),
            ],
            grid=(128, (Dv + 15) // 16, B * Hv),
            threadgroup=(128, 1, 1),
            output_shapes=[(B, T, Hv, Dv), codes_in.shape, scales_in.shape],
            output_dtypes=[input_type, mx.int8, mx.bfloat16],
        )
        return out_y, (codes_out, scales_out)

    assert dense_state is not None and state_out_target_dtype is not None
    state_type = dense_state.dtype
    staged_kernel = _GATED_DELTA_KERNEL_STAGED_SEG if is_seg else _GATED_DELTA_KERNEL_STAGED
    if staged_shape_ok and staged_kernel is not None:
        eff_tile_rows = _effective_gdn_tile_rows(Dk, input_type, tile_rows)
        T_arr = mx.array([T], dtype=mx.int32)
        inputs = [q, k, v, g, beta, dense_state, T_arr]
        if is_seg:
            inputs.append(eff_flags)
        out_y, out_state = staged_kernel(  # guard: allow-fp32
            inputs=inputs,
            template=[
                ("InT", input_type),
                ("StT", state_type),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
                ("TILE_ROWS", eff_tile_rows),
                ("TG_DV", 16),
            ],
            grid=(128, (Dv + 15) // 16, B * Hv),
            threadgroup=(128, 1, 1),
            output_shapes=[(B, T, Hv, Dv), dense_state.shape],
            output_dtypes=[input_type, state_type],
        )
        return out_y, out_state

    if seg_flags is not None:
        out_y, out_st_f32 = gated_delta_ops(
            q, k, v, g, beta, dense_state.astype(mx.float32), mask, seg_flags=seg_flags
        )
        if out_st_f32.dtype != state_out_target_dtype:
            out_st_f32 = out_st_f32.astype(state_out_target_dtype)
        return out_y, out_st_f32

    if mask is not None and _GATED_DELTA_KERNEL_MASKED is not None:
        kernel = _GATED_DELTA_KERNEL_MASKED
        inputs = [q, k, v, g, beta, dense_state, T, mask]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    else:
        kernel = _GATED_DELTA_KERNEL
        inputs = [q, k, v, g, beta, dense_state, T]
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
        output_shapes=[(B, T, Hv, Dv), dense_state.shape],
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
        D = int(x.shape[-1])
        if (
            _RMS_NORM_KERNEL is not None
            and x.size > 0
            and D >= 32
            and D % 32 == 0
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            rows = int(x.size // D)
            return _RMS_NORM_KERNEL(
                inputs=[
                    x.reshape(rows, D),
                    self.weight,
                    mx.array([float(self.eps)], dtype=mx.float32),
                ],
                template=[("T", x.dtype), ("WgtT", self.weight.dtype), ("D", D)],
                grid=(32, rows, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]
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
        D = int(x.shape[-1])
        if (
            _GATED_RMS_NORM_KERNEL is not None
            and x.size > 0
            and x.shape == gate.shape
            and D >= 32
            and D % 32 == 0
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            rows = int(x.size // D)
            gate_in = gate if gate.dtype == x.dtype else gate.astype(x.dtype)
            return _GATED_RMS_NORM_KERNEL(
                inputs=[
                    x.reshape(rows, D),
                    gate_in.reshape(rows, D),
                    self.weight,
                    mx.array([float(self.eps)], dtype=mx.float32),
                ],
                template=[("T", x.dtype), ("WgtT", self.weight.dtype), ("D", D)],
                grid=(32, rows, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]
        x_f32 = x.astype(mx.float32)
        g_f32 = gate.astype(mx.float32)
        w_f32 = self.weight.astype(mx.float32)
        normed_f32 = mx.fast.rms_norm(x_f32, w_f32, self.eps)
        out_f32 = normed_f32 * nn.silu(g_f32)
        return out_f32.astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _norm_qk_head_hsd(x: mx.array, weight_f32: mx.array, eps: float) -> mx.array:
    """Evaluate per-head Q/K RMSNorm in float32 and transpose `[B, S, H, D] -> [B, H, S, D]` without Q/K concat."""
    normed_f32 = mx.fast.rms_norm(x.astype(mx.float32), weight_f32, eps)
    return normed_f32.astype(x.dtype).swapaxes(1, 2)


@mx.compile
def _apply_custom_partial_rope(
    qk_hsd: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    rope_dim: int,
) -> mx.array:
    half = rope_dim // 2
    x1 = qk_hsd[..., :half]
    x2 = qk_hsd[..., half:rope_dim]
    rest = qk_hsd[..., rope_dim:]
    o1 = x1 * rope_cos - x2 * rope_sin
    o2 = x2 * rope_cos + x1 * rope_sin
    return mx.concatenate([o1, o2, rest], axis=-1)


@mx.compile
def _build_varlen_mask_and_rope(
    doc_ids_mx: mx.array,
    pos_ids_mx: mx.array,
    cos_tbl_mx: mx.array,
    sin_tbl_mx: mx.array,
    zero_val: mx.array,
    neg_val: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    t_len = doc_ids_mx.shape[1]
    idx = mx.arange(t_len, dtype=mx.int32)
    d_col = doc_ids_mx[:, None, :, None]
    d_row = doc_ids_mx[:, None, None, :]
    same_doc = (d_col >= 0) & (d_col == d_row)
    pad_diag = (d_col < 0) & (idx[None, None, :, None] == idx[None, None, None, :])
    sdpa_mask = mx.where(same_doc | pad_diag, zero_val, neg_val)
    rope_cos = cos_tbl_mx[pos_ids_mx][:, None, :, :]
    rope_sin = sin_tbl_mx[pos_ids_mx][:, None, :, :]
    return sdpa_mask, rope_cos, rope_sin


class PplxAttention(nn.Module):
    """Bidirectional grouped-query attention with output gating, fused [Q,K] norm+RoPE, and Apple9/Apple10 dispatch."""

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
        self.rms_norm_eps = float(cfg.rms_norm_eps)
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
        *,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
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

        keys_4d = keys.reshape(B, S, self.num_kv_heads, self.head_dim)
        qw = self.q_norm.weight
        kw = self.k_norm.weight
        qw_f32 = qw if qw.dtype == mx.float32 else qw.astype(mx.float32)
        kw_f32 = kw if kw.dtype == mx.float32 else kw.astype(mx.float32)
        queries_hsd = _norm_qk_head_hsd(queries, qw_f32, self.rms_norm_eps)
        keys_hsd = _norm_qk_head_hsd(keys_4d, kw_f32, self.rms_norm_eps)
        values = values.reshape(B, S, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

        qk_hsd = mx.concatenate([queries_hsd, keys_hsd], axis=1)
        if rope_cos is not None and rope_sin is not None:
            qk_hsd = _apply_custom_partial_rope(qk_hsd, rope_cos, rope_sin, self.rope_dim)
        else:
            qk_hsd = mx.fast.rope(
                qk_hsd,
                self.rope_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=0,
            )
        queries, keys = mx.split(qk_hsd, [self.num_heads], axis=1)

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
    """Linear-attention GatedDeltaNet layer with fused `in_proj_all`, causal Conv1d, and configurable state streaming."""

    def __init__(self, cfg: PplxConfig, *, state_dtype: str | None = None) -> None:
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.total_in_dim = self.conv_dim + self.value_dim + 2 * self.num_v_heads
        self.inv_scale = float(self.head_k_dim) ** -0.5
        self.state_dtype = normalize_state_dtype(state_dtype)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_all = nn.Linear(cfg.hidden_size, self.total_in_dim, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,), dtype=mx.float32)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=mx.float32)
        self.norm = PplxGatedRMSNorm(self.head_v_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.use_kernel = True

    def _run_step(
        self,
        x_slice: mx.array,
        conv_state: mx.array,
        rec_state: mx.array | tuple[mx.array, mx.array] | None,
        mask_slice: mx.array | None,
        *,
        seg_flags_slice: mx.array | None = None,
        keep_conv_state_fp32: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array | tuple[mx.array, mx.array]]:
        B, S_sub, _ = x_slice.shape

        proj = self.in_proj_all(x_slice)
        s_qkv = self.conv_dim
        s_z = s_qkv + self.value_dim
        s_b = s_z + self.num_v_heads
        mixed_qkv, z_flat, b_slice, a_slice = mx.split(proj, [s_qkv, s_z, s_b], axis=-1)
        z = z_flat.reshape(B, S_sub, self.num_v_heads, self.head_v_dim)

        conv_w = self.conv1d.weight.squeeze(-1)
        use_fused_conv = (
            self.use_kernel
            and _CONV_SILU_QKNORM_KERNEL is not None
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and B > 0
            and S_sub > 0
            and self.conv_kernel_size == 4
            and getattr(self.conv1d, "bias", None) is None
            and 32 <= self.head_k_dim <= 256
            and self.head_k_dim % 32 == 0
            and conv_w.shape == (self.conv_dim, 4)
            and conv_state.shape == (B, 3, self.conv_dim)
        )
        if use_fused_conv:
            t_blk = 32
            n_blks = (S_sub + t_blk - 1) // t_blk
            tiles = 2 * self.num_k_heads + (self.value_dim + self.head_k_dim - 1) // self.head_k_dim
            conv_out_dtype = mx.float32 if keep_conv_state_fp32 else mixed_qkv.dtype
            q, k, v, new_conv_state = _CONV_SILU_QKNORM_KERNEL(  # guard: allow-fp32
                inputs=[
                    mixed_qkv,
                    conv_state,
                    conv_w,
                    mx.array([float(self.inv_scale)], dtype=mx.float32),
                    mx.array([S_sub], dtype=mx.int32),
                    mx.array([n_blks], dtype=mx.int32),
                ],
                template=[
                    ("InT", mixed_qkv.dtype),
                    ("StInT", conv_state.dtype),
                    ("StOutT", conv_out_dtype),
                    ("WgtT", conv_w.dtype),
                    ("Dk", self.head_k_dim),
                    ("Dv", self.head_v_dim),
                    ("Hk", self.num_k_heads),
                    ("Hv", self.num_v_heads),
                    ("T_BLK", t_blk),
                ],
                grid=(self.head_k_dim, tiles, B * n_blks),
                threadgroup=(self.head_k_dim, 1, 1),
                output_shapes=[
                    (B, S_sub, self.num_k_heads, self.head_k_dim),
                    (B, S_sub, self.num_k_heads, self.head_k_dim),
                    (B, S_sub, self.num_v_heads, self.head_v_dim),
                    (B, 3, self.conv_dim),
                ],
                output_dtypes=[mixed_qkv.dtype, mixed_qkv.dtype, mixed_qkv.dtype, conv_out_dtype],
            )
        else:
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
            seg_flags=seg_flags_slice,
            use_kernel=self.use_kernel,
            state_dtype=self.state_dtype,
        )
        normed = self.norm(out, z).reshape(B, S_sub, self.value_dim)
        return self.out_proj(normed), new_conv_state, new_rec_state

    def __call__(
        self,
        x: mx.array,
        ssm_mask: mx.array | None = None,
        *,
        seg_flags: mx.array | None = None,
    ) -> mx.array:
        B, S, _ = x.shape
        if seg_flags is not None:
            zero_val = mx.array(0.0, dtype=x.dtype)
            x = mx.where((seg_flags != 0)[..., None], x, zero_val)
        elif ssm_mask is not None:
            zero_val = mx.array(0.0, dtype=x.dtype)
            x = mx.where(ssm_mask[..., None], x, zero_val)

        if S <= _STREAMED_SEQ_THRESHOLD:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
            out, _, _ = self._run_step(
                x,
                conv_state,
                None,
                ssm_mask,
                seg_flags_slice=seg_flags,
                keep_conv_state_fp32=False,
            )
            return out

        if self.state_dtype == "int8" and self.head_k_dim >= 32 and self.head_k_dim % 32 == 0:
            rec_state: mx.array | tuple[mx.array, mx.array] | None = (
                mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.int8),
                mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim // 32), dtype=mx.bfloat16),
            )
        else:
            init_state_dtype = mx.bfloat16 if self.state_dtype == "bfloat16" else mx.float32
            rec_state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=init_state_dtype)
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=mx.float32)
        outs: list[mx.array] = []
        for s0, s1 in tile_aligned_slices(S, slice_len=_GDN_SLICE_LEN, min_tail=_GDN_SLICE_LEN):
            mask_slice = None if ssm_mask is None else ssm_mask[:, s0:s1]
            flags_slice = None if seg_flags is None else seg_flags[:, s0:s1]
            y_slice, conv_state, rec_state = self._run_step(
                x[:, s0:s1, :],
                conv_state,
                rec_state,
                mask_slice,
                seg_flags_slice=flags_slice,
                keep_conv_state_fp32=True,
            )
            rec_eval = rec_state if isinstance(rec_state, tuple) else (rec_state,)
            mx.async_eval(y_slice, conv_state, *rec_eval)
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
        h = self.intermediate_size
        rows = gate_up.size // (2 * h) if h > 0 else 0
        if (
            _SWIGLU_SPLIT_KERNEL is not None
            and rows > 0
            and h >= 4
            and h % 4 == 0
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            act = _SWIGLU_SPLIT_KERNEL(
                inputs=[gate_up.reshape(rows, 2 * h)],
                template=[("T", gate_up.dtype), ("H", h)],
                grid=(((h // 4 + 31) // 32) * 32, rows, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(*x.shape[:-1], h)],
                output_dtypes=[gate_up.dtype],
            )[0]
            return self.down_proj(act)
        gate, up = mx.split(gate_up, [h], axis=-1)
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
        state_dtype: str | None = None,
    ) -> None:
        super().__init__()
        self.is_linear = is_linear
        self.input_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = PplxMLP(cfg)
        if is_linear:
            self.linear_attn = PplxGatedDeltaNet(cfg, state_dtype=state_dtype)
        else:
            self.self_attn = PplxAttention(cfg, gpu_family=gpu_family)

    def __call__(
        self,
        h: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        seg_flags: mx.array | None = None,
        sdpa_mask: mx.array | str | None = None,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
    ) -> mx.array:
        x_in = self.input_layernorm(h)
        if self.is_linear:
            r = self.linear_attn(x_in, ssm_mask=ssm_mask, seg_flags=seg_flags)
        else:
            r = self.self_attn(x_in, sdpa_mask=sdpa_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        h = h + r
        return h + self.mlp(self.post_attention_layernorm(h))


class PplxModel(nn.Module):
    def __init__(
        self,
        cfg: PplxConfig,
        *,
        has_embed_module: bool = True,
        gpu_family: Mapping[str, Any] | None = None,
        state_dtype: str | None = None,
    ) -> None:
        super().__init__()
        self.config = cfg
        self.gpu_family = _detect_gpu_family_info() if gpu_family is None else dict(gpu_family)
        self.state_dtype = normalize_state_dtype(state_dtype)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) if has_embed_module else None
        self.layers = [
            PplxDecoderLayer(
                cfg,
                is_linear=(cfg.layer_types[i] == "linear_attention"),
                gpu_family=self.gpu_family,
                state_dtype=self.state_dtype,
            )
            for i in range(cfg.num_hidden_layers)
        ]
        self.norm = PplxRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.contextual_projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def load_weights(self, file_or_weights: Any, strict: bool = True) -> "PplxModel":
        if not isinstance(file_or_weights, (str, Path)):
            w_dict = dict(file_or_weights) if not isinstance(file_or_weights, Mapping) else dict(file_or_weights.items())
            for i in range(self.config.num_hidden_layers):
                gdn_p = f"layers.{i}.linear_attn"
                unfused = (
                    f"{gdn_p}.in_proj_qkv.weight",
                    f"{gdn_p}.in_proj_z.weight",
                    f"{gdn_p}.in_proj_b.weight",
                    f"{gdn_p}.in_proj_a.weight",
                )
                if all(k in w_dict for k in unfused):
                    w_dict[f"{gdn_p}.in_proj_all.weight"] = mx.concatenate([w_dict.pop(k) for k in unfused], axis=0)
            return super().load_weights(list(w_dict.items()), strict=strict)
        return super().load_weights(file_or_weights, strict=strict)

    def forward_hidden(
        self,
        hidden: mx.array,
        *,
        ssm_mask: mx.array | None = None,
        seg_flags: mx.array | None = None,
        sdpa_mask: mx.array | str | None = None,
        rope_cos: mx.array | None = None,
        rope_sin: mx.array | None = None,
        return_trace: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        trace: list[mx.array] = []
        if return_trace:
            trace.append(hidden)
        h = hidden
        is_streamed = hidden.shape[1] > _STREAMED_SEQ_THRESHOLD
        for layer_idx, layer in enumerate(self.layers):
            h = layer(
                h,
                ssm_mask=ssm_mask,
                seg_flags=seg_flags,
                sdpa_mask=sdpa_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
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

    # Fuse linear_attn in_proj_{qkv,z,b,a} -> in_proj_all, q_proj + k_proj + v_proj -> qkv_proj,
    # and gate_proj + up_proj -> gate_up_proj idempotently.
    for i in range(num_layers):
        gdn_prefix = f"layers.{i}.linear_attn"
        in_projs = (
            f"{gdn_prefix}.in_proj_qkv.weight",
            f"{gdn_prefix}.in_proj_z.weight",
            f"{gdn_prefix}.in_proj_b.weight",
            f"{gdn_prefix}.in_proj_a.weight",
        )
        if all(k in normalized for k in in_projs):
            normalized[f"{gdn_prefix}.in_proj_all.weight"] = mx.concatenate(
                [normalized.pop(k) for k in in_projs], axis=0
            )

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


def _resolve_max_length(max_length: int | None, max_seq_len: int) -> int:
    if max_length is None:
        return max_seq_len
    ml = int(max_length)
    if ml <= 0:
        raise ValueError("max_length must be > 0")
    return min(ml, max_seq_len)


def _quantize_pplx_linear_weights(model: PplxModel, *, bits: int = 8, group_size: int = 64) -> None:
    """Quantize backbone linear projections while keeping embeddings, RMSNorms, conv1d, and contextual head native."""
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, mod: (
            isinstance(mod, nn.Linear)
            and "contextual_projection" not in path
            and int(mod.weight.shape[-1]) % group_size == 0
        ),
    )


class FastPplxGPU:
    """High-throughput MLX Metal GPU backend for `pplx-embed-v2-context-0.6b`."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        model_dir: str | Path | None = None,
        dtype: str = "float16",
        state_dtype: str | None = None,
        quantize_weights: Literal["none", "w8"] = "none",
        use_mapped_embedding: bool = True,
        cache_limit_mb: int = 512,
        local_files_only: bool = True,
        max_batch_size: int = 32,
        max_seq_len: int = 32768,
    ) -> None:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype {dtype!r}; expected 'float16' or 'bfloat16'")
        eff_quant_weights = str(quantize_weights).strip().lower()
        if eff_quant_weights not in _VALID_QUANTIZE_WEIGHTS:
            raise ValueError(
                f"Unsupported quantize_weights {quantize_weights!r}; expected one of {_VALID_QUANTIZE_WEIGHTS}"
            )

        self.paths = resolve_paths(home)
        self.dtype_name = dtype
        self.state_dtype_name = normalize_state_dtype(state_dtype)
        self.quantize_weights: Literal["none", "w8"] = "w8" if eff_quant_weights == "w8" else "none"
        self.act_dtype = mx.float16 if dtype == "float16" else mx.bfloat16
        self.use_mapped_embedding = bool(use_mapped_embedding)
        self.cache_limit_mb = int(cache_limit_mb)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.runtime_telemetry = {
            **get_mlx_runtime_telemetry(),
            "state_dtype": self.state_dtype_name,
            "quantize_weights": self.quantize_weights,
        }

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
                state_dtype=self.state_dtype_name,
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
            if self.quantize_weights == "w8":
                _quantize_pplx_linear_weights(self.model, bits=8, group_size=64)
                mx.eval(self.model.parameters())
        except Exception:
            self.close()
            raise

    def _embed_inputs(self, input_ids: np.ndarray) -> mx.array:
        if self._mapped_embed is not None:
            # Gather as float32 when F32/BF16 weights feed non-FP16 activations to avoid F16 truncation/subnormal loss.
            src_dtype = str(getattr(self._mapped_embed, "dtype", "F16")).upper()
            use_f32_gather = src_dtype not in ("F16", "FLOAT16") and self.act_dtype != mx.float16
            out_dt = "float32" if use_f32_gather else "float16"
            gathered = self._mapped_embed.gather(input_ids, out_dtype=out_dt)
            arr = mx.array(gathered, dtype=mx.float32 if use_f32_gather else mx.float16)
            return arr if arr.dtype == self.act_dtype else arr.astype(self.act_dtype)
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

    def _prepare_batch_inputs_and_mask(
        self, rows: Sequence[Sequence[int]]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Avoid pad buffer and all-True mask allocation when every sequence in the wave has equal positive length."""
        if rows:
            first_len = len(rows[0])
            if first_len > 0 and all(len(r) == first_len for r in rows):
                return np.asarray(rows, dtype=np.int32), None
        return self.tokenizer.pad(rows)

    def _project_mean_pool(self, hidden: mx.array, attention_mask: np.ndarray | None = None) -> np.ndarray:
        hidden_f32 = hidden.astype(mx.float32)
        if attention_mask is None:
            pooled = mx.mean(hidden_f32, axis=1)
        else:
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

    def _can_pack_varlen_1d(self) -> bool:
        return (
            _ENABLE_VARLEN_PACK
            and type(self).forward_hidden is FastPplxGPU.forward_hidden
            and isinstance(getattr(self, "model", None), PplxModel)
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    def _ensure_varlen_rope_table(self, max_pos: int) -> tuple[mx.array, mx.array]:
        cached_mx = getattr(self, "_varlen_rope_mx", None)
        if cached_mx is not None and int(cached_mx[0].shape[0]) >= max_pos:
            return cached_mx
        cap = max(max_pos, _STREAMED_SEQ_THRESHOLD)
        inv_freq = 1.0 / (
            float(self.config.rope_theta) ** (np.arange(0, self.config.rope_dim, 2, dtype=np.float64) / self.config.rope_dim)
        )
        ang = np.arange(cap, dtype=np.float64)[:, None] * inv_freq[None, :]
        cos_mx = mx.array(np.cos(ang).astype(np.float32), dtype=self.act_dtype)
        sin_mx = mx.array(np.sin(ang).astype(np.float32), dtype=self.act_dtype)
        mx.eval(cos_mx, sin_mx)
        self._varlen_rope_mx = (cos_mx, sin_mx)
        return cos_mx, sin_mx

    def _pack_varlen_token_bins(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        spans_list: Sequence[Sequence[tuple[int, int]]] | None = None,
        max_packed_tokens: int | None = None,
        align_to: int = 64,
    ) -> list[dict[str, Any]]:
        """FFD-pack variable-length docs into `1024 / 2048` 1D lanes with `conv_gap` isolation."""
        conv_gap = max(3, int(self.config.linear_conv_kernel_dim) - 1)
        pad_id = int(self.tokenizer.pad_id)
        hard_max = _STREAMED_SEQ_THRESHOLD if max_packed_tokens is None else int(max_packed_tokens)
        short_lane_cap = min(1024, hard_max) if max_packed_tokens is None else int(max_packed_tokens)

        indexed: list[tuple[int, list[int], list[tuple[int, int]]]] = []
        for idx, row in enumerate(token_lists):
            ids = list(row)
            if not ids:
                continue
            sp = [(0, len(ids))] if spans_list is None else [(int(a), int(b)) for a, b in spans_list[idx]]
            indexed.append((idx, ids, sp))

        # Sort descending by token length for tight First-Fit Decreasing (FFD) bin packing.
        indexed.sort(key=lambda item: len(item[1]), reverse=True)

        open_bins: list[dict[str, Any]] = []
        finalized: list[dict[str, Any]] = []

        for orig_idx, ids, spans in indexed:
            ln = len(ids)
            if ln > hard_max:
                finalized.append(
                    {
                        "single_unpacked": True,
                        "orig_indices": [orig_idx],
                        "doc_spans_counts": [len(spans)],
                        "token_ids": np.asarray(ids, dtype=np.int32),
                        "flat_spans": spans,
                    }
                )
                continue

            placed = False
            for b in open_bins:
                st = b["cursor"] + conv_gap
                if st + ln <= b["cap"]:
                    b["entries"].append((orig_idx, st, ids, spans))
                    b["cursor"] = st + ln
                    placed = True
                    break
            if not placed:
                bin_cap = short_lane_cap if ln <= short_lane_cap else hard_max
                open_bins.append(
                    {
                        "entries": [(orig_idx, 0, ids, spans)],
                        "cursor": ln,
                        "cap": bin_cap,
                    }
                )

        only_one_short_doc = len(open_bins) == 1 and len(open_bins[0]["entries"]) == 1
        for b in open_bins:
            entries = b["entries"]
            if only_one_short_doc:
                orig_idx, _, ids, spans = entries[0]
                finalized.append(
                    {
                        "single_unpacked": True,
                        "orig_indices": [orig_idx],
                        "doc_spans_counts": [len(spans)],
                        "token_ids": np.asarray(ids, dtype=np.int32),
                        "flat_spans": spans,
                    }
                )
                continue

            used = int(b["cursor"])
            t_cap = min(int(b["cap"]), max(align_to, ((used + align_to - 1) // align_to) * align_to))
            tok_arr = np.full((t_cap,), pad_id, dtype=np.int32)
            doc_arr = np.full((t_cap,), -1, dtype=np.int32)
            pos_arr = np.zeros((t_cap,), dtype=np.int32)
            flg_arr = np.zeros((t_cap,), dtype=np.uint8)
            orig_indices: list[int] = []
            doc_spans_counts: list[int] = []
            flat_spans: list[tuple[int, int]] = []

            for d_slot, (orig_idx, st, ids, spans) in enumerate(entries):
                ln = len(ids)
                tok_arr[st : st + ln] = np.asarray(ids, dtype=np.int32)
                doc_arr[st : st + ln] = d_slot
                pos_arr[st : st + ln] = np.arange(ln, dtype=np.int32)
                flg_arr[st : st + ln] = 1
                if d_slot > 0:
                    flg_arr[st] = 2  # Reset FP32 GDN state at document boundary
                orig_indices.append(orig_idx)
                doc_spans_counts.append(len(spans))
                for s0, s1 in spans:
                    if s1 > s0:
                        flat_spans.append((st + s0, st + min(s1, ln)))
                    else:
                        flat_spans.append((0, 0))

            finalized.append(
                {
                    "single_unpacked": False,
                    "orig_indices": orig_indices,
                    "doc_spans_counts": doc_spans_counts,
                    "token_ids": tok_arr,
                    "doc_ids": doc_arr,
                    "pos_ids": pos_arr,
                    "seg_flags": flg_arr,
                    "flat_spans": flat_spans,
                }
            )
        return finalized

    def _run_varlen_bins(
        self,
        bins: Sequence[dict[str, Any]],
        *,
        max_wave_tokens: int = 8192,
    ) -> list[np.ndarray]:
        """Execute varlen packed bins in `[B_bins, T_wave]` waves to saturate all GPU cores."""
        n_bins = len(bins)
        if n_bins == 0:
            return []
        out_per_bin: list[np.ndarray | None] = [None] * n_bins
        packed_items: list[tuple[int, dict[str, Any]]] = []

        for b_idx, bin_spec in enumerate(bins):
            flat_spans: list[tuple[int, int]] = bin_spec["flat_spans"]
            if not flat_spans:
                out_per_bin[b_idx] = np.zeros((0, self.dim), dtype=np.float32)
            elif bin_spec["single_unpacked"]:
                tok_2d = bin_spec["token_ids"][None, :]
                hidden = self.forward_hidden(tok_2d, None)
                span_arr = np.asarray(flat_spans, dtype=np.int32)[None, :, :]
                out_per_bin[b_idx] = self._project_span_pool(hidden, span_arr)[0]
            else:
                packed_items.append((b_idx, bin_spec))

        if not packed_items:
            return [r if r is not None else np.zeros((0, self.dim), dtype=np.float32) for r in out_per_bin]

        # Sort descending by t_cap so uniform-width bins (e.g. 8 x 1024) batch with zero inter-lane pad.
        packed_items.sort(key=lambda item: int(item[1]["token_ids"].shape[0]), reverse=True)
        pad_id = int(self.tokenizer.pad_id)
        zero_val = mx.array(0.0, dtype=self.act_dtype)
        neg_val = mx.array(-65504.0 if self.dtype_name == "float16" else -1e9, dtype=self.act_dtype)

        p_pos = 0
        total_p = len(packed_items)
        while p_pos < total_p:
            t_wave = int(packed_items[p_pos][1]["token_ids"].shape[0])
            max_b = max(1, min(self.max_batch_size, max_wave_tokens // max(1, t_wave)))
            wave: list[tuple[int, dict[str, Any]]] = [packed_items[p_pos]]
            p_pos += 1
            while p_pos < total_p and len(wave) < max_b:
                cand_t = int(packed_items[p_pos][1]["token_ids"].shape[0])
                rem_after = total_p - (p_pos + 1)
                # Avoid inflating many 1024-tok lanes into a 2048-tok wave, while absorbing a single orphan tail.
                if cand_t < (t_wave * 3) // 4 and not (rem_after == 0 and t_wave <= 1024):
                    break
                wave.append(packed_items[p_pos])
                p_pos += 1

            b_w = len(wave)
            uniform_t = all(int(spec["token_ids"].shape[0]) == t_wave for _, spec in wave)
            if uniform_t:
                tok_2d = np.stack([spec["token_ids"] for _, spec in wave], axis=0)
                doc_2d = np.stack([spec["doc_ids"] for _, spec in wave], axis=0)
                pos_2d = np.stack([spec["pos_ids"] for _, spec in wave], axis=0)
                flg_2d = np.stack([spec["seg_flags"] for _, spec in wave], axis=0)
            else:
                tok_2d = np.full((b_w, t_wave), pad_id, dtype=np.int32)
                doc_2d = np.full((b_w, t_wave), -1, dtype=np.int32)
                pos_2d = np.zeros((b_w, t_wave), dtype=np.int32)
                flg_2d = np.zeros((b_w, t_wave), dtype=np.uint8)
                for w_i, (_, spec) in enumerate(wave):
                    bt = int(spec["token_ids"].shape[0])
                    tok_2d[w_i, :bt] = spec["token_ids"]
                    doc_2d[w_i, :bt] = spec["doc_ids"]
                    pos_2d[w_i, :bt] = spec["pos_ids"]
                    flg_2d[w_i, :bt] = spec["seg_flags"]

            max_pos_req = int(pos_2d.max()) + 1
            cos_mx, sin_mx = self._ensure_varlen_rope_table(max_pos_req)
            sdpa_mask, rope_cos, rope_sin = _build_varlen_mask_and_rope(
                mx.array(doc_2d, dtype=mx.int32),
                mx.array(pos_2d, dtype=mx.int32),
                cos_mx,
                sin_mx,
                zero_val,
                neg_val,
            )
            hidden = self.model.forward_hidden(
                self._embed_inputs(tok_2d),
                seg_flags=mx.array(flg_2d, dtype=mx.uint8),
                sdpa_mask=sdpa_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

            max_sp = max(1, *(len(spec["flat_spans"]) for _, spec in wave))
            span_arr = np.zeros((b_w, max_sp, 2), dtype=np.int32)
            for w_i, (_, spec) in enumerate(wave):
                f_sp = spec["flat_spans"]
                if f_sp:
                    span_arr[w_i, : len(f_sp)] = np.asarray(f_sp, dtype=np.int32)

            proj_3d = self._project_span_pool(hidden, span_arr)
            for w_i, (orig_b_idx, spec) in enumerate(wave):
                n_sp = len(spec["flat_spans"])
                out_per_bin[orig_b_idx] = proj_3d[w_i, :n_sp]

        return [r if r is not None else np.zeros((0, self.dim), dtype=np.float32) for r in out_per_bin]

    def _run_varlen_bin(self, bin_spec: dict[str, Any]) -> np.ndarray:
        return self._run_varlen_bins([bin_spec])[0]

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
        bs = max(1, min(int(batch_size), self.max_batch_size))
        proj_all = np.zeros((n, self.dim), dtype=np.float32)

        # Use 1D varlen sequence packing when batch has multiple unequal-length docs (avoids pad waste + mask fallback).
        has_varlen = n > 1 and int(lengths.min()) != int(lengths.max())
        if has_varlen and self._can_pack_varlen_1d():
            bins = self._pack_varlen_token_bins(token_lists)
            for bin_spec, proj_bin in zip(bins, self._run_varlen_bins(bins), strict=True):
                for pos_i, orig_idx in enumerate(bin_spec["orig_indices"]):
                    proj_all[int(orig_idx)] = proj_bin[pos_i]
        else:
            uniform_pos_len = int(lengths[0]) if int(lengths.min()) == int(lengths.max()) > 0 else 0
            eff_bs = min(bs, max(1, 8192 // uniform_pos_len)) if uniform_pos_len > 512 else bs
            order = np.argsort(lengths, kind="stable")
            for start in range(0, n, eff_bs):
                idxs = order[start : start + eff_bs]
                rows = [token_lists[int(i)] for i in idxs]
                input_ids, attention_mask = self._prepare_batch_inputs_and_mask(rows)
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
        max_length: int | None = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        eff_max = _resolve_max_length(max_length, self.max_seq_len)
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
        max_length: int | None = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> list[np.ndarray]:
        eff_max = _resolve_max_length(max_length, self.max_seq_len)
        encoded_docs = [
            self.tokenizer.encode_contextual_document(chunks, return_spans=True, max_length=eff_max)
            for chunks in documents
        ]
        return self.encode_contextual_tokens(
            encoded_docs, batch_size=batch_size, dimensions=dimensions, output_dtype=output_dtype, normalize=normalize
        )

    def encode_contextual_tokens(
        self,
        encoded_docs: Sequence[tuple[Sequence[int], Sequence[tuple[int, int]]]],
        *,
        batch_size: int = 8,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
    ) -> list[np.ndarray]:
        """Pool one vector per supplied span, preserving document and chunk order."""
        if not encoded_docs:
            return []
        lengths = np.array([len(item[0]) for item in encoded_docs], dtype=np.int64)
        bs = max(1, min(int(batch_size), self.max_batch_size))
        results: list[np.ndarray | None] = [None] * len(encoded_docs)

        has_varlen = len(encoded_docs) > 1 and int(lengths.min()) != int(lengths.max())
        if has_varlen and self._can_pack_varlen_1d():
            tok_lists = [item[0] for item in encoded_docs]
            all_spans = [item[1] for item in encoded_docs]
            for orig_i, (ids, sp) in enumerate(encoded_docs):
                if len(ids) == 0:
                    results[orig_i] = self._format_output(
                        np.zeros((len(sp), self.dim), dtype=np.float32),
                        dimensions=dimensions,
                        output_dtype=output_dtype,
                        normalize=normalize,
                    )
            bins = self._pack_varlen_token_bins(tok_lists, spans_list=all_spans)
            for bin_spec, proj_bin in zip(bins, self._run_varlen_bins(bins), strict=True):
                cur = 0
                for orig_idx, n_sp in zip(bin_spec["orig_indices"], bin_spec["doc_spans_counts"], strict=True):
                    doc_proj = proj_bin[cur : cur + n_sp]
                    cur += n_sp
                    results[int(orig_idx)] = self._format_output(
                        doc_proj,
                        dimensions=dimensions,
                        output_dtype=output_dtype,
                        normalize=normalize,
                    )
        else:
            uniform_pos_len = int(lengths[0]) if int(lengths.min()) == int(lengths.max()) > 0 else 0
            eff_bs = min(bs, max(1, 8192 // uniform_pos_len)) if uniform_pos_len > 512 else bs
            order = np.argsort(lengths, kind="stable")
            for start in range(0, len(encoded_docs), eff_bs):
                idxs = order[start : start + eff_bs]
                rows = [encoded_docs[int(i)][0] for i in idxs]
                spans_list = [encoded_docs[int(i)][1] for i in idxs]
                input_ids, attention_mask = self._prepare_batch_inputs_and_mask(rows)
                max_c = max(1, *(len(s) for s in spans_list))
                span_arr = np.zeros((len(idxs), max_c, 2), dtype=np.int32)
                for row_i, sp in enumerate(spans_list):
                    for col_j, (t0, t1) in enumerate(sp):
                        span_arr[row_i, col_j] = (t0, t1)

                hidden = self.forward_hidden(input_ids, attention_mask)
                proj = self._project_span_pool(hidden, span_arr)
                for row_i, orig_idx in enumerate(idxs):
                    n_chunks = len(spans_list[row_i])
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
    "dequantize_gdn_state_i8",
    "gated_delta_ops",
    "gated_delta_update",
    "get_mlx_runtime_telemetry",
    "normalize_qk",
    "normalize_state_dtype",
    "quantize_gdn_state_i8",
    "set_mlx_cache_limit",
    "tile_aligned_slices",
    "wired_limit_scope",
]
