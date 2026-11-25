"""Standalone Gemma-4 E2B/E4B/12B/26B-A4B/31B MLX + Metal 4 MPP model loader and ragged runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Sequence
import warnings

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from .kernels import (
    fused_geglu_split,
    fused_kv_only_norm_rope_ragged,
    fused_norm_add_norm,
    fused_ple_geglu_indexed,
    fused_q_norm_rope_ragged,
    fused_qkv_packed_norm_rope,
    fused_qkv_packed_norm_rope_ragged,
    fused_res_rms_norm,
    fused_softcap_argmax_bf16,
    gather_dequant_kv_pair_i8_bf16,
    mpp_fused_geglu_bf16,
    mpp_linear_bf16,
    mqa8_sdpa_prefix_ragged,
    mqa8_sdpa_prefix_ragged_slot,
    multi_row_gemv_exact_bf16,
    multi_row_qmv_exact_bf16,
    quantize_kv_pair_i8,
    scatter_kv_i8_slot_precomputed,
    scatter_kv_ragged,
    scatter_kv_slot_precomputed,
)
from .packager import is_ple_gate_or_proj_key, quantize_q4_0_sym_u32
from .paths import Gemma4Paths, resolve_paths
from .tokenizer import DEFAULT_KEPT_VOCAB_SIZE, FULL_VOCAB_SIZE, VocabRemapper

try:
    from fast_shared.metal_mpp_int8 import (
        get_or_pack_affine_module_weights,
        mpp_quantized_linear,
        should_use_mpp_int8_linear,
    )
except ImportError:
    get_or_pack_affine_module_weights = None  # type: ignore[assignment]
    mpp_quantized_linear = None  # type: ignore[assignment]
    should_use_mpp_int8_linear = None  # type: ignore[assignment]


def _macos_version_tuple() -> tuple[int, int]:
    ver_str = platform.mac_ver()[0]
    if not ver_str:
        return (0, 0)
    parts = ver_str.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except ValueError:
        return (0, 0)


def _supports_mpp_bf16() -> bool:
    """Metal 4 MPP BF16 tensor operands require macOS >= 26.1."""
    return _macos_version_tuple() >= (26, 1)


def _detect_gpu_family() -> str:
    """Classify Apple Silicon GPU family (`apple9` for M4, `apple10` for M5+/NAX) via `fast_shared.hw_check`."""
    try:
        from fast_shared.hw_check import probe_host_hardware, supports_apple10_nax

        hw = probe_host_hardware()
        if supports_apple10_nax(hw):
            return "apple10"
        fam = str(hw.get("gpu_family") or "").lower()
        arch = str(hw.get("mlx_architecture") or "").lower()
        name = str(hw.get("brand") or "").lower()
        if "apple10" in fam or "m5" in name or "m6" in name or "g17" in arch:
            return "apple10"
        if "apple9" in fam or "m4" in name or "g16" in arch:
            return "apple9"
    except Exception:
        pass
    return "unknown"


def _detect_host_ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_sz = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_sz, int) and pages > 0 and page_sz > 0:
            return float(pages * page_sz) / float(1024**3)
    except Exception:
        pass
    try:
        dev_info_fn = getattr(mx, "device_info", None)
        if callable(dev_info_fn):
            mem = int(dev_info_fn().get("memory_size", 0) or 0)
            if mem > 0:
                return float(mem) / float(1024**3)
    except Exception:
        pass
    return 48.0


def should_use_bf16_mpp_for_mlp(
    *,
    mlp_dtype: str,
    gpu_family: str,
    host_ram_gib: float,
    batch_size: int,
    mlp_bits: int | None,
) -> bool:
    """Hardware- and checkpoint-aware policy deciding whether MLPs should promote to BF16 MPP.

    Under `mlp_dtype="auto"`:
      - `Apple10` (`NAX`, M5+): NEVER promote (`QuantizedLinear` wins across `B=1..128`).
      - `Apple9` (M4): keep `QuantizedLinear` whenever `B < 64`, `RAM < 48 GB`, or `mlp_bits <= 4`
        (`q4_0` / `qat-mobile` / `mxfp4`); promote ONLY when `B >= 64` on `>= 48 GB` `Apple9`
        and weights are `8-bit` or BF16-backed (`mlp_bits is None or mlp_bits >= 8`).
    """
    mode = str(mlp_dtype or "auto").strip().lower()
    fam = str(gpu_family or "unknown").strip().lower()
    if mode == "quantized":
        return False
    if mode == "bfloat16":
        return fam != "apple10"
    if mode != "auto":
        raise ValueError(f"Unsupported mlp_dtype={mlp_dtype!r}")
    if fam != "apple9":
        return False
    if int(batch_size) < 64:
        return False
    if float(host_ram_gib) < 46.0:
        return False
    if mlp_bits is not None and int(mlp_bits) <= 4:
        return False
    return True


def singleton_quantized_linear(linear: nn.Module, x: mx.array) -> mx.array | None:
    """Execute single-pass weight-reusing per-token `qmv` across `[B, L_q, D]` (`0.0` ULP vs `L_q=1`)."""
    if not isinstance(linear, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return None
    if x.ndim != 3:
        return None
    bsz, seq_len, in_dim = (int(d) for d in x.shape)
    m_rows = bsz * seq_len
    if m_rows == 1:
        if isinstance(linear, nn.QuantizedLinear):
            return linear(x)
        out_1 = mx.quantized_matmul(
            x,
            linear.weight,
            scales=linear.scales,
            biases=linear.biases,
            transpose=True,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
        bias = getattr(linear, "bias", None)
        return (out_1 + bias) if bias is not None else out_1

    if mx.default_device() == mx.gpu and linear.mode == "affine" and linear.biases is not None:
        fast_out = multi_row_qmv_exact_bf16(
            x,
            linear.weight,
            linear.scales,
            linear.biases,
            group_size=int(linear.group_size),
            bits=int(linear.bits),
        )
        if fast_out is not None:
            bias = getattr(linear, "bias", None)
            return (fast_out + bias) if bias is not None else fast_out

    lhs_idx = mx.arange(m_rows, dtype=mx.uint32)
    rhs_idx = mx.zeros((m_rows,), dtype=mx.uint32)
    projected = mx.contiguous(x).reshape((m_rows, 1, 1, in_dim))
    out = mx.gather_qmm(  # guard: allow-unsorted-qmm
        projected,
        linear.weight[None],
        linear.scales[None],
        None if linear.biases is None else linear.biases[None],
        lhs_indices=lhs_idx,
        rhs_indices=rhs_idx,
        transpose=True,
        group_size=linear.group_size,
        bits=linear.bits,
        mode=linear.mode,
        sorted_indices=False,
    )
    out = out.reshape((bsz, seq_len, -1))
    bias = getattr(linear, "bias", None)
    if bias is not None:
        out = out + bias
    return out


def _singleton_verify_linear(mod: nn.Module, x: mx.array) -> mx.array:
    """Apply `mod` to `[B, L_q, D]` using single-pass weight-reusing reduction matching `L_q=1` bit-for-bit."""
    q_out = singleton_quantized_linear(mod, x)
    if q_out is not None:
        return q_out
    if isinstance(mod, nn.Linear) and x.ndim == 3:
        m_rows = int(x.shape[0]) * int(x.shape[1])
        if m_rows == 1:
            return mod(x)
        if mx.default_device() == mx.gpu:
            fast_lin = multi_row_gemv_exact_bf16(x, mod.weight)
            if fast_lin is not None:
                if "bias" in mod and mod.bias is not None:
                    return fast_lin + mod.bias
                return fast_lin
        out = mx.matmul(mx.expand_dims(mx.contiguous(x), -2), mod.weight.T).squeeze(-2)
        if "bias" in mod and mod.bias is not None:
            out = out + mod.bias
        return out
    if isinstance(mod, nn.Embedding) and x.ndim == 3:
        if mx.default_device() == mx.gpu and int(x.shape[0]) * int(x.shape[1]) > 1:
            fast_emb = multi_row_gemv_exact_bf16(x, mod.weight)
            if fast_emb is not None:
                return fast_emb
        return mx.matmul(mx.expand_dims(mx.contiguous(x), -2), mod.weight.T).squeeze(-2)
    return mod(x)


def _next_pow2(n: int) -> int:
    v = max(1, int(n))
    return 1 << (v - 1).bit_length()


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: int | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    """Additive `0 / -inf` lower-triangular causal (+ optional sliding window) attention mask in `dtype`."""
    rinds = mx.arange(offset + N, dtype=mx.int32)[None, :]
    linds = mx.arange(offset, offset + N, dtype=mx.int32)[:, None]
    vis = linds >= rinds
    if window_size is not None:
        vis = vis & (linds < (rinds + int(window_size)))
    zero = mx.array(0.0, dtype=dtype)
    neg_inf = mx.array(-float("inf"), dtype=dtype)
    return mx.where(vis, zero, neg_inf)


def _select_sdpa_mask(
    lc: int,
    offset: int,
    *,
    is_full: bool,
    window_size: int | None,
    dtype: mx.Dtype,
) -> str | mx.array | None:
    """Select fastest non-bool SDPA mask: `'causal'` for full/unwindowed `lc > 1`, `None` for `lc == 1`, `0/-inf` `dtype` for SWA."""
    if lc <= 1:
        return None
    if is_full or window_size is None:
        return "causal"
    if offset + lc <= int(window_size):
        return "causal"
    return create_causal_mask(lc, offset=offset, window_size=window_size, dtype=dtype)


def safe_argmax_with_nan_guard(logits: mx.array, axis: int = -1) -> mx.array:
    """Compute argmax along `axis`, returning `-1` (`int32`) for any row containing `NaN`."""
    am = mx.argmax(logits, axis=axis).astype(mx.int32)
    isnan_fn = getattr(mx, "isnan", None)
    any_fn = getattr(mx, "any", None)
    where_fn = getattr(mx, "where", None)
    if callable(isnan_fn) and callable(any_fn) and callable(where_fn):
        has_nan = any_fn(isnan_fn(logits), axis=axis)
        neg1 = mx.full(am.shape, -1, dtype=mx.int32)
        am = where_fn(has_nan, neg1, am)
    return am


def compute_safe_fp16_residual_scales(
    num_early_layers: int,
    w_pg_max_abs: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute power-of-2 per-layer residual scales keeping Gemma-4 `L13` (`215K`) safely below FP16 max (`65,504`)."""
    scales = np.ones((int(num_early_layers),), dtype=np.float32)
    if num_early_layers < 12:
        return scales
    # Gemma-4 residual norms surge around L11..L14 (`~215K > 65,504` at L13).
    # Stepping to `0.0625` (`1/16`, exact power-of-2) brings 215K down to `13,437 << 65,504`.
    for i in range(num_early_layers):
        if i == 10:
            target = 0.5
        elif i == 11:
            target = 0.25
        elif i >= 12:
            target = 0.0625
        else:
            target = 1.0
        if w_pg_max_abs is not None and i < len(w_pg_max_abs) and w_pg_max_abs[i] > 0.0:
            # Ensure folded `w_pg / alpha` stays comfortably within FP16 weight range (<= 4096.0).
            while target < 1.0 and (float(w_pg_max_abs[i]) / target) > 4096.0:
                target *= 2.0
        scales[i] = float(target)
    return scales


@mx.compile
def logit_softcap(softcap: float, x: mx.array) -> mx.array:
    inv_sc = 1.0 / float(softcap)
    return mx.tanh(x.astype(mx.float32) * inv_sc) * float(softcap)


def _candidate_quant_names(prefix: str) -> tuple[str, ...]:
    cands = [prefix]
    if prefix.startswith("language_model.model."):
        tail = prefix[len("language_model.model.") :]
        cands.extend((f"language_model.{tail}", f"model.{tail}", tail))
    elif prefix.startswith("model."):
        tail = prefix[len("model.") :]
        cands.extend((f"language_model.model.{tail}", f"language_model.{tail}", tail))
    elif prefix.startswith("language_model."):
        tail = prefix[len("language_model.") :]
        cands.extend((f"language_model.model.{tail}", tail))
    return tuple(dict.fromkeys(cands))


def _resolve_quant_spec(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    *,
    expected_in_f: int | None = None,
) -> tuple[int, int, str]:
    """Resolve `(bits, group_size, mode)` from explicit per-module overrides, regexes (`qat-mobile`), and tensor shapes."""
    w = raw[f"{prefix}.weight"]
    s = raw[f"{prefix}.scales"]
    has_biases = f"{prefix}.biases" in raw

    meta: dict[str, Any] = {}
    for cand in _candidate_quant_names(prefix):
        v = qcfg.get(cand)
        if isinstance(v, dict):
            meta = v
            break

    if not meta and isinstance(qcfg.get("module_quant_configs"), dict):
        mqc: dict[str, Any] = qcfg["module_quant_configs"]
        for pat, pat_cfg in mqc.items():
            if not isinstance(pat_cfg, dict):
                continue
            if any(re.search(pat, cand) for cand in _candidate_quant_names(prefix)):
                meta = pat_cfg
                break

    u8_dt = getattr(mx, "uint8", "uint8")
    raw_mode = meta.get("mode", qcfg.get("mode"))
    if raw_mode in ("affine", "mxfp4", "mxfp8", "nvfp4"):
        mode = str(raw_mode)
    elif s.dtype == u8_dt and not has_biases:
        mode = "mxfp4"
    else:
        mode = "affine"

    default_gs = 32 if mode in ("mxfp4", "mxfp8") else (16 if mode == "nvfp4" else 64)
    gs = int(meta.get("group_size", qcfg.get("group_size", default_gs)))
    bits = int(meta.get("bits", meta.get("num_bits", qcfg.get("bits", qcfg.get("num_bits", 4)))))

    w_cols = int(w.shape[-1])
    s_cols = int(s.shape[-1])
    if expected_in_f is not None and expected_in_f > 0 and s_cols > 0 and w_cols > 0:
        if expected_in_f % s_cols == 0 and (w_cols * 32) % expected_in_f == 0:
            cand_gs = expected_in_f // s_cols
            cand_bits = (w_cols * 32) // expected_in_f
            if cand_bits in (2, 3, 4, 5, 6, 8) and cand_gs in (16, 32, 64, 128, 256):
                gs = cand_gs
                bits = cand_bits
    elif s_cols > 0 and w_cols > 0:
        # Even without `expected_in_f`, `in_f = s_cols * gs = w_cols * (32 / bits)` -> `bits * gs = (32 * w_cols) / s_cols`.
        if (32 * w_cols) % s_cols == 0:
            prod = (32 * w_cols) // s_cols
            if bits * gs != prod:
                if prod % gs == 0 and (prod // gs) in (2, 3, 4, 6, 8):
                    bits = prod // gs
                elif prod % bits == 0 and (prod // bits) in (16, 32, 64, 128, 256):
                    gs = prod // bits

    if s.dtype == u8_dt and not has_biases and mode == "affine":
        mode = "mxfp8" if bits == 8 else ("nvfp4" if gs == 16 else "mxfp4")
    return bits, gs, mode


def _cast_quant_scales(scales: mx.array, mode: str, dtype: mx.Dtype) -> mx.array:
    if mode in ("mxfp4", "mxfp8", "nvfp4") or scales.dtype == getattr(mx, "uint8", "uint8"):
        return scales
    return scales.astype(dtype)


def _quantize_unpacked_qat_linear(w: mx.array, prefix: str, dtype: mx.Dtype) -> nn.QuantizedLinear:
    """Quantize unpacked `*-qat-q4_0-unquantized` 2D linear onto symmetric `q4_0 g32` (or `8-bit` PLE gate/proj)."""
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if is_ple_gate_or_proj_key(f"{prefix}.weight") and in_f % 32 == 0:
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight = w.astype(dtype)
        return nn.QuantizedLinear.from_linear(lin, group_size=32, bits=8, mode="affine")
    w_np = np.asarray(w.astype(mx.float32), dtype=np.float32)
    qw, qs, qb = quantize_q4_0_sym_u32(w_np, group_size=32)
    qlin = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=32, bits=4, mode="affine")
    qlin.weight = mx.array(qw.array, dtype=mx.uint32)
    qlin.scales = mx.array(qs.to_float32(), dtype=dtype)
    qlin.biases = mx.array(qb.to_float32(), dtype=dtype)
    return qlin


def _quantize_unpacked_qat_embedding(w: mx.array, dtype: mx.Dtype) -> nn.QuantizedEmbedding:
    num_emb, dims = int(w.shape[0]), int(w.shape[1])
    w_np = np.asarray(w.astype(mx.float32), dtype=np.float32)
    qw, qs, qb = quantize_q4_0_sym_u32(w_np, group_size=32)
    qemb = nn.QuantizedEmbedding(num_emb, dims, group_size=32, bits=4, mode="affine")
    qemb.weight = mx.array(qw.array, dtype=mx.uint32)
    qemb.scales = mx.array(qs.to_float32(), dtype=dtype)
    qemb.biases = mx.array(qb.to_float32(), dtype=dtype)
    return qemb


def _dequantize_linear(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    expected_in_f: int | None = None,
) -> mx.array:
    w_key = f"{prefix}.weight"
    if f"{prefix}.scales" not in raw:
        return raw[w_key].astype(dtype)
    bits, gs, mode = _resolve_quant_spec(raw, prefix, qcfg, expected_in_f=expected_in_f)
    scales = _cast_quant_scales(raw[f"{prefix}.scales"], mode, dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    return mx.dequantize(
        raw[w_key],
        scales,
        biases,
        group_size=gs,
        bits=bits,
        mode=mode,
    ).astype(dtype)


def _load_linear(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    promote_bf16: bool = False,
    expected_in_f: int | None = None,
    auto_qat_q4_0: bool = False,
) -> nn.Module:
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.scales"
    if s_key not in raw:
        w = raw[w_key].astype(dtype)
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if auto_qat_q4_0 and not promote_bf16 and in_f % 32 == 0:
            return _quantize_unpacked_qat_linear(w, prefix, dtype)
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight = w
        return lin

    if promote_bf16:
        w = _dequantize_linear(raw, prefix, qcfg, dtype, expected_in_f=expected_in_f)
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight = w
        return lin

    bits, gs, mode = _resolve_quant_spec(raw, prefix, qcfg, expected_in_f=expected_in_f)
    scales = _cast_quant_scales(raw[s_key], mode, dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    out_f = int(scales.shape[0])
    in_f = int(scales.shape[1]) * gs
    qlin = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=gs, bits=bits, mode=mode)
    qlin.weight = raw[w_key]
    qlin.scales = scales
    qlin.biases = biases
    return qlin


def _load_qembedding(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    expected_in_f: int | None = None,
    auto_qat_q4_0: bool = False,
) -> nn.Module:
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.scales"
    if s_key not in raw:
        w = raw[w_key].astype(dtype)
        if auto_qat_q4_0 and int(w.shape[1]) % 32 == 0:
            return _quantize_unpacked_qat_embedding(w, dtype)
        emb = nn.Embedding(int(w.shape[0]), int(w.shape[1]))
        emb.weight = w
        return emb
    bits, gs, mode = _resolve_quant_spec(raw, prefix, qcfg, expected_in_f=expected_in_f)
    scales = _cast_quant_scales(raw[s_key], mode, dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    num_emb = int(scales.shape[0])
    dims = int(scales.shape[1]) * gs
    qemb = nn.QuantizedEmbedding(num_emb, dims, group_size=gs, bits=bits, mode=mode)
    qemb.weight = raw[w_key]
    qemb.scales = scales
    qemb.biases = biases
    return qemb


class _SwitchProj:
    """3D expert projection (`[num_experts, out_f, in_f]`), supporting quantized (`affine`/`mxfp4`) and dense BF16."""

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array | None = None,
        biases: mx.array | None = None,
        *,
        group_size: int = 32,
        bits: int = 4,
        mode: str = "affine",
    ) -> None:
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)
        self.is_quantized = scales is not None

    def parameters(self) -> list[mx.array]:
        arrs = [self.weight]
        if self.scales is not None:
            arrs.append(self.scales)
        if self.biases is not None:
            arrs.append(self.biases)
        return arrs

    def __call__(
        self,
        x: mx.array,
        *,
        lhs_indices: mx.array,
        rhs_indices: mx.array,
        sorted_indices: bool = True,
    ) -> mx.array:
        if self.is_quantized:
            return mx.gather_qmm(
                x,
                self.weight,
                self.scales,
                self.biases,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        return mx.gather_mm(
            x,
            self.weight.swapaxes(-1, -2),
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            sorted_indices=sorted_indices,
        )


def _load_3d_switch_proj(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    expected_in_f: int,
) -> _SwitchProj:
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.scales"
    if s_key not in raw:
        return _SwitchProj(raw[w_key].astype(dtype))
    bits, gs, mode = _resolve_quant_spec(raw, prefix, qcfg, expected_in_f=expected_in_f)
    scales = _cast_quant_scales(raw[s_key], mode, dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    return _SwitchProj(raw[w_key], scales, biases, group_size=gs, bits=bits, mode=mode)


def _concat_switch_projs(p_a: _SwitchProj, p_b: _SwitchProj) -> _SwitchProj:
    return _SwitchProj(
        mx.contiguous(mx.concatenate([p_a.weight, p_b.weight], axis=1)),
        mx.contiguous(mx.concatenate([p_a.scales, p_b.scales], axis=1)) if p_a.scales is not None else None,
        mx.contiguous(mx.concatenate([p_a.biases, p_b.biases], axis=1)) if p_a.biases is not None else None,
        group_size=p_a.group_size,
        bits=p_a.bits,
        mode=p_a.mode,
    )


class Gemma4Router:
    """Gemma-4 MoE expert router (`rms_norm(x, scale * H^-0.5) -> proj -> top-k -> softmax * per_expert_scale`)."""

    def __init__(
        self,
        proj: nn.Module,
        scale: mx.array,
        per_expert_scale: mx.array,
        *,
        hidden_size: int,
        top_k: int,
        eps: float,
        dtype: mx.Dtype,
    ) -> None:
        self.proj = proj
        self.scale = scale.astype(dtype)
        self.per_expert_scale = per_expert_scale.astype(dtype)
        self.per_expert_scale_f32 = per_expert_scale.astype(mx.float32)
        self.scaled_norm_w_f32 = scale.astype(mx.float32) * (float(hidden_size) ** -0.5)
        self.scaled_norm_w = self.scaled_norm_w_f32.astype(dtype)
        self.top_k = int(top_k)
        self.eps = float(eps)

    def parameters(self) -> list[Any]:
        return [self.proj.parameters(), self.scale, self.per_expert_scale, self.scaled_norm_w]

    def __call__(self, x: mx.array, *, singleton: bool = False) -> tuple[mx.array, mx.array]:
        x_f32 = x.astype(mx.float32)
        x_norm = mx.fast.rms_norm(x_f32, self.scaled_norm_w_f32, self.eps)
        if isinstance(self.proj, nn.Linear):
            w_f32 = self.proj.weight.astype(mx.float32)
            scores = mx.matmul(x_norm, w_f32.T)
            if "bias" in self.proj and self.proj.bias is not None:
                scores = scores + self.proj.bias.astype(mx.float32)
        elif isinstance(self.proj, nn.QuantizedLinear):
            scores = self.proj(x_norm).astype(mx.float32)
        else:
            scores = (_singleton_verify_linear(self.proj, x_norm) if singleton else self.proj(x_norm)).astype(mx.float32)
        top_k_idx = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        top_k_w = mx.softmax(mx.take_along_axis(scores, top_k_idx, axis=-1), axis=-1, precise=True)
        top_k_w = top_k_w * self.per_expert_scale_f32[top_k_idx]
        return top_k_idx, top_k_w


class Gemma4SwitchGLU:
    """Sparse MoE SwitchGLU using fused `gate_up_proj` + `fused_geglu_split` + sorted `lhs_indices`/`rhs_indices` `gather_qmm`."""

    def __init__(self, gate_up_proj: _SwitchProj, down_proj: _SwitchProj) -> None:
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj

    def parameters(self) -> list[Any]:
        return [self.gate_up_proj.parameters(), self.down_proj.parameters()]

    def _forward_tokens(self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array) -> mx.array:
        orig_idx_shape = top_k_indices.shape
        k_top = int(orig_idx_shape[-1])
        in_dim = int(x.shape[-1])
        m_tokens = int(x.size // in_dim)
        x_flat = mx.contiguous(x).reshape((m_tokens, 1, 1, in_dim))

        flat_experts = top_k_indices.reshape((-1,))
        order = mx.argsort(flat_experts)
        inv_order = mx.argsort(order)
        rhs_indices = flat_experts[order]
        lhs_indices = order // k_top

        gu = self.gate_up_proj(
            x_flat,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            sorted_indices=True,
        )
        act = fused_geglu_split(gu)
        down_lhs = mx.arange(int(rhs_indices.size), dtype=order.dtype)
        y_sorted = self.down_proj(
            act,
            lhs_indices=down_lhs,
            rhs_indices=rhs_indices,
            sorted_indices=True,
        )
        y = y_sorted[inv_order].reshape((*orig_idx_shape, -1))
        return (mx.expand_dims(top_k_weights.astype(mx.float32), -1) * y.astype(mx.float32)).sum(axis=-2).astype(
            y.dtype
        )

    def __call__(
        self,
        x: mx.array,
        top_k_indices: mx.array,
        top_k_weights: mx.array,
        *,
        singleton: bool = False,
    ) -> mx.array:
        if singleton and x.ndim == 3 and int(x.shape[1]) > 1:
            cols = [
                self._forward_tokens(x[:, t : t + 1], top_k_indices[:, t : t + 1], top_k_weights[:, t : t + 1])
                for t in range(int(x.shape[1]))
            ]
            return mx.concatenate(cols, axis=1)
        return self._forward_tokens(x, top_k_indices, top_k_weights)


def _load_moe_experts(
    raw: dict[str, mx.array],
    lp: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    hidden_size: int,
    moe_interm: int,
    num_experts: int,
) -> Gemma4SwitchGLU:
    sg_gate = f"{lp}.experts.switch_glu.gate_proj"
    sg_up = f"{lp}.experts.switch_glu.up_proj"
    sg_down = f"{lp}.experts.switch_glu.down_proj"
    if f"{sg_gate}.weight" in raw:
        p_g = _load_3d_switch_proj(raw, sg_gate, qcfg, dtype, expected_in_f=hidden_size)
        p_u = _load_3d_switch_proj(raw, sg_up, qcfg, dtype, expected_in_f=hidden_size)
        p_d = _load_3d_switch_proj(raw, sg_down, qcfg, dtype, expected_in_f=moe_interm)
        return Gemma4SwitchGLU(_concat_switch_projs(p_g, p_u), p_d)

    fused_gu = f"{lp}.experts.gate_up_proj"
    fused_dn = f"{lp}.experts.down_proj"
    if f"{fused_gu}.weight" in raw:
        p_gu = _load_3d_switch_proj(raw, fused_gu, qcfg, dtype, expected_in_f=hidden_size)
        p_d = _load_3d_switch_proj(raw, fused_dn, qcfg, dtype, expected_in_f=moe_interm)
        return Gemma4SwitchGLU(p_gu, p_d)

    # Per-expert keys (`experts.0.gate_proj`, ..., `experts.{E-1}.down_proj`, including REAP pruned counts)
    e_cnt = 0
    while f"{lp}.experts.{e_cnt}.gate_proj.weight" in raw and e_cnt < max(num_experts, 1024):
        e_cnt += 1
    if e_cnt == 0:
        raise KeyError(f"Missing MoE expert weights under {lp}.experts")

    def _stack_expert(proj_name: str, exp_in: int) -> _SwitchProj:
        p0 = f"{lp}.experts.0.{proj_name}"
        has_sc = f"{p0}.scales" in raw
        has_bi = f"{p0}.biases" in raw
        w_st = mx.stack([raw[f"{lp}.experts.{e}.{proj_name}.weight"] for e in range(e_cnt)], axis=0)
        if not has_sc:
            return _SwitchProj(w_st.astype(dtype))
        bits, gs, mode = _resolve_quant_spec(raw, p0, qcfg, expected_in_f=exp_in)
        s_st = _cast_quant_scales(
            mx.stack([raw[f"{lp}.experts.{e}.{proj_name}.scales"] for e in range(e_cnt)], axis=0), mode, dtype
        )
        b_st = (
            mx.stack([raw[f"{lp}.experts.{e}.{proj_name}.biases"] for e in range(e_cnt)], axis=0).astype(dtype)
            if has_bi
            else None
        )
        return _SwitchProj(w_st, s_st, b_st, group_size=gs, bits=bits, mode=mode)

    p_g = _stack_expert("gate_proj", hidden_size)
    p_u = _stack_expert("up_proj", hidden_size)
    p_d = _stack_expert("down_proj", moe_interm)
    return Gemma4SwitchGLU(_concat_switch_projs(p_g, p_u), p_d)


def _fuse_linears(mods: Sequence[nn.Module], dtype: mx.Dtype) -> nn.Module:
    first = mods[0]
    if all(
        isinstance(m, nn.QuantizedLinear)
        and m.bits == first.bits
        and m.group_size == first.group_size
        and m.mode == first.mode
        and (m.biases is None) == (first.biases is None)
        for m in mods
    ):
        in_f = int(first.scales.shape[1]) * int(first.group_size)
        out_f = sum(int(m.weight.shape[0]) for m in mods)
        fused = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=first.group_size, bits=first.bits, mode=first.mode)
        fused.weight = mx.concatenate([m.weight for m in mods], axis=0)
        fused.scales = mx.concatenate([m.scales for m in mods], axis=0)
        fused.biases = mx.concatenate([m.biases for m in mods], axis=0) if first.biases is not None else None
        return fused
    weights = []
    for m in mods:
        if isinstance(m, nn.QuantizedLinear):
            weights.append(
                mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits, mode=m.mode).astype(dtype)
            )
        else:
            weights.append(m.weight.astype(dtype))
    fused = nn.Linear(int(weights[0].shape[1]), sum(int(w.shape[0]) for w in weights), bias=False)
    fused.weight = mx.concatenate(weights, axis=0)
    return fused


def _slice_q_from_qkv(qkv_proj: nn.Module, q_dim: int) -> nn.Module:
    """Extract [q_proj] rows [:q_dim] from fused qkv_proj without losing quantization."""
    if isinstance(qkv_proj, nn.QuantizedLinear):
        in_f = int(qkv_proj.scales.shape[1]) * int(qkv_proj.group_size)
        q_mod = nn.QuantizedLinear(
            in_f, q_dim, bias=False, group_size=qkv_proj.group_size, bits=qkv_proj.bits, mode=qkv_proj.mode
        )
        q_mod.weight = qkv_proj.weight[:q_dim]
        q_mod.scales = qkv_proj.scales[:q_dim]
        q_mod.biases = qkv_proj.biases[:q_dim] if qkv_proj.biases is not None else None
        return q_mod
    w = qkv_proj.weight[:q_dim]
    q_mod = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    q_mod.weight = w
    return q_mod


def _slice_kv_from_qkv(qkv_proj: nn.Module, q_dim: int) -> nn.Module:
    """Extract [k_proj; v_proj] rows [q_dim:] from fused qkv_proj without losing quantization."""
    if isinstance(qkv_proj, nn.QuantizedLinear):
        in_f = int(qkv_proj.scales.shape[1]) * int(qkv_proj.group_size)
        out_f = int(qkv_proj.weight.shape[0]) - q_dim
        kv_q = nn.QuantizedLinear(
            in_f, out_f, bias=False, group_size=qkv_proj.group_size, bits=qkv_proj.bits, mode=qkv_proj.mode
        )
        kv_q.weight = qkv_proj.weight[q_dim:]
        kv_q.scales = qkv_proj.scales[q_dim:]
        kv_q.biases = qkv_proj.biases[q_dim:] if qkv_proj.biases is not None else None
        return kv_q
    w = qkv_proj.weight[q_dim:]
    kv = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    kv.weight = w
    return kv


def _slice_to_bf16_padded(mod: nn.Module, pad_mult: int = 64) -> mx.array:
    if isinstance(mod, nn.QuantizedLinear):
        w = mx.dequantize(
            mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode
        ).astype(mx.bfloat16)
    else:
        w = mod.weight.astype(mx.bfloat16)
    N = int(w.shape[0])
    pad_n = (pad_mult - (N % pad_mult)) % pad_mult
    if pad_n:
        w = mx.pad(w, [(0, pad_n), (0, 0)])
    return w


def _build_rope_inv_freq(tcfg: dict[str, Any], is_full: bool) -> mx.array:
    rope_map = tcfg.get("rope_parameters", {})
    if is_full:
        D = int(tcfg["global_head_dim"])
        p = rope_map.get("full_attention", {})
        theta = float(p.get("rope_theta", 1_000_000.0))
        factor = float(p.get("partial_rotary_factor", 0.25))
        rotary_dims = int(D * factor)
        inv_rot = 1.0 / (theta ** (mx.arange(0, rotary_dims, 2, dtype=mx.float32) / float(D)))
        inv_zero = mx.zeros(((D - rotary_dims) // 2,), dtype=mx.float32)
        inv = mx.concatenate([inv_rot, inv_zero], axis=0).astype(mx.float32)
    else:
        D = int(tcfg["head_dim"])
        p = rope_map.get("sliding_attention", {})
        theta = float(p.get("rope_theta", 10_000.0))
        inv = (1.0 / (theta ** (mx.arange(0, D, 2, dtype=mx.float32) / float(D)))).astype(mx.float32)
    mx.eval(inv)
    return inv


def _compute_previous_kvs(layer_types: Sequence[str], num_kv_shared: int) -> list[int]:
    n = len(layer_types)
    first_shared = n - num_kv_shared
    prev = list(range(n))
    if num_kv_shared > 0 and first_shared > 0:
        by_type: dict[str, int] = {}
        for i in range(first_shared):
            by_type[layer_types[i]] = i
        for j in range(first_shared, n):
            prev[j] = by_type[layer_types[j]]
    return prev


class FastGemma4Model:
    """
    Standalone hardware-limit Gemma-4 E2B/E4B/12B/26B-A4B/31B runner for Apple Silicon (M4 / M5+):
      - Zero `transformers`/`torch`/`mlx_vlm` runtime dependencies
      - Shared `KV_sys` prefix broadcast across KV-owning layers + ragged per-doc `KV_doc` `[B, Cap, n_kv, D]`
      - Exact power-of-2 bitmask SWA ring buffer + Full KV linear buffer
      - `EarlyExit` prefill with final KV-owning layer `kv_proj_only`
      - Hardware- & checkpoint-aware `mlp_dtype="auto"` (`QuantizedLinear` across `B=1..128` on M5 NAX and `<48GB` / `<=4-bit` / `B<64` on M4; lazy BF16 MPP promotion only on `>=48GB` M4 `B>=64` with `8-bit`/BF16 MLPs)
      - Full checkpoint `262,144` LM head by default, with optional domain/SFT `vocab_remap.npz` or explicit `allow_english_vocab_prune=True`
    """

    def __init__(
        self,
        *,
        paths: Gemma4Paths | None = None,
        model_dir: str | Path | None = None,
        mlp_dtype: str = "auto",
        vocab_out: int | None = None,
        allow_english_vocab_prune: bool = False,
        use_mpp: bool = True,
        lm_head_mode: str = "q4",
        local_files_only: bool = True,
        kv_i8: bool | str = "auto",
        vec4_reduce: bool = True,
        mpp_m32_cutoff: int = 96,
        use_fused_softcap_argmax: bool = True,
        allow_mpp_on_apple10: bool = False,
    ) -> None:
        if mlp_dtype not in ("auto", "bfloat16", "quantized"):
            raise ValueError(f"Unsupported mlp_dtype={mlp_dtype!r}")
        if lm_head_mode not in ("q4", "bf16_mpp"):
            raise ValueError(f"Unsupported lm_head_mode={lm_head_mode!r}")
        if not isinstance(kv_i8, bool) and str(kv_i8).strip().lower() != "auto":
            raise ValueError(f"Unsupported kv_i8={kv_i8!r}; expected bool or 'auto'")
        if vocab_out is not None and int(vocab_out) <= 0:
            raise ValueError(f"vocab_out must be a positive integer (or None), got {vocab_out!r}")

        self.paths = paths or resolve_paths()
        self.model_dir = self.paths.resolve_model_dir(model_dir, local_files_only=local_files_only)
        self.tokenizer_json = self.paths.resolve_tokenizer_json(self.model_dir, local_files_only=local_files_only)
        self.mlp_dtype_name = str(mlp_dtype)
        self.allow_english_vocab_prune = bool(allow_english_vocab_prune)
        self.act_dtype = mx.bfloat16
        self._macos_version = _macos_version_tuple()
        self._mpp_bf16_supported = _supports_mpp_bf16()
        self._gpu_family = _detect_gpu_family()
        self._host_ram_gib = _detect_host_ram_gib()
        self._mpp_apple10_enabled = bool(allow_mpp_on_apple10)
        self._mpp_prefill_min_m = 2048
        self._lm_head_mpp_min_m = 64
        self._use_mpp_requested = bool(use_mpp)
        self.use_mpp = bool(use_mpp and (mlp_dtype in ("bfloat16", "auto")) and self._mpp_bf16_supported)
        self._mpp_apple10_int8_enabled = bool(
            self._use_mpp_requested and self._gpu_family == "apple10" and mpp_quantized_linear is not None
        )
        self.lm_head_mode = lm_head_mode
        if isinstance(kv_i8, bool):
            self.kv_i8_mode = "explicit"
            self.kv_i8 = kv_i8
        else:
            self.kv_i8_mode = "auto"
            self.kv_i8 = False
        self.vec4_reduce = bool(vec4_reduce)
        self.mpp_m32_cutoff = int(mpp_m32_cutoff)
        self.use_fused_softcap_argmax = bool(use_fused_softcap_argmax)

        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        tcfg = cfg.get("text_config", cfg)
        qcfg: dict[str, Any] = {}
        if isinstance(cfg.get("quantization_config"), dict):
            qcfg.update(cfg["quantization_config"])
        if isinstance(cfg.get("quantization"), dict):
            qcfg.update(cfg["quantization"])

        self.num_layers = int(tcfg["num_hidden_layers"])
        self.hidden_size = int(tcfg["hidden_size"])
        self.intermediate_size = int(tcfg.get("intermediate_size", self.hidden_size * 4))
        self.use_double_wide_mlp = bool(tcfg.get("use_double_wide_mlp", False))
        self.num_kv_shared = int(tcfg.get("num_kv_shared_layers", 0))
        self.first_kv_shared = self.num_layers - self.num_kv_shared
        self.layer_types = list(tcfg["layer_types"])
        self.previous_kvs = _compute_previous_kvs(self.layer_types, self.num_kv_shared)
        self.window_size = int(tcfg.get("sliding_window", 512))
        self.softcap = float(tcfg.get("final_logit_softcapping") or 30.0)
        self.eps = float(tcfg.get("rms_norm_eps", 1e-6))
        self.ple_dim = int(tcfg.get("hidden_size_per_layer_input") or 0)
        self.enable_moe = bool(tcfg.get("enable_moe_block", False))
        self.num_experts = int(tcfg.get("num_experts") or 0)
        self.top_k_experts = int(tcfg.get("top_k_experts") or 8)
        self.moe_intermediate_size = int(tcfg.get("moe_intermediate_size") or 704)
        self.n_q_heads = int(tcfg.get("num_attention_heads", 8))
        self.n_kv_heads = int(tcfg.get("num_key_value_heads", 1))
        self.n_global_kv_heads = int(tcfg.get("num_global_key_value_heads") or self.n_kv_heads)
        self.attention_k_eq_v = bool(tcfg.get("attention_k_eq_v", False))
        self.swa_head_dim = int(tcfg.get("head_dim", 256))
        self.full_head_dim = int(tcfg.get("global_head_dim") or self.swa_head_dim)
        self.tie_word_embeddings = bool(tcfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)))
        self.use_bidirectional_vision = bool(tcfg.get("use_bidirectional_attention") == "vision")
        self.boi_token_id = int(cfg.get("boi_token_id", 255999))
        self.image_token_id = int(cfg.get("image_token_id", cfg.get("image_token_index", 258880)))
        self.eoi_token_id = int(cfg.get("eoi_token_id", 258882))

        sf_files = sorted(self.model_dir.glob("model*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"Missing model*.safetensors in {self.model_dir}")
        raw: dict[str, mx.array] = {}
        for sf_path in sf_files:
            raw_all: dict[str, mx.array] = mx.load(str(sf_path))
            for k, v in raw_all.items():
                if (
                    k.startswith("language_model.model.")
                    or k.startswith("model.")
                    or k.startswith("language_model.lm_head.")
                    or k.startswith("lm_head.")
                ):
                    raw[k] = v
            del raw_all

        has_any_scales = any(k.endswith(".scales") for k in raw)
        dir_lower = str(self.model_dir).lower()
        prov_qmode = ""
        prov_file = self.model_dir / "provenance.json"
        if prov_file.is_file():
            try:
                prov_obj = json.loads(prov_file.read_text(encoding="utf-8"))
                prov_qmode = str((prov_obj.get("quantization_policy") or {}).get("quant_mode", ""))
            except Exception:
                prov_qmode = ""
        auto_qat_q4_0 = (not has_any_scales) and ("qat-q4_0" in dir_lower or "qat_q4_0" in dir_lower)
        self.is_qat_checkpoint = bool(
            auto_qat_q4_0
            or "qat" in dir_lower
            or prov_qmode == "qat_q4_0"
            or qcfg.get("quant_method") == "gemma"
        )

        base_p = "language_model.model" if any(k.startswith("language_model.model.") for k in raw) else "model"
        self.embed_tokens = _load_qembedding(
            raw,
            f"{base_p}.embed_tokens",
            qcfg,
            self.act_dtype,
            expected_in_f=self.hidden_size,
            auto_qat_q4_0=auto_qat_q4_0,
        )
        self.embed_scale = mx.array(self.hidden_size**0.5, dtype=self.act_dtype)
        self.norm_w = raw[f"{base_p}.norm.weight"].astype(self.act_dtype)

        self._ple_mmap_i8: np.memmap | None = None
        self._ple_mmap_scales: np.memmap | None = None
        self.embed_tokens_per_layer: nn.Module | None = None
        self.per_layer_model_projection: nn.Module | None = None
        self.pli_proj_14: nn.Module | None = None
        self._etpl_14_groups = 0
        self._etpl_14_wcols = 0
        self._n_early_pli_layers = max(0, self.first_kv_shared - 1)

        if self.ple_dim > 0:
            self.embed_tokens_per_layer_scale = mx.array(self.ple_dim**0.5, dtype=self.act_dtype)
            self.per_layer_projection_scale = mx.array(self.hidden_size**-0.5, dtype=self.act_dtype)
            self.per_layer_projection_norm_w = raw[f"{base_p}.per_layer_projection_norm.weight"].astype(self.act_dtype)
            self.per_layer_input_scale = mx.array(2.0**-0.5, dtype=self.act_dtype)
            self.per_layer_model_projection = _load_linear(
                raw,
                f"{base_p}.per_layer_model_projection",
                qcfg,
                self.act_dtype,
                expected_in_f=self.hidden_size,
                auto_qat_q4_0=auto_qat_q4_0,
            )

            sidecar_name = cfg.get("ple_sidecar") or "embed_tokens_per_layer_i8.json"
            sidecar_json = self.model_dir / str(sidecar_name)
            if sidecar_json.is_file() and f"{base_p}.embed_tokens_per_layer.weight" not in raw:
                sc_meta = json.loads(sidecar_json.read_text(encoding="utf-8"))
                sc_shape = tuple(int(x) for x in sc_meta["shape"])
                self._ple_mmap_i8 = np.memmap(
                    self.model_dir / str(sc_meta["data_file"]),
                    dtype=np.int8,
                    mode="r",
                    shape=sc_shape,
                )
                self._ple_mmap_scales = np.memmap(
                    self.model_dir / str(sc_meta["scales_file"]),
                    dtype=np.float16,
                    mode="r",
                    shape=(sc_shape[0],),
                )
            else:
                self.embed_tokens_per_layer = _load_qembedding(
                    raw,
                    f"{base_p}.embed_tokens_per_layer",
                    qcfg,
                    self.act_dtype,
                    expected_in_f=self.num_layers * self.ple_dim,
                    auto_qat_q4_0=auto_qat_q4_0,
                )

            n_early_pli_dim = self._n_early_pli_layers * self.ple_dim
            pmp = self.per_layer_model_projection
            if n_early_pli_dim > 0 and pmp is not None:
                if isinstance(pmp, nn.QuantizedLinear):
                    pli_proj_14 = nn.QuantizedLinear(
                        int(pmp.scales.shape[1]) * int(pmp.group_size),
                        n_early_pli_dim,
                        bias=False,
                        group_size=pmp.group_size,
                        bits=pmp.bits,
                        mode=pmp.mode,
                    )
                    pli_proj_14.weight = pmp.weight[:n_early_pli_dim]
                    pli_proj_14.scales = pmp.scales[:n_early_pli_dim]
                    pli_proj_14.biases = pmp.biases[:n_early_pli_dim] if pmp.biases is not None else None
                else:
                    pli_proj_14 = nn.Linear(int(pmp.weight.shape[1]), n_early_pli_dim, bias=False)
                    pli_proj_14.weight = pmp.weight[:n_early_pli_dim]
                self.pli_proj_14 = pli_proj_14

            etpl = self.embed_tokens_per_layer
            if isinstance(etpl, nn.QuantizedEmbedding) and n_early_pli_dim > 0 and n_early_pli_dim % int(etpl.group_size) == 0:
                self._etpl_14_groups = n_early_pli_dim // int(etpl.group_size)
                self._etpl_14_wcols = (int(etpl.weight.shape[1]) * self._etpl_14_groups) // int(etpl.scales.shape[1])
        else:
            self.embed_tokens_per_layer_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_projection_norm_w = None
            self.per_layer_input_scale = None

        self._layer_idx_mx = [mx.array([i], dtype=mx.int32) for i in range(self.num_layers)]

        # Load explicit untied/materialized `lm_head.weight` when present; otherwise share rows from tied `embed_tokens`.
        lm_prefix: str | None = None
        if "language_model.lm_head.weight" in raw:
            lm_prefix = "language_model.lm_head"
        elif "lm_head.weight" in raw:
            lm_prefix = "lm_head"

        full_vocab_rows = int(self.embed_tokens.weight.shape[0])
        ckpt_remap = self.model_dir / "vocab_remap.npz"
        explicit_lm_mod: nn.Module | None = (
            _load_linear(
                raw,
                lm_prefix,
                qcfg,
                self.act_dtype,
                expected_in_f=self.hidden_size,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            if lm_prefix is not None
            else None
        )
        self._full_lm_head: nn.Module = self.embed_tokens

        def _slice_linear_or_emb(src_mod: nn.Module, kept_idx: mx.array | None) -> nn.Module:
            if isinstance(src_mod, (nn.QuantizedEmbedding, nn.QuantizedLinear)):
                in_f = int(src_mod.scales.shape[1]) * int(src_mod.group_size)
                out_f = int(kept_idx.shape[0]) if kept_idx is not None else int(src_mod.weight.shape[0])
                head = nn.QuantizedLinear(
                    in_f, out_f, bias=False, group_size=src_mod.group_size, bits=src_mod.bits, mode=src_mod.mode
                )
                if kept_idx is not None:
                    head.weight = src_mod.weight[kept_idx]
                    head.scales = src_mod.scales[kept_idx]
                    head.biases = src_mod.biases[kept_idx] if src_mod.biases is not None else None
                else:
                    head.weight = src_mod.weight
                    head.scales = src_mod.scales
                    head.biases = src_mod.biases
                return head
            w_src = src_mod.weight[kept_idx] if kept_idx is not None else src_mod.weight
            head = nn.Linear(self.hidden_size, int(w_src.shape[0]), bias=False)
            head.weight = w_src
            return head

        if explicit_lm_mod is not None and int(explicit_lm_mod.weight.shape[0]) < full_vocab_rows:
            lm_rows = int(explicit_lm_mod.weight.shape[0])
            if not ckpt_remap.is_file():
                raise FileNotFoundError(
                    f"Checkpoint {self.model_dir} contains materialized pruned lm_head ({lm_rows} rows) "
                    f"but is missing {ckpt_remap}."
                )
            self.remapper: VocabRemapper | None = VocabRemapper.load_or_build(
                self.tokenizer_json, ckpt_remap, target_size=None
            )
            if len(self.remapper.kept_old_ids) != lm_rows:
                raise ValueError(
                    f"Materialized lm_head rows ({lm_rows}) disagrees with {ckpt_remap} size "
                    f"({len(self.remapper.kept_old_ids)})."
                )
            if vocab_out is not None and int(vocab_out) != lm_rows:
                if int(vocab_out) == DEFAULT_KEPT_VOCAB_SIZE:
                    warnings.warn(
                        f"Checkpoint {self.model_dir} has materialized pruned lm_head ({lm_rows} rows); "
                        f"using checkpoint vocab size {lm_rows} instead of legacy preset vocab_out={vocab_out}.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"Requested vocab_out={vocab_out} disagrees with checkpoint materialized lm_head rows={lm_rows}."
                    )
            self.lm_head = explicit_lm_mod
            self.new_to_old = self.remapper.kept_old_ids
            self.old_to_new = self.remapper.old_to_new
        else:
            head_source: nn.Module = explicit_lm_mod if explicit_lm_mod is not None else self.embed_tokens
            self._full_lm_head = head_source
            if vocab_out is not None and int(vocab_out) < full_vocab_rows:
                if ckpt_remap.is_file():
                    with np.load(ckpt_remap) as rdata:
                        ckpt_v = int(len(rdata["kept_old_ids"] if "kept_old_ids" in rdata else rdata["old_ids"]))
                    if int(vocab_out) == ckpt_v:
                        self.remapper = VocabRemapper.load_or_build(self.tokenizer_json, ckpt_remap, target_size=None)
                    elif int(vocab_out) == DEFAULT_KEPT_VOCAB_SIZE:
                        warnings.warn(
                            f"Checkpoint {self.model_dir} provides vocab_remap.npz with {ckpt_v} tokens "
                            f"(differs from legacy preset vocab_out={vocab_out}); loading checkpoint remap ({ckpt_v}).",
                            UserWarning,
                            stacklevel=2,
                        )
                        self.remapper = VocabRemapper.load_or_build(self.tokenizer_json, ckpt_remap, target_size=None)
                    else:
                        warnings.warn(
                            f"Explicit vocab_out={vocab_out} disagrees with checkpoint vocab_remap.npz ({ckpt_v}); "
                            f"honoring explicit vocab_out={vocab_out}.",
                            UserWarning,
                            stacklevel=2,
                        )
                        self.remapper = VocabRemapper.load_or_build(
                            self.tokenizer_json, self.paths.vocab_remap_npz, target_size=int(vocab_out)
                        )
                else:
                    self.remapper = VocabRemapper.load_or_build(
                        self.tokenizer_json,
                        self.paths.vocab_remap_npz,
                        target_size=int(vocab_out),
                    )
                kept = mx.array(self.remapper.kept_old_ids, dtype=mx.int32)
                self.lm_head = _slice_linear_or_emb(head_source, kept)
                self.new_to_old = self.remapper.kept_old_ids
                self.old_to_new = self.remapper.old_to_new
            elif vocab_out is None and ckpt_remap.is_file():
                # Domain-packed SFT checkpoint shipping checkpoint-local `vocab_remap.npz`.
                self.remapper = VocabRemapper.load_or_build(self.tokenizer_json, ckpt_remap, target_size=None)
                kept = mx.array(self.remapper.kept_old_ids, dtype=mx.int32)
                self.lm_head = _slice_linear_or_emb(head_source, kept)
                self.new_to_old = self.remapper.kept_old_ids
                self.old_to_new = self.remapper.old_to_new
            elif vocab_out is None and self.allow_english_vocab_prune and self.paths.vocab_remap_npz.is_file():
                self.remapper = VocabRemapper.load_or_build(
                    self.tokenizer_json, self.paths.vocab_remap_npz, target_size=None
                )
                kept = mx.array(self.remapper.kept_old_ids, dtype=mx.int32)
                self.lm_head = _slice_linear_or_emb(head_source, kept)
                self.new_to_old = self.remapper.kept_old_ids
                self.old_to_new = self.remapper.old_to_new
            else:
                self.remapper = None
                self.lm_head = _slice_linear_or_emb(head_source, None)
                self.new_to_old = np.arange(int(self.lm_head.weight.shape[0]), dtype=np.int32)
                self.old_to_new = self.new_to_old

        self.vocab_size = int(len(self.new_to_old))
        if self.lm_head_mode == "bf16_mpp" and self._mpp_bf16_supported:
            self.w_lm_bf16_pad: mx.array | None = _slice_to_bf16_padded(self.lm_head, pad_mult=64)
        else:
            self.w_lm_bf16_pad = None

        # Under `mlp_dtype="auto"`, keep quantized MLPs intact at load time (never unconditionally dequantize!).
        promote_mlp = self.mlp_dtype_name == "bfloat16"
        inv_swa = _build_rope_inv_freq(tcfg, is_full=False)
        inv_full = _build_rope_inv_freq(tcfg, is_full=True)

        self.layers: list[dict[str, Any]] = []
        for i in range(self.num_layers):
            lp = f"{base_p}.layers.{i}"
            is_full = self.layer_types[i] == "full_attention"
            head_dim = int(self.full_head_dim if is_full else self.swa_head_dim)
            is_shared = i >= self.first_kv_shared
            layer_k_eq_v = self.attention_k_eq_v and is_full
            layer_n_kv = self.n_global_kv_heads if layer_k_eq_v else self.n_kv_heads
            eff_interm = self.intermediate_size * (2 if (self.use_double_wide_mlp and is_shared) else 1)

            q_proj_raw = _load_linear(
                raw,
                f"{lp}.self_attn.q_proj",
                qcfg,
                self.act_dtype,
                expected_in_f=self.hidden_size,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            if not is_shared:
                k_proj_raw = _load_linear(
                    raw,
                    f"{lp}.self_attn.k_proj",
                    qcfg,
                    self.act_dtype,
                    expected_in_f=self.hidden_size,
                    auto_qat_q4_0=auto_qat_q4_0,
                )
                if layer_k_eq_v and f"{lp}.self_attn.v_proj.weight" not in raw:
                    # Fuse ONLY [q_proj, k_proj] (`(n_q + n_kv)*D`) — zero wk duplication!
                    qkv_proj = _fuse_linears([q_proj_raw, k_proj_raw], self.act_dtype)
                    eff_k_eq_v = True
                else:
                    v_proj_raw = _load_linear(
                        raw,
                        f"{lp}.self_attn.v_proj",
                        qcfg,
                        self.act_dtype,
                        expected_in_f=self.hidden_size,
                        auto_qat_q4_0=auto_qat_q4_0,
                    )
                    qkv_proj = _fuse_linears([q_proj_raw, k_proj_raw, v_proj_raw], self.act_dtype)
                    eff_k_eq_v = False
                q_proj = None
                w_kn = raw[f"{lp}.self_attn.k_norm.weight"].astype(self.act_dtype)
            else:
                qkv_proj = None
                q_proj = q_proj_raw
                w_kn = None
                eff_k_eq_v = False

            o_proj = _load_linear(
                raw,
                f"{lp}.self_attn.o_proj",
                qcfg,
                self.act_dtype,
                expected_in_f=self.n_q_heads * head_dim,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            gate_p = _load_linear(
                raw,
                f"{lp}.mlp.gate_proj",
                qcfg,
                self.act_dtype,
                promote_bf16=promote_mlp,
                expected_in_f=self.hidden_size,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            up_p = _load_linear(
                raw,
                f"{lp}.mlp.up_proj",
                qcfg,
                self.act_dtype,
                promote_bf16=promote_mlp,
                expected_in_f=self.hidden_size,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            down_p = _load_linear(
                raw,
                f"{lp}.mlp.down_proj",
                qcfg,
                self.act_dtype,
                promote_bf16=promote_mlp,
                expected_in_f=eff_interm,
                auto_qat_q4_0=auto_qat_q4_0,
            )
            gate_up_proj = _fuse_linears([gate_p, up_p], self.act_dtype)
            mlp_bits: int | None = None
            if isinstance(gate_up_proj, nn.QuantizedLinear):
                mlp_bits = int(gate_up_proj.bits)
                if isinstance(down_p, nn.QuantizedLinear):
                    mlp_bits = min(mlp_bits, int(down_p.bits))

            if self.ple_dim > 0:
                ple_gate: nn.Module | None = _load_linear(
                    raw,
                    f"{lp}.per_layer_input_gate",
                    qcfg,
                    self.act_dtype,
                    expected_in_f=self.hidden_size,
                    auto_qat_q4_0=auto_qat_q4_0,
                )
                ple_proj: nn.Module | None = _load_linear(
                    raw,
                    f"{lp}.per_layer_projection",
                    qcfg,
                    self.act_dtype,
                    expected_in_f=self.ple_dim,
                    auto_qat_q4_0=auto_qat_q4_0,
                )
                w_post_ple: mx.array | None = raw[f"{lp}.post_per_layer_input_norm.weight"].astype(self.act_dtype)
            else:
                ple_gate = None
                ple_proj = None
                w_post_ple = None

            if self.enable_moe and f"{lp}.router.proj.weight" in raw:
                r_proj = _load_linear(
                    raw,
                    f"{lp}.router.proj",
                    qcfg,
                    self.act_dtype,
                    expected_in_f=self.hidden_size,
                    auto_qat_q4_0=auto_qat_q4_0,
                )
                layer_experts = int(r_proj.weight.shape[0])
                moe_router: Gemma4Router | None = Gemma4Router(
                    r_proj,
                    raw[f"{lp}.router.scale"],
                    raw[f"{lp}.router.per_expert_scale"],
                    hidden_size=self.hidden_size,
                    top_k=self.top_k_experts,
                    eps=self.eps,
                    dtype=self.act_dtype,
                )
                moe_experts: Gemma4SwitchGLU | None = _load_moe_experts(
                    raw,
                    lp,
                    qcfg,
                    self.act_dtype,
                    hidden_size=self.hidden_size,
                    moe_interm=self.moe_intermediate_size,
                    num_experts=layer_experts,
                )
                w_post_ff_1: mx.array | None = raw[f"{lp}.post_feedforward_layernorm_1.weight"].astype(self.act_dtype)
                w_pre_ff_2: mx.array | None = raw[f"{lp}.pre_feedforward_layernorm_2.weight"].astype(self.act_dtype)
                w_post_ff_2: mx.array | None = raw[f"{lp}.post_feedforward_layernorm_2.weight"].astype(self.act_dtype)
            else:
                moe_router = None
                moe_experts = None
                w_post_ff_1 = None
                w_pre_ff_2 = None
                w_post_ff_2 = None

            if i + 1 < self.num_layers:
                next_norm_w = raw[f"{base_p}.layers.{i + 1}.input_layernorm.weight"].astype(self.act_dtype)
            else:
                next_norm_w = self.norm_w

            layer_scalar_raw = raw.get(f"{lp}.layer_scalar")
            layer_scalar = (
                layer_scalar_raw.astype(self.act_dtype).reshape((1,))
                if layer_scalar_raw is not None
                else mx.ones((1,), dtype=self.act_dtype)
            )

            ld: dict[str, Any] = {
                "idx": i,
                "is_shared": is_shared,
                "prev_kv": self.previous_kvs[i],
                "is_full": is_full,
                "head_dim": head_dim,
                "n_q_heads": self.n_q_heads,
                "n_kv_heads": layer_n_kv,
                "k_eq_v": eff_k_eq_v,
                "mlp_bits": mlp_bits,
                "inv_freq": inv_full if is_full else inv_swa,
                "w_in": raw[f"{lp}.input_layernorm.weight"].astype(self.act_dtype),
                "w_post_attn": raw[f"{lp}.post_attention_layernorm.weight"].astype(self.act_dtype),
                "w_pre_ff": raw[f"{lp}.pre_feedforward_layernorm.weight"].astype(self.act_dtype),
                "w_post_ff": raw[f"{lp}.post_feedforward_layernorm.weight"].astype(self.act_dtype),
                "w_post_ff_1": w_post_ff_1,
                "w_pre_ff_2": w_pre_ff_2,
                "w_post_ff_2": w_post_ff_2,
                "w_post_ple": w_post_ple,
                "w_next_norm": next_norm_w,
                "layer_scalar": layer_scalar,
                "w_qn": raw[f"{lp}.self_attn.q_norm.weight"].astype(self.act_dtype),
                "w_kn": w_kn,
                "qkv_proj": qkv_proj,
                "q_proj": q_proj,
                "o_proj": o_proj,
                "gate_up_proj": gate_up_proj,
                "down_proj": down_p,
                "ple_gate": ple_gate,
                "ple_proj": ple_proj,
                "moe_router": moe_router,
                "moe_experts": moe_experts,
            }
            if self.use_mpp and isinstance(gate_up_proj, nn.Linear) and isinstance(down_p, nn.Linear):
                w_gu = gate_up_proj.weight
                h_mid = int(w_gu.shape[0]) // 2
                ld["w_gu"] = w_gu
                ld["w_g"] = w_gu[:h_mid]
                ld["w_u"] = w_gu[h_mid:]
                ld["w_dn"] = down_p.weight
            if i == self.first_kv_shared - 1 and qkv_proj is not None:
                q_dim = self.n_q_heads * head_dim
                ld["kv_proj_only"] = _slice_kv_from_qkv(qkv_proj, q_dim)
                ld["q_proj_split"] = _slice_q_from_qkv(qkv_proj, q_dim)
            self.layers.append(ld)

        del raw
        self.swa_layers = [i for i in range(self.first_kv_shared) if not self.layers[i]["is_full"]]
        self.full_layers = [i for i in range(self.first_kv_shared) if self.layers[i]["is_full"]]
        self.swa_slot = {l: k for k, l in enumerate(self.swa_layers)}
        self.full_slot = {l: k for k, l in enumerate(self.full_layers)}

        mx.eval(self._eval_targets())
        self.param_gb = sum(int(v.nbytes) for _, v in tree_flatten(self._eval_targets())) / (1024**3)

    def _eval_targets(self) -> list[Any]:
        arrs: list[Any] = [
            self.embed_tokens.parameters(),
            self.embed_scale,
            self.norm_w,
            self.lm_head.parameters(),
            self._layer_idx_mx,
        ]
        if self.embed_tokens_per_layer is not None:
            arrs.append(self.embed_tokens_per_layer.parameters())
        if self.per_layer_model_projection is not None:
            arrs.append(self.per_layer_model_projection.parameters())
        if self.pli_proj_14 is not None:
            arrs.append(self.pli_proj_14.parameters())
        for sc_attr in (
            "embed_tokens_per_layer_scale",
            "per_layer_projection_scale",
            "per_layer_projection_norm_w",
            "per_layer_input_scale",
        ):
            val = getattr(self, sc_attr, None)
            if val is not None:
                arrs.append(val)
        if self.w_lm_bf16_pad is not None:
            arrs.append(self.w_lm_bf16_pad)
        for ld in self.layers:
            for k in (
                "inv_freq",
                "w_in",
                "w_post_attn",
                "w_pre_ff",
                "w_post_ff",
                "w_post_ff_1",
                "w_pre_ff_2",
                "w_post_ff_2",
                "w_post_ple",
                "w_next_norm",
                "layer_scalar",
                "w_qn",
                "w_kn",
            ):
                if ld.get(k) is not None:
                    arrs.append(ld[k])
            for k in (
                "qkv_proj",
                "q_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
                "ple_gate",
                "ple_proj",
                "kv_proj_only",
                "q_proj_split",
                "moe_router",
                "moe_experts",
            ):
                m = ld.get(k)
                if m is not None:
                    arrs.append(m.parameters())
            for k in ("w_g", "w_u"):
                if k in ld:
                    arrs.append(ld[k])
        return arrs

    def _gather_mmap_ple_rows(self, toks: mx.array, n_cols: int) -> mx.array:
        assert self._ple_mmap_i8 is not None and self._ple_mmap_scales is not None
        mx.eval(toks)
        toks_np = np.asarray(toks, dtype=np.int32)
        rows_i8 = np.ascontiguousarray(self._ple_mmap_i8[toks_np][..., :n_cols])
        rows_sc = np.ascontiguousarray(self._ple_mmap_scales[toks_np])
        return (
            mx.array(rows_i8).astype(self.act_dtype) * mx.array(rows_sc).astype(self.act_dtype)[..., None]
        ) * self.embed_tokens_per_layer_scale

    def _compute_embed_and_pli(self, toks: mx.array, *, singleton: bool = False) -> tuple[mx.array, mx.array | None]:
        h = self.embed_tokens(toks) * self.embed_scale
        if getattr(self, "ple_dim", 256) <= 0:
            return h, None
        tot_cols = self.num_layers * self.ple_dim
        if getattr(self, "_ple_mmap_i8", None) is not None:
            pli_raw = self._gather_mmap_ple_rows(toks, tot_cols).reshape(*toks.shape, self.num_layers, self.ple_dim)
        else:
            assert self.embed_tokens_per_layer is not None
            pli_raw = (self.embed_tokens_per_layer(toks) * self.embed_tokens_per_layer_scale).reshape(
                *toks.shape, self.num_layers, self.ple_dim
            )
        assert self.per_layer_model_projection is not None
        pmp_out = (
            _singleton_verify_linear(self.per_layer_model_projection, h)
            if singleton
            else self.per_layer_model_projection(h)
        )
        proj = (pmp_out * self.per_layer_projection_scale).reshape(*toks.shape, self.num_layers, self.ple_dim)
        proj = mx.fast.rms_norm(proj, self.per_layer_projection_norm_w, self.eps)
        pli = (proj + pli_raw) * self.per_layer_input_scale
        return h, pli

    def _compute_embed_and_pli_early14(self, toks: mx.array) -> tuple[mx.array, mx.array | None]:
        """Compute token embeddings + PLI strictly for Layers `0..first_kv_shared-2` (`EarlyExit14`)."""
        B, L = toks.shape
        h = self.embed_tokens(toks) * self.embed_scale
        if getattr(self, "ple_dim", 256) <= 0 or self._n_early_pli_layers <= 0:
            return h, None
        n_l = self._n_early_pli_layers
        n_cols = n_l * self.ple_dim
        if getattr(self, "_ple_mmap_i8", None) is not None:
            pli_tok = self._gather_mmap_ple_rows(toks, n_cols)
        else:
            etpl = self.embed_tokens_per_layer
            assert etpl is not None
            if self._etpl_14_groups > 0:
                w_sub = etpl.weight[toks][..., : self._etpl_14_wcols]
                s_sub = etpl.scales[toks][..., : self._etpl_14_groups]
                b_sub = etpl.biases[toks][..., : self._etpl_14_groups] if etpl.biases is not None else None
                pli_tok = mx.dequantize(
                    w_sub,
                    scales=s_sub,
                    biases=b_sub,
                    group_size=etpl.group_size,
                    bits=etpl.bits,
                    mode=etpl.mode,
                ) * self.embed_tokens_per_layer_scale
            else:
                pli_tok = (
                    etpl.weight[toks][..., :n_cols].astype(self.act_dtype)
                    * self.embed_tokens_per_layer_scale
                )
        pli_tok = pli_tok.reshape(B, L, n_l, self.ple_dim)
        assert self.pli_proj_14 is not None
        pli_mod = (self.pli_proj_14(h) * self.per_layer_projection_scale).reshape(B, L, n_l, self.ple_dim)
        pli_mod = mx.fast.rms_norm(pli_mod, self.per_layer_projection_norm_w, self.eps)
        pli = (pli_mod + pli_tok) * self.per_layer_input_scale
        return h, pli

    def _can_use_lm_head_mpp(self, x_in: mx.array) -> bool:
        if self.w_lm_bf16_pad is None:
            return False
        if not getattr(self, "_mpp_bf16_supported", True):
            return False
        if getattr(self, "_gpu_family", "apple9") == "apple10" and not getattr(self, "_mpp_apple10_enabled", False):
            return False
        M = x_in.size // x_in.shape[-1] if x_in.size > 0 else 0
        min_m = int(getattr(self, "_lm_head_mpp_min_m", 64))
        return M >= min_m and x_in.dtype == getattr(self, "act_dtype", mx.bfloat16)

    def _apply_lm_head(self, x_in: mx.array, apply_softcap: bool = True) -> mx.array:
        if self._can_use_lm_head_mpp(x_in):
            assert self.w_lm_bf16_pad is not None
            lg = mpp_linear_bf16(x_in, self.w_lm_bf16_pad, tm=64, tn=64)[:, :, : self.vocab_size]
        else:
            lg = self.lm_head(x_in)
        if apply_softcap:
            lg = logit_softcap(self.softcap, lg)
        return lg

    def compute_full_vocab_logits_1d(self, x_row: mx.array, *, apply_softcap: bool = True) -> np.ndarray:
        """Compute lazy full `262,144`-vocab softcapped `float32 [V_full]` logits from a single hidden row `[D]`."""
        x_2d = mx.contiguous(x_row).reshape(1, self.hidden_size)
        full_head = getattr(self, "_full_lm_head", self.embed_tokens)
        if isinstance(full_head, (nn.QuantizedEmbedding, nn.QuantizedLinear)):
            lg = mx.quantized_matmul(
                x_2d,
                full_head.weight,
                scales=full_head.scales,
                biases=full_head.biases,
                transpose=True,
                group_size=full_head.group_size,
                bits=full_head.bits,
                mode=full_head.mode,
            )
        else:
            lg = x_2d @ full_head.weight.T
        if apply_softcap:
            lg = logit_softcap(self.softcap, lg)
        lg_f32 = lg.reshape(-1).astype(mx.float32)
        mx.eval(lg_f32)
        return np.asarray(lg_f32)

    def _apply_lm_head_argmax(
        self,
        x_in: mx.array,
        *,
        collect_telemetry: bool = False,
        valid_mask_u32: mx.array | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Compute greedy argmax (+ optional zero-copy GPU telemetry) without allocating `[B, 1, V]` FP32 buffer."""
        want_telem = bool(collect_telemetry or valid_mask_u32 is not None)
        if self.use_fused_softcap_argmax and x_in.dtype == mx.bfloat16:
            if self._can_use_lm_head_mpp(x_in):
                assert self.w_lm_bf16_pad is not None
                raw_lg = mpp_linear_bf16(x_in, self.w_lm_bf16_pad, tm=64, tn=64)
                return fused_softcap_argmax_bf16(
                    raw_lg,
                    self.softcap,
                    vocab_limit=self.vocab_size,
                    collect_telemetry=want_telem,
                    valid_mask_u32=valid_mask_u32,
                )
            raw_lg = self.lm_head(x_in)
            if raw_lg.dtype == mx.bfloat16:
                return fused_softcap_argmax_bf16(
                    raw_lg,
                    self.softcap,
                    collect_telemetry=want_telem,
                    valid_mask_u32=valid_mask_u32,
                )
        lg = self._apply_lm_head(x_in, apply_softcap=True)
        am = safe_argmax_with_nan_guard(lg[:, 0, :], axis=-1)
        if not want_telem:
            return am
        z = lg[:, 0, :].astype(mx.float32)
        lse = mx.logsumexp(z, axis=-1)
        p = mx.softmax(z, axis=-1, precise=True)
        h_pre = -mx.sum(p * (z - lse[:, None]), axis=-1)
        top2 = mx.sort(z, axis=-1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        cols = [lse, h_pre, margin]
        if valid_mask_u32 is not None:
            B_cur, V_cur = int(z.shape[0]), int(z.shape[1])
            m_np = np.asarray(valid_mask_u32, dtype=np.uint32).reshape(B_cur, -1)
            j_idx = np.arange(V_cur, dtype=np.uint32)
            vis_np = ((m_np[:, j_idx >> 5] >> (j_idx & 31)) & 1) != 0
            z_masked = mx.where(mx.array(vis_np), z, mx.array(-float("inf"), dtype=mx.float32))
            cols.append(mx.logsumexp(z_masked, axis=-1))
        return am, mx.stack(cols, axis=-1)

    def attach_dual_mlp_q4(self, *, q4_max_m: int = 32, skip_layers: frozenset[int] = frozenset()) -> int:
        """Lazily quantize missing MLPs to Q4 (~1.07 GB max) for auto B>=48 M<=q4_max_m decode tails."""
        self.dual_q4_max_m = int(q4_max_m)
        attached = 0
        for i, l in enumerate(self.layers):
            if i in skip_layers:
                continue
            gu, dn = l["gate_up_proj"], l["down_proj"]
            if isinstance(gu, nn.QuantizedLinear) or "q4_gate_up" in l:
                continue
            q_gu = nn.QuantizedLinear.from_linear(gu, group_size=64, bits=4, mode="affine")
            q_dn = nn.QuantizedLinear.from_linear(dn, group_size=64, bits=4, mode="affine")
            mx.eval(q_gu.parameters(), q_dn.parameters())
            l["q4_gate_up"] = q_gu
            l["q4_down"] = q_dn
            attached += 1
        return attached

    def _ensure_lazy_bf16_mlp_weights(self, ld: dict[str, Any]) -> bool:
        if "w_g" in ld and "w_dn" in ld:
            return True
        gu, dn = ld["gate_up_proj"], ld["down_proj"]
        if isinstance(gu, nn.QuantizedLinear):
            w_gu = mx.dequantize(
                gu.weight, gu.scales, gu.biases, group_size=gu.group_size, bits=gu.bits, mode=gu.mode
            ).astype(self.act_dtype)
        else:
            w_gu = gu.weight.astype(self.act_dtype)
        if isinstance(dn, nn.QuantizedLinear):
            w_dn = mx.dequantize(
                dn.weight, dn.scales, dn.biases, group_size=dn.group_size, bits=dn.bits, mode=dn.mode
            ).astype(self.act_dtype)
        else:
            w_dn = dn.weight.astype(self.act_dtype)
        mx.eval(w_gu, w_dn)
        h_mid = int(w_gu.shape[0]) // 2
        ld["w_gu"] = w_gu
        ld["w_g"] = w_gu[:h_mid]
        ld["w_u"] = w_gu[h_mid:]
        ld["w_dn"] = w_dn
        ld["_lazy_bf16_promoted"] = True
        return True

    def release_lazy_bf16_mlp_weights(self) -> int:
        """Release lazily promoted BF16 MLP mirrors (`w_gu`/`w_g`/`w_u`/`w_dn`) on quantized layers."""
        released = 0
        for ld in getattr(self, "layers", ()):
            if ld.pop("_lazy_bf16_promoted", False) or isinstance(ld.get("gate_up_proj"), nn.QuantizedLinear):
                had_any = False
                for k in ("w_gu", "w_g", "w_u", "w_dn"):
                    if k in ld:
                        ld.pop(k, None)
                        had_any = True
                if had_any:
                    released += 1
        return released

    def _apply_qlinear(self, ld: dict[str, Any], key: str, x: mx.array) -> mx.array:
        """Apply Apple10+ `W8A8`/`W4A8` MPP linear projection when enabled and profitable, else `mod(x)`."""
        mod = ld[key]
        if (
            getattr(self, "_mpp_apple10_int8_enabled", False)
            and mpp_quantized_linear is not None
            and isinstance(mod, nn.QuantizedLinear)
            and getattr(mod, "biases", None) is not None
            and mx.default_device() == mx.gpu
        ):
            bits = int(mod.bits)
            gs = int(mod.group_size)
            mode = str(mod.mode)
            k = int(x.shape[-1])
            m = x.size // k if k > 0 and x.size > 0 else 0
            n = int(mod.weight.shape[0])
            if should_use_mpp_int8_linear(
                m,
                k,
                n,
                bits=bits,
                group_size=gs,
                mode=mode,
                apple10_nax=True,
            ):
                packed = get_or_pack_affine_module_weights(
                    ld, f"_mpp_i8_{key}", mod, bits=bits, group_size=gs
                )
                return mpp_quantized_linear(
                    x,
                    mod.weight,
                    mod.scales,
                    mod.biases,
                    bits=bits,
                    group_size=gs,
                    mode=mode,
                    bias=getattr(mod, "bias", None),
                    out_dtype=x.dtype,
                    packed_weights=packed,
                )
        return mod(x)

    def _run_mlp(
        self,
        mlp_in: mx.array,
        ld: dict[str, Any],
        allow_mpp: bool = True,
        *,
        singleton: bool = False,
        batch_size: int | None = None,
    ) -> mx.array:
        if singleton:
            gu_out = _singleton_verify_linear(ld["gate_up_proj"], mlp_in)
            return _singleton_verify_linear(ld["down_proj"], fused_geglu_split(gu_out))
        M = mlp_in.size // mlp_in.shape[-1] if mlp_in.size > 0 else 0
        eff_b = int(batch_size) if batch_size is not None else (int(mlp_in.shape[0]) if mlp_in.ndim >= 3 else 1)
        if getattr(self, "_dual_q4_active", False) and "q4_gate_up" in ld:
            q4_cap = getattr(self, "dual_q4_max_m", 32)
            if M <= q4_cap:
                mlp_act = fused_geglu_split(ld["q4_gate_up"](mlp_in))
                return ld["q4_down"](mlp_act)
        mpp_os_ok = getattr(self, "_mpp_bf16_supported", True)
        gpu_fam = getattr(self, "_gpu_family", "apple9")
        gpu_ok = gpu_fam != "apple10" or getattr(self, "_mpp_apple10_enabled", False)
        prefill_min_m = int(getattr(self, "_mpp_prefill_min_m", 2048))
        mlp_mode = getattr(self, "mlp_dtype_name", "bfloat16")
        if self.use_mpp and allow_mpp and mpp_os_ok and gpu_ok:
            use_bf16_mpp = False
            if mlp_mode == "bfloat16":
                use_bf16_mpp = "w_g" in ld
            elif mlp_mode == "auto" and eff_b >= 64:
                if should_use_bf16_mpp_for_mlp(
                    mlp_dtype="auto",
                    gpu_family=gpu_fam,
                    host_ram_gib=float(getattr(self, "_host_ram_gib", 48.0)),
                    batch_size=eff_b,
                    mlp_bits=ld.get("mlp_bits"),
                ):
                    use_bf16_mpp = self._ensure_lazy_bf16_mlp_weights(ld)
            if use_bf16_mpp and "w_g" in ld:
                if 16 <= M <= 64:
                    mlp_act = mpp_fused_geglu_bf16(
                        mlp_in, ld["w_g"], ld["w_u"], ld["w_gu"], m32_cutoff=self.mpp_m32_cutoff
                    )
                    if isinstance(ld["down_proj"], nn.QuantizedLinear):
                        return mlp_act @ ld["w_dn"].T
                    return ld["down_proj"](mlp_act)
                if M >= prefill_min_m:
                    mlp_act = mpp_fused_geglu_bf16(
                        mlp_in, ld["w_g"], ld["w_u"], ld["w_gu"], m32_cutoff=self.mpp_m32_cutoff
                    )
                    return mpp_linear_bf16(mlp_act, ld["w_dn"], 64, 64)
                if mlp_mode == "auto" and isinstance(ld["gate_up_proj"], nn.QuantizedLinear):
                    mlp_act = fused_geglu_split(mlp_in @ ld["w_gu"].T)
                    return mlp_act @ ld["w_dn"].T
        mlp_act = fused_geglu_split(
            self._apply_qlinear(ld, "gate_up_proj", mlp_in) if allow_mpp else ld["gate_up_proj"](mlp_in)
        )
        return self._apply_qlinear(ld, "down_proj", mlp_act) if allow_mpp else ld["down_proj"](mlp_act)

    def _run_ffn_and_ple(
        self,
        h: mx.array,
        attn_out: mx.array,
        pli: mx.array | None,
        i: int,
        l: dict[str, Any],
        *,
        singleton: bool = False,
        batch_size: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
        if singleton:
            mlp_out = self._run_mlp(mlp_in, l, singleton=True)
        elif batch_size is not None:
            mlp_out = self._run_mlp(mlp_in, l, batch_size=batch_size)
        else:
            mlp_out = self._run_mlp(mlp_in, l)
        moe_router = l.get("moe_router")
        if moe_router is not None:
            moe_experts = l["moe_experts"]
            h1 = mx.fast.rms_norm(mlp_out, l["w_post_ff_1"], self.eps)
            top_k_idx, top_k_w = moe_router(h_mid, singleton=singleton)
            h2_in = mx.fast.rms_norm(h_mid, l["w_pre_ff_2"], self.eps)
            h2 = mx.fast.rms_norm(
                moe_experts(h2_in, top_k_idx, top_k_w, singleton=singleton),
                l["w_post_ff_2"],
                self.eps,
            )
            ff_out = h1 + h2
        else:
            ff_out = mlp_out

        if pli is not None and l.get("ple_gate") is not None:
            h_ff = fused_res_rms_norm(h_mid, ff_out, l["w_post_ff"], self.eps)
            pg_raw = _singleton_verify_linear(l["ple_gate"], h_ff) if singleton else l["ple_gate"](h_ff)
            ple_g = fused_ple_geglu_indexed(pg_raw, pli, self._layer_idx_mx[i])
            ple_out = _singleton_verify_linear(l["ple_proj"], ple_g) if singleton else l["ple_proj"](ple_g)
            return fused_norm_add_norm(
                h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"]
            )
        return fused_norm_add_norm(
            h_mid, ff_out, l["w_post_ff"], l["w_next_norm"], self.eps, l["layer_scalar"]
        )

    def _sdpa_multi_query_prefix_cache(
        self,
        q_rot: mx.array,
        k_sys: mx.array,
        v_sys: mx.array,
        c: dict[str, Any],
        off_b: mx.array,
        l_sys: int,
        *,
        is_full: bool,
        slot_idx: mx.array | None = None,
        use_kv_i8: bool = False,
    ) -> mx.array:
        """Delegate `L_q > 1` SDPA to `mx.fast.scaled_dot_product_attention` (`NAX` on M5 + `steel_attention` `WN=2` on M4)."""
        B, L_q, n_q, D = (int(x) for x in q_rot.shape)
        n_kv = int(c["k"].shape[2]) if c["k"].ndim == 4 else 1
        cap = int(c["cap"])
        is_4d = c["k"].ndim == 4
        mx.eval(off_b)
        offs = [int(x) for x in off_b.tolist()]

        def _run_uniform(rows_b: list[int], off_val: int) -> mx.array:
            Bg = len(rows_b)
            if Bg == B and rows_b == list(range(B)):
                q_sub = q_rot.transpose(0, 2, 1, 3)
                s_sub = slot_idx if slot_idx is not None else mx.arange(B, dtype=mx.int32)
                use_slice_all = slot_idx is None
            else:
                r_mx = mx.array(rows_b, dtype=mx.int32)
                q_sub = q_rot[r_mx].transpose(0, 2, 1, 3)
                s_sub = slot_idx[r_mx] if slot_idx is not None else r_mx
                use_slice_all = False

            parts_k: list[mx.array] = []
            parts_v: list[mx.array] = []
            if is_full:
                if l_sys > 0:
                    ks = k_sys[:l_sys].transpose(1, 0, 2)[None] if k_sys.ndim == 3 else k_sys[:l_sys][None, None]
                    vs = v_sys[:l_sys].transpose(1, 0, 2)[None] if v_sys.ndim == 3 else v_sys[:l_sys][None, None]
                    parts_k.append(mx.broadcast_to(ks, (Bg, n_kv, l_sys, D)))
                    parts_v.append(mx.broadcast_to(vs, (Bg, n_kv, l_sys, D)))
                priv_end = off_val + L_q - l_sys
                if priv_end > 0:
                    if use_kv_i8 and "k_s" in c:
                        p_pos = mx.arange(0, priv_end, dtype=mx.int32)
                        kd, vd = gather_dequant_kv_pair_i8_bf16(
                            c["k"], c["k_s"], c["v"], c["v_s"], s_sub, p_pos
                        )
                    elif use_slice_all:
                        kp = c["k"][:, :priv_end]
                        vp = c["v"][:, :priv_end]
                        kd = kp.transpose(0, 2, 1, 3) if is_4d else kp[:, None, :, :]
                        vd = vp.transpose(0, 2, 1, 3) if is_4d else vp[:, None, :, :]
                    else:
                        kp = c["k"][s_sub, :priv_end]
                        vp = c["v"][s_sub, :priv_end]
                        kd = kp.transpose(0, 2, 1, 3) if is_4d else kp[:, None, :, :]
                        vd = vp.transpose(0, 2, 1, 3) if is_4d else vp[:, None, :, :]
                    parts_k.append(kd)
                    parts_v.append(vd)
                mask = _select_sdpa_mask(L_q, off_val, is_full=True, window_size=None, dtype=self.act_dtype)
            else:
                t_start = max(0, off_val - self.window_size + 1)
                t_end = off_val + L_q
                if t_start < l_sys:
                    ks_sl = k_sys[t_start:l_sys]
                    vs_sl = v_sys[t_start:l_sys]
                    ks = ks_sl.transpose(1, 0, 2)[None] if ks_sl.ndim == 3 else ks_sl[None, None]
                    vs = vs_sl.transpose(1, 0, 2)[None] if vs_sl.ndim == 3 else vs_sl[None, None]
                    parts_k.append(mx.broadcast_to(ks, (Bg, n_kv, l_sys - t_start, D)))
                    parts_v.append(mx.broadcast_to(vs, (Bg, n_kv, l_sys - t_start, D)))
                priv_lo = max(0, t_start - l_sys)
                priv_hi = t_end - l_sys
                if priv_hi > priv_lo:
                    p_slots = mx.arange(priv_lo, priv_hi, dtype=mx.int32) & (cap - 1)
                    if use_kv_i8 and "k_s" in c:
                        kd, vd = gather_dequant_kv_pair_i8_bf16(
                            c["k"], c["k_s"], c["v"], c["v_s"], s_sub, p_slots
                        )
                    elif use_slice_all:
                        kp = c["k"][:, p_slots]
                        vp = c["v"][:, p_slots]
                        kd = kp.transpose(0, 2, 1, 3) if is_4d else kp[:, None, :, :]
                        vd = vp.transpose(0, 2, 1, 3) if is_4d else vp[:, None, :, :]
                    else:
                        kp = c["k"][s_sub[:, None], p_slots[None, :]]
                        vp = c["v"][s_sub[:, None], p_slots[None, :]]
                        kd = kp.transpose(0, 2, 1, 3) if is_4d else kp[:, None, :, :]
                        vd = vp.transpose(0, 2, 1, 3) if is_4d else vp[:, None, :, :]
                    parts_k.append(kd)
                    parts_v.append(vd)
                mask = _select_sdpa_mask(
                    L_q, off_val - t_start, is_full=False, window_size=self.window_size, dtype=self.act_dtype
                )

            kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
            vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
            return (
                mx.fast.scaled_dot_product_attention(q_sub, kf, vf, scale=1.0, mask=mask)
                .transpose(0, 2, 1, 3)
                .reshape(Bg, L_q, n_q * D)
            )

        if len(set(offs)) == 1:
            return _run_uniform(list(range(B)), offs[0])

        by_off: dict[int, list[int]] = {}
        for idx_b, ov in enumerate(offs):
            by_off.setdefault(ov, []).append(idx_b)
        if len(by_off) == B:
            return mx.concatenate([_run_uniform([b], offs[b]) for b in range(B)], axis=0)
        out_rows: list[tuple[int, mx.array]] = []
        for ov, grp in by_off.items():
            sub_att = _run_uniform(grp, ov)
            for pos_g, orig_b in enumerate(grp):
                out_rows.append((orig_b, sub_att[pos_g : pos_g + 1]))
        out_rows.sort(key=lambda item: item[0])
        return mx.concatenate([item[1] for item in out_rows], axis=0)

    # ------------------------------------------------------------------
    # Shared Prefix KV_sys Prefill (runs ONCE per system prompt)
    # ------------------------------------------------------------------
    def prefill_prefix(self, sys_ids: Sequence[int], chunk_sz: int | None = None) -> dict[str, Any]:
        """Compute shared `kv_sys [first_kv_shared][L_sys, n_kv, D]` for system prompt using EarlyExit."""
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)
        if not sys_ids:
            empty_k = [
                mx.zeros((16, self.layers[i].get("n_kv_heads", default_n_kv), self.layers[i]["head_dim"]), dtype=self.act_dtype)
                for i in range(self.first_kv_shared)
            ]
            empty_v = [
                mx.zeros((16, self.layers[i].get("n_kv_heads", default_n_kv), self.layers[i]["head_dim"]), dtype=self.act_dtype)
                for i in range(self.first_kv_shared)
            ]
            mx.eval(*empty_k, *empty_v)
            return {"l_sys": 0, "k_sys": empty_k, "v_sys": empty_v}

        ids = mx.array(sys_ids, dtype=mx.int32)[None, :]
        l_sys = int(ids.shape[1])
        eff_chunk_sz = int(chunk_sz) if chunk_sz is not None else (1024 if l_sys >= 1024 else max(256, l_sys))
        k_sys_list = [
            mx.zeros((l_sys, self.layers[i].get("n_kv_heads", default_n_kv), self.layers[i]["head_dim"]), dtype=self.act_dtype)
            for i in range(self.first_kv_shared)
        ]
        v_sys_list = [
            mx.zeros((l_sys, self.layers[i].get("n_kv_heads", default_n_kv), self.layers[i]["head_dim"]), dtype=self.act_dtype)
            for i in range(self.first_kv_shared)
        ]
        mx.eval(*k_sys_list, *v_sys_list)

        for s0 in range(0, l_sys, eff_chunk_sz):
            s1 = min(l_sys, s0 + eff_chunk_sz)
            lc = s1 - s0
            tok_c = ids[:, s0:s1]
            off_meta = mx.array([s0, lc], dtype=mx.int32)
            off_1 = mx.array([s0], dtype=mx.int32)
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                n_q = l.get("n_q_heads", default_n_q)
                n_kv = l.get("n_kv_heads", default_n_kv)
                if i == self.first_kv_shared - 1:
                    kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_1, self.eps, n_kv=n_kv
                    )
                    k_sys_list[i][s0:s1] = k_rot[0]
                    v_sys_list[i][s0:s1] = v_norm[0]
                    break

                qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps, n_q=n_q, n_kv=n_kv
                )
                k_sys_list[i][s0:s1] = k_rot[0]
                v_sys_list[i][s0:s1] = v_norm[0]

                if l["is_full"]:
                    kf = k_sys_list[i][:s1].transpose(1, 0, 2)[None, :, :, :]
                    vf = v_sys_list[i][:s1].transpose(1, 0, 2)[None, :, :, :]
                    mk = ("full", s0, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, s0, is_full=True, window_size=None, dtype=self.act_dtype
                        )
                else:
                    st = max(0, s0 - self.window_size + 1)
                    kf = k_sys_list[i][st:s1].transpose(1, 0, 2)[None, :, :, :]
                    vf = v_sys_list[i][st:s1].transpose(1, 0, 2)[None, :, :, :]
                    mk = ("swa", s0 - st, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, s0 - st, is_full=False, window_size=self.window_size, dtype=self.act_dtype
                        )

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(1, lc, n_q * D)
                )
                attn_out = self._apply_qlinear(l, "o_proj", att)
                h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)
            mx.eval(*k_sys_list, *v_sys_list)
        if l_sys > eff_chunk_sz and callable(getattr(mx, "clear_cache", None)):
            mx.clear_cache()

        return {"l_sys": l_sys, "k_sys": k_sys_list, "v_sys": v_sys_list}

    # ------------------------------------------------------------------
    # Ragged Per-Doc Cache Allocation & Prefix-Shared Decode/Verify Step
    # ------------------------------------------------------------------
    def make_batch_cache(
        self,
        prefix_kv: dict[str, Any],
        batch_size: int,
        doc_cap_full: int = 1152,
        swa_ring_cap: int = 512,
    ) -> dict[str, Any]:
        if swa_ring_cap < self.window_size or (swa_ring_cap & (swa_ring_cap - 1)) != 0:
            raise ValueError(f"swa_ring_cap={swa_ring_cap} must be a power of 2 >= {self.window_size}")
        default_n_kv = getattr(self, "n_kv_heads", 1)
        layers = []
        for i in range(self.first_kv_shared):
            l = self.layers[i]
            D = l["head_dim"]
            n_kv = l.get("n_kv_heads", default_n_kv)
            ring = not l["is_full"]
            cap = swa_ring_cap if ring else doc_cap_full
            layers.append(
                {
                    "k": mx.zeros((batch_size, cap, n_kv, D), dtype=self.act_dtype),
                    "v": mx.zeros((batch_size, cap, n_kv, D), dtype=self.act_dtype),
                    "cap": cap,
                    "ring": ring,
                }
            )
        mx.eval(*[(c["k"], c["v"]) for c in layers])
        l_sys = int(prefix_kv["l_sys"])
        off_b = mx.full((batch_size,), l_sys, dtype=mx.int32)
        ns_swa = 1 if batch_size >= 128 else (2 if batch_size >= 96 else 4)
        ns_full = 1 if batch_size >= 128 else (4 if batch_size >= 96 else 8)
        return {
            "batch_size": batch_size,
            "l_sys": l_sys,
            "k_sys": prefix_kv["k_sys"],
            "v_sys": prefix_kv["v_sys"],
            "layers": layers,
            "off_b": off_b,
            "ns_swa": ns_swa,
            "ns_full": ns_full,
            "meta_swa": mx.array([l_sys, swa_ring_cap, self.window_size, 1, ns_swa, swa_ring_cap - 1], dtype=mx.int32),
            "meta_full": mx.array([l_sys, doc_cap_full, 0, 0, ns_full, -1], dtype=mx.int32),
        }

    def forward_step_ragged(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        *,
        apply_softcap: bool = True,
        keep_from_col: int = 0,
    ) -> mx.array:
        """Full multi-layer ragged batched step for `L_q >= 1` (`1` decode, `L_q > 1` MLX SDPA, or `keep_from_col` slice jump)."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if keep_from_col < 0 or keep_from_col >= L_q:
            raise ValueError(f"Expected 0 <= keep_from_col < L_q ({L_q}), got {keep_from_col}")
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            cap = c["cap"]
            ring_cap = cap if c["ring"] else 0
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)

            if not l["is_shared"]:
                if i == split_idx and keep_from_col > 0:
                    kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                    )
                    scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    if pli is not None:
                        pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = self._apply_qlinear(l, "q_proj_split", x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)
                else:
                    qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                    )
                    scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            else:
                q_raw = self._apply_qlinear(l, "q_proj", x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            if int(q_rot.shape[1]) > 1:
                att = self._sdpa_multi_query_prefix_cache(
                    q_rot, k_sys_l[kv_idx], v_sys_l[kv_idx], c, off_b, l_sys, is_full=bool(l["is_full"])
                )
            elif l["is_full"]:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0,
                    doc_ring_cap=0,
                    n_splits=cache["ns_full"],
                    meta=cache["meta_full"],
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=self.window_size,
                    doc_ring_cap=cap,
                    n_splits=cache["ns_swa"],
                    meta=cache["meta_swa"],
                    vec4_reduce=self.vec4_reduce,
                )

            attn_out = self._apply_qlinear(l, "o_proj", att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)

        cache["_last_x_in"] = x_in
        return self._apply_lm_head(x_in, apply_softcap=apply_softcap)

    # ------------------------------------------------------------------
    # GPU Suffix Prefill (Shared KV_sys + EarlyExit14 + Batched LastTok)
    # ------------------------------------------------------------------
    def _prefill_early14_group(
        self,
        sub_toks: list[list[int]],
        doc_indices: list[int],
        cache: dict[str, Any],
        chunk_sz: int = 256,
    ) -> None:
        Bg = len(doc_indices)
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        pre_lens = [len(s) - 1 for s in sub_toks]
        max_pre = max(pre_lens)
        if max_pre <= 0:
            return
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        pad_arr = np.zeros((Bg, max_pre), dtype=np.int32)
        for j, s in enumerate(sub_toks):
            if pre_lens[j] > 0:
                pad_arr[j, : pre_lens[j]] = s[: pre_lens[j]]
        toks_mx = mx.array(pad_arr, dtype=mx.int32)
        b_idx = mx.array(doc_indices, dtype=mx.int32)

        for s0 in range(0, max_pre, chunk_sz):
            s1 = min(max_pre, s0 + chunk_sz)
            lc = s1 - s0
            off = l_sys + s0
            off_meta = mx.array([off, lc], dtype=mx.int32)
            off_bg = mx.full((Bg,), off, dtype=mx.int32)
            tok_c = toks_mx[:, s0:s1]
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                n_q = l.get("n_q_heads", default_n_q)
                n_kv = l.get("n_kv_heads", default_n_kv)
                c = doc_layers[i]
                cap = c["cap"]
                is_4d = c["k"].ndim == 4

                if i == self.first_kv_shared - 1:
                    kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_bg, self.eps, n_kv=n_kv
                    )
                    k_store = k_rot if is_4d else k_rot.reshape(Bg, lc, D)
                    v_store = v_norm if is_4d else v_norm.reshape(Bg, lc, D)
                    if c["ring"]:
                        slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                        c["k"][b_idx[:, None], slots[None, :]] = k_store
                        c["v"][b_idx[:, None], slots[None, :]] = v_store
                    else:
                        c["k"][b_idx, s0:s1] = k_store
                        c["v"][b_idx, s0:s1] = v_store
                    break

                qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps, n_q=n_q, n_kv=n_kv
                )
                k_store = k_rot if is_4d else k_rot.reshape(Bg, lc, D)
                v_store = v_norm if is_4d else v_norm.reshape(Bg, lc, D)
                k_chunk_sdpa = k_rot.transpose(0, 2, 1, 3) if k_rot.ndim == 4 else k_rot[:, None, :, :]
                v_chunk_sdpa = v_norm.transpose(0, 2, 1, 3) if v_norm.ndim == 4 else v_norm[:, None, :, :]

                if not c["ring"]:
                    c["k"][b_idx, s0:s1] = k_store
                    c["v"][b_idx, s0:s1] = v_store
                    parts_k = []
                    parts_v = []
                    if l_sys > 0:
                        ks_i = k_sys_l[i].transpose(1, 0, 2)[None] if k_sys_l[i].ndim == 3 else k_sys_l[i][None, None]
                        vs_i = v_sys_l[i].transpose(1, 0, 2)[None] if v_sys_l[i].ndim == 3 else v_sys_l[i][None, None]
                        parts_k.append(mx.broadcast_to(ks_i, (Bg, n_kv, l_sys, D)))
                        parts_v.append(mx.broadcast_to(vs_i, (Bg, n_kv, l_sys, D)))
                    if s0 == 0:
                        parts_k.append(k_chunk_sdpa)
                        parts_v.append(v_chunk_sdpa)
                    else:
                        kp_prev = c["k"][b_idx, :s1]
                        vp_prev = c["v"][b_idx, :s1]
                        parts_k.append(kp_prev.transpose(0, 2, 1, 3) if is_4d else kp_prev[:, None, :, :])
                        parts_v.append(vp_prev.transpose(0, 2, 1, 3) if is_4d else vp_prev[:, None, :, :])
                    kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
                    vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
                    mk = ("full", off, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, off, is_full=True, window_size=None, dtype=self.act_dtype
                        )
                else:
                    slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                    t_start = max(0, off - self.window_size + 1)
                    k_parts = []
                    v_parts = []
                    if t_start < l_sys:
                        sys_lo = t_start
                        ks_sl = k_sys_l[i][sys_lo:l_sys]
                        vs_sl = v_sys_l[i][sys_lo:l_sys]
                        ks_b = ks_sl.transpose(1, 0, 2)[None] if ks_sl.ndim == 3 else ks_sl[None, None]
                        vs_b = vs_sl.transpose(1, 0, 2)[None] if vs_sl.ndim == 3 else vs_sl[None, None]
                        k_parts.append(mx.broadcast_to(ks_b, (Bg, n_kv, l_sys - sys_lo, D)))
                        v_parts.append(mx.broadcast_to(vs_b, (Bg, n_kv, l_sys - sys_lo, D)))
                    priv_lo = max(0, t_start - l_sys)
                    if priv_lo < s0:
                        p_slots = mx.arange(priv_lo, s0, dtype=mx.int32) & (cap - 1)
                        kp_old = c["k"][b_idx[:, None], p_slots[None, :]]
                        vp_old = c["v"][b_idx[:, None], p_slots[None, :]]
                        k_parts.append(kp_old.transpose(0, 2, 1, 3) if is_4d else kp_old[:, None, :, :])
                        v_parts.append(vp_old.transpose(0, 2, 1, 3) if is_4d else vp_old[:, None, :, :])
                    if priv_lo < s1:
                        chunk_lo = max(0, priv_lo - s0)
                        k_parts.append(k_chunk_sdpa if chunk_lo == 0 else k_chunk_sdpa[:, :, chunk_lo:, :])
                        v_parts.append(v_chunk_sdpa if chunk_lo == 0 else v_chunk_sdpa[:, :, chunk_lo:, :])
                    c["k"][b_idx[:, None], slots[None, :]] = k_store
                    c["v"][b_idx[:, None], slots[None, :]] = v_store
                    kf = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                    vf = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                    mk = ("swa", off - t_start, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, off - t_start, is_full=False, window_size=self.window_size, dtype=self.act_dtype
                        )

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(Bg, lc, n_q * D)
                )
                attn_out = self._apply_qlinear(l, "o_proj", att)
                h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)
            mx.eval(*[(c["k"], c["v"]) for c in doc_layers])

    def prefill_suffix_batch(
        self,
        suf_ids_list: list[list[int]],
        cache: dict[str, Any],
        *,
        group_size: int = 16,
        apply_softcap: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Prefill unique doc suffixes via Shared KV_sys + EarlyExit14 + batched LastTok."""
        B = len(suf_ids_list)
        if B != cache["batch_size"]:
            raise ValueError(f"Batch size mismatch: {B} vs cache {cache['batch_size']}")
        l_sys = cache["l_sys"]

        order = sorted(range(B), key=lambda b: len(suf_ids_list[b]))
        g0 = 0
        while g0 < B:
            l0 = len(suf_ids_list[order[g0]])
            g1 = g0 + 1
            while g1 < B and (g1 - g0) < group_size and (len(suf_ids_list[order[g1]]) - l0) <= 48:
                g1 += 1
            idxs = order[g0:g1]
            sub_toks = [suf_ids_list[b] for b in idxs]
            self._prefill_early14_group(sub_toks, idxs, cache, chunk_sz=256)
            g0 = g1
        if callable(getattr(mx, "clear_cache", None)):
            mx.clear_cache()

        last_toks = mx.array([suf_ids_list[b][-1] for b in range(B)], dtype=mx.int32)[:, None]
        last_off = mx.array([l_sys + len(suf_ids_list[b]) - 1 for b in range(B)], dtype=mx.int32)
        lg = self.forward_step_ragged(last_toks, cache, last_off, apply_softcap=apply_softcap)
        next_off = last_off + 1
        mx.eval(lg, next_off)
        return lg, next_off

    # ------------------------------------------------------------------
    # ANE KV Prep, Ingestion, Cache Compaction & EarlyExit14 Tail
    # ------------------------------------------------------------------
    def ingest_ane_chunk_kv(
        self,
        cache: dict[str, Any],
        doc_indices: Sequence[int],
        priv_off: int,
        k_swa_f16: Any,
        v_swa_f16: Any,
        k_full_f16: Any,
        v_full_f16: Any,
        *,
        valid_len: int = 256,
        eval_cache: bool = True,
    ) -> None:
        b_idx = mx.array(doc_indices, dtype=mx.int32)
        use_kv_i8 = bool(cache.get("kv_i8", False))
        doc_layers = cache["layers"]

        ks_is_5d = len(k_swa_f16.shape) == 5
        kf_is_5d = len(k_full_f16.shape) == 5
        ks_mx = (
            mx.array(k_swa_f16[:, :, :, :valid_len, :]).astype(self.act_dtype)
            if ks_is_5d
            else mx.array(k_swa_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        )
        vs_mx = (
            mx.array(v_swa_f16[:, :, :, :valid_len, :]).astype(self.act_dtype)
            if ks_is_5d
            else mx.array(v_swa_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        )
        kf_mx = (
            mx.array(k_full_f16[:, :, :, :valid_len, :]).astype(self.act_dtype)
            if kf_is_5d
            else mx.array(k_full_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        )
        vf_mx = (
            mx.array(v_full_f16[:, :, :, :valid_len, :]).astype(self.act_dtype)
            if kf_is_5d
            else mx.array(v_full_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        )

        def _align_layer_kv(k_sl: mx.array, v_sl: mx.array, is_5d: bool, dst_is_4d: bool) -> tuple[mx.array, mx.array]:
            if is_5d:
                k_sl = k_sl.transpose(0, 2, 1, 3)
                v_sl = v_sl.transpose(0, 2, 1, 3)
                if not dst_is_4d:
                    B_i, L_i, _, D_i = k_sl.shape
                    k_sl = k_sl.reshape(B_i, L_i, D_i)
                    v_sl = v_sl.reshape(B_i, L_i, D_i)
            elif dst_is_4d:
                B_i, L_i, D_i = k_sl.shape
                k_sl = k_sl.reshape(B_i, L_i, 1, D_i)
                v_sl = v_sl.reshape(B_i, L_i, 1, D_i)
            return k_sl, v_sl

        for slot_i, l_idx in enumerate(self.swa_layers):
            c = doc_layers[l_idx]
            cap = c["cap"]
            slots = mx.arange(priv_off, priv_off + valid_len, dtype=mx.int32) & (cap - 1)
            k_sl, v_sl = _align_layer_kv(ks_mx[:, slot_i], vs_mx[:, slot_i], ks_is_5d, c["k"].ndim == 4)
            if use_kv_i8 and "k_s" in c:
                kq, ks, vq, vs = quantize_kv_pair_i8(k_sl, v_sl)
                c["k"][b_idx[:, None], slots[None, :]] = kq
                c["k_s"][b_idx[:, None], slots[None, :]] = ks
                c["v"][b_idx[:, None], slots[None, :]] = vq
                c["v_s"][b_idx[:, None], slots[None, :]] = vs
            else:
                c["k"][b_idx[:, None], slots[None, :]] = k_sl
                c["v"][b_idx[:, None], slots[None, :]] = v_sl
        for slot_i, l_idx in enumerate(self.full_layers):
            c = doc_layers[l_idx]
            k_sl, v_sl = _align_layer_kv(kf_mx[:, slot_i], vf_mx[:, slot_i], kf_is_5d, c["k"].ndim == 4)
            if use_kv_i8 and "k_s" in c:
                kq, ks, vq, vs = quantize_kv_pair_i8(k_sl, v_sl)
                c["k"][b_idx, priv_off : priv_off + valid_len] = kq
                c["k_s"][b_idx, priv_off : priv_off + valid_len] = ks
                c["v"][b_idx, priv_off : priv_off + valid_len] = vq
                c["v_s"][b_idx, priv_off : priv_off + valid_len] = vs
            else:
                c["k"][b_idx, priv_off : priv_off + valid_len] = k_sl
                c["v"][b_idx, priv_off : priv_off + valid_len] = v_sl
        if eval_cache:
            self._eval_doc_layers(doc_layers, use_kv_i8 and bool(doc_layers and "k_s" in doc_layers[0]))

    def compact_cache(self, cache: dict[str, Any], keep_indices: Sequence[int]) -> dict[str, Any]:
        idx = mx.array(keep_indices, dtype=mx.int32)
        B_new = len(keep_indices)
        l_sys = cache["l_sys"]
        swa_cap = cache["layers"][self.swa_layers[0]]["cap"]
        full_cap = cache["layers"][self.full_layers[0]]["cap"]
        new_layers = [{"k": c["k"][idx], "v": c["v"][idx], "cap": c["cap"], "ring": c["ring"]} for c in cache["layers"]]
        ns_swa = 1 if B_new >= 128 else (2 if B_new >= 96 else 4)
        ns_full = 1 if B_new >= 128 else (4 if B_new >= 96 else 8)
        mx.eval(*[(c["k"], c["v"]) for c in new_layers])
        return {
            "batch_size": B_new,
            "l_sys": l_sys,
            "k_sys": cache["k_sys"],
            "v_sys": cache["v_sys"],
            "layers": new_layers,
            "off_b": cache["off_b"][idx],
            "ns_swa": ns_swa,
            "ns_full": ns_full,
            "meta_swa": mx.array([l_sys, swa_cap, self.window_size, 1, ns_swa, swa_cap - 1], dtype=mx.int32),
            "meta_full": mx.array([l_sys, full_cap, 0, 0, ns_full, -1], dtype=mx.int32),
        }

    def forward_early14_ragged(self, toks: mx.array, cache: dict[str, Any], off_b: mx.array) -> None:
        """Populate KV layers `0..first_kv_shared-1` only (`L_q >= 1`), exiting at `kv_proj_only`."""
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        h, pli = self._compute_embed_and_pli_early14(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)

        for i in range(self.first_kv_shared):
            l = self.layers[i]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)
            c = doc_layers[i]
            cap = c["cap"]
            ring_cap = cap if c["ring"] else 0

            if i == self.first_kv_shared - 1:
                kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                    kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                )
                scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
                break

            qkv = self._apply_qlinear(l, "qkv_proj", x_in)
            q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
            )
            scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            if int(q_rot.shape[1]) > 1:
                att = self._sdpa_multi_query_prefix_cache(
                    q_rot, k_sys_l[i], v_sys_l[i], c, off_b, l_sys, is_full=bool(l["is_full"])
                )
            elif l["is_full"]:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[i],
                    v_sys_l[i],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0,
                    doc_ring_cap=0,
                    n_splits=cache["ns_full"],
                    meta=cache["meta_full"],
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[i],
                    v_sys_l[i],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=self.window_size,
                    doc_ring_cap=cap,
                    n_splits=cache["ns_swa"],
                    meta=cache["meta_swa"],
                    vec4_reduce=self.vec4_reduce,
                )
            attn_out = self._apply_qlinear(l, "o_proj", att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)

    # ------------------------------------------------------------------
    # Slot-Indexed Continuous Cache (BF16 or INT8) + EarlyExit14 & Prefix Split
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_doc_layers(doc_layers: list[dict[str, Any]], kv_i8: bool) -> None:
        arrs: list[mx.array] = []
        if kv_i8:
            for c in doc_layers:
                arrs.extend((c["k"], c["k_s"], c["v"], c["v_s"]))
        else:
            for c in doc_layers:
                arrs.extend((c["k"], c["v"]))
        mx.eval(*arrs)

    def make_slot_cache(
        self,
        prefix_kv: dict[str, Any],
        batch_size: int,
        doc_cap_full: int = 1440,
        swa_ring_cap: int = 512,
        kv_i8: bool | None = None,
        drain_slack: int = 16,
    ) -> dict[str, Any]:
        """Allocate fixed `B_slots` persistent KV cache `(B_slots, Cap, n_kv, D)` for continuous rolling scheduling."""
        if swa_ring_cap < self.window_size or (swa_ring_cap & (swa_ring_cap - 1)) != 0:
            raise ValueError(f"swa_ring_cap={swa_ring_cap} must be a power of 2 >= {self.window_size}")
        slack = int(drain_slack)
        if slack < 0:
            raise ValueError(f"drain_slack must be >= 0, got {slack}")
        use_kv_i8 = self.kv_i8 if kv_i8 is None else bool(kv_i8)
        default_n_kv = getattr(self, "n_kv_heads", 1)
        full_cap_alloc = doc_cap_full + slack
        layers: list[dict[str, Any]] = []
        eval_arrs: list[mx.array] = []
        for i in range(self.first_kv_shared):
            l = self.layers[i]
            D = l["head_dim"]
            n_kv = l.get("n_kv_heads", default_n_kv)
            ring = not l["is_full"]
            cap = swa_ring_cap if ring else full_cap_alloc
            if use_kv_i8:
                c: dict[str, Any] = {
                    "k": mx.zeros((batch_size, cap, n_kv, D), dtype=mx.int8),
                    "k_s": mx.ones((batch_size, cap, n_kv), dtype=mx.float16),
                    "v": mx.zeros((batch_size, cap, n_kv, D), dtype=mx.int8),
                    "v_s": mx.ones((batch_size, cap, n_kv), dtype=mx.float16),
                    "cap": cap,
                    "ring": ring,
                }
                eval_arrs.extend((c["k"], c["k_s"], c["v"], c["v_s"]))
            else:
                c = {
                    "k": mx.zeros((batch_size, cap, n_kv, D), dtype=self.act_dtype),
                    "v": mx.zeros((batch_size, cap, n_kv, D), dtype=self.act_dtype),
                    "cap": cap,
                    "ring": ring,
                }
                eval_arrs.extend((c["k"], c["v"]))
            layers.append(c)
        mx.eval(*eval_arrs)

        return {
            "batch_size": batch_size,
            "l_sys": int(prefix_kv["l_sys"]),
            "k_sys": prefix_kv["k_sys"],
            "v_sys": prefix_kv["v_sys"],
            "layers": layers,
            "swa_cap": swa_ring_cap,
            "doc_cap_full": doc_cap_full,
            "full_cap": full_cap_alloc,
            "drain_slack": slack,
            "kv_i8": use_kv_i8,
            "meta_cache": {},
        }

    def _get_sdpa_meta(self, cache: dict[str, Any], rows_cur: int) -> tuple[int, int, mx.array, mx.array]:
        ns_swa = 1 if rows_cur >= 128 else (2 if rows_cur >= 96 else 4)
        ns_full = 1 if rows_cur >= 128 else (4 if rows_cur >= 96 else 8)
        key = (ns_swa, ns_full)
        m = cache["meta_cache"].get(key)
        if m is None:
            l_sys = cache["l_sys"]
            swa_cap = cache["swa_cap"]
            full_cap = cache["full_cap"]
            m = (
                ns_swa,
                ns_full,
                mx.array([l_sys, swa_cap, self.window_size, 1, ns_swa, swa_cap - 1], dtype=mx.int32),
                mx.array([l_sys, full_cap, 0, 0, ns_full, -1], dtype=mx.int32),
            )
            mx.eval(m[2], m[3])
            cache["meta_cache"][key] = m
        return m

    @staticmethod
    def _precompute_scatter_pos(off_b: mx.array, l_sys: int, swa_cap: int, L_q: int) -> tuple[mx.array, mx.array]:
        if L_q == 1:
            priv = off_b - l_sys
        else:
            priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        return priv & (swa_cap - 1), priv

    def forward_early14_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
    ) -> None:
        """Advance active slots through KV-owning layers only (`L_q >= 1`)."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        drain_slack = int(cache.get("drain_slack", 16))
        if L_q > 1 and drain_slack > 0 and L_q > drain_slack:
            raise ValueError(
                f"forward_early14_slot L_q={L_q} exceeds slot cache drain_slack={drain_slack}; "
                "pass a larger `drain_slack` to `make_slot_cache`."
            )
        rows = B * L_q
        use_singleton = B <= 8 and L_q > 1
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B if use_singleton else rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx if L_q == 1 else slot_idx[:, None]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        h, pli = (
            self._compute_embed_and_pli(toks, singleton=True)
            if use_singleton
            else self._compute_embed_and_pli_early14(toks)
        )
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)

        for i in range(self.first_kv_shared):
            l = self.layers[i]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)
            c = doc_layers[i]
            pos_slot = swa_slot if c["ring"] else full_slot

            if i == self.first_kv_shared - 1:
                kv = (
                    _singleton_verify_linear(l["kv_proj_only"], x_in)
                    if use_singleton
                    else self._apply_qlinear(l, "kv_proj_only", x_in)
                )
                k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                    kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                )
                if use_kv_i8:
                    scatter_kv_i8_slot_precomputed(
                        c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                    )
                else:
                    scatter_kv_slot_precomputed(
                        c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                    )
                break

            qkv = (
                _singleton_verify_linear(l["qkv_proj"], x_in)
                if use_singleton
                else self._apply_qlinear(l, "qkv_proj", x_in)
            )
            q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
            )
            if use_kv_i8:
                scatter_kv_i8_slot_precomputed(
                    c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                )
            else:
                scatter_kv_slot_precomputed(
                    c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                )

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[i],
                v_sys_l[i],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )
            attn_out = (
                _singleton_verify_linear(l["o_proj"], att)
                if use_singleton
                else self._apply_qlinear(l, "o_proj", att)
            )
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l, singleton=use_singleton)

        self._eval_doc_layers(doc_layers, use_kv_i8)

    def forward_step_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        apply_softcap: bool = True,
        keep_from_col: int = 0,
    ) -> mx.array:
        """Full slot-indexed step for `L_q >= 1` -> `[B, L_q - keep_from_col, V]`."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if keep_from_col < 0 or keep_from_col >= L_q:
            raise ValueError(f"Expected 0 <= keep_from_col < L_q ({L_q}), got {keep_from_col}")
        rows = B * L_q
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx if L_q == 1 else slot_idx[:, None]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                if i == split_idx and keep_from_col > 0:
                    kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    if pli is not None:
                        pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = self._apply_qlinear(l, "q_proj_split", x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B * (L_q - keep_from_col))
                else:
                    qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
            else:
                q_raw = self._apply_qlinear(l, "q_proj", x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = self._apply_qlinear(l, "o_proj", att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)

        cache["_last_x_in"] = x_in
        return self._apply_lm_head(x_in, apply_softcap=apply_softcap)

    def forward_ff_jump_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        return_logits: bool = False,
        collect_telemetry: bool = False,
        valid_mask_u32: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Single-launch causal slice jump over `k >= 1` forced `llguidance` tokens (`[B, k]`)."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        drain_slack = int(cache.get("drain_slack", 16))
        if L_q > 1 and drain_slack > 0 and L_q > drain_slack:
            raise ValueError(
                f"forward_ff_jump_slot k={L_q} exceeds slot cache drain_slack={drain_slack}; "
                "pass a larger `drain_slack` to `make_slot_cache`."
            )
        if L_q == 1:
            return self.forward_decode_step_slot(
                toks,
                cache,
                off_b,
                slot_idx,
                n_keep=B,
                return_logits=return_logits,
                collect_telemetry=collect_telemetry,
                valid_mask_u32=valid_mask_u32,
            )
        rows = B * L_q
        use_singleton = B <= 8
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B if use_singleton else rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx[:, None]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        h, pli = self._compute_embed_and_pli(toks, singleton=use_singleton)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        keep_from_col = L_q - 1

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)
            still_multi = int(x_in.shape[1]) > 1
            step_single = use_singleton and still_multi

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                if i == split_idx:
                    kv = (
                        _singleton_verify_linear(l["kv_proj_only"], x_in)
                        if step_single
                        else self._apply_qlinear(l, "kv_proj_only", x_in)
                    )
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    if pli is not None:
                        pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = self._apply_qlinear(l, "q_proj_split", x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
                    step_single = False
                else:
                    qkv = (
                        _singleton_verify_linear(l["qkv_proj"], x_in)
                        if step_single
                        else self._apply_qlinear(l, "qkv_proj", x_in)
                    )
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
            else:
                q_raw = self._apply_qlinear(l, "q_proj", x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = (
                _singleton_verify_linear(l["o_proj"], att)
                if step_single
                else self._apply_qlinear(l, "o_proj", att)
            )
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l, singleton=step_single)

        cache["_last_x_in"] = x_in
        want_telem = bool(collect_telemetry or valid_mask_u32 is not None)
        if return_logits:
            lg = self._apply_lm_head(x_in, apply_softcap=True)
            if want_telem:
                _, telem_mx = self._apply_lm_head_argmax(
                    x_in, collect_telemetry=True, valid_mask_u32=valid_mask_u32
                )
                cache["_last_telemetry"] = telem_mx
            am_mx = safe_argmax_with_nan_guard(lg[:, 0, :], axis=-1)
            return lg, am_mx
        if want_telem:
            am_mx, telem_mx = self._apply_lm_head_argmax(
                x_in, collect_telemetry=True, valid_mask_u32=valid_mask_u32
            )
            cache["_last_telemetry"] = telem_mx
            return x_in, am_mx
        am_mx = self._apply_lm_head_argmax(x_in)
        assert isinstance(am_mx, mx.array)
        return x_in, am_mx

    def forward_decode_step_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        n_keep: int,
        return_logits: bool = False,
        collect_telemetry: bool = False,
        valid_mask_u32: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Run `L_q=1` slot-indexed decode step with Layer-15 Prefix Split (`n_keep <= B`) & lazy logits.
        When `return_logits=False`, returns `(x_samp [n_keep, 1, D], am_mx [n_keep])`.
        When `return_logits=True`, returns `(logits [n_keep, 1, V], am_mx [n_keep])`.
        When `collect_telemetry=True` (or `valid_mask_u32` is provided), populates `cache['_last_telemetry']`
        in the same fused kernel pass with zero extra `[B, V]` reads.
        """
        B = int(toks.shape[0])
        if not (0 < n_keep <= B):
            raise ValueError(f"Expected 0 < n_keep <= B, got n_keep={n_keep}, B={B}")
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, 1)
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                )
                if use_kv_i8:
                    scatter_kv_i8_slot_precomputed(
                        c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, slot_idx, pos_slot
                    )
                else:
                    scatter_kv_slot_precomputed(
                        c["k"], c["v"], k_rot, v_norm, slot_idx, pos_slot
                    )
                if i == split_idx and n_keep < B:
                    q_rot = q_rot[:n_keep]
                    h = h[:n_keep]
                    if pli is not None:
                        pli = pli[:n_keep]
                    off_b = off_b[:n_keep]
                    slot_idx = slot_idx[:n_keep]
                    B = n_keep
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
            else:
                q_raw = self._apply_qlinear(l, "q_proj", x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = self._apply_qlinear(l, "o_proj", att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)

        cache["_last_x_in"] = x_in
        want_telem = bool(collect_telemetry or valid_mask_u32 is not None)
        if return_logits:
            lg = self._apply_lm_head(x_in, apply_softcap=True)
            if want_telem:
                _, telem_mx = self._apply_lm_head_argmax(
                    x_in, collect_telemetry=True, valid_mask_u32=valid_mask_u32
                )
                cache["_last_telemetry"] = telem_mx
            am_mx = safe_argmax_with_nan_guard(lg[:, 0, :], axis=-1)
            return lg, am_mx
        if want_telem:
            am_mx, telem_mx = self._apply_lm_head_argmax(
                x_in, collect_telemetry=True, valid_mask_u32=valid_mask_u32
            )
            cache["_last_telemetry"] = telem_mx
            return x_in, am_mx
        am_mx = self._apply_lm_head_argmax(x_in)
        assert isinstance(am_mx, mx.array)
        return x_in, am_mx

    def forward_verify_mtp_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array | None = None,
        *,
        return_logits: bool = False,
        keep_from_col: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Exact `K+1` MTP verification (+ optional `keep_from_col` FF jump) matching `L_q=1` `0.0` ULP."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if keep_from_col < 0 or keep_from_col >= L_q:
            raise ValueError(f"Expected 0 <= keep_from_col < L_q ({L_q}), got {keep_from_col}")
        is_slot_cache = "swa_cap" in cache
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)
        split_idx = self.first_kv_shared - 1

        if is_slot_cache:
            swa_cap = cache["swa_cap"]
            use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
            eff_slot_idx = mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx
            # Key split count on `B` (matching `L_q=1` `forward_decode_step_slot` reduction tree!).
            ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
            swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
            s_idx_sc = eff_slot_idx if L_q == 1 else eff_slot_idx[:, None]
        else:
            use_kv_i8 = False
            eff_slot_idx = mx.arange(B, dtype=mx.int32)
            ns_swa = cache["ns_swa"]
            ns_full = cache["ns_full"]
            meta_swa = cache["meta_swa"]
            meta_full = cache["meta_full"]

        use_singleton = B == 1

        def _verify_lin(mod: nn.Module, x: mx.array) -> mx.array:
            return _singleton_verify_linear(mod, x) if use_singleton else mod(x)

        h, pli = self._compute_embed_and_pli(toks, singleton=use_singleton)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)

            if not l["is_shared"]:
                if i == split_idx and keep_from_col > 0 and "kv_proj_only" in l:
                    kv = _verify_lin(l["kv_proj_only"], x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps, n_kv=n_kv
                    )
                    if is_slot_cache:
                        pos_slot = swa_slot if c["ring"] else full_slot
                        if use_kv_i8:
                            scatter_kv_i8_slot_precomputed(
                                c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                            )
                        else:
                            scatter_kv_slot_precomputed(
                                c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                            )
                    else:
                        ring_cap = c["cap"] if c["ring"] else 0
                        scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    if pli is not None:
                        pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = _verify_lin(l["q_proj_split"], x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)
                else:
                    qkv = _verify_lin(l["qkv_proj"], x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                    )
                    if is_slot_cache:
                        pos_slot = swa_slot if c["ring"] else full_slot
                        if use_kv_i8:
                            scatter_kv_i8_slot_precomputed(
                                c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                            )
                        else:
                            scatter_kv_slot_precomputed(
                                c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                            )
                    else:
                        ring_cap = c["cap"] if c["ring"] else 0
                        scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            else:
                q_raw = _verify_lin(l["q_proj"], x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            if is_slot_cache:
                att = mqa8_sdpa_prefix_ragged_slot(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    eff_slot_idx,
                    ns_full if l["is_full"] else ns_swa,
                    meta_full if l["is_full"] else meta_swa,
                    k_s_doc=c.get("k_s"),
                    v_s_doc=c.get("v_s"),
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0 if l["is_full"] else self.window_size,
                    doc_ring_cap=0 if l["is_full"] else c["cap"],
                    n_splits=ns_full if l["is_full"] else ns_swa,
                    meta=meta_full if l["is_full"] else meta_swa,
                    vec4_reduce=self.vec4_reduce,
                )

            attn_out = _verify_lin(l["o_proj"], att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l, singleton=use_singleton)

        L_out = int(x_in.shape[1])
        cache["_last_x_in"] = x_in
        raw_lg = _verify_lin(self.lm_head, x_in)
        if return_logits:
            lg = logit_softcap(self.softcap, raw_lg)
            am_flat = safe_argmax_with_nan_guard(lg.reshape(B * L_out, -1), axis=-1).reshape(B, L_out)
            return lg, am_flat
        if self.use_fused_softcap_argmax and raw_lg.dtype == mx.bfloat16:
            am_flat = fused_softcap_argmax_bf16(raw_lg, self.softcap)
            assert isinstance(am_flat, mx.array)
            return x_in, am_flat.reshape(B, L_out)
        lg = logit_softcap(self.softcap, raw_lg)
        am_flat = safe_argmax_with_nan_guard(lg.reshape(B * L_out, -1), axis=-1).reshape(B, L_out)
        return x_in, am_flat

    @staticmethod
    def _real_token_pairs(pre_lens: Sequence[int], s0: int, s1: int) -> tuple[np.ndarray, np.ndarray] | None:
        """`(doc row, chunk column)` of the real tokens in chunk `[s0, s1)`, or None when every doc fills it."""
        if min(pre_lens) >= s1:
            return None
        rows, cols = np.nonzero(np.arange(s0, s1)[None, :] < np.asarray(pre_lens)[:, None])
        return rows.astype(np.int32), cols.astype(np.int32)

    @staticmethod
    def _scatter_ring_chunk(c: dict[str, Any], b_idx: Any, slots: Any, values: dict[str, Any], real: Any) -> None:
        if real is None:
            for key, val in values.items():
                c[key][b_idx[:, None], slots[None, :]] = val
            return
        rows, cols = real
        bb, ss = b_idx[rows], slots[cols]
        for key, val in values.items():
            c[key][bb, ss] = val[rows, cols]

    @staticmethod
    def _kv_cache_values(k_rot: mx.array, v_norm: mx.array, use_kv_i8: bool) -> dict[str, mx.array]:
        if not use_kv_i8:
            return {"k": k_rot, "v": v_norm}
        kq, ks, vq, vs = quantize_kv_pair_i8(k_rot, v_norm)
        return {"k": kq, "k_s": ks, "v": vq, "v_s": vs}

    def _prefill_early14_slot_group(
        self,
        sub_toks: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        chunk_sz: int = 256,
        start_s0: int = 0,
    ) -> None:
        Bg = len(slot_indices)
        l_sys = cache["l_sys"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        pre_lens = [len(s) - 1 for s in sub_toks]
        max_pre = max(pre_lens)
        if max_pre <= start_s0:
            return
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        pad_arr = np.zeros((Bg, max_pre), dtype=np.int32)
        for j, s in enumerate(sub_toks):
            if pre_lens[j] > 0:
                pad_arr[j, : pre_lens[j]] = s[: pre_lens[j]]
        toks_mx = mx.array(pad_arr, dtype=mx.int32)
        b_idx = mx.array(slot_indices, dtype=mx.int32)

        for s0 in range(start_s0, max_pre, chunk_sz):
            s1 = min(max_pre, s0 + chunk_sz)
            lc = s1 - s0
            off = l_sys + s0
            off_meta = mx.array([off, lc], dtype=mx.int32)
            off_bg = mx.full((Bg,), off, dtype=mx.int32)
            tok_c = toks_mx[:, s0:s1]
            real_np = self._real_token_pairs(pre_lens, s0, s1)
            real = None if real_np is None else (mx.array(real_np[0]), mx.array(real_np[1]))
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                n_q = l.get("n_q_heads", default_n_q)
                n_kv = l.get("n_kv_heads", default_n_kv)
                c = doc_layers[i]
                cap = c["cap"]
                is_4d = c["k"].ndim == 4

                if i == self.first_kv_shared - 1:
                    kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_bg, self.eps, n_kv=n_kv
                    )
                    k_store = k_rot if is_4d else k_rot.reshape(Bg, lc, D)
                    v_store = v_norm if is_4d else v_norm.reshape(Bg, lc, D)
                    if c["ring"]:
                        slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                        self._scatter_ring_chunk(c, b_idx, slots, self._kv_cache_values(k_store, v_store, use_kv_i8), real)
                    else:
                        if use_kv_i8:
                            kq, ks, vq, vs = quantize_kv_pair_i8(k_store, v_store)
                            c["k"][b_idx, s0:s1] = kq
                            c["k_s"][b_idx, s0:s1] = ks
                            c["v"][b_idx, s0:s1] = vq
                            c["v_s"][b_idx, s0:s1] = vs
                        else:
                            c["k"][b_idx, s0:s1] = k_store
                            c["v"][b_idx, s0:s1] = v_store
                    break

                qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps, n_q=n_q, n_kv=n_kv
                )
                k_store = k_rot if is_4d else k_rot.reshape(Bg, lc, D)
                v_store = v_norm if is_4d else v_norm.reshape(Bg, lc, D)
                k_chunk_sdpa = k_rot.transpose(0, 2, 1, 3) if k_rot.ndim == 4 else k_rot[:, None, :, :]
                v_chunk_sdpa = v_norm.transpose(0, 2, 1, 3) if v_norm.ndim == 4 else v_norm[:, None, :, :]

                if not c["ring"]:
                    if use_kv_i8:
                        kq, ks, vq, vs = quantize_kv_pair_i8(k_store, v_store)
                        c["k"][b_idx, s0:s1] = kq
                        c["k_s"][b_idx, s0:s1] = ks
                        c["v"][b_idx, s0:s1] = vq
                        c["v_s"][b_idx, s0:s1] = vs
                        if s0 == 0:
                            kd = k_chunk_sdpa
                            vd = v_chunk_sdpa
                        else:
                            full_pos = mx.arange(0, s1, dtype=mx.int32)
                            kd, vd = gather_dequant_kv_pair_i8_bf16(
                                c["k"], c["k_s"], c["v"], c["v_s"], b_idx, full_pos
                            )
                    else:
                        c["k"][b_idx, s0:s1] = k_store
                        c["v"][b_idx, s0:s1] = v_store
                        if s0 == 0:
                            kd = k_chunk_sdpa
                            vd = v_chunk_sdpa
                        else:
                            kp_prev = c["k"][b_idx, :s1]
                            vp_prev = c["v"][b_idx, :s1]
                            kd = kp_prev.transpose(0, 2, 1, 3) if is_4d else kp_prev[:, None, :, :]
                            vd = vp_prev.transpose(0, 2, 1, 3) if is_4d else vp_prev[:, None, :, :]
                    parts_k = []
                    parts_v = []
                    if l_sys > 0:
                        ks_i = k_sys_l[i].transpose(1, 0, 2)[None] if k_sys_l[i].ndim == 3 else k_sys_l[i][None, None]
                        vs_i = v_sys_l[i].transpose(1, 0, 2)[None] if v_sys_l[i].ndim == 3 else v_sys_l[i][None, None]
                        parts_k.append(mx.broadcast_to(ks_i, (Bg, n_kv, l_sys, D)))
                        parts_v.append(mx.broadcast_to(vs_i, (Bg, n_kv, l_sys, D)))
                    parts_k.append(kd)
                    parts_v.append(vd)
                    kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
                    vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
                    mk = ("full", off, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, off, is_full=True, window_size=None, dtype=self.act_dtype
                        )
                else:
                    slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                    t_start = max(0, off - self.window_size + 1)
                    k_parts = []
                    v_parts = []
                    if t_start < l_sys:
                        sys_lo = t_start
                        ks_sl = k_sys_l[i][sys_lo:l_sys]
                        vs_sl = v_sys_l[i][sys_lo:l_sys]
                        ks_b = ks_sl.transpose(1, 0, 2)[None] if ks_sl.ndim == 3 else ks_sl[None, None]
                        vs_b = vs_sl.transpose(1, 0, 2)[None] if vs_sl.ndim == 3 else vs_sl[None, None]
                        k_parts.append(mx.broadcast_to(ks_b, (Bg, n_kv, l_sys - sys_lo, D)))
                        v_parts.append(mx.broadcast_to(vs_b, (Bg, n_kv, l_sys - sys_lo, D)))
                    priv_lo = max(0, t_start - l_sys)
                    if priv_lo < s0:
                        p_slots = mx.arange(priv_lo, s0, dtype=mx.int32) & (cap - 1)
                        if use_kv_i8:
                            kd_old, vd_old = gather_dequant_kv_pair_i8_bf16(
                                c["k"], c["k_s"], c["v"], c["v_s"], b_idx, p_slots
                            )
                        else:
                            kp_old = c["k"][b_idx[:, None], p_slots[None, :]]
                            vp_old = c["v"][b_idx[:, None], p_slots[None, :]]
                            kd_old = kp_old.transpose(0, 2, 1, 3) if is_4d else kp_old[:, None, :, :]
                            vd_old = vp_old.transpose(0, 2, 1, 3) if is_4d else vp_old[:, None, :, :]
                        k_parts.append(kd_old)
                        v_parts.append(vd_old)
                    if priv_lo < s1:
                        chunk_lo = max(0, priv_lo - s0)
                        k_parts.append(k_chunk_sdpa if chunk_lo == 0 else k_chunk_sdpa[:, :, chunk_lo:, :])
                        v_parts.append(v_chunk_sdpa if chunk_lo == 0 else v_chunk_sdpa[:, :, chunk_lo:, :])
                    self._scatter_ring_chunk(c, b_idx, slots, self._kv_cache_values(k_store, v_store, use_kv_i8), real)
                    kf = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                    vf = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                    mk = ("swa", off - t_start, lc)
                    if mk not in masks:
                        masks[mk] = _select_sdpa_mask(
                            lc, off - t_start, is_full=False, window_size=self.window_size, dtype=self.act_dtype
                        )

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(Bg, lc, n_q * D)
                )
                attn_out = self._apply_qlinear(l, "o_proj", att)
                h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)
            self._eval_doc_layers(doc_layers, use_kv_i8)

    def _prefill_early14_slot_groups(
        self,
        suf_ids_list: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        group_size: int = 16,
        start_s0: int = 0,
    ) -> None:
        """Prefill in length-sorted groups whose suffixes differ by at most 48 tokens."""
        K = len(suf_ids_list)
        if K != len(slot_indices):
            raise ValueError(f"Mismatched K={K} vs slot_indices={len(slot_indices)}")
        order = sorted(range(K), key=lambda b: len(suf_ids_list[b]))
        g0 = 0
        while g0 < K:
            l0 = len(suf_ids_list[order[g0]])
            g1 = g0 + 1
            while g1 < K and (g1 - g0) < group_size and (len(suf_ids_list[order[g1]]) - l0) <= 48:
                g1 += 1
            sub_idxs = order[g0:g1]
            self._prefill_early14_slot_group(
                [suf_ids_list[b] for b in sub_idxs],
                [slot_indices[b] for b in sub_idxs],
                cache,
                chunk_sz=256,
                start_s0=start_s0,
            )
            g0 = g1
        if K > 0 and callable(getattr(mx, "clear_cache", None)):
            mx.clear_cache()

    def prefill_slots_lazy_lg(
        self,
        suf_ids_list: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        *,
        group_size: int = 16,
        return_logits: bool = False,
        async_eval: bool = False,
        collect_telemetry: bool = False,
        valid_mask_u32: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, list[int]]:
        """Prefill `K` docs into `slot_indices` and run batched `LastTok` (`return_logits=False` skips `[K, 1, V]`)."""
        K = len(suf_ids_list)
        l_sys = cache["l_sys"]
        self._prefill_early14_slot_groups(suf_ids_list, slot_indices, cache, group_size=group_size)
        last_toks = mx.array([suf_ids_list[b][-1] for b in range(K)], dtype=mx.int32)[:, None]
        last_off = mx.array([l_sys + len(suf_ids_list[b]) - 1 for b in range(K)], dtype=mx.int32)
        slot_mx = mx.array(slot_indices, dtype=mx.int32)
        want_telem = bool(collect_telemetry or valid_mask_u32 is not None)
        if want_telem:
            x_or_lg, am = self.forward_decode_step_slot(
                last_toks,
                cache,
                last_off,
                slot_mx,
                n_keep=K,
                return_logits=return_logits,
                collect_telemetry=collect_telemetry,
                valid_mask_u32=valid_mask_u32,
            )
        else:
            x_or_lg, am = self.forward_decode_step_slot(
                last_toks, cache, last_off, slot_mx, n_keep=K, return_logits=return_logits
            )
        next_off = [l_sys + len(suf_ids_list[b]) for b in range(K)]
        telem_mx = cache.get("_last_telemetry") if want_telem else None
        if async_eval:
            if telem_mx is not None:
                mx.async_eval(x_or_lg, am, telem_mx)
            else:
                mx.async_eval(x_or_lg, am)
        else:
            if telem_mx is not None:
                mx.eval(x_or_lg, am, telem_mx)
            else:
                mx.eval(x_or_lg, am)
        return x_or_lg, am, next_off

    def build_cpu_ane_embed_tables(self):
        """One-time dequantization of `embed_tokens` + PLI layers `0..13` to CPU FP16 tables (~2.68 GB)."""
        emb = self.embed_tokens
        pli = self.embed_tokens_per_layer
        if isinstance(emb, nn.QuantizedEmbedding):
            w_emb = (
                mx.dequantize(emb.weight, emb.scales, emb.biases, group_size=emb.group_size, bits=emb.bits, mode=emb.mode)
                * self.embed_scale
            ).astype(mx.float16)
        else:
            w_emb = (emb.weight * self.embed_scale).astype(mx.float16)
        mx.eval(w_emb)

        n_pli_cols = max(0, (self.first_kv_shared - 1) * self.ple_dim)
        if n_pli_cols == 0:
            return np.array(w_emb, copy=True), np.zeros((int(w_emb.shape[0]), 0), dtype=np.float16)
        if getattr(self, "_ple_mmap_i8", None) is not None and getattr(self, "_ple_mmap_scales", None) is not None:
            sc = float(self.embed_tokens_per_layer_scale.item())
            w_pli14_np = (
                self._ple_mmap_i8[:, :n_pli_cols].astype(np.float32)
                * self._ple_mmap_scales[:, None].astype(np.float32)
                * sc
            ).astype(np.float16)
            return np.array(w_emb, copy=True), np.ascontiguousarray(w_pli14_np)
        assert pli is not None
        if isinstance(pli, nn.QuantizedEmbedding):
            pack_div = 32 // pli.bits
            w_words = n_pli_cols // pack_div
            g_cnt = n_pli_cols // pli.group_size
            w_pli14 = (
                mx.dequantize(
                    pli.weight[:, :w_words],
                    pli.scales[:, :g_cnt],
                    pli.biases[:, :g_cnt] if pli.biases is not None else None,
                    group_size=pli.group_size,
                    bits=pli.bits,
                    mode=pli.mode,
                )
                * self.embed_tokens_per_layer_scale
            ).astype(mx.float16)
        else:
            w_pli14 = (pli.weight[:, :n_pli_cols] * self.embed_tokens_per_layer_scale).astype(mx.float16)

        mx.eval(w_pli14)
        return np.array(w_emb, copy=True), np.array(w_pli14, copy=True)

    # ------------------------------------------------------------------
    # Standalone Exact Export (Zero mlx_vlm) for CoreML ANE Compilation
    # ------------------------------------------------------------------
    def export_early14_f16(
        self,
        out_npz: str | Path,
        *,
        safe_fp16_residual: bool = True,
        residual_scales: Sequence[float] | None = None,
    ) -> Path:
        """Export EarlyExit layers `0..first_kv_shared-1` FP16 weights in CoreML `[O, I, 1, 1]` layout.

        When `safe_fp16_residual=True` (default), folds exact power-of-2 residual scaling across deep
        layers (e.g. `L10..L14`) into `w_pa`, `w_poff`, `w_pple`, `w_pg`, and `layer_scalar` so Gemma-4
        `L13` residuals (`~215K`) stay safely within FP16 range (`<= 65,504`) without breaking ANE placement.
        """
        import numpy as np

        if int(getattr(self, "ple_dim", 0)) <= 0 or self.per_layer_model_projection is None:
            raise ValueError(
                "export_early14_f16 requires ple_dim > 0 (Early14 ANE offload is only supported on PLE-enabled Gemma-4 models)."
            )

        def _lin_w_f32(mod: nn.Module) -> np.ndarray:
            if isinstance(mod, nn.QuantizedLinear):
                w = mx.dequantize(
                    mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode
                ).astype(mx.float32)
            else:
                w = mod.weight.astype(mx.float32)
            mx.eval(w)
            return np.ascontiguousarray(np.asarray(w, dtype=np.float32))[:, :, None, None]

        def _lin_w_f16(mod: nn.Module) -> np.ndarray:
            return np.ascontiguousarray(_lin_w_f32(mod).astype(np.float16))

        k_layers = int(self.first_kv_shared)
        w_pg_f32_map: dict[int, np.ndarray] = {}
        w_pg_max_abs: list[float] = []
        for i in range(k_layers - 1):
            w_pg_f32 = _lin_w_f32(self.layers[i]["ple_gate"])
            w_pg_f32_map[i] = w_pg_f32
            w_pg_max_abs.append(float(np.max(np.abs(w_pg_f32))) if w_pg_f32.size else 0.0)

        if residual_scales is not None:
            r_scales = np.asarray(residual_scales, dtype=np.float32)
            if r_scales.shape != (k_layers,) or np.any(r_scales <= 0.0) or not np.all(np.isfinite(r_scales)):
                raise ValueError(f"residual_scales must be positive finite length {k_layers}, got {r_scales}")
        elif safe_fp16_residual:
            r_scales = compute_safe_fp16_residual_scales(k_layers, w_pg_max_abs=w_pg_max_abs)
        else:
            r_scales = np.ones((k_layers,), dtype=np.float32)

        arrays: dict[str, np.ndarray] = {
            "per_layer_model_proj_w": _lin_w_f16(self.per_layer_model_projection)[: (k_layers - 1) * self.ple_dim],
            "per_layer_proj_scale": np.array([float(self.per_layer_projection_scale.item())], dtype=np.float16),
            "per_layer_proj_norm_w": np.array(self.per_layer_projection_norm_w.astype(mx.float16), copy=True),
            "per_layer_input_scale": np.array([float(self.per_layer_input_scale.item())], dtype=np.float16),
            "residual_scales": r_scales.astype(np.float32),
            "fp16_residual_protected": np.array([1 if np.any(r_scales != 1.0) else 0], dtype=np.int32),
        }
        for i in range(k_layers):
            l = self.layers[i]
            p = f"l{i}_"
            alpha_i = float(r_scales[i])
            arrays[p + "w_in"] = np.array(l["w_in"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
            if i == k_layers - 1:
                arrays[p + "kv"] = _lin_w_f16(l["kv_proj_only"])
            else:
                arrays[p + "qkv"] = _lin_w_f16(l["qkv_proj"])
            arrays[p + "w_kn"] = np.array(l["w_kn"].astype(mx.float16), copy=True).reshape(1, 1, 1, l["head_dim"])
            if i < k_layers - 1:
                alpha_next = float(r_scales[i + 1])
                arrays[p + "w_qn"] = np.array(l["w_qn"].astype(mx.float16), copy=True).reshape(1, 1, 1, l["head_dim"])
                arrays[p + "wo"] = _lin_w_f16(l["o_proj"])
                w_pa = np.asarray(l["w_post_attn"].astype(mx.float32), dtype=np.float32) * alpha_i
                arrays[p + "w_pa"] = np.ascontiguousarray(w_pa, dtype=np.float16).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "w_pff"] = np.array(l["w_pre_ff"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "gu"] = _lin_w_f16(l["gate_up_proj"])
                arrays[p + "dn"] = _lin_w_f16(l["down_proj"])
                w_poff = np.asarray(l["w_post_ff"].astype(mx.float32), dtype=np.float32) * alpha_i
                arrays[p + "w_poff"] = np.ascontiguousarray(w_poff, dtype=np.float16).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "w_pg"] = np.ascontiguousarray((w_pg_f32_map[i] / alpha_i).astype(np.float16))
                arrays[p + "w_pp"] = _lin_w_f16(l["ple_proj"])
                w_pple = np.asarray(l["w_post_ple"].astype(mx.float32), dtype=np.float32) * alpha_i
                arrays[p + "w_pple"] = np.ascontiguousarray(w_pple, dtype=np.float16).reshape(1, self.hidden_size, 1, 1)
                sc_val = float(l["layer_scalar"].astype(mx.float32).item()) * (alpha_next / alpha_i)
                arrays[p + "scalar"] = np.array([sc_val], dtype=np.float16)

        out_p = Path(out_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_p, **arrays)
        return out_p

    # ------------------------------------------------------------------
    # Lazy Multimodal Prefill (Isolated from Text-Only Kernels & Decode)
    # ------------------------------------------------------------------
    def _compute_pli_from_embeds(
        self,
        inputs_embeds: mx.array,
        ple_toks: mx.array,
        *,
        early14: bool = False,
    ) -> mx.array | None:
        """Compute per-layer inputs from pre-scattered `inputs_embeds` and text-only `ple_toks`."""
        if getattr(self, "ple_dim", 256) <= 0:
            return None
        b_sz, seq_l = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        if early14:
            if self._n_early_pli_layers <= 0:
                return None
            n_l = self._n_early_pli_layers
            n_cols = n_l * self.ple_dim
            if getattr(self, "_ple_mmap_i8", None) is not None:
                pli_tok = self._gather_mmap_ple_rows(ple_toks, n_cols)
            else:
                etpl = self.embed_tokens_per_layer
                assert etpl is not None
                if self._etpl_14_groups > 0:
                    w_sub = etpl.weight[ple_toks][..., : self._etpl_14_wcols]
                    s_sub = etpl.scales[ple_toks][..., : self._etpl_14_groups]
                    b_sub = etpl.biases[ple_toks][..., : self._etpl_14_groups] if etpl.biases is not None else None
                    pli_tok = mx.dequantize(
                        w_sub,
                        scales=s_sub,
                        biases=b_sub,
                        group_size=etpl.group_size,
                        bits=etpl.bits,
                        mode=etpl.mode,
                    ) * self.embed_tokens_per_layer_scale
                else:
                    raw_etpl = etpl(ple_toks)[..., :n_cols] if isinstance(etpl, nn.QuantizedEmbedding) else etpl.weight[ple_toks][..., :n_cols]
                    pli_tok = raw_etpl.astype(self.act_dtype) * self.embed_tokens_per_layer_scale
            pli_tok = pli_tok.reshape(b_sz, seq_l, n_l, self.ple_dim)
            assert self.pli_proj_14 is not None
            pli_mod = (self.pli_proj_14(inputs_embeds) * self.per_layer_projection_scale).reshape(
                b_sz, seq_l, n_l, self.ple_dim
            )
            pli_mod = mx.fast.rms_norm(pli_mod, self.per_layer_projection_norm_w, self.eps)
            return (pli_mod + pli_tok) * self.per_layer_input_scale

        tot_cols = self.num_layers * self.ple_dim
        if getattr(self, "_ple_mmap_i8", None) is not None:
            pli_raw = self._gather_mmap_ple_rows(ple_toks, tot_cols).reshape(b_sz, seq_l, self.num_layers, self.ple_dim)
        else:
            assert self.embed_tokens_per_layer is not None
            pli_raw = (self.embed_tokens_per_layer(ple_toks) * self.embed_tokens_per_layer_scale).reshape(
                b_sz, seq_l, self.num_layers, self.ple_dim
            )
        assert self.per_layer_model_projection is not None
        pmp_out = self.per_layer_model_projection(inputs_embeds)
        proj = (pmp_out * self.per_layer_projection_scale).reshape(b_sz, seq_l, self.num_layers, self.ple_dim)
        proj = mx.fast.rms_norm(proj, self.per_layer_projection_norm_w, self.eps)
        return (proj + pli_raw) * self.per_layer_input_scale

    def _build_vision_sdpa_mask(
        self,
        q_start: int,
        q_len: int,
        k_start: int,
        kv_len: int,
        vis_group_ids: np.ndarray,
        *,
        is_full: bool,
    ) -> mx.array:
        """Build `0.0 / -inf` `act_dtype` mask combining causal/SWA rules with same-image bidirectional overlay."""
        q_pos = np.arange(q_start, q_start + q_len, dtype=np.int32)[:, None]
        k_pos = np.arange(k_start, k_start + kv_len, dtype=np.int32)[None, :]
        allowed = q_pos >= k_pos
        if not is_full:
            allowed = allowed & ((q_pos - k_pos) < int(self.window_size))

        total_g = int(vis_group_ids.shape[0])
        q_clamped = np.clip(np.arange(q_start, q_start + q_len, dtype=np.int32), 0, max(total_g - 1, 0))
        q_valid = (np.arange(q_start, q_start + q_len, dtype=np.int32) >= 0) & (
            np.arange(q_start, q_start + q_len, dtype=np.int32) < total_g
        )
        q_gid = np.where(q_valid, vis_group_ids[q_clamped], -1)[:, None]

        k_idx = np.arange(k_start, k_start + kv_len, dtype=np.int32)
        k_valid = (k_idx >= 0) & (k_idx < total_g)
        k_gid = np.where(k_valid, vis_group_ids[np.clip(k_idx, 0, max(total_g - 1, 0))], -1)[None, :]

        same_block = (q_gid >= 0) & (q_gid == k_gid)
        if not is_full:
            same_block = same_block & (np.abs(q_pos - k_pos) < int(self.window_size))
        allowed = allowed | same_block
        zero = mx.array(0.0, dtype=self.act_dtype)
        neg_inf = mx.array(-float("inf"), dtype=self.act_dtype)
        return mx.where(mx.array(allowed, dtype=mx.bool_), zero, neg_inf)

    def _forward_step_embeds_b1(
        self,
        inputs_embeds: mx.array,
        ple_toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        *,
        apply_softcap: bool = True,
    ) -> mx.array:
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]

        h = inputs_embeds.astype(self.act_dtype)
        pli = self._compute_pli_from_embeds(h, ple_toks, early14=False)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            cap = c["cap"]
            ring_cap = cap if c["ring"] else 0
            n_q = l.get("n_q_heads", default_n_q)
            n_kv = l.get("n_kv_heads", default_n_kv)

            if not l["is_shared"]:
                qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps, n_q=n_q, n_kv=n_kv
                )
                scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            else:
                q_raw = self._apply_qlinear(l, "q_proj", x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps, n_q=n_q)

            if l["is_full"]:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0,
                    doc_ring_cap=0,
                    n_splits=cache["ns_full"],
                    meta=cache["meta_full"],
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=self.window_size,
                    doc_ring_cap=cap,
                    n_splits=cache["ns_swa"],
                    meta=cache["meta_swa"],
                    vec4_reduce=self.vec4_reduce,
                )
            attn_out = self._apply_qlinear(l, "o_proj", att)
            h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)

        cache["_last_x_in"] = x_in
        return self._apply_lm_head(x_in, apply_softcap=apply_softcap)

    def prefill_multimodal_b1(
        self,
        prompt_ids: Sequence[int],
        inputs_embeds: mx.array,
        cache: dict[str, Any],
        *,
        image_token_id: int | None = None,
        chunk_sz: int = 256,
        apply_softcap: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Prefill `[1, L, H]` multimodal prompt into `batch_size=1` cache and emit last-token logits."""
        if int(cache["batch_size"]) != 1:
            raise ValueError(f"prefill_multimodal_b1 requires cache batch_size=1, got {cache['batch_size']}")
        seq_len = len(prompt_ids)
        if seq_len <= 0 or int(inputs_embeds.shape[1]) != seq_len:
            raise ValueError(f"Mismatched prompt_ids len ({seq_len}) vs inputs_embeds ({inputs_embeds.shape})")

        l_sys = int(cache["l_sys"])
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        default_n_q = getattr(self, "n_q_heads", 8)
        default_n_kv = getattr(self, "n_kv_heads", 1)
        img_tid = self.image_token_id if image_token_id is None else int(image_token_id)
        ple_ids_np = np.array([0 if int(t) == img_tid else int(t) for t in prompt_ids], dtype=np.int32)

        pre_len = seq_len - 1
        if pre_len > 0:
            vis_group_ids = np.full((seq_len,), -1, dtype=np.int32)
            grp_id = -1
            in_grp = False
            for idx_t in range(seq_len):
                if int(prompt_ids[idx_t]) == img_tid:
                    if not in_grp:
                        grp_id += 1
                        in_grp = True
                    vis_group_ids[idx_t] = grp_id
                else:
                    in_grp = False

            vg_list = [int(x) for x in vis_group_ids]
            eff_step = max(chunk_sz, 2048) if self.use_bidirectional_vision else chunk_sz
            chunks: list[tuple[int, int]] = []
            s0 = 0
            while s0 < pre_len:
                s1 = min(pre_len, s0 + eff_step)
                if self.use_bidirectional_vision and s1 < pre_len:
                    g_edge = vg_list[s1 - 1]
                    if g_edge >= 0 and vg_list[s1] == g_edge:
                        # Avoid splitting a bidirectional vision block across chunks
                        start_of_g = s1 - 1
                        while start_of_g > s0 and vg_list[start_of_g - 1] == g_edge:
                            start_of_g -= 1
                        if start_of_g > s0:
                            s1 = start_of_g
                        else:
                            while s1 < pre_len and vg_list[s1] == g_edge:
                                s1 += 1
                chunks.append((s0, s1))
                s0 = s1

            b_idx = mx.array([0], dtype=mx.int32)
            for s0, s1 in chunks:
                lc = s1 - s0
                off = l_sys + s0
                off_meta = mx.array([off, lc], dtype=mx.int32)
                off_bg = mx.array([off], dtype=mx.int32)
                h = inputs_embeds[:, s0:s1, :].astype(self.act_dtype)
                ple_c = mx.array(ple_ids_np[s0:s1][None, :], dtype=mx.int32)
                pli = self._compute_pli_from_embeds(h, ple_c, early14=True)
                x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
                masks: dict[tuple[str, int, int, int], Any] = {}

                for i in range(self.first_kv_shared):
                    l = self.layers[i]
                    d_head = l["head_dim"]
                    n_q = l.get("n_q_heads", default_n_q)
                    n_kv = l.get("n_kv_heads", default_n_kv)
                    c = doc_layers[i]
                    cap = c["cap"]
                    is_4d = c["k"].ndim == 4

                    if i == self.first_kv_shared - 1:
                        kv = self._apply_qlinear(l, "kv_proj_only", x_in)
                        k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                            kv, l["w_kn"], l["inv_freq"], off_bg, self.eps, n_kv=n_kv
                        )
                        k_store = k_rot if is_4d else k_rot.reshape(1, lc, d_head)
                        v_store = v_norm if is_4d else v_norm.reshape(1, lc, d_head)
                        if c["ring"]:
                            slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                            c["k"][b_idx[:, None], slots[None, :]] = k_store
                            c["v"][b_idx[:, None], slots[None, :]] = v_store
                        else:
                            c["k"][b_idx, s0:s1] = k_store
                            c["v"][b_idx, s0:s1] = v_store
                        break

                    qkv = self._apply_qlinear(l, "qkv_proj", x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps, n_q=n_q, n_kv=n_kv
                    )
                    k_store = k_rot if is_4d else k_rot.reshape(1, lc, d_head)
                    v_store = v_norm if is_4d else v_norm.reshape(1, lc, d_head)
                    k_chunk_sdpa = k_rot.transpose(0, 2, 1, 3) if k_rot.ndim == 4 else k_rot[:, None, :, :]
                    v_chunk_sdpa = v_norm.transpose(0, 2, 1, 3) if v_norm.ndim == 4 else v_norm[:, None, :, :]

                    if not c["ring"]:
                        c["k"][b_idx, s0:s1] = k_store
                        c["v"][b_idx, s0:s1] = v_store
                        parts_k: list[mx.array] = []
                        parts_v: list[mx.array] = []
                        if l_sys > 0:
                            ks_i = (
                                k_sys_l[i].transpose(1, 0, 2)[None]
                                if k_sys_l[i].ndim == 3
                                else k_sys_l[i][None, None]
                            )
                            vs_i = (
                                v_sys_l[i].transpose(1, 0, 2)[None]
                                if v_sys_l[i].ndim == 3
                                else v_sys_l[i][None, None]
                            )
                            parts_k.append(mx.broadcast_to(ks_i, (1, n_kv, l_sys, d_head)))
                            parts_v.append(mx.broadcast_to(vs_i, (1, n_kv, l_sys, d_head)))
                        if s0 == 0:
                            parts_k.append(k_chunk_sdpa)
                            parts_v.append(v_chunk_sdpa)
                        else:
                            kp_prev = c["k"][b_idx, :s1]
                            vp_prev = c["v"][b_idx, :s1]
                            parts_k.append(kp_prev.transpose(0, 2, 1, 3) if is_4d else kp_prev[:, None, :, :])
                            parts_v.append(vp_prev.transpose(0, 2, 1, 3) if is_4d else vp_prev[:, None, :, :])
                        kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
                        vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
                        mk = ("full", off, lc, int(kf.shape[2]))
                        if mk not in masks:
                            if self.use_bidirectional_vision:
                                masks[mk] = self._build_vision_sdpa_mask(
                                    s0, lc, -l_sys, int(kf.shape[2]), vis_group_ids, is_full=True
                                )
                            else:
                                masks[mk] = _select_sdpa_mask(
                                    lc, off, is_full=True, window_size=None, dtype=self.act_dtype
                                )
                    else:
                        slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                        t_start = max(0, off - self.window_size + 1)
                        k_parts: list[mx.array] = []
                        v_parts: list[mx.array] = []
                        if t_start < l_sys:
                            ks_sl = k_sys_l[i][t_start:l_sys]
                            vs_sl = v_sys_l[i][t_start:l_sys]
                            ks_b = ks_sl.transpose(1, 0, 2)[None] if ks_sl.ndim == 3 else ks_sl[None, None]
                            vs_b = vs_sl.transpose(1, 0, 2)[None] if vs_sl.ndim == 3 else vs_sl[None, None]
                            k_parts.append(mx.broadcast_to(ks_b, (1, n_kv, l_sys - t_start, d_head)))
                            v_parts.append(mx.broadcast_to(vs_b, (1, n_kv, l_sys - t_start, d_head)))
                        priv_lo = max(0, t_start - l_sys)
                        if priv_lo < s0:
                            p_slots = mx.arange(priv_lo, s0, dtype=mx.int32) & (cap - 1)
                            kp_old = c["k"][b_idx[:, None], p_slots[None, :]]
                            vp_old = c["v"][b_idx[:, None], p_slots[None, :]]
                            k_parts.append(kp_old.transpose(0, 2, 1, 3) if is_4d else kp_old[:, None, :, :])
                            v_parts.append(vp_old.transpose(0, 2, 1, 3) if is_4d else vp_old[:, None, :, :])
                        if priv_lo < s1:
                            chunk_lo = max(0, priv_lo - s0)
                            k_parts.append(k_chunk_sdpa if chunk_lo == 0 else k_chunk_sdpa[:, :, chunk_lo:, :])
                            v_parts.append(v_chunk_sdpa if chunk_lo == 0 else v_chunk_sdpa[:, :, chunk_lo:, :])
                        c["k"][b_idx[:, None], slots[None, :]] = k_store
                        c["v"][b_idx[:, None], slots[None, :]] = v_store
                        kf = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                        vf = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                        mk = ("swa", off - t_start, lc, int(kf.shape[2]))
                        if mk not in masks:
                            if self.use_bidirectional_vision:
                                masks[mk] = self._build_vision_sdpa_mask(
                                    s0, lc, t_start - l_sys, int(kf.shape[2]), vis_group_ids, is_full=False
                                )
                            else:
                                masks[mk] = _select_sdpa_mask(
                                    lc, off - t_start, is_full=False, window_size=self.window_size, dtype=self.act_dtype
                                )

                    att = (
                        mx.fast.scaled_dot_product_attention(
                            q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                        )
                        .transpose(0, 2, 1, 3)
                        .reshape(1, lc, n_q * d_head)
                    )
                    attn_out = self._apply_qlinear(l, "o_proj", att)
                    h, x_in = self._run_ffn_and_ple(h, attn_out, pli, i, l)
                mx.eval(*[(c["k"], c["v"]) for c in doc_layers])
            if callable(getattr(mx, "clear_cache", None)):
                mx.clear_cache()

        last_tid = int(prompt_ids[-1])
        last_off = mx.array([l_sys + seq_len - 1], dtype=mx.int32)
        if last_tid != img_tid:
            lg = self.forward_step_ragged(
                mx.array([[last_tid]], dtype=mx.int32),
                cache,
                last_off,
                apply_softcap=apply_softcap,
            )
        else:
            lg = self._forward_step_embeds_b1(
                inputs_embeds[:, -1:, :],
                mx.zeros((1, 1), dtype=mx.int32),
                cache,
                last_off,
                apply_softcap=apply_softcap,
            )
        next_off = mx.array([l_sys + seq_len], dtype=mx.int32)
        mx.eval(lg, next_off)
        return lg, next_off
