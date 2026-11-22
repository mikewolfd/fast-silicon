"""Fused projections, general per-layer spec-bucketing, `lora_accel`, and GDN/RoPE kernels for `fast-qwen3`."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
import inspect
import os
import platform
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class QwenDoubleQuantError(ValueError):
    """Raised when attempting 2-pass `dequant -> merge LoRA -> 4-bit requant` (grid-snap erasure)."""


def _parse_ver_tuple(ver_str: str) -> tuple[int, ...]:
    out: list[int] = []
    for tok in str(ver_str).split("."):
        m = re.match(r"(\d+)", tok)
        if not m:
            break
        out.append(int(m.group(1)))
    return tuple(out)


def detect_apple10_d256_force_fused(
    device_info: Mapping[str, Any] | None = None,
    macos_version: str | None = None,
) -> bool:
    """Return True on Apple10 (M5+) + macOS >= 26.2 + MLX supporting `force_fused` D=256 SDPA."""
    if os.environ.get("MLX_METAL_NO_NAX") == "1":
        return False
    try:
        sig = inspect.signature(mx.fast.scaled_dot_product_attention)
        if "force_fused" not in sig.parameters:
            return False
    except (TypeError, ValueError):
        return False

    info = dict(device_info) if device_info is not None else {}
    if not info:
        try:
            info = dict(mx.device_info())
        except Exception:
            return False
    dev_name = str(info.get("device_name", "")).lower()
    arch = str(info.get("architecture", "")).lower()
    is_m5_plus = ("m5" in dev_name) or ("m6" in dev_name) or ("apple10" in arch) or ("g17" in arch)
    if not is_m5_plus:
        return False
    mac_ver = _parse_ver_tuple(platform.mac_ver()[0] if macos_version is None else macos_version)
    if mac_ver and mac_ver < (26, 2):
        return False
    return True


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


class RMSNormGated(nn.Module):
    """Unshifted Gated RMSNorm (`y_norm * silu(z)` in `float32`) used inside `GatedDeltaNet`."""

    def __init__(self, weight: mx.array, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight.astype(mx.bfloat16)
        self.eps = float(eps)

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        yf = mx.fast.rms_norm(y.astype(mx.float32), self.weight.astype(mx.float32), self.eps)
        zf = nn.silu(z.astype(mx.float32))
        return (yf * zf).astype(y.dtype)


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


def linear_quant_spec_key(lin: nn.Module) -> tuple[bool, int, int, bool, bool]:
    """Return `(is_quant, bits, group_size, has_qbiases, has_linear_bias)` spec key for safe axis-0 fusion."""
    has_lin_bias = getattr(lin, "bias", None) is not None
    if isinstance(lin, nn.QuantizedLinear):
        has_qb = getattr(lin, "biases", None) is not None
        return (True, int(lin.bits), int(lin.group_size), has_qb, has_lin_bias)
    if isinstance(lin, nn.Linear):
        return (False, 16, 0, False, has_lin_bias)
    raise TypeError(f"Unsupported linear module for spec-bucketing: {type(lin)}")


def concat_compatible_linears(mods: Sequence[nn.Module]) -> nn.Module:
    """Horizontally concatenate axis-0 rows of compatible `Linear` or `QuantizedLinear` modules (`0.0` ULP)."""
    if len(mods) == 1:
        return mods[0]
    first = mods[0]
    spec0 = linear_quant_spec_key(first)
    for m in mods[1:]:
        if linear_quant_spec_key(m) != spec0:
            raise ValueError("Cannot concatenate linears with mismatched quantization specs")

    is_quant, bits, group_size, has_qb, has_lin_bias = spec0
    if is_quant:
        in_dims = (int(first.weight.shape[1]) * 32) // bits
        total_out = sum(int(m.weight.shape[0]) for m in mods)
        fused = nn.QuantizedLinear(in_dims, total_out, bias=has_lin_bias, group_size=group_size, bits=bits)
        fused.weight = mx.contiguous(mx.concatenate([m.weight for m in mods], axis=0))
        fused.scales = mx.contiguous(mx.concatenate([m.scales for m in mods], axis=0))
        if has_qb:
            fused.biases = mx.contiguous(mx.concatenate([m.biases for m in mods], axis=0))
        if has_lin_bias:
            fused.bias = mx.contiguous(mx.concatenate([m.bias for m in mods], axis=0))
        return fused

    in_dims = int(first.weight.shape[1])
    total_out = sum(int(m.weight.shape[0]) for m in mods)
    fused_d = nn.Linear(in_dims, total_out, bias=has_lin_bias)
    fused_d.weight = mx.contiguous(mx.concatenate([m.weight for m in mods], axis=0))
    if has_lin_bias:
        fused_d.bias = mx.contiguous(mx.concatenate([m.bias for m in mods], axis=0))
    return fused_d


class SpecBucketedLinearGroup(nn.Module):
    """General arbitrary-order per-layer spec-bucketing over named projections sharing input `x`.

    Handles uniform checkpoints (`1` GEMM) as well as real `OptiQ` non-contiguous mixed-bit
    patterns such as `(4, 8, 4, 4)` in `GatedDeltaNet`, `(4, 4, 8)` / `(8, 4, 8)` in `SelfAttention`,
    and mixed `(gate_proj, up_proj)` MLPs with `0.0` ULP quantization drift.
    """

    def __init__(
        self,
        named_linears: Sequence[tuple[str, nn.Module]],
        *,
        enable_fusion: bool = True,
    ) -> None:
        super().__init__()
        if not named_linears:
            raise ValueError("named_linears must be non-empty")

        self.names: tuple[str, ...] = tuple(nm for nm, _ in named_linears)
        if not enable_fusion:
            self.buckets = [lin for _, lin in named_linears]
            self._bucket_meta: list[tuple[list[str], list[int]]] = [
                ([nm], []) for nm, _ in named_linears
            ]
        else:
            grouped: OrderedDict[tuple[bool, int, int, bool, bool], list[tuple[str, nn.Module]]] = OrderedDict()
            for nm, lin in named_linears:
                key = linear_quant_spec_key(lin)
                grouped.setdefault(key, []).append((nm, lin))

            self.buckets = []
            self._bucket_meta = []
            for entries in grouped.values():
                sub_names = [nm for nm, _ in entries]
                sub_mods = [lin for _, lin in entries]
                fused_mod = concat_compatible_linears(sub_mods)
                stops: list[int] = []
                acc = 0
                for m in sub_mods[:-1]:
                    acc += int(m.weight.shape[0])
                    stops.append(acc)
                self.buckets.append(fused_mod)
                self._bucket_meta.append((sub_names, stops))

        self._lora_stacked_A: mx.array | None = None
        self._lora_meta: list[tuple[str, int, int, mx.array, float]] = []

    @property
    def num_gemms(self) -> int:
        return len(self.buckets)

    def clear_lora(self) -> None:
        self._lora_stacked_A = None
        self._lora_meta = []

    def set_lora_slices(
        self,
        entries: Sequence[tuple[str, mx.array, mx.array, float]],
    ) -> None:
        """Attach hot-swappable `lora_accel` adapters (`(target_name, lora_A[r, in], lora_B[out, r], scale)`)."""
        self.clear_lora()
        if not entries:
            return
        valid_names = set(self.names)
        a_list: list[mx.array] = []
        meta: list[tuple[str, int, int, mx.array, float]] = []
        offset = 0
        for nm, la, lb, scale in entries:
            if nm not in valid_names:
                raise KeyError(f"Unknown target {nm!r} for SpecBucketedLinearGroup({self.names})")
            la_bf = la.astype(mx.bfloat16)
            lb_t = mx.contiguous(lb.astype(mx.bfloat16).T)
            r = int(la_bf.shape[0])
            a_list.append(la_bf)
            meta.append((nm, offset, offset + r, lb_t, float(scale)))
            offset += r
        self._lora_stacked_A = mx.contiguous(mx.concatenate(a_list, axis=0).T)
        self._lora_meta = meta

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        outputs: dict[str, mx.array] = {}
        for bucket_mod, (sub_names, stops) in zip(self.buckets, self._bucket_meta):
            y = bucket_mod(x)
            if not stops:
                outputs[sub_names[0]] = y
            else:
                parts = mx.split(y, stops, axis=-1)
                for nm, part in zip(sub_names, parts):
                    outputs[nm] = part

        if self._lora_stacked_A is not None:
            low = mx.matmul(x.astype(mx.bfloat16), self._lora_stacked_A)
            for nm, s0, s1, lb_t, sc in self._lora_meta:
                delta = (mx.matmul(low[..., s0:s1], lb_t) * sc).astype(outputs[nm].dtype)
                outputs[nm] = outputs[nm] + delta
        return outputs


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
    """Block-32 symmetric `INT8` compression (`4x` memory reduction) for GDN recurrent `[B, Hv, Dv, Dk]` state."""
    b, hv, dv, dk = state.shape
    if dk % 32 != 0:
        raise ValueError(f"GDN Dk must be divisible by 32 for block-32 INT8 codec, got {dk}")
    n_blocks = dk // 32
    blocks = state.astype(mx.float32).reshape(b, hv, dv, n_blocks, 32)
    amax = mx.max(mx.abs(blocks), axis=-1)
    scales_bf16 = (amax * (1.0 / 127.0)).astype(mx.bfloat16)
    scales_f32 = scales_bf16.astype(mx.float32)[..., None]
    safe_denom = mx.where(scales_f32 > 0.0, scales_f32, mx.array(1.0, dtype=mx.float32))
    codes = mx.clip(mx.round(blocks / safe_denom), -127.0, 127.0).astype(mx.int8)
    codes = mx.where(scales_f32 > 0.0, codes, mx.zeros_like(codes))
    return codes.reshape(b, hv, dv, dk), scales_bf16


def dequantize_gdn_state_int8(
    codes: mx.array,
    scales: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Restore `[B, Hv, Dv, Dk]` GDN recurrent state from block-32 `(int8 codes, bfloat16 scales)`."""
    b, hv, dv, dk = codes.shape
    n_blocks = dk // 32
    rec = codes.astype(mx.float32).reshape(b, hv, dv, n_blocks, 32) * scales.astype(mx.float32)[..., None]
    return rec.reshape(b, hv, dv, dk).astype(dtype)


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
    window: int = 128,
    min_repeats: int = 2,
    penalty: float = 4.5,
) -> dict[int, float]:
    """Identify next-token candidates that would repeat an already-repeated 4-gram within `window` tokens."""
    hist = token_history[-window:] if len(token_history) > window else token_history
    n = len(hist)
    if n < 6:
        return {}
    tail3 = (int(hist[-3]), int(hist[-2]), int(hist[-1]))
    counts: dict[int, int] = {}
    for i in range(n - 3):
        if (int(hist[i]), int(hist[i + 1]), int(hist[i + 2])) == tail3:
            nxt = int(hist[i + 3])
            counts[nxt] = counts.get(nxt, 0) + 1
    return {tok: float(penalty) * cnt for tok, cnt in counts.items() if cnt >= min_repeats}
