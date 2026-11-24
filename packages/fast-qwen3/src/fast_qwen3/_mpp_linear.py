"""Apple10+ NAX `W4A8`/`W8A8` prefill acceleration and memory-guarded weight packing for `FastQwen3Model`."""

from __future__ import annotations

import os
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from fast_shared.hw_check import supports_apple10_nax
import fast_shared.metal_mpp_int8 as _metal_mpp_int8
from fast_shared.metal_mpp_int8 import (
    mpp_quantized_linear,
    pack_affine_weights_for_mpp_i8,
    quantize_activations_i8_k64,
)
import fast_shared.mpp_linear as _shared_mpp
from fast_shared.mpp_linear import resolve_mpp_env_bool

_ORIG_PACK_FN = pack_affine_weights_for_mpp_i8
_ORIG_QUANT_FN = quantize_activations_i8_k64
_ORIG_MPP_LINEAR_FN = mpp_quantized_linear

_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})

# Empirically calibrated Apple10 NAX thresholds (Base M5 24 GB + M4 Pro 48 GB parity/Cat-4 suite):
# - 4-bit >= 8M MACs (`Qwen3.5-4B/9B` dense & GDN): `M <= 64` loses `0.33x..0.90x`; `M >= 128` wins `1.08x..1.24x` with 0.0 paired Cat-4 delta.
# - 8-bit >= 8M MACs (`>= 1536` dims): `M >= 64` wins `1.25x..1.50x` (`+8.3%` Cat-4 EOT, delta `-0.00153 <= 0.003` lossless tolerance).
# - 8-bit medium tall/wide (`min >= 1024`, `max >= 3584`, `>= 3.6M` MACs, e.g. OvisOCR2-8bit): `M=64` loses `0.55x..0.68x`; `M >= 128` wins `1.06x..2.48x`.
_MIN_MACS_LARGE = 8_000_000
_MIN_DIM_LARGE = 1536
_MIN_M_W4A8 = 128
_MIN_M_W8A8_LARGE = 64
_MIN_DIM_MEDIUM = 1024
_MAX_DIM_MEDIUM = 3584
_MIN_MACS_MEDIUM = 3_600_000
_MIN_M_W8A8_MEDIUM = 128


def resolve_qwen_mpp_mode(
    *,
    enabled: bool | None = None,
    allow_w8a8: bool | None = None,
) -> tuple[bool, bool]:
    """Resolve `(mpp_enabled, w8a8_enabled)` from explicit args and `FAST_SILICON_QWEN3_MPP*` env overrides."""
    raw_mode = str(os.environ.get("FAST_SILICON_QWEN3_MPP", "")).strip().lower()
    if enabled is not None:
        mpp_on = bool(enabled)
    elif raw_mode in _FALSY:
        mpp_on = False
    else:
        mpp_on = True

    if not mpp_on:
        return False, False

    if allow_w8a8 is not None:
        return True, bool(allow_w8a8)
    if raw_mode in ("w4a8", "4bit", "w4"):
        return True, False
    for env_name in ("FAST_SILICON_QWEN3_MPP_W8A8", "FAST_SILICON_MPP_W8A8"):
        parsed = resolve_mpp_env_bool(env_name)
        if parsed is not None:
            return True, parsed
    return True, False


def _w8a8_min_m(k: int, n: int) -> int | None:
    if (k % 64) != 0 or (n % 64) != 0:
        return None
    elems = k * n
    if k >= _MIN_DIM_LARGE and n >= _MIN_DIM_LARGE and elems >= _MIN_MACS_LARGE:
        return _MIN_M_W8A8_LARGE
    if min(k, n) >= _MIN_DIM_MEDIUM and max(k, n) >= _MAX_DIM_MEDIUM and elems >= _MIN_MACS_MEDIUM:
        return _MIN_M_W8A8_MEDIUM
    return None


def _w4a8_min_m(k: int, n: int) -> int | None:
    if (k % 64) != 0 or (n % 64) != 0:
        return None
    if k >= _MIN_DIM_LARGE and n >= _MIN_DIM_LARGE and (k * n) >= _MIN_MACS_LARGE:
        return _MIN_M_W4A8
    return None


def _qwen_min_m_fn(k: int, n: int, bits: int) -> int | None:
    b = int(bits)
    if b == 4:
        return _w4a8_min_m(int(k), int(n))
    if b == 8:
        return _w8a8_min_m(int(k), int(n))
    return None


def is_eligible_qwen_mpp_shape(
    k: int,
    n: int,
    *,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
) -> bool:
    """Return True if static weight shape/format can ever accelerate under Qwen3 Apple10 MPP prefill."""
    if str(mode) != "affine" or int(group_size) != 64:
        return False
    return _qwen_min_m_fn(int(k), int(n), int(bits)) is not None


def should_use_qwen_mpp_linear(
    m: int,
    k: int,
    n: int,
    *,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
    apple10_nax: bool | None = None,
    allow_w8a8: bool | None = None,
    enabled: bool | None = None,
) -> bool:
    """Gate `FastQwen3Model` Apple10+ NAX `W4A8` (`default`) and `W8A8` (`opt-in`) prefill GEMMs."""
    mpp_on, w8_on = resolve_qwen_mpp_mode(enabled=enabled, allow_w8a8=allow_w8a8)
    if not mpp_on:
        return False
    has_nax = bool(supports_apple10_nax()) if apple10_nax is None else bool(apple10_nax)
    if not has_nax:
        return False
    if str(mode) != "affine" or int(group_size) != 64:
        return False
    b = int(bits)
    if b == 8 and not w8_on:
        return False
    min_m = _qwen_min_m_fn(int(k), int(n), b)
    return min_m is not None and int(m) >= min_m


class QwenMppPackState(_shared_mpp.MppPackState):
    """Shared lazy weight-pack cache, budget guard, and group activation-quantization slot."""

    def __init__(
        self,
        *,
        apple10_nax: bool,
        allow_w8a8: bool,
        enabled: bool = True,
        max_pack_bytes: int | None = None,
        min_avail_bytes: int | None = None,
    ) -> None:
        super().__init__(
            apple10_nax=apple10_nax,
            allow_w8a8=allow_w8a8,
            enabled=enabled,
            min_m_fn=_qwen_min_m_fn,
            max_pack_bytes=max_pack_bytes,
            min_avail_bytes=min_avail_bytes,
        )

    def get_or_pack(
        self,
        lin: nn.QuantizedLinear,
        *,
        bits: int,
        k: int,
        n: int,
    ) -> tuple[mx.array, mx.array] | None:
        if pack_affine_weights_for_mpp_i8 is not _ORIG_PACK_FN:
            orig = _shared_mpp.pack_affine_weights_for_mpp_i8
            _shared_mpp.pack_affine_weights_for_mpp_i8 = pack_affine_weights_for_mpp_i8
            try:
                return super().get_or_pack(lin, bits=bits, k=k, n=n)
            finally:
                _shared_mpp.pack_affine_weights_for_mpp_i8 = orig
        return super().get_or_pack(lin, bits=bits, k=k, n=n)


class MppQuantizedLinear(_shared_mpp.MppQuantizedLinear):
    """Delegate `MppQuantizedLinear` to `fast_shared.mpp_linear` while honoring local monkeypatches."""

    def __call__(self, x: mx.array) -> mx.array:
        if mpp_quantized_linear is not _ORIG_MPP_LINEAR_FN:
            orig = _shared_mpp.mpp_quantized_linear
            _shared_mpp.mpp_quantized_linear = mpp_quantized_linear
            try:
                return super().__call__(x)
            finally:
                _shared_mpp.mpp_quantized_linear = orig
        return super().__call__(x)


class MppBucketedLinearGroup(_shared_mpp.MppBucketedLinearGroup):
    """Delegate `MppBucketedLinearGroup` to `fast_shared.mpp_linear` while honoring local monkeypatches."""

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        if quantize_activations_i8_k64 is not _ORIG_QUANT_FN or mpp_quantized_linear is not _ORIG_MPP_LINEAR_FN:
            orig_q = _shared_mpp.quantize_activations_i8_k64
            orig_m = _shared_mpp.mpp_quantized_linear
            _shared_mpp.quantize_activations_i8_k64 = quantize_activations_i8_k64
            _shared_mpp.mpp_quantized_linear = mpp_quantized_linear
            try:
                return super().__call__(x)
            finally:
                _shared_mpp.quantize_activations_i8_k64 = orig_q
                _shared_mpp.mpp_quantized_linear = orig_m
        return super().__call__(x)


def wrap_linear_if_mpp_eligible(
    mod: Any,
    state: _shared_mpp.MppPackState,
    *,
    is_eligible_fn: Callable[[int, int, int], bool] | None = None,
    qualifies_fn: Callable[[int, int, int, int], bool] | None = None,
) -> Any:
    """Delegate linear wrapping to `fast_shared.mpp_linear.wrap_linear_if_mpp_eligible`."""
    elig = is_eligible_fn or (lambda k, n, b: is_eligible_qwen_mpp_shape(k, n, bits=b))
    res = _shared_mpp.wrap_linear_if_mpp_eligible(
        mod,
        state,
        is_eligible_fn=elig,
        qualifies_fn=qualifies_fn,
    )
    if type(res) is _shared_mpp.MppQuantizedLinear:
        res.__class__ = MppQuantizedLinear
    return res


def wrap_bucketed_group_if_mpp_eligible(
    group: Any,
    state: _shared_mpp.MppPackState,
    *,
    is_eligible_fn: Callable[[int, int, int], bool] | None = None,
    qualifies_fn: Callable[[int, int, int, int], bool] | None = None,
    min_shared_m: int = _MIN_M_W8A8_LARGE,
) -> Any:
    """Delegate bucketed-group wrapping to `fast_shared.mpp_linear.wrap_bucketed_group_if_mpp_eligible`."""
    if isinstance(group, MppBucketedLinearGroup):
        return group
    elig = is_eligible_fn or (lambda k, n, b: is_eligible_qwen_mpp_shape(k, n, bits=b))
    res = _shared_mpp.wrap_bucketed_group_if_mpp_eligible(
        group,
        state,
        is_eligible_fn=elig,
        qualifies_fn=qualifies_fn,
        min_shared_m=min_shared_m,
    )
    if type(res) is _shared_mpp.MppBucketedLinearGroup:
        for b in res.group.buckets:
            if type(b) is _shared_mpp.MppQuantizedLinear:
                b.__class__ = MppQuantizedLinear
        res.__class__ = MppBucketedLinearGroup
    return res


def install_qwen_mpp_linears(
    model: Any,
    *,
    enabled: bool | None = None,
    allow_w8a8: bool | None = None,
    apple10_nax: bool | None = None,
    max_pack_bytes: int | None = None,
    min_avail_bytes: int | None = None,
) -> QwenMppPackState:
    """Attach memory-bounded Apple10 NAX prefill MPP wrappers across decoder linears & bucketed groups."""
    eff_on, eff_w8 = resolve_qwen_mpp_mode(enabled=enabled, allow_w8a8=allow_w8a8)
    has_nax = bool(supports_apple10_nax()) if apple10_nax is None else bool(apple10_nax)
    state = QwenMppPackState(
        apple10_nax=has_nax,
        allow_w8a8=eff_w8,
        enabled=eff_on,
        max_pack_bytes=max_pack_bytes,
        min_avail_bytes=min_avail_bytes,
    )
    if not state.enabled:
        return state

    for layer in getattr(model, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            if hasattr(mlp, "gate_up"):
                mlp.gate_up = wrap_bucketed_group_if_mpp_eligible(mlp.gate_up, state)
            if hasattr(mlp, "down_proj"):
                mlp.down_proj = wrap_linear_if_mpp_eligible(mlp.down_proj, state)
            sh_exp = getattr(mlp, "shared_expert", None)
            if sh_exp is not None:
                if hasattr(sh_exp, "gate_up"):
                    sh_exp.gate_up = wrap_bucketed_group_if_mpp_eligible(sh_exp.gate_up, state)
                if hasattr(sh_exp, "down_proj"):
                    sh_exp.down_proj = wrap_linear_if_mpp_eligible(sh_exp.down_proj, state)
        lin_attn = getattr(layer, "linear_attn", None)
        if lin_attn is not None:
            if hasattr(lin_attn, "in_group"):
                lin_attn.in_group = wrap_bucketed_group_if_mpp_eligible(lin_attn.in_group, state)
            if hasattr(lin_attn, "out_proj"):
                lin_attn.out_proj = wrap_linear_if_mpp_eligible(lin_attn.out_proj, state)
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            if hasattr(self_attn, "qkvg_group"):
                self_attn.qkvg_group = wrap_bucketed_group_if_mpp_eligible(self_attn.qkvg_group, state)
            if hasattr(self_attn, "o_proj"):
                self_attn.o_proj = wrap_linear_if_mpp_eligible(self_attn.o_proj, state)
    return state


__all__ = [
    "MppBucketedLinearGroup",
    "MppQuantizedLinear",
    "QwenMppPackState",
    "install_qwen_mpp_linears",
    "is_eligible_qwen_mpp_shape",
    "resolve_qwen_mpp_mode",
    "should_use_qwen_mpp_linear",
]
