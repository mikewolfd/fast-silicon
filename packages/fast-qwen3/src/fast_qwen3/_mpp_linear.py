"""Apple10+ NAX `W4A8`/`W8A8` prefill acceleration and memory-guarded weight packing for `FastQwen3Model`."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from fast_shared.hw_check import supports_apple10_nax
from fast_shared.kv_quant import SpecBucketedLinearGroup
import fast_shared.metal_mpp_int8 as _metal_mpp_int8
from fast_shared.metal_mpp_int8 import (
    mpp_quantized_linear,
    pack_affine_weights_for_mpp_i8,
    quantize_activations_i8_k64,
)
from fast_shared.runtime_lessons import sample_host_memory

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
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


def _parse_env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    return None


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
        parsed = _parse_env_bool(env_name)
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
    b = int(bits)
    if b == 4:
        return _w4a8_min_m(int(k), int(n)) is not None
    if b == 8:
        return _w8a8_min_m(int(k), int(n)) is not None
    return False


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
    m_i, k_i, n_i = int(m), int(k), int(n)
    if b == 4:
        min_m = _w4a8_min_m(k_i, n_i)
        return min_m is not None and m_i >= min_m
    if b == 8:
        if not w8_on:
            return False
        min_m = _w8a8_min_m(k_i, n_i)
        return min_m is not None and m_i >= min_m
    return False


def _default_pack_budgets() -> tuple[int, int]:
    try:
        host_ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        host_ram_bytes = 24 * (1 << 30)
    if 0 < host_ram_bytes <= 32 * (1 << 30):
        return int(0.28 * host_ram_bytes), int(0.15 * host_ram_bytes)
    return int(0.35 * host_ram_bytes), int(0.12 * host_ram_bytes)


class QwenMppPackState:
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
        def_max, def_min = _default_pack_budgets()
        self.enabled = bool(enabled and apple10_nax)
        self.apple10_nax = bool(apple10_nax)
        self.allow_w8a8 = bool(allow_w8a8)
        self.max_pack_bytes = def_max if max_pack_bytes is None else max(0, int(max_pack_bytes))
        self.min_avail_bytes = def_min if min_avail_bytes is None else max(0, int(min_avail_bytes))
        self.packed_weights: dict[int, tuple[mx.array, mx.array, int]] = {}
        self.packed_bytes: int = 0
        self.active_shared_xq: tuple[int, int, tuple[mx.array, mx.array]] | None = None
        self.w4_calls: int = 0
        self.w8_calls: int = 0
        self.shared_quant_calls: int = 0
        self.budget_skips: int = 0

    @staticmethod
    def estimate_pack_bytes(k: int, n: int, *, bits: int = 8) -> int:
        if bool(getattr(_metal_mpp_int8, "PACKS_META_ONLY", False)):
            return 4 * int(n) * (int(k) // 64)
        return (int(k) * int(n) * int(bits)) // 8 + 2 * int(n) * (int(k) // 64)

    def can_pack_estimate(self, k: int, n: int, *, bits: int = 8) -> bool:
        return (self.packed_bytes + self.estimate_pack_bytes(k, n, bits=bits)) <= self.max_pack_bytes

    def get_or_pack(
        self,
        lin: nn.QuantizedLinear,
        *,
        bits: int,
        k: int,
        n: int,
    ) -> tuple[mx.array, mx.array] | None:
        mid = id(lin)
        cached = self.packed_weights.get(mid)
        if cached is not None:
            return cached[0], cached[1]
        est_bytes = self.estimate_pack_bytes(k, n, bits=bits)
        if self.packed_bytes + est_bytes > self.max_pack_bytes:
            self.budget_skips += 1
            return None
        try:
            snap = sample_host_memory()
            avail = int(getattr(snap, "available_bytes", 0) or 0)
            if avail > 0 and (avail - est_bytes) < self.min_avail_bytes:
                self.budget_skips += 1
                return None
        except Exception:
            pass
        wp, wm = pack_affine_weights_for_mpp_i8(
            mx.stop_gradient(lin.weight),
            mx.stop_gradient(lin.scales),
            mx.stop_gradient(lin.biases),
            bits=int(bits),
            group_size=64,
        )
        mx.eval(wp, wm)  # guard: allow-eval
        actual_bytes = int(wp.nbytes) + int(wm.nbytes)
        self.packed_weights[mid] = (wp, wm, actual_bytes)
        self.packed_bytes += actual_bytes
        return wp, wm

    def clear(self) -> int:
        released = len(self.packed_weights)
        self.packed_weights.clear()
        self.packed_bytes = 0
        self.active_shared_xq = None
        return released


class MppQuantizedLinear(nn.Module):
    """Wrap `nn.QuantizedLinear` to accelerate large Apple10 prefill GEMMs while preserving decode and autograd VJP."""

    def __init__(
        self,
        linear: nn.QuantizedLinear,
        *,
        bits: int,
        k: int,
        n: int,
        state: QwenMppPackState,
    ) -> None:
        super().__init__()
        self.linear = linear
        self._mpp_bits = int(bits)
        self._mpp_k = int(k)
        self._mpp_n = int(n)
        self._mpp_state = state

        @mx.custom_function
        def _mpp_op(x_in: mx.array) -> mx.array:
            lin = self.linear
            packed_ent = self._mpp_state.packed_weights.get(id(lin))
            packed = (packed_ent[0], packed_ent[1]) if packed_ent is not None else None
            m_i = x_in.size // self._mpp_k
            shared_slot = self._mpp_state.active_shared_xq
            pre_xq = (
                shared_slot[2]
                if (shared_slot is not None and shared_slot[0] == m_i and shared_slot[1] == self._mpp_k)
                else None
            )
            bias_arr = lin.bias if "bias" in lin else None
            out = mpp_quantized_linear(
                x_in,
                mx.stop_gradient(lin.weight),
                mx.stop_gradient(lin.scales),
                mx.stop_gradient(lin.biases),
                bits=self._mpp_bits,
                group_size=64,
                mode="affine",
                bias=mx.stop_gradient(bias_arr) if bias_arr is not None else None,
                out_dtype=x_in.dtype,
                packed_weights=packed,
                prequantized_x=pre_xq,
            )
            return out

        @_mpp_op.vjp
        def _mpp_vjp(_primals: Any, cotangent: mx.array, _outputs: Any) -> tuple[mx.array]:
            lin = self.linear
            dx = mx.quantized_matmul(
                cotangent,
                mx.stop_gradient(lin.weight),
                scales=mx.stop_gradient(lin.scales),
                biases=mx.stop_gradient(lin.biases),
                transpose=False,
                group_size=64,
                bits=self._mpp_bits,
                mode="affine",
            )
            return (dx,)

        self._mpp_op = _mpp_op

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @property
    def scales(self) -> mx.array:
        return self.linear.scales

    @property
    def biases(self) -> Any:
        return getattr(self.linear, "biases", None)

    @property
    def bias(self) -> Any:
        return self.linear.bias if "bias" in self.linear else None

    @property
    def bits(self) -> int:
        return int(self.linear.bits)

    @property
    def group_size(self) -> int:
        return int(self.linear.group_size)

    @property
    def mode(self) -> str:
        return str(self.linear.mode)

    def _qualifies_for_m(self, m: int) -> bool:
        st = self._mpp_state
        return should_use_qwen_mpp_linear(
            m,
            self._mpp_k,
            self._mpp_n,
            bits=self._mpp_bits,
            group_size=64,
            mode="affine",
            apple10_nax=st.apple10_nax,
            allow_w8a8=st.allow_w8a8,
            enabled=st.enabled,
        )

    def __call__(self, x: mx.array) -> mx.array:
        st = self._mpp_state
        if not st.enabled:
            return self.linear(x)
        m = x.size // self._mpp_k if (x.size > 0 and self._mpp_k > 0) else 0
        if not self._qualifies_for_m(m):
            return self.linear(x)
        packed = st.get_or_pack(self.linear, bits=self._mpp_bits, k=self._mpp_k, n=self._mpp_n)
        if packed is None:
            return self.linear(x)
        if self._mpp_bits == 4:
            st.w4_calls += 1
        else:
            st.w8_calls += 1
        return self._mpp_op(x)


class MppBucketedLinearGroup(nn.Module):
    """Wrap multi-bucket `SpecBucketedLinearGroup` so qualifying buckets share one `i8` activation quant pass."""

    def __init__(self, group: SpecBucketedLinearGroup, state: QwenMppPackState) -> None:
        super().__init__()
        self.group = group
        self._state = state

    @property
    def buckets(self) -> list[Any]:
        return self.group.buckets

    def set_lora_slices(self, slices: Any) -> None:
        self.group.set_lora_slices(slices)

    def clear_lora(self) -> None:
        self.group.clear_lora()

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        st = self._state
        if not st.enabled or st.active_shared_xq is not None or len(self.group.buckets) < 2:
            return self.group(x)
        k = int(x.shape[-1])
        if k <= 0 or (k % 64) != 0:
            return self.group(x)
        m = x.size // k if x.size > 0 else 0
        if m < _MIN_M_W8A8_LARGE:
            return self.group(x)
        elig = 0
        for bucket in self.group.buckets:
            if isinstance(bucket, MppQuantizedLinear) and bucket._mpp_k == k and bucket._qualifies_for_m(m):
                if id(bucket.linear) in st.packed_weights or st.can_pack_estimate(
                    k, bucket._mpp_n, bits=bucket._mpp_bits
                ):
                    elig += 1
        if elig < 2:
            return self.group(x)
        st.shared_quant_calls += 1
        xq, xm = quantize_activations_i8_k64(mx.stop_gradient(x).reshape(m, k))
        st.active_shared_xq = (m, k, (xq, xm))
        try:
            return self.group(x)
        finally:
            st.active_shared_xq = None


def _wrap_linear_if_eligible(mod: Any, state: QwenMppPackState) -> Any:
    if not isinstance(mod, nn.QuantizedLinear):
        return mod
    if getattr(mod, "biases", None) is None:
        return mod
    bits = int(getattr(mod, "bits", 0))
    gs = int(getattr(mod, "group_size", 0))
    mode = str(getattr(mod, "mode", ""))
    if bits not in (4, 8) or gs != 64 or mode != "affine":
        return mod
    k = int(mod.scales.shape[-1]) * 64
    n = int(mod.weight.shape[0])
    if not is_eligible_qwen_mpp_shape(k, n, bits=bits, group_size=gs, mode=mode):
        return mod
    return MppQuantizedLinear(mod, bits=bits, k=k, n=n, state=state)


def _wrap_group_if_eligible(group: Any, state: QwenMppPackState) -> Any:
    if isinstance(group, MppBucketedLinearGroup):
        return group
    if not isinstance(group, SpecBucketedLinearGroup):
        return group
    group.buckets = [_wrap_linear_if_eligible(b, state) for b in group.buckets]
    mpp_count = sum(1 for b in group.buckets if isinstance(b, MppQuantizedLinear))
    if mpp_count >= 2:
        return MppBucketedLinearGroup(group, state)
    return group


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
                mlp.gate_up = _wrap_group_if_eligible(mlp.gate_up, state)
            if hasattr(mlp, "down_proj"):
                mlp.down_proj = _wrap_linear_if_eligible(mlp.down_proj, state)
            sh_exp = getattr(mlp, "shared_expert", None)
            if sh_exp is not None:
                if hasattr(sh_exp, "gate_up"):
                    sh_exp.gate_up = _wrap_group_if_eligible(sh_exp.gate_up, state)
                if hasattr(sh_exp, "down_proj"):
                    sh_exp.down_proj = _wrap_linear_if_eligible(sh_exp.down_proj, state)
        lin_attn = getattr(layer, "linear_attn", None)
        if lin_attn is not None:
            if hasattr(lin_attn, "in_group"):
                lin_attn.in_group = _wrap_group_if_eligible(lin_attn.in_group, state)
            if hasattr(lin_attn, "out_proj"):
                lin_attn.out_proj = _wrap_linear_if_eligible(lin_attn.out_proj, state)
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            if hasattr(self_attn, "qkvg_group"):
                self_attn.qkvg_group = _wrap_group_if_eligible(self_attn.qkvg_group, state)
            if hasattr(self_attn, "o_proj"):
                self_attn.o_proj = _wrap_linear_if_eligible(self_attn.o_proj, state)
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
