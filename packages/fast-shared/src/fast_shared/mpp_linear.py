"""Generic Apple10+ NAX `W4A8`/`W8A8` affine linear wrapper, pack-state cache, and bucketed group."""

from __future__ import annotations

import os
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from .hw_check import supports_apple10_nax
from .kv_quant import SpecBucketedLinearGroup
from . import metal_mpp_int8 as _metal_mpp_int8
from .metal_mpp_int8 import (
    mpp_quantized_linear,
    pack_affine_weights_for_mpp_i8,
    quantize_activations_i8_k64,
)
from .runtime_lessons import sample_host_memory

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})


def resolve_mpp_env_bool(name: str) -> bool | None:
    """Parse a boolean environment override (`None` when unset or unrecognized)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    return None


def _default_pack_budgets() -> tuple[int, int]:
    try:
        host_ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        host_ram_bytes = 24 * (1 << 30)
    if 0 < host_ram_bytes <= 32 * (1 << 30):
        return int(0.28 * host_ram_bytes), int(0.15 * host_ram_bytes)
    return int(0.35 * host_ram_bytes), int(0.12 * host_ram_bytes)


class MppPackState:
    """Shared lazy weight-pack cache, budget guard, and shared group activation-quantization slot."""

    def __init__(
        self,
        *,
        apple10_nax: bool,
        allow_w8a8: bool = False,
        enabled: bool = True,
        min_m_fn: Callable[[int, int, int], int | None] | None = None,
        max_pack_bytes: int | None = None,
        min_avail_bytes: int | None = None,
    ) -> None:
        def_max, def_min = _default_pack_budgets()
        self.enabled = bool(enabled and apple10_nax)
        self.apple10_nax = bool(apple10_nax)
        self.allow_w8a8 = bool(allow_w8a8)
        self.min_m_fn = min_m_fn
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

    def min_m_for(self, k: int, n: int, bits: int) -> int | None:
        b = int(bits)
        if b == 8 and not self.allow_w8a8:
            return None
        if self.min_m_fn is not None:
            return self.min_m_fn(int(k), int(n), b)
        if (int(k) % 64) != 0 or (int(n) % 64) != 0:
            return None
        return 128 if b == 4 else (64 if b == 8 else None)

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
    """Wrap `nn.QuantizedLinear` to accelerate large Apple10 NAX GEMMs while preserving small-M & VJP."""

    def __init__(
        self,
        linear: nn.QuantizedLinear,
        *,
        bits: int,
        k: int,
        n: int,
        state: MppPackState,
        qualifies_fn: Callable[[int], bool] | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear
        self._mpp_bits = int(bits)
        self._mpp_k = int(k)
        self._mpp_n = int(n)
        self._mpp_state = state
        self._qualifies_fn = qualifies_fn

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
            return mpp_quantized_linear(
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
        if self._qualifies_fn is not None:
            return bool(self._qualifies_fn(int(m)))
        st = self._mpp_state
        if not st.enabled:
            return False
        min_m = st.min_m_for(self._mpp_k, self._mpp_n, self._mpp_bits)
        return min_m is not None and int(m) >= int(min_m)

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
    """Wrap `SpecBucketedLinearGroup` so qualifying buckets share one `i8` activation quant pass."""

    def __init__(self, group: SpecBucketedLinearGroup, state: MppPackState, *, min_shared_m: int = 64) -> None:
        super().__init__()
        self.group = group
        self._state = state
        self._min_shared_m = max(1, int(min_shared_m))

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
        if m < self._min_shared_m:
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


def wrap_linear_if_mpp_eligible(
    mod: Any,
    state: MppPackState,
    *,
    is_eligible_fn: Callable[[int, int, int], bool] | None = None,
    qualifies_fn: Callable[[int, int, int, int], bool] | None = None,
) -> Any:
    """Wrap `nn.QuantizedLinear` with `MppQuantizedLinear` when shape and quant specs qualify."""
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
    if is_eligible_fn is not None:
        if not is_eligible_fn(k, n, bits):
            return mod
    elif state.min_m_fn is not None:
        if state.min_m_fn(k, n, bits) is None:
            return mod
    elif state.min_m_for(k, n, bits) is None:
        return mod
    q_fn = (lambda m, _k=k, _n=n, _b=bits: bool(qualifies_fn(m, _k, _n, _b))) if qualifies_fn is not None else None
    return MppQuantizedLinear(mod, bits=bits, k=k, n=n, state=state, qualifies_fn=q_fn)


def wrap_bucketed_group_if_mpp_eligible(
    group: Any,
    state: MppPackState,
    *,
    is_eligible_fn: Callable[[int, int, int], bool] | None = None,
    qualifies_fn: Callable[[int, int, int, int], bool] | None = None,
    min_shared_m: int = 64,
) -> Any:
    """Wrap compatible buckets in `SpecBucketedLinearGroup` to share one `i8` activation quant pass."""
    if isinstance(group, MppBucketedLinearGroup):
        return group
    if not isinstance(group, SpecBucketedLinearGroup):
        return group
    group.buckets = [
        wrap_linear_if_mpp_eligible(
            b,
            state,
            is_eligible_fn=is_eligible_fn,
            qualifies_fn=qualifies_fn,
        )
        for b in group.buckets
    ]
    mpp_count = sum(1 for b in group.buckets if isinstance(b, MppQuantizedLinear))
    if mpp_count >= 2:
        return MppBucketedLinearGroup(group, state, min_shared_m=min_shared_m)
    return group


def wrap_quantized_linears_with_mpp(
    module: nn.Module,
    *,
    min_m_fn: Callable[[int, int, int], int | None],
    apple10_nax: bool | None = None,
    allow_w8a8: bool = True,
    enabled: bool = True,
    max_pack_bytes: int | None = None,
    min_avail_bytes: int | None = None,
    state: MppPackState | None = None,
    skip_attr_substrings: tuple[str, ...] = (),
) -> MppPackState:
    """Recursively wrap eligible `nn.QuantizedLinear` submodules on `module` with `MppQuantizedLinear`."""
    eff_nax = bool(supports_apple10_nax()) if apple10_nax is None else bool(apple10_nax)
    eff_state = state if state is not None else MppPackState(
        apple10_nax=eff_nax,
        allow_w8a8=allow_w8a8,
        enabled=enabled,
        min_m_fn=min_m_fn,
        max_pack_bytes=max_pack_bytes,
        min_avail_bytes=min_avail_bytes,
    )
    if not eff_state.enabled:
        return eff_state

    visited: set[int] = set()

    def _should_skip(name: str) -> bool:
        return any(sub in name for sub in skip_attr_substrings)

    def _visit(mod: Any, prefix: str = "") -> None:
        mid = id(mod)
        if mid in visited:
            return
        visited.add(mid)
        if isinstance(mod, (MppQuantizedLinear, MppBucketedLinearGroup)):
            return
        if isinstance(mod, list):
            for idx, item in enumerate(mod):
                if isinstance(item, nn.Module):
                    _visit(item, f"{prefix}.{idx}" if prefix else str(idx))
            return
        if not isinstance(mod, nn.Module):
            return
        for attr_name, child in list(mod.items()):
            full = f"{prefix}.{attr_name}" if prefix else str(attr_name)
            if _should_skip(full):
                continue
            if isinstance(child, nn.QuantizedLinear):
                wrapped = wrap_linear_if_mpp_eligible(child, eff_state)
                if wrapped is not child:
                    mod[attr_name] = wrapped
            elif isinstance(child, list):
                _visit(child, full)
            elif isinstance(child, nn.Module):
                _visit(child, full)

    _visit(module)
    return eff_state


__all__ = [
    "MppBucketedLinearGroup",
    "MppPackState",
    "MppQuantizedLinear",
    "resolve_mpp_env_bool",
    "wrap_bucketed_group_if_mpp_eligible",
    "wrap_linear_if_mpp_eligible",
    "wrap_quantized_linears_with_mpp",
]
