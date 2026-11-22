"""Shared quantized KV cache (`q8` / `q4_quarot`) and online `FWHT-64` `QuaRot` rotations for MLX LLMs.

Key empirical invariants on Apple Silicon (`MLX 0.32.4+`):
  1. Fused `mx.quantized_matmul` directly on packed `(k_w, k_s, k_b)` and `(v_w, v_s, v_b)`
     runs `4.72x` faster than Python-dequant `INT8` SDPA at `B=4, T=16,384` (`2.53 ms` vs `11.94 ms`)
     and `4.42x` faster for `4-bit g64` (`2.70 ms`), with `< 1e-3` strided-slice max diff.
  2. Raw `4-bit` KV (`88.9%` passkey) and `K-only QuaRot` (`55.6%` passkey) drop needles at `>=4K`,
     whereas online `(Q, K, V)` block-64 Walsh-Hadamard `QuaRot` (`kv_q4_quarot`, `3.56x-3.8x` smaller)
     recovers `100%` (`27/27`) multi-depth passkeys and `100%` exact greedy matches.
  3. On gated-attention layers (`Qwen3.5` `FullAttentionLayer` with `q_norm`/`k_norm` + `sigmoid(gate)`),
     Hadamard rotation does NOT commute with `RMSNorm` (`rel_frob 0.12-0.28`) or `sigmoid(gate)`:
     apply `FWHT-64` to `(Q, K, V)` *post-RoPE*, and invert `FWHT-64` on `attn_out` *immediately before*
     the elementwise `sigmoid(gate)` multiply.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
import math
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .runtime_lessons import MASK_NEG_VAL_FP16

KVMode = Literal["bf16", "q8", "q4_quarot"]
VALID_KV_MODES: frozenset[str] = frozenset({"bf16", "q8", "q4_quarot"})


def normalize_kv_mode(kv_i8_or_mode: bool | str) -> KVMode:
    """Normalize legacy `bool` or string KV spec (`False/'bf16'`, `True/'q8'`, `'q4_quarot'`) to `KVMode`."""
    if isinstance(kv_i8_or_mode, bool):
        return "q8" if kv_i8_or_mode else "bf16"
    s = str(kv_i8_or_mode).strip().lower()
    if s in ("false", "0", "off", "none", "bf16", "fp16"):
        return "bf16"
    if s in ("true", "1", "on", "i8", "int8", "q8", "kv_i8", "kv_q8"):
        return "q8"
    if s in ("q4", "q4_quarot", "quarot", "kv_q4", "kv_q4_quarot"):
        return "q4_quarot"
    raise ValueError(
        f"Unsupported KV mode {kv_i8_or_mode!r}; expected False/'bf16', True/'q8', or 'q4_quarot'."
    )


def build_sylvester_hadamard(n: int = 64) -> np.ndarray:
    """Build normalized `n x n` Sylvester Walsh-Hadamard matrix (`H @ H.T = I_n`, `H = H.T`)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard block size must be a power of 2, got {n}")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(float(n))


def build_rademacher_signs(dim: int, *, seed: int = 20261122) -> np.ndarray:
    """Deterministic Rademacher `{-1, +1}` vector for randomized `QuaRot` block-64 Hadamard."""
    if dim <= 0 or dim % 64 != 0:
        raise ValueError(f"Hadamard head_dim must be a positive multiple of 64, got {dim}")
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(int(dim),))


_H64_NP: np.ndarray = build_sylvester_hadamard(64)

_METAL_FWHT64_FWD_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint vec_idx = thread_position_in_grid.y;
    uint n_vecs = params[0];
    uint n_blocks = params[1];
    if (vec_idx >= n_vecs) return;
    uint blk = vec_idx % n_blocks;
    uint base = vec_idx * 64u;
    float s0 = float(signs[blk * 64u + lane]);
    float s1 = float(signs[blk * 64u + lane + 32u]);
    float x0 = float(inp[base + lane]) * s0;
    float x1 = float(inp[base + lane + 32u]) * s1;
    float t0 = x0 + x1;
    float t1 = x0 - x1;
    x0 = t0;
    x1 = t1;
    for (ushort st = 16; st >= 1; st >>= 1) {
        float o0 = simd_shuffle_xor(x0, st);
        float o1 = simd_shuffle_xor(x1, st);
        bool up = (lane & st) != 0;
        x0 = up ? (o0 - x0) : (x0 + o0);
        x1 = up ? (o1 - x1) : (x1 + o1);
    }
    out[base + lane] = T(x0 * 0.125f);
    out[base + lane + 32u] = T(x1 * 0.125f);
"""

_METAL_FWHT64_INV_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint vec_idx = thread_position_in_grid.y;
    uint n_vecs = params[0];
    uint n_blocks = params[1];
    if (vec_idx >= n_vecs) return;
    uint blk = vec_idx % n_blocks;
    uint base = vec_idx * 64u;
    float x0 = float(inp[base + lane]);
    float x1 = float(inp[base + lane + 32u]);
    float t0 = x0 + x1;
    float t1 = x0 - x1;
    x0 = t0;
    x1 = t1;
    for (ushort st = 16; st >= 1; st >>= 1) {
        float o0 = simd_shuffle_xor(x0, st);
        float o1 = simd_shuffle_xor(x1, st);
        bool up = (lane & st) != 0;
        x0 = up ? (o0 - x0) : (x0 + o0);
        x1 = up ? (o1 - x1) : (x1 + o1);
    }
    float s0 = float(signs[blk * 64u + lane]);
    float s1 = float(signs[blk * 64u + lane + 32u]);
    out[base + lane] = T(x0 * 0.125f * s0);
    out[base + lane + 32u] = T(x1 * 0.125f * s1);
"""

_METAL_KERNELS: tuple[Any, Any] | None = None


def _get_metal_fwht64_kernels() -> tuple[Any, Any]:
    global _METAL_KERNELS
    if _METAL_KERNELS is None:
        fwd = mx.fast.metal_kernel(
            name="fast_shared_fwht64_simd_fwd",
            input_names=["inp", "signs", "params"],
            output_names=["out"],
            source=_METAL_FWHT64_FWD_SRC,
        )
        inv = mx.fast.metal_kernel(
            name="fast_shared_fwht64_simd_inv",
            input_names=["inp", "signs", "params"],
            output_names=["out"],
            source=_METAL_FWHT64_INV_SRC,
        )
        _METAL_KERNELS = (fwd, inv)
    return _METAL_KERNELS


@partial(mx.compile, shapeless=True)
def _compiled_fwht64_fwd_3d(xr: mx.array, h64: mx.array, signs_3d: mx.array) -> mx.array:
    return mx.matmul(xr * signs_3d, h64)


@partial(mx.compile, shapeless=True)
def _compiled_fwht64_inv_3d(yr: mx.array, h64: mx.array, signs_3d: mx.array) -> mx.array:
    return mx.matmul(yr, h64) * signs_3d


class HadamardTransformBank:
    """Reusable block-64 Walsh-Hadamard + Rademacher bank for any `head_dim % 64 == 0` (`64..512`)."""

    def __init__(
        self,
        head_dim: int = 256,
        *,
        use_random_signs: bool = True,
        seed: int = 20261122,
        use_metal: bool = True,
    ) -> None:
        if head_dim <= 0 or head_dim % 64 != 0:
            raise ValueError(f"head_dim must be a positive multiple of 64, got {head_dim}")
        self.head_dim: int = int(head_dim)
        self.n_blocks: int = self.head_dim // 64
        self.use_metal: bool = bool(use_metal)
        signs_np = (
            build_rademacher_signs(self.head_dim, seed=seed)
            if use_random_signs
            else np.ones((self.head_dim,), dtype=np.float32)
        )
        self.signs_np: np.ndarray = signs_np
        self._h64_by_dtype: dict[mx.Dtype, mx.array] = {
            mx.bfloat16: mx.array(_H64_NP, dtype=mx.bfloat16),
            mx.float16: mx.array(_H64_NP, dtype=mx.float16),
            mx.float32: mx.array(_H64_NP, dtype=mx.float32),
        }
        self._signs_flat_by_dtype: dict[mx.Dtype, mx.array] = {
            mx.bfloat16: mx.array(signs_np, dtype=mx.bfloat16),
            mx.float16: mx.array(signs_np, dtype=mx.float16),
            mx.float32: mx.array(signs_np, dtype=mx.float32),
        }
        s3d = signs_np.reshape(1, self.n_blocks, 64)
        self._signs_3d_by_dtype: dict[mx.Dtype, mx.array] = {
            mx.bfloat16: mx.array(s3d, dtype=mx.bfloat16),
            mx.float16: mx.array(s3d, dtype=mx.float16),
            mx.float32: mx.array(s3d, dtype=mx.float32),
        }

    def transform_compile(self, x: mx.array, *, inverse: bool = False) -> mx.array:
        """Portable `@mx.compile(shapeless=True)` block-64 Hadamard transform (CPU & GPU)."""
        if int(x.shape[-1]) != self.head_dim:
            raise ValueError(f"Expected last dim {self.head_dim}, got {x.shape[-1]}")
        orig_shape = x.shape
        dt = x.dtype if x.dtype in self._h64_by_dtype else mx.float32
        x_cast = x if x.dtype == dt else x.astype(dt)
        xr = x_cast.reshape(-1, self.n_blocks, 64)
        h64 = self._h64_by_dtype[dt]
        s3d = self._signs_3d_by_dtype[dt]
        out = _compiled_fwht64_inv_3d(xr, h64, s3d) if inverse else _compiled_fwht64_fwd_3d(xr, h64, s3d)
        res = out.reshape(orig_shape)
        return res if res.dtype == x.dtype else res.astype(x.dtype)

    def transform_metal(self, x: mx.array, *, inverse: bool = False) -> mx.array:
        """In-register `simd_shuffle_xor` Metal `FWHT-64` (`0` shared memory, `1D` params buffer)."""
        if int(x.shape[-1]) != self.head_dim:
            raise ValueError(f"Expected last dim {self.head_dim}, got {x.shape[-1]}")
        if x.size == 0:
            return x
        xc = mx.contiguous(x)
        dt = xc.dtype if xc.dtype in (mx.bfloat16, mx.float16, mx.float32) else mx.bfloat16
        if xc.dtype != dt:
            xc = xc.astype(dt)
        n_vecs = xc.size // 64
        tg_y = 8
        grid_y = ((n_vecs + tg_y - 1) // tg_y) * tg_y
        fwd_k, inv_k = _get_metal_fwht64_kernels()
        kern = inv_k if inverse else fwd_k
        params = mx.array([n_vecs, self.n_blocks], dtype=mx.uint32)
        signs_t = self._signs_flat_by_dtype[dt]
        out = kern(
            inputs=[xc, signs_t, params],
            template=[("T", dt)],
            grid=(32, grid_y, 1),
            threadgroup=(32, tg_y, 1),
            output_shapes=[xc.shape],
            output_dtypes=[dt],
        )[0]
        return out if out.dtype == x.dtype else out.astype(x.dtype)

    def transform(self, x: mx.array, *, inverse: bool = False) -> mx.array:
        """Dispatch `transform_metal` on `mx.gpu` (when enabled) and `transform_compile` on `mx.cpu`."""
        if (
            self.use_metal
            and mx.metal.is_available()
            and mx.default_device().type == mx.DeviceType.gpu
            and x.dtype in (mx.bfloat16, mx.float16)
        ):
            return self.transform_metal(x, inverse=inverse)
        return self.transform_compile(x, inverse=inverse)


_HADAMARD_CACHE: dict[tuple[int, int, bool], HadamardTransformBank] = {}


def get_hadamard_bank(
    head_dim: int,
    *,
    seed: int = 20261122,
    use_metal: bool = True,
) -> HadamardTransformBank:
    """Return process-wide cached `HadamardTransformBank` for `(head_dim, seed, use_metal)`."""
    key = (int(head_dim), int(seed), bool(use_metal))
    bank = _HADAMARD_CACHE.get(key)
    if bank is None:
        bank = HadamardTransformBank(head_dim, seed=seed, use_metal=use_metal)
        _HADAMARD_CACHE[key] = bank
    return bank


def fused_quantized_sdpa(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    *,
    scale: float,
    mask: str | mx.array | None = None,
    k_bits: int = 8,
    k_group_size: int = 64,
    v_bits: int = 8,
    v_group_size: int = 64,
) -> mx.array:
    """Zero-dequantized-KV SDPA executing `mx.quantized_matmul` directly on packed `(K_q, V_q)`."""
    b, n_q, l_q, d = queries.shape
    n_kv = int(q_keys[0].shape[1])
    n_rep = n_q // n_kv

    q_scaled = queries * mx.array(scale, dtype=queries.dtype)
    if n_rep > 1:
        q_scaled = q_scaled.reshape(b, n_kv, n_rep, l_q, d)
        kq = (
            mx.expand_dims(q_keys[0], axis=2),
            mx.expand_dims(q_keys[1], axis=2),
            mx.expand_dims(q_keys[2], axis=2),
        )
        vq = (
            mx.expand_dims(q_values[0], axis=2),
            mx.expand_dims(q_values[1], axis=2),
            mx.expand_dims(q_values[2], axis=2),
        )
    else:
        kq = q_keys
        vq = q_values

    scores = mx.quantized_matmul(
        q_scaled,
        kq[0],
        kq[1],
        kq[2],
        transpose=True,
        group_size=k_group_size,
        bits=k_bits,
    )

    if mask is not None:
        neg_inf = mx.array(MASK_NEG_VAL_FP16, dtype=scores.dtype)
        if isinstance(mask, str):
            if mask == "causal" and l_q > 1:
                k_len = int(scores.shape[-1])
                q_idx = mx.arange(k_len - l_q, k_len, dtype=mx.int32)[:, None]
                k_idx = mx.arange(k_len, dtype=mx.int32)[None, :]
                scores = mx.where(k_idx <= q_idx, scores, neg_inf)
        else:
            m = mx.expand_dims(mask, axis=2) if (n_rep > 1 and mask.ndim == 4) else mask
            if m.dtype == mx.bool_:
                scores = mx.where(m, scores, neg_inf)
            else:
                scores = scores + m.astype(scores.dtype)

    probs = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        probs,
        vq[0],
        vq[1],
        vq[2],
        transpose=False,
        group_size=v_group_size,
        bits=v_bits,
    )
    if n_rep > 1:
        out = out.reshape(b, n_q, l_q, d)
    return out


class QuantizedKVCache:
    """Chunk-allocated `O(1)` affine quantized KV cache (`bits=8` or `4`, `group_size=64`) with fused-`qmm` SDPA."""

    def __init__(
        self,
        step: int = 256,
        *,
        bits: int = 8,
        group_size: int = 64,
        use_quarot: bool = False,
        max_fused_q_len: int = 8,
    ) -> None:
        if bits not in (4, 8):
            raise ValueError(f"QuantizedKVCache bits must be 4 or 8, got {bits}")
        if group_size not in (32, 64, 128):
            raise ValueError(f"QuantizedKVCache group_size must be 32, 64, or 128, got {group_size}")
        self.step: int = int(step)
        self.bits: int = int(bits)
        self.group_size: int = int(group_size)
        self.use_quarot: bool = bool(use_quarot)
        self.max_fused_q_len: int = int(max_fused_q_len)

        self.k_w: mx.array | None = None
        self.k_s: mx.array | None = None
        self.k_b: mx.array | None = None
        self.v_w: mx.array | None = None
        self.v_s: mx.array | None = None
        self.v_b: mx.array | None = None
        self.head_dim: int = 0
        self.offset: int = 0

    @property
    def kv_mode(self) -> KVMode:
        return "q4_quarot" if (self.bits == 4 and self.use_quarot) else "q8"

    def _ensure_capacity(self, b: int, h_kv: int, needed: int, d: int, dtype: mx.Dtype) -> int:
        if d % self.group_size != 0:
            raise ValueError(f"head_dim={d} is not divisible by group_size={self.group_size}")
        self.head_dim = int(d)
        prev = self.offset
        cur_cap = 0 if self.k_w is None else int(self.k_w.shape[2])
        if self.k_w is None or needed > cur_cap or int(self.k_w.shape[0]) != b:
            alloc = ((needed + self.step - 1) // self.step) * self.step
            pack_d = (d * self.bits) // 32
            n_grp = d // self.group_size
            new_kw = mx.zeros((b, h_kv, alloc, pack_d), dtype=mx.uint32)
            new_ks = mx.zeros((b, h_kv, alloc, n_grp), dtype=dtype)
            new_kb = mx.zeros((b, h_kv, alloc, n_grp), dtype=dtype)
            new_vw = mx.zeros((b, h_kv, alloc, pack_d), dtype=mx.uint32)
            new_vs = mx.zeros((b, h_kv, alloc, n_grp), dtype=dtype)
            new_vb = mx.zeros((b, h_kv, alloc, n_grp), dtype=dtype)
            if self.k_w is not None and prev > 0 and int(self.k_w.shape[0]) == b:
                new_kw[:, :, :prev, :] = self.k_w[:, :, :prev, :]
                new_ks[:, :, :prev, :] = self.k_s[:, :, :prev, :]  # type: ignore[index]
                new_kb[:, :, :prev, :] = self.k_b[:, :, :prev, :]  # type: ignore[index]
                new_vw[:, :, :prev, :] = self.v_w[:, :, :prev, :]  # type: ignore[index]
                new_vs[:, :, :prev, :] = self.v_s[:, :, :prev, :]  # type: ignore[index]
                new_vb[:, :, :prev, :] = self.v_b[:, :, :prev, :]  # type: ignore[index]
            self.k_w, self.k_s, self.k_b = new_kw, new_ks, new_kb
            self.v_w, self.v_s, self.v_b = new_vw, new_vs, new_vb
        return prev

    def write(self, k: mx.array, v: mx.array) -> int:
        b, h_kv, s, d = k.shape
        needed = self.offset + s
        prev = self._ensure_capacity(b, h_kv, needed, d, k.dtype)
        assert self.k_w is not None and self.k_s is not None and self.k_b is not None
        assert self.v_w is not None and self.v_s is not None and self.v_b is not None
        kw, ks, kb = mx.quantize(k, group_size=self.group_size, bits=self.bits)
        vw, vs, vb = mx.quantize(v, group_size=self.group_size, bits=self.bits)
        self.k_w[:, :, prev:needed, :] = kw
        self.k_s[:, :, prev:needed, :] = ks
        self.k_b[:, :, prev:needed, :] = kb
        self.v_w[:, :, prev:needed, :] = vw
        self.v_s[:, :, prev:needed, :] = vs
        self.v_b[:, :, prev:needed, :] = vb
        self.offset = needed
        return needed

    def append_cached_quantized(
        self,
        kw: mx.array,
        ks: mx.array,
        kb: mx.array,
        vw: mx.array,
        vs: mx.array,
        vb: mx.array,
    ) -> None:
        """Append pre-quantized `(kw, ks, kb, vw, vs, vb)` slices directly (`0` double-quantization)."""
        b, h_kv, s, pack_d = kw.shape
        d = (pack_d * 32) // self.bits
        needed = self.offset + s
        prev = self._ensure_capacity(b, h_kv, needed, d, ks.dtype)
        assert self.k_w is not None and self.k_s is not None and self.k_b is not None
        assert self.v_w is not None and self.v_s is not None and self.v_b is not None
        self.k_w[:, :, prev:needed, :] = kw
        self.k_s[:, :, prev:needed, :] = ks
        self.k_b[:, :, prev:needed, :] = kb
        self.v_w[:, :, prev:needed, :] = vw
        self.v_s[:, :, prev:needed, :] = vs
        self.v_b[:, :, prev:needed, :] = vb
        self.offset = needed

    def slice_quantized(
        self, start: int, end: int
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Extract contiguous packed slice `[start:end]` for `PrefixTree` / `HybridPrefixSnapshot`."""
        assert self.k_w is not None and self.k_s is not None and self.k_b is not None
        assert self.v_w is not None and self.v_s is not None and self.v_b is not None
        return (
            mx.contiguous(self.k_w[:, :, start:end, :]),
            mx.contiguous(self.k_s[:, :, start:end, :]),
            mx.contiguous(self.k_b[:, :, start:end, :]),
            mx.contiguous(self.v_w[:, :, start:end, :]),
            mx.contiguous(self.v_s[:, :, start:end, :]),
            mx.contiguous(self.v_b[:, :, start:end, :]),
        )

    def fetch_dequantized(self, out_dtype: mx.Dtype = mx.bfloat16) -> tuple[mx.array, mx.array]:
        needed = self.offset
        assert self.k_w is not None and self.k_s is not None and self.k_b is not None
        assert self.v_w is not None and self.v_s is not None and self.v_b is not None
        k_out = mx.dequantize(
            self.k_w[:, :, :needed, :],
            self.k_s[:, :, :needed, :],
            self.k_b[:, :, :needed, :],
            group_size=self.group_size,
            bits=self.bits,
            dtype=out_dtype,
        )
        v_out = mx.dequantize(
            self.v_w[:, :, :needed, :],
            self.v_s[:, :, :needed, :],
            self.v_b[:, :, :needed, :],
            group_size=self.group_size,
            bits=self.bits,
            dtype=out_dtype,
        )
        return k_out, v_out

    def attend(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        *,
        scale: float,
        mask: str | mx.array | None = "causal",
        use_apple10_force_fused: bool = False,
    ) -> mx.array:
        """Write `(k_new, v_new)` and execute fused-`qmm` SDPA for `s_len <= max_fused_q_len` else Flash SDPA."""
        s_len = int(q.shape[2])
        needed = self.write(k_new, v_new)
        eff_mask: str | mx.array | None = None if (s_len == 1 and mask == "causal") else mask

        if s_len <= self.max_fused_q_len:
            assert self.k_w is not None and self.k_s is not None and self.k_b is not None
            assert self.v_w is not None and self.v_s is not None and self.v_b is not None
            return fused_quantized_sdpa(
                q,
                (self.k_w[:, :, :needed, :], self.k_s[:, :, :needed, :], self.k_b[:, :, :needed, :]),
                (self.v_w[:, :, :needed, :], self.v_s[:, :, :needed, :], self.v_b[:, :, :needed, :]),
                scale=scale,
                mask=eff_mask,
                k_bits=self.bits,
                k_group_size=self.group_size,
                v_bits=self.bits,
                v_group_size=self.group_size,
            )

        k_dq, v_dq = self.fetch_dequantized(out_dtype=q.dtype)
        if use_apple10_force_fused and int(q.shape[-1]) == 256:
            return mx.fast.scaled_dot_product_attention(
                q, k_dq, v_dq, scale=scale, mask=eff_mask, force_fused=True
            )
        return mx.fast.scaled_dot_product_attention(q, k_dq, v_dq, scale=scale, mask=eff_mask)

    def active_nbytes(self) -> int:
        if self.k_w is None or self.offset <= 0:
            return 0
        off = self.offset
        return int(
            self.k_w[:, :, :off, :].nbytes
            + self.k_s[:, :, :off, :].nbytes  # type: ignore[index]
            + self.k_b[:, :, :off, :].nbytes  # type: ignore[index]
            + self.v_w[:, :, :off, :].nbytes  # type: ignore[index]
            + self.v_s[:, :, :off, :].nbytes  # type: ignore[index]
            + self.v_b[:, :, :off, :].nbytes  # type: ignore[index]
        )

    def allocated_nbytes(self) -> int:
        if self.k_w is None:
            return 0
        return int(
            self.k_w.nbytes
            + self.k_s.nbytes  # type: ignore[union-attr]
            + self.k_b.nbytes  # type: ignore[union-attr]
            + self.v_w.nbytes  # type: ignore[union-attr]
            + self.v_s.nbytes  # type: ignore[union-attr]
            + self.v_b.nbytes  # type: ignore[union-attr]
        )

    def rewind_to(self, new_offset: int) -> None:
        if new_offset < 0 or new_offset > self.offset:
            raise ValueError(f"Invalid rewind offset {new_offset} for current offset {self.offset}")
        self.offset = int(new_offset)


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
    """Arbitrary-order per-layer spec-bucketing over named projections sharing input `x` (`0.0` ULP drift)."""

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
