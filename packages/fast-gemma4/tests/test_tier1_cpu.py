"""Pure-CPU & static unit tests for Tier 1 `fast-gemma4` engine optimizations (zero GPU/Metal/ANE calls)."""

from __future__ import annotations

import ast
import concurrent.futures
import inspect
import json
from pathlib import Path
import py_compile
import sys
import threading
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

import fast_gemma4
import fast_gemma4.engine as fg4_engine
import fast_gemma4.kernels as fg4_kernels
import fast_gemma4.model as fg4_model
import fast_gemma4.speculative as fg4_spec
import fast_shared.metal_mpp_bf16 as fs_mpp_bf16
from fast_gemma4.engine import FastGemma4Engine
from fast_gemma4.grammar import GrammarHelper
from fast_gemma4.model import FastGemma4Model
from fast_gemma4.paths import resolve_paths
from fast_gemma4.speculative import run_decode_rolling
from fast_gemma4.tokenizer import DEFAULT_KEPT_VOCAB_SIZE, FULL_VOCAB_SIZE, FastGemma4Tokenizer, VocabRemapper

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeArray:
    """Pure-NumPy array stand-in supporting MLX array attributes and slicing without GPU dispatch."""

    def __init__(self, data: np.ndarray | Any, dtype: Any = None) -> None:
        arr = data.data if isinstance(data, _FakeArray) else np.asarray(data)
        self.data = arr
        self._dtype = dtype if dtype is not None else str(arr.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def T(self) -> "_FakeArray":
        return _FakeArray(self.data.T, dtype=self._dtype)

    def reshape(self, *shape: int) -> "_FakeArray":
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return _FakeArray(self.data.reshape(target), dtype=self._dtype)

    def transpose(self, *axes: int) -> "_FakeArray":
        if not axes:
            return _FakeArray(self.data.T, dtype=self._dtype)
        ax = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        return _FakeArray(np.transpose(self.data, axes=ax), dtype=self._dtype)

    def astype(self, dtype: Any) -> "_FakeArray":
        s_dt = str(dtype)
        if s_dt.endswith("float32"):
            return _FakeArray(self.data.astype(np.float32), dtype=dtype)
        if s_dt.endswith("float16"):
            return _FakeArray(self.data.astype(np.float16), dtype=dtype)
        if s_dt.endswith("int8"):
            return _FakeArray(self.data.astype(np.int8), dtype=dtype)
        if s_dt.endswith("uint32"):
            return _FakeArray(self.data.astype(np.uint32), dtype=dtype)
        if s_dt.endswith("int32"):
            return _FakeArray(self.data.astype(np.int32), dtype=dtype)
        return _FakeArray(self.data.astype(np.float32), dtype=dtype)

    def tolist(self) -> Any:
        return self.data.tolist()

    def item(self) -> Any:
        return self.data.item()

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = np.asarray(self.data, dtype=dtype)
        return arr.copy() if copy else arr

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Any) -> "_FakeArray":
        k = _unwrap_index(key)
        return _FakeArray(self.data[k], dtype=self._dtype)

    def __setitem__(self, key: Any, value: Any) -> None:
        k = _unwrap_index(key)
        val = value.data if isinstance(value, _FakeArray) else np.asarray(value)
        self.data[k] = val

    def __matmul__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else np.asarray(other)
        return _FakeArray(self.data.astype(np.float32) @ rhs.astype(np.float32), dtype=self._dtype)

    def __add__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data + rhs, dtype=self._dtype)

    def __sub__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data - rhs, dtype=self._dtype)

    def __mul__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data * rhs, dtype=self._dtype)

    def __truediv__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data.astype(np.float32) / np.asarray(rhs, dtype=np.float32), dtype=self._dtype)

    def __rtruediv__(self, other: Any) -> "_FakeArray":
        lhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(np.asarray(lhs, dtype=np.float32) / self.data.astype(np.float32), dtype=self._dtype)

    def __pow__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(np.power(self.data.astype(np.float32), rhs), dtype=self._dtype)

    def __rpow__(self, other: Any) -> "_FakeArray":
        lhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(np.power(lhs, self.data.astype(np.float32)), dtype=self._dtype)

    def __and__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data & rhs, dtype=self._dtype)

    def __or__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data | rhs, dtype=self._dtype)

    def __invert__(self) -> "_FakeArray":
        return _FakeArray(~self.data, dtype=self._dtype)

    def __ge__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data >= rhs, dtype="bool")

    def __le__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data <= rhs, dtype="bool")

    def __lt__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data < rhs, dtype="bool")

    def __gt__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data > rhs, dtype="bool")

    def __radd__(self, other: Any) -> "_FakeArray":
        lhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(lhs + self.data, dtype=self._dtype)

    def __rsub__(self, other: Any) -> "_FakeArray":
        lhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(lhs - self.data, dtype=self._dtype)

    def __rmul__(self, other: Any) -> "_FakeArray":
        lhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(lhs * self.data, dtype=self._dtype)

    def __mod__(self, other: Any) -> "_FakeArray":
        rhs = other.data if isinstance(other, _FakeArray) else other
        return _FakeArray(self.data % rhs, dtype=self._dtype)


def _unwrap_index(key: Any) -> Any:
    if isinstance(key, _FakeArray):
        return key.data
    if isinstance(key, tuple):
        return tuple(_unwrap_index(x) for x in key)
    return key


class _FakeMx:
    """Minimal pure-NumPy MLX module shim guaranteeing zero GPU/Metal calls."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    int32 = "int32"
    int8 = "int8"

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def array(self, data: Any, dtype: Any = None) -> _FakeArray:
        if isinstance(data, _FakeArray):
            return data.astype(dtype) if dtype is not None else _FakeArray(data.data.copy(), dtype=data.dtype)
        np_dt = np.int32 if dtype == self.int32 else (np.float32 if dtype in (self.bfloat16, self.float32, None) else np.float16)
        return _FakeArray(np.array(data, dtype=np_dt), dtype=dtype or self.float32)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> _FakeArray:
        np_dt = np.int8 if dtype == self.int8 else (np.int32 if dtype == self.int32 else (np.float16 if dtype == self.float16 else np.float32))
        return _FakeArray(np.zeros(shape, dtype=np_dt), dtype=dtype or self.float32)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> _FakeArray:
        np_dt = np.float16 if dtype == self.float16 else np.float32
        return _FakeArray(np.ones(shape, dtype=np_dt), dtype=dtype or self.float32)

    def full(self, shape: tuple[int, ...], val: Any, dtype: Any = None) -> _FakeArray:
        np_dt = np.int32 if dtype == self.int32 else np.float32
        return _FakeArray(np.full(shape, val, dtype=np_dt), dtype=dtype or self.int32)

    def arange(self, start: int, stop: int | None = None, step: int = 1, dtype: Any = None) -> _FakeArray:
        arr = np.arange(start, dtype=np.int32) if stop is None else np.arange(start, stop, step, dtype=np.int32)
        return _FakeArray(arr, dtype=dtype or self.int32)

    def pad(self, x: _FakeArray, pad_width: list[tuple[int, int]]) -> _FakeArray:
        return _FakeArray(np.pad(x.data, pad_width, mode="constant"), dtype=x.dtype)

    def concatenate(self, arrs: list[_FakeArray], axis: int = 0) -> _FakeArray:
        return _FakeArray(np.concatenate([a.data for a in arrs], axis=axis), dtype=arrs[0].dtype)

    def broadcast_to(self, x: _FakeArray, shape: tuple[int, ...]) -> _FakeArray:
        return _FakeArray(np.broadcast_to(x.data, shape), dtype=x.dtype)

    def contiguous(self, x: _FakeArray) -> _FakeArray:
        return _FakeArray(np.ascontiguousarray(x.data), dtype=x.dtype)

    def argmax(self, x: _FakeArray, axis: int = -1) -> _FakeArray:
        return _FakeArray(np.argmax(x.data, axis=axis).astype(np.int32), dtype=self.int32)

    def where(self, cond: Any, x: Any, y: Any) -> _FakeArray:
        c = cond.data if isinstance(cond, _FakeArray) else np.asarray(cond)
        xv = x.data if isinstance(x, _FakeArray) else x
        yv = y.data if isinstance(y, _FakeArray) else y
        dt = x.dtype if isinstance(x, _FakeArray) else (y.dtype if isinstance(y, _FakeArray) else self.float32)
        return _FakeArray(np.where(c, xv, yv), dtype=dt)

    def isnan(self, x: _FakeArray) -> _FakeArray:
        return _FakeArray(np.isnan(x.data), dtype="bool")

    def any(self, x: _FakeArray, axis: int | None = None) -> _FakeArray:
        return _FakeArray(np.asarray(np.any(x.data, axis=axis)), dtype="bool")

    def abs(self, x: _FakeArray) -> _FakeArray:
        return _FakeArray(np.abs(x.data), dtype=x.dtype)

    def eval(self, *args: Any) -> None:
        self.events.append(("eval", len(args)))

    def async_eval(self, *args: Any) -> None:
        self.events.append(("async_eval", len(args)))

    def clear_cache(self) -> None:
        self.events.append(("clear_cache", 0))

    def get_cache_memory(self) -> int:
        return 0

    def reset_peak_memory(self) -> None:
        pass

    def get_peak_memory(self) -> int:
        return 0

    class fast:
        @staticmethod
        def rms_norm(x: _FakeArray, w: _FakeArray, eps: float) -> _FakeArray:
            arr = x.data.astype(np.float32)
            inv = 1.0 / np.sqrt(np.mean(arr * arr, axis=-1, keepdims=True) + eps)
            return _FakeArray(arr * inv * w.data.astype(np.float32), dtype=x.dtype)

        @staticmethod
        def scaled_dot_product_attention(q: Any, k: Any, v: Any, scale: float = 1.0, mask: Any = None) -> _FakeArray:
            B, H, Lq, D = q.shape
            return _FakeArray(np.zeros((B, H, Lq, D), dtype=np.float32), dtype=q.dtype)


def _cpu_geglu(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    g32 = g.astype(np.float32)
    u32 = u.astype(np.float32)
    t = np.tanh(0.7978845608 * g32 * (1.0 + 0.044715 * g32 * g32))
    return 0.5 * g32 * (1.0 + t) * u32


def test_tier1_resolves_from_worktree() -> None:
    """Guardrail: verify imported `fast_gemma4` resolves to this isolated worktree."""
    resolved = Path(fast_gemma4.__file__).resolve()
    assert str(resolved).startswith(str((REPO_ROOT / "src" / "fast_gemma4").resolve())), resolved


def test_py_compile_all_modified_modules_cpu() -> None:
    """Verify every Python source file in `src/fast_gemma4` byte-compiles cleanly."""
    for py_file in sorted((REPO_ROOT / "src" / "fast_gemma4").rglob("*.py")):
        py_compile.compile(str(py_file), doraise=True)


def test_mlp_dispatch_16_to_64_and_64x64_prefill_cpu() -> None:
    """Verify Deliverable #1 & #3: `16<=M<=64` (`MPP_pad32` GeGLU + MLX `down_proj`), `64<M<512` (`32x32`), `M>=512` (`64x64`)."""
    fake_mx = _FakeMx()
    geglu_calls: list[tuple[int, int, int, int]] = []
    mpp_lin_calls: list[tuple[int, int, int]] = []
    fused_split_calls: list[int] = []

    def _fake_make_geglu_kern(tm: int, tn: int):
        def _run(*, inputs: list[_FakeArray], output_shapes: list[tuple[int, int]], grid: Any, threadgroup: Any, output_dtypes: Any):
            a2, wg, wu = inputs
            geglu_calls.append((tm, tn, int(a2.shape[0]), int(wg.shape[0])))
            assert int(a2.shape[0]) % tm == 0
            assert output_shapes == [(int(a2.shape[0]), int(wg.shape[0]))]
            out = _cpu_geglu(a2.data @ wg.data.T, a2.data @ wu.data.T)
            return [_FakeArray(out, dtype=fake_mx.bfloat16)]

        return _run

    def _fake_mpp_linear(x: _FakeArray, w_nk: _FakeArray, tm: int = 32, tn: int = 32) -> _FakeArray:
        M = x.size // x.shape[-1]
        mpp_lin_calls.append((M, tm, tn))
        return _FakeArray(x.data @ w_nk.data.T, dtype=x.dtype)

    def _fake_fused_geglu_split(gate_up: _FakeArray) -> _FakeArray:
        H = gate_up.shape[-1] // 2
        fused_split_calls.append(gate_up.size // gate_up.shape[-1])
        return _FakeArray(_cpu_geglu(gate_up.data[..., :H], gate_up.data[..., H:]), dtype=gate_up.dtype)

    rng = np.random.default_rng(17)
    K, H = 64, 64
    w_g = _FakeArray(rng.standard_normal((H, K)).astype(np.float32) * 0.1, dtype=fake_mx.bfloat16)
    w_u = _FakeArray(rng.standard_normal((H, K)).astype(np.float32) * 0.1, dtype=fake_mx.bfloat16)
    w_gu = _FakeArray(np.concatenate([w_g.data, w_u.data], axis=0), dtype=fake_mx.bfloat16)
    w_dn = _FakeArray(rng.standard_normal((K, H)).astype(np.float32) * 0.1, dtype=fake_mx.bfloat16)

    assert fg4_kernels.mpp_fused_geglu_bf16_v4 is fs_mpp_bf16.mpp_fused_geglu_bf16
    with (
        patch.object(fs_mpp_bf16, "mx", fake_mx),
        patch.object(fs_mpp_bf16, "_mpp_bf16_supported", return_value=True),
        patch.object(fs_mpp_bf16, "_is_apple10_nax", return_value=False),
        patch.object(fs_mpp_bf16, "make_mpp_bf16_geglu_kernel", side_effect=_fake_make_geglu_kern),
        patch.object(fs_mpp_bf16, "fused_geglu_split", side_effect=_fake_fused_geglu_split),
        patch.object(fg4_kernels, "mx", fake_mx),
        patch.object(fg4_kernels, "mpp_linear_bf16", side_effect=_fake_mpp_linear),
        patch.object(fg4_kernels, "fused_geglu_split", side_effect=_fake_fused_geglu_split),
    ):
        # 1. Direct mpp_fused_geglu_bf16_v4 sweeps (Apple9 min_m=16 + Apple10 min_m=64 thresholds)
        for M in (0, 1, 8, 15, 16, 20, 24, 32, 48, 64, 80, 96, 97, 128, 512):
            geglu_calls.clear()
            mpp_lin_calls.clear()
            fused_split_calls.clear()
            x = _FakeArray(rng.standard_normal((M, K)).astype(np.float32) * 0.2, dtype=fake_mx.bfloat16)
            got = fg4_kernels.mpp_fused_geglu_bf16_v4(x, w_g, w_u, w_gu, m32_cutoff=96)
            assert got.shape == (M, H)
            if M == 0:
                assert not geglu_calls and not fused_split_calls
            elif M < 16:
                assert not geglu_calls and not mpp_lin_calls
                assert fused_split_calls == [M]
            elif M <= 96:
                exp_mp = M + ((32 - (M % 32)) % 32)
                assert geglu_calls == [(32, 32, exp_mp, H)], f"M={M}: {geglu_calls}"
                assert not mpp_lin_calls and not fused_split_calls
            else:
                exp_mp = M + ((64 - (M % 64)) % 64)
                assert geglu_calls == [(64, 32, exp_mp, H)], f"M={M}: {geglu_calls}"
                assert not mpp_lin_calls and not fused_split_calls
            if M > 0:
                ref = _cpu_geglu(x.data @ w_g.data.T, x.data @ w_u.data.T)
                assert np.allclose(got.data, ref, atol=1e-5)

        # Apple10 (NAX) default min_m=64: M=32 uses fused_geglu_split, M=64 enters 32x32 MPP
        with patch.object(fs_mpp_bf16, "_is_apple10_nax", return_value=True):
            for M_nax, use_mpp in ((32, False), (64, True)):
                geglu_calls.clear()
                fused_split_calls.clear()
                x_nax = _FakeArray(rng.standard_normal((M_nax, K)).astype(np.float32) * 0.2, dtype=fake_mx.bfloat16)
                got_nax = fg4_kernels.mpp_fused_geglu_bf16_v4(x_nax, w_g, w_u, w_gu, m32_cutoff=96)
                ref_nax = _cpu_geglu(x_nax.data @ w_g.data.T, x_nax.data @ w_u.data.T)
                assert np.allclose(got_nax.data, ref_nax, atol=1e-5)
                if use_mpp:
                    assert geglu_calls == [(32, 32, 64, H)] and not fused_split_calls
                else:
                    assert not geglu_calls and fused_split_calls == [32]

        # Divisibility guard: H not divisible by 32 falls back to MLX
        geglu_calls.clear()
        fused_split_calls.clear()
        w_g_odd = _FakeArray(rng.standard_normal((48, K)).astype(np.float32), dtype=fake_mx.bfloat16)
        w_u_odd = _FakeArray(rng.standard_normal((48, K)).astype(np.float32), dtype=fake_mx.bfloat16)
        w_gu_odd = _FakeArray(np.concatenate([w_g_odd.data, w_u_odd.data], axis=0), dtype=fake_mx.bfloat16)
        x24 = _FakeArray(rng.standard_normal((24, K)).astype(np.float32), dtype=fake_mx.bfloat16)
        got_odd = fg4_kernels.mpp_fused_geglu_bf16_v4(x24, w_g_odd, w_u_odd, w_gu_odd)
        assert got_odd.shape == (24, 48)
        assert not geglu_calls and fused_split_calls == [24]

    # 2. FastGemma4Model._run_mlp full routing table
    m = FastGemma4Model.__new__(FastGemma4Model)
    m.use_mpp = True
    m.mpp_m32_cutoff = 96

    down_mlx_calls: list[int] = []
    gu_mlx_calls: list[int] = []
    v4_geglu_calls: list[int] = []
    model_mpp_lin_calls: list[tuple[int, int, int]] = []

    def _ld_gu(x: _FakeArray) -> _FakeArray:
        gu_mlx_calls.append(x.size // x.shape[-1])
        return x @ w_gu.T

    def _ld_dn(x: _FakeArray) -> _FakeArray:
        down_mlx_calls.append(x.size // x.shape[-1])
        return x @ w_dn.T

    def _model_v4(x: _FakeArray, wg: _FakeArray, wu: _FakeArray, wgu: _FakeArray, m32_cutoff: int = 96) -> _FakeArray:
        v4_geglu_calls.append(x.size // x.shape[-1])
        return _FakeArray(_cpu_geglu(x.data @ wg.data.T, x.data @ wu.data.T), dtype=x.dtype)

    def _model_mpp_lin(x: _FakeArray, w: _FakeArray, tm: int = 32, tn: int = 32) -> _FakeArray:
        model_mpp_lin_calls.append((x.size // x.shape[-1], tm, tn))
        return _FakeArray(x.data @ w.data.T, dtype=x.dtype)

    ld = {
        "gate_up_proj": _ld_gu,
        "down_proj": _ld_dn,
        "w_g": w_g,
        "w_u": w_u,
        "w_gu": w_gu,
        "w_dn": w_dn,
    }

    with (
        patch.object(fg4_model, "_supports_mpp_bf16", return_value=True),
        patch.object(fg4_model, "_detect_gpu_family", return_value="apple9"),
        patch.object(fg4_model, "mpp_fused_geglu_bf16", side_effect=_model_v4),
        patch.object(fg4_model, "mpp_linear_bf16", side_effect=_model_mpp_lin),
        patch.object(fg4_model, "fused_geglu_split", side_effect=_fake_fused_geglu_split),
    ):
        for M, allow_mpp, expect in [
            (1, True, "all_mlx"),
            (15, True, "all_mlx"),
            (16, True, "hybrid_32_mlx_dn"),
            (24, True, "hybrid_32_mlx_dn"),
            (32, True, "hybrid_32_mlx_dn"),
            (64, True, "hybrid_32_mlx_dn"),
            (65, True, "all_mlx"),
            (256, True, "all_mlx"),
            (511, True, "all_mlx"),
            (512, True, "all_mlx"),
            (1024, True, "all_mlx"),
            (2047, True, "all_mlx"),
            (2048, True, "mpp_64x64"),
            (4096, True, "mpp_64x64"),
            (4096, False, "all_mlx"),
        ]:
            down_mlx_calls.clear()
            gu_mlx_calls.clear()
            v4_geglu_calls.clear()
            model_mpp_lin_calls.clear()
            x = _FakeArray(rng.standard_normal((M, 1, K)).astype(np.float32), dtype=fake_mx.bfloat16)
            out = m._run_mlp(x, ld, allow_mpp=allow_mpp)
            assert out.shape == (M, 1, K)
            if expect == "all_mlx":
                assert gu_mlx_calls == [M] and down_mlx_calls == [M]
                assert not v4_geglu_calls and not model_mpp_lin_calls
            elif expect == "hybrid_32_mlx_dn":
                assert v4_geglu_calls == [M] and down_mlx_calls == [M]
                assert not gu_mlx_calls and not model_mpp_lin_calls
            elif expect == "mpp_64x64":
                assert v4_geglu_calls == [M] and model_mpp_lin_calls == [(M, 64, 64)]
                assert not gu_mlx_calls and not down_mlx_calls


def test_prefill_allow_mpp_gate_and_skip_s0_self_dequant_cpu() -> None:
    """Verify Deliverable #3 (`_run_mlp` single-owner `16<=M<=64` / `M>=2048`) & #4 (skip `s0==0` i8 self-dequant + SWA pre-scatter gather)."""
    src = (REPO_ROOT / "src" / "fast_gemma4" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Single owner: zero duplicated call-site `allow_mpp=` expressions in model.py!
    allow_mpp_calls: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "allow_mpp":
            allow_mpp_calls.append(node.lineno)
    assert allow_mpp_calls == [], f"Unexpected duplicated allow_mpp kwargs at lines {allow_mpp_calls}"
    run_mlp_src = inspect.getsource(FastGemma4Model._run_mlp)
    assert "16 <= M <= 64" in run_mlp_src and "M >= prefill_min_m" in run_mlp_src

    # Verify `_prefill_early14_slot_group` runtime behavior on s0==0 and s0>0 (with swa_ring_cap=512 wrap)
    fake_mx = _FakeMx()
    m = FastGemma4Model.__new__(FastGemma4Model)
    m.first_kv_shared = 3
    m.window_size = 512
    m.eps = 1e-6
    m.act_dtype = fake_mx.bfloat16
    m.kv_i8 = True
    m.use_mpp = True
    m.hidden_size = 8
    m.ple_dim = 4
    m._layer_idx_mx = [_FakeArray([i], dtype=fake_mx.int32) for i in range(3)]

    def _build_layers() -> list[dict[str, Any]]:
        ident = lambda x: x
        return [
            {
                "idx": 0,
                "is_full": True,
                "head_dim": 8,
                "inv_freq": _FakeArray(np.ones((4,), dtype=np.float32)),
                "w_in": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_qn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_kn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_attn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_pre_ff": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_ff": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_ple": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_next_norm": _FakeArray(np.ones((8,), dtype=np.float32)),
                "layer_scalar": _FakeArray([1.0], dtype=fake_mx.bfloat16),
                "qkv_proj": lambda x: _FakeArray(np.zeros((*x.shape[:-1], 80), dtype=np.float32), dtype=fake_mx.bfloat16),
                "o_proj": ident,
                "ple_gate": lambda x: _FakeArray(np.zeros((*x.shape[:-1], 4), dtype=np.float32), dtype=fake_mx.bfloat16),
                "ple_proj": ident,
            },
            {
                "idx": 1,
                "is_full": False,
                "head_dim": 8,
                "inv_freq": _FakeArray(np.ones((4,), dtype=np.float32)),
                "w_in": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_qn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_kn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_attn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_pre_ff": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_ff": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_post_ple": _FakeArray(np.ones((8,), dtype=np.float32)),
                "w_next_norm": _FakeArray(np.ones((8,), dtype=np.float32)),
                "layer_scalar": _FakeArray([1.0], dtype=fake_mx.bfloat16),
                "qkv_proj": lambda x: _FakeArray(np.zeros((*x.shape[:-1], 80), dtype=np.float32), dtype=fake_mx.bfloat16),
                "o_proj": ident,
                "ple_gate": lambda x: _FakeArray(np.zeros((*x.shape[:-1], 4), dtype=np.float32), dtype=fake_mx.bfloat16),
                "ple_proj": ident,
            },
            {
                "idx": 2,
                "is_full": False,
                "head_dim": 8,
                "inv_freq": _FakeArray(np.ones((4,), dtype=np.float32)),
                "w_kn": _FakeArray(np.ones((8,), dtype=np.float32)),
                "kv_proj_only": lambda x: _FakeArray(np.zeros((*x.shape[:-1], 16), dtype=np.float32), dtype=fake_mx.bfloat16),
            },
        ]

    m.layers = _build_layers()
    allow_mpp_log: list[tuple[int, bool]] = []
    sdpa_k_markers: list[tuple[int, np.ndarray]] = []
    quant_calls = 0
    gather_calls: list[tuple[int, list[int]]] = []

    def _fake_embed_pli(toks: _FakeArray) -> tuple[_FakeArray, _FakeArray]:
        B, L = toks.shape
        # Encode token position marker directly so we can assert exact K keys passed to SDPA
        h = _FakeArray(np.broadcast_to(toks.data[:, :, None].astype(np.float32), (B, L, 8)).copy(), dtype=fake_mx.bfloat16)
        pli = _FakeArray(np.zeros((B, L, 2, 4), dtype=np.float32), dtype=fake_mx.bfloat16)
        return h, pli

    def _fake_qkv_nr(qkv: _FakeArray, wqn: Any, wkn: Any, inv: Any, off_meta: _FakeArray, eps: float, **_kw: Any):
        Bg, lc, _ = qkv.shape
        off_val = int(off_meta.data[0])
        pos_ids = (off_val + np.arange(lc, dtype=np.float32))[None, :, None]
        q = _FakeArray(np.zeros((Bg, lc, 8, 8), dtype=np.float32), dtype=fake_mx.bfloat16)
        k = _FakeArray(np.broadcast_to(pos_ids, (Bg, lc, 8)).copy(), dtype=fake_mx.bfloat16)
        v = _FakeArray(np.broadcast_to(pos_ids + 10000.0, (Bg, lc, 8)).copy(), dtype=fake_mx.bfloat16)
        return q, k, v

    def _fake_kv_only(kv: _FakeArray, wkn: Any, inv: Any, off_bg: _FakeArray, eps: float, **_kw: Any):
        Bg, lc, _ = kv.shape
        k = _FakeArray(np.zeros((Bg, lc, 8), dtype=np.float32), dtype=fake_mx.bfloat16)
        v = _FakeArray(np.zeros((Bg, lc, 8), dtype=np.float32), dtype=fake_mx.bfloat16)
        return k, v

    def _fake_quant(k_rot: _FakeArray, v_norm: _FakeArray):
        nonlocal quant_calls
        quant_calls += 1
        Bg, lc, D = k_rot.shape
        # Preserve token marker in int8/float32-compatible slot (use mod/marker or store exact offset in scale!)
        kq = _FakeArray(np.ones((Bg, lc, D), dtype=np.int8), dtype=fake_mx.int8)
        ks = _FakeArray(k_rot.data[:, :, 0].astype(np.float16), dtype=fake_mx.float16)
        vq = _FakeArray(np.ones((Bg, lc, D), dtype=np.int8), dtype=fake_mx.int8)
        vs = _FakeArray(v_norm.data[:, :, 0].astype(np.float16), dtype=fake_mx.float16)
        return kq, ks, vq, vs

    def _fake_gather_deq(kc: _FakeArray, ksc: _FakeArray, vc: _FakeArray, vsc: _FakeArray, b_idx: _FakeArray, p_slots: _FakeArray):
        slots = p_slots.data.tolist()
        gather_calls.append((len(slots), slots))
        vals_k = ksc.data[b_idx.data[:, None], p_slots.data[None, :]].astype(np.float32)
        vals_v = vsc.data[b_idx.data[:, None], p_slots.data[None, :]].astype(np.float32)
        kd = np.broadcast_to(vals_k[:, None, :, None], (len(b_idx), 1, len(slots), 8)).copy()
        vd = np.broadcast_to(vals_v[:, None, :, None], (len(b_idx), 1, len(slots), 8)).copy()
        return _FakeArray(kd, dtype=fake_mx.bfloat16), _FakeArray(vd, dtype=fake_mx.bfloat16)

    def _fake_sdpa(q: _FakeArray, k: _FakeArray, v: _FakeArray, scale: float = 1.0, mask: Any = None):
        sdpa_k_markers.append((int(k.shape[2]), k.data[0, 0, :, 0].copy()))
        B, H, Lq, D = q.shape
        return _FakeArray(np.zeros((B, H, Lq, D), dtype=np.float32), dtype=q.dtype)

    def _fake_run_mlp(mlp_in: _FakeArray, ld: dict[str, Any], allow_mpp: bool = True) -> _FakeArray:
        m_val = mlp_in.size // mlp_in.shape[-1]
        allow_mpp_log.append((m_val, bool(allow_mpp and (16 <= m_val <= 64 or m_val >= 2048))))
        return _FakeArray(np.zeros(mlp_in.shape, dtype=np.float32), dtype=mlp_in.dtype)

    fake_mx.fast.scaled_dot_product_attention = _fake_sdpa  # type: ignore[method-assign]
    m._compute_embed_and_pli_early14 = _fake_embed_pli  # type: ignore[method-assign]
    m._run_mlp = _fake_run_mlp  # type: ignore[method-assign]

    cache = {
        "batch_size": 16,
        "l_sys": 0,
        "k_sys": [],
        "v_sys": [],
        "swa_cap": 512,
        "full_cap": 1024,
        "kv_i8": True,
        "layers": [
            {
                "k": fake_mx.zeros((16, 1024, 8), dtype=fake_mx.int8),
                "k_s": fake_mx.ones((16, 1024), dtype=fake_mx.float16),
                "v": fake_mx.zeros((16, 1024, 8), dtype=fake_mx.int8),
                "v_s": fake_mx.ones((16, 1024), dtype=fake_mx.float16),
                "cap": 1024,
                "ring": False,
            },
            {
                "k": fake_mx.zeros((16, 512, 8), dtype=fake_mx.int8),
                "k_s": fake_mx.ones((16, 512), dtype=fake_mx.float16),
                "v": fake_mx.zeros((16, 512, 8), dtype=fake_mx.int8),
                "v_s": fake_mx.ones((16, 512), dtype=fake_mx.float16),
                "cap": 512,
                "ring": True,
            },
            {
                "k": fake_mx.zeros((16, 512, 8), dtype=fake_mx.int8),
                "k_s": fake_mx.ones((16, 512), dtype=fake_mx.float16),
                "v": fake_mx.zeros((16, 512, 8), dtype=fake_mx.int8),
                "v_s": fake_mx.ones((16, 512), dtype=fake_mx.float16),
                "cap": 512,
                "ring": True,
            },
        ],
    }

    with (
        patch.object(fg4_model, "mx", fake_mx),
        patch.object(fg4_model, "fused_qkv_packed_norm_rope", side_effect=_fake_qkv_nr),
        patch.object(fg4_model, "fused_kv_only_norm_rope_ragged", side_effect=_fake_kv_only),
        patch.object(fg4_model, "quantize_kv_pair_i8", side_effect=_fake_quant),
        patch.object(fg4_model, "gather_dequant_kv_pair_i8_bf16", side_effect=_fake_gather_deq),
        patch.object(fg4_model, "fused_norm_add_norm", side_effect=lambda h, x, *a: (h, x)),
        patch.object(fg4_model, "fused_res_rms_norm", side_effect=lambda r, x, *a: r),
        patch.object(fg4_model, "fused_ple_geglu_indexed", side_effect=lambda g, *a: g),
    ):
        # Run 640-token suffix (3 chunks: 0..256, 256..512, 512..640) at Bg=16
        # -> Chunk 0 & 1 have Bg*lc = 4096 >= 2048 -> allow_mpp=True!
        # -> Chunk 2 has Bg*lc = 16*128 = 2048 >= 2048 -> allow_mpp=True!
        sufs16 = [list(range(641)) for _ in range(16)]
        m._prefill_early14_slot_group(sufs16, list(range(16)), cache, chunk_sz=256)
        assert quant_calls == 3 * 3  # 3 layers x 3 chunks
        assert all(flag is True for m_val, flag in allow_mpp_log if m_val == 4096 or m_val == 2048)

        # At chunk 0 (s0=0): zero gather calls, SDPA received k_rot [0..255] directly!
        assert sdpa_k_markers[0][0] == 256
        assert np.array_equal(sdpa_k_markers[0][1], np.arange(256, dtype=np.float32))
        assert sdpa_k_markers[1][0] == 256
        assert np.array_equal(sdpa_k_markers[1][1], np.arange(256, dtype=np.float32))

        # At chunk 2 (s0=512, s1=640, SWA layer idx=1 with cap=512!):
        # Pre-scatter gather MUST fetch [1..511] BEFORE scattering [512..639], so SDPA sees exact [1..639]!
        swa_chunk2_keys = sdpa_k_markers[-1][1]
        assert len(swa_chunk2_keys) == 639
        assert np.array_equal(swa_chunk2_keys, np.arange(1, 640, dtype=np.float32))

        # Run scattered Bg=4, lc=256 (Bg*lc = 1024 -> 512 <= M < 2048 -> allow_mpp MUST be False!)
        allow_mpp_log.clear()
        sufs4 = [list(range(257)) for _ in range(4)]
        m._prefill_early14_slot_group(sufs4, [0, 1, 2, 3], cache, chunk_sz=256)
        assert all(flag is False for m_val, flag in allow_mpp_log if m_val == 1024)


def test_defaults_swa512_and_roll_slots256_cpu() -> None:
    """Verify Deliverable #2: aligned `swa_ring_cap=512`, `b_slots=256`, `waves=1`, `select_execution_plan`."""
    from fast_gemma4.engine import select_execution_plan

    assert inspect.signature(FastGemma4Model.make_batch_cache).parameters["swa_ring_cap"].default == 512
    assert inspect.signature(FastGemma4Model.make_slot_cache).parameters["swa_ring_cap"].default == 512
    sig_roll = inspect.signature(run_decode_rolling).parameters
    assert sig_roll["b_slots"].default == 256
    assert sig_roll["swa_ring_cap"].default == 512
    assert sig_roll["group_size"].default == 16
    assert sig_roll["sort_by_length"].default is True
    assert inspect.signature(FastGemma4Engine.run_rolling).parameters["B_slots"].default == 256
    assert inspect.signature(FastGemma4Engine.run_rolling).parameters["swa_ring_cap"].default == 512

    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=256, batch_size=None) == (False, 256)
    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=384, batch_size=None) == (False, 256)
    assert select_execution_plan("hetero", True, lcp=1024, max_suf=400, n_docs=128, batch_size=None) == (True, 64)

    # Validate power-of-2 & >= window_size enforcement
    m = FastGemma4Model.__new__(FastGemma4Model)
    m.window_size = 512
    m.first_kv_shared = 0
    m.layers = []
    m.act_dtype = "bfloat16"
    m.kv_i8 = True
    fake_mx = _FakeMx()
    with patch.object(fg4_model, "mx", fake_mx):
        with pytest.raises(ValueError):
            m.make_batch_cache({"l_sys": 0, "k_sys": [], "v_sys": []}, batch_size=2, swa_ring_cap=256)
        with pytest.raises(ValueError):
            m.make_slot_cache({"l_sys": 0, "k_sys": [], "v_sys": []}, batch_size=2, swa_ring_cap=600)
        c512 = m.make_slot_cache({"l_sys": 0, "k_sys": [], "v_sys": []}, batch_size=2, swa_ring_cap=512)
        c1024 = m.make_slot_cache({"l_sys": 0, "k_sys": [], "v_sys": []}, batch_size=2, swa_ring_cap=1024)
        assert c512["swa_cap"] == 512 and c1024["swa_cap"] == 1024

    # Verify engine.generate_batch passes roll_slots=256 and sets stats["waves"] == 1
    eng = FastGemma4Engine.__new__(FastGemma4Engine)
    eng._gpu_lock = threading.Lock()
    eng.default_mode = "auto"
    eng.ane_available = False
    eng._prefix_cache = {}  # type: ignore[assignment]
    eng.max_prefix_cache = 4
    eng.model = type("_M", (), {"kv_i8": True, "window_size": 512})()
    eng.tokenizer = type(
        "_T",
        (),
        {
            "bos_token_id": 2,
            "stop_token_ids": frozenset({1}),
            "decode": staticmethod(lambda ids, skip_special_tokens=True: "{}"),
        },
    )()
    eng.encode_prompts = lambda prompts, **kw: [[2, 10, 11] for _ in prompts]  # type: ignore[method-assign]
    eng.get_prefix_kv = lambda sys_ids: {"l_sys": 0, "k_sys": [], "v_sys": []}  # type: ignore[method-assign]

    seen_roll_kwargs: list[dict[str, Any]] = []

    def _fake_roll(model, stop_ids, grammar, pkv, sufs, **kwargs):
        seen_roll_kwargs.append(kwargs)
        return {
            "prefill_sec": 0.01,
            "decode_sec": 0.02,
            "gen_toks": [[12] for _ in sufs],
            "valid_schemas": [True for _ in sufs],
            "tot_out_tok": len(sufs),
            "llm_steps": 1,
            "ff_early14_steps": 0,
            "ff_tokens": 0,
            "mask_fallbacks": 0,
            "used_speculative": False,
        }

    with (
        patch.object(fg4_engine, "mx", fake_mx),
        patch.object(fg4_engine, "run_decode_rolling", side_effect=_fake_roll),
    ):
        res300 = eng.generate_batch(["doc"] * 300, mode="gpu")
        assert len(res300.items) == 300
        assert seen_roll_kwargs[0]["b_slots"] == 256
        assert res300.stats["waves"] == 1
        assert "tail_b_handoff" not in seen_roll_kwargs[0]


def _build_tiny_json_tokenizer(tmp_dir: Path) -> FastGemma4Tokenizer:
    """Create a tiny real Rust `tokenizers` BPE vocab for CPU `llguidance` + `FastGemma4Tokenizer` tests."""
    chars = [chr(c) for c in range(32, 127)]
    pieces = ["<pad>", "<eos>", "<bos>", "<turn|>"] + chars + ["alpha", "beta", "gamma"]
    vocab = {p: i for i, p in enumerate(pieces)}
    merges: list[tuple[str, str]] = []
    for word in ("alpha", "beta", "gamma"):
        cur = word[0]
        for ch in word[1:]:
            merges.append((cur, ch))
            if (cur + ch) not in vocab:
                vocab[cur + ch] = len(vocab)
            cur = cur + ch
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<pad>"))
    tok.add_special_tokens([AddedToken(s, special=True) for s in ("<pad>", "<eos>", "<bos>", "<turn|>")])
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[A-Za-z]+|."), behavior="isolated")
    tok.decoder = decoders.ByteLevel()
    tok_p = tmp_dir / "tokenizer.json"
    tok.save(str(tok_p))
    ft = FastGemma4Tokenizer(tok_p)
    ft.eos_token_id = 1
    ft.bos_token_id = 2
    ft.eot_token_id = 3
    ft.stop_token_ids = frozenset({1, 3})
    return ft


def test_drain_ff_slots_mixed_ks_swa512_wrap_cpu() -> None:
    """Verify BLOCKER fix: skewed mixed k_s (`priv=508, k_s=1` + `k_s=5`, `max_k=5`) splits before SWA=512 wrap."""
    fake_mx = _FakeMx()
    scatter_writes: list[tuple[int, list[int]]] = []
    early14_shapes: list[tuple[int, int]] = []
    decode_call_count = 0

    class _MixedKsModel:
        window_size = 512
        kv_i8 = True
        new_to_old = np.arange(32, dtype=np.int32)

        def __init__(self, suf_lens: list[int], ff_queues: dict[int, list[int]]) -> None:
            self.suf_lens = suf_lens
            self.ff_queues = ff_queues
            self.slot_to_doc: dict[int, int] = {}

        def make_slot_cache(self, prefix_kv, batch_size, doc_cap_full=1440, swa_ring_cap=512, kv_i8=None):
            # Track slot 0..511 marker values (`-1` = untouched, `0..511` = live key)
            ring_buf = np.arange(swa_ring_cap, dtype=np.int32)[None, :].repeat(batch_size, axis=0)
            return {
                "batch_size": batch_size,
                "l_sys": prefix_kv["l_sys"],
                "k_sys": prefix_kv["k_sys"],
                "v_sys": prefix_kv["v_sys"],
                "swa_cap": swa_ring_cap,
                "full_cap": doc_cap_full,
                "kv_i8": True,
                "swa_ring_shadow": ring_buf,
                "layers": [{"k": fake_mx.zeros((batch_size, swa_ring_cap, 2), dtype=fake_mx.int8), "cap": swa_ring_cap, "ring": True}],
            }

        def prefill_slots_lazy_lg(self, sub_sufs, target_slots, cache, *, group_size=16, return_logits=False, async_eval=False):
            for r, s in enumerate(target_slots):
                self.slot_to_doc[s] = int(sub_sufs[r][0]) - 100
            K = len(sub_sufs)
            x = _FakeArray(np.zeros((K, 1, 32), dtype=np.float32), dtype=fake_mx.float32)
            am = _FakeArray(np.full((K,), 10, dtype=np.int32), dtype=fake_mx.int32)
            next_offs = [cache["l_sys"] + len(s) for s in sub_sufs]
            return x, am, next_offs

        def forward_early14_slot(self, toks, cache, off_b, slot_idx):
            B_cur, L_q = toks.shape
            early14_shapes.append((B_cur, L_q))
            l_sys = cache["l_sys"]
            cap = cache["swa_cap"]
            for r in range(B_cur):
                s = int(slot_idx.data[r])
                off_s = int(off_b.data[r])
                slots_written = [((off_s - l_sys) + j) & (cap - 1) for j in range(L_q)]
                scatter_writes.append((s, slots_written))
                for j, sl in enumerate(slots_written):
                    # Mirror Model scatter: token 0 dummy would overwrite live marker if written!
                    cache["swa_ring_shadow"][s, sl] = int(toks.data[r, j])

        def forward_decode_step_slot(self, toks, cache, off_b, slot_idx, *, n_keep, return_logits=False):
            nonlocal decode_call_count
            decode_call_count += 1
            # Return stop token `1` for all rows once FF drained
            x = _FakeArray(np.zeros((n_keep, 1, 32), dtype=np.float32), dtype=fake_mx.float32)
            am = _FakeArray(np.ones((n_keep,), dtype=np.int32), dtype=fake_mx.int32)
            return x, am

    # Mock GrammarHelper returning exact mixed FF lengths:
    # Doc 0: off-l_sys after prefill + tok10 = 508, emits 1 FF token (+1 queued -> k_s=1)!
    # Doc 1: off-l_sys after prefill + tok10 = 100, emits 5 FF tokens (+1 queued -> k_s=5)!
    # Under padded L_q = max_k = 5 without the fix: doc 0 would scatter indices [508, 509, 510, 511, 0], overwriting live idx 0!
    class _SkewGH:
        upfront_ff: list[int] = []

        def __init__(self) -> None:
            self._idx = 0

        def new_matcher(self):
            doc_n = self._idx
            self._idx += 1
            ff_seq = [21] if doc_n == 0 else [21, 22, 23, 24, 25]

            class _M:
                def __init__(self, seq: list[int]) -> None:
                    self.seq = seq
                    self.used = False
                    self.stopped = False

                def try_consume_tokens(self, toks):
                    if toks == [1]:
                        self.stopped = True
                    return 1

                def is_stopped(self):
                    return self.stopped

                def is_accepting(self):
                    return self.stopped

                def compute_ff_tokens(self):
                    if not self.used:
                        self.used = True
                        return list(self.seq)
                    return []

                def consume_tokens(self, ff):
                    pass

            return _M(ff_seq)

    sufs = [
        [100] + [11] * 507,  # len=508 -> priv_s=508!
        [101] + [11] * 99,   # len=100 -> priv_s=100!
    ]
    model = _MixedKsModel([len(s) for s in sufs], {})
    with patch.object(fg4_spec, "mx", fake_mx):
        res = run_decode_rolling(
            model,
            frozenset({1}),
            _SkewGH(),
            {"l_sys": 10, "k_sys": [], "v_sys": []},
            sufs,
            max_tokens=16,
            b_slots=2,
            swa_ring_cap=512,
            max_ff_drain=8,
            sort_by_length=False,
        )
    assert res["valid_schemas"] == [True, True]
    # All forward_early14_slot dispatches MUST split to L_q=1!
    assert early14_shapes == [(2, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    # Slot 0 ONLY wrote slot 508 during drain (`c0=0`), NEVER `509..512 (0)`!
    slot0_written = [sl for s, sls in scatter_writes if s == 0 for sl in sls]
    assert slot0_written == [508]
    assert 0 not in slot0_written


def test_default_grammar_preserves_field_values_and_explicit_layouts(tmp_path: Path) -> None:
    from types import SimpleNamespace

    # Use real byte-level BPE merges for the two delimiter tokens Gemma predicts.
    pieces = ["<pad>", "<eos>", "<bos>"] + sorted(pre_tokenizers.ByteLevel.alphabet()) + ['":', 'Ġ"']
    tokenizer = Tokenizer(BPE(vocab={p: i for i, p in enumerate(pieces)}, merges=[('"', ':'), ('Ġ', '"')]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    for piece in pieces[:3]:
        tokenizer.add_special_tokens([AddedToken(piece, special=True)])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tok = FastGemma4Tokenizer(tokenizer_path)
    tok.eos_token_id = 1
    properties = {"a": {"type": "string"}, "b": {"type": "string"}}
    schema = {"type": "object", "properties": properties, "required": ["a", "b"], "additionalProperties": False}
    engine = object.__new__(FastGemma4Engine)
    engine.tokenizer = tok
    engine.model = SimpleNamespace(new_to_old=np.arange(tok.vocab_size, dtype=np.int32))
    engine._grammar_cache = {}
    fixed = engine.get_grammar(schema)
    minified = engine.get_grammar(schema, whitespace_flexible=False)
    flexible = engine.get_grammar(schema, whitespace_flexible=True)
    values = {"a": "alpha: beta", "b": ": "}
    spaced_ids = tok.encode(json.dumps(values, separators=(",", ": ")))
    minified_ids = tok.encode(json.dumps(values, separators=(",", ":")))
    assert fixed.validate_token_sequence(spaced_ids)
    assert not fixed.validate_token_sequence(minified_ids)
    assert minified.validate_token_sequence(minified_ids)
    assert not minified.validate_token_sequence(spaced_ids)
    assert flexible.validate_token_sequence(spaced_ids) and flexible.validate_token_sequence(minified_ids)
    assert not fixed.validate_token_sequence(tok.encode('{"a": 42,"b": "beta"}'))
    assert not fixed.validate_token_sequence(tok.encode('{"a": "alpha"}'))
    reversed_schema = {**schema, "properties": dict(reversed(properties.items()))}
    assert engine.get_grammar(reversed_schema) is not fixed
    assert engine.get_grammar(schema) is fixed


def test_rolling_valid_schema_propagation_and_per_doc_grammars_cpu(tmp_path: Path) -> None:
    """Verify A4 removal: valid_schemas propagated directly without validate_token_sequence + per_doc_grammars."""
    ft = _build_tiny_json_tokenizer(tmp_path)
    schema_ab = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    schema_a = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    gh_ab = GrammarHelper(ft, schema_ab, np.arange(ft.vocab_size, dtype=np.int32), whitespace_flexible=False)
    gh_a = GrammarHelper(ft, schema_a, np.arange(ft.vocab_size, dtype=np.int32), whitespace_flexible=False)

    def _non_ff_suffix(gh: GrammarHelper, full_ids: list[int]) -> list[int]:
        m = gh.new_matcher()
        pos = len(gh.upfront_ff)
        out: list[int] = []
        while pos < len(full_ids):
            tid = full_ids[pos]
            if not m.consume_token(tid):
                raise AssertionError(f"Failed consuming token {tid!r} at pos={pos}")
            out.append(tid)
            pos += 1
            if m.is_stopped():
                break
            ff = m.compute_ff_tokens()
            if ff:
                m.consume_tokens(ff)
                pos += len(ff)
                if m.is_stopped():
                    break
        return out

    fake_mx = _FakeMx()
    id_a = ft.encode("alpha", add_special_tokens=False)[0]
    full_ab = list(gh_ab.upfront_ff) + ft.encode('alpha","b":"beta"}', add_special_tokens=False) + [ft.eos_token_id]
    full_a = list(gh_a.upfront_ff) + ft.encode('alpha"}', add_special_tokens=False) + [ft.eos_token_id]
    seq_ab_ok = _non_ff_suffix(gh_ab, full_ab)
    seq_a_ok = _non_ff_suffix(gh_a, full_a)
    seq_trunc = [id_a] * 40  # exceeds max_tokens inside open string -> valid_schema=False!

    class _RollSchemaModel:
        window_size = 512
        kv_i8 = True
        new_to_old = np.arange(ft.vocab_size, dtype=np.int32)

        def __init__(self, scripts: dict[int, list[int]]) -> None:
            self.scripts = scripts
            self.slot_to_doc: dict[int, int] = {}
            self.doc_cur: dict[int, int] = {}

        def make_slot_cache(self, prefix_kv, batch_size, doc_cap_full=1440, swa_ring_cap=512, kv_i8=None):
            return {
                "batch_size": batch_size,
                "l_sys": prefix_kv["l_sys"],
                "k_sys": prefix_kv["k_sys"],
                "v_sys": prefix_kv["v_sys"],
                "swa_cap": swa_ring_cap,
                "full_cap": doc_cap_full,
                "kv_i8": True,
                "layers": [],
            }

        def prefill_slots_lazy_lg(self, sub_sufs, target_slots, cache, *, group_size=16, return_logits=False, async_eval=False):
            am_vals = np.zeros((len(sub_sufs),), dtype=np.int32)
            next_offs = []
            for r, s in enumerate(target_slots):
                d = int(sub_sufs[r][0]) - 1000
                self.slot_to_doc[s] = d
                self.doc_cur[d] = 0
                am_vals[r] = self._next(d)
                next_offs.append(cache["l_sys"] + len(sub_sufs[r]))
            return _FakeArray(np.zeros((len(sub_sufs), 1, ft.vocab_size), dtype=np.float32)), _FakeArray(am_vals, dtype=fake_mx.int32), next_offs

        def _next(self, d: int) -> int:
            c = self.doc_cur[d]
            seq = self.scripts[d]
            t = seq[min(c, len(seq) - 1)]
            self.doc_cur[d] = c + 1
            return t

        def forward_early14_slot(self, toks, cache, off_b, slot_idx):
            pass

        def forward_decode_step_slot(self, toks, cache, off_b, slot_idx, *, n_keep, return_logits=False):
            am_vals = np.array([self._next(self.slot_to_doc[int(slot_idx.data[r])]) for r in range(n_keep)], dtype=np.int32)
            return _FakeArray(np.zeros((n_keep, 1, ft.vocab_size), dtype=np.float32)), _FakeArray(am_vals, dtype=fake_mx.int32)

        def _apply_lm_head(self, x_sub, apply_softcap=True):
            return x_sub

    # 1. N=0 and max_tokens<=0 edge cases
    with patch.object(fg4_spec, "mx", fake_mx):
        empty_res = run_decode_rolling(_RollSchemaModel({}), frozenset({1}), gh_ab, {"l_sys": 0, "k_sys": [], "v_sys": []}, [])
        assert empty_res["gen_toks"] == [] and empty_res["valid_schemas"] == []

        zero_res = run_decode_rolling(
            _RollSchemaModel({}),
            frozenset({1}),
            gh_ab,
            {"l_sys": 0, "k_sys": [], "v_sys": []},
            [[10, 11], [12, 13]],
            max_tokens=0,
        )
        assert zero_res["gen_toks"] == [[], []] and zero_res["valid_schemas"] == [False, False]

    # 2. Mixed per_doc_grammars (`gh_ab`, `gh_a`, `None`, truncated `gh_a`) via FastGemma4Engine.generate_batch
    scripts = {
        0: list(seq_ab_ok),
        1: list(seq_a_ok),
        2: [id_a, ft.eos_token_id],
        3: list(seq_trunc),
    }
    eng = FastGemma4Engine.__new__(FastGemma4Engine)
    eng._gpu_lock = threading.Lock()
    eng.default_mode = "auto"
    eng.ane_available = False
    eng._prefix_cache = {}  # type: ignore[assignment]
    eng._grammar_cache = {}
    eng.max_prefix_cache = 4
    eng.model = _RollSchemaModel(scripts)  # type: ignore[assignment]
    eng.tokenizer = ft
    eng.encode_prompts = lambda prompts, **kw: [[2, 1000 + i] for i in range(len(prompts))]  # type: ignore[method-assign]
    eng.get_prefix_kv = lambda sys_ids: {"l_sys": 0, "k_sys": [], "v_sys": []}  # type: ignore[method-assign]

    # Verify GrammarHelper.validate_token_sequence is NEVER called during rolling post-processing!
    with (
        patch.object(fg4_spec, "mx", fake_mx),
        patch.object(fg4_engine, "mx", fake_mx),
        patch.object(GrammarHelper, "validate_token_sequence", side_effect=AssertionError("Must not re-validate from token 0")),
    ):
        batch_res = eng.generate_batch(
            ["p0", "p1", "p2", "p3"],
            per_doc_grammars=[gh_ab, gh_a, None, gh_a],
            max_tokens=14,
            mode="gpu",
        )
    assert [it.valid_schema for it in batch_res.items] == [True, True, True, False]
    assert batch_res.items[0].valid_json is True and batch_res.items[0].parsed == {"a": "alpha", "b": "beta"}
    assert batch_res.items[1].valid_json is True and batch_res.items[1].parsed == {"a": "alpha"}
    assert batch_res.items[2].valid_json is False and batch_res.items[2].valid_schema is True
    assert batch_res.items[3].finish_reason == "length"
    assert batch_res.stats["valid_schema_count"] == 3
    assert batch_res.stats["waves"] == 1


def test_ane_custom_dir_kv_i8_ingest_and_q14_trim_cpu(tmp_path: Path) -> None:
    """Verify the 3 custom-checkpoint Early14 fixes: bundle lookup, ingest_ane_chunk_kv i8, Q14 trim."""
    home = tmp_path / "home"
    (home / "artifacts" / "ane_models").mkdir(parents=True)
    (home / "artifacts" / "weights").mkdir(parents=True)
    (home / "build").mkdir(parents=True)
    (home / "build" / "libgemma4_ane_prefix.dylib").write_bytes(b"stub")
    (home / "artifacts" / "ane_models" / "gemma4_early14_prefix_S256_fp16.mlmodelc").mkdir()

    default_dir = home / "artifacts" / "hf_model"
    default_dir.mkdir(parents=True)
    (default_dir / "config.json").write_text("{}", encoding="utf-8")
    (default_dir / "model.safetensors").write_bytes(b"m")

    bf16_dir = tmp_path / "bf16_custom"
    bf16_dir.mkdir()
    (bf16_dir / "config.json").write_text("{}", encoding="utf-8")

    fused_ckpt_dir = tmp_path / "fused_lora"
    fused_ckpt_dir.mkdir()
    (fused_ckpt_dir / "config.json").write_text("{}", encoding="utf-8")

    p = resolve_paths(home)
    # 1a. Default dir uses artifacts/ane_models base bundle
    assert p.resolve_ane_bundle("fp16", model_dir=None) == p.ane_fp16_bundle
    assert p.resolve_ane_bundle("fp16", model_dir=default_dir) == p.ane_fp16_bundle
    # 1b. Overridden unbundled dir (`e2b-it-bf16`) NEVER picks up base E2B-4bit bundle!
    assert p.resolve_ane_bundle("fp16", model_dir=bf16_dir) is None
    # 1c. Checkpoint-scoped custom bundle (e.g. fused LoRA checkpoint) resolves cleanly!
    fused_bundle_dir = p.ane_models_dir / p.checkpoint_ane_key(fused_ckpt_dir) / "gemma4_early14_prefix_S256_fp16.mlmodelc"
    fused_bundle_dir.mkdir(parents=True)
    assert p.resolve_ane_bundle("fp16", model_dir=fused_ckpt_dir) == fused_bundle_dir

    # 2. ingest_ane_chunk_kv with kv_i8=True quantizes FP16 ANE KV and scatters SWA wrap + Full
    fake_mx = _FakeMx()
    m = FastGemma4Model.__new__(FastGemma4Model)
    m.act_dtype = fake_mx.bfloat16
    m.swa_layers = [0]
    m.full_layers = [1]
    m.first_kv_shared = 2
    m.hidden_size = 8
    m.ple_dim = 4
    cache_i8 = {
        "kv_i8": True,
        "layers": [
            {
                "k": fake_mx.zeros((2, 512, 4), dtype=fake_mx.int8),
                "k_s": fake_mx.zeros((2, 512), dtype=fake_mx.float16),
                "v": fake_mx.zeros((2, 512, 4), dtype=fake_mx.int8),
                "v_s": fake_mx.zeros((2, 512), dtype=fake_mx.float16),
                "cap": 512,
                "ring": True,
            },
            {
                "k": fake_mx.zeros((2, 768, 8), dtype=fake_mx.int8),
                "k_s": fake_mx.zeros((2, 768), dtype=fake_mx.float16),
                "v": fake_mx.zeros((2, 768, 8), dtype=fake_mx.int8),
                "v_s": fake_mx.zeros((2, 768), dtype=fake_mx.float16),
                "cap": 768,
                "ring": False,
            },
        ],
    }
    quant_shapes: list[tuple[int, int, int]] = []

    def _q_ane(k_in: _FakeArray, v_in: _FakeArray):
        quant_shapes.append(k_in.shape)
        B_i, L_i, D_i = k_in.shape
        return (
            _FakeArray(np.full((B_i, L_i, D_i), 7, dtype=np.int8), dtype=fake_mx.int8),
            _FakeArray(k_in.data[:, :, 0].astype(np.float16), dtype=fake_mx.float16),
            _FakeArray(np.full((B_i, L_i, D_i), -3, dtype=np.int8), dtype=fake_mx.int8),
            _FakeArray(v_in.data[:, :, 0].astype(np.float16), dtype=fake_mx.float16),
        )

    k_swa_in = np.full((1, 1, 10, 4), 42.0, dtype=np.float16)
    v_swa_in = np.full((1, 1, 10, 4), 11.0, dtype=np.float16)
    k_full_in = np.full((1, 1, 10, 8), 99.0, dtype=np.float16)
    v_full_in = np.full((1, 1, 10, 8), 55.0, dtype=np.float16)
    with (
        patch.object(fg4_model, "mx", fake_mx),
        patch.object(fg4_model, "quantize_kv_pair_i8", side_effect=_q_ane),
    ):
        # Cross 512 wrap on SWA (priv_off=508, valid_len=6 -> slots 508,509,510,511,0,1)
        m.ingest_ane_chunk_kv(cache_i8, [1], 508, k_swa_in, v_swa_in, k_full_in, v_full_in, valid_len=6, eval_cache=True)
    assert quant_shapes == [(1, 6, 4), (1, 6, 8)]
    assert float(cache_i8["layers"][0]["k_s"].data[1, 0]) == 42.0
    assert int(cache_i8["layers"][0]["k"].data[1, 511, 0]) == 7
    assert float(cache_i8["layers"][1]["v_s"].data[1, 513]) == 55.0

    # 3. export_early14_f16 emits `l14_kv` (K14+V14 only, 2*D_full channels), omitting `l14_qkv`
    class _FakeMod:
        def __init__(self, out_c: int, in_c: int) -> None:
            self.weight = _FakeArray(np.zeros((out_c, in_c), dtype=np.float16), dtype=fake_mx.float16)

    m.per_layer_model_projection = _FakeMod(8, 8)
    m.per_layer_projection_scale = type("_S", (), {"item": lambda s: 1.0})()
    m.per_layer_projection_norm_w = _FakeArray(np.ones((4,), dtype=np.float16), dtype=fake_mx.float16)
    m.per_layer_input_scale = type("_S", (), {"item": lambda s: 1.0})()
    m.layers = [
        {
            "head_dim": 4,
            "w_in": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "qkv_proj": _FakeMod(40, 8),
            "w_kn": _FakeArray(np.ones((4,), dtype=np.float16), dtype=fake_mx.float16),
            "w_qn": _FakeArray(np.ones((4,), dtype=np.float16), dtype=fake_mx.float16),
            "o_proj": _FakeMod(8, 32),
            "w_post_attn": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "w_pre_ff": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "gate_up_proj": _FakeMod(16, 8),
            "down_proj": _FakeMod(8, 8),
            "w_post_ff": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "ple_gate": _FakeMod(4, 8),
            "ple_proj": _FakeMod(8, 4),
            "w_post_ple": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "layer_scalar": _FakeArray([1.0], dtype=fake_mx.float32),
        },
        {
            "head_dim": 8,
            "w_in": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
            "qkv_proj": _FakeMod(80, 8),
            "kv_proj_only": _FakeMod(16, 8),
            "w_kn": _FakeArray(np.ones((8,), dtype=np.float16), dtype=fake_mx.float16),
        },
    ]
    with patch.object(fg4_model, "mx", fake_mx):
        npz_path = m.export_early14_f16(tmp_path / "e14.npz")
    with np.load(npz_path) as data:
        assert "l0_qkv" in data
        assert "l1_kv" in data and data["l1_kv"].shape == (16, 8, 1, 1)
        assert "l1_qkv" not in data


def test_mixed_8bit_g64_non_mlp_bf16_mlp_vocab150786_loader_cpu(tmp_path: Path) -> None:
    """Verify Deliverable #6: 8-bit g64 non-MLP + BF16 MLP + 150,786 vocab remap loader support on CPU."""
    assert DEFAULT_KEPT_VOCAB_SIZE == 150786
    assert FULL_VOCAB_SIZE == 262144

    # 1. Verify real VocabRemapper if artifacts tokenizer is present locally or via sibling resolve_paths, else synthetic
    from fast_gemma4.paths import resolve_paths

    repo_root = Path(__file__).resolve().parents[1]
    real_tok = repo_root / "artifacts" / "tokenizer" / "tokenizer.json"
    real_remap = repo_root / "artifacts" / "vocab" / "vocab_remap_150786.npz"
    if not real_tok.exists():
        try:
            paths_r = resolve_paths()
            cand_tok = paths_r.resolve_tokenizer_json(local_files_only=True)
            if cand_tok.exists():
                real_tok = cand_tok
            if not real_remap.exists() and paths_r.vocab_remap_npz.exists():
                real_remap = paths_r.vocab_remap_npz
        except FileNotFoundError:
            pass
    if real_tok.exists():
        tok_path = real_tok
        rem = VocabRemapper.load_or_build(tok_path, real_remap if real_remap.exists() else None, target_size=150786)
        assert rem.vocab_size == 150786
        assert rem.old_to_new.shape == (262144,)
        ft = FastGemma4Tokenizer(tok_path)
        sample_json = '{"status":"usable","primary_kind":"technical_report","year":2025}'
        ids = ft.encode(sample_json, add_special_tokens=False)
        new_ids, misses = rem.encode_remap(ids)
        assert misses == 0
        assert ft.decode(rem.decode_to_old(new_ids), skip_special_tokens=False) == sample_json
    else:
        # Synthetic tokenizer JSON fallback
        tok_path = tmp_path / "tokenizer.json"
        _build_tiny_json_tokenizer(tmp_path)

    # 2. Pure-CPU FastGemma4Model loader verification for 8-bit g64 non-MLP + BF16 MLP + 150,786 remap
    H_DIM, D_SWA, D_FULL, PLE_D = 128, 64, 64, 64
    N_LAYERS, N_SHARED = 16, 1  # 15 KV-owning (0..14) + 1 KV-shared (15)
    MLP_H = 128

    model_dir = tmp_path / "mixed_8bit_g64_bf16_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "text_config": {
            "num_hidden_layers": N_LAYERS,
            "hidden_size": H_DIM,
            "num_kv_shared_layers": N_SHARED,
            "layer_types": ["sliding_attention"] * 14 + ["full_attention", "sliding_attention"],
            "sliding_window": 512,
            "head_dim": D_SWA,
            "global_head_dim": D_FULL,
            "hidden_size_per_layer_input": PLE_D,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
        },
        "quantization": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "language_model.model.layers.0.mlp.gate_proj": False,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"mock")
    # Build synthetic kept_old_ids npz of length 150786 (or smaller if synthetic tokenizer)
    kept_len = 150786 if real_tok.exists() else 16
    remap_npz = tmp_path / "vocab_remap_150786.npz"
    kept_arr = np.arange(kept_len, dtype=np.int32)
    np.savez(remap_npz, kept_old_ids=kept_arr)

    fake_mx = _FakeMx()

    class _FakeLinear:
        def __init__(self, in_f: int, out_f: int, bias: bool = False) -> None:
            self.in_features = in_f
            self.out_features = out_f
            self.weight = None

        def parameters(self) -> list[Any]:
            return [self.weight]

    class _FakeQLinear:
        def __init__(self, in_f: int, out_f: int, bias: bool = False, group_size: int = 64, bits: int = 4, mode: str = "affine") -> None:
            self.in_features = in_f
            self.out_features = out_f
            self.group_size = group_size
            self.bits = bits
            self.mode = mode
            self.weight = None
            self.scales = None
            self.biases = None

        def parameters(self) -> list[Any]:
            return [self.weight, self.scales, self.biases]

    class _FakeQEmbedding:
        def __init__(self, num_emb: int, dims: int, group_size: int = 64, bits: int = 4, mode: str = "affine") -> None:
            self.num_embeddings = num_emb
            self.dims = dims
            self.group_size = group_size
            self.bits = bits
            self.mode = mode
            self.weight = None
            self.scales = None
            self.biases = None

        def parameters(self) -> list[Any]:
            return [self.weight, self.scales, self.biases]

    class _FakeNN:
        Linear = _FakeLinear
        QuantizedLinear = _FakeQLinear
        QuantizedEmbedding = _FakeQEmbedding
        Embedding = _FakeLinear

    def _q8_tensors(prefix: str, out_f: int, in_f: int) -> dict[str, _FakeArray]:
        return {
            f"{prefix}.weight": _FakeArray(np.zeros((out_f, in_f // 4), dtype=np.uint32), dtype="uint32"),
            f"{prefix}.scales": _FakeArray(np.ones((out_f, in_f // 64), dtype=np.float16), dtype=fake_mx.bfloat16),
            f"{prefix}.biases": _FakeArray(np.zeros((out_f, in_f // 64), dtype=np.float16), dtype=fake_mx.bfloat16),
        }

    def _bf16_tensors(prefix: str, out_f: int, in_f: int) -> dict[str, _FakeArray]:
        return {f"{prefix}.weight": _FakeArray(np.zeros((out_f, in_f), dtype=np.float32), dtype=fake_mx.bfloat16)}

    V_RAW = 262144 if real_tok.exists() else 64
    raw_weights: dict[str, _FakeArray] = {}
    raw_weights.update(_q8_tensors("language_model.model.embed_tokens", V_RAW, H_DIM))
    raw_weights.update(_q8_tensors("language_model.model.embed_tokens_per_layer", V_RAW, N_LAYERS * PLE_D))
    raw_weights.update(_q8_tensors("language_model.model.per_layer_model_projection", N_LAYERS * PLE_D, H_DIM))
    raw_weights["language_model.model.per_layer_projection_norm.weight"] = _FakeArray(np.ones((PLE_D,), dtype=np.float32), dtype=fake_mx.bfloat16)
    raw_weights["language_model.model.norm.weight"] = _FakeArray(np.ones((H_DIM,), dtype=np.float32), dtype=fake_mx.bfloat16)

    for i in range(N_LAYERS):
        lp = f"language_model.model.layers.{i}"
        d_h = D_FULL if i == 14 else D_SWA
        raw_weights.update(_q8_tensors(f"{lp}.self_attn.q_proj", 8 * d_h, H_DIM))
        if i < N_LAYERS - N_SHARED:
            raw_weights.update(_q8_tensors(f"{lp}.self_attn.k_proj", d_h, H_DIM))
            raw_weights.update(_q8_tensors(f"{lp}.self_attn.v_proj", d_h, H_DIM))
            raw_weights[f"{lp}.self_attn.k_norm.weight"] = _FakeArray(np.ones((d_h,), dtype=np.float32), dtype=fake_mx.bfloat16)
        raw_weights.update(_q8_tensors(f"{lp}.self_attn.o_proj", H_DIM, 8 * d_h))
        # MLPs stay native BF16 (no .scales/.biases)!
        raw_weights.update(_bf16_tensors(f"{lp}.mlp.gate_proj", MLP_H, H_DIM))
        raw_weights.update(_bf16_tensors(f"{lp}.mlp.up_proj", MLP_H, H_DIM))
        raw_weights.update(_bf16_tensors(f"{lp}.mlp.down_proj", H_DIM, MLP_H))
        raw_weights.update(_q8_tensors(f"{lp}.per_layer_input_gate", PLE_D, H_DIM))
        raw_weights.update(_q8_tensors(f"{lp}.per_layer_projection", H_DIM, PLE_D))
        raw_weights[f"{lp}.self_attn.q_norm.weight"] = _FakeArray(np.ones((d_h,), dtype=np.float32), dtype=fake_mx.bfloat16)
        for nw in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"):
            raw_weights[f"{lp}.{nw}.weight"] = _FakeArray(np.ones((H_DIM,), dtype=np.float32), dtype=fake_mx.bfloat16)
        raw_weights[f"{lp}.layer_scalar"] = _FakeArray([1.0], dtype=fake_mx.bfloat16)

    fake_mx.load = lambda path: raw_weights  # type: ignore[attr-defined]
    fake_mx.power = lambda a, b: _FakeArray(np.power(a, b.data), dtype=fake_mx.float32)  # type: ignore[attr-defined]
    paths_obj = resolve_paths(tmp_path)
    # Point vocab_remap_npz and tokenizer_dir to our test files
    from dataclasses import replace

    (tmp_path / "artifacts" / "tokenizer").mkdir(parents=True, exist_ok=True)
    target_tok = tmp_path / "artifacts" / "tokenizer" / "tokenizer.json"
    if real_tok.exists():
        target_tok.symlink_to(real_tok)
    else:
        target_tok.symlink_to(tok_path)
    paths_obj = replace(paths_obj, vocab_remap_npz=remap_npz)

    with (
        patch.object(fg4_model, "mx", fake_mx),
        patch.object(fg4_model, "nn", _FakeNN),
        patch.object(fg4_model, "tree_flatten", side_effect=lambda x: []),
    ):
        loaded = FastGemma4Model(
            paths=paths_obj,
            model_dir=model_dir,
            mlp_dtype="bfloat16",
            vocab_out=kept_len,
            use_mpp=True,
            lm_head_mode="q4",
            local_files_only=True,
        )

    assert loaded.use_mpp is True
    assert loaded.vocab_size == kept_len
    assert isinstance(loaded.embed_tokens, _FakeQEmbedding)
    assert loaded.embed_tokens.bits == 8 and loaded.embed_tokens.group_size == 64
    assert isinstance(loaded.lm_head, _FakeQLinear)
    assert loaded.lm_head.bits == 8 and loaded.lm_head.group_size == 64
    assert loaded.lm_head.out_features == kept_len
    assert isinstance(loaded.embed_tokens_per_layer, _FakeQEmbedding)
    assert loaded.embed_tokens_per_layer.bits == 8 and loaded.embed_tokens_per_layer.group_size == 64
    # Early14 PLE slicing: 14 * 64 = 896 -> 14 groups of 64, 224 uint32 cols (with real PLE_D=256: 56 groups, 896 cols)
    assert loaded._etpl_14_groups == (14 * PLE_D) // 64
    assert loaded._etpl_14_wcols == (14 * PLE_D * 8) // 32
    assert isinstance(loaded.pli_proj_14, _FakeQLinear) and loaded.pli_proj_14.bits == 8
    assert isinstance(loaded.layers[0]["qkv_proj"], _FakeQLinear) and loaded.layers[0]["qkv_proj"].bits == 8
    assert isinstance(loaded.layers[0]["o_proj"], _FakeQLinear) and loaded.layers[0]["o_proj"].bits == 8
    assert isinstance(loaded.layers[14]["kv_proj_only"], _FakeQLinear) and loaded.layers[14]["kv_proj_only"].bits == 8
    assert isinstance(loaded.layers[0]["gate_up_proj"], _FakeLinear)
    assert isinstance(loaded.layers[0]["down_proj"], _FakeLinear)
    assert "w_g" in loaded.layers[0] and "w_u" in loaded.layers[0] and "w_dn" in loaded.layers[0]


def test_prefill_slot_groups_bound_length_spread_and_forward_start_cpu() -> None:
    """A padded prefill group may not mix lengths: pad KVs wrap in the SWA ring and overwrite a short doc's tail."""
    from fast_gemma4.model import FastGemma4Model

    calls: list[tuple[list[int], list[int], int]] = []

    class _Recorder:
        def _prefill_early14_slot_group(self, sub_toks, slot_indices, cache, chunk_sz=256, start_s0=0):
            calls.append(([len(s) for s in sub_toks], list(slot_indices), start_s0))

    lens = [800, 2400, 810, 1500, 1540, 2390, 830]
    FastGemma4Model._prefill_early14_slot_groups(
        _Recorder(), [[1] * n for n in lens], list(range(len(lens))), {}, group_size=16, start_s0=768
    )
    assert sorted(slot for _, slots, _ in calls for slot in slots) == list(range(len(lens)))
    assert all(max(group) - min(group) <= 48 for group, _, _ in calls)
    assert all(start == 768 for _, _, start in calls)
    assert len(calls) == 3

    # The ANE cohort tail goes through the grouped entry point, not one batch per cohort.
    src = (REPO_ROOT / "src" / "fast_gemma4" / "speculative.py").read_text(encoding="utf-8")
    admit = src[src.index("def _admit_ane_cohort") : src.index("def _pull_ready_ane")]
    assert "_prefill_early14_slot_groups(" in admit and "_prefill_early14_slot_group(" not in admit


def test_ane_ring_variant_selects_the_ring_bundle_cpu() -> None:
    """`--ane-variant fp16` must put the ring on the FP16 bundle; it used to resolve Linear-W8 whatever was asked."""
    with pytest.raises(ValueError, match="ane_ring_variant"):
        FastGemma4Engine(ane_ring_variant="lut8")  # rejected before any model or bundle is loaded

    class _Paths:
        lib_ane_prefix = Path("/nonexistent/libgemma4_ane_prefix.dylib")

        @staticmethod
        def resolve_ane_bundle(variant: str, *, model_dir: Any, local_files_only: bool) -> Path:
            return Path(f"/nonexistent/{Path(model_dir).name}/early14_{variant}.mlmodelc")

    class _Runner:
        def __init__(self, lib_path: Path, bundle: Path, **_kw: Any) -> None:
            self.bundle = bundle

    def _engine(variant: str | None) -> FastGemma4Engine:
        eng = FastGemma4Engine.__new__(FastGemma4Engine)
        eng.paths = _Paths()  # type: ignore[assignment]
        eng.model = type("_M", (), {"model_dir": Path("/nonexistent/ckpt")})()  # type: ignore[assignment]
        if variant is not None:
            eng.ane_ring_variant = variant
        return eng

    with patch.object(fg4_engine, "AnePrefixRunner", _Runner), patch.object(
        fg4_engine, "is_ane_resident_bundle", lambda lib, bundle: True
    ):
        for variant, expect in ((None, "fp16"), ("w8", "w8"), ("fp16", "fp16")):
            eng = _engine(variant)
            assert eng.ane_ring_bundle is None
            if expect == "w8":
                with pytest.warns(UserWarning, match="Linear-W8"):
                    runner = eng._resolve_ane_ring_runner(require=True)
            else:
                runner = eng._resolve_ane_ring_runner(require=True)
            assert runner.bundle.name == f"early14_{expect}.mlmodelc"  # type: ignore[union-attr]
            assert eng.ane_ring_bundle == runner.bundle  # type: ignore[union-attr]
            if expect == "w8":
                with pytest.warns(UserWarning, match="Linear-W8"):
                    assert eng._resolve_ane_ring_runner(require=False) is runner
            else:
                assert eng._resolve_ane_ring_runner(require=False) is runner

    # A bundle that is missing or not fully on the ANE: `auto` falls back to GPU only and says so through
    # `ane_ring_bundle`; an explicit ring request fails loudly and names the variant.
    with patch.object(fg4_engine, "is_ane_resident_bundle", lambda lib, bundle: False):
        eng = _engine("fp16")
        assert eng._resolve_ane_ring_runner(require=False) is None and eng.ane_ring_bundle is None
        with pytest.raises(RuntimeError, match="fp16"):
            eng._resolve_ane_ring_runner(require=True)


def test_prefill_ring_chunk_never_writes_pad_positions_cpu() -> None:
    """A right-padded prefill group must not write pads into a ring layer: they wrap onto a shorter doc's window."""
    cap, dim = 512, 2
    pre = [600, 640, 648]  # within 48 tokens of each other, so the grouping puts them in one padded batch
    docs = np.arange(len(pre), dtype=np.int32)

    def prefill(skip_pads: bool) -> np.ndarray:
        c = {"k": np.zeros((len(pre), cap, dim), dtype=np.float32)}
        for s0 in range(0, max(pre), 256):
            s1 = min(max(pre), s0 + 256)
            pos = np.arange(s0, s1)
            is_real = pos[None, :] < np.asarray(pre)[:, None]
            val = np.where(is_real, 1000.0 * (docs[:, None] + 1) + pos[None, :], -1.0)  # pads carry -1
            real = FastGemma4Model._real_token_pairs(pre, s0, s1) if skip_pads else None
            assert (real is None) == (not skip_pads or bool(is_real.all()))
            FastGemma4Model._scatter_ring_chunk(
                c, docs, pos & (cap - 1), {"k": np.repeat(val[..., None], dim, axis=-1).astype(np.float32)}, real
            )
        return c["k"]

    def lost_window_positions(ring: np.ndarray) -> list[int]:
        out = []
        for d, n in enumerate(pre):
            window = np.arange(n - (cap - 1), n)  # the positions the next query, at position n, attends
            out.append(int((ring[d, window & (cap - 1), 0] != 1000.0 * (d + 1) + window).sum()))
        return out

    assert lost_window_positions(prefill(skip_pads=True)) == [0, 0, 0]
    # What the unguarded write did: each doc lost one window position per pad after its first.
    assert lost_window_positions(prefill(skip_pads=False)) == [47, 7, 0]

    # A doc that ended before the chunk starts contributes no rows at all.
    rows, cols = FastGemma4Model._real_token_pairs([500, 540], 512, 540)  # type: ignore[misc]
    assert set(rows.tolist()) == {1} and cols.tolist() == list(range(28))

    # Both ring write sites of the slot prefill go through the guarded helper.
    src = (REPO_ROOT / "src" / "fast_gemma4" / "model.py").read_text(encoding="utf-8")
    body = src[src.index("def _prefill_early14_slot_group(") : src.index("def _prefill_early14_slot_groups(")]
    assert body.count("self._scatter_ring_chunk(") == 2 and "slots[None, :]] =" not in body


def test_ane_ring_chunk_count_and_previous_chunk_bookkeeping_cpu() -> None:
    """The ring runs as many chunks as the bundle's full-attention past holds, each fed every earlier chunk."""
    import queue

    from fast_gemma4.ane_pipeline import (
        ANE_CHUNK_S,
        ANE_P_FULL,
        ANE_P_SWA,
        AsyncHybridPrefillRing,
        ane_ring_chunk_count,
        make_chunk_geom,
    )

    # Chunk k starts at lcp + 256k and needs that many past positions out of 1,920.
    counts = {lcp: ane_ring_chunk_count(lcp) for lcp in (0, 278, 384, 385, 1408, 1409, 1664, 1920, 1921)}
    assert counts == {0: 8, 278: 7, 384: 7, 385: 6, 1408: 3, 1409: 2, 1664: 2, 1920: 1, 1921: 0}

    lcp = 278
    n_chunks = ane_ring_chunk_count(lcp)
    for k in range(n_chunks):
        off = lcp + k * ANE_CHUNK_S
        assert off <= ANE_P_FULL
        *_, mask_swa, mask_full = make_chunk_geom(off)
        t = np.arange(ANE_CHUNK_S)
        # Full: all `off` earlier positions plus the causal part of this chunk; never a slot past `off`.
        assert ((mask_full[:, :ANE_P_FULL] == 0).sum(1) == off).all() and (mask_full[:, off:ANE_P_FULL] != 0).all()
        assert ((mask_full[:, ANE_P_FULL:] == 0).sum(1) == t + 1).all()
        # Sliding window: the last 512 positions up to and including the query.
        assert ((mask_swa == 0).sum(1) == np.minimum(ANE_P_SWA, off + t + 1)).all()

    class _Runner:
        def __init__(self, out_ring_size: int) -> None:
            self.out_ring_size = out_ring_size
            self.calls: list[tuple[int, Any, list[int], bool]] = []
            self.scattered: tuple[list[int], list[int], list[int], list[int]] | None = None

        def run_into_ring(self, B, chunk_idx, lcp_, h_in, pli_raw, geom, swa_k, swa_v, kf, vf, prev, out, *, pipe_depth=4):
            assert h_in.shape[0] == B == len(out) and lcp_ == lcp
            self.calls.append((chunk_idx, None if prev is None else np.array(prev), list(out), swa_k is None))
            return 0.01

        def scatter_ring_kv_i8(self, binding, ring_ids, doc_slots, offs, vlens):
            self.scattered = (list(ring_ids), list(doc_slots), list(offs), list(vlens))
            return 0.0

    def _ring(out_ring_size: int, *, min_chunk_tokens: int = 1) -> AsyncHybridPrefillRing:
        ring = AsyncHybridPrefillRing.__new__(AsyncHybridPrefillRing)
        ring.runner = _Runner(out_ring_size)  # type: ignore[assignment]
        ring.lcp, ring.n_chunks = lcp, n_chunks
        ring.min_chunk_tokens = min_chunk_tokens
        ring.geoms = {k: make_chunk_geom(lcp + k * ANE_CHUNK_S) for k in range(n_chunks)}
        ring.w_emb = np.zeros((8, 4), dtype=np.float16)
        ring.w_pli14 = np.zeros((8, 6), dtype=np.float16)
        ring.sys_ks_512 = ring.sys_vs_512 = ring.sys_ks_256 = ring.sys_vs_256 = np.zeros(1, dtype=np.float16)
        ring.sys_kf = ring.sys_vf = None
        ring._free_ring_q = queue.Queue()
        for r in range(out_ring_size):
            ring._free_ring_q.put(r)
        return ring

    lens = [60, 300, 769, 1300, 1793, 2500]  # suffix tokens; the last one is fed by LastTok, not prefilled
    sufs = [[1 + (i % 7)] * n for i, n in enumerate(lens)]
    doc_chunks = [1, 2, 3, 6, 7, 7]

    # Default min_chunk_tokens=64 spills short tails (<64 toks: 59, 43, 19) cleanly to GPU instead of wasting S=256 buckets!
    ring_default = _ring(96, min_chunk_tokens=AsyncHybridPrefillRing.min_chunk_tokens)
    assert AsyncHybridPrefillRing.min_chunk_tokens == 64
    item_def = ring_default.prefill_cohort_into_ring(list(range(len(lens))), sufs)
    assert item_def["covered"] == [0, 256, 768, 1280, 1792, 1792]
    ring_default.scatter_cohort_to_slot_cache(object(), item_def, [10, 11, 12, 13, 14, 15])  # type: ignore[arg-type]
    assert ring_default._free_ring_q.qsize() == 96

    # With min_chunk_tokens=1, verify exact partial-chunk previous-chunk bookkeeping across all 7 chunks:
    ring = _ring(96, min_chunk_tokens=1)
    item = ring.prefill_cohort_into_ring(list(range(len(lens))), sufs)
    assert item["covered"] == [59, 299, 768, 1299, 1792, 1792]
    calls = ring.runner.calls  # type: ignore[attr-defined]
    assert [c[0] for c in calls] == list(range(n_chunks))
    slot_of: list[dict[int, int]] = []
    for k, prev, out, no_sys_swa in calls:
        act = [d for d, n in enumerate(doc_chunks) if n > k]
        assert len(out) == len(act) and no_sys_swa == (k >= 2)
        if k == 0:
            assert prev is None
        else:
            assert prev.shape == (k, len(act)) and prev.dtype == np.int32
            for j in range(k):  # row j is each active doc's own chunk j, in the same doc order as this call
                assert prev[j].tolist() == [slot_of[j][d] for d in act]
        slot_of.append(dict(zip(act, out)))
    held = [r for _, _, out, _ in calls for r in out]
    assert len(held) == len(set(held)) == sum(doc_chunks) == len(item["ring_slots_held"])
    assert ring._free_ring_q.qsize() == 96 - sum(doc_chunks)

    ring.scatter_cohort_to_slot_cache(object(), item, [10, 11, 12, 13, 14, 15])  # type: ignore[arg-type]
    ring_ids, doc_slots, offs, vlens = ring.runner.scattered  # type: ignore[attr-defined,misc]
    assert ring_ids == held and ring._free_ring_q.qsize() == 96
    by_doc = {s: sorted((o, v) for o, v, ds in zip(offs, vlens, doc_slots) if ds == s) for s in set(doc_slots)}
    assert by_doc[10] == [(0, 59)] and by_doc[12] == [(0, 256), (256, 256), (512, 256)]
    assert by_doc[13][-1] == (1280, 19) and by_doc[15] == [(256 * k, 256) for k in range(7)]

    # A cohort that cannot fit the ring fails at once and leaks no slot: waiting would never return, because only
    # scattering this same cohort frees slots.
    small = _ring(20, min_chunk_tokens=1)
    with pytest.raises(ValueError, match="26 ring slots"):
        small.prefill_cohort_into_ring(list(range(len(lens))), sufs)
    assert small._free_ring_q.qsize() == 20 and small.runner.calls == []  # type: ignore[attr-defined]


def test_mtp_drafter_apple10_nax_vs_apple9_gating_and_cache_cpu() -> None:
    """Verify Apple10 NAX (`B == 1`) vs Apple9 (`1 <= B <= 4`) MTP gating, `K` selection, and cached host probe."""
    import fast_gemma4.mtp_drafter as fg4_mtp
    import fast_shared.hw_check as fs_hw

    sentinel = object()

    # 1. Explicit Apple10 NAX (M5+): strictly B == 1; B=2..4 disabled so NAX TensorOps accelerate M >= 8 directly.
    assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 1, 0.0, apple10_nax=True) is True
    assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 1, 0.0, speculative=True, apple10_nax=True) is True
    assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 1, 0.0, speculative=False, apple10_nax=True) is False
    assert fg4_mtp.should_use_mtp_drafter(sentinel, True, 1, 0.0, apple10_nax=True) is False
    assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 1, 0.1, apple10_nax=True) is False
    assert fg4_mtp.should_use_mtp_drafter(None, False, 1, 0.0, apple10_nax=True) is False
    for b in (0, 2, 3, 4, 8):
        assert fg4_mtp.should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=True) is False
        assert fg4_mtp.should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=True, apple10_nax=True) is False

    assert fg4_mtp.recommended_mtp_draft_k(1, has_grammar=False, apple10_nax=True) == 3
    assert fg4_mtp.recommended_mtp_draft_k(1, has_grammar=True, apple10_nax=True) == 3
    for bad_b in (0, 2, 3, 4, 8):
        with pytest.raises(ValueError, match="strictly gated"):
            fg4_mtp.recommended_mtp_draft_k(bad_b, apple10_nax=True)

    # 2. Explicit Apple9 (M4): preserve 1 <= B <= 4 (K=3/4 at B=1, K=2 at B=2..4), B >= 5 gated out.
    for b in (1, 2, 3, 4):
        assert fg4_mtp.should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False) is True
    for b in (0, 5, 8, 16):
        assert fg4_mtp.should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False) is False

    assert fg4_mtp.recommended_mtp_draft_k(1, has_grammar=False, apple10_nax=False) == 4
    assert fg4_mtp.recommended_mtp_draft_k(1, has_grammar=True, apple10_nax=False) == 3
    for b in (2, 3, 4):
        assert fg4_mtp.recommended_mtp_draft_k(b, has_grammar=False, apple10_nax=False) == 2
        assert fg4_mtp.recommended_mtp_draft_k(b, has_grammar=True, apple10_nax=False) == 2
    for bad_b in (0, 5, 8):
        with pytest.raises(ValueError, match="strictly gated"):
            fg4_mtp.recommended_mtp_draft_k(bad_b, apple10_nax=False)

    # 3. Cached host probe (`apple10_nax=None`): probes once per process with safe False fallback.
    prev_cache = fg4_mtp._APPLE10_NAX_CACHE
    try:
        m5_calls = 0

        def _probe_m5() -> dict[str, Any]:
            nonlocal m5_calls
            m5_calls += 1
            return {
                "system": "Darwin",
                "machine": "arm64",
                "brand": "Apple M5",
                "gpu_family": "Apple10",
                "supports_apple10": True,
                "macos_version": "26.6",
            }

        fg4_mtp._APPLE10_NAX_CACHE = None
        with patch.object(fs_hw, "probe_host_hardware", side_effect=_probe_m5):
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 1, 0.0) is True
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 2, 0.0) is False
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 4, 0.0) is False
            assert fg4_mtp.recommended_mtp_draft_k(1, has_grammar=True) == 3
            with pytest.raises(ValueError, match="strictly gated"):
                fg4_mtp.recommended_mtp_draft_k(2)
        assert m5_calls == 1

        m4_calls = 0

        def _probe_m4() -> dict[str, Any]:
            nonlocal m4_calls
            m4_calls += 1
            return {
                "system": "Darwin",
                "machine": "arm64",
                "brand": "Apple M4 Pro",
                "gpu_family": "Apple9",
                "supports_apple10": False,
                "macos_version": "26.6",
            }

        fg4_mtp._APPLE10_NAX_CACHE = None
        with patch.object(fs_hw, "probe_host_hardware", side_effect=_probe_m4):
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 4, 0.0) is True
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 8, 0.0) is False
            assert fg4_mtp.recommended_mtp_draft_k(3, has_grammar=False) == 2
        assert m4_calls == 1

        fg4_mtp._APPLE10_NAX_CACHE = None
        with patch.object(fs_hw, "probe_host_hardware", side_effect=RuntimeError("probe failed")):
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 4, 0.0) is True
            assert fg4_mtp.should_use_mtp_drafter(sentinel, False, 8, 0.0) is False
            assert fg4_mtp.recommended_mtp_draft_k(2) == 2
    finally:
        fg4_mtp._APPLE10_NAX_CACHE = prev_cache


def test_ane_ring_variant_auto_dynamic_silicon_policy_cpu(tmp_path: Path) -> None:
    """Verify default `ane_ring_variant='auto'` on 24 GB (calibrated w8 vs uncalibrated/missing -> fp16) vs 48 GB."""
    import fast_shared.hw_check as fs_hw

    assert inspect.signature(FastGemma4Engine.__init__).parameters["ane_ring_variant"].default == "auto"

    ane_dir = tmp_path / "ane_models"
    ane_dir.mkdir()
    fp16_bundle = ane_dir / "gemma4_early14_prefix_S256_fp16.mlmodelc"
    w8_bundle = ane_dir / "gemma4_early14_prefix_S256_w8.mlmodelc"
    fp16_bundle.mkdir()
    w8_bundle.mkdir()
    dylib_stub = tmp_path / "libgemma4_ane_prefix.dylib"
    dylib_stub.write_bytes(b"stub")

    manifest_path = ane_dir / "ane_calibration_manifest.json"

    def _write_manifest(*, calibrated: bool) -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "bundles": {
                        w8_bundle.name: {
                            "awq_mse": True,
                            "calibration_source": "domain_corpus" if calibrated else "synthetic",
                            "calibration_docs": 16,
                            "qat_validated": False,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    class _Paths:
        def __init__(self, *, has_w8: bool = True) -> None:
            self.lib_ane_prefix = dylib_stub
            self.has_w8 = has_w8

        def resolve_ane_bundle(
            self,
            variant: str,
            *,
            model_dir: Any = None,
            local_files_only: bool = True,
            exit_layer: int | None = None,
        ) -> Path | None:
            if variant == "w8":
                return w8_bundle if self.has_w8 else None
            if variant == "fp16":
                return fp16_bundle
            return None

    class _DummyModel:
        hidden_size = 1536
        num_layers = 35
        n_kv_heads = 1
        first_kv_shared = 15
        model_dir = tmp_path / "ckpt"
        tokenizer_json = tmp_path / "tokenizer.json"
        kv_i8 = True

    class _Runner:
        def __init__(self, lib_path: Path, bundle: Path, **_kw: Any) -> None:
            self.bundle = bundle

    fake_mx = _FakeMx()
    hw_24gb = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M5",
        "gpu_family": "Apple10",
        "supports_apple10": True,
        "mem_gb": 24,
        "macos_version": "26.6",
    }
    hw_48gb = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M4 Pro",
        "gpu_family": "Apple9",
        "supports_apple10": False,
        "mem_gb": 48,
        "macos_version": "26.6",
    }

    def _build_engine(hw_dict: dict[str, Any], *, has_w8: bool = True) -> FastGemma4Engine:
        with (
            patch.object(fs_hw, "probe_host_hardware", return_value=hw_dict),
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "_host_ram_gib", return_value=float(hw_dict.get("mem_gb", 0))),
            patch.object(fg4_engine, "resolve_paths", return_value=_Paths(has_w8=has_w8)),
            patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel()),
            patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=object()),
        ):
            return FastGemma4Engine(warmup=False)

    # 1. 24 GB + calibrated W8 bundle -> selects "w8" cleanly via DynamicSiliconPolicy.
    _write_manifest(calibrated=True)
    eng_24_cal = _build_engine(hw_24gb, has_w8=True)
    assert eng_24_cal.ane_ring_variant == "w8"
    assert eng_24_cal.silicon_policy is not None
    assert eng_24_cal.silicon_policy.host.ane_ring_variant == "w8"
    with (
        patch.object(fg4_engine, "AnePrefixRunner", _Runner),
        patch.object(fg4_engine, "is_ane_resident_bundle", return_value=True),
    ):
        runner = eng_24_cal._resolve_ane_ring_runner(require=True)
        assert runner is not None and runner.bundle == w8_bundle

    # 2. 24 GB + calibrated W8 fails residency check while FP16 passes -> falls back to "fp16".
    eng_24_nonres = _build_engine(hw_24gb, has_w8=True)
    assert eng_24_nonres.ane_ring_variant == "w8"
    with (
        patch.object(fg4_engine, "AnePrefixRunner", _Runner),
        patch.object(fg4_engine, "is_ane_resident_bundle", side_effect=lambda _lib, b: b == fp16_bundle),
    ):
        runner = eng_24_nonres._resolve_ane_ring_runner(require=True)
        assert runner is not None and runner.bundle == fp16_bundle
        assert eng_24_nonres.ane_ring_variant == "fp16"

    # 3. 24 GB + uncalibrated W8 bundle -> falls back to "fp16".
    _write_manifest(calibrated=False)
    eng_24_uncal = _build_engine(hw_24gb, has_w8=True)
    assert eng_24_uncal.ane_ring_variant == "fp16"
    assert eng_24_uncal.silicon_policy is not None
    assert eng_24_uncal.silicon_policy.host.ane_ring_variant == "fp16"

    # 4. 24 GB + missing W8 bundle -> falls back to "fp16".
    _write_manifest(calibrated=True)
    eng_24_missing = _build_engine(hw_24gb, has_w8=False)
    assert eng_24_missing.ane_ring_variant == "fp16"

    # 5. 48 GB + calibrated W8 bundle -> stays on "fp16".
    eng_48_cal = _build_engine(hw_48gb, has_w8=True)
    assert eng_48_cal.ane_ring_variant == "fp16"
    assert eng_48_cal.silicon_policy is not None
    assert eng_48_cal.silicon_policy.host.ane_ring_variant == "fp16"
    with (
        patch.object(fg4_engine, "AnePrefixRunner", _Runner),
        patch.object(fg4_engine, "is_ane_resident_bundle", return_value=True),
    ):
        runner = eng_48_cal._resolve_ane_ring_runner(require=True)
        assert runner is not None and runner.bundle == fp16_bundle

    # 6. Unknown RAM (0 GB) + calibrated W8 bundle -> stays safely on "fp16".
    eng_0_ram = _build_engine({**hw_24gb, "mem_gb": 0}, has_w8=True)
    assert eng_0_ram.ane_ring_variant == "fp16"


def test_mtp_drafter_auto_lazy_load_cpu(tmp_path: Path) -> None:
    """Verify default `mtp_drafter='auto'` pays 0 init load, skips B=8/T>0/absent drafter, lazy-loads on B=1, T=0, and None/False disable it."""
    import fast_gemma4.mtp_drafter as fg4_mtp
    import fast_shared.hw_check as fs_hw

    assert inspect.signature(FastGemma4Engine.__init__).parameters["mtp_drafter"].default == "auto"

    fake_mx = _FakeMx()

    class _DummyPaths:
        lib_ane_prefix = tmp_path / "missing.dylib"

        def resolve_ane_bundle(self, *_a: Any, **_kw: Any) -> None:
            return None

    class _DummyModel:
        hidden_size = 1536
        num_layers = 35
        n_kv_heads = 1
        first_kv_shared = 15
        window_size = 512
        kv_i8 = True
        model_dir = tmp_path / "ckpt"
        tokenizer_json = tmp_path / "tokenizer.json"

        def prefill_prefix(self, key: Any) -> dict[str, Any]:
            return {"l_sys": len(key), "k_sys": [], "v_sys": []}

        def make_batch_cache(self, pkv: dict[str, Any], *, batch_size: int, doc_cap_full: int, swa_ring_cap: int) -> dict[str, Any]:
            return {"batch_size": batch_size, "l_sys": pkv["l_sys"], "layers": []}

        def prefill_suffix_batch(self, sufs: Any, cache: dict[str, Any], *, group_size: int = 8, apply_softcap: bool = True) -> tuple[_FakeArray, _FakeArray]:
            B = len(sufs)
            return (
                _FakeArray(np.zeros((B, 1, 8), dtype=np.float32)),
                _FakeArray(np.full((B,), 4, dtype=np.int32), dtype=fake_mx.int32),
            )

    class _DummyTokenizer:
        bos_token_id = 2
        stop_token_ids = frozenset({1})

        def format_chat(self, msgs: Any, **_kw: Any) -> str:
            return str(msgs)

        def encode_batch(self, texts: list[str], **_kw: Any) -> list[list[int]]:
            return [[2, 10, 11 + i] for i in range(len(texts))]

        @staticmethod
        def decode(ids: list[int], skip_special_tokens: bool = True) -> str:
            return "ok"

    def _fake_roll(_model: Any, _stop: Any, _g: Any, _pkv: Any, sufs: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "prefill_sec": 0.01,
            "decode_sec": 0.01,
            "gen_toks": [[5] for _ in sufs],
            "valid_schemas": [True for _ in sufs],
            "tot_out_tok": len(sufs),
            "llm_steps": 1,
            "ff_early14_steps": 0,
            "ff_tokens": 0,
            "mask_fallbacks": 0,
            "used_speculative": False,
        }

    def _fake_prompt_lookup(_model: Any, _stop: Any, _g: Any, _lg: Any, _ob: Any, _c: Any, _wp: Any, B: int, **_kw: Any) -> dict[str, Any]:
        return {
            "decode_sec": 0.01,
            "gen_toks": [[6] for _ in range(B)],
            "tot_out_tok": B,
            "ff_tokens": 0,
            "used_speculative": True,
        }

    mtp_decode_calls: list[int] = []

    def _fake_mtp_decode(_model: Any, drafter: Any, _stop: Any, _g: Any, _lg: Any, _ob: Any, _c: Any, B: int, **_kw: Any) -> dict[str, Any]:
        mtp_decode_calls.append(B)
        return {
            "decode_sec": 0.01,
            "gen_toks": [[7] for _ in range(B)],
            "tot_out_tok": B,
            "ff_tokens": 0,
            "mtp_rounds": 1,
            "mtp_accepted": 2,
            "mtp_proposed": 3,
            "used_speculative": True,
            "speculative_backend": "mtp",
        }

    hw_m5 = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M5",
        "gpu_family": "Apple10",
        "supports_apple10": True,
        "mem_gb": 48,
        "macos_version": "26.6",
    }

    def _make_auto_engine(*, mtp_drafter: Any = "auto", use_default: bool = False) -> FastGemma4Engine:
        with (
            patch.object(fs_hw, "probe_host_hardware", return_value=hw_m5),
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "_host_ram_gib", return_value=48.0),
            patch.object(fg4_engine, "resolve_paths", return_value=_DummyPaths()),
            patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel()),
            patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=_DummyTokenizer()),
        ):
            if use_default:
                return FastGemma4Engine(default_mode="gpu", warmup=False)
            return FastGemma4Engine(default_mode="gpu", mtp_drafter=mtp_drafter, warmup=False)

    resolve_calls: list[tuple[Any, bool]] = []
    drafter_init_calls: list[Path] = []
    staged_dir = tmp_path / "staged_drafter"
    staged_dir.mkdir()
    (staged_dir / "model.safetensors").write_bytes(b"stub")

    def _fake_resolve_present(repo: Any, *, local_files_only: bool = True) -> Path:
        resolve_calls.append((repo, local_files_only))
        return staged_dir

    def _fake_resolve_absent(repo: Any, *, local_files_only: bool = True) -> Path:
        resolve_calls.append((repo, local_files_only))
        raise FileNotFoundError("no local drafter staged")

    def _fake_drafter_ctor(_model: Any, ddir: Any, **_kw: Any) -> object:
        drafter_init_calls.append(Path(ddir))
        return object()

    prev_cache = fg4_mtp._APPLE10_NAX_CACHE
    try:
        fg4_mtp._APPLE10_NAX_CACHE = True  # Also verify Apple10 NAX skips B=2 lazy-load!

        # 1. Default init (`mtp_drafter="auto"`) performs ZERO drafter resolution or loading.
        with (
            patch.object(fg4_engine, "resolve_mtp_drafter_dir", side_effect=_fake_resolve_present),
            patch.object(fg4_engine, "Gemma4MtpDrafter", side_effect=_fake_drafter_ctor),
        ):
            eng = _make_auto_engine(use_default=True)
        assert eng._mtp_drafter_auto is True
        assert eng.mtp_drafter is None
        assert resolve_calls == [] and drafter_init_calls == []

        with (
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "resolve_mtp_drafter_dir", side_effect=_fake_resolve_present),
            patch.object(fg4_engine, "Gemma4MtpDrafter", side_effect=_fake_drafter_ctor),
            patch.object(fg4_engine, "run_decode_rolling", side_effect=_fake_roll),
            patch.object(fg4_engine, "run_decode_prompt_lookup", side_effect=_fake_prompt_lookup),
            patch.object(fg4_engine, "run_decode_mtp", side_effect=_fake_mtp_decode),
        ):
            # 2. B=8 batch ETL -> 0 lazy-load, uses rolling.
            r_b8 = eng.generate_batch(["doc"] * 8, temperature=0.0, mode="gpu")
            assert r_b8.stats["rolling"] is True
            assert eng.mtp_drafter is None and resolve_calls == [] and drafter_init_calls == []

            # 3. B=1 with temperature > 0 -> 0 lazy-load, uses rolling.
            r_temp = eng.generate_batch(["doc"], temperature=0.7, mode="gpu")
            assert r_temp.stats["rolling"] is True
            assert eng.mtp_drafter is None and resolve_calls == [] and drafter_init_calls == []

            # 4. B=2 on Apple10 NAX -> 0 lazy-load, falls back to PromptLookup.
            r_b2_nax = eng.generate_batch(["d1", "d2"], temperature=0.0, mode="gpu")
            assert r_b2_nax.stats.get("speculative_backend") != "mtp"
            assert eng.mtp_drafter is None and resolve_calls == [] and drafter_init_calls == []

            # 5. First qualifying call (B=1, temperature=0.0) -> triggers lazy-load once!
            r_b1_a = eng.generate_batch(["doc"], temperature=0.0, mode="gpu")
            assert r_b1_a.stats.get("speculative_backend") == "mtp"
            assert eng.mtp_drafter is not None
            assert len(resolve_calls) == 1 and resolve_calls[0][1] is True
            assert drafter_init_calls == [staged_dir]
            assert mtp_decode_calls == [1]

            # 6. Second B=1, T=0 call reuses cached drafter (still 1 init call).
            r_b1_b = eng.generate_batch(["doc"], temperature=0.0, mode="gpu")
            assert r_b1_b.stats.get("speculative_backend") == "mtp"
            assert len(resolve_calls) == 1 and len(drafter_init_calls) == 1
            assert mtp_decode_calls == [1, 1]

            # 7. Subsequent B=8 call never invokes MTP even after drafter is resident.
            r_b8_after = eng.generate_batch(["doc"] * 8, temperature=0.0, mode="gpu")
            assert r_b8_after.stats["rolling"] is True
            assert mtp_decode_calls == [1, 1]

        # 8. Explicit mtp_drafter=None or False completely disables MTP even on B=1, T=0.
        for disabled_val in (None, False):
            resolve_calls.clear()
            drafter_init_calls.clear()
            mtp_decode_calls.clear()
            eng_off = _make_auto_engine(mtp_drafter=disabled_val)
            assert eng_off._mtp_drafter_auto is False and eng_off.mtp_drafter is None
            with (
                patch.object(fg4_engine, "mx", fake_mx),
                patch.object(fg4_engine, "resolve_mtp_drafter_dir", side_effect=_fake_resolve_present),
                patch.object(fg4_engine, "Gemma4MtpDrafter", side_effect=_fake_drafter_ctor),
                patch.object(fg4_engine, "run_decode_prompt_lookup", side_effect=_fake_prompt_lookup),
                patch.object(fg4_engine, "run_decode_mtp", side_effect=_fake_mtp_decode),
            ):
                r_off = eng_off.generate_batch(["doc"], temperature=0.0, mode="gpu")
                assert r_off.stats.get("speculative_backend") != "mtp"
                assert eng_off.mtp_drafter is None
                assert resolve_calls == [] and drafter_init_calls == [] and mtp_decode_calls == []

        # 9. Absent local drafter under mtp_drafter="auto": falls back cleanly without raising or re-probing.
        resolve_calls.clear()
        drafter_init_calls.clear()
        mtp_decode_calls.clear()
        eng_absent = _make_auto_engine(use_default=True)
        with (
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "resolve_mtp_drafter_dir", side_effect=_fake_resolve_absent),
            patch.object(fg4_engine, "Gemma4MtpDrafter", side_effect=_fake_drafter_ctor),
            patch.object(fg4_engine, "run_decode_prompt_lookup", side_effect=_fake_prompt_lookup),
            patch.object(fg4_engine, "run_decode_mtp", side_effect=_fake_mtp_decode),
        ):
            r_miss1 = eng_absent.generate_batch(["doc"], temperature=0.0, mode="gpu")
            r_miss2 = eng_absent.generate_batch(["doc"], temperature=0.0, mode="gpu")
            assert r_miss1.stats.get("speculative_backend") != "mtp"
            assert r_miss2.stats.get("speculative_backend") != "mtp"
            assert eng_absent.mtp_drafter is None
            assert len(resolve_calls) == 1 and resolve_calls[0][1] is True
            assert drafter_init_calls == [] and mtp_decode_calls == []
    finally:
        fg4_mtp._APPLE10_NAX_CACHE = prev_cache


def test_use_ane_ring_auto_chip_ram_batch_and_explicit_overrides_cpu(tmp_path: Path) -> None:
    """Verify chip-, RAM-, kv_i8-, and batch-aware default `use_ane_ring='auto'` plus explicit True/False overrides."""
    import fast_gemma4.mtp_drafter as fg4_mtp
    import fast_shared.hw_check as fs_hw

    assert inspect.signature(FastGemma4Engine.generate_batch).parameters["use_ane_ring"].default == "auto"

    ane_dir = tmp_path / "ane_models"
    ane_dir.mkdir()
    fp16_bundle = ane_dir / "gemma4_early14_prefix_S256_fp16.mlmodelc"
    fp16_bundle.mkdir()
    dylib_stub = tmp_path / "libgemma4_ane_prefix.dylib"
    dylib_stub.write_bytes(b"stub")

    fake_mx = _FakeMx()

    class _Paths:
        def __init__(self, *, has_bundle: bool = True) -> None:
            self.lib_ane_prefix = dylib_stub
            self.has_bundle = has_bundle

        def resolve_ane_bundle(
            self,
            variant: str,
            *,
            model_dir: Any = None,
            local_files_only: bool = True,
            exit_layer: int | None = None,
        ) -> Path | None:
            if not self.has_bundle:
                return None
            return fp16_bundle if variant == "fp16" else None

    class _DummyModel:
        hidden_size = 1536
        num_layers = 35
        n_kv_heads = 1
        first_kv_shared = 15
        window_size = 512
        model_dir = tmp_path / "ckpt"
        tokenizer_json = tmp_path / "tokenizer.json"

        def __init__(self, *, kv_i8: bool = True) -> None:
            self.kv_i8 = kv_i8
            self.embed_table_builds = 0

        def prefill_prefix(self, key: Any) -> dict[str, Any]:
            return {"l_sys": len(key), "k_sys": [], "v_sys": []}

        def build_cpu_ane_embed_tables(self) -> tuple[np.ndarray, np.ndarray]:
            self.embed_table_builds += 1
            return np.zeros((2, 2), dtype=np.float16), np.zeros((2, 2), dtype=np.float16)

        def make_batch_cache(self, pkv: dict[str, Any], *, batch_size: int, doc_cap_full: int, swa_ring_cap: int) -> dict[str, Any]:
            return {"batch_size": batch_size, "l_sys": pkv["l_sys"], "layers": []}

        def prefill_suffix_batch(self, sufs: Any, cache: dict[str, Any], *, group_size: int = 8, apply_softcap: bool = True) -> tuple[_FakeArray, _FakeArray]:
            B = len(sufs)
            return (
                _FakeArray(np.zeros((B, 1, 8), dtype=np.float32)),
                _FakeArray(np.full((B,), 4, dtype=np.int32), dtype=fake_mx.int32),
            )

    class _DummyTokenizer:
        bos_token_id = 2
        stop_token_ids = frozenset({1})

        def __init__(self, suffix_len: int = 96) -> None:
            self.suffix_len = suffix_len

        def format_chat(self, msgs: Any, **_kw: Any) -> str:
            return str(msgs)

        def encode_batch(self, texts: list[str], **_kw: Any) -> list[list[int]]:
            return [[2, 10] + [20 + ((i + j) % 50) for j in range(self.suffix_len)] for i in range(len(texts))]

        @staticmethod
        def decode(ids: list[int], skip_special_tokens: bool = True) -> str:
            return "ok"

    class _FakeRunner:
        def __init__(self, lib_path: Path, bundle: Path, **_kw: Any) -> None:
            self.bundle = bundle

        def close(self) -> None:
            pass

    class _FakeRing:
        def __init__(self, model: Any, runner: Any, pkv: Any, w_emb: Any, w_pli14: Any) -> None:
            self.model = model
            self.runner = runner
            self.pkv = pkv

    hw_apple9_48 = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M4 Pro",
        "gpu_family": "Apple9",
        "supports_apple10": False,
        "mem_gb": 48,
        "macos_version": "26.6",
    }
    hw_apple10_48 = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M5 Pro",
        "gpu_family": "Apple10",
        "supports_apple10": True,
        "mem_gb": 48,
        "macos_version": "27.2.1",
    }
    hw_apple9_24 = {**hw_apple9_48, "mem_gb": 24}

    def _build_engine(
        hw_dict: dict[str, Any],
        *,
        kv_i8: bool = True,
        has_bundle: bool = True,
        allow_ane_on_low_ram: bool = False,
        suffix_len: int = 96,
    ) -> FastGemma4Engine:
        with (
            patch.object(fs_hw, "probe_host_hardware", return_value=hw_dict),
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "_host_ram_gib", return_value=float(hw_dict["mem_gb"])),
            patch.object(fg4_engine, "resolve_paths", return_value=_Paths(has_bundle=has_bundle)),
            patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel(kv_i8=kv_i8)),
            patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=_DummyTokenizer(suffix_len=suffix_len)),
        ):
            return FastGemma4Engine(
                kv_i8=kv_i8,
                allow_ane_on_low_ram=allow_ane_on_low_ram,
                mtp_drafter=None,
                warmup=False,
            )

    seen_rings: list[tuple[Any, Any]] = []

    def _fake_roll(_model: Any, _stop: Any, _g: Any, _pkv: Any, sufs: Any, **kw: Any) -> dict[str, Any]:
        ring_arg = kw.get("ane_ring")
        seen_rings.append((kw.get("use_ane_ring"), ring_arg))
        return {
            "prefill_sec": 0.01,
            "decode_sec": 0.01,
            "gen_toks": [[5] for _ in sufs],
            "valid_schemas": [True for _ in sufs],
            "tot_out_tok": len(sufs),
            "llm_steps": 1,
            "ff_early14_steps": 0,
            "ff_tokens": 0,
            "mask_fallbacks": 0,
            "used_speculative": False,
            "used_ane_ring": ring_arg is not None,
        }

    def _fake_prompt_lookup(_model: Any, _stop: Any, _g: Any, _lg: Any, _ob: Any, _c: Any, _wp: Any, B: int, **_kw: Any) -> dict[str, Any]:
        return {
            "decode_sec": 0.01,
            "gen_toks": [[6] for _ in range(B)],
            "tot_out_tok": B,
            "ff_tokens": 0,
            "used_speculative": True,
        }

    prev_cache = fg4_mtp._APPLE10_NAX_CACHE
    try:
        fg4_mtp._APPLE10_NAX_CACHE = None
        with (
            patch.object(fg4_engine, "mx", fake_mx),
            patch.object(fg4_engine, "AnePrefixRunner", _FakeRunner),
            patch.object(fg4_engine, "AsyncHybridPrefillRing", _FakeRing),
            patch.object(fg4_engine, "is_ane_resident_bundle", return_value=True),
            patch.object(fg4_engine, "run_decode_rolling", side_effect=_fake_roll),
            patch.object(fg4_engine, "run_decode_prompt_lookup", side_effect=_fake_prompt_lookup),
        ):
            # 1. Apple9 48 GB, B=32, default use_ane_ring="auto" -> ENABLED.
            eng_a9 = _build_engine(hw_apple9_48)
            r_a9_b32 = eng_a9.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_a9_b32.stats["used_ane_ring"] is True
            assert r_a9_b32.stats["mode_used"] == "hetero_ane_gpu"
            assert seen_rings[-1][0] == "auto" and isinstance(seen_rings[-1][1], _FakeRing)

            # 2. Apple10 48 GB, B=32, default use_ane_ring="auto" -> DISABLED (pure GPU wins on NAX).
            eng_a10 = _build_engine(hw_apple10_48)
            r_a10_b32 = eng_a10.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_a10_b32.stats["used_ane_ring"] is False
            assert r_a10_b32.stats["mode_used"] == "gpu"
            assert seen_rings[-1] == (False, None)
            assert eng_a10.model.embed_table_builds == 0  # type: ignore[attr-defined]

            # 3. Apple9 48 GB, B=4 and B=15 (< max(16, ane_cohort_size)) -> DISABLED.
            eng_a9_small = _build_engine(hw_apple9_48)
            r_a9_b4 = eng_a9_small.generate_batch([f"doc-{i}" for i in range(4)])
            assert r_a9_b4.stats.get("used_ane_ring", False) is False
            assert eng_a9_small.model.embed_table_builds == 0  # type: ignore[attr-defined]
            r_a9_b15 = eng_a9_small.generate_batch([f"doc-{i}" for i in range(15)])
            assert r_a9_b15.stats["used_ane_ring"] is False
            assert seen_rings[-1] == (False, None)
            # Also honors larger ane_cohort_size=32 when B=24 < max(16, 32).
            r_a9_cohort = eng_a9_small.generate_batch([f"doc-{i}" for i in range(24)], ane_cohort_size=32)
            assert r_a9_cohort.stats["used_ane_ring"] is False
            assert seen_rings[-1] == (False, None)

            # 4. Apple9 24 GB (low RAM): disabled unless allow_ane_on_low_ram=True.
            eng_a9_24_off = _build_engine(hw_apple9_24, allow_ane_on_low_ram=False)
            r_24_off = eng_a9_24_off.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_24_off.stats["used_ane_ring"] is False
            eng_a9_24_on = _build_engine(hw_apple9_24, allow_ane_on_low_ram=True)
            r_24_on = eng_a9_24_on.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_24_on.stats["used_ane_ring"] is True

            # 5. kv_i8=False under default use_ane_ring="auto" -> None cleanly without raising!
            eng_no_i8 = _build_engine(hw_apple9_48, kv_i8=False)
            r_no_i8 = eng_no_i8.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_no_i8.stats["used_ane_ring"] is False
            assert seen_rings[-1] == (False, None)
            # Explicit use_ane_ring=True or "require" with kv_i8=False -> raises ValueError!
            with pytest.raises(ValueError, match="use_ane_ring=True requires kv_i8=True"):
                eng_no_i8.generate_batch([f"doc-{i}" for i in range(32)], use_ane_ring=True)
            with pytest.raises(ValueError, match="use_ane_ring=True requires kv_i8=True"):
                eng_no_i8.generate_batch([f"doc-{i}" for i in range(32)], use_ane_ring="require")

            # 6. Missing ANE bundle under default use_ane_ring="auto" -> None cleanly without raising.
            eng_no_bundle = _build_engine(hw_apple9_48, has_bundle=False)
            r_no_bundle = eng_no_bundle.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_no_bundle.stats["used_ane_ring"] is False
            assert seen_rings[-1] == (False, None)

            # 7. Short suffix (<64 tokens) under "auto" -> None cleanly without building CPU tables.
            eng_short_suf = _build_engine(hw_apple9_48, suffix_len=16)
            r_short = eng_short_suf.generate_batch([f"doc-{i}" for i in range(32)])
            assert r_short.stats["used_ane_ring"] is False
            assert eng_short_suf.model.embed_table_builds == 0  # type: ignore[attr-defined]

            # 8. Explicit mode="gpu" or use_ane_ring=False disables the ring on Apple9 48 GB B=32.
            r_mode_gpu = eng_a9.generate_batch([f"doc-{i}" for i in range(32)], mode="gpu")
            assert r_mode_gpu.stats["used_ane_ring"] is False
            r_explicit_off = eng_a9.generate_batch([f"doc-{i}" for i in range(32)], use_ane_ring=False)
            assert r_explicit_off.stats["used_ane_ring"] is False

            # 9. Explicit use_ane_ring=True forces the ring even on Apple10 and small B=4 (and mode="gpu").
            r_a10_forced = eng_a10.generate_batch([f"doc-{i}" for i in range(4)], mode="gpu", use_ane_ring=True)
            assert r_a10_forced.stats["used_ane_ring"] is True
            assert r_a10_forced.stats["mode_used"] == "hetero_ane_gpu"
            assert seen_rings[-1][0] is True and isinstance(seen_rings[-1][1], _FakeRing)
    finally:
        fg4_mtp._APPLE10_NAX_CACHE = prev_cache


def test_tied_checkpoint_vocab_remap_auto_detect_and_validation_cpu(tmp_path: Path) -> None:
    """Verify tied-weight ckpt `vocab_remap.npz` auto-detection, full-vocab override, `vocab_out <= 0` validation, and typed `extract_parquet`."""
    from functools import partial
    from fast_gemma4.cli import build_parser
    from fast_gemma4.pipeline import extract_parquet

    # 1. Typed `extract_parquet` wrapper (not functools.partial) defaulting `whitespace_flexible=None`
    assert not isinstance(extract_parquet, partial)
    assert inspect.isfunction(extract_parquet)
    sig = inspect.signature(extract_parquet)
    assert sig.parameters["whitespace_flexible"].default is None

    # 2. CLI rejects `--vocab-out 0` and negative values cleanly
    parser = build_parser()
    for bad in ("0", "-1", "-150786"):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "hi", "--vocab-out", bad])
        assert exc_info.value.code == 2

    # 3. `FastGemma4Model.__init__` rejects `vocab_out <= 0` immediately before touching weights
    for bad_v in (0, -1, -100):
        with pytest.raises(ValueError, match="vocab_out"):
            FastGemma4Model(vocab_out=bad_v, local_files_only=True)

    # 4. Synthetic tied-weight checkpoint (2 layers, 0 shared) using pure-CPU `_FakeMx` / `_FakeNN`
    V_FULL = 64
    V_KEPT = 24
    H_DIM = 64
    PLE_D = 16
    D_SWA = 16
    D_FULL = 32
    MLP_H = 64
    N_LAYERS = 2

    model_dir = tmp_path / "tied_ckpt"
    model_dir.mkdir()

    vocab = {f"tok{i}": i for i in range(V_FULL)}
    tok = Tokenizer(BPE(vocab=vocab, merges=[]))
    tok.save(str(model_dir / "tokenizer.json"))

    cfg = {
        "text_config": {
            "hidden_size": H_DIM,
            "num_hidden_layers": N_LAYERS,
            "num_kv_shared_layers": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 512,
            "head_dim": D_SWA,
            "global_head_dim": D_FULL,
            "hidden_size_per_layer_input": PLE_D,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
        }
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"mock")

    fake_mx = _FakeMx()

    class _FakeLinear:
        def __init__(self, in_f: int, out_f: int, bias: bool = False) -> None:
            self.in_features = in_f
            self.out_features = out_f
            self.weight = None

        def parameters(self) -> list[Any]:
            return [self.weight]

    class _FakeNN:
        Linear = _FakeLinear
        QuantizedLinear = type("_NeverQLinear", (), {})
        QuantizedEmbedding = type("_NeverQEmb", (), {})
        Embedding = _FakeLinear

    def _bf16(out_f: int, in_f: int) -> _FakeArray:
        return _FakeArray(np.zeros((out_f, in_f), dtype=np.float32), dtype=fake_mx.bfloat16)

    raw_weights: dict[str, _FakeArray] = {
        "language_model.model.embed_tokens.weight": _bf16(V_FULL, H_DIM),
        "language_model.model.embed_tokens_per_layer.weight": _bf16(V_FULL, N_LAYERS * PLE_D),
        "language_model.model.per_layer_model_projection.weight": _bf16(N_LAYERS * PLE_D, H_DIM),
        "language_model.model.per_layer_projection_norm.weight": _FakeArray(np.ones((PLE_D,), dtype=np.float32), dtype=fake_mx.bfloat16),
        "language_model.model.norm.weight": _FakeArray(np.ones((H_DIM,), dtype=np.float32), dtype=fake_mx.bfloat16),
    }
    for i, hd in enumerate((D_SWA, D_FULL)):
        lp = f"language_model.model.layers.{i}"
        raw_weights[f"{lp}.self_attn.q_proj.weight"] = _bf16(2 * hd, H_DIM)
        raw_weights[f"{lp}.self_attn.k_proj.weight"] = _bf16(hd, H_DIM)
        raw_weights[f"{lp}.self_attn.v_proj.weight"] = _bf16(hd, H_DIM)
        raw_weights[f"{lp}.self_attn.k_norm.weight"] = _FakeArray(np.ones((hd,), dtype=np.float32), dtype=fake_mx.bfloat16)
        raw_weights[f"{lp}.self_attn.q_norm.weight"] = _FakeArray(np.ones((hd,), dtype=np.float32), dtype=fake_mx.bfloat16)
        raw_weights[f"{lp}.self_attn.o_proj.weight"] = _bf16(H_DIM, 2 * hd)
        raw_weights[f"{lp}.mlp.gate_proj.weight"] = _bf16(MLP_H, H_DIM)
        raw_weights[f"{lp}.mlp.up_proj.weight"] = _bf16(MLP_H, H_DIM)
        raw_weights[f"{lp}.mlp.down_proj.weight"] = _bf16(H_DIM, MLP_H)
        raw_weights[f"{lp}.per_layer_input_gate.weight"] = _bf16(PLE_D, H_DIM)
        raw_weights[f"{lp}.per_layer_projection.weight"] = _bf16(H_DIM, PLE_D)
        for nw in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"):
            raw_weights[f"{lp}.{nw}.weight"] = _FakeArray(np.ones((H_DIM,), dtype=np.float32), dtype=fake_mx.bfloat16)
        raw_weights[f"{lp}.layer_scalar"] = _FakeArray([1.0], dtype=fake_mx.bfloat16)

    fake_mx.load = lambda path: raw_weights  # type: ignore[attr-defined]
    fake_mx.power = lambda a, b: _FakeArray(np.power(a, b.data), dtype=fake_mx.float32)  # type: ignore[attr-defined]
    paths_obj = resolve_paths(tmp_path / "home")

    with (
        patch.object(fg4_model, "mx", fake_mx),
        patch.object(fg4_model, "nn", _FakeNN),
        patch.object(fg4_model, "tree_flatten", side_effect=lambda x: []),
    ):
        # (a) Stock tied checkpoint (no vocab_remap.npz in model_dir) + vocab_out=None -> full vocab
        m_stock = FastGemma4Model(paths=paths_obj, model_dir=model_dir, vocab_out=None, use_mpp=False)
        assert m_stock.remapper is None
        assert m_stock.vocab_size == V_FULL
        assert int(m_stock.lm_head.weight.shape[0]) == V_FULL

        # (b) Packed tied checkpoint WITH model_dir / "vocab_remap.npz" + vocab_out=None -> auto-loads ckpt_remap
        kept_ids = np.arange(0, V_KEPT * 2, 2, dtype=np.int32)
        np.savez(model_dir / "vocab_remap.npz", kept_old_ids=kept_ids, old_ids=kept_ids)

        m_packed = FastGemma4Model(paths=paths_obj, model_dir=model_dir, vocab_out=None, use_mpp=False)
        assert m_packed.remapper is not None
        assert m_packed.vocab_size == V_KEPT
        assert int(m_packed.lm_head.weight.shape[0]) == V_KEPT
        np.testing.assert_array_equal(m_packed.new_to_old, kept_ids)

        # (c) Explicit vocab_out=full_vocab_rows overrides ckpt_remap to full vocab
        m_override = FastGemma4Model(paths=paths_obj, model_dir=model_dir, vocab_out=V_FULL, use_mpp=False)
        assert m_override.remapper is None
        assert m_override.vocab_size == V_FULL
        assert int(m_override.lm_head.weight.shape[0]) == V_FULL
