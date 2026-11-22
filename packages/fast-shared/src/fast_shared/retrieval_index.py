"""Compact MRL/INT8/Binary retrieval index and multi-vector MaxSim document scoring.

Implements the empirically validated retrieval winners across `fast-silicon` encoders:
  1. Universal lossless full-dim symmetric `INT8` (`int8` + `fp16` scale = `4.0x` compression,
     `|delta nDCG@10| <= 0.0007` across `FastBgeSmall` `384d`, `FastEmbeddingGemma2` `768d`,
     and `FastPplx` `1024d`).
  2. Model-aware `MRL` truncation (`512d int8` = `6.0x` compression, `+0.38 pp` `LegalBench-RAG`
     on MRL-trained `FastEmbeddingGemma2`, while protecting non-MRL `FastPplx` & `FastBgeSmall`
     from premature dimension truncation).
  3. Two-stage `1-bit` packed binary Hamming pre-filter (`vcntq_u8` ARM NEON / `uint8` LUT)
     followed by exact `INT8` dot-product rerank (`0.43-0.54 ms` at `100K` chunks = `9x` faster).
  4. Multi-vector `MaxSim` / `Hybrid` / `Top-2` window aggregation over `[doc_offsets[d]:doc_offsets[d+1]]`,
     lifting `32K` `LongEmbed` passkey `nDCG@10` from `0.047 -> 1.000` (`0.7510 -> 0.9923` combined).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Literal

import numpy as np

from .embeddings import truncate_and_normalize

IndexMode = Literal["auto", "int8", "mrl_int8", "binary_int8", "fp16", "fp32"]
WindowAgg = Literal["maxsim", "hybrid", "top2_mean", "mean"]

VALID_INDEX_MODES: frozenset[str] = frozenset(
    {"auto", "int8", "mrl_int8", "binary_int8", "fp16", "fp32"}
)
VALID_WINDOW_AGGS: frozenset[str] = frozenset({"maxsim", "hybrid", "top2_mean", "mean"})

_POPCOUNT_LUT: np.ndarray = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

_NEON_CPP_SOURCE = r"""
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

extern "C" {

void fast_neon_int8_scores(
    const int8_t* __restrict__ q_i8,
    const _Float16* __restrict__ q_sc,
    const int8_t* __restrict__ d_i8,
    const _Float16* __restrict__ d_sc,
    int Q,
    int N,
    int D,
    float* __restrict__ out_scores
) {
    for (int qi = 0; qi < Q; ++qi) {
        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D;
        const float q_s = static_cast<float>(q_sc[qi]);
        float* out_r = out_scores + static_cast<size_t>(qi) * N;
        for (int di = 0; di < N; ++di) {
            const int8_t* dr = d_i8 + static_cast<size_t>(di) * D;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 64 <= D; k += 64) {
                acc0 = vdotq_s32(acc0, vld1q_s8(qr + k + 0), vld1q_s8(dr + k + 0));
                acc1 = vdotq_s32(acc1, vld1q_s8(qr + k + 16), vld1q_s8(dr + k + 16));
                acc2 = vdotq_s32(acc2, vld1q_s8(qr + k + 32), vld1q_s8(dr + k + 32));
                acc3 = vdotq_s32(acc3, vld1q_s8(qr + k + 48), vld1q_s8(dr + k + 48));
            }
            for (; k + 16 <= D; k += 16) {
                acc0 = vdotq_s32(acc0, vld1q_s8(qr + k), vld1q_s8(dr + k));
            }
            int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
            for (; k < D; ++k) {
                dot += static_cast<int32_t>(qr[k]) * static_cast<int32_t>(dr[k]);
            }
            out_r[di] = static_cast<float>(dot) * (q_s * static_cast<float>(d_sc[di]));
        }
    }
}

void fast_neon_binary_topr_rerank_int8(
    const uint8_t* __restrict__ q_bits,
    const int8_t* __restrict__ q_i8,
    const _Float16* __restrict__ q_sc,
    const uint8_t* __restrict__ d_bits,
    const int8_t* __restrict__ d_i8,
    const _Float16* __restrict__ d_sc,
    int Q,
    int N,
    int n_bytes,
    int D_rerank,
    int top_r,
    int top_k,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_scores
) {
    const int eff_r = std::min(top_r, N);
    const int eff_k = std::min(top_k, eff_r);
    std::vector<uint64_t> packed_cand(N);
    std::vector<std::pair<float, int32_t>> reranked(eff_r);

    for (int qi = 0; qi < Q; ++qi) {
        const uint8_t* qb = q_bits + static_cast<size_t>(qi) * n_bytes;
        for (int di = 0; di < N; ++di) {
            const uint8_t* db = d_bits + static_cast<size_t>(di) * n_bytes;
            uint32_t dist = 0;
            int b = 0;
            for (; b + 16 <= n_bytes; b += 16) {
                uint8x16_t xr = veorq_u8(vld1q_u8(qb + b), vld1q_u8(db + b));
                dist += vaddlvq_u8(vcntq_u8(xr));
            }
            for (; b < n_bytes; ++b) {
                dist += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(qb[b] ^ db[b])));
            }
            packed_cand[di] = (static_cast<uint64_t>(dist) << 32) | static_cast<uint32_t>(di);
        }

        if (eff_r < N) {
            std::nth_element(packed_cand.begin(), packed_cand.begin() + eff_r, packed_cand.end());
        }

        const int8_t* qr = q_i8 + static_cast<size_t>(qi) * D_rerank;
        const float q_s = static_cast<float>(q_sc[qi]);
        for (int r = 0; r < eff_r; ++r) {
            int32_t di = static_cast<int32_t>(packed_cand[r] & 0xFFFFFFFFull);
            const int8_t* dr = d_i8 + static_cast<size_t>(di) * D_rerank;
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 64 <= D_rerank; k += 64) {
                acc0 = vdotq_s32(acc0, vld1q_s8(qr + k + 0), vld1q_s8(dr + k + 0));
                acc1 = vdotq_s32(acc1, vld1q_s8(qr + k + 16), vld1q_s8(dr + k + 16));
                acc2 = vdotq_s32(acc2, vld1q_s8(qr + k + 32), vld1q_s8(dr + k + 32));
                acc3 = vdotq_s32(acc3, vld1q_s8(qr + k + 48), vld1q_s8(dr + k + 48));
            }
            for (; k + 16 <= D_rerank; k += 16) {
                acc0 = vdotq_s32(acc0, vld1q_s8(qr + k), vld1q_s8(dr + k));
            }
            int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
            for (; k < D_rerank; ++k) {
                dot += static_cast<int32_t>(qr[k]) * static_cast<int32_t>(dr[k]);
            }
            float sc = static_cast<float>(dot) * (q_s * static_cast<float>(d_sc[di]));
            reranked[r] = {-sc, di};
        }

        std::partial_sort(reranked.begin(), reranked.begin() + eff_k, reranked.end());
        int32_t* out_idx_r = out_indices + static_cast<size_t>(qi) * top_k;
        float* out_sc_r = out_scores + static_cast<size_t>(qi) * top_k;
        for (int k = 0; k < eff_k; ++k) {
            out_idx_r[k] = reranked[k].second;
            out_sc_r[k] = -reranked[k].first;
        }
        for (int k = eff_k; k < top_k; ++k) {
            out_idx_r[k] = -1;
            out_sc_r[k] = -1e30f;
        }
    }
}

} // extern "C"
"""

_NEON_LOCK = threading.Lock()
_NEON_LIB: ctypes.CDLL | None = None
_NEON_TRIED: bool = False


def _get_neon_lib() -> ctypes.CDLL | None:
    """Lazily compile & load ARM64 NEON `vcntq_u8` + `vdotq_s32` helper on Apple Silicon (`None` fallback)."""
    global _NEON_LIB, _NEON_TRIED
    if _NEON_TRIED:
        return _NEON_LIB
    with _NEON_LOCK:
        if _NEON_TRIED:
            return _NEON_LIB
        _NEON_TRIED = True
        if os.environ.get("FAST_SILICON_DISABLE_NEON_INDEX") == "1":
            return None
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return None
        cxx = shutil.which("clang++")
        if not cxx:
            return None
        try:
            digest = hashlib.sha256(_NEON_CPP_SOURCE.encode("utf-8")).hexdigest()[:12]
            cache_dir = Path(os.environ.get("FAST_SILICON_CACHE_DIR") or (Path.home() / ".cache" / "fast_silicon"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            dylib_path = cache_dir / f"libhamming_int8_neon_{digest}.dylib"
            if not dylib_path.is_file():
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".cpp", dir=str(cache_dir), delete=False, encoding="utf-8"
                ) as src_f:
                    src_f.write(_NEON_CPP_SOURCE)
                    src_path = Path(src_f.name)
                tmp_dylib = cache_dir / f".tmp_{os.getpid()}_{digest}.dylib"
                try:
                    subprocess.run(
                        [
                            cxx,
                            "-O3",
                            "-std=c++17",
                            "-arch",
                            "arm64",
                            "-march=armv8.4-a+dotprod+fp16",
                            "-dynamiclib",
                            str(src_path),
                            "-o",
                            str(tmp_dylib),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=20,
                    )
                    os.replace(tmp_dylib, dylib_path)
                finally:
                    src_path.unlink(missing_ok=True)
                    tmp_dylib.unlink(missing_ok=True)

            lib = ctypes.CDLL(str(dylib_path))
            lib.fast_neon_int8_scores.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.fast_neon_int8_scores.restype = None
            lib.fast_neon_binary_topr_rerank_int8.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            lib.fast_neon_binary_topr_rerank_int8.restype = None
            _NEON_LIB = lib
        except Exception:
            _NEON_LIB = None
        return _NEON_LIB


def quantize_int8_symmetric(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize 2D embeddings `[N, D]` to L2-normalized symmetric `int8` + per-row `float16` scale."""
    arr = np.asarray(vecs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D embedding array, got shape {arr.shape}")
    if arr.dtype == np.int8:
        q = np.ascontiguousarray(arr, dtype=np.int8)
        scales = np.full((q.shape[0],), np.float16(1.0 / 127.0), dtype=np.float16)
        return q, scales
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    if f32.shape[0] == 0:
        return np.empty(f32.shape, dtype=np.int8), np.empty((0,), dtype=np.float16)
    norms = np.linalg.norm(f32, axis=-1, keepdims=True)
    normed = np.where(norms > 1e-12, f32 / np.maximum(norms, 1e-12), np.float32(0.0))
    absmax = np.max(np.abs(normed), axis=-1)
    raw_scale = np.where(absmax > 1e-12, absmax / np.float32(127.0), np.float32(1.0 / 127.0))
    scales_f16 = np.ascontiguousarray(raw_scale.astype(np.float16), dtype=np.float16)
    eff_scale = np.where(
        scales_f16.astype(np.float32) > 1e-12,
        scales_f16.astype(np.float32),
        np.float32(1.0 / 127.0),
    )
    q_i8 = np.clip(np.rint(normed / eff_scale[:, None]), -127.0, 127.0).astype(np.int8)
    return np.ascontiguousarray(q_i8, dtype=np.int8), scales_f16


def dequantize_int8_symmetric(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize symmetric `int8` codes `[N, D]` + `float16` scales `[N]` back to `float32`."""
    c = np.asarray(codes, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32).reshape(-1, 1)
    return np.ascontiguousarray(c * s, dtype=np.float32)


def pack_binary_bits(vecs: np.ndarray, bits: int | None = None) -> np.ndarray:
    """Pack sign bits (`x > 0`) of `[N, D]` embeddings into `[N, ceil(bits / 8)]` `uint8` bytes."""
    arr = np.asarray(vecs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if bits is not None:
        if bits <= 0 or bits > arr.shape[1]:
            raise ValueError(f"bits must be in [1, {arr.shape[1]}], got {bits}")
        arr = arr[:, :bits]
    if arr.shape[0] == 0:
        n_bytes = (arr.shape[1] + 7) // 8
        return np.empty((0, n_bytes), dtype=np.uint8)
    return np.ascontiguousarray(np.packbits(arr > 0, axis=-1), dtype=np.uint8)


def _int8_dot_scores(
    q_i8: np.ndarray,
    q_sc: np.ndarray,
    d_i8: np.ndarray,
    d_sc: np.ndarray,
    *,
    use_neon: bool = True,
) -> np.ndarray:
    q_n, dim = q_i8.shape
    d_n = d_i8.shape[0]
    if q_n == 0 or d_n == 0:
        return np.empty((q_n, d_n), dtype=np.float32)
    lib = _get_neon_lib() if use_neon else None
    if lib is not None:
        qc = np.ascontiguousarray(q_i8, dtype=np.int8)
        qs = np.ascontiguousarray(q_sc, dtype=np.float16)
        dc = np.ascontiguousarray(d_i8, dtype=np.int8)
        ds = np.ascontiguousarray(d_sc, dtype=np.float16)
        out = np.empty((q_n, d_n), dtype=np.float32)
        lib.fast_neon_int8_scores(
            qc.ctypes.data,
            qs.ctypes.data,
            dc.ctypes.data,
            ds.ctypes.data,
            int(q_n),
            int(d_n),
            int(dim),
            out.ctypes.data,
        )
        return out
    q_f32 = dequantize_int8_symmetric(q_i8, q_sc)
    d_f32 = dequantize_int8_symmetric(d_i8, d_sc)
    return np.ascontiguousarray(q_f32 @ d_f32.T, dtype=np.float32)


def _binary_topr_rerank_int8(
    q_bits: np.ndarray,
    q_i8: np.ndarray,
    q_sc: np.ndarray,
    d_bits: np.ndarray,
    d_i8: np.ndarray,
    d_sc: np.ndarray,
    *,
    top_r: int,
    top_k: int,
    use_neon: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q_n = q_i8.shape[0]
    d_n, dim = d_i8.shape
    if q_n == 0 or top_k <= 0:
        return np.empty((q_n, max(0, top_k)), dtype=np.int32), np.empty((q_n, max(0, top_k)), dtype=np.float32)
    if d_n == 0:
        return np.full((q_n, top_k), -1, dtype=np.int32), np.full((q_n, top_k), -1e30, dtype=np.float32)

    eff_r = min(max(top_k, int(top_r)), d_n)
    eff_k = min(int(top_k), eff_r)

    lib = _get_neon_lib() if use_neon else None
    if lib is not None:
        qb = np.ascontiguousarray(q_bits, dtype=np.uint8)
        qc = np.ascontiguousarray(q_i8, dtype=np.int8)
        qs = np.ascontiguousarray(q_sc, dtype=np.float16)
        db = np.ascontiguousarray(d_bits, dtype=np.uint8)
        dc = np.ascontiguousarray(d_i8, dtype=np.int8)
        ds = np.ascontiguousarray(d_sc, dtype=np.float16)
        out_idx = np.empty((q_n, top_k), dtype=np.int32)
        out_sc = np.empty((q_n, top_k), dtype=np.float32)
        lib.fast_neon_binary_topr_rerank_int8(
            qb.ctypes.data,
            qc.ctypes.data,
            qs.ctypes.data,
            db.ctypes.data,
            dc.ctypes.data,
            ds.ctypes.data,
            int(q_n),
            int(d_n),
            int(qb.shape[1]),
            int(dim),
            int(eff_r),
            int(top_k),
            out_idx.ctypes.data,
            out_sc.ctypes.data,
        )
        return out_idx, out_sc

    out_idx = np.full((q_n, top_k), -1, dtype=np.int32)
    out_sc = np.full((q_n, top_k), -1e30, dtype=np.float32)
    doc_arange = np.arange(d_n, dtype=np.uint64)
    for qi in range(q_n):
        xor_bytes = np.bitwise_xor(d_bits, q_bits[qi : qi + 1])
        dists = _POPCOUNT_LUT[xor_bytes].sum(axis=1, dtype=np.uint32)
        packed = (dists.astype(np.uint64) << np.uint64(32)) | doc_arange
        if eff_r < d_n:
            cand_pos = np.argpartition(packed, eff_r - 1)[:eff_r]
        else:
            cand_pos = np.arange(d_n, dtype=np.int64)
        cand_ids = (packed[cand_pos] & np.uint64(0xFFFFFFFF)).astype(np.int32)
        dots = (d_i8[cand_ids].astype(np.int32) @ q_i8[qi].astype(np.int32)).astype(np.float32)
        sc = dots * (float(q_sc[qi]) * d_sc[cand_ids].astype(np.float32))
        order = np.lexsort((cand_ids, -sc))[:eff_k]
        out_idx[qi, :eff_k] = cand_ids[order]
        out_sc[qi, :eff_k] = sc[order]
    return out_idx, out_sc


def score_document_windows(
    window_scores: np.ndarray,
    doc_offsets: np.ndarray | list[int] | tuple[int, ...],
    *,
    agg: WindowAgg = "maxsim",
    hybrid_alpha: float = 0.7,
) -> np.ndarray:
    """Aggregate `[Q, W]` window similarity scores into `[Q, N_docs]` using ragged `doc_offsets`."""
    if agg not in VALID_WINDOW_AGGS:
        raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")
    ws = np.asarray(window_scores, dtype=np.float32)
    was_1d = ws.ndim == 1
    if was_1d:
        ws = ws[None, :]
    offsets = np.asarray(doc_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 1:
        raise ValueError("doc_offsets must be a 1D cumulative offset array of length N_docs + 1")
    n_docs = int(offsets.size) - 1
    q_n = int(ws.shape[0])
    if n_docs <= 0 or q_n == 0:
        res = np.empty((q_n, max(0, n_docs)), dtype=np.float32)
        return res[0] if was_1d else res
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(ws.shape[1]) or np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"doc_offsets must start at 0, be non-decreasing, and end at W={ws.shape[1]} "
            f"(got start={offsets[0]}, end={offsets[-1]})"
        )

    counts = np.diff(offsets)
    non_empty = counts > 0
    if np.all(non_empty) and agg in ("maxsim", "mean", "hybrid"):
        starts = offsets[:-1]
        max_s = np.maximum.reduceat(ws, starts, axis=1)
        if agg == "maxsim":
            return max_s[0] if was_1d else max_s
        mean_s = np.add.reduceat(ws, starts, axis=1) / counts.astype(np.float32)[None, :]
        if agg == "mean":
            return mean_s[0] if was_1d else mean_s
        alpha = float(hybrid_alpha)
        hyb = alpha * max_s + (1.0 - alpha) * mean_s
        return hyb[0] if was_1d else hyb

    out = np.full((q_n, n_docs), -1e30, dtype=np.float32)
    alpha = float(hybrid_alpha)
    for d_idx in range(n_docs):
        w0 = int(offsets[d_idx])
        w1 = int(offsets[d_idx + 1])
        if w1 <= w0:
            continue
        sl = ws[:, w0:w1]
        if agg == "maxsim":
            out[:, d_idx] = np.max(sl, axis=1)
        elif agg == "mean":
            out[:, d_idx] = np.mean(sl, axis=1)
        elif agg == "hybrid":
            out[:, d_idx] = alpha * np.max(sl, axis=1) + (1.0 - alpha) * np.mean(sl, axis=1)
        else:
            if (w1 - w0) == 1:
                out[:, d_idx] = sl[:, 0]
            else:
                part = np.partition(sl, (w1 - w0) - 2, axis=1)[:, -2:]
                out[:, d_idx] = 0.5 * (part[:, 0] + part[:, 1])
    return out[0] if was_1d else out


def resolve_index_spec(
    *,
    raw_dim: int,
    mode: IndexMode = "auto",
    is_mrl_trained: bool = False,
    mrl_dim: int | None = None,
    prefer_binary: bool = False,
) -> tuple[str, int]:
    """Resolve `(resolved_mode, effective_dim)` honoring whether the encoder is MRL-trained."""
    if mode not in VALID_INDEX_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_INDEX_MODES)}, got {mode!r}")
    if raw_dim <= 0:
        raise ValueError(f"raw_dim must be positive, got {raw_dim}")

    if mode == "auto":
        if is_mrl_trained and raw_dim > 512:
            eff_dim = min(int(mrl_dim or 512), raw_dim)
            eff_mode = "binary_int8" if prefer_binary else "mrl_int8"
            return eff_mode, eff_dim
        eff_dim = min(int(mrl_dim), raw_dim) if mrl_dim is not None else raw_dim
        eff_mode = "binary_int8" if prefer_binary else ("mrl_int8" if eff_dim < raw_dim else "int8")
        return eff_mode, eff_dim

    if mode == "mrl_int8":
        target = int(mrl_dim) if mrl_dim is not None else min(512, raw_dim)
        if target <= 0 or target > raw_dim:
            raise ValueError(f"mrl_dim must be in [1, {raw_dim}], got {target}")
        return "mrl_int8", target

    if mode == "binary_int8":
        default_d = min(512, raw_dim) if (is_mrl_trained and raw_dim > 512) else raw_dim
        target = int(mrl_dim) if mrl_dim is not None else default_d
        if target <= 0 or target > raw_dim:
            raise ValueError(f"mrl_dim must be in [1, {raw_dim}], got {target}")
        return "binary_int8", target

    eff_d = int(mrl_dim) if mrl_dim is not None else raw_dim
    if eff_d <= 0 or eff_d > raw_dim:
        raise ValueError(f"mrl_dim must be in [1, {raw_dim}], got {eff_d}")
    return mode, eff_d


class CompactRetrievalIndex:
    """In-memory retrieval index supporting `INT8`, `MRL+INT8`, `Binary+INT8`, and `MaxSim` windows."""

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        doc_offsets: np.ndarray | list[int] | tuple[int, ...] | None = None,
        mode: IndexMode = "auto",
        is_mrl_trained: bool = False,
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
        use_neon: bool = True,
    ) -> None:
        arr = np.asarray(vectors)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be a 2D array [N, D], got shape {arr.shape}")
        if agg not in VALID_WINDOW_AGGS:
            raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")

        self.raw_dim: int = int(arr.shape[1])
        self.num_windows: int = int(arr.shape[0])
        self.is_mrl_trained: bool = bool(is_mrl_trained)
        self.default_agg: WindowAgg = agg
        self.default_hybrid_alpha: float = float(hybrid_alpha)
        self.use_neon: bool = bool(use_neon)

        if doc_offsets is not None:
            off_arr = np.ascontiguousarray(doc_offsets, dtype=np.int32)
            if off_arr.ndim != 1 or off_arr.size < 1 or int(off_arr[0]) != 0 or int(off_arr[-1]) != self.num_windows:
                raise ValueError(
                    f"doc_offsets must start at 0 and end at num_windows={self.num_windows}"
                )
            self.doc_offsets: np.ndarray | None = off_arr
            self.window_to_doc: np.ndarray | None = np.repeat(
                np.arange(off_arr.size - 1, dtype=np.int32),
                np.diff(off_arr),
            )
            self.num_docs: int = int(off_arr.size) - 1
        else:
            self.doc_offsets = None
            self.window_to_doc = None
            self.num_docs = self.num_windows

        prefer_bin = self.num_windows >= 16384 and self.doc_offsets is None
        resolved_mode, eff_dim = resolve_index_spec(
            raw_dim=self.raw_dim,
            mode=mode,
            is_mrl_trained=self.is_mrl_trained,
            mrl_dim=mrl_dim,
            prefer_binary=prefer_bin,
        )
        self.mode: str = resolved_mode
        self.dim: int = eff_dim

        work = (
            truncate_and_normalize(arr.astype(np.float32, copy=False), eff_dim)
            if eff_dim < self.raw_dim
            else truncate_and_normalize(arr.astype(np.float32, copy=False), self.raw_dim)
        )

        self.codes_i8: np.ndarray | None = None
        self.scales_f16: np.ndarray | None = None
        self.binary_u8: np.ndarray | None = None
        self.dense_vecs: np.ndarray | None = None
        self.binary_bits: int = 0

        if resolved_mode in ("int8", "mrl_int8", "binary_int8"):
            self.codes_i8, self.scales_f16 = quantize_int8_symmetric(work)
            if resolved_mode == "binary_int8":
                b_bits = int(binary_bits) if binary_bits is not None else eff_dim
                if b_bits > self.raw_dim or b_bits <= 0:
                    raise ValueError(f"binary_bits must be in [1, {self.raw_dim}], got {b_bits}")
                src_for_bits = arr if b_bits > eff_dim else work
                self.binary_u8 = pack_binary_bits(src_for_bits, bits=b_bits)
                self.binary_bits = b_bits
        elif resolved_mode == "fp16":
            self.dense_vecs = np.ascontiguousarray(work, dtype=np.float16)
        else:
            self.dense_vecs = np.ascontiguousarray(work, dtype=np.float32)

    @property
    def nbytes(self) -> int:
        total = 0
        for buf in (self.codes_i8, self.scales_f16, self.binary_u8, self.dense_vecs, self.doc_offsets):
            if buf is not None:
                total += int(buf.nbytes)
        return total

    @property
    def compression_ratio_vs_fp32(self) -> float:
        fp32_bytes = self.num_windows * self.raw_dim * 4
        if fp32_bytes <= 0 or self.nbytes <= 0:
            return 1.0
        return float(fp32_bytes) / float(self.nbytes)

    def _prepare_queries(self, query_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        raw_q = np.asarray(query_vecs)
        was_1d = raw_q.ndim == 1
        if was_1d:
            raw_q = raw_q[None, :]
        if raw_q.ndim != 2:
            raise ValueError(f"query_vecs must be 1D or 2D, got shape {raw_q.shape}")
        if raw_q.shape[1] < self.dim:
            raise ValueError(f"query_vecs dim {raw_q.shape[1]} < index dim {self.dim}")
        q_trunc = truncate_and_normalize(raw_q.astype(np.float32, copy=False), self.dim)
        return raw_q, q_trunc, was_1d

    def score_all(
        self,
        query_vecs: np.ndarray,
        *,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> np.ndarray:
        """Compute exact `[Q, N_docs]` (`[N_docs]` if 1D query) similarity scores across all docs."""
        _, q_trunc, was_1d = self._prepare_queries(query_vecs)
        if self.codes_i8 is not None and self.scales_f16 is not None:
            q_i8, q_sc = quantize_int8_symmetric(q_trunc)
            win_scores = _int8_dot_scores(
                q_i8, q_sc, self.codes_i8, self.scales_f16, use_neon=self.use_neon
            )
        else:
            assert self.dense_vecs is not None
            win_scores = np.ascontiguousarray(
                q_trunc @ self.dense_vecs.astype(np.float32, copy=False).T, dtype=np.float32
            )

        if self.doc_offsets is not None:
            doc_scores = score_document_windows(
                win_scores,
                self.doc_offsets,
                agg=agg or self.default_agg,
                hybrid_alpha=self.default_hybrid_alpha if hybrid_alpha is None else float(hybrid_alpha),
            )
        else:
            doc_scores = win_scores
        return doc_scores[0] if was_1d else doc_scores

    def search(
        self,
        query_vecs: np.ndarray,
        k: int = 10,
        *,
        rerank_top_r: int = 64,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(top_indices, top_scores)` of shape `[Q, k]` (`[k]` when `query_vecs` is 1D)."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        raw_q, q_trunc, was_1d = self._prepare_queries(query_vecs)
        q_n = int(q_trunc.shape[0])
        eff_k = min(int(k), max(1, self.num_docs)) if self.num_docs > 0 else int(k)

        if self.num_docs == 0 or q_n == 0:
            empty_i = np.full((q_n, k), -1, dtype=np.int32)
            empty_s = np.full((q_n, k), -1e30, dtype=np.float32)
            return (empty_i[0], empty_s[0]) if was_1d else (empty_i, empty_s)

        if self.mode == "binary_int8" and self.binary_u8 is not None and self.doc_offsets is None:
            assert self.codes_i8 is not None and self.scales_f16 is not None
            q_i8, q_sc = quantize_int8_symmetric(q_trunc)
            src_q = raw_q if self.binary_bits > self.dim else q_trunc
            q_bits = pack_binary_bits(src_q, bits=self.binary_bits)
            idx, sc = _binary_topr_rerank_int8(
                q_bits,
                q_i8,
                q_sc,
                self.binary_u8,
                self.codes_i8,
                self.scales_f16,
                top_r=max(int(rerank_top_r), int(k)),
                top_k=int(k),
                use_neon=self.use_neon,
            )
            return (idx[0], sc[0]) if was_1d else (idx, sc)

        if (
            self.mode == "binary_int8"
            and self.binary_u8 is not None
            and self.doc_offsets is not None
            and self.window_to_doc is not None
        ):
            assert self.codes_i8 is not None and self.scales_f16 is not None
            mean_w = max(1, int(np.ceil(self.num_windows / max(1, self.num_docs))))
            win_top_r = min(self.num_windows, max(int(rerank_top_r), int(k) * 4) * mean_w)
            q_i8, q_sc = quantize_int8_symmetric(q_trunc)
            src_q = raw_q if self.binary_bits > self.dim else q_trunc
            q_bits = pack_binary_bits(src_q, bits=self.binary_bits)
            cand_win_idx, _ = _binary_topr_rerank_int8(
                q_bits,
                q_i8,
                q_sc,
                self.binary_u8,
                self.codes_i8,
                self.scales_f16,
                top_r=win_top_r,
                top_k=win_top_r,
                use_neon=self.use_neon,
            )
            eff_agg = agg or self.default_agg
            eff_alpha = self.default_hybrid_alpha if hybrid_alpha is None else float(hybrid_alpha)
            out_idx = np.full((q_n, k), -1, dtype=np.int32)
            out_sc = np.full((q_n, k), -1e30, dtype=np.float32)
            for qi in range(q_n):
                valid_w = cand_win_idx[qi][cand_win_idx[qi] >= 0]
                cand_docs = np.unique(self.window_to_doc[valid_w])
                if cand_docs.size == 0:
                    continue
                # Score all windows of shortlisted parent documents so hybrid/top2/maxsim stays exact.
                spans = [np.arange(self.doc_offsets[d], self.doc_offsets[d + 1], dtype=np.int32) for d in cand_docs]
                sub_offsets = np.zeros((cand_docs.size + 1,), dtype=np.int32)
                np.cumsum([sp.size for sp in spans], out=sub_offsets[1:])
                flat_w = np.concatenate(spans) if sub_offsets[-1] > 0 else np.empty((0,), dtype=np.int32)
                w_sc = _int8_dot_scores(
                    q_i8[qi : qi + 1],
                    q_sc[qi : qi + 1],
                    self.codes_i8[flat_w],
                    self.scales_f16[flat_w],
                    use_neon=self.use_neon,
                )
                d_sc = score_document_windows(w_sc, sub_offsets, agg=eff_agg, hybrid_alpha=eff_alpha)[0]
                take = min(int(k), cand_docs.size)
                order = np.lexsort((cand_docs, -d_sc))[:take]
                out_idx[qi, :take] = cand_docs[order]
                out_sc[qi, :take] = d_sc[order]
            return (out_idx[0], out_sc[0]) if was_1d else (out_idx, out_sc)

        scores = self.score_all(q_trunc, agg=agg, hybrid_alpha=hybrid_alpha)
        out_idx = np.full((q_n, k), -1, dtype=np.int32)
        out_sc = np.full((q_n, k), -1e30, dtype=np.float32)
        doc_ids = np.arange(self.num_docs, dtype=np.int32)
        for qi in range(q_n):
            row = scores[qi]
            if eff_k < self.num_docs:
                part = np.argpartition(-row, eff_k - 1)[:eff_k]
                sub_sc = row[part]
                sub_id = doc_ids[part]
                order = np.lexsort((sub_id, -sub_sc))
                out_idx[qi, :eff_k] = sub_id[order]
                out_sc[qi, :eff_k] = sub_sc[order]
            else:
                order = np.lexsort((doc_ids, -row))[:eff_k]
                out_idx[qi, :eff_k] = doc_ids[order]
                out_sc[qi, :eff_k] = row[order]
        return (out_idx[0], out_sc[0]) if was_1d else (out_idx, out_sc)


def build_engine_index(
    engine: Any,
    corpus: Any,
    *,
    doc_offsets: np.ndarray | list[int] | tuple[int, ...] | None = None,
    mode: IndexMode = "auto",
    mrl_dim: int | None = None,
    binary_bits: int | None = None,
    agg: WindowAgg = "maxsim",
    hybrid_alpha: float = 0.7,
    encode_mode: str | None = None,
    **encode_kwargs: Any,
) -> CompactRetrievalIndex:
    """DRY helper that encodes `corpus` (or accepts pre-encoded `np.ndarray`) and builds `CompactRetrievalIndex`."""
    is_mrl = bool(getattr(engine, "is_mrl_trained", False))
    if isinstance(corpus, np.ndarray):
        vecs = corpus
    else:
        kw = dict(encode_kwargs)
        if encode_mode is not None:
            kw["mode"] = encode_mode
        if hasattr(engine, "encode_documents"):
            vecs = engine.encode_documents(corpus, **kw)
        else:
            vecs = engine.encode_corpus(corpus, **kw)
    return CompactRetrievalIndex(
        vecs,
        doc_offsets=doc_offsets,
        mode=mode,
        is_mrl_trained=is_mrl,
        mrl_dim=mrl_dim,
        binary_bits=binary_bits,
        agg=agg,
        hybrid_alpha=hybrid_alpha,
    )
