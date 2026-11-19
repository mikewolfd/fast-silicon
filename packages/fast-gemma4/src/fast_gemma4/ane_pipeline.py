"""Concurrent CoreML ANE Early14/Early23 prefix prefill runner and zero-copy IOSurface -> Metal KV_i8 bridge."""

from __future__ import annotations

import collections
import ctypes
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Sequence
import warnings

import mlx.core as mx
import numpy as np

from .model import FastGemma4Model

ANE_CHUNK_S = 256
ANE_P_SWA = 512
ANE_P_FULL = 1920
ANE_MAX_CHUNKS = 3  # `AnePrefixWorker` only: `gemma4_px_run_batch` stitches at most two earlier chunks
ANE_RING_MIN_CHUNK_TOKENS = 64  # Spill <64-token tails to GPU to avoid wasting full S=256 ANE slots


def ane_ring_chunk_count(lcp: int, S: int = ANE_CHUNK_S, P_full: int = ANE_P_FULL) -> int:
    """Chunks the prefill ring can run after an `lcp`-token shared prefix.

    Chunk `k` starts at position `lcp + k * S`, and its full-attention past, every position before it, must fit
    the bundle's `P_full` input: 3 chunks after the old 1,408-token prefix, 7 (1,792 suffix tokens) after 278.
    """
    return 0 if lcp > P_full else (P_full - lcp) // S + 1


def make_chunk_geom(
    off: int,
    S: int = ANE_CHUNK_S,
    P_swa: int = ANE_P_SWA,
    P_full: int = ANE_P_FULL,
    *,
    hd_swa: int = 256,
    hd_full: int = 512,
    window_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build CoreML ANE FP16 RoPE tables and causal/SWA masks for chunk starting at global offset `off`."""
    pos = np.arange(off, off + S, dtype=np.float32)[:, None]
    inv_swa = 1.0 / (10000.0 ** (np.arange(0, hd_swa, 2, dtype=np.float32) / float(hd_swa)))
    ang_swa = np.concatenate([pos * inv_swa[None, :], pos * inv_swa[None, :]], axis=-1)
    cos_swa = np.ascontiguousarray(np.cos(ang_swa), dtype=np.float16)
    sin_swa = np.ascontiguousarray(np.sin(ang_swa), dtype=np.float16)

    rotary_full = max(2, hd_full // 4)
    inv_full_part = 1.0 / (1000000.0 ** (np.arange(0, rotary_full, 2, dtype=np.float32) / float(hd_full)))
    inv_full = np.concatenate([inv_full_part, np.zeros((hd_full // 2) - len(inv_full_part), dtype=np.float32)])
    ang_full = np.concatenate([pos * inv_full[None, :], pos * inv_full[None, :]], axis=-1)
    cos_full = np.ascontiguousarray(np.cos(ang_full), dtype=np.float16)
    sin_full = np.ascontiguousarray(np.sin(ang_full), dtype=np.float16)

    q_pos = (off + np.arange(S))[:, None]
    k_past_pos_swa = (off - P_swa + np.arange(P_swa))[None, :]
    k_new_pos = (off + np.arange(S))[None, :]
    k_all_swa = np.concatenate([k_past_pos_swa, k_new_pos], axis=1)
    vis_swa = (k_all_swa >= 0) & (k_all_swa <= q_pos) & ((q_pos - k_all_swa) < window_size)
    mask_swa = np.ascontiguousarray(np.where(vis_swa, 0.0, -10000.0), dtype=np.float16)

    past_idx = np.arange(P_full)[None, :]
    vis_full_past = past_idx < off
    vis_full_new = np.arange(S)[None, :] <= np.arange(S)[:, None]
    vis_full = np.concatenate([np.broadcast_to(vis_full_past, (S, P_full)), vis_full_new], axis=1)
    mask_full = np.ascontiguousarray(np.where(vis_full, 0.0, -10000.0), dtype=np.float16)
    return cos_swa, sin_swa, cos_full, sin_full, mask_swa, mask_full


def validate_ane_kv_finite(*arrays: np.ndarray, max_abs: float = 65504.0) -> bool:
    """Return True iff every ANE FP16/BF16 KV array contains only finite values within `[-max_abs, max_abs]`."""
    for arr in arrays:
        if arr is None or arr.size == 0:
            continue
        if not bool(np.all(np.isfinite(arr))):
            return False
        if float(np.max(np.abs(arr))) > max_abs:
            return False
    return True


def validate_slot_kv_scales_finite(
    cache: dict[str, Any],
    target_slots: Sequence[int],
    *,
    max_scale: float = 65504.0,
) -> bool:
    """Validate that scattered `KV_i8` scales (or BF16 KV tensors) at `target_slots` contain no `NaN`/`Inf`."""
    slots = [int(s) for s in target_slots]
    if not slots:
        return True
    layers = cache.get("layers", ())
    if not layers:
        return True

    use_i8 = bool(cache.get("kv_i8", False))
    mx_flags: list[mx.array] = []
    slot_mx: mx.array | None = None

    for cl in layers:
        if use_i8 and "k_s" in cl and "v_s" in cl:
            ks_raw, vs_raw = cl["k_s"], cl["v_s"]
        elif "k" in cl and "v" in cl:
            ks_raw, vs_raw = cl["k"], cl["v"]
        else:
            continue

        if isinstance(ks_raw, np.ndarray) or isinstance(vs_raw, np.ndarray):
            ks_sub = np.asarray(ks_raw)[slots]
            vs_sub = np.asarray(vs_raw)[slots]
            if not (np.all(np.isfinite(ks_sub)) and np.all(np.isfinite(vs_sub))):
                return False
            if ks_sub.size and float(np.max(np.abs(ks_sub))) > max_scale:
                return False
            if vs_sub.size and float(np.max(np.abs(vs_sub))) > max_scale:
                return False
            if use_i8 and "k_s" in cl:
                if ks_sub.size and float(np.min(ks_sub)) < 0.0:
                    return False
                if vs_sub.size and float(np.min(vs_sub)) < 0.0:
                    return False
        else:
            if slot_mx is None:
                slot_mx = mx.array(slots, dtype=mx.int32)
            ks_sub = ks_raw[slot_mx]
            vs_sub = vs_raw[slot_mx]
            flag = (
                mx.any(mx.isnan(ks_sub))
                | mx.any(mx.isnan(vs_sub))
                | mx.any(mx.abs(ks_sub) > max_scale)
                | mx.any(mx.abs(vs_sub) > max_scale)
            )
            if use_i8 and "k_s" in cl:
                flag = flag | mx.any(ks_sub < 0.0) | mx.any(vs_sub < 0.0)
            mx_flags.append(flag)

    if not mx_flags:
        return True
    any_bad = mx_flags[0]
    for f in mx_flags[1:]:
        any_bad = any_bad | f
    mx.eval(any_bad)
    return not bool(any_bad.item())


def is_w8_bundle_calibrated(mlmodelc_path: str | Path | None) -> bool:
    """Return True iff `ane_calibration_manifest.json` certifies the W8 bundle as domain-calibrated or QAT-validated."""
    if mlmodelc_path is None:
        return False
    mp = Path(mlmodelc_path)
    for manifest_p in (mp.parent / "ane_calibration_manifest.json", mp / "ane_calibration_manifest.json"):
        if not manifest_p.is_file():
            continue
        try:
            data = json.loads(manifest_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bundles = data.get("bundles", {})
        entry = bundles.get(mp.name) if isinstance(bundles, dict) else None
        if not isinstance(entry, dict):
            entry = data if data.get("bundle_name") == mp.name else None
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("qat_validated", False)):
            return True
        src = str(entry.get("calibration_source", "")).strip().lower()
        docs = int(entry.get("calibration_docs", 0))
        if bool(entry.get("awq_mse", False)) and src in ("corpus", "domain_corpus", "file", "fixtures") and docs >= 8:
            return True
    return False


def _stack_sys_kv_fp16(
    prefix_kv: dict[str, Any],
    key: str,
    layer_indices: Sequence[int],
) -> np.ndarray:
    """Stack prefix KV tensors into contiguous FP16 `(n_layers, n_kv, l_sys, hd)`."""
    mats: list[np.ndarray] = []
    for l_idx in layer_indices:
        arr = np.asarray(prefix_kv[key][l_idx].astype(mx.float16), dtype=np.float16)
        if arr.ndim == 2:
            arr = arr[None, :, :]  # legacy (l_sys, hd) -> (1, l_sys, hd)
        elif arr.ndim == 3:
            arr = np.transpose(arr, (1, 0, 2))  # (l_sys, n_kv, hd) -> (n_kv, l_sys, hd)
        else:
            raise ValueError(f"Expected 2D or 3D prefix_kv[{key!r}][{l_idx}], got shape {arr.shape}")
        mats.append(arr)
    return np.ascontiguousarray(np.stack(mats, axis=0), dtype=np.float16)


def _mx_uma_ptr_and_nbytes(arr: mx.array) -> tuple[int, int]:
    """Extract unified-memory virtual address and byte size without retaining a `memoryview` on `arr`."""
    mv = memoryview(arr)
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        nbytes = int(mv.nbytes)
        return addr, nbytes
    finally:
        del mv


_PLAN_INSPECT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def inspect_ane_bundle_plan(
    lib_path: str | Path,
    mlmodelc_path: str | Path,
    *,
    compute_mode: int = 0,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Inspect CoreML `MLComputePlan` with 30s native timeout guard + check MIL quantization ops."""
    lp = Path(lib_path)
    mp = Path(mlmodelc_path)
    if not lp.exists():
        raise FileNotFoundError(f"Missing native ANE library: {lp}")
    if not mp.exists():
        raise FileNotFoundError(f"Missing CoreML bundle: {mp}")

    mil_path = mp / "model.mil"
    mtime_ns = mil_path.stat().st_mtime_ns if mil_path.exists() else mp.stat().st_mtime_ns
    cache_key = (str(mp.resolve()), int(mtime_ns), int(compute_mode))
    if use_cache and cache_key in _PLAN_INSPECT_CACHE:
        return dict(_PLAN_INSPECT_CACHE[cache_key])

    has_legacy_lut = False
    has_linear_w8 = False
    if mil_path.exists():
        mil_txt = mil_path.read_text(encoding="utf-8", errors="ignore")
        has_legacy_lut = "constexpr_lut_to_dense" in mil_txt
        has_linear_w8 = "constexpr_blockwise_shift_scale" in mil_txt

    lib = ctypes.CDLL(str(lp))
    i32p = ctypes.POINTER(ctypes.c_int)
    lib.gemma4_px_inspect_plan_ex.argtypes = [ctypes.c_char_p, ctypes.c_int, i32p, i32p, i32p, i32p]
    lib.gemma4_px_inspect_plan_ex.restype = ctypes.c_int
    ane = ctypes.c_int(0)
    cpu_rt = ctypes.c_int(0)
    cpu_ce = ctypes.c_int(0)
    gpu = ctypes.c_int(0)
    rc = -1
    max_tries = 2 if (has_linear_w8 and not has_legacy_lut) else 1
    for attempt in range(max_tries):
        rc = int(
            lib.gemma4_px_inspect_plan_ex(
                str(mp.resolve()).encode("utf-8"),
                int(compute_mode),
                ctypes.byref(ane),
                ctypes.byref(cpu_rt),
                ctypes.byref(cpu_ce),
                ctypes.byref(gpu),
            )
        )
        if rc >= 0:
            break
        if attempt + 1 < max_tries:
            time.sleep(1.5)

    res = {
        "ok": rc >= 0,
        "ane_ops": ane.value,
        "cpu_runtime_ops": cpu_rt.value,
        "cpu_constexpr_ops": cpu_ce.value,
        "gpu_ops": gpu.value,
        "has_legacy_kmeans_lut": has_legacy_lut,
        "has_linear_w8": has_linear_w8,
    }
    if rc >= 0:
        _PLAN_INSPECT_CACHE[cache_key] = dict(res)
    return res


def is_ane_resident_bundle(lib_path: str | Path, mlmodelc_path: str | Path | None) -> bool:
    """Return True iff `mlmodelc_path` exists, has 0 CPU runtime ops on ANE, and is not legacy LUT8.

    Weight format is not checked: a Linear-W8 and an FP16 bundle both pass when every op is placed on the ANE.
    """
    if mlmodelc_path is None:
        return False
    lp = Path(lib_path)
    mp = Path(mlmodelc_path)
    if not lp.exists() or not mp.exists():
        return False
    mil_path = mp / "model.mil"
    if mil_path.exists():
        mil_txt = mil_path.read_text(encoding="utf-8", errors="ignore")
        if "constexpr_lut_to_dense" in mil_txt:
            return False
    try:
        info = inspect_ane_bundle_plan(lp, mp, compute_mode=0)
    except Exception:
        return False
    return bool(
        info["ok"]
        and info["ane_ops"] > 0
        and info["cpu_runtime_ops"] == 0
        and not info["has_legacy_kmeans_lut"]
    )


class SlotCacheMetalBinding:
    """Zero-copy 16 KiB page-aligned `MTLBuffer` binding over MLX `make_slot_cache(kv_i8=True)` layers."""

    def __init__(self, runner: "AnePrefixRunner", model: FastGemma4Model, cache: dict[str, Any]) -> None:
        if not cache.get("kv_i8", False):
            raise ValueError("Zero-copy IOSurface -> Metal scatter requires kv_i8=True slot cache")
        self._runner = runner
        self._layers_ref = cache["layers"]  # keep mx.arrays alive without holding a memoryview
        B_slots = int(cache["batch_size"])
        swa_cap = int(cache["swa_cap"])
        full_cap = int(cache["full_cap"])

        n_swa = len(model.swa_layers)
        n_full = len(model.full_layers)
        swa_ptrs = np.empty(n_swa * 4, dtype=np.uint64)
        swa_nbytes = np.empty(n_swa * 4, dtype=np.uint64)
        for i, l_idx in enumerate(model.swa_layers):
            c = self._layers_ref[l_idx]
            for j, key in enumerate(("k", "k_s", "v", "v_s")):
                addr, nb = _mx_uma_ptr_and_nbytes(c[key])
                swa_ptrs[i * 4 + j] = addr
                swa_nbytes[i * 4 + j] = nb

        full_ptrs = np.empty(n_full * 4, dtype=np.uint64)
        full_nbytes = np.empty(n_full * 4, dtype=np.uint64)
        for i, l_idx in enumerate(model.full_layers):
            c = self._layers_ref[l_idx]
            for j, key in enumerate(("k", "k_s", "v", "v_s")):
                addr, nb = _mx_uma_ptr_and_nbytes(c[key])
                full_ptrs[i * 4 + j] = addr
                full_nbytes[i * 4 + j] = nb

        u64p = ctypes.POINTER(ctypes.c_uint64)
        self._bind_handle = runner._lib.gemma4_px_bind_slot_cache_i8(
            runner._handle,
            B_slots,
            swa_cap,
            full_cap,
            n_swa,
            swa_ptrs.ctypes.data_as(u64p),
            swa_nbytes.ctypes.data_as(u64p),
            n_full,
            full_ptrs.ctypes.data_as(u64p),
            full_nbytes.ctypes.data_as(u64p),
        )
        if not self._bind_handle:
            raise RuntimeError("gemma4_px_bind_slot_cache_i8 failed to bind MLX UMA slot-cache buffers")

    def __enter__(self) -> "SlotCacheMetalBinding":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        bh = getattr(self, "_bind_handle", None)
        if bh:
            self._runner._lib.gemma4_px_unbind_slot_cache_i8(bh)
            self._bind_handle = None

    def __del__(self) -> None:
        self.close()


class AnePrefixRunner:
    """Thread-safe `ctypes` wrapper around `libgemma4_ane_prefix.dylib` with zero-copy IOSurface ring."""

    def __init__(
        self,
        lib_path: str | Path,
        mlmodelc_path: str | Path,
        *,
        S: int = ANE_CHUNK_S,
        P_swa: int = ANE_P_SWA,
        P_full: int = ANE_P_FULL,
        max_pipe: int = 4,
        pipe_depth: int | None = None,
        out_ring_slots: int = 8,
    ) -> None:
        if pipe_depth is not None:
            max_pipe = int(pipe_depth)
        self.lib_path = Path(lib_path)
        self.mlmodelc_path = Path(mlmodelc_path)
        if not self.lib_path.exists():
            raise FileNotFoundError(f"Missing native ANE library: {self.lib_path} (run `make native`)")
        if not self.mlmodelc_path.exists():
            raise FileNotFoundError(f"Missing CoreML bundle: {self.mlmodelc_path}")

        self._lib = ctypes.CDLL(str(self.lib_path))
        u16p = ctypes.POINTER(ctypes.c_uint16)
        i32p = ctypes.POINTER(ctypes.c_int32)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        self._u16p = u16p
        self._i32p = i32p

        self._lib.gemma4_px_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.gemma4_px_inspect_plan.restype = ctypes.c_int
        self._lib.gemma4_px_inspect_plan_ex.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.gemma4_px_inspect_plan_ex.restype = ctypes.c_int
        self._lib.gemma4_px_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.gemma4_px_load.restype = ctypes.c_void_p
        self._lib.gemma4_px_load_ex.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.gemma4_px_load_ex.restype = ctypes.c_void_p
        self._lib.gemma4_px_is_zero_copy.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_px_is_zero_copy.restype = ctypes.c_int
        self._lib.gemma4_px_out_ring_size.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_px_out_ring_size.restype = ctypes.c_int
        self._lib.gemma4_px_free.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_px_free.restype = None
        self._lib.gemma4_px_run_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            u16p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.gemma4_px_run_batch.restype = ctypes.c_double
        try:
            run_chunk = self._lib.gemma4_px_run_chunk_into_ring
        except AttributeError as exc:
            raise RuntimeError(
                f"{self.lib_path} predates `gemma4_px_run_chunk_into_ring`; rebuild it with `make native`."
            ) from exc
        run_chunk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, *([u16p] * 12), i32p, i32p, ctypes.c_int]
        run_chunk.restype = ctypes.c_double
        self._lib.gemma4_px_bind_slot_cache_i8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            u64p,
            u64p,
            ctypes.c_int,
            u64p,
            u64p,
        ]
        self._lib.gemma4_px_bind_slot_cache_i8.restype = ctypes.c_void_p
        self._lib.gemma4_px_unbind_slot_cache_i8.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_px_unbind_slot_cache_i8.restype = None
        self._lib.gemma4_px_scatter_ring_kv_i8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            i32p,
            i32p,
            i32p,
            i32p,
        ]
        self._lib.gemma4_px_scatter_ring_kv_i8.restype = ctypes.c_double

        self._handle = self._lib.gemma4_px_load_ex(
            str(self.mlmodelc_path).encode("utf-8"), S, P_swa, P_full, max_pipe, max(4, int(out_ring_slots))
        )
        if not self._handle:
            raise RuntimeError(f"gemma4_px_load_ex failed for {self.mlmodelc_path}")
        self.zero_copy = bool(self._lib.gemma4_px_is_zero_copy(self._handle))
        self.out_ring_size = int(self._lib.gemma4_px_out_ring_size(self._handle))

    def close(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            self._lib.gemma4_px_free(h)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _ptr_u16(self, arr: np.ndarray | None):
        if arr is None:
            return ctypes.cast(0, self._u16p)
        return np.ascontiguousarray(arr, dtype=np.float16).ctypes.data_as(self._u16p)

    def _ptr_i32(self, arr: Sequence[int] | np.ndarray | None):
        if arr is None:
            return ctypes.cast(0, self._i32p)
        a = arr if isinstance(arr, np.ndarray) and arr.dtype == np.int32 and arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr, dtype=np.int32)
        return a.ctypes.data_as(self._i32p), a

    def bind_slot_cache_i8(self, model: FastGemma4Model, cache: dict[str, Any]) -> SlotCacheMetalBinding:
        if not self._handle:
            raise RuntimeError("AnePrefixRunner is closed")
        return SlotCacheMetalBinding(self, model, cache)

    def run_into_ring(
        self,
        B: int,
        chunk_idx: int,
        lcp: int,
        h_in: np.ndarray,
        pli_raw: np.ndarray,
        geom: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        sys_swa_k: np.ndarray | None,
        sys_swa_v: np.ndarray | None,
        sys_full_k: np.ndarray | None,
        sys_full_v: np.ndarray | None,
        prev_ring_ids: np.ndarray | None,
        out_ring_ids: Sequence[int] | np.ndarray,
        *,
        pipe_depth: int = 4,
    ) -> float:
        """Run chunk `chunk_idx` for `B` docs. `prev_ring_ids[j, b]` is the ring slot holding doc `b`'s chunk `j`."""
        if not self._handle:
            raise RuntimeError("AnePrefixRunner is closed")
        if chunk_idx > 0 and (prev_ring_ids is None or np.shape(prev_ring_ids) != (chunk_idx, B)):
            raise ValueError(f"chunk {chunk_idx} of {B} docs needs prev_ring_ids of shape ({chunk_idx}, {B})")
        cs, ss, cf, sf, ms, mf = geom
        p_prev = self._ptr_i32(prev_ring_ids if chunk_idx > 0 else None)
        p_out, _out_keep = self._ptr_i32(out_ring_ids)
        ptr_prev = p_prev[0] if isinstance(p_prev, tuple) else p_prev
        dt = float(
            self._lib.gemma4_px_run_chunk_into_ring(
                self._handle,
                int(B),
                int(chunk_idx),
                int(lcp),
                self._ptr_u16(h_in),
                self._ptr_u16(pli_raw),
                self._ptr_u16(cs),
                self._ptr_u16(ss),
                self._ptr_u16(cf),
                self._ptr_u16(sf),
                self._ptr_u16(sys_swa_k),
                self._ptr_u16(sys_swa_v),
                self._ptr_u16(sys_full_k),
                self._ptr_u16(sys_full_v),
                self._ptr_u16(ms),
                self._ptr_u16(mf),
                ptr_prev,
                p_out,
                int(pipe_depth),
            )
        )
        if dt < 0.0:
            raise RuntimeError(f"gemma4_px_run_chunk_into_ring failed on chunk_idx={chunk_idx}, B={B}")
        return dt

    def scatter_ring_kv_i8(
        self,
        binding: SlotCacheMetalBinding,
        ring_slot_ids: Sequence[int] | np.ndarray,
        doc_slots: Sequence[int] | np.ndarray,
        priv_offs: Sequence[int] | np.ndarray,
        valid_lens: Sequence[int] | np.ndarray,
    ) -> float:
        if not self._handle or not binding._bind_handle:
            raise RuntimeError("AnePrefixRunner or SlotCacheMetalBinding is closed")
        n = len(ring_slot_ids)
        if n == 0:
            return 0.0
        p_r, _kr = self._ptr_i32(ring_slot_ids)
        p_d, _kd = self._ptr_i32(doc_slots)
        p_o, _ko = self._ptr_i32(priv_offs)
        p_v, _kv = self._ptr_i32(valid_lens)
        gpu_sec = float(
            self._lib.gemma4_px_scatter_ring_kv_i8(
                self._handle,
                binding._bind_handle,
                int(n),
                p_r,
                p_d,
                p_o,
                p_v,
            )
        )
        if gpu_sec < 0.0:
            raise RuntimeError("gemma4_px_scatter_ring_kv_i8 failed")
        return gpu_sec

    def run_batch(
        self,
        B: int,
        h_in: np.ndarray,
        pli_raw: np.ndarray,
        cos_swa: np.ndarray,
        sin_swa: np.ndarray,
        cos_full: np.ndarray,
        sin_full: np.ndarray,
        kp_swa: np.ndarray,
        vp_swa: np.ndarray,
        kp_full: np.ndarray,
        vp_full: np.ndarray,
        mask_swa: np.ndarray,
        mask_full: np.ndarray,
        ok_swa: np.ndarray,
        ov_swa: np.ndarray,
        ok_full: np.ndarray,
        ov_full: np.ndarray,
        *,
        shared_past_kv: bool,
        shared_geom: bool = True,
        pipe_depth: int = 4,
    ) -> float:
        if not self._handle:
            raise RuntimeError("AnePrefixRunner is closed")
        dt = float(
            self._lib.gemma4_px_run_batch(
                self._handle,
                B,
                self._ptr_u16(h_in),
                self._ptr_u16(pli_raw),
                self._ptr_u16(cos_swa),
                self._ptr_u16(sin_swa),
                self._ptr_u16(cos_full),
                self._ptr_u16(sin_full),
                self._ptr_u16(kp_swa),
                self._ptr_u16(vp_swa),
                self._ptr_u16(kp_full),
                self._ptr_u16(vp_full),
                self._ptr_u16(mask_swa),
                self._ptr_u16(mask_full),
                self._ptr_u16(ok_swa),
                self._ptr_u16(ov_swa),
                self._ptr_u16(ok_full),
                self._ptr_u16(ov_full),
                int(shared_past_kv),
                int(shared_geom),
                int(pipe_depth),
            )
        )
        if dt < 0.0:
            raise RuntimeError("gemma4_px_run_batch failed")
        return dt


def can_use_ane_prefix(lcp: int, max_suf_len: int) -> bool:
    """Return True if prefix + up to 2..3 ANE chunks fit within `P_FULL=1920` and suffix warrants ANE."""
    if lcp > ANE_P_FULL - 2 * ANE_CHUNK_S:
        return False
    return (max_suf_len - 1) >= 32


class AsyncHybridPrefillRing:
    """Zero-copy background ANE EarlyExit14/Early23 producer + IOSurface -> Metal KV_i8 scatter ring.

    Keeps CPU SME2 strictly parked for standard suffixes (`S_suffix <= 1,200`), streaming
    `pipe_depth=4` ANE chunks directly into unlocked `CVPixelBuffer`/`IOSurface` ring slots
    and scattering them into freed MLX `KV_i8` rolling slots via 2 Metal dispatches (`0` CPU copies).
    Short suffix tails `< min_chunk_tokens` (default `64`) spill cleanly to GPU BF16 prefill.
    """

    min_chunk_tokens: int = ANE_RING_MIN_CHUNK_TOKENS

    def __init__(
        self,
        model: FastGemma4Model,
        runner: AnePrefixRunner,
        prefix_kv: dict[str, Any],
        w_emb: np.ndarray,
        w_pli14: np.ndarray,
        *,
        min_chunk_tokens: int = ANE_RING_MIN_CHUNK_TOKENS,
        secondary_runner: AnePrefixRunner | None = None,
        secondary_chunk_s: int | None = None,
    ) -> None:
        self.model = model
        self.runner = runner
        self.w_emb = w_emb
        self.w_pli14 = w_pli14
        self.min_chunk_tokens = max(1, int(min_chunk_tokens))
        self.secondary_runner = secondary_runner
        self.secondary_chunk_s = int(secondary_chunk_s) if secondary_chunk_s else None
        self.n_kv = int(getattr(model, "n_kv_heads", 1))
        self.hd_swa = int(getattr(model, "swa_head_dim", 256))
        self.hd_full = int(getattr(model, "full_head_dim", 512))
        self.window_size = int(getattr(model, "window_size", 512))
        self.lcp = int(prefix_kv["l_sys"])
        if self.lcp > ANE_P_FULL - ANE_CHUNK_S:
            raise ValueError(f"Prefix length {self.lcp} exceeds CoreML P_FULL={ANE_P_FULL}")

        self.sys_ks = _stack_sys_kv_fp16(prefix_kv, "k_sys", model.swa_layers)
        self.sys_vs = _stack_sys_kv_fp16(prefix_kv, "v_sys", model.swa_layers)
        n_fp = len(model.full_layers) - 1
        if self.lcp > 0 and n_fp > 0:
            self.sys_kf: np.ndarray | None = _stack_sys_kv_fp16(prefix_kv, "k_sys", model.full_layers[:n_fp])
            self.sys_vf: np.ndarray | None = _stack_sys_kv_fp16(prefix_kv, "v_sys", model.full_layers[:n_fp])
        else:
            self.sys_kf = None
            self.sys_vf = None

        self.sys_ks_512 = self._tail_swa(self.sys_ks, ANE_P_SWA)
        self.sys_vs_512 = self._tail_swa(self.sys_vs, ANE_P_SWA)
        self.sys_ks_256 = self._tail_swa(self.sys_ks, ANE_CHUNK_S)
        self.sys_vs_256 = self._tail_swa(self.sys_vs, ANE_CHUNK_S)

        self.n_chunks = ane_ring_chunk_count(self.lcp)
        self.geoms: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
            k: make_chunk_geom(
                self.lcp + k * ANE_CHUNK_S,
                ANE_CHUNK_S,
                ANE_P_SWA,
                ANE_P_FULL,
                hd_swa=self.hd_swa,
                hd_full=self.hd_full,
                window_size=self.window_size,
            )
            for k in range(self.n_chunks)
        }

        self._ring_lock = threading.Lock()
        self._free_ring_q: queue.Queue[int] = queue.Queue()
        for r_id in range(self.runner.out_ring_size):
            self._free_ring_q.put(r_id)

    @property
    def bucket_sizes(self) -> tuple[int, ...]:
        if self.secondary_runner is not None and self.secondary_chunk_s:
            return (ANE_CHUNK_S, self.secondary_chunk_s)
        return (ANE_CHUNK_S,)

    def _tail_swa(self, arr: np.ndarray, n_take: int) -> np.ndarray:
        n_swa, n_kv, _, hd = arr.shape
        if self.lcp >= n_take:
            return np.ascontiguousarray(arr[:, :, -n_take:, :], dtype=np.float16)
        out = np.zeros((n_swa, n_kv, n_take, hd), dtype=np.float16)
        if self.lcp > 0:
            out[:, :, n_take - self.lcp :, :] = arr
        return out

    def acquire_ring_slots(self, count: int, stop_event: threading.Event | None = None) -> list[int]:
        acquired: list[int] = []
        try:
            for _ in range(count):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise RuntimeError("AsyncHybridPrefillRing stopped while acquiring ring slots")
                    try:
                        acquired.append(self._free_ring_q.get(timeout=0.05))
                        break
                    except queue.Empty:
                        continue
            return acquired
        except BaseException:
            self.release_ring_slots(acquired)
            raise

    def release_ring_slots(self, slot_ids: Sequence[int]) -> None:
        for r_id in slot_ids:
            self._free_ring_q.put(int(r_id))

    def prefill_cohort_into_ring(
        self,
        doc_ids: list[int],
        sufs: list[list[int]],
        *,
        pipe_depth: int = 4,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run up to `n_chunks` ANE chunks per doc directly into unlocked IOSurface ring slots.

        A doc keeps one ring slot per chunk until the cohort is scattered: later chunks read the earlier ones as
        their past, so a cohort of `B` docs can hold `B * n_chunks` slots at once. Trailing tails shorter than
        `min_chunk_tokens` (default 64) are spilled cleanly to GPU BF16 prefill instead of wasting a 256-tok bucket.
        """
        B = len(doc_ids)
        covered = [0] * B
        chunk_records: list[tuple[int, list[int], list[int], list[int]]] = []
        ring_by_chunk: list[dict[int, int]] = []  # chunk -> doc row -> ring slot
        min_tok = max(1, int(getattr(self, "min_chunk_tokens", ANE_RING_MIN_CHUNK_TOKENS)))
        chunk_keys = sorted(self.geoms.keys())
        need = 0
        for k in chunk_keys:
            c_off = k * ANE_CHUNK_S
            if self.lcp + c_off > ANE_P_FULL:
                break
            need += sum(1 for s in sufs if (len(s) - 1 - c_off) >= min_tok)
        if need > self.runner.out_ring_size:
            # Slots are only freed by scattering this cohort, so waiting for more would never return.
            raise ValueError(
                f"Cohort of {B} docs needs {need} ring slots at once; the runner has {self.runner.out_ring_size}."
            )
        ane_sec = 0.0
        allocated_all: list[int] = []

        try:
            for k in chunk_keys:
                c_off = k * ANE_CHUNK_S
                if self.lcp + c_off > ANE_P_FULL:
                    break
                # Gate partial chunks at `min_chunk_tokens` (default 64); shorter tails spill to GPU BF16 prefill.
                act_ris = [ri for ri in range(B) if (len(sufs[ri]) - 1 - c_off) >= min_tok]
                if not act_ris:
                    break

                n_act = len(act_ris)
                ring_ids = self.acquire_ring_slots(n_act, stop_event=stop_event)
                allocated_all.extend(ring_ids)

                pad_toks = np.zeros((n_act, ANE_CHUNK_S), dtype=np.int32)
                vlens: list[int] = []
                for j, ri in enumerate(act_ris):
                    s_toks = sufs[ri]
                    vl = min(ANE_CHUNK_S, len(s_toks) - 1 - c_off)
                    vlens.append(vl)
                    pad_toks[j, :vl] = s_toks[c_off : c_off + vl]
                    if vl < ANE_CHUNK_S:
                        pad_toks[j, vl:] = s_toks[c_off + vl - 1]

                h_in = np.ascontiguousarray(np.transpose(self.w_emb[pad_toks], (0, 2, 1)), dtype=np.float16)
                pli_raw = np.ascontiguousarray(np.transpose(self.w_pli14[pad_toks], (0, 2, 1)), dtype=np.float16)

                # Sliding-window past: the prefix tail for chunk 0, prefix tail + chunk 0 for chunk 1, then the two
                # chunks before (read from the ring). A doc active here filled every earlier chunk completely.
                if k == 0:
                    swa_k, swa_v = self.sys_ks_512, self.sys_vs_512
                elif k == 1:
                    swa_k, swa_v = self.sys_ks_256, self.sys_vs_256
                else:
                    swa_k, swa_v = None, None
                prev = (
                    np.asarray([[ring_by_chunk[j][ri] for ri in act_ris] for j in range(k)], dtype=np.int32)
                    if k > 0
                    else None
                )

                dt = self.runner.run_into_ring(
                    n_act,
                    k,
                    self.lcp,
                    h_in,
                    pli_raw,
                    self.geoms[k],
                    swa_k,
                    swa_v,
                    self.sys_kf,
                    self.sys_vf,
                    prev,
                    ring_ids,
                    pipe_depth=pipe_depth,
                )
                ane_sec += dt
                ring_by_chunk.append(dict(zip(act_ris, ring_ids)))

                for j, ri in enumerate(act_ris):
                    covered[ri] = c_off + vlens[j]
                chunk_records.append((c_off, act_ris, vlens, ring_ids))

            return {
                "doc_ids": doc_ids,
                "sufs": sufs,
                "chunk_records": chunk_records,
                "covered": covered,
                "ring_slots_held": allocated_all,
                "ane_sec": ane_sec,
            }
        except BaseException:
            self.release_ring_slots(allocated_all)
            raise

    def scatter_cohort_to_slot_cache(
        self,
        binding: SlotCacheMetalBinding,
        item: dict[str, Any],
        target_slots: Sequence[int],
    ) -> float:
        """Scatter all ANE IOSurface chunks in `item` directly into `target_slots` and free the ring slots."""
        flat_ring: list[int] = []
        flat_doc_slot: list[int] = []
        flat_off: list[int] = []
        flat_vlen: list[int] = []
        try:
            for c_off, act_ris, vlens, ring_ids in item["chunk_records"]:
                for j, ri in enumerate(act_ris):
                    flat_ring.append(ring_ids[j])
                    flat_doc_slot.append(int(target_slots[ri]))
                    flat_off.append(int(c_off))
                    flat_vlen.append(int(vlens[j]))
            if not flat_ring:
                return 0.0
            return self.runner.scatter_ring_kv_i8(binding, flat_ring, flat_doc_slot, flat_off, flat_vlen)
        finally:
            held = item.get("ring_slots_held", ())
            if held:
                self.release_ring_slots(held)
                item["ring_slots_held"] = []


class AnePrefixWorker:
    """Pure CPU + CoreML ANE chunk prefill worker for a shared prefix `KV_sys` (zero GPU contention)."""

    def __init__(
        self,
        model: FastGemma4Model,
        runner: AnePrefixRunner,
        prefix_kv: dict[str, Any],
        w_emb: np.ndarray,
        w_pli14: np.ndarray,
        *,
        min_chunk_tokens: int = 32,
    ) -> None:
        self.model = model
        self.runner = runner
        self.w_emb = w_emb
        self.w_pli14 = w_pli14
        self.min_chunk_tokens = max(1, int(min_chunk_tokens))
        self.n_swa = len(model.swa_layers)
        self.n_full = len(model.full_layers)
        self.n_full_past = max(0, self.n_full - 1)
        self.n_kv = int(getattr(model, "n_kv_heads", 1))
        self.hd_swa = int(getattr(model, "swa_head_dim", 256))
        self.hd_full = int(getattr(model, "full_head_dim", 512))
        self.window_size = int(getattr(model, "window_size", 512))
        self.lcp = int(prefix_kv["l_sys"])
        if self.lcp > ANE_P_FULL - ANE_CHUNK_S:
            raise ValueError(f"Prefix length {self.lcp} exceeds CoreML P_FULL={ANE_P_FULL}")

        self.sys_ks = _stack_sys_kv_fp16(prefix_kv, "k_sys", model.swa_layers)
        self.sys_vs = _stack_sys_kv_fp16(prefix_kv, "v_sys", model.swa_layers)
        self.sys_kf = _stack_sys_kv_fp16(prefix_kv, "k_sys", model.full_layers[: self.n_full_past])
        self.sys_vf = _stack_sys_kv_fp16(prefix_kv, "v_sys", model.full_layers[: self.n_full_past])

        self.geoms = {}
        for k in range(ANE_MAX_CHUNKS):
            off = self.lcp + k * ANE_CHUNK_S
            if off <= ANE_P_FULL:
                self.geoms[k] = make_chunk_geom(
                    off,
                    ANE_CHUNK_S,
                    ANE_P_SWA,
                    ANE_P_FULL,
                    hd_swa=self.hd_swa,
                    hd_full=self.hd_full,
                    window_size=self.window_size,
                )

    def _sys_swa_tail512(self, k_arr: np.ndarray, n_take: int) -> np.ndarray:
        """Return rightmost `n_take` slots of `sys_ks/sys_vs`, zero-left-padded if `lcp < n_take`."""
        n_swa, n_kv, _, hd = k_arr.shape
        if self.lcp >= n_take:
            return np.ascontiguousarray(k_arr[:, :, -n_take:, :], dtype=np.float16)
        out = np.zeros((n_swa, n_kv, n_take, hd), dtype=np.float16)
        if self.lcp > 0:
            out[:, :, n_take - self.lcp :, :] = k_arr
        return out

    def run_wave_cpu_ane(self, suf_ids_list: Sequence[Sequence[int]]) -> dict[str, Any]:
        """Run pure CPU embed lookup + CoreML ANE Early14/Early23 prefill for a wave of documents."""
        t0 = time.perf_counter()
        B = len(suf_ids_list)
        lcp = self.lcp
        n_swa = self.n_swa
        n_full = self.n_full
        n_fp = self.n_full_past
        n_kv = self.n_kv
        hd_s = self.hd_swa
        hd_f = self.hd_full

        chunks_out: dict[int, tuple[list[int], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        covered = [0] * B
        ane_ks_prev = np.empty((B, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
        ane_vs_prev = np.empty((B, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
        ane_kf_prev = np.empty((B, n_fp, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)
        ane_vf_prev = np.empty((B, n_fp, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)
        ane_ks_prev1 = np.empty((B, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
        ane_vs_prev1 = np.empty((B, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
        ane_kf_prev1 = np.empty((B, n_fp, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)
        ane_vf_prev1 = np.empty((B, n_fp, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)

        t_emb = 0.0
        t_ane = 0.0
        non_finite_detected = False
        for k in sorted(self.geoms.keys()):
            c0 = k * ANE_CHUNK_S
            if lcp + c0 > ANE_P_FULL:
                break
            act = [b for b in range(B) if (len(suf_ids_list[b]) - 1 - c0) >= self.min_chunk_tokens]
            if not act:
                break
            n_act = len(act)
            pad_toks = np.zeros((n_act, ANE_CHUNK_S), dtype=np.int32)
            vlen: list[int] = []
            for ri, b in enumerate(act):
                vl = min(ANE_CHUNK_S, len(suf_ids_list[b]) - 1 - c0)
                vlen.append(vl)
                pad_toks[ri, :vl] = suf_ids_list[b][c0 : c0 + vl]

            te0 = time.perf_counter()
            h_in = np.ascontiguousarray(np.transpose(self.w_emb[pad_toks], (0, 2, 1)))
            pli_raw = np.ascontiguousarray(np.transpose(self.w_pli14[pad_toks], (0, 2, 1)))
            t_emb += time.perf_counter() - te0

            cs, ss, cf, sf, ms, mf = self.geoms[k]
            if k == 0:
                kp_s = self._sys_swa_tail512(self.sys_ks, ANE_P_SWA)
                vp_s = self._sys_swa_tail512(self.sys_vs, ANE_P_SWA)
                kp_f = np.zeros((n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                vp_f = np.zeros((n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                if lcp > 0:
                    kp_f[:, :, :lcp, :] = self.sys_kf
                    vp_f[:, :, :lcp, :] = self.sys_vf
                shared_past = True
            elif k == 1:
                kp_s = np.empty((n_act, n_swa, n_kv, ANE_P_SWA, hd_s), dtype=np.float16)
                vp_s = np.empty((n_act, n_swa, n_kv, ANE_P_SWA, hd_s), dtype=np.float16)
                kp_f = np.zeros((n_act, n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                vp_f = np.zeros((n_act, n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                kp_s[:, :, :, :ANE_CHUNK_S, :] = self._sys_swa_tail512(self.sys_ks, ANE_CHUNK_S)
                vp_s[:, :, :, :ANE_CHUNK_S, :] = self._sys_swa_tail512(self.sys_vs, ANE_CHUNK_S)
                if lcp > 0:
                    kp_f[:, :, :, :lcp, :] = self.sys_kf
                    vp_f[:, :, :, :lcp, :] = self.sys_vf
                act_idx = np.asarray(act, dtype=np.int32)
                kp_s[:, :, :, ANE_CHUNK_S:, :] = ane_ks_prev[act_idx]
                vp_s[:, :, :, ANE_CHUNK_S:, :] = ane_vs_prev[act_idx]
                kp_f[:, :, :, lcp : lcp + ANE_CHUNK_S, :] = ane_kf_prev[act_idx]
                vp_f[:, :, :, lcp : lcp + ANE_CHUNK_S, :] = ane_vf_prev[act_idx]
                shared_past = False
            else:
                kp_s = np.empty((n_act, n_swa, n_kv, ANE_P_SWA, hd_s), dtype=np.float16)
                vp_s = np.empty((n_act, n_swa, n_kv, ANE_P_SWA, hd_s), dtype=np.float16)
                kp_f = np.zeros((n_act, n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                vp_f = np.zeros((n_act, n_fp, n_kv, ANE_P_FULL, hd_f), dtype=np.float16)
                act_idx = np.asarray(act, dtype=np.int32)
                kp_s[:, :, :, :ANE_CHUNK_S, :] = ane_ks_prev[act_idx]
                kp_s[:, :, :, ANE_CHUNK_S:, :] = ane_ks_prev1[act_idx]
                vp_s[:, :, :, :ANE_CHUNK_S, :] = ane_vs_prev[act_idx]
                vp_s[:, :, :, ANE_CHUNK_S:, :] = ane_vs_prev1[act_idx]
                if lcp > 0:
                    kp_f[:, :, :, :lcp, :] = self.sys_kf
                    vp_f[:, :, :, :lcp, :] = self.sys_vf
                kp_f[:, :, :, lcp : lcp + ANE_CHUNK_S, :] = ane_kf_prev[act_idx]
                vp_f[:, :, :, lcp : lcp + ANE_CHUNK_S, :] = ane_vf_prev[act_idx]
                kp_f[:, :, :, lcp + ANE_CHUNK_S : lcp + 2 * ANE_CHUNK_S, :] = ane_kf_prev1[act_idx]
                vp_f[:, :, :, lcp + ANE_CHUNK_S : lcp + 2 * ANE_CHUNK_S, :] = ane_vf_prev1[act_idx]
                shared_past = False

            ok_s = np.empty((n_act, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
            ov_s = np.empty((n_act, n_swa, n_kv, ANE_CHUNK_S, hd_s), dtype=np.float16)
            ok_f = np.empty((n_act, n_full, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)
            ov_f = np.empty((n_act, n_full, n_kv, ANE_CHUNK_S, hd_f), dtype=np.float16)

            dt = self.runner.run_batch(
                n_act,
                h_in,
                pli_raw,
                cs,
                ss,
                cf,
                sf,
                kp_s,
                vp_s,
                kp_f,
                vp_f,
                ms,
                mf,
                ok_s,
                ov_s,
                ok_f,
                ov_f,
                shared_past_kv=shared_past,
                shared_geom=True,
                pipe_depth=4,
            )
            t_ane += dt

            if not validate_ane_kv_finite(ok_s, ov_s, ok_f, ov_f):
                warnings.warn(
                    f"Non-finite (Inf/NaN) ANE FP16 KV detected at chunk {k}; falling back to GPU BF16 prefill.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                non_finite_detected = True
                chunks_out.clear()
                covered = [0] * B
                break

            act_idx = np.asarray(act, dtype=np.int32)
            if k == 0:
                ane_ks_prev[act_idx] = ok_s
                ane_vs_prev[act_idx] = ov_s
                ane_kf_prev[act_idx] = ok_f[:, :n_fp]
                ane_vf_prev[act_idx] = ov_f[:, :n_fp]
            elif k == 1:
                ane_ks_prev1[act_idx] = ok_s
                ane_vs_prev1[act_idx] = ov_s
                ane_kf_prev1[act_idx] = ok_f[:, :n_fp]
                ane_vf_prev1[act_idx] = ov_f[:, :n_fp]

            chunks_out[k] = (act, vlen, ok_s, ov_s, ok_f, ov_f)
            for ri, b in enumerate(act):
                covered[b] = c0 + vlen[ri]

        return {
            "chunks": chunks_out,
            "covered": covered,
            "non_finite_detected": non_finite_detected,
            "worker_sec": time.perf_counter() - t0,
            "ane_sec": t_ane,
            "emb_sec": t_emb,
        }


def ingest_ane_finish_tail_and_last(
    model: FastGemma4Model,
    cache: dict[str, Any],
    suf_ids_list: Sequence[Sequence[int]],
    ane_res: dict[str, Any],
    lcp: int,
) -> tuple[mx.array, mx.array, float, float, float]:
    """Ingest ANE chunk KVs into GPU cache, finish any short tail via EarlyExit14/23, and run batched LastTok.

    Automatically detects non-finite (`Inf`/`NaN`) ANE outputs and falls back cleanly to GPU BF16 prefill.
    """
    t0 = time.perf_counter()
    B = len(suf_ids_list)
    chunks = ane_res["chunks"]
    covered = list(ane_res["covered"])
    non_finite = bool(ane_res.get("non_finite_detected", False))

    if not non_finite:
        for _, (_, _, ok_s, ov_s, ok_f, ov_f) in chunks.items():
            if not validate_ane_kv_finite(ok_s, ov_s, ok_f, ov_f):
                warnings.warn(
                    "Non-finite (Inf/NaN) ANE KV chunk detected during ingestion; falling back to GPU BF16 prefill.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                non_finite = True
                break

    if non_finite:
        covered = [0] * B
    else:
        for k, (act, vlen, ok_s, ov_s, ok_f, ov_f) in chunks.items():
            c0 = k * ANE_CHUNK_S
            if all(v == ANE_CHUNK_S for v in vlen):
                model.ingest_ane_chunk_kv(cache, act, c0, ok_s, ov_s, ok_f, ov_f, valid_len=ANE_CHUNK_S, eval_cache=False)
            else:
                by_vl: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
                for ri, b in enumerate(act):
                    by_vl[vlen[ri]].append((ri, b))
                for vl, pairs in by_vl.items():
                    ris = np.asarray([p[0] for p in pairs], dtype=np.int32)
                    bs = [p[1] for p in pairs]
                    model.ingest_ane_chunk_kv(
                        cache, bs, c0, ok_s[ris], ov_s[ris], ok_f[ris], ov_f[ris], valid_len=vl, eval_cache=False
                    )
        mx.eval(*[(c["k"], c["v"]) for c in cache["layers"]])
    t_ing = time.perf_counter() - t0

    t_tail0 = time.perf_counter()
    tail_docs = sorted(
        [b for b in range(B) if (len(suf_ids_list[b]) - 1) > covered[b]],
        key=lambda b: -(len(suf_ids_list[b]) - 1 - covered[b]),
    )
    if tail_docs:
        rem_seqs = [list(suf_ids_list[b][covered[b] : len(suf_ids_list[b]) - 1]) for b in tail_docs]
        rem_lens = [len(x) for x in rem_seqs]
        max_rem = rem_lens[0]
        sub_c = model.compact_cache(cache, tail_docs)
        ob_sub = mx.array([lcp + covered[b] for b in tail_docs], dtype=mx.int32)
        for step_t in range(max_rem):
            n_act_t = sum(1 for rl in rem_lens if step_t < rl)
            if n_act_t < sub_c["batch_size"]:
                fin_orig = tail_docs[n_act_t : sub_c["batch_size"]]
                fin_rows = mx.array(fin_orig, dtype=mx.int32)
                for li, cl in enumerate(cache["layers"]):
                    cl["k"][fin_rows] = sub_c["layers"][li]["k"][n_act_t : sub_c["batch_size"]]
                    cl["v"][fin_rows] = sub_c["layers"][li]["v"][n_act_t : sub_c["batch_size"]]
                tail_docs = tail_docs[:n_act_t]
                rem_seqs = rem_seqs[:n_act_t]
                rem_lens = rem_lens[:n_act_t]
                sub_c = model.compact_cache(sub_c, list(range(n_act_t)))
                ob_sub = ob_sub[:n_act_t]
            toks_step = mx.array([[rem_seqs[j][step_t]] for j in range(len(tail_docs))], dtype=mx.int32)
            model.forward_early14_ragged(toks_step, sub_c, ob_sub)  # guard: allow-eval
            ob_sub = ob_sub + 1
            mx.eval(*[(c["k"], c["v"]) for c in sub_c["layers"]], ob_sub)  # guard: allow-eval
        if tail_docs:
            fin_rows = mx.array(tail_docs, dtype=mx.int32)
            for li, cl in enumerate(cache["layers"]):
                cl["k"][fin_rows] = sub_c["layers"][li]["k"]
                cl["v"][fin_rows] = sub_c["layers"][li]["v"]
            mx.eval(*[(c["k"], c["v"]) for c in cache["layers"]])
    t_tail = time.perf_counter() - t_tail0

    t_last0 = time.perf_counter()
    last_toks = mx.array([suf_ids_list[b][-1] for b in range(B)], dtype=mx.int32)[:, None]
    last_off = mx.array([lcp + len(suf_ids_list[b]) - 1 for b in range(B)], dtype=mx.int32)
    lg = model.forward_step_ragged(last_toks, cache, last_off, apply_softcap=True)
    next_off = last_off + 1
    mx.eval(lg, next_off)
    t_last = time.perf_counter() - t_last0
    return lg, next_off, t_ing, t_tail, t_last
