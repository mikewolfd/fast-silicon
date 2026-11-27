"""Unified Apple Silicon BGE-small-en-v1.5 inference engine combining Apple Neural Engine and Metal GPU."""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any, Sequence
import warnings

import mlx.core as mx
import numpy as np

from fast_shared.bench import clear_mlx_arena
from fast_shared.cpu_kernels import CpuBackendMode
from fast_shared.embeddings import set_mlx_cache_limit
from fast_shared.retrieval_index import CompactRetrievalIndex, IndexMode, WindowAgg

from .gpu import FastBgeSmallGPU, pack_sequences
from .paths import ensure_prebuilt_artifacts, load_or_mmap_host_tables, require_runtime_paths, resolve_paths
from .tokenizer import FastBgeTokenizer

_MULTIFUNCTION_CANDIDATES: tuple[str, ...] = (
    "bge_ane_w8a16.mlmodelc",
    "bge_ane_multifunction.mlmodelc",
)

_ANE_BUCKETS: tuple[tuple[int, str, int, int], ...] = (
    (24, "bge_b1_s24_fp16.mlmodelc", 1, 1),
    (64, "bge_b1_s64_fp16.mlmodelc", 1, 4),
    (128, "bge_b1_s128_fp16.mlmodelc", 1, 4),
    (192, "bge_b1_s192_fp16.mlmodelc", 1, 4),
    (256, "bge_b8_s256_fp16.mlmodelc", 8, 4),
    (320, "bge_b1_s320_fp16.mlmodelc", 1, 4),
    (384, "bge_b1_s384_fp16.mlmodelc", 1, 4),
    (448, "bge_b1_s448_fp16.mlmodelc", 1, 4),
    (512, "bge_b1_s512_fp16.mlmodelc", 1, 4),
)

# Single-query B=1 S=256 bucket (included by default in W8A16 multifunction builds; optional on legacy FP16).
_OPTIONAL_QUERY_BUCKETS: tuple[tuple[int, str, int, int], ...] = (
    (256, "bge_b1_s256_fp16.mlmodelc", 1, 1),
)

_QUERY_CAPS: tuple[int, ...] = (24, 64, 128, 192, 256, 320, 384, 448, 512)
_CORPUS_CAPS: tuple[int, ...] = (64, 128, 192, 256, 320, 384, 448, 512)
_BGE_QUERY_BATCH_MIN_SIZE: int = 8
_ANE_TABLE_BIND_LOCK = threading.Lock()
_LAST_BOUND_ANE_TABLE_KEY: tuple[int, int, int, int, int] | None = None


def _apply_mlx_wired_limit() -> int | None:
    """Set MLX wired limit to max_recommended_working_set_size safely; return prior limit or None."""
    set_wired = getattr(mx, "set_wired_limit", None)
    dev_info_fn = getattr(mx, "device_info", None)
    if not callable(set_wired) or not callable(dev_info_fn):
        return None
    try:
        info = dev_info_fn()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            return int(set_wired(rec))
    except Exception:
        return None
    return None


def _restore_mlx_wired_limit(prev_limit: int | None) -> None:
    if prev_limit is None:
        return
    set_wired = getattr(mx, "set_wired_limit", None)
    if callable(set_wired):
        try:
            set_wired(int(prev_limit))
        except Exception:
            pass


def _maybe_clear_mlx_arena(threshold_mib: int = 256) -> None:
    get_cache = getattr(mx, "get_cache_memory", None)
    if callable(get_cache):
        try:
            if int(get_cache()) > (threshold_mib << 20):
                clear_mlx_arena(threshold_mib)
        except Exception:
            pass


def _load_ane_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.bge_ane_init_tables.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.bge_ane_init_tables.restype = ctypes.c_int
    if hasattr(lib, "bge_ane_init_tables_borrowed"):
        lib.bge_ane_init_tables_borrowed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.bge_ane_init_tables_borrowed.restype = ctypes.c_int
    lib.bge_ane_load_bucket.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.bge_ane_load_bucket.restype = ctypes.c_int
    if hasattr(lib, "bge_ane_load_bucket_fn"):
        lib.bge_ane_load_bucket_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.bge_ane_load_bucket_fn.restype = ctypes.c_int
    lib.bge_ane_query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.bge_ane_query.restype = ctypes.c_double
    lib.bge_ane_encode_bucket.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.bge_ane_encode_bucket.restype = ctypes.c_double
    lib.bge_ane_free_all.argtypes = []
    lib.bge_ane_free_all.restype = None
    if hasattr(lib, "bge_ane_free_bucket"):
        lib.bge_ane_free_bucket.argtypes = [ctypes.c_int]
        lib.bge_ane_free_bucket.restype = None
    if hasattr(lib, "bge_ane_clear_borrowed_tables"):
        lib.bge_ane_clear_borrowed_tables.argtypes = [ctypes.c_void_p]
        lib.bge_ane_clear_borrowed_tables.restype = None
    if hasattr(lib, "bge_ane_inspect_plan"):
        lib.bge_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.bge_ane_inspect_plan.restype = None
    if hasattr(lib, "bge_ane_get_plan_counts"):
        lib.bge_ane_get_plan_counts.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.bge_ane_get_plan_counts.restype = ctypes.c_int
    if hasattr(lib, "bge_ane_get_plan_counts_fn"):
        lib.bge_ane_get_plan_counts_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.bge_ane_get_plan_counts_fn.restype = ctypes.c_int
    if hasattr(lib, "bge_ane_last_error"):
        lib.bge_ane_last_error.argtypes = []
        lib.bge_ane_last_error.restype = ctypes.c_char_p
    return lib


def _ane_err_detail(lib: ctypes.CDLL | None) -> str:
    if lib is not None and hasattr(lib, "bge_ane_last_error"):
        raw = lib.bge_ane_last_error()
        if raw:
            return raw.decode("utf-8", errors="replace")
    return ""


class FastBgeSmall:
    """Heterogeneous Apple Silicon BGE-small-en-v1.5 engine (Metal GPU + 16-core ANE)."""

    dim: int = 384
    max_dim: int = 384
    is_mrl_trained: bool = False
    model_name: str = "BAAI/bge-small-en-v1.5"
    max_seq_len: int = 512
    ane_query_max_len: int = 512
    cpu_sme_available: bool = False

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        default_mode: str = "hetero",
        warmup: bool = True,
        enable_ane: bool | None = None,
        verify_ane_plan: bool = True,
        mlx_cache_limit_mb: int | None = 512,
        set_wired_limit: bool = True,
        auto_prebuilt: bool = True,
        local_files_only: bool = False,
    ) -> None:
        if auto_prebuilt and home is None:
            cand = resolve_paths(None)
            want_ane = (enable_ane is not False) and (default_mode != "gpu")
            if not cand.weights_npz.exists() or not cand.tokenizer_json.exists() or (
                want_ane and not cand.ane_models_dir.exists()
            ):
                try:
                    ensure_prebuilt_artifacts(
                        None,
                        include_ane=want_ane,
                        local_files_only=local_files_only,
                    )
                except Exception:
                    pass
        self.paths = require_runtime_paths(home, require_ane=False)
        self.tokenizer = FastBgeTokenizer(self.paths.tokenizer_json)
        self._gpu: FastBgeSmallGPU | None = None
        self._gpu_init_lock = threading.Lock()
        self._mlx_cache_limit_mb = mlx_cache_limit_mb
        self._set_wired_limit = set_wired_limit
        self._prev_wired_limit: int | None = None
        self._memory_hygiene_applied = False

        self.ane: ctypes.CDLL | None = None
        self.buckets: dict[int, int] = {}
        self.query_buckets: dict[int, int] = {}
        self._owned_bucket_ids: set[int] = set()
        self._bucket_paths: dict[int, Path] = {}
        self._bucket_fn_names: dict[int, str | None] = {}
        self.ane_plan_stats: dict[int, dict[str, int]] = {}
        self.ane_available = False
        self._wt0_f16: np.ndarray | None = None
        self._pos_f16: np.ndarray | None = None
        self._q_buf = np.empty(384, dtype=np.float32)
        self._query_lock = threading.Lock()
        self._corpus_lock = threading.Lock()
        self.last_stats: dict[str, int] | None = None
        self.cpu_sme_available = self.paths.cpu_dylib.exists()

        norm_default = "hetero" if default_mode == "hetero2" else default_mode
        if norm_default not in ("hetero", "gpu", "ane"):
            raise ValueError(f"Unknown default_mode {default_mode!r}; expected 'hetero', 'hetero2', 'gpu', or 'ane'")

        eff_enable_ane = (norm_default != "gpu") if enable_ane is None else bool(enable_ane)
        self._ane_enabled = eff_enable_ane

        if eff_enable_ane and self.paths.ane_dylib.exists() and self.paths.ane_models_dir.exists():
            self._init_ane(verify_plan=verify_ane_plan)

        if norm_default == "hetero" and not self.ane_available:
            self.default_mode = "gpu"
        elif norm_default == "ane" and not self.ane_available:
            if not eff_enable_ane:
                raise RuntimeError("ANE requested (default_mode='ane') but enable_ane=False")
            raise RuntimeError(
                f"ANE requested but native dylib or .mlmodelc bundles are missing under {self.paths.home}"
            )
        else:
            self.default_mode = norm_default

        # Skip eager GPU weight/kernel initialization when running pure-ANE mode.
        if self.default_mode != "ane":
            _ = self.gpu

        if warmup:
            self.warmup()

    def _ensure_memory_hygiene(self) -> None:
        if self._memory_hygiene_applied:
            return
        self._memory_hygiene_applied = True
        if self._mlx_cache_limit_mb is not None:
            set_mlx_cache_limit(self._mlx_cache_limit_mb)
        if self._set_wired_limit:
            self._prev_wired_limit = _apply_mlx_wired_limit()

    @property
    def gpu(self) -> FastBgeSmallGPU:
        if self._gpu is None:
            with self._gpu_init_lock:
                if self._gpu is None:
                    self._ensure_memory_hygiene()
                    if self._wt0_f16 is None or self._pos_f16 is None:
                        self._wt0_f16, self._pos_f16 = load_or_mmap_host_tables(self.paths.weights_npz)
                    self._gpu = FastBgeSmallGPU(
                        self.paths.weights_npz,
                        wt0_f16=self._wt0_f16,
                        pos_f16=self._pos_f16,
                    )
        return self._gpu

    @gpu.setter
    def gpu(self, value: Any) -> None:
        self._gpu = value

    def close(self) -> None:
        ane_lib = getattr(self, "ane", None)
        if ane_lib is not None:
            owned_ids = sorted(getattr(self, "_owned_bucket_ids", ()))
            if hasattr(ane_lib, "bge_ane_free_bucket"):
                for bid in owned_ids:
                    ane_lib.bge_ane_free_bucket(bid)
            if hasattr(ane_lib, "bge_ane_clear_borrowed_tables"):
                wt0 = getattr(self, "_wt0_f16", None)
                if wt0 is not None:
                    ane_lib.bge_ane_clear_borrowed_tables(wt0.ctypes.data)
            wt0_ptr = int(getattr(self, "_wt0_f16", None).ctypes.data) if getattr(self, "_wt0_f16", None) is not None else 0
            with _ANE_TABLE_BIND_LOCK:
                global _LAST_BOUND_ANE_TABLE_KEY
                if _LAST_BOUND_ANE_TABLE_KEY is not None and (
                    _LAST_BOUND_ANE_TABLE_KEY[0] == id(ane_lib)
                    or (wt0_ptr != 0 and _LAST_BOUND_ANE_TABLE_KEY[1] == wt0_ptr)
                ):
                    _LAST_BOUND_ANE_TABLE_KEY = None
            self._owned_bucket_ids.clear()
            self.buckets.clear()
            self.query_buckets.clear()
            self.ane = None
        self.ane_available = False

        for attr in ("_wt0_f16", "_pos_f16"):
            arr = getattr(self, attr, None)
            setattr(self, attr, None)
            if isinstance(arr, np.memmap):
                backing = getattr(arr, "_mmap", None)
                if backing is not None and not getattr(backing, "closed", False):
                    backing.close()

        prev = getattr(self, "_prev_wired_limit", None)
        self._prev_wired_limit = None
        _restore_mlx_wired_limit(prev)
        if getattr(self, "_gpu", None) is not None:
            _maybe_clear_mlx_arena(256)

    def __enter__(self) -> "FastBgeSmall":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _bind_ane_tables(self, ane: Any | None = None) -> None:
        global _LAST_BOUND_ANE_TABLE_KEY
        lib = ane if ane is not None else self.ane
        if lib is None or self._wt0_f16 is None or self._pos_f16 is None:
            return
        key = (
            id(lib),
            int(self._wt0_f16.ctypes.data),
            int(self._wt0_f16.shape[0]),
            int(self._pos_f16.ctypes.data),
            int(self._pos_f16.shape[0]),
        )
        with _ANE_TABLE_BIND_LOCK:
            if _LAST_BOUND_ANE_TABLE_KEY == key:
                return
            init_fn = getattr(lib, "bge_ane_init_tables_borrowed", None) or getattr(lib, "bge_ane_init_tables")
            rc = init_fn(
                self._wt0_f16.ctypes.data,
                self._wt0_f16.shape[0],
                self._pos_f16.ctypes.data,
                self._pos_f16.shape[0],
            )
            if rc != 0:
                raise RuntimeError(f"bge_ane_init_tables failed with code {rc}")
            _LAST_BOUND_ANE_TABLE_KEY = key

    def _init_ane(self, *, verify_plan: bool = True) -> None:
        self._wt0_f16, self._pos_f16 = load_or_mmap_host_tables(self.paths.weights_npz)

        ane = _load_ane_lib(self.paths.ane_dylib)
        self._bind_ane_tables(ane)

        mf_path: Path | None = None
        if hasattr(ane, "bge_ane_load_bucket_fn"):
            for mf_name in _MULTIFUNCTION_CANDIDATES:
                cand = self.paths.ane_models_dir / mf_name
                if cand.exists():
                    mf_path = cand
                    break

        def _try_load_bucket(batch_size: int, cap: int, legacy_filename: str, slots: int) -> tuple[int, Path | None, str | None]:
            fn_name = f"bge_b{batch_size}_s{cap}"
            if mf_path is not None:
                bid_mf = ane.bge_ane_load_bucket_fn(
                    str(mf_path).encode("utf-8"),
                    fn_name.encode("utf-8"),
                    batch_size,
                    cap,
                    slots,
                    1,
                )
                if bid_mf >= 0:
                    return bid_mf, mf_path, fn_name
            w8_path = self.paths.ane_models_dir / f"bge_b{batch_size}_s{cap}_w8a16.mlmodelc"
            for model_path in (w8_path, self.paths.ane_models_dir / legacy_filename):
                if not model_path.exists():
                    continue
                bid_sep = ane.bge_ane_load_bucket(str(model_path).encode("utf-8"), batch_size, cap, slots, 1)
                if bid_sep >= 0:
                    return bid_sep, model_path, None
            return -1, None, None

        loaded: dict[int, int] = {}
        query_loaded: dict[int, int] = {}
        bucket_paths: dict[int, Path] = {}
        bucket_fns: dict[int, str | None] = {}
        for cap, filename, batch_size, slots in _ANE_BUCKETS:
            bid, model_path, fn_name = _try_load_bucket(batch_size, cap, filename, slots)
            if bid >= 0 and model_path is not None:
                loaded[cap] = bid
                bucket_paths[cap] = model_path
                bucket_fns[cap] = fn_name
                if batch_size == 1:
                    query_loaded[cap] = bid

        for cap, filename, batch_size, slots in _OPTIONAL_QUERY_BUCKETS:
            bid, _, _ = _try_load_bucket(batch_size, cap, filename, slots)
            if bid >= 0:
                query_loaded[cap] = bid

        if loaded:
            self.ane = ane
            self.buckets = loaded
            self.query_buckets = query_loaded
            self._owned_bucket_ids = set(loaded.values()) | set(query_loaded.values())
            self._bucket_paths = bucket_paths
            self._bucket_fn_names = bucket_fns
            self.ane_available = True
            if verify_plan:
                self.inspect_ane_plans(warn_on_cpu_fallback=True)
        else:
            if hasattr(ane, "bge_ane_clear_borrowed_tables") and self._wt0_f16 is not None:
                ane.bge_ane_clear_borrowed_tables(self._wt0_f16.ctypes.data)

    def inspect_ane_plans(self, *, warn_on_cpu_fallback: bool = True) -> dict[int, dict[str, int]]:
        """Inspect MLComputePlan placement for loaded ANE buckets; warn if cpu_ops > 0 or ane_ops == 0."""
        plans: dict[int, dict[str, int]] = {}
        if not self.ane_available or self.ane is None or not hasattr(self.ane, "bge_ane_get_plan_counts"):
            return plans
        has_fn_plan = hasattr(self.ane, "bge_ane_get_plan_counts_fn")
        for cap in sorted(self._bucket_paths):
            model_path = self._bucket_paths[cap]
            fn_name = self._bucket_fn_names.get(cap)
            ane_c = ctypes.c_int(0)
            gpu_c = ctypes.c_int(0)
            cpu_c = ctypes.c_int(0)
            if fn_name and has_fn_plan:
                rc = self.ane.bge_ane_get_plan_counts_fn(
                    str(model_path).encode("utf-8"),
                    fn_name.encode("utf-8"),
                    1,
                    ctypes.byref(ane_c),
                    ctypes.byref(gpu_c),
                    ctypes.byref(cpu_c),
                )
            else:
                rc = self.ane.bge_ane_get_plan_counts(
                    str(model_path).encode("utf-8"),
                    1,
                    ctypes.byref(ane_c),
                    ctypes.byref(gpu_c),
                    ctypes.byref(cpu_c),
                )
            if rc != 0:
                if warn_on_cpu_fallback:
                    warnings.warn(
                        f"MLComputePlan inspection failed for {model_path.name} (rc={rc}: {_ane_err_detail(self.ane)})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            entry = {"ane_ops": int(ane_c.value), "gpu_ops": int(gpu_c.value), "cpu_ops": int(cpu_c.value)}
            plans[cap] = entry
            if warn_on_cpu_fallback and (entry["ane_ops"] == 0 or entry["cpu_ops"] > 0):
                warnings.warn(
                    f"CoreML ANE plan for {model_path.name} has ANE={entry['ane_ops']} GPU={entry['gpu_ops']} "
                    f"CPU={entry['cpu_ops']} (expected ane_ops > 0 and cpu_ops == 0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.ane_plan_stats = plans
        return plans

    def warmup(self) -> None:
        if self.ane_available:
            self.encode_query(list(range(10, 34)), prefer_ane=True)
        if self.default_mode == "ane" and self.ane_available:
            return
        w_seqs = [list(range(10, 266)) for _ in range(32)]
        b0 = pack_sequences(w_seqs, max_tokens=8192)[0]
        mx.eval(
            self.gpu.forward_packed(
                b0["tok_mx"], b0["pos_mx"], b0["tiles_mx"], b0["bounds_mx"], b0["cls_mx"], b0["M_tok"], b0["M_pad"]
            )
        )
        _maybe_clear_mlx_arena(256)

    def encode_query(self, token_ids: Sequence[int] | np.ndarray, prefer_ane: bool = True) -> np.ndarray:
        """Low-latency single-query encoder returning a `(384,)` L2-normalized float32 vector."""
        arr = np.ascontiguousarray(token_ids[:512], dtype=np.int32)
        if arr.shape[0] == 0:
            arr = np.array([101, 102], dtype=np.int32)
        sl = int(arr.shape[0])
        with self._query_lock:
            if prefer_ane and self.ane_available and self.ane is not None:
                self._bind_ane_tables()
                q_map = getattr(self, "query_buckets", None) or self.buckets
                for cap in _QUERY_CAPS:
                    if sl <= cap and cap in q_map:
                        rc = self.ane.bge_ane_query(q_map[cap], arr.ctypes.data, sl, self._q_buf.ctypes.data, 1)
                        if rc < 0.0:
                            detail = _ane_err_detail(self.ane)
                            raise RuntimeError(f"bge_ane_query(cap={cap}) failed with {rc}: {detail}")
                        return self._q_buf.copy()
            return self.gpu.encode_single(arr)

    def _ensure_ane_ready(self) -> None:
        if self.ane_available and self.ane is not None:
            return
        if not getattr(self, "_ane_enabled", True):
            raise RuntimeError(
                "ANE requested but ANE was not initialized (pass enable_ane=True or use default_mode!='gpu')"
            )
        raise RuntimeError(
            f"ANE requested but native dylib or .mlmodelc bundles are missing under {self.paths.home}"
        )

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        mode: str | None = None,
    ) -> np.ndarray:
        """Embed a single string `(384,)` or sequence of strings `[N, 384]` into L2-normalized float32 vectors."""
        eff_mode = mode or self.default_mode
        if eff_mode == "hetero2":
            eff_mode = "hetero"
        if eff_mode not in ("hetero", "gpu", "ane"):
            raise ValueError(f"Unknown mode {eff_mode!r}; expected 'hetero', 'hetero2', 'gpu', or 'ane'")
        if eff_mode == "ane":
            self._ensure_ane_ready()
        prefer_ane = eff_mode != "gpu"

        if isinstance(texts, str):
            ids = self.tokenizer.encode(texts, add_query_prefix=add_query_prefix, max_length=512)
            return self.encode_query(ids, prefer_ane=prefer_ane)

        items = list(texts)
        if not items:
            return np.empty((0, 384), dtype=np.float32)

        if is_query or len(items) == 1:
            tok_lists = self.tokenizer.encode_batch(items, add_query_prefix=add_query_prefix, max_length=512)
            if len(tok_lists) == 1:
                return self.encode_query(tok_lists[0], prefer_ane=prefer_ane)[None, :]
            if len(tok_lists) >= _BGE_QUERY_BATCH_MIN_SIZE or eff_mode == "gpu" or not self.ane_available:
                return self.encode_corpus(tok_lists, mode=eff_mode)
            return np.stack([self.encode_query(ids, prefer_ane=prefer_ane) for ids in tok_lists], axis=0)

        tok_lists = self.tokenizer.encode_batch(items, add_query_prefix=add_query_prefix, max_length=512)
        return self.encode_corpus(tok_lists, mode=eff_mode)

    def _run_ane_subdocs(self, doc_lists: Sequence[Sequence[int]], doc_indices: list[int], out_vecs: np.ndarray) -> None:
        self._ensure_ane_ready()
        assert self.ane is not None
        self._bind_ane_tables()
        avail_caps = [c for c in _CORPUS_CAPS if c in self.buckets]
        if not avail_caps:
            raise RuntimeError("No ANE corpus buckets loaded")

        by_bkt: dict[int, list[int]] = {c: [] for c in avail_caps}
        max_cap = avail_caps[-1]
        for i in doc_indices:
            sl = len(doc_lists[i])
            for c in avail_caps:
                if sl <= c or c == max_cap:
                    by_bkt[c].append(i)
                    break

        for cap in avail_caps:
            idxs = by_bkt[cap]
            if not idxs:
                continue
            lens = [min(len(doc_lists[i]), 512) for i in idxs]
            off = np.zeros(len(idxs) + 1, dtype=np.int32)
            off[1:] = np.cumsum(lens, dtype=np.int32)
            flat = np.empty(int(off[-1]), dtype=np.int32)
            for k, i in enumerate(idxs):
                sl = lens[k]
                if sl > 0:
                    flat[off[k] : off[k + 1]] = doc_lists[i][:sl]
            idx_arr = np.asarray(idxs, dtype=np.int32)
            rc = self.ane.bge_ane_encode_bucket(
                self.buckets[cap],
                len(idxs),
                flat.ctypes.data,
                off.ctypes.data,
                idx_arr.ctypes.data,
                out_vecs.ctypes.data,
                4,
            )
            if rc < 0:
                detail = _ane_err_detail(self.ane)
                raise RuntimeError(f"bge_ane_encode_bucket(cap={cap}) failed with {rc}: {detail}")

    def encode_corpus(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        mode: str | None = None,
        gpu_max_tokens: int = 8192,
        ane_chunk_docs: int = 96,
    ) -> np.ndarray:
        """Encode variable-length pre-tokenized corpus into `[N, 384]` L2-normalized float32 vectors."""
        n = len(token_lists)
        if n == 0:
            self.last_stats = {"gpu_docs": 0, "ane_docs": 0, "gpu_toks": 0, "ane_toks": 0}
            return np.empty((0, 384), dtype=np.float32)

        norm_toks: list[Sequence[int]] = [
            (t[:512] if len(t) > 0 else [101, 102]) for t in token_lists
        ]
        eff_mode = mode or self.default_mode
        if eff_mode == "hetero2":
            eff_mode = "hetero"
        if eff_mode == "hetero" and not self.ane_available:
            eff_mode = "gpu"

        with self._corpus_lock:
            if eff_mode == "gpu":
                tot_toks = sum(len(t) for t in norm_toks)
                self.last_stats = {"gpu_docs": n, "ane_docs": 0, "gpu_toks": tot_toks, "ane_toks": 0}
                res = self.gpu.encode_corpus(norm_toks, max_tokens=gpu_max_tokens)
                _maybe_clear_mlx_arena(256)
                return res

            out = np.empty((n, 384), dtype=np.float32)
            if eff_mode == "ane":
                self._run_ane_subdocs(norm_toks, list(range(n)), out)
                tot_toks = sum(len(t) for t in norm_toks)
                self.last_stats = {"gpu_docs": 0, "ane_docs": n, "gpu_toks": 0, "ane_toks": tot_toks}
                return out

            if eff_mode != "hetero":
                raise ValueError(f"Unknown mode {eff_mode!r}; expected 'hetero', 'hetero2', 'gpu', or 'ane'")

            order = sorted(range(n), key=lambda i: len(norm_toks[i]), reverse=True)
            lock = threading.Lock()
            state = {
                "gpu_head": 0,
                "ane_tail": n,
                "gpu_docs": 0,
                "ane_docs": 0,
                "gpu_toks": 0,
                "ane_toks": 0,
            }
            errors: list[Exception] = []

            def take_gpu_batch():
                with lock:
                    gh = state["gpu_head"]
                    at = state["ane_tail"]
                    if gh >= at:
                        return None
                    end = gh
                    tok_acc = 0
                    while end < at:
                        sl = len(norm_toks[order[end]])
                        if end > gh and tok_acc + sl > gpu_max_tokens:
                            break
                        tok_acc += sl
                        end += 1
                    state["gpu_head"] = end
                    state["gpu_docs"] += end - gh
                    state["gpu_toks"] += tok_acc
                    return order[gh:end]

            def take_ane_chunk():
                with lock:
                    gh = state["gpu_head"]
                    at = state["ane_tail"]
                    if gh >= at:
                        return None
                    shortest_sl = len(norm_toks[order[at - 1]])
                    if shortest_sl <= 192:
                        c_sz = max(ane_chunk_docs, 128)
                    elif shortest_sl <= 256:
                        c_sz = ane_chunk_docs
                    elif shortest_sl <= 320:
                        c_sz = 32
                    else:
                        c_sz = 16
                    start = max(gh, at - c_sz)
                    state["ane_tail"] = start
                    state["ane_docs"] += at - start
                    idxs = order[start:at]
                    state["ane_toks"] += sum(len(norm_toks[i]) for i in idxs)
                    return idxs

            def gpu_worker() -> None:
                try:
                    pending = None
                    while True:
                        batch_orig = take_gpu_batch()
                        if batch_orig is None:
                            break
                        sub_toks = [norm_toks[i] for i in batch_orig]
                        batch_orig_arr = np.asarray(batch_orig, dtype=np.int32)
                        for b in pack_sequences(sub_toks, max_tokens=gpu_max_tokens + 512):
                            vecs = self.gpu.forward_packed(
                                b["tok_mx"], b["pos_mx"], b["tiles_mx"], b["bounds_mx"], b["cls_mx"], b["M_tok"], b["M_pad"]
                            )
                            mx.async_eval(vecs)
                            if pending is not None:
                                pv, p_orig = pending
                                mx.eval(pv)  # guard: allow-eval (materialize previous async-dispatched batch)
                                out[p_orig] = np.array(pv, dtype=np.float32)
                            pending = (vecs, batch_orig_arr[b["orig_idx"]])
                    if pending is not None:
                        pv, p_orig = pending
                        mx.eval(pv)
                        out[p_orig] = np.array(pv, dtype=np.float32)
                    _maybe_clear_mlx_arena(256)
                except Exception as exc:
                    errors.append(exc)

            def ane_worker() -> None:
                try:
                    while True:
                        idxs = take_ane_chunk()
                        if idxs is None:
                            break
                        self._run_ane_subdocs(norm_toks, idxs, out)
                except Exception as exc:
                    errors.append(exc)

            t_ane = threading.Thread(target=ane_worker, daemon=False)
            t_ane.start()
            gpu_worker()
            t_ane.join()
            if errors:
                raise errors[0]

            self.last_stats = {
                "gpu_docs": state["gpu_docs"],
                "ane_docs": state["ane_docs"],
                "gpu_toks": state["gpu_toks"],
                "ane_toks": state["ane_toks"],
            }
            return out

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        mode: str | None = None,
    ) -> np.ndarray:
        """Encode document strings `[N]` into `[N, 384]` L2-normalized float32 embeddings."""
        items = list(texts)
        if not items:
            return np.empty((0, 384), dtype=np.float32)
        return self.encode_texts(items, is_query=False, add_query_prefix=False, mode=mode)

    def build_index(
        self,
        corpus: Sequence[str] | Sequence[Sequence[int]] | np.ndarray,
        *,
        sub_chunk_chars: int | None = None,
        doc_ids: Sequence[str] | None = None,
        doc_titles: Sequence[str | None] | None = None,
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        mode: IndexMode = "auto",
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
        cpu_backend: CpuBackendMode = "auto",
        max_threads: int = 0,
        encode_mode: str | None = None,
    ) -> CompactRetrievalIndex:
        """Build a `CompactRetrievalIndex` (`384d int8` `4.0x` lossless default, or `binary_int8` / `MaxSim`)."""
        if sub_chunk_chars is not None and not isinstance(corpus, np.ndarray) and corpus:
            if not isinstance(corpus[0], str):
                raise TypeError("sub_chunk_chars requires a sequence of raw document strings")
            from fast_shared.hybrid_retrieval import build_contextual_subchunk_index

            return build_contextual_subchunk_index(
                self,
                corpus,  # type: ignore[arg-type]
                doc_ids=doc_ids,
                doc_titles=doc_titles,
                sub_chunk_chars=int(sub_chunk_chars),
                index_mode=mode,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                agg=agg,
                hybrid_alpha=hybrid_alpha,
                cpu_backend=cpu_backend,
                max_threads=max_threads,
                encode_mode=encode_mode,
            )
        if isinstance(corpus, np.ndarray):
            vecs = corpus
        elif not corpus:
            vecs = np.empty((0, 384), dtype=np.float32)
        elif isinstance(corpus[0], str):
            vecs = self.encode_documents(corpus, mode=encode_mode)  # type: ignore[arg-type]
        else:
            vecs = self.encode_corpus(corpus, mode=encode_mode)  # type: ignore[arg-type]
        return CompactRetrievalIndex(
            vecs,
            doc_offsets=list(doc_offsets) if doc_offsets is not None and not isinstance(doc_offsets, np.ndarray) else doc_offsets,
            mode=mode,
            is_mrl_trained=self.is_mrl_trained,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            cpu_backend=cpu_backend,
            max_threads=max_threads,
        )
