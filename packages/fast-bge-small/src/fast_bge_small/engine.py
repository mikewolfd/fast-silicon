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
from fast_shared.embeddings import set_mlx_cache_limit

from .gpu import FastBgeSmallGPU, pack_sequences
from .paths import require_runtime_paths
from .tokenizer import FastBgeTokenizer

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

# Optional B=1 S=256 single-query bucket if generated via `scripts/build_ane_models.py --include-s256-query`.
# When absent on disk, 193..256-token queries safely pad to S=320 without touching the B=8 corpus bucket.
_OPTIONAL_QUERY_BUCKETS: tuple[tuple[int, str, int, int], ...] = (
    (256, "bge_b1_s256_fp16.mlmodelc", 1, 1),
)

_QUERY_CAPS: tuple[int, ...] = (24, 64, 128, 192, 256, 320, 384, 448, 512)
_CORPUS_CAPS: tuple[int, ...] = (64, 128, 192, 256, 320, 384, 448, 512)


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
    lib.bge_ane_load_bucket.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.bge_ane_load_bucket.restype = ctypes.c_int
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
        enable_ane: bool = True,
        verify_ane_plan: bool = True,
        mlx_cache_limit_mb: int | None = 512,
        set_wired_limit: bool = True,
    ) -> None:
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
        self._bucket_paths: dict[int, Path] = {}
        self.ane_plan_stats: dict[int, dict[str, int]] = {}
        self.ane_available = False
        self._wt0_f16: np.ndarray | None = None
        self._pos_f16: np.ndarray | None = None
        self._q_buf = np.empty(384, dtype=np.float32)
        self._query_lock = threading.Lock()
        self._corpus_lock = threading.Lock()
        self.last_stats: dict[str, int] | None = None

        norm_default = "hetero" if default_mode == "hetero2" else default_mode

        if enable_ane and self.paths.ane_dylib.exists() and self.paths.ane_models_dir.exists():
            self._init_ane(verify_plan=verify_ane_plan)

        if norm_default == "hetero" and not self.ane_available:
            self.default_mode = "gpu"
        elif norm_default == "ane" and not self.ane_available:
            raise RuntimeError(
                f"ANE requested but native dylib or .mlmodelc bundles are missing under {self.paths.home}"
            )
        elif norm_default not in ("hetero", "gpu", "ane"):
            raise ValueError(f"Unknown default_mode {default_mode!r}; expected 'hetero', 'hetero2', 'gpu', or 'ane'")
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
                    self._gpu = FastBgeSmallGPU(self.paths.weights_npz)
        return self._gpu

    @gpu.setter
    def gpu(self, value: Any) -> None:
        self._gpu = value

    def close(self) -> None:
        prev = getattr(self, "_prev_wired_limit", None)
        self._prev_wired_limit = None
        _restore_mlx_wired_limit(prev)
        if getattr(self, "_gpu", None) is not None:
            _maybe_clear_mlx_arena(256)

    def __enter__(self) -> "FastBgeSmall":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _init_ane(self, *, verify_plan: bool = True) -> None:
        f16 = dict(np.load(self.paths.weights_npz))
        self._wt0_f16 = np.ascontiguousarray(
            f16["embeddings.word_embeddings.weight"].astype(np.float16)
            + f16["embeddings.token_type_embeddings.weight"][0:1].astype(np.float16)
        )
        self._pos_f16 = np.ascontiguousarray(f16["embeddings.position_embeddings.weight"].astype(np.float16))

        ane = _load_ane_lib(self.paths.ane_dylib)
        rc = ane.bge_ane_init_tables(
            self._wt0_f16.ctypes.data,
            self._wt0_f16.shape[0],
            self._pos_f16.ctypes.data,
            self._pos_f16.shape[0],
        )
        if rc != 0:
            raise RuntimeError(f"bge_ane_init_tables failed with code {rc}")

        loaded: dict[int, int] = {}
        query_loaded: dict[int, int] = {}
        bucket_paths: dict[int, Path] = {}
        for cap, filename, batch_size, slots in _ANE_BUCKETS:
            model_path = self.paths.ane_models_dir / filename
            if not model_path.exists():
                continue
            bid = ane.bge_ane_load_bucket(str(model_path).encode("utf-8"), batch_size, cap, slots, 1)
            if bid >= 0:
                loaded[cap] = bid
                bucket_paths[cap] = model_path
                if batch_size == 1:
                    query_loaded[cap] = bid

        for cap, filename, batch_size, slots in _OPTIONAL_QUERY_BUCKETS:
            model_path = self.paths.ane_models_dir / filename
            if not model_path.exists():
                continue
            bid = ane.bge_ane_load_bucket(str(model_path).encode("utf-8"), batch_size, cap, slots, 1)
            if bid >= 0:
                query_loaded[cap] = bid

        if loaded:
            self.ane = ane
            self.buckets = loaded
            self.query_buckets = query_loaded
            self._bucket_paths = bucket_paths
            self.ane_available = True
            if verify_plan:
                self.inspect_ane_plans(warn_on_cpu_fallback=True)

    def inspect_ane_plans(self, *, warn_on_cpu_fallback: bool = True) -> dict[int, dict[str, int]]:
        """Inspect MLComputePlan placement for loaded ANE buckets; warn if cpu_ops > 0 or ane_ops == 0."""
        plans: dict[int, dict[str, int]] = {}
        if not self.ane_available or self.ane is None or not hasattr(self.ane, "bge_ane_get_plan_counts"):
            return plans
        for cap in sorted(self._bucket_paths):
            model_path = self._bucket_paths[cap]
            ane_c = ctypes.c_int(0)
            gpu_c = ctypes.c_int(0)
            cpu_c = ctypes.c_int(0)
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
                q_map = getattr(self, "query_buckets", None) or self.buckets
                for cap in _QUERY_CAPS:
                    if sl <= cap and cap in q_map:
                        rc = self.ane.bge_ane_query(q_map[cap], arr.ctypes.data, sl, self._q_buf.ctypes.data, 1)
                        if rc < 0.0:
                            detail = _ane_err_detail(self.ane)
                            raise RuntimeError(f"bge_ane_query(cap={cap}) failed with {rc}: {detail}")
                        return self._q_buf.copy()
            return self.gpu.encode_single(arr)

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
            return np.stack([self.encode_query(ids, prefer_ane=prefer_ane) for ids in tok_lists], axis=0)

        tok_lists = self.tokenizer.encode_batch(items, add_query_prefix=add_query_prefix, max_length=512)
        return self.encode_corpus(tok_lists, mode=eff_mode)

    def _run_ane_subdocs(self, doc_lists: Sequence[Sequence[int]], doc_indices: list[int], out_vecs: np.ndarray) -> None:
        if not self.ane_available or self.ane is None:
            raise RuntimeError("ANE runner not initialized")
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
                        b = pack_sequences(sub_toks, max_tokens=gpu_max_tokens + 512)[0]
                        vecs = self.gpu.forward_packed(
                            b["tok_mx"], b["pos_mx"], b["tiles_mx"], b["bounds_mx"], b["cls_mx"], b["M_tok"], b["M_pad"]
                        )
                        mx.async_eval(vecs)
                        if pending is not None:
                            pv, p_orig = pending
                            mx.eval(pv)  # guard: allow-eval (materialize previous async-dispatched batch)
                            out[p_orig] = np.array(pv, dtype=np.float32)
                        pending = (vecs, np.asarray(batch_orig, dtype=np.int32))
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
