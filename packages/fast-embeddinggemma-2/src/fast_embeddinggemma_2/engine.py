"""Unified Apple Silicon EmbeddingGemma-2 engine (Metal GPU + 16-core ANE + opt-in CPU SME)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Any, Sequence
import warnings

import mlx.core as mx
import numpy as np
from fast_shared.bench import clear_mlx_arena
from fast_shared.embeddings import resolve_embedding_dim, set_mlx_cache_limit, truncate_and_normalize

from .gpu import (
    ANE_MS_COST,
    EMBED_DIM,
    GPU_B_OPTS,
    HIDDEN_SIZE,
    MAX_SEQ_LEN,
    OVER_B_OPTS,
    FastEmbeddingGemma2GPU,
    build_over_chunks,
    fast_pack_cost_aware,
    make_rope_tables_np,
)
from .paths import (
    MmapTokenTable,
    load_or_mmap_token_table,
    require_runtime_paths,
    resolve_openinfinity_dir,
    resolve_paths,
)
from .tokenizer import FastEmbeddingGemma2Tokenizer

_MULTIFUNCTION_CANDIDATES: tuple[str, ...] = (
    "emb2_ane_w8.mlmodelc",
    "emb2_ane_multifunction.mlmodelc",
)

_OPENINFINITY_CAPS: tuple[int, ...] = (32, 48, 64, 128, 256, 512)

_ANE_BUNDLES: tuple[tuple[int, str, int], ...] = (
    (32, "emb2_pk_s32_w8.mlmodelc", 1),
    (64, "emb2_pk_s64_w8.mlmodelc", 4),
    (128, "emb2_pk_s128_w8.mlmodelc", 4),
    (192, "emb2_pk_s192_w8.mlmodelc", 4),
    (256, "emb2_pk_s256_w8.mlmodelc", 4),
    (320, "emb2_pk_s320_w8.mlmodelc", 4),
    (384, "emb2_pk_s384_w8.mlmodelc", 4),
    (448, "emb2_pk_s448_w8.mlmodelc", 4),
    (512, "emb2_pk_s512_w8.mlmodelc", 4),
)

# Cap CPU-SME concurrent slots / BLAS workers to <= 2 (2 physical shared SME2 units on M4/M5).
MAX_CPU_SME_THREADS: int = 2

_CPU_BUNDLES: tuple[tuple[int, str, int], ...] = (
    (256, "emb2_cpu_s256_mix_w8.mlmodelc", MAX_CPU_SME_THREADS),
    (384, "emb2_cpu_s384_mix_w8.mlmodelc", MAX_CPU_SME_THREADS),
    (512, "emb2_cpu_s512_mix_w8.mlmodelc", MAX_CPU_SME_THREADS),
)

_QUERY_CAPS: tuple[int, ...] = (32, 64, 128, 192, 256, 320, 384, 448, 512)

HETERO_CPU_MAX_SEC: float = 0.75
HETERO_CPU_MAX_DOCS: int = 192
HETERO_CPU_MAX_TOKS: int = 32768

# Minimum workload size before opt-in 3-way hetero3 fires CPU-SME alongside GPU+ANE.
HETERO3_CPU_MIN_DOCS: int = 256
HETERO3_CPU_MIN_TOKS: int = 65536


def _sysctl_str(key: str) -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except Exception:
        return ""


def _sysctl_int(key: str) -> int | None:
    raw = _sysctl_str(key)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def probe_cpu_topology() -> dict[str, Any]:
    """Probe perflevel0/perflevel1 names and core counts without assuming E-cores exist.

    Base M5 reports 4 Super + 6 Efficiency, whereas M5 Pro/Max report 6 Super + 12 Performance (zero E-cores).
    """
    p0_name = _sysctl_str("hw.perflevel0.name") or "Performance"
    p0_cnt = _sysctl_int("hw.perflevel0.physicalcpu")
    p1_name = _sysctl_str("hw.perflevel1.name") or ""
    p1_cnt = _sysctl_int("hw.perflevel1.physicalcpu")
    has_eff_cores = bool(p1_cnt and "eff" in p1_name.lower())
    return {
        "brand": _sysctl_str("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "perflevel0_name": p0_name,
        "perflevel0_cores": p0_cnt,
        "perflevel1_name": p1_name or None,
        "perflevel1_cores": p1_cnt,
        "has_efficiency_cores": has_eff_cores,
        "max_cpu_sme_threads": MAX_CPU_SME_THREADS,
    }


def _apply_mlx_wired_limit() -> int | None:
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


def _should_stop_hetero_cpu(elapsed_s: float, cpu_docs: int, cpu_toks: int, *, eff_mode: str) -> bool:
    """Keep CPU-SME on short bursts only in hetero/hetero3 mode so sustained runs do not throttle ANE/GPU."""
    if eff_mode not in ("hetero", "hetero3"):
        return False
    return (
        (elapsed_s >= HETERO_CPU_MAX_SEC)
        or (cpu_docs >= HETERO_CPU_MAX_DOCS)
        or (cpu_toks >= HETERO_CPU_MAX_TOKS)
    )


def _allow_hetero3_cpu_worker(
    n_docs: int,
    total_toks: int,
    *,
    cpu_sme_available: bool,
    llm_coexec: bool = False,
) -> bool:
    """Retired: 3-way CPU-SME (`hetero3`) slowed OCR backfill +13.6%; heterogeneous mode routes strictly 2-way GPU+ANE."""
    _ = (n_docs, total_toks, cpu_sme_available, llm_coexec)
    return False


def _load_coreml_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.emb2_pk_load_units.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.emb2_pk_load_units.restype = ctypes.c_void_p
    if hasattr(lib, "emb2_pk_load_units_fn"):
        lib.emb2_pk_load_units_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.emb2_pk_load_units_fn.restype = ctypes.c_void_p
    if hasattr(lib, "emb2_oi_load_embed"):
        lib.emb2_oi_load_embed.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.emb2_oi_load_embed.restype = ctypes.c_void_p
        lib.emb2_oi_exec_embed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.emb2_oi_exec_embed.restype = ctypes.c_double
        lib.emb2_oi_free.argtypes = [ctypes.c_void_p]
        lib.emb2_oi_free.restype = None
    lib.emb2_pk_submit_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pk_submit_queue.restype = ctypes.c_void_p
    lib.emb2_pk_wait_queue.argtypes = [ctypes.c_void_p]
    lib.emb2_pk_wait_queue.restype = ctypes.c_double
    lib.emb2_pk_exec_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pk_exec_queue.restype = ctypes.c_double
    lib.emb2_pk_free.argtypes = [ctypes.c_void_p]
    lib.emb2_pk_free.restype = None
    if hasattr(lib, "emb2_inspect_plan"):
        lib.emb2_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.emb2_inspect_plan.restype = None
    if hasattr(lib, "emb2_get_plan_counts"):
        lib.emb2_get_plan_counts.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.emb2_get_plan_counts.restype = ctypes.c_int
    if hasattr(lib, "emb2_get_plan_counts_fn"):
        lib.emb2_get_plan_counts_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.emb2_get_plan_counts_fn.restype = ctypes.c_int
    if hasattr(lib, "emb2_last_error"):
        lib.emb2_last_error.argtypes = []
        lib.emb2_last_error.restype = ctypes.c_char_p
    return lib


def _coreml_err_detail(lib: ctypes.CDLL | None) -> str:
    if lib is not None and hasattr(lib, "emb2_last_error"):
        raw = lib.emb2_last_error()
        if raw:
            return raw.decode("utf-8", errors="replace")
    return ""


def _load_io_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.emb2_prep_batch.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_prep_batch.restype = None
    if hasattr(lib, "emb2_prep_batch_i8"):
        lib.emb2_prep_batch_i8.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.emb2_prep_batch_i8.restype = None
    if hasattr(lib, "emb2_gather_scaled_i8"):
        lib.emb2_gather_scaled_i8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.emb2_gather_scaled_i8.restype = None
    lib.emb2_pool_project_batch.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pool_project_batch.restype = None
    lib.emb2_pool_project_batch_dim.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.emb2_pool_project_batch_dim.restype = None
    return lib


class CoreMLScratch:
    """Reusable aligned buffers for C++/NEON prep -> CoreML -> Accelerate projection."""

    def __init__(self, max_N: int, max_S: int) -> None:
        self.h_all = np.empty((max_N, 512, max_S), dtype=np.float16)
        self.b_all = np.empty((max_N, max_S, max_S), dtype=np.float16)
        self.cs_all = np.empty((max_N, max_S, 128), dtype=np.float16)
        self.ss_all = np.empty((max_N, max_S, 128), dtype=np.float16)
        self.cf_all = np.empty((max_N, max_S, 256), dtype=np.float16)
        self.sf_all = np.empty((max_N, max_S, 256), dtype=np.float16)
        self.out_all = np.empty((max_N, max_S, 512), dtype=np.float16)
        self.ids_flat = np.empty(max_N * max_S, dtype=np.int32)
        self.pos_flat = np.empty(max_N * max_S, dtype=np.int32)
        self.arange_s = np.arange(max_S, dtype=np.int32)
        self.span_offsets = np.empty(max_N + 1, dtype=np.int32)
        self.spans_se = np.empty((max_N * 16, 2), dtype=np.int32)
        self.doc_lane_se = np.empty((max_N * 16, 3), dtype=np.int32)
        self.scratch_pooled = np.empty((max_N * 16, 512), dtype=np.float32)
        self.out_embs = np.empty((max_N * 16, EMBED_DIM), dtype=np.float32)

    def ensure_docs(self, tot_docs: int) -> None:
        if tot_docs > len(self.spans_se):
            cap = tot_docs + 64
            self.spans_se = np.empty((cap, 2), dtype=np.int32)
            self.doc_lane_se = np.empty((cap, 3), dtype=np.int32)
            self.scratch_pooled = np.empty((cap, 512), dtype=np.float32)
            self.out_embs = np.empty((cap, EMBED_DIM), dtype=np.float32)


MAX_EMBED_DIM: int = EMBED_DIM
MRL_DIMENSIONS: tuple[int, ...] = (128, 256, 512, 768)


class FastEmbeddingGemma2:
    """Heterogeneous Apple Silicon EmbeddingGemma-2 engine (GPU + ANE W8 + opt-in CPU-SME mix-W8)."""

    dim: int = EMBED_DIM
    max_dim: int = EMBED_DIM
    supports_dimensions: bool = True
    mrl_dimensions: tuple[int, ...] = MRL_DIMENSIONS
    model_name: str = "google/embeddinggemma-2"
    max_seq_len: int = MAX_SEQ_LEN
    ane_query_max_len: int = 512

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        openinfinity_dir: str | Path | None = None,
        default_mode: str = "hetero",
        warmup: bool = True,
        enable_ane: bool = True,
        enable_cpu_sme: bool = False,
        dimensions: int | None = None,
        verify_ane_plan: bool = True,
        mlx_cache_limit_mb: int | None = 512,
        set_wired_limit: bool = True,
        llm_coexec: bool = False,
    ) -> None:
        self.dim = resolve_embedding_dim(dimensions, max_dim=EMBED_DIM)
        self._openinfinity_dir: Path | None = Path(openinfinity_dir).expanduser().resolve() if openinfinity_dir else None
        if self._openinfinity_dir is not None:
            self.paths = resolve_paths(home)
            oi_text_dir = self._openinfinity_dir / "text" if (self._openinfinity_dir / "text").is_dir() else self._openinfinity_dir
            oi_tok = oi_text_dir / "tokenizer.json"
            if not oi_tok.is_file():
                oi_tok = self.paths.tokenizer_json
            self.tokenizer = FastEmbeddingGemma2Tokenizer(oi_tok)
        else:
            self.paths = require_runtime_paths(home, require_ane=False)
            self.tokenizer = FastEmbeddingGemma2Tokenizer(self.paths.tokenizer_json)

        self._gpu: FastEmbeddingGemma2GPU | None = None
        self._gpu_init_lock = threading.Lock()
        self._mlx_cache_limit_mb = mlx_cache_limit_mb
        self._set_wired_limit = set_wired_limit
        self._prev_wired_limit: int | None = None
        self._memory_hygiene_applied = False
        self.llm_coexec = bool(llm_coexec)
        self.cpu_topology = probe_cpu_topology()

        self._coreml: ctypes.CDLL | None = None
        self._io: ctypes.CDLL | None = None
        self.ane_handles: dict[int, int] = {}
        self.cpu_handles: dict[int, int] = {}
        self._oi_handles: dict[int, int] = {}
        self._ane_bundle_paths: dict[int, Path] = {}
        self._ane_bundle_fns: dict[int, str | None] = {}
        self.ane_plan_stats: dict[int, dict[str, int]] = {}
        self._handle_locks: dict[tuple[str, int], threading.Lock] = {}
        self.ane_available = False
        self.cpu_sme_available = False

        self._token_table: MmapTokenTable | None = None
        self._emb_scaled_np: np.ndarray | None = None
        self._emb_proj_T32_np: np.ndarray | None = None
        self._c256_np: np.ndarray | None = None
        self._s256_np: np.ndarray | None = None
        self._c512_np: np.ndarray | None = None
        self._s512_np: np.ndarray | None = None
        self._neg_inf_u16 = int(np.float16(-40000.0).view(np.uint16))

        self._query_scratch: CoreMLScratch | None = None
        self._ane_scratch: CoreMLScratch | None = None
        self._ane_scratch_b: CoreMLScratch | None = None
        self._cpu_scratch: CoreMLScratch | None = None
        self._gpu_lock = threading.Lock()
        self._corpus_lock = threading.Lock()
        self.last_stats: dict[str, int] | None = None

        # Retire hetero3 CPU-SME 3rd leg; only enable CPU bundles for explicit standalone mode="cpu".
        eff_enable_cpu = bool(enable_cpu_sme) and (default_mode == "cpu")
        if eff_enable_cpu:
            cur_blas = os.environ.get("VECLIB_MAXIMUM_THREADS")
            if not cur_blas or (cur_blas.isdigit() and int(cur_blas) > MAX_CPU_SME_THREADS):
                os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_CPU_SME_THREADS)

        if self._openinfinity_dir is not None:
            self._init_openinfinity(self._openinfinity_dir, verify_plan=verify_ane_plan)
            if default_mode not in ("ane", "cpu") and self.paths.weights_npz.exists():
                _ = self.gpu
        else:
            # Skip eager Metal GPU weight/kernel initialization when default_mode is pure ANE or pure CPU.
            if default_mode not in ("ane", "cpu"):
                _ = self.gpu

            if (
                (enable_ane or eff_enable_cpu)
                and self.paths.coreml_dylib.exists()
                and self.paths.io_dylib.exists()
                and self.paths.ane_models_dir.exists()
            ):
                self._init_coreml(
                    enable_ane=enable_ane,
                    enable_cpu_sme=eff_enable_cpu,
                    verify_plan=verify_ane_plan,
                )

        self.default_mode = self._resolve_mode(default_mode)
        if self.default_mode not in ("ane", "cpu") and self._gpu is None and self.paths.weights_npz.exists():
            _ = self.gpu

        if warmup:
            self.warmup()

    @classmethod
    def from_openinfinity(
        cls,
        model_dir: str | Path | None = None,
        *,
        local_files_only: bool = True,
        default_mode: str = "ane",
        warmup: bool = False,
        **kwargs: Any,
    ) -> "FastEmbeddingGemma2":
        """Instantiate zero-compile ANE engine from prebuilt `OpenInfinity/embeddinggemma-2-coreml-int8`."""
        oi_dir = resolve_openinfinity_dir(model_dir, local_files_only=local_files_only)
        return cls(openinfinity_dir=oi_dir, default_mode=default_mode, warmup=warmup, **kwargs)

    def _ensure_memory_hygiene(self) -> None:
        if self._memory_hygiene_applied:
            return
        self._memory_hygiene_applied = True
        if self._mlx_cache_limit_mb is not None:
            set_mlx_cache_limit(self._mlx_cache_limit_mb)
        if self._set_wired_limit:
            self._prev_wired_limit = _apply_mlx_wired_limit()

    @property
    def gpu(self) -> FastEmbeddingGemma2GPU:
        if self._gpu is None:
            with self._gpu_init_lock:
                if self._gpu is None:
                    self._ensure_memory_hygiene()
                    if self._token_table is None:
                        self._token_table = load_or_mmap_token_table(self.paths.weights_npz, io_lib=self._io)
                    self._gpu = FastEmbeddingGemma2GPU(
                        self.paths.weights_npz,
                        token_table=self._token_table,
                    )
        return self._gpu

    @gpu.setter
    def gpu(self, value: Any) -> None:
        self._gpu = value

    @property
    def _has_gpu_weights(self) -> bool:
        return self._gpu is not None or self.paths.weights_npz.is_file()

    def _resolve_dim(self, dimensions: int | None) -> int:
        return resolve_embedding_dim(dimensions, max_dim=EMBED_DIM, default_dim=self.dim)

    def _resolve_mode(self, mode: str) -> str:
        # Canonical 2-way heterogeneous execution ("hetero", "hetero2", and retired alias "hetero3" -> 2-way GPU + ANE)
        if mode in ("hetero", "hetero2", "hetero3"):
            if getattr(self, "_oi_handles", None) and not self._has_gpu_weights:
                return "ane"
            return "hetero" if self.ane_available else "gpu"
        if mode == "ane" and not self.ane_available:
            raise RuntimeError(
                f"ANE requested but CoreML runner dylibs or .mlmodelc bundles are missing under {self.paths.home}"
            )
        if mode == "gpu" and getattr(self, "_oi_handles", None) and not self._has_gpu_weights:
            raise FileNotFoundError(
                f"mode='gpu' requested on OpenInfinity engine without exported GPU weights at {self.paths.weights_npz}"
            )
        if mode == "cpu" and not self.cpu_sme_available:
            raise RuntimeError(
                f"CPU-SME requested but CoreML runner dylibs, cpu .mlmodelc bundles, or enable_cpu_sme=True are missing under {self.paths.home}"
            )
        if mode not in ("gpu", "ane", "cpu", "hetero", "hetero2", "hetero3"):
            raise ValueError(f"Unsupported mode: {mode!r}")
        return mode

    def _load_coreml_host_tables(self) -> None:
        if self._gpu is not None:
            self._token_table = self._gpu.token_table
            self._token_table.bind_io_lib(self._io)
            self._emb_scaled_np = None
            self._emb_proj_T32_np = np.ascontiguousarray(np.array(self._gpu.emb_proj_wT32))
            self._c256_np = np.ascontiguousarray(np.array(self._gpu.c256))
            self._s256_np = np.ascontiguousarray(np.array(self._gpu.s256))
            self._c512_np = np.ascontiguousarray(np.array(self._gpu.c512))
            self._s512_np = np.ascontiguousarray(np.array(self._gpu.s512))
            return

        # Clean-page mmap token table + pure-NumPy projection/RoPE tables so mode="ane"/"cpu" never touches GPU.
        if self._token_table is None:
            self._token_table = load_or_mmap_token_table(self.paths.weights_npz, io_lib=self._io)
        else:
            self._token_table.bind_io_lib(self._io)
        self._emb_scaled_np = None
        with np.load(self.paths.weights_npz) as raw:
            proj_raw = raw["emb_proj"]
            proj_2d = proj_raw[:, :, 0, 0] if proj_raw.ndim == 4 else proj_raw
            self._emb_proj_T32_np = np.ascontiguousarray(proj_2d.astype(np.float32).T)
        self._c256_np, self._s256_np, self._c512_np, self._s512_np = make_rope_tables_np(MAX_SEQ_LEN)

    def _init_openinfinity(self, oi_root: Path, *, verify_plan: bool = True) -> None:
        if not (self.paths.coreml_dylib.exists() and self.paths.io_dylib.exists()):
            raise FileNotFoundError(
                f"Missing native runner libraries under {self.paths.build_dir}. Run `make native`."
            )
        coreml = _load_coreml_lib(self.paths.coreml_dylib)
        io_lib = _load_io_lib(self.paths.io_dylib)
        text_dir = oi_root / "text" if (oi_root / "text").is_dir() else oi_root
        mlmodelc = text_dir / "EmbeddingGemma2Text.mlmodelc"
        if not mlmodelc.exists():
            raise FileNotFoundError(f"Missing EmbeddingGemma2Text.mlmodelc in {text_dir}")

        self._token_table = load_or_mmap_token_table(oi_root, io_lib=io_lib)
        self._emb_scaled_np = None

        for cap in _OPENINFINITY_CAPS:
            fn_name = f"embed_{cap}"
            pd = 1 if cap == 32 else 4
            h = coreml.emb2_oi_load_embed(
                str(mlmodelc).encode("utf-8"),
                fn_name.encode("utf-8"),
                cap,
                pd,
                0,
            )
            if h:
                self._oi_handles[cap] = h
                self._ane_bundle_paths[cap] = mlmodelc
                self._ane_bundle_fns[cap] = fn_name
                self._handle_locks[("oi", cap)] = threading.Lock()

        if not self._oi_handles:
            raise RuntimeError(f"Failed to load OpenInfinity functions from {mlmodelc}: {_coreml_err_detail(coreml)}")

        self._coreml = coreml
        self._io = io_lib
        self.ane_available = True
        if verify_plan:
            self.inspect_ane_plans(warn_on_cpu_fallback=True)

    def _init_coreml(self, *, enable_ane: bool, enable_cpu_sme: bool, verify_plan: bool = True) -> None:
        coreml = _load_coreml_lib(self.paths.coreml_dylib)
        io_lib = _load_io_lib(self.paths.io_dylib)

        if enable_ane:
            mf_path: Path | None = None
            if hasattr(coreml, "emb2_pk_load_units_fn"):
                for mf_name in _MULTIFUNCTION_CANDIDATES:
                    cand = self.paths.ane_models_dir / mf_name
                    if cand.exists():
                        mf_path = cand
                        break

            for seq_len, fname, pd in _ANE_BUNDLES:
                h = None
                chosen_path: Path | None = None
                chosen_fn: str | None = None
                if mf_path is not None:
                    fn_name = f"emb2_pk_s{seq_len}"
                    h = coreml.emb2_pk_load_units_fn(
                        str(mf_path).encode("utf-8"),
                        fn_name.encode("utf-8"),
                        seq_len,
                        pd,
                        0,
                    )
                    if h:
                        chosen_path = mf_path
                        chosen_fn = fn_name
                if not h:
                    p = self.paths.ane_models_dir / fname
                    if p.exists():
                        h = coreml.emb2_pk_load_units(str(p).encode("utf-8"), seq_len, pd, 0)
                        if h:
                            chosen_path = p
                            chosen_fn = None
                if h and chosen_path is not None:
                    self.ane_handles[seq_len] = h
                    self._ane_bundle_paths[seq_len] = chosen_path
                    self._ane_bundle_fns[seq_len] = chosen_fn
                    self._handle_locks[("ane", seq_len)] = threading.Lock()

        if enable_cpu_sme:
            for seq_len, fname, pd in _CPU_BUNDLES:
                p = self.paths.ane_models_dir / fname
                if not p.exists():
                    continue
                h = coreml.emb2_pk_load_units(str(p).encode("utf-8"), seq_len, min(pd, MAX_CPU_SME_THREADS), 1)
                if h:
                    self.cpu_handles[seq_len] = h
                    self._handle_locks[("cpu", seq_len)] = threading.Lock()

        if self.ane_handles or self.cpu_handles:
            self._coreml = coreml
            self._io = io_lib
            self._load_coreml_host_tables()
            self._query_scratch = CoreMLScratch(1, 512)
            self._ane_scratch = CoreMLScratch(32, 512)
            self._ane_scratch_b = CoreMLScratch(32, 512)
            self._cpu_scratch = CoreMLScratch(24, 512) if self.cpu_handles else None
            self.ane_available = bool(self.ane_handles)
            self.cpu_sme_available = bool(self.cpu_handles)
            if verify_plan and self.ane_available:
                self.inspect_ane_plans(warn_on_cpu_fallback=True)

    def inspect_ane_plans(self, *, warn_on_cpu_fallback: bool = True) -> dict[int, dict[str, int]]:
        """Inspect MLComputePlan placement for loaded ANE bundles; warn if cpu_ops > 0 or ane_ops == 0."""
        plans: dict[int, dict[str, int]] = {}
        if not self.ane_available or self._coreml is None or not hasattr(self._coreml, "emb2_get_plan_counts"):
            return plans
        has_fn = hasattr(self._coreml, "emb2_get_plan_counts_fn")
        for seq_len in sorted(self._ane_bundle_paths):
            model_path = self._ane_bundle_paths[seq_len]
            fn_name = self._ane_bundle_fns.get(seq_len)
            ane_c = ctypes.c_int(0)
            gpu_c = ctypes.c_int(0)
            cpu_c = ctypes.c_int(0)
            if fn_name and has_fn:
                rc = self._coreml.emb2_get_plan_counts_fn(
                    str(model_path).encode("utf-8"),
                    fn_name.encode("utf-8"),
                    0,
                    ctypes.byref(ane_c),
                    ctypes.byref(gpu_c),
                    ctypes.byref(cpu_c),
                )
            else:
                rc = self._coreml.emb2_get_plan_counts(
                    str(model_path).encode("utf-8"),
                    0,
                    ctypes.byref(ane_c),
                    ctypes.byref(gpu_c),
                    ctypes.byref(cpu_c),
                )
            if rc != 0:
                if warn_on_cpu_fallback:
                    warnings.warn(
                        f"MLComputePlan inspection failed for {model_path.name} (rc={rc}: {_coreml_err_detail(self._coreml)})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            entry = {"ane_ops": int(ane_c.value), "gpu_ops": int(gpu_c.value), "cpu_ops": int(cpu_c.value)}
            plans[seq_len] = entry
            if warn_on_cpu_fallback and (entry["ane_ops"] == 0 or entry["cpu_ops"] > 0):
                warnings.warn(
                    f"CoreML ANE plan for {model_path.name} has ANE={entry['ane_ops']} GPU={entry['gpu_ops']} "
                    f"CPU={entry['cpu_ops']} (expected ane_ops > 0 and cpu_ops == 0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.ane_plan_stats = plans
        return plans

    def close(self) -> None:
        coreml = getattr(self, "_coreml", None)
        if coreml is not None:
            for h in getattr(self, "ane_handles", {}).values():
                coreml.emb2_pk_free(h)
            for h in getattr(self, "cpu_handles", {}).values():
                coreml.emb2_pk_free(h)
            if hasattr(coreml, "emb2_oi_free"):
                for h in getattr(self, "_oi_handles", {}).values():
                    coreml.emb2_oi_free(h)
            self.ane_handles.clear()
            self.cpu_handles.clear()
            if hasattr(self, "_oi_handles"):
                self._oi_handles.clear()
        tbl = getattr(self, "_token_table", None)
        if tbl is not None:
            tbl.close()
            self._token_table = None
        self.ane_available = False
        self.cpu_sme_available = False
        prev = getattr(self, "_prev_wired_limit", None)
        self._prev_wired_limit = None
        _restore_mlx_wired_limit(prev)
        if getattr(self, "_gpu", None) is not None:
            _maybe_clear_mlx_arena(256)

    def __enter__(self) -> "FastEmbeddingGemma2":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _prep_coreml_lanes(
        self,
        S: int,
        lanes: list,
        scratch: CoreMLScratch,
    ) -> tuple[int, int, int, list[int]]:
        assert self._io is not None
        assert self._c256_np is not None and self._s256_np is not None
        assert self._c512_np is not None and self._s512_np is not None

        N = len(lanes)
        ids = scratch.ids_flat[: N * S].reshape(N, S)
        pos = scratch.pos_flat[: N * S].reshape(N, S)
        ids.fill(-1)
        pos.fill(0)
        span_offsets = scratch.span_offsets
        span_offsets[0] = 0
        sp_i = 0
        doc_idxs: list[int] = []
        tot_docs = sum(len(ln) for ln in lanes)
        scratch.ensure_docs(tot_docs)

        spans_se = scratch.spans_se
        doc_lane_se = scratch.doc_lane_se
        ar = scratch.arange_s
        real_toks = 0
        for n, ln in enumerate(lanes):
            cur = 0
            for idx, tok in ln:
                ll = len(tok)
                real_toks += ll
                ids[n, cur : cur + ll] = tok
                pos[n, cur : cur + ll] = ar[:ll]
                spans_se[sp_i, 0] = cur
                spans_se[sp_i, 1] = cur + ll
                doc_lane_se[sp_i, 0] = n
                doc_lane_se[sp_i, 1] = cur
                doc_lane_se[sp_i, 2] = cur + ll
                doc_idxs.append(idx)
                sp_i += 1
                cur += ll
            span_offsets[n + 1] = sp_i

        max_pos = int(self._c256_np.shape[0])
        if self._token_table is not None:
            vocab_sz = int(self._token_table.vocab_size)
        elif self._emb_scaled_np is not None:
            vocab_sz = int(self._emb_scaled_np.shape[0])
        else:
            vocab_sz = 262144

        if N * S > 0:
            oob_tok = (ids < -1) | (ids >= vocab_sz)
            if np.any(oob_tok):
                ids[oob_tok] = -1
            oob_pos = (pos < 0) | (pos >= max_pos)
            if np.any(oob_pos):
                np.clip(pos, 0, max_pos - 1, out=pos)

        if self._token_table is not None and hasattr(self._io, "emb2_prep_batch_i8"):
            self._io.emb2_prep_batch_i8(
                N,
                S,
                self._token_table.emb_i8.ctypes.data,
                self._token_table.scale_f16.ctypes.data,
                vocab_sz,
                max_pos,
                float(self._token_table.embed_scale),
                self._c256_np.ctypes.data,
                self._s256_np.ctypes.data,
                self._c512_np.ctypes.data,
                self._s512_np.ctypes.data,
                ids.ctypes.data,
                pos.ctypes.data,
                span_offsets.ctypes.data,
                spans_se.ctypes.data,
                self._neg_inf_u16,
                scratch.h_all.ctypes.data,
                scratch.b_all.ctypes.data,
                scratch.cs_all.ctypes.data,
                scratch.ss_all.ctypes.data,
                scratch.cf_all.ctypes.data,
                scratch.sf_all.ctypes.data,
            )
        else:
            assert self._emb_scaled_np is not None
            self._io.emb2_prep_batch(
                N,
                S,
                self._emb_scaled_np.ctypes.data,
                self._c256_np.ctypes.data,
                self._s256_np.ctypes.data,
                self._c512_np.ctypes.data,
                self._s512_np.ctypes.data,
                ids.ctypes.data,
                pos.ctypes.data,
                span_offsets.ctypes.data,
                spans_se.ctypes.data,
                self._neg_inf_u16,
                scratch.h_all.ctypes.data,
                scratch.b_all.ctypes.data,
                scratch.cs_all.ctypes.data,
                scratch.ss_all.ctypes.data,
                scratch.cf_all.ctypes.data,
                scratch.sf_all.ctypes.data,
            )
        return N, tot_docs, real_toks, doc_idxs

    def _finish_coreml_lanes(
        self,
        S: int,
        tot_docs: int,
        scratch: CoreMLScratch,
        out_dest: np.ndarray,
        doc_idxs: list[int],
        eff_dim: int,
    ) -> None:
        assert self._io is not None and self._emb_proj_T32_np is not None
        self._io.emb2_pool_project_batch_dim(
            S,
            scratch.out_all.ctypes.data,
            tot_docs,
            scratch.doc_lane_se.ctypes.data,
            self._emb_proj_T32_np.ctypes.data,
            scratch.scratch_pooled.ctypes.data,
            scratch.out_embs.ctypes.data,
            eff_dim,
        )
        out_dest[doc_idxs] = scratch.out_embs.ravel()[: tot_docs * eff_dim].reshape(tot_docs, eff_dim)

    def _exec_coreml_lanes(
        self,
        kind: str,
        S: int,
        lanes: list,
        scratch: CoreMLScratch,
        out_dest: np.ndarray,
        *,
        eff_dim: int = EMBED_DIM,
    ) -> tuple[int, int]:
        assert self._coreml is not None
        handle = self.ane_handles[S] if kind == "ane" else self.cpu_handles[S]
        lock = self._handle_locks[(kind, S)]
        N, tot_docs, real_toks, doc_idxs = self._prep_coreml_lanes(S, lanes, scratch)
        with lock:
            ms = self._coreml.emb2_pk_exec_queue(
                handle,
                N,
                scratch.h_all.ctypes.data,
                scratch.b_all.ctypes.data,
                scratch.cs_all.ctypes.data,
                scratch.ss_all.ctypes.data,
                scratch.cf_all.ctypes.data,
                scratch.sf_all.ctypes.data,
                scratch.out_all.ctypes.data,
            )
            if ms < 0.0:
                detail = _coreml_err_detail(self._coreml)
                raise RuntimeError(f"emb2_pk_exec_queue(kind={kind}, S={S}) failed: {detail}")
        self._finish_coreml_lanes(S, tot_docs, scratch, out_dest, doc_idxs, eff_dim)
        return real_toks, N * S

    def warmup(
        self,
        *,
        prewarm_gpu_shapes: bool = True,
        prewarm_over_shapes: bool = False,
    ) -> None:
        if self.ane_available:
            self.encode_query([2, 3250, 236787, 7293, 1109, 1816, 236787, 1], prefer_ane=True)
        if self.default_mode == "ane" and self.ane_available:
            return
        if self.default_mode == "cpu" and self.cpu_sme_available:
            return
        if getattr(self, "_oi_handles", None) and not self._has_gpu_weights:
            return
        with self._gpu_lock:
            self.gpu.encode_single([2, 3250, 236787, 7293, 1109, 1816, 236787, 1])
            if prewarm_gpu_shapes:
                # Serialize shape warmup sequentially per (L_lane, B) with arena hygiene between
                # buckets so 23 packed graphs do not accumulate intermediates in a single spike.
                for L_l, b_tup in GPU_B_OPTS.items():
                    for B in b_tup:
                        _, _, _, o = self.gpu.exec_packed(L_l, [[(0, [2, 1])]] * B)
                        mx.eval(o)
                    _maybe_clear_mlx_arena(256)
                if prewarm_over_shapes:
                    for ml, b_tup in OVER_B_OPTS.items():
                        for B in b_tup:
                            _, _, _, o = self.gpu.exec_over(ml, [(0, [2, 1])] * B)
                            mx.eval(o)
                        _maybe_clear_mlx_arena(256)
                mx.synchronize()

    def _exec_openinfinity_docs(
        self,
        docs: Sequence[tuple[int, Sequence[int]]],
        out_dest: np.ndarray,
        eff_dim: int,
    ) -> int:
        assert self._coreml is not None and self._token_table is not None
        by_cap: dict[int, list[tuple[int, Sequence[int]]]] = {}
        real_toks = 0
        for idx, tok in docs:
            sl = len(tok)
            real_toks += sl
            chosen_cap = 512
            for cap in _OPENINFINITY_CAPS:
                if sl <= cap and cap in self._oi_handles:
                    chosen_cap = cap
                    break
            by_cap.setdefault(chosen_cap, []).append((idx, tok))

        for cap, group in by_cap.items():
            N = len(group)
            ids_np = np.full((N, cap), -1, dtype=np.int32)
            msk_np = np.zeros((N, cap), dtype=np.float16)
            doc_idxs: list[int] = []
            for i, (idx, tok) in enumerate(group):
                ll = min(len(tok), cap)
                ids_np[i, :ll] = tok[:ll]
                msk_np[i, :ll] = 1.0
                doc_idxs.append(idx)
            emb_f16 = self._token_table.gather_scaled_f16(ids_np)
            out_768 = np.empty((N, EMBED_DIM), dtype=np.float32)
            with self._handle_locks[("oi", cap)]:
                ms = self._coreml.emb2_oi_exec_embed(
                    self._oi_handles[cap],
                    N,
                    emb_f16.ctypes.data,
                    msk_np.ctypes.data,
                    out_768.ctypes.data,
                )
                if ms < 0.0:
                    raise RuntimeError(f"emb2_oi_exec_embed(S={cap}) failed: {_coreml_err_detail(self._coreml)}")
            out_dest[doc_idxs] = truncate_and_normalize(out_768, eff_dim) if eff_dim < EMBED_DIM else out_768
        return real_toks

    def encode_query(
        self,
        token_ids: Sequence[int] | np.ndarray,
        prefer_ane: bool = True,
        *,
        dimensions: int | None = None,
        strict_ane: bool = False,
    ) -> np.ndarray:
        """Low-latency single-query encoder returning a `(dim,)` L2-normalized float32 vector."""
        eff_dim = self._resolve_dim(dimensions)
        arr = np.ascontiguousarray(token_ids[:MAX_SEQ_LEN], dtype=np.int32)
        if arr.shape[0] == 0:
            arr = np.array([self.tokenizer.bos_id, self.tokenizer.eos_id], dtype=np.int32)
        sl = int(arr.shape[0])

        if prefer_ane and self.ane_available and sl <= self.ane_query_max_len:
            if getattr(self, "_oi_handles", None):
                out = np.empty((1, eff_dim), dtype=np.float32)
                self._exec_openinfinity_docs([(0, arr)], out, eff_dim)
                return out[0]
            assert self._query_scratch is not None
            for cap in _QUERY_CAPS:
                if sl <= cap and cap in self.ane_handles:
                    out = np.empty((1, eff_dim), dtype=np.float32)
                    self._exec_coreml_lanes("ane", cap, [[(0, arr)]], self._query_scratch, out, eff_dim=eff_dim)
                    return out[0]

        if strict_ane:
            raise ValueError(
                f"Strict ANE mode cannot encode query of length {sl} (max ANE length is {self.ane_query_max_len}); "
                "use mode='hetero'/'gpu' or truncate to <=512 tokens."
            )
        if getattr(self, "_oi_handles", None) and not self._has_gpu_weights:
            raise ValueError(
                f"OpenInfinity ANE bundles cap at {self.ane_query_max_len} tokens (got {sl}) and GPU weights are not staged at {self.paths.weights_npz}."
            )

        with self._gpu_lock:
            if isinstance(self.gpu, FastEmbeddingGemma2GPU):
                return self.gpu.encode_single(arr, dimensions=eff_dim)
            vec = self.gpu.encode_single(arr)
        return truncate_and_normalize(vec, eff_dim) if vec.shape[0] != eff_dim else vec

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        prefix: str | None = None,
        prompt_name: str | None = None,
        mode: str | None = None,
        max_length: int | None = None,
        dimensions: int | None = None,
    ) -> np.ndarray:
        """Embed a single string `(dim,)` or sequence of strings `[N, dim]` into L2-normalized float32 vectors."""
        eff_dim = self._resolve_dim(dimensions)
        eff_mode = self._resolve_mode(mode or self.default_mode)
        prefer_ane = eff_mode != "gpu"
        strict_ane = eff_mode == "ane"
        ml = MAX_SEQ_LEN if max_length is None else min(MAX_SEQ_LEN, int(max_length))

        if isinstance(texts, str):
            ids = self.tokenizer.encode(
                texts,
                prefix=prefix,
                prompt_name=prompt_name,
                add_query_prefix=add_query_prefix,
                add_document_prefix=add_document_prefix,
                max_length=ml,
            )
            return self.encode_query(ids, prefer_ane=prefer_ane, dimensions=eff_dim, strict_ane=strict_ane)

        items = list(texts)
        if not items:
            return np.empty((0, eff_dim), dtype=np.float32)

        tok_lists = self.tokenizer.encode_batch(
            items,
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
            max_length=ml,
        )
        if is_query or len(tok_lists) == 1:
            if len(tok_lists) == 1:
                return self.encode_query(tok_lists[0], prefer_ane=prefer_ane, dimensions=eff_dim, strict_ane=strict_ane)[None, :]
            return np.stack(
                [self.encode_query(ids, prefer_ane=prefer_ane, dimensions=eff_dim, strict_ane=strict_ane) for ids in tok_lists],
                axis=0,
            )

        return self.encode_corpus(tok_lists, mode=eff_mode, dimensions=eff_dim)

    def encode_corpus(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        mode: str | None = None,
        dimensions: int | None = None,
        allow_gpu_overflow: bool = False,
    ) -> np.ndarray:
        """Batch-embed pre-tokenized sequences `[N]` into `[N, dim]` float32 L2-normalized vectors.

        When `mode="ane"` or `mode="cpu"`, sequences exceeding `512` tokens raise `ValueError`
        unless `allow_gpu_overflow=True` is explicitly set (preventing silent GPU execution).
        """
        eff_dim = self._resolve_dim(dimensions)
        n = len(token_lists)
        if n == 0:
            self.last_stats = {"gpu_toks": 0, "ane_toks": 0, "cpu_toks": 0, "gpu_docs": 0, "ane_docs": 0, "cpu_docs": 0}
            return np.empty((0, eff_dim), dtype=np.float32)

        eff_mode = self._resolve_mode(mode or self.default_mode)
        empty_seq = (self.tokenizer.bos_id, self.tokenizer.eos_id)
        seqs = [
            (i, t[:MAX_SEQ_LEN] if len(t) > 0 else empty_seq)
            for i, t in enumerate(token_lists)
        ]

        if len(seqs) == 1 and eff_mode in ("gpu", "hetero", "hetero2", "hetero3"):
            v = self.encode_query(seqs[0][1], prefer_ane=(eff_mode != "gpu"), dimensions=eff_dim)
            on_ane = eff_mode != "gpu" and self.ane_available and len(seqs[0][1]) <= self.ane_query_max_len
            self.last_stats = {
                "gpu_toks": 0 if on_ane else len(seqs[0][1]),
                "ane_toks": len(seqs[0][1]) if on_ane else 0,
                "cpu_toks": 0,
                "gpu_docs": 0 if on_ane else 1,
                "ane_docs": 1 if on_ane else 0,
                "cpu_docs": 0,
            }
            return v[None, :]

        outs = np.empty((n, eff_dim), dtype=np.float32)
        total_toks = sum(len(t) for _, t in seqs)

        with self._corpus_lock:
            if getattr(self, "_oi_handles", None) and eff_mode in ("ane", "hetero", "hetero2", "hetero3"):
                max_oi_cap = max(self._oi_handles)
                over_oi = [s for s in seqs if len(s[1]) > max_oi_cap]
                if over_oi:
                    if eff_mode == "ane" and not allow_gpu_overflow:
                        raise ValueError(
                            f"mode='ane' received {len(over_oi)} sequence(s) longer than {max_oi_cap} tokens; "
                            "OpenInfinity ANE functions cap at 512 tokens."
                        )
                    if not self._has_gpu_weights:
                        raise ValueError(
                            f"OpenInfinity ANE functions cap at {max_oi_cap} tokens (got {len(over_oi)} longer sequence(s)) "
                            f"and GPU weights are not staged at {self.paths.weights_npz}."
                        )
                le_oi = [s for s in seqs if len(s[1]) <= max_oi_cap]
                ane_rt = self._exec_openinfinity_docs(le_oi, outs, eff_dim) if le_oi else 0
                gpu_rt = 0
                if over_oi:
                    with self._gpu_lock:
                        for ml, chk in build_over_chunks(over_oi):
                            rt, _, di, cur_o = self.gpu.exec_over(ml, chk, dimensions=eff_dim)
                            mx.eval(cur_o)
                            outs[di] = np.asarray(cur_o, dtype=np.float32)
                            gpu_rt += rt
                self.last_stats = {
                    "gpu_toks": gpu_rt,
                    "ane_toks": ane_rt,
                    "cpu_toks": 0,
                    "gpu_docs": len(over_oi),
                    "ane_docs": len(le_oi),
                    "cpu_docs": 0,
                }
                return outs

            by_b, over = fast_pack_cost_aware(seqs, ANE_MS_COST)
            over_q = build_over_chunks(over)
            if len(over_q) > 0 and eff_mode in ("ane", "cpu") and not allow_gpu_overflow:
                raise ValueError(
                    f"mode={eff_mode!r} received {len(over)} sequence(s) longer than 512 tokens; "
                    "ANE/CPU CoreML bundles cap at 512 tokens. Use mode='hetero'/'gpu', truncate to <=512, "
                    "or pass allow_gpu_overflow=True to explicitly route >512 overflow onto GPU."
                )

            qlock = threading.Lock()
            use_ane = eff_mode in ("hetero", "hetero2", "ane") and bool(self.ane_handles)
            use_cpu = (eff_mode == "cpu") and self.cpu_sme_available
            use_gpu = eff_mode in ("hetero", "hetero2", "gpu") or (len(over_q) > 0)

            stats = {"gpu": 0, "ane": 0, "cpu": 0, "gpu_docs": 0, "ane_docs": 0, "cpu_docs": 0}
            errors: list[BaseException] = []

            def pop_lanes(L_l: int, count: int) -> list:
                q = by_b[L_l]
                take = min(count, len(q))
                return [q.popleft() for _ in range(take)]

            def total_shared_rem() -> int:
                return sum(len(q) for q in by_b.values())

            def gpu_worker() -> None:
                try:
                    prev_o: mx.array | None = None
                    prev_idxs: list[int] | None = None
                    gpu_only_mode = (eff_mode == "gpu") or (not use_ane and not use_cpu)
                    gpu_pref = sorted(by_b.keys()) if gpu_only_mode else (448, 320, 192, 512, 384, 256, 128)
                    with self._gpu_lock:
                        while not errors:
                            job = None
                            with qlock:
                                if over_q:
                                    ml, chk = over_q.popleft()
                                    job = ("over", ml, chk)
                                elif eff_mode in ("hetero", "hetero2", "hetero3", "gpu"):
                                    for L_l in gpu_pref:
                                        q_b = by_b.get(L_l)
                                        n_av = len(q_b) if q_b is not None else 0
                                        if n_av == 0:
                                            continue
                                        opts = GPU_B_OPTS[L_l]
                                        cand_opts = opts if (gpu_only_mode or not use_ane) else opts[:-1]
                                        chosen = 0
                                        for opt in cand_opts:
                                            if opt <= n_av:
                                                chosen = opt
                                                break
                                        if chosen > 0:
                                            job = ("packed", L_l, pop_lanes(L_l, chosen))
                                            break
                            if job is None:
                                break
                            kind, param, chk = job
                            if kind == "over":
                                rt, _, di, cur_o = self.gpu.exec_over(param, chk, dimensions=eff_dim)
                            else:
                                rt, _, di, cur_o = self.gpu.exec_packed(param, chk, dimensions=eff_dim)
                            mx.async_eval(cur_o)
                            if prev_o is not None and prev_idxs is not None:
                                outs[prev_idxs] = np.asarray(prev_o, dtype=np.float32)
                            prev_o, prev_idxs = cur_o, di
                            stats["gpu"] += rt
                            stats["gpu_docs"] += len(di)
                        if prev_o is not None and prev_idxs is not None:
                            outs[prev_idxs] = np.asarray(prev_o, dtype=np.float32)
                        mx.synchronize()
                        _maybe_clear_mlx_arena(self._mlx_cache_limit_mb or 512)
                except BaseException as exc:
                    errors.append(exc)

            def ane_worker() -> None:
                assert self._ane_scratch is not None and self._ane_scratch_b is not None
                assert self._coreml is not None
                ane_pref = [(128, 24), (192, 20), (256, 20), (320, 16), (384, 16), (448, 12), (512, 12)]
                fallback = [(192, 256), (320, 384), (448, 512), (128, 128), (256, 256), (384, 384), (512, 512)]
                scratches = (self._ane_scratch, self._ane_scratch_b)
                buf_idx = 0
                inflight = None

                def pop_ane_job():
                    with qlock:
                        rem_sh = total_shared_rem()
                        for S, batch_n in ane_pref:
                            if S not in self.ane_handles:
                                continue
                            n_av = len(by_b.get(S, ()))
                            if n_av > 0:
                                if eff_mode == "ane":
                                    n_take = min(batch_n, n_av)
                                elif rem_sh <= 20:
                                    n_take = max(4, min(batch_n, (n_av + 1) // 2)) if n_av > 4 else n_av
                                elif rem_sh <= 48:
                                    n_take = min(batch_n, max(8, (n_av + 1) // 2))
                                else:
                                    n_take = min(batch_n, n_av)
                                chk = pop_lanes(S, n_take)
                                S_exec = 64 if (S == 128 and 64 in self.ane_handles and all(sum(len(t) for _, t in ln) <= 64 for ln in chk)) else S
                                return (S_exec, chk)
                        for src_b, s_tgt in fallback:
                            if s_tgt not in self.ane_handles:
                                continue
                            n_av = len(by_b.get(src_b, ()))
                            if n_av > 0:
                                n_take = min(8, n_av) if eff_mode == "ane" else min(8, max(1, (n_av + 1) // 2))
                                return (s_tgt, pop_lanes(src_b, n_take))
                    return None

                try:
                    while not errors:
                        job = pop_ane_job()
                        if job is None:
                            break
                        S_exec, chk = job
                        cur_sc = scratches[buf_idx]
                        buf_idx ^= 1
                        N_c, tot_docs_c, real_toks_c, doc_idxs_c = self._prep_coreml_lanes(S_exec, chk, cur_sc)

                        prev = inflight
                        inflight = None
                        if prev is not None:
                            p_lock, p_flight, p_S, p_tot_docs, p_doc_idxs, p_rt, p_sc = prev
                            try:
                                wait_ms = self._coreml.emb2_pk_wait_queue(p_flight)
                            finally:
                                p_lock.release()
                            if wait_ms < 0.0:
                                raise RuntimeError(
                                    f"emb2_pk_wait_queue(ane, S={p_S}) failed: {_coreml_err_detail(self._coreml)}"
                                )

                        c_lock = self._handle_locks[("ane", S_exec)]
                        c_handle = self.ane_handles[S_exec]
                        c_lock.acquire()
                        try:
                            c_flight = self._coreml.emb2_pk_submit_queue(
                                c_handle,
                                N_c,
                                cur_sc.h_all.ctypes.data,
                                cur_sc.b_all.ctypes.data,
                                cur_sc.cs_all.ctypes.data,
                                cur_sc.ss_all.ctypes.data,
                                cur_sc.cf_all.ctypes.data,
                                cur_sc.sf_all.ctypes.data,
                                cur_sc.out_all.ctypes.data,
                            )
                            if not c_flight:
                                raise RuntimeError(
                                    f"emb2_pk_submit_queue(ane, S={S_exec}) returned null: {_coreml_err_detail(self._coreml)}"
                                )
                        except BaseException:
                            c_lock.release()
                            raise
                        inflight = (c_lock, c_flight, S_exec, tot_docs_c, doc_idxs_c, real_toks_c, cur_sc)

                        if prev is not None:
                            self._finish_coreml_lanes(p_S, p_tot_docs, p_sc, outs, p_doc_idxs, eff_dim)
                            stats["ane"] += p_rt
                            stats["ane_docs"] += p_tot_docs

                    if inflight is not None:
                        p_lock, p_flight, p_S, p_tot_docs, p_doc_idxs, p_rt, p_sc = inflight
                        inflight = None
                        try:
                            wait_ms = self._coreml.emb2_pk_wait_queue(p_flight)
                        finally:
                            p_lock.release()
                        if wait_ms < 0.0:
                            raise RuntimeError(
                                f"emb2_pk_wait_queue(ane, S={p_S}) failed: {_coreml_err_detail(self._coreml)}"
                            )
                        self._finish_coreml_lanes(p_S, p_tot_docs, p_sc, outs, p_doc_idxs, eff_dim)
                        stats["ane"] += p_rt
                        stats["ane_docs"] += p_tot_docs
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    if inflight is not None:
                        p_lock, p_flight, *_ = inflight
                        try:
                            self._coreml.emb2_pk_wait_queue(p_flight)
                        finally:
                            p_lock.release()

            def cpu_worker() -> None:
                try:
                    assert self._cpu_scratch is not None
                    cpu_pref_hi = [(256, 16), (384, 12), (512, 8)]
                    cpu_pref_lo = [(256, 8), (384, 6), (512, 4)]
                    cpu_fallback = [(128, 256), (192, 256), (320, 384), (448, 512), (256, 256), (384, 384), (512, 512)]
                    cpu_t0 = time.perf_counter()
                    while not errors:
                        # Bound CPU-SME whenever co-executing with GPU (with or without ANE).
                        if use_gpu and _should_stop_hetero_cpu(
                            time.perf_counter() - cpu_t0, stats["cpu_docs"], stats["cpu"], eff_mode=eff_mode
                        ):
                            break
                        job = None
                        with qlock:
                            rem_sh = total_shared_rem()
                            cpu_pref = cpu_pref_hi if rem_sh > 64 else cpu_pref_lo
                            for S, batch_n in cpu_pref:
                                if S not in self.cpu_handles:
                                    continue
                                n_av = len(by_b.get(S, ()))
                                if n_av > 0:
                                    if eff_mode == "cpu":
                                        n_take = min(batch_n, n_av)
                                    elif rem_sh <= 20:
                                        n_take = max(4, min(batch_n, (n_av + 2) // 3)) if n_av > 4 else n_av
                                    else:
                                        n_take = min(batch_n, n_av)
                                    job = (S, pop_lanes(S, n_take))
                                    break
                            if job is None and (eff_mode == "cpu" or not use_ane):
                                for src_b, s_tgt in cpu_fallback:
                                    if s_tgt not in self.cpu_handles:
                                        continue
                                    n_av = len(by_b.get(src_b, ()))
                                    if n_av > 0:
                                        n_take = min(8, n_av) if eff_mode == "cpu" else min(4, max(1, (n_av + 1) // 2))
                                        job = (s_tgt, pop_lanes(src_b, n_take))
                                        break
                            elif job is None:
                                for src_b, s_tgt in ((192, 256), (320, 384), (448, 512)):
                                    if s_tgt not in self.cpu_handles:
                                        continue
                                    n_av = len(by_b.get(src_b, ()))
                                    if n_av > 0:
                                        n_take = min(4, max(1, (n_av + 1) // 2))
                                        job = (s_tgt, pop_lanes(src_b, n_take))
                                        break
                        if job is None:
                            break
                        S, chk = job
                        rt, _ = self._exec_coreml_lanes("cpu", S, chk, self._cpu_scratch, outs, eff_dim=eff_dim)
                        stats["cpu"] += rt
                        stats["cpu_docs"] += sum(len(ln) for ln in chk)
                except BaseException as exc:
                    errors.append(exc)

            threads: list[threading.Thread] = []
            if use_gpu:
                threads.append(threading.Thread(target=gpu_worker, name="emb2-gpu"))
            if use_ane:
                threads.append(threading.Thread(target=ane_worker, name="emb2-ane"))
            if use_cpu:
                threads.append(threading.Thread(target=cpu_worker, name="emb2-cpu"))
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            if errors:
                raise errors[0]

            self.last_stats = {
                "gpu_toks": stats["gpu"],
                "ane_toks": stats["ane"],
                "cpu_toks": stats["cpu"],
                "gpu_docs": stats["gpu_docs"],
                "ane_docs": stats["ane_docs"],
                "cpu_docs": stats["cpu_docs"],
            }
        return outs
