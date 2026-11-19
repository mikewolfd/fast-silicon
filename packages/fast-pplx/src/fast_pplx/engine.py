"""Unified Apple Silicon heterogeneous (Metal MLX GPU + 16-core ANE) engine for pplx-embed-v2-context-0.6b."""

from __future__ import annotations

from collections import deque
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any, Sequence

import numpy as np
from fast_shared.embeddings import resolve_embedding_dim, truncate_and_normalize

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .gpu import FastPplxGPU, _find_local_hf_snapshot
from .paths import RepoPaths, resolve_paths
from .tokenizer import ContextualTokenizer

EMBED_DIM = 1024
MAX_EMBED_DIM = 1024
MRL_DIMENSIONS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_CHUNK_SIZE = 32
DEFAULT_ROPE_HALF = 32
DEFAULT_PIPE_DEPTH = 4
DEFAULT_ANE_MAX_SEQ_LEN = 512
_VALID_MODES: tuple[str, ...] = ("auto", "gpu", "ane", "hetero", "hetero2")


@dataclass(frozen=True)
class PackedSpanTarget:
    item_index: int
    chunk_index: int
    start: int
    end: int


@dataclass
class PackedBucket:
    seq_len: int
    token_ids: np.ndarray  # [S] int32
    doc_ids: np.ndarray    # [S] int32 (-1 = PAD)
    pos_ids: np.ndarray    # [S] int32
    targets: list[PackedSpanTarget]
    n_docs: int
    used_tokens: int


def pack_ane_buckets(
    items: Sequence[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]],
    *,
    supported_caps: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    conv_gap: int = 3,
) -> list[PackedBucket]:
    """Pack short `<= max_cap` token sequences into fixed ANE buckets with strict doc isolation.

    Each packed document after the first starts at a `chunk_size` boundary with at least
    `conv_gap` (`3` for `conv1d(k=4)`) zero-pad tokens after the previous document end.
    `doc_ids[pad] = -1`, `pos_ids` reset to `0..L-1` per doc, and empty spans stay `(0, 0)`.
    """
    caps = tuple(sorted(int(c) for c in supported_caps))
    if not caps:
        raise ValueError("supported_caps must be non-empty")
    max_cap = caps[-1]
    if not items:
        return []

    buckets: list[PackedBucket] = []
    cur_pos = 0
    doc_slot = 0
    cur_toks: list[tuple[int, Sequence[int], int]] = []  # (start, ids, doc_slot)
    cur_targets: list[PackedSpanTarget] = []

    def _select_cap(used: int) -> int:
        for cap in caps:
            if cap >= used:
                return cap
        return max_cap

    def _flush() -> None:
        nonlocal cur_pos, doc_slot, cur_toks, cur_targets
        if not cur_toks and not cur_targets:
            return
        cap = _select_cap(max(1, cur_pos))
        tok_arr = np.zeros((cap,), dtype=np.int32)
        doc_arr = np.full((cap,), -1, dtype=np.int32)
        pos_arr = np.zeros((cap,), dtype=np.int32)
        for st, ids, d_slot in cur_toks:
            ln = len(ids)
            if ln > 0:
                tok_arr[st : st + ln] = np.asarray(ids, dtype=np.int32)
                doc_arr[st : st + ln] = d_slot
                pos_arr[st : st + ln] = np.arange(ln, dtype=np.int32)
        buckets.append(
            PackedBucket(
                seq_len=cap,
                token_ids=tok_arr,
                doc_ids=doc_arr,
                pos_ids=pos_arr,
                targets=cur_targets,
                n_docs=doc_slot,
                used_tokens=cur_pos,
            )
        )
        cur_pos = 0
        doc_slot = 0
        cur_toks = []
        cur_targets = []

    for orig_idx, tok_ids, spans in items:
        n_tok = len(tok_ids)
        if n_tok > max_cap:
            raise ValueError(f"Item {orig_idx} length ({n_tok}) exceeds max ANE bucket ({max_cap})")

        if n_tok == 0:
            # No token run needed; record empty zero-vector target(s).
            n_sp = 1 if spans is None else len(spans)
            for c_idx in range(n_sp):
                cur_targets.append(PackedSpanTarget(orig_idx, c_idx, 0, 0))
            continue

        if doc_slot == 0:
            start = 0
        else:
            min_next = cur_pos + conv_gap
            start = ((min_next + chunk_size - 1) // chunk_size) * chunk_size

        if doc_slot > 0 and start + n_tok > max_cap:
            _flush()
            start = 0

        cur_toks.append((start, tok_ids, doc_slot))
        if spans is None:
            cur_targets.append(PackedSpanTarget(orig_idx, 0, start, start + n_tok))
        else:
            for c_idx, (s0, s1) in enumerate(spans):
                if s1 > s0:
                    cur_targets.append(
                        PackedSpanTarget(orig_idx, c_idx, start + int(s0), start + min(int(s1), n_tok))
                    )
                else:
                    cur_targets.append(PackedSpanTarget(orig_idx, c_idx, 0, 0))

        cur_pos = start + n_tok
        doc_slot += 1

    _flush()
    return buckets


def _load_io_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.pplx_build_rope_table_f16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.pplx_build_rope_table_f16.restype = None

    lib.pplx_pack_bucket_f16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.pplx_pack_bucket_f16.restype = None

    lib.pplx_pool_and_project_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.pplx_pool_and_project_f32.restype = None

    lib.pplx_l2_normalize_f32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.pplx_l2_normalize_f32.restype = None
    return lib


def _load_coreml_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.pplx_ane_load_stages.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.pplx_ane_load_stages.restype = ctypes.c_void_p

    lib.pplx_ane_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.pplx_ane_submit.restype = ctypes.c_void_p

    lib.pplx_ane_wait.argtypes = [ctypes.c_void_p]
    lib.pplx_ane_wait.restype = ctypes.c_double

    lib.pplx_ane_exec.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.pplx_ane_exec.restype = ctypes.c_double

    lib.pplx_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.pplx_ane_inspect_plan.restype = ctypes.c_int

    lib.pplx_ane_free.argtypes = [ctypes.c_void_p]
    lib.pplx_ane_free.restype = None
    return lib


def _discover_ane_stage_paths(ane_dir: Path) -> dict[int, list[Path]]:
    """Discover per-seq-len staged .mlmodelc bundle chains from manifest.json or glob fallback."""
    if not ane_dir.is_dir():
        return {}
    manifest_path = ane_dir / "manifest.json"
    discovered: dict[int, list[Path]] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k_str, entry in manifest.get("buckets", {}).items():
            cap = int(entry.get("seq_len", k_str))
            stages = [ane_dir / name for name in entry.get("stages", [])]
            if stages and all(p.exists() for p in stages):
                discovered[cap] = stages
        if discovered:
            return dict(sorted(discovered.items()))

    # Fallback: scan pplx_s{S}_st{idx}_*.mlmodelc
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for p in sorted(ane_dir.glob("pplx_s*_st*.mlmodelc")):
        parts = p.stem.split("_")
        try:
            cap = int(parts[1][1:])
            st_idx = int(parts[2][2:])
        except (IndexError, ValueError):
            continue
        grouped.setdefault(cap, []).append((st_idx, p))
    for cap, pairs in sorted(grouped.items()):
        pairs.sort(key=lambda t: t[0])
        if [idx for idx, _ in pairs] == list(range(len(pairs))):
            discovered[cap] = [p for _, p in pairs]
    return discovered


def _load_f16_memmap_from_safetensors(sf_path: Path) -> tuple[np.ndarray, int, int]:
    """Open read-only F16 numpy memmap over embed_tokens safetensors."""
    with sf_path.open("rb") as fp:
        hdr_len = struct.unpack("<Q", fp.read(8))[0]
        hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
    for key in ("language_model.embed_tokens.weight", "model.embed_tokens.weight", "embed_tokens.weight", "weight"):
        entry = hdr.get(key)
        if isinstance(entry, dict) and entry.get("dtype") == "F16":
            shape = entry["shape"]
            b0 = int(entry["data_offsets"][0])
            mm = np.memmap(
                sf_path,
                dtype=np.dtype("<f2"),
                mode="r",
                offset=8 + hdr_len + b0,
                shape=(int(shape[0]), int(shape[1])),
            )
            return mm, int(shape[0]), int(shape[1])
    raise KeyError(f"No F16 embedding matrix found in {sf_path}")


class _ANERunnerBackend:
    """Thread-safe double/quad-buffered C++ + CoreML ANE backend using staged .mlmodelc bundles."""

    def __init__(
        self,
        paths: RepoPaths,
        config: PplxConfig,
        *,
        contextual_proj_f32: np.ndarray,
        embed_f16_mm: np.ndarray,
        pipe_depth: int = DEFAULT_PIPE_DEPTH,
        compute_units: int = 0,
    ) -> None:
        self.io_lib = _load_io_lib(paths.io_dylib)
        self.coreml_lib = _load_coreml_lib(paths.coreml_dylib)
        self.config = config
        self.pipe_depth = max(1, min(int(pipe_depth), 8))
        self.embed_mm = embed_f16_mm
        self.vocab_size, self.hidden_dim = int(embed_f16_mm.shape[0]), int(embed_f16_mm.shape[1])
        self.contextual_proj_f32 = np.ascontiguousarray(contextual_proj_f32, dtype=np.float32)

        stage_map = _discover_ane_stage_paths(paths.ane_models_dir)
        if not stage_map:
            raise FileNotFoundError(f"No compiled ANE .mlmodelc stages found in {paths.ane_models_dir}")

        self.runners: dict[int, int] = {}
        try:
            for cap, stage_paths in stage_map.items():
                arr = (ctypes.c_char_p * len(stage_paths))(
                    *(str(p).encode("utf-8") for p in stage_paths)
                )
                handle = self.coreml_lib.pplx_ane_load_stages(
                    arr,
                    len(stage_paths),
                    cap,
                    config.hidden_size,
                    DEFAULT_CHUNK_SIZE,
                    DEFAULT_ROPE_HALF,
                    self.pipe_depth,
                    compute_units,
                )
                if not handle:
                    raise RuntimeError(f"Failed loading ANE pipeline for S={cap} ({stage_paths})")
                self.runners[cap] = int(handle)
        except Exception:
            self.close()
            raise

        self.caps: tuple[int, ...] = tuple(sorted(self.runners))
        self.max_cap: int = self.caps[-1]

        self.rope_cos_u16 = np.empty((self.max_cap, DEFAULT_ROPE_HALF), dtype=np.uint16)
        self.rope_sin_u16 = np.empty((self.max_cap, DEFAULT_ROPE_HALF), dtype=np.uint16)
        self.io_lib.pplx_build_rope_table_f16(
            self.rope_cos_u16.ctypes.data,
            self.rope_sin_u16.ctypes.data,
            self.max_cap,
            DEFAULT_ROPE_HALF,
            float(config.rope_theta),
        )

    def run_buckets(
        self,
        buckets: Sequence[PackedBucket],
        *,
        out_dim: int,
        apply_tanh_int8: bool,
        apply_l2_norm: bool,
    ) -> list[tuple[PackedBucket, np.ndarray]]:
        """Execute packed buckets through the async `pipe_depth` ANE ring and project in F32."""
        if not buckets:
            return []

        embed_ptr = int(self.embed_mm.ctypes.data)
        cos_ptr = int(self.rope_cos_u16.ctypes.data)
        sin_ptr = int(self.rope_sin_u16.ctypes.data)
        proj_ptr = int(self.contextual_proj_f32.ctypes.data)
        proj_rows = int(self.contextual_proj_f32.shape[0])

        # Maintain an async inflight window up to pipe_depth overlapping CPU prep + ANE exec.
        inflight: deque[tuple[PackedBucket, np.ndarray, int]] = deque()
        completed: list[tuple[PackedBucket, np.ndarray]] = []

        def _drain_one() -> None:
            bkt, h_out_u16, job = inflight.popleft()
            dt = self.coreml_lib.pplx_ane_wait(ctypes.c_void_p(job))
            if dt < 0.0:
                raise RuntimeError(f"ANE prediction failed for bucket S={bkt.seq_len}")
            live_targets = [t for t in bkt.targets if t.end > t.start]
            if not live_targets:
                completed.append((bkt, np.zeros((0, out_dim), dtype=np.float32)))
                return
            starts = np.array([t.start for t in live_targets], dtype=np.int32)
            ends = np.array([t.end for t in live_targets], dtype=np.int32)
            out_f32 = np.zeros((len(live_targets), out_dim), dtype=np.float32)
            # Final stage already executed final RMSNorm -> pass final_norm_f32=None (0)
            self.io_lib.pplx_pool_and_project_f32(
                h_out_u16.ctypes.data,
                self.hidden_dim,
                bkt.seq_len,
                len(live_targets),
                starts.ctypes.data,
                ends.ctypes.data,
                None,
                proj_ptr,
                proj_rows,
                out_dim,
                1 if apply_tanh_int8 else 0,
                1 if apply_l2_norm else 0,
                out_f32.ctypes.data,
            )
            completed.append((bkt, out_f32))

        try:
            for bkt in buckets:
                if len(inflight) >= self.pipe_depth:
                    _drain_one()

                seq_len = bkt.seq_len
                if seq_len not in self.runners:
                    raise KeyError(f"No ANE runner loaded for bucket S={seq_len} (have {self.caps})")

                # Staging buffers (copied immediately inside pplx_ane_submit before it returns).
                h_in_u16 = np.empty((self.hidden_dim, seq_len), dtype=np.uint16)
                bias_u16 = np.empty((seq_len, seq_len), dtype=np.uint16)
                c32_u16 = np.empty((seq_len, DEFAULT_ROPE_HALF), dtype=np.uint16)
                s32_u16 = np.empty((seq_len, DEFAULT_ROPE_HALF), dtype=np.uint16)
                tvalid_u16 = np.empty((seq_len,), dtype=np.uint16)
                carry_u16 = np.empty((seq_len // DEFAULT_CHUNK_SIZE,), dtype=np.uint16)
                h_out_u16 = np.empty((self.hidden_dim, seq_len), dtype=np.uint16)

                self.io_lib.pplx_pack_bucket_f16(
                    embed_ptr,
                    self.vocab_size,
                    self.hidden_dim,
                    seq_len,
                    DEFAULT_CHUNK_SIZE,
                    DEFAULT_ROPE_HALF,
                    bkt.token_ids.ctypes.data,
                    bkt.doc_ids.ctypes.data,
                    bkt.pos_ids.ctypes.data,
                    cos_ptr,
                    sin_ptr,
                    self.max_cap,
                    h_in_u16.ctypes.data,
                    bias_u16.ctypes.data,
                    c32_u16.ctypes.data,
                    s32_u16.ctypes.data,
                    tvalid_u16.ctypes.data,
                    carry_u16.ctypes.data,
                )

                job = self.coreml_lib.pplx_ane_submit(
                    ctypes.c_void_p(self.runners[seq_len]),
                    h_in_u16.ctypes.data,
                    bias_u16.ctypes.data,
                    c32_u16.ctypes.data,
                    s32_u16.ctypes.data,
                    tvalid_u16.ctypes.data,
                    carry_u16.ctypes.data,
                    h_out_u16.ctypes.data,
                )
                if not job:
                    raise RuntimeError(f"pplx_ane_submit returned NULL for S={seq_len}")
                inflight.append((bkt, h_out_u16, int(job)))

            while inflight:
                _drain_one()
        except Exception:
            while inflight:
                _, _, job = inflight.popleft()
                try:
                    self.coreml_lib.pplx_ane_wait(ctypes.c_void_p(job))
                except Exception:
                    pass
            raise

        return completed

    def close(self) -> None:
        runners = getattr(self, "runners", None)
        if runners:
            for handle in runners.values():
                try:
                    self.coreml_lib.pplx_ane_free(ctypes.c_void_p(handle))
                except Exception:
                    pass
            runners.clear()
        mm = getattr(self, "embed_mm", None)
        if isinstance(mm, np.memmap):
            backing = getattr(mm, "_mmap", None)
            if backing is not None and not getattr(backing, "closed", False):
                backing.close()
        self.embed_mm = None


class FastPplx:
    """Unified M4 Pro `pplx-embed-v2-context-0.6b` engine (Metal MLX GPU + 16-core ANE `hetero2`)."""

    model_name: str = DEFAULT_MODEL_NAME
    max_dim: int = MAX_EMBED_DIM
    supports_dimensions: bool = True
    mrl_dimensions: tuple[int, ...] = MRL_DIMENSIONS
    cpu_sme_available: bool = False

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        mode: str = "auto",
        model_dir: str | Path | None = None,
        dtype: str = "float16",
        use_mapped_embedding: bool = True,
        cache_limit_mb: int = 512,
        local_files_only: bool = True,
        enable_ane: bool = True,
        dimensions: int | None = None,
        pipe_depth: int = DEFAULT_PIPE_DEPTH,
        compute_units: int = 0,
        _gpu_backend: Any | None = None,
        _ane_backend: Any | None = None,
        _tokenizer: Any | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; choose from {_VALID_MODES}")

        self.paths: RepoPaths = resolve_paths(home)
        self.dtype_name = dtype
        self.pipe_depth = int(pipe_depth)
        self.last_stats: dict[str, Any] = {}

        # 1. GPU / MLX backend (always loaded unless explicitly testing ANE-only with mock)
        self.gpu: Any = _gpu_backend
        if self.gpu is None and mode != "ane":
            self.gpu = FastPplxGPU(
                home=self.paths.home,
                model_dir=model_dir,
                dtype=dtype,
                use_mapped_embedding=use_mapped_embedding,
                cache_limit_mb=cache_limit_mb,
                local_files_only=local_files_only,
            )

        if _tokenizer is not None:
            self.tokenizer: Any = _tokenizer
            self.config: PplxConfig = getattr(_tokenizer, "config", DEFAULT_PPLX_CONFIG)
        elif self.gpu is not None:
            self.tokenizer = self.gpu.tokenizer
            self.config = self.gpu.config
        else:
            self.config = self._load_config(model_dir)
            tok_path = self._find_tokenizer_json(model_dir)
            self.tokenizer = ContextualTokenizer(tok_path, self.config)

        self.dim: int = resolve_embedding_dim(dimensions, max_dim=MAX_EMBED_DIM, supports_dimensions=True)
        self.max_seq_len: int = int(getattr(self.gpu, "max_seq_len", 32768))

        # 2. Optional ANE staged runner
        self.ane: Any = _ane_backend
        self._ane_init_error: str | None = None
        want_ane = enable_ane and mode in ("auto", "ane", "hetero", "hetero2")
        if want_ane and self.ane is None:
            self.ane = self._try_init_ane(pipe_depth=self.pipe_depth, compute_units=compute_units)
            if self.ane is None and mode == "ane":
                raise FileNotFoundError(
                    self._ane_init_error
                    or f"ANE mode requested but CoreML runner/bundles missing under {self.paths.ane_models_dir}"
                )

        if mode in ("auto", "gpu") and self.gpu is None and self.ane is None:
            raise RuntimeError("Neither GPU nor ANE backend is available.")

        self.ane_available: bool = self.ane is not None
        self.ane_query_max_len: int = (
            int(getattr(self.ane, "max_cap", DEFAULT_ANE_MAX_SEQ_LEN))
            if self.ane_available
            else DEFAULT_ANE_MAX_SEQ_LEN
        )
        if mode == "auto":
            self.default_mode = "hetero2" if self.ane_available else "gpu"
        elif mode == "ane":
            self.default_mode = "ane"
        elif mode in ("hetero", "hetero2"):
            self.default_mode = mode if self.ane_available else "gpu"
        else:
            self.default_mode = "gpu"

    def _load_config(self, model_dir: str | Path | None) -> PplxConfig:
        for cand in (
            Path(model_dir) / "config.json" if model_dir is not None else None,
            self.paths.tokenizer_dir / "config.json",
            self.paths.model_dir / "config.json",
        ):
            if cand is not None and cand.is_file():
                return PplxConfig.from_json(cand)
        return DEFAULT_PPLX_CONFIG

    def _find_tokenizer_json(self, model_dir: str | Path | None) -> Path:
        for cand in (
            Path(model_dir) / "tokenizer.json" if model_dir is not None else None,
            self.paths.tokenizer_json,
            self.paths.model_dir / "tokenizer.json",
        ):
            if cand is not None and cand.is_file():
                return cand
        snap = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
        if snap is not None and (snap / "tokenizer.json").is_file():
            return snap / "tokenizer.json"
        raise FileNotFoundError(f"Could not find tokenizer.json at {self.paths.tokenizer_json}")

    def _try_init_ane(self, *, pipe_depth: int, compute_units: int) -> _ANERunnerBackend | None:
        try:
            if not (self.paths.coreml_dylib.is_file() and self.paths.io_dylib.is_file()):
                self._ane_init_error = (
                    f"Missing native dylibs ({self.paths.io_dylib}, {self.paths.coreml_dylib}); run `make native`."
                )
                return None
            stage_map = _discover_ane_stage_paths(self.paths.ane_models_dir)
            if not stage_map:
                self._ane_init_error = f"No .mlmodelc stages in {self.paths.ane_models_dir}; run `setup-ane`."
                return None
            if not self.paths.embed_tokens_safetensors.is_file():
                self._ane_init_error = (
                    f"Missing F16 mmap embeddings at {self.paths.embed_tokens_safetensors}; run `setup-weights`."
                )
                return None

            embed_mm, _, _ = _load_f16_memmap_from_safetensors(self.paths.embed_tokens_safetensors)
            proj_w = self._resolve_contextual_proj_f32()
            return _ANERunnerBackend(
                self.paths,
                self.config,
                contextual_proj_f32=proj_w,
                embed_f16_mm=embed_mm,
                pipe_depth=pipe_depth,
                compute_units=compute_units,
            )
        except Exception as exc:
            self._ane_init_error = f"ANE initialization failed: {exc}"
            return None

    def _resolve_contextual_proj_f32(self) -> np.ndarray:
        if self.gpu is not None and hasattr(self.gpu, "model"):
            w_mx = getattr(getattr(self.gpu.model, "contextual_projection", None), "weight", None)
            if w_mx is not None:
                return np.ascontiguousarray(np.array(w_mx, dtype=np.float32, copy=True))
        if self.paths.weights_npz.is_file():
            with np.load(self.paths.weights_npz) as npz:
                for key in ("contextual_projection.weight", "contextual_proj"):
                    if key in npz:
                        return np.asarray(npz[key], dtype=np.float32)
        head_sf = self.paths.model_dir / "contextual_head.safetensors"
        if head_sf.is_file():
            with head_sf.open("rb") as fp:
                hlen = struct.unpack("<Q", fp.read(8))[0]
                hdr = json.loads(fp.read(hlen).decode("utf-8"))
                entry = hdr["contextual_projection.weight"]
                b0, b1 = entry["data_offsets"]
                fp.seek(8 + hlen + b0)
                return np.frombuffer(fp.read(b1 - b0), dtype="<f4").reshape(tuple(entry["shape"])).copy()
        raise FileNotFoundError(f"Could not locate contextual_projection.weight in {self.paths.weights_npz}")

    def _resolve_mode(self, mode: str | None) -> str:
        eff = self.default_mode if mode is None else mode
        if eff not in _VALID_MODES:
            raise ValueError(f"Unsupported mode {eff!r}; choose from {_VALID_MODES}")
        if eff == "auto":
            return "hetero2" if self.ane_available else "gpu"
        if eff in ("hetero", "hetero2") and not self.ane_available:
            if self.gpu is not None:
                return "gpu"
            raise RuntimeError("No GPU or ANE backend available")
        if eff == "ane" and not self.ane_available:
            raise RuntimeError(self._ane_init_error or "ANE backend is not initialized")
        return eff

    def _format_empty(self, n_rows: int, target_dim: int, output_dtype: str) -> np.ndarray:
        np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
        return np.zeros((n_rows, target_dim), dtype=np_dt)

    def _finalize_ane_array(
        self,
        raw_vecs: np.ndarray,
        *,
        output_dtype: str,
    ) -> np.ndarray:
        if output_dtype == "int8":
            return np.clip(np.rint(raw_vecs), -128, 127).astype(np.int8)
        if output_dtype == "float16":
            return raw_vecs.astype(np.float16)
        return raw_vecs.astype(np.float32, copy=False)

    def _dispatch_token_items(
        self,
        items: Sequence[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]],
        *,
        mode: str | None,
        batch_size: int,
        dimensions: int | None,
        output_dtype: str,
        normalize: bool,
        contextual_docs: Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        eff_mode = self._resolve_mode(mode)
        if output_dtype not in ("float32", "float16", "int8"):
            raise ValueError(f"Unsupported output_dtype {output_dtype!r}")
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.max_dim, default_dim=self.dim, supports_dimensions=True)

        t0 = time.perf_counter()
        n_items = len(items)
        if n_items == 0:
            return [] if contextual_docs is not None else self._format_empty(0, target_dim, output_dtype)

        ane_caps = tuple(getattr(self.ane, "caps", ())) if self.ane is not None else ()
        max_ane_cap = ane_caps[-1] if ane_caps else 0

        if eff_mode == "gpu":
            self.last_stats = {"mode": "gpu", "ane_items": 0, "gpu_items": n_items}
            if contextual_docs is not None:
                res = self.gpu.encode_contextual_chunks(
                    contextual_docs,
                    batch_size=batch_size,
                    max_length=self.max_seq_len,
                    dimensions=target_dim,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )
            else:
                res = self.gpu.encode_tokens(
                    [it[1] for it in items],
                    batch_size=batch_size,
                    dimensions=target_dim,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )
            self.last_stats["elapsed_s"] = round(time.perf_counter() - t0, 4)
            return res

        # Partition short vs >max_ane_cap items.
        short_items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = []
        long_items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = []
        for it in items:
            if len(it[1]) <= max_ane_cap:
                short_items.append(it)
            else:
                if eff_mode == "ane":
                    raise ValueError(
                        f"Item {it[0]} length ({len(it[1])}) exceeds max ANE bucket ({max_ane_cap}); use mode='hetero2' or 'gpu'"
                    )
                long_items.append(it)

        # Prepare result containers.
        np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
        if contextual_docs is not None:
            ctx_out: list[np.ndarray] = [np.zeros((len(contextual_docs[i]), target_dim), dtype=np_dt) for i in range(n_items)]
        else:
            flat_out = np.zeros((n_items, target_dim), dtype=np_dt)

        apply_tanh_int8 = output_dtype == "int8"
        apply_l2 = normalize and output_dtype != "int8"

        def _store_ane_results(done_list: list[tuple[PackedBucket, np.ndarray]]) -> None:
            for bkt, raw_mat in done_list:
                fin = self._finalize_ane_array(raw_mat, output_dtype=output_dtype)
                live_pos = 0
                for tgt in bkt.targets:
                    if tgt.end > tgt.start:
                        vec = fin[live_pos]
                        live_pos += 1
                    else:
                        vec = np.zeros((target_dim,), dtype=np_dt)
                    if contextual_docs is not None:
                        ctx_out[tgt.item_index][tgt.chunk_index] = vec
                    else:
                        flat_out[tgt.item_index] = vec

        def _run_gpu_subset(sub_items: Sequence[tuple[int, Sequence[int], Any]]) -> None:
            if not sub_items:
                return
            idxs = [it[0] for it in sub_items]
            if contextual_docs is not None:
                sub_docs = [contextual_docs[i] for i in idxs]
                vecs_list = self.gpu.encode_contextual_chunks(
                    sub_docs,
                    batch_size=batch_size,
                    max_length=self.max_seq_len,
                    dimensions=target_dim,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )
                for orig_i, v_mat in zip(idxs, vecs_list, strict=True):
                    ctx_out[orig_i] = v_mat
            else:
                toks = [it[1] for it in sub_items]
                vecs = self.gpu.encode_tokens(
                    toks,
                    batch_size=batch_size,
                    dimensions=target_dim,
                    output_dtype=output_dtype,
                    normalize=normalize,
                )
                for pos, orig_i in enumerate(idxs):
                    flat_out[orig_i] = vecs[pos]

        if eff_mode == "ane" or self.gpu is None:
            buckets = pack_ane_buckets(short_items, supported_caps=ane_caps)
            done = self.ane.run_buckets(
                buckets,
                out_dim=target_dim,
                apply_tanh_int8=apply_tanh_int8,
                apply_l2_norm=apply_l2,
            )
            _store_ane_results(done)
            self.last_stats = {
                "mode": eff_mode,
                "ane_items": len(short_items),
                "ane_buckets": len(buckets),
                "gpu_items": 0,
                "elapsed_s": round(time.perf_counter() - t0, 4),
            }
            return ctx_out if contextual_docs is not None else flat_out

        # Concurrent hetero / hetero2 scheduler:
        # - ANE drains packed buckets from head of short_dq via async pipe_depth ring.
        # - GPU processes all >max_ane_cap items first, then steals batches from tail of short_dq
        #   whenever enough short items remain to amortize a Metal launch.
        short_dq = deque(sorted(short_items, key=lambda it: len(it[1])))
        mu = threading.Lock()
        errors: list[BaseException] = []
        counts = {"ane_items": 0, "ane_buckets": 0, "gpu_items": 0}
        steal_min_chunk = max(4, min(batch_size, 16))
        ane_chunk_take = max(8, self.pipe_depth * 2)

        def ane_worker() -> None:
            try:
                while True:
                    with mu:
                        if not short_dq:
                            break
                        taken = [short_dq.popleft() for _ in range(min(len(short_dq), ane_chunk_take))]
                    bkts = pack_ane_buckets(taken, supported_caps=ane_caps)
                    done = self.ane.run_buckets(
                        bkts,
                        out_dim=target_dim,
                        apply_tanh_int8=apply_tanh_int8,
                        apply_l2_norm=apply_l2,
                    )
                    _store_ane_results(done)
                    with mu:
                        counts["ane_items"] += len(taken)
                        counts["ane_buckets"] += len(bkts)
            except BaseException as exc:
                errors.append(exc)

        def gpu_worker() -> None:
            try:
                if long_items:
                    _run_gpu_subset(long_items)
                    with mu:
                        counts["gpu_items"] += len(long_items)
                if eff_mode in ("hetero", "hetero2"):
                    while True:
                        with mu:
                            # Leave at least 2 small items if almost done so ANE finishes quickly cleanly.
                            if len(short_dq) < steal_min_chunk:
                                break
                            take_n = min(batch_size, max(1, len(short_dq) // 2))
                            stolen = [short_dq.pop() for _ in range(take_n)]
                        _run_gpu_subset(stolen)
                        with mu:
                            counts["gpu_items"] += len(stolen)
            except BaseException as exc:
                errors.append(exc)

        t_ane = threading.Thread(target=ane_worker, name="fast-pplx-ane", daemon=True)
        t_gpu = threading.Thread(target=gpu_worker, name="fast-pplx-gpu", daemon=True)
        t_ane.start()
        t_gpu.start()
        t_ane.join()
        t_gpu.join()
        if errors:
            raise errors[0]

        self.last_stats = {
            "mode": eff_mode,
            **counts,
            "elapsed_s": round(time.perf_counter() - t0, 4),
        }
        return ctx_out if contextual_docs is not None else flat_out

    def encode_tokens(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        batch_size: int = 16,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray:
        items = [(idx, list(toks), None) for idx, toks in enumerate(token_lists)]
        res = self._dispatch_token_items(
            items,
            mode=mode,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
        )
        assert isinstance(res, np.ndarray)
        return res

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        is_query: bool = False,
        mode: str | None = None,
    ) -> np.ndarray:
        items_txt = [texts] if isinstance(texts, str) else list(texts)
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        if is_query:
            token_lists = [self.tokenizer.encode_query(t, max_length=eff_max) for t in items_txt]
        else:
            token_lists = [self.tokenizer.encode_document(t, max_length=eff_max) for t in items_txt]
        return self.encode_tokens(
            token_lists,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
        )

    def encode_queries(
        self,
        queries: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray:
        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=True,
            mode=mode,
        )

    def encode_documents(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray:
        return self.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            is_query=False,
            mode=mode,
        )

    def encode_contextual_chunks(
        self,
        documents: Sequence[Sequence[str]],
        *,
        batch_size: int = 8,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> list[np.ndarray]:
        eff_max = min(int(max_length), self.max_seq_len) if max_length else self.max_seq_len
        items: list[tuple[int, list[int], list[tuple[int, int]]]] = []
        for idx, chunks in enumerate(documents):
            tok_ids, spans = self.tokenizer.encode_contextual_document(
                chunks, return_spans=True, max_length=eff_max
            )
            items.append((idx, tok_ids, spans))
        res = self._dispatch_token_items(
            items,
            mode=mode,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            contextual_docs=documents,
        )
        assert isinstance(res, list)
        return res

    def encode_texts(
        self,
        texts: Sequence[str] | str,
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        prefix: str | None = None,
        prompt_name: str | None = None,
        max_length: int | None = None,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        batch_size: int = 16,
        mode: str | None = None,
    ) -> np.ndarray:
        items_txt = [texts] if isinstance(texts, str) else list(texts)
        eff_max = self.max_seq_len if max_length is None else min(int(max_length), self.max_seq_len)
        eff_prefix = self.tokenizer.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=bool(is_query or add_query_prefix),
            add_document_prefix=add_document_prefix,
        )
        tok_lists = self.tokenizer.encode_batch(items_txt, prefix=eff_prefix, max_length=eff_max)
        return self.encode_tokens(
            tok_lists,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
        )

    def encode_query(
        self,
        query: str | Sequence[int],
        *,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray:
        if isinstance(query, str):
            mat = self.encode_queries(
                [query],
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
            )
        else:
            mat = self.encode_tokens(
                [list(query)],
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
            )
        return np.asarray(mat[0])

    def encode_corpus(
        self,
        corpus: Sequence[str] | Sequence[Sequence[int]],
        *,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray:
        if not corpus:
            target_dim = resolve_embedding_dim(dimensions, max_dim=self.max_dim, default_dim=self.dim, supports_dimensions=True)
            return self._format_empty(0, target_dim, output_dtype)
        first = corpus[0]
        if isinstance(first, str):
            return self.encode_documents(
                corpus,  # type: ignore[arg-type]
                batch_size=batch_size,
                max_length=max_length,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
            )
        return self.encode_tokens(
            corpus,  # type: ignore[arg-type]
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
        )

    def embed(
        self,
        inputs: str | Sequence[str] | Sequence[Sequence[str]],
        *,
        is_query: bool = False,
        batch_size: int = 16,
        max_length: int = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(inputs, str):
            return self.encode(
                inputs,
                batch_size=batch_size,
                max_length=max_length,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                is_query=is_query,
                mode=mode,
            )
        seq = list(inputs)
        if not seq:
            target_dim = resolve_embedding_dim(dimensions, max_dim=self.max_dim, default_dim=self.dim, supports_dimensions=True)
            return self._format_empty(0, target_dim, output_dtype)
        if isinstance(seq[0], str):
            return self.encode(
                seq,  # type: ignore[arg-type]
                batch_size=batch_size,
                max_length=max_length,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                is_query=is_query,
                mode=mode,
            )
        return self.encode_contextual_chunks(
            seq,  # type: ignore[arg-type]
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
        )

    def close(self) -> None:
        if getattr(self, "ane", None) is not None:
            try:
                self.ane.close()
            finally:
                self.ane = None
        if getattr(self, "gpu", None) is not None:
            try:
                self.gpu.close()
            finally:
                self.gpu = None

    def __enter__(self) -> FastPplx:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "DEFAULT_MODEL_NAME",
    "EMBED_DIM",
    "FastPplx",
    "MAX_EMBED_DIM",
    "MRL_DIMENSIONS",
    "PackedBucket",
    "PackedSpanTarget",
    "pack_ane_buckets",
]
