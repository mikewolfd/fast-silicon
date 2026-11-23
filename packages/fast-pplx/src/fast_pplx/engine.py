"""Unified Apple Silicon heterogeneous (Metal MLX GPU + 16-core ANE) engine for pplx-embed-v2-context-0.6b."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, Literal, Sequence

import numpy as np
from fast_shared.embeddings import accepts_kwarg, resolve_embedding_dim, truncate_and_normalize
from fast_shared.retrieval_index import CompactRetrievalIndex, IndexMode, WindowAgg

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .gpu import FastPplxGPU, _find_local_hf_snapshot, _resolve_max_length, get_mlx_runtime_telemetry
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
_VALID_PACK_STRATEGIES: tuple[str, ...] = ("ffd", "sequential")


def resolve_hetero_schedule(
    *,
    is_apple10: bool,
    batch_size: int,
    pipe_depth: int,
) -> tuple[int, int, int, int, int]:
    """Return host-aware `(ane_chunk_take, steal_min_chunk, steal_cap, steal_num, steal_den)` for `hetero2`.

    Base M5 (`Apple10` NAX) runs Metal GPU ~1.8x faster than M4 Pro (`Apple9`), so M5 GPU steals
    earlier (down to smaller tail thresholds) in larger batches while ANE pulls lighter bites.
    """
    bs = max(1, int(batch_size))
    pd = max(1, int(pipe_depth))
    if is_apple10:
        ane_take = max(4, min(8, pd * 2 - 2 if pd >= 4 else pd * 2))
        steal_min = max(2, min(bs, 6))
        steal_cap = max(bs * 2, 16)
        return ane_take, steal_min, steal_cap, 2, 3
    ane_take = max(8, pd * 2)
    steal_min = max(4, min(bs, 16))
    return ane_take, steal_min, bs, 1, 2


class _PplxSlotBuffers(ctypes.Structure):
    _fields_ = [
        ("slot_idx", ctypes.c_int),
        ("in_h", ctypes.c_void_p),
        ("in_bias", ctypes.c_void_p),
        ("in_cos32", ctypes.c_void_p),
        ("in_sin32", ctypes.c_void_p),
        ("in_tok_valid", ctypes.c_void_p),
        ("in_gdn_carry", ctypes.c_void_p),
    ]


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


@dataclass
class _BucketStagingSlot:
    h_in_u16: np.ndarray
    bias_u16: np.ndarray
    c32_u16: np.ndarray
    s32_u16: np.ndarray
    tvalid_u16: np.ndarray
    carry_u16: np.ndarray
    h_out_u16: np.ndarray


def _append_packed_targets(
    targets: list[PackedSpanTarget],
    orig_idx: int,
    n_tok: int,
    start: int,
    spans: Sequence[tuple[int, int]] | None,
) -> None:
    if spans is None:
        targets.append(PackedSpanTarget(orig_idx, 0, start, start + n_tok))
        return
    for c_idx, (s0, s1) in enumerate(spans):
        if s1 > s0:
            targets.append(PackedSpanTarget(orig_idx, c_idx, start + int(s0), start + min(int(s1), n_tok)))
        else:
            targets.append(PackedSpanTarget(orig_idx, c_idx, 0, 0))


def _finalize_packed_bin(
    cur_toks: list[tuple[int, Sequence[int], int]],
    cur_targets: list[PackedSpanTarget],
    cur_pos: int,
    doc_slot: int,
    caps: tuple[int, ...],
) -> PackedBucket:
    max_cap = caps[-1]
    cap = next((c for c in caps if c >= max(1, cur_pos)), max_cap)
    tok_arr = np.zeros((cap,), dtype=np.int32)
    doc_arr = np.full((cap,), -1, dtype=np.int32)
    pos_arr = np.zeros((cap,), dtype=np.int32)
    for st, ids, d_slot in cur_toks:
        ln = len(ids)
        if ln > 0:
            tok_arr[st : st + ln] = np.asarray(ids, dtype=np.int32)
            doc_arr[st : st + ln] = d_slot
            pos_arr[st : st + ln] = np.arange(ln, dtype=np.int32)
    return PackedBucket(
        seq_len=cap,
        token_ids=tok_arr,
        doc_ids=doc_arr,
        pos_ids=pos_arr,
        targets=cur_targets,
        n_docs=doc_slot,
        used_tokens=cur_pos,
    )


def pack_ane_buckets(
    items: Sequence[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]],
    *,
    supported_caps: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    conv_gap: int = 3,
    strategy: Literal["ffd", "sequential"] = "ffd",
) -> list[PackedBucket]:
    """Pack short `<= max_cap` token sequences into fixed ANE buckets with strict doc isolation.

    Defaults to First-Fit Decreasing (`strategy='ffd'`) across open buckets to maximize slot
    density while preserving `conv_gap >= 3` zero-pad boundaries (`doc_ids[pad] = -1`), chunk
    alignment, exact `PackedSpanTarget` item/chunk indices, and empty-doc targets.
    """
    if strategy not in _VALID_PACK_STRATEGIES:
        raise ValueError(f"Unsupported pack strategy {strategy!r}; choose from {_VALID_PACK_STRATEGIES}")
    caps = tuple(sorted(int(c) for c in supported_caps))
    if not caps:
        raise ValueError("supported_caps must be non-empty")
    max_cap = caps[-1]
    if not items:
        return []

    def _next_aligned_start(cur_pos: int, doc_slot: int) -> int:
        if doc_slot == 0:
            return 0
        min_next = cur_pos + conv_gap
        return ((min_next + chunk_size - 1) // chunk_size) * chunk_size

    if strategy == "sequential":
        buckets: list[PackedBucket] = []
        cur_pos = 0
        doc_slot = 0
        cur_toks: list[tuple[int, Sequence[int], int]] = []
        cur_targets: list[PackedSpanTarget] = []

        def _flush_seq() -> None:
            nonlocal cur_pos, doc_slot, cur_toks, cur_targets
            if not cur_toks and not cur_targets:
                return
            buckets.append(_finalize_packed_bin(cur_toks, cur_targets, cur_pos, doc_slot, caps))
            cur_pos = 0
            doc_slot = 0
            cur_toks = []
            cur_targets = []

        for orig_idx, tok_ids, spans in items:
            n_tok = len(tok_ids)
            if n_tok > max_cap:
                raise ValueError(f"Item {orig_idx} length ({n_tok}) exceeds max ANE bucket ({max_cap})")
            if n_tok == 0:
                n_sp = 1 if spans is None else len(spans)
                for c_idx in range(n_sp):
                    cur_targets.append(PackedSpanTarget(orig_idx, c_idx, 0, 0))
                continue
            start = _next_aligned_start(cur_pos, doc_slot)
            if doc_slot > 0 and start + n_tok > max_cap:
                _flush_seq()
                start = 0
            cur_toks.append((start, tok_ids, doc_slot))
            _append_packed_targets(cur_targets, orig_idx, n_tok, start, spans)
            cur_pos = start + n_tok
            doc_slot += 1

        _flush_seq()
        return buckets

    empty_targets: list[PackedSpanTarget] = []
    non_empty: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = []
    for orig_idx, tok_ids, spans in items:
        n_tok = len(tok_ids)
        if n_tok > max_cap:
            raise ValueError(f"Item {orig_idx} length ({n_tok}) exceeds max ANE bucket ({max_cap})")
        if n_tok == 0:
            n_sp = 1 if spans is None else len(spans)
            for c_idx in range(n_sp):
                empty_targets.append(PackedSpanTarget(orig_idx, c_idx, 0, 0))
        else:
            non_empty.append((orig_idx, tok_ids, spans))

    non_empty.sort(key=lambda it: len(it[1]), reverse=True)
    open_bins: list[dict[str, Any]] = []
    for orig_idx, tok_ids, spans in non_empty:
        n_tok = len(tok_ids)
        placed_bin: dict[str, Any] | None = None
        placed_start = 0
        for b in open_bins:
            start = _next_aligned_start(b["cur_pos"], b["doc_slot"])
            if start + n_tok <= max_cap:
                placed_bin = b
                placed_start = start
                break
        if placed_bin is None:
            placed_bin = {"cur_pos": 0, "doc_slot": 0, "toks": [], "targets": []}
            open_bins.append(placed_bin)
            placed_start = 0
        d_slot = placed_bin["doc_slot"]
        placed_bin["toks"].append((placed_start, tok_ids, d_slot))
        _append_packed_targets(placed_bin["targets"], orig_idx, n_tok, placed_start, spans)
        placed_bin["cur_pos"] = placed_start + n_tok
        placed_bin["doc_slot"] = d_slot + 1

    if not open_bins:
        return [_finalize_packed_bin([], empty_targets, 0, 0, caps)] if empty_targets else []

    if empty_targets:
        open_bins[0]["targets"].extend(empty_targets)

    return [
        _finalize_packed_bin(b["toks"], b["targets"], b["cur_pos"], b["doc_slot"], caps)
        for b in open_bins
    ]


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

    if hasattr(lib, "pplx_ane_load_stages_fn"):
        lib.pplx_ane_load_stages_fn.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.pplx_ane_load_stages_fn.restype = ctypes.c_void_p

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

    if hasattr(lib, "pplx_ane_inspect_plan_fn"):
        lib.pplx_ane_inspect_plan_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.pplx_ane_inspect_plan_fn.restype = ctypes.c_int

    lib.pplx_ane_free.argtypes = [ctypes.c_void_p]
    lib.pplx_ane_free.restype = None

    if hasattr(lib, "pplx_ane_last_error"):
        lib.pplx_ane_last_error.argtypes = []
        lib.pplx_ane_last_error.restype = ctypes.c_char_p

    if all(
        hasattr(lib, fn)
        for fn in (
            "pplx_ane_acquire_slot",
            "pplx_ane_submit_slot",
            "pplx_ane_wait_slot",
            "pplx_ane_release_slot",
        )
    ):
        lib.pplx_ane_acquire_slot.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PplxSlotBuffers)]
        lib.pplx_ane_acquire_slot.restype = ctypes.c_int

        lib.pplx_ane_submit_slot.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.pplx_ane_submit_slot.restype = ctypes.c_void_p

        lib.pplx_ane_wait_slot.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.pplx_ane_wait_slot.restype = ctypes.c_double

        lib.pplx_ane_release_slot.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.pplx_ane_release_slot.restype = None

    return lib


def _coreml_last_error(lib: ctypes.CDLL) -> str:
    fn = getattr(lib, "pplx_ane_last_error", None)
    if fn is None:
        return ""
    try:
        raw = fn()
        return raw.decode("utf-8", errors="replace") if raw else ""
    except Exception:
        return ""


def read_ane_manifest(ane_dir: Path) -> dict[str, Any] | None:
    """Read and normalize `ane_models/manifest.json` (including legacy schema v1 manifests)."""
    manifest_path = ane_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    silu_mode = str(raw.get("silu_mode") or "silu")
    rmsnorm_mode = str(raw.get("rmsnorm_mode") or "prescale")
    profile = raw.get("ane_profile")
    if not profile:
        if (silu_mode, rmsnorm_mode) == ("silu", "prescale"):
            profile = "fast"
        elif (silu_mode, rmsnorm_mode) == ("tanh_gdn", "max_abs"):
            profile = "balanced"
        elif (silu_mode, rmsnorm_mode) == ("tanh_all", "max_abs"):
            profile = "parity"
        else:
            profile = "custom"
    out = dict(raw)
    out.setdefault("schema_version", 1)
    out["silu_mode"] = silu_mode
    out["rmsnorm_mode"] = rmsnorm_mode
    out["ane_profile"] = str(profile)
    out.setdefault("config_hash", None)
    return out


def _discover_ane_stage_specs(
    ane_dir: Path,
    *,
    expected_profile: str | None = None,
    expected_config_hash: str | None = None,
) -> dict[int, tuple[list[Path], str | None]]:
    """Discover per-seq-len `(stage_paths, optional_function_name)` chains from `manifest.json` or glob fallback.

    Supports both multifunction stage bundles (`function_name="pplx_s32"`, sharing 4 stage `.mlmodelc`
    bundles across all buckets) and legacy per-bucket `.mlmodelc` stage dirs (`function_name=None`).
    """
    if not ane_dir.is_dir():
        return {}
    manifest_path = ane_dir / "manifest.json"
    discovered: dict[int, tuple[list[Path], str | None]] = {}
    if manifest_path.is_file():
        manifest = read_ane_manifest(ane_dir)
        if manifest is None:
            return {}
        if expected_profile is not None and manifest.get("ane_profile") != expected_profile:
            return {}
        top_hash = manifest.get("config_hash")
        if expected_config_hash is not None and top_hash != expected_config_hash:
            return {}
        expected_stages = manifest.get("num_stages")
        for k_str, entry in manifest.get("buckets", {}).items():
            if not isinstance(entry, dict):
                continue
            bucket_hash = entry.get("config_hash")
            if top_hash is not None and bucket_hash is not None and bucket_hash != top_hash:
                continue
            cap = int(entry.get("seq_len", k_str))
            stage_names = entry.get("stages", [])
            if expected_stages is not None and len(stage_names) != int(expected_stages):
                continue
            stages = [ane_dir / name for name in stage_names]
            fn_raw = entry.get("function_name")
            fn_name = str(fn_raw) if fn_raw else None
            if stages and all(p.exists() for p in stages):
                discovered[cap] = (stages, fn_name)
        return dict(sorted(discovered.items()))

    if expected_profile is not None or expected_config_hash is not None:
        return {}

    # Fallback when no manifest.json exists: scan pplx_s{S}_st{idx}_<variant>.mlmodelc,
    # grouping by (cap, variant_suffix) so distinct modes/configs never mix or collide.
    grouped: dict[int, dict[str, list[tuple[int, Path]]]] = {}
    for p in sorted(ane_dir.glob("pplx_s*_st*.mlmodelc")):
        parts = p.stem.split("_")
        try:
            cap = int(parts[1][1:])
            st_idx = int(parts[2][2:])
        except (IndexError, ValueError):
            continue
        suffix = "_".join(parts[3:])
        grouped.setdefault(cap, {}).setdefault(suffix, []).append((st_idx, p))
    for cap, by_suffix in sorted(grouped.items()):
        valid_groups: list[list[Path]] = []
        for pairs in by_suffix.values():
            pairs.sort(key=lambda t: t[0])
            if [idx for idx, _ in pairs] == list(range(len(pairs))):
                valid_groups.append([p for _, p in pairs])
        # Only accept unambiguous single-variant stage chains in manifest-less fallback.
        if len(valid_groups) == 1:
            discovered[cap] = (valid_groups[0], None)
    return discovered


def _discover_ane_stage_paths(
    ane_dir: Path,
    *,
    expected_profile: str | None = None,
    expected_config_hash: str | None = None,
) -> dict[int, list[Path]]:
    """Discover per-seq-len staged .mlmodelc bundle chains from manifest.json or glob fallback."""
    specs = _discover_ane_stage_specs(
        ane_dir,
        expected_profile=expected_profile,
        expected_config_hash=expected_config_hash,
    )
    return {cap: stages for cap, (stages, _) in specs.items()}


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
    """Thread-safe double/quad-buffered C++ + CoreML ANE backend using staged .mlmodelc bundles.

    Eliminates per-bucket NumPy allocations and MLMultiArray copies by packing directly into
    dense ring-slot MLMultiArray pointers when supported, with preallocated per-bucket
    staging pools as fallback.
    """

    def __init__(
        self,
        paths: RepoPaths,
        config: PplxConfig,
        *,
        contextual_proj_f32: np.ndarray,
        embed_f16_mm: np.ndarray,
        pipe_depth: int = DEFAULT_PIPE_DEPTH,
        compute_units: int = 0,
        expected_profile: str | None = None,
    ) -> None:
        self.io_lib = _load_io_lib(paths.io_dylib)
        self.coreml_lib = _load_coreml_lib(paths.coreml_dylib)
        self.config = config
        self.pipe_depth = max(1, min(int(pipe_depth), 8))
        self.embed_mm = embed_f16_mm
        self.vocab_size, self.hidden_dim = int(embed_f16_mm.shape[0]), int(embed_f16_mm.shape[1])
        self.contextual_proj_f32 = np.ascontiguousarray(contextual_proj_f32, dtype=np.float32)
        self._run_lock = threading.Lock()
        self._supports_zero_copy_slots = all(
            hasattr(self.coreml_lib, fn)
            for fn in (
                "pplx_ane_acquire_slot",
                "pplx_ane_submit_slot",
                "pplx_ane_wait_slot",
                "pplx_ane_release_slot",
            )
        )

        self.manifest: dict[str, Any] | None = read_ane_manifest(paths.ane_models_dir)
        stage_specs = _discover_ane_stage_specs(paths.ane_models_dir, expected_profile=expected_profile)
        if not stage_specs:
            if expected_profile is not None and self.manifest is not None:
                staged = self.manifest.get("ane_profile")
                raise FileNotFoundError(
                    f"Staged ANE bundle profile {staged!r} does not match requested ane_profile={expected_profile!r}; "
                    f"rebuild with `fast-pplx setup-ane --ane-profile {expected_profile}`."
                )
            raise FileNotFoundError(f"No compiled ANE .mlmodelc stages found in {paths.ane_models_dir}")

        self.runners: dict[int, int] = {}
        has_load_fn = hasattr(self.coreml_lib, "pplx_ane_load_stages_fn")
        try:
            for cap, (stage_paths, fn_name) in stage_specs.items():
                arr = (ctypes.c_char_p * len(stage_paths))(
                    *(str(p).encode("utf-8") for p in stage_paths)
                )
                if fn_name and has_load_fn:
                    handle = self.coreml_lib.pplx_ane_load_stages_fn(
                        arr,
                        fn_name.encode("utf-8"),
                        len(stage_paths),
                        cap,
                        config.hidden_size,
                        DEFAULT_CHUNK_SIZE,
                        DEFAULT_ROPE_HALF,
                        self.pipe_depth,
                        compute_units,
                    )
                else:
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
                    detail = _coreml_last_error(self.coreml_lib)
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(f"Failed loading ANE pipeline for S={cap} ({stage_paths}){suffix}")
                self.runners[cap] = int(handle)
        except Exception:
            self.close()
            raise

        self.caps: tuple[int, ...] = tuple(sorted(self.runners))
        self.max_cap: int = self.caps[-1]

        # Preallocate reusable fallback staging rings (one per cap, pipe_depth slots) to eliminate per-bucket np.empty.
        self._staging_rings: dict[int, list[_BucketStagingSlot]] = {}
        for cap in self.caps:
            n_chunks = max(1, cap // DEFAULT_CHUNK_SIZE)
            self._staging_rings[cap] = [
                _BucketStagingSlot(
                    h_in_u16=np.empty((self.hidden_dim, cap), dtype=np.uint16),
                    bias_u16=np.empty((cap, cap), dtype=np.uint16),
                    c32_u16=np.empty((cap, DEFAULT_ROPE_HALF), dtype=np.uint16),
                    s32_u16=np.empty((cap, DEFAULT_ROPE_HALF), dtype=np.uint16),
                    tvalid_u16=np.empty((cap,), dtype=np.uint16),
                    carry_u16=np.empty((n_chunks,), dtype=np.uint16),
                    h_out_u16=np.empty((self.hidden_dim, cap), dtype=np.uint16),
                )
                for _ in range(self.pipe_depth)
            ]
        self._ring_cursors: dict[int, int] = {cap: 0 for cap in self.caps}

        self.rope_cos_u16 = np.empty((self.max_cap, DEFAULT_ROPE_HALF), dtype=np.uint16)
        self.rope_sin_u16 = np.empty((self.max_cap, DEFAULT_ROPE_HALF), dtype=np.uint16)
        self.io_lib.pplx_build_rope_table_f16(
            self.rope_cos_u16.ctypes.data,
            self.rope_sin_u16.ctypes.data,
            self.max_cap,
            DEFAULT_ROPE_HALF,
            float(config.rope_theta),
        )

    def _next_staging_slot(self, seq_len: int) -> _BucketStagingSlot:
        ring = self._staging_rings[seq_len]
        idx = self._ring_cursors[seq_len]
        self._ring_cursors[seq_len] = (idx + 1) % len(ring)
        return ring[idx]

    def _pool_from_h_ptr(
        self,
        bkt: PackedBucket,
        h_out_ptr: int,
        *,
        out_dim: int,
        apply_tanh_int8: bool,
        apply_l2_norm: bool,
    ) -> np.ndarray:
        live_targets = [t for t in bkt.targets if t.end > t.start]
        if not live_targets:
            return np.zeros((0, out_dim), dtype=np.float32)
        starts = np.fromiter((t.start for t in live_targets), dtype=np.int32, count=len(live_targets))
        ends = np.fromiter((t.end for t in live_targets), dtype=np.int32, count=len(live_targets))
        out_f32 = np.zeros((len(live_targets), out_dim), dtype=np.float32)
        # Final stage already executed final RMSNorm -> pass final_norm_f32=None (0)
        self.io_lib.pplx_pool_and_project_f32(
            h_out_ptr,
            self.hidden_dim,
            bkt.seq_len,
            len(live_targets),
            starts.ctypes.data,
            ends.ctypes.data,
            None,
            int(self.contextual_proj_f32.ctypes.data),
            int(self.contextual_proj_f32.shape[0]),
            out_dim,
            1 if apply_tanh_int8 else 0,
            1 if apply_l2_norm else 0,
            out_f32.ctypes.data,
        )
        return out_f32

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

        with self._run_lock:
            embed_ptr = int(self.embed_mm.ctypes.data)
            cos_ptr = int(self.rope_cos_u16.ctypes.data)
            sin_ptr = int(self.rope_sin_u16.ctypes.data)

            # Inflight queue entries: (is_zero_copy, runner_handle, bkt, slot_idx_or_staging_h_out, job_ptr)
            inflight: deque[tuple[bool, int, PackedBucket, Any, int]] = deque()
            completed: list[tuple[PackedBucket, np.ndarray]] = []

            def _drain_one() -> None:
                is_zc, runner_handle, bkt, aux, job = inflight.popleft()
                if is_zc:
                    out_ptr = ctypes.c_void_p(0)
                    slot_out = ctypes.c_int(int(aux))
                    dt = self.coreml_lib.pplx_ane_wait_slot(
                        ctypes.c_void_p(job),
                        ctypes.byref(out_ptr),
                        ctypes.byref(slot_out),
                    )
                    try:
                        if dt < 0.0 or not out_ptr.value:
                            detail = _coreml_last_error(self.coreml_lib)
                            suffix = f": {detail}" if detail else ""
                            raise RuntimeError(f"ANE prediction failed for bucket S={bkt.seq_len}{suffix}")
                        out_f32 = self._pool_from_h_ptr(
                            bkt,
                            int(out_ptr.value),
                            out_dim=out_dim,
                            apply_tanh_int8=apply_tanh_int8,
                            apply_l2_norm=apply_l2_norm,
                        )
                    finally:
                        self.coreml_lib.pplx_ane_release_slot(ctypes.c_void_p(runner_handle), int(slot_out.value))
                    completed.append((bkt, out_f32))
                else:
                    h_out_u16: np.ndarray = aux
                    dt = self.coreml_lib.pplx_ane_wait(ctypes.c_void_p(job))
                    if dt < 0.0:
                        detail = _coreml_last_error(self.coreml_lib)
                        suffix = f": {detail}" if detail else ""
                        raise RuntimeError(f"ANE prediction failed for bucket S={bkt.seq_len}{suffix}")
                    out_f32 = self._pool_from_h_ptr(
                        bkt,
                        int(h_out_u16.ctypes.data),
                        out_dim=out_dim,
                        apply_tanh_int8=apply_tanh_int8,
                        apply_l2_norm=apply_l2_norm,
                    )
                    completed.append((bkt, out_f32))

            try:
                for bkt in buckets:
                    if len(inflight) >= self.pipe_depth:
                        _drain_one()

                    seq_len = bkt.seq_len
                    if seq_len not in self.runners:
                        raise KeyError(f"No ANE runner loaded for bucket S={seq_len} (have {self.caps})")
                    runner_handle = self.runners[seq_len]

                    submitted_zc = False
                    if self._supports_zero_copy_slots:
                        slot_bufs = _PplxSlotBuffers()
                        slot_idx = int(
                            self.coreml_lib.pplx_ane_acquire_slot(
                                ctypes.c_void_p(runner_handle), ctypes.byref(slot_bufs)
                            )
                        )
                        if slot_idx >= 0:
                            try:
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
                                    slot_bufs.in_h,
                                    slot_bufs.in_bias,
                                    slot_bufs.in_cos32,
                                    slot_bufs.in_sin32,
                                    slot_bufs.in_tok_valid,
                                    slot_bufs.in_gdn_carry,
                                )
                                job = self.coreml_lib.pplx_ane_submit_slot(
                                    ctypes.c_void_p(runner_handle),
                                    slot_idx,
                                )
                                if not job:
                                    detail = _coreml_last_error(self.coreml_lib)
                                    suffix = f": {detail}" if detail else ""
                                    raise RuntimeError(f"pplx_ane_submit_slot returned NULL for S={seq_len}{suffix}")
                            except BaseException:
                                self.coreml_lib.pplx_ane_release_slot(ctypes.c_void_p(runner_handle), slot_idx)
                                raise
                            inflight.append((True, runner_handle, bkt, slot_idx, int(job)))
                            submitted_zc = True

                    if not submitted_zc:
                        slot = self._next_staging_slot(seq_len)
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
                            slot.h_in_u16.ctypes.data,
                            slot.bias_u16.ctypes.data,
                            slot.c32_u16.ctypes.data,
                            slot.s32_u16.ctypes.data,
                            slot.tvalid_u16.ctypes.data,
                            slot.carry_u16.ctypes.data,
                        )
                        job = self.coreml_lib.pplx_ane_submit(
                            ctypes.c_void_p(runner_handle),
                            slot.h_in_u16.ctypes.data,
                            slot.bias_u16.ctypes.data,
                            slot.c32_u16.ctypes.data,
                            slot.s32_u16.ctypes.data,
                            slot.tvalid_u16.ctypes.data,
                            slot.carry_u16.ctypes.data,
                            slot.h_out_u16.ctypes.data,
                        )
                        if not job:
                            detail = _coreml_last_error(self.coreml_lib)
                            suffix = f": {detail}" if detail else ""
                            raise RuntimeError(f"pplx_ane_submit returned NULL for S={seq_len}{suffix}")
                        inflight.append((False, runner_handle, bkt, slot.h_out_u16, int(job)))

                while inflight:
                    _drain_one()
            except Exception:
                while inflight:
                    is_zc, runner_handle, _, aux, job = inflight.popleft()
                    try:
                        if is_zc:
                            out_ptr = ctypes.c_void_p(0)
                            slot_out = ctypes.c_int(int(aux))
                            self.coreml_lib.pplx_ane_wait_slot(
                                ctypes.c_void_p(job),
                                ctypes.byref(out_ptr),
                                ctypes.byref(slot_out),
                            )
                            self.coreml_lib.pplx_ane_release_slot(ctypes.c_void_p(runner_handle), int(slot_out.value))
                        else:
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
    """Unified Apple Silicon `pplx-embed-v2-context-0.6b` engine (Metal MLX GPU + 16-core ANE).

    Routing modes:
      - ``gpu``: deterministic high-precision Metal MLX GPU (recommended for exact reproducibility / verify).
      - ``ane``: static short-bucket (``<= 512``) pure-ANE CoreML pipeline (ANE quantization drift tolerance).
      - ``hetero``: deterministic static split (``<= max_ane_cap`` -> ANE, ``> max_ane_cap`` -> GPU; no stealing).
      - ``hetero2``: dynamic work-stealing ANE + GPU (maximum mixed throughput; non-deterministic stealing).
      - ``auto``: selects ``hetero2`` when ANE stages are available (or ``gpu`` when ``deterministic=True``).
    """

    model_name: str = DEFAULT_MODEL_NAME
    dim: int = EMBED_DIM
    max_dim: int = MAX_EMBED_DIM
    supports_dimensions: bool = True
    is_mrl_trained: bool = False
    mrl_dimensions: tuple[int, ...] = MRL_DIMENSIONS
    cpu_sme_available: bool = False

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        mode: str = "auto",
        model_dir: str | Path | None = None,
        dtype: str = "float16",
        state_dtype: str | None = None,
        quantize_weights: Literal["none", "w8"] = "none",
        use_mapped_embedding: bool = True,
        cache_limit_mb: int = 512,
        local_files_only: bool = True,
        enable_ane: bool = True,
        ane_profile: str | None = None,
        deterministic: bool = False,
        dimensions: int | None = None,
        pipe_depth: int = DEFAULT_PIPE_DEPTH,
        compute_units: int = 0,
        _gpu_backend: Any | None = None,
        _ane_backend: Any | None = None,
        _tokenizer: Any | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; choose from {_VALID_MODES}")
        if ane_profile is not None and ane_profile not in ("fast", "balanced", "parity", "custom"):
            raise ValueError(
                f"Unsupported ane_profile {ane_profile!r}; choose from ('fast', 'balanced', 'parity', 'custom')"
            )
        eff_qw = str(quantize_weights).strip().lower()
        if eff_qw not in ("none", "w8"):
            raise ValueError(f"Unsupported quantize_weights {quantize_weights!r}; expected 'none' or 'w8'")

        self.paths: RepoPaths = resolve_paths(home)
        self._requested_mode: str = mode
        self.dtype_name = dtype
        self.quantize_weights: Literal["none", "w8"] = "w8" if eff_qw == "w8" else "none"
        self.pipe_depth = int(pipe_depth)
        self.deterministic = bool(deterministic)
        self._requested_ane_profile: str | None = ane_profile
        self._local_stats = threading.local()
        self._stats_lock = threading.Lock()
        self._latest_stats: dict[str, Any] = {}
        self._gpu_lock = threading.Lock()
        self.runtime_telemetry: dict[str, Any] = {
            **get_mlx_runtime_telemetry(),
            "quantize_weights": self.quantize_weights,
        }

        # 1. GPU / MLX backend (always loaded unless explicitly testing ANE-only with mock)
        self.gpu: Any = _gpu_backend
        if self.gpu is None and mode != "ane":
            self.gpu = FastPplxGPU(
                home=self.paths.home,
                model_dir=model_dir,
                dtype=dtype,
                state_dtype=state_dtype,
                quantize_weights=self.quantize_weights,
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
        want_ane = enable_ane and (not self.deterministic or mode in ("ane", "hetero")) and mode in (
            "auto",
            "ane",
            "hetero",
            "hetero2",
        )
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
        self.ane_manifest: dict[str, Any] | None = (
            getattr(self.ane, "manifest", None) if self.ane_available else read_ane_manifest(self.paths.ane_models_dir)
        )
        self.staged_ane_profile: str | None = (
            str(self.ane_manifest["ane_profile"])
            if isinstance(self.ane_manifest, dict) and self.ane_manifest.get("ane_profile")
            else None
        )
        self.ane_profile: str | None = self.staged_ane_profile if self.ane_available else None
        self.ane_query_max_len: int = (
            int(getattr(self.ane, "max_cap", DEFAULT_ANE_MAX_SEQ_LEN))
            if self.ane_available
            else DEFAULT_ANE_MAX_SEQ_LEN
        )
        if mode == "auto":
            if self.deterministic:
                self.default_mode = "gpu"
            else:
                self.default_mode = "hetero2" if self.ane_available else "gpu"
        elif mode == "ane":
            self.default_mode = "ane"
        elif mode == "hetero2":
            if self.deterministic:
                self.default_mode = "gpu" if self.gpu is not None else "hetero"
            else:
                self.default_mode = "hetero2" if self.ane_available else "gpu"
        elif mode == "hetero":
            self.default_mode = "hetero" if self.ane_available else "gpu"
        else:
            self.default_mode = "gpu"

    @property
    def last_stats(self) -> dict[str, Any]:
        """Thread-safe snapshot of the calling thread's most recent dispatch stats."""
        local_val = getattr(self._local_stats, "value", None)
        if local_val is not None:
            return dict(local_val)
        with self._stats_lock:
            return dict(self._latest_stats)

    @last_stats.setter
    def last_stats(self, value: dict[str, Any]) -> None:
        snap = dict(value)
        self._local_stats.value = snap
        with self._stats_lock:
            self._latest_stats = snap

    def _set_call_stats(self, stats: dict[str, Any]) -> None:
        self.last_stats = stats

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
            manifest = read_ane_manifest(self.paths.ane_models_dir)
            if (
                self._requested_ane_profile is not None
                and manifest is not None
                and manifest.get("ane_profile") != self._requested_ane_profile
            ):
                self._ane_init_error = (
                    f"Staged ANE bundle profile {manifest.get('ane_profile')!r} does not match requested "
                    f"ane_profile={self._requested_ane_profile!r}; rebuild with "
                    f"`fast-pplx setup-ane --ane-profile {self._requested_ane_profile}`."
                )
                return None
            stage_map = _discover_ane_stage_paths(
                self.paths.ane_models_dir,
                expected_profile=self._requested_ane_profile,
            )
            if not stage_map:
                if self._requested_ane_profile is not None:
                    self._ane_init_error = (
                        f"No .mlmodelc stages matching ane_profile={self._requested_ane_profile!r} in "
                        f"{self.paths.ane_models_dir}; run `fast-pplx setup-ane --ane-profile {self._requested_ane_profile}`."
                    )
                else:
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
                expected_profile=self._requested_ane_profile,
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
            if self.deterministic:
                return "gpu" if self.gpu is not None else "ane"
            return "hetero2" if self.ane_available else "gpu"
        if eff == "hetero2" and self.deterministic and self.gpu is not None:
            return "gpu"
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
        contextual: bool = False,
        apply_tanh: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        eff_mode = self._resolve_mode(mode)
        if output_dtype not in ("float32", "float16", "int8"):
            raise ValueError(f"Unsupported output_dtype {output_dtype!r}")
        target_dim = resolve_embedding_dim(dimensions, max_dim=self.max_dim, default_dim=self.dim, supports_dimensions=True)

        t0 = time.perf_counter()
        n_items = len(items)
        if n_items == 0:
            self._set_call_stats(
                {"mode": eff_mode, "ane_items": 0, "gpu_items": 0, "deterministic": eff_mode in ("gpu", "ane", "hetero"), "elapsed_s": 0.0}
            )
            return [] if contextual else self._format_empty(0, target_dim, output_dtype)

        ane_caps = tuple(getattr(self.ane, "caps", ())) if self.ane is not None else ()
        max_ane_cap = ane_caps[-1] if ane_caps else 0

        def _encode_gpu(
            sub_items: Sequence[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]],
        ) -> np.ndarray | list[np.ndarray]:
            with self._gpu_lock:
                if contextual:
                    fn = self.gpu.encode_contextual_tokens
                    kw: dict[str, Any] = {
                        "batch_size": batch_size,
                        "dimensions": target_dim,
                        "output_dtype": output_dtype,
                        "normalize": normalize,
                    }
                    if accepts_kwarg(fn, "apply_tanh"):
                        kw["apply_tanh"] = apply_tanh
                    return fn([(ids, spans or ()) for _, ids, spans in sub_items], **kw)
                if any(it[2] is not None for it in sub_items) and hasattr(self.gpu, "encode_contextual_tokens"):
                    fn = self.gpu.encode_contextual_tokens
                    kw = {
                        "batch_size": batch_size,
                        "dimensions": target_dim,
                        "output_dtype": output_dtype,
                        "normalize": normalize,
                    }
                    if accepts_kwarg(fn, "apply_tanh"):
                        kw["apply_tanh"] = apply_tanh
                    mats = fn([(ids, spans or [(0, len(ids))]) for _, ids, spans in sub_items], **kw)
                    if not mats:
                        return self._format_empty(0, target_dim, output_dtype)
                    return np.vstack([np.asarray(m)[0:1] for m in mats])
                fn = self.gpu.encode_tokens
                kw = {
                    "batch_size": batch_size,
                    "dimensions": target_dim,
                    "output_dtype": output_dtype,
                    "normalize": normalize,
                }
                if accepts_kwarg(fn, "apply_tanh"):
                    kw["apply_tanh"] = apply_tanh
                return fn([it[1] for it in sub_items], **kw)

        if eff_mode == "gpu":
            res = _encode_gpu(items)
            self._set_call_stats(
                {
                    "mode": "gpu",
                    "ane_items": 0,
                    "gpu_items": n_items,
                    "deterministic": True,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
            )
            return res

        # Partition short vs >max_ane_cap items.
        short_items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = []
        long_items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = []
        for it in items:
            if len(it[1]) <= max_ane_cap:
                short_items.append(it)
            else:
                if eff_mode == "ane" or self.gpu is None:
                    raise ValueError(
                        f"Item {it[0]} length ({len(it[1])}) exceeds max ANE bucket ({max_ane_cap}); use mode='hetero2' or 'gpu'"
                    )
                long_items.append(it)

        # Prepare result containers.
        np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8}[output_dtype]
        if contextual:
            ctx_out = [np.zeros((len(spans or ()), target_dim), dtype=np_dt) for _, _, spans in items]
        else:
            flat_out = np.zeros((n_items, target_dim), dtype=np_dt)

        apply_tanh_int8 = (output_dtype == "int8") or bool(apply_tanh)
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
                    if contextual:
                        ctx_out[tgt.item_index][tgt.chunk_index] = vec
                    else:
                        flat_out[tgt.item_index] = vec

        def _run_gpu_subset(sub_items: Sequence[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]]) -> None:
            for (orig_i, _, _), vec in zip(sub_items, _encode_gpu(sub_items), strict=True):
                if contextual:
                    ctx_out[orig_i] = vec
                else:
                    flat_out[orig_i] = vec

        if eff_mode == "ane" or self.gpu is None:
            buckets = pack_ane_buckets(short_items, supported_caps=ane_caps)
            done = self.ane.run_buckets(
                buckets,
                out_dim=target_dim,
                apply_tanh_int8=apply_tanh_int8,
                apply_l2_norm=apply_l2,
            )
            _store_ane_results(done)
            self._set_call_stats(
                {
                    "mode": eff_mode,
                    "ane_items": len(short_items),
                    "ane_buckets": len(buckets),
                    "gpu_items": 0,
                    "deterministic": True,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
            )
            return ctx_out if contextual else flat_out

        # Zero-thread inline fast path for single-item short inputs:
        # - Base M5+ (Apple10 NAX) `hetero2`: Metal GPU is ~1.5x faster (~6.1ms vs ~9.0ms ANE) at exact FP16 parity.
        # - M4 Pro (Apple9) or deterministic `hetero`: single ANE slot beats Metal GPU without thread-spawn overhead.
        if n_items == 1 and not contextual and len(short_items) == 1:
            is_apple10_q = bool(self.runtime_telemetry.get("is_apple10", False))
            if eff_mode == "hetero2" and self.gpu is not None and is_apple10_q:
                res = _encode_gpu(items)
                self._set_call_stats(
                    {
                        "mode": eff_mode,
                        "ane_items": 0,
                        "ane_buckets": 0,
                        "gpu_items": 1,
                        "gpu_stolen_items": 1,
                        "deterministic": False,
                        "elapsed_s": round(time.perf_counter() - t0, 4),
                    }
                )
                return res
            buckets = pack_ane_buckets(short_items, supported_caps=ane_caps)
            done = self.ane.run_buckets(
                buckets,
                out_dim=target_dim,
                apply_tanh_int8=apply_tanh_int8,
                apply_l2_norm=apply_l2,
            )
            _store_ane_results(done)
            self._set_call_stats(
                {
                    "mode": eff_mode,
                    "ane_items": 1,
                    "ane_buckets": len(buckets),
                    "gpu_items": 0,
                    "gpu_stolen_items": 0,
                    "deterministic": eff_mode == "hetero",
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                }
            )
            return flat_out

        # Concurrent hetero (deterministic static split) vs hetero2 (dynamic work-stealing):
        # - `hetero`: ANE owns all short <= max_ane_cap items deterministically; GPU owns > max_ane_cap items.
        # - `hetero2`: ANE drains from head of short_dq while GPU finishes long items and steals from tail of short_dq.
        allow_stealing = eff_mode == "hetero2"
        short_dq = deque(sorted(short_items, key=lambda it: len(it[1])))
        mu = threading.Lock()
        errors: list[BaseException] = []
        counts = {"ane_items": 0, "ane_buckets": 0, "gpu_items": 0, "gpu_stolen_items": 0}
        is_apple10 = bool(self.runtime_telemetry.get("is_apple10", False))
        ane_chunk_take, steal_min_chunk, steal_cap, steal_num, steal_den = resolve_hetero_schedule(
            is_apple10=is_apple10,
            batch_size=batch_size,
            pipe_depth=self.pipe_depth,
        )

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
                if allow_stealing:
                    while True:
                        with mu:
                            # Leave small tail items when almost empty so ANE finishes cleanly without racing.
                            if len(short_dq) < steal_min_chunk:
                                break
                            take_n = min(len(short_dq), steal_cap, max(1, (len(short_dq) * steal_num) // steal_den))
                            stolen = [short_dq.pop() for _ in range(take_n)]
                        _run_gpu_subset(stolen)
                        with mu:
                            counts["gpu_items"] += len(stolen)
                            counts["gpu_stolen_items"] += len(stolen)
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

        self._set_call_stats(
            {
                "mode": eff_mode,
                **counts,
                "deterministic": not allow_stealing,
                "elapsed_s": round(time.perf_counter() - t0, 4),
            }
        )
        return ctx_out if contextual else flat_out

    def encode_tokens(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        batch_size: int = 16,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
        exclude_document_prefix: bool = False,
        apply_tanh: bool = True,
    ) -> np.ndarray:
        use_span = bool(exclude_document_prefix and hasattr(self.tokenizer, "document_prefix_span"))
        items = [
            (idx, list(toks), [self.tokenizer.document_prefix_span(toks)] if use_span else None)
            for idx, toks in enumerate(token_lists)
        ]
        res = self._dispatch_token_items(
            items,
            mode=mode,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            apply_tanh=apply_tanh,
        )
        assert isinstance(res, np.ndarray)
        return res

    def encode(
        self,
        texts: Sequence[str] | str,
        *,
        batch_size: int = 16,
        max_length: int | None = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        is_query: bool = False,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> np.ndarray:
        items_txt = [texts] if isinstance(texts, str) else list(texts)
        eff_max = _resolve_max_length(max_length, self.max_seq_len)
        if is_query:
            items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = [
                (idx, self.tokenizer.encode_query(t, max_length=eff_max), None) for idx, t in enumerate(items_txt)
            ]
        else:
            items = [
                (idx, *self.tokenizer.encode_contextual_document([t], return_spans=True, max_length=eff_max))
                for idx, t in enumerate(items_txt)
            ]
        res = self._dispatch_token_items(
            items,
            mode=mode,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            contextual=False,
            apply_tanh=apply_tanh,
        )
        assert isinstance(res, np.ndarray)
        return res

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
        apply_tanh: bool = True,
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
            apply_tanh=apply_tanh,
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
        apply_tanh: bool = True,
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
            apply_tanh=apply_tanh,
        )

    def encode_contextual_chunks(
        self,
        documents: Sequence[Sequence[str]],
        *,
        batch_size: int = 8,
        max_length: int | None = 32768,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> list[np.ndarray]:
        eff_max = _resolve_max_length(max_length, self.max_seq_len)
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
            contextual=True,
            apply_tanh=apply_tanh,
        )
        assert isinstance(res, list)
        return res

    def encode_contextual_tokens(
        self,
        encoded_docs: Sequence[tuple[Sequence[int], Sequence[tuple[int, int]]]],
        *,
        batch_size: int = 8,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> list[np.ndarray]:
        """Encode pre-tokenized `(token_ids, chunk_spans)` contextual documents without re-tokenization."""
        items = [
            (idx, list(tok_ids), [(int(s0), int(s1)) for s0, s1 in spans])
            for idx, (tok_ids, spans) in enumerate(encoded_docs)
        ]
        res = self._dispatch_token_items(
            items,
            mode=mode,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            contextual=True,
            apply_tanh=apply_tanh,
        )
        assert isinstance(res, list)
        return res

    def _ensure_gpu(self) -> Any:
        if self.gpu is None:
            if self._requested_mode == "ane":
                raise RuntimeError(
                    "encode_images requires the Metal GPU backend (not available when mode='ane')"
                )
            raise RuntimeError("Metal GPU backend is not initialized")
        return self.gpu

    def encode_images(
        self,
        images: Sequence[Any],
        *,
        chunks: int = 8,
        batch_size: int = 16,
        min_pixels: int = 16384,
        max_pixels: int = 1_800_000,
        crop_margins: bool = False,
        rotations_cw: int | Sequence[int] | None = None,
        grid_bucket_step: int | None = None,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        apply_tanh: bool = True,
    ) -> list[np.ndarray]:
        """Encode page/document images into contextual chunk embeddings via shared Qwen3.5 ViT + MRoPE."""
        from .gpu import normalize_rotations_cw

        gpu = self._ensure_gpu()
        img_list = list(images)
        norm_rotations = normalize_rotations_cw(rotations_cw, len(img_list))
        with self._gpu_lock:
            return gpu.encode_images(
                img_list,
                chunks=chunks,
                batch_size=batch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                crop_margins=crop_margins,
                rotations_cw=norm_rotations,
                grid_bucket_step=grid_bucket_step,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                apply_tanh=apply_tanh,
            )

    def encode_texts(
        self,
        texts: Sequence[str] | str,
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        add_document_prefix: bool | None = None,
        prefix: str | None = None,
        prompt_name: str | None = None,
        max_length: int | None = None,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        batch_size: int = 16,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> np.ndarray:
        items_txt = [texts] if isinstance(texts, str) else list(texts)
        eff_max = _resolve_max_length(max_length, self.max_seq_len)
        eff_add_doc = (
            (not is_query and not add_query_prefix and prefix is None and prompt_name is None)
            if add_document_prefix is None
            else bool(add_document_prefix)
        )
        eff_prefix = self.tokenizer.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=bool(is_query or add_query_prefix),
            add_document_prefix=eff_add_doc,
        )
        if (
            eff_prefix
            and eff_prefix.startswith(self.tokenizer.document_prefix)
            and hasattr(self.tokenizer, "encode_contextual_document")
        ):
            user_pfx = eff_prefix[len(self.tokenizer.document_prefix) :]
            items: list[tuple[int, Sequence[int], Sequence[tuple[int, int]] | None]] = [
                (idx, *self.tokenizer.encode_contextual_document([user_pfx + t], return_spans=True, max_length=eff_max))
                for idx, t in enumerate(items_txt)
            ]
            res = self._dispatch_token_items(
                items,
                mode=mode,
                batch_size=batch_size,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                contextual=False,
                apply_tanh=apply_tanh,
            )
            assert isinstance(res, np.ndarray)
            return res
        tok_lists = self.tokenizer.encode_batch(items_txt, prefix=eff_prefix, max_length=eff_max)
        return self.encode_tokens(
            tok_lists,
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
            apply_tanh=apply_tanh,
        )

    def encode_query(
        self,
        query: str | Sequence[int],
        *,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> np.ndarray:
        if isinstance(query, str):
            mat = self.encode_queries(
                [query],
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
                apply_tanh=apply_tanh,
            )
        else:
            mat = self.encode_tokens(
                [list(query)],
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
                apply_tanh=apply_tanh,
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
        exclude_document_prefix: bool = False,
        apply_tanh: bool = True,
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
                apply_tanh=apply_tanh,
            )
        return self.encode_tokens(
            corpus,  # type: ignore[arg-type]
            batch_size=batch_size,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
            exclude_document_prefix=exclude_document_prefix,
            apply_tanh=apply_tanh,
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
        apply_tanh: bool = True,
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
                apply_tanh=apply_tanh,
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
                apply_tanh=apply_tanh,
            )
        return self.encode_contextual_chunks(
            seq,  # type: ignore[arg-type]
            batch_size=batch_size,
            max_length=max_length,
            dimensions=dimensions,
            output_dtype=output_dtype,
            normalize=normalize,
            mode=mode,
            apply_tanh=apply_tanh,
        )

    def encode_documents_multivector(
        self,
        texts: Sequence[str] | str,
        *,
        window_tokens: int = 1024,
        context_tokens: int = 2048,
        overlap: int = 0,
        repeat_header_tokens: int | None = 128,
        batch_size: int = 16,
        dimensions: int | None = None,
        output_dtype: str = "float32",
        normalize: bool = True,
        mode: str | None = None,
        apply_tanh: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode `4K-32K` documents into `(window_vecs [W, D], doc_offsets [N_docs + 1])` for `MaxSim`.

        Packs `<= 512`-token sub-chunks into `context_tokens=2048` contextual windows with
        `repeat_header_tokens=128` (`2.31x` faster GPU encode and `2.48x` lower Metal peak RAM
        than `32K` whole-doc attention, while lifting `32K` `LongEmbed` passkey `nDCG@10` from
        `0.047 -> 1.000` per-split / `0.674` pooled and `2wikimqa` from `0.8945 -> 0.985-0.987`).
        """
        if window_tokens <= 0:
            raise ValueError(f"window_tokens must be positive, got {window_tokens}")
        if context_tokens < 256:
            raise ValueError(f"context_tokens must be >= 256, got {context_tokens}")

        docs = [texts] if isinstance(texts, str) else list(texts)
        target_dim = resolve_embedding_dim(
            dimensions, max_dim=self.max_dim, default_dim=self.dim, supports_dimensions=True
        )
        if not docs:
            return self._format_empty(0, target_dim, output_dtype), np.zeros((1,), dtype=np.int32)

        base_chunk = min(512, int(window_tokens))
        group_k = max(1, (int(window_tokens) + base_chunk - 1) // base_chunk)

        def _encode_doc_windows(txt: str) -> list[tuple[list[int], list[tuple[int, int]]]]:
            if not txt or not txt.strip():
                return []
            planned = self.tokenizer.plan_contextual_windows(
                txt,
                chunk_tokens=base_chunk,
                context_tokens=int(context_tokens),
                overlap=int(overlap),
                repeat_header_tokens=repeat_header_tokens,
            )
            doc_wins: list[tuple[list[int], list[tuple[int, int]]]] = []
            for w in planned:
                ids, spans = self.tokenizer.encode_contextual_document(
                    w.chunks, return_spans=True, max_length=int(context_tokens)
                )
                raw_emit = [
                    (int(a), int(b))
                    for a, b in spans[w.emit_start : w.emit_start + len(w.spans)]
                    if int(b) > int(a)
                ]
                if not raw_emit:
                    continue
                if group_k <= 1:
                    grouped_spans = raw_emit
                else:
                    grouped_spans = [
                        (raw_emit[i][0], raw_emit[min(i + group_k - 1, len(raw_emit) - 1)][1])
                        for i in range(0, len(raw_emit), group_k)
                    ]
                doc_wins.append((ids, grouped_spans))
            return doc_wins

        encoded_wins: list[tuple[list[int], list[tuple[int, int]]]] = []
        win_owners: list[int] = []
        if len(docs) >= 8:
            max_workers = min(4, max(2, os.cpu_count() or 2), len(docs))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                per_doc_wins = list(pool.map(_encode_doc_windows, docs))
        else:
            per_doc_wins = [_encode_doc_windows(txt) for txt in docs]
        for d_idx, doc_wins in enumerate(per_doc_wins):
            for win_item in doc_wins:
                encoded_wins.append(win_item)
                win_owners.append(d_idx)

        per_doc: list[list[np.ndarray]] = [[] for _ in range(len(docs))]
        if encoded_wins:
            mats = self.encode_contextual_tokens(
                encoded_wins,
                batch_size=batch_size,
                dimensions=dimensions,
                output_dtype=output_dtype,
                normalize=normalize,
                mode=mode,
                apply_tanh=apply_tanh,
            )
            for owner, mat in zip(win_owners, mats):
                per_doc[owner].append(np.asarray(mat))

        offsets = [0]
        chunks_out: list[np.ndarray] = []
        for d_idx, blks in enumerate(per_doc):
            if blks:
                cat = np.vstack(blks)
                chunks_out.append(cat)
                offsets.append(offsets[-1] + int(cat.shape[0]))
            else:
                # Guarantee >=1 vector even for empty/whitespace-only docs so doc_offsets stays strictly increasing.
                z = self._format_empty(1, target_dim, output_dtype)
                z.fill(0)
                chunks_out.append(z)
                offsets.append(offsets[-1] + 1)

        all_vecs = np.vstack(chunks_out) if chunks_out else self._format_empty(0, target_dim, output_dtype)
        return all_vecs, np.asarray(offsets, dtype=np.int32)

    def build_index(
        self,
        corpus: Sequence[str] | Sequence[Sequence[int]] | np.ndarray,
        *,
        multivector: bool = False,
        window_tokens: int = 1024,
        context_tokens: int = 2048,
        repeat_header_tokens: int | None = 128,
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        mode: IndexMode = "auto",
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
        batch_size: int = 16,
        encode_mode: str | None = None,
    ) -> CompactRetrievalIndex:
        """Build a `CompactRetrievalIndex` (`1024d sym_int8` default, `binary_int8`, or `MaxSim` multivector)."""
        eff_offsets: np.ndarray | list[int] | None = (
            list(doc_offsets) if doc_offsets is not None and not isinstance(doc_offsets, np.ndarray) else doc_offsets
        )
        if isinstance(corpus, np.ndarray):
            vecs = corpus
        elif not corpus:
            vecs = np.empty((0, self.max_dim), dtype=np.float32)
        elif multivector:
            if not isinstance(corpus[0], str):
                raise TypeError("multivector=True requires a sequence of raw document strings")
            vecs, eff_offsets = self.encode_documents_multivector(
                corpus,  # type: ignore[arg-type]
                window_tokens=window_tokens,
                context_tokens=context_tokens,
                repeat_header_tokens=repeat_header_tokens,
                batch_size=batch_size,
                dimensions=self.max_dim,
                output_dtype="float32",
                normalize=True,
                mode=encode_mode,
            )
        else:
            vecs = self.encode_corpus(
                corpus,
                batch_size=batch_size,
                dimensions=self.max_dim,
                output_dtype="float32",
                normalize=True,
                mode=encode_mode,
            )
        return CompactRetrievalIndex(
            vecs,
            doc_offsets=eff_offsets,
            mode=mode,
            is_mrl_trained=self.is_mrl_trained,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
        )

    def build_image_index(
        self,
        images: Sequence[Any],
        *,
        chunks: int = 8,
        batch_size: int = 16,
        min_pixels: int = 16384,
        max_pixels: int = 1_800_000,
        crop_margins: bool = True,
        rotations_cw: int | Sequence[int] | None = None,
        grid_bucket_step: int | None = None,
        dimensions: int | None = None,
        mode: IndexMode = "auto",
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
    ) -> CompactRetrievalIndex:
        """Encode document/page images into contextual bands and build a CompactRetrievalIndex."""
        img_list = list(images)
        if dimensions is None:
            enc_dim = self.max_dim
        else:
            if isinstance(dimensions, bool) or not isinstance(dimensions, (int, np.integer)):
                raise ValueError(f"dimensions must be an integer, got {dimensions!r}")
            enc_dim = int(dimensions)
            if enc_dim <= 0 or enc_dim > self.max_dim:
                raise ValueError(
                    f"dimensions must be in [1, {self.max_dim}], got {enc_dim}"
                )

        per_page = self.encode_images(
            img_list,
            chunks=chunks,
            batch_size=batch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            crop_margins=crop_margins,
            rotations_cw=rotations_cw,
            grid_bucket_step=grid_bucket_step,
            dimensions=enc_dim,
            output_dtype="float32",
            normalize=True,
        )
        if not per_page:
            return CompactRetrievalIndex(
                np.empty((0, enc_dim), dtype=np.float32),
                doc_offsets=np.zeros((1,), dtype=np.int32),
                mode=mode,
                is_mrl_trained=self.is_mrl_trained,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                agg=agg,
                hybrid_alpha=hybrid_alpha,
            )

        vec_dim = enc_dim
        for mat in per_page:
            arr_m = np.asarray(mat)
            if arr_m.ndim == 2 and arr_m.shape[1] > 0:
                vec_dim = int(arr_m.shape[1])
                break

        offsets = [0]
        chunks_out: list[np.ndarray] = []
        for mat in per_page:
            arr = np.asarray(mat, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                chunks_out.append(arr)
                offsets.append(offsets[-1] + int(arr.shape[0]))
            else:
                chunks_out.append(np.zeros((1, vec_dim), dtype=np.float32))
                offsets.append(offsets[-1] + 1)

        all_vecs = np.vstack(chunks_out)
        return CompactRetrievalIndex(
            all_vecs,
            doc_offsets=np.asarray(offsets, dtype=np.int32),
            mode=mode,
            is_mrl_trained=self.is_mrl_trained,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
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
