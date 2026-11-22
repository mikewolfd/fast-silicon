"""Streaming sharded Parquet & optional Lance embedding pipeline optimized for Apple Silicon."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import ExitStack
from dataclasses import dataclass
import glob
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .bench import clear_mlx_arena, wired_limit
from .embeddings import accepts_kwarg, resolve_embedding_dim, set_mlx_cache_limit, truncate_and_normalize
from .hw_check import probe_host_hardware
from .runtime_lessons import (
    DEFAULT_MLX_CACHE_LIMIT_MB,
    WIRED_MEMORY_CEILING_FRACTION,
    LiveMemoryGovernor,
    nocache_open,
)
from .tokenizer import plan_header_anchored_windows

DEFAULT_TEXT_COLUMNS = ("body", "page_text", "text", "content")
_SUPPORTED_DTYPES = ("float32", "float16", "int8")
_MANIFEST_FILENAME = ".embed_manifest.json"


def _import_lance() -> Any | None:
    try:
        import lance  # type: ignore[import-not-found]

        return lance
    except ImportError:
        return None


LANCE_AVAILABLE: bool = importlib.util.find_spec("lance") is not None


def recommended_backfill_mrl_dim(engine: Any | None, *, default_mrl_dim: int = 512) -> int | None:
    """Return `512` only when `engine.is_mrl_trained` is True; `None` (full native dim) otherwise.

    Non-MRL encoders (`FastPplx` `1024d`, `FastBgeSmall` `384d`) lose `-1.10 pp` to `-3.77 pp`
    when truncated to `512d`, whereas full-dim symmetric `int8` is CI95-lossless (`4.0x` smaller).
    """
    if engine is None:
        return int(default_mrl_dim)
    if not bool(getattr(engine, "is_mrl_trained", False)):
        return None
    raw_dim = getattr(engine, "max_dim", None) or getattr(engine, "dim", None)
    if raw_dim is not None and int(raw_dim) <= int(default_mrl_dim):
        return None
    return int(default_mrl_dim)


@dataclass(frozen=True)
class CorpusBackfillConfig:
    """10M-document scale defaults (`int8` + `fp16` scale, model-aware MRL via `.for_engine()`, macro-window sort)."""

    dtype: str = "int8"
    mrl_dim: int | None = 512
    sort_macro_window: bool = True
    macro_window_tokens: int | None = None
    use_wired_limit: bool = True
    compression: str = "zstd"
    batch_rows: int = 4096
    encode_batch_rows: int = 4096
    shard_rows: int = 65536
    sink_format: str | None = None
    snap_sentences: bool = False
    header_anchored: bool = False
    anchor_max_chars: int = 2048
    pin_non_hetero_cutoffs: bool | None = None
    non_hetero_min_batch: int = 12
    non_hetero_min_tokens: int = 1024

    @classmethod
    def for_engine(cls, engine: Any, **overrides: Any) -> CorpusBackfillConfig:
        """Build a model-aware `CorpusBackfillConfig` (`512d int8` if MRL-trained else full-dim `int8`)."""
        kw: dict[str, Any] = {"mrl_dim": recommended_backfill_mrl_dim(engine)}
        kw.update(overrides)
        return cls(**kw)


@dataclass
class ParquetEmbedSummary:
    input_files: int
    rows_read: int
    rows_emitted: int
    total_tokens: int
    elapsed_seconds: float
    output_path: Path

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def rows_per_sec(self) -> float:
        return self.rows_emitted / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def resolve_macro_window_tokens(
    macro_window_tokens: int | None = None,
    *,
    host_mem_gb: float | None = None,
) -> int:
    """Auto-size token budget per global length-sorted macro-window (`1M` on `<=24 GB`, `2M` on `>=36 GB`)."""
    if macro_window_tokens is not None:
        val = int(macro_window_tokens)
        if val <= 0:
            raise ValueError(f"macro_window_tokens must be positive, got {macro_window_tokens!r}")
        return val
    mem = host_mem_gb
    if mem is None:
        try:
            mem = float(probe_host_hardware().get("mem_gb") or 24.0)
        except Exception:
            mem = 24.0
    return 2_000_000 if float(mem) >= 36.0 else 1_000_000


def resolve_input_files(input_path: str | Path) -> list[Path]:
    raw = str(input_path)
    p = Path(raw).expanduser()
    if any(ch in raw for ch in ("*", "?", "[")):
        cands = (Path(m).resolve() for m in glob.glob(raw, recursive=True) if m.endswith(".parquet"))
    elif p.is_dir():
        top = [f.resolve() for f in p.glob("*.parquet") if not f.name.startswith(".")]
        cands = top if top else [f.resolve() for f in p.rglob("*.parquet") if not f.name.startswith(".")]
    elif p.is_file():
        cands = [p.resolve()]
    else:
        raise FileNotFoundError(f"Input parquet path does not exist: {input_path}")

    files = sorted(f for f in cands if f.is_file() and f.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"No non-empty .parquet files found under {input_path}")
    return files


def resolve_text_column(schema: pa.Schema, text_col: str | None) -> str:
    names = set(schema.names)
    if text_col is not None:
        if text_col not in names:
            raise KeyError(f"Requested text column {text_col!r} not in schema columns: {schema.names}")
        return text_col
    for cand in DEFAULT_TEXT_COLUMNS:
        if cand in names:
            return cand
    raise KeyError(f"Could not auto-detect text column among {DEFAULT_TEXT_COLUMNS}; schema has {schema.names}")


def slice_ocr_pages_utf8(text: str | None, page_offsets: Sequence[int] | None, ocr_pages: Sequence[int] | None) -> str:
    """Extract only 1-based `ocr_pages` byte spans using UTF-8 `page_offsets` (body_text table convention)."""
    if not text or not page_offsets or not ocr_pages:
        return ""
    raw = text.encode("utf-8")
    total_len = len(raw)
    n_pages = len(page_offsets)
    parts: list[str] = []
    for p in ocr_pages:
        idx = int(p) - 1
        if 0 <= idx < n_pages:
            start = max(0, min(total_len, int(page_offsets[idx])))
            end = total_len if (idx + 1 >= n_pages) else max(start, min(total_len, int(page_offsets[idx + 1])))
            seg = raw[start:end].decode("utf-8", errors="replace").strip()
            if seg:
                parts.append(seg)
    return "\n\n".join(parts)


def _iter_source_batches(
    files: list[Path],
    columns: list[str],
    batch_rows: int,
    *,
    filter_ocr: bool,
    limit: int | None,
    use_nocache: bool = False,
) -> Iterable[pa.RecordBatch]:
    emitted = 0
    for fpath in files:
        with ExitStack() as fstack:
            if use_nocache:
                fh = fstack.enter_context(nocache_open(fpath, "rb", buffering=0))
                pf = pq.ParquetFile(fh)
            else:
                pf = pq.ParquetFile(fpath)
            names = set(pf.schema_arrow.names)
            read_cols = [c for c in columns if c in names]
            for rb in pf.iter_batches(batch_size=batch_rows, columns=read_cols):
                if filter_ocr:
                    if "is_ocr" in names:
                        mask = pc.equal(rb.column("is_ocr"), True)
                        rb = rb.filter(mask)
                    elif "ocr_pages" in names:
                        mask = pc.greater(pc.list_value_length(rb.column("ocr_pages")), 0)
                        rb = rb.filter(mask)
                if rb.num_rows == 0:
                    continue
                if limit is not None and emitted + rb.num_rows > limit:
                    rb = rb.slice(0, max(0, limit - emitted))
                if rb.num_rows > 0:
                    emitted += rb.num_rows
                    yield rb
                if limit is not None and emitted >= limit:
                    return


def _quantize_symmetric_int8(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize L2-normalized float32 embeddings to symmetric `int8` + per-row `float16` `embedding_scale`."""
    arr = np.asarray(vecs)
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
    scales_f16 = raw_scale.astype(np.float16)
    eff_scale = np.where(scales_f16.astype(np.float32) > 1e-12, scales_f16.astype(np.float32), np.float32(1.0 / 127.0))
    q_i8 = np.clip(np.rint(normed / eff_scale[:, None]), -127.0, 127.0).astype(np.int8)
    return np.ascontiguousarray(q_i8, dtype=np.int8), scales_f16


def _make_vector_array(vecs: np.ndarray, dim: int, dtype: str) -> tuple[pa.Array, pa.Array | None]:
    if dtype == "int8":
        q_i8, scales_f16 = _quantize_symmetric_int8(vecs)
        flat_i8 = q_i8.reshape(-1)
        vec_arr = pa.FixedSizeListArray.from_arrays(pa.array(flat_i8, type=pa.int8()), dim)
        scale_arr = pa.array(scales_f16, type=pa.float16())
        return vec_arr, scale_arr
    if dtype == "float16":
        flat = vecs.astype(np.float16).reshape(-1)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float16()), dim), None
    flat = np.ascontiguousarray(vecs, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dim), None


def _append_vector_columns(meta_tbl: pa.Table, vecs: np.ndarray, *, dim: int, dtype: str) -> pa.Table:
    vec_arr, scale_arr = _make_vector_array(vecs, dim=dim, dtype=dtype)
    out_tbl = meta_tbl.append_column("vector", vec_arr)
    if scale_arr is not None:
        out_tbl = out_tbl.append_column("embedding_scale", scale_arr)
    return out_tbl


def _resolve_parquet_compression(
    schema: pa.Schema,
    compression: str | Mapping[str, str] | None,
) -> str | dict[str, str]:
    """Never waste Zstd CPU compressing dense float/int8 vector columns; keep Zstd on text/metadata."""
    if compression is None:
        return "NONE"
    if isinstance(compression, Mapping):
        norm_map = {str(k): str(v).upper() for k, v in compression.items()}
        default_codec = norm_map.get("*", "ZSTD")
        out: dict[str, str] = {}
        for name in schema.names:
            if name in norm_map:
                out[name] = norm_map[name]
            elif name in ("vector", "embedding") and ("embedding" in norm_map or "vector" in norm_map):
                out[name] = norm_map.get("embedding", norm_map.get("vector", "NONE"))
            else:
                out[name] = default_codec
        return out
    codec = str(compression).strip()
    if codec.lower() in ("none", "uncompressed"):
        return "NONE"
    return {
        name: ("NONE" if name in ("vector", "embedding") else codec)
        for name in schema.names
    }


_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?…][\"'\)\]]*|\n)(?=\s|$)")
_WHITESPACE_BOUNDARY_RE = re.compile(r"\s+")
_DEFAULT_NON_HETERO_MIN_BATCH = 12
_DEFAULT_NON_HETERO_MIN_TOKENS = 1024


def resolve_embedding_device_mode(
    engine: Any,
    tok_lists: Sequence[Sequence[int]],
    *,
    mode: str | None = None,
    pin_non_hetero_cutoffs: bool | None = None,
    non_hetero_min_batch: int | None = None,
    non_hetero_min_tokens: int | None = None,
) -> str | None:
    """Resolve `mode='auto'` and small/ANE-ineligible sub-batches to the fastest pinned single device."""
    norm_mode = str(mode).strip().lower() if mode is not None else None
    if norm_mode in ("gpu", "ane", "cpu"):
        return norm_mode

    ane_attr = getattr(engine, "ane_available", None)
    if norm_mode is None and pin_non_hetero_cutoffs is False:
        return None
    if norm_mode is None and pin_non_hetero_cutoffs is None and ane_attr is None:
        return None
    if norm_mode in ("hetero", "hetero2") and pin_non_hetero_cutoffs is False:
        return "hetero"
    if norm_mode in ("hetero", "hetero2") and pin_non_hetero_cutoffs is None:
        return norm_mode
    if not tok_lists:
        if norm_mode == "auto":
            return "ane" if bool(ane_attr) else "gpu"
        return norm_mode

    ane_avail = bool(ane_attr)
    if not ane_avail:
        return "gpu" if (norm_mode == "auto" or ane_attr is not None) else norm_mode

    ane_max_len = int(
        getattr(engine, "ane_query_max_len", None)
        or getattr(engine, "ane_max_seq_len", None)
        or getattr(engine, "max_seq_len", None)
        or 512
    )
    lengths = [len(t) for t in tok_lists]
    n_docs = len(lengths)
    tot_toks = sum(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    if min_len > ane_max_len:
        return "gpu"

    min_batch = (
        int(non_hetero_min_batch)
        if non_hetero_min_batch is not None
        else int(getattr(engine, "non_hetero_min_batch", _DEFAULT_NON_HETERO_MIN_BATCH))
    )
    min_tokens = (
        int(non_hetero_min_tokens)
        if non_hetero_min_tokens is not None
        else int(getattr(engine, "non_hetero_min_tokens", _DEFAULT_NON_HETERO_MIN_TOKENS))
    )
    if n_docs < min_batch or tot_toks < min_tokens:
        return "ane" if max_len <= ane_max_len else "gpu"
    return "hetero"


def _snap_end_char(
    text: str,
    c0: int,
    raw_c1: int,
    *,
    min_fraction: float = 0.55,
    prefer_paragraph: bool = True,
) -> int:
    n_txt = len(text)
    hi = min(n_txt, max(c0, raw_c1))
    if hi >= n_txt or hi <= c0:
        return hi
    span = hi - c0
    min_c1 = c0 + max(1, int(span * max(0.25, min(0.95, float(min_fraction)))))
    window = text[c0:hi]
    rel_min = max(0, min_c1 - c0)

    if prefer_paragraph:
        para = window.rfind("\n\n", rel_min)
        if para != -1:
            return c0 + para + 2

    last_sent_end: int | None = None
    for m in _SENTENCE_BOUNDARY_RE.finditer(window, rel_min):
        last_sent_end = m.end()
    if last_sent_end is not None:
        return c0 + last_sent_end

    last_ws_start: int | None = None
    for m in _WHITESPACE_BOUNDARY_RE.finditer(window, rel_min):
        if m.start() > rel_min or last_sent_end is None:
            last_ws_start = m.start()
    if last_ws_start is not None and (c0 + last_ws_start) > c0:
        return c0 + last_ws_start
    return hi


def _chunk_from_token_offsets(
    text: str,
    body_ids: list[int],
    offsets: list[tuple[int, int]],
    *,
    tokenizer: Any,
    lead_ids: list[int],
    tail_ids: list[int],
    eff_max_body: int,
    overlap: int,
    max_chunk_tokens: int,
    snap_sentences: bool,
    snap_min_fraction: float,
) -> list[tuple[list[int], int, int]]:
    finalize = tokenizer._finalize_ids
    n = len(body_ids)
    if not n:
        return [(finalize([*lead_ids, *tail_ids], max_chunk_tokens), 0, len(text))]

    doc_chunks: list[tuple[list[int], int, int]] = []
    pos = 0
    while pos < n:
        end = min(n, pos + eff_max_body)
        c0 = 0 if pos == 0 else offsets[pos][0]
        while end > pos and end < n and offsets[end - 1][1] > offsets[end][0]:
            end -= 1
        if end <= pos:
            raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")

        raw_c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
        if snap_sentences and end < n and raw_c1 > c0:
            snap_c1 = _snap_end_char(text, c0, raw_c1, min_fraction=snap_min_fraction)
            if snap_c1 < raw_c1:
                tok_ends = [offsets[k][1] for k in range(pos, end)]
                k_fit = bisect_right(tok_ends, snap_c1)
                cand_end = pos + k_fit
                while cand_end > pos and cand_end < n and offsets[cand_end - 1][1] > offsets[cand_end][0]:
                    cand_end -= 1
                if cand_end > pos:
                    end = cand_end

        c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
        win_ids = finalize([*lead_ids, *body_ids[pos:end], *tail_ids], max_chunk_tokens)
        doc_chunks.append((win_ids, c0, c1))
        if end >= n:
            break

        if overlap <= 0:
            pos = end
            continue

        raw_next = max(pos + 1, end - min(overlap, eff_max_body - 1))
        next_pos = raw_next
        while next_pos > 0 and next_pos < n and offsets[next_pos - 1][1] > offsets[next_pos][0]:
            next_pos -= 1
        if snap_sentences and next_pos > pos and next_pos < end:
            # Prefer snapping the next window start onto the start of the nearest sentence/word boundary.
            start_ch = offsets[next_pos][0]
            scan_hi = offsets[min(end - 1, next_pos + max(4, overlap // 3))][1]
            sub_win = text[start_ch:scan_hi]
            m_sent = _SENTENCE_BOUNDARY_RE.search(sub_win)
            target_ch = None
            if m_sent is not None and m_sent.end() < len(sub_win):
                target_ch = start_ch + m_sent.end()
            else:
                m_ws = _WHITESPACE_BOUNDARY_RE.search(sub_win)
                if m_ws is not None and m_ws.end() < len(sub_win):
                    target_ch = start_ch + m_ws.end()
            if target_ch is not None:
                cand_starts = [offsets[k][0] for k in range(next_pos, end)]
                k_off = bisect_right(cand_starts, target_ch - 1)
                cand_pos = next_pos + k_off
                if pos < cand_pos < end:
                    while cand_pos < end and offsets[cand_pos - 1][1] > offsets[cand_pos][0]:
                        cand_pos += 1
                    if pos < cand_pos < end:
                        next_pos = cand_pos
        if next_pos <= pos:
            next_pos = raw_next
            while next_pos < end and offsets[next_pos - 1][1] > offsets[next_pos][0]:
                next_pos += 1
        pos = next_pos
    return doc_chunks


def _compute_header_anchor(
    unit_text: str,
    *,
    tokenizer: Any,
    anchor_max_chars: int,
    max_anchor_tokens: int,
) -> tuple[str, int]:
    raw_hdr = unit_text[: max(0, int(anchor_max_chars))]
    if not raw_hdr:
        return "", 0
    if len(raw_hdr) < len(unit_text):
        snapped_len = _snap_end_char(raw_hdr, 0, len(raw_hdr), min_fraction=0.35, prefer_paragraph=True)
        raw_hdr = raw_hdr[:snapped_len].rstrip() or raw_hdr.rstrip()
    raw_tok = getattr(tokenizer, "_tok", None)
    if raw_tok is not None and hasattr(raw_tok, "encode"):
        enc = raw_tok.encode(raw_hdr, add_special_tokens=False)
        if len(enc.ids) > max_anchor_tokens > 0 and enc.offsets:
            cutoff = enc.offsets[max_anchor_tokens - 1][1]
            clipped = raw_hdr[:cutoff]
            snapped = _snap_end_char(clipped, 0, len(clipped), min_fraction=0.45, prefer_paragraph=True)
            cand = clipped[:snapped].rstrip() or clipped.rstrip()
            if cand:
                raw_hdr = cand
                enc = raw_tok.encode(raw_hdr, add_special_tokens=False)
        return raw_hdr, len(enc.ids)
    toks = tokenizer.encode(raw_hdr)
    if len(toks) > max_anchor_tokens > 0:
        ratio = max_anchor_tokens / max(1, len(toks))
        cand_len = max(1, int(len(raw_hdr) * ratio))
        raw_hdr = raw_hdr[:_snap_end_char(raw_hdr, 0, cand_len, min_fraction=0.45)].rstrip() or raw_hdr[:cand_len]
    return raw_hdr, min(len(toks), max_anchor_tokens)


def _chunk_units_with_options(
    unit_texts: Sequence[str],
    *,
    tokenizer: Any,
    eff_prefix: str,
    max_body_tokens: int | None,
    overlap: int,
    max_chunk_tokens: int,
    snap_sentences: bool = False,
    snap_min_fraction: float = 0.55,
    header_anchored: bool = False,
    anchor_max_chars: int = 2048,
    anchor_max_tokens: int | None = None,
) -> tuple[list[list[tuple[list[int], int, int]]], list[list[bool]]]:
    """Chunk texts with optional sentence-boundary snapping and `plan_header_anchored_windows`."""
    if not unit_texts:
        return [], []

    if not snap_sentences and not header_anchored:
        if hasattr(tokenizer, "chunk_texts_batch"):
            wins = tokenizer.chunk_texts_batch(
                unit_texts,
                max_body_tokens=max_body_tokens,
                overlap=overlap,
                prefix=eff_prefix,
                max_chunk_tokens=max_chunk_tokens,
            )
        else:
            wins = [
                tokenizer.chunk_text(
                    u_txt,
                    max_body_tokens=max_body_tokens,
                    overlap=overlap,
                    prefix=eff_prefix,
                    max_chunk_tokens=max_chunk_tokens,
                )
                for u_txt in unit_texts
            ]
        return wins, [[False] * len(w) for w in wins]

    has_fast_tok = (
        getattr(tokenizer, "_tok", None) is not None
        and callable(getattr(tokenizer, "_resolve_special_and_prefix_ids", None))
        and callable(getattr(tokenizer, "_finalize_ids", None))
    )
    if not has_fast_tok:
        if hasattr(tokenizer, "chunk_texts_batch"):
            base_wins = tokenizer.chunk_texts_batch(
                unit_texts,
                max_body_tokens=max_body_tokens,
                overlap=overlap,
                prefix=eff_prefix,
                max_chunk_tokens=max_chunk_tokens,
            )
        else:
            base_wins = [
                tokenizer.chunk_text(
                    u_txt,
                    max_body_tokens=max_body_tokens,
                    overlap=overlap,
                    prefix=eff_prefix,
                    max_chunk_tokens=max_chunk_tokens,
                )
                for u_txt in unit_texts
            ]
        if not header_anchored:
            return base_wins, [[False] * len(w) for w in base_wins]
        # Generic fallback with plan_header_anchored_windows + re-encoding via encode_batch/encode
        all_windows: list[list[tuple[list[int], int, int]]] = []
        all_flags: list[list[bool]] = []
        for u_txt, w_list in zip(unit_texts, base_wins):
            if len(w_list) <= 1:
                all_windows.append(w_list)
                all_flags.append([False] * len(w_list))
                continue
            piece_strs = [u_txt[c0:c1] for _, c0, c1 in w_list]
            planned = plan_header_anchored_windows(
                piece_strs,
                max_window_tokens=max_chunk_tokens,
                header_max_chars=anchor_max_chars,
                tokenizer=tokenizer,
                overhead_tokens=2,
                per_chunk_extra_tokens=0,
            )
            doc_out: list[tuple[list[int], int, int]] = []
            doc_flags: list[bool] = []
            for pw in planned:
                c0 = w_list[pw.chunk_indices[0]][1]
                c1 = w_list[pw.chunk_indices[-1]][2]
                ids = tokenizer.encode("\n".join(pw.chunks), prefix=eff_prefix, max_length=max_chunk_tokens)
                doc_out.append((ids, c0, c1))
                doc_flags.append(bool(pw.has_anchor))
            all_windows.append(doc_out)
            all_flags.append(doc_flags)
        return all_windows, all_flags

    lead_ids, tail_ids = tokenizer._resolve_special_and_prefix_ids(eff_prefix)
    overhead = len(lead_ids) + len(tail_ids)
    eff_max_body = int(max_body_tokens) if max_body_tokens is not None else max(1, max_chunk_tokens - overhead)
    if eff_max_body <= 0:
        raise ValueError("max_chunk_tokens is too small to fit special tokens and prefix")
    if eff_max_body + overhead > max_chunk_tokens:
        raise ValueError("max_body_tokens plus special/prefix tokens exceeds max_chunk_tokens")
    if overlap < 0 or overlap >= eff_max_body:
        raise ValueError(f"overlap must be in [0, {eff_max_body - 1}]")

    raw_tok = tokenizer._tok
    encs = raw_tok.encode_batch(list(unit_texts), add_special_tokens=False)
    if not header_anchored:
        wins_out = [
            _chunk_from_token_offsets(
                u_txt,
                enc.ids,
                enc.offsets,
                tokenizer=tokenizer,
                lead_ids=lead_ids,
                tail_ids=tail_ids,
                eff_max_body=eff_max_body,
                overlap=overlap,
                max_chunk_tokens=max_chunk_tokens,
                snap_sentences=snap_sentences,
                snap_min_fraction=snap_min_fraction,
            )
            for u_txt, enc in zip(unit_texts, encs)
        ]
        return wins_out, [[False] * len(w) for w in wins_out]

    # Header-anchored window planning: reserve compact header anchor tokens so continuation chunks fit clean in max_chunk_tokens.
    max_anc_cap = int(anchor_max_tokens) if anchor_max_tokens is not None else min(96, max(16, eff_max_body // 4))
    anchored_encode_texts: list[str] = []
    anchored_meta: list[tuple[int, int, int, int, bool]] = []  # (doc_i, chunk_idx_in_doc, c0, c1, has_anchor)
    doc_results: list[list[tuple[list[int], int, int]]] = [[] for _ in unit_texts]
    doc_flags: list[list[bool]] = [[] for _ in unit_texts]

    for doc_i, (u_txt, enc) in enumerate(zip(unit_texts, encs)):
        if len(enc.ids) <= eff_max_body:
            base = _chunk_from_token_offsets(
                u_txt,
                enc.ids,
                enc.offsets,
                tokenizer=tokenizer,
                lead_ids=lead_ids,
                tail_ids=tail_ids,
                eff_max_body=eff_max_body,
                overlap=overlap,
                max_chunk_tokens=max_chunk_tokens,
                snap_sentences=snap_sentences,
                snap_min_fraction=snap_min_fraction,
            )
            doc_results[doc_i] = base
            doc_flags[doc_i] = [False] * len(base)
            continue

        hdr_text, hdr_toks = _compute_header_anchor(
            u_txt,
            tokenizer=tokenizer,
            anchor_max_chars=anchor_max_chars,
            max_anchor_tokens=max_anc_cap,
        )
        eff_sub_body = max(16, eff_max_body - hdr_toks - 2)
        eff_sub_overlap = min(overlap, max(0, eff_sub_body - 1))
        sub_chunks = _chunk_from_token_offsets(
            u_txt,
            enc.ids,
            enc.offsets,
            tokenizer=tokenizer,
            lead_ids=lead_ids,
            tail_ids=tail_ids,
            eff_max_body=eff_sub_body,
            overlap=eff_sub_overlap,
            max_chunk_tokens=max_chunk_tokens,
            snap_sentences=snap_sentences,
            snap_min_fraction=snap_min_fraction,
        )
        if len(sub_chunks) <= 1 or not hdr_text:
            doc_results[doc_i] = sub_chunks
            doc_flags[doc_i] = [False] * len(sub_chunks)
            continue

        # Ensure piece_strs[0][:eff_hdr_chars] equals hdr_text cleanly for plan_header_anchored_windows.
        piece_strs = [u_txt[c0:c1] for _, c0, c1 in sub_chunks]
        eff_hdr_chars = min(len(hdr_text), len(piece_strs[0]))
        planned = plan_header_anchored_windows(
            piece_strs,
            max_window_tokens=max_chunk_tokens,
            header_max_chars=eff_hdr_chars,
            tokenizer=tokenizer,
            overhead_tokens=overhead,
            per_chunk_extra_tokens=1,
        )
        for pw in planned:
            c0 = sub_chunks[pw.chunk_indices[0]][1]
            c1 = sub_chunks[pw.chunk_indices[-1]][2]
            if not pw.has_anchor and len(pw.chunks) == 1:
                slot = len(doc_results[doc_i])
                doc_results[doc_i].append((sub_chunks[pw.chunk_indices[0]][0], c0, c1))
                doc_flags[doc_i].append(False)
                continue
            slot = len(doc_results[doc_i])
            doc_results[doc_i].append(([], c0, c1))
            doc_flags[doc_i].append(bool(pw.has_anchor))
            anchored_encode_texts.append("\n".join(pw.chunks))
            anchored_meta.append((doc_i, slot, c0, c1, bool(pw.has_anchor)))

    if anchored_encode_texts:
        encoded_batch = tokenizer.encode_batch(anchored_encode_texts, prefix=eff_prefix, max_length=max_chunk_tokens)
        for (doc_i, slot, c0, c1, _), tok_ids in zip(anchored_meta, encoded_batch):
            doc_results[doc_i][slot] = (tok_ids, c0, c1)

    return doc_results, doc_flags


def _prepare_batch_rows(
    rb: pa.RecordBatch,
    *,
    tokenizer: Any,
    text_col: str,
    keep_cols: list[str] | None,
    drop_text_col: bool,
    ocr_pages_only: bool,
    per_ocr_page: bool,
    chunk: bool,
    max_body_tokens: int | None,
    overlap: int,
    max_encode_length: int,
    max_chunk_tokens: int,
    skip_empty: bool,
    include_chunk_text: bool,
    prefix: str | None,
    prompt_name: str | None,
    add_query_prefix: bool,
    add_document_prefix: bool,
    snap_sentences: bool = False,
    snap_min_fraction: float = 0.55,
    header_anchored: bool = False,
    anchor_max_chars: int = 2048,
    anchor_max_tokens: int | None = None,
) -> tuple[pa.Table, list[list[int]]]:
    schema_names = rb.schema.names
    if keep_cols is not None:
        meta_cols = [c for c in keep_cols if c in schema_names]
    else:
        meta_cols = [c for c in schema_names if not (drop_text_col and c == text_col)]

    raw_texts = rb.column(text_col).to_pylist()
    if ocr_pages_only or per_ocr_page:
        offsets_col = rb.column("page_offsets").to_pylist() if "page_offsets" in schema_names else [None] * rb.num_rows
        ocr_pages_col = rb.column("ocr_pages").to_pylist() if "ocr_pages" in schema_names else [None] * rb.num_rows
    else:
        offsets_col = None
        ocr_pages_col = None

    eff_prefix = tokenizer.resolve_prefix(
        prefix=prefix,
        prompt_name=prompt_name,
        add_query_prefix=add_query_prefix,
        add_document_prefix=add_document_prefix,
    )

    if not chunk and not per_ocr_page:
        if ocr_pages_only:
            texts = [
                slice_ocr_pages_utf8(raw_texts[i], offsets_col[i], ocr_pages_col[i])
                for i in range(rb.num_rows)
            ]
        else:
            texts = [t if isinstance(t, str) else "" for t in raw_texts]

        if skip_empty:
            keep_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not keep_indices:
                return pa.Table.from_batches([rb.slice(0, 0)]).select(meta_cols), []
            if len(keep_indices) < len(texts):
                rb = rb.take(keep_indices)
                texts = [texts[i] for i in keep_indices]

        tok_lists = tokenizer.encode_batch(texts, prefix=eff_prefix, max_length=max_encode_length)
        meta_table = pa.Table.from_batches([rb]).select(meta_cols)
        return meta_table, tok_lists

    meta_only_cols = [c for c in meta_cols if c != "page_offsets"]
    all_units: list[tuple[int, int, str]] = []
    for i in range(rb.num_rows):
        t = raw_texts[i] if isinstance(raw_texts[i], str) else ""
        if per_ocr_page and offsets_col is not None and ocr_pages_col is not None:
            raw_bytes = t.encode("utf-8")
            po = offsets_col[i] or []
            op = ocr_pages_col[i] or []
            n_pages = len(po)
            total_b = len(raw_bytes)
            for pnum in op:
                pidx = int(pnum) - 1
                if 0 <= pidx < n_pages:
                    b0 = max(0, min(total_b, int(po[pidx])))
                    b1 = total_b if (pidx + 1 >= n_pages) else max(b0, min(total_b, int(po[pidx + 1])))
                    seg = raw_bytes[b0:b1].decode("utf-8", errors="replace")
                    if seg.strip() or not skip_empty:
                        all_units.append((i, int(pnum), seg))
        elif ocr_pages_only and offsets_col is not None and ocr_pages_col is not None:
            seg = slice_ocr_pages_utf8(t, offsets_col[i], ocr_pages_col[i])
            if seg or not skip_empty:
                all_units.append((i, -1, seg))
        else:
            if t.strip() or not skip_empty:
                all_units.append((i, -1, t))

    if not all_units:
        empty = pa.Table.from_batches([rb.slice(0, 0)]).select(meta_only_cols)
        return empty, []

    expanded_row_idx: list[int] = []
    expanded_tok_lists: list[list[int]] = []
    page_numbers: list[int] = []
    chunk_indices: list[int] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    tok_counts: list[int] = []
    chunk_texts: list[str] = []
    anchor_flags: list[bool] = []

    row_chunk_counter: dict[int, int] = {}
    unit_texts = [u_txt for _, _, u_txt in all_units]

    if chunk:
        unit_windows, unit_anchor_flags = _chunk_units_with_options(
            unit_texts,
            tokenizer=tokenizer,
            eff_prefix=eff_prefix,
            max_body_tokens=max_body_tokens,
            overlap=overlap,
            max_chunk_tokens=max_chunk_tokens,
            snap_sentences=snap_sentences,
            snap_min_fraction=snap_min_fraction,
            header_anchored=header_anchored,
            anchor_max_chars=anchor_max_chars,
            anchor_max_tokens=anchor_max_tokens,
        )
        for (row_i, pnum, unit_text), win_list, fl_list in zip(all_units, unit_windows, unit_anchor_flags):
            c_idx = row_chunk_counter.get(row_i, 0)
            for (win_ids, c0, c1), has_anc in zip(win_list, fl_list):
                expanded_row_idx.append(row_i)
                expanded_tok_lists.append(win_ids)
                if per_ocr_page:
                    page_numbers.append(pnum)
                chunk_indices.append(c_idx)
                char_starts.append(c0)
                char_ends.append(c1)
                tok_counts.append(len(win_ids))
                if include_chunk_text:
                    chunk_texts.append(unit_text[c0:c1])
                if header_anchored:
                    anchor_flags.append(bool(has_anc))
                c_idx += 1
            row_chunk_counter[row_i] = c_idx
    else:
        if hasattr(tokenizer, "encode_batch"):
            encoded_units = tokenizer.encode_batch(unit_texts, prefix=eff_prefix, max_length=max_encode_length)
        else:
            encoded_units = [tokenizer.encode(u_txt, prefix=eff_prefix, max_length=max_encode_length) for u_txt in unit_texts]
        for (row_i, pnum, unit_text), ids in zip(all_units, encoded_units):
            c_idx = row_chunk_counter.get(row_i, 0)
            expanded_row_idx.append(row_i)
            expanded_tok_lists.append(ids)
            if per_ocr_page:
                page_numbers.append(pnum)
            chunk_indices.append(c_idx)
            char_starts.append(0)
            char_ends.append(len(unit_text))
            tok_counts.append(len(ids))
            if include_chunk_text:
                chunk_texts.append(unit_text)
            row_chunk_counter[row_i] = c_idx + 1

    if not expanded_row_idx:
        empty = pa.Table.from_batches([rb.slice(0, 0)]).select(meta_only_cols)
        return empty, []

    rb_meta = rb.select(meta_only_cols)
    idx_arr = pa.array(expanded_row_idx, type=pa.int32())
    if rb_meta.num_columns == 0:
        meta_table = pa.Table.from_batches([rb_meta.take(idx_arr)])
    elif len(expanded_row_idx) > rb.num_rows:
        taken_cols: list[pa.Array] = []
        taken_fields: list[pa.Field] = []
        for field, col in zip(rb_meta.schema, rb_meta.columns):
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                orig_tag = b"large_string" if pa.types.is_large_string(field.type) else b"string"
                d_col = col.dictionary_encode().take(idx_arr)
                new_meta = dict(field.metadata or {})
                new_meta[_FAST_DICT_ORIG_META_KEY] = orig_tag
                taken_cols.append(d_col)
                taken_fields.append(field.with_type(d_col.type).with_metadata(new_meta))
            else:
                taken_cols.append(col.take(idx_arr))
                taken_fields.append(field)
        meta_table = pa.Table.from_arrays(
            taken_cols,
            schema=pa.schema(taken_fields, metadata=rb_meta.schema.metadata),
        )
    else:
        meta_table = pa.Table.from_batches([rb_meta.take(idx_arr)])
    if per_ocr_page:
        meta_table = meta_table.append_column("ocr_page_number", pa.array(page_numbers, type=pa.int32()))
    if chunk:
        meta_table = meta_table.append_column("chunk_index", pa.array(chunk_indices, type=pa.int32()))
        meta_table = meta_table.append_column("char_start", pa.array(char_starts, type=pa.int32()))
        meta_table = meta_table.append_column("char_end", pa.array(char_ends, type=pa.int32()))
        meta_table = meta_table.append_column("token_count", pa.array(tok_counts, type=pa.int32()))
        if header_anchored:
            meta_table = meta_table.append_column("has_header_anchor", pa.array(anchor_flags, type=pa.bool_()))
    if include_chunk_text:
        meta_table = meta_table.append_column("chunk_text", pa.array(chunk_texts, type=pa.string()))
    return meta_table, expanded_tok_lists


_FAST_DICT_ORIG_META_KEY = b"_fast_dict_orig"


def _decode_temp_dict_columns(tbl: pa.Table) -> pa.Table:
    """Decode temporary dictionary-encoded string metadata columns back to exact original `string`/`large_string` types."""
    if not any(f.metadata and _FAST_DICT_ORIG_META_KEY in f.metadata for f in tbl.schema):
        return tbl
    out_cols: list[pa.ChunkedArray] = []
    out_fields: list[pa.Field] = []
    for field, col in zip(tbl.schema, tbl.columns):
        if field.metadata and _FAST_DICT_ORIG_META_KEY in field.metadata:
            orig_kind = field.metadata[_FAST_DICT_ORIG_META_KEY]
            target_type = pa.large_string() if orig_kind == b"large_string" else pa.string()
            rem_meta = {k: v for k, v in field.metadata.items() if k != _FAST_DICT_ORIG_META_KEY}
            clean_field = field.with_type(target_type).with_metadata(rem_meta if rem_meta else None)
            decoded_chunks = [
                ch.dictionary_decode() if pa.types.is_dictionary(ch.type) else ch.cast(target_type)
                for ch in col.chunks
            ]
            out_cols.append(pa.chunked_array(decoded_chunks, type=target_type))
            out_fields.append(clean_field)
        else:
            out_cols.append(col)
            out_fields.append(field)
    return pa.Table.from_arrays(out_cols, schema=pa.schema(out_fields, metadata=tbl.schema.metadata))


def _align_dict_meta_parts(parts: list[pa.Table]) -> list[pa.Table]:
    """Ensure multi-batch `pending_meta_parts` agree on temporary dictionary encoding before `pa.concat_tables`."""
    if len(parts) <= 1:
        return parts
    dict_orig_by_col: dict[str, bytes] = {}
    for pt in parts:
        for f in pt.schema:
            if f.metadata and _FAST_DICT_ORIG_META_KEY in f.metadata:
                dict_orig_by_col[f.name] = f.metadata[_FAST_DICT_ORIG_META_KEY]
    if not dict_orig_by_col:
        return parts
    aligned: list[pa.Table] = []
    for pt in parts:
        needs_fix = any(
            f.name in dict_orig_by_col and not pa.types.is_dictionary(f.type)
            for f in pt.schema
        )
        if not needs_fix:
            aligned.append(pt)
            continue
        cols: list[pa.ChunkedArray] = []
        fields: list[pa.Field] = []
        for f, col in zip(pt.schema, pt.columns):
            if f.name in dict_orig_by_col and not pa.types.is_dictionary(f.type):
                enc_chunks = [ch.dictionary_encode() for ch in col.chunks]
                d_col = pa.chunked_array(enc_chunks) if enc_chunks else pa.chunked_array([], type=pa.dictionary(pa.int32(), f.type))
                new_meta = dict(f.metadata or {})
                new_meta[_FAST_DICT_ORIG_META_KEY] = dict_orig_by_col[f.name]
                cols.append(d_col)
                fields.append(f.with_type(d_col.type).with_metadata(new_meta))
            else:
                cols.append(col)
                fields.append(f)
        aligned.append(pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=pt.schema.metadata)))
    return aligned


def _enter_mlx_limits(
    stack: ExitStack,
    *,
    mlx_cache_limit_mb: int | None,
    use_wired_limit: bool,
) -> None:
    if mlx_cache_limit_mb is not None:
        prev_limit: int | None = None
        try:
            import mlx.core as mx

            set_limit = getattr(mx, "set_cache_limit", None)
            if callable(set_limit):
                prev = set_limit(max(0, int(mlx_cache_limit_mb)) * 1024 * 1024)
                if isinstance(prev, int):
                    prev_limit = prev
        except Exception:
            set_mlx_cache_limit(int(mlx_cache_limit_mb))
        if prev_limit is not None:
            def _restore(val: int = prev_limit) -> None:
                try:
                    import mlx.core as mx

                    set_limit = getattr(mx, "set_cache_limit", None)
                    if callable(set_limit):
                        set_limit(val)
                except Exception:
                    pass

            stack.callback(_restore)
    if use_wired_limit:
        stack.enter_context(wired_limit())


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_manifest(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _expected_arrow_val_type(expected_dtype: str) -> pa.DataType:
    if expected_dtype == "int8":
        return pa.int8()
    if expected_dtype == "float16":
        return pa.float16()
    return pa.float32()


def _shard_matches_expected(
    shard_path: Path,
    *,
    expected_rows: int,
    expected_meta_cols: list[str],
    expected_dim: int,
    expected_dtype: str,
    manifest: dict[str, Any],
    run_config: dict[str, Any],
) -> bool:
    if not shard_path.is_file():
        return False
    if manifest and isinstance(manifest.get("config"), dict) and manifest["config"] != run_config:
        return False
    manifest_shards = manifest.get("shards") if isinstance(manifest.get("shards"), dict) else {}
    if manifest_shards and shard_path.name in manifest_shards:
        entry = manifest_shards[shard_path.name]
        if (
            not isinstance(entry, dict)
            or entry.get("num_rows") != expected_rows
            or entry.get("dim") != expected_dim
            or entry.get("dtype") != expected_dtype
        ):
            return False
    try:
        pf = pq.ParquetFile(shard_path)
        if pf.metadata.num_rows != expected_rows:
            return False
        sch = pf.schema_arrow
        if "vector" not in sch.names:
            return False
        if expected_dtype == "int8" and "embedding_scale" not in sch.names:
            return False
        vec_type = sch.field("vector").type
        exp_val_type = _expected_arrow_val_type(expected_dtype)
        if not pa.types.is_fixed_size_list(vec_type):
            return False
        if vec_type.list_size != expected_dim or vec_type.value_type != exp_val_type:
            return False
        actual_meta_cols = [n for n in sch.names if n not in ("vector", "embedding_scale")]
        if actual_meta_cols != expected_meta_cols:
            return False
        return True
    except Exception:
        return False


def _normalize_truncated(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings to `[:dim]` and L2-normalize (preserving native `int8` pass-through)."""
    arr = np.asarray(vecs)
    if arr.dtype == np.int8:
        return np.ascontiguousarray(arr[..., :dim], dtype=np.int8)
    sub = truncate_and_normalize(arr, dim)
    if sub.shape[0] == 0:
        return sub
    norms = np.linalg.norm(sub, axis=-1, keepdims=True)
    return np.ascontiguousarray(sub / np.maximum(norms, 1e-12), dtype=np.float32)


def _partition_macro_windows(tok_lists: Sequence[Sequence[int]], budget_tokens: int) -> list[tuple[int, int]]:
    n = len(tok_lists)
    if n == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    used = 0
    for idx in range(n):
        tlen = max(1, len(tok_lists[idx]))
        if idx > start and (used + tlen) > budget_tokens:
            spans.append((start, idx))
            start = idx
            used = tlen
        else:
            used += tlen
    spans.append((start, n))
    return spans


def embed_parquet(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_col: str | None = None,
    keep_cols: list[str] | None = None,
    drop_text_col: bool = False,
    mode: str | None = None,
    dtype: str = "float32",
    batch_rows: int = 4096,
    encode_batch_rows: int = 4096,
    shard_rows: int = 65536,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    per_ocr_page: bool = False,
    chunk: bool = False,
    max_body_tokens: int | None = None,
    overlap: int = 51,
    snap_sentences: bool = False,
    snap_min_fraction: float = 0.55,
    header_anchored: bool = False,
    anchor_max_chars: int = 2048,
    anchor_max_tokens: int | None = None,
    pin_non_hetero_cutoffs: bool | None = None,
    non_hetero_min_batch: int | None = None,
    non_hetero_min_tokens: int | None = None,
    max_length: int | None = None,
    max_chunk_tokens: int = 512,
    skip_empty: bool = False,
    include_chunk_text: bool = False,
    prefix: str | None = None,
    prompt_name: str | None = None,
    add_query_prefix: bool = False,
    add_document_prefix: bool = False,
    dimensions: int | None = None,
    mrl_dim: int | None = None,
    truncate_dim: int | None = None,
    compression: str | Mapping[str, str] = "zstd",
    resume: bool = False,
    limit: int | None = None,
    progress_every: int = 1,
    mlx_cache_limit_mb: int | None = 512,
    use_wired_limit: bool = True,
    sort_macro_window: bool = True,
    macro_window_tokens: int | None = None,
    lookahead_batches: int | None = None,
    lookahead_shard_multiplier: int | None = None,
    lookahead_max_tokens: int | None = None,
    use_nocache: bool = False,
    memory_governor: LiveMemoryGovernor | bool | None = True,
    uma_ceiling_fraction: float = WIRED_MEMORY_CEILING_FRACTION,
    sink_format: str | None = None,
    _clamp_mrl_dim: bool = False,
) -> ParquetEmbedSummary:
    """Prepare source batches, length-sort token-budgeted macro-windows, and stream Parquet or Lance."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_SUPPORTED_DTYPES}")
    for name, value in (("batch_rows", batch_rows), ("encode_batch_rows", encode_batch_rows), ("shard_rows", shard_rows)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, opt_val in (
        ("lookahead_batches", lookahead_batches),
        ("lookahead_shard_multiplier", lookahead_shard_multiplier),
        ("lookahead_max_tokens", lookahead_max_tokens),
    ):
        if opt_val is not None and int(opt_val) <= 0:
            raise ValueError(f"{name} must be positive")

    dim_candidates = [d for d in (dimensions, mrl_dim, truncate_dim) if d is not None]
    if len(set(dim_candidates)) > 1:
        raise ValueError(
            f"Conflicting dimension overrides: dimensions={dimensions!r}, mrl_dim={mrl_dim!r}, truncate_dim={truncate_dim!r}"
        )
    req_dim: Any = dim_candidates[0] if dim_candidates else None

    default_dim = int(getattr(engine, "dim"))
    max_dim = int(getattr(engine, "max_dim", default_dim))
    supports_dims = bool(getattr(engine, "supports_dimensions", False))
    if _clamp_mrl_dim and isinstance(req_dim, (int, np.integer)) and not isinstance(req_dim, bool):
        if int(req_dim) > max_dim or not supports_dims:
            req_dim = default_dim if not supports_dims else min(int(req_dim), max_dim)

    dim = resolve_embedding_dim(
        req_dim,
        max_dim=max_dim,
        default_dim=default_dim,
        supports_dimensions=supports_dims,
    )
    encode_fn = getattr(engine, "encode_corpus", None) or getattr(engine, "encode_tokens")
    corpus_kw: dict[str, Any] = {}
    accepts_mode = accepts_kwarg(encode_fn, "mode")
    if accepts_mode:
        corpus_kw["mode"] = mode
    if req_dim is not None and accepts_kwarg(encode_fn, "dimensions"):
        corpus_kw["dimensions"] = dim

    def _call_encode_fn(sub_toks: list[list[int]]) -> np.ndarray:
        if not accepts_mode:
            return encode_fn(sub_toks, **corpus_kw)
        eff_batch_mode = resolve_embedding_device_mode(
            engine,
            sub_toks,
            mode=mode,
            pin_non_hetero_cutoffs=pin_non_hetero_cutoffs,
            non_hetero_min_batch=non_hetero_min_batch,
            non_hetero_min_tokens=non_hetero_min_tokens,
        )
        call_kw = dict(corpus_kw)
        call_kw["mode"] = eff_batch_mode
        return encode_fn(sub_toks, **call_kw)

    eff_macro_tokens = resolve_macro_window_tokens(macro_window_tokens)
    engine_max_length = int(getattr(engine, "max_seq_len", 512))
    max_encode_length = int(max_length if max_length is not None else engine_max_length)
    if max_encode_length <= 0 or max_encode_length > engine_max_length:
        raise ValueError(f"max_length must be positive and no greater than the engine limit ({engine_max_length})")
    if max_chunk_tokens <= 0 or (chunk and max_chunk_tokens > engine_max_length):
        raise ValueError(f"max_chunk_tokens must be positive and no greater than the engine limit ({engine_max_length}) when chunking")

    out_p = Path(output_path).expanduser().resolve()
    eff_sink = (sink_format or ("lance" if out_p.suffix.lower() == ".lance" else "parquet")).strip().lower()
    if eff_sink not in ("parquet", "lance"):
        raise ValueError(f"Unsupported sink_format {sink_format!r}; expected 'parquet' or 'lance'")
    lance_mod = None
    if eff_sink == "lance":
        lance_mod = _import_lance()
        if lance_mod is None:
            raise RuntimeError(
                "Lance sink requested (`sink_format='lance'` or `.lance` output suffix), "
                "but optional dependency `pylance` (`import lance`) is not installed."
            )

    files = resolve_input_files(input_path)
    first_schema = pq.ParquetFile(files[0]).schema_arrow
    eff_text_col = resolve_text_column(first_schema, text_col)

    lance_mode = eff_sink == "lance"
    single_file_mode = (not lance_mode) and (out_p.suffix.lower() == ".parquet")
    if single_file_mode or lance_mode:
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_p.mkdir(parents=True, exist_ok=True)

    run_config: dict[str, Any] = {
        "dim": dim,
        "dtype": dtype,
        "text_col": eff_text_col,
        "keep_cols": keep_cols,
        "drop_text_col": drop_text_col,
        "shard_rows": shard_rows,
        "filter_ocr": filter_ocr,
        "ocr_pages_only": ocr_pages_only,
        "per_ocr_page": per_ocr_page,
        "chunk": chunk,
        "max_body_tokens": max_body_tokens,
        "overlap": overlap,
        "snap_sentences": snap_sentences,
        "header_anchored": header_anchored,
        "anchor_max_chars": anchor_max_chars if header_anchored else None,
        "max_encode_length": max_encode_length,
        "max_chunk_tokens": max_chunk_tokens,
        "skip_empty": skip_empty,
        "include_chunk_text": include_chunk_text,
        "prefix": prefix,
        "prompt_name": prompt_name,
        "add_query_prefix": add_query_prefix,
        "add_document_prefix": add_document_prefix,
        "input_files": [[f.name, f.stat().st_size] for f in files],
    }
    manifest_path = out_p / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path) if (not single_file_mode and not lance_mode and resume) else {}
    manifest_shards: dict[str, Any] = (
        dict(manifest.get("shards", {}))
        if (manifest.get("config") == run_config and isinstance(manifest.get("shards"), dict))
        else {}
    )
    manifest_lock = threading.Lock()

    if keep_cols is not None:
        needed = list(dict.fromkeys([*keep_cols, eff_text_col]))
    else:
        needed = list(first_schema.names)
    for aux in ("is_ocr", "ocr_pages", "page_offsets"):
        if aux in first_schema.names and aux not in needed:
            needed.append(aux)

    if isinstance(memory_governor, LiveMemoryGovernor):
        governor: LiveMemoryGovernor | None = memory_governor
    elif memory_governor:
        governor = LiveMemoryGovernor(uma_ceiling_fraction=uma_ceiling_fraction)
    else:
        governor = None

    eff_cache_mb = DEFAULT_MLX_CACHE_LIMIT_MB if mlx_cache_limit_mb is None else int(mlx_cache_limit_mb)
    if not sort_macro_window:
        eff_lookahead_batches = 1
        eff_shard_mult = 1
    else:
        eff_lookahead_batches = (
            int(lookahead_batches)
            if lookahead_batches is not None
            else (4 if batch_rows >= 64 else 1)
        )
        if lookahead_shard_multiplier is not None:
            eff_shard_mult = int(lookahead_shard_multiplier)
        elif lookahead_batches is not None:
            eff_shard_mult = max(1, int(lookahead_batches))
        else:
            eff_shard_mult = 4 if (shard_rows >= 64 and batch_rows >= 64) else 1
    eff_lookahead_toks = (
        min(eff_macro_tokens, int(lookahead_max_tokens))
        if lookahead_max_tokens is not None
        else eff_macro_tokens
    )

    prep_queue: queue.Queue[tuple[int, pa.Table, list[list[int]]] | None | BaseException] = queue.Queue(
        maxsize=max(2, min(8, max(eff_lookahead_batches, eff_shard_mult)))
    )
    prod_stop = threading.Event()
    rows_read_box = [0]

    def _put_prep(item: tuple[int, pa.Table, list[list[int]]] | None | BaseException) -> bool:
        while not prod_stop.is_set():
            try:
                prep_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        try:
            for rb in _iter_source_batches(
                files,
                needed,
                batch_rows,
                filter_ocr=filter_ocr,
                limit=limit,
                use_nocache=use_nocache,
            ):
                if prod_stop.is_set():
                    break
                rows_read_box[0] += rb.num_rows
                meta_tbl, tok_lists = _prepare_batch_rows(
                    rb,
                    tokenizer=engine.tokenizer,
                    text_col=eff_text_col,
                    keep_cols=keep_cols,
                    drop_text_col=drop_text_col,
                    ocr_pages_only=ocr_pages_only,
                    per_ocr_page=per_ocr_page,
                    chunk=chunk,
                    max_body_tokens=max_body_tokens,
                    overlap=overlap,
                    snap_sentences=snap_sentences,
                    snap_min_fraction=snap_min_fraction,
                    header_anchored=header_anchored,
                    anchor_max_chars=anchor_max_chars,
                    anchor_max_tokens=anchor_max_tokens,
                    max_encode_length=max_encode_length,
                    max_chunk_tokens=max_chunk_tokens,
                    skip_empty=skip_empty,
                    include_chunk_text=include_chunk_text,
                    prefix=prefix,
                    prompt_name=prompt_name,
                    add_query_prefix=add_query_prefix,
                    add_document_prefix=add_document_prefix,
                )
                if len(tok_lists) > 0 and not _put_prep((rb.num_rows, meta_tbl, tok_lists)):
                    break
        except BaseException as exc:
            _put_prep(exc)
        finally:
            _put_prep(None)

    t0 = time.perf_counter()
    prod_thread = threading.Thread(target=producer, daemon=True)

    single_writer: pq.ParquetWriter | None = None
    tmp_single_path = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    lance_created = False
    shard_idx = 0
    pending_meta_parts: list[pa.Table] = []
    pending_meta_rows = 0
    pending_toks: list[list[int]] = []
    pending_tok_sum = 0
    sl_meta_parts: list[pa.Table] = []
    sl_toks: list[list[int]] = []
    sl_tok_sum = 0

    rows_emitted = 0
    total_tokens = 0
    batch_num = 0

    write_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=2)
    writer_err: list[BaseException] = []
    writer_stop = threading.Event()

    def _writer_worker() -> None:
        nonlocal single_writer, lance_created
        try:
            while not writer_stop.is_set():
                try:
                    w_item = write_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if w_item is None:
                    break
                kind = w_item[0]
                if kind == "lance":
                    _, m_tbl, vecs = w_item
                    out_tbl = _append_vector_columns(_decode_temp_dict_columns(m_tbl), vecs, dim=dim, dtype=dtype)
                    assert lance_mod is not None
                    w_mode = "append" if lance_created else "create"
                    lance_mod.write_dataset(out_tbl, str(out_p), mode=w_mode)
                    lance_created = True
                elif kind == "single":
                    _, m_tbl, vecs = w_item
                    out_tbl = _append_vector_columns(_decode_temp_dict_columns(m_tbl), vecs, dim=dim, dtype=dtype)
                    if single_writer is None:
                        col_comp = _resolve_parquet_compression(out_tbl.schema, compression)
                        single_writer = pq.ParquetWriter(tmp_single_path, out_tbl.schema, compression=col_comp)
                    single_writer.write_table(out_tbl)
                else:
                    _, m_slice, vecs, shard_path, tmp_shard, n_shard_toks = w_item
                    out_tbl = _append_vector_columns(_decode_temp_dict_columns(m_slice), vecs, dim=dim, dtype=dtype)
                    col_comp = _resolve_parquet_compression(out_tbl.schema, compression)
                    try:
                        pq.write_table(out_tbl, tmp_shard, compression=col_comp)
                        os.replace(tmp_shard, shard_path)
                        with manifest_lock:
                            manifest_shards[shard_path.name] = {
                                "num_rows": out_tbl.num_rows,
                                "tokens": int(n_shard_toks),
                                "dim": dim,
                                "dtype": dtype,
                            }
                            _save_manifest(manifest_path, {"config": run_config, "shards": manifest_shards})
                    except BaseException:
                        tmp_shard.unlink(missing_ok=True)
                        raise
        except BaseException as exc:
            writer_err.append(exc)
            writer_stop.set()
        finally:
            if single_writer is not None:
                try:
                    single_writer.close()
                finally:
                    single_writer = None

    writer_thread = threading.Thread(target=_writer_worker, daemon=True)

    def _enqueue_write(job: tuple | None) -> None:
        while True:
            if writer_err:
                raise writer_err[0]
            try:
                write_queue.put(job, timeout=0.05)
                return
            except queue.Full:
                continue

    def _encode_sorted_chunk(sub_toks: list[list[int]]) -> np.ndarray:
        if len(sub_toks) <= encode_batch_rows:
            out_vecs = _normalize_truncated(_call_encode_fn(sub_toks), dim)
            clear_mlx_arena(256)
            return out_vecs
        first = _normalize_truncated(_call_encode_fn(sub_toks[:encode_batch_rows]), dim)
        clear_mlx_arena(256)
        vecs = np.empty((len(sub_toks), dim), dtype=first.dtype)
        vecs[:encode_batch_rows] = first
        for start in range(encode_batch_rows, len(sub_toks), encode_batch_rows):
            end = min(start + encode_batch_rows, len(sub_toks))
            vecs[start:end] = _normalize_truncated(_call_encode_fn(sub_toks[start:end]), dim)
            clear_mlx_arena(256)
        return vecs

    def encode_rows(tok_lists: list[list[int]]) -> np.ndarray:
        if not tok_lists:
            return np.empty((0, dim), dtype=np.float32)
        if not sort_macro_window or len(tok_lists) <= 1:
            return _encode_sorted_chunk(tok_lists)

        macro_spans = _partition_macro_windows(tok_lists, eff_macro_tokens)
        out_all: np.ndarray | None = None
        for w_start, w_end in macro_spans:
            win_toks = tok_lists[w_start:w_end]
            order = sorted(range(len(win_toks)), key=lambda idx: len(win_toks[idx]))
            sorted_toks = [win_toks[idx] for idx in order]
            sorted_vecs = _encode_sorted_chunk(sorted_toks)
            if out_all is None:
                out_all = np.empty((len(tok_lists), dim), dtype=sorted_vecs.dtype)
            for sorted_i, orig_local_i in enumerate(order):
                out_all[w_start + orig_local_i] = sorted_vecs[sorted_i]
        assert out_all is not None
        return out_all

    def _govern_lookahead(base_mult: int) -> tuple[int, int]:
        if governor is None or base_mult <= 1:
            return base_mult, eff_lookahead_toks
        ad = governor.adapt(
            base_b_slots=base_mult,
            base_cache_limit_mb=eff_cache_mb,
            apply_mlx=True,
        )
        eff_mult = max(1, min(base_mult, int(ad.b_slots)))
        eff_toks = max(1, (eff_lookahead_toks * eff_mult) // max(1, base_mult))
        return eff_mult, eff_toks

    def drain_stream(final: bool = False) -> None:
        nonlocal sl_meta_parts, sl_toks, sl_tok_sum, rows_emitted, total_tokens
        if not sl_toks:
            return
        cur_mult, cur_toks_cap = _govern_lookahead(eff_lookahead_batches)
        if not final and len(sl_toks) < (batch_rows * cur_mult) and sl_tok_sum < cur_toks_cap:
            return
        combined_meta = (
            sl_meta_parts[0]
            if len(sl_meta_parts) == 1
            else pa.concat_tables(_align_dict_meta_parts(sl_meta_parts), promote_options="default")
        )
        toks_out = sl_toks
        tok_cnt = sl_tok_sum
        sl_meta_parts = []
        sl_toks = []
        sl_tok_sum = 0
        vecs = encode_rows(toks_out)
        rows_emitted += len(toks_out)
        total_tokens += tok_cnt
        _enqueue_write(("lance" if lance_mode else "single", combined_meta, vecs))

    def drain_shards(final: bool = False) -> None:
        nonlocal pending_meta_parts, pending_meta_rows, pending_toks, pending_tok_sum, shard_idx, rows_emitted, total_tokens
        if not pending_meta_parts:
            return
        cur_mult, cur_toks_cap = _govern_lookahead(eff_shard_mult)
        if not final:
            if pending_meta_rows < shard_rows:
                return
            if cur_mult > 1 and pending_meta_rows < (shard_rows * cur_mult) and pending_tok_sum < cur_toks_cap:
                return

        combined = (
            pending_meta_parts[0]
            if len(pending_meta_parts) == 1
            else pa.concat_tables(_align_dict_meta_parts(pending_meta_parts), promote_options="default")
        )
        pending_meta_parts = []
        offset = 0
        total_avail = len(pending_toks)
        slices_to_process: list[tuple[pa.Table, list[list[int]]]] = []
        while (total_avail - offset) >= shard_rows:
            slices_to_process.append(
                (combined.slice(offset, shard_rows), pending_toks[offset : offset + shard_rows])
            )
            offset += shard_rows
        rem = total_avail - offset
        if final and rem > 0:
            slices_to_process.append((combined.slice(offset, rem), pending_toks[offset:]))
            pending_meta_rows = 0
            pending_toks = []
            pending_tok_sum = 0
        elif rem > 0:
            pending_meta_parts = [combined.slice(offset, rem)]
            pending_meta_rows = rem
            pending_toks = pending_toks[offset:]
            pending_tok_sum = sum(len(x) for x in pending_toks)
        else:
            pending_meta_rows = 0
            pending_toks = []
            pending_tok_sum = 0

        undone_slices: list[tuple[pa.Table, list[list[int]], Path, Path, int]] = []
        for meta_slice, tok_slice in slices_to_process:
            shard_path = out_p / f"part-{shard_idx:05d}.parquet"
            tmp_shard = out_p / f".part-{shard_idx:05d}.parquet.tmp"
            shard_idx += 1
            if resume and _shard_matches_expected(
                shard_path,
                expected_rows=len(tok_slice),
                expected_meta_cols=list(meta_slice.schema.names),
                expected_dim=dim,
                expected_dtype=dtype,
                manifest=manifest,
                run_config=run_config,
            ):
                continue
            undone_slices.append((meta_slice, tok_slice, shard_path, tmp_shard, sum(len(x) for x in tok_slice)))

        if not undone_slices:
            return
        if cur_mult <= 1 or len(undone_slices) == 1 or not sort_macro_window:
            for meta_slice, tok_slice, shard_path, tmp_shard, n_shard_toks in undone_slices:
                vecs = encode_rows(tok_slice)
                rows_emitted += len(tok_slice)
                total_tokens += n_shard_toks
                _enqueue_write(("shard", meta_slice, vecs, shard_path, tmp_shard, n_shard_toks))
            return

        # Cross-shard lookahead length sort: pack and sort across all undone slices once, then slice exact shards
        combined_toks = [t for _, tok_slice, *_ in undone_slices for t in tok_slice]
        vecs_all = encode_rows(combined_toks)
        vec_pos = 0
        for meta_slice, tok_slice, shard_path, tmp_shard, n_shard_toks in undone_slices:
            n_slice = len(tok_slice)
            vecs_i = vecs_all[vec_pos : vec_pos + n_slice]
            vec_pos += n_slice
            rows_emitted += n_slice
            total_tokens += n_shard_toks
            _enqueue_write(("shard", meta_slice, vecs_i, shard_path, tmp_shard, n_shard_toks))

    with ExitStack() as stack:
        _enter_mlx_limits(
            stack,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )
        prod_thread.start()
        writer_thread.start()
        try:
            while True:
                if writer_err:
                    raise writer_err[0]
                item = prep_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item

                _, meta_tbl, tok_lists = item
                batch_num += 1

                if lance_mode or single_file_mode:
                    sl_meta_parts.append(meta_tbl)
                    sl_toks.extend(tok_lists)
                    sl_tok_sum += sum(len(x) for x in tok_lists)
                    drain_stream(final=False)
                else:
                    pending_meta_parts.append(meta_tbl)
                    pending_meta_rows += meta_tbl.num_rows
                    pending_toks.extend(tok_lists)
                    pending_tok_sum += sum(len(x) for x in tok_lists)
                    drain_shards(final=False)

                if progress_every > 0 and (batch_num % progress_every == 0):
                    dt = time.perf_counter() - t0
                    print(
                        f"[parquet batch {batch_num}] rows={rows_emitted:,} toks={total_tokens:,} "
                        f"elapsed={dt:.2f}s ({total_tokens/max(dt, 1e-9):,.0f} tok/s)",
                        flush=True,
                    )

            if lance_mode or single_file_mode:
                drain_stream(final=True)
            else:
                drain_shards(final=True)

            _enqueue_write(None)
            writer_thread.join()
            if writer_err:
                raise writer_err[0]

            if single_file_mode:
                if rows_emitted > 0 and tmp_single_path is not None and tmp_single_path.exists():
                    os.replace(tmp_single_path, out_p)
                else:
                    extra_fields = [pa.field("vector", pa.list_(_expected_arrow_val_type(dtype), dim))]
                    if dtype == "int8":
                        extra_fields.append(pa.field("embedding_scale", pa.float16()))
                    empty_schema = pa.schema(
                        [f for f in first_schema if f.name != eff_text_col or not drop_text_col] + extra_fields
                    )
                    col_comp = _resolve_parquet_compression(empty_schema, compression)
                    pq.write_table(pa.Table.from_pylist([], schema=empty_schema), out_p, compression=col_comp)
            elif lance_mode and rows_emitted == 0 and lance_mod is not None:
                extra_fields = [pa.field("vector", pa.list_(_expected_arrow_val_type(dtype), dim))]
                if dtype == "int8":
                    extra_fields.append(pa.field("embedding_scale", pa.float16()))
                empty_schema = pa.schema(
                    [f for f in first_schema if f.name != eff_text_col or not drop_text_col] + extra_fields
                )
                lance_mod.write_dataset(pa.Table.from_pylist([], schema=empty_schema), str(out_p), mode="create")
        finally:
            prod_stop.set()
            writer_stop.set()
            for q in (prep_queue, write_queue):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            try:
                write_queue.put_nowait(None)
            except queue.Full:
                pass
            writer_thread.join(timeout=2.0)
            if tmp_single_path is not None and tmp_single_path.exists():
                tmp_single_path.unlink(missing_ok=True)
            prod_thread.join(timeout=2.0)
            clear_mlx_arena(256)

    elapsed = time.perf_counter() - t0
    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read_box[0],
        rows_emitted=rows_emitted,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        output_path=out_p,
    )


def build_backfill_cli_parser(
    parser: argparse.ArgumentParser | None = None,
    *,
    default_config: CorpusBackfillConfig | None = None,
    mode_choices: Sequence[str] = ("auto", "gpu", "ane", "hetero", "hetero2"),
    default_mode: str = "hetero",
) -> argparse.ArgumentParser:
    """Build 10M-doc scale corpus backfill CLI parser (`--dtype int8 --mrl-dim 512` default; `--dtype float32` one flag away)."""
    cfg = default_config or CorpusBackfillConfig()
    p = parser if parser is not None else argparse.ArgumentParser(description="10M-doc scale corpus embedding backfill.")
    p.add_argument("--input", required=True, help="Parquet file, directory of .parquet files, or glob")
    p.add_argument("--output", required=True, help="Output .parquet/.lance path or shard directory")
    p.add_argument("--dtype", choices=_SUPPORTED_DTYPES, default=cfg.dtype, help="Vector storage dtype (default: int8; pass --dtype float32 for full precision)")
    p.add_argument("--mrl-dim", "--truncate-dim", dest="mrl_dim", type=int, default=cfg.mrl_dim, help="Matryoshka output width (default: 512; pass 0 for full model dim)")
    p.add_argument("--mode", choices=tuple(mode_choices), default=default_mode)
    p.add_argument("--sink-format", choices=("parquet", "lance"), default=cfg.sink_format)
    p.add_argument("--batch-rows", type=int, default=cfg.batch_rows)
    p.add_argument("--encode-batch-rows", type=int, default=cfg.encode_batch_rows)
    p.add_argument("--shard-rows", type=int, default=cfg.shard_rows)
    p.add_argument("--macro-window-tokens", type=int, default=cfg.macro_window_tokens)
    p.add_argument("--no-sort-macro-window", action="store_true")
    p.add_argument("--chunk", action="store_true")
    p.add_argument("--max-chunk-tokens", type=int, default=512)
    p.add_argument("--overlap", type=int, default=51)
    p.add_argument("--snap-sentences", action="store_true", default=cfg.snap_sentences)
    p.add_argument("--header-anchored", action="store_true", default=cfg.header_anchored)
    p.add_argument("--anchor-max-chars", type=int, default=cfg.anchor_max_chars)
    p.add_argument("--pin-non-hetero-cutoffs", action="store_true", default=cfg.pin_non_hetero_cutoffs)
    p.add_argument("--non-hetero-min-batch", type=int, default=cfg.non_hetero_min_batch)
    p.add_argument("--non-hetero-min-tokens", type=int, default=cfg.non_hetero_min_tokens)
    p.add_argument("--compression", default=cfg.compression)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=1)
    return p


def run_corpus_backfill(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: CorpusBackfillConfig | None = None,
    **overrides: Any,
) -> ParquetEmbedSummary:
    """Execute `embed_parquet` with 10M-doc `CorpusBackfillConfig` defaults (`d512 int8`, uncompressed vectors)."""
    cfg = config or CorpusBackfillConfig()
    eff_mrl = overrides.pop("mrl_dim", cfg.mrl_dim)
    if eff_mrl is not None and int(eff_mrl) <= 0:
        eff_mrl = None
    kw: dict[str, Any] = {
        "dtype": cfg.dtype,
        "mrl_dim": eff_mrl,
        "sort_macro_window": cfg.sort_macro_window,
        "macro_window_tokens": cfg.macro_window_tokens,
        "use_wired_limit": cfg.use_wired_limit,
        "compression": cfg.compression,
        "batch_rows": cfg.batch_rows,
        "encode_batch_rows": cfg.encode_batch_rows,
        "shard_rows": cfg.shard_rows,
        "sink_format": cfg.sink_format,
        "snap_sentences": cfg.snap_sentences,
        "header_anchored": cfg.header_anchored,
        "anchor_max_chars": cfg.anchor_max_chars,
        "pin_non_hetero_cutoffs": cfg.pin_non_hetero_cutoffs,
        "non_hetero_min_batch": cfg.non_hetero_min_batch,
        "non_hetero_min_tokens": cfg.non_hetero_min_tokens,
        "_clamp_mrl_dim": True,
    }
    kw.update(overrides)
    return embed_parquet(engine, input_path, output_path, **kw)
