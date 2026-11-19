"""Streaming sharded Parquet embedding pipeline plus two-knob contextual windowing."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fast_shared.embeddings import resolve_embedding_dim
from fast_shared.parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    ParquetEmbedSummary,
    embed_parquet,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)

from .config import PplxConfig


def embed_contextual_parquet(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_col: str | None = None,
    keep_cols: Sequence[str] | None = None,
    drop_text_col: bool = False,
    chunk_tokens: int = 256,
    context_tokens: int = 2048,
    overlap: int = 32,
    mode: str | None = None,
    dtype: str = "float32",
    batch_rows: int = 256,
    encode_batch_windows: int = 16,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    compression: str = "zstd",
    limit: int | None = None,
) -> ParquetEmbedSummary:
    """Split each document into `<= chunk_tokens` spans, pack into `<= context_tokens`
    windows, and run ``engine.encode_contextual_chunks`` so every chunk vector reflects
    its surrounding window context without exceeding memory/retrieval sweet spots.
    """
    if int(chunk_tokens) <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if int(context_tokens) <= 0:
        raise ValueError(f"context_tokens must be positive, got {context_tokens}")

    t0 = time.perf_counter()
    files = resolve_input_files(input_path)
    first_pf = pq.ParquetFile(files[0])
    first_schema = first_pf.schema_arrow
    eff_text_col = resolve_text_column(first_schema, text_col)

    cols_to_read = list(first_schema.names if keep_cols is None else keep_cols)
    if eff_text_col not in cols_to_read:
        cols_to_read.append(eff_text_col)
    if ocr_pages_only:
        for req_col in ("page_offsets", "ocr_pages"):
            if req_col in first_schema.names and req_col not in cols_to_read:
                cols_to_read.append(req_col)

    max_dim = int(getattr(engine, "max_dim", getattr(engine, "dim", PplxConfig().embedding_dim)))
    default_dim = int(getattr(engine, "dim", max_dim))
    target_dim = resolve_embedding_dim(dimensions, max_dim=max_dim, default_dim=default_dim, supports_dimensions=True)
    vec_dtype = "float16" if dtype == "float16" else "float32"
    pa_vec_type = pa.list_(pa.float16() if dtype == "float16" else pa.float32(), target_dim)

    tokenizer = engine.tokenizer
    rows_read = 0
    rows_emitted = 0
    total_tokens = 0
    out_tables: list[pa.Table] = []

    keep_names = [c for c in (list(first_schema.names) if keep_cols is None else list(keep_cols)) if not (drop_text_col and c == eff_text_col)]

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rb in pf.iter_batches(batch_size=max(1, int(batch_rows)), columns=cols_to_read):
            py_rows = rb.to_pylist()
            for r in py_rows:
                if limit is not None and rows_read >= int(limit):
                    break
                rows_read += 1
                if filter_ocr:
                    is_ocr = bool(r.get("is_ocr"))
                    ocr_p = r.get("ocr_pages")
                    if not (is_ocr or (ocr_p and len(ocr_p) > 0)):
                        continue
                raw_text = r.get(eff_text_col)
                if ocr_pages_only:
                    raw_text = slice_ocr_pages_utf8(raw_text, r.get("page_offsets"), r.get("ocr_pages"))
                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue

                grouped = tokenizer.chunk_and_group_document(
                    raw_text,
                    chunk_tokens=int(chunk_tokens),
                    context_tokens=int(context_tokens),
                    overlap=int(overlap),
                )
                if not grouped:
                    continue

                win_chunks = [win for win, _ in grouped]
                win_spans = [spans for _, spans in grouped]
                win_vecs: list[np.ndarray] = []
                for start in range(0, len(win_chunks), max(1, int(encode_batch_windows))):
                    sub = win_chunks[start : start + max(1, int(encode_batch_windows))]
                    enc_kw: dict[str, Any] = {
                        "dimensions": target_dim,
                        "output_dtype": vec_dtype,
                        "normalize": normalize,
                    }
                    if mode is not None:
                        enc_kw["mode"] = mode
                    win_vecs.extend(engine.encode_contextual_chunks(sub, **enc_kw))

                emit_dicts: list[dict[str, Any]] = []
                for w_idx, (chunks_w, spans_w, vec_mat) in enumerate(
                    zip(win_chunks, win_spans, win_vecs, strict=True)
                ):
                    total_tokens += len(
                        tokenizer.encode_contextual_document(chunks_w, return_spans=False)
                    )
                    vec_np = np.asarray(vec_mat, dtype=np.float16 if dtype == "float16" else np.float32)
                    for c_idx, (c_text, (c0, c1)) in enumerate(zip(chunks_w, spans_w, strict=True)):
                        row_out = {k: r.get(k) for k in keep_names}
                        row_out["window_index"] = int(w_idx)
                        row_out["chunk_index"] = int(c_idx)
                        row_out["char_start"] = int(c0)
                        row_out["char_end"] = int(c1)
                        row_out["chunk_text"] = c_text
                        row_out["vector"] = vec_np[c_idx].tolist()
                        emit_dicts.append(row_out)
                if emit_dicts:
                    rows_emitted += len(emit_dicts)
                    out_tables.append(pa.Table.from_pylist(emit_dicts))
            if limit is not None and rows_read >= int(limit):
                break
        if limit is not None and rows_read >= int(limit):
            break

    out_p = Path(output_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_tables:
        merged = pa.concat_tables(out_tables, promote_options="default")
        vec_idx = merged.schema.get_field_index("vector")
        merged = merged.set_column(vec_idx, "vector", merged.column(vec_idx).cast(pa_vec_type))
    else:
        fields = [f for f in first_schema if f.name in keep_names]
        fields.extend(
            [
                pa.field("window_index", pa.int64()),
                pa.field("chunk_index", pa.int64()),
                pa.field("char_start", pa.int64()),
                pa.field("char_end", pa.int64()),
                pa.field("chunk_text", pa.string()),
                pa.field("vector", pa_vec_type),
            ]
        )
        merged = pa.Table.from_pylist([], schema=pa.schema(fields))

    pq.write_table(merged, out_p, compression=compression)
    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read,
        rows_emitted=rows_emitted,
        total_tokens=total_tokens,
        elapsed_seconds=time.perf_counter() - t0,
        output_path=out_p,
    )


__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "ParquetEmbedSummary",
    "embed_contextual_parquet",
    "embed_parquet",
    "resolve_input_files",
    "resolve_text_column",
    "slice_ocr_pages_utf8",
]
