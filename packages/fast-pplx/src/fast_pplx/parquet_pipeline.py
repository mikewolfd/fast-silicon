"""Streaming sharded Parquet embedding pipeline plus two-knob contextual windowing."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fast_shared.bench import clear_mlx_arena
from fast_shared.embeddings import resolve_embedding_dim
from fast_shared.parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    ParquetEmbedSummary,
    embed_parquet as _shared_embed_parquet,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)

from .config import PplxConfig

_SUPPORTED_VECTOR_DTYPES = ("float32", "float16", "int8")


def _arrow_vector_types(dtype: str, dim: int) -> tuple[pa.DataType, pa.DataType, np.dtype[Any]]:
    if dtype == "int8":
        return pa.int8(), pa.list_(pa.int8(), dim), np.dtype(np.int8)
    if dtype == "float16":
        return pa.float16(), pa.list_(pa.float16(), dim), np.dtype(np.float16)
    if dtype == "float32":
        return pa.float32(), pa.list_(pa.float32(), dim), np.dtype(np.float32)
    raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_SUPPORTED_VECTOR_DTYPES}")


def _make_vector_column(vecs: np.ndarray, *, dim: int, dtype: str) -> pa.Array:
    elem_type, _, np_dt = _arrow_vector_types(dtype, dim)
    mat = np.ascontiguousarray(vecs, dtype=np_dt)
    if mat.ndim == 1:
        mat = mat.reshape(1, dim)
    if mat.shape[1] != dim:
        raise ValueError(f"Vector width mismatch: got {mat.shape[1]}, expected {dim}")
    flat = mat.reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=elem_type), dim)


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
    max_length: int | None = None,
    max_chunk_tokens: int = 512,
    skip_empty: bool = False,
    include_chunk_text: bool = False,
    prefix: str | None = None,
    prompt_name: str | None = None,
    add_query_prefix: bool = False,
    add_document_prefix: bool = False,
    dimensions: int | None = None,
    compression: str = "zstd",
    resume: bool = False,
    limit: int | None = None,
    progress_every: int = 1,
) -> ParquetEmbedSummary:
    """Stream-embed Parquet rows into `float32`, `float16`, or signed `int8` fixed-size vectors."""
    if dtype not in _SUPPORTED_VECTOR_DTYPES:
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_SUPPORTED_VECTOR_DTYPES}")

    if dtype in ("float32", "float16"):
        summary = _shared_embed_parquet(
            engine,
            input_path,
            output_path,
            text_col=text_col,
            keep_cols=keep_cols,
            drop_text_col=drop_text_col,
            mode=mode,
            dtype=dtype,
            batch_rows=batch_rows,
            encode_batch_rows=encode_batch_rows,
            shard_rows=shard_rows,
            filter_ocr=filter_ocr,
            ocr_pages_only=ocr_pages_only,
            per_ocr_page=per_ocr_page,
            chunk=chunk,
            max_body_tokens=max_body_tokens,
            overlap=overlap,
            max_length=max_length,
            max_chunk_tokens=max_chunk_tokens,
            skip_empty=skip_empty,
            include_chunk_text=include_chunk_text,
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
            dimensions=dimensions,
            compression=compression,
            resume=resume,
            limit=limit,
            progress_every=progress_every,
        )
        clear_mlx_arena(256)
        return summary

    # Streaming int8 Parquet path (preserves signed int8 tanh*127 codes via pq.ParquetWriter).
    if int(batch_rows) <= 0 or int(encode_batch_rows) <= 0 or int(shard_rows) <= 0:
        raise ValueError("batch_rows, encode_batch_rows, and shard_rows must be positive")

    t0 = time.perf_counter()
    files = resolve_input_files(input_path)
    first_schema = pq.ParquetFile(files[0]).schema_arrow
    eff_text_col = resolve_text_column(first_schema, text_col)
    max_dim = int(getattr(engine, "max_dim", getattr(engine, "dim", PplxConfig().embedding_dim)))
    default_dim = int(getattr(engine, "dim", max_dim))
    target_dim = resolve_embedding_dim(dimensions, max_dim=max_dim, default_dim=default_dim, supports_dimensions=True)
    _, pa_vec_type, _ = _arrow_vector_types("int8", target_dim)

    out_p = Path(output_path).expanduser().resolve()
    single_file_mode = out_p.suffix.lower() == ".parquet"
    if single_file_mode:
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_p.mkdir(parents=True, exist_ok=True)

    keep_names = [
        c
        for c in (list(first_schema.names) if keep_cols is None else [c for c in keep_cols if c in first_schema.names])
        if not (drop_text_col and c == eff_text_col)
    ]
    read_cols = list(dict.fromkeys([*keep_names, eff_text_col]))
    for aux in ("is_ocr", "ocr_pages", "page_offsets"):
        if aux in first_schema.names and aux not in read_cols:
            read_cols.append(aux)

    tokenizer = engine.tokenizer
    engine_max_len = int(getattr(engine, "max_seq_len", 32768))
    eff_max_len = engine_max_len if max_length is None else min(int(max_length), engine_max_len)
    eff_prefix = tokenizer.resolve_prefix(
        prefix=prefix,
        prompt_name=prompt_name,
        add_query_prefix=add_query_prefix,
        add_document_prefix=add_document_prefix,
    )

    tmp_single = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    single_writer: pq.ParquetWriter | None = None
    rows_read = 0
    rows_emitted = 0
    total_tokens = 0
    shard_idx = 0
    pending_meta: list[pa.Table] = []
    pending_toks: list[list[int]] = []

    def _encode_int8_toks(tok_lists: list[list[int]]) -> np.ndarray:
        out_mat = np.empty((len(tok_lists), target_dim), dtype=np.int8)
        for st in range(0, len(tok_lists), max(1, int(encode_batch_rows))):
            en = min(st + max(1, int(encode_batch_rows)), len(tok_lists))
            kw: dict[str, Any] = {
                "dimensions": target_dim,
                "output_dtype": "int8",
                "normalize": False,
            }
            if mode is not None:
                kw["mode"] = mode
            out_mat[st:en] = np.asarray(engine.encode_corpus(tok_lists[st:en], **kw), dtype=np.int8)
        clear_mlx_arena(256)
        return out_mat

    def _flush_shard(meta_tbl: pa.Table, tok_slice: list[list[int]]) -> None:
        nonlocal shard_idx, rows_emitted, total_tokens
        shard_path = out_p / f"part-{shard_idx:05d}.parquet"
        tmp_shard = out_p / f".part-{shard_idx:05d}.parquet.tmp"
        shard_idx += 1
        if resume and shard_path.exists():
            try:
                if pq.ParquetFile(shard_path).metadata.num_rows == len(tok_slice):
                    return
            except Exception:
                pass
        vecs = _encode_int8_toks(tok_slice)
        out_tbl = meta_tbl.append_column("vector", _make_vector_column(vecs, dim=target_dim, dtype="int8"))
        try:
            with pq.ParquetWriter(tmp_shard, out_tbl.schema, compression=compression) as writer:
                writer.write_table(out_tbl)
            os.replace(tmp_shard, shard_path)
        except BaseException:
            tmp_shard.unlink(missing_ok=True)
            raise
        rows_emitted += len(tok_slice)
        total_tokens += sum(len(t) for t in tok_slice)

    def _drain_pending(final: bool = False) -> None:
        nonlocal pending_meta, pending_toks
        if not pending_toks:
            return
        if not final and len(pending_toks) < int(shard_rows):
            return
        merged_meta = pa.concat_tables(pending_meta, promote_options="default") if len(pending_meta) > 1 else pending_meta[0]
        cur_meta = merged_meta
        cur_toks = pending_toks
        pending_meta = []
        pending_toks = []
        while len(cur_toks) >= int(shard_rows):
            _flush_shard(cur_meta.slice(0, int(shard_rows)), cur_toks[: int(shard_rows)])
            rem = len(cur_toks) - int(shard_rows)
            cur_meta = cur_meta.slice(int(shard_rows), rem) if rem > 0 else cur_meta.slice(0, 0)
            cur_toks = cur_toks[int(shard_rows) :]
        if cur_toks:
            if final:
                _flush_shard(cur_meta, cur_toks)
            else:
                pending_meta = [cur_meta]
                pending_toks = cur_toks

    try:
        for fpath in files:
            pf = pq.ParquetFile(fpath)
            avail = [c for c in read_cols if c in pf.schema_arrow.names]
            for rb in pf.iter_batches(batch_size=max(1, int(batch_rows)), columns=avail):
                if limit is not None and rows_read >= int(limit):
                    break
                if limit is not None and rows_read + rb.num_rows > int(limit):
                    rb = rb.slice(0, max(0, int(limit) - rows_read))
                rows_read += rb.num_rows
                py_rows = rb.to_pylist()
                keep_idx: list[int] = []
                texts: list[str] = []
                for idx_r, r in enumerate(py_rows):
                    if filter_ocr:
                        is_ocr = bool(r.get("is_ocr"))
                        ocr_p = r.get("ocr_pages")
                        if not (is_ocr or (ocr_p and len(ocr_p) > 0)):
                            continue
                    txt = r.get(eff_text_col)
                    if ocr_pages_only:
                        txt = slice_ocr_pages_utf8(txt, r.get("page_offsets"), r.get("ocr_pages"))
                    txt_s = txt if isinstance(txt, str) else ""
                    if skip_empty and not txt_s.strip():
                        continue
                    keep_idx.append(idx_r)
                    texts.append(txt_s)
                if not keep_idx:
                    continue
                sub_rb = rb.take(keep_idx)
                meta_tbl = pa.Table.from_batches([sub_rb]).select([c for c in keep_names if c in sub_rb.schema.names])
                tok_lists = tokenizer.encode_batch(texts, prefix=eff_prefix, max_length=eff_max_len)
                if single_file_mode:
                    vecs = _encode_int8_toks(tok_lists)
                    out_tbl = meta_tbl.append_column("vector", _make_vector_column(vecs, dim=target_dim, dtype="int8"))
                    if single_writer is None:
                        single_writer = pq.ParquetWriter(tmp_single, out_tbl.schema, compression=compression)
                    single_writer.write_table(out_tbl)
                    rows_emitted += len(tok_lists)
                    total_tokens += sum(len(t) for t in tok_lists)
                else:
                    pending_meta.append(meta_tbl)
                    pending_toks.extend(tok_lists)
                    _drain_pending(final=False)
            if limit is not None and rows_read >= int(limit):
                break

        if not single_file_mode:
            _drain_pending(final=True)
        else:
            if single_writer is not None:
                single_writer.close()
                single_writer = None
            if tmp_single is not None and tmp_single.exists():
                os.replace(tmp_single, out_p)
            else:
                empty_schema = pa.schema(
                    [f for f in first_schema if f.name in keep_names] + [pa.field("vector", pa_vec_type)]
                )
                pq.write_table(pa.Table.from_pylist([], schema=empty_schema), out_p, compression=compression)
    finally:
        if single_writer is not None:
            single_writer.close()
        if tmp_single is not None and tmp_single.exists():
            tmp_single.unlink(missing_ok=True)

    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read,
        rows_emitted=rows_emitted,
        total_tokens=total_tokens,
        elapsed_seconds=time.perf_counter() - t0,
        output_path=out_p,
    )


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
    encode_batch_rows: int | None = None,
    shard_rows: int = 65536,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    skip_empty: bool = True,
    include_chunk_text: bool = True,
    prefix: str | None = None,
    prompt_name: str | None = None,
    add_query_prefix: bool = False,
    add_document_prefix: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    compression: str = "zstd",
    resume: bool = False,
    limit: int | None = None,
    progress_every: int = 0,
) -> ParquetEmbedSummary:
    """Split each document into `<= chunk_tokens` spans, pack into `<= context_tokens`
    windows, and stream-write `float32`/`float16`/`int8` vectors incrementally via `pq.ParquetWriter`.
    """
    if int(chunk_tokens) <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if int(context_tokens) <= 0:
        raise ValueError(f"context_tokens must be positive, got {context_tokens}")
    if int(shard_rows) <= 0:
        raise ValueError(f"shard_rows must be positive, got {shard_rows}")
    if dtype not in _SUPPORTED_VECTOR_DTYPES:
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_SUPPORTED_VECTOR_DTYPES}")

    t0 = time.perf_counter()
    files = resolve_input_files(input_path)
    first_pf = pq.ParquetFile(files[0])
    first_schema = first_pf.schema_arrow
    eff_text_col = resolve_text_column(first_schema, text_col)

    cols_to_read = list(first_schema.names if keep_cols is None else keep_cols)
    if eff_text_col not in cols_to_read:
        cols_to_read.append(eff_text_col)
    if ocr_pages_only or filter_ocr:
        for req_col in ("is_ocr", "page_offsets", "ocr_pages"):
            if req_col in first_schema.names and req_col not in cols_to_read:
                cols_to_read.append(req_col)

    max_dim = int(getattr(engine, "max_dim", getattr(engine, "dim", PplxConfig().embedding_dim)))
    default_dim = int(getattr(engine, "dim", max_dim))
    target_dim = resolve_embedding_dim(dimensions, max_dim=max_dim, default_dim=default_dim, supports_dimensions=True)
    _, pa_vec_type, _ = _arrow_vector_types(dtype, target_dim)

    keep_names = [
        c
        for c in (list(first_schema.names) if keep_cols is None else [c for c in keep_cols if c in first_schema.names])
        if not (drop_text_col and c == eff_text_col)
    ]
    keep_fields = [f for f in first_schema if f.name in keep_names]
    out_fields: list[pa.Field] = [
        *keep_fields,
        pa.field("window_index", pa.int64()),
        pa.field("chunk_index", pa.int64()),
        pa.field("char_start", pa.int64()),
        pa.field("char_end", pa.int64()),
    ]
    if include_chunk_text:
        out_fields.append(pa.field("chunk_text", pa.string()))
    out_fields.append(pa.field("vector", pa_vec_type))
    out_schema = pa.schema(out_fields)

    out_p = Path(output_path).expanduser().resolve()
    single_file_mode = out_p.suffix.lower() == ".parquet"
    if single_file_mode:
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_p.mkdir(parents=True, exist_ok=True)

    tokenizer = engine.tokenizer
    eff_prefix = ""
    if prefix is not None or prompt_name is not None or add_query_prefix or add_document_prefix:
        eff_prefix = tokenizer.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
        )

    eff_window_batch = max(1, int(encode_batch_windows))
    if encode_batch_rows is not None and int(encode_batch_rows) > 0:
        # Honor CLI --encode-batch-rows as a chunk budget per contextual wave when tighter.
        chunks_per_win_est = max(1, int(context_tokens) // max(1, int(chunk_tokens)))
        eff_window_batch = max(1, min(eff_window_batch, max(1, int(encode_batch_rows) // chunks_per_win_est)))

    eff_normalize = False if dtype == "int8" else bool(normalize)

    tmp_single = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    single_writer: pq.ParquetWriter | None = None
    shard_idx = 0
    rows_read = 0
    rows_emitted = 0
    total_tokens = 0
    wave_num = 0

    # Pending prepared windows (bounded to <= shard_rows in sharded mode or flushed per batch in single-file mode).
    # Each entry: (meta_row_dict, w_idx, chunks_w, spans_w)
    pending_windows: list[tuple[dict[str, Any], int, list[str], list[tuple[int, int]]]] = []
    pending_chunk_count = 0

    def _build_wave_table(
        entries: Sequence[tuple[dict[str, Any], int, list[str], list[tuple[int, int]]]],
        vec_mats: Sequence[np.ndarray],
    ) -> pa.Table:
        col_data: dict[str, list[Any]] = {k: [] for k in keep_names}
        win_indices: list[int] = []
        chunk_indices: list[int] = []
        char_starts: list[int] = []
        char_ends: list[int] = []
        chunk_texts: list[str] = []
        vec_blocks: list[np.ndarray] = []

        _, _, np_dt = _arrow_vector_types(dtype, target_dim)
        for (meta_row, w_idx, chunks_w, spans_w), vec_mat in zip(entries, vec_mats, strict=True):
            vec_np = np.asarray(vec_mat, dtype=np_dt)
            if vec_np.ndim == 1:
                vec_np = vec_np.reshape(1, target_dim)
            vec_blocks.append(vec_np[: len(chunks_w)])
            for c_idx, (c_text, (c0, c1)) in enumerate(zip(chunks_w, spans_w, strict=True)):
                for k in keep_names:
                    col_data[k].append(meta_row.get(k))
                win_indices.append(int(w_idx))
                chunk_indices.append(int(c_idx))
                char_starts.append(int(c0))
                char_ends.append(int(c1))
                if include_chunk_text:
                    chunk_texts.append(c_text)

        arrays: list[pa.Array] = []
        for f in keep_fields:
            arrays.append(pa.array(col_data[f.name], type=f.type))
        arrays.append(pa.array(win_indices, type=pa.int64()))
        arrays.append(pa.array(chunk_indices, type=pa.int64()))
        arrays.append(pa.array(char_starts, type=pa.int64()))
        arrays.append(pa.array(char_ends, type=pa.int64()))
        if include_chunk_text:
            arrays.append(pa.array(chunk_texts, type=pa.string()))
        stacked = np.concatenate(vec_blocks, axis=0) if vec_blocks else np.zeros((0, target_dim), dtype=np_dt)
        arrays.append(_make_vector_column(stacked, dim=target_dim, dtype=dtype))
        return pa.Table.from_arrays(arrays, schema=out_schema)

    def _encode_entries(
        entries: Sequence[tuple[dict[str, Any], int, list[str], list[tuple[int, int]]]],
    ) -> tuple[list[np.ndarray], int]:
        win_chunks = [e[2] for e in entries]
        win_vecs: list[np.ndarray] = []
        tok_count = 0
        encode_ctx_toks_fn = getattr(engine, "encode_contextual_tokens", None)
        eff_max_len = getattr(engine, "max_seq_len", None)
        for st in range(0, len(win_chunks), eff_window_batch):
            sub = win_chunks[st : st + eff_window_batch]
            enc_kw: dict[str, Any] = {
                "dimensions": target_dim,
                "output_dtype": dtype,
                "normalize": eff_normalize,
            }
            if mode is not None:
                enc_kw["mode"] = mode
            if callable(encode_ctx_toks_fn):
                encoded_sub = [
                    tokenizer.encode_contextual_document(chunks_w, return_spans=True, max_length=eff_max_len)
                    for chunks_w in sub
                ]
                tok_count += sum(len(ids) for ids, _ in encoded_sub)
                win_vecs.extend(encode_ctx_toks_fn(encoded_sub, **enc_kw))
            else:
                win_vecs.extend(engine.encode_contextual_chunks(sub, **enc_kw))
                tok_count += sum(len(chunks_w) for chunks_w in sub)
        clear_mlx_arena(256)
        return win_vecs, tok_count

    def _flush_wave(entries: list[tuple[dict[str, Any], int, list[str], list[tuple[int, int]]]]) -> None:
        nonlocal single_writer, shard_idx, rows_emitted, total_tokens, wave_num
        if not entries:
            return
        n_chunks_wave = sum(len(e[2]) for e in entries)
        wave_num += 1

        if single_file_mode:
            win_vecs, tok_cnt = _encode_entries(entries)
            tbl = _build_wave_table(entries, win_vecs)
            if single_writer is None:
                single_writer = pq.ParquetWriter(tmp_single, out_schema, compression=compression)
            single_writer.write_table(tbl)
            rows_emitted += n_chunks_wave
            total_tokens += tok_cnt
        else:
            shard_path = out_p / f"part-{shard_idx:05d}.parquet"
            tmp_shard = out_p / f".part-{shard_idx:05d}.parquet.tmp"
            shard_idx += 1
            if resume and shard_path.exists():
                try:
                    if pq.ParquetFile(shard_path).metadata.num_rows == n_chunks_wave:
                        return
                except Exception:
                    pass
            try:
                with pq.ParquetWriter(tmp_shard, out_schema, compression=compression) as writer:
                    for st in range(0, len(entries), eff_window_batch):
                        sub_entries = entries[st : st + eff_window_batch]
                        win_vecs, tok_cnt = _encode_entries(sub_entries)
                        writer.write_table(_build_wave_table(sub_entries, win_vecs))
                        rows_emitted += sum(len(e[2]) for e in sub_entries)
                        total_tokens += tok_cnt
                os.replace(tmp_shard, shard_path)
            except BaseException:
                tmp_shard.unlink(missing_ok=True)
                raise

        if progress_every > 0 and (wave_num % progress_every == 0):
            dt = time.perf_counter() - t0
            print(
                f"[contextual parquet wave {wave_num}] chunks={rows_emitted:,} toks={total_tokens:,} "
                f"elapsed={dt:.2f}s ({total_tokens / max(dt, 1e-9):,.0f} tok/s)",
                flush=True,
            )

    try:
        for fpath in files:
            pf = pq.ParquetFile(fpath)
            avail_cols = [c for c in cols_to_read if c in pf.schema_arrow.names]
            for rb in pf.iter_batches(batch_size=max(1, int(batch_rows)), columns=avail_cols):
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
                    if not isinstance(raw_text, str):
                        if skip_empty:
                            continue
                        raw_text = ""
                    if not raw_text.strip():
                        if skip_empty:
                            continue
                        grouped = [([""], [(0, 0)])]
                    else:
                        doc_input = (eff_prefix + raw_text) if eff_prefix else raw_text
                        grouped = tokenizer.chunk_and_group_document(
                            doc_input,
                            chunk_tokens=int(chunk_tokens),
                            context_tokens=int(context_tokens),
                            overlap=int(overlap),
                        )
                        if not grouped:
                            continue

                    meta_row = {k: r.get(k) for k in keep_names}
                    for w_idx, (chunks_w, spans_w) in enumerate(grouped):
                        pending_windows.append((meta_row, w_idx, chunks_w, spans_w))
                        pending_chunk_count += len(chunks_w)
                        if single_file_mode and len(pending_windows) >= eff_window_batch:
                            _flush_wave(pending_windows)
                            pending_windows = []
                            pending_chunk_count = 0
                        elif not single_file_mode and pending_chunk_count >= int(shard_rows):
                            _flush_wave(pending_windows)
                            pending_windows = []
                            pending_chunk_count = 0
                if limit is not None and rows_read >= int(limit):
                    break
            if limit is not None and rows_read >= int(limit):
                break

        if pending_windows:
            _flush_wave(pending_windows)
            pending_windows = []
            pending_chunk_count = 0

        if single_file_mode:
            if single_writer is not None:
                single_writer.close()
                single_writer = None
            if tmp_single is not None and tmp_single.exists():
                os.replace(tmp_single, out_p)
            else:
                pq.write_table(pa.Table.from_pylist([], schema=out_schema), out_p, compression=compression)
    finally:
        if single_writer is not None:
            single_writer.close()
        if tmp_single is not None and tmp_single.exists():
            tmp_single.unlink(missing_ok=True)

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
