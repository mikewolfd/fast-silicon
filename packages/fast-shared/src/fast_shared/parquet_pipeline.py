"""Streaming sharded Parquet embedding pipeline optimized for Apple Silicon."""

from __future__ import annotations

import glob
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .embeddings import accepts_kwarg, resolve_embedding_dim, truncate_and_normalize

DEFAULT_TEXT_COLUMNS = ("body", "page_text", "text", "content")


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
) -> Iterable[pa.RecordBatch]:
    emitted = 0
    for fpath in files:
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


def _make_vector_array(vecs_f32: np.ndarray, dim: int, dtype: str) -> pa.Array:
    if dtype == "float16":
        flat = vecs_f32.astype(np.float16).reshape(-1)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float16()), dim)
    flat = np.ascontiguousarray(vecs_f32, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dim)


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

    expanded_row_idx: list[int] = []
    expanded_tok_lists: list[list[int]] = []
    page_numbers: list[int] = []
    chunk_indices: list[int] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    tok_counts: list[int] = []
    chunk_texts: list[str] = []

    meta_only_cols = [c for c in meta_cols if c != "page_offsets"]
    for i in range(rb.num_rows):
        t = raw_texts[i] if isinstance(raw_texts[i], str) else ""
        units: list[tuple[int, str]] = []
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
                        units.append((int(pnum), seg))
        elif ocr_pages_only and offsets_col is not None and ocr_pages_col is not None:
            seg = slice_ocr_pages_utf8(t, offsets_col[i], ocr_pages_col[i])
            if seg or not skip_empty:
                units.append((-1, seg))
        else:
            if t.strip() or not skip_empty:
                units.append((-1, t))

        c_idx = 0
        for pnum, unit_text in units:
            if chunk:
                win_list = tokenizer.chunk_text(
                    unit_text,
                    max_body_tokens=max_body_tokens,
                    overlap=overlap,
                    prefix=eff_prefix,
                    max_chunk_tokens=max_chunk_tokens,
                )
                for win_ids, c0, c1 in win_list:
                    expanded_row_idx.append(i)
                    expanded_tok_lists.append(win_ids)
                    if per_ocr_page:
                        page_numbers.append(pnum)
                    chunk_indices.append(c_idx)
                    char_starts.append(c0)
                    char_ends.append(c1)
                    tok_counts.append(len(win_ids))
                    if include_chunk_text:
                        chunk_texts.append(unit_text[c0:c1])
                    c_idx += 1
            else:
                ids = tokenizer.encode(unit_text, prefix=eff_prefix, max_length=max_encode_length)
                expanded_row_idx.append(i)
                expanded_tok_lists.append(ids)
                if per_ocr_page:
                    page_numbers.append(pnum)
                chunk_indices.append(c_idx)
                char_starts.append(0)
                char_ends.append(len(unit_text))
                tok_counts.append(len(ids))
                if include_chunk_text:
                    chunk_texts.append(unit_text)
                c_idx += 1

    if not expanded_row_idx:
        empty = pa.Table.from_batches([rb.slice(0, 0)]).select(meta_only_cols)
        return empty, []

    meta_table = pa.Table.from_batches([rb.take(expanded_row_idx)]).select(meta_only_cols)
    if per_ocr_page:
        meta_table = meta_table.append_column("ocr_page_number", pa.array(page_numbers, type=pa.int32()))
    if chunk:
        meta_table = meta_table.append_column("chunk_index", pa.array(chunk_indices, type=pa.int32()))
        meta_table = meta_table.append_column("char_start", pa.array(char_starts, type=pa.int32()))
        meta_table = meta_table.append_column("char_end", pa.array(char_ends, type=pa.int32()))
        meta_table = meta_table.append_column("token_count", pa.array(tok_counts, type=pa.int32()))
    if include_chunk_text:
        meta_table = meta_table.append_column("chunk_text", pa.array(chunk_texts, type=pa.string()))
    return meta_table, expanded_tok_lists


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
    """Prepare source batches, encode capped chunk batches, and write Parquet.

    batch_rows limits source rows read at once; encode_batch_rows limits each
    model call after chunk expansion; shard_rows controls stored output rows.
    Source record size and chunk expansion still affect preparation memory.
    """
    if dtype not in ("float32", "float16"):
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected 'float32' or 'float16'")
    for name, value in (("batch_rows", batch_rows), ("encode_batch_rows", encode_batch_rows), ("shard_rows", shard_rows)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    default_dim = int(getattr(engine, "dim"))
    max_dim = int(getattr(engine, "max_dim", default_dim))
    supports_dims = bool(getattr(engine, "supports_dimensions", False))
    dim = resolve_embedding_dim(
        dimensions,
        max_dim=max_dim,
        default_dim=default_dim,
        supports_dimensions=supports_dims,
    )
    corpus_kw: dict[str, Any] = {"mode": mode}
    if dimensions is not None and accepts_kwarg(engine.encode_corpus, "dimensions"):
        corpus_kw["dimensions"] = dim
    engine_max_length = int(getattr(engine, "max_seq_len", 512))
    max_encode_length = int(max_length if max_length is not None else engine_max_length)
    if max_encode_length <= 0 or max_encode_length > engine_max_length:
        raise ValueError(f"max_length must be positive and no greater than the engine limit ({engine_max_length})")
    if max_chunk_tokens <= 0 or (chunk and max_chunk_tokens > engine_max_length):
        raise ValueError(f"max_chunk_tokens must be positive and no greater than the engine limit ({engine_max_length}) when chunking")
    files = resolve_input_files(input_path)
    first_schema = pq.ParquetFile(files[0]).schema_arrow
    eff_text_col = resolve_text_column(first_schema, text_col)

    out_p = Path(output_path).expanduser().resolve()
    single_file_mode = out_p.suffix.lower() == ".parquet"
    if single_file_mode:
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_p.mkdir(parents=True, exist_ok=True)

    if keep_cols is not None:
        needed = list(dict.fromkeys([*keep_cols, eff_text_col]))
    else:
        needed = list(first_schema.names)
    for aux in ("is_ocr", "ocr_pages", "page_offsets"):
        if aux in first_schema.names and aux not in needed:
            needed.append(aux)

    prep_queue: queue.Queue[tuple[int, pa.Table, list[list[int]]] | None | BaseException] = queue.Queue(maxsize=2)
    rows_read_box = [0]

    def producer() -> None:
        try:
            for rb in _iter_source_batches(
                files,
                needed,
                batch_rows,
                filter_ocr=filter_ocr,
                limit=limit,
            ):
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
                    max_encode_length=max_encode_length,
                    max_chunk_tokens=max_chunk_tokens,
                    skip_empty=skip_empty,
                    include_chunk_text=include_chunk_text,
                    prefix=prefix,
                    prompt_name=prompt_name,
                    add_query_prefix=add_query_prefix,
                    add_document_prefix=add_document_prefix,
                )
                if len(tok_lists) > 0:
                    prep_queue.put((rb.num_rows, meta_tbl, tok_lists))
        except BaseException as exc:
            prep_queue.put(exc)
        finally:
            prep_queue.put(None)

    t0 = time.perf_counter()
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    single_writer: pq.ParquetWriter | None = None
    tmp_single_path = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    shard_idx = 0
    pending_meta: pa.Table | None = None
    pending_toks: list[list[int]] = []

    rows_emitted = 0
    total_tokens = 0
    batch_num = 0

    write_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=2)
    writer_err: list[BaseException] = []
    writer_stop = threading.Event()

    def _writer_worker() -> None:
        nonlocal single_writer
        try:
            while not writer_stop.is_set():
                try:
                    w_item = write_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if w_item is None:
                    break
                kind = w_item[0]
                if kind == "single":
                    _, m_tbl, vecs = w_item
                    vec_arr = _make_vector_array(vecs, dim=dim, dtype=dtype)
                    out_tbl = m_tbl.append_column("vector", vec_arr)
                    if single_writer is None:
                        single_writer = pq.ParquetWriter(tmp_single_path, out_tbl.schema, compression=compression)
                    single_writer.write_table(out_tbl)
                else:
                    _, m_slice, vecs, shard_path, tmp_shard = w_item
                    vec_arr = _make_vector_array(vecs, dim=dim, dtype=dtype)
                    out_tbl = m_slice.append_column("vector", vec_arr)
                    try:
                        pq.write_table(out_tbl, tmp_shard, compression=compression)
                        os.replace(tmp_shard, shard_path)
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
    writer_thread.start()

    def _enqueue_write(job: tuple | None) -> None:
        while True:
            if writer_err:
                raise writer_err[0]
            try:
                write_queue.put(job, timeout=0.05)
                return
            except queue.Full:
                continue

    def encode_rows(tok_lists: list[list[int]]) -> np.ndarray:
        if len(tok_lists) <= encode_batch_rows:
            return truncate_and_normalize(engine.encode_corpus(tok_lists, **corpus_kw), dim)
        vecs = np.empty((len(tok_lists), dim), dtype=np.float32)
        for start in range(0, len(tok_lists), encode_batch_rows):
            end = min(start + encode_batch_rows, len(tok_lists))
            vecs[start:end] = truncate_and_normalize(engine.encode_corpus(tok_lists[start:end], **corpus_kw), dim)
        return vecs

    def write_shard_slice(meta_slice: pa.Table, tok_slice: list[list[int]]) -> None:
        nonlocal shard_idx
        shard_path = out_p / f"part-{shard_idx:05d}.parquet"
        tmp_shard = out_p / f".part-{shard_idx:05d}.parquet.tmp"
        shard_idx += 1
        if resume and shard_path.exists():
            try:
                if pq.ParquetFile(shard_path).metadata.num_rows == len(tok_slice):
                    return
            except Exception:
                pass
        vecs = encode_rows(tok_slice)
        _enqueue_write(("shard", meta_slice, vecs, shard_path, tmp_shard))

    def drain_shards(final: bool = False) -> None:
        nonlocal pending_meta, pending_toks
        while pending_meta is not None and len(pending_toks) >= shard_rows:
            slice_meta = pending_meta.slice(0, shard_rows)
            slice_toks = pending_toks[:shard_rows]
            rem = len(pending_toks) - shard_rows
            pending_meta = pending_meta.slice(shard_rows, rem) if rem > 0 else None
            pending_toks = pending_toks[shard_rows:]
            write_shard_slice(slice_meta, slice_toks)
        if final and pending_meta is not None and len(pending_toks) > 0:
            write_shard_slice(pending_meta, pending_toks)
            pending_meta = None
            pending_toks = []

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
            n_items = len(tok_lists)
            n_toks = sum(len(x) for x in tok_lists)
            total_tokens += n_toks
            rows_emitted += n_items
            batch_num += 1

            if single_file_mode:
                vecs = encode_rows(tok_lists)
                _enqueue_write(("single", meta_tbl, vecs))
            else:
                if pending_meta is None:
                    pending_meta = meta_tbl
                else:
                    pending_meta = pa.concat_tables([pending_meta, meta_tbl], promote_options="default")
                pending_toks.extend(tok_lists)
                drain_shards(final=False)

            if progress_every > 0 and (batch_num % progress_every == 0):
                dt = time.perf_counter() - t0
                print(
                    f"[parquet batch {batch_num}] rows={rows_emitted:,} toks={total_tokens:,} "
                    f"elapsed={dt:.2f}s ({total_tokens/dt:,.0f} tok/s)",
                    flush=True,
                )

        if not single_file_mode:
            drain_shards(final=True)

        _enqueue_write(None)
        writer_thread.join()
        if writer_err:
            raise writer_err[0]

        if single_file_mode:
            if rows_emitted > 0 and tmp_single_path is not None and tmp_single_path.exists():
                os.replace(tmp_single_path, out_p)
            else:
                empty_schema = pa.schema(
                    [f for f in first_schema if f.name != eff_text_col or not drop_text_col]
                    + [pa.field("vector", pa.list_(pa.float16() if dtype == "float16" else pa.float32(), dim))]
                )
                pq.write_table(pa.Table.from_pylist([], schema=empty_schema), out_p, compression=compression)
    finally:
        writer_stop.set()
        try:
            write_queue.put_nowait(None)
        except queue.Full:
            pass
        writer_thread.join(timeout=2.0)
        if tmp_single_path is not None and tmp_single_path.exists():
            tmp_single_path.unlink(missing_ok=True)
        prod_thread.join(timeout=2.0)

    elapsed = time.perf_counter() - t0
    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read_box[0],
        rows_emitted=rows_emitted,
        total_tokens=total_tokens,
        elapsed_seconds=elapsed,
        output_path=out_p,
    )
