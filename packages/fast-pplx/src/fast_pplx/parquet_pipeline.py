"""Streaming sharded Parquet embedding pipeline plus two-knob contextual windowing."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterator, Sequence, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fast_shared.bench import clear_mlx_arena
from fast_shared.embeddings import accepts_kwarg, resolve_embedding_dim
from fast_shared.parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    ParquetEmbedSummary,
    _decode_temp_dict_columns,
    _iter_source_batches,
    _prepare_batch_rows,
    embed_parquet as _shared_embed_parquet,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)

from .config import PplxConfig
from .tokenizer import ContextualTokenizer, ContextualWindow

_SUPPORTED_VECTOR_DTYPES = ("float32", "float16", "int8")
_T = TypeVar("_T")


def _prefetch_iter(it: Iterator[_T], *, max_prefetch: int = 2) -> Iterator[_T]:
    """Yield items from a synchronous producer iterator with bounded lookahead."""
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=max(1, int(max_prefetch)))
    stop_event = threading.Event()

    def _producer() -> None:
        try:
            for item in it:
                while not stop_event.is_set():
                    try:
                        q.put((True, item), timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop_event.is_set():
                    return
            while not stop_event.is_set():
                try:
                    q.put((False, None), timeout=0.05)
                    return
                except queue.Full:
                    continue
        except BaseException as exc:
            while not stop_event.is_set():
                try:
                    q.put((False, exc), timeout=0.05)
                    return
                except queue.Full:
                    continue

    worker = threading.Thread(target=_producer, daemon=True)
    worker.start()
    try:
        while True:
            ok, payload = q.get()
            if ok:
                yield payload
                continue
            if payload is not None:
                raise payload
            return
    finally:
        stop_event.set()
        worker.join()


class _AsyncOrderedWriter:
    """Single-worker ordered background Parquet writer with bounded backpressure."""

    def __init__(self, *, max_in_flight: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pplx-pq-write")
        self._pending: deque[Future[tuple[int, int]]] = deque()
        self._max_in_flight = max(1, int(max_in_flight))
        self.rows_written = 0
        self.tokens_written = 0

    def submit(self, fn: Callable[..., tuple[int, int]], *args: Any) -> None:
        while self._pending and (self._pending[0].done() or len(self._pending) >= self._max_in_flight):
            r, t = self._pending.popleft().result()
            self.rows_written += int(r)
            self.tokens_written += int(t)
        self._pending.append(self._pool.submit(fn, *args))

    def drain(self) -> tuple[int, int]:
        while self._pending:
            r, t = self._pending.popleft().result()
            self.rows_written += int(r)
            self.tokens_written += int(t)
        return self.rows_written, self.tokens_written

    def abort_and_close(self) -> None:
        for fut in self._pending:
            fut.cancel()
        self._pool.shutdown(wait=True)
        self._pending.clear()

    def close(self) -> None:
        self._pool.shutdown(wait=True)


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


class _DocPrefixSpanEngineProxy:
    """Delegate engine calls while stripping leading `[D] ` tokens from pooling."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def encode_corpus(self, *args: Any, **kwargs: Any) -> np.ndarray:
        kwargs.setdefault("exclude_document_prefix", True)
        return self._engine.encode_corpus(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


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
    add_document_prefix: bool | None = None,
    header_col: str | None = None,
    header_text: str | None = None,
    dimensions: int | None = None,
    compression: str = "zstd",
    resume: bool = False,
    limit: int | None = None,
    progress_every: int = 1,
) -> ParquetEmbedSummary:
    """Stream-embed Parquet rows into `float32`, `float16`, or signed `int8` fixed-size vectors."""
    if dtype not in _SUPPORTED_VECTOR_DTYPES:
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_SUPPORTED_VECTOR_DTYPES}")

    eff_add_doc_prefix = (
        (not add_query_prefix and prefix is None and prompt_name is None)
        if add_document_prefix is None
        else bool(add_document_prefix)
    )
    use_doc_span = bool(eff_add_doc_prefix and accepts_kwarg(engine.encode_corpus, "exclude_document_prefix"))

    if dtype in ("float32", "float16"):
        eff_engine = _DocPrefixSpanEngineProxy(engine) if use_doc_span else engine
        summary = _shared_embed_parquet(
            eff_engine,
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
            add_document_prefix=eff_add_doc_prefix,
            header_col=header_col,
            header_text=header_text,
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
    if header_col is not None and header_col not in first_schema.names:
        raise KeyError(f"Requested header column {header_col!r} not in schema columns: {first_schema.names}")
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
    read_cols_raw = [*keep_names, eff_text_col]
    if header_col is not None:
        read_cols_raw.append(header_col)
    read_cols = list(dict.fromkeys(read_cols_raw))
    for aux in ("is_ocr", "ocr_pages", "page_offsets"):
        if aux in first_schema.names and aux not in read_cols:
            read_cols.append(aux)

    tokenizer = engine.tokenizer
    engine_max_len = int(getattr(engine, "max_seq_len", 32768))
    eff_max_len = engine_max_len if max_length is None else min(int(max_length), engine_max_len)

    tmp_single = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    single_writer: pq.ParquetWriter | None = None
    rows_read_box = [0]
    rows_emitted = 0
    total_tokens = 0
    shard_idx = 0
    pending_meta: list[pa.Table] = []
    pending_toks: list[list[int]] = []
    async_writer = _AsyncOrderedWriter(max_in_flight=2)

    def _encode_int8_toks(tok_lists: list[list[int]]) -> np.ndarray:
        out_mat = np.empty((len(tok_lists), target_dim), dtype=np.int8)
        for st in range(0, len(tok_lists), max(1, int(encode_batch_rows))):
            en = min(st + max(1, int(encode_batch_rows)), len(tok_lists))
            kw: dict[str, Any] = {
                "dimensions": target_dim,
                "output_dtype": "int8",
                "normalize": False,
            }
            if use_doc_span:
                kw["exclude_document_prefix"] = True
            if mode is not None:
                kw["mode"] = mode
            out_mat[st:en] = np.asarray(engine.encode_corpus(tok_lists[st:en], **kw), dtype=np.int8)
        clear_mlx_arena(256)
        return out_mat

    def _write_shard_sync(
        meta_tbl: pa.Table,
        vecs: np.ndarray,
        tok_count: int,
        shard_path: Path,
        tmp_shard: Path,
    ) -> tuple[int, int]:
        out_tbl = _decode_temp_dict_columns(meta_tbl).append_column(
            "vector", _make_vector_column(vecs, dim=target_dim, dtype="int8")
        )
        try:
            with pq.ParquetWriter(tmp_shard, out_tbl.schema, compression=compression) as writer:
                writer.write_table(out_tbl)
            os.replace(tmp_shard, shard_path)
        except BaseException:
            tmp_shard.unlink(missing_ok=True)
            raise
        return out_tbl.num_rows, tok_count

    def _write_single_sync(meta_tbl: pa.Table, vecs: np.ndarray, tok_count: int) -> tuple[int, int]:
        nonlocal single_writer
        out_tbl = _decode_temp_dict_columns(meta_tbl).append_column(
            "vector", _make_vector_column(vecs, dim=target_dim, dtype="int8")
        )
        if single_writer is None:
            single_writer = pq.ParquetWriter(tmp_single, out_tbl.schema, compression=compression)
        single_writer.write_table(out_tbl)
        return out_tbl.num_rows, tok_count

    def _close_single_sync() -> tuple[int, int]:
        nonlocal single_writer
        if single_writer is not None:
            single_writer.close()
            single_writer = None
        return 0, 0

    def _flush_shard(meta_tbl: pa.Table, tok_slice: list[list[int]]) -> None:
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
        vecs = _encode_int8_toks(tok_slice)
        tok_cnt = sum(len(t) for t in tok_slice)
        async_writer.submit(_write_shard_sync, meta_tbl, vecs, tok_cnt, shard_path, tmp_shard)

    def _drain_pending(final: bool = False) -> None:
        nonlocal pending_meta, pending_toks
        if not pending_toks:
            return
        if not final and len(pending_toks) < int(shard_rows):
            return
        merged_meta = pa.concat_tables([_decode_temp_dict_columns(t) for t in pending_meta], promote_options="default") if len(pending_meta) > 1 else _decode_temp_dict_columns(pending_meta[0])
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

    def _iter_prepared_batches() -> Iterator[tuple[pa.Table, list[list[int]]]]:
        for rb in _iter_source_batches(
            files,
            read_cols,
            max(1, int(batch_rows)),
            filter_ocr=filter_ocr,
            limit=limit,
        ):
            rows_read_box[0] += rb.num_rows
            meta_tbl, tok_lists = _prepare_batch_rows(
                rb,
                tokenizer=tokenizer,
                text_col=eff_text_col,
                keep_cols=keep_cols,
                drop_text_col=drop_text_col,
                ocr_pages_only=ocr_pages_only,
                per_ocr_page=per_ocr_page,
                chunk=chunk,
                max_body_tokens=max_body_tokens,
                overlap=overlap,
                max_encode_length=eff_max_len,
                max_chunk_tokens=max_chunk_tokens,
                skip_empty=skip_empty,
                include_chunk_text=include_chunk_text,
                prefix=prefix,
                prompt_name=prompt_name,
                add_query_prefix=add_query_prefix,
                add_document_prefix=eff_add_doc_prefix,
                header_col=header_col,
                header_text=header_text,
            )
            if tok_lists:
                yield meta_tbl, tok_lists

    try:
        for meta_tbl, tok_lists in _prefetch_iter(_iter_prepared_batches(), max_prefetch=2):
            if single_file_mode:
                vecs = _encode_int8_toks(tok_lists)
                tok_cnt = sum(len(t) for t in tok_lists)
                async_writer.submit(_write_single_sync, meta_tbl, vecs, tok_cnt)
            else:
                pending_meta.append(meta_tbl)
                pending_toks.extend(tok_lists)
                _drain_pending(final=False)

        if not single_file_mode:
            _drain_pending(final=True)
            rows_emitted, total_tokens = async_writer.drain()
        else:
            async_writer.submit(_close_single_sync)
            rows_emitted, total_tokens = async_writer.drain()
            if tmp_single is not None and tmp_single.exists():
                os.replace(tmp_single, out_p)
            else:
                empty_schema = pa.schema(
                    [f for f in first_schema if f.name in keep_names] + [pa.field("vector", pa_vec_type)]
                )
                pq.write_table(pa.Table.from_pylist([], schema=empty_schema), out_p, compression=compression)
    except BaseException:
        async_writer.abort_and_close()
        raise
    finally:
        async_writer.close()
        if single_writer is not None:
            single_writer.close()
        if tmp_single is not None and tmp_single.exists():
            tmp_single.unlink(missing_ok=True)

    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read_box[0],
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
    overlap: int = 0,
    header_text: str | None = None,
    header_col: str | None = None,
    repeat_header_tokens: int | None = None,
    window_overlap_chunks: int = 0,
    mode: str | None = None,
    dtype: str = "float32",
    batch_rows: int = 256,
    encode_batch_windows: int = 64,
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
    if int(window_overlap_chunks) < 0:
        raise ValueError(f"window_overlap_chunks must be non-negative, got {window_overlap_chunks}")
    if repeat_header_tokens is not None and int(repeat_header_tokens) < 0:
        raise ValueError(f"repeat_header_tokens must be non-negative, got {repeat_header_tokens}")
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
    eff_header_col = str(header_col) if header_col is not None else None
    if eff_header_col is not None:
        if eff_header_col not in first_schema.names:
            raise KeyError(f"Header column {eff_header_col!r} not found in schema {first_schema.names}")
        if eff_header_col not in cols_to_read:
            cols_to_read.append(eff_header_col)
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
    # encode_contextual_document always prepends [D] and excludes it from pooling; do not double-prepend document_prefix.
    if prefix is not None or prompt_name is not None or add_query_prefix:
        eff_prefix = tokenizer.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=False,
        )

    eff_window_batch = max(1, int(encode_batch_windows))
    if encode_batch_rows is not None and int(encode_batch_rows) > 0:
        # Honor CLI --encode-batch-rows as a chunk budget per contextual wave when tighter.
        chunks_per_win_est = max(1, int(context_tokens) // max(1, int(chunk_tokens)))
        eff_window_batch = max(1, min(eff_window_batch, max(1, int(encode_batch_rows) // chunks_per_win_est)))

    eff_normalize = False if dtype == "int8" else bool(normalize)

    tmp_single = out_p.with_suffix(".parquet.tmp") if single_file_mode else None
    single_writer_box: list[pq.ParquetWriter | None] = [None]
    shard_writer_box: list[pq.ParquetWriter | None] = [None]
    shard_tmp_box: list[Path | None] = [None]
    shard_idx = 0
    rows_read_box = [0]
    rows_emitted = 0
    total_tokens = 0
    wave_num = 0
    async_writer = _AsyncOrderedWriter(max_in_flight=2)
    encode_ctx_toks_fn = getattr(engine, "encode_contextual_tokens", None)
    eff_max_len = getattr(engine, "max_seq_len", None)
    plan_max_workers = min(4, max(1, os.cpu_count() or 1)) if isinstance(tokenizer, ContextualTokenizer) else 1
    plan_pool = (
        ThreadPoolExecutor(max_workers=plan_max_workers, thread_name_prefix="pplx-pq-plan")
        if plan_max_workers > 1
        else None
    )
    tok_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="pplx-pq-tok")
        if callable(encode_ctx_toks_fn)
        else None
    )

    # Pending prepared windows (bounded to <= shard_rows in sharded mode or flushed per batch in single-file mode).
    eff_header_text = header_text.strip() if (header_text is not None and header_text.strip()) else None
    default_repeat_tokens: int | None
    if repeat_header_tokens is not None:
        default_repeat_tokens = int(repeat_header_tokens)
    elif eff_header_text is not None:
        default_repeat_tokens = None
    elif int(context_tokens) >= 512:
        # Validated LegalBench-RAG default: anchor lead tokens on long continuation windows unless disabled (0).
        default_repeat_tokens = 128
    else:
        default_repeat_tokens = None

    # Each entry: (meta_row_dict, w_idx, ContextualWindow)
    pending_windows: list[tuple[dict[str, Any], int, ContextualWindow]] = []
    pending_chunk_count = 0

    def _build_wave_table(
        entries: Sequence[tuple[dict[str, Any], int, ContextualWindow]],
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
        for (meta_row, w_idx, win), vec_mat in zip(entries, vec_mats, strict=True):
            vec_np = np.asarray(vec_mat, dtype=np_dt)
            if vec_np.ndim == 1:
                vec_np = vec_np.reshape(1, target_dim)
            st = int(win.emit_start)
            n_emit = len(win.spans)
            if len(vec_np) < st + n_emit:
                raise RuntimeError(
                    f"Expected at least {st + n_emit} vectors for window {w_idx}, got {len(vec_np)}"
                )
            vec_blocks.append(vec_np[st : st + n_emit])
            emitted_chunks = win.chunks[st : st + n_emit]
            for c_idx, (c_text, (c0, c1)) in enumerate(zip(emitted_chunks, win.spans, strict=True)):
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

    def _encode_ctx_window(chunks_w: Sequence[str]) -> tuple[list[int], list[tuple[int, int]]]:
        return tokenizer.encode_contextual_document(
            chunks_w,
            return_spans=True,
            max_length=eff_max_len,
        )

    def _prepare_sub(
        entries: Sequence[tuple[dict[str, Any], int, ContextualWindow]],
    ) -> tuple[bool, Any, int]:
        win_chunks = [e[2].chunks for e in entries]
        if callable(encode_ctx_toks_fn):
            if plan_pool is not None and len(win_chunks) >= 8:
                encoded_sub = list(plan_pool.map(_encode_ctx_window, win_chunks))
            else:
                encoded_sub = [_encode_ctx_window(chunks_w) for chunks_w in win_chunks]
            tok_cnt = sum(len(ids) for ids, _ in encoded_sub)
            return True, encoded_sub, tok_cnt
        tok_cnt = sum(len(chunks_w) for chunks_w in win_chunks)
        return False, win_chunks, tok_cnt

    def _encode_prepared_sub(prepared: tuple[bool, Any, int]) -> tuple[list[np.ndarray], int]:
        is_toks, payload, tok_cnt = prepared
        enc_kw: dict[str, Any] = {
            "dimensions": target_dim,
            "output_dtype": dtype,
            "normalize": eff_normalize,
        }
        if mode is not None:
            enc_kw["mode"] = mode
        if is_toks:
            win_vecs = list(encode_ctx_toks_fn(payload, **enc_kw))
        else:
            win_vecs = list(engine.encode_contextual_chunks(payload, **enc_kw))
        clear_mlx_arena(256)
        return win_vecs, tok_cnt

    def _write_single_wave_sync(
        entries: Sequence[tuple[dict[str, Any], int, ContextualWindow]],
        win_vecs: Sequence[np.ndarray],
        tok_cnt: int,
    ) -> tuple[int, int]:
        tbl = _build_wave_table(entries, win_vecs)
        if single_writer_box[0] is None:
            single_writer_box[0] = pq.ParquetWriter(tmp_single, out_schema, compression=compression)
        single_writer_box[0].write_table(tbl)
        return tbl.num_rows, tok_cnt

    def _close_single_sync() -> tuple[int, int]:
        if single_writer_box[0] is not None:
            single_writer_box[0].close()
            single_writer_box[0] = None
        return 0, 0

    def _write_shard_sub_sync(
        sub_entries: Sequence[tuple[dict[str, Any], int, ContextualWindow]],
        win_vecs: Sequence[np.ndarray],
        tok_cnt: int,
        tmp_shard: Path,
        finalize_shard_path: Path | None,
    ) -> tuple[int, int]:
        tbl = _build_wave_table(sub_entries, win_vecs)
        try:
            if shard_writer_box[0] is None:
                shard_tmp_box[0] = tmp_shard
                shard_writer_box[0] = pq.ParquetWriter(tmp_shard, out_schema, compression=compression)
            shard_writer_box[0].write_table(tbl)
            if finalize_shard_path is not None:
                shard_writer_box[0].close()
                shard_writer_box[0] = None
                os.replace(tmp_shard, finalize_shard_path)
                shard_tmp_box[0] = None
        except BaseException:
            if shard_writer_box[0] is not None:
                shard_writer_box[0].close()
                shard_writer_box[0] = None
            tmp_shard.unlink(missing_ok=True)
            shard_tmp_box[0] = None
            raise
        return tbl.num_rows, tok_cnt

    def _log_progress(progress_chunks: int, progress_tokens: int) -> None:
        if progress_every > 0 and (wave_num % progress_every == 0):
            dt = time.perf_counter() - t0
            print(
                f"[contextual parquet wave {wave_num}] chunks={progress_chunks:,} toks={progress_tokens:,} "
                f"elapsed={dt:.2f}s ({progress_tokens / max(dt, 1e-9):,.0f} tok/s)",
                flush=True,
            )

    progress_chunks = 0
    progress_tokens = 0

    def _flush_shard_wave(entries: list[tuple[dict[str, Any], int, ContextualWindow]]) -> None:
        nonlocal shard_idx, wave_num, progress_chunks, progress_tokens
        if not entries:
            return
        n_chunks_wave = sum(len(e[2].spans) for e in entries)
        wave_num += 1
        shard_path = out_p / f"part-{shard_idx:05d}.parquet"
        tmp_shard = out_p / f".part-{shard_idx:05d}.parquet.tmp"
        shard_idx += 1
        if resume and shard_path.exists():
            try:
                if pq.ParquetFile(shard_path).metadata.num_rows == n_chunks_wave:
                    return
            except Exception:
                pass
        subs = [entries[st : st + eff_window_batch] for st in range(0, len(entries), eff_window_batch)]
        cur_fut: Future[tuple[bool, Any, int]] | None = (
            tok_pool.submit(_prepare_sub, subs[0]) if tok_pool is not None else None
        )
        for i, sub in enumerate(subs):
            prepared = cur_fut.result() if cur_fut is not None else _prepare_sub(sub)
            cur_fut = (
                tok_pool.submit(_prepare_sub, subs[i + 1])
                if (tok_pool is not None and i + 1 < len(subs))
                else None
            )
            win_vecs, tok_cnt = _encode_prepared_sub(prepared)
            finalize_path = shard_path if i + 1 == len(subs) else None
            async_writer.submit(_write_shard_sub_sync, sub, win_vecs, tok_cnt, tmp_shard, finalize_path)
            progress_chunks += sum(len(e[2].spans) for e in sub)
            progress_tokens += tok_cnt
        _log_progress(progress_chunks, progress_tokens)

    plan_fn = getattr(tokenizer, "plan_contextual_windows", None)

    def _plan_one_doc(args: tuple[str, str | None, int | None]) -> list[ContextualWindow]:
        doc_input, row_hdr_opt, row_rep = args
        if callable(plan_fn):
            return plan_fn(
                doc_input,
                chunk_tokens=int(chunk_tokens),
                context_tokens=int(context_tokens),
                overlap=int(overlap),
                header_text=row_hdr_opt,
                repeat_header_tokens=row_rep,
                window_overlap_chunks=int(window_overlap_chunks),
            )
        return [
            ContextualWindow(chunks=c_w, spans=s_w, emit_start=0)
            for c_w, s_w in tokenizer.chunk_and_group_document(
                doc_input,
                chunk_tokens=int(chunk_tokens),
                context_tokens=int(context_tokens),
                overlap=int(overlap),
                header_text=row_hdr_opt,
                repeat_header_tokens=row_rep,
            )
        ]

    def _iter_planned_docs() -> Iterator[tuple[dict[str, Any], list[ContextualWindow]]]:
        for fpath in files:
            pf = pq.ParquetFile(fpath)
            avail_cols = [c for c in cols_to_read if c in pf.schema_arrow.names]
            for rb in pf.iter_batches(batch_size=max(1, int(batch_rows)), columns=avail_cols):
                py_rows = rb.to_pylist()
                staged: list[tuple[dict[str, Any], tuple[str, str | None, int | None] | None]] = []
                plan_indices: list[int] = []
                plan_args: list[tuple[str, str | None, int | None]] = []
                for r in py_rows:
                    if limit is not None and rows_read_box[0] >= int(limit):
                        break
                    rows_read_box[0] += 1
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
                    meta_row = {k: r.get(k) for k in keep_names}
                    if not raw_text.strip():
                        if skip_empty:
                            continue
                        staged.append((meta_row, None))
                    else:
                        doc_input = (eff_prefix + raw_text) if eff_prefix else raw_text
                        raw_hdr = r.get(eff_header_col) if eff_header_col is not None else eff_header_text
                        row_hdr = str(raw_hdr).strip() if raw_hdr is not None else ""
                        if not row_hdr:
                            row_hdr = eff_header_text or ""
                        row_hdr_opt = row_hdr or None
                        row_rep = (
                            int(repeat_header_tokens)
                            if repeat_header_tokens is not None
                            else (None if row_hdr_opt is not None else default_repeat_tokens)
                        )
                        plan_indices.append(len(staged))
                        args = (doc_input, row_hdr_opt, row_rep)
                        plan_args.append(args)
                        staged.append((meta_row, args))

                if staged:
                    if plan_pool is not None and len(plan_args) >= 4:
                        planned_list = list(plan_pool.map(_plan_one_doc, plan_args))
                    else:
                        planned_list = [_plan_one_doc(a) for a in plan_args]
                    planned_by_pos = dict(zip(plan_indices, planned_list, strict=True))
                    for pos, (meta_row, maybe_args) in enumerate(staged):
                        if maybe_args is None:
                            grouped = [ContextualWindow(chunks=[""], spans=[(0, 0)], emit_start=0)]
                        else:
                            grouped = planned_by_pos[pos]
                        if grouped:
                            yield meta_row, grouped

                if limit is not None and rows_read_box[0] >= int(limit):
                    return

    def _iter_single_waves() -> Iterator[list[tuple[dict[str, Any], int, ContextualWindow]]]:
        nonlocal pending_windows, pending_chunk_count
        for meta_row, grouped in _prefetch_iter(_iter_planned_docs(), max_prefetch=2):
            for w_idx, win in enumerate(grouped):
                pending_windows.append((meta_row, w_idx, win))
                pending_chunk_count += len(win.spans)
                if len(pending_windows) >= eff_window_batch:
                    wave = pending_windows
                    pending_windows = []
                    pending_chunk_count = 0
                    yield wave
        if pending_windows:
            wave = pending_windows
            pending_windows = []
            pending_chunk_count = 0
            yield wave

    try:
        if not single_file_mode:
            for meta_row, grouped in _prefetch_iter(_iter_planned_docs(), max_prefetch=2):
                for w_idx, win in enumerate(grouped):
                    pending_windows.append((meta_row, w_idx, win))
                    pending_chunk_count += len(win.spans)
                    if pending_chunk_count >= int(shard_rows):
                        _flush_shard_wave(pending_windows)
                        pending_windows = []
                        pending_chunk_count = 0
            if pending_windows:
                _flush_shard_wave(pending_windows)
                pending_windows = []
                pending_chunk_count = 0
            rows_emitted, total_tokens = async_writer.drain()
        else:
            wave_iter = _iter_single_waves()
            cur_wave = next(wave_iter, None)
            cur_fut: Future[tuple[bool, Any, int]] | None = (
                tok_pool.submit(_prepare_sub, cur_wave)
                if (cur_wave is not None and tok_pool is not None)
                else None
            )
            while cur_wave is not None:
                wave_num += 1
                next_wave = next(wave_iter, None)
                prepared = cur_fut.result() if cur_fut is not None else _prepare_sub(cur_wave)
                next_fut = (
                    tok_pool.submit(_prepare_sub, next_wave)
                    if (next_wave is not None and tok_pool is not None)
                    else None
                )
                win_vecs, tok_cnt = _encode_prepared_sub(prepared)
                async_writer.submit(_write_single_wave_sync, cur_wave, win_vecs, tok_cnt)
                progress_chunks += sum(len(e[2].spans) for e in cur_wave)
                progress_tokens += tok_cnt
                _log_progress(progress_chunks, progress_tokens)
                cur_wave = next_wave
                cur_fut = next_fut
            async_writer.submit(_close_single_sync)
            rows_emitted, total_tokens = async_writer.drain()
            if tmp_single is not None and tmp_single.exists():
                os.replace(tmp_single, out_p)
            else:
                pq.write_table(pa.Table.from_pylist([], schema=out_schema), out_p, compression=compression)
    except BaseException:
        async_writer.abort_and_close()
        raise
    finally:
        if plan_pool is not None:
            plan_pool.shutdown(wait=True)
        if tok_pool is not None:
            tok_pool.shutdown(wait=True)
        async_writer.close()
        if single_writer_box[0] is not None:
            single_writer_box[0].close()
            single_writer_box[0] = None
        if shard_writer_box[0] is not None:
            shard_writer_box[0].close()
            shard_writer_box[0] = None
        if shard_tmp_box[0] is not None and shard_tmp_box[0].exists():
            shard_tmp_box[0].unlink(missing_ok=True)
        if tmp_single is not None and tmp_single.exists():
            tmp_single.unlink(missing_ok=True)

    return ParquetEmbedSummary(
        input_files=len(files),
        rows_read=rows_read_box[0],
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
