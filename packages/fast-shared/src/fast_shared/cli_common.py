"""Shared CLI parser builders and command runners for `embed`, `parquet`, and `serve`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from .embeddings import accepts_kwarg, truncate_and_normalize
from .parquet_pipeline import embed_parquet
from .server import EmbeddingHTTPServer


def add_embed_arguments(
    parser: argparse.ArgumentParser,
    *,
    mode_choices: Sequence[str],
    default_mode: str,
    allow_doc_prefix: bool = False,
    allow_dimensions: bool = False,
) -> None:
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--query-prefix", action="store_true")
    parser.add_argument(
        "--header-col",
        default=None,
        help="Optional input Parquet column containing per-document header/title text prepended to each row or chunk",
    )
    parser.add_argument(
        "--header-text",
        default=None,
        help="Optional static document header/title text prepended to each row or chunk",
    )
    if allow_doc_prefix:
        parser.add_argument("--document-prefix", action="store_true")
        parser.add_argument("--prefix", default=None)
        parser.add_argument("--prompt-name", default=None)
    if allow_dimensions:
        parser.add_argument(
            "--dimensions",
            type=int,
            default=None,
            help="Matryoshka output embedding dimensions (e.g. 128, 256, 512, 768)",
        )
    parser.add_argument("--mode", choices=tuple(mode_choices), default=default_mode)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true")


def add_parquet_arguments(
    parser: argparse.ArgumentParser,
    *,
    mode_choices: Sequence[str],
    default_mode: str,
    default_max_body_tokens: int | None = None,
    allow_doc_prefix: bool = False,
    allow_dimensions: bool = False,
) -> None:
    parser.add_argument("--input", required=True, help="Parquet file, directory of .parquet files, or glob")
    parser.add_argument("--output", required=True, help="Output .parquet file or shard directory")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--keep-cols", nargs="+", default=None)
    parser.add_argument("--drop-text-col", action="store_true")
    parser.add_argument("--mode", choices=tuple(mode_choices), default=default_mode)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    if allow_dimensions:
        parser.add_argument(
            "--dimensions",
            type=int,
            default=None,
            help="Matryoshka output embedding dimensions (e.g. 128, 256, 512, 768)",
        )
    parser.add_argument("--batch-rows", type=int, default=4096, help="Source rows per preparation batch; use a smaller value for long documents")
    parser.add_argument("--encode-batch-rows", type=int, default=4096, help="Maximum chunks per model call, after chunk expansion")
    parser.add_argument("--shard-rows", type=int, default=65536, help="Output rows per stored shard")
    parser.add_argument("--filter-ocr", action="store_true", help="Filter is_ocr==True or len(ocr_pages)>0")
    parser.add_argument("--ocr-pages-only", action="store_true", help="Slice body_text UTF-8 page_offsets to ocr_pages")
    parser.add_argument("--per-ocr-page", action="store_true", help="Emit 1 row (or chunk) per OCR page")
    parser.add_argument("--chunk", action="store_true", help="Split documents into overlapping token windows")
    parser.add_argument("--max-chunk-tokens", type=int, default=512, help="Maximum encoded chunk length, including prompt and special tokens (default: 512)")
    parser.add_argument("--max-body-tokens", type=int, default=default_max_body_tokens, help="Optional extra limit on original body tokens per chunk")
    parser.add_argument("--overlap", "--chunk-overlap", type=int, default=51, help="Overlap in original body tokens; must be smaller than the usable body window")
    parser.add_argument("--max-length", type=int, default=None, help="Token limit for unchunked input; --max-chunk-tokens controls chunked input")
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--include-chunk-text", action="store_true")
    parser.add_argument("--query-prefix", action="store_true")
    parser.add_argument(
        "--header-col",
        default=None,
        help="Optional input Parquet column containing per-document header/title text prepended to each row or chunk",
    )
    parser.add_argument(
        "--header-text",
        default=None,
        help="Optional static document header/title text prepended to each row or chunk",
    )
    if allow_doc_prefix:
        parser.add_argument("--document-prefix", action="store_true")
        parser.add_argument("--prefix", default=None)
        parser.add_argument("--prompt-name", default=None)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)


def add_serve_arguments(
    parser: argparse.ArgumentParser,
    *,
    mode_choices: Sequence[str],
    default_mode: str,
    allow_dimensions: bool = False,
) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    if allow_dimensions:
        parser.add_argument(
            "--dimensions",
            type=int,
            default=None,
            help="Default Matryoshka output embedding dimensions (e.g. 128, 256, 512, 768)",
        )
    parser.add_argument("--mode", choices=tuple(mode_choices), default=default_mode)
    parser.add_argument("--no-warmup", action="store_true")


def run_embed_command(args: argparse.Namespace, make_engine: Callable[[argparse.Namespace], Any]) -> int:
    texts = list(args.texts)
    if args.stdin:
        texts.extend(line.rstrip("\r\n") for line in sys.stdin if line.rstrip("\r\n"))
    if not texts:
        print("No input text provided. Pass text arguments or --stdin.", file=sys.stderr)
        return 2

    eng = make_engine(args)
    kw: dict[str, Any] = {
        "is_query": args.query,
        "add_query_prefix": args.query_prefix,
        "mode": args.mode,
    }
    if hasattr(args, "document_prefix") and args.document_prefix:
        kw["add_document_prefix"] = True
    if getattr(args, "prefix", None) is not None:
        kw["prefix"] = args.prefix
    if getattr(args, "prompt_name", None) is not None:
        kw["prompt_name"] = args.prompt_name
    req_dim = getattr(args, "dimensions", None)
    if req_dim is not None and accepts_kwarg(eng.encode_texts, "dimensions"):
        kw["dimensions"] = req_dim

    t0 = time.perf_counter()
    vecs = eng.encode_texts(texts, **kw)
    dt_ms = (time.perf_counter() - t0) * 1e3
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    if req_dim is not None:
        vecs = truncate_and_normalize(vecs, req_dim)

    if args.json:
        print(
            json.dumps(
                {
                    "count": int(vecs.shape[0]),
                    "dim": int(vecs.shape[1]),
                    "mode": args.mode,
                    "latency_ms": round(dt_ms, 3),
                    "embeddings": vecs.tolist(),
                }
            )
        )
    else:
        for i, row in enumerate(vecs):
            print(f"[{i}] norm={np.linalg.norm(row):.6f} vec[:6]={np.round(row[:6], 5).tolist()}")
        print(f"Encoded {vecs.shape[0]} text(s) in {dt_ms:.2f} ms (mode={args.mode})")
    return 0


def run_parquet_command(args: argparse.Namespace, make_engine: Callable[[argparse.Namespace], Any]) -> int:
    eng = make_engine(args)
    summary = embed_parquet(
        eng,
        args.input,
        args.output,
        text_col=args.text_col,
        keep_cols=args.keep_cols,
        drop_text_col=args.drop_text_col,
        mode=args.mode,
        dtype=args.dtype,
        batch_rows=args.batch_rows,
        encode_batch_rows=args.encode_batch_rows,
        shard_rows=args.shard_rows,
        filter_ocr=args.filter_ocr,
        ocr_pages_only=args.ocr_pages_only,
        per_ocr_page=args.per_ocr_page,
        chunk=args.chunk,
        max_body_tokens=args.max_body_tokens,
        overlap=args.overlap,
        max_length=args.max_length,
        max_chunk_tokens=args.max_chunk_tokens,
        skip_empty=args.skip_empty,
        include_chunk_text=args.include_chunk_text,
        prefix=getattr(args, "prefix", None),
        prompt_name=getattr(args, "prompt_name", None),
        add_query_prefix=args.query_prefix,
        add_document_prefix=bool(getattr(args, "document_prefix", False)),
        header_col=getattr(args, "header_col", None),
        header_text=getattr(args, "header_text", None),
        dimensions=getattr(args, "dimensions", None),
        compression=args.compression,
        resume=args.resume,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        f"DONE: files={summary.input_files} rows_read={summary.rows_read:,} "
        f"rows_emitted={summary.rows_emitted:,} tokens={summary.total_tokens:,} "
        f"elapsed={summary.elapsed_seconds:.2f}s throughput={summary.tokens_per_sec:,.0f} tok/s "
        f"({summary.rows_per_sec:,.0f} rows/s) -> {summary.output_path}"
    )
    return 0


def run_serve_command(
    args: argparse.Namespace,
    make_engine: Callable[[argparse.Namespace], Any],
    *,
    label: str,
) -> int:
    eng = make_engine(args)
    mode = getattr(args, "mode", None)
    srv = EmbeddingHTTPServer(eng, host=args.host, port=args.port, mode=mode)
    print(
        f"Serving {label} on http://{srv.host}:{srv.port} "
        f"(dim={getattr(eng, 'dim', '?')}, ane={eng.ane_available}, mode={mode or eng.default_mode})",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0
