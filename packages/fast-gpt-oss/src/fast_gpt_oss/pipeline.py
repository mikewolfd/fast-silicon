"""Streaming and resumable Parquet + JSONL extraction built DRY on `fast_shared.pipeline_llm`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fast_shared.pipeline_llm import (
    DEFAULT_TEXT_COLUMNS,
    ParquetExtractSummary,
    extract_jsonl as _shared_extract_jsonl,
    extract_parquet as _shared_extract_parquet,
    slice_first_pages_utf8,
)
from fast_shared.parquet_pipeline import (
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)


def extract_parquet(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    system_prompt: str,
    json_schema: str | dict[str, Any] | None = None,
    text_col: str | None = None,
    keep_cols: list[str] | None = None,
    drop_text_col: bool = False,
    mode: str | None = None,
    speculative: bool | None = None,
    batch_rows: int | None = None,
    gpu_batch_size: int | None = None,
    shard_rows: int = 4096,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
    max_chars: int = 8000,
    max_pages: int = 0,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    skip_empty: bool = True,
    resume: bool = False,
    mlx_cache_limit_mb: int | None = 512,
    use_wired_limit: bool = False,
    **extra_gen_kwargs: Any,
) -> ParquetExtractSummary:
    """Stream-extract structured JSON from Parquet input(s), defaulting to spaced `(", ", ": ")` JSON."""
    return _shared_extract_parquet(
        engine,
        input_path,
        output_path,
        system_prompt=system_prompt,
        json_schema=json_schema,
        text_col=text_col,
        keep_cols=keep_cols,
        drop_text_col=drop_text_col,
        mode=mode,
        speculative=speculative,
        batch_rows=batch_rows,
        gpu_batch_size=gpu_batch_size,
        shard_rows=shard_rows,
        max_tokens=max_tokens,
        temperature=temperature,
        whitespace_flexible=whitespace_flexible,
        max_chars=max_chars,
        max_pages=max_pages,
        filter_ocr=filter_ocr,
        ocr_pages_only=ocr_pages_only,
        skip_empty=skip_empty,
        resume=resume,
        mlx_cache_limit_mb=mlx_cache_limit_mb,
        use_wired_limit=use_wired_limit,
        **extra_gen_kwargs,
    )


def extract_jsonl(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    system_prompt: str,
    json_schema: str | dict[str, Any] | None = None,
    text_key: str = "text",
    mode: str | None = None,
    speculative: bool | None = None,
    batch_size: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
    response_text_key: str = "output_text",
    response_json_key: str = "parsed",
    mlx_cache_limit_mb: int | None = 512,
    use_wired_limit: bool = False,
    **extra_gen_kwargs: Any,
) -> dict[str, Any]:
    """Batch-extract structured JSON from JSON/JSONL into JSONL, defaulting to spaced `(", ", ": ")` JSON."""
    return _shared_extract_jsonl(
        engine,
        input_path,
        output_path,
        system_prompt=system_prompt,
        json_schema=json_schema,
        text_key=text_key,
        mode=mode,
        speculative=speculative,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        whitespace_flexible=whitespace_flexible,
        response_text_key=response_text_key,
        response_json_key=response_json_key,
        mlx_cache_limit_mb=mlx_cache_limit_mb,
        use_wired_limit=use_wired_limit,
        **extra_gen_kwargs,
    )


__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "ParquetExtractSummary",
    "extract_jsonl",
    "extract_parquet",
    "resolve_input_files",
    "resolve_text_column",
    "slice_first_pages_utf8",
    "slice_ocr_pages_utf8",
]
