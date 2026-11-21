"""Streaming and resumable Parquet + JSONL structured extraction pipeline for `fast-gemma4`."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from fast_shared.pipeline_llm import (
    DEFAULT_TEXT_COLUMNS,
    ParquetExtractSummary,
    extract_jsonl as _shared_extract_jsonl,
    extract_parquet as _shared_extract_parquet,
    resolve_input_files,
    resolve_llm_text_column as _resolve_gemma4_text_column,
    resolve_text_column,
    slice_first_pages_utf8,
    slice_ocr_pages_utf8,
)


extract_parquet = partial(_shared_extract_parquet, whitespace_flexible=None)


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
    batch_size: int = 128,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
) -> dict[str, Any]:
    """Batch-extract structured JSON from a JSON or JSONL file into JSONL."""
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
        response_text_key="response_text",
        response_json_key="response_json",
    )


__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "ParquetExtractSummary",
    "_resolve_gemma4_text_column",
    "extract_jsonl",
    "extract_parquet",
    "resolve_input_files",
    "resolve_text_column",
    "slice_first_pages_utf8",
    "slice_ocr_pages_utf8",
]
