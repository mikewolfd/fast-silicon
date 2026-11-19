"""Streaming and resumable Parquet + JSONL extraction built DRY on `fast_shared.pipeline_llm`."""

from __future__ import annotations

from fast_shared.pipeline_llm import (
    DEFAULT_TEXT_COLUMNS,
    ParquetExtractSummary,
    extract_jsonl,
    extract_parquet,
    slice_first_pages_utf8,
)
from fast_shared.parquet_pipeline import (
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
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
