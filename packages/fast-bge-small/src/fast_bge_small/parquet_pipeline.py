"""Streaming sharded Parquet embedding pipeline (backed by `fast_shared.parquet_pipeline`)."""

from __future__ import annotations

from fast_shared.parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    ParquetEmbedSummary,
    embed_parquet,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)

__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "ParquetEmbedSummary",
    "embed_parquet",
    "resolve_input_files",
    "resolve_text_column",
    "slice_ocr_pages_utf8",
]
