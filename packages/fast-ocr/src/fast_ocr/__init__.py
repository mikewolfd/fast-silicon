"""Fast zero-training Apple Silicon OCR (`AppleVisionOCR` + `FastOvisOCREngine` + `AutoOCRPipeline`)."""

from __future__ import annotations

from typing import Any

from .apple_vision import AppleVisionOCR
from .markdown_struct import (
    DynamicTableRowGrammar,
    canonicalize_markdown,
    synthesize_markdown_from_lines,
    validate_markdown,
)
from .numeric_recovery import (
    align_json_values_to_lines,
    compute_missing_numeric_crop_box_px,
    compute_missing_numeric_crop_boxes_px,
    extract_normalized_numbers,
    find_filtered_missing_numbers,
)
from .pipeline import AutoOCRPipeline, FastOCR, ocr_parquet
from .types import (
    MarkdownTableMode,
    MarkdownValidation,
    OCRBackend,
    OCRBox,
    OCRLine,
    OCRResult,
    OCRTimings,
    OutputFormat,
    PageAnalysis,
    SpeculativeMode,
)

__version__ = "0.1.0"

__all__ = [
    "AppleVisionOCR",
    "AutoOCRPipeline",
    "DynamicTableRowGrammar",
    "FastOCR",
    "FastOvisOCREngine",
    "FastQwen35VisionModel",
    "MarkdownTableMode",
    "MarkdownValidation",
    "OCRBackend",
    "OCRBox",
    "OCRLine",
    "OCRResult",
    "OCRTimings",
    "OutputFormat",
    "PageAnalysis",
    "Qwen35VisionConfig",
    "SpeculativeMode",
    "align_json_values_to_lines",
    "canonicalize_markdown",
    "compute_missing_numeric_crop_box_px",
    "compute_missing_numeric_crop_boxes_px",
    "extract_normalized_numbers",
    "find_filtered_missing_numbers",
    "ocr_parquet",
    "resolve_ocr_mlx_limits",
    "synthesize_markdown_from_lines",
    "validate_markdown",
]


def __getattr__(name: str) -> Any:
    """Lazily import MLX/VLM classes so standalone `AppleVisionOCR` imports zero MLX weights or frameworks."""
    if name in ("FastOvisOCREngine", "resolve_ocr_mlx_limits"):
        from . import ovis_engine

        return getattr(ovis_engine, name)
    if name in ("FastQwen35VisionModel", "Qwen35VisionConfig"):
        from . import vision

        return getattr(vision, name)
    raise AttributeError(f"module 'fast_ocr' has no attribute {name!r}")
