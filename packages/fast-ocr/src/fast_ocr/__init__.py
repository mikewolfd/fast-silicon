"""Fast zero-training Apple Silicon OCR (`AppleVisionOCR` + `FastOvisOCREngine` + `AutoOCRPipeline`)."""

from __future__ import annotations

from typing import Any

from .apple_vision import (
    AppleVisionOCR,
    deskew_document_image,
    detect_document_quad,
    is_likely_camera_photo,
    normalize_deskew_mode,
    perspective_deskew_image,
)
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
from .paths import (
    DEFAULT_OVIS_GFM_REPO,
    DEFAULT_OVIS_REPO,
    OCR_MODEL_PRESETS,
    OCR_PRESET_FALLBACKS,
    resolve_ocr_model_dir,
    resolve_v7final_local_dir,
)
from .pipeline import AutoOCRPipeline, FastOCR, ocr_parquet
from .preprocess import normalize_faded_scan_image, strip_running_headers_footers
from .region_router import (
    REGION_CROP_MIN_PIXELS,
    REGION_FORMULA_PROMPT,
    REGION_TABLE_PROMPT,
    PageSegment,
    build_ordered_pld_corpus,
    compute_adaptive_max_pixels,
    estimate_visual_tokens,
    extract_escalation_regions,
    merge_region_markdown,
)
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
    "DEFAULT_OVIS_GFM_REPO",
    "DEFAULT_OVIS_REPO",
    "DynamicTableRowGrammar",
    "FastOCR",
    "FastOvisOCREngine",
    "FastQwen35VisionModel",
    "MarkdownTableMode",
    "MarkdownValidation",
    "OCRBackend",
    "OCRBox",
    "OCRLine",
    "OCR_MODEL_PRESETS",
    "OCR_PRESET_FALLBACKS",
    "OCRResult",
    "OCRTimings",
    "OutputFormat",
    "PageAnalysis",
    "PageSegment",
    "Qwen35VisionConfig",
    "REGION_CROP_MIN_PIXELS",
    "REGION_FORMULA_PROMPT",
    "REGION_TABLE_PROMPT",
    "SpeculativeMode",
    "align_json_values_to_lines",
    "build_ordered_pld_corpus",
    "canonicalize_markdown",
    "compute_adaptive_max_pixels",
    "compute_missing_numeric_crop_box_px",
    "compute_missing_numeric_crop_boxes_px",
    "deskew_document_image",
    "detect_document_quad",
    "estimate_visual_tokens",
    "extract_escalation_regions",
    "extract_normalized_numbers",
    "find_filtered_missing_numbers",
    "is_likely_camera_photo",
    "merge_region_markdown",
    "normalize_deskew_mode",
    "normalize_faded_scan_image",
    "ocr_parquet",
    "perspective_deskew_image",
    "resolve_ocr_mlx_limits",
    "resolve_ocr_model_dir",
    "resolve_v7final_local_dir",
    "strip_running_headers_footers",
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
