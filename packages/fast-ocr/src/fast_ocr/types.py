"""Core data structures for `fast-ocr`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OCRBackend = Literal["apple-vision", "ovis", "auto"]
OutputFormat = Literal["markdown", "raw", "text"]
MarkdownTableMode = Literal["gfm", "html"]
SpeculativeMode = Literal["auto", "off", "self", "vision+self"]


@dataclass(frozen=True)
class OCRBox:
    """Normalized top-left image bounding box in `[0.0, 1.0]`."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class OCRLine:
    """Single recognized line of text with confidence and top-left bounding box."""

    text: str
    confidence: float
    box: OCRBox


@dataclass(frozen=True)
class PageAnalysis:
    """Lightweight layout, orientation, ink-crop, and complexity summary derived from Apple Vision."""

    width: int
    height: int
    line_count: int
    char_count: int
    digit_ratio: float
    currency_hits: int
    mean_confidence: float
    low_conf_line_ratio: float
    median_line_height: float
    inferred_rotation_cw: int  # 0, 90, 180, 270
    ink_crop_box_px: tuple[int, int, int, int] | None  # (left, top, right, bottom)
    ink_area_ratio: float
    estimated_columns: int
    table_score: float
    needs_vlm: bool
    escalation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarkdownValidation:
    """Deterministic structural validation receipt for Markdown/HTML OCR output."""

    valid: bool
    table_count: int
    gfm_table_count: int
    html_table_count: int
    fence_balanced: bool
    math_balanced: bool
    html_tables_balanced: bool
    gfm_tables_rectangular: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class OCRTimings:
    """Stage-by-stage timing and throughput breakdown for one OCR page."""

    preprocess_ms: float = 0.0
    apple_vision_ms: float = 0.0
    vit_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    decode_tps: float = 0.0
    prompt_tokens: int = 0
    visual_tokens: int = 0
    generated_tokens: int = 0
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    verify_passes: int = 0
    peak_memory_gb: float = 0.0


@dataclass(frozen=True)
class OCRResult:
    """Result for a single OCR page across native PDF text, Apple Vision (ANE), or Qwen3.5 VLM."""

    text: str
    raw_text: str
    backend_used: str
    page_index: int = 0
    output_format: OutputFormat = "markdown"
    table_mode: MarkdownTableMode = "gfm"
    rotation_applied_cw: int = 0
    cropped_box_px: tuple[int, int, int, int] | None = None
    lines: tuple[OCRLine, ...] = ()
    analysis: PageAnalysis | None = None
    markdown_validation: MarkdownValidation | None = None
    timings: OCRTimings = field(default_factory=OCRTimings)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)
