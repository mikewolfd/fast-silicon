"""Core data structures for `fast-ocr`."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Sequence

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


DEFAULT_CONFIDENCE_WINDOW: int = 16
DEFAULT_LOW_LOGPROB_THRESHOLD: float = -0.7
DEFAULT_LOW_MARGIN_THRESHOLD: float = 1.0


@dataclass(frozen=True)
class LogprobConfidence:
    """Aggregated per-page token log-probability and greedy logit-margin statistics."""

    n_tokens: int
    mean_logprob: float
    mean_prob: float
    min_window_logprob: float
    p05_logprob: float
    low_conf_frac: float
    window_size: int = DEFAULT_CONFIDENCE_WINDOW
    low_logprob_threshold: float = DEFAULT_LOW_LOGPROB_THRESHOLD
    mean_margin: float | None = None
    min_window_margin: float | None = None
    low_margin_frac: float | None = None
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD
    mean_entropy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_tokens": int(self.n_tokens),
            "mean_logprob": round(float(self.mean_logprob), 6),
            "mean_prob": round(float(self.mean_prob), 6),
            "min_window_logprob": round(float(self.min_window_logprob), 6),
            "p05_logprob": round(float(self.p05_logprob), 6),
            "low_conf_frac": round(float(self.low_conf_frac), 6),
            "window_size": int(self.window_size),
            "low_logprob_threshold": float(self.low_logprob_threshold),
        }
        if self.mean_margin is not None:
            out["mean_margin"] = round(float(self.mean_margin), 6)
        if self.min_window_margin is not None:
            out["min_window_margin"] = round(float(self.min_window_margin), 6)
        if self.low_margin_frac is not None:
            out["low_margin_frac"] = round(float(self.low_margin_frac), 6)
            out["low_margin_threshold"] = float(self.low_margin_threshold)
        if self.mean_entropy is not None:
            out["mean_entropy"] = round(float(self.mean_entropy), 6)
        return out


def _min_sliding_window_mean(values: Sequence[float], window: int) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    w = min(max(1, int(window)), n)
    cur_sum = sum(float(v) for v in values[:w])
    min_sum = cur_sum
    for i in range(w, n):
        cur_sum += float(values[i]) - float(values[i - w])
        if cur_sum < min_sum:
            min_sum = cur_sum
    return min_sum / float(w)


def summarize_logprob_confidence(
    logprobs: Sequence[float],
    *,
    probs: Sequence[float] | None = None,
    margins: Sequence[float] | None = None,
    entropies: Sequence[float] | None = None,
    window_size: int = DEFAULT_CONFIDENCE_WINDOW,
    low_logprob_threshold: float = DEFAULT_LOW_LOGPROB_THRESHOLD,
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD,
) -> LogprobConfidence:
    """Compute per-page token confidence summary without requiring heavy numeric imports."""
    n = len(logprobs)
    w_eff = min(max(1, int(window_size)), max(1, n))
    if n == 0:
        return LogprobConfidence(
            n_tokens=0,
            mean_logprob=0.0,
            mean_prob=1.0,
            min_window_logprob=0.0,
            p05_logprob=0.0,
            low_conf_frac=0.0,
            window_size=int(window_size),
            low_logprob_threshold=float(low_logprob_threshold),
            mean_margin=0.0 if margins is not None else None,
            min_window_margin=0.0 if margins is not None else None,
            low_margin_frac=0.0 if margins is not None else None,
            low_margin_threshold=float(low_margin_threshold),
            mean_entropy=0.0 if entropies is not None else None,
        )

    lps = [float(x) for x in logprobs]
    mean_lp = sum(lps) / float(n)
    if probs is not None and len(probs) == n:
        mean_p = sum(float(p) for p in probs) / float(n)
    else:
        mean_p = sum(math.exp(max(-60.0, min(0.0, lp))) for lp in lps) / float(n)

    min_win_lp = _min_sliding_window_mean(lps, w_eff)
    sorted_lps = sorted(lps)
    p05_idx = min(n - 1, max(0, int(0.05 * (n - 1))))
    p05_lp = sorted_lps[p05_idx]
    low_lp_thresh = float(low_logprob_threshold)
    low_conf_frac = sum(1 for lp in lps if lp < low_lp_thresh) / float(n)

    mean_m: float | None = None
    min_win_m: float | None = None
    low_m_frac: float | None = None
    low_m_thresh = float(low_margin_threshold)
    if margins is not None and len(margins) == n:
        m_vals = [float(m) for m in margins]
        mean_m = sum(m_vals) / float(n)
        min_win_m = _min_sliding_window_mean(m_vals, w_eff)
        low_m_frac = sum(1 for m in m_vals if m < low_m_thresh) / float(n)

    mean_ent: float | None = None
    if entropies is not None and len(entropies) == n:
        mean_ent = sum(float(e) for e in entropies) / float(n)

    return LogprobConfidence(
        n_tokens=n,
        mean_logprob=mean_lp,
        mean_prob=mean_p,
        min_window_logprob=min_win_lp,
        p05_logprob=p05_lp,
        low_conf_frac=low_conf_frac,
        window_size=int(window_size),
        low_logprob_threshold=low_lp_thresh,
        mean_margin=mean_m,
        min_window_margin=min_win_m,
        low_margin_frac=low_m_frac,
        low_margin_threshold=low_m_thresh,
        mean_entropy=mean_ent,
    )
