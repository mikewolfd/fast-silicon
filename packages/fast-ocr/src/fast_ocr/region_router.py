"""Zero-MLX adaptive visual budgeting, reading-order PLD seeding, and region-only VLM escalation routing.

Distills model-agnostic document-parsing patterns (`MinerU2.5` / `PaddleOCR-VL` region escalation,
`GLM-OCR` / `DeepSeek-OCR-2` adaptive visual-token budgets, and `olmOCR-2` reading-order / native-text
PLD anchoring) into pure Python + NumPy helpers usable with `AppleVisionOCR` + `FastOvisOCREngine`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

import numpy as np

from .markdown_struct import (
    _group_visual_rows,
    _xy_cut_order,
    canonicalize_markdown,
    synthesize_markdown_from_lines,
)
from .preprocess import rotate_box_normalized, smart_resize_qwen
from .types import MarkdownTableMode, OCRLine, OCRResult, PageAnalysis

RegionKind = Literal["prose", "table", "formula"]

REGION_TABLE_PROMPT: str = (
    "Transcribe only the table in this cropped image as a single clean HTML table: <table>...</table>. "
    "Preserve every row, column header, alignment, and numeric value verbatim without commentary."
)
REGION_FORMULA_PROMPT: str = (
    "Transcribe all text and mathematical formulas in this cropped image cleanly as Markdown with exact "
    "LaTeX equations ($...$ for inline, $$...$$ for display). Preserve every symbol, subscript, and superscript verbatim."
)
# Small region crops (<256x256) blow up N_vis 1.65x-2.31x under default min_pixels=65536;
# pass min_pixels=REGION_CROP_MIN_PIXELS (4096 = 64x64) to keep crop ViT work proportional to area.
REGION_CROP_MIN_PIXELS: int = 4096

_MATH_SYMBOL_RE = re.compile(
    r"[=±×÷≤≥≠≈∞∑∏∫∂∇√∈∉⊂⊃∪∩→←⇒⇔∀∃αβγδεζηθλμνξπρστφχψωΓΔΘΛΞΠΣΦΨΩ^_\\]|"
    r"\b(?:sin|cos|tan|log|ln|exp|lim|sup|inf|min|max|arg|det|dim|ker)\b"
)
_CURRENCY_OR_NUM_RE = re.compile(
    r"(?:\$|USD|€|£)\s*\d[\d,]*(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2,}\b|\b\d+(?:\.\d+)?%"
)


@dataclass(frozen=True)
class PageSegment:
    """Single reading-order segment produced by ANE layout partitioning (`prose`, `table`, or `formula`)."""

    kind: RegionKind
    markdown: str | None = None
    crop_box_px: tuple[int, int, int, int] | None = None
    prompt: str | None = None
    line_count: int = 0


def build_ordered_pld_corpus(
    vision_res: OCRResult | None,
    *,
    native_pdf_text: str | None = None,
) -> tuple[str, list[str]]:
    """Build `(primary_text, extra_texts)` for `OCRPLDDrafter` in reading order + native-PDF fallback.

    Places XY-cut reading-order Vision Markdown first so monotonic cursor tracking matches natural
    multi-column decode order, with clean native PDF prose (when available) and raw Vision text as
    separated secondary lookup corpora.
    """
    primary = ""
    extras: list[str] = []

    if vision_res is not None:
        if vision_res.lines:
            primary = synthesize_markdown_from_lines(vision_res.lines).strip()
        if not primary and vision_res.raw_text:
            primary = vision_res.raw_text.strip()
        raw_txt = (vision_res.raw_text or "").strip()
        if raw_txt and raw_txt != primary:
            extras.append(raw_txt)

    if native_pdf_text and native_pdf_text.strip():
        clean_pdf = native_pdf_text.strip()
        if not primary:
            primary = clean_pdf
        elif clean_pdf != primary:
            # Prefer native PDF prose ahead of unsorted raw Vision lines when both exist.
            extras.insert(0, clean_pdf)

    return primary, extras


def compute_adaptive_max_pixels(
    analysis: PageAnalysis | None,
    image_size: tuple[int, int],
    *,
    cropped_size: tuple[int, int] | None = None,
    target_line_height_px: float = 24.0,
    table_line_height_px: float = 28.0,
    min_max_pixels: int = 786_432,
    default_max_pixels: int = 16_777_216,
    min_scale: float = 0.25,
) -> int:
    """Compute a lossless `max_pixels` cap for `build_qwen_vision_inputs` from ANE `median_line_height`.

    Large-print clean pages and 600-DPI scans shrink down to keep `>= target_line_height_px`
    pixels per text line (cutting ViT + prefill tokens without blurring characters), while tiny
    fonts, low-confidence scans, and dense tables retain full native pixel resolution.
    When `cropped_size` (post-`crop_to_ink` `(cw, ch)`) is supplied, applies the oriented line-height
    scale factor to `cropped_size` so white-margin trimming and resolution scaling compose cleanly.
    """
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0 or analysis is None:
        return int(default_max_pixels)
    if analysis.line_count < 6 or analysis.median_line_height <= 0.0:
        return int(default_max_pixels)
    reasons = set(analysis.escalation_reasons)
    if "tiny_font_scan" in reasons or "low_vision_confidence" in reasons:
        return int(default_max_pixels)

    line_h_px = float(analysis.median_line_height) * float(h)
    eff_target = float(target_line_height_px)
    if analysis.digit_ratio >= 0.25 or analysis.table_score >= 0.45 or "financial_table" in reasons:
        eff_target = max(eff_target, float(table_line_height_px))

    if line_h_px <= eff_target:
        return int(default_max_pixels)

    scale = min(1.0, max(float(min_scale), eff_target / line_h_px))
    if scale >= 0.94:
        return int(default_max_pixels)
    eff_w, eff_h = (int(cropped_size[0]), int(cropped_size[1])) if cropped_size is not None else (w, h)
    if eff_w <= 0 or eff_h <= 0:
        eff_w, eff_h = w, h
    target_px = int(round((float(eff_w) * scale) * (float(eff_h) * scale)))
    return max(int(min_max_pixels), min(int(default_max_pixels), target_px))


def estimate_visual_tokens(
    image_size: tuple[int, int],
    *,
    max_pixels: int = 16_777_216,
    min_pixels: int = 65_536,
    patch_size: int = 16,
    merge_size: int = 2,
) -> int:
    """Estimate Qwen3.5 merged visual token count for `(width, height)` under `max_pixels`."""
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return 0
    factor = int(patch_size) * int(merge_size)
    rh, rw = smart_resize_qwen(h, w, factor=factor, min_pixels=int(min_pixels), max_pixels=int(max_pixels))
    return (rh // factor) * (rw // factor)


def _row_is_table_like(row: Sequence[OCRLine]) -> bool:
    if len(row) >= 2:
        ordered = sorted(row, key=lambda l: l.box.x0)
        sep_cells = sum(
            1
            for i in range(1, len(ordered))
            if ordered[i].box.x0 > ordered[i - 1].box.x1 + 0.015
        ) + 1
        if sep_cells >= 3:
            return True
        if sep_cells == 2:
            joined = " ".join(l.text for l in ordered)
            if _CURRENCY_OR_NUM_RE.search(joined):
                return True
    return False


def _row_is_formula_like(row: Sequence[OCRLine]) -> bool:
    joined = " ".join(l.text.strip() for l in row if l.text.strip())
    if not joined:
        return False
    mean_conf = float(np.mean([float(l.confidence) for l in row]))
    math_hits = len(_MATH_SYMBOL_RE.findall(joined))
    if math_hits >= 3:
        return True
    if math_hits >= 1 and mean_conf < 0.72:
        return True
    if mean_conf < 0.60 and len(joined) >= 4:
        return True
    return False


def _lines_to_px_box(
    lines: Sequence[OCRLine],
    image_size: tuple[int, int],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.025,
) -> tuple[int, int, int, int] | None:
    if not lines:
        return None
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return None
    rot = int(rotation_cw) % 360
    boxes = [
        rotate_box_normalized((ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1), rot)
        if rot != 0
        else (ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1)
        for ln in lines
    ]
    pad = max(0.0, float(pad_frac))
    x0 = max(0, int(round((min(b[0] for b in boxes) - pad) * w)))
    y0 = max(0, int(round((min(b[1] for b in boxes) - pad) * h)))
    x1 = min(w, int(round((max(b[2] for b in boxes) + pad) * w)))
    y1 = min(h, int(round((max(b[3] for b in boxes) + pad) * h)))
    if x1 - x0 < 24 or y1 - y0 < 16:
        return None
    return (x0, y0, x1, y1)


def extract_escalation_regions(
    vision_res: OCRResult | None,
    image_size: tuple[int, int],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.025,
    max_regions: int = 4,
    max_total_region_area_frac: float = 0.58,
    min_prose_lines: int = 4,
    allow_formula_regions: bool = False,
) -> tuple[list[PageSegment], bool]:
    """Partition an ANE `OCRResult` into reading-order prose segments and cropped table/formula VLM regions.

    Returns `(segments, should_use_region_routing)`. By default (`allow_formula_regions=False`), pages
    containing mathematical formula bands return `([], False)` so math-heavy pages escalate full-page
    to the VLM (inline LaTeX inside prose degrades sharply if left on ANE). Also returns `([], False)`
    when complex regions cover `>= max_total_region_area_frac`, exceed `max_regions`, or leave fewer
    than `min_prose_lines` clean prose lines on ANE.
    """
    if vision_res is None or not vision_res.lines:
        return [], False
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return [], False

    ordered_lines = _xy_cut_order(vision_res.lines)
    heights = [ln.box.height for ln in ordered_lines if ln.box.height > 0.0]
    med_h = float(np.median(heights)) if heights else 0.015
    rows = _group_visual_rows(ordered_lines, med_h)
    if not rows:
        return [], False

    # Classify each visual row, bridging a 1-row gap inside a table block (e.g. sub-heading row).
    row_kinds: list[RegionKind] = []
    for row in rows:
        if _row_is_table_like(row):
            row_kinds.append("table")
        elif _row_is_formula_like(row):
            row_kinds.append("formula")
        else:
            row_kinds.append("prose")

    for i in range(1, len(row_kinds) - 1):
        if row_kinds[i] == "prose" and row_kinds[i - 1] == row_kinds[i + 1] != "prose":
            prev_y1 = max(l.box.y1 for l in rows[i - 1])
            next_y0 = min(l.box.y0 for l in rows[i + 1])
            if (next_y0 - prev_y1) <= max(0.06, med_h * 4.0):
                row_kinds[i] = row_kinds[i - 1]

    # Require at least 2 consecutive rows for a table region; single isolated table-like row stays prose.
    i = 0
    while i < len(row_kinds):
        if row_kinds[i] == "table":
            j = i
            while j < len(row_kinds) and row_kinds[j] == "table":
                j += 1
            if (j - i) < 2:
                row_kinds[i] = "prose"
            i = j
        else:
            i += 1

    # Merge contiguous rows of identical kind into ordered segments.
    grouped: list[tuple[RegionKind, list[OCRLine]]] = []
    for kind, row in zip(row_kinds, rows):
        if grouped and grouped[-1][0] == kind:
            grouped[-1][1].extend(row)
        else:
            grouped.append((kind, list(row)))

    vlm_groups = [g for g in grouped if g[0] != "prose"]
    prose_line_count = sum(len(lines) for kind, lines in grouped if kind == "prose")
    if not allow_formula_regions and any(g[0] == "formula" for g in vlm_groups):
        return [], False
    if not vlm_groups or len(vlm_groups) > int(max_regions) or prose_line_count < int(min_prose_lines):
        return [], False

    segments: list[PageSegment] = []
    total_region_area = 0.0
    page_area = float(w * h)
    for kind, seg_lines in grouped:
        if kind == "prose":
            md = synthesize_markdown_from_lines(seg_lines).strip()
            if md:
                segments.append(PageSegment(kind="prose", markdown=md, line_count=len(seg_lines)))
        else:
            box = _lines_to_px_box(seg_lines, (w, h), rotation_cw=rotation_cw, pad_frac=pad_frac)
            if box is None:
                return [], False
            total_region_area += float((box[2] - box[0]) * (box[3] - box[1]))
            prompt = REGION_TABLE_PROMPT if kind == "table" else REGION_FORMULA_PROMPT
            segments.append(
                PageSegment(
                    kind=kind,
                    crop_box_px=box,
                    prompt=prompt,
                    line_count=len(seg_lines),
                )
            )

    if (total_region_area / max(page_area, 1.0)) >= float(max_total_region_area_frac):
        return [], False
    return segments, True


def merge_region_markdown(
    segments: Sequence[PageSegment],
    region_outputs: Sequence[str],
    *,
    table_mode: MarkdownTableMode = "gfm",
) -> str:
    """Splice VLM outputs for escalated regions back into ANE prose segments in exact reading order."""
    parts: list[str] = []
    r_idx = 0
    for seg in segments:
        if seg.kind == "prose":
            if seg.markdown and seg.markdown.strip():
                parts.append(seg.markdown.strip())
        else:
            if r_idx >= len(region_outputs):
                raise ValueError(f"Expected VLM output for region {r_idx}, only received {len(region_outputs)}")
            txt = (region_outputs[r_idx] or "").strip()
            r_idx += 1
            if txt:
                parts.append(txt)
    if r_idx != len(region_outputs):
        raise ValueError(f"Unused VLM region outputs: consumed {r_idx} of {len(region_outputs)}")
    return canonicalize_markdown("\n\n".join(parts), table_mode=table_mode)
