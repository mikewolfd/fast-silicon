"""Zero-MLX adaptive visual budgeting, reading-order PLD seeding, and region-only VLM escalation routing.

Distills model-agnostic document-parsing patterns (`MinerU2.5` / `PaddleOCR-VL` region escalation,
`GLM-OCR` / `DeepSeek-OCR-2` adaptive visual-token budgets, and `olmOCR-2` reading-order / native-text
PLD anchoring) into pure Python + NumPy helpers usable with `AppleVisionOCR` + `FastOvisOCREngine`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Sequence

import numpy as np

from .markdown_struct import (
    _group_visual_rows,
    _xy_cut_order,
    canonicalize_markdown,
    synthesize_markdown_from_lines,
)
from .preprocess import RenderedPDFPage, rotate_box_normalized, smart_resize_qwen
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


_ISOLATED_DELIMITER_TEXTS = frozenset({"(", ")", "[", "]", "{", "}", "|", "§", "•", "·", "-", "–", "—"})
_PURE_NUMERIC_CELL_RE = re.compile(r"^[\s\$€£¥₹%+\-–—.,:/()0-9]+$")


def _filter_layout_noise_lines(lines: Sequence[OCRLine]) -> list[OCRLine]:
    """Exclude vertical margin stamps and court-caption delimiter tokens before gutter analysis."""
    kept: list[OCRLine] = []
    for ln in lines:
        txt = ln.text.strip()
        if not txt or txt in _ISOLATED_DELIMITER_TEXTS:
            continue
        w = ln.box.width
        h = ln.box.height
        if w > 0.0 and h > 0.0 and w < 0.035 and h > 3.0 * w:
            continue
        kept.append(ln)
    return kept


def _is_narrative_prose_line(ln: OCRLine) -> bool:
    txt = ln.text.strip()
    if not txt or len(txt.split()) < 4:
        return False
    non_ws = sum(1 for c in txt if not c.isspace())
    if non_ws < 12:
        return False
    digits = sum(1 for c in txt if c.isdigit())
    alpha = sum(1 for c in txt if c.isalpha())
    if alpha < 8 or (digits / float(non_ws)) >= 0.28:
        return False
    return float(ln.confidence) >= 0.72


def _is_narrative_prose_row(row: Sequence[OCRLine]) -> bool:
    joined = " ".join(ln.text.strip() for ln in row if ln.text.strip())
    if not joined or len(joined.split()) < 4:
        return False
    non_ws = sum(1 for c in joined if not c.isspace())
    if non_ws < 14:
        return False
    digits = sum(1 for c in joined if c.isdigit())
    alpha = sum(1 for c in joined if c.isalpha())
    if alpha < 10 or (digits / float(non_ws)) >= 0.28:
        return False
    mean_conf = float(np.mean([float(ln.confidence) for ln in row]))
    return mean_conf >= 0.72


def _find_vertical_gutter(items: Sequence[OCRLine], *, min_gutter: float = 0.025) -> float | None:
    if len(items) < 4:
        return None
    min_x = min(l.box.x0 for l in items)
    max_x = max(l.box.x1 for l in items)
    if (max_x - min_x) <= 0.35:
        return None
    n_bins = 80
    occ = np.zeros(n_bins, dtype=np.int32)
    for l in items:
        b0 = max(0, min(n_bins - 1, int(l.box.x0 * n_bins)))
        b1 = max(b0 + 1, min(n_bins, math.ceil(l.box.x1 * n_bins)))
        occ[b0:b1] += 1
    lo = int(0.20 * n_bins)
    hi = int(0.80 * n_bins)
    best_start, best_len = -1, 0
    cur_start, cur_len = -1, 0
    for b in range(lo, hi):
        if occ[b] == 0:
            if cur_start < 0:
                cur_start = b
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_start, cur_len = -1, 0
    if (best_len / n_bins) < float(min_gutter):
        return None
    return (best_start + 0.5 * best_len) / n_bins


def _detect_multi_column_prose(lines: Sequence[OCRLine], *, min_gutter: float = 0.025) -> bool:
    """Detect whether clean lines contain 2+ side-by-side narrative prose columns."""
    clean = _filter_layout_noise_lines(lines)
    if len(clean) < 8:
        return False

    def _has_parallel_prose_split(subset: Sequence[OCRLine], depth: int = 0) -> bool:
        if len(subset) < 8 or depth > 4:
            return False
        split_x = _find_vertical_gutter(subset, min_gutter=min_gutter)
        if split_x is not None:
            left = [l for l in subset if l.box.x1 <= split_x + 0.008]
            right = [l for l in subset if l.box.x0 >= split_x - 0.008]
            if len(left) >= 4 and len(right) >= 4 and (len(left) + len(right)) == len(subset):
                left_prose = sum(1 for l in left if _is_narrative_prose_line(l))
                right_prose = sum(1 for l in right if _is_narrative_prose_line(l))
                if left_prose >= 4 and right_prose >= 4:
                    y_ov = min(max(l.box.y1 for l in left), max(l.box.y1 for l in right)) - max(
                        min(l.box.y0 for l in left), min(l.box.y0 for l in right)
                    )
                    if y_ov >= 0.04:
                        return True
        heights = [l.box.height for l in subset if l.box.height > 0.0]
        med_h = float(np.median(heights)) if heights else 0.015
        by_y = sorted(subset, key=lambda l: (l.box.y0, l.box.x0))
        running_y1 = by_y[0].box.y1
        best_gap = 0.0
        best_split = -1
        min_h_gap = max(0.014, med_h * 1.1)
        for k in range(1, len(by_y)):
            gap = by_y[k].box.y0 - running_y1
            if gap >= min_h_gap and gap > best_gap:
                best_gap = gap
                best_split = k
            if by_y[k].box.y1 > running_y1:
                running_y1 = by_y[k].box.y1
        if best_split > 0:
            return _has_parallel_prose_split(by_y[:best_split], depth + 1) or _has_parallel_prose_split(
                by_y[best_split:], depth + 1
            )
        # Check same-band fragmented 2-col academic lines when inline spans occasionally touch center
        left_half = [l for l in subset if l.box.x1 <= 0.51]
        right_half = [l for l in subset if l.box.x0 >= 0.49]
        if len(left_half) >= 6 and len(right_half) >= 6:
            lp = sum(1 for l in left_half if _is_narrative_prose_line(l))
            rp = sum(1 for l in right_half if _is_narrative_prose_line(l))
            if lp >= 5 and rp >= 5:
                y_ov = min(max(l.box.y1 for l in left_half), max(l.box.y1 for l in right_half)) - max(
                    min(l.box.y0 for l in left_half), min(l.box.y0 for l in right_half)
                )
                if y_ov >= 0.12:
                    return True
        return False

    return _has_parallel_prose_split(clean)


def _should_split_vertical_gutter(left: Sequence[OCRLine], right: Sequence[OCRLine]) -> bool:
    """Split on a vertical gutter only for multi-column prose, never across aligned table columns."""
    if len(left) < 4 or len(right) < 4:
        return False
    left_narr = sum(1 for l in left if _is_narrative_prose_line(l))
    right_narr = sum(1 for l in right if _is_narrative_prose_line(l))
    return left_narr >= 3 and right_narr >= 3


def _xy_cut_leaf_rows(
    lines: Sequence[OCRLine],
    med_h: float,
    *,
    min_gutter: float = 0.03,
) -> list[list[OCRLine]]:
    """Group lines into visual rows within each XY-cut leaf so columns never bleed horizontally."""
    items = list(lines)
    if not items:
        return []
    if len(items) <= 2:
        return _group_visual_rows(items, med_h)

    split_x = _find_vertical_gutter(items, min_gutter=min_gutter)
    if split_x is not None:
        left = [l for l in items if l.box.x1 <= split_x + 0.005]
        right = [l for l in items if l.box.x0 >= split_x - 0.005]
        if (len(left) + len(right)) == len(items) and _should_split_vertical_gutter(left, right):
            return _xy_cut_leaf_rows(left, med_h, min_gutter=min_gutter) + _xy_cut_leaf_rows(
                right, med_h, min_gutter=min_gutter
            )

    by_y = sorted(items, key=lambda l: (l.box.y0, l.box.x0))
    running_y1 = by_y[0].box.y1
    best_gap = 0.0
    best_split = -1
    min_h_gap = max(0.014, med_h * 1.1)
    for k in range(1, len(by_y)):
        gap = by_y[k].box.y0 - running_y1
        if gap >= min_h_gap and gap > best_gap:
            best_gap = gap
            best_split = k
        if by_y[k].box.y1 > running_y1:
            running_y1 = by_y[k].box.y1
    if best_split > 0:
        return _xy_cut_leaf_rows(by_y[:best_split], med_h, min_gutter=min_gutter) + _xy_cut_leaf_rows(
            by_y[best_split:], med_h, min_gutter=min_gutter
        )

    return _group_visual_rows(items, med_h)


def _lines_to_norm_box(
    lines: Sequence[OCRLine],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.025,
) -> tuple[float, float, float, float] | None:
    if not lines:
        return None
    rot = int(rotation_cw) % 360
    boxes = [
        rotate_box_normalized((ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1), rot)
        if rot != 0
        else (ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1)
        for ln in lines
    ]
    pad = max(0.0, float(pad_frac))
    nx0 = max(0.0, min(b[0] for b in boxes) - pad)
    ny0 = max(0.0, min(b[1] for b in boxes) - pad)
    nx1 = min(1.0, max(b[2] for b in boxes) + pad)
    ny1 = min(1.0, max(b[3] for b in boxes) + pad)
    if (nx1 - nx0) <= 0.01 or (ny1 - ny0) <= 0.01:
        return None
    return (nx0, ny0, nx1, ny1)


def _lines_to_px_box(
    lines: Sequence[OCRLine],
    image_size: tuple[int, int],
    *,
    rotation_cw: int = 0,
    pad_frac: float = 0.025,
) -> tuple[int, int, int, int] | None:
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return None
    norm = _lines_to_norm_box(lines, rotation_cw=rotation_cw, pad_frac=pad_frac)
    if norm is None:
        return None
    x0 = max(0, int(round(norm[0] * w)))
    y0 = max(0, int(round(norm[1] * h)))
    x1 = min(w, int(round(norm[2] * w)))
    y1 = min(h, int(round(norm[3] * h)))
    if x1 - x0 < 24 or y1 - y0 < 16:
        return None
    return (x0, y0, x1, y1)


def _classify_rows_and_group(
    lines: Sequence[OCRLine],
) -> tuple[list[tuple[RegionKind, list[OCRLine]]], list[RegionKind], list[list[OCRLine]], float]:
    ordered_lines = _xy_cut_order(lines)
    heights = [ln.box.height for ln in ordered_lines if ln.box.height > 0.0]
    med_h = float(np.median(heights)) if heights else 0.015
    rows = _xy_cut_leaf_rows(lines, med_h)
    if not rows:
        return [], [], [], med_h

    row_kinds: list[RegionKind] = []
    for row in rows:
        if _row_is_table_like(row):
            row_kinds.append("table")
        elif _row_is_formula_like(row):
            row_kinds.append("formula")
        else:
            row_kinds.append("prose")

    # Bridge single prose/sub-header row sandwiched tightly inside a table or formula band.
    for i in range(1, len(row_kinds) - 1):
        if row_kinds[i] == "prose" and row_kinds[i - 1] == row_kinds[i + 1] != "prose":
            prev_y1 = max(l.box.y1 for l in rows[i - 1])
            next_y0 = min(l.box.y0 for l in rows[i + 1])
            if 0.0 <= (next_y0 - prev_y1) <= max(0.06, med_h * 4.0):
                row_kinds[i] = row_kinds[i - 1]

    # Require at least 2 consecutive rows for a table region; isolated multi-span prose stays prose.
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

    grouped: list[tuple[RegionKind, list[OCRLine]]] = []
    for kind, row in zip(row_kinds, rows):
        if grouped and grouped[-1][0] == kind:
            grouped[-1][1].extend(row)
        else:
            grouped.append((kind, list(row)))
    return grouped, row_kinds, rows, med_h


def _format_prose_segment_lines(seg_lines: Sequence[OCRLine]) -> str:
    """Format native/Vision lines in a prose band into paragraph-preserved markdown."""
    ordered = _xy_cut_order(seg_lines)
    if not ordered:
        return ""
    return synthesize_markdown_from_lines(ordered).strip()


def _passes_region_guards(
    grouped: Sequence[tuple[RegionKind, list[OCRLine]]],
    row_kinds: Sequence[RegionKind],
    rows: Sequence[Sequence[OCRLine]],
    *,
    max_regions: int,
    min_prose_lines: int,
    max_vlm_row_frac: float,
    allow_formula_regions: bool,
) -> tuple[bool, str]:
    """Validate that VLM regions are compact, non-fragmented, and leave enough clean prose."""
    vlm_groups = [g for g in grouped if g[0] != "prose"]
    if not vlm_groups:
        return False, "single_column_clean"
    if not allow_formula_regions and any(g[0] == "formula" for g in vlm_groups):
        return False, "formula_or_complex"
    if len(vlm_groups) > int(max_regions):
        return False, "table_heavy"

    prose_line_count = sum(len(lines) for kind, lines in grouped if kind == "prose")
    vlm_line_count = sum(len(lines) for kind, lines in grouped if kind != "prose")
    total_line_count = prose_line_count + vlm_line_count
    narrative_prose_rows = sum(
        1 for kind, row in zip(row_kinds, rows) if kind == "prose" and _is_narrative_prose_row(row)
    )
    vlm_rows = sum(1 for kind in row_kinds if kind != "prose")
    if (
        prose_line_count < int(min_prose_lines)
        or narrative_prose_rows < 3
        or (vlm_rows / max(1, len(row_kinds))) >= float(max_vlm_row_frac)
        or (vlm_line_count / max(1, total_line_count)) >= 0.65
    ):
        return False, "table_heavy"

    if len(vlm_groups) >= 2:
        vlm_indices = [idx_g for idx_g, (k, _ls) in enumerate(grouped) if k != "prose"]
        first_v, last_v = vlm_indices[0], vlm_indices[-1]
        for mid_i in range(first_v + 1, last_v):
            k_mid, ls_mid = grouped[mid_i]
            if k_mid == "prose":
                mid_txt = " ".join(l.text.strip() for l in ls_mid if l.text.strip())
                non_ws_m = sum(1 for c in mid_txt if not c.isspace())
                dig_m = sum(1 for c in mid_txt if c.isdigit())
                if non_ws_m > 0 and (dig_m / float(non_ws_m)) >= 0.30 and len(ls_mid) >= 3:
                    return False, "table_heavy"
        y0_span = min(l.box.y0 for idx_v in vlm_indices for l in grouped[idx_v][1])
        y1_span = max(l.box.y1 for idx_v in vlm_indices for l in grouped[idx_v][1])
        if (y1_span - y0_span) >= 0.65:
            return False, "table_heavy"

    return True, "splice_tables"


def analyze_digital_pdf_splice(
    page: RenderedPDFPage,
    *,
    max_regions: int = 4,
    max_total_region_area_frac: float = 0.58,
    min_prose_lines: int = 4,
    max_vlm_row_frac: float = 0.55,
    allow_formula_regions: bool = False,
) -> tuple[list[tuple[RegionKind, str | None, tuple[float, float, float, float] | None, str | None, int]], str]:
    """Classify a clean digital PDF page for native fast-path, VLM table crop splice, or full-page escalation.

    Returns `(segment_specs, disposition)` where `disposition` is one of:
      - `"not_eligible"`: missing lines, control/garbled text, or embedded images (`image_count > 0`)
      - `"multi_column"`: genuine multi-column prose detected -> escalate cleanly full-page
      - `"single_column_clean"`: verified clean single-column prose with zero table regions -> pure native fast path
      - `"splice_tables"`: clean single-column prose + compact embedded table(s) -> native prose + VLM crops
      - `"table_heavy"` / `"formula_or_complex"`: full-page / dense table or math page -> escalate full-page
    Each `segment_specs` element is `(kind, markdown, norm_box, prompt, line_count)`.
    """
    if (
        page.native_control_char_count > 0
        or page.native_garbled_ratio >= 0.02
        or page.image_count > 0
        or not page.lines
    ):
        return [], "not_eligible"

    clean_lines = _filter_layout_noise_lines(page.lines)
    if len(clean_lines) < 4:
        return [], "not_eligible"
    if _detect_multi_column_prose(clean_lines):
        return [], "multi_column"

    grouped, row_kinds, rows, _med_h = _classify_rows_and_group(clean_lines)
    if not grouped:
        return [], "not_eligible"

    ok_guards, reason = _passes_region_guards(
        grouped,
        row_kinds,
        rows,
        max_regions=max_regions,
        min_prose_lines=min_prose_lines,
        max_vlm_row_frac=max_vlm_row_frac,
        allow_formula_regions=allow_formula_regions,
    )
    if not ok_guards:
        return [], reason

    specs: list[tuple[RegionKind, str | None, tuple[float, float, float, float] | None, str | None, int]] = []
    total_norm_area = 0.0
    prev_table_y1 = -1.0
    for kind, seg_lines in grouped:
        if kind == "prose":
            md = _format_prose_segment_lines(seg_lines)
            if md:
                specs.append(("prose", md, None, None, len(seg_lines)))
        else:
            nbox = _lines_to_norm_box(seg_lines, pad_frac=0.025)
            if nbox is None or (prev_table_y1 >= 0.0 and nbox[1] < prev_table_y1 - 0.015):
                return [], "table_heavy"
            prev_table_y1 = max(prev_table_y1, nbox[3])
            total_norm_area += (nbox[2] - nbox[0]) * (nbox[3] - nbox[1])
            prompt = REGION_TABLE_PROMPT if kind == "table" else REGION_FORMULA_PROMPT
            specs.append((kind, None, nbox, prompt, len(seg_lines)))

    if total_norm_area >= float(max_total_region_area_frac):
        return [], "table_heavy"
    if not any(s[0] != "prose" for s in specs):
        return [], "single_column_clean"
    return specs, "splice_tables"


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
    allow_multi_column: bool = False,
) -> tuple[list[PageSegment], bool]:
    """Partition an ANE `OCRResult` into reading-order prose segments and cropped table/formula VLM regions.

    Returns `(segments, should_use_region_routing)`. Declines formula-bearing pages (by default),
    multi-column prose pages (unless `allow_multi_column=True`), low-confidence/table-dominant
    scans, and sparse-prose layouts so those escalate full-page to the VLM without interleaving
    columns or clipping table headers.
    """
    if vision_res is None or not vision_res.lines:
        return [], False
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return [], False

    if vision_res.analysis is not None:
        reasons = set(vision_res.analysis.escalation_reasons)
        if "low_vision_confidence" in reasons or "tiny_font_scan" in reasons:
            return [], False

    if not allow_multi_column and _detect_multi_column_prose(vision_res.lines):
        return [], False

    grouped, row_kinds, rows, _med_h = _classify_rows_and_group(vision_res.lines)
    if not grouped:
        return [], False

    ok_guards, _reason = _passes_region_guards(
        grouped,
        row_kinds,
        rows,
        max_regions=max_regions,
        min_prose_lines=min_prose_lines,
        max_vlm_row_frac=0.55,
        allow_formula_regions=allow_formula_regions,
    )
    if not ok_guards:
        return [], False

    segments: list[PageSegment] = []
    total_region_area = 0.0
    page_area = float(w * h)
    prev_table_y1 = -1
    for kind, seg_lines in grouped:
        if kind == "prose":
            md = synthesize_markdown_from_lines(seg_lines).strip()
            if md:
                segments.append(PageSegment(kind="prose", markdown=md, line_count=len(seg_lines)))
        else:
            box = _lines_to_px_box(seg_lines, (w, h), rotation_cw=rotation_cw, pad_frac=pad_frac)
            if box is None or (prev_table_y1 >= 0 and box[1] < prev_table_y1 - 8):
                return [], False
            prev_table_y1 = max(prev_table_y1, box[3])
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
