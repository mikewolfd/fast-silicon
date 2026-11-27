"""Image loading, PDF rendering, ink-tight cropping, orientation rotation, and Qwen3.5 vision patchification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from fast_shared.vision_preprocess import (
    QwenVisionInputs,
    build_qwen_vision_inputs,
    crop_to_ink,
    detect_ink_bbox,
    load_image,
    rotate_image_cw,
    smart_resize_qwen,
)

from .types import OCRBox, OCRLine

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_MAX_PDF_PIXELS = 20_000_000
DEFAULT_PDF_LINE_MERGE_GAP = 0.015

_CONTROL_GARBLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")
_STAMP_BOILERPLATE_RES = (
    re.compile(r"^\s*VerDate\s+\S+.*?\bFrm\s+\S+.*?\bFmt\s+\S+.*$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z]{2,12}[-_]\d{4,10}\s+with\s+(?:DISTILLER|PROD\S*)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Insert\s+offset\s+folio\s+\d+\s+here\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:page\s+)?\d{1,5}(?:\s*(?:of|/)\s*\d{1,5})?\s*$", re.IGNORECASE),
    re.compile(r"^[\s\-_=\u2013\u2014\u00b7\u2022*]+$"),
    re.compile(r"^\s*[A-Z]{2,8}[-_]\d{4,10}\s*$"),
    re.compile(r"^\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\s*$"),
)

_PAGE_NUM_ONLY_RE = re.compile(
    r"^\s*(?:[-\u2013\u2014\[\(]\s*)?(?:p(?:age|\.)?\s*)?(\d{1,5})(?:\s*(?:of|/)\s*\d{1,5})?(?:\s*[-\u2013\u2014\]\)])?\s*$",
    re.IGNORECASE,
)
_EQUATION_NUM_ONLY_RE = re.compile(r"^\s*[\(\[]\d{1,4}[\)\]]\s*$")
_BARE_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_HF_BOILERPLATE_RES = (
    re.compile(r"(?:\u00a9|\(c\)|\bcopyright\b|\ball\s+rights\s+reserved\b)", re.IGNORECASE),
    re.compile(r"\b(?:ISSN|ISBN)\s*[:#]?\s*\d{4}[\s\-X\d]*\b", re.IGNORECASE),
    re.compile(r"\b(?:biodiversity\s+heritage\s+library|biodiversitylibrary\.org|zobodat\.at)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:for\s+personal\s+use\s+only|authorized\s+licensed\s+use|downloaded\s+from\s+https?://|"
        r"pers\u00f6nlichen\s+gebrauch|persoenlichen\s+gebrauch|keiner\s+form\s+vervielf\u00e4ltigt|"
        r"dritte\s+weitergegeben)\b",
        re.IGNORECASE,
    ),
)
_LEADING_MEASURE_WORDS: frozenset[str] = frozenset({
    "mg", "g", "kg", "ml", "l", "cm", "mm", "m", "km", "px", "pt", "hz", "khz", "mhz",
    "pm", "am", "sec", "min", "hr", "hrs", "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "page", "pages", "item", "items", "row", "rows",
    "percent", "usd", "eur", "cases", "patients", "samples", "tables", "figures",
})
_MD_DECOR_EDGE_RE = re.compile(r"^(?:#{1,6}\s+|>\s+)")


@dataclass(frozen=True)
class RenderedPDFPage:
    """A single page extracted from a PDF with optional native text layer and raster image."""

    page_index: int
    width_pt: float
    height_pt: float
    native_text: str
    native_char_count: int
    native_garbled_ratio: float
    image: Image.Image | None
    image_count: int = 0
    path_count: int = 0
    effective_dpi: int | None = None
    native_control_char_count: int = 0
    native_body_char_count: int | None = None
    lines: tuple[OCRLine, ...] = ()


def rotate_box_normalized(
    box: tuple[float, float, float, float],
    degrees_cw: int,
) -> tuple[float, float, float, float]:
    """Rotate a normalized top-left `(x0, y0, x1, y1)` box clockwise."""
    deg = int(degrees_cw) % 360
    x0, y0, x1, y1 = box
    if deg == 0:
        return (x0, y0, x1, y1)
    if deg == 90:
        return (1.0 - y1, x0, 1.0 - y0, x1)
    if deg == 180:
        return (1.0 - x1, 1.0 - y1, 1.0 - x0, 1.0 - y0)
    if deg == 270:
        return (y0, 1.0 - x1, y1, 1.0 - x0)
    raise ValueError(f"Unsupported clockwise rotation angle: {degrees_cw}")


def count_control_garble_chars(text: str) -> int:
    """Count C0/C1 control characters (excluding \\t, \\n, \\r) and U+FFFD replacement glyphs."""
    if not text:
        return 0
    return len(_CONTROL_GARBLE_RE.findall(text))


def strip_running_pdf_boilerplate(text: str) -> str:
    """Strip standalone typesetting stamps, page numbers, separators, and BATES/date lines."""
    if not text:
        return ""
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pat.match(line) for pat in _STAMP_BOILERPLATE_RES):
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _norm_hf_cmp(text: str) -> str:
    """Normalize text for running header/footer comparison (lowercase, stripped markdown/punct)."""
    s = str(text or "").strip()
    while True:
        nxt = _MD_DECOR_EDGE_RE.sub("", s).strip()
        if nxt == s:
            break
        s = nxt
    s = s.strip("*_`~")
    return " ".join(re.findall(r"[a-z0-9\u00c0-\u024f]+", s.lower()))


def _is_isolated_page_num_token(tok: str, *, allow_year_like: bool = False) -> bool:
    t = str(tok or "").strip()
    if not _BARE_PAGE_NUM_RE.match(t):
        return False
    val = int(t)
    if val <= 0:
        return False
    if not allow_year_like and 1900 <= val <= 2099:
        return False
    return True


def _is_hf_boilerplate_line(clean_line: str) -> bool:
    if not clean_line or len(clean_line) > 320:
        return False
    if any(pat.search(clean_line) for pat in _HF_BOILERPLATE_RES):
        return True
    if re.search(r"\(\s*97[89]\d{10}\s*\)", clean_line):
        return True
    return False


def _is_standalone_page_number_line(clean_line: str) -> bool:
    if not clean_line or _EQUATION_NUM_ONLY_RE.match(clean_line):
        return False
    m = _PAGE_NUM_ONLY_RE.match(clean_line)
    if not m:
        return False
    num = int(m.group(1))
    return 1 <= num <= 99999


def _is_text_paired_header_line(clean_line: str, *, raw_line: str = "") -> bool:
    """Detect inline running header paired with an isolated page number (e.g. `Title   1`, `164 Adam A. ZYCH`)."""
    if not clean_line or len(clean_line) > 160 or "|" in clean_line:
        return False
    if _is_hf_boilerplate_line(clean_line) or _is_standalone_page_number_line(clean_line):
        return True
    # Multi-space separated `<header>   <page_num>` or `<page_num>   <header>`
    m_trail_gap = re.match(r"^(.+?)\s{2,}(\d{1,4})\s*$", clean_line)
    if m_trail_gap:
        body, num_s = m_trail_gap.group(1).strip(), m_trail_gap.group(2)
        if body and len(body) >= 3 and _is_isolated_page_num_token(num_s, allow_year_like=True):
            return True
    m_lead_gap = re.match(r"^(\d{1,4})\s{2,}(.+)$", clean_line)
    if m_lead_gap:
        num_s, body = m_lead_gap.group(1), m_lead_gap.group(2).strip()
        if body and len(body) >= 3 and _is_isolated_page_num_token(num_s, allow_year_like=True):
            return True

    toks = clean_line.split()
    if len(toks) < 2:
        return False
    # Leading bare page number + non-sentence running header (`164 Adam A. ZYCH`), never numbered section headings (`# 1 Grundlagen`)
    is_numbered_section_heading = bool(raw_line and re.match(r"^\s*#{1,6}\s+\d{1,2}\s+\S+$", raw_line.strip()))
    first, rest_first = toks[0], toks[1].lower().strip(".,;:")
    has_caps_running_tag = any(len(t) >= 2 and t.isupper() and t.isalpha() for t in toks[1:])
    if (
        not is_numbered_section_heading
        and _is_isolated_page_num_token(first, allow_year_like=False)
        and "." not in first
        and ")" not in first
        and (int(first) >= 10 or has_caps_running_tag)
        and rest_first not in _LEADING_MEASURE_WORDS
        and toks[1][0].isalpha()
        and toks[1][0].isupper()
        and not clean_line.endswith(("?", "!"))
        and len(toks) <= 10
        and len(toks) >= 3
    ):
        return True
    # Trailing bare page number (`Inhaltsverzeichnis 11`, `Programs of Graduate Study ... 1`)
    last = toks[-1]
    prefix = " ".join(toks[:-1]).strip()
    if (
        _is_isolated_page_num_token(last, allow_year_like=False)
        and len(prefix) >= 4
        and not prefix.endswith((".", ",", ";", ":", "?", "!", "=", "(", "[", "/"))
        and not re.search(r"(?:chapter|section|part|fig(?:ure)?|table|vol(?:ume)?|no|page|pp)\.?$", prefix, re.IGNORECASE)
        and prefix[0].isalpha()
        and len(toks) <= 14
    ):
        return True
    return False


def _extract_spatial_hf_candidates(
    vision_lines: Sequence[OCRLine],
    *,
    line_rotation_cw: int = 0,
) -> tuple[set[str], set[str]]:
    """Identify normalized top running-header and bottom running-footer texts from spatial page bands."""
    if not vision_lines:
        return set(), set()
    rot = int(line_rotation_cw) % 360
    oriented: list[OCRLine] = []
    for ln in vision_lines:
        txt = str(getattr(ln, "text", "") or "").strip()
        if not txt:
            continue
        b = getattr(ln, "box", None)
        if b is None:
            continue
        rx0, ry0, rx1, ry1 = rotate_box_normalized((float(b.x0), float(b.y0), float(b.x1), float(b.y1)), rot)
        oriented.append(OCRLine(text=txt, box=OCRBox(rx0, ry0, rx1, ry1), confidence=float(getattr(ln, "confidence", 1.0))))
    if len(oriented) < 5:
        return set(), set()

    heights = [max(0.004, ln.box.y1 - ln.box.y0) for ln in oriented]
    med_h = float(np.median(np.asarray(heights, dtype=np.float32)))
    min_y = min(ln.box.y0 for ln in oriented)
    max_y = max(ln.box.y1 for ln in oriented)
    span_y = max_y - min_y
    if span_y < 0.20:
        return set(), set()

    oriented.sort(key=lambda ln: (0.5 * (ln.box.y0 + ln.box.y1), ln.box.x0))
    bands: list[list[OCRLine]] = []
    for ln in oriented:
        cy = 0.5 * (ln.box.y0 + ln.box.y1)
        if bands:
            anchor = bands[-1][0]
            acy = 0.5 * (anchor.box.y0 + anchor.box.y1)
            y_tol = max(0.008, 0.55 * min(max(0.004, anchor.box.y1 - anchor.box.y0), max(0.004, ln.box.y1 - ln.box.y0)))
            if abs(cy - acy) <= y_tol:
                bands[-1].append(ln)
                continue
        bands.append([ln])
    if len(bands) < 3:
        return set(), set()

    for band in bands:
        band.sort(key=lambda ln: ln.box.x0)

    def _band_bounds(band: Sequence[OCRLine]) -> tuple[float, float]:
        return min(ln.box.y0 for ln in band), max(ln.box.y1 for ln in band)

    def _band_is_hf(band: Sequence[OCRLine], gap_to_body: float, *, is_top: bool) -> bool:
        joined = " ".join(ln.text for ln in band).strip()
        if not joined or len(joined) > 280:
            return False
        if _is_hf_boilerplate_line(joined) or _is_standalone_page_number_line(joined):
            return True
        if len(band) >= 2:
            max_dx = max(band[j + 1].box.x0 - band[j].box.x1 for j in range(len(band) - 1))
            first_tok = band[0].text.split()[0] if band[0].text.split() else ""
            last_tok = band[-1].text.split()[-1] if band[-1].text.split() else ""
            has_split_num = (
                _is_isolated_page_num_token(first_tok, allow_year_like=(max_dx >= 0.12))
                or _is_isolated_page_num_token(last_tok, allow_year_like=(max_dx >= 0.12))
            )
            if max_dx >= 0.05 and has_split_num and len(joined) <= 160:
                return True
        if is_top and gap_to_body >= max(0.012, 0.65 * med_h) and _is_text_paired_header_line(joined):
            return True
        return False

    top_cands: set[str] = set()
    bot_cands: set[str] = set()

    top_limit_y = min_y + 0.22 * span_y
    for bi in range(min(2, len(bands) - 1)):
        by0, by1 = _band_bounds(bands[bi])
        if by1 > top_limit_y:
            break
        nxt_y0, _ = _band_bounds(bands[bi + 1])
        gap = max(0.0, nxt_y0 - by1)
        if _band_is_hf(bands[bi], gap, is_top=True):
            joined_norm = _norm_hf_cmp(" ".join(ln.text for ln in bands[bi]))
            if joined_norm:
                top_cands.add(joined_norm)
            for ln in bands[bi]:
                sn = _norm_hf_cmp(ln.text)
                if sn:
                    top_cands.add(sn)
        elif bi == 0:
            break

    bot_limit_y = max_y - 0.22 * span_y
    for bi in range(len(bands) - 1, max(0, len(bands) - 4), -1):
        by0, by1 = _band_bounds(bands[bi])
        if by0 < bot_limit_y:
            break
        _, prv_y1 = _band_bounds(bands[bi - 1])
        gap = max(0.0, by0 - prv_y1)
        if _band_is_hf(bands[bi], gap, is_top=False):
            joined_norm = _norm_hf_cmp(" ".join(ln.text for ln in bands[bi]))
            if joined_norm:
                bot_cands.add(joined_norm)
            for ln in bands[bi]:
                sn = _norm_hf_cmp(ln.text)
                if sn:
                    bot_cands.add(sn)
        elif bi == len(bands) - 1:
            break

    return top_cands, bot_cands


def _matches_spatial_candidate(line_norm: str, candidates: set[str]) -> bool:
    if not line_norm or not candidates:
        return False
    if line_norm in candidates:
        return True
    line_toks = line_norm.split()
    if not line_toks:
        return False
    line_set = set(line_toks)
    for cand in candidates:
        cand_toks = cand.split()
        if not cand_toks:
            continue
        cand_set = set(cand_toks)
        overlap = len(line_set & cand_set)
        if overlap == 0:
            continue
        if (overlap / float(len(line_set))) >= 0.80 and (overlap / float(len(cand_set))) >= 0.65:
            return True
        if line_norm in cand and len(line_toks) >= 2 and (len(line_toks) / float(len(cand_toks))) >= 0.50:
            return True
    return False


def strip_running_headers_footers(
    markdown_text: str,
    *,
    vision_lines: Sequence[OCRLine] | None = None,
    line_rotation_cw: int = 0,
) -> str:
    """Strip likely running page headers and footers from leading/trailing OCR Markdown edges.

    Preserves tables, code fences, display math, equation numbers `(2)`, titles without running
    page-number markers, and body prose. Never empties a non-empty document.
    """
    if not markdown_text or not markdown_text.strip():
        return markdown_text

    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    n = len(lines)
    protected = [False] * n
    in_fence: tuple[str, int] | None = None
    in_math = False
    for i, raw in enumerate(lines):
        st = raw.strip()
        if st.startswith(("```", "~~~")):
            marker = "```" if st.startswith("```") else "~~~"
            if in_fence is None:
                in_fence = (marker, len(st) - len(st.lstrip(marker[0])))
            elif marker == in_fence[0] and st.strip(marker[0]).strip() == "":
                in_fence = None
            protected[i] = True
            continue
        if in_fence is not None:
            protected[i] = True
            continue
        if st.startswith("$$") or st.startswith(r"\[") or st.endswith(r"\]"):
            if st == "$$" or (st.startswith("$$") and not st[2:].rstrip().endswith("$$")):
                in_math = not in_math
            protected[i] = True
            continue
        if in_math:
            protected[i] = True
            continue
        if st.startswith("|") or st.lower().startswith(("<table", "</table", "<tr", "<td", "<th", "<img")):
            protected[i] = True
            continue
        if _EQUATION_NUM_ONLY_RE.match(st) or (
            ("\\frac" in st or "\\sum" in st or "\\int" in st or "=" in st) and not _is_hf_boilerplate_line(st)
        ):
            protected[i] = True

    ne_indices = [i for i, raw in enumerate(lines) if raw.strip()]
    if len(ne_indices) <= 1:
        return markdown_text

    top_spatial, bot_spatial = (
        _extract_spatial_hf_candidates(vision_lines, line_rotation_cw=line_rotation_cw)
        if vision_lines
        else (set(), set())
    )

    drop_mask = [False] * n

    def _clean_edge(raw_line: str) -> str:
        s = raw_line.strip()
        while True:
            nxt = _MD_DECOR_EDGE_RE.sub("", s).strip()
            if nxt == s:
                break
            s = nxt
        return s.strip("*_`~").strip()

    # Scan top edge inward (max 4 non-empty lines); stop as soon as a kept body line is hit.
    for pos, idx in enumerate(ne_indices[:4]):
        if protected[idx]:
            break
        clean = _clean_edge(lines[idx])
        norm = _norm_hf_cmp(clean)
        is_hit = (
            _is_hf_boilerplate_line(clean)
            or _is_standalone_page_number_line(clean)
            or _is_text_paired_header_line(clean, raw_line=lines[idx])
            or _matches_spatial_candidate(norm, top_spatial)
        )
        if is_hit:
            drop_mask[idx] = True
        else:
            break

    # Scan bottom edge inward (max 5 non-empty lines); stop as soon as a kept body line is hit.
    for idx in reversed(ne_indices[-5:]):
        if drop_mask[idx]:
            continue
        if protected[idx]:
            break
        clean = _clean_edge(lines[idx])
        norm = _norm_hf_cmp(clean)
        # Do not drop numbered bibliography items in References sections unless explicit copyright/ISSN stamp.
        is_ref_entry = bool(re.match(r"^(?:\[\d+\]|\d+\.\s+[A-Z])", clean))
        is_hit = (
            _is_hf_boilerplate_line(clean)
            or _is_standalone_page_number_line(clean)
            or (not is_ref_entry and _matches_spatial_candidate(norm, bot_spatial))
        )
        if is_hit:
            drop_mask[idx] = True
        else:
            break

    kept = [raw for i, raw in enumerate(lines) if not drop_mask[i]]
    # Never delete every line of a non-empty transcription.
    if not any(ln.strip() for ln in kept):
        return markdown_text
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def normalize_faded_scan_image(
    source: str | Path | Image.Image | np.ndarray,
    *,
    white_target: float = 248.0,
) -> tuple[Image.Image, dict[str, Any] | None]:
    """Normalize low-contrast faded archival scans or gray-paper scans without touching clean pages or photos.

    Strictly skips small cropped tables (`FinTabNet`), camera photos (`CORD-v2`), high-chroma slides/charts,
    and clean digital pages. Preserves subtle stroke anti-aliasing via local background equalization + gentle
    contrast stretch (never hard binary thresholding).
    """
    from PIL import Image as _PILImage
    from PIL import ImageFilter as _PILImageFilter

    from .apple_vision import is_likely_camera_photo

    img = load_image(source)
    passthrough_img = source if (isinstance(source, _PILImage.Image) and source.size == img.size and source.mode == "RGB") else img
    w, h = img.size
    if min(w, h) < 350 or (w * h) < 200_000:
        return passthrough_img, None
    if is_likely_camera_photo(img):
        return passthrough_img, None

    raw = np.asarray(img, dtype=np.uint8)
    step = max(1, max(h, w) // 256)
    sub = raw[::step, ::step, :3].astype(np.float32)
    y_sub = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
    chroma = np.max(sub, axis=2) - np.min(sub, axis=2)
    ch75 = float(np.percentile(chroma, 75))
    if ch75 > 80.0:
        return passthrough_img, None

    sh, sw = y_sub.shape
    bw = max(2, int(round(min(sh, sw) * 0.04)))
    border = np.concatenate(
        [
            y_sub[:bw, :].ravel(),
            y_sub[-bw:, :].ravel(),
            y_sub[bw:-bw, :bw].ravel(),
            y_sub[bw:-bw, -bw:].ravel(),
        ]
    )
    center = y_sub[bw:-bw, bw:-bw] if (sh > 2 * bw and sw > 2 * bw) else y_sub
    p005, p02, p05, p50, p99 = (float(v) for v in np.percentile(y_sub, [0.5, 2.0, 5.0, 50.0, 99.0]))
    b50 = float(np.median(border))
    c50 = float(np.median(center))

    is_aged_faded = (p02 >= 50.0) and (p50 <= 235.0) and (p99 <= 245.0) and ((p99 - p005) < 175.0)
    is_faint_pencil = (p005 >= 75.0) and (p02 <= 175.0) and (p05 <= 235.0) and (p99 < 253.5) and ((p99 - p005) < 170.0)
    is_dark_paper = (b50 >= 225.0) and (120.0 <= c50 <= 205.0) and (p02 >= 30.0)

    if not (is_aged_faded or is_faint_pencil or is_dark_paper):
        return passthrough_img, None

    reason = "dark_paper" if is_dark_paper else ("aged_faded" if is_aged_faded else "faint_pencil")
    arr_f = raw.astype(np.float32)

    # Local paper illumination equalization when background is dark or unevenly shaded.
    if is_dark_paper or p50 < 238.0 or (p99 - p50) >= 20.0:
        bg_long = 256
        scale = float(bg_long) / float(max(w, h))
        bg_w = max(32, int(round(w * scale)))
        bg_h = max(32, int(round(h * scale)))
        gray_small = img.convert("L").resize((bg_w, bg_h), _PILImage.Resampling.BILINEAR)
        # MaxFilter traces local bright paper level over thick ink strokes; GaussianBlur smooths tile edges.
        env_small = gray_small.filter(_PILImageFilter.MaxFilter(size=15)).filter(_PILImageFilter.GaussianBlur(radius=8))
        env_full = env_small.resize((w, h), _PILImage.Resampling.BICUBIC)
        bg = np.maximum(np.asarray(env_full, dtype=np.float32), 105.0)
        gain = np.clip(float(white_target) / bg, 0.85, 2.20)[:, :, None]
        arr_f = np.clip(arr_f * gain, 0.0, 255.0)

    y_eq = 0.299 * arr_f[::step, ::step, 0] + 0.587 * arr_f[::step, ::step, 1] + 0.114 * arr_f[::step, ::step, 2]
    lo = float(np.percentile(y_eq, 0.3))
    hi = float(np.percentile(y_eq, 96.0))
    hi_eff = max(210.0, hi)
    lo_eff = max(12.0, min(lo, hi_eff - 32.0))
    inv_span = 1.0 / max(1.0, hi_eff - lo_eff)
    norm01 = np.clip((arr_f - lo_eff) * inv_span, 0.0, 1.0)
    # Mild gamma > 1 deepens faint ink strokes slightly while keeping bright paper clean.
    stretched = np.clip(np.power(norm01, 1.12) * 255.0, 0.0, 255.0).astype(np.uint8)
    out_img = _PILImage.fromarray(stretched, mode="RGB")
    meta = {
        "applied": True,
        "reason": reason,
        "p005": round(p005, 1),
        "p50": round(p50, 1),
        "p99": round(p99, 1),
    }
    return out_img, meta


def compute_adaptive_render_dpi(
    width_pt: float,
    height_pt: float,
    *,
    dpi: int = 300,
    max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
    min_dpi: int = 36,
) -> int:
    """Clamp `dpi` down to keep rendered page pixels within `max_pixels`, flooring at `min_dpi`."""
    target_dpi = int(dpi)
    if max_pixels is None or max_pixels <= 0 or width_pt <= 0.0 or height_pt <= 0.0 or target_dpi <= 0:
        return target_dpi
    raw_px = (float(width_pt) * target_dpi / 72.0) * (float(height_pt) * target_dpi / 72.0)
    if raw_px <= float(max_pixels):
        return target_dpi
    scaled = int(math.floor(target_dpi * math.sqrt(float(max_pixels) / raw_px)))
    floor_dpi = min(int(min_dpi), target_dpi)
    return max(floor_dpi, min(target_dpi, scaled))


def _count_pdf_page_images(page: object) -> int:
    """Count embedded image objects on a `pypdfium2` page without failing on unsupported builds."""
    try:
        import pypdfium2.raw as pdfium_raw

        objs = page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE])  # type: ignore[attr-defined]
        return sum(1 for _ in objs)
    except Exception:
        return 0


def _count_pdf_page_paths(page: object) -> int:
    """Count vector path objects on a `pypdfium2` page (detects vector-outlined glyph streams)."""
    try:
        import pypdfium2.raw as pdfium_raw

        objs = page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_PATH])  # type: ignore[attr-defined]
        return sum(1 for _ in objs)
    except Exception:
        return 0


def _garbled_text_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        code = ord(ch)
        if ch == "\ufffd" or (code < 32 and ch not in "\r\n\t") or (0x7F <= code <= 0x9F):
            bad += 1
    return bad / max(1, total)


def extract_native_pdf_lines(
    textpage: object,
    width_pt: float,
    height_pt: float,
    *,
    merge_gap: float = DEFAULT_PDF_LINE_MERGE_GAP,
) -> tuple[OCRLine, ...]:
    """Extract baseline-merged normalized `OCRLine` boxes from a `pypdfium2` text page."""
    count_fn = getattr(textpage, "count_rects", None)
    rect_fn = getattr(textpage, "get_rect", None)
    bounded_fn = getattr(textpage, "get_text_bounded", None)
    if not callable(count_fn) or not callable(rect_fn) or not callable(bounded_fn):
        return ()
    w_pt = max(float(width_pt), 1.0)
    h_pt = max(float(height_pt), 1.0)
    raw_lines: list[OCRLine] = []
    n_rects = int(count_fn())
    for i in range(n_rects):
        l, b, r, t = rect_fn(i)
        raw = bounded_fn(l, b, r, t) or ""
        parts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not parts:
            continue
        x0 = max(0.0, min(1.0, float(l) / w_pt))
        x1 = max(x0, min(1.0, float(r) / w_pt))
        y0 = max(0.0, min(1.0, 1.0 - float(t) / h_pt))
        y1 = max(y0, min(1.0, 1.0 - float(b) / h_pt))
        step = (y1 - y0) / float(len(parts))
        for k, part in enumerate(parts):
            ly0 = y0 + k * step
            ly1 = min(1.0, max(ly0, y0 + (k + 1) * step))
            raw_lines.append(OCRLine(text=part, box=OCRBox(x0, ly0, x1, ly1), confidence=1.0))

    if len(raw_lines) <= 1:
        return tuple(raw_lines)

    # Group same-baseline lines (handling descender height differences) before sorting left-to-right by `x0`
    # so taller text on the left is never sorted after shorter numbers on the right and merged backwards.
    raw_lines.sort(key=lambda ln: (0.5 * (ln.box.y0 + ln.box.y1), ln.box.x0))
    gap_tol = max(0.0, float(merge_gap))
    bands: list[list[OCRLine]] = []
    for ln in raw_lines:
        cy = 0.5 * (ln.box.y0 + ln.box.y1)
        if bands:
            anchor = bands[-1][0]
            acy = 0.5 * (anchor.box.y0 + anchor.box.y1)
            y_tol = max(0.006, 0.5 * min(anchor.box.height, ln.box.height))
            if abs(cy - acy) <= y_tol:
                bands[-1].append(ln)
                continue
        bands.append([ln])

    merged: list[OCRLine] = []
    for band in bands:
        band.sort(key=lambda ln: ln.box.x0)
        band_merged: list[OCRLine] = []
        for ln in band:
            if band_merged:
                prev = band_merged[-1]
                dx = ln.box.x0 - prev.box.x1
                if -0.015 <= dx <= gap_tol:
                    band_merged[-1] = OCRLine(
                        text=f"{prev.text} {ln.text}",
                        box=OCRBox(
                            min(prev.box.x0, ln.box.x0),
                            min(prev.box.y0, ln.box.y0),
                            max(prev.box.x1, ln.box.x1),
                            max(prev.box.y1, ln.box.y1),
                        ),
                        confidence=1.0,
                    )
                    continue
            band_merged.append(ln)
        merged.extend(band_merged)
    return tuple(merged)


def extract_native_pdf_text(
    pdf_path: str | Path,
    *,
    page_indices: Sequence[int] | None = None,
    include_lines: bool = True,
) -> list[RenderedPDFPage]:
    """Extract per-page native text metadata and optional merged `OCRLine` boxes from a PDF without rasterizing."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(Path(pdf_path).expanduser()))
    try:
        n_pages = len(doc)
        indices = list(range(n_pages)) if page_indices is None else [int(i) for i in page_indices]
        out: list[RenderedPDFPage] = []
        for idx in indices:
            if idx < 0 or idx >= n_pages:
                raise IndexError(f"Page index {idx} out of range for {pdf_path} ({n_pages} pages)")
            page = doc[idx]
            try:
                w_pt, h_pt = page.get_size()
                tp = page.get_textpage()
                try:
                    raw_txt = tp.get_text_bounded() or ""
                    stripped = raw_txt.strip()
                    body_stripped = strip_running_pdf_boilerplate(stripped)
                    char_count = sum(1 for c in stripped if not c.isspace())
                    garbled_ratio = _garbled_text_ratio(stripped)
                    ctrl_count = count_control_garble_chars(stripped)
                    body_count = sum(1 for c in body_stripped if not c.isspace())
                    page_lines: tuple[OCRLine, ...] = (
                        extract_native_pdf_lines(tp, float(w_pt), float(h_pt))
                        if (include_lines and ctrl_count == 0 and garbled_ratio < 0.02 and body_count > 0)
                        else ()
                    )
                finally:
                    tp.close()
                out.append(
                    RenderedPDFPage(
                        page_index=idx,
                        width_pt=float(w_pt),
                        height_pt=float(h_pt),
                        native_text=stripped,
                        native_char_count=char_count,
                        native_garbled_ratio=garbled_ratio,
                        image=None,
                        image_count=_count_pdf_page_images(page),
                        path_count=_count_pdf_page_paths(page),
                        native_control_char_count=ctrl_count,
                        native_body_char_count=body_count,
                        lines=page_lines,
                    )
                )
            finally:
                page.close()
        return out
    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: str | Path,
    *,
    dpi: int = 300,
    page_indices: Sequence[int] | None = None,
    include_native_text: bool = True,
    max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
) -> list[RenderedPDFPage]:
    """Render PDF pages to RGB `PIL.Image` at `dpi` (adaptively capped by `max_pixels`) via `pypdfium2`."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(Path(pdf_path).expanduser()))
    try:
        n_pages = len(doc)
        indices = list(range(n_pages)) if page_indices is None else [int(i) for i in page_indices]
        out: list[RenderedPDFPage] = []
        for idx in indices:
            if idx < 0 or idx >= n_pages:
                raise IndexError(f"Page index {idx} out of range for {pdf_path} ({n_pages} pages)")
            page = doc[idx]
            try:
                w_pt, h_pt = page.get_size()
                stripped = ""
                img_count = 0
                path_count = 0
                ctrl_count = 0
                body_count: int | None = None
                if include_native_text:
                    tp = page.get_textpage()
                    try:
                        stripped = (tp.get_text_bounded() or "").strip()
                    finally:
                        tp.close()
                    img_count = _count_pdf_page_images(page)
                    path_count = _count_pdf_page_paths(page)
                    ctrl_count = count_control_garble_chars(stripped)
                    body_count = sum(1 for c in strip_running_pdf_boilerplate(stripped) if not c.isspace())
                eff_dpi = compute_adaptive_render_dpi(
                    float(w_pt),
                    float(h_pt),
                    dpi=int(dpi),
                    max_pixels=max_pixels,
                )
                bmp = page.render(scale=float(eff_dpi) / 72.0, rotation=0)
                try:
                    pil_img = bmp.to_pil().convert("RGB")
                finally:
                    bmp.close()
                out.append(
                    RenderedPDFPage(
                        page_index=idx,
                        width_pt=float(w_pt),
                        height_pt=float(h_pt),
                        native_text=stripped,
                        native_char_count=sum(1 for c in stripped if not c.isspace()),
                        native_garbled_ratio=_garbled_text_ratio(stripped),
                        image=pil_img,
                        image_count=img_count,
                        path_count=path_count,
                        effective_dpi=eff_dpi,
                        native_control_char_count=ctrl_count,
                        native_body_char_count=body_count,
                    )
                )
            finally:
                page.close()
        return out
    finally:
        doc.close()
