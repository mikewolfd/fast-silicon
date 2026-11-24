"""Image loading, PDF rendering, ink-tight cropping, orientation rotation, and Qwen3.5 vision patchification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence

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
    re.compile(r"^[\s\-_=–—·•*]+$"),
    re.compile(r"^\s*[A-Z]{2,8}[-_]\d{4,10}\s*$"),
    re.compile(r"^\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\s*$"),
)


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
