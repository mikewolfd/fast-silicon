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

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_MAX_PDF_PIXELS = 20_000_000

_CONTROL_GARBLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")
_STAMP_BOILERPLATE_RES = (
    re.compile(r"^\s*VerDate\s+\S+.*?\bFrm\s+\S+.*?\bFmt\s+\S+.*$", re.IGNORECASE),
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
    effective_dpi: int | None = None
    native_control_char_count: int = 0
    native_body_char_count: int | None = None


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


def extract_native_pdf_text(
    pdf_path: str | Path,
    *,
    page_indices: Sequence[int] | None = None,
) -> list[RenderedPDFPage]:
    """Extract per-page native text metadata from a PDF via `pypdfium2` without rasterizing."""
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
                finally:
                    tp.close()
                stripped = raw_txt.strip()
                body_stripped = strip_running_pdf_boilerplate(stripped)
                out.append(
                    RenderedPDFPage(
                        page_index=idx,
                        width_pt=float(w_pt),
                        height_pt=float(h_pt),
                        native_text=stripped,
                        native_char_count=sum(1 for c in stripped if not c.isspace()),
                        native_garbled_ratio=_garbled_text_ratio(stripped),
                        image=None,
                        image_count=_count_pdf_page_images(page),
                        native_control_char_count=count_control_garble_chars(stripped),
                        native_body_char_count=sum(1 for c in body_stripped if not c.isspace()),
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
                ctrl_count = 0
                body_count: int | None = None
                if include_native_text:
                    tp = page.get_textpage()
                    try:
                        stripped = (tp.get_text_bounded() or "").strip()
                    finally:
                        tp.close()
                    img_count = _count_pdf_page_images(page)
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
