"""Independent macOS Apple Vision (`VNRecognizeTextRequest`) ANE OCR backend (zero MLX dependencies)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import ctypes
import functools
import math
import os
from pathlib import Path
import plistlib
import platform
import re
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from .preprocess import crop_to_ink, detect_ink_bbox, load_image, rotate_box_normalized, rotate_image_cw
from .types import OCRBox, OCRLine, OCRResult, OCRTimings, OutputFormat, PageAnalysis

_BAND_OVERLAP = 0.02
_BAND_EDGE = 0.005
_CURRENCY_RE = re.compile(
    r"(?:\$|USD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b|\b\d{1,3}(?:,\d{3})+\.\d{2}\b|\b\d+\.\d{2}\b"
)


def _same_line(box_a: list[float], box_b: list[float]) -> bool:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    across = min(ax + aw, bx + bw) - max(ax, bx)
    down = min(ay + ah, by + bh) - max(ay, by)
    return across > min(aw, bw) * 0.5 and down > min(ah, bh) * 0.5


def _infer_rotation_cw_from_detections(
    detections: Sequence[tuple[str, float, list[float]]],
    *,
    vectors: Sequence[tuple[float, float]] | None = None,
) -> int:
    """Infer clockwise rotation needed (`0`, `90`, `180`, or `270`) from Vision detections.

    When oriented baseline vectors (`topLeft -> topRight` in pixel-scaled bottom-left coordinates) are
    available from `_NativeVisionRunner`, distinguishes all four cardinal orientations unambiguously.
    Otherwise falls back to axis-aligned aspect ratio (`h / w > 1.8 -> 90`) for synthetic engines.
    """
    if vectors is not None and len(vectors) == len(detections):
        sum_dx = 0.0
        sum_dy = 0.0
        qualifying = 0
        for (txt, _conf, _box), (dx, dy) in zip(detections, vectors):
            n_chars = len(txt.strip())
            norm = math.hypot(dx, dy)
            if n_chars >= 4 and norm > 1e-6:
                weight = float(min(40, n_chars))
                sum_dx += (dx / norm) * weight
                sum_dy += (dy / norm) * weight
                qualifying += 1
        if qualifying >= 3:
            if abs(sum_dy) > abs(sum_dx) * 1.25:
                return 90 if sum_dy > 0.0 else 270
            if sum_dx < -abs(sum_dy) * 1.25:
                return 180
            return 0

    ratios: list[float] = []
    for txt, _conf, (_x, _y, w, h) in detections:
        if len(txt.strip()) >= 4 and w > 1e-6 and h > 1e-6:
            ratios.append(h / w)
    if len(ratios) >= 4:
        med = float(np.median(np.asarray(ratios, dtype=np.float32)))
        if med > 1.8:
            return 90
    return 0


@functools.lru_cache(maxsize=1)
def _get_macos_vision_provenance() -> dict[str, str]:
    """Return cached macOS version and build strings without importing MLX."""
    info: dict[str, str] = {}
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        info["macos_version"] = mac_ver
    plist_path = Path("/System/Library/CoreServices/SystemVersion.plist")
    if plist_path.is_file():
        try:
            with plist_path.open("rb") as fh:
                plist = plistlib.load(fh)
            prod_ver = plist.get("ProductVersion")
            prod_build = plist.get("ProductBuildVersion")
            if isinstance(prod_ver, str) and prod_ver:
                info["macos_version"] = prod_ver
            if isinstance(prod_build, str) and prod_build:
                info["macos_build"] = prod_build
        except Exception:
            pass
    return info


class _NativeVisionRunner:
    """Single-worker PyObjC `VNRecognizeTextRequest` wrapper with zero `CGDataProviderCreateWithData` leak."""

    def __init__(
        self,
        *,
        level: str = "accurate",
        languages: tuple[str, ...] = ("en-US",),
        language_correction: bool = True,
    ) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("AppleVisionOCR requires macOS")
        import Foundation
        import Quartz
        import Vision
        import objc

        self.foundation = Foundation
        self.quartz = Quartz
        self.vision = Vision
        self.objc = objc

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(1 if level == "fast" else 0)
        supported = req.supportedRecognitionLanguagesAndReturnError_(None)[0]
        if not set(languages) <= set(supported):
            raise ValueError(f"Unsupported Apple Vision languages {languages}; supported={sorted(supported)}")
        req.setRecognitionLanguages_(list(languages))
        req.setUsesLanguageCorrection_(bool(language_correction))
        rev_fn = getattr(req, "revision", None)
        self.revision: int | None = int(rev_fn()) if callable(rev_fn) else None
        self.request = req
        self.last_vectors: list[tuple[float, float]] = []

    def _make_cgimage(self, rgb_bytes: bytes, width: int, height: int) -> Any:
        q = self.quartz
        ns_data = self.foundation.NSData.dataWithBytes_length_(rgb_bytes, len(rgb_bytes))
        provider = q.CGDataProviderCreateWithCFData(ns_data)
        return q.CGImageCreate(
            width,
            height,
            8,
            24,
            width * 3,
            q.CGColorSpaceCreateDeviceRGB(),
            q.kCGBitmapByteOrderDefault | q.kCGImageAlphaNone,
            provider,
            None,
            False,
            q.kCGRenderingIntentDefault,
        )

    def __call__(self, rgb_bytes: bytes, width: int, height: int) -> list[tuple[str, float, list[float]]]:
        with self.objc.autorelease_pool():
            handler = self.vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
                self._make_cgimage(rgb_bytes, width, height),
                None,
            )
            done = handler.performRequests_error_([self.request], None)
            ok, error = done if isinstance(done, tuple) else (bool(done), None)
            if not ok or error is not None:
                raise RuntimeError(f"Apple Vision recognition failed: {error}")
            out: list[tuple[str, float, list[float]]] = []
            vecs: list[tuple[float, float]] = []
            fw = float(width)
            fh = float(height)
            for res in self.request.results() or ():
                box = res.boundingBox()
                tl = res.topLeft()
                tr = res.topRight()
                out.append(
                    (
                        str(res.text()),
                        float(res.confidence()),
                        [float(box.origin.x), float(box.origin.y), float(box.size.width), float(box.size.height)],
                    )
                )
                vecs.append(((float(tr.x) - float(tl.x)) * fw, (float(tr.y) - float(tl.y)) * fh))
            self.last_vectors = vecs
            return out


def _recognize_banded_with_vectors(
    engine: Callable[[bytes, int, int], list[tuple[str, float, list[float]]]],
    rgb_bytes: bytes,
    width: int,
    height: int,
    *,
    bands: int = 1,
) -> tuple[list[tuple[str, float, list[float]]], list[tuple[float, float]] | None]:
    n_bands = max(1, int(bands))
    if n_bands == 1:
        dets = engine(rgb_bytes, width, height)
        raw_vecs = getattr(engine, "last_vectors", None)
        vecs = list(raw_vecs) if raw_vecs is not None and len(raw_vecs) == len(dets) else None
        return dets, vecs

    row_bytes = int(width) * 3
    reach = math.ceil(int(height) * _BAND_OVERLAP)
    strips: list[list[tuple[str, float, list[float]]]] = []
    strip_vecs: list[list[tuple[float, float]] | None] = []
    own: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    edge: list[tuple[int, int]] = []

    for i in range(n_bands):
        start = height * i / n_bands
        end = height * (i + 1) / n_bands
        top = max(0, math.floor(start) - reach)
        bottom = min(height, math.ceil(end) + reach)
        strip_h = bottom - top
        strip_bytes = rgb_bytes[top * row_bytes : bottom * row_bytes]
        found: list[tuple[str, float, list[float]]] = []
        raw_found = engine(strip_bytes, width, strip_h)
        raw_vecs = getattr(engine, "last_vectors", None)
        has_vecs = raw_vecs is not None and len(raw_vecs) == len(raw_found)
        strip_vecs.append(list(raw_vecs) if has_vecs else None)
        for text, conf, (x, y, w, h) in raw_found:
            upper = top + (1.0 - y - h) * strip_h
            lower = top + (1.0 - y) * strip_h
            middle = 0.5 * (upper + lower)
            full_box = [x, 1.0 - lower / height, w, (lower - upper) / height]
            j = len(found)
            found.append((text, conf, full_box))
            cut = (top > 0 and (upper - top) < height * _BAND_EDGE) or (
                bottom < height and (bottom - lower) < height * _BAND_EDGE
            )
            if start <= middle < end or (i == n_bands - 1 and middle == end):
                own.append((i, j))
            elif not cut:
                other.append((i, j))
            else:
                edge.append((i, j))
        strips.append(found)

    kept: set[tuple[int, int]] = set()
    for candidates, adjacent_only in ((own + other, True), (edge, False)):
        for i, j in candidates:
            box = strips[i][j][2]
            if not any(
                (abs(k - i) == 1 if adjacent_only else k != i) and _same_line(box, strips[k][m][2])
                for k, m in kept
            ):
                kept.add((i, j))

    all_have_vecs = all(v is not None for v in strip_vecs)
    out_dets: list[tuple[str, float, list[float]]] = []
    out_vecs: list[tuple[float, float]] = []
    for i, found in enumerate(strips):
        sv = strip_vecs[i]
        for j in range(len(found)):
            if (i, j) in kept:
                out_dets.append(found[j])
                if all_have_vecs and sv is not None:
                    out_vecs.append(sv[j])
    return out_dets, (out_vecs if all_have_vecs else None)


def recognize_banded_detections(
    engine: Callable[[bytes, int, int], list[tuple[str, float, list[float]]]],
    rgb_bytes: bytes,
    width: int,
    height: int,
    *,
    bands: int = 1,
) -> list[tuple[str, float, list[float]]]:
    """Run `engine` across `1` or more overlapping horizontal bands and deduplicate seam lines.

    Input `engine` returns bottom-left normalized `[x, y, w, h]`; output keeps that exact coordinate
    convention before normalization to top-left `OCRBox`.
    """
    dets, _vecs = _recognize_banded_with_vectors(engine, rgb_bytes, width, height, bands=bands)
    return dets


def analyze_page_from_lines(
    lines: Sequence[OCRLine],
    image: Image.Image | None,
    width: int,
    height: int,
    *,
    raw_detections: Sequence[tuple[str, float, list[float]]] = (),
    vectors: Sequence[tuple[float, float]] | None = None,
) -> PageAnalysis:
    """Compute layout complexity, ink crop bbox, orientation, and VLM escalation heuristics from Vision lines."""
    n_lines = len(lines)
    joined = "\n".join(ln.text for ln in lines)
    non_ws = sum(1 for c in joined if not c.isspace())
    digits = sum(1 for c in joined if c.isdigit())
    digit_ratio = digits / max(1, non_ws)
    currency_hits = len(_CURRENCY_RE.findall(joined))

    confs = [float(ln.confidence) for ln in lines]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    low_conf_ratio = (sum(1 for c in confs if c < 0.75) / max(1, len(confs))) if confs else 0.0
    heights = [ln.box.height for ln in lines if ln.box.height > 0.0]
    med_h = float(np.median(heights)) if heights else 0.0

    # Estimate horizontal band alignment & column structure
    x_starts = sorted(ln.box.x0 for ln in lines)
    estimated_columns = 1
    if len(lines) >= 6:
        # Count distinct horizontal bands that contain multiple horizontally separated detections
        sorted_y = sorted(lines, key=lambda ln: ln.box.y0)
        band_tol = max(0.008, med_h * 0.7)
        multi_cell_bands = 0
        max_cells_in_band = 1
        cur_band: list[OCRLine] = []
        for ln in sorted_y:
            cy = 0.5 * (ln.box.y0 + ln.box.y1)
            if not cur_band or abs(cy - 0.5 * (cur_band[0].box.y0 + cur_band[0].box.y1)) <= band_tol:
                cur_band.append(ln)
            else:
                if len(cur_band) >= 2:
                    # Check distinct non-overlapping x intervals
                    cur_sorted = sorted(cur_band, key=lambda c: c.box.x0)
                    sep = sum(
                        1
                        for idx in range(1, len(cur_sorted))
                        if cur_sorted[idx].box.x0 > cur_sorted[idx - 1].box.x1 + 0.015
                    ) + 1
                    if sep >= 2:
                        multi_cell_bands += 1
                        max_cells_in_band = max(max_cells_in_band, sep)
                cur_band = [ln]
        if len(cur_band) >= 2:
            cur_sorted = sorted(cur_band, key=lambda c: c.box.x0)
            sep = sum(
                1
                for idx in range(1, len(cur_sorted))
                if cur_sorted[idx].box.x0 > cur_sorted[idx - 1].box.x1 + 0.015
            ) + 1
            if sep >= 2:
                multi_cell_bands += 1
                max_cells_in_band = max(max_cells_in_band, sep)
        estimated_columns = max_cells_in_band if multi_cell_bands >= 3 else (2 if len(x_starts) >= 12 and (sum(1 for l in lines if l.box.x0 > 0.48) >= 4 and sum(1 for l in lines if l.box.x1 < 0.48) >= 4) else 1)
        table_score = min(1.0, (multi_cell_bands / 6.0) * 0.6 + min(1.0, currency_hits / 8.0) * 0.4)
    else:
        table_score = 0.0

    crop_box = detect_ink_bbox(image) if image is not None else None
    if crop_box is not None and width > 0 and height > 0:
        ink_area = ((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) / float(width * height)
    else:
        ink_area = 1.0

    inferred_rot = _infer_rotation_cw_from_detections(raw_detections, vectors=vectors) if raw_detections else 0

    reasons: list[str] = []
    if n_lines == 0 or non_ws < 10:
        reasons.append("empty_or_tiny_vision_text")
    if currency_hits >= 5 and table_score >= 0.35:
        reasons.append("financial_table")
    elif estimated_columns >= 3 and table_score >= 0.4:
        reasons.append("multi_column_table")
    if digit_ratio >= 0.28 and n_lines >= 10:
        reasons.append("high_digit_density")
    if low_conf_ratio >= 0.25 and n_lines >= 5:
        reasons.append("low_vision_confidence")
    if med_h > 0.0 and med_h < 0.0085 and n_lines >= 20:
        reasons.append("tiny_font_scan")
    if inferred_rot != 0:
        reasons.append("rotated_page")

    return PageAnalysis(
        width=int(width),
        height=int(height),
        line_count=n_lines,
        char_count=non_ws,
        digit_ratio=float(digit_ratio),
        currency_hits=int(currency_hits),
        mean_confidence=float(mean_conf),
        low_conf_line_ratio=float(low_conf_ratio),
        median_line_height=float(med_h),
        inferred_rotation_cw=int(inferred_rot),
        ink_crop_box_px=crop_box,
        ink_area_ratio=float(ink_area),
        estimated_columns=int(estimated_columns),
        table_score=float(table_score),
        needs_vlm=bool(reasons),
        escalation_reasons=tuple(reasons),
    )


def _worker_recognize_batch(args: tuple[str, int, bool, tuple[str, ...], str, tuple[str, ...] | list[str]]) -> list[dict[str, Any]]:
    """Top-level helper for multiprocessing multi-worker ANE scaling across `AppleVisionOCR`."""
    level, bands, language_correction, languages, output_format, paths = args
    ocr = AppleVisionOCR(
        level=level,
        bands=bands,
        language_correction=language_correction,
        languages=languages,
    )
    out: list[dict[str, Any]] = []
    for idx, p in enumerate(paths):
        res = ocr.recognize_image(p, page_index=idx, output_format=output_format)  # type: ignore[arg-type]
        out.append(
            {
                "text": res.text,
                "raw_text": res.raw_text,
                "page_index": idx,
                "total_ms": res.timings.total_ms,
                "valid": res.markdown_validation.valid if res.markdown_validation else True,
            }
        )
    return out


class AppleVisionOCR:
    """First-class independent macOS Neural Engine (`VNRecognizeTextRequest`) OCR provider.

    Imports ZERO MLX or VLM modules and allocates ZERO GPU weights. Safe to construct directly when
    using `--backend apple-vision`, or injected with a synthetic `engine` callable for deterministic CPU tests.
    """

    def __init__(
        self,
        *,
        level: str = "accurate",
        bands: int = 1,
        language_correction: bool = True,
        languages: tuple[str, ...] = ("en-US",),
        auto_orient: bool = False,
        min_long_edge: int = 960,
        engine: Callable[[bytes, int, int], list[tuple[str, float, list[float]]]] | None = None,
    ) -> None:
        if level not in ("accurate", "fast"):
            raise ValueError(f"level must be 'accurate' or 'fast', got {level!r}")
        self.level = level
        self.bands = max(1, int(bands))
        self.language_correction = bool(language_correction)
        self.languages = tuple(languages)
        self.auto_orient = bool(auto_orient)
        self.min_long_edge = max(0, int(min_long_edge))
        self._engine = engine

    @staticmethod
    def _prepare_vision_image(
        img: Image.Image,
        *,
        min_long_edge: int = 960,
        min_short_edge: int = 240,
        max_scale: float = 3.0,
    ) -> tuple[Image.Image, float]:
        """Bicubic-upscale tiny low-res crops for macOS Vision ANE stability; full pages pass through untouched."""
        w, h = img.size
        if min_long_edge <= 0 or w <= 0 or h <= 0:
            return img, 1.0
        eff_short = min(int(min_short_edge), int(min_long_edge))
        if max(w, h) >= min_long_edge and min(w, h) >= eff_short:
            return img, 1.0
        scale = min(
            float(max_scale),
            max(1.0, float(min_long_edge) / float(max(w, h)), float(eff_short) / float(min(w, h))),
        )
        if scale <= 1.05:
            return img, 1.0
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), resample=Image.Resampling.BICUBIC), round(scale, 3)

    def _get_engine(self) -> Callable[[bytes, int, int], list[tuple[str, float, list[float]]]]:
        if self._engine is None:
            self._engine = _NativeVisionRunner(
                level=self.level,
                languages=self.languages,
                language_correction=self.language_correction,
            )
        return self._engine

    def recognize_image(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        page_index: int = 0,
        output_format: OutputFormat = "markdown",
    ) -> OCRResult:
        """Recognize text on one image via ANE Vision, optionally synthesizing structured Markdown."""
        t_start = time.perf_counter()
        t_prep0 = t_start
        img = load_image(source)
        w, h = img.size
        vis_img, vis_scale = self._prepare_vision_image(img, min_long_edge=self.min_long_edge)
        vw, vh = vis_img.size
        rgb_bytes = vis_img.tobytes()
        prep_ms = (time.perf_counter() - t_prep0) * 1000.0

        t_vis0 = time.perf_counter()
        engine = self._get_engine()
        raw_dets, raw_vecs = _recognize_banded_with_vectors(engine, rgb_bytes, vw, vh, bands=self.bands)
        applied_rot = 0
        if self.auto_orient:
            rot_deg = _infer_rotation_cw_from_detections(raw_dets, vectors=raw_vecs)
            if rot_deg != 0:
                img = rotate_image_cw(img, rot_deg)
                w, h = img.size
                vis_img, vis_scale = self._prepare_vision_image(img, min_long_edge=self.min_long_edge)
                vw, vh = vis_img.size
                raw_dets, raw_vecs = _recognize_banded_with_vectors(engine, vis_img.tobytes(), vw, vh, bands=self.bands)
                applied_rot = rot_deg
        vis_ms = (time.perf_counter() - t_vis0) * 1000.0

        t_post0 = time.perf_counter()
        lines: list[OCRLine] = []
        for txt, conf, (bx, by, bw, bh) in raw_dets:
            # Convert Vision bottom-left [x, y, w, h] -> clamped top-left OCRBox(x0, y0, x1, y1)
            x0 = max(0.0, min(1.0, float(bx)))
            y0 = max(0.0, min(1.0, 1.0 - float(by) - float(bh)))
            x1 = max(0.0, min(1.0, float(bx) + float(bw)))
            y1 = max(0.0, min(1.0, 1.0 - float(by)))
            lines.append(OCRLine(text=str(txt), confidence=float(conf), box=OCRBox(x0, y0, x1, y1)))

        raw_text = "\n".join(ln.text for ln in lines)
        analysis = analyze_page_from_lines(lines, img, w, h, raw_detections=raw_dets, vectors=raw_vecs)

        validation = None
        if output_format == "markdown":
            from .markdown_struct import synthesize_markdown_from_lines, validate_markdown

            rendered = synthesize_markdown_from_lines(lines)
            validation = validate_markdown(rendered, allow_html_tables=False)
        else:
            rendered = raw_text

        post_ms = (time.perf_counter() - t_post0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return OCRResult(
            text=rendered,
            raw_text=raw_text,
            backend_used=f"apple-vision:{self.level}:b{self.bands}",
            page_index=int(page_index),
            output_format=output_format,
            table_mode="gfm",
            rotation_applied_cw=applied_rot,
            lines=tuple(lines),
            analysis=analysis,
            markdown_validation=validation,
            timings=OCRTimings(
                preprocess_ms=prep_ms,
                apple_vision_ms=vis_ms,
                postprocess_ms=post_ms,
                total_ms=total_ms,
            ),
            metadata={
                "bands": self.bands,
                "level": self.level,
                "language_correction": self.language_correction,
                "vision_upscale": vis_scale,
                **_get_macos_vision_provenance(),
                "request_revision": getattr(engine, "revision", None),
            },
        )

    def recognize_image_batch_parallel(
        self,
        image_paths: Sequence[str | Path],
        *,
        workers: int = 8,
        output_format: OutputFormat = "markdown",
    ) -> list[dict[str, Any]]:
        """Process many image paths concurrently across ANE worker processes."""
        paths = [str(Path(p).expanduser()) for p in image_paths]
        if not paths:
            return []
        n_workers = max(1, min(int(workers), len(paths)))
        chunks: list[list[str]] = [paths[i::n_workers] for i in range(n_workers)]
        tasks = [
            (self.level, self.bands, self.language_correction, self.languages, output_format, chunk)
            for chunk in chunks
            if chunk
        ]
        if len(tasks) == 1:
            return _worker_recognize_batch(tasks[0])
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            chunk_results = list(pool.map(_worker_recognize_batch, tasks))
        ordered: list[dict[str, Any]] = [{}] * len(paths)
        for worker_idx, res_list in enumerate(chunk_results):
            for local_i, item in enumerate(res_list):
                orig_idx = worker_idx + local_i * n_workers
                item["page_index"] = orig_idx
                ordered[orig_idx] = item
        return ordered
