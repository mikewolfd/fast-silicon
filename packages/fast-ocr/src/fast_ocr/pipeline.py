"""Unified zero-training OCR pipeline (`AutoOCRPipeline` + `FastOCR`) routing native PDF, ANE Apple Vision, and Qwen3.5 VLM."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import inspect
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .apple_vision import (
    AppleVisionOCR,
    _NativeVisionRunner,
    analyze_page_from_lines,
    deskew_document_image,
    normalize_deskew_mode,
)
from .markdown_struct import canonicalize_markdown, validate_markdown
from .preprocess import (
    DEFAULT_MAX_PDF_PIXELS,
    RenderedPDFPage,
    extract_native_pdf_text,
    load_image,
    render_pdf_pages,
    rotate_image_cw,
)
from .region_router import (
    REGION_CROP_MIN_PIXELS,
    PageSegment,
    extract_escalation_regions,
    merge_region_markdown,
)
from .types import (
    MarkdownTableMode,
    OCRBackend,
    OCRResult,
    OCRTimings,
    OutputFormat,
    PageAnalysis,
    SpeculativeMode,
)

_VALID_BACKENDS = frozenset({"apple-vision", "ovis", "auto"})
_VALID_OUTPUT_FORMATS = frozenset({"markdown", "raw", "text"})
_VALID_TABLE_MODES = frozenset({"gfm", "html"})
_VALID_SPECULATIVE_MODES = frozenset({"auto", "off", "self", "vision+self"})
_VALID_LEVELS = frozenset({"accurate", "fast"})

_NATIVE_MIN_CHARS = 80
_NATIVE_MAX_GARBLED_RATIO = 0.02

_NATIVE_BULLET_RE = re.compile(r"^(?:[-*•·]|(?:\d{1,3}|[a-zA-Z])[.)])\s+")
_NATIVE_ATX_RE = re.compile(r"^#{1,6}\s+")


def _filter_accepted_kwargs(fn: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Return only keyword arguments accepted by `fn`, preserving all if `**kwargs` is present."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {k: v for k, v in candidates.items() if v is not None}

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _estimate_source_pixels(src: Any) -> int:
    """Cheaply estimate pixel count (`W * H`) for parallel ANE triage routing."""
    if isinstance(src, Image.Image):
        return int(src.width) * int(src.height)
    if isinstance(src, np.ndarray):
        return (int(src.shape[0]) * int(src.shape[1])) if src.ndim >= 2 else 0
    if isinstance(src, (str, Path)):
        try:
            with Image.open(Path(src).expanduser()) as im:
                return int(im.width) * int(im.height)
        except Exception:
            return 0
    return 0


def _recognize_vision_worker_chunk(
    args: tuple[str, int, bool, tuple[str, ...], bool, int, OutputFormat, list[tuple[int, str]]],
) -> list[OCRResult]:
    """Top-level picklable worker running full `AppleVisionOCR.recognize_image` (no MLX import)."""
    level, bands, language_correction, languages, auto_orient, min_long_edge, output_format, chunk = args
    ocr = AppleVisionOCR(
        level=level,
        bands=bands,
        language_correction=language_correction,
        languages=languages,
        auto_orient=auto_orient,
        min_long_edge=min_long_edge,
    )
    return [ocr.recognize_image(path, page_index=page_idx, output_format=output_format) for page_idx, path in chunk]


def run_vision_batch_parallel(
    vision: Any,
    sources: Sequence[str | Path | Image.Image | np.ndarray],
    page_indices: Sequence[int],
    *,
    output_format: OutputFormat,
    workers: int = 4,
    min_batch: int = 4,
    min_total_pixels: int = 2_000_000,
) -> list[OCRResult] | None:
    """Dispatch `>= min_batch` full-page Apple Vision requests across `ProcessPoolExecutor`, else `None`."""
    if len(sources) < min_batch:
        return None
    if not isinstance(vision, AppleVisionOCR):
        return None
    injected_engine = getattr(vision, "_engine", None)
    if injected_engine is not None and not isinstance(injected_engine, _NativeVisionRunner):
        return None
    if sum(_estimate_source_pixels(src) for src in sources) < min_total_pixels:
        return None

    with tempfile.TemporaryDirectory(prefix="fast_ocr_ane_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        paths: list[str] = []
        for pos, src in enumerate(sources):
            if isinstance(src, (str, Path)):
                paths.append(str(Path(src).expanduser()))
            else:
                if isinstance(src, np.ndarray):
                    arr = np.asarray(src)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    pil_im = Image.fromarray(arr)
                elif isinstance(src, Image.Image):
                    pil_im = src
                else:
                    return None
                if pil_im.mode != "RGB":
                    pil_im = pil_im.convert("RGB")
                tmp_path = tmp_root / f"page_{pos}.png"
                pil_im.save(tmp_path, format="PNG", compress_level=1)
                paths.append(str(tmp_path))

        n_workers = max(1, min(int(workers), len(paths)))
        if n_workers <= 1:
            return None
        indexed = list(zip((int(i) for i in page_indices), paths))
        chunks = [indexed[w::n_workers] for w in range(n_workers)]
        tasks = [
            (
                str(vision.level),
                int(vision.bands),
                bool(vision.language_correction),
                tuple(vision.languages),
                bool(vision.auto_orient),
                int(vision.min_long_edge),
                output_format,
                chunk,
            )
            for chunk in chunks
            if chunk
        ]
        try:
            with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
                chunk_results = list(pool.map(_recognize_vision_worker_chunk, tasks))
        except (OSError, PermissionError):
            return None

        ordered: list[OCRResult | None] = [None] * len(paths)
        for worker_idx, res_list in enumerate(chunk_results):
            for local_i, res in enumerate(res_list):
                orig_pos = worker_idx + local_i * n_workers
                ordered[orig_pos] = res
        return [r for r in ordered if r is not None]


def _is_likely_native_heading(line: str, *, prev_blank: bool, next_blank: bool) -> bool:
    if not line or len(line) > 80 or line[-1] in ".,;":
        return False
    if _NATIVE_BULLET_RE.match(line) or _NATIVE_ATX_RE.match(line):
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    if alpha_count < 3:
        return False
    is_all_caps = line.isupper() and alpha_count >= 4
    if is_all_caps and (prev_blank or next_blank):
        return True
    if prev_blank and next_blank and len(line) <= 65:
        words = line.split()
        if 1 <= len(words) <= 9:
            return True
    return False


def _format_native_pdf_markdown(raw_text: str, *, table_mode: MarkdownTableMode = "gfm") -> str:
    """Apply lightweight heading, list, de-hyphenation, and paragraph preservation to native PDF text."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    raw_lines = [ln.rstrip() for ln in text.split("\n")]
    blocks: list[list[str]] = []
    para_buf: list[str] = []
    heading_count = 0

    def _flush_para() -> None:
        if not para_buf:
            return
        joined = ""
        for frag in para_buf:
            if joined.endswith("-") and frag and frag[0].islower():
                joined = joined[:-1] + frag
            elif joined:
                joined = joined + " " + frag
            else:
                joined = frag
        blocks.append([joined])
        para_buf.clear()

    n = len(raw_lines)
    for idx, raw_ln in enumerate(raw_lines):
        stripped = " ".join(raw_ln.split())
        if not stripped:
            _flush_para()
            continue

        prev_blank = idx == 0 or not raw_lines[idx - 1].strip()
        next_blank = idx + 1 == n or not raw_lines[idx + 1].strip()

        if _NATIVE_ATX_RE.match(stripped):
            _flush_para()
            blocks.append([stripped])
            heading_count += 1
        elif _NATIVE_BULLET_RE.match(stripped):
            _flush_para()
            norm_item = re.sub(r"^[•·*]\s+", "- ", stripped)
            blocks.append([norm_item])
        elif _is_likely_native_heading(stripped, prev_blank=prev_blank, next_blank=next_blank):
            _flush_para()
            prefix = "#" if heading_count == 0 and not blocks else "##"
            blocks.append([f"{prefix} {stripped}"])
            heading_count += 1
        else:
            para_buf.append(stripped)

    _flush_para()

    out_lines: list[str] = []
    for idx, blk in enumerate(blocks):
        if idx > 0:
            if (
                len(blk) == 1
                and len(blocks[idx - 1]) == 1
                and _NATIVE_BULLET_RE.match(blk[0])
                and _NATIVE_BULLET_RE.match(blocks[idx - 1][0])
            ):
                pass
            else:
                out_lines.append("")
        out_lines.extend(blk)
    return canonicalize_markdown("\n".join(out_lines), table_mode=table_mode)


def _merge_vision_prepass_timing(result: OCRResult, vision_result: OCRResult | None) -> OCRResult:
    """Fold ANE Vision pre-analysis latency into an escalated VLM `OCRResult` if not already counted."""
    if vision_result is None:
        return result
    vis_ms = float(vision_result.timings.apple_vision_ms)
    if vis_ms <= 0.0 or result.timings.apple_vision_ms > 0.0:
        return result
    updated_timings = replace(
        result.timings,
        apple_vision_ms=vis_ms,
        total_ms=float(result.timings.total_ms) + vis_ms,
    )
    return replace(result, timings=updated_timings)


class AutoOCRPipeline:
    """Hybrid OCR router combining native PDF text bypass, ANE Apple Vision, and lazy Qwen3.5 VLM escalation."""

    def __init__(
        self,
        backend: OCRBackend = "auto",
        output_format: OutputFormat = "markdown",
        table_mode: MarkdownTableMode = "gfm",
        strict_markdown_grammar: bool = False,
        zoom_recover: bool = False,
        model_path: str | Path | None = None,
        bands: int | None = None,
        level: str = "accurate",
        language_correction: bool | None = None,
        crop_margins: bool = True,
        auto_orient: bool = True,
        adaptive_resolution: bool = True,
        deskew: bool | str = "auto",
        region_routing: bool = False,
        speculative: SpeculativeMode = "auto",
        vocab_size: int | None = None,
        vit_quant_bits: int | None = None,
        vision_ocr: Any | None = None,
        ovis_engine: Any | None = None,
        draft_k: int = 4,
        languages: tuple[str, ...] = ("en-US",),
        max_tokens: int = 8192,
        loop_breaker: bool = False,
        auto_loop_recover: bool = True,
        memory_limit_gb: float | None = None,
        cache_limit_gb: float | None = None,
        kv_i8: bool | str | None = None,
        gdn_state_mode: str | None = None,
        drop_white_patches: bool | None = None,
        latin_only: bool | None = None,
    ) -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unsupported backend {backend!r}; expected one of {sorted(_VALID_BACKENDS)}")
        if output_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output_format {output_format!r}; expected one of {sorted(_VALID_OUTPUT_FORMATS)}")
        if table_mode not in _VALID_TABLE_MODES:
            raise ValueError(f"Unsupported table_mode {table_mode!r}; expected one of {sorted(_VALID_TABLE_MODES)}")
        if speculative not in _VALID_SPECULATIVE_MODES:
            raise ValueError(f"Unsupported speculative mode {speculative!r}; expected one of {sorted(_VALID_SPECULATIVE_MODES)}")
        if level not in _VALID_LEVELS:
            raise ValueError(f"Unsupported Apple Vision level {level!r}; expected one of {sorted(_VALID_LEVELS)}")

        # P3 receipt: `bands=3` + `language_correction=False` catches tight/multi-column tables & gives 98.6% rotated ledger tokens for auto escalation;
        # standalone `apple-vision` keeps `bands=1` + `language_correction=True` for fast clean reading.
        eff_bands = (3 if backend == "auto" else 1) if bands is None else max(1, int(bands))
        eff_lc = (False if backend == "auto" else True) if language_correction is None else bool(language_correction)

        self.backend: OCRBackend = backend
        self.output_format: OutputFormat = output_format
        self.table_mode: MarkdownTableMode = table_mode
        self.strict_markdown_grammar: bool = bool(strict_markdown_grammar)
        self.zoom_recover: bool = bool(zoom_recover)
        self.model_path: str | Path | None = model_path
        self.bands: int = eff_bands
        self.level: str = level
        self.language_correction: bool = eff_lc
        self.crop_margins: bool = bool(crop_margins)
        self.auto_orient: bool = bool(auto_orient)
        self.adaptive_resolution: bool = bool(adaptive_resolution)
        self.deskew: str = normalize_deskew_mode(deskew)
        self.region_routing: bool = bool(region_routing)
        self.speculative: SpeculativeMode = speculative
        self.vocab_size: int | None = vocab_size
        self.vit_quant_bits: int | None = vit_quant_bits
        self.draft_k: int = max(1, int(draft_k))
        self.languages: tuple[str, ...] = tuple(languages)
        self.max_tokens: int = max(1, int(max_tokens))
        self.loop_breaker: bool = bool(loop_breaker)
        self.auto_loop_recover: bool = bool(auto_loop_recover)
        self.memory_limit_gb: float | None = memory_limit_gb
        self.cache_limit_gb: float | None = cache_limit_gb
        self.kv_i8: bool | str | None = kv_i8
        self.gdn_state_mode: str | None = gdn_state_mode
        self.drop_white_patches: bool | None = drop_white_patches
        self.latin_only: bool | None = latin_only

        self._vision: Any | None = vision_ocr
        self._ovis: Any | None = ovis_engine

    def _get_vision_ocr(self) -> Any:
        if self._vision is None:
            self._vision = AppleVisionOCR(
                level=self.level,
                bands=self.bands,
                language_correction=self.language_correction,
                languages=self.languages,
                auto_orient=self.auto_orient,
            )
        return self._vision

    def _get_ovis_engine(self) -> Any:
        if self._ovis is None:
            from .ovis_engine import FastOvisOCREngine

            sig = inspect.signature(FastOvisOCREngine.__init__)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            candidates: dict[str, Any] = {
                "model_path": self.model_path,
                "output_format": self.output_format,
                "table_mode": self.table_mode,
                "strict_markdown_grammar": self.strict_markdown_grammar,
                "zoom_recover": self.zoom_recover,
                "crop_margins": self.crop_margins,
                "auto_orient": self.auto_orient,
                "adaptive_resolution": self.adaptive_resolution,
                "deskew": self.deskew,
                "speculative": self.speculative,
                "vocab_size": self.vocab_size,
                "vit_quant_bits": self.vit_quant_bits,
                "draft_k": self.draft_k,
                "max_tokens": self.max_tokens,
                "loop_breaker": self.loop_breaker,
                "auto_loop_recover": self.auto_loop_recover,
                "memory_limit_gb": self.memory_limit_gb,
                "cache_limit_gb": self.cache_limit_gb,
                "kv_i8": self.kv_i8,
                "gdn_state_mode": self.gdn_state_mode,
                "drop_white_patches": self.drop_white_patches,
                "latin_only": self.latin_only,
            }
            if not has_var_kw:
                if "model_path" not in params and "model_dir" in params:
                    candidates["model_dir"] = candidates.pop("model_path")
                if "draft_k" not in params and "max_draft_k" in params:
                    candidates["max_draft_k"] = candidates.pop("draft_k")
                if "speculative" not in params and "speculative_mode" in params:
                    candidates["speculative_mode"] = candidates.pop("speculative")
                filtered = {k: v for k, v in candidates.items() if k in params and (v is not None or params[k].default is inspect.Parameter.empty)}
            else:
                filtered = {k: v for k, v in candidates.items() if v is not None}
            self._ovis = FastOvisOCREngine(**filtered)
        return self._ovis

    def _maybe_deskew_source(
        self,
        source: str | Path | Image.Image | np.ndarray,
    ) -> tuple[str | Path | Image.Image | np.ndarray, dict[str, Any] | None]:
        if self.deskew == "off":
            return source, None
        try:
            warped, info = deskew_document_image(source, mode=self.deskew)
            if info.get("applied"):
                return warped, info
            return (source, info) if self.deskew == "on" else (source, None)
        except Exception:
            return (source, {"applied": False, "reason": "deskew_error"}) if self.deskew == "on" else (source, None)

    def _run_vision_image(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        page_index: int,
        output_format: OutputFormat,
        apply_deskew: bool = True,
    ) -> OCRResult:
        eff_src, deskew_info = self._maybe_deskew_source(source) if apply_deskew else (source, None)
        vision = self._get_vision_ocr()
        fn = getattr(vision, "recognize_image", None)
        if fn is None:
            fn = getattr(vision, "ocr_image", None)
        if fn is None:
            raise TypeError("Injected `vision_ocr` must provide `recognize_image` or `ocr_image`")
        kw = _filter_accepted_kwargs(
            fn,
            {
                "page_index": int(page_index),
                "output_format": output_format,
            },
        )
        res = fn(eff_src, **kw)
        if deskew_info is not None:
            meta = dict(res.metadata)
            meta["deskew"] = deskew_info
            res = replace(res, metadata=meta)
        return res

    def _run_ovis_image(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        page_index: int,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        pre_analysis: OCRResult | None = None,
        native_pdf_text: str | None = None,
        deskew_override: bool | None = None,
    ) -> OCRResult:
        ovis = self._get_ovis_engine()
        fn = getattr(ovis, "ocr_image", None)
        if fn is None:
            fn = getattr(ovis, "recognize_image", None)
        if fn is None:
            raise TypeError("`ovis_engine` must provide `ocr_image` or `recognize_image`")
        candidates: dict[str, Any] = {
            "page_index": int(page_index),
            "output_format": output_format,
            "table_mode": table_mode,
            "strict_markdown_grammar": self.strict_markdown_grammar,
            "zoom_recover": self.zoom_recover,
            "crop_margins": self.crop_margins,
            "auto_orient": self.auto_orient,
            "adaptive_resolution": self.adaptive_resolution,
            "deskew": self.deskew if deskew_override is None else bool(deskew_override),
            "speculative": self.speculative,
            "draft_k": self.draft_k,
            "max_tokens": self.max_tokens,
        }
        if pre_analysis is not None:
            candidates["pre_analysis"] = pre_analysis
        if native_pdf_text:
            candidates["native_pdf_text"] = native_pdf_text
        kw = _filter_accepted_kwargs(fn, candidates)
        res = fn(source, **kw)
        return _merge_vision_prepass_timing(res, pre_analysis)

    def _run_vision_batch(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        page_indices: Sequence[int],
        *,
        output_format: OutputFormat,
        apply_deskew: bool = True,
    ) -> list[OCRResult]:
        eff_sources: Sequence[Any] = sources
        deskew_metas: list[dict[str, Any] | None] | None = None
        if apply_deskew and self.deskew != "off":
            eff_list: list[Any] = []
            meta_list: list[dict[str, Any] | None] = []
            for src in sources:
                ds, dinfo = self._maybe_deskew_source(src)
                eff_list.append(ds)
                meta_list.append(dinfo)
            eff_sources = eff_list
            if any(m is not None for m in meta_list):
                deskew_metas = meta_list
        par = run_vision_batch_parallel(
            self._get_vision_ocr(),
            eff_sources,
            page_indices,
            output_format=output_format,
            workers=4,
        )
        if par is not None:
            if deskew_metas is None:
                return par
            out_par: list[OCRResult] = []
            for r, dm in zip(par, deskew_metas):
                if dm is not None:
                    meta = dict(r.metadata)
                    meta["deskew"] = dm
                    r = replace(r, metadata=meta)
                out_par.append(r)
            return out_par
        out_seq: list[OCRResult] = []
        for pos, (src, idx) in enumerate(zip(eff_sources, page_indices)):
            r = self._run_vision_image(src, page_index=int(idx), output_format=output_format, apply_deskew=False)
            if deskew_metas is not None and deskew_metas[pos] is not None:
                meta = dict(r.metadata)
                meta["deskew"] = deskew_metas[pos]
                r = replace(r, metadata=meta)
            out_seq.append(r)
        return out_seq

    def _run_ovis_batch(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        page_indices: Sequence[int],
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        pre_analyses: Sequence[OCRResult | None] | None = None,
        native_pdf_texts: Sequence[str | None] | None = None,
        batch_size: int | None = None,
        deskew_override: bool | None = None,
    ) -> list[OCRResult]:
        if not sources:
            return []
        ovis = self._get_ovis_engine()
        batch_fn = getattr(ovis, "ocr_images", None)
        has_pre = pre_analyses is not None and any(p is not None for p in pre_analyses)
        if callable(batch_fn) and (not has_pre or _supports_kwarg(batch_fn, "pre_analyses")):
            candidates: dict[str, Any] = {
                "page_indices": list(page_indices),
                "output_format": output_format,
                "table_mode": table_mode,
                "strict_markdown_grammar": self.strict_markdown_grammar,
                "zoom_recover": self.zoom_recover,
                "crop_margins": self.crop_margins,
                "auto_orient": self.auto_orient,
                "adaptive_resolution": self.adaptive_resolution,
                "deskew": self.deskew if deskew_override is None else bool(deskew_override),
                "speculative": self.speculative,
                "draft_k": self.draft_k,
                "max_tokens": self.max_tokens,
            }
            if batch_size is not None:
                candidates["batch_size"] = int(batch_size)
            if pre_analyses is not None:
                candidates["pre_analyses"] = list(pre_analyses)
            if native_pdf_texts is not None and any(native_pdf_texts):
                candidates["native_pdf_texts"] = list(native_pdf_texts)
            kw = _filter_accepted_kwargs(batch_fn, candidates)
            results = list(batch_fn(list(sources), **kw))
            if pre_analyses is not None:
                results = [_merge_vision_prepass_timing(r, p) for r, p in zip(results, pre_analyses)]
            return results

        out: list[OCRResult] = []
        pre_seq: Sequence[OCRResult | None] = pre_analyses if pre_analyses is not None else [None] * len(sources)
        nat_seq: Sequence[str | None] = native_pdf_texts if native_pdf_texts is not None else [None] * len(sources)
        for src, idx, pre, nat in zip(sources, page_indices, pre_seq, nat_seq):
            out.append(
                self._run_ovis_image(
                    src,
                    page_index=idx,
                    output_format=output_format,
                    table_mode=table_mode,
                    pre_analysis=pre,
                    native_pdf_text=nat,
                    deskew_override=deskew_override,
                )
            )
        return out

    def _try_region_route_pages(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        page_indices: Sequence[int],
        vision_results: Sequence[OCRResult],
        *,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        batch_size: int | None = None,
    ) -> dict[int, OCRResult]:
        """Opt-in prose+table region-only VLM escalation (`region_routing=True`, formula pages excluded)."""
        if not self.region_routing or output_format != "markdown" or not sources:
            return {}

        page_plans: list[tuple[int, int, OCRResult, int, list[PageSegment], int, int]] = []
        all_crops: list[Image.Image] = []
        all_prompts: list[str | None] = []

        for pos, (src, idx, v_res) in enumerate(zip(sources, page_indices, vision_results)):
            if not v_res.lines:
                continue
            img = load_image(src)
            rot_cw = 0
            if self.auto_orient:
                applied = int(v_res.rotation_applied_cw) % 360
                if applied != 0:
                    rot_cw = applied
                elif v_res.analysis is not None:
                    rot_cw = int(v_res.analysis.inferred_rotation_cw) % 360
            oriented = rotate_image_cw(img, rot_cw) if rot_cw != 0 else img
            line_rot = (rot_cw - int(v_res.rotation_applied_cw)) % 360
            segs, ok = extract_escalation_regions(
                v_res,
                oriented.size,
                rotation_cw=line_rot,
                allow_formula_regions=False,
            )
            if not ok:
                continue
            start_c = len(all_crops)
            for seg in segs:
                if seg.kind != "prose" and seg.crop_box_px is not None:
                    all_crops.append(oriented.crop(seg.crop_box_px))
                    all_prompts.append(seg.prompt)
            end_c = len(all_crops)
            if end_c > start_c:
                page_plans.append((pos, int(idx), v_res, rot_cw, segs, start_c, end_c))

        if not page_plans:
            return {}

        ovis = self._get_ovis_engine()
        crop_tok_cap = min(1024, self.max_tokens)
        crop_spec: SpeculativeMode = "off" if self.speculative == "off" else "self"
        batch_fn = getattr(ovis, "ocr_images", None)
        if len(all_crops) >= 2 and callable(batch_fn) and _supports_kwarg(batch_fn, "prompts"):
            batch_kw = _filter_accepted_kwargs(
                batch_fn,
                {
                    "prompts": all_prompts,
                    "output_format": "raw",
                    "table_mode": table_mode,
                    "strict_markdown_grammar": False,
                    "zoom_recover": False,
                    "crop_margins": False,
                    "auto_orient": False,
                    "adaptive_resolution": False,
                    "deskew": False,
                    "min_pixels": REGION_CROP_MIN_PIXELS,
                    "speculative": crop_spec,
                    "draft_k": self.draft_k,
                    "max_tokens": crop_tok_cap,
                    "batch_size": (
                        max(2, len(all_crops))
                        if batch_size is None
                        else max(1, min(int(batch_size), len(all_crops)))
                    ),
                },
            )
            crop_results = list(batch_fn(all_crops, **batch_kw))
        else:
            img_fn = getattr(ovis, "ocr_image", None) or getattr(ovis, "recognize_image", None)
            if img_fn is None:
                return {}
            crop_results = []
            for crop_im, crop_pr in zip(all_crops, all_prompts):
                kw = _filter_accepted_kwargs(
                    img_fn,
                    {
                        "prompt": crop_pr,
                        "output_format": "raw",
                        "table_mode": table_mode,
                        "strict_markdown_grammar": False,
                        "zoom_recover": False,
                        "crop_margins": False,
                        "auto_orient": False,
                        "adaptive_resolution": False,
                        "deskew": False,
                        "min_pixels": REGION_CROP_MIN_PIXELS,
                        "speculative": crop_spec,
                        "draft_k": self.draft_k,
                        "max_tokens": crop_tok_cap,
                    },
                )
                crop_results.append(img_fn(crop_im, **kw))

        routed: dict[int, OCRResult] = {}
        for pos, page_idx, v_res, rot_cw, segs, start_c, end_c in page_plans:
            sub = crop_results[start_c:end_c]
            merged = merge_region_markdown(segs, [r.text for r in sub], table_mode=table_mode)
            validation = validate_markdown(merged, allow_html_tables=(table_mode == "html"))
            vis_ms = float(v_res.timings.apple_vision_ms)
            prep_ms = sum(float(r.timings.preprocess_ms) for r in sub)
            vit_ms = sum(float(r.timings.vit_ms) for r in sub)
            prefill_ms = sum(float(r.timings.prefill_ms) for r in sub)
            dec_ms = sum(float(r.timings.decode_ms) for r in sub)
            post_ms = sum(float(r.timings.postprocess_ms) for r in sub)
            gen_toks = sum(int(r.timings.generated_tokens) for r in sub)
            tot_ms = vis_ms + sum(float(r.timings.total_ms) for r in sub)
            dec_tps = (gen_toks / (dec_ms / 1000.0)) if dec_ms > 0.0 else 0.0
            meta = dict(v_res.metadata)
            meta["region_routing"] = {
                "enabled": True,
                "crops": len(sub),
                "min_pixels": REGION_CROP_MIN_PIXELS,
                "segment_kinds": [s.kind for s in segs],
            }
            routed[pos] = OCRResult(
                text=merged,
                raw_text=merged,
                backend_used=f"{sub[0].backend_used}+ane-regions",
                page_index=page_idx,
                output_format=output_format,
                table_mode=table_mode,
                rotation_applied_cw=rot_cw,
                lines=v_res.lines,
                analysis=v_res.analysis,
                markdown_validation=validation,
                timings=OCRTimings(
                    preprocess_ms=round(prep_ms, 3),
                    apple_vision_ms=round(vis_ms, 3),
                    vit_ms=round(vit_ms, 3),
                    prefill_ms=round(prefill_ms, 3),
                    decode_ms=round(dec_ms, 3),
                    postprocess_ms=round(post_ms, 3),
                    total_ms=round(tot_ms, 3),
                    decode_tps=round(dec_tps, 2),
                    prompt_tokens=sum(int(r.timings.prompt_tokens) for r in sub),
                    visual_tokens=sum(int(r.timings.visual_tokens) for r in sub),
                    generated_tokens=gen_toks,
                    peak_memory_gb=max((float(r.timings.peak_memory_gb) for r in sub), default=0.0),
                ),
                finish_reason="stop",
                metadata=meta,
            )
        return routed

    def ocr_image(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        page_index: int = 0,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        pre_analysis: OCRResult | None = None,
    ) -> OCRResult:
        """Run OCR on a single image using the configured backend (`apple-vision`, `ovis`, or `auto`)."""
        fmt = output_format or self.output_format
        tmode = table_mode or self.table_mode

        if self.backend == "apple-vision":
            return self._run_vision_image(source, page_index=page_index, output_format=fmt)

        if self.backend == "ovis":
            return self._run_ovis_image(
                source,
                page_index=page_index,
                output_format=fmt,
                table_mode=tmode,
                pre_analysis=pre_analysis,
            )

        eff_src, deskew_info = self._maybe_deskew_source(source) if pre_analysis is None else (source, None)
        if pre_analysis is not None:
            vision_res = pre_analysis
        else:
            vision = self._get_vision_ocr()
            fn = getattr(vision, "recognize_image", None) or getattr(vision, "ocr_image", None)
            if fn is None:
                raise TypeError("Injected `vision_ocr` must provide `recognize_image` or `ocr_image`")
            kw = _filter_accepted_kwargs(fn, {"page_index": int(page_index), "output_format": fmt})
            vision_res = fn(eff_src, **kw)
            if deskew_info is not None:
                meta = dict(vision_res.metadata)
                meta["deskew"] = deskew_info
                vision_res = replace(vision_res, metadata=meta)

        if vision_res.analysis is not None and vision_res.analysis.needs_vlm:
            if self.region_routing and fmt == "markdown":
                routed = self._try_region_route_pages(
                    [eff_src],
                    [page_index],
                    [vision_res],
                    output_format=fmt,
                    table_mode=tmode,
                )
                if 0 in routed:
                    return routed[0]
            return self._run_ovis_image(
                eff_src,
                page_index=page_index,
                output_format=fmt,
                table_mode=tmode,
                pre_analysis=vision_res,
                deskew_override=False if (self.deskew != "off" and pre_analysis is None) else None,
            )
        return vision_res

    def ocr_images(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        page_indices: Sequence[int] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        batch_size: int = 16,
        native_pdf_texts: Mapping[int, str] | None = None,
    ) -> list[OCRResult]:
        """Run OCR across multiple images, parallelizing ANE pre-analysis and batching VLM escalation."""
        if not sources:
            return []
        indices = list(range(len(sources))) if page_indices is None else [int(i) for i in page_indices]
        if len(indices) != len(sources):
            raise ValueError(f"Expected {len(sources)} page_indices, got {len(indices)}")

        fmt = output_format or self.output_format
        tmode = table_mode or self.table_mode
        bs = max(1, int(batch_size))

        if self.backend == "apple-vision":
            return self._run_vision_batch(sources, indices, output_format=fmt)

        if self.backend == "ovis":
            nat_list = [native_pdf_texts.get(int(idx)) for idx in indices] if native_pdf_texts else None
            return self._run_ovis_batch(
                sources,
                page_indices=indices,
                output_format=fmt,
                table_mode=tmode,
                native_pdf_texts=nat_list,
                batch_size=bs,
            )

        # If deskew is active (`auto` or `on`) in `auto` mode, run once upfront so ANE Vision and VLM share identical coordinates.
        deskew_active = self.deskew != "off"
        eff_sources: Sequence[Any] = (
            [self._maybe_deskew_source(src)[0] for src in sources] if deskew_active else sources
        )

        # `auto` mode: pipeline ANE Vision pre-analysis of batch `k + 1` while VLM runs escalated pages of batch `k`
        batches: list[tuple[Sequence[Any], list[int]]] = [
            (eff_sources[start : start + bs], indices[start : start + bs])
            for start in range(0, len(eff_sources), bs)
        ]

        def _vision_batch(
            batch_srcs: Sequence[str | Path | Image.Image | np.ndarray],
            batch_idxs: list[int],
        ) -> list[OCRResult]:
            return self._run_vision_batch(batch_srcs, batch_idxs, output_format=fmt, apply_deskew=False)

        results: list[OCRResult] = []
        if len(batches) == 1:
            vis_batch = _vision_batch(batches[0][0], batches[0][1])
            return self._resolve_auto_batch(
                batches[0][0],
                batches[0][1],
                vis_batch,
                output_format=fmt,
                table_mode=tmode,
                batch_size=bs,
                native_pdf_texts=native_pdf_texts,
                deskew_already_applied=deskew_active,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            cur_vis = _vision_batch(batches[0][0], batches[0][1])
            for b_idx, (b_srcs, b_idxs) in enumerate(batches):
                next_future: Future[list[OCRResult]] | None = None
                if b_idx + 1 < len(batches):
                    nb_srcs, nb_idxs = batches[b_idx + 1]
                    next_future = pool.submit(_vision_batch, nb_srcs, nb_idxs)
                results.extend(
                    self._resolve_auto_batch(
                        b_srcs,
                        b_idxs,
                        cur_vis,
                        output_format=fmt,
                        table_mode=tmode,
                        batch_size=bs,
                        native_pdf_texts=native_pdf_texts,
                        deskew_already_applied=deskew_active,
                    )
                )
                if next_future is not None:
                    cur_vis = next_future.result()
        return results

    def _dispatch_escalated_vlm(
        self,
        esc_srcs: Sequence[str | Path | Image.Image | np.ndarray],
        esc_idxs: Sequence[int],
        esc_vis: Sequence[OCRResult],
        *,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        batch_size: int | None = None,
        native_pdf_texts: Mapping[int, str] | None = None,
        deskew_override: bool | None = None,
    ) -> list[OCRResult]:
        """Run opt-in region routing on mixed prose+table pages, then batch remaining pages through Ovis."""
        if not esc_srcs:
            return []
        eff_bs = max(1, int(batch_size)) if batch_size is not None else max(1, len(esc_srcs))
        out: list[OCRResult | None] = [None] * len(esc_srcs)
        rem_pos: list[int] = list(range(len(esc_srcs)))
        rem_srcs: list[str | Path | Image.Image | np.ndarray] = list(esc_srcs)
        rem_idxs: list[int] = [int(i) for i in esc_idxs]
        rem_vis: list[OCRResult] = list(esc_vis)

        if self.region_routing and output_format == "markdown":
            routed_map = self._try_region_route_pages(
                rem_srcs,
                rem_idxs,
                rem_vis,
                output_format=output_format,
                table_mode=table_mode,
                batch_size=eff_bs,
            )
            if routed_map:
                next_pos: list[int] = []
                next_srcs: list[str | Path | Image.Image | np.ndarray] = []
                next_idxs: list[int] = []
                next_vis: list[OCRResult] = []
                for local_i, (orig_pos, src, idx, v_res) in enumerate(
                    zip(rem_pos, rem_srcs, rem_idxs, rem_vis)
                ):
                    if local_i in routed_map:
                        out[orig_pos] = routed_map[local_i]
                    else:
                        next_pos.append(orig_pos)
                        next_srcs.append(src)
                        next_idxs.append(idx)
                        next_vis.append(v_res)
                rem_pos, rem_srcs, rem_idxs, rem_vis = next_pos, next_srcs, next_idxs, next_vis

        if rem_srcs:
            esc_nat = [native_pdf_texts.get(int(idx)) for idx in rem_idxs] if native_pdf_texts else None
            vlm_results = self._run_ovis_batch(
                rem_srcs,
                page_indices=rem_idxs,
                output_format=output_format,
                table_mode=table_mode,
                pre_analyses=rem_vis,
                native_pdf_texts=esc_nat,
                batch_size=min(eff_bs, len(rem_srcs)),
                deskew_override=deskew_override,
            )
            for orig_pos, vlm_res in zip(rem_pos, vlm_results):
                out[orig_pos] = vlm_res

        return [r for r in out if r is not None]

    def _resolve_auto_batch(
        self,
        batch_srcs: Sequence[str | Path | Image.Image | np.ndarray],
        batch_idxs: Sequence[int],
        vision_results: Sequence[OCRResult],
        *,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        batch_size: int | None = None,
        native_pdf_texts: Mapping[int, str] | None = None,
        deskew_already_applied: bool = False,
    ) -> list[OCRResult]:
        resolved: list[OCRResult | None] = [None] * len(batch_srcs)
        esc_pos: list[int] = []
        esc_srcs: list[str | Path | Image.Image | np.ndarray] = []
        esc_idxs: list[int] = []
        esc_vis: list[OCRResult] = []

        for pos, (src, idx, v_res) in enumerate(zip(batch_srcs, batch_idxs, vision_results)):
            if v_res.analysis is not None and v_res.analysis.needs_vlm:
                esc_pos.append(pos)
                esc_srcs.append(src)
                esc_idxs.append(idx)
                esc_vis.append(v_res)
            else:
                resolved[pos] = v_res

        if esc_srcs:
            esc_outs = self._dispatch_escalated_vlm(
                esc_srcs,
                esc_idxs,
                esc_vis,
                output_format=output_format,
                table_mode=table_mode,
                batch_size=batch_size,
                native_pdf_texts=native_pdf_texts,
                deskew_override=False if deskew_already_applied else None,
            )
            for pos, vlm_res in zip(esc_pos, esc_outs):
                resolved[pos] = vlm_res

        return [r for r in resolved if r is not None]

    def _build_native_result(
        self,
        page: RenderedPDFPage,
        *,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        inspect_ms: float,
        native_analysis: PageAnalysis | None = None,
    ) -> OCRResult:
        t0 = time.perf_counter()
        raw_text = page.native_text
        validation = None
        if output_format == "markdown":
            rendered = _format_native_pdf_markdown(raw_text, table_mode=table_mode)
            validation = validate_markdown(rendered, allow_html_tables=(table_mode == "html"))
        else:
            rendered = raw_text
        post_ms = (time.perf_counter() - t0) * 1000.0
        meta: dict[str, Any] = {
            "native_char_count": page.native_char_count,
            "native_body_char_count": (
                page.native_body_char_count
                if page.native_body_char_count is not None
                else page.native_char_count
            ),
            "native_control_char_count": page.native_control_char_count,
            "image_count": page.image_count,
            "native_garbled_ratio": page.native_garbled_ratio,
            "width_pt": page.width_pt,
            "height_pt": page.height_pt,
        }
        if native_analysis is not None:
            meta["native_estimated_columns"] = int(native_analysis.estimated_columns)
            meta["native_table_score"] = float(native_analysis.table_score)
        return OCRResult(
            text=rendered,
            raw_text=raw_text,
            backend_used="native",
            page_index=page.page_index,
            output_format=output_format,
            table_mode=table_mode,
            markdown_validation=validation,
            timings=OCRTimings(
                preprocess_ms=inspect_ms,
                postprocess_ms=post_ms,
                total_ms=inspect_ms + post_ms,
            ),
            metadata=meta,
        )

    def _render_pdf_subset(
        self,
        pdf_path: str | Path,
        *,
        dpi: int,
        page_indices: Sequence[int] | None,
        max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
    ) -> tuple[list[Image.Image], list[int], float]:
        """Rasterize selected PDF pages at `dpi` (capped by `max_pixels`) and return `(images, page_indices, per_page_ms)`."""
        t_ren0 = time.perf_counter()
        render_kw: dict[str, Any] = {
            "dpi": int(dpi),
            "page_indices": page_indices,
            "include_native_text": False,
        }
        if max_pixels != DEFAULT_MAX_PDF_PIXELS:
            render_kw["max_pixels"] = max_pixels
        rendered_pages = render_pdf_pages(pdf_path, **render_kw)
        ren_ms = (time.perf_counter() - t_ren0) * 1000.0
        if not rendered_pages:
            return [], [], 0.0
        per_page_ren = ren_ms / len(rendered_pages)
        imgs: list[Image.Image] = []
        img_idxs: list[int] = []
        for rp in rendered_pages:
            if rp.image is None:
                raise RuntimeError(f"Failed to rasterize page {rp.page_index} of {pdf_path}")
            imgs.append(rp.image)
            img_idxs.append(rp.page_index)
        return imgs, img_idxs, per_page_ren

    @staticmethod
    def _fold_pdf_overhead(
        result: OCRResult,
        *,
        preprocess_extra_ms: float = 0.0,
        triage_dpi: int | None = None,
    ) -> OCRResult:
        extra = float(preprocess_extra_ms)
        updated_timings = (
            replace(
                result.timings,
                preprocess_ms=float(result.timings.preprocess_ms) + extra,
                total_ms=float(result.timings.total_ms) + extra,
            )
            if extra != 0.0
            else result.timings
        )
        if triage_dpi is None:
            return replace(result, timings=updated_timings)
        updated_meta = dict(result.metadata)
        updated_meta["triage_dpi"] = int(triage_dpi)
        return replace(result, timings=updated_timings, metadata=updated_meta)

    def _run_raster_pdf_pages(
        self,
        pdf_path: str | Path,
        *,
        dpi: int,
        triage_dpi: int | None,
        page_indices: Sequence[int] | None,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        batch_size: int,
        max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
        native_pdf_texts: Mapping[int, str] | None = None,
    ) -> list[OCRResult]:
        eff_triage = (
            int(triage_dpi)
            if (self.backend == "auto" and triage_dpi is not None and 0 < int(triage_dpi) < int(dpi))
            else None
        )
        if eff_triage is None:
            imgs, img_idxs, per_page_ren = self._render_pdf_subset(
                pdf_path,
                dpi=int(dpi),
                page_indices=page_indices,
                max_pixels=max_pixels,
            )
            if not imgs:
                return []
            ocr_outs = self.ocr_images(
                imgs,
                page_indices=img_idxs,
                output_format=output_format,
                table_mode=table_mode,
                batch_size=batch_size,
                native_pdf_texts=native_pdf_texts,
            )
            return [self._fold_pdf_overhead(r, preprocess_extra_ms=per_page_ren) for r in ocr_outs]

        # Two-stage ANE triage: render non-native pages at `triage_dpi` first, and re-render only escalated pages at `dpi`.
        low_imgs, low_idxs, per_page_low = self._render_pdf_subset(
            pdf_path,
            dpi=eff_triage,
            page_indices=page_indices,
            max_pixels=max_pixels,
        )
        if not low_imgs:
            return []

        low_vis = self._run_vision_batch(low_imgs, low_idxs, output_format=output_format)
        by_page: dict[int, OCRResult] = {}
        esc_idxs: list[int] = []
        esc_vis: list[OCRResult] = []
        for idx, v_res in zip(low_idxs, low_vis):
            if v_res.analysis is not None and v_res.analysis.needs_vlm:
                esc_idxs.append(idx)
                esc_vis.append(v_res)
            else:
                by_page[idx] = self._fold_pdf_overhead(
                    v_res,
                    preprocess_extra_ms=per_page_low,
                    triage_dpi=eff_triage,
                )

        if esc_idxs:
            hi_imgs, hi_idxs, per_page_hi = self._render_pdf_subset(
                pdf_path,
                dpi=int(dpi),
                page_indices=esc_idxs,
                max_pixels=max_pixels,
            )
            bs = max(1, int(batch_size))
            chunk_outs = self._dispatch_escalated_vlm(
                hi_imgs,
                hi_idxs,
                esc_vis,
                output_format=output_format,
                table_mode=table_mode,
                batch_size=bs,
                native_pdf_texts=native_pdf_texts,
            )
            for res in chunk_outs:
                by_page[res.page_index] = self._fold_pdf_overhead(
                    res,
                    preprocess_extra_ms=per_page_low + per_page_hi,
                    triage_dpi=eff_triage,
                )

        return [by_page[idx] for idx in low_idxs]

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        *,
        dpi: int = 300,
        triage_dpi: int | None = None,
        page_indices: Sequence[int] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        batch_size: int = 16,
        use_native_text: bool | None = None,
        max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
    ) -> list[OCRResult]:
        """OCR a PDF: emit clean native digital pages without rasterizing, then route remaining pages."""
        fmt = output_format or self.output_format
        tmode = table_mode or self.table_mode
        check_native = (self.backend == "auto") if use_native_text is None else bool(use_native_text)

        if check_native:
            t0 = time.perf_counter()
            native_pages = extract_native_pdf_text(pdf_path, page_indices=page_indices)
            inspect_ms = (time.perf_counter() - t0) * 1000.0
            per_page_inspect = inspect_ms / max(1, len(native_pages))

            ordered_results: dict[int, OCRResult] = {}
            ordered_indices = [p.page_index for p in native_pages]
            raster_indices: list[int] = []
            raster_native_hints: dict[int, str] = {}
            for p in native_pages:
                eff_chars = p.native_body_char_count if p.native_body_char_count is not None else p.native_char_count
                is_clean = p.native_control_char_count == 0 and p.native_garbled_ratio < _NATIVE_MAX_GARBLED_RATIO
                native_pa = (
                    analyze_page_from_lines(
                        p.lines,
                        None,
                        int(round(p.width_pt)),
                        int(round(p.height_pt)),
                    )
                    if (is_clean and p.lines)
                    else None
                )
                layout_needs_vlm = bool(native_pa is not None and native_pa.needs_vlm)
                if (
                    is_clean
                    and (not layout_needs_vlm)
                    and (eff_chars >= _NATIVE_MIN_CHARS or (eff_chars > 0 and p.image_count == 0))
                ):
                    ordered_results[p.page_index] = self._build_native_result(
                        p,
                        output_format=fmt,
                        table_mode=tmode,
                        inspect_ms=per_page_inspect,
                        native_analysis=native_pa,
                    )
                else:
                    raster_indices.append(p.page_index)
                    if is_clean and p.native_text and len(p.native_text.strip()) >= 20:
                        raster_native_hints[p.page_index] = p.native_text.strip()

            if raster_indices:
                for res in self._run_raster_pdf_pages(
                    pdf_path,
                    dpi=int(dpi),
                    triage_dpi=triage_dpi,
                    page_indices=raster_indices,
                    output_format=fmt,
                    table_mode=tmode,
                    batch_size=batch_size,
                    max_pixels=max_pixels,
                    native_pdf_texts=raster_native_hints or None,
                ):
                    ordered_results[res.page_index] = res

            return [ordered_results[idx] for idx in ordered_indices]

        return self._run_raster_pdf_pages(
            pdf_path,
            dpi=int(dpi),
            triage_dpi=triage_dpi,
            page_indices=page_indices,
            output_format=fmt,
            table_mode=tmode,
            batch_size=batch_size,
            max_pixels=max_pixels,
        )

    def ocr_parquet(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        input_col: str | None = None,
        shard_rows: int = 64,
        limit: int | None = None,
        resume: bool = True,
        compression: str = "zstd",
        dpi: int = 180,
        triage_dpi: int = 72,
        max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
        batch_size: int = 16,
        prompt: str | None = None,
        schema: str | dict[str, Any] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        skip_errors: bool = False,
    ) -> dict[str, Any]:
        """Stream PDF/image rows from Parquet file(s) with atomic sharded checkpoints and resume."""
        import io
        import json
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq
        from fast_shared.parquet_pipeline import (
            _load_manifest,
            _resolve_parquet_compression,
            _save_manifest,
            resolve_input_files,
        )

        files = resolve_input_files(input_path)
        out_raw = str(output_path).strip()
        single_file_target = out_raw.endswith(".parquet")
        out_target = Path(out_raw).expanduser().resolve()
        shard_dir = (
            out_target.parent / f".{out_target.stem}_shards"
            if single_file_target
            else out_target
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = shard_dir / "manifest.json"
        if single_file_target:
            out_target.parent.mkdir(parents=True, exist_ok=True)

        step = max(1, int(shard_rows))
        row_budget = None if limit is None else max(0, int(limit))
        fmt: OutputFormat = output_format or self.output_format
        tmode: MarkdownTableMode = table_mode or self.table_mode

        manifest: dict[str, Any] = _load_manifest(manifest_path) if resume else {}
        completed_raw = manifest.get("completed_shards")
        completed_shards: dict[str, dict[str, Any]] = (
            dict(completed_raw) if isinstance(completed_raw, dict) else {}
        )
        manifest = {
            "version": 1,
            "input_path": str(input_path),
            "backend": self.backend,
            "output_format": fmt,
            "table_mode": tmode,
            "shard_rows": step,
            "completed_shards": completed_shards,
        }

        total_rows = 0
        processed_rows = 0
        skipped_rows = 0
        valid_rows = 0
        completed_count = 0
        skipped_count = 0
        total_ms = 0.0
        shard_files: list[Path] = []
        global_shard_idx = 0

        def _is_valid_json_text(txt: str) -> bool:
            try:
                json.loads(txt)
                return True
            except Exception:
                return False

        def _resolve_rel_path(raw_val: str, base_file: Path) -> Path:
            p = Path(raw_val).expanduser()
            if not p.is_file():
                rel = base_file.parent / p
                if rel.is_file():
                    return rel
            return p

        def _ocr_single_row(
            cols: dict[str, list[Any]],
            idx: int,
            src_file: Path,
        ) -> tuple[str, str, bool, float, dict[str, Any]]:
            cand_cols = (
                [input_col]
                if input_col is not None
                else [c for c in ("pdf_path", "image_path", "image_bytes", "image") if c in cols]
            )
            chosen_col: str | None = None
            raw_val: Any = None
            for col_name in cand_cols:
                if col_name not in cols:
                    raise KeyError(f"Column {col_name!r} not found in {src_file.name}; available={list(cols)}")
                val = cols[col_name][idx]
                if val is not None:
                    chosen_col = col_name
                    raw_val = val
                    break
            if chosen_col is None:
                raise ValueError(f"No non-null OCR source found at row {idx} of {src_file.name}")

            if chosen_col == "pdf_path" or (
                isinstance(raw_val, (str, Path)) and str(raw_val).lower().endswith(".pdf")
            ):
                pdf_p = _resolve_rel_path(str(raw_val), src_file)
                page_idxs: list[int] | None = None
                if "page_index" in cols and cols["page_index"][idx] is not None:
                    page_idxs = [int(cols["page_index"][idx])]
                if schema is not None or prompt is not None:
                    if self.backend == "apple-vision":
                        raise ValueError("Parquet `schema`/`prompt` extraction requires backend 'ovis' or 'auto'.")
                    rendered_pages = render_pdf_pages(
                        pdf_p,
                        dpi=int(dpi),
                        page_indices=page_idxs,
                        include_native_text=False,
                        max_pixels=max_pixels,
                    )
                    ovis = self._get_ovis_engine()
                    pages = ovis.ocr_images(
                        [rp.image for rp in rendered_pages if rp.image is not None],
                        page_indices=[rp.page_index for rp in rendered_pages if rp.image is not None],
                        prompt=prompt,
                        schema=schema,
                        output_format=fmt,
                        table_mode=tmode,
                    )
                else:
                    pages = self.ocr_pdf(
                        pdf_p,
                        dpi=int(dpi),
                        triage_dpi=int(triage_dpi),
                        page_indices=page_idxs,
                        output_format=fmt,
                        table_mode=tmode,
                        batch_size=int(batch_size),
                        max_pixels=max_pixels,
                    )
                ocr_txt = "\n\n".join(r.text for r in pages)
                backends = ",".join(dict.fromkeys(r.backend_used for r in pages)) if pages else self.backend
                if schema is not None:
                    is_valid = bool(pages) and all(_is_valid_json_text(r.text) for r in pages)
                else:
                    is_valid = bool(pages) and all(
                        (r.markdown_validation.valid if r.markdown_validation is not None else bool(r.text.strip()))
                        for r in pages
                    )
                row_ms = sum(float(r.timings.total_ms) for r in pages)
                meta = {
                    "page_count": len(pages),
                    "page_indices": [int(r.page_index) for r in pages],
                    "output_format": pages[0].output_format if pages else fmt,
                    "table_mode": pages[0].table_mode if pages else tmode,
                    "finish_reasons": [r.finish_reason for r in pages],
                    "generated_tokens": sum(int(r.timings.generated_tokens) for r in pages),
                    "metadata": pages[0].metadata if len(pages) == 1 else [r.metadata for r in pages],
                }
                return ocr_txt, backends, is_valid, row_ms, meta

            img_src: Image.Image | Path
            if isinstance(raw_val, dict):
                if raw_val.get("bytes") is not None:
                    img_src = Image.open(io.BytesIO(bytes(raw_val["bytes"]))).convert("RGB")
                elif raw_val.get("path"):
                    img_src = _resolve_rel_path(str(raw_val["path"]), src_file)
                else:
                    raise ValueError(f"Unsupported image dict at row {idx}: keys={list(raw_val)}")
            elif isinstance(raw_val, (bytes, bytearray, memoryview)):
                img_src = Image.open(io.BytesIO(bytes(raw_val))).convert("RGB")
            else:
                img_src = _resolve_rel_path(str(raw_val), src_file)

            if schema is not None or prompt is not None:
                if self.backend == "apple-vision":
                    raise ValueError("Parquet `schema`/`prompt` extraction requires backend 'ovis' or 'auto'.")
                res = self._get_ovis_engine().ocr_image(
                    img_src,
                    prompt=prompt,
                    schema=schema,
                    output_format=fmt,
                    table_mode=tmode,
                )
            else:
                res = self.ocr_image(
                    img_src,
                    output_format=fmt,
                    table_mode=tmode,
                )
            if schema is not None:
                is_valid = _is_valid_json_text(res.text)
            elif res.markdown_validation is not None:
                is_valid = bool(res.markdown_validation.valid)
            else:
                is_valid = bool(res.text.strip())
            meta = {
                "page_count": 1,
                "page_index": int(res.page_index),
                "output_format": res.output_format,
                "table_mode": res.table_mode,
                "rotation_applied_cw": int(res.rotation_applied_cw),
                "finish_reason": res.finish_reason,
                "generated_tokens": int(res.timings.generated_tokens),
                "decode_tps": float(res.timings.decode_tps),
                "metadata": res.metadata,
            }
            return res.text, res.backend_used, is_valid, float(res.timings.total_ms), meta

        for file_idx, src_file in enumerate(files):
            pf = pq.ParquetFile(src_file)
            file_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
            if file_rows <= 0:
                continue
            remaining_for_file = file_rows if row_budget is None else min(file_rows, max(0, row_budget - total_rows))
            if remaining_for_file <= 0:
                break

            file_table = pf.read()
            for row_start in range(0, remaining_for_file, step):
                row_end = min(remaining_for_file, row_start + step)
                shard_len = row_end - row_start
                shard_key = f"{file_idx:04d}:{src_file.name}:{row_start}:{row_end}"
                shard_name = f"part-{global_shard_idx:05d}.parquet"
                shard_path = shard_dir / shard_name
                global_shard_idx += 1

                prev_entry = completed_shards.get(shard_key) if resume else None
                if prev_entry is not None:
                    prev_file = shard_dir / str(prev_entry.get("file", shard_name))
                    if prev_file.is_file() and prev_file.stat().st_size > 0:
                        shard_files.append(prev_file)
                        skipped_count += 1
                        skipped_rows += shard_len
                        total_rows += shard_len
                        valid_rows += int(prev_entry.get("valid_rows", 0))
                        total_ms += float(prev_entry.get("total_ms", 0.0))
                        continue

                slice_table = file_table.slice(row_start, shard_len)
                cols = slice_table.to_pydict()
                ocr_texts: list[str] = []
                backends: list[str] = []
                valids: list[bool] = []
                row_times: list[float] = []
                meta_jsons: list[str] = []
                shard_valid = 0
                shard_ms = 0.0

                for r_idx in range(shard_len):
                    try:
                        txt, b_used, ok, r_ms, meta = _ocr_single_row(cols, r_idx, src_file)
                    except Exception as exc:
                        if not skip_errors:
                            raise
                        txt = ""
                        b_used = f"error:{self.backend}"
                        ok = False
                        r_ms = 0.0
                        meta = {"error": f"{type(exc).__name__}: {exc}"}
                    ocr_texts.append(txt)
                    backends.append(b_used)
                    valids.append(bool(ok))
                    row_times.append(round(float(r_ms), 3))
                    meta_jsons.append(json.dumps(meta, ensure_ascii=False))
                    if ok:
                        shard_valid += 1
                    shard_ms += float(r_ms)

                out_cols = {
                    name: slice_table.column(name)
                    for name in slice_table.column_names
                    if name not in ("ocr_text", "backend_used", "ocr_valid", "total_ms", "metadata_json")
                }
                out_cols["ocr_text"] = pa.array(ocr_texts, type=pa.string())
                out_cols["backend_used"] = pa.array(backends, type=pa.string())
                out_cols["ocr_valid"] = pa.array(valids, type=pa.bool_())
                out_cols["total_ms"] = pa.array(row_times, type=pa.float64())
                out_cols["metadata_json"] = pa.array(meta_jsons, type=pa.string())
                out_table = pa.table(out_cols)

                tmp_shard = shard_path.with_name(f".{shard_path.name}.tmp.{os.getpid()}")
                try:
                    pq.write_table(
                        out_table,
                        str(tmp_shard),
                        compression=_resolve_parquet_compression(out_table.schema, compression),
                    )
                    os.replace(tmp_shard, shard_path)
                except BaseException:
                    tmp_shard.unlink(missing_ok=True)
                    raise

                completed_shards[shard_key] = {
                    "file": shard_name,
                    "source_file": str(src_file),
                    "row_start": int(row_start),
                    "row_end": int(row_end),
                    "rows": int(shard_len),
                    "valid_rows": int(shard_valid),
                    "total_ms": round(shard_ms, 3),
                }
                _save_manifest(manifest_path, manifest)

                shard_files.append(shard_path)
                completed_count += 1
                processed_rows += shard_len
                total_rows += shard_len
                valid_rows += shard_valid
                total_ms += shard_ms

        final_output: Path = out_target if single_file_target else shard_dir
        if single_file_target:
            tmp_final = out_target.with_name(f".{out_target.name}.tmp.{os.getpid()}")
            writer: pq.ParquetWriter | None = None
            try:
                if shard_files:
                    for sp in shard_files:
                        tbl = pq.read_table(sp)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                str(tmp_final),
                                tbl.schema,
                                compression=_resolve_parquet_compression(tbl.schema, compression),
                            )
                        writer.write_table(tbl)
                else:
                    empty_tbl = pa.table(
                        {
                            "ocr_text": pa.array([], type=pa.string()),
                            "backend_used": pa.array([], type=pa.string()),
                            "ocr_valid": pa.array([], type=pa.bool_()),
                            "total_ms": pa.array([], type=pa.float64()),
                            "metadata_json": pa.array([], type=pa.string()),
                        }
                    )
                    pq.write_table(empty_tbl, str(tmp_final))
                if writer is not None:
                    writer.close()
                    writer = None
                os.replace(tmp_final, out_target)
            except BaseException:
                if writer is not None:
                    writer.close()
                tmp_final.unlink(missing_ok=True)
                raise

        return {
            "input_files": [str(f) for f in files],
            "output_path": str(final_output),
            "shard_dir": str(shard_dir),
            "manifest_path": str(manifest_path),
            "total_rows": int(total_rows),
            "processed_rows": int(processed_rows),
            "skipped_rows": int(skipped_rows),
            "completed_shards": int(completed_count),
            "skipped_shards": int(skipped_count),
            "valid_rows": int(valid_rows),
            "total_ms": round(total_ms, 3),
            "shard_files": [str(p) for p in shard_files],
        }

    def close(self) -> None:
        """Release the lazily initialized VLM engine if loaded."""
        if self._ovis is not None:
            close_fn = getattr(self._ovis, "close", None)
            if callable(close_fn):
                close_fn()
            self._ovis = None

    def __enter__(self) -> AutoOCRPipeline:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class FastOCR(AutoOCRPipeline):
    """High-level facade for zero-training Apple Silicon OCR (`apple-vision`, `ovis`, or `auto`)."""

    @property
    def pipeline(self) -> AutoOCRPipeline:
        return self


def ocr_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    backend: OCRBackend = "auto",
    model_path: str | Path | None = None,
    input_col: str | None = None,
    shard_rows: int = 64,
    limit: int | None = None,
    resume: bool = True,
    compression: str = "zstd",
    dpi: int = 180,
    triage_dpi: int = 72,
    max_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
    batch_size: int = 16,
    prompt: str | None = None,
    schema: str | dict[str, Any] | None = None,
    output_format: OutputFormat = "markdown",
    table_mode: MarkdownTableMode = "gfm",
    strict_markdown_grammar: bool = False,
    zoom_recover: bool = False,
    adaptive_resolution: bool = True,
    deskew: bool | str = "auto",
    region_routing: bool = False,
    speculative: SpeculativeMode = "auto",
    kv_i8: bool | str | None = None,
    gdn_state_mode: str | None = None,
    drop_white_patches: bool | None = None,
    latin_only: bool | None = None,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
    skip_errors: bool = False,
    pipeline: AutoOCRPipeline | None = None,
) -> dict[str, Any]:
    """Stream OCR across Parquet document/image rows with sharded checkpoints and resume."""
    if pipeline is not None:
        return pipeline.ocr_parquet(
            input_path,
            output_path,
            input_col=input_col,
            shard_rows=shard_rows,
            limit=limit,
            resume=resume,
            compression=compression,
            dpi=dpi,
            triage_dpi=triage_dpi,
            max_pixels=max_pixels,
            batch_size=batch_size,
            prompt=prompt,
            schema=schema,
            output_format=output_format,
            table_mode=table_mode,
            skip_errors=skip_errors,
        )
    with AutoOCRPipeline(
        backend=backend,
        output_format=output_format,
        table_mode=table_mode,
        strict_markdown_grammar=strict_markdown_grammar,
        zoom_recover=zoom_recover,
        model_path=model_path,
        adaptive_resolution=adaptive_resolution,
        deskew=deskew,
        region_routing=region_routing,
        speculative=speculative,
        kv_i8=kv_i8,
        gdn_state_mode=gdn_state_mode,
        drop_white_patches=drop_white_patches,
        latin_only=latin_only,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
    ) as pipe:
        return pipe.ocr_parquet(
            input_path,
            output_path,
            input_col=input_col,
            shard_rows=shard_rows,
            limit=limit,
            resume=resume,
            compression=compression,
            dpi=dpi,
            triage_dpi=triage_dpi,
            max_pixels=max_pixels,
            batch_size=batch_size,
            prompt=prompt,
            schema=schema,
            output_format=output_format,
            table_mode=table_mode,
            skip_errors=skip_errors,
        )


__all__ = [
    "AutoOCRPipeline",
    "FastOCR",
    "ocr_parquet",
]
