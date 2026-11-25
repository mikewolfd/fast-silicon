"""Zero-training Qwen3.5 VLM OCR engine (`OvisOCR2`, `NuExtract3`) built on `fast-qwen3`."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import AbstractSet, Any, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from fast_shared.grammar import (
    GrammarHelper,
    apply_loop_breaker_penalties_1d,
    compute_loop_breaker_penalties,
    _token_tail_repeat_4gram_rate,
    detect_catastrophic_token_loop,
    normalize_json_schema,
    safe_argmax_1d,
    sample_unconstrained,
)
from fast_shared.pld_drafter import filter_drafts_with_grammar
from fast_shared.kv_quant import KVMode, normalize_kv_mode
from fast_qwen3.kernels import gpu_argmax_conf, gpu_argmax_nan, stats_to_logprob_margin
from fast_qwen3.model import FastQwen3Model, Qwen3Config
from fast_qwen3.mtp_drafter import verify_draft_sequence_b1
from fast_qwen3.multimodal_batch import (
    _make_hybrid_cache,
    batched_multimodal_generate,
    compute_safe_white_patch_keep_mask,
    host_phys_memory_gb,
)
from fast_qwen3.tokenizer import ENDOFTEXT, IM_END, IM_START, THINK_CLOSE, THINK_OPEN, FastQwen3Tokenizer, QwenVocabRemapper

from .apple_vision import AppleVisionOCR, deskew_document_image, normalize_deskew_mode
from .markdown_struct import DynamicTableRowGrammar, canonicalize_markdown, validate_markdown
from .mrope import compute_mrope_positions, precompute_mrope_trig
from .numeric_recovery import (
    align_json_values_to_lines,
    compute_missing_numeric_crop_boxes_px,
    find_filtered_missing_numbers,
    find_recovered_numbers,
)
from .pld_drafter import OCRPLDDrafter
from .preprocess import (
    build_qwen_vision_inputs,
    crop_to_ink,
    load_image,
    render_pdf_pages,
    rotate_image_cw,
    smart_resize_qwen,
)
from .region_router import (
    REGION_CROP_MIN_PIXELS,
    build_ordered_pld_corpus,
    compute_adaptive_max_pixels,
)
from .types import (
    LogprobConfidence,
    MarkdownTableMode,
    OCRResult,
    OCRTimings,
    OutputFormat,
    SpeculativeMode,
    summarize_logprob_confidence,
)
from .vision import FastQwen35VisionModel

DEFAULT_OVIS_REPO: str = "mlx-community/OvisOCR2-8bit"
OCR_MODEL_PRESETS: dict[str, str] = {
    "ovis": DEFAULT_OVIS_REPO,
    "ovisocr2": DEFAULT_OVIS_REPO,
    "ovis2": DEFAULT_OVIS_REPO,
    "nuextract": "numind/NuExtract3-mlx-4bits",
    "nuextract3": "numind/NuExtract3-mlx-4bits",
}
OCR_PRESET_FALLBACKS: dict[str, tuple[str, ...]] = {
    "nuextract": ("mlx-community/NuExtract3-bf16",),
    "nuextract3": ("mlx-community/NuExtract3-bf16",),
}

DEFAULT_OVIS_PROMPT: str = (
    "Extract all readable content from the image in natural human reading order and output the result as a "
    "single Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are '
    "bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: "
    "<table>...</table>. Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

DEFAULT_GFM_STRICT_PROMPT: str = (
    "Extract all readable content from the image in natural human reading order and output the result as a "
    "single GitHub-Flavored Markdown document. Format every table strictly as a standard GFM pipe table "
    "(| col1 | col2 |). Format formulas as LaTeX. Preserve all original text and numbers verbatim."
)

_OCR_EXTRA_CHARS: str = (
    "—–“”‘’•§¶°±×÷€£¥¢½¼¾©®™✓✔✗☐☑"
    "−→←↑↓≤≥≠≈∞√∑∫†‡‑"
    "αβγδεζηθλμνξπρστφχψωΔΘΛΞΠΣΦΨΩ"
    "₀₁₂₃₄₅₆₇₈₉"
)

_LATIN_LANG_CODES: frozenset[str] = frozenset({
    "en", "eng", "fr", "fra", "fre", "de", "deu", "ger", "es", "spa", "pt", "por",
    "it", "ita", "nl", "nld", "dut", "da", "dan", "sv", "swe", "no", "nob", "nno",
    "nb", "nn", "fi", "fin", "is", "isl", "pl", "pol", "cs", "ces", "cze", "sk",
    "slk", "slo", "sl", "slv", "hr", "hrv", "hu", "hun", "ro", "ron", "rum",
    "tr", "tur", "az", "aze", "et", "est", "lv", "lav", "lt", "lit", "ca", "cat",
    "eu", "eus", "gl", "glg", "ga", "gle", "cy", "cym", "mt", "mlt", "sq", "sqi",
    "id", "ind", "ms", "msa", "tl", "tgl", "sw", "swa", "af", "afr", "la", "lat",
    "latin", "latn",
})
_NON_LATIN_SCRIPTS: frozenset[str] = frozenset({
    "hans", "hant", "jpan", "kore", "arab", "cyrl", "deva", "grek", "hebr",
    "thai", "hang", "kana", "hira", "beng", "taml", "telu", "gujr", "guru",
    "knda", "mlym", "orya", "sinh", "mymr", "khmr", "laoo", "geor", "armn", "ethi",
})


def _languages_use_latin_only(languages: Sequence[str] | None) -> bool:
    """Return False when `languages` contains any non-Latin language or script tag."""
    if not languages:
        return True
    for raw in languages:
        tag = str(raw or "").strip().lower()
        if not tag:
            continue
        parts = [p for p in re.split(r"[-_]", tag) if p]
        if not parts:
            continue
        if "latn" in parts:
            continue
        if any(p in _NON_LATIN_SCRIPTS for p in parts):
            return False
        if parts[0] not in _LATIN_LANG_CODES:
            return False
    return True


_CACHED_APPLE10_NAX: bool | None = None


def _host_supports_apple10_nax() -> bool:
    global _CACHED_APPLE10_NAX
    if _CACHED_APPLE10_NAX is None:
        try:
            from fast_shared.hw_check import supports_apple10_nax

            _CACHED_APPLE10_NAX = bool(supports_apple10_nax())
        except Exception:
            _CACHED_APPLE10_NAX = False
    return _CACHED_APPLE10_NAX


def resolve_ovis_kv_mode(
    kv_i8: KVMode | bool | str,
    *,
    batch_size: int,
    alloc_ctx: int,
    host_mem_gb: float | None = None,
    head_dim: int = 256,
    apple10_nax: bool | None = None,
) -> KVMode:
    """Resolve KV cache precision for Ovis OCR, keeping NAX/medium batches on BF16 and guarding extreme pressure."""
    if isinstance(kv_i8, bool):
        return "q8" if kv_i8 else "bf16"
    raw_s = str(kv_i8).strip().lower()
    if raw_s != "auto":
        return normalize_kv_mode(kv_i8)
    if int(head_dim) % 64 != 0:
        return "bf16"
    host_gb = float(host_mem_gb) if host_mem_gb is not None else host_phys_memory_gb()
    is_nax = _host_supports_apple10_nax() if apple10_nax is None else bool(apple10_nax)
    bs = max(1, int(batch_size))
    ctx = max(1, int(alloc_ctx))
    if is_nax:
        # OvisOCR2 has only 6 Full-Attn KV layers (18 GDN): Apple10 NAX BF16 SDPA beats q4_quarot through L=5120 / B=64.
        if host_gb <= 28.0:
            if ctx >= 16384 or (bs * ctx) >= 262_144:
                return "q4_quarot"
        elif ctx >= 32768 or (bs * ctx) >= 524_288:
            return "q4_quarot"
        return "bf16"
    ctx_cutoff = 8192 if host_gb <= 28.0 else 16384
    pressure_cutoff = 262_144 if host_gb <= 28.0 else 524_288
    if ctx >= ctx_cutoff or (bs * ctx) >= pressure_cutoff:
        return "q4_quarot"
    return "bf16"


def _resolve_ovis_min_pixels(
    image_size: tuple[int, int],
    *,
    min_pixels: int | None,
    max_pixels: int | None,
    output_format: OutputFormat = "markdown",
    schema: Any = None,
) -> int:
    """Resolve `min_pixels` for ViT preprocessing, auto-upscaling compact Markdown tables (<150K px) when unpinned."""
    if min_pixels is not None:
        return max(1024, int(min_pixels))
    if max_pixels is None and output_format == "markdown" and schema is None:
        w, h = int(image_size[0]), int(image_size[1])
        area = w * h
        min_side = min(w, h)
        if 50_000 <= area < 150_000 and min_side >= 100 and (float(max(w, h)) / float(min_side)) <= 4.0:
            return 262_144
    return 65_536


def _find_local_hf_snapshot(repo_id: str) -> Path | None:
    slug = "models--" + repo_id.replace("/", "--")
    hub_snaps = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    if hub_snaps.is_dir():
        for snap in reversed(sorted(s for s in hub_snaps.iterdir() if s.is_dir())):
            if (snap / "config.json").is_file() and (snap / "tokenizer.json").is_file() and any(snap.glob("*.safetensors")):
                return snap.resolve()
    return None


def resolve_ocr_model_dir(
    model_path_or_repo: str | Path | None = None,
    *,
    local_files_only: bool = True,
) -> Path:
    """Resolve an OCR VLM snapshot path from local path, preset alias, or Hugging Face Hub cache."""
    raw = str(model_path_or_repo).strip() if model_path_or_repo is not None else os.environ.get("FAST_OCR_MODEL_DIR", "")
    preset_key = raw.lower() if raw else ""
    if raw:
        cand = Path(raw).expanduser()
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
        repo_id = OCR_MODEL_PRESETS.get(preset_key, raw)
    else:
        repo_id = DEFAULT_OVIS_REPO

    as_dir = Path(repo_id).expanduser()
    if as_dir.is_dir() and (as_dir / "config.json").is_file():
        return as_dir.resolve()

    local_snap = _find_local_hf_snapshot(repo_id)
    if local_snap is not None:
        return local_snap

    for fb_repo in OCR_PRESET_FALLBACKS.get(preset_key, ()):
        fb_snap = _find_local_hf_snapshot(fb_repo)
        if fb_snap is not None:
            return fb_snap

    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"],
        )
    ).resolve()
    if not (downloaded / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete OCR model snapshot for {repo_id!r} at {downloaded}")
    return downloaded


def resolve_ocr_mlx_limits(
    *,
    host_mem_gb: float | None = None,
    metal_working_set_gb: float | None = None,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
) -> tuple[int | None, int | None, Any | None]:
    """Compute recommended MLX memory and wired-cache byte limits for Apple Silicon OCR workloads."""
    from fast_shared.runtime_lessons import recommend_silicon_budget

    eff_host_gb = float(host_mem_gb) if host_mem_gb is not None else 0.0
    if eff_host_gb <= 0.0:
        try:
            from fast_shared.hw_check import probe_host_hardware

            hw = probe_host_hardware()
            raw_gb = hw.get("mem_gb") or (float(hw.get("mem_bytes", 0)) / float(1024**3))
            eff_host_gb = float(raw_gb or 0.0)
        except Exception:
            eff_host_gb = 0.0

    eff_metal_ws_gb = metal_working_set_gb
    if eff_metal_ws_gb is None:
        try:
            dev_info = mx.device_info()
            ws_bytes = float(dev_info.get("max_recommended_working_set_size") or 0.0)
            if ws_bytes > 0.0:
                eff_metal_ws_gb = ws_bytes / float(1024**3)
        except Exception:
            eff_metal_ws_gb = None

    budget = (
        recommend_silicon_budget(
            eff_host_gb,
            metal_working_set_gb=eff_metal_ws_gb,
            workload="extraction",
        )
        if eff_host_gb > 0.0
        else None
    )

    mem_gb = memory_limit_gb if memory_limit_gb is not None else (budget.memory_limit_gb if budget is not None else None)
    cache_gb = cache_limit_gb if cache_limit_gb is not None else (budget.cache_limit_gb if budget is not None else None)

    mem_bytes = int(round(float(mem_gb) * float(1024**3))) if (mem_gb is not None and float(mem_gb) > 0.0) else None
    cache_bytes = int(round(float(cache_gb) * float(1024**3))) if (cache_gb is not None and float(cache_gb) > 0.0) else None
    return mem_bytes, cache_bytes, budget


def build_ocr_vocab_subset(
    tokenizer: FastQwen3Tokenizer,
    *,
    full_vocab_size: int,
    special_ids: Sequence[int] = (),
    target_size: int | None = None,
    latin_only: bool = True,
    languages: Sequence[str] | None = None,
) -> QwenVocabRemapper | None:
    """Build a lossless OCR vocabulary subset preserving special tokens, HTML/LaTeX markup, and symbols.

    Returns `None` (keep full vocabulary) when `latin_only=False` or `languages` includes non-Latin scripts.
    """
    if not latin_only or not _languages_use_latin_only(languages):
        return None
    base = QwenVocabRemapper.build_ascii_latin_subset(
        tokenizer.backend_tokenizer,
        full_vocab_size=int(full_vocab_size),
        target_size=target_size,
    )
    if target_size is None:
        # Ensure explicit vision/ChatML specials & OCR symbols (bare and space-prefixed) are never dropped
        extra_keep: set[int] = {int(x) for x in base.new_to_old.tolist()}
        for sid in special_ids:
            if 0 <= int(sid) < int(full_vocab_size):
                extra_keep.add(int(sid))
        for sym in _OCR_EXTRA_CHARS:
            for cand in (sym, " " + sym):
                for tid in tokenizer.encode(cand, add_special_tokens=False):
                    if 0 <= int(tid) < int(full_vocab_size):
                        extra_keep.add(int(tid))
        for tok_str in (IM_START, IM_END, ENDOFTEXT, THINK_OPEN, THINK_CLOSE):
            tid = tokenizer.backend_tokenizer.token_to_id(tok_str)
            if tid is not None and 0 <= int(tid) < int(full_vocab_size):
                extra_keep.add(int(tid))
        if len(extra_keep) == base.vocab_size:
            return base
        return QwenVocabRemapper(np.array(sorted(extra_keep), dtype=np.int32), full_vocab_size=int(full_vocab_size))
    return base


def resolve_vision_rotation_cw(vision_res: OCRResult | None, *, auto_orient: bool) -> int:
    """Resolve required clockwise rotation (0/90/180/270) from either raw or auto-oriented pre-analysis."""
    if not auto_orient or vision_res is None:
        return 0
    applied = int(vision_res.rotation_applied_cw) % 360
    if applied != 0:
        return applied
    if vision_res.analysis is not None:
        return int(vision_res.analysis.inferred_rotation_cw) % 360
    return 0


def _filter_table_safe_drafts(
    table_gram: DynamicTableRowGrammar,
    drafts: Sequence[int],
    *,
    decode_fn: Callable[[Sequence[int]], str],
    vocab_size: int,
    stop_ids: AbstractSet[int] = (),
    old_to_new: np.ndarray | None = None,
) -> list[int]:
    """Prefix-filter `drafts` so every step is table-grammar legal and triggers zero FF tokens."""
    if not drafts:
        return []
    sim = table_gram.clone()
    safe: list[int] = []
    for raw_dt in drafts:
        dt = int(raw_dt)
        if dt in stop_ids or dt < 0 or dt >= int(vocab_size):
            break
        if old_to_new is not None and (dt >= len(old_to_new) or int(old_to_new[dt]) < 0):
            break
        mask = sim.prepare_mask(vocab_size=vocab_size)
        if mask is not None and (dt >= mask.shape[0] or not bool(mask[dt])):
            break
        # Stop before any lock-activating or row-closing FF token so the verify bonus token inserts FF cleanly.
        if sim.advance_text_and_maybe_lock(decode_fn([dt]), dt):
            break
        safe.append(dt)
    return safe


def _consume_table_token_and_ff(
    table_gram: DynamicTableRowGrammar,
    token_id: int,
    *,
    decode_fn: Callable[[Sequence[int]], str],
    stop_ids: AbstractSet[int],
) -> list[int]:
    """Advance `table_gram` by `token_id` and return filtered fast-forward token IDs."""
    delta = decode_fn([int(token_id)])
    return [int(t) for t in table_gram.advance_text_and_maybe_lock(delta, int(token_id)) if int(t) not in stop_ids]


def _filter_json_safe_drafts(
    json_matcher: Any,
    json_gram: Any,
    drafts: Sequence[int],
    *,
    vocab_size: int,
    stop_ids: AbstractSet[int] = (),
    new_to_old: np.ndarray,
    old_to_new: np.ndarray | None = None,
) -> list[int]:
    """Prefix-filter `drafts` so every speculative step matches greedy `json_matcher` semantics."""
    return filter_drafts_with_grammar(
        drafts,
        json_matcher,
        stop_ids,
        json_gram=json_gram,
        vocab_size=vocab_size,
        new_to_old=new_to_old,
        old_to_new=old_to_new,
    )


def _normalize_per_item_int(
    value: int | Sequence[int | None] | None,
    n_items: int,
    name: str,
) -> tuple[int | None, list[int | None] | None]:
    """Normalize a scalar or per-item int sequence into `(scalar_override, per_item_list)`."""
    if value is None:
        return None, None
    if isinstance(value, (int, np.integer)):
        return int(value), None
    seq = [None if x is None else int(x) for x in value]
    if len(seq) != n_items:
        raise ValueError(f"Expected {n_items} {name} entries, got {len(seq)}")
    return None, seq


def _normalize_confidence_mode(mode: bool | str | None, *, default: str = "auto") -> str:
    if mode is None:
        return default
    if isinstance(mode, bool):
        return "on" if mode else "off"
    norm = str(mode).strip().lower()
    if norm in {"auto", "on", "true", "1", "yes"}:
        return "auto" if norm == "auto" else "on"
    if norm in {"off", "false", "0", "no", "none"}:
        return "off"
    raise ValueError(f"Invalid collect_logprob_confidence mode {mode!r}; expected 'auto', True, or False")


def _load_accelerate_vdsp() -> Any | None:
    if sys.platform != "darwin":
        return None
    try:
        acc = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        acc.vDSP_maxv.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.POINTER(ctypes.c_float), ctypes.c_ulong]
        acc.vDSP_vsadd.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_ulong,
        ]
        acc.vvexpf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        acc.vDSP_sve.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.POINTER(ctypes.c_float), ctypes.c_ulong]
        return acc
    except Exception:
        return None


_VDSP_LIB: Any | None = _load_accelerate_vdsp()


class _StepConfidenceState:
    """Reusable per-thread scratch & Accelerate vDSP kernels for zero-copy step logprob + margin."""

    def __init__(self) -> None:
        self._vdsp = _VDSP_LIB
        self._scratch = np.empty(160_000, dtype=np.float32)
        self._c_max = ctypes.c_float()
        self._c_shift = ctypes.c_float()
        self._c_sum = ctypes.c_float()
        self._c_m_l = ctypes.c_float()
        self._c_m_r = ctypes.c_float()
        self._c_n = ctypes.c_int()

    def compute(self, logits_1d: np.ndarray, chosen_idx: int) -> tuple[float, float]:
        n = int(logits_1d.shape[0]) if getattr(logits_1d, "ndim", 0) == 1 else 0
        if n <= 0 or chosen_idx < 0 or chosen_idx >= n:
            return 0.0, 0.0
        arr = (
            logits_1d
            if (logits_1d.dtype == np.float32 and logits_1d.flags.c_contiguous)
            else np.ascontiguousarray(logits_1d, dtype=np.float32)
        )
        chosen = float(arr[chosen_idx])
        if not math.isfinite(chosen):
            return -60.0, 0.0

        vdsp = self._vdsp
        if vdsp is not None and n >= 64:
            if n > int(self._scratch.shape[0]):
                self._scratch = np.empty(n, dtype=np.float32)
            ptr = arr.ctypes.data
            sptr = self._scratch.ctypes.data
            vdsp.vDSP_maxv(ptr, 1, ctypes.byref(self._c_max), n)
            m = float(self._c_max.value)
            if not math.isfinite(m):
                m = chosen
            self._c_shift.value = -m
            vdsp.vDSP_vsadd(ptr, 1, ctypes.byref(self._c_shift), sptr, 1, n)
            self._c_n.value = n
            vdsp.vvexpf(sptr, sptr, ctypes.byref(self._c_n))
            vdsp.vDSP_sve(sptr, 1, ctypes.byref(self._c_sum), n)
            z = float(self._c_sum.value)
            lse = m + math.log(max(z, 1e-30))
            lp = min(0.0, chosen - lse)

            other_max = -1e30
            if chosen_idx > 0:
                vdsp.vDSP_maxv(ptr, 1, ctypes.byref(self._c_m_l), chosen_idx)
                v_l = float(self._c_m_l.value)
                if v_l > other_max:
                    other_max = v_l
            if chosen_idx + 1 < n:
                vdsp.vDSP_maxv(ptr + (chosen_idx + 1) * 4, 1, ctypes.byref(self._c_m_r), n - chosen_idx - 1)
                v_r = float(self._c_m_r.value)
                if v_r > other_max:
                    other_max = v_r
            margin = (chosen - other_max) if other_max > -1e29 else 0.0
            return lp, margin

        m = float(np.max(arr))
        z = float(np.sum(np.exp(arr - m)))
        lse = m + math.log(max(z, 1e-30))
        lp = min(0.0, chosen - lse)
        other_max = -1e30
        if chosen_idx > 0:
            other_max = max(other_max, float(np.max(arr[:chosen_idx])))
        if chosen_idx + 1 < n:
            other_max = max(other_max, float(np.max(arr[chosen_idx + 1 :])))
        margin = (chosen - other_max) if other_max > -1e29 else 0.0
        return lp, margin


def compute_step_logprob_margin(logits_1d: np.ndarray, chosen_idx: int) -> tuple[float, float]:
    """Compute `(logprob, logit_margin)` for a single step without full-softmax float64 copies."""
    return _StepConfidenceState().compute(logits_1d, chosen_idx)


class _StepConfidenceWorker:
    """Persistent single background thread computing step logprobs/margins overlapped with GPU steps."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._state = _StepConfidenceState()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="fast-ocr-logprob-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            kind = item[0]
            if kind == "stop":
                return
            if kind == "flush":
                item[1].set()
                continue
            _, arr, chosen_idx, out_lps, out_margins = item
            lp, margin = self._state.compute(arr, chosen_idx)
            out_lps.append(lp)
            out_margins.append(margin)

    def submit(
        self,
        logits_1d: np.ndarray,
        chosen_idx: int,
        out_lps: list[float],
        out_margins: list[float],
    ) -> None:
        if self._closed:
            lp, margin = self._state.compute(logits_1d, chosen_idx)
            out_lps.append(lp)
            out_margins.append(margin)
            return
        self._queue.put(("step", logits_1d, int(chosen_idx), out_lps, out_margins))

    def flush(self) -> None:
        if self._closed:
            return
        done = threading.Event()
        self._queue.put(("flush", done))
        done.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(("stop",))
        self._thread.join(timeout=1.0)


class _StepConfidenceSession:
    """Per-page accumulator backed by GPU scalar stats, `_StepConfidenceWorker`, or synchronous evaluation."""

    def __init__(self, worker: _StepConfidenceWorker | None) -> None:
        self._worker = worker
        self._sync_state = _StepConfidenceState() if worker is None else None
        self._used_worker = False
        self.logprobs: list[float] = []
        self.margins: list[float] = []

    def submit(self, logits_1d: np.ndarray, chosen_idx: int) -> None:
        n = int(logits_1d.shape[0]) if getattr(logits_1d, "ndim", 0) == 1 else 0
        if n <= 0 or chosen_idx < 0 or chosen_idx >= n:
            return
        if self._worker is not None:
            self._used_worker = True
            self._worker.submit(logits_1d, chosen_idx, self.logprobs, self.margins)
        else:
            assert self._sync_state is not None
            lp, margin = self._sync_state.compute(logits_1d, chosen_idx)
            self.logprobs.append(lp)
            self.margins.append(margin)

    def record_direct(self, lp: float, margin: float) -> None:
        """Append GPU-computed `(logprob, margin)` directly, flushing any prior CPU-fallback worker items first."""
        if self._used_worker and self._worker is not None:
            self._worker.flush()
        self.logprobs.append(float(lp))
        self.margins.append(float(margin))

    def finish(self) -> LogprobConfidence:
        if self._used_worker and self._worker is not None:
            self._worker.flush()
        return summarize_logprob_confidence(self.logprobs, margins=self.margins)


class FastOvisOCREngine:
    """Zero-training high-speed Qwen3.5 VLM OCR engine built on `fast-qwen3` + `FastQwen35VisionModel`."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        output_format: OutputFormat = "markdown",
        table_mode: MarkdownTableMode = "gfm",
        strict_markdown_grammar: bool = False,
        zoom_recover: bool = False,
        crop_margins: bool = True,
        crop_pad_px: int = 24,
        auto_orient: bool = True,
        adaptive_resolution: bool = True,
        deskew: bool | str = "auto",
        speculative: SpeculativeMode = "auto",
        draft_k: int = 4,
        vocab_size: int | None = None,
        vit_quant_bits: int | None = None,
        prefill_chunk_size: int = 1024,
        max_tokens: int = 8192,
        loop_breaker: bool = False,
        auto_loop_recover: bool = True,
        kv_i8: KVMode | bool | str = "auto",
        gdn_state_mode: str = "auto",
        drop_white_patches: bool = False,
        latin_only: bool = True,
        languages: Sequence[str] | None = None,
        enable_fusion: bool = True,
        warmup: bool = False,
        memory_limit_gb: float | None = None,
        cache_limit_gb: float | None = None,
        apply_memory_limits: bool = True,
        collect_logprob_confidence: bool | str = "auto",
    ) -> None:
        self._prev_mlx_memory_limit: int | None = None
        self._prev_mlx_cache_limit: int | None = None
        self._mlx_budget: Any | None = None
        if apply_memory_limits:
            mem_bytes, cache_bytes, budget = resolve_ocr_mlx_limits(
                memory_limit_gb=memory_limit_gb,
                cache_limit_gb=cache_limit_gb,
            )
            self._mlx_budget = budget
            if mem_bytes is not None and callable(getattr(mx, "set_memory_limit", None)):
                self._prev_mlx_memory_limit = int(mx.set_memory_limit(mem_bytes))
            if cache_bytes is not None and callable(getattr(mx, "set_cache_limit", None)):
                self._prev_mlx_cache_limit = int(mx.set_cache_limit(cache_bytes))

        if not isinstance(kv_i8, bool) and str(kv_i8).strip().lower() != "auto":
            _ = normalize_kv_mode(kv_i8)

        self.model_dir = resolve_ocr_model_dir(model_path)
        self.raw_config: dict[str, Any] = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.output_format: OutputFormat = output_format
        self.table_mode: MarkdownTableMode = table_mode
        self.strict_markdown_grammar: bool = bool(strict_markdown_grammar)
        self.zoom_recover: bool = bool(zoom_recover)
        self.crop_margins: bool = bool(crop_margins)
        self.crop_pad_px: int = max(0, int(crop_pad_px))
        self.auto_orient: bool = bool(auto_orient)
        self.adaptive_resolution: bool = bool(adaptive_resolution)
        self.deskew: str = normalize_deskew_mode(deskew)
        self.speculative: SpeculativeMode = speculative
        self.draft_k: int = max(1, int(draft_k))
        self.vit_quant_bits: int | None = vit_quant_bits
        self.prefill_chunk_size: int = max(0, int(prefill_chunk_size))
        self.max_tokens: int = max(1, int(max_tokens))
        self.loop_breaker: bool = bool(loop_breaker)
        self.auto_loop_recover: bool = bool(auto_loop_recover)
        self.kv_i8: KVMode | bool | str = kv_i8
        self.gdn_state_mode: str = str(gdn_state_mode)
        self.drop_white_patches: bool = bool(drop_white_patches)
        self.latin_only: bool = bool(latin_only)
        self.languages: tuple[str, ...] | None = tuple(languages) if languages is not None else None
        self.collect_logprob_confidence: str = _normalize_confidence_mode(collect_logprob_confidence)
        self._confidence_worker: _StepConfidenceWorker | None = None

        self.image_token_id: int = int(self.raw_config.get("image_token_id", 248056))
        self.vision_start_token_id: int = int(self.raw_config.get("vision_start_token_id", 248053))
        self.vision_end_token_id: int = int(self.raw_config.get("vision_end_token_id", 248054))
        self.video_token_id: int = int(self.raw_config.get("video_token_id", 248057))

        lm_cfg = Qwen3Config.from_dict(self.raw_config)
        self.tokenizer = FastQwen3Tokenizer(self.model_dir, full_vocab_size=lm_cfg.vocab_size)
        specials = (
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            self.video_token_id,
        )
        self.vocab_remap: QwenVocabRemapper | None = build_ocr_vocab_subset(
            self.tokenizer,
            full_vocab_size=lm_cfg.vocab_size,
            special_ids=specials,
            target_size=vocab_size,
            latin_only=self.latin_only,
            languages=self.languages,
        )
        if self.vocab_remap is not None and self.vocab_remap.vocab_size >= lm_cfg.vocab_size:
            self.vocab_remap = None

        self.vision_model = FastQwen35VisionModel.from_pretrained(
            self.model_dir,
            quantize_bits=vit_quant_bits,
        )
        self.lm, _ = FastQwen3Model.from_pretrained(
            self.model_dir,
            vocab_remap=self.vocab_remap,
            enable_fusion=enable_fusion,
            load_mtp_tensors=False,
        )
        self._new_to_old: np.ndarray = (
            self.vocab_remap.new_to_old
            if self.vocab_remap is not None
            else np.arange(self.lm.cfg.vocab_size, dtype=np.int32)
        )
        self._new_to_old_mx: mx.array | None = (
            mx.array(self.vocab_remap.new_to_old, dtype=mx.int32)
            if self.vocab_remap is not None
            else None
        )
        self._old_to_new: np.ndarray | None = (
            self.vocab_remap.old_to_new if self.vocab_remap is not None else None
        )
        self._vision_prepass: AppleVisionOCR | None = None
        if warmup:
            self._warmup()

    def _get_confidence_worker(self) -> _StepConfidenceWorker:
        worker = getattr(self, "_confidence_worker", None)
        if worker is None:
            worker = _StepConfidenceWorker()
            self._confidence_worker = worker
        return worker

    def _get_vision_prepass(self) -> AppleVisionOCR:
        if self._vision_prepass is None:
            # 3-band inaccurate/fast-LC Vision gives ~98% token recall for PLD + accurate 360-deg orientation
            self._vision_prepass = AppleVisionOCR(
                level="accurate",
                bands=3,
                language_correction=False,
                auto_orient=False,
            )
        return self._vision_prepass

    def _warmup(self) -> None:
        tiny = Image.new("RGB", (64, 64), (255, 255, 255))
        _ = self.ocr_image(
            tiny,
            output_format="raw",
            crop_margins=False,
            auto_orient=False,
            speculative="off",
            max_tokens=2,
        )
        mx.clear_cache()

    def _build_prompt_ids(self, num_visual_tokens: int, instruction: str) -> list[int]:
        prefix = "<|im_start|>user\n<|vision_start|>"
        suffix = (
            f"<|vision_end|>\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        pre_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suf_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        return pre_ids + ([self.image_token_id] * int(num_visual_tokens)) + suf_ids

    def ocr_image(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        page_index: int = 0,
        prompt: str | None = None,
        schema: str | Mapping[str, Any] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        strict_markdown_grammar: bool | None = None,
        zoom_recover: bool | None = None,
        crop_margins: bool | None = None,
        auto_orient: bool | None = None,
        adaptive_resolution: bool | None = None,
        deskew: bool | str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        drop_white_patches: bool | None = None,
        native_pdf_text: str | None = None,
        speculative: SpeculativeMode | None = None,
        draft_k: int | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        pre_analysis: OCRResult | None = None,
        auto_loop_recover: bool | None = None,
        collect_logprob_confidence: bool | str | None = None,
        _force_loop_breaker: bool = False,
        _loop_retry_depth: int = 0,
    ) -> OCRResult:
        """Run end-to-end OCR on one image with Stage-0 crop/orient/adaptive-res, MRoPE VLM decode, PLD, and Markdown cleanup."""
        fmt: OutputFormat = output_format or self.output_format
        tmode: MarkdownTableMode = table_mode or self.table_mode
        strict_md = self.strict_markdown_grammar if strict_markdown_grammar is None else bool(strict_markdown_grammar)
        do_zoom = (self.zoom_recover if zoom_recover is None else bool(zoom_recover)) and (schema is None)
        do_crop = self.crop_margins if crop_margins is None else bool(crop_margins)
        do_orient = self.auto_orient if auto_orient is None else bool(auto_orient)
        do_adaptive = bool(getattr(self, "adaptive_resolution", True)) if adaptive_resolution is None else bool(adaptive_resolution)
        eff_deskew = normalize_deskew_mode(getattr(self, "deskew", "off") if deskew is None else deskew)
        eff_drop_white = (
            bool(getattr(self, "drop_white_patches", False))
            if drop_white_patches is None
            else bool(drop_white_patches)
        )
        spec_req: SpeculativeMode = self.speculative if speculative is None else speculative
        has_native_pdf = bool(native_pdf_text and str(native_pdf_text).strip())
        if draft_k is None:
            k_draft = max(self.draft_k, 6) if has_native_pdf else self.draft_k
        else:
            k_draft = max(1, int(draft_k))
        tok_cap = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        do_loop_recover = bool(getattr(self, "auto_loop_recover", True)) if auto_loop_recover is None else bool(auto_loop_recover)
        eff_loop_breaker = bool(getattr(self, "loop_breaker", False) or _force_loop_breaker)
        eff_conf_mode = _normalize_confidence_mode(
            getattr(self, "collect_logprob_confidence", "auto")
            if collect_logprob_confidence is None
            else collect_logprob_confidence
        )
        conf_session = (
            _StepConfidenceSession(self._get_confidence_worker())
            if eff_conf_mode in ("auto", "on")
            else None
        )

        if prompt is not None:
            instruction = str(prompt)
        elif strict_md and tmode == "gfm":
            instruction = DEFAULT_GFM_STRICT_PROMPT
        else:
            instruction = DEFAULT_OVIS_PROMPT

        mx.clear_cache()
        t_total0 = time.perf_counter()

        t_prep0 = time.perf_counter()
        img = load_image(source)
        vision_res = pre_analysis
        deskew_meta: dict[str, Any] | None = None
        if eff_deskew != "off":
            try:
                img, d_meta = deskew_document_image(img, mode=eff_deskew)
                if d_meta.get("applied"):
                    vision_res = None
                    deskew_meta = d_meta
                elif eff_deskew == "on":
                    deskew_meta = d_meta
            except Exception:
                if eff_deskew == "on":
                    deskew_meta = {"applied": False, "reason": "deskew_error"}
        vision_ms = float(vision_res.timings.apple_vision_ms) if vision_res is not None else 0.0

        # Run lightweight ANE Vision pre-pass when needed for orientation, zoom recovery, or vision+self speculative drafting
        if vision_res is None and (do_orient or do_zoom or spec_req in ("auto", "vision+self")):
            try:
                vision_res = self._get_vision_prepass().recognize_image(img, page_index=page_index, output_format="raw")
                vision_ms = float(vision_res.timings.apple_vision_ms)
            except Exception:
                vision_res = None

        rot_cw = resolve_vision_rotation_cw(vision_res, auto_orient=do_orient)
        if rot_cw != 0:
            img = rotate_image_cw(img, rot_cw)
        oriented_img = img

        crop_box: tuple[int, int, int, int] | None = None
        if do_crop:
            img, crop_box = crop_to_ink(img, pad_px=self.crop_pad_px)

        eff_min_pixels = _resolve_ovis_min_pixels(
            img.size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            output_format=fmt,
            schema=schema,
        )
        if max_pixels is not None:
            eff_max_pixels = max(eff_min_pixels, int(max_pixels))
        elif do_adaptive and vision_res is not None and vision_res.analysis is not None:
            eff_max_pixels = compute_adaptive_max_pixels(
                vision_res.analysis,
                oriented_img.size,
                cropped_size=img.size,
            )
        else:
            eff_max_pixels = 16_777_216

        if eff_min_pixels == 65_536 and eff_max_pixels == 16_777_216:
            vi = build_qwen_vision_inputs(img)
        else:
            vi = build_qwen_vision_inputs(img, min_pixels=eff_min_pixels, max_pixels=eff_max_pixels)
        prompt_ids = self._build_prompt_ids(vi.num_visual_tokens, instruction)
        pos_3d, rope_delta = compute_mrope_positions(
            prompt_ids,
            image_grid_thw=[vi.grid_thw],
            image_token_id=self.image_token_id,
            spatial_merge_size=self.vision_model.spatial_merge_size,
        )
        mrope_sec = self.raw_config.get("text_config", {}).get("rope_parameters", {}).get("mrope_section", (11, 11, 10))
        cos_a, sin_a = precompute_mrope_trig(
            pos_3d,
            rot_dims=self.lm.cfg.rot_dims,
            rope_theta=self.lm.cfg.rope_theta,
            mrope_section=tuple(int(x) for x in mrope_sec),
            dtype=mx.bfloat16,
        )
        patch_sz = int(getattr(getattr(self.vision_model, "cfg", None), "patch_size", 16))
        merge_sz = int(getattr(self.vision_model, "spatial_merge_size", 2))
        block_sz = patch_sz * merge_sz
        keep_mask: np.ndarray | None = None
        if (
            eff_drop_white
            and block_sz > 0
            and vi.resized_height > 0
            and vi.resized_width > 0
            and (vi.resized_height % block_sz) == 0
            and (vi.resized_width % block_sz) == 0
            and ((vi.resized_height // block_sz) * (vi.resized_width // block_sz)) == vi.num_visual_tokens
        ):
            km = compute_safe_white_patch_keep_mask(
                img,
                vi.resized_height,
                vi.resized_width,
                patch_size=patch_sz,
                merge_size=merge_sz,
            )
            if int(km.sum()) < vi.num_visual_tokens:
                keep_mask = km

        px_mx = mx.array(vi.pixel_values)
        mx.eval(cos_a, sin_a, px_mx)
        prep_ms = (time.perf_counter() - t_prep0) * 1000.0

        t_vit0 = time.perf_counter()
        vis_feats = self.vision_model(px_mx, [vi.grid_thw])
        mx.eval(vis_feats)
        vit_ms = (time.perf_counter() - t_vit0) * 1000.0

        t_prefill0 = time.perf_counter()
        img_start = prompt_ids.index(self.image_token_id)
        n_full_vis = int(vis_feats.shape[0])
        n_dropped_vis = 0
        if keep_mask is not None and len(keep_mask) == n_full_vis:
            kept_vis_idx = np.flatnonzero(keep_mask).astype(np.int32)
            n_kept_vis = int(kept_vis_idx.size)
            if 0 < n_kept_vis < n_full_vis:
                n_dropped_vis = n_full_vis - n_kept_vis
                full_seq_len = len(prompt_ids)
                vis_feats = mx.take(vis_feats, mx.array(kept_vis_idx, dtype=mx.int32), axis=0)
                prompt_ids = (
                    prompt_ids[:img_start]
                    + ([self.image_token_id] * n_kept_vis)
                    + prompt_ids[img_start + n_full_vis :]
                )
                seq_keep = np.concatenate(
                    [
                        np.arange(0, img_start, dtype=np.int32),
                        int(img_start) + kept_vis_idx,
                        np.arange(img_start + n_full_vis, full_seq_len, dtype=np.int32),
                    ],
                    axis=0,
                )
                seq_keep_mx = mx.array(seq_keep, dtype=mx.int32)
                cos_a = mx.take(cos_a, seq_keep_mx, axis=2)
                sin_a = mx.take(sin_a, seq_keep_mx, axis=2)
                rope_delta = int(rope_delta) + int(n_dropped_vis)

        seq_len = len(prompt_ids)
        prompt_mx = mx.array([prompt_ids], dtype=mx.int32)
        text_emb = self.lm.embed_tokens(prompt_mx)
        img_end = img_start + int(vis_feats.shape[0])
        inputs_embeds = mx.concatenate(
            [
                text_emb[:, :img_start, :],
                vis_feats[None, :, :].astype(text_emb.dtype),
                text_emb[:, img_end:, :],
            ],
            axis=1,
        )

        kv_step = max(256, min(512, seq_len + tok_cap + 64))
        head_dim = int(getattr(getattr(self.lm, "cfg", None), "head_dim", 256))
        use_kv_mode = resolve_ovis_kv_mode(
            getattr(self, "kv_i8", "auto"),
            batch_size=1,
            alloc_ctx=seq_len + tok_cap,
            head_dim=head_dim,
        )
        use_gdn_mode = str(getattr(self, "gdn_state_mode", "auto"))
        cache = _make_hybrid_cache(
            self.lm,
            kv_step=kv_step,
            kv_i8=use_kv_mode,
            gdn_state_mode=use_gdn_mode,
            batch_size=1,
        )
        if self.prefill_chunk_size > 0 and seq_len > self.prefill_chunk_size:
            last_h: mx.array | None = None
            for c0 in range(0, seq_len, self.prefill_chunk_size):
                c1 = min(seq_len, c0 + self.prefill_chunk_size)
                ch = self.lm.forward_hidden(
                    inputs_embeds=inputs_embeds[:, c0:c1, :],
                    cache=cache,
                    rope_trig=(cos_a[:, :, c0:c1, :], sin_a[:, :, c0:c1, :]),
                )
                mx.eval(ch)  # guard: allow-eval
                last_h = ch[:, -1:, :]
            assert last_h is not None
        else:
            h_full = self.lm.forward_hidden(
                inputs_embeds=inputs_embeds,
                cache=cache,
                rope_trig=(cos_a, sin_a),
            )
            last_h = h_full[:, -1:, :]

        vocab_limit = len(self._new_to_old) if getattr(self, "vocab_remap", None) is None else None
        need_conf = conf_session is not None

        def _eval_logits_argmax(
            logits_mx: mx.array,
            h_keep: mx.array,
        ) -> tuple[int, tuple[float, float, float] | None, mx.array]:
            l2d = logits_mx[0]
            if vocab_limit is not None and int(l2d.shape[-1]) > vocab_limit:
                l2d = l2d[:, :vocab_limit]
            if need_conf:
                i_mx, s_mx = gpu_argmax_conf(l2d)
                mx.eval(h_keep, i_mx, s_mx)  # guard: allow-eval
                i_val = int(i_mx.tolist()[0])  # guard: allow-sync
                s_row = s_mx.tolist()[0]  # guard: allow-sync
                return i_val, (float(s_row[0]), float(s_row[1]), float(s_row[2])), l2d
            i_mx = gpu_argmax_nan(l2d)
            mx.eval(h_keep, i_mx)  # guard: allow-eval
            return int(i_mx.tolist()[0]), None, l2d  # guard: allow-sync

        logits0_mx = self.lm.project_logits(last_h)
        idx0_val, stats0_val, logits0_2d = _eval_logits_argmax(logits0_mx, last_h)
        prefill_ms = (time.perf_counter() - t_prefill0) * 1000.0

        # Prepare grammars / drafter for decode
        t_dec0 = time.perf_counter()
        stop_ids = self.tokenizer.stop_token_ids

        json_gram: GrammarHelper | None = None
        json_matcher: Any | None = None
        if schema is not None:
            schema_norm = normalize_json_schema(schema)
            json_gram = GrammarHelper(self.tokenizer.llguidance_tokenizer(), schema_norm, new_to_old=self._new_to_old)
            json_matcher = json_gram.new_matcher()

        table_gram: DynamicTableRowGrammar | None = (
            DynamicTableRowGrammar(self.tokenizer.llguidance_tokenizer())
            if (strict_md and json_matcher is None)
            else None
        )

        primary_pld_ids: list[int] = []
        extra_pld_ids: list[list[int]] = []
        primary_pld_txt, extra_pld_txts = build_ordered_pld_corpus(
            vision_res,
            native_pdf_text=native_pdf_text,
        )
        if primary_pld_txt:
            primary_pld_ids = self.tokenizer.encode(primary_pld_txt, add_special_tokens=False)
        for extra_txt in extra_pld_txts:
            eids = self.tokenizer.encode(extra_txt, add_special_tokens=False)
            if eids:
                extra_pld_ids.append(eids)

        if spec_req == "off" or float(temperature) > 0.0:
            eff_spec = "off"
        elif spec_req == "auto":
            eff_spec = "vision+self" if primary_pld_ids else "self"
        elif spec_req == "vision+self" and not primary_pld_ids:
            eff_spec = "self"
        else:
            eff_spec = spec_req

        drafter = (
            OCRPLDDrafter(
                primary_pld_ids if eff_spec == "vision+self" else (),
                extra_token_ids=extra_pld_ids if eff_spec == "vision+self" else (),
                max_ngram=4,
                min_ngram=2,
                max_k=k_draft,
            )
            if eff_spec != "off"
            else None
        )

        def _decode_tokens(tids: Sequence[int]) -> str:
            return self.tokenizer.decode(tids, skip_special_tokens=False)

        def _pick_token(
            step_logits_1d: np.ndarray,
            *,
            hidden_1d: mx.array | None,
            recent_tokens: Sequence[int],
            table_gram_override: DynamicTableRowGrammar | None = None,
            json_matcher_override: Any | None = None,
        ) -> int:
            work = step_logits_1d
            if self.vocab_remap is None and len(work) > len(self._new_to_old):
                work = work[: len(self._new_to_old)]
            argmax_idx = safe_argmax_1d(work)
            if eff_loop_breaker:
                pen = compute_loop_breaker_penalties(
                    recent_tokens,
                    window=96,
                    min_prior_hits=4,
                    base_penalty=8.0,
                    stop_token_ids=stop_ids,
                )
                if pen:
                    work, argmax_idx = apply_loop_breaker_penalties_1d(work, pen, self._old_to_new)

            active_json_matcher = json_matcher_override if json_matcher_override is not None else json_matcher
            chosen_tok: int | None = None
            if active_json_matcher is not None and json_gram is not None:
                def _lazy_full() -> np.ndarray:
                    assert hidden_1d is not None
                    fl = self.lm.project_logits(hidden_1d[None, None, :], full_vocab=True)
                    mx.eval(fl)
                    return np.asarray(fl[0, 0].astype(mx.float32))

                can_fb = bool(self.lm.has_full_vocab_fallback and hidden_1d is not None)
                tok_i, _ = json_gram.pick_valid_token(
                    active_json_matcher,
                    work,
                    argmax_idx,
                    temperature=temperature,
                    top_p=1.0,
                    fallback_to_full_vocab=can_fb,
                    full_logits_1d=_lazy_full if can_fb else None,
                )
                chosen_tok = int(tok_i)
            else:
                active_table_gram = table_gram_override if table_gram_override is not None else table_gram
                if active_table_gram is not None and active_table_gram.is_active:
                    full_mask = active_table_gram.prepare_mask(vocab_size=self.tokenizer.vocab_size)
                    if full_mask is not None:
                        active_mask = full_mask[self._new_to_old] if self.vocab_remap is not None else full_mask[: len(work)]
                        if np.any(active_mask):
                            masked = np.where(active_mask, work, -1e9)
                            chosen_tok = int(self._new_to_old[safe_argmax_1d(masked)])

            if chosen_tok is None:
                if float(temperature) <= 0.0:
                    chosen_tok = int(self._new_to_old[argmax_idx])
                else:
                    chosen_tok = int(
                        sample_unconstrained(
                            work,
                            self._new_to_old,
                            argmax_idx,
                            temperature=temperature,
                            top_p=1.0,
                        )
                    )

            if conf_session is not None and chosen_tok not in stop_ids:
                old_to_new = self._old_to_new
                c_idx = int(old_to_new[chosen_tok]) if old_to_new is not None else int(chosen_tok)
                conf_session.submit(work, c_idx)
            return chosen_tok

        def _pick_token_gpu(
            top1_idx: int,
            stats: tuple[float, float, float] | None,
            lazy_logits_fn: Callable[[], np.ndarray],
            *,
            hidden_1d: mx.array | None,
            recent_tokens: Sequence[int],
            table_gram_override: DynamicTableRowGrammar | None = None,
            json_matcher_override: Any | None = None,
        ) -> int:
            vocab_len = len(self._new_to_old)
            if float(temperature) > 0.0 or top1_idx < 0 or top1_idx >= vocab_len:
                return _pick_token(
                    lazy_logits_fn(),
                    hidden_1d=hidden_1d,
                    recent_tokens=recent_tokens,
                    table_gram_override=table_gram_override,
                    json_matcher_override=json_matcher_override,
                )

            if eff_loop_breaker:
                pen = compute_loop_breaker_penalties(
                    recent_tokens,
                    window=96,
                    min_prior_hits=4,
                    base_penalty=8.0,
                    stop_token_ids=stop_ids,
                )
                if pen:
                    return _pick_token(
                        lazy_logits_fn(),
                        hidden_1d=hidden_1d,
                        recent_tokens=recent_tokens,
                        table_gram_override=table_gram_override,
                        json_matcher_override=json_matcher_override,
                    )

            active_json_matcher = json_matcher_override if json_matcher_override is not None else json_matcher
            if active_json_matcher is not None and json_gram is not None:
                return _pick_token(
                    lazy_logits_fn(),
                    hidden_1d=hidden_1d,
                    recent_tokens=recent_tokens,
                    table_gram_override=table_gram_override,
                    json_matcher_override=json_matcher_override,
                )

            active_table_gram = table_gram_override if table_gram_override is not None else table_gram
            if active_table_gram is not None and active_table_gram.is_active:
                full_mask = active_table_gram.prepare_mask(vocab_size=self.tokenizer.vocab_size)
                if full_mask is not None:
                    cand_tok = int(self._new_to_old[top1_idx])
                    if not bool(full_mask[cand_tok]):
                        return _pick_token(
                            lazy_logits_fn(),
                            hidden_1d=hidden_1d,
                            recent_tokens=recent_tokens,
                            table_gram_override=table_gram_override,
                            json_matcher_override=json_matcher_override,
                        )

            chosen_tok = int(self._new_to_old[top1_idx])
            if conf_session is not None and chosen_tok not in stop_ids and stats is not None:
                lp, margin = stats_to_logprob_margin(stats[0], stats[1], stats[2])
                conf_session.record_direct(lp, margin)
            return chosen_tok

        gen_ids: list[int] = []
        drafted_tokens = 0
        accepted_draft_tokens = 0
        verify_passes = 0
        hit_eos = False

        # Fold any deterministic upfront JSON grammar prefix before sampling first token
        pending_tokens: list[int] = []
        if json_matcher is not None and json_gram is not None and json_gram.upfront_ff:
            for ft in json_gram.upfront_ff:
                fti = int(ft)
                if fti in stop_ids or len(gen_ids) >= tok_cap:
                    hit_eos = fti in stop_ids
                    break
                gen_ids.append(fti)
                pending_tokens.append(fti)

        if not hit_eos and len(gen_ids) < tok_cap:
            if pending_tokens:
                sh_up = self.lm.forward_hidden(
                    mx.array([pending_tokens], dtype=mx.int32),
                    cache,
                    rope_offset_delta=rope_delta,
                )
                last_h = sh_up[:, -1:, :]
                logits0_mx = self.lm.project_logits(last_h)
                idx0_val, stats0_val, logits0_2d = _eval_logits_argmax(logits0_mx, last_h)
                pending_tokens = []

            def _lazy0() -> np.ndarray:
                f32 = logits0_2d[0].astype(mx.float32)
                mx.eval(f32)  # guard: allow-eval
                return np.asarray(f32)  # guard: allow-sync

            first_tok = _pick_token_gpu(
                idx0_val,
                stats0_val,
                _lazy0,
                hidden_1d=last_h[0, 0],
                recent_tokens=gen_ids,
            )
            del logits0_mx, logits0_2d
            if first_tok in stop_ids:
                hit_eos = True
            else:
                gen_ids.append(first_tok)
                pending_tokens = [first_tok]
                if json_matcher is not None:
                    json_matcher.consume_token(first_tok)
                    if not json_matcher.is_stopped():
                        ff = [int(t) for t in json_matcher.compute_ff_tokens() if int(t) not in stop_ids]
                        if ff:
                            used = ff[: max(0, tok_cap - len(gen_ids))]
                            json_matcher.consume_tokens(used)
                            gen_ids.extend(used)
                            pending_tokens.extend(used)
                    else:
                        pending_tokens = []
                elif table_gram is not None:
                    ff = _consume_table_token_and_ff(
                        table_gram,
                        first_tok,
                        decode_fn=_decode_tokens,
                        stop_ids=stop_ids,
                    )
                    if ff:
                        used = ff[: max(0, tok_cap - len(gen_ids))]
                        gen_ids.extend(used)
                        pending_tokens.extend(used)

        can_loop_recover = bool(
            do_loop_recover
            and int(_loop_retry_depth) < 2
            and float(temperature) <= 0.0
            and tok_cap >= 32
            and (crop_box is not None or not eff_loop_breaker)
        )
        aborted_loop: tuple[int, int] | None = None
        next_loop_check = 16

        while pending_tokens and len(gen_ids) < tok_cap and not hit_eos:
            if json_matcher is not None and json_matcher.is_stopped():
                break
            if can_loop_recover and len(gen_ids) >= next_loop_check:
                aborted_loop = detect_catastrophic_token_loop(gen_ids, decode_fn=_decode_tokens)
                if aborted_loop is not None:
                    break
                next_loop_check = len(gen_ids) + 8
            rem = tok_cap - len(gen_ids)
            if drafter is not None and len(pending_tokens) == 1 and rem >= 2:
                drafts = drafter.draft(gen_ids, mode=eff_spec, max_k=min(k_draft, rem - 1))
                if json_matcher is not None and json_gram is not None and drafts:
                    drafts = _filter_json_safe_drafts(
                        json_matcher,
                        json_gram,
                        drafts,
                        vocab_size=self.tokenizer.vocab_size,
                        stop_ids=stop_ids,
                        new_to_old=self._new_to_old,
                        old_to_new=self._old_to_new,
                    )
                elif table_gram is not None and drafts:
                    drafts = _filter_table_safe_drafts(
                        table_gram,
                        drafts,
                        decode_fn=_decode_tokens,
                        vocab_size=self.tokenizer.vocab_size,
                        stop_ids=stop_ids,
                        old_to_new=self._old_to_new,
                    )
                if drafts:
                    last_tok = pending_tokens[0]
                    drafted_tokens += len(drafts)
                    verify_passes += 1
                    spec_gram = table_gram.clone() if table_gram is not None else None
                    spec_json_matcher = json_matcher.deep_copy() if json_matcher is not None else None

                    def _pick_spec_gpu(
                        step_idx: int,
                        top1_idx: int,
                        stats: tuple[float, float, float] | None,
                        lazy_fn: Callable[[], np.ndarray],
                    ) -> int:
                        tok = _pick_token_gpu(
                            top1_idx,
                            stats,
                            lazy_fn,
                            hidden_1d=None,
                            recent_tokens=gen_ids + drafts[:step_idx],
                            table_gram_override=spec_gram,
                            json_matcher_override=spec_json_matcher,
                        )
                        if step_idx < len(drafts) and tok == int(drafts[step_idx]) and tok not in stop_ids:
                            if spec_gram is not None:
                                _ = spec_gram.advance_text_and_maybe_lock(_decode_tokens([tok]), tok)
                            if spec_json_matcher is not None:
                                spec_json_matcher.consume_token(tok)
                        return tok

                    emitted, _acc_h, n_matched = verify_draft_sequence_b1(
                        self.lm,
                        cache,
                        last_tok,
                        drafts,
                        pick_token_gpu_fn=_pick_spec_gpu,
                        need_conf=need_conf,
                        vocab_limit=vocab_limit,
                        stop_token_ids=stop_ids,
                        rope_offset_delta=rope_delta,
                    )
                    accepted_draft_tokens += int(n_matched)
                    next_pending: list[int] = []
                    for idx_e, etok in enumerate(emitted):
                        last_e = idx_e == (len(emitted) - 1)
                        if etok in stop_ids:
                            hit_eos = True
                            break
                        if len(gen_ids) >= tok_cap:
                            break
                        gen_ids.append(etok)
                        next_pending = [etok]
                        if json_matcher is not None:
                            json_matcher.consume_token(etok)
                            if json_matcher.is_stopped():
                                if last_e:
                                    next_pending = []
                                    break
                                continue
                            if last_e:
                                ff = [int(t) for t in json_matcher.compute_ff_tokens() if int(t) not in stop_ids]
                                if ff:
                                    used = ff[: max(0, tok_cap - len(gen_ids))]
                                    if used:
                                        json_matcher.consume_tokens(used)
                                        gen_ids.extend(used)
                                        next_pending.extend(used)
                                        if json_matcher.is_stopped():
                                            next_pending = []
                                            break
                        elif table_gram is not None:
                            ff = _consume_table_token_and_ff(
                                table_gram,
                                etok,
                                decode_fn=_decode_tokens,
                                stop_ids=stop_ids,
                            )
                            if last_e and ff:
                                used = ff[: max(0, tok_cap - len(gen_ids))]
                                gen_ids.extend(used)
                                next_pending.extend(used)
                    if hit_eos or not next_pending or len(gen_ids) >= tok_cap:
                        break
                    pending_tokens = next_pending
                    continue

            step_arr = mx.array([pending_tokens], dtype=mx.int32)
            sh = self.lm.forward_hidden(step_arr, cache, rope_offset_delta=rope_delta)
            last_h = sh[:, -1:, :]
            step_logits_mx = self.lm.project_logits(last_h)
            s_idx_val, s_stats_val, step_logits_2d = _eval_logits_argmax(step_logits_mx, last_h)

            def _lazy_step() -> np.ndarray:
                f32 = step_logits_2d[0].astype(mx.float32)
                mx.eval(f32)  # guard: allow-eval
                return np.asarray(f32)  # guard: allow-sync

            nxt_tok = _pick_token_gpu(
                s_idx_val,
                s_stats_val,
                _lazy_step,
                hidden_1d=last_h[0, 0],
                recent_tokens=gen_ids,
            )
            del step_logits_mx, step_logits_2d
            if nxt_tok in stop_ids:
                hit_eos = True
                break
            gen_ids.append(nxt_tok)
            pending_tokens = [nxt_tok]
            if json_matcher is not None:
                json_matcher.consume_token(nxt_tok)
                if json_matcher.is_stopped():
                    pending_tokens = []
                    break
                ff = [int(t) for t in json_matcher.compute_ff_tokens() if int(t) not in stop_ids]
                if ff:
                    used = ff[: max(0, tok_cap - len(gen_ids))]
                    json_matcher.consume_tokens(used)
                    gen_ids.extend(used)
                    pending_tokens.extend(used)
            elif table_gram is not None:
                ff = _consume_table_token_and_ff(
                    table_gram,
                    nxt_tok,
                    decode_fn=_decode_tokens,
                    stop_ids=stop_ids,
                )
                if ff:
                    used = ff[: max(0, tok_cap - len(gen_ids))]
                    gen_ids.extend(used)
                    pending_tokens.extend(used)

        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        peak_gb = float(mx.get_peak_memory() / 1e9)
        hit_length = (not hit_eos) and not (json_matcher is not None and json_matcher.is_stopped())
        tail_rep4 = max(
            _token_tail_repeat_4gram_rate(gen_ids, tail_len=256),
            _token_tail_repeat_4gram_rate(gen_ids, tail_len=128),
        )
        if (
            aborted_loop is None
            and can_loop_recover
            and hit_length
            and len(gen_ids) >= tok_cap
            and tail_rep4 >= 0.35
        ):
            aborted_loop = (0, len(gen_ids))

        if aborted_loop is not None and can_loop_recover:
            if conf_session is not None:
                conf_session.finish()
            aborted_tokens = len(gen_ids)
            del cache
            mx.clear_cache()
            if crop_box is not None and int(_loop_retry_depth) == 0:
                retry_action = "retry_uncropped"
                retry_crop = False
                retry_lb = False
            else:
                retry_action = "retry_loop_breaker"
                retry_crop = False
                retry_lb = True
            retry_res = self.ocr_image(
                oriented_img,
                page_index=page_index,
                prompt=prompt,
                schema=schema,
                output_format=fmt,
                table_mode=tmode,
                strict_markdown_grammar=strict_md,
                zoom_recover=do_zoom,
                crop_margins=retry_crop,
                auto_orient=False,
                adaptive_resolution=adaptive_resolution,
                deskew=False,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                drop_white_patches=eff_drop_white,
                native_pdf_text=native_pdf_text,
                speculative=spec_req,
                draft_k=k_draft,
                max_tokens=tok_cap,
                temperature=temperature,
                pre_analysis=vision_res,
                auto_loop_recover=do_loop_recover,
                collect_logprob_confidence=collect_logprob_confidence,
                _force_loop_breaker=retry_lb,
                _loop_retry_depth=int(_loop_retry_depth) + 1,
            )
            rt = retry_res.timings
            merged_dec_ms = decode_ms + float(rt.decode_ms)
            merged_gen = int(rt.generated_tokens)
            merged_timings = replace(
                rt,
                preprocess_ms=round(prep_ms + float(rt.preprocess_ms), 3),
                apple_vision_ms=round(vision_ms, 3),
                vit_ms=round(vit_ms + float(rt.vit_ms), 3),
                prefill_ms=round(prefill_ms + float(rt.prefill_ms), 3),
                decode_ms=round(merged_dec_ms, 3),
                total_ms=round((time.perf_counter() - t_total0) * 1000.0, 3),
                decode_tps=round((merged_gen / (merged_dec_ms / 1000.0)) if merged_dec_ms > 0.0 else 0.0, 2),
                peak_memory_gb=round(max(peak_gb, float(rt.peak_memory_gb)), 4),
            )
            merged_meta = dict(retry_res.metadata)
            merged_meta["loop_recovery"] = {
                "action": retry_action,
                "aborted_at_tokens": aborted_tokens,
                "detected_period": int(aborted_loop[0]),
                "detected_repeats": int(aborted_loop[1]),
                "prior_crop_box": list(crop_box) if crop_box is not None else None,
            }
            return replace(
                retry_res,
                rotation_applied_cw=rot_cw,
                timings=merged_timings,
                metadata=merged_meta,
            )

        total_gen = len(gen_ids) + (1 if hit_eos else 0)
        decode_tps = (total_gen / (decode_ms / 1000.0)) if decode_ms > 0.0 else 0.0

        t_post0 = time.perf_counter()
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        validation = None
        if fmt == "markdown" and schema is None:
            rendered = canonicalize_markdown(raw_text, table_mode=tmode)
            validation = validate_markdown(rendered, allow_html_tables=(tmode == "html"))
        else:
            rendered = raw_text
        conf_summary = conf_session.finish() if conf_session is not None else None
        post_ms = (time.perf_counter() - t_post0) * 1000.0

        zoom_meta: dict[str, Any] | None = None
        if do_zoom and vision_res is not None and vision_res.lines:
            rendered, raw_text, z_res, zoom_meta = self._run_zoom_recovery(
                oriented_img,
                vision_res,
                rot_cw=rot_cw,
                rendered=rendered,
                raw_text=raw_text,
                page_index=page_index,
                prompt=prompt,
                output_format=fmt,
                table_mode=tmode,
                strict_markdown_grammar=strict_md,
                speculative=spec_req,
                draft_k=k_draft,
                max_tokens=tok_cap,
                temperature=temperature,
            )
            if z_res is not None:
                prep_ms += float(z_res.timings.preprocess_ms)
                vit_ms += float(z_res.timings.vit_ms)
                prefill_ms += float(z_res.timings.prefill_ms)
                decode_ms += float(z_res.timings.decode_ms)
                post_ms += float(z_res.timings.postprocess_ms)
                total_gen += int(z_res.timings.generated_tokens)
                drafted_tokens += int(z_res.timings.drafted_tokens)
                accepted_draft_tokens += int(z_res.timings.accepted_draft_tokens)
                verify_passes += int(z_res.timings.verify_passes)
                peak_gb = max(peak_gb, float(z_res.timings.peak_memory_gb))
                if decode_ms > 0.0:
                    decode_tps = total_gen / (decode_ms / 1000.0)
                if fmt == "markdown" and schema is None and zoom_meta and zoom_meta.get("appended"):
                    validation = validate_markdown(rendered, allow_html_tables=(tmode == "html"))

        total_ms = (time.perf_counter() - t_total0) * 1000.0
        metadata: dict[str, Any] = {
            "grid_thw": list(vi.grid_thw),
            "rope_delta": int(rope_delta),
            "speculative_mode": eff_spec,
            "vocab_size": int(self.lm.vocab_n),
            "kv_mode": use_kv_mode,
            "dropped_visual_tokens": int(n_dropped_vis),
        }
        if eff_max_pixels < 16_777_216:
            metadata["adaptive_max_pixels"] = int(eff_max_pixels)
        if deskew_meta is not None:
            metadata["deskew"] = deskew_meta
        if schema is not None and vision_res is not None:
            line_rot = (int(rot_cw) - int(vision_res.rotation_applied_cw)) % 360
            metadata["field_boxes"] = align_json_values_to_lines(
                raw_text,
                vision_res.lines,
                image_size=oriented_img.size,
                rotation_cw=line_rot,
            )
        if zoom_meta is not None:
            metadata["zoom_recovery"] = zoom_meta
        if conf_summary is not None:
            metadata["logprob_confidence"] = conf_summary.to_dict()

        finish_reason = "stop" if (hit_eos or (json_matcher is not None and json_matcher.is_stopped())) else "length"
        return OCRResult(
            text=rendered,
            raw_text=raw_text,
            backend_used=f"ovis:{self.model_dir.name}",
            page_index=int(page_index),
            output_format=fmt,
            table_mode=tmode,
            rotation_applied_cw=rot_cw,
            cropped_box_px=crop_box,
            lines=vision_res.lines if vision_res is not None else (),
            analysis=vision_res.analysis if vision_res is not None else None,
            markdown_validation=validation,
            timings=OCRTimings(
                preprocess_ms=round(prep_ms, 3),
                apple_vision_ms=round(vision_ms, 3),
                vit_ms=round(vit_ms, 3),
                prefill_ms=round(prefill_ms, 3),
                decode_ms=round(decode_ms, 3),
                postprocess_ms=round(post_ms, 3),
                total_ms=round(total_ms, 3),
                decode_tps=round(decode_tps, 2),
                prompt_tokens=int(seq_len),
                visual_tokens=int(vis_feats.shape[0]),
                generated_tokens=int(total_gen),
                drafted_tokens=int(drafted_tokens),
                accepted_draft_tokens=int(accepted_draft_tokens),
                verify_passes=int(verify_passes),
                peak_memory_gb=round(peak_gb, 4),
            ),
            finish_reason=finish_reason,
            metadata=metadata,
        )

    def _run_zoom_recovery(
        self,
        oriented_img: Image.Image,
        vision_res: OCRResult,
        *,
        rot_cw: int,
        rendered: str,
        raw_text: str,
        page_index: int,
        prompt: str | None,
        output_format: OutputFormat,
        table_mode: MarkdownTableMode,
        strict_markdown_grammar: bool,
        speculative: SpeculativeMode,
        draft_k: int,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str, OCRResult | None, dict[str, Any] | None]:
        """Run targeted 2x zoom crop pass(es) when ANE Vision detects multi-digit numbers absent from VLM text."""
        vision_text = vision_res.text or vision_res.raw_text
        missing_nums = find_filtered_missing_numbers(vision_text, rendered)
        if not missing_nums:
            return rendered, raw_text, None, None
        line_rot = (int(rot_cw) - int(vision_res.rotation_applied_cw)) % 360
        crop_boxes = compute_missing_numeric_crop_boxes_px(
            vision_res.lines,
            missing_nums,
            oriented_img.size,
            rotation_cw=line_rot,
            max_crops=2,
        )
        if not crop_boxes:
            return rendered, raw_text, None, None

        z_spec: SpeculativeMode = "off" if speculative == "off" else "self"
        cur_raw = raw_text
        cur_rendered = rendered
        remaining_missing = list(missing_nums)
        all_recovered: list[str] = []
        appended = False
        z_results: list[OCRResult] = []
        used_boxes: list[list[int]] = []

        for zoom_box in crop_boxes:
            crop = oriented_img.crop(zoom_box)
            if min(crop.size) < 400:
                crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)
            z_res = self.ocr_image(
                crop,
                page_index=page_index,
                prompt=prompt,
                schema=None,
                output_format=output_format,
                table_mode=table_mode,
                strict_markdown_grammar=strict_markdown_grammar,
                zoom_recover=False,
                crop_margins=False,
                auto_orient=False,
                adaptive_resolution=False,
                deskew=False,
                min_pixels=REGION_CROP_MIN_PIXELS,
                speculative=z_spec,
                draft_k=draft_k,
                max_tokens=min(max_tokens, 2048),
                temperature=temperature,
            )
            z_results.append(z_res)
            used_boxes.append(list(zoom_box))
            crop_recovered = find_recovered_numbers(remaining_missing, z_res.text)
            if crop_recovered and z_res.text.strip():
                appended = True
                cur_raw = f"{cur_raw.rstrip()}\n\n{z_res.raw_text.strip()}".strip() if z_res.raw_text.strip() else cur_raw
                cur_rendered = f"{cur_rendered.rstrip()}\n\n{z_res.text.strip()}".strip()
                if output_format == "markdown":
                    cur_rendered = canonicalize_markdown(cur_rendered, table_mode=table_mode)
                for rec in crop_recovered:
                    if rec not in all_recovered:
                        all_recovered.append(rec)
                rec_set = set(crop_recovered)
                remaining_missing = [m for m in remaining_missing if m not in rec_set]
                if not remaining_missing:
                    break

        if not z_results:
            return rendered, raw_text, None, None
        if len(z_results) == 1:
            agg_z_res = z_results[0]
        else:
            sum_dec_ms = sum(float(r.timings.decode_ms) for r in z_results)
            sum_gen = sum(int(r.timings.generated_tokens) for r in z_results)
            agg_timings = OCRTimings(
                preprocess_ms=sum(float(r.timings.preprocess_ms) for r in z_results),
                apple_vision_ms=sum(float(r.timings.apple_vision_ms) for r in z_results),
                vit_ms=sum(float(r.timings.vit_ms) for r in z_results),
                prefill_ms=sum(float(r.timings.prefill_ms) for r in z_results),
                decode_ms=sum_dec_ms,
                postprocess_ms=sum(float(r.timings.postprocess_ms) for r in z_results),
                total_ms=sum(float(r.timings.total_ms) for r in z_results),
                decode_tps=(sum_gen / (sum_dec_ms / 1000.0)) if sum_dec_ms > 0.0 else 0.0,
                prompt_tokens=sum(int(r.timings.prompt_tokens) for r in z_results),
                visual_tokens=sum(int(r.timings.visual_tokens) for r in z_results),
                generated_tokens=sum_gen,
                drafted_tokens=sum(int(r.timings.drafted_tokens) for r in z_results),
                accepted_draft_tokens=sum(int(r.timings.accepted_draft_tokens) for r in z_results),
                verify_passes=sum(int(r.timings.verify_passes) for r in z_results),
                peak_memory_gb=max(float(r.timings.peak_memory_gb) for r in z_results),
            )
            agg_z_res = replace(z_results[-1], timings=agg_timings)

        meta = {
            "triggered": True,
            "appended": appended,
            "missing_numbers": missing_nums,
            "recovered_numbers": all_recovered,
            "crop_box_px": used_boxes[0],
            "crop_boxes_px": used_boxes,
            "zoom_ms": round(float(agg_z_res.timings.total_ms), 3),
        }
        return cur_rendered, cur_raw, agg_z_res, meta

    def ocr_images(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        page_indices: Sequence[int] | None = None,
        prompt: str | None = None,
        prompts: Sequence[str | None] | None = None,
        schema: str | Mapping[str, Any] | None = None,
        schemas: Sequence[str | Mapping[str, Any] | None] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        strict_markdown_grammar: bool | None = None,
        zoom_recover: bool | None = None,
        crop_margins: bool | None = None,
        auto_orient: bool | None = None,
        adaptive_resolution: bool | None = None,
        deskew: bool | str | None = None,
        min_pixels: int | Sequence[int | None] | None = None,
        max_pixels: int | Sequence[int | None] | None = None,
        drop_white_patches: bool | None = None,
        native_pdf_texts: Sequence[str | None] | None = None,
        speculative: SpeculativeMode | None = None,
        draft_k: int | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        pre_analyses: Sequence[OCRResult | None] | None = None,
        batch_size: int | None = 16,
        compact_active: bool = True,
        max_pad_ratio: float | None = 1.8,
        max_wave_tokens: int | None = None,
        auto_loop_recover: bool | None = None,
        collect_logprob_confidence: bool | str | None = None,
    ) -> list[OCRResult]:
        """Process multiple images: true `B >= 2` batched VLM waves or double-buffered ANE+PLD when `B=1`."""
        if not sources:
            return []
        n_sources = len(sources)
        indices = list(range(n_sources)) if page_indices is None else [int(i) for i in page_indices]
        if len(indices) != n_sources:
            raise ValueError(f"Expected {n_sources} page_indices, got {len(indices)}")

        pre_seq: list[OCRResult | None] = (
            list(pre_analyses) if pre_analyses is not None else [None] * n_sources
        )
        if len(pre_seq) != n_sources:
            raise ValueError(f"Expected {n_sources} pre_analyses, got {len(pre_seq)}")

        prompt_seq: list[str | None] = (
            list(prompts) if prompts is not None else [prompt] * n_sources
        )
        if len(prompt_seq) != n_sources:
            raise ValueError(f"Expected {n_sources} prompts, got {len(prompt_seq)}")

        schema_seq: list[str | Mapping[str, Any] | None] = (
            list(schemas) if schemas is not None else [schema] * n_sources
        )
        if len(schema_seq) != n_sources:
            raise ValueError(f"Expected {n_sources} schemas, got {len(schema_seq)}")

        native_seq: list[str | None] = (
            list(native_pdf_texts) if native_pdf_texts is not None else [None] * n_sources
        )
        if len(native_seq) != n_sources:
            raise ValueError(f"Expected {n_sources} native_pdf_texts, got {len(native_seq)}")

        min_px_scalar, min_px_per_item = _normalize_per_item_int(min_pixels, n_sources, "min_pixels")
        max_px_scalar, max_px_per_item = _normalize_per_item_int(max_pixels, n_sources, "max_pixels")

        fmt: OutputFormat = output_format or self.output_format
        tmode: MarkdownTableMode = table_mode or self.table_mode
        strict_md = self.strict_markdown_grammar if strict_markdown_grammar is None else bool(strict_markdown_grammar)
        do_zoom_req = self.zoom_recover if zoom_recover is None else bool(zoom_recover)
        do_crop = self.crop_margins if crop_margins is None else bool(crop_margins)
        do_orient = self.auto_orient if auto_orient is None else bool(auto_orient)
        do_adaptive = bool(getattr(self, "adaptive_resolution", True)) if adaptive_resolution is None else bool(adaptive_resolution)
        eff_deskew = normalize_deskew_mode(getattr(self, "deskew", "off") if deskew is None else deskew)
        eff_drop_white = (
            bool(getattr(self, "drop_white_patches", False))
            if drop_white_patches is None
            else bool(drop_white_patches)
        )
        spec_req: SpeculativeMode = self.speculative if speculative is None else speculative
        k_draft = self.draft_k if draft_k is None else max(1, int(draft_k))
        tok_cap = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        do_loop_recover = bool(getattr(self, "auto_loop_recover", True)) if auto_loop_recover is None else bool(auto_loop_recover)
        eff_conf_mode = _normalize_confidence_mode(
            getattr(self, "collect_logprob_confidence", "auto")
            if collect_logprob_confidence is None
            else collect_logprob_confidence
        )
        eff_loop_breaker = bool(getattr(self, "loop_breaker", False))
        eff_bs = 1 if batch_size is None else max(1, int(batch_size))
        eff_min_pixels = 65_536 if min_px_scalar is None else max(1024, int(min_px_scalar))
        eff_max_pixels = 16_777_216 if max_px_scalar is None else max(eff_min_pixels, int(max_px_scalar))

        def _item_min_px(pos_i: int) -> int | None:
            if min_px_per_item is not None:
                return min_px_per_item[pos_i]
            return min_px_scalar

        def _item_max_px(pos_i: int) -> int | None:
            if max_px_per_item is not None:
                return max_px_per_item[pos_i]
            return max_px_scalar

        # Single image or explicit B=1 path keeps double-buffered ANE Vision + B=1 PLD fast path
        if n_sources == 1 or eff_bs <= 1:
            need_ane_prepass = do_orient or (do_zoom_req and any(sc is None for sc in schema_seq)) or spec_req in ("auto", "vision+self")
            if n_sources == 1 or eff_deskew == "on" or not need_ane_prepass or all(p is not None for p in pre_seq):
                return [
                    self.ocr_image(
                        src,
                        page_index=idx,
                        prompt=pr_i,
                        schema=sc_i,
                        output_format=fmt,
                        table_mode=tmode,
                        strict_markdown_grammar=strict_md,
                        zoom_recover=(do_zoom_req and sc_i is None),
                        crop_margins=do_crop,
                        auto_orient=do_orient,
                        adaptive_resolution=do_adaptive,
                        deskew=eff_deskew,
                        min_pixels=_item_min_px(pos_i),
                        max_pixels=_item_max_px(pos_i),
                        drop_white_patches=eff_drop_white,
                        native_pdf_text=nat_i,
                        speculative=spec_req,
                        draft_k=draft_k,
                        max_tokens=tok_cap,
                        temperature=temperature,
                        pre_analysis=pre,
                        auto_loop_recover=do_loop_recover,
                        collect_logprob_confidence=collect_logprob_confidence,
                    )
                    for pos_i, (src, idx, pre, pr_i, sc_i, nat_i) in enumerate(
                        zip(sources, indices, pre_seq, prompt_seq, schema_seq, native_seq)
                    )
                ]

            def _run_ane(src: Any, idx: int) -> OCRResult | None:
                try:
                    return self._get_vision_prepass().recognize_image(src, page_index=idx, output_format="raw")
                except Exception:
                    return None

            results_b1: list[OCRResult] = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                cur_pre = pre_seq[0] if pre_seq[0] is not None else _run_ane(sources[0], indices[0])
                for pos in range(n_sources):
                    nxt_fut: Future[OCRResult | None] | None = None
                    if pos + 1 < n_sources and pre_seq[pos + 1] is None:
                        nxt_fut = pool.submit(_run_ane, sources[pos + 1], indices[pos + 1])
                    sc_pos = schema_seq[pos]
                    results_b1.append(
                        self.ocr_image(
                            sources[pos],
                            page_index=indices[pos],
                            prompt=prompt_seq[pos],
                            schema=sc_pos,
                            output_format=fmt,
                            table_mode=tmode,
                            strict_markdown_grammar=strict_md,
                            zoom_recover=(do_zoom_req and sc_pos is None),
                            crop_margins=do_crop,
                            auto_orient=do_orient,
                            adaptive_resolution=do_adaptive,
                            deskew=False,
                            min_pixels=_item_min_px(pos),
                            max_pixels=_item_max_px(pos),
                            drop_white_patches=eff_drop_white,
                            native_pdf_text=native_seq[pos],
                            speculative=spec_req,
                            draft_k=draft_k,
                            max_tokens=tok_cap,
                            temperature=temperature,
                            pre_analysis=cur_pre,
                            auto_loop_recover=do_loop_recover,
                            collect_logprob_confidence=collect_logprob_confidence,
                        )
                    )
                    if nxt_fut is not None:
                        cur_pre = nxt_fut.result()
                    elif pos + 1 < n_sources:
                        cur_pre = pre_seq[pos + 1]
            return results_b1

        # Batched `B >= 2` path: parallel ANE prepass + shared `batched_multimodal_generate` + per-page recovery
        mx.clear_cache()
        n_items = len(sources)

        t_load0 = time.perf_counter()
        raw_imgs = [load_image(src) for src in sources]
        deskew_metas: list[dict[str, Any] | None] = [None] * n_items
        if eff_deskew != "off":
            for i in range(n_items):
                try:
                    warped_i, d_meta = deskew_document_image(raw_imgs[i], mode=eff_deskew)
                    if d_meta.get("applied"):
                        deskew_metas[i] = d_meta
                        raw_imgs[i] = warped_i
                        pre_seq[i] = None
                    elif eff_deskew == "on":
                        deskew_metas[i] = d_meta
                except Exception:
                    if eff_deskew == "on":
                        deskew_metas[i] = {"applied": False, "reason": "deskew_error"}
        load_ms_each = ((time.perf_counter() - t_load0) * 1000.0) / max(1, n_items)

        rolling_eligible = bool(
            float(temperature) <= 0.0
            and not strict_md
            and all(sc is None for sc in schema_seq)
            and not eff_loop_breaker
            and eff_conf_mode != "on"
            and not eff_drop_white
        )
        use_batch_pld = bool(rolling_eligible and spec_req != "off")

        need_batch_ane = (
            do_orient
            or (do_zoom_req and any(sc is None for sc in schema_seq))
            or any(sc is not None for sc in schema_seq)
            or (use_batch_pld and spec_req in ("auto", "vision+self"))
        )
        if need_batch_ane:
            missing_pos = [i for i, pre in enumerate(pre_seq) if pre is None]
            if missing_pos:
                par_res: list[OCRResult] | None = None
                try:
                    from .pipeline import run_vision_batch_parallel

                    par_res = run_vision_batch_parallel(
                        self._get_vision_prepass(),
                        [raw_imgs[p] for p in missing_pos],
                        [indices[p] for p in missing_pos],
                        output_format="raw",
                        workers=4,
                    )
                except Exception:
                    par_res = None
                if par_res is not None:
                    for pos_i, vis_r in zip(missing_pos, par_res):
                        pre_seq[pos_i] = vis_r
                else:
                    vp = self._get_vision_prepass()
                    for pos_i in missing_pos:
                        try:
                            pre_seq[pos_i] = vp.recognize_image(
                                raw_imgs[pos_i],
                                page_index=indices[pos_i],
                                output_format="raw",
                            )
                        except Exception:
                            pre_seq[pos_i] = None

        oriented_imgs: list[Image.Image] = []
        proc_imgs: list[Image.Image] = []
        rot_cws: list[int] = []
        crop_boxes: list[tuple[int, int, int, int] | None] = []
        adaptive_caps: list[int | None] = []
        vision_ms_arr: list[float] = []
        prep_ms_arr: list[float] = []

        for i in range(n_items):
            t_p0 = time.perf_counter()
            im = raw_imgs[i]
            pre_i = pre_seq[i]
            vision_ms_arr.append(float(pre_i.timings.apple_vision_ms) if pre_i is not None else 0.0)
            rot_cw = resolve_vision_rotation_cw(pre_i, auto_orient=do_orient)
            if rot_cw != 0:
                im = rotate_image_cw(im, rot_cw)
            oriented_imgs.append(im)
            crop_box: tuple[int, int, int, int] | None = None
            if do_crop:
                im, crop_box = crop_to_ink(im, pad_px=self.crop_pad_px)
            adap_cap: int | None = None
            if max_px_scalar is None and max_px_per_item is None and do_adaptive and pre_i is not None and pre_i.analysis is not None:
                apx = compute_adaptive_max_pixels(
                    pre_i.analysis,
                    oriented_imgs[i].size,
                    cropped_size=im.size,
                )
                if apx < 16_777_216:
                    adap_cap = apx
                    if im.width * im.height > apx:
                        eff_item_min_px = _item_min_px(i)
                        eff_adapt_min_px = _resolve_ovis_min_pixels(
                            im.size,
                            min_pixels=eff_item_min_px,
                            max_pixels=_item_max_px(i),
                            output_format=fmt,
                            schema=schema_seq[i],
                        )
                        rh, rw = smart_resize_qwen(
                            im.height,
                            im.width,
                            factor=32,
                            min_pixels=eff_adapt_min_px,
                            max_pixels=apx,
                        )
                        if (rw, rh) != im.size:
                            im = im.resize((rw, rh), Image.Resampling.BICUBIC)
            proc_imgs.append(im)
            rot_cws.append(rot_cw)
            crop_boxes.append(crop_box)
            adaptive_caps.append(adap_cap)
            prep_ms_arr.append(load_ms_each + (time.perf_counter() - t_p0) * 1000.0)

        default_instruction = (
            DEFAULT_GFM_STRICT_PROMPT if (strict_md and tmode == "gfm") else DEFAULT_OVIS_PROMPT
        )
        pre_ids = self.tokenizer.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
        suf_cache: dict[str, list[int]] = {}
        suf_ids_per_item: list[list[int]] = []
        for pr_i in prompt_seq:
            inst_i = str(pr_i) if pr_i is not None else default_instruction
            cached_suf = suf_cache.get(inst_i)
            if cached_suf is None:
                cached_suf = self.tokenizer.encode(
                    f"<|vision_end|>\n{inst_i}<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    add_special_tokens=False,
                )
                suf_cache[inst_i] = cached_suf
            suf_ids_per_item.append(cached_suf)

        def _prompt_builder(item_idx: int, vis_counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
            nv = int(vis_counts[0]) if vis_counts else 0
            s0 = len(pre_ids)
            suf_i = suf_ids_per_item[item_idx]
            return pre_ids + ([self.image_token_id] * nv) + suf_i, ([(s0, s0 + nv)] if nv > 0 else [])

        gram_cache: dict[str, GrammarHelper] = {}
        json_grams: list[GrammarHelper | None] = []
        for sc_i in schema_seq:
            if sc_i is None:
                json_grams.append(None)
                continue
            schema_norm = normalize_json_schema(sc_i)
            cache_k = (
                schema_norm
                if isinstance(schema_norm, str)
                else json.dumps(schema_norm, sort_keys=True, separators=(",", ":"))
            )
            gh = gram_cache.get(cache_k)
            if gh is None:
                gh = GrammarHelper(self.tokenizer.llguidance_tokenizer(), schema_norm, new_to_old=self._new_to_old)
                gram_cache[cache_k] = gh
            json_grams.append(gh)

        table_grammars = (
            [
                DynamicTableRowGrammar(self.tokenizer.llguidance_tokenizer()) if g_i is None else None
                for g_i in json_grams
            ]
            if (strict_md and any(g_i is None for g_i in json_grams))
            else None
        )
        mrope_sec = tuple(
            int(x)
            for x in self.raw_config.get("text_config", {}).get("rope_parameters", {}).get("mrope_section", (11, 11, 10))
        )

        kv_req = getattr(self, "kv_i8", "auto")
        head_dim = int(getattr(getattr(self.lm, "cfg", None), "head_dim", 256))
        use_gdn_mode = str(getattr(self, "gdn_state_mode", "auto"))

        def _ovis_kv_i8(b_slots: int, alloc_ctx: int) -> KVMode:
            return resolve_ovis_kv_mode(
                kv_req,
                batch_size=b_slots,
                alloc_ctx=alloc_ctx,
                head_dim=head_dim,
            )

        eff_min_px_per_item = min_px_per_item
        if min_px_scalar is None and min_px_per_item is None:
            auto_mins = [
                _resolve_ovis_min_pixels(
                    proc_imgs[i].size,
                    min_pixels=None,
                    max_pixels=_item_max_px(i),
                    output_format=fmt,
                    schema=schema_seq[i],
                )
                for i in range(n_items)
            ]
            if any(m != eff_min_pixels for m in auto_mins):
                eff_min_px_per_item = auto_mins

        eff_rolling_max = 8
        if (
            rolling_eligible
            and int(eff_bs) >= 16
            and n_items >= 16
            and int(tok_cap) <= 2048
            and host_phys_memory_gb() > 28.0
        ):
            est_vis = [
                (
                    lambda hw: (hw[0] // 32) * (hw[1] // 32)
                )(
                    smart_resize_qwen(
                        proc_imgs[i].height,
                        proc_imgs[i].width,
                        factor=32,
                        min_pixels=(
                            eff_min_px_per_item[i]
                            if eff_min_px_per_item is not None and eff_min_px_per_item[i] is not None
                            else eff_min_pixels
                        ),
                        max_pixels=(
                            max_px_per_item[i]
                            if max_px_per_item is not None and max_px_per_item[i] is not None
                            else eff_max_pixels
                        ),
                    )
                )
                for i in range(n_items)
            ]
            if max(est_vis, default=0) <= 512:
                # MS4b: uniform compact tables run faster at B=16 AR (K=0) than B=8 PLD.
                eff_rolling_max = min(int(eff_bs), 16)
                use_batch_pld = False

        has_any_native = any(bool(n and str(n).strip()) for n in native_seq)
        batch_k_draft = max(k_draft, 6) if (draft_k is None and has_any_native) else k_draft
        batch_drafters: list[OCRPLDDrafter | None] | None = None
        batch_spec_modes: list[str] | None = None
        if use_batch_pld:
            batch_drafters = []
            batch_spec_modes = []
            for i in range(n_items):
                primary_ids: list[int] = []
                extra_ids: list[list[int]] = []
                if spec_req in ("auto", "vision+self"):
                    primary_txt, extra_txts = build_ordered_pld_corpus(
                        pre_seq[i],
                        native_pdf_text=native_seq[i],
                    )
                    if primary_txt:
                        primary_ids = self.tokenizer.encode(primary_txt, add_special_tokens=False)
                    for extra_t in extra_txts:
                        eids = self.tokenizer.encode(extra_t, add_special_tokens=False)
                        if eids:
                            extra_ids.append(eids)
                eff_sp = "vision+self" if primary_ids else "self"
                batch_spec_modes.append(eff_sp)
                batch_drafters.append(
                    OCRPLDDrafter(
                        primary_ids if eff_sp == "vision+self" else (),
                        extra_token_ids=extra_ids if eff_sp == "vision+self" else (),
                        max_ngram=4,
                        min_ngram=2,
                        max_k=batch_k_draft,
                    )
                )

        mm_results, mm_stats = batched_multimodal_generate(
            lm=self.lm,
            vision_model=self.vision_model,
            tokenizer=self.tokenizer,
            images_per_item=[[im] for im in proc_imgs],
            prompt_builder=_prompt_builder,
            image_token_id=self.image_token_id,
            new_to_old=self._new_to_old,
            old_to_new=self._old_to_new,
            grammars=json_grams,
            table_grammars=table_grammars,
            drafters=batch_drafters,
            spec_modes=batch_spec_modes,
            draft_k=batch_k_draft,
            rolling=("auto" if rolling_eligible else False),
            rolling_max_batch_size=eff_rolling_max,
            max_tokens=tok_cap,
            temperature=temperature,
            repeat_penalty=eff_loop_breaker,
            collect_telemetry=(eff_conf_mode == "on"),
            batch_size=eff_bs,
            kv_step=512,
            kv_i8_resolver=_ovis_kv_i8,
            compact_active=compact_active,
            max_pad_ratio=max_pad_ratio,
            max_wave_tokens=max_wave_tokens,
            min_pixels=eff_min_pixels,
            max_pixels=eff_max_pixels,
            drop_white_patches=eff_drop_white,
            min_pixels_per_item=eff_min_px_per_item,
            max_pixels_per_item=max_px_per_item,
            gdn_state_mode=use_gdn_mode,
            stop_ids=self.tokenizer.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            prefill_chunk_size=min(512, int(self.prefill_chunk_size)) if self.prefill_chunk_size > 0 else 512,
            mrope_section=(int(mrope_sec[0]), int(mrope_sec[1]), int(mrope_sec[2])),
        )
        batch_peak_gb = float(mx.get_peak_memory() / 1e9)
        vocab_n = int(getattr(self.lm, "vocab_n", len(self._new_to_old)))
        decode_fn = lambda tids: self.tokenizer.decode(tids, skip_special_tokens=True)

        results: list[OCRResult] = []
        for i, mm_res in enumerate(mm_results):
            gen_ids = mm_res.generated_ids
            can_loop_recover = bool(
                do_loop_recover
                and tok_cap >= 32
                and float(temperature) <= 0.0
                and (not eff_loop_breaker or crop_boxes[i] is not None or rot_cws[i] != 0)
            )
            aborted_loop = detect_catastrophic_token_loop(gen_ids, decode_fn=decode_fn) if can_loop_recover else None
            hit_length = bool(mm_res.hit_max_tokens) or (
                (not mm_res.hit_eos) and len(gen_ids) >= tok_cap and schema_seq[i] is None
            )
            if aborted_loop is None and can_loop_recover and hit_length:
                tail_rep4 = max(
                    _token_tail_repeat_4gram_rate(gen_ids, tail_len=256),
                    _token_tail_repeat_4gram_rate(gen_ids, tail_len=128),
                )
                if tail_rep4 >= 0.35:
                    aborted_loop = (0, len(gen_ids))

            vision_res = pre_seq[i]
            if aborted_loop is not None and can_loop_recover:
                retry_rot_reset = rot_cws[i] != 0
                retry_src = raw_imgs[i] if retry_rot_reset else oriented_imgs[i]
                retry_crop = False
                retry_lb = (crop_boxes[i] is None) and (not retry_rot_reset)
                if retry_rot_reset:
                    retry_action = "retry_unrotated"
                elif crop_boxes[i] is not None:
                    retry_action = "retry_uncropped"
                else:
                    retry_action = "retry_loop_breaker"
                retry_res = self.ocr_image(
                    retry_src,
                    page_index=indices[i],
                    prompt=prompt_seq[i],
                    schema=schema_seq[i],
                    output_format=fmt,
                    table_mode=tmode,
                    strict_markdown_grammar=strict_md,
                    zoom_recover=(do_zoom_req and schema_seq[i] is None),
                    crop_margins=retry_crop,
                    auto_orient=False,
                    adaptive_resolution=do_adaptive,
                    deskew=False,
                    min_pixels=_item_min_px(i),
                    max_pixels=_item_max_px(i),
                    drop_white_patches=eff_drop_white,
                    native_pdf_text=native_seq[i],
                    speculative=spec_req,
                    draft_k=draft_k,
                    max_tokens=tok_cap,
                    temperature=temperature,
                    pre_analysis=(None if retry_rot_reset else vision_res),
                    auto_loop_recover=True,
                    collect_logprob_confidence=collect_logprob_confidence,
                    _force_loop_breaker=retry_lb,
                    _loop_retry_depth=1,
                )
                rt = retry_res.timings
                merged_dec = mm_res.decode_ms + float(rt.decode_ms)
                merged_gen = int(rt.generated_tokens)
                merged_timings = replace(
                    rt,
                    preprocess_ms=round(prep_ms_arr[i] + float(rt.preprocess_ms), 3),
                    apple_vision_ms=round(vision_ms_arr[i], 3),
                    vit_ms=round(mm_res.vit_ms + float(rt.vit_ms), 3),
                    prefill_ms=round(mm_res.prefill_ms + float(rt.prefill_ms), 3),
                    decode_ms=round(merged_dec, 3),
                    decode_tps=round((merged_gen / (merged_dec / 1000.0)) if merged_dec > 0.0 else 0.0, 2),
                    peak_memory_gb=round(max(batch_peak_gb, float(rt.peak_memory_gb)), 4),
                )
                merged_meta = dict(retry_res.metadata)
                merged_meta["loop_recovery"] = {
                    "action": retry_action,
                    "aborted_at_tokens": len(gen_ids),
                    "detected_period": int(aborted_loop[0]),
                    "detected_repeats": int(aborted_loop[1]),
                    "prior_crop_box": list(crop_boxes[i]) if crop_boxes[i] is not None else None,
                    "prior_rotation_cw": int(rot_cws[i]),
                }
                results.append(
                    replace(
                        retry_res,
                        rotation_applied_cw=(0 if retry_rot_reset else rot_cws[i]),
                        timings=merged_timings,
                        metadata=merged_meta,
                    )
                )
                continue

            t_post0 = time.perf_counter()
            raw_text = decode_fn(gen_ids)
            validation = None
            if fmt == "markdown" and schema_seq[i] is None:
                rendered = canonicalize_markdown(raw_text, table_mode=tmode)
                validation = validate_markdown(rendered, allow_html_tables=(tmode == "html"))
            else:
                rendered = raw_text
            post_ms = (time.perf_counter() - t_post0) * 1000.0

            prep_ms = prep_ms_arr[i]
            vit_ms = mm_res.vit_ms
            prefill_ms = mm_res.prefill_ms
            dec_ms = mm_res.decode_ms
            total_gen = len(gen_ids) + (1 if mm_res.hit_eos else 0)
            drafted_tokens = int(getattr(mm_res, "drafted_tokens", 0))
            accepted_draft_tokens = int(getattr(mm_res, "accepted_draft_tokens", 0))
            verify_passes = int(getattr(mm_res, "verify_passes", 0))
            peak_gb = batch_peak_gb
            decode_tps = (total_gen / (dec_ms / 1000.0)) if dec_ms > 0.0 else 0.0

            zoom_meta: dict[str, Any] | None = None
            if do_zoom_req and (schema_seq[i] is None) and vision_res is not None and vision_res.lines:
                rendered, raw_text, z_res, zoom_meta = self._run_zoom_recovery(
                    oriented_imgs[i],
                    vision_res,
                    rot_cw=rot_cws[i],
                    rendered=rendered,
                    raw_text=raw_text,
                    page_index=indices[i],
                    prompt=prompt_seq[i],
                    output_format=fmt,
                    table_mode=tmode,
                    strict_markdown_grammar=strict_md,
                    speculative=spec_req,
                    draft_k=k_draft,
                    max_tokens=tok_cap,
                    temperature=temperature,
                )
                if z_res is not None:
                    prep_ms += float(z_res.timings.preprocess_ms)
                    vit_ms += float(z_res.timings.vit_ms)
                    prefill_ms += float(z_res.timings.prefill_ms)
                    dec_ms += float(z_res.timings.decode_ms)
                    post_ms += float(z_res.timings.postprocess_ms)
                    total_gen += int(z_res.timings.generated_tokens)
                    drafted_tokens += int(z_res.timings.drafted_tokens)
                    accepted_draft_tokens += int(z_res.timings.accepted_draft_tokens)
                    verify_passes += int(z_res.timings.verify_passes)
                    peak_gb = max(peak_gb, float(z_res.timings.peak_memory_gb))
                    if dec_ms > 0.0:
                        decode_tps = total_gen / (dec_ms / 1000.0)
                    if fmt == "markdown" and schema_seq[i] is None and zoom_meta and zoom_meta.get("appended"):
                        validation = validate_markdown(rendered, allow_html_tables=(tmode == "html"))

            grid_thw = list(mm_res.image_grids[0]) if mm_res.image_grids else [1, 0, 0]
            full_vis_toks = (grid_thw[0] * grid_thw[1] * grid_thw[2]) // 4 if len(grid_thw) == 3 else 0
            dropped_vis = int(getattr(mm_res, "dropped_visual_tokens", 0))
            vis_toks = max(0, full_vis_toks - dropped_vis)
            metadata: dict[str, Any] = {
                "grid_thw": grid_thw,
                "speculative_mode": "batched",
                "vocab_size": vocab_n,
                "wave_stats": mm_stats,
                "dropped_visual_tokens": dropped_vis,
            }
            if adaptive_caps[i] is not None:
                metadata["adaptive_max_pixels"] = int(adaptive_caps[i])  # type: ignore[arg-type]
            if deskew_metas[i] is not None:
                metadata["deskew"] = deskew_metas[i]
            if schema_seq[i] is not None and vision_res is not None:
                line_rot = (int(rot_cws[i]) - int(vision_res.rotation_applied_cw)) % 360
                metadata["field_boxes"] = align_json_values_to_lines(
                    raw_text,
                    vision_res.lines,
                    image_size=oriented_imgs[i].size,
                    rotation_cw=line_rot,
                )
            if zoom_meta is not None:
                metadata["zoom_recovery"] = zoom_meta
            mm_tel = getattr(mm_res, "telemetry", None)
            if mm_tel:
                metadata["logprob_confidence"] = summarize_logprob_confidence(
                    [float(step["logprob"]) for step in mm_tel],
                    probs=[float(step["prob"]) for step in mm_tel],
                    entropies=[float(step["entropy"]) for step in mm_tel],
                ).to_dict()

            item_total_ms = prep_ms + vision_ms_arr[i] + vit_ms + prefill_ms + dec_ms + post_ms
            results.append(
                OCRResult(
                    text=rendered,
                    raw_text=raw_text,
                    backend_used=f"ovis:{self.model_dir.name}",
                    page_index=int(indices[i]),
                    output_format=fmt,
                    table_mode=tmode,
                    rotation_applied_cw=rot_cws[i],
                    cropped_box_px=crop_boxes[i],
                    lines=vision_res.lines if vision_res is not None else (),
                    analysis=vision_res.analysis if vision_res is not None else None,
                    markdown_validation=validation,
                    timings=OCRTimings(
                        preprocess_ms=round(prep_ms, 3),
                        apple_vision_ms=round(vision_ms_arr[i], 3),
                        vit_ms=round(vit_ms, 3),
                        prefill_ms=round(prefill_ms, 3),
                        decode_ms=round(dec_ms, 3),
                        postprocess_ms=round(post_ms, 3),
                        total_ms=round(item_total_ms, 3),
                        decode_tps=round(decode_tps, 2),
                        prompt_tokens=int(mm_res.prompt_tokens),
                        visual_tokens=int(vis_toks),
                        generated_tokens=int(total_gen),
                        drafted_tokens=int(drafted_tokens),
                        accepted_draft_tokens=int(accepted_draft_tokens),
                        verify_passes=int(verify_passes),
                        peak_memory_gb=round(peak_gb, 4),
                    ),
                    finish_reason=("length" if mm_res.hit_max_tokens else "stop"),
                    metadata=metadata,
                )
            )
        return results

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        *,
        dpi: int = 300,
        page_indices: Sequence[int] | None = None,
        prompt: str | None = None,
        schema: str | Mapping[str, Any] | None = None,
        output_format: OutputFormat | None = None,
        table_mode: MarkdownTableMode | None = None,
        strict_markdown_grammar: bool | None = None,
        zoom_recover: bool | None = None,
    ) -> list[OCRResult]:
        """Render a PDF via `pypdfium2` and run double-buffered Qwen3.5 VLM OCR across pages."""
        rendered = render_pdf_pages(pdf_path, dpi=int(dpi), page_indices=page_indices, include_native_text=True)
        valid_pages = [rp for rp in rendered if rp.image is not None]
        imgs = [rp.image for rp in valid_pages]
        idxs = [rp.page_index for rp in valid_pages]
        nat_hints = [
            rp.native_text.strip()
            if (
                rp.native_control_char_count == 0
                and rp.native_garbled_ratio < 0.02
                and rp.native_text
                and len(rp.native_text.strip()) >= 20
            )
            else None
            for rp in valid_pages
        ]
        return self.ocr_images(
            imgs,
            page_indices=idxs,
            prompt=prompt,
            schema=schema,
            output_format=output_format,
            table_mode=table_mode,
            strict_markdown_grammar=strict_markdown_grammar,
            zoom_recover=zoom_recover,
            native_pdf_texts=nat_hints if any(nat_hints) else None,
        )

    def close(self) -> None:
        """Clear MLX caches and restore any host memory/cache limits applied at init."""
        worker = getattr(self, "_confidence_worker", None)
        if worker is not None:
            worker.close()
            self._confidence_worker = None
        prev_mem = getattr(self, "_prev_mlx_memory_limit", None)
        if prev_mem is not None and callable(getattr(mx, "set_memory_limit", None)):
            mx.set_memory_limit(int(prev_mem))
            self._prev_mlx_memory_limit = None
        prev_cache = getattr(self, "_prev_mlx_cache_limit", None)
        if prev_cache is not None and callable(getattr(mx, "set_cache_limit", None)):
            mx.set_cache_limit(int(prev_cache))
            self._prev_mlx_cache_limit = None
        mx.clear_cache()

    def __enter__(self) -> FastOvisOCREngine:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
