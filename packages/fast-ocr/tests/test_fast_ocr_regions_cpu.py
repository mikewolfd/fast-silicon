"""Pure CPU unit tests for `fast-ocr` Phase A+B enhancements:
  - Cursor-tracked multi-corpus `OCRPLDDrafter` (`pld_drafter.py`)
  - Adaptive visual budgeting, reading-order PLD seeding & region-only VLM routing (`region_router.py`)
  - Perspective homography deskew (`apple_vision.py`, zero MLX)
  - Phase B `FastOvisOCREngine`, `FastOCR`, and `cli.py` wiring
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from fast_ocr.apple_vision import (
    deskew_document_image,
    is_likely_camera_photo,
    normalize_deskew_mode,
    perspective_deskew_image,
)
from fast_ocr.pld_drafter import OCRPLDDrafter
from fast_ocr.region_router import (
    REGION_CROP_MIN_PIXELS,
    REGION_FORMULA_PROMPT,
    REGION_TABLE_PROMPT,
    build_ordered_pld_corpus,
    compute_adaptive_max_pixels,
    estimate_visual_tokens,
    extract_escalation_regions,
    merge_region_markdown,
)
from fast_ocr.types import OCRBox, OCRLine, OCRResult, PageAnalysis

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_cursor_tracked_multi_corpus_pld_drafter() -> None:
    # Repeated 2-gram (10, 11) appears in 3 distinct paragraphs with different continuations.
    corpus = [
        10, 11, 100, 101, 102, 103,
        50, 51, 52, 53,
        10, 11, 200, 201, 202, 203,
        60, 61, 62, 63,
        10, 11, 300, 301, 302, 303,
    ]
    drafter = OCRPLDDrafter(corpus, extra_token_ids=[[900, 901, 902, 903, 904]], max_ngram=4, min_ngram=2, max_k=4)

    # First occurrence at start -> drafts [100, 101, 102, 103]
    assert drafter.draft([10, 11], mode="vision") == [100, 101, 102, 103]
    # Advance past paragraph 1 into paragraph 2 -> cursor moves to [50, 51] then picks 2nd (10, 11)!
    assert drafter.draft([10, 11, 100, 101, 102, 103, 50, 51], mode="vision") == [52, 53, 10, 11]
    assert drafter.draft([50, 51, 52, 53, 10, 11], mode="vision") == [200, 201, 202, 203]
    # Advance into paragraph 3 -> picks 3rd (10, 11)!
    assert drafter.draft([60, 61, 62, 63, 10, 11], mode="vision") == [300, 301, 302, 303]

    # Secondary corpus lookup works and NEVER bleeds across corpus boundary sentinel.
    assert drafter.draft([900, 901], mode="vision") == [902, 903, 904]
    assert drafter.draft([302, 303], mode="vision") == []

    # Tight 56-token windowed 1-gram cursor re-lock recovers immediately across a formatting/punctuation break!
    drafter.reset_cursor()
    assert drafter.draft([10, 11], mode="vision") == [100, 101, 102, 103]
    # Insert divergent formatting token 9999 before unique token 102 -> 2-gram fails, 1-gram window re-locks cleanly!
    assert drafter.draft([10, 11, 100, 9999, 102], mode="vision") == [103, 50, 51, 52]
    # Disabling window_relock turns off 1-gram fallback cleanly.
    drafter_no_rl = OCRPLDDrafter(corpus, max_ngram=4, min_ngram=2, max_k=4, window_relock=False)
    assert drafter_no_rl.draft([10, 11], mode="vision") == [100, 101, 102, 103]
    assert drafter_no_rl.draft([10, 11, 100, 9999, 102], mode="vision") == []

    # Incremental O(1) self-history drafts repeated local structure cleanly.
    drafter.reset_cursor()
    gen = [1, 2, 3, 4, 5, 1, 2]
    assert drafter.draft(gen, mode="self") == [3, 4, 5, 1]
    # Shortened sequence triggers safe self-index reset.
    assert drafter.draft([7, 8, 9, 7, 8], mode="self") == [9, 7, 8]
    # Unrelated equal-or-longer sequence also triggers O(1) anchor reset cleanly.
    assert drafter.draft([21, 22, 23, 24, 25, 21, 22], mode="self") == [23, 24, 25, 21]


def test_ordered_pld_corpus_and_adaptive_max_pixels() -> None:
    # Unsorted 2-column Vision lines (right column detected before left column).
    lines = (
        OCRLine("Right column line 1.", 0.98, OCRBox(0.56, 0.10, 0.92, 0.13)),
        OCRLine("Right column line 2.", 0.98, OCRBox(0.56, 0.14, 0.92, 0.17)),
        OCRLine("Left column line 1.", 0.98, OCRBox(0.06, 0.10, 0.42, 0.13)),
        OCRLine("Left column line 2.", 0.98, OCRBox(0.06, 0.14, 0.42, 0.17)),
    )
    vis = OCRResult(
        text="",
        raw_text="\n".join(l.text for l in lines),
        backend_used="apple-vision:accurate:b3",
        lines=lines,
    )
    primary, extras = build_ordered_pld_corpus(vis, native_pdf_text="Clean native PDF prose.")
    assert primary.index("Left column line 1.") < primary.index("Right column line 1.")
    assert extras[0] == "Clean native PDF prose."
    assert len(extras) == 2

    def _mk_analysis(*, med_h: float, digit_ratio: float = 0.05, table_score: float = 0.0, reasons: tuple[str, ...] = ()) -> PageAnalysis:
        return PageAnalysis(
            width=1700,
            height=2200,
            line_count=40,
            char_count=2000,
            digit_ratio=digit_ratio,
            currency_hits=0,
            mean_confidence=0.96,
            low_conf_line_ratio=0.0,
            median_line_height=med_h,
            inferred_rotation_cw=0,
            ink_crop_box_px=None,
            ink_area_ratio=1.0,
            estimated_columns=1,
            table_score=table_score,
            needs_vlm=False,
            escalation_reasons=reasons,
        )

    # Large clean font (med_h=0.020 * 2200 = 44 px >> 24 px target) -> shrinks max_pixels cleanly!
    large_mp = compute_adaptive_max_pixels(_mk_analysis(med_h=0.020), (1700, 2200))
    assert large_mp < 1_600_000
    assert estimate_visual_tokens((1700, 2200), max_pixels=large_mp) < estimate_visual_tokens((1700, 2200)) * 0.45

    # Tiny font (med_h=0.008 * 2200 = 17.6 px) or low confidence -> keeps full 16.78M native budget!
    assert compute_adaptive_max_pixels(_mk_analysis(med_h=0.008), (1700, 2200)) == 16_777_216
    assert compute_adaptive_max_pixels(
        _mk_analysis(med_h=0.020, reasons=("low_vision_confidence",)),
        (1700, 2200),
    ) == 16_777_216
    # Dense financial table raises target line height to 28 px -> retains more pixels than prose.
    tbl_mp = compute_adaptive_max_pixels(_mk_analysis(med_h=0.016, table_score=0.8), (1700, 2200))
    prose_mp = compute_adaptive_max_pixels(_mk_analysis(med_h=0.016, table_score=0.0), (1700, 2200))
    assert tbl_mp > prose_mp
    # Post-crop composition (`cropped_size`) scales cropped dimensions cleanly using oriented line height.
    cropped_mp = compute_adaptive_max_pixels(
        _mk_analysis(med_h=0.020), (4900, 6100), cropped_size=(3600, 4800)
    )
    uncropped_mp = compute_adaptive_max_pixels(_mk_analysis(med_h=0.020), (4900, 6100))
    assert 786_432 <= cropped_mp < uncropped_mp < 3_000_000
    assert REGION_CROP_MIN_PIXELS == 4096


def test_region_only_escalation_and_merge_roundtrip() -> None:
    # Mixed page: 4 prose lines top, compact 3-row table in middle, 2 prose lines bottom, 1 equation band.
    lines = (
        OCRLine("Executive Summary", 0.99, OCRBox(0.08, 0.05, 0.40, 0.075)),
        OCRLine("Revenue expanded across core segments during the fiscal year.", 0.98, OCRBox(0.08, 0.09, 0.85, 0.11)),
        OCRLine("Operating margins remained resilient despite supply shifts.", 0.98, OCRBox(0.08, 0.12, 0.82, 0.14)),
        OCRLine("Key quarterly highlights are tabulated below for reference.", 0.98, OCRBox(0.08, 0.15, 0.80, 0.17)),
        # Table region (3 aligned rows, y in [0.24, 0.34])
        OCRLine("Metric", 0.96, OCRBox(0.10, 0.24, 0.25, 0.26)),
        OCRLine("2025", 0.96, OCRBox(0.50, 0.24, 0.62, 0.26)),
        OCRLine("2026", 0.96, OCRBox(0.74, 0.24, 0.86, 0.26)),
        OCRLine("Revenue", 0.96, OCRBox(0.10, 0.28, 0.28, 0.30)),
        OCRLine("$1,250.00", 0.95, OCRBox(0.50, 0.28, 0.64, 0.30)),
        OCRLine("$1,480.00", 0.95, OCRBox(0.74, 0.28, 0.88, 0.30)),
        OCRLine("Net Income", 0.96, OCRBox(0.10, 0.32, 0.30, 0.34)),
        OCRLine("$310.00", 0.95, OCRBox(0.50, 0.32, 0.64, 0.34)),
        OCRLine("$405.00", 0.95, OCRBox(0.74, 0.32, 0.88, 0.34)),
        # Prose after table
        OCRLine("The growth factor satisfies the normalized identity:", 0.98, OCRBox(0.08, 0.42, 0.78, 0.44)),
        # Formula region (low-conf math symbols)
        OCRLine("∑ α_i * x_i = √(β^2 + γ^2)", 0.58, OCRBox(0.20, 0.50, 0.68, 0.53)),
        # Closing prose
        OCRLine("All figures are audited in thousands of dollars.", 0.98, OCRBox(0.08, 0.62, 0.72, 0.64)),
    )
    vis = OCRResult(text="", raw_text="", backend_used="apple-vision:accurate:b3", lines=lines)
    # Default (`allow_formula_regions=False`) declines formula-bearing pages so inline math stays full-page VLM.
    segs_def, ok_def = extract_escalation_regions(vis, (1000, 1400))
    assert ok_def is False and segs_def == []

    # Pure prose + table page (no formula bands) routes cleanly under default settings.
    prose_table_lines = tuple(ln for ln in lines if "∑" not in ln.text)
    vis_pt = OCRResult(text="", raw_text="", backend_used="apple-vision:accurate:b3", lines=prose_table_lines)
    segs_pt, ok_pt = extract_escalation_regions(vis_pt, (1000, 1400))
    assert ok_pt is True
    assert [s.kind for s in segs_pt if s.kind != "prose"] == ["table"]

    # Explicit opt-in (`allow_formula_regions=True`) exposes both table and formula crops in reading order.
    segs, ok = extract_escalation_regions(vis, (1000, 1400), allow_formula_regions=True)
    assert ok is True
    vlm_segs = [s for s in segs if s.kind != "prose"]
    assert [s.kind for s in vlm_segs] == ["table", "formula"]
    assert vlm_segs[0].prompt == REGION_TABLE_PROMPT and vlm_segs[0].crop_box_px is not None
    assert vlm_segs[1].prompt == REGION_FORMULA_PROMPT and vlm_segs[1].crop_box_px is not None

    merged = merge_region_markdown(
        segs,
        [
            "<table><tr><th>Metric</th><th>2025</th><th>2026</th></tr><tr><td>Revenue</td><td>$1,250.00</td><td>$1,480.00</td></tr></table>",
            r"$$\sum \alpha_i x_i = \sqrt{\beta^2 + \gamma^2}$$",
        ],
        table_mode="gfm",
    )
    assert "| Metric | 2025 | 2026 |" in merged
    assert r"$$\sum \alpha_i x_i = \sqrt{\beta^2 + \gamma^2}$$" in merged
    assert merged.index("Executive Summary") < merged.index("| Metric |") < merged.index("All figures are audited")

    # Full-page table (>58% area / <4 prose lines) declines region routing -> caller escalates whole page.
    table_only_lines = tuple(
        OCRLine(f"C{c}_{r}", 0.95, OCRBox(0.05 + 0.30 * c, 0.05 + 0.12 * r, 0.25 + 0.30 * c, 0.08 + 0.12 * r))
        for r in range(7)
        for c in range(3)
    )
    vis_full_tbl = OCRResult(text="", raw_text="", backend_used="apple-vision:accurate:b3", lines=table_only_lines)
    segs_no, ok_no = extract_escalation_regions(vis_full_tbl, (1000, 1400))
    assert ok_no is False and segs_no == []


def test_perspective_deskew_homography_and_zero_mlx() -> None:
    # Draw a bright red marker at the top-left of a tilted white quad inside a dark background.
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    arr[90:310, 80:320] = 240
    arr[90:120, 80:110] = [255, 0, 0]
    img = Image.fromarray(arr, mode="RGB")

    tilted_quad = ((80 / 400, 90 / 400), (320 / 400, 95 / 400), (315 / 400, 310 / 400), (85 / 400, 305 / 400))
    warped, info = deskew_document_image(img, quad_detector=lambda _im: (tilted_quad, 0.96))
    assert info["applied"] is True
    assert info["confidence"] == pytest.approx(0.96)
    assert 220 <= warped.width <= 250 and 205 <= warped.height <= 230
    # Top-left corner of warped output must map back to the red marker!
    tl_px = warped.getpixel((5, 5))
    assert tl_px[0] > 200 and tl_px[1] < 50

    # Nearly full-frame upright quad stays untouched (`applied=False`).
    flat_quad = ((0.005, 0.005), (0.995, 0.005), (0.995, 0.995), (0.005, 0.995))
    out_flat, applied_flat = perspective_deskew_image(img, flat_quad)
    assert applied_flat is False and out_flat is img

    # Low-confidence quad is rejected cleanly.
    out_low, info_low = deskew_document_image(img, min_confidence=0.80, quad_detector=lambda _im: (tilted_quad, 0.45))
    assert info_low["applied"] is False and info_low["reason"] == "low_confidence" and out_low.size == img.size

    # Camera-photo auto gate: rejects white-margin scans/PDFs & solid saturated slides without calling quad_detector!
    assert normalize_deskew_mode(False) == "off"
    assert normalize_deskew_mode(True) == "on"
    assert normalize_deskew_mode("auto") == "auto"

    white_pdf = Image.new("RGB", (400, 500), (255, 255, 255))
    assert is_likely_camera_photo(white_pdf) is False
    called_det = []
    _, info_skip = deskew_document_image(
        white_pdf,
        mode="auto",
        quad_detector=lambda _im: called_det.append(1) or (tilted_quad, 0.99),  # type: ignore[func-returns-value]
    )
    assert info_skip["applied"] is False and info_skip["reason"] == "not_camera_photo" and called_det == []

    solid_ppt = Image.new("RGB", (400, 300), (0, 0, 255))
    assert is_likely_camera_photo(solid_ppt) is False

    # Simulated camera receipt photo: warm wood/dark textured border + bright paper center -> auto gate triggers!
    rng = np.random.default_rng(7)
    photo_arr = rng.integers(55, 115, size=(400, 400, 3), dtype=np.uint8)
    photo_arr[:, :, 0] = np.clip(photo_arr[:, :, 0].astype(np.int32) + 25, 0, 255).astype(np.uint8)
    photo_arr[90:310, 80:320] = 235
    photo_img = Image.fromarray(photo_arr, mode="RGB")
    assert is_likely_camera_photo(photo_img) is True
    # Tilted receipt quad (area ~0.33 <= 0.84) warps under mode="auto".
    w_auto, info_auto = deskew_document_image(photo_img, mode="auto", quad_detector=lambda _im: (tilted_quad, 0.97))
    assert info_auto["applied"] is True and info_auto["area_ratio"] <= 0.84 and w_auto.size != photo_img.size
    # High-area full-frame scan quad (area ~0.92 > 0.84) is blocked under mode="auto"!
    big_quad = ((0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98))
    _, info_big = deskew_document_image(photo_img, mode="auto", quad_detector=lambda _im: (big_quad, 0.99))
    assert info_big["applied"] is False and info_big["reason"] == "full_frame_area"

    # Verify zero MLX imports when using deskew_document_image standalone.
    sub_code = (
        f"import sys; sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "from PIL import Image\n"
        "from fast_ocr.apple_vision import deskew_document_image\n"
        "im = Image.new('RGB', (200, 200), (255, 255, 255))\n"
        "_, info = deskew_document_image(im, quad_detector=lambda _: (((0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9)), 0.95))\n"
        "assert info['applied'] is True\n"
        "assert 'mlx' not in sys.modules\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", sub_code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_phase_b_engine_pipeline_cli_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect
    from typing import Any
    import mlx.core as mx
    from tokenizers import AddedToken, Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_qwen3.multimodal_batch import BatchedMMItemResult
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    from fast_ocr.cli import build_parser
    from fast_ocr.ovis_engine import FastOvisOCREngine
    from fast_ocr.pipeline import FastOCR
    from fast_ocr.types import OCRTimings

    # 1. Verified optimal defaults: adaptive_resolution=True, deskew="auto", region_routing=False.
    assert inspect.signature(FastOvisOCREngine.__init__).parameters["adaptive_resolution"].default is True
    assert inspect.signature(FastOvisOCREngine.__init__).parameters["deskew"].default == "auto"
    assert inspect.signature(FastOCR.__init__).parameters["adaptive_resolution"].default is True
    assert inspect.signature(FastOCR.__init__).parameters["deskew"].default == "auto"
    assert inspect.signature(FastOCR.__init__).parameters["region_routing"].default is False

    args_def = build_parser().parse_args(["run", "dummy.png"])
    assert args_def.batch_size == 16
    assert args_def.no_adaptive_resolution is False
    assert args_def.deskew == "auto"
    assert args_def.region_routing is False
    args_opt = build_parser().parse_args(["run", "dummy.png", "--no-adaptive-resolution", "--deskew", "--region-routing"])
    assert args_opt.no_adaptive_resolution is True
    assert args_opt.deskew == "on"
    assert args_opt.region_routing is True
    args_off = build_parser().parse_args(["run", "dummy.png", "--no-deskew"])
    assert args_off.deskew == "off"

    # 2. FastOvisOCREngine.ocr_image & ocr_images: adaptive_24 cap, min_pixels hook, per-item prompts, ordered PLD + native_pdf_text.
    vocab: dict[str, int] = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    bpe = HfTokenizer(BPE(vocab=vocab, merges=[]))
    bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bpe.decoder = ByteLevelDecoder(add_prefix_space=False)
    bpe.add_special_tokens(
        [
            AddedToken("<|endoftext|>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
            AddedToken("<think>", special=True),
            AddedToken("</think>", special=True),
            AddedToken("<|vision_start|>", special=True),
            AddedToken("<|image_pad|>", special=True),
            AddedToken("<|vision_end|>", special=True),
        ]
    )
    tok_p = tmp_path / "tok_phase_b.json"
    bpe.save(str(tok_p))
    ftok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))

    class _StubLM:
        vocab_n = ftok.vocab_size
        has_full_vocab_fallback = False

        class cfg:
            rot_dims = 64
            rope_theta = 1000000.0
            mrope_section = (11, 11, 10)

        def make_cache(self, *, kv_step: int = 256, kv_i8: bool | str = False):
            return []

        def embed_tokens(self, x: mx.array) -> mx.array:
            return mx.zeros((int(x.shape[0]), int(x.shape[1]), 16), dtype=mx.bfloat16)

        def forward_hidden(self, _ids: Any = None, _cache: Any = None, *, inputs_embeds: mx.array | None = None, **_kw: Any) -> mx.array:
            b = int(inputs_embeds.shape[0]) if inputs_embeds is not None else 1
            s = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 1
            return mx.zeros((b, s, 16), dtype=mx.bfloat16)

        def project_logits(self, h: mx.array, **_kw: Any) -> mx.array:
            out = np.full((int(h.shape[0]), int(h.shape[1]), ftok.vocab_size), -20.0, dtype=np.float32)
            out[:, :, ftok.eos_token_id] = 10.0
            return mx.array(out)

    class _StubVision:
        spatial_merge_size = 2

        def __call__(self, px: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
            n_vis = sum((t * h * w) // 4 for t, h, w in grids)
            return mx.zeros((n_vis, 16), dtype=mx.bfloat16)

    eng = object.__new__(FastOvisOCREngine)
    eng.model_dir = tmp_path
    eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    eng.output_format = "markdown"
    eng.table_mode = "gfm"
    eng.strict_markdown_grammar = False
    eng.zoom_recover = False
    eng.crop_margins = False
    eng.crop_pad_px = 0
    eng.auto_orient = False
    eng.speculative = "vision+self"
    eng.draft_k = 4
    eng.prefill_chunk_size = 1024
    eng.max_tokens = 128
    eng.loop_breaker = False
    eng.auto_loop_recover = False
    eng.image_token_id = ftok.backend_tokenizer.token_to_id("<|image_pad|>")
    eng.tokenizer = ftok
    eng.vocab_remap = None
    eng.vision_model = _StubVision()
    eng.lm = _StubLM()
    eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    eng._new_to_old_mx = None
    eng._old_to_new = None
    eng._vision_prepass = None

    big_analysis = PageAnalysis(
        width=1700,
        height=2200,
        line_count=24,
        char_count=1200,
        digit_ratio=0.05,
        currency_hits=0,
        mean_confidence=0.98,
        low_conf_line_ratio=0.0,
        median_line_height=0.025,
        inferred_rotation_cw=0,
        ink_crop_box_px=None,
        ink_area_ratio=1.0,
        estimated_columns=1,
        table_score=0.0,
        needs_vlm=True,
        escalation_reasons=(),
    )
    pre_big = OCRResult(
        text="Alpha Beta",
        raw_text="Alpha Beta",
        backend_used="apple-vision:accurate:b3",
        lines=(OCRLine("Alpha Beta", 0.99, OCRBox(0.1, 0.1, 0.5, 0.125)),),
        analysis=big_analysis,
    )
    seen_vi: list[tuple[int, int]] = []
    orig_build_vi = __import__("fast_ocr.ovis_engine", fromlist=["build_qwen_vision_inputs"]).build_qwen_vision_inputs

    def _spy_build_vi(image: Image.Image, **kw: Any) -> Any:
        seen_vi.append((int(kw.get("min_pixels", 65536)), int(kw.get("max_pixels", 16777216))))
        return orig_build_vi(image, **kw)

    seen_drafter_k: list[int] = []
    orig_drafter = OCRPLDDrafter

    def _spy_drafter(*args: Any, **kw: Any) -> OCRPLDDrafter:
        seen_drafter_k.append(int(kw.get("max_k", 4)))
        return orig_drafter(*args, **kw)

    monkeypatch.setattr("fast_ocr.ovis_engine.build_qwen_vision_inputs", _spy_build_vi)
    monkeypatch.setattr("fast_ocr.ovis_engine.OCRPLDDrafter", _spy_drafter)

    big_im = Image.new("RGB", (1700, 2200), "white")
    res_adap = eng.ocr_image(big_im, pre_analysis=pre_big, native_pdf_text="Clean native prose text for drafting.")
    assert seen_vi[-1][0] == 65_536
    assert seen_vi[-1][1] < 1_600_000
    assert res_adap.metadata["adaptive_max_pixels"] == seen_vi[-1][1]
    assert seen_drafter_k[-1] == 6  # native PDF prose auto-bumps draft_k to 6 when draft_k is None

    # Explicit crop min_pixels=REGION_CROP_MIN_PIXELS + adaptive_resolution=False
    crop_im = Image.new("RGB", (128, 96), "white")
    _ = eng.ocr_image(crop_im, adaptive_resolution=False, min_pixels=REGION_CROP_MIN_PIXELS, speculative="off")
    assert seen_vi[-1] == (REGION_CROP_MIN_PIXELS, 16_777_216)

    # Batched ocr_images forwards min_pixels + per-item prompts cleanly into batched_multimodal_generate
    mm_seen: dict[str, Any] = {}

    def _spy_batched_mm(**kw: Any):
        mm_seen["min_pixels"] = kw["min_pixels"]
        mm_seen["prompts_distinct"] = len({tuple(kw["prompt_builder"](i, [4])[0]) for i in range(len(kw["images_per_item"]))})
        ids_hi = ftok.encode("| Col |\n| --- |\n| 1 |", add_special_tokens=False)
        return [
            BatchedMMItemResult(
                generated_ids=ids_hi,
                prompt_tokens=20,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=1.0,
                image_grids=[(1, 4, 4)],
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _spy_batched_mm)
    _ = eng.ocr_images(
        [crop_im, crop_im],
        prompts=[REGION_TABLE_PROMPT, REGION_FORMULA_PROMPT],
        min_pixels=REGION_CROP_MIN_PIXELS,
        batch_size=4,
    )
    assert mm_seen == {"min_pixels": REGION_CROP_MIN_PIXELS, "prompts_distinct": 2}

    # 3. FastOCR opt-in region_routing=True batches table crops with min_pixels=4096 & falls back on formula pages.
    pt_lines = (
        OCRLine("Executive Summary", 0.99, OCRBox(0.08, 0.05, 0.40, 0.075)),
        OCRLine("Revenue expanded across core segments during the fiscal year.", 0.98, OCRBox(0.08, 0.09, 0.85, 0.11)),
        OCRLine("Operating margins remained resilient despite supply shifts.", 0.98, OCRBox(0.08, 0.12, 0.82, 0.14)),
        OCRLine("Key quarterly highlights are tabulated below for reference.", 0.98, OCRBox(0.08, 0.15, 0.80, 0.17)),
        OCRLine("Metric", 0.96, OCRBox(0.10, 0.24, 0.25, 0.26)),
        OCRLine("2025", 0.96, OCRBox(0.50, 0.24, 0.62, 0.26)),
        OCRLine("2026", 0.96, OCRBox(0.74, 0.24, 0.86, 0.26)),
        OCRLine("Revenue", 0.96, OCRBox(0.10, 0.28, 0.28, 0.30)),
        OCRLine("$1,250.00", 0.95, OCRBox(0.50, 0.28, 0.64, 0.30)),
        OCRLine("$1,480.00", 0.95, OCRBox(0.74, 0.28, 0.88, 0.30)),
        OCRLine("All figures are audited in thousands of dollars.", 0.98, OCRBox(0.08, 0.62, 0.72, 0.64)),
    )
    esc_analysis = PageAnalysis(
        width=1000,
        height=1400,
        line_count=len(pt_lines),
        char_count=500,
        digit_ratio=0.25,
        currency_hits=2,
        mean_confidence=0.97,
        low_conf_line_ratio=0.0,
        median_line_height=0.02,
        inferred_rotation_cw=0,
        ink_crop_box_px=None,
        ink_area_ratio=1.0,
        estimated_columns=1,
        table_score=0.7,
        needs_vlm=True,
        escalation_reasons=("financial_table",),
    )
    vis_pt = OCRResult(
        text="raw",
        raw_text="raw",
        backend_used="apple-vision:accurate:b3",
        lines=pt_lines,
        analysis=esc_analysis,
        timings=OCRTimings(apple_vision_ms=12.0, total_ms=12.0),
    )

    class _FakeVision:
        def recognize_image(self, _src: Any, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            return OCRResult(
                text=vis_pt.text,
                raw_text=vis_pt.raw_text,
                backend_used=vis_pt.backend_used,
                page_index=page_index,
                output_format=output_format,  # type: ignore[arg-type]
                lines=vis_pt.lines,
                analysis=vis_pt.analysis,
                timings=vis_pt.timings,
            )

    crop_spy: list[dict[str, Any]] = []

    class _FakeOvis:
        def ocr_image(self, src: Image.Image, **kw: Any) -> OCRResult:
            crop_spy.append({"size": src.size, **kw})
            return OCRResult(
                text="<table><tr><th>Metric</th><th>2025</th><th>2026</th></tr><tr><td>Revenue</td><td>$1,250.00</td><td>$1,480.00</td></tr></table>",
                raw_text="table",
                backend_used="ovis:stub",
                timings=OCRTimings(decode_ms=5.0, total_ms=10.0, generated_tokens=20),
            )

        def ocr_images(self, srcs: list[Image.Image], **kw: Any) -> list[OCRResult]:
            return [self.ocr_image(s, **kw) for s in srcs]

    page_im = Image.new("RGB", (1000, 1400), "white")
    with FastOCR(backend="auto", region_routing=True, vision_ocr=_FakeVision(), ovis_engine=_FakeOvis()) as pipe:
        out_r = pipe.ocr_image(page_im)
    assert out_r.backend_used == "ovis:stub+ane-regions"
    assert out_r.metadata["region_routing"]["min_pixels"] == REGION_CROP_MIN_PIXELS
    assert len(crop_spy) == 1 and crop_spy[0]["min_pixels"] == REGION_CROP_MIN_PIXELS
    assert "| Metric | 2025 | 2026 |" in out_r.text
    assert out_r.text.index("Executive Summary") < out_r.text.index("| Metric |") < out_r.text.index("All figures are audited")

    # Verify top-level `import fast_ocr` still imports ZERO MLX!
    sub_zero = (
        f"import sys; sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import fast_ocr\n"
        "assert fast_ocr.REGION_CROP_MIN_PIXELS == 4096\n"
        "assert 'mlx' not in sys.modules\n"
    )
    proc_z = subprocess.run([sys.executable, "-B", "-c", sub_zero], capture_output=True, text=True, check=False)
    assert proc_z.returncode == 0, proc_z.stderr


def test_ovis_auto_upscales_small_markdown_tables_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from PIL import Image
    from tokenizers import AddedToken, Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_qwen3.multimodal_batch import BatchedMMItemResult
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    from fast_ocr.ovis_engine import FastOvisOCREngine, _resolve_ovis_min_pixels

    # Pure resolver contract: upscale compact unpinned Markdown tables (<150K px, >=50K px, min_side>=100, aspect<=4),
    # while skipping tiny/narrow table strips (like FinTabNet[11] 376x87) and preserving overrides/raw/JSON.
    assert _resolve_ovis_min_pixels((320, 200), min_pixels=None, max_pixels=None, output_format="markdown", schema=None) == 262_144
    assert _resolve_ovis_min_pixels((376, 87), min_pixels=None, max_pixels=None, output_format="markdown", schema=None) == 65_536
    assert _resolve_ovis_min_pixels((320, 200), min_pixels=REGION_CROP_MIN_PIXELS, max_pixels=None, output_format="markdown", schema=None) == REGION_CROP_MIN_PIXELS
    assert _resolve_ovis_min_pixels((320, 200), min_pixels=None, max_pixels=786_432, output_format="markdown", schema=None) == 65_536
    assert _resolve_ovis_min_pixels((320, 200), min_pixels=None, max_pixels=None, output_format="raw", schema=None) == 65_536
    assert _resolve_ovis_min_pixels((320, 200), min_pixels=None, max_pixels=None, output_format="markdown", schema={"type": "object"}) == 65_536
    assert _resolve_ovis_min_pixels((64, 64), min_pixels=None, max_pixels=None, output_format="markdown", schema=None) == 65_536
    assert _resolve_ovis_min_pixels((600, 400), min_pixels=None, max_pixels=None, output_format="markdown", schema=None) == 65_536

    vocab = {ch: idx for idx, ch in enumerate(sorted(ByteLevel.alphabet()))}
    bpe = HfTokenizer(BPE(vocab=vocab, merges=[]))
    bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bpe.decoder = ByteLevelDecoder(add_prefix_space=False)
    bpe.add_special_tokens(
        [AddedToken(tok, special=True) for tok in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
    )
    tok_p = tmp_path / "tok_tiny.json"
    bpe.save(str(tok_p))
    ftok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))
    img_tok_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))
    eng = object.__new__(FastOvisOCREngine)
    eng.model_dir = tmp_path
    eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    eng.output_format = "markdown"
    eng.table_mode = "gfm"
    eng.strict_markdown_grammar = False
    eng.zoom_recover = False
    eng.crop_margins = False
    eng.crop_pad_px = 0
    eng.auto_orient = False
    eng.adaptive_resolution = True
    eng.deskew = "off"
    eng.speculative = "off"
    eng.draft_k = 4
    eng.prefill_chunk_size = 512
    eng.max_tokens = 256
    eng.loop_breaker = False
    eng.auto_loop_recover = False
    eng.kv_i8 = "auto"
    eng.gdn_state_mode = "auto"
    eng.drop_white_patches = False
    eng.image_token_id = img_tok_id
    eng.tokenizer = ftok
    eng.vocab_remap = None
    eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    eng._new_to_old_mx = None
    eng._old_to_new = None
    eng._vision_prepass = None
    eng.vision_model = object()  # type: ignore[assignment]

    class _DummyLM:
        vocab_n = ftok.vocab_size

        class cfg:
            head_dim = 256

    eng.lm = _DummyLM()  # type: ignore[assignment]

    seen_per_item: list[Any] = []

    def _spy_mm(**kw: Any):
        seen_per_item.append(kw["min_pixels_per_item"])
        return [
            BatchedMMItemResult(
                generated_ids=ftok.encode("| A |\n| --- |\n| 1 |", add_special_tokens=False),
                prompt_tokens=20,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=1.0,
                image_grids=[(1, 4, 4)],
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _spy_mm)
    tiny_tbl = Image.new("RGB", (320, 200), "white")
    big_pg = Image.new("RGB", (800, 800), "white")
    _ = eng.ocr_images([tiny_tbl, big_pg], batch_size=4)
    assert seen_per_item[-1] == [262_144, 65_536]
    _ = eng.ocr_images([tiny_tbl, big_pg], schemas=[{"type": "object"}, None], batch_size=4)
    assert seen_per_item[-1] is None
