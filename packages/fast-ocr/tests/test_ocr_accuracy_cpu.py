"""CPU-only unit tests for Rank #4a/#4b/#6/#7 fast-ocr accuracy improvements."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from fast_ocr import (
    DEFAULT_OVIS_GFM_REPO,
    DEFAULT_OVIS_REPO,
    FastOCR,
    normalize_faded_scan_image,
    resolve_ocr_model_dir,
    strip_running_headers_footers,
)
from fast_ocr.ovis_engine import FastOvisOCREngine
from fast_ocr.paths import resolve_v7final_local_dir
from fast_ocr.types import OCRBox, OCRLine, OCRResult, OCRTimings


def test_strip_running_headers_footers_catches_real_patterns_and_preserves_content() -> None:
    # 1. Split header + page number + bottom German personal-use / copyright footer (b08d60393 pattern)
    doc_de = (
        "Inhaltsverzeichnis 11\n\n"
        "# 1 Grundlagen\n\n"
        "Kinder wachsen in unterschiedlichen Familien auf.\n\n"
        "Dieses Dokument ist nur fuer den persoenlichen Gebrauch bestimmt und darf in keiner Form "
        "vervielfaeltigt und an Dritte weitergegeben werden.\n\n"
        "Aus M. Zobel: Kinder (9783840928307) \u00a9 2017 Hogrefe Verlag, Goettingen."
    )
    out_de = strip_running_headers_footers(doc_de)
    assert "Inhaltsverzeichnis" not in out_de
    assert "persoenlichen Gebrauch" not in out_de
    assert "Hogrefe Verlag" not in out_de
    assert "# 1 Grundlagen" in out_de
    assert "Kinder wachsen" in out_de

    # 2. Top running header with trailing page number vs same-named title guard via spatial bands (6e6caf4cc)
    vis_lines = [
        OCRLine(text="Programs of Graduate Study in the Basic Biological Sciences", box=OCRBox(0.10, 0.055, 0.65, 0.070), confidence=0.99),
        OCRLine(text="1", box=OCRBox(0.86, 0.055, 0.89, 0.070), confidence=0.99),
        OCRLine(text="Faculty Roster and Courses", box=OCRBox(0.10, 0.140, 0.45, 0.165), confidence=0.99),
        OCRLine(text="Biology offers graduate fellowships.", box=OCRBox(0.10, 0.220, 0.70, 0.240), confidence=0.99),
        OCRLine(text="Admission requires undergraduate calculus.", box=OCRBox(0.10, 0.260, 0.70, 0.280), confidence=0.99),
    ]
    doc_split = (
        "Programs of Graduate Study in the Basic Biological Sciences\n"
        "1\n\n"
        "## Faculty Roster and Courses\n\n"
        "Biology offers graduate fellowships."
    )
    out_split = strip_running_headers_footers(doc_split, vision_lines=vis_lines)
    assert "Programs of Graduate Study" not in out_split
    assert "## Faculty Roster and Courses" in out_split
    assert "Biology offers graduate fellowships." in out_split

    # 3. Leading page number + author header (164 Adam A. ZYCH) + bottom ISSN footer (4b6eb432b)
    doc_auth = (
        "# 164 Adam A. ZYCH\n\n"
        "Older adults participated across three longitudinal waves.\n\n"
        "\u00a9 JVE INTERNATIONAL LTD. ISSN 1392-8716, KAUNAS, LITHUANIA"
    )
    out_auth = strip_running_headers_footers(doc_auth)
    assert "Adam A. ZYCH" not in out_auth
    assert "ISSN 1392-8716" not in out_auth
    assert "Older adults participated" in out_auth

    # 4. Strict preservation: titles without page numbers, GFM tables, equation numbers, citations, body years
    doc_clean = (
        "# Rubric (continued)\n\n"
        "Annual Report 2024\n\n"
        "100 mg daily dosage was administered.\n\n"
        "| Model | Score |\n"
        "| --- | ---: |\n"
        "| Alpha | 94.2 |\n\n"
        "(2)\n\n"
        "[1] Smith, J. (2020). Archival Journal."
    )
    assert strip_running_headers_footers(doc_clean) == doc_clean


def test_normalize_faded_scan_image_gating_and_enhancement() -> None:
    # 1. Clean high-contrast synthetic page -> NO-OP
    clean = Image.new("RGB", (600, 800), (255, 255, 255))
    draw_c = ImageDraw.Draw(clean)
    draw_c.rectangle([60, 80, 500, 120], fill=(10, 10, 10))
    draw_c.rectangle([60, 160, 480, 185], fill=(20, 20, 20))
    out_c, meta_c = normalize_faded_scan_image(clean)
    assert meta_c is None
    assert out_c is clean

    # 2. Small cropped FinTabNet-like table -> NO-OP (size guard)
    tiny_tbl = Image.new("RGB", (240, 130), (232, 232, 236))
    ImageDraw.Draw(tiny_tbl).rectangle([10, 20, 200, 40], fill=(80, 80, 85))
    out_t, meta_t = normalize_faded_scan_image(tiny_tbl)
    assert meta_t is None
    assert out_t is tiny_tbl

    # 3. Faded sepia/low-contrast scan -> triggers & expands dynamic range
    faded = Image.new("RGB", (600, 800), (218, 208, 190))
    draw_f = ImageDraw.Draw(faded)
    for y in range(120, 620, 40):
        draw_f.rectangle([80, y, 520, y + 6], fill=(115, 110, 102))
    out_f, meta_f = normalize_faded_scan_image(faded)
    assert meta_f is not None and meta_f["applied"] is True
    assert meta_f["reason"] in ("aged_faded", "faint_pencil")
    arr_f = np.asarray(out_f.convert("L"), dtype=np.float32)
    assert float(np.percentile(arr_f, 1.0)) < 60.0
    assert float(np.percentile(arr_f, 90.0)) > 235.0

    # 4. Dark/gray paper page inside white outer border (22.pdf telegram pattern)
    dark_paper = Image.new("RGB", (700, 500), (249, 249, 249))
    draw_d = ImageDraw.Draw(dark_paper)
    draw_d.rectangle([45, 40, 655, 460], fill=(176, 176, 176))
    for y in range(110, 390, 45):
        draw_d.rectangle([100, y, 580, y + 6], fill=(85, 85, 85))
    out_d, meta_d = normalize_faded_scan_image(dark_paper)
    assert meta_d is not None and meta_d["applied"] is True and meta_d["reason"] == "dark_paper"
    arr_d = np.asarray(out_d.convert("L"), dtype=np.float32)
    center_med = float(np.median(arr_d[80:-80, 80:-80]))
    assert center_med > 220.0

    # 5. Camera photo (dark/textured non-white outer border) -> strictly skipped
    rng = np.random.default_rng(7)
    photo_arr = rng.integers(55, 145, size=(500, 500, 3), dtype=np.uint8)
    photo_arr[70:-70, 70:-70] = 210
    photo_arr[120:140, 120:360] = 95
    photo = Image.fromarray(photo_arr, mode="RGB")
    out_p, meta_p = normalize_faded_scan_image(photo)
    assert meta_p is None
    assert out_p is photo


def test_continuation_stitching_and_opt_in_truncation_recovery() -> None:
    prefix = (
        "| ID | Val |\n"
        "| --- | ---: |\n"
        "| row1 | 10 |\n"
        "| row2 | 20 |"
    )
    suffix = (
        "| ID | Val |\n"
        "| --- | ---: |\n"
        "| row3 | 30 |\n"
        "| row4 | 40 |"
    )
    merged = FastOvisOCREngine._stitch_continuation_markdown(prefix, suffix, table_mode="gfm")
    assert merged.count("| ID | Val |") == 1
    assert "| row1 | 10 |" in merged
    assert "| row4 | 40 |" in merged

    # Exact text overlap deduplication
    prefix_p = "Paragraph one ends smoothly here and continues"
    suffix_p = "here and continues with the rest of paragraph one."
    merged_p = FastOvisOCREngine._stitch_continuation_markdown(prefix_p, suffix_p, table_mode="gfm")
    assert merged_p == "Paragraph one ends smoothly here and continues with the rest of paragraph one."

    # _continue_truncated_ocr_image stitches non-overlapping continuation & aggregates timings
    eng = FastOvisOCREngine.__new__(FastOvisOCREngine)
    calls: list[dict[str, object]] = []

    def _fake_ocr_image(_img: object, **kw: object) -> OCRResult:
        calls.append(kw)
        return OCRResult(
            text="| ID | Val |\n| --- | ---: |\n| row3 | 30 |",
            raw_text="| ID | Val |\n| --- | ---: |\n| row3 | 30 |",
            backend_used="ovis:stub",
            finish_reason="stop",
            timings=OCRTimings(decode_ms=12.5, generated_tokens=14),
        )

    eng.ocr_image = _fake_ocr_image  # type: ignore[method-assign]
    merged_c, raw_c, c_res, cont_meta = eng._continue_truncated_ocr_image(
        Image.new("RGB", (64, 64), "white"),
        None,
        rot_cw=0,
        rendered=prefix,
        raw_text=prefix,
        page_index=0,
        table_mode="gfm",
        strict_markdown_grammar=False,
        do_strip_hf=True,
        speculative="auto",
        draft_k=4,
        max_tokens=4096,
        temperature=0.0,
    )
    assert len(calls) == 1
    assert calls[0]["_continuation_depth"] == 1
    assert calls[0]["max_tokens"] == 2048
    assert c_res is not None
    assert cont_meta == {"applied": True, "continuation_tokens": 14, "continuation_finish_reason": "stop"}
    assert "| row1 | 10 |" in merged_c and "| row3 | 30 |" in merged_c
    assert merged_c.count("| ID | Val |") == 1
    assert raw_c.endswith("| row3 | 30 |")


def test_resolve_ocr_model_dir_v7final_presets_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _make_snap(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"stub")
        return root

    fake_home = tmp_path / "home"
    hub_root = fake_home / ".cache" / "huggingface" / "hub"
    base_snap = _make_snap(hub_root / ("models--" + DEFAULT_OVIS_REPO.replace("/", "--")) / "snapshots" / "abc123")
    gfm_local = _make_snap(tmp_path / "ovisocr2_merged_gfm_v7final_8bit")

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_root))
    monkeypatch.delenv("FAST_OCR_MODEL_DIR", raising=False)
    monkeypatch.delenv("FAST_OCR_GFM_MODEL_DIR", raising=False)
    monkeypatch.delenv("FAST_OCR_PREFER_GFM", raising=False)

    # Default resolves base OvisOCR2 snapshot
    assert resolve_ocr_model_dir(local_files_only=True) == base_snap.resolve()

    # FAST_OCR_PREFER_GFM=1 cleanly falls back to base snapshot when v7final is absent locally
    monkeypatch.setenv("FAST_OCR_PREFER_GFM", "1")
    assert resolve_v7final_local_dir() is None
    assert resolve_ocr_model_dir(local_files_only=True) == base_snap.resolve()

    # Setting FAST_OCR_GFM_MODEL_DIR resolves validated v7final for both v7final preset and FAST_OCR_PREFER_GFM=1
    monkeypatch.setenv("FAST_OCR_GFM_MODEL_DIR", str(gfm_local))
    assert resolve_v7final_local_dir() == gfm_local.resolve()
    assert resolve_ocr_model_dir("v7final", local_files_only=True) == gfm_local.resolve()
    assert resolve_ocr_model_dir("ovis-gfm", local_files_only=True) == gfm_local.resolve()
    assert resolve_ocr_model_dir(local_files_only=True) == gfm_local.resolve()

    # Explicit "ovis" also picks v7final only when FAST_OCR_PREFER_GFM=1, else base
    monkeypatch.delenv("FAST_OCR_PREFER_GFM", raising=False)
    assert resolve_ocr_model_dir("ovis", local_files_only=True) == base_snap.resolve()
    assert resolve_ocr_model_dir("v7final", local_files_only=True) == gfm_local.resolve()
    assert DEFAULT_OVIS_GFM_REPO == "mikewolfd/OvisOCR2-8bit-gfm"


def test_fast_ocr_forwards_accuracy_flags_and_normalizes_faded_inputs() -> None:
    captured_ovis_kwargs: dict[str, object] = {}

    class StubOvis:
        def ocr_images(self, sources: object, **kwargs: object) -> list[OCRResult]:
            captured_ovis_kwargs.update(kwargs)
            return [
                OCRResult(
                    text="Main Title\n\nBody text.",
                    raw_text="Main Title\n\nBody text.",
                    backend_used="ovis:stub",
                    page_index=0,
                    timings=OCRTimings(generated_tokens=8),
                )
            ]

    faded = Image.new("RGB", (600, 800), (218, 208, 190))
    ImageDraw.Draw(faded).rectangle([80, 120, 520, 130], fill=(115, 110, 102))

    seen_vis_pils: list[Image.Image] = []

    class StubVision:
        def recognize_image(self, src: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            assert isinstance(src, Image.Image)
            seen_vis_pils.append(src)
            return OCRResult(
                text="",
                raw_text="",
                backend_used="apple-vision",
                page_index=page_index,
                analysis=SimpleNamespace(needs_vlm=True),  # type: ignore[arg-type]
            )

    with FastOCR(
        backend="auto",
        vision_ocr=StubVision(),
        ovis_engine=StubOvis(),
        strip_headers_footers=True,
        normalize_faded_scans=True,
        continue_on_truncation=True,
    ) as pipe:
        res = pipe.ocr_images([faded], batch_size=1)

    assert len(res) == 1
    assert len(seen_vis_pils) == 1
    # Vision pre-pass received contrast-stretched image (>235 bright paper, <60 ink)
    arr = np.asarray(seen_vis_pils[0].convert("L"), dtype=np.float32)
    assert float(np.percentile(arr, 1.0)) < 60.0
    assert float(np.percentile(arr, 90.0)) > 235.0
    assert captured_ovis_kwargs["strip_headers_footers"] is True
    assert captured_ovis_kwargs["normalize_faded_scans"] is True
    assert captured_ovis_kwargs["continue_on_truncation"] is True
