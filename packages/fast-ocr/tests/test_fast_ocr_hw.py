"""Hardware ANE & MLX GPU integration tests for `fast-ocr` (serialized via `/tmp/m4_gpu_validation.lock`)."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterator

from PIL import Image, ImageDraw, ImageFont
import pytest

from fast_ocr.apple_vision import AppleVisionOCR
from fast_ocr.ovis_engine import FastOvisOCREngine, resolve_ocr_model_dir
from fast_ocr.pipeline import FastOCR
from fast_ocr.types import OCRBox, OCRLine, OCRResult, PageAnalysis

pytestmark = pytest.mark.gpu_ane

_GPU_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


@contextmanager
def _gpu_lock() -> Iterator[None]:
    _GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_GPU_LOCK_PATH, "w", encoding="utf-8") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _render_synthetic_table_page(path: Path, *, rotate_cw: int = 0) -> Path:
    img = Image.new("RGB", (1100, 760), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=20)
    draw.text((80, 50), "QUARTERLY LEDGER SUMMARY", fill=(0, 0, 0), font=font)
    draw.text((80, 120), "Line Item", fill=(0, 0, 0), font=small)
    draw.text((540, 120), "Status", fill=(0, 0, 0), font=small)
    draw.text((800, 120), "Amount", fill=(0, 0, 0), font=small)
    rows = [
        ("Office Supplies", "Paid", "$1,250.00"),
        ("Cloud Compute", "Paid", "$3,400.50"),
        ("Travel Expense", "Pending", "$780.25"),
    ]
    y = 185
    for item, status, amt in rows:
        draw.text((80, y), item, fill=(0, 0, 0), font=small)
        draw.text((540, y), status, fill=(0, 0, 0), font=small)
        draw.text((800, y), amt, fill=(0, 0, 0), font=small)
        y += 55
    if rotate_cw == 90:
        img = img.transpose(Image.Transpose.ROTATE_270)
    elif rotate_cw == 180:
        img = img.transpose(Image.Transpose.ROTATE_180)
    elif rotate_cw == 270:
        img = img.transpose(Image.Transpose.ROTATE_90)
    img.save(path)
    return path


def _has_cached_ovis() -> Path | None:
    try:
        return resolve_ocr_model_dir(None, local_files_only=True)
    except Exception:
        return None


def test_apple_vision_independent_ane_orientation_and_markdown_hw(tmp_path: Path) -> None:
    if platform.system() != "Darwin":
        pytest.skip("Apple Vision requires macOS")

    upright = _render_synthetic_table_page(tmp_path / "ledger.png", rotate_cw=0)
    rotated = _render_synthetic_table_page(tmp_path / "ledger_rot90.png", rotate_cw=90)

    # Confirm standalone `apple-vision` reads accurate text + valid GFM without loading MLX
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sub_code = (
        "import sys\n"
        f"sys.path.insert(0, {str(src_dir)!r})\n"
        "from fast_ocr import FastOCR\n"
        "with FastOCR(backend='apple-vision', output_format='markdown') as ocr:\n"
        f"    r0 = ocr.ocr_image({str(upright)!r})\n"
        "    assert 'Office Supplies' in r0.text\n"
        "    assert '3,400.50' in r0.text or '$3,400.50' in r0.text\n"
        "    assert r0.markdown_validation is not None and r0.markdown_validation.valid\n"
        "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", sub_code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Verify auto-orient on real macOS Vision vectors
    vis = AppleVisionOCR(level="accurate", bands=1, auto_orient=True)
    res_rot = vis.recognize_image(rotated, output_format="markdown")
    assert res_rot.rotation_applied_cw in (90, 270)
    assert "Office Supplies" in res_rot.text
    assert res_rot.markdown_validation is not None and res_rot.markdown_validation.valid is True


def test_ovis_engine_speculative_bit_identity_and_structured_markdown_hw(tmp_path: Path) -> None:
    if platform.system() != "Darwin":
        pytest.skip("Apple Silicon MLX test requires macOS")
    model_dir = _has_cached_ovis()
    if model_dir is None:
        pytest.skip("Cached OvisOCR2 snapshot unavailable")

    img_path = _render_synthetic_table_page(tmp_path / "table_hw.png", rotate_cw=0)

    with _gpu_lock():
        with FastOvisOCREngine(
            model_path=model_dir,
            crop_margins=True,
            auto_orient=True,
            draft_k=4,
            max_tokens=1024,
        ) as eng:
            res_off = eng.ocr_image(img_path, output_format="markdown", table_mode="gfm", speculative="off")
            res_spec = eng.ocr_image(img_path, output_format="markdown", table_mode="gfm", speculative="vision+self")
            res_html = eng.ocr_image(img_path, output_format="markdown", table_mode="html", speculative="off")
            res_strict_off = eng.ocr_image(
                img_path,
                output_format="markdown",
                table_mode="gfm",
                strict_markdown_grammar=True,
                speculative="off",
            )
            res_strict = eng.ocr_image(
                img_path,
                output_format="markdown",
                table_mode="gfm",
                strict_markdown_grammar=True,
                speculative="vision+self",
            )
            res_zoom_clean = eng.ocr_image(
                img_path,
                output_format="markdown",
                table_mode="gfm",
                zoom_recover=True,
            )

            # Deterministic real-GPU verification of targeted 2x zoom crop numeric recovery
            src_img = Image.open(img_path).convert("RGB")
            fake_pre = OCRResult(
                text="Travel Expense Pending $780.25",
                raw_text="Travel Expense Pending $780.25",
                backend_used="apple-vision:test",
                lines=(
                    OCRLine(
                        text="Travel Expense Pending $780.25",
                        confidence=0.95,
                        box=OCRBox(0.05, 0.34, 0.90, 0.46),
                    ),
                ),
            )
            recovered_md, _, z_res, z_meta = eng._run_zoom_recovery(
                src_img,
                fake_pre,
                rot_cw=0,
                rendered="| Line Item | Status | Amount |\n| --- | --- | ---: |\n| Office Supplies | Paid | $1,250.00 |",
                raw_text="| Line Item | Status | Amount |\n| --- | --- | ---: |\n| Office Supplies | Paid | $1,250.00 |",
                page_index=0,
                prompt=None,
                output_format="markdown",
                table_mode="gfm",
                strict_markdown_grammar=False,
                speculative="vision+self",
                draft_k=4,
                max_tokens=512,
                temperature=0.0,
            )

        assert res_off.raw_text == res_spec.raw_text
        assert res_off.text == res_spec.text
        assert res_spec.timings.accepted_draft_tokens > 0
        assert res_strict_off.raw_text == res_strict.raw_text
        assert res_strict_off.text == res_strict.text
        assert res_strict.timings.accepted_draft_tokens > 0
        assert "zoom_recovery" not in res_zoom_clean.metadata
        assert z_res is not None
        assert z_meta is not None and z_meta["triggered"] is True and z_meta["appended"] is True
        assert "780.25" in z_meta["recovered_numbers"]
        assert "780.25" in recovered_md
        for item in (res_off, res_spec, res_html, res_strict_off, res_strict, res_zoom_clean):
            assert item.markdown_validation is not None
            assert item.markdown_validation.valid is True
            assert "1,250.00" in item.text
            assert "3,400.50" in item.text
