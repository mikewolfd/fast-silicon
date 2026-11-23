"""Pure CPU / synthetic unit tests for `fast-ocr` (zero model downloads, zero macOS Vision hardware calls)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest

from fast_qwen3 import FastQwen3Model, Qwen3Config

from fast_ocr.apple_vision import (
    AppleVisionOCR,
    analyze_page_from_lines,
    recognize_banded_detections,
)
from fast_ocr.cli import main as cli_main
from fast_ocr.markdown_struct import (
    DynamicTableRowGrammar,
    build_gfm_table_lock_lark,
    canonicalize_markdown,
    synthesize_markdown_from_lines,
    validate_markdown,
)
from fast_ocr.mrope import (
    compute_mrope_position_selector,
    compute_mrope_positions,
    precompute_mrope_trig,
)
from fast_ocr.pipeline import AutoOCRPipeline, FastOCR
from fast_ocr.pld_drafter import OCRPLDDrafter
from fast_ocr.preprocess import (
    build_qwen_vision_inputs,
    compute_adaptive_render_dpi,
    count_control_garble_chars,
    crop_to_ink,
    detect_ink_bbox,
    render_pdf_pages,
    rotate_box_normalized,
    rotate_image_cw,
    smart_resize_qwen,
    strip_running_pdf_boilerplate,
)
from fast_ocr.types import (
    OCRBox,
    OCRLine,
    OCRResult,
    OCRTimings,
    PageAnalysis,
)
from fast_ocr.vision import FastQwen35VisionModel, Qwen35VisionConfig

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _make_tiny_qwen3_model(*, seed: int = 7) -> tuple[Qwen3Config, FastQwen3Model]:
    """Construct a tiny synthetic `FastQwen3Model` (`hidden=64`, `vocab=320`) for CPU VLM hook tests."""
    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=320,
        tie_word_embeddings=True,
        full_attention_interval=4,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.5,
    )
    mx.random.seed(seed)
    h = cfg.hidden_size
    v = cfg.vocab_size
    inter = cfg.intermediate_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = 2 * k_dim + v_dim

    def r_w(out_d: int, in_d: int) -> mx.array:
        return (mx.random.normal((out_d, in_d)) * 0.04).astype(mx.bfloat16)

    def r_n(dim: int) -> mx.array:
        return (mx.ones((dim,), dtype=mx.float32) + mx.random.normal((dim,)) * 0.02).astype(mx.bfloat16)

    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": r_w(v, h),
        "model.norm.weight": r_n(h),
    }
    for i in range(cfg.num_hidden_layers):
        px = f"model.layers.{i}"
        weights[f"{px}.input_layernorm.weight"] = r_n(h)
        weights[f"{px}.post_attention_layernorm.weight"] = r_n(h)
        if cfg.layer_types[i] == "linear_attention":
            gpx = f"{px}.linear_attn"
            weights[f"{gpx}.in_proj_qkv.weight"] = r_w(conv_dim, h)
            weights[f"{gpx}.in_proj_z.weight"] = r_w(v_dim, h)
            weights[f"{gpx}.in_proj_b.weight"] = r_w(cfg.linear_num_value_heads, h)
            weights[f"{gpx}.in_proj_a.weight"] = r_w(cfg.linear_num_value_heads, h)
            weights[f"{gpx}.out_proj.weight"] = r_w(h, v_dim)
            weights[f"{gpx}.conv1d.weight"] = (
                mx.random.normal((conv_dim, 1, cfg.linear_conv_kernel_dim)) * 0.05
            ).astype(mx.bfloat16)
            weights[f"{gpx}.dt_bias"] = mx.ones((cfg.linear_num_value_heads,), dtype=mx.bfloat16) * 0.5
            weights[f"{gpx}.A_log"] = mx.zeros((cfg.linear_num_value_heads,), dtype=mx.bfloat16)
            weights[f"{gpx}.norm.weight"] = mx.ones((cfg.linear_value_head_dim,), dtype=mx.bfloat16)
        else:
            apx = f"{px}.self_attn"
            weights[f"{apx}.q_proj.weight"] = r_w(2 * q_dim, h)
            weights[f"{apx}.k_proj.weight"] = r_w(kv_dim, h)
            weights[f"{apx}.v_proj.weight"] = r_w(kv_dim, h)
            weights[f"{apx}.o_proj.weight"] = r_w(h, q_dim)
            weights[f"{apx}.q_norm.weight"] = r_n(cfg.head_dim)
            weights[f"{apx}.k_norm.weight"] = r_n(cfg.head_dim)
        mpx = f"{px}.mlp"
        weights[f"{mpx}.gate_proj.weight"] = r_w(inter, h)
        weights[f"{mpx}.up_proj.weight"] = r_w(inter, h)
        weights[f"{mpx}.down_proj.weight"] = r_w(h, inter)

    mx.eval(*weights.values())
    return cfg, FastQwen3Model(cfg, weights, enable_fusion=True)


def _write_mixed_two_page_pdf(path: Path, page0_lines: list[str]) -> None:
    """Create a valid 2-page PDF: page 0 holds native Helvetica text, page 1 holds no text (scanned)."""
    ops = ["BT", "/F1 12 Tf", "72 740 Td"]
    for idx, line in enumerate(page0_lines):
        esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if idx > 0:
            ops.append("0 -18 Td")
        ops.append(f"({esc}) Tj")
    ops.append("ET")
    s0 = "\n".join(ops).encode("latin1")
    s1 = b""
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R >>\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"6 0 obj\n<< /Length {len(s0)} >>\nstream\n".encode("latin1") + s0 + b"\nendstream\nendobj\n",
        f"7 0 obj\n<< /Length {len(s1)} >>\nstream\n".encode("latin1") + s1 + b"\nendstream\nendobj\n",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    buf.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
    path.write_bytes(bytes(buf))


def _make_prose_result(page_index: int = 0, *, text: str = "Clean prose paragraph.") -> OCRResult:
    line = OCRLine(text=text, confidence=0.98, box=OCRBox(0.1, 0.1, 0.7, 0.12))
    analysis = PageAnalysis(
        width=800,
        height=1000,
        line_count=1,
        char_count=len(text),
        digit_ratio=0.0,
        currency_hits=0,
        mean_confidence=0.98,
        low_conf_line_ratio=0.0,
        median_line_height=0.02,
        inferred_rotation_cw=0,
        ink_crop_box_px=None,
        ink_area_ratio=1.0,
        estimated_columns=1,
        table_score=0.0,
        needs_vlm=False,
        escalation_reasons=(),
    )
    return OCRResult(
        text=text,
        raw_text=text,
        backend_used="apple-vision:accurate:b1",
        page_index=page_index,
        output_format="markdown",
        table_mode="gfm",
        lines=(line,),
        analysis=analysis,
        markdown_validation=validate_markdown(text, allow_html_tables=False),
        timings=OCRTimings(apple_vision_ms=3.5, total_ms=3.5),
    )


def _make_table_vision_result(page_index: int = 0) -> OCRResult:
    lines = (
        OCRLine(text="Item", confidence=0.97, box=OCRBox(0.1, 0.2, 0.25, 0.22)),
        OCRLine(text="$120.00", confidence=0.97, box=OCRBox(0.6, 0.2, 0.8, 0.22)),
    )
    analysis = PageAnalysis(
        width=800,
        height=1000,
        line_count=12,
        char_count=140,
        digit_ratio=0.35,
        currency_hits=6,
        mean_confidence=0.95,
        low_conf_line_ratio=0.0,
        median_line_height=0.018,
        inferred_rotation_cw=0,
        ink_crop_box_px=(40, 60, 760, 920),
        ink_area_ratio=0.77,
        estimated_columns=3,
        table_score=0.75,
        needs_vlm=True,
        escalation_reasons=("financial_table",),
    )
    return OCRResult(
        text="| Item | Amount |\n| --- | ---: |\n| A | $120.00 |",
        raw_text="Item\n$120.00",
        backend_used="apple-vision:accurate:b1",
        page_index=page_index,
        output_format="markdown",
        table_mode="gfm",
        lines=lines,
        analysis=analysis,
        markdown_validation=validate_markdown("| Item | Amount |\n| --- | ---: |\n| A | $120.00 |", allow_html_tables=False),
        timings=OCRTimings(apple_vision_ms=4.0, total_ms=4.0),
    )


# 1. mrope tests ---------------------------------------------------------------------------------


def test_mrope_positions_interleaving_and_trig() -> None:
    selector = compute_mrope_position_selector(mrope_section=(11, 11, 10), freq_dim=32)
    sel_list = [int(x) for x in selector.tolist()]
    assert len(sel_list) == 32
    assert sel_list.count(0) == 11
    assert sel_list.count(1) == 11
    assert sel_list.count(2) == 10
    assert sel_list[:9] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert sel_list[30:] == [0, 1]

    # Text-only positions yield identity 1D range and delta=0
    pos_txt, delta_txt = compute_mrope_positions(
        [10, 11, 12, 13],
        image_grid_thw=[],
        image_token_id=999,
    )
    assert pos_txt.shape == (3, 1, 4)
    assert delta_txt == 0
    assert pos_txt[:, 0, :].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]

    # Multimodal: 2 text + 6 visual (grid_thw=(1, 4, 6), merge=2 -> 2x3) + 2 text
    img_tok = 999
    prompt = [1, 2] + [img_tok] * 6 + [3, 4]
    pos_mm, delta_mm = compute_mrope_positions(
        prompt,
        image_grid_thw=[(1, 4, 6)],
        image_token_id=img_tok,
        spatial_merge_size=2,
    )
    assert pos_mm.shape == (3, 1, 10)
    # Visual block starts at st_idx=2: t=2, h=[2,2,2,3,3,3], w=[2,3,4,2,3,4], max_pos=4 -> suffix starts at 5
    assert pos_mm[0, 0, :].tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 5, 6]
    assert pos_mm[1, 0, :].tolist() == [0, 1, 2, 2, 2, 3, 3, 3, 5, 6]
    assert pos_mm[2, 0, :].tolist() == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]
    # Next decode pos is 7 at cache.offset=10 -> delta = 7 - 10 = -3
    assert delta_mm == -3

    cos_a, sin_a = precompute_mrope_trig(
        pos_mm,
        rot_dims=64,
        rope_theta=10000000.0,
        mrope_section=(11, 11, 10),
        dtype=mx.bfloat16,
    )
    mx.eval(cos_a, sin_a)
    assert cos_a.shape == (1, 1, 10, 32)
    assert sin_a.shape == (1, 1, 10, 32)
    assert cos_a.dtype == mx.bfloat16
    assert sin_a.dtype == mx.bfloat16
    unit_norm = (cos_a.astype(mx.float32) ** 2) + (sin_a.astype(mx.float32) ** 2)
    assert float(mx.max(mx.abs(unit_norm - 1.0)).item()) < 2e-2


# 2. preprocess & vision tests -------------------------------------------------------------------


def test_preprocess_resize_patchify_crop_and_rotate() -> None:
    h_r, w_r = smart_resize_qwen(241, 319, factor=32, min_pixels=65536, max_pixels=16777216)
    assert h_r % 32 == 0 and w_r % 32 == 0
    assert h_r * w_r >= 65536

    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize_qwen(10, 3000, factor=32)

    # 320x256 fits min_pixels=65536 without resize -> (1, 16, 20) patches -> 80 merged tokens
    img = Image.new("RGB", (320, 256), color=(255, 255, 255))
    vin = build_qwen_vision_inputs(img)
    assert vin.grid_thw == (1, 16, 20)
    assert vin.resized_height == 256 and vin.resized_width == 320
    assert vin.pixel_values.shape == (320, 1536)
    assert vin.pixel_values.dtype == np.float32
    assert vin.num_visual_tokens == 80
    assert np.allclose(vin.pixel_values, 1.0, atol=1e-5)

    # Tiny synthetic VisionTower parity check on preprocessed patches
    vcfg = Qwen35VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        num_position_embeddings=16,
        out_hidden_size=64,
    )
    vis_model = FastQwen35VisionModel(vcfg)
    vis_out = vis_model(mx.array(vin.pixel_values), [vin.grid_thw])
    mx.eval(vis_out)
    assert vis_out.shape == (vin.num_visual_tokens, 64)

    # Ink bbox detection & tight cropping
    arr = np.full((400, 400, 3), 255, dtype=np.uint8)
    arr[120:260, 100:280] = 20
    ink_img = Image.fromarray(arr, mode="RGB")
    bbox = detect_ink_bbox(ink_img)
    assert bbox is not None
    left, top, right, bottom = bbox
    assert left <= 100 and top <= 120 and right >= 280 and bottom >= 260
    cropped, applied_box = crop_to_ink(ink_img)
    assert applied_box == bbox
    assert cropped.size == (right - left, bottom - top)

    # Blank white image should not crop
    blank = Image.new("RGB", (400, 400), color=(255, 255, 255))
    assert detect_ink_bbox(blank) is None
    unchanged, box_none = crop_to_ink(blank)
    assert box_none is None and unchanged.size == (400, 400)

    # Clockwise image & normalized box rotation
    asym = Image.new("RGB", (80, 40), color=(0, 0, 0))
    asym.putpixel((8, 4), (255, 0, 0))
    rot90 = rotate_image_cw(asym, 90)
    assert rot90.size == (40, 80)
    # Original normalized point near (x=0.1, y=0.1) maps under 90 CW to (x=0.9, y=0.1)
    assert rot90.getpixel((40 - 1 - 4, 8)) == (255, 0, 0)

    box0 = (0.1, 0.2, 0.3, 0.5)
    b90 = rotate_box_normalized(box0, 90)
    b180 = rotate_box_normalized(b90, 90)
    b270 = rotate_box_normalized(b180, 90)
    b360 = rotate_box_normalized(b270, 90)
    assert np.allclose(b90, (0.5, 0.1, 0.8, 0.3))
    assert np.allclose(b360, box0)


# 3. fast_qwen3 VLM hooks ------------------------------------------------------------------------


def test_fast_qwen3_vlm_inputs_embeds_and_rope_offset_delta() -> None:
    cfg, model = _make_tiny_qwen3_model(seed=13)
    ids = mx.array([[5, 19, 44, 71, 88, 102]], dtype=mx.int32)

    h_ids = model.forward_hidden(input_ids=ids)
    h_emb = model.forward_hidden(inputs_embeds=model.embed_tokens(ids))
    mx.eval(h_ids, h_emb)
    diff_embeds = float(mx.max(mx.abs(h_ids.astype(mx.float32) - h_emb.astype(mx.float32))).item())
    assert diff_embeds == 0.0

    # Verify `rope_offset_delta` shifts cached decode RoPE identically to explicit `rope_trig`
    c_delta = model.make_cache()
    c_trig = model.make_cache()
    c_zero = model.make_cache()
    _ = model.forward_hidden(ids, c_delta)
    _ = model.forward_hidden(ids, c_trig)
    _ = model.forward_hidden(ids, c_zero)

    delta = -2  # cache.offset=6 -> effective position=4
    dec_id = mx.array([[27]], dtype=mx.int32)
    h_delta = model.forward_hidden(dec_id, c_delta, rope_offset_delta=delta)

    pos_3d = mx.full((3, 1, 1), 4, dtype=mx.int32)
    trig_mrope = precompute_mrope_trig(
        pos_3d,
        rot_dims=cfg.rot_dims,
        rope_theta=cfg.rope_theta,
        mrope_section=(6, 5, 5),
        dtype=mx.bfloat16,
    )
    trig_1d = model.precompute_rope_trig(mx.array([[4]], dtype=mx.int32))
    mx.eval(trig_mrope[0], trig_mrope[1], trig_1d[0], trig_1d[1])
    assert float(mx.max(mx.abs(trig_mrope[0].astype(mx.float32) - trig_1d[0].astype(mx.float32))).item()) == 0.0
    assert float(mx.max(mx.abs(trig_mrope[1].astype(mx.float32) - trig_1d[1].astype(mx.float32))).item()) == 0.0

    h_trig = model.forward_hidden(dec_id, c_trig, rope_trig=trig_mrope)
    h_unshifted = model.forward_hidden(dec_id, c_zero, rope_offset_delta=0)
    mx.eval(h_delta, h_trig, h_unshifted)

    diff_delta_trig = float(mx.max(mx.abs(h_delta.astype(mx.float32) - h_trig.astype(mx.float32))).item())
    diff_delta_unshifted = float(mx.max(mx.abs(h_delta.astype(mx.float32) - h_unshifted.astype(mx.float32))).item())
    assert diff_delta_trig < 2e-2
    assert diff_delta_unshifted > diff_delta_trig


# 4. pld_drafter tests ---------------------------------------------------------------------------


def test_pld_drafter_self_vision_priority_and_max_k() -> None:
    vision_ids = [10, 20, 30, 40, 50, 60, 70]
    drafter = OCRPLDDrafter(vision_ids, max_ngram=3, min_ngram=2, max_k=4)

    # Too short to match min_ngram=2
    assert drafter.draft([10]) == []

    # Vision-PLD match when self-history has no repeat
    assert drafter.draft([99, 20, 30], mode="vision") == [40, 50, 60, 70]
    assert drafter.draft([99, 20, 30], mode="vision+self", max_k=2) == [40, 50]

    # Self-PLD match on repeating table row suffix
    generated = [1, 2, 3, 4, 5, 1, 2]
    assert drafter.draft(generated, mode="self") == [3, 4, 5, 1]

    # Combined priority: when both Self-PLD and Vision-PLD match `(20, 30)`, Self-PLD wins
    both_gen = [20, 30, 81, 82, 83, 20, 30]
    assert drafter.draft(both_gen, mode="vision+self", max_k=3) == [81, 82, 83]
    assert drafter.draft(both_gen, mode="vision", max_k=3) == [40, 50, 60]


# 5. apple_vision tests --------------------------------------------------------------------------


def test_apple_vision_banded_dedup_escalation_and_zero_mlx() -> None:
    # 3 overlapping horizontal bands with duplicate seam line near y=1/3 and y=2/3
    band_calls: list[tuple[int, int]] = []

    def banded_fake_engine(_rgb: bytes, w: int, strip_h: int) -> list[tuple[str, float, list[float]]]:
        idx = len(band_calls)
        band_calls.append((w, strip_h))
        if idx == 0:
            # Top band: one line near top, one near seam (bottom of band 0 -> y=0.04 in strip)
            return [
                ("Top line", 0.99, [0.1, 0.70, 0.4, 0.06]),
                ("Seam line", 0.98, [0.1, 0.04, 0.4, 0.06]),
            ]
        if idx == 1:
            # Middle band: sees same seam line near top of strip (y=0.90) + middle line
            return [
                ("Seam line", 0.98, [0.1, 0.90, 0.4, 0.06]),
                ("Middle line", 0.99, [0.1, 0.45, 0.4, 0.06]),
            ]
        return [("Bottom line", 0.99, [0.1, 0.20, 0.4, 0.06])]

    rgb = bytes(100 * 300 * 3)
    deduped = recognize_banded_detections(banded_fake_engine, rgb, 100, 300, bands=3)
    texts = [t for t, _, _ in deduped]
    assert len(band_calls) == 3
    assert texts.count("Seam line") == 1
    assert set(texts) == {"Top line", "Seam line", "Middle line", "Bottom line"}

    # Page analysis: prose vs financial table escalation
    prose_lines = [
        OCRLine(text=f"Sentence line {i} in a clean paragraph.", confidence=0.98, box=OCRBox(0.1, 0.1 + i * 0.05, 0.85, 0.12 + i * 0.05))
        for i in range(6)
    ]
    prose_analysis = analyze_page_from_lines(prose_lines, None, 800, 1000)
    assert prose_analysis.needs_vlm is False
    assert prose_analysis.escalation_reasons == ()

    table_lines: list[OCRLine] = []
    for r in range(6):
        y0 = 0.2 + r * 0.04
        table_lines.append(OCRLine(text=f"Row{r}", confidence=0.96, box=OCRBox(0.08, y0, 0.22, y0 + 0.02)))
        table_lines.append(OCRLine(text=f"${100 + r}.50", confidence=0.96, box=OCRBox(0.45, y0, 0.62, y0 + 0.02)))
        table_lines.append(OCRLine(text=f"${200 + r}.00", confidence=0.96, box=OCRBox(0.72, y0, 0.88, y0 + 0.02)))
    table_analysis = analyze_page_from_lines(table_lines, None, 800, 1000)
    assert table_analysis.needs_vlm is True
    assert "financial_table" in table_analysis.escalation_reasons
    assert table_analysis.estimated_columns >= 3

    # Verify AppleVisionOCR standalone import & execution never imports mlx or VLM modules
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "from PIL import Image\n"
        "from fast_ocr.apple_vision import AppleVisionOCR\n"
        "def fake(_b, _w, _h):\n"
        "    return [('Clean standalone line.', 0.99, [0.1, 0.8, 0.6, 0.03])]\n"
        "ocr = AppleVisionOCR(engine=fake)\n"
        "res = ocr.recognize_image(Image.new('RGB', (200, 200), (255, 255, 255)))\n"
        "assert res.text == 'Clean standalone line.'\n"
        "for banned in ('mlx', 'mlx.core', 'fast_ocr.vision', 'fast_ocr.ovis_engine'):\n"
        "    assert banned not in sys.modules, f'Unexpected import: {banned}'\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# 6. markdown_struct tests -----------------------------------------------------------------------


def test_markdown_synthesis_canonicalization_validation_and_grammar() -> None:
    # 2-column XY-cut + headings + bullets + de-hyphenation + geometric table with currency & escaped pipe
    lines = [
        OCRLine(text="QUARTERLY REPORT", confidence=0.99, box=OCRBox(0.28, 0.04, 0.72, 0.08)),
        # Left column (x in [0.06, 0.42])
        OCRLine(text="First paragraph wraps across a hyphen-", confidence=0.98, box=OCRBox(0.06, 0.14, 0.42, 0.155)),
        OCRLine(text="ated word cleanly.", confidence=0.98, box=OCRBox(0.06, 0.165, 0.30, 0.18)),
        OCRLine(text="• Bullet alpha item.", confidence=0.98, box=OCRBox(0.06, 0.21, 0.35, 0.225)),
        OCRLine(text="• Bullet beta item.", confidence=0.98, box=OCRBox(0.06, 0.235, 0.35, 0.25)),
        # Right column (x in [0.56, 0.92])
        OCRLine(text="Right column summary text.", confidence=0.98, box=OCRBox(0.56, 0.14, 0.92, 0.155)),
        OCRLine(text="More right column notes.", confidence=0.98, box=OCRBox(0.56, 0.165, 0.90, 0.18)),
        OCRLine(text="Final right note.", confidence=0.98, box=OCRBox(0.56, 0.19, 0.85, 0.205)),
        # Geometric 2-column table below
        OCRLine(text="Category", confidence=0.99, box=OCRBox(0.10, 0.35, 0.28, 0.368)),
        OCRLine(text="Amount", confidence=0.99, box=OCRBox(0.68, 0.35, 0.85, 0.368)),
        OCRLine(text="Alpha|Unit", confidence=0.99, box=OCRBox(0.10, 0.385, 0.30, 0.403)),
        OCRLine(text="$1,250.00", confidence=0.99, box=OCRBox(0.68, 0.385, 0.86, 0.403)),
        OCRLine(text="Beta Unit", confidence=0.99, box=OCRBox(0.10, 0.42, 0.30, 0.438)),
        OCRLine(text="$450.00", confidence=0.99, box=OCRBox(0.68, 0.42, 0.86, 0.438)),
    ]
    md = synthesize_markdown_from_lines(lines)
    assert "# QUARTERLY REPORT" in md
    assert "hyphenated word cleanly." in md
    assert "- Bullet alpha item." in md
    assert md.index("Bullet beta item.") < md.index("Right column summary text.")
    assert "| Category | Amount |" in md
    assert "| --- | ---: |" in md
    assert "Alpha\\|Unit" in md

    val_ok = validate_markdown(md, allow_html_tables=False)
    assert val_ok.valid is True
    assert val_ok.gfm_table_count == 1
    assert val_ok.gfm_tables_rectangular is True

    # Canonicalizer: colspan + rowspan 2D occupancy grid + uneven rows + unclosed fences/math
    html_mixed = (
        "<table>"
        "<tr><th colspan='2'>Metrics</th><th rowspan='2'>Notes</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>$10.00</td><td>$20.00</td><td>OK</td></tr>"
        "<tr><td>$15.00</td><td>$25.00</td></tr>"
        "</table>\n"
        "$$\nx = y + 1\n"
        "```python\nprint(1)"
    )
    gfm_canon = canonicalize_markdown(html_mixed, table_mode="gfm")
    val_gfm = validate_markdown(gfm_canon, allow_html_tables=False)
    assert val_gfm.valid is True
    assert val_gfm.gfm_table_count == 1
    assert val_gfm.fence_balanced is True
    assert val_gfm.math_balanced is True
    assert "| Metrics / Q1 | Metrics / Q2 | Notes |" in gfm_canon

    html_canon = canonicalize_markdown(html_mixed, table_mode="html")
    val_html = validate_markdown(html_canon, allow_html_tables=True)
    assert val_html.valid is True
    assert val_html.html_table_count == 1
    assert val_html.html_tables_balanced is True

    # Malformed validation cases
    assert validate_markdown("```python\nx = 1").fence_balanced is False
    assert validate_markdown("$$\nx = 1").math_balanced is False
    bad_gfm = validate_markdown("| a | b |\n| --- | --- |\n| 1 |", allow_html_tables=False)
    assert bad_gfm.valid is False and bad_gfm.gfm_tables_rectangular is False
    bad_html = validate_markdown("<table><tr><td>x</table>", allow_html_tables=True)
    assert bad_html.valid is False and bad_html.html_tables_balanced is False
    disallowed_html = validate_markdown("<table><tr><td>x</td></tr></table>", allow_html_tables=False)
    assert disallowed_html.valid is False

    # DynamicTableRowGrammar with tiny synthetic Rust ByteLevel tokenizer
    llguidance = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    assert "start: SEP row+ TERM" in build_gfm_table_lock_lark(3)
    vocab: dict[str, int] = {"<|endoftext|>": 0}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llguidance.LLTokenizer(bt.to_str(), eos_token=0)
    gram = DynamicTableRowGrammar(ltok)
    vocab_sz = bt.get_vocab_size()
    nl_id = bt.encode("\n").ids[0]

    header = "| ColA | ColB | ColC |\n"
    ff_ids: list[int] = []
    for tid in bt.encode(header).ids:
        ff_ids.extend(gram.advance_text_and_maybe_lock(bt.decode([tid]), tid))
    assert gram.is_active is True
    assert gram.locked_cols == 3
    assert bt.decode(ff_ids) == "| --- | --- | --- |\n|"

    # Before 3 cells are emitted, newline is blocked by the 3-column Lark lock
    mask_start = gram.prepare_mask(vocab_size=vocab_sz)
    assert mask_start is not None
    assert bool(mask_start[nl_id]) is False

    # Feed 2 cells only (" a | 1 |") -> newline still blocked!
    for tid in bt.encode(" a | 1 |").ids:
        _ = gram.advance_text_and_maybe_lock(bt.decode([tid]), tid)
    mask_mid = gram.prepare_mask(vocab_size=vocab_sz)
    assert mask_mid is not None
    assert bool(mask_mid[nl_id]) is False

    # Feed final cell + closing pipe (" 2 |") -> fast-forwards row newline, then allows TERM newline
    tail_ff: list[int] = []
    for tid in bt.encode(" 2 |").ids:
        tail_ff.extend(gram.advance_text_and_maybe_lock(bt.decode([tid]), tid))
    assert bt.decode(tail_ff) == "\n"
    mask_end = gram.prepare_mask(vocab_size=vocab_sz)
    assert mask_end is not None
    assert bool(mask_end[nl_id]) is True

    # Blank line TERM closes the table lock
    _ = gram.advance_text_and_maybe_lock("\n", nl_id)
    assert gram.is_active is False
    assert gram.locked_cols is None


# 7. pipeline & cli tests ------------------------------------------------------------------------


def test_pipeline_lazy_isolation_auto_routing_and_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Subprocess isolation test: `FastOCR(backend="apple-vision")` never imports `mlx` or `ovis_engine`
    iso_code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "from PIL import Image\n"
        "from fast_ocr.apple_vision import AppleVisionOCR\n"
        "from fast_ocr.pipeline import FastOCR\n"
        "v = AppleVisionOCR(engine=lambda _b, _w, _h: [('Solo page.', 0.99, [0.1, 0.8, 0.5, 0.04])])\n"
        "with FastOCR(backend='apple-vision', vision_ocr=v) as ocr:\n"
        "    res = ocr.ocr_image(Image.new('RGB', (160, 120), (255, 255, 255)))\n"
        "    assert ocr._ovis is None\n"
        "    assert res.text == 'Solo page.'\n"
        "for banned in ('mlx', 'mlx.core', 'fast_ocr.vision', 'fast_ocr.ovis_engine'):\n"
        "    assert banned not in sys.modules, f'Unexpected import: {banned}'\n"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", iso_code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Auto routing: prose stays on Apple Vision; table escalates to Ovis reusing pre_analysis
    class FakeVision:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def recognize_image(self, source: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            self.calls.append(page_index)
            tag = getattr(source, "_tag", "prose")
            if tag == "table":
                return _make_table_vision_result(page_index=page_index)
            return _make_prose_result(page_index=page_index, text=f"Prose page {page_index}.")

    class FakeOvis:
        def __init__(self) -> None:
            self.single_calls: list[tuple[int, OCRResult | None]] = []
            self.batch_calls: list[tuple[list[int], list[OCRResult | None]]] = []
            self.closed = False

        def ocr_image(
            self,
            _source: object,
            *,
            page_index: int = 0,
            output_format: str = "markdown",
            table_mode: str = "gfm",
            pre_analysis: OCRResult | None = None,
        ) -> OCRResult:
            self.single_calls.append((page_index, pre_analysis))
            md = "| Col1 | Col2 |\n| --- | ---: |\n| VLM | $99.00 |"
            return OCRResult(
                text=md,
                raw_text=md,
                backend_used="ovis:fast-ocr-test",
                page_index=page_index,
                output_format=output_format,  # type: ignore[arg-type]
                table_mode=table_mode,  # type: ignore[arg-type]
                markdown_validation=validate_markdown(md, allow_html_tables=False),
                timings=OCRTimings(vit_ms=6.0, prefill_ms=3.0, decode_ms=10.0, total_ms=19.0),
            )

        def ocr_images(
            self,
            sources: list[object],
            *,
            page_indices: list[int],
            output_format: str = "markdown",
            table_mode: str = "gfm",
            pre_analyses: list[OCRResult | None] | None = None,
        ) -> list[OCRResult]:
            pres = list(pre_analyses) if pre_analyses is not None else [None] * len(sources)
            self.batch_calls.append((list(page_indices), pres))
            return [
                self.ocr_image(s, page_index=idx, output_format=output_format, table_mode=table_mode, pre_analysis=pre)
                for s, idx, pre in zip(sources, page_indices, pres)
            ]

        def close(self) -> None:
            self.closed = True

    class TaggedImg:
        def __init__(self, tag: str) -> None:
            self._tag = tag

    fake_vis = FakeVision()
    fake_ovis = FakeOvis()
    with AutoOCRPipeline(backend="auto", vision_ocr=fake_vis, ovis_engine=fake_ovis) as pipe:
        r_prose = pipe.ocr_image(TaggedImg("prose"), page_index=0)  # type: ignore[arg-type]
        assert r_prose.backend_used.startswith("apple-vision")
        assert len(fake_ovis.single_calls) == 0

        r_table = pipe.ocr_image(TaggedImg("table"), page_index=1)  # type: ignore[arg-type]
        assert r_table.backend_used == "ovis:fast-ocr-test"
        assert len(fake_ovis.single_calls) == 1
        assert fake_ovis.single_calls[0][1] is not None
        assert fake_ovis.single_calls[0][1].analysis is not None
        assert fake_ovis.single_calls[0][1].analysis.needs_vlm is True
        assert r_table.timings.apple_vision_ms == 4.0
        assert r_table.timings.total_ms == 23.0

        # Multi-batch `ocr_images` double-buffering across 2 batches
        batch_imgs = [TaggedImg("prose"), TaggedImg("table"), TaggedImg("table"), TaggedImg("prose")]
        batch_out = pipe.ocr_images(batch_imgs, batch_size=2)  # type: ignore[arg-type]
        assert [r.page_index for r in batch_out] == [0, 1, 2, 3]
        assert batch_out[0].backend_used.startswith("apple-vision")
        assert batch_out[1].backend_used == "ovis:fast-ocr-test"
        assert batch_out[2].backend_used == "ovis:fast-ocr-test"
        assert batch_out[3].backend_used.startswith("apple-vision")
        assert [idxs for idxs, _ in fake_ovis.batch_calls] == [[1], [2]]
    assert fake_ovis.closed is True

    # PDF routing: native digital page 0 emitted directly WITHOUT rasterizing; page 1 rasterized -> escalated
    pdf_path = tmp_path / "mixed_doc.pdf"
    _write_mixed_two_page_pdf(
        pdf_path,
        [
            "QUARTERLY FINANCIAL OVERVIEW",
            "Alpha Corp delivered steady operating performance across all regional divisions during the quarter.",
            "- Revenue rose across North America and Europe.",
        ],
    )

    from fast_ocr import pipeline as pipeline_mod

    real_render = pipeline_mod.render_pdf_pages
    render_calls: list[tuple[list[int] | None, int]] = []

    def spy_render(path: str | Path, *, dpi: int = 300, page_indices=None, include_native_text: bool = True):
        render_calls.append((list(page_indices) if page_indices is not None else None, dpi))
        return real_render(path, dpi=dpi, page_indices=page_indices, include_native_text=include_native_text)

    monkeypatch.setattr(pipeline_mod, "render_pdf_pages", spy_render)

    class PdfSplitVision:
        def recognize_image(self, _img: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            return _make_table_vision_result(page_index=page_index)

    fake_ovis_pdf = FakeOvis()
    with FastOCR(backend="auto", vision_ocr=PdfSplitVision(), ovis_engine=fake_ovis_pdf) as pdf_ocr:
        pdf_res = pdf_ocr.ocr_pdf(pdf_path, dpi=150)

    assert len(pdf_res) == 2
    assert render_calls == [([1], 150)]  # Page 0 was never rasterized!
    assert pdf_res[0].backend_used == "native"
    assert pdf_res[0].page_index == 0
    assert "# QUARTERLY FINANCIAL OVERVIEW" in pdf_res[0].text
    assert pdf_res[0].markdown_validation is not None and pdf_res[0].markdown_validation.valid is True
    assert pdf_res[1].backend_used == "ovis:fast-ocr-test"
    assert pdf_res[1].page_index == 1


def test_cli_help_verify_run_and_bench(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    capsys.readouterr()

    rc_verify = cli_main(["verify", "--json"])
    assert rc_verify == 0
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["ok"] is True

    # Prepare synthetic input image & monkeypatch FastOCR in `cli` so no hardware Vision/GPU runs
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 80), (255, 255, 255)).save(img_path)
    out_md = tmp_path / "out.md"

    class FakeCLIFastOCR:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> FakeCLIFastOCR:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ocr_images(self, sources: list[Path], **_kwargs: object) -> list[OCRResult]:
            return [_make_prose_result(page_index=i, text=f"# Page {i}\n\nSynthetic text.") for i, _ in enumerate(sources)]

        def ocr_pdf(self, _path: Path, **_kwargs: object) -> list[OCRResult]:
            return [_make_prose_result(page_index=0, text="PDF page.")]

    monkeypatch.setattr("fast_ocr.cli.FastOCR", FakeCLIFastOCR)

    rc_run = cli_main(
        [
            "run",
            str(img_path),
            "--backend",
            "apple-vision",
            "--format",
            "markdown",
            "--out",
            str(out_md),
            "--json",
        ]
    )
    assert rc_run == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["backend"] == "apple-vision"
    assert run_payload["page_count"] == 1
    assert run_payload["results"][0]["markdown_validation"]["valid"] is True
    assert out_md.exists() and "Synthetic text." in out_md.read_text(encoding="utf-8")

    rc_bench = cli_main(
        [
            "bench",
            str(img_path),
            "--backend",
            "auto",
            "--repeat",
            "2",
            "--json",
        ]
    )
    assert rc_bench == 0
    bench_payload = json.loads(capsys.readouterr().out)
    assert len(bench_payload["benchmarks"]) == 1
    assert bench_payload["benchmarks"][0]["pages"] == 2
    assert bench_payload["benchmarks"][0]["validation"]["valid"] == 2


def test_html_table_unclosed_cells_and_auto_orient_rotation_resolution() -> None:
    from fast_ocr.ovis_engine import FastOvisOCREngine, resolve_vision_rotation_cw
    import inspect

    # Unclosed <td>/<tr> HTML fragments must flush all cell values rather than silently dropping text
    malformed = (
        "<table><tr><th>Item<th>Amount"
        "<tr><td>Services<td>$144,233.02"
        "<tr><td>Equipment<td>$633,807.27</table>"
    )
    gfm = canonicalize_markdown(malformed, table_mode="gfm")
    val = validate_markdown(gfm)
    assert val.valid is True
    assert val.gfm_table_count == 1
    assert "$144,233.02" in gfm
    assert "$633,807.27" in gfm

    # FastOvisOCREngine defaults `loop_breaker=False` so repeated <td>/<tr> n-grams never corrupt tables
    sig = inspect.signature(FastOvisOCREngine.__init__)
    assert "loop_breaker" in sig.parameters
    assert sig.parameters["loop_breaker"].default is False

    # Pre-oriented AutoOCRPipeline results report rotation_applied_cw=90 with post-rotation analysis.inferred_rotation_cw=0
    rotated_pre = OCRResult(
        text="rotated",
        raw_text="rotated",
        backend_used="apple-vision:accurate:b3",
        rotation_applied_cw=90,
        analysis=PageAnalysis(
            width=3259,
            height=2032,
            line_count=100,
            char_count=1200,
            digit_ratio=0.4,
            currency_hits=40,
            mean_confidence=0.95,
            low_conf_line_ratio=0.05,
            median_line_height=0.01,
            inferred_rotation_cw=0,
            ink_crop_box_px=None,
            ink_area_ratio=0.6,
            estimated_columns=6,
            table_score=1.0,
            needs_vlm=True,
        ),
    )
    assert resolve_vision_rotation_cw(rotated_pre, auto_orient=True) == 90
    assert resolve_vision_rotation_cw(rotated_pre, auto_orient=False) == 0


def test_table_grammar_clone_safe_drafts_numeric_recovery_and_triage_dpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llguidance = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    from fast_ocr.numeric_recovery import (
        compute_missing_numeric_crop_box_px,
        extract_normalized_numbers,
        find_filtered_missing_numbers,
        find_recovered_numbers,
    )
    from fast_ocr.ovis_engine import _filter_table_safe_drafts

    # 1. DynamicTableRowGrammar.clone isolation + _filter_table_safe_drafts boundaries
    vocab: dict[str, int] = {"<|endoftext|>": 0}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llguidance.LLTokenizer(bt.to_str(), eos_token=0)
    gram = DynamicTableRowGrammar(ltok)
    vocab_sz = bt.get_vocab_size()
    decode_fn = lambda tids: bt.decode(list(tids), skip_special_tokens=False)

    # Inactive phase: stop right before header-closing `\n` that triggers lock + separator FF
    hdr_prefix = bt.encode("| ColA | ColB |").ids
    for tid in hdr_prefix:
        assert gram.advance_text_and_maybe_lock(bt.decode([tid]), tid) == []
    assert gram.is_active is False
    safe_hdr = _filter_table_safe_drafts(
        gram,
        bt.encode(" ").ids + bt.encode("\n").ids + bt.encode("more").ids,
        decode_fn=decode_fn,
        vocab_size=vocab_sz,
        stop_ids={0},
    )
    assert bt.decode(safe_hdr) == " "

    # Activate lock, verify clone isolation and active-table body vs illegal `\n` vs row-close FF `|`
    nl_id = bt.encode("\n").ids[0]
    ff_sep = gram.advance_text_and_maybe_lock("\n", nl_id)
    assert bt.decode(ff_sep) == "| --- | --- |\n|"
    assert gram.is_active is True

    for tid in bt.encode(" 12").ids:
        assert gram.advance_text_and_maybe_lock(bt.decode([tid]), tid) == []
    cloned = gram.clone()
    assert cloned.is_active is True and cloned.locked_cols == 2
    for tid in bt.encode(" | 34 |").ids:
        _ = cloned.advance_text_and_maybe_lock(bt.decode([tid]), tid)
    # Original grammar still expects remainder of row 0, unaffected by `cloned`
    assert gram.locked_cols == 2

    # Body tokens after ` 12` stay legal up to closing `|` (which triggers FF `\n`)
    drafts_body = bt.encode(".5 | 99 ").ids + bt.encode("|").ids
    safe_body = _filter_table_safe_drafts(gram, drafts_body, decode_fn=decode_fn, vocab_size=vocab_sz, stop_ids={0})
    assert bt.decode(safe_body) == ".5 | 99 "
    # Grammar-illegal `\n` mid-row is rejected at index 0
    assert _filter_table_safe_drafts(gram, [nl_id], decode_fn=decode_fn, vocab_size=vocab_sz, stop_ids={0}) == []

    # Pruned vocab cutoff (`old_to_new[tok] < 0`) halts draft prefix cleanly
    o2n = np.arange(vocab_sz, dtype=np.int32)
    tail_id = bt.encode("x").ids[0]
    o2n[tail_id] = -1
    assert (
        _filter_table_safe_drafts(
            gram,
            bt.encode(".5x").ids,
            decode_fn=decode_fn,
            vocab_size=vocab_sz,
            stop_ids={0},
            old_to_new=o2n,
        )
        == bt.encode(".5").ids
    )

    # 2. Digit-stem filtered numeric recovery + rotation-aware crop box
    assert extract_normalized_numbers("Total $1,250.00 (-15%)") == ["1250.00", "-15%"]
    vis_txt = "Year 2,016\nItem 5 19.29\nCode 5111\nTotal $4,250.75"
    ovis_txt = "| Year | Ref | Other |\n| --- | --- | --- |\n| 2.016 | 1929 | $4,250.75 |"
    missing = find_filtered_missing_numbers(vis_txt, ovis_txt)
    assert missing == ["5111"]  # Ignores `2,016` vs `2.016`, 1-digit `5`, and `19.29` inside `1929`
    assert find_recovered_numbers(missing, "| Code | 5111 |") == ["5111"]
    assert find_recovered_numbers(["99.99", "5111"], "| Code | 5111 |") == ["5111"]

    lines = [
        OCRLine(text="Code 5111", confidence=0.95, box=OCRBox(0.10, 0.20, 0.40, 0.25)),
        OCRLine(text="Clean 200", confidence=0.99, box=OCRBox(0.10, 0.80, 0.40, 0.85)),
    ]
    assert compute_missing_numeric_crop_box_px(lines, missing, (1000, 1000), rotation_cw=0, pad_frac=0.04) == (
        60,
        160,
        440,
        290,
    )
    assert compute_missing_numeric_crop_box_px(lines, missing, (1000, 1000), rotation_cw=90, pad_frac=0.04) == (
        710,
        60,
        840,
        440,
    )

    # 3. Two-stage `triage_dpi` PDF screening + CLI flags
    pdf_path = tmp_path / "scan_triage.pdf"
    _write_mixed_two_page_pdf(pdf_path, ["Header only", "Table body"])
    from fast_ocr import pipeline as pipeline_mod

    real_render = pipeline_mod.render_pdf_pages
    render_log: list[tuple[list[int] | None, int]] = []

    def spy_render(path: str | Path, *, dpi: int = 300, page_indices=None, include_native_text: bool = True):
        render_log.append((list(page_indices) if page_indices is not None else None, int(dpi)))
        return real_render(path, dpi=dpi, page_indices=page_indices, include_native_text=include_native_text)

    monkeypatch.setattr(pipeline_mod, "render_pdf_pages", spy_render)

    class TriageVision:
        def recognize_image(self, _img: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            return _make_prose_result(page_index=0) if page_index == 0 else _make_table_vision_result(page_index=1)

    class TriageOvis:
        def __init__(self, **_kwargs: object) -> None:
            self.seen: list[tuple[int, OCRResult | None]] = []

        def ocr_images(self, sources: list[object], *, page_indices: list[int], pre_analyses=None, **_kwargs: object):
            outs: list[OCRResult] = []
            for _src, idx, pre in zip(sources, page_indices, pre_analyses or [None] * len(sources)):
                self.seen.append((idx, pre))
                md = "| Item | Amount |\n| --- | ---: |\n| Esc | $5,111.00 |"
                outs.append(
                    OCRResult(
                        text=md,
                        raw_text=md,
                        backend_used="ovis:test",
                        page_index=idx,
                        markdown_validation=validate_markdown(md, allow_html_tables=False),
                        timings=OCRTimings(total_ms=12.0),
                    )
                )
            return outs

    ovis_mock = TriageOvis()
    with FastOCR(backend="auto", vision_ocr=TriageVision(), ovis_engine=ovis_mock, zoom_recover=True) as pipe:
        res_triage = pipe.ocr_pdf(pdf_path, dpi=300, triage_dpi=100, use_native_text=False)

    assert render_log == [(None, 100), ([1], 300)]
    assert [r.page_index for r in res_triage] == [0, 1]
    assert res_triage[0].backend_used.startswith("apple-vision") and res_triage[0].metadata["triage_dpi"] == 100
    assert res_triage[1].backend_used == "ovis:test" and res_triage[1].metadata["triage_dpi"] == 100
    assert len(ovis_mock.seen) == 1 and ovis_mock.seen[0][0] == 1 and ovis_mock.seen[0][1] is not None

    captured_cli: dict[str, object] = {}

    class SpyCLIFastOCR:
        def __init__(self, **kwargs: object) -> None:
            captured_cli["init"] = kwargs

        def __enter__(self) -> SpyCLIFastOCR:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ocr_pdf(self, _path: Path, **kwargs: object) -> list[OCRResult]:
            captured_cli["pdf_kw"] = kwargs
            return [_make_prose_result(page_index=0)]

    monkeypatch.setattr("fast_ocr.cli.FastOCR", SpyCLIFastOCR)
    rc = cli_main(["run", str(pdf_path), "--backend", "auto", "--triage-dpi", "100", "--zoom-recover", "--json"])
    assert rc == 0
    assert captured_cli["init"]["zoom_recover"] is True  # type: ignore[index]
    assert captured_cli["pdf_kw"]["triage_dpi"] == 100  # type: ignore[index]


def _write_custom_pdf(path: Path, page_specs: list[dict[str, object]]) -> None:
    """Create a multi-page synthetic PDF with optional image XObjects and custom page sizes."""
    n = len(page_specs)
    font_obj_num = 3 + 2 * n
    img_obj_num = font_obj_num + 1
    page_obj_nums = [3 + 2 * i for i in range(n)]
    kids = " ".join(f"{num} 0 R" for num in page_obj_nums)

    objs: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n} >>\nendobj\n".encode("latin1"),
    ]
    for idx, spec in enumerate(page_specs):
        page_num = 3 + 2 * idx
        content_num = page_num + 1
        w_pt = float(spec.get("width_pt", 612.0))  # type: ignore[arg-type]
        h_pt = float(spec.get("height_pt", 792.0))  # type: ignore[arg-type]
        lines = list(spec.get("lines", []))  # type: ignore[arg-type]
        with_image = bool(spec.get("with_image", False))

        ops: list[str] = []
        if with_image:
            ops.append("q 120 0 0 120 72 500 cm /Im1 Do Q")
        if lines:
            ops.extend(["BT", "/F1 12 Tf", "72 740 Td"])
            for l_idx, line in enumerate(lines):
                esc = str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                if l_idx > 0:
                    ops.append("0 -18 Td")
                ops.append(f"({esc}) Tj")
            ops.append("ET")
        stream_bytes = "\n".join(ops).encode("latin1")
        xobj_part = f" /XObject << /Im1 {img_obj_num} 0 R >>" if with_image else ""
        page_obj = (
            f"{page_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:g} {h_pt:g}] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >>{xobj_part} >> /Contents {content_num} 0 R >>\nendobj\n"
        ).encode("latin1")
        content_obj = (
            f"{content_num} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        objs.extend([page_obj, content_obj])

    objs.append(
        f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1")
    )
    img_payload = b"\xff\x00\xff"
    objs.append(
        (
            f"{img_obj_num} 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {len(img_payload)} >>\nstream\n"
        ).encode("latin1")
        + img_payload
        + b"\nendstream\nendobj\n"
    )

    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    buf.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
    path.write_bytes(bytes(buf))


def test_adaptive_pdf_dpi_boilerplate_control_garble_and_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_ocr import pipeline as pipeline_mod
    from fast_ocr.preprocess import RenderedPDFPage

    # 1. Adaptive DPI clamping
    assert compute_adaptive_render_dpi(612.0, 792.0, dpi=300) == 300
    poster_dpi = compute_adaptive_render_dpi(3600.0, 4800.0, dpi=300)
    assert 36 <= poster_dpi < 300
    assert (3600.0 * poster_dpi / 72.0) * (4800.0 * poster_dpi / 72.0) <= 20_000_000
    assert compute_adaptive_render_dpi(3600.0, 4800.0, dpi=300, max_pixels=None) == 300
    assert compute_adaptive_render_dpi(3600.0, 4800.0, dpi=300, max_pixels=100, min_dpi=36) == 36

    # 2. Running boilerplate strip & C0/C1/U+FFFD control garble detector
    raw_sample = "\n".join(
        [
            "VerDate Mar 15 2016 14:22 Jkt 000001 PO 00000 Frm 00001 Fmt 6601 Sfmt 6601",
            "Page 3 of 12",
            "42",
            "---",
            "ABC-000123",
            "2026-11-23",
            "PART II - EXHIBITS",
            "This page intentionally left blank",
        ]
    )
    assert strip_running_pdf_boilerplate(raw_sample) == "PART II - EXHIBITS\nThis page intentionally left blank"
    assert count_control_garble_chars("Clean\tline\r\nNext line.") == 0
    assert count_control_garble_chars("a\x00b\x0bc\x7fd\x9fe\ufffdf") == 5

    # 3. Native routing on boilerplate-only w/ image vs clean blank divider vs oversized raster page
    pdf_path = tmp_path / "routing_edge_cases.pdf"
    _write_custom_pdf(
        pdf_path,
        [
            {
                "lines": [
                    "VerDate Mar 15 2016 14:22 Jkt 000001 PO 00000 Frm 00001 Fmt 6601 Sfmt 6601",
                    "Page 1 of 3",
                    "ABC-000123",
                ],
                "with_image": True,
            },
            {
                "lines": ["PART II - EXHIBITS"],
                "with_image": False,
            },
            {
                "lines": [],
                "width_pt": 3600.0,
                "height_pt": 4800.0,
                "with_image": False,
            },
        ],
    )

    rendered_oversized = render_pdf_pages(pdf_path, page_indices=[2], dpi=300, max_pixels=10_000_000)
    assert len(rendered_oversized) == 1
    rp = rendered_oversized[0]
    assert rp.effective_dpi is not None and 36 < rp.effective_dpi < 300
    assert rp.image is not None and rp.image.width * rp.image.height <= 10_000_000

    class FakeVisionRouter:
        def recognize_image(self, _img: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            return _make_prose_result(page_index=page_index, text=f"Vision page {page_index}")

    class FakeOvisNever:
        def ocr_images(self, *_args: object, **_kwargs: object) -> list[OCRResult]:
            raise AssertionError("Ovis should not be invoked for Vision prose pages")

    with FastOCR(backend="auto", vision_ocr=FakeVisionRouter(), ovis_engine=FakeOvisNever()) as pipe:
        outs = pipe.ocr_pdf(pdf_path, dpi=150)

    # Page 0: >80 raw chars of pure boilerplate + 1 image -> NOT native (routed to Vision)
    # Page 1: short clean divider (<80 chars, 0 images) -> native fast path
    # Page 2: zero text -> rasterized to Vision
    assert [r.page_index for r in outs] == [0, 1, 2]
    assert outs[0].backend_used.startswith("apple-vision")
    assert outs[1].backend_used == "native"
    assert outs[1].metadata["image_count"] == 0
    assert outs[1].metadata["native_body_char_count"] == len("PARTII-EXHIBITS")
    assert outs[2].backend_used.startswith("apple-vision")

    # 4. Single control/replacement char disqualifies native fast path even when garbled ratio < 0.02
    long_prose = "clean word " * 25  # 275 chars -> 1 bad char is ratio 0.004 < 0.02
    monkeypatch.setattr(
        pipeline_mod,
        "extract_native_pdf_text",
        lambda _p, page_indices=None: [
            RenderedPDFPage(
                page_index=0,
                width_pt=612.0,
                height_pt=792.0,
                native_text=long_prose + "\ufffd",
                native_char_count=251,
                native_garbled_ratio=1.0 / 251.0,
                image=None,
                image_count=0,
                native_control_char_count=1,
                native_body_char_count=251,
            )
        ],
    )
    single_pdf = tmp_path / "single.pdf"
    _write_custom_pdf(single_pdf, [{"lines": ["Placeholder"], "with_image": False}])
    with FastOCR(backend="auto", vision_ocr=FakeVisionRouter(), ovis_engine=FakeOvisNever()) as pipe:
        ctrl_outs = pipe.ocr_pdf(single_pdf, dpi=100)
    assert len(ctrl_outs) == 1
    assert ctrl_outs[0].backend_used.startswith("apple-vision")


def test_apple_vision_provenance_and_cli_vision_pdf_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_engine(_rgb: bytes, _w: int, _h: int) -> list[tuple[str, float, list[float]]]:
        return [("Provenance test line.", 0.99, [0.1, 0.8, 0.5, 0.05])]

    fake_engine.revision = 3  # type: ignore[attr-defined]
    vis = AppleVisionOCR(engine=fake_engine, bands=1, language_correction=False)
    res = vis.recognize_image(Image.new("RGB", (200, 100), (255, 255, 255)), page_index=0)
    assert res.metadata["request_revision"] == 3
    assert res.metadata["language_correction"] is False
    if sys.platform == "darwin":
        assert isinstance(res.metadata.get("macos_version"), str) and res.metadata["macos_version"]
        assert isinstance(res.metadata.get("macos_build"), str) and res.metadata["macos_build"]

    # Fresh subprocess: verify AppleVisionOCR provenance never imports mlx
    sub_code = (
        "import sys\n"
        "from PIL import Image\n"
        "from fast_ocr.apple_vision import AppleVisionOCR\n"
        "def f(_b, _w, _h):\n"
        "    return [('Clean line.', 0.99, [0.1, 0.8, 0.4, 0.05])]\n"
        "f.revision = 3\n"
        "r = AppleVisionOCR(engine=f).recognize_image(Image.new('RGB', (120, 80), (255, 255, 255)))\n"
        "assert 'mlx' not in sys.modules, f'mlx imported: {list(sys.modules)}'\n"
        "assert r.metadata['request_revision'] == 3\n"
    )
    proc = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    # CLI flags `--no-language-correction` and `--max-pdf-pixels`
    pdf_path = tmp_path / "cli_flags.pdf"
    _write_custom_pdf(pdf_path, [{"lines": ["Test"], "with_image": False}])
    captured: dict[str, object] = {}

    class SpyCLIFastOCR:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def __enter__(self) -> SpyCLIFastOCR:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ocr_pdf(self, _path: Path, **kwargs: object) -> list[OCRResult]:
            captured["pdf_kw"] = kwargs
            return [_make_prose_result(page_index=0)]

    monkeypatch.setattr("fast_ocr.cli.FastOCR", SpyCLIFastOCR)
    rc1 = cli_main(
        [
            "run",
            str(pdf_path),
            "--backend",
            "apple-vision",
            "--no-language-correction",
            "--max-pdf-pixels",
            "5000000",
            "--json",
        ]
    )
    assert rc1 == 0
    assert captured["init"]["language_correction"] is False  # type: ignore[index]
    assert captured["pdf_kw"]["max_pixels"] == 5_000_000  # type: ignore[index]

    rc2 = cli_main(
        [
            "run",
            str(pdf_path),
            "--backend",
            "apple-vision",
            "--max-pdf-pixels",
            "0",
            "--json",
        ]
    )
    assert rc2 == 0
    assert captured["init"]["language_correction"] is None  # type: ignore[index]
    assert captured["pdf_kw"]["max_pixels"] is None  # type: ignore[index]


def test_numeric_recovery_hallucination_pruning_and_band_crops() -> None:
    """Single-digit substitutions (>= 3 digits) are pruned while genuine omissions and tight band crops stay intact."""
    from fast_ocr.numeric_recovery import (
        compute_missing_numeric_crop_box_px,
        compute_missing_numeric_crop_boxes_px,
        find_filtered_missing_numbers,
    )

    # FT[3]/FT[5]-style ANE 1-digit substitution hallucinations must be suppressed
    vision_halluc = "Total 445 and 310 with item 105 and 375 plus 223 at rate 37.8"
    cand_text = "Total 448 and 314 with item 102 and 575 plus 225 at rate 37.5"
    assert find_filtered_missing_numbers(vision_halluc, cand_text) == []

    # Genuine multi-digit omissions (Hamming >= 2, like FT[1] '1362' and FT[4] '1320') remain detected
    assert find_filtered_missing_numbers("Value 1362 and 1320", "Value 1345 and 1319") == ["1362", "1320"]

    # 2-digit numbers must NOT be suppressed by >=3-digit Hamming pruning
    assert find_filtered_missing_numbers("Count 42", "Count 43") == ["42"]

    # Isolated lone low-conf noise line at top must not stretch real bottom missing band into full-column crop
    lines = [
        OCRLine(text="99999", confidence=0.30, box=OCRBox(0.10, 0.02, 0.20, 0.05)),
        OCRLine(text="1320", confidence=0.95, box=OCRBox(0.70, 0.78, 0.85, 0.81)),
    ]
    box_single = compute_missing_numeric_crop_box_px(lines, ["99999", "1320"], (1000, 1000), pad_frac=0.04)
    assert box_single is not None
    assert box_single[1] >= 700
    assert (box_single[3] - box_single[1]) <= 200

    # Two distinct high-conf distant bands split into <=2 tight crops instead of giant union
    band_lines = [
        OCRLine(text="1362", confidence=0.98, box=OCRBox(0.70, 0.05, 0.85, 0.09)),
        OCRLine(text="1320", confidence=0.96, box=OCRBox(0.70, 0.85, 0.85, 0.89)),
    ]
    band_boxes = compute_missing_numeric_crop_boxes_px(band_lines, ["1362", "1320"], (1000, 1000), max_crops=2)
    assert len(band_boxes) == 2
    assert all((b[3] - b[1]) <= 200 for b in band_boxes)


def test_align_json_values_to_lines_provenance_and_rotation() -> None:
    """Leaf JSON alignment anchors array rows and respects clockwise coordinate rotation."""
    from fast_ocr.numeric_recovery import align_json_values_to_lines

    lines = [
        OCRLine(text="LATTE", confidence=0.99, box=OCRBox(0.15, 0.20, 0.40, 0.24)),
        OCRLine(text="1", confidence=0.98, box=OCRBox(0.05, 0.20, 0.09, 0.24)),
        OCRLine(text="15,000", confidence=0.99, box=OCRBox(0.75, 0.20, 0.92, 0.24)),
        OCRLine(text="MOCHA", confidence=0.99, box=OCRBox(0.15, 0.50, 0.42, 0.54)),
        OCRLine(text="1", confidence=0.98, box=OCRBox(0.05, 0.50, 0.09, 0.54)),
        OCRLine(text="22,000", confidence=0.99, box=OCRBox(0.75, 0.50, 0.92, 0.54)),
        OCRLine(text="TOTAL 37,000", confidence=0.99, box=OCRBox(0.45, 0.80, 0.92, 0.84)),
    ]
    raw_json = json.dumps(
        {
            "menu": [
                {"nm": "LATTE", "cnt": "1", "price": "15,000"},
                {"nm": "MOCHA", "cnt": "1", "price": "22,000"},
            ],
            "total": {"total_price": "37,000"},
        }
    )
    boxes = align_json_values_to_lines(raw_json, lines, image_size=(800, 1000), rotation_cw=0)
    assert boxes["menu[0].nm"] == [0.15, 0.2, 0.4, 0.24]
    assert boxes["menu[0].cnt"] == [0.05, 0.2, 0.09, 0.24]
    assert boxes["menu[1].nm"] == [0.15, 0.5, 0.42, 0.54]
    assert boxes["menu[1].cnt"] == [0.05, 0.5, 0.09, 0.54]
    assert boxes["total.total_price"] == [0.45, 0.8, 0.92, 0.84]

    boxes_90 = align_json_values_to_lines(raw_json, lines, image_size=(1000, 800), rotation_cw=90)
    exp = rotate_box_normalized((0.15, 0.20, 0.40, 0.24), 90)
    assert np.allclose(boxes_90["menu[0].nm"], list(exp), atol=1e-5)


def test_json_matcher_pld_prefilter_and_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_filter_json_safe_drafts` invariants and bit-identical greedy parity under JSON grammar + PLD."""
    from fast_ocr.ovis_engine import FastOvisOCREngine, _filter_json_safe_drafts
    from fast_ocr.preprocess import QwenVisionInputs

    class ToyJsonMatcher:
        """Minimal `llguidance`-compatible stateful matcher over an integer token vocab."""

        def __init__(self, target_seq: list[int], ff_after: dict[int, list[int]] | None = None) -> None:
            self.target_seq = list(target_seq)
            self.ff_after = ff_after or {}
            self.pos = 0
            self.stopped = False

        def deep_copy(self) -> ToyJsonMatcher:
            c = ToyJsonMatcher(self.target_seq, self.ff_after)
            c.pos = self.pos
            c.stopped = self.stopped
            return c

        def is_stopped(self) -> bool:
            return self.stopped

        def validate_tokens(self, tokens: list[int]) -> int:
            if self.stopped or self.pos >= len(self.target_seq):
                return 0
            return 1 if tokens and int(tokens[0]) == self.target_seq[self.pos] else 0

        def consume_token(self, tok: int) -> bool:
            if self.validate_tokens([tok]) == 0:
                self.stopped = True
                return False
            self.pos += 1
            if self.pos >= len(self.target_seq):
                self.stopped = True
            return True

        def consume_tokens(self, toks: list[int]) -> bool:
            for t in toks:
                if not self.consume_token(int(t)):
                    return False
            return True

        def compute_ff_tokens(self) -> list[int]:
            return list(self.ff_after.get(self.pos, []))

    class ToyJsonGram:
        def __init__(self, target_seq: list[int], ff_after: dict[int, list[int]], eos_id: int) -> None:
            self.target_seq = list(target_seq)
            self.ff_after = ff_after
            self.upfront_ff = [target_seq[0]]
            self.eos_token_id = eos_id
            self.empty_pruned_at: set[int] = set()

        def new_matcher(self) -> ToyJsonMatcher:
            m = ToyJsonMatcher(self.target_seq, self.ff_after)
            m.consume_tokens(self.upfront_ff)
            return m

        def has_any_allowed(self, matcher: ToyJsonMatcher) -> bool:
            return matcher.pos not in self.empty_pruned_at

        def pick_valid_token(
            self,
            matcher: ToyJsonMatcher,
            logits_1d: np.ndarray,
            _argmax_idx: int,
            *,
            fallback_to_full_vocab: bool = False,
            full_logits_1d: object = None,
            **_kwargs: object,
        ) -> tuple[int, bool]:
            _ = (logits_1d, fallback_to_full_vocab, full_logits_1d)
            if matcher.is_stopped() or matcher.pos >= len(matcher.target_seq):
                return self.eos_token_id, True
            return matcher.target_seq[matcher.pos], False

    # Sequence of tokens: upfront=10, step tokens [11, 12, 13, 14, 15, 16, 17, 18], with FF [14, 15] triggered after pos=4 (token 13)
    target = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    gram_unit = ToyJsonGram(target, ff_after={4: [14, 15]}, eos_id=2)
    matcher_u = gram_unit.new_matcher()  # pos == 1
    new_to_old = np.arange(320, dtype=np.int32)
    old_to_new = np.arange(320, dtype=np.int32)
    old_to_new[99] = -1  # simulate pruned-out token

    # Drafts [11, 12, 13, 14]: stops BEFORE 13 because consuming 13 triggers FF [14, 15]
    assert _filter_json_safe_drafts(
        matcher_u,
        gram_unit,
        [11, 12, 13, 14],
        vocab_size=320,
        stop_ids=frozenset({2}),
        new_to_old=new_to_old,
        old_to_new=old_to_new,
    ) == [11, 12]

    # Drafts hitting pruned-out token 99 or stopped state terminate cleanly
    assert _filter_json_safe_drafts(
        matcher_u,
        gram_unit,
        [99, 11],
        vocab_size=320,
        stop_ids=frozenset({2}),
        new_to_old=new_to_old,
        old_to_new=old_to_new,
    ) == []
    gram_unit.empty_pruned_at.add(2)
    assert _filter_json_safe_drafts(
        matcher_u,
        gram_unit,
        [11, 12],
        vocab_size=320,
        stop_ids=frozenset({2}),
        new_to_old=new_to_old,
        old_to_new=old_to_new,
    ) == []

    # End-to-end synthetic engine parity: speculative="off" vs "self" vs "vision+self"
    _, model = _make_tiny_qwen3_model(seed=19)
    eos_id = 2

    class ToyTokenizer:
        vocab_size = 320
        stop_token_ids = frozenset({eos_id})
        _tok_map = {
            10: '{"',
            11: "item",
            12: '":',
            13: '"',
            14: "LAT",
            15: "TE",
            16: '"',
            17: "}",
            18: "",
        }

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if "LATTE" in text:
                return [10, 11, 12, 13, 14, 15, 16, 17]
            return [20, 21, 22, 23]

        def decode(self, ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
            _ = skip_special_tokens
            return "".join(self._tok_map.get(int(i), "") for i in ids)

        def llguidance_tokenizer(self) -> object:
            return object()

    class StubVisionModel:
        spatial_merge_size = 2

        def __call__(self, _px: mx.array, _grid: list[tuple[int, int, int]]) -> mx.array:
            return mx.zeros((4, 64), dtype=mx.float32)

    eng = FastOvisOCREngine.__new__(FastOvisOCREngine)
    eng.model_dir = Path("toy-nuextract")
    eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    eng.output_format = "raw"
    eng.table_mode = "gfm"
    eng.strict_markdown_grammar = False
    eng.zoom_recover = False
    eng.crop_margins = False
    eng.crop_pad_px = 0
    eng.auto_orient = False
    eng.speculative = "auto"
    eng.draft_k = 4
    eng.prefill_chunk_size = 0
    eng.max_tokens = 32
    eng.loop_breaker = False
    eng.image_token_id = 5
    eng.vision_start_token_id = 3
    eng.vision_end_token_id = 4
    eng.video_token_id = 6
    eng.tokenizer = ToyTokenizer()  # type: ignore[assignment]
    eng.vocab_remap = None  # type: ignore[assignment]
    eng._new_to_old = np.arange(320, dtype=np.int32)
    eng._new_to_old_mx = None
    eng._old_to_new = None
    eng.vision_model = StubVisionModel()  # type: ignore[assignment]
    eng.lm = model
    eng._vision_prepass = None
    eng._prev_mlx_memory_limit = None
    eng._prev_mlx_cache_limit = None

    caches_seen: list[list[object]] = []
    orig_make_cache = model.make_cache

    def _spy_make_cache(**kwargs: object) -> list[object]:
        c = orig_make_cache(**kwargs)  # type: ignore[arg-type]
        caches_seen.append(c)
        return c

    monkeypatch.setattr(model, "make_cache", _spy_make_cache)

    def _build_prompt_ids(num_vis: int, _inst: str) -> list[int]:
        return [1, 3] + ([5] * num_vis) + [4, 7]

    eng._build_prompt_ids = _build_prompt_ids  # type: ignore[method-assign]
    monkeypatch.setattr(
        "fast_ocr.ovis_engine.build_qwen_vision_inputs",
        lambda _img: QwenVisionInputs(
            pixel_values=np.zeros((1, 4, 4, 3), dtype=np.float32),
            grid_thw=(1, 4, 4),
            resized_height=56,
            resized_width=56,
            num_visual_tokens=4,
        ),
    )

    seq = [10, 11, 12, 13, 14, 15, 16, 17]
    # Pos 4 triggers [14, 15] FF after consuming token 13; pos 8 stops cleanly at '}'.
    monkeypatch.setattr(
        "fast_ocr.ovis_engine.GrammarHelper",
        lambda *_a, **_kw: ToyJsonGram(seq, ff_after={4: [14, 15]}, eos_id=eos_id),
    )

    pre_vis = OCRResult(
        text="LATTE",
        raw_text="LATTE",
        backend_used="apple-vision",
        lines=(OCRLine(text="LATTE", confidence=0.99, box=OCRBox(0.2, 0.3, 0.5, 0.36)),),
    )
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    res_off = eng.ocr_image(img, schema='{"type": "object"}', speculative="off", pre_analysis=pre_vis)
    res_vis = eng.ocr_image(img, schema='{"type": "object"}', speculative="vision+self", pre_analysis=pre_vis)

    assert res_off.raw_text == '{"item":"LATTE"}'
    assert res_vis.raw_text == res_off.raw_text
    assert res_vis.timings.drafted_tokens > 0
    assert res_vis.timings.accepted_draft_tokens > 0
    assert res_vis.metadata["speculative_mode"] == "vision+self"
    assert res_vis.metadata["field_boxes"] == {"item": [0.2, 0.3, 0.5, 0.36]}
    assert caches_seen[0][-1].offset == caches_seen[1][-1].offset  # type: ignore[attr-defined]


def test_resolve_ocr_mlx_limits_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 24 GB M5 budget caps to 19.2 GB, explicit opt-outs work, and close() restores MLX limits."""
    from fast_ocr.ovis_engine import FastOvisOCREngine, resolve_ocr_mlx_limits

    mem_24, cache_24, budget_24 = resolve_ocr_mlx_limits(host_mem_gb=24.0, metal_working_set_gb=20.0)
    assert mem_24 == int(round(19.2 * (1024**3)))
    assert cache_24 == int(round(1.5 * (1024**3)))
    assert budget_24 is not None and budget_24.memory_limit_gb == 19.2

    mem_tight, _, _ = resolve_ocr_mlx_limits(host_mem_gb=24.0, metal_working_set_gb=16.0)
    assert mem_tight == int(round(16.0 * (1024**3)))

    mem_48, cache_48, _ = resolve_ocr_mlx_limits(host_mem_gb=48.0, metal_working_set_gb=36.0)
    assert mem_48 == int(round(24.0 * (1024**3)))
    assert cache_48 == int(round(2.0 * (1024**3)))

    mem_off, _, _ = resolve_ocr_mlx_limits(host_mem_gb=24.0, memory_limit_gb=0.0)
    assert mem_off is None

    # Verify close() restores saved MLX limits
    recorded: list[tuple[str, int]] = []
    monkeypatch.setattr("fast_ocr.ovis_engine.mx.set_memory_limit", lambda v: recorded.append(("mem", int(v))) or 999)
    monkeypatch.setattr("fast_ocr.ovis_engine.mx.set_cache_limit", lambda v: recorded.append(("cache", int(v))) or 888)
    monkeypatch.setattr("fast_ocr.ovis_engine.mx.clear_cache", lambda: recorded.append(("clear", 0)))

    eng = FastOvisOCREngine.__new__(FastOvisOCREngine)
    eng._prev_mlx_memory_limit = 999
    eng._prev_mlx_cache_limit = 888
    eng.close()
    assert ("mem", 999) in recorded
    assert ("cache", 888) in recorded
    assert ("clear", 0) in recorded
    assert eng._prev_mlx_memory_limit is None


def test_ocr_parquet_checkpoint_resume_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify sharded Parquet OCR with atomic manifest checkpoints, resume, CLI subcommand, and MLX isolation."""
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fast_ocr.pipeline import ocr_parquet

    # Build 3 synthetic PNG rows (`image_bytes` + `image_path`)
    img_bytes_list: list[bytes] = []
    for _i in range(2):
        buf = io.BytesIO()
        Image.new("RGB", (64, 32), (255, 255, 255)).save(buf, format="PNG")
        img_bytes_list.append(buf.getvalue())

    img_file = tmp_path / "doc3.png"
    Image.new("RGB", (64, 32), (255, 255, 255)).save(img_file)

    in_pq1 = tmp_path / "part0.parquet"
    in_pq2 = tmp_path / "part1.parquet"
    pq.write_table(
        pa.table({"doc_id": ["d0", "d1"], "image_bytes": img_bytes_list}),
        str(in_pq1),
    )
    pq.write_table(
        pa.table({"doc_id": ["d2"], "image_path": [str(img_file)]}),
        str(in_pq2),
    )

    calls = {"count": 0}
    fail_on_third = {"enabled": True}

    def fake_engine(_buf: bytes, _w: int, _h: int) -> list[tuple[str, float, list[float]]]:
        calls["count"] += 1
        if fail_on_third["enabled"] and calls["count"] == 3:
            raise RuntimeError("simulated mid-run crash on shard 1")
        return [(f"Page line {calls['count']}", 0.99, [0.1, 0.8, 0.5, 0.05])]

    out_dir = tmp_path / "ocr_out"
    pipe = AutoOCRPipeline(
        backend="apple-vision",
        vision_ocr=AppleVisionOCR(engine=fake_engine, bands=1),
    )

    with pytest.raises(RuntimeError, match="simulated mid-run crash"):
        ocr_parquet(
            str(tmp_path / "part*.parquet"),
            out_dir,
            shard_rows=2,
            resume=True,
            pipeline=pipe,
        )

    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "part-00000.parquet").is_file()
    assert calls["count"] == 3

    # Resume: shard 0 (2 rows) must be skipped cleanly; only row 2 (part1) runs.
    fail_on_third["enabled"] = False
    summary = ocr_parquet(
        str(tmp_path / "part*.parquet"),
        out_dir,
        shard_rows=2,
        resume=True,
        pipeline=pipe,
    )
    assert summary["skipped_shards"] == 1
    assert summary["skipped_rows"] == 2
    assert summary["completed_shards"] == 1
    assert summary["processed_rows"] == 1
    assert summary["total_rows"] == 3
    assert summary["valid_rows"] == 3
    assert calls["count"] == 4  # only 1 new call for row 2!

    t_out = pq.read_table(str(out_dir / "part-00000.parquet"))
    assert {"doc_id", "ocr_text", "backend_used", "ocr_valid", "total_ms", "metadata_json"} <= set(t_out.column_names)
    assert t_out["doc_id"].to_pylist() == ["d0", "d1"]
    assert all(t_out["ocr_valid"].to_pylist())

    # Verify CLI `fast-ocr parquet` wiring & `--memory-limit-gb`
    cli_calls: dict[str, object] = {}

    class SpyParquetFastOCR:
        def __init__(self, **kwargs: object) -> None:
            cli_calls["init"] = kwargs

        def __enter__(self) -> SpyParquetFastOCR:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def ocr_parquet(self, inp: str, outp: str, **kwargs: object) -> dict[str, object]:
            cli_calls["inp"] = inp
            cli_calls["outp"] = outp
            cli_calls["kw"] = kwargs
            return {
                "output_path": outp,
                "total_rows": 3,
                "processed_rows": 3,
                "skipped_rows": 0,
                "valid_rows": 3,
                "total_ms": 12.0,
            }

    monkeypatch.setattr("fast_ocr.cli.FastOCR", SpyParquetFastOCR)
    rc = cli_main(
        [
            "parquet",
            str(in_pq1),
            str(tmp_path / "out.parquet"),
            "--backend",
            "auto",
            "--memory-limit-gb",
            "19.2",
            "--shard-rows",
            "32",
            "--json",
        ]
    )
    assert rc == 0
    assert cli_calls["init"]["memory_limit_gb"] == 19.2  # type: ignore[index]
    assert cli_calls["kw"]["shard_rows"] == 32  # type: ignore[index]

    # Verify fresh subprocess using ocr_parquet(backend="apple-vision") never imports mlx
    sub_code = (
        "import io, sys, tempfile\n"
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "import pyarrow as pa\n"
        "import pyarrow.parquet as pq\n"
        "from fast_ocr import AutoOCRPipeline, AppleVisionOCR, ocr_parquet\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    p = Path(td)\n"
        "    buf = io.BytesIO()\n"
        "    Image.new('RGB', (48, 32), (255, 255, 255)).save(buf, format='PNG')\n"
        "    pq.write_table(pa.table({'image_bytes': [buf.getvalue()]}), str(p / 'in.parquet'))\n"
        "    vis = AppleVisionOCR(engine=lambda _b, _w, _h: [('Hello.', 0.99, [0.1, 0.8, 0.5, 0.05])], bands=1)\n"
        "    pipe = AutoOCRPipeline(backend='apple-vision', vision_ocr=vis)\n"
        "    s = ocr_parquet(p / 'in.parquet', p / 'out.parquet', pipeline=pipe)\n"
        "    assert s['valid_rows'] == 1\n"
        "assert 'mlx' not in sys.modules, f'mlx imported: {list(sys.modules)}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_catastrophic_loop_detector_and_vision_tiny_crop_upscale() -> None:
    from fast_ocr.ovis_engine import detect_catastrophic_token_loop

    # 1. Catches degenerate p=1 and p=2..4 loops while exempting normal repeated table markup (<40x)
    assert detect_catastrophic_token_loop([1, 2, 3, 4, 5, 6, 7, 8] * 3) is None
    assert detect_catastrophic_token_loop([99] + [2007] * 16) == (1, 16)
    assert detect_catastrophic_token_loop([10] + [32, 2007] * 16) == (2, 16)

    tok_map = {100: "<td>", 101: "</td>", 2007: " 2007"}
    dec = lambda ids: "".join(tok_map.get(int(t), "x") for t in ids)
    assert detect_catastrophic_token_loop([100, 101] * 20, decode_fn=dec) is None
    assert detect_catastrophic_token_loop([100, 101] * 40, decode_fn=dec) == (2, 40)
    assert detect_catastrophic_token_loop([2007] * 16, decode_fn=dec) == (1, 16)

    # 2. AutoOCRPipeline exposes loop_breaker=False and auto_loop_recover=True defaults
    pipe = FastOCR(backend="ovis", loop_breaker=True, auto_loop_recover=False)
    assert pipe.loop_breaker is True
    assert pipe.auto_loop_recover is False
    pipe_default = FastOCR(backend="ovis")
    assert pipe_default.loop_breaker is False
    assert pipe_default.auto_loop_recover is True

    # 3. AppleVisionOCR upscales tiny low-res crops while preserving original analysis dimensions and skipping full pages
    seen_dims: list[tuple[int, int]] = []

    def rec_engine(_rgb: bytes, w: int, h: int) -> list[tuple[str, float, list[float]]]:
        seen_dims.append((w, h))
        return [("Total $123.45", 0.99, [0.1, 0.5, 0.6, 0.1])]

    vis = AppleVisionOCR(engine=rec_engine, bands=1)
    r_tiny = vis.recognize_image(Image.new("RGB", (258, 61), (255, 255, 255)))
    assert seen_dims[-1] == (774, 183)
    assert r_tiny.metadata["vision_upscale"] == pytest.approx(3.0)
    assert r_tiny.analysis is not None and (r_tiny.analysis.width, r_tiny.analysis.height) == (258, 61)

    r_med = vis.recognize_image(Image.new("RGB", (339, 106), (255, 255, 255)))
    assert seen_dims[-1] == (960, 300)
    assert r_med.metadata["vision_upscale"] == pytest.approx(2.832, abs=1e-3)

    r_full = vis.recognize_image(Image.new("RGB", (1200, 1600), (255, 255, 255)))
    assert seen_dims[-1] == (1200, 1600)
    assert r_full.metadata["vision_upscale"] == 1.0

    vis_off = AppleVisionOCR(engine=rec_engine, bands=1, min_long_edge=0)
    r_off = vis_off.recognize_image(Image.new("RGB", (258, 61), (255, 255, 255)))
    assert seen_dims[-1] == (258, 61)
    assert r_off.metadata["vision_upscale"] == 1.0


def test_b_gt_8_defaults_parallel_ane_gate_and_ovis_batch_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect
    import numpy as np
    from fast_ocr.ovis_engine import FastOvisOCREngine
    from fast_ocr.pipeline import run_vision_batch_parallel
    from fast_ocr.types import OCRTimings

    assert inspect.signature(FastOvisOCREngine.ocr_images).parameters["batch_size"].default == 16
    assert inspect.signature(FastOCR.ocr_images).parameters["batch_size"].default == 16
    assert inspect.signature(FastOCR.ocr_pdf).parameters["batch_size"].default == 16

    # 1. Parallel ANE helper skips small batches (< 4), tiny total pixels (< 2.0 MP), and injected mock engines,
    #    while still allowing an already-initialized `_NativeVisionRunner` instance.
    from fast_ocr.apple_vision import _NativeVisionRunner

    small_im = Image.new("RGB", (200, 200), "white")
    real_vis_with_mock_engine = AppleVisionOCR(
        engine=lambda _b, _w, _h: [("ok", 0.99, [0.1, 0.5, 0.4, 0.1])],
        bands=1,
    )
    assert run_vision_batch_parallel(real_vis_with_mock_engine, [small_im] * 4, [0, 1, 2, 3], output_format="markdown") is None

    big_ims = [Image.new("RGB", (1000, 800), "white") for _ in range(4)]  # 3.2 MP >= 2.0 MP
    assert run_vision_batch_parallel(real_vis_with_mock_engine, big_ims, [0, 1, 2, 3], output_format="markdown") is None

    pool_invoked: list[int] = []

    class _FakePool:
        def __init__(self, max_workers: int = 1) -> None:
            pool_invoked.append(int(max_workers))

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def map(self, _fn: Any, tasks: list[Any]) -> list[list[OCRResult]]:
            return [
                [
                    OCRResult(text=f"N {idx}", raw_text=f"N {idx}", backend_used="apple-vision:native", page_index=int(idx))
                    for idx, _p in task[-1]
                ]
                for task in tasks
            ]

    vis_native_init = AppleVisionOCR(bands=1)
    vis_native_init._engine = object.__new__(_NativeVisionRunner)  # type: ignore[assignment]
    with monkeypatch.context() as mp_pool:
        mp_pool.setattr("fast_ocr.pipeline.ProcessPoolExecutor", _FakePool)
        nat_res = run_vision_batch_parallel(vis_native_init, big_ims, [0, 1, 2, 3], output_format="markdown")
    assert pool_invoked == [4] and nat_res is not None
    assert [r.text for r in nat_res] == ["N 0", "N 1", "N 2", "N 3"]
    seen_parallel: dict[str, Any] = {}

    def _fake_par(vision: Any, sources: Any, page_indices: Any, *, output_format: str, workers: int = 4):
        seen_parallel["count"] = len(sources)
        seen_parallel["workers"] = workers
        seen_parallel["fmt"] = output_format
        return [
            OCRResult(
                text=f"Page {idx}",
                raw_text=f"Page {idx}",
                backend_used="apple-vision:par",
                page_index=int(idx),
                output_format=output_format,  # type: ignore[arg-type]
                timings=OCRTimings(total_ms=1.0),
            )
            for idx in page_indices
        ]

    monkeypatch.setattr("fast_ocr.pipeline.run_vision_batch_parallel", _fake_par)
    with FastOCR(backend="apple-vision", vision_ocr=real_vis_with_mock_engine) as pipe:
        par_outs = pipe.ocr_images(big_ims)
        assert [r.text for r in par_outs] == ["Page 0", "Page 1", "Page 2", "Page 3"]
    assert seen_parallel == {"count": 4, "workers": 4, "fmt": "markdown"}

    # 2. FastOCR(backend="ovis") forwards batch_size=16 to FastOvisOCREngine.ocr_images
    ovis_calls: list[dict[str, Any]] = []

    class _SpyOvis:
        def ocr_images(self, sources: Any, **kwargs: Any) -> list[OCRResult]:
            ovis_calls.append({"n": len(sources), **kwargs})
            return [
                OCRResult(
                    text=f"Ovis {idx}",
                    raw_text=f"Ovis {idx}",
                    backend_used="ovis:spy",
                    page_index=int(idx),
                    timings=OCRTimings(total_ms=2.0),
                )
                for idx in kwargs.get("page_indices", range(len(sources)))
            ]

    with FastOCR(backend="ovis", ovis_engine=_SpyOvis()) as pipe:
        res_ovis = pipe.ocr_images([small_im, small_im, small_im])
    assert len(res_ovis) == 3
    assert len(ovis_calls) == 1 and ovis_calls[0]["batch_size"] == 16 and ovis_calls[0]["n"] == 3

    # 3. Real FastOvisOCREngine.ocr_image caps kv_step <= 512 and FastOvisOCREngine.ocr_images delegates B>=2 to batched_multimodal_generate
    import mlx.core as mx
    from tokenizers import Tokenizer as HfTokenizer, AddedToken
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    from fast_qwen3.multimodal_batch import BatchedMMItemResult

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
    tok_p = tmp_path / "tok_ovis.json"
    bpe.save(str(tok_p))
    ftok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))

    kv_steps_seen: list[int] = []

    class _StubLM:
        vocab_n = ftok.vocab_size
        has_full_vocab_fallback = False

        class cfg:
            rot_dims = 64
            rope_theta = 1000000.0
            mrope_section = (11, 11, 10)

        def make_cache(self, *, kv_step: int = 256, kv_i8: bool | str = False):
            kv_steps_seen.append(int(kv_step))
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

        class cfg:
            patch_size = 16
            temporal_patch_size = 2
            spatial_merge_size = 2

        def __call__(self, px: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
            n_vis = sum((t * h * w) // 4 for t, h, w in grids)
            return mx.zeros((n_vis, 16), dtype=mx.bfloat16)

    stub_eng = object.__new__(FastOvisOCREngine)
    stub_eng.model_dir = tmp_path
    stub_eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    stub_eng.output_format = "markdown"
    stub_eng.table_mode = "gfm"
    stub_eng.strict_markdown_grammar = False
    stub_eng.zoom_recover = False
    stub_eng.crop_margins = False
    stub_eng.crop_pad_px = 0
    stub_eng.auto_orient = False
    stub_eng.speculative = "off"
    stub_eng.draft_k = 4
    stub_eng.prefill_chunk_size = 1024
    stub_eng.max_tokens = 4096
    stub_eng.loop_breaker = False
    stub_eng.auto_loop_recover = False
    stub_eng.image_token_id = ftok.backend_tokenizer.token_to_id("<|image_pad|>")
    stub_eng.tokenizer = ftok
    stub_eng.vocab_remap = None
    stub_eng.vision_model = _StubVision()
    stub_eng.lm = _StubLM()
    stub_eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub_eng._new_to_old_mx = None
    stub_eng._old_to_new = None
    stub_eng._vision_prepass = None

    stub_eng.ocr_image(Image.new("RGB", (64, 64), "white"), max_tokens=4096)
    assert kv_steps_seen and kv_steps_seen[-1] == 512

    batched_calls: dict[str, Any] = {}

    def _fake_batched_mm(**kw: Any):
        batched_calls["n"] = len(kw["images_per_item"])
        batched_calls["batch_size"] = kw["batch_size"]
        batched_calls["kv_step"] = kw["kv_step"]
        ids_hi = ftok.encode("Table text", add_special_tokens=False)
        return [
            BatchedMMItemResult(
                generated_ids=ids_hi,
                prompt_tokens=30,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=2.0,
                decode_ms=3.0,
                image_grids=[(1, 4, 4)],
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1, "compactions": 0, "vit_ms": 1.0, "prefill_ms": 2.0, "decode_ms": 3.0}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _fake_batched_mm)
    bat_res = stub_eng.ocr_images([Image.new("RGB", (64, 64), "white")] * 3, batch_size=16)
    assert batched_calls == {"n": 3, "batch_size": 16, "kv_step": 512}
    assert len(bat_res) == 3 and all("Table text" in r.text for r in bat_res)
    assert all(r.metadata["speculative_mode"] == "batched" for r in bat_res)


def test_native_pdf_layout_gate_merges_inline_runs_and_escalates_multicol_tables(tmp_path: Path) -> None:
    from fast_ocr.apple_vision import analyze_page_from_lines
    from fast_ocr.preprocess import extract_native_pdf_lines, extract_native_pdf_text

    def _build_positioned_pdf(out_pdf: Path, pages_ops: list[list[str]]) -> None:
        objs: list[bytes] = [
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            b"",
        ]
        kids: list[str] = []
        font_num = 3 + len(pages_ops) * 2
        for idx, ops in enumerate(pages_ops):
            pg_num = 3 + idx * 2
            ct_num = pg_num + 1
            kids.append(f"{pg_num} 0 R")
            stream = "\n".join(ops).encode("latin1")
            objs.append(
                f"{pg_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {ct_num} 0 R >>\nendobj\n".encode("latin1")
            )
            objs.append(
                f"{ct_num} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin1") + stream + b"\nendstream\nendobj\n"
            )
        objs[1] = f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages_ops)} >>\nendobj\n".encode("latin1")
        objs.append(f"{font_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1"))
        buf = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = []
        for obj in objs:
            offsets.append(len(buf))
            buf.extend(obj)
        xref_pos = len(buf)
        buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin1"))
        for off in offsets:
            buf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
        buf.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
        out_pdf.write_bytes(bytes(buf))

    # Page 0: single-column court opinion prose with split inline runs at the same baseline (must stay native)
    p0_ops: list[str] = []
    for row_i in range(10):
        y = 720 - row_i * 20
        p0_ops.extend(
            [
                f"BT /F1 11 Tf 72 {y} Td (Court opinion ) Tj ET",
                f"BT /F1 11 Tf 143 {y} Td (section {row_i} continuing cleanly across column center.) Tj ET",
            ]
        )

    # Page 1: clean 3-column table with >80 clean chars and 0 images (must NOT bypass to Tier-0)
    p1_ops: list[str] = []
    for row_i in range(8):
        y = 720 - row_i * 22
        p1_ops.extend(
            [
                f"BT /F1 10 Tf 40 {y} Td (Budget Row {row_i}) Tj ET",
                f"BT /F1 10 Tf 230 {y} Td (FY26 {1000 + row_i}.00) Tj ET",
                f"BT /F1 10 Tf 420 {y} Td (FY27 {2000 + row_i}.00) Tj ET",
            ]
        )

    pdf_p = tmp_path / "native_layout_gate.pdf"
    _build_positioned_pdf(pdf_p, [p0_ops, p1_ops])

    pages = extract_native_pdf_text(pdf_p)
    assert len(pages) == 2 and pages[0].lines and pages[1].lines
    pa0 = analyze_page_from_lines(pages[0].lines, None, 612, 792)
    pa1 = analyze_page_from_lines(pages[1].lines, None, 612, 792)
    assert pa0.estimated_columns == 1 and pa0.needs_vlm is False
    assert pa1.estimated_columns >= 3 and pa1.needs_vlm is True

    # Also verify graceful empty return for stub textpages without count_rects
    assert extract_native_pdf_lines(object(), 612.0, 792.0) == ()

    class FakeVisionTablePrepass:
        def recognize_image(self, _img: object, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            from dataclasses import replace as _dc_replace

            base = _make_prose_result(page_index=page_index)
            assert base.analysis is not None
            return _dc_replace(
                base,
                analysis=_dc_replace(base.analysis, estimated_columns=3, table_score=1.0, needs_vlm=True),
            )

    seen_hints: list[str | None] = []

    class FakeOvisTableRouter:
        def ocr_images(self, imgs: list[object], **kwargs: Any) -> list[OCRResult]:
            hints = kwargs.get("native_pdf_texts") or []
            seen_hints.extend(hints)
            return [
                OCRResult(
                    text="| Line | FY26 | FY27 |\n| --- | --- | --- |\n| Item 0 | $1000.00 | $2000.00 |",
                    raw_text="| Line |",
                    backend_used="ovis:stub",
                    page_index=int((kwargs.get("page_indices") or [0])[i]),
                    finish_reason="stop",
                    timings=OCRTimings(generated_tokens=16, decode_tps=200.0, decode_ms=80.0),
                )
                for i, _ in enumerate(imgs)
            ]

    with FastOCR(backend="auto", vision_ocr=FakeVisionTablePrepass(), ovis_engine=FakeOvisTableRouter()) as pipe:
        outs = pipe.ocr_pdf(pdf_p, dpi=100)

    assert [r.page_index for r in outs] == [0, 1]
    assert outs[0].backend_used == "native"
    assert outs[0].metadata["native_estimated_columns"] == 1
    assert outs[1].backend_used == "ovis:stub"
    assert len(seen_hints) == 1 and seen_hints[0] and "Budget Row 0" in seen_hints[0]


def test_ovis_engine_ocr_images_per_item_schemas_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    import numpy as np
    from tokenizers import AddedToken, Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_ocr.ovis_engine import FastOvisOCREngine
    from fast_qwen3.multimodal_batch import BatchedMMItemResult
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    from fast_shared.grammar import normalize_json_schema

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
    tok_p = tmp_path / "tok_ovis_schemas.json"
    bpe.save(str(tok_p))
    ftok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))

    built_schemas: list[str] = []

    class _FakeGrammarHelper:
        def __init__(self, _ltok: Any, schema_norm: str, new_to_old: Any = None) -> None:
            self.schema = str(schema_norm)
            built_schemas.append(self.schema)

    monkeypatch.setattr("fast_ocr.ovis_engine.GrammarHelper", _FakeGrammarHelper)
    monkeypatch.setattr("fast_ocr.ovis_engine.normalize_json_schema", normalize_json_schema)

    stub_eng = FastOvisOCREngine.__new__(FastOvisOCREngine)
    stub_eng.model_dir = tmp_path
    stub_eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [1, 1, 1]}}}
    stub_eng.output_format = "markdown"
    stub_eng.table_mode = "gfm"
    stub_eng.strict_markdown_grammar = True
    stub_eng.zoom_recover = False
    stub_eng.crop_margins = False
    stub_eng.crop_pad_px = 0
    stub_eng.auto_orient = False
    stub_eng.adaptive_resolution = False
    stub_eng.deskew = "off"
    stub_eng.speculative = "off"
    stub_eng.draft_k = 4
    stub_eng.prefill_chunk_size = 512
    stub_eng.max_tokens = 256
    stub_eng.loop_breaker = False
    stub_eng.auto_loop_recover = False
    stub_eng.kv_i8 = "bf16"
    stub_eng.gdn_state_mode = "fp32"
    stub_eng.drop_white_patches = False
    stub_eng.image_token_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))
    stub_eng.vision_model = SimpleNamespace()
    stub_eng.lm = SimpleNamespace(cfg=SimpleNamespace(head_dim=256), vocab_n=ftok.vocab_size)
    stub_eng.tokenizer = ftok
    stub_eng.vocab_remap = None
    stub_eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub_eng._new_to_old_mx = None
    stub_eng._old_to_new = None
    stub_eng._vision_prepass = None

    seen_mm: dict[str, Any] = {}
    ans_a = ftok.encode('{"amount":"12.50"}', add_special_tokens=False)
    ans_txt = ftok.encode("Table text", add_special_tokens=False)

    def _fake_batched_mm(**kw: Any):
        seen_mm["grammars"] = list(kw["grammars"])
        seen_mm["table_grammars"] = list(kw["table_grammars"]) if kw["table_grammars"] is not None else None
        return [
            BatchedMMItemResult(
                generated_ids=ans_a if seen_mm["grammars"][idx_m] is not None else ans_txt,
                prompt_tokens=12,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=1.0,
                image_grids=[(1, 2, 2)],
            )
            for idx_m in range(len(kw["images_per_item"]))
        ], {"waves": 1, "compactions": 0}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _fake_batched_mm)

    schema_a = {"type": "object", "properties": {"amount": {"type": "string"}}, "required": ["amount"]}
    schema_b = {"type": "object", "properties": {"vendor": {"type": "string"}}, "required": ["vendor"]}
    imgs = [Image.new("RGB", (32, 32), "white")] * 3
    outs = stub_eng.ocr_images(
        imgs,
        prompts=["extract a", "extract b", "read page"],
        schemas=[dict(schema_a), schema_a, None],
        batch_size=4,
    )
    assert len(outs) == 3
    # Identical normalized schemas share one compiled GrammarHelper; None leaves slot open for table_grammars
    assert len(built_schemas) == 1
    assert seen_mm["grammars"][0] is seen_mm["grammars"][1]
    assert seen_mm["grammars"][2] is None
    assert seen_mm["table_grammars"] is not None
    assert seen_mm["table_grammars"][0] is None and seen_mm["table_grammars"][1] is None and seen_mm["table_grammars"][2] is not None

    outs_diff = stub_eng.ocr_images(
        imgs[:2],
        prompts=["extract a", "extract b"],
        schemas=[schema_a, schema_b],
        batch_size=4,
    )
    assert len(outs_diff) == 2
    assert seen_mm["grammars"][0] is not seen_mm["grammars"][1]
    assert seen_mm["table_grammars"] is None

    # B=1 fallback forwards per-item schema into ocr_image
    seen_b1_schemas: list[Any] = []
    monkeypatch.setattr(
        stub_eng,
        "ocr_image",
        lambda _im, **kw: (seen_b1_schemas.append(kw.get("schema")) or _make_prose_result()),
    )
    _ = stub_eng.ocr_images(imgs[:2], schemas=[schema_a, schema_b], batch_size=1)
    assert seen_b1_schemas == [schema_a, schema_b]


def test_triage_dpi_refreshes_only_rotated_or_low_conf_escalations(tmp_path: Path) -> None:
    from fast_ocr.types import OCRBox, OCRLine, OCRResult, OCRTimings, PageAnalysis

    pdf_p = tmp_path / "triage_mixed.pdf"
    p0 = Image.new("RGB", (600, 800), "white")
    p1 = Image.new("RGB", (450, 100), "white")
    p0.save(pdf_p, "PDF", resolution=180.0, save_all=True, append_images=[p1])

    vision_calls: list[tuple[int, tuple[int, int]]] = []

    class _MockVision:
        def recognize_image(self, src: Image.Image, *, page_index: int = 0, output_format: str = "markdown") -> OCRResult:
            vision_calls.append((int(page_index), src.size))
            is_low_p1 = (page_index == 1 and min(src.size) < 90)
            return OCRResult(
                text="raw",
                raw_text="raw",
                backend_used="apple-vision:accurate",
                output_format=output_format,  # type: ignore[arg-type]
                page_index=int(page_index),
                lines=(OCRLine("cell", 0.99 if not is_low_p1 else 0.55, OCRBox(0.1, 0.1, 0.4, 0.2)),),
                analysis=PageAnalysis(
                    width=src.width,
                    height=src.height,
                    line_count=10,
                    char_count=80,
                    digit_ratio=0.3,
                    currency_hits=1,
                    mean_confidence=0.99 if not is_low_p1 else 0.65,
                    low_conf_line_ratio=0.0 if not is_low_p1 else 0.4,
                    median_line_height=0.02,
                    inferred_rotation_cw=(90 if is_low_p1 else 0),
                    ink_crop_box_px=None,
                    ink_area_ratio=1.0,
                    estimated_columns=2,
                    table_score=0.6,
                    needs_vlm=True,
                    escalation_reasons=(("rotated_page", "low_vision_confidence") if is_low_p1 else ("financial_table",)),
                ),
                timings=OCRTimings(apple_vision_ms=3.0, total_ms=3.0),
            )

    seen_pre: list[tuple[int, int]] = []

    class _MockOvis:
        def ocr_images(self, sources: list[Image.Image], **kw: Any) -> list[OCRResult]:
            for pre in kw["pre_analyses"]:
                seen_pre.append((int(pre.page_index), int(pre.analysis.inferred_rotation_cw)))
            return [
                OCRResult(
                    text=f"pg{pre.page_index}",
                    raw_text=f"pg{pre.page_index}",
                    backend_used="ovis:stub",
                    page_index=int(pre.page_index),
                    timings=OCRTimings(decode_ms=2.0, total_ms=3.0, generated_tokens=2),
                )
                for pre in kw["pre_analyses"]
            ]

    with FastOCR(backend="auto", vision_ocr=_MockVision(), ovis_engine=_MockOvis()) as pipe:
        outs = pipe.ocr_pdf(pdf_p, dpi=180, triage_dpi=100, batch_size=4)

    assert len(outs) == 2
    # Page 0 stays at single low-DPI pass; Page 1 gets refreshed at 180 DPI so Ovis receives rot=0 for both.
    assert [idx for idx, _sz in vision_calls] == [0, 1, 1]
    assert seen_pre == [(0, 0), (1, 0)]


