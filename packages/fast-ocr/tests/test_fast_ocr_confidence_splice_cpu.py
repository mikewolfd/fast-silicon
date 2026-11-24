"""CPU unit tests for Stream 4 (L-8): logprob confidence telemetry, opt-in escalation, and hybrid digital-PDF splicing."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

from fast_ocr.types import (
    OCRBox,
    OCRLine,
    OCRResult,
    OCRTimings,
    PageAnalysis,
    summarize_logprob_confidence,
)


def _write_positioned_pdf(path: Path, pages: list[list[tuple[float, float, str, int]]]) -> None:
    """Write a minimal uncompressed PDF placing Helvetica text at exact `(x_pt, y_pt)` coordinates."""
    n = len(pages)
    font_obj = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {n} >>\nendobj\n".encode("latin1"),
    ]
    for idx, items in enumerate(pages):
        pnum = 3 + 2 * idx
        cnum = pnum + 1
        ops: list[str] = []
        for x_pt, y_pt, txt, fsize in items:
            esc = str(txt).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"BT /F1 {int(fsize)} Tf 1 0 0 1 {x_pt:g} {y_pt:g} Tm ({esc}) Tj ET")
        stream = "\n".join(ops).encode("latin1")
        objs.append(
            f"{pnum} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {cnum} 0 R >>\nendobj\n".encode("latin1")
        )
        objs.append(
            f"{cnum} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin1") + stream + b"\nendstream\nendobj\n"
        )
    objs.append(f"{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1"))

    buf = bytearray(b"%PDF-1.4\n")
    offs: list[int] = []
    for obj in objs:
        offs.append(len(buf))
        buf.extend(obj)
    xref = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offs:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    buf.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin1"))
    path.write_bytes(bytes(buf))


def test_logprob_confidence_summary_and_step_kernel() -> None:
    from fast_ocr.ovis_engine import _StepConfidenceSession, _StepConfidenceWorker, compute_step_logprob_margin

    empty = summarize_logprob_confidence([])
    assert empty.n_tokens == 0
    assert empty.mean_logprob == 0.0
    assert empty.low_conf_frac == 0.0

    lps = [-0.05] * 10 + [-1.5, -1.5, -1.2, -0.8]
    margins = [8.0] * 10 + [0.4, 0.3, 0.6, 0.9]
    summ = summarize_logprob_confidence(lps, margins=margins, window_size=4)
    assert summ.n_tokens == 14
    assert summ.min_window_logprob == pytest.approx((-1.5 - 1.5 - 1.2 - 0.8) / 4.0, abs=1e-6)
    assert summ.low_conf_frac == pytest.approx(4.0 / 14.0, abs=1e-6)
    assert summ.min_window_margin == pytest.approx((0.4 + 0.3 + 0.6 + 0.9) / 4.0, abs=1e-6)
    assert summ.low_margin_frac == pytest.approx(4.0 / 14.0, abs=1e-6)
    d = summ.to_dict()
    assert d["n_tokens"] == 14
    assert d["window_size"] == 4

    # Verify Accelerate vDSP path (N >= 64) and numpy fallback match float64 reference logprob + margin
    rng = np.random.default_rng(7)
    for vocab_n, idx in ((32, 5), (140_307, 12345)):
        logits = (rng.standard_normal(vocab_n).astype(np.float32) * 2.5)
        logits[idx] = 15.0
        ref = logits.astype(np.float64)
        mx_r = float(np.max(ref))
        ref_lse = mx_r + math.log(float(np.sum(np.exp(ref - mx_r))))
        ref_lp = float(ref[idx]) - ref_lse
        ref_margin = float(ref[idx] - np.max(np.delete(ref, idx)))
        lp, margin = compute_step_logprob_margin(logits, idx)
        assert lp == pytest.approx(ref_lp, abs=1e-4)
        assert margin == pytest.approx(ref_margin, abs=1e-4)

    # Async worker session matches direct summary
    worker = _StepConfidenceWorker()
    try:
        sess = _StepConfidenceSession(worker)
        logits_a = np.full(128, -5.0, dtype=np.float32)
        logits_a[10] = 6.0
        sess.submit(logits_a, 10)
        logits_b = np.full(128, -1.0, dtype=np.float32)
        logits_b[20] = 0.0
        sess.submit(logits_b, 20)
        res = sess.finish()
        assert res.n_tokens == 2
        assert res.mean_margin is not None and res.mean_margin > 5.0
    finally:
        worker.close()


def test_ovis_engine_confidence_plumbing_b1_and_batched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx
    from tokenizers import AddedToken, Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_qwen3.multimodal_batch import BatchedMMItemResult
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    from fast_ocr.ovis_engine import FastOvisOCREngine

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
    tok_p = tmp_path / "tok_conf.json"
    bpe.save(str(tok_p))
    ftok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))
    tok_hello = ftok.encode("ok", add_special_tokens=False)

    class _StubLM:
        vocab_n = ftok.vocab_size
        has_full_vocab_fallback = False

        class cfg:
            rot_dims = 64
            rope_theta = 1000000.0
            mrope_section = (11, 11, 10)

        def __init__(self) -> None:
            self.step = 0

        def make_cache(self, *, kv_step: int = 256, kv_i8: bool | str = False):
            return []

        def embed_tokens(self, x: mx.array) -> mx.array:
            return mx.zeros((int(x.shape[0]), int(x.shape[1]), 16), dtype=mx.bfloat16)

        def forward_hidden(self, _ids: Any = None, _cache: Any = None, *, inputs_embeds: mx.array | None = None, **_kw: Any) -> mx.array:
            b = int(inputs_embeds.shape[0]) if inputs_embeds is not None else 1
            s = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 1
            return mx.zeros((b, s, 16), dtype=mx.bfloat16)

        def project_logits(self, h: mx.array, **_kw: Any) -> mx.array:
            out = np.full((int(h.shape[0]), int(h.shape[1]), ftok.vocab_size), -12.0, dtype=np.float32)
            idx = tok_hello[self.step] if self.step < len(tok_hello) else ftok.eos_token_id
            out[:, :, idx] = 8.0
            self.step += 1
            return mx.array(out)

    class _StubVision:
        spatial_merge_size = 2

        def __call__(self, _px: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
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
    eng.adaptive_resolution = False
    eng.deskew = False
    eng.speculative = "off"
    eng.draft_k = 4
    eng.prefill_chunk_size = 512
    eng.max_tokens = 32
    eng.loop_breaker = False
    eng.auto_loop_recover = False
    eng.collect_logprob_confidence = "auto"
    eng._confidence_worker = None
    eng.image_token_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))
    eng.tokenizer = ftok
    eng.vocab_remap = None
    eng.vision_model = _StubVision()
    eng.lm = _StubLM()
    eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    eng._new_to_old_mx = None
    eng._old_to_new = None
    eng._vision_prepass = None

    im = Image.new("RGB", (64, 64), "white")
    try:
        res_on = eng.ocr_image(im)
        assert "logprob_confidence" in res_on.metadata
        assert res_on.metadata["logprob_confidence"]["n_tokens"] == len(tok_hello)
        assert res_on.metadata["logprob_confidence"]["mean_margin"] > 10.0

        eng.lm.step = 0
        res_off = eng.ocr_image(im, collect_logprob_confidence="off")
        assert "logprob_confidence" not in res_off.metadata
    finally:
        eng.close()

    # Batched B>=2 keeps collect_telemetry=False in "auto" mode and consumes mm_res.telemetry when "on"
    seen_collect: list[bool] = []

    def _spy_mm(**kw: Any):
        flag = bool(kw.get("collect_telemetry", False))
        seen_collect.append(flag)
        return [
            BatchedMMItemResult(
                generated_ids=list(tok_hello),
                prompt_tokens=10,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=1.0,
                image_grids=[(1, 2, 2)],
                telemetry=(
                    [{"logprob": -0.2, "prob": 0.8187, "entropy": 0.5}] * len(tok_hello)
                    if flag
                    else None
                ),
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _spy_mm)
    outs_auto = eng.ocr_images([im, im], batch_size=2)
    assert seen_collect[-1] is False
    assert "logprob_confidence" not in outs_auto[0].metadata

    outs_on = eng.ocr_images([im, im], batch_size=2, collect_logprob_confidence=True)
    assert seen_collect[-1] is True
    assert outs_on[0].metadata["logprob_confidence"]["n_tokens"] == len(tok_hello)
    assert outs_on[0].metadata["logprob_confidence"]["mean_logprob"] == pytest.approx(-0.2, abs=1e-6)


def test_pipeline_confidence_escalation_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_ocr.pipeline import AutoOCRPipeline

    # Gate requirement: auto-escalation stays opt-in (default None)
    assert inspect.signature(AutoOCRPipeline.__init__).parameters["confidence_escalate_dpi"].default is None

    pdf_path = tmp_path / "scan_conf.pdf"
    _write_positioned_pdf(pdf_path, [[(72, 720, "Low contrast line page 0", 11)], [(72, 720, "Clean line page 1", 11)]])

    render_dpis: list[tuple[int, list[int]]] = []
    ovis_calls: list[dict[str, Any]] = []

    def _fake_render(_p: Any, *, dpi: int, page_indices: Any = None, **_kw: Any):
        idxs = [0, 1] if page_indices is None else [int(i) for i in page_indices]
        render_dpis.append((int(dpi), idxs))
        return [Image.new("RGB", (100, 100), "white") for _ in idxs], idxs, 2.0

    class _FakeOvis:
        def ocr_images(self, sources: Sequence[Any], **kw: Any) -> list[OCRResult]:
            ovis_calls.append(dict(kw))
            out: list[OCRResult] = []
            for idx in kw.get("page_indices", list(range(len(sources)))):
                is_retry = kw.get("crop_margins") is False and kw.get("adaptive_resolution") is False
                lp = -0.05 if (idx == 1 or is_retry) else -0.45
                out.append(
                    OCRResult(
                        text=f"Page {idx} retry={is_retry}",
                        raw_text=f"Page {idx} retry={is_retry}",
                        backend_used="ovis:stub",
                        page_index=int(idx),
                        timings=OCRTimings(decode_ms=5.0, total_ms=10.0, generated_tokens=40),
                        metadata={
                            "logprob_confidence": summarize_logprob_confidence([lp] * 10).to_dict(),
                        },
                    )
                )
            return out

    pipe = AutoOCRPipeline(
        backend="ovis",
        confidence_escalate_dpi=300,
        confidence_escalate_threshold=-0.15,
        ovis_engine=_FakeOvis(),
    )
    monkeypatch.setattr(pipe, "_render_pdf_subset", _fake_render)
    outs = pipe.ocr_pdf(pdf_path, dpi=180, use_native_text=False)

    assert render_dpis == [(180, [0, 1]), (300, [0])]
    assert outs[0].text == "Page 0 retry=True"
    assert outs[0].metadata["confidence_escalation"]["triggered"] is True
    assert outs[0].metadata["confidence_escalation"]["escalate_dpi"] == 300
    assert outs[1].text == "Page 1 retry=False"
    assert "confidence_escalation" not in outs[1].metadata


def test_region_router_multicolumn_guard_and_digital_pdf_hybrid_splice(tmp_path: Path) -> None:
    from fast_ocr.pipeline import AutoOCRPipeline
    from fast_ocr.preprocess import extract_native_pdf_text
    from fast_ocr.region_router import analyze_digital_pdf_splice, extract_escalation_regions

    # 1. Two-column prose with fragmented horizontal spans (00f6c2-like) must NEVER interleave or crop as tables!
    two_col_lines = []
    for row_i in range(10):
        y0 = 0.08 + row_i * 0.04
        y1 = y0 + 0.015
        two_col_lines.append(
            OCRLine(f"Left column sentence number {row_i} explains model design.", 0.98, OCRBox(0.07, y0, 0.48, y1))
        )
        # Split right column row into 2 horizontal fragments like inline bold/math spans
        two_col_lines.append(OCRLine(f"Right col phrase {row_i}", 0.98, OCRBox(0.52, y0, 0.70, y1)))
        two_col_lines.append(OCRLine(f"continues topic cleanly {row_i}.", 0.98, OCRBox(0.72, y0, 0.93, y1)))

    vr_mc = OCRResult(text="", raw_text="", backend_used="apple-vision:accurate:b3", lines=tuple(two_col_lines))
    segs_mc, ok_mc = extract_escalation_regions(vr_mc, (1500, 2000))
    assert ok_mc is False and segs_mc == []

    # 2. Mixed digital PDF: Page 0 = prose+table splice; Page 1 = court caption single-col native; Page 2 = 2-col prose
    pdf_path = tmp_path / "hybrid_splice.pdf"
    p0 = [
        (56, 730, "Quarterly Operations Report", 14),
        (56, 700, "Global software revenue expanded solidly during fiscal year 2026.", 11),
        (56, 682, "Operating leverage improved sequentially as manufacturing bottlenecks eased.", 11),
        (56, 664, "Regional teams maintained strict expense discipline across core hubs.", 11),
        (56, 646, "Management expects sustained free cash flow growth over next year.", 11),
        (64, 580, "Segment", 11),
        (280, 580, "FY2025", 11),
        (430, 580, "FY2026", 11),
        (64, 554, "Cloud Platform", 11),
        (280, 554, "$1,250.00", 11),
        (430, 554, "$1,680.00", 11),
        (64, 528, "Edge Silicon", 11),
        (280, 528, "$840.50", 11),
        (430, 528, "$1,120.75", 11),
        (56, 440, "All figures follow audited consolidated GAAP financial reporting standards.", 11),
        (56, 422, "Subsidiary adjustments are reconciled inside footnote disclosures.", 11),
    ]
    p1 = [
        (180, 735, "UNITED STATES DISTRICT COURT", 13),
        (72, 695, "ACME CORPORATION,", 11),
        (245, 695, ")", 11),
        (350, 695, "Judge A. Smith", 11),
        (120, 675, "Plaintiff,", 11),
        (245, 675, ")", 11),
        (350, 675, "Case No. 1:26-cv-00412", 11),
        (72, 655, "WIDGET HOLDINGS LLC,", 11),
        (245, 655, ")", 11),
        (350, 655, "Civil Action", 11),
        (72, 590, "Because no genuine dispute of material fact remains, summary judgment is GRANTED.", 11),
    ]
    p2 = [
        (56, 700, "First column paragraph one describes", 11),
        (56, 682, "long-term productivity growth trends", 11),
        (56, 664, "across domestic fabrication plants", 11),
        (56, 646, "and regional equipment supply chains.", 11),
        (56, 628, "Capital utilization remained elevated.", 11),
        (330, 700, "Second column commentary begins cleanly", 11),
        (330, 682, "after the first column completes", 11),
        (330, 664, "with quarterly monetary policy notes", 11),
        (330, 646, "and sovereign yield curve analysis.", 11),
        (330, 628, "Inflation expectations stayed anchored.", 11),
    ]
    _write_positioned_pdf(pdf_path, [p0, p1, p2])

    native_pages = extract_native_pdf_text(pdf_path)
    specs0, disp0 = analyze_digital_pdf_splice(native_pages[0])
    specs1, disp1 = analyze_digital_pdf_splice(native_pages[1])
    specs2, disp2 = analyze_digital_pdf_splice(native_pages[2])
    assert disp0 == "splice_tables" and any(s[0] == "table" for s in specs0)
    assert disp1 == "single_column_clean" and specs1 == []
    assert disp2 == "multi_column" and specs2 == []

    # Wire into AutoOCRPipeline(region_routing=True): Page 0 splices 1 crop at min_pixels=4096, Page 1 runs native!
    crop_calls: list[dict[str, Any]] = []

    class _FakeVisionFailIfCalled:
        def recognize_image(self, *_a: Any, **_kw: Any) -> OCRResult:
            raise AssertionError("ANE Vision should not run on clean digital splice/single-column pages")

    class _FakeSpliceOvis:
        def ocr_image(self, src: Image.Image, **kw: Any) -> OCRResult:
            crop_calls.append({"size": src.size, "min_pixels": kw.get("min_pixels")})
            return OCRResult(
                text=(
                    "| Segment | FY2025 | FY2026 |\n"
                    "| --- | --- | --- |\n"
                    "| Cloud Platform | $1,250.00 | $1,680.00 |\n"
                    "| Edge Silicon | $840.50 | $1,120.75 |"
                ),
                raw_text="table",
                backend_used="ovis:stub",
                timings=OCRTimings(decode_ms=4.0, total_ms=8.0, generated_tokens=32),
            )

        def ocr_images(self, srcs: list[Image.Image], **kw: Any) -> list[OCRResult]:
            return [self.ocr_image(s, **kw) for s in srcs]

    with AutoOCRPipeline(
        backend="auto",
        region_routing=True,
        vision_ocr=_FakeVisionFailIfCalled(),
        ovis_engine=_FakeSpliceOvis(),
    ) as pipe:
        res_list = pipe.ocr_pdf(pdf_path, dpi=150, page_indices=[0, 1])

    assert len(res_list) == 2
    assert len(crop_calls) == 1
    assert crop_calls[0]["min_pixels"] == 4096
    r0, r1 = res_list
    assert r0.backend_used == "ovis:stub+native-splice"
    assert r0.metadata["region_routing"]["mode"] == "digital_pdf_splice"
    assert "Global software revenue expanded solidly" in r0.text
    assert "| Cloud Platform | $1,250.00 | $1,680.00 |" in r0.text
    assert "Subsidiary adjustments are reconciled" in r0.text
    assert r0.text.index("Global software revenue") < r0.text.index("| Cloud Platform |") < r0.text.index("Subsidiary adjustments")
    assert r1.backend_used == "native"
    assert "ACME CORPORATION" in r1.text and "summary judgment is GRANTED" in r1.text
