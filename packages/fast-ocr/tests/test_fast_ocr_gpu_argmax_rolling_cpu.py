"""CPU unit tests for OvisOCR2 A1 GPU/CPU argmax+confidence & B1/B2 rolling PLD routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest

from fast_ocr.ovis_engine import (
    FastOvisOCREngine,
    _StepConfidenceSession,
    _StepConfidenceWorker,
)
from fast_ocr.types import OCRBox, OCRLine, OCRResult, OCRTimings
from fast_qwen3.multimodal_batch import BatchedMMItemResult


def _write_tiny_tokenizer(tok_dir: Path) -> Path:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    tok_dir.mkdir(parents=True, exist_ok=True)
    vocab: dict[str, int] = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}
    for tok_s in ("Hello", "Ġworld", "!", "row"):
        if tok_s not in vocab:
            vocab[tok_s] = len(vocab)
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    tok = Tokenizer(BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder(add_prefix_space=False)
    tok.add_special_tokens(
        [
            AddedToken("<|endoftext|>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
            AddedToken("<think>", special=True),
            AddedToken("</think>", special=True),
        ]
    )
    tok.save(str(tok_dir / "tokenizer.json"))
    return tok_dir


def test_step_confidence_session_record_direct_and_mixed_flush() -> None:
    worker = _StepConfidenceWorker()
    try:
        sess = _StepConfidenceSession(worker)
        logits_fallback = np.array([1.0, 4.0, 2.0, -1.0] + [-5.0] * 60, dtype=np.float32)
        sess.submit(logits_fallback, 1)
        # Mixed direct call must flush prior worker item first so ordering is preserved
        sess.record_direct(-0.25, 1.75)
        summary = sess.finish()
        assert len(sess.logprobs) == 2
        assert len(sess.margins) == 2
        assert sess.margins[0] == pytest.approx(2.0, abs=1e-4)
        assert sess.logprobs[1] == pytest.approx(-0.25, abs=1e-6)
        assert sess.margins[1] == pytest.approx(1.75, abs=1e-6)
        assert summary.n_tokens == 2
    finally:
        worker.close()


def test_ovis_ocr_image_a1_gpu_argmax_confidence_and_fallback_cpu(tmp_path: Path) -> None:
    from fast_qwen3.tokenizer import FastQwen3Tokenizer

    tok_dir = _write_tiny_tokenizer(tmp_path / "tok")
    ftok = FastQwen3Tokenizer(tok_dir)

    stub = FastOvisOCREngine.__new__(FastOvisOCREngine)
    stub.model_dir = tok_dir
    stub.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [1, 1, 2]}}}
    stub.output_format = "raw"
    stub.table_mode = "gfm"
    stub.strict_markdown_grammar = False
    stub.zoom_recover = False
    stub.crop_margins = False
    stub.crop_pad_px = 0
    stub.auto_orient = False
    stub.adaptive_resolution = False
    stub.deskew = "off"
    stub.speculative = "off"
    stub.draft_k = 4
    stub.prefill_chunk_size = 0
    stub.max_tokens = 16
    stub.loop_breaker = False
    stub.auto_loop_recover = False
    stub.kv_i8 = "bf16"
    stub.gdn_state_mode = "fp32"
    stub.drop_white_patches = False
    stub.collect_logprob_confidence = "on"
    stub._confidence_worker = None
    stub.vocab_remap = None
    stub.image_token_id = 4
    stub.tokenizer = ftok
    stub._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub._new_to_old_mx = None
    stub._old_to_new = None
    stub._vision_prepass = None

    class _Cfg:
        rot_dims = 8
        rope_theta = 10000.0
        head_dim = 32
        vocab_size = ftok.vocab_size

    class _DummyVision:
        spatial_merge_size = 2

        def __call__(self, _px: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
            n_tok = sum((t * h * w) // 4 for t, h, w in grids)
            return mx.zeros((n_tok, 16), dtype=mx.bfloat16)

    seq_ids = ftok.encode("Hello world!", add_special_tokens=False) + [ftok.eos_token_id]

    class _DummyLM:
        cfg = _Cfg()
        vocab_n = ftok.vocab_size
        has_full_vocab_fallback = False

        def __init__(self) -> None:
            self.step = 0

        def embed_tokens(self, ids: mx.array) -> mx.array:
            return mx.zeros((int(ids.shape[0]), int(ids.shape[1]), 16), dtype=mx.bfloat16)

        def make_cache(self, **_kw: Any) -> list[Any]:
            return []

        def forward_hidden(self, ids: mx.array | None = None, _cache: Any = None, *, inputs_embeds: mx.array | None = None, **_kw: Any) -> mx.array:
            src = inputs_embeds if inputs_embeds is not None else ids
            assert src is not None
            return mx.zeros((int(src.shape[0]), int(src.shape[1]), 16), dtype=mx.bfloat16)

        def project_logits(self, h: mx.array, **_kw: Any) -> mx.array:
            b, s, _ = int(h.shape[0]), int(h.shape[1]), int(h.shape[2])
            out = np.full((b, s, ftok.vocab_size), -10.0, dtype=np.float32)
            for t in range(s):
                tok = seq_ids[min(self.step, len(seq_ids) - 1)]
                out[:, t, tok] = 6.0
                out[:, t, (tok + 1) % ftok.vocab_size] = 3.0
                self.step += 1
            return mx.array(out, dtype=mx.bfloat16)

    stub.vision_model = _DummyVision()  # type: ignore[assignment]
    stub.lm = _DummyLM()  # type: ignore[assignment]

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        res = stub.ocr_image(Image.new("RGB", (64, 64), "white"), collect_logprob_confidence="on")
        assert res.text == "Hello world!"
        conf = res.metadata["logprob_confidence"]
        assert conf["n_tokens"] == len(seq_ids) - 1
        assert conf["mean_margin"] == pytest.approx(3.0, abs=1e-3)
    finally:
        mx.set_default_device(prev_dev)
        stub.close()


def test_ovis_ocr_images_rolling_and_pld_gating_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_qwen3.tokenizer import FastQwen3Tokenizer
    tok_dir = _write_tiny_tokenizer(tmp_path / "tok")
    ftok = FastQwen3Tokenizer(tok_dir)

    stub = FastOvisOCREngine.__new__(FastOvisOCREngine)
    stub.model_dir = tok_dir
    stub.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [1, 1, 2]}}}
    stub.output_format = "markdown"
    stub.table_mode = "gfm"
    stub.strict_markdown_grammar = False
    stub.zoom_recover = False
    stub.crop_margins = False
    stub.crop_pad_px = 0
    stub.auto_orient = False
    stub.adaptive_resolution = False
    stub.deskew = "off"
    stub.speculative = "auto"
    stub.draft_k = 4
    stub.prefill_chunk_size = 0
    stub.max_tokens = 128
    stub.loop_breaker = False
    stub.auto_loop_recover = False
    stub.kv_i8 = "bf16"
    stub.gdn_state_mode = "fp32"
    stub.drop_white_patches = False
    stub.collect_logprob_confidence = "auto"
    stub._confidence_worker = None
    stub.vocab_remap = None
    stub.image_token_id = 4
    stub.tokenizer = ftok
    stub._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub._new_to_old_mx = None
    stub._old_to_new = None
    stub._vision_prepass = None
    stub.vision_model = object()  # type: ignore[assignment]
    stub.lm = type("LM", (), {"vocab_n": ftok.vocab_size, "cfg": type("Cfg", (), {"head_dim": 256})()})()  # type: ignore[assignment]

    captured: list[dict[str, Any]] = []

    def _spy_mm(**kw: Any):
        captured.append(
            {
                "rolling": kw.get("rolling"),
                "drafters": kw.get("drafters"),
                "spec_modes": kw.get("spec_modes"),
                "draft_k": kw.get("draft_k"),
                "batch_size": kw.get("batch_size"),
            }
        )
        return [
            BatchedMMItemResult(
                generated_ids=ftok.encode("Hello", add_special_tokens=False),
                prompt_tokens=16,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=2.0,
                image_grids=[(1, 4, 4)],
                drafted_tokens=4,
                accepted_draft_tokens=3,
                verify_passes=1,
            )
            for _ in kw["images_per_item"]
        ], {"rolling": bool(kw.get("rolling")), "use_pld": kw.get("drafters") is not None, "waves": 1}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _spy_mm)

    pre = OCRResult(
        text="Hello row",
        raw_text="Hello row",
        backend_used="apple-vision:accurate",
        lines=(OCRLine(text="Hello row", confidence=0.95, box=OCRBox(0.1, 0.1, 0.5, 0.2)),),
    )
    imgs = [Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white")]

    # 1. Default greedy unconstrained batch -> rolling="auto" + OCRPLDDrafter + spec_modes
    outs = stub.ocr_images(imgs, pre_analyses=[pre, pre], batch_size=16)
    assert captured[-1]["rolling"] == "auto"
    assert captured[-1]["batch_size"] == 16
    assert captured[-1]["draft_k"] == 4
    assert captured[-1]["spec_modes"] == ["vision+self", "vision+self"]
    assert captured[-1]["drafters"] is not None and len(captured[-1]["drafters"]) == 2
    assert outs[0].metadata["speculative_mode"] == "batched"
    assert outs[0].timings.drafted_tokens == 4
    assert outs[0].timings.accepted_draft_tokens == 3
    assert outs[0].timings.verify_passes == 1

    # 2. speculative="off" -> rolling="auto" (rolling AR) with drafters=None
    _ = stub.ocr_images(imgs, pre_analyses=[pre, pre], speculative="off", batch_size=8)
    assert captured[-1]["rolling"] == "auto"
    assert captured[-1]["drafters"] is None

    # 3. Non-eligible cases disable rolling cleanly (schema, temperature > 0, drop_white_patches, confidence="on")
    _ = stub.ocr_images(imgs, pre_analyses=[pre, pre], temperature=0.2, batch_size=8)
    assert captured[-1]["rolling"] is False and captured[-1]["drafters"] is None

    _ = stub.ocr_images(imgs, pre_analyses=[pre, pre], collect_logprob_confidence="on", batch_size=8)
    assert captured[-1]["rolling"] is False and captured[-1]["drafters"] is None

    _ = stub.ocr_images(imgs, pre_analyses=[pre, pre], drop_white_patches=True, batch_size=8)
    assert captured[-1]["rolling"] is False and captured[-1]["drafters"] is None

    _ = stub.ocr_images(
        imgs,
        pre_analyses=[pre, pre],
        schemas=[{"type": "object", "properties": {"x": {"type": "string"}}}, None],
        batch_size=8,
    )
    assert captured[-1]["rolling"] is False and captured[-1]["drafters"] is None
