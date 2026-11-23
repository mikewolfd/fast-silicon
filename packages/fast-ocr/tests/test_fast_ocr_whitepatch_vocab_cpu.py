"""Pure CPU unit tests for safe white-patch dropping, OCR vocab subsetting, and kv_i8='auto'/GDN wiring."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
import pytest
from tokenizers import AddedToken, Tokenizer as HfTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from fast_ocr.ovis_engine import (
    _OCR_EXTRA_CHARS,
    FastOvisOCREngine,
    build_ocr_vocab_subset,
    resolve_ovis_kv_mode,
)
from fast_ocr.types import OCRTimings
from fast_qwen3 import FastQwen3Model, Qwen3Config
from fast_qwen3.multimodal_batch import (
    BatchedMMItemResult,
    batched_multimodal_generate,
    compute_safe_white_patch_keep_mask,
)
from fast_qwen3.tokenizer import FastQwen3Tokenizer
from fast_shared.qwen35_vision import (
    FastQwen35VisionModel,
    Qwen35VisionConfig,
    compute_mrope_positions,
    precompute_mrope_trig,
)
from fast_shared.vision_preprocess import QwenVisionInputs


def _make_tiny_qwen3_model(*, vocab_size: int = 320, head_dim: int = 64, seed: int = 11) -> tuple[Qwen3Config, FastQwen3Model]:
    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        vocab_size=vocab_size,
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


def _make_bpe_tokenizer(tmp_path: Path, *, extra_strings: tuple[str, ...] = ()) -> FastQwen3Tokenizer:
    bl = ByteLevel(add_prefix_space=False)
    vocab: dict[str, int] = {ch: i for i, ch in enumerate(sorted(ByteLevel.alphabet()))}
    merges: list[tuple[str, str]] = []

    def _add_word(piece: str) -> None:
        if len(piece) <= 1 or piece in vocab:
            return
        if piece.startswith("Ġ") and piece[1:] in vocab:
            merges.append(("Ġ", piece[1:]))
            vocab[piece] = len(vocab)
            return
        cur = piece[0]
        for nxt in piece[1:]:
            merged = cur + nxt
            if merged not in vocab:
                merges.append((cur, nxt))
                vocab[merged] = len(vocab)
            cur = merged

    # Push custom symbol merges above the 1024 must_prefix cutoff so build_ascii_latin_subset cannot keep them via tid < 1024
    for idx in range(900):
        tok_s = f"w{idx:03d}"
        _add_word(tok_s)

    for s in extra_strings:
        for piece, _span in bl.pre_tokenize_str(s):
            _add_word(piece)

    bpe = HfTokenizer(BPE(vocab=vocab, merges=merges))
    bpe.pre_tokenizer = bl
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
    tok_p = tmp_path / "tok_test.json"
    bpe.save(str(tok_p))
    return FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))


def test_compute_safe_white_patch_keep_mask_hairline_dot_margin_and_guard() -> None:
    # 256x256 image -> 8x8 = 64 merged 32x32 visual tokens
    arr = np.full((256, 256, 3), 255, dtype=np.uint8)

    # Block (0, 0): 1px chromatic hairline rule -> min_rgb=201, luma_std ~ 1.17 -> MUST KEEP
    arr[16, 0:32] = [255, 254, 201]
    blk0 = arr[0:32, 0:32].astype(np.float32)
    luma0 = 0.299 * blk0[:, :, 0] + 0.587 * blk0[:, :, 1] + 0.114 * blk0[:, :, 2]
    assert int(blk0.min()) == 201
    assert 1.1 < float(luma0.std()) < 1.25

    # Block (0, 1): faint dark period `.` -> min_rgb=240, std ~ 0.50 -> MUST KEEP
    arr[20, 48] = [240, 240, 240]
    arr[20, 49] = [245, 245, 245]
    blk1 = arr[0:32, 32:64].astype(np.float32)
    assert int(blk1.min()) == 240
    assert 0.4 < float(blk1.std(axis=(0, 1)).max()) < 0.6

    # Block (0, 2): very faint neutral 1px rule with min_rgb=250 (saved solely by edge_grad_max_rgb=5 > 4) -> MUST KEEP
    arr[10, 64:96] = [250, 250, 250]

    # Populate rows 2..3 (16 blocks) with dark ink so >= 16 kept tokens prevent safety fallback
    arr[64:128, 0:256] = 20

    img = Image.fromarray(arr, mode="RGB")
    keep = compute_safe_white_patch_keep_mask(img, 256, 256, patch_size=16, merge_size=2, min_keep_tokens=16)
    assert keep.shape == (64,)
    assert keep.dtype == np.bool_
    grid = keep.reshape(8, 8)
    assert bool(grid[0, 0]) is True  # chromatic hairline rule
    assert bool(grid[0, 1]) is True  # dark period `.`
    assert bool(grid[0, 2]) is True  # faint neutral edge_grad=5 rule
    assert not np.any(grid[0, 3:])  # pure white margin blocks dropped
    assert np.all(grid[2:4, :])  # ink blocks kept
    assert not np.any(grid[4:, :])  # pure white bottom margins dropped

    # Resized input path works cleanly
    keep_resized = compute_safe_white_patch_keep_mask(
        img.resize((512, 512), resample=Image.Resampling.BICUBIC),
        256,
        256,
        patch_size=16,
        merge_size=2,
        min_keep_tokens=16,
    )
    assert keep_resized.shape == (64,)
    assert int(keep_resized.sum()) >= 18

    # Safety guard: pure white image (0 < min_keep_tokens) returns all-True mask (never 0 visual tokens)
    blank = Image.new("RGB", (256, 256), (255, 255, 255))
    keep_blank = compute_safe_white_patch_keep_mask(blank, 256, 256)
    assert keep_blank.shape == (64,)
    assert np.all(keep_blank)

    # Sparse image with < min_keep_tokens (e.g. 5 blocks) also returns all-True mask
    sparse_arr = np.full((256, 256, 3), 255, dtype=np.uint8)
    sparse_arr[0:32, 0:160] = 30
    keep_sparse = compute_safe_white_patch_keep_mask(
        Image.fromarray(sparse_arr, mode="RGB"),
        256,
        256,
        min_keep_tokens=16,
    )
    assert np.all(keep_sparse)


def test_batched_multimodal_generate_white_patch_mrope_and_gdn_mode(tmp_path: Path) -> None:
    ftok = _make_bpe_tokenizer(tmp_path)
    cfg, model = _make_tiny_qwen3_model(vocab_size=ftok.vocab_size, head_dim=64, seed=17)
    vcfg = Qwen35VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=cfg.hidden_size,
        num_position_embeddings=64,
    )
    vis = FastQwen35VisionModel(vcfg)
    mx.eval(vis.parameters())

    img_tok = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))
    pre_ids = ftok.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
    suf_ids = ftok.encode("<|vision_end|>\nTranscribe<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    # Middle band ink (rows 2..5 = 32 blocks), top & bottom white margins (rows 0..1 & 6..7 = 32 blocks)
    arr = np.full((256, 256, 3), 255, dtype=np.uint8)
    arr[64:192, :] = 15
    img_half = Image.fromarray(arr, mode="RGB")

    def _pb(_idx: int, vis_counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        nv = int(vis_counts[0])
        s0 = len(pre_ids)
        return pre_ids + ([img_tok] * nv) + suf_ids, [(s0, s0 + nv)]

    seen_rope_trigs: list[tuple[mx.array, mx.array]] = []
    seen_make_cache_kwargs: list[dict[str, Any]] = []
    orig_forward_hidden = model.forward_hidden
    orig_make_cache = model.make_cache

    def _spy_make_cache(
        *,
        kv_step: int = 256,
        kv_i8: bool | str = False,
        gdn_state_mode: str = "auto",
        batch_size: int = 1,
    ):
        seen_make_cache_kwargs.append(
            {
                "kv_step": kv_step,
                "kv_i8": kv_i8,
                "gdn_state_mode": gdn_state_mode,
                "batch_size": batch_size,
            }
        )
        return orig_make_cache(kv_step=kv_step, kv_i8=kv_i8)

    def _spy_forward_hidden(*args: Any, **kwargs: Any) -> mx.array:
        if kwargs.get("rope_trig") is not None:
            seen_rope_trigs.append(kwargs["rope_trig"])
        return orig_forward_hidden(*args, **kwargs)

    model.make_cache = _spy_make_cache  # type: ignore[method-assign]
    model.forward_hidden = _spy_forward_hidden  # type: ignore[method-assign]

    res_keep, stats_keep = batched_multimodal_generate(
        lm=model,
        vision_model=vis,
        tokenizer=ftok,
        images_per_item=[[img_half]],
        prompt_builder=_pb,
        image_token_id=img_tok,
        new_to_old=np.arange(ftok.vocab_size, dtype=np.int32),
        old_to_new=None,
        grammars=[None],
        max_tokens=3,
        batch_size=4,
        drop_white_patches=False,
        gdn_state_mode="int8",
        stop_ids=ftok.stop_token_ids,
        pad_token_id=ftok.eos_token_id,
    )
    trigs_keep = list(seen_rope_trigs)
    seen_rope_trigs.clear()

    res_drop, stats_drop = batched_multimodal_generate(
        lm=model,
        vision_model=vis,
        tokenizer=ftok,
        images_per_item=[[img_half], [img_half]],
        prompt_builder=_pb,
        image_token_id=img_tok,
        new_to_old=np.arange(ftok.vocab_size, dtype=np.int32),
        old_to_new=None,
        grammars=[None, None],
        max_tokens=3,
        batch_size=4,
        drop_white_patches=True,
        min_pixels_per_item=[65536, 4096],
        max_pixels_per_item=[65536, 16384],
        gdn_state_mode="int8",
        stop_ids=ftok.stop_token_ids,
        pad_token_id=ftok.eos_token_id,
    )
    assert stats_keep["dropped_visual_tokens"] == 0
    assert res_keep[0].dropped_visual_tokens == 0
    assert res_drop[0].dropped_visual_tokens == 32
    assert res_drop[0].prompt_tokens == res_keep[0].prompt_tokens - 32
    assert stats_drop["dropped_visual_tokens"] >= 32
    assert res_drop[0].image_grids == [(1, 16, 16)]
    assert res_drop[1].image_grids == [(1, 8, 8)]  # max_pixels_per_item=16384 -> 128x128 -> (1, 8, 8)
    assert all(kw["gdn_state_mode"] == "int8" for kw in seen_make_cache_kwargs)
    assert [kw["batch_size"] for kw in seen_make_cache_kwargs] == [1, 2]

    # Verify MRoPE invariance: full-grid prefill suffix + decode positions match dropped prefill suffix + decode positions
    # Note: `res_drop` sorts shorter item 1 (`max_pixels=16384`) into wave row 0 and item 0 (`max_pixels=65536`) into row 1.
    prefill_cos_keep, prefill_sin_keep = trigs_keep[0]
    prefill_cos_drop, prefill_sin_drop = seen_rope_trigs[0]
    decode_cos_keep, decode_sin_keep = trigs_keep[1]
    decode_cos_drop, decode_sin_drop = seen_rope_trigs[1]
    mx.eval(prefill_cos_keep, prefill_cos_drop, decode_cos_keep, decode_cos_drop)
    suf_len = len(suf_ids)
    pre_len = len(pre_ids)
    # Kept middle-band visual tokens (indices 16..48) retain exact original 2D (H, W) MRoPE coordinates!
    assert float(
        mx.max(
            mx.abs(
                prefill_cos_keep[0, :, pre_len + 16 : pre_len + 48, :]
                - prefill_cos_drop[1, :, pre_len : pre_len + 32, :]
            )
        ).item()
    ) == 0.0
    # Suffix RoPE cos/sin at end of item 0's prompt (row 1) must be bit-identical after white-patch drop!
    assert float(mx.max(mx.abs(prefill_cos_keep[0, :, -suf_len:, :] - prefill_cos_drop[1, :, -suf_len:, :])).item()) == 0.0
    assert float(mx.max(mx.abs(prefill_sin_keep[0, :, -suf_len:, :] - prefill_sin_drop[1, :, -suf_len:, :])).item()) == 0.0
    # First decode step RoPE position (`max_pos + 1`) for item 0 (row 1) must also be bit-identical!
    assert float(mx.max(mx.abs(decode_cos_keep[0] - decode_cos_drop[1])).item()) == 0.0
    assert float(mx.max(mx.abs(decode_sin_keep[0] - decode_sin_drop[1])).item()) == 0.0


def test_ovis_b1_white_patch_drop_and_kv_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ftok = _make_bpe_tokenizer(tmp_path)
    cfg, model = _make_tiny_qwen3_model(vocab_size=ftok.vocab_size, head_dim=64, seed=23)
    vcfg = Qwen35VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=cfg.hidden_size,
        num_position_embeddings=64,
    )
    vis = FastQwen35VisionModel(vcfg)
    mx.eval(vis.parameters())

    eng = object.__new__(FastOvisOCREngine)
    eng.model_dir = tmp_path
    eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    eng.output_format = "raw"
    eng.table_mode = "gfm"
    eng.strict_markdown_grammar = False
    eng.zoom_recover = False
    eng.crop_margins = False
    eng.crop_pad_px = 0
    eng.auto_orient = False
    eng.speculative = "off"
    eng.draft_k = 4
    eng.prefill_chunk_size = 0
    eng.max_tokens = 3
    eng.loop_breaker = False
    eng.auto_loop_recover = False
    eng.kv_i8 = "auto"
    eng.gdn_state_mode = "int8"
    eng.drop_white_patches = True
    eng.image_token_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))
    eng.tokenizer = ftok
    eng.vocab_remap = None
    eng.vision_model = vis
    eng.lm = model
    eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    eng._new_to_old_mx = None
    eng._old_to_new = None
    eng._vision_prepass = None

    arr = np.full((256, 256, 3), 255, dtype=np.uint8)
    arr[0:128, :] = 25
    img = Image.fromarray(arr, mode="RGB")

    deltas_seen: list[int] = []
    cache_calls: list[dict[str, Any]] = []
    orig_fw = model.forward_hidden
    orig_mc = model.make_cache

    def _spy_mc(*, kv_step: int = 256, kv_i8: bool | str = False, gdn_state_mode: str = "auto"):
        cache_calls.append({"kv_step": kv_step, "kv_i8": kv_i8, "gdn_state_mode": gdn_state_mode})
        return orig_mc(kv_step=kv_step, kv_i8=kv_i8)

    def _spy_fw(*args: Any, **kwargs: Any) -> mx.array:
        if "rope_offset_delta" in kwargs:
            deltas_seen.append(int(kwargs["rope_offset_delta"]))
        return orig_fw(*args, **kwargs)

    monkeypatch.setattr(model, "make_cache", _spy_mc)
    monkeypatch.setattr(model, "forward_hidden", _spy_fw)
    monkeypatch.setattr("fast_ocr.ovis_engine.host_phys_memory_gb", lambda: 24.0)

    res_keep = eng.ocr_image(img, drop_white_patches=False, max_tokens=2)
    res_drop = eng.ocr_image(img, drop_white_patches=True, max_tokens=2)

    assert res_keep.metadata["dropped_visual_tokens"] == 0
    assert res_drop.metadata["dropped_visual_tokens"] == 32
    assert res_drop.timings.visual_tokens == 32
    assert res_drop.timings.prompt_tokens == res_keep.timings.prompt_tokens - 32
    assert res_drop.metadata["rope_delta"] == res_keep.metadata["rope_delta"] + 32
    assert cache_calls[0]["kv_i8"] == "bf16"
    assert cache_calls[0]["gdn_state_mode"] == "int8"

    # At max_tokens=4096 on 24 GB host -> B=1 selects q4_quarot!
    eng.ocr_image(img, drop_white_patches=True, max_tokens=4096)
    assert cache_calls[-1]["kv_i8"] == "q4_quarot"


def test_build_ocr_vocab_subset_extra_symbols_leading_space_and_cjk_bypass(tmp_path: Path) -> None:
    # Ensure all expanded scientific/math/footnote OCR symbols are registered
    for req in ("−", "→", "≤", "≥", "≠", "≈", "∞", "√", "∑", "∫", "†", "‡", "‑", "α", "Ω", "₀", "₉"):
        assert req in _OCR_EXTRA_CHARS

    extra_toks = ("✓", " ✓", "✔", " ✔", "™", " ™", "−", " −", "≤", " ≤", "中文", "日本")
    ftok = _make_bpe_tokenizer(tmp_path, extra_strings=extra_toks)
    full_v = ftok.vocab_size

    # Verify each leading-space symbol is a single BPE token with ID >= 1024
    space_check_ids = {s: ftok.encode(s, add_special_tokens=False) for s in (" ✓", " ✔", " ™", " −", " ≤")}
    for s, ids in space_check_ids.items():
        assert len(ids) == 1 and ids[0] >= 1024, f"Expected single high-ID token for {s!r}, got {ids}"

    zh_ids = ftok.encode("中文", add_special_tokens=False)
    assert len(zh_ids) == 1 and zh_ids[0] >= 1024

    remap = build_ocr_vocab_subset(ftok, full_vocab_size=full_v)
    assert remap is not None
    kept_set = set(remap.new_to_old.tolist())
    for s, ids in space_check_ids.items():
        assert ids[0] in kept_set, f"Missing leading-space symbol token for {s!r}"
    assert zh_ids[0] not in kept_set

    # Latin languages still subset; non-Latin languages or latin_only=False bypass to None
    assert build_ocr_vocab_subset(ftok, full_vocab_size=full_v, languages=["en-US", "fr"]) is not None
    for lang in ("zh", "zh-Hans", "ja", "ko", "ar", "ru", "hi"):
        assert build_ocr_vocab_subset(ftok, full_vocab_size=full_v, languages=[lang]) is None
    assert build_ocr_vocab_subset(ftok, full_vocab_size=full_v, latin_only=False) is None


def test_ocr_images_signature_prompt_builder_and_kv_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sig_init = inspect.signature(FastOvisOCREngine.__init__)
    assert sig_init.parameters["kv_i8"].default == "auto"
    assert sig_init.parameters["gdn_state_mode"].default == "auto"
    assert sig_init.parameters["drop_white_patches"].default is False
    assert sig_init.parameters["latin_only"].default is True
    assert sig_init.parameters["languages"].default is None

    sig_batch = inspect.signature(FastOvisOCREngine.ocr_images)
    for name in ("prompts", "min_pixels", "max_pixels", "drop_white_patches"):
        assert name in sig_batch.parameters
        assert sig_batch.parameters[name].default is None

    ftok = _make_bpe_tokenizer(tmp_path)
    img_tok_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))

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
    stub_eng.prefill_chunk_size = 512
    stub_eng.max_tokens = 2048
    stub_eng.loop_breaker = False
    stub_eng.auto_loop_recover = False
    stub_eng.kv_i8 = "auto"
    stub_eng.gdn_state_mode = "auto"
    stub_eng.drop_white_patches = True
    stub_eng.image_token_id = img_tok_id
    stub_eng.tokenizer = ftok
    stub_eng.vocab_remap = None
    stub_eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub_eng._new_to_old_mx = None
    stub_eng._old_to_new = None
    stub_eng._vision_prepass = None
    stub_eng.vision_model = object()  # type: ignore[assignment]

    class _DummyLM:
        vocab_n = ftok.vocab_size

        class cfg:
            head_dim = 256

    stub_eng.lm = _DummyLM()  # type: ignore[assignment]

    captured: dict[str, Any] = {}

    def _spy_batched_mm(**kw: Any):
        captured.update(kw)
        return [
            BatchedMMItemResult(
                generated_ids=ftok.encode("ok", add_special_tokens=False),
                prompt_tokens=32,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=2.0,
                decode_ms=3.0,
                image_grids=[(1, 8, 8)],
                dropped_visual_tokens=4,
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1, "compactions": 0, "vit_ms": 1.0, "prefill_ms": 2.0, "decode_ms": 3.0, "dropped_visual_tokens": 8}

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _spy_batched_mm)

    imgs = [Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white")]
    outs = stub_eng.ocr_images(
        imgs,
        prompts=["TRANSCRIBE_TABLE", "TRANSCRIBE_FORMULA"],
        min_pixels=4096,
        max_pixels=[500_000, 900_000],
        batch_size=8,
    )
    assert len(outs) == 2
    assert outs[0].timings.visual_tokens == 12  # (8*8//4)=16 - 4 dropped
    assert outs[0].metadata["dropped_visual_tokens"] == 4
    assert captured["min_pixels"] == 4096
    assert captured["min_pixels_per_item"] is None
    assert captured["max_pixels_per_item"] == [500_000, 900_000]
    assert captured["drop_white_patches"] is True
    assert captured["gdn_state_mode"] == "auto"

    pb = captured["prompt_builder"]
    ids0, spans0 = pb(0, [5])
    ids1, spans1 = pb(1, [7])
    assert spans0 == [(spans0[0][0], spans0[0][0] + 5)]
    assert spans1 == [(spans1[0][0], spans1[0][0] + 7)]
    assert ids0[spans0[0][0] : spans0[0][1]] == [img_tok_id] * 5
    assert ids1[spans1[0][0] : spans1[0][1]] == [img_tok_id] * 7
    assert "TRANSCRIBE_TABLE" in ftok.decode(ids0, skip_special_tokens=False)
    assert "TRANSCRIBE_FORMULA" in ftok.decode(ids1, skip_special_tokens=False)

    kv_res = captured["kv_i8_resolver"]
    monkeypatch.setattr("fast_ocr.ovis_engine.host_phys_memory_gb", lambda: 24.0)
    assert kv_res(4, 2048) == "bf16"
    assert kv_res(8, 2048) == "q4_quarot"
    assert kv_res(16, 2048) == "q4_quarot"
    assert kv_res(2, 4096) == "q4_quarot"

    monkeypatch.setattr("fast_ocr.ovis_engine.host_phys_memory_gb", lambda: 64.0)
    assert kv_res(4, 4096) == "bf16"
    assert kv_res(8, 2048) == "q4_quarot"
    assert kv_res(16, 2048) == "q4_quarot"
    assert kv_res(2, 8192) == "q4_quarot"

    # Explicit kv_i8 overrides auto
    assert resolve_ovis_kv_mode("bf16", batch_size=32, alloc_ctx=16384, host_mem_gb=24.0) == "bf16"
    assert resolve_ovis_kv_mode(False, batch_size=32, alloc_ctx=16384, host_mem_gb=24.0) == "bf16"
    assert resolve_ovis_kv_mode("q8", batch_size=2, alloc_ctx=512, host_mem_gb=64.0) == "q8"
    assert resolve_ovis_kv_mode("q4_quarot", batch_size=1, alloc_ctx=256, host_mem_gb=64.0) == "q4_quarot"

    # B=1 path in ocr_images forwards per-item prompt, min_pixels, max_pixels, and drop_white_patches
    b1_calls: list[dict[str, Any]] = []

    def _spy_b1(src: Any, **kw: Any):
        b1_calls.append(kw)
        return type("Res", (), {"timings": OCRTimings(), "metadata": {}})()

    monkeypatch.setattr(stub_eng, "ocr_image", _spy_b1)
    stub_eng.ocr_images(
        [imgs[0]],
        prompts=["SINGLE_CROP"],
        min_pixels=[4096],
        max_pixels=[786432],
        drop_white_patches=False,
    )
    assert len(b1_calls) == 1
    assert b1_calls[0]["prompt"] == "SINGLE_CROP"
    assert b1_calls[0]["min_pixels"] == 4096
    assert b1_calls[0]["max_pixels"] == 786432
    assert b1_calls[0]["drop_white_patches"] is False


def test_pipeline_passes_kv_gdn_whitepatch_args_to_ovis_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_ocr import pipeline as pipe_mod

    seen: list[dict[str, Any]] = []

    class _FakeOvisEngine:
        def __init__(
            self,
            *,
            model_path: Any = None,
            kv_i8: Any = "auto",
            gdn_state_mode: str = "auto",
            drop_white_patches: bool = False,
            **kw: Any,
        ) -> None:
            seen.append(
                {
                    "kv_i8": kv_i8,
                    "gdn_state_mode": gdn_state_mode,
                    "drop_white_patches": drop_white_patches,
                    **kw,
                }
            )

    monkeypatch.setattr("fast_ocr.ovis_engine.FastOvisOCREngine", _FakeOvisEngine)

    # Explicit overrides forward through `_get_ovis_engine()` while defaults (`None`) preserve engine defaults.
    p_explicit = pipe_mod.AutoOCRPipeline(
        backend="auto",
        kv_i8="q4_quarot",
        gdn_state_mode="bf16",
        drop_white_patches=True,
    )
    assert p_explicit.adaptive_resolution is True
    assert p_explicit.deskew is False
    assert p_explicit.region_routing is False
    _ = p_explicit._get_ovis_engine()
    assert seen[-1]["kv_i8"] == "q4_quarot"
    assert seen[-1]["gdn_state_mode"] == "bf16"
    assert seen[-1]["drop_white_patches"] is True

    p_default = pipe_mod.AutoOCRPipeline(backend="auto")
    _ = p_default._get_ovis_engine()
    assert seen[-1]["kv_i8"] == "auto"
    assert seen[-1]["gdn_state_mode"] == "auto"
    assert seen[-1]["drop_white_patches"] is False


def test_ocr_images_zoom_recovery_forwards_per_item_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_ocr.types import OCRBox, OCRLine, OCRResult

    ftok = _make_bpe_tokenizer(tmp_path)
    img_tok_id = int(ftok.backend_tokenizer.token_to_id("<|image_pad|>"))

    stub_eng = object.__new__(FastOvisOCREngine)
    stub_eng.model_dir = tmp_path
    stub_eng.raw_config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
    stub_eng.output_format = "markdown"
    stub_eng.table_mode = "gfm"
    stub_eng.strict_markdown_grammar = False
    stub_eng.zoom_recover = True
    stub_eng.crop_margins = False
    stub_eng.crop_pad_px = 0
    stub_eng.auto_orient = False
    stub_eng.adaptive_resolution = False
    stub_eng.deskew = False
    stub_eng.speculative = "off"
    stub_eng.draft_k = 4
    stub_eng.prefill_chunk_size = 512
    stub_eng.max_tokens = 256
    stub_eng.loop_breaker = False
    stub_eng.auto_loop_recover = False
    stub_eng.kv_i8 = "auto"
    stub_eng.gdn_state_mode = "auto"
    stub_eng.drop_white_patches = False
    stub_eng.image_token_id = img_tok_id
    stub_eng.tokenizer = ftok
    stub_eng._new_to_old = np.arange(ftok.vocab_size, dtype=np.int32)
    stub_eng._old_to_new = None
    stub_eng._vision_prepass = None
    stub_eng.vision_model = object()  # type: ignore[assignment]
    stub_eng.lm = type("LM", (), {"vocab_n": ftok.vocab_size, "cfg": type("Cfg", (), {"head_dim": 256})()})()  # type: ignore[assignment]

    def _fake_mm(**kw: Any):
        return [
            BatchedMMItemResult(
                generated_ids=ftok.encode("ok", add_special_tokens=False),
                prompt_tokens=16,
                hit_eos=True,
                hit_max_tokens=False,
                vit_ms=1.0,
                prefill_ms=1.0,
                decode_ms=1.0,
                image_grids=[(1, 4, 4)],
            )
            for _ in kw["images_per_item"]
        ], {"waves": 1}

    seen_zoom_prompts: list[Any] = []

    def _spy_zoom(*args: Any, **kw: Any):
        seen_zoom_prompts.append(kw.get("prompt"))
        return str(kw.get("rendered", "")), str(kw.get("raw_text", "")), None, None

    monkeypatch.setattr("fast_ocr.ovis_engine.batched_multimodal_generate", _fake_mm)
    monkeypatch.setattr(stub_eng, "_run_zoom_recovery", _spy_zoom)

    pre = OCRResult(
        text="row",
        raw_text="row",
        backend_used="apple-vision:accurate",
        lines=(OCRLine(text="row 123", confidence=0.95, box=OCRBox(0.1, 0.1, 0.8, 0.2)),),
    )
    imgs = [Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white")]
    _ = stub_eng.ocr_images(
        imgs,
        prompts=["ZOOM_PROMPT_0", "ZOOM_PROMPT_1"],
        pre_analyses=[pre, pre],
        zoom_recover=True,
        batch_size=2,
    )
    assert seen_zoom_prompts == ["ZOOM_PROMPT_0", "ZOOM_PROMPT_1"]


def test_pipeline_region_routing_crop_batch_cap_and_pdf_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_ocr import pipeline as pipe_mod
    from fast_ocr.region_router import PageSegment
    from fast_ocr.types import OCRBox, OCRLine, OCRResult, PageAnalysis

    crop_batch_sizes: list[int] = []

    class _FakeCropOvis:
        def ocr_images(self, sources: Any, **kw: Any):
            crop_batch_sizes.append(int(kw["batch_size"]))
            return [
                OCRResult(
                    text=f"| h |\n| --- |\n| {i} |",
                    raw_text=f"| h |\n| --- |\n| {i} |",
                    backend_used="ovis:test",
                    timings=OCRTimings(generated_tokens=8),
                )
                for i in range(len(sources))
            ]

    monkeypatch.setattr(
        pipe_mod,
        "extract_escalation_regions",
        lambda *_a, **_kw: (
            [
                PageSegment(kind="prose", markdown="Intro", line_count=4),
                PageSegment(kind="table", crop_box_px=(0, 10, 40, 30), prompt="T1", line_count=2),
                PageSegment(kind="table", crop_box_px=(0, 30, 40, 50), prompt="T2", line_count=2),
            ],
            True,
        ),
    )

    pipe = pipe_mod.AutoOCRPipeline(backend="auto", region_routing=True, ovis_engine=_FakeCropOvis())
    vis = OCRResult(
        text="Intro",
        raw_text="Intro",
        backend_used="apple-vision:accurate",
        lines=(OCRLine(text="Intro", confidence=0.95, box=OCRBox(0.0, 0.0, 0.5, 0.1)),),
    )
    imgs = [Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "white")]
    routed = pipe._try_region_route_pages(
        imgs, [0, 1], [vis, vis], output_format="markdown", table_mode="gfm", batch_size=2
    )
    assert set(routed.keys()) == {0, 1}
    assert crop_batch_sizes == [2]

    # Two-stage PDF triage (`triage_dpi < dpi`) routes escalated pages through `_dispatch_escalated_vlm`
    # so opt-in `region_routing=True` runs on re-rendered high-DPI pages and preserves order.
    rendered_calls: list[tuple[int, list[int] | None]] = []

    def _fake_render_subset(_path: Any, *, dpi: int, page_indices: Any = None, **_kw: Any):
        idxs = [0, 1] if page_indices is None else list(page_indices)
        rendered_calls.append((int(dpi), idxs))
        return [Image.new("RGB", (64, 64), "white") for _ in idxs], idxs, 5.0

    res_keep = OCRResult(
        text="Clean page 0",
        raw_text="Clean page 0",
        backend_used="apple-vision:accurate",
        page_index=0,
        analysis=PageAnalysis(
            width=64, height=64, line_count=5, char_count=40, digit_ratio=0.05,
            currency_hits=0, mean_confidence=0.98, low_conf_line_ratio=0.0,
            median_line_height=0.03, inferred_rotation_cw=0, ink_crop_box_px=None,
            ink_area_ratio=0.4, estimated_columns=1, table_score=0.0,
            needs_vlm=False, escalation_reasons=(),
        ),
    )
    res_esc = OCRResult(
        text="Table page 1",
        raw_text="Table page 1",
        backend_used="apple-vision:accurate",
        page_index=1,
        analysis=PageAnalysis(
            width=64, height=64, line_count=6, char_count=45, digit_ratio=0.25,
            currency_hits=1, mean_confidence=0.90, low_conf_line_ratio=0.0,
            median_line_height=0.03, inferred_rotation_cw=0, ink_crop_box_px=None,
            ink_area_ratio=0.4, estimated_columns=2, table_score=0.60,
            needs_vlm=True, escalation_reasons=("financial_table",),
        ),
    )
    routed_pdf_res = OCRResult(
        text="Routed PDF table 1",
        raw_text="Routed PDF table 1",
        backend_used="ovis:test+ane-regions",
        page_index=1,
        metadata={"region_routing": {"enabled": True, "crops": 1}},
    )
    monkeypatch.setattr(pipe, "_render_pdf_subset", _fake_render_subset)
    monkeypatch.setattr(pipe, "_run_vision_batch", lambda *_a, **_kw: [res_keep, res_esc])
    seen_route_bs: list[int | None] = []

    def _spy_route(srcs: Any, idxs: Any, vrs: Any, *, output_format: Any, table_mode: Any, batch_size: int | None = None):
        seen_route_bs.append(batch_size)
        return {0: routed_pdf_res}

    monkeypatch.setattr(pipe, "_try_region_route_pages", _spy_route)
    monkeypatch.setattr(pipe, "_run_ovis_batch", lambda *_a, **_kw: pytest.fail("Should not call full-page Ovis when region routing handled page"))

    pdf_outs = pipe._run_raster_pdf_pages(
        Path("dummy.pdf"),
        dpi=300,
        triage_dpi=150,
        page_indices=None,
        output_format="markdown",
        table_mode="gfm",
        batch_size=3,
    )
    assert rendered_calls == [(150, [0, 1]), (300, [1])]
    assert seen_route_bs == [3]
    assert [r.page_index for r in pdf_outs] == [0, 1]
    assert pdf_outs[1].text == "Routed PDF table 1"
    assert pdf_outs[1].metadata["triage_dpi"] == 150
