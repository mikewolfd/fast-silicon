"""Shared batched multimodal ViT + 3D MRoPE chunked prefill & compacting decode wave execution.

Reused by both `fast_qwen3.engine.FastQwen3Engine` and `fast_ocr.ovis_engine.FastOvisOCREngine`
so neither duplicates ragged prefill, ViT frame bucketing, MRoPE decode, or cache row compaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

from fast_shared.grammar import (
    GrammarHelper,
    apply_loop_breaker_penalties_1d,
    compute_loop_breaker_penalties,
    safe_argmax_1d,
    sample_unconstrained,
)
from fast_shared.kv_quant import KVMode
from fast_shared.qwen35_vision import compute_mrope_positions, precompute_mrope_trig
from fast_shared.vision_preprocess import QwenVisionInputs, build_qwen_vision_inputs, load_image

from .model import DeltaSlotCache, GrowingKVCache, compact_hybrid_cache


def host_phys_memory_gb() -> float:
    """Return physical unified memory size in GiB."""
    return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0**3)


def default_wave_token_budget() -> int:
    """Bounded per-wave prompt token budget protecting 24 GB Macs while saturating >=32 GB UMA."""
    return 16384 if host_phys_memory_gb() <= 28.0 else 24576


def compute_step_telemetry(logits_1d: np.ndarray, chosen_idx: int) -> dict[str, float]:
    """Compute scalar next-token logprob, probability, and entropy."""
    arr = logits_1d.astype(np.float64)
    mx_v = float(np.max(arr))
    shifted = arr - mx_v
    exp_v = np.exp(shifted)
    z = float(np.sum(exp_v))
    probs = exp_v / max(z, 1e-30)
    p_tok = float(probs[chosen_idx]) if 0 <= chosen_idx < len(probs) else 0.0
    nz = probs[probs > 1e-12]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "logprob": float(np.log(max(p_tok, 1e-30))),
        "prob": p_tok,
        "entropy": entropy,
    }


def select_next_token_1d(
    logits_1d: np.ndarray,
    *,
    lm: Any,
    new_to_old: np.ndarray,
    old_to_new: np.ndarray | None,
    stop_token_ids: frozenset[int],
    vocab_size: int,
    full_hidden_1d: mx.array | None,
    matcher: Any | None,
    gram: GrammarHelper | None,
    table_gram: Any | None,
    recent_tokens: Sequence[int],
    temperature: float,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
    repeat_penalty: bool = False,
    collect_telemetry: bool = False,
) -> tuple[int, dict[str, float] | None]:
    """Sample/argmax one token from mapped logits with optional repeat penalty, JSON grammar, and table lock."""
    vocab_remap = getattr(lm, "vocab_remap", None)
    work = logits_1d[: len(new_to_old)] if (vocab_remap is None and len(logits_1d) > len(new_to_old)) else logits_1d
    argmax_idx = safe_argmax_1d(work)
    if repeat_penalty:
        min_hits = 5 if matcher is not None else 4
        pen = compute_loop_breaker_penalties(
            recent_tokens,
            window=96,
            min_prior_hits=min_hits,
            base_penalty=8.0,
            stop_token_ids=stop_token_ids,
        )
        if pen:
            work, argmax_idx = apply_loop_breaker_penalties_1d(work, pen, old_to_new)

    if matcher is not None and gram is not None:
        def _lazy_full() -> np.ndarray:
            assert full_hidden_1d is not None
            fl = lm.project_logits(full_hidden_1d[None, None, :], full_vocab=True)
            mx.eval(fl)  # guard: allow-eval
            return np.asarray(fl[0, 0].astype(mx.float32))

        can_fb = bool(getattr(lm, "has_full_vocab_fallback", False) and full_hidden_1d is not None)
        tok_id, _ = gram.pick_valid_token(
            matcher,
            work,
            argmax_idx,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
            fallback_to_full_vocab=can_fb,
            full_logits_1d=_lazy_full if can_fb else None,
        )
    elif table_gram is not None and getattr(table_gram, "is_active", False):
        full_mask = table_gram.prepare_mask(vocab_size=vocab_size)
        if full_mask is not None:
            active_mask = full_mask[new_to_old] if vocab_remap is not None else full_mask[: len(work)]
            if np.any(active_mask):
                work = np.where(active_mask, work, -1e9)
                argmax_idx = safe_argmax_1d(work)
        tok_id = sample_unconstrained(
            work,
            new_to_old,
            argmax_idx,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
        )
    else:
        tok_id = sample_unconstrained(
            work,
            new_to_old,
            argmax_idx,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
        )

    tel: dict[str, float] | None = None
    if collect_telemetry:
        c_idx = int(old_to_new[tok_id]) if old_to_new is not None else int(tok_id)
        tel = (
            compute_step_telemetry(work, c_idx)
            if 0 <= c_idx < len(work)
            else {"logprob": 0.0, "prob": 0.0, "entropy": 0.0}
        )
    return int(tok_id), tel


def pack_length_waves(
    items: Sequence[Any],
    length_fn: Callable[[Any], int],
    max_batch_size: int,
    *,
    max_pad_ratio: float | None = 1.8,
    max_wave_tokens: int | None = None,
) -> list[list[Any]]:
    """Group length-ordered items into waves bounded by batch size, pad ratio, and prompt token budget."""
    b_cap = max(1, int(max_batch_size))
    tok_budget = int(max_wave_tokens) if max_wave_tokens is not None else default_wave_token_budget()
    waves: list[list[Any]] = []
    cur: list[Any] = []
    min_l = 0
    max_l = 0
    for row in items:
        l_row = max(1, int(length_fn(row)))
        if not cur:
            cur = [row]
            min_l = l_row
            max_l = l_row
            continue
        new_min = min(min_l, l_row)
        new_max = max(max_l, l_row)
        exceed_b = len(cur) >= b_cap
        exceed_ratio = max_pad_ratio is not None and (float(new_max) / float(max(1, new_min))) > float(max_pad_ratio)
        exceed_toks = ((len(cur) + 1) * new_max) > tok_budget
        if exceed_b or exceed_ratio or exceed_toks:
            waves.append(cur)
            cur = [row]
            min_l = l_row
            max_l = l_row
        else:
            cur.append(row)
            min_l = new_min
            max_l = new_max
    if cur:
        waves.append(cur)
    return waves


def compute_safe_white_patch_keep_mask(
    image: Image.Image,
    target_h: int,
    target_w: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    min_keep_tokens: int = 16,
) -> np.ndarray:
    """Compute a conservative row-major visual-token keep mask preserving hairlines, faint punctuation, and ink.

    Resizes `image` (RGB) to `(target_w, target_h)` with `BICUBIC` if needed, groups pixels into
    `(grid_h // merge_size) * (grid_w // merge_size)` row-major `(patch_size * merge_size)^2` blocks
    matching `build_qwen_vision_inputs`, and drops a merged visual token iff:
    `(max_ch_std < 1.5) & (min_rgb >= 250) & (edge_grad_max_rgb <= 4)`.

    If fewer than `min_keep_tokens` (or 0) tokens survive, returns an all-True mask so visual spans
    are never empty.
    """
    from PIL import Image as _PILImage

    ps = int(patch_size)
    ms = int(merge_size)
    block = ps * ms
    th = int(target_h)
    tw = int(target_w)
    if ps <= 0 or ms <= 0 or th <= 0 or tw <= 0 or (th % block) != 0 or (tw % block) != 0:
        raise ValueError(
            f"target_h={target_h} and target_w={target_w} must be positive multiples of patch_size*merge_size={block}"
        )

    rgb = image.convert("RGB") if image.mode != "RGB" else image
    if rgb.size != (tw, th):
        rgb = rgb.resize((tw, th), resample=_PILImage.Resampling.BICUBIC)

    grid_h = th // ps
    grid_w = tw // ps
    mh = grid_h // ms
    mw = grid_w // ms
    n_tok = mh * mw

    arr_u8 = np.asarray(rgb, dtype=np.uint8)
    blocks = arr_u8.reshape(mh, block, mw, block, 3).transpose(0, 2, 1, 3, 4).reshape(n_tok, block, block, 3)

    max_ch_std = blocks.astype(np.float32).std(axis=(1, 2)).max(axis=1)
    min_rgb = blocks.min(axis=(1, 2, 3))
    if block > 1:
        blk_i16 = blocks.astype(np.int16)
        gx = np.abs(blk_i16[:, :, 1:, :] - blk_i16[:, :, :-1, :]).max(axis=(1, 2, 3))
        gy = np.abs(blk_i16[:, 1:, :, :] - blk_i16[:, :-1, :, :]).max(axis=(1, 2, 3))
        edge_grad_max_rgb = np.maximum(gx, gy)
    else:
        edge_grad_max_rgb = np.zeros((n_tok,), dtype=np.int16)

    drop = (max_ch_std < 1.5) & (min_rgb >= 250) & (edge_grad_max_rgb <= 4)
    keep = np.asarray(~drop, dtype=np.bool_)
    req_keep = max(1, int(min_keep_tokens))
    if int(keep.sum()) < req_keep:
        return np.ones((n_tok,), dtype=np.bool_)
    return keep


def _make_hybrid_cache(
    lm: Any,
    *,
    kv_step: int,
    kv_i8: KVMode | bool,
    gdn_state_mode: str = "auto",
    batch_size: int = 1,
) -> list[DeltaSlotCache | GrowingKVCache]:
    """Call `lm.make_cache` passing `gdn_state_mode`/`batch_size` safely whether or not the LM supports them."""
    make_fn = lm.make_cache
    supports_gdn = False
    supports_bs = False
    has_var_kw = False
    try:
        sig = inspect.signature(make_fn)
        params = sig.parameters
        supports_gdn = "gdn_state_mode" in params
        supports_bs = "batch_size" in params
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        pass
    kw: dict[str, Any] = {"kv_step": kv_step, "kv_i8": kv_i8}
    if supports_gdn or (has_var_kw and str(gdn_state_mode) != "auto"):
        kw["gdn_state_mode"] = gdn_state_mode
    if supports_bs:
        kw["batch_size"] = int(batch_size)
    try:
        return make_fn(**kw)
    except TypeError:
        if supports_gdn or supports_bs:
            raise
        return make_fn(kv_step=kv_step, kv_i8=kv_i8)


@dataclass
class BatchedMMItemResult:
    """Per-item completion outputs and attributed stage latencies from batched multimodal decode."""

    generated_ids: list[int]
    prompt_tokens: int
    hit_eos: bool
    hit_max_tokens: bool
    vit_ms: float
    prefill_ms: float
    decode_ms: float
    image_grids: list[tuple[int, int, int]] = field(default_factory=list)
    telemetry: list[dict[str, float]] | None = None
    dropped_visual_tokens: int = 0


def batched_multimodal_generate(
    *,
    lm: Any,
    vision_model: Any | None,
    tokenizer: Any,
    images_per_item: Sequence[Sequence[Any] | None],
    prompt_builder: Callable[[int, list[int]], tuple[list[int], list[tuple[int, int]]]],
    image_token_id: int,
    new_to_old: np.ndarray,
    old_to_new: np.ndarray | None,
    grammars: Sequence[GrammarHelper | None],
    table_grammars: Sequence[Any | None] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    slot_seeds: Sequence[int | None] | None = None,
    repeat_penalty: bool = False,
    collect_telemetry: bool = False,
    batch_size: int = 16,
    kv_step: int = 512,
    kv_i8_resolver: Callable[[int, int], KVMode] | KVMode = "bf16",
    compact_active: bool = True,
    max_pad_ratio: float | None = 1.8,
    max_wave_tokens: int | None = None,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    drop_white_patches: bool = False,
    min_pixels_per_item: Sequence[int | None] | None = None,
    max_pixels_per_item: Sequence[int | None] | None = None,
    gdn_state_mode: str = "auto",
    stop_ids: frozenset[int],
    pad_token_id: int,
    prefill_chunk_size: int = 512,
    mrope_section: tuple[int, int, int] | None = None,
) -> tuple[list[BatchedMMItemResult], dict[str, Any]]:
    """Run grid-bucketed ViT + 3D MRoPE chunked prefill + active-compacted decode over mixed items."""
    n_items = len(images_per_item)
    if n_items == 0:
        return [], {
            "waves": 0,
            "compactions": 0,
            "vit_ms": 0.0,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "dropped_visual_tokens": 0,
        }

    if min_pixels_per_item is not None and len(min_pixels_per_item) != n_items:
        raise ValueError(f"Expected {n_items} min_pixels_per_item entries, got {len(min_pixels_per_item)}")
    if max_pixels_per_item is not None and len(max_pixels_per_item) != n_items:
        raise ValueError(f"Expected {n_items} max_pixels_per_item entries, got {len(max_pixels_per_item)}")

    patch_sz = int(getattr(getattr(vision_model, "cfg", None), "patch_size", 16))
    tps = int(getattr(getattr(vision_model, "cfg", None), "temporal_patch_size", 2))
    merge_sz = int(getattr(getattr(vision_model, "cfg", None), "spatial_merge_size", 2))
    block_sz = patch_sz * merge_sz

    # 1. Preprocess every item's image list into CPU QwenVisionInputs
    flat_slots: list[tuple[int, int, QwenVisionInputs]] = []
    item_vins: list[list[QwenVisionInputs]] = [[] for _ in range(n_items)]
    item_keep_masks: list[list[np.ndarray | None]] = [[] for _ in range(n_items)]
    for i, imgs in enumerate(images_per_item):
        if not imgs:
            continue
        item_min_px = (
            int(min_pixels_per_item[i])
            if (min_pixels_per_item is not None and min_pixels_per_item[i] is not None)
            else int(min_pixels)
        )
        item_max_px = (
            int(max_pixels_per_item[i])
            if (max_pixels_per_item is not None and max_pixels_per_item[i] is not None)
            else int(max_pixels)
        )
        for j, im in enumerate(imgs):
            pil_im = load_image(im)
            vi = build_qwen_vision_inputs(
                pil_im,
                patch_size=patch_sz,
                temporal_patch_size=tps,
                merge_size=merge_sz,
                min_pixels=item_min_px,
                max_pixels=item_max_px,
            )
            item_vins[i].append(vi)
            flat_slots.append((i, j, vi))
            km_or_none: np.ndarray | None = None
            if (
                drop_white_patches
                and block_sz > 0
                and vi.resized_height > 0
                and vi.resized_width > 0
                and (vi.resized_height % block_sz) == 0
                and (vi.resized_width % block_sz) == 0
                and ((vi.resized_height // block_sz) * (vi.resized_width // block_sz)) == vi.num_visual_tokens
            ):
                km = compute_safe_white_patch_keep_mask(
                    pil_im,
                    vi.resized_height,
                    vi.resized_width,
                    patch_size=patch_sz,
                    merge_size=merge_sz,
                )
                if int(km.sum()) < vi.num_visual_tokens:
                    km_or_none = km
            item_keep_masks[i].append(km_or_none)

    # 2. Uniform-grid batched ViT encoding (B <= 16 frames per call)
    t_vit0 = time.perf_counter()
    vis_feats: list[list[mx.array | None]] = [[None] * len(item_vins[i]) for i in range(n_items)]
    if flat_slots:
        if vision_model is None:
            raise ValueError("vision_model is required when images are provided")
        groups: dict[tuple[int, int, int], list[tuple[int, int, QwenVisionInputs]]] = {}
        for rec in flat_slots:
            groups.setdefault(rec[2].grid_thw, []).append(rec)
        vit_sub_cap = max(1, min(int(batch_size), 16))
        for recs in groups.values():
            for s in range(0, len(recs), vit_sub_cap):
                sub = recs[s : s + vit_sub_cap]
                px_cat = mx.array(np.concatenate([r[2].pixel_values for r in sub], axis=0))
                grids = [r[2].grid_thw for r in sub]
                vf = vision_model(px_cat, grids)
                mx.eval(vf)  # guard: allow-eval
                cursor = 0
                for it_i, im_j, vi in sub:
                    nv = vi.num_visual_tokens
                    vis_feats[it_i][im_j] = vf[cursor : cursor + nv, :]
                    cursor += nv
        mx.synchronize()
    total_vit_ms = (time.perf_counter() - t_vit0) * 1000.0
    total_vis_toks = max(1, sum(vi.num_visual_tokens for _, _, vi in flat_slots))

    # 3. Expand prompts + fold upfront_ff BEFORE MRoPE & embedding assembly
    ans_all: list[list[int]] = [[] for _ in range(n_items)]
    hit_eos_all = [False] * n_items
    hit_max_all = [False] * n_items
    prompt_len_raw = [0] * n_items
    dropped_vis_all = [0] * n_items
    matchers_all: list[Any | None] = [
        grammars[i].new_matcher() if grammars[i] is not None else None for i in range(n_items)
    ]
    tgrams_all: list[Any | None] = (
        list(table_grammars) if table_grammars is not None else [None] * n_items
    )

    pending_meta: list[tuple[int, list[int], list[tuple[int, int]], int, mx.array, mx.array]] = []
    for i in range(n_items):
        counts_i = [vi.num_visual_tokens for vi in item_vins[i]]
        p_ids, v_spans = prompt_builder(i, counts_i)
        raw_len_i = len(p_ids)
        prompt_len_raw[i] = raw_len_i
        g_i = grammars[i]
        m_i = matchers_all[i]
        up_clean: list[int] = []
        hit_stop_up = False
        if g_i is not None and g_i.upfront_ff:
            for t in g_i.upfront_ff:
                ti = int(t)
                if ti in stop_ids:
                    hit_stop_up = True
                    break
                up_clean.append(ti)
        if len(up_clean) >= max_tokens:
            ans_all[i] = up_clean[:max_tokens]
            is_ok = len(up_clean) == max_tokens and (
                hit_stop_up or (m_i is not None and m_i.is_stopped() and m_i.is_accepting())
            )
            hit_eos_all[i] = hit_stop_up
            hit_max_all[i] = not is_ok
            continue
        if hit_stop_up or (m_i is not None and m_i.is_stopped()):
            ans_all[i] = up_clean
            hit_eos_all[i] = hit_stop_up
            continue
        if up_clean:
            ans_all[i] = list(up_clean)
            p_ids = list(p_ids) + up_clean

        pos_3d, rope_delta = compute_mrope_positions(
            p_ids,
            image_grid_thw=[vi.grid_thw for vi in item_vins[i]],
            image_token_id=int(image_token_id),
            spatial_merge_size=merge_sz,
        )
        r_cos, r_sin = precompute_mrope_trig(
            pos_3d,
            rot_dims=int(lm.cfg.rot_dims),
            rope_theta=float(lm.cfg.rope_theta),
            mrope_section=mrope_section if mrope_section is not None else getattr(lm.cfg, "mrope_section", (11, 11, 10)),
            dtype=mx.bfloat16,
        )
        eff_rope_delta = int(rope_delta)
        masks_i = item_keep_masks[i]
        if v_spans and any(m is not None for m in masks_i):
            new_p_ids: list[int] = []
            new_v_spans: list[tuple[int, int]] = []
            seq_keep_parts: list[np.ndarray] = []
            cur_pos = 0
            actual_dropped = 0
            for k_im, (s0, s1) in enumerate(v_spans):
                if s0 > cur_pos:
                    new_p_ids.extend(p_ids[cur_pos:s0])
                    seq_keep_parts.append(np.arange(cur_pos, s0, dtype=np.int32))
                m_k = masks_i[k_im] if k_im < len(masks_i) else None
                span_len = s1 - s0
                vf_k = vis_feats[i][k_im] if k_im < len(vis_feats[i]) else None
                if (
                    m_k is not None
                    and vf_k is not None
                    and len(m_k) == span_len
                    and int(vf_k.shape[0]) == span_len
                    and int(m_k.sum()) < span_len
                ):
                    kept_local = np.flatnonzero(m_k).astype(np.int32)
                    n_kept = int(kept_local.size)
                    vis_feats[i][k_im] = mx.take(vf_k, mx.array(kept_local, dtype=mx.int32), axis=0)
                    ns0 = len(new_p_ids)
                    new_p_ids.extend([int(image_token_id)] * n_kept)
                    new_v_spans.append((ns0, ns0 + n_kept))
                    seq_keep_parts.append(int(s0) + kept_local)
                    actual_dropped += span_len - n_kept
                else:
                    ns0 = len(new_p_ids)
                    new_p_ids.extend(p_ids[s0:s1])
                    new_v_spans.append((ns0, ns0 + span_len))
                    seq_keep_parts.append(np.arange(s0, s1, dtype=np.int32))
                cur_pos = s1
            if cur_pos < len(p_ids):
                new_p_ids.extend(p_ids[cur_pos:])
                seq_keep_parts.append(np.arange(cur_pos, len(p_ids), dtype=np.int32))
            if actual_dropped > 0:
                seq_keep_mx = mx.array(np.concatenate(seq_keep_parts, axis=0), dtype=mx.int32)
                r_cos = mx.take(r_cos, seq_keep_mx, axis=2)
                r_sin = mx.take(r_sin, seq_keep_mx, axis=2)
                p_ids = new_p_ids
                v_spans = new_v_spans
                eff_rope_delta = int(rope_delta) + int(actual_dropped)
                dropped_vis_all[i] = int(actual_dropped)
                prompt_len_raw[i] = raw_len_i - int(actual_dropped)
        pending_meta.append((i, p_ids, v_spans, eff_rope_delta, r_cos[0], r_sin[0]))

    pending_meta.sort(key=lambda row: len(row[1]))
    waves = pack_length_waves(
        pending_meta,
        lambda row: len(row[1]),
        batch_size,
        max_pad_ratio=max_pad_ratio,
        max_wave_tokens=max_wave_tokens,
    )

    rngs_all = [
        np.random.default_rng(int(slot_seeds[i])) if (slot_seeds is not None and slot_seeds[i] is not None) else None
        for i in range(n_items)
    ]
    tels_all: list[list[dict[str, float]] | None] = [
        ([] if collect_telemetry else None) for _ in range(n_items)
    ]
    pf_ms_item = [0.0] * n_items
    dec_ms_item = [0.0] * n_items
    pf_ms_total = 0.0
    dec_ms_total = 0.0
    compactions = 0
    vocab_size = int(tokenizer.vocab_size)
    chunk_sz = max(64, int(prefill_chunk_size))

    for w_idx, wave in enumerate(waves):
        b_sz = len(wave)
        prompt_lens = [len(r[1]) for r in wave]
        min_s = min(prompt_lens)
        max_s = max(prompt_lens)
        any_ragged = any(sl < max_s for sl in prompt_lens)

        t_pf0 = time.perf_counter()
        emb_rows: list[mx.array] = []
        cos_rows: list[mx.array] = []
        sin_rows: list[mx.array] = []
        valid_2d = np.zeros((b_sz, max_s, 1), dtype=np.float32)

        for b_i, (orig_i, p_ids, v_spans, _rd, c_i, s_i) in enumerate(wave):
            l_i = len(p_ids)
            pad_len = max_s - l_i
            base_i = lm.embed_tokens(mx.array([p_ids], dtype=mx.int32))[0]
            vf_list = vis_feats[orig_i]
            if v_spans:
                parts: list[mx.array] = []
                vf_k = 0
                tok_cur = 0
                for s0, s1 in v_spans:
                    if s0 > tok_cur:
                        parts.append(base_i[tok_cur:s0, :])
                    vf_k_arr = vf_list[vf_k]
                    assert vf_k_arr is not None
                    parts.append(vf_k_arr.astype(base_i.dtype))
                    vf_k += 1
                    tok_cur = s1
                if tok_cur < l_i:
                    parts.append(base_i[tok_cur:, :])
                comb = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            else:
                comb = base_i
            if pad_len > 0:
                h_dim = int(comb.shape[-1])
                r_dim = int(c_i.shape[-1])
                comb = mx.concatenate([comb, mx.zeros((pad_len, h_dim), dtype=comb.dtype)], axis=0)
                c_i = mx.concatenate([c_i, mx.ones((1, pad_len, r_dim), dtype=c_i.dtype)], axis=1)
                s_i = mx.concatenate([s_i, mx.zeros((1, pad_len, r_dim), dtype=s_i.dtype)], axis=1)
            emb_rows.append(comb)
            cos_rows.append(c_i)
            sin_rows.append(s_i)
            valid_2d[b_i, :l_i, 0] = 1.0

        wave_embeds = mx.stack(emb_rows, axis=0)
        wave_cos = mx.stack(cos_rows, axis=0)
        wave_sin = mx.stack(sin_rows, axis=0)
        valid_scale = mx.array(valid_2d, dtype=mx.bfloat16) if any_ragged else None

        est_ctx = max_s + max_tokens + 64
        use_kvi8: KVMode | bool = (
            kv_i8_resolver(b_sz, est_ctx) if callable(kv_i8_resolver) else kv_i8_resolver
        )
        eff_kv_step = max(256, min(int(kv_step), est_ctx))
        cache = _make_hybrid_cache(
            lm,
            kv_step=eff_kv_step,
            kv_i8=use_kvi8,
            gdn_state_mode=gdn_state_mode,
            batch_size=b_sz,
        )

        slot_cur_h: list[mx.array | None] = [None] * b_sz
        for c0 in range(0, max_s, chunk_sz):
            c1 = min(max_s, c0 + chunk_sz)
            emb_c = wave_embeds[:, c0:c1, :]
            trig_c = (wave_cos[:, :, c0:c1, :], wave_sin[:, :, c0:c1, :])
            chunk_has_pad = c1 > min_s
            v_scale_c = valid_scale[:, c0:c1, :] if (chunk_has_pad and valid_scale is not None) else None
            v_lens_c = (
                mx.array([max(0, min(sl - c0, c1 - c0)) for sl in prompt_lens], dtype=mx.int32)
                if chunk_has_pad
                else None
            )
            if chunk_has_pad:
                q_kpos = mx.arange(c0, c1, dtype=mx.int32)[None, None, :, None]
                k_pos_c = mx.arange(c1, dtype=mx.int32)[None, None, None, :]
                suf_end_c = mx.array(prompt_lens, dtype=mx.int32)[:, None, None, None]
                allow_c = (k_pos_c < suf_end_c) & (k_pos_c <= q_kpos)
                mask_c: str | mx.array = mx.where(
                    allow_c,
                    mx.array(0.0, dtype=mx.bfloat16),
                    mx.array(-1e9, dtype=mx.bfloat16),
                )
            else:
                mask_c = "causal"

            ch_h = lm.forward_hidden(
                None,
                cache,
                inputs_embeds=emb_c,
                valid_token_scale=v_scale_c,
                valid_lengths=v_lens_c,
                attn_mask=mask_c,
                rope_trig=trig_c,
            )
            mx.eval(ch_h)  # guard: allow-eval
            for b_i, sl in enumerate(prompt_lens):
                pos_i = max(0, sl - 1)
                if c0 <= pos_i < c1:
                    slot_cur_h[b_i] = ch_h[b_i, pos_i - c0, :]

        cur_h = mx.stack([slot_cur_h[b_i] for b_i in range(b_sz)], axis=0)  # type: ignore[misc]
        logits0 = lm.project_logits(cur_h[:, None, :])[:, 0, :]
        mx.eval(cur_h, logits0)  # guard: allow-eval
        mx.synchronize()
        pf_ms_w = (time.perf_counter() - t_pf0) * 1000.0
        pf_ms_total += pf_ms_w
        for r_w in wave:
            pf_ms_item[r_w[0]] = pf_ms_w / max(1, b_sz)

        del wave_embeds, wave_cos, wave_sin, emb_rows, cos_rows, sin_rows, slot_cur_h

        t_dec0 = time.perf_counter()
        done_w = [False] * b_sz
        pending_q: list[list[int]] = [[] for _ in range(b_sz)]
        arr_logits0 = np.asarray(logits0.astype(mx.float32))
        cur_logits_map: dict[int, np.ndarray] = {b_i: arr_logits0[b_i] for b_i in range(b_sz)}
        true_pos = [len(r[1]) + int(r[3]) for r in wave]
        kv_cursor = max_s
        live_rows = list(range(b_sz))
        any_ragged_live = any_ragged
        suf_end_arr = (
            mx.array([prompt_lens[w_i] for w_i in live_rows], dtype=mx.int32)[:, None, None, None]
            if any_ragged_live
            else None
        )

        def _advance_slot_token(w_i: int, logits_1d: np.ndarray, h_1d: mx.array) -> None:
            orig_i = wave[w_i][0]
            m_i = matchers_all[orig_i]
            g_i = grammars[orig_i]
            tg_i = tgrams_all[orig_i]
            if len(ans_all[orig_i]) >= max_tokens:
                done_w[w_i] = True
                hit_max_all[orig_i] = not (m_i is not None and m_i.is_stopped() and m_i.is_accepting())
                return
            if m_i is not None and m_i.is_stopped():
                done_w[w_i] = True
                return

            tid, tel = select_next_token_1d(
                logits_1d,
                lm=lm,
                new_to_old=new_to_old,
                old_to_new=old_to_new,
                stop_token_ids=stop_ids,
                vocab_size=vocab_size,
                full_hidden_1d=h_1d,
                matcher=m_i,
                gram=g_i,
                table_gram=tg_i,
                recent_tokens=ans_all[orig_i],
                temperature=temperature,
                top_p=top_p,
                rng=rngs_all[orig_i],
                repeat_penalty=repeat_penalty,
                collect_telemetry=collect_telemetry,
            )
            if tel is not None and tels_all[orig_i] is not None:
                tels_all[orig_i].append(tel)  # type: ignore[union-attr]
            if tid in stop_ids:
                done_w[w_i] = True
                hit_eos_all[orig_i] = True
                return
            ans_all[orig_i].append(tid)
            ff_clean: list[int] = []
            stopped_ff = False
            if m_i is not None:
                m_i.consume_token(tid)
                if m_i.is_stopped():
                    done_w[w_i] = True
                    return
                if len(ans_all[orig_i]) >= max_tokens:
                    done_w[w_i] = True
                    hit_max_all[orig_i] = True
                    return
                for ft in m_i.compute_ff_tokens():
                    fti = int(ft)
                    if fti in stop_ids:
                        stopped_ff = True
                        break
                    if len(ans_all[orig_i]) + len(ff_clean) >= max_tokens:
                        break
                    ff_clean.append(fti)
                if ff_clean:
                    m_i.consume_tokens(ff_clean)
                    ans_all[orig_i].extend(ff_clean)
                if stopped_ff or m_i.is_stopped():
                    done_w[w_i] = True
                    hit_eos_all[orig_i] = stopped_ff
                    return
            elif tg_i is not None:
                delta_str = tokenizer.decode([tid], skip_special_tokens=False)
                for fti_raw in tg_i.advance_text_and_maybe_lock(delta_str, tid):
                    fti = int(fti_raw)
                    if fti in stop_ids:
                        stopped_ff = True
                        break
                    if len(ans_all[orig_i]) + len(ff_clean) >= max_tokens:
                        break
                    ff_clean.append(fti)
                if ff_clean:
                    ans_all[orig_i].extend(ff_clean)
                if stopped_ff:
                    done_w[w_i] = True
                    hit_eos_all[orig_i] = True
                    return
            if len(ans_all[orig_i]) >= max_tokens:
                done_w[w_i] = True
                hit_max_all[orig_i] = True
                return
            pending_q[w_i] = [tid] + ff_clean

        for _step in range(max_tokens + 2):
            cur_b = len(live_rows)
            need_sample_r = [
                r for r, w_i in enumerate(live_rows) if not done_w[w_i] and not pending_q[w_i]
            ]
            for r in need_sample_r:
                _advance_slot_token(live_rows[r], cur_logits_map[r], cur_h[r])

            alive_r = [
                r for r, w_i in enumerate(live_rows) if not done_w[w_i] and pending_q[w_i]
            ]
            if not alive_r:
                break

            if compact_active and cur_b >= 4 and len(alive_r) < cur_b:
                if len(alive_r) <= (cur_b * 5) // 8 or (cur_b - len(alive_r)) >= 4:
                    compact_hybrid_cache(cache, alive_r)
                    live_rows = [live_rows[r] for r in alive_r]
                    cur_b = len(live_rows)
                    alive_r = list(range(cur_b))
                    any_ragged_live = any(prompt_lens[w_i] < max_s for w_i in live_rows)
                    suf_end_arr = (
                        mx.array([prompt_lens[w_i] for w_i in live_rows], dtype=mx.int32)[:, None, None, None]
                        if any_ragged_live
                        else None
                    )
                    compactions += 1

            step_w = min(len(pending_q[live_rows[r]]) for r in alive_r)
            step_rows: list[list[int]] = []
            for r in range(cur_b):
                w_i = live_rows[r]
                if done_w[w_i] or not pending_q[w_i]:
                    step_rows.append([pad_token_id] * step_w)
                else:
                    step_rows.append(pending_q[w_i][:step_w])
                    del pending_q[w_i][:step_w]

            step_inp = mx.array(step_rows, dtype=mx.int32)
            kv_prev = kv_cursor
            kv_now = kv_prev + step_w
            kv_cursor = kv_now

            p_mat = np.array([true_pos[w_i] for w_i in live_rows], dtype=np.int32)[:, None] + np.arange(step_w, dtype=np.int32)[None, :]
            r_trig_s = lm.precompute_rope_trig(mx.array(p_mat, dtype=mx.int32))
            if any_ragged_live:
                assert suf_end_arr is not None
                k_pos = mx.arange(kv_now, dtype=mx.int32)[None, None, None, :]
                in_pad = (k_pos >= suf_end_arr) & (k_pos < max_s)
                if step_w == 1:
                    allow = ~in_pad
                else:
                    q_kv = mx.arange(step_w, dtype=mx.int32)[None, None, :, None] + kv_prev
                    allow = (~in_pad) & (k_pos <= q_kv)
                d_mask_s: str | mx.array = mx.where(
                    allow,
                    mx.array(0.0, dtype=mx.bfloat16),
                    mx.array(-1e9, dtype=mx.bfloat16),
                )
            else:
                d_mask_s = "causal"

            v_scale_s = (
                mx.array(
                    [[[0.0] if done_w[live_rows[r]] else [1.0]] * step_w for r in range(cur_b)],
                    dtype=mx.bfloat16,
                )
                if len(alive_r) < cur_b
                else None
            )
            sh = lm.forward_hidden(
                step_inp,
                cache,
                valid_token_scale=v_scale_s,
                attn_mask=d_mask_s,
                rope_trig=r_trig_s,
            )
            cur_h = sh[:, -1, :]
            need_next_r = [
                r for r, w_i in enumerate(live_rows) if not done_w[w_i] and not pending_q[w_i]
            ]
            cur_logits_map = {}
            if not need_next_r:
                mx.eval(cur_h)  # guard: allow-eval
            elif len(need_next_r) == cur_b:
                sl_all = lm.project_logits(cur_h[:, None, :])[:, 0, :]
                mx.eval(cur_h, sl_all)  # guard: allow-eval
                arr_all = np.asarray(sl_all.astype(mx.float32))
                cur_logits_map = {r: arr_all[r] for r in range(cur_b)}
            else:
                idx_mx = mx.array(need_next_r, dtype=mx.int32)
                sub_h = mx.take(cur_h, idx_mx, axis=0)[:, None, :]
                sl_sub = lm.project_logits(sub_h)[:, 0, :]
                mx.eval(cur_h, sl_sub)  # guard: allow-eval
                arr_sub = np.asarray(sl_sub.astype(mx.float32))
                cur_logits_map = {r: arr_sub[pos] for pos, r in enumerate(need_next_r)}

            for w_i in live_rows:
                if not done_w[w_i]:
                    true_pos[w_i] += step_w

        mx.synchronize()
        dec_ms_w = (time.perf_counter() - t_dec0) * 1000.0
        dec_ms_total += dec_ms_w
        for r_w in wave:
            dec_ms_item[r_w[0]] = dec_ms_w / max(1, b_sz)
        del cache
        if w_idx + 1 < len(waves) and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    results = [
        BatchedMMItemResult(
            generated_ids=ans_all[i],
            prompt_tokens=prompt_len_raw[i],
            hit_eos=hit_eos_all[i],
            hit_max_tokens=hit_max_all[i],
            vit_ms=(
                total_vit_ms * (sum(vi.num_visual_tokens for vi in item_vins[i]) / total_vis_toks)
                if item_vins[i]
                else 0.0
            ),
            prefill_ms=pf_ms_item[i],
            decode_ms=dec_ms_item[i],
            image_grids=[vi.grid_thw for vi in item_vins[i]],
            telemetry=tels_all[i],
            dropped_visual_tokens=dropped_vis_all[i],
        )
        for i in range(n_items)
    ]
    return results, {
        "waves": len(waves),
        "compactions": compactions,
        "vit_ms": round(total_vit_ms, 2),
        "prefill_ms": round(pf_ms_total, 2),
        "decode_ms": round(dec_ms_total, 2),
        "dropped_visual_tokens": int(sum(dropped_vis_all)),
    }


__all__ = [
    "BatchedMMItemResult",
    "batched_multimodal_generate",
    "compute_safe_white_patch_keep_mask",
    "compute_step_telemetry",
    "default_wave_token_budget",
    "host_phys_memory_gb",
    "pack_length_waves",
    "select_next_token_1d",
]
