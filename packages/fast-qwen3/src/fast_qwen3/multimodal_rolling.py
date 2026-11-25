"""Batched Speculative PLD (`B=2..8`) and Continuous Rolling Multimodal Batching for `fast-qwen3`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import mlx.core as mx
import numpy as np

from fast_shared.grammar import safe_argmax_1d
from fast_shared.kv_quant import KVMode
from fast_shared.qwen35_vision import compute_mrope_positions, precompute_mrope_trig
from fast_shared.vision_preprocess import QwenVisionInputs, build_qwen_vision_inputs, load_image

from .gdn_kernels import gdn_delta_update, resolve_gdn_state_mode
from .kernels import gpu_argmax_nan
from .model import DeltaSlotCache, GrowingKVCache, compact_hybrid_cache

if TYPE_CHECKING:
    from .multimodal_batch import BatchedMMItemResult


def scatter_kv_and_rewind(
    cache: Sequence[DeltaSlotCache | GrowingKVCache],
    slot_kv_len: list[int],
    n_acc: list[int],
    kv_prev: int,
    l_q: int,
    cur_b: int,
) -> list[int]:
    """Maintain contiguous `[0, slot_kv_len[r])` KV layout per slot after batched verify/decode."""
    new_slot_kv_len = [int(slot_kv_len[r]) + int(n_acc[r]) for r in range(cur_b)]
    new_max_kv = max(new_slot_kv_len) if new_slot_kv_len else 0

    all_aligned = all(sl == kv_prev for sl in slot_kv_len)
    all_full = all(na == l_q for na in n_acc)
    if all_aligned and all_full:
        return new_slot_kv_len

    need_scatter = any(slot_kv_len[r] < kv_prev and n_acc[r] > 0 for r in range(cur_b))
    if need_scatter:
        b_3d = mx.arange(cur_b, dtype=mx.int32)[:, None, None]
        t_3d = (
            mx.array(slot_kv_len, dtype=mx.int32)[:, None, None]
            + mx.arange(l_q, dtype=mx.int32)[None, None, :]
        )
        to_eval: list[mx.array] = []
        for entry in cache:
            if not isinstance(entry, GrowingKVCache):
                continue
            qc = entry._qcache
            if qc is not None:
                for attr in ("k_w", "k_s", "k_b", "v_w", "v_s", "v_b"):
                    arr = getattr(qc, attr, None)
                    if arr is not None:
                        h_3d = mx.arange(int(arr.shape[1]), dtype=mx.int32)[None, :, None]
                        chunk = mx.contiguous(arr[:, :, kv_prev : kv_prev + l_q, :])
                        arr[b_3d, h_3d, t_3d] = chunk
                        to_eval.append(arr)
            else:
                assert entry.keys is not None and entry.values is not None
                h_3d = mx.arange(int(entry.keys.shape[1]), dtype=mx.int32)[None, :, None]
                k_chunk = mx.contiguous(entry.keys[:, :, kv_prev : kv_prev + l_q, :])
                v_chunk = mx.contiguous(entry.values[:, :, kv_prev : kv_prev + l_q, :])
                entry.keys[b_3d, h_3d, t_3d] = k_chunk
                entry.values[b_3d, h_3d, t_3d] = v_chunk
                to_eval.extend([entry.keys, entry.values])
            entry.rewind_to(new_max_kv)
        if to_eval:
            mx.eval(*to_eval)  # guard: allow-eval
    else:
        for entry in cache:
            if isinstance(entry, GrowingKVCache):
                entry.rewind_to(new_max_kv)
    return new_slot_kv_len


def rollback_gdn_batched(
    cache: Sequence[DeltaSlotCache | GrowingKVCache],
    n_acc: list[int],
    l_q: int,
    conv_k: int,
    cur_b: int,
) -> None:
    """Zero-GEMM per-slot GDN state freeze via `gdn_delta_update(..., mask=...)` + exact conv slice."""
    if all(na == l_q for na in n_acc):
        for entry in cache:
            if isinstance(entry, DeltaSlotCache):
                entry.clear_verify_trace()
        return

    max_acc = max(n_acc) if n_acc else 0
    if max_acc <= 0:
        to_eval_zero: list[mx.array] = []
        for entry in cache:
            if not isinstance(entry, DeltaSlotCache):
                continue
            if entry._verify_trace is not None:
                conv_in, _, _, _, _, _, prev_rec = entry._verify_trace
                entry.conv_state = mx.contiguous(conv_in[:, : conv_k - 1, :])
                to_eval_zero.append(entry.conv_state)
                entry._store_rec_output(prev_rec)
                if isinstance(prev_rec, tuple):
                    to_eval_zero.extend([prev_rec[0], prev_rec[1]])
                elif prev_rec is not None:
                    to_eval_zero.append(prev_rec)
            entry.clear_verify_trace()
        if to_eval_zero:
            mx.eval(*to_eval_zero)  # guard: allow-eval
        return

    mask_np = np.zeros((cur_b, max_acc), dtype=np.uint8)
    for r, na in enumerate(n_acc):
        if na > 0:
            mask_np[r, :na] = 1
    mask_mx = mx.array(mask_np, dtype=mx.uint8)
    n_acc_mx = mx.array(n_acc, dtype=mx.int32)
    tap_offsets = mx.arange(conv_k - 1, dtype=mx.int32)[None, :, None]
    gather_idx = n_acc_mx[:, None, None] + tap_offsets

    to_eval: list[mx.array] = []
    for entry in cache:
        if not isinstance(entry, DeltaSlotCache):
            continue
        if entry._verify_trace is None:
            continue
        conv_in, q, k, v, g, beta, prev_rec = entry._verify_trace
        entry.conv_state = mx.contiguous(mx.take_along_axis(conv_in, gather_idx, axis=1))
        to_eval.append(entry.conv_state)
        _, new_rec = gdn_delta_update(
            q[:, :max_acc],
            k[:, :max_acc],
            v[:, :max_acc],
            g[:, :max_acc],
            beta[:, :max_acc],
            prev_rec,
            mask=mask_mx,
            state_mode=entry.state_mode,
        )
        entry._store_rec_output(new_rec)
        if isinstance(new_rec, tuple):
            to_eval.extend([new_rec[0], new_rec[1]])
        elif new_rec is not None:
            to_eval.append(new_rec)
        entry.clear_verify_trace()
    if to_eval:
        mx.eval(*to_eval)  # guard: allow-eval


def build_ragged_attn_mask(
    slot_kv_len: list[int],
    kv_prev: int,
    l_q: int,
) -> str | mx.array:
    """Build exact BF16 additive mask allowing `[0, slot_kv_len[r])` and causal `[kv_prev, kv_prev + l_q)`."""
    if all(sl == kv_prev for sl in slot_kv_len):
        return "causal"
    k_pos = mx.arange(kv_prev + l_q, dtype=mx.int32)[None, None, None, :]
    slot_end_arr = mx.array(slot_kv_len, dtype=mx.int32)[:, None, None, None]
    if l_q == 1:
        allow = (k_pos < slot_end_arr) | (k_pos == kv_prev)
    else:
        q_pos = mx.arange(l_q, dtype=mx.int32)[None, None, :, None] + kv_prev
        allow = (k_pos < slot_end_arr) | ((k_pos >= kv_prev) & (k_pos <= q_pos))
    return mx.where(allow, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16))


def scatter_single_item_cache_into_slot(
    active_cache: list[DeltaSlotCache | GrowingKVCache],
    item_cache: list[DeltaSlotCache | GrowingKVCache],
    slot_r: int,
    b_slots: int,
    slot_kv_len: list[int],
    new_len: int,
    kv_step: int,
) -> None:
    """Scatter a `B=1` prefilled hybrid cache into slot `slot_r` of `active_cache`."""
    to_eval: list[mx.array] = []
    other_max = max((slot_kv_len[i] for i in range(b_slots) if i != slot_r), default=0)
    new_max_kv = max(other_max, int(new_len))

    for ac, ic in zip(active_cache, item_cache):
        if isinstance(ac, DeltaSlotCache) and isinstance(ic, DeltaSlotCache):
            if ic.conv_state is not None:
                if ac.conv_state is None:
                    _, c_k_1, c_dim = ic.conv_state.shape
                    ac.conv_state = mx.zeros((b_slots, int(c_k_1), int(c_dim)), dtype=ic.conv_state.dtype)
                ac.conv_state[slot_r : slot_r + 1] = ic.conv_state
                to_eval.append(ac.conv_state)

            if ac.state_mode == "int8_blk32":
                if ic.rec_codes is not None and ic.rec_scales is not None:
                    if ac.rec_codes is None or ac.rec_scales is None:
                        _, hv, dv, dk = ic.rec_codes.shape
                        _, _, _, nq = ic.rec_scales.shape
                        ac.rec_codes = mx.zeros((b_slots, int(hv), int(dv), int(dk)), dtype=mx.int8)
                        ac.rec_scales = mx.zeros((b_slots, int(hv), int(dv), int(nq)), dtype=mx.bfloat16)
                    ac.rec_codes[slot_r : slot_r + 1] = ic.rec_codes
                    ac.rec_scales[slot_r : slot_r + 1] = ic.rec_scales
                    to_eval.extend([ac.rec_codes, ac.rec_scales])
            else:
                if ic._rec_state is not None:
                    if ac._rec_state is None:
                        _, hv, dv, dk = ic._rec_state.shape
                        ac._rec_state = mx.zeros((b_slots, int(hv), int(dv), int(dk)), dtype=ic._rec_state.dtype)
                    ac._rec_state[slot_r : slot_r + 1] = ic._rec_state
                    to_eval.append(ac._rec_state)
        elif isinstance(ac, GrowingKVCache) and isinstance(ic, GrowingKVCache):
            if ac._qcache is not None and ic._qcache is not None:
                iq = ic._qcache
                if iq.k_w is not None and iq.k_s is not None and new_len > 0:
                    _, h_kv, _, pack_d = iq.k_w.shape
                    d_head = (int(pack_d) * 32) // int(iq.bits)
                    needed_cap = max(new_max_kv, kv_step)
                    ac._qcache._ensure_capacity(b_slots, int(h_kv), int(needed_cap), d_head, iq.k_s.dtype)
                    for attr in ("k_w", "k_s", "k_b", "v_w", "v_s", "v_b"):
                        a_arr = getattr(ac._qcache, attr, None)
                        i_arr = getattr(iq, attr, None)
                        if a_arr is not None and i_arr is not None:
                            a_arr[slot_r : slot_r + 1, :, :new_len, :] = i_arr[:, :, :new_len, :]
                            to_eval.append(a_arr)
                ac._qcache.offset = int(new_max_kv)
            elif ac._qcache is None and ic._qcache is None:
                if ic.keys is not None and ic.values is not None and new_len > 0:
                    _, h_kv, _, d = ic.keys.shape
                    needed_cap = max(new_max_kv, kv_step)
                    ac._ensure_capacity(b_slots, int(h_kv), int(needed_cap), int(d), ic.keys.dtype)
                    assert ac.keys is not None and ac.values is not None
                    ac.keys[slot_r : slot_r + 1, :, :new_len, :] = ic.keys[:, :, :new_len, :]
                    ac.values[slot_r : slot_r + 1, :, :new_len, :] = ic.values[:, :, :new_len, :]
                    to_eval.extend([ac.keys, ac.values])
                ac.offset = int(new_max_kv)
            else:
                raise TypeError("Mismatched GrowingKVCache quantization modes in slot scatter")
        else:
            raise TypeError(f"Mismatched cache types {type(ac)} vs {type(ic)}")

    slot_kv_len[slot_r] = int(new_len)
    if to_eval:
        mx.eval(*to_eval)  # guard: allow-eval


def rolling_multimodal_generate(
    *,
    lm: Any,
    vision_model: Any | None,
    tokenizer: Any,
    images_per_item: Sequence[Sequence[Any] | None],
    prompt_builder: Callable[[int, list[int]], tuple[list[int], list[tuple[int, int]]]],
    image_token_id: int,
    new_to_old: np.ndarray,
    drafters: Sequence[Any | None] | None = None,
    spec_modes: Sequence[str] | None = None,
    draft_k: int = 4,
    use_pld: bool = True,
    max_tokens: int = 512,
    batch_size: int = 8,
    rolling_max_batch_size: int = 8,
    kv_step: int = 512,
    kv_i8_resolver: Callable[[int, int], KVMode] | KVMode = "bf16",
    gdn_state_mode: str = "auto",
    stop_ids: frozenset[int],
    pad_token_id: int,
    prefill_chunk_size: int = 512,
    mrope_section: tuple[int, int, int] | None = None,
    compact_tail: bool = True,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    min_pixels_per_item: Sequence[int | None] | None = None,
    max_pixels_per_item: Sequence[int | None] | None = None,
) -> tuple[list[BatchedMMItemResult], dict[str, Any]]:
    """Continuous rolling multimodal decode with immediate finished-slot admission and batched speculative PLD."""
    from .multimodal_batch import BatchedMMItemResult, _make_hybrid_cache

    n_items = len(images_per_item)
    if n_items == 0:
        return [], {
            "rolling": True,
            "use_pld": bool(use_pld),
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

    # 1. Preprocess images and run grid-bucketed ViT
    flat_slots: list[tuple[int, int, QwenVisionInputs]] = []
    item_vins: list[list[QwenVisionInputs]] = [[] for _ in range(n_items)]
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

    # 2. Build prompt metadata preserving caller submission order
    pending_meta: list[tuple[int, list[int], list[tuple[int, int]], int, mx.array, mx.array]] = []
    prompt_len_raw = [0] * n_items
    default_mrope = getattr(lm.cfg, "mrope_section", (11, 11, 10))
    eff_mrope = mrope_section if mrope_section is not None else default_mrope
    for i in range(n_items):
        counts_i = [vi.num_visual_tokens for vi in item_vins[i]]
        p_ids, v_spans = prompt_builder(i, counts_i)
        prompt_len_raw[i] = len(p_ids)
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
            mrope_section=eff_mrope,
            dtype=mx.bfloat16,
        )
        pending_meta.append((i, p_ids, v_spans, int(rope_delta), r_cos[0], r_sin[0]))

    b_slots = max(1, min(int(batch_size), max(1, int(rolling_max_batch_size)), n_items))
    max_prompt_len = max((len(r[1]) for r in pending_meta), default=1)
    est_ctx = max_prompt_len + int(max_tokens) + 64
    use_kvi8: KVMode | bool = (
        kv_i8_resolver(b_slots, est_ctx) if callable(kv_i8_resolver) else kv_i8_resolver
    )
    eff_gdn_mode = resolve_gdn_state_mode(
        gdn_state_mode,
        batch_size=b_slots,
        head_k_dim=int(lm.cfg.linear_key_head_dim),
    )
    active_cache = _make_hybrid_cache(
        lm,
        kv_step=kv_step,
        kv_i8=use_kvi8,
        gdn_state_mode=eff_gdn_mode,
        batch_size=b_slots,
    )

    ans_all: list[list[int]] = [[] for _ in range(n_items)]
    hit_eos_all = [False] * n_items
    hit_max_all = [False] * n_items
    drafted_all = [0] * n_items
    accepted_all = [0] * n_items
    verify_passes_all = [0] * n_items
    pf_ms_item = [0.0] * n_items
    dec_ms_item = [0.0] * n_items
    pf_ms_total = 0.0

    drafters_all = list(drafters) if drafters is not None else [None] * n_items
    spec_modes_all = list(spec_modes) if spec_modes is not None else ["off"] * n_items
    eff_use_pld = bool(use_pld and any(d is not None and sm != "off" for d, sm in zip(drafters_all, spec_modes_all)))
    vocab_remap = getattr(lm, "vocab_remap", None)
    vocab_n = len(new_to_old)
    chunk_sz = max(64, int(prefill_chunk_size))
    conv_k = int(lm.cfg.linear_conv_kernel_dim)

    slot_orig_idx: list[int] = [-1] * b_slots
    slot_done: list[bool] = [True] * b_slots
    slot_pending: list[list[int]] = [[] for _ in range(b_slots)]
    slot_true_pos: list[int] = [0] * b_slots
    slot_kv_len: list[int] = [0] * b_slots

    def _prefill_and_admit(
        slot_r: int,
        meta_row: tuple[int, list[int], list[tuple[int, int]], int, mx.array, mx.array],
    ) -> None:
        nonlocal pf_ms_total
        orig_i, p_ids, v_spans, rope_delta_i, c_i, s_i = meta_row
        l_i = len(p_ids)
        t_p0 = time.perf_counter()

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

        emb_1 = comb[None, :, :]
        cos_1 = c_i[None, :, :, :]
        sin_1 = s_i[None, :, :, :]

        item_cache = _make_hybrid_cache(
            lm,
            kv_step=max(256, min(kv_step, l_i + 64)),
            kv_i8=use_kvi8,
            gdn_state_mode=eff_gdn_mode,
            batch_size=1,
        )
        last_h: mx.array | None = None
        for c0 in range(0, l_i, chunk_sz):
            c1 = min(l_i, c0 + chunk_sz)
            ch = lm.forward_hidden(
                None,
                item_cache,
                inputs_embeds=emb_1[:, c0:c1, :],
                attn_mask="causal",
                rope_trig=(cos_1[:, :, c0:c1, :], sin_1[:, :, c0:c1, :]),
            )
            mx.eval(ch)  # guard: allow-eval
            last_h = ch[:, -1:, :]
        assert last_h is not None
        logits0 = lm.project_logits(last_h)[:, 0, :]
        if vocab_remap is None and int(logits0.shape[-1]) > vocab_n:
            logits0 = logits0[:, :vocab_n]
        idx0_mx = gpu_argmax_nan(logits0)
        mx.eval(idx0_mx)  # guard: allow-eval
        idx0 = int(idx0_mx.tolist()[0])  # guard: allow-sync
        if 0 <= idx0 < vocab_n:
            first_tok = int(new_to_old[idx0])
        else:
            row_f32 = logits0[0].astype(mx.float32)
            mx.eval(row_f32)  # guard: allow-eval
            first_tok = int(new_to_old[safe_argmax_1d(np.asarray(row_f32))])  # guard: allow-sync

        scatter_single_item_cache_into_slot(
            active_cache,
            item_cache,
            slot_r,
            len(slot_orig_idx),
            slot_kv_len,
            l_i,
            kv_step,
        )
        del item_cache, emb_1, cos_1, sin_1, comb, base_i
        vis_feats[orig_i] = []

        mx.synchronize()
        pf_ms = (time.perf_counter() - t_p0) * 1000.0
        pf_ms_total += pf_ms
        pf_ms_item[orig_i] = pf_ms

        slot_orig_idx[slot_r] = orig_i
        slot_true_pos[slot_r] = l_i + rope_delta_i
        if first_tok in stop_ids:
            hit_eos_all[orig_i] = True
            slot_done[slot_r] = True
            slot_pending[slot_r] = []
        else:
            ans_all[orig_i].append(first_tok)
            if len(ans_all[orig_i]) >= max_tokens:
                hit_max_all[orig_i] = True
                slot_done[slot_r] = True
                slot_pending[slot_r] = []
            else:
                slot_done[slot_r] = False
                slot_pending[slot_r] = [first_tok]

    queue_idx = 0
    for r in range(b_slots):
        while queue_idx < n_items and slot_done[r]:
            _prefill_and_admit(r, pending_meta[queue_idx])
            queue_idx += 1

    dec_ms_acc = 0.0
    total_decode_steps = 0
    slot_step_capacity = 0
    active_slot_steps = 0
    replacements = 0
    compactions = 0

    while True:
        admitted_any = False
        for r in range(len(slot_orig_idx)):
            while slot_done[r] and queue_idx < n_items:
                _prefill_and_admit(r, pending_meta[queue_idx])
                replacements += 1
                queue_idx += 1
                admitted_any = True
        if admitted_any and hasattr(mx, "clear_cache"):
            mx.clear_cache()

        cur_b = len(slot_orig_idx)
        alive_r = [r for r in range(cur_b) if not slot_done[r] and slot_pending[r]]
        if not alive_r:
            break

        if compact_tail and queue_idx >= n_items and cur_b >= 4 and len(alive_r) < cur_b:
            if len(alive_r) <= (cur_b * 5) // 8 or (cur_b - len(alive_r)) >= 4:
                compact_hybrid_cache(active_cache, alive_r)
                slot_orig_idx = [slot_orig_idx[r] for r in alive_r]
                slot_done = [slot_done[r] for r in alive_r]
                slot_pending = [slot_pending[r] for r in alive_r]
                slot_true_pos = [slot_true_pos[r] for r in alive_r]
                slot_kv_len = [slot_kv_len[r] for r in alive_r]
                cur_b = len(slot_orig_idx)
                alive_r = list(range(cur_b))
                new_max_kv = max(slot_kv_len) if slot_kv_len else 0
                for entry in active_cache:
                    if isinstance(entry, GrowingKVCache) and entry.offset > new_max_kv:
                        entry.rewind_to(new_max_kv)
                compactions += 1

        t_step0 = time.perf_counter()
        total_decode_steps += 1
        slot_step_capacity += cur_b
        active_slot_steps += len(alive_r)

        drafts_r: list[list[int]] = [[] for _ in range(cur_b)]
        if eff_use_pld:
            for r in range(cur_b):
                if slot_done[r] or len(slot_pending[r]) != 1:
                    continue
                orig_i = slot_orig_idx[r]
                dr = drafters_all[orig_i]
                sp_mode = spec_modes_all[orig_i]
                rem_r = max_tokens - len(ans_all[orig_i])
                if dr is not None and sp_mode != "off" and rem_r >= 2:
                    drafts_r[r] = dr.draft(ans_all[orig_i], mode=sp_mode, max_k=min(draft_k, rem_r - 1))

        k_max = max((len(d) for d in drafts_r), default=0) if eff_use_pld else 0
        l_q = (k_max + 1) if k_max > 0 else 1

        step_rows: list[list[int]] = []
        valid_masks: list[list[float]] = []
        has_invalid = False
        for r in range(cur_b):
            if slot_done[r] or not slot_pending[r]:
                step_rows.append([pad_token_id] * l_q)
                valid_masks.append([0.0] * l_q)
                has_invalid = True
            elif k_max > 0:
                seq_r = [slot_pending[r][0]] + drafts_r[r]
                pad_r = l_q - len(seq_r)
                if pad_r > 0:
                    step_rows.append(seq_r + [pad_token_id] * pad_r)
                    valid_masks.append([1.0] * len(seq_r) + [0.0] * pad_r)
                    has_invalid = True
                else:
                    step_rows.append(seq_r)
                    valid_masks.append([1.0] * l_q)
            else:
                step_rows.append([slot_pending[r][0]])
                valid_masks.append([1.0])

        step_inp = mx.array(step_rows, dtype=mx.int32)
        kv_prev = max(slot_kv_len) if slot_kv_len else 0
        p_mat = (
            np.array(slot_true_pos, dtype=np.int32)[:, None]
            + np.arange(l_q, dtype=np.int32)[None, :]
        )
        r_trig_s = lm.precompute_rope_trig(mx.array(p_mat, dtype=mx.int32))
        d_mask_s = build_ragged_attn_mask(slot_kv_len, kv_prev, l_q)
        v_scale_s = (
            mx.array([[[v] for v in vm] for vm in valid_masks], dtype=mx.bfloat16)
            if has_invalid
            else None
        )

        for entry in active_cache:
            if isinstance(entry, DeltaSlotCache):
                entry.record_verify_trace = k_max > 0

        sh = lm.forward_hidden(
            step_inp,
            active_cache,
            valid_token_scale=v_scale_s,
            attn_mask=d_mask_s,
            rope_trig=r_trig_s,
        )
        logits_step = lm.project_logits(sh)
        if vocab_remap is None and int(logits_step.shape[-1]) > vocab_n:
            logits_step = logits_step[:, :, :vocab_n]
        flat_logits = logits_step.reshape(cur_b * l_q, int(logits_step.shape[-1]))
        flat_idx_mx = gpu_argmax_nan(flat_logits)
        mx.eval(flat_idx_mx)  # guard: allow-eval
        flat_idxs = flat_idx_mx.tolist()  # guard: allow-sync

        tok_step_np = np.empty((cur_b, l_q), dtype=np.int32)
        for r_i in range(cur_b):
            if slot_done[r_i]:
                continue
            for q_i in range(l_q):
                top1 = int(flat_idxs[r_i * l_q + q_i])
                if 0 <= top1 < vocab_n:
                    tok_step_np[r_i, q_i] = int(new_to_old[top1])
                else:
                    row_f32 = logits_step[r_i, q_i].astype(mx.float32)
                    mx.eval(row_f32)  # guard: allow-eval
                    tok_step_np[r_i, q_i] = int(new_to_old[safe_argmax_1d(np.asarray(row_f32))])  # guard: allow-sync

        n_acc = [0] * cur_b
        if k_max > 0:
            for r in range(cur_b):
                if slot_done[r]:
                    n_acc[r] = 0
                    continue
                orig_i = slot_orig_idx[r]
                d_r = drafts_r[r]
                k_r = len(d_r)
                if k_r > 0:
                    drafted_all[orig_i] += k_r
                    verify_passes_all[orig_i] += 1
                emitted_r: list[int] = []
                consumed_r = 0
                for idx_d in range(k_r):
                    target_tok = int(tok_step_np[r, idx_d])
                    consumed_r = idx_d + 1
                    if target_tok == int(d_r[idx_d]):
                        emitted_r.append(target_tok)
                        if target_tok in stop_ids:
                            break
                    else:
                        emitted_r.append(target_tok)
                        break
                else:
                    bonus_tok = int(tok_step_np[r, k_r])
                    emitted_r.append(bonus_tok)
                    consumed_r = k_r + 1

                if k_r > 0:
                    accepted_all[orig_i] += max(0, consumed_r - 1)

                next_pending: list[int] = []
                for etok in emitted_r:
                    if etok in stop_ids:
                        hit_eos_all[orig_i] = True
                        slot_done[r] = True
                        break
                    if len(ans_all[orig_i]) >= max_tokens:
                        hit_max_all[orig_i] = True
                        slot_done[r] = True
                        break
                    ans_all[orig_i].append(etok)
                    next_pending = [etok]

                if slot_done[r] or len(ans_all[orig_i]) >= max_tokens:
                    if not hit_eos_all[orig_i] and len(ans_all[orig_i]) >= max_tokens:
                        hit_max_all[orig_i] = True
                    slot_done[r] = True
                    slot_pending[r] = []
                    n_acc[r] = 0
                else:
                    slot_pending[r] = next_pending
                    slot_true_pos[r] += consumed_r
                    n_acc[r] = consumed_r

            rollback_gdn_batched(active_cache, n_acc, l_q, conv_k, cur_b)
            slot_kv_len = scatter_kv_and_rewind(active_cache, slot_kv_len, n_acc, kv_prev, l_q, cur_b)
        else:
            for r in range(cur_b):
                if slot_done[r]:
                    n_acc[r] = 0
                    continue
                orig_i = slot_orig_idx[r]
                nxt_tok = int(tok_step_np[r, 0])
                if nxt_tok in stop_ids:
                    hit_eos_all[orig_i] = True
                    slot_done[r] = True
                    slot_pending[r] = []
                    n_acc[r] = 0
                else:
                    ans_all[orig_i].append(nxt_tok)
                    if len(ans_all[orig_i]) >= max_tokens:
                        hit_max_all[orig_i] = True
                        slot_done[r] = True
                        slot_pending[r] = []
                        n_acc[r] = 0
                    else:
                        slot_pending[r] = [nxt_tok]
                        slot_true_pos[r] += 1
                        n_acc[r] = 1
            slot_kv_len = scatter_kv_and_rewind(active_cache, slot_kv_len, n_acc, kv_prev, 1, cur_b)
        dec_ms_acc += (time.perf_counter() - t_step0) * 1000.0

    mx.synchronize()
    dec_ms_total = dec_ms_acc
    for i in range(n_items):
        dec_ms_item[i] = dec_ms_total / max(1, n_items)
    del active_cache
    if hasattr(mx, "clear_cache"):
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
            telemetry=None,
            dropped_visual_tokens=0,
            drafted_tokens=drafted_all[i],
            accepted_draft_tokens=accepted_all[i],
            verify_passes=verify_passes_all[i],
        )
        for i in range(n_items)
    ]
    occ_pct = 100.0 * active_slot_steps / max(1, slot_step_capacity)
    return results, {
        "rolling": True,
        "use_pld": eff_use_pld,
        "waves": 1,
        "b_slots": b_slots,
        "replacements": replacements,
        "compactions": compactions,
        "total_decode_steps": total_decode_steps,
        "slot_step_capacity": slot_step_capacity,
        "active_slot_steps": active_slot_steps,
        "occupancy_pct": round(occ_pct, 2),
        "vit_ms": round(total_vit_ms, 2),
        "prefill_ms": round(pf_ms_total, 2),
        "decode_ms": round(dec_ms_total, 2),
        "dropped_visual_tokens": 0,
        "drafted_tokens": int(sum(drafted_all)),
        "accepted_draft_tokens": int(sum(accepted_all)),
        "verify_passes": int(sum(verify_passes_all)),
    }


__all__ = [
    "build_ragged_attn_mask",
    "rollback_gdn_batched",
    "rolling_multimodal_generate",
    "scatter_kv_and_rewind",
    "scatter_single_item_cache_into_slot",
]
