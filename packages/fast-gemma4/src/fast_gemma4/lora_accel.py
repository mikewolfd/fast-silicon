"""Accelerated MLX training primitives for Gemma-4 LoRA / QLoRA fine-tuning.

Implements four composable training accelerations on Apple Silicon (M4 / M5+):
  - P0.3: ``@mx.compile`` fused BF16->FP32 logit softcap + chunked cross-entropy
          and compiled ``AdamW`` + gradient accumulation/clipping updates.
  - P1.5: Fixed-bin varlen sequence packing with per-document RoPE ``position_ids``
          reset, block-diagonal Full + sliding-window (SWA) additive masks,
          per-token PLE alignment, and exact token/example answer loss gathering.
  - P1.6: ``8-bit g64`` frozen base trunk quantization (QLoRA) with ``BF16`` LoRA
          adapters, RAM-aware ``--ckpt-tokens`` defaults (``512`` on ``<= 24 GB``,
          ``1280`` on ``>= 36 GB``), pre-stripped/pre-quantized text-only checkpoint
          loading, plus a sorted ``mx.gather_qmm(..., lhs_indices=..., sorted_indices=True)``
          MoE QLoRA projection helper.
  - P1.7: Differentiable shared-prefix KV broadcast across ``grad_accum``
          micro-batches using exact two-stage VJP (``O(1)`` memory in ``K``) and
          automatic KV-shared upper-layer skipping on Gemma-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Callable, Literal, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

MASK_NEG_VAL = -65504.0


# ---------------------------------------------------------------------------
# P0.3: Compiled Softcap + Chunked CE & Compiled Optimizer / Grad Helpers
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def compiled_softcap_ce_sum(
    logits: mx.array, targets: mx.array, softcap_f32: mx.array
) -> mx.array:
    """Fused BF16->FP32 tanh softcap + cross-entropy sum (keeps FP32 logits in registers)."""
    x = logits.astype(mx.float32)
    x = mx.tanh(x / softcap_f32) * softcap_f32
    return nn.losses.cross_entropy(x, targets, reduction="sum")


@partial(mx.compile, shapeless=True)
def compiled_softcap_ce_weighted(
    logits: mx.array,
    targets: mx.array,
    weights_f32: mx.array,
    softcap_f32: mx.array,
) -> mx.array:
    """Fused BF16->FP32 tanh softcap + weighted cross-entropy sum."""
    x = logits.astype(mx.float32)
    x = mx.tanh(x / softcap_f32) * softcap_f32
    nll = nn.losses.cross_entropy(x, targets, reduction="none")
    return (nll * weights_f32).sum()


@partial(mx.compile, shapeless=True)
def compiled_ce_sum(logits: mx.array, targets: mx.array) -> mx.array:
    """Fused BF16->FP32 cross-entropy sum without softcap."""
    return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="sum")


@partial(mx.compile, shapeless=True)
def compiled_ce_weighted(
    logits: mx.array, targets: mx.array, weights_f32: mx.array
) -> mx.array:
    """Fused BF16->FP32 weighted cross-entropy sum without softcap."""
    nll = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return (nll * weights_f32).sum()


def _uncompiled_softcap_ce(
    logits: mx.array,
    targets: mx.array,
    *,
    softcap: float | None,
    weights_f32: mx.array | None,
) -> mx.array:
    x = logits.astype(mx.float32)
    if softcap is not None:
        cap = float(softcap)
        x = mx.tanh(x / cap) * cap
    if weights_f32 is not None:
        return (nn.losses.cross_entropy(x, targets, reduction="none") * weights_f32).sum()
    return nn.losses.cross_entropy(x, targets, reduction="sum")


def chunked_softcap_ce_loss(
    lm_head_fn: Callable[[mx.array], mx.array],
    h_ans_2d: mx.array,
    y_flat: mx.array,
    *,
    softcap: float | None = None,
    token_weights: mx.array | None = None,
    chunk_size: int = 256,
    compiled: bool = True,
) -> mx.array:
    """Compute answer-only chunked softcap cross-entropy without materializing full ``[S, V]`` FP32 logits.

    When ``token_weights`` is ``None``, returns the mean over all answer tokens.
    When ``token_weights`` is provided (shape ``(N_ans,)``, ``float32``), returns
    ``sum_t token_weights[t] * nll[t]`` (so ``sum(token_weights) == 1.0`` preserves
    exact loss scale across variable-length packed examples).
    """
    n_tot_ans = int(h_ans_2d.shape[0])
    if n_tot_ans == 0:
        return mx.array(0.0, dtype=mx.float32)

    softcap_arr = mx.array(float(softcap), dtype=mx.float32) if softcap is not None else None
    chunk_losses: list[mx.array] = []
    for c0 in range(0, n_tot_ans, chunk_size):
        c1 = min(n_tot_ans, c0 + chunk_size)
        h_sub = h_ans_2d[c0:c1]
        lg_sub = lm_head_fn(h_sub)
        y_sub = y_flat[c0:c1]
        w_sub = token_weights[c0:c1] if token_weights is not None else None
        if compiled:
            if softcap_arr is not None:
                if w_sub is not None:
                    c_loss = compiled_softcap_ce_weighted(lg_sub, y_sub, w_sub, softcap_arr)
                else:
                    c_loss = compiled_softcap_ce_sum(lg_sub, y_sub, softcap_arr)
            else:
                if w_sub is not None:
                    c_loss = compiled_ce_weighted(lg_sub, y_sub, w_sub)
                else:
                    c_loss = compiled_ce_sum(lg_sub, y_sub)
        else:
            c_loss = _uncompiled_softcap_ce(
                lg_sub, y_sub, softcap=softcap, weights_f32=w_sub
            )
        chunk_losses.append(c_loss)

    total = chunk_losses[0]
    for cl in chunk_losses[1:]:
        total = total + cl
    if token_weights is not None:
        return total
    return total * (1.0 / float(n_tot_ans))


class CompiledOptimizerStep:
    """Compiles gradient scaling, global-norm clipping, and ``AdamW`` parameter updates into one graph."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        *,
        max_grad_norm: float = 1.0,
        compiled: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.compiled = bool(compiled)
        trainable = model.trainable_parameters()
        self.optimizer.init(trainable)
        self._state = [self.optimizer.state]

        def _raw_step(
            params: Any,
            grads: Any,
            inv_accum_f32: mx.array,
            lr_f32: mx.array,
        ) -> tuple[Any, mx.array]:
            scaled = tree_map(lambda g: g * inv_accum_f32.astype(g.dtype), grads)
            if self.max_grad_norm > 0.0:
                clipped, grad_norm = optim.clip_grad_norm(
                    scaled, max_norm=self.max_grad_norm
                )
            else:
                clipped = scaled
                grad_norm = mx.array(0.0, dtype=mx.float32)
            self.optimizer.learning_rate = lr_f32
            new_params = self.optimizer.apply_gradients(clipped, params)
            return new_params, grad_norm

        if self.compiled:
            self._step_fn = mx.compile(
                _raw_step, inputs=self._state, outputs=self._state
            )
        else:
            self._step_fn = _raw_step

        def _raw_accum(acc: Any, g: Any) -> Any:
            return tree_map(mx.add, acc, g)

        self._accum_fn = mx.compile(_raw_accum) if self.compiled else _raw_accum

    def accumulate(self, acc_grads: Any, new_grads: Any) -> Any:
        """Accumulate a gradient pytree using the compiled elementwise adder."""
        if acc_grads is None:
            return new_grads
        return self._accum_fn(acc_grads, new_grads)

    def step(
        self,
        model: nn.Module,
        grads: Any,
        *,
        lr: float,
        accum_divisor: int = 1,
    ) -> mx.array:
        """Scale, clip, and apply one optimizer update to ``model.trainable_parameters()``."""
        inv_acc = mx.array(1.0 / float(max(1, accum_divisor)), dtype=mx.float32)
        lr_arr = mx.array(float(lr), dtype=mx.float32)
        new_params, grad_norm = self._step_fn(
            model.trainable_parameters(), grads, inv_acc, lr_arr
        )
        model.update(new_params)
        return grad_norm


# ---------------------------------------------------------------------------
# P1.5: Varlen Sequence Packing, Per-Token RoPE Reset & Block-Diagonal Masks
# ---------------------------------------------------------------------------


@dataclass
class RopeTable:
    """Precomputed per-token rotary cosine/sine table for a packed or offset sequence."""

    cos: mx.array
    sin: mx.array
    dims: int
    traditional: bool
    mscale: float = 1.0


@dataclass
class PackedBin:
    """Fixed-bin packed sequence batch with segment/position metadata and exact answer indices."""

    input_ids: mx.array
    position_ids: mx.array
    segment_ids: mx.array
    ans_indices: mx.array
    y_flat: mx.array
    token_weights: mx.array | None
    num_examples: int
    real_tokens: int
    bin_tokens: int
    spans: list[tuple[int, int]]
    segment_spans: list[tuple[int, int, int]]


@dataclass
class SegmentedAttentionPlan:
    """Block-diagonal per-segment RoPE + SDPA plan avoiding O((sum L_i)^2) cross-doc mask waste."""

    segment_spans: list[tuple[int, int, int]]
    segment_masks: list[Any]
    total_tokens: int


def _extract_rope_params(rope_module: Any) -> tuple[mx.array, int, bool, float]:
    """Extract inverse frequencies, rotated dims, traditional flag, and mscale from an MLX RoPE module."""
    dims = int(getattr(rope_module, "dims", getattr(rope_module, "dim", 0)))
    traditional = bool(getattr(rope_module, "traditional", False))
    mscale = float(getattr(rope_module, "mscale", getattr(rope_module, "_scale", 1.0)))
    freqs = getattr(rope_module, "_freqs", None)
    if freqs is not None:
        f32 = freqs.astype(mx.float32)
        zero_f32 = mx.array(0.0, dtype=mx.float32)
        one_f32 = mx.array(1.0, dtype=mx.float32)
        inv_freqs = mx.where(mx.isinf(f32), zero_f32, one_f32 / f32)
        if dims <= 0:
            dims = int(f32.shape[0]) * 2
    else:
        base = float(getattr(rope_module, "base", 10000.0))
        scale = float(getattr(rope_module, "scale", 1.0))
        exponents = mx.arange(0, dims, 2, dtype=mx.float32) / float(dims)
        inv_freqs = mx.array(scale, dtype=mx.float32) / (base**exponents)
    return inv_freqs, dims, traditional, mscale


def build_rope_table(
    rope_module: Any, position_ids: mx.array, dtype: mx.Dtype | None = None
) -> RopeTable:
    """Precompute FP32 ``(cos, sin)`` matching Metal ``mx.fast.rope`` for arbitrary per-token ``position_ids``."""
    del dtype  # Angles stay in FP32 and fuse in registers inside _compiled_rotate_pair.
    inv_freqs, dims, traditional, mscale = _extract_rope_params(rope_module)
    pos_f32 = position_ids.astype(mx.float32)
    if pos_f32.ndim == 1:
        pos_f32 = pos_f32[None, :]
    # Shape: (B, 1, L, dims // 2)
    angles = pos_f32[:, None, :, None] * inv_freqs[None, None, None, :]
    return RopeTable(
        cos=mx.cos(angles),
        sin=mx.sin(angles),
        dims=dims,
        traditional=traditional,
        mscale=mscale,
    )


@partial(mx.compile, shapeless=True)
def _compiled_rotate_pair(
    x1: mx.array, x2: mx.array, cos_f32: mx.array, sin_f32: mx.array
) -> tuple[mx.array, mx.array]:
    """Fused FP32-register rotary pair multiply-add matching Metal ``mx.fast.rope`` precision."""
    out_dt = x1.dtype
    a = x1.astype(mx.float32)
    b = x2.astype(mx.float32)
    y1 = (a * cos_f32 - b * sin_f32).astype(out_dt)
    y2 = (b * cos_f32 + a * sin_f32).astype(out_dt)
    return y1, y2


def apply_rope_with_table(x: mx.array, table: RopeTable) -> mx.array:
    """Apply precomputed per-token ``RopeTable`` to ``x`` of shape ``(B, H, L, D)``."""
    head_dim = int(x.shape[-1])
    rot = x if table.dims == head_dim else x[..., : table.dims]
    if table.mscale != 1.0:
        rot = rot * mx.array(table.mscale, dtype=x.dtype)
    if table.traditional:
        y1, y2 = _compiled_rotate_pair(rot[..., 0::2], rot[..., 1::2], table.cos, table.sin)
        rot = mx.stack([y1, y2], axis=-1).reshape(rot.shape)
    else:
        half = int(rot.shape[-1]) // 2
        y1, y2 = _compiled_rotate_pair(rot[..., :half], rot[..., half:], table.cos, table.sin)
        rot = mx.concatenate([y1, y2], axis=-1)
    if table.dims == head_dim:
        return rot
    return mx.concatenate([rot, x[..., table.dims :]], axis=-1)


def build_rope_tables_for_model(
    tm: Any, position_ids: mx.array, dtype: mx.Dtype
) -> dict[str, RopeTable]:
    """Build one ``RopeTable`` per distinct layer type (``sliding_attention``, ``full_attention``)."""
    tables: dict[str, RopeTable] = {}
    for layer in tm.layers:
        ltype = getattr(layer, "layer_type", "full_attention")
        if ltype not in tables:
            tables[ltype] = build_rope_table(layer.self_attn.rope, position_ids, dtype)
    return tables


def build_packed_groups(
    items: Sequence[tuple[list[int], int]],
    *,
    pack_tokens: int = 2048,
    pool_size: int = 256,
    seed: int = 20261117,
) -> list[list[tuple[list[int], int]]]:
    """Pack ``(full_ids, prompt_len)`` items into capacity-``pack_tokens`` bins via Best-Fit Decreasing.

    Oversized items (``len(full_ids) - 1 > pack_tokens``) are placed intact in singleton bins
    so no tokens are silently truncated.
    """
    if pack_tokens <= 0:
        raise ValueError(f"pack_tokens must be positive, got {pack_tokens}")
    rng = random.Random(seed)
    perm = list(items)
    rng.shuffle(perm)
    bins: list[list[tuple[list[int], int]]] = []
    p_sz = max(1, int(pool_size))

    for p0 in range(0, len(perm), p_sz):
        pool = sorted(perm[p0 : p0 + p_sz], key=lambda it: len(it[0]) - 1, reverse=True)
        open_bins: list[list[tuple[list[int], int]]] = []
        open_rem: list[int] = []

        for it in pool:
            l_in = len(it[0]) - 1
            if l_in > pack_tokens:
                bins.append([it])
                continue
            best_idx = -1
            best_rem = pack_tokens + 1
            for bi, rem in enumerate(open_rem):
                if l_in <= rem < best_rem:
                    best_idx = bi
                    best_rem = rem
            if best_idx >= 0:
                open_bins[best_idx].append(it)
                open_rem[best_idx] -= l_in
            else:
                open_bins.append([it])
                open_rem.append(pack_tokens - l_in)
        bins.extend(open_bins)

    rng.shuffle(bins)
    return bins


def pack_examples_to_bin(
    examples: Sequence[tuple[list[int], int]],
    *,
    pack_tokens: int = 2048,
    pad_to_fixed: bool = True,
    pad_multiple: int = 1,
    pad_token_id: int = 0,
    loss_norm: Literal["token", "example"] = "token",
) -> PackedBin:
    """Concatenate ``examples`` into a single packed sequence with per-doc RoPE reset & answer indices."""
    if not examples:
        raise ValueError("Cannot pack an empty example list.")
    if loss_norm not in ("token", "example"):
        raise ValueError(f"Unsupported loss_norm={loss_norm!r}; expected 'token' or 'example'.")

    in_ids: list[int] = []
    pos_ids: list[int] = []
    seg_ids: list[int] = []
    ans_indices: list[int] = []
    y_flat: list[int] = []
    weights_list: list[float] = []
    spans: list[tuple[int, int]] = []
    segment_spans: list[tuple[int, int, int]] = []

    k_docs = len(examples)
    cursor = 0
    for seg_i, (full_ids, p_len) in enumerate(examples):
        inp = full_ids[:-1]
        l_in = len(inp)
        if p_len < 1 or p_len >= len(full_ids):
            raise ValueError(
                f"Invalid prompt_len={p_len} for sequence of length {len(full_ids)}"
            )
        in_ids.extend(inp)
        pos_ids.extend(range(l_in))
        seg_ids.extend([seg_i] * l_in)
        segment_spans.append((cursor, cursor + l_in, 0))

        ans_s = cursor + (p_len - 1)
        ans_e = cursor + l_in
        spans.append((ans_s, ans_e))
        ans_indices.extend(range(ans_s, ans_e))
        targets_i = full_ids[p_len:]
        y_flat.extend(targets_i)
        if loss_norm == "example":
            w_i = 1.0 / float(k_docs * len(targets_i))
            weights_list.extend([w_i] * len(targets_i))
        cursor += l_in

    real_tokens = cursor
    if pad_to_fixed and pack_tokens > 0:
        bin_tokens = max(pack_tokens, real_tokens)
    elif pad_multiple > 1:
        bin_tokens = ((real_tokens + pad_multiple - 1) // pad_multiple) * pad_multiple
    else:
        bin_tokens = real_tokens
    pad_n = bin_tokens - real_tokens
    if pad_n > 0:
        # Distinct segment id (-1) isolates pad tokens from all real docs while
        # preserving diagonal self-attention (mask[q, q] == 0) so softmax never sees all -inf.
        in_ids.extend([pad_token_id] * pad_n)
        pos_ids.extend([0] * pad_n)
        seg_ids.extend([-1] * pad_n)

    token_weights = (
        mx.array(weights_list, dtype=mx.float32) if loss_norm == "example" else None
    )
    return PackedBin(
        input_ids=mx.array([in_ids], dtype=mx.int32),
        position_ids=mx.array([pos_ids], dtype=mx.int32),
        segment_ids=mx.array([seg_ids], dtype=mx.int32),
        ans_indices=mx.array(ans_indices, dtype=mx.int32),
        y_flat=mx.array(y_flat, dtype=mx.int32),
        token_weights=token_weights,
        num_examples=k_docs,
        real_tokens=real_tokens,
        bin_tokens=bin_tokens,
        spans=spans,
        segment_spans=segment_spans,
    )


def build_packed_attention_masks(
    segment_ids: mx.array,
    *,
    window_size: int | None,
    dtype: mx.Dtype,
) -> dict[str, mx.array]:
    """Build additive ``dtype`` block-diagonal Full and SWA causal masks of shape ``(1, 1, M, M)``."""
    seg = segment_ids[0] if segment_ids.ndim == 2 else segment_ids
    seq_len = int(seg.shape[0])
    q_idx = mx.arange(seq_len, dtype=mx.int32)[:, None]
    k_idx = mx.arange(seq_len, dtype=mx.int32)[None, :]
    same_seg = seg[:, None] == seg[None, :]
    allowed_full = same_seg & (q_idx >= k_idx)

    zero = mx.array(0.0, dtype=dtype)
    neg = mx.array(MASK_NEG_VAL, dtype=dtype)
    masks: dict[str, mx.array] = {
        "full_attention": mx.where(allowed_full, zero, neg)[None, None, :, :]
    }
    if window_size is not None:
        allowed_swa = allowed_full & (q_idx < (k_idx + int(window_size)))
        masks["sliding_attention"] = mx.where(allowed_swa, zero, neg)[None, None, :, :]
    else:
        masks["sliding_attention"] = masks["full_attention"]
    return masks


def build_segmented_attention_plans(
    segment_spans: Sequence[tuple[int, int, int]],
    total_tokens: int,
    *,
    sys_len: int = 0,
    window_size: int | None,
    dtype: mx.Dtype,
) -> dict[str, SegmentedAttentionPlan]:
    """Build per-segment SDPA plans using fused ``mask='causal'`` wherever exact and small SWA masks otherwise."""
    spans = list(segment_spans)
    full_masks: list[Any] = ["causal"] * len(spans)
    swa_masks: list[Any] = []

    zero = mx.array(0.0, dtype=dtype)
    neg = mx.array(MASK_NEG_VAL, dtype=dtype)
    swa_cache: dict[tuple[int, int], mx.array] = {}
    pfx_swa_start = _swa_prefix_start(sys_len, window_size) if sys_len > 0 else 0
    pfx_swa_len = (sys_len - pfx_swa_start) if sys_len > 0 else 0

    for s, e, off in spans:
        l_seg = e - s
        if window_size is None or (pfx_swa_len + l_seg) <= int(window_size):
            swa_masks.append("causal")
        else:
            w = int(window_size)
            key = (l_seg, off)
            cached = swa_cache.get(key)
            if cached is None:
                q_pos = (off + mx.arange(l_seg, dtype=mx.int32))[:, None]
                if sys_len > 0:
                    k_pfx = mx.arange(pfx_swa_start, sys_len, dtype=mx.int32)
                    k_suf = off + mx.arange(l_seg, dtype=mx.int32)
                    k_pos = mx.concatenate([k_pfx, k_suf], axis=0)[None, :]
                else:
                    k_pos = (off + mx.arange(l_seg, dtype=mx.int32))[None, :]
                allowed = (q_pos >= k_pos) & (q_pos < (k_pos + w))
                cached = mx.where(allowed, zero, neg)[None, None, :, :]
                swa_cache[key] = cached
            swa_masks.append(cached)

    return {
        "full_attention": SegmentedAttentionPlan(
            segment_spans=spans, segment_masks=full_masks, total_tokens=total_tokens
        ),
        "sliding_attention": SegmentedAttentionPlan(
            segment_spans=spans, segment_masks=swa_masks, total_tokens=total_tokens
        ),
    }


def _apply_segmented_rope(
    attn: Any,
    tensor: mx.array,
    segment_spans: Sequence[tuple[int, int, int]],
    total_len: int,
) -> mx.array:
    """Apply native ``attn.rope`` per packed segment (bit-exact with ``mlx_lm``)."""
    if (
        len(segment_spans) == 1
        and segment_spans[0][0] == 0
        and segment_spans[0][1] == total_len
    ):
        return attn.rope(tensor, offset=segment_spans[0][2])
    pieces = [
        attn.rope(tensor[:, :, s:e, :], offset=off) for s, e, off in segment_spans
    ]
    real_end = segment_spans[-1][1]
    if real_end < total_len:
        pieces.append(tensor[:, :, real_end:total_len, :])
    return mx.concatenate(pieces, axis=2)


def forward_gemma4_attention(
    attn: Any,
    x: mx.array,
    mask: Any,
    *,
    shared_kv: tuple[mx.array, mx.array] | None = None,
    rope_table: RopeTable | None = None,
    prefix_kv: tuple[mx.array, mx.array] | None = None,
    offset: int = 0,
) -> tuple[mx.array, tuple[mx.array, mx.array], int]:
    """Execute Gemma-4 attention with segmented block-diagonal SDPA, per-token RoPE, or prefix KV broadcast."""
    bsz, seq_len, _ = x.shape
    queries = attn.q_proj(x).reshape(bsz, seq_len, attn.n_heads, attn.head_dim)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)

    is_segmented = isinstance(mask, SegmentedAttentionPlan)
    if shared_kv is not None:
        keys, values = shared_kv
    else:
        keys = attn.k_proj(x).reshape(bsz, seq_len, attn.n_kv_heads, attn.head_dim)
        if getattr(attn, "use_k_eq_v", False):
            values = keys
        else:
            values = attn.v_proj(x).reshape(bsz, seq_len, attn.n_kv_heads, attn.head_dim)

        keys = attn.k_norm(keys).transpose(0, 2, 1, 3)
        if is_segmented:
            keys = _apply_segmented_rope(attn, keys, mask.segment_spans, seq_len)
        elif rope_table is not None:
            keys = apply_rope_with_table(keys, rope_table)
        else:
            keys = attn.rope(keys, offset=offset)

        values = attn.v_norm(values).transpose(0, 2, 1, 3)

        if prefix_kv is not None:
            k_pfx, v_pfx = prefix_kv
            keys = mx.concatenate([k_pfx, keys], axis=2)
            values = mx.concatenate([v_pfx, values], axis=2)

    if is_segmented:
        queries = _apply_segmented_rope(attn, queries, mask.segment_spans, seq_len)
        spans = mask.segment_spans
        seg_masks = mask.segment_masks
        pfx_len = int(keys.shape[2]) - seq_len
        if (
            len(spans) == 1
            and spans[0][0] == 0
            and spans[0][1] == seq_len
        ):
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask=seg_masks[0]
            )
        else:
            k_pfx_slice = keys[:, :, :pfx_len, :] if pfx_len > 0 else None
            v_pfx_slice = values[:, :, :pfx_len, :] if pfx_len > 0 else None
            seg_outs: list[mx.array] = []
            for (s, e, _), seg_m in zip(spans, seg_masks):
                # Materialize contiguous segment tensors before Metal SDPA VJP (MLX steel_attention_vjp
                # silently corrupts gradients on non-contiguous axis-2 slices with stride != L_seg * D).
                q_seg = mx.contiguous(queries[:, :, s:e, :])
                if k_pfx_slice is not None and v_pfx_slice is not None:
                    k_seg = mx.concatenate(
                        [k_pfx_slice, keys[:, :, pfx_len + s : pfx_len + e, :]], axis=2
                    )
                    v_seg = mx.concatenate(
                        [v_pfx_slice, values[:, :, pfx_len + s : pfx_len + e, :]], axis=2
                    )
                else:
                    k_seg = mx.contiguous(keys[:, :, s:e, :])
                    v_seg = mx.contiguous(values[:, :, s:e, :])
                seg_outs.append(
                    mx.fast.scaled_dot_product_attention(
                        q_seg, k_seg, v_seg, scale=attn.scale, mask=seg_m
                    )
                )
            real_end = spans[-1][1]
            if real_end < seq_len:
                seg_outs.append(
                    mx.zeros(
                        (bsz, attn.n_heads, seq_len - real_end, attn.head_dim),
                        dtype=queries.dtype,
                    )
                )
            out = mx.concatenate(seg_outs, axis=2)
    else:
        if rope_table is not None:
            queries = apply_rope_with_table(queries, rope_table)
        else:
            queries = attn.rope(queries, offset=offset)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=attn.scale, mask=mask
        )

    out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
    return attn.o_proj(out), (keys, values), offset


def forward_gemma4_layer(
    layer: Any,
    x: mx.array,
    mask: Any,
    *,
    per_layer_input: mx.array | None = None,
    shared_kv: tuple[mx.array, mx.array] | None = None,
    rope_table: RopeTable | None = None,
    prefix_kv: tuple[mx.array, mx.array] | None = None,
    offset: int = 0,
) -> tuple[mx.array, tuple[mx.array, mx.array], int]:
    """Execute one Gemma-4 decoder layer matching ``mlx_lm.models.gemma4_text.DecoderLayer``."""
    residual = x
    h = layer.input_layernorm(x)
    h, kvs, out_offset = forward_gemma4_attention(
        layer.self_attn,
        h,
        mask,
        shared_kv=shared_kv,
        rope_table=rope_table,
        prefix_kv=prefix_kv,
        offset=offset,
    )
    h = layer.post_attention_layernorm(h)
    h = residual + h

    residual = h
    if getattr(layer, "enable_moe", False):
        h1 = layer.pre_feedforward_layernorm(h)
        h1 = layer.mlp(h1)
        h1 = layer.post_feedforward_layernorm_1(h1)

        top_k_indices, top_k_weights = layer.router(h)
        h2 = layer.pre_feedforward_layernorm_2(h)
        h2 = layer.experts(h2, top_k_indices, top_k_weights)
        h2 = layer.post_feedforward_layernorm_2(h2)
        h = h1 + h2
    else:
        h = layer.pre_feedforward_layernorm(h)
        h = layer.mlp(h)

    h = layer.post_feedforward_layernorm(h)
    h = residual + h

    if (
        getattr(layer, "per_layer_input_gate", None) is not None
        and getattr(layer, "per_layer_projection", None) is not None
        and getattr(layer, "post_per_layer_input_norm", None) is not None
        and per_layer_input is not None
    ):
        residual = h
        gate = layer.per_layer_input_gate(h)
        gate = nn.gelu_approx(gate)
        gate = mx.multiply(gate, per_layer_input)
        gate = layer.per_layer_projection(gate)
        gate = layer.post_per_layer_input_norm(gate)
        h = residual + gate

    if getattr(layer, "layer_scalar", None) is not None:
        h = h * layer.layer_scalar

    return h, kvs, out_offset


def _compute_embed_and_ple(
    tm: Any, input_ids: mx.array
) -> tuple[mx.array, list[mx.array | None]]:
    """Compute token embeddings and per-token PLE inputs aligned to ``input_ids``."""
    input_embeddings = tm.embed_tokens(input_ids)
    h = input_embeddings * mx.array(tm.embed_scale, dtype=input_embeddings.dtype)
    if getattr(tm, "hidden_size_per_layer_input", 0):
        pli = tm._get_per_layer_inputs(input_ids, input_embeddings)
        pli = tm._project_per_layer_inputs(h, pli)
        pli_list: list[mx.array | None] = [
            pli[:, :, i, :] for i in range(len(tm.layers))
        ]
    else:
        pli_list = [None] * len(tm.layers)
    return h, pli_list


def run_frozen_prefix_packed(
    model: Any,
    packed_bin: PackedBin,
    first_lora_layer: int,
) -> tuple[mx.array, list[Any], list[tuple[Any, Any]], list[mx.array | None], dict[str, RopeTable | None]]:
    """Run frozen bottom layers ``0..first_lora_layer-1`` on a packed bin with per-doc RoPE reset & PLE alignment."""
    tm = model.language_model.model
    h, pli_list = _compute_embed_and_ple(tm, packed_bin.input_ids)
    plan_by_type = build_segmented_attention_plans(
        packed_bin.segment_spans,
        packed_bin.bin_tokens,
        sys_len=0,
        window_size=getattr(tm, "window_size", None),
        dtype=h.dtype,
    )
    masks: list[Any] = [
        plan_by_type[getattr(layer, "layer_type", "full_attention")]
        for layer in tm.layers
    ]
    rope_tables: dict[str, RopeTable | None] = {
        "full_attention": None,
        "sliding_attention": None,
    }
    intermediates: list[tuple[Any, Any]] = [(None, None)] * len(tm.layers)

    for idx in range(first_lora_layer):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        kvs, off = intermediates[prev_idx]
        h, kvs, off = forward_gemma4_layer(
            layer,
            h,
            masks[idx],
            per_layer_input=pli_list[idx],
            shared_kv=kvs,
            offset=off or 0,
        )
        intermediates[idx] = (kvs, off)

    return h, masks, intermediates, pli_list, rope_tables


def compute_packed_tail_ce_loss(
    model: Any,
    h: mx.array,
    masks: list[Any],
    prefix_intermediates: list[tuple[Any, Any]],
    pli_list: list[mx.array | None],
    rope_tables: dict[str, RopeTable | None],
    ans_indices: mx.array,
    y_flat: mx.array,
    token_weights: mx.array | None,
    first_lora_layer: int,
    *,
    use_ckpt: bool = False,
    chunk_size: int = 256,
    compiled_ce: bool = True,
) -> mx.array:
    """Run trainable upper layers on a packed bin, gather exact answer tokens, and compute compiled CE."""
    lm = model.language_model
    tm = lm.model
    intermediates = list(prefix_intermediates)

    for idx in range(first_lora_layer, len(tm.layers)):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        kvs, off = intermediates[prev_idx]
        ltype = getattr(layer, "layer_type", "full_attention")
        r_table = rope_tables.get(ltype)
        m_layer = masks[idx]
        if use_ckpt:

            def _ckpt_fn(
                h_in: mx.array,
                pli_in: mx.array | None,
                kvs_in: Any,
                _l: Any = layer,
                _m: Any = m_layer,
                _rt: RopeTable | None = r_table,
            ) -> tuple[mx.array, tuple[mx.array, mx.array], int]:
                return forward_gemma4_layer(
                    _l,
                    h_in,
                    _m,
                    per_layer_input=pli_in,
                    shared_kv=kvs_in,
                    rope_table=_rt,
                )

            h, kvs, off = nn.utils.checkpoint(layer, _ckpt_fn)(
                h, pli_list[idx], kvs
            )
        else:
            h, kvs, off = forward_gemma4_layer(
                layer,
                h,
                m_layer,
                per_layer_input=pli_list[idx],
                shared_kv=kvs,
                rope_table=r_table,
            )
        intermediates[idx] = (kvs, off)

    h_norm = tm.norm(h)
    h_ans_2d = h_norm[0, ans_indices, :]
    lm_head_fn = (
        tm.embed_tokens.as_linear
        if getattr(lm, "tie_word_embeddings", True)
        else lm.lm_head
    )
    return chunked_softcap_ce_loss(
        lm_head_fn,
        h_ans_2d,
        y_flat,
        softcap=getattr(lm, "final_logit_softcapping", None),
        token_weights=token_weights,
        chunk_size=chunk_size,
        compiled=compiled_ce,
    )


# ---------------------------------------------------------------------------
# P1.6: 8-bit g64 QLoRA Trunk, BF16 Adapters & Sorted MoE gather_qmm QLoRA
# ---------------------------------------------------------------------------


_MULTIMODAL_TENSOR_MARKERS: tuple[str, ...] = (
    "vision_tower",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    "multi_modal_projector",
)


def _detect_host_ram_gib() -> float:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0**3)
    except Exception:
        return 0.0


def _is_multimodal_tensor_key(key: str) -> bool:
    return any(key.startswith(m) or f".{m}." in key or key.endswith(f".{m}") for m in _MULTIMODAL_TENSOR_MARKERS)


def compute_effective_ckpt_tokens(
    requested_ckpt_tokens: int | None,
    *,
    qlora_8bit: bool,
    host_ram_gib: float | None = None,
    default_bf16: int = 2200,
    default_qlora_low_ram: int = 512,
    default_qlora_high_ram: int = 1280,
) -> int:
    """Resolve ``--ckpt-tokens``: ``512`` on ``<= 24 GB`` (``< 36 GB``) and ``1280`` on ``>= 36 GB`` for ``--qlora-8bit``."""
    if requested_ckpt_tokens is not None:
        return int(requested_ckpt_tokens)
    if not qlora_8bit:
        return int(default_bf16)
    ram = float(_detect_host_ram_gib() if host_ram_gib is None else host_ram_gib)
    if 0.0 < ram < 36.0:
        return int(default_qlora_low_ram)
    return int(default_qlora_high_ram)


def inspect_lora_base_checkpoint(base_dir: Path | str) -> dict[str, Any]:
    """Inspect a Gemma-4 base checkpoint via JSON + safetensors headers without allocating Metal tensors."""
    from .packager import inspect_safetensors_headers

    bdir = Path(base_dir)
    cfg_path = bdir / "config.json"
    cfg: dict[str, Any] = (
        json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    )
    tcfg = cfg.get("text_config", cfg)
    has_quant_cfg = bool(
        cfg.get("quantization")
        or cfg.get("quantization_config")
        or tcfg.get("quantization")
        or tcfg.get("quantization_config")
    )
    has_vision_cfg = bool(cfg.get("vision_config") or cfg.get("audio_config"))

    headers = inspect_safetensors_headers(bdir) if any(bdir.glob("*.safetensors")) else {}
    has_scales = False
    has_mm_tensors = False
    text_bytes = 0
    mm_bytes = 0
    for key, info in headers.items():
        nbytes = int(info.get("nbytes", 0))
        if _is_multimodal_tensor_key(key):
            has_mm_tensors = True
            mm_bytes += nbytes
        else:
            text_bytes += nbytes
            if key.endswith(".scales"):
                has_scales = True

    return {
        "is_prequantized": bool(has_quant_cfg or has_scales),
        "has_multimodal_tensors": bool(has_mm_tensors),
        "has_vision_config": bool(has_vision_cfg),
        "text_only_bytes": int(text_bytes),
        "multimodal_bytes": int(mm_bytes),
    }


def load_text_only_lora_base_model(
    base_dir: Path | str,
    *,
    strict: bool = False,
    load_model_fn: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a pre-stripped/pre-quantized or multimodal Gemma-4 checkpoint without materializing 10.24 GB multimodal BF16."""
    bdir = Path(base_dir)
    info = inspect_lora_base_checkpoint(bdir)
    if load_model_fn is None:
        from mlx_lm.utils import load_model as _mlx_load_model

        load_model_fn = _mlx_load_model

    supports_lazy = False
    try:
        supports_lazy = "lazy" in inspect.signature(load_model_fn).parameters
    except (TypeError, ValueError):
        supports_lazy = False

    # Fast path: already text-only (and optionally pre-quantized) on disk -> load directly!
    if not info["has_multimodal_tensors"] and not info["has_vision_config"]:
        call_kw: dict[str, Any] = {"strict": strict}
        if supports_lazy:
            call_kw["lazy"] = False
        loaded = load_model_fn(bdir, **call_kw)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        return model, info

    # Stage a lightweight header-only view with vision/audio stripped from config.json and shard index
    # so `mlx_lm` never instantiates vision/audio towers or reads pure-multimodal shards into RAM!
    with tempfile.TemporaryDirectory(prefix="fg4_text_base_") as tmp_s:
        sdir = Path(tmp_s)
        cfg_p = bdir / "config.json"
        if cfg_p.is_file():
            cfg_d = json.loads(cfg_p.read_text(encoding="utf-8"))
            cfg_d.pop("vision_config", None)
            cfg_d.pop("audio_config", None)
            (sdir / "config.json").write_text(json.dumps(cfg_d, indent=2) + "\n", encoding="utf-8")

        idx_p = bdir / "model.safetensors.index.json"
        referenced_shards: set[str] | None = None
        if idx_p.is_file():
            idx_d = json.loads(idx_p.read_text(encoding="utf-8"))
            wmap = idx_d.get("weight_map", {})
            filtered_wmap = {
                k: v for k, v in wmap.items() if not _is_multimodal_tensor_key(k)
            }
            idx_d["weight_map"] = filtered_wmap
            referenced_shards = set(filtered_wmap.values())
            (sdir / "model.safetensors.index.json").write_text(
                json.dumps(idx_d, indent=2) + "\n", encoding="utf-8"
            )

        for sf in sorted(bdir.glob("*.safetensors")):
            if referenced_shards is not None and sf.name not in referenced_shards:
                continue
            os.symlink(sf.resolve(), sdir / sf.name)

        for tok_name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            tp = bdir / tok_name
            if tp.is_file():
                shutil.copy2(tp, sdir / tok_name)

        call_kw = {"strict": False}
        if supports_lazy:
            call_kw["lazy"] = True
        loaded = load_model_fn(sdir, **call_kw)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        for mm_attr in _MULTIMODAL_TENSOR_MARKERS:
            if hasattr(model, mm_attr):
                setattr(model, mm_attr, None)
        if supports_lazy:
            mx.eval(model.parameters())
        mx.clear_cache()
        return model, info


def strip_quantization_from_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `config.json` with top-level and `text_config` quantization blocks removed."""
    out = dict(cfg)
    out.pop("quantization", None)
    out.pop("quantization_config", None)
    out.pop("ple_sidecar", None)
    if isinstance(out.get("text_config"), dict):
        tcfg = dict(out["text_config"])
        tcfg.pop("quantization", None)
        tcfg.pop("quantization_config", None)
        out["text_config"] = tcfg
    return out


def dequantize_prequantized_shard_tensors(
    shard_tensors: dict[str, mx.array],
    base_cfg: dict[str, Any],
    *,
    lora_by_key: dict[str, Any] | None = None,
    quant_mod_by_key: dict[str, Any] | None = None,
) -> tuple[dict[str, mx.array], int]:
    """Convert one safetensors shard (pre-quantized or BF16) + LoRA deltas into clean BF16 tensors."""
    from .model import _resolve_quant_spec

    tcfg = base_cfg.get("text_config", base_cfg)
    qcfg: dict[str, Any] = dict(
        base_cfg.get("quantization")
        or base_cfg.get("quantization_config")
        or tcfg.get("quantization")
        or tcfg.get("quantization_config")
        or {}
    )
    lora_map = lora_by_key or {}
    qmod_map = quant_mod_by_key or {}
    out_shard: dict[str, mx.array] = {}
    fused_count = 0

    for k, t_val in shard_tensors.items():
        if _is_multimodal_tensor_key(k):
            continue
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        if k in lora_map:
            mod = lora_map[k]
            delta_f32 = float(mod.scale) * (
                mod.lora_b.astype(mx.float32).T @ mod.lora_a.astype(mx.float32).T
            )
            qlin = getattr(mod, "linear", None)
            if isinstance(qlin, nn.QuantizedLinear):
                base_f32 = mx.dequantize(
                    qlin.weight,
                    qlin.scales,
                    qlin.biases,
                    group_size=qlin.group_size,
                    bits=qlin.bits,
                    mode=qlin.mode,
                ).astype(mx.float32)
            elif k.endswith(".weight") and f"{k[:-7]}.scales" in shard_tensors:
                pref = k[:-7]
                bits, gs, mode = _resolve_quant_spec(shard_tensors, pref, qcfg)
                base_f32 = mx.dequantize(
                    t_val,
                    shard_tensors[f"{pref}.scales"],
                    shard_tensors.get(f"{pref}.biases"),
                    group_size=gs,
                    bits=bits,
                    mode=mode,
                ).astype(mx.float32)
            else:
                base_f32 = t_val.astype(mx.float32)
            merged_bf16 = (base_f32 + delta_f32).astype(mx.bfloat16)
            mx.eval(merged_bf16)
            out_shard[k] = merged_bf16
            fused_count += 1
        elif k in qmod_map and isinstance(qmod_map[k], (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            qm = qmod_map[k]
            deq_bf16 = mx.dequantize(
                qm.weight,
                qm.scales,
                qm.biases,
                group_size=qm.group_size,
                bits=qm.bits,
                mode=qm.mode,
            ).astype(mx.bfloat16)
            mx.eval(deq_bf16)
            out_shard[k] = deq_bf16
        elif k.endswith(".weight") and f"{k[:-7]}.scales" in shard_tensors:
            pref = k[:-7]
            bits, gs, mode = _resolve_quant_spec(shard_tensors, pref, qcfg)
            deq_bf16 = mx.dequantize(
                t_val,
                shard_tensors[f"{pref}.scales"],
                shard_tensors.get(f"{pref}.biases"),
                group_size=gs,
                bits=bits,
                mode=mode,
            ).astype(mx.bfloat16)
            mx.eval(deq_bf16)
            out_shard[k] = deq_bf16
        else:
            out_shard[k] = t_val
    return out_shard, fused_count


def quantize_frozen_trunk_8bit(
    model: Any,
    *,
    group_size: int = 64,
    bits: int = 8,
    quantize_ple_embedding: bool = True,
    quantize_embed_tokens: bool = True,
) -> dict[str, int]:
    """Quantize frozen trunk ``nn.Linear`` (and PLE + token embeddings by default) to ``8-bit g64``.

    Already-quantized ``QuantizedLinear`` / ``QuantizedEmbedding`` modules are skipped cleanly
    without double-quantizing. Must be called BEFORE wrapping trainable layers with LoRA adapters.
    """
    stats = {
        "linear_quantized": 0,
        "embedding_quantized": 0,
        "skipped_quantized": 0,
        "skipped_lora": 0,
        "skipped_shape": 0,
    }
    target = model.language_model if hasattr(model, "language_model") else model

    def _predicate(path: str, module: nn.Module) -> bool:
        if hasattr(module, "lora_a") or hasattr(module, "lora_b"):
            stats["skipped_lora"] += 1
            return False
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            stats["skipped_quantized"] += 1
            return False
        if isinstance(module, nn.Linear):
            in_dim = int(module.weight.shape[-1])
            if in_dim % group_size != 0:
                stats["skipped_shape"] += 1
                return False
            stats["linear_quantized"] += 1
            return True
        if isinstance(module, nn.Embedding):
            leaf = path.split(".")[-1]
            if leaf == "embed_tokens_per_layer" and quantize_ple_embedding:
                if int(module.weight.shape[-1]) % group_size == 0:
                    stats["embedding_quantized"] += 1
                    return True
                stats["skipped_shape"] += 1
                return False
            if leaf == "embed_tokens" and quantize_embed_tokens:
                if int(module.weight.shape[-1]) % group_size == 0:
                    stats["embedding_quantized"] += 1
                    return True
                stats["skipped_shape"] += 1
                return False
        return False

    nn.quantize(target, group_size=group_size, bits=bits, class_predicate=_predicate)
    return stats


def cast_lora_adapters_dtype(model: nn.Module, dtype: mx.Dtype = mx.bfloat16) -> int:
    """Cast all trainable LoRA ``(lora_a, lora_b)`` adapter matrices to ``dtype`` (default ``mx.bfloat16``)."""
    casted = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
            mod.lora_a = mod.lora_a.astype(dtype)
            mod.lora_b = mod.lora_b.astype(dtype)
            casted += 1
    return casted


def unwrap_kv_shared_lora_modules(model: Any) -> int:
    """Unwrap dead ``k_proj`` / ``v_proj`` LoRA wrappers on KV-shared layers while preserving base (quantized or BF16) linears."""
    tm = model.language_model.model
    unwrapped = 0
    for idx, layer in enumerate(tm.layers):
        if tm.previous_kvs[idx] != idx:
            attn = layer.self_attn
            if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "lora_a") and hasattr(attn.k_proj, "linear"):
                attn.k_proj = attn.k_proj.linear
                unwrapped += 1
            if hasattr(attn, "v_proj") and hasattr(attn.v_proj, "lora_a") and hasattr(attn.v_proj, "linear"):
                attn.v_proj = attn.v_proj.linear
                unwrapped += 1
    return unwrapped


class QuantizedMoELoRALinear(nn.Module):
    """Frozen quantized MoE expert bank + trainable ``BF16`` LoRA using sorted ``mx.gather_qmm``.

    Always sorts expert assignments with ``mx.argsort`` and passes explicit ``lhs_indices``,
    ``rhs_indices``, and ``sorted_indices=True`` so batched ``M >= 2`` MoE forward and VJP
    avoid redundant token copies and unsorted expert scatter penalties.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        *,
        rank: int = 8,
        scale: float = 2.0,
        dropout: float = 0.0,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.scale = float(scale)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)

        init_s = 1.0 / math.sqrt(self.input_dims)
        w_init = mx.random.uniform(
            low=-init_s,
            high=init_s,
            shape=(self.num_experts, self.output_dims, self.input_dims),
            dtype=dtype,
        )
        wq, sc, *maybe_b = mx.quantize(
            w_init, group_size=self.group_size, bits=self.bits, mode=self.mode
        )
        self.weight = wq
        self.scales = sc
        self.biases = maybe_b[0] if maybe_b else None
        self.freeze()

        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = mx.random.uniform(
            low=-init_s,
            high=init_s,
            shape=(self.num_experts, self.rank, self.input_dims),
            dtype=dtype,
        )
        self.lora_b = mx.zeros(
            (self.num_experts, self.output_dims, self.rank), dtype=dtype
        )

    @classmethod
    def from_dense(
        cls,
        weight: mx.array,
        *,
        rank: int = 8,
        scale: float = 2.0,
        dropout: float = 0.0,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine",
    ) -> "QuantizedMoELoRALinear":
        """Quantize a dense ``(E, D_out, D_in)`` expert weight tensor and attach ``BF16`` LoRA adapters."""
        num_experts, output_dims, input_dims = (int(d) for d in weight.shape)
        mod = cls(
            input_dims=input_dims,
            output_dims=output_dims,
            num_experts=num_experts,
            rank=rank,
            scale=scale,
            dropout=dropout,
            group_size=group_size,
            bits=bits,
            mode=mode,
            dtype=weight.dtype,
        )
        wq, sc, *maybe_b = mx.quantize(
            weight, group_size=group_size, bits=bits, mode=mode
        )
        mod.weight = wq
        mod.scales = sc
        mod.biases = maybe_b[0] if maybe_b else None
        mod.freeze(keys=["weight", "scales", "biases"])
        return mod

    def __call__(self, x: mx.array, expert_indices: mx.array) -> mx.array:
        """Project ``x`` (``(..., D_in)``) through selected ``expert_indices`` (``(..., K)``)."""
        orig_idx_shape = expert_indices.shape
        k_top = int(orig_idx_shape[-1])
        m_tokens = int(x.size // self.input_dims)
        x_flat = x.reshape((m_tokens, 1, 1, self.input_dims))

        flat_experts = mx.stop_gradient(expert_indices.reshape((-1,)))
        order = mx.argsort(flat_experts)
        inv_order = mx.argsort(order)
        rhs_indices = flat_experts[order]
        lhs_indices = order // k_top

        y_base = mx.gather_qmm(
            x_flat,
            self.weight,
            self.scales,
            self.biases,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=True,
        )

        x_drop = self.dropout(x_flat)
        z = mx.gather_mm(
            x_drop,
            self.lora_a.swapaxes(-1, -2),
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            sorted_indices=True,
        )
        z = mx.gather_mm(
            z,
            self.lora_b.swapaxes(-1, -2),
            rhs_indices=rhs_indices,
            sorted_indices=True,
        )
        scale_arr = mx.array(self.scale, dtype=x.dtype)
        y_sorted = y_base + scale_arr * z
        y = y_sorted[inv_order]
        return y.reshape((*orig_idx_shape, self.output_dims))


# ---------------------------------------------------------------------------
# P1.7: Differentiable Shared-Prefix KV Broadcast Across grad_accum
# ---------------------------------------------------------------------------


@dataclass
class SharedSuffixBin:
    """One suffix micro-batch (single doc or packed suffix bin) attending to a shared system prefix."""

    suffix_input_ids: mx.array
    suffix_position_ids: mx.array
    suffix_segment_ids: mx.array
    ans_indices: mx.array
    y_flat: mx.array
    token_weights: mx.array | None
    num_examples: int
    real_suffix_tokens: int
    logical_total_tokens: int
    bin_suffix_tokens: int
    segment_spans: list[tuple[int, int, int]]


def find_shared_prefix_len(
    examples: Sequence[tuple[list[int], int]], *, min_len: int = 32
) -> int:
    """Return longest common prefix length strictly before every example's first answer predictor.

    Capped at ``min(prompt_len - 1)`` so every answer predictor position ``prompt_len - 1``
    lies strictly inside the suffix stream (never inside ``sys_ids``).
    """
    if not examples:
        return 0
    max_cap = min(int(p_len) - 1 for _, p_len in examples)
    if max_cap < min_len:
        return 0
    ref = examples[0][0]
    common = min(max_cap, len(ref) - 1)
    for ids, _ in examples[1:]:
        limit = min(common, len(ids) - 1)
        i = 0
        while i < limit and ids[i] == ref[i]:
            i += 1
        common = i
        if common < min_len:
            return 0
    return common if common >= min_len else 0


def pack_suffixes_for_shared_prefix(
    examples: Sequence[tuple[list[int], int]],
    *,
    sys_len: int,
    pack_tokens: int = 0,
    pad_to_fixed: bool = False,
    pad_multiple: int = 1,
    pad_token_id: int = 0,
    loss_norm: Literal["token", "example"] = "token",
) -> SharedSuffixBin:
    """Strip ``sys_len`` leading tokens from each example and pack the remaining suffixes."""
    if sys_len <= 0:
        raise ValueError(f"sys_len must be positive, got {sys_len}")
    if not examples:
        raise ValueError("Cannot build SharedSuffixBin from empty examples.")

    in_ids: list[int] = []
    pos_ids: list[int] = []
    seg_ids: list[int] = []
    ans_indices: list[int] = []
    y_flat: list[int] = []
    weights_list: list[float] = []
    segment_spans: list[tuple[int, int, int]] = []
    logical_total = 0

    k_docs = len(examples)
    cursor = 0
    for seg_i, (full_ids, p_len) in enumerate(examples):
        if p_len <= sys_len:
            raise ValueError(
                f"prompt_len={p_len} must be strictly greater than sys_len={sys_len}"
            )
        logical_total += len(full_ids) - 1
        suf_inp = full_ids[sys_len:-1]
        l_suf = len(suf_inp)
        in_ids.extend(suf_inp)
        pos_ids.extend(range(sys_len, sys_len + l_suf))
        seg_ids.extend([seg_i] * l_suf)
        segment_spans.append((cursor, cursor + l_suf, sys_len))

        ans_s = cursor + (p_len - 1 - sys_len)
        ans_e = cursor + l_suf
        ans_indices.extend(range(ans_s, ans_e))
        targets_i = full_ids[p_len:]
        y_flat.extend(targets_i)
        if loss_norm == "example":
            w_i = 1.0 / float(k_docs * len(targets_i))
            weights_list.extend([w_i] * len(targets_i))
        cursor += l_suf

    real_suf = cursor
    if pad_to_fixed and pack_tokens > 0:
        bin_suf = max(pack_tokens, real_suf)
    elif pad_multiple > 1:
        bin_suf = ((real_suf + pad_multiple - 1) // pad_multiple) * pad_multiple
    else:
        bin_suf = real_suf
    pad_n = bin_suf - real_suf
    if pad_n > 0:
        in_ids.extend([pad_token_id] * pad_n)
        pos_ids.extend([sys_len] * pad_n)
        seg_ids.extend([-1] * pad_n)

    token_weights = (
        mx.array(weights_list, dtype=mx.float32) if loss_norm == "example" else None
    )
    return SharedSuffixBin(
        suffix_input_ids=mx.array([in_ids], dtype=mx.int32),
        suffix_position_ids=mx.array([pos_ids], dtype=mx.int32),
        suffix_segment_ids=mx.array([seg_ids], dtype=mx.int32),
        ans_indices=mx.array(ans_indices, dtype=mx.int32),
        y_flat=mx.array(y_flat, dtype=mx.int32),
        token_weights=token_weights,
        num_examples=k_docs,
        real_suffix_tokens=real_suf,
        logical_total_tokens=logical_total,
        bin_suffix_tokens=bin_suf,
        segment_spans=segment_spans,
    )


def _swa_prefix_start(sys_len: int, window_size: int | None) -> int:
    if window_size is None:
        return 0
    return max(0, int(sys_len) - (int(window_size) - 1))


def build_shared_prefix_suffix_masks(
    sys_len: int,
    suffix_segment_ids: mx.array,
    suffix_position_ids: mx.array,
    *,
    window_size: int | None,
    dtype: mx.Dtype,
) -> dict[str, mx.array]:
    """Build additive ``dtype`` masks for suffix tokens attending to ``[prefix_kv, suffix_kv]``.

    Prefix columns are attended to by every real suffix segment (respecting SWA distance),
    while suffix columns use strict block-diagonal segment causal (+ SWA) masking.
    """
    seg = suffix_segment_ids[0] if suffix_segment_ids.ndim == 2 else suffix_segment_ids
    pos = suffix_position_ids[0] if suffix_position_ids.ndim == 2 else suffix_position_ids
    l_suf = int(seg.shape[0])

    q_idx = mx.arange(l_suf, dtype=mx.int32)[:, None]
    k_idx = mx.arange(l_suf, dtype=mx.int32)[None, :]
    same_seg = seg[:, None] == seg[None, :]
    suf_causal = same_seg & (q_idx >= k_idx)

    zero = mx.array(0.0, dtype=dtype)
    neg = mx.array(MASK_NEG_VAL, dtype=dtype)
    is_real_row = (seg >= 0)[:, None]

    # Full attention: real suffix tokens attend to all sys_len prefix columns; pad rows only self-attend.
    pfx_full_ok = mx.broadcast_to(is_real_row, (l_suf, sys_len))
    full_ok = mx.concatenate([pfx_full_ok, suf_causal], axis=1)
    masks: dict[str, mx.array] = {
        "full_attention": mx.where(full_ok, zero, neg)[None, None, :, :]
    }

    if window_size is not None:
        w = int(window_size)
        k_start = _swa_prefix_start(sys_len, w)
        k_pfx_pos = mx.arange(k_start, sys_len, dtype=mx.int32)[None, :]
        q_pos = pos[:, None]
        pfx_swa_ok = is_real_row & (q_pos >= k_pfx_pos) & (q_pos < (k_pfx_pos + w))
        suf_swa_ok = suf_causal & (q_idx < (k_idx + w))
        swa_ok = mx.concatenate([pfx_swa_ok, suf_swa_ok], axis=1)
        masks["sliding_attention"] = mx.where(swa_ok, zero, neg)[None, None, :, :]
    else:
        masks["sliding_attention"] = masks["full_attention"]
    return masks


def _slice_kv_for_layer_type(
    kv: tuple[mx.array, mx.array],
    *,
    is_sliding: bool,
    sys_len: int,
    window_size: int | None,
) -> tuple[mx.array, mx.array]:
    if not is_sliding or window_size is None:
        return kv
    k_start = _swa_prefix_start(sys_len, window_size)
    if k_start <= 0:
        return kv
    k, v = kv
    return k[:, :, k_start:, :], v[:, :, k_start:, :]


def _run_sys_frozen_prefix(
    model: Any,
    sys_ids: mx.array,
    first_lora_layer: int,
) -> tuple[
    mx.array,
    list[Any],
    list[tuple[Any, Any]],
    list[mx.array | None],
    dict[str, RopeTable | None],
    int,
]:
    tm = model.language_model.model
    sys_len = int(sys_ids.shape[1])
    h, pli_list = _compute_embed_and_ple(tm, sys_ids)
    plan_by_type = build_segmented_attention_plans(
        [(0, sys_len, 0)],
        sys_len,
        sys_len=0,
        window_size=getattr(tm, "window_size", None),
        dtype=h.dtype,
    )
    masks: list[Any] = [
        plan_by_type[getattr(layer, "layer_type", "full_attention")]
        for layer in tm.layers
    ]
    rope_tables: dict[str, RopeTable | None] = {
        "full_attention": None,
        "sliding_attention": None,
    }
    intermediates: list[tuple[Any, Any]] = [(None, None)] * len(tm.layers)

    last_storing = max(tm.previous_kvs)
    stop_frozen = min(first_lora_layer, last_storing + 1)
    for idx in range(stop_frozen):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        kvs, off = intermediates[prev_idx]
        h, kvs, off = forward_gemma4_layer(
            layer,
            h,
            masks[idx],
            per_layer_input=pli_list[idx],
            shared_kv=kvs,
            offset=off or 0,
        )
        intermediates[idx] = (kvs, off)

    return h, masks, intermediates, pli_list, rope_tables, last_storing


def _run_sys_upper_storing(
    model: Any,
    h_sys: mx.array,
    sys_masks: list[Any],
    sys_prefix_intermediates: list[tuple[Any, Any]],
    sys_pli_list: list[mx.array | None],
    sys_rope_tables: dict[str, RopeTable | None],
    *,
    first_lora_layer: int,
    last_storing_layer: int,
    sys_len: int,
    use_ckpt: bool = False,
) -> tuple[list[int], list[tuple[mx.array, mx.array]]]:
    """Run ``sys_ids`` through trainable storing layers only (skipping KV-shared upper layers)."""
    tm = model.language_model.model
    window_size = getattr(tm, "window_size", None)
    intermediates = list(sys_prefix_intermediates)
    h = h_sys
    storing_indices: list[int] = []
    storing_kvs: list[tuple[mx.array, mx.array]] = []

    for idx in range(first_lora_layer, last_storing_layer + 1):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        kvs, off = intermediates[prev_idx]
        ltype = getattr(layer, "layer_type", "full_attention")
        r_table = sys_rope_tables.get(ltype)
        m_layer = sys_masks[idx]
        if use_ckpt:

            def _ckpt_fn(
                h_in: mx.array,
                pli_in: mx.array | None,
                kvs_in: Any,
                _l: Any = layer,
                _m: Any = m_layer,
                _rt: RopeTable | None = r_table,
            ) -> tuple[mx.array, tuple[mx.array, mx.array], int]:
                return forward_gemma4_layer(
                    _l,
                    h_in,
                    _m,
                    per_layer_input=pli_in,
                    shared_kv=kvs_in,
                    rope_table=_rt,
                )

            h, kvs, off = nn.utils.checkpoint(layer, _ckpt_fn)(
                h, sys_pli_list[idx], kvs
            )
        else:
            h, kvs, off = forward_gemma4_layer(
                layer,
                h,
                m_layer,
                per_layer_input=sys_pli_list[idx],
                shared_kv=kvs,
                rope_table=r_table,
            )
        intermediates[idx] = (kvs, off)
        if prev_idx == idx:
            storing_indices.append(idx)
            storing_kvs.append(
                _slice_kv_for_layer_type(
                    kvs,
                    is_sliding=(ltype == "sliding_attention"),
                    sys_len=sys_len,
                    window_size=window_size,
                )
            )
    return storing_indices, storing_kvs


def _run_suffix_frozen_prefix(
    model: Any,
    suffix_bin: SharedSuffixBin,
    sys_prefix_intermediates: list[tuple[Any, Any]],
    *,
    sys_len: int,
    first_lora_layer: int,
) -> tuple[
    mx.array,
    list[Any],
    list[tuple[Any, Any]],
    list[mx.array | None],
    dict[str, RopeTable | None],
]:
    tm = model.language_model.model
    window_size = getattr(tm, "window_size", None)
    h, pli_list = _compute_embed_and_ple(tm, suffix_bin.suffix_input_ids)
    plan_by_type = build_segmented_attention_plans(
        suffix_bin.segment_spans,
        suffix_bin.bin_suffix_tokens,
        sys_len=sys_len,
        window_size=window_size,
        dtype=h.dtype,
    )
    masks: list[Any] = [
        plan_by_type[getattr(layer, "layer_type", "full_attention")]
        for layer in tm.layers
    ]
    rope_tables: dict[str, RopeTable | None] = {
        "full_attention": None,
        "sliding_attention": None,
    }
    intermediates: list[tuple[Any, Any]] = [(None, None)] * len(tm.layers)

    for idx in range(first_lora_layer):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        ltype = getattr(layer, "layer_type", "full_attention")
        if prev_idx == idx:
            pfx_raw = sys_prefix_intermediates[idx][0]
            pfx_kv = _slice_kv_for_layer_type(
                pfx_raw,
                is_sliding=(ltype == "sliding_attention"),
                sys_len=sys_len,
                window_size=window_size,
            )
            shared_kv = None
        else:
            pfx_kv = None
            shared_kv = intermediates[prev_idx][0]

        h, kvs, off = forward_gemma4_layer(
            layer,
            h,
            masks[idx],
            per_layer_input=pli_list[idx],
            shared_kv=shared_kv,
            prefix_kv=pfx_kv,
        )
        intermediates[idx] = (kvs, off)

    return h, masks, intermediates, pli_list, rope_tables


def _compute_suffix_tail_ce_loss(
    model: Any,
    kv_sys_upper: list[tuple[mx.array, mx.array]],
    storing_indices: list[int],
    h: mx.array,
    masks: list[Any],
    suffix_prefix_intermediates: list[tuple[Any, Any]],
    pli_list: list[mx.array | None],
    rope_tables: dict[str, RopeTable | None],
    ans_indices: mx.array,
    y_flat: mx.array,
    token_weights: mx.array | None,
    first_lora_layer: int,
    *,
    use_ckpt: bool = False,
    chunk_size: int = 256,
    compiled_ce: bool = True,
) -> mx.array:
    lm = model.language_model
    tm = lm.model
    intermediates = list(suffix_prefix_intermediates)
    kv_by_layer = {
        layer_idx: kv_pair for layer_idx, kv_pair in zip(storing_indices, kv_sys_upper)
    }

    for idx in range(first_lora_layer, len(tm.layers)):
        layer = tm.layers[idx]
        prev_idx = tm.previous_kvs[idx]
        ltype = getattr(layer, "layer_type", "full_attention")
        r_table = rope_tables.get(ltype)
        m_layer = masks[idx]
        if prev_idx == idx:
            pfx_kv = kv_by_layer.get(idx)
            shared_kv = None
        else:
            pfx_kv = None
            shared_kv = intermediates[prev_idx][0]

        if use_ckpt:

            def _ckpt_fn(
                h_in: mx.array,
                pli_in: mx.array | None,
                kvs_in: Any,
                pfx_in: Any,
                _l: Any = layer,
                _m: Any = m_layer,
                _rt: RopeTable | None = r_table,
            ) -> tuple[mx.array, tuple[mx.array, mx.array], int]:
                return forward_gemma4_layer(
                    _l,
                    h_in,
                    _m,
                    per_layer_input=pli_in,
                    shared_kv=kvs_in,
                    rope_table=_rt,
                    prefix_kv=pfx_in,
                )

            h, kvs, off = nn.utils.checkpoint(layer, _ckpt_fn)(
                h, pli_list[idx], shared_kv, pfx_kv
            )
        else:
            h, kvs, off = forward_gemma4_layer(
                layer,
                h,
                m_layer,
                per_layer_input=pli_list[idx],
                shared_kv=shared_kv,
                rope_table=r_table,
                prefix_kv=pfx_kv,
            )
        intermediates[idx] = (kvs, off)

    h_norm = tm.norm(h)
    h_ans_2d = h_norm[0, ans_indices, :]
    lm_head_fn = (
        tm.embed_tokens.as_linear
        if getattr(lm, "tie_word_embeddings", True)
        else lm.lm_head
    )
    return chunked_softcap_ce_loss(
        lm_head_fn,
        h_ans_2d,
        y_flat,
        softcap=getattr(lm, "final_logit_softcapping", None),
        token_weights=token_weights,
        chunk_size=chunk_size,
        compiled=compiled_ce,
    )


def value_and_grad_model_and_aux(
    model: nn.Module, fn: Callable[..., mx.array]
) -> Callable[..., tuple[mx.array, tuple[Any, Any]]]:
    """Differentiate ``fn(model, aux, *args, **kwargs)`` w.r.t. both ``model.trainable_parameters()`` and ``aux``."""

    def _wrapped(params: Any, aux: Any, *args: Any, **kwargs: Any) -> mx.array:
        model.update(params)
        return fn(model, aux, *args, **kwargs)

    vg = mx.value_and_grad(_wrapped, argnums=(0, 1))

    def _call(aux: Any, *args: Any, **kwargs: Any) -> tuple[mx.array, tuple[Any, Any]]:
        orig = model.trainable_parameters()
        try:
            return vg(orig, aux, *args, **kwargs)
        finally:
            model.update(orig)

    return _call


def run_shared_prefix_grad_accum_step(
    model: Any,
    sys_ids: list[int],
    suffix_bins: Sequence[SharedSuffixBin],
    first_lora_layer: int,
    *,
    micro_weights: Sequence[float] | None = None,
    ckpt_tokens: int = 2200,
    chunk_size: int = 256,
    compiled_ce: bool = True,
    sys_dropout_seed: int | None = None,
) -> tuple[float, Any, list[float]]:
    """Compute exact accumulated loss & LoRA gradients over ``K`` suffix micro-batches sharing ``sys_ids``.

    Uses:
      - Single-graph ``nn.value_and_grad`` when ``len(suffix_bins) == 1`` (zero second prefix forward).
      - Exact two-stage VJP when ``len(suffix_bins) > 1`` so peak activation memory stays ``O(1)``
        in ``grad_accum`` while ``sys_ids`` executes only once through storing layers ``0..last_storing``.
    """
    k_bins = len(suffix_bins)
    if k_bins == 0:
        raise ValueError("suffix_bins must be non-empty")
    if micro_weights is None:
        weights = [1.0 / float(k_bins)] * k_bins
    else:
        weights = [float(w) for w in micro_weights]

    sys_arr = mx.array([sys_ids], dtype=mx.int32)
    sys_len = len(sys_ids)
    (
        h_sys_f,
        sys_masks,
        sys_inter_f,
        sys_pli,
        sys_rope,
        last_storing,
    ) = _run_sys_frozen_prefix(model, sys_arr, first_lora_layer)
    mx.eval(h_sys_f, sys_inter_f)

    # Fast single-microbatch path (e.g. grad_accum=1 with packed suffixes over shared prefix):
    if k_bins == 1:
        sb = suffix_bins[0]
        h_suf_f, suf_masks, suf_inter_f, suf_pli, suf_rope = _run_suffix_frozen_prefix(
            model, sb, sys_inter_f, sys_len=sys_len, first_lora_layer=first_lora_layer
        )
        use_ckpt_sys = (sys_len + sb.bin_suffix_tokens) > ckpt_tokens
        use_ckpt_suf = (sys_len + sb.bin_suffix_tokens) > ckpt_tokens

        def _single_loss(m: Any) -> mx.array:
            s_idx, s_kvs = _run_sys_upper_storing(
                m,
                h_sys_f,
                sys_masks,
                sys_inter_f,
                sys_pli,
                sys_rope,
                first_lora_layer=first_lora_layer,
                last_storing_layer=last_storing,
                sys_len=sys_len,
                use_ckpt=use_ckpt_sys,
            )
            return _compute_suffix_tail_ce_loss(
                m,
                s_kvs,
                s_idx,
                h_suf_f,
                suf_masks,
                suf_inter_f,
                suf_pli,
                suf_rope,
                sb.ans_indices,
                sb.y_flat,
                sb.token_weights,
                first_lora_layer,
                use_ckpt=use_ckpt_suf,
                chunk_size=chunk_size,
                compiled_ce=compiled_ce,
            )

        loss_arr, grads = nn.value_and_grad(model, _single_loss)(model)
        if weights[0] != 1.0:
            w0 = mx.array(weights[0], dtype=mx.float32)
            grads = tree_map(lambda g: g * w0.astype(g.dtype), grads)
        mx.eval(loss_arr, grads)
        l_val = float(loss_arr.item())
        return l_val * weights[0], grads, [l_val]

    # Multi-microbatch exact Two-Stage VJP path:
    use_ckpt_sys = sys_len > ckpt_tokens
    if sys_dropout_seed is not None and model.training:
        mx.random.seed(sys_dropout_seed)
    storing_indices, kv_sys_upper = _run_sys_upper_storing(
        model,
        h_sys_f,
        sys_masks,
        sys_inter_f,
        sys_pli,
        sys_rope,
        first_lora_layer=first_lora_layer,
        last_storing_layer=last_storing,
        sys_len=sys_len,
        use_ckpt=use_ckpt_sys,
    )
    if kv_sys_upper:
        mx.eval(kv_sys_upper)

    def _suffix_fn(
        m: Any,
        kv_upper: list[tuple[mx.array, mx.array]],
        h_suf_f_: mx.array,
        suf_masks_: list[mx.array],
        suf_inter_f_: list[tuple[Any, Any]],
        suf_pli_: list[mx.array | None],
        suf_rope_: dict[str, RopeTable],
        ans_idx_: mx.array,
        y_flat_: mx.array,
        tok_w_: mx.array | None,
        use_ckpt_: bool,
    ) -> mx.array:
        return _compute_suffix_tail_ce_loss(
            m,
            kv_upper,
            storing_indices,
            h_suf_f_,
            suf_masks_,
            suf_inter_f_,
            suf_pli_,
            suf_rope_,
            ans_idx_,
            y_flat_,
            tok_w_,
            first_lora_layer,
            use_ckpt=use_ckpt_,
            chunk_size=chunk_size,
            compiled_ce=compiled_ce,
        )

    if storing_indices:
        vg_suffix = value_and_grad_model_and_aux(model, _suffix_fn)
    else:

        def _suffix_only_model(
            m: Any,
            h_suf_f_: mx.array,
            suf_masks_: list[mx.array],
            suf_inter_f_: list[tuple[Any, Any]],
            suf_pli_: list[mx.array | None],
            suf_rope_: dict[str, RopeTable],
            ans_idx_: mx.array,
            y_flat_: mx.array,
            tok_w_: mx.array | None,
            use_ckpt_: bool,
        ) -> mx.array:
            return _suffix_fn(
                m,
                [],
                h_suf_f_,
                suf_masks_,
                suf_inter_f_,
                suf_pli_,
                suf_rope_,
                ans_idx_,
                y_flat_,
                tok_w_,
                use_ckpt_,
            )

        vg_suffix_model = nn.value_and_grad(model, _suffix_only_model)

    acc_param_grads: Any = None
    acc_kv_grads: list[tuple[mx.array, mx.array]] | None = None
    per_bin_losses: list[float] = []
    weighted_loss = 0.0

    for sb, w_m in zip(suffix_bins, weights):
        h_suf_f, suf_masks, suf_inter_f, suf_pli, suf_rope = _run_suffix_frozen_prefix(
            model, sb, sys_inter_f, sys_len=sys_len, first_lora_layer=first_lora_layer
        )
        use_ckpt_suf = (sys_len + sb.bin_suffix_tokens) > ckpt_tokens
        if storing_indices:
            loss_m, (g_params_m, g_kv_m) = vg_suffix(
                kv_sys_upper,
                h_suf_f,
                suf_masks,
                suf_inter_f,
                suf_pli,
                suf_rope,
                sb.ans_indices,
                sb.y_flat,
                sb.token_weights,
                use_ckpt_suf,
            )
        else:
            loss_m, g_params_m = vg_suffix_model(
                model,
                h_suf_f,
                suf_masks,
                suf_inter_f,
                suf_pli,
                suf_rope,
                sb.ans_indices,
                sb.y_flat,
                sb.token_weights,
                use_ckpt_suf,
            )
            g_kv_m = []
        w_f32 = mx.array(w_m, dtype=mx.float32)
        scaled_gp = tree_map(lambda g, _w=w_f32: g * _w.astype(g.dtype), g_params_m)
        acc_param_grads = (
            scaled_gp
            if acc_param_grads is None
            else tree_map(mx.add, acc_param_grads, scaled_gp)
        )
        if storing_indices:
            scaled_gkv = [
                (
                    gk.astype(mx.float32) * w_f32,
                    gv.astype(mx.float32) * w_f32,
                )
                for gk, gv in g_kv_m
            ]
            if acc_kv_grads is None:
                acc_kv_grads = scaled_gkv
            else:
                acc_kv_grads = [
                    (ak + gk, av + gv)
                    for (ak, av), (gk, gv) in zip(acc_kv_grads, scaled_gkv)
                ]
            mx.eval(loss_m, acc_param_grads, acc_kv_grads)
        else:
            mx.eval(loss_m, acc_param_grads)

        l_val = float(loss_m.item())
        per_bin_losses.append(l_val)
        weighted_loss += w_m * l_val

    if storing_indices and acc_kv_grads is not None:
        if sys_dropout_seed is not None and model.training:
            mx.random.seed(sys_dropout_seed)

        def _sys_vjp_surrogate(m: Any) -> mx.array:
            _, s_kvs = _run_sys_upper_storing(
                m,
                h_sys_f,
                sys_masks,
                sys_inter_f,
                sys_pli,
                sys_rope,
                first_lora_layer=first_lora_layer,
                last_storing_layer=last_storing,
                sys_len=sys_len,
                use_ckpt=use_ckpt_sys,
            )
            total = mx.array(0.0, dtype=mx.float32)
            for (k_t, v_t), (gk_t, gv_t) in zip(s_kvs, acc_kv_grads):
                total = total + (k_t.astype(mx.float32) * gk_t).sum()
                total = total + (v_t.astype(mx.float32) * gv_t).sum()
            return total

        _, sys_param_grads = nn.value_and_grad(model, _sys_vjp_surrogate)(model)
        total_grads = tree_map(mx.add, acc_param_grads, sys_param_grads)
        mx.eval(total_grads)
        if sys_dropout_seed is not None and model.training:
            mx.random.seed(sys_dropout_seed + 1)
    else:
        total_grads = acc_param_grads

    return weighted_loss, total_grads, per_bin_losses


def grad_tree_cosine(grads_a: Any, grads_b: Any) -> tuple[float, float, int]:
    """Compute global cosine similarity and worst-tensor cosine similarity between two gradient pytrees."""
    flat_a = dict(tree_flatten(grads_a))
    flat_b = dict(tree_flatten(grads_b))
    common_keys = sorted(set(flat_a.keys()) & set(flat_b.keys()))
    dot_sum = 0.0
    na_sq = 0.0
    nb_sq = 0.0
    min_cos = 1.0
    compared = 0
    for k in common_keys:
        a = flat_a[k].astype(mx.float32)
        b = flat_b[k].astype(mx.float32)
        aa = float((a * a).sum().item())
        bb = float((b * b).sum().item())
        if aa <= 1e-24 and bb <= 1e-24:
            continue
        ab = float((a * b).sum().item())
        dot_sum += ab
        na_sq += aa
        nb_sq += bb
        denom = math.sqrt(max(aa * bb, 1e-30))
        c_k = ab / denom
        if c_k < min_cos:
            min_cos = c_k
        compared += 1
    global_cos = dot_sum / math.sqrt(max(na_sq * nb_sq, 1e-30)) if compared > 0 else 1.0
    return global_cos, min_cos, compared


def group_batches_by_prefix_signature(
    batches: Sequence[list[tuple[list[int], int]]],
    *,
    prefix_probe_len: int = 32,
    grad_accum: int = 2,
    seed: int = 20261117,
) -> list[list[tuple[list[int], int]]]:
    """Group batches with identical leading prefix tokens into contiguous ``grad_accum`` windows."""
    by_sig: dict[tuple[int, ...], list[list[tuple[list[int], int]]]] = {}
    for b in batches:
        if not b:
            continue
        first_ids = b[0][0]
        sig = tuple(first_ids[: min(prefix_probe_len, len(first_ids))])
        by_sig.setdefault(sig, []).append(b)

    rng = random.Random(seed)
    chunks: list[list[list[tuple[list[int], int]]]] = []
    step_w = max(1, int(grad_accum))
    for group in by_sig.values():
        rng.shuffle(group)
        for i in range(0, len(group), step_w):
            chunks.append(group[i : i + step_w])
    rng.shuffle(chunks)
    ordered: list[list[tuple[list[int], int]]] = []
    for ch in chunks:
        ordered.extend(ch)
    return ordered


def build_shared_prefix_packed_groups(
    items: Sequence[tuple[list[int], int]],
    *,
    pack_tokens: int = 0,
    min_shared_prefix: int = 32,
    grad_accum: int = 2,
    seed: int = 20261117,
) -> list[list[tuple[list[int], int]]]:
    """Pack items sharing a common system prefix into suffix-budget bins grouped by ``grad_accum``.

    When ``pack_tokens > 0``, suffix tokens (excluding the shared prefix) are packed up to
    ``max(64, pack_tokens - common_prefix_len)`` per bin, eliminating repeated system-prompt
    tokens both within each packed bin and across ``grad_accum`` micro-batches.
    """
    by_sig: dict[tuple[int, ...], list[tuple[list[int], int]]] = {}
    for it in items:
        ids = it[0]
        sig = tuple(ids[: min(min_shared_prefix, len(ids))])
        by_sig.setdefault(sig, []).append(it)

    rng = random.Random(seed)
    chunks: list[list[list[tuple[list[int], int]]]] = []
    step_w = max(1, int(grad_accum))

    for group_idx, group in enumerate(by_sig.values()):
        common_len = find_shared_prefix_len(group, min_len=min_shared_prefix)
        if pack_tokens > 0 and common_len >= min_shared_prefix:
            suf_cap = max(64, int(pack_tokens))
            rng_g = random.Random(seed + group_idx + 1)
            perm = list(group)
            rng_g.shuffle(perm)
            group_bins: list[list[tuple[list[int], int]]] = []
            for p0 in range(0, len(perm), 256):
                pool = sorted(
                    perm[p0 : p0 + 256],
                    key=lambda it: max(1, (len(it[0]) - 1) - common_len),
                    reverse=True,
                )
                open_bins: list[list[tuple[list[int], int]]] = []
                open_rem: list[int] = []
                for it in pool:
                    l_suf = max(1, (len(it[0]) - 1) - common_len)
                    if l_suf > suf_cap:
                        group_bins.append([it])
                        continue
                    best_i = -1
                    best_rem = suf_cap + 1
                    for bi, rem in enumerate(open_rem):
                        if l_suf <= rem < best_rem:
                            best_i = bi
                            best_rem = rem
                    if best_i >= 0:
                        open_bins[best_i].append(it)
                        open_rem[best_i] -= l_suf
                    else:
                        open_bins.append([it])
                        open_rem.append(suf_cap - l_suf)
                group_bins.extend(open_bins)
            rng_g.shuffle(group_bins)
        elif pack_tokens > 0:
            group_bins = build_packed_groups(
                group, pack_tokens=pack_tokens, seed=seed + group_idx + 1
            )
        else:
            perm = list(group)
            rng.shuffle(perm)
            group_bins = [[it] for it in perm]

        for i in range(0, len(group_bins), step_w):
            chunks.append(group_bins[i : i + step_w])

    rng.shuffle(chunks)
    ordered: list[list[tuple[list[int], int]]] = []
    for ch in chunks:
        ordered.extend(ch)
    return ordered

