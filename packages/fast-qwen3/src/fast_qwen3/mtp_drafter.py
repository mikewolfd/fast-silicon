"""Native 1-Layer Qwen3.5/3.6 `MTP` Drafter + `PromptLookup` fallback + Zero-GEMM GDN Rollback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .kernels import gpu_argmax_conf, gpu_argmax_nan
from .model import (
    DeltaSlotCache,
    FastQwen3Model,
    GrowingKVCache,
    Qwen3Config,
    Qwen3DecoderBlock,
    build_linear_from_weights,
)

MAX_MTP_BATCH_APPLE9: int = 4


_CACHED_APPLE10_NAX: bool | None = None


def _host_supports_apple10_nax() -> bool:
    global _CACHED_APPLE10_NAX
    if _CACHED_APPLE10_NAX is None:
        try:
            from fast_shared.hw_check import supports_apple10_nax

            _CACHED_APPLE10_NAX = bool(supports_apple10_nax())
        except Exception:
            _CACHED_APPLE10_NAX = False
    return _CACHED_APPLE10_NAX


def should_use_speculative_qwen(
    batch_size: int,
    temperature: float,
    *,
    speculative: bool | None = None,
    has_drafter_or_lookup: bool = True,
    has_native_mtp: bool = False,
    hidden_size: int | None = None,
    is_moe: bool = False,
    apple10_nax: bool | None = None,
) -> bool:
    """Gate speculative drafting to `B == 1` on Apple10 NAX, `1 <= B <= 4` on Apple9 (`T == 0`), and `>=9B` dense native MTP in `auto`."""
    if not has_drafter_or_lookup or speculative is False or temperature > 0.0:
        return False
    if speculative is None:
        # Empirical ABBA gate: on <=4B models (H <= 2560) or sparse MoE, verify lm_head / expert overhead exceeds single-step decode,
        # and PromptLookup without native MTP regresses non-repetitive prompts on M4 Pro (0.863x). Auto-enable only on >=9B dense native MTP.
        if is_moe or not has_native_mtp or (hidden_size is not None and int(hidden_size) < 3584):
            return False
    is_nax = _host_supports_apple10_nax() if apple10_nax is None else bool(apple10_nax)
    max_b = 1 if is_nax else MAX_MTP_BATCH_APPLE9
    if speculative is None and int(batch_size) > 1 and is_nax:
        return False
    return 1 <= int(batch_size) <= max_b


def recommended_draft_k(
    batch_size: int,
    *,
    has_grammar: bool = False,
    apple10_nax: bool | None = None,
) -> int:
    b = int(batch_size)
    is_nax = _host_supports_apple10_nax() if apple10_nax is None else bool(apple10_nax)
    max_b = 1 if is_nax else MAX_MTP_BATCH_APPLE9
    if not (1 <= b <= max_b):
        raise ValueError(f"Speculative drafting unsupported for batch_size={b} (max_b={max_b})")
    if b == 1:
        # k=3 outperforms k=4 by +10-13% for 1-layer MTP on both M4 Pro and Base M5 (with or without grammar).
        return 3
    return 2


def detect_mtp_norm_add_one(
    mtp_weights: Mapping[str, mx.array],
    *,
    fallback: bool = False,
) -> bool:
    """Detect zero-centered `(1 + w)` vs pre-shifted `w` independently on `mtp.*` tensors.

    Quantized conversions (e.g., `OptiQ`) may fold `+1.0` into main-model RMSNorm weights
    while shipping `optiq/mtp.safetensors` with raw zero-centered `(1 + w)` HF weights.
    """
    probe_keys = (
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    )
    means: list[float] = []
    pre_fc_means: list[float] = []
    min_val = float("inf")
    for k in probe_keys:
        w = mtp_weights.get(k)
        if w is None:
            continue
        arr = np.asarray(w.astype(mx.float32))
        m = float(np.mean(arr))
        means.append(m)
        min_val = min(min_val, float(np.min(arr)))
        if k.startswith("mtp.pre_fc_norm_"):
            pre_fc_means.append(m)
    if not means:
        return bool(fallback)
    # Raw HF pre-FC norms sit negative (~ -0.5..-0.2); pre-shifted norms sit > +0.5 with min > 0.
    if pre_fc_means and (min(pre_fc_means) < 0.15 or min_val < -0.15):
        return True
    overall_mean = sum(means) / len(means)
    return abs(overall_mean - 1.0) > abs(overall_mean)


class Qwen3MtpDrafter(nn.Module):
    """Native 1-layer Qwen3.5 / Qwen3.6 Multi-Token Predictor (`mtp.*` / `optiq/mtp.safetensors`).

    Architecture verified against `Qwen/Qwen3.5-*` and `mlx-community/Qwen3.5-*-OptiQ`:
      - `pre_fc_norm_embedding` (`RMSNorm[H]`) + `pre_fc_norm_hidden` (`RMSNorm[H]`)
      - `fc`: `Linear(2 * H -> H)` over `concat([norm_emb(next_tok), norm_hid(prev_hidden)], axis=-1)`
      - `layers.0`: 1 Full-Attention `Qwen3DecoderBlock` (`force_full_attention=True`)
      - `norm`: `RMSNorm[H]`, projected through `target_model.project_logits`
    """

    def __init__(
        self,
        cfg: Qwen3Config,
        mtp_weights: Mapping[str, mx.array],
        *,
        norm_add_one: bool,
        use_apple10_force_fused: bool = False,
        enable_fusion: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(cfg.hidden_size)
        eps = float(cfg.rms_norm_eps)

        self.pre_fc_norm_embedding = nn.RMSNorm(hidden, eps=eps)
        ew = mtp_weights["mtp.pre_fc_norm_embedding.weight"].astype(mx.float32)
        self.pre_fc_norm_embedding.weight = (ew + 1.0 if norm_add_one else ew).astype(mx.bfloat16)

        self.pre_fc_norm_hidden = nn.RMSNorm(hidden, eps=eps)
        hw = mtp_weights["mtp.pre_fc_norm_hidden.weight"].astype(mx.float32)
        self.pre_fc_norm_hidden.weight = (hw + 1.0 if norm_add_one else hw).astype(mx.bfloat16)

        self.fc = build_linear_from_weights(mtp_weights, "mtp.fc", 2 * hidden, hidden)
        self.block = Qwen3DecoderBlock(
            0,
            cfg,
            mtp_weights,
            norm_add_one=norm_add_one,
            use_apple10_force_fused=use_apple10_force_fused,
            layer_prefix="mtp.layers.0",
            force_full_attention=True,
            enable_fusion=enable_fusion,
        )
        self.norm = nn.RMSNorm(hidden, eps=eps)
        nw = mtp_weights["mtp.norm.weight"].astype(mx.float32)
        self.norm.weight = (nw + 1.0 if norm_add_one else nw).astype(mx.bfloat16)

    @classmethod
    def from_mtp_weights(
        cls,
        target_model: FastQwen3Model,
        mtp_weights: Mapping[str, mx.array],
    ) -> Qwen3MtpDrafter | None:
        required = (
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc.weight",
            "mtp.norm.weight",
        )
        if not all(k in mtp_weights for k in required):
            return None
        mtp_norm_add_one = detect_mtp_norm_add_one(mtp_weights, fallback=target_model.norm_add_one)
        return cls(
            target_model.cfg,
            mtp_weights,
            norm_add_one=mtp_norm_add_one,
            use_apple10_force_fused=target_model.use_apple10_force_fused,
            enable_fusion=getattr(target_model, "enable_fusion", True),
        )

    def make_cache(self, *, kv_step: int = 256) -> GrowingKVCache:
        return GrowingKVCache(step=kv_step)

    def forward_hidden(
        self,
        next_token_ids: mx.array,
        prev_hidden: mx.array,
        target_model: FastQwen3Model,
        cache: GrowingKVCache | None = None,
    ) -> mx.array:
        emb = target_model.embed_tokens(next_token_ids)
        ne = self.pre_fc_norm_embedding(emb)
        nh = self.pre_fc_norm_hidden(prev_hidden)
        fused_in = self.fc(mx.concatenate([ne, nh], axis=-1))
        out = self.block(fused_in, cache)
        return self.norm(out)


def draft_prompt_lookup_tokens(
    history_ids: Sequence[int],
    max_k: int,
    *,
    max_ngram: int = 4,
    min_ngram: int = 2,
) -> list[int]:
    """Zero-parameter prompt/history `n`-gram lookup drafter when a checkpoint omits `mtp.*` weights."""
    n = len(history_ids)
    if n < min_ngram + 1 or max_k <= 0:
        return []
    for ng in range(min(max_ngram, n - 1), min_ngram - 1, -1):
        needle = tuple(history_ids[-ng:])
        # Search most recent match first
        for pos in range(n - ng - 1, -1, -1):
            if tuple(history_ids[pos : pos + ng]) == needle:
                end = min(n, pos + ng + max_k)
                return [int(t) for t in history_ids[pos + ng : end]]
    return []


def verify_draft_sequence_b1(
    target_model: FastQwen3Model,
    cache: Sequence[DeltaSlotCache | GrowingKVCache],
    last_verified_token: int,
    draft_tokens: Sequence[int],
    *,
    pick_token_fn: Any = None,
    stop_token_ids: frozenset[int],
    rope_offset_delta: int = 0,
    pick_token_gpu_fn: (
        Callable[[int, int, tuple[float, float, float] | None, Callable[[], np.ndarray]], int] | None
    ) = None,
    need_conf: bool = False,
    vocab_limit: int | None = None,
) -> tuple[list[int], mx.array, int]:
    """Verify `K` draft tokens on `target_model` (`B=1`) in one `L_q = K + 1` pass with exact GDN rollback.

    Invariants:
      1. Before the verify pass (`S = K + 1`), every `GrowingKVCache` records `kv_prev_offset` and
         every `DeltaSlotCache` sets `record_verify_trace = True`.
      2. All linear/MoE/conv/attention projections execute ONCE across `[last_verified_token, *draft_tokens]`.
      3. At each draft step `i in 0..K-1`, `pick_token_fn(step_idx=i, logits_1d=...)` or
         `pick_token_gpu_fn(step_idx, top1_idx, stats, lazy_logits_fn)` selects the exact greedy
         (or grammar-constrained greedy) target token `t_target`.
      4. If all `K` drafts match (`n_accepted_drafts == K`), `cache` already holds the exact state after
         all `K + 1` input tokens, and we emit `[d_0, ..., d_{K-1}, bonus_tok]` (`K + 1` tokens in 1 pass,
         `0` rollback ops!).
      5. If mismatch (or EOS) occurs at draft `j < K`, we emit `[d_0, ..., d_{j-1}, t_target_j]`, rewind
         every `GrowingKVCache` to `kv_prev_offset + j + 1`, and advance every `DeltaSlotCache` over only
         the `j + 1` verified steps using its cached `(q, k, v, g, beta)` slices (`0` GEMM recompute!).
    """
    k_len = len(draft_tokens)
    seq = [int(last_verified_token)] + [int(d) for d in draft_tokens]
    inp = mx.array([seq], dtype=mx.int32)

    kv_prev_offsets: list[int] = []
    for entry in cache:
        if isinstance(entry, GrowingKVCache):
            kv_prev_offsets.append(entry.offset)
        elif isinstance(entry, DeltaSlotCache):
            entry.record_verify_trace = True

    hidden = target_model.forward_hidden(inp, cache, rope_offset_delta=rope_offset_delta)
    logits = target_model.project_logits(hidden)

    emitted: list[int] = []
    consumed_inputs = 0  # How many tokens of `seq` remain valid in `cache`

    if pick_token_gpu_fn is not None:
        logits_2d = logits[0]
        if vocab_limit is not None and int(logits_2d.shape[-1]) > int(vocab_limit):
            logits_2d = logits_2d[:, : int(vocab_limit)]
        if need_conf:
            idx_mx, stats_mx = gpu_argmax_conf(logits_2d)
            mx.eval(hidden, idx_mx, stats_mx)  # guard: allow-eval
            idx_list = idx_mx.tolist()  # guard: allow-sync
            stats_list: list[list[float]] | None = stats_mx.tolist()  # guard: allow-sync
        else:
            idx_mx = gpu_argmax_nan(logits_2d)
            mx.eval(hidden, idx_mx)  # guard: allow-eval
            idx_list = idx_mx.tolist()  # guard: allow-sync
            stats_list = None

        def _lazy_row_np(step_i: int) -> np.ndarray:
            row_f32 = logits_2d[step_i].astype(mx.float32)
            mx.eval(row_f32)  # guard: allow-eval
            return np.asarray(row_f32)  # guard: allow-sync

        for i in range(k_len):
            st_i = (
                (float(stats_list[i][0]), float(stats_list[i][1]), float(stats_list[i][2]))
                if stats_list is not None
                else None
            )
            target_tok = int(pick_token_gpu_fn(i, int(idx_list[i]), st_i, lambda idx=i: _lazy_row_np(idx)))
            consumed_inputs = i + 1
            if target_tok == int(draft_tokens[i]):
                emitted.append(target_tok)
                if target_tok in stop_token_ids:
                    break
            else:
                emitted.append(target_tok)
                break
        else:
            st_k = (
                (float(stats_list[k_len][0]), float(stats_list[k_len][1]), float(stats_list[k_len][2]))
                if stats_list is not None
                else None
            )
            bonus_tok = int(
                pick_token_gpu_fn(k_len, int(idx_list[k_len]), st_k, lambda idx=k_len: _lazy_row_np(idx))
            )
            emitted.append(bonus_tok)
            consumed_inputs = k_len + 1
    else:
        if pick_token_fn is None:
            raise ValueError("verify_draft_sequence_b1 requires pick_token_fn or pick_token_gpu_fn")
        mx.eval(hidden, logits)  # guard: allow-eval
        logits_np = np.asarray(logits[0].astype(mx.float32))  # guard: allow-sync

        for i in range(k_len):
            target_tok = int(pick_token_fn(i, logits_np[i]))
            consumed_inputs = i + 1
            if target_tok == int(draft_tokens[i]):
                emitted.append(target_tok)
                if target_tok in stop_token_ids:
                    break
            else:
                emitted.append(target_tok)
                break
        else:
            # All K drafts matched without hitting stop_token_ids -> sample bonus token K!
            bonus_tok = int(pick_token_fn(k_len, logits_np[k_len]))
            emitted.append(bonus_tok)
            consumed_inputs = k_len + 1

    total_verify_len = k_len + 1
    if consumed_inputs == total_verify_len:
        for entry in cache:
            if isinstance(entry, DeltaSlotCache):
                entry.clear_verify_trace()
    else:
        kv_idx = 0
        conv_k = target_model.cfg.linear_conv_kernel_dim
        for entry in cache:
            if isinstance(entry, GrowingKVCache):
                entry.rewind_to(kv_prev_offsets[kv_idx] + consumed_inputs)
                kv_idx += 1
            elif isinstance(entry, DeltaSlotCache):
                entry.rollback_to_accepted(consumed_inputs, conv_k)
        # Ensure rolled-back GDN state is evaluated cleanly
        rec_arrays = [e.rec_state for e in cache if isinstance(e, DeltaSlotCache) and e.rec_state is not None]
        conv_arrays = [e.conv_state for e in cache if isinstance(e, DeltaSlotCache) and e.conv_state is not None]
        if rec_arrays or conv_arrays:
            mx.eval(*rec_arrays, *conv_arrays)

    accepted_hidden = hidden[:, :consumed_inputs, :]
    return emitted, accepted_hidden, consumed_inputs - 1
