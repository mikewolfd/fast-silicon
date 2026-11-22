"""Batched ragged decode + Prompt-Lookup N-gram, Causal Forced-Span Jump & Async Hybrid ANE+GPU Rolling Decode."""

from __future__ import annotations

import collections
import os
import queue
import threading
import time
from typing import Any, Sequence
import warnings

import mlx.core as mx
import numpy as np

from .ane_pipeline import (
    ANE_CHUNK_S,
    ANE_P_FULL,
    ANE_P_SWA,
    AnePrefixRunner,
    AsyncHybridPrefillRing,
    SlotCacheMetalBinding,
    is_ane_resident_bundle,
    is_w8_bundle_calibrated,
    validate_slot_kv_scales_finite,
)
from .grammar import GrammarHelper, sample_unconstrained
from .model import FastGemma4Model, safe_argmax_with_nan_guard


def build_ngram_table(
    ids_list: Sequence[int],
) -> tuple[dict[tuple[int, int, int], list[int]], dict[tuple[int, int], list[int]]]:
    t3: dict[tuple[int, int, int], list[int]] = {}
    t2: dict[tuple[int, int], list[int]] = {}
    n = len(ids_list)
    for i in range(n - 3):
        k3 = (int(ids_list[i]), int(ids_list[i + 1]), int(ids_list[i + 2]))
        if k3 not in t3:
            t3[k3] = [
                int(ids_list[i + 3]),
                int(ids_list[i + 4]) if i + 4 < n else int(ids_list[i + 3]),
                int(ids_list[i + 5]) if i + 5 < n else int(ids_list[i + 3]),
            ]
    for i in range(n - 2):
        k2 = (int(ids_list[i]), int(ids_list[i + 1]))
        if k2 not in t2:
            t2[k2] = [
                int(ids_list[i + 2]),
                int(ids_list[i + 3]) if i + 3 < n else int(ids_list[i + 2]),
                int(ids_list[i + 4]) if i + 4 < n else int(ids_list[i + 2]),
            ]
    return t3, t2


_REPEAT_4GRAM_BREAKER_DEFAULT: bool = True
_REPEAT_4GRAM_WINDOW: int = 96
_REPEAT_4GRAM_MIN_HITS: int = 4
_REPEAT_4GRAM_MIN_HITS_CONSTRAINED: int = 5
_REPEAT_4GRAM_PENALTY: float = 8.0


def _repeated_4gram_penalties(
    token_ids: Sequence[int] | None,
    *,
    window: int = _REPEAT_4GRAM_WINDOW,
    min_prior_hits: int = _REPEAT_4GRAM_MIN_HITS,
    base_penalty: float = _REPEAT_4GRAM_PENALTY,
    stop_token_ids: frozenset[int] = frozenset(),
) -> dict[int, float]:
    """Return `{next_token_id: penalty}` only on true cyclic/dense degenerate repeat loops."""
    if not token_ids:
        return {}
    n = len(token_ids)
    out: dict[int, float] = {}

    # 1. Exact consecutive periodic suffix detector: suppresses stuck `p=1..8` token cycles
    # without touching legitimate prose, lists, CSV rows, or 2-hit math derivations.
    if n >= 6:
        tail = [int(t) for t in token_ids[-min(n, 32) :]]
        nt = len(tail)
        # p = 1: >= 6 consecutive identical tokens (avoids false hits on markdown/ellipsis)
        if nt >= 6 and all(t == tail[-1] for t in tail[-6:-1]):
            d = tail[-1]
            if d not in stop_token_ids:
                out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)
        # p = 2: >= 4 consecutive 2-token copies
        if nt >= 8 and tail[-2:] == tail[-4:-2] == tail[-6:-4] == tail[-8:-6]:
            d = tail[-2]
            if d not in stop_token_ids:
                out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)
        # p = 3..8: >= 3 consecutive p-token copies
        for p in range(3, 9):
            if nt >= 3 * p:
                pat = tail[-p:]
                if tail[-2 * p : -p] == pat and tail[-3 * p : -2 * p] == pat:
                    d = pat[0]
                    if d not in stop_token_ids:
                        out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)

    # 2. High-threshold 4-gram frequency check inside trailing window (5th+ occurrence).
    if n >= (3 + min_prior_hits):
        start = max(0, n - int(window))
        a, b, c = int(token_ids[-3]), int(token_ids[-2]), int(token_ids[-1])
        if c in token_ids[start : n - 1]:
            hits: dict[int, int] = {}
            for i in range(start, n - 3):
                if int(token_ids[i]) == a and int(token_ids[i + 1]) == b and int(token_ids[i + 2]) == c:
                    d = int(token_ids[i + 3])
                    if d not in stop_token_ids:
                        hits[d] = hits.get(d, 0) + 1
            for d, cnt in hits.items():
                if cnt >= min_prior_hits:
                    pen = float(base_penalty) * float(min(cnt - min_prior_hits + 1, 3))
                    out[d] = max(out.get(d, 0.0), pen)
    return out


def _apply_repeat_penalties_row(
    row_lg: np.ndarray,
    penalties: dict[int, float],
    old_to_new: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Copy a 1D logit row, apply repeat penalties in vocab space, and return `(adj_lg, argmax_j)`."""
    adj = np.array(row_lg, copy=True)
    n_o2n = int(len(old_to_new)) if old_to_new is not None else -1
    v_len = int(adj.shape[0])
    for orig_id, pen in penalties.items():
        idx = int(old_to_new[orig_id]) if (old_to_new is not None and 0 <= orig_id < n_o2n) else orig_id
        if 0 <= idx < v_len:
            adj[idx] -= float(pen)
    return adj, int(np.argmax(adj))


def _host_ram_gib() -> float:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0**3)
    except Exception:
        return 0.0


def should_auto_dual_mlp_q4(
    model: FastGemma4Model,
    batch_size: int,
    *,
    temperature: float = 0.0,
    min_batch: int = 48,
    min_ram_gib: float = 32.0,
) -> bool:
    """Auto-enable Dual-MLP Q4 tails ONLY when large BF16+MPP batches have >=32GiB RAM and M>32 prefix/head protection."""
    return (
        batch_size >= min_batch
        and temperature <= 0.0
        and getattr(model, "mlp_dtype_name", getattr(model, "mlp_dtype", "")) == "bfloat16"
        and bool(getattr(model, "use_mpp", False))
        and _host_ram_gib() >= min_ram_gib
    )


def _seed_tables(
    doc_prompts: Sequence[Sequence[int]],
) -> list[tuple[dict[tuple[int, int, int], list[int]], dict[tuple[int, int], list[int]]]]:
    return [build_ngram_table(p) for p in doc_prompts]


def _update_ngrams_from_tail(t3: dict, t2: dict, ctx: list[int], prev_len: int, max_draft: int = 8) -> int:
    """Incrementally index newly appended context tokens once they have a full draft continuation."""
    n = len(ctx)
    end_idx = n - max_draft - 3
    if end_idx <= prev_len - 3:
        return prev_len
    start = max(0, prev_len - 3)
    for idx in range(start, end_idx):
        rem = ctx[idx + 2 : idx + 2 + max_draft]
        k2 = (ctx[idx], ctx[idx + 1])
        if k2 not in t2:
            t2[k2] = rem
        rem3 = ctx[idx + 3 : idx + 3 + max_draft]
        k3 = (ctx[idx], ctx[idx + 1], ctx[idx + 2])
        if k3 not in t3:
            t3[k3] = rem3
    return end_idx


def _pick_valid_token_safe(
    g_slot: Any,
    model: Any,
    matcher: Any,
    logits_1d: np.ndarray,
    argmax_j: int,
    *,
    x_row_mx: mx.array | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, bool]:
    """Invoke `pick_valid_token` with lazy full-262,144-vocab logit fallback when pruned vocab misses."""
    can_full = (
        x_row_mx is not None
        and getattr(model, "remapper", None) is not None
        and hasattr(model, "compute_full_vocab_logits_1d")
    )
    lazy_full = (lambda: model.compute_full_vocab_logits_1d(x_row_mx, apply_softcap=True)) if can_full else None
    try:
        return g_slot.pick_valid_token(
            matcher,
            logits_1d,
            argmax_j,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
            fallback_to_full_vocab=can_full,
            full_logits_1d=lazy_full,
        )
    except TypeError:
        if temperature > 0.0:
            return g_slot.pick_valid_token(
                matcher, logits_1d, argmax_j, temperature=temperature, top_p=top_p, rng=rng
            )
        return g_slot.pick_valid_token(matcher, logits_1d, argmax_j)


def _parse_use_ane_ring(use_ane_ring: bool | str) -> tuple[bool, bool]:
    """Return `(want_ane, require_ane)` for `False | True | 'auto' | 'require'`."""
    if isinstance(use_ane_ring, bool):
        return use_ane_ring, use_ane_ring
    mode = str(use_ane_ring).strip().lower()
    if mode in ("false", "0", "off", "gpu", "none"):
        return False, False
    if mode in ("true", "1", "on", "ane", "hybrid", "require"):
        return True, True
    if mode == "auto":
        return True, False
    raise ValueError(f"Unsupported use_ane_ring={use_ane_ring!r}; expected False, True, 'auto', or 'require'.")


def _resolve_or_build_ane_ring(
    model: Any,
    prefix_kv: dict[str, Any],
    cache: dict[str, Any],
    suf_list: list[list[int]],
    *,
    use_ane_ring: bool | str,
    ane_ring: Any | None,
    ane_ring_variant: str = "fp16",
    ane_cohort_size: int = 16,
) -> AsyncHybridPrefillRing | None:
    want_ane, require_ane = _parse_use_ane_ring(use_ane_ring)
    if not want_ane:
        return None

    use_kv_i8 = bool(cache.get("kv_i8", False))
    if not use_kv_i8:
        if require_ane:
            raise ValueError("use_ane_ring=True requires kv_i8=True slot cache.")
        return None

    l_sys = int(prefix_kv["l_sys"])
    if l_sys > ANE_P_FULL - ANE_CHUNK_S:
        if require_ane:
            raise ValueError(
                f"Prefix length l_sys={l_sys} exceeds ANE Full geometry (max {ANE_P_FULL - ANE_CHUNK_S} for P_FULL={ANE_P_FULL})."
            )
        return None

    min_auto_batch = max(16, int(ane_cohort_size))
    max_suf = max((len(s) for s in suf_list), default=0)
    if not require_ane and (len(suf_list) < min_auto_batch or max_suf < 64):
        return None

    if ane_ring is not None:
        return ane_ring

    if not require_ane:
        if 0.0 < _host_ram_gib() < 32.0:
            return None
        from .mtp_drafter import _host_supports_apple10_nax

        if _host_supports_apple10_nax():
            return None

    paths = getattr(model, "paths", None)
    model_dir = getattr(model, "model_dir", None)
    if paths is None or not paths.lib_ane_prefix.exists():
        if require_ane:
            raise RuntimeError("libgemma4_ane_prefix.dylib is unavailable for use_ane_ring=True.")
        return None

    exit_layer = getattr(model, "first_kv_shared", 15) - 1
    try:
        bundle = paths.resolve_ane_bundle(
            ane_ring_variant, model_dir=model_dir, local_files_only=True, exit_layer=exit_layer
        )
    except TypeError:
        bundle = paths.resolve_ane_bundle(ane_ring_variant, model_dir=model_dir, local_files_only=True)
    if not is_ane_resident_bundle(paths.lib_ane_prefix, bundle):
        if require_ane:
            raise RuntimeError(
                f"No valid {ane_ring_variant} ANE bundle (`0` CPU runtime ops) found for model_dir={model_dir} (checked {bundle})."
            )
        return None
    assert bundle is not None

    if ane_ring_variant == "w8" and not is_w8_bundle_calibrated(bundle):
        warnings.warn(
            f"ANE ring bundle {bundle.name} is Linear-W8 without a domain-corpus/QAT calibration manifest; "
            "consider ane_ring_variant='fp16' or calibrating with `--awq-mse --calib-file ...`.",
            UserWarning,
            stacklevel=2,
        )

    cached_runner: AnePrefixRunner | None = getattr(model, "_auto_ane_runner", None)
    cached_bundle: str | None = getattr(model, "_auto_ane_bundle", None)
    if cached_runner is None or cached_bundle != str(bundle):
        if cached_runner is not None:
            cached_runner.close()
        cached_runner = AnePrefixRunner(paths.lib_ane_prefix, bundle, pipe_depth=4, out_ring_slots=96)
        model._auto_ane_runner = cached_runner
        model._auto_ane_bundle = str(bundle)

    tables = getattr(model, "_auto_cpu_ane_tables", None)
    if tables is None:
        tables = model.build_cpu_ane_embed_tables()
        model._auto_cpu_ane_tables = tables
    w_emb, w_pli14 = tables
    return AsyncHybridPrefillRing(model, cached_runner, prefix_kv, w_emb, w_pli14)


def run_decode_batch(
    model: FastGemma4Model,
    stop_token_ids: frozenset[int],
    grammar: GrammarHelper | None,
    init_lg: mx.array,
    init_off: mx.array,
    cache: dict[str, Any],
    B: int,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    compact: bool = True,
    compact_ratio: float = 0.55,
    tail_handoff_b: int = 4,
    draft_k: int = 4,
    doc_prompts: Sequence[Sequence[int]] | None = None,
    auto_dual_mlp_q4: bool = False,
    dual_min_batch: int = 48,
    enable_ff_jump: bool = True,
    repeat_4gram_breaker: bool | None = None,
) -> dict[str, Any]:
    """Optimized batched decode: lazy init_lg argmax, eager compaction down to B=1, optional PromptLookup tail handoff."""
    eff_repeat_breaker = _REPEAT_4GRAM_BREAKER_DEFAULT if repeat_4gram_breaker is None else bool(repeat_4gram_breaker)
    use_q4 = auto_dual_mlp_q4 and should_auto_dual_mlp_q4(model, B, temperature=temperature, min_batch=dual_min_batch)
    prev_q4 = bool(getattr(model, "_dual_q4_active", False))
    if use_q4:
        model.attach_dual_mlp_q4()
    model._dual_q4_active = use_q4  # type: ignore[attr-defined]
    try:
        t_dec0 = time.perf_counter()
        new_to_old = model.new_to_old
        old_to_new = getattr(model, "old_to_new", None)
        fallback_pad = grammar.upfront_ff[0] if (grammar is not None and grammar.upfront_ff) else int(new_to_old[0])

        matchers = [grammar.new_matcher() for _ in range(B)] if grammar is not None else None
        queues: list[collections.deque[int]] = [collections.deque() for _ in range(B)]
        gen_toks: list[list[int]] = [[] for _ in range(B)]
        contexts: list[list[int]] = [list(p) for p in doc_prompts] if doc_prompts is not None else [[] for _ in range(B)]
        done = [False] * B
        active = list(range(B))
        ff_skipped = 0
        mask_fallbacks = 0
        llm_steps = 1
        ff_jump_steps = 0
        ff_jump_tokens = 0
        rng = np.random.default_rng(seed)

        def consume_sampled(
            orig_b: int,
            row_lg: np.ndarray | None,
            argmax_j: int,
            x_row_mx: mx.array | None = None,
        ) -> None:
            nonlocal ff_skipped, mask_fallbacks
            if grammar is None:
                if temperature <= 0.0:
                    t_orig = int(new_to_old[argmax_j])
                else:
                    assert row_lg is not None
                    t_orig = sample_unconstrained(
                        row_lg, new_to_old, argmax_j, temperature=temperature, top_p=top_p, rng=rng
                    )
                if t_orig in stop_token_ids:
                    done[orig_b] = True
                    queues[orig_b].clear()
                    return
                gen_toks[orig_b].append(t_orig)
                contexts[orig_b].append(t_orig)
                if len(gen_toks[orig_b]) >= max_tokens:
                    done[orig_b] = True
                    queues[orig_b].clear()
                    return
                queues[orig_b].append(t_orig)
                return

            assert matchers is not None
            m = matchers[orig_b]
            if temperature > 0.0:
                assert row_lg is not None
                t_orig, fell = _pick_valid_token_safe(
                    grammar,
                    model,
                    m,
                    row_lg,
                    argmax_j,
                    x_row_mx=x_row_mx,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                )
                if fell:
                    mask_fallbacks += 1
            elif row_lg is None:
                t_orig = int(new_to_old[argmax_j])
            else:
                t_orig, fell = _pick_valid_token_safe(
                    grammar, model, m, row_lg, argmax_j, x_row_mx=x_row_mx
                )
                if fell:
                    mask_fallbacks += 1

            m.consume_token(t_orig)
            gen_toks[orig_b].append(t_orig)
            contexts[orig_b].append(t_orig)
            if m.is_stopped() and m.is_accepting():
                done[orig_b] = True
                queues[orig_b].clear()
                return
            queues[orig_b].append(t_orig)
            ff = m.compute_ff_tokens()
            if ff:
                m.consume_tokens(ff)
                gen_toks[orig_b].extend(ff)
                contexts[orig_b].extend(ff)
                ff_skipped += len(ff)
                if m.is_stopped() and m.is_accepting():
                    done[orig_b] = True
                    queues[orig_b].clear()
                    return
                queues[orig_b].extend(ff)
            if len(gen_toks[orig_b]) >= max_tokens:
                done[orig_b] = True
                queues[orig_b].clear()

        def _row_repeat_penalties(orig_b: int, *, constrained: bool) -> dict[int, float]:
            if not eff_repeat_breaker:
                return {}
            min_hits = _REPEAT_4GRAM_MIN_HITS_CONSTRAINED if constrained else _REPEAT_4GRAM_MIN_HITS
            return _repeated_4gram_penalties(
                gen_toks[orig_b], min_prior_hits=min_hits, stop_token_ids=stop_token_ids
            )

        def _sample_rows(logits_2d: mx.array, row_indices: Sequence[int]) -> None:
            """Sample `row_indices` from `[len(active), V_eff]` logits, copying float32 rows only when needed."""
            last_x = cache.get("_last_x_in")
            if temperature > 0.0:
                lg_np = np.array(logits_2d.astype(mx.float32))
                nan_rows = [active[r] for r in row_indices if bool(np.isnan(lg_np[r]).any())]
                if nan_rows:
                    raise FloatingPointError(f"NaN logits detected in run_decode_batch for rows {nan_rows}")
                for r in row_indices:
                    b = active[r]
                    row_lg = lg_np[r]
                    pen = _row_repeat_penalties(b, constrained=grammar is not None)
                    if pen:
                        row_lg, am_r = _apply_repeat_penalties_row(row_lg, pen, old_to_new)
                    else:
                        am_r = int(np.argmax(row_lg))
                    xr = last_x[r, -1] if last_x is not None else None
                    consume_sampled(b, row_lg, am_r, x_row_mx=xr)
                return

            am_mx = safe_argmax_with_nan_guard(logits_2d, axis=-1)
            mx.eval(am_mx)
            am_list = am_mx.tolist()
            del am_mx
            bad_rows = [active[r] for r in row_indices if int(am_list[r]) < 0]
            if bad_rows:
                raise FloatingPointError(f"NaN logits detected in run_decode_batch for rows {bad_rows}")
            if grammar is None:
                need_rep_rows: list[tuple[int, dict[int, float]]] = []
                for r in row_indices:
                    b = active[r]
                    t_orig = int(new_to_old[am_list[r]])
                    pen = _row_repeat_penalties(b, constrained=False) if t_orig not in stop_token_ids else {}
                    if pen and t_orig in pen:
                        need_rep_rows.append((r, pen))
                    else:
                        consume_sampled(b, None, am_list[r])
                if need_rep_rows:
                    lg_sub = np.array(logits_2d[mx.array([x[0] for x in need_rep_rows], dtype=mx.int32), :].astype(mx.float32))
                    for k_idx, (r, pen) in enumerate(need_rep_rows):
                        _, am_adj = _apply_repeat_penalties_row(lg_sub[k_idx], pen, old_to_new)
                        consume_sampled(active[r], None, am_adj)
                return

            assert matchers is not None
            need_mask_rows: list[tuple[int, dict[int, float]]] = []
            for r in row_indices:
                b = active[r]
                t_orig = int(new_to_old[am_list[r]])
                pen = _row_repeat_penalties(b, constrained=True) if t_orig not in stop_token_ids else {}
                if (not pen or t_orig not in pen) and matchers[b].validate_tokens([t_orig]):
                    consume_sampled(b, None, am_list[r])
                else:
                    need_mask_rows.append((r, pen))
            if need_mask_rows:
                lg_sub = np.array(logits_2d[mx.array([x[0] for x in need_mask_rows], dtype=mx.int32), :].astype(mx.float32))
                for k_idx, (r, pen) in enumerate(need_mask_rows):
                    row_lg = lg_sub[k_idx]
                    am_j = am_list[r]
                    if pen:
                        row_lg, am_j = _apply_repeat_penalties_row(row_lg, pen, old_to_new)
                    xr = last_x[r, -1] if last_x is not None else None
                    consume_sampled(active[r], row_lg, am_j, x_row_mx=xr)

        # Step-0 lazy argmax (skips full [B, V_eff] float32 copy on Temp=0 unless fallback row hits!)
        _sample_rows(init_lg[:, 0, :], range(len(active)))
        del init_lg

        off_cur = init_off

        while True:
            keep_rows = [r for r, b in enumerate(active) if not done[b]]
            if not keep_rows:
                break

            # Eager dynamic compaction down to B=1, keeping B>16 ratio conservative (<=0.55) to avoid peak RAM spikes
            if compact and len(keep_rows) < len(active):
                need_compact = (len(active) <= 16) or (len(keep_rows) <= int(len(active) * compact_ratio))
                if need_compact:
                    cache = model.compact_cache(cache, keep_rows)
                    off_cur = off_cur[mx.array(keep_rows, dtype=mx.int32)]
                    active = [active[r] for r in keep_rows]

            # In-place tail handoff to PromptLookup L_q>1 once batch shrinks <= tail_handoff_b (Temp=0)!
            if (
                tail_handoff_b > 0
                and temperature <= 0.0
                and doc_prompts is not None
                and len(active) <= tail_handoff_b
            ):
                res_tail = run_decode_prompt_lookup(
                    model,
                    stop_token_ids,
                    grammar,
                    None,
                    off_cur,
                    cache,
                    doc_prompts,
                    B,
                    max_tokens=max_tokens,
                    draft_k=draft_k,
                    enable_ff_jump=enable_ff_jump,
                    repeat_4gram_breaker=eff_repeat_breaker,
                    state={
                        "matchers": matchers,
                        "queues": queues,
                        "gen_toks": gen_toks,
                        "contexts": contexts,
                        "done": done,
                        "active": active,
                        "ff_skipped": ff_skipped,
                        "mask_fallbacks": mask_fallbacks,
                        "verify_steps": llm_steps,
                        "total_acc": sum(len(g) for g in gen_toks),
                        "ff_jump_steps": ff_jump_steps,
                        "ff_jump_tokens": ff_jump_tokens,
                    },
                )
                dec_sec = time.perf_counter() - t_dec0
                res_tail["decode_sec"] = dec_sec
                res_tail["llm_steps"] = res_tail["verify_steps"]
                res_tail["decode_tok_s"] = res_tail["tot_out_tok"] / max(dec_sec, 1e-9)
                res_tail["used_speculative"] = True
                return res_tail

            # Low-batch causal slice jump when all active rows share a forced span `k >= 2`
            live_bs = [b for b in active if not done[b]]
            if enable_ff_jump and 1 <= len(active) <= 8 and len(live_bs) == len(active):
                q_lens = [len(queues[b]) for b in active]
                min_q, max_q = min(q_lens), max(q_lens)
                if min_q >= 2 and min_q == max_q:
                    cur_swa_cap = int(cache["layers"][model.swa_layers[0]]["cap"]) if cache.get("layers") else 512
                    full_cap = int(cache["layers"][model.full_layers[0]]["cap"]) if cache.get("layers") else 1024
                    k_cand = min(64, min_q)
                    swa_safe = (cur_swa_cap >= model.window_size + k_cand - 1) or (full_cap <= cur_swa_cap)
                    if not swa_safe:
                        priv_max = int(mx.max(off_cur).item()) - int(cache["l_sys"])
                        swa_safe = (priv_max + k_cand - 1) < cur_swa_cap
                    if swa_safe:
                        seq_np = np.empty((len(active), k_cand), dtype=np.int32)
                        need_row_idx_j: list[int] = []
                        for r, b in enumerate(active):
                            q = queues[b]
                            for cj in range(k_cand):
                                seq_np[r, cj] = q.popleft()
                            if len(q) == 0:
                                need_row_idx_j.append(r)
                        toks_mx = mx.array(seq_np, dtype=mx.int32)
                        lg = model.forward_step_ragged(
                            toks_mx, cache, off_cur, apply_softcap=True, keep_from_col=k_cand - 1
                        )
                        off_cur = off_cur + k_cand
                        llm_steps += 1
                        ff_jump_steps += 1
                        ff_jump_tokens += k_cand * len(active)
                        if need_row_idx_j:
                            _sample_rows(lg[:, 0, :], need_row_idx_j)
                        del lg
                        continue

            in_toks = []
            need_row_idx: list[int] = []
            for r, b in enumerate(active):
                if done[b]:
                    in_toks.append(fallback_pad)
                else:
                    q = queues[b]
                    tok_id = q.popleft()
                    in_toks.append(tok_id)
                    if len(q) == 0:
                        need_row_idx.append(r)

            toks_mx = mx.array(in_toks, dtype=mx.int32)[:, None]
            lg = model.forward_step_ragged(toks_mx, cache, off_cur, apply_softcap=True)
            off_cur = off_cur + 1
            llm_steps += 1
            _sample_rows(lg[:, 0, :], need_row_idx)
            del lg

        dec_sec = time.perf_counter() - t_dec0
        upfront_len = len(grammar.upfront_ff) if grammar is not None else 0
        tot_gen = sum(len(g) for g in gen_toks) + upfront_len * B
        return {
            "decode_sec": dec_sec,
            "gen_toks": gen_toks,
            "tot_out_tok": tot_gen,
            "llm_steps": llm_steps,
            "ff_tokens": ff_skipped + upfront_len * B,
            "ff_jump_steps": ff_jump_steps,
            "ff_jump_tokens": ff_jump_tokens,
            "mask_fallbacks": mask_fallbacks,
            "decode_tok_s": tot_gen / max(dec_sec, 1e-9),
            "used_speculative": False,
        }
    finally:
        model._dual_q4_active = prev_q4  # type: ignore[attr-defined]


def run_decode_prompt_lookup(
    model: FastGemma4Model,
    stop_token_ids: frozenset[int],
    grammar: GrammarHelper | None,
    init_lg: mx.array | None,
    init_off: mx.array,
    cache: dict[str, Any],
    doc_prompts: Sequence[Sequence[int]],
    B: int,
    *,
    max_tokens: int = 512,
    draft_k: int = 4,
    enable_ff_jump: bool = True,
    repeat_4gram_breaker: bool | None = None,
    # Optional continuation state when handed off from run_decode_batch:
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimized PromptLookup L_q<=draft_k+1 verifier + causal slice jumping (`keep_from_col`) on forced spans."""
    eff_repeat_breaker = _REPEAT_4GRAM_BREAKER_DEFAULT if repeat_4gram_breaker is None else bool(repeat_4gram_breaker)
    t_dec0 = time.perf_counter()
    new_to_old = model.new_to_old
    old_to_new = getattr(model, "old_to_new", None)
    fallback_pad = grammar.upfront_ff[0] if (grammar is not None and grammar.upfront_ff) else int(new_to_old[0])
    L_step = int(draft_k)

    if state is None:
        matchers = [grammar.new_matcher() for _ in range(B)] if grammar is not None else None
        queues: list[collections.deque[int]] = [collections.deque() for _ in range(B)]
        gen_toks: list[list[int]] = [[] for _ in range(B)]
        contexts = [list(p) for p in doc_prompts]
        tables = _seed_tables(doc_prompts)
        idx_pos = [len(c) for c in contexts]
        done = [False] * B
        active = list(range(B))
        ff_skipped = 0
        mask_fallbacks = 0
        verify_steps = 1
        total_acc = 0
        ff_jump_steps = 0
        ff_jump_tokens = 0
    else:
        matchers = state["matchers"]
        queues = state["queues"]
        gen_toks = state["gen_toks"]
        contexts = state["contexts"]
        tables: list[Any] = [None] * B
        idx_pos = [0] * B
        for b in state["active"]:
            t3, t2 = build_ngram_table(doc_prompts[b])
            tables[b] = (t3, t2)
            idx_pos[b] = _update_ngrams_from_tail(t3, t2, contexts[b], len(doc_prompts[b]), max_draft=L_step)
        done = state["done"]
        active = list(state["active"])
        ff_skipped = state["ff_skipped"]
        mask_fallbacks = state["mask_fallbacks"]
        verify_steps = state["verify_steps"]
        total_acc = state["total_acc"]
        ff_jump_steps = int(state.get("ff_jump_steps", 0))
        ff_jump_tokens = int(state.get("ff_jump_tokens", 0))

    def _pl_repeat_penalties(orig_b: int) -> dict[int, float]:
        if not eff_repeat_breaker:
            return {}
        min_hits = _REPEAT_4GRAM_MIN_HITS_CONSTRAINED if grammar is not None else _REPEAT_4GRAM_MIN_HITS
        return _repeated_4gram_penalties(
            gen_toks[orig_b], min_prior_hits=min_hits, stop_token_ids=stop_token_ids
        )

    def sample_and_queue(
        orig_b: int,
        row_logits: np.ndarray | None,
        argmax_j: int,
        x_row_mx: mx.array | None = None,
    ) -> bool:
        nonlocal ff_skipped, mask_fallbacks
        if grammar is None:
            t_orig = int(new_to_old[argmax_j])
            if row_logits is None and t_orig not in stop_token_ids:
                pen = _pl_repeat_penalties(orig_b)
                if pen and t_orig in pen:
                    return False
            if t_orig in stop_token_ids:
                done[orig_b] = True
                queues[orig_b].clear()
                return True
            gen_toks[orig_b].append(t_orig)
            contexts[orig_b].append(t_orig)
            if len(gen_toks[orig_b]) >= max_tokens:
                done[orig_b] = True
                queues[orig_b].clear()
                return True
            queues[orig_b].append(t_orig)
            return True

        assert matchers is not None
        m = matchers[orig_b]
        if row_logits is None:
            t_orig = int(new_to_old[argmax_j])
            pen = _pl_repeat_penalties(orig_b) if t_orig not in stop_token_ids else {}
            if (pen and t_orig in pen) or not m.validate_tokens([t_orig]):
                return False  # Signal caller to fetch lazy row_logits!
        else:
            t_orig, fell_back = _pick_valid_token_safe(
                grammar, model, m, row_logits, argmax_j, x_row_mx=x_row_mx
            )
            if fell_back:
                mask_fallbacks += 1
        m.consume_token(t_orig)
        gen_toks[orig_b].append(t_orig)
        contexts[orig_b].append(t_orig)
        if m.is_stopped() and m.is_accepting():
            done[orig_b] = True
            queues[orig_b].clear()
            return True
        queues[orig_b].append(t_orig)
        ff = m.compute_ff_tokens()
        if ff:
            m.consume_tokens(ff)
            gen_toks[orig_b].extend(ff)
            contexts[orig_b].extend(ff)
            ff_skipped += len(ff)
            if m.is_stopped() and m.is_accepting():
                done[orig_b] = True
                queues[orig_b].clear()
                return True
            queues[orig_b].extend(ff)
        if len(gen_toks[orig_b]) >= max_tokens:
            done[orig_b] = True
            queues[orig_b].clear()
        return True

    def _fallback_sample_step_row(
        orig_b: int,
        r_idx: int,
        pos_idx: int,
        lg_full_np: np.ndarray | None,
        lg_step_mx: Any,
        last_x: Any,
        am_rows: list[list[int]],
    ) -> np.ndarray:
        if lg_full_np is None:
            lg_full_np = np.array(lg_step_mx.astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
        row_lg = lg_full_np[r_idx, pos_idx]
        am_j = am_rows[r_idx][pos_idx]
        pen = _pl_repeat_penalties(orig_b)
        if pen:
            row_lg, am_j = _apply_repeat_penalties_row(row_lg, pen, old_to_new)
        xr = last_x[r_idx, pos_idx] if last_x is not None else None
        sample_and_queue(orig_b, row_lg, am_j, x_row_mx=xr)
        return lg_full_np

    if init_lg is not None:
        am_mx0 = safe_argmax_with_nan_guard(init_lg[:, 0, :], axis=-1)
        mx.eval(am_mx0)
        am_list0 = am_mx0.tolist()
        bad0 = [active[r] for r in range(len(active)) if int(am_list0[r]) < 0]
        if bad0:
            raise FloatingPointError(f"NaN logits detected in run_decode_prompt_lookup init_lg for rows {bad0}")
        last_x0 = cache.get("_last_x_in")
        if grammar is None:
            for r, b in enumerate(active):
                sample_and_queue(b, None, am_list0[r])
        else:
            assert matchers is not None
            need_mask0: list[int] = []
            for r, b in enumerate(active):
                if not sample_and_queue(b, None, am_list0[r]):
                    need_mask0.append(r)
            if need_mask0:
                lg_sub0 = np.array(init_lg[mx.array(need_mask0, dtype=mx.int32), 0, :].astype(mx.float32))
                for k_idx, r in enumerate(need_mask0):
                    xr0 = last_x0[r, 0] if last_x0 is not None else None
                    sample_and_queue(active[r], lg_sub0[k_idx], am_list0[r], x_row_mx=xr0)
        del init_lg

    off_cur = init_off

    while True:
        keep_rows = [r for r, b in enumerate(active) if not done[b]]
        if not keep_rows:
            break
        if len(keep_rows) < len(active) and (len(active) <= 6 or len(keep_rows) <= int(len(active) * 0.78)):
            cache = model.compact_cache(cache, keep_rows)
            off_cur = off_cur[mx.array(keep_rows, dtype=mx.int32)]
            active = [active[r] for r in keep_rows]

        B_cur = len(active)
        eff_step = L_step
        batch_seq = np.zeros((B_cur, eff_step), dtype=np.int32)
        n_ff_feed = [0] * B_cur
        drafts: list[list[int]] = [[] for _ in range(B_cur)]

        for r, b in enumerate(active):
            if done[b]:
                batch_seq[r] = fallback_pad
                continue
            t3, t2 = tables[b]
            if len(contexts[b]) - idx_pos[b] >= L_step + 4:
                idx_pos[b] = _update_ngrams_from_tail(t3, t2, contexts[b], idx_pos[b], max_draft=L_step)
            q = queues[b]
            fed: list[int] = []
            while q and len(fed) < eff_step:
                fed.append(q.popleft())
            n_ff_feed[r] = len(fed)
            ctx = contexts[b]
            if not q and len(fed) < eff_step:
                cand = None
                if len(ctx) >= 3:
                    cand = t3.get((ctx[-3], ctx[-2], ctx[-1]))
                if cand is None and len(ctx) >= 2:
                    cand = t2.get((ctx[-2], ctx[-1]))
                if cand is None:
                    pad_t = ctx[-1] if ctx else fallback_pad
                    cand = [pad_t] * (eff_step - 1)
                for d_tok in cand:
                    if len(fed) >= eff_step:
                        break
                    fed.append(d_tok)
                    drafts[r].append(d_tok)
            pad_t = fed[-1] if fed else fallback_pad
            while len(fed) < eff_step:
                fed.append(pad_t)
            batch_seq[r] = fed

        keep_from_col = 0
        if enable_ff_jump and eff_step >= 2:
            live_nfs = [n_ff_feed[r] for r, b in enumerate(active) if not done[b]]
            if live_nfs:
                min_nf = min(live_nfs)
                if min_nf >= 2:
                    keep_from_col = min(eff_step - 1, min_nf - 1)
                    ff_jump_steps += 1
                    ff_jump_tokens += (keep_from_col + 1) * len(live_nfs)

        batch_mx = mx.array(batch_seq, dtype=mx.int32)
        if keep_from_col > 0:
            lg_step = model.forward_step_ragged(
                batch_mx, cache, off_cur, apply_softcap=True, keep_from_col=keep_from_col
            )
        else:
            lg_step = model.forward_step_ragged(batch_mx, cache, off_cur, apply_softcap=True)
        am_mx = safe_argmax_with_nan_guard(lg_step, axis=-1)
        mx.eval(am_mx)  # guard: allow-eval (speculative draft verification step requires synchronous token IDs)
        am_list = am_mx.tolist()
        if np.any(np.asarray(am_list, dtype=np.int32) < 0):
            raise FloatingPointError("NaN logits detected in run_decode_prompt_lookup verification step")
        verify_steps += 1
        last_x_step = cache.get("_last_x_in")

        # Lazy float32 logits fetched ONLY if grammar validation fails on argmax!
        lg_full_np = None
        acc_lens = [0] * B_cur
        for r, b in enumerate(active):
            if done[b]:
                continue
            nf = n_ff_feed[r]
            dr = drafts[r]
            if queues[b]:
                acc_lens[r] = nf
                total_acc += nf
                continue

            matched = 0
            stopped = False
            for j, d_tok in enumerate(dr):
                pos = (nf - 1 - keep_from_col) + j
                pred_orig = int(new_to_old[am_list[r][pos]])
                pen = _pl_repeat_penalties(b) if pred_orig not in stop_token_ids else {}
                if pen and pred_orig in pen:
                    lg_full_np = _fallback_sample_step_row(b, r, pos, lg_full_np, lg_step, last_x_step, am_list)
                    acc_lens[r] = nf + matched
                    total_acc += nf + matched
                    stopped = True
                    break
                if grammar is not None:
                    assert matchers is not None
                    if not matchers[b].validate_tokens([pred_orig]):
                        lg_full_np = _fallback_sample_step_row(b, r, pos, lg_full_np, lg_step, last_x_step, am_list)
                        acc_lens[r] = nf + matched
                        total_acc += nf + matched
                        stopped = True
                        break
                else:
                    if pred_orig in stop_token_ids:
                        done[b] = True
                        queues[b].clear()
                        acc_lens[r] = nf + matched
                        total_acc += nf + matched
                        stopped = True
                        break

                if pred_orig == d_tok:
                    if grammar is not None:
                        assert matchers is not None
                        matchers[b].consume_token(pred_orig)
                    gen_toks[b].append(pred_orig)
                    contexts[b].append(pred_orig)
                    matched += 1
                    if len(gen_toks[b]) >= max_tokens:
                        done[b] = True
                        queues[b].clear()
                        acc_lens[r] = nf + matched
                        total_acc += nf + matched
                        stopped = True
                        break
                    if grammar is not None:
                        assert matchers is not None
                        if matchers[b].is_stopped() and matchers[b].is_accepting():
                            done[b] = True
                            queues[b].clear()
                            acc_lens[r] = nf + matched
                            total_acc += nf + matched
                            stopped = True
                            break
                        ff = matchers[b].compute_ff_tokens()
                        if ff:
                            matchers[b].consume_tokens(ff)
                            gen_toks[b].extend(ff)
                            contexts[b].extend(ff)
                            ff_skipped += len(ff)
                            if (matchers[b].is_stopped() and matchers[b].is_accepting()) or len(gen_toks[b]) >= max_tokens:
                                done[b] = True
                                queues[b].clear()
                            else:
                                queues[b].extend(ff)
                            acc_lens[r] = nf + matched
                            total_acc += nf + matched
                            stopped = True
                            break
                else:
                    if not sample_and_queue(b, None, am_list[r][pos]):
                        lg_full_np = _fallback_sample_step_row(b, r, pos, lg_full_np, lg_step, last_x_step, am_list)
                    acc_lens[r] = nf + matched
                    total_acc += nf + matched
                    stopped = True
                    break

            if not stopped:
                pos = (nf - 1 - keep_from_col) + len(dr)
                if pos >= 0:
                    ok = sample_and_queue(b, None, am_list[r][pos])
                    if not ok:
                        lg_full_np = _fallback_sample_step_row(b, r, pos, lg_full_np, lg_step, last_x_step, am_list)
                acc_lens[r] = nf + len(dr)
                total_acc += nf + len(dr)

            if len(gen_toks[b]) >= max_tokens:
                done[b] = True
                queues[b].clear()

        # Notice: NO separate mx.eval(off_cur) sync stall! Evaluates inside next forward_step_ragged!
        off_cur = off_cur + mx.array(acc_lens, dtype=mx.int32)
        del lg_step, am_mx

    dec_sec = (time.perf_counter() - t_dec0) if state is None else 0.0
    upfront_len = len(grammar.upfront_ff) if grammar is not None else 0
    tot_gen = sum(len(g) for g in gen_toks) + upfront_len * B
    return {
        "decode_sec": dec_sec,
        "gen_toks": gen_toks,
        "tot_out_tok": tot_gen,
        "verify_steps": verify_steps,
        "acc_tok_per_step": tot_gen / max(B * verify_steps, 1),
        "ff_tokens": ff_skipped + upfront_len * B,
        "ff_jump_steps": ff_jump_steps,
        "ff_jump_tokens": ff_jump_tokens,
        "mask_fallbacks": mask_fallbacks,
        "decode_tok_s": tot_gen / max(dec_sec, 1e-9),
    }


def run_decode_rolling(
    model: FastGemma4Model,
    stop_token_ids: frozenset[int],
    grammar: GrammarHelper | Any | None,
    prefix_kv: dict[str, Any],
    suf_ids_list: Sequence[Sequence[int]],
    *,
    max_tokens: int = 512,
    b_slots: int = 256,
    refill_min: int | None = None,
    swa_ring_cap: int = 512,
    group_size: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    sort_by_length: bool = True,
    layer15_prefix_split: bool = True,
    max_ff_drain: int = 8,
    tail_ff_max_b: int = 24,
    kv_i8: bool | None = None,
    doc_cap_full: int = 0,
    per_doc_grammars: Sequence[Any] | None = None,
    use_ane_ring: bool | str = False,
    ane_ring: Any | None = None,
    ane_cohort_size: int = 16,
    enable_ff_jump: bool = True,
    ane_ring_variant: str = "fp16",
    collect_telemetry: bool = False,
    collect_valid_lse: bool = False,
    repeat_4gram_breaker: bool | None = None,
) -> dict[str, Any]:
    """
    Continuous length-sorted slot-refill rolling GPU (+ optional Async Hybrid ANE+GPU) scheduler:
      - Wave-0 `50/50` simultaneous `GPU + ANE` bootstrap + background `AsyncHybridPrefillRing` slot refills (`use_ane_ring`)
      - zero-copy `IOSurface -> Metal KV_i8` ring scatter (`<1 ms` GPU) + `1`-token `LastTok` admission
      - single-launch causal slice jumping (`forward_ff_jump_slot`) across forced `llguidance` spans (`B <= 8`)
      - exits forced llguidance FF slots after Layer 14 KV scatter (`layer15_prefix_split`)
      - batch-drains pure FF runs (`max_ff_drain`) via 14-layer `forward_early14_slot`
      - avoids `[B, 1, V]` logit writes on greedy steps via `fused_softcap_argmax_bf16` (with NaN `-1` guard)
      - optional zero-copy GPU pre-mask telemetry (`lse_all`, `H_pre`, `top2_margin`, + `lse_valid` via `uint32` bitmask)
      - propagates `valid_schemas` directly from live matchers without re-parsing from token 0
    """
    eff_repeat_breaker = _REPEAT_4GRAM_BREAKER_DEFAULT if repeat_4gram_breaker is None else bool(repeat_4gram_breaker)
    N = len(suf_ids_list)
    want_telem = bool(collect_telemetry or collect_valid_lse)
    if per_doc_grammars is not None and len(per_doc_grammars) != N:
        raise ValueError(f"Mismatched per_doc_grammars={len(per_doc_grammars)} vs N={N}")

    def _doc_grammar(d: int) -> Any:
        return per_doc_grammars[d] if per_doc_grammars is not None else grammar

    def _doc_upfront_len(d: int) -> int:
        g = _doc_grammar(d)
        if g is None:
            return 0
        up_fn = getattr(g, "upfront_ff_for_doc", None)
        if callable(up_fn):
            return len(up_fn(d))
        return len(getattr(g, "upfront_ff", ()))

    def _new_doc_matcher(d: int, g: Any) -> Any:
        if g is None:
            return None
        by_doc = getattr(g, "new_matcher_for_doc", None)
        if callable(by_doc):
            return by_doc(d)
        return g.new_matcher()

    if N == 0:
        empty_res: dict[str, Any] = {
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "gen_toks": [],
            "valid_schemas": [],
            "tot_out_tok": 0,
            "llm_steps": 0,
            "ff_early14_steps": 0,
            "ff_tokens": 0,
            "ff_jump_steps": 0,
            "ff_jump_tokens": 0,
            "mask_fallbacks": 0,
            "decode_tok_s": 0.0,
            "used_speculative": False,
            "used_ane_ring": False,
            "ane_cohorts": 0,
            "ane_busy_sec": 0.0,
            "ane_nan_fallbacks": 0,
            "gpu_boot_cohorts": 0,
        }
        if want_telem:
            empty_res["telemetry"] = []
        return empty_res

    suf_list = [list(s) for s in suf_ids_list]
    if max_tokens <= 0:
        tot_up = sum(_doc_upfront_len(d) for d in range(N))
        valid_zero: list[bool] = []
        for d in range(N):
            g_d = _doc_grammar(d)
            if g_d is None:
                valid_zero.append(True)
            else:
                m0 = _new_doc_matcher(d, g_d)
                valid_zero.append(bool(m0.is_stopped() and m0.is_accepting()))
        zero_res: dict[str, Any] = {
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "gen_toks": [[] for _ in range(N)],
            "valid_schemas": valid_zero,
            "tot_out_tok": tot_up,
            "llm_steps": 0,
            "ff_early14_steps": 0,
            "ff_tokens": tot_up,
            "ff_jump_steps": 0,
            "ff_jump_tokens": 0,
            "mask_fallbacks": 0,
            "decode_tok_s": 0.0,
            "used_speculative": False,
            "used_ane_ring": False,
            "ane_cohorts": 0,
            "ane_busy_sec": 0.0,
            "ane_nan_fallbacks": 0,
            "gpu_boot_cohorts": 0,
        }
        if want_telem:
            zero_res["telemetry"] = [[] for _ in range(N)]
        return zero_res

    B_slots = max(1, min(int(b_slots), N))
    eff_refill_min = min(32, max(1, B_slots // 4)) if refill_min is None else max(1, min(B_slots, int(refill_min)))
    eff_group_size = min(group_size, 8 if B_slots <= 32 else 16)
    max_suf = max(len(s) for s in suf_list)
    eff_doc_cap_full = max(int(doc_cap_full), max_suf + max_tokens + 64, 128)
    win_sz = int(getattr(model, "window_size", 512) or 512)
    min_swa = 1 << (max(512, win_sz) - 1).bit_length()
    eff_swa_ring_cap = max(int(swa_ring_cap), min_swa)
    slot_cache_kwargs: dict[str, Any] = {
        "batch_size": B_slots,
        "doc_cap_full": eff_doc_cap_full,
        "swa_ring_cap": eff_swa_ring_cap,
        "kv_i8": kv_i8,
    }
    if int(max_ff_drain) > 16:
        slot_cache_kwargs["drain_slack"] = int(max_ff_drain)
    cache = model.make_slot_cache(prefix_kv, **slot_cache_kwargs)

    active_ane_ring = _resolve_or_build_ane_ring(
        model,
        prefix_kv,
        cache,
        suf_list,
        use_ane_ring=use_ane_ring,
        ane_ring=ane_ring,
        ane_ring_variant=ane_ring_variant,
        ane_cohort_size=ane_cohort_size,
    )

    doc_order = (
        sorted(range(N), key=lambda d: -len(suf_list[d]))
        if (sort_by_length and N > B_slots)
        else list(range(N))
    )

    new_to_old = model.new_to_old
    gen_toks: list[list[int]] = [[] for _ in range(N)]
    valid_schemas: list[bool] = [_doc_grammar(d) is None for d in range(N)]

    slot_doc = [-1] * B_slots
    slot_grammar: list[Any] = [None] * B_slots
    slot_matcher: list[Any] = [None] * B_slots
    slot_queue: list[collections.deque[int]] = [collections.deque() for _ in range(B_slots)]
    slot_off = [0] * B_slots
    free_slots = list(range(B_slots))

    in_toks_np = np.empty((B_slots, 1), dtype=np.int32)
    offs_np = np.empty(B_slots, dtype=np.int32)
    slots_np = np.empty(B_slots, dtype=np.int32)
    ff_pad_cap = max(1, max_ff_drain, 32 if enable_ff_jump else 1)
    ff_pad_np = np.zeros((B_slots, ff_pad_cap), dtype=np.int32)

    return_logits = temperature > 0.0
    rng = np.random.default_rng(seed)
    can_ff_jump = bool(enable_ff_jump and hasattr(model, "forward_ff_jump_slot"))

    ff_skipped = 0
    mask_fallbacks = 0
    llm_steps = 0
    ff_early14_steps = 0
    ff_jump_steps = 0
    ff_jump_tokens = 0
    prefill_sec = 0.0
    decode_sec = 0.0
    ane_cohorts = 0
    ane_busy_sec = 0.0
    ane_nan_fallbacks = 0
    gpu_boot_cohorts = 0

    stop_event = threading.Event()
    ready_q: queue.Queue[tuple[str, Any]] = queue.Queue()
    ane_thread: threading.Thread | None = None
    pending_ane_item: dict[str, Any] | None = None
    ane_done_producing = active_ane_ring is None

    def _free_slot(slot_s: int) -> None:
        slot_doc[slot_s] = -1
        slot_grammar[slot_s] = None
        slot_matcher[slot_s] = None
        slot_queue[slot_s].clear()
        free_slots.append(slot_s)

    def _process_sampled_token(slot_s: int, orig_id: int) -> None:
        nonlocal ff_skipped
        d = slot_doc[slot_s]
        g_slot = slot_grammar[slot_s]
        if g_slot is None:
            if orig_id in stop_token_ids:
                valid_schemas[d] = True
                _free_slot(slot_s)
                return
            gen_toks[d].append(orig_id)
            if len(gen_toks[d]) >= max_tokens:
                valid_schemas[d] = True
                _free_slot(slot_s)
                return
            slot_queue[slot_s].append(orig_id)
            return

        m = slot_matcher[slot_s]
        gen_toks[d].append(orig_id)
        if m.is_stopped() and m.is_accepting():
            valid_schemas[d] = True
            _free_slot(slot_s)
            return
        slot_queue[slot_s].append(orig_id)
        ff = m.compute_ff_tokens()
        if ff:
            m.consume_tokens(ff)
            gen_toks[d].extend(ff)
            ff_skipped += len(ff)
            if m.is_stopped() and m.is_accepting():
                valid_schemas[d] = True
                _free_slot(slot_s)
                return
            slot_queue[slot_s].extend(ff)
        if len(gen_toks[d]) >= max_tokens:
            valid_schemas[d] = bool(m.is_stopped() and m.is_accepting())
            _free_slot(slot_s)

    doc_telemetry: list[list[dict[str, float]]] = [[] for _ in range(N)] if want_telem else []
    model_vocab_size = int(getattr(model, "vocab_size", len(new_to_old)))
    old_to_new = getattr(model, "old_to_new", None)
    head_mask_words = (model_vocab_size + 31) // 32
    new_to_old_u32 = np.asarray(new_to_old, dtype=np.uint32) if want_telem else None

    def _build_valid_mask_for_slots(slots_seq: Sequence[int], n_rows: int) -> mx.array | None:
        if not collect_valid_lse or n_rows <= 0:
            return None
        mask_mat = np.full((n_rows, head_mask_words), np.uint32(0xFFFFFFFF), dtype=np.uint32)
        for ri in range(n_rows):
            s_i = int(slots_seq[ri]) if ri < len(slots_seq) else -1
            if s_i < 0 or slot_grammar[s_i] is None or slot_matcher[s_i] is None:
                continue
            g_i = slot_grammar[s_i]
            m_i = slot_matcher[s_i]
            full_v = int(getattr(getattr(g_i, "ltok", None), "vocab_size", model_vocab_size))
            full_words = (full_v + 31) // 32
            raw_u32 = np.zeros(full_words, dtype=np.uint32)
            m_i.unsafe_compute_mask_ptr(raw_u32.ctypes.data, raw_u32.nbytes)
            if full_v == model_vocab_size and new_to_old_u32 is not None and len(new_to_old_u32) == full_v and int(new_to_old_u32[-1]) == full_v - 1:
                mask_mat[ri, : min(head_mask_words, full_words)] = raw_u32[: min(head_mask_words, full_words)]
            elif new_to_old_u32 is not None:
                bits_1d = (raw_u32[new_to_old_u32 >> np.uint32(5)] >> (new_to_old_u32 & np.uint32(31))) & np.uint32(1)
                row_u32 = np.zeros(head_mask_words, dtype=np.uint32)
                nz = np.flatnonzero(bits_1d != 0).astype(np.uint32)
                if nz.size > 0:
                    np.bitwise_or.at(row_u32, nz >> np.uint32(5), np.uint32(1) << (nz & np.uint32(31)))
                mask_mat[ri] = row_u32
        return mx.array(mask_mat, dtype=mx.uint32)

    def _record_step_telemetry(target_slots: Sequence[int], telem_mx: mx.array | None) -> None:
        if telem_mx is None or not want_telem:
            return
        t_np = np.asarray(telem_mx, dtype=np.float32)
        has_v = t_np.ndim == 2 and t_np.shape[1] >= 4
        for r, s in enumerate(target_slots):
            d = slot_doc[s]
            if d < 0 or r >= t_np.shape[0]:
                continue
            entry: dict[str, float] = {
                "lse_all": float(t_np[r, 0]),
                "h_pre": float(t_np[r, 1]),
                "top2_margin": float(t_np[r, 2]),
            }
            if has_v:
                entry["lse_valid"] = float(t_np[r, 3])
                entry["log_valid_mass"] = float(t_np[r, 3] - t_np[r, 0])
            doc_telemetry[d].append(entry)

    def _handle_step_outputs(target_slots: Sequence[int], x_or_lg: mx.array, am_mx: mx.array) -> None:
        nonlocal mask_fallbacks
        telem_mx = cache.pop("_last_telemetry", None) if isinstance(cache, dict) else None
        if telem_mx is not None:
            mx.eval(x_or_lg, am_mx, telem_mx)
            _record_step_telemetry(target_slots, telem_mx)
        else:
            mx.eval(x_or_lg, am_mx)
        am_np = np.asarray(am_mx, dtype=np.int32)
        if np.any(am_np < 0):
            bad_slots = [int(target_slots[i]) for i in range(len(target_slots)) if int(am_np[i]) < 0]
            raise FloatingPointError(f"NaN logits detected in fused_softcap_argmax_bf16 for slots {bad_slots}")

        def _slot_repeat_penalties(slot_s: int, *, constrained: bool) -> dict[int, float]:
            if not eff_repeat_breaker:
                return {}
            min_hits = _REPEAT_4GRAM_MIN_HITS_CONSTRAINED if constrained else _REPEAT_4GRAM_MIN_HITS
            return _repeated_4gram_penalties(
                gen_toks[slot_doc[slot_s]],
                min_prior_hits=min_hits,
                stop_token_ids=stop_token_ids,
            )

        if return_logits:
            lg_np = np.asarray(x_or_lg[:, 0, :].astype(mx.float32))
            last_x = cache.get("_last_x_in")
            for r, s in enumerate(target_slots):
                g_slot = slot_grammar[s]
                row_lg = lg_np[r]
                am_j = int(am_np[r])
                pen = _slot_repeat_penalties(s, constrained=g_slot is not None)
                if pen:
                    row_lg, am_j = _apply_repeat_penalties_row(row_lg, pen, old_to_new)
                if g_slot is None:
                    oid = sample_unconstrained(
                        row_lg, new_to_old, am_j, temperature=temperature, top_p=top_p, rng=rng
                    )
                else:
                    m = slot_matcher[s]
                    xr = last_x[r, 0] if last_x is not None else None
                    oid, fell = _pick_valid_token_safe(
                        g_slot,
                        model,
                        m,
                        row_lg,
                        am_j,
                        x_row_mx=xr,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                    )
                    if fell:
                        mask_fallbacks += 1
                    m.consume_token(oid)
                _process_sampled_token(s, oid)
            return

        orig_ids = new_to_old[am_np]
        need_fb: list[tuple[int, int, Any, int, dict[int, float]]] = []
        for r, s in enumerate(target_slots):
            oid = int(orig_ids[r])
            g_slot = slot_grammar[s]
            pen = (
                _slot_repeat_penalties(s, constrained=g_slot is not None)
                if oid not in stop_token_ids
                else {}
            )
            if g_slot is None:
                if pen and oid in pen:
                    need_fb.append((r, s, None, int(am_np[r]), pen))
                else:
                    _process_sampled_token(s, oid)
                continue
            m = slot_matcher[s]
            if (not pen or oid not in pen) and m.try_consume_tokens([oid]) == 1:
                _process_sampled_token(s, oid)
            else:
                need_fb.append((r, s, g_slot, int(am_np[r]), pen))
        if need_fb:
            fb_rows = mx.array([x[0] for x in need_fb], dtype=mx.int32)
            x_fb = x_or_lg[fb_rows]
            lg_fb = model._apply_lm_head(x_fb, apply_softcap=True)
            mx.eval(lg_fb)
            lg_sub = np.asarray(lg_fb[:, 0, :].astype(mx.float32))
            for k_i, (_, s, g_slot, am_j, pen) in enumerate(need_fb):
                row_lg = lg_sub[k_i]
                if pen:
                    row_lg, am_j = _apply_repeat_penalties_row(row_lg, pen, old_to_new)
                if g_slot is None:
                    _process_sampled_token(s, int(new_to_old[am_j]))
                else:
                    m = slot_matcher[s]
                    oid, fell = _pick_valid_token_safe(
                        g_slot, model, m, row_lg, am_j, x_row_mx=x_fb[k_i, 0]
                    )
                    if fell:
                        mask_fallbacks += 1
                    m.consume_token(oid)
                    _process_sampled_token(s, oid)

    def _admit_batch(doc_indices: list[int], target_slots: list[int]) -> None:
        nonlocal prefill_sec, decode_sec, llm_steps
        tp0 = time.perf_counter()
        sub_sufs = [suf_list[d] for d in doc_indices]
        if want_telem:
            for r, s in enumerate(target_slots):
                d = doc_indices[r]
                g_slot = _doc_grammar(d)
                slot_doc[s] = d
                slot_grammar[s] = g_slot
                slot_matcher[s] = _new_doc_matcher(d, g_slot)
                slot_queue[s].clear()
            vmask_0 = _build_valid_mask_for_slots(target_slots, len(target_slots))
            x_or_lg, am, next_offs = model.prefill_slots_lazy_lg(
                sub_sufs,
                target_slots,
                cache,
                group_size=eff_group_size,
                return_logits=return_logits,
                async_eval=True,
                collect_telemetry=True,
                valid_mask_u32=vmask_0,
            )
            prefill_sec += time.perf_counter() - tp0
            llm_steps += 1
            td0 = time.perf_counter()
            for r, s in enumerate(target_slots):
                slot_off[s] = next_offs[r]
            _handle_step_outputs(target_slots, x_or_lg, am)
            decode_sec += time.perf_counter() - td0
        else:
            x_or_lg, am, next_offs = model.prefill_slots_lazy_lg(
                sub_sufs,
                target_slots,
                cache,
                group_size=eff_group_size,
                return_logits=return_logits,
                async_eval=True,
            )
            prefill_sec += time.perf_counter() - tp0
            llm_steps += 1

            # Overlap CPU slot/matcher creation behind async LastTok execution
            td0 = time.perf_counter()
            for r, s in enumerate(target_slots):
                d = doc_indices[r]
                g_slot = _doc_grammar(d)
                slot_doc[s] = d
                slot_off[s] = next_offs[r]
                slot_grammar[s] = g_slot
                slot_matcher[s] = _new_doc_matcher(d, g_slot)
                slot_queue[s].clear()
            _handle_step_outputs(target_slots, x_or_lg, am)
            decode_sec += time.perf_counter() - td0
        if len(doc_indices) >= 8 and callable(getattr(mx, "get_cache_memory", None)) and callable(getattr(mx, "clear_cache", None)):
            if mx.get_cache_memory() > (256 << 20):
                mx.clear_cache()

    def _admit_ane_cohort(item: dict[str, Any], target_slots: list[int]) -> None:
        nonlocal prefill_sec, decode_sec, llm_steps, ane_cohorts, ane_busy_sec, ane_nan_fallbacks
        tp0 = time.perf_counter()
        doc_indices = item["doc_ids"]
        sub_sufs = item["sufs"]
        covered = list(item["covered"])
        K = len(doc_indices)
        assert active_ane_ring is not None
        model._eval_doc_layers(cache["layers"], bool(cache.get("kv_i8", True)))
        mx.synchronize()
        with SlotCacheMetalBinding(active_ane_ring.runner, model, cache) as live_binding:
            active_ane_ring.scatter_cohort_to_slot_cache(live_binding, item, target_slots)
        ane_cohorts += 1
        ane_busy_sec += float(item.get("ane_sec", 0.0))

        # Validate scattered KV_i8 scales (or BF16 KV) for Inf/NaN; if poisoned, recompute cohort cleanly on GPU!
        if not validate_slot_kv_scales_finite(cache, target_slots):
            warnings.warn(
                "Non-finite (Inf/NaN) ANE KV or KV_i8 scale detected after ring scatter; "
                "falling back to GPU BF16 prefill for this cohort.",
                RuntimeWarning,
                stacklevel=2,
            )
            ane_nan_fallbacks += 1
            covered = [0] * K

        l_sys_i = int(cache["l_sys"])
        # Finish on the GPU whatever suffix the ANE chunks did not cover.
        tail_by_cov: dict[int, list[int]] = collections.defaultdict(list)
        for ri in range(K):
            if (len(sub_sufs[ri]) - 1) > covered[ri]:
                tail_by_cov[int(covered[ri])].append(ri)
        # Length-grouped like the GPU admit path: one padded batch over a mixed-length cohort writes pad KVs past
        # the shorter docs' ends, which wrap in the SWA ring and overwrite their last 512 real positions.
        for cov_start, ris in tail_by_cov.items():
            model._prefill_early14_slot_groups(
                [sub_sufs[ri] for ri in ris],
                [target_slots[ri] for ri in ris],
                cache,
                group_size=eff_group_size,
                start_s0=cov_start,
            )

        for r, s in enumerate(target_slots):
            d = doc_indices[r]
            g_slot = _doc_grammar(d)
            slot_doc[s] = d
            slot_off[s] = l_sys_i + len(sub_sufs[r])
            slot_grammar[s] = g_slot
            slot_matcher[s] = _new_doc_matcher(d, g_slot)
            slot_queue[s].clear()

        last_toks = mx.array([sub_sufs[ri][-1] for ri in range(K)], dtype=mx.int32)[:, None]
        last_off = mx.array([l_sys_i + len(sub_sufs[ri]) - 1 for ri in range(K)], dtype=mx.int32)
        slot_mx = mx.array(target_slots, dtype=mx.int32)
        if want_telem:
            adm_mask = _build_valid_mask_for_slots(target_slots, K)
            x_or_lg, am = model.forward_decode_step_slot(
                last_toks,
                cache,
                last_off,
                slot_mx,
                n_keep=K,
                return_logits=return_logits,
                collect_telemetry=collect_telemetry,
                valid_mask_u32=adm_mask,
            )
            telem_adm = cache.get("_last_telemetry")
            if telem_adm is not None:
                mx.async_eval(x_or_lg, am, telem_adm)
            else:
                mx.async_eval(x_or_lg, am)
        else:
            x_or_lg, am = model.forward_decode_step_slot(
                last_toks, cache, last_off, slot_mx, n_keep=K, return_logits=return_logits
            )
            mx.async_eval(x_or_lg, am)
        prefill_sec += time.perf_counter() - tp0
        llm_steps += 1

        td0 = time.perf_counter()
        _handle_step_outputs(target_slots, x_or_lg, am)
        decode_sec += time.perf_counter() - td0

    def _pull_ready_ane(*, block: bool) -> bool:
        """Pull at most one ready ANE item into `pending_ane_item`; return True if an item was admitted."""
        nonlocal pending_ane_item, ane_done_producing
        admitted_any = False
        while True:
            if pending_ane_item is None and not ane_done_producing:
                try:
                    kind, payload = ready_q.get(block=block)
                except queue.Empty:
                    break
                if kind == "err":
                    ane_done_producing = True
                    raise payload
                if kind == "eof":
                    ane_done_producing = True
                    break
                pending_ane_item = payload
                block = False

            if pending_ane_item is None:
                break
            k_need = len(pending_ane_item["doc_ids"])
            if len(free_slots) < k_need:
                break
            r_slots = free_slots[:k_need]
            del free_slots[:k_need]
            item = pending_ane_item
            pending_ane_item = None
            _admit_ane_cohort(item, r_slots)
            admitted_any = True
            block = False
        return admitted_any

    def _drain_ff_slots(target_ff_slots: list[int]) -> None:
        nonlocal ff_early14_steps
        n_ff = len(target_ff_slots)
        eff_drain = (
            max(max_ff_drain, 15 // max(1, n_ff))
            if (can_ff_jump and n_ff <= 4 and max_ff_drain > 0)
            else max_ff_drain
        )
        eff_drain = min(eff_drain, int(cache.get("drain_slack", 16)), ff_pad_cap)
        max_k = 1
        k_lens = [0] * n_ff
        max_priv_start = 0
        l_sys = int(cache["l_sys"])
        cur_swa_cap = int(cache["swa_cap"])
        for r, s in enumerate(target_ff_slots):
            q = slot_queue[s]
            k_s = min(eff_drain, len(q) - 1)
            k_lens[r] = k_s
            if k_s > max_k:
                max_k = k_s
            off_s = slot_off[s]
            offs_np[r] = off_s
            slots_np[r] = s
            slot_off[s] = off_s + k_s
            priv_start = off_s - l_sys
            if priv_start > max_priv_start:
                max_priv_start = priv_start
            for c_j in range(k_s):
                ff_pad_np[r, c_j] = q.popleft()
            if k_s < eff_drain:
                ff_pad_np[r, k_s:eff_drain] = 0

        max_priv_pad_end = max_priv_start + max_k - 1
        if max_k > 1 and cur_swa_cap < model.window_size + max_k - 1 and max_priv_pad_end >= cur_swa_cap:
            # SWA ring wrap safety: guard using max_priv_pad_end because L_q=max_k scatters padded columns.
            step_w = max(1, cur_swa_cap - model.window_size + 1)
            for c0 in range(0, max_k, step_w):
                c1 = min(max_k, c0 + step_w)
                act_rows = [r for r in range(n_ff) if k_lens[r] > c0]
                if not act_rows:
                    break
                toks_sub = mx.array(ff_pad_np[act_rows, c0:c1])
                off_sub = mx.array(offs_np[act_rows] + c0)
                slot_sub = mx.array(slots_np[act_rows])
                model.forward_early14_slot(toks_sub, cache, off_sub, slot_sub)
        else:
            toks_ff = mx.array(ff_pad_np[:n_ff, :max_k])
            off_ff = mx.array(offs_np[:n_ff])
            slot_ff = mx.array(slots_np[:n_ff])
            model.forward_early14_slot(toks_ff, cache, off_ff, slot_ff)
        ff_early14_steps += 1

    try:
        init_k = min(B_slots, N)
        if active_ane_ring is not None:
            # A cohort holds one ring slot per doc per chunk until it is scattered.
            max_ring_cohort = max(1, active_ane_ring.runner.out_ring_size // max(1, active_ane_ring.n_chunks))
            eff_ane_cohort = max(1, min(int(ane_cohort_size), B_slots, max_ring_cohort))
            gpu_boot_k = (init_k // 2) if init_k >= 2 else 0
            ane_start = gpu_boot_k

            def _ane_producer() -> None:
                assert active_ane_ring is not None
                try:
                    for p0 in range(ane_start, N, eff_ane_cohort):
                        if stop_event.is_set():
                            return
                        c_docs = doc_order[p0 : min(N, p0 + eff_ane_cohort)]
                        c_sufs = [suf_list[d] for d in c_docs]
                        item = active_ane_ring.prefill_cohort_into_ring(
                            c_docs, c_sufs, pipe_depth=4, stop_event=stop_event
                        )
                        ready_q.put(("cohort", item))
                    ready_q.put(("eof", None))
                except BaseException as exc:
                    stop_event.set()
                    ready_q.put(("err", exc))

            ane_thread = threading.Thread(target=_ane_producer, name="fg4-ane-ring-producer", daemon=True)
            ane_thread.start()

            # Wave-0 50/50 simultaneous GPU + ANE bootstrap
            if gpu_boot_k > 0:
                boot_slots = free_slots[:gpu_boot_k]
                del free_slots[:gpu_boot_k]
                _admit_batch(doc_order[:gpu_boot_k], boot_slots)
                gpu_boot_cohorts = 1

            # Admit the ANE half of Wave-0 so decode starts at full B_slots occupancy
            ane_wave0_target = init_k - gpu_boot_k
            ane_wave0_done = 0
            while ane_wave0_done < ane_wave0_target and not ane_done_producing:
                prev_cohorts = ane_cohorts
                if not _pull_ready_ane(block=True):
                    break
                if ane_cohorts > prev_cohorts:
                    ane_wave0_done = init_k - len(free_slots) - gpu_boot_k
            next_doc = N  # All remaining Wave-1+ documents are owned by `_ane_producer`!
        else:
            init_slots = free_slots[:init_k]
            del free_slots[:init_k]
            _admit_batch(doc_order[:init_k], init_slots)
            next_doc = init_k

        while True:
            if active_ane_ring is not None:
                if pending_ane_item is not None or not ane_done_producing:
                    _pull_ready_ane(block=False)
            elif next_doc < N and len(free_slots) >= eff_refill_min:
                k = min(len(free_slots), N - next_doc)
                r_slots = free_slots[:k]
                del free_slots[:k]
                _admit_batch(doc_order[next_doc : next_doc + k], r_slots)
                next_doc += k

            samp_slots: list[int] = []
            ff_slots: list[int] = []
            for s in range(B_slots):
                if slot_doc[s] >= 0:
                    if len(slot_queue[s]) == 1:
                        samp_slots.append(s)
                    else:
                        ff_slots.append(s)

            if not samp_slots and not ff_slots:
                if active_ane_ring is not None and (pending_ane_item is not None or not ane_done_producing):
                    _pull_ready_ane(block=True)
                    for s in range(B_slots):
                        if slot_doc[s] >= 0:
                            if len(slot_queue[s]) == 1:
                                samp_slots.append(s)
                            else:
                                ff_slots.append(s)
                    if not samp_slots and not ff_slots and pending_ane_item is None and ane_done_producing:
                        break
                    if not samp_slots and not ff_slots:
                        continue
                elif next_doc < N:
                    k = min(len(free_slots), N - next_doc)
                    r_slots = free_slots[:k]
                    del free_slots[:k]
                    _admit_batch(doc_order[next_doc : next_doc + k], r_slots)
                    next_doc += k
                    for s in range(B_slots):
                        if slot_doc[s] >= 0:
                            if len(slot_queue[s]) == 1:
                                samp_slots.append(s)
                            else:
                                ff_slots.append(s)
                    if not samp_slots and not ff_slots:
                        break
                else:
                    break

            td0 = time.perf_counter()

            # Case 1: Only FF tokens pending -> single-launch causal slice jump (B<=8) or 14-layer EarlyExit14 drain
            if not samp_slots:
                if can_ff_jump and 1 <= len(ff_slots) <= 7:
                    q_lens = [len(slot_queue[s]) for s in ff_slots]
                    q_min, q_max = min(q_lens), max(q_lens)
                    max_k_jump = 15 // len(ff_slots)
                    if q_min >= 2 and q_min == q_max and q_min <= max_k_jump:
                        l_sys_j = int(cache["l_sys"])
                        cur_swa_cap = int(cache["swa_cap"])
                        max_priv_j = max(slot_off[s] - l_sys_j for s in ff_slots)
                        if cur_swa_cap >= model.window_size + q_min - 1 or (max_priv_j + q_min - 1) < cur_swa_cap:
                            n_jf = len(ff_slots)
                            jump_np = np.empty((n_jf, q_min), dtype=np.int32)
                            for r_j, s_j in enumerate(ff_slots):
                                q_j = slot_queue[s_j]
                                offs_np[r_j] = slot_off[s_j]
                                slots_np[r_j] = s_j
                                slot_off[s_j] += q_min
                                for c_j in range(q_min):
                                    jump_np[r_j, c_j] = q_j.popleft()
                            if want_telem:
                                jmp_mask = _build_valid_mask_for_slots(ff_slots, n_jf)
                                x_or_lg, am_mx = model.forward_ff_jump_slot(
                                    mx.array(jump_np),
                                    cache,
                                    mx.array(offs_np[:n_jf]),
                                    mx.array(slots_np[:n_jf]),
                                    return_logits=return_logits,
                                    collect_telemetry=collect_telemetry,
                                    valid_mask_u32=jmp_mask,
                                )
                            else:
                                x_or_lg, am_mx = model.forward_ff_jump_slot(
                                    mx.array(jump_np),
                                    cache,
                                    mx.array(offs_np[:n_jf]),
                                    mx.array(slots_np[:n_jf]),
                                    return_logits=return_logits,
                                )
                            llm_steps += 1
                            ff_jump_steps += 1
                            ff_jump_tokens += q_min * n_jf
                            _handle_step_outputs(ff_slots, x_or_lg, am_mx)
                            decode_sec += time.perf_counter() - td0
                            continue

                if max_ff_drain > 0:
                    _drain_ff_slots(ff_slots)
                    decode_sec += time.perf_counter() - td0
                    continue
                n_ff = len(ff_slots)
                for r, s in enumerate(ff_slots):
                    in_toks_np[r, 0] = slot_queue[s].popleft()
                    offs_np[r] = slot_off[s]
                    slots_np[r] = s
                    slot_off[s] += 1
                model.forward_early14_slot(
                    mx.array(in_toks_np[:n_ff]),
                    cache,
                    mx.array(offs_np[:n_ff]),
                    mx.array(slots_np[:n_ff]),
                )
                ff_early14_steps += 1
                decode_sec += time.perf_counter() - td0
                continue

            # Case 2: Tail FF drain when no pending docs remain and active batch is small
            no_pending_docs = (
                (pending_ane_item is None and ane_done_producing)
                if active_ane_ring is not None
                else (next_doc >= N)
            )
            if (
                max_ff_drain > 1
                and ff_slots
                and no_pending_docs
                and (len(samp_slots) + len(ff_slots)) <= tail_ff_max_b
                and any(len(slot_queue[s]) >= 3 for s in ff_slots)
            ):
                _drain_ff_slots(ff_slots)
                new_ff = []
                for s in ff_slots:
                    if len(slot_queue[s]) == 1:
                        samp_slots.append(s)
                    else:
                        new_ff.append(s)
                ff_slots = new_ff

            B_samp = len(samp_slots)
            B_act = B_samp + len(ff_slots)
            r = 0
            for s in samp_slots:
                in_toks_np[r, 0] = slot_queue[s].popleft()
                offs_np[r] = slot_off[s]
                slots_np[r] = s
                slot_off[s] += 1
                r += 1
            for s in ff_slots:
                in_toks_np[r, 0] = slot_queue[s].popleft()
                offs_np[r] = slot_off[s]
                slots_np[r] = s
                slot_off[s] += 1
                r += 1

            toks_mx = mx.array(in_toks_np[:B_act])
            off_mx = mx.array(offs_np[:B_act])
            slot_mx = mx.array(slots_np[:B_act])
            n_keep = B_samp if layer15_prefix_split else B_act

            if want_telem:
                vmask_step = _build_valid_mask_for_slots(
                    samp_slots if n_keep == B_samp else (samp_slots + ff_slots), n_keep
                )
                x_or_lg, am_mx = model.forward_decode_step_slot(
                    toks_mx,
                    cache,
                    off_mx,
                    slot_mx,
                    n_keep=n_keep,
                    return_logits=return_logits,
                    collect_telemetry=True,
                    valid_mask_u32=vmask_step,
                )
            else:
                x_or_lg, am_mx = model.forward_decode_step_slot(
                    toks_mx, cache, off_mx, slot_mx, n_keep=n_keep, return_logits=return_logits
                )
            llm_steps += 1
            _handle_step_outputs(samp_slots, x_or_lg, am_mx)
            decode_sec += time.perf_counter() - td0

        tot_up = sum(_doc_upfront_len(d) for d in range(N))
        tot_gen = sum(len(g) for g in gen_toks) + tot_up
        roll_out: dict[str, Any] = {
            "prefill_sec": prefill_sec,
            "decode_sec": decode_sec,
            "gen_toks": gen_toks,
            "valid_schemas": valid_schemas,
            "tot_out_tok": tot_gen,
            "llm_steps": llm_steps,
            "ff_early14_steps": ff_early14_steps,
            "ff_tokens": ff_skipped + tot_up,
            "ff_jump_steps": ff_jump_steps,
            "ff_jump_tokens": ff_jump_tokens,
            "mask_fallbacks": mask_fallbacks,
            "decode_tok_s": tot_gen / max(decode_sec, 1e-9),
            "used_speculative": False,
            "used_ane_ring": active_ane_ring is not None,
            "ane_cohorts": ane_cohorts,
            "ane_busy_sec": ane_busy_sec,
            "ane_nan_fallbacks": ane_nan_fallbacks,
            "gpu_boot_cohorts": gpu_boot_cohorts,
        }
        if want_telem:
            roll_out["telemetry"] = doc_telemetry
        return roll_out
    finally:
        stop_event.set()
        if ane_thread is not None:
            ane_thread.join(timeout=5.0)
        if active_ane_ring is not None:
            if pending_ane_item is not None and pending_ane_item.get("ring_slots_held"):
                active_ane_ring.release_ring_slots(pending_ane_item["ring_slots_held"])
            while True:
                try:
                    kind, payload = ready_q.get_nowait()
                except queue.Empty:
                    break
                if kind == "cohort" and isinstance(payload, dict) and payload.get("ring_slots_held"):
                    active_ane_ring.release_ring_slots(payload["ring_slots_held"])
        del cache
