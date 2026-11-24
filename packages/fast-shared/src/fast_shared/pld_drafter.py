"""Shared cursor-tracked n-gram PromptLookupDrafter and grammar-safe draft prefix filtering."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import AbstractSet, Any

import numpy as np

_CORPUS_SENTINEL = -1


class PromptLookupDrafter:
    """Zero-parameter speculative drafter combining:
      1. Cursor-tracked prompt/context corpus n-gram lookup (disambiguates repeated n-grams monotonically).
      2. Incremental O(1) online Self-PLD over already generated tokens (captures repeating structure
         such as JSON keys, HTML/Markdown rows, and repeated phrases without O(N) Python scans).
    """

    def __init__(
        self,
        vision_token_ids: Sequence[int] = (),
        *,
        extra_token_ids: Sequence[Sequence[int]] = (),
        max_ngram: int = 4,
        min_ngram: int = 2,
        max_k: int = 6,
        max_hits_per_ngram: int = 32,
        use_cursor: bool = True,
        window_relock: bool = True,
    ) -> None:
        self.max_ngram = max(2, int(max_ngram))
        self.min_ngram = max(1, min(self.max_ngram, int(min_ngram)))
        self.max_k = max(1, int(max_k))
        self.max_hits_per_ngram = max(1, int(max_hits_per_ngram))
        self.use_cursor = bool(use_cursor)
        self.window_relock = bool(window_relock)

        merged: list[int] = []
        primary = [int(t) for t in vision_token_ids if int(t) >= 0]
        if primary:
            merged.extend(primary)
        for extra in extra_token_ids:
            clean = [int(t) for t in extra if int(t) >= 0]
            if clean:
                if merged:
                    merged.append(_CORPUS_SENTINEL)
                merged.extend(clean)
        self._vision_ids: list[int] = merged
        self._vision_index: dict[tuple[int, ...], list[int]] = {}
        self._cursor: int = 0
        self._self_indexed_upto: int = 0
        self._self_anchors: list[tuple[int, int]] = []
        self._self_index: dict[tuple[int, ...], int] = {}
        if self._vision_ids:
            self._index_vision_tokens()

    def reset_cursor(self) -> None:
        """Reset monotonic reading-order cursor and incremental self-history index."""
        self._cursor = 0
        self._self_indexed_upto = 0
        self._self_anchors.clear()
        self._self_index.clear()

    def _index_vision_tokens(self) -> None:
        n = len(self._vision_ids)
        v = self._vision_ids
        cap = self.max_hits_per_ngram
        for ng in range(self.min_ngram, self.max_ngram + 1):
            for pos in range(0, n - ng):
                span = v[pos : pos + ng]
                if _CORPUS_SENTINEL in span or v[pos + ng] == _CORPUS_SENTINEL:
                    continue
                key = tuple(span)
                slots = self._vision_index.setdefault(key, [])
                if len(slots) < cap:
                    slots.append(pos)

    def _self_prefix_matches(self, generated_ids: Sequence[int]) -> bool:
        n = self._self_indexed_upto
        if n == 0:
            return True
        if len(generated_ids) < n:
            return False
        for idx, val in self._self_anchors:
            if int(generated_ids[idx]) != val:
                return False
        return True

    def _sync_self_index(self, generated_ids: Sequence[int]) -> None:
        g_len = len(generated_ids)
        if not self._self_prefix_matches(generated_ids):
            self._self_indexed_upto = 0
            self._self_anchors.clear()
            self._self_index.clear()
        start_from = self._self_indexed_upto
        for ng in range(self.min_ngram, self.max_ngram + 1):
            lo = max(0, start_from - ng)
            hi = g_len - ng
            for pos in range(lo, hi):
                key = tuple(int(generated_ids[pos + off]) for off in range(ng))
                self._self_index[key] = pos
        self._self_indexed_upto = g_len
        if g_len > 0:
            anchor_idxs = sorted({0, g_len // 2, max(0, g_len - 2), g_len - 1})
            self._self_anchors = [(idx, int(generated_ids[idx])) for idx in anchor_idxs]
        else:
            self._self_anchors.clear()

    def _pick_vision_hit(self, hits: list[int], ng: int) -> int:
        if not self.use_cursor or len(hits) == 1:
            chosen = hits[0]
        else:
            floor = max(0, self._cursor - 32)
            idx = bisect_left(hits, floor)
            chosen = hits[idx] if idx < len(hits) else hits[-1]
        if self.use_cursor:
            self._cursor = chosen + ng
        return chosen

    def _slice_vision_draft(self, start: int, k: int) -> list[int]:
        v = self._vision_ids
        end = min(len(v), start + k)
        out: list[int] = []
        for idx in range(start, end):
            tok = v[idx]
            if tok == _CORPUS_SENTINEL:
                break
            out.append(tok)
        return out

    def draft(
        self,
        generated_ids: Sequence[int],
        *,
        mode: str = "vision+self",
        max_k: int | None = None,
    ) -> list[int]:
        """Propose up to `k` speculative next token IDs using longest n-gram match."""
        k = self.max_k if max_k is None else max(1, int(max_k))
        g_len = len(generated_ids)
        if g_len < self.min_ngram:
            return []

        use_self = mode in ("self", "vision+self")
        use_vision = mode in ("vision", "vision+self") and bool(self._vision_ids)
        if use_self:
            self._sync_self_index(generated_ids)

        for ng in range(min(self.max_ngram, g_len), self.min_ngram - 1, -1):
            needle = tuple(int(t) for t in generated_ids[-ng:])

            if use_self and g_len >= ng + 1:
                self_pos = self._self_index.get(needle)
                if self_pos is not None and self_pos < g_len - ng:
                    end = min(g_len, self_pos + ng + k)
                    if end > self_pos + ng:
                        return [int(generated_ids[idx]) for idx in range(self_pos + ng, end)]

            if use_vision:
                vis_hits = self._vision_index.get(needle)
                if vis_hits:
                    chosen = self._pick_vision_hit(vis_hits, ng)
                    cand = self._slice_vision_draft(chosen + ng, k)
                    if cand:
                        return cand

        if self.window_relock and self.use_cursor and use_vision and g_len >= 1:
            last_tok = int(generated_ids[-1])
            lo = max(0, self._cursor - 8)
            hi = min(len(self._vision_ids) - 1, self._cursor + 48)
            relock_hits = [p for p in range(lo, hi) if self._vision_ids[p] == last_tok]
            if len(relock_hits) == 1:
                pos = relock_hits[0]
                if self._vision_ids[pos + 1] != _CORPUS_SENTINEL:
                    cand = self._slice_vision_draft(pos + 1, k)
                    if cand:
                        self._cursor = pos + 1
                        return cand

        return []


def _matcher_has_any_pruned_token(
    matcher: Any,
    json_gram: Any,
    new_to_old: np.ndarray | None,
) -> bool:
    """Return True if at least one token in `new_to_old` is legal under `matcher`."""
    _ = new_to_old
    if json_gram is None:
        return True
    mask_lock = getattr(json_gram, "_mask_lock", None)
    mask_ptr = getattr(json_gram, "_mask_ptr", None)
    mask_nbytes = getattr(json_gram, "_mask_nbytes", None)
    mask_buf = getattr(json_gram, "_mask_buf", None)
    k_word = getattr(json_gram, "k_word", None)
    k_bit = getattr(json_gram, "k_bit", None)
    if (
        mask_lock is not None
        and mask_ptr is not None
        and mask_nbytes is not None
        and mask_buf is not None
        and k_word is not None
        and k_bit is not None
    ):
        with mask_lock:
            matcher.unsafe_compute_mask_ptr(mask_ptr, mask_nbytes)
            allowed = ((mask_buf[k_word] >> k_bit) & 1) != 0
            return bool(np.any(allowed))
    probe_fn = getattr(json_gram, "has_any_allowed", None)
    if callable(probe_fn):
        return bool(probe_fn(matcher))
    return True


def filter_drafts_with_grammar(
    drafts: Sequence[int],
    matcher: Any,
    stop_token_ids: AbstractSet[int] = frozenset(),
    *,
    json_gram: Any = None,
    vocab_size: int | None = None,
    new_to_old: np.ndarray | None = None,
    old_to_new: np.ndarray | None = None,
    max_len: int | None = None,
) -> list[int]:
    """Prefix-filter `drafts` so every speculative step matches greedy `LLMatcher` semantics.

    Stops BEFORE any token that is outside bounds/`old_to_new`, rejected by `matcher`, would put
    `matcher` into `is_stopped()`, would emit non-empty `compute_ff_tokens()`, or would leave zero
    valid tokens in `new_to_old`.
    """
    if not drafts:
        return []
    limit = len(drafts) if max_len is None else max(0, int(max_len))
    if limit == 0:
        return []
    if matcher is None:
        safe_unconstrained: list[int] = []
        for raw_dt in drafts[:limit]:
            dt = int(raw_dt)
            if dt in stop_token_ids or dt < 0 or (vocab_size is not None and dt >= int(vocab_size)):
                break
            if old_to_new is not None and (dt >= len(old_to_new) or int(old_to_new[dt]) < 0):
                break
            safe_unconstrained.append(dt)
        return safe_unconstrained

    if matcher.is_stopped():
        return []
    sim = matcher.deep_copy()
    if not _matcher_has_any_pruned_token(sim, json_gram, new_to_old):
        return []
    safe: list[int] = []
    for raw_dt in drafts[:limit]:
        dt = int(raw_dt)
        if dt in stop_token_ids or dt < 0 or (vocab_size is not None and dt >= int(vocab_size)):
            break
        if old_to_new is not None and (dt >= len(old_to_new) or int(old_to_new[dt]) < 0):
            break
        if not sim.validate_tokens([dt]):
            break
        if not sim.consume_token(dt):
            break
        if sim.is_stopped():
            break
        ff = [int(t) for t in sim.compute_ff_tokens() if int(t) not in stop_token_ids]
        if ff:
            break
        if not _matcher_has_any_pruned_token(sim, json_gram, new_to_old):
            break
        safe.append(dt)
    return safe


OCRPLDDrafter = PromptLookupDrafter

__all__ = [
    "OCRPLDDrafter",
    "PromptLookupDrafter",
    "filter_drafts_with_grammar",
]
