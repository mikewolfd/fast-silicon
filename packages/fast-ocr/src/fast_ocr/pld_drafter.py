"""Zero-training Cursor-Tracked Prompt & Vision-Text Suffix Lookup Drafter (`OCRPLDDrafter`)."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

_CORPUS_SENTINEL = -1


class OCRPLDDrafter:
    """Zero-parameter speculative drafter combining:
      1. Cursor-tracked Apple-Vision ANE / native-PDF corpus n-gram lookup (disambiguates repeated
         common n-grams monotonically in XY-cut reading order across dense multi-column pages).
      2. Incremental O(1) online Self-PLD over already generated tokens (captures repeating
         HTML/Markdown table rows & cells without O(N) Python scans).
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
    ) -> None:
        self.max_ngram = max(2, int(max_ngram))
        self.min_ngram = max(1, min(self.max_ngram, int(min_ngram)))
        self.max_k = max(1, int(max_k))
        self.max_hits_per_ngram = max(1, int(max_hits_per_ngram))
        self.use_cursor = bool(use_cursor)

        # Concatenate primary reading-order corpus + optional fallback corpora (e.g. raw Vision / PDF text)
        # separated by `_CORPUS_SENTINEL` so n-grams never span across corpus boundaries.
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
        # Index every position whose successor token exists (`pos + ng < g_len`).
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
            # Prefer earliest occurrence at or just behind the monotonic cursor (handles small backtracks).
            floor = max(0, self._cursor - 32)
            idx = bisect_left(hits, floor)
            if idx < len(hits):
                chosen = hits[idx]
            else:
                chosen = hits[-1]
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

            # Check online self-history first (prefer the most recent table-structure or repeated pattern).
            if use_self and g_len >= ng + 1:
                self_pos = self._self_index.get(needle)
                if self_pos is not None and self_pos < g_len - ng:
                    end = min(g_len, self_pos + ng + k)
                    if end > self_pos + ng:
                        return [int(generated_ids[idx]) for idx in range(self_pos + ng, end)]

            # Fall back to cursor-tracked Apple Vision ANE / native-PDF corpus match.
            if use_vision:
                vis_hits = self._vision_index.get(needle)
                if vis_hits:
                    chosen = self._pick_vision_hit(vis_hits, ng)
                    cand = self._slice_vision_draft(chosen + ng, k)
                    if cand:
                        return cand

        return []
