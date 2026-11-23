"""Zero-training Prompt & Vision-Text Suffix Lookup Drafter (`OCRPLDDrafter`)."""

from __future__ import annotations

from collections.abc import Sequence


class OCRPLDDrafter:
    """Zero-parameter speculative drafter combining:
      1. Apple-Vision ANE corpus n-gram lookup (captures printed words, names, and numbers on first pass).
      2. Online Self-PLD over already generated tokens (captures repeating HTML/Markdown table rows & cells).
    """

    def __init__(
        self,
        vision_token_ids: Sequence[int] = (),
        *,
        max_ngram: int = 4,
        min_ngram: int = 2,
        max_k: int = 6,
    ) -> None:
        self.max_ngram = max(2, int(max_ngram))
        self.min_ngram = max(1, min(self.max_ngram, int(min_ngram)))
        self.max_k = max(1, int(max_k))
        self._vision_ids: list[int] = [int(t) for t in vision_token_ids]
        # Map n-gram tuple -> list of start indices in `_vision_ids` (keep up to 4 occurrences for disambiguation)
        self._vision_index: dict[tuple[int, ...], list[int]] = {}
        if self._vision_ids:
            self._index_vision_tokens()

    def _index_vision_tokens(self) -> None:
        n = len(self._vision_ids)
        v = self._vision_ids
        for ng in range(self.min_ngram, self.max_ngram + 1):
            for pos in range(0, n - ng):
                key = tuple(v[pos : pos + ng])
                slots = self._vision_index.setdefault(key, [])
                if len(slots) < 4:
                    slots.append(pos)

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

        for ng in range(min(self.max_ngram, g_len), self.min_ngram - 1, -1):
            needle = tuple(int(t) for t in generated_ids[-ng:])

            # Check online self-history first (prefer the most recent table-structure or repeated pattern)
            if use_self and g_len >= ng + 1:
                for pos in range(g_len - ng - 1, -1, -1):
                    if all(int(generated_ids[pos + off]) == needle[off] for off in range(ng)):
                        end = min(g_len, pos + ng + k)
                        return [int(generated_ids[idx]) for idx in range(pos + ng, end)]

            # Fall back to Apple Vision ANE corpus match
            if use_vision:
                hits = self._vision_index.get(needle)
                if hits:
                    pos = hits[0]
                    end = min(len(self._vision_ids), pos + ng + k)
                    if end > pos + ng:
                        return [int(t) for t in self._vision_ids[pos + ng : end]]

        return []
