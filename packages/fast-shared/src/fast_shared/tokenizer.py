"""Shared Rust-backed `tokenizers` wrapper with 1-pass batch chunking and header-anchored windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tokenizers import Tokenizer


@dataclass(frozen=True)
class HeaderAnchoredWindow:
    """Contextual chunk window with optional repeated document header anchor."""

    window_index: int
    has_anchor: bool
    chunk_indices: tuple[int, ...]
    chunks: list[str]
    estimated_tokens: int


def plan_header_anchored_windows(
    chunks: Sequence[str],
    max_window_tokens: int = 2048,
    header_max_chars: int = 2048,
    *,
    token_count: Callable[[str], int] | None = None,
    tokenizer: Any | None = None,
    overhead_tokens: int | None = None,
    per_chunk_extra_tokens: int = 2,
) -> list[HeaderAnchoredWindow]:
    """Group ordered chunks into `<= max_window_tokens` windows, anchoring continuation windows with `chunks[0][:header_max_chars]`."""
    max_win = int(max_window_tokens)
    hdr_chars = int(header_max_chars)
    extra = max(0, int(per_chunk_extra_tokens))
    if max_win <= 0:
        raise ValueError("max_window_tokens must be positive")
    if hdr_chars < 0:
        raise ValueError("header_max_chars must be nonnegative")
    if not chunks:
        return []

    chunk_list = [str(c) for c in chunks]
    eff_overhead = int(overhead_tokens) if overhead_tokens is not None else (64 if max_win >= 512 else 0)
    budget = max(1, max_win - max(0, eff_overhead))

    if token_count is not None:
        sizes = [max(1, int(token_count(c))) + extra for c in chunk_list]

        def _count_one(text: str) -> int:
            return max(1, int(token_count(text))) + extra

    elif tokenizer is not None:
        raw_tok = getattr(tokenizer, "_tok", None)
        if raw_tok is not None and hasattr(raw_tok, "encode_batch"):
            encs = raw_tok.encode_batch(chunk_list, add_special_tokens=False)
            sizes = [len(e.ids) + extra for e in encs]

            def _count_one(text: str) -> int:
                return len(raw_tok.encode(text, add_special_tokens=False).ids) + extra

        elif hasattr(tokenizer, "encode_batch"):
            encs_b = tokenizer.encode_batch(chunk_list)
            sizes = [len(ids) + extra for ids in encs_b]

            def _count_one(text: str) -> int:
                return len(tokenizer.encode(text)) + extra

        else:
            sizes = [len(tokenizer.encode(c)) + extra for c in chunk_list]

            def _count_one(text: str) -> int:
                return len(tokenizer.encode(text)) + extra

    else:
        sizes = [max(1, (len(c.encode("utf-8")) + 3) // 4) + extra for c in chunk_list]

        def _count_one(text: str) -> int:
            return max(1, (len(text.encode("utf-8")) + 3) // 4) + extra

    anchor = chunk_list[0][:hdr_chars] if hdr_chars > 0 else ""
    anchor_size = 0
    if anchor:
        anchor_size = sizes[0] if anchor == chunk_list[0] else _count_one(anchor)
    use_anchor = bool(anchor) and anchor_size < budget

    windows: list[HeaderAnchoredWindow] = []
    current_chunks: list[str] = [chunk_list[0]]
    current_indices: list[int] = [0]
    used = sizes[0]
    has_anchor = False

    for idx in range(1, len(chunk_list)):
        chunk_str = chunk_list[idx]
        size = sizes[idx]
        if used + size > budget:
            windows.append(
                HeaderAnchoredWindow(
                    window_index=len(windows),
                    has_anchor=has_anchor,
                    chunk_indices=tuple(current_indices),
                    chunks=current_chunks,
                    estimated_tokens=used,
                )
            )
            if use_anchor:
                current_chunks = [anchor, chunk_str]
                used = anchor_size + size
                has_anchor = True
            else:
                current_chunks = [chunk_str]
                used = size
                has_anchor = False
            current_indices = [idx]
        else:
            current_chunks.append(chunk_str)
            current_indices.append(idx)
            used += size

    windows.append(
        HeaderAnchoredWindow(
            window_index=len(windows),
            has_anchor=has_anchor,
            chunk_indices=tuple(current_indices),
            chunks=current_chunks,
            estimated_tokens=used,
        )
    )
    return windows


class BaseFastTokenizer:
    """Fast `tokenizers` wrapper with zero `transformers`/`torch` dependency."""

    def __init__(
        self,
        tokenizer_json: str | Path,
        *,
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        default_bos_id: int = 101,
        default_eos_id: int = 102,
        default_pad_id: int = 0,
        default_max_length: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
        prompts: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(tokenizer_json)
        self._tok = Tokenizer.from_file(str(self.path))
        self._tok.no_padding()
        self._tok.no_truncation()

        cls_id = self._tok.token_to_id(bos_token)
        sep_id = self._tok.token_to_id(eos_token)
        pad_id = self._tok.token_to_id(pad_token)
        self.bos_id = int(cls_id) if cls_id is not None else int(default_bos_id)
        self.eos_id = int(sep_id) if sep_id is not None else int(default_eos_id)
        self.pad_id = int(pad_id) if pad_id is not None else int(default_pad_id)
        # Keep BERT-compatible aliases so callers don't break
        self.cls_id = self.bos_id
        self.sep_id = self.eos_id

        self.default_max_length = int(default_max_length)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.prompts = dict(prompts or {})
        self._trunc_cache: dict[int, Tokenizer] = {}
        self._prefix_frame_cache: dict[str, tuple[list[int], list[int]]] = {}
        self._get_trunc_tok(self.default_max_length)

    def _get_trunc_tok(self, max_length: int) -> Tokenizer:
        tok = self._trunc_cache.get(max_length)
        if tok is None:
            tok = Tokenizer.from_file(str(self.path))
            tok.no_padding()
            tok.enable_truncation(max_length=max_length)
            self._trunc_cache[max_length] = tok
        return tok

    def resolve_prefix(
        self,
        *,
        prefix: str | None = None,
        prompt_name: str | None = None,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
    ) -> str:
        if prefix is not None:
            return prefix
        if prompt_name is not None:
            if prompt_name not in self.prompts:
                raise KeyError(f"Unknown prompt_name {prompt_name!r}; available: {sorted(self.prompts)}")
            return self.prompts[prompt_name]
        if add_query_prefix:
            return self.query_prefix
        if add_document_prefix:
            return self.document_prefix
        return ""

    def _finalize_ids(self, ids: list[int], max_length: int) -> list[int]:
        if not ids:
            return [self.bos_id, self.eos_id]
        if len(ids) > max_length:
            return ids[: max_length - 1] + [self.eos_id]
        return ids

    def _resolve_special_and_prefix_ids(self, eff_prefix: str) -> tuple[list[int], list[int]]:
        cached = self._prefix_frame_cache.get(eff_prefix)
        if cached is not None:
            return cached

        probe_raw = self._tok.encode("a", add_special_tokens=False).ids
        probe_full = self._tok.encode("a", add_special_tokens=True).ids
        lead_special: list[int] | None = None
        tail_special: list[int] | None = None
        if probe_raw and len(probe_full) >= len(probe_raw):
            m = len(probe_raw)
            for idx in range(len(probe_full) - m + 1):
                if probe_full[idx : idx + m] == probe_raw:
                    lead_special = list(probe_full[:idx])
                    tail_special = list(probe_full[idx + m :])
                    break
        if lead_special is None or tail_special is None:
            empty_ids = self._tok.encode("", add_special_tokens=True).ids
            if len(empty_ids) >= 2:
                lead_special = list(empty_ids[:-1])
                tail_special = list(empty_ids[-1:])
            else:
                lead_special = [self.bos_id]
                tail_special = [self.eos_id]

        prefix_raw = self._tok.encode(eff_prefix, add_special_tokens=False).ids if eff_prefix else []
        frame = ([*lead_special, *prefix_raw], tail_special)
        self._prefix_frame_cache[eff_prefix] = frame
        return frame

    def encode(
        self,
        text: str,
        *,
        prefix: str | None = None,
        prompt_name: str | None = None,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ml = self.default_max_length if max_length is None else int(max_length)
        eff_prefix = self.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
        )
        if eff_prefix:
            text = eff_prefix + text
        ids = self._get_trunc_tok(ml).encode(text, add_special_tokens=True).ids
        return self._finalize_ids(ids, ml)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        prefix: str | None = None,
        prompt_name: str | None = None,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        max_length: int | None = None,
        batch_size: int = 2048,
    ) -> list[list[int]]:
        if not texts:
            return []
        ml = self.default_max_length if max_length is None else int(max_length)
        eff_prefix = self.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
        )
        inputs = [eff_prefix + t for t in texts] if eff_prefix else list(texts)
        tok = self._get_trunc_tok(ml)
        out: list[list[int]] = []
        for start in range(0, len(inputs), batch_size):
            encs = tok.encode_batch(inputs[start : start + batch_size], add_special_tokens=True)
            out.extend(self._finalize_ids(e.ids, ml) for e in encs)
        return out

    def chunk_text(
        self,
        text: str,
        *,
        max_body_tokens: int | None = None,
        overlap: int = 51,
        prefix: str | None = None,
        prompt_name: str | None = None,
        add_document_prefix: bool = False,
        max_chunk_tokens: int = 512,
    ) -> list[tuple[list[int], int, int]]:
        """Return `(token_ids, char_start, char_end)` chunks of the original text.

        Body windows and overlap use tokens from the original, unprefixed text.
        Each source slice is then encoded with its prefix and native special
        tokens, just like :meth:`encode`, without truncation. Boundary changes
        can shorten a window to keep its final encoding within max_chunk_tokens.
        max_body_tokens is an optional additional limit on the body window.
        Character ranges include intervening whitespace and cover all of text.
        """
        max_chunk_tokens = int(max_chunk_tokens)
        overlap = int(overlap)
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if max_body_tokens is not None and int(max_body_tokens) <= 0:
            raise ValueError("max_body_tokens must be positive")
        if overlap < 0:
            raise ValueError("overlap must be nonnegative")
        eff_prefix = self.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_document_prefix=add_document_prefix,
        )
        prefix_ids = self._tok.encode(eff_prefix, add_special_tokens=True).ids
        prefix_size = len(prefix_ids) if prefix_ids else 2
        max_allowed_body = max_chunk_tokens - prefix_size
        if max_allowed_body <= 0:
            raise ValueError("max_chunk_tokens must leave room for body tokens after the prefix and special tokens")
        eff_max_body = max_allowed_body if max_body_tokens is None else min(int(max_body_tokens), max_allowed_body)
        if overlap >= eff_max_body:
            raise ValueError(f"overlap must be smaller than the usable body window ({eff_max_body} tokens)")

        enc = self._tok.encode(text, add_special_tokens=False)
        offsets = enc.offsets
        chunks: list[tuple[list[int], int, int]] = []
        n = len(enc.ids)
        if not n:
            ids = self._tok.encode(eff_prefix + text, add_special_tokens=True).ids
            if len(ids) > max_chunk_tokens:
                raise ValueError("Text without body token offsets exceeds max_chunk_tokens with its prefix")
            return [(self._finalize_ids(ids, max_chunk_tokens), 0, len(text))]

        pos = 0
        while pos < n:
            end = min(n, pos + eff_max_body)
            c0 = 0 if pos == 0 else offsets[pos][0]
            # Step back off mid-character overlapping token boundaries first.
            while end > pos and end < n and offsets[end - 1][1] > offsets[end][0]:
                end -= 1
            if end <= pos:
                raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")

            c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
            ids = self._tok.encode(eff_prefix + text[c0:c1], add_special_tokens=True).ids
            if len(ids) > max_chunk_tokens:
                # Binary-search over valid character-boundary end positions in [pos + 1, end - 1].
                valid_ends = [
                    e for e in range(pos + 1, end) if e == n or offsets[e - 1][1] <= offsets[e][0]
                ]
                best: tuple[int, int, list[int]] | None = None
                lo = 0
                hi = len(valid_ends) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand_end = valid_ends[mid]
                    cand_c1 = len(text) if cand_end == n else max(offsets[cand_end - 1][1], offsets[cand_end][0])
                    cand_ids = self._tok.encode(eff_prefix + text[c0:cand_c1], add_special_tokens=True).ids
                    if len(cand_ids) <= max_chunk_tokens:
                        best = (cand_end, cand_c1, cand_ids)
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best is None:
                    raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")
                end, c1, ids = best

            chunks.append((self._finalize_ids(ids, max_chunk_tokens), c0, c1))
            if end >= n:
                break
            next_pos = end - overlap
            if next_pos <= pos:
                raise ValueError("overlap leaves no forward progress after boundary encoding; reduce overlap or increase max_chunk_tokens")
            pos = next_pos
        return chunks

    def chunk_texts_batch(
        self,
        texts: Sequence[str],
        *,
        max_body_tokens: int | None = None,
        overlap: int = 51,
        prefix: str | None = None,
        prompt_name: str | None = None,
        add_document_prefix: bool = False,
        max_chunk_tokens: int = 512,
        batch_size: int = 2048,
    ) -> list[list[tuple[list[int], int, int]]]:
        """1-pass Rust `Tokenizer.encode_batch(..., add_special_tokens=False)` sliding-window chunker.

        Encodes batches of strings once in multi-threaded Rust and slices token ID spans
        directly with pre-encoded prefix and special token IDs (zero per-slice Python re-tokenization).
        """
        max_chunk_tokens = int(max_chunk_tokens)
        overlap = int(overlap)
        batch_sz = int(batch_size)
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if max_body_tokens is not None and int(max_body_tokens) <= 0:
            raise ValueError("max_body_tokens must be positive")
        if overlap < 0:
            raise ValueError("overlap must be nonnegative")
        if batch_sz <= 0:
            raise ValueError("batch_size must be positive")

        eff_prefix = self.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_document_prefix=add_document_prefix,
        )
        lead_ids, tail_ids = self._resolve_special_and_prefix_ids(eff_prefix)
        frame_size = (len(lead_ids) + len(tail_ids)) if (lead_ids or tail_ids) else 2
        max_allowed_body = max_chunk_tokens - frame_size
        if max_allowed_body <= 0:
            raise ValueError("max_chunk_tokens must leave room for body tokens after the prefix and special tokens")
        eff_max_body = max_allowed_body if max_body_tokens is None else min(int(max_body_tokens), max_allowed_body)
        if overlap >= eff_max_body:
            raise ValueError(f"overlap must be smaller than the usable body window ({eff_max_body} tokens)")

        if not texts:
            return []

        norm_texts = [t if isinstance(t, str) else str(t) for t in texts]
        empty_chunk_ids = self._finalize_ids([*lead_ids, *tail_ids], max_chunk_tokens)
        results: list[list[tuple[list[int], int, int]]] = []

        for start in range(0, len(norm_texts), batch_sz):
            sub_texts = norm_texts[start : start + batch_sz]
            encs = self._tok.encode_batch(sub_texts, add_special_tokens=False)
            for text, enc in zip(sub_texts, encs):
                body_ids = enc.ids
                offsets = enc.offsets
                n = len(body_ids)
                if not n:
                    results.append([(list(empty_chunk_ids), 0, len(text))])
                    continue

                doc_chunks: list[tuple[list[int], int, int]] = []
                pos = 0
                while pos < n:
                    end = min(n, pos + eff_max_body)
                    c0 = 0 if pos == 0 else offsets[pos][0]
                    while end > pos and end < n and offsets[end - 1][1] > offsets[end][0]:
                        end -= 1
                    if end <= pos:
                        raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")

                    c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
                    win_ids = self._finalize_ids([*lead_ids, *body_ids[pos:end], *tail_ids], max_chunk_tokens)
                    doc_chunks.append((win_ids, c0, c1))
                    if end >= n:
                        break

                    raw_next = end - overlap
                    if raw_next <= pos:
                        raise ValueError(
                            "overlap leaves no forward progress after boundary encoding; reduce overlap or increase max_chunk_tokens"
                        )
                    next_pos = raw_next
                    while next_pos > 0 and next_pos < n and offsets[next_pos - 1][1] > offsets[next_pos][0]:
                        next_pos -= 1
                    if next_pos <= pos:
                        next_pos = raw_next
                        while next_pos < end and offsets[next_pos - 1][1] > offsets[next_pos][0]:
                            next_pos += 1
                    pos = next_pos
                results.append(doc_chunks)

        return results

    def plan_header_anchored_windows(
        self,
        chunks: Sequence[str],
        max_window_tokens: int = 2048,
        header_max_chars: int = 2048,
        *,
        overhead_tokens: int | None = None,
        per_chunk_extra_tokens: int = 2,
    ) -> list[HeaderAnchoredWindow]:
        return plan_header_anchored_windows(
            chunks,
            max_window_tokens=max_window_tokens,
            header_max_chars=header_max_chars,
            tokenizer=self,
            overhead_tokens=overhead_tokens,
            per_chunk_extra_tokens=per_chunk_extra_tokens,
        )
