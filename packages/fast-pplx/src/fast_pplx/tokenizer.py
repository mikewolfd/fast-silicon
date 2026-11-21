"""Pure-Rust `tokenizers` wrapper matching pplx-embed-v2-context-0.6b text/span semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tokenizers import Tokenizer

from .config import DEFAULT_PPLX_CONFIG, PplxConfig


def _resolve_tokenizer_json(path_or_file: str | Path) -> Path:
    p = Path(path_or_file)
    if p.is_dir():
        return p / "tokenizer.json"
    return p


class ContextualTokenizer:
    """Upstream `pplx-embed-v2-context` tokenizer with zero `torch`/`transformers` dependencies.

    Uses ``encode_special_tokens=True`` (equivalent to ``split_special_tokens=True``):
    - Queries prepend reserved prefix token ID ``248077`` (``[Q] ``).
    - Documents prepend literal ``[D] `` so user text cannot forge reserved control tokens;
      inter-chunk boundaries insert ``<|chunk_sep|>`` (``248079``), aligned to exact token spans.
    """

    def __init__(
        self,
        path_or_file: str | Path,
        config: PplxConfig = DEFAULT_PPLX_CONFIG,
        *,
        pad_token: str = "<|endoftext|>",
        default_max_length: int = 32768,
    ) -> None:
        self.path = _resolve_tokenizer_json(path_or_file)
        if not self.path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.path}")
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._tokenizer.encode_special_tokens = True
        self.config = config
        self.default_max_length = int(default_max_length)

        self.pad_id = self.reserved_id(pad_token)
        self.eos_id = self.pad_id
        self.bos_id = self.reserved_id(config.query_prefix)
        self.query_prefix_id = self.bos_id
        self.chunk_sep_id = self.reserved_id(config.boundary_marker)
        self.query_prefix = config.query_prefix
        self.document_prefix = config.document_prefix
        self.prompts = {
            "query": config.query_prefix,
            "document": config.document_prefix,
        }

    def reserved_id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Tokenizer has no reserved token {token!r}")
        return int(value)

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

    def pad(self, rows: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.bool_)
        width = max(1, max(len(row) for row in rows))
        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(rows), width), dtype=np.bool_)
        for index, row in enumerate(rows):
            if row:
                input_ids[index, : len(row)] = row
                attention_mask[index, : len(row)] = True
        return input_ids, attention_mask

    def encode_query(self, text: str, *, max_length: int | None = None) -> list[int]:
        ids = [self.query_prefix_id, *self._tokenizer.encode(text, add_special_tokens=False).ids]
        if max_length is not None:
            return ids[: int(max_length)]
        if len(ids) > self.config.query_length:
            raise ValueError(f"Query input exceeds the {self.config.query_length}-token limit")
        return ids

    def encode_document(self, text: str, *, max_length: int | None = None) -> list[int]:
        prefixed = self.config.document_prefix + text
        ids = list(self._tokenizer.encode(prefixed, add_special_tokens=False).ids)
        if max_length is not None:
            return ids[: int(max_length)]
        if len(ids) > self.config.document_length:
            raise ValueError(f"Document input exceeds the {self.config.document_length}-token limit")
        return ids

    def _align_chunk_spans(
        self,
        offsets_raw: Sequence[tuple[int, int]],
        char_spans: Sequence[tuple[int, int]],
        chunks: Sequence[str],
        max_length: int | None = None,
    ) -> list[tuple[int, int]]:
        if not char_spans:
            return []
        offsets = np.array(offsets_raw, dtype=np.int64).reshape(-1, 2)
        starts, ends = offsets[:, 0], offsets[:, 1]
        valid = ends > starts
        spans: list[tuple[int, int]] = []
        for col, (c_start, c_end) in enumerate(char_spans):
            indices = np.flatnonzero(valid & (starts < c_end) & (ends > c_start))
            if indices.size:
                t0, t1 = int(indices[0]), int(indices[-1]) + 1
                if max_length is not None:
                    if t0 >= max_length:
                        spans.append((0, 0))
                    else:
                        spans.append((t0, min(t1, max_length)))
                else:
                    spans.append((t0, t1))
            elif chunks[col]:
                raise ValueError(f"Chunk {col} produced no tokens")
            else:
                spans.append((0, 0))
        return spans

    def encode_contextual_document(
        self,
        chunks: Sequence[str],
        *,
        return_spans: bool = True,
        max_length: int | None = None,
    ) -> tuple[list[int], list[tuple[int, int]]] | list[int]:
        prefix, marker = self.config.document_prefix, self.config.boundary_marker
        parts: list[str] = [prefix]
        char_spans: list[tuple[int, int]] = []
        cursor = len(prefix)
        for index, chunk in enumerate(chunks):
            if index:
                parts.append(marker)
                cursor += len(marker)
            char_spans.append((cursor, cursor + len(chunk)))
            parts.append(chunk)
            cursor += len(chunk)

        encoded = self._tokenizer.encode("".join(parts), add_special_tokens=False)
        ids = list(encoded.ids)
        if max_length is None and len(ids) > self.config.document_length:
            raise ValueError(
                f"Input exceeds the {self.config.document_length}-token limit; split the document into context windows"
            )
        token_spans = self._align_chunk_spans(encoded.offsets, char_spans, chunks, max_length=max_length)
        if max_length is not None and len(ids) > max_length:
            ids = ids[: int(max_length)]
        return (ids, token_spans) if return_spans else ids

    def encode_queries(
        self,
        queries: Sequence[str] | Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        flat: list[str] = []
        for item in queries:
            if isinstance(item, str):
                flat.append(item)
            else:
                if len(item) != 1:
                    raise ValueError("Each query must contain exactly one string")
                flat.append(item[0])
        encoded = self._tokenizer.encode_batch(flat, add_special_tokens=False)
        rows = [[self.query_prefix_id, *item.ids] for item in encoded]
        if max_length is not None:
            ml = int(max_length)
            rows = [r[:ml] for r in rows]
        input_ids, attention_mask = self.pad(rows)
        if max_length is None and input_ids.shape[1] > self.config.query_length:
            raise ValueError(f"Input exceeds the {self.config.query_length}-token limit")
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_documents(
        self,
        documents: Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        prefix, marker = self.config.document_prefix, self.config.boundary_marker
        texts: list[str] = []
        all_char_spans: list[list[tuple[int, int]]] = []
        for chunks in documents:
            parts: list[str] = [prefix]
            spans: list[tuple[int, int]] = []
            cursor = len(prefix)
            for index, chunk in enumerate(chunks):
                if index:
                    parts.append(marker)
                    cursor += len(marker)
                spans.append((cursor, cursor + len(chunk)))
                parts.append(chunk)
                cursor += len(chunk)
            texts.append("".join(parts))
            all_char_spans.append(spans)

        encoded = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        rows = [list(item.ids) for item in encoded]
        if max_length is not None:
            ml = int(max_length)
            rows = [r[:ml] for r in rows]
        input_ids, attention_mask = self.pad(rows)
        if max_length is None and input_ids.shape[1] > self.config.document_length:
            raise ValueError(
                f"Input exceeds the {self.config.document_length}-token limit; split the document into context windows"
            )
        max_chunks = max(1, *(len(doc) for doc in documents)) if documents else 1
        token_spans = np.zeros((len(documents), max_chunks, 2), dtype=np.int32)
        for row, char_spans in enumerate(all_char_spans):
            aligned = self._align_chunk_spans(
                encoded[row].offsets,
                char_spans,
                documents[row],
                max_length=max_length,
            )
            for col, (t0, t1) in enumerate(aligned):
                token_spans[row, col] = (t0, t1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chunk_token_spans": token_spans,
        }

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
        if eff_prefix == self.config.query_prefix:
            return self.encode_query(text, max_length=ml)
        ids = list(self._tokenizer.encode(eff_prefix + text, add_special_tokens=False).ids)
        return ids[:ml]

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
        is_query = eff_prefix == self.config.query_prefix
        inputs = list(texts) if is_query else ([eff_prefix + t for t in texts] if eff_prefix else list(texts))
        out: list[list[int]] = []
        for start in range(0, len(inputs), batch_size):
            encs = self._tokenizer.encode_batch(inputs[start : start + batch_size], add_special_tokens=False)
            if is_query:
                out.extend(([self.query_prefix_id, *e.ids])[:ml] for e in encs)
            else:
                out.extend(list(e.ids)[:ml] for e in encs)
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
        """Return `(token_ids, char_start, char_end)` windows suitable for `parquet_pipeline`."""
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
        is_query = eff_prefix == self.config.query_prefix
        if is_query:
            prefix_size = 1
        elif eff_prefix:
            prefix_ids = self._tokenizer.encode(eff_prefix, add_special_tokens=False).ids
            prefix_size = max(1, len(prefix_ids))
        else:
            prefix_size = 0

        max_allowed_body = max_chunk_tokens - prefix_size
        if max_allowed_body <= 0:
            raise ValueError("max_chunk_tokens must leave room for body tokens after the prefix")
        eff_max_body = max_allowed_body if max_body_tokens is None else min(int(max_body_tokens), max_allowed_body)
        if overlap >= eff_max_body:
            raise ValueError(f"overlap must be smaller than the usable body window ({eff_max_body} tokens)")

        enc = self._tokenizer.encode(text, add_special_tokens=False)
        offsets = enc.offsets
        n = len(enc.ids)
        if not n:
            ids = (
                [self.query_prefix_id]
                if is_query
                else list(self._tokenizer.encode(eff_prefix + text, add_special_tokens=False).ids)
            )
            if len(ids) > max_chunk_tokens:
                raise ValueError("Text without body token offsets exceeds max_chunk_tokens with its prefix")
            return [(ids, 0, len(text))]

        chunks: list[tuple[list[int], int, int]] = []
        pos = 0
        while pos < n:
            end = min(n, pos + eff_max_body)
            c0 = 0 if pos == 0 else offsets[pos][0]
            while end > pos:
                if end < n and offsets[end - 1][1] > offsets[end][0]:
                    end -= 1
                    continue
                c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
                sub = text[c0:c1]
                if is_query:
                    ids = [self.query_prefix_id, *self._tokenizer.encode(sub, add_special_tokens=False).ids]
                else:
                    ids = list(self._tokenizer.encode(eff_prefix + sub, add_special_tokens=False).ids)
                if len(ids) <= max_chunk_tokens:
                    break
                end -= 1
            else:
                raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")

            chunks.append((ids, c0, c1))
            if end >= n:
                break
            next_pos = end - overlap
            if next_pos <= pos:
                raise ValueError(
                    "overlap leaves no forward progress after boundary encoding; reduce overlap or increase max_chunk_tokens"
                )
            pos = next_pos
        return chunks

    def group_chunks_by_context_windows(
        self,
        chunks: Sequence[str],
        *,
        max_context_tokens: int = 2048,
        context_tokens: int | None = None,
        header_text: str | None = None,
        repeat_header_tokens: int | Sequence[int] | None = None,
    ) -> list[list[str]]:
        """Greedily pack ordered document chunks into contextual windows <= `max_context_tokens`.

        Mirrors ``evals/vidore.py`` / ``evals/conteb.py`` two-knob contextual packing:
        accounts for ``[D] `` prefix tokens once per window plus one ``<|chunk_sep|>``
        token between adjacent chunks, without ever splitting an individual chunk.
        Supports optional header-anchoring via ``header_text`` (prepended to every window)
        or ``repeat_header_tokens`` (repeats the leading header tokens of ``chunks[0]`` or
        an explicit token sequence at the start of subsequent windows ``w >= 1``).
        """
        if not chunks:
            return []
        raw_limit = max_context_tokens if context_tokens is None else context_tokens
        limit = int(raw_limit)
        if limit <= 0:
            raise ValueError(f"max_context_tokens must be positive, got {raw_limit}")

        prefix_len = max(1, len(self._tokenizer.encode(self.config.document_prefix, add_special_tokens=False).ids))
        encs = self._tokenizer.encode_batch(list(chunks), add_special_tokens=False)

        anchor_w0: str | None = None
        anchor_w0_cost = 0
        anchor_subseq: str | None = None
        anchor_subseq_cost = 0

        if header_text is not None and header_text.strip():
            hdr_str = header_text.strip()
            hdr_len = len(self._tokenizer.encode(hdr_str, add_special_tokens=False).ids)
            anchor_subseq = hdr_str
            anchor_subseq_cost = hdr_len + 1
            if chunks[0].strip() != hdr_str:
                anchor_w0 = hdr_str
                anchor_w0_cost = hdr_len + 1
        elif repeat_header_tokens is not None:
            if isinstance(repeat_header_tokens, int):
                n_rep = int(repeat_header_tokens)
                if n_rep < 0:
                    raise ValueError("repeat_header_tokens must be non-negative")
                if n_rep > 0 and encs[0].ids:
                    rep_ids = list(encs[0].ids[:n_rep])
                    rep_str = self._tokenizer.decode(rep_ids, skip_special_tokens=False).strip()
                    if rep_str:
                        rep_len = len(self._tokenizer.encode(rep_str, add_special_tokens=False).ids)
                        anchor_subseq = rep_str
                        anchor_subseq_cost = rep_len + 1
            else:
                rep_ids = [int(t) for t in repeat_header_tokens]
                if rep_ids:
                    rep_str = self._tokenizer.decode(rep_ids, skip_special_tokens=False).strip()
                    if rep_str:
                        rep_len = len(self._tokenizer.encode(rep_str, add_special_tokens=False).ids)
                        anchor_subseq = rep_str
                        anchor_subseq_cost = rep_len + 1

        windows: list[list[str]] = [[]]
        used = prefix_len
        if anchor_w0 is not None and prefix_len + anchor_w0_cost + len(encs[0].ids) <= limit:
            windows[0].append(anchor_w0)
            used += anchor_w0_cost - 1

        for chunk, enc in zip(chunks, encs, strict=True):
            c_len = len(enc.ids)
            sep_cost = 1 if windows[-1] else 0
            cost = c_len + sep_cost
            if windows[-1] and used + cost > limit:
                # Only split when current window already has at least one body chunk (not just anchor_w0).
                has_body = len(windows[-1]) > (1 if (len(windows) == 1 and anchor_w0 is not None and windows[0][:1] == [anchor_w0]) else 0)
                if has_body:
                    windows.append([])
                    used = prefix_len
                    if anchor_subseq is not None and prefix_len + anchor_subseq_cost + c_len <= limit:
                        windows[-1].append(anchor_subseq)
                        used += anchor_subseq_cost - 1
                        cost = c_len + 1
                    else:
                        cost = c_len
            windows[-1].append(chunk)
            used += cost
        return windows

    def chunk_and_group_document(
        self,
        text: str,
        *,
        chunk_tokens: int = 256,
        max_context_tokens: int = 2048,
        context_tokens: int | None = None,
        overlap: int = 32,
        header_text: str | None = None,
        repeat_header_tokens: int | Sequence[int] | None = None,
    ) -> list[tuple[list[str], list[tuple[int, int]]]]:
        """Split `text` into <= `chunk_tokens` pieces and pack them into <= `max_context_tokens` windows.

        Returns ``[(window_chunks, [(char_start, char_end), ...]), ...]``.
        """
        eff_ctx = max_context_tokens if context_tokens is None else int(context_tokens)
        raw_pieces = self.chunk_text(
            text,
            max_chunk_tokens=int(chunk_tokens),
            overlap=int(overlap),
            add_document_prefix=False,
        )
        texts = [text[c0:c1] for _, c0, c1 in raw_pieces]
        offsets = [(c0, c1) for _, c0, c1 in raw_pieces]
        windows = self.group_chunks_by_context_windows(
            texts,
            max_context_tokens=eff_ctx,
            header_text=header_text,
            repeat_header_tokens=repeat_header_tokens,
        )
        grouped: list[tuple[list[str], list[tuple[int, int]]]] = []
        cursor = 0
        for win in windows:
            # Check whether window starts with an injected header anchor before texts[cursor]
            if (
                (header_text is not None or repeat_header_tokens is not None)
                and cursor < len(texts)
                and len(win) >= 2
                and win[0] != texts[cursor]
                and win[1] == texts[cursor]
            ):
                n_body = len(win) - 1
                hdr_end = min(len(text), offsets[0][0] + len(win[0])) if offsets else 0
                span_slice = [(offsets[0][0] if offsets else 0, hdr_end), *offsets[cursor : cursor + n_body]]
                grouped.append((win, span_slice))
                cursor += n_body
            else:
                span_slice = offsets[cursor : cursor + len(win)]
                grouped.append((win, span_slice))
                cursor += len(win)
        return grouped


FastPplxTokenizer = ContextualTokenizer

__all__ = ["ContextualTokenizer", "FastPplxTokenizer"]
