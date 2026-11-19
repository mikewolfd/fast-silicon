"""Shared Rust-backed `tokenizers` wrapper with sliding-window chunking and prompt prefixes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from tokenizers import Tokenizer


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
            while end > pos:
                if end < n and offsets[end - 1][1] > offsets[end][0]:
                    end -= 1
                    continue
                # Include whitespace ignored by the tokenizer, and never split
                # a Unicode character with multiple tokens sharing an offset.
                c1 = len(text) if end == n else max(offsets[end - 1][1], offsets[end][0])
                ids = self._tok.encode(eff_prefix + text[c0:c1], add_special_tokens=True).ids
                if len(ids) <= max_chunk_tokens:
                    break
                end -= 1
            else:
                raise ValueError("A source character or token span cannot fit max_chunk_tokens with its prefix")

            chunks.append((self._finalize_ids(ids, max_chunk_tokens), c0, c1))
            if end >= n:
                break
            next_pos = end - overlap
            if next_pos <= pos:
                raise ValueError("overlap leaves no forward progress after boundary encoding; reduce overlap or increase max_chunk_tokens")
            pos = next_pos
        return chunks
