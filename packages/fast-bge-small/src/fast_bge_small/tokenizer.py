"""Lightweight WordPiece tokenizer backed by `fast_shared.BaseFastTokenizer`."""

from __future__ import annotations

from pathlib import Path

from fast_shared.tokenizer import BaseFastTokenizer

from .paths import resolve_paths

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastBgeTokenizer(BaseFastTokenizer):
    def __init__(self, tokenizer_json: str | Path | None = None, home: str | Path | None = None):
        if tokenizer_json is None:
            tokenizer_json = resolve_paths(home).tokenizer_json
        super().__init__(
            tokenizer_json,
            bos_token="[CLS]",
            eos_token="[SEP]",
            pad_token="[PAD]",
            default_bos_id=101,
            default_eos_id=102,
            default_pad_id=0,
            default_max_length=512,
            query_prefix=BGE_QUERY_PREFIX,
            prompts={"query": BGE_QUERY_PREFIX},
        )
