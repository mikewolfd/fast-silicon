"""Fast Rust-backed SentencePiece/BPE tokenizer for EmbeddingGemma-2."""

from __future__ import annotations

import json
from pathlib import Path

from fast_shared.tokenizer import BaseFastTokenizer

from .paths import resolve_paths

QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

DEFAULT_PROMPTS: dict[str, str] = {
    "query": QUERY_PREFIX,
    "document": DOCUMENT_PREFIX,
    "Retrieval-query": QUERY_PREFIX,
    "Retrieval-document": DOCUMENT_PREFIX,
    "SearchQuery": QUERY_PREFIX,
    "Document": DOCUMENT_PREFIX,
}


class FastEmbeddingGemma2Tokenizer(BaseFastTokenizer):
    def __init__(self, tokenizer_json: str | Path | None = None, home: str | Path | None = None) -> None:
        if tokenizer_json is None:
            tokenizer_json = resolve_paths(home).tokenizer_json
        tok_path = Path(tokenizer_json)
        prompts = dict(DEFAULT_PROMPTS)
        st_cfg = tok_path.parent / "config_sentence_transformers.json"
        if st_cfg.exists():
            try:
                loaded = json.loads(st_cfg.read_text()).get("prompts") or {}
                for k, v in loaded.items():
                    if isinstance(v, str):
                        prompts[k] = v
            except Exception:
                pass
        super().__init__(
            tok_path,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            default_bos_id=2,
            default_eos_id=1,
            default_pad_id=0,
            default_max_length=2048,
            query_prefix=QUERY_PREFIX,
            document_prefix=DOCUMENT_PREFIX,
            prompts=prompts,
        )

    def format_document_prefix(self, title: str | None = None) -> str:
        """Return native EmbeddingGemma-2 document prefix (`title: {title} | text: `)."""
        clean = title.strip() if isinstance(title, str) and title.strip() else "none"
        return f"title: {clean} | text: "

    def format_document_text(self, text: str, title: str | None = None) -> str:
        """Format document `text` with native `title: {title} | text: ` framing."""
        return f"{self.format_document_prefix(title)}{text}"
