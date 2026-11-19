"""Hardware-limit Apple Silicon (Metal GPU + 16-core ANE + opt-in CPU SME) EmbeddingGemma-2 engine."""

from .engine import MAX_EMBED_DIM, MRL_DIMENSIONS, FastEmbeddingGemma2
from .paths import RepoPaths, resolve_home, resolve_paths
from .server import FastEmbeddingGemma2Server
from .tokenizer import DOCUMENT_PREFIX, QUERY_PREFIX, FastEmbeddingGemma2Tokenizer

__all__ = [
    "DOCUMENT_PREFIX",
    "FastEmbeddingGemma2",
    "FastEmbeddingGemma2Server",
    "FastEmbeddingGemma2Tokenizer",
    "MAX_EMBED_DIM",
    "MRL_DIMENSIONS",
    "QUERY_PREFIX",
    "RepoPaths",
    "resolve_home",
    "resolve_paths",
]
