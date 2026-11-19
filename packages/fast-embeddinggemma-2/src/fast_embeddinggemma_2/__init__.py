"""Hardware-limit M4 Pro (Metal MPP GPU + 16-core ANE + CPU SME) EmbeddingGemma-2 engine."""

from .engine import MAX_EMBED_DIM, MRL_DIMENSIONS, FastEmbeddingGemma2
from .paths import RepoPaths, resolve_paths
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
    "resolve_paths",
]
