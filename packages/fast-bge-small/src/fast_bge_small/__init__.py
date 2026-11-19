"""fast-bge-small: Apple Silicon hardware-limit BGE-small-en-v1.5 inference engine."""

from .engine import FastBgeSmall
from .gpu import FastBgeSmallGPU, pack_sequences
from .paths import RepoPaths, resolve_paths
from .tokenizer import BGE_QUERY_PREFIX, FastBgeTokenizer

__all__ = [
    "BGE_QUERY_PREFIX",
    "FastBgeSmall",
    "FastBgeSmallGPU",
    "FastBgeTokenizer",
    "RepoPaths",
    "pack_sequences",
    "resolve_paths",
]

__version__ = "0.1.0"
