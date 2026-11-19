"""Hardware-limit Apple Silicon contextual embedding engine for pplx-embed-v2-context-0.6b."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .gpu import FastPplxGPU
from .paths import RepoPaths, require_runtime_paths, resolve_paths
from .server import FastPplxServer
from .tokenizer import ContextualTokenizer, FastPplxTokenizer

if TYPE_CHECKING:
    from .engine import FastPplx


def __getattr__(name: str) -> Any:
    if name == "FastPplx":
        from .engine import FastPplx

        return FastPplx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContextualTokenizer",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PPLX_CONFIG",
    "FastPplx",
    "FastPplxGPU",
    "FastPplxServer",
    "FastPplxTokenizer",
    "PplxConfig",
    "RepoPaths",
    "require_runtime_paths",
    "resolve_paths",
]
