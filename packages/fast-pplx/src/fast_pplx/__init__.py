"""Hardware-limit Apple Silicon contextual embedding engine for pplx-embed-v2-context-0.6b."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import DEFAULT_MODEL_NAME, DEFAULT_PPLX_CONFIG, PplxConfig
from .paths import RepoPaths, require_runtime_paths, resolve_home, resolve_paths

if TYPE_CHECKING:
    from .engine import FastPplx
    from .gpu import FastPplxGPU
    from .server import FastPplxServer
    from .tokenizer import ContextualTokenizer, FastPplxTokenizer

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ContextualTokenizer": (".tokenizer", "ContextualTokenizer"),
    "FastPplx": (".engine", "FastPplx"),
    "FastPplxGPU": (".gpu", "FastPplxGPU"),
    "FastPplxServer": (".server", "FastPplxServer"),
    "FastPplxTokenizer": (".tokenizer", "FastPplxTokenizer"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        from importlib import import_module

        mod = import_module(target[0], __name__)
        return getattr(mod, target[1])
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
    "resolve_home",
    "resolve_paths",
]
