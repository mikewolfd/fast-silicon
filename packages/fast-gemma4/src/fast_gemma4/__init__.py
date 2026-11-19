"""Hardware-limit Apple Silicon Gemma-4 E2B inference engine."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnePrefixRunner": (".ane_pipeline", "AnePrefixRunner"),
    "AnePrefixWorker": (".ane_pipeline", "AnePrefixWorker"),
    "DEFAULT_HF_REPO": (".paths", "DEFAULT_HF_REPO"),
    "FastGemma4Engine": (".engine", "FastGemma4Engine"),
    "FastGemma4Model": (".model", "FastGemma4Model"),
    "FastGemma4OpenAIAdapter": (".server", "FastGemma4OpenAIAdapter"),
    "FastGemma4Server": (".server", "FastGemma4Server"),
    "FastGemma4Tokenizer": (".tokenizer", "FastGemma4Tokenizer"),
    "Gemma4Paths": (".paths", "Gemma4Paths"),
    "GenerationBatchResult": (".engine", "GenerationBatchResult"),
    "GenerationItem": (".engine", "GenerationItem"),
    "GrammarHelper": (".grammar", "GrammarHelper"),
    "PackReport": (".packager", "PackReport"),
    "ParquetExtractSummary": (".pipeline", "ParquetExtractSummary"),
    "StripReport": (".packager", "StripReport"),
    "VocabRemapper": (".tokenizer", "VocabRemapper"),
    "extract_jsonl": (".pipeline", "extract_jsonl"),
    "extract_parquet": (".pipeline", "extract_parquet"),
    "format_gemma4_chat": (".tokenizer", "format_gemma4_chat"),
    "inspect_safetensors_headers": (".packager", "inspect_safetensors_headers"),
    "make_gemma4_handler": (".server", "make_gemma4_handler"),
    "pack_checkpoint_dir": (".packager", "pack_checkpoint_dir"),
    "resolve_paths": (".paths", "resolve_paths"),
}

try:
    from .ane_pipeline import AnePrefixRunner, AnePrefixWorker
    from .engine import FastGemma4Engine, GenerationBatchResult, GenerationItem
    from .grammar import GrammarHelper
    from .model import FastGemma4Model
    from .paths import DEFAULT_HF_REPO, Gemma4Paths, resolve_paths
    from .pipeline import ParquetExtractSummary, extract_jsonl, extract_parquet
    from .server import FastGemma4OpenAIAdapter, FastGemma4Server, make_gemma4_handler
    from .tokenizer import FastGemma4Tokenizer, VocabRemapper, format_gemma4_chat
except ModuleNotFoundError:
    # Allow the stdlib-only module (`fast_gemma4.per_doc_grammar`) to import
    # where mlx is not installed (`/usr/bin/python3 -B -I -S`).
    pass


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        mod_name, attr_name = target
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnePrefixRunner",
    "AnePrefixWorker",
    "DEFAULT_HF_REPO",
    "FastGemma4Engine",
    "FastGemma4Model",
    "FastGemma4OpenAIAdapter",
    "FastGemma4Server",
    "FastGemma4Tokenizer",
    "Gemma4Paths",
    "GenerationBatchResult",
    "GenerationItem",
    "GrammarHelper",
    "PackReport",
    "ParquetExtractSummary",
    "StripReport",
    "VocabRemapper",
    "extract_jsonl",
    "extract_parquet",
    "format_gemma4_chat",
    "inspect_safetensors_headers",
    "make_gemma4_handler",
    "pack_checkpoint_dir",
    "resolve_paths",
]
