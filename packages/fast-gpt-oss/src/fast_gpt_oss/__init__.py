"""Hardware-limit Apple Silicon `openai/gpt-oss-20b` inference engine."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_HF_REPO": (".paths", "DEFAULT_HF_REPO"),
    "FALLBACK_HF_REPO": (".paths", "FALLBACK_HF_REPO"),
    "FastGptOssEngine": (".engine", "FastGptOssEngine"),
    "FastGptOssModel": (".model", "FastGptOssModel"),
    "FastGptOssOpenAIAdapter": (".server", "FastGptOssOpenAIAdapter"),
    "FastGptOssServer": (".server", "FastGptOssServer"),
    "FastGptOssTokenizer": (".tokenizer", "FastGptOssTokenizer"),
    "GenerationBatchResult": (".engine", "GenerationBatchResult"),
    "GenerationItem": (".engine", "GenerationItem"),
    "GptOssConfig": (".model", "GptOssConfig"),
    "GptOssPaths": (".paths", "GptOssPaths"),
    "GrammarHelper": (".grammar", "GrammarHelper"),
    "KernelSpec": (".kernels", "KernelSpec"),
    "O200kVocabRemapper": (".tokenizer", "O200kVocabRemapper"),
    "ParquetExtractSummary": (".pipeline", "ParquetExtractSummary"),
    "WhitespaceGuard": (".grammar", "WhitespaceGuard"),
    "WhitespaceGuardedMatcher": (".grammar", "WhitespaceGuardedMatcher"),
    "WhitespaceProfile": (".grammar", "WhitespaceProfile"),
    "build_kernels": (".kernels", "build_kernels"),
    "extract_jsonl": (".pipeline", "extract_jsonl"),
    "extract_parquet": (".pipeline", "extract_parquet"),
    "format_harmony_chat": (".tokenizer", "format_harmony_chat"),
    "make_gpt_oss_handler": (".server", "make_gpt_oss_handler"),
    "parse_harmony_output": (".tokenizer", "parse_harmony_output"),
    "resolve_paths": (".paths", "resolve_paths"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        mod_name, attr_name = target
        if mod_name.startswith("."):
            mod = importlib.import_module(mod_name, __name__)
        else:
            mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS.keys())
