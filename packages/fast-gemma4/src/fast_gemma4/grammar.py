"""Backward-compatible re-export of `fast_shared.grammar` for `fast-gemma4`."""

from __future__ import annotations

from fast_shared.grammar import (
    GrammarHelper,
    _sample_from_logits,
    normalize_json_schema as _normalize_schema,
    sample_unconstrained,
    validate_json_against_schema,
)

__all__ = [
    "GrammarHelper",
    "_normalize_schema",
    "_sample_from_logits",
    "sample_unconstrained",
    "validate_json_against_schema",
]
