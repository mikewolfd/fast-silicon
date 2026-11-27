"""Backwards-compatible re-export of shared test-time self-consistency utilities."""

from __future__ import annotations

from fast_shared.self_consistency import (
    SelfConsistencyBatchResult,
    SelfConsistencySelection,
    generate_self_consistent,
    select_majority_answer,
)

__all__ = [
    "SelfConsistencyBatchResult",
    "SelfConsistencySelection",
    "generate_self_consistent",
    "select_majority_answer",
]