"""Thin re-export of shared `fast_shared.qwen35_vision` 3D interleaved MRoPE helpers for `fast-ocr`."""

from __future__ import annotations

from fast_shared.qwen35_vision import (
    compute_mrope_position_selector,
    compute_mrope_positions,
    precompute_mrope_trig,
)

__all__ = [
    "compute_mrope_position_selector",
    "compute_mrope_positions",
    "precompute_mrope_trig",
]
