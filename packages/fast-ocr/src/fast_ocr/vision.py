"""Thin re-export of shared `fast_shared.qwen35_vision` Qwen3.5 Vision Tower for `fast-ocr`."""

from __future__ import annotations

from fast_shared.qwen35_vision import (
    FastQwen35VisionModel,
    Qwen35VisionConfig,
    VisionAttention,
    VisionBlock,
    VisionMLP,
    VisionPatchEmbed,
    VisionPatchMerger,
    _apply_vision_rope_qk,
    _normalize_vision_key,
    _rotate_half,
)

__all__ = [
    "FastQwen35VisionModel",
    "Qwen35VisionConfig",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
    "VisionPatchEmbed",
    "VisionPatchMerger",
    "_apply_vision_rope_qk",
    "_normalize_vision_key",
    "_rotate_half",
]
