"""Unified stdlib-safe facade exposing typed-question decision-model backends."""

from __future__ import annotations

from .kev_mlx import Kev08bMlxBackend, Kev4bMlxBackend, KevMlxBackend
from .laya_ane import LayaAneBackend
from .laya_hetero import LayaHeteroEngine, LayaMlxGpuBackend
from .paths import ensure_laya_dylib, resolve_artifacts_dir, resolve_home

resolve_validators_dir = resolve_artifacts_dir

__all__ = [
    "Kev08bMlxBackend",
    "Kev4bMlxBackend",
    "KevMlxBackend",
    "LayaAneBackend",
    "LayaHeteroEngine",
    "LayaMlxGpuBackend",
    "ensure_laya_dylib",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_validators_dir",
]
