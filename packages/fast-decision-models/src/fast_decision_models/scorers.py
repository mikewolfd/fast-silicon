"""Unified stdlib-safe facade exposing both typed-question decision-model backends."""

from __future__ import annotations

from .kev_mlx import Kev08bMlxBackend
from .laya_ane import LayaAneBackend
from .paths import ensure_laya_dylib, resolve_artifacts_dir, resolve_home

resolve_validators_dir = resolve_artifacts_dir

__all__ = [
    "Kev08bMlxBackend",
    "LayaAneBackend",
    "ensure_laya_dylib",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_validators_dir",
]
