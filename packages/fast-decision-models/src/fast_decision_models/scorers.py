"""Unified stdlib-safe facade exposing typed-question decision-model backends."""

from __future__ import annotations

from .calibration import (
    CalibrationProfile,
    DecisionCascadeScorer,
    apply_multiclass_temperature,
    apply_platt_scaling,
    check_probability_validity,
    fit_multiclass_temperature,
    fit_platt_scaling,
)
from .kev_mlx import Kev08bMlxBackend, Kev4bMlxBackend, KevMlxBackend
from .laya_ane import LayaAneBackend
from .laya_hetero import LayaHeteroEngine, LayaMlxGpuBackend
from .paths import ensure_laya_dylib, resolve_artifacts_dir, resolve_home

resolve_validators_dir = resolve_artifacts_dir

__all__ = [
    "CalibrationProfile",
    "DecisionCascadeScorer",
    "Kev08bMlxBackend",
    "Kev4bMlxBackend",
    "KevMlxBackend",
    "LayaAneBackend",
    "LayaHeteroEngine",
    "LayaMlxGpuBackend",
    "apply_multiclass_temperature",
    "apply_platt_scaling",
    "check_probability_validity",
    "ensure_laya_dylib",
    "fit_multiclass_temperature",
    "fit_platt_scaling",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_validators_dir",
]
