"""Unified stdlib-safe facade exposing typed-question decision-model backends."""

from __future__ import annotations

from .calibration import (
    CalibrationProfile,
    DecisionCascadeScorer,
    align_oos_probabilities,
    apply_multiclass_temperature,
    apply_oos_decision,
    apply_platt_scaling,
    check_probability_validity,
    fit_calibrated_profile,
    fit_multiclass_temperature,
    fit_oof_calibrated_scores,
    fit_oos_threshold,
    fit_platt_scaling,
    stratified_kfold_indices,
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
    "align_oos_probabilities",
    "apply_multiclass_temperature",
    "apply_oos_decision",
    "apply_platt_scaling",
    "check_probability_validity",
    "ensure_laya_dylib",
    "fit_calibrated_profile",
    "fit_multiclass_temperature",
    "fit_oof_calibrated_scores",
    "fit_oos_threshold",
    "fit_platt_scaling",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_validators_dir",
    "stratified_kfold_indices",
]
