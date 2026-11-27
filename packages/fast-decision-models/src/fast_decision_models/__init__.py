"""fast-decision-models: Apple Silicon ANE/Hetero (`Laya-Typed-421M`) & MLX (`Kev-0.8B` / `Kev-4B`) decision runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .paths import (
    DEFAULT_KEV_08B_MLX_REPO,
    DEFAULT_KEV_4B_ADAPTER_REPO,
    DEFAULT_KEV_4B_BASE_REPO,
    DEFAULT_KEV_4B_MLX_REPO,
    DEFAULT_KEV_ADAPTER_REPO,
    DEFAULT_KEV_BASE_REPO,
    DEFAULT_LAYA_COREML_REPO,
    DEFAULT_LAYA_NMTZ_COREML_REPO,
    DEFAULT_LAYA_REPO,
    DecisionModelPaths,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
    resolve_kev_4b_dir,
    resolve_kev_dir,
    resolve_laya_dir,
    resolve_paths,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BEST_OF_K_GENRM_SYSTEM": (".genrm", "BEST_OF_K_GENRM_SYSTEM"),
    "CalibrationProfile": (".calibration", "CalibrationProfile"),
    "DecisionCascadeScorer": (".calibration", "DecisionCascadeScorer"),
    "GenRMBestOfKResult": (".genrm", "GenRMBestOfKResult"),
    "GenRMScorer": (".genrm", "GenRMScorer"),
    "KEV_SPECIAL_TOKENS": (".kev_mlx", "KEV_SPECIAL_TOKENS"),
    "Kev08bMlxBackend": (".kev_mlx", "Kev08bMlxBackend"),
    "Kev4bMlxBackend": (".kev_mlx", "Kev4bMlxBackend"),
    "KevMlxBackend": (".kev_mlx", "KevMlxBackend"),
    "LayaAneBackend": (".laya_ane", "LayaAneBackend"),
    "LayaHeteroEngine": (".laya_hetero", "LayaHeteroEngine"),
    "LayaMlxGpuBackend": (".laya_hetero", "LayaMlxGpuBackend"),
    "LayaTokenizerHelper": (".laya_ane", "LayaTokenizerHelper"),
    "PrefixMatch": (".prefix_tree", "PrefixMatch"),
    "PrefixTree": (".prefix_tree", "PrefixTree"),
    "RAGTRUTH_GENRM_SYSTEM": (".genrm", "RAGTRUTH_GENRM_SYSTEM"),
    "acquire_hw_lock": ("._common", "acquire_hw_lock"),
    "align_oos_probabilities": (".calibration", "align_oos_probabilities"),
    "apply_multiclass_temperature": (".calibration", "apply_multiclass_temperature"),
    "apply_oos_decision": (".calibration", "apply_oos_decision"),
    "apply_platt_scaling": (".calibration", "apply_platt_scaling"),
    "build_kev_mlx_artifacts": (".builder", "build_kev_mlx_artifacts"),
    "build_laya_ane_artifacts": (".builder", "build_laya_ane_artifacts"),
    "check_probability_validity": (".calibration", "check_probability_validity"),
    "compile_native_dylib": (".builder", "compile_native_dylib"),
    "encode_kev_questions": (".kev_mlx", "encode_kev_questions"),
    "fit_calibrated_profile": (".calibration", "fit_calibrated_profile"),
    "fit_multiclass_temperature": (".calibration", "fit_multiclass_temperature"),
    "fit_oof_calibrated_scores": (".calibration", "fit_oof_calibrated_scores"),
    "fit_oos_threshold": (".calibration", "fit_oos_threshold"),
    "fit_platt_scaling": (".calibration", "fit_platt_scaling"),
    "quantize_kev_w8_artifacts": (".builder", "quantize_kev_w8_artifacts"),
    "recommended_prefix_budget_bytes": (".prefix_tree", "recommended_prefix_budget_bytes"),
    "resolve_kev_special_ids": (".kev_mlx", "resolve_kev_special_ids"),
    "stratified_kfold_indices": (".calibration", "stratified_kfold_indices"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(target[0], __name__)
    val = getattr(mod, target[1])
    globals()[name] = val
    return val


__all__ = [
    "BEST_OF_K_GENRM_SYSTEM",
    "CalibrationProfile",
    "DEFAULT_KEV_08B_MLX_REPO",
    "DEFAULT_KEV_4B_ADAPTER_REPO",
    "DEFAULT_KEV_4B_BASE_REPO",
    "DEFAULT_KEV_4B_MLX_REPO",
    "DEFAULT_KEV_ADAPTER_REPO",
    "DEFAULT_KEV_BASE_REPO",
    "DEFAULT_LAYA_COREML_REPO",
    "DEFAULT_LAYA_NMTZ_COREML_REPO",
    "DEFAULT_LAYA_REPO",
    "DecisionCascadeScorer",
    "DecisionModelPaths",
    "GenRMBestOfKResult",
    "GenRMScorer",
    "KEV_SPECIAL_TOKENS",
    "Kev08bMlxBackend",
    "Kev4bMlxBackend",
    "KevMlxBackend",
    "LayaAneBackend",
    "LayaHeteroEngine",
    "LayaMlxGpuBackend",
    "LayaTokenizerHelper",
    "PrefixMatch",
    "PrefixTree",
    "RAGTRUTH_GENRM_SYSTEM",
    "acquire_hw_lock",
    "align_oos_probabilities",
    "apply_multiclass_temperature",
    "apply_oos_decision",
    "apply_platt_scaling",
    "build_kev_mlx_artifacts",
    "build_laya_ane_artifacts",
    "check_probability_validity",
    "compile_native_dylib",
    "encode_kev_questions",
    "ensure_laya_dylib",
    "fit_calibrated_profile",
    "fit_multiclass_temperature",
    "fit_oof_calibrated_scores",
    "fit_oos_threshold",
    "fit_platt_scaling",
    "quantize_kev_w8_artifacts",
    "recommended_prefix_budget_bytes",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_kev_4b_dir",
    "resolve_kev_dir",
    "resolve_kev_special_ids",
    "resolve_laya_dir",
    "resolve_paths",
    "stratified_kfold_indices",
]
