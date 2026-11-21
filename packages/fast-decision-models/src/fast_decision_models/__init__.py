"""fast-decision-models: Apple Silicon ANE (`Laya-Typed-421M`) & MLX (`Kev-0.8B`) decision runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .paths import (
    DEFAULT_KEV_ADAPTER_REPO,
    DEFAULT_KEV_BASE_REPO,
    DEFAULT_LAYA_REPO,
    DecisionModelPaths,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
    resolve_kev_dir,
    resolve_laya_dir,
    resolve_paths,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "KEV_SPECIAL_TOKENS": (".kev_mlx", "KEV_SPECIAL_TOKENS"),
    "Kev08bMlxBackend": (".kev_mlx", "Kev08bMlxBackend"),
    "LayaAneBackend": (".laya_ane", "LayaAneBackend"),
    "LayaTokenizerHelper": (".laya_ane", "LayaTokenizerHelper"),
    "PrefixMatch": (".prefix_tree", "PrefixMatch"),
    "PrefixTree": (".prefix_tree", "PrefixTree"),
    "acquire_hw_lock": ("._common", "acquire_hw_lock"),
    "build_kev_mlx_artifacts": (".builder", "build_kev_mlx_artifacts"),
    "build_laya_ane_artifacts": (".builder", "build_laya_ane_artifacts"),
    "compile_native_dylib": (".builder", "compile_native_dylib"),
    "encode_kev_questions": (".kev_mlx", "encode_kev_questions"),
    "quantize_kev_w8_artifacts": (".builder", "quantize_kev_w8_artifacts"),
    "recommended_prefix_budget_bytes": (".prefix_tree", "recommended_prefix_budget_bytes"),
    "resolve_kev_special_ids": (".kev_mlx", "resolve_kev_special_ids"),
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
    "DEFAULT_KEV_ADAPTER_REPO",
    "DEFAULT_KEV_BASE_REPO",
    "DEFAULT_LAYA_REPO",
    "DecisionModelPaths",
    "KEV_SPECIAL_TOKENS",
    "Kev08bMlxBackend",
    "LayaAneBackend",
    "LayaTokenizerHelper",
    "PrefixMatch",
    "PrefixTree",
    "acquire_hw_lock",
    "build_kev_mlx_artifacts",
    "build_laya_ane_artifacts",
    "compile_native_dylib",
    "encode_kev_questions",
    "ensure_laya_dylib",
    "quantize_kev_w8_artifacts",
    "recommended_prefix_budget_bytes",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_kev_dir",
    "resolve_kev_special_ids",
    "resolve_laya_dir",
    "resolve_paths",
]
