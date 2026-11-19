"""Filesystem discovery for weights, tokenizer assets, ANE CoreML bundles, and native dylibs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fast_shared.paths import resolve_home_dir


@dataclass(frozen=True)
class RepoPaths:
    home: Path
    weights_dir: Path
    model_dir: Path
    embed_tokens_safetensors: Path
    weights_npz: Path
    tokenizer_dir: Path
    tokenizer_json: Path
    ane_models_dir: Path
    build_dir: Path
    coreml_dylib: Path
    io_dylib: Path
    ane_cli: Path


def resolve_paths(home: str | Path | None = None) -> RepoPaths:
    root = resolve_home_dir(
        __file__,
        env_var="FAST_PPLX_HOME",
        cache_name="fast-pplx",
        home=home,
    )
    artifacts = root / "artifacts"
    weights_dir = artifacts / "weights"
    tokenizer_dir = artifacts / "tokenizer"
    build_dir = root / "build"
    return RepoPaths(
        home=root,
        weights_dir=weights_dir,
        model_dir=weights_dir / "model",
        embed_tokens_safetensors=weights_dir / "embed_tokens.safetensors",
        weights_npz=weights_dir / "pplx_0_6b_f16.npz",
        tokenizer_dir=tokenizer_dir,
        tokenizer_json=tokenizer_dir / "tokenizer.json",
        ane_models_dir=artifacts / "ane_models",
        build_dir=build_dir,
        coreml_dylib=build_dir / "libpplx_coreml_runner.dylib",
        io_dylib=build_dir / "libpplx_fast_io.dylib",
        ane_cli=build_dir / "pplx-ane-cli",
    )


def require_runtime_paths(home: str | Path | None = None, require_ane: bool = False) -> RepoPaths:
    p = resolve_paths(home)
    has_weights = p.weights_npz.exists() or (p.model_dir / "config.json").exists()
    if not has_weights:
        raise FileNotFoundError(
            f"Missing exported weights at {p.weights_npz} or {p.model_dir}. "
            "Run `fast-pplx setup-weights` or `scripts/export_weights.py`."
        )
    has_tok = p.tokenizer_json.exists() or (p.model_dir / "tokenizer.json").exists()
    if not has_tok:
        raise FileNotFoundError(
            f"Missing tokenizer file: {p.tokenizer_json}. "
            "Run `fast-pplx setup-weights` or `scripts/export_weights.py`."
        )
    if require_ane:
        if not p.coreml_dylib.exists():
            raise FileNotFoundError(
                f"Missing native ANE runner library: {p.coreml_dylib}. Run `make native` in {p.home}."
            )
        if not p.ane_models_dir.exists():
            raise FileNotFoundError(f"Missing ANE models directory: {p.ane_models_dir}.")
    return p


__all__ = ["RepoPaths", "require_runtime_paths", "resolve_paths"]
