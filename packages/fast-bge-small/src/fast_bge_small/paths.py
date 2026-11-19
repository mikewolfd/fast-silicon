"""Filesystem discovery for weights, tokenizer assets, ANE CoreML bundles, and native dylibs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from fast_shared.paths import resolve_home_dir

_PACKAGE_CACHE_NAME = "fast-bge-small"
_HOME_ENV_VAR = "FAST_BGE_SMALL_HOME"


@dataclass(frozen=True)
class RepoPaths:
    home: Path
    weights_npz: Path
    tokenizer_dir: Path
    tokenizer_json: Path
    vocab_txt: Path
    ane_models_dir: Path
    build_dir: Path
    ane_dylib: Path
    cpu_dylib: Path
    ane_cli: Path


def resolve_home(home: str | Path | None = None, *, anchor_file: str | Path | None = None) -> Path:
    """Resolve the runtime data root (explicit > env > checkout with artifacts/ > FAST_ARTIFACTS_ROOT > sibling > ~/.cache)."""
    return resolve_home_dir(
        anchor_file if anchor_file is not None else __file__,
        env_var=_HOME_ENV_VAR,
        cache_name=_PACKAGE_CACHE_NAME,
        home=home,
        marker_dirs=("artifacts",),
    )


def resolve_paths(home: str | Path | None = None) -> RepoPaths:
    root = resolve_home(home)
    checkout_root = Path(__file__).resolve().parent.parent.parent

    artifacts = root / "artifacts"
    build_dir = root / "build"
    if (
        home is None
        and not os.environ.get(_HOME_ENV_VAR)
        and not build_dir.exists()
        and (checkout_root / "build").exists()
    ):
        build_dir = checkout_root / "build"

    return RepoPaths(
        home=root,
        weights_npz=artifacts / "weights" / "bge_small_f16.npz",
        tokenizer_dir=artifacts / "tokenizer",
        tokenizer_json=artifacts / "tokenizer" / "tokenizer.json",
        vocab_txt=artifacts / "tokenizer" / "vocab.txt",
        ane_models_dir=artifacts / "ane_models",
        build_dir=build_dir,
        ane_dylib=build_dir / "libbge_ane_runner.dylib",
        cpu_dylib=build_dir / "libbge_cpu_query.dylib",
        ane_cli=build_dir / "bge-ane-cli",
    )


def require_runtime_paths(home: str | Path | None = None, require_ane: bool = False) -> RepoPaths:
    p = resolve_paths(home)
    if not p.weights_npz.exists():
        raise FileNotFoundError(
            f"Missing weights file: {p.weights_npz}. Run `bge-small setup-weights` or `scripts/export_weights.py`."
        )
    if not p.tokenizer_json.exists():
        raise FileNotFoundError(
            f"Missing tokenizer file: {p.tokenizer_json}. Run `bge-small setup-weights` or `scripts/export_weights.py`."
        )
    if require_ane:
        if not p.ane_dylib.exists():
            raise FileNotFoundError(f"Missing native ANE runner library: {p.ane_dylib}. Run `make native` in {p.home}.")
        if not p.ane_models_dir.exists():
            raise FileNotFoundError(f"Missing ANE models directory: {p.ane_models_dir}.")
    return p
