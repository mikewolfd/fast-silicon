"""Filesystem discovery for weights, tokenizer assets, ANE CoreML bundles, and native dylibs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

_PKG_NAME = "fast-pplx"
_ARTIFACT_SUBDIRS = ("weights", "tokenizer", "ane_models")


def _has_direct_artifact_subdirs(path: Path) -> bool:
    return path.is_dir() and any((path / sub).is_dir() for sub in _ARTIFACT_SUBDIRS)


def resolve_home(
    home: str | Path | None = None,
    *,
    _anchor_file: str | Path | None = None,
) -> Path:
    """Resolve `fast-pplx` home directory requiring `artifacts/` before accepting a checkout.

    Precedence:
      1. Explicit `home` argument.
      2. `FAST_PPLX_HOME` environment variable.
      3. Package checkout (`packages/fast-pplx`) ONLY if `<checkout>/artifacts` exists.
      4. `FAST_ARTIFACTS_ROOT` (`<root>/fast-pplx` or `<root>`) if it contains `artifacts/`
         or direct artifact subdirectories (`weights/`, `tokenizer/`, `ane_models/`).
      5. Sibling checkout (`../fast-pplx` alongside monorepo root or packages dir) if `artifacts/` exists.
      6. `~/.cache/fast-pplx` if `artifacts/` exists.
      7. Fallback: `$FAST_ARTIFACTS_ROOT/fast-pplx` when set, else `~/.cache/fast-pplx`.
    """
    if home is not None:
        return Path(home).expanduser().resolve()

    env_home = os.environ.get("FAST_PPLX_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    anchor = Path(_anchor_file if _anchor_file is not None else __file__).resolve()
    pkg_root = anchor.parents[2] if len(anchor.parents) >= 3 else anchor.parent

    # 3. Package checkout only if `artifacts/` actually exists there (do not stop on `native/`/`pyproject.toml`).
    if (pkg_root / "artifacts").is_dir():
        return pkg_root

    # 4. Shared FAST_ARTIFACTS_ROOT override.
    env_art_root = os.environ.get("FAST_ARTIFACTS_ROOT")
    if env_art_root:
        art_root = Path(env_art_root).expanduser().resolve()
        pkg_art_home = art_root / _PKG_NAME
        if (pkg_art_home / "artifacts").is_dir() or _has_direct_artifact_subdirs(pkg_art_home):
            return pkg_art_home
        if (art_root / "artifacts").is_dir():
            return art_root

    # 5. Sibling checkout (e.g., `<work>/fast-pplx` next to `<work>/fast-silicon/packages/fast-pplx`).
    sibling_candidates: list[Path] = []
    if len(pkg_root.parents) >= 2:
        sibling_candidates.append(pkg_root.parents[1] / _PKG_NAME)
    if len(pkg_root.parents) >= 3:
        sibling_candidates.append(pkg_root.parents[2] / _PKG_NAME)
    if pkg_root.parent != pkg_root:
        sibling_candidates.append(pkg_root.parent / _PKG_NAME)

    seen: set[Path] = {pkg_root}
    for sib in sibling_candidates:
        sib_res = sib.resolve()
        if sib_res in seen:
            continue
        seen.add(sib_res)
        if (sib_res / "artifacts").is_dir():
            return sib_res

    # 6. Cache directory if populated.
    cache_home = (Path("~/.cache") / _PKG_NAME).expanduser().resolve()
    if (cache_home / "artifacts").is_dir():
        return cache_home

    # 7. Default unpopulated target (prefer FAST_ARTIFACTS_ROOT if configured, else ~/.cache/fast-pplx).
    if env_art_root:
        return (Path(env_art_root).expanduser().resolve() / _PKG_NAME).resolve()
    return cache_home


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


def resolve_paths(
    home: str | Path | None = None,
    *,
    _anchor_file: str | Path | None = None,
) -> RepoPaths:
    explicit_or_env = home is not None or bool(os.environ.get("FAST_PPLX_HOME"))
    root = resolve_home(home, _anchor_file=_anchor_file)

    if not explicit_or_env and not (root / "artifacts").is_dir() and _has_direct_artifact_subdirs(root):
        artifacts = root
    else:
        artifacts = root / "artifacts"

    anchor = Path(_anchor_file if _anchor_file is not None else __file__).resolve()
    pkg_root = anchor.parents[2] if len(anchor.parents) >= 3 else anchor.parent
    build_dir = root / "build"
    if not explicit_or_env and (pkg_root / "build").is_dir():
        build_dir = pkg_root / "build"

    weights_dir = artifacts / "weights"
    tokenizer_dir = artifacts / "tokenizer"
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


__all__ = ["RepoPaths", "require_runtime_paths", "resolve_home", "resolve_paths"]
