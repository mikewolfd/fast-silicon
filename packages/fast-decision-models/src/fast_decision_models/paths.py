"""Stdlib-only home, artifact directory, and native runner path resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

DEFAULT_LAYA_REPO = "convaiinnovations/laya-typed-decisions"
DEFAULT_KEV_ADAPTER_REPO = "jaredpalmer/kev-0.8b"
DEFAULT_KEV_BASE_REPO = "Qwen/Qwen3.5-0.8B-Base"


@dataclass(frozen=True)
class DecisionModelPaths:
    """Resolved workspace/cache locations for `fast-decision-models`."""

    home: Path
    artifacts_dir: Path
    laya_dir: Path
    kev_dir: Path
    dylib_path: Path


def resolve_home(home: str | Path | None = None) -> Path:
    """Resolve `fast-decision-models` home directory (`explicit > env > checkout > ~/.cache`)."""
    if home is not None:
        return Path(home).expanduser().resolve()
    env_home = os.environ.get("FAST_DECISION_MODELS_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    cand = Path(__file__).resolve().parents[2]
    if any((cand / m).exists() for m in ("native", "artifacts", "pyproject.toml")):
        return cand
    return (Path("~/.cache") / "fast-decision-models").expanduser().resolve()


def resolve_artifacts_dir(
    artifacts_dir: str | Path | None = None,
    *,
    home: str | Path | None = None,
) -> Path:
    """Resolve staged weights/bundles directory (`explicit > env > <home>/artifacts`)."""
    if artifacts_dir is not None:
        return Path(artifacts_dir).expanduser().resolve()
    env_dir = os.environ.get("FAST_DECISION_MODELS_ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    pkg_home = resolve_home(home)
    return pkg_home / "artifacts"


def resolve_laya_dir(
    laya_dir: str | Path | None = None,
    *,
    artifacts_dir: str | Path | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve directory containing `laya_typed` CoreML ANE bundles and tokenizer."""
    if laya_dir is not None:
        return Path(laya_dir).expanduser().resolve()
    env_dir = os.environ.get("FAST_DECISION_MODELS_LAYA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return resolve_artifacts_dir(artifacts_dir, home=home) / "laya_typed"


def resolve_kev_dir(
    kev_dir: str | Path | None = None,
    *,
    artifacts_dir: str | Path | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve directory containing `kev_08b` merged BF16/W8 MLX safetensors and tokenizer."""
    if kev_dir is not None:
        return Path(kev_dir).expanduser().resolve()
    env_dir = os.environ.get("FAST_DECISION_MODELS_KEV_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return resolve_artifacts_dir(artifacts_dir, home=home) / "kev_08b"


def resolve_paths(
    home: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> DecisionModelPaths:
    """Return resolved `DecisionModelPaths` bundle."""
    pkg_home = resolve_home(home)
    art = resolve_artifacts_dir(artifacts_dir, home=pkg_home)
    return DecisionModelPaths(
        home=pkg_home,
        artifacts_dir=art,
        laya_dir=art / "laya_typed",
        kev_dir=art / "kev_08b",
        dylib_path=pkg_home / "build" / "liblaya_ane_runner.dylib",
    )


def ensure_laya_dylib(
    home_dir: str | Path | None = None,
    *,
    shared_include_dir: str | Path | None = None,
) -> Path:
    """Locate or compile `build/liblaya_ane_runner.dylib` from `native/laya_ane_runner.mm`."""
    pkg_home = resolve_home(home_dir)
    dylib = pkg_home / "build" / "liblaya_ane_runner.dylib"
    src_mm = pkg_home / "native" / "laya_ane_runner.mm"
    if dylib.is_file():
        return dylib
    if not src_mm.is_file():
        pkg_checkout = Path(__file__).resolve().parents[2]
        fallback_dylib = pkg_checkout / "build" / "liblaya_ane_runner.dylib"
        if fallback_dylib.is_file():
            return fallback_dylib
        fallback_mm = pkg_checkout / "native" / "laya_ane_runner.mm"
        if fallback_mm.is_file():
            pkg_home = pkg_checkout
            dylib = fallback_dylib
            src_mm = fallback_mm
        else:
            raise FileNotFoundError(f"Missing {dylib} and native source {src_mm}")

    dylib.parent.mkdir(parents=True, exist_ok=True)
    include_flags = [f"-I{pkg_home / 'native'}"]
    cand_inc = (
        Path(shared_include_dir).expanduser().resolve()
        if shared_include_dir is not None
        else pkg_home.parent / "fast-shared" / "src" / "fast_shared" / "include"
    )
    if cand_inc.is_dir():
        include_flags.append(f"-I{cand_inc}")

    subprocess.check_call(
        [
            "clang++",
            "-std=c++17",
            "-O3",
            "-fobjc-arc",
            "-dynamiclib",
            "-framework",
            "Foundation",
            "-framework",
            "CoreML",
            *include_flags,
            str(src_mm),
            "-o",
            str(dylib),
        ]
    )
    return dylib


__all__ = [
    "DEFAULT_KEV_ADAPTER_REPO",
    "DEFAULT_KEV_BASE_REPO",
    "DEFAULT_LAYA_REPO",
    "DecisionModelPaths",
    "ensure_laya_dylib",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_kev_dir",
    "resolve_laya_dir",
    "resolve_paths",
]
