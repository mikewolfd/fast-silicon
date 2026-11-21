"""Stdlib-only home, artifact directory, and native runner path resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Sequence

DEFAULT_LAYA_REPO = "convaiinnovations/laya-typed-decisions"
DEFAULT_LAYA_NMTZ_COREML_REPO = "NMTZ/laya-typed-decisions-421m-coreml-ane"
DEFAULT_KEV_ADAPTER_REPO = "jaredpalmer/kev-0.8b"
DEFAULT_KEV_BASE_REPO = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_KEV_08B_MLX_REPO = "AbstractFramework/kev-0.8b-mlx-8bit"
DEFAULT_KEV_4B_ADAPTER_REPO = "jaredpalmer/kev-4b"
DEFAULT_KEV_4B_BASE_REPO = "Qwen/Qwen3.5-4B-Base"
DEFAULT_KEV_4B_MLX_REPO = "AbstractFramework/kev-4b-mlx-4bit"
DEFAULT_MACOSX_DEPLOYMENT_TARGET = "15.0"


@dataclass(frozen=True)
class DecisionModelPaths:
    """Resolved workspace/cache locations for `fast-decision-models`."""

    home: Path
    artifacts_dir: Path
    laya_dir: Path
    kev_dir: Path
    dylib_path: Path
    kev_4b_dir: Path | None = None


def resolve_home(home: str | Path | None = None) -> Path:
    """Resolve `fast-decision-models` home directory (`explicit > env > checkout w/ artifacts > FAST_ARTIFACTS_ROOT > sibling > cache`)."""
    if home is not None:
        return Path(home).expanduser().resolve()
    env_home = os.environ.get("FAST_DECISION_MODELS_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    cand = Path(__file__).resolve().parents[2]
    if (cand / "artifacts").is_dir():
        return cand
    env_art = os.environ.get("FAST_ARTIFACTS_ROOT")
    if env_art:
        art_root = Path(env_art).expanduser().resolve()
        sub = art_root / "fast-decision-models"
        if (sub / "artifacts").is_dir() or not (art_root / "artifacts").is_dir():
            return sub
        return art_root
    if cand.parent.name == "packages" and len(cand.parents) >= 3:
        sib = cand.parent.parent.parent / "fast-decision-models"
        if (sib / "artifacts").is_dir():
            return sib.resolve()
    return (Path.home() / ".cache" / "fast-decision-models").resolve()


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
    variant: str = "0.8b",
) -> Path:
    """Resolve directory containing `kev_08b` or `kev_4b` MLX safetensors and tokenizer."""
    if kev_dir is not None:
        return Path(kev_dir).expanduser().resolve()
    norm_var = str(variant).strip().lower()
    if norm_var in ("4b", "kev-4b", "kev_4b"):
        env_4b = os.environ.get("FAST_DECISION_MODELS_KEV_4B_DIR")
        if env_4b:
            return Path(env_4b).expanduser().resolve()
        return resolve_artifacts_dir(artifacts_dir, home=home) / "kev_4b"
    env_dir = os.environ.get("FAST_DECISION_MODELS_KEV_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return resolve_artifacts_dir(artifacts_dir, home=home) / "kev_08b"


def resolve_kev_4b_dir(
    kev_dir: str | Path | None = None,
    *,
    artifacts_dir: str | Path | None = None,
    home: str | Path | None = None,
) -> Path:
    """Resolve directory containing `kev_4b` merged/quantized MLX safetensors and tokenizer."""
    return resolve_kev_dir(kev_dir, artifacts_dir=artifacts_dir, home=home, variant="4b")


def resolve_paths(
    home: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> DecisionModelPaths:
    """Return resolved `DecisionModelPaths` bundle."""
    pkg_home = resolve_home(home)
    art = resolve_artifacts_dir(artifacts_dir, home=pkg_home)
    pkg_checkout = Path(__file__).resolve().parents[2]
    build_root = (
        pkg_checkout
        if (
            home is None
            and not os.environ.get("FAST_DECISION_MODELS_HOME")
            and not (pkg_home / "build" / "liblaya_ane_runner.dylib").is_file()
            and ((pkg_checkout / "build" / "liblaya_ane_runner.dylib").is_file() or (pkg_checkout / "native").is_dir())
        )
        else pkg_home
    )
    return DecisionModelPaths(
        home=pkg_home,
        artifacts_dir=art,
        laya_dir=art / "laya_typed",
        kev_dir=art / "kev_08b",
        dylib_path=build_root / "build" / "liblaya_ane_runner.dylib",
        kev_4b_dir=art / "kev_4b",
    )


def _is_dylib_fresh(dylib: Path, deps: Sequence[Path]) -> bool:
    """Return True iff `dylib` exists and is at least as new as every existing native dependency."""
    if not dylib.is_file():
        return False
    dylib_mtime = dylib.stat().st_mtime
    for dep in deps:
        if dep.is_file() and dep.stat().st_mtime > dylib_mtime:
            return False
    return True


def ensure_laya_dylib(
    home_dir: str | Path | None = None,
    *,
    shared_include_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Locate or (re)compile `build/liblaya_ane_runner.dylib` when missing or older than native sources."""
    pkg_home = resolve_home(home_dir)
    pkg_checkout = Path(__file__).resolve().parents[2]

    if (
        home_dir is None
        and not os.environ.get("FAST_DECISION_MODELS_HOME")
        and (pkg_checkout / "native" / "laya_ane_runner.mm").is_file()
    ):
        src_root = pkg_checkout
    elif (pkg_home / "native" / "laya_ane_runner.mm").is_file():
        src_root = pkg_home
    else:
        src_root = pkg_checkout
    src_mm = src_root / "native" / "laya_ane_runner.mm"
    src_h = src_root / "native" / "laya_ane_runner.h"
    dylib = src_root / "build" / "liblaya_ane_runner.dylib"

    if not src_mm.is_file():
        if dylib.is_file() and not force_rebuild:
            return dylib
        fallback_dylib = pkg_home / "build" / "liblaya_ane_runner.dylib"
        if fallback_dylib.is_file() and not force_rebuild:
            return fallback_dylib
        raise FileNotFoundError(f"Missing {dylib} and native source {src_mm}")

    cand_inc = (
        Path(shared_include_dir).expanduser().resolve()
        if shared_include_dir is not None
        else src_root.parent / "fast-shared" / "src" / "fast_shared" / "include"
    )
    deps: list[Path] = [src_mm, src_h]
    shared_apple_h = cand_inc / "fast_shared_apple.hpp"
    if shared_apple_h.is_file():
        deps.append(shared_apple_h)

    if not force_rebuild and _is_dylib_fresh(dylib, deps):
        return dylib

    dylib.parent.mkdir(parents=True, exist_ok=True)
    include_flags = [f"-I{src_root / 'native'}"]
    if cand_inc.is_dir():
        include_flags.append(f"-I{cand_inc}")

    deploy_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", DEFAULT_MACOSX_DEPLOYMENT_TARGET)
    tmp_dylib = dylib.with_name(f".{dylib.name}.{os.getpid()}.tmp")
    try:
        subprocess.check_call(
            [
                "clang++",
                "-std=c++17",
                "-O3",
                "-arch",
                "arm64",
                f"-mmacosx-version-min={deploy_target}",
                "-fobjc-arc",
                "-dynamiclib",
                "-framework",
                "Foundation",
                "-framework",
                "CoreML",
                "-framework",
                "CoreVideo",
                *include_flags,
                str(src_mm),
                "-o",
                str(tmp_dylib),
            ]
        )
        os.replace(tmp_dylib, dylib)
    finally:
        if tmp_dylib.exists():
            try:
                tmp_dylib.unlink()
            except OSError:
                pass
    return dylib


__all__ = [
    "DEFAULT_KEV_08B_MLX_REPO",
    "DEFAULT_KEV_4B_ADAPTER_REPO",
    "DEFAULT_KEV_4B_BASE_REPO",
    "DEFAULT_KEV_4B_MLX_REPO",
    "DEFAULT_KEV_ADAPTER_REPO",
    "DEFAULT_KEV_BASE_REPO",
    "DEFAULT_LAYA_NMTZ_COREML_REPO",
    "DEFAULT_LAYA_REPO",
    "DEFAULT_MACOSX_DEPLOYMENT_TARGET",
    "DecisionModelPaths",
    "ensure_laya_dylib",
    "resolve_artifacts_dir",
    "resolve_home",
    "resolve_kev_4b_dir",
    "resolve_kev_dir",
    "resolve_laya_dir",
    "resolve_paths",
]
