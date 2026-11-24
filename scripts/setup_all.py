#!/usr/bin/env python3
"""One-command workspace bootstrap for `fast-silicon`.

Builds native libraries, downloads required HuggingFace checkpoints/tokenizers,
exports stripped/packed weight files, builds vocabulary remaps, and compiles
CoreML Neural Engine (`.mlmodelc`) bundles across all workspace packages so tests,
CLIs, and benchmarks run end-to-end with zero skipped tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ORDER: tuple[str, ...] = (
    "bge-small",
    "embeddinggemma-2",
    "pplx",
    "decision-models",
    "gemma4",
    "gpt-oss",
    "qwen3",
)

_PACKAGE_ALIASES: dict[str, str] = {
    "fast-bge-small": "bge-small",
    "bge-small": "bge-small",
    "fast-embeddinggemma-2": "embeddinggemma-2",
    "embeddinggemma-2": "embeddinggemma-2",
    "emb2": "embeddinggemma-2",
    "fast-pplx": "pplx",
    "pplx": "pplx",
    "fast-decision-models": "decision-models",
    "decision-models": "decision-models",
    "fast-gemma4": "gemma4",
    "gemma4": "gemma4",
    "fast-gpt-oss": "gpt-oss",
    "gpt-oss": "gpt-oss",
    "fast-qwen3": "qwen3",
    "qwen3": "qwen3",
}

_BGE_REQUIRED_BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 24),
    (1, 64),
    (1, 128),
    (1, 192),
    (8, 256),
    (1, 320),
    (1, 384),
    (1, 448),
    (1, 512),
)
_BGE_BUCKETS: tuple[tuple[int, int], ...] = _BGE_REQUIRED_BUCKETS + ((1, 256),)
_EMB2_ANE_SEQS: tuple[int, ...] = (32, 64, 128, 192, 256, 320, 384, 448, 512)
_EMB2_CPU_SEQS: tuple[int, ...] = (256, 384, 512)

_BGE_HF_REPO = "mlx-community/bge-small-en-v1.5-bf16"
_EMB2_HF_REPO = "mlx-community/embeddinggemma-2-bf16"
_PPLX_HF_REPO = "perplexity-ai/pplx-embed-v2-context-0.6b"
_GEMMA4_HF_REPO = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
_GEMMA4_TEST_HF_REPOS: tuple[str, ...] = (
    "mlx-community/gemma-4-e2b-it-bf16",
    "mlx-community/gemma-4-E2B-it-qat-4bit",
)
_GPT_OSS_HF_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q4"
_GPT_OSS_PREBUILT_HF_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q8"
_QWEN3_HF_REPO = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
_LAYA_HF_REPO = "convaiinnovations/laya-typed-decisions"
_KEV_ADAPTER_HF_REPO = "jaredpalmer/kev-0.8b"
_KEV_BASE_HF_REPO = "Qwen/Qwen3.5-0.8B-Base"

VETTED_PREBUILT_HF_REPOS: dict[str, tuple[str, ...]] = {
    "bge-small": (
        "BAAI/bge-small-en-v1.5",
        "sidekickd/bge-small-en-v1.5-coreml",
    ),
    "embeddinggemma-2": (
        "OpenInfinity/embeddinggemma-2-coreml-int8",
    ),
    "pplx": (
        "perplexity-ai/pplx-embed-v2-context-0.6b",
    ),
    "decision-models": (
        "NMTZ/laya-typed-decisions-421m-coreml-ane",
        "AbstractFramework/kev-0.8b-mlx-8bit",
        "AbstractFramework/kev-4b-mlx-4bit",
    ),
    "gemma4": (
        "google/gemma-4-E2B-it-qat-q4_0-unquantized",
        "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
    ),
    "gpt-oss": (
        _GPT_OSS_PREBUILT_HF_REPO,
    ),
    "qwen3": (
        "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen3.5-0.8B-4bit",
    ),
}

_PREBUILT_APPROX_GB: dict[str, float] = {
    "bge-small": 0.35,
    "embeddinggemma-2": 0.29,
    "pplx": 2.38,
    "decision-models": 4.90,
    "gemma4": 10.40,
    "gpt-oss": 12.10,
    "qwen3": 3.10,
}


def resolve_prebuilt_hf_repos(package: str) -> tuple[str, ...]:
    """Return vetted zero-compile Hugging Face repositories for ``package``."""
    key = _PACKAGE_ALIASES.get(package.strip().lower())
    if key is None or key not in VETTED_PREBUILT_HF_REPOS:
        raise ValueError(f"Unknown package {package!r}; choose from: {', '.join(PACKAGE_ORDER)}")
    return VETTED_PREBUILT_HF_REPOS[key]


def pull_prebuilt_package(
    package: str,
    *,
    local_files_only: bool = False,
    downloader: Callable[..., str | Path] | None = None,
) -> dict[str, Path]:
    """Auto-pull vetted zero-compile prebuilt HF snapshots for ``package``."""
    repos = resolve_prebuilt_hf_repos(package)
    if downloader is None:
        from huggingface_hub import snapshot_download

        dl_fn: Callable[..., str | Path] = snapshot_download
    else:
        dl_fn = downloader
    return {
        repo_id: Path(dl_fn(repo_id=repo_id, local_files_only=local_files_only))
        for repo_id in repos
    }


@dataclass(frozen=True)
class StepAction:
    kind: str  # "native" | "hf_snapshot" | "subprocess"
    label: str
    needed: bool
    reason: str
    repo_id: str | None = None
    command: tuple[str, ...] = ()
    requires_coreml: bool = False


@dataclass(frozen=True)
class PackagePlan:
    package: str
    target_home: Path
    approx_size_gb: float
    hf_repos: tuple[str, ...]
    actions: tuple[StepAction, ...]

    @property
    def has_pending(self) -> bool:
        return any(a.needed for a in self.actions)


def _is_hf_snapshot_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return False
    try:
        p = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception:
        return False
    return p.is_dir()


def _resolve_target_home(
    pkg_key: str,
    *,
    pkg_dir_name: str,
    target_mode: str,
    artifacts_root: Path | None,
    paths_module: str,
) -> Path:
    if artifacts_root is not None:
        return (artifacts_root.expanduser().resolve() / pkg_dir_name).resolve()
    if target_mode == "in-repo":
        return (REPO_ROOT / "packages" / pkg_dir_name).resolve()
    if target_mode == "cache":
        return (Path.home() / ".cache" / pkg_dir_name).resolve()
    if target_mode != "auto":
        raise ValueError(f"Unsupported target mode {target_mode!r}; expected auto|in-repo|cache")

    mod = importlib.import_module(paths_module)
    if pkg_key in ("bge-small", "embeddinggemma-2", "pplx", "decision-models"):
        return Path(mod.resolve_home()).resolve()
    return Path(mod.resolve_paths().home).resolve()


def parse_packages_arg(raw: str, *, skip_gpt_oss: bool = False) -> tuple[str, ...]:
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens or "all" in tokens:
        selected = list(PACKAGE_ORDER)
    else:
        selected = []
        for tok in tokens:
            if tok not in _PACKAGE_ALIASES:
                valid = ", ".join(PACKAGE_ORDER)
                raise ValueError(f"Unknown package {tok!r}; choose from: all, {valid}")
            canonical = _PACKAGE_ALIASES[tok]
            if canonical not in selected:
                selected.append(canonical)
    if skip_gpt_oss:
        selected = [p for p in selected if p != "gpt-oss"]
    return tuple(selected)


def _plan_bge_small(
    home: Path,
    *,
    force: bool,
    skip_ane: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    pkg_dir = REPO_ROOT / "packages" / "fast-bge-small"
    art = home / "artifacts"
    weights_ok = (art / "weights" / "bge_small_f16.npz").is_file() and (art / "tokenizer" / "tokenizer.json").is_file()
    ane_dir = art / "ane_models"
    ane_ok = all((ane_dir / f"bge_b{b}_s{s}_fp16.mlmodelc").is_dir() for b, s in _BGE_REQUIRED_BUCKETS)
    actions = [
        StepAction(
            kind="subprocess",
            label="bge-small: export weights + tokenizer",
            needed=force or not weights_ok,
            reason="forced" if force else ("present" if weights_ok else "missing bge_small_f16.npz/tokenizer"),
            repo_id=_BGE_HF_REPO,
            command=(python_exe, str(pkg_dir / "scripts" / "export_weights.py"), "--home", str(home)),
        )
    ]
    if not skip_ane:
        actions.append(
            StepAction(
                kind="subprocess",
                label="bge-small: compile ANE buckets (B=1/8, S=24..512 + S=256)",
                needed=force or not ane_ok,
                reason="forced" if force else ("present" if ane_ok else "missing .mlmodelc bucket(s)"),
                command=(
                    python_exe,
                    str(pkg_dir / "scripts" / "build_ane_models.py"),
                    "--home",
                    str(home),
                    "--include-s256-query",
                ),
                requires_coreml=True,
            )
        )
    _ = hf_checker
    return PackagePlan(
        package="bge-small",
        target_home=home,
        approx_size_gb=0.15,
        hf_repos=(_BGE_HF_REPO,),
        actions=tuple(actions),
    )


def _plan_embeddinggemma_2(
    home: Path,
    *,
    force: bool,
    skip_ane: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    pkg_dir = REPO_ROOT / "packages" / "fast-embeddinggemma-2"
    art = home / "artifacts"
    weights_ok = (art / "weights" / "embeddinggemma_2_f16.npz").is_file() and (art / "tokenizer" / "tokenizer.json").is_file()
    ane_dir = art / "ane_models"
    ane_ok = all((ane_dir / f"emb2_pk_s{s}_w8.mlmodelc").is_dir() for s in _EMB2_ANE_SEQS) and all(
        (ane_dir / f"emb2_cpu_s{s}_mix_w8.mlmodelc").is_dir() for s in _EMB2_CPU_SEQS
    )
    actions = [
        StepAction(
            kind="subprocess",
            label="embeddinggemma-2: export weights + tokenizer",
            needed=force or not weights_ok,
            reason="forced" if force else ("present" if weights_ok else "missing embeddinggemma_2_f16.npz/tokenizer"),
            repo_id=_EMB2_HF_REPO,
            command=(python_exe, str(pkg_dir / "scripts" / "export_weights.py"), "--home", str(home)),
        )
    ]
    if not skip_ane:
        actions.append(
            StepAction(
                kind="subprocess",
                label="embeddinggemma-2: compile ANE + CPU-SME CoreML bundles",
                needed=force or not ane_ok,
                reason="forced" if force else ("present" if ane_ok else "missing .mlmodelc bucket(s)"),
                command=(python_exe, str(pkg_dir / "scripts" / "build_ane_models.py"), "--home", str(home), "--mode", "all"),
                requires_coreml=True,
            )
        )
    _ = hf_checker
    return PackagePlan(
        package="embeddinggemma-2",
        target_home=home,
        approx_size_gb=0.65,
        hf_repos=(_EMB2_HF_REPO,),
        actions=tuple(actions),
    )


def _plan_pplx(
    home: Path,
    *,
    force: bool,
    skip_ane: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    pkg_dir = REPO_ROOT / "packages" / "fast-pplx"
    art = home / "artifacts"
    hf_cached = hf_checker(_PPLX_HF_REPO)
    weights_ok = (
        (art / "weights" / "pplx_0_6b_f16.npz").is_file()
        and (art / "weights" / "embed_tokens.safetensors").is_file()
        and (art / "tokenizer" / "tokenizer.json").is_file()
    )
    manifest_p = art / "ane_models" / "manifest.json"
    ane_ok = manifest_p.is_file() and any((art / "ane_models").glob("*.mlmodelc"))
    actions = [
        StepAction(
            kind="hf_snapshot",
            label=f"pplx: cache HF snapshot ({_PPLX_HF_REPO})",
            needed=force or (not weights_ok and not hf_cached),
            reason=(
                "forced"
                if force
                else ("cached" if hf_cached else ("exported weights present" if weights_ok else "missing HF cache"))
            ),
            repo_id=_PPLX_HF_REPO,
        ),
        StepAction(
            kind="subprocess",
            label="pplx: export FP16 weights + mmap embed_tokens",
            needed=force or not weights_ok,
            reason="forced" if force else ("present" if weights_ok else "missing pplx_0_6b_f16.npz/embed_tokens"),
            repo_id=_PPLX_HF_REPO,
            command=(python_exe, str(pkg_dir / "scripts" / "export_weights.py"), "--home", str(home)),
        ),
    ]
    if not skip_ane:
        actions.append(
            StepAction(
                kind="subprocess",
                label="pplx: compile CoreML ANE w8a16 stage bundles",
                needed=force or not ane_ok,
                reason="forced" if force else ("present" if ane_ok else "missing ane_models/manifest.json"),
                command=(
                    python_exe,
                    str(pkg_dir / "scripts" / "build_ane_models.py"),
                    "--home",
                    str(home),
                    "--precision",
                    "w8a16",
                ),
                requires_coreml=True,
            )
        )
    return PackagePlan(
        package="pplx",
        target_home=home,
        approx_size_gb=1.8,
        hf_repos=(_PPLX_HF_REPO,),
        actions=tuple(actions),
    )


def _plan_decision_models(
    home: Path,
    *,
    force: bool,
    skip_ane: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    pkg_dir = REPO_ROOT / "packages" / "fast-decision-models"
    art = home / "artifacts"
    laya_dir = art / "laya_typed"
    laya_ok = (
        (laya_dir / "laya_typed_s256_w8.mlmodelc").is_dir()
        and (laya_dir / "laya_typed_s512_w8.mlmodelc").is_dir()
        and (laya_dir / "laya_typed_emb_f16.npz").is_file()
        and (laya_dir / "tokenizer" / "tokenizer.json").is_file()
    )
    kev_dir = art / "kev_08b"
    kev_ok = (
        ((kev_dir / "model_w8.safetensors").is_file() or (kev_dir / "model.safetensors").is_file())
        and (kev_dir / "tokenizer.json").is_file()
    )
    build_script = str(pkg_dir / "scripts" / "build_decision_models.py")
    actions: list[StepAction] = []
    if not skip_ane:
        actions.append(
            StepAction(
                kind="subprocess",
                label="decision-models: download Laya-Typed & compile ANE s256/s512 W8 bundles",
                needed=force or not laya_ok,
                reason="forced" if force else ("present" if laya_ok else "missing laya_typed bundles"),
                repo_id=_LAYA_HF_REPO,
                command=(
                    python_exe,
                    build_script,
                    "--home",
                    str(pkg_dir),
                    "--artifacts-dir",
                    str(art),
                    "--build-laya-ane",
                ),
                requires_coreml=True,
            )
        )
    actions.append(
        StepAction(
            kind="subprocess",
            label="decision-models: download Kev-0.8B adapter + Qwen3.5-0.8B base & merge BF16/W8",
            needed=force or not kev_ok,
            reason="forced" if force else ("present" if kev_ok else "missing kev_08b safetensors"),
            repo_id=f"{_KEV_ADAPTER_HF_REPO} + {_KEV_BASE_HF_REPO}",
            command=(
                python_exe,
                build_script,
                "--home",
                str(pkg_dir),
                "--artifacts-dir",
                str(art),
                "--build-kev-mlx",
            ),
        )
    )
    _ = hf_checker
    return PackagePlan(
        package="decision-models",
        target_home=home,
        approx_size_gb=3.4,
        hf_repos=(_LAYA_HF_REPO, _KEV_ADAPTER_HF_REPO, _KEV_BASE_HF_REPO),
        actions=tuple(actions),
    )


def _plan_gemma4(
    home: Path,
    *,
    force: bool,
    skip_ane: bool,
    skip_test_extras: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    pkg_dir = REPO_ROOT / "packages" / "fast-gemma4"
    art = home / "artifacts"
    hf_4bit = hf_checker(_GEMMA4_HF_REPO)
    vocab_ok = (art / "vocab" / "vocab_remap_150786.npz").is_file() and (art / "tokenizer" / "tokenizer.json").is_file()
    early14_ok = (art / "weights" / "gemma4_early14_f16.npz").is_file() and (art / "tokenizer" / "tokenizer.json").is_file()
    ane_ok = (
        (art / "ane_models" / "gemma4_early14_prefix_S256_fp16.mlmodelc").is_dir()
        and (art / "ane_models" / "gemma4_early14_prefix_S256_w8.mlmodelc").is_dir()
    )
    actions: list[StepAction] = [
        StepAction(
            kind="hf_snapshot",
            label=f"gemma4: download E2B-4bit runtime ({_GEMMA4_HF_REPO})",
            needed=force or not hf_4bit,
            reason="forced" if force else ("cached" if hf_4bit else "missing HF cache"),
            repo_id=_GEMMA4_HF_REPO,
        ),
        StepAction(
            kind="subprocess",
            label="gemma4: setup-weights (tokenizer + vocab_remap_150786)",
            needed=force or not vocab_ok,
            reason="forced" if force else ("present" if vocab_ok else "missing vocab_remap_150786.npz/tokenizer"),
            command=(
                python_exe,
                "-m",
                "fast_gemma4.cli",
                "--home",
                str(home),
                "setup-weights",
                "--skip-early14",
            ),
        ),
        StepAction(
            kind="subprocess",
            label="gemma4: export Early14 FP16 weights",
            needed=force or (not early14_ok and not ane_ok and not skip_ane),
            reason=(
                "forced"
                if force
                else (
                    "present"
                    if early14_ok
                    else (
                        "already compiled into ANE bundles"
                        if ane_ok
                        else ("skipped with --skip-ane" if skip_ane else "missing gemma4_early14_f16.npz")
                    )
                )
            ),
            command=(
                python_exe,
                str(pkg_dir / "scripts" / "export_early14_f16.py"),
                "--home",
                str(home),
            ),
        ),
    ]
    if not skip_ane:
        actions.append(
            StepAction(
                kind="subprocess",
                label="gemma4: compile ANE Early14 prefix bundles (fp16 + w8)",
                needed=force or not ane_ok,
                reason="forced" if force else ("present" if ane_ok else "missing S256 fp16/w8 .mlmodelc"),
                command=(
                    python_exe,
                    str(pkg_dir / "scripts" / "build_ane_models.py"),
                    "--home",
                    str(home),
                    "--variant",
                    "both",
                ),
                requires_coreml=True,
            )
        )
    hf_repos = [_GEMMA4_HF_REPO]
    if not skip_test_extras:
        for repo in _GEMMA4_TEST_HF_REPOS:
            hf_repos.append(repo)
            cached = hf_checker(repo)
            actions.append(
                StepAction(
                    kind="hf_snapshot",
                    label=f"gemma4: cache full-test snapshot ({repo})",
                    needed=force or not cached,
                    reason="forced" if force else ("cached" if cached else "missing HF cache"),
                    repo_id=repo,
                )
            )
    approx_gb = 4.2 if skip_test_extras else 17.0
    return PackagePlan(
        package="gemma4",
        target_home=home,
        approx_size_gb=approx_gb,
        hf_repos=tuple(hf_repos),
        actions=tuple(actions),
    )


def _plan_gpt_oss(
    home: Path,
    *,
    force: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    art = home / "artifacts"
    hf_cached = hf_checker(_GPT_OSS_PREBUILT_HF_REPO) or hf_checker(_GPT_OSS_HF_REPO)
    local_hf_ok = (art / "hf_model" / "config.json").is_file() and any((art / "hf_model").glob("*.safetensors"))
    vocab_ok = (art / "vocab" / "o200k_remap_131072.npz").is_file() and (art / "tokenizer" / "tokenizer.json").is_file()
    actions = [
        StepAction(
            kind="subprocess",
            label="gpt-oss: download 20B-MXFP4-Q8/Q4 + build o200k_remap_131072",
            needed=force or not (vocab_ok and (hf_cached or local_hf_ok)),
            reason=(
                "forced"
                if force
                else ("present" if (vocab_ok and (hf_cached or local_hf_ok)) else "missing HF checkpoint or o200k remap")
            ),
            repo_id=_GPT_OSS_PREBUILT_HF_REPO,
            command=(
                python_exe,
                "-m",
                "fast_gpt_oss.cli",
                "setup-weights",
                "--home",
                str(home),
                "--no-local-files-only",
            ),
        )
    ]
    return PackagePlan(
        package="gpt-oss",
        target_home=home,
        approx_size_gb=12.0,
        hf_repos=(_GPT_OSS_PREBUILT_HF_REPO, _GPT_OSS_HF_REPO),
        actions=tuple(actions),
    )


def _is_qwen3_snapshot_cached(home: Path) -> bool:
    try:
        from fast_qwen3.paths import resolve_paths

        resolve_paths(home).resolve_model_dir(repo_id=_QWEN3_HF_REPO, local_files_only=True)
        return True
    except Exception:
        return False


def _plan_qwen3(
    home: Path,
    *,
    force: bool,
    hf_checker: Callable[[str], bool],
    python_exe: str,
) -> PackagePlan:
    art = home / "artifacts"
    hf_cached = bool(hf_checker(_QWEN3_HF_REPO)) or (
        hf_checker is _is_hf_snapshot_cached and _is_qwen3_snapshot_cached(home)
    )
    tok_ok = (art / "tokenizer" / "tokenizer.json").is_file()
    actions = [
        StepAction(
            kind="subprocess",
            label="qwen3: download Qwen3.5-4B-OptiQ-4bit + stage tokenizer",
            needed=force or not (tok_ok and hf_cached),
            reason="forced" if force else ("present" if (tok_ok and hf_cached) else "missing Qwen3.5-4B-OptiQ-4bit snapshot"),
            repo_id=_QWEN3_HF_REPO,
            command=(
                python_exe,
                "-m",
                "fast_qwen3.cli",
                "setup-weights",
                "--home",
                str(home),
                "--allow-download",
            ),
        )
    ]
    return PackagePlan(
        package="qwen3",
        target_home=home,
        approx_size_gb=2.6,
        hf_repos=(_QWEN3_HF_REPO,),
        actions=tuple(actions),
    )


def _plan_prebuilt_package(
    pkg: str,
    home: Path,
    *,
    force: bool,
    hf_checker: Callable[[str], bool],
) -> PackagePlan:
    repos = VETTED_PREBUILT_HF_REPOS[pkg]
    actions = tuple(
        StepAction(
            kind="hf_snapshot",
            label=f"{pkg}: pull prebuilt zero-compile HF snapshot ({repo_id})",
            needed=force or not hf_checker(repo_id),
            reason="forced" if force else ("cached" if hf_checker(repo_id) else "missing HF cache"),
            repo_id=repo_id,
            requires_coreml=False,
        )
        for repo_id in repos
    )
    return PackagePlan(
        package=pkg,
        target_home=home,
        approx_size_gb=_PREBUILT_APPROX_GB.get(pkg, 1.0),
        hf_repos=repos,
        actions=actions,
    )


def build_setup_plan(
    *,
    packages: Sequence[str],
    target_mode: str = "auto",
    artifacts_root: Path | None = None,
    force: bool = False,
    prebuilt: bool = False,
    skip_native: bool = False,
    skip_ane: bool = False,
    skip_test_extras: bool = False,
    hf_checker: Callable[[str], bool] = _is_hf_snapshot_cached,
    python_exe: str = sys.executable,
) -> tuple[list[StepAction], list[PackagePlan]]:
    """Build deterministic idempotent setup actions across selected workspace packages."""
    native_actions: list[StepAction] = []
    if not skip_native:
        native_actions.append(
            StepAction(
                kind="native",
                label="workspace: build native dylibs + CLIs (`make native` + `fast-bge-small cpu-query`)",
                needed=True,
                reason="incremental make",
                command=("make", "-C", str(REPO_ROOT), f"PYTHON={python_exe}", "native"),
            )
        )
        if "bge-small" in packages:
            native_actions.append(
                StepAction(
                    kind="native",
                    label="bge-small: build vectorized CPU query dylib (`make cpu-query`)",
                    needed=True,
                    reason="incremental make",
                    command=(
                        "make",
                        "-C",
                        str(REPO_ROOT / "packages" / "fast-bge-small"),
                        f"PYTHON={python_exe}",
                        "cpu-query",
                    ),
                )
            )

    pkg_specs: dict[str, tuple[str, str]] = {
        "bge-small": ("fast-bge-small", "fast_bge_small.paths"),
        "embeddinggemma-2": ("fast-embeddinggemma-2", "fast_embeddinggemma_2.paths"),
        "pplx": ("fast-pplx", "fast_pplx.paths"),
        "decision-models": ("fast-decision-models", "fast_decision_models.paths"),
        "gemma4": ("fast-gemma4", "fast_gemma4.paths"),
        "gpt-oss": ("fast-gpt-oss", "fast_gpt_oss.paths"),
        "qwen3": ("fast-qwen3", "fast_qwen3.paths"),
    }

    plans: list[PackagePlan] = []
    for pkg in packages:
        dir_name, pmod = pkg_specs[pkg]
        home = _resolve_target_home(
            pkg,
            pkg_dir_name=dir_name,
            target_mode=target_mode,
            artifacts_root=artifacts_root,
            paths_module=pmod,
        )
        if prebuilt:
            plans.append(_plan_prebuilt_package(pkg, home, force=force, hf_checker=hf_checker))
        elif pkg == "bge-small":
            plans.append(_plan_bge_small(home, force=force, skip_ane=skip_ane, hf_checker=hf_checker, python_exe=python_exe))
        elif pkg == "embeddinggemma-2":
            plans.append(
                _plan_embeddinggemma_2(home, force=force, skip_ane=skip_ane, hf_checker=hf_checker, python_exe=python_exe)
            )
        elif pkg == "pplx":
            plans.append(_plan_pplx(home, force=force, skip_ane=skip_ane, hf_checker=hf_checker, python_exe=python_exe))
        elif pkg == "decision-models":
            plans.append(
                _plan_decision_models(home, force=force, skip_ane=skip_ane, hf_checker=hf_checker, python_exe=python_exe)
            )
        elif pkg == "gemma4":
            plans.append(
                _plan_gemma4(
                    home,
                    force=force,
                    skip_ane=skip_ane,
                    skip_test_extras=skip_test_extras,
                    hf_checker=hf_checker,
                    python_exe=python_exe,
                )
            )
        elif pkg == "gpt-oss":
            plans.append(_plan_gpt_oss(home, force=force, hf_checker=hf_checker, python_exe=python_exe))
        elif pkg == "qwen3":
            plans.append(_plan_qwen3(home, force=force, hf_checker=hf_checker, python_exe=python_exe))

    return native_actions, plans


def _preflight_dependencies(plans: Sequence[PackagePlan]) -> None:
    if importlib.util.find_spec("huggingface_hub") is None:
        raise RuntimeError("Missing `huggingface_hub`. Run `uv sync --extra dev --extra extract --extra build` first.")
    needs_coreml = any(a.needed and a.requires_coreml for p in plans for a in p.actions)
    if needs_coreml and importlib.util.find_spec("coremltools") is None:
        raise RuntimeError(
            "Missing `coremltools` needed to compile CoreML ANE `.mlmodelc` bundles. "
            "Run `uv sync --extra dev --extra extract --extra build` (or re-run with `--skip-ane`)."
        )


def _format_plan_summary(native_actions: Sequence[StepAction], plans: Sequence[PackagePlan]) -> str:
    lines = ["=== fast-silicon setup plan ==="]
    total_pending_gb = sum(p.approx_size_gb for p in plans if p.has_pending)
    for act in native_actions:
        lines.append(f"  [native] {act.label}")
    for plan in plans:
        state = "PENDING" if plan.has_pending else "READY"
        lines.append(f"  [{state:7s}] {plan.package} (~{plan.approx_size_gb:.2f} GB) -> {plan.target_home}")
        for act in plan.actions:
            tag = "DO" if act.needed else "OK"
            lines.append(f"      [{tag}] {act.label} ({act.reason})")
    lines.append(f"Estimated network/disk needed for pending packages: ~{total_pending_gb:.2f} GB")
    return "\n".join(lines)


def _execute_plan(
    native_actions: Sequence[StepAction],
    plans: Sequence[PackagePlan],
    *,
    env: dict[str, str],
) -> None:
    _preflight_dependencies(plans)

    for act in native_actions:
        print(f"==> {act.label}", flush=True)
        subprocess.check_call(list(act.command), cwd=str(REPO_ROOT), env=env)

    from huggingface_hub import snapshot_download

    for plan in plans:
        if not plan.has_pending:
            print(f"==> [READY] {plan.package} ({plan.target_home})", flush=True)
            continue
        # Pre-create `artifacts/` so subsequent `resolve_paths()` consistently sees `<home>/artifacts`.
        (plan.target_home / "artifacts").mkdir(parents=True, exist_ok=True)
        for act in plan.actions:
            if not act.needed:
                print(f"    [skip] {act.label} ({act.reason})", flush=True)
                continue
            print(f"==> {act.label}", flush=True)
            if act.kind == "hf_snapshot":
                assert act.repo_id is not None
                path = snapshot_download(repo_id=act.repo_id, local_files_only=False)
                print(f"    cached -> {path}", flush=True)
            elif act.kind == "subprocess":
                subprocess.check_call(list(act.command), cwd=str(REPO_ROOT), env=env)
            else:
                raise ValueError(f"Unknown action kind {act.kind!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command download, weight export, native build, and CoreML ANE setup for fast-silicon."
    )
    parser.add_argument(
        "--packages",
        default="all",
        help="Comma-separated package list (all or subset of: bge-small,embeddinggemma-2,pplx,decision-models,gemma4,gpt-oss)",
    )
    parser.add_argument(
        "--target",
        choices=("auto", "in-repo", "cache"),
        default="auto",
        help="Artifact destination root (`auto` uses existing checkout/sibling/<home>/.cache layout; default=auto)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Custom root directory (`<root>/fast-<pkg>/artifacts`) overriding --target",
    )
    parser.add_argument(
        "--prebuilt",
        action="store_true",
        help="Zero-compile mode: pull vetted prebuilt HF checkpoints/CoreML bundles without local compilation",
    )
    parser.add_argument("--skip-native", action="store_true", help="Skip `make native` and `cpu-query` builds")
    parser.add_argument("--skip-ane", action="store_true", help="Skip CoreML `.mlmodelc` compilation steps")
    parser.add_argument("--skip-gpt-oss", action="store_true", help="Skip ~12 GB GPT-OSS-20B download & o200k remap")
    parser.add_argument(
        "--skip-test-extras",
        action="store_true",
        help="Skip test-only Gemma-4 BF16/QAT-4bit snapshots (~12.8 GB)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run weight export and CoreML ANE builds even if staged")
    parser.add_argument("--dry-run", action="store_true", help="Inspect staged state and print the plan without changes")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON plan with --dry-run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    packages = parse_packages_arg(args.packages, skip_gpt_oss=args.skip_gpt_oss)

    env = dict(os.environ)
    art_root = args.artifacts_root.expanduser().resolve() if args.artifacts_root is not None else None
    if art_root is not None:
        env["FAST_ARTIFACTS_ROOT"] = str(art_root)

    native_actions, plans = build_setup_plan(
        packages=packages,
        target_mode=args.target,
        artifacts_root=art_root,
        force=bool(args.force),
        prebuilt=bool(args.prebuilt),
        skip_native=bool(args.skip_native),
        skip_ane=bool(args.skip_ane),
        skip_test_extras=bool(args.skip_test_extras),
    )

    if args.dry_run:
        if args.json:
            payload: dict[str, Any] = {
                "dry_run": True,
                "prebuilt": bool(args.prebuilt),
                "packages": list(packages),
                "target": args.target,
                "native_actions": [asdict(a) for a in native_actions],
                "plans": [
                    {
                        "package": p.package,
                        "target_home": str(p.target_home),
                        "approx_size_gb": p.approx_size_gb,
                        "hf_repos": list(p.hf_repos),
                        "has_pending": p.has_pending,
                        "actions": [asdict(a) for a in p.actions],
                    }
                    for p in plans
                ],
            }
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(_format_plan_summary(native_actions, plans))
        return 0

    _execute_plan(native_actions, plans, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
