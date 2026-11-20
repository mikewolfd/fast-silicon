"""Pure-CPU unit tests for the unified `scripts/setup_all.py` workspace bootstrap."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SETUP_ALL_PATH = _REPO_ROOT / "scripts" / "setup_all.py"


def _load_setup_all() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_setup_all_mod", _SETUP_ALL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_setup_all_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_setup_all_dry_run_and_target_planning_cpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_setup_all()

    rc = mod.main(
        [
            "--dry-run",
            "--json",
            "--artifacts-root",
            str(tmp_path / "empty_root"),
            "--packages",
            "bge-small,pplx,decision-models,gemma4",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert [p["package"] for p in payload["plans"]] == ["bge-small", "pplx", "decision-models", "gemma4"]
    assert all(p["has_pending"] for p in payload["plans"])
    assert any(a["kind"] == "native" for a in payload["native_actions"])
    g4_plan = next(p for p in payload["plans"] if p["package"] == "gemma4")
    assert set(g4_plan["hf_repos"]) == {
        "lmstudio-community/gemma-4-E2B-it-MLX-4bit",
        "mlx-community/gemma-4-e2b-it-bf16",
        "mlx-community/gemma-4-E2B-it-qat-4bit",
    }


def test_setup_all_marks_staged_artifacts_ready_and_validates_flags_cpu(tmp_path: Path) -> None:
    mod = _load_setup_all()
    art_root = tmp_path / "staged"

    # Stage every required artifact expected for bge-small, pplx, gemma4, and gpt-oss under artifacts-root.
    bge_art = art_root / "fast-bge-small" / "artifacts"
    (bge_art / "weights").mkdir(parents=True)
    (bge_art / "tokenizer").mkdir(parents=True)
    (bge_art / "weights" / "bge_small_f16.npz").write_bytes(b"npz")
    (bge_art / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    for b, s in mod._BGE_REQUIRED_BUCKETS:
        (bge_art / "ane_models" / f"bge_b{b}_s{s}_fp16.mlmodelc").mkdir(parents=True)

    pplx_art = art_root / "fast-pplx" / "artifacts"
    (pplx_art / "weights").mkdir(parents=True)
    (pplx_art / "tokenizer").mkdir(parents=True)
    (pplx_art / "ane_models" / "stage_0.mlmodelc").mkdir(parents=True)
    (pplx_art / "weights" / "pplx_0_6b_f16.npz").write_bytes(b"npz")
    (pplx_art / "weights" / "embed_tokens.safetensors").write_bytes(b"st")
    (pplx_art / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (pplx_art / "ane_models" / "manifest.json").write_text("{}", encoding="utf-8")

    g4_art = art_root / "fast-gemma4" / "artifacts"
    (g4_art / "vocab").mkdir(parents=True)
    (g4_art / "tokenizer").mkdir(parents=True)
    (g4_art / "ane_models" / "gemma4_early14_prefix_S256_fp16.mlmodelc").mkdir(parents=True)
    (g4_art / "ane_models" / "gemma4_early14_prefix_S256_w8.mlmodelc").mkdir(parents=True)
    (g4_art / "vocab" / "vocab_remap_150786.npz").write_bytes(b"npz")
    (g4_art / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    oss_art = art_root / "fast-gpt-oss" / "artifacts"
    (oss_art / "vocab").mkdir(parents=True)
    (oss_art / "tokenizer").mkdir(parents=True)
    (oss_art / "hf_model").mkdir(parents=True)
    (oss_art / "vocab" / "o200k_remap_131072.npz").write_bytes(b"npz")
    (oss_art / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (oss_art / "hf_model" / "config.json").write_text("{}", encoding="utf-8")
    (oss_art / "hf_model" / "model.safetensors").write_bytes(b"st")

    native_ready, plans_ready = mod.build_setup_plan(
        packages=("bge-small", "pplx", "gemma4", "gpt-oss"),
        artifacts_root=art_root,
        skip_native=True,
        skip_test_extras=True,
        hf_checker=lambda repo: repo == mod._GEMMA4_HF_REPO,
    )
    assert native_ready == []
    assert [p.has_pending for p in plans_ready] == [False, False, False, False]

    _, plans_forced = mod.build_setup_plan(
        packages=("bge-small", "gpt-oss"),
        artifacts_root=art_root,
        force=True,
        skip_native=True,
        hf_checker=lambda _repo: True,
    )
    assert [p.has_pending for p in plans_forced] == [True, True]

    assert mod.parse_packages_arg("all", skip_gpt_oss=True) == (
        "bge-small",
        "embeddinggemma-2",
        "pplx",
        "decision-models",
        "gemma4",
    )
    with pytest.raises(ValueError, match="Unknown package"):
        mod.parse_packages_arg("bge-small,nonexistent")
