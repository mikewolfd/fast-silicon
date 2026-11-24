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
        "qwen3",
        "ocr",
    )
    with pytest.raises(ValueError, match="Unknown package"):
        mod.parse_packages_arg("bge-small,nonexistent")


def test_sync_from_siblings_transform_content_markdown_links_and_idempotence() -> None:
    sync_path = _REPO_ROOT / "scripts" / "sync_from_siblings.py"
    spec = importlib.util.spec_from_file_location("test_sync_from_siblings_mod", sync_path)
    assert spec is not None and spec.loader is not None
    sync_mod = importlib.util.module_from_spec(spec)
    sys.modules["test_sync_from_siblings_mod"] = sync_mod
    spec.loader.exec_module(sync_mod)

    src_md = (
        "# Sample README\n\n"
        "Inline file: [Benchmarks](../fast-silicon/docs/BENCHMARKS.md#retrieval)\n"
        "Inline root anchor: [Overview](../fast-silicon/#overview)\n"
        "Inline root dir: [Monorepo](../fast-silicon/)\n"
        "Inline subdir: [Docs](../fast-silicon/docs/)\n\n"
        "```markdown\n"
        "[Do Not Rewrite](../fast-silicon/docs/BENCHMARKS.md#inside-fence)\n"
        "[ref-fence]: ../fast-silicon/docs/BENCHMARKS.md\n"
        "```\n\n"
        "~~~text\n"
        "[Also Skip Tilde](../fast-silicon/#tilde)\n"
        "~~~\n\n"
        "[bench]: ../fast-silicon/docs/BENCHMARKS.md#structured-generation\n"
        "[root-anch]: ../fast-silicon/#top\n"
        "[root-dir]: ../fast-silicon/\n"
        "[angle]: <../fast-silicon/docs/benchmarks/review-20261121.md#decision-scoring>\n"
    ).encode("utf-8")

    once = sync_mod._transform_content("fast-gemma4", "README.md", src_md)
    text = once.decode("utf-8")

    assert "[Benchmarks](../../docs/BENCHMARKS.md#retrieval)" in text
    assert "[Overview](../../#overview)" in text
    assert "[Monorepo](../../)" in text
    assert "[Docs](../../docs/)" in text
    assert "[Do Not Rewrite](../fast-silicon/docs/BENCHMARKS.md#inside-fence)" in text
    assert "[ref-fence]: ../fast-silicon/docs/BENCHMARKS.md" in text
    assert "[Also Skip Tilde](../fast-silicon/#tilde)" in text
    assert "[bench]: ../../docs/BENCHMARKS.md#structured-generation" in text
    assert "[root-anch]: ../../#top" in text
    assert "[root-dir]: ../../" in text
    assert "[angle]: <../../docs/benchmarks/review-20261121.md#decision-scoring>" in text

    # Idempotence: running _transform_content a second time changes zero bytes.
    twice = sync_mod._transform_content("fast-gemma4", "README.md", once)
    assert twice == once

    # Deeper nested markdown path rewrites `../../fast-silicon/` to `../../../...`
    nested_src = b"See [Guide](../../fast-silicon/docs/BENCHMARKS.md#top) and [Ref][r]\n\n[r]: ../../fast-silicon/#root\n"
    nested_once = sync_mod._transform_content("fast-gemma4", "docs/guide.md", nested_src)
    assert nested_once.decode("utf-8") == "See [Guide](../../../docs/BENCHMARKS.md#top) and [Ref][r]\n\n[r]: ../../../#root\n"
    assert sync_mod._transform_content("fast-gemma4", "docs/guide.md", nested_once) == nested_once

    # pyproject.toml workspace transform
    py_raw = b'fast-shared = { path = "../fast-shared", editable = true }\n'
    py_out = sync_mod._transform_content("fast-gemma4", "pyproject.toml", py_raw)
    assert py_out == b"fast-shared = { workspace = true }\n"
    assert sync_mod._transform_content("fast-shared", "pyproject.toml", py_raw) == py_raw


def test_setup_all_prebuilt_zero_compile_repos_and_offline_plan_cpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_setup_all()

    expected_repos = {
        "OpenInfinity/embeddinggemma-2-coreml-int8",
        "NMTZ/laya-typed-decisions-421m-coreml-ane",
        "AbstractFramework/kev-0.8b-mlx-8bit",
        "AbstractFramework/kev-4b-mlx-4bit",
        "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    }
    all_registered = {repo for repos in mod.VETTED_PREBUILT_HF_REPOS.values() for repo in repos}
    assert expected_repos <= all_registered

    # Offline --prebuilt --dry-run --json builds zero-compile HF snapshot plan without network calls
    rc = mod.main(
        [
            "--dry-run",
            "--json",
            "--prebuilt",
            "--skip-native",
            "--artifacts-root",
            str(tmp_path / "prebuilt_root"),
            "--packages",
            "embeddinggemma-2,decision-models,gemma4,gpt-oss",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prebuilt"] is True
    planned_repos = { repo for plan in payload["plans"] for repo in plan["hf_repos"] }
    assert expected_repos <= planned_repos
    for plan in payload["plans"]:
        assert all(not act["requires_coreml"] for act in plan["actions"])

    # Programmatic pull_prebuilt_package delegates cleanly to injected downloader
    pulled: list[tuple[str, bool]] = []

    def _fake_downloader(*, repo_id: str, local_files_only: bool = False) -> Path:
        pulled.append((repo_id, local_files_only))
        d = tmp_path / repo_id.replace("/", "--")
        d.mkdir(parents=True, exist_ok=True)
        return d

    out_map = mod.pull_prebuilt_package("decision-models", local_files_only=True, downloader=_fake_downloader)
    assert "NMTZ/laya-typed-decisions-421m-coreml-ane" in out_map
    assert "AbstractFramework/kev-0.8b-mlx-8bit" in out_map
    assert "AbstractFramework/kev-4b-mlx-4bit" in out_map
    assert all(local_only is True for _, local_only in pulled)


def test_setup_all_qwen3_uses_complete_snapshot_check_for_partial_hf_cache_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial HF snapshots with config+tokenizer+safetensors satisfy _plan_qwen3 without full optiq_vision downloads."""
    import fast_qwen3.paths as q3_paths

    mod = _load_setup_all()
    home = tmp_path / "fast-qwen3"
    (home / "artifacts" / "tokenizer").mkdir(parents=True)
    (home / "artifacts" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    # Simulate huggingface_hub.snapshot_download(..., local_files_only=True) failing when optional vision weights
    # are absent from the local cache, while fast_qwen3.paths.resolve_paths(...).resolve_model_dir confirms complete.
    fake_hub_mod = ModuleType("huggingface_hub")

    def _fail_full_snapshot(*_a, **_kw):
        raise FileNotFoundError("optiq/optiq_vision.safetensors missing")

    fake_hub_mod.snapshot_download = _fail_full_snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub_mod)

    snap = tmp_path / "qwen3_snap"
    (snap / "optiq").mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"w")

    class _FakeQwen3Paths:
        def __init__(self, h: Path) -> None:
            self.home = h

        def resolve_model_dir(self, *, repo_id: str | None = None, local_files_only: bool = True) -> Path:
            assert repo_id == mod._QWEN3_HF_REPO
            assert local_files_only is True
            assert q3_paths.Qwen3Paths.is_complete_snapshot(snap)
            return snap

    monkeypatch.setattr(q3_paths, "resolve_paths", lambda h=None: _FakeQwen3Paths(Path(h)))
    plan = mod._plan_qwen3(
        home,
        force=False,
        hf_checker=mod._is_hf_snapshot_cached,
        python_exe=sys.executable,
    )
    assert plan.has_pending is False

    # Custom hf_checker returning False (e.g., unit test overrides) still triggers pending download.
    plan_missing = mod._plan_qwen3(
        home,
        force=False,
        hf_checker=lambda _repo: False,
        python_exe=sys.executable,
    )
    assert plan_missing.has_pending is True


def test_setup_all_decision_models_multifunction_and_ocr_plan_cpu(tmp_path: Path) -> None:
    mod = _load_setup_all()

    # Empty .mlmodelc directory must NOT mark decision-models ready
    dm_home = tmp_path / "fast-decision-models"
    laya_dir = dm_home / "artifacts" / "laya_typed"
    kev_dir = dm_home / "artifacts" / "kev_08b"
    (laya_dir / "tokenizer").mkdir(parents=True)
    (laya_dir / "laya_typed_emb_f16.npz").write_bytes(b"npz")
    (laya_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (laya_dir / "laya_typed_s256_w8.mlmodelc").mkdir()
    (laya_dir / "laya_typed_s512_w8.mlmodelc").mkdir()
    kev_dir.mkdir(parents=True)
    (kev_dir / "model_w8.safetensors").write_bytes(b"st")
    (kev_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    plan_empty = mod._plan_decision_models(
        dm_home,
        force=False,
        skip_ane=False,
        hf_checker=lambda _r: True,
        python_exe=sys.executable,
    )
    assert plan_empty.has_pending is True

    # Valid multifunction laya_typed_w8.mlmodelc + symlinks marks decision-models ready
    mf_bundle = laya_dir / "laya_typed_w8.mlmodelc"
    mf_bundle.mkdir()
    (mf_bundle / "coremldata.bin").write_bytes(b"bin")
    for s in (64, 128, 256, 512):
        link = laya_dir / f"laya_typed_s{s}_w8.mlmodelc"
        if link.is_dir() and not link.is_symlink():
            link.rmdir()
        link.symlink_to(mf_bundle, target_is_directory=True)

    plan_mf = mod._plan_decision_models(
        dm_home,
        force=False,
        skip_ane=False,
        hf_checker=lambda _r: True,
        python_exe=sys.executable,
    )
    assert plan_mf.has_pending is False

    # OCR package plan tracks HF cache
    ocr_plan_miss = mod._plan_ocr(tmp_path / "fast-ocr", force=False, hf_checker=lambda _r: False)
    ocr_plan_hit = mod._plan_ocr(tmp_path / "fast-ocr", force=False, hf_checker=lambda r: r == mod._OCR_HF_REPO)
    assert ocr_plan_miss.has_pending is True
    assert ocr_plan_hit.has_pending is False


