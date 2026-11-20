"""CPU unit tests for integrated `benchmark_outcomes.py`, `public_baselines.py`, and `benchmark_capabilities.py`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _load_script(name: str, filename: str) -> ModuleType:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_public_baselines_snapshot_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _load_script("test_pub_baselines_mod", "public_baselines.py")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    repo_dir = tmp_path / "hub" / "models--zainmerchan--pplx-embed-v2-context-0.6b-mlx"
    snap_dir = repo_dir / "snapshots" / "abc1234"
    snap_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc1234\n", encoding="utf-8")

    assert pub.resolve_hf_snapshot("zainmerchan/pplx-embed-v2-context-0.6b-mlx") == snap_dir
    assert pub.resolve_hf_snapshot("missing/model") is None


def test_outcomes_public_baselines_and_flags(capsys: pytest.CaptureFixture[str]) -> None:
    out = _load_script("test_outcomes_mod", "benchmark_outcomes.py")

    # 1. --include-public-baselines expands fast-* packages cleanly in --dry-run
    rc = out.main(["run", "--dry-run", "--packages", "fast-bge-small,fast-pplx,fast-decision-models", "--include-public-baselines"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["packages"] == [
        "fast-bge-small",
        "public-bge-small",
        "fast-pplx",
        "public-pplx",
        "fast-decision-models",
        "public-laya",
        "coreai-laya",
        "public-kev-0.8b",
    ]

    # 2. --packages public selects stock baseline specs directly
    rc_pub = out.main(["run", "--dry-run", "--packages", "public", "--skip-gpt-oss"])
    assert rc_pub == 0
    plan_pub = json.loads(capsys.readouterr().out)
    assert "public-bge-small" in plan_pub["packages"]
    assert "public-gpt-oss" not in plan_pub["packages"]

    # 3. Extra engine capability flags (--mtp-drafter, --mlp-dtype, --lm-head-mode, --coreai-variant)
    ns = out._parse_extra_flags(["--mtp-drafter", "--mlp-dtype", "quantized", "--lm-head-mode", "bf16_mpp", "--coreai-variant", "fp4_fp8"])
    assert ns.mtp_drafter is True
    assert ns.mlp_dtype == "quantized"
    assert ns.lm_head_mode == "bf16_mpp"
    assert ns.coreai_variant == "fp4_fp8"

    # 4. Paired public comparison math (EOT speedup + paired SE)
    comps = out.compute_public_comparisons(
        [
            {
                "package": "fast-pplx",
                "config_label": "hetero2",
                "category": 1,
                "benchmark": "NanoBEIR+LegalBench",
                "status": "ok",
                "objective_score_e2e": 0.7244,
                "effective_outcome_per_sec": 1.724,
                "per_item_effective_scores": [0.8, 0.7, 0.9, 0.5],
            },
            {
                "package": "public-pplx",
                "config_label": "pplx-context-mlx-zainmerchan-f16",
                "category": 1,
                "benchmark": "NanoBEIR+LegalBench",
                "status": "ok",
                "objective_score_e2e": 0.7240,
                "effective_outcome_per_sec": 0.862,
                "per_item_effective_scores": [0.8, 0.7, 0.9, 0.5],
            },
        ]
    )
    assert len(comps) == 1
    assert comps[0]["eot_speedup_vs_stock"] == pytest.approx(2.0, rel=1e-3)
    assert comps[0]["se_paired"] == 0.0


def test_capabilities_suite_dry_run_and_formatting(capsys: pytest.CaptureFixture[str]) -> None:
    cap = _load_script("test_capabilities_mod", "benchmark_capabilities.py")

    rc = cap.main(["--dry-run", "--suites", "qlora-packing,mtp-speculative,prefix-tree"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["suites"] == ["qlora-packing", "mtp-speculative", "prefix-tree"]

    with pytest.raises(ValueError, match="Unknown suite"):
        cap.select_suites("nonexistent-suite")

    md = cap.format_capabilities_markdown(
        [
            {
                "suite": "prefix-tree",
                "fast_metric": "B=1 46.3 q/s | B=8 22.6 q/s",
                "stock_baseline": "mlx-lm kev-8bit: B=8 8.73 q/s",
                "speedup_vs_stock": 2.59,
                "peak_gb": 1.78,
                "status": "ok",
            }
        ]
    )
    assert "`prefix-tree`" in md
    assert "`2.59x`" in md


def test_coreai_subprocess_client_preamble_and_telemetry_cpu(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _outcome_public_backends import (  # type: ignore[import-not-found]
        _CoreAiSubprocessClient,
        _build_coreai_daemon_env,
    )

    env = _build_coreai_daemon_env({})
    assert "fast-shared/src" in env["PYTHONPATH"]

    fake_daemon = """
telem = _DaemonTelemetry([sys.argv[1]])
telem.mark_loaded(None, 12.5)
sys.stdout.write("READY\\n")
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
        sys.stdout.flush()
        break
    telem.record_infer(float(req["ms"]))
    sys.stdout.write(json.dumps({"ok": True, **telem.payload()}) + "\\n")
    sys.stdout.flush()
"""
    client = _CoreAiSubprocessClient(
        cmd=[sys.executable],
        daemon_script=fake_daemon,
        script_name="fake_coreai_daemon.py",
        script_args=[tmp_path],
        prefix="test_coreai_client_",
    )
    try:
        r1 = client.request({"ms": 4.0})
        assert r1["ok"] is True
        client.request({"ms": 1.0})
        client.request({"ms": 1.5})
    finally:
        telem = client.close()

    assert telem["peak_gb"] > 0.0
    assert telem["four_bucket_metrics"]["rusage_version"] == 6
    assert telem["cold_warm_latency"]["cold_load_ms"] == 12.5
    assert telem["cold_warm_latency"]["cold_first_infer_ms"] == 4.0
    assert telem["cold_warm_latency"]["warm_count"] == 2
    assert telem["cold_warm_latency"]["warm_median_ms"] == 1.25

    class _BadProc:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            import io

            self.stdin = io.StringIO()
            self.stdout = io.StringIO("CRASHED\n")
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    with pytest.raises(RuntimeError, match="failed to start"):
        _CoreAiSubprocessClient(
            cmd=[sys.executable],
            daemon_script="pass",
            popen_cls=_BadProc,
        )


def test_benchmark_outcomes_merge_daemon_telemetry_schema_cpu() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_outcomes as bo  # type: ignore[import-not-found]

    mem4: dict[str, float] = {
        "mlx_peak_gb": 0.0,
        "phys_footprint_gb": 0.05,
        "peak_footprint_gb": 0.05,
        "model_mapped_gb": 0.0,
        "neural_nofootprint_gb": 0.0,
        "combined_with_model_mapped_gb": 0.05,
    }
    diag: dict[str, object] = {}
    telem = {
        "peak_gb": 1.75,
        "four_bucket_metrics": {
            "rusage_version": 6,
            "neural_nofootprint_mb": 128.0,
            "mapped_model_mb": 256.0,
            "ane_daemon_footprint_delta_mb": 42.5,
            "combined_with_model_mapped_gb": 1.85,
        },
        "cold_warm_latency": {
            "cold_load_ms": 50.0,
            "cold_first_infer_ms": 10.0,
            "warm_count": 3,
            "warm_median_ms": 2.0,
        },
        "asset_summary": {"asset_valid": True, "storage_types": {"fp16": 4}},
        "profiler_available": True,
    }
    bo._merge_daemon_telemetry(mem4, diag, telem)
    mib_to_gb = (1024.0 * 1024.0) / 1e9
    assert mem4["peak_footprint_gb"] == 1.75
    assert mem4["combined_with_model_mapped_gb"] == 1.85
    assert mem4["neural_nofootprint_gb"] == round(128.0 * mib_to_gb, 4)
    assert mem4["model_mapped_gb"] == round(256.0 * mib_to_gb, 4)
    assert mem4["ane_daemon_footprint_delta_mb"] == 42.5
    assert diag["daemon_four_bucket_metrics"]["rusage_version"] == 6
    assert diag["cold_warm_latency"]["warm_median_ms"] == 2.0
    assert diag["asset_summary"]["asset_valid"] is True
    assert diag["profiler_available"] is True


def test_inspect_aimodel_asset_uses_authoring_load_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _outcome_public_backends as opb  # type: ignore[import-not-found]

    fake_bundle = tmp_path / "demo.aimodel"
    fake_bundle.mkdir()

    class _FakeSummary:
        storage_types = [("Float16", 12), ("Int8", 34)]
        compute_types = ["Float16", "Int8"]
        operation_distribution = [("conv2d", 4)]

    class _FakeAsset:
        @classmethod
        def load(cls, path: str) -> "_FakeAsset":
            assert path == str(fake_bundle)
            return cls()

        def is_valid(self, path: Path) -> bool:
            assert path == fake_bundle
            return True

        def summary(self, *, include_statistics: bool = False) -> _FakeSummary:
            assert include_statistics is True
            return _FakeSummary()

    # Execute _COREAI_DAEMON_PREAMBLE directly so we exercise the subprocess daemon code path
    # and confirm coreai.authoring.AIModelAsset.load is preferred without requiring coreai.runtime.
    monkeypatch.setitem(sys.modules, "coreai", types.ModuleType("coreai"))
    monkeypatch.setitem(sys.modules, "coreai.authoring", types.SimpleNamespace(AIModelAsset=_FakeAsset))
    monkeypatch.setitem(sys.modules, "coreai.runtime", types.ModuleType("coreai.runtime"))

    daemon_ns: dict[str, object] = {}
    exec(compile(opb._COREAI_DAEMON_PREAMBLE, "<_COREAI_DAEMON_PREAMBLE>", "exec"), daemon_ns)
    inspect_fn = daemon_ns["_inspect_aimodel_asset"]
    assert callable(inspect_fn)
    res = inspect_fn(fake_bundle)
    assert res == {
        "asset_valid": True,
        "path": str(fake_bundle),
        "storage_types": [("Float16", 12), ("Int8", 34)],
        "compute_types": ["Float16", "Int8"],
        "operation_distribution": [("conv2d", 4)],
    }
