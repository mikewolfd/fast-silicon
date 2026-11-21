"""CPU unit tests for integrated `benchmark_outcomes.py`, `public_baselines.py`, and `benchmark_capabilities.py`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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


def test_public_baseline_preserves_worker_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess

    pub = _load_script("test_public_worker_exit", "public_baselines.py")
    for code, stdout, stderr, expected in (
        (1, "", "ModuleNotFoundError: missing dependency", "worker_error"),
        (0, "model initialized\n", "", "parse_error"),
        (0, '{"status":"skipped","runtime":"unavailable"}\n', "", "unavailable"),
    ):
        completed = subprocess.CompletedProcess([], code, stdout, stderr)
        monkeypatch.setattr(pub, "run_subprocess_with_killpg", lambda *a, **kw: completed)
        rc = pub.main(["--package", "coreml-laya"])
        result = json.loads(capsys.readouterr().out)
        assert result["runtime"] == expected
        assert rc == (0 if expected == "unavailable" else 1)

    def _timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(pub, "run_subprocess_with_killpg", _timeout)
    rc_to = pub.main(["--package", "coreml-laya", "--timeout-sec", "1"])
    res_to = json.loads(capsys.readouterr().out)
    assert rc_to == 1 and res_to["runtime"] == "timeout"


def test_coreml_empty_compute_plan_preserves_unknown_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    from enum import Enum
    from fast_shared import coreai_pipeline
    from types import SimpleNamespace

    bench = _load_script("test_coreml_plan", "_coreml_bench.py")
    units = Enum("ComputeUnit", ["CPU_AND_GPU"])
    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace(ComputeUnit=units))
    monkeypatch.setattr(coreai_pipeline, "inspect_coreml_bundle_placement", lambda *a, **kw: {"total_ops": 0})
    plan = bench._placement({"compiled_path": "Model.mlmodelc", "compute_units": "CPU_AND_GPU"})
    assert "no inspectable operations" in plan["text"]["error"]


def test_outcomes_public_baselines_and_flags(capsys: pytest.CaptureFixture[str]) -> None:
    out = _load_script("test_outcomes_mod", "benchmark_outcomes.py")

    # 1. --include-public-baselines expands fast-* packages cleanly in --dry-run
    rc = out.main(["run", "--dry-run", "--packages", "fast-bge-small,fast-pplx,fast-decision-models", "--include-public-baselines"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["packages"] == [
        "fast-bge-small",
        "public-bge-small",
        "coreml-bge-small",
        "fast-pplx",
        "public-pplx",
        "fast-decision-models",
        "public-laya",
        "coreai-laya",
        "public-kev-0.8b",
        "coreml-laya",
        "coreml-kev-0.8b",
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
                "objective_score_e2e": 0.725,
                "effective_outcome_per_sec": 1.724,
                "per_item_effective_scores": [0.8, 0.7, 0.9, 0.5],
            },
            {
                "package": "public-pplx",
                "config_label": "pplx-context-mlx-zainmerchan-f16",
                "category": 1,
                "benchmark": "NanoBEIR+LegalBench",
                "status": "ok",
                "objective_score_e2e": 0.725,
                "effective_outcome_per_sec": 0.862,
                "per_item_effective_scores": [0.8, 0.7, 0.9, 0.5],
            },
        ]
    )
    assert len(comps) == 1
    assert comps[0]["eot_speedup_vs_stock"] == pytest.approx(2.0, rel=1e-3)
    assert comps[0]["se_paired"] == 0.0


def test_required_generation_grammar_failure_identifies_the_item() -> None:
    out = _load_script("required_grammar_runner", "benchmark_outcomes.py")
    seen = []
    def compile_schema(schema, *, whitespace_flexible):
        seen.append(whitespace_flexible)
        if "not" in schema:
            raise ValueError("unsupported keyword")
        return "compiled"

    assert out._compile_required_grammars("fast-gemma4", [{}], compile_schema, whitespace_flexible=None) == ["compiled"]
    with pytest.raises(ValueError, match="fast-gemma4 input 1: unsupported keyword"):
        out._compile_required_grammars("fast-gemma4", [{}, {"not": {}}], compile_schema, whitespace_flexible=False)
    with pytest.raises(ValueError, match="Missing required schema for fast-gemma4 input 1"):
        out._compile_required_grammars("fast-gemma4", [{}, None], compile_schema, whitespace_flexible=True)
    assert seen == [None, False, False, True]


def test_retrieval_run_abba_and_compare_require_all_selected_queries(tmp_path, monkeypatch, capsys) -> None:
    out = _load_script("retrieval_gate_runner", "benchmark_outcomes.py")
    ids = ["unjudged", "miss", "hit"]
    ranking = {"per_query_ids": ["hit", "miss"], "per_query_valid": [True, True], "per_query_ndcg": [1.0, 0.0]}
    valid, scores, coverage = out._retrieval_query_outcomes(ids, ranking, "per_query_ndcg")
    assert valid == [False, True, True] and scores == [0, 0, 1]
    assert coverage["unscored_query_ids"] == ["unjudged"]
    assert coverage["n_queries"] == 3 and coverage["n_valid_queries"] == 2

    def record(valid_flags, parts):
        return out._build_record(
            package="fast-pplx", category=1, config_label="gpu", benchmark="retrieval",
            valid_flags=valid_flags, item_scores=scores, wall_sec=1.0, diagnostics={},
            memory_4bucket_gb={}, sub_benchmarks=parts, host_mem_gb=48.0,
        )

    complete = record([True] * 3, {})
    partial = record(valid, {"short": coverage})
    assert complete["gate_verdict"] == "PASS"
    assert partial["retrieval_coverage_status"] == "INCOMPLETE" and partial["gate_verdict"] == "FAIL"
    assert record([True] * 3, {"isolated": coverage})["gate_verdict"] == "FAIL"
    repeated = out._aggregate_abba_records([[complete], [partial]])[0]
    assert repeated["gate_verdict"] == "FAIL" and repeated["incomplete_retrieval_repeats"] == [2]

    monkeypatch.setattr(out, "_hw_lock", out._null_ctx)
    monkeypatch.setattr(out, "_spawn_isolated_worker", lambda **kw: [partial])
    assert out.main(["run", "--packages", "fast-pplx", "--allow-contended"]) == 1
    assert "2/3" in capsys.readouterr().out

    report = tmp_path / "archive.json"
    # Legacy snapshots could report PASS even for incomplete retrieval.
    report.write_text(json.dumps({"results": [dict(partial, gate_verdict="PASS")]}))
    assert out.main(["compare", str(report), str(report)]) == 1
    assert json.loads(capsys.readouterr().out)["comparisons"][0]["retrieval_coverage_pass"] is False

    # Same counts with a different query order cannot establish paired non-inferiority.
    _, _, ordered = out._retrieval_query_outcomes(["hit", "miss"], ranking, "per_query_ndcg")
    _, _, reordered = out._retrieval_query_outcomes(["miss", "hit"], ranking, "per_query_ndcg")
    assert ordered["ordered_query_ids_sha256"] != reordered["ordered_query_ids_sha256"]
    a, b = [record([True] * 3, {"short": c}) for c in (ordered, reordered)]
    changed = out.evaluate_promotion_gates(a, b)
    assert changed.verdict == "FAIL" and changed.quality_status == "UNRESOLVED" and changed.n_paired == 0


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


def test_mtp_counts_each_timed_arm_and_preserves_per_prompt_agreement(tmp_path, monkeypatch):
    cap = _load_script("capability_mtp_counts", "benchmark_capabilities.py")
    clock = [0.0]
    monkeypatch.setattr(cap, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate_batch(self, prompts, *, speculative, **kwargs):
            clock[0] += 1.0 if speculative else 2.0
            items = [SimpleNamespace(token_ids=[1] if speculative and i % 2 else [1, 2]) for i in range(len(prompts))]
            return SimpleNamespace(items=items)

        def close(self):
            pass

    fixture = ("system", "schema", list("abcd"))
    monkeypatch.setitem(sys.modules, "fast_gemma4.cli", SimpleNamespace(_load_fixture_dataset=lambda _: fixture))
    monkeypatch.setitem(sys.modules, "fast_gemma4.engine", SimpleNamespace(FastGemma4Engine=Engine))
    monkeypatch.setitem(sys.modules, "fast_gemma4.mtp_drafter", SimpleNamespace(resolve_mtp_drafter_dir=lambda **kw: tmp_path))
    monkeypatch.setitem(sys.modules, "fast_gemma4.paths", SimpleNamespace(resolve_paths=lambda home: home))
    monkeypatch.setattr(cap, "resolve_pkg_home", lambda *args: tmp_path)
    monkeypatch.setattr(cap, "get_unified_memory_footprint", lambda: {"peak_footprint_gb": 1})
    args = SimpleNamespace(abba_rounds=1, public_python=str(tmp_path / "missing"))
    result = cap._run_mtp_speculative(args)
    assert result["greedy_exact_token_matches"] == "6/10"
    assert result["speedup_vs_stock"] is None and result["b4_speedup_vs_stock"] is None
    four = result["workloads"]["json_b4"]
    assert four["ar_output_tokens"] == 8 and four["mtp_output_tokens"] == four["total_out_tokens"] == 6
    assert four["ar_tok_s"] == 4 and four["mtp_tok_s"] == 6
    assert four["elapsed_speedup_vs_ar"] == four["speedup_vs_ar"] == 2
    assert four["token_rate_ratio_vs_ar"] == 1.5
    assert four["exact_output_by_prompt"] == [True, False, True, False]
    assert len(four["samples"]["mtp"]) == 2

    stock_python = tmp_path / "stock"
    stock_python.touch()
    args.public_python = str(stock_python)
    receipt = {"status": "ok", "json_b1_tok_s": 1, "json_b4_tok_s": 3,
               "json_b1": {"median_seconds": 3}, "json_b4": {"median_seconds": 4}}
    monkeypatch.setattr(cap.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(receipt)))
    result = cap._run_mtp_speculative(args)
    assert result["b4_speedup_vs_stock"] == result["workloads"]["json_b4"]["token_rate_ratio_vs_stock"] == 2
    assert result["workloads"]["json_b4"]["elapsed_speedup_vs_stock"] == 4


def test_decoding_reports_missing_counts_and_short_batches():
    cap = _load_script("capability_token_count_failures", "benchmark_capabilities.py")
    assert cap._stock_generated_token_count(SimpleNamespace(generation_tokens=12), 1) == 12
    assert cap._stock_generated_token_count(SimpleNamespace(texts=["a", "b"], generation_tokens=[2, 3]), 2) == 5
    assert cap._stock_generated_token_count(SimpleNamespace(texts=["a", "b"], stats=SimpleNamespace(generation_tokens=7)), 2) == 7
    with pytest.raises(ValueError, match="generated-token count"):
        cap._stock_generated_token_count(SimpleNamespace(text="an ambiguous token count"), 1)
    with pytest.raises(ValueError, match="number of completions"):
        cap._stock_generated_token_count(SimpleNamespace(texts=["short"]), 2)
    with pytest.raises(ValueError, match="1 completions for 2"):
        result = SimpleNamespace(items=[SimpleNamespace(token_ids=[1])])
        cap._measure_mtp_pair(lambda: result, lambda: result, batch_size=2, rounds=1)


def test_qlora_report_checks_targets_and_distinguishes_repeated_inputs():
    cap = _load_script("capability_training_counts", "benchmark_capabilities.py")
    original = [([1, 2, 3, 4, 5], 2), ([1, 2, 8, 9], 3)]
    run = {"examples": 2, "supervised_tokens": 4, "logical_input_tokens": 7,
           "optimizer_steps": 2, "elapsed_seconds": 2, "peak_gb": 1.0}
    stock = cap._qlora_regime_report(original, [run, run], prompt_copies=1, pack_tokens=0)
    packed = cap._qlora_regime_report(original, [{**run, "optimizer_steps": 1}], prompt_copies=1, pack_tokens=2048)
    repeated = [([1, 1, 1, 1, 2, 3, 4, 5], 5)]
    shared_run = dict(run, examples=1, supervised_tokens=3, logical_input_tokens=7, optimizer_steps=1)
    shared = cap._qlora_regime_report(repeated, [shared_run], prompt_copies=4, pack_tokens=2048)
    assert stock["input_ids_and_prompt_lengths_sha256"] == packed["input_ids_and_prompt_lengths_sha256"]
    assert shared["input_ids_and_prompt_lengths_sha256"] != packed["input_ids_and_prompt_lengths_sha256"]
    assert stock["supervised_tokens_per_sec"] == 2 and stock["examples_per_sec"] == 1
    assert stock["tok_s"] == 3.5 and stock["elapsed_seconds"] == 2
    assert stock["optimizer_steps_per_pass"] == [2, 2] and packed["optimizer_steps_per_pass"] == [1]
    assert shared["examples"] == 1 and shared["system_instruction_copies"] == 4
    with pytest.raises(ValueError, match="QLoRA pass trained on"):
        cap._qlora_regime_report(original, [{**run, "supervised_tokens": 7}], prompt_copies=1, pack_tokens=0)


def test_parallel_coreml_reports_failed_predictions_and_changed_vectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    import numpy as np

    cap = _load_script("test_parallel_coreml_capability", "benchmark_capabilities.py")
    weights = tmp_path / "artifacts" / "weights" / "bge_small_f16.npz"
    weights.parent.mkdir(parents=True)
    weights.touch()
    monkeypatch.setattr(cap, "resolve_pkg_home", lambda *_: tmp_path)
    monkeypatch.setattr(cap.subprocess, "check_call", lambda *a, **kw: 0)
    monkeypatch.setattr(cap, "get_unified_memory_footprint", lambda: {"peak_footprint_gb": 1.0})
    args = argparse.Namespace(abba_rounds=1)

    def failed_prediction(*_args):
        raise AttributeError("embedding call unavailable")

    monkeypatch.setattr(cap, "_predict_bge_buckets", failed_prediction)
    failed = cap._run_parallel_coreml(args)
    assert failed["status"] == "failed"
    assert "embedding call unavailable" in failed["embedding_comparison"]["reason"]
    assert "min_cosine" not in failed["embedding_comparison"]

    def changed_prediction(_home, build):
        vector = [0, 1] if build.name.endswith("j4") else [1, 0]
        return {"example.mlmodelc": np.array([vector], dtype=np.float32)}

    monkeypatch.setattr(cap, "_predict_bge_buckets", changed_prediction)
    changed = cap._run_parallel_coreml(args)
    assert changed["status"] == "failed"
    assert changed["embedding_comparison"]["min_cosine"] == 0.0
    assert changed["embedding_comparison"]["all_bit_identical"] is False


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


def test_coreml_package_expansion_and_comparisons_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    out = _load_script("test_outcomes_coreml_mod", "benchmark_outcomes.py")

    rc = out.main(["run", "--dry-run", "--packages", "coreml"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["packages"] == [
        "coreml-bge-small",
        "coreml-embeddinggemma-2-fp16",
        "coreml-embeddinggemma-2-int8",
        "coreml-laya",
        "coreml-kev-0.8b",
    ]

    comps = out.compute_public_comparisons(
        [
            {
                "package": "fast-embeddinggemma-2",
                "config_label": "hetero",
                "category": 1,
                "benchmark": "NanoBEIR+LegalBench",
                "status": "ok",
                "objective_score_e2e": 0.54,
                "effective_outcome_per_sec": 4.0,
                "per_item_effective_scores": [0.6, 0.5, 0.7, 0.36],
            },
            {
                "package": "coreml-embeddinggemma-2-int8",
                "config_label": "coreml-emb2-int8-ane",
                "category": 1,
                "benchmark": "NanoBEIR+LegalBench",
                "status": "ok",
                "objective_score_e2e": 0.50,
                "effective_outcome_per_sec": 2.0,
                "per_item_effective_scores": [0.55, 0.45, 0.65, 0.35],
            },
            {
                "package": "fast-decision-models",
                "config_label": "laya_ane",
                "category": 2,
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "status": "ok",
                "diagnostics": {"backend": "laya-ane"},
                "objective_score_e2e": 0.72,
                "effective_outcome_per_sec": 6.0,
                "per_item_effective_scores": [1.0, 0.0, 1.0, 0.88],
            },
            {
                "package": "coreml-laya",
                "config_label": "coreml_laya_ane",
                "category": 2,
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "status": "ok",
                "diagnostics": {"backend": "coreml-laya"},
                "objective_score_e2e": 0.72,
                "effective_outcome_per_sec": 3.0,
                "per_item_effective_scores": [1.0, 0.0, 1.0, 0.88],
            },
            {
                "package": "fast-decision-models",
                "config_label": "kev_08b_w8",
                "category": 2,
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "status": "ok",
                "diagnostics": {"backend": "kev-0.8b"},
                "objective_score_e2e": 0.78,
                "effective_outcome_per_sec": 5.0,
                "per_item_effective_scores": [1.0, 1.0, 0.6, 0.52],
            },
            {
                "package": "coreml-kev-0.8b",
                "config_label": "coreml_kev_08b_fp16_gpu",
                "category": 2,
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "status": "ok",
                "diagnostics": {"backend": "coreml-kev-0.8b"},
                "objective_score_e2e": 0.78,
                "effective_outcome_per_sec": 2.5,
                "per_item_effective_scores": [1.0, 1.0, 0.6, 0.52],
            },
        ]
    )
    by_stock = {c["stock_package"]: c for c in comps}
    assert set(by_stock) == {"coreml-embeddinggemma-2-int8", "coreml-laya", "coreml-kev-0.8b"}
    assert by_stock["coreml-embeddinggemma-2-int8"]["eot_speedup_vs_stock"] == pytest.approx(2.0, rel=1e-3)
    assert by_stock["coreml-laya"]["eot_speedup_vs_stock"] == pytest.approx(2.0, rel=1e-3)
    assert by_stock["coreml-kev-0.8b"]["eot_speedup_vs_stock"] == pytest.approx(2.0, rel=1e-3)

    common = {"status": "ok", "category": 1, "benchmark": "retrieval"}
    native = {**common, "package": "fast-bge-small", "n_items": 325}
    short_run = {**common, "package": "coreml-bge-small", "n_items": 12}
    assert out.compute_public_comparisons([native, short_run]) == []

    # _merge_daemon_telemetry preserves compile_sec and coreml_info
    mem4 = {"peak_footprint_gb": 0.0, "combined_with_model_mapped_gb": 0.0, "neural_nofootprint_gb": 0.0, "model_mapped_gb": 0.0}
    diag: dict[str, object] = {}
    out._merge_daemon_telemetry(
        mem4,
        diag,
        {
            "peak_gb": 0.85,
            "compile_sec": 2.345,
            "coreml_info": {"repo_id": "OpenInfinity/embeddinggemma-2-coreml-int8", "compute_units": "CPU_AND_NE"},
            "four_bucket_metrics": {"neural_nofootprint_mb": 64.0, "mapped_model_mb": 128.0, "combined_with_model_mapped_gb": 0.9},
        },
    )
    assert diag["compile_sec"] == 2.345
    assert diag["coreml_info"]["compute_units"] == "CPU_AND_NE"  # type: ignore[index]


def test_coreml_materialization_and_packers_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _load_script("test_pub_coreml_mod", "public_baselines.py")
    opb = _load_script("test_opb_coreml_mod", "_outcome_public_backends.py")

    # 1. Precompiled .mlmodelc with HF-style symlinks materializes symlink-free into cache
    hf_home = tmp_path / "hf"
    cache_root = tmp_path / "cm_cache"
    monkeypatch.setenv("HF_HOME", str(hf_home))

    repo_dir = hf_home / "hub" / "models--demo--encoder-coreml"
    blob_dir = repo_dir / "blobs"
    blob_dir.mkdir(parents=True)
    blob_w = blob_dir / "deadbeef"
    blob_w.write_bytes(b"weights123")

    snap = repo_dir / "snapshots" / "rev1"
    mc_src = snap / "text" / "Model.mlmodelc"
    (mc_src / "weights").mkdir(parents=True)
    (mc_src / "coremldata.bin").write_bytes(b"meta")
    (mc_src / "weights" / "weight.bin").symlink_to(blob_w)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("rev1\n", encoding="utf-8")

    out_mc, compile_s = pub.resolve_coreml_compiled_model(
        "demo/encoder-coreml", "text/Model.mlmodelc", cache_root=cache_root
    )
    assert compile_s == 0.0
    assert (out_mc / "coremldata.bin").is_file()
    assert not (out_mc / "weights" / "weight.bin").is_symlink()
    assert (out_mc / "weights" / "weight.bin").read_bytes() == b"weights123"
    # Second call hits cache directly
    out_mc2, compile_s2 = pub.resolve_coreml_compiled_model(
        "demo/encoder-coreml", "text/Model.mlmodelc", cache_root=cache_root
    )
    assert out_mc2 == out_mc and compile_s2 == 0.0

    # 2. EmbeddingGemma-2 FFD pack_256 packer
    packed, singles = opb.pack_emb2_sequences_ffd([300, 120, 120, 40, 30, 30, 30, 30, 30, 30, 30, 20])
    assert 0 in singles  # >256 tokens goes to embed_512
    flat = [i for b in packed for i in b] + singles
    assert sorted(flat) == list(range(12))
    for b in packed:
        assert len(b) <= 8
        assert sum([300, 120, 120, 40, 30, 30, 30, 30, 30, 30, 30, 20][i] for i in b) <= 256

    # Lone short sequence falls back to embed_S instead of pack_256
    packed_lone, singles_lone = opb.pack_emb2_sequences_ffd([40])
    assert packed_lone == [] and singles_lone == [0]

    # 3. KevFused lane starts & group packing (matching KevFastManager.swift)
    assert opb.kev_lane_starts([60, 60, 60], lane=128) == [0, 60, 128]
    groups = opb.kev_pack_groups([60, 60, 60, 60], packed_len=192, lane=128, max_readouts=16)
    assert groups == [[0, 1, 2], [3]]

    # 4. Pure-Python Kev question encoder escapes control tokens and tracks option/decide indices
    class _FakeEnc:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    class _FakeTok:
        def encode(self, txt: str, add_special_tokens: bool = False) -> _FakeEnc:
            assert not add_special_tokens
            assert "<|decide|>" not in txt
            return _FakeEnc([10 + i for i in range(max(1, len(txt.split())))])

    s_ids, q_rows = opb.encode_kev_questions_pure(
        _FakeTok(),
        [248060, 248061, 248049, 248050, 248062],
        "hello <|decide|> world",
        [{"type": "noul", "instructions": "is it valid?"}],
    )
    assert s_ids[0] == 248060
    assert len(q_rows) == 1
    assert q_rows[0]["ids"][0] == 248061
    assert q_rows[0]["ids"][q_rows[0]["decide"]] == 248062
    assert len(q_rows[0]["opts"]) == 2
    assert all(q_rows[0]["ids"][p] == 248050 for p in q_rows[0]["opts"])


def test_coreml_microbench_uses_native_corpus_ids_and_in_process_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    import numpy as np

    bench = _load_script("test_coreml_bench", "_coreml_bench.py")
    for bge, n_docs, tokens, first in ((True, 1000, 249303, [101, 23632]), (False, 2000, 551889, [2, 175638])):
        docs = bench.encoder_workload(bge)
        assert (len(docs), sum(map(len, docs)), docs[0][:2]) == (n_docs, tokens, first)

    # Return different durations for packing and one-at-a-time execution.
    def encode(inputs, pack):
        vectors = np.zeros((len(inputs), 768), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors, 2.0 if pack else 4.0

    def pair(a, b, **kwargs):
        return {"order": ["A", "B", "B", "A"], "a_runs": [{"payload": a()}], "b_runs": [{"payload": b()}]}

    monkeypatch.setattr(bench, "encoder_workload", lambda _: [[2, 1], [2, 3, 4, 1]])
    adapter = SimpleNamespace(encode_token_ids=encode, harness_label="test", close=lambda: None, telemetry=dict)
    monkeypatch.setattr(bench, "build_public_cat1_adapter", lambda *args, **kwargs: adapter)
    metrics, _ = bench._encoder("coreml-embeddinggemma-2-int8", pair, 0)
    assert (metrics["tokens"], metrics["tok_s"], metrics["single_embed_tok_s"]) == (6, 3.0, 1.5)
    assert (metrics["pack_256_speedup"], metrics["min_pack_single_cosine"]) == (2.0, 1.0)


def test_killpg_timeout_kills_grandchild_and_capabilities_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os
    import subprocess
    import time

    pub = _load_script("test_pub_killpg_mod", "public_baselines.py")
    cap = _load_script("test_cap_killpg_mod", "benchmark_capabilities.py")

    pid_file = tmp_path / "grandchild.pid"
    parent_script = (
        "import pathlib, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid), encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        pub.run_subprocess_with_killpg([sys.executable, "-c", parent_script], timeout_sec=0.5)

    assert pid_file.is_file()
    gc_pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Wait briefly for kernel process-group teardown after SIGKILL.
    dead = False
    for _ in range(40):
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.05)
    assert dead, f"grandchild pid {gc_pid} survived run_subprocess_with_killpg timeout"

    # Capabilities coordinator uses run_subprocess_with_killpg and fails cleanly on timeout.
    monkeypatch.setattr(cap, "_hw_lock", cap._null_ctx)
    monkeypatch.setattr(cap, "check_system_quiescence", lambda: [])

    def _raise_to(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(cap, "run_subprocess_with_killpg", _raise_to)
    rc = cap.main(["--suites", "parallel-coreml", "--allow-contended", "--timeout-sec", "1"])
    assert rc == 1
    assert "timeout (1s)" in capsys.readouterr().out


def test_coreai_client_close_kills_hung_daemon_and_cat12_finally_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types
    import numpy as np

    opb = _load_script("test_opb_close_mod", "_outcome_public_backends.py")
    out = _load_script("test_out_close_mod", "benchmark_outcomes.py")

    # 1. Hung daemon is SIGKILLed and its tempdir is cleaned up on close(), and close() is idempotent.
    hung_daemon = (
        "import time\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    client = opb._CoreAiSubprocessClient(
        cmd=[sys.executable],
        daemon_script=hung_daemon,
        script_name="hung.py",
        prefix="test_hung_daemon_",
    )
    client._close_timeout = 0.2
    tmp_dir = client.tmp_dir
    assert tmp_dir.is_dir()
    client.close()
    assert client._proc.poll() is not None
    assert not tmp_dir.exists()
    # Second close() is a clean no-op.
    assert isinstance(client.close(), dict)

    # 2. Startup failure calls both kill() and wait().
    waited = {"kill": 0, "wait": 0}

    class _StartFailProc:
        def __init__(self, *_a, **_kw):
            import io

            self.stdin = io.StringIO()
            self.stdout = io.StringIO("NOT_READY\n")

        def kill(self) -> None:
            waited["kill"] += 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            waited["wait"] += 1
            return -9

    with pytest.raises(RuntimeError, match="failed to start"):
        opb._CoreAiSubprocessClient(cmd=[sys.executable], daemon_script="pass", popen_cls=_StartFailProc)
    assert waited == {"kill": 1, "wait": 1}

    # 3. Cat 1 finally: closes coreml-* adapter even when mid-eval raises.
    cat1_closed = {"count": 0}

    def _boom_encode(_txts: list[str]) -> np.ndarray:
        raise RuntimeError("mid-eval failure")

    fake_adapter = opb.Cat1Adapter(
        harness_label="coreml-bge-small-fp16-ane",
        mrl_dims=(128, "max"),
        int8_mode="sym_int8",
        nb_pre_normalized=True,
        enc_q_nb=_boom_encode,
        enc_d_nb=lambda txts: np.ones((len(txts), 4), dtype=np.float32),
        enc_q=lambda txts: np.ones((len(txts), 4), dtype=np.float32),
        enc_d=lambda txts: np.ones((len(txts), 4), dtype=np.float32),
        close_fn=lambda: (cat1_closed.__setitem__("count", cat1_closed["count"] + 1) or 0.0),
        telemetry_fn=dict,
    )
    monkeypatch.setitem(
        sys.modules,
        "_outcome_public_backends",
        SimpleNamespace(
            build_public_cat1_adapter=lambda *a, **kw: fake_adapter,
            stock_hf_model_files=lambda _pkg: [],
            stock_decision_model_files=lambda _b: [],
        ),
    )
    monkeypatch.setattr(
        out,
        "load_nanobeir_subset",
        lambda sub, **kw: {"queries": {"q1": "hello"}, "corpus": {"d1": "world"}, "qrels": {"q1": {"d1": 1.0}}},
    )
    monkeypatch.setattr(
        out,
        "load_legalbench_rag",
        lambda **kw: {
            "subsets": ["cuad"],
            "corpus_texts": {"doc1": "Some legal clause text."},
            "queries": [{"id": "lb1", "query": "clause?", "snippets": [{"file_path": "doc1", "span": [0, 10]}]}],
        },
    )
    monkeypatch.setattr(out, "FourBucketMemoryProbe", lambda **kw: SimpleNamespace(snapshot=lambda **k: {}))
    monkeypatch.setattr(out, "DeviceMemoryGuard", lambda: SimpleNamespace(arm=lambda: None, check=lambda: (True, None)))
    extra = out._parse_extra_flags([])
    lock = {"datasets": {"nanobeir_en": {"tier1_subsets": ["NanoSciFact"], "tier2_subsets": ["NanoSciFact"]}}}
    with pytest.raises(RuntimeError, match="mid-eval failure"):
        out._run_cat1_worker(
            "coreml-bge-small",
            tier="tier1",
            config_label="",
            extra=extra,
            lock=lock,
            cache_dir=tmp_path,
            local_files_only=True,
            max_items=None,
            smoke=True,
            home=tmp_path,
            host_mem_gb=48.0,
        )
    assert cat1_closed["count"] == 1

    # 4. Cat 2 finally: closes coreml-laya backend even when mid-eval raises.
    cat2_closed = {"count": 0}
    cat2_calls = {"n": 0}

    class _FakeCoreMlLaya:
        peak_gb = 0.0

        def encode_item(self, state, qdef, *, on_truncate="ignore"):
            del state, qdef, on_truncate
            return {"ids": [1, 2], "markers": [0, 1], "original_state_tokens": 2}

        def score_noul_batch(self, encs):
            del encs
            cat2_calls["n"] += 1
            if cat2_calls["n"] > 1:
                raise RuntimeError("cat2 mid-eval failure")
            return [0.5]

        def telemetry(self):
            return {}

        def close(self):
            cat2_closed["count"] += 1
            return 0.0

    monkeypatch.setitem(
        sys.modules,
        "_outcome_public_backends",
        SimpleNamespace(
            CoreMlLayaBackend=_FakeCoreMlLaya,
            stock_decision_model_files=lambda _b: [],
        ),
    )
    monkeypatch.setitem(sys.modules, "fast_decision_models.kev_mlx", types.SimpleNamespace(Kev08bMlxBackend=object))
    monkeypatch.setitem(sys.modules, "fast_decision_models.laya_ane", types.SimpleNamespace(LayaAneBackend=object))
    monkeypatch.setitem(
        sys.modules,
        "fast_decision_models.paths",
        types.SimpleNamespace(ensure_laya_dylib=lambda: None, resolve_kev_dir=lambda *a, **k: tmp_path, resolve_laya_dir=lambda *a, **k: tmp_path),
    )
    monkeypatch.setattr(out, "load_ragtruth", lambda **kw: [{"id": "r1", "source": "s", "response": "r", "label": 1}])
    monkeypatch.setattr(out, "load_clinc_oos", lambda **kw: {"items": []})
    monkeypatch.setattr(out, "load_reward_bench_2", lambda **kw: [])
    with pytest.raises(RuntimeError, match="cat2 mid-eval failure"):
        out._run_cat2_single_backend(
            "coreml-laya",
            pkg_name="coreml-laya",
            tier="tier1",
            config_label="",
            extra=extra,
            lock=lock,
            cache_dir=tmp_path,
            local_files_only=True,
            max_items=None,
            smoke=True,
            home=tmp_path,
            host_mem_gb=48.0,
        )
    assert cat2_closed["count"] == 1


def test_merge_daemon_telemetry_keeps_phys_peak_and_combined_distinct() -> None:
    out = _load_script("test_out_merge_distinct", "benchmark_outcomes.py")
    mib_to_gb = (1024.0 * 1024.0) / 1e9
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
        "peak_gb": 3.04,
        "four_bucket_metrics": {
            "rusage_version": 6,
            "phys_footprint_mb": 400.0,
            "lifetime_max_phys_footprint_mb": 450.0,
            "neural_nofootprint_mb": 2400.0,
            "mapped_model_mb": 100.0,
            "combined_with_model_mapped_mb": 2900.0,
        },
    }
    out._merge_daemon_telemetry(mem4, diag, telem)
    assert mem4["phys_footprint_gb"] == round(400.0 * mib_to_gb, 4)
    assert mem4["peak_footprint_gb"] == round(450.0 * mib_to_gb, 4)
    assert mem4["combined_with_model_mapped_gb"] == round(2900.0 * mib_to_gb, 4)
    assert mem4["phys_footprint_gb"] < mem4["peak_footprint_gb"] < mem4["combined_with_model_mapped_gb"]

    # Fallback to external_peak_gb ONLY when lifetime_max_phys_footprint_mb is missing.
    mem4_fb: dict[str, float] = {
        "phys_footprint_gb": 0.05,
        "peak_footprint_gb": 0.05,
        "combined_with_model_mapped_gb": 0.05,
    }
    out._merge_daemon_telemetry(
        mem4_fb,
        {},
        {
            "external_peak_gb": 2.25,
            "four_bucket_metrics": {"phys_footprint_mb": 100.0, "combined_with_model_mapped_mb": 2500.0},
        },
    )
    assert mem4_fb["peak_footprint_gb"] == 2.25
    assert mem4_fb["combined_with_model_mapped_gb"] == round(2500.0 * mib_to_gb, 4)


def test_max_coarse_tokens_unjudged_filter_abba_public_python_and_smoke_unresolved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _load_script("test_out_coarse_and_smoke", "benchmark_outcomes.py")

    # 1. --max-coarse-tokens CLI flag & dry-run propagation
    rc_dry = out.main(["run", "--dry-run", "--packages", "fast-embeddinggemma-2", "--max-coarse-tokens", "512"])
    assert rc_dry == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_coarse_tokens"] == 512
    assert out._parse_extra_flags(["--max-coarse-tokens", "512"]).max_coarse_tokens == 512

    # 2. coarse_max_tokens_mismatch in compute_public_comparisons
    base_rec = {
        "category": 1,
        "benchmark": "NanoBEIR+LegalBench",
        "n_items": 4,
        "status": "ok",
        "objective_score_e2e": 0.52,
        "effective_outcome_per_sec": 3.0,
        "per_item_effective_scores": [0.5, 0.5, 0.5, 0.58],
    }
    mismatch = out.compute_public_comparisons(
        [
            {**base_rec, "package": "fast-embeddinggemma-2", "config_label": "hetero", "diagnostics": {"coarse_max_tokens": 2048}},
            {**base_rec, "package": "coreml-embeddinggemma-2-int8", "config_label": "int8", "diagnostics": {"coarse_max_tokens": 512}},
        ]
    )
    assert len(mismatch) == 1 and mismatch[0]["coarse_max_tokens_mismatch"] is True

    matched = out.compute_public_comparisons(
        [
            {**base_rec, "package": "fast-embeddinggemma-2", "config_label": "hetero", "diagnostics": {"coarse_max_tokens": 512}},
            {**base_rec, "package": "coreml-embeddinggemma-2-int8", "config_label": "int8", "diagnostics": {"coarse_max_tokens": 512}},
        ]
    )
    assert len(matched) == 1 and matched[0]["coarse_max_tokens_mismatch"] is False

    # 3. MTEB unjudged query filtering
    judged, unjudged = out._split_judged_query_ids(
        ["q_missing", "q_zero", "q_pos"],
        {"q_zero": {"d1": 0.0}, "q_pos": {"d1": 1.0}},
    )
    assert judged == ["q_pos"]
    assert unjudged == ["q_missing", "q_zero"]

    # 4. --abba-compare honors --public-python AND UNRESOLVED exits 0 under --smoke / --max-items, 1 on full runs.
    seen_pythons: list[str] = []

    def _fake_spawn(*, python_exe: str, pkg_name: str, label: str, extra_flags: str, args) -> list[dict[str, object]]:
        del extra_flags, args
        seen_pythons.append(python_exe)
        # Small noisy delta whose 95% CI spans -0.003 -> UNRESOLVED.
        scores = [0.5, 0.5, 0.5, 0.5] if label == "base" else [0.49, 0.51, 0.49, 0.51]
        return [
            out._build_record(
                package=pkg_name,
                category=2,
                config_label=label,
                benchmark="decisions",
                valid_flags=[True] * 4,
                item_scores=scores,
                wall_sec=1.0,
                diagnostics={"repeat_loop_count": 0},
                memory_4bucket_gb={"combined_with_model_mapped_gb": 1.0},
                sub_benchmarks={},
                host_mem_gb=48.0,
            )
        ]

    monkeypatch.setattr(out, "_hw_lock", out._null_ctx)
    monkeypatch.setattr(out, "check_system_quiescence", lambda: [])
    monkeypatch.setattr(out, "_spawn_isolated_worker", _fake_spawn)

    rc_smoke = out.main([
        "run", "--packages", "public-laya", "--public-python", "/opt/stock/python",
        "--abba-compare", "base:", "cand:", "--smoke", "--cooldown-sec", "0",
    ])
    _ = capsys.readouterr()
    assert rc_smoke == 0
    assert seen_pythons and all(p == "/opt/stock/python" for p in seen_pythons)

    rc_max = out.main([
        "run", "--packages", "public-laya", "--public-python", "/opt/stock/python",
        "--abba-compare", "base:", "cand:", "--max-items", "4", "--cooldown-sec", "0",
    ])
    _ = capsys.readouterr()
    assert rc_max == 0

    rc_full = out.main([
        "run", "--packages", "public-laya", "--public-python", "/opt/stock/python",
        "--abba-compare", "base:", "cand:", "--cooldown-sec", "0",
    ])
    _ = capsys.readouterr()
    assert rc_full == 1


def test_coreml_decision_probe_excludes_ipc_and_stock_hf_model_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _load_script("test_coreml_dec_ipc", "_coreml_bench.py")
    opb = _load_script("test_opb_hf_files", "_outcome_public_backends.py")

    # 1. _decision uses in-process last_encode_sec rather than outer IPC wall time
    class _FakeLaya:
        last_encode_sec = 0.0

        def encode_item(self, text, _qdef):
            return {"ids": [1] * (40 if len(text) < 100 else 400), "markers": [0, 1]}

        def score_noul_batch(self, encs):
            self.last_encode_sec = 0.01 if len(encs[0]["ids"]) < 100 else 0.05
            return [0.5]

        def close(self):
            return 0.0

        def telemetry(self):
            return {"coreml_info": {"buckets": [192, 384, 512]}}

    def _pair(a, b, **_kw):
        # Deliberately return huge outer wall time vs small in-process encode_sec payload.
        return {
            "order": ["A", "B", "B", "A"],
            "a_median_s": 10.0,
            "b_median_s": 20.0,
            "a_runs": [{"payload": a()}],
            "b_runs": [{"payload": b()}],
        }

    monkeypatch.setattr(bench, "CoreMlLayaBackend", _FakeLaya)
    metrics, _ = bench._decision("coreml-laya", _pair, 0.0)
    assert metrics["short_ms"] == pytest.approx(10.0, rel=1e-6)
    assert metrics["long_ms"] == pytest.approx(50.0, rel=1e-6)
    assert "excluding temp-JSON IPC" in metrics["timing_scope"]

    # 2. stock_hf_model_files resolves blobs symlinks for Cat 1 and Cat 3/4 public-* packages
    snap = tmp_path / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    blob = tmp_path / "blobs" / "weights.safetensors"
    blob.parent.mkdir()
    blob.write_bytes(b"ok")
    (snap / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr(opb, "resolve_hf_snapshot", lambda _repo: snap)
    for pkg in ("public-bge-small", "public-embeddinggemma-2", "public-pplx", "public-gemma4", "public-gpt-oss"):
        assert opb.stock_hf_model_files(pkg) == [blob]
    assert opb.stock_hf_model_files("fast-bge-small") == []


def test_llm_worker_merges_final_telemetry_before_gate_and_grpo_error_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect
    import subprocess
    import types

    out = _load_script("test_out_llm_gate_mod", "benchmark_outcomes.py")
    cap = _load_script("test_cap_grpo_err_mod", "benchmark_capabilities.py")

    # 1. _run_llm_worker merges final daemon telemetry from pub_llm.close() BEFORE _build_record computes gate_verdict.
    state = {"closed": False}

    def _fake_gen(prompts, **_kw):
        return ["Answer: A"] * len(prompts), ["stop"] * len(prompts), [[1, 2]] * len(prompts), {"completion_tokens": 2 * len(prompts), "mask_fallbacks": 0}

    def _fake_close():
        state["closed"] = True
        return 22.0  # Exceeds 80% of 24 GB host_mem_gb (= 19.2 GB)

    def _fake_telem():
        if not state["closed"]:
            return {"peak_gb": 1.0, "four_bucket_metrics": {"combined_with_model_mapped_gb": 1.0}}
        return {"peak_gb": 22.0, "four_bucket_metrics": {"combined_with_model_mapped_gb": 22.0, "lifetime_max_phys_footprint_mb": 20000.0}}

    fake_llm = SimpleNamespace(
        default_label="coreai_qwen35_4b_4bit",
        generate_fn=_fake_gen,
        close=_fake_close,
        telemetry=_fake_telem,
    )
    monkeypatch.setitem(
        sys.modules,
        "_outcome_public_backends",
        SimpleNamespace(build_public_llm_adapter=lambda *a, **k: fake_llm, stock_hf_model_files=lambda _p: []),
    )
    monkeypatch.setattr(
        out,
        "FourBucketMemoryProbe",
        lambda **kw: SimpleNamespace(
            snapshot=lambda **k: {
                "phys_footprint": {"phys_footprint_gb": 0.1, "peak_footprint_gb": 0.1},
                "task_vm_info": {"ledger_tag_neural_nofootprint_bytes": 0},
                "mapped_files": {"model_bytes": 0},
                "combined_with_model_mapped_gb": 0.1,
            }
        ),
    )
    monkeypatch.setattr(out, "DeviceMemoryGuard", lambda: SimpleNamespace(arm=lambda: None, check=lambda: (True, None)))
    monkeypatch.setattr(out, "load_ifbench_test", lambda **kw: [{"id": "i1", "prompt": "Say ok", "instruction_id_list": [], "kwargs": []}])
    monkeypatch.setattr(out, "load_mmlu_redux", lambda **kw: [{"id": "m1", "subject": "math", "question": "1+1?", "choices": ["2", "3", "4", "5"], "answer": 0}])
    monkeypatch.setattr(out, "load_math_500", lambda **kw: [{"id": "ma1", "problem": "1+1", "answer": "2"}])
    monkeypatch.setattr(out, "evaluate_instruction_following", lambda *a, **k: {"prompt_loose_acc": 1.0, "repeat_4gram_rate": 0.0, "repeat_loop_count": 0, "per_item_valid": [True], "per_item_scores": [1.0]})
    monkeypatch.setattr(out, "evaluate_mmlu_redux", lambda *a, **k: {"exact_accuracy": 1.0, "repeat_loop_count": 0, "per_item_valid": [True], "per_item_scores": [1.0]})
    monkeypatch.setattr(out, "evaluate_math_500", lambda *a, **k: {"verified_accuracy": 1.0, "repeat_loop_count": 0, "per_item_valid": [True], "per_item_scores": [1.0]})

    recs = out._run_llm_worker(
        "coreai-qwen3.5-4b",
        tier="tier1",
        config_label="",
        extra=out._parse_extra_flags(["--categories", "3"]),
        lock={"datasets": {"mmlu_redux_2": {"subjects": ["math"]}}},
        cache_dir=tmp_path,
        local_files_only=True,
        max_items=None,
        smoke=True,
        include_unconstrained=False,
        home=tmp_path,
        host_mem_gb=24.0,
    )
    assert len(recs) == 1
    assert recs[0]["memory_4bucket_gb"]["combined_with_model_mapped_gb"] == 22.0
    assert recs[0]["gate_verdict"] == "FAIL"
    assert any("19.2" in r or "ceiling" in r.lower() or "mem" in r.lower() for r in recs[0]["gate_reasons"])

    # 2. grpo-ane-critic warms PublicKev08bMlxBackend before t1, captures non-zero subprocess stderr,
    #    and qlora-packing / encoder-finetune standardize peak_gb to decimal GB (/ 1e9).
    grpo_src = inspect.getsource(cap._run_grpo_ane_critic)
    assert grpo_src.index('"warmup"') < grpo_src.index("t1 = time.perf_counter()")
    qlora_src = inspect.getsource(cap._run_qlora_packing)
    enc_src = inspect.getsource(cap._run_encoder_finetune)
    assert "peak_bytes / 1e9" in qlora_src and "peak_bytes / 1e9" in enc_src

    laya_bundle = tmp_path / "laya_typed_s512_w8.mlmodelc"
    laya_bundle.mkdir()
    fake_py = tmp_path / "fake_py"
    fake_py.touch()

    class _FakeG4:
        def __init__(self, **_kw):
            pass

        def generate_batch(self, prompts, **_kw):
            return SimpleNamespace(items=[SimpleNamespace(text="{}", token_ids=[1, 2]) for _ in prompts])

        def close(self):
            pass

    class _FakeLayaAne:
        def __init__(self, *_a, **_kw):
            pass

        def encode_item(self, *_a, **_kw):
            return {"ids": [1, 2], "markers": [0, 1]}

        def score_noul_batch(self, encs):
            return [0.8] * len(encs)

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "fast_decision_models.laya_ane", types.SimpleNamespace(LayaAneBackend=_FakeLayaAne))
    monkeypatch.setitem(sys.modules, "fast_decision_models.paths", types.SimpleNamespace(ensure_laya_dylib=lambda: None, resolve_laya_dir=lambda *a, **k: tmp_path))
    monkeypatch.setitem(sys.modules, "fast_gemma4.cli", types.SimpleNamespace(_load_fixture_dataset=lambda _p: ("sys", "{}", ["p0", "p1", "p2", "p3"])))
    monkeypatch.setitem(sys.modules, "fast_gemma4.engine", types.SimpleNamespace(FastGemma4Engine=_FakeG4))
    monkeypatch.setitem(sys.modules, "fast_gemma4.paths", types.SimpleNamespace(resolve_paths=lambda h: h))
    monkeypatch.setattr(cap, "resolve_pkg_home", lambda *_a: tmp_path)
    monkeypatch.setattr(cap, "get_unified_memory_footprint", lambda: {"peak_footprint_gb": 1.23})
    monkeypatch.setattr(
        cap,
        "run_subprocess_with_killpg",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 2, "", "RuntimeError: critic exploded\n"),
    )
    grpo_out = cap._run_grpo_ane_critic(SimpleNamespace(abba_rounds=1, public_python=str(fake_py), lora_python=str(fake_py)))
    assert grpo_out["stock_detail"]["status"] == "failed"
    assert grpo_out["stock_detail"]["returncode"] == 2
    assert "critic exploded" in grpo_out["stock_detail"]["reason"]
    assert grpo_out["grpo_step_overlap"]["status"] == "failed"
    assert grpo_out["grpo_step_overlap"]["returncode"] == 2
    assert "critic exploded" in grpo_out["grpo_step_overlap"]["reason"]
