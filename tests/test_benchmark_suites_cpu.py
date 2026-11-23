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


def test_hp04_coreai_cache_probe_gpt_oss_ungate_budgets_and_grouping_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opb = _load_script("test_hp04_opb", "_outcome_public_backends.py")
    out = _load_script("test_hp04_out", "benchmark_outcomes.py")

    # 1. CoreAI cache path included via Path.home() (zero literal tilde-slash)
    coreai_cache = opb.resolve_coreai_cache_dir()
    assert coreai_cache == Path.home() / "Library" / "Caches" / "coreai-cache"
    assert (chr(126) + "/") not in Path(opb.__file__).read_text(encoding="utf-8")

    # 2. GPT-OSS-20B un-gated on 24 GB Base M5 (>= 23.0 GB), gated on 16 GB under group token
    monkeypatch.setattr(out, "collect_env_telemetry", lambda: {"mem_gb": 24.0})
    assert out.main(["run", "--dry-run", "--packages", "all"]) == 0
    plan_24 = json.loads(capsys.readouterr().out)
    assert "fast-gpt-oss" in plan_24["packages"]

    monkeypatch.setattr(out, "collect_env_telemetry", lambda: {"mem_gb": 16.0})
    assert out.main(["run", "--dry-run", "--packages", "all"]) == 0
    plan_16 = json.loads(capsys.readouterr().out)
    assert "fast-gpt-oss" not in plan_16["packages"]

    # 3. MXFP4-Q8 preferred over MXFP4-Q4 in _STOCK_HF_REPOS
    assert opb._STOCK_HF_REPOS["public-gpt-oss"][0] == "mlx-community/gpt-oss-20b-MXFP4-Q8"
    assert "mlx-community/gpt-oss-20b-MXFP4-Q4" in opb._STOCK_HF_REPOS["public-gpt-oss"]

    # 4. Stock finish_length_count checks raw token length BEFORE stripping <|channel|>analysis<|message|>
    raw_truncated = "<|channel|>analysis<|message|>" + ("tok " * 62) + "<|channel|>final<|message|>short"
    clean_tx, hit_len, raw_cnt = opb._extract_harmony_final_and_length(
        raw_truncated,
        encode_fn=lambda s: s.split(),
        eff_max=64,
    )
    assert clean_tx == "short"
    assert raw_cnt >= 63 and hit_len is True

    raw_stopped = "<|channel|>analysis<|message|>brief<|channel|>final<|message|>done<|return|>"
    clean_ok, hit_ok, _ = opb._extract_harmony_final_and_length(
        raw_stopped,
        encode_fn=lambda s: s.split(),
        eff_max=64,
    )
    assert clean_ok == "done" and hit_ok is False

    # 5. RewardBench-2 (4x) and RAGTruth (<=6x) shared-prefix grouping preserves deterministic item order
    rt_sample = [
        {"id": "0", "context": "P_long " * 20, "query": "Q1", "output": "a"},
        {"id": "1", "context": "P_short", "query": "Q2", "output": "b"},
        {"id": "2", "context": "P_long " * 20, "query": "Q1", "output": "c"},
        {"id": "3", "context": "P_short", "query": "Q2", "output": "d"},
    ]
    order_rt, inv_rt = out._group_shared_prefix_indices(
        rt_sample,
        key_fn=lambda it: (it["context"], it["query"]),
        max_group_size=6,
        length_fn=lambda it: len(it["context"]) + len(it["query"]),
    )
    # Sibling items sharing (context, query) must be adjacent in grouped order
    assert order_rt in ([1, 3, 0, 2], [0, 2, 1, 3])
    scored_grouped = [f"score_{idx}" for idx in order_rt]
    restored = [scored_grouped[inv_rt[orig_i]] for orig_i in range(len(rt_sample))]
    assert restored == ["score_0", "score_1", "score_2", "score_3"]

    # 6. CoreAiGemma4Backend OS gate (< 27.2 raises FileNotFoundError -> SKIP in benchmark_outcomes)
    with pytest.raises(FileNotFoundError, match="27.2"):
        opb.CoreAiGemma4Backend(macos_version="26.6")

    # 7. CoreAiQwen35Backend & pplx ctx_max_len=2048 + GPT-OSS budgeted reasoning in _run_llm_worker
    opb_src = Path(opb.__file__).read_text(encoding="utf-8")
    assert '"--repetition-penalty", "1.05"' in opb_src or "'--repetition-penalty', '1.05'" in opb_src
    assert "<|endoftext|>" in opb_src
    assert "np.arange(cur_len, dtype=np.int32).reshape(1, cur_len)" in opb_src

    out_src = Path(out.__file__).read_text(encoding="utf-8")
    assert "ctx_max_len = 2048" in out_src
    assert "reasoning_budget=96" in out_src
    assert "reasoning_budget=192" in out_src
    assert "reasoning_budget=384" in out_src
    assert "reasoning_budget=64" in out_src

    # 8. #347 checks: structured.py maxItems=8 + coerce_one_of=True, fast-gemma4 mlp_dtype=auto & kv_i8=False,
    #    benchmark_monorepo.py <=24GB fast-gpt-oss --batches 1,8,16, and CoreAiQwen35Backend enable_thinking=False
    from fast_shared.outcome_eval.structured import build_bfcl_response_schema, evaluate_jsonschemabench_compiler

    bf_sch = build_bfcl_response_schema([{"name": "ping", "parameters": {"type": "object", "properties": {}}}])
    assert bf_sch["properties"]["tool_calls"]["maxItems"] == 8
    oneof_items = [{"id": "oneof_case", "json_schema": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}]
    jsb_res = evaluate_jsonschemabench_compiler(oneof_items)
    assert jsb_res["compiled_ok"] == 1 and jsb_res["compile_success_rate"] == 1.0

    def_flags = out._parse_extra_flags([])
    assert def_flags.mlp_dtype == "auto"
    assert def_flags.kv_i8 is None and def_flags.no_kv_i8 is False
    assert def_flags.batch_size is None

    mono = _load_script("test_hp04_mono", "benchmark_monorepo.py")
    mono_parser = mono.build_parser()
    mono_args = mono_parser.parse_args([])
    gpt_spec = mono._SPEC_BY_NAME["fast-gpt-oss"]
    assert mono._effective_bench_args(gpt_spec, mono_args, host_mem_gb=24.0) == ["bench", "--batches", "1,8,16", "--local-files-only"]
    assert mono._effective_bench_args(gpt_spec, mono_args, host_mem_gb=48.0) == ["bench", "--batches", "8,64,128", "--local-files-only"]
    assert '"enable_thinking": eff_think' in opb_src


def test_capabilities_qlora_24gb_gate_and_authentic_modernbert_packed_rope_cpu() -> None:
    import numpy as np

    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    cap = _load_script("test_cap_hp04", "benchmark_capabilities.py")

    # 1. QLoRA 24 GB vs 48 GB policy: ckpt_tokens == 512, BF16 baseline gated to >= 36.0 GB
    pol_24 = cap.select_qlora_policy(24.0)
    pol_48 = cap.select_qlora_policy(48.0)
    assert pol_24["ckpt_tokens"] == 512 and pol_24["run_bf16_baseline"] is False
    assert pol_48["ckpt_tokens"] == 512 and pol_48["run_bf16_baseline"] is True

    # 2. Authentic ModernBERT-large architecture + per-segment RoPE reset on Packed-512 (CPU stream)
    with mx.stream(mx.cpu):
        mx.random.seed(11)
        # Small hidden/layer scale of the exact _ModernBertLargeValidator class for sub-second CPU grad check,
        # plus structural check of default 28L, RoPE 160_000 / 10_000, local_attention=128, biasless LayerNorm, 2L head + scorer
        full_spec = cap._ModernBertLargeValidator(
            vocab=128, hidden=64, n_heads=4, d_ff=128, n_layers=28, freeze_below=18, head_ff=128
        )
        assert len(full_spec.layers) == 28
        assert full_spec.layers[0].attn_norm is None
        assert isinstance(full_spec.layers[1].attn_norm, nn.LayerNorm)
        assert "bias" not in full_spec.layers[1].attn_norm
        assert full_spec.layers[0].rope_theta == 160_000.0
        assert full_spec.layers[1].rope_theta == 10_000.0
        assert full_spec.local_attention == 128
        assert len(full_spec.head_layers) == 2

        full_spec.set_dtype(mx.float32)
        mx.eval(full_spec.parameters())

        seqs = [
            np.array([101, 11, 12, 13, 14, 15, 16, 102], dtype=np.int32),
            np.array([101, 21, 22, 23, 24, 25, 102], dtype=np.int32),
        ]
        markers = [[1, 3], [1, 3]]
        qtypes = np.array([0, 2], dtype=np.int32)
        labels = np.array([1, 0], dtype=np.int32)

        unpacked, packed = cap._prepare_modernbert_step_pair(
            seqs, markers, qtypes, labels, pack_cap=32, local_attention=128, dtype=mx.float32
        )
        # Verify per-segment position reset in Packed-512
        pos_np = np.asarray(packed["position_ids"])[0]
        assert list(pos_np[:8]) == list(range(8))
        assert list(pos_np[8:15]) == list(range(7))

        def _loss_fn(m, batch):
            h_f = m.forward_frozen(batch["input_ids"], batch["position_ids"], batch["mask_full"], batch["mask_local"])
            logits = m.forward_upper(
                h_f,
                batch["position_ids"],
                batch["mask_full"],
                batch["mask_local"],
                batch["qtypes"],
                batch["marker_positions"],
                fp32_head=True,
            )
            return mx.mean(nn.losses.cross_entropy(logits, batch["labels"]))

        vg = nn.value_and_grad(full_spec, _loss_fn)
        lu, gu = vg(full_spec, unpacked)
        lp, gp = vg(full_spec, packed)
        mx.eval(lu, gu, lp, gp)
        assert abs(float(lu.item()) - float(lp.item())) < 1e-5
        fu, fp = dict(tree_flatten(gu)), dict(tree_flatten(gp))
        dot = sum(float(mx.sum(fu[k] * fp[k]).item()) for k in fu if k in fp)
        nu = sum(float(mx.sum(fu[k] ** 2).item()) for k in fu if k in fp)
        np_ = sum(float(mx.sum(fp[k] ** 2).item()) for k in fu if k in fp)
        cos = dot / max((nu * np_) ** 0.5, 1e-12)
        assert cos > 0.9999


def test_jsonschema_fallback_without_jsonschema_package_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_shared.outcome_eval.structured import validate_against_jsonschema

    monkeypatch.setitem(sys.modules, "jsonschema", None)

    schema = {
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "minLength": 2, "maxLength": 6},
                    "qty": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["code", "qty"],
                "additionalProperties": False,
            }
        },
        "definitions": {
            "Tag": {"enum": ["alpha", "beta", 1]}
        },
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "kind": {"const": "order"},
                    "note": {"type": ["string", "null"]},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "tag": {"$ref": "#/definitions/Tag"},
                    "choice": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                    "either": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"$ref": "#/$defs/Item"},
                    },
                },
                "required": ["kind", "score", "items"],
                "additionalProperties": False,
            }
        ],
    }

    valid_obj = {
        "kind": "order",
        "note": None,
        "score": 0.75,
        "tag": "alpha",
        "choice": 42,
        "either": True,
        "items": [{"code": "AB1", "qty": 5}],
    }
    assert validate_against_jsonschema(valid_obj, schema) is True

    # Reject bool for integer/number and 1 vs True in enum/const
    assert validate_against_jsonschema({**valid_obj, "score": True}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": [{"code": "AB1", "qty": True}]}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "tag": True}, schema) is False

    # Reject extra key, missing required, minItems/maxItems, minLength/maxLength, minimum/maximum
    assert validate_against_jsonschema({**valid_obj, "extra": 1}, schema) is False
    assert validate_against_jsonschema({k: v for k, v in valid_obj.items() if k != "score"}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": []}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": [{"code": "AB", "qty": 1}] * 4}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": [{"code": "A", "qty": 1}]}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": [{"code": "ABCDEFG", "qty": 1}]}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "items": [{"code": "AB", "qty": 0}]}, schema) is False
    assert validate_against_jsonschema({**valid_obj, "choice": True}, schema) is False


def test_coreai_gemma4_llm_runner_bundle_preference_and_telemetry_parsing_cpu(tmp_path: Path) -> None:
    opb = _load_script("test_coreai_g4_fix_mod", "_outcome_public_backends.py")

    # 1. Create fake coreai tree with llm-runner binary + both 4-input mlboydaisuke and 2-input mikedoise bundles
    ca_root = tmp_path / "coreai_repo"
    ca_py = ca_root / "sub" / "venv" / "bin_py"
    ca_py.parent.mkdir(parents=True)
    ca_py.write_text("#!/bin/sh\n", encoding="utf-8")
    llm_runner = ca_root / ".build" / "out" / "Products" / "Release" / "llm-runner"
    llm_runner.parent.mkdir(parents=True)
    llm_runner.write_text("#!/bin/sh\n", encoding="utf-8")

    mlboy_dir = tmp_path / "mlboydaisuke_gemma4"
    (mlboy_dir / "z_second.aimodel").mkdir(parents=True)
    (mlboy_dir / "a_first.aimodel").mkdir(parents=True)

    mike_dir = tmp_path / "mikedoise_gemma4_4bit"
    (mike_dir / "text.aimodel").mkdir(parents=True)
    (mike_dir / "metadata.json").write_text(json.dumps({"per_layer_embeddings": "ple.safetensors"}), encoding="utf-8")

    snaps = {
        "mlboydaisuke/gemma-4-E2B-CoreAI": mlboy_dir,
        "mikedoise/gemma-4-E2B-it-coreai-4bit-128k-text": mike_dir,
    }
    sel_bdir, sel_aim_dir, llm_run_s, lbl = opb._select_coreai_gemma4_bundle(
        ca_py,
        resolve_snap_fn=lambda repo: snaps.get(repo),
        resolve_bundle_fn=lambda _rel: None,
    )
    assert sel_bdir == mike_dir
    assert sel_aim_dir == mike_dir
    assert llm_run_s == str(llm_runner)
    assert lbl == "coreai_gemma4_e2b_4bit"

    # Fallback to mlboydaisuke when mikedoise is absent uses deterministic sorted .aimodel discovery
    sel_b2, sel_aim2, _, lbl2 = opb._select_coreai_gemma4_bundle(
        ca_py,
        resolve_snap_fn=lambda repo: mlboy_dir if repo == "mlboydaisuke/gemma-4-E2B-CoreAI" else None,
        resolve_bundle_fn=lambda _rel: None,
    )
    assert sel_b2 == mlboy_dir and sel_aim2 == mlboy_dir and lbl2 == "coreai_gemma4_e2b_qat"

    # 2. Parse llm-runner stdout/stderr: cold Model Load subtracted from infer_ms, max_rss_gb + peak_footprint_gb + error
    stdout_ok = (
        "Loading model...\n"
        "Generating...\n"
        "Answer: B\n"
        "Done\n"
        "Performance Summary:\n"
        "  Model Load: 107250.5ms\n"
        "  Prefill: 80.0ms, 16 tokens (200.0 tok/s)\n"
        "  Generation: 180.0ms, 10 tokens (55.5 tok/s)\n"
    )
    stderr_ok = (
        "      3450000000  maximum resident set size\n"
        "      2850000000  peak memory footprint\n"
    )
    parsed_ok = opb._parse_coreai_llm_runner_result(stdout_ok, stderr_ok, 0, 107520.5)
    assert parsed_ok["ok"] is True
    assert parsed_ok["text"] == "Answer: B"
    assert parsed_ok["gen_tokens"] == 10
    assert parsed_ok["model_load_ms"] == pytest.approx(107250.5)
    assert parsed_ok["infer_ms"] == pytest.approx(270.0)
    assert parsed_ok["peak_footprint_gb"] == pytest.approx(2.85)
    assert parsed_ok["max_rss_gb"] == pytest.approx(3.45)

    daemon_ns: dict[str, object] = {}
    exec(compile(opb._COREAI_DAEMON_PREAMBLE, "<_COREAI_DAEMON_PREAMBLE>", "exec"), daemon_ns)
    telem_cls = daemon_ns["_DaemonTelemetry"]
    apply_fn = daemon_ns["_apply_runner_telemetry"]
    telem = telem_cls([mike_dir])  # type: ignore[operator]
    apply_fn(telem, None, parsed_ok)  # type: ignore[operator]
    pay = telem.payload(final=True)
    assert pay["cold_warm_latency"]["cold_load_ms"] == pytest.approx(107250.5)
    assert pay["cold_warm_latency"]["cold_first_infer_ms"] == pytest.approx(270.0)
    assert pay["external_peak_gb"] == pytest.approx(2.85)
    assert pay["max_rss_gb"] == pytest.approx(3.45)

    # 3. _CoreAiSubprocessClient._merge_response + benchmark_outcomes._merge_daemon_telemetry preserve max_rss_gb & external_peak_gb
    stub_client = object.__new__(opb._CoreAiSubprocessClient)
    stub_client._telemetry = {"peak_gb": 0.0, "four_bucket_metrics": {}, "cold_warm_latency": {}, "profiler_available": False}
    stub_client._merge_response({"max_rss_gb": 3.45, "external_peak_gb": 1.29})
    client_telem = stub_client.telemetry()
    assert client_telem["max_rss_gb"] == pytest.approx(3.45)
    assert client_telem["external_peak_gb"] == pytest.approx(1.29)

    out_mod = _load_script("test_coreai_g4_out_mod", "benchmark_outcomes.py")
    diag_check: dict[str, object] = {}
    out_mod._merge_daemon_telemetry({"phys_footprint_gb": 0.1, "peak_footprint_gb": 0.1, "combined_with_model_mapped_gb": 0.1}, diag_check, client_telem)
    assert diag_check["max_rss_gb"] == pytest.approx(3.45)
    assert diag_check["external_peak_gb"] == pytest.approx(1.29)

    parsed_err = opb._parse_coreai_llm_runner_result("", "Fatal error: Expected 2 inputs, got 4\n", 1, 50.0)
    assert parsed_err["ok"] is False
    assert "Expected 2 inputs, got 4" in parsed_err["error"]

    opb_src = Path(opb.__file__).read_text(encoding="utf-8")
    assert "timeout=600" in opb_src


def test_cat5_cross_family_and_m4_vs_m5_rates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _load_script("test_cat5_cross_family_out", "benchmark_outcomes.py")
    opb = _load_script("test_cat5_cross_family_opb", "_outcome_public_backends.py")

    # 1. --packages ocr-families --dry-run expands all 14 cross-family packages
    rc_dry = out.main(["run", "--dry-run", "--packages", "ocr-families"])
    assert rc_dry == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["packages"] == list(out.OCR_FAMILY_PACKAGES)
    assert len(plan["packages"]) == 14
    for expected in (
        "fast-ocr",
        "public-ovis-mlx-vlm",
        "public-glm-ocr-mlx-vlm",
        "public-lightonocr2-mlx-vlm",
        "public-deepseek-ocr2-mlx-vlm",
        "public-paddleocr-vl-mlx-vlm",
        "public-nemotron-parse-mlx-vlm",
        "public-rapidocr",
        "public-tesseract",
        "public-docling",
        "public-marker",
    ):
        assert expected in plan["packages"]

    # 2. Family-native OCR prompt resolution (page vs table)
    assert opb.resolve_cat5_native_prompt("public-glm-ocr-mlx-vlm", task_kind="page") == "Text Recognition:"
    assert opb.resolve_cat5_native_prompt("public-glm-ocr-mlx-vlm", task_kind="table") == "Table Recognition:"
    assert opb.resolve_cat5_native_prompt("public-lightonocr2-mlx-vlm", task_kind="page") == ""
    assert opb.resolve_cat5_native_prompt("public-paddleocr-vl-mlx-vlm", task_kind="page") == "OCR:"
    assert opb.resolve_cat5_native_prompt("public-paddleocr-vl-mlx-vlm", task_kind="table") == "Table Recognition:"
    assert "<|grounding|>" in opb.resolve_cat5_native_prompt("public-deepseek-ocr2-mlx-vlm", task_kind="page")

    # 3. _summarize_cat5_rates + compute_public_comparisons + format_outcomes_markdown
    rates = out._summarize_cat5_rates(
        [
            {"backend_used": "ovis:8bit", "model_family": "qwen3.5-ovis", "generated_tokens": 300, "prompt_tokens": 600, "visual_tokens": 500, "decode_ms": 2000.0, "decode_tps": 150.0},
            {"backend_used": "ovis:8bit", "model_family": "qwen3.5-ovis", "generated_tokens": 500, "prompt_tokens": 800, "visual_tokens": 700, "decode_ms": 2000.0, "decode_tps": 250.0},
        ],
        n_pages=2,
        wall_sec=5.0,
        default_family="qwen3.5-ovis",
    )
    assert rates["pages_per_sec"] == pytest.approx(0.4)
    assert rates["sec_per_page"] == pytest.approx(2.5)
    assert rates["total_generated_tokens"] == 800
    assert rates["tokens_per_page"] == pytest.approx(400.0)
    assert rates["decode_tok_s"] == pytest.approx(200.0)
    assert rates["e2e_tok_s"] == pytest.approx(160.0)

    def _cat5_rec(pkg: str, family: str, cfg: str, wall: float, gen_toks: int, dec_tps: float, score: float, mem: float) -> dict[str, object]:
        n = 4
        return {
            "package": pkg,
            "category": 5,
            "config_label": cfg,
            "benchmark": "FinTabNet-OTSL(4)",
            "status": "ok",
            "n_items": n,
            "n_valid": n,
            "wall_sec": wall,
            "raw_items_per_sec": round(n / wall, 4),
            "valid_outputs_per_sec": round(n / wall, 4),
            "objective_score_e2e": score,
            "effective_outcome_per_sec": round((n / wall) * score, 4),
            "per_item_effective_scores": [score] * n,
            "memory_4bucket_gb": {"combined_with_model_mapped_gb": mem},
            "diagnostics": {
                "backend": "ovis" if pkg == "fast-ocr" else pkg,
                "model_family": family,
                "pages_per_sec": round(n / wall, 4),
                "sec_per_page": round(wall / n, 4),
                "total_generated_tokens": gen_toks,
                "tokens_per_page": round(gen_toks / n, 2),
                "decode_tok_s": dec_tps,
                "e2e_tok_s": round(gen_toks / wall, 2) if gen_toks > 0 else 0.0,
            },
            "gate_verdict": "PASS",
        }

    m4_rows = [
        _cat5_rec("fast-ocr", "qwen3.5-ovis", "ovis_gfm_render", wall=4.0, gen_toks=1200, dec_tps=400.0, score=0.90, mem=1.8),
        _cat5_rec("public-glm-ocr-mlx-vlm", "glm-ocr", "glm_ocr_mlx_vlm", wall=8.0, gen_toks=1000, dec_tps=200.0, score=0.88, mem=2.6),
        _cat5_rec("public-rapidocr", "rapidocr-onnx", "rapidocr_onnx", wall=2.0, gen_toks=0, dec_tps=0.0, score=0.45, mem=0.4),
        {"package": "public-deepseek-ocr2-mlx-vlm", "category": 5, "config_label": "deepseek", "benchmark": "all", "status": "skipped", "gate_verdict": "SKIP"},
    ]
    m5_rows = [
        _cat5_rec("fast-ocr", "qwen3.5-ovis", "ovis_gfm_render", wall=5.0, gen_toks=1200, dec_tps=300.0, score=0.90, mem=1.75),
        _cat5_rec("public-glm-ocr-mlx-vlm", "glm-ocr", "glm_ocr_mlx_vlm", wall=10.0, gen_toks=1000, dec_tps=150.0, score=0.88, mem=2.55),
        _cat5_rec("public-rapidocr", "rapidocr-onnx", "rapidocr_onnx", wall=1.6, gen_toks=0, dec_tps=0.0, score=0.45, mem=0.38),
        {"package": "public-deepseek-ocr2-mlx-vlm", "category": 5, "config_label": "deepseek", "benchmark": "all", "status": "skipped", "gate_verdict": "SKIP"},
    ]

    pub_comps = out.compute_public_comparisons(m4_rows)
    assert len(pub_comps) == 2
    glm_pair = next(c for c in pub_comps if c["stock_package"] == "public-glm-ocr-mlx-vlm")
    assert glm_pair["fast_family"] == "qwen3.5-ovis"
    assert glm_pair["stock_family"] == "glm-ocr"
    assert glm_pair["pages_per_sec_ratio"] == pytest.approx(2.0)
    assert glm_pair["decode_tok_s_ratio"] == pytest.approx(2.0)
    assert glm_pair["e2e_tok_s_ratio"] == pytest.approx(2.4)
    assert glm_pair["fast_tokens_per_page"] == pytest.approx(300.0)
    assert glm_pair["stock_tokens_per_page"] == pytest.approx(250.0)

    md_run = out.format_outcomes_markdown(m4_rows)
    assert "### Category 5 OCR Cross-Family Token & Page Rates" in md_run
    assert "`qwen3.5-ovis`" in md_run and "`glm-ocr`" in md_run and "`rapidocr-onnx`" in md_run

    # 4. Cross-chip compare (Apple M4 Pro vs Apple M5) JSON & Markdown tables
    m4_json = tmp_path / "m4_ocr.json"
    m5_json = tmp_path / "m5_ocr.json"
    md_file = tmp_path / "m4_vs_m5.md"
    m4_json.write_text(
        json.dumps({"environment": {"chip": "Apple M4 Pro", "mem_gb": 48.0}, "dataset_lock_sha256": "abc", "results": m4_rows}),
        encoding="utf-8",
    )
    m5_json.write_text(
        json.dumps({"environment": {"chip": "Apple M5", "mem_gb": 24.0}, "dataset_lock_sha256": "abc", "results": m5_rows}),
        encoding="utf-8",
    )

    rc_json = out.main(["compare", str(m4_json), str(m5_json), "--min-eot-ratio", "0.0", "--markdown-out", str(md_file)])
    assert rc_json == 0
    cmp_payload = json.loads(capsys.readouterr().out)
    assert cmp_payload["verdict"] == "PASS"
    assert len(cmp_payload["comparisons"]) == 3  # skipped package cleanly ignored
    fast_cmp = next(c for c in cmp_payload["comparisons"] if c["package"] == "fast-ocr")
    assert fast_cmp["baseline_chip"] == "Apple M4 Pro"
    assert fast_cmp["candidate_chip"] == "Apple M5"
    assert fast_cmp["baseline_pages_per_sec"] == pytest.approx(1.0)
    assert fast_cmp["candidate_pages_per_sec"] == pytest.approx(0.8)
    assert fast_cmp["pages_per_sec_ratio"] == pytest.approx(0.8)
    assert fast_cmp["baseline_tokens_per_page"] == pytest.approx(300.0)
    assert fast_cmp["candidate_tokens_per_page"] == pytest.approx(300.0)
    assert fast_cmp["baseline_decode_tok_s"] == pytest.approx(400.0)
    assert fast_cmp["candidate_decode_tok_s"] == pytest.approx(300.0)
    assert fast_cmp["decode_tok_s_ratio"] == pytest.approx(0.75)
    assert fast_cmp["baseline_e2e_tok_s"] == pytest.approx(300.0)
    assert fast_cmp["candidate_e2e_tok_s"] == pytest.approx(240.0)
    assert fast_cmp["e2e_tok_s_ratio"] == pytest.approx(0.8)

    rc_md = out.main(["compare", str(m4_json), str(m5_json), "--min-eot-ratio", "0.0", "--markdown"])
    assert rc_md == 0
    md_text = capsys.readouterr().out
    assert "### Category 5 Cross-Silicon OCR Token & Page Rates (`Apple M4 Pro` vs `Apple M5`)" in md_text
    assert "M4 Pro `pp/s`" in md_text and "M5 `pp/s`" in md_text
    assert "M4 Pro Dec `tok/s`" in md_text and "M5 Dec `tok/s`" in md_text
    assert "0.80x" in md_text and "0.75x" in md_text
    assert md_file.read_text(encoding="utf-8").strip() == md_text.strip()


def test_cat5_gov_suite_routing_and_stage_gov_parser() -> None:
    out = _load_script("test_gov_suite_routing", "benchmark_outcomes.py")
    ns_gov = out._parse_extra_flags(["--ocr-suites", "gov"])
    assert out._resolve_cat5_suites("fast-ocr", ns_gov) == ["doclaynet_gov", "us_gov_ocr"]
    assert out._resolve_cat5_suites("public-pypdfium2", ns_gov) == ["doclaynet_gov", "us_gov_ocr"]
    assert out._resolve_cat5_suites("public-nuextract3-mlx-vlm", ns_gov) == [
        "vrdu_ad_buy",
        "vrdu_registration",
        "doclaynet_gov",
        "us_gov_ocr",
    ]

    ns_gov_all = out._parse_extra_flags(["--ocr-suites", "gov-all"])
    assert out._resolve_cat5_suites("fast-ocr", ns_gov_all) == [
        "vrdu_ad_buy",
        "vrdu_registration",
        "doclaynet_gov",
        "us_gov_ocr",
    ]
    assert out._resolve_cat5_suites("public-pypdfium2", ns_gov_all) == ["doclaynet_gov", "us_gov_ocr"]

    ns_pdb = out._parse_extra_flags(
        ["--ocr-suites", "puredocbench_legal_gov", "--puredocbench-track", "real", "--puredocbench-images-root", "/tmp/pdb"]
    )
    assert out._resolve_cat5_suites("fast-ocr", ns_pdb) == ["puredocbench_legal_gov"]
    assert ns_pdb.puredocbench_track == "real"
    assert ns_pdb.puredocbench_images_root == "/tmp/pdb"

    parser = out.build_parser()
    parsed = parser.parse_args(
        [
            "stage-gov",
            "--branches",
            "legislative,executive",
            "--per-source",
            "2",
            "--year",
            "2025",
            "--loc-btp-path",
            "/tmp/btp.parquet",
        ]
    )
    assert parsed.cmd == "stage-gov"
    assert parsed.branches == "legislative,executive"
    assert parsed.per_source == 2
    assert parsed.year == 2025
    assert parsed.loc_btp_path == "/tmp/btp.parquet"


def test_cat5_optional_all_skipping_and_pypdfium_gov_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _outcome_public_backends as opb

    out = _load_script("test_gov_worker_cpu", "benchmark_outcomes.py")
    # Stub heavy/network core loaders so `--ocr-suites all` exercises optional skip paths cleanly on CPU.
    monkeypatch.setattr(out, "load_olmocr_bench", lambda **kw: [])
    monkeypatch.setattr(out, "load_fintabnet_otsl", lambda **kw: [])
    monkeypatch.setattr(out, "load_omnidocbench_en", lambda **kw: [])
    monkeypatch.setattr(out, "load_cord_v2", lambda **kw: [])
    monkeypatch.setattr(out, "load_vrdu", lambda *a, **kw: [])
    monkeypatch.setattr(
        out,
        "load_doclaynet_gov",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("missing doclaynet_gov shard")),
    )
    monkeypatch.setattr(
        out,
        "load_puredocbench_legal_gov",
        lambda **kw: [{"id": "pdb_0", "image_path": None, "text_blocks": [], "tables": [], "equations": []}],
    )
    monkeypatch.setattr(
        out,
        "load_us_gov_ocr",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("missing manifest.json")),
    )

    recs = out._run_cat5_worker(
        "fast-ocr",
        tier="tier1",
        config_label="cpu_smoke",
        extra=out._parse_extra_flags(["--ocr-suites", "all"]),
        lock={"datasets": {}},
        cache_dir=tmp_path,
        local_files_only=True,
        max_items=2,
        smoke=True,
        home=tmp_path,
        host_mem_gb=48.0,
    )
    assert recs == []

    # Explicit `--ocr-suites puredocbench_legal_gov` must fail loudly when images are missing.
    with pytest.raises(FileNotFoundError, match="PureDocBench"):
        out._run_cat5_worker(
            "fast-ocr",
            tier="tier1",
            config_label="cpu_smoke",
            extra=out._parse_extra_flags(["--ocr-suites", "puredocbench_legal_gov"]),
            lock={"datasets": {}},
            cache_dir=tmp_path,
            local_files_only=True,
            max_items=2,
            smoke=True,
            home=tmp_path,
            host_mem_gb=48.0,
        )

    # `public-pypdfium2` (`supports_images=False`) must skip image-only `loc_btp` entries inside `us_gov_ocr`.
    fake_pdf = tmp_path / "fr.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    fake_img = tmp_path / "scan.jpg"
    fake_img.write_bytes(b"jpg")
    monkeypatch.setattr(
        out,
        "load_us_gov_ocr",
        lambda **kw: [
            {
                "id": "fr:2026-1",
                "branch": "executive",
                "source": "govinfo_fr",
                "pdf_path": str(fake_pdf),
                "image_path": None,
                "page_index": 0,
                "text_blocks": [{"category_type": "text_block", "order": 0, "text": "Federal rule notice text."}],
                "tables": [],
                "equations": [],
            },
            {
                "id": "loc_btp:a1",
                "branch": "executive",
                "source": "loc_btp",
                "pdf_path": None,
                "image_path": str(fake_img),
                "text_blocks": [{"category_type": "text_block", "order": 0, "text": "Scanned letter text."}],
                "tables": [],
                "equations": [],
            },
        ],
    )
    pdf_calls: list[str] = []
    img_calls: list[Any] = []

    def _fake_pdf_fn(pdf_path: str | Path, **kw: Any) -> tuple[str, str, dict[str, Any]]:
        pdf_calls.append(str(pdf_path))
        return "Federal rule notice text.", "stop", {"generated_tokens": 5}

    def _fake_img_fn(*args: Any, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
        img_calls.append(args)
        raise AssertionError("ocr_image_fn must not be called for supports_images=False")

    fake_ad = SimpleNamespace(
        package="public-pypdfium2",
        default_label="pypdfium2",
        model_family="pypdfium2-text",
        supports_images=False,
        supports_pdf_pages=True,
        ocr_image_fn=_fake_img_fn,
        ocr_pdf_page_fn=_fake_pdf_fn,
        close=lambda: None,
    )
    monkeypatch.setattr(opb, "build_public_cat5_adapter", lambda *a, **kw: fake_ad)
    monkeypatch.setattr(opb, "stock_hf_model_files", lambda *a, **kw: [])

    pdf_recs = out._run_cat5_worker(
        "public-pypdfium2",
        tier="tier1",
        config_label="pypdfium2",
        extra=out._parse_extra_flags(["--ocr-suites", "us_gov_ocr"]),
        lock={"datasets": {}},
        cache_dir=tmp_path,
        local_files_only=True,
        max_items=4,
        smoke=False,
        home=tmp_path,
        host_mem_gb=48.0,
    )
    assert len(pdf_recs) == 1
    assert pdf_recs[0]["benchmark"] == "US-Gov-OCR(1)"
    assert len(pdf_calls) == 1 and img_calls == []




