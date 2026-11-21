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
        monkeypatch.setattr(pub.subprocess, "run", lambda *a, **kw: completed)
        rc = pub.main(["--package", "coreml-laya"])
        result = json.loads(capsys.readouterr().out)
        assert result["runtime"] == expected
        assert rc == (0 if expected == "unavailable" else 1)


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
