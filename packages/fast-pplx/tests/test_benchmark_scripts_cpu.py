"""CPU unit tests for `packages/fast-pplx/scripts/benchmark.py` and `scripts/benchmark_monorepo.py`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PPLX_BENCH_PATH = _REPO_ROOT / "packages" / "fast-pplx" / "scripts" / "benchmark.py"
_MONO_BENCH_PATH = _REPO_ROOT / "scripts" / "benchmark_monorepo.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pplx_benchmark_dry_run_and_helpers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_module("test_pplx_bench_mod", _PPLX_BENCH_PATH)

    # 1. --dry-run executes cleanly on CPU without loading weights/GPU/ANE
    rc = mod.main(["speed", "--dry-run", "--modes", "gpu,hetero2"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["modes"] == ["gpu", "hetero2"]

    rc_all = mod.main(["all", "--dry-run"])
    assert rc_all == 0
    payload_all = json.loads(capsys.readouterr().out)
    assert payload_all["suite"] == "all"
    assert payload_all["context_limits"] == [512, 2048, 8192, 16384, 32768]

    # 2. Vector & int8 parity computations
    ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    vp = mod.compute_vector_parity(ref + 1e-4, ref)
    assert vp["min_cos"] >= 0.9999
    assert vp["max_abs_err"] == pytest.approx(1e-4, abs=1e-6)

    i8_ref = np.array([[10, -20, 30]], dtype=np.int8)
    ip = mod.compute_int8_parity(np.array([[10, -19, 30]], dtype=np.int8), i8_ref)
    assert ip["max_abs_err"] == 1
    assert ip["exact_fraction"] == pytest.approx(2.0 / 3.0)

    # 3. Synthetic context corpus + retrieval metric scoring
    docs, queries = mod.build_synthetic_context_corpus(filler_repeats=3)
    assert len(docs) == len(queries) >= 6
    assert all(len(d["chunks"]) == 5 for d in docs)

    q_vecs = np.eye(len(queries), dtype=np.float32)
    c_vecs = np.repeat(q_vecs, 2, axis=0)
    c_doc_ids = [d["doc_id"] for d in docs for _ in range(2)]
    scores = mod.score_retrieval_metrics(q_vecs, c_vecs, c_doc_ids, [q["doc_id"] for q in queries])
    assert scores["mrr"] == 1.0
    assert scores["recall_at_1"] == 1.0
    assert scores["ndcg_at_5"] == 1.0

    # 4. Markdown formatters
    md_speed = mod.format_speed_markdown(
        [
            {
                "label": "gpu query B=1",
                "status": "ok",
                "tok_s": 420.5,
                "latency_ms": 16.65,
                "peak_active_gb": 1.25,
                "peak_footprint_gb": 1.48,
                "rss_gb": 0.41,
                "notes": "load=0.80s",
            }
        ]
    )
    assert "OS Footprint (GB)" in md_speed
    assert "gpu query B=1" in md_speed

    md_lim = mod.format_context_limit_markdown(
        [
            {
                "target_tokens": 32768,
                "actual_tokens": 32768,
                "elapsed_s": 2.5,
                "tok_s": 13107.2,
                "peak_active_gb": 1.9,
                "peak_footprint_gb": 2.3,
                "status": "ok",
            }
        ]
    )
    assert "32,768" in md_lim

    # 5. Bit-identity dump & exact vs drifted comparison CLI
    base_dir = tmp_path / "base"
    same_dir = tmp_path / "same"
    diff_dir = tmp_path / "diff"
    sample = {"query_single/projected": ref}
    mod.dump_bit_identity(sample, {"mode": "gpu"}, base_dir)
    mod.dump_bit_identity(sample, {"mode": "gpu"}, same_dir)
    mod.dump_bit_identity({"query_single/projected": ref + 0.01}, {"mode": "gpu"}, diff_dir)

    assert mod.main(["bit-identity", "compare", "--baseline-dir", str(base_dir), "--candidate-dir", str(same_dir)]) == 0
    _ = capsys.readouterr()
    assert mod.main(["bit-identity", "compare", "--baseline-dir", str(base_dir), "--candidate-dir", str(diff_dir)]) == 1
    diff_out = json.loads(capsys.readouterr().out)
    assert diff_out["all_bit_identical"] is False

    # 6. Baseline annotators (speed, context-limit, context-windows)
    s_rows = mod.annotate_speed_comparisons(
        [
            {"workload": "doc_s512", "label": "gpu-no-mmap doc S=512", "mode": "gpu-no-mmap", "status": "ok", "tok_s": 5000.0, "latency_ms": 10.0, "peak_footprint_gb": 2.4},
            {"workload": "doc_s512", "label": "gpu-bf16 doc S=512", "mode": "gpu-bf16", "status": "ok", "tok_s": 4000.0, "latency_ms": 12.0, "peak_footprint_gb": 1.6},
            {"workload": "doc_s512", "label": "gpu doc S=512", "mode": "gpu", "status": "ok", "tok_s": 8000.0, "latency_ms": 6.0, "peak_footprint_gb": 1.4},
            {"workload": "doc_s512", "label": "torch-cpu doc S=512", "mode": "torch-cpu", "status": "ok", "tok_s": 1000.0, "latency_ms": 48.0, "peak_footprint_gb": 3.0},
        ]
    )
    gpu_row = next(r for r in s_rows if r["mode"] == "gpu")
    assert gpu_row["vs_gpu_bf16"] == 2.0
    assert gpu_row["vs_torch_cpu"] == 8.0
    assert gpu_row["footprint_delta_vs_no_mmap_gb"] == pytest.approx(-1.0)
    assert "2.00x vs bf16" in mod.format_speed_markdown(s_rows)

    cl_rows = mod.annotate_context_limit_comparisons(
        [
            {"target_tokens": 512, "actual_tokens": 512, "elapsed_s": 0.1, "tok_s": 5000.0, "peak_footprint_gb": 1.2, "status": "ok"},
            {"target_tokens": 32768, "actual_tokens": 32768, "elapsed_s": 8.0, "tok_s": 4000.0, "peak_footprint_gb": 1.8, "status": "ok"},
        ]
    )
    assert cl_rows[1]["footprint_delta_vs_base_gb"] == pytest.approx(0.6)
    assert cl_rows[1]["tok_s_ratio_vs_base"] == pytest.approx(0.8)

    cw_rows = mod.annotate_context_window_comparisons(
        [
            {"window_budget": "isolated", "windows": 40, "tok_s": 3000.0, "mrr": 0.4, "recall_at_1": 0.3, "ndcg_at_5": 0.45},
            {"window_budget": "2048", "windows": 12, "tok_s": 6000.0, "mrr": 0.9, "recall_at_1": 0.85, "ndcg_at_5": 0.92},
        ]
    )
    assert cw_rows[1]["delta_mrr_vs_isolated"] == pytest.approx(0.5)
    assert "+0.5000" in mod.format_context_windows_markdown(cw_rows)


def test_monorepo_benchmark_dry_run_and_parsers(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_module("test_mono_bench_mod", _MONO_BENCH_PATH)

    rc = mod.main(["--dry-run", "--packages", "fast-pplx,fast-gemma4,fast-decision-models", "--include-verify"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert [p["name"] for p in plan["packages"]] == ["fast-pplx", "fast-gemma4", "fast-decision-models"]
    assert plan["packages"][0]["verify_args"]
    assert "no_kv_i8" in plan["packages"][1]["baseline_runs"]
    assert "bf16_b8" in plan["packages"][2]["baseline_runs"]

    with pytest.raises(ValueError, match="Unknown package"):
        mod.select_specs("not-a-package")

    raw_out = "Throughput = 12345.6 tok/s\n__MONOREPO_METRICS__=" + json.dumps(
        {
            "rc": 0,
            "elapsed_s": 1.5,
            "mlx_peak_gb": 1.25,
            "peak_footprint_gb": 1.55,
            "phys_footprint_gb": 1.4,
            "rss_gb": 0.4,
            "error": None,
        }
    )
    metrics, clean = mod.parse_child_metrics(raw_out)
    assert metrics is not None and metrics["peak_footprint_gb"] == 1.55
    assert "12345.6 tok/s" in mod.summarize_throughput("fast-bge-small", clean)
    assert "41.72 q/s" in mod.summarize_throughput(
        "fast-decision-models", json.dumps({"laya_ane": {"q_per_sec": 41.72, "ms_per_question": 23.97}})
    )
    assert "462.8 dec tok/s" in mod.summarize_throughput(
        "fast-gpt-oss", json.dumps({"batched_B128": {"decode_tok_s": 462.8, "docs_per_sec": 0.71}})
    )
    assert "hetero 200,000 vs gpu 120,000 tok/s (1.67x)" in mod.extract_baseline_comparison(
        "fast-bge-small",
        {
            "status": "ok",
            "stdout_full": "Query [ANE_B1_S24]: median=0.500ms min=0.48ms\nQuery [GPU_B1_S24]: median=2.000ms min=1.9ms\n"
            "Corpus n=1,000 toks=200,000 [mode=gpu]: 1.667s -> 120,000 tok/s\n"
            "Corpus n=1,000 toks=200,000 [mode=hetero]: 1.000s -> 200,000 tok/s\n",
        },
        {},
    )
    assert "batched_B128 vs B=1: 6.48x tok/s" in mod.extract_baseline_comparison(
        "fast-gpt-oss",
        {
            "status": "ok",
            "parsed_json": {
                "b1_seq": {"decode_tok_s": 71.4, "docs_per_sec": 0.2},
                "batched_B128": {"decode_tok_s": 462.8, "docs_per_sec": 0.71},
            },
        },
        {},
    )
    assert "w8 B=8 vs bf16 B=8: 2.00x; dFootprint=-0.500GB" in mod.extract_baseline_comparison(
        "fast-decision-models",
        {"status": "ok", "peak_footprint_gb": 1.1, "parsed_json": {"kev_08b": {"q_per_sec": 60.0}}},
        {"bf16_b8": {"status": "ok", "peak_footprint_gb": 1.6, "parsed_json": {"kev_08b": {"q_per_sec": 30.0}}}},
    )
    assert "kv_i8 mem=-1.20GB" in mod.extract_baseline_comparison(
        "fast-gemma4",
        {"status": "ok", "stdout_full": "[GPUOnly B=64] (770 tok/s) 2.10 docs/s peak=6.80GB\n[Concurrent ANE+GPU 1xB=64] (900 tok/s) 2.52 docs/s peak=6.80GB"},
        {"no_kv_i8": {"status": "ok", "stdout_full": "[GPUOnly B=64] (750 tok/s) 2.00 docs/s peak=8.00GB"}},
    )
    assert any(p["resolved_home"] for p in plan["packages"])
    assert "vs pplx-ctx-mlx: 1.50x" in mod.extract_public_comparison(
        "fast-pplx",
        {
            "status": "ok",
            "parsed_json": {
                "rows": [
                    {"workload": "doc_s512", "mode": "gpu", "status": "ok", "tok_s": 9000.0},
                    {"workload": "doc_s512", "mode": "pplx-context-mlx-f16", "status": "ok", "tok_s": 6000.0},
                ]
            },
        },
        None,
    )
    assert "vs stock-mlx_vlm: 10.00x" in mod.extract_public_comparison(
        "fast-gemma4",
        {"status": "ok", "stdout_full": "[Concurrent ANE+GPU 1xB=64] (900 tok/s) 2.50 docs/s"},
        {"status": "ok", "runtime": "stock-mlx_vlm", "docs_per_sec": 0.25},
    )
    assert "vs stock-mlx_lm: 4.00x" in mod.extract_public_comparison(
        "fast-gpt-oss",
        {"status": "ok", "parsed_json": {"batched_B128": {"decode_tok_s": 400.0}}},
        {"status": "ok", "runtime": "stock-mlx_lm", "decode_tok_s": 100.0},
    )
    assert "vs transformers-mps-bf16: 3.00x" in mod.extract_public_comparison(
        "fast-decision-models",
        {"status": "ok", "peak_footprint_gb": 1.1, "parsed_json": {"kev_08b": {"q_per_sec": 60.0}}},
        {"status": "ok", "runtime": "transformers-mps-bf16", "q_per_sec": 20.0, "peak_footprint_gb": 2.1},
    )
    assert mod.classify_failure("", "FileNotFoundError: missing snapshot", None) == "skipped"
    assert mod.classify_failure("", "AssertionError: parity mismatch", None) == "failed"

    g4_summary = mod.summarize_throughput(
        "fast-gemma4",
        "[GPUOnly B=128] (1265 tok/s) 7.36 docs/s peak=6.718GB\n"
        "[PromptLookup B=32] (516 eff tok/s) 2.50 docs/s peak=6.718GB\n"
        "[Concurrent ANE+GPU 2xB=128] (1192 tok/s) 8.92 docs/s peak=6.718GB",
    )
    assert "GPU: 1,265 tok/s (7.36 docs/s)" in g4_summary
    assert "Spec: 516 eff tok/s" in g4_summary
    assert "ANE+GPU: 1,192 tok/s (8.92 docs/s)" in g4_summary

    pub_dec = mod.extract_public_comparison(
        "fast-decision-models",
        {
            "status": "ok",
            "peak_footprint_gb": 1.1,
            "parsed_json": {
                "kev_08b": {"q_per_sec": 60.0},
                "laya_ane": {"q_per_sec": 40.0},
            },
        },
        {"status": "ok", "runtime": "transformers-mps-bf16", "q_per_sec": 20.0, "peak_footprint_gb": 2.1},
    )
    assert "vs transformers-mps-bf16: 3.00x" in pub_dec
    assert "laya_ane 2.00x" in pub_dec

    import inspect

    pplx_bench = _load_module("test_pplx_bench_cw_check", _PPLX_BENCH_PATH)
    cw_src = inspect.getsource(pplx_bench.cmd_context_windows)
    assert "group_chunks_by_context_windows" in cw_src
    assert "_pack_contextual_windows" not in cw_src

    # Verify telemetry, gpu-bf16 + S=4096 parity defaults, symmetric timing scope, and no forced --allow-contended
    rc_par = pplx_bench.main(["parity", "--dry-run"])
    assert rc_par == 0
    par_payload = json.loads(capsys.readouterr().out)
    assert "gpu-bf16" in par_payload["modes"]
    assert par_payload["include_s4096"] is True
    assert "mlx_enable_tf32" in par_payload["telemetry"]

    rc_spd = pplx_bench.main(["speed", "--dry-run", "--modes", "gpu,hetero2"])
    assert rc_spd == 0
    spd_payload = json.loads(capsys.readouterr().out)
    assert spd_payload["timing_scope"] == "e2e"
    assert "mlx_enable_tf32" in spd_payload["telemetry"]

    spd_src = inspect.getsource(pplx_bench.cmd_speed)
    clim_src = inspect.getsource(pplx_bench.cmd_context_limit)
    assert "if args.allow_contended:" in spd_src
    assert "if args.allow_contended:" in clim_src
    assert "interleaved_abba" in inspect.getsource(pplx_bench.run_abba_speed_pair)
