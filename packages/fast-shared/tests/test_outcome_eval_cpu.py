"""Offline pure-CPU unit tests for ``fast_shared.outcome_eval`` and ``scripts/benchmark_outcomes.py``."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq
import pytest

from fast_shared.outcome_eval import (
    CLINC150_DOMAINS,
    CLINC150_INTENT_TO_DOMAIN,
    CLINC_STAGE1_DOMAINS,
    INSTRUCTION_VERIFIERS,
    build_bfcl_response_schema,
    build_clinc_stage1_qdef,
    build_clinc_stage2_qdef,
    check_bfcl_ast_match,
    check_chat_validity,
    check_embedding_validity,
    check_probability_validity,
    chunk_document_char_spans,
    compute_auroc,
    compute_auroc_delong_se,
    compute_binary_macro_f1,
    compute_brier_score,
    compute_contextual_lift,
    compute_corpus_outcome_metrics,
    compute_ece,
    compute_file_sha256,
    compute_multiclass_nll,
    compute_outcome_metrics,
    compute_paired_se,
    dcg_at_k,
    download_pinned_file,
    evaluate_bfcl_v3,
    evaluate_clinc_two_stage,
    evaluate_instruction_following,
    evaluate_jsonschemabench_compiler,
    evaluate_legalbench_spans,
    evaluate_livebench_reasoning,
    evaluate_math_500,
    evaluate_memory_gate,
    evaluate_mmlu_redux,
    evaluate_mrl_dtype_grid,
    evaluate_promotion_gates,
    evaluate_ragtruth_noul,
    evaluate_retrieval_rankings,
    evaluate_reward_bench_2,
    evaluate_structured_extraction,
    expected_reward_score,
    extract_boxed_answer,
    extract_mmlu_choice,
    filter_legalbench_subsets,
    filter_longembed_subsets,
    grounded_field_f1,
    hungarian_min_cost,
    linear_sum_assignment_max,
    load_lockfile,
    math_answers_equivalent,
    merge_char_spans,
    parse_bfcl_tool_calls,
    partition_banking77_clusters,
    project_mrl_dtype,
    read_arrow_ipc_rows,
    read_jsonl_rows,
    read_parquet_rows,
    repeat_4gram_rate,
    split_sentences,
    stratified_sha256_sample,
    validate_against_jsonschema,
    verify_instruction,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. Lockfile, Pure-Arrow IO & SHA-256 Stratified Sampling
# ---------------------------------------------------------------------------


def test_lockfile_pins_all_16_datasets_and_excludes_restricted_subsets(tmp_path: Path) -> None:
    lock_p = _REPO_ROOT / "configs" / "outcome_benchmarks.lock.json"
    if not lock_p.is_file():
        # Standalone fast-shared mirror without monorepo configs/: verify load_lockfile contract on synthetic lockfile.
        synth_p = tmp_path / "outcome_benchmarks.lock.json"
        synth_p.write_text(json.dumps({"schema_version": 1, "datasets": {"demo": {"revision": "a" * 40}}}), encoding="utf-8")
        lock, sha = load_lockfile(synth_p)
        assert len(sha) == 64 and "demo" in lock["datasets"]
        return
    lock, sha = load_lockfile(lock_p)
    assert len(sha) == 64
    ds = lock["datasets"]
    expected_keys = {
        "nanobeir_en",
        "legalbench_rag",
        "longembed",
        "ragtruth",
        "clinc_oos",
        "banking77",
        "reward_bench_2",
        "ifbench_test",
        "ifeval",
        "mmlu_redux_2",
        "math_500",
        "livebench_reasoning",
        "liquidai_extraction",
        "bfcl_v3",
        "jsonschemabench",
        "extractbench",
    }
    assert set(ds.keys()) == expected_keys
    for key, entry in ds.items():
        assert len(entry["revision"]) == 40, f"{key} missing 40-char commit SHA"
        assert entry["gated"] is False, f"{key} must be un-gated"
        assert entry.get("license"), f"{key} missing permissive license tag"
        assert entry.get("pinned_files"), f"{key} missing pinned_files"

    assert "maud" in ds["legalbench_rag"]["excluded_subsets"]
    assert "maud" not in ds["legalbench_rag"]["allowed_subsets"]
    assert "summ_screen_fd" in ds["longembed"]["excluded_subsets"]
    assert "summ_screen_fd" not in ds["longembed"]["allowed_subsets"]
    assert ds["mmlu_redux_2"]["error_type_filter"] == "ok"
    assert len(ds["mmlu_redux_2"]["subjects"]) == 57


def test_forbidden_subsets_rejected_before_network(tmp_path: Path) -> None:
    assert filter_legalbench_subsets(["cuad", "contractnli", "privacy_qa"]) == ["cuad", "contractnli", "privacy_qa"]
    with pytest.raises(ValueError, match="maud"):
        filter_legalbench_subsets(["cuad", "maud"])

    assert filter_longembed_subsets(["2wikimqa", "passkey"]) == ["2wikimqa", "passkey"]
    with pytest.raises(ValueError, match="summ_screen_fd"):
        filter_longembed_subsets(["passkey", "summ_screen_fd"])

    fake_lock = {
        "datasets": {
            "legalbench_rag": {"repo_id": "awinml/legalbench-rag", "revision": "0" * 40, "gated": False, "pinned_files": {}},
            "longembed": {"repo_id": "dwzhu/LongEmbed", "revision": "0" * 40, "gated": False, "pinned_files": {}},
        }
    }
    with pytest.raises(ValueError, match="maud"):
        download_pinned_file("legalbench_rag", "benchmarks/maud.json", lock=fake_lock, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="summ_screen_fd"):
        download_pinned_file("longembed", "summ_screen_fd/corpus.jsonl", lock=fake_lock, cache_dir=tmp_path)


def test_pure_arrow_parquet_ipc_jsonl_and_checksum(tmp_path: Path) -> None:
    tbl = pa.table({"id": ["a", "b"], "val": [10, 20]})
    pq_path = tmp_path / "sample.parquet"
    pq.write_table(tbl, pq_path)
    assert read_parquet_rows(pq_path) == [{"id": "a", "val": 10}, {"id": "b", "val": 20}]

    arrow_path = tmp_path / "sample.arrow"
    with arrow_path.open("wb") as fp:
        with pa_ipc.new_stream(fp, tbl.schema) as writer:
            writer.write_table(tbl)
    assert read_arrow_ipc_rows(arrow_path) == [{"id": "a", "val": 10}, {"id": "b", "val": 20}]

    jl_path = tmp_path / "sample.jsonl"
    jl_path.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    assert read_jsonl_rows(jl_path) == [{"x": 1}, {"x": 2}]

    good_sha = compute_file_sha256(pq_path)
    cache_ds = tmp_path / "cache" / "demo"
    cache_ds.mkdir(parents=True)
    cached_file = cache_ds / "sample.parquet"
    cached_file.write_bytes(pq_path.read_bytes())

    fake_lock = {
        "datasets": {
            "demo": {
                "repo_id": "org/demo",
                "revision": "a" * 40,
                "gated": False,
                "pinned_files": {"sample.parquet": {"sha256": good_sha}},
            }
        }
    }
    resolved = download_pinned_file("demo", "sample.parquet", lock=fake_lock, cache_dir=tmp_path / "cache")
    assert resolved == cached_file

    fake_lock["datasets"]["demo"]["pinned_files"]["sample.parquet"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_pinned_file("demo", "sample.parquet", lock=fake_lock, cache_dir=tmp_path / "cache")


def test_stratified_sha256_sampling_is_order_invariant_and_balanced() -> None:
    items = [{"id": f"item_{i}", "grp": f"g{i % 4}"} for i in range(40)]
    s1 = stratified_sha256_sample(items, dataset_key="ds", id_fn=lambda r: r["id"], stratum_fn=lambda r: r["grp"], total_n=12)
    s2 = stratified_sha256_sample(list(reversed(items)), dataset_key="ds", id_fn=lambda r: r["id"], stratum_fn=lambda r: r["grp"], total_n=12)
    assert [x["id"] for x in s1] == [x["id"] for x in s2]
    assert len(s1) == 12
    counts = {g: sum(1 for x in s1 if x["grp"] == g) for g in ("g0", "g1", "g2", "g3")}
    assert counts == {"g0": 3, "g1": 3, "g2": 3, "g3": 3}

    per_s = stratified_sha256_sample(items, dataset_key="ds", id_fn=lambda r: r["id"], stratum_fn=lambda r: r["grp"], per_stratum=2)
    assert len(per_s) == 8


# ---------------------------------------------------------------------------
# 2. Composite EOT & Two-Gate Promotion Math
# ---------------------------------------------------------------------------


def test_eot_avoids_vrate_squared_trap_and_checks_identities() -> None:
    # 4 items, 2 valid (scores 0.8 and 0.6), 2 invalid (raw scores 1.0 and 0.9 must be zeroed!).
    valid = [True, True, False, False]
    scores = [0.8, 0.6, 1.0, 0.9]
    wall = 2.0
    res = compute_outcome_metrics(valid, scores, wall)

    assert res.n_items == 4
    assert res.n_valid == 2
    assert res.valid_rate == pytest.approx(0.5)
    assert res.objective_score_given_valid == pytest.approx(0.7)
    assert res.objective_score_e2e == pytest.approx(0.35)
    # Exact EOT = (0.8 + 0.6) / 2.0 = 0.70 (NOT valid_ops * s_e2e = 1.0 * 0.35 = 0.35!).
    assert res.effective_outcome_per_sec == pytest.approx(0.70)
    assert res.effective_outcome_per_sec == pytest.approx(res.valid_outputs_per_sec * res.objective_score_given_valid)
    assert res.effective_outcome_per_sec == pytest.approx(res.raw_items_per_sec * res.objective_score_e2e)

    corp = compute_corpus_outcome_metrics(n_items=100, n_valid=80, corpus_score_given_valid=0.85, wall_sec=4.0)
    assert corp.objective_score_e2e == pytest.approx(0.68)
    assert corp.effective_outcome_per_sec == pytest.approx((80 / 4.0) * 0.85)
    assert corp.effective_outcome_per_sec == pytest.approx((100 / 4.0) * 0.68)


def test_paired_se_and_two_gate_promotion_contract() -> None:
    rng = np.random.default_rng(7)
    base_items = rng.uniform(0.3, 0.95, size=200)
    # Tightly correlated candidate (+0.002 shift with tiny noise)
    cand_items = np.clip(base_items + 0.002 + rng.normal(0.0, 0.001, size=200), 0.0, 1.0)
    paired = compute_paired_se(base_items, cand_items)
    # Closed-form check: sqrt(sum((d - mean)^2) / (N*(N-1)))
    diff = cand_items - base_items
    expected_se = float(np.sqrt(np.sum((diff - diff.mean()) ** 2) / (200 * 199)))
    assert paired["se_paired"] == pytest.approx(expected_se, rel=1e-12)
    assert paired["se_paired"] < 0.001

    base_rec = {
        "objective_score_e2e": 0.850,
        "effective_outcome_per_sec": 10.0,
        "diagnostics": {"mask_fallback_rate": 0.001, "repeat_loop_count": 0},
        "memory_4bucket_gb": {"combined_with_model_mapped_gb": 12.0},
    }
    # Lossless PASS
    cand_pass = {
        "objective_score_e2e": 0.849,
        "effective_outcome_per_sec": 10.5,
        "diagnostics": {"mask_fallback_rate": 0.001, "repeat_loop_count": 0},
        "memory_4bucket_gb": {"combined_with_model_mapped_gb": 12.5},
    }
    g_pass = evaluate_promotion_gates(
        base_rec, cand_pass, profile="lossless", min_eot_ratio=1.03, host_mem_gb=48.0,
        base_effective_scores=[0.85, 0.85], cand_effective_scores=[0.849, 0.849],
    )
    assert g_pass.verdict == "PASS"

    # Lossless FAIL on repeat loop
    cand_loop = dict(cand_pass, diagnostics={"mask_fallback_rate": 0.001, "repeat_loop_count": 1})
    assert evaluate_promotion_gates(base_rec, cand_loop, profile="lossless", host_mem_gb=48.0).verdict == "FAIL"

    # Quantized PASS within -1.0 pp & fallback <= max(0.005, 2*base)
    cand_q_ok = {
        "objective_score_e2e": 0.842,
        "effective_outcome_per_sec": 12.0,
        "diagnostics": {"mask_fallback_rate": 0.004, "repeat_loop_count": 0},
        "memory_4bucket_gb": {"combined_with_model_mapped_gb": 10.0},
    }
    assert evaluate_promotion_gates(base_rec, cand_q_ok, profile="quantized", host_mem_gb=48.0).verdict == "PASS"

    # Quantized FAIL on mask_fallback_rate explosion (grammar-masked collapse!)
    cand_q_fb = dict(cand_q_ok, diagnostics={"mask_fallback_rate": 0.025, "repeat_loop_count": 0})
    g_fb = evaluate_promotion_gates(base_rec, cand_q_fb, profile="quantized", host_mem_gb=48.0)
    assert g_fb.verdict == "FAIL" and not g_fb.gate1_quality_pass

    # Memory Gate FAIL (> 80% of 24 GB = 19.2 GB)
    cand_oom = dict(cand_q_ok, memory_4bucket_gb={"combined_with_model_mapped_gb": 19.5})
    g_oom = evaluate_promotion_gates(base_rec, cand_oom, profile="quantized", host_mem_gb=24.0)
    assert g_oom.verdict == "FAIL" and not g_oom.gate2_memory_pass

    mem_ok, ceil_gb, _, _ = evaluate_memory_gate(
        {"combined_with_model_mapped_gb": 10.0}, host_mem_gb=48.0, device_guard_tripped=True
    )
    assert not mem_ok and ceil_gb == pytest.approx(38.4)


# ---------------------------------------------------------------------------
# 3. Category 1: Retrieval, Stable Tie-Breaking, MRL Grid & LegalBench Spans
# ---------------------------------------------------------------------------


def test_lossless_gate_uses_fixed_tolerance_and_requires_paired_evidence() -> None:
    base = {"objective_score_e2e": 0.5, "effective_outcome_per_sec": 1.0, "n_items": 4}
    cand = dict(base, effective_outcome_per_sec=2.0)

    def gate(b, c, **kwargs):
        return evaluate_promotion_gates(base, cand, base_effective_scores=b, cand_effective_scores=c, **kwargs)

    wide = gate([0, 1, 0, 1], [1, 0, 1, 0])
    assert wide.quality_floor == -0.003
    assert wide.verdict == wide.quality_status == "UNRESOLVED"
    assert wide.ci95_low < wide.quality_floor < wide.ci95_high
    assert wide.n_paired == 4
    for b, c in ((None, None), ([0.5], [0.5]), ([0.5] * 2, [0.5] * 2)):
        unknown = gate(b, c)
        assert unknown.quality_status == "UNRESOLVED" and unknown.ci95_low is None
    assert gate([0, 1, 0, 1], [0, 1, 0, 1]).verdict == "PASS"
    cand["objective_score_e2e"] = 0.497
    assert gate([0.5] * 4, [0.497] * 4).verdict == "PASS"
    cand["objective_score_e2e"] = 0.495
    assert gate([0.5] * 4, [0.495] * 4).quality_status == "FAIL"
    assert gate([0.5] * 4, [0.495] * 4, lossless_tolerance=0.01).verdict == "PASS"

    # Mean-mismatch between per_item_effective_scores and objective_score_e2e reports an explicit diagnostic reason.
    bad_base = {
        "objective_score_e2e": 0.5,
        "effective_outcome_per_sec": 1.0,
        "n_items": 4,
        "per_item_effective_scores": [0.8, 0.8, 0.8, 0.8],
    }
    good_cand = {
        "objective_score_e2e": 0.5,
        "effective_outcome_per_sec": 2.0,
        "n_items": 4,
        "per_item_effective_scores": [0.5, 0.5, 0.5, 0.5],
    }
    mismatch = evaluate_promotion_gates(bad_base, good_cand, profile="lossless")
    assert mismatch.verdict == mismatch.quality_status == "UNRESOLVED"
    assert mismatch.n_paired == 0 and mismatch.ci95_low is None
    assert any("baseline paired item scores mean (0.800000) disagrees with objective_score_e2e (0.500000)" in r for r in mismatch.reasons)
    assert not any("complete paired scores for at least two items" in r for r in mismatch.reasons)

    mismatch_both = evaluate_promotion_gates(
        bad_base,
        dict(good_cand, per_item_effective_scores=[0.2, 0.2, 0.2, 0.2]),
        profile="lossless",
    )
    joined = " | ".join(mismatch_both.reasons)
    assert "baseline paired item scores mean (0.800000) disagrees with objective_score_e2e (0.500000)" in joined
    assert "candidate paired item scores mean (0.200000) disagrees with objective_score_e2e (0.500000)" in joined

    mismatch_quant = evaluate_promotion_gates(bad_base, good_cand, profile="quantized")
    assert mismatch_quant.verdict == "PASS"
    assert any("disagrees with objective_score_e2e" in r for r in mismatch_quant.reasons)


def test_retrieval_coverage_blocks_promotion_of_partial_historical_rows() -> None:
    complete = {"category": 1, "n_items": 325, "n_valid": 325, "objective_score_e2e": 1.0,
                "effective_outcome_per_sec": 1.0}
    partial = dict(complete, n_valid=150, gate_verdict="PASS")
    for base, candidate in ((partial, complete), (complete, partial)):
        result = evaluate_promotion_gates(base, candidate, profile="quantized")
        assert result.verdict == "FAIL" and result.retrieval_coverage_pass is False
        assert "150/325" in " ".join(result.reasons)
    assert evaluate_promotion_gates(complete, complete, profile="quantized").verdict == "PASS"


def test_retrieval_ndcg_mrr_recall_stable_ties_and_mrl_grid() -> None:
    # 3 queries (q3 has empty qrels and must be skipped), 4 docs (d1 and d2 tie identically for q1).
    q_ids = ["q1", "q2", "q_empty"]
    d_ids = ["d0", "d1", "d2", "d3"]
    q_vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    d_vecs = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],  # d0: sim 0 with q1
            [1.0, 0.0, 0.0, 0.0],  # d1: sim 1 with q1 (wins tie over d2 via mergesort!)
            [1.0, 0.0, 0.0, 0.0],  # d2: sim 1 with q1
            [0.0, 1.0, 0.0, 0.0],  # d3: sim 1 with q2
        ],
        dtype=np.float32,
    )
    qrels = {
        "q1": {"d2": 1.0},  # d1 comes first (rank 1, rel 0), d2 at rank 2 (rel 1) -> DCG = 1/log2(3)
        "q2": {"d3": 2.0},  # d3 at rank 1 (rel 2) -> nDCG = 1.0
        "q_empty": {},
    }
    res = evaluate_retrieval_rankings(q_ids, q_vecs, d_ids, d_vecs, qrels, k=10)
    assert res["n_queries"] == 2
    expected_q1_ndcg = dcg_at_k([0.0, 1.0], 10) / dcg_at_k([1.0], 10)
    assert res["per_query_ndcg"][0] == pytest.approx(expected_q1_ndcg, rel=1e-9)
    assert res["per_query_mrr"][0] == pytest.approx(0.5)
    assert res["per_query_ndcg"][1] == pytest.approx(1.0)
    assert res["recall@10"] == pytest.approx(1.0)

    # Non-unit-norm / zero-norm / NaN vector detection
    bad_vecs = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [np.nan, 0.0]], dtype=np.float32)
    assert check_embedding_validity(bad_vecs).tolist() == [True, False, False, False]
    assert check_embedding_validity(bad_vecs, expect_unit_norm=False).tolist() == [True, True, False, False]

    # Zero-recompute MRL {128, 256, 512, max} x {f32, f16, int8} grid (both sym_int8 and unnormalized native_tanh_int8)
    rng = np.random.default_rng(11)
    q_raw = rng.standard_normal((12, 768)).astype(np.float32) * 1.5
    d_raw = np.vstack([q_raw, rng.standard_normal((28, 768)).astype(np.float32) * 1.5])
    q_big = q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True)
    d_big = d_raw / np.linalg.norm(d_raw, axis=1, keepdims=True)
    big_qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(12)}

    # Guard: native_tanh_int8 MUST reject pre_normalized=True vectors!
    with pytest.raises(ValueError, match="native_tanh_int8"):
        project_mrl_dtype(q_big, dim=128, dtype_mode="int8", int8_mode="native_tanh_int8", pre_normalized=True)

    # Exact bit-for-bit parity with fast-pplx clip(rint(tanh(x[:, :dim]) * 127), -128, 127)
    proj_i8 = project_mrl_dtype(q_raw, dim=128, dtype_mode="int8", int8_mode="native_tanh_int8", pre_normalized=False)
    ref_i8 = np.clip(np.rint(np.tanh(q_raw[:, :128].astype(np.float64)) * 127.0), -128.0, 127.0).astype(np.int8).astype(np.float64)
    ref_i8 /= np.linalg.norm(ref_i8, axis=1, keepdims=True)
    assert np.allclose(proj_i8, ref_i8, atol=1e-12)

    grid_sym = evaluate_mrl_dtype_grid(
        [f"q{i}" for i in range(12)],
        q_big,
        [f"d{i}" for i in range(40)],
        d_big,
        big_qrels,
        dims=(128, 256, 512, "max"),
        dtypes=("f32", "f16", "int8"),
        int8_mode="sym_int8",
        pre_normalized=True,
        k=10,
    )
    assert len(grid_sym["cells"]) == 12
    assert grid_sym["cells"]["d768_f32"]["ndcg@10"] == pytest.approx(1.0)

    grid_tanh = evaluate_mrl_dtype_grid(
        [f"q{i}" for i in range(12)],
        q_raw,
        [f"d{i}" for i in range(40)],
        d_raw,
        big_qrels,
        dims=(128, 256, 512, "max"),
        dtypes=("f32", "f16", "int8"),
        int8_mode="native_tanh_int8",
        pre_normalized=False,
        k=10,
    )
    assert len(grid_tanh["cells"]) == 12
    assert grid_tanh["cells"]["d768_f32"]["ndcg@10"] == pytest.approx(1.0)
    assert grid_tanh["mrl_128_int8_delta_ndcg@10"] is not None


def test_pplx_benchmark_windows_restore_source_rows_and_refuse_partial_batches() -> None:
    from fast_shared.outcome_eval.retrieval import collect_pplx_window_vectors, plan_pplx_windows

    docs = [["Header", "first", "second", "third"], [], ["short"]]
    windows = plan_pplx_windows(docs, 600, lambda text: 300 if text != "Header" else 20)
    assert [(w.doc_index, w.has_anchor, w.chunks) for w in windows] == [
        (0, False, ["Header", "first"]), (0, True, ["Header", "second"]),
        (0, True, ["Header", "third"]), (2, False, ["short"]),
    ]
    values = [np.array([[1], [2]]), np.array([[10], [3]]), np.array([[11], [4]]), np.array([[5]])]
    restored = collect_pplx_window_vectors(windows, values, len(docs))
    assert restored[0][:, 0].tolist() == [1, 2, 3, 4]
    assert restored[1].shape == (0, 1) and restored[2][:, 0].tolist() == [5]
    # Infer non-1024 embedding width (e.g. 768-d) for empty docs, falling back to 1024 only when all docs are empty.
    values_768 = [
        np.ones((2, 768), dtype=np.float32),
        np.ones((2, 768), dtype=np.float32) * 2.0,
        np.ones((2, 768), dtype=np.float32) * 3.0,
        np.ones((1, 768), dtype=np.float32) * 4.0,
    ]
    restored_768 = collect_pplx_window_vectors(windows, values_768, len(docs))
    assert restored_768[0].shape == (4, 768)
    assert restored_768[1].shape == (0, 768) and restored_768[1].dtype == np.float32
    assert restored_768[2].shape == (1, 768)
    all_empty = collect_pplx_window_vectors([], [], 2)
    assert [m.shape for m in all_empty] == [(0, 1024), (0, 1024)]
    with pytest.raises(ValueError):
        collect_pplx_window_vectors(windows, values[1:], len(docs))
    with pytest.raises(ValueError, match="contextual chunk vectors"):
        collect_pplx_window_vectors(windows, [values[0][:1], *values[1:]], len(docs))


def test_retrieval_and_legalbench_reject_nan_corpus_and_chunk_embeddings() -> None:
    # Clean unit-norm queries vs corrupted (all-NaN or partially-NaN) corpus MUST fail validity (0 valid queries!).
    q_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d_nan = np.full((3, 2), np.nan, dtype=np.float32)
    qrels = {"q0": {"d0": 1.0}, "q1": {"d1": 1.0}}
    r_all_nan = evaluate_retrieval_rankings(["q0", "q1"], q_vecs, ["d0", "d1", "d2"], d_nan, qrels, k=10)
    assert r_all_nan["n_valid_queries"] == 0
    assert r_all_nan["per_query_valid"] == [False, False]
    assert r_all_nan["ndcg@10"] == pytest.approx(0.0)
    out_of_scope = evaluate_retrieval_rankings(["q0", "q1"], q_vecs, ["d0", "other"], q_vecs, qrels)
    assert out_of_scope["per_query_valid"] == [True, False]
    empty_corpus = evaluate_retrieval_rankings(["q0", "q1"], q_vecs, [], q_vecs[:0], qrels)
    assert empty_corpus["n_queries"] == 2 and empty_corpus["per_query_valid"] == [False, False]

    # Even a single NaN document corrupts the ranking matrix and invalidates queries.
    d_one_nan = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, 0.0]], dtype=np.float32)
    r_one_nan = evaluate_retrieval_rankings(["q0", "q1"], q_vecs, ["d0", "d1", "d2"], d_one_nan, qrels, k=10)
    assert r_one_nan["n_valid_queries"] == 0
    assert r_one_nan["per_query_valid"] == [False, False]
    assert r_one_nan["ndcg@10"] == pytest.approx(0.0)

    # LegalBench-RAG corrupted chunk embeddings MUST also invalidate all queries.
    chunks = chunk_document_char_spans("doc_nan", "Alpha clause text here " * 20, chunk_chars=128, overlap_chars=0)
    lb_q = [{"id": "q0", "query": "Alpha", "gold_spans": [("doc_nan", 0, 20)]}]
    c_nan = np.full((len(chunks), 2), np.nan, dtype=np.float32)
    r_lb_nan = evaluate_legalbench_spans(lb_q, chunks, q_vecs[:1], c_nan, k=10)
    assert r_lb_nan["n_valid_queries"] == 0
    assert r_lb_nan["per_query_valid"] == [False]
    assert r_lb_nan["span_ndcg@10"] == pytest.approx(0.0)
    wrong_document = [dict(lb_q[0], gold_spans=[("absent_document", 0, 20)])]
    valid_chunks = np.tile(q_vecs[0], (len(chunks), 1))
    assert evaluate_legalbench_spans(wrong_document, chunks, q_vecs[:1], valid_chunks)["per_query_valid"] == [False]
    no_chunks = evaluate_legalbench_spans(lb_q, [], q_vecs[:1], q_vecs[:0])
    assert no_chunks["n_queries"] == 1 and no_chunks["per_query_valid"] == [False]

    # project_mrl_dtype preserves non-finite rows cleanly without int8 RuntimeWarning.
    p_nan = project_mrl_dtype(np.array([[1.0, 2.0], [np.nan, 0.5]], dtype=np.float32), dim=2, dtype_mode="int8", int8_mode="sym_int8")
    assert check_embedding_validity(p_nan, expect_unit_norm=False).tolist() == [True, False]


def test_legalbench_span_ndcg_and_contextual_lift() -> None:
    doc_a = ("Alpha Party NDA boilerplate clause. " * 20) + "UNIQUE_SECRET_CLAUSE_CARDLYTICS " + ("End terms. " * 20)
    chunks = chunk_document_char_spans("doc_a", doc_a, chunk_chars=256, overlap_chars=0)
    assert len(chunks) >= 3
    idx_secret = doc_a.index("UNIQUE_SECRET_CLAUSE_CARDLYTICS")
    gold_span = (idx_secret, idx_secret + len("UNIQUE_SECRET_CLAUSE_CARDLYTICS"))
    assert merge_char_spans([(10, 25), (20, 40), (50, 60)]) == [(10, 40), (50, 60)]

    queries = [{"id": "q0", "query": "Find unique clause", "gold_spans": [("doc_a", gold_span[0], gold_span[1])]}]
    target_ci = next(i for i, ch in enumerate(chunks) if ch.start_char <= gold_span[0] < ch.end_char)

    # Isolated places target chunk at rank 2; contextual places it at rank 1.
    q_vec = np.array([[1.0, 0.0]], dtype=np.float32)
    iso_vecs = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(chunks), 1))
    iso_vecs[0] = [1.0, 0.0]  # wrong boilerplate chunk ranked #1
    iso_vecs[target_ci] = [0.70710678, 0.70710678]  # target ranked #2

    ctx_vecs = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(chunks), 1))
    ctx_vecs[target_ci] = [1.0, 0.0]  # target ranked #1!

    r_iso = evaluate_legalbench_spans(queries, chunks, q_vec, iso_vecs, k=10)
    r_ctx = evaluate_legalbench_spans(queries, chunks, q_vec, ctx_vecs, k=10)
    lift = compute_contextual_lift(r_iso, r_ctx, k=10)
    assert r_ctx["span_ndcg@10"] == pytest.approx(1.0)
    assert r_ctx["char_recall@10"] == pytest.approx(1.0)
    assert lift["contextual_delta_ndcg@10"] > 0.25


# ---------------------------------------------------------------------------
# 4. Category 2: SciPy-Free AUROC/DeLong/ECE, CLINC150 2-Stage & RewardBench-2
# ---------------------------------------------------------------------------


def test_scipy_free_auroc_ties_delong_brier_ece_and_ragtruth() -> None:
    assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert compute_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    # Tied middle pair: pairs = (0.1<0.5:1), (0.1<0.9:1), (0.5==0.5:0.5), (0.5<0.9:1) => 3.5 / 4 = 0.875
    assert compute_auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)

    # Exact verification of fast O((m+n)log(m+n)) DeLong SE against O(m*n) textbook definition
    labels = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    scores = [0.2, 0.8, 0.5, 0.5, 0.9, 0.1, 0.7, 0.4, 0.3, 0.6]
    se_fast = compute_auroc_delong_se(labels, scores)
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    v10 = np.array([np.mean([(1.0 if p > n else (0.5 if p == n else 0.0)) for n in neg]) for p in pos])
    v01 = np.array([np.mean([(1.0 if p > n else (0.5 if p == n else 0.0)) for p in pos]) for n in neg])
    se_ref = float(np.sqrt(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg)))
    assert se_fast == pytest.approx(se_ref, rel=1e-12)

    assert compute_binary_macro_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["macro_f1"] == pytest.approx(1.0)
    assert compute_brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert compute_brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    assert compute_ece([0, 1], [0.0, 1.0], n_bins=15) == pytest.approx(0.0)
    assert compute_ece([0, 0], [1.0, 1.0], n_bins=15) == pytest.approx(1.0)
    assert compute_multiclass_nll([0, 1], [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(0.0)
    assert check_probability_validity([[0.2, 0.8], [0.5, 0.6], [float("nan"), 1.0]]) == [True, False, False]

    rt_items = [{"id": str(i), "is_faithful": bool(y)} for i, y in enumerate(labels)]
    r_rt = evaluate_ragtruth_noul(rt_items, scores, token_counts=[128] * 5 + [768] * 5)
    assert "le_512" in r_rt["length_slices"] and "gt_512" in r_rt["length_slices"]


def test_clinc150_two_stage_and_rewardbench2_tie_breaking() -> None:
    assert len(CLINC150_DOMAINS) == 10
    assert all(len(v) == 15 for v in CLINC150_DOMAINS.values())
    assert len(CLINC150_INTENT_TO_DOMAIN) == 151

    # Verify concise stage-1 (K=11) and stage-2 (K=15) criteria stay compact for Laya head_max_len=192
    q1 = build_clinc_stage1_qdef()
    assert len(q1["criteria"]) == 11
    q2 = build_clinc_stage2_qdef("banking")
    assert len(q2["criteria"]) == 15

    # Verify Banking77 8-cluster partition covers all 77 labels cleanly
    clusters = partition_banking77_clusters([f"intent_{i}" for i in range(77)], n_clusters=8)
    assert len(clusters) == 8 and sum(len(c) for c in clusters) == 77 and max(len(c) for c in clusters) <= 10

    items = [
        {"id": "0", "domain": "banking", "intent_name": "balance", "is_oos": False},
        {"id": "1", "domain": "out_of_scope", "intent_name": "oos", "is_oos": True},
        {"id": "2", "domain": "travel", "intent_name": "book_flight", "is_oos": False},
    ]
    p1_bank = [0.9] + [0.01] * 10
    p1_oos = [0.01] * 10 + [0.9]
    p1_travel = [0.01] * 11
    p1_travel[CLINC_STAGE1_DOMAINS.index("travel")] = 0.9
    p2_bank = [0.0] * 15
    p2_bank[CLINC150_DOMAINS["banking"].index("balance")] = 1.0
    p2_travel_wrong = [0.0] * 15
    p2_travel_wrong[CLINC150_DOMAINS["travel"].index("book_hotel")] = 1.0

    r_cl = evaluate_clinc_two_stage(
        items,
        [p1_bank, p1_oos, p1_travel],
        [("banking", p2_bank), None, ("travel", p2_travel_wrong)],
    )
    assert r_cl["stage1_domain_acc"] == pytest.approx(1.0)
    assert r_cl["end_to_end_151_acc"] == pytest.approx(2.0 / 3.0)
    assert r_cl["oos_auroc"] == pytest.approx(1.0)

    # RewardBench-2 continuous expectation + FP16 saturated softmax tie-break via pre-softmax raw logits!
    er, tr = expected_reward_score([0.0, 0.0, 0.0, 0.0, 1.0], raw_logits=[0.0, 0.0, 0.0, 1.0, 12.0])
    assert er == pytest.approx(4.0)
    rb_items = [
        {"id": "0", "subset": "Factuality", "gold_idx": 2},
        {"id": "1", "subset": "Math", "gold_idx": 1},
    ]
    # Prompt 0: candidates 0 and 2 both saturate FP16 softmax to [0,0,0,0,1], but candidate 2 has higher raw logit!
    probs_0 = [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    logits_0 = [
        [0.0, 0.0, 0.0, 1.0, 10.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 15.0],  # gold_idx=2 wins exact tie!
        [5.0, 0.0, 0.0, 0.0, 0.0],
    ]
    probs_1 = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.8],  # gold_idx=1 wins continuous R=3.8
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    r_rb = evaluate_reward_bench_2(rb_items, [probs_0, probs_1], [logits_0, None])
    assert r_rb["macro_best_of_4_acc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Category 3: 83 IFEval+IFBench Verifiers, MMLU-Redux, MATH-500, Repeat Loops
# ---------------------------------------------------------------------------


def test_83_if_verifiers_and_loose_vs_strict_evaluation() -> None:
    assert len(INSTRUCTION_VERIFIERS) == 83
    assert len(split_sentences("Dr. Smith wrote 3.14 notes. Really?! Yes!")) == 3

    # Representative IFEval checks
    assert verify_instruction("detectable_format:json_format", '{"a": 1}')
    assert not verify_instruction("detectable_format:json_format", "not json")
    assert verify_instruction("detectable_format:title", "Intro <<My Title>> body")
    assert verify_instruction("punctuation:no_comma", "Hello world without punctuation")
    assert not verify_instruction("punctuation:no_comma", "Hello, world")
    assert verify_instruction("keywords:forbidden_words", "Clean text", {"forbidden_words": ["banned"]})
    assert not verify_instruction("keywords:forbidden_words", "Has banned word", {"forbidden_words": ["banned"]})
    assert verify_instruction("language:response_language", "Bonjour, le chat est dans la maison pour nous.", {"language": "fr"})
    # Loose strips intro line so startend/quotation passes loosely even when strict fails
    assert not verify_instruction("startend:quotation", 'Sure!\n"Quoted response"', loose=False)
    assert verify_instruction("startend:quotation", 'Sure!\n"Quoted response"', loose=True)

    # Representative IFBench checks
    assert verify_instruction("count:conjunctions", "for and nor but or yet", {"small_n": 5})
    assert verify_instruction("count:punctuation", "Wait, stop; look: go-now. Really? Yes! What?!")
    assert not verify_instruction("count:punctuation", "No interrobang or semicolon here.")
    assert verify_instruction("count:word_count_range", "one two three four", {"min_words": 3, "max_words": 5})
    assert verify_instruction("custom:multiples", "14, 21, 28, 35, 42, 49")
    assert not verify_instruction("custom:multiples", "7, 14, 21, 28, 35, 42, 49")
    assert verify_instruction("custom:date_format_list", "1796-04-12, 1805-12-02, 1815-06-18")
    assert verify_instruction("format:parentheses", "a (b [c {d (e [f])}])")
    assert not verify_instruction("format:parentheses", "a (b [c])")
    assert verify_instruction("format:no_whitespace", "NoSpacesHere123")
    assert verify_instruction("format:output_template", "My Answer: x My Conclusion: y Future Outlook: z")
    assert verify_instruction("format:options", "maybe", {"options": "yes/no/maybe"})
    assert verify_instruction("format:sub-bullets", "* Main\n  - Sub\n* Next\n  - Sub2")
    assert verify_instruction("repeat:repeat_span", "code and theory behind", {"prompt_to_repeat": "explain the context, code and theory behind the paper", "n_start": 3, "n_end": 6})
    assert verify_instruction("repeat:repeat_change", "Create a script now", {"prompt_to_repeat": "Write a script now"})
    assert verify_instruction("sentence:increment", "One. Two three four. Five six seven eight nine.", {"small_n": 2})
    assert verify_instruction("words:alphabet", "Apple banana cherry date")
    assert verify_instruction("words:last_first", "Cats sleep. Sleep helps.")
    assert verify_instruction("words:prime_lengths", "to cat apple")
    assert not verify_instruction("words:prime_lengths", "to cat four")
    assert verify_instruction("words:words_position", "The vibrant sky stays vibrant today", {"keyword": "vibrant"})

    # Full IF evaluator + repeat loop invalidation
    items = [
        {"id": "0", "prompt": "p0", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        {"id": "1", "prompt": "p1", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
    ]
    loop_txt = " ".join(["loop token drift error"] * 12)
    r_if = evaluate_instruction_following(items, ["Clean output without commas", loop_txt])
    assert r_if["prompt_strict_acc"] == pytest.approx(0.5)
    assert r_if["repeat_loop_count"] == 1
    assert r_if["per_item_valid"] == [True, False]


def test_mmlu_math500_livebench_and_repeat_4gram_rate() -> None:
    assert repeat_4gram_rate("This is a completely normal varied technical paragraph with distinct words.") == 0.0
    assert repeat_4gram_rate("alpha beta gamma delta " * 8) > 0.8
    ok_loop, rate_loop = check_chat_validity("a b c d " * 10, finish_reason="stop")
    assert not ok_loop and rate_loop > 0.5
    ok_len, _ = check_chat_validity("short valid text", finish_reason="length")
    assert not ok_len

    assert extract_mmlu_choice("Answer: C") == "C"
    assert extract_mmlu_choice("Therefore the answer is \\boxed{B}.") == "B"
    assert extract_mmlu_choice("D. Mitochondria") == "D"
    assert extract_mmlu_choice("Either A or B might be true") is None
    r_mm = evaluate_mmlu_redux(
        [{"id": "0", "subject": "algebra", "answer_letter": "B"}],
        ["Answer: B"],
    )
    assert r_mm["exact_accuracy"] == pytest.approx(1.0)

    # Nested-brace \boxed{} + canonical equivalence
    sol = r"Step 1... Therefore the final value is \boxed{\frac{1 + \sqrt{4}}{3}}."
    assert extract_boxed_answer(sol) == r"\frac{1 + \sqrt{4}}{3}"
    assert math_answers_equivalent(extract_boxed_answer(sol), "1")
    assert math_answers_equivalent(r"\dfrac{2}{4}", "0.5")
    assert math_answers_equivalent(r"2x + 3", r"3 + 2\cdot x")
    assert math_answers_equivalent(r"(1, \frac{4}{2})", "(1, 2)")
    assert not math_answers_equivalent("[-1, 3)", "(-1, 3]")
    r_math = evaluate_math_500([{"id": "m0", "answer": r"\frac{1}{2}", "level": 3}], [r"So $\boxed{0.5}$."])
    assert r_math["verified_accuracy"] == pytest.approx(1.0)

    r_lb = evaluate_livebench_reasoning(
        [{"id": "lb0", "task": "zebra_puzzle", "ground_truth": "1, filmmaking, journalist"}],
        ["Reasoning...\n**1, filmmaking, journalist**"],
    )
    assert r_lb["verified_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Category 4: Rectangular Hungarian, Grounded Field-F1, BFCL v3 & llguidance
# ---------------------------------------------------------------------------


def test_rectangular_hungarian_matches_brute_force_optimum() -> None:
    rng = np.random.default_rng(123)
    for shape in ((4, 4), (3, 5), (5, 3), (1, 4), (4, 1), (3, 3)):
        mat = rng.uniform(0.0, 1.0, size=shape)
        r_ind, c_ind = linear_sum_assignment_max(mat)
        got = float(np.sum(mat[r_ind, c_ind]))
        # Exact brute-force check over all injections of smaller side into larger side
        n_r, n_c = shape
        best = -1.0
        if n_r <= n_c:
            for perm in itertools.permutations(range(n_c), n_r):
                best = max(best, sum(float(mat[r, perm[r]]) for r in range(n_r)))
        else:
            for perm in itertools.permutations(range(n_r), n_c):
                best = max(best, sum(float(mat[perm[c], c]) for c in range(n_c)))
        assert got == pytest.approx(best, abs=1e-12)

    r0, c0 = hungarian_min_cost(np.zeros((0, 3)))
    assert r0.size == 0 and c0.size == 0


def test_grounded_field_f1_penalizes_hallucinated_nulls_and_matches_unordered_arrays() -> None:
    schema = {
        "type": "object",
        "properties": {
            "customer": {"type": "string"},
            "renewal_date": {"type": "string", "format": "date"},
            "amount": {"type": "number"},
            "optional_note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
                    "required": ["sku", "qty"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["customer", "renewal_date", "amount", "optional_note", "line_items"],
        "additionalProperties": False,
    }
    gold = {
        "customer": "Jan Nowicki",
        "renewal_date": "2025-01-15",
        "amount": 120.50,
        "optional_note": None,
        "line_items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 5}],
    }
    # Reordered array + normalized ISO date + float within 1e-5 => 1.0!
    pred_exact = {
        "customer": "  jan   nowicki ",
        "renewal_date": "2025/01/15",
        "amount": 120.500001,
        "optional_note": None,
        "line_items": [{"sku": "B2", "qty": 5}, {"sku": "A1", "qty": 2}],
    }
    assert grounded_field_f1(pred_exact, gold, schema) == pytest.approx(1.0)

    # Hallucinated non-null on gold null strictly penalizes score (4 exact + 1 hallucinated = 4/5 = 0.8)
    pred_halluc = dict(pred_exact, optional_note="Hallucinated clause")
    assert grounded_field_f1(pred_halluc, gold, schema) == pytest.approx(0.8)

    # Invalid schema immediately returns 0.0
    pred_bad_schema = dict(pred_exact, extra_illegal_key=123)
    assert grounded_field_f1(pred_bad_schema, gold, schema) == pytest.approx(0.0)

    r_ext = evaluate_structured_extraction(
        [{"id": "e0", "schema": schema, "extraction": gold}],
        [json.dumps(pred_exact)],
    )
    assert r_ext["grounded_field_f1"] == pytest.approx(1.0)
    assert r_ext["valid_schema_rate"] == pytest.approx(1.0)


def test_bfcl_v3_ast_parallel_abstention_and_jsonschemabench_compiler() -> None:
    fn_defs = [
        {
            "name": "calc_area",
            "parameters": {
                "type": "dict",
                "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}, "unit": {"type": "string"}},
                "required": ["base", "height"],
            },
        },
        {
            "name": "get_weather",
            "parameters": {
                "type": "dict",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]
    bfcl_schema = build_bfcl_response_schema(fn_defs)
    # Explicit empty abstention branch MUST validate!
    assert validate_against_jsonschema({"tool_calls": []}, bfcl_schema)
    assert validate_against_jsonschema(
        {"tool_calls": [{"name": "calc_area", "arguments": {"base": 10, "height": 5}}]}, bfcl_schema
    )

    ok_p, calls_p = parse_bfcl_tool_calls('[get_weather(city="Boston"), calc_area(base=10, height=5)]')
    assert ok_p and len(calls_p) == 2
    gt_parallel = [
        {"calc_area": {"base": [10], "height": [5], "unit": ["units", ""]}},
        {"get_weather": {"city": ["Boston", "Boston, MA"]}},
    ]
    assert check_bfcl_ast_match(calls_p, gt_parallel, fn_defs, is_irrelevance=False)
    assert not check_bfcl_ast_match(calls_p, [], fn_defs, is_irrelevance=True)
    assert check_bfcl_ast_match([], [], fn_defs, is_irrelevance=True)

    r_bf = evaluate_bfcl_v3(
        [
            {"id": "pm0", "split": "parallel_multiple", "functions": fn_defs, "ground_truth": gt_parallel, "is_irrelevance": False},
            {"id": "ir0", "split": "irrelevance", "functions": fn_defs, "ground_truth": [], "is_irrelevance": True},
            {"id": "ir1", "split": "irrelevance", "functions": fn_defs, "ground_truth": [], "is_irrelevance": True},
        ],
        [
            json.dumps({"tool_calls": [{"name": "get_weather", "arguments": {"city": "Boston"}}, {"name": "calc_area", "arguments": {"base": 10, "height": 5}}]}),
            json.dumps({"tool_calls": []}),
            json.dumps({"tool_calls": [{"name": "get_weather", "arguments": {"city": "Boston"}}]}),  # hallucinated tool!
        ],
    )
    assert r_bf["overall_ast_acc"] == pytest.approx(2.0 / 3.0)
    assert r_bf["abstention_acc"] == pytest.approx(0.5)

    # Real llguidance compilation diagnostic on complex schemas ($ref, anyOf, additionalProperties: false, pattern)
    complex_schemas = [
        {
            "id": "s0",
            "subset": "Github_hard",
            "json_schema": json.dumps(
                {
                    "$defs": {"Tag": {"type": "string", "pattern": "^[a-z0-9-]+$"}},
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}},
                        "mode": {"anyOf": [{"type": "string", "enum": ["fast", "parity"]}, {"type": "null"}]},
                    },
                    "required": ["name", "tags"],
                    "additionalProperties": False,
                }
            ),
        },
        {
            "id": "s1",
            "subset": "Kubernetes",
            "json_schema": json.dumps(bfcl_schema),
        },
    ]
    diag = evaluate_jsonschemabench_compiler(complex_schemas)
    assert diag["compiled_ok"] == 2
    assert diag["compile_success_rate"] == pytest.approx(1.0)
    assert diag["keyword_coverage"]["has_ref"]["compiled"] == 1
    assert diag["keyword_coverage"]["has_anyof"]["compiled"] == 2


# ---------------------------------------------------------------------------
# 7. CLI Harness Dry-Run & Offline Compare Subcommand
# ---------------------------------------------------------------------------


def test_benchmark_outcomes_cli_dry_run_and_compare(tmp_path: Path) -> None:
    script = _REPO_ROOT / "scripts" / "benchmark_outcomes.py"
    if not script.is_file():
        # Standalone fast-shared mirror: verify GateEvaluation serialization contract directly.
        ev = evaluate_promotion_gates(
            {"objective_score_e2e": 0.62, "effective_outcome_per_sec": 50.0},
            {"objective_score_e2e": 0.619, "effective_outcome_per_sec": 62.0},
            profile="quantized",
            min_eot_ratio=1.03,
        )
        assert ev.verdict == "PASS" and ev.eot_ratio == pytest.approx(1.24)
        return

    proc = subprocess.run(
        [sys.executable, str(script), "run", "--tier", "tier1", "--packages", "fast-bge-small", "--dry-run"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(proc.stdout)
    assert plan["dry_run"] is True
    assert plan["packages"] == ["fast-bge-small"]
    assert len(plan["dataset_lock_sha256"]) == 64

    base_p = tmp_path / "base.json"
    cand_p = tmp_path / "cand.json"
    base_doc = {
        "environment": {"mem_gb": 48},
        "results": [
            {
                "package": "fast-pplx",
                "category": 1,
                "config_label": "gpu",
                "benchmark": "NanoBEIR",
                "n_items": 20,
                "n_valid": 20,
                "objective_score_e2e": 0.620,
                "effective_outcome_per_sec": 50.0,
                "per_item_effective_scores": [0.62] * 20,
                "diagnostics": {"mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 2.5},
            }
        ],
    }
    cand_doc = {
        "environment": {"mem_gb": 48},
        "results": [
            {
                "package": "fast-pplx",
                "category": 1,
                "config_label": "hetero2",
                "benchmark": "NanoBEIR",
                "n_items": 20,
                "n_valid": 20,
                "objective_score_e2e": 0.619,
                "effective_outcome_per_sec": 62.0,
                "per_item_effective_scores": [0.619] * 20,
                "diagnostics": {"mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 2.8},
            }
        ],
    }
    base_p.write_text(json.dumps(base_doc), encoding="utf-8")
    cand_p.write_text(json.dumps(cand_doc), encoding="utf-8")
    cmp_proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "compare",
            str(base_p),
            str(cand_p),
            "--gate-profile",
            "quantized",
            "--min-eot-ratio",
            "1.03",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    cmp_out = json.loads(cmp_proc.stdout)
    assert cmp_out["verdict"] == "PASS"
    assert cmp_out["comparisons"][0]["eot_ratio"] == pytest.approx(1.24)

    # Multi-config self-compare (Cat 2 Laya + Kev, Cat 4 llg_on + llg_off): ZERO key overwrites!
    multi_doc = {
        "environment": {"mem_gb": 48},
        "results": [
            {
                "package": "fast-decision-models",
                "category": 2,
                "config_label": "laya_ane",
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "objective_score_e2e": 0.65,
                "effective_outcome_per_sec": 80.0,
                "per_item_effective_scores": [0.65] * 10,
                "diagnostics": {"backend": "laya-ane", "mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 2.0},
            },
            {
                "package": "fast-decision-models",
                "category": 2,
                "config_label": "kev_08b_w8",
                "benchmark": "RAGTruth+CLINC150+RewardBench2",
                "objective_score_e2e": 0.85,
                "effective_outcome_per_sec": 30.0,
                "per_item_effective_scores": [0.85] * 10,
                "diagnostics": {"backend": "kev-0.8b", "mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 3.0},
            },
            {
                "package": "fast-gemma4",
                "category": 4,
                "config_label": "fast_gemma4_llg_on",
                "benchmark": "LiquidAI+BFCL",
                "objective_score_e2e": 0.72,
                "effective_outcome_per_sec": 4.0,
                "per_item_effective_scores": [0.72] * 10,
                "diagnostics": {"llguidance": True, "mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 10.0},
            },
            {
                "package": "fast-gemma4",
                "category": 4,
                "config_label": "fast_gemma4_llg_off",
                "benchmark": "LiquidAI+BFCL",
                "objective_score_e2e": 0.68,
                "effective_outcome_per_sec": 3.8,
                "per_item_effective_scores": [0.68] * 10,
                "diagnostics": {"llguidance": False, "mask_fallback_rate": 0.0, "repeat_loop_count": 0},
                "memory_4bucket_gb": {"combined_with_model_mapped_gb": 10.0},
            },
        ],
    }
    multi_p = tmp_path / "multi.json"
    multi_p.write_text(json.dumps(multi_doc), encoding="utf-8")
    self_cmp = subprocess.run(
        [sys.executable, str(script), "compare", str(multi_p), str(multi_p)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    self_out = json.loads(self_cmp.stdout)
    assert self_out["verdict"] == "PASS"
    assert len(self_out["comparisons"]) == 4
    for row in self_out["comparisons"]:
        assert row["baseline_label"] == row["candidate_label"]
        assert row["delta_objective_score_e2e"] == pytest.approx(0.0)

    # Disjoint reports MUST FAIL with exit code 1 (no vacuous PASS on empty comparisons!).
    disjoint_p = tmp_path / "disjoint.json"
    disjoint_p.write_text(json.dumps({"environment": {"mem_gb": 48}, "results": []}), encoding="utf-8")
    disj_proc = subprocess.run(
        [sys.executable, str(script), "compare", str(base_p), str(disjoint_p)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert disj_proc.returncode == 1
    disj_out = json.loads(disj_proc.stdout)
    assert disj_out["verdict"] == "FAIL"
    assert disj_out["comparisons"] == []


def test_all_83_verifiers_safe_and_zero_forbidden_imports() -> None:
    import ast as py_ast

    sample = "Hello world. This is a sample sentence! Really? Maybe."
    for iid in sorted(INSTRUCTION_VERIFIERS):
        res_empty = verify_instruction(iid, "", {}, prompt="Sample prompt", loose=False)
        res_sample = verify_instruction(iid, sample, {}, prompt="Sample prompt", loose=True)
        assert isinstance(res_empty, bool) and isinstance(res_sample, bool)

    forbidden = {"datasets", "torch", "scipy", "nltk", "transformers"}
    pkg_dir = Path(__file__).resolve().parents[1] / "src" / "fast_shared" / "outcome_eval"
    files = list(pkg_dir.glob("*.py"))
    cli_script = _REPO_ROOT / "scripts" / "benchmark_outcomes.py"
    if cli_script.is_file():
        files.append(cli_script)
    for fp in files:
        tree = py_ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))
        for node in py_ast.walk(tree):
            if isinstance(node, py_ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, f"Forbidden import {top!r} in {fp.name}"
            elif isinstance(node, py_ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in forbidden, f"Forbidden import {top!r} in {fp.name}"


def test_extractbench_middle_table_drop_and_bfcl_edge_cases() -> None:
    # Simulate multi-page repeating table extraction (ExtractBench) where SWA-128 drops middle rows.
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"line_no": {"type": "integer"}, "amount": {"type": "number"}},
                    "required": ["line_no", "amount"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }
    gold = {"rows": [{"line_no": i, "amount": float(i * 10)} for i in range(1, 11)]}
    # Middle rows 4..7 dropped (6/10 retrieved); schema is still 100% valid, but F1 must drop to 2*6/(6+10) = 0.75!
    pred_drop = {"rows": [r for r in gold["rows"] if r["line_no"] not in (4, 5, 6, 7)]}
    assert validate_against_jsonschema(pred_drop, schema) is True
    assert grounded_field_f1(pred_drop, gold, schema) == pytest.approx(0.75)

    # BFCL missing required arg, extra arg, and dot-vs-underscore normalization
    fn_defs = [
        {
            "name": "math.add",
            "parameters": {
                "type": "dict",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        }
    ]
    gt = [{"math.add": {"a": [2], "b": [3]}}]
    assert check_bfcl_ast_match([("math_add", {"a": 2, "b": 3})], gt, fn_defs) is True
    assert check_bfcl_ast_match([("math.add", {"a": 2})], gt, fn_defs) is False
    assert check_bfcl_ast_match([("math.add", {"a": 2, "b": 3, "extra": 9})], gt, fn_defs) is False


def test_hungarian_nonfinite_guard_large_nested_lists_and_bfcl_nested_dicts() -> None:
    # 1. hungarian_min_cost MUST reject NaN / Inf instead of spinning in an infinite loop.
    with pytest.raises(ValueError, match="finite"):
        hungarian_min_cost(np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        hungarian_min_cost(np.array([[1.0, np.inf], [0.0, 1.0]]))

    # 2. Large (>64) array of nested lists MUST NOT raise TypeError: unhashable type 'list'.
    gold_big_lists = {"matrix": [[i, i + 1] for i in range(70)]}
    pred_big_lists = {"matrix": list(reversed(gold_big_lists["matrix"]))}
    assert grounded_field_f1(pred_big_lists, gold_big_lists, None, require_schema_valid=False) == pytest.approx(1.0)
    # Large primitive array distinguishes bool True from int 1.
    assert grounded_field_f1([True] * 70, [1] * 70, None, require_schema_valid=False) == pytest.approx(0.0)

    # 3. BFCL v3 nested-dict possible_answer specs (multiple_8, parallel_multiple_135, multiple_119).
    fn_realestate = [
        {
            "name": "realestate.find_properties",
            "parameters": {"type": "dict", "properties": {"budget": {"type": "dict"}}, "required": ["budget"]},
        }
    ]
    gt_budget = [{"realestate.find_properties": {"budget": [{"min": [300000], "max": [400000], "currency": ["USD", ""]}]}}]
    assert check_bfcl_ast_match([("realestate.find_properties", {"budget": {"min": 300000, "max": 400000}})], gt_budget, fn_realestate) is True
    assert check_bfcl_ast_match([("realestate.find_properties", {"budget": {"min": 300000, "max": 400000, "currency": "USD"}})], gt_budget, fn_realestate) is True
    assert check_bfcl_ast_match([("realestate.find_properties", {"budget": {"min": 300000, "max": 500000}})], gt_budget, fn_realestate) is False
    assert check_bfcl_ast_match([("realestate.find_properties", {"budget": {"min": [300000], "max": [400000]}})], gt_budget, fn_realestate) is False
    assert check_bfcl_ast_match([("realestate.find_properties", {"budget": {"min": 300000, "max": 400000, "bogus": 1}})], gt_budget, fn_realestate) is False

    # Nested dict containing list-typed values (parallel_multiple_135 cards).
    gt_cards = [{"poker_game_winner": {"cards": [{"John": [["2 of hearts", "3 of diamonds"]]}]}}]
    assert check_bfcl_ast_match([("poker_game_winner", {"cards": {"John": ["2 of hearts", "3 of diamonds"]}})], gt_cards) is True

    # List of nested dicts (multiple_119 conditions).
    gt_conds = [{"database.query": {"conditions": [[{"field": ["age"], "operation": [">"], "value": ["25"]}]]}}]
    assert check_bfcl_ast_match([("database.query", {"conditions": [{"field": "age", "operation": ">", "value": "25"}]})], gt_conds) is True

    # 4. Qualified declared name vs unqualified GT (simple_363), rejecting ambiguous namespaces.
    fn_rest = [
        {
            "name": "restaurant_search.find_closest",
            "parameters": {"type": "dict", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        }
    ]
    gt_rest = [{"find_closest": {"location": ["Boston", "Boston, MA"]}}]
    assert check_bfcl_ast_match([("restaurant_search.find_closest", {"location": "Boston"})], gt_rest, fn_rest) is True
    assert check_bfcl_ast_match([("restaurant_search.find_closest", {"location": "Boston", "bad": 1})], gt_rest, fn_rest) is False

    fn_ambig = [
        {"name": "a.find_closest", "parameters": {"type": "dict", "properties": {"location": {"type": "string"}}}},
        {"name": "b.find_closest", "parameters": {"type": "dict", "properties": {"location": {"type": "string"}}}},
    ]
    assert check_bfcl_ast_match([("a.find_closest", {"location": "Boston"})], gt_rest, fn_ambig) is False

