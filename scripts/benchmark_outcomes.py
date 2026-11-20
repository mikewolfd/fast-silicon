#!/usr/bin/env python3
"""External Objective Outcome Benchmark Suite CLI for ``fast-silicon``.

Subcommands:
  - ``setup-data``: Stage & verify immutable HF dataset shards into local cache (0 GPU work).
  - ``run``: Execute Tier-1 / Tier-2 objective outcome benchmarks in isolated child subprocesses
    serialized under ``/tmp/m4_gpu_validation.lock``, capturing ``FourBucketMemoryProbe``,
    ``DeviceMemoryGuard``, Effective Outcome Throughput (``EOT``), and optional ``ABBA`` comparison.
  - ``compare``: Offline Two-Gate + Memory Ceiling promotion evaluation between two JSON reports.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
import urllib.request

import numpy as np

from fast_shared.bench import (
    DeviceMemoryGuard,
    FourBucketMemoryProbe,
    check_system_quiescence,
    collect_env_telemetry,
    wired_limit,
)
from fast_shared.outcome_eval import (
    CLINC150_DOMAINS,
    CLINC_STAGE1_DOMAINS,
    build_banking77_flat_qdef,
    build_bfcl_response_schema,
    build_clinc_stage1_qdef,
    build_clinc_stage2_qdef,
    build_ragtruth_noul_qdef,
    build_ragtruth_state,
    build_rewardbench2_score_qdef,
    check_embedding_validity,
    check_probability_validity,
    chunk_document_char_spans,
    compute_contextual_lift,
    compute_effective_item_scores,
    compute_outcome_metrics,
    compute_paired_se,
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
    evaluate_multiclass_choice,
    evaluate_promotion_gates,
    evaluate_ragtruth_noul,
    evaluate_retrieval_rankings,
    evaluate_reward_bench_2,
    evaluate_structured_extraction,
    load_banking77,
    load_bfcl_v3,
    load_clinc_oos,
    load_ifbench_test,
    load_ifeval,
    load_jsonschemabench,
    load_legalbench_rag,
    load_liquidai_extraction,
    load_livebench_reasoning,
    load_lockfile,
    load_longembed_subset,
    load_math_500,
    load_mmlu_redux,
    load_nanobeir_subset,
    load_ragtruth,
    load_reward_bench_2,
    partition_banking77_clusters,
    resolve_cache_dir,
    stage_tier_datasets,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"
_WORKER_MARKER = "__OUTCOME_WORKER_JSON__="
_ARTIFACT_SUBDIRS = ("weights", "ane_models", "kev_08b", "laya_typed", "models", "tokenizer", "vocab")


@dataclass(frozen=True)
class OutcomePackageSpec:
    name: str
    category: int
    home_env_var: str
    description: str


PACKAGE_SPECS: tuple[OutcomePackageSpec, ...] = (
    OutcomePackageSpec("fast-bge-small", 1, "FAST_BGE_SMALL_HOME", "33M BERT bi-encoder (Cat 1)"),
    OutcomePackageSpec("fast-embeddinggemma-2", 1, "FAST_EMBEDDINGGEMMA2_HOME", "308M MRL encoder <=2K (Cat 1)"),
    OutcomePackageSpec("fast-pplx", 1, "FAST_PPLX_HOME", "0.6B 32K contextual encoder (Cat 1)"),
    OutcomePackageSpec("fast-decision-models", 2, "FAST_DECISION_MODELS_HOME", "Laya-ANE & Kev-0.8B decisions (Cat 2)"),
    OutcomePackageSpec("fast-gemma4", 4, "FAST_GEMMA4_HOME", "Gemma-4 chat/IF/reasoning & structured JSON (Cat 3+4)"),
    OutcomePackageSpec("fast-gpt-oss", 4, "FAST_GPT_OSS_HOME", "GPT-OSS-20B chat/reasoning & structured JSON (Cat 3+4)"),
)
_SPEC_BY_NAME: dict[str, OutcomePackageSpec] = {s.name: s for s in PACKAGE_SPECS}


def _has_staged_artifacts(home_dir: Path) -> bool:
    art = home_dir / "artifacts"
    if not art.is_dir():
        return False
    # For fast-gemma4, weights/models/ane_models may live in sibling repo while monorepo only has vocab/.
    strong_dirs = ("weights", "ane_models", "kev_08b", "laya_typed", "models")
    if any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in strong_dirs):
        return True
    return any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in _ARTIFACT_SUBDIRS)


def _has_strong_artifacts(home_dir: Path) -> bool:
    art = home_dir / "artifacts"
    if not art.is_dir():
        return False
    strong_dirs = ("weights", "ane_models", "kev_08b", "laya_typed", "models")
    return any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in strong_dirs)


def resolve_package_home(spec: OutcomePackageSpec, base_env: Mapping[str, str] | None = None) -> Path:
    """Resolve effective home dir for ``spec``, preferring explicit env or staged weights/models."""
    env = os.environ if base_env is None else base_env
    explicit = env.get(spec.home_env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    mono_home = _REPO_ROOT / "packages" / spec.name
    sibling_home = _REPO_ROOT.parent / spec.name
    if not _has_strong_artifacts(mono_home) and _has_strong_artifacts(sibling_home):
        return sibling_home
    if not _has_staged_artifacts(mono_home) and _has_staged_artifacts(sibling_home):
        return sibling_home
    return mono_home


@contextmanager
def _hw_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    import fcntl

    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev = os.environ.get("M4_GPU_LOCK_HELD")
        os.environ["M4_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("M4_GPU_LOCK_HELD", None)
            else:
                os.environ["M4_GPU_LOCK_HELD"] = prev
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _mx_sync() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "synchronize"):
            mx.synchronize()
    except Exception:
        pass


def _mx_peak_gb() -> float:
    try:
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
            return round(float(mx.get_peak_memory()) / 1e9, 4)
    except Exception:
        pass
    return 0.0


def _mx_reset_peak() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    except Exception:
        pass


def _memory_4bucket_dict(probe: FourBucketMemoryProbe) -> dict[str, float]:
    snap = probe.snapshot(evict_mapped=False)
    fp = snap["phys_footprint"]
    vm = snap["task_vm_info"]
    mf = snap["mapped_files"]
    return {
        "mlx_peak_gb": _mx_peak_gb(),
        "phys_footprint_gb": round(float(fp["phys_footprint_gb"]), 4),
        "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
        "model_mapped_gb": round(float(mf["model_bytes"]) / 1e9, 4),
        "neural_nofootprint_gb": round(max(0, int(vm["ledger_tag_neural_nofootprint_bytes"])) / 1e9, 4),
        "combined_with_model_mapped_gb": round(float(snap["combined_with_model_mapped_gb"]), 4),
    }


def _host_mem_gb(env_meta: Mapping[str, Any]) -> float:
    val = env_meta.get("mem_gb")
    return float(val) if val else 0.0


def _build_record(
    *,
    package: str,
    category: int,
    config_label: str,
    benchmark: str,
    valid_flags: Sequence[bool],
    item_scores: Sequence[float],
    wall_sec: float,
    diagnostics: Mapping[str, Any],
    memory_4bucket_gb: Mapping[str, float],
    sub_benchmarks: Mapping[str, Any],
    host_mem_gb: float,
) -> dict[str, Any]:
    summary = compute_outcome_metrics(valid_flags, item_scores, max(wall_sec, 1e-6))
    rec = summary.to_dict()

    eff_vec = compute_effective_item_scores(valid_flags, item_scores)
    mem_ok, _, _, mem_reasons = evaluate_memory_gate(
        memory_4bucket_gb,
        host_mem_gb=host_mem_gb,
        device_guard_tripped=bool(diagnostics.get("device_guard_tripped", False)),
    )
    reasons = list(mem_reasons)
    if rec["n_valid"] <= 0:
        reasons.append("n_valid == 0 (all outputs failed validity checks)")
    rep_loops = int(diagnostics.get("repeat_loop_count", 0) or 0)
    if rep_loops > 0:
        reasons.append(f"repeat_loop_count={rep_loops} > 0")
    verdict = "PASS" if (mem_ok and not reasons) else "FAIL"
    return {
        "package": package,
        "category": int(category),
        "config_label": config_label,
        "benchmark": benchmark,
        **rec,
        "se_paired_vs_baseline": None,
        "diagnostics": dict(diagnostics),
        "memory_4bucket_gb": dict(memory_4bucket_gb),
        "sub_benchmarks": dict(sub_benchmarks),
        "per_item_effective_scores": [round(float(x), 6) for x in eff_vec.tolist()],
        "gate_verdict": verdict,
        "gate_reasons": reasons,
        "status": "ok",
    }


def _parse_extra_flags(extra_args: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", default=None)
    p.add_argument("--ane-profile", default=None)
    p.add_argument("--ane-variant", default="fp16")
    p.add_argument("--no-kv-i8", action="store_true")
    p.add_argument("--backend", choices=("all", "laya-ane", "kev-0.8b"), default="all")
    p.add_argument("--kev-quant", choices=("auto", "w8", "bf16"), default="w8")
    p.add_argument("--batch-docs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--categories", default="all")
    p.add_argument("--no-llguidance", action="store_true")
    p.add_argument("--reasoning-effort", default=None)
    ns, _ = p.parse_known_args(list(extra_args))
    return ns


# ---------------------------------------------------------------------------
# Category 1 Workers: fast-bge-small, fast-embeddinggemma-2, fast-pplx
# ---------------------------------------------------------------------------


def _run_cat1_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
) -> list[dict[str, Any]]:
    probe = FourBucketMemoryProbe(model_directories=[home / "artifacts"])
    guard = DeviceMemoryGuard()
    guard_tripped = False

    ds_nb = lock["datasets"]["nanobeir_en"]
    subsets = list(ds_nb["tier1_subsets"] if tier == "tier1" else ds_nb["tier2_subsets"])
    if smoke:
        subsets = subsets[:1]

    lb_max_docs = 6 if smoke else (min(12, max_items) if max_items else None)
    lb_max_q = 12 if smoke else max_items

    nb_loaded = {
        sub: load_nanobeir_subset(sub, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
        for sub in subsets
    }
    if smoke or max_items:
        q_cap = 12 if smoke else int(max_items or 50)
        d_cap = 120 if smoke else None
        for sub, pack in nb_loaded.items():
            q_ids_keep = sorted(pack["queries"].keys())[:q_cap]
            pack["queries"] = {qid: pack["queries"][qid] for qid in q_ids_keep}
            pack["qrels"] = {qid: pack["qrels"][qid] for qid in q_ids_keep if qid in pack["qrels"]}
            if d_cap is not None:
                rel_docs = {cid for mp in pack["qrels"].values() for cid in mp}
                extra_docs = [cid for cid in sorted(pack["corpus"].keys()) if cid not in rel_docs][: max(0, d_cap - len(rel_docs))]
                keep_docs = rel_docs | set(extra_docs)
                pack["corpus"] = {cid: pack["corpus"][cid] for cid in sorted(keep_docs) if cid in pack["corpus"]}

    lb_pack = load_legalbench_rag(
        tier=tier,
        max_docs=lb_max_docs,
        max_queries=lb_max_q,
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Chunk LegalBench-RAG documents deterministically (~256 tokens ~= 1024 chars).
    lb_doc_ids = sorted(lb_pack["corpus_texts"].keys())
    lb_chunks_by_doc = {
        did: chunk_document_char_spans(did, lb_pack["corpus_texts"][did], chunk_chars=1024, overlap_chars=0)
        for did in lb_doc_ids
    }
    lb_flat_chunks = [ch for did in lb_doc_ids for ch in lb_chunks_by_doc[did]]
    lb_queries = lb_pack["queries"]
    lb_q_texts = [q["query"] for q in lb_queries]

    mode = extra.mode
    nb_pre_normalized = True
    _enc_coarse_2k = None
    if pkg_name == "fast-bge-small":
        from fast_bge_small.engine import FastBgeSmall

        eff_mode = mode or "hetero"
        eng = FastBgeSmall(home=home, default_mode=eff_mode, warmup=True)
        mrl_dims: tuple[int | str, ...] = (128, 256, "max")
        int8_mode = "sym_int8"

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=True, add_query_prefix=True, mode=eff_mode)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=False, add_query_prefix=False, mode=eff_mode)

        _enc_q_nb = _enc_q
        _enc_d_nb = _enc_d
        _enc_ctx = None
    elif pkg_name == "fast-embeddinggemma-2":
        from fast_embeddinggemma_2.engine import FastEmbeddingGemma2

        eff_mode = mode or "hetero"
        eng = FastEmbeddingGemma2(home=home, default_mode=eff_mode, warmup=True)
        mrl_dims = (128, 256, 512, "max")
        int8_mode = "sym_int8"

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=True, add_query_prefix=True, mode=eff_mode)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=False, add_document_prefix=True, mode=eff_mode)

        def _enc_coarse_2k(txts: list[str]) -> np.ndarray:
            # Exercise native 2K-token context via encode_corpus(..., allow_gpu_overflow=True).
            tok_lists = eng.tokenizer.encode_batch(txts, add_document_prefix=True, max_length=2048)
            return eng.encode_corpus(tok_lists, mode=eff_mode, allow_gpu_overflow=True)

        _enc_q_nb = _enc_q
        _enc_d_nb = _enc_d
        _enc_ctx = None
    elif pkg_name == "fast-pplx":
        from fast_pplx.engine import FastPplx

        eff_mode = mode or "hetero2"
        eng = FastPplx(home=home, mode=eff_mode, ane_profile=extra.ane_profile, local_files_only=True)
        mrl_dims = (128, 256, 512, "max")
        int8_mode = "native_tanh_int8"
        nb_pre_normalized = False

        def _enc_q_nb(txts: list[str]) -> np.ndarray:
            # Raw unnormalized projections required for exact native clip(rint(tanh(x)*127)) int8 parity.
            return eng.encode_queries(txts, mode=eff_mode, output_dtype="float32", normalize=False)

        def _enc_d_nb(txts: list[str]) -> np.ndarray:
            return eng.encode_documents(txts, mode=eff_mode, output_dtype="float32", normalize=False)

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_queries(txts, mode=eff_mode, output_dtype="float32", normalize=True)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_documents(txts, mode=eff_mode, output_dtype="float32", normalize=True)

        ctx_max_len = 4096 if tier == "tier1" else 16384

        def _enc_ctx(doc_groups: list[list[str]]) -> list[np.ndarray]:
            return eng.encode_contextual_chunks(
                doc_groups, max_length=ctx_max_len, mode=eff_mode, output_dtype="float32", normalize=True
            )
    else:
        raise ValueError(f"Unsupported Cat 1 package: {pkg_name}")

    try:
        guard.arm()
        # Untimed warmup batch
        _ = _enc_q(["warmup query"])
        _ = _enc_d(["warmup document passage"])
        _mx_sync()
        _mx_reset_peak()

        all_valid: list[bool] = []
        all_scores: list[float] = []
        sub_reports: dict[str, Any] = {}
        mrl_deltas: list[float] = []

        t0 = time.perf_counter()

        # 1A: NanoBEIR-en + zero-recompute MRL x dtype sweep
        nb_ndcgs: list[float] = []
        for sub in subsets:
            pack = nb_loaded[sub]
            q_ids = sorted(pack["queries"].keys())
            d_ids = sorted(pack["corpus"].keys())
            q_vecs = _enc_q_nb([pack["queries"][qid] for qid in q_ids])
            d_vecs = _enc_d_nb([pack["corpus"][did] for did in d_ids])
            _mx_sync()
            _, ref = guard.check()
            if ref is not None:
                guard_tripped = True

            r_main = evaluate_retrieval_rankings(
                q_ids, q_vecs, d_ids, d_vecs, pack["qrels"], k=10, expect_unit_norm=nb_pre_normalized
            )
            grid = evaluate_mrl_dtype_grid(
                q_ids,
                q_vecs,
                d_ids,
                d_vecs,
                pack["qrels"],
                dims=mrl_dims,
                dtypes=("f32", "f16", "int8"),
                int8_mode=int8_mode,
                pre_normalized=nb_pre_normalized,
                k=10,
            )
            nb_ndcgs.append(float(r_main["ndcg@10"]))
            if grid.get("mrl_128_int8_delta_ndcg@10") is not None:
                mrl_deltas.append(float(grid["mrl_128_int8_delta_ndcg@10"]))
            all_valid.extend(r_main["per_query_valid"])
            all_scores.extend(r_main["per_query_ndcg"])
            sub_reports[sub] = {
                "n_queries": r_main["n_queries"],
                "n_docs": r_main["n_docs"],
                "ndcg@10": round(float(r_main["ndcg@10"]), 6),
                "mrr@10": round(float(r_main["mrr@10"]), 6),
                "recall@10": round(float(r_main["recall@10"]), 6),
                "mrl_grid": grid,
            }

        # 1B: LegalBench-RAG isolated (plus ~2K coarse chunks for fast-embeddinggemma-2 and contextual lift for fast-pplx)
        contextual_lift = None
        coarse_2k_lift = None
        if lb_queries and lb_flat_chunks:
            lb_q_vecs = _enc_q(lb_q_texts)
            lb_iso_vecs = _enc_d([ch.text for ch in lb_flat_chunks])
            _mx_sync()
            r_iso = evaluate_legalbench_spans(lb_queries, lb_flat_chunks, lb_q_vecs, lb_iso_vecs, k=10)
            sub_reports["legalbench_rag_isolated"] = {
                "n_queries": r_iso["n_queries"],
                "n_chunks": len(lb_flat_chunks),
                "chunk_chars": 1024,
                "span_ndcg@10": round(float(r_iso["span_ndcg@10"]), 6),
                "char_recall@10": round(float(r_iso["char_recall@10"]), 6),
            }
            if _enc_ctx is not None:
                doc_chunk_texts = [[ch.text for ch in lb_chunks_by_doc[did]] for did in lb_doc_ids]
                ctx_mats = _enc_ctx(doc_chunk_texts)
                _mx_sync()
                lb_ctx_vecs = np.vstack(ctx_mats) if ctx_mats else np.zeros((0, lb_q_vecs.shape[1]), dtype=np.float32)
                r_ctx = evaluate_legalbench_spans(lb_queries, lb_flat_chunks, lb_q_vecs, lb_ctx_vecs, k=10)
                contextual_lift = compute_contextual_lift(r_iso, r_ctx, k=10)
                sub_reports["legalbench_rag_contextual"] = {
                    "n_queries": r_ctx["n_queries"],
                    "span_ndcg@10": round(float(r_ctx["span_ndcg@10"]), 6),
                    "char_recall@10": round(float(r_ctx["char_recall@10"]), 6),
                    **contextual_lift,
                }
                all_valid.extend(r_ctx["per_query_valid"])
                all_scores.extend(r_ctx["per_query_span_ndcg"])
            elif _enc_coarse_2k is not None:
                lb_coarse_chunks = [
                    ch
                    for did in lb_doc_ids
                    for ch in chunk_document_char_spans(did, lb_pack["corpus_texts"][did], chunk_chars=6144, overlap_chars=0)
                ]
                lb_coarse_vecs = _enc_coarse_2k([ch.text for ch in lb_coarse_chunks])
                _mx_sync()
                r_coarse = evaluate_legalbench_spans(lb_queries, lb_coarse_chunks, lb_q_vecs, lb_coarse_vecs, k=10)
                coarse_2k_lift = compute_contextual_lift(r_iso, r_coarse, k=10)
                sub_reports["legalbench_rag_coarse_2k"] = {
                    "n_queries": r_coarse["n_queries"],
                    "n_chunks": len(lb_coarse_chunks),
                    "chunk_chars": 6144,
                    "span_ndcg@10": round(float(r_coarse["span_ndcg@10"]), 6),
                    "char_recall@10": round(float(r_coarse["char_recall@10"]), 6),
                    "coarse_2k_delta_ndcg@10": coarse_2k_lift["contextual_delta_ndcg@10"],
                    "se_paired": coarse_2k_lift["se_paired"],
                }
                all_valid.extend(r_iso["per_query_valid"])
                all_scores.extend(r_iso["per_query_span_ndcg"])
                all_valid.extend(r_coarse["per_query_valid"])
                all_scores.extend(r_coarse["per_query_span_ndcg"])
            else:
                all_valid.extend(r_iso["per_query_valid"])
                all_scores.extend(r_iso["per_query_span_ndcg"])

        # 1C: Tier-2 LongEmbed (fast-pplx 4K-32K)
        if tier == "tier2" and pkg_name == "fast-pplx" and not smoke:
            le_cfg = lock["datasets"]["longembed"]["tier2"]
            for le_sub in le_cfg["subsets"]:
                ctx_lens = le_cfg.get("passkey_context_lengths") if le_sub == "passkey" else None
                le_max_q = int(le_cfg.get(f"max_queries_{le_sub}", 200))
                if max_items is not None:
                    le_max_q = min(le_max_q, int(max_items))
                le_pack = load_longembed_subset(
                    le_sub,
                    context_lengths=ctx_lens,
                    max_queries=le_max_q,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                q_ids = sorted(le_pack["queries"].keys())
                d_ids = sorted(le_pack["corpus"].keys())
                q_vecs = _enc_q([le_pack["queries"][qid] for qid in q_ids])
                d_vecs = _enc_d([le_pack["corpus"][did] for did in d_ids])
                _mx_sync()
                r_le = evaluate_retrieval_rankings(q_ids, q_vecs, d_ids, d_vecs, le_pack["qrels"], k=10)
                all_valid.extend(r_le["per_query_valid"])
                all_scores.extend(r_le["per_query_ndcg"])
                sub_reports[f"longembed_{le_sub}"] = {
                    "n_queries": r_le["n_queries"],
                    "n_docs": r_le["n_docs"],
                    "ndcg@10": round(float(r_le["ndcg@10"]), 6),
                    "mrr@10": round(float(r_le["mrr@10"]), 6),
                    "recall@10": round(float(r_le["recall@10"]), 6),
                }

        _mx_sync()
        wall_sec = time.perf_counter() - t0
        _, ref_end = guard.check()
        if ref_end is not None:
            guard_tripped = True

        mem4 = _memory_4bucket_dict(probe)
        diag: dict[str, Any] = {
            "mode": eff_mode,
            "nanobeir_macro_ndcg10": round(float(np.mean(nb_ndcgs)) if nb_ndcgs else 0.0, 6),
            "mrl_128_int8_mean_delta_ndcg10": round(float(np.mean(mrl_deltas)) if mrl_deltas else 0.0, 6),
            "contextual_delta_ndcg10": contextual_lift["contextual_delta_ndcg@10"] if contextual_lift else None,
            "coarse_2k_delta_ndcg10": coarse_2k_lift["contextual_delta_ndcg@10"] if coarse_2k_lift else None,
            "device_guard_tripped": guard_tripped,
            "repeat_loop_count": 0,
        }
        bench_name = f"NanoBEIR-en({len(subsets)})+LegalBench-RAG({','.join(lb_pack['subsets'])})"
        return [
            _build_record(
                package=pkg_name,
                category=1,
                config_label=config_label or eff_mode,
                benchmark=bench_name,
                valid_flags=all_valid,
                item_scores=all_scores,
                wall_sec=wall_sec,
                diagnostics=diag,
                memory_4bucket_gb=mem4,
                sub_benchmarks=sub_reports,
                host_mem_gb=host_mem_gb,
            )
        ]
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Category 2 Worker: fast-decision-models (Laya-ANE & Kev-0.8B)
# ---------------------------------------------------------------------------


def _run_cat2_single_backend(
    backend_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
) -> dict[str, Any]:
    from fast_decision_models.kev_mlx import Kev08bMlxBackend
    from fast_decision_models.laya_ane import LayaAneBackend
    from fast_decision_models.paths import ensure_laya_dylib, resolve_kev_dir, resolve_laya_dir

    art_dir = home / "artifacts"
    probe = FourBucketMemoryProbe(model_directories=[art_dir])
    guard = DeviceMemoryGuard()
    guard_tripped = False

    rt_cap = 16 if smoke else max_items
    cl_cap = 16 if smoke else max_items
    rb_cap = 10 if smoke else max_items

    rt_items = load_ragtruth(tier=tier, max_items=rt_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    cl_pack = load_clinc_oos(tier=tier, max_items=cl_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    cl_items = cl_pack["items"]
    rb_items = load_reward_bench_2(tier=tier, max_items=rb_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    b77_pack = (
        load_banking77(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
        if (tier == "tier2" and not smoke)
        else {"items": [], "label_names": []}
    )

    rt_qdef = build_ragtruth_noul_qdef()
    cl_s1_qdef = build_clinc_stage1_qdef()
    cl_s2_qdefs = {dom: build_clinc_stage2_qdef(dom) for dom in CLINC150_DOMAINS}
    rb_qdef = build_rewardbench2_score_qdef()

    is_laya = backend_name == "laya-ane"
    if is_laya:
        laya_dir = resolve_laya_dir(None, artifacts_dir=art_dir)
        eng: Any = LayaAneBackend(laya_dir, ensure_laya_dylib(), pipe_depth=4)
        eff_label = config_label or "laya_ane"
    else:
        kev_dir = resolve_kev_dir(None, artifacts_dir=art_dir)
        eng = Kev08bMlxBackend(kev_dir, quant_mode=extra.kev_quant, batch_docs=int(extra.batch_docs))
        eff_label = config_label or f"kev_08b_{eng.quant_mode}"

    try:
        guard.arm()
        # Untimed warmup
        if is_laya:
            w_enc = eng.encode_item("warmup", rt_qdef, on_truncate="ignore")
            _ = eng.score_noul_batch([w_enc])
        else:
            _ = eng.score_multi_states_noul([("warmup", [rt_qdef])], batch_docs=1)
        _mx_sync()
        _mx_reset_peak()

        # Precompute exact real tokenizer token counts for RAGTruth length slices (untimed so Kev is not double-charged).
        rt_pairs = [(build_ragtruth_state(it), [rt_qdef]) for it in rt_items]
        if is_laya:
            rt_header_toks = len(eng.encode_item("", rt_qdef, on_truncate="ignore")["ids"])
            kev_rt_tok_counts: list[int] = []
        else:
            rt_header_toks = 0
            kev_rt_tok_counts = []
            for st, qlist in rt_pairs:
                s_ids, q_rows = eng.encode_questions(st, qlist)
                kev_rt_tok_counts.append(len(s_ids) + (len(q_rows[0]["ids"]) if q_rows else 0))

        t0 = time.perf_counter()
        sub_reports: dict[str, Any] = {}
        all_valid: list[bool] = []
        all_scores: list[float] = []
        laya_opt_truncations = 0

        # 2A: RAGTruth (`noul`)
        rt_p_true: list[float] = []
        rt_tok_counts: list[int] = []
        if is_laya:
            rt_encs = [eng.encode_item(st, rt_qdef, on_truncate="ignore") for st, _ in rt_pairs]
            rt_tok_counts = [rt_header_toks + int(e.get("original_state_tokens", 0)) for e in rt_encs]
            rt_p_true = [float(p) for p in eng.score_noul_batch(rt_encs)]
        else:
            rt_tok_counts = kev_rt_tok_counts
            out_rt = eng.score_multi_states_noul(rt_pairs, batch_docs=int(extra.batch_docs))
            rt_p_true = [float(r[0][0]) for r in out_rt]
        _mx_sync()
        r_rt = evaluate_ragtruth_noul(rt_items, rt_p_true, token_counts=rt_tok_counts)
        sub_reports["ragtruth_noul"] = {k: v for k, v in r_rt.items() if not k.startswith("per_item_")}
        all_valid.extend(r_rt["per_item_valid"])
        all_scores.extend(r_rt["per_item_scores"])

        # 2B: CLINC150 2-stage (`K=11 -> K=15`)
        if is_laya:
            s1_encs = [eng.encode_item(it["text"], cl_s1_qdef, on_truncate="ignore") for it in cl_items]
            laya_opt_truncations += sum(1 for e in s1_encs if e.get("truncated"))
            s1_probs = eng.score_choice_batch(s1_encs)
            s2_preds: list[tuple[str, Sequence[float]] | None] = [None] * len(cl_items)
            s2_indices: list[int] = []
            s2_doms: list[str] = []
            s2_encs: list[dict[str, Any]] = []
            for idx, probs in enumerate(s1_probs):
                if not probs:
                    continue
                pred_dom = CLINC_STAGE1_DOMAINS[int(np.argmax(np.asarray(probs, dtype=np.float64)))]
                if pred_dom != "out_of_scope" and pred_dom in cl_s2_qdefs:
                    enc2 = eng.encode_item(cl_items[idx]["text"], cl_s2_qdefs[pred_dom], on_truncate="ignore")
                    if enc2.get("truncated"):
                        laya_opt_truncations += 1
                    s2_indices.append(idx)
                    s2_doms.append(pred_dom)
                    s2_encs.append(enc2)
            if s2_encs:
                s2_probs_batch = eng.score_choice_batch(s2_encs)
                for idx, dom, p2 in zip(s2_indices, s2_doms, s2_probs_batch):
                    s2_preds[idx] = (dom, p2)
        else:
            s1_raw = eng.score_multi_states([(it["text"], [cl_s1_qdef]) for it in cl_items], batch_docs=int(extra.batch_docs))
            s1_probs = [r[0][0] for r in s1_raw]
            s2_preds = [None] * len(cl_items)
            s2_indices = []
            s2_doms = []
            s2_pairs: list[tuple[Any, list[Mapping[str, Any]]]] = []
            for idx, probs in enumerate(s1_probs):
                if not probs:
                    continue
                pred_dom = CLINC_STAGE1_DOMAINS[int(np.argmax(np.asarray(probs, dtype=np.float64)))]
                if pred_dom != "out_of_scope" and pred_dom in cl_s2_qdefs:
                    s2_indices.append(idx)
                    s2_doms.append(pred_dom)
                    s2_pairs.append((cl_items[idx]["text"], [cl_s2_qdefs[pred_dom]]))
            if s2_pairs:
                s2_raw = eng.score_multi_states(s2_pairs, batch_docs=int(extra.batch_docs))
                for idx, dom, r2 in zip(s2_indices, s2_doms, s2_raw):
                    s2_preds[idx] = (dom, r2[0][0])
        _mx_sync()
        r_cl = evaluate_clinc_two_stage(cl_items, s1_probs, s2_preds)
        sub_reports["clinc150_2stage"] = {k: v for k, v in r_cl.items() if not k.startswith("per_item_")}
        all_valid.extend(r_cl["per_item_valid"])
        all_scores.extend(r_cl["per_item_scores"])

        # 2C: RewardBench-2 Best-of-4 (`score`)
        rb_cand_probs: list[list[list[float]]] = []
        rb_cand_logits: list[list[list[float]] | None] = []
        flat_states = [
            {"prompt": it["prompt"], "response": cand}
            for it in rb_items
            for cand in it["candidates"]
        ]
        if is_laya:
            rb_encs = [eng.encode_item(st, rb_qdef, on_truncate="ignore") for st in flat_states]
            flat_p = eng.score_choice_batch(rb_encs)
            for idx in range(len(rb_items)):
                rb_cand_probs.append(flat_p[4 * idx : 4 * idx + 4])
                rb_cand_logits.append(None)
        else:
            rb_raw = eng.score_multi_states([(st, [rb_qdef]) for st in flat_states], batch_docs=int(extra.batch_docs))
            for idx in range(len(rb_items)):
                sl = rb_raw[4 * idx : 4 * idx + 4]
                rb_cand_probs.append([x[0][0] for x in sl])
                rb_cand_logits.append([x[0][1] for x in sl])
        _mx_sync()
        r_rb = evaluate_reward_bench_2(rb_items, rb_cand_probs, rb_cand_logits)
        sub_reports["reward_bench_2"] = {k: v for k, v in r_rb.items() if not k.startswith("per_item_")}
        all_valid.extend(r_rb["per_item_valid"])
        all_scores.extend(r_rb["per_item_scores"])

        # Optional Tier-2 Banking77
        if b77_pack["items"]:
            b77_items = b77_pack["items"]
            b77_names = b77_pack["label_names"]
            if not is_laya:
                q_b77 = build_banking77_flat_qdef(b77_names)
                raw_b77 = eng.score_multi_states([(it["text"], [q_b77]) for it in b77_items], batch_docs=int(extra.batch_docs))
                probs_b77 = [r[0][0] for r in raw_b77]
                r_b77 = evaluate_multiclass_choice([int(it["label_id"]) for it in b77_items], probs_b77)
                sub_reports["banking77_k77"] = {k: v for k, v in r_b77.items() if not k.startswith("per_item_")}
                all_valid.extend(r_b77["per_item_valid"])
                all_scores.extend(r_b77["per_item_scores"])
            else:
                # Batched 2-stage 8-cluster hierarchy for Laya parity (0% option truncation).
                clusters = partition_banking77_clusters(b77_names, n_clusters=8)
                c_qdef = {
                    "id": "bank77_c1",
                    "type": "choice",
                    "instructions": "Select the banking intent cluster.",
                    "criteria": [f"cluster_{ci}: {', '.join(cl[:3]).replace('_', ' ')}" for ci, cl in enumerate(clusters)],
                }
                c2_qdefs = {
                    ci: {
                        "id": f"bank77_c2_{ci}",
                        "type": "choice",
                        "instructions": "Select the exact banking intent.",
                        "criteria": [nm.replace("_", " ") for nm in cl],
                    }
                    for ci, cl in enumerate(clusters)
                }
                c_encs = [eng.encode_item(it["text"], c_qdef, on_truncate="ignore") for it in b77_items]
                laya_opt_truncations += sum(1 for e in c_encs if e.get("truncated"))
                c_probs = eng.score_choice_batch(c_encs)
                b77_s1_valid = check_probability_validity(c_probs)

                b77_s2_idxs: list[int] = []
                b77_s2_cis: list[int] = []
                b77_s2_encs: list[dict[str, Any]] = []
                for idx, it in enumerate(b77_items):
                    if not b77_s1_valid[idx]:
                        continue
                    ci = int(np.argmax(np.asarray(c_probs[idx], dtype=np.float64)))
                    enc2 = eng.encode_item(it["text"], c2_qdefs[ci], on_truncate="ignore")
                    if enc2.get("truncated"):
                        laya_opt_truncations += 1
                    b77_s2_idxs.append(idx)
                    b77_s2_cis.append(ci)
                    b77_s2_encs.append(enc2)

                b77_valid = [False] * len(b77_items)
                b77_scores = [0.0] * len(b77_items)
                if b77_s2_encs:
                    s2_probs_batch = eng.score_choice_batch(b77_s2_encs)
                    s2_valid_batch = check_probability_validity(s2_probs_batch)
                    for idx, ci, p2, s2_ok in zip(b77_s2_idxs, b77_s2_cis, s2_probs_batch, s2_valid_batch):
                        is_ok = bool(b77_s1_valid[idx] and s2_ok)
                        b77_valid[idx] = is_ok
                        if is_ok:
                            pred_name = clusters[ci][int(np.argmax(np.asarray(p2, dtype=np.float64)))]
                            b77_scores[idx] = 1.0 if pred_name == b77_items[idx]["label_name"] else 0.0
                sub_reports["banking77_2stage"] = {
                    "n_items": len(b77_items),
                    "n_valid": int(sum(1 for v in b77_valid if v)),
                    "top1_accuracy": round(float(np.mean(b77_scores)) if b77_scores else 0.0, 6),
                }
                all_valid.extend(b77_valid)
                all_scores.extend(b77_scores)

        _mx_sync()
        wall_sec = time.perf_counter() - t0
        _, ref = guard.check()
        if ref is not None:
            guard_tripped = True

        # Headline Cat 2 macro composite stored in diagnostics; EOT uses exact sum(v_i * s_i) / wall_sec.
        headline_s = float(np.mean([r_rt["auroc"], r_cl["end_to_end_151_acc"], r_rb["macro_best_of_4_acc"]]))
        mem4 = _memory_4bucket_dict(probe)
        diag = {
            "backend": backend_name,
            "composite_macro_score": round(headline_s, 6),
            "ragtruth_auroc": r_rt["auroc"],
            "ragtruth_macro_f1": r_rt["macro_f1"],
            "ragtruth_brier": r_rt["brier"],
            "ragtruth_ece_15": r_rt["ece_15"],
            "clinc150_151_acc": r_cl["end_to_end_151_acc"],
            "clinc150_oos_auroc": r_cl["oos_auroc"],
            "rewardbench2_macro_bo4": r_rb["macro_best_of_4_acc"],
            "laya_clinc_option_truncations": laya_opt_truncations if is_laya else 0,
            "device_guard_tripped": guard_tripped,
            "repeat_loop_count": 0,
        }
        return _build_record(
            package="fast-decision-models",
            category=2,
            config_label=eff_label,
            benchmark="RAGTruth+CLINC150+RewardBench2",
            valid_flags=all_valid,
            item_scores=all_scores,
            wall_sec=wall_sec,
            diagnostics=diag,
            memory_4bucket_gb=mem4,
            sub_benchmarks=sub_reports,
            host_mem_gb=host_mem_gb,
        )
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Categories 3 & 4 Workers: fast-gemma4, fast-gpt-oss, optional OpenAI-compatible
# ---------------------------------------------------------------------------


def _format_mmlu_prompt(item: Mapping[str, Any]) -> str:
    letters = ("A", "B", "C", "D")
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    return (
        f"Question ({item['subject'].replace('_', ' ')}):\n{item['question']}\n\n"
        f"{opts}\n\nRespond with only 'Answer: <A|B|C|D>'."
    )


def _format_math_prompt(item: Mapping[str, Any]) -> str:
    return f"Solve the following math problem step by step and put your final answer inside \\boxed{{}}.\n\n{item['problem']}"


def _format_liquidai_prompt(item: Mapping[str, Any]) -> str:
    sch_str = json.dumps(item["schema"], ensure_ascii=False)
    return (
        f"Extract the requested fields from the document as a single valid JSON object matching this schema:\n"
        f"{sch_str}\n\nDocument:\n{item['document']}"
    )


def _format_bfcl_prompt(item: Mapping[str, Any]) -> str:
    q_turns = item.get("question") or []
    user_txt = ""
    if q_turns and isinstance(q_turns[0], list):
        user_txt = "\n".join(str(m.get("content") or "") for m in q_turns[0] if isinstance(m, Mapping))
    elif isinstance(q_turns, list):
        user_txt = "\n".join(str(m) for m in q_turns)
    tools_str = json.dumps(item.get("functions") or [], ensure_ascii=False)
    return (
        f"Available tools (JSON definitions):\n{tools_str}\n\n"
        f"User request:\n{user_txt}\n\n"
        f'Return a JSON object {{"tool_calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}. '
        f'If none of the tools match the user request, return {{"tool_calls": []}}.'
    )


def _run_llm_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    include_unconstrained: bool,
    home: Path,
    host_mem_gb: float,
    openai_url: str | None = None,
    openai_model: str | None = None,
) -> list[dict[str, Any]]:
    want_cats = {3, 4} if extra.categories == "all" else {int(x.strip()) for x in extra.categories.split(",") if x.strip()}
    probe = FourBucketMemoryProbe(model_directories=[home / "artifacts"] if home.exists() else None)
    guard = DeviceMemoryGuard()
    guard_tripped = False

    eng: Any = None
    ltok: Any = None
    eff_label = config_label
    batch_sz = int(extra.batch_size)

    if pkg_name == "fast-gemma4":
        from fast_gemma4.engine import FastGemma4Engine

        eff_mode = extra.mode or "gpu"
        eng = FastGemma4Engine(
            home=home,
            default_mode=eff_mode,
            ane_variant=extra.ane_variant,
            ane_ring_variant=extra.ane_variant,
            kv_i8=not bool(extra.no_kv_i8),
            local_files_only=True,
            warmup=True,
            warmup_ane=(eff_mode == "hetero"),
        )
        ltok = eng.tokenizer.llguidance_tokenizer()
        eff_label = config_label or f"{eff_mode}_{extra.ane_variant}_{'kvi8' if not extra.no_kv_i8 else 'kvbf16'}"
    elif pkg_name == "fast-gpt-oss":
        from fast_gpt_oss.engine import FastGptOssEngine

        eff_mode = extra.mode or "auto"
        eng = FastGptOssEngine(
            home=home,
            default_mode=eff_mode,
            default_reasoning_effort=extra.reasoning_effort or "low",
            local_files_only=True,
            warmup=True,
        )
        ltok = eng.tokenizer.llguidance_tokenizer()
        eff_label = config_label or f"gpt_oss_{eff_mode}"
    elif pkg_name == "openai-chat":
        if not openai_url:
            raise ValueError("openai-chat package requires --openai-url")
        eff_label = config_label or (openai_model or "openai_chat")
    else:
        raise ValueError(f"Unsupported LLM package: {pkg_name}")

    def _generate_batch(
        prompts: list[str],
        *,
        system_prompt: str | None = None,
        per_doc_schemas: list[dict[str, Any] | None] | None = None,
        max_tokens: int = 256,
        reasoning_effort: str | None = None,
    ) -> tuple[list[str], list[str], list[list[int]], dict[str, Any]]:
        if not prompts:
            return [], [], [], {"completion_tokens": 0, "mask_fallbacks": 0}
        if pkg_name == "openai-chat":
            texts: list[str] = []
            reasons: list[str] = []
            tot_comp = 0
            for idx, pr in enumerate(prompts):
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": pr})
                body: dict[str, Any] = {
                    "model": openai_model or "default",
                    "messages": msgs,
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                }
                if per_doc_schemas and per_doc_schemas[idx] is not None:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "extraction", "schema": per_doc_schemas[idx]},
                    }
                req = urllib.request.Request(
                    f"{openai_url.rstrip('/')}/v1/chat/completions",  # type: ignore[union-attr]
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                ch = payload["choices"][0]
                texts.append(str(ch["message"].get("content") or ""))
                reasons.append(str(ch.get("finish_reason") or "stop"))
                tot_comp += int(payload.get("usage", {}).get("completion_tokens", 0))
            return texts, reasons, [[] for _ in texts], {"completion_tokens": tot_comp, "mask_fallbacks": 0}

        # Compile per-doc schemas defensively so an unsupported upstream schema keyword never aborts a run.
        safe_grammars: list[Any] | None = None
        compile_fallbacks = 0
        if per_doc_schemas is not None:
            safe_grammars = []
            for sch in per_doc_schemas:
                if sch is None:
                    safe_grammars.append(None)
                    continue
                try:
                    if pkg_name == "fast-gemma4":
                        safe_grammars.append(eng.get_grammar(sch, whitespace_flexible=False))
                    else:
                        safe_grammars.append(eng._get_grammar(sch, whitespace_flexible=False))
                except Exception:
                    safe_grammars.append(None)
                    compile_fallbacks += 1

        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "per_doc_grammars": safe_grammars,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "batch_size": min(batch_sz, len(prompts)),
        }
        if pkg_name == "fast-gpt-oss" and reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        elif pkg_name == "fast-gemma4":
            kwargs["enable_thinking"] = False
            kwargs["speculative"] = False

        res = eng.generate_batch(prompts, **kwargs)
        _mx_sync()
        texts = [it.text for it in res.items]
        reasons = [it.finish_reason for it in res.items]
        tok_ids = [list(it.token_ids) for it in res.items]
        st = dict(res.stats)
        st["grammar_compile_fallbacks"] = compile_fallbacks
        return texts, reasons, tok_ids, st

    records: list[dict[str, Any]] = []
    try:
        guard.arm()
        # Untimed warmup
        _ = _generate_batch(["Reply OK."], max_tokens=8, reasoning_effort="skip")
        _mx_sync()
        _mx_reset_peak()

        # ---------------- Category 3: Chat, IF & Reasoning ----------------
        if 3 in want_cats:
            if_cap = 6 if smoke else max_items
            mm_cap = 12 if smoke else max_items
            math_cap = 4 if smoke else max_items

            ifb_items = load_ifbench_test(tier=tier, max_items=if_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            mm_items = load_mmlu_redux(
                tier=tier,
                subjects=lock["datasets"]["mmlu_redux_2"]["subjects"][:4] if smoke else None,
                max_items=mm_cap,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            math_items = load_math_500(tier=tier, max_items=math_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            ife_items = (
                load_ifeval(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
                if (tier == "tier2" and not smoke)
                else []
            )
            lb_items = (
                load_livebench_reasoning(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
                if (tier == "tier2" and not smoke)
                else []
            )

            t0 = time.perf_counter()
            sub3: dict[str, Any] = {}
            all_v3: list[bool] = []
            all_s3: list[float] = []
            tot_comp_toks = 0
            tot_rep_loops = 0
            tot_fin_len = 0

            # 3A: IFBench (+ IFEval in Tier-2)
            ifb_txt, ifb_fr, ifb_ids, ifb_st = _generate_batch(
                [it["prompt"] for it in ifb_items], max_tokens=256 if smoke else 512, reasoning_effort="skip"
            )
            r_ifb = evaluate_instruction_following(ifb_items, ifb_txt, finish_reasons=ifb_fr, token_ids_list=ifb_ids)
            sub3["ifbench_test"] = {k: v for k, v in r_ifb.items() if not k.startswith("per_item_")}
            all_v3.extend(r_ifb["per_item_valid"])
            all_s3.extend(r_ifb["per_item_scores"])
            tot_comp_toks += int(ifb_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_ifb["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in ifb_fr if fr == "length")

            if_scores_for_comp = [float(r_ifb["prompt_loose_acc"])]
            if ife_items:
                ife_txt, ife_fr, ife_ids, ife_st = _generate_batch(
                    [it["prompt"] for it in ife_items], max_tokens=512, reasoning_effort="skip"
                )
                r_ife = evaluate_instruction_following(ife_items, ife_txt, finish_reasons=ife_fr, token_ids_list=ife_ids)
                sub3["ifeval"] = {k: v for k, v in r_ife.items() if not k.startswith("per_item_")}
                all_v3.extend(r_ife["per_item_valid"])
                all_s3.extend(r_ife["per_item_scores"])
                tot_comp_toks += int(ife_st.get("completion_tokens", 0))
                tot_rep_loops += int(r_ife["repeat_loop_count"])
                tot_fin_len += sum(1 for fr in ife_fr if fr == "length")
                if_scores_for_comp.append(float(r_ife["prompt_loose_acc"]))

            # 3B: MMLU-Redux 2.0 (max_tokens=16)
            mm_txt, mm_fr, mm_ids, mm_st = _generate_batch(
                [_format_mmlu_prompt(it) for it in mm_items], max_tokens=16, reasoning_effort="skip"
            )
            r_mm = evaluate_mmlu_redux(mm_items, mm_txt, finish_reasons=mm_fr, token_ids_list=mm_ids)
            sub3["mmlu_redux_2"] = {k: v for k, v in r_mm.items() if not k.startswith("per_item_")}
            all_v3.extend(r_mm["per_item_valid"])
            all_s3.extend(r_mm["per_item_scores"])
            tot_comp_toks += int(mm_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_mm["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in mm_fr if fr == "length")

            # 3C: MATH-500 (+ LiveBench-Reasoning in Tier-2)
            math_txt, math_fr, math_ids, math_st = _generate_batch(
                [_format_math_prompt(it) for it in math_items], max_tokens=512, reasoning_effort="low"
            )
            r_math = evaluate_math_500(math_items, math_txt, finish_reasons=math_fr, token_ids_list=math_ids)
            sub3["math_500"] = {k: v for k, v in r_math.items() if not k.startswith("per_item_")}
            all_v3.extend(r_math["per_item_valid"])
            all_s3.extend(r_math["per_item_scores"])
            tot_comp_toks += int(math_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_math["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in math_fr if fr == "length")

            reason_scores = [float(r_math["verified_accuracy"])]
            if lb_items:
                lb_txt, lb_fr, lb_ids, lb_st = _generate_batch(
                    [it["prompt"] for it in lb_items], max_tokens=512, reasoning_effort="low"
                )
                r_lb = evaluate_livebench_reasoning(lb_items, lb_txt, finish_reasons=lb_fr, token_ids_list=lb_ids)
                sub3["livebench_reasoning"] = {k: v for k, v in r_lb.items() if not k.startswith("per_item_")}
                all_v3.extend(r_lb["per_item_valid"])
                all_s3.extend(r_lb["per_item_scores"])
                tot_comp_toks += int(lb_st.get("completion_tokens", 0))
                tot_rep_loops += int(r_lb["repeat_loop_count"])
                tot_fin_len += sum(1 for fr in lb_fr if fr == "length")
                reason_scores.append(float(r_lb["verified_accuracy"]))

            _mx_sync()
            wall3 = time.perf_counter() - t0
            _, ref3 = guard.check()
            if ref3 is not None:
                guard_tripped = True
            mem3 = _memory_4bucket_dict(probe)
            comp3 = float(np.mean([float(np.mean(if_scores_for_comp)), float(r_mm["exact_accuracy"]), float(np.mean(reason_scores))]))
            diag3 = {
                "composite_macro_score": round(comp3, 6),
                "if_prompt_loose_acc": round(float(np.mean(if_scores_for_comp)), 6),
                "mmlu_redux_exact_acc": r_mm["exact_accuracy"],
                "math_reasoning_acc": round(float(np.mean(reason_scores)), 6),
                "avg_completion_tokens": round(float(tot_comp_toks) / float(max(1, len(all_v3))), 2),
                "repeat_4gram_rate": round(float(r_ifb["repeat_4gram_rate"]), 6),
                "repeat_loop_count": tot_rep_loops,
                "finish_length_count": tot_fin_len,
                "device_guard_tripped": guard_tripped,
            }
            records.append(
                _build_record(
                    package=pkg_name,
                    category=3,
                    config_label=f"{eff_label}_cat3",
                    benchmark="IFBench+MMLU-Redux+MATH-500",
                    valid_flags=all_v3,
                    item_scores=all_s3,
                    wall_sec=wall3,
                    diagnostics=diag3,
                    memory_4bucket_gb=mem3,
                    sub_benchmarks=sub3,
                    host_mem_gb=host_mem_gb,
                )
            )

        # ---------------- Category 4: Structured JSON & Tool AST ----------------
        if 4 in want_cats:
            lq_cap = 8 if smoke else max_items
            bfcl_per = 2 if smoke else (max(1, int(max_items) // 5) if max_items else None)

            lq_items = load_liquidai_extraction(tier=tier, max_items=lq_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            bfcl_items = load_bfcl_v3(tier=tier, per_split=bfcl_per, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            jsb_items = load_jsonschemabench(tier="tier1" if smoke else tier, per_subset=10 if smoke else None, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)

            # Auxiliary JSONSchemaBench llguidance compiler diagnostic (< 5s CPU)
            jsb_diag = evaluate_jsonschemabench_compiler(jsb_items, llguidance_tokenizer=ltok)

            llg_modes = [not bool(extra.no_llguidance)]
            if include_unconstrained and True in llg_modes and False not in llg_modes:
                llg_modes.append(False)

            for use_llg in llg_modes:
                t0 = time.perf_counter()
                sub4: dict[str, Any] = {"jsonschemabench_compiler": jsb_diag}
                all_v4: list[bool] = []
                all_s4: list[float] = []
                tot_comp_4 = 0
                tot_fb_4 = 0
                tot_rep_4 = 0
                tot_len_4 = 0

                # 4A: LiquidAI/structured-extraction
                lq_prompts = [_format_liquidai_prompt(it) for it in lq_items]
                lq_schemas = [it["schema"] for it in lq_items] if use_llg else None
                lq_txt, lq_fr, lq_ids, lq_st = _generate_batch(
                    lq_prompts,
                    system_prompt="Extract fields strictly from the document as compact JSON matching the schema.",
                    per_doc_schemas=lq_schemas,
                    max_tokens=256 if smoke else 512,
                    reasoning_effort="skip",
                )
                r_lq = evaluate_structured_extraction(lq_items, lq_txt, finish_reasons=lq_fr, token_ids_list=lq_ids)
                sub4["liquidai_extraction"] = {k: v for k, v in r_lq.items() if not k.startswith("per_item_")}
                all_v4.extend(r_lq["per_item_valid"])
                all_s4.extend(r_lq["per_item_scores"])
                tot_comp_4 += int(lq_st.get("completion_tokens", 0))
                tot_fb_4 += int(lq_st.get("mask_fallbacks", lq_st.get("mask_hits", 0)) or 0)
                tot_rep_4 += int(r_lq["repeat_loop_count"])
                tot_len_4 += sum(1 for fr in lq_fr if fr == "length")

                # 4B: BFCL v3 AST + Irrelevance
                bf_prompts = [_format_bfcl_prompt(it) for it in bfcl_items]
                bf_schemas = [build_bfcl_response_schema(it["functions"]) for it in bfcl_items] if use_llg else None
                bf_txt, bf_fr, bf_ids, bf_st = _generate_batch(
                    bf_prompts,
                    system_prompt="Select matching tool calls or return empty tool_calls if none apply. Output raw JSON only.",
                    per_doc_schemas=bf_schemas,
                    max_tokens=192 if smoke else 384,
                    reasoning_effort="skip",
                )
                r_bf = evaluate_bfcl_v3(bfcl_items, bf_txt, finish_reasons=bf_fr, token_ids_list=bf_ids)
                sub4["bfcl_v3"] = {k: v for k, v in r_bf.items() if not k.startswith("per_item_")}
                all_v4.extend(r_bf["per_item_valid"])
                all_s4.extend(r_bf["per_item_scores"])
                tot_comp_4 += int(bf_st.get("completion_tokens", 0))
                tot_fb_4 += int(bf_st.get("mask_fallbacks", bf_st.get("mask_hits", 0)) or 0)
                tot_rep_4 += int(r_bf["repeat_loop_count"])
                tot_len_4 += sum(1 for fr in bf_fr if fr == "length")

                _mx_sync()
                wall4 = time.perf_counter() - t0
                _, ref4 = guard.check()
                if ref4 is not None:
                    guard_tripped = True
                mem4 = _memory_4bucket_dict(probe)
                fb_rate = float(tot_fb_4) / float(max(1, tot_comp_4))
                diag4 = {
                    "llguidance": bool(use_llg),
                    "grounded_field_f1": r_lq["grounded_field_f1"],
                    "valid_schema_rate": r_lq["valid_schema_rate"],
                    "bfcl_ast_acc": r_bf["overall_ast_acc"],
                    "bfcl_abstention_acc": r_bf["abstention_acc"],
                    "jsb_compile_rate": jsb_diag["compile_success_rate"],
                    "jsb_gct_p50_ms": jsb_diag["gct_p50_ms"],
                    "avg_completion_tokens": round(float(tot_comp_4) / float(max(1, len(all_v4))), 2),
                    "mask_fallback_rate": round(fb_rate, 6),
                    "repeat_4gram_rate": round(0.5 * (float(r_lq["repeat_4gram_rate"]) + float(r_bf["repeat_4gram_rate"])), 6),
                    "repeat_loop_count": tot_rep_4,
                    "finish_length_count": tot_len_4,
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=4,
                        config_label=f"{eff_label}_{'llg_on' if use_llg else 'llg_off'}",
                        benchmark=f"LiquidAI/structured-extraction({len(lq_items)})+BFCL_v3({len(bfcl_items)})",
                        valid_flags=all_v4,
                        item_scores=all_s4,
                        wall_sec=wall4,
                        diagnostics=diag4,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks=sub4,
                        host_mem_gb=host_mem_gb,
                    )
                )
    finally:
        if eng is not None and hasattr(eng, "close"):
            eng.close()
    return records


def _run_worker_main(args: argparse.Namespace) -> int:
    """Isolated child subprocess entrypoint executing one package+config under wired_limit & 4-bucket probes."""
    lock, _ = load_lockfile(args.lockfile)
    cache_dir = resolve_cache_dir(args.cache_dir)
    env_meta = collect_env_telemetry()
    host_mem = _host_mem_gb(env_meta)
    pkg_name = str(args.worker_package)
    extra = _parse_extra_flags(shlex.split(args.worker_extra_flags or ""))

    if pkg_name == "openai-chat":
        home = _REPO_ROOT
    else:
        spec = _SPEC_BY_NAME[pkg_name]
        home = resolve_package_home(spec)

    use_wl = not bool(args.no_wired_limit)
    ctx = wired_limit() if use_wl else _null_ctx()
    with ctx:
        if pkg_name in ("fast-bge-small", "fast-embeddinggemma-2", "fast-pplx"):
            recs = _run_cat1_worker(
                pkg_name,
                tier=args.tier,
                config_label=args.worker_label or "",
                extra=extra,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=bool(args.local_files_only),
                max_items=args.max_items,
                smoke=bool(args.smoke),
                home=home,
                host_mem_gb=host_mem,
            )
        elif pkg_name == "fast-decision-models":
            backends = ["laya-ane", "kev-0.8b"] if extra.backend == "all" else [extra.backend]
            recs = []
            for bname in backends:
                try:
                    recs.append(
                        _run_cat2_single_backend(
                            bname,
                            tier=args.tier,
                            config_label=args.worker_label or "",
                            extra=extra,
                            lock=lock,
                            cache_dir=cache_dir,
                            local_files_only=bool(args.local_files_only),
                            max_items=args.max_items,
                            smoke=bool(args.smoke),
                            home=home,
                            host_mem_gb=host_mem,
                        )
                    )
                except FileNotFoundError as exc:
                    recs.append(
                        {
                            "package": pkg_name,
                            "category": 2,
                            "config_label": bname,
                            "benchmark": "RAGTruth+CLINC150+RewardBench2",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    )
        else:
            recs = _run_llm_worker(
                pkg_name,
                tier=args.tier,
                config_label=args.worker_label or "",
                extra=extra,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=bool(args.local_files_only),
                max_items=args.max_items,
                smoke=bool(args.smoke),
                include_unconstrained=bool(args.include_unconstrained_baseline),
                home=home,
                host_mem_gb=host_mem,
                openai_url=args.openai_url,
                openai_model=args.openai_model,
            )

    print(_WORKER_MARKER + json.dumps({"records": recs}, ensure_ascii=False), flush=True)
    return 0


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield None


# ---------------------------------------------------------------------------
# Coordinator Subprocess Spawning, ABBA Interleaving & Formatting
# ---------------------------------------------------------------------------


def _spawn_isolated_worker(
    *,
    python_exe: str,
    pkg_name: str,
    label: str,
    extra_flags: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cmd = [
        python_exe,
        str(Path(__file__).resolve()),
        "_worker",
        "--worker-package",
        pkg_name,
        "--worker-label",
        label,
        "--worker-extra-flags",
        extra_flags,
        "--tier",
        args.tier,
    ]
    if args.lockfile:
        cmd.extend(["--lockfile", str(args.lockfile)])
    if args.cache_dir:
        cmd.extend(["--cache-dir", str(args.cache_dir)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    if args.smoke:
        cmd.append("--smoke")
    if args.include_unconstrained_baseline:
        cmd.append("--include-unconstrained-baseline")
    if args.no_wired_limit:
        cmd.append("--no-wired-limit")
    if args.openai_url:
        cmd.extend(["--openai-url", str(args.openai_url)])
    if args.openai_model:
        cmd.extend(["--openai-model", str(args.openai_model)])

    env = dict(os.environ)
    env["M4_GPU_LOCK_HELD"] = "1"
    if pkg_name in _SPEC_BY_NAME:
        spec = _SPEC_BY_NAME[pkg_name]
        env.setdefault(spec.home_env_var, str(resolve_package_home(spec)))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(args.timeout_sec),
        )
    except subprocess.TimeoutExpired:
        return [
            {
                "package": pkg_name,
                "category": _SPEC_BY_NAME[pkg_name].category if pkg_name in _SPEC_BY_NAME else 3,
                "config_label": label or "default",
                "status": "failed",
                "error": f"timeout after {args.timeout_sec}s",
                "gate_verdict": "FAIL",
            }
        ]

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_WORKER_MARKER):
            payload = json.loads(line[len(_WORKER_MARKER) :])
            return list(payload.get("records", []))

    err_tail = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip().splitlines()
    reason = err_tail[-1] if err_tail else f"exit={proc.returncode}"
    low = (proc.stderr + "\n" + proc.stdout).lower()
    is_skip = any(tok in low for tok in ("filenotfounderror", "not found", "missing", "not staged"))
    return [
        {
            "package": pkg_name,
            "category": _SPEC_BY_NAME[pkg_name].category if pkg_name in _SPEC_BY_NAME else 3,
            "config_label": label or "default",
            "status": "skipped" if is_skip else "failed",
            "error": reason[:200],
            "gate_verdict": "SKIP" if is_skip else "FAIL",
        }
    ]


def _aggregate_abba_records(runs: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate repeated ABBA worker outputs by median wall_sec & EOT while preserving item scores."""
    if not runs:
        return []
    if len(runs) == 1:
        return [dict(r) for r in runs[0]]

    n_slots = len(runs[0])
    aggregated: list[dict[str, Any]] = []
    for slot_i in range(n_slots):
        slot_runs = [r[slot_i] for r in runs if slot_i < len(r) and r[slot_i].get("status") == "ok"]
        if not slot_runs:
            aggregated.append(dict(runs[0][slot_i]))
            continue
        med_wall = float(statistics.median([float(r["wall_sec"]) for r in slot_runs]))
        rep = dict(min(slot_runs, key=lambda r: abs(float(r["wall_sec"]) - med_wall)))
        rep["wall_sec"] = round(med_wall, 4)
        rep["raw_items_per_sec"] = round(float(rep["n_items"]) / max(med_wall, 1e-6), 4)
        rep["valid_outputs_per_sec"] = round(float(rep["n_valid"]) / max(med_wall, 1e-6), 4)
        rep["effective_outcome_per_sec"] = round(
            float(rep["raw_items_per_sec"]) * float(rep["objective_score_e2e"]), 4
        )
        rep["abba_repeats"] = len(slot_runs)
        rep["wall_series_sec"] = [round(float(r["wall_sec"]), 4) for r in slot_runs]
        aggregated.append(rep)
    return aggregated


def parse_abba_arm(raw: str) -> tuple[str, str]:
    """Parse ``label:flags`` CLI spec for ``--abba-compare``."""
    if ":" not in raw:
        return raw.strip(), ""
    lbl, flags = raw.split(":", 1)
    return lbl.strip(), flags.strip()


def format_outcomes_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    """Render concise Markdown summary table of External Objective Outcome Benchmark results."""
    lines = [
        "| Package | Cat | Config | Benchmark | N (Valid) | `S_obj,e2e` | `EOT` (u/s) | Wall (s) | 4-Bucket Mem (GB) | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        pkg = r.get("package", "-")
        cat = r.get("category", "-")
        cfg = r.get("config_label", "-")
        bench = r.get("benchmark", "-")
        if r.get("status") != "ok":
            lines.append(f"| `{pkg}` | {cat} | `{cfg}` | {bench} | - | - | - | - | - | `{r.get('gate_verdict', r.get('status', 'FAIL'))}` |")
            continue
        n_str = f"{r['n_items']} ({r['n_valid']})"
        s_val = f"{float(r['objective_score_e2e']):.4f}"
        eot_val = f"{float(r['effective_outcome_per_sec']):.3f}"
        wall = f"{float(r['wall_sec']):.2f}"
        mem = float((r.get("memory_4bucket_gb") or {}).get("combined_with_model_mapped_gb", 0.0))
        lines.append(
            f"| `{pkg}` | {cat} | `{cfg}` | {bench} | {n_str} | {s_val} | {eot_val} | {wall} | {mem:.3f} | `{r.get('gate_verdict', 'PASS')}` |"
        )
    return "\n".join(lines)


def _cmd_setup_data(args: argparse.Namespace) -> int:
    ds_list = None if args.datasets in (None, "", "all") else [x.strip() for x in args.datasets.split(",") if x.strip()]
    report = stage_tier_datasets(
        tier=args.tier,
        datasets=ds_list,
        lock_path=args.lockfile,
        cache_dir=args.cache_dir,
        verify_checksums=not bool(args.no_verify_checksums),
    )
    print(json.dumps(report, indent=2))
    return 0


def _compare_record_key(rec: Mapping[str, Any]) -> tuple[Any, ...]:
    """Disambiguating comparison key preventing Cat 2 (Laya/Kev) and Cat 4 (llg_on/llg_off) collisions."""
    diag = rec.get("diagnostics") or {}
    return (
        rec.get("package"),
        rec.get("category"),
        rec.get("benchmark"),
        diag.get("backend"),
        diag.get("llguidance"),
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    base_doc = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand_doc = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    host_mem = float(args.host_mem_gb) if args.host_mem_gb is not None else float(
        (cand_doc.get("environment") or {}).get("mem_gb") or 0.0
    )

    base_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for br in base_doc.get("results", []):
        base_by_key.setdefault(_compare_record_key(br), []).append(br)

    comparisons: list[dict[str, Any]] = []
    any_fail = False
    for cr in cand_doc.get("results", []):
        key = _compare_record_key(cr)
        bucket = base_by_key.get(key)
        if not bucket:
            continue
        if len(bucket) == 1:
            br = bucket[0]
        else:
            exact = [r for r in bucket if r.get("config_label") == cr.get("config_label")]
            if len(exact) != 1:
                any_fail = True
                comparisons.append(
                    {
                        "package": cr.get("package"),
                        "category": cr.get("category"),
                        "benchmark": cr.get("benchmark"),
                        "baseline_label": None,
                        "candidate_label": cr.get("config_label"),
                        "verdict": "FAIL",
                        "reasons": [f"Ambiguous baseline match ({len(bucket)} records) for key {key}"],
                    }
                )
                continue
            br = exact[0]
        b_eff = br.get("per_item_effective_scores")
        c_eff = cr.get("per_item_effective_scores")
        use_paired = isinstance(b_eff, list) and isinstance(c_eff, list) and len(b_eff) == len(c_eff) and len(b_eff) > 0
        ev = evaluate_promotion_gates(
            br,
            cr,
            profile=args.gate_profile,
            min_eot_ratio=float(args.min_eot_ratio),
            host_mem_gb=host_mem if host_mem > 0 else None,
            base_effective_scores=b_eff if use_paired else None,
            cand_effective_scores=c_eff if use_paired else None,
        )
        if ev.verdict != "PASS":
            any_fail = True
        comparisons.append(
            {
                "package": cr.get("package"),
                "category": cr.get("category"),
                "benchmark": cr.get("benchmark"),
                "baseline_label": br.get("config_label"),
                "candidate_label": cr.get("config_label"),
                **ev.to_dict(),
            }
        )
    if not comparisons:
        out_empty = {
            "comparisons": [],
            "verdict": "FAIL",
            "error": "No matching records found between baseline and candidate reports.",
        }
        print(json.dumps(out_empty, indent=2))
        return 1
    print(json.dumps({"comparisons": comparisons, "verdict": "FAIL" if any_fail else "PASS"}, indent=2))
    return 1 if any_fail else 0


def _cmd_run(args: argparse.Namespace) -> int:
    lock, lock_sha = load_lockfile(args.lockfile)
    env_meta = collect_env_telemetry()
    host_mem = _host_mem_gb(env_meta)
    quiescence = check_system_quiescence()

    raw_pkgs = args.packages.strip()
    if raw_pkgs == "all":
        pkg_names = [s.name for s in PACKAGE_SPECS]
    else:
        pkg_names = [p.strip() for p in raw_pkgs.split(",") if p.strip()]
        for pn in pkg_names:
            if pn not in _SPEC_BY_NAME and pn != "openai-chat":
                raise ValueError(f"Unknown package {pn!r}; choose from all, {', '.join(_SPEC_BY_NAME)}, openai-chat")

    if args.skip_gpt_oss or (0 < host_mem <= 24.5 and "fast-gpt-oss" in pkg_names and raw_pkgs == "all"):
        pkg_names = [p for p in pkg_names if p != "fast-gpt-oss"]

    abba_arms = [parse_abba_arm(x) for x in (args.abba_compare or [])]
    if args.dry_run:
        plan = {
            "dry_run": True,
            "tier": args.tier,
            "smoke": bool(args.smoke),
            "packages": pkg_names,
            "abba_arms": [{"label": lbl, "flags": fl} for lbl, fl in abba_arms],
            "dataset_lock_sha256": lock_sha,
            "cache_dir": str(resolve_cache_dir(args.cache_dir)),
            "environment": env_meta,
            "quiescence_warnings": quiescence,
        }
        print(json.dumps(plan, indent=2))
        return 0

    if quiescence and not args.allow_contended:
        print(f"QUIESCENCE WARNING ({len(quiescence)} active contenders):", file=sys.stderr)
        for line in quiescence:
            print(f"  {line}", file=sys.stderr)
        print("Aborting: re-run when quiet or pass --allow-contended.", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    cooldown = max(0.0, float(args.cooldown_sec))

    with _hw_lock():
        for idx, pkg in enumerate(pkg_names, 1):
            if idx > 1 and cooldown > 0.0:
                time.sleep(cooldown)
            print(f"[{idx}/{len(pkg_names)}] Running outcome benchmarks for {pkg} ({args.tier})...", file=sys.stderr, flush=True)

            if len(abba_arms) >= 2:
                # Interleaved A-B-B-A execution in isolated child subprocesses.
                (lbl_a, fl_a), (lbl_b, fl_b) = abba_arms[0], abba_arms[1]
                rounds = max(2, int(args.abba_rounds or 2))
                schedule = ["A", "B", "B", "A"] * max(1, rounds // 2)
                if rounds % 2 == 1:
                    schedule.extend(["A", "B"])
                runs_a: list[list[dict[str, Any]]] = []
                runs_b: list[list[dict[str, Any]]] = []
                for step_i, tag in enumerate(schedule):
                    if step_i > 0 and cooldown > 0.0:
                        time.sleep(cooldown)
                    lbl, fl = (lbl_a, fl_a) if tag == "A" else (lbl_b, fl_b)
                    print(f"  ABBA step {step_i + 1}/{len(schedule)} [{tag}:{lbl}]...", file=sys.stderr, flush=True)
                    recs = _spawn_isolated_worker(
                        python_exe=args.python, pkg_name=pkg, label=lbl, extra_flags=fl, args=args
                    )
                    (runs_a if tag == "A" else runs_b).append(recs)
                agg_a = _aggregate_abba_records(runs_a)
                agg_b = _aggregate_abba_records(runs_b)
                for ra, rb in zip(agg_a, agg_b):
                    if ra.get("status") == "ok" and rb.get("status") == "ok":
                        ea = ra.get("per_item_effective_scores")
                        eb = rb.get("per_item_effective_scores")
                        if isinstance(ea, list) and isinstance(eb, list) and len(ea) == len(eb) and len(ea) > 0:
                            paired = compute_paired_se(ea, eb)
                            rb["se_paired_vs_baseline"] = round(float(paired["se_paired"]), 6)
                        ev = evaluate_promotion_gates(
                            ra,
                            rb,
                            profile=args.gate_profile,
                            min_eot_ratio=float(args.min_eot_ratio),
                            host_mem_gb=host_mem if host_mem > 0 else None,
                            base_effective_scores=ea if (isinstance(ea, list) and isinstance(eb, list) and len(ea) == len(eb)) else None,
                            cand_effective_scores=eb if (isinstance(ea, list) and isinstance(eb, list) and len(ea) == len(eb)) else None,
                        )
                        rb["gate_verdict"] = ev.verdict
                        rb["gate_reasons"] = list(ev.reasons)
                        comparisons.append(
                            {
                                "package": pkg,
                                "category": rb.get("category"),
                                "benchmark": rb.get("benchmark"),
                                "baseline_label": ra.get("config_label"),
                                "candidate_label": rb.get("config_label"),
                                "abba_order": schedule,
                                **ev.to_dict(),
                            }
                        )
                results.extend(agg_a)
                results.extend(agg_b)
            else:
                lbl, fl = abba_arms[0] if abba_arms else ("", args.extra_args or "")
                recs = _spawn_isolated_worker(
                    python_exe=args.python, pkg_name=pkg, label=lbl, extra_flags=fl, args=args
                )
                results.extend(recs)

    # Strip verbose per-item score vectors only when requested; keep compact by default for `compare`.
    if args.omit_item_scores:
        for r in results:
            r.pop("per_item_effective_scores", None)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_epoch": int(time.time()),
        "tier": args.tier,
        "smoke": bool(args.smoke),
        "environment": {
            "hw_model": env_meta.get("hw_model"),
            "chip": env_meta.get("brand"),
            "mem_gb": env_meta.get("mem_gb"),
            "macos_version": env_meta.get("macos_version"),
            "mlx_version": env_meta.get("mlx_version"),
            "python_version": platform.python_version(),
        },
        "dataset_lock_sha256": lock_sha,
        "results": results,
        "comparisons": comparisons,
    }
    md = format_outcomes_markdown(results)

    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_m = Path(args.markdown_out)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md + "\n", encoding="utf-8")

    print(md)
    failed = [r for r in results if r.get("status") == "failed"]
    comp_failed = [c for c in comparisons if c.get("verdict") == "FAIL"]
    return 1 if (failed or comp_failed) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark-outcomes",
        description="External Public Objective Outcome Benchmark Suite for fast-silicon.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sd = sub.add_parser("setup-data", help="Stage & verify pinned HF dataset shards (0 GPU work)")
    sd.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    sd.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'")
    sd.add_argument("--cache-dir", default=None)
    sd.add_argument("--lockfile", default=None)
    sd.add_argument("--no-verify-checksums", action="store_true")
    sd.set_defaults(func=_cmd_setup_data)

    rn = sub.add_parser("run", help="Run isolated Tier-1 / Tier-2 objective outcome benchmarks under GPU lock")
    rn.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    rn.add_argument("--packages", default="all", help="Comma-separated package list or 'all'")
    rn.add_argument("--abba-compare", nargs="+", default=None, help="Two 'label:flags' arms for interleaved ABBA")
    rn.add_argument("--abba-rounds", type=int, default=2, help="Interleaved ABBA rounds when --abba-compare is set")
    rn.add_argument("--gate-profile", choices=("lossless", "quantized"), default="lossless")
    rn.add_argument("--min-eot-ratio", type=float, default=1.0)
    rn.add_argument("--include-unconstrained-baseline", action="store_true")
    rn.add_argument("--extra-args", default="", help="Forwarded worker flags (e.g. '--mode hetero2')")
    rn.add_argument("--max-items", type=int, default=None, help="Optional per-sub-benchmark item cap")
    rn.add_argument("--smoke", action="store_true", help="Fast end-to-end smoke slice across selected packages")
    rn.add_argument("--openai-url", default=None)
    rn.add_argument("--openai-model", default=None)
    rn.add_argument("--cache-dir", default=None)
    rn.add_argument("--lockfile", default=None)
    rn.add_argument("--local-files-only", action="store_true")
    rn.add_argument("--python", default=sys.executable)
    rn.add_argument("--cooldown-sec", type=float, default=1.0)
    rn.add_argument("--timeout-sec", type=int, default=900)
    rn.add_argument("--no-wired-limit", action="store_true")
    rn.add_argument("--allow-contended", action="store_true")
    rn.add_argument("--skip-gpt-oss", action="store_true")
    rn.add_argument("--omit-item-scores", action="store_true")
    rn.add_argument("--dry-run", action="store_true")
    rn.add_argument("--json-out", default=None)
    rn.add_argument("--markdown-out", default=None)
    rn.set_defaults(func=_cmd_run)

    cp = sub.add_parser("compare", help="Compare two saved outcomes_*.json reports via Two-Gate promotion rules")
    cp.add_argument("baseline", help="Path to baseline outcomes JSON")
    cp.add_argument("candidate", help="Path to candidate outcomes JSON")
    cp.add_argument("--gate-profile", choices=("lossless", "quantized"), default="lossless")
    cp.add_argument("--min-eot-ratio", type=float, default=1.0)
    cp.add_argument("--host-mem-gb", type=float, default=None)
    cp.set_defaults(func=_cmd_compare)

    wk = sub.add_parser("_worker", help=argparse.SUPPRESS)
    wk.add_argument("--worker-package", required=True)
    wk.add_argument("--worker-label", default="")
    wk.add_argument("--worker-extra-flags", default="")
    wk.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    wk.add_argument("--lockfile", default=None)
    wk.add_argument("--cache-dir", default=None)
    wk.add_argument("--local-files-only", action="store_true")
    wk.add_argument("--max-items", type=int, default=None)
    wk.add_argument("--smoke", action="store_true")
    wk.add_argument("--include-unconstrained-baseline", action="store_true")
    wk.add_argument("--no-wired-limit", action="store_true")
    wk.add_argument("--openai-url", default=None)
    wk.add_argument("--openai-model", default=None)
    wk.set_defaults(func=_run_worker_main)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
