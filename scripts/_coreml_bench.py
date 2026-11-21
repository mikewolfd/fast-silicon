"""Warm throughput, latency and compute plans for the public Core ML benchmark adapters."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from _outcome_public_backends import CoreMlKev08bBackend, CoreMlLayaBackend, build_public_cat1_adapter


def encoder_workload(bge: bool) -> list[list[int]]:
    """Match the seed, IDs and document counts of the native monorepo encoder CLIs."""
    rng = np.random.default_rng(42)
    if bge:
        lengths = rng.integers(32, 465, size=1000)
        bos, eos, low, high = 101, 102, 1000, 29000
    else:
        lengths = rng.integers(48, 505, size=2000)
        bos, eos, low, high = 2, 1, 100, 200000
    return [
        [bos, *rng.integers(low, high, size=int(length) - 2, dtype=np.int32).tolist(), eos]
        for length in lengths
    ]


def _placement(info: dict[str, Any]) -> dict[str, Any]:
    import coremltools as ct
    from fast_shared.coreai_pipeline import inspect_coreml_bundle_placement

    paths = info.get("compiled_paths") or {"text": info["compiled_path"]}
    units = ct.ComputeUnit[info["compute_units"]]
    target = "gpu" if units == ct.ComputeUnit.CPU_AND_GPU else "ane"
    plans = {}
    for label, path in paths.items():
        try:
            plan = inspect_coreml_bundle_placement(
                path, target_device=target, compute_units=units, include_ops=True
            )
            if not plan["total_ops"]:
                raise ValueError("Core ML returned no inspectable operations for this bundle")
            by_function = defaultdict(Counter)
            for op in plan["operations"]:
                by_function[op["function"]][op["preferred_device"]] += 1
            plans[label] = {
                "bundle": Path(path).name,
                "device_counts": plan["device_counts"],
                "total_ops": plan["total_ops"],
                "by_function": dict(by_function),
            }
        except (RuntimeError, ValueError, OSError) as exc:
            plans[label] = {"bundle": Path(path).name, "error": str(exc)}
    return plans


def _encoder(pkg: str, run_pair: Callable, cooldown_sec: float) -> tuple[dict, dict]:
    bge = pkg == "coreml-bge-small"
    docs = encoder_workload(bge)
    query = [docs[0][:24]]
    tokens = sum(map(len, docs))
    adapter = build_public_cat1_adapter(pkg, tier="tier1")
    encode = adapter.encode_token_ids
    last_vectors = {}
    try:
        if encode is None:
            raise ValueError(f"{pkg} has no token-ID entry point")
        for _ in range(3):
            encode(query, False)
        query_ms = [1000.0 * encode(query, False)[1] for _ in range(30)]

        def corpus(pack: bool) -> float:
            vectors, elapsed = encode(docs, pack)
            if vectors.shape != (len(docs), 384 if bge else 768) or not np.isfinite(vectors).all():
                raise ValueError(f"{pkg} returned invalid corpus embeddings")
            last_vectors[pack] = vectors
            return elapsed

        if bge:
            corpus(False)
            times = [corpus(False) for _ in range(2)]
            sweep = {"timing_order": ["single", "single"]}
        else:
            pair = run_pair(lambda: corpus(True), lambda: corpus(False), cooldown_sec=cooldown_sec,
                            rounds=2, use_mlx_abba=False)
            times = [r["payload"] for r in pair["a_runs"]]
            single = [r["payload"] for r in pair["b_runs"]]
            # The adapters return L2-normalized vectors in original document order.
            cosines = np.einsum("ij,ij->i", last_vectors[True], last_vectors[False])
            sweep = {
                "timing_order": ["packed" if arm == "A" else "single" for arm in pair["order"]],
                "single_embed_sec": single,
                "single_embed_tok_s": round(tokens / median(single), 1),
                "single_embed_docs_per_sec": round(len(docs) / median(single), 2),
                "pack_256_speedup": round(median(single) / median(times), 4),
                "min_pack_single_cosine": round(float(cosines.min()), 7),
            }
        tok_s = tokens / median(times)
        b1_ms = median(query_ms)
        metrics = {
            "runtime": adapter.harness_label,
            "seed": 42,
            "documents": len(docs),
            "tokens": tokens,
            "query_tokens": len(query[0]),
            "timing_scope": "token IDs through normalized vectors inside the model process",
            "query_samples_ms": query_ms,
            "b1_ms": round(b1_ms, 3),
            "corpus_sec": times,
            "tok_s": round(tok_s, 1),
            "docs_per_sec": round(len(docs) / median(times), 2),
            **sweep,
            "highlight": f"{tok_s:,.0f} tok/s, {b1_ms:.2f} ms for one query",
        }
    finally:
        adapter.close()
    return metrics, adapter.telemetry() or {}


def _decision(pkg: str, run_pair: Callable, cooldown_sec: float) -> tuple[dict, dict]:
    if pkg == "coreml-laya":
        eng = CoreMlLayaBackend()
    else:
        eng = CoreMlKev08bBackend()
    try:
        if pkg == "coreml-laya":
            qdef = {"type": "noul", "instructions": "Does the passage state that ocean currents transport heat?"}
            short = eng.encode_item("Ocean currents move warm tropical water toward the poles.", qdef)
            long = eng.encode_item("Ocean currents transfer thermal energy across hemispheres. " * 55, qdef)
            first = lambda: eng.score_noul_batch([short])
            second = lambda: eng.score_noul_batch([long])
        else:
            state = "Ocean currents transfer heat across hemispheres and regulate coastal climates."
            qs = [{"type": "noul", "instructions": f"Question {i}: does the passage mention ocean currents?"}
                  for i in range(8)]
            first = lambda: eng.score_multi_states_noul([(state, qs[:1])])
            second = lambda: eng.score_multi_states_noul([(state, qs)])
        a = run_pair(lambda: [first() for _ in range(4)], lambda: [second() for _ in range(4)],
                     cooldown_sec=cooldown_sec, rounds=2, use_mlx_abba=False)
    finally:
        eng.close()
    telemetry = eng.telemetry()
    short_s, long_s = a["a_median_s"] / 4, a["b_median_s"] / 4
    if pkg == "coreml-laya":
        buckets = telemetry["coreml_info"]["buckets"]
        sizes = [len(item["ids"]) for item in (short, long)]
        matched = [next(b for b in buckets if n <= b) for n in sizes]
        metrics = {"short_tokens": sizes[0], "long_tokens": sizes[1], "buckets": matched,
                   "short_ms": 1000 * short_s, "long_ms": 1000 * long_s,
                   "q_per_sec_short": 1 / short_s, "q_per_sec_long": 1 / long_s,
                   "highlight": f"L{matched[0]} {1000*short_s:.2f} ms, L{matched[1]} {1000*long_s:.2f} ms"}
    else:
        metrics = {"b1_q_per_sec": 1 / short_s, "q_per_sec": 8 / long_s,
                   "ms_per_question": 1000 * long_s / 8,
                   "highlight": f"{8/long_s:.2f} questions/s for eight questions on one state"}
    return {"runtime": pkg, "timing_order": a["order"], **metrics}, telemetry


def run_coreml_benchmark(pkg: str, run_pair: Callable, cooldown_sec: float) -> dict:
    if pkg in ("coreml-laya", "coreml-kev-0.8b"):
        metrics, telemetry = _decision(pkg, run_pair, cooldown_sec)
    else:
        metrics, telemetry = _encoder(pkg, run_pair, cooldown_sec)
    info = telemetry["coreml_info"]
    return {
        "status": "ok",
        "repo_id": info["repo_id"],
        "compile_sec": telemetry["compile_sec"],
        "cold_warm_latency": telemetry["cold_warm_latency"],
        "daemon_four_bucket_metrics": telemetry["four_bucket_metrics"],
        "placement": _placement(info),
        **metrics,
    }
