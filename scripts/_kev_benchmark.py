"""Matched input and score capture for Kev decision throughput and prefix-cache probes."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys
import time

# Both the project environment and the standalone public MLX environment run these probes.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("fast-shared", "fast-decision-models"):
    sys.path.insert(0, str(_ROOT / "packages" / _pkg / "src"))


def _token_layout(scorer, pairs):
    layouts = []
    for state, questions in pairs:
        state_ids, branches = scorer.encode_questions(state, questions)
        payload = json.dumps([state_ids, branches], separators=(",", ":")).encode()
        layouts.append({
            "state_tokens": len(state_ids),
            "branch_tokens": [len(row["ids"]) for row in branches],
            "input_sha256": hashlib.sha256(payload).hexdigest(),
        })
    return layouts


def _timed_scores(scorer, pairs, batch_docs):
    started = time.perf_counter()
    rows = scorer.score_multi_states(pairs, batch_docs=batch_docs)
    elapsed = time.perf_counter() - started
    scores = []
    for doc, (_, questions) in zip(rows, pairs, strict=True):
        for (probabilities, logits), question in zip(doc, questions, strict=True):
            options = 2 if question["type"] == "noul" else len(question["criteria"])
            if (
                len(probabilities) != options
                or len(logits) != options
                or not all(math.isfinite(x) for x in probabilities + logits)
                or not all(0 <= value <= 1 for value in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)
            ):
                raise ValueError("Kev returned invalid option probabilities or logits")
            scores.append({"probabilities": probabilities, "logits": logits})
    return {"seconds": elapsed, "scores": scores}


def score_differences(left, right):
    """Compare every recorded option in order, including the uncalibrated logits."""
    changes = {"max_abs_probability": 0.0, "max_abs_logit": 0.0, "top_option_changes": 0}
    for first, second in zip(left, right, strict=True):
        for field, metric in (("probabilities", "max_abs_probability"), ("logits", "max_abs_logit")):
            gap = max(abs(a - b) for a, b in zip(first[field], second[field], strict=True))
            changes[metric] = max(changes[metric], gap)
        first_top = max(range(len(first["probabilities"])), key=first["probabilities"].__getitem__)
        second_top = max(range(len(second["probabilities"])), key=second["probabilities"].__getitem__)
        changes["top_option_changes"] += int(first_top != second_top)
    return changes


def stock_short_probe(kev_dir, run_abba_pair, cooldown_sec):
    import mlx.core as mx
    from _outcome_public_backends import PublicKev08bMlxBackend
    from fast_decision_models.cli import _SAMPLE_ITEMS
    from fast_decision_models.paths import resolve_kev_dir
    from fast_shared.bench import get_unified_memory_footprint

    started = time.perf_counter()
    if not (kev_dir / "tokenizer.json").is_file():
        kev_dir = resolve_kev_dir()
    scorer = PublicKev08bMlxBackend(kev_dir, batch_docs=8)
    load_sec = time.perf_counter() - started
    # These are the same 64 passage/question pairs as benchmark_monorepo's project CLI arm.
    pairs = [(st, [q]) for st, q, _ in _SAMPLE_ITEMS]
    repeated = (pairs * 64)[:64]
    try:
        layouts = _token_layout(scorer, pairs)
        mx.reset_peak_memory()
        abba = run_abba_pair(
            lambda: _timed_scores(scorer, repeated[:1], 1),
            lambda: _timed_scores(scorer, repeated, 8),
            rounds=2, cooldown_sec=cooldown_sec,
        )
        t1 = abba["a_median_s"]
        t8 = abba["b_median_s"]
        peak = get_unified_memory_footprint()
        return {
            "status": "ok", "runtime": "stock-mlx_lm-kev-8bit", "timing_order": abba["order"],
            "b1_drift": abba["a_drift"], "b8_drift": abba["b_drift"],
            "b1_q_per_sec": 1 / t1, "q_per_sec": len(repeated) / t8,
            "ms_per_question": t8 * 1000 / len(repeated),
            "peak_active_gb": mx.get_peak_memory() / 1e9,
            "peak_footprint_gb": peak["peak_footprint_gb"], "rss_gb": peak["rss_bytes"] / 1e9,
            "load_s": load_sec, "b1_median_s": t1, "b8_median_s": t8,
            "fixture_ids": [q["id"] for _, q, _ in _SAMPLE_ITEMS], "fixture_tokens": layouts,
            "n_questions": len(repeated), "b8_scores": abba["b_runs"][-1]["payload"]["scores"],
            "highlight": f"stock MLX, 64 scored questions in batches of 8: {len(repeated)/t8:.2f} q/s",
        }
    finally:
        scorer.close()


def cache_probe_inputs():
    from fast_shared.outcome_eval.decisions import build_ragtruth_noul_qdef, build_ragtruth_state
    from fast_shared.outcome_eval.hub_loader import load_ragtruth

    # Eight actual candidate responses from two repeated contexts in the locked test split.
    ids = ["10350", "10351", "10352", "10353", "10354", "15685", "15686", "15687"]
    items = {row["id"]: row for row in load_ragtruth(local_files_only=True)}
    question = build_ragtruth_noul_qdef()
    return [{"id": key, "state": build_ragtruth_state(items[key]), "questions": [question]} for key in ids]


def cache_probe_arm(arm, records, kev_dir):
    import mlx.core as mx
    from fast_shared.bench import FourBucketMemoryProbe, get_unified_memory_footprint
    from _outcome_public_backends import PublicKev08bMlxBackend, stock_decision_model_files
    from fast_decision_models.kev_mlx import Kev08bMlxBackend

    stock = arm == "stock"
    model_files = stock_decision_model_files("public-kev-0.8b") if stock else [Path(kev_dir)]
    probe = FourBucketMemoryProbe(model_directories=model_files)
    started = time.perf_counter()
    if stock:
        scorer = PublicKev08bMlxBackend(Path(kev_dir), batch_docs=8)
    else:
        scorer = Kev08bMlxBackend(kev_dir, quant_mode="w8", batch_docs=8,
                                 prefix_cache_mb=0 if arm == "plain" else "auto")
    load_seconds = time.perf_counter() - started
    tree = getattr(scorer, "prefix_tree", None)
    if (tree is not None) != (arm == "cached"):
        raise RuntimeError(f"unexpected cache configuration for {arm}")
    pairs = [(r["state"], r["questions"]) for r in records]
    try:
        layouts = _token_layout(scorer, pairs)
        # Separate model warmup from a cold prefix tree, then time an immediate repeat.
        scorer.score_multi_states([("warmup", pairs[0][1])], batch_docs=1)
        mx.reset_peak_memory()
        workloads = {}
        for n in (1, len(pairs)):
            if tree is not None:
                tree.clear()
            cold = _timed_scores(scorer, pairs[:n], n)
            warm = _timed_scores(scorer, pairs[:n], n)
            workloads[f"b{n}"] = {"cold": cold, "warm": warm}
            if tree is not None:
                workloads[f"b{n}"]["tree_after_warm"] = tree.stats()
        snapshot = probe.snapshot()
        return {
            "arm": arm, "prefix_cache_enabled": tree is not None, "load_seconds": load_seconds,
            "token_layouts": layouts, "workloads": workloads,
            "memory_gb": {
                "mlx_peak": mx.get_peak_memory()/1e9,
                "process_peak_footprint": get_unified_memory_footprint()["peak_footprint_gb"],
                "process_footprint": snapshot["phys_footprint"]["phys_footprint_gb"],
                "model_mapped": snapshot["mapped_files"]["model_bytes"]/1e9,
                "neural_nofootprint": snapshot["task_vm_info"]["ledger_tag_neural_nofootprint_bytes"]/1e9,
                "combined_with_model_mapped": snapshot["combined_with_model_mapped_gb"],
            },
        }
    finally:
        scorer.close()


if __name__ == "__main__":
    from fast_shared.bench import wired_limit

    request = json.load(sys.stdin)
    try:
        with wired_limit() if request["use_wired_limit"] else nullcontext():
            result = cache_probe_arm(request["arm"], request["records"], request["kev_dir"])
    except (FileNotFoundError, ImportError) as exc:
        result = {"status": "skipped", "reason": str(exc)}
    print(json.dumps(result, allow_nan=False))
