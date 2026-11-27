"""CLI for `fast-decision-models` (`setup-weights`, `score`, `verify`, `bench`)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from ._common import acquire_hw_lock
from .builder import add_setup_weights_arguments, run_build_args
from .kev_mlx import Kev08bMlxBackend, Kev4bMlxBackend
from .laya_ane import LayaAneBackend
from .laya_hetero import LayaHeteroEngine, LayaMlxGpuBackend
from .paths import ensure_laya_dylib, resolve_kev_4b_dir, resolve_kev_dir, resolve_laya_dir

_SAMPLE_ITEMS = [
    (
        "Incident Report #402: Service auth-api experienced 502 errors for 14 minutes due to TLS certificate expiration on edge-lb-02.",
        {
            "id": "root_cause_match",
            "type": "noul",
            "instructions": "Does the incident report state that an expired TLS certificate caused the outage?",
            "criteria": {
                "false": "The report attributes the outage to another cause or contradicts TLS expiration.",
                "true": "The report explicitly states TLS certificate expiration caused the outage.",
            },
        },
        True,
    ),
    (
        "Quarterly Invoice #8812: Total due is $1,240.00 for cloud object storage (20 TB). Status: PAID via ACH.",
        {
            "id": "unpaid_balance",
            "type": "noul",
            "instructions": "Does the invoice state that payment is currently overdue and unpaid?",
            "criteria": {
                "false": "The invoice shows paid or settled status, not overdue.",
                "true": "The invoice explicitly shows an unpaid or overdue balance.",
            },
        },
        False,
    ),
    (
        "Clinical Triage Note: Patient reports 3 days of dry cough and fever of 38.6C; chest X-ray unremarkable; discharged with rest.",
        {
            "id": "admission_required",
            "type": "noul",
            "instructions": "Was the patient admitted to the intensive care unit?",
            "criteria": {
                "false": "The patient was discharged and not admitted to the ICU.",
                "true": "The patient was admitted to the ICU.",
            },
        },
        False,
    ),
]


def _parse_criteria(raw: str | None, qtype: str) -> Any:
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        return json.loads(stripped)
    if qtype in ("choice", "score"):
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return json.loads(stripped)


def _cmd_score(args: argparse.Namespace) -> int:
    qdef: dict[str, Any] = {
        "id": "cli_query",
        "type": args.type,
        "instructions": args.question,
    }
    crit = _parse_criteria(args.criteria, args.type)
    if crit is not None:
        qdef["criteria"] = crit

    with acquire_hw_lock(f"cli_score_{args.backend}"):
        t0 = time.perf_counter()
        if args.backend in ("laya-ane", "laya-gpu", "laya-hetero"):
            laya_dir = resolve_laya_dir(args.laya_dir, artifacts_dir=args.artifacts_dir)
            if args.backend == "laya-ane":
                laya: Any = LayaAneBackend(laya_dir)
            elif args.backend == "laya-gpu":
                laya = LayaMlxGpuBackend(args.laya_dir, batch_size=1)
            else:
                laya = LayaHeteroEngine(laya_dir=laya_dir, mode="hetero", profile="backfill")
            try:
                enc = laya.encode_item(args.state, qdef)
                if args.type == "noul":
                    p_true = laya.score_noul_batch([enc])[0]
                    probs = [1.0 - p_true, p_true]
                else:
                    probs = laya.score_choice_batch([enc])[0]
                payload: dict[str, Any] = {
                    "backend": args.backend,
                    "type": args.type,
                    "tokens": len(enc["ids"]),
                    "probs": probs,
                }
                if args.type == "noul":
                    payload["p_true"] = probs[1]
            finally:
                laya.close()
        elif args.backend == "kev-4b":
            kev4 = Kev4bMlxBackend(
                resolve_kev_4b_dir(args.kev_dir, artifacts_dir=args.artifacts_dir) if args.kev_dir else None,
                quant_mode=args.kev_quant,
                batch_docs=1,
            )
            try:
                probs, logits = kev4.score_multi_states([(args.state, [qdef])])[0][0]
                payload = {
                    "backend": f"kev-4b-{kev4.quant_mode}",
                    "type": args.type,
                    "probs": probs,
                    "logits": logits,
                }
                if args.type == "noul" and len(probs) >= 2:
                    payload["p_true"] = probs[1]
            finally:
                kev4.close()
        elif args.backend == "genrm":
            from .genrm import GenRMScorer

            with GenRMScorer(model_dir=getattr(args, "qwen_model_dir", None)) as genrm:
                if args.type == "noul":
                    prompt = f"Context:\n{args.state}\n\nQuestion: {args.question}"
                    p_true = float(genrm.score_yes_no_logprob([prompt], verdict_suffix="Answer (Yes or No):")[0])
                    probs = [1.0 - p_true, p_true]
                    payload = {
                        "backend": "genrm",
                        "type": args.type,
                        "probs": probs,
                        "p_true": p_true,
                    }
                else:
                    if isinstance(crit, dict):
                        opt_items = list(crit.items())
                        labels = [chr(ord("A") + i) for i in range(len(opt_items))]
                        opts_txt = "\n".join(f"{lbl}. {k}: {v}" for lbl, (k, v) in zip(labels, opt_items))
                    elif isinstance(crit, list) and crit:
                        labels = [chr(ord("A") + i) for i in range(len(crit))]
                        opts_txt = "\n".join(f"{lbl}. {v}" for lbl, v in zip(labels, crit))
                    else:
                        labels = ["A", "B"]
                        opts_txt = "A. Option A\nB. Option B"
                    prompt = (
                        f"Context:\n{args.state}\n\nQuestion: {args.question}\n\n"
                        f"Options:\n{opts_txt}"
                    )
                    probs = genrm.score_choice_logprob([prompt], labels)[0]
                    payload = {
                        "backend": "genrm",
                        "type": args.type,
                        "options": labels,
                        "probs": probs,
                    }
        else:
            kev = Kev08bMlxBackend(
                resolve_kev_dir(args.kev_dir, artifacts_dir=args.artifacts_dir) if args.kev_dir else None,
                quant_mode=args.kev_quant,
                batch_docs=1,
            )
            try:
                probs, logits = kev.score_multi_states([(args.state, [qdef])])[0][0]
                payload = {
                    "backend": f"kev-0.8b-{kev.quant_mode}",
                    "type": args.type,
                    "probs": probs,
                    "logits": logits,
                }
                if args.type == "noul" and len(probs) >= 2:
                    payload["p_true"] = probs[1]
            finally:
                kev.close()
        payload["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 3)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"backend={payload['backend']} type={args.type} probs={[round(p, 4) for p in payload['probs']]} ({payload['latency_ms']:.2f} ms)")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from fast_shared.bench import check_system_quiescence

    warnings = check_system_quiescence()
    if warnings:
        print("System quiescence warnings:", *warnings, sep="\n  ")

    checked = 0
    if args.backend in ("all", "laya-ane"):
        laya_dir = resolve_laya_dir(args.laya_dir, artifacts_dir=args.artifacts_dir)
        has_local = (laya_dir / "laya_typed_s512_w8.mlmodelc").exists() or (laya_dir / "laya_typed_w8.mlmodelc").is_dir()
        if not has_local and args.backend == "all":
            print(f"[SKIP] Laya-Typed local ANE bundles not found in {laya_dir}")
        else:
            with acquire_hw_lock("cli_verify_laya"):
                dylib_p = ensure_laya_dylib()
                laya = LayaAneBackend(laya_dir, dylib_p, pipe_depth=4)
                try:
                    plans = laya.inspect_compute_plans()
                    for bucket, (ane, gpu, cpu) in plans.items():
                        assert gpu == 0 and cpu == 0 and ane > 0, f"Non-pure ANE plan {bucket}: {(ane, gpu, cpu)}"
                    encs = [laya.encode_item(st, qd) for st, qd, _ in _SAMPLE_ITEMS]
                    ps = laya.score_noul_batch(encs)
                    for i, ((_, _, exp), p) in enumerate(zip(_SAMPLE_ITEMS, ps)):
                        pred = p >= 0.5
                        assert pred == exp, f"Laya sample[{i}] expected {exp}, got P(true)={p:.4f}"
                    print(f"[PASS] laya-ane plans={plans} probs={[round(p, 4) for p in ps]}")
                    checked += 1
                finally:
                    laya.close()

    if args.backend in ("all", "kev-0.8b"):
        kev_dir = resolve_kev_dir(args.kev_dir, artifacts_dir=args.artifacts_dir)
        has_local_kev = (kev_dir / "kev_meta.json").exists() or (kev_dir / "kev_meta_w8.json").exists()
        if not has_local_kev and args.backend == "all":
            print(f"[SKIP] Kev-0.8B local artifacts not found in {kev_dir}")
        else:
            with acquire_hw_lock("cli_verify_kev"):
                kev = Kev08bMlxBackend(kev_dir if has_local_kev else None, quant_mode=args.kev_quant, batch_docs=4)
                try:
                    pairs = [(st, [qd]) for st, qd, _ in _SAMPLE_ITEMS]
                    single = [kev.score_multi_states_noul([pair], batch_docs=1)[0][0][0] for pair in pairs]
                    batched = [res[0][0] for res in kev.score_multi_states_noul(pairs, batch_docs=len(pairs))]
                    diffs = [abs(a - b) for a, b in zip(single, batched)]
                    assert max(diffs) < 1e-2, f"Kev B=1 vs B={len(pairs)} drift too large: {diffs}"
                    for i, ((_, _, exp), p) in enumerate(zip(_SAMPLE_ITEMS, batched)):
                        assert (p >= 0.5) == exp, f"Kev sample[{i}] expected {exp}, got P(true)={p:.4f}"
                    print(f"[PASS] kev-0.8b ({kev.quant_mode}) B=1 vs B={len(pairs)} max_diff={max(diffs):.6f} probs={[round(p, 4) for p in batched]}")
                    checked += 1
                finally:
                    kev.close()

    if checked == 0:
        print("No decision-model artifacts found to verify. Run 'decision-models setup-weights'.", file=sys.stderr)
        return 1
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    items = [(_SAMPLE_ITEMS[i % len(_SAMPLE_ITEMS)][0], _SAMPLE_ITEMS[i % len(_SAMPLE_ITEMS)][1]) for i in range(args.n)]
    results: dict[str, Any] = {}

    if args.backend in ("all", "laya-ane"):
        laya_dir = resolve_laya_dir(args.laya_dir, artifacts_dir=args.artifacts_dir)
        if (laya_dir / "laya_typed_s512_w8.mlmodelc").exists() or (laya_dir / "laya_typed_w8.mlmodelc").is_dir():
            with acquire_hw_lock("cli_bench_laya"):
                laya = LayaAneBackend(laya_dir, ensure_laya_dylib(), pipe_depth=4)
                try:
                    encs = [laya.encode_item(st, qd) for st, qd in items]
                    _ = laya.score_noul_batch(encs[: min(4, len(encs))])
                    t0 = time.perf_counter()
                    _ = laya.score_noul_batch(encs)
                    dt = time.perf_counter() - t0
                    results["laya_ane"] = {
                        "n_questions": len(encs),
                        "elapsed_sec": round(dt, 4),
                        "ms_per_question": round((dt * 1e3) / len(encs), 2),
                        "q_per_sec": round(len(encs) / max(dt, 1e-9), 2),
                    }
                finally:
                    laya.close()

    if args.backend in ("all", "laya-hetero"):
        laya_dir = resolve_laya_dir(args.laya_dir, artifacts_dir=args.artifacts_dir)
        if args.backend == "laya-hetero" or (laya_dir / "laya_typed_s512_w8.mlmodelc").exists():
            with acquire_hw_lock("cli_bench_laya_hetero"):
                het = LayaHeteroEngine(laya_dir=laya_dir, mode="hetero", profile="backfill")
                try:
                    encs = [het.encode_item(st, qd) for st, qd in items]
                    _ = het.score_noul_batch(encs[: min(4, len(encs))])
                    t0 = time.perf_counter()
                    _ = het.score_noul_batch(encs)
                    dt = time.perf_counter() - t0
                    results["laya_hetero"] = {
                        "n_questions": len(encs),
                        "elapsed_sec": round(dt, 4),
                        "ms_per_question": round((dt * 1e3) / len(encs), 2),
                        "q_per_sec": round(len(encs) / max(dt, 1e-9), 2),
                    }
                finally:
                    het.close()

    if args.backend in ("all", "kev-0.8b"):
        kev_dir = resolve_kev_dir(args.kev_dir, artifacts_dir=args.artifacts_dir)
        if (kev_dir / "kev_meta.json").exists() or (kev_dir / "kev_meta_w8.json").exists() or args.backend == "kev-0.8b":
            with acquire_hw_lock("cli_bench_kev"):
                has_local = (kev_dir / "kev_meta.json").exists() or (kev_dir / "kev_meta_w8.json").exists()
                kev = Kev08bMlxBackend(kev_dir if has_local else None, quant_mode=args.kev_quant, batch_docs=args.batch_docs)
                try:
                    pairs = [(st, [qd]) for st, qd in items]
                    _ = kev.score_multi_states_noul(pairs[: min(2, len(pairs))])
                    t0 = time.perf_counter()
                    _ = kev.score_multi_states_noul(pairs)
                    dt = time.perf_counter() - t0
                    results["kev_08b"] = {
                        "quant_mode": kev.quant_mode,
                        "batch_docs": args.batch_docs,
                        "n_questions": len(pairs),
                        "elapsed_sec": round(dt, 4),
                        "ms_per_question": round((dt * 1e3) / len(pairs), 2),
                        "q_per_sec": round(len(pairs) / max(dt, 1e-9), 2),
                    }
                finally:
                    kev.close()

    print(json.dumps(results, indent=2))
    return 0 if results else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decision-models", description="fast-decision-models ANE + MLX CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Compile native dylib and/or build CoreML ANE / MLX W8 weights")
    add_setup_weights_arguments(sw)
    sw.set_defaults(func=run_build_args)

    sc = sub.add_parser("score", help="Score one typed question (noul, choice, score) against a state")
    sc.add_argument("--backend", choices=("laya-ane", "laya-gpu", "laya-hetero", "kev-0.8b", "kev-4b", "genrm"), default="kev-0.8b")
    sc.add_argument("--type", choices=("noul", "choice", "score"), default="noul")
    sc.add_argument("--question", required=True)
    sc.add_argument("--criteria", default=None, help="JSON object/array or comma-separated options")
    sc.add_argument("--state", required=True)
    sc.add_argument("--artifacts-dir", default=None)
    sc.add_argument("--laya-dir", default=None)
    sc.add_argument("--kev-dir", default=None)
    sc.add_argument("--qwen-model-dir", default=None, help="Optional local Qwen3.5 model dir for --backend genrm")
    sc.add_argument("--kev-quant", choices=("auto", "w8", "w4", "bf16"), default="auto")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=_cmd_score)

    vf = sub.add_parser("verify", help="Verify ANE ComputePlan, B=1 vs batched parity, and scoring sanity")
    vf.add_argument("--backend", choices=("all", "laya-ane", "kev-0.8b"), default="all")
    vf.add_argument("--artifacts-dir", default=None)
    vf.add_argument("--laya-dir", default=None)
    vf.add_argument("--kev-dir", default=None)
    vf.add_argument("--kev-quant", choices=("auto", "w8", "w4", "bf16"), default="auto")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark ANE, Hetero, and/or MLX throughput on synthetic typed queries")
    bn.add_argument("--backend", choices=("all", "laya-ane", "laya-hetero", "kev-0.8b"), default="all")
    bn.add_argument("--n", type=int, default=24)
    bn.add_argument("--batch-docs", type=int, default=8)
    bn.add_argument("--artifacts-dir", default=None)
    bn.add_argument("--laya-dir", default=None)
    bn.add_argument("--kev-dir", default=None)
    bn.add_argument("--kev-quant", choices=("auto", "w8", "w4", "bf16"), default="auto")
    bn.set_defaults(func=_cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    from fast_shared.hw_check import warn_if_untested_hardware

    parser = build_parser()
    args = parser.parse_args(argv)
    warn_if_untested_hardware()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
