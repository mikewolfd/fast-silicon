"""Hardware-backed ANE (`Laya-Typed`) & MLX (`Kev-0.8B`) verification tests."""

from __future__ import annotations

import math

import pytest

from fast_decision_models._common import acquire_hw_lock
from fast_decision_models.kev_mlx import Kev08bMlxBackend
from fast_decision_models.laya_ane import LayaAneBackend
from fast_decision_models.paths import ensure_laya_dylib, resolve_kev_dir, resolve_laya_dir

_SAMPLE_CASES = [
    (
        "Incident Report #402: auth-api experienced 502 errors for 14 minutes due to TLS certificate expiration on edge-lb-02.",
        {
            "id": "tls_outage",
            "type": "noul",
            "instructions": "Does the report state that TLS certificate expiration caused the outage?",
            "criteria": {
                "false": "The outage was not caused by TLS certificate expiration.",
                "true": "TLS certificate expiration caused the outage.",
            },
        },
        True,
    ),
    (
        "Invoice #8812: Total $1,240.00 for cloud storage (20 TB). Payment status: PAID via ACH on Nov 12.",
        {
            "id": "invoice_overdue",
            "type": "noul",
            "instructions": "Does the invoice state that the balance is overdue and unpaid?",
            "criteria": {
                "false": "The invoice shows paid status and is not overdue.",
                "true": "The invoice shows an overdue unpaid balance.",
            },
        },
        False,
    ),
    (
        "Deployment Log: Release v2.8.1 rolled back after canary health checks exceeded 5% 5xx threshold.",
        {
            "id": "canary_rollback",
            "type": "noul",
            "instructions": "Was release v2.8.1 rolled back due to failing canary health checks?",
            "criteria": {
                "false": "The release promoted cleanly without rollback.",
                "true": "The release was rolled back due to failing canary health checks.",
            },
        },
        True,
    ),
    (
        "Clinical Triage Note: Dry cough and fever 38.6C for 3 days; chest X-ray clear; discharged home with hydration instructions.",
        {
            "id": "icu_admitted",
            "type": "noul",
            "instructions": "Was the patient admitted to the intensive care unit?",
            "criteria": {
                "false": "The patient was discharged home and not admitted to the ICU.",
                "true": "The patient was admitted to the ICU.",
            },
        },
        False,
    ),
]


def test_kev08b_left_pad_batch_parity_and_noul_accuracy() -> None:
    kev_dir = resolve_kev_dir()
    if not (kev_dir / "model_w8.safetensors").exists() and not (kev_dir / "model.safetensors").exists():
        pytest.skip(f"Missing Kev-0.8B artifacts in {kev_dir}")

    pairs = [(st, [qd]) for st, qd, _ in _SAMPLE_CASES]
    with acquire_hw_lock("test_kev08b_batch_parity"):
        kev = Kev08bMlxBackend(kev_dir, quant_mode="auto", batch_docs=4)
        try:
            single_ps = [kev.score_multi_states_noul([p], batch_docs=1)[0][0][0] for p in pairs]
            batch_ps = [res[0][0] for res in kev.score_multi_states_noul(pairs, batch_docs=len(pairs))]
        finally:
            kev.close()

    for idx, (p1, pb, (_, _, expected)) in enumerate(zip(single_ps, batch_ps, _SAMPLE_CASES)):
        assert math.isfinite(p1) and math.isfinite(pb)
        assert abs(p1 - pb) < 1e-2, f"Sample[{idx}] B=1 vs B={len(pairs)} drift: {p1:.5f} vs {pb:.5f}"
        assert (pb >= 0.5) == expected, f"Sample[{idx}] expected {expected}, got P(true)={pb:.4f}"


def test_kev08b_choice_and_score_distributions() -> None:
    kev_dir = resolve_kev_dir()
    if not (kev_dir / "model_w8.safetensors").exists() and not (kev_dir / "model.safetensors").exists():
        pytest.skip(f"Missing Kev-0.8B artifacts in {kev_dir}")

    state = "System Alert: Primary database disk utilization reached 99%, writes paused until volume expanded."
    questions = [
        {
            "id": "subsystem",
            "type": "choice",
            "instructions": "Which subsystem caused the alert?",
            "criteria": ["network", "storage", "authentication"],
        },
        {
            "id": "severity",
            "type": "score",
            "instructions": "Rate operational severity of writes pausing.",
            "criteria": ["no impact", "minor warning", "degraded", "critical outage"],
        },
    ]
    with acquire_hw_lock("test_kev08b_choice_score"):
        kev = Kev08bMlxBackend(kev_dir, quant_mode="auto", batch_docs=1)
        try:
            res = kev.score_multi_states([(state, questions)], batch_docs=1)[0]
        finally:
            kev.close()

    probs_choice, logits_choice = res[0]
    probs_score, logits_score = res[1]
    assert len(probs_choice) == 3 and len(logits_choice) == 3
    assert len(probs_score) == 4 and len(logits_score) == 4
    assert abs(sum(probs_choice) - 1.0) < 1e-5
    assert abs(sum(probs_score) - 1.0) < 1e-5
    assert probs_choice[1] == max(probs_choice), f"Expected 'storage', got {probs_choice}"


def test_laya_ane_compute_plan_and_batch_parity() -> None:
    laya_dir = resolve_laya_dir()
    if not (laya_dir / "laya_typed_s512_w8.mlmodelc").exists():
        pytest.skip(f"Missing Laya ANE bundles in {laya_dir}")

    with acquire_hw_lock("test_laya_ane_hw"):
        laya = LayaAneBackend(laya_dir, ensure_laya_dylib(), pipe_depth=4)
        try:
            plans = laya.inspect_compute_plans()
            assert "s512" in plans
            for bucket, (ane_ops, gpu_ops, cpu_ops) in plans.items():
                assert gpu_ops == 0 and cpu_ops == 0 and ane_ops > 1000, f"Unexpected {bucket} plan: {(ane_ops, gpu_ops, cpu_ops)}"

            encs = [laya.encode_item(st, qd) for st, qd, _ in _SAMPLE_CASES]
            sync_ps = [laya.score_noul_batch([e])[0] for e in encs]
            pipe_ps = laya.score_noul_batch(encs)
            for idx, (ps, pp) in enumerate(zip(sync_ps, pipe_ps)):
                assert math.isfinite(ps) and math.isfinite(pp)
                assert abs(ps - pp) < 2e-2, f"Sample[{idx}] N=1 vs pipe_depth=4 drift: {ps:.5f} vs {pp:.5f}"

            # Exercise S=512 bucket alongside S=256 and Choice(3)
            long_state = " ".join(["Telemetry healthy across cluster nodes."] * 60)
            c_enc = laya.encode_item(
                long_state,
                {
                    "id": "domain",
                    "type": "choice",
                    "instructions": "Select the matching domain.",
                    "criteria": ["billing", "infrastructure", "clinical"],
                },
            )
            assert len(c_enc["ids"]) > 256
            c_probs = laya.score_choice_batch([c_enc])[0]
            assert len(c_probs) == 3 and abs(sum(c_probs) - 1.0) < 1e-5
        finally:
            laya.close()
