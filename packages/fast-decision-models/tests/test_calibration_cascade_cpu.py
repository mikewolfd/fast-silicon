"""Comprehensive CPU unit tests for OOS threshold calibration and 3-tier DecisionCascadeScorer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fast_decision_models.calibration import (
    CalibrationProfile,
    DecisionCascadeScorer,
    align_oos_probabilities,
    apply_multiclass_temperature,
    apply_oos_decision,
    fit_calibrated_profile,
    fit_multiclass_temperature,
    fit_oof_calibrated_scores,
    fit_oos_threshold,
    stratified_kfold_indices,
)
from fast_shared.outcome_eval.decisions import (
    CLINC150_DOMAINS,
    CLINC_STAGE1_DOMAINS,
    compute_auroc,
    compute_binary_macro_f1,
    compute_ece,
    compute_multiclass_nll,
    evaluate_clinc_two_stage,
)


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_calibration_module_stdlib_only_import() -> None:
    """Importing `fast_decision_models.calibration` must not eagerly import `numpy` or `mlx`."""
    src_dir = _pkg_root() / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src_dir)!r}); "
        "import fast_decision_models.calibration, fast_decision_models.scorers; "
        "banned = {'numpy', 'mlx', 'torch', 'coremltools'}; "
        "leaked = sorted(banned.intersection(sys.modules)); "
        "assert not leaked, f'Unexpected imports: {leaked}'; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_multiclass_temperature_extreme_laya_and_underconfident_regimes() -> None:
    """Verify temperature fitting covers both Laya-like T~0.10 overconfidence and T<0.5 underconfidence."""
    rng = np.random.default_rng(101)
    n, k = 300, 11
    y = rng.integers(0, k, size=n)
    logits = rng.normal(0.0, 0.8, size=(n, k))
    logits[np.arange(n), y] += 1.6

    # Case A: Simulate Laya choice:11+ T=0.10058 over-sharpening (requires T_post > 3.5)
    t_laya = 0.10058
    scaled_a = logits / t_laya
    scaled_a -= np.max(scaled_a, axis=1, keepdims=True)
    p_sharp = np.exp(scaled_a)
    p_sharp /= np.sum(p_sharp, axis=1, keepdims=True)

    t_fit_a = fit_multiclass_temperature(p_sharp, y, from_probs=True)
    assert t_fit_a > 3.5
    p_cal_a = apply_multiclass_temperature(p_sharp, t_fit_a, from_probs=True)
    assert np.array_equal(np.argmax(p_cal_a, axis=1), np.argmax(p_sharp, axis=1))
    assert compute_multiclass_nll(y.tolist(), p_cal_a.tolist()) < compute_multiclass_nll(y.tolist(), p_sharp.tolist()) - 1.0

    # Case B: Underconfident logits scaled by T=3.0 (requires T_fit < 0.5)
    scaled_b = logits / 3.0
    scaled_b -= np.max(scaled_b, axis=1, keepdims=True)
    p_flat = np.exp(scaled_b)
    p_flat /= np.sum(p_flat, axis=1, keepdims=True)

    t_fit_b = fit_multiclass_temperature(p_flat, y, from_probs=True)
    assert 0.25 <= t_fit_b < 0.5
    p_cal_b = apply_multiclass_temperature(p_flat, t_fit_b, from_probs=True)
    assert np.array_equal(np.argmax(p_cal_b, axis=1), np.argmax(p_flat, axis=1))
    assert compute_multiclass_nll(y.tolist(), p_cal_b.tolist()) < compute_multiclass_nll(y.tolist(), p_flat.tolist())


def test_fit_oos_threshold_and_align_invariants() -> None:
    """Verify OOS threshold search, boundary tie safety, AUROC preservation, and argmax consistency."""
    rng = np.random.default_rng(77)
    n, k, oos_idx = 220, 11, 10
    y_oos = (rng.random(n) < 0.18).astype(np.int64)

    logits = rng.normal(0.0, 0.7, size=(n, k))
    # Give OOS items a higher OOS logit that still stays < 0.5 under 11-way softmax on many items
    logits[y_oos == 1, oos_idx] += 1.65
    logits[y_oos == 0, 0] += 1.5
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)

    tau_f1 = fit_oos_threshold(probs, y_oos, oos_index=oos_idx, metric="f1")
    tau_mf1 = fit_oos_threshold(probs[:, oos_idx], y_oos, metric="macro_f1")
    tau_youden = fit_oos_threshold(probs, y_oos, oos_index=oos_idx, metric="youden")
    assert 0.0 < tau_f1 < 1.0
    assert 0.0 < tau_mf1 < 1.0
    assert 0.0 < tau_youden < 1.0

    # Degenerate single-class split returns fallback cleanly
    assert fit_oos_threshold(probs[:, oos_idx], np.zeros(n, dtype=np.int64), fallback_threshold=0.42) == 0.42

    # Verify align_oos_probabilities invariants
    aligned = align_oos_probabilities(probs, oos_index=oos_idx, oos_threshold=tau_f1)
    assert aligned.shape == probs.shape
    np.testing.assert_allclose(np.sum(aligned, axis=1), 1.0, atol=1e-9)
    assert np.all(aligned >= 0.0)

    # Exact AUROC invariance
    auroc_before = compute_auroc(y_oos.tolist(), probs[:, oos_idx].tolist())
    auroc_after = compute_auroc(y_oos.tolist(), aligned[:, oos_idx].tolist())
    assert abs(auroc_after - auroc_before) < 1e-12

    # Exact consistency between >= 0.5, argmax == oos_idx, and raw >= tau_f1
    is_oos, pred_idx = apply_oos_decision(probs, oos_index=oos_idx, oos_threshold=tau_f1)
    assert np.array_equal(aligned[:, oos_idx] >= 0.5, is_oos)
    assert np.array_equal(np.argmax(aligned, axis=1) == oos_idx, is_oos)
    assert np.array_equal(np.argmax(aligned, axis=1), pred_idx)

    # Relative order among in-scope classes is preserved
    in_cols = [c for c in range(k) if c != oos_idx]
    assert np.array_equal(np.argmax(probs[:, in_cols], axis=1), np.argmax(aligned[:, in_cols], axis=1))

    # Boundary test: p_oos == tau_f1 with 100% of remaining mass in class 0 must STILL select oos_idx!
    boundary_vec = np.zeros(k, dtype=np.float64)
    boundary_vec[oos_idx] = tau_f1
    boundary_vec[0] = 1.0 - tau_f1
    aligned_b = align_oos_probabilities(boundary_vec, oos_index=oos_idx, oos_threshold=tau_f1)
    assert aligned_b[oos_idx] > 0.5
    assert int(np.argmax(aligned_b)) == oos_idx


def test_calibration_profile_oos_roundtrip_and_leakage_free_oof() -> None:
    """Verify CalibrationProfile serialization with OOS fields and leakage-free 5-fold OOF calibration."""
    rng = np.random.default_rng(202611)
    n, k = 400, len(CLINC_STAGE1_DOMAINS)
    oos_idx = CLINC_STAGE1_DOMAINS.index("out_of_scope")

    domains = [CLINC_STAGE1_DOMAINS[i % k] for i in range(n)]
    y_dom = np.array([CLINC_STAGE1_DOMAINS.index(d) for d in domains], dtype=np.int64)
    y_oos = (y_dom == oos_idx).astype(np.int64)

    # Simulate over-sharpened Laya Stage-1 probabilities where OOS has strong AUROC but low raw prob < 0.5
    logits = rng.normal(0.0, 0.7, size=(n, k))
    logits[np.arange(n), y_dom] += 1.4
    logits[:, oos_idx] -= 0.75  # OOS logit bias so raw p_oos < 0.5 on most OOS items
    scaled = logits / 0.12
    scaled -= np.max(scaled, axis=1, keepdims=True)
    raw_probs = np.exp(scaled)
    raw_probs /= np.sum(raw_probs, axis=1, keepdims=True)

    # Profile serialization round-trip
    prof = fit_calibrated_profile(
        "laya_hetero",
        "clinc150_stage1",
        method="temperature",
        values=raw_probs[:200],
        labels=y_dom[:200],
        oos_labels=y_oos[:200],
        oos_index=oos_idx,
    )
    assert prof.oos_index == oos_idx
    assert prof.oos_threshold is not None and 0.0 < prof.oos_threshold < 1.0
    assert CalibrationProfile.from_dict(prof.to_dict()) == prof

    # Verify stratified K-fold has zero overlap and covers every index once
    folds = stratified_kfold_indices(y_dom, n_splits=5, seed=42)
    seen_val: list[int] = []
    for tr, va in folds:
        assert len(set(tr.tolist()).intersection(set(va.tolist()))) == 0
        seen_val.extend(va.tolist())
    assert sorted(seen_val) == list(range(n))

    # Leakage-free 5-fold OOF calibration + OOS alignment
    oof = fit_oof_calibrated_scores(
        raw_probs,
        y_dom,
        model_id="laya_hetero",
        task_key="clinc150_stage1",
        method="temperature",
        n_splits=5,
        seed=42,
        oos_labels=y_oos,
        oos_index=oos_idx,
        align_oos=True,
    )
    nll_raw = compute_multiclass_nll(y_dom.tolist(), raw_probs.tolist())
    nll_cal = compute_multiclass_nll(y_dom.tolist(), oof["calibrated"].tolist())
    assert nll_cal < nll_raw - 0.8

    f1_raw = compute_binary_macro_f1(y_oos.tolist(), raw_probs[:, oos_idx].tolist(), threshold=0.5)["f1_pos"]
    f1_oof = compute_binary_macro_f1(y_oos.tolist(), oof["aligned"][:, oos_idx].tolist(), threshold=0.5)["f1_pos"]
    assert f1_oof > f1_raw + 0.10

    # Also verify binary Platt OOF calibration cuts ECE cleanly
    raw_bin = np.clip(1.0 / (1.0 + np.exp(-(rng.normal(0.0, 1.0, size=200) * 3.5 + 0.8))), 1e-6, 1.0 - 1e-6)
    y_bin = (rng.random(200) < (1.0 / (1.0 + np.exp(-((np.log(raw_bin / (1.0 - raw_bin)) - 0.8) / 3.5))))).astype(np.int64)
    oof_bin = fit_oof_calibrated_scores(
        raw_bin,
        y_bin,
        model_id="laya_hetero",
        task_key="ragtruth",
        method="platt",
        n_splits=5,
        seed=7,
    )
    assert compute_ece(y_bin.tolist(), oof_bin["calibrated"].tolist()) < compute_ece(y_bin.tolist(), raw_bin.tolist())


def test_decision_cascade_3tier_parity_and_clinc_integration_cpu() -> None:
    """Verify LayaHetero -> Kev-0.8B -> Kev-4B cascade calibration parity and evaluate_clinc_two_stage."""
    # Item 0: Primary confident in-scope ("banking", intent idx 1) -> stays on primary
    # Item 1: Primary confident OOS (calibrated p_oos >= 0.28 though < 0.5!) -> stays on primary, routed to OOS
    # Item 2: Primary uncertain -> escalates to Kev-0.8B, which is confident ("travel", intent idx 0) -> stops at Tier 2
    # Item 3: Primary uncertain AND Kev-0.8B uncertain -> escalates to Kev-4B ("work", intent idx 2) -> Tier 3
    # Item 4: Primary uncertain -> Kev-0.8B uncertain -> Kev-4B predicts OOS via Kev-4B's calibrated OOS threshold
    oos_idx = CLINC_STAGE1_DOMAINS.index("out_of_scope")
    k = len(CLINC_STAGE1_DOMAINS)

    def _onehot_ish(top_i: int, top_p: float, oos_p: float = 0.02) -> list[float]:
        vec = [0.0] * k
        vec[top_i] = top_p
        if top_i != oos_idx:
            vec[oos_idx] = oos_p
            rem = max(0.0, 1.0 - top_p - oos_p)
            other_idxs = [j for j in range(k) if j not in (top_i, oos_idx)]
        else:
            rem = max(0.0, 1.0 - top_p)
            other_idxs = [j for j in range(k) if j != oos_idx]
        for j in other_idxs:
            vec[j] = rem / float(len(other_idxs))
        return vec

    # Primary (Laya) raw scores (under T=0.10058 overconfidence, softened by primary profile T=6.0!)
    # Note: after T=6.0, a vector with [0.999, ...] drops below tau=0.60, whereas [0.999999, ...] stays >= 0.60.
    s1_qdef = {"id": "clinc_s1", "type": "choice", "criteria": {d: d for d in CLINC_STAGE1_DOMAINS}}
    s2_qdefs = {
        dom: {"id": f"clinc_s2_{dom}", "type": "choice", "criteria": list(CLINC150_DOMAINS[dom])}
        for dom in CLINC150_DOMAINS
    }

    prim_s1_map = {
        "item0": _onehot_ish(0, 0.999999, oos_p=1e-7),     # banking: stays confident (>0.60) even after T=6.0
        "item1": [0.35 if j == 0 else (0.35 if j == oos_idx else 0.30 / (k - 2)) for j in range(k)],  # Wait: make primary confident OOS at T=1.0
        "item2": [0.30 if j == 5 else (0.10 if j == oos_idx else 0.60 / (k - 2)) for j in range(k)],
        "item3": [0.22 if j == 7 else (0.10 if j == oos_idx else 0.68 / (k - 2)) for j in range(k)],
        "item4": [0.20 if j == 3 else (0.18 if j == oos_idx else 0.62 / (k - 2)) for j in range(k)],
    }
    # Make item1 clearly >= 0.60 max prob after T=1.5 while p_oos=0.35 < 0.5 yet >= oos_threshold=0.28!
    # Wait: if item1 has top1=0.62 on class 0 and p_oos=0.32 -> after T=1.0, max=0.62 >= 0.60 (no esc!) and p_oos=0.32 >= 0.28 (OOS!).
    prim_s1_map["item1"] = _onehot_ish(0, 0.62, oos_p=0.32)

    close_events: list[str] = []

    class _FakePrimary:
        def encode_item(self, st, qdef, on_truncate="ignore"):
            return (st, qdef)

        def score_noul_batch(self, encs):
            return [0.48 if "edge" in st else 0.90 for st, _ in encs]

        def score_choice_batch(self, encs):
            out = []
            for st, qdef in encs:
                if qdef["id"] == "clinc_s1":
                    out.append(prim_s1_map[st])
                else:
                    # banking intent idx 1
                    p2 = [0.01] * 15
                    p2[1] = 0.86
                    out.append(p2)
            return out

        def close(self):
            close_events.append("primary")

    kev08_calls: list[list[str]] = []
    kev4b_calls: list[list[str]] = []

    class _FakeKev08b:
        def score_multi_states(self, pairs, batch_docs=8):
            kev08_calls.append([st for st, _ in pairs])
            out = []
            for st, qdefs in pairs:
                qid = qdefs[0]["id"]
                if qid == "clinc_s1":
                    if st == "item2":
                        # Sharp -> even after Kev-0.8B T=2.0 stays >= 0.55 (confident travel)
                        out.append([(_onehot_ish(5, 0.97, oos_p=0.005), [0.0] * k)])
                    elif st == "item3":
                        # Uncalibrated raw max=0.65 looks confident, BUT after Kev-0.8B T=2.0 drops < 0.55 -> escalates to Kev-4B!
                        out.append([(_onehot_ish(7, 0.65, oos_p=0.05), [0.0] * k)])
                    else:
                        out.append([(_onehot_ish(3, 0.40, oos_p=0.15), [0.0] * k)])
                elif qid == "clinc_s2_travel":
                    p2 = [0.01] * 15
                    p2[0] = 0.86
                    out.append([(p2, [0.0] * 15)])
                else:
                    out.append([([0.60, 0.40], [0.0, 0.0])])
            return out

        def close(self):
            close_events.append("kev08b")

    class _FakeKev4b:
        def score_multi_states(self, pairs, batch_docs=4):
            kev4b_calls.append([st for st, _ in pairs])
            out = []
            for st, qdefs in pairs:
                qid = qdefs[0]["id"]
                if qid == "clinc_s1":
                    if st == "item3":
                        out.append([(_onehot_ish(7, 0.88, oos_p=0.02), [0.0] * k)])
                    else:
                        # item4: Kev-4B raw p_oos=0.35 >= kev4b oos_threshold=0.30 -> OOS!
                        out.append([(_onehot_ish(3, 0.55, oos_p=0.35), [0.0] * k)])
                elif qid == "clinc_s2_work":
                    p2 = [0.01] * 15
                    p2[2] = 0.86
                    out.append([(p2, [0.0] * 15)])
                else:
                    out.append([([0.15, 0.85], [0.0, 0.0])])
            return out

        def close(self):
            close_events.append("kev4b")

    prof_laya = CalibrationProfile(
        model_id="laya_hetero",
        task_key="clinc_s1",
        method="temperature",
        temperature=1.0,
        oos_index=oos_idx,
        oos_threshold=0.28,
    )
    prof_kev08 = CalibrationProfile(
        model_id="kev_08b",
        task_key="clinc_s1",
        method="temperature",
        temperature=2.0,
        oos_index=oos_idx,
        oos_threshold=0.35,
    )
    prof_kev4b = CalibrationProfile(
        model_id="kev_4b",
        task_key="clinc_s1",
        method="temperature",
        temperature=1.0,
        oos_index=oos_idx,
        oos_threshold=0.30,
    )

    with DecisionCascadeScorer(
        _FakePrimary(),
        escalation_factory=_FakeKev08b,
        tertiary_factory=_FakeKev4b,
        multiclass_profile=prof_laya,
        escalation_multiclass_profile=prof_kev08,
        tertiary_multiclass_profile=prof_kev4b,
        binary_profile=CalibrationProfile(model_id="laya", task_key="rt", method="platt", platt_a=1.0, platt_b=0.0),
        escalation_binary_profile=CalibrationProfile(model_id="kev08", task_key="rt", method="platt", platt_a=1.0, platt_b=0.0),
        tertiary_binary_profile=CalibrationProfile(model_id="kev4b", task_key="rt", method="platt", platt_a=1.2, platt_b=0.1),
        confidence_threshold=0.60,
        tertiary_threshold=0.55,
    ) as cascade:
        res = cascade.score_multiclass_cascade(
            ["item0", "item1", "item2", "item3", "item4"],
            s1_qdef,
            stage1_labels=CLINC_STAGE1_DOMAINS,
            stage2_qdefs=s2_qdefs,
        )
        assert res["escalated_indices"] == [2, 3, 4]
        assert res["tertiary_escalated_indices"] == [3, 4]
        assert res["tier_counts"] == {"primary": 2, "escalation": 1, "tertiary": 2}
        assert kev08_calls == [["item2", "item3", "item4"], ["item2"]]
        assert kev4b_calls == [["item3", "item4"], ["item3"]]
        assert res["oos_preds"] == [False, True, False, False, True]
        assert res["stage1_pred_domains"] == ["banking", "out_of_scope", "travel", "work", "out_of_scope"]
        assert res["stage2_preds"][1] is None and res["stage2_preds"][4] is None
        assert res["stage2_preds"][0][0] == "banking"
        assert res["stage2_preds"][2][0] == "travel"
        assert res["stage2_preds"][3][0] == "work"

        # Verify seamless compatibility with fast_shared evaluate_clinc_two_stage!
        cl_items = [
            {"domain": "banking", "intent_name": CLINC150_DOMAINS["banking"][1], "is_oos": False},
            {"domain": "out_of_scope", "intent_name": "oos", "is_oos": True},
            {"domain": "travel", "intent_name": CLINC150_DOMAINS["travel"][0], "is_oos": False},
            {"domain": "work", "intent_name": CLINC150_DOMAINS["work"][2], "is_oos": False},
            {"domain": "out_of_scope", "intent_name": "oos", "is_oos": True},
        ]
        cl_report = evaluate_clinc_two_stage(cl_items, res["stage1_probs"], res["stage2_preds"])
        assert cl_report["n_valid"] == 5
        assert cl_report["end_to_end_151_acc"] == 1.0
        assert cl_report["oos_f1"] == 1.0
        assert cl_report["oos_auroc"] == 1.0

        # Verify 3-tier binary cascade
        bin_res = cascade.score_binary_cascade(
            ["easy", "edge"],
            {"id": "rt", "type": "noul"},
            uncertainty_margin=0.05,
            tertiary_margin=0.15,
        )
        assert bin_res["escalated_indices"] == [1]
        assert bin_res["tertiary_escalated_indices"] == [1]
        assert bin_res["tier_counts"] == {"primary": 1, "escalation": 0, "tertiary": 1}
        assert bin_res["probs"][1] > 0.80

    assert close_events == ["primary", "kev08b", "kev4b"]
    # Idempotent close does not re-close
    cascade.close()
    assert close_events == ["primary", "kev08b", "kev4b"]


def test_decision_cascade_lazy_skip_and_direct_tertiary_cpu() -> None:
    """Verify tertiary_factory is never invoked when Tier 2 handles all escalations, and direct primary->tertiary works."""
    tert_calls = 0

    class _Prim:
        def encode_item(self, st, qdef, on_truncate="ignore"):
            return (st, qdef)

        def score_choice_batch(self, encs):
            return [[0.45, 0.35, 0.20] for _ in encs]

    class _Esc:
        def score_multi_states(self, pairs, batch_docs=8):
            return [[([0.85, 0.10, 0.05], [0.0, 0.0, 0.0])] for _ in pairs]

    class _Tert:
        def score_multi_states(self, pairs, batch_docs=4):
            return [[([0.10, 0.85, 0.05], [0.0, 0.0, 0.0])] for _ in pairs]

    def _tert_factory():
        nonlocal tert_calls
        tert_calls += 1
        return _Tert()

    # Tier 2 confident -> tertiary_factory NOT called!
    c1 = DecisionCascadeScorer(
        _Prim(),
        escalation=_Esc(),
        tertiary_factory=_tert_factory,
        confidence_threshold=0.60,
        tertiary_threshold=0.60,
    )
    r1 = c1.score_multiclass_cascade(["x"], {"id": "s1", "options": ["a", "b", "out_of_scope"]})
    assert tert_calls == 0
    assert r1["tier_counts"] == {"primary": 0, "escalation": 1, "tertiary": 0}

    # Direct primary -> tertiary when escalation is None
    c2 = DecisionCascadeScorer(
        _Prim(),
        tertiary_factory=_tert_factory,
        tertiary_threshold=0.60,
    )
    r2 = c2.score_multiclass_cascade(["x"], {"id": "s1", "options": ["a", "b", "out_of_scope"]})
    assert tert_calls == 1
    assert r2["tertiary_escalated_indices"] == [0]
    assert r2["stage1_pred_domains"] == ["b"]
