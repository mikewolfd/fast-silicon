"""Category 2 deterministic decision scorers: SciPy-free AUROC/DeLong/ECE, CLINC150 2-stage, RewardBench-2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .hub_loader import ALLOWED_REWARDBENCH2_SUBSETS, CLINC150_DOMAINS

CLINC_STAGE1_DOMAINS: tuple[str, ...] = (
    "banking",
    "credit_cards",
    "kitchen_and_dining",
    "home",
    "auto_and_commute",
    "travel",
    "utility",
    "work",
    "small_talk",
    "meta",
    "out_of_scope",
)

# Concise domain hints chosen so all 11 options + instructions stay well below Laya head_max_len=192 (0% truncation).
CLINC_STAGE1_LABELS: dict[str, str] = {
    "banking": "bank accounts, bills, transfers, fraud",
    "credit_cards": "cards, APR, limits, rewards, credit score",
    "kitchen_and_dining": "recipes, calories, restaurants, reservations",
    "home": "music, reminders, calendar, shopping, smart home",
    "auto_and_commute": "car maintenance, gas, tires, traffic, directions",
    "travel": "flights, hotels, visas, luggage, currency",
    "utility": "weather, alarms, time, calls, math, spelling",
    "work": "PTO, payday, taxes, W2, meetings, insurance",
    "small_talk": "greetings, jokes, bot identity, chat",
    "meta": "assistant settings, volume, language, yes/no",
    "out_of_scope": "falls outside all supported domains",
}


def check_probability_validity(
    probs_batch: Sequence[Sequence[float]],
    *,
    tol: float = 1e-3,
) -> list[bool]:
    """Validate that each probability vector is non-empty, finite, in [0, 1], and sums to 1 within ``tol``."""
    out: list[bool] = []
    for vec in probs_batch:
        if not vec:
            out.append(False)
            continue
        arr = np.asarray(vec, dtype=np.float64)
        ok = (
            bool(np.all(np.isfinite(arr)))
            and bool(np.all(arr >= -tol))
            and bool(np.all(arr <= 1.0 + tol))
            and abs(float(np.sum(arr)) - 1.0) < float(tol)
        )
        out.append(ok)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """SciPy-free 1-based average midranks (exact tie handling) in O(N log N)."""
    n = int(values.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_v = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        avg_rank = 0.5 * float((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Exact SciPy-free Mann-Whitney / trapezoidal AUROC with exact tie midranks."""
    y = np.asarray([1 if bool(v) else 0 for v in labels], dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"Mismatched labels/scores shapes: {y.shape} vs {s.shape}")
    if y.size == 0:
        return 0.5
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(s)
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    u_stat = sum_pos_ranks - 0.5 * float(n_pos * (n_pos + 1))
    return float(u_stat / float(n_pos * n_neg))


def compute_auroc_delong_se(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Exact SciPy-free DeLong (1988) standard error for AUROC using structural placement values."""
    y = np.asarray([1 if bool(v) else 0 for v in labels], dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    m = int(pos.size)
    n = int(neg.size)
    if m < 2 or n < 2:
        return 0.0
    # Compute placement values via O((m+n) log(m+n)) searchsorted rather than O(m*n) matrix.
    neg_sorted = np.sort(neg)
    pos_sorted = np.sort(pos)

    # V10[i] = P(pos[i] > neg) + 0.5 * P(pos[i] == neg)
    lt_neg = np.searchsorted(neg_sorted, pos, side="left")
    le_neg = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (lt_neg.astype(np.float64) + 0.5 * (le_neg - lt_neg).astype(np.float64)) / float(n)

    # V01[j] = P(pos > neg[j]) + 0.5 * P(pos == neg[j])
    gt_pos = m - np.searchsorted(pos_sorted, neg, side="right")
    eq_pos = np.searchsorted(pos_sorted, neg, side="right") - np.searchsorted(pos_sorted, neg, side="left")
    v01 = (gt_pos.astype(np.float64) + 0.5 * eq_pos.astype(np.float64)) / float(m)

    var_v10 = float(np.var(v10, ddof=1))
    var_v01 = float(np.var(v01, ddof=1))
    var_auc = var_v10 / float(m) + var_v01 / float(n)
    return float(math.sqrt(max(0.0, var_auc)))


def compute_binary_macro_f1(
    labels: Sequence[int | bool],
    probs: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary ``macro_f1``, ``f1_pos``, ``f1_neg``, and ``accuracy`` at ``p >= threshold``."""
    y = np.asarray([1 if bool(v) else 0 for v in labels], dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if y.size == 0:
        return {"macro_f1": 0.0, "f1_pos": 0.0, "f1_neg": 0.0, "accuracy": 0.0}
    pred = (p >= float(threshold)).astype(np.int64)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    f1_pos = (2.0 * tp / float(2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    f1_neg = (2.0 * tn / float(2 * tn + fn + fp)) if (2 * tn + fn + fp) > 0 else 0.0
    acc = float(tp + tn) / float(y.size)
    return {
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "accuracy": acc,
    }


def compute_brier_score(labels: Sequence[int | bool], probs: Sequence[float]) -> float:
    """Mean squared probability error ``mean((p_true - y)^2)``."""
    y = np.asarray([1.0 if bool(v) else 0.0 for v in labels], dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def compute_ece(
    labels: Sequence[int | bool],
    probs: Sequence[float],
    *,
    n_bins: int = 15,
) -> float:
    """Exact equal-width ``n_bins`` Expected Calibration Error (default ``15`` bins)."""
    y = np.asarray([1.0 if bool(v) else 0.0 for v in labels], dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    n = int(y.size)
    if n == 0:
        return 0.0
    bins = max(1, int(n_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            avg_conf = float(np.mean(p[mask]))
            avg_acc = float(np.mean(y[mask]))
            ece += (float(cnt) / float(n)) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_multiclass_nll(
    gold_indices: Sequence[int],
    probs_batch: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood ``-mean(log(max(p[gold], eps)))``."""
    if not gold_indices:
        return 0.0
    losses: list[float] = []
    for g, pvec in zip(gold_indices, probs_batch, strict=True):
        if 0 <= int(g) < len(pvec) and math.isfinite(float(pvec[int(g)])):
            pr = max(float(eps), min(1.0, float(pvec[int(g)])))
        else:
            pr = float(eps)
        losses.append(-math.log(pr))
    return float(np.mean(losses)) if losses else 0.0


# ---------------------------------------------------------------------------
# Sub-Track 2A: RAGTruth (`noul`)
# ---------------------------------------------------------------------------


def build_ragtruth_noul_qdef() -> dict[str, Any]:
    """Canonical ``noul`` typed question definition for ``wandb/RAGTruth-processed``."""
    return {
        "id": "ragtruth_faithfulness",
        "type": "noul",
        "instructions": (
            "Is every factual claim in the response fully supported by the provided context "
            "without hallucination or contradiction?"
        ),
        "criteria": {
            "false": "Contains unsupported or contradicted claims.",
            "true": "Fully grounded and faithful to the context.",
        },
    }


def build_ragtruth_state(item: Mapping[str, Any]) -> dict[str, str]:
    """Structured state payload passed to ``LayaAneBackend`` / ``Kev08bMlxBackend``."""
    return {
        "context": str(item.get("context") or ""),
        "question": str(item.get("query") or ""),
        "response": str(item.get("output") or ""),
    }


def evaluate_ragtruth_noul(
    items: Sequence[Mapping[str, Any]],
    p_true_list: Sequence[float],
    *,
    valid_flags: Sequence[bool] | None = None,
    token_counts: Sequence[int] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score RAGTruth binary faithfulness predictions with AUROC, Macro-F1, Brier, 15-bin ECE, and length slices."""
    if len(items) != len(p_true_list):
        raise ValueError(f"Mismatched items={len(items)} vs p_true_list={len(p_true_list)}")
    n = len(items)
    valids = (
        [bool(v) for v in valid_flags]
        if valid_flags is not None
        else [math.isfinite(float(p)) and (-1e-3 <= float(p) <= 1.0 + 1e-3) for p in p_true_list]
    )
    labels = [int(bool(it.get("is_faithful", it.get("label", 0)))) for it in items]
    probs = [float(p) if v else 0.5 for p, v in zip(p_true_list, valids)]

    valid_labels = [y for y, v in zip(labels, valids) if v]
    valid_probs = [p for p, v in zip(probs, valids) if v]

    auroc = compute_auroc(valid_labels, valid_probs) if valid_labels else 0.0
    auroc_se = compute_auroc_delong_se(valid_labels, valid_probs) if valid_labels else 0.0
    f1_res = compute_binary_macro_f1(valid_labels, valid_probs, threshold=threshold)
    brier = compute_brier_score(valid_labels, valid_probs)
    ece15 = compute_ece(valid_labels, valid_probs, n_bins=15)

    per_item_correct = [
        (1.0 if ((p >= threshold) == bool(y)) else 0.0) if v else 0.0
        for y, p, v in zip(labels, probs, valids)
    ]

    slices: dict[str, dict[str, float | int]] = {}
    if token_counts is not None and len(token_counts) == n:
        for sl_name, pred_fn in (("le_512", lambda t: t <= 512), ("gt_512", lambda t: t > 512)):
            idxs = [i for i, tc in enumerate(token_counts) if pred_fn(int(tc)) and valids[i]]
            if idxs:
                sy = [labels[i] for i in idxs]
                sp = [probs[i] for i in idxs]
                sf1 = compute_binary_macro_f1(sy, sp, threshold=threshold)
                slices[sl_name] = {
                    "n": len(idxs),
                    "auroc": round(compute_auroc(sy, sp), 6),
                    "macro_f1": round(sf1["macro_f1"], 6),
                    "accuracy": round(sf1["accuracy"], 6),
                }

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in valids if v)),
        "auroc": round(auroc, 6),
        "auroc_se_delong": round(auroc_se, 6),
        "macro_f1": round(f1_res["macro_f1"], 6),
        "accuracy": round(f1_res["accuracy"], 6),
        "brier": round(brier, 6),
        "ece_15": round(ece15, 6),
        "length_slices": slices,
        "per_item_valid": valids,
        "per_item_scores": per_item_correct,
    }


# ---------------------------------------------------------------------------
# Sub-Track 2B: CLINC150 2-Stage (K=11 -> K=15) & Banking77 (K=77 / 2-Stage)
# ---------------------------------------------------------------------------


def build_clinc_stage1_qdef() -> dict[str, Any]:
    """Stage 1 ``choice`` (K=11: 10 domains + out_of_scope) designed for 0% Laya ``head_max_len=192`` truncation."""
    return {
        "id": "clinc_stage1_domain",
        "type": "choice",
        "instructions": "Classify the user utterance into one domain or out_of_scope.",
        "criteria": {d: CLINC_STAGE1_LABELS[d] for d in CLINC_STAGE1_DOMAINS},
    }


def build_clinc_stage2_qdef(domain: str) -> dict[str, Any]:
    """Stage 2 ``choice`` (K=15 fine intents within ``domain``) with 0% Laya ``head_max_len=192`` truncation."""
    if domain not in CLINC150_DOMAINS:
        raise ValueError(f"Unknown CLINC150 domain {domain!r}; expected one of {sorted(CLINC150_DOMAINS)}")
    intents = CLINC150_DOMAINS[domain]
    return {
        "id": f"clinc_stage2_{domain}",
        "type": "choice",
        "instructions": f"Select the exact {domain.replace('_', ' ')} intent.",
        "criteria": [name.replace("_", " ") for name in intents],
    }


def evaluate_clinc_two_stage(
    items: Sequence[Mapping[str, Any]],
    stage1_probs: Sequence[Sequence[float]],
    stage2_preds: Sequence[tuple[str, Sequence[float]] | None],
    *,
    stage1_valid: Sequence[bool] | None = None,
    stage2_valid: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Score 2-stage ``K=11 -> K=15`` CLINC150 predictions (151-class accuracy, OOS AUROC/F1, NLL).

    ``stage2_preds[i]`` is ``(pred_domain, stage2_intent_probs)`` for items predicted in-scope at Stage 1,
    or ``None`` when Stage 1 predicted ``out_of_scope``.
    """
    n = len(items)
    if len(stage1_probs) != n or len(stage2_preds) != n:
        raise ValueError("Mismatched lengths across items, stage1_probs, and stage2_preds")

    v1 = list(stage1_valid) if stage1_valid is not None else check_probability_validity(stage1_probs)
    oos_idx = CLINC_STAGE1_DOMAINS.index("out_of_scope")
    domain_to_idx = {d: i for i, d in enumerate(CLINC_STAGE1_DOMAINS)}

    valids: list[bool] = []
    per_item_correct: list[float] = []
    stage1_hits = 0
    oos_labels: list[int] = []
    oos_scores: list[float] = []
    stage1_gold_idxs: list[int] = []
    stage1_valid_probs: list[Sequence[float]] = []

    for i, item in enumerate(items):
        p1 = stage1_probs[i]
        ok1 = bool(v1[i]) and len(p1) == len(CLINC_STAGE1_DOMAINS)
        gold_domain = str(item.get("domain") or "out_of_scope")
        gold_intent = str(item.get("intent_name") or "oos")
        gold_is_oos = bool(item.get("is_oos", gold_domain == "out_of_scope"))

        if not ok1:
            valids.append(False)
            per_item_correct.append(0.0)
            continue

        oos_labels.append(1 if gold_is_oos else 0)
        oos_scores.append(float(p1[oos_idx]))
        stage1_gold_idxs.append(domain_to_idx.get(gold_domain, oos_idx))
        stage1_valid_probs.append(p1)

        pred_dom_idx = int(np.argmax(np.asarray(p1, dtype=np.float64)))
        pred_dom = CLINC_STAGE1_DOMAINS[pred_dom_idx]
        if pred_dom == gold_domain:
            stage1_hits += 1

        if pred_dom == "out_of_scope":
            valids.append(True)
            per_item_correct.append(1.0 if gold_is_oos else 0.0)
            continue

        s2_entry = stage2_preds[i]
        ok2 = (stage2_valid[i] if stage2_valid is not None else True) and s2_entry is not None
        if not ok2 or s2_entry is None:
            valids.append(False)
            per_item_correct.append(0.0)
            continue

        s2_dom, p2 = s2_entry
        if s2_dom not in CLINC150_DOMAINS or len(p2) != len(CLINC150_DOMAINS[s2_dom]):
            valids.append(False)
            per_item_correct.append(0.0)
            continue
        if not check_probability_validity([p2])[0]:
            valids.append(False)
            per_item_correct.append(0.0)
            continue

        pred_intent = CLINC150_DOMAINS[s2_dom][int(np.argmax(np.asarray(p2, dtype=np.float64)))]
        valids.append(True)
        per_item_correct.append(1.0 if (not gold_is_oos and pred_intent == gold_intent) else 0.0)

    n_valid = sum(1 for v in valids if v)
    acc_e2e = (sum(per_item_correct) / float(n)) if n > 0 else 0.0
    acc_valid = (sum(per_item_correct) / float(n_valid)) if n_valid > 0 else 0.0
    s1_acc = (float(stage1_hits) / float(max(1, len(stage1_gold_idxs)))) if stage1_gold_idxs else 0.0
    oos_auroc = compute_auroc(oos_labels, oos_scores) if oos_labels else 0.5
    oos_f1 = compute_binary_macro_f1(oos_labels, oos_scores, threshold=0.5)["f1_pos"] if oos_labels else 0.0
    s1_nll = compute_multiclass_nll(stage1_gold_idxs, stage1_valid_probs)

    return {
        "n_items": n,
        "n_valid": n_valid,
        "stage1_domain_acc": round(s1_acc, 6),
        "end_to_end_151_acc": round(acc_e2e, 6),
        "end_to_end_151_acc_given_valid": round(acc_valid, 6),
        "oos_auroc": round(oos_auroc, 6),
        "oos_f1": round(oos_f1, 6),
        "stage1_nll": round(s1_nll, 6),
        "per_item_valid": valids,
        "per_item_scores": per_item_correct,
    }


def partition_banking77_clusters(label_names: Sequence[str], *, n_clusters: int = 8) -> list[list[str]]:
    """Deterministically partition 77 Banking77 labels into ``n_clusters`` groups (<=10 options each for Laya)."""
    sorted_labels = list(label_names)
    k = max(1, int(n_clusters))
    base = len(sorted_labels) // k
    rem = len(sorted_labels) % k
    clusters: list[list[str]] = []
    pos = 0
    for i in range(k):
        sz = base + (1 if i < rem else 0)
        clusters.append(sorted_labels[pos : pos + sz])
        pos += sz
    return clusters


def build_banking77_flat_qdef(label_names: Sequence[str]) -> dict[str, Any]:
    """Flat K=77 ``choice`` question for ``Kev-0.8B`` ``PrefixTree`` branching stress."""
    return {
        "id": "banking77_flat",
        "type": "choice",
        "instructions": "Select the banking customer support intent that best matches the query.",
        "criteria": [name.replace("_", " ") for name in label_names],
    }


def evaluate_multiclass_choice(
    gold_indices: Sequence[int],
    probs_batch: Sequence[Sequence[float]],
    *,
    valid_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Generic K-way ``choice`` evaluator (used for Banking77 K=77 and single-stage choice tasks)."""
    n = len(gold_indices)
    if len(probs_batch) != n:
        raise ValueError(f"Mismatched gold_indices={n} vs probs_batch={len(probs_batch)}")
    valids = list(valid_flags) if valid_flags is not None else check_probability_validity(probs_batch)
    scores: list[float] = []
    valid_gold: list[int] = []
    valid_probs: list[Sequence[float]] = []
    for g, pvec, v in zip(gold_indices, probs_batch, valids):
        if not v or not pvec:
            scores.append(0.0)
            continue
        pred = int(np.argmax(np.asarray(pvec, dtype=np.float64)))
        scores.append(1.0 if pred == int(g) else 0.0)
        valid_gold.append(int(g))
        valid_probs.append(pvec)
    n_valid = sum(1 for v in valids if v)
    return {
        "n_items": n,
        "n_valid": n_valid,
        "top1_accuracy": round(sum(scores) / float(max(1, n)), 6),
        "top1_accuracy_given_valid": round(sum(scores) / float(max(1, n_valid)), 6),
        "nll": round(compute_multiclass_nll(valid_gold, valid_probs), 6),
        "per_item_valid": valids,
        "per_item_scores": scores,
    }


# ---------------------------------------------------------------------------
# Sub-Track 2C: RewardBench-2 Best-of-4 (`score`)
# ---------------------------------------------------------------------------


def build_rewardbench2_score_qdef() -> dict[str, Any]:
    """Canonical 5-level ordinal ``score`` rubric for ``allenai/reward-bench-2`` Best-of-4."""
    return {
        "id": "rewardbench2_quality",
        "type": "score",
        "instructions": "Rate the assistant response for accuracy, helpfulness, safety, and instruction following.",
        "criteria": [
            "very poor or harmful",
            "poor with major flaws",
            "acceptable with minor issues",
            "good and accurate",
            "excellent, thorough, and faithful",
        ],
    }


def expected_reward_score(
    probs: Sequence[float],
    raw_logits: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Compute continuous expected reward ``R = sum_{l=0..L-1} l * P(l)`` plus raw-logit tie-breaker.

    When FP16 softmax saturates to one-hot on two competing candidates, raw pre-softmax logits
    ``sum_{l} l * (z_l - mean(z))`` break ties deterministically without random jitter.
    """
    p = np.asarray(probs, dtype=np.float64)
    levels = np.arange(p.size, dtype=np.float64)
    exp_r = float(np.dot(levels, p))
    if raw_logits is not None and len(raw_logits) == p.size:
        z = np.asarray(raw_logits, dtype=np.float64)
        if bool(np.all(np.isfinite(z))):
            z_centered = z - float(np.mean(z))
            tie_r = float(np.dot(levels, z_centered))
            return exp_r, tie_r
    return exp_r, exp_r


def evaluate_reward_bench_2(
    items: Sequence[Mapping[str, Any]],
    candidate_probs: Sequence[Sequence[Sequence[float]]],
    candidate_logits: Sequence[Sequence[Sequence[float]] | None] | None = None,
    *,
    valid_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Score ``allenai/reward-bench-2`` Best-of-4 prompts (5-domain unweighted macro accuracy)."""
    n = len(items)
    if len(candidate_probs) != n:
        raise ValueError(f"Mismatched items={n} vs candidate_probs={len(candidate_probs)}")

    valids: list[bool] = []
    per_item_correct: list[float] = []
    by_subset: dict[str, list[float]] = {}

    for i, item in enumerate(items):
        sub = str(item.get("subset") or "unknown")
        gold_idx = int(item["gold_idx"])
        c_probs = candidate_probs[i]
        c_logits = candidate_logits[i] if (candidate_logits is not None and i < len(candidate_logits)) else None

        ok = (
            (valid_flags[i] if valid_flags is not None else True)
            and len(c_probs) == 4
            and all(check_probability_validity(c_probs))
        )
        if not ok:
            valids.append(False)
            per_item_correct.append(0.0)
            by_subset.setdefault(sub, []).append(0.0)
            continue

        scored: list[tuple[float, float, int]] = []
        for j in range(4):
            lg = c_logits[j] if (c_logits is not None and j < len(c_logits)) else None
            er, tr = expected_reward_score(c_probs[j], lg)
            # Negative j ensures deterministic lowest-index tie-break only if both R and raw logit tie identically.
            scored.append((er, tr, -j))
        best = max(scored)
        pred_idx = -best[2]
        hit = 1.0 if pred_idx == gold_idx else 0.0
        valids.append(True)
        per_item_correct.append(hit)
        by_subset.setdefault(sub, []).append(hit)

    subset_acc: dict[str, float] = {}
    for sub in ALLOWED_REWARDBENCH2_SUBSETS:
        if sub in by_subset and by_subset[sub]:
            subset_acc[sub] = round(float(np.mean(by_subset[sub])), 6)
    for sub, vals in sorted(by_subset.items()):
        if sub not in subset_acc and vals:
            subset_acc[sub] = round(float(np.mean(vals)), 6)

    macro_vals = [subset_acc[s] for s in ALLOWED_REWARDBENCH2_SUBSETS if s in subset_acc]
    macro_acc = float(np.mean(macro_vals)) if macro_vals else (float(np.mean(per_item_correct)) if per_item_correct else 0.0)
    micro_acc = float(np.mean(per_item_correct)) if per_item_correct else 0.0

    return {
        "n_items": n,
        "n_valid": int(sum(1 for v in valids if v)),
        "macro_best_of_4_acc": round(macro_acc, 6),
        "micro_best_of_4_acc": round(micro_acc, 6),
        "subset_accuracy": subset_acc,
        "per_item_valid": valids,
        "per_item_scores": per_item_correct,
    }


__all__ = [
    "CLINC_STAGE1_DOMAINS",
    "CLINC_STAGE1_LABELS",
    "build_banking77_flat_qdef",
    "build_clinc_stage1_qdef",
    "build_clinc_stage2_qdef",
    "build_ragtruth_noul_qdef",
    "build_ragtruth_state",
    "build_rewardbench2_score_qdef",
    "check_probability_validity",
    "compute_auroc",
    "compute_auroc_delong_se",
    "compute_binary_macro_f1",
    "compute_brier_score",
    "compute_ece",
    "compute_multiclass_nll",
    "evaluate_clinc_two_stage",
    "evaluate_multiclass_choice",
    "evaluate_ragtruth_noul",
    "evaluate_reward_bench_2",
    "expected_reward_score",
    "partition_banking77_clusters",
]
