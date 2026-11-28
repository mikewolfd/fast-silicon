"""Effective Outcome Throughput (EOT), paired-SE statistics, and Two-Gate promotion logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

DEFAULT_LOSSLESS_TOLERANCE = 0.003


@dataclass(frozen=True)
class OutcomeSummary:
    """Dimensionally exact Effective Outcome Throughput summary for one benchmark run."""

    n_items: int
    n_valid: int
    valid_rate: float
    wall_sec: float
    raw_items_per_sec: float
    valid_outputs_per_sec: float
    objective_score_e2e: float
    objective_score_given_valid: float
    effective_outcome_per_sec: float
    se_unpaired: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": int(self.n_items),
            "n_valid": int(self.n_valid),
            "valid_rate": round(float(self.valid_rate), 6),
            "wall_sec": round(float(self.wall_sec), 4),
            "raw_items_per_sec": round(float(self.raw_items_per_sec), 4),
            "valid_outputs_per_sec": round(float(self.valid_outputs_per_sec), 4),
            "objective_score_e2e": round(float(self.objective_score_e2e), 6),
            "objective_score_given_valid": round(float(self.objective_score_given_valid), 6),
            "effective_outcome_per_sec": round(float(self.effective_outcome_per_sec), 4),
            "se_unpaired": round(float(self.se_unpaired), 6),
        }


def compute_effective_item_scores(
    valid_flags: Sequence[bool | int],
    item_scores: Sequence[float],
) -> np.ndarray:
    """Compute per-item effective scores ``e_i = v_i * s_i`` with ``e_i = 0`` if invalid or non-finite."""
    if len(valid_flags) != len(item_scores):
        raise ValueError(f"Mismatched lengths: valid_flags={len(valid_flags)} vs item_scores={len(item_scores)}")
    n = len(valid_flags)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    v = np.fromiter((1.0 if bool(x) else 0.0 for x in valid_flags), dtype=np.float64, count=n)
    s = np.asarray(item_scores, dtype=np.float64)
    s = np.where(np.isfinite(s), np.clip(s, 0.0, 1.0), 0.0)
    return v * s


def compute_unpaired_se(values: Sequence[float] | np.ndarray) -> float:
    """Standard error of the sample mean (``ddof=1`` when ``N >= 2``, else ``0.0``)."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    return float(std / math.sqrt(float(n)))


def compute_paired_se(
    baseline_scores: Sequence[float] | np.ndarray,
    candidate_scores: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Compute paired-difference mean, exact ``SE_paired``, and 95% normal confidence interval.

    ``SE_paired = sqrt((1 / (N * (N - 1))) * sum_i (d_i - mean(d))^2)`` where ``d_i = cand_i - base_i``.
    Works identically across continuous metrics (``nDCG@10``, ``Field-F1``) and binary accuracy.
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    cand = np.asarray(candidate_scores, dtype=np.float64)
    if base.shape != cand.shape or base.ndim != 1:
        raise ValueError(f"Expected matching 1D score arrays, got {base.shape} vs {cand.shape}")
    n = int(base.size)
    if n == 0:
        return {"n_paired": 0, "delta_mean": 0.0, "se_paired": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    diff = cand - base
    delta_mean = float(np.mean(diff))
    if n == 1:
        se = 0.0
    else:
        centered = diff - delta_mean
        se = float(math.sqrt(float(np.dot(centered, centered)) / float(n * (n - 1))))
    half_ci = 1.96 * se
    return {
        "n_paired": n,
        "delta_mean": delta_mean,
        "se_paired": se,
        "ci95_low": delta_mean - half_ci,
        "ci95_high": delta_mean + half_ci,
    }


def compute_outcome_metrics(
    valid_flags: Sequence[bool | int],
    item_scores: Sequence[float],
    wall_sec: float,
) -> OutcomeSummary:
    """Compute dimensionally exact Effective Outcome Throughput (avoiding ``V_rate^2`` double-penalization).

    ``EOT = sum_i(v_i * s_i) / T_e2e = (N_valid / T_e2e) * S_{obj|valid} = (N / T_e2e) * S_{obj,e2e}``.
    """
    eff = compute_effective_item_scores(valid_flags, item_scores)
    n = int(eff.size)
    n_valid = int(sum(1 for x in valid_flags if bool(x)))
    t_e2e = float(wall_sec)
    if t_e2e <= 0.0 or not math.isfinite(t_e2e):
        raise ValueError(f"wall_sec must be finite and > 0, got {wall_sec!r}")
    if n == 0:
        return OutcomeSummary(
            n_items=0,
            n_valid=0,
            valid_rate=0.0,
            wall_sec=t_e2e,
            raw_items_per_sec=0.0,
            valid_outputs_per_sec=0.0,
            objective_score_e2e=0.0,
            objective_score_given_valid=0.0,
            effective_outcome_per_sec=0.0,
            se_unpaired=0.0,
        )

    total_credit = float(np.sum(eff))
    valid_rate = float(n_valid) / float(n)
    raw_ips = float(n) / t_e2e
    valid_ops = float(n_valid) / t_e2e
    s_e2e = total_credit / float(n)
    s_given_valid = (total_credit / float(n_valid)) if n_valid > 0 else 0.0
    eot = total_credit / t_e2e
    se = compute_unpaired_se(eff)
    return OutcomeSummary(
        n_items=n,
        n_valid=n_valid,
        valid_rate=valid_rate,
        wall_sec=t_e2e,
        raw_items_per_sec=raw_ips,
        valid_outputs_per_sec=valid_ops,
        objective_score_e2e=s_e2e,
        objective_score_given_valid=s_given_valid,
        effective_outcome_per_sec=eot,
        se_unpaired=se,
    )


def compute_corpus_outcome_metrics(
    *,
    n_items: int,
    n_valid: int,
    corpus_score_given_valid: float,
    wall_sec: float,
    se_unpaired: float = 0.0,
) -> OutcomeSummary:
    """Compute exact EOT for corpus-level ranking metrics (e.g. ``AUROC``) without squaring ``V_rate``."""
    n = int(n_items)
    nv = max(0, min(n, int(n_valid)))
    t_e2e = float(wall_sec)
    if t_e2e <= 0.0 or not math.isfinite(t_e2e):
        raise ValueError(f"wall_sec must be finite and > 0, got {wall_sec!r}")
    s_val = float(corpus_score_given_valid)
    if not math.isfinite(s_val):
        s_val = 0.0
    s_val = max(0.0, min(1.0, s_val))
    if n <= 0:
        return OutcomeSummary(0, 0, 0.0, t_e2e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    valid_rate = float(nv) / float(n)
    s_e2e = valid_rate * s_val
    raw_ips = float(n) / t_e2e
    valid_ops = float(nv) / t_e2e
    eot = valid_ops * s_val  # Identical to raw_ips * s_e2e; never multiplies valid_ops * s_e2e.
    return OutcomeSummary(
        n_items=n,
        n_valid=nv,
        valid_rate=valid_rate,
        wall_sec=t_e2e,
        raw_items_per_sec=raw_ips,
        valid_outputs_per_sec=valid_ops,
        objective_score_e2e=s_e2e,
        objective_score_given_valid=s_val if nv > 0 else 0.0,
        effective_outcome_per_sec=eot,
        se_unpaired=max(0.0, float(se_unpaired)),
    )


@dataclass(frozen=True)
class GateEvaluation:
    """Two-Gate + Memory Ceiling promotion verdict for a candidate vs baseline."""

    verdict: str  # PASS, FAIL, or UNRESOLVED
    profile: str  # "lossless" or "quantized"
    gate1_quality_pass: bool
    gate2_memory_pass: bool
    gate3_eot_pass: bool
    quality_status: str
    retrieval_coverage_pass: bool | None
    delta_objective_score_e2e: float | None
    quality_floor: float
    n_paired: int
    se_paired: float | None
    ci95_low: float | None
    ci95_high: float | None
    eot_ratio: float
    min_eot_ratio: float
    memory_ceiling_gb: float | None
    combined_with_model_mapped_gb: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "profile": self.profile,
            "gate1_quality_pass": bool(self.gate1_quality_pass),
            "gate2_memory_pass": bool(self.gate2_memory_pass),
            "gate3_eot_pass": bool(self.gate3_eot_pass),
            "quality_status": self.quality_status,
            "retrieval_coverage_pass": self.retrieval_coverage_pass,
            "delta_objective_score_e2e": round(self.delta_objective_score_e2e, 6)
            if self.delta_objective_score_e2e is not None else None,
            "quality_floor": round(float(self.quality_floor), 6),
            "n_paired": self.n_paired,
            "se_paired": round(self.se_paired, 6) if self.se_paired is not None else None,
            "ci95_low": round(self.ci95_low, 6) if self.ci95_low is not None else None,
            "ci95_high": round(self.ci95_high, 6) if self.ci95_high is not None else None,
            "eot_ratio": round(float(self.eot_ratio), 4),
            "min_eot_ratio": round(float(self.min_eot_ratio), 4),
            "memory_ceiling_gb": round(float(self.memory_ceiling_gb), 3)
            if self.memory_ceiling_gb is not None
            else None,
            "combined_with_model_mapped_gb": round(float(self.combined_with_model_mapped_gb), 4)
            if self.combined_with_model_mapped_gb is not None
            else None,
            "reasons": list(self.reasons),
        }


def evaluate_memory_gate(
    memory_4bucket: Mapping[str, Any] | None,
    *,
    host_mem_gb: float | None,
    max_host_fraction: float = 0.80,
    device_guard_tripped: bool = False,
) -> tuple[bool, float | None, float | None, list[str]]:
    """Verify ``FourBucketMemoryProbe`` peak/combined GB ``<= 0.80 * host_mem_gb`` and guard status."""
    reasons: list[str] = []
    if device_guard_tripped:
        reasons.append("DeviceMemoryGuard tripped (exceeded 0.98 * recommendedMaxWorkingSetSize)")
    combined_gb: float | None = None
    if isinstance(memory_4bucket, Mapping):
        candidates_gb: list[float] = []
        for k in (
            "combined_with_model_mapped_gb",
            "peak_footprint_gb",
            "peak_combined_with_model_mapped_gb",
        ):
            val = memory_4bucket.get(k)
            if val is not None:
                candidates_gb.append(float(val))
        phys_fp = memory_4bucket.get("phys_footprint")
        if isinstance(phys_fp, Mapping) and phys_fp.get("peak_footprint_gb") is not None:
            candidates_gb.append(float(phys_fp["peak_footprint_gb"]))
        if candidates_gb:
            combined_gb = max(candidates_gb)
    ceiling_gb: float | None = None
    if host_mem_gb is not None and float(host_mem_gb) > 0.0:
        ceiling_gb = float(max_host_fraction) * float(host_mem_gb)
        if combined_gb is not None and combined_gb > ceiling_gb:
            reasons.append(
                f"FourBucketMemoryProbe effective_peak_gb={combined_gb:.3f}GB "
                f"> {max_host_fraction * 100.0:.0f}% ceiling ({ceiling_gb:.2f}GB of {host_mem_gb:.1f}GB)"
            )
    return (len(reasons) == 0), ceiling_gb, combined_gb, reasons


def retrieval_coverage_errors(record: Mapping[str, Any]) -> list[str]:
    """Check intended query counts, including optional component results and repeated runs."""
    if record.get("category") != 1:
        return []
    intended, valid = record.get("n_items"), record.get("n_valid")
    errors = []
    if intended is None or valid is None:
        errors.append("retrieval coverage unavailable: intended and valid query counts are required")
    elif intended <= 0 or valid != intended:
        errors.append(f"retrieval coverage: {valid}/{intended} intended queries produced usable results")
    for name, sub in (record.get("sub_benchmarks") or {}).items():
        if "n_valid_queries" in sub and (not sub.get("n_queries") or sub["n_valid_queries"] != sub["n_queries"]):
            errors.append(f"{name} retrieval coverage: {sub['n_valid_queries']}/{sub.get('n_queries')} queries")
    if record.get("incomplete_retrieval_repeats"):
        errors.append(f"incomplete retrieval repeats: {record['incomplete_retrieval_repeats']}")
    return errors


def _paired_score_evidence(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    base_items: Sequence[float] | None, cand_items: Sequence[float] | None,
) -> tuple[dict[str, float | int] | None, list[str]]:
    arrays: list[np.ndarray] = []
    mismatch_reasons: list[str] = []
    valid = True
    for arm, record, items in (("baseline", baseline, base_items), ("candidate", candidate, cand_items)):
        if items is None:
            items = record.get("per_item_effective_scores")
        if items is None:
            valid = False
            continue
        scores = np.asarray(items, dtype=np.float64)
        if scores.ndim != 1 or scores.size < 2 or not np.all(np.isfinite(scores)):
            valid = False
            continue
        # Stored item scores and headlines are rounded independently to six decimal places.
        mean = record.get("objective_score_e2e")
        if mean is None or not math.isfinite(float(mean)):
            valid = False
            continue
        item_mean = float(scores.mean())
        if not math.isclose(item_mean, float(mean), abs_tol=2e-6):
            mismatch_reasons.append(
                f"{arm} paired item scores mean ({item_mean:.6f}) disagrees with "
                f"objective_score_e2e ({float(mean):.6f})"
            )
            valid = False
            continue
        if scores.size != record.get("n_items", scores.size):
            valid = False
            continue
        arrays.append(scores)
    if not valid or len(arrays) != 2 or arrays[0].shape != arrays[1].shape:
        return None, mismatch_reasons
    return compute_paired_se(*arrays), mismatch_reasons


def evaluate_promotion_gates(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    profile: str = "lossless",
    lossless_tolerance: float = DEFAULT_LOSSLESS_TOLERANCE,
    min_eot_ratio: float = 1.0,
    host_mem_gb: float | None = None,
    max_host_fraction: float = 0.80,
    base_effective_scores: Sequence[float] | None = None,
    cand_effective_scores: Sequence[float] | None = None,
) -> GateEvaluation:
    """Apply the Two-Gate + Unified-Memory Promotion Contract.

    Profiles:
      - ``lossless``: the paired 95% lower bound clears the fixed score-loss tolerance and there are no repeat loops.
        Missing pairs or an interval crossing the floor leave non-inferiority unresolved.
      - ``quantized``: ``delta_S >= -0.010`` and ``mask_fallback_rate <= max(0.005, 2.0 * base_fb_rate)``.
    Memory Gate:
      - ``combined_with_model_mapped_gb <= 0.80 * host_mem_gb`` and ``device_guard_tripped == False``.
    Throughput Gate:
      - ``EOT_candidate / EOT_baseline >= min_eot_ratio`` (``1.00`` or ``1.03`` for custom kernels).
    """
    prof = str(profile).strip().lower()
    if prof not in ("lossless", "quantized"):
        raise ValueError(f"Unsupported gate profile={profile!r}; expected 'lossless' or 'quantized'")

    if not 0.0 <= lossless_tolerance <= 1.0:
        raise ValueError("lossless_tolerance must be a fixed score difference in [0, 1]")

    reasons: list[str] = []
    base_s = baseline.get("objective_score_e2e")
    cand_s = candidate.get("objective_score_e2e")
    delta_s = float(cand_s) - float(base_s) if base_s is not None and cand_s is not None else math.nan

    base_ids = baseline.get("retrieval_query_order_sha256")
    cand_ids = candidate.get("retrieval_query_order_sha256")
    different_queries = bool(base_ids and cand_ids and base_ids != cand_ids)
    paired: dict[str, float | int] | None
    paired_mismatch_reasons: list[str]
    if different_queries:
        paired, paired_mismatch_reasons = None, []
    else:
        paired, paired_mismatch_reasons = _paired_score_evidence(
            baseline, candidate, base_effective_scores, cand_effective_scores,
        )

    base_diag = baseline.get("diagnostics") if isinstance(baseline.get("diagnostics"), Mapping) else {}
    cand_diag = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), Mapping) else {}
    base_repeats = int(base_diag.get("repeat_loop_count", 0) or 0)
    cand_repeats = int(cand_diag.get("repeat_loop_count", 0) or 0)
    base_fb = float(base_diag.get("mask_fallback_rate", 0.0) or 0.0)
    cand_fb = float(cand_diag.get("mask_fallback_rate", 0.0) or 0.0)

    # Gate 1: Objective Quality Non-Inferiority
    quality_floor = -lossless_tolerance if prof == "lossless" else -0.010
    # A score exactly on the floor passes despite floating-point subtraction roundoff.
    threshold = quality_floor - 1e-12
    quality_status = "PASS"
    if not math.isfinite(delta_s):
        quality_status = "UNRESOLVED"
        reasons.append("Gate1 quality unresolved: finite baseline and candidate scores are required")
    elif different_queries:
        quality_status = "UNRESOLVED"
        reasons.append("Gate1 quality unresolved: intended query IDs or order differ between runs")
    elif prof == "lossless":
        if paired is None:
            quality_status = "UNRESOLVED"
            if paired_mismatch_reasons:
                reasons.extend(
                    f"Gate1(lossless) quality unresolved: {reason}"
                    for reason in paired_mismatch_reasons
                )
            else:
                reasons.append("Gate1(lossless) quality unresolved: complete paired scores for at least two items "
                               "must match both reported means")
        elif paired["ci95_high"] < threshold:
            quality_status = "FAIL"
            reasons.append(f"Gate1(lossless) quality regression: paired 95% upper bound "
                           f"{paired['ci95_high']:+.4f} < fixed floor {quality_floor:+.4f}")
        elif paired["ci95_low"] < threshold:
            quality_status = "UNRESOLVED"
            reasons.append(f"Gate1(lossless) quality unresolved: paired 95% interval "
                           f"[{paired['ci95_low']:+.4f}, {paired['ci95_high']:+.4f}] crosses fixed floor {quality_floor:+.4f}")
    else:
        if paired_mismatch_reasons:
            reasons.extend(
                f"Paired score evidence ignored: {reason}"
                for reason in paired_mismatch_reasons
            )
        if delta_s < threshold:
            quality_status = "FAIL"
            reasons.append(f"Gate1(quantized) quality regression: delta_S={delta_s:+.4f} < floor {quality_floor:+.4f}")

    diagnostics_ok = True
    if prof == "lossless":
        if cand_repeats > base_repeats:
            diagnostics_ok = False
            reasons.append(f"Gate1(lossless) repeat_loop_count={cand_repeats} > baseline={base_repeats}")
    else:
        fb_limit = max(0.005, 2.0 * base_fb)
        if cand_fb > fb_limit:
            diagnostics_ok = False
            reasons.append(
                f"Gate1(quantized) mask_fallback_rate={cand_fb:.4f} > limit {fb_limit:.4f} (baseline={base_fb:.4f})"
            )
    g1_ok = quality_status == "PASS" and diagnostics_ok

    coverage_ok = None
    if baseline.get("category") == 1 or candidate.get("category") == 1:
        coverage_errors = [
            f"{arm} {error}" for arm, rec in (("baseline", baseline), ("candidate", candidate))
            for error in retrieval_coverage_errors(rec)
        ]
        if baseline.get("n_items") != candidate.get("n_items") or different_queries:
            coverage_errors.append("retrieval comparison requires the same intended queries")
        coverage_ok = not coverage_errors
        reasons.extend(coverage_errors)

    # Gate 2: Unified Memory Headroom
    mem_map = candidate.get("memory_4bucket_gb") if isinstance(candidate.get("memory_4bucket_gb"), Mapping) else None
    guard_tripped = bool(cand_diag.get("device_guard_tripped", False))
    g2_ok, ceiling_gb, combined_gb, mem_reasons = evaluate_memory_gate(
        mem_map,
        host_mem_gb=host_mem_gb,
        max_host_fraction=max_host_fraction,
        device_guard_tripped=guard_tripped,
    )
    reasons.extend(mem_reasons)

    # Gate 3: Effective Outcome Throughput
    base_eot = float(baseline.get("effective_outcome_per_sec", 0.0))
    cand_eot = float(candidate.get("effective_outcome_per_sec", 0.0))
    if base_eot > 0.0:
        eot_ratio = cand_eot / base_eot
    else:
        eot_ratio = float("inf") if cand_eot > 0.0 else 0.0
    g3_ok = eot_ratio >= float(min_eot_ratio)
    if not g3_ok:
        reasons.append(
            f"Gate3(EOT) ratio={eot_ratio:.4f} ({cand_eot:.4f} / {base_eot:.4f}) < required {float(min_eot_ratio):.4f}"
        )

    verdict = quality_status if diagnostics_ok and g2_ok and g3_ok and coverage_ok is not False else "FAIL"
    return GateEvaluation(
        verdict=verdict,
        profile=prof,
        gate1_quality_pass=g1_ok,
        gate2_memory_pass=g2_ok,
        gate3_eot_pass=g3_ok,
        quality_status=quality_status,
        retrieval_coverage_pass=coverage_ok,
        delta_objective_score_e2e=delta_s if math.isfinite(delta_s) else None,
        quality_floor=quality_floor,
        n_paired=int(paired["n_paired"]) if paired else 0,
        se_paired=float(paired["se_paired"]) if paired else None,
        ci95_low=float(paired["ci95_low"]) if paired else None,
        ci95_high=float(paired["ci95_high"]) if paired else None,
        eot_ratio=eot_ratio if math.isfinite(eot_ratio) else 0.0,
        min_eot_ratio=float(min_eot_ratio),
        memory_ceiling_gb=ceiling_gb,
        combined_with_model_mapped_gb=combined_gb,
        reasons=reasons,
    )


_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF via Acklam's rational approximation (<1e-8 error, pure stdlib)."""
    pf = float(p)
    if not math.isfinite(pf) or pf <= 0.0 or pf >= 1.0:
        raise ValueError(f"p must be finite and strictly in (0, 1), got {p!r}")
    p_low = 0.02425
    p_high = 1.0 - p_low
    if pf < p_low:
        q = math.sqrt(-2.0 * math.log(pf))
        num = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5])
        den = ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
        return num / den
    if pf <= p_high:
        q = pf - 0.5
        r = q * q
        num = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q
        den = (((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - pf))
    num = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5])
    den = ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    return -(num / den)


@dataclass(frozen=True)
class PPIComparisonResult:
    """Prediction-Powered Inference (PPI++, arXiv:2311.01453) comparison summary."""

    theta_ppi: float
    theta_classical: float
    theta_judge_unlabeled: float
    theta_judge_labeled: float
    se_ppi: float
    se_classical: float
    ci_low: float
    ci_high: float
    classical_ci_low: float
    classical_ci_high: float
    lambda_hat: float
    alpha: float
    n_labeled: int
    n_unlabeled: int
    n_effective_multiplier: float
    judge_bias_estimate: float
    judge_misaligned: bool
    ci_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_ppi": float(self.theta_ppi),
            "theta_classical": float(self.theta_classical),
            "theta_judge_unlabeled": float(self.theta_judge_unlabeled),
            "theta_judge_labeled": float(self.theta_judge_labeled),
            "se_ppi": float(self.se_ppi),
            "se_classical": float(self.se_classical),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "classical_ci_low": float(self.classical_ci_low),
            "classical_ci_high": float(self.classical_ci_high),
            "lambda_hat": float(self.lambda_hat),
            "alpha": float(self.alpha),
            "n_labeled": int(self.n_labeled),
            "n_unlabeled": int(self.n_unlabeled),
            "n_effective_multiplier": float(self.n_effective_multiplier),
            "judge_bias_estimate": float(self.judge_bias_estimate),
            "judge_misaligned": bool(self.judge_misaligned),
            "ci_method": str(self.ci_method),
        }


def _row_var_ddof1(arr: np.ndarray) -> np.ndarray:
    """Row-wise sample variance (``ddof=1``), guarding exact-constant rows against float noise."""
    return np.where(np.ptp(arr, axis=1) == 0.0, 0.0, np.var(arr, axis=1, ddof=1))


def _ppi_batch_stats(
    f_u: np.ndarray,
    f_l: np.ndarray,
    y_l: np.ndarray,
    *,
    power_tune: bool,
    lam_ov: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized ``(theta_ppi, se_ppi, lam, y_n, f_n, f_N)`` over ``(B, N)`` and ``(B, n)`` batches."""
    b, n = y_l.shape
    n_unlabeled = f_u.shape[1]
    y_n = y_l.mean(axis=1)
    f_n = f_l.mean(axis=1)
    f_N = f_u.mean(axis=1)
    var_f_l = _row_var_ddof1(f_l)

    if lam_ov is not None:
        lam = np.full(b, lam_ov, dtype=np.float64)
    elif not power_tune:
        lam = np.ones(b, dtype=np.float64)
    else:
        cov_yf = np.sum((y_l - y_n[:, None]) * (f_l - f_n[:, None]), axis=1) / float(n - 1)
        cov_yf = np.where((np.ptp(y_l, axis=1) == 0.0) | (np.ptp(f_l, axis=1) == 0.0), 0.0, cov_yf)
        f_all = np.concatenate([f_l, f_u], axis=1)
        var_f_total = _row_var_ddof1(f_all)
        denom = (1.0 + float(n) / float(n_unlabeled)) * var_f_total
        pos = denom > 1e-15
        lam = np.where(pos, np.clip(cov_yf / np.where(pos, denom, 1.0), 0.0, 1.0), 0.0)

    theta = lam * f_N + (y_n - lam * f_n)
    rectifier = y_l - lam[:, None] * f_l
    var_rect = _row_var_ddof1(rectifier)
    var_unlab = _row_var_ddof1(f_u) if n_unlabeled >= 2 else var_f_l
    se = np.sqrt(np.maximum(0.0, var_rect / float(n) + (lam * lam) * var_unlab / float(n_unlabeled)))
    return theta, se, lam, y_n, f_n, f_N


def _validate_ppi_inputs(
    judge_unlabeled: Sequence[float] | np.ndarray,
    judge_labeled: Sequence[float] | np.ndarray,
    human_labeled: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    lambda_override: float | None,
    misaligned_lambda_threshold: float,
    bootstrap_reps: int,
    ci_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float | None, float, int]:
    """Validate and normalize inputs for :func:`compute_ppi_comparison`."""
    f_u = np.asarray(judge_unlabeled, dtype=np.float64)
    f_l = np.asarray(judge_labeled, dtype=np.float64)
    y_l = np.asarray(human_labeled, dtype=np.float64)

    if f_l.ndim != 1 or y_l.ndim != 1 or f_l.shape != y_l.shape:
        raise ValueError(
            f"judge_labeled and human_labeled must be matching 1D arrays, got {f_l.shape} vs {y_l.shape}"
        )
    if f_u.ndim != 1:
        raise ValueError(f"judge_unlabeled must be a 1D array, got shape {f_u.shape}")
    if y_l.size < 2:
        raise ValueError(f"At least 2 paired labeled samples are required, got n={y_l.size}")
    if f_u.size < 1:
        raise ValueError(f"At least 1 unlabeled judge score is required, got N={f_u.size}")
    if not (np.all(np.isfinite(y_l)) and np.all(np.isfinite(f_l)) and np.all(np.isfinite(f_u))):
        raise ValueError("All judge_unlabeled, judge_labeled, and human_labeled values must be finite")

    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or alpha_f <= 0.0 or alpha_f >= 1.0:
        raise ValueError(f"alpha must be strictly in (0, 1), got {alpha!r}")

    lam_ov: float | None = None
    if lambda_override is not None:
        lam_ov = float(lambda_override)
        if not math.isfinite(lam_ov) or lam_ov < 0.0 or lam_ov > 1.0:
            raise ValueError(f"lambda_override must be in [0, 1], got {lambda_override!r}")

    thresh = float(misaligned_lambda_threshold)
    if not math.isfinite(thresh) or thresh < 0.0 or thresh > 1.0:
        raise ValueError(f"misaligned_lambda_threshold must be in [0, 1], got {misaligned_lambda_threshold!r}")

    if ci_method not in ("asymptotic", "bootstrap_percentile", "bootstrap_t"):
        raise ValueError(
            f"Unsupported ci_method={ci_method!r}; expected 'asymptotic', 'bootstrap_percentile', or 'bootstrap_t'"
        )

    reps = int(bootstrap_reps)
    if reps < 0:
        raise ValueError(f"bootstrap_reps must be >= 0, got {bootstrap_reps!r}")
    if ci_method != "asymptotic":
        if reps == 0:
            reps = 1000
        elif reps < 50:
            raise ValueError(f"bootstrap_reps must be 0 (default 1000) or >= 50, got {bootstrap_reps!r}")

    return f_u, f_l, y_l, alpha_f, lam_ov, thresh, reps


def _ppi_bootstrap_ci(
    f_u: np.ndarray,
    f_l: np.ndarray,
    y_l: np.ndarray,
    *,
    theta_ppi: float,
    se_ppi: float,
    alpha: float,
    power_tune: bool,
    lam_ov: float | None,
    n_reps: int,
    ci_method: Literal["bootstrap_percentile", "bootstrap_t"],
    seed: int,
) -> tuple[float, float]:
    """Compute percentile or studentized bootstrap confidence interval for PPI++."""
    if ci_method == "bootstrap_t" and se_ppi <= 1e-15:
        return theta_ppi, theta_ppi

    n = int(y_l.size)
    n_unlabeled = int(f_u.size)
    rng = np.random.default_rng(int(seed))
    # Cap per-chunk allocation (~16 MB) so large unlabeled pools remain memory-bounded.
    chunk_size = max(1, min(n_reps, 2_000_000 // max(n, n_unlabeled)))
    stats = np.empty(n_reps, dtype=np.float64)
    deg_thresh = min(1e-12, 1e-8 * se_ppi) if ci_method == "bootstrap_t" else 0.0

    offset = 0
    while offset < n_reps:
        cur = min(chunk_size, n_reps - offset)
        idx_l = rng.integers(0, n, size=(cur, n))
        idx_u = rng.integers(0, n_unlabeled, size=(cur, n_unlabeled))
        theta_b, se_b, _, _, _, _ = _ppi_batch_stats(
            f_u[idx_u], f_l[idx_l], y_l[idx_l], power_tune=power_tune, lam_ov=lam_ov,
        )
        if ci_method == "bootstrap_percentile":
            stats[offset : offset + cur] = theta_b
        else:
            valid_se = se_b > deg_thresh
            safe_se = np.where(valid_se, se_b, se_ppi)
            t_b = (theta_b - theta_ppi) / safe_se
            stats[offset : offset + cur] = np.where(valid_se, t_b, np.clip(t_b, -50.0, 50.0))
        offset += cur

    if ci_method == "bootstrap_percentile":
        q_lo, q_hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(q_lo), float(q_hi)

    q_hi, q_lo = np.quantile(stats, [1.0 - alpha / 2.0, alpha / 2.0])
    return float(theta_ppi - q_hi * se_ppi), float(theta_ppi - q_lo * se_ppi)


def compute_ppi_comparison(
    judge_unlabeled: Sequence[float] | np.ndarray,
    judge_labeled: Sequence[float] | np.ndarray,
    human_labeled: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    power_tune: bool = True,
    lambda_override: float | None = None,
    misaligned_lambda_threshold: float = 0.2,
    bootstrap_reps: int = 0,
    ci_method: Literal["asymptotic", "bootstrap_percentile", "bootstrap_t"] = "asymptotic",
    seed: int = 0,
) -> PPIComparisonResult:
    """Compute Prediction-Powered Inference (PPI++) comparison from bulk judge + small human anchor scores.

    Defaults to ``ci_method="asymptotic"`` (passes >=92% empirical 95% coverage for n=30-40); pass
    ``ci_method="bootstrap_t"`` when exact >=95% finite-sample coverage is needed on small audits (n <= 40).
    """
    f_u, f_l, y_l, alpha_f, lam_ov, thresh, n_reps = _validate_ppi_inputs(
        judge_unlabeled,
        judge_labeled,
        human_labeled,
        alpha=alpha,
        lambda_override=lambda_override,
        misaligned_lambda_threshold=misaligned_lambda_threshold,
        bootstrap_reps=bootstrap_reps,
        ci_method=ci_method,
    )
    n = int(y_l.size)
    n_unlabeled = int(f_u.size)

    theta_arr, se_arr, lam_arr, y_n_arr, f_n_arr, f_N_arr = _ppi_batch_stats(
        f_u[None, :], f_l[None, :], y_l[None, :], power_tune=power_tune, lam_ov=lam_ov,
    )
    theta_ppi = float(theta_arr[0])
    se_ppi = float(se_arr[0])
    lam = float(lam_arr[0])
    y_n = float(y_n_arr[0])
    f_n = float(f_n_arr[0])
    f_N = float(f_N_arr[0])

    var_y = float(_row_var_ddof1(y_l[None, :])[0])
    se_classical = math.sqrt(max(0.0, var_y / float(n)))

    if se_ppi > 1e-15:
        n_effective_multiplier = (se_classical / se_ppi) ** 2
    elif se_classical <= 1e-15:
        n_effective_multiplier = 1.0
    else:
        n_effective_multiplier = float("inf")

    z = _normal_ppf(1.0 - alpha_f / 2.0)
    classical_ci_low = y_n - z * se_classical
    classical_ci_high = y_n + z * se_classical

    if ci_method == "asymptotic":
        ci_low = theta_ppi - z * se_ppi
        ci_high = theta_ppi + z * se_ppi
    else:
        ci_low, ci_high = _ppi_bootstrap_ci(
            f_u,
            f_l,
            y_l,
            theta_ppi=theta_ppi,
            se_ppi=se_ppi,
            alpha=alpha_f,
            power_tune=power_tune,
            lam_ov=lam_ov,
            n_reps=n_reps,
            ci_method=ci_method,
            seed=seed,
        )

    return PPIComparisonResult(
        theta_ppi=theta_ppi,
        theta_classical=y_n,
        theta_judge_unlabeled=f_N,
        theta_judge_labeled=f_n,
        se_ppi=se_ppi,
        se_classical=se_classical,
        ci_low=ci_low,
        ci_high=ci_high,
        classical_ci_low=classical_ci_low,
        classical_ci_high=classical_ci_high,
        lambda_hat=lam,
        alpha=alpha_f,
        n_labeled=n,
        n_unlabeled=n_unlabeled,
        n_effective_multiplier=n_effective_multiplier,
        judge_bias_estimate=f_n - y_n,
        judge_misaligned=lam < thresh,
        ci_method=ci_method,
    )


__all__ = [
    "DEFAULT_LOSSLESS_TOLERANCE",
    "GateEvaluation",
    "OutcomeSummary",
    "PPIComparisonResult",
    "compute_corpus_outcome_metrics",
    "compute_effective_item_scores",
    "compute_outcome_metrics",
    "compute_paired_se",
    "compute_ppi_comparison",
    "compute_unpaired_se",
    "evaluate_memory_gate",
    "evaluate_promotion_gates",
    "retrieval_coverage_errors",
]
