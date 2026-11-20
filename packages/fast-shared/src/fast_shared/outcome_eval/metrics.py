"""Effective Outcome Throughput (EOT), paired-SE statistics, and Two-Gate promotion logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


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

    verdict: str  # "PASS" or "FAIL"
    profile: str  # "lossless" or "quantized"
    gate1_quality_pass: bool
    gate2_memory_pass: bool
    gate3_eot_pass: bool
    delta_objective_score_e2e: float
    quality_floor: float
    se_paired: float
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
            "delta_objective_score_e2e": round(float(self.delta_objective_score_e2e), 6),
            "quality_floor": round(float(self.quality_floor), 6),
            "se_paired": round(float(self.se_paired), 6),
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
    """Verify ``FourBucketMemoryProbe`` ``combined_with_model_mapped_gb <= 0.80 * host_mem_gb`` and guard status."""
    reasons: list[str] = []
    if device_guard_tripped:
        reasons.append("DeviceMemoryGuard tripped (exceeded 0.98 * recommendedMaxWorkingSetSize)")
    combined_gb: float | None = None
    if isinstance(memory_4bucket, Mapping) and memory_4bucket.get("combined_with_model_mapped_gb") is not None:
        combined_gb = float(memory_4bucket["combined_with_model_mapped_gb"])
    ceiling_gb: float | None = None
    if host_mem_gb is not None and float(host_mem_gb) > 0.0:
        ceiling_gb = float(max_host_fraction) * float(host_mem_gb)
        if combined_gb is not None and combined_gb > ceiling_gb:
            reasons.append(
                f"FourBucketMemoryProbe combined_with_model_mapped_gb={combined_gb:.3f}GB "
                f"> {max_host_fraction * 100.0:.0f}% ceiling ({ceiling_gb:.2f}GB of {host_mem_gb:.1f}GB)"
            )
    return (len(reasons) == 0), ceiling_gb, combined_gb, reasons


def evaluate_promotion_gates(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    profile: str = "lossless",
    min_eot_ratio: float = 1.0,
    host_mem_gb: float | None = None,
    max_host_fraction: float = 0.80,
    base_effective_scores: Sequence[float] | None = None,
    cand_effective_scores: Sequence[float] | None = None,
) -> GateEvaluation:
    """Apply the Two-Gate + Unified-Memory Promotion Contract.

    Profiles:
      - ``lossless``: ``delta_S >= -max(0.003, 1.96 * SE_paired)`` and ``repeat_loop_count == 0``.
      - ``quantized``: ``delta_S >= -0.010`` and ``mask_fallback_rate <= max(0.005, 2.0 * base_fb_rate)``.
    Memory Gate:
      - ``combined_with_model_mapped_gb <= 0.80 * host_mem_gb`` and ``device_guard_tripped == False``.
    Throughput Gate:
      - ``EOT_candidate / EOT_baseline >= min_eot_ratio`` (``1.00`` or ``1.03`` for custom kernels).
    """
    prof = str(profile).strip().lower()
    if prof not in ("lossless", "quantized"):
        raise ValueError(f"Unsupported gate profile={profile!r}; expected 'lossless' or 'quantized'")

    reasons: list[str] = []

    base_s = float(baseline.get("objective_score_e2e", 0.0))
    cand_s = float(candidate.get("objective_score_e2e", 0.0))
    delta_s = cand_s - base_s

    if base_effective_scores is not None and cand_effective_scores is not None:
        paired = compute_paired_se(base_effective_scores, cand_effective_scores)
        se_paired = float(paired["se_paired"])
    else:
        se_paired = float(candidate.get("se_paired_vs_baseline") or 0.0)

    base_diag = baseline.get("diagnostics") if isinstance(baseline.get("diagnostics"), Mapping) else {}
    cand_diag = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), Mapping) else {}
    repeat_loops = int(cand_diag.get("repeat_loop_count", 0) or 0)
    base_fb = float(base_diag.get("mask_fallback_rate", 0.0) or 0.0)
    cand_fb = float(cand_diag.get("mask_fallback_rate", 0.0) or 0.0)

    # Gate 1: Objective Quality Non-Inferiority
    g1_ok = True
    if prof == "lossless":
        quality_floor = -max(0.003, 1.96 * se_paired)
        if delta_s < quality_floor:
            g1_ok = False
            reasons.append(f"Gate1(lossless) quality regression: delta_S={delta_s:+.4f} < floor {quality_floor:+.4f}")
        if repeat_loops > 0:
            g1_ok = False
            reasons.append(f"Gate1(lossless) repeat_loop_count={repeat_loops} > 0")
    else:
        quality_floor = -0.010
        if delta_s < quality_floor:
            g1_ok = False
            reasons.append(f"Gate1(quantized) quality regression: delta_S={delta_s:+.4f} < floor -0.0100")
        fb_limit = max(0.005, 2.0 * base_fb)
        if cand_fb > fb_limit:
            g1_ok = False
            reasons.append(
                f"Gate1(quantized) mask_fallback_rate={cand_fb:.4f} > limit {fb_limit:.4f} (baseline={base_fb:.4f})"
            )

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

    verdict = "PASS" if (g1_ok and g2_ok and g3_ok) else "FAIL"
    return GateEvaluation(
        verdict=verdict,
        profile=prof,
        gate1_quality_pass=g1_ok,
        gate2_memory_pass=g2_ok,
        gate3_eot_pass=g3_ok,
        delta_objective_score_e2e=delta_s,
        quality_floor=quality_floor,
        se_paired=se_paired,
        eot_ratio=eot_ratio if math.isfinite(eot_ratio) else 0.0,
        min_eot_ratio=float(min_eot_ratio),
        memory_ceiling_gb=ceiling_gb,
        combined_with_model_mapped_gb=combined_gb,
        reasons=reasons,
    )


__all__ = [
    "GateEvaluation",
    "OutcomeSummary",
    "compute_corpus_outcome_metrics",
    "compute_effective_item_scores",
    "compute_outcome_metrics",
    "compute_paired_se",
    "compute_unpaired_se",
    "evaluate_memory_gate",
    "evaluate_promotion_gates",
]
