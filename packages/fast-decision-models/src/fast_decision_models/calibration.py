"""Pure-NumPy post-hoc calibration (Platt + multiclass temperature) and Laya-to-Kev confidence cascade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _np() -> Any:
    import numpy as np

    return np


def check_probability_validity(probs: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Validate finite non-negative probability vectors/matrices and normalize cleanly."""
    np = _np()
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Expected non-empty probability array.")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("Probability array contains NaN or Inf values.")
    if arr.ndim == 1:
        clipped = np.clip(arr, float(eps), 1.0 - float(eps) if arr.size == 1 else None)
        if arr.size == 1:
            return clipped
        total = float(np.sum(np.maximum(arr, 0.0)))
        if total <= 0.0:
            raise ValueError("Probability vector has non-positive sum.")
        normed = np.maximum(arr, 0.0) / total
        return normed
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return np.clip(arr, float(eps), 1.0 - float(eps))
        pos = np.maximum(arr, 0.0)
        row_sums = np.sum(pos, axis=1, keepdims=True)
        if not bool(np.all(row_sums > 0.0)):
            raise ValueError("Probability matrix contains a row with non-positive sum.")
        return pos / row_sums
    raise ValueError(f"Expected 1D or 2D probabilities, got shape={arr.shape}")


def fit_platt_scaling(
    scores: Any,
    y: Any,
    *,
    from_probs: bool = False,
    reg: float = 1e-2,
    max_iter: int = 60,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """Fit damped-Newton L2-regularized 1D Platt scaling `sigmoid(a * z + b)`."""
    np = _np()
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    lbl = np.asarray(y, dtype=np.float64).reshape(-1)
    if raw.shape != lbl.shape:
        raise ValueError(f"Shape mismatch between scores={raw.shape} and y={lbl.shape}")
    if raw.size == 0:
        raise ValueError("Cannot fit Platt scaling on empty data.")
    if not bool(np.all(np.isfinite(raw))) or not bool(np.all(np.isfinite(lbl))):
        raise ValueError("Platt inputs contain NaN or Inf.")
    if from_probs:
        p_clamped = np.clip(raw, float(eps), 1.0 - float(eps))
        logits = np.log(p_clamped / (1.0 - p_clamped))
    else:
        logits = raw

    a, b = 1.0, 0.0
    reg_f = max(1e-8, float(reg))
    for _ in range(max(1, int(max_iter))):
        z = np.clip(a * logits + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-4)
        diff = p - lbl
        ga = float(np.sum(diff * logits) + reg_f * a)
        gb = float(np.sum(diff) + reg_f * b)
        haa = float(np.sum(w * logits * logits) + reg_f)
        hab = float(np.sum(w * logits))
        hbb = float(np.sum(w) + reg_f)
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = float(np.clip((hbb * ga - hab * gb) / det, -1.0, 1.0))
        db = float(np.clip((-hab * ga + haa * gb) / det, -1.0, 1.0))
        a -= 0.5 * da
        b -= 0.5 * db
        if max(abs(da), abs(db)) < 1e-6:
            break
    return float(a), float(b)


def apply_platt_scaling(
    scores: Any,
    a: float,
    b: float,
    *,
    from_probs: bool = False,
    eps: float = 1e-7,
) -> np.ndarray:
    """Apply fitted `(a, b)` Platt parameters to logits or probabilities."""
    np = _np()
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    if from_probs:
        p_clamped = np.clip(arr, float(eps), 1.0 - float(eps))
        logits = np.log(p_clamped / (1.0 - p_clamped))
    else:
        logits = arr
    z = np.clip(float(a) * logits + float(b), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_multiclass_temperature(
    probs_or_logp: Any,
    temperature: float,
    *,
    from_probs: bool = True,
    eps: float = 1e-30,
) -> np.ndarray:
    """Apply positive scalar temperature `T` to multiclass log-probs (`0` top-1 rank flips)."""
    np = _np()
    t = float(temperature)
    if t <= 0.0 or not np.isfinite(t):
        raise ValueError(f"Expected positive finite temperature, got {temperature}")
    arr = np.asarray(probs_or_logp, dtype=np.float64)
    was_1d = arr.ndim == 1
    mat = arr[None, :] if was_1d else arr
    if mat.ndim != 2 or mat.shape[1] < 2:
        raise ValueError(f"Expected 2D [N, K>=2] multiclass inputs, got shape={arr.shape}")
    if from_probs:
        mat = check_probability_validity(mat, eps=eps)
        logp = np.log(np.clip(mat, float(eps), 1.0))
    else:
        if not bool(np.all(np.isfinite(mat))):
            raise ValueError("Multiclass logits/logp contain NaN or Inf.")
        logp = mat
    sc = logp / t
    sc = sc - np.max(sc, axis=1, keepdims=True)
    ex = np.exp(sc)
    out = ex / np.sum(ex, axis=1, keepdims=True)
    return out[0] if was_1d else out


def fit_multiclass_temperature(
    probs_or_logp: Any,
    y: Any,
    *,
    from_probs: bool = True,
    t_min: float = 0.25,
    t_max: float = 40.0,
    num_grid: int = 79,
    eps: float = 1e-30,
) -> float:
    """Fit NLL-optimal positive temperature scale over `[t_min, t_max]` via log-grid + golden-section."""
    np = _np()
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_lo <= 0.0 or not math.isfinite(t_lo) or not math.isfinite(t_hi) or t_hi < t_lo:
        raise ValueError(f"Expected positive finite temperature bounds t_min <= t_max, got [{t_min}, {t_max}]")
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot fit multiclass temperature on empty data.")
    arr = np.asarray(probs_or_logp, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != labels.shape[0]:
        raise ValueError(f"Shape mismatch: probs_or_logp={arr.shape} vs y={labels.shape}")
    if from_probs:
        arr = check_probability_validity(arr, eps=eps)
        logp = np.log(np.clip(arr, float(eps), 1.0))
    else:
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("Multiclass logits contain NaN or Inf.")
        logp = arr
    if not bool(np.all((labels >= 0) & (labels < arr.shape[1]))):
        raise ValueError(f"Class labels out of bounds for K={arr.shape[1]}")

    row_idx = np.arange(labels.shape[0], dtype=np.int64)

    def _nll_for_t(cand_t: float) -> float:
        pr = apply_multiclass_temperature(logp, cand_t, from_probs=False, eps=eps)
        p_true = np.clip(pr[row_idx, labels], float(eps), 1.0)
        return float(-np.mean(np.log(p_true)))

    if t_hi == t_lo:
        return t_lo

    grid = np.geomspace(t_lo, t_hi, max(3, int(num_grid)))
    nlls = [_nll_for_t(float(cand_t)) for cand_t in grid]
    best_idx = int(np.argmin(np.asarray(nlls, dtype=np.float64)))
    left_t = float(grid[max(0, best_idx - 1)])
    right_t = float(grid[min(len(grid) - 1, best_idx + 1)])

    # Golden-section search in log(T) space (NLL is convex in 1/T).
    a = math.log(left_t)
    b = math.log(right_t)
    phi = 0.5 * (math.sqrt(5.0) - 1.0)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _nll_for_t(math.exp(c))
    fd = _nll_for_t(math.exp(d))
    for _ in range(30):
        if abs(b - a) < 1e-6:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _nll_for_t(math.exp(c))
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _nll_for_t(math.exp(d))

    best_log_t = 0.5 * (a + b)
    best_t = math.exp(best_log_t)
    if nlls[best_idx] < _nll_for_t(best_t):
        return float(grid[best_idx])
    return float(best_t)


def fit_oos_threshold(
    scores_or_probs: Any,
    y_oos: Any,
    *,
    oos_index: int | None = None,
    metric: str = "f1",
    fallback_threshold: float = 0.5,
) -> float:
    """Fit exact threshold `tau_oos` on calibrated OOS probabilities without leakage."""
    np = _np()
    if metric not in ("f1", "macro_f1", "youden"):
        raise ValueError(f"Unsupported OOS threshold metric {metric!r}; expected 'f1', 'macro_f1', or 'youden'")
    arr = np.asarray(scores_or_probs, dtype=np.float64)
    if arr.ndim == 2:
        if oos_index is None:
            raise ValueError("oos_index is required when passing 2D probability matrix to fit_oos_threshold.")
        idx = int(oos_index)
        if idx < 0 or idx >= arr.shape[1]:
            raise ValueError(f"oos_index={idx} out of bounds for K={arr.shape[1]}")
        scores = arr[:, idx]
    elif arr.ndim == 1:
        scores = arr
    else:
        raise ValueError(f"Expected 1D scores or 2D probabilities, got shape={arr.shape}")

    lbl = np.asarray(y_oos, dtype=np.int64).reshape(-1)
    scores = scores.reshape(-1)
    if scores.shape != lbl.shape:
        raise ValueError(f"Shape mismatch: scores={scores.shape} vs y_oos={lbl.shape}")
    if scores.size == 0:
        raise ValueError("Cannot fit OOS threshold on empty data.")
    if not bool(np.all(np.isfinite(scores))):
        raise ValueError("OOS scores contain NaN or Inf.")

    y_bin = (lbl != 0).astype(np.int64)
    n_total = int(y_bin.size)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float(fallback_threshold)

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_bin[order]

    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)

    # Candidate midpoints between distinct sorted score values + clamped extremes.
    diff_mask = np.diff(s_sorted) > 1e-12
    split_idxs = np.flatnonzero(diff_mask)
    cand_thresholds: list[float] = [float(np.clip(s_sorted[0] - 1e-6, 1e-6, 1.0 - 1e-6))]
    cand_metrics: list[float] = []

    def _eval_counts(tp: int, fp: int, fn: int, tn: int) -> float:
        if metric == "f1":
            denom = 2 * tp + fp + fn
            return (2.0 * tp / float(denom)) if denom > 0 else 0.0
        if metric == "macro_f1":
            d_pos = 2 * tp + fp + fn
            d_neg = 2 * tn + fn + fp
            f1_p = (2.0 * tp / float(d_pos)) if d_pos > 0 else 0.0
            f1_n = (2.0 * tn / float(d_neg)) if d_neg > 0 else 0.0
            return 0.5 * (f1_p + f1_n)
        return (float(tp) / float(n_pos)) - (float(fp) / float(n_neg))

    cand_metrics.append(_eval_counts(n_pos, n_neg, 0, 0))
    for i in split_idxs:
        fn = int(cum_pos[i])
        tn = int(cum_neg[i])
        tp = n_pos - fn
        fp = n_neg - tn
        t_mid = 0.5 * float(s_sorted[i] + s_sorted[i + 1])
        cand_thresholds.append(float(np.clip(t_mid, 1e-6, 1.0 - 1e-6)))
        cand_metrics.append(_eval_counts(tp, fp, fn, tn))
    cand_thresholds.append(float(np.clip(s_sorted[-1] + 1e-6, 1e-6, 1.0 - 1e-6)))
    cand_metrics.append(_eval_counts(0, 0, n_pos, n_neg))

    best_val = max(cand_metrics)
    tied = [t for t, m in zip(cand_thresholds, cand_metrics) if m >= best_val - 1e-12]
    return float(tied[len(tied) // 2])


def apply_oos_decision(
    probs: Any,
    *,
    oos_index: int,
    oos_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict `(is_oos_mask, stage1_pred_indices)` using calibrated `p[oos_index] >= oos_threshold`."""
    np = _np()
    mat = check_probability_validity(probs)
    was_1d = mat.ndim == 1
    arr2d = mat[None, :] if was_1d else mat
    k_dim = arr2d.shape[1]
    if k_dim < 2:
        raise ValueError(f"Expected K>=2 classes for OOS decision, got K={k_dim}")
    idx = int(oos_index)
    if idx < 0 or idx >= k_dim:
        raise ValueError(f"oos_index={idx} out of bounds for K={k_dim}")
    tau = float(oos_threshold)
    if not math.isfinite(tau) or tau <= 0.0 or tau >= 1.0:
        raise ValueError(f"Expected finite oos_threshold in (0, 1), got {oos_threshold}")

    is_oos = arr2d[:, idx] >= tau
    inscope_scores = arr2d.copy()
    inscope_scores[:, idx] = -1.0
    best_inscope = np.argmax(inscope_scores, axis=1)
    pred_indices = np.where(is_oos, idx, best_inscope)
    if was_1d:
        return is_oos[0:1], pred_indices[0:1]
    return is_oos, pred_indices


def align_oos_probabilities(
    probs: Any,
    *,
    oos_index: int,
    oos_threshold: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Monotonically align OOS mass so `p'[oos] >= 0.5` and `argmax(p') == oos` iff `p[oos] >= oos_threshold`.

    Maps `[0, tau_oos)` into `[0, 1/K - eps)` and `[tau_oos, 1]` into `[0.5 + eps, 1]` while scaling
    in-scope probabilities proportionally. Preserves exact `oos_auroc`, in-scope class order, and simplex sum.
    """
    np = _np()
    arr = np.asarray(probs, dtype=np.float64)
    was_1d = arr.ndim == 1
    mat = check_probability_validity(arr[None, :] if was_1d else arr)
    k_dim = mat.shape[1]
    if k_dim < 2:
        raise ValueError(f"Expected K>=2 classes for OOS alignment, got K={k_dim}")
    idx = int(oos_index)
    if idx < 0 or idx >= k_dim:
        raise ValueError(f"oos_index={idx} out of bounds for K={k_dim}")
    tau = float(oos_threshold)
    if not math.isfinite(tau) or tau <= 0.0 or tau >= 1.0:
        raise ValueError(f"Expected finite oos_threshold in (0, 1), got {oos_threshold}")

    eps_f = max(1e-9, min(1e-3, float(eps)))
    lo_cap = (1.0 / float(k_dim)) - eps_f
    hi_floor = 0.5 + eps_f

    p_oos = mat[:, idx]
    p_new_oos = np.where(
        p_oos < tau,
        lo_cap * (p_oos / tau),
        hi_floor + (1.0 - hi_floor) * ((p_oos - tau) / (1.0 - tau)),
    )
    p_new_oos = np.clip(p_new_oos, 0.0, 1.0)

    out = mat.copy()
    rem_old = 1.0 - p_oos
    rem_new = 1.0 - p_new_oos
    safe_scale = np.where(rem_old > 1e-12, rem_new / np.maximum(rem_old, 1e-12), 0.0)
    out = out * safe_scale[:, None]
    # Handle degenerate p_oos == 1.0 rows cleanly
    deg = rem_old <= 1e-12
    if bool(np.any(deg)):
        out[deg, :] = 0.0
        p_new_oos = np.where(deg, 1.0, p_new_oos)
    out[:, idx] = p_new_oos
    # Renormalize strictly to 1.0 against tiny float drift
    row_sums = np.sum(out, axis=1, keepdims=True)
    out = out / np.maximum(row_sums, 1e-12)
    return out[0] if was_1d else out


@dataclass(frozen=True)
class CalibrationProfile:
    """Opt-in calibration profile keyed by `(model_id, task_key)` with optional OOS threshold."""

    model_id: str
    task_key: str
    method: str
    platt_a: float = 1.0
    platt_b: float = 0.0
    temperature: float = 1.0
    oos_index: int | None = None
    oos_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.method not in ("platt", "temperature"):
            raise ValueError(f"Unsupported calibration method {self.method!r}")
        if self.method == "temperature" and (self.temperature <= 0.0 or not math.isfinite(self.temperature)):
            raise ValueError(f"Expected positive finite temperature, got {self.temperature}")
        if self.oos_index is not None and int(self.oos_index) < 0:
            raise ValueError(f"Expected non-negative oos_index, got {self.oos_index}")
        if self.oos_threshold is not None:
            tau = float(self.oos_threshold)
            if not math.isfinite(tau) or tau <= 0.0 or tau >= 1.0:
                raise ValueError(f"Expected finite oos_threshold in (0, 1), got {self.oos_threshold}")

    def apply(self, values: Any, *, from_probs: bool = True) -> np.ndarray:
        if self.method == "platt":
            return apply_platt_scaling(values, self.platt_a, self.platt_b, from_probs=from_probs)
        return apply_multiclass_temperature(values, self.temperature, from_probs=from_probs)

    def predict_oos(self, values: Any, *, from_probs: bool = True) -> np.ndarray:
        """Apply calibration and evaluate the calibrated OOS decision rule."""
        np = _np()
        cal = self.apply(values, from_probs=from_probs)
        if self.method == "platt":
            tau = 0.5 if self.oos_threshold is None else float(self.oos_threshold)
            return np.asarray(cal, dtype=np.float64) >= tau
        if self.oos_index is None:
            raise ValueError("Cannot predict OOS on multiclass profile without oos_index.")
        if self.oos_threshold is not None:
            is_oos, _ = apply_oos_decision(cal, oos_index=int(self.oos_index), oos_threshold=float(self.oos_threshold))
            return is_oos
        mat = cal[None, :] if cal.ndim == 1 else cal
        return np.argmax(mat, axis=1) == int(self.oos_index)

    def apply_with_oos_alignment(self, values: Any, *, from_probs: bool = True) -> np.ndarray:
        """Return calibrated multiclass probs aligned to `oos_threshold` at 0.5/argmax when configured."""
        cal = self.apply(values, from_probs=from_probs)
        if self.method == "temperature" and self.oos_index is not None and self.oos_threshold is not None:
            return align_oos_probabilities(
                cal, oos_index=int(self.oos_index), oos_threshold=float(self.oos_threshold)
            )
        return cal

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "task_key": self.task_key,
            "method": self.method,
            "platt_a": float(self.platt_a),
            "platt_b": float(self.platt_b),
            "temperature": float(self.temperature),
        }
        if self.oos_index is not None:
            payload["oos_index"] = int(self.oos_index)
        if self.oos_threshold is not None:
            payload["oos_threshold"] = float(self.oos_threshold)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationProfile":
        raw_idx = raw.get("oos_index")
        raw_tau = raw.get("oos_threshold")
        return cls(
            model_id=str(raw["model_id"]),
            task_key=str(raw["task_key"]),
            method=str(raw["method"]),
            platt_a=float(raw.get("platt_a", 1.0)),
            platt_b=float(raw.get("platt_b", 0.0)),
            temperature=float(raw.get("temperature", 1.0)),
            oos_index=int(raw_idx) if raw_idx is not None else None,
            oos_threshold=float(raw_tau) if raw_tau is not None else None,
        )


def fit_calibrated_profile(
    model_id: str,
    task_key: str,
    *,
    method: str,
    values: Any,
    labels: Any,
    from_probs: bool = True,
    oos_labels: Any = None,
    oos_index: int | None = None,
    oos_metric: str = "f1",
    reg: float = 1e-2,
) -> CalibrationProfile:
    """Fit a `CalibrationProfile` (Platt or multiclass temperature + optional OOS threshold) on one split."""
    if method == "platt":
        a, b = fit_platt_scaling(values, labels, from_probs=from_probs, reg=reg)
        tau_oos = None
        if oos_labels is not None:
            cal_p = apply_platt_scaling(values, a, b, from_probs=from_probs)
            tau_oos = fit_oos_threshold(cal_p, oos_labels, metric=oos_metric)
        return CalibrationProfile(
            model_id=model_id,
            task_key=task_key,
            method="platt",
            platt_a=a,
            platt_b=b,
            oos_threshold=tau_oos,
        )
    if method == "temperature":
        temp = fit_multiclass_temperature(values, labels, from_probs=from_probs)
        tau_oos = None
        if oos_labels is not None:
            if oos_index is None:
                raise ValueError("oos_index is required when oos_labels are provided for temperature scaling.")
            cal_p = apply_multiclass_temperature(values, temp, from_probs=from_probs)
            tau_oos = fit_oos_threshold(cal_p, oos_labels, oos_index=int(oos_index), metric=oos_metric)
        return CalibrationProfile(
            model_id=model_id,
            task_key=task_key,
            method="temperature",
            temperature=temp,
            oos_index=int(oos_index) if oos_index is not None else None,
            oos_threshold=tau_oos,
        )
    raise ValueError(f"Unsupported calibration method {method!r}")


def stratified_kfold_indices(
    labels: Any,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified K-fold `(train_idx, val_idx)` split using pure NumPy."""
    np = _np()
    y = np.asarray(labels).reshape(-1)
    n = int(y.size)
    k = int(n_splits)
    if k < 2:
        raise ValueError(f"Expected n_splits >= 2, got {n_splits}")
    if n < k:
        raise ValueError(f"Cannot split {n} items into {k} folds.")

    rng = np.random.default_rng(int(seed))
    fold_bins: list[list[int]] = [[] for _ in range(k)]
    # Group deterministically by sorted unique label string/value
    unique_vals = sorted(set(y.tolist()), key=str)
    for cls_pos, val in enumerate(unique_vals):
        cls_idxs = np.flatnonzero(y == val)
        perm = cls_idxs[rng.permutation(cls_idxs.size)]
        for offset, item_idx in enumerate(perm.tolist()):
            fold_bins[(offset + cls_pos) % k].append(int(item_idx))

    all_idx = np.arange(n, dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(k):
        val_idx = np.asarray(sorted(fold_bins[f]), dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        splits.append((train_idx, val_idx))
    return splits


def fit_oof_calibrated_scores(
    values: Any,
    labels: Any,
    *,
    model_id: str,
    task_key: str,
    method: str,
    from_probs: bool = True,
    n_splits: int = 5,
    seed: int = 42,
    oos_labels: Any = None,
    oos_index: int | None = None,
    oos_metric: str = "f1",
    align_oos: bool = False,
    reg: float = 1e-2,
) -> dict[str, Any]:
    """Leakage-free out-of-fold calibration: every item is calibrated by a profile fit on other folds."""
    np = _np()
    arr = np.asarray(values, dtype=np.float64)
    lbl = np.asarray(labels).reshape(-1)
    if arr.shape[0] != lbl.shape[0]:
        raise ValueError(f"Shape mismatch: values={arr.shape} vs labels={lbl.shape}")
    oos_arr = np.asarray(oos_labels, dtype=np.int64).reshape(-1) if oos_labels is not None else None
    if oos_arr is not None and oos_arr.shape[0] != lbl.shape[0]:
        raise ValueError(f"Shape mismatch: oos_labels={oos_arr.shape} vs labels={lbl.shape}")

    folds = stratified_kfold_indices(lbl, n_splits=n_splits, seed=seed)
    cal_out = np.zeros_like(arr, dtype=np.float64)
    aligned_out = np.zeros_like(arr, dtype=np.float64)
    oos_preds = np.zeros(lbl.shape[0], dtype=bool)
    fold_profiles: list[CalibrationProfile] = []

    for train_idx, val_idx in folds:
        prof = fit_calibrated_profile(
            model_id,
            task_key,
            method=method,
            values=arr[train_idx],
            labels=lbl[train_idx],
            from_probs=from_probs,
            oos_labels=oos_arr[train_idx] if oos_arr is not None else None,
            oos_index=oos_index,
            oos_metric=oos_metric,
            reg=reg,
        )
        fold_profiles.append(prof)
        val_cal = prof.apply(arr[val_idx], from_probs=from_probs)
        cal_out[val_idx] = val_cal
        if align_oos and method == "temperature" and prof.oos_index is not None and prof.oos_threshold is not None:
            aligned_out[val_idx] = align_oos_probabilities(
                val_cal, oos_index=int(prof.oos_index), oos_threshold=float(prof.oos_threshold)
            )
        else:
            aligned_out[val_idx] = val_cal
        if oos_arr is not None or (method == "temperature" and oos_index is not None):
            oos_preds[val_idx] = prof.predict_oos(arr[val_idx], from_probs=from_probs)

    return {
        "calibrated": cal_out,
        "aligned": aligned_out,
        "oos_preds": oos_preds,
        "fold_profiles": fold_profiles,
    }


class DecisionCascadeScorer:
    """Task-aware 2-tier / 3-tier (`LayaHetero -> Kev-0.8B -> optional Kev-4B`) cascade with strict parity."""

    def __init__(
        self,
        primary: Any,
        escalation: Any | None = None,
        *,
        escalation_factory: Callable[[], Any] | None = None,
        tertiary: Any | None = None,
        tertiary_factory: Callable[[], Any] | None = None,
        binary_profile: CalibrationProfile | None = None,
        multiclass_profile: CalibrationProfile | None = None,
        escalation_binary_profile: CalibrationProfile | None = None,
        escalation_multiclass_profile: CalibrationProfile | None = None,
        tertiary_binary_profile: CalibrationProfile | None = None,
        tertiary_multiclass_profile: CalibrationProfile | None = None,
        confidence_threshold: float = 0.50,
        tertiary_threshold: float = 0.50,
        kev_batch_docs: int = 8,
        tertiary_batch_docs: int = 4,
    ) -> None:
        self.primary = primary
        self._escalation = escalation
        self._escalation_factory = escalation_factory
        self._owns_escalation = escalation is None and escalation_factory is not None
        self._tertiary = tertiary
        self._tertiary_factory = tertiary_factory
        self._owns_tertiary = tertiary is None and tertiary_factory is not None
        self.binary_profile = binary_profile
        self.multiclass_profile = multiclass_profile
        self.escalation_binary_profile = escalation_binary_profile
        self.escalation_multiclass_profile = escalation_multiclass_profile
        self.tertiary_binary_profile = tertiary_binary_profile
        self.tertiary_multiclass_profile = tertiary_multiclass_profile
        self.confidence_threshold = float(confidence_threshold)
        self.tertiary_threshold = float(tertiary_threshold)
        self.kev_batch_docs = max(1, int(kev_batch_docs))
        self.tertiary_batch_docs = max(1, int(tertiary_batch_docs))
        self._closed = False

    def _get_escalation(self) -> Any | None:
        if self._escalation is None and self._escalation_factory is not None:
            self._escalation = self._escalation_factory()
        return self._escalation

    def _get_tertiary(self) -> Any | None:
        if self._tertiary is None and self._tertiary_factory is not None:
            self._tertiary = self._tertiary_factory()
        return self._tertiary

    @staticmethod
    def _extract_noul_prob(raw_entry: Any) -> float:
        probs = raw_entry[0][0]
        if len(probs) >= 2:
            return float(probs[1])
        return float(probs[0])

    @staticmethod
    def _resolve_stage1_labels(
        stage1_qdef: Mapping[str, Any],
        stage1_labels: Sequence[str] | None,
        k_dim: int,
    ) -> list[str]:
        if stage1_labels is not None:
            return list(stage1_labels)
        opts = stage1_qdef.get("options")
        if opts is not None:
            return [str(opt) for opt in opts]
        crit = stage1_qdef.get("criteria")
        if isinstance(crit, Mapping):
            return [str(k) for k in crit.keys()]
        if isinstance(crit, Sequence) and not isinstance(crit, (str, bytes)):
            return [str(c) for c in crit]
        return [str(i) for i in range(k_dim)]

    @staticmethod
    def _finalize_tier_multiclass(
        p_cal_row: Any,
        profile: CalibrationProfile | None,
        labels: Sequence[str],
        oos_label: str,
        align_oos_probs: bool,
    ) -> tuple[list[float], list[float], str, bool]:
        np = _np()
        p_cal = np.asarray(p_cal_row, dtype=np.float64).reshape(-1)
        oos_idx: int | None = None
        if profile is not None and profile.oos_index is not None:
            oos_idx = int(profile.oos_index)
        elif oos_label in labels:
            oos_idx = list(labels).index(oos_label)

        if profile is not None and profile.oos_threshold is not None and oos_idx is not None:
            tau_oos = float(profile.oos_threshold)
            is_oos_arr, pred_idx_arr = apply_oos_decision(p_cal, oos_index=oos_idx, oos_threshold=tau_oos)
            is_oos = bool(is_oos_arr[0])
            dom = labels[int(pred_idx_arr[0])]
            p_out = (
                align_oos_probabilities(p_cal, oos_index=oos_idx, oos_threshold=tau_oos)
                if align_oos_probs
                else p_cal
            )
        else:
            pred_idx = int(np.argmax(p_cal))
            dom = labels[pred_idx]
            is_oos = dom == oos_label
            p_out = p_cal

        return (
            [float(x) for x in p_out.tolist()],
            [float(x) for x in p_cal.tolist()],
            str(dom),
            bool(is_oos),
        )

    def score_binary_batch(
        self,
        states: Sequence[Any],
        qdef: Mapping[str, Any],
        *,
        on_truncate: str = "ignore",
    ) -> list[float]:
        """Score binary (`noul`) tasks on primary with optional Platt calibration (no Kev escalation)."""
        if not states:
            return []
        np = _np()
        encoded = [self.primary.encode_item(st, qdef, on_truncate=on_truncate) for st in states]
        raw_probs = np.asarray(self.primary.score_noul_batch(encoded), dtype=np.float64)
        if self.binary_profile is not None:
            cal = self.binary_profile.apply(raw_probs, from_probs=True)
            return [float(x) for x in cal.tolist()]
        return [float(x) for x in raw_probs.tolist()]

    def score_binary_cascade(
        self,
        states: Sequence[Any],
        qdef: Mapping[str, Any],
        *,
        uncertainty_margin: float = 0.0,
        tertiary_margin: float = 0.0,
        on_truncate: str = "ignore",
    ) -> dict[str, Any]:
        """Score binary (`noul`) tasks across primary -> escalation -> optional tertiary with per-tier Platt parity."""
        n_items = len(states)
        margin_esc = max(0.0, float(uncertainty_margin))
        margin_tert = max(0.0, float(tertiary_margin))
        if n_items == 0:
            return {
                "probs": [],
                "escalated_indices": [],
                "tertiary_escalated_indices": [],
                "tier_counts": {"primary": 0, "escalation": 0, "tertiary": 0},
            }

        np = _np()
        encoded = [self.primary.encode_item(st, qdef, on_truncate=on_truncate) for st in states]
        raw_probs = np.asarray(self.primary.score_noul_batch(encoded), dtype=np.float64)
        prim_cal = (
            np.asarray(self.binary_profile.apply(raw_probs, from_probs=True), dtype=np.float64)
            if self.binary_profile is not None
            else np.clip(raw_probs, 1e-12, 1.0 - 1e-12)
        )
        final_probs = [float(x) for x in prim_cal.tolist()]

        has_esc = self._escalation is not None or self._escalation_factory is not None
        has_tert = self._tertiary is not None or self._tertiary_factory is not None
        esc_idx: list[int] = []
        tert_idx: list[int] = []
        if has_esc and margin_esc > 0.0:
            esc_idx = [i for i in range(n_items) if abs(final_probs[i] - 0.5) < margin_esc]
        elif (not has_esc) and has_tert and margin_tert > 0.0:
            tert_idx = [i for i in range(n_items) if abs(final_probs[i] - 0.5) < margin_tert]

        if esc_idx:
            esc_backend = self._get_escalation()
            if esc_backend is None:
                raise RuntimeError("Escalation items requested without an escalation backend.")
            esc_raw = esc_backend.score_multi_states(
                [(states[i], [qdef]) for i in esc_idx],
                batch_docs=self.kev_batch_docs,
            )
            esc_p_raw = np.asarray([self._extract_noul_prob(r) for r in esc_raw], dtype=np.float64)
            esc_cal = (
                np.asarray(self.escalation_binary_profile.apply(esc_p_raw, from_probs=True), dtype=np.float64)
                if self.escalation_binary_profile is not None
                else np.clip(esc_p_raw, 1e-12, 1.0 - 1e-12)
            )
            for idx, p_val in zip(esc_idx, esc_cal.tolist()):
                final_probs[idx] = float(p_val)
                if has_tert and margin_tert > 0.0 and abs(float(p_val) - 0.5) < margin_tert:
                    tert_idx.append(idx)

        if tert_idx:
            tert_backend = self._get_tertiary()
            if tert_backend is None:
                raise RuntimeError("Tertiary escalation items requested without a tertiary backend.")
            tert_raw = tert_backend.score_multi_states(
                [(states[i], [qdef]) for i in tert_idx],
                batch_docs=self.tertiary_batch_docs,
            )
            tert_p_raw = np.asarray([self._extract_noul_prob(r) for r in tert_raw], dtype=np.float64)
            tert_cal = (
                np.asarray(self.tertiary_binary_profile.apply(tert_p_raw, from_probs=True), dtype=np.float64)
                if self.tertiary_binary_profile is not None
                else np.clip(tert_p_raw, 1e-12, 1.0 - 1e-12)
            )
            for idx, p_val in zip(tert_idx, tert_cal.tolist()):
                final_probs[idx] = float(p_val)

        esc_set = set(esc_idx) - set(tert_idx)
        tert_set = set(tert_idx)
        prim_cnt = n_items - len(esc_set) - len(tert_set)
        return {
            "probs": final_probs,
            "escalated_indices": esc_idx,
            "tertiary_escalated_indices": tert_idx,
            "tier_counts": {
                "primary": int(prim_cnt),
                "escalation": int(len(esc_set)),
                "tertiary": int(len(tert_set)),
            },
        }

    def score_multiclass_cascade(
        self,
        states: Sequence[Any],
        stage1_qdef: Mapping[str, Any],
        *,
        stage1_labels: Sequence[str] | None = None,
        stage2_qdefs: Mapping[str, Mapping[str, Any]] | None = None,
        oos_label: str = "out_of_scope",
        tau: float | None = None,
        tertiary_tau: float | None = None,
        align_oos_probs: bool = True,
        on_truncate: str = "ignore",
    ) -> dict[str, Any]:
        """Run Stage-1 on primary, calibrate, and escalate uncertain items (`< tau` -> Kev-0.8B -> `< tertiary_tau` -> Kev-4B)."""
        n_items = len(states)
        eff_tau = self.confidence_threshold if tau is None else float(tau)
        eff_tert_tau = self.tertiary_threshold if tertiary_tau is None else float(tertiary_tau)
        if n_items == 0:
            return {
                "stage1_probs": [],
                "stage1_probs_calibrated": [],
                "stage1_pred_domains": [],
                "oos_preds": [],
                "stage2_preds": [],
                "escalated_indices": [],
                "tertiary_escalated_indices": [],
                "tier_counts": {"primary": 0, "escalation": 0, "tertiary": 0},
                "esc_count": 0,
                "esc_rate": 0.0,
                "tertiary_count": 0,
                "tertiary_rate": 0.0,
                "tau": eff_tau,
                "tertiary_tau": eff_tert_tau,
            }

        np = _np()
        s1_encs = [self.primary.encode_item(st, stage1_qdef, on_truncate=on_truncate) for st in states]
        s1_raw = check_probability_validity(self.primary.score_choice_batch(s1_encs))
        if self.multiclass_profile is not None:
            s1_cal = self.multiclass_profile.apply(s1_raw, from_probs=True)
        else:
            s1_cal = s1_raw

        labels = self._resolve_stage1_labels(stage1_qdef, stage1_labels, s1_cal.shape[1])
        top1_vals = np.max(s1_cal, axis=1)

        has_escalation = self._escalation is not None or self._escalation_factory is not None
        has_tertiary = self._tertiary is not None or self._tertiary_factory is not None
        esc_idx: list[int] = []
        tert_idx: list[int] = []
        if has_escalation and eff_tau > 0.0:
            keep_idx = [i for i in range(n_items) if float(top1_vals[i]) >= eff_tau]
            esc_idx = [i for i in range(n_items) if float(top1_vals[i]) < eff_tau]
        elif (not has_escalation) and has_tertiary and eff_tert_tau > 0.0:
            keep_idx = [i for i in range(n_items) if float(top1_vals[i]) >= eff_tert_tau]
            tert_idx = [i for i in range(n_items) if float(top1_vals[i]) < eff_tert_tau]
        else:
            keep_idx = list(range(n_items))

        s1_final: list[list[float]] = [[] for _ in range(n_items)]
        s1_cal_final: list[list[float]] = [[] for _ in range(n_items)]
        s1_domains: list[str] = ["" for _ in range(n_items)]
        oos_preds: list[bool] = [False] * n_items
        s2_final: list[tuple[str, list[float]] | None] = [None] * n_items

        # Primary Stage 2 for confident items
        prim_s2_idxs: list[int] = []
        prim_s2_doms: list[str] = []
        prim_s2_encs: list[Any] = []
        for idx in keep_idx:
            p_out, p_cal_row, dom, is_oos = self._finalize_tier_multiclass(
                s1_cal[idx], self.multiclass_profile, labels, oos_label, align_oos_probs
            )
            s1_final[idx] = p_out
            s1_cal_final[idx] = p_cal_row
            s1_domains[idx] = dom
            oos_preds[idx] = is_oos
            if stage2_qdefs is not None and not is_oos and dom in stage2_qdefs:
                prim_s2_idxs.append(idx)
                prim_s2_doms.append(dom)
                prim_s2_encs.append(
                    self.primary.encode_item(states[idx], stage2_qdefs[dom], on_truncate=on_truncate)
                )
        if prim_s2_encs:
            p2_batch = self.primary.score_choice_batch(prim_s2_encs)
            for idx, dom, p2 in zip(prim_s2_idxs, prim_s2_doms, p2_batch):
                p2_valid = check_probability_validity(p2)
                s2_final[idx] = (dom, [float(x) for x in p2_valid.tolist()])

        # Tier-2 lazy escalation (e.g. Kev-0.8B) for low-confidence primary items
        esc_kept_count = 0
        if esc_idx:
            esc_backend = self._get_escalation()
            if esc_backend is None:
                raise RuntimeError("Escalation items requested without an escalation backend.")
            esc_s1_raw = esc_backend.score_multi_states(
                [(states[i], [stage1_qdef]) for i in esc_idx],
                batch_docs=self.kev_batch_docs,
            )
            esc_s1_mat = check_probability_validity([r[0][0] for r in esc_s1_raw])
            if self.escalation_multiclass_profile is not None:
                esc_s1_cal = self.escalation_multiclass_profile.apply(esc_s1_mat, from_probs=True)
            else:
                esc_s1_cal = esc_s1_mat

            esc_s2_idxs: list[int] = []
            esc_s2_doms: list[str] = []
            esc_s2_pairs: list[tuple[Any, list[Mapping[str, Any]]]] = []
            for row_pos, idx in enumerate(esc_idx):
                p_row = esc_s1_cal[row_pos]
                if has_tertiary and eff_tert_tau > 0.0 and float(np.max(p_row)) < eff_tert_tau:
                    tert_idx.append(idx)
                    continue
                esc_kept_count += 1
                p_out, p_cal_row, dom, is_oos = self._finalize_tier_multiclass(
                    p_row, self.escalation_multiclass_profile, labels, oos_label, align_oos_probs
                )
                s1_final[idx] = p_out
                s1_cal_final[idx] = p_cal_row
                s1_domains[idx] = dom
                oos_preds[idx] = is_oos
                if stage2_qdefs is not None and not is_oos and dom in stage2_qdefs:
                    esc_s2_idxs.append(idx)
                    esc_s2_doms.append(dom)
                    esc_s2_pairs.append((states[idx], [stage2_qdefs[dom]]))
            if esc_s2_pairs:
                esc_s2_raw = esc_backend.score_multi_states(
                    esc_s2_pairs, batch_docs=self.kev_batch_docs
                )
                for idx, dom, r2 in zip(esc_s2_idxs, esc_s2_doms, esc_s2_raw):
                    p2_valid = check_probability_validity(r2[0][0])
                    s2_final[idx] = (dom, [float(x) for x in p2_valid.tolist()])

        # Tier-3 optional lazy escalation (e.g. Kev-4B) for items still uncertain after Tier 2
        if tert_idx:
            tert_backend = self._get_tertiary()
            if tert_backend is None:
                raise RuntimeError("Tertiary escalation items requested without a tertiary backend.")
            tert_s1_raw = tert_backend.score_multi_states(
                [(states[i], [stage1_qdef]) for i in tert_idx],
                batch_docs=self.tertiary_batch_docs,
            )
            tert_s1_mat = check_probability_validity([r[0][0] for r in tert_s1_raw])
            if self.tertiary_multiclass_profile is not None:
                tert_s1_cal = self.tertiary_multiclass_profile.apply(tert_s1_mat, from_probs=True)
            else:
                tert_s1_cal = tert_s1_mat

            tert_s2_idxs: list[int] = []
            tert_s2_doms: list[str] = []
            tert_s2_pairs: list[tuple[Any, list[Mapping[str, Any]]]] = []
            for row_pos, idx in enumerate(tert_idx):
                p_out, p_cal_row, dom, is_oos = self._finalize_tier_multiclass(
                    tert_s1_cal[row_pos], self.tertiary_multiclass_profile, labels, oos_label, align_oos_probs
                )
                s1_final[idx] = p_out
                s1_cal_final[idx] = p_cal_row
                s1_domains[idx] = dom
                oos_preds[idx] = is_oos
                if stage2_qdefs is not None and not is_oos and dom in stage2_qdefs:
                    tert_s2_idxs.append(idx)
                    tert_s2_doms.append(dom)
                    tert_s2_pairs.append((states[idx], [stage2_qdefs[dom]]))
            if tert_s2_pairs:
                tert_s2_raw = tert_backend.score_multi_states(
                    tert_s2_pairs, batch_docs=self.tertiary_batch_docs
                )
                for idx, dom, r2 in zip(tert_s2_idxs, tert_s2_doms, tert_s2_raw):
                    p2_valid = check_probability_validity(r2[0][0])
                    s2_final[idx] = (dom, [float(x) for x in p2_valid.tolist()])

        return {
            "stage1_probs": s1_final,
            "stage1_probs_calibrated": s1_cal_final,
            "stage1_pred_domains": s1_domains,
            "oos_preds": oos_preds,
            "stage2_preds": s2_final,
            "escalated_indices": esc_idx,
            "tertiary_escalated_indices": tert_idx,
            "tier_counts": {
                "primary": len(keep_idx),
                "escalation": esc_kept_count,
                "tertiary": len(tert_idx),
            },
            "esc_count": len(esc_idx),
            "esc_rate": float(len(esc_idx)) / float(n_items),
            "tertiary_count": len(tert_idx),
            "tertiary_rate": float(len(tert_idx)) / float(n_items),
            "tau": eff_tau,
            "tertiary_tau": eff_tert_tau,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_exc: BaseException | None = None
        for obj, owned in (
            (self.primary, True),
            (self._escalation, self._owns_escalation),
            (self._tertiary, self._owns_tertiary),
        ):
            if owned and obj is not None and hasattr(obj, "close") and callable(obj.close):
                try:
                    obj.close()
                except BaseException as exc:
                    if first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc

    def __enter__(self) -> "DecisionCascadeScorer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "CalibrationProfile",
    "DecisionCascadeScorer",
    "align_oos_probabilities",
    "apply_multiclass_temperature",
    "apply_oos_decision",
    "apply_platt_scaling",
    "check_probability_validity",
    "fit_calibrated_profile",
    "fit_multiclass_temperature",
    "fit_oof_calibrated_scores",
    "fit_oos_threshold",
    "fit_platt_scaling",
    "stratified_kfold_indices",
]
