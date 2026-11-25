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
    if not bool(np.all(np.isfinite(raw))):
        raise ValueError("Platt scores contain NaN or Inf.")
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
    t_min: float = 0.5,
    t_max: float = 20.0,
    num_grid: int = 79,
    eps: float = 1e-30,
) -> float:
    """Fit NLL-optimal positive temperature scale over `[t_min, t_max]`."""
    np = _np()
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
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
    best_t = 1.0
    best_nll = float("inf")
    for cand_t in np.linspace(float(t_min), float(t_max), max(2, int(num_grid))):
        pr = apply_multiclass_temperature(logp, float(cand_t), from_probs=False, eps=eps)
        p_true = np.clip(pr[row_idx, labels], 1e-30, 1.0)
        nll = float(-np.mean(np.log(p_true)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(cand_t)
    return best_t


@dataclass(frozen=True)
class CalibrationProfile:
    """Opt-in calibration profile keyed by `(model_id, task_key)`."""

    model_id: str
    task_key: str
    method: str
    platt_a: float = 1.0
    platt_b: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in ("platt", "temperature"):
            raise ValueError(f"Unsupported calibration method {self.method!r}")
        if self.method == "temperature" and (self.temperature <= 0.0 or not math.isfinite(self.temperature)):
            raise ValueError(f"Expected positive finite temperature, got {self.temperature}")

    def apply(self, values: Any, *, from_probs: bool = True) -> np.ndarray:
        if self.method == "platt":
            return apply_platt_scaling(values, self.platt_a, self.platt_b, from_probs=from_probs)
        return apply_multiclass_temperature(values, self.temperature, from_probs=from_probs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_key": self.task_key,
            "method": self.method,
            "platt_a": float(self.platt_a),
            "platt_b": float(self.platt_b),
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationProfile":
        return cls(
            model_id=str(raw["model_id"]),
            task_key=str(raw["task_key"]),
            method=str(raw["method"]),
            platt_a=float(raw.get("platt_a", 1.0)),
            platt_b=float(raw.get("platt_b", 0.0)),
            temperature=float(raw.get("temperature", 1.0)),
        )


class DecisionCascadeScorer:
    """Task-aware fast-primary to Kev-escalation decision cascade with lazy secondary loading."""

    def __init__(
        self,
        primary: Any,
        escalation: Any | None = None,
        *,
        escalation_factory: Callable[[], Any] | None = None,
        binary_profile: CalibrationProfile | None = None,
        multiclass_profile: CalibrationProfile | None = None,
        confidence_threshold: float = 0.50,
        kev_batch_docs: int = 8,
    ) -> None:
        self.primary = primary
        self._escalation = escalation
        self._escalation_factory = escalation_factory
        self._owns_escalation = escalation is None and escalation_factory is not None
        self.binary_profile = binary_profile
        self.multiclass_profile = multiclass_profile
        self.confidence_threshold = float(confidence_threshold)
        self.kev_batch_docs = max(1, int(kev_batch_docs))

    def _get_escalation(self) -> Any | None:
        if self._escalation is None and self._escalation_factory is not None:
            self._escalation = self._escalation_factory()
        return self._escalation

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

    def score_multiclass_cascade(
        self,
        states: Sequence[Any],
        stage1_qdef: Mapping[str, Any],
        *,
        stage1_labels: Sequence[str] | None = None,
        stage2_qdefs: Mapping[str, Mapping[str, Any]] | None = None,
        oos_label: str = "out_of_scope",
        tau: float | None = None,
        on_truncate: str = "ignore",
    ) -> dict[str, Any]:
        """Run Stage-1 on primary, calibrate, keep confident items on primary, and escalate `< tau` to Kev."""
        n_items = len(states)
        eff_tau = self.confidence_threshold if tau is None else float(tau)
        if n_items == 0:
            return {
                "stage1_probs": [],
                "stage2_preds": [],
                "escalated_indices": [],
                "esc_count": 0,
                "esc_rate": 0.0,
                "tau": eff_tau,
            }

        np = _np()
        s1_encs = [self.primary.encode_item(st, stage1_qdef, on_truncate=on_truncate) for st in states]
        s1_raw = np.asarray(self.primary.score_choice_batch(s1_encs), dtype=np.float64)
        if self.multiclass_profile is not None:
            s1_cal = self.multiclass_profile.apply(s1_raw, from_probs=True)
        else:
            s1_cal = check_probability_validity(s1_raw)

        labels = (
            list(stage1_labels)
            if stage1_labels is not None
            else [str(opt) for opt in (stage1_qdef.get("options") or range(s1_cal.shape[1]))]
        )
        top1_vals = np.max(s1_cal, axis=1)

        has_escalation = self._escalation is not None or self._escalation_factory is not None
        if has_escalation and eff_tau > 0.0:
            keep_idx = [i for i in range(n_items) if float(top1_vals[i]) >= eff_tau]
            esc_idx = [i for i in range(n_items) if float(top1_vals[i]) < eff_tau]
        else:
            keep_idx = list(range(n_items))
            esc_idx = []

        s1_final: list[list[float]] = [[] for _ in range(n_items)]
        s2_final: list[tuple[str, list[float]] | None] = [None] * n_items

        # Primary Stage 2 for confident items
        prim_s2_idxs: list[int] = []
        prim_s2_doms: list[str] = []
        prim_s2_encs: list[Any] = []
        for idx in keep_idx:
            s1_final[idx] = [float(x) for x in s1_cal[idx].tolist()]
            dom = labels[int(np.argmax(s1_cal[idx]))]
            if stage2_qdefs is not None and dom != oos_label and dom in stage2_qdefs:
                prim_s2_idxs.append(idx)
                prim_s2_doms.append(dom)
                prim_s2_encs.append(
                    self.primary.encode_item(states[idx], stage2_qdefs[dom], on_truncate=on_truncate)
                )
        if prim_s2_encs:
            p2_batch = self.primary.score_choice_batch(prim_s2_encs)
            for idx, dom, p2 in zip(prim_s2_idxs, prim_s2_doms, p2_batch):
                s2_final[idx] = (dom, [float(x) for x in p2])

        # Lazy escalation to Kev for low-confidence items
        if esc_idx:
            esc_backend = self._get_escalation()
            if esc_backend is None:
                raise RuntimeError("Escalation items requested without an escalation backend.")
            esc_s1_raw = esc_backend.score_multi_states(
                [(states[i], [stage1_qdef]) for i in esc_idx],
                batch_docs=self.kev_batch_docs,
            )
            esc_s2_idxs: list[int] = []
            esc_s2_doms: list[str] = []
            esc_s2_pairs: list[tuple[Any, list[Mapping[str, Any]]]] = []
            for idx, r1 in zip(esc_idx, esc_s1_raw):
                p1 = [float(x) for x in r1[0][0]]
                s1_final[idx] = p1
                dom = labels[int(np.argmax(np.asarray(p1, dtype=np.float64)))]
                if stage2_qdefs is not None and dom != oos_label and dom in stage2_qdefs:
                    esc_s2_idxs.append(idx)
                    esc_s2_doms.append(dom)
                    esc_s2_pairs.append((states[idx], [stage2_qdefs[dom]]))
            if esc_s2_pairs:
                esc_s2_raw = esc_backend.score_multi_states(
                    esc_s2_pairs, batch_docs=self.kev_batch_docs
                )
                for idx, dom, r2 in zip(esc_s2_idxs, esc_s2_doms, esc_s2_raw):
                    s2_final[idx] = (dom, [float(x) for x in r2[0][0]])

        return {
            "stage1_probs": s1_final,
            "stage2_preds": s2_final,
            "escalated_indices": esc_idx,
            "esc_count": len(esc_idx),
            "esc_rate": float(len(esc_idx)) / float(n_items),
            "tau": eff_tau,
        }

    def close(self) -> None:
        if hasattr(self.primary, "close") and callable(self.primary.close):
            self.primary.close()
        if self._owns_escalation and self._escalation is not None and hasattr(self._escalation, "close"):
            self._escalation.close()

    def __enter__(self) -> "DecisionCascadeScorer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "CalibrationProfile",
    "DecisionCascadeScorer",
    "apply_multiclass_temperature",
    "apply_platt_scaling",
    "check_probability_validity",
    "fit_multiclass_temperature",
    "fit_platt_scaling",
]
