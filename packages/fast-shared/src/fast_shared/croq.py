"""Domain-generic Conformal Risk-Controlling Per-Document Grammar Lock Router (`CROQ`).

Provides distribution-free split-conformal prediction sets (`aps_det`, `lac`, `raps_det`)
that narrow per-document JSON Schema enum fields to calibrated `singleton` or `compact_set`
(`<= max_lock_k`) grammars when confidence warrants, falling back cleanly to the open schema
whenever the conformal set is wider than `max_lock_k` (or empty under `lac`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

ScoreMode = Literal["aps_det", "lac", "raps_det"]
LockType = Literal["singleton", "compact_set", "open"]
VALID_SCORE_MODES: tuple[ScoreMode, ...] = ("aps_det", "lac", "raps_det")


def split_conformal_quantile(scores: Sequence[float] | Any, alpha: float = 0.05) -> float:
    """Return exact finite-sample split-conformal quantile `ceil((n + 1)(1 - alpha))`-th order statistic.

    Uses exact `Decimal` arithmetic so binary floating-point rounding never under-selects the rank.
    Returns `+inf` when `ceil((n + 1)(1 - alpha)) > n` to preserve finite-sample coverage `>= 1 - alpha`.
    """
    if hasattr(scores, "ravel"):
        flat = [float(x) for x in scores.ravel().tolist()]
    else:
        flat = [float(x) for x in scores]
    n = len(flat)
    if n == 0:
        raise ValueError("Calibration score sequence must be non-empty.")
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}.")
    if any(not math.isfinite(x) for x in flat):
        raise ValueError("Calibration scores must all be finite.")

    target = Decimal(n + 1) * (Decimal("1") - Decimal(str(alpha)))
    rank = int(math.ceil(target))
    if rank <= 0:
        return float("-inf")
    if rank > n:
        return float("inf")
    flat.sort()
    return float(flat[rank - 1])


def _normalize_prob_row(row: Sequence[float] | Any, n_classes: int | None = None) -> list[float]:
    vals = [float(x) for x in (row.tolist() if hasattr(row, "tolist") else row)]
    k = n_classes if n_classes is not None else len(vals)
    if len(vals) != k:
        raise ValueError(f"Expected probability vector of length {k}, got {len(vals)}.")
    if k == 0:
        raise ValueError("Probability vector must be non-empty.")
    clean = [v if (math.isfinite(v) and v > 0.0) else 0.0 for v in vals]
    s = sum(clean)
    if s > 0.0:
        return [v / s for v in clean]
    return [1.0 / k] * k


def _row_nonconformity_scores(
    probs_row: list[float],
    *,
    score_mode: ScoreMode = "aps_det",
    k_reg: int = 4,
    lam: float = 0.05,
) -> tuple[list[float], list[int]]:
    """Return `(scores_by_class_idx, order_desc_indices)` for one normalized probability vector."""
    if score_mode not in VALID_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; choose from {VALID_SCORE_MODES}")
    n_c = len(probs_row)
    order = sorted(range(n_c), key=lambda idx: (-probs_row[idx], idx))
    scores = [0.0] * n_c
    cprev = 0.0
    for rank_0, cls_idx in enumerate(order):
        p = probs_row[cls_idx]
        if score_mode == "lac":
            s_val = 1.0 - p
        elif score_mode == "aps_det":
            s_val = cprev
        else:
            s_val = cprev + float(lam) * max(0.0, float(rank_0 + 1 - int(k_reg)))
        scores[cls_idx] = s_val
        cprev += p
    return scores, order


def compute_nonconformity_scores(
    probs: Sequence[Sequence[float]] | Any,
    *,
    score_mode: ScoreMode = "aps_det",
    k_reg: int = 4,
    lam: float = 0.05,
) -> np.ndarray:
    """Compute per-class nonconformity score matrix `(N, K)` for `probs`."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"Expected 2D non-empty probability matrix, got shape {arr.shape!r}.")
    out = np.empty_like(arr, dtype=np.float64)
    n_c = int(arr.shape[1])
    for i in range(int(arr.shape[0])):
        norm_r = _normalize_prob_row(arr[i], n_c)
        row_s, _ = _row_nonconformity_scores(norm_r, score_mode=score_mode, k_reg=k_reg, lam=lam)
        out[i] = row_s
    return out


def build_enum_locked_schema(
    base_schema: Mapping[str, Any] | str,
    field_name: str,
    allowed_values: Sequence[str],
) -> dict[str, Any] | None:
    """Return a deep-copied JSON Schema restricting `field_name` (or dotted path) to `allowed_values`."""
    deduped = list(dict.fromkeys(str(v) for v in allowed_values if str(v)))
    if not deduped:
        return None

    schema_obj: dict[str, Any] = (
        json.loads(base_schema) if isinstance(base_schema, str) else copy.deepcopy(dict(base_schema))
    )
    parts = [p for p in str(field_name).split(".") if p]
    if not parts:
        raise ValueError("`field_name` must be a non-empty property name.")

    cursor: dict[str, Any] = schema_obj
    for depth, part in enumerate(parts):
        props = cursor.get("properties")
        if not isinstance(props, dict) or part not in props or not isinstance(props[part], dict):
            raise KeyError(f"Schema property {'.'.join(parts[: depth + 1])!r} not found in base_schema.")
        cursor = props[part]

    existing_enum = cursor.get("enum")
    if isinstance(existing_enum, list):
        valid_set = {str(x) for x in existing_enum}
        unknown = [v for v in deduped if v not in valid_set]
        if unknown:
            raise ValueError(f"Allowed values {unknown!r} are not members of schema enum for {field_name!r}.")
    cursor["enum"] = deduped
    return schema_obj


@dataclass(frozen=True)
class CroqDecision:
    """Per-document conformal grammar routing decision."""

    allowed_set: tuple[str, ...]
    set_size: int
    lock_type: LockType
    lock_spec: str | None
    top1_label: str
    top1_prob: float
    set_prob_mass: float
    q_hat: float

    @property
    def conformal_set(self) -> list[str]:
        return list(self.allowed_set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_set": list(self.allowed_set),
            "set_size": self.set_size,
            "lock_type": self.lock_type,
            "lock_spec": self.lock_spec,
            "top1_label": self.top1_label,
            "top1_prob": self.top1_prob,
            "set_prob_mass": self.set_prob_mass,
            "q_hat": self.q_hat,
        }


class ConformalGrammarRouter:
    """Distribution-free conformal risk-controlling grammar router (`CROQ`)."""

    def __init__(
        self,
        labels: Sequence[str],
        *,
        field_name: str = "category",
        alpha: float = 0.05,
        score_mode: ScoreMode = "aps_det",
        max_lock_k: int = 4,
        k_reg: int = 4,
        lam: float = 0.05,
        allow_empty: bool = True,
        q_hat: float | None = None,
        n_cal: int = 0,
        cal_summary: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}.")
        if score_mode not in VALID_SCORE_MODES:
            raise ValueError(f"Unsupported score_mode={score_mode!r}; choose from {VALID_SCORE_MODES}")
        if isinstance(max_lock_k, bool) or int(max_lock_k) < 1:
            raise ValueError(f"max_lock_k must be >= 1, got {max_lock_k!r}.")
        clean_labels = tuple(dict.fromkeys(str(lbl) for lbl in labels if str(lbl)))
        if not clean_labels:
            raise ValueError("`labels` must contain at least one non-empty label.")

        self.labels: tuple[str, ...] = clean_labels
        self.field_name: str = str(field_name)
        self.alpha: float = float(alpha)
        self.score_mode: ScoreMode = score_mode
        self.max_lock_k: int = int(max_lock_k)
        self.k_reg: int = int(k_reg)
        self.lam: float = float(lam)
        self.allow_empty: bool = bool(allow_empty)
        self._label_to_idx: dict[str, int] = {lbl: idx for idx, lbl in enumerate(self.labels)}
        self.q_hat: float | None = float(q_hat) if q_hat is not None else None
        self.n_cal: int = int(n_cal)
        self.cal_summary: dict[str, Any] = dict(cal_summary or {})

    def _resolve_label_indices(self, labels: Sequence[int | str]) -> list[int]:
        out: list[int] = []
        n_c = len(self.labels)
        for y in labels:
            if isinstance(y, str):
                if y not in self._label_to_idx:
                    raise KeyError(f"Unknown calibration label {y!r}")
                out.append(self._label_to_idx[y])
            else:
                if isinstance(y, bool):
                    raise TypeError(f"Boolean is not a valid class label: {y!r}")
                yi = int(y)
                if not (0 <= yi < n_c):
                    raise IndexError(f"Calibration label index {yi} out of bounds [0, {n_c}).")
                out.append(yi)
        return out

    def fit(
        self,
        cal_probs: Sequence[Sequence[float]] | Any,
        cal_labels: Sequence[int | str],
    ) -> "ConformalGrammarRouter":
        """Calibrate split-conformal threshold `q_hat` on held-out `(cal_probs, cal_labels)`."""
        rows = cal_probs.tolist() if hasattr(cal_probs, "tolist") else list(cal_probs)
        y_idx = self._resolve_label_indices(cal_labels)
        if len(rows) != len(y_idx):
            raise ValueError(f"cal_probs row count ({len(rows)}) != cal_labels count ({len(y_idx)}).")
        if not rows:
            raise ValueError("Calibration dataset cannot be empty.")

        cal_scores: list[float] = []
        for row, yi in zip(rows, y_idx):
            norm_r = _normalize_prob_row(row, len(self.labels))
            row_s, _ = _row_nonconformity_scores(
                norm_r,
                score_mode=self.score_mode,
                k_reg=self.k_reg,
                lam=self.lam,
            )
            cal_scores.append(row_s[yi])

        self.q_hat = split_conformal_quantile(cal_scores, alpha=self.alpha)
        self.n_cal = len(rows)
        self.cal_summary = self.evaluate(rows, y_idx)
        return self

    def predict_allowed_set(self, probs_1d: Sequence[float] | Any) -> list[str]:
        """Predict ordered conformal label set `C(x)` (descending probability) for one input."""
        if self.q_hat is None:
            raise RuntimeError("ConformalGrammarRouter must be fit or loaded with `q_hat` before prediction.")
        norm_r = _normalize_prob_row(probs_1d, len(self.labels))
        row_s, order = _row_nonconformity_scores(
            norm_r,
            score_mode=self.score_mode,
            k_reg=self.k_reg,
            lam=self.lam,
        )
        allowed = [self.labels[idx] for idx in order if row_s[idx] <= self.q_hat + 1e-12]
        if not allowed and not self.allow_empty:
            allowed = [self.labels[order[0]]]
        return allowed

    def predict_allowed_sets(self, probs_2d: Sequence[Sequence[float]] | Any) -> list[list[str]]:
        """Predict ordered conformal label sets `C(x_i)` across a batch of probability vectors."""
        rows = probs_2d.tolist() if hasattr(probs_2d, "tolist") else list(probs_2d)
        return [self.predict_allowed_set(row) for row in rows]

    def route_one(self, probs_1d: Sequence[float] | Any) -> CroqDecision:
        """Route one probability vector to `singleton`, `compact_set` (`<= max_lock_k`), or `open`."""
        if self.q_hat is None:
            raise RuntimeError("ConformalGrammarRouter must be fit or loaded with `q_hat` before routing.")
        norm_r = _normalize_prob_row(probs_1d, len(self.labels))
        allowed = self.predict_allowed_set(norm_r)
        top1_idx = max(range(len(self.labels)), key=lambda i: (norm_r[i], -i))
        top1_label = self.labels[top1_idx]
        top1_prob = float(norm_r[top1_idx])
        set_mass = float(sum(norm_r[self._label_to_idx[lbl]] for lbl in allowed))
        n_set = len(allowed)

        if n_set == 1:
            lock_type: LockType = "singleton"
            lock_spec: str | None = allowed[0]
        elif 1 < n_set <= self.max_lock_k:
            lock_type = "compact_set"
            lock_spec = "|".join(sorted(allowed))
        else:
            lock_type = "open"
            lock_spec = None

        return CroqDecision(
            allowed_set=tuple(allowed),
            set_size=n_set,
            lock_type=lock_type,
            lock_spec=lock_spec,
            top1_label=top1_label,
            top1_prob=round(top1_prob, 6),
            set_prob_mass=round(set_mass, 6),
            q_hat=float(self.q_hat),
        )

    def route_batch(self, probs_2d: Sequence[Sequence[float]] | Any) -> list[CroqDecision]:
        """Route a 2D probability matrix into per-document `CroqDecision` records."""
        rows = probs_2d.tolist() if hasattr(probs_2d, "tolist") else list(probs_2d)
        return [self.route_one(row) for row in rows]

    def lock_schema(
        self,
        base_schema: Mapping[str, Any] | str,
        probs_or_decision: CroqDecision | Sequence[float] | Any,
        *,
        field_name: str | None = None,
    ) -> dict[str, Any]:
        """Return enum-locked JSON Schema when `1 <= |C(x)| <= max_lock_k`, else open `base_schema`."""
        decision = (
            probs_or_decision
            if isinstance(probs_or_decision, CroqDecision)
            else self.route_one(probs_or_decision)
        )
        base_dict: dict[str, Any] = (
            json.loads(base_schema) if isinstance(base_schema, str) else copy.deepcopy(dict(base_schema))
        )
        if decision.lock_spec is None:
            return base_dict
        eff_field = self.field_name if field_name is None else str(field_name)
        locked = build_enum_locked_schema(base_dict, eff_field, sorted(decision.allowed_set))
        return locked if locked is not None else base_dict

    def evaluate(
        self,
        probs_2d: Sequence[Sequence[float]] | Any,
        labels: Sequence[int | str],
    ) -> dict[str, Any]:
        """Evaluate empirical coverage, mean set size, and lock-regime rates."""
        rows = probs_2d.tolist() if hasattr(probs_2d, "tolist") else list(probs_2d)
        y_idx = self._resolve_label_indices(labels)
        if len(rows) != len(y_idx):
            raise ValueError(f"probs_2d rows ({len(rows)}) != labels ({len(y_idx)}).")
        if not rows:
            return {
                "n": 0,
                "empirical_coverage": 0.0,
                "target_coverage": round(1.0 - self.alpha, 6),
                "mean_set_size": 0.0,
                "singleton_rate": 0.0,
                "compact_rate": 0.0,
                "open_fallback_rate": 0.0,
            }

        covered = 0
        sizes: list[int] = []
        singletons = 0
        compacts = 0
        opens = 0
        for row, yi in zip(rows, y_idx):
            dec = self.route_one(row)
            if self.labels[yi] in dec.allowed_set:
                covered += 1
            sizes.append(dec.set_size)
            if dec.lock_type == "singleton":
                singletons += 1
            elif dec.lock_type == "compact_set":
                compacts += 1
            else:
                opens += 1

        n = len(rows)
        return {
            "n": n,
            "empirical_coverage": round(covered / n, 6),
            "target_coverage": round(1.0 - self.alpha, 6),
            "mean_set_size": round(sum(sizes) / n, 4),
            "singleton_rate": round(singletons / n, 6),
            "compact_rate": round((singletons + compacts) / n, 6),
            "open_fallback_rate": round(opens / n, 6),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "labels": list(self.labels),
            "field_name": self.field_name,
            "alpha": self.alpha,
            "score_mode": self.score_mode,
            "max_lock_k": self.max_lock_k,
            "k_reg": self.k_reg,
            "lam": self.lam,
            "allow_empty": self.allow_empty,
            "q_hat": self.q_hat,
            "n_cal": self.n_cal,
            "cal_summary": dict(self.cal_summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformalGrammarRouter":
        q_raw = payload.get("q_hat")
        return cls(
            labels=tuple(payload["labels"]),
            field_name=str(payload.get("field_name", "category")),
            alpha=float(payload.get("alpha", 0.05)),
            score_mode=payload.get("score_mode", "aps_det"),
            max_lock_k=int(payload.get("max_lock_k", 4)),
            k_reg=int(payload.get("k_reg", 4)),
            lam=float(payload.get("lam", 0.05)),
            allow_empty=bool(payload.get("allow_empty", True)),
            q_hat=float(q_raw) if q_raw is not None else None,
            n_cal=int(payload.get("n_cal", 0)),
            cal_summary=dict(payload.get("cal_summary") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ConformalGrammarRouter":
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if not isinstance(payload, Mapping):
            raise ValueError("Expected JSON object for ConformalGrammarRouter.")
        return cls.from_dict(payload)

    def save_json(self, path: str | Path) -> Path:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return dst

    @classmethod
    def load_json(cls, path: str | Path) -> "ConformalGrammarRouter":
        src = Path(path)
        return cls.from_json(src.read_text(encoding="utf-8"))


CroqRouter = ConformalGrammarRouter

__all__ = [
    "ConformalGrammarRouter",
    "CroqDecision",
    "CroqRouter",
    "LockType",
    "ScoreMode",
    "VALID_SCORE_MODES",
    "build_enum_locked_schema",
    "compute_nonconformity_scores",
    "split_conformal_quantile",
]
