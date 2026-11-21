"""Domain-generic `0 ms` CPU Extraction Risk Scorer (`ExtractConf`).

Combines pre-mask token telemetry (`h_pre`, `neg_log_z_valid` / `lse`, `top2_margin`, mask fallback rate)
with JSON Schema structural diagnostics (null density, empty strings, empty containers, missing required
fields, length outlier score, and repaired/fallback status) to output calibrated defect risk in `[0, 1]`.

Supports both zero-config heuristic scoring (`compute_extractconf_risk`) and supervised calibration
via L2-regularized logistic regression, Pool-Adjacent-Violators (PAVA) isotonic calibration, or
combined `logistic_isotonic` fitting (`ExtractConfCalibrator`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

CalibrationMethod = Literal["logistic", "isotonic", "logistic_isotonic", "heuristic"]
VALID_CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = (
    "logistic",
    "isotonic",
    "logistic_isotonic",
    "heuristic",
)

EXTRACTCONF_FEATURE_NAMES: tuple[str, ...] = (
    "has_telemetry",
    "mean_h_pre",
    "max_h_pre",
    "mean_neg_log_z_valid",
    "max_neg_log_z_valid",
    "mean_top2_margin",
    "min_top2_margin",
    "mask_fallback_rate",
    "invalid_json",
    "invalid_schema",
    "truncated_length",
    "repaired_or_fallback",
    "null_density",
    "empty_string_density",
    "empty_container_density",
    "missing_required_rate",
    "length_outlier_score",
    "log1p_prompt_tokens",
)

_NEUTRAL_MARGIN: float = 4.0


def _sigmoid(z: float) -> float:
    clamped = max(-35.0, min(35.0, float(z)))
    return 1.0 / (1.0 + math.exp(-clamped))


def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _as_finite_floats(val: Any) -> list[float]:
    if val is None or isinstance(val, bool):
        return []
    if isinstance(val, (int, float, np.number)):
        f = float(val)
        return [f] if math.isfinite(f) else []
    if hasattr(val, "tolist"):
        raw = val.tolist()
        if isinstance(raw, (int, float)):
            return [float(raw)] if math.isfinite(float(raw)) else []
        return [float(x) for x in raw if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        out: list[float] = []
        for x in val:
            if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
                fx = float(x)
                if math.isfinite(fx):
                    out.append(fx)
        return out
    return []


@dataclass(frozen=True)
class ExtractionTelemetry:
    """Normalized pre-mask token uncertainty & grammar-conflict summary for one extracted item."""

    mean_h_pre: float = 0.0
    max_h_pre: float = 0.0
    mean_neg_log_z_valid: float = 0.0
    max_neg_log_z_valid: float = 0.0
    mean_top2_margin: float = _NEUTRAL_MARGIN
    min_top2_margin: float = _NEUTRAL_MARGIN
    mask_fallbacks: int = 0
    steps: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_h_pre": self.mean_h_pre,
            "max_h_pre": self.max_h_pre,
            "mean_neg_log_z_valid": self.mean_neg_log_z_valid,
            "max_neg_log_z_valid": self.max_neg_log_z_valid,
            "mean_top2_margin": self.mean_top2_margin,
            "min_top2_margin": self.min_top2_margin,
            "mask_fallbacks": self.mask_fallbacks,
            "steps": self.steps,
        }


def normalize_extraction_telemetry(telemetry: Any) -> tuple[bool, ExtractionTelemetry]:
    """Normalize an `ExtractionTelemetry`, dict, or step-wise arrays into `ExtractionTelemetry`."""
    if telemetry is None:
        return False, ExtractionTelemetry()
    if isinstance(telemetry, ExtractionTelemetry):
        return True, telemetry

    if hasattr(telemetry, "to_flat_features") and callable(telemetry.to_flat_features):
        telemetry = telemetry.to_flat_features()
    elif not isinstance(telemetry, Mapping) and hasattr(telemetry, "__dict__"):
        telemetry = vars(telemetry)

    if not isinstance(telemetry, Mapping):
        return False, ExtractionTelemetry()

    h_seq = _as_finite_floats(telemetry.get("h_pre"))
    mean_h_vals = _as_finite_floats(
        telemetry.get("mean_h_pre", telemetry.get("tel_mean_h_pre"))
    )
    max_h_vals = _as_finite_floats(
        telemetry.get("max_h_pre", telemetry.get("tel_max_h_pre"))
    )

    nlz_seq = _as_finite_floats(
        telemetry.get("neg_log_z_valid", telemetry.get("neg_log_z"))
    )
    if not nlz_seq:
        lse_all = _as_finite_floats(telemetry.get("lse_all", telemetry.get("lse")))
        lse_valid = _as_finite_floats(telemetry.get("lse_valid"))
        if lse_all and lse_valid and len(lse_all) == len(lse_valid):
            nlz_seq = [max(0.0, a - v) for a, v in zip(lse_all, lse_valid)]
    mean_nlz_vals = _as_finite_floats(
        telemetry.get("mean_neg_log_z_valid", telemetry.get("mean_neg_log_z", telemetry.get("tel_mean_neg_log_z")))
    )
    max_nlz_vals = _as_finite_floats(
        telemetry.get("max_neg_log_z_valid", telemetry.get("max_neg_log_z", telemetry.get("tel_max_neg_log_z")))
    )
    if not nlz_seq and not mean_nlz_vals and telemetry.get("min_z_valid") is not None:
        z_vals = _as_finite_floats(telemetry.get("min_z_valid"))
        if z_vals:
            nlz_from_z = max(0.0, -math.log(max(z_vals[0], 1e-12)))
            mean_nlz_vals = [nlz_from_z * 0.5]
            max_nlz_vals = [nlz_from_z]

    margin_seq = _as_finite_floats(telemetry.get("top2_margin"))
    mean_margin_vals = _as_finite_floats(telemetry.get("mean_top2_margin"))
    min_margin_vals = _as_finite_floats(telemetry.get("min_top2_margin"))

    has_any = bool(
        h_seq or mean_h_vals or max_h_vals or nlz_seq or mean_nlz_vals or max_nlz_vals or margin_seq or mean_margin_vals or min_margin_vals
    )
    if not has_any and telemetry.get("mask_fallbacks") is None:
        return False, ExtractionTelemetry()

    mean_h = max(0.0, mean_h_vals[0] if mean_h_vals else (sum(h_seq) / len(h_seq) if h_seq else 0.0))
    max_h = max(mean_h, max_h_vals[0] if max_h_vals else (max(h_seq) if h_seq else mean_h))

    mean_nlz = max(0.0, mean_nlz_vals[0] if mean_nlz_vals else (sum(nlz_seq) / len(nlz_seq) if nlz_seq else 0.0))
    max_nlz = max(mean_nlz, max_nlz_vals[0] if max_nlz_vals else (max(nlz_seq) if nlz_seq else mean_nlz))

    if mean_margin_vals:
        mean_m = max(0.0, mean_margin_vals[0])
    elif margin_seq:
        mean_m = max(0.0, sum(margin_seq) / len(margin_seq))
    else:
        mean_m = _NEUTRAL_MARGIN

    if min_margin_vals:
        min_m = max(0.0, min_margin_vals[0])
    elif margin_seq:
        min_m = max(0.0, min(margin_seq))
    else:
        min_m = mean_m

    steps_raw = telemetry.get("steps", telemetry.get("tel_steps", len(h_seq) or len(nlz_seq) or len(margin_seq) or 1))
    steps = max(1, int(steps_raw))
    fallbacks = max(0, int(telemetry.get("mask_fallbacks", telemetry.get("was_masked_count", 0)) or 0))
    return True, ExtractionTelemetry(
        mean_h_pre=mean_h,
        max_h_pre=max_h,
        mean_neg_log_z_valid=mean_nlz,
        max_neg_log_z_valid=max_nlz,
        mean_top2_margin=mean_m,
        min_top2_margin=min_m,
        mask_fallbacks=fallbacks,
        steps=steps,
    )


def _compute_structural_densities(
    parsed: Any,
    schema_obj: Mapping[str, Any] | None,
) -> tuple[float, float, float, float]:
    """Return `(null_density, empty_string_density, empty_container_density, missing_required_rate)`."""
    if not isinstance(parsed, Mapping):
        if parsed is None:
            return 1.0, 0.0, 0.0, 1.0
        return 0.0, 0.0, 0.0, 0.0

    required_keys: list[str] = []
    if isinstance(schema_obj, Mapping) and isinstance(schema_obj.get("required"), list):
        required_keys = [str(k) for k in schema_obj["required"]]
    missing_req = (
        sum(1 for k in required_keys if k not in parsed) / len(required_keys)
        if required_keys
        else 0.0
    )

    if isinstance(schema_obj, Mapping) and isinstance(schema_obj.get("properties"), Mapping):
        keys = list(schema_obj["properties"].keys())
    else:
        keys = list(parsed.keys())

    if not keys:
        return 0.0, 0.0, 0.0, missing_req

    n_null = 0
    n_str = 0
    n_empty_str = 0
    n_container = 0
    n_empty_container = 0

    for k in keys:
        val = parsed.get(k)
        if val is None:
            n_null += 1
        elif isinstance(val, str):
            n_str += 1
            if not val.strip():
                n_empty_str += 1
        elif isinstance(val, (list, tuple, Mapping)):
            n_container += 1
            if len(val) == 0:
                n_empty_container += 1

    null_density = n_null / len(keys)
    empty_str_density = (n_empty_str / n_str) if n_str > 0 else 0.0
    empty_cont_density = (n_empty_container / n_container) if n_container > 0 else 0.0
    return null_density, empty_str_density, empty_cont_density, missing_req


def _compute_length_outlier_score(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    max_tokens: int,
    response_text: str | None,
) -> float:
    score = 0.0
    eff_max = max(1, int(max_tokens))
    comp = max(0, int(completion_tokens))
    budget_ratio = comp / float(eff_max)
    if budget_ratio > 0.85:
        score = max(score, min(1.0, (budget_ratio - 0.85) / 0.15))
    if int(prompt_tokens) >= 16 and 0 < comp <= 2:
        score = max(score, 0.65)
    if isinstance(response_text, str) and len(response_text) >= 120:
        # Detect degenerate trailing character/ngram repetition
        tail = response_text[-120:]
        if len(set(tail)) <= 4:
            score = 1.0
    return float(max(0.0, min(1.0, score)))


def parse_json_schema_dict(
    json_schema: Mapping[str, Any] | str | None,
) -> Mapping[str, Any] | None:
    """Pre-parse a JSON Schema string into a `Mapping` once, or pass through an existing `Mapping`."""
    if isinstance(json_schema, Mapping):
        return json_schema
    if isinstance(json_schema, str) and json_schema.strip():
        try:
            loaded_s = json.loads(json_schema)
            if isinstance(loaded_s, Mapping):
                return loaded_s
        except Exception:
            return None
    return None


def extract_extractconf_features(
    *,
    parsed: Any = None,
    response_text: str | None = None,
    json_schema: Mapping[str, Any] | str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    max_tokens: int = 512,
    valid_json: bool = True,
    valid_schema: bool = True,
    finish_reason: str = "stop",
    repaired: bool = False,
    fallback_used: bool = False,
    telemetry: Any = None,
    _already_parsed: bool = False,
) -> list[float]:
    """Extract canonical 18-feature `ExtractConf` vector in ~0 ms pure Python."""
    eff_parsed = parsed
    eff_valid_json = bool(valid_json)
    if (
        not _already_parsed
        and eff_parsed is None
        and eff_valid_json
        and isinstance(response_text, str)
        and response_text.strip()
    ):
        try:
            eff_parsed = json.loads(response_text)
        except Exception:
            eff_valid_json = False

    schema_obj = parse_json_schema_dict(json_schema)

    has_tel, tel = normalize_extraction_telemetry(telemetry)
    denom_steps = max(1, tel.steps, int(completion_tokens))
    fallback_rate = min(1.0, float(tel.mask_fallbacks) / float(denom_steps))

    fr_norm = str(finish_reason or "stop").strip().lower()
    is_trunc = 1.0 if fr_norm == "length" else 0.0
    is_rep_fb = 1.0 if (repaired or fallback_used or fr_norm in ("repaired", "fallback")) else 0.0

    null_d, empty_str_d, empty_cont_d, miss_req = _compute_structural_densities(eff_parsed, schema_obj)
    len_outlier = _compute_length_outlier_score(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        max_tokens=max_tokens,
        response_text=response_text,
    )

    return [
        1.0 if has_tel else 0.0,
        float(tel.mean_h_pre),
        float(tel.max_h_pre),
        float(tel.mean_neg_log_z_valid),
        float(tel.max_neg_log_z_valid),
        float(tel.mean_top2_margin),
        float(tel.min_top2_margin),
        float(fallback_rate),
        0.0 if eff_valid_json else 1.0,
        0.0 if valid_schema else 1.0,
        is_trunc,
        is_rep_fb,
        float(null_d),
        float(empty_str_d),
        float(empty_cont_d),
        float(miss_req),
        float(len_outlier),
        math.log1p(max(0, int(prompt_tokens))),
    ]


def _heuristic_risk_from_features(feats: Sequence[float]) -> float:
    """Zero-config calibrated heuristic mapping `EXTRACTCONF_FEATURE_NAMES` -> `[0, 1]`."""
    (
        has_tel,
        mean_h,
        max_h,
        mean_nlz,
        max_nlz,
        mean_margin,
        min_margin,
        fallback_rate,
        inv_json,
        inv_schema,
        is_trunc,
        is_rep_fb,
        null_d,
        empty_str_d,
        empty_cont_d,
        miss_req,
        len_outlier,
        _log_prompt,
    ) = feats

    if inv_json > 0.5 or inv_schema > 0.5 or is_trunc > 0.5:
        hard = 0.88 + 0.05 * inv_json + 0.05 * inv_schema + 0.02 * is_trunc
        return float(min(1.0, max(0.85, hard)))

    z = -2.85
    if has_tel > 0.5:
        margin_deficit = max(0.0, 1.5 - min(1.5, float(mean_margin))) + 0.5 * max(0.0, 0.8 - min(0.8, float(min_margin)))
        z += (
            0.55 * min(4.0, float(mean_h))
            + 0.18 * min(6.0, float(max_h))
            + 0.45 * min(5.0, float(mean_nlz))
            + 0.16 * min(8.0, float(max_nlz))
            + 0.42 * margin_deficit
            + 1.60 * float(fallback_rate)
        )

    z += (
        1.45 * float(is_rep_fb)
        + 0.85 * float(null_d)
        + 1.15 * float(empty_str_d)
        + 0.55 * float(empty_cont_d)
        + 1.80 * float(miss_req)
        + 0.95 * float(len_outlier)
    )
    return float(max(0.0, min(1.0, _sigmoid(z))))


def _fit_pava_isotonic(scores: np.ndarray, targets: np.ndarray) -> tuple[list[float], list[float]]:
    """Pool-Adjacent-Violators Algorithm (PAVA) for monotone non-decreasing probability calibration."""
    order = np.argsort(scores, kind="mergesort")
    x_sorted = scores[order].astype(np.float64)
    y_sorted = np.clip(targets[order].astype(np.float64), 0.0, 1.0)

    # Blocks of [sum_y, count, x_min, x_max]
    blocks: list[list[float]] = []
    for xi, yi in zip(x_sorted.tolist(), y_sorted.tolist()):
        if blocks and abs(xi - blocks[-1][3]) <= 1e-12:
            blocks[-1][0] += yi
            blocks[-1][1] += 1.0
            blocks[-1][3] = xi
        else:
            blocks.append([yi, 1.0, xi, xi])
        while len(blocks) >= 2:
            mean_prev = blocks[-2][0] / blocks[-2][1]
            mean_curr = blocks[-1][0] / blocks[-1][1]
            if mean_prev <= mean_curr + 1e-15:
                break
            curr = blocks.pop()
            blocks[-1][0] += curr[0]
            blocks[-1][1] += curr[1]
            blocks[-1][3] = curr[3]

    knots_x: list[float] = []
    knots_y: list[float] = []
    for sum_y, cnt, x_lo, x_hi in blocks:
        val = max(0.0, min(1.0, sum_y / cnt))
        knots_x.append(0.5 * (x_lo + x_hi))
        knots_y.append(val)
    return knots_x, knots_y


def _predict_isotonic(scores: np.ndarray, knots_x: Sequence[float], knots_y: Sequence[float]) -> np.ndarray:
    if not knots_x:
        return np.clip(scores, 0.0, 1.0)
    if len(knots_x) == 1:
        return np.full_like(scores, float(knots_y[0]), dtype=np.float64)
    kx = np.asarray(knots_x, dtype=np.float64)
    ky = np.asarray(knots_y, dtype=np.float64)
    return np.clip(np.interp(scores, kx, ky, left=float(ky[0]), right=float(ky[-1])), 0.0, 1.0)


def _record_to_feature_vector(rec: Any) -> list[float]:
    if isinstance(rec, Mapping):
        if "features" in rec and isinstance(rec["features"], Sequence):
            vec = [float(x) for x in rec["features"]]
            if len(vec) != len(EXTRACTCONF_FEATURE_NAMES):
                raise ValueError(f"Expected {len(EXTRACTCONF_FEATURE_NAMES)} features, got {len(vec)}.")
            return vec
        return extract_extractconf_features(
            parsed=rec.get("parsed"),
            response_text=rec.get("response_text", rec.get("text")),
            json_schema=rec.get("json_schema"),
            prompt_tokens=int(rec.get("prompt_tokens", 0) or 0),
            completion_tokens=int(rec.get("completion_tokens", 0) or 0),
            max_tokens=int(rec.get("max_tokens", 512) or 512),
            valid_json=bool(rec.get("valid_json", True)),
            valid_schema=bool(rec.get("valid_schema", True)),
            finish_reason=str(rec.get("finish_reason", "stop")),
            repaired=bool(rec.get("repaired", False)),
            fallback_used=bool(rec.get("fallback_used", False)),
            telemetry=rec.get("telemetry"),
        )
    vec = [float(x) for x in rec]
    if len(vec) != len(EXTRACTCONF_FEATURE_NAMES):
        raise ValueError(f"Expected {len(EXTRACTCONF_FEATURE_NAMES)} features, got {len(vec)}.")
    return vec


class ExtractConfCalibrator:
    """Calibrated `ExtractConf` risk scorer supporting `logistic`, `isotonic`, and `logistic_isotonic`."""

    def __init__(
        self,
        *,
        method: CalibrationMethod = "logistic_isotonic",
        l2_reg: float = 1.0,
        risk_threshold: float = 0.50,
        mean: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        iso_knots_x: Sequence[float] | None = None,
        iso_knots_y: Sequence[float] | None = None,
        fitted: bool = False,
    ) -> None:
        if method not in VALID_CALIBRATION_METHODS:
            raise ValueError(f"Unsupported method={method!r}; choose from {VALID_CALIBRATION_METHODS}")
        if l2_reg <= 0.0 or not math.isfinite(float(l2_reg)):
            raise ValueError(f"l2_reg must be finite and positive, got {l2_reg!r}")
        if not (0.0 <= float(risk_threshold) <= 1.0):
            raise ValueError(f"risk_threshold must lie in [0, 1], got {risk_threshold!r}")

        self.method: CalibrationMethod = method
        self.l2_reg: float = float(l2_reg)
        self.risk_threshold: float = float(risk_threshold)
        self.mean: np.ndarray = (
            np.asarray(mean, dtype=np.float64)
            if mean is not None
            else np.zeros(len(EXTRACTCONF_FEATURE_NAMES), dtype=np.float64)
        )
        self.scale: np.ndarray = (
            np.asarray(scale, dtype=np.float64)
            if scale is not None
            else np.ones(len(EXTRACTCONF_FEATURE_NAMES), dtype=np.float64)
        )
        self.weights: np.ndarray = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.zeros(len(EXTRACTCONF_FEATURE_NAMES), dtype=np.float64)
        )
        self.bias: float = float(bias)
        self.iso_knots_x: tuple[float, ...] = tuple(float(x) for x in (iso_knots_x or ()))
        self.iso_knots_y: tuple[float, ...] = tuple(float(y) for y in (iso_knots_y or ()))
        self.fitted: bool = bool(fitted)

    def fit(
        self,
        records_or_features: Sequence[Any] | np.ndarray,
        labels: Sequence[int | float],
    ) -> "ExtractConfCalibrator":
        """Fit logistic weights and/or PAVA isotonic calibration curve on labeled examples."""
        if isinstance(records_or_features, np.ndarray) and records_or_features.ndim == 2:
            x_mat = np.asarray(records_or_features, dtype=np.float64)
            if x_mat.shape[1] != len(EXTRACTCONF_FEATURE_NAMES):
                raise ValueError(
                    f"Expected feature matrix width {len(EXTRACTCONF_FEATURE_NAMES)}, got {x_mat.shape[1]}"
                )
        else:
            rows = [_record_to_feature_vector(r) for r in records_or_features]
            if not rows:
                raise ValueError("Training records cannot be empty.")
            x_mat = np.asarray(rows, dtype=np.float64)

        y_arr = np.asarray([float(v) for v in labels], dtype=np.float64)
        if x_mat.shape[0] == 0 or x_mat.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Feature count ({x_mat.shape[0]}) must match label count ({y_arr.shape[0]}).")
        if np.any(~np.isfinite(y_arr)) or np.any((y_arr < 0.0) | (y_arr > 1.0)):
            raise ValueError("Labels must be finite values in [0, 1].")

        if self.method == "heuristic":
            self.fitted = True
            return self

        if self.method in ("logistic", "logistic_isotonic"):
            mu = x_mat.mean(axis=0)
            sd = x_mat.std(axis=0)
            sd = np.where(sd > 1e-8, sd, 1.0)
            z_mat = (x_mat - mu) / sd

            n_samples, n_dim = z_mat.shape
            # Augment with intercept column at index n_dim (unpenalized intercept)
            design = np.concatenate([z_mat, np.ones((n_samples, 1), dtype=np.float64)], axis=1)
            theta = np.zeros(n_dim + 1, dtype=np.float64)
            pos_rate = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
            theta[-1] = math.log(pos_rate / (1.0 - pos_rate))
            reg_diag = np.diag([self.l2_reg] * n_dim + [1e-6])

            for _ in range(30):
                logits = design @ theta
                probs = _sigmoid_vec(logits)
                w_irls = np.maximum(probs * (1.0 - probs), 1e-6)
                grad = design.T @ (probs - y_arr) + reg_diag @ theta
                hess = (design.T * w_irls) @ design + reg_diag
                try:
                    step = np.linalg.solve(hess, grad)
                except np.linalg.LinAlgError:
                    break
                theta -= step
                if float(np.max(np.abs(step))) < 1e-7:
                    break

            self.mean = mu
            self.scale = sd
            self.weights = theta[:n_dim]
            self.bias = float(theta[-1])

            if self.method == "logistic_isotonic":
                raw_logits = z_mat @ self.weights + self.bias
                kx, ky = _fit_pava_isotonic(raw_logits, y_arr)
                self.iso_knots_x = tuple(kx)
                self.iso_knots_y = tuple(ky)
            else:
                self.iso_knots_x = ()
                self.iso_knots_y = ()
        elif self.method == "isotonic":
            base_scores = np.asarray(
                [_heuristic_risk_from_features(row) for row in x_mat.tolist()],
                dtype=np.float64,
            )
            kx, ky = _fit_pava_isotonic(base_scores, y_arr)
            self.iso_knots_x = tuple(kx)
            self.iso_knots_y = tuple(ky)

        self.fitted = True
        return self

    def predict_risk(self, record_or_features: Any) -> float:
        """Predict calibrated defect risk in `[0, 1]` for one feature vector or record."""
        feats = _record_to_feature_vector(record_or_features)
        if not self.fitted or self.method == "heuristic":
            return _heuristic_risk_from_features(feats)

        x_vec = np.asarray(feats, dtype=np.float64)
        if self.method == "logistic":
            z_val = float(np.dot((x_vec - self.mean) / self.scale, self.weights) + self.bias)
            return float(_sigmoid(z_val))
        if self.method == "logistic_isotonic":
            z_val = float(np.dot((x_vec - self.mean) / self.scale, self.weights) + self.bias)
            return float(_predict_isotonic(np.asarray([z_val], dtype=np.float64), self.iso_knots_x, self.iso_knots_y)[0])
        base = _heuristic_risk_from_features(feats)
        return float(_predict_isotonic(np.asarray([base], dtype=np.float64), self.iso_knots_x, self.iso_knots_y)[0])

    def predict_risks(self, records_or_features: Sequence[Any] | np.ndarray) -> list[float]:
        """Predict calibrated defect risks in `[0, 1]` across a batch."""
        if isinstance(records_or_features, np.ndarray) and records_or_features.ndim == 2:
            return [self.predict_risk(row) for row in records_or_features.tolist()]
        return [self.predict_risk(r) for r in records_or_features]

    def score_item(
        self,
        item: Any,
        *,
        json_schema: Mapping[str, Any] | str | None = None,
        max_tokens: int = 512,
        telemetry: Any = None,
        repaired: bool | None = None,
        fallback_used: bool | None = None,
    ) -> dict[str, Any]:
        """Score an engine `GenerationItem` (or dict) and return risk + escalation flag."""
        eff_tel = telemetry if telemetry is not None else getattr(item, "telemetry", None)
        eff_rep = bool(getattr(item, "repaired", False)) if repaired is None else bool(repaired)
        eff_fb = bool(getattr(item, "fallback_used", False)) if fallback_used is None else bool(fallback_used)
        feats = extract_extractconf_features(
            parsed=getattr(item, "parsed", None),
            response_text=getattr(item, "text", None),
            json_schema=json_schema,
            prompt_tokens=int(getattr(item, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(item, "completion_tokens", 0) or 0),
            max_tokens=max_tokens,
            valid_json=bool(getattr(item, "valid_json", True)),
            valid_schema=bool(getattr(item, "valid_schema", True)),
            finish_reason=str(getattr(item, "finish_reason", "stop")),
            repaired=eff_rep,
            fallback_used=eff_fb,
            telemetry=eff_tel,
        )
        risk = self.predict_risk(feats)
        return {
            "risk": round(float(risk), 6),
            "escalate": bool(risk >= self.risk_threshold),
            "has_telemetry": bool(feats[0] > 0.5),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_names": list(EXTRACTCONF_FEATURE_NAMES),
            "method": self.method,
            "l2_reg": self.l2_reg,
            "risk_threshold": self.risk_threshold,
            "mean": [float(x) for x in self.mean.tolist()],
            "scale": [float(x) for x in self.scale.tolist()],
            "weights": [float(x) for x in self.weights.tolist()],
            "bias": self.bias,
            "iso_knots_x": list(self.iso_knots_x),
            "iso_knots_y": list(self.iso_knots_y),
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExtractConfCalibrator":
        feat_names = tuple(payload.get("feature_names") or EXTRACTCONF_FEATURE_NAMES)
        if feat_names != EXTRACTCONF_FEATURE_NAMES:
            raise ValueError(
                f"ExtractConf feature schema mismatch: expected {len(EXTRACTCONF_FEATURE_NAMES)} features."
            )
        return cls(
            method=payload.get("method", "logistic_isotonic"),
            l2_reg=float(payload.get("l2_reg", 1.0)),
            risk_threshold=float(payload.get("risk_threshold", 0.50)),
            mean=payload.get("mean"),
            scale=payload.get("scale"),
            weights=payload.get("weights"),
            bias=float(payload.get("bias", 0.0)),
            iso_knots_x=payload.get("iso_knots_x"),
            iso_knots_y=payload.get("iso_knots_y"),
            fitted=bool(payload.get("fitted", True)),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ExtractConfCalibrator":
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if not isinstance(payload, Mapping):
            raise ValueError("Expected JSON object for ExtractConfCalibrator.")
        return cls.from_dict(payload)

    def save_json(self, path: str | Path) -> Path:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return dst

    @classmethod
    def load_json(cls, path: str | Path) -> "ExtractConfCalibrator":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def compute_extractconf_risk(
    *,
    parsed: Any = None,
    response_text: str | None = None,
    json_schema: Mapping[str, Any] | str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    max_tokens: int = 512,
    valid_json: bool = True,
    valid_schema: bool = True,
    finish_reason: str = "stop",
    repaired: bool = False,
    fallback_used: bool = False,
    telemetry: Any = None,
    calibrator: ExtractConfCalibrator | None = None,
    _already_parsed: bool = False,
) -> float:
    """Compute `ExtractConf` extraction defect risk in `[0, 1]` at ~0 ms CPU overhead."""
    feats = extract_extractconf_features(
        parsed=parsed,
        response_text=response_text,
        json_schema=json_schema,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        max_tokens=max_tokens,
        valid_json=valid_json,
        valid_schema=valid_schema,
        finish_reason=finish_reason,
        repaired=repaired,
        fallback_used=fallback_used,
        telemetry=telemetry,
        _already_parsed=_already_parsed,
    )
    if calibrator is not None:
        return float(calibrator.predict_risk(feats))
    return _heuristic_risk_from_features(feats)


__all__ = [
    "CalibrationMethod",
    "EXTRACTCONF_FEATURE_NAMES",
    "ExtractConfCalibrator",
    "ExtractionTelemetry",
    "VALID_CALIBRATION_METHODS",
    "compute_extractconf_risk",
    "extract_extractconf_features",
    "normalize_extraction_telemetry",
    "parse_json_schema_dict",
]
