"""Pure-CPU unit tests for domain-generic `fast_shared.croq` and `fast_shared.extractconf`."""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pytest

from fast_shared.croq import (
    ConformalGrammarRouter,
    CroqDecision,
    CroqRouter,
    build_enum_locked_schema,
    compute_nonconformity_scores,
    split_conformal_quantile,
)
from fast_shared.extractconf import (
    EXTRACTCONF_FEATURE_NAMES,
    ExtractConfCalibrator,
    ExtractionTelemetry,
    compute_extractconf_risk,
    extract_extractconf_features,
)


def test_split_conformal_quantile_exact_rank_and_bounds_cpu() -> None:
    scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # n=9, alpha=0.10 -> ceil((9 + 1) * 0.90) = 9 -> 9th order statistic = 0.90
    assert split_conformal_quantile(scores, alpha=0.10) == pytest.approx(0.90)
    # n=19, alpha=0.05 -> ceil(20 * 0.95) = 19 -> max score
    s19 = [i / 20.0 for i in range(1, 20)]
    assert split_conformal_quantile(s19, alpha=0.05) == pytest.approx(19.0 / 20.0)
    # Small n where ceil((n + 1)(1 - alpha)) > n returns +inf to preserve coverage >= 1 - alpha
    assert math.isinf(split_conformal_quantile([0.1, 0.2, 0.3], alpha=0.05))

    with pytest.raises(ValueError, match="non-empty"):
        split_conformal_quantile([], alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_quantile([0.1, 0.2], alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        split_conformal_quantile([0.1, 0.2], alpha=1.0)
    with pytest.raises(ValueError, match="finite"):
        split_conformal_quantile([0.1, float("nan")], alpha=0.05)


@pytest.mark.parametrize("score_mode", ["aps_det", "lac"])
def test_croq_router_coverage_lock_modes_schema_and_json_roundtrip_cpu(
    tmp_path: Path,
    score_mode: str,
) -> None:
    assert CroqRouter is ConformalGrammarRouter
    rng = np.random.default_rng(2026)
    labels = [f"type_{i}" for i in range(8)]
    n_cal = 400
    n_test = 400

    def _sample_dataset(n: int) -> tuple[np.ndarray, list[str]]:
        y_idx = rng.integers(0, len(labels), size=n)
        logits = rng.normal(0.0, 1.0, size=(n, len(labels)))
        # Mix high-confidence, two-way ambiguous, and uniform rows
        regimes = rng.choice(["sharp", "pair", "flat"], size=n, p=[0.55, 0.30, 0.15])
        for i in range(n):
            yi = int(y_idx[i])
            if regimes[i] == "sharp":
                logits[i, yi] += 4.5
            elif regimes[i] == "pair":
                partner = (yi + 1) % len(labels)
                logits[i, yi] += 2.4
                logits[i, partner] += 2.3
            else:
                logits[i] *= 0.2
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs, [labels[int(i)] for i in y_idx]

    cal_probs, cal_y = _sample_dataset(n_cal)
    test_probs, test_y = _sample_dataset(n_test)

    scores_mat = compute_nonconformity_scores(cal_probs[:5], score_mode=score_mode)  # type: ignore[arg-type]
    assert scores_mat.shape == (5, len(labels))

    router = ConformalGrammarRouter(
        labels,
        field_name="record_kind",
        alpha=0.05,
        score_mode=score_mode,  # type: ignore[arg-type]
        max_lock_k=4,
    ).fit(cal_probs, cal_y)

    assert router.q_hat is not None
    assert router.n_cal == n_cal

    sets = router.predict_allowed_sets(test_probs)
    assert len(sets) == n_test
    covered = sum(1 for s, y in zip(sets, test_y) if y in s) / n_test
    # Exchangeable finite-sample conformal guarantee (allowing normal binomial slack around 0.95)
    assert covered >= 0.91

    # Explicit routing regime checks: singleton, compact_set (<=4), and open (>4 or empty)
    sharp_p = [0.995, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0005]
    dec_single = router.route_one(sharp_p)
    assert isinstance(dec_single, CroqDecision)
    assert dec_single.lock_type == "singleton"
    assert dec_single.lock_spec == "type_0"
    assert dec_single.allowed_set == ("type_0",)

    pair_p = [0.49, 0.49, 0.005, 0.005, 0.0025, 0.0025, 0.0025, 0.0025]
    dec_pair = router.route_one(pair_p)
    assert dec_pair.lock_type == "compact_set"
    assert dec_pair.lock_spec == "type_0|type_1"
    assert set(dec_pair.allowed_set) == {"type_0", "type_1"}

    flat_p = [1.0 / len(labels)] * len(labels)
    dec_open = router.route_one(flat_p)
    assert dec_open.lock_type == "open"
    assert dec_open.lock_spec is None

    # Schema locking narrows enum on singleton/compact_set and leaves open schema intact on open
    base_schema = {
        "type": "object",
        "properties": {
            "record_kind": {"type": "string", "enum": list(labels)},
            "summary": {"type": "string"},
        },
        "required": ["record_kind", "summary"],
    }
    locked_pair = router.lock_schema(base_schema, dec_pair)
    assert locked_pair["properties"]["record_kind"]["enum"] == ["type_0", "type_1"]
    # Base schema must not be mutated in place
    assert len(base_schema["properties"]["record_kind"]["enum"]) == len(labels)

    open_schema = router.lock_schema(base_schema, dec_open)
    assert open_schema["properties"]["record_kind"]["enum"] == list(labels)

    narrowed = build_enum_locked_schema(base_schema, "record_kind", ["type_3"])
    assert narrowed is not None
    assert narrowed["properties"]["record_kind"]["enum"] == ["type_3"]

    # JSON serialization roundtrip (dict, json string, and file)
    json_path = tmp_path / f"croq_{score_mode}.json"
    router.save_json(json_path)
    reloaded = ConformalGrammarRouter.load_json(json_path)
    assert reloaded.to_dict() == router.to_dict()
    assert reloaded.predict_allowed_sets(test_probs[:16]) == sets[:16]
    assert ConformalGrammarRouter.from_json(router.to_json()).q_hat == pytest.approx(router.q_hat)
    ev = reloaded.evaluate(test_probs[:50], test_y[:50])
    assert ev["n"] == 50 and 0.0 <= ev["empirical_coverage"] <= 1.0


def test_extractconf_zero_config_heuristic_and_telemetry_monotonicity_cpu() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "category": {"type": "string", "enum": ["alpha", "beta", "gamma"]},
            "author": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "category", "author", "tags"],
    }

    clean_parsed = {
        "title": "Quarterly Subsurface Pressure Telemetry Summary",
        "category": "alpha",
        "author": "Dr. Robin Mercer",
        "tags": ["pressure", "sensor"],
    }
    clean_tel = ExtractionTelemetry(
        mean_h_pre=0.04,
        max_h_pre=0.12,
        mean_neg_log_z_valid=0.002,
        max_neg_log_z_valid=0.01,
        mean_top2_margin=6.5,
        min_top2_margin=3.8,
        mask_fallbacks=0,
        steps=24,
    )
    r_clean = compute_extractconf_risk(
        parsed=clean_parsed,
        json_schema=schema,
        prompt_tokens=180,
        completion_tokens=24,
        max_tokens=256,
        valid_json=True,
        valid_schema=True,
        finish_reason="stop",
        telemetry=clean_tel,
    )
    assert 0.0 <= r_clean < 0.15

    # High pre-mask entropy + heavy grammar mask conflict (-log Z_valid via lse_all - lse_valid) + tiny margin
    conflict_tel = {
        "h_pre": [1.8, 2.4, 2.9],
        "lse_all": [10.0, 12.0, 11.5],
        "lse_valid": [7.5, 8.0, 7.0],  # neg_log_z_valid = [2.5, 4.0, 4.5]
        "top2_margin": [0.05, 0.02, 0.08],
        "mask_fallbacks": 2,
        "steps": 12,
    }
    r_conflict = compute_extractconf_risk(
        parsed=clean_parsed,
        json_schema=schema,
        prompt_tokens=180,
        completion_tokens=24,
        max_tokens=256,
        valid_json=True,
        valid_schema=True,
        finish_reason="stop",
        telemetry=conflict_tel,
    )
    assert r_clean < r_conflict <= 1.0
    assert r_conflict > 0.35

    # Structural defects: null density, empty strings, empty containers, repaired/fallback flag
    sparse_parsed = {
        "title": "   ",
        "category": "alpha",
        "author": None,
        "tags": [],
    }
    r_sparse = compute_extractconf_risk(
        parsed=sparse_parsed,
        json_schema=schema,
        prompt_tokens=400,
        completion_tokens=248,  # near max_tokens=256 outlier
        max_tokens=256,
        valid_json=True,
        valid_schema=True,
        finish_reason="stop",
        repaired=True,
    )
    assert r_clean < r_sparse <= 1.0
    assert r_sparse > 0.40

    # Hard failures (truncated length / invalid schema / invalid json) hit top risk tier
    r_trunc = compute_extractconf_risk(
        parsed=None,
        response_text='{"title": "cut',
        json_schema=schema,
        prompt_tokens=200,
        completion_tokens=256,
        max_tokens=256,
        valid_json=False,
        valid_schema=False,
        finish_reason="length",
    )
    assert 0.85 <= r_trunc <= 1.0

    feats = extract_extractconf_features(
        parsed=sparse_parsed,
        json_schema=schema,
        prompt_tokens=400,
        completion_tokens=248,
        max_tokens=256,
        repaired=True,
        telemetry=conflict_tel,
    )
    assert len(feats) == len(EXTRACTCONF_FEATURE_NAMES)
    assert all(math.isfinite(x) for x in feats)


@pytest.mark.parametrize("method", ["logistic", "isotonic", "logistic_isotonic"])
def test_extractconf_calibrator_fit_predict_and_serialization_cpu(
    tmp_path: Path,
    method: str,
) -> None:
    rng = np.random.default_rng(99)
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "note": {"type": ["string", "null"]},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "note", "items"],
    }

    records: list[dict[str, object]] = []
    labels: list[int] = []
    for _ in range(240):
        is_bad = int(rng.random() < 0.35)
        labels.append(is_bad)
        if is_bad:
            h_val = float(rng.uniform(1.4, 3.0))
            nlz_val = float(rng.uniform(1.2, 4.5))
            margin = float(rng.uniform(0.01, 0.35))
            parsed = {"name": "" if rng.random() < 0.5 else "x", "note": None, "items": []}
            repaired = bool(rng.random() < 0.4)
        else:
            h_val = float(rng.uniform(0.01, 0.35))
            nlz_val = float(rng.uniform(0.0, 0.15))
            margin = float(rng.uniform(2.5, 8.0))
            parsed = {"name": "Sensor Unit 42", "note": "Verified normal", "items": ["a", "b"]}
            repaired = False
        records.append(
            {
                "parsed": parsed,
                "json_schema": schema,
                "prompt_tokens": 150,
                "completion_tokens": 30,
                "max_tokens": 256,
                "valid_json": True,
                "valid_schema": True,
                "finish_reason": "stop",
                "repaired": repaired,
                "telemetry": {
                    "mean_h_pre": h_val,
                    "max_h_pre": h_val * 1.3,
                    "mean_neg_log_z_valid": nlz_val,
                    "max_neg_log_z_valid": nlz_val * 1.5,
                    "mean_top2_margin": margin,
                    "min_top2_margin": margin * 0.5,
                },
            }
        )

    cal = ExtractConfCalibrator(method=method, risk_threshold=0.5).fit(records[:160], labels[:160])  # type: ignore[arg-type]
    preds = cal.predict_risks(records[160:])
    assert len(preds) == 80
    assert all(0.0 <= p <= 1.0 for p in preds)

    pos_mean = float(np.mean([p for p, y in zip(preds, labels[160:]) if y == 1]))
    neg_mean = float(np.mean([p for p, y in zip(preds, labels[160:]) if y == 0]))
    assert pos_mean > neg_mean + 0.35

    # Roundtrip JSON persistence
    out_p = tmp_path / f"extractconf_{method}.json"
    cal.save_json(out_p)
    reloaded = ExtractConfCalibrator.load_json(out_p)
    assert np.allclose(reloaded.predict_risks(records[160:175]), preds[:15], atol=1e-6)
    assert np.isclose(
        compute_extractconf_risk(**records[160], calibrator=reloaded),  # type: ignore[arg-type]
        preds[0],
        atol=1e-6,
    )


def test_extractconf_preparsed_schema_dict_parity_and_single_json_loads_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    import fast_shared.extractconf as ec
    import fast_shared.pipeline_llm as pllm

    schema_dict = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "tag": {"type": ["string", "null"]},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "tag", "items"],
    }
    schema_str = json.dumps(schema_dict)
    assert ec.parse_json_schema_dict(schema_str) == schema_dict
    assert ec.parse_json_schema_dict(schema_dict) is schema_dict
    assert ec.parse_json_schema_dict("not-valid-json") is None

    sample_text = json.dumps({"title": "Telemetry Node", "tag": None, "items": []})
    f_str = extract_extractconf_features(response_text=sample_text, json_schema=schema_str, prompt_tokens=64, completion_tokens=16)
    f_dict = extract_extractconf_features(
        parsed=json.loads(sample_text),
        response_text=sample_text,
        json_schema=schema_dict,
        prompt_tokens=64,
        completion_tokens=16,
    )
    assert f_str == f_dict
    assert compute_extractconf_risk(response_text=sample_text, json_schema=schema_str, prompt_tokens=64, completion_tokens=16) == pytest.approx(
        compute_extractconf_risk(
            parsed=json.loads(sample_text),
            response_text=sample_text,
            json_schema=schema_dict,
            prompt_tokens=64,
            completion_tokens=16,
        )
    )

    # Verify extract_parquet parses json_schema string ONCE per call and parses each item.text ONCE when item.parsed is None!
    loads_inputs: list[str] = []
    real_loads = json.loads

    def _spy_loads(s: str | bytes, *args, **kwargs):
        if isinstance(s, str):
            loads_inputs.append(s)
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(ec.json, "loads", _spy_loads)
    monkeypatch.setattr(pllm.json, "loads", _spy_loads)

    class _UnparsedItemEngine:
        default_max_b_slots = 16
        seen_telemetry_flags: list[bool] = []

        def generate_batch(self, prompts: list[str], *, collect_telemetry: bool = False, **_kwargs):
            self.seen_telemetry_flags.append(bool(collect_telemetry))
            items = [
                SimpleNamespace(
                    text='{"title": "' + p + '", "tag": null, "items": ["ok"]}',
                    parsed=None,
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    prompt_tokens=20,
                    completion_tokens=10,
                    telemetry={"mean_h_pre": 0.1, "mean_neg_log_z_valid": 0.01, "mean_top2_margin": 4.5},
                )
                for p in prompts
            ]
            return SimpleNamespace(items=items, stats={"decode_sec": 0.01})

    in_pq = tmp_path / "single_parse_in.parquet"
    out_pq = tmp_path / "single_parse_out.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": i, "body_text": f"doc_{i}"} for i in range(6)]),
        in_pq,
    )
    loads_inputs.clear()
    eng_spy = _UnparsedItemEngine()
    eng_spy.seen_telemetry_flags = []
    pllm.extract_parquet(
        eng_spy,
        in_pq,
        out_pq,
        system_prompt="sys",
        json_schema=schema_str,
        emit_extractconf=True,
    )
    # Schema string parsed exactly once across all 6 rows, each unique item text parsed once, and collect_telemetry=True auto-forwarded!
    assert eng_spy.seen_telemetry_flags == [True]
    assert loads_inputs.count(schema_str) == 1
    assert len(loads_inputs) == 1 + 6

