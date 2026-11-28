"""Offline pure-CPU unit tests for Prediction-Powered Inference (PPI++) metrics."""

from __future__ import annotations

import json
import math
import statistics

import numpy as np
import pytest

from fast_shared.outcome_eval import PPIComparisonResult, compute_ppi_comparison
from fast_shared.outcome_eval.metrics import _normal_ppf


def test_normal_ppf_acklam_accuracy_and_symmetry() -> None:
    assert _normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-8)
    assert _normal_ppf(0.995) == pytest.approx(2.5758293035489004, abs=1e-8)
    assert _normal_ppf(0.5) == pytest.approx(0.0, abs=1e-12)

    nd = statistics.NormalDist()
    for p in (1e-8, 1e-4, 0.01, 0.02424, 0.02425, 0.025, 0.1, 0.25, 0.49):
        low = _normal_ppf(p)
        high = _normal_ppf(1.0 - p)
        assert low == pytest.approx(-high, abs=1e-8)
        assert low == pytest.approx(nd.inv_cdf(p), abs=1e-8)
        assert high == pytest.approx(nd.inv_cdf(1.0 - p), abs=1e-8)

    for bad_p in (0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _normal_ppf(bad_p)


def test_ppi_unbiased_accurate_judge() -> None:
    rng = np.random.default_rng(101)
    true_theta = 0.65
    n_labeled, n_unlabeled = 100, 2500

    y_all = np.clip(rng.normal(true_theta, 0.20, size=n_labeled + n_unlabeled), 0.0, 1.0)
    f_all = y_all + rng.normal(0.0, 0.03, size=n_labeled + n_unlabeled)
    y_l, f_l, f_u = y_all[:n_labeled], f_all[:n_labeled], f_all[n_labeled:]

    res = compute_ppi_comparison(f_u, f_l, y_l)
    assert isinstance(res, PPIComparisonResult)
    assert res.lambda_hat > 0.85
    assert res.n_effective_multiplier > 1.5
    assert not res.judge_misaligned
    assert res.se_ppi < res.se_classical
    assert res.ci_low < true_theta < res.ci_high
    assert abs(res.judge_bias_estimate) < 0.05
    assert res.n_labeled == n_labeled
    assert res.n_unlabeled == n_unlabeled
    assert res.ci_method == "asymptotic"

    as_dict = res.to_dict()
    assert set(as_dict) == {
        "theta_ppi",
        "theta_classical",
        "theta_judge_unlabeled",
        "theta_judge_labeled",
        "se_ppi",
        "se_classical",
        "ci_low",
        "ci_high",
        "classical_ci_low",
        "classical_ci_high",
        "lambda_hat",
        "alpha",
        "n_labeled",
        "n_unlabeled",
        "n_effective_multiplier",
        "judge_bias_estimate",
        "judge_misaligned",
        "ci_method",
    }
    assert json.loads(json.dumps(as_dict))["theta_ppi"] == pytest.approx(res.theta_ppi)


def test_ppi_systematically_biased_correlated_judge() -> None:
    rng = np.random.default_rng(202)
    true_theta = 0.42
    bias = 0.35
    n_labeled, n_unlabeled = 120, 3000

    y_all = rng.normal(true_theta, 0.25, size=n_labeled + n_unlabeled)
    f_all = y_all + bias + rng.normal(0.0, 0.04, size=n_labeled + n_unlabeled)
    y_l, f_l, f_u = y_all[:n_labeled], f_all[:n_labeled], f_all[n_labeled:]

    res = compute_ppi_comparison(f_u, f_l, y_l)
    assert res.lambda_hat > 0.80
    assert not res.judge_misaligned
    assert res.judge_bias_estimate == pytest.approx(bias, abs=0.02)
    assert abs(res.theta_ppi - true_theta) < 0.03
    assert abs(res.theta_judge_unlabeled - (true_theta + bias)) < 0.03
    assert res.ci_low < true_theta < res.ci_high
    assert (res.ci_high - res.ci_low) < 0.35 * (res.classical_ci_high - res.classical_ci_low)


def test_ppi_broken_uncorrelated_noise_judge() -> None:
    rng = np.random.default_rng(303)
    n_labeled, n_unlabeled = 80, 1500
    y_l = rng.normal(0.5, 0.2, size=n_labeled)
    f_l = rng.normal(0.5, 0.4, size=n_labeled)
    f_u = rng.normal(0.5, 0.4, size=n_unlabeled)

    res = compute_ppi_comparison(f_u, f_l, y_l)
    assert res.lambda_hat < 0.2
    assert res.judge_misaligned
    assert res.se_ppi <= res.se_classical * 1.05

    untuned = compute_ppi_comparison(f_u, f_l, y_l, power_tune=False)
    assert untuned.lambda_hat == 1.0
    assert untuned.se_ppi > res.se_ppi

    overridden = compute_ppi_comparison(f_u, f_l, y_l, lambda_override=0.25, misaligned_lambda_threshold=0.25)
    assert overridden.lambda_hat == pytest.approx(0.25)
    assert not overridden.judge_misaligned


def test_ppi_zero_variance_constant_judge_fallback() -> None:
    y_l = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.7])
    # Use 0.3 specifically to verify IEEE-754 mean rounding does not leak non-zero covariance.
    f_l = np.full(6, 0.3)
    f_u = np.full(50, 0.3)

    res = compute_ppi_comparison(f_u, f_l, y_l)
    assert res.lambda_hat == 0.0
    assert res.judge_misaligned
    assert res.theta_ppi == pytest.approx(res.theta_classical)
    assert res.se_ppi == pytest.approx(res.se_classical)
    assert res.n_effective_multiplier == pytest.approx(1.0)
    assert res.ci_low == pytest.approx(res.classical_ci_low)
    assert res.ci_high == pytest.approx(res.classical_ci_high)

    # Constant labeled + non-constant unlabeled still falls back cleanly to lambda=0.
    res_const_l = compute_ppi_comparison(np.linspace(0.1, 0.9, 20), f_l, y_l)
    assert res_const_l.lambda_hat == 0.0
    assert res_const_l.se_ppi == pytest.approx(res_const_l.se_classical)

    # Constant human AND constant judge across all CI methods.
    y_const = np.full(8, 0.3)
    for method in ("asymptotic", "bootstrap_percentile", "bootstrap_t"):
        r_const = compute_ppi_comparison(
            np.full(20, 0.3),
            np.full(8, 0.3),
            y_const,
            ci_method=method,
            bootstrap_reps=60 if method != "asymptotic" else 0,
        )
        assert r_const.lambda_hat == 0.0
        assert r_const.se_ppi == 0.0
        assert r_const.se_classical == 0.0
        assert r_const.n_effective_multiplier == 1.0
        assert r_const.ci_low == pytest.approx(0.3)
        assert r_const.ci_high == pytest.approx(0.3)

    # N=1 unlabeled edge case + zero-se_ppi with non-zero se_classical (inf multiplier).
    r_n1 = compute_ppi_comparison([0.5], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
    assert math.isfinite(r_n1.se_ppi) and r_n1.n_unlabeled == 1
    r_inf = compute_ppi_comparison([0.5, 0.5, 0.5], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6], lambda_override=1.0)
    assert r_inf.se_ppi == 0.0 and r_inf.se_classical > 0.0
    assert math.isinf(r_inf.n_effective_multiplier)


def test_ppi_validation_errors() -> None:
    ok_u = [0.5, 0.6, 0.7]
    ok_l = [0.5, 0.6]
    ok_y = [0.4, 0.7]

    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, [0.5, 0.6, 0.7], ok_y)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, [0.5], [0.4])
    with pytest.raises(ValueError):
        compute_ppi_comparison([], ok_l, ok_y)
    with pytest.raises(ValueError):
        compute_ppi_comparison([[0.5, 0.6]], ok_l, ok_y)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, [[0.5, 0.6]], [[0.4, 0.7]])
    with pytest.raises(ValueError):
        compute_ppi_comparison([0.5, math.nan], ok_l, ok_y)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, [0.5, math.inf], ok_y)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, ok_l, [0.4, math.nan])
    for bad_alpha in (0.0, 1.0, -0.05, 1.05, math.nan):
        with pytest.raises(ValueError):
            compute_ppi_comparison(ok_u, ok_l, ok_y, alpha=bad_alpha)
    for bad_lam in (-0.01, 1.01, math.nan):
        with pytest.raises(ValueError):
            compute_ppi_comparison(ok_u, ok_l, ok_y, lambda_override=bad_lam)
    for bad_thresh in (-0.01, 1.01, math.nan):
        with pytest.raises(ValueError):
            compute_ppi_comparison(ok_u, ok_l, ok_y, misaligned_lambda_threshold=bad_thresh)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, ok_l, ok_y, ci_method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, ok_l, ok_y, bootstrap_reps=-1)
    with pytest.raises(ValueError):
        compute_ppi_comparison(ok_u, ok_l, ok_y, ci_method="bootstrap_t", bootstrap_reps=10)


def test_ppi_paired_ab_win_rate_monte_carlo_coverage() -> None:
    rng = np.random.default_rng(777)
    # Paired A/B win-rate differences in {-1, 0, +1}: p(-1)=0.25, p(0)=0.30, p(+1)=0.45 -> true theta = +0.20.
    true_theta = 0.20
    probs = [0.25, 0.30, 0.45]
    outcomes = np.array([-1.0, 0.0, 1.0])
    n_labeled, n_unlabeled = 40, 400
    n_trials = 250

    covered_asym = 0
    covered_boot_t = 0
    ppi_se_sum = 0.0
    classical_se_sum = 0.0

    for trial in range(n_trials):
        y_all = rng.choice(outcomes, size=n_labeled + n_unlabeled, p=probs)
        noise = rng.choice(outcomes, size=n_labeled + n_unlabeled, p=probs)
        # Judge agrees 80% of the time with a systematic +0.15 optimism offset.
        f_all = np.where(rng.random(n_labeled + n_unlabeled) < 0.80, y_all, noise) + 0.15
        y_l, f_l, f_u = y_all[:n_labeled], f_all[:n_labeled], f_all[n_labeled:]

        res_asym = compute_ppi_comparison(f_u, f_l, y_l, alpha=0.05, ci_method="asymptotic")
        res_boot = compute_ppi_comparison(
            f_u, f_l, y_l, alpha=0.05, ci_method="bootstrap_t", bootstrap_reps=200, seed=trial,
        )
        assert res_boot.ci_low <= res_boot.ci_high
        if res_asym.ci_low <= true_theta <= res_asym.ci_high:
            covered_asym += 1
        if res_boot.ci_low <= true_theta <= res_boot.ci_high:
            covered_boot_t += 1
        ppi_se_sum += res_asym.se_ppi
        classical_se_sum += res_asym.se_classical

    cov_asym = covered_asym / n_trials
    cov_boot_t = covered_boot_t / n_trials
    assert 0.90 <= cov_asym <= 0.99
    assert 0.90 <= cov_boot_t <= 0.99
    assert ppi_se_sum < classical_se_sum * 0.8

    # Verify default bootstrap_reps=0 (uses 1000 reps) + deterministic seed + bootstrap_percentile.
    sample_u = rng.normal(0.3, 0.2, size=300)
    sample_l = rng.normal(0.3, 0.2, size=30)
    sample_y = 0.8 * sample_l + rng.normal(0.0, 0.08, size=30)
    p1 = compute_ppi_comparison(sample_u, sample_l, sample_y, ci_method="bootstrap_percentile", seed=19)
    p2 = compute_ppi_comparison(sample_u, sample_l, sample_y, ci_method="bootstrap_percentile", seed=19)
    assert p1 == p2
    assert p1.ci_low < p1.theta_ppi < p1.ci_high
    b_def = compute_ppi_comparison(sample_u, sample_l, sample_y, ci_method="bootstrap_t", seed=19)
    assert b_def.ci_low < b_def.theta_ppi < b_def.ci_high
