"""CPU parity and edge-case tests for exact `_sample_from_logits` top-K argpartition fast path."""

from __future__ import annotations

import numpy as np

from fast_shared.grammar import _sample_from_logits


def _naive_reference_sample(
    logits_1d: np.ndarray,
    new_to_old: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    scaled = logits_1d.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= float(probs.sum())
    if top_p < 1.0:
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cdf = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(cdf, top_p, side="right")) + 1
        keep = order[: max(1, cutoff)]
        sub_p = probs[keep]
        sub_p /= sub_p.sum()
        j = int(rng.choice(keep, p=sub_p))
        return int(new_to_old[j])
    j = int(rng.choice(len(probs), p=probs))
    return int(new_to_old[j])


def test_sample_from_logits_argpartition_exact_parity_peaked_cpu() -> None:
    rng_data = np.random.default_rng(7)
    logits = rng_data.normal(loc=-8.0, scale=0.5, size=2048).astype(np.float32)
    # Place clear peaked candidates in top-256 so argpartition covers >= 0.95 mass.
    for idx, val in zip((31, 145, 777, 1599, 1901), (6.0, 5.2, 4.9, 4.4, 3.8)):
        logits[idx] = val
    new_to_old = (np.arange(2048, dtype=np.int64) * 3) + 10

    for seed in range(25):
        got = _sample_from_logits(
            logits, new_to_old, temperature=0.6, top_p=0.95, rng=np.random.default_rng(seed)
        )
        ref = _naive_reference_sample(
            logits, new_to_old, temperature=0.6, top_p=0.95, rng=np.random.default_rng(seed)
        )
        assert got == ref


def test_sample_from_logits_flat_tail_and_nan_fallback_cpu() -> None:
    # Flat uniform logits over 1000 tokens force argpartition top-256 mass < 0.95 -> full-sort fallback.
    logits = np.zeros(1000, dtype=np.float32)
    new_to_old = np.arange(1000, dtype=np.int64)
    for seed in range(10):
        got = _sample_from_logits(
            logits, new_to_old, temperature=0.8, top_p=0.95, rng=np.random.default_rng(seed)
        )
        ref = _naive_reference_sample(
            logits, new_to_old, temperature=0.8, top_p=0.95, rng=np.random.default_rng(seed)
        )
        assert got == ref

    # Small vocab (<= 256) and top_p=1.0 and temperature=0.0 and NaN guard.
    small = np.array([1.0, 3.0, 2.0, -4.0], dtype=np.float32)
    nto_s = np.array([10, 20, 30, 40], dtype=np.int64)
    assert _sample_from_logits(small, nto_s, temperature=0.0) == 20
    assert _sample_from_logits(small, nto_s, temperature=0.7, top_p=1.0, rng=np.random.default_rng(3)) == _naive_reference_sample(
        small, nto_s, temperature=0.7, top_p=1.0, rng=np.random.default_rng(3)
    )
    bad = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    assert _sample_from_logits(bad, np.array([0, 1, 2]), temperature=0.6) == -1


def test_try_sample_topk_row_and_gather_mlx_tiebreak_parity_cpu() -> None:
    import mlx.core as mx
    from fast_shared.grammar import gather_topk_stats_mlx, try_sample_topk_row

    vocab = 512
    logits = np.full(vocab, -10.0, dtype=np.float32)
    # Deliberate exact ties at indices 50 and 200 to verify ascending-index tiebreak parity.
    logits[50] = 5.0
    logits[200] = 5.0
    logits[12] = 4.5
    nto = np.arange(vocab, dtype=np.int64) * 3 + 5
    lg_mx = mx.array(logits[None, :], dtype=mx.float32)
    gathered = gather_topk_stats_mlx(lg_mx, temperature=0.6, top_k=128)
    assert gathered is not None
    ti, tv, lse = gathered

    for seed in range(25):
        got = try_sample_topk_row(
            ti[0], tv[0], float(lse[0]), nto, temperature=0.6, top_p=0.95, rng=np.random.default_rng(seed)
        )
        ref = _naive_reference_sample(
            logits, nto, temperature=0.6, top_p=0.95, rng=np.random.default_rng(seed)
        )
        assert got == ref

    # Small vocab <= top_k returns None to fall back safely.
    assert gather_topk_stats_mlx(mx.zeros((1, 64), dtype=mx.float32), temperature=0.6, top_k=128) is None
    assert gather_topk_stats_mlx(mx.zeros((1, 256), dtype=mx.float32), temperature=float("nan"), top_k=128) is None
    assert gather_topk_stats_mlx(mx.zeros((1, 256), dtype=mx.float32), temperature=float("inf"), top_k=128) is None


def test_boundary_tie_group_excluded_uniform_fallback_cpu() -> None:
    """When top-k slice ends inside a tie group, try_sample_topk_row falls back cleanly unless resolved strictly above the tie."""
    import mlx.core as mx
    from fast_shared.grammar import _sample_from_logits, gather_topk_stats_mlx, try_sample_topk_row

    vocab = 512
    nto = np.arange(vocab, dtype=np.int64)
    # 1. Completely flat logits at top_p=0.20 -> strict_n == 0 -> top-k helper falls back (None),
    # while _sample_from_logits falls back safely to full-vocab distribution across all tied tokens.
    flat = np.zeros(vocab, dtype=np.float32)
    ti_f, tv_f, lse_f = gather_topk_stats_mlx(mx.array(flat[None, :], dtype=mx.float32), temperature=0.6, top_k=128)
    assert try_sample_topk_row(ti_f[0], tv_f[0], float(lse_f[0]), nto, temperature=0.6, top_p=0.20, rng=np.random.default_rng(0)) is None
    samples_cpu = {
        _sample_from_logits(flat, nto, temperature=0.6, top_p=0.20, rng=np.random.default_rng(seed))
        for seed in range(60)
    }
    assert len(samples_cpu) > 1

    # 2. Boundary-crossing tie (1 high token ~0.205, 250 tied tail tokens summing ~0.795):
    # At top_p=0.50, cumulative mass strictly above the tie (0.205) does not reach top_p=0.50,
    # so narrowing within top-128 would truncate the 250-way tie -> must return None to fall back.
    # At top_p=0.15, cumulative mass strictly above the tie already exceeds top_p -> resolves to token 0.
    tie_logits = np.full(vocab, -20.0, dtype=np.float32)
    tie_logits[0] = 1.0
    tie_logits[1:251] = -1.5
    ti_t, tv_t, lse_t = gather_topk_stats_mlx(mx.array(tie_logits[None, :], dtype=mx.float32), temperature=0.6, top_k=128)
    assert try_sample_topk_row(ti_t[0], tv_t[0], float(lse_t[0]), nto, temperature=0.6, top_p=0.50, rng=np.random.default_rng(0)) is None
    assert try_sample_topk_row(ti_t[0], tv_t[0], float(lse_t[0]), nto, temperature=0.6, top_p=0.15, rng=np.random.default_rng(0)) == 0

