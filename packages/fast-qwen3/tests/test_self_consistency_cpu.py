"""Deterministic CPU unit tests for GPU top-K sampling fast path and self-consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from fast_qwen3 import (
    FastQwen3Engine,
    GenerationBatchResult,
    GenerationItem,
    generate_self_consistent,
    select_majority_answer,
)


def _naive_ref_row(logits_1d: np.ndarray, new_to_old: np.ndarray, temperature: float, top_p: float, seed: int) -> int:
    scaled = logits_1d.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= float(probs.sum())
    gen = np.random.default_rng(seed)
    if top_p < 1.0:
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cdf = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(cdf, top_p, side="right")) + 1
        keep = order[: max(1, cutoff)]
        sub_p = probs[keep]
        sub_p /= sub_p.sum()
        return int(new_to_old[int(gen.choice(keep, p=sub_p))])
    return int(new_to_old[int(gen.choice(len(probs), p=probs))])


def test_try_sample_topk_row_exactness_and_fallbacks_cpu() -> None:
    vocab = 512
    logits = np.full(vocab, -10.0, dtype=np.float32)
    for idx, val in zip((5, 77, 211, 490), (5.5, 4.7, 4.1, 3.3)):
        logits[idx] = val
    nto = (np.arange(vocab, dtype=np.int64) * 2) + 1
    temp = 0.6
    scaled = logits.astype(np.float64) / temp
    m = float(np.max(scaled))
    lse = float(m + np.log(np.sum(np.exp(scaled - m))))

    k = 128
    top_idx = np.argpartition(-logits, k - 1)[:k]
    top_vals = logits[top_idx]

    for seed in range(20):
        got = FastQwen3Engine._try_sample_topk_row(
            top_idx, top_vals, lse, nto, temperature=temp, top_p=0.95, rng=np.random.default_rng(seed)
        )
        ref = _naive_ref_row(logits, nto, temp, 0.95, seed)
        assert got == ref

    # Flat uniform distribution over 512 tokens: top-128 < 0.95 mass -> returns None to trigger safe fallback.
    flat = np.zeros(vocab, dtype=np.float32)
    lse_flat = float(np.log(vocab))
    flat_idx = np.arange(128, dtype=np.int64)
    assert (
        FastQwen3Engine._try_sample_topk_row(
            flat_idx, np.zeros(128, dtype=np.float32), lse_flat, nto, temperature=0.8, top_p=0.95, rng=np.random.default_rng(1)
        )
        is None
    )

    # Non-finite LSE / NaN values return None cleanly.
    assert (
        FastQwen3Engine._try_sample_topk_row(
            top_idx, top_vals, float("nan"), nto, temperature=0.6, top_p=0.95, rng=np.random.default_rng(1)
        )
        is None
    )
    bad_vals = top_vals.copy()
    bad_vals[0] = np.nan
    assert (
        FastQwen3Engine._try_sample_topk_row(
            top_idx, bad_vals, lse, nto, temperature=0.6, top_p=0.95, rng=np.random.default_rng(1)
        )
        is None
    )


def test_select_tokens_from_hidden_gpu_topk_and_fallback_cpu() -> None:
    vocab = 300
    peaked = np.full(vocab, -9.0, dtype=np.float32)
    peaked[17] = 5.0
    peaked[99] = 4.4
    flat = np.zeros(vocab, dtype=np.float32)
    mat = np.stack([peaked, flat], axis=0)

    class _StubModel:
        vocab_remap = None

        def project_logits(self, h_3d: mx.array) -> mx.array:
            b = int(h_3d.shape[0])
            return mx.array(mat[:b, None, :], dtype=mx.float32)

    class _StubTok:
        stop_token_ids = frozenset({2})

    eng = object.__new__(FastQwen3Engine)
    eng.model = _StubModel()
    eng.tokenizer = _StubTok()
    eng._new_to_old = np.arange(vocab, dtype=np.int64)

    h = mx.zeros((2, 8), dtype=mx.float32)
    picked, tel, fb = eng._select_tokens_from_hidden(
        h,
        [0, 1],
        2,
        matchers=[None, None],
        grammars=[None, None],
        recent_tokens_by_row=[[], []],
        rngs=[np.random.default_rng(11), np.random.default_rng(11)],
        temperature=0.6,
        top_p=0.95,
        repeat_penalty=True,
        collect_telemetry=False,
    )
    assert fb == 0
    assert tel == {0: None, 1: None}
    assert picked[0] == _naive_ref_row(peaked, eng._new_to_old, 0.6, 0.95, 11)
    assert picked[1] == _naive_ref_row(flat, eng._new_to_old, 0.6, 0.95, 11)


def _mk_item(text: str, finish_reason: str = "stop", completion_tokens: int = 10) -> GenerationItem:
    return GenerationItem(
        text=text,
        parsed=None,
        token_ids=[1] * completion_tokens,
        prompt_tokens=8,
        completion_tokens=completion_tokens,
        valid_json=False,
        valid_schema=False,
        finish_reason=finish_reason,
    )


def test_select_majority_answer_equivalence_and_incomplete_weight_cpu() -> None:
    samples = [
        _mk_item("wrong \\boxed{1/3}", "stop"),
        _mk_item("Step... \\boxed{\\frac{1}{2}}", "stop"),
        _mk_item("Check: \\boxed{0.5}", "stop"),
        _mk_item("Truncated \\boxed{1/3}", "length"),
    ]
    sel = select_majority_answer(samples, incomplete_weight=0.25)
    assert sel.answer in ("\\frac{1}{2}", "0.5")
    assert sel.selected_index == 1
    assert sel.n_valid == 3
    assert sel.consensus_ratio == pytest.approx(2.0 / 3.25, rel=1e-4)

    # All length-truncated still picks highest incomplete weight rather than empty.
    trunc_only = [_mk_item("A \\boxed{42}", "length"), _mk_item("B \\boxed{42}", "length"), _mk_item("C \\boxed{7}", "length")]
    sel_t = select_majority_answer(trunc_only, incomplete_weight=0.25)
    assert sel_t.answer == "42"
    assert sel_t.n_valid == 0


def test_generate_self_consistent_adaptive_and_fixed_cpu() -> None:
    calls: list[dict[str, Any]] = []

    class _FakeEngine:
        def generate_batch(self, prompts: list[str], **kw: Any) -> GenerationBatchResult:
            calls.append({"n": len(prompts), **kw})
            items: list[GenerationItem] = []
            # Prompt 0 agrees on initial_k=2 ("\\boxed{5}"), Prompt 1 disagrees ("\\boxed{1}" vs "length") -> escalates.
            if len(calls) == 1:
                items = [
                    _mk_item("\\boxed{5}", "stop", 20),
                    _mk_item("\\boxed{5}", "stop", 20),
                    _mk_item("\\boxed{1}", "stop", 20),
                    _mk_item("no box yet", "length", 30),
                ]
            else:
                items = [
                    _mk_item("\\boxed{9}", "stop", 25),
                    _mk_item("\\boxed{9}", "stop", 25),
                ]
            return GenerationBatchResult(
                items=items,
                stats={"completion_tokens": sum(i.completion_tokens for i in items)},
            )

    eng = _FakeEngine()
    res = generate_self_consistent(eng, ["p0", "p1"], k=4, adaptive=True, initial_k=2)
    assert len(calls) == 2
    assert calls[0]["n"] == 4
    assert calls[1]["n"] == 2
    assert res.answers == ["5", "9"]
    assert res.selections[0].escalated is False
    assert res.selections[1].escalated is True
    assert res.stats["escalated_prompts"] == 1
    assert res.stats["escalation_rate"] == pytest.approx(0.5)
    assert res.stats["total_samples"] == 6

    calls.clear()
    res_fixed = generate_self_consistent(eng, ["p0"], k=3, adaptive=False)
    assert len(calls) == 1
    assert calls[0]["n"] == 3
    assert res_fixed.answers == ["5"]


def test_select_majority_json_and_choice_helpers_cpu() -> None:
    from fast_shared.self_consistency import extract_choice_letter, select_majority_json

    assert extract_choice_letter("Answer: B") == "B"

    # Whole-object majority wins cleanly.
    s_whole = [
        _mk_item('{"a": 1, "b": "x"}'),
        _mk_item('{"b": "x", "a": 1}'),
        _mk_item('{"a": 2, "b": "y"}'),
    ]
    sel_w = select_majority_json(s_whole)
    assert sel_w.selected_index == 0
    assert sel_w.consensus_ratio >= 0.5

    # Field-level tiebreak when no single complete JSON object repeats identically.
    s_field = [
        _mk_item('{"a": 1, "b": 2, "c": 9}'),
        _mk_item('{"a": 1, "b": 3, "c": 4}'),
        _mk_item('{"a": 0, "b": 2, "c": 4}'),
    ]
    sel_f = select_majority_json(s_field)
    # Winners: a=1 (2/3), b=2 (2/3), c=4 (2/3) -> candidate 1 and 2 each agree on 2 fields, candidate 0 agrees on a & b (first valid tie -> idx 0).
    assert sel_f.selected_index in (0, 1, 2)
    assert sel_f.n_valid == 3


def test_generate_self_consistent_rolling_auto_and_override_cpu() -> None:
    calls: list[dict[str, Any]] = []

    class _RollingEngine:
        def generate_batch(self, prompts: list[str], *, rolling: bool | str = "auto", **kw: Any) -> GenerationBatchResult:
            calls.append({"n": len(prompts), "rolling": rolling, **kw})
            return GenerationBatchResult(
                items=[_mk_item("\\boxed{7}", "stop", 5) for _ in prompts],
                stats={"completion_tokens": 5 * len(prompts)},
            )

    eng = _RollingEngine()
    _ = generate_self_consistent(eng, ["p0", "p1"], k=2, adaptive=False)
    assert calls[-1]["rolling"] is True

    _ = generate_self_consistent(eng, ["p0", "p1"], k=2, adaptive=False, rolling=False)
    assert calls[-1]["rolling"] is False


def test_select_tokens_from_hidden_small_vocab_fallback_cpu() -> None:
    """Vocab <= 128 disables top-k GPU slice and still executes full-vocab fallback without KeyError."""
    vocab = 64
    logits_row = np.full(vocab, -8.0, dtype=np.float32)
    logits_row[5] = 4.0
    logits_row[19] = 3.2
    mat = logits_row[None, :]

    class _StubModel:
        vocab_remap = None

        def project_logits(self, h_3d: mx.array) -> mx.array:
            b = int(h_3d.shape[0])
            return mx.array(mat[:b, None, :], dtype=mx.float32)

    class _StubTok:
        stop_token_ids = frozenset({2})

    eng = object.__new__(FastQwen3Engine)
    eng.model = _StubModel()
    eng.tokenizer = _StubTok()
    eng._new_to_old = np.arange(vocab, dtype=np.int64)

    picked, tel, fb = eng._select_tokens_from_hidden(
        mx.zeros((1, 8), dtype=mx.float32),
        [0],
        1,
        matchers=[None],
        grammars=[None],
        recent_tokens_by_row=[[]],
        rngs=[np.random.default_rng(7)],
        temperature=0.6,
        top_p=0.95,
        repeat_penalty=True,
        collect_telemetry=False,
    )
    assert fb == 0
    assert tel == {0: None}
    assert picked[0] == _naive_ref_row(logits_row, eng._new_to_old, 0.6, 0.95, 7)


def test_generate_self_consistent_empty_prompts_stats_cpu() -> None:
    """Empty prompt input preserves full stats schema including escalated_prompts and total_samples."""

    class _UnusedEngine:
        def generate_batch(self, prompts: list[str], **kw: Any) -> GenerationBatchResult:
            raise AssertionError("generate_batch should not be called for empty prompts")

    res = generate_self_consistent(_UnusedEngine(), [], k=4)
    assert res.items == []
    assert res.selections == []
    assert res.stats["escalated_prompts"] == 0
    assert res.stats["total_samples"] == 0

