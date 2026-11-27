"""Deterministic CPU parity and NaN-guard tests for `fast-gemma4` GPU top-K sampling fast path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import fast_gemma4.speculative as g4_spec
from fast_gemma4.speculative import run_decode_batch


class _TinySamplingModel:
    vocab_size: int = 512

    def __init__(self, logits_seq: list[np.ndarray]) -> None:
        self.new_to_old = np.arange(self.vocab_size, dtype=np.int32)
        self.old_to_new = np.arange(self.vocab_size, dtype=np.int32)
        self._logits_seq = logits_seq
        self._step = 0

    def compact_cache(self, cache: dict[str, Any], keep_rows: list[int]) -> dict[str, Any]:
        return cache

    def forward_step_ragged(
        self,
        toks_mx: mx.array,
        cache: dict[str, Any],
        off_cur: mx.array,
        *,
        apply_softcap: bool = True,
        keep_from_col: int | None = None,
    ) -> mx.array:
        idx = min(self._step, len(self._logits_seq) - 1)
        self._step += 1
        b_cur = int(toks_mx.shape[0])
        arr = self._logits_seq[idx][:b_cur, None, :]
        return mx.array(arr, dtype=mx.float32)


def test_gemma4_run_decode_batch_topk_parity_and_flat_fallback_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = 512
    rng = np.random.default_rng(123)
    steps: list[np.ndarray] = []
    for _ in range(5):
        row0 = rng.normal(loc=-8.0, scale=0.4, size=vocab).astype(np.float32)
        for tok, sc in zip((9, 25, 111, 277, 405), (6.0, 5.1, 4.5, 4.0, 3.2)):
            row0[tok] = sc
        row1 = np.zeros(vocab, dtype=np.float32)  # flat uniform forces safe host fallback
        steps.append(np.stack([row0, row1], axis=0))

    init_lg = mx.array(steps[0][:, None, :], dtype=mx.float32)
    m_fast = _TinySamplingModel(steps[1:])
    res_fast = run_decode_batch(
        m_fast,  # type: ignore[arg-type]
        frozenset({0}),
        None,
        init_lg,
        mx.array([4, 4], dtype=mx.int32),
        {"swa_cap": 16, "l_sys": 0},
        2,
        max_tokens=4,
        temperature=0.6,
        top_p=0.95,
        seed=77,
        compact=False,
        tail_handoff_b=0,
    )

    monkeypatch.setattr(g4_spec, "gather_topk_stats_mlx", lambda *_a, **_kw: None)
    m_ref = _TinySamplingModel(steps[1:])
    res_ref = run_decode_batch(
        m_ref,  # type: ignore[arg-type]
        frozenset({0}),
        None,
        init_lg,
        mx.array([4, 4], dtype=mx.int32),
        {"swa_cap": 16, "l_sys": 0},
        2,
        max_tokens=4,
        temperature=0.6,
        top_p=0.95,
        seed=77,
        compact=False,
        tail_handoff_b=0,
    )
    assert res_fast["gen_toks"] == res_ref["gen_toks"]


def test_gemma4_run_decode_batch_topk_raises_on_nan_cpu() -> None:
    vocab = 512
    bad_row = np.full((1, 1, vocab), -4.0, dtype=np.float32)
    bad_row[0, 0, 10] = np.nan
    m = _TinySamplingModel([bad_row[:, 0, :]])
    with pytest.raises(FloatingPointError, match="NaN"):
        run_decode_batch(
            m,  # type: ignore[arg-type]
            frozenset({0}),
            None,
            mx.array(bad_row, dtype=mx.float32),
            mx.array([4], dtype=mx.int32),
            {"swa_cap": 16, "l_sys": 0},
            1,
            max_tokens=2,
            temperature=0.6,
            top_p=0.95,
            seed=1,
            compact=False,
            tail_handoff_b=0,
        )


def test_gemma4_run_decode_rolling_topk_parity_cpu() -> None:
    """Rolling decode top-k GPU sampling matches naive reference for T=0.6, top_p=0.95."""
    from fast_gemma4.speculative import run_decode_rolling
    from fast_shared.grammar import gather_topk_stats_mlx, try_sample_topk_row

    vocab = 512
    row0 = np.full(vocab, -9.0, dtype=np.float32)
    row0[42] = 4.0
    row0[111] = 3.2
    row1 = np.full(vocab, -8.5, dtype=np.float32)
    row1[77] = 5.0
    row1[205] = 4.1
    nto = np.arange(vocab, dtype=np.int32)

    class _RollSamplingModel:
        window_size = 512
        kv_i8 = True
        final_logit_softcap = None
        new_to_old = nto

        def make_slot_cache(self, prefix_kv, batch_size, doc_cap_full=128, swa_ring_cap=512, kv_i8=None):
            return {
                "batch_size": batch_size,
                "l_sys": 0,
                "k_sys": [],
                "v_sys": [],
                "swa_cap": swa_ring_cap,
                "full_cap": doc_cap_full,
                "kv_i8": True,
                "layers": [],
            }

        def prefill_slots_lazy_lg(self, sub_sufs, target_slots, cache, **kw):
            logits = mx.array(row0[None, None, :], dtype=mx.bfloat16)
            am = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            return logits, am, [len(s) for s in sub_sufs]

        def forward_decode_step_slot(self, toks, cache, off, slots, n_keep=1, return_logits=False):
            logits = mx.array(row1[None, None, :], dtype=mx.bfloat16)
            am = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            return logits, am

    res = run_decode_rolling(
        _RollSamplingModel(),  # type: ignore[arg-type]
        frozenset({0}),
        None,
        {"l_sys": 0, "k_sys": [], "v_sys": []},
        [[10, 11]],
        max_tokens=2,
        b_slots=1,
        swa_ring_cap=512,
        temperature=0.6,
        top_p=0.95,
        seed=123,
    )
    rng = np.random.default_rng(123)
    ti0, tv0, lse0 = gather_topk_stats_mlx(mx.array(row0[None, :], dtype=mx.float32), temperature=0.6, top_k=128)
    expected0 = try_sample_topk_row(ti0[0], tv0[0], float(lse0[0]), nto, temperature=0.6, top_p=0.95, rng=rng)
    ti1, tv1, lse1 = gather_topk_stats_mlx(mx.array(row1[None, :], dtype=mx.float32), temperature=0.6, top_k=128)
    expected1 = try_sample_topk_row(ti1[0], tv1[0], float(lse1[0]), nto, temperature=0.6, top_p=0.95, rng=rng)
    assert res["gen_toks"] == [[expected0, expected1]]
