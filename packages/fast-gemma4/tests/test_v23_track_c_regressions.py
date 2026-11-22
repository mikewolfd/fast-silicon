"""Regression tests for Track C accuracy & stability promotions in `fast-gemma4`."""

from __future__ import annotations

import contextlib
import fcntl
import inspect
import os
from pathlib import Path
import subprocess
from typing import Iterator

import mlx.core as mx
import numpy as np
import pytest

from fast_gemma4.kernels import fused_softcap_argmax_bf16
from fast_gemma4.model import logit_softcap
from fast_gemma4.mtp_drafter import _run_decode_mtp_unconstrained, run_decode_mtp
from fast_gemma4.speculative import (
    _REPEAT_4GRAM_BREAKER_DEFAULT,
    _REPEAT_4GRAM_MIN_HITS,
    _REPEAT_4GRAM_MIN_HITS_CONSTRAINED,
    _REPEAT_4GRAM_PENALTY,
    _REPEAT_4GRAM_WINDOW,
    _repeated_4gram_penalties,
    run_decode_batch,
    run_decode_prompt_lookup,
    run_decode_rolling,
)

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def _ancestor_holds_lock(lock_path: Path) -> bool:
    try:
        anc: set[int] = set()
        pid = os.getpid()
        while pid > 1 and pid not in anc:
            anc.add(pid)
            ppid_s = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
            pid = int(ppid_s) if ppid_s else 0
        holders = {
            int(x)
            for x in subprocess.check_output(["lsof", "-t", str(lock_path)], text=True).split()
            if x.strip().isdigit()
        }
        return bool((anc - {os.getpid()}) & holders)
    except Exception:
        return False


@contextlib.contextmanager
def _gpu_lock() -> Iterator[None]:
    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(HW_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o666)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            if not _ancestor_holds_lock(HW_LOCK_PATH):
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def test_repeat_4gram_defaults_and_decode_signatures() -> None:
    assert _REPEAT_4GRAM_BREAKER_DEFAULT is True
    assert _REPEAT_4GRAM_WINDOW == 96
    assert _REPEAT_4GRAM_MIN_HITS == 4
    assert _REPEAT_4GRAM_MIN_HITS_CONSTRAINED == 5
    assert _REPEAT_4GRAM_PENALTY == pytest.approx(8.0)
    for fn in (run_decode_batch, run_decode_prompt_lookup, run_decode_rolling, run_decode_mtp):
        sig = inspect.signature(fn)
        assert "repeat_4gram_breaker" in sig.parameters
        assert sig.parameters["repeat_4gram_breaker"].default is None


def test_run_decode_mtp_unconstrained_breaks_degenerate_cycle_lazily() -> None:
    vocab = np.arange(32, dtype=np.int32)
    lm_calls = 0

    class _FakeModel:
        new_to_old = vocab
        old_to_new = vocab

        def compact_cache(self, cache, keep_rows):
            return cache

        def forward_verify_mtp_slot(self, verify_toks, cache, off_cur, slot_idx=None, return_logits=False):
            b_cur, l_q = int(verify_toks.shape[0]), int(verify_toks.shape[1])
            x_step = mx.zeros((b_cur, l_q, 8), dtype=mx.bfloat16)
            am_mx = mx.full((b_cur, l_q), 13, dtype=mx.int32)
            return x_step, am_mx

        def _apply_lm_head(self, x_cell, apply_softcap=True):
            nonlocal lm_calls
            lm_calls += 1
            lg = np.full((1, 1, 32), -5.0, dtype=np.float32)
            lg[0, 0, 13] = 20.0
            lg[0, 0, 21] = 17.0
            return mx.array(lg)

    class _FakeDrafter:
        def draft_k(self, anchor_mx, seed_h, cache, off_cur, cur_k, slot_idx=None):
            return mx.full((int(anchor_mx.shape[0]), int(cur_k)), 13, dtype=mx.int32)

    cache = {"swa_cap": 16, "l_sys": 0}
    init_lg = mx.zeros((1, 1, 32), dtype=mx.float32)
    init_x = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
    init_off = mx.array([4], dtype=mx.int32)
    init_am = mx.array([13], dtype=mx.int32)

    res = _run_decode_mtp_unconstrained(
        _FakeModel(),  # type: ignore[arg-type]
        _FakeDrafter(),  # type: ignore[arg-type]
        frozenset({0}),
        init_lg,
        init_x,
        init_off,
        cache,
        1,
        max_tokens=10,
        draft_k=3,
        t_dec0=0.0,
        init_am=init_am,
    )
    gen = res["gen_toks"][0]
    assert len(gen) == 10
    # Six 13s trigger the p=1 period breaker, switching step 6 lazily via _apply_lm_head to token 21.
    assert gen[:7] == [13, 13, 13, 13, 13, 13, 21]
    assert lm_calls == 1


def test_repeat_4gram_penalties_breaks_consecutive_cycles_and_preserves_stops() -> None:
    # Period p=1 (>= 6 identical tokens) penalizes repeating token unless in stop_token_ids.
    assert _repeated_4gram_penalties([7, 7, 7, 7, 7]) == {}
    pen_p1 = _repeated_4gram_penalties([7, 7, 7, 7, 7, 7])
    assert 7 in pen_p1 and pen_p1[7] >= 8.0
    assert _repeated_4gram_penalties([7, 7, 7, 7, 7, 7], stop_token_ids=frozenset({7})) == {}

    # Period p=2 (>= 4 pairs = 8 tokens) catches alternating loops (e.g., 10, 20 x 4 -> next is 10).
    assert _repeated_4gram_penalties([10, 20] * 3) == {}
    pen_p2 = _repeated_4gram_penalties([10, 20] * 4)
    assert 10 in pen_p2 and pen_p2[10] >= 8.0

    # Period p=3..8 (>= 3 copies) catches multi-token loops cleanly.
    motif = [101, 102, 103, 104, 105, 106]
    assert _repeated_4gram_penalties(motif * 2) == {}
    pen_p6 = _repeated_4gram_penalties(motif * 3)
    assert pen_p6.get(101, 0.0) >= 8.0


def test_repeat_4gram_penalties_threshold_avoids_benign_structured_repetition() -> None:
    # Structured 4-gram repeating only twice/three times (below min_prior_hits=4) must NOT trigger.
    # Simulates benign CSV/JSON keys like `, "name": "a", "name": "b", "name": "c"`.
    seq = [1, 2, 3, 4, 50, 1, 2, 3, 4, 51, 1, 2, 3, 4, 52, 1, 2, 3]
    assert _repeated_4gram_penalties(seq, min_prior_hits=4) == {}

    # Once a degenerate non-adjacent 4-gram suffix hits >= min_prior_hits prior copies, suppress continuation.
    degenerate = [1, 2, 3, 4, 60, 1, 2, 3, 4, 61, 1, 2, 3, 4, 62, 1, 2, 3, 4, 63, 1, 2, 3]
    pen = _repeated_4gram_penalties(degenerate, min_prior_hits=4)
    assert 4 in pen and pen[4] >= 8.0
    assert _repeated_4gram_penalties(degenerate, min_prior_hits=5) == {}


def test_logit_softcap_fp32_precision_cpu() -> None:
    raw = mx.array([[100.0, 105.0, 29.5, -120.0]], dtype=mx.bfloat16)
    out = logit_softcap(30.0, raw)
    assert out.dtype == mx.float32
    raw_np = np.asarray(raw.astype(mx.float32))
    expected = np.tanh(raw_np * np.float32(1.0 / 30.0)) * np.float32(30.0)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6, atol=1e-5)


def test_fused_softcap_argmax_bf16_matches_fp32_softcap_near_boundary_gpu() -> None:
    with _gpu_lock():
        # Values well above softcap saturate near 30.0 in BF16 (where 29.875 == 30.0 in coarse ULP),
        # so FP32 tanh precision in fused_softcap_argmax_bf16 is required to preserve ordering.
        row = np.full((256,), -50.0, dtype=np.float32)
        row[11] = 100.0
        row[37] = 104.0
        bf = mx.array(row[None, :], dtype=mx.bfloat16)
        raw32 = np.asarray(bf.astype(mx.float32))
        ref_softcap = np.tanh(raw32 * np.float32(1.0 / 30.0)) * np.float32(30.0)
        expected_idx = int(np.argmax(ref_softcap, axis=-1)[0])
        assert expected_idx == 37

        out_plain = fused_softcap_argmax_bf16(bf, softcap=30.0, vocab_limit=256)
        out_telem, stats = fused_softcap_argmax_bf16(
            bf, softcap=30.0, vocab_limit=256, collect_telemetry=True
        )
        mx.eval(out_plain, out_telem, stats)
        assert int(out_plain[0]) == expected_idx
        assert int(out_telem[0]) == expected_idx
