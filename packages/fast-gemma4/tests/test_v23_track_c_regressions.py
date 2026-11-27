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


def test_gemma4_channel_thought_stripping_and_token_id_extraction_cpu() -> None:
    from fast_gemma4.tokenizer import FastGemma4Tokenizer, _strip_thinking

    assert _strip_thinking("<|channel>thought\nstep 1\n<channel|>Answer: B") == "Answer: B"
    assert _strip_thinking("<|thought>legacy\n<thought|>Answer: C") == "Answer: C"
    assert _strip_thinking("<|channel>thought\nunclosed thought") == ""
    assert _strip_thinking("prior<channel|>Final") == "Final"

    class _FakeBackendTok:
        def __init__(self) -> None:
            self._special = {100: "<|channel>", 101: "<channel|>", 106: "<turn|>"}
            self._words = {10: "thought\n", 11: "option A or ", 12: "B", 20: "Answer: ", 21: "B"}

        def token_to_id(self, tok: str) -> int | None:
            for k, v in self._special.items():
                if v == tok:
                    return k
            return None

        def get_vocab_size(self, with_added_tokens: bool = True) -> int:
            return 256

        def no_padding(self) -> None:
            pass

        def no_truncation(self) -> None:
            pass

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            out: list[str] = []
            for tid in ids:
                if tid in self._special:
                    if not skip_special_tokens:
                        out.append(self._special[tid])
                else:
                    out.append(self._words.get(tid, f"<{tid}>"))
            return "".join(out)

    tok = FastGemma4Tokenizer.__new__(FastGemma4Tokenizer)
    tok._tok = _FakeBackendTok()  # type: ignore[assignment]
    tok.channel_open_id = 100
    tok.channel_close_id = 101
    tok.think_token_id = 98

    # With skip_special_tokens=True directly, `<|channel>` and `<channel|>` disappear leaving `"thought..."`.
    # `decode_answer_and_reasoning` strips `[100..101]` BEFORE `skip_special_tokens=True`.
    ans, reasoning, ans_ids = tok.decode_answer_and_reasoning([100, 10, 11, 12, 101, 20, 21, 106])
    assert ans == "Answer: B"
    assert reasoning == "option A or B"
    assert ans_ids == [20, 21, 106]

    # Unclosed `<|channel>thought` returns empty answer and preserves reasoning.
    ans_u, reas_u, ans_ids_u = tok.decode_answer_and_reasoning([100, 10, 11])
    assert ans_u == ""
    assert reas_u == "option A or"
    assert ans_ids_u == []

    # Backwards-compatible fast path when no channel tokens appear.
    ans_p, reas_p, ans_ids_p = tok.decode_answer_and_reasoning([20, 21])
    assert ans_p == "Answer: B"
    assert reas_p is None
    assert ans_ids_p == [20, 21]

    # Stray `<channel|>` without `<|channel>` treats prior tokens as reasoning (matching `_strip_thinking`),
    # and multi-block `<|channel>...<channel|>` strips per-block headers and joins cleanly.
    ans_stray, reas_stray, ans_ids_stray = tok.decode_answer_and_reasoning([10, 11, 101, 20, 21])
    assert ans_stray == "Answer: B"
    assert reas_stray == "option A or"
    assert ans_ids_stray == [20, 21]

    ans_multi, reas_multi, ans_ids_multi = tok.decode_answer_and_reasoning(
        [100, 10, 11, 101, 100, 10, 12, 101, 20, 21]
    )
    assert ans_multi == "Answer: B"
    assert reas_multi == "option A or\n\nB"
    assert ans_ids_multi == [20, 21]

    # String-fallback path when channel token IDs are unset also extracts unclosed `<|channel>` / `<|thought>` tails.
    class _StrTok:
        def __init__(self, text: str) -> None:
            self._text = text

        def decode(self, _ids: list[int], skip_special_tokens: bool = True) -> str:
            _ = skip_special_tokens
            return self._text

    tok_str = FastGemma4Tokenizer.__new__(FastGemma4Tokenizer)
    tok_str.channel_open_id = None
    tok_str.channel_close_id = None
    tok_str.think_token_id = None
    tok_str._tok = _StrTok("<|channel>thought\npartial trace")  # type: ignore[assignment]
    ans_sc, reas_sc, _ = tok_str.decode_answer_and_reasoning([1, 2])
    assert ans_sc == ""
    assert reas_sc == "partial trace"

    tok_str._tok = _StrTok("<|thought>inner trace")  # type: ignore[assignment]
    ans_st, reas_st, _ = tok_str.decode_answer_and_reasoning([1, 2])
    assert ans_st == ""
    assert reas_st == "inner trace"


def test_gemma4_bounded_two_pass_thinking_rolling_cpu() -> None:
    from fast_gemma4.engine import FastGemma4Engine, GenerationBatchResult, GenerationItem

    class _FakeTok:
        channel_open_id = 100
        channel_close_id = 101

        def encode_channel_close_suffix(self) -> list[int]:
            return [107, 101]

        def decode_answer_and_reasoning(self, ids: list[int]) -> tuple[str, str | None, list[int]]:
            if 100 in ids and 101 in ids:
                return "Answer: C", "step by step", [30, 31]
            return "Answer: A", None, list(ids)

    eng = FastGemma4Engine.__new__(FastGemma4Engine)
    eng.tokenizer = _FakeTok()  # type: ignore[assignment]
    eng.default_mode = "gpu"
    eng.profile = "auto"
    eng.default_max_b_slots = 32

    seen_encode: list[bool] = []
    seen_roll: list[dict[str, object]] = []

    def _encode_prompts(prompts, *, enable_thinking: bool = False, **_kw):
        seen_encode.append(enable_thinking)
        return [[2, 105, 50 + i] for i in range(len(prompts))]

    def _gen_batch(prompts, **kwargs):
        assert kwargs["enable_thinking"] is True
        assert kwargs["max_tokens"] == 16
        assert kwargs.get("thinking_budget_tokens") is None
        return GenerationBatchResult(
            items=[
                # Item 0: finished inside Pass 1
                GenerationItem("Answer: A", None, [100, 10, 101, 20], 3, 4, False, True, "stop", reasoning="quick"),
                # Item 1: hit thinking_budget=16 with unclosed <|channel>
                GenerationItem("", None, [100, 11, 12], 3, 3, False, True, "length", reasoning="partial"),
            ],
            stats={"completion_tokens": 7, "decode_sec": 0.1, "prefill_sec": 0.05},
        )

    def _run_rolling(*, sys_ids, suf_ids_list, max_tokens, B_slots, **_kw):
        seen_roll.append({"sys_ids": sys_ids, "suf_ids_list": suf_ids_list, "max_tokens": max_tokens, "B_slots": B_slots})
        return {"gen_toks": [[30, 31]], "prefill_sec": 0.02, "decode_sec": 0.03, "tot_out_tok": 2}

    eng.encode_prompts = _encode_prompts  # type: ignore[method-assign]
    eng.generate_batch = _gen_batch  # type: ignore[method-assign]
    eng.run_rolling = _run_rolling  # type: ignore[method-assign]

    res = FastGemma4Engine.generate_batch(
        eng,
        ["p0", "p1"],
        enable_thinking=True,
        thinking_budget_tokens=16,
        max_tokens=8,
        batch_size=32,
    )
    assert seen_encode == [True]
    assert len(seen_roll) == 1
    assert seen_roll[0]["max_tokens"] == 8
    assert seen_roll[0]["B_slots"] == 1
    # Item 1 prompt + [100, 11, 12] + injected [107, 101] (`\n<channel|>`)
    full_p2 = list(seen_roll[0]["sys_ids"]) + list(seen_roll[0]["suf_ids_list"][0])  # type: ignore[index]
    assert full_p2 == [2, 105, 51, 100, 11, 12, 107, 101]
    assert res.items[0].text == "Answer: A" and res.items[0].finish_reason == "stop"
    assert res.items[1].text == "Answer: C" and res.items[1].reasoning == "step by step"
    assert res.items[1].finish_reason == "stop"
    assert res.stats["two_pass_thinking"] is True
    assert res.stats["pass1_done_count"] == 1
    assert res.stats["pass2_count"] == 1
    assert res.stats["pass2_forced_close"] == 1


def test_gemma4_two_pass_review_fixes_and_legacy_reasoning_budget_cpu() -> None:
    from fast_gemma4.engine import FastGemma4Engine, GenerationBatchResult, GenerationItem

    class _EdgeTok:
        bos_token_id = 2
        channel_open_id = 100
        channel_close_id = 101
        stop_token_ids = frozenset({1})

        def encode_channel_close_suffix(self) -> list[int]:
            return [107, 101]

        def decode_answer_and_reasoning(self, ids: list[int]) -> tuple[str, str | None, list[int]]:
            if 30 in ids and 31 in ids:
                return '{"ok": 1}', "done", [30, 31]
            return "trunc", "still thinking", [40, 41]

    eng = FastGemma4Engine.__new__(FastGemma4Engine)
    eng.tokenizer = _EdgeTok()  # type: ignore[assignment]
    eng.default_mode = "auto"
    eng.profile = "auto"
    eng.default_max_b_slots = 16

    recorded_roll: list[dict[str, object]] = []

    def _encode_empty(prompts, **_kw):
        return [[] for _ in prompts]

    def _gen_p1(prompts, **kwargs):
        items = [
            GenerationItem("", None, [100, 11, 12], 1, 3, False, True, "length", reasoning="t")
            for _ in prompts
        ]
        return GenerationBatchResult(
            items=items,
            stats={"completion_tokens": 3 * len(prompts), "decode_sec": 0.1, "prefill_sec": 0.05, "mode_used": "gpu"},
        )

    def _roll2(*, sys_ids, suf_ids_list, max_tokens, B_slots, **kw):
        recorded_roll.append({"sys_ids": sys_ids, "suf_ids_list": suf_ids_list, "max_tokens": max_tokens, "kw": kw})
        if len(suf_ids_list) == 1:
            # Exact budget length ending in stop_token_id=1 -> must finish as "stop".
            return {"gen_toks": [[30, 31, 1]], "prefill_sec": 0.01, "decode_sec": 0.02, "tot_out_tok": 3}
        # Doc 0 hits stop token on exact budget=2; doc 1 hits length without stop token.
        return {"gen_toks": [[30, 1], [40, 41]], "prefill_sec": 0.01, "decode_sec": 0.02, "tot_out_tok": 4}

    eng.encode_prompts = _encode_empty  # type: ignore[method-assign]
    eng.generate_batch = _gen_p1  # type: ignore[method-assign]
    eng.run_rolling = _roll2  # type: ignore[method-assign]

    # Case 1: Empty raw prompt gets [bos_id], singleton Pass 2 forwards mode="gpu" + cache_prefix=False,
    # exact stop token yields finish_reason="stop", valid_json_count recomputed, injected tokens separated.
    res1 = FastGemma4Engine.generate_batch(
        eng,
        [""],
        use_chat_template=False,
        enable_thinking=True,
        thinking_budget_tokens=8,
        answer_max_tokens=3,
        mode="gpu",
    )
    assert recorded_roll[0]["sys_ids"][0] == 2  # type: ignore[index]
    assert recorded_roll[0]["kw"]["mode"] == "gpu"  # type: ignore[index]
    assert recorded_roll[0]["kw"]["cache_prefix"] is False  # type: ignore[index]
    assert res1.items[0].finish_reason == "stop"
    assert res1.items[0].valid_json is True
    assert res1.items[0].prompt_tokens == 1
    assert res1.items[0].completion_tokens == 6  # 3 (Pass 1) + 3 (Pass 2), excluding 2 injected suffix tokens
    assert res1.stats["valid_json_count"] == 1
    assert res1.stats["injected_close_tokens"] == 2
    assert res1.stats["completion_tokens"] == 6
    assert res1.stats["mode_used"] == "gpu"

    # Case 2: Boundary length without stop -> "length"; 2 docs -> cache_prefix=True.
    res2 = FastGemma4Engine.generate_batch(
        eng,
        ["", "x"],
        use_chat_template=False,
        enable_thinking=True,
        thinking_budget_tokens=8,
        answer_max_tokens=2,
        mode="gpu",
    )
    assert recorded_roll[1]["kw"]["cache_prefix"] is True  # type: ignore[index]
    assert res2.items[0].finish_reason == "stop"
    assert res2.items[1].finish_reason == "length"

    # Case 3: answer_max_tokens=0 skips Pass 2 completely and reports pass1_done_count=0.
    roll_before = len(recorded_roll)
    res_zero = FastGemma4Engine.generate_batch(
        eng,
        ["a"],
        enable_thinking=True,
        thinking_budget_tokens=8,
        answer_max_tokens=0,
    )
    assert len(recorded_roll) == roll_before
    assert res_zero.stats["pass2_count"] == 0
    assert res_zero.stats["pass1_done_count"] == 0

    # Case 4: Legacy `reasoning_budget` without `thinking_budget_tokens` must NOT enter two-pass.
    called_two_pass: list[bool] = []
    eng._generate_two_pass_bounded_thinking = lambda *_a, **_kw: called_two_pass.append(True)  # type: ignore[method-assign]
    eng.encode_prompts = lambda prompts, **kw: (  # type: ignore[method-assign]
        [_ for _ in prompts],  # trigger single-pass path cleanly until _gpu_lock
    )[0]
    import contextlib

    class _StopEarly(Exception):
        pass

    @contextlib.contextmanager
    def _abort_lock():
        raise _StopEarly()
        yield

    eng._gpu_lock = _abort_lock()  # type: ignore[assignment]
    with pytest.raises(_StopEarly):
        FastGemma4Engine.generate_batch(eng, ["hi"], reasoning_budget=64, max_tokens=128)
    assert called_two_pass == []


def test_gemma4_cli_bounded_thinking_forwarding_and_json_reasoning_cpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json
    from types import SimpleNamespace

    from fast_gemma4 import cli as g4_cli

    seen_gen: list[dict[str, object]] = []

    class _FakeCliEngine:
        def generate(self, prompt, **kwargs):
            seen_gen.append({"prompt": prompt, **kwargs})
            return SimpleNamespace(
                text="42",
                parsed=None,
                prompt_tokens=12,
                completion_tokens=6,
                valid_json=False,
                valid_schema=True,
                finish_reason="stop",
                reasoning="computed cleanly",
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(g4_cli, "_make_engine", lambda _args: _FakeCliEngine())
    rc = g4_cli.main(
        [
            "generate",
            "--enable-thinking",
            "--thinking-budget-tokens",
            "1024",
            "--answer-max-tokens",
            "256",
            "--json",
            "What is 6 * 7?",
        ]
    )
    assert rc == 0
    assert seen_gen[0]["enable_thinking"] is True
    assert seen_gen[0]["thinking_budget_tokens"] == 1024
    assert seen_gen[0]["answer_max_tokens"] == 256
    out_obj = json.loads(capsys.readouterr().out)
    assert out_obj["text"] == "42"
    assert out_obj["reasoning"] == "computed cleanly"
