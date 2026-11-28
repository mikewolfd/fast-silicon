"""CPU unit tests for fast_decision_models.genrm (GenRMScorer and logprob utilities)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fast_decision_models
from fast_decision_models import (
    BEST_OF_K_GENRM_SYSTEM,
    RAGTRUTH_GENRM_SYSTEM,
    GenRMBestOfKResult,
    GenRMScorer,
)
from fast_decision_models.genrm import (
    GenRMPairwiseResult,
    __all__ as genrm_all,
    default_option_variants,
    format_best_of_k_prompt,
    format_pairwise_prompt,
    format_ragtruth_prompt,
    logsumexp_group_probs,
    resolve_option_token_groups,
    truncate_balanced,
)


class _MockChoiceTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, list[int]] = {
            "Yes": [10],
            " Yes": [11],
            "yes": [12],
            " yes": [13],
            "No": [20],
            " No": [21],
            "no": [22],
            " no": [23],
            "A": [30],
            " A": [31],
            "B": [40],
            " B": [41],
            "C": [50],
            " C": [51],
            "D": [60],
            " D": [61],
            "multi": [70, 71],
            "Faithful (Yes or No):": [90, 91],
            "Best option (A, B, C, or D):": [92, 93],
            "Better response (A or B):": [94, 95],
        }
        self.chat_calls: list[dict[str, object]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text in self._vocab:
            return list(self._vocab[text])
        return [100 + idx for idx, _ in enumerate(str(text).split())]

    def encode_chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> list[int]:
        self.chat_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        # Encode short deterministic tag IDs reflecting prompt content for fake engine.
        return [1 if "grounded_fact" in prompt else 2, len(prompt) % 97]


class _FakeLogprobEngine:
    def __init__(self) -> None:
        self.tokenizer = _MockChoiceTokenizer()
        self.calls: list[tuple[list[list[int]], list[list[int]], int]] = []
        self.closed = False

    def score_next_token_logprobs(self, full_seqs, cand_groups, *, batch_size: int = 8):
        self.calls.append(([list(s) for s in full_seqs], [list(g) for g in cand_groups], batch_size))
        rows: list[list[float]] = []
        for seq in full_seqs:
            logits = np.full((128,), -20.0, dtype=np.float32)
            if seq[0] == 1:
                logits[10] = 3.0
                logits[11] = 2.0
                logits[20] = -1.0
            else:
                logits[10] = -1.0
                logits[20] = 3.0
                logits[21] = 2.5
            rows.append(logsumexp_group_probs(logits, cand_groups))
        return rows

    def close(self) -> None:
        self.closed = True


def test_lazy_top_level_exports() -> None:
    assert fast_decision_models.GenRMScorer is GenRMScorer
    assert fast_decision_models.GenRMBestOfKResult is GenRMBestOfKResult
    assert fast_decision_models.RAGTRUTH_GENRM_SYSTEM == RAGTRUTH_GENRM_SYSTEM
    assert fast_decision_models.BEST_OF_K_GENRM_SYSTEM == BEST_OF_K_GENRM_SYSTEM


def test_logsumexp_group_probs_exact_and_nan_safe() -> None:
    logits = np.full((8,), -100.0, dtype=np.float32)
    logits[1] = 2.0
    logits[2] = 1.0
    logits[4] = 0.5
    logits[6] = float("nan")
    logits[7] = float("inf")

    probs = logsumexp_group_probs(logits, [[1, 2], [4, 6, 7], [99]])
    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    g0 = math.log(math.exp(2.0) + math.exp(1.0))
    g1 = 0.5  # non-finite 6 and 7 are masked to -1e9
    denom = math.exp(g0) + math.exp(g1)
    assert math.isclose(probs[0], math.exp(g0) / denom, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(probs[1], math.exp(g1) / denom, rel_tol=1e-6, abs_tol=1e-6)
    assert probs[2] < 1e-12

    # All-empty or all-invalid groups return uniform distribution.
    uniform = logsumexp_group_probs(np.array([float("nan"), float("-inf")]), [[0], [1], []])
    assert np.allclose(uniform, [1.0 / 3.0] * 3)


def test_resolve_option_token_groups_and_prompt_formatters() -> None:
    tok = _MockChoiceTokenizer()
    groups = resolve_option_token_groups(tok, [["Yes", " Yes", "multi"], ["No", " No"], ["multi"]])
    assert groups == [[10, 11], [20, 21], []]

    assert default_option_variants(["A", "B"]) == [[" A", "A"], [" B", "B"]]

    short = truncate_balanced("alpha beta gamma", 100)
    assert short == "alpha beta gamma"
    clipped = truncate_balanced("A" * 80 + "MIDDLE" + "Z" * 80, 80)
    assert "[...truncated...]" in clipped and clipped.startswith("A") and clipped.endswith("Z")

    rt_prompt = format_ragtruth_prompt(
        {"context": "Source body.", "query": "Summarize.", "response": "Output text."}
    )
    assert "[Reference Context]\nSource body." in rt_prompt
    assert "[User Query]\nSummarize." in rt_prompt
    assert "[Candidate Response]\nOutput text." in rt_prompt

    bok_prompt = format_best_of_k_prompt("User prompt", ["Cand 0", "Cand 1"])
    assert "[Option A]\nCand 0\n[/Option A]" in bok_prompt
    assert "[Option B]\nCand 1\n[/Option B]" in bok_prompt

    pw_prompt = format_pairwise_prompt("User prompt", "Alpha", "Beta")
    assert "[Option A]\nAlpha\n[/Option A]" in pw_prompt
    assert "[Option B]\nBeta\n[/Option B]" in pw_prompt


def test_genrm_scorer_fake_engine_lifecycle_and_ragtruth() -> None:
    fake_eng = _FakeLogprobEngine()
    scorer = GenRMScorer(engine_factory=lambda: fake_eng, batch_size=2)
    assert not scorer.is_initialized

    scores = scorer.score_ragtruth_faithfulness(
        [
            {"context": "grounded_fact: Alice signed.", "query": "Who signed?", "response": "Alice signed."},
            {"context": "Bob signed.", "query": "Who signed?", "response": "Mallory signed."},
        ]
    )
    assert scorer.is_initialized
    assert len(scores) == 2
    assert scores[0] > 0.95
    assert scores[1] < 0.05
    assert fake_eng.calls[0][2] == 2
    assert all(call["enable_thinking"] is False for call in fake_eng.tokenizer.chat_calls)

    with scorer:
        pass
    assert fake_eng.closed
    assert not scorer.is_initialized


def test_genrm_scorer_best_of_k_and_swap_invariant_pairwise_tiebreak() -> None:
    calls: list[tuple[str, list[str]]] = []

    def _mock_logit_scorer(prompts, system_prompt, verdict_suffix, option_variants):
        _ = system_prompt, option_variants
        calls.append((verdict_suffix, list(prompts)))
        if verdict_suffix.startswith("Best option"):
            return [
                # Item 0: clear winner C (margin 0.70 - 0.15 = 0.55 >= tau=0.20)
                [0.10, 0.15, 0.70, 0.05],
                # Item 1: borderline A vs B (margin 0.36 - 0.34 = 0.02 < tau=0.20)
                [0.36, 0.34, 0.15, 0.15],
            ]
        # Pairwise swap-invariant tiebreak on Item 1 (i1=A, i2=B):
        # Simulate slight position bias toward slot A, while candidate B is genuinely better:
        # Forward (slot A=cand A, slot B=cand B): [0.35, 0.65] -> B favored
        # Reverse (slot A=cand B, slot B=cand A): [0.85, 0.15] -> B favored strongly
        # Averaged P(cand A) = 0.5*(0.35 + 0.15) = 0.25, P(cand B) = 0.75 -> flips winner to B!
        assert len(prompts) == 2
        return [[0.35, 0.65], [0.85, 0.15]]

    scorer = GenRMScorer(logit_scorer=_mock_logit_scorer)
    items = [
        {"prompt": "Clear prompt", "candidates": ["a0", "b0", "c0", "d0"]},
        ("Tie prompt", ["cand_A_orig", "cand_B_orig", "cand_C_orig", "cand_D_orig"]),
    ]

    results = scorer.score_best_of_k(items, pairwise_margin_tau=0.20)
    assert len(results) == 2

    # Item 0 should NOT trigger pairwise tiebreak.
    r0 = results[0]
    assert r0.best_idx == 2
    assert r0.pairwise_refined is False
    assert r0.pairwise_probs is None
    assert math.isclose(r0.margin_top1_top2, 0.55, abs_tol=1e-9)

    # Item 1 triggered swap-invariant tiebreak and flipped winner from 0 (A) to 1 (B).
    r1 = results[1]
    assert r1.best_idx == 1
    assert r1.pairwise_refined is True
    assert r1.pairwise_probs is not None
    assert math.isclose(r1.pairwise_probs[0], 0.25, abs_tol=1e-9)
    assert math.isclose(r1.pairwise_probs[1], 0.75, abs_tol=1e-9)
    assert math.isclose(sum(r1.probs), 1.0, abs_tol=1e-9)
    assert r1.probs[1] > r1.probs[0]

    assert [c[0] for c in calls] == [
        "Best option (A, B, C, or D):",
        "Better response (A or B):",
    ]
    # Verify reverse prompt actually swapped candidates A and B.
    fwd_p, rev_p = calls[1][1]
    assert "[Option A]\ncand_A_orig\n[/Option A]" in fwd_p and "[Option B]\ncand_B_orig\n[/Option B]" in fwd_p
    assert "[Option A]\ncand_B_orig\n[/Option A]" in rev_p and "[Option B]\ncand_A_orig\n[/Option B]" in rev_p


def test_genrm_scorer_validation_errors() -> None:
    with pytest.raises(ValueError, match="at most one"):
        GenRMScorer(engine=object(), engine_factory=lambda: object())
    with pytest.raises(TypeError, match="engine_factory"):
        GenRMScorer(engine_factory="not_callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="logit_scorer"):
        GenRMScorer(logit_scorer="not_callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        GenRMScorer(batch_size=0)

    scorer = GenRMScorer(logit_scorer=lambda p, *_a: [[0.5, 0.5] for _ in p])
    with pytest.raises(ValueError, match="at least 2 options"):
        scorer.score_choice_logprob(["hi"], ["OnlyOne"])
    with pytest.raises(ValueError, match="option_variants length"):
        scorer.score_choice_logprob(["hi"], ["A", "B"], option_variants=[["A"]])
    with pytest.raises(ValueError, match="2..8 candidates"):
        scorer.score_best_of_k([("p", ["only_one"])])
    with pytest.raises(ValueError, match="share candidate count K"):
        scorer.score_best_of_k([("p1", ["a", "b"]), ("p2", ["a", "b", "c"])])


def test_normalize_choice_probs_and_raw_logit_scorers() -> None:
    from fast_decision_models.genrm import _normalize_choice_probs

    # Negative log-probabilities must normalize via softmax rather than clamping to 0.
    neg_lp = _normalize_choice_probs([-0.2, -2.5], 2)
    exp0, exp1 = math.exp(-0.2), math.exp(-2.5)
    assert math.isclose(neg_lp[0], exp0 / (exp0 + exp1), rel_tol=1e-9)
    assert math.isclose(neg_lp[1], exp1 / (exp0 + exp1), rel_tol=1e-9)

    # -inf paired with 0.0 must assign 1.0 to 0.0 instead of collapsing to [0.5, 0.5].
    masked = _normalize_choice_probs([0.0, float("-inf")], 2)
    assert masked == [1.0, 0.0]

    # Mixed raw logits trigger softmax automatically, and explicit flag forces softmax on [0, 1] values.
    scorer_auto = GenRMScorer(logit_scorer=lambda p, *_a: [[2.5, -1.0] for _ in p])
    p_auto = scorer_auto.score_choice_logprob(["q"], ["A", "B"])[0]
    e_hi, e_lo = math.exp(2.5), math.exp(-1.0)
    assert math.isclose(p_auto[0], e_hi / (e_hi + e_lo), rel_tol=1e-9)

    scorer_logits = GenRMScorer(
        logit_scorer=lambda p, *_a: [[0.7, 0.3] for _ in p],
        logit_scorer_returns_logits=True,
    )
    p_explicit = scorer_logits.score_choice_logprob(["q"], ["A", "B"])[0]
    e0, e1 = math.exp(0.7), math.exp(0.3)
    assert math.isclose(p_explicit[0], e0 / (e0 + e1), rel_tol=1e-9)
    assert math.isclose(p_explicit[1], e1 / (e0 + e1), rel_tol=1e-9)

    assert _normalize_choice_probs([float("nan"), float("-inf")], 2) == [0.5, 0.5]


def test_score_next_token_logprobs_mlx_chunked_and_vocab_remap_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import threading
    from types import ModuleType, SimpleNamespace

    from fast_decision_models.genrm import score_next_token_logprobs_mlx

    class _FakeArray:
        def __init__(self, data) -> None:
            self._np = np.asarray(data)

        @property
        def shape(self) -> tuple[int, ...]:
            return self._np.shape

        def astype(self, dtype) -> "_FakeArray":
            return _FakeArray(self._np.astype(dtype))

        def __getitem__(self, key):
            return _FakeArray(self._np[key])

        def __array__(self, dtype=None):
            return self._np if dtype is None else self._np.astype(dtype)

    fake_mx = ModuleType("mlx.core")
    fake_mx.bfloat16 = np.float32  # type: ignore[attr-defined]
    fake_mx.float32 = np.float32  # type: ignore[attr-defined]
    fake_mx.int32 = np.int32  # type: ignore[attr-defined]
    fake_mx.array = lambda data, dtype=None: _FakeArray(np.asarray(data, dtype=dtype))  # type: ignore[attr-defined]
    fake_mx.take = lambda arr, idx, axis=-1: _FakeArray(np.take(np.asarray(arr), np.asarray(idx), axis=axis))  # type: ignore[attr-defined]
    fake_mx.take_along_axis = lambda arr, idx, axis=-1: _FakeArray(np.take_along_axis(np.asarray(arr), np.asarray(idx), axis=axis))  # type: ignore[attr-defined]
    fake_mx.stack = lambda seq, axis=0: _FakeArray(np.stack([np.asarray(x) for x in seq], axis=axis))  # type: ignore[attr-defined]
    fake_mx.eval = lambda *_args: None  # type: ignore[attr-defined]
    fake_mx.clear_cache = lambda: None  # type: ignore[attr-defined]

    fake_mlx_pkg = ModuleType("mlx")
    fake_mlx_pkg.core = fake_mx  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    # Part A: empty sequence [] + short [1, 2] + long >512 sequence across chunk boundary.
    seen_chunks: list[tuple[int, list[int] | None]] = []

    class _FakeChunkModel:
        vocab_remap = None
        has_full_vocab_fallback = False

        def make_cache(self, *, kv_step: int, kv_i8: bool, batch_size: int):
            return {"kv_step": kv_step, "kv_i8": kv_i8, "batch_size": batch_size}

        def forward_hidden(self, inp, cache, *, valid_token_scale=None, valid_lengths=None, **_kw):
            _ = cache, valid_token_scale
            arr = np.asarray(inp)
            vl = np.asarray(valid_lengths).tolist() if valid_lengths is not None else None
            seen_chunks.append((arr.shape[1], vl))
            # Encode last valid token ID * 0.1 into hidden[:, :, 0] so chunk selection is observable.
            out = np.zeros((arr.shape[0], arr.shape[1], 2), dtype=np.float32)
            out[:, :, 0] = arr.astype(np.float32)
            return _FakeArray(out)

        def project_logits(self, cur_h):
            h_np = np.asarray(cur_h)
            logits = np.full((h_np.shape[0], 1, 64), -10.0, dtype=np.float32)
            # When last token == 59 (from >512 seq), boost token 20; otherwise boost token 10.
            for r in range(h_np.shape[0]):
                if h_np[r, 0, 0] == 59.0:
                    logits[r, 0, 20] = 6.0
                else:
                    logits[r, 0, 10] = 6.0
            return _FakeArray(logits)

    eng_chunk = SimpleNamespace(
        _lock=threading.Lock(),
        tokenizer=SimpleNamespace(eos_token_id=9),
        model=_FakeChunkModel(),
        _resolve_kv_i8=lambda b_sz, max_len: False,
    )
    long_seq = [1] * 599 + [59]
    res_chunk = score_next_token_logprobs_mlx(
        eng_chunk,
        [long_seq, [], [1, 2]],
        [[10], [20]],
        batch_size=2,
    )
    assert len(res_chunk) == 3
    assert res_chunk[0][1] > 0.99  # long_seq picked token 59 in chunk 2
    assert res_chunk[1][0] > 0.99  # empty [] padded with eos_token_id=9
    assert res_chunk[2][0] > 0.99  # short [1, 2] finished in chunk 1 while long_seq continued
    assert any(c_len == 88 for c_len, _ in seen_chunks)

    # Part B: vocab_remap prunes candidate token 20 -> -1; falls back to project_logits(..., full_vocab=True).
    full_vocab_calls: list[bool] = []

    class _FakeRemapModel(_FakeChunkModel):
        vocab_remap = SimpleNamespace(old_to_new=np.array([0 if i == 10 else -1 for i in range(64)], dtype=np.int32))
        has_full_vocab_fallback = True

        def project_logits(self, cur_h, *, full_vocab: bool = False):
            full_vocab_calls.append(full_vocab)
            h_np = np.asarray(cur_h)
            logits = np.full((h_np.shape[0], 1, 64), -10.0, dtype=np.float32)
            if full_vocab:
                logits[:, 0, 20] = 5.0
            return _FakeArray(logits)

    eng_remap = SimpleNamespace(
        _lock=threading.Lock(),
        tokenizer=SimpleNamespace(eos_token_id=9),
        model=_FakeRemapModel(),
        _resolve_kv_i8=lambda b_sz, max_len: False,
    )
    res_remap = score_next_token_logprobs_mlx(eng_remap, [[1, 2]], [[10], [20]], batch_size=2)
    assert full_vocab_calls == [True]
    assert res_remap[0][1] > 0.99


def test_decision_models_cli_genrm_score_cpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from fast_decision_models import cli as dm_cli
    import fast_decision_models.genrm as genrm_mod

    fake_eng = _FakeLogprobEngine()
    monkeypatch.setattr(
        genrm_mod,
        "GenRMScorer",
        lambda **_kw: GenRMScorer(engine_factory=lambda: fake_eng),
    )

    rc1 = dm_cli.main(
        [
            "score",
            "--backend",
            "genrm",
            "--type",
            "noul",
            "--question",
            "Is this claim supported?",
            "--state",
            "grounded_fact: Alice signed.",
            "--json",
        ]
    )
    assert rc1 == 0
    out1 = json.loads(capsys.readouterr().out)
    assert out1["backend"] == "genrm"
    assert out1["p_true"] > 0.95

    rc2 = dm_cli.main(
        [
            "score",
            "--backend",
            "genrm",
            "--type",
            "choice",
            "--question",
            "Which failed?",
            "--criteria",
            "storage,network",
            "--state",
            "Disk full error.",
            "--json",
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["options"] == ["A", "B"]
    assert math.isclose(sum(out2["probs"]), 1.0, abs_tol=1e-9)


def test_genrm_scorer_pairwise_swap_invariant_and_validation() -> None:
    assert "GenRMPairwiseResult" in genrm_all

    calls: list[tuple[str, list[str], str | None]] = []

    def _mock_pairwise(prompts, system_prompt, verdict_suffix, option_variants):
        _ = option_variants
        calls.append((verdict_suffix, list(prompts), system_prompt))
        assert len(prompts) == 6
        return [
            [0.80, 0.20],  # item 0 fwd: p_fwd_a = 0.80
            [0.20, 0.80],  # item 0 rev: p_rev_a = 0.80 -> p_a=0.80, p_b=0.20, disagree=0.0
            [0.85, 0.15],  # item 1 fwd: p_fwd_a = 0.85
            [0.75, 0.25],  # item 1 rev: p_rev_a = 0.25 -> p_a=0.55, p_b=0.45, disagree=0.60
            [0.35, 0.65],  # item 2 fwd: p_fwd_a = 0.35
            [0.85, 0.15],  # item 2 rev: p_rev_a = 0.15 -> p_a=0.25, p_b=0.75, disagree=0.20
        ]

    scorer = GenRMScorer(logit_scorer=_mock_pairwise)
    assert scorer.score_pairwise_swap_invariant([]) == []
    assert calls == []

    items = [
        ("Prompt 0", "Alpha0", "Beta0"),
        {"prompt": "Prompt 1", "cand_a": "Alpha1", "cand_b": "Beta1"},
        {"prompt": "Prompt 2", "candidates": ["Alpha2", "Beta2"]},
    ]
    res = scorer.score_pairwise_swap_invariant(items)
    assert len(calls) == 1
    assert calls[0][0] == "Better response (A or B):"
    assert calls[0][2] == BEST_OF_K_GENRM_SYSTEM

    prompts_sent = calls[0][1]
    assert "[Option A]\nAlpha0\n[/Option A]" in prompts_sent[0]
    assert "[Option B]\nBeta0\n[/Option B]" in prompts_sent[0]
    assert "[Option A]\nBeta0\n[/Option A]" in prompts_sent[1]
    assert "[Option B]\nAlpha0\n[/Option B]" in prompts_sent[1]
    assert "[Option A]\nAlpha2\n[/Option A]" in prompts_sent[4]
    assert "[Option B]\nBeta2\n[/Option B]" in prompts_sent[4]
    assert "[Option A]\nBeta2\n[/Option A]" in prompts_sent[5]
    assert "[Option B]\nAlpha2\n[/Option B]" in prompts_sent[5]

    expected = [
        (0.80, 0.20, 0.80, 0.80, 0.0),
        (0.55, 0.45, 0.85, 0.25, 0.60),
        (0.25, 0.75, 0.35, 0.15, 0.20),
    ]
    assert len(res) == 3
    for r, (p_a, p_b, p_fwd_a, p_rev_a, disagree) in zip(res, expected):
        assert isinstance(r, GenRMPairwiseResult)
        assert math.isclose(r.p_a, p_a, abs_tol=1e-12)
        assert math.isclose(r.p_b, p_b, abs_tol=1e-12)
        assert math.isclose(r.p_fwd_a, p_fwd_a, abs_tol=1e-12)
        assert math.isclose(r.p_rev_a, p_rev_a, abs_tol=1e-12)
        assert math.isclose(r.order_disagreement, disagree, abs_tol=1e-12)

    with pytest.raises(ValueError, match="both 'cand_a' and 'cand_b'"):
        scorer.score_pairwise_swap_invariant([{"prompt": "p", "cand_a": "a"}])
    with pytest.raises(ValueError, match="length 2"):
        scorer.score_pairwise_swap_invariant([{"prompt": "p", "candidates": ["a", "b", "c"]}])
    with pytest.raises(ValueError, match=r"\(prompt, cand_a, cand_b\)"):
        scorer.score_pairwise_swap_invariant([("p", "a")])  # type: ignore[list-item]


def test_genrm_best_of_k_backward_compat_and_decision_margin() -> None:
    pos_res = GenRMBestOfKResult(1, (0.2, 0.8), 0.6)
    assert pos_res.pairwise_refined is False
    assert pos_res.pairwise_probs is None
    assert pos_res.decision_margin == 0.0
    assert pos_res.pairwise_order_disagreement is None
    assert pos_res.abstain is False
    assert pos_res.abstain_reason is None

    kw_res = GenRMBestOfKResult(best_idx=0, probs=(0.9, 0.1), margin_top1_top2=0.8, pairwise_refined=False)
    assert kw_res.decision_margin == 0.0
    assert kw_res.abstain is False
    assert kw_res.abstain_reason is None

    def _mock_bok(prompts, system_prompt, verdict_suffix, option_variants):
        _ = system_prompt, option_variants
        if verdict_suffix.startswith("Best option"):
            return [
                [0.10, 0.15, 0.70, 0.05],  # clear winner -> no tiebreak
                [0.36, 0.34, 0.15, 0.15],  # tiebreak flips 0 -> 1
                [0.36, 0.34, 0.28, 0.02],  # tiebreak sinks idx 1 below idx 2
            ]
        assert len(prompts) == 4
        return [
            [0.35, 0.65],  # item 1 fwd
            [0.85, 0.15],  # item 1 rev -> p_a=0.25, p_b=0.75, disagree=0.20
            [0.90, 0.10],  # item 2 fwd
            [0.10, 0.90],  # item 2 rev -> p_a=0.90, p_b=0.10, disagree=0.0
        ]

    scorer = GenRMScorer(logit_scorer=_mock_bok)
    items = [
        ("p0", ["a", "b", "c", "d"]),
        ("p1", ["a", "b", "c", "d"]),
        ("p2", ["a", "b", "c", "d"]),
    ]
    res = scorer.score_best_of_k(items, pairwise_margin_tau=0.20)
    assert len(res) == 3

    # Item 0: unchanged listwise margin == decision_margin, no pairwise telemetry.
    assert res[0].best_idx == 2
    assert math.isclose(res[0].margin_top1_top2, 0.55, abs_tol=1e-12)
    assert math.isclose(res[0].decision_margin, 0.55, abs_tol=1e-12)
    assert res[0].pairwise_order_disagreement is None
    assert res[0].abstain is False
    assert res[0].abstain_reason is None

    # Item 1: margin_top1_top2 preserves initial 0.02; decision_margin reflects final [0.175, 0.525, 0.15, 0.15].
    assert res[1].best_idx == 1
    assert math.isclose(res[1].margin_top1_top2, 0.02, abs_tol=1e-12)
    assert math.isclose(res[1].decision_margin, 0.35, abs_tol=1e-12)
    assert res[1].pairwise_order_disagreement is not None
    assert math.isclose(res[1].pairwise_order_disagreement, 0.20, abs_tol=1e-12)
    assert res[1].abstain is False
    assert res[1].abstain_reason is None

    # Item 2: final probs [0.63, 0.07, 0.28, 0.02] -> top2 is idx 2 (0.28), so decision_margin = 0.63 - 0.28 = 0.35.
    assert res[2].best_idx == 0
    assert math.isclose(res[2].margin_top1_top2, 0.02, abs_tol=1e-12)
    assert math.isclose(res[2].decision_margin, 0.35, abs_tol=1e-12)
    assert res[2].pairwise_order_disagreement is not None
    assert math.isclose(res[2].pairwise_order_disagreement, 0.0, abs_tol=1e-12)
    assert res[2].abstain is False
    assert res[2].abstain_reason is None


def test_genrm_best_of_k_selective_abstention_and_threshold_validation() -> None:
    def _mock_abstain(prompts, system_prompt, verdict_suffix, option_variants):
        _ = system_prompt, option_variants
        if verdict_suffix.startswith("Best option"):
            return [
                [0.45, 0.25, 0.20, 0.10],  # item 0: margin=0.20 (not < 0.20), decision_margin=0.20 < 0.30 -> low_margin
                [0.36, 0.34, 0.15, 0.15],  # item 1: refined -> low_margin only
                [0.36, 0.34, 0.15, 0.15],  # item 2: refined -> order_disagreement only
                [0.36, 0.34, 0.15, 0.15],  # item 3: refined -> low_margin+order_disagreement
                [0.36, 0.34, 0.15, 0.15],  # item 4: refined -> passes both thresholds
            ]
        assert len(prompts) == 8
        return [
            [0.60, 0.40], [0.40, 0.60],  # item 1: p_a=0.60, disagree=0.0, final=[0.42, 0.28, 0.15, 0.15], dm=0.14
            [0.95, 0.05], [0.35, 0.65],  # item 2: p_a=0.80, disagree=0.30, final=[0.56, 0.14, 0.15, 0.15], dm=0.41
            [0.85, 0.15], [0.75, 0.25],  # item 3: p_a=0.55, disagree=0.60, final=[0.385, 0.315, 0.15, 0.15], dm=0.07
            [0.90, 0.10], [0.20, 0.80],  # item 4: p_a=0.85, disagree=0.10, final=[0.595, 0.105, 0.15, 0.15], dm=0.445
        ]

    scorer = GenRMScorer(logit_scorer=_mock_abstain)
    items = [(f"p{i}", ["a", "b", "c", "d"]) for i in range(5)]
    res = scorer.score_best_of_k(
        items,
        pairwise_margin_tau=0.20,
        abstain_margin_tau=0.30,
        max_order_disagreement=0.25,
    )
    assert [(r.abstain, r.abstain_reason) for r in res] == [
        (True, "low_margin"),
        (True, "low_margin"),
        (True, "order_disagreement"),
        (True, "low_margin+order_disagreement"),
        (False, None),
    ]
    assert res[0].pairwise_order_disagreement is None
    assert math.isclose(res[1].decision_margin, 0.14, abs_tol=1e-12)
    assert math.isclose(res[2].decision_margin, 0.41, abs_tol=1e-12)
    assert math.isclose(res[3].decision_margin, 0.07, abs_tol=1e-12)
    assert math.isclose(res[4].decision_margin, 0.445, abs_tol=1e-12)

    # Boundary equality does not abstain (strict `<` for margin and strict `>` for disagreement).
    def _mock_exact_boundary(prompts, system_prompt, verdict_suffix, option_variants):
        _ = system_prompt, option_variants
        if verdict_suffix.startswith("Best option"):
            return [[0.50, 0.50]]
        # fwd=[0.875, 0.125], rev=[0.375, 0.625] -> p_fwd_a=0.875, p_rev_a=0.625 -> p_a=0.75, p_b=0.25
        # decision_margin = 0.75 - 0.25 = 0.50, order_disagreement = 0.25
        return [[0.875, 0.125], [0.375, 0.625]]

    boundary_scorer = GenRMScorer(logit_scorer=_mock_exact_boundary)
    b_res = boundary_scorer.score_best_of_k(
        [("p", ["a", "b"])],
        pairwise_margin_tau=0.25,
        abstain_margin_tau=0.50,
        max_order_disagreement=0.25,
    )[0]
    assert b_res.decision_margin == 0.50
    assert b_res.pairwise_order_disagreement == 0.25
    assert b_res.abstain is False
    assert b_res.abstain_reason is None

    # 0.0 thresholds are valid non-negative numbers.
    zero_res = boundary_scorer.score_best_of_k(
        [("p", ["a", "b"])],
        pairwise_margin_tau=0.25,
        abstain_margin_tau=0.0,
        max_order_disagreement=0.0,
    )[0]
    assert zero_res.abstain is True
    assert zero_res.abstain_reason == "order_disagreement"

    # Invalid negative, NaN, Inf, and bool thresholds raise ValueError even on empty items.
    for bad in (-0.01, -1.0, float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError, match="abstain_margin_tau"):
            scorer.score_best_of_k([], abstain_margin_tau=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_order_disagreement"):
            scorer.score_best_of_k(items[:1], max_order_disagreement=bad)  # type: ignore[arg-type]
