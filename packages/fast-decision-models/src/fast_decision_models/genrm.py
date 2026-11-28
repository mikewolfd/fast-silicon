"""Generative Reward Model (GenRM) 1-pass calibrated logprob verifier for decisions & pairwise judging."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np

RAGTRUTH_GENRM_SYSTEM = (
    "You are a strict factual consistency auditor. Compare the Candidate Response against the Reference Context and User Query. "
    "Verify every number, entity, date, attribute, and causal statement. "
    "If ANY claim in the Candidate Response is unsupported by, extrapolated beyond, or contradicts the Reference Context, "
    "the response is NOT faithful (No). Only if EVERY claim is grounded in the Reference Context is it faithful (Yes)."
)

BEST_OF_K_GENRM_SYSTEM = (
    "You are an expert alignment and quality judge evaluating AI assistant responses. "
    "Compare candidate responses strictly on factual accuracy, mathematical/logical correctness, "
    "precise instruction-following (every formatting/length/constraint), appropriate safety without unnecessary over-refusal, "
    "and direct helpful focus."
)

DEFAULT_YES_VARIANTS: tuple[str, ...] = (" Yes", " yes", "Yes", "yes")
DEFAULT_NO_VARIANTS: tuple[str, ...] = (" No", " no", "No", "no")


@dataclass(frozen=True)
class GenRMPairwiseResult:
    """Result from :meth:`GenRMScorer.score_pairwise_swap_invariant`."""

    p_a: float
    p_b: float
    p_fwd_a: float
    p_rev_a: float
    order_disagreement: float


@dataclass(frozen=True)
class GenRMBestOfKResult:
    """Result from :meth:`GenRMScorer.score_best_of_k`."""

    best_idx: int
    probs: tuple[float, ...]
    margin_top1_top2: float
    pairwise_refined: bool = False
    pairwise_probs: tuple[float, float] | None = None
    decision_margin: float = 0.0
    pairwise_order_disagreement: float | None = None
    abstain: bool = False
    abstain_reason: str | None = None


def truncate_balanced(text: str, max_chars: int) -> str:
    """Truncate text preserving 65% head and 35% tail when exceeding ``max_chars``."""
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head_n = int(max_chars * 0.65)
    tail_n = max_chars - head_n - 15
    if tail_n <= 0:
        return s[:max_chars]
    return f"{s[:head_n]}\n[...truncated...]\n{s[-tail_n:]}"


def logsumexp_group_probs(
    logits_1d: Sequence[float] | np.ndarray,
    token_id_groups: Sequence[Sequence[int]],
) -> list[float]:
    """Compute NaN-safe calibrated softmax probabilities over token-ID option groups via logsumexp."""
    n_groups = len(token_id_groups)
    if n_groups == 0:
        return []
    arr = np.asarray(logits_1d, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return [1.0 / float(n_groups)] * n_groups

    group_lse: list[float] = []
    for tids in token_id_groups:
        valid = [int(t) for t in tids if 0 <= int(t) < arr.size and math.isfinite(float(arr[int(t)]))]
        if not valid:
            group_lse.append(float("-inf"))
            continue
        vals = arr[valid]
        m = float(np.max(vals))
        sum_exp = float(np.sum(np.exp(vals - m)))
        if not math.isfinite(sum_exp) or sum_exp <= 0.0:
            group_lse.append(float("-inf"))
        else:
            group_lse.append(m + math.log(sum_exp))

    finite_vals = [v for v in group_lse if math.isfinite(v)]
    if not finite_vals:
        return [1.0 / float(n_groups)] * n_groups
    max_g = max(finite_vals)
    exps = [math.exp(v - max_g) if math.isfinite(v) else 0.0 for v in group_lse]
    z = sum(exps)
    if not math.isfinite(z) or z <= 0.0:
        return [1.0 / float(n_groups)] * n_groups
    return [float(v / z) for v in exps]


def resolve_option_token_groups(
    tokenizer: Any,
    option_variants: Sequence[Sequence[str]],
) -> list[list[int]]:
    """Resolve single-token IDs for each option label across spacing/case variants."""
    enc_fn = getattr(tokenizer, "encode", None)
    if not callable(enc_fn):
        raise TypeError(f"Tokenizer {type(tokenizer).__name__!r} must implement encode()")

    groups: list[list[int]] = []
    for variants in option_variants:
        seen: list[int] = []
        for raw_s in variants:
            try:
                ids = enc_fn(str(raw_s), add_special_tokens=False)
            except TypeError:
                ids = enc_fn(str(raw_s))
            if len(ids) == 1:
                tid = int(ids[0])
                if tid not in seen:
                    seen.append(tid)
        groups.append(seen)
    return groups


def default_option_variants(option_labels: Sequence[str]) -> list[list[str]]:
    """Build default single-token spacing variants (`' A'`, `'A'`) for choice labels."""
    out: list[list[str]] = []
    for lbl in option_labels:
        s = str(lbl).strip()
        variants = [f" {s}", s] if s else [str(lbl)]
        out.append(variants)
    return out


def format_ragtruth_prompt(
    item: Mapping[str, Any] | None = None,
    *,
    context: str | None = None,
    query: str | None = None,
    response: str | None = None,
    max_context_chars: int = 8000,
) -> str:
    """Format a RAGTruth faithfulness audit prompt preserving full context up to ``max_context_chars``."""
    src = item or {}
    ctx_raw = src.get("context", "") if context is None else context
    qry_raw = src.get("query", "") if query is None else query
    out_raw = src.get("output", src.get("response", "")) if response is None else response
    ctx = truncate_balanced(str(ctx_raw or ""), max_context_chars)
    qry = str(qry_raw or "").strip()
    out = str(out_raw or "").strip()
    return (
        f"[Reference Context]\n{ctx}\n\n"
        f"[User Query]\n{qry}\n\n"
        f"[Candidate Response]\n{out}\n\n"
        "Audit every factual claim in [Candidate Response] against [Reference Context]."
    )


def format_best_of_k_prompt(
    user_prompt: str,
    candidates: Sequence[str],
    *,
    max_prompt_chars: int = 2400,
    max_cand_chars: int = 2200,
) -> str:
    """Format a listwise Best-of-K preference verification prompt with labeled options A, B, ..."""
    if len(candidates) < 2 or len(candidates) > 8:
        raise ValueError(f"candidates length must be in [2, 8], got {len(candidates)}")
    labels = tuple(chr(ord("A") + i) for i in range(len(candidates)))
    p_txt = truncate_balanced(str(user_prompt or "").strip(), max_prompt_chars)
    blocks = [
        f"[Option {labels[i]}]\n{truncate_balanced(str(cand or '').strip(), max_cand_chars)}\n[/Option {labels[i]}]"
        for i, cand in enumerate(candidates)
    ]
    joined_labels = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else f"{labels[0]} or {labels[1]}"
    return (
        f"[User Prompt]\n{p_txt}\n\n"
        + "\n\n".join(blocks)
        + f"\n\nSelect the single best response among Options {joined_labels}."
    )


def format_pairwise_prompt(
    user_prompt: str,
    cand_a: str,
    cand_b: str,
    *,
    max_prompt_chars: int = 2400,
    max_cand_chars: int = 2600,
) -> str:
    """Format a pairwise Option A vs Option B comparison prompt."""
    return format_best_of_k_prompt(
        user_prompt,
        (cand_a, cand_b),
        max_prompt_chars=max_prompt_chars,
        max_cand_chars=max_cand_chars,
    )


def score_next_token_logprobs_mlx(
    engine: Any,
    prompt_id_seqs: Sequence[Sequence[int]],
    candidate_groups: Sequence[Sequence[int]],
    *,
    batch_size: int = 8,
) -> list[list[float]]:
    """Run batched prefill-only forward passes on ``FastQwen3Engine`` and return option probabilities."""
    if not prompt_id_seqs:
        return []
    n_groups = len(candidate_groups)
    if n_groups == 0:
        return [[] for _ in prompt_id_seqs]

    import mlx.core as mx

    n = len(prompt_id_seqs)
    pad_id = int(engine.tokenizer.eos_token_id)
    vocab_remap = getattr(engine.model, "vocab_remap", None)
    old_to_new = vocab_remap.old_to_new if vocab_remap is not None else None
    use_full_vocab = False
    if old_to_new is not None:
        has_unmapped = any(
            int(old_to_new[int(tid)]) < 0
            for grp in candidate_groups
            for tid in grp
        )
        if has_unmapped and bool(getattr(engine.model, "has_full_vocab_fallback", False)):
            use_full_vocab = True

    mapped_groups: list[list[int]] = []
    flat_cand_ids: list[int] = []
    for grp in candidate_groups:
        m_grp: list[int] = []
        for tid in grp:
            idx = int(tid) if (old_to_new is None or use_full_vocab) else int(old_to_new[int(tid)])
            if idx >= 0:
                m_grp.append(idx)
                flat_cand_ids.append(idx)
        mapped_groups.append(m_grp)

    uniq_cands = sorted(set(flat_cand_ids))
    if not uniq_cands:
        return [[1.0 / float(n_groups)] * n_groups for _ in range(n)]

    cand_pos_map = {cid: i for i, cid in enumerate(uniq_cands)}
    local_groups = [[cand_pos_map[cid] for cid in grp] for grp in mapped_groups]
    cand_mx = mx.array(uniq_cands, dtype=mx.int32)

    order = sorted(range(n), key=lambda i: len(prompt_id_seqs[i]))
    results: list[list[float]] = [[1.0 / float(n_groups)] * n_groups for _ in range(n)]
    eff_bs = max(1, int(batch_size))

    with engine._lock:
        for start in range(0, n, eff_bs):
            idxs = order[start : start + eff_bs]
            seqs = [list(prompt_id_seqs[i]) for i in idxs]
            b_sz = len(seqs)
            lens = [max(1, len(s)) for s in seqs]
            seqs = [s if s else [pad_id] for s in seqs]
            min_len = min(lens)
            max_len = max(lens)
            any_ragged = any(l < max_len for l in lens)
            padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]

            kv_alloc = max(256, min(512, max_len + 8))
            use_kvi8 = engine._resolve_kv_i8(b_sz, max_len + 8)
            cache = engine.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8, batch_size=b_sz)

            if any_ragged:
                valid_2d = np.zeros((b_sz, max_len, 1), dtype=np.float32)
                for r, l in enumerate(lens):
                    valid_2d[r, :l, 0] = 1.0
                valid_scale: mx.array | None = mx.array(valid_2d, dtype=mx.bfloat16)
                valid_lens_mx: mx.array | None = mx.array(lens, dtype=mx.int32)
            else:
                valid_scale = None
                valid_lens_mx = None

            if max_len <= 512:
                inp = mx.array(padded, dtype=mx.int32)
                h = engine.model.forward_hidden(
                    inp,
                    cache,
                    valid_token_scale=valid_scale,
                    valid_lengths=valid_lens_mx,
                    attn_mask="causal",
                    rope_trig=None,
                )
                if any_ragged:
                    last_idx = mx.array([l - 1 for l in lens], dtype=mx.int32)
                    cur_h = mx.take_along_axis(h, last_idx[:, None, None], axis=1)[:, 0, :]
                else:
                    cur_h = h[:, -1, :]
            else:
                slot_h: list[mx.array | None] = [None] * b_sz
                for c0 in range(0, max_len, 512):
                    c1 = min(max_len, c0 + 512)
                    inp_c = mx.array([row[c0:c1] for row in padded], dtype=mx.int32)
                    chunk_has_pad = any_ragged and (c1 > min_len)
                    v_sc_c = valid_scale[:, c0:c1, :] if (chunk_has_pad and valid_scale is not None) else None
                    v_ln_c = (
                        mx.array([max(0, min(l - c0, c1 - c0)) for l in lens], dtype=mx.int32)
                        if chunk_has_pad
                        else None
                    )
                    ch_h = engine.model.forward_hidden(
                        inp_c,
                        cache,
                        valid_token_scale=v_sc_c,
                        valid_lengths=v_ln_c,
                        attn_mask="causal",
                        rope_trig=None,
                    )
                    mx.eval(ch_h)
                    for r, l in enumerate(lens):
                        pos_r = max(0, l - 1)
                        if c0 <= pos_r < c1:
                            slot_h[r] = ch_h[r, pos_r - c0, :]
                cur_h = mx.stack([slot_h[r] for r in range(b_sz)], axis=0)  # type: ignore[misc]

            if use_full_vocab:
                logits_2d = engine.model.project_logits(cur_h[:, None, :], full_vocab=True)[:, 0, :]
            else:
                logits_2d = engine.model.project_logits(cur_h[:, None, :])[:, 0, :]
            sub_logits = mx.take(logits_2d, cand_mx, axis=-1).astype(mx.float32)
            mx.eval(sub_logits)
            sub_np = np.asarray(sub_logits, dtype=np.float64)

            for r, orig_i in enumerate(idxs):
                results[orig_i] = logsumexp_group_probs(sub_np[r], local_groups)

            del cache, cur_h, logits_2d, sub_logits
            if start + eff_bs < n and hasattr(mx, "clear_cache"):
                mx.clear_cache()

    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return results


def _normalize_choice_probs(
    raw_probs: Sequence[float],
    k: int,
    *,
    raw_logits: bool = False,
) -> list[float]:
    if k <= 0:
        return []
    arr = np.asarray(raw_probs, dtype=np.float64)
    if arr.shape != (k,):
        raise ValueError(f"Expected {k} choice probabilities, got shape {arr.shape}")

    pos_inf_mask = np.isinf(arr) & (arr > 0.0)
    if np.any(pos_inf_mask):
        w = pos_inf_mask.astype(np.float64)
        return [float(x) for x in (w / float(np.sum(w))).tolist()]

    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]
    has_neg_inf = bool(np.any(np.isinf(arr) & (arr < 0.0)))
    # Treat negative values (logprobs/logits), -inf masks, or >1.0 scores as log-space
    # logits so negative log-probabilities are never silently clamped to 0.0.
    is_log_space = bool(raw_logits) or has_neg_inf or bool(
        np.any(finite_vals < 0.0) or np.any(finite_vals > 1.0)
    )
    if is_log_space:
        if not np.any(finite_mask):
            return [1.0 / float(k)] * k
        m = float(np.max(finite_vals))
        exps = np.where(finite_mask, np.exp(np.clip(arr - m, -700.0, 0.0)), 0.0)
        total = float(np.sum(exps))
        if not math.isfinite(total) or total <= 0.0:
            return [1.0 / float(k)] * k
        return [float(x / total) for x in exps.tolist()]

    clean = np.where(finite_mask & (arr >= 0.0), arr, 0.0)
    total = float(np.sum(clean))
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / float(k)] * k
    return [float(x / total) for x in clean.tolist()]


def _validate_non_negative_threshold(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative float, got {value!r}")
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative float, got {value!r}") from exc
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return val


def _parse_pairwise_item(
    idx: int,
    item: Mapping[str, Any] | tuple[str, str, str],
) -> tuple[str, str, str]:
    if isinstance(item, Mapping):
        p_txt = "" if item.get("prompt") is None else str(item.get("prompt"))
        has_pair_keys = "cand_a" in item or "cand_b" in item
        has_cands_key = "candidates" in item and item.get("candidates") is not None
        if has_pair_keys and (
            item.get("cand_a") is not None or item.get("cand_b") is not None or not has_cands_key
        ):
            if "cand_a" not in item or "cand_b" not in item:
                raise ValueError(f"Item {idx} mapping must provide both 'cand_a' and 'cand_b'")
            cand_a = "" if item["cand_a"] is None else str(item["cand_a"])
            cand_b = "" if item["cand_b"] is None else str(item["cand_b"])
            return p_txt, cand_a, cand_b
        if has_cands_key:
            raw_cands = item["candidates"]
            if isinstance(raw_cands, (str, bytes)) or not isinstance(raw_cands, Sequence):
                raise ValueError(f"Item {idx} 'candidates' must be a 2-element sequence")
            cands = ["" if c is None else str(c) for c in raw_cands]
            if len(cands) != 2:
                raise ValueError(f"Item {idx} pairwise 'candidates' must have length 2, got {len(cands)}")
            return p_txt, cands[0], cands[1]
        raise ValueError(
            f"Item {idx} mapping must provide ('cand_a', 'cand_b') or 2-element 'candidates'"
        )

    if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 3:
        raise ValueError(
            f"Item {idx} must be a (prompt, cand_a, cand_b) tuple or mapping, got {type(item).__name__}"
        )
    return (
        "" if item[0] is None else str(item[0]),
        "" if item[1] is None else str(item[1]),
        "" if item[2] is None else str(item[2]),
    )


def _compute_abstention(
    decision_margin: float,
    pairwise_order_disagreement: float | None,
    abstain_margin_tau: float | None,
    max_order_disagreement: float | None,
) -> tuple[bool, str | None]:
    low_margin = abstain_margin_tau is not None and decision_margin < abstain_margin_tau
    high_flip = (
        max_order_disagreement is not None
        and pairwise_order_disagreement is not None
        and pairwise_order_disagreement > max_order_disagreement
    )
    if low_margin and high_flip:
        return True, "low_margin+order_disagreement"
    if low_margin:
        return True, "low_margin"
    if high_flip:
        return True, "order_disagreement"
    return False, None


class GenRMScorer:
    """Reusable 1-pass prefill logprob GenRM verifier for binary faithfulness and listwise Best-of-K judging.

    Supports:
      - Injected ``FastQwen3Engine`` (or compatible ``model`` + ``tokenizer`` engine)
      - Lazy ``engine_factory`` (or default ``FastQwen3Engine`` instantiation on first call)
      - Pluggable ``logit_scorer(prompts, system_prompt, verdict_suffix, option_variants)``
        returning calibrated ``[N, K]`` probabilities or raw group logits for zero-MLX CPU testing.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        engine_factory: Callable[[], Any] | None = None,
        logit_scorer: Callable[
            [Sequence[str], str | None, str, Sequence[Sequence[str]]],
            Sequence[Sequence[float]],
        ]
        | None = None,
        model_dir: str | Path | None = None,
        batch_size: int = 8,
        local_files_only: bool = True,
        yes_variants: Sequence[str] = DEFAULT_YES_VARIANTS,
        no_variants: Sequence[str] = DEFAULT_NO_VARIANTS,
        logit_scorer_returns_logits: bool = False,
    ) -> None:
        if engine is not None and engine_factory is not None:
            raise ValueError("Provide at most one of `engine` or `engine_factory`")
        if engine_factory is not None and not callable(engine_factory):
            raise TypeError("`engine_factory` must be callable")
        if logit_scorer is not None and not callable(logit_scorer):
            raise TypeError("`logit_scorer` must be callable")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")

        self._engine: Any | None = engine
        self._engine_factory: Callable[[], Any] | None = engine_factory
        self._owns_engine: bool = False
        self._logit_scorer = logit_scorer
        self._logit_scorer_returns_logits: bool = bool(logit_scorer_returns_logits)
        self._model_dir = model_dir
        self.batch_size: int = int(batch_size)
        self.local_files_only: bool = bool(local_files_only)
        self.yes_variants: tuple[str, ...] = tuple(str(v) for v in yes_variants)
        self.no_variants: tuple[str, ...] = tuple(str(v) for v in no_variants)
        self._lock = threading.Lock()

    @property
    def is_initialized(self) -> bool:
        return self._engine is not None

    def _get_engine(self) -> Any:
        eng = self._engine
        if eng is not None:
            return eng
        with self._lock:
            eng = self._engine
            if eng is not None:
                return eng
            if self._engine_factory is not None:
                eng = self._engine_factory()
            else:
                from fast_qwen3.engine import FastQwen3Engine

                eng = FastQwen3Engine(
                    model_dir=self._model_dir,
                    default_mode="auto",
                    default_enable_thinking=False,
                    kv_i8="auto",
                    local_files_only=self.local_files_only,
                    warmup=True,
                )
            self._engine = eng
            self._owns_engine = True
            return eng

    def score_choice_logprob(
        self,
        prompts: Sequence[str],
        option_labels: Sequence[str],
        *,
        system_prompt: str | None = None,
        verdict_suffix: str = "Answer:",
        option_variants: Sequence[Sequence[str]] | None = None,
    ) -> list[list[float]]:
        """Score each prompt over ``option_labels`` via 1-pass prefill next-token logprobs."""
        k = len(option_labels)
        if k < 2:
            raise ValueError(f"option_labels must contain at least 2 options, got {k}")
        prompt_list = [str(p) for p in prompts]
        if not prompt_list:
            return []

        eff_variants = (
            [list(v) for v in option_variants]
            if option_variants is not None
            else default_option_variants(option_labels)
        )
        if len(eff_variants) != k:
            raise ValueError(f"option_variants length ({len(eff_variants)}) != option_labels ({k})")

        if self._logit_scorer is not None:
            raw_rows = self._logit_scorer(prompt_list, system_prompt, verdict_suffix, eff_variants)
            if len(raw_rows) != len(prompt_list):
                raise ValueError(
                    f"logit_scorer returned {len(raw_rows)} rows for {len(prompt_list)} prompts"
                )
            return [
                _normalize_choice_probs(row, k, raw_logits=self._logit_scorer_returns_logits)
                for row in raw_rows
            ]

        eng = self._get_engine()
        tok = eng.tokenizer
        cand_groups = resolve_option_token_groups(tok, eff_variants)
        suffix_ids = tok.encode(verdict_suffix, add_special_tokens=False)
        full_seqs = [
            tok.encode_chat(p, system_prompt=system_prompt, enable_thinking=False) + suffix_ids
            for p in prompt_list
        ]

        custom_fn = getattr(eng, "score_next_token_logprobs", None)
        if callable(custom_fn):
            raw_rows = custom_fn(full_seqs, cand_groups, batch_size=self.batch_size)
        else:
            raw_rows = score_next_token_logprobs_mlx(
                eng, full_seqs, cand_groups, batch_size=self.batch_size
            )
        return [_normalize_choice_probs(row, k) for row in raw_rows]

    def score_yes_no_logprob(
        self,
        prompts: Sequence[str],
        *,
        system_prompt: str | None = None,
        verdict_suffix: str = "Faithful (Yes or No):",
        yes_variants: Sequence[str] | None = None,
        no_variants: Sequence[str] | None = None,
    ) -> list[float]:
        """Return calibrated ``P(Yes) in [0, 1]`` for each binary verification prompt."""
        if not prompts:
            return []
        y_var = list(self.yes_variants if yes_variants is None else yes_variants)
        n_var = list(self.no_variants if no_variants is None else no_variants)
        rows = self.score_choice_logprob(
            prompts,
            ("Yes", "No"),
            system_prompt=system_prompt,
            verdict_suffix=verdict_suffix,
            option_variants=(y_var, n_var),
        )
        return [float(min(1.0, max(0.0, row[0]))) for row in rows]

    def score_ragtruth_faithfulness(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        max_context_chars: int = 8000,
        system_prompt: str = RAGTRUTH_GENRM_SYSTEM,
        verdict_suffix: str = "Faithful (Yes or No):",
    ) -> list[float]:
        """Score RAGTruth items for factual faithfulness using calibrated Yes/No logprobs."""
        prompts = [format_ragtruth_prompt(it, max_context_chars=max_context_chars) for it in items]
        return self.score_yes_no_logprob(
            prompts,
            system_prompt=system_prompt,
            verdict_suffix=verdict_suffix,
        )

    def score_pairwise_swap_invariant(
        self,
        items: Sequence[Mapping[str, Any] | tuple[str, str, str]],
        *,
        system_prompt: str = BEST_OF_K_GENRM_SYSTEM,
        max_prompt_chars: int = 2400,
        max_cand_chars: int = 2600,
    ) -> list[GenRMPairwiseResult]:
        """Score pairwise items in both ``(a, b)`` and ``(b, a)`` order in one batched call."""
        if not items:
            return []

        norm_items = [_parse_pairwise_item(idx, it) for idx, it in enumerate(items)]
        pair_prompts: list[str] = []
        for p_txt, cand_a, cand_b in norm_items:
            pair_prompts.append(
                format_pairwise_prompt(
                    p_txt,
                    cand_a,
                    cand_b,
                    max_prompt_chars=max_prompt_chars,
                    max_cand_chars=max_cand_chars,
                )
            )
            pair_prompts.append(
                format_pairwise_prompt(
                    p_txt,
                    cand_b,
                    cand_a,
                    max_prompt_chars=max_prompt_chars,
                    max_cand_chars=max_cand_chars,
                )
            )

        pair_probs = self.score_choice_logprob(
            pair_prompts,
            ("A", "B"),
            system_prompt=system_prompt,
            verdict_suffix="Better response (A or B):",
        )

        results: list[GenRMPairwiseResult] = []
        for idx in range(len(norm_items)):
            pr_fwd = pair_probs[2 * idx]
            pr_rev = pair_probs[2 * idx + 1]
            p_fwd_a = float(pr_fwd[0])
            # Reversed prompt puts (cand_b, cand_a) as (A, B), so original A corresponds to B (index 1).
            p_rev_a = float(pr_rev[1])
            p_a = 0.5 * (p_fwd_a + p_rev_a)
            p_b = 1.0 - p_a
            results.append(
                GenRMPairwiseResult(
                    p_a=p_a,
                    p_b=p_b,
                    p_fwd_a=p_fwd_a,
                    p_rev_a=p_rev_a,
                    order_disagreement=abs(p_fwd_a - p_rev_a),
                )
            )
        return results

    def score_best_of_k(
        self,
        items: Sequence[Mapping[str, Any] | tuple[str, Sequence[str]]],
        *,
        pairwise_margin_tau: float | None = None,
        abstain_margin_tau: float | None = None,
        max_order_disagreement: float | None = None,
        system_prompt: str = BEST_OF_K_GENRM_SYSTEM,
        max_prompt_chars: int = 2400,
        max_cand_chars: int = 2200,
    ) -> list[GenRMBestOfKResult]:
        """Score Best-of-K preference items via listwise logprob with optional swap-invariant pairwise tiebreak."""
        eff_abstain_tau = _validate_non_negative_threshold("abstain_margin_tau", abstain_margin_tau)
        eff_max_disagree = _validate_non_negative_threshold("max_order_disagreement", max_order_disagreement)
        if not items:
            return []

        norm_items: list[tuple[str, list[str]]] = []
        for idx, it in enumerate(items):
            if isinstance(it, Mapping):
                p_txt = str(it.get("prompt") or "")
                cands = [str(c) for c in (it.get("candidates") or ())]
            else:
                p_txt = str(it[0])
                cands = [str(c) for c in it[1]]
            if len(cands) < 2 or len(cands) > 8:
                raise ValueError(f"Item {idx} must have 2..8 candidates, got {len(cands)}")
            norm_items.append((p_txt, cands))

        k_set = {len(cands) for _, cands in norm_items}
        if len(k_set) != 1:
            raise ValueError(f"All items in score_best_of_k batch must share candidate count K, got {sorted(k_set)}")
        k = next(iter(k_set))
        labels = tuple(chr(ord("A") + i) for i in range(k))
        joined = ", ".join(labels[:-1]) + f", or {labels[-1]}" if k > 2 else f"{labels[0]} or {labels[1]}"
        list_suffix = f"Best option ({joined}):"

        list_prompts = [
            format_best_of_k_prompt(
                p_txt, cands, max_prompt_chars=max_prompt_chars, max_cand_chars=max_cand_chars
            )
            for p_txt, cands in norm_items
        ]
        probs_batch = self.score_choice_logprob(
            list_prompts,
            labels,
            system_prompt=system_prompt,
            verdict_suffix=list_suffix,
        )

        top_pairs: list[tuple[int, int, float]] = []
        ambig_indices: list[int] = []
        ambig_pairs: list[tuple[str, str, str]] = []
        for idx, row in enumerate(probs_batch):
            arr = np.asarray(row, dtype=np.float64)
            order = np.argsort(-arr, kind="mergesort")
            i1, i2 = int(order[0]), int(order[1])
            margin = float(arr[i1] - arr[i2])
            top_pairs.append((i1, i2, margin))
            if pairwise_margin_tau is not None and margin < float(pairwise_margin_tau):
                p_txt, cands = norm_items[idx]
                ambig_indices.append(idx)
                ambig_pairs.append((p_txt, cands[i1], cands[i2]))

        refined_map: dict[int, GenRMPairwiseResult] = {}
        if ambig_pairs:
            pw_results = self.score_pairwise_swap_invariant(
                ambig_pairs,
                system_prompt=system_prompt,
                max_prompt_chars=max_prompt_chars,
                max_cand_chars=max_cand_chars,
            )
            refined_map = dict(zip(ambig_indices, pw_results))

        results: list[GenRMBestOfKResult] = []
        for idx, row in enumerate(probs_batch):
            i1, i2, margin = top_pairs[idx]
            cur_probs = list(row)
            pw_res = refined_map.get(idx)
            pw_pair: tuple[float, float] | None = None
            pw_disagree: float | None = None
            if pw_res is not None:
                pw_pair = (pw_res.p_a, pw_res.p_b)
                pw_disagree = pw_res.order_disagreement
                mass = max(1e-6, cur_probs[i1] + cur_probs[i2])
                cur_probs[i1] = mass * pw_res.p_a
                cur_probs[i2] = mass * pw_res.p_b
                cur_probs = _normalize_choice_probs(cur_probs, k)
            arr = np.asarray(cur_probs, dtype=np.float64)
            final_order = np.argsort(-arr, kind="mergesort")
            best_idx = int(final_order[0])
            decision_margin = float(arr[final_order[0]] - arr[final_order[1]])
            abstain, abstain_reason = _compute_abstention(
                decision_margin,
                pw_disagree,
                eff_abstain_tau,
                eff_max_disagree,
            )
            results.append(
                GenRMBestOfKResult(
                    best_idx=best_idx,
                    probs=tuple(cur_probs),
                    margin_top1_top2=margin,
                    pairwise_refined=pw_res is not None,
                    pairwise_probs=pw_pair,
                    decision_margin=decision_margin,
                    pairwise_order_disagreement=pw_disagree,
                    abstain=abstain,
                    abstain_reason=abstain_reason,
                )
            )
        return results

    def close(self) -> None:
        eng = self._engine
        owns = self._owns_engine
        self._engine = None
        self._owns_engine = False
        if owns and eng is not None:
            close_fn = getattr(eng, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> GenRMScorer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "BEST_OF_K_GENRM_SYSTEM",
    "DEFAULT_NO_VARIANTS",
    "DEFAULT_YES_VARIANTS",
    "GenRMBestOfKResult",
    "GenRMPairwiseResult",
    "GenRMScorer",
    "RAGTRUTH_GENRM_SYSTEM",
    "default_option_variants",
    "format_best_of_k_prompt",
    "format_pairwise_prompt",
    "format_ragtruth_prompt",
    "logsumexp_group_probs",
    "resolve_option_token_groups",
    "score_next_token_logprobs_mlx",
    "truncate_balanced",
]
