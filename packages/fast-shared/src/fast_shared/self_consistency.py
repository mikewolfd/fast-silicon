"""Engine-agnostic batched test-time self-consistency (majority voting and adaptive escalation)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
import json
import time
from typing import Any


@dataclass(slots=True)
class SelfConsistencySelection:
    """Winning candidate and cluster telemetry for a single prompt."""

    answer: Any | None
    selected_index: int
    selected_item: Any
    clusters: list[dict[str, Any]]
    n_samples: int
    n_valid: int
    consensus_ratio: float
    escalated: bool


@dataclass(slots=True)
class SelfConsistencyBatchResult:
    """Batch output from `generate_self_consistent` with per-prompt winners and stats."""

    selections: list[SelfConsistencySelection]
    answers: list[Any | None]
    items: list[Any]
    stats: dict[str, Any]


def _default_is_valid(item: Any) -> bool:
    fr = getattr(item, "finish_reason", "stop")
    return str(fr or "stop") != "length"


def _default_equiv(a: Any, b: Any) -> bool:
    if a == b:
        return True
    try:
        from fast_shared.outcome_eval.chat_reasoning import math_answers_equivalent

        return bool(math_answers_equivalent(str(a), str(b)))
    except Exception:
        return str(a).strip() == str(b).strip()


def extract_choice_letter(text: str) -> str | None:
    """Extract normalized A-D multiple-choice option letter from model output."""
    try:
        from fast_shared.outcome_eval.chat_reasoning import extract_mmlu_choice

        return extract_mmlu_choice(text)
    except Exception:
        return None


def canonical_json_key(text: str) -> str | None:
    """Parse JSON text and return canonical sorted-key string representation, or None if invalid."""
    try:
        obj = json.loads(text)
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return None


def select_majority_answer(
    samples: Sequence[Any],
    *,
    extract_fn: Callable[[str], Any | None] | None = None,
    equiv_fn: Callable[[Any, Any], bool] | None = None,
    valid_fn: Callable[[Any], bool] | None = None,
    incomplete_weight: float = 0.25,
    escalated: bool = False,
) -> SelfConsistencySelection:
    """Cluster extracted answers by equivalence and select the highest-weight majority cluster."""
    if not samples:
        raise ValueError("select_majority_answer requires at least one sample")
    if extract_fn is None:
        from fast_shared.outcome_eval.chat_reasoning import extract_boxed_answer

        ext_fn: Callable[[str], Any | None] = extract_boxed_answer
    else:
        ext_fn = extract_fn
    eq_fn = equiv_fn or _default_equiv
    val_fn = valid_fn or _default_is_valid
    inc_w = max(0.0, float(incomplete_weight))

    clusters: list[dict[str, Any]] = []
    n_valid_ans = 0
    total_weight = 0.0
    for idx, samp in enumerate(samples):
        text = str(getattr(samp, "text", samp) or "")
        is_valid = bool(val_fn(samp))
        ans = ext_fn(text)
        if ans is None:
            continue
        w = 1.0 if is_valid else inc_w
        if is_valid:
            n_valid_ans += 1
        total_weight += w
        matched = None
        for cl in clusters:
            if eq_fn(ans, cl["rep"]):
                matched = cl
                break
        if matched is None:
            clusters.append(
                {
                    "rep": ans,
                    "count": 1,
                    "weight": w,
                    "valid_count": 1 if is_valid else 0,
                    "first_valid_idx": idx if is_valid else None,
                    "first_idx": idx,
                }
            )
        else:
            matched["count"] += 1
            matched["weight"] += w
            if is_valid:
                matched["valid_count"] += 1
                if matched["first_valid_idx"] is None:
                    matched["first_valid_idx"] = idx

    if not clusters:
        first_valid = next((i for i, s in enumerate(samples) if val_fn(s)), 0)
        return SelfConsistencySelection(
            answer=None,
            selected_index=first_valid,
            selected_item=samples[first_valid],
            clusters=[],
            n_samples=len(samples),
            n_valid=0,
            consensus_ratio=0.0,
            escalated=bool(escalated),
        )

    best = max(clusters, key=lambda c: (float(c["weight"]), int(c["valid_count"]), int(c["count"])))
    sel_idx = int(best["first_valid_idx"] if best["first_valid_idx"] is not None else best["first_idx"])
    ratio = float(best["weight"]) / total_weight if total_weight > 0.0 else 0.0
    summary_clusters = [
        {"answer": c["rep"], "count": c["count"], "weight": round(float(c["weight"]), 4), "valid_count": c["valid_count"]}
        for c in sorted(clusters, key=lambda c: (-float(c["weight"]), -int(c["count"])))
    ]
    return SelfConsistencySelection(
        answer=best["rep"],
        selected_index=sel_idx,
        selected_item=samples[sel_idx],
        clusters=summary_clusters,
        n_samples=len(samples),
        n_valid=n_valid_ans,
        consensus_ratio=round(ratio, 6),
        escalated=bool(escalated),
    )


def select_majority_json(
    samples: Sequence[Any],
    *,
    valid_fn: Callable[[Any], bool] | None = None,
    incomplete_weight: float = 0.25,
    escalated: bool = False,
) -> SelfConsistencySelection:
    """Select consensus JSON response using exact canonical vote with field-level agreement tiebreak."""
    val_fn = valid_fn or _default_is_valid
    sel = select_majority_answer(
        samples,
        extract_fn=canonical_json_key,
        equiv_fn=lambda a, b: str(a) == str(b),
        valid_fn=val_fn,
        incomplete_weight=incomplete_weight,
        escalated=escalated,
    )
    if sel.clusters and sel.consensus_ratio >= 0.5:
        return sel

    # Field-level agreement vote when valid JSON dicts have no strict whole-object majority.
    valid_pairs: list[tuple[int, dict[str, Any]]] = []
    for idx, s in enumerate(samples):
        if not val_fn(s):
            continue
        try:
            obj = json.loads(str(getattr(s, "text", s) or ""))
            if isinstance(obj, dict):
                valid_pairs.append((idx, obj))
        except Exception:
            continue
    if len(valid_pairs) < 2:
        return sel

    all_keys: list[str] = []
    for _, obj in valid_pairs:
        for k in obj.keys():
            if k not in all_keys:
                all_keys.append(k)
    field_winners: dict[str, str] = {}
    for k in all_keys:
        vals = [json.dumps(obj.get(k), sort_keys=True, ensure_ascii=False) for _, obj in valid_pairs if k in obj]
        if vals:
            field_winners[k] = Counter(vals).most_common(1)[0][0]
    best_idx = valid_pairs[0][0]
    best_agree = -1
    for idx, obj in valid_pairs:
        agree = sum(
            1
            for k, win_val in field_winners.items()
            if k in obj and json.dumps(obj[k], sort_keys=True, ensure_ascii=False) == win_val
        )
        if agree > best_agree:
            best_agree = agree
            best_idx = idx
    return SelfConsistencySelection(
        answer=canonical_json_key(str(getattr(samples[best_idx], "text", samples[best_idx]) or "")),
        selected_index=best_idx,
        selected_item=samples[best_idx],
        clusters=sel.clusters,
        n_samples=sel.n_samples,
        n_valid=sel.n_valid,
        consensus_ratio=sel.consensus_ratio,
        escalated=bool(escalated),
    )


def _supports_kwarg(fn: Any, kw: str) -> bool:
    try:
        return kw in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def generate_self_consistent(
    engine: Any,
    prompts: Sequence[Any],
    *,
    k: int = 4,
    adaptive: bool = False,
    initial_k: int = 2,
    extract_fn: Callable[[str], Any | None] | None = None,
    equiv_fn: Callable[[Any, Any], bool] | None = None,
    valid_fn: Callable[[Any], bool] | None = None,
    incomplete_weight: float = 0.25,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    seed: int = 42,
    **generate_kwargs: Any,
) -> SelfConsistencyBatchResult:
    """Run batched test-time self-consistency across any generative engine (`generate_batch`)."""
    n_prompts = len(prompts)
    k_eff = max(1, int(k))
    init_k = max(1, min(k_eff, int(initial_k))) if adaptive else k_eff
    if extract_fn is None:
        from fast_shared.outcome_eval.chat_reasoning import extract_boxed_answer

        ext_fn: Callable[[str], Any | None] = extract_boxed_answer
    else:
        ext_fn = extract_fn
    eq_fn = equiv_fn or _default_equiv
    val_fn = valid_fn or _default_is_valid

    if n_prompts == 0:
        return SelfConsistencyBatchResult(
            selections=[],
            answers=[],
            items=[],
            stats={
                "prompts": 0,
                "k": k_eff,
                "initial_k": init_k,
                "adaptive": bool(adaptive),
                "escalated_prompts": 0,
                "escalation_rate": 0.0,
                "total_samples": 0,
                "completion_tokens": 0,
                "elapsed_sec": 0.0,
            },
        )

    gen_kw = dict(generate_kwargs)
    gen_fn = getattr(engine, "generate_batch", None)
    if "enable_thinking" in gen_kw and callable(gen_fn) and not _supports_kwarg(gen_fn, "enable_thinking"):
        gen_kw.pop("enable_thinking", None)
    if callable(gen_fn) and _supports_kwarg(gen_fn, "rolling"):
        if "rolling" not in gen_kw and n_prompts * init_k >= 4:
            gen_kw["rolling"] = True
    elif "rolling" in gen_kw:
        gen_kw.pop("rolling", None)

    t0 = time.perf_counter()
    stage1_prompts = [p for p in prompts for _ in range(init_k)]
    b1 = engine.generate_batch(
        stage1_prompts,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
        **gen_kw,
    )
    samples_by_prompt: list[list[Any]] = [
        list(b1.items[init_k * i : init_k * (i + 1)]) for i in range(n_prompts)
    ]
    total_completion_tokens = int(
        b1.stats.get("completion_tokens", 0) or sum(getattr(it, "completion_tokens", 0) for it in b1.items)
    )

    esc_indices: list[int] = []
    esc_set: set[int] = set()
    rem_k = k_eff - init_k
    if adaptive and rem_k > 0:
        for idx in range(n_prompts):
            samps = samples_by_prompt[idx]
            all_valid = all(bool(val_fn(s)) for s in samps)
            extracted = [ext_fn(str(getattr(s, "text", s) or "")) for s in samps]
            all_have_ans = all(a is not None for a in extracted)
            agreed = all_valid and all_have_ans and all(eq_fn(extracted[0], a) for a in extracted[1:])
            if not agreed:
                esc_indices.append(idx)
                esc_set.add(idx)

        if esc_indices:
            stage2_prompts = [prompts[idx] for idx in esc_indices for _ in range(rem_k)]
            b2 = engine.generate_batch(
                stage2_prompts,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=int(seed) + max(1000, n_prompts * init_k),
                **gen_kw,
            )
            for pos, orig_idx in enumerate(esc_indices):
                samples_by_prompt[orig_idx].extend(b2.items[rem_k * pos : rem_k * (pos + 1)])
            total_completion_tokens += int(
                b2.stats.get("completion_tokens", 0) or sum(getattr(it, "completion_tokens", 0) for it in b2.items)
            )

    selections = [
        select_majority_answer(
            samples_by_prompt[i],
            extract_fn=ext_fn,
            equiv_fn=eq_fn,
            valid_fn=val_fn,
            incomplete_weight=incomplete_weight,
            escalated=(i in esc_set),
        )
        for i in range(n_prompts)
    ]
    elapsed = time.perf_counter() - t0
    return SelfConsistencyBatchResult(
        selections=selections,
        answers=[s.answer for s in selections],
        items=[s.selected_item for s in selections],
        stats={
            "prompts": n_prompts,
            "k": k_eff,
            "initial_k": init_k,
            "adaptive": bool(adaptive),
            "escalated_prompts": len(esc_indices),
            "escalation_rate": round(float(len(esc_indices)) / float(n_prompts), 6),
            "total_samples": sum(len(x) for x in samples_by_prompt),
            "completion_tokens": int(total_completion_tokens),
            "elapsed_sec": round(elapsed, 4),
        },
    )


__all__ = [
    "SelfConsistencyBatchResult",
    "SelfConsistencySelection",
    "canonical_json_key",
    "extract_choice_letter",
    "generate_self_consistent",
    "select_majority_answer",
    "select_majority_json",
]