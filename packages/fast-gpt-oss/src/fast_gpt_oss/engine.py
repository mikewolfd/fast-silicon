"""High-throughput `openai/gpt-oss-20b` engine for Apple Silicon M4 & M5.

Combines:
  - `FastGptOssModel` (`MXFP4-Q8` default with offline `MXFP4-Q4` fallback, streamed
    per-shard loading, Apple10 `NAX` + Apple9 Metal 4 `MPP::matmul2d` Hybrid MoE,
    Hydragen v2 zero-prefix-clone SDPA, circular SWA-128, asymmetric `o200k` vocab
    with `_cold_lm_head` fallback)
  - Shared-prefix LRU KV cache (`LCP` split across waves and repeat calls)
  - Canonical OpenAI Harmony multi-channel framing + persistent `in_final_channel`
    stop tracking + two-phase batched budgeted reasoning (`wave_bs=8..128`)
  - `fast_gpt_oss.grammar.GrammarHelper` defaulting to spaced `(", ", ": ")` JSON +
    `WhitespaceGuard`, lazy 4-byte `int32` argmax fast-path, `-5.0` trailing 4-gram
    repeat-loop breaker, upfront + per-step `compute_ff_tokens()` skipping, dynamic
    slot compaction, and automatic `B=1` tail handoff (`extract_b1_cache` -> `forward_b1`).
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import gc
import json
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import warnings

import numpy as np

from fast_shared.grammar import (
    compute_loop_breaker_penalties,
    normalize_json_schema,
    sample_unconstrained,
    validate_json_against_schema,
)

from .grammar import GrammarHelper
from .model import FastGptOssModel
from .paths import DEFAULT_HF_REPO, GptOssPaths, resolve_paths
from .tokenizer import (
    DEFAULT_KEPT_VOCAB_SIZE,
    FastGptOssTokenizer,
    parse_harmony_output,
)

PromptInput = str | Sequence[dict[str, Any]]

_EFFORT_DEFAULT_BUDGET = {
    "skip": 0,
    "none": 0,
    "low": 48,
    "medium": 80,
    "high": 128,
}

_REPEAT_4GRAM_WINDOW = 64
_REPEAT_4GRAM_MIN_HITS = 2
_REPEAT_4GRAM_PENALTY = 5.0


@dataclass
class GenerationItem:
    text: str
    parsed: Any | None
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    valid_json: bool
    valid_schema: bool
    finish_reason: str
    reasoning: str | None = None
    telemetry: Any | None = None


@dataclass
class GenerationBatchResult:
    items: list[GenerationItem]
    stats: dict[str, Any] = field(default_factory=dict)


def _find_repeated_4gram_penalties(
    recent_ids: Sequence[int] | None,
    *,
    window: int = _REPEAT_4GRAM_WINDOW,
    min_prior_hits: int = _REPEAT_4GRAM_MIN_HITS,
    base_penalty: float = _REPEAT_4GRAM_PENALTY,
    stop_token_ids: frozenset[int] = frozenset(),
) -> dict[int, float]:
    """Delegate to shared `compute_loop_breaker_penalties` (periodic `p=1..8` + trailing 4-gram check)."""
    return compute_loop_breaker_penalties(
        recent_ids,
        window=window,
        min_prior_hits=min_prior_hits,
        base_penalty=base_penalty,
        stop_token_ids=stop_token_ids,
    )


def split_lcp_ids(prompt_ids_list: Sequence[Sequence[int]]) -> tuple[list[int], list[list[int]]]:
    """Extract longest common token prefix leaving at least 1 suffix token per prompt."""
    if not prompt_ids_list:
        return [], []
    if len(prompt_ids_list) == 1:
        return [], [list(prompt_ids_list[0])]

    min_len = min(len(x) for x in prompt_ids_list)
    if min_len <= 1:
        return [], [list(x) for x in prompt_ids_list]

    first = prompt_ids_list[0]
    lcp = min_len - 1
    for other in prompt_ids_list[1:]:
        cur = 0
        limit = min(lcp, len(other) - 1)
        while cur < limit and first[cur] == other[cur]:
            cur += 1
        lcp = cur
        if lcp == 0:
            break

    if lcp == 0:
        return [], [list(x) for x in prompt_ids_list]
    prefix = list(first[:lcp])
    suffixes = [list(x[lcp:]) for x in prompt_ids_list]
    return prefix, suffixes


def resolve_reasoning_params(
    *,
    has_schema: bool,
    reasoning_effort: str | None,
    reasoning_budget: int | None,
    enable_thinking: bool,
    default_effort: str = "low",
) -> tuple[str, int]:
    """Resolve effective `(reasoning_effort, analysis_budget)` for Harmony generation."""
    if has_schema:
        req_bud = None if reasoning_budget is None else int(reasoning_budget)
        if req_bud is not None and req_bud <= 0:
            return "skip", 0
        if enable_thinking:
            eff = str(reasoning_effort or default_effort or "low").lower().strip()
            if eff in ("skip", "none"):
                eff = "low"
            bud = req_bud if req_bud is not None else _EFFORT_DEFAULT_BUDGET.get(eff, 48)
            return eff, max(1, int(bud))
        if reasoning_effort is not None:
            eff = str(reasoning_effort).lower().strip()
            if eff in ("skip", "none"):
                return eff, 0
            bud = req_bud if req_bud is not None else _EFFORT_DEFAULT_BUDGET.get(eff, 48)
            return eff, max(1, int(bud))
        if req_bud is not None and req_bud > 0:
            return "low", req_bud
        return "skip", 0

    if enable_thinking and reasoning_effort in (None, "skip", "none"):
        eff = "low"
    else:
        eff = str(reasoning_effort or default_effort or "low").lower().strip()
    if eff not in _EFFORT_DEFAULT_BUDGET:
        raise ValueError(f"Unsupported reasoning_effort={reasoning_effort!r}")
    bud = int(reasoning_budget) if reasoning_budget is not None else 0
    return eff, max(0, bud)


def recommended_max_wave_batch(memory_bytes: int | None) -> int:
    """Return safe wave batch ceiling for unified memory capacity (`16/24/32/48+ GiB`)."""
    if memory_bytes is None or memory_bytes <= 0:
        return 64
    mem_gib = float(memory_bytes) / float(1 << 30)
    if mem_gib <= 18.0:
        return 8
    if mem_gib <= 26.0:
        return 16
    if mem_gib <= 36.0:
        return 32
    return 64


def select_gpt_oss_plan(
    mode: str | None,
    n_docs: int,
    batch_size: int | None,
    *,
    memory_bytes: int | None = None,
) -> tuple[str, int]:
    """Choose `(engine_path, wave_batch_size)` across `auto`, `b1`, and `batched` with `<=24 GiB` clamping."""
    eff_mode = (mode or "auto").lower().strip()
    if eff_mode not in ("auto", "b1", "batched", "batch"):
        raise ValueError(f"Unsupported mode={mode!r}; expected auto|b1|batched")
    if eff_mode == "b1":
        return "b1", 1
    if eff_mode == "auto" and n_docs <= 1:
        return "b1", 1

    mem_cap = recommended_max_wave_batch(memory_bytes)
    default_wb = min(max(n_docs, 1), mem_cap)
    wb = int(batch_size) if batch_size is not None else default_wb
    wb = max(1, wb)

    if memory_bytes is not None and memory_bytes > 0:
        mem_gib = float(memory_bytes) / float(1 << 30)
        if mem_gib <= 26.0 and wb > mem_cap:
            warnings.warn(
                f"Clamping wave batch_size from {wb} to {mem_cap} on {mem_gib:.1f} GiB unified-memory host "
                f"to avoid OS swapout thrashing (gpt-oss-20b MXFP4 weights ~11.3 GiB).",
                UserWarning,
                stacklevel=2,
            )
            wb = mem_cap
    return "batched", wb


class FastGptOssEngine:
    """Thread-safe hardware-limit `openai/gpt-oss-20b` inference engine."""

    DEFAULT_CACHE_LIMIT_BYTES: int = 256 * 1024 * 1024

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        model_dir: str | Path | None = None,
        vocab_out: int | None = None,
        default_reasoning_effort: str = "low",
        default_mode: str = "auto",
        min_prefix_cache_tokens: int = 8,
        local_files_only: bool = True,
        warmup: bool = True,
        max_prefix_cache_entries: int = 4,
        max_grammar_cache_entries: int = 64,
    ) -> None:
        self.paths: GptOssPaths = resolve_paths(home)
        resolved_dir = self.paths.resolve_model_dir(
            model_dir,
            repo_id=DEFAULT_HF_REPO,
            local_files_only=local_files_only,
        )
        tok_json = self.paths.resolve_tokenizer_json(resolved_dir)
        self.model_dir = Path(resolved_dir)
        self.model_name = DEFAULT_HF_REPO
        self.default_reasoning_effort = default_reasoning_effort
        self.default_mode = default_mode
        self.min_prefix_cache_tokens = int(min_prefix_cache_tokens)
        self._lock = threading.RLock()

        import mlx.core as mx

        self._prev_cache_limit: int | None = None
        self._prev_wired_limit: int | None = None
        self._wired_limit_restored = False
        self._host_memory_bytes: int = 0
        self._max_rec_ws_bytes: int = 0
        set_cache = getattr(mx, "set_cache_limit", None)
        if callable(set_cache):
            try:
                self._prev_cache_limit = int(set_cache(self.DEFAULT_CACHE_LIMIT_BYTES))
            except Exception:
                self._prev_cache_limit = None
        try:
            dev_info = mx.device_info()
            self._host_memory_bytes = int(dev_info.get("memory_size", 0))
            self._max_rec_ws_bytes = int(dev_info.get("max_recommended_working_set_size", 0))
            set_wired = getattr(mx, "set_wired_limit", None)
            if callable(set_wired) and self._max_rec_ws_bytes > 0:
                if 0 < self._host_memory_bytes <= (24 << 30):
                    wired_target = min(self._max_rec_ws_bytes, int(self._host_memory_bytes * 0.72))
                else:
                    wired_target = self._max_rec_ws_bytes
                self._prev_wired_limit = int(set_wired(wired_target))
        except Exception:
            self._prev_wired_limit = None

        try:
            self.tokenizer = FastGptOssTokenizer(tok_json)
            self.model = FastGptOssModel(
                self.model_dir,
                vocab_out=vocab_out,
                vocab_remap_npz=self.paths.vocab_remap_npz,
                tokenizer_json_path=tok_json,
                compile_kernels=True,
            )
            self._prefix_cache: OrderedDict[tuple[int, ...], dict[str, Any]] = OrderedDict()
            self._grammar_cache: OrderedDict[tuple[str, bool | None, int | None], GrammarHelper] = OrderedDict()
            self._max_prefix_cache = max(1, int(max_prefix_cache_entries))
            self._max_grammar_cache = max(1, int(max_grammar_cache_entries))

            self._final_channel_seq: tuple[int, ...] = self.tokenizer.final_channel_seq
            self._start_assistant_seq: tuple[int, ...] = tuple(self.tokenizer.encode("<|start|>assistant"))
            self._trans_ids = self.tokenizer.encode("<|end|><|start|>assistant<|channel|>final<|message|>")
            self._dummy_pad_id = int(self.tokenizer.encode(" ")[0]) if self.tokenizer.encode(" ") else 0

            if warmup:
                self.warmup()
        except Exception:
            self._restore_wired_limit()
            raise

    @property
    def default_max_b_slots(self) -> int:
        return recommended_max_wave_batch(getattr(self, "_host_memory_bytes", 0) or None)

    def _restore_wired_limit(self) -> None:
        if getattr(self, "_wired_limit_restored", True):
            return
        self._wired_limit_restored = True
        prev_cl = getattr(self, "_prev_cache_limit", None)
        if prev_cl is not None:
            try:
                import mlx.core as mx

                set_cache = getattr(mx, "set_cache_limit", None)
                if callable(set_cache):
                    set_cache(prev_cl)
            except Exception:
                pass
            self._prev_cache_limit = None
        prev = getattr(self, "_prev_wired_limit", None)
        if prev is not None:
            try:
                import mlx.core as mx

                set_wired = getattr(mx, "set_wired_limit", None)
                if callable(set_wired):
                    set_wired(prev)
            except Exception:
                pass
            self._prev_wired_limit = None

    def close(self) -> None:
        with self._lock:
            self._prefix_cache.clear()
            self._grammar_cache.clear()
            gc.collect()
            self._restore_wired_limit()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass

    def __enter__(self) -> "FastGptOssEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def warmup(self) -> None:
        import mlx.core as mx

        with self._lock:
            c = self.model.make_b1_cache(4)
            lg = self.model.forward_b1(mx.array([[self._dummy_pad_id, self._dummy_pad_id]], dtype=mx.int32), c, last_only=True)
            mx.eval(lg)
            self.model._last_hf_2d = None
            del c, lg
            mx.clear_cache()

    def _get_grammar(
        self,
        json_schema: str | dict[str, Any] | Any | None,
        *,
        whitespace_flexible: bool | None = None,
        max_whitespace_run: int | None = None,
    ) -> GrammarHelper | Any | None:
        if json_schema is None:
            return None
        if not isinstance(json_schema, (str, dict)):
            return json_schema
        schema_str = normalize_json_schema(json_schema)
        norm_ws: bool | None = None if whitespace_flexible is None else bool(whitespace_flexible)
        key = (schema_str, norm_ws, max_whitespace_run)
        helper = self._grammar_cache.get(key)
        if helper is None:
            helper = GrammarHelper(
                self.tokenizer,
                schema_str,
                new_to_old=self.model.new_to_old,
                whitespace_flexible=norm_ws,
                max_whitespace_run=max_whitespace_run,
                eos_token_id=self.tokenizer.return_token_id,
            )
            self._grammar_cache[key] = helper
            if len(self._grammar_cache) > self._max_grammar_cache:
                self._grammar_cache.popitem(last=False)
        else:
            self._grammar_cache.move_to_end(key)
        return helper

    get_grammar = _get_grammar

    def _get_or_compute_prefix(self, prefix_ids: list[int]) -> tuple[dict[str, Any], float, bool]:
        if len(prefix_ids) < self.min_prefix_cache_tokens:
            return self.model.compute_shared_prefix([]), 0.0, False
        key = tuple(prefix_ids)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            self._prefix_cache.move_to_end(key)
            return cached, 0.0, True
        t0 = time.perf_counter()
        sp = self.model.compute_shared_prefix(prefix_ids)
        dt = time.perf_counter() - t0
        self._prefix_cache[key] = sp
        if len(self._prefix_cache) > self._max_prefix_cache:
            self._prefix_cache.popitem(last=False)
        return sp, dt, False

    def _match_cached_prefix_for_single(self, ids: list[int]) -> tuple[dict[str, Any] | None, list[int]]:
        """Return longest cached shared prefix `sp` that strictly prefixes `ids`, if any."""
        best_len = 0
        best_key: tuple[int, ...] | None = None
        for key in self._prefix_cache:
            klen = len(key)
            if 0 < klen < len(ids) and klen > best_len and tuple(ids[:klen]) == key:
                best_len = klen
                best_key = key
        if best_key is None:
            return None, ids
        self._prefix_cache.move_to_end(best_key)
        return self._prefix_cache[best_key], ids[best_len:]

    def encode_prompts(
        self,
        prompts: Sequence[PromptInput],
        *,
        system_prompt: str | None = None,
        use_chat_template: bool = True,
        reasoning_effort: str = "low",
    ) -> list[list[int]]:
        raw_texts: list[str] = []
        for item in prompts:
            if isinstance(item, str):
                if use_chat_template:
                    msgs: list[dict[str, Any]] = []
                    if system_prompt:
                        msgs.append({"role": "system", "content": system_prompt})
                    msgs.append({"role": "user", "content": item})
                    raw_texts.append(
                        self.tokenizer.format_chat(
                            msgs,
                            add_generation_prompt=True,
                            reasoning_effort=reasoning_effort,
                        )
                    )
                else:
                    raw_texts.append(f"{system_prompt}\n\n{item}" if system_prompt else item)
            else:
                msgs = list(item)
                if system_prompt and msgs and msgs[0].get("role") not in ("system", "developer"):
                    msgs = [{"role": "system", "content": system_prompt}] + msgs
                raw_texts.append(
                    self.tokenizer.format_chat(
                        msgs,
                        add_generation_prompt=True,
                        reasoning_effort=reasoning_effort,
                    )
                )
        return self.tokenizer.encode_batch(raw_texts, add_special_tokens=False)

    def _should_use_gpu_top128(self, lg_2d: Any, *, temperature: float, top_p: float) -> bool:
        import mlx.core as mx

        return bool(
            0.0 < float(temperature) <= 100.0
            and 0.0 < float(top_p) < 1.0
            and len(lg_2d.shape) == 2
            and int(lg_2d.shape[-1]) > 128
            and hasattr(mx, "argpartition")
            and hasattr(mx, "take_along_axis")
            and hasattr(mx, "logsumexp")
        )

    def _eval_logits_for_sampling(
        self,
        lg_2d: Any,
        *,
        temperature: float,
        top_p: float,
        has_grammar: bool = False,
    ) -> dict[str, Any]:
        import mlx.core as mx

        hf_2d = getattr(self.model, "_last_hf_2d", None)
        if hasattr(self.model, "_last_hf_2d"):
            self.model._last_hf_2d = None
        keep_hf = hf_2d if (has_grammar and hf_2d is not None and getattr(self.model, "_cold_lm_head", None) is not None) else None

        if self._should_use_gpu_top128(lg_2d, temperature=temperature, top_p=top_p):
            k_cand = 128
            lg_f32 = lg_2d.astype(mx.float32)
            inv_temp = float(1.0 / float(temperature))
            scaled = lg_f32 * inv_temp
            lse_mx = mx.logsumexp(scaled, axis=-1)
            top_idx_mx = mx.argpartition(-scaled, kth=k_cand - 1, axis=-1)[:, :k_cand]
            top_scaled_mx = mx.take_along_axis(scaled, top_idx_mx, axis=-1)
            if keep_hf is not None:
                mx.eval(top_idx_mx, top_scaled_mx, lse_mx, keep_hf)
            else:
                mx.eval(top_idx_mx, top_scaled_mx, lse_mx)
            return {
                "mode": "top128",
                "lg_2d": lg_2d,
                "hf_2d": keep_hf,
                "top_idx_np": np.asarray(top_idx_mx, dtype=np.int32),
                "top_scaled_np": np.asarray(top_scaled_mx, dtype=np.float32),
                "lse_np": np.asarray(lse_mx, dtype=np.float32),
            }

        if hasattr(lg_2d, "ndim") and hasattr(lg_2d, "dtype"):
            from fast_shared.argmax import gpu_argmax_nan

            top1_mx = gpu_argmax_nan(lg_2d)
        else:
            top1_mx = mx.argmax(lg_2d, axis=-1)
        if keep_hf is not None:
            mx.eval(top1_mx, keep_hf)
        else:
            mx.eval(top1_mx)
        top1_list = [int(t) for t in top1_mx.tolist()]
        if any(t < 0 for t in top1_list):
            eos_old = int(getattr(self.tokenizer, "return_token_id", 0))
            eos_pruned = self._map_old_to_pruned_idx(eos_old, self.model.new_to_old)
            safe_fallback = eos_pruned if eos_pruned >= 0 else 0
            top1_list = [safe_fallback if t < 0 else t for t in top1_list]
        return {
            "mode": "top1",
            "lg_2d": lg_2d,
            "hf_2d": keep_hf,
            "top1_cpu": top1_list,
        }

    def _map_old_to_pruned_idx(self, old_tid: int, n2o: np.ndarray) -> int:
        o2n = getattr(self.model, "old_to_new", None)
        if o2n is not None and 0 <= old_tid < len(o2n) and n2o is self.model.new_to_old:
            return int(o2n[old_tid])
        matches = np.flatnonzero(n2o == int(old_tid))
        return int(matches[0]) if matches.size > 0 else -1

    def _sample_row_from_eval(
        self,
        eval_pack: dict[str, Any],
        row_idx: int,
        *,
        grammar: Any | None,
        matcher: Any | None,
        temperature: float,
        top_p: float,
        rng: np.random.Generator,
        stats_counters: dict[str, int] | None = None,
        recent_ids: Sequence[int] | None = None,
    ) -> tuple[int, bool]:
        import mlx.core as mx

        lg_2d = eval_pack["lg_2d"]
        step_was_masked = [False]
        tok = getattr(self, "tokenizer", None)
        raw_stops = getattr(tok, "final_stop_token_ids", None) if tok is not None else None
        stop_ids = (
            raw_stops
            if isinstance(raw_stops, frozenset)
            else (frozenset(raw_stops) if raw_stops is not None else frozenset())
        )
        penalties = compute_loop_breaker_penalties(
            recent_ids,
            window=_REPEAT_4GRAM_WINDOW,
            min_prior_hits=_REPEAT_4GRAM_MIN_HITS,
            base_penalty=_REPEAT_4GRAM_PENALTY,
            stop_token_ids=stop_ids,
        )
        n2o = getattr(grammar, "new_to_old", self.model.new_to_old) if grammar is not None else self.model.new_to_old
        cached_row: list[np.ndarray | None] = [None]

        def _lazy_full_row() -> np.ndarray:
            if cached_row[0] is not None:
                return cached_row[0]
            if stats_counters is not None:
                stats_counters["full_vocab_logit_copies"] = (
                    stats_counters.get("full_vocab_logit_copies", 0) + 1
                )
                if not step_was_masked[0]:
                    stats_counters["unmasked_full_vocab_logit_copies"] = (
                        stats_counters.get("unmasked_full_vocab_logit_copies", 0) + 1
                    )
            arr = np.array(lg_2d[row_idx].astype(mx.float32), copy=bool(penalties))
            if penalties:
                for old_tid, pen in penalties.items():
                    if len(arr) > len(n2o) and 0 <= old_tid < len(arr):
                        arr[old_tid] -= pen
                    else:
                        pj = self._map_old_to_pruned_idx(old_tid, n2o)
                        if 0 <= pj < len(arr):
                            arr[pj] -= pen
            cached_row[0] = arr
            return arr

        def _lazy_full_vocab_row() -> np.ndarray:
            compute_fn = getattr(self.model, "compute_full_logits_1d", None)
            if callable(compute_fn):
                raw_pruned = np.array(lg_2d[row_idx].astype(mx.float32), copy=False)
                full_arr = compute_fn(raw_pruned, row_idx=row_idx, hf_2d=eval_pack.get("hf_2d"))
                if full_arr is not None:
                    if penalties:
                        full_arr = np.array(full_arr, copy=True)
                        for old_tid, pen in penalties.items():
                            if 0 <= old_tid < len(full_arr):
                                full_arr[old_tid] -= pen
                    return full_arr
            return _lazy_full_row()

        has_cold_head = getattr(self.model, "_cold_lm_head", None) is not None
        full_logits_cb = _lazy_full_vocab_row if has_cold_head else None

        if eval_pack["mode"] == "top1":
            top1_new = int(eval_pack["top1_cpu"][row_idx])
            if penalties:
                top1_old = int(n2o[top1_new]) if 0 <= top1_new < len(n2o) else -1
                if top1_old in penalties or temperature > 0.0:
                    step_was_masked[0] = grammar is not None
                    biased = _lazy_full_row()
                    top1_new = int(np.argmax(biased))
            if grammar is not None:
                step_was_masked[0] = True
                return grammar.pick_valid_token(
                    matcher,
                    _lazy_full_row,
                    top1_new,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                    fallback_to_full_vocab=True,
                    full_logits_1d=full_logits_cb,
                )
            if temperature <= 0.0:
                return int(self.model.new_to_old[top1_new]), False
            row_np = _lazy_full_row()
            return (
                int(
                    sample_unconstrained(
                        row_np,
                        self.model.new_to_old,
                        top1_new,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                    )
                ),
                False,
            )

        raw_idx = eval_pack["top_idx_np"][row_idx]
        raw_sc = eval_pack["top_scaled_np"][row_idx].astype(np.float64)
        lse_val = float(eval_pack["lse_np"][row_idx])
        ord128 = np.lexsort((raw_idx, -raw_sc))
        s_sorted = raw_sc[ord128]
        idx_sorted = raw_idx[ord128]
        top1_new = int(idx_sorted[0])
        m_max = float(s_sorted[0])
        s_kth = float(s_sorted[-1])

        cand_old_all = n2o[idx_sorted]
        has_top128_penalty = bool(penalties) and any(int(t) in penalties for t in cand_old_all)

        if not has_top128_penalty and np.isfinite(m_max) and np.isfinite(lse_val):
            w_128 = np.exp(s_sorted - m_max)
            sum_top128 = float(w_128.sum())
            z_all = float(np.exp(lse_val - m_max))
            z_rem = max(0.0, z_all - sum_top128)
            n_tail = max(0, int(lg_2d.shape[-1]) - len(idx_sorted))
            tail_cap_kth = float(n_tail) * float(np.exp(s_kth - m_max))
            z_tail_hi = min(z_rem + 1e-6 * max(z_all, 1.0), tail_cap_kth)

            if grammar is not None:
                cand_old = cand_old_all
                mask_owner = grammar if hasattr(grammar, "_mask_ptr") else getattr(grammar, "_gh_std", None)
                if mask_owner is not None and hasattr(matcher, "unsafe_compute_mask_ptr"):
                    lock_ctx = getattr(mask_owner, "_mask_lock", None)
                    if lock_ctx is not None:
                        with lock_ctx:
                            matcher.unsafe_compute_mask_ptr(mask_owner._mask_ptr, mask_owner._mask_nbytes)
                            cand_ok = ((mask_owner._mask_buf[cand_old >> 5] >> (cand_old & 31)) & 1) != 0
                    else:
                        matcher.unsafe_compute_mask_ptr(mask_owner._mask_ptr, mask_owner._mask_nbytes)
                        cand_ok = ((mask_owner._mask_buf[cand_old >> 5] >> (cand_old & 31)) & 1) != 0
                elif hasattr(matcher, "validate_tokens"):
                    cand_ok = np.array(
                        [bool(matcher.validate_tokens([int(t)])) for t in cand_old],
                        dtype=bool,
                    )
                else:
                    cand_ok = None

                if cand_ok is not None:
                    step_was_masked[0] = not bool(cand_ok[0])
                if cand_ok is not None and bool(np.any(cand_ok)):
                    was_masked = not bool(cand_ok[0])
                    valid_new = idx_sorted[cand_ok]
                    valid_s = s_sorted[cand_ok]
                    valid_w = w_128[cand_ok]
                    m_v = len(valid_new)
                    cum_w = np.cumsum(valid_w)
                    z_top_valid = float(cum_w[-1])
                    if z_top_valid > 0.0:
                        z_lo = z_top_valid
                        z_hi = z_top_valid + z_tail_hi
                        cut_lo = max(1, int(np.searchsorted(cum_w / z_lo, top_p, side="right")) + 1)
                        cut_hi = max(1, int(np.searchsorted(cum_w / z_hi, top_p, side="right")) + 1)
                        if cut_lo == cut_hi and cut_hi <= m_v and float(valid_s[cut_lo - 1]) > s_kth:
                            keep_new = valid_new[:cut_lo]
                            sub_p = valid_w[:cut_lo]
                            sub_p = sub_p / float(sub_p.sum())
                            picked_new = int(rng.choice(keep_new, p=sub_p))
                            if stats_counters is not None:
                                stats_counters["top128_fast_hits"] = (
                                    stats_counters.get("top128_fast_hits", 0) + 1
                                )
                            return int(n2o[picked_new]), was_masked

                return grammar.pick_valid_token(
                    matcher,
                    _lazy_full_row,
                    top1_new,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                    fallback_to_full_vocab=True,
                    full_logits_1d=full_logits_cb,
                )

            cum_w = np.cumsum(w_128)
            z_lo = sum_top128
            z_hi = sum_top128 + z_tail_hi
            if z_lo > 0.0:
                cut_lo = max(1, int(np.searchsorted(cum_w / z_lo, top_p, side="right")) + 1)
                cut_hi = max(1, int(np.searchsorted(cum_w / z_hi, top_p, side="right")) + 1)
                if cut_lo == cut_hi and cut_hi <= len(idx_sorted) and float(s_sorted[cut_lo - 1]) > s_kth:
                    keep_new = idx_sorted[:cut_lo]
                    sub_p = w_128[:cut_lo]
                    sub_p = sub_p / float(sub_p.sum())
                    picked_new = int(rng.choice(keep_new, p=sub_p))
                    if stats_counters is not None:
                        stats_counters["top128_fast_hits"] = (
                            stats_counters.get("top128_fast_hits", 0) + 1
                        )
                    return int(self.model.new_to_old[picked_new]), False

        if has_top128_penalty:
            step_was_masked[0] = grammar is not None
            biased_row = _lazy_full_row()
            top1_new = int(np.argmax(biased_row))
        if grammar is not None:
            return grammar.pick_valid_token(
                matcher,
                _lazy_full_row,
                top1_new,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
                fallback_to_full_vocab=True,
                full_logits_1d=full_logits_cb,
            )
        row_np = _lazy_full_row()
        return (
            int(
                sample_unconstrained(
                    row_np,
                    self.model.new_to_old,
                    top1_new,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                )
            ),
            False,
        )

    def _advance_channel_and_check_stop(
        self,
        tok_id: int,
        out_ids: list[int],
        in_final_channel: bool,
    ) -> tuple[bool, bool]:
        """Advance persistent per-slot `in_final_channel` state on `tok_id` (already appended to `out_ids`)
        and return `(in_final_channel, should_stop)`."""
        if not in_final_channel:
            fseq = getattr(self, "_final_channel_seq", None) or getattr(self.tokenizer, "final_channel_seq", ())
            n_f = len(fseq)
            if n_f > 0 and tok_id == fseq[-1] and len(out_ids) >= n_f:
                if tuple(out_ids[-n_f:]) == fseq:
                    in_final_channel = True
        if tok_id in self.tokenizer.analysis_stop_token_ids:
            return in_final_channel, True
        if in_final_channel and tok_id == self.tokenizer.end_token_id:
            return in_final_channel, True
        return in_final_channel, False

    def _strip_trailing_final_header(self, ids: list[int]) -> list[int]:
        fseq = tuple(getattr(self, "_final_channel_seq", None) or getattr(self.tokenizer, "final_channel_seq", ()))
        n_f = len(fseq)
        if n_f <= 0 or len(ids) < n_f or tuple(ids[-n_f:]) != fseq:
            return ids
        trimmed = ids[:-n_f]
        sa_seq = getattr(self, "_start_assistant_seq", None)
        if sa_seq is None and hasattr(self.tokenizer, "encode"):
            try:
                sa_seq = tuple(self.tokenizer.encode("<|start|>assistant"))
            except Exception:
                sa_seq = ()
        if sa_seq and len(trimmed) >= len(sa_seq) and tuple(trimmed[-len(sa_seq) :]) == tuple(sa_seq):
            trimmed = trimmed[: -len(sa_seq)]
        return trimmed

    def _is_unconstrained_stop(self, tok_id: int, out_ids: list[int], in_final_channel: bool) -> bool:
        if tok_id in self.tokenizer.analysis_stop_token_ids:
            return True
        if tok_id == self.tokenizer.end_token_id:
            if in_final_channel:
                return True
            fseq = getattr(self, "_final_channel_seq", None) or getattr(self.tokenizer, "final_channel_seq", ())
            n_f = len(fseq)
            if n_f > 0 and len(out_ids) >= n_f:
                f0 = fseq[0]
                for k in range(len(out_ids) - n_f, -1, -1):
                    if out_ids[k] == f0 and tuple(out_ids[k : k + n_f]) == fseq:
                        return True
        return False

    @staticmethod
    def _doc_upfront_ff(doc_grammar: Any, doc_idx: int) -> list[int]:
        if doc_grammar is None:
            return []
        up_fn = getattr(doc_grammar, "upfront_ff_for_doc", None)
        if callable(up_fn):
            return list(up_fn(doc_idx))
        return list(getattr(doc_grammar, "upfront_ff", ()))

    @staticmethod
    def _new_doc_matcher(doc_grammar: Any, doc_idx: int) -> Any:
        if doc_grammar is None:
            return None
        by_doc = getattr(doc_grammar, "new_matcher_for_doc", None)
        if callable(by_doc):
            return by_doc(doc_idx)
        return doc_grammar.new_matcher()

    def _run_b1_single(
        self,
        prompt_ids: list[int],
        *,
        grammar: GrammarHelper | Any | None,
        per_doc_grammars: Sequence[str | dict[str, Any] | Any | None] | None = None,
        doc_index: int = 0,
        whitespace_flexible: bool | None = None,
        shared_prefix: dict[str, Any] | None,
        prefix_len: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        rng: np.random.Generator,
        started_in_final: bool,
        analysis_budget: int,
    ) -> dict[str, Any]:
        import mlx.core as mx

        eff_grammar = grammar
        if per_doc_grammars is not None and len(per_doc_grammars) > 0:
            sel = per_doc_grammars[doc_index if doc_index < len(per_doc_grammars) else 0]
            if sel is not None:
                eff_grammar = self._get_grammar(sel, whitespace_flexible=whitespace_flexible)

        suf = list(prompt_ids[prefix_len:])
        ff_skipped = 0
        mask_hits = 0
        fallbacks = 0
        steps = 0
        ana_ids: list[int] = []
        samp_stats: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}

        t0 = time.perf_counter()
        ff0_est = len(self._doc_upfront_ff(eff_grammar, doc_index)) if eff_grammar is not None else 0
        max_out_cap = max(max_tokens, ff0_est) + len(getattr(self, "_trans_ids", ())) + 32
        if shared_prefix is not None and prefix_len > 0:
            c1 = self.model.clone_b1_cache_from_prefix(
                shared_prefix, prefix_len + len(suf) + analysis_budget + max_out_cap
            )
        else:
            c1 = self.model.make_b1_cache(len(suf) + analysis_budget + max_out_cap)

        t_dec0 = time.perf_counter()
        t_pp = 0.0

        # Case 1: Two-phase budgeted Harmony reasoning (`analysis_budget > 0`) for constrained & unconstrained!
        if analysis_budget > 0 and not started_in_final:
            lg_2d = self.model.forward_b1(mx.array(suf, dtype=mx.int32)[None, :], c1, last_only=True)[0]
            eval_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p)
            t_pp = time.perf_counter() - t0
            t_dec0 = time.perf_counter()

            p1_ids: list[int] = []
            pending_last_ana: int | None = None
            already_in_final = False
            stopped_in_p1 = False
            out_ids: list[int] = []
            for _ in range(analysis_budget):
                tok_orig, _ = self._sample_row_from_eval(
                    eval_pack,
                    0,
                    grammar=None,
                    matcher=None,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                    stats_counters=samp_stats,
                    recent_ids=p1_ids,
                )
                lg_2d = None
                eval_pack = None  # type: ignore[assignment]
                if eff_grammar is None and tok_orig == self.tokenizer.call_token_id:
                    out_ids = [tok_orig]
                    stopped_in_p1 = True
                    break
                if tok_orig in self.tokenizer.final_stop_token_ids:
                    break
                p1_ids.append(tok_orig)
                in_fin_p1, _ = self._advance_channel_and_check_stop(tok_orig, p1_ids, False)
                if in_fin_p1:
                    p1_ids = self._strip_trailing_final_header(p1_ids)
                    already_in_final = True
                    pending_last_ana = tok_orig
                    break
                if len(p1_ids) >= analysis_budget:
                    pending_last_ana = tok_orig
                    break
                steps += 1
                lg_2d = self.model.forward_b1(mx.array([[tok_orig]], dtype=mx.int32), c1, last_only=True)[0]
                eval_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p)

            ana_ids = p1_ids
            if stopped_in_p1:
                t_dec = time.perf_counter() - t_dec0
                del c1
                return {
                    "out_ids": out_ids,
                    "ana_ids": ana_ids,
                    "two_phase_final": True,
                    "ff_skipped": 0,
                    "mask_hits": 0,
                    "fallbacks": 0,
                    "prefill_sec": t_pp,
                    "decode_sec": t_dec,
                    "steps": steps,
                    "finished": True,
                    "ff_lm_head_skipped_steps": 0,
                    "top128_fast_hits": samp_stats["top128_fast_hits"],
                    "full_vocab_logit_copies": samp_stats["full_vocab_logit_copies"],
                    "unmasked_full_vocab_logit_copies": samp_stats.get("unmasked_full_vocab_logit_copies", 0),
                }

            inj_prefix = [pending_last_ana] if pending_last_ana is not None else []
            trans_part = [] if already_in_final else list(self._trans_ids)
            if eff_grammar is not None:
                matcher = self._new_doc_matcher(eff_grammar, doc_index)
                ff0 = self._doc_upfront_ff(eff_grammar, doc_index)
                if ff0:
                    ff_skipped += len(ff0)
                chunk = inj_prefix + trans_part + ff0
                lg_2d = self.model.forward_b1(mx.array(chunk, dtype=mx.int32)[None, :], c1, last_only=True)[0]
                eval_pack = self._eval_logits_for_sampling(
                    lg_2d, temperature=temperature, top_p=top_p, has_grammar=True
                )
                out_ids = list(ff0)
                ff_ref = [ff_skipped]
                out_ids, mask_hits, fallbacks, steps_p2 = self._finish_b1_grammar_loop(
                    c1,
                    eval_pack,
                    matcher,
                    eff_grammar,
                    out_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                    mask_hits=mask_hits,
                    fallbacks=fallbacks,
                    ff_skipped_ref=ff_ref,
                    stats_counters=samp_stats,
                )
                ff_skipped = ff_ref[0]
                finished_p2 = bool(matcher.is_stopped())
            else:
                chunk = inj_prefix + trans_part
                lg_2d = self.model.forward_b1(mx.array(chunk, dtype=mx.int32)[None, :], c1, last_only=True)[0]
                eval_pack = self._eval_logits_for_sampling(
                    lg_2d, temperature=temperature, top_p=top_p, has_grammar=False
                )
                out_ids = []
                finished_p2 = False
                in_final_p2 = True
                steps_p2 = 0
                while len(out_ids) < max_tokens:
                    tok_orig, _ = self._sample_row_from_eval(
                        eval_pack,
                        0,
                        grammar=None,
                        matcher=None,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=out_ids,
                    )
                    lg_2d = None
                    eval_pack = None  # type: ignore[assignment]
                    out_ids.append(tok_orig)
                    in_final_p2, should_stop = self._advance_channel_and_check_stop(
                        tok_orig, out_ids, in_final_p2
                    )
                    if should_stop:
                        finished_p2 = True
                        break
                    if len(out_ids) >= max_tokens:
                        break
                    lg_2d = self.model.forward_b1(mx.array([[tok_orig]], dtype=mx.int32), c1, last_only=True)[0]
                    eval_pack = self._eval_logits_for_sampling(
                        lg_2d, temperature=temperature, top_p=top_p, has_grammar=False
                    )
                    steps_p2 += 1

            t_dec = time.perf_counter() - t_dec0
            del c1
            return {
                "out_ids": out_ids,
                "ana_ids": ana_ids,
                "two_phase_final": True,
                "ff_skipped": ff_skipped,
                "mask_hits": mask_hits,
                "fallbacks": fallbacks,
                "prefill_sec": t_pp,
                "decode_sec": t_dec,
                "steps": steps + steps_p2,
                "finished": finished_p2,
                "ff_lm_head_skipped_steps": 0,
                "top128_fast_hits": samp_stats["top128_fast_hits"],
                "full_vocab_logit_copies": samp_stats["full_vocab_logit_copies"],
                "unmasked_full_vocab_logit_copies": samp_stats.get("unmasked_full_vocab_logit_copies", 0),
            }

        # Case 2: Schema-constrained starting immediately in `final` channel (`skip`).
        if eff_grammar is not None:
            matcher = self._new_doc_matcher(eff_grammar, doc_index)
            ff0 = self._doc_upfront_ff(eff_grammar, doc_index)
            if ff0:
                ff_skipped += len(ff0)
            suf_eff = suf + ff0
            lg_2d = self.model.forward_b1(mx.array(suf_eff, dtype=mx.int32)[None, :], c1, last_only=True)[0]
            eval_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p, has_grammar=True)
            t_pp = time.perf_counter() - t0
            t_dec0 = time.perf_counter()
            ff_ref = [ff_skipped]
            out_ids, mask_hits, fallbacks, steps = self._finish_b1_grammar_loop(
                c1,
                eval_pack,
                matcher,
                eff_grammar,
                list(ff0),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
                mask_hits=mask_hits,
                fallbacks=fallbacks,
                ff_skipped_ref=ff_ref,
                stats_counters=samp_stats,
            )
            t_dec = time.perf_counter() - t_dec0
            del c1
            return {
                "out_ids": out_ids,
                "ana_ids": ana_ids,
                "ff_skipped": ff_ref[0],
                "mask_hits": mask_hits,
                "fallbacks": fallbacks,
                "prefill_sec": t_pp,
                "decode_sec": t_dec,
                "steps": steps,
                "finished": bool(matcher.is_stopped()),
                "ff_lm_head_skipped_steps": 0,
                "top128_fast_hits": samp_stats["top128_fast_hits"],
                "full_vocab_logit_copies": samp_stats["full_vocab_logit_copies"],
                "unmasked_full_vocab_logit_copies": samp_stats.get("unmasked_full_vocab_logit_copies", 0),
            }

        # Case 3: Unconstrained generation (chat / multi-channel Harmony).
        lg_2d = self.model.forward_b1(mx.array(suf, dtype=mx.int32)[None, :], c1, last_only=True)[0]
        eval_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p)
        t_pp = time.perf_counter() - t0
        t_dec0 = time.perf_counter()

        out_ids = []
        finished = False
        in_final = bool(started_in_final)
        while len(out_ids) < max_tokens:
            tok_orig, _ = self._sample_row_from_eval(
                eval_pack,
                0,
                grammar=None,
                matcher=None,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
                stats_counters=samp_stats,
                recent_ids=out_ids,
            )
            lg_2d = None
            eval_pack = None  # type: ignore[assignment]
            out_ids.append(tok_orig)
            in_final, should_stop = self._advance_channel_and_check_stop(tok_orig, out_ids, in_final)
            if should_stop:
                finished = True
                break
            if len(out_ids) >= max_tokens:
                break
            lg_2d = self.model.forward_b1(mx.array([[tok_orig]], dtype=mx.int32), c1, last_only=True)[0]
            eval_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p)
            steps += 1

        t_dec = time.perf_counter() - t_dec0
        del c1
        return {
            "out_ids": out_ids,
            "ana_ids": ana_ids,
            "ff_skipped": 0,
            "mask_hits": 0,
            "fallbacks": 0,
            "prefill_sec": t_pp,
            "decode_sec": t_dec,
            "steps": steps,
            "finished": finished,
            "ff_lm_head_skipped_steps": 0,
            "top128_fast_hits": samp_stats["top128_fast_hits"],
            "full_vocab_logit_copies": samp_stats["full_vocab_logit_copies"],
        }

    def _finish_b1_grammar_loop(
        self,
        c1: list[dict[str, Any]],
        eval_pack: dict[str, Any],
        matcher: Any,
        grammar: GrammarHelper | Any,
        out_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        rng: np.random.Generator,
        mask_hits: int,
        fallbacks: int,
        ff_skipped_ref: list[int],
        stats_counters: dict[str, int] | None = None,
    ) -> tuple[list[int], int, int, int]:
        import mlx.core as mx

        steps = 0
        stop_ids = self.tokenizer.final_stop_token_ids
        cur_pack: dict[str, Any] | None = eval_pack
        while not matcher.is_stopped() and len(out_ids) < max_tokens:
            assert cur_pack is not None
            tok_orig, was_masked = self._sample_row_from_eval(
                cur_pack,
                0,
                grammar=grammar,
                matcher=matcher,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
                stats_counters=stats_counters,
                recent_ids=out_ids,
            )
            cur_pack = None
            mask_hits += int(was_masked)
            if not matcher.consume_token(tok_orig):
                break
            out_ids.append(tok_orig)
            if tok_orig in stop_ids or matcher.is_stopped() or len(out_ids) >= max_tokens:
                break

            ff = matcher.compute_ff_tokens() or []
            if ff:
                rem = max(0, max_tokens - len(out_ids))
                if rem < len(ff):
                    take = ff[:rem]
                    if take:
                        matcher.consume_tokens(take)
                        out_ids.extend(take)
                        ff_skipped_ref[0] += len(take)
                    break
                matcher.consume_tokens(ff)
                out_ids.extend(ff)
                ff_skipped_ref[0] += len(ff)
                if matcher.is_stopped() or any(t in stop_ids for t in ff) or len(out_ids) >= max_tokens:
                    break
                feed = [tok_orig] + ff
            else:
                feed = [tok_orig]

            lg_2d = self.model.forward_b1(mx.array([feed], dtype=mx.int32), c1, last_only=True)[0]
            cur_pack = self._eval_logits_for_sampling(lg_2d, temperature=temperature, top_p=top_p, has_grammar=True)
            steps += 1
        return out_ids, mask_hits, fallbacks, steps

    def _run_batched_wave(
        self,
        prompt_ids_wave: list[list[int]],
        *,
        grammar: GrammarHelper | Any | None,
        per_doc_grammars: Sequence[str | dict[str, Any] | Any | None] | None = None,
        whitespace_flexible: bool | None = None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        rng: np.random.Generator,
        started_in_final: bool,
        analysis_budget: int = 0,
        chunk_docs: int = 8,
    ) -> dict[str, Any]:
        import mlx.core as mx

        b_sz = len(prompt_ids_wave)
        if per_doc_grammars is not None:
            if len(per_doc_grammars) != b_sz:
                raise ValueError(
                    f"Mismatched per_doc_grammars length={len(per_doc_grammars)} vs wave={b_sz}"
                )
            doc_grammars: list[Any] = [
                grammar if g is None else self._get_grammar(g, whitespace_flexible=whitespace_flexible)
                for g in per_doc_grammars
            ]
        else:
            doc_grammars = [grammar] * b_sz

        prefix_ids, _ = split_lcp_ids(prompt_ids_wave)
        sp, prefix_sec, prefix_hit = self._get_or_compute_prefix(prefix_ids)
        l_sys = int(sp["L_sys"])
        suf_lists = [list(ids[l_sys:]) for ids in prompt_ids_wave] if l_sys > 0 else [list(ids) for ids in prompt_ids_wave]

        ff_q: list[deque[int]] = [deque() for _ in range(b_sz)]
        out_ids: list[list[int]] = [[] for _ in range(b_sz)]
        ana_ids: list[list[int]] = [[] for _ in range(b_sz)]
        ff_skipped = [0] * b_sz
        mask_hits = [0] * b_sz
        finished_flags = [False] * b_sz
        in_final_slots = [bool(started_in_final)] * b_sz
        matchers = [self._new_doc_matcher(doc_grammars[i], i) for i in range(b_sz)]
        samp_stats: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}
        any_wave_grammar = any(g is not None for g in doc_grammars)
        two_phase_budgeted = bool(analysis_budget > 0 and not started_in_final)

        steps = 0
        t_pp = 0.0
        t_dec0 = time.perf_counter()

        if two_phase_budgeted:
            ff0_per_doc = [
                self._doc_upfront_ff(doc_grammars[i], i) if doc_grammars[i] is not None else []
                for i in range(b_sz)
            ]
            max_ff0 = max((len(f) for f in ff0_per_doc), default=0)
            suf_max_alloc = max(len(s) for s in suf_lists) + analysis_budget + len(self._trans_ids) + max_ff0 + 4
            bc = self.model.alloc_batch_cache(sp, B=b_sz, suf_max=suf_max_alloc, max_dec=max_tokens)

            t_pp0 = time.perf_counter()
            lg_p1 = self.model.prefill_batch_suffixes(suf_lists, bc, chunk_docs=chunk_docs)
            eval_p1 = self._eval_logits_for_sampling(lg_p1, temperature=temperature, top_p=top_p, has_grammar=False)
            t_pp = time.perf_counter() - t_pp0
            t_dec0 = time.perf_counter()

            p1_ids: list[list[int]] = [[] for _ in range(b_sz)]
            p1_done_steps = [0] * b_sz
            pending_last_ana: list[int | None] = [None] * b_sz
            already_in_final = [False] * b_sz
            already_stopped_p1 = [False] * b_sz
            saved_swa: list[list[tuple[Any, Any]] | None] = [None] * b_sz
            active_p1 = list(range(b_sz))

            def _snapshot_slot_swa(slot_b: int) -> None:
                layers_bc = bc.get("layers") or []
                if not layers_bc or "swa_init_lens_arr" not in bc:
                    return
                d_b = int(bc.get("swa_write", 0))
                init_b = int(bc["swa_init_lens_arr"][slot_b].item())
                swa_cap = int(getattr(self.model, "swa", 128))
                valid_b = min(swa_cap, init_b + d_b)
                snaps: list[tuple[Any, Any]] = []
                to_ev: list[Any] = []
                for lc in layers_bc:
                    if lc.get("is_full"):
                        continue
                    if valid_b > 0:
                        ring_idx = mx.arange(d_b - valid_b, d_b, dtype=mx.int32) % swa_cap
                        kp = mx.take(lc["k_swa"][slot_b : slot_b + 1], ring_idx, axis=2)
                        vp = mx.take(lc["v_swa"][slot_b : slot_b + 1], ring_idx, axis=2)
                        snaps.append((kp, vp))
                        to_ev.extend((kp, vp))
                    else:
                        snaps.append((None, None))  # type: ignore[arg-type]
                if to_ev:
                    mx.eval(*to_ev)  # guard: allow-eval
                saved_swa[slot_b] = snaps

            for _ in range(analysis_budget):
                next_active_p1: list[int] = []
                next_feed_toks: list[int] = []
                just_finished: list[int] = []
                cur_dec_cnt = int(bc.get("dec_cnt", 0))

                for r_i, orig_i in enumerate(active_p1):
                    tok_a, _ = self._sample_row_from_eval(
                        eval_p1,
                        r_i,
                        grammar=None,
                        matcher=None,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=p1_ids[orig_i],
                    )
                    if doc_grammars[orig_i] is None and tok_a == self.tokenizer.call_token_id:
                        p1_done_steps[orig_i] = cur_dec_cnt
                        out_ids[orig_i].append(tok_a)
                        finished_flags[orig_i] = True
                        already_stopped_p1[orig_i] = True
                        just_finished.append(orig_i)
                        continue
                    if tok_a in self.tokenizer.final_stop_token_ids:
                        p1_done_steps[orig_i] = cur_dec_cnt
                        just_finished.append(orig_i)
                        continue
                    p1_ids[orig_i].append(tok_a)
                    in_fin_p1, _ = self._advance_channel_and_check_stop(tok_a, p1_ids[orig_i], False)
                    if in_fin_p1:
                        p1_ids[orig_i] = self._strip_trailing_final_header(p1_ids[orig_i])
                        p1_done_steps[orig_i] = cur_dec_cnt
                        pending_last_ana[orig_i] = tok_a
                        already_in_final[orig_i] = True
                        just_finished.append(orig_i)
                        continue
                    if len(p1_ids[orig_i]) >= analysis_budget:
                        p1_done_steps[orig_i] = cur_dec_cnt
                        pending_last_ana[orig_i] = tok_a
                        just_finished.append(orig_i)
                        continue
                    next_active_p1.append(orig_i)
                    next_feed_toks.append(tok_a)

                lg_p1 = None
                eval_p1 = None  # type: ignore[assignment]
                if not next_active_p1:
                    break
                for done_i in just_finished:
                    _snapshot_slot_swa(done_i)

                padded_p1 = [self._dummy_pad_id] * b_sz
                for orig_i, nt in zip(next_active_p1, next_feed_toks):
                    padded_p1[orig_i] = nt
                samp_idx_p1 = (
                    mx.array(next_active_p1, dtype=mx.int32)
                    if len(next_active_p1) < b_sz
                    else None
                )
                lg_p1 = self.model.decode_step_batch(
                    mx.array(padded_p1, dtype=mx.int32), bc, samp_indices=samp_idx_p1
                )
                eval_p1 = self._eval_logits_for_sampling(
                    lg_p1, temperature=temperature, top_p=top_p, has_grammar=False
                )
                steps += 1
                active_p1 = next_active_p1

            ana_ids = p1_ids
            extra_lists: list[list[int]] = []
            for i in range(b_sz):
                if already_stopped_p1[i]:
                    extra_lists.append([self._dummy_pad_id])
                    in_final_slots[i] = True
                    continue
                chunk_i = [pending_last_ana[i]] if pending_last_ana[i] is not None else []
                if not already_in_final[i]:
                    chunk_i.extend(self._trans_ids)
                ff0 = ff0_per_doc[i]
                if ff0:
                    chunk_i.extend(ff0)
                    out_ids[i].extend(ff0)
                    ff_skipped[i] += len(ff0)
                if not chunk_i:
                    chunk_i.append(self._dummy_pad_id)
                extra_lists.append(chunk_i)
                in_final_slots[i] = True

            lg = self.model.append_batch_suffixes(
                extra_lists,
                bc,
                p1_done_steps=p1_done_steps,
                saved_swa=saved_swa,
                chunk_docs=chunk_docs,
            )
            del saved_swa
            eval_pack = self._eval_logits_for_sampling(
                lg, temperature=temperature, top_p=top_p, has_grammar=any_wave_grammar
            )
        else:
            for i in range(b_sz):
                g_i = doc_grammars[i]
                if g_i is not None:
                    ff0 = self._doc_upfront_ff(g_i, i)
                    if ff0:
                        suf_lists[i] = suf_lists[i] + ff0
                        out_ids[i].extend(ff0)
                        ff_skipped[i] += len(ff0)

            suf_max = max(len(s) for s in suf_lists)
            bc = self.model.alloc_batch_cache(sp, B=b_sz, suf_max=suf_max, max_dec=max_tokens)

            t_pp0 = time.perf_counter()
            lg = self.model.prefill_batch_suffixes(suf_lists, bc, chunk_docs=chunk_docs)
            eval_pack = self._eval_logits_for_sampling(
                lg, temperature=temperature, top_p=top_p, has_grammar=any_wave_grammar
            )
            t_pp = time.perf_counter() - t_pp0
            t_dec0 = time.perf_counter()

        active_orig = list(range(b_sz))
        b1_tail_mode = False
        b1_cache: list[dict[str, Any]] | None = None
        b1_orig_idx: int | None = None
        ff_lm_head_skipped_steps = 0
        compactions = 0
        stop_ids = self.tokenizer.final_stop_token_ids
        async_eval_fn = getattr(mx, "async_eval", mx.eval)

        def _consume_logits_step(
            cur_pack: dict[str, Any] | None,
            step_locals: list[int],
            step_origs: list[int],
        ) -> tuple[list[int], list[int], list[int]]:
            keep_local: list[int] = []
            keep_orig: list[int] = []
            next_toks: list[int] = []
            samp_row = 0
            for loc_i, orig_i in zip(step_locals, step_origs):
                if finished_flags[orig_i]:
                    if cur_pack is not None:
                        samp_row += 1
                    continue
                g_i = doc_grammars[orig_i]
                if g_i is not None:
                    if ff_q[orig_i]:
                        tok = ff_q[orig_i].popleft()
                        out_ids[orig_i].append(tok)
                        ff_skipped[orig_i] += 1
                        if (
                            tok in stop_ids
                            or (matchers[orig_i].is_stopped() and not ff_q[orig_i])
                            or len(out_ids[orig_i]) >= max_tokens
                        ):
                            finished_flags[orig_i] = bool(
                                tok in stop_ids or (matchers[orig_i].is_stopped() and not ff_q[orig_i])
                            )
                            continue
                        keep_local.append(loc_i)
                        keep_orig.append(orig_i)
                        next_toks.append(tok)
                        continue
                    if matchers[orig_i].is_stopped():
                        finished_flags[orig_i] = True
                        if cur_pack is not None:
                            samp_row += 1
                        continue

                    assert cur_pack is not None
                    tok, was_masked = self._sample_row_from_eval(
                        cur_pack,
                        samp_row,
                        grammar=g_i,
                        matcher=matchers[orig_i],
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=out_ids[orig_i],
                    )
                    samp_row += 1
                    mask_hits[orig_i] += int(was_masked)
                    if not matchers[orig_i].consume_token(tok):
                        finished_flags[orig_i] = True
                        continue
                    out_ids[orig_i].append(tok)
                    if tok in stop_ids or matchers[orig_i].is_stopped() or len(out_ids[orig_i]) >= max_tokens:
                        finished_flags[orig_i] = bool(tok in stop_ids or matchers[orig_i].is_stopped())
                        continue
                    ff = matchers[orig_i].compute_ff_tokens() or []
                    if ff:
                        matchers[orig_i].consume_tokens(ff)
                        ff_q[orig_i].extend(ff)
                        if matchers[orig_i].is_stopped() and not ff_q[orig_i]:
                            finished_flags[orig_i] = True
                            continue
                    keep_local.append(loc_i)
                    keep_orig.append(orig_i)
                    next_toks.append(tok)
                else:
                    assert cur_pack is not None
                    tok, _ = self._sample_row_from_eval(
                        cur_pack,
                        samp_row,
                        grammar=None,
                        matcher=None,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=out_ids[orig_i],
                    )
                    samp_row += 1
                    out_ids[orig_i].append(tok)
                    in_fin, should_stop = self._advance_channel_and_check_stop(
                        tok, out_ids[orig_i], in_final_slots[orig_i]
                    )
                    in_final_slots[orig_i] = in_fin
                    if should_stop:
                        finished_flags[orig_i] = True
                        continue
                    if len(out_ids[orig_i]) >= max_tokens:
                        continue
                    keep_local.append(loc_i)
                    keep_orig.append(orig_i)
                    next_toks.append(tok)
            return keep_local, keep_orig, next_toks

        keep_local, keep_orig, next_toks = _consume_logits_step(
            eval_pack, list(range(b_sz)), active_orig
        )
        lg = None
        eval_pack = None  # type: ignore[assignment]

        while keep_orig:
            cur_b = int(bc["B"]) if not b1_tail_mode else 1
            if not b1_tail_mode:
                if len(keep_orig) == 1 and cur_b > 1:
                    b1_orig_idx = keep_orig[0]
                    b1_cache = self.model.extract_b1_cache(
                        bc, keep_local[0], max_rem=max_tokens - len(out_ids[b1_orig_idx]) + 8
                    )
                    b1_tail_mode = True
                    del bc
                    gc.collect()
                    mx.clear_cache()
                elif len(keep_local) < cur_b and (len(keep_local) <= int(cur_b * 0.75) or cur_b <= 8):
                    self.model.compact_batch_cache(bc, keep_local)
                    compactions += 1
                    keep_local = list(range(len(keep_local)))
                    active_orig = keep_orig

            if b1_tail_mode:
                assert b1_cache is not None and b1_orig_idx is not None
                orig_i = b1_orig_idx
                g_i = doc_grammars[orig_i]
                feed = [next_toks[0]]
                while ff_q[orig_i] and len(out_ids[orig_i]) < max_tokens:
                    nxt_ff = ff_q[orig_i].popleft()
                    out_ids[orig_i].append(nxt_ff)
                    ff_skipped[orig_i] += 1
                    feed.append(nxt_ff)
                if (
                    (g_i is not None and matchers[orig_i].is_stopped() and not ff_q[orig_i])
                    or len(out_ids[orig_i]) >= max_tokens
                    or any(t in stop_ids for t in feed[1:])
                ):
                    finished_flags[orig_i] = bool(
                        (g_i is not None and matchers[orig_i].is_stopped() and not ff_q[orig_i])
                        or any(t in stop_ids for t in feed[1:])
                    )
                    keep_orig = []
                    break

                lg1_2d = self.model.forward_b1(mx.array([feed], dtype=mx.int32), b1_cache, last_only=True)[0]
                pack_1 = self._eval_logits_for_sampling(
                    lg1_2d, temperature=temperature, top_p=top_p, has_grammar=(g_i is not None)
                )
                steps += 1

                if g_i is not None:
                    tok, was_masked = self._sample_row_from_eval(
                        pack_1,
                        0,
                        grammar=g_i,
                        matcher=matchers[orig_i],
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=out_ids[orig_i],
                    )
                    mask_hits[orig_i] += int(was_masked)
                    if not matchers[orig_i].consume_token(tok):
                        finished_flags[orig_i] = True
                        keep_orig = []
                        break
                    out_ids[orig_i].append(tok)
                    if tok in stop_ids or matchers[orig_i].is_stopped() or len(out_ids[orig_i]) >= max_tokens:
                        finished_flags[orig_i] = bool(tok in stop_ids or matchers[orig_i].is_stopped())
                        keep_orig = []
                        break
                    ff = matchers[orig_i].compute_ff_tokens() or []
                    if ff:
                        matchers[orig_i].consume_tokens(ff)
                        ff_q[orig_i].extend(ff)
                else:
                    tok, _ = self._sample_row_from_eval(
                        pack_1,
                        0,
                        grammar=None,
                        matcher=None,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        stats_counters=samp_stats,
                        recent_ids=out_ids[orig_i],
                    )
                    out_ids[orig_i].append(tok)
                    in_fin, should_stop = self._advance_channel_and_check_stop(
                        tok, out_ids[orig_i], in_final_slots[orig_i]
                    )
                    in_final_slots[orig_i] = in_fin
                    if should_stop or len(out_ids[orig_i]) >= max_tokens:
                        finished_flags[orig_i] = should_stop
                        keep_orig = []
                        break
                lg1_2d = None
                pack_1 = None  # type: ignore[assignment]
                next_toks = [tok]
                continue

            cur_bc_b = int(bc["B"])
            if len(keep_local) == cur_bc_b:
                tok_mx = mx.array(next_toks, dtype=mx.int32)
                active_step_orig = keep_orig
                active_step_local = list(range(cur_bc_b))
            else:
                padded = [self._dummy_pad_id] * cur_bc_b
                for loc_i, nt in zip(keep_local, next_toks):
                    padded[loc_i] = nt
                tok_mx = mx.array(padded, dtype=mx.int32)
                active_step_orig = keep_orig
                active_step_local = keep_local

            samp_locs = [
                loc_i
                for loc_i, orig_i in zip(active_step_local, active_step_orig)
                if doc_grammars[orig_i] is None or not ff_q[orig_i]
            ]

            if len(samp_locs) == 0:
                self.model.decode_step_batch(tok_mx, bc, skip_lm_head=True)
                dirty_bufs: list[Any] = []
                if "rope_pos_arr" in bc:
                    dirty_bufs.append(bc["rope_pos_arr"])
                for layer_c in bc.get("layers", ()):
                    if layer_c.get("is_full"):
                        dirty_bufs.extend((layer_c["k_suf"], layer_c["v_suf"]))
                    else:
                        dirty_bufs.extend((layer_c["k_swa"], layer_c["v_swa"]))
                if dirty_bufs:
                    async_eval_fn(*dirty_bufs)
                steps += 1
                ff_lm_head_skipped_steps += 1
                step_pack: dict[str, Any] | None = None
            else:
                samp_idx_mx = (
                    mx.array(samp_locs, dtype=mx.int32)
                    if len(samp_locs) < cur_bc_b
                    else None
                )
                lg = self.model.decode_step_batch(tok_mx, bc, samp_indices=samp_idx_mx)
                step_pack = self._eval_logits_for_sampling(
                    lg, temperature=temperature, top_p=top_p, has_grammar=any_wave_grammar
                )
                steps += 1

            keep_local, keep_orig, next_toks = _consume_logits_step(
                step_pack, active_step_local, active_step_orig
            )
            lg = None
            step_pack = None

        t_dec = time.perf_counter() - t_dec0
        if b1_cache is not None:
            del b1_cache
        else:
            del bc
        gc.collect()
        mx.clear_cache()
        return {
            "out_ids": out_ids,
            "ana_ids": ana_ids,
            "two_phase_final": two_phase_budgeted,
            "ff_skipped": ff_skipped,
            "mask_hits": mask_hits,
            "finished": finished_flags,
            "lcp": l_sys,
            "prefix_sec": prefix_sec,
            "prefix_hit": prefix_hit,
            "prefill_sec": t_pp,
            "decode_sec": t_dec,
            "steps": steps,
            "ff_lm_head_skipped_steps": ff_lm_head_skipped_steps,
            "top128_fast_hits": samp_stats["top128_fast_hits"],
            "full_vocab_logit_copies": samp_stats["full_vocab_logit_copies"],
            "unmasked_full_vocab_logit_copies": samp_stats.get("unmasked_full_vocab_logit_copies", 0),
            "compactions": compactions,
        }

    def generate(
        self,
        prompt: PromptInput,
        *,
        system_prompt: str | None = None,
        json_schema: str | dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        mode: str | None = None,
        speculative: bool | None = None,
        use_chat_template: bool = True,
        whitespace_flexible: bool | None = None,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
        compact: bool | None = None,
        collect_telemetry: bool = False,
        collect_valid_lse: bool = False,
    ) -> GenerationItem:
        res = self.generate_batch(
            [prompt],
            system_prompt=system_prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            mode=mode,
            speculative=speculative,
            use_chat_template=use_chat_template,
            whitespace_flexible=whitespace_flexible,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            reasoning_budget=reasoning_budget,
            compact=compact,
            collect_telemetry=collect_telemetry,
            collect_valid_lse=collect_valid_lse,
        )
        return res.items[0]

    def generate_batch(
        self,
        prompts: Sequence[PromptInput],
        *,
        system_prompt: str | None = None,
        json_schema: str | dict[str, Any] | None = None,
        per_doc_grammars: Sequence[str | dict[str, Any] | Any | None] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        mode: str | None = None,
        batch_size: int | None = None,
        speculative: bool | None = None,
        use_chat_template: bool = True,
        whitespace_flexible: bool | None = None,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
        compact: bool | None = None,
        chunk_docs: int = 8,
        collect_telemetry: bool = False,
        collect_valid_lse: bool = False,
    ) -> GenerationBatchResult:
        del speculative, collect_telemetry, collect_valid_lse  # Accepted for cross-engine API parity.
        n_docs = len(prompts)
        if per_doc_grammars is not None and len(per_doc_grammars) != n_docs:
            raise ValueError(
                f"Mismatched per_doc_grammars length={len(per_doc_grammars)} vs prompts={n_docs}"
            )
        if n_docs == 0:
            return GenerationBatchResult(items=[], stats={"docs": 0})

        import mlx.core as mx

        if compact is not None:
            eff_ws: bool | None = False if bool(compact) else (
                None if whitespace_flexible is None else bool(whitespace_flexible)
            )
        else:
            eff_ws = None if whitespace_flexible is None else bool(whitespace_flexible)

        has_any_schema = (json_schema is not None) or (
            per_doc_grammars is not None and any(g is not None for g in per_doc_grammars)
        )
        eff_effort, ana_budget = resolve_reasoning_params(
            has_schema=has_any_schema,
            reasoning_effort=reasoning_effort,
            reasoning_budget=reasoning_budget,
            enable_thinking=bool(enable_thinking),
            default_effort=self.default_reasoning_effort,
        )
        started_in_final = eff_effort in ("skip", "none")

        with self._lock:
            mx.reset_peak_memory()
            t_all0 = time.perf_counter()

            default_grammar = self._get_grammar(json_schema, whitespace_flexible=eff_ws)
            if per_doc_grammars is not None:
                doc_grammars = [
                    default_grammar if g is None else self._get_grammar(g, whitespace_flexible=eff_ws)
                    for g in per_doc_grammars
                ]
            else:
                doc_grammars = [default_grammar] * n_docs

            prompt_ids_list = self.encode_prompts(
                prompts,
                system_prompt=system_prompt,
                use_chat_template=use_chat_template,
                reasoning_effort=eff_effort,
            )

            host_mem = getattr(self, "_host_memory_bytes", 0) or None
            path_mode, wave_bs = select_gpt_oss_plan(
                mode or self.default_mode,
                n_docs=n_docs,
                batch_size=batch_size,
                memory_bytes=host_mem,
            )

            rng = np.random.default_rng(int(seed))
            all_out_ids: list[list[int]] = [[] for _ in range(n_docs)]
            all_ana_ids: list[list[int]] = [[] for _ in range(n_docs)]
            all_out_in_final: list[bool] = [bool(started_in_final)] * n_docs
            all_finished: list[bool] = [False] * n_docs
            tot_ff = 0
            tot_mask_hits = 0
            tot_pre_sec = 0.0
            tot_dec_sec = 0.0
            tot_prefix_sec = 0.0
            tot_steps = 0
            tot_ff_lm_head_skipped_steps = 0
            tot_top128_fast_hits = 0
            tot_full_vocab_logit_copies = 0
            tot_unmasked_full_copies = 0
            tot_compactions = 0
            lcp_val = 0
            waves: list[tuple[int, int]] = []

            if path_mode == "b1":
                shared_sp = None
                shared_len = 0
                if n_docs >= 2:
                    prefix_ids, _ = split_lcp_ids(prompt_ids_list)
                    shared_sp, prefix_sec, _ = self._get_or_compute_prefix(prefix_ids)
                    tot_prefix_sec += prefix_sec
                    shared_len = int(shared_sp["L_sys"])
                    lcp_val = shared_len
                for idx in range(n_docs):
                    waves.append((idx, idx + 1))
                    ids = prompt_ids_list[idx]
                    if n_docs == 1:
                        matched_sp, _ = self._match_cached_prefix_for_single(ids)
                        sp_use = matched_sp
                        sp_len = int(matched_sp["L_sys"]) if matched_sp is not None else 0
                    else:
                        sp_use = shared_sp
                        sp_len = shared_len
                    res_1 = self._run_b1_single(
                        ids,
                        grammar=doc_grammars[idx],
                        doc_index=idx,
                        whitespace_flexible=eff_ws,
                        shared_prefix=sp_use,
                        prefix_len=sp_len,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        rng=rng,
                        started_in_final=started_in_final,
                        analysis_budget=ana_budget,
                    )
                    all_out_ids[idx] = res_1["out_ids"]
                    all_ana_ids[idx] = res_1["ana_ids"]
                    if res_1.get("two_phase_final"):
                        all_out_in_final[idx] = True
                    all_finished[idx] = bool(res_1["finished"])
                    tot_ff += int(res_1["ff_skipped"])
                    tot_mask_hits += int(res_1["mask_hits"])
                    tot_pre_sec += float(res_1["prefill_sec"])
                    tot_dec_sec += float(res_1["decode_sec"])
                    tot_steps += int(res_1["steps"])
                    tot_ff_lm_head_skipped_steps += int(res_1.get("ff_lm_head_skipped_steps", 0))
                    tot_top128_fast_hits += int(res_1.get("top128_fast_hits", 0))
                    tot_full_vocab_logit_copies += int(res_1.get("full_vocab_logit_copies", 0))
                    tot_unmasked_full_copies += int(res_1.get("unmasked_full_vocab_logit_copies", 0))
                    if getattr(mx, "get_cache_memory", lambda: 0)() > (256 << 20):
                        mx.clear_cache()
            else:
                order = sorted(range(n_docs), key=lambda idx: len(prompt_ids_list[idx]))
                for start in range(0, n_docs, wave_bs):
                    end = min(n_docs, start + wave_bs)
                    wave_orig = order[start:end]
                    waves.append((start, end))
                    sub_ids = [prompt_ids_list[i] for i in wave_orig]
                    sub_grammars = [doc_grammars[i] for i in wave_orig]
                    if len(sub_ids) == 1:
                        orig_0 = wave_orig[0]
                        matched_sp, _ = self._match_cached_prefix_for_single(sub_ids[0])
                        sp_len = int(matched_sp["L_sys"]) if matched_sp is not None else 0
                        res_1 = self._run_b1_single(
                            sub_ids[0],
                            grammar=sub_grammars[0],
                            doc_index=orig_0,
                            whitespace_flexible=eff_ws,
                            shared_prefix=matched_sp,
                            prefix_len=sp_len,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            rng=rng,
                            started_in_final=started_in_final,
                            analysis_budget=ana_budget,
                        )
                        all_out_ids[orig_0] = res_1["out_ids"]
                        all_ana_ids[orig_0] = res_1["ana_ids"]
                        if res_1.get("two_phase_final"):
                            all_out_in_final[orig_0] = True
                        all_finished[orig_0] = bool(res_1["finished"])
                        tot_ff += int(res_1["ff_skipped"])
                        tot_mask_hits += int(res_1["mask_hits"])
                        tot_pre_sec += float(res_1["prefill_sec"])
                        tot_dec_sec += float(res_1["decode_sec"])
                        tot_steps += int(res_1["steps"])
                        tot_ff_lm_head_skipped_steps += int(res_1.get("ff_lm_head_skipped_steps", 0))
                        tot_top128_fast_hits += int(res_1.get("top128_fast_hits", 0))
                        tot_full_vocab_logit_copies += int(res_1.get("full_vocab_logit_copies", 0))
                        tot_unmasked_full_copies += int(res_1.get("unmasked_full_vocab_logit_copies", 0))
                    else:
                        w_res = self._run_batched_wave(
                            sub_ids,
                            grammar=default_grammar,
                            per_doc_grammars=sub_grammars,
                            whitespace_flexible=eff_ws,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            rng=rng,
                            started_in_final=started_in_final,
                            analysis_budget=ana_budget,
                            chunk_docs=chunk_docs,
                        )
                        w_ana = w_res.get("ana_ids")
                        w_tp_fin = bool(w_res.get("two_phase_final", False))
                        for loc_i, orig_i in enumerate(wave_orig):
                            all_out_ids[orig_i] = w_res["out_ids"][loc_i]
                            if w_ana is not None and loc_i < len(w_ana):
                                all_ana_ids[orig_i] = w_ana[loc_i]
                            if w_tp_fin:
                                all_out_in_final[orig_i] = True
                            all_finished[orig_i] = bool(w_res["finished"][loc_i])
                            tot_ff += int(w_res["ff_skipped"][loc_i])
                            tot_mask_hits += int(w_res["mask_hits"][loc_i])
                        lcp_val = max(lcp_val, int(w_res["lcp"]))
                        tot_prefix_sec += float(w_res["prefix_sec"])
                        tot_pre_sec += float(w_res["prefill_sec"])
                        tot_dec_sec += float(w_res["decode_sec"])
                        tot_steps += int(w_res["steps"])
                        tot_ff_lm_head_skipped_steps += int(w_res.get("ff_lm_head_skipped_steps", 0))
                        tot_top128_fast_hits += int(w_res.get("top128_fast_hits", 0))
                        tot_full_vocab_logit_copies += int(w_res.get("full_vocab_logit_copies", 0))
                        tot_unmasked_full_copies += int(w_res.get("unmasked_full_vocab_logit_copies", 0))
                        tot_compactions += int(w_res["compactions"])
                    mx.clear_cache()

            items: list[GenerationItem] = []
            valid_json_cnt = 0
            valid_schema_cnt = 0
            tot_out_toks = 0
            for idx in range(n_docs):
                doc_g = doc_grammars[idx]
                gen_ids = all_out_ids[idx]
                ana_ids = all_ana_ids[idx]
                comp_toks = len(gen_ids) + len(ana_ids)
                tot_out_toks += comp_toks

                if doc_g is not None:
                    raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    text = raw_text.strip()
                    reasoning = (
                        self.tokenizer.decode(ana_ids, skip_special_tokens=True).strip() or None
                        if ana_ids
                        else None
                    )
                else:
                    raw_full = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                    text, reasoning = parse_harmony_output(
                        raw_full,
                        started_in_analysis=(use_chat_template and not all_out_in_final[idx]),
                    )
                    if reasoning is None and ana_ids:
                        reasoning = self.tokenizer.decode(ana_ids, skip_special_tokens=True).strip() or None

                parsed = None
                is_json = False
                is_schema = doc_g is None
                try:
                    parsed = json.loads(text)
                    is_json = True
                    valid_json_cnt += 1
                except Exception:
                    pass
                if doc_g is not None:
                    tok_ok = doc_g.validate_token_sequence(gen_ids)
                    schema_ok = True
                    if is_json:
                        sch_obj = getattr(doc_g, "schema_obj", None)
                        if sch_obj is not None:
                            try:
                                validate_json_against_schema(parsed, sch_obj)
                            except Exception:
                                schema_ok = False
                    else:
                        schema_ok = False
                    is_schema = bool(tok_ok and schema_ok)
                    if is_schema:
                        valid_schema_cnt += 1
                else:
                    valid_schema_cnt += 1

                finish_reason = "stop" if (all_finished[idx] or (doc_g is not None and is_schema)) else (
                    "length" if len(gen_ids) >= max_tokens else "stop"
                )
                items.append(
                    GenerationItem(
                        text=text,
                        parsed=parsed,
                        token_ids=ana_ids + gen_ids,
                        prompt_tokens=len(prompt_ids_list[idx]),
                        completion_tokens=comp_toks,
                        valid_json=is_json,
                        valid_schema=is_schema,
                        finish_reason=finish_reason,
                        reasoning=reasoning,
                    )
                )

            elapsed = time.perf_counter() - t_all0
            prompt_tok_tot = sum(len(x) for x in prompt_ids_list)
            peak_gb = mx.get_peak_memory() / (1024**3)
            stats = {
                "mode_requested": mode or self.default_mode,
                "mode_used": path_mode,
                "reasoning_effort": eff_effort,
                "reasoning_budget": ana_budget,
                "docs": n_docs,
                "waves": len(waves),
                "lcp": lcp_val,
                "prompt_tokens": prompt_tok_tot,
                "completion_tokens": tot_out_toks,
                "ff_tokens": tot_ff,
                "ff_lm_head_skipped_steps": tot_ff_lm_head_skipped_steps,
                "top128_fast_hits": tot_top128_fast_hits,
                "full_vocab_logit_copies": tot_full_vocab_logit_copies,
                "unmasked_full_vocab_logit_copies": tot_unmasked_full_copies,
                "mask_hits": tot_mask_hits,
                "prefix_sec": tot_prefix_sec,
                "prefill_sec": tot_pre_sec,
                "decode_sec": tot_dec_sec,
                "elapsed_sec": elapsed,
                "decode_tok_s": tot_out_toks / max(tot_dec_sec, 1e-9),
                "total_tok_s": (prompt_tok_tot + tot_out_toks) / max(elapsed, 1e-9),
                "docs_per_sec": n_docs / max(elapsed, 1e-9),
                "valid_json_count": valid_json_cnt,
                "valid_schema_count": valid_schema_cnt,
                "steps": tot_steps,
                "compactions": tot_compactions,
                "peak_gb": peak_gb,
            }
            if getattr(mx, "get_cache_memory", lambda: 0)() > (256 << 20):
                mx.clear_cache()
            return GenerationBatchResult(items=items, stats=stats)


__all__ = [
    "FastGptOssEngine",
    "GenerationBatchResult",
    "GenerationItem",
    "PromptInput",
    "recommended_max_wave_batch",
    "resolve_reasoning_params",
    "select_gpt_oss_plan",
    "split_lcp_ids",
]
