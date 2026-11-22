"""High-throughput `FastQwen3Engine` for Qwen3.5 & Qwen3.6 (`0.8B`..`35B-A3B`) on Apple Silicon M4 & M5."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import mlx.core as mx
import numpy as np

from fast_shared.grammar import (
    GrammarHelper,
    apply_loop_breaker_penalties_1d,
    compute_loop_breaker_penalties,
    normalize_json_schema,
    sample_unconstrained,
    validate_json_against_schema,
)
from fast_shared.prefix_tree import (
    DEFAULT_CHUNK_TOKENS,
    PrefixTree,
    dequantize_gdn_state_block32,
    quantize_gdn_state_block32,
    recommended_prefix_budget_bytes,
)
from fast_shared.runtime_lessons import LiveMemoryGovernor

from .model import (
    DeltaSlotCache,
    FastQwen3Model,
    GrowingKVCache,
    HybridPrefixSnapshot,
    restore_hybrid_cache,
    snapshot_hybrid_cache,
)
from .mtp_drafter import (
    Qwen3MtpDrafter,
    draft_prompt_lookup_tokens,
    recommended_draft_k,
    should_use_speculative_qwen,
    verify_draft_sequence_b1,
)
from .paths import DEFAULT_HF_REPO, Qwen3Paths, resolve_paths
from .tokenizer import (
    FastQwen3Tokenizer,
    QwenVocabRemapper,
    inject_reasoning_prefix_property,
    validate_thinking_budget,
)

PromptInput = str | Sequence[Mapping[str, Any]]


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
    telemetry: list[dict[str, float]] | None = None


@dataclass
class GenerationBatchResult:
    items: list[GenerationItem]
    stats: dict[str, Any] = field(default_factory=dict)


def split_lcp_ids(prompt_ids_list: Sequence[Sequence[int]], *, min_lcp: int = 32) -> tuple[list[int], list[list[int]]]:
    """Extract longest common token prefix (`>= min_lcp`) leaving at least 1 suffix token per prompt."""
    if len(prompt_ids_list) < 2:
        return [], [list(x) for x in prompt_ids_list]
    min_len = min(len(x) for x in prompt_ids_list)
    if min_len <= min_lcp:
        return [], [list(x) for x in prompt_ids_list]

    first = prompt_ids_list[0]
    lcp = min_len - 1
    for other in prompt_ids_list[1:]:
        cur = 0
        lim = min(lcp, len(other) - 1)
        while cur < lim and first[cur] == other[cur]:
            cur += 1
        lcp = cur
        if lcp < min_lcp:
            return [], [list(x) for x in prompt_ids_list]
    return list(first[:lcp]), [list(x[lcp:]) for x in prompt_ids_list]


def _compute_step_telemetry(logits_1d: np.ndarray, chosen_idx: int) -> dict[str, float]:
    arr = logits_1d.astype(np.float64)
    mx_v = float(np.max(arr))
    shifted = arr - mx_v
    exp_v = np.exp(shifted)
    z = float(np.sum(exp_v))
    probs = exp_v / max(z, 1e-30)
    p_tok = float(probs[chosen_idx]) if 0 <= chosen_idx < len(probs) else 0.0
    nz = probs[probs > 1e-12]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "logprob": float(np.log(max(p_tok, 1e-30))),
        "prob": p_tok,
        "entropy": entropy,
    }


class FastQwen3Engine:
    """Unified Qwen3.5 / Qwen3.6 inference & structured-extraction engine (`M4` / `M5`)."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        model_dir: str | Path | None = None,
        repo_id: str | None = None,
        model: FastQwen3Model | None = None,
        tokenizer: FastQwen3Tokenizer | None = None,
        mtp_drafter: Qwen3MtpDrafter | None = None,
        lora_adapter: str | Path | None = None,
        vocab_out: int | None = None,
        default_mode: str = "auto",
        default_enable_thinking: bool = False,
        default_thinking_budget: int = 0,
        max_prefix_cache_entries: int = 8,
        prefix_cache_mb: int | str | None = "auto",
        kv_i8: bool | str = "auto",
        mlx_cache_limit_mb: int = 512,
        local_files_only: bool = True,
        allow_oversubscribe: bool = False,
        warmup: bool = True,
    ) -> None:
        self.paths: Qwen3Paths = resolve_paths(home)
        self.default_mode = str(default_mode)
        self.default_enable_thinking = bool(default_enable_thinking)
        self.default_thinking_budget = int(default_thinking_budget)
        self.max_prefix_cache_entries = max(0, int(max_prefix_cache_entries))
        self.kv_i8_mode = kv_i8
        self._mlx_cache_limit_mb = int(mlx_cache_limit_mb)
        self._lock = threading.RLock()
        self._grammar_cache: OrderedDict[tuple[str, bool | None], GrammarHelper] = OrderedDict()
        self._prefix_cache: OrderedDict[tuple[int, ...], HybridPrefixSnapshot] = OrderedDict()
        self._memory_governor = LiveMemoryGovernor(min_b_slots=1)

        try:
            host_ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            host_ram_bytes = 24 * (1 << 30)
        if prefix_cache_mb is None or prefix_cache_mb == 0:
            tree_budget = 0
        elif str(prefix_cache_mb).strip().lower() == "auto":
            tree_budget = recommended_prefix_budget_bytes(host_ram_bytes)
        else:
            tree_budget = max(0, int(prefix_cache_mb)) << 20
        self.prefix_tree: PrefixTree | None = (
            PrefixTree(chunk_tokens=DEFAULT_CHUNK_TOKENS, byte_budget=tree_budget)
            if tree_budget > 0
            else None
        )

        if mlx_cache_limit_mb > 0 and hasattr(mx, "set_cache_limit"):
            try:
                mx.set_cache_limit(int(mlx_cache_limit_mb) << 20)
            except Exception:
                pass

        if model is not None and tokenizer is not None:
            self.model_dir: Path | None = model.model_dir
            self.model = model
            self.tokenizer = tokenizer
            self.mtp_drafter = mtp_drafter
            self.model_name = str(repo_id or (model.model_dir.name if model.model_dir else "fast-qwen3"))
        else:
            resolved_dir = self.paths.resolve_model_dir(
                model_dir,
                repo_id=repo_id or DEFAULT_HF_REPO,
                local_files_only=local_files_only,
            )
            self.model_dir = resolved_dir
            self.model_name = str(repo_id or resolved_dir.name)
            tok_path = self.paths.resolve_tokenizer_json(resolved_dir)
            raw_cfg = json.loads((resolved_dir / "config.json").read_text(encoding="utf-8"))
            tcfg = raw_cfg.get("text_config") or raw_cfg
            full_v = int(tcfg.get("vocab_size", 248320))
            self.tokenizer = tokenizer or FastQwen3Tokenizer(tok_path, full_vocab_size=full_v)

            remap: QwenVocabRemapper | None = None
            if vocab_out is not None and 0 < int(vocab_out) < full_v:
                remap = QwenVocabRemapper.build_ascii_latin_subset(
                    self.tokenizer.backend_tokenizer,
                    full_vocab_size=full_v,
                    target_size=int(vocab_out),
                )

            loaded_model, mtp_w = FastQwen3Model.from_pretrained(
                resolved_dir,
                vocab_remap=remap,
                allow_oversubscribe=allow_oversubscribe,
            )
            self.model = loaded_model
            self.mtp_drafter = mtp_drafter or Qwen3MtpDrafter.from_mtp_weights(loaded_model, mtp_w)

        if lora_adapter is not None:
            self.model.load_lora_adapter(lora_adapter)

        tok_real_vocab = int(self.tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True))
        self._new_to_old: np.ndarray = (
            self.model.vocab_remap.new_to_old
            if self.model.vocab_remap is not None
            else np.arange(min(self.model.vocab_n, tok_real_vocab), dtype=np.int32)
        )

        if warmup:
            self._run_warmup()

    def _run_warmup(self) -> None:
        cache = self.model.make_cache(kv_step=32)
        dummy = mx.array([[1, 2]], dtype=mx.int32)
        logits = self.model(dummy, cache)
        mx.eval(logits)
        mx.clear_cache()

    def close(self) -> None:
        self._prefix_cache.clear()
        if self.prefix_tree is not None:
            self.prefix_tree.clear()
        self._grammar_cache.clear()
        gc.collect()
        mx.clear_cache()

    def _resolve_kv_i8(self, batch_size: int, kv_alloc: int) -> bool:
        if isinstance(self.kv_i8_mode, bool):
            return self.kv_i8_mode
        return int(batch_size) >= 16 or int(kv_alloc) >= 2048

    def get_grammar(
        self,
        schema: str | Mapping[str, Any],
        *,
        whitespace_flexible: bool | None = None,
        **_ignored: Any,
    ) -> GrammarHelper:
        return self._get_grammar_helper(schema, whitespace_flexible)

    def _get_grammar(
        self,
        schema: str | Mapping[str, Any],
        *,
        whitespace_flexible: bool | None = None,
        **_ignored: Any,
    ) -> GrammarHelper:
        return self._get_grammar_helper(schema, whitespace_flexible)

    def _get_grammar_helper(
        self,
        schema: str | Mapping[str, Any],
        whitespace_flexible: bool | None,
    ) -> GrammarHelper:
        norm_s = normalize_json_schema(dict(schema) if isinstance(schema, Mapping) else schema)
        key = (norm_s, whitespace_flexible)
        if key in self._grammar_cache:
            self._grammar_cache.move_to_end(key)
            return self._grammar_cache[key]
        gh = GrammarHelper(
            self.tokenizer,
            norm_s,
            new_to_old=self._new_to_old,
            whitespace_flexible=whitespace_flexible,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self._grammar_cache[key] = gh
        if len(self._grammar_cache) > 32:
            self._grammar_cache.popitem(last=False)
        return gh

    def _lookup_or_build_prefix(self, prefix_ids: Sequence[int]) -> HybridPrefixSnapshot:
        key = tuple(int(t) for t in prefix_ids)
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            return self._prefix_cache[key]
        c1 = self.model.make_cache(kv_step=max(256, len(key) + 64))
        inp = mx.array([list(key)], dtype=mx.int32)
        h = self.model.forward_hidden(inp, c1)
        mx.eval(h)
        snap = snapshot_hybrid_cache(key, c1, compress_gdn_int8=True)
        if self.max_prefix_cache_entries > 0:
            self._prefix_cache[key] = snap
            while len(self._prefix_cache) > self.max_prefix_cache_entries:
                self._prefix_cache.popitem(last=False)
        return snap

    def _select_next_token(
        self,
        logits_1d: np.ndarray,
        *,
        full_hidden_1d: mx.array | None,
        matcher: Any | None,
        gram: GrammarHelper | None,
        recent_tokens: Sequence[int],
        temperature: float,
        repeat_penalty: bool,
        collect_telemetry: bool,
    ) -> tuple[int, dict[str, float] | None]:
        work = logits_1d[: len(self._new_to_old)] if (self.model.vocab_remap is None and len(logits_1d) > len(self._new_to_old)) else logits_1d
        argmax_idx = int(np.argmax(work))
        if repeat_penalty:
            min_hits = 5 if matcher is not None else 4
            pen = compute_loop_breaker_penalties(
                recent_tokens,
                window=96,
                min_prior_hits=min_hits,
                base_penalty=8.0,
                stop_token_ids=self.tokenizer.stop_token_ids,
            )
            if pen:
                old_to_new = self.model.vocab_remap.old_to_new if self.model.vocab_remap is not None else None
                work, argmax_idx = apply_loop_breaker_penalties_1d(work, pen, old_to_new)

        if matcher is not None and gram is not None:
            def _lazy_full() -> np.ndarray:
                assert full_hidden_1d is not None
                fl = self.model.project_logits(full_hidden_1d[None, None, :], full_vocab=True)
                mx.eval(fl)
                return np.asarray(fl[0, 0].astype(mx.float32))

            tok_id, _ = gram.pick_valid_token(
                matcher,
                work,
                argmax_idx,
                temperature=temperature,
                fallback_to_full_vocab=self.model._cold_lm_head is not None,
                full_logits_1d=_lazy_full if self.model._cold_lm_head is not None else None,
            )
        else:
            tok_id = sample_unconstrained(
                work,
                self._new_to_old,
                argmax_idx,
                temperature=temperature,
            )

        tel: dict[str, float] | None = None
        if collect_telemetry:
            old_to_new = self.model.vocab_remap.old_to_new if self.model.vocab_remap is not None else None
            c_idx = int(old_to_new[tok_id]) if old_to_new is not None else int(tok_id)
            tel = _compute_step_telemetry(work, c_idx)
        return int(tok_id), tel

    def _finalize_item(
        self,
        *,
        prompt_tokens: int,
        answer_ids: list[int],
        reasoning_text: str | None,
        hit_max_tokens: bool,
        schema_obj: dict[str, Any] | None,
        telemetry: list[dict[str, float]] | None,
    ) -> GenerationItem:
        clean_ids = [t for t in answer_ids if t not in self.tokenizer.stop_token_ids]
        text = self.tokenizer.decode(clean_ids, skip_special_tokens=True)
        parsed: Any | None = None
        valid_json = False
        valid_schema = False
        if schema_obj is not None:
            try:
                parsed = json.loads(text)
                valid_json = True
                validate_json_against_schema(parsed, schema_obj)
                valid_schema = True
            except Exception:
                pass
        else:
            stripped = text.strip()
            if stripped.startswith(("{", "[")):
                try:
                    parsed = json.loads(stripped)
                    valid_json = True
                    valid_schema = True
                except Exception:
                    pass

        # Preserve `finish_reason="length"` whenever `max_tokens` was exhausted, even under active grammars!
        finish_reason = "length" if hit_max_tokens else "stop"
        return GenerationItem(
            text=text,
            parsed=parsed,
            token_ids=clean_ids,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=len(clean_ids) + (len(self.tokenizer.encode(reasoning_text)) if reasoning_text else 0),
            valid_json=valid_json,
            valid_schema=valid_schema,
            finish_reason=finish_reason,
            reasoning=reasoning_text,
            telemetry=telemetry,
        )

    def _prefill_single_with_caches(
        self,
        prompt_ids: list[int],
        *,
        prefix_snap: HybridPrefixSnapshot | None,
        suffix_ids: list[int],
        max_tokens: int,
        extra_tail_ids: Sequence[int] = (),
    ) -> tuple[list[DeltaSlotCache | GrowingKVCache], mx.array, list[int]]:
        """Prefill `B=1` using `prefix_snap` (`>=32` LCP), `PrefixTree` (`256`-tok chunks), or cold chunked publish."""
        ext = [int(t) for t in extra_tail_ids]
        kv_alloc = max(256, len(prompt_ids) + len(ext) + max_tokens + 64)
        use_kvi8 = self._resolve_kv_i8(1, kv_alloc)

        if prefix_snap is not None:
            cache = restore_hybrid_cache(prefix_snap, batch_size=1, kv_step=kv_alloc, kv_i8=use_kvi8)
            run_ids = list(suffix_ids) + ext
            inp = mx.array([run_ids], dtype=mx.int32)
            hidden = self.model.forward_hidden(inp, cache)
            return cache, hidden, run_ids

        ptree = self.prefix_tree
        ct = ptree.chunk_tokens if ptree is not None else 0
        if ptree is not None and ct > 0 and len(prompt_ids) > ct:
            match = ptree.match(prompt_ids, limit=len(prompt_ids) - 1, needs_snapshot=True)
            if not match.is_empty and match.snapshot is not None:
                ptree.record(match, restored=True)
                cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)
                gdn_idx = 0
                kv_idx = 0
                for entry in cache:
                    if isinstance(entry, DeltaSlotCache):
                        conv, codes, scales = match.snapshot[gdn_idx]
                        entry.conv_state = conv
                        entry.rec_state = dequantize_gdn_state_block32(codes, scales) if codes is not None else None
                        gdn_idx += 1
                    elif isinstance(entry, GrowingKVCache):
                        k_cat = mx.concatenate([ch[kv_idx][0] for ch in match.chunks], axis=2)
                        v_cat = mx.concatenate([ch[kv_idx][1] for ch in match.chunks], axis=2)
                        entry.update_and_fetch(k_cat, v_cat)
                        kv_idx += 1
                start_pos = match.reused_tokens
                parent_key: str | None = match.keys[-1]
                depth = len(match.keys)
            else:
                ptree.record(match, restored=False)
                cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)
                start_pos = 0
                parent_key = None
                depth = 0

            # Publish any newly crossed 256-token chunks of `prompt_ids` (schema-independent!) when circuit breaker allows
            n_full_chunks = (len(prompt_ids) - 1) // ct
            if ptree.should_write_snapshots() and n_full_chunks > depth:
                last_inserted: str | None = None
                for d in range(depth + 1, n_full_chunks + 1):
                    c_start = (d - 1) * ct
                    c_end = d * ct
                    c_toks = prompt_ids[c_start:c_end]
                    _ = self.model.forward_hidden(mx.array([c_toks], dtype=mx.int32), cache)
                    kv_payload: list[tuple[mx.array, mx.array]] = []
                    for entry in cache:
                        if isinstance(entry, GrowingKVCache) and entry.keys is not None and entry.values is not None:
                            if entry.kv_i8 and entry.k_scales is not None and entry.v_scales is not None:
                                kc = mx.contiguous(entry.keys[:, :, c_start:c_end, :].astype(mx.bfloat16) * entry.k_scales[:, :, c_start:c_end, :])
                                vc = mx.contiguous(entry.values[:, :, c_start:c_end, :].astype(mx.bfloat16) * entry.v_scales[:, :, c_start:c_end, :])
                            else:
                                kc = mx.contiguous(entry.keys[:, :, c_start:c_end, :])
                                vc = mx.contiguous(entry.values[:, :, c_start:c_end, :])
                            mx.eval(kc, vc)
                            kv_payload.append((kc, vc))
                    parent_key = ptree.insert(parent=parent_key, tokens=c_toks, depth=d, chunk=tuple(kv_payload))
                    if parent_key is None:
                        start_pos = c_end
                        break
                    last_inserted = parent_key
                    start_pos = c_end

                if last_inserted is not None:
                    gdn_snap: list[tuple[Any, Any, Any]] = []
                    for entry in cache:
                        if isinstance(entry, DeltaSlotCache):
                            cv = mx.contiguous(entry.conv_state) if entry.conv_state is not None else None
                            if entry.rec_state is not None:
                                q_c, s_c = quantize_gdn_state_block32(entry.rec_state)
                                mx.eval(q_c, s_c)
                                gdn_snap.append((cv, q_c, s_c))
                            else:
                                gdn_snap.append((cv, None, None))
                    ptree.attach_snapshot(last_inserted, tuple(gdn_snap))

            tail_ids = prompt_ids[start_pos:] + ext
            inp = mx.array([tail_ids], dtype=mx.int32)
            hidden = self.model.forward_hidden(inp, cache)
            return cache, hidden, tail_ids

        cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)
        run_all = list(prompt_ids) + ext
        inp = mx.array([run_all], dtype=mx.int32)
        hidden = self.model.forward_hidden(inp, cache)
        return cache, hidden, run_all

    def _decode_single(
        self,
        prompt_ids: list[int],
        *,
        prefix_snap: HybridPrefixSnapshot | None,
        suffix_ids: list[int],
        gram: GrammarHelper | None,
        schema_obj: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        thinking_budget: int,
        speculative: bool | None,
        collect_telemetry: bool,
    ) -> GenerationItem:
        matcher = gram.new_matcher() if gram is not None else None
        telemetry: list[dict[str, float]] | None = [] if collect_telemetry else None
        folded_ff: list[int] = []
        if thinking_budget <= 0 and matcher is not None and gram is not None and gram.upfront_ff:
            hit_stop_up = False
            for t in gram.upfront_ff:
                ti = int(t)
                if ti in self.tokenizer.stop_token_ids:
                    hit_stop_up = True
                    break
                folded_ff.append(ti)
            if len(folded_ff) >= max_tokens:
                cap_ff = folded_ff[:max_tokens]
                is_done_ok = len(folded_ff) == max_tokens and (
                    hit_stop_up or (matcher.is_stopped() and matcher.is_accepting())
                )
                return self._finalize_item(
                    prompt_tokens=len(prompt_ids),
                    answer_ids=cap_ff,
                    reasoning_text=None,
                    hit_max_tokens=not is_done_ok,
                    schema_obj=schema_obj,
                    telemetry=telemetry,
                )
            if hit_stop_up or matcher.is_stopped():
                return self._finalize_item(
                    prompt_tokens=len(prompt_ids),
                    answer_ids=folded_ff,
                    reasoning_text=None,
                    hit_max_tokens=False,
                    schema_obj=schema_obj,
                    telemetry=telemetry,
                )

        cache, hidden, prefill_in = self._prefill_single_with_caches(
            prompt_ids,
            prefix_snap=prefix_snap,
            suffix_ids=suffix_ids,
            max_tokens=max_tokens,
            extra_tail_ids=folded_ff,
        )
        last_h = hidden[:, -1:, :]
        logits = self.model.project_logits(last_h)
        mx.eval(last_h, logits)

        reasoning_text: str | None = None
        # Optional Phase 1: Two-phase budgeted thinking (`thinking_budget >= 384`)
        if thinking_budget > 0:
            think_ids: list[int] = []
            closed_naturally = False
            cur_logits = np.asarray(logits[0, 0].astype(mx.float32))
            cur_h = last_h[0, 0]
            for _ in range(thinking_budget):
                tid, _ = self._select_next_token(
                    cur_logits,
                    full_hidden_1d=cur_h,
                    matcher=None,
                    gram=None,
                    recent_tokens=think_ids,
                    temperature=temperature,
                    repeat_penalty=True,
                    collect_telemetry=False,
                )
                if tid in self.tokenizer.stop_token_ids:
                    closed_naturally = True
                    break
                think_ids.append(tid)
                if self.tokenizer.think_close_id is not None and tid == self.tokenizer.think_close_id:
                    closed_naturally = True
                    break
                # Check string-level `</think>` for multi-token BPE splits
                if len(think_ids) >= 2 and "</think>" in self.tokenizer.decode(think_ids[-4:], skip_special_tokens=False):
                    closed_naturally = True
                    break
                step_in = mx.array([[tid]], dtype=mx.int32)
                sh = self.model.forward_hidden(step_in, cache)
                sl = self.model.project_logits(sh)
                mx.eval(sh, sl)
                cur_h = sh[0, 0]
                cur_logits = np.asarray(sl[0, 0].astype(mx.float32))

            raw_think = self.tokenizer.decode(think_ids, skip_special_tokens=False)
            reasoning_text = raw_think.split("</think>")[0].strip()
            close_suffix = (
                self.tokenizer.encode("\n\n", add_special_tokens=False)
                if closed_naturally
                else self.tokenizer.encode_think_close_suffix(trailing_newline_before_close=True)
            )
            suf_arr = mx.array([close_suffix], dtype=mx.int32)
            sh = self.model.forward_hidden(suf_arr, cache)
            last_h = sh[:, -1:, :]
            logits = self.model.project_logits(last_h)
            mx.eval(last_h, logits)

        answer_ids: list[int] = list(folded_ff)
        hit_max_tokens = False

        # When two-phase thinking ran first, advance over upfront deterministic grammar prefix after `</think>\n\n`
        if thinking_budget > 0 and matcher is not None and gram is not None and gram.upfront_ff:
            ff0: list[int] = []
            hit_stop_t = False
            for t in gram.upfront_ff:
                ti = int(t)
                if ti in self.tokenizer.stop_token_ids:
                    hit_stop_t = True
                    break
                ff0.append(ti)
            ff0_cap = ff0[:max_tokens]
            answer_ids.extend(ff0_cap)
            if len(answer_ids) >= max_tokens:
                hit_max_tokens = not (
                    len(ff0) == max_tokens and (hit_stop_t or (matcher.is_stopped() and matcher.is_accepting()))
                )
            elif not hit_stop_t and not matcher.is_stopped() and ff0_cap:
                ff_arr = mx.array([ff0_cap], dtype=mx.int32)
                sh = self.model.forward_hidden(ff_arr, cache)
                last_h = sh[:, -1:, :]
                logits = self.model.project_logits(last_h)
                mx.eval(last_h, logits)

        use_spec = (
            should_use_speculative_qwen(
                1,
                temperature,
                speculative=speculative,
                has_drafter_or_lookup=True,
                has_native_mtp=self.mtp_drafter is not None,
                hidden_size=self.model.cfg.hidden_size,
            )
            and matcher is None
            and not collect_telemetry
        )

        mtp_cache = self.mtp_drafter.make_cache() if (use_spec and self.mtp_drafter is not None) else None
        if mtp_cache is not None and self.mtp_drafter is not None and len(prefill_in) >= 2:
            # Seed 1-layer MTP cache over prompt tail (capped to 128 tokens for O(1) latency)
            tail_n = min(len(prefill_in) - 1, 128)
            seed_next = mx.array([prefill_in[-tail_n:]], dtype=mx.int32)
            seed_prev_h = hidden[:, -(tail_n + 1) : -1, :]
            _ = self.mtp_drafter.forward_hidden(seed_next, seed_prev_h, self.model, mtp_cache)

        cur_logits_1d = np.asarray(logits[0, 0].astype(mx.float32))
        cur_h_1d = last_h[0, 0]
        history_ids = list(prompt_ids) + list(answer_ids)

        # Sample first non-FF token from prefill logits
        if len(answer_ids) < max_tokens and (matcher is None or not matcher.is_stopped()):
            first_tok, first_tel = self._select_next_token(
                cur_logits_1d,
                full_hidden_1d=cur_h_1d,
                matcher=matcher,
                gram=gram,
                recent_tokens=answer_ids,
                temperature=temperature,
                repeat_penalty=True,
                collect_telemetry=collect_telemetry,
            )
            if first_tel is not None and telemetry is not None:
                telemetry.append(first_tel)
            if first_tok not in self.tokenizer.stop_token_ids:
                answer_ids.append(first_tok)
                history_ids.append(first_tok)
                if matcher is not None:
                    matcher.consume_token(first_tok)
                    if not matcher.is_stopped():
                        ff_more = [int(t) for t in matcher.compute_ff_tokens() if int(t) not in self.tokenizer.stop_token_ids]
                        if ff_more:
                            rem = max_tokens - len(answer_ids)
                            used_ff = ff_more[:rem]
                            matcher.consume_tokens(used_ff)
                            answer_ids.extend(used_ff)
                            history_ids.extend(used_ff)
                            pending_tokens = [first_tok] + used_ff
                        else:
                            pending_tokens = [first_tok]
                    else:
                        pending_tokens = []
                else:
                    pending_tokens = [first_tok]
            else:
                pending_tokens = []
        else:
            pending_tokens = []

        draft_k = recommended_draft_k(1, has_grammar=gram is not None) if use_spec else 0

        for _step in range(max_tokens):
            if not pending_tokens or len(answer_ids) >= max_tokens or (matcher is not None and matcher.is_stopped()):
                break
            if use_spec and len(pending_tokens) == 1 and (max_tokens - len(answer_ids)) >= 2:
                last_tok = pending_tokens[0]
                cap_k = min(draft_k, max_tokens - len(answer_ids))
                drafts: list[int] = []
                if self.mtp_drafter is not None and mtp_cache is not None:
                    mtp_prev_off = mtp_cache.offset
                    d_tok = last_tok
                    d_prev_h = cur_h_1d[None, None, :]
                    for _ in range(cap_k):
                        dh = self.mtp_drafter.forward_hidden(
                            mx.array([[d_tok]], dtype=mx.int32),
                            d_prev_h,
                            self.model,
                            mtp_cache,
                        )
                        dl = self.model.project_logits(dh)
                        d_idx_arr = mx.argmax(dl[0, 0], axis=-1)
                        mx.eval(dh, d_idx_arr)
                        d_idx = int(np.asarray(d_idx_arr).item())
                        d_next = int(self._new_to_old[d_idx])
                        drafts.append(d_next)
                        if d_next in self.tokenizer.stop_token_ids:
                            break
                        d_tok = d_next
                        d_prev_h = dh
                else:
                    drafts = draft_prompt_lookup_tokens(history_ids, cap_k)

                if drafts:
                    def _pick_greedy(step_idx: int, step_logits: np.ndarray) -> int:
                        hist_slice = answer_ids + drafts[:step_idx]
                        t_out, _ = self._select_next_token(
                            step_logits,
                            full_hidden_1d=None,
                            matcher=None,
                            gram=None,
                            recent_tokens=hist_slice,
                            temperature=0.0,
                            repeat_penalty=True,
                            collect_telemetry=False,
                        )
                        return t_out

                    emitted, acc_h, n_drafts_matched = verify_draft_sequence_b1(
                        self.model,
                        cache,
                        last_tok,
                        drafts,
                        pick_token_fn=_pick_greedy,
                        stop_token_ids=self.tokenizer.stop_token_ids,
                    )
                    if mtp_cache is not None and self.mtp_drafter is not None:
                        # Keep MTP KV cache aligned with accepted drafts (`1` last_tok + `n_drafts_matched` drafts)
                        mtp_cache.rewind_to(min(mtp_cache.offset, mtp_prev_off + 1 + n_drafts_matched))

                    cur_h_1d = acc_h[0, -1]
                    stopped = False
                    last_emitted: int | None = None
                    for etok in emitted:
                        if etok in self.tokenizer.stop_token_ids:
                            stopped = True
                            break
                        if len(answer_ids) >= max_tokens:
                            hit_max_tokens = True
                            stopped = True
                            break
                        answer_ids.append(etok)
                        history_ids.append(etok)
                        last_emitted = etok

                    if stopped or last_emitted is None:
                        pending_tokens = []
                        break
                    pending_tokens = [last_emitted]
                    continue

            # Standard 1-step (or multi-FF-token) advance
            step_arr = mx.array([pending_tokens], dtype=mx.int32)
            sh = self.model.forward_hidden(step_arr, cache)
            last_h = sh[:, -1:, :]
            sl = self.model.project_logits(last_h)
            mx.eval(last_h, sl)
            cur_h_1d = last_h[0, 0]
            cur_logits_1d = np.asarray(sl[0, 0].astype(mx.float32))

            nxt_tok, nxt_tel = self._select_next_token(
                cur_logits_1d,
                full_hidden_1d=cur_h_1d,
                matcher=matcher,
                gram=gram,
                recent_tokens=answer_ids,
                temperature=temperature,
                repeat_penalty=True,
                collect_telemetry=collect_telemetry,
            )
            if nxt_tel is not None and telemetry is not None:
                telemetry.append(nxt_tel)
            if nxt_tok in self.tokenizer.stop_token_ids:
                pending_tokens = []
                break
            answer_ids.append(nxt_tok)
            history_ids.append(nxt_tok)

            if matcher is not None:
                matcher.consume_token(nxt_tok)
                if matcher.is_stopped():
                    pending_tokens = []
                    break
                ff_more = [int(t) for t in matcher.compute_ff_tokens() if int(t) not in self.tokenizer.stop_token_ids]
                if ff_more and len(answer_ids) < max_tokens:
                    rem = max_tokens - len(answer_ids)
                    used_ff = ff_more[:rem]
                    matcher.consume_tokens(used_ff)
                    answer_ids.extend(used_ff)
                    history_ids.extend(used_ff)
                    pending_tokens = [nxt_tok] + used_ff
                else:
                    pending_tokens = [nxt_tok]
            else:
                pending_tokens = [nxt_tok]

        if len(answer_ids) >= max_tokens and not (
            matcher is not None and matcher.is_stopped() and matcher.is_accepting()
        ):
            hit_max_tokens = True

        return self._finalize_item(
            prompt_tokens=len(prompt_ids),
            answer_ids=answer_ids,
            reasoning_text=reasoning_text,
            hit_max_tokens=hit_max_tokens,
            schema_obj=schema_obj,
            telemetry=telemetry,
        )

    def _decode_wave_batched(
        self,
        wave_prompt_ids: list[list[int]],
        *,
        prefix_snap: HybridPrefixSnapshot | None,
        wave_suffix_ids: list[list[int]],
        grammars: list[GrammarHelper | None],
        schema_objs: list[dict[str, Any] | None],
        max_tokens: int,
        temperature: float,
        collect_telemetry: bool,
    ) -> list[GenerationItem]:
        """Right-padded batched prefill + multi-token FF + active-row `lm_head` decode wave (`B >= 2`, zero pad leakage)."""
        b_sz = len(wave_prompt_ids)
        pad_id = self.tokenizer.eos_token_id
        matchers = [g.new_matcher() if g is not None else None for g in grammars]
        ans_ids: list[list[int]] = [[] for _ in range(b_sz)]
        tels: list[list[dict[str, float]] | None] = [
            ([] if collect_telemetry else None) for _ in range(b_sz)
        ]
        done = [False] * b_sz
        hit_max = [False] * b_sz

        # Fold each slot's deterministic `upfront_ff` directly into its prefill suffix (saves 1 forward + 1 lm_head!)
        aug_sufs: list[list[int]] = []
        for i in range(b_sz):
            g = grammars[i]
            m = matchers[i]
            up_clean: list[int] = []
            hit_stop_up = False
            if g is not None and g.upfront_ff:
                for t in g.upfront_ff:
                    ti = int(t)
                    if ti in self.tokenizer.stop_token_ids:
                        hit_stop_up = True
                        break
                    up_clean.append(ti)
            if len(up_clean) >= max_tokens:
                ans_ids[i] = up_clean[:max_tokens]
                done[i] = True
                is_done_ok = len(up_clean) == max_tokens and (
                    hit_stop_up or (m is not None and m.is_stopped() and m.is_accepting())
                )
                hit_max[i] = not is_done_ok
                aug_sufs.append(list(wave_suffix_ids[i]))
            elif hit_stop_up or (m is not None and m.is_stopped()):
                ans_ids[i] = up_clean
                done[i] = True
                aug_sufs.append(list(wave_suffix_ids[i]))
            else:
                ans_ids[i] = list(up_clean)
                aug_sufs.append(list(wave_suffix_ids[i]) + up_clean)

        if all(done):
            return [
                self._finalize_item(
                    prompt_tokens=len(wave_prompt_ids[i]),
                    answer_ids=ans_ids[i],
                    reasoning_text=None,
                    hit_max_tokens=hit_max[i],
                    schema_obj=schema_objs[i],
                    telemetry=tels[i],
                )
                for i in range(b_sz)
            ]

        suf_lens = [len(s) for s in aug_sufs]
        max_suf = max(suf_lens)
        any_ragged = any(sl < max_suf for sl in suf_lens)
        padded_suf = [s + [pad_id] * (max_suf - len(s)) for s in aug_sufs]

        lcp_len = len(prefix_snap.prefix_tokens) if prefix_snap is not None else 0
        kv_alloc = max(256, lcp_len + max_suf + max_tokens + 64)
        use_kvi8 = self._resolve_kv_i8(b_sz, kv_alloc)
        if prefix_snap is not None:
            cache = restore_hybrid_cache(prefix_snap, batch_size=b_sz, kv_step=kv_alloc, kv_i8=use_kvi8)
        else:
            cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)

        # Right-padded suffixes stay contiguous with `LCP[-3:]` (`0` conv1d boundary corruption, `"causal"` SDPA!)
        if any_ragged:
            valid_2d = np.zeros((b_sz, max_suf, 1), dtype=np.float32)
            for i, sl in enumerate(suf_lens):
                valid_2d[i, :sl, 0] = 1.0
            valid_scale: mx.array | None = mx.array(valid_2d, dtype=mx.bfloat16)
            valid_lens_mx: mx.array | None = mx.array(suf_lens, dtype=mx.int32)
        else:
            valid_scale = None
            valid_lens_mx = None

        inp = mx.array(padded_suf, dtype=mx.int32)
        h = self.model.forward_hidden(
            inp,
            cache,
            valid_token_scale=valid_scale,
            valid_lengths=valid_lens_mx,
            attn_mask="causal",
            rope_trig=None,
        )
        if any_ragged:
            last_idx = mx.array([sl - 1 for sl in suf_lens], dtype=mx.int32)
            cur_h = mx.take_along_axis(h, last_idx[:, None, None], axis=1)[:, 0, :]
        else:
            cur_h = h[:, -1, :]

        def _project_needed_rows(h_2d: mx.array, need_rows: list[int]) -> dict[int, np.ndarray]:
            if not need_rows:
                mx.eval(h_2d)
                return {}
            if len(need_rows) == b_sz:
                sl_all = self.model.project_logits(h_2d[:, None, :])[:, 0, :]
                mx.eval(h_2d, sl_all)
                arr_all = np.asarray(sl_all.astype(mx.float32))
                return {r: arr_all[r] for r in range(b_sz)}
            idx_mx = mx.array(need_rows, dtype=mx.int32)
            sub_h = mx.take(h_2d, idx_mx, axis=0)[:, None, :]
            sl_sub = self.model.project_logits(sub_h)[:, 0, :]
            mx.eval(h_2d, sl_sub)
            arr_sub = np.asarray(sl_sub.astype(mx.float32))
            return {r: arr_sub[pos] for pos, r in enumerate(need_rows)}

        cur_logits_map = _project_needed_rows(cur_h, [i for i in range(b_sz) if not done[i]])
        true_pos = np.array([lcp_len + sl for sl in suf_lens], dtype=np.int32)
        kv_cursor = lcp_len + max_suf
        suf_end_arr = mx.array([lcp_len + sl for sl in suf_lens], dtype=mx.int32)[:, None, None, None] if any_ragged else None
        pending_q: list[list[int]] = [[] for _ in range(b_sz)]

        for _step in range(max_tokens + 1):
            for i in range(b_sz):
                if done[i] or pending_q[i]:
                    continue
                m = matchers[i]
                g = grammars[i]
                if len(ans_ids[i]) >= max_tokens:
                    done[i] = True
                    hit_max[i] = not (m is not None and m.is_stopped() and m.is_accepting())
                    continue
                if m is not None and m.is_stopped():
                    done[i] = True
                    continue

                tid, tel = self._select_next_token(
                    cur_logits_map[i],
                    full_hidden_1d=cur_h[i],
                    matcher=m,
                    gram=g,
                    recent_tokens=ans_ids[i],
                    temperature=temperature,
                    repeat_penalty=True,
                    collect_telemetry=collect_telemetry,
                )
                if tel is not None and tels[i] is not None:
                    tels[i].append(tel)  # type: ignore[union-attr]
                if tid in self.tokenizer.stop_token_ids:
                    done[i] = True
                    continue
                ans_ids[i].append(tid)

                if m is not None:
                    m.consume_token(tid)
                    if m.is_stopped():
                        done[i] = True
                        continue
                    if len(ans_ids[i]) >= max_tokens:
                        done[i] = True
                        hit_max[i] = True
                        continue
                    ff_clean: list[int] = []
                    stopped_by_ff = False
                    for ft in m.compute_ff_tokens():
                        fti = int(ft)
                        if fti in self.tokenizer.stop_token_ids:
                            stopped_by_ff = True
                            break
                        if len(ans_ids[i]) + len(ff_clean) >= max_tokens:
                            break
                        ff_clean.append(fti)
                    if ff_clean:
                        m.consume_tokens(ff_clean)
                        ans_ids[i].extend(ff_clean)
                    if stopped_by_ff or m.is_stopped():
                        done[i] = True
                        continue
                    if len(ans_ids[i]) >= max_tokens:
                        done[i] = True
                        hit_max[i] = True
                        continue
                    pending_q[i] = [tid] + ff_clean
                else:
                    if len(ans_ids[i]) >= max_tokens:
                        done[i] = True
                        hit_max[i] = True
                        continue
                    pending_q[i] = [tid]

            alive = [i for i in range(b_sz) if not done[i]]
            if not alive:
                break

            # Advance `step_w >= 1` common queued tokens across all alive slots in a single forward pass!
            step_w = min(len(pending_q[i]) for i in alive)
            step_rows: list[list[int]] = []
            for i in range(b_sz):
                if done[i]:
                    step_rows.append([pad_id] * step_w)
                else:
                    step_rows.append(pending_q[i][:step_w])
                    del pending_q[i][:step_w]

            step_inp = mx.array(step_rows, dtype=mx.int32)
            kv_prev = kv_cursor
            kv_now = kv_prev + step_w
            kv_cursor = kv_now

            if any_ragged:
                assert suf_end_arr is not None
                p_mat = true_pos[:, None] + np.arange(step_w, dtype=np.int32)[None, :]
                r_trig_s: tuple[mx.array, mx.array] | None = self.model.precompute_rope_trig(mx.array(p_mat, dtype=mx.int32))
                k_pos = mx.arange(kv_now, dtype=mx.int32)[None, None, None, :]
                in_pad = (k_pos >= suf_end_arr) & (k_pos < (lcp_len + max_suf))
                if step_w == 1:
                    allow = ~in_pad
                else:
                    q_kv = mx.arange(step_w, dtype=mx.int32)[None, None, :, None] + kv_prev
                    allow = (~in_pad) & (k_pos <= q_kv)
                d_mask_s: str | mx.array = mx.where(
                    allow, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16)
                )
            else:
                r_trig_s = None
                d_mask_s = "causal"

            v_scale_s = (
                mx.array([[[0.0] if done[i] else [1.0]] * step_w for i in range(b_sz)], dtype=mx.bfloat16)
                if len(alive) < b_sz
                else None
            )
            sh = self.model.forward_hidden(
                step_inp,
                cache,
                valid_token_scale=v_scale_s,
                attn_mask=d_mask_s,
                rope_trig=r_trig_s,
            )
            cur_h = sh[:, -1, :]
            need_rows = [i for i in alive if not pending_q[i]]
            cur_logits_map = _project_needed_rows(cur_h, need_rows)
            true_pos += step_w

        return [
            self._finalize_item(
                prompt_tokens=len(wave_prompt_ids[i]),
                answer_ids=ans_ids[i],
                reasoning_text=None,
                hit_max_tokens=hit_max[i],
                schema_obj=schema_objs[i],
                telemetry=tels[i],
            )
            for i in range(b_sz)
        ]

    def generate_batch(
        self,
        prompts: Sequence[PromptInput],
        *,
        system_prompt: str | None = None,
        json_schema: str | Mapping[str, Any] | None = None,
        json_schemas: Sequence[str | Mapping[str, Any] | None] | None = None,
        per_doc_grammars: Sequence[str | Mapping[str, Any] | GrammarHelper | None] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        batch_size: int | None = None,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool | None = None,
        enable_thinking: bool | None = None,
        thinking_budget_tokens: int | None = None,
        allow_short_thinking_budget: bool = False,
        reasoning_prefix_field: str | None = None,
        collect_telemetry: bool = False,
        **_ignored_kwargs: Any,
    ) -> GenerationBatchResult:
        """Generate structured or free-text completions over a batch of prompts with LCP hybrid caching."""
        if not prompts:
            return GenerationBatchResult(items=[], stats={"docs": 0, "elapsed_sec": 0.0})

        t0 = time.perf_counter()
        eff_think = self.default_enable_thinking if enable_thinking is None else bool(enable_thinking)
        raw_budget = self.default_thinking_budget if thinking_budget_tokens is None else int(thinking_budget_tokens)
        eff_budget = validate_thinking_budget(
            raw_budget if raw_budget > 0 else (None if eff_think else 0),
            enable_thinking=eff_think,
            allow_short_thinking_budget=allow_short_thinking_budget,
        )
        use_two_phase = eff_budget > 0

        n_docs = len(prompts)
        grammars: list[GrammarHelper | None] = []
        schema_objs: list[dict[str, Any] | None] = []
        if per_doc_grammars is not None:
            if len(per_doc_grammars) != n_docs:
                raise ValueError(f"per_doc_grammars len {len(per_doc_grammars)} != prompts len {n_docs}")
            for g_spec in per_doc_grammars:
                if g_spec is None:
                    grammars.append(None)
                    schema_objs.append(None)
                elif isinstance(g_spec, GrammarHelper):
                    grammars.append(g_spec)
                    schema_objs.append(g_spec.schema_obj)
                else:
                    gh = self._get_grammar_helper(g_spec, whitespace_flexible)
                    grammars.append(gh)
                    schema_objs.append(gh.schema_obj)
        else:
            per_item_schemas: list[str | Mapping[str, Any] | None]
            if json_schemas is not None:
                if len(json_schemas) != n_docs:
                    raise ValueError(f"json_schemas len {len(json_schemas)} != prompts len {n_docs}")
                per_item_schemas = list(json_schemas)
            else:
                per_item_schemas = [json_schema] * n_docs

            for sch in per_item_schemas:
                if sch is None:
                    grammars.append(None)
                    schema_objs.append(None)
                else:
                    eff_sch = (
                        inject_reasoning_prefix_property(sch, field_name=reasoning_prefix_field)
                        if reasoning_prefix_field
                        else sch
                    )
                    gh = self._get_grammar_helper(eff_sch, whitespace_flexible)
                    grammars.append(gh)
                    schema_objs.append(gh.schema_obj)

        encoded = [
            self.tokenizer.encode_chat(
                p,
                system_prompt=system_prompt,
                enable_thinking=use_two_phase,
            )
            for p in prompts
        ]

        sys_prefix_ids: list[int] = (
            self.tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
            if system_prompt and all(isinstance(p, str) for p in prompts)
            else []
        )

        with self._lock:
            base_wave_bs = int(batch_size) if batch_size is not None else min(n_docs, 64)
            eff_wave_bs = self._memory_governor.adapt(
                max(1, base_wave_bs),
                base_cache_limit_mb=self._mlx_cache_limit_mb,
                apply_mlx=True,
            ).b_slots
            results: list[GenerationItem | None] = [None] * n_docs
            max_lcp_seen = 0

            if n_docs == 1 or eff_wave_bs <= 1 or use_two_phase:
                b1_snap: HybridPrefixSnapshot | None = None
                if (
                    self.max_prefix_cache_entries > 0
                    and len(sys_prefix_ids) >= 32
                    and all(len(enc) > len(sys_prefix_ids) and enc[: len(sys_prefix_ids)] == sys_prefix_ids for enc in encoded)
                ):
                    b1_snap = self._lookup_or_build_prefix(sys_prefix_ids)
                    max_lcp_seen = len(sys_prefix_ids)
                for idx in range(n_docs):
                    suf_i = encoded[idx][len(sys_prefix_ids) :] if b1_snap is not None else encoded[idx]
                    results[idx] = self._decode_single(
                        encoded[idx],
                        prefix_snap=b1_snap,
                        suffix_ids=suf_i,
                        gram=grammars[idx],
                        schema_obj=schema_objs[idx],
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        thinking_budget=eff_budget,
                        speculative=speculative,
                        collect_telemetry=collect_telemetry,
                    )
            else:
                # Group identical GrammarHelper instances first so batched upfront_ff & multi-token FF jumps align!
                order = sorted(
                    range(n_docs),
                    key=lambda i: (0 if grammars[i] is None else id(grammars[i]), len(encoded[i])),
                )
                cursor = 0
                for _wave in range(n_docs):
                    if cursor >= n_docs:
                        break
                    cur_bs = self._memory_governor.adapt(
                        eff_wave_bs,
                        base_cache_limit_mb=self._mlx_cache_limit_mb,
                        apply_mlx=True,
                    ).b_slots
                    chunk_idx = order[cursor : cursor + max(1, cur_bs)]
                    cursor += len(chunk_idx)
                    if len(chunk_idx) == 1:
                        i0 = chunk_idx[0]
                        b1_s = (
                            self._lookup_or_build_prefix(sys_prefix_ids)
                            if (self.max_prefix_cache_entries > 0 and len(sys_prefix_ids) >= 32 and len(encoded[i0]) > len(sys_prefix_ids) and encoded[i0][: len(sys_prefix_ids)] == sys_prefix_ids)
                            else None
                        )
                        if b1_s is not None:
                            max_lcp_seen = max(max_lcp_seen, len(sys_prefix_ids))
                        results[i0] = self._decode_single(
                            encoded[i0],
                            prefix_snap=b1_s,
                            suffix_ids=encoded[i0][len(sys_prefix_ids) :] if b1_s is not None else encoded[i0],
                            gram=grammars[i0],
                            schema_obj=schema_objs[i0],
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            thinking_budget=0,
                            speculative=speculative,
                            collect_telemetry=collect_telemetry,
                        )
                    else:
                        wave_encs = [encoded[i] for i in chunk_idx]
                        w_lcp, w_sufs = split_lcp_ids(wave_encs, min_lcp=32)
                        w_snap = self._lookup_or_build_prefix(w_lcp) if (w_lcp and self.max_prefix_cache_entries > 0) else None
                        if w_snap is None and w_lcp:
                            w_sufs = wave_encs
                        elif w_snap is not None:
                            max_lcp_seen = max(max_lcp_seen, len(w_lcp))
                        wave_items = self._decode_wave_batched(
                            wave_encs,
                            prefix_snap=w_snap,
                            wave_suffix_ids=w_sufs,
                            grammars=[grammars[i] for i in chunk_idx],
                            schema_objs=[schema_objs[i] for i in chunk_idx],
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            collect_telemetry=collect_telemetry,
                        )
                        for i, it in zip(chunk_idx, wave_items):
                            results[i] = it

        elapsed = time.perf_counter() - t0
        final_items = [r for r in results if r is not None]
        total_comp = sum(it.completion_tokens for it in final_items)
        return GenerationBatchResult(
            items=final_items,
            stats={
                "docs": n_docs,
                "lcp_tokens": max_lcp_seen,
                "prefix_tree_reused_tokens": int(self.prefix_tree.reused_tokens) if self.prefix_tree is not None else 0,
                "elapsed_sec": round(elapsed, 4),
                "docs_per_sec": round(n_docs / max(elapsed, 1e-9), 3),
                "decode_tok_s": round(total_comp / max(elapsed, 1e-9), 2),
                "mode": mode or self.default_mode,
            },
        )

    def generate(
        self,
        prompt: PromptInput,
        *,
        system_prompt: str | None = None,
        json_schema: str | Mapping[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool | None = None,
        enable_thinking: bool | None = None,
        thinking_budget_tokens: int | None = None,
        allow_short_thinking_budget: bool = False,
        reasoning_prefix_field: str | None = None,
        collect_telemetry: bool = False,
        **extra_kwargs: Any,
    ) -> GenerationItem:
        batch_res = self.generate_batch(
            [prompt],
            system_prompt=system_prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=1,
            mode=mode,
            speculative=speculative,
            whitespace_flexible=whitespace_flexible,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            allow_short_thinking_budget=allow_short_thinking_budget,
            reasoning_prefix_field=reasoning_prefix_field,
            collect_telemetry=collect_telemetry,
            **extra_kwargs,
        )
        return batch_res.items[0]
