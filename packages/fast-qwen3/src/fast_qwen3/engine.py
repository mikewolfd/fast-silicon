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
    safe_argmax_1d,
    sample_unconstrained,
    validate_json_against_schema,
)
from fast_shared.kv_quant import KVMode, normalize_kv_mode
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
    compact_hybrid_cache,
    restore_hybrid_cache,
    snapshot_hybrid_cache,
)
from .multimodal_batch import (
    batched_multimodal_generate,
    pack_length_waves,
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


def _safe_argmax_with_nan_guard(logits_2d: mx.array) -> mx.array:
    """Compute `int32` argmax along `axis=-1`, returning `-1` for any row containing `NaN`."""
    am = mx.argmax(logits_2d, axis=-1).astype(mx.int32)
    isnan_fn = getattr(mx, "isnan", None)
    any_fn = getattr(mx, "any", None)
    where_fn = getattr(mx, "where", None)
    full_fn = getattr(mx, "full", None)
    if callable(isnan_fn) and callable(any_fn) and callable(where_fn) and callable(full_fn):
        has_nan = any_fn(isnan_fn(logits_2d), axis=-1)
        neg1 = full_fn(am.shape, -1, dtype=mx.int32)
        am = where_fn(has_nan, neg1, am)
    return am


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
        mtp_drafter: Qwen3MtpDrafter | bool | None = None,
        lora_adapter: str | Path | None = None,
        vocab_out: int | None = None,
        default_mode: str = "auto",
        default_enable_thinking: bool = False,
        default_thinking_budget: int = 0,
        max_prefix_cache_entries: int = 8,
        prefix_cache_mb: int | str | None = "auto",
        kv_i8: bool | str = "auto",
        mlx_cache_limit_mb: int = 256,
        max_b_slots: int | None = None,
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
        self._vision_model: Any | None = None
        self._memory_governor = LiveMemoryGovernor(min_b_slots=1)

        try:
            host_ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            host_ram_bytes = 24 * (1 << 30)
        self._is_low_ram: bool = 0 < host_ram_bytes <= (32 * (1 << 30))
        try:
            from fast_shared.hw_check import supports_apple10_nax

            self._is_apple10: bool = bool(supports_apple10_nax())
        except Exception:
            self._is_apple10 = False
        self.default_max_b_slots: int = (
            max(1, int(max_b_slots)) if max_b_slots is not None else (8 if self._is_low_ram else 64)
        )
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

        self._prev_cache_limit: int | None = None
        if mlx_cache_limit_mb > 0 and hasattr(mx, "set_cache_limit"):
            try:
                self._prev_cache_limit = int(mx.set_cache_limit(int(mlx_cache_limit_mb) << 20))
            except Exception:
                self._prev_cache_limit = None

        self._prev_wired_limit: int | None = None
        if hasattr(mx, "set_wired_limit") and hasattr(mx, "device_info"):
            try:
                rec_ws = int(mx.device_info().get("max_recommended_working_set_size", 0))
                if rec_ws > 0:
                    self._prev_wired_limit = int(mx.set_wired_limit(rec_ws))
            except Exception:
                self._prev_wired_limit = None

        self._governor_shed_cb = self._on_memory_shed
        self._memory_governor.register_shed_callback(self._governor_shed_cb)

        if model is not None and tokenizer is not None:
            self.model_dir: Path | None = model.model_dir
            self.model = model
            self.tokenizer = tokenizer
            self.mtp_drafter = None if isinstance(mtp_drafter, bool) else mtp_drafter
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
            if mtp_drafter is False:
                self.mtp_drafter = None
            elif mtp_drafter is None or mtp_drafter is True:
                self.mtp_drafter = Qwen3MtpDrafter.from_mtp_weights(loaded_model, mtp_w)
            else:
                self.mtp_drafter = mtp_drafter

        if lora_adapter is not None:
            self.model.load_lora_adapter(lora_adapter)

        tok_real_vocab = int(self.tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True))
        self._new_to_old: np.ndarray = (
            self.model.vocab_remap.new_to_old
            if self.model.vocab_remap is not None
            else np.arange(min(self.model.vocab_n, tok_real_vocab), dtype=np.int32)
        )
        self._new_to_old_mx: mx.array | None = (
            mx.array(self._new_to_old, dtype=mx.int32) if self.model.vocab_remap is not None else None
        )

        if warmup:
            self._run_warmup()

    def _run_warmup(self) -> None:
        cache = self.model.make_cache(kv_step=32)
        dummy = mx.array([[1, 2]], dtype=mx.int32)
        logits = self.model(dummy, cache)
        mx.eval(logits)
        mx.clear_cache()

    def _on_memory_shed(self, *_args: Any) -> None:
        self._grammar_cache.clear()
        self._prefix_cache.clear()
        if self.prefix_tree is not None:
            self.prefix_tree.clear()
        if hasattr(self, "model") and hasattr(self.model, "clear_mpp_packed_weights"):
            self.model.clear_mpp_packed_weights()

    def close(self) -> None:
        cb = getattr(self, "_governor_shed_cb", None)
        if cb is not None and hasattr(self, "_memory_governor"):
            self._memory_governor.unregister_shed_callback(cb)
        self._prefix_cache.clear()
        if self.prefix_tree is not None:
            self.prefix_tree.clear()
        self._grammar_cache.clear()
        self._vision_model = None
        if hasattr(self, "model") and hasattr(self.model, "clear_mpp_packed_weights"):
            self.model.clear_mpp_packed_weights()
        gc.collect()
        prev_cl = getattr(self, "_prev_cache_limit", None)
        if prev_cl is not None and hasattr(mx, "set_cache_limit"):
            try:
                mx.set_cache_limit(prev_cl)
            except Exception:
                pass
            self._prev_cache_limit = None
        prev_wl = getattr(self, "_prev_wired_limit", None)
        if prev_wl is not None and hasattr(mx, "set_wired_limit"):
            try:
                mx.set_wired_limit(prev_wl)
            except Exception:
                pass
            self._prev_wired_limit = None
        mx.clear_cache()

    def __enter__(self) -> FastQwen3Engine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _resolve_kv_i8(self, batch_size: int, kv_alloc: int) -> KVMode:
        """Resolve lossless KV precision (`'bf16'` short, `'q4_quarot'` long/heavy batch `3.56x` KV & `2.13x` faster than `q8`)."""
        if isinstance(self.kv_i8_mode, bool):
            return "q8" if self.kv_i8_mode else "bf16"
        mode_s = str(self.kv_i8_mode).strip().lower()
        if mode_s == "auto":
            q_ctx_cutoff = 8192 if self._is_low_ram else 16384
            pressure_cutoff = 262_144 if self._is_low_ram else 524_288
            bs = max(1, int(batch_size))
            ctx = max(1, int(kv_alloc))
            if ctx >= q_ctx_cutoff or (bs * ctx) >= pressure_cutoff:
                return "q4_quarot" if (int(self.model.cfg.head_dim) % 64 == 0) else "q8"
            return "bf16"
        return normalize_kv_mode(mode_s)

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

    def _restore_from_tree_and_advance_chunks(
        self,
        token_ids: Sequence[int],
        *,
        kv_alloc: int,
        use_kvi8: bool | str,
        limit: int,
    ) -> tuple[list[DeltaSlotCache | GrowingKVCache], int]:
        """Restore longest cached `PrefixTree` branch and atomically publish newly crossed `256`-tok chunks."""
        ptree = self.prefix_tree
        ct = ptree.chunk_tokens if ptree is not None else 0
        if ptree is None or ct <= 0 or limit < ct:
            return self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8), 0

        ids_list = [int(t) for t in token_ids]
        match = ptree.match(ids_list, limit=limit, needs_snapshot=True)
        if not match.is_empty and match.snapshot is not None:
            ptree.record(match, restored=True)
            cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)
            gdn_idx = 0
            kv_idx = 0
            for entry in cache:
                if isinstance(entry, DeltaSlotCache):
                    conv, codes, scales = match.snapshot[gdn_idx]
                    entry.conv_state = conv
                    entry.rec_state = (
                        dequantize_gdn_state_block32(codes, scales, out_dtype=mx.bfloat16)
                        if codes is not None
                        else None
                    )
                    gdn_idx += 1
                elif isinstance(entry, GrowingKVCache):
                    entry.append_chunk_payloads([ch[kv_idx] for ch in match.chunks])
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

        # Publish each newly crossed chunk atomically with its exact GDN snapshot (roll back if budget exceeded).
        n_full_chunks = limit // ct
        if ptree.should_write_snapshots() and n_full_chunks > depth:
            for d in range(depth + 1, n_full_chunks + 1):
                c_start = (d - 1) * ct
                c_end = d * ct
                c_toks = ids_list[c_start:c_end]
                _ = self.model.forward_hidden(mx.array([c_toks], dtype=mx.int32), cache)
                kv_payload: list[tuple[Any, ...]] = []
                for entry in cache:
                    if isinstance(entry, GrowingKVCache) and entry.offset >= c_end:
                        kv_payload.append(entry.extract_chunk_payload(c_start, c_end))
                new_key = ptree.insert(parent=parent_key, tokens=c_toks, depth=d, chunk=tuple(kv_payload))
                if new_key is None:
                    start_pos = c_end
                    break
                gdn_snap: list[tuple[Any, Any, Any]] = []
                for entry in cache:
                    if isinstance(entry, DeltaSlotCache):
                        cv = mx.contiguous(entry.conv_state) if entry.conv_state is not None else None
                        if entry.rec_state is not None:
                            q_c, s_c = quantize_gdn_state_block32(entry.rec_state)
                            mx.eval(q_c, s_c)  # guard: allow-eval
                            gdn_snap.append((cv, q_c, s_c))
                        else:
                            if cv is not None:
                                mx.eval(cv)  # guard: allow-eval
                            gdn_snap.append((cv, None, None))
                if not ptree.attach_snapshot(new_key, tuple(gdn_snap), rollback_keys_on_fail=[new_key]):
                    start_pos = c_end
                    break
                parent_key = new_key
                start_pos = c_end

        return cache, start_pos

    def _prefill_tail_chunked(
        self,
        cache: list[DeltaSlotCache | GrowingKVCache],
        run_ids: Sequence[int],
        *,
        chunk_size: int = 256,
        keep_tail_hidden: int = 129,
    ) -> mx.array:
        """Prefill `run_ids` in `<= 256`-token chunks to cap `S > 2048` peak activation RAM."""
        ids = [int(t) for t in run_ids]
        if len(ids) <= chunk_size:
            h = self.model.forward_hidden(mx.array([ids], dtype=mx.int32), cache)
            return h[:, -keep_tail_hidden:, :] if int(h.shape[1]) > keep_tail_hidden else h

        tail_buf: mx.array | None = None
        for c0 in range(0, len(ids), chunk_size):
            chunk_ids = ids[c0 : c0 + chunk_size]
            ch_h = self.model.forward_hidden(mx.array([chunk_ids], dtype=mx.int32), cache)
            mx.eval(ch_h)  # guard: allow-eval
            need_keep = min(keep_tail_hidden, int(ch_h.shape[1]))
            slice_h = ch_h[:, -need_keep:, :]
            if tail_buf is None:
                tail_buf = slice_h
            else:
                tail_buf = mx.concatenate([tail_buf, slice_h], axis=1)
                if int(tail_buf.shape[1]) > keep_tail_hidden:
                    tail_buf = tail_buf[:, -keep_tail_hidden:, :]
        assert tail_buf is not None
        return tail_buf

    def _lookup_or_build_prefix(self, prefix_ids: Sequence[int]) -> HybridPrefixSnapshot:
        key = tuple(int(t) for t in prefix_ids)
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            return self._prefix_cache[key]
        use_kvi8 = self._resolve_kv_i8(1, max(256, len(key) + 64))
        c1, start_pos = self._restore_from_tree_and_advance_chunks(
            key,
            kv_alloc=max(256, len(key) + 64),
            use_kvi8=use_kvi8,
            limit=max(0, len(key) - 1),
        )
        tail = list(key[start_pos:])
        if tail:
            h = self._prefill_tail_chunked(c1, tail, keep_tail_hidden=1)
            mx.eval(h)
        snap = snapshot_hybrid_cache(key, c1, compress_gdn_int8=True)
        if self.max_prefix_cache_entries > 0:
            self._prefix_cache[key] = snap
            while len(self._prefix_cache) > self.max_prefix_cache_entries:
                self._prefix_cache.popitem(last=False)
        return snap

    def _select_next_token_with_meta(
        self,
        logits_1d: np.ndarray,
        *,
        full_hidden_1d: mx.array | None,
        matcher: Any | None,
        gram: GrammarHelper | None,
        recent_tokens: Sequence[int],
        temperature: float,
        top_p: float = 1.0,
        rng: np.random.Generator | None = None,
        repeat_penalty: bool,
        collect_telemetry: bool,
    ) -> tuple[int, dict[str, float] | None, bool]:
        work = logits_1d[: len(self._new_to_old)] if (self.model.vocab_remap is None and len(logits_1d) > len(self._new_to_old)) else logits_1d
        argmax_idx = safe_argmax_1d(work)
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

        was_masked = False
        if matcher is not None and gram is not None:
            def _lazy_full() -> np.ndarray:
                assert full_hidden_1d is not None
                fl = self.model.project_logits(full_hidden_1d[None, None, :], full_vocab=True)
                mx.eval(fl)
                return np.asarray(fl[0, 0].astype(mx.float32))

            can_fallback = bool(self.model.has_full_vocab_fallback and full_hidden_1d is not None)
            tok_id, was_masked = gram.pick_valid_token(
                matcher,
                work,
                argmax_idx,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
                fallback_to_full_vocab=can_fallback,
                full_logits_1d=_lazy_full if can_fallback else None,
            )
        else:
            tok_id = sample_unconstrained(
                work,
                self._new_to_old,
                argmax_idx,
                temperature=temperature,
                top_p=top_p,
                rng=rng,
            )

        tel: dict[str, float] | None = None
        if collect_telemetry:
            old_to_new = self.model.vocab_remap.old_to_new if self.model.vocab_remap is not None else None
            c_idx = int(old_to_new[tok_id]) if old_to_new is not None else int(tok_id)
            tel = (
                _compute_step_telemetry(work, c_idx)
                if 0 <= c_idx < len(work)
                else {"logprob": 0.0, "prob": 0.0, "entropy": 0.0}
            )
        return int(tok_id), tel, bool(was_masked)

    def _select_next_token(
        self,
        logits_1d: np.ndarray,
        *,
        full_hidden_1d: mx.array | None,
        matcher: Any | None,
        gram: GrammarHelper | None,
        recent_tokens: Sequence[int],
        temperature: float,
        top_p: float = 1.0,
        rng: np.random.Generator | None = None,
        repeat_penalty: bool,
        collect_telemetry: bool,
    ) -> tuple[int, dict[str, float] | None]:
        tok_id, tel, _ = self._select_next_token_with_meta(
            logits_1d,
            full_hidden_1d=full_hidden_1d,
            matcher=matcher,
            gram=gram,
            recent_tokens=recent_tokens,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
            repeat_penalty=repeat_penalty,
            collect_telemetry=collect_telemetry,
        )
        return tok_id, tel

    def _project_needed_rows(
        self,
        h_2d: mx.array,
        need_rows: list[int],
        row_count: int,
        *,
        eager_fp32: bool = False,
    ) -> tuple[mx.array | None, dict[int, int], dict[int, np.ndarray]]:
        """Project active rows; evaluate `[M]` GPU argmax only when `eager_fp32=False`."""
        if not need_rows:
            mx.eval(h_2d)  # guard: allow-eval
            return None, {}, {}
        if len(need_rows) == row_count:
            sl_raw = self.model.project_logits(h_2d[:, None, :])[:, 0, :]
        else:
            idx_mx = mx.array(need_rows, dtype=mx.int32)
            sub_h = mx.take(h_2d, idx_mx, axis=0)[:, None, :]
            sl_raw = self.model.project_logits(sub_h)[:, 0, :]

        v_limit = len(self._new_to_old)
        sl_eff = (
            sl_raw[:, :v_limit]
            if (self.model.vocab_remap is None and int(sl_raw.shape[-1]) > v_limit)
            else sl_raw
        )
        if eager_fp32:
            sl_f32 = sl_eff.astype(mx.float32)
            mx.eval(h_2d, sl_f32)  # guard: allow-eval
            arr = np.asarray(sl_f32)
            return None, {}, {r: arr[pos] for pos, r in enumerate(need_rows)}

        am_mx = _safe_argmax_with_nan_guard(sl_eff)
        mx.eval(h_2d, am_mx)  # guard: allow-eval
        am_list = [int(x) for x in am_mx.tolist()]
        return sl_eff, {r: am_list[pos] for pos, r in enumerate(need_rows)}, {}

    def _select_tokens_from_hidden(
        self,
        h_2d: mx.array,
        need_rows: list[int],
        row_count: int,
        *,
        matchers: Sequence[Any | None],
        grammars: Sequence[GrammarHelper | None],
        recent_tokens_by_row: Sequence[Sequence[int]],
        rngs: Sequence[np.random.Generator | None],
        temperature: float,
        top_p: float,
        repeat_penalty: bool,
        collect_telemetry: bool,
    ) -> tuple[dict[int, int], dict[int, dict[str, float] | None], int]:
        """Select next tokens via lazy GPU argmax, copying FP32 logits only for fallback rows."""
        eager_fp32 = bool(float(temperature) > 0.0 or collect_telemetry)
        sl_eff, greedy_map, logits_map = self._project_needed_rows(
            h_2d,
            need_rows,
            row_count,
            eager_fp32=eager_fp32,
        )
        if not need_rows:
            return {}, {}, 0

        picked_tid: dict[int, int] = {}
        picked_tel: dict[int, dict[str, float] | None] = {}
        mask_fallbacks = 0

        if not eager_fp32:
            n_vocab = len(self._new_to_old)
            fb_rows: list[tuple[int, int]] = []
            for pos, r in enumerate(need_rows):
                argmax_idx = greedy_map[r]
                m = matchers[r]
                if 0 <= argmax_idx < n_vocab:
                    cand_tid = int(self._new_to_old[argmax_idx])
                    is_penalized = False
                    if repeat_penalty:
                        min_hits = 5 if m is not None else 4
                        pen = compute_loop_breaker_penalties(
                            recent_tokens_by_row[r],
                            window=96,
                            min_prior_hits=min_hits,
                            base_penalty=8.0,
                            stop_token_ids=self.tokenizer.stop_token_ids,
                        )
                        is_penalized = bool(pen and cand_tid in pen)
                    if not is_penalized and (m is None or bool(m.validate_tokens([cand_tid]))):
                        picked_tid[r] = cand_tid
                        picked_tel[r] = None
                        continue
                fb_rows.append((pos, r))

            if fb_rows:
                assert sl_eff is not None
                fb_pos_mx = mx.array([pos for pos, _ in fb_rows], dtype=mx.int32)
                fb_f32 = mx.take(sl_eff, fb_pos_mx, axis=0).astype(mx.float32)
                mx.eval(fb_f32)  # guard: allow-eval
                fb_arr = np.asarray(fb_f32)
                for k_idx, (_, r) in enumerate(fb_rows):
                    logits_map[r] = fb_arr[k_idx]
            del sl_eff

        for r in need_rows:
            if r in picked_tid:
                continue
            tid, tel, was_masked = self._select_next_token_with_meta(
                logits_map[r],
                full_hidden_1d=h_2d[r],
                matcher=matchers[r],
                gram=grammars[r],
                recent_tokens=recent_tokens_by_row[r],
                temperature=temperature,
                top_p=top_p,
                rng=rngs[r],
                repeat_penalty=repeat_penalty,
                collect_telemetry=collect_telemetry,
            )
            picked_tid[r] = tid
            picked_tel[r] = tel
            if was_masked:
                mask_fallbacks += 1

        return picked_tid, picked_tel, mask_fallbacks

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
            hidden = self._prefill_tail_chunked(cache, run_ids)
            return cache, hidden, run_ids

        cache, start_pos = self._restore_from_tree_and_advance_chunks(
            prompt_ids,
            kv_alloc=kv_alloc,
            use_kvi8=use_kvi8,
            limit=max(0, len(prompt_ids) - 1),
        )
        tail_ids = list(prompt_ids[start_pos:]) + ext
        hidden = self._prefill_tail_chunked(cache, tail_ids)
        return cache, hidden, tail_ids

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
        top_p: float = 1.0,
        seed: int | None = None,
        thinking_budget: int,
        speculative: bool | None,
        collect_telemetry: bool,
        inputs_embeds: mx.array | None = None,
        rope_trig: tuple[mx.array, mx.array] | None = None,
        rope_offset_delta: int = 0,
        fallback_counter: list[int] | None = None,
    ) -> GenerationItem:
        rng = np.random.default_rng(int(seed)) if seed is not None else None
        matcher = gram.new_matcher() if gram is not None else None
        telemetry: list[dict[str, float]] | None = [] if collect_telemetry else None
        single_mask_fallbacks = 0
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

        if inputs_embeds is not None:
            est_mm_ctx = len(prompt_ids) + len(folded_ff) + max_tokens + 64
            kv_alloc = max(256, min(512, est_mm_ctx))
            use_kvi8 = self._resolve_kv_i8(1, est_mm_ctx)
            cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8)
            h_mm = self.model.forward_hidden(
                None,
                cache,
                inputs_embeds=inputs_embeds,
                rope_trig=rope_trig,
                rope_offset_delta=rope_offset_delta,
            )
            hidden = h_mm[:, -129:, :] if int(h_mm.shape[1]) > 129 else h_mm
            mx.eval(hidden)
            prefill_in = list(prompt_ids)
            if folded_ff:
                hidden = self._prefill_tail_chunked(cache, folded_ff) if rope_offset_delta == 0 else self.model.forward_hidden(
                    mx.array([folded_ff], dtype=mx.int32),
                    cache,
                    rope_offset_delta=rope_offset_delta,
                )
                prefill_in.extend(folded_ff)
        else:
            cache, hidden, prefill_in = self._prefill_single_with_caches(
                prompt_ids,
                prefix_snap=prefix_snap,
                suffix_ids=suffix_ids,
                max_tokens=max_tokens,
                extra_tail_ids=folded_ff,
            )
        last_h = hidden[:, -1:, :]
        logits: mx.array | None = None
        reasoning_text: str | None = None
        # Optional Phase 1: Two-phase budgeted thinking (`thinking_budget >= 384`)
        if thinking_budget > 0:
            logits = self.model.project_logits(last_h)
            mx.eval(last_h, logits)
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
                    top_p=top_p,
                    rng=rng,
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
                sh = self.model.forward_hidden(step_in, cache, rope_offset_delta=rope_offset_delta)
                sl = self.model.project_logits(sh)
                mx.eval(sh, sl)  # guard: allow-eval
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
            sh = self.model.forward_hidden(suf_arr, cache, rope_offset_delta=rope_offset_delta)
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
            elif hit_stop_t or matcher.is_stopped():
                return self._finalize_item(
                    prompt_tokens=len(prompt_ids),
                    answer_ids=answer_ids,
                    reasoning_text=reasoning_text,
                    hit_max_tokens=False,
                    schema_obj=schema_obj,
                    telemetry=telemetry,
                )
            elif ff0_cap:
                ff_arr = mx.array([ff0_cap], dtype=mx.int32)
                sh = self.model.forward_hidden(ff_arr, cache, rope_offset_delta=rope_offset_delta)
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
                is_moe=self.model.cfg.is_moe,
            )
            and matcher is None
            and not collect_telemetry
        )

        mtp_cache = self.mtp_drafter.make_cache() if (use_spec and self.mtp_drafter is not None) else None
        history_ids = list(prompt_ids) + list(answer_ids)

        # Sample first non-FF token from prefill logits (lazy GPU argmax when thinking_budget <= 0)
        if len(answer_ids) < max_tokens and (matcher is None or not matcher.is_stopped()):
            if logits is not None:
                cur_h_1d = last_h[0, 0]
                cur_logits_1d = np.asarray(logits[0, 0].astype(mx.float32))
                first_tok, first_tel, first_masked = self._select_next_token_with_meta(
                    cur_logits_1d,
                    full_hidden_1d=cur_h_1d,
                    matcher=matcher,
                    gram=gram,
                    recent_tokens=answer_ids,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                    repeat_penalty=True,
                    collect_telemetry=collect_telemetry,
                )
                if first_masked:
                    single_mask_fallbacks += 1
            else:
                h_2d_0 = last_h[:, 0, :]
                tids0, tels0, fb0 = self._select_tokens_from_hidden(
                    h_2d_0,
                    [0],
                    1,
                    matchers=[matcher],
                    grammars=[gram],
                    recent_tokens_by_row=[answer_ids],
                    rngs=[rng],
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=True,
                    collect_telemetry=collect_telemetry,
                )
                single_mask_fallbacks += fb0
                cur_h_1d = h_2d_0[0]
                first_tok, first_tel = tids0[0], tels0[0]
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
            if logits is None:
                mx.eval(last_h)
            cur_h_1d = last_h[0, 0]
            pending_tokens = []

        if (
            mtp_cache is not None
            and self.mtp_drafter is not None
            and thinking_budget <= 0
            and len(prefill_in) >= 2
            and int(hidden.shape[1]) >= 2
        ):
            # Seed 1-layer MTP cache over prompt tail (capped to 128 tokens and bounded by chunked hidden buffer)
            tail_n = min(len(prefill_in) - 1, int(hidden.shape[1]) - 1, 128)
            seed_next = mx.array([prefill_in[-tail_n:]], dtype=mx.int32)
            seed_prev_h = hidden[:, -(tail_n + 1) : -1, :]
            _ = self.mtp_drafter.forward_hidden(seed_next, seed_prev_h, self.model, mtp_cache)

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
                    pen_near = compute_loop_breaker_penalties(
                        answer_ids,
                        window=96,
                        min_prior_hits=3,
                        base_penalty=8.0,
                        stop_token_ids=self.tokenizer.stop_token_ids,
                    )
                    if not pen_near:
                        d_tok_mx = mx.array([[last_tok]], dtype=mx.int32)
                        d_prev_h = cur_h_1d[None, None, :]
                        step_arrs: list[tuple[mx.array, mx.array]] = []
                        for _ in range(cap_k):
                            dh = self.mtp_drafter.forward_hidden(
                                d_tok_mx,
                                d_prev_h,
                                self.model,
                                mtp_cache,
                            )
                            dl = self.model.project_logits(dh)
                            idx_m = mx.argmax(dl[0, 0], axis=-1)
                            has_nan_m = mx.any(mx.isnan(dl[0, 0]))
                            nxt_m = (self._new_to_old_mx[idx_m] if self._new_to_old_mx is not None else idx_m).astype(mx.int32)
                            step_arrs.append((nxt_m, has_nan_m))
                            d_tok_mx = nxt_m.reshape(1, 1)
                            d_prev_h = dh
                        mx.eval(*[arr for pair in step_arrs for arr in pair])  # guard: allow-eval
                        for nxt_m, has_nan_m in step_arrs:
                            if bool(np.asarray(has_nan_m).item()):
                                break
                            d_next = int(np.asarray(nxt_m).item())
                            drafts.append(d_next)
                            if d_next in self.tokenizer.stop_token_ids:
                                break
                        if not drafts:
                            mtp_cache.rewind_to(mtp_prev_off)
                    else:
                        d_tok = last_tok
                        d_prev_h = cur_h_1d[None, None, :]
                        old_to_new = self.model.vocab_remap.old_to_new if self.model.vocab_remap is not None else None
                        for _ in range(cap_k):
                            dh = self.mtp_drafter.forward_hidden(
                                mx.array([[d_tok]], dtype=mx.int32),
                                d_prev_h,
                                self.model,
                                mtp_cache,
                            )
                            dl = self.model.project_logits(dh)
                            mx.eval(dh, dl)  # guard: allow-eval
                            dl_np = np.asarray(dl[0, 0].astype(mx.float32))
                            d_pen = compute_loop_breaker_penalties(
                                answer_ids + drafts,
                                window=96,
                                min_prior_hits=4,
                                base_penalty=8.0,
                                stop_token_ids=self.tokenizer.stop_token_ids,
                            )
                            if d_pen:
                                dl_np, d_idx = apply_loop_breaker_penalties_1d(dl_np, d_pen, old_to_new)
                            else:
                                d_idx = safe_argmax_1d(dl_np)
                            if d_idx < 0:
                                break
                            d_next = int(self._new_to_old[d_idx])
                            drafts.append(d_next)
                            if d_next in self.tokenizer.stop_token_ids:
                                break
                            d_tok = d_next
                            d_prev_h = dh
                        if not drafts:
                            mtp_cache.rewind_to(mtp_prev_off)
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
                        rope_offset_delta=rope_offset_delta,
                    )

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

                    if mtp_cache is not None and self.mtp_drafter is not None:
                        # Re-sync MTP cache cleanly with exact target hidden states `acc_h` (fixes offset/bonus drift).
                        mtp_cache.rewind_to(min(mtp_cache.offset, mtp_prev_off + 1))
                        if n_drafts_matched > 0 and not stopped:
                            sync_toks = mx.array([drafts[:n_drafts_matched]], dtype=mx.int32)
                            _ = self.mtp_drafter.forward_hidden(
                                sync_toks,
                                acc_h[:, :n_drafts_matched, :],
                                self.model,
                                mtp_cache,
                            )

                    if stopped or last_emitted is None:
                        pending_tokens = []
                        break
                    pending_tokens = [last_emitted]
                    continue

            # Standard 1-step (or multi-FF-token) advance
            step_arr = mx.array([pending_tokens], dtype=mx.int32)
            sh = self.model.forward_hidden(step_arr, cache, rope_offset_delta=rope_offset_delta)
            last_h = sh[:, -1:, :]
            h_2d = last_h[:, 0, :]
            tids_step, tels_step, step_fb = self._select_tokens_from_hidden(
                h_2d,
                [0],
                1,
                matchers=[matcher],
                grammars=[gram],
                recent_tokens_by_row=[answer_ids],
                rngs=[rng],
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=True,
                collect_telemetry=collect_telemetry,
            )
            single_mask_fallbacks += step_fb
            cur_h_1d = h_2d[0]
            nxt_tok, nxt_tel = tids_step[0], tels_step[0]
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

        if fallback_counter is not None:
            fallback_counter[0] += single_mask_fallbacks

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
        top_p: float = 1.0,
        slot_seeds: Sequence[int | None] | None = None,
        collect_telemetry: bool,
        compact_active: bool = True,
        fallback_counter: list[int] | None = None,
    ) -> list[GenerationItem]:
        """Right-padded batched prefill + multi-token FF + active-row `lm_head` decode wave (`B >= 2`, zero pad leakage)."""
        b_sz = len(wave_prompt_ids)
        rngs = [
            np.random.default_rng(int(slot_seeds[i])) if (slot_seeds is not None and slot_seeds[i] is not None) else None
            for i in range(b_sz)
        ]
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
        min_suf = min(suf_lens)
        max_suf = max(suf_lens)
        any_ragged = any(sl < max_suf for sl in suf_lens)
        padded_suf = [s + [pad_id] * (max_suf - len(s)) for s in aug_sufs]

        lcp_len = len(prefix_snap.prefix_tokens) if prefix_snap is not None else 0
        est_ctx = lcp_len + max_suf + max_tokens + 64
        kv_alloc = max(256, min(512, est_ctx))
        use_kvi8 = self._resolve_kv_i8(b_sz, est_ctx)
        if prefix_snap is not None:
            cache = restore_hybrid_cache(prefix_snap, batch_size=b_sz, kv_step=kv_alloc, kv_i8=use_kvi8)
        else:
            cache = self.model.make_cache(kv_step=kv_alloc, kv_i8=use_kvi8, batch_size=b_sz)

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

        if max_suf <= 512:
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
        else:
            # Chunked 512-token wave prefill bounds `S > 2048` peak intermediate RAM while capturing exact last-token `h[i]`.
            slot_cur_h: list[mx.array | None] = [None] * b_sz
            for c0 in range(0, max_suf, 512):
                c1 = min(max_suf, c0 + 512)
                inp_c = mx.array([row[c0:c1] for row in padded_suf], dtype=mx.int32)
                chunk_has_pad = any_ragged and (c1 > min_suf)
                v_scale_c = valid_scale[:, c0:c1, :] if (chunk_has_pad and valid_scale is not None) else None
                v_lens_c = (
                    mx.array([max(0, min(sl - c0, c1 - c0)) for sl in suf_lens], dtype=mx.int32)
                    if chunk_has_pad
                    else None
                )
                ch_h = self.model.forward_hidden(
                    inp_c,
                    cache,
                    valid_token_scale=v_scale_c,
                    valid_lengths=v_lens_c,
                    attn_mask="causal",
                    rope_trig=None,
                )
                mx.eval(ch_h)  # guard: allow-eval
                for i, sl in enumerate(suf_lens):
                    pos_i = max(0, sl - 1)
                    if c0 <= pos_i < c1:
                        slot_cur_h[i] = ch_h[i, pos_i - c0, :]
            cur_h = mx.stack([slot_cur_h[i] for i in range(b_sz)], axis=0)  # type: ignore[misc]

        true_pos = [lcp_len + sl for sl in suf_lens]
        kv_cursor = lcp_len + max_suf
        live_rows = list(range(b_sz))
        any_ragged_live = any_ragged
        suf_end_arr = (
            mx.array([lcp_len + suf_lens[w_i] for w_i in live_rows], dtype=mx.int32)[:, None, None, None]
            if any_ragged_live
            else None
        )
        pending_q: list[list[int]] = [[] for _ in range(b_sz)]
        wave_mask_fallbacks = 0

        for _step in range(max_tokens + 1):
            cur_b = len(live_rows)
            need_rows: list[int] = []
            for r in range(cur_b):
                w_i = live_rows[r]
                if done[w_i] or pending_q[w_i]:
                    continue
                m = matchers[w_i]
                if len(ans_ids[w_i]) >= max_tokens:
                    done[w_i] = True
                    hit_max[w_i] = not (m is not None and m.is_stopped() and m.is_accepting())
                    continue
                if m is not None and m.is_stopped():
                    done[w_i] = True
                    continue
                need_rows.append(r)

            picked_tid, picked_tel, step_fb = self._select_tokens_from_hidden(
                cur_h,
                need_rows,
                cur_b,
                matchers=[matchers[w_i] for w_i in live_rows],
                grammars=[grammars[w_i] for w_i in live_rows],
                recent_tokens_by_row=[ans_ids[w_i] for w_i in live_rows],
                rngs=[rngs[w_i] for w_i in live_rows],
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=True,
                collect_telemetry=collect_telemetry,
            )
            wave_mask_fallbacks += step_fb

            for r in need_rows:
                w_i = live_rows[r]
                m = matchers[w_i]
                tid = picked_tid[r]
                tel = picked_tel[r]
                if tel is not None and tels[w_i] is not None:
                    tels[w_i].append(tel)  # type: ignore[union-attr]
                if tid in self.tokenizer.stop_token_ids:
                    done[w_i] = True
                    continue
                ans_ids[w_i].append(tid)

                if m is not None:
                    m.consume_token(tid)
                    if m.is_stopped():
                        done[w_i] = True
                        continue
                    if len(ans_ids[w_i]) >= max_tokens:
                        done[w_i] = True
                        hit_max[w_i] = True
                        continue
                    ff_clean: list[int] = []
                    stopped_by_ff = False
                    for ft in m.compute_ff_tokens():
                        fti = int(ft)
                        if fti in self.tokenizer.stop_token_ids:
                            stopped_by_ff = True
                            break
                        if len(ans_ids[w_i]) + len(ff_clean) >= max_tokens:
                            break
                        ff_clean.append(fti)
                    if ff_clean:
                        m.consume_tokens(ff_clean)
                        ans_ids[w_i].extend(ff_clean)
                    if stopped_by_ff or m.is_stopped():
                        done[w_i] = True
                        continue
                    if len(ans_ids[w_i]) >= max_tokens:
                        done[w_i] = True
                        hit_max[w_i] = True
                        continue
                    pending_q[w_i] = [tid] + ff_clean
                else:
                    if len(ans_ids[w_i]) >= max_tokens:
                        done[w_i] = True
                        hit_max[w_i] = True
                        continue
                    pending_q[w_i] = [tid]

            alive_r = [r for r, w_i in enumerate(live_rows) if not done[w_i]]
            if not alive_r:
                break

            if compact_active and cur_b >= 4 and len(alive_r) < cur_b:
                if len(alive_r) <= (cur_b * 5) // 8 or (cur_b - len(alive_r)) >= 4:
                    compact_hybrid_cache(cache, alive_r)
                    live_rows = [live_rows[r] for r in alive_r]
                    cur_b = len(live_rows)
                    alive_r = list(range(cur_b))
                    any_ragged_live = any(suf_lens[w_i] < max_suf for w_i in live_rows)
                    suf_end_arr = (
                        mx.array([lcp_len + suf_lens[w_i] for w_i in live_rows], dtype=mx.int32)[:, None, None, None]
                        if any_ragged_live
                        else None
                    )

            # Advance `step_w >= 1` common queued tokens across all alive slots in a single forward pass!
            step_w = min(len(pending_q[live_rows[r]]) for r in alive_r)
            step_rows: list[list[int]] = []
            for r in range(cur_b):
                w_i = live_rows[r]
                if done[w_i]:
                    step_rows.append([pad_id] * step_w)
                else:
                    step_rows.append(pending_q[w_i][:step_w])
                    del pending_q[w_i][:step_w]

            step_inp = mx.array(step_rows, dtype=mx.int32)
            kv_prev = kv_cursor
            kv_now = kv_prev + step_w
            kv_cursor = kv_now

            if any_ragged_live:
                assert suf_end_arr is not None
                p_mat = np.array([true_pos[w_i] for w_i in live_rows], dtype=np.int32)[:, None] + np.arange(step_w, dtype=np.int32)[None, :]
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
                mx.array([[[0.0] if done[live_rows[r]] else [1.0]] * step_w for r in range(cur_b)], dtype=mx.bfloat16)
                if len(alive_r) < cur_b
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
            for w_i in live_rows:
                if not done[w_i]:
                    true_pos[w_i] += step_w

        if fallback_counter is not None:
            fallback_counter[0] += wave_mask_fallbacks

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
        images: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        json_schema: str | Mapping[str, Any] | None = None,
        json_schemas: Sequence[str | Mapping[str, Any] | None] | None = None,
        per_doc_grammars: Sequence[str | Mapping[str, Any] | GrammarHelper | None] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        batch_size: int | None = None,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool | None = None,
        enable_thinking: bool | None = None,
        thinking_budget_tokens: int | None = None,
        allow_short_thinking_budget: bool = False,
        reasoning_prefix_field: str | None = None,
        collect_telemetry: bool = False,
        compact_active: bool = True,
        max_pad_ratio: float | None = 1.8,
        max_wave_tokens: int | None = None,
        **_ignored_kwargs: Any,
    ) -> GenerationBatchResult:
        """Generate structured or free-text completions over a batch of prompts with LCP hybrid caching."""
        if not prompts:
            return GenerationBatchResult(
                items=[],
                stats={"docs": 0, "prompt_tokens": 0, "completion_tokens": 0, "mask_fallbacks": 0, "elapsed_sec": 0.0},
            )

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

        norm_images: list[list[Any]] | None = None
        if images is not None:
            if len(images) != n_docs:
                raise ValueError(f"images len {len(images)} != prompts len {n_docs}")

            def _norm_item_images(raw: Any) -> list[Any]:
                if raw is None:
                    return []
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, np.ndarray)):
                    return list(raw)
                return [raw]

            norm_images = [_norm_item_images(x) for x in images]

        extra_stats: dict[str, Any] = {}
        with self._lock:
            base_wave_bs = int(batch_size) if batch_size is not None else min(n_docs, self.default_max_b_slots)
            eff_wave_bs = self._memory_governor.adapt(
                max(1, base_wave_bs),
                base_cache_limit_mb=self._mlx_cache_limit_mb,
                apply_mlx=True,
            ).b_slots
            results: list[GenerationItem | None] = [None] * n_docs
            max_lcp_seen = 0
            fb_counter = [0]

            if norm_images is not None:
                if n_docs == 1 or eff_wave_bs <= 1 or use_two_phase:
                    for idx in range(n_docs):
                        seed_i = (int(seed) + idx) if seed is not None else None
                        if norm_images[idx]:
                            p_ids, inp_emb, r_trig, r_delta = self._prepare_multimodal_prefill(
                                prompts[idx],
                                norm_images[idx],
                                system_prompt=system_prompt,
                                enable_thinking=use_two_phase,
                            )
                            results[idx] = self._decode_single(
                                p_ids,
                                prefix_snap=None,
                                suffix_ids=[],
                                gram=grammars[idx],
                                schema_obj=schema_objs[idx],
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                                top_p=float(top_p),
                                seed=seed_i,
                                thinking_budget=eff_budget,
                                speculative=speculative,
                                collect_telemetry=collect_telemetry,
                                inputs_embeds=inp_emb,
                                rope_trig=r_trig,
                                rope_offset_delta=r_delta,
                                fallback_counter=fb_counter,
                            )
                        else:
                            enc_i = self.tokenizer.encode_chat(
                                prompts[idx],
                                system_prompt=system_prompt,
                                enable_thinking=use_two_phase,
                            )
                            results[idx] = self._decode_single(
                                enc_i,
                                prefix_snap=None,
                                suffix_ids=enc_i,
                                gram=grammars[idx],
                                schema_obj=schema_objs[idx],
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                                top_p=float(top_p),
                                seed=seed_i,
                                thinking_budget=eff_budget,
                                speculative=speculative,
                                collect_telemetry=collect_telemetry,
                                fallback_counter=fb_counter,
                            )
                else:
                    has_any_images = any(bool(xs) for xs in norm_images)
                    vis = self._ensure_vision_model() if has_any_images else None
                    img_pad_id, vis_start_id, vis_end_id, user_hdr_ids, nl_ids = self._mm_special_ids()
                    if img_pad_id is None:
                        raise ValueError("Tokenizer is missing `<|image_pad|>` token required for multimodal prompts")

                    def _mm_prompt_builder(item_idx: int, vis_counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
                        return self._expand_multimodal_prompt_ids(
                            prompts[item_idx],
                            vis_counts,
                            system_prompt=system_prompt,
                            enable_thinking=False,
                            img_pad_id=int(img_pad_id),
                            vis_start_id=vis_start_id,
                            vis_end_id=vis_end_id,
                            user_hdr_ids=user_hdr_ids,
                            nl_ids=nl_ids,
                        )

                    old_to_new = self.model.vocab_remap.old_to_new if self.model.vocab_remap is not None else None
                    mm_items, extra_stats = batched_multimodal_generate(
                        lm=self.model,
                        vision_model=vis,
                        tokenizer=self.tokenizer,
                        images_per_item=norm_images,
                        prompt_builder=_mm_prompt_builder,
                        image_token_id=int(img_pad_id),
                        new_to_old=self._new_to_old,
                        old_to_new=old_to_new,
                        grammars=grammars,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        slot_seeds=[(int(seed) + i) if seed is not None else None for i in range(n_docs)],
                        repeat_penalty=True,
                        collect_telemetry=collect_telemetry,
                        batch_size=eff_wave_bs,
                        kv_step=512,
                        kv_i8_resolver=self._resolve_kv_i8,
                        compact_active=compact_active,
                        max_pad_ratio=max_pad_ratio,
                        max_wave_tokens=max_wave_tokens,
                        stop_ids=self.tokenizer.stop_token_ids,
                        pad_token_id=self.tokenizer.eos_token_id,
                        prefill_chunk_size=512,
                    )
                    for idx, mm_res in enumerate(mm_items):
                        results[idx] = self._finalize_item(
                            prompt_tokens=mm_res.prompt_tokens,
                            answer_ids=mm_res.generated_ids,
                            reasoning_text=None,
                            hit_max_tokens=mm_res.hit_max_tokens,
                            schema_obj=schema_objs[idx],
                            telemetry=mm_res.telemetry,
                        )
            else:
                encoded = [
                    self.tokenizer.encode_chat(
                        p,
                        system_prompt=system_prompt,
                        enable_thinking=use_two_phase,
                    )
                    for p in prompts
                ]

                sys_prefix_ids: list[int] = (
                    self.tokenizer.encode(
                        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n",
                        add_special_tokens=False,
                    )
                    if system_prompt and all(isinstance(p, str) for p in prompts)
                    else []
                )

                if n_docs == 1 or eff_wave_bs <= 1 or use_two_phase:
                    b1_snap: HybridPrefixSnapshot | None = None
                    if (
                        (self.max_prefix_cache_entries > 0 or self.prefix_tree is not None)
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
                            top_p=float(top_p),
                            seed=(int(seed) + idx if seed is not None else None),
                            thinking_budget=eff_budget,
                            speculative=speculative,
                            collect_telemetry=collect_telemetry,
                            fallback_counter=fb_counter,
                        )
                else:
                    # Bucket by prompt length (`128`-tok buckets) first to bound ragged pad waste across mixed-schema batches,
                    # then group identical GrammarHelper instances inside each length bucket so FF steps align cleanly.
                    order = sorted(
                        range(n_docs),
                        key=lambda i: (len(encoded[i]) // 128, 0 if grammars[i] is None else id(grammars[i]), len(encoded[i])),
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
                        apply_pad_ratio = any(
                            len(encoded[i]) > 512 for i in order[cursor : cursor + max(1, cur_bs)]
                        )
                        packed = pack_length_waves(
                            order[cursor:],
                            lambda idx: len(encoded[idx]),
                            max(1, cur_bs),
                            max_pad_ratio=(max_pad_ratio if apply_pad_ratio else None),
                            max_wave_tokens=max_wave_tokens,
                        )
                        chunk_idx = packed[0] if packed else order[cursor : cursor + 1]
                        cursor += len(chunk_idx)
                        if len(chunk_idx) == 1:
                            i0 = chunk_idx[0]
                            b1_s = (
                                self._lookup_or_build_prefix(sys_prefix_ids)
                                if (
                                    (self.max_prefix_cache_entries > 0 or self.prefix_tree is not None)
                                    and len(sys_prefix_ids) >= 32
                                    and len(encoded[i0]) > len(sys_prefix_ids)
                                    and encoded[i0][: len(sys_prefix_ids)] == sys_prefix_ids
                                )
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
                                top_p=float(top_p),
                                seed=(int(seed) + i0 if seed is not None else None),
                                thinking_budget=0,
                                speculative=speculative,
                                collect_telemetry=collect_telemetry,
                                fallback_counter=fb_counter,
                            )
                        else:
                            wave_encs = [encoded[i] for i in chunk_idx]
                            w_lcp, w_sufs = split_lcp_ids(wave_encs, min_lcp=32)
                            w_snap = (
                                self._lookup_or_build_prefix(w_lcp)
                                if (w_lcp and (self.max_prefix_cache_entries > 0 or self.prefix_tree is not None))
                                else None
                            )
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
                                top_p=float(top_p),
                                slot_seeds=[(int(seed) + i) if seed is not None else None for i in chunk_idx],
                                collect_telemetry=collect_telemetry,
                                compact_active=compact_active,
                                fallback_counter=fb_counter,
                            )
                            for i, it in zip(chunk_idx, wave_items):
                                results[i] = it
                        if cursor < n_docs and hasattr(mx, "clear_cache"):
                            mx.clear_cache()

        elapsed = time.perf_counter() - t0
        final_items = [r for r in results if r is not None]
        total_prompt = sum(it.prompt_tokens for it in final_items)
        total_comp = sum(it.completion_tokens for it in final_items)
        total_mask_fallbacks = fb_counter[0] + int(extra_stats.get("mask_fallbacks", 0) or 0)
        stats: dict[str, Any] = {
            "docs": n_docs,
            "lcp_tokens": max_lcp_seen,
            "prefix_tree_reused_tokens": int(self.prefix_tree.reused_tokens) if self.prefix_tree is not None else 0,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "mask_fallbacks": total_mask_fallbacks,
            "elapsed_sec": round(elapsed, 4),
            "docs_per_sec": round(n_docs / max(elapsed, 1e-9), 3),
            "decode_tok_s": round(total_comp / max(elapsed, 1e-9), 2),
            "mode": mode or self.default_mode,
        }
        if extra_stats:
            stats.update(extra_stats)
        return GenerationBatchResult(items=final_items, stats=stats)

    def _ensure_vision_model(self) -> Any:
        if self._vision_model is None:
            if self.model_dir is None:
                raise ValueError(
                    "Multimodal image inputs require a model checkpoint directory (`model_dir`) with `vision_config`"
                )
            cfg_p = Path(self.model_dir) / "config.json"
            if not cfg_p.is_file():
                raise ValueError(f"Checkpoint config.json missing in {self.model_dir}")
            raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            if "vision_config" not in raw_cfg:
                raise ValueError(
                    f"Checkpoint {self.model_dir} is text-only (missing 'vision_config' in config.json)"
                )
            from fast_shared.qwen35_vision import FastQwen35VisionModel

            self._vision_model = FastQwen35VisionModel.from_pretrained(self.model_dir, dtype=mx.bfloat16)
        return self._vision_model

    def _mm_special_ids(self) -> tuple[int | None, int | None, int | None, list[int], list[int]]:
        bt = self.tokenizer.backend_tokenizer
        return (
            bt.token_to_id("<|image_pad|>"),
            bt.token_to_id("<|vision_start|>"),
            bt.token_to_id("<|vision_end|>"),
            self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False),
            self.tokenizer.encode("\n", add_special_tokens=False),
        )

    def _expand_multimodal_prompt_ids(
        self,
        prompt: PromptInput,
        vis_token_counts: Sequence[int],
        *,
        system_prompt: str | None,
        enable_thinking: bool,
        img_pad_id: int,
        vis_start_id: int | None,
        vis_end_id: int | None,
        user_hdr_ids: list[int],
        nl_ids: list[int],
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Sanitize ChatML prompt markers and expand image placeholders into token IDs + visual spans."""
        placeholder = "\x00__FAST_QWEN3_IMG_PAD__\x00"
        has_explicit_markers = False
        if isinstance(prompt, str):
            has_explicit_markers = "<|image_pad|>" in prompt
            san_prompt: PromptInput = (
                prompt.replace("<|vision_start|>", "")
                .replace("<|vision_end|>", "")
                .replace("<|image_pad|>", placeholder)
                if has_explicit_markers
                else prompt
            )
        else:
            msgs_out: list[dict[str, Any]] = []
            for m in prompt:
                md = dict(m)
                raw_c = md.get("content", "")
                if isinstance(raw_c, list):
                    norm_parts: list[dict[str, Any]] = []
                    for part in raw_c:
                        if isinstance(part, Mapping) and part.get("type") == "image":
                            has_explicit_markers = True
                            norm_parts.append({"type": "text", "text": placeholder})
                        elif isinstance(part, Mapping):
                            txt = str(part.get("text", ""))
                            if "<|image_pad|>" in txt:
                                has_explicit_markers = True
                                txt = (
                                    txt.replace("<|vision_start|>", "")
                                    .replace("<|vision_end|>", "")
                                    .replace("<|image_pad|>", placeholder)
                                )
                            norm_parts.append({"type": "text", "text": txt})
                    md["content"] = norm_parts
                else:
                    s_c = str(raw_c)
                    if "<|image_pad|>" in s_c:
                        has_explicit_markers = True
                        md["content"] = (
                            s_c.replace("<|vision_start|>", "")
                            .replace("<|vision_end|>", "")
                            .replace("<|image_pad|>", placeholder)
                        )
                msgs_out.append(md)
            san_prompt = msgs_out

        rendered = self.tokenizer.format_chat(
            san_prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            sanitize=True,
        )
        expanded_ids: list[int] = []
        vis_spans: list[tuple[int, int]] = []
        img_cursor = 0

        if has_explicit_markers:
            for seg_idx, seg in enumerate(rendered.split(placeholder)):
                if seg_idx > 0 and img_cursor < len(vis_token_counts):
                    if vis_start_id is not None:
                        expanded_ids.append(int(vis_start_id))
                    n_vis_i = int(vis_token_counts[img_cursor])
                    s0 = len(expanded_ids)
                    expanded_ids.extend([int(img_pad_id)] * n_vis_i)
                    vis_spans.append((s0, s0 + n_vis_i))
                    if vis_end_id is not None:
                        expanded_ids.append(int(vis_end_id))
                    img_cursor += 1
                if seg:
                    expanded_ids.extend(self.tokenizer.encode(seg, add_special_tokens=False))
        else:
            base_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            insert_at = 0
            hdr_len = len(user_hdr_ids)
            for idx in range(len(base_ids) - hdr_len, -1, -1):
                if base_ids[idx : idx + hdr_len] == user_hdr_ids:
                    insert_at = idx + hdr_len
                    break
            expanded_ids.extend(base_ids[:insert_at])
            for n_vis_raw in vis_token_counts:
                n_vis_i = int(n_vis_raw)
                if vis_start_id is not None:
                    expanded_ids.append(int(vis_start_id))
                s0 = len(expanded_ids)
                expanded_ids.extend([int(img_pad_id)] * n_vis_i)
                vis_spans.append((s0, s0 + n_vis_i))
                if vis_end_id is not None:
                    expanded_ids.append(int(vis_end_id))
                expanded_ids.extend(nl_ids)
                img_cursor += 1
            expanded_ids.extend(base_ids[insert_at:])

        if img_cursor != len(vis_token_counts):
            raise ValueError(
                f"Expected {len(vis_token_counts)} `<|image_pad|>` marker(s) in prompt, expanded {img_cursor}"
            )
        return expanded_ids, vis_spans

    def _prepare_multimodal_prefill(
        self,
        prompt: PromptInput,
        images: Sequence[Any],
        *,
        system_prompt: str | None,
        enable_thinking: bool,
    ) -> tuple[list[int], mx.array, tuple[mx.array, mx.array], int]:
        from fast_shared.qwen35_vision import build_qwen_vision_inputs, compute_mrope_positions, precompute_mrope_trig

        vis = self._ensure_vision_model()
        img_pad_id, vis_start_id, vis_end_id, user_hdr_ids, nl_ids = self._mm_special_ids()
        if img_pad_id is None:
            raise ValueError("Tokenizer is missing `<|image_pad|>` token required for multimodal prompts")

        vinputs = [
            build_qwen_vision_inputs(
                im,
                patch_size=int(vis.cfg.patch_size),
                temporal_patch_size=int(vis.cfg.temporal_patch_size),
                merge_size=int(vis.cfg.spatial_merge_size),
            )
            for im in images
        ]
        px_cat = mx.array(np.concatenate([vi.pixel_values for vi in vinputs], axis=0))
        grids = [vi.grid_thw for vi in vinputs]
        vis_feats = vis(px_cat, grids)

        expanded_ids, vis_spans = self._expand_multimodal_prompt_ids(
            prompt,
            [vi.num_visual_tokens for vi in vinputs],
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            img_pad_id=int(img_pad_id),
            vis_start_id=vis_start_id,
            vis_end_id=vis_end_id,
            user_hdr_ids=user_hdr_ids,
            nl_ids=nl_ids,
        )

        base_emb = self.model.embed_tokens(mx.array([expanded_ids], dtype=mx.int32))
        parts: list[mx.array] = []
        vf_cursor = 0
        tok_cursor = 0
        for s0, s1 in vis_spans:
            if s0 > tok_cursor:
                parts.append(base_emb[:, tok_cursor:s0, :])
            slen = s1 - s0
            parts.append(vis_feats[None, vf_cursor : vf_cursor + slen, :].astype(base_emb.dtype))
            vf_cursor += slen
            tok_cursor = s1
        if tok_cursor < len(expanded_ids):
            parts.append(base_emb[:, tok_cursor:, :])
        inputs_embeds = mx.concatenate(parts, axis=1)

        pos_3d, rope_delta = compute_mrope_positions(
            expanded_ids,
            image_grid_thw=grids,
            image_token_id=int(img_pad_id),
            spatial_merge_size=int(vis.cfg.spatial_merge_size),
        )
        rope_cos, rope_sin = precompute_mrope_trig(
            pos_3d,
            rot_dims=int(self.model.cfg.rot_dims),
            rope_theta=float(self.model.cfg.rope_theta),
            mrope_section=getattr(self.model.cfg, "mrope_section", (11, 11, 10)),
            dtype=inputs_embeds.dtype,
        )
        return expanded_ids, inputs_embeds, (rope_cos, rope_sin), rope_delta

    def generate(
        self,
        prompt: PromptInput,
        *,
        images: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        json_schema: str | Mapping[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
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
        if images:
            eff_think = self.default_enable_thinking if enable_thinking is None else bool(enable_thinking)
            raw_budget = (
                self.default_thinking_budget if thinking_budget_tokens is None else int(thinking_budget_tokens)
            )
            eff_budget = validate_thinking_budget(
                raw_budget if raw_budget > 0 else (None if eff_think else 0),
                enable_thinking=eff_think,
                allow_short_thinking_budget=allow_short_thinking_budget,
            )
            gram: GrammarHelper | None = None
            schema_obj: dict[str, Any] | None = None
            if json_schema is not None:
                eff_sch = (
                    inject_reasoning_prefix_property(json_schema, field_name=reasoning_prefix_field)
                    if reasoning_prefix_field
                    else json_schema
                )
                gram = self._get_grammar_helper(eff_sch, whitespace_flexible)
                schema_obj = gram.schema_obj
            prompt_ids, inputs_embeds, rope_trig, rope_delta = self._prepare_multimodal_prefill(
                prompt,
                list(images),
                system_prompt=system_prompt,
                enable_thinking=(eff_budget > 0),
            )
            with self._lock:
                return self._decode_single(
                    prompt_ids,
                    prefix_snap=None,
                    suffix_ids=[],
                    gram=gram,
                    schema_obj=schema_obj,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    seed=seed,
                    thinking_budget=eff_budget,
                    speculative=speculative,
                    collect_telemetry=collect_telemetry,
                    inputs_embeds=inputs_embeds,
                    rope_trig=rope_trig,
                    rope_offset_delta=rope_delta,
                )

        batch_res = self.generate_batch(
            [prompt],
            system_prompt=system_prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
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
