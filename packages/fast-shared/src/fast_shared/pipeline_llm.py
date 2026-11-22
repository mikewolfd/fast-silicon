"""Streaming and resumable Parquet + JSONL structured extraction pipeline for generative engines."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
import copy
from dataclasses import dataclass, field, is_dataclass, replace
import inspect
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any
from weakref import WeakKeyDictionary

import pyarrow as pa
import pyarrow.parquet as pq

from .bench import clear_mlx_arena, wired_limit
from .embeddings import set_mlx_cache_limit
from .extractconf import ExtractConfCalibrator, compute_extractconf_risk, parse_json_schema_dict
from .parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
)
from .runtime_lessons import (
    DEFAULT_MLX_CACHE_LIMIT_MB,
    WIRED_MEMORY_CEILING_FRACTION,
    DynamicSiliconPolicy,
    LiveMemoryGovernor,
    nocache_open,
)


@dataclass
class ParquetExtractSummary:
    input_files: list[Path]
    output_path: Path
    docs_processed: int
    valid_json: int
    valid_schema: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float
    docs_per_sec: float
    decode_tok_s: float
    retries_attempted: int = 0
    retries_succeeded: int = 0


@dataclass(frozen=True)
class ExtractRetryRequest:
    """Deferred single-model extraction retry payload."""

    row_index: int
    prompt: str
    system_prompt: str
    json_schema: str | dict[str, Any] | None
    json_schema_dict: Mapping[str, Any] | None
    attempt: int
    max_tokens: int
    finish_reason: str
    valid_json: bool
    valid_schema: bool
    extractconf_risk: float
    response_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def slice_first_pages_utf8(text: str, page_offsets: list[int] | None, max_pages: int) -> str:
    """Slice `text` cleanly to its first `max_pages` using UTF-8 `page_offsets`."""
    if max_pages <= 0 or not page_offsets or len(page_offsets) <= max_pages:
        return text
    raw = text.encode("utf-8")
    end_b = int(page_offsets[max_pages])
    if 0 < end_b < len(raw):
        return raw[:end_b].decode("utf-8", errors="ignore")
    return text


def next_shard_index(out_dir: Path) -> int:
    max_idx = -1
    for p in out_dir.glob("part-*.parquet"):
        stem = p.stem
        if stem.startswith("part-"):
            try:
                max_idx = max(max_idx, int(stem.split("-")[1]))
            except ValueError:
                pass
    return max_idx + 1


def resolve_llm_text_column(schema: pa.Schema, text_col: str | None) -> str:
    try:
        return resolve_text_column(schema, text_col)
    except KeyError:
        if text_col is None and "body_text" in set(schema.names):
            return "body_text"
        raise


def _iter_engine_provenance_dicts(engine: Any) -> Iterator[dict[str, Any]]:
    candidates: list[Any] = [
        getattr(engine, "model_dir", None),
        getattr(engine, "ckpt_dir", None),
        getattr(engine, "checkpoint_dir", None),
        getattr(engine, "path", None),
    ]
    model_obj = getattr(engine, "model", None)
    if model_obj is not None:
        candidates.extend(
            [
                getattr(model_obj, "model_dir", None),
                getattr(model_obj, "ckpt_dir", None),
                getattr(model_obj, "checkpoint_dir", None),
                getattr(model_obj, "path", None),
            ]
        )
    paths_obj = getattr(engine, "paths", None)
    if paths_obj is not None:
        candidates.append(getattr(paths_obj, "model_dir", None))

    seen: set[Path] = set()
    for cand in candidates:
        if not cand or not isinstance(cand, (str, Path)):
            continue
        cpath = Path(cand)
        prov_path = cpath if cpath.name == "provenance.json" else (cpath / "provenance.json")
        if prov_path in seen or not prov_path.is_file():
            continue
        seen.add(prov_path)
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(prov, dict):
            yield prov


def resolve_engine_whitespace_flexible(engine: Any, whitespace_flexible: bool | None) -> bool | None:
    """Resolve `whitespace_flexible`: `None` selects `': '` unless checkpoint `provenance.json` declares compact wire."""
    if whitespace_flexible is not None:
        return bool(whitespace_flexible)

    for prov in _iter_engine_provenance_dicts(engine):
        if prov.get("whitespace_flexible") is False or prov.get("compact_wire") is True:
            return False
        if str(prov.get("wire_format", "")).strip().lower() == "compact":
            return False
        seps = prov.get("json_separators")
        if isinstance(seps, (list, tuple)) and len(seps) >= 2 and str(seps[1]) == ":":
            return False
    return None


def _is_nullable_schema_prop(prop: Any) -> bool:
    if not isinstance(prop, Mapping):
        return False
    if prop.get("nullable") is True:
        return True
    ptype = prop.get("type")
    if ptype == "null":
        return True
    if isinstance(ptype, (list, tuple)) and "null" in ptype:
        return True
    for comb in ("anyOf", "oneOf"):
        branches = prop.get(comb)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping):
                    btype = branch.get("type")
                    if btype == "null" or (isinstance(btype, (list, tuple)) and "null" in btype):
                        return True
    return False


def prune_optional_nullable_fields(
    schema: Mapping[str, Any] | str,
    *,
    prune_required_nullable: bool = False,
) -> dict[str, Any]:
    """Return a deep-copied JSON Schema with optional/nullable object properties pruned.

    Prevents QAT/compact checkpoints from exhaustively emitting placeholder values
    (`\"N/A\"`, `[]`, `false`, `null`) for unconstrained optional keys while keeping
    every emitted instance valid against the original unpruned schema.
    """
    if isinstance(schema, str):
        loaded = json.loads(schema)
        if not isinstance(loaded, dict):
            raise ValueError("JSON schema string must decode to a JSON object.")
        root_copy: dict[str, Any] = copy.deepcopy(loaded)
    else:
        root_copy = copy.deepcopy(dict(schema))

    def _prune_node(node: Any) -> Any:
        if isinstance(node, list):
            return [_prune_node(item) for item in node]
        if not isinstance(node, dict):
            return node

        out = dict(node)
        for defs_key in ("$defs", "definitions"):
            defs_map = out.get(defs_key)
            if isinstance(defs_map, dict):
                out[defs_key] = {k: _prune_node(v) for k, v in defs_map.items()}
        if "items" in out and isinstance(out["items"], (dict, list)):
            out["items"] = _prune_node(out["items"])
        if "prefixItems" in out and isinstance(out["prefixItems"], list):
            out["prefixItems"] = [_prune_node(x) for x in out["prefixItems"]]
        for comb_key in ("anyOf", "oneOf", "allOf"):
            if comb_key in out and isinstance(out[comb_key], list):
                out[comb_key] = [_prune_node(x) for x in out[comb_key]]

        props = out.get("properties")
        if isinstance(props, dict):
            req_raw = out.get("required")
            if isinstance(req_raw, list) and len(req_raw) > 0:
                req_list = [str(k) for k in req_raw]
                if prune_required_nullable:
                    kept_req = [k for k in req_list if not _is_nullable_schema_prop(props.get(k))]
                else:
                    kept_req = req_list
                kept_set = set(kept_req)
                out["properties"] = {k: _prune_node(v) for k, v in props.items() if k in kept_set}
                out["required"] = kept_req
                if out.get("additionalProperties") is not True:
                    out["additionalProperties"] = False
            else:
                out["properties"] = {
                    k: _prune_node(v) for k, v in props.items() if not _is_nullable_schema_prop(v)
                }
                if out.get("additionalProperties") is not True:
                    out["additionalProperties"] = False
        return out

    return _prune_node(root_copy)


def _resolve_compact_optional_schema(engine: Any, compact_optional_schema: bool) -> bool:
    if compact_optional_schema:
        return True
    for prov in _iter_engine_provenance_dicts(engine):
        if prov.get("compact_optional_schema") is True or prov.get("prune_optional_nullable_fields") is True:
            return True
        schema_pol = prov.get("schema_policy")
        if isinstance(schema_pol, Mapping) and schema_pol.get("compact_optional_schema") is True:
            return True
    return False


def _resolve_effective_schema(
    engine: Any,
    json_schema: str | dict[str, Any] | None,
    compact_optional_schema: bool,
) -> str | dict[str, Any] | None:
    if json_schema is None or not _resolve_compact_optional_schema(engine, compact_optional_schema):
        return json_schema
    pruned = prune_optional_nullable_fields(json_schema)
    return json.dumps(pruned, ensure_ascii=False) if isinstance(json_schema, str) else pruned


class _SchemaResolver:
    """Cache normalized JSON schemas and parsed dicts across multi-schema batches."""

    def __init__(
        self,
        engine: Any,
        default_schema: str | dict[str, Any] | None,
        *,
        compact_optional_schema: bool,
        emit_extractconf: bool,
    ) -> None:
        self._compact = _resolve_compact_optional_schema(engine, compact_optional_schema)
        self._emit_extractconf = bool(emit_extractconf)
        self._cache: dict[tuple[str, str], tuple[str | dict[str, Any], Mapping[str, Any] | None, str]] = {}
        self.default_raw = default_schema
        self.default_spec, self.default_dict, self.default_key = self._resolve_uncached(default_schema)

    def _resolve_uncached(
        self, raw_spec: str | Mapping[str, Any] | None
    ) -> tuple[str | dict[str, Any] | None, Mapping[str, Any] | None, str | None]:
        if raw_spec is None:
            return None, None, None
        eff: str | dict[str, Any]
        if isinstance(raw_spec, str):
            raw_key = ("str", raw_spec)
            cached = self._cache.get(raw_key)
            if cached is not None:
                return cached
            if self._compact:
                pruned = prune_optional_nullable_fields(raw_spec)
                eff = json.dumps(pruned, ensure_ascii=False)
                parsed_dict: Mapping[str, Any] | None = pruned if self._emit_extractconf else None
            else:
                eff = raw_spec
                parsed_dict = parse_json_schema_dict(eff) if self._emit_extractconf else None
            canon_key = eff
        else:
            raw_dict = dict(raw_spec)
            raw_str = json.dumps(raw_dict, sort_keys=True, ensure_ascii=False)
            raw_key = ("dict", raw_str)
            cached = self._cache.get(raw_key)
            if cached is not None:
                return cached
            if self._compact:
                eff = prune_optional_nullable_fields(raw_dict)
                canon_key = json.dumps(eff, sort_keys=True, ensure_ascii=False)
            else:
                eff = raw_dict
                canon_key = raw_str
            parsed_dict = eff if self._emit_extractconf else None
        out = (eff, parsed_dict, canon_key)
        self._cache[raw_key] = out
        return out

    def resolve(
        self, raw_spec: str | Mapping[str, Any] | None
    ) -> tuple[str | dict[str, Any] | None, Mapping[str, Any] | None, str | None]:
        if raw_spec is None:
            return self.default_spec, self.default_dict, self.default_key
        if raw_spec is self.default_raw:
            return self.default_spec, self.default_dict, self.default_key
        return self._resolve_uncached(raw_spec)


def _resolve_memory_governor(
    memory_governor: LiveMemoryGovernor | bool | None,
    uma_ceiling_fraction: float,
    *,
    engine: Any = None,
) -> LiveMemoryGovernor | None:
    if isinstance(memory_governor, LiveMemoryGovernor):
        return memory_governor
    if not memory_governor:
        return None
    if engine is not None and not (
        _engine_supports_rolling(engine)
        or hasattr(engine, "tokenizer")
        or hasattr(engine, "model")
        or hasattr(engine, "engine")
    ):
        return None
    return LiveMemoryGovernor(uma_ceiling_fraction=uma_ceiling_fraction)


def _engine_supports_rolling(engine: Any) -> bool:
    return (
        callable(getattr(engine, "run_rolling", None))
        and callable(getattr(engine, "encode_prompts", None))
        and callable(getattr(engine, "get_grammar", None))
        and callable(getattr(getattr(engine, "tokenizer", None), "decode", None))
    )


def _resolve_effective_thinking(
    *,
    enable_thinking: bool,
    has_schema: bool,
    clean_extra: Mapping[str, Any],
) -> bool:
    eff_thinking = bool(enable_thinking)
    if not eff_thinking and not has_schema:
        effort = clean_extra.get("reasoning_effort")
        budget = clean_extra.get("reasoning_budget")
        if effort is not None and str(effort).strip().lower() not in ("", "none", "skip", "off", "0"):
            return True
        if budget is not None and int(budget) > 0:
            return True
    return eff_thinking


_ENGINE_LOCKS_GUARD = threading.Lock()
_ENGINE_LOCKS: WeakKeyDictionary[Any, threading.RLock] = WeakKeyDictionary()
DEFAULT_RETRY_REASONS: tuple[str, ...] = ("length", "invalid_schema", "invalid_json")


def _get_engine_lock(engine: Any) -> threading.RLock:
    with _ENGINE_LOCKS_GUARD:
        lock = _ENGINE_LOCKS.get(engine)
        if lock is None:
            lock = threading.RLock()
            try:
                _ENGINE_LOCKS[engine] = lock
            except TypeError:
                pass
        return lock


def _resolve_stop_token_ids(tokenizer: Any) -> frozenset[int]:
    if tokenizer is None:
        return frozenset()
    out: set[int] = set()
    raw_stops = getattr(tokenizer, "stop_token_ids", None)
    if raw_stops is not None and not isinstance(raw_stops, (str, bytes)):
        try:
            for tid in raw_stops:
                if isinstance(tid, int) and not isinstance(tid, bool):
                    out.add(int(tid))
        except TypeError:
            pass
    for attr in ("eos_token_id", "eot_id", "eot_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and not isinstance(val, bool):
            out.add(int(val))
    return frozenset(out)


def _resolve_rolling_finish_reason(
    gen_tokens: Sequence[int],
    *,
    max_tokens: int,
    stop_ids: frozenset[int],
    explicit_finish_reason: str | None = None,
    valid_json: bool = True,
    valid_schema: bool = True,
) -> str:
    if explicit_finish_reason == "length":
        return "length"
    ended_with_stop = bool(gen_tokens and stop_ids and int(gen_tokens[-1]) in stop_ids)
    if max_tokens > 0 and len(gen_tokens) >= max_tokens and not ended_with_stop:
        return "length"
    if max_tokens > 0 and len(gen_tokens) >= max_tokens and not (valid_json and valid_schema):
        return "length"
    return explicit_finish_reason or "stop"


def _update_item_finish_reason(item: Any, finish_reason: str) -> Any:
    if str(getattr(item, "finish_reason", "stop")) == finish_reason:
        return item
    try:
        item.finish_reason = finish_reason
        return item
    except Exception:
        if is_dataclass(item):
            try:
                return replace(item, finish_reason=finish_reason)
            except Exception:
                pass
        return SimpleNamespace(
            text=getattr(item, "text", ""),
            parsed=getattr(item, "parsed", None),
            token_ids=getattr(item, "token_ids", None),
            prompt_tokens=int(getattr(item, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(item, "completion_tokens", 0) or 0),
            valid_json=bool(getattr(item, "valid_json", True)),
            valid_schema=bool(getattr(item, "valid_schema", True)),
            finish_reason=finish_reason,
            telemetry=getattr(item, "telemetry", None),
            repaired=bool(getattr(item, "repaired", False)),
            fallback_used=bool(getattr(item, "fallback_used", False)),
        )


def _normalize_batch_finish_reasons(
    engine: Any,
    items: Sequence[Any],
    *,
    max_tokens: int,
    entry_schemas: Sequence[str | dict[str, Any] | None],
) -> list[Any]:
    if max_tokens <= 0 or not items:
        return list(items)
    stop_ids = _resolve_stop_token_ids(getattr(engine, "tokenizer", None))
    upfront_cache: dict[str | None, tuple[int, ...]] = {}
    has_get_grammar = callable(getattr(engine, "get_grammar", None))

    def _upfront_for(spec: str | dict[str, Any] | None) -> tuple[int, ...]:
        if spec is None or not has_get_grammar:
            return ()
        sk = spec if isinstance(spec, str) else json.dumps(spec, sort_keys=True, ensure_ascii=False)
        cached = upfront_cache.get(sk)
        if cached is not None:
            return cached
        up: tuple[int, ...] = ()
        try:
            gh = engine.get_grammar(spec)
            raw_up = getattr(gh, "upfront_ff", ())
            if raw_up:
                up = tuple(int(t) for t in raw_up)
        except Exception:
            up = ()
        upfront_cache[sk] = up
        return up

    out: list[Any] = []
    for idx, it in enumerate(items):
        raw_fr = str(getattr(it, "finish_reason", "stop"))
        if raw_fr == "length":
            out.append(it)
            continue
        spec = entry_schemas[idx] if idx < len(entry_schemas) else None
        tok_ids = getattr(it, "token_ids", None)
        c_toks = int(getattr(it, "completion_tokens", 0) or 0)
        v_json = bool(getattr(it, "valid_json", True))
        v_schema = bool(getattr(it, "valid_schema", True))
        repaired = bool(getattr(it, "repaired", False))

        hit_len = False
        if isinstance(tok_ids, Sequence) and not isinstance(tok_ids, (str, bytes)):
            ended_with_stop = bool(tok_ids and stop_ids and int(tok_ids[-1]) in stop_ids)
            upfront_ff = _upfront_for(spec)
            matched_up = (
                len(upfront_ff)
                if (
                    upfront_ff
                    and len(tok_ids) >= len(upfront_ff)
                    and tuple(int(t) for t in tok_ids[: len(upfront_ff)]) == upfront_ff
                )
                else 0
            )
            eff_gen_len = len(tok_ids) - matched_up
            if (eff_gen_len >= max_tokens and not ended_with_stop) or (
                c_toks >= max_tokens and spec is not None and (not v_json or not v_schema)
            ):
                hit_len = True
        elif c_toks >= max_tokens and spec is not None and (not v_json or not v_schema or repaired):
            hit_len = True

        out.append(_update_item_finish_reason(it, "length") if hit_len else it)
    return out


def _resolve_retry_reasons(retry_on: Sequence[str] | str | None) -> tuple[str, ...]:
    if retry_on is None:
        return DEFAULT_RETRY_REASONS
    if isinstance(retry_on, str):
        parts = [p.strip().lower() for p in retry_on.split(",") if p.strip()]
        return tuple(parts) if parts else DEFAULT_RETRY_REASONS
    return tuple(str(x).strip().lower() for x in retry_on if str(x).strip())


def _should_retry_row(
    *,
    finish_reason: str,
    valid_json: bool,
    valid_schema: bool,
    extractconf_risk: float,
    retry_reasons: tuple[str, ...],
    retry_risk_threshold: float | None,
) -> bool:
    if finish_reason in ("skipped_empty", "empty"):
        return False
    if "always" in retry_reasons:
        return True
    if "length" in retry_reasons and finish_reason == "length":
        return True
    if "invalid_json" in retry_reasons and not valid_json:
        return True
    if "invalid_schema" in retry_reasons and not valid_schema:
        return True
    if retry_risk_threshold is not None and float(extractconf_risk) >= float(retry_risk_threshold):
        return True
    return False


def _run_awaitables_sync(coros: Sequence[Any]) -> None:
    if not coros:
        return

    async def _gather_all() -> None:
        for c in coros:
            await c

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_gather_all())
        return
    for c in coros:
        loop.create_task(c)


def _dispatch_retry_hooks(
    requests: Sequence[ExtractRetryRequest],
    *,
    deferred_retries: list[ExtractRetryRequest] | None = None,
    retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
    async_retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
) -> None:
    if not requests:
        return
    if deferred_retries is not None:
        deferred_retries.extend(requests)
    pending_coros: list[Any] = []
    if retry_hook is not None:
        for req in requests:
            res = retry_hook(req)
            if inspect.isawaitable(res):
                pending_coros.append(res)
    if async_retry_hook is not None:
        for req in requests:
            res = async_retry_hook(req)
            if inspect.isawaitable(res):
                pending_coros.append(res)
    _run_awaitables_sync(pending_coros)


def _schema_key(spec: str | dict[str, Any] | None) -> str | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    return json.dumps(spec, sort_keys=True, ensure_ascii=False)


def _execute_rolling_batch(
    engine: Any,
    prompts: Sequence[str],
    *,
    pre_encoded_ids: list[list[int]] | None,
    system_prompt: str,
    entry_schemas: Sequence[str | dict[str, Any] | None],
    grammar_cache: dict[tuple[Any, ...], Any],
    max_tokens: int,
    temperature: float,
    whitespace_flexible: bool | None,
    max_whitespace_run: int | None,
    b_slots: int,
    sort_by_length: bool,
    clean_extra: Mapping[str, Any],
    use_chat_template: bool,
    enable_thinking: bool,
) -> tuple[Any, list[int]]:
    """Run pre-tokenized, per-doc-schema rolling decode directly via `engine.run_rolling`."""
    n_docs = len(prompts)
    if n_docs == 0:
        return SimpleNamespace(items=[], stats={"docs": 0, "decode_sec": 0.0}), []

    def _get_gh(spec: str | dict[str, Any]) -> Any:
        s_key = spec if isinstance(spec, str) else json.dumps(spec, sort_keys=True, ensure_ascii=False)
        ck = (id(engine), s_key, whitespace_flexible, max_whitespace_run)
        hit = grammar_cache.get(ck)
        if hit is not None:
            return hit
        try:
            gh_obj = engine.get_grammar(
                spec,
                whitespace_flexible=whitespace_flexible,
                max_whitespace_run=max_whitespace_run,
            )
        except TypeError:
            gh_obj = engine.get_grammar(spec, whitespace_flexible=whitespace_flexible)
        if len(grammar_cache) > 128:
            grammar_cache.clear()
        grammar_cache[ck] = gh_obj
        return gh_obj

    prompt_ids_list = (
        [list(ids) for ids in pre_encoded_ids]
        if pre_encoded_ids is not None
        else [
            list(ids)
            for ids in engine.encode_prompts(
                prompts,
                system_prompt=system_prompt,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
        ]
    )

    non_none = [s for s in entry_schemas if s is not None]
    if not non_none:
        single_gh = None
        resolved_grammars: list[Any] | None = None
        upfront_by_doc: list[list[int]] = [[] for _ in range(n_docs)]
    elif len(non_none) == n_docs and len(
        {s if isinstance(s, str) else json.dumps(s, sort_keys=True, ensure_ascii=False) for s in non_none}
    ) == 1:
        single_gh = _get_gh(non_none[0])
        resolved_grammars = None
        up_list = list(getattr(single_gh, "upfront_ff", ()))
        upfront_by_doc = [list(up_list) for _ in range(n_docs)]
    else:
        single_gh = None
        resolved_grammars = [None if s is None else _get_gh(s) for s in entry_schemas]
        upfront_by_doc = [
            [] if g is None else list(getattr(g, "upfront_ff", ()))
            for g in resolved_grammars
        ]

    if sort_by_length and n_docs > 1:
        order = sorted(
            range(n_docs),
            key=lambda i: (-(len(prompt_ids_list[i]) + len(upfront_by_doc[i])), i),
        )
    else:
        order = list(range(n_docs))

    prompt_ids_ord = [prompt_ids_list[i] for i in order]
    upfront_ord = [upfront_by_doc[i] for i in order]
    grammars_ord = [resolved_grammars[i] for i in order] if resolved_grammars is not None else None

    bos_id = int(getattr(engine.tokenizer, "bos_token_id", 2))
    prompt_ids_ord = [
        x if (x or upfront_ord[i]) else [bos_id]
        for i, x in enumerate(prompt_ids_ord)
    ]

    lcp = min(len(x) for x in prompt_ids_ord)
    for k in range(lcp):
        t0_k = prompt_ids_ord[0][k]
        if any(x[k] != t0_k for x in prompt_ids_ord[1:]):
            lcp = k
            break
    min_raw_len = min(len(x) for x in prompt_ids_ord)
    if any(not u for u in upfront_ord) and lcp >= min_raw_len:
        lcp = max(0, min_raw_len - 1)

    sys_ids = prompt_ids_ord[0][:lcp]
    suf_all = [x[lcp:] + upfront_ord[i] for i, x in enumerate(prompt_ids_ord)]
    max_suf = max(len(s) for s in suf_all)
    eff_roll_slots = max(1, min(int(b_slots), len(suf_all)))
    win_sz = int(getattr(getattr(engine, "model", None), "window_size", 512))
    explicit_swa_cap = clean_extra.get("swa_ring_cap")
    if explicit_swa_cap is not None:
        roll_swa = int(explicit_swa_cap)
    elif (
        clean_extra.get("enable_ff_jump", True)
        and eff_roll_slots <= 8
        and (max_suf + max_tokens + 64) > win_sz
    ):
        roll_swa = max(1024, win_sz)
    else:
        roll_swa = max(512, win_sz)

    roll_kw: dict[str, Any] = {
        "sys_ids": sys_ids,
        "suf_ids_list": suf_all,
        "max_tokens": max_tokens,
        "B_slots": max(1, min(int(b_slots), n_docs)),
        "swa_ring_cap": roll_swa,
        "temperature": temperature,
        "gh": single_gh,
        "per_doc_grammars": grammars_ord,
        "sort_by_length": False,
        "use_ane_ring": clean_extra.get("use_ane_ring", "auto"),
    }
    for k in (
        "top_p",
        "seed",
        "kv_i8",
        "ane_cohort_size",
        "enable_ff_jump",
        "collect_telemetry",
        "collect_valid_lse",
        "override_memory_cap",
    ):
        if k in clean_extra:
            roll_kw[k] = clean_extra[k]

    roll_res = engine.run_rolling(**roll_kw)
    gen_toks_all = roll_res["gen_toks"]
    roll_telem = roll_res.get("telemetry")
    roll_valid_schemas = roll_res.get("valid_schemas")
    roll_finish_reasons = roll_res.get("finish_reasons")
    stop_ids = _resolve_stop_token_ids(getattr(engine, "tokenizer", None))

    items: list[Any] = []
    valid_json_cnt = 0
    valid_schema_cnt = 0
    for idx in range(n_docs):
        g_toks = gen_toks_all[idx]
        doc_g = grammars_ord[idx] if grammars_ord is not None else single_gh
        full_out_ids = upfront_ord[idx] + g_toks
        txt = engine.tokenizer.decode(full_out_ids, skip_special_tokens=True)
        parsed = None
        is_json = False
        is_schema = doc_g is None
        try:
            parsed = json.loads(txt)
            is_json = True
            valid_json_cnt += 1
        except Exception:
            pass
        if doc_g is not None:
            if roll_valid_schemas is not None:
                is_schema = bool(roll_valid_schemas[idx])
            else:
                is_schema = bool(doc_g.validate_token_sequence(full_out_ids))
        if is_schema:
            valid_schema_cnt += 1
        explicit_fr = (
            str(roll_finish_reasons[idx])
            if (
                roll_finish_reasons is not None
                and idx < len(roll_finish_reasons)
                and roll_finish_reasons[idx] is not None
            )
            else None
        )
        finish_reason = _resolve_rolling_finish_reason(
            g_toks,
            max_tokens=max_tokens,
            stop_ids=stop_ids,
            explicit_finish_reason=explicit_fr,
            valid_json=is_json,
            valid_schema=is_schema,
        )
        telem_i = roll_telem[idx] if (roll_telem is not None and idx < len(roll_telem)) else None
        items.append(
            SimpleNamespace(
                text=txt,
                parsed=parsed,
                token_ids=full_out_ids,
                prompt_tokens=len(prompt_ids_ord[idx]),
                completion_tokens=len(full_out_ids),
                valid_json=is_json,
                valid_schema=is_schema,
                finish_reason=finish_reason,
                telemetry=telem_i,
            )
        )

    stats = {
        "docs": n_docs,
        "valid_json_count": valid_json_cnt,
        "valid_schema_count": valid_schema_cnt,
        "prompt_tokens": sum(len(x) for x in prompt_ids_ord),
        "completion_tokens": int(roll_res.get("tot_out_tok", sum(len(it.token_ids) for it in items))),
        "decode_sec": float(roll_res.get("decode_sec", 0.0)),
        "swa_ring_cap": roll_swa,
    }
    return SimpleNamespace(items=items, stats=stats), order


def _generate_batch_with_schema_fallback(
    engine: Any,
    prompts: Sequence[str],
    *,
    entry_schemas: Sequence[str | dict[str, Any] | None],
    fallback_schema: str | dict[str, Any] | None,
    base_gen_kw: Mapping[str, Any],
    accepts_per_doc: bool,
    eff_b_slots: int,
) -> Any:
    """Route multi-schema batches natively via `per_doc_grammars` or group cleanly by schema."""
    distinct_specs = {_schema_key(s): s for s in entry_schemas}
    single_schema = next(iter(distinct_specs.values())) if len(distinct_specs) == 1 else fallback_schema
    kw = dict(base_gen_kw)
    eff_max_tokens = int(kw.get("max_tokens", 0) or 0)
    if len(distinct_specs) <= 1:
        kw["json_schema"] = single_schema
        raw_res = engine.generate_batch(prompts, **kw)
        norm_items = _normalize_batch_finish_reasons(
            engine, raw_res.items, max_tokens=eff_max_tokens, entry_schemas=entry_schemas
        )
        return SimpleNamespace(items=norm_items, stats=getattr(raw_res, "stats", {}) or {})
    if accepts_per_doc:
        kw["json_schema"] = single_schema
        kw["per_doc_grammars"] = list(entry_schemas)
        raw_res = engine.generate_batch(prompts, **kw)
        norm_items = _normalize_batch_finish_reasons(
            engine, raw_res.items, max_tokens=eff_max_tokens, entry_schemas=entry_schemas
        )
        return SimpleNamespace(items=norm_items, stats=getattr(raw_res, "stats", {}) or {})

    grouped: dict[str | None, list[int]] = {}
    for idx_e, sc in enumerate(entry_schemas):
        grouped.setdefault(_schema_key(sc), []).append(idx_e)

    combined_items: list[Any] = [None] * len(prompts)
    combined_stats: dict[str, float] = {
        "docs": 0,
        "valid_json_count": 0,
        "valid_schema_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "decode_sec": 0.0,
    }
    for idx_list in grouped.values():
        g_kw = dict(kw)
        g_kw["json_schema"] = entry_schemas[idx_list[0]]
        g_kw["batch_size"] = min(len(idx_list), eff_b_slots)
        sub_schemas = [entry_schemas[k] for k in idx_list]
        sub_res = engine.generate_batch([prompts[k] for k in idx_list], **g_kw)
        norm_sub_items = _normalize_batch_finish_reasons(
            engine, sub_res.items, max_tokens=eff_max_tokens, entry_schemas=sub_schemas
        )
        for pos_sub, orig_e in enumerate(idx_list):
            combined_items[orig_e] = norm_sub_items[pos_sub]
        sub_stats = getattr(sub_res, "stats", {}) or {}
        combined_stats["docs"] += int(sub_stats.get("docs", len(norm_sub_items)))
        combined_stats["valid_json_count"] += int(
            sub_stats.get("valid_json_count", sum(1 for it in norm_sub_items if it.valid_json))
        )
        combined_stats["valid_schema_count"] += int(
            sub_stats.get("valid_schema_count", sum(1 for it in norm_sub_items if it.valid_schema))
        )
        combined_stats["prompt_tokens"] += int(
            sub_stats.get("prompt_tokens", sum(it.prompt_tokens for it in norm_sub_items))
        )
        combined_stats["completion_tokens"] += int(
            sub_stats.get("completion_tokens", sum(it.completion_tokens for it in norm_sub_items))
        )
        combined_stats["decode_sec"] += float(sub_stats.get("decode_sec", 0.0))
    return SimpleNamespace(items=combined_items, stats=combined_stats)


def _resolve_default_b_slots(engine: Any) -> int:
    for attr in ("default_max_b_slots", "max_b_slots"):
        val = getattr(engine, attr, None)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return int(val)
    try:
        return int(DynamicSiliconPolicy().host.extraction_batch_size)
    except Exception:
        return 128


_UNSET_PARSED = object()


def _engine_accepts_explicit_kwarg(engine: Any, kwarg_name: str) -> bool:
    fn = getattr(engine, "generate_batch", None)
    if not callable(fn):
        return False
    try:
        param = inspect.signature(fn).parameters.get(kwarg_name)
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _score_item_extractconf(
    item: Any,
    *,
    json_schema: Mapping[str, Any] | str | None,
    max_tokens: int,
    calibrator: ExtractConfCalibrator | None,
    parsed: Any = _UNSET_PARSED,
    valid_json: bool | None = None,
) -> float:
    eff_valid_json = bool(getattr(item, "valid_json", True)) if valid_json is None else bool(valid_json)
    resp_text = getattr(item, "text", None)
    if parsed is _UNSET_PARSED:
        eff_parsed = getattr(item, "parsed", None)
        if eff_parsed is None and eff_valid_json and isinstance(resp_text, str) and resp_text.strip():
            try:
                eff_parsed = json.loads(resp_text)
            except Exception:
                eff_valid_json = False
    else:
        eff_parsed = parsed
    return compute_extractconf_risk(
        parsed=eff_parsed,
        response_text=resp_text,
        json_schema=json_schema,
        prompt_tokens=int(getattr(item, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(item, "completion_tokens", 0) or 0),
        max_tokens=max_tokens,
        valid_json=eff_valid_json,
        valid_schema=bool(getattr(item, "valid_schema", True)),
        finish_reason=str(getattr(item, "finish_reason", "stop")),
        repaired=bool(getattr(item, "repaired", False)),
        fallback_used=bool(getattr(item, "fallback_used", False)),
        telemetry=getattr(item, "telemetry", None),
        calibrator=calibrator,
        _already_parsed=True,
    )


def _is_retry_better(
    *,
    cur_valid_schema: bool,
    cur_valid_json: bool,
    cur_finish_reason: str,
    cur_risk: float,
    cand_valid_schema: bool,
    cand_valid_json: bool,
    cand_finish_reason: str,
    cand_risk: float,
) -> bool:
    cur_rank = (
        1 if cur_valid_schema else 0,
        1 if cur_valid_json else 0,
        1 if cur_finish_reason != "length" else 0,
        -float(cur_risk),
    )
    cand_rank = (
        1 if cand_valid_schema else 0,
        1 if cand_valid_json else 0,
        1 if cand_finish_reason != "length" else 0,
        -float(cand_risk),
    )
    return cand_rank > cur_rank


def run_deferred_retries(
    engine: Any,
    requests: Sequence[ExtractRetryRequest],
    *,
    system_prompt: str | None = None,
    retry_max_tokens: int | None = None,
    temperature: float | None = None,
    whitespace_flexible: bool | None = None,
    max_whitespace_run: int | None = None,
    mode: str | None = None,
    speculative: bool | None = None,
    b_slots: int | None = None,
    use_rolling: bool | str = "auto",
    emit_extractconf: bool = True,
    extractconf_calibrator: ExtractConfCalibrator | None = None,
    retry_prompt_builder: Callable[[ExtractRetryRequest], str] | None = None,
    on_retry_complete: Callable[[list[dict[str, Any]]], Any] | None = None,
    clear_cache_mb: int = 256,
    grammar_cache: dict[tuple[Any, ...], Any] | None = None,
    **extra_gen_kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute deferred extraction retries sequentially/batched on the single already-loaded `engine`."""
    if not requests:
        return []

    req_list = list(requests)
    eff_ws = resolve_engine_whitespace_flexible(engine, whitespace_flexible)
    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
    if max_whitespace_run is not None and _engine_accepts_explicit_kwarg(engine, "max_whitespace_run"):
        clean_extra.setdefault("max_whitespace_run", int(max_whitespace_run))
    if emit_extractconf and "collect_telemetry" not in clean_extra and _engine_accepts_explicit_kwarg(engine, "collect_telemetry"):
        clean_extra["collect_telemetry"] = True

    eff_slots = int(b_slots) if (b_slots is not None and int(b_slots) > 0) else _resolve_default_b_slots(engine)
    eff_mode = str(mode or getattr(engine, "default_mode", "auto") or "auto").lower()
    accepts_per_doc = _engine_accepts_explicit_kwarg(engine, "per_doc_grammars")
    can_roll = _engine_supports_rolling(engine)
    eff_use_chat = bool(clean_extra.get("use_chat_template", True))
    g_cache = grammar_cache if grammar_cache is not None else {}

    # Deduplicate identical retry `(system_prompt, prompt, schema_key, budget)` pairs cleanly.
    groups: dict[tuple[str, int], list[int]] = {}
    for idx_r, req in enumerate(req_list):
        sys_p = req.system_prompt if system_prompt is None else system_prompt
        budget = (
            int(retry_max_tokens)
            if retry_max_tokens is not None
            else max(int(req.max_tokens) * 2, 256)
        )
        groups.setdefault((sys_p, budget), []).append(idx_r)

    results: list[dict[str, Any]] = [{} for _ in req_list]
    eff_temp = 0.0 if temperature is None else float(temperature)
    lock = _get_engine_lock(engine)
    with lock:
        for (sys_p, budget), group_indices in groups.items():
            clear_mlx_arena(clear_cache_mb)
            prompts = [
                retry_prompt_builder(req_list[i]) if retry_prompt_builder is not None else req_list[i].prompt
                for i in group_indices
            ]
            entry_schemas = [req_list[i].json_schema for i in group_indices]
            schema_dicts = [req_list[i].json_schema_dict for i in group_indices]
            has_any_schema = any(s is not None for s in entry_schemas)
            eff_thinking = _resolve_effective_thinking(
                enable_thinking=bool(clean_extra.get("enable_thinking", False)),
                has_schema=has_any_schema,
                clean_extra=clean_extra,
            )

            use_roll = False
            if can_roll:
                if use_rolling is True:
                    use_roll = True
                elif str(use_rolling).strip().lower() == "auto":
                    use_roll = (
                        not eff_thinking
                        and (has_any_schema or len(prompts) > 4 or speculative is False)
                        and eff_mode != "hetero"
                    )

            if use_roll:
                res_roll, order = _execute_rolling_batch(
                    engine,
                    prompts,
                    pre_encoded_ids=None,
                    system_prompt=sys_p,
                    entry_schemas=entry_schemas,
                    grammar_cache=g_cache,
                    max_tokens=budget,
                    temperature=eff_temp,
                    whitespace_flexible=eff_ws,
                    max_whitespace_run=max_whitespace_run,
                    b_slots=eff_slots,
                    sort_by_length=True,
                    clean_extra=clean_extra,
                    use_chat_template=eff_use_chat,
                    enable_thinking=eff_thinking,
                )
                ordered_items: list[Any] = [None] * len(prompts)
                for pos_r, orig_r in enumerate(order):
                    ordered_items[orig_r] = res_roll.items[pos_r]
                out_items = ordered_items
                dec_sec = float(res_roll.stats.get("decode_sec", 0.0))
            else:
                gen_kw: dict[str, Any] = {
                    "system_prompt": sys_p,
                    "max_tokens": budget,
                    "temperature": eff_temp,
                    "whitespace_flexible": eff_ws,
                    "batch_size": min(len(prompts), eff_slots),
                    **clean_extra,
                }
                if max_whitespace_run is not None and not _engine_accepts_explicit_kwarg(engine, "max_whitespace_run"):
                    gen_kw.pop("max_whitespace_run", None)
                if mode is not None:
                    gen_kw["mode"] = mode
                if speculative is not None:
                    gen_kw["speculative"] = speculative
                res_b = _generate_batch_with_schema_fallback(
                    engine,
                    prompts,
                    entry_schemas=entry_schemas,
                    fallback_schema=entry_schemas[0] if len(entry_schemas) == 1 else None,
                    base_gen_kw=gen_kw,
                    accepts_per_doc=accepts_per_doc,
                    eff_b_slots=eff_slots,
                )
                out_items = list(res_b.items)
                dec_sec = float((getattr(res_b, "stats", {}) or {}).get("decode_sec", 0.0))

            clear_mlx_arena(clear_cache_mb)
            per_item_dec = dec_sec / max(1, len(group_indices))
            for g_pos, orig_r in enumerate(group_indices):
                req = req_list[orig_r]
                item = out_items[g_pos]
                r_text = str(getattr(item, "text", ""))
                v_json = bool(getattr(item, "valid_json", False))
                v_schema = bool(getattr(item, "valid_schema", True))
                f_reason = str(getattr(item, "finish_reason", "stop"))
                p_toks = int(getattr(item, "prompt_tokens", 0) or 0)
                c_toks = int(getattr(item, "completion_tokens", 0) or 0)
                it_parsed = getattr(item, "parsed", None)
                if it_parsed is None and v_json and r_text.strip():
                    try:
                        it_parsed = json.loads(r_text)
                    except Exception:
                        v_json = False
                it_risk = (
                    _score_item_extractconf(
                        item,
                        parsed=it_parsed,
                        valid_json=v_json,
                        json_schema=schema_dicts[g_pos],
                        max_tokens=budget,
                        calibrator=extractconf_calibrator,
                    )
                    if emit_extractconf
                    else 0.0
                )
                improved = _is_retry_better(
                    cur_valid_schema=req.valid_schema,
                    cur_valid_json=req.valid_json,
                    cur_finish_reason=req.finish_reason,
                    cur_risk=req.extractconf_risk,
                    cand_valid_schema=v_schema,
                    cand_valid_json=v_json,
                    cand_finish_reason=f_reason,
                    cand_risk=it_risk,
                )
                results[orig_r] = {
                    "row_index": req.row_index,
                    "attempt": req.attempt,
                    "max_tokens": budget,
                    "prompt": prompts[g_pos],
                    "text": r_text,
                    "parsed": it_parsed,
                    "valid_json": v_json,
                    "valid_schema": v_schema,
                    "finish_reason": f_reason,
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "extractconf_risk": float(it_risk),
                    "decode_sec": per_item_dec,
                    "improved": improved,
                    "metadata": dict(req.metadata),
                }

    if on_retry_complete is not None:
        hook_res = on_retry_complete(results)
        if inspect.isawaitable(hook_res):
            _run_awaitables_sync([hook_res])
    return results


async def run_deferred_retries_async(
    engine: Any,
    requests: Sequence[ExtractRetryRequest],
    *,
    async_complete_hook: Callable[[list[dict[str, Any]]], Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run deferred single-model retries off the asyncio loop via `asyncio.to_thread` without co-loading models."""
    results = await asyncio.to_thread(run_deferred_retries, engine, requests, **kwargs)
    if async_complete_hook is not None:
        res = async_complete_hook(results)
        if inspect.isawaitable(res):
            await res
    return results


def _register_governor_shed_hooks(
    stack: ExitStack,
    governor: Any,
    engine: Any,
    grammar_cache: dict[tuple[Any, ...], Any] | None,
) -> None:
    if governor is None or not hasattr(governor, "register_shed_callback"):
        return
    if grammar_cache is not None:
        governor.register_shed_callback(grammar_cache.clear)
        stack.callback(governor.unregister_shed_callback, grammar_cache.clear)
    for pt_attr in ("prefix_tree", "_prefix_tree"):
        pt_obj = getattr(engine, pt_attr, None)
        shed_fn = getattr(pt_obj, "shed", None) if pt_obj is not None else None
        if callable(shed_fn):
            governor.register_shed_callback(shed_fn)
            stack.callback(governor.unregister_shed_callback, shed_fn)
            break


def _enter_mlx_limits(
    stack: ExitStack,
    *,
    mlx_cache_limit_mb: int | None,
    use_wired_limit: bool,
) -> None:
    if mlx_cache_limit_mb is not None:
        prev_limit: int | None = None
        try:
            import mlx.core as mx

            set_limit = getattr(mx, "set_cache_limit", None)
            if callable(set_limit):
                prev = set_limit(max(0, int(mlx_cache_limit_mb)) * 1024 * 1024)
                if isinstance(prev, int):
                    prev_limit = prev
        except Exception:
            set_mlx_cache_limit(int(mlx_cache_limit_mb))
        if prev_limit is not None:

            def _restore(val: int = prev_limit) -> None:
                try:
                    import mlx.core as mx

                    set_limit = getattr(mx, "set_cache_limit", None)
                    if callable(set_limit):
                        set_limit(val)
                except Exception:
                    pass

            stack.callback(_restore)
    if use_wired_limit:
        stack.enter_context(wired_limit())


def _iter_filtered_batches(
    pf: pq.ParquetFile,
    *,
    read_cols: list[str],
    batch_size: int,
    done_ids: set[Any],
    resume_key_col: str | None,
    filter_ocr: bool,
) -> Iterator[pa.RecordBatch]:
    """Yield filtered `RecordBatch` slices, pre-checking ID/OCR columns per row group before decompressing text."""
    avail = set(pf.schema_arrow.names)
    kcol: str | None = None
    if done_ids:
        if resume_key_col and resume_key_col in avail:
            kcol = resume_key_col
        elif "id" in avail:
            kcol = "id"
        elif "body_sha" in avail:
            kcol = "body_sha"

    has_iso = filter_ocr and ("is_ocr" in avail)
    has_op = filter_ocr and ("ocr_pages" in avail)

    if kcol is None and not (has_iso or has_op):
        yield from pf.iter_batches(batch_size=batch_size, columns=read_cols)
        return

    pre_cols: list[str] = []
    if kcol is not None:
        pre_cols.append(kcol)
    if has_iso and "is_ocr" not in pre_cols:
        pre_cols.append("is_ocr")
    if has_op and "ocr_pages" not in pre_cols:
        pre_cols.append("ocr_pages")

    for rg_idx in range(pf.num_row_groups):
        pre_tbl = pf.read_row_group(rg_idx, columns=pre_cols)
        n_rg = pre_tbl.num_rows
        if n_rg == 0:
            continue

        keep_flags = [True] * n_rg
        if kcol is not None:
            ids_list = pre_tbl.column(kcol).to_pylist()
            for i, val in enumerate(ids_list):
                if val in done_ids:
                    keep_flags[i] = False

        if has_iso or has_op:
            iso_list = pre_tbl.column("is_ocr").to_pylist() if has_iso else None
            op_list = pre_tbl.column("ocr_pages").to_pylist() if has_op else None
            for i in range(n_rg):
                if not keep_flags[i]:
                    continue
                v_iso = bool(iso_list[i]) if iso_list is not None and iso_list[i] is not None else False
                v_op = len(op_list[i]) > 0 if op_list is not None and op_list[i] is not None else False
                keep_flags[i] = v_iso or v_op

        if not any(keep_flags):
            # Skip decompressing/reading heavy text columns on completed/filtered row groups!
            continue

        offset = 0
        for rb in pf.iter_batches(batch_size=batch_size, row_groups=[rg_idx], columns=read_cols):
            sub_flags = keep_flags[offset : offset + rb.num_rows]
            offset += rb.num_rows
            if all(sub_flags):
                yield rb
            elif any(sub_flags):
                yield rb.filter(pa.array(sub_flags, type=pa.bool_()))


def extract_parquet(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    system_prompt: str,
    json_schema: str | dict[str, Any] | None = None,
    compact_optional_schema: bool = False,
    text_col: str | None = None,
    schema_col: str | None = None,
    per_doc_grammars: (
        Callable[[Mapping[str, Any]], str | dict[str, Any] | None]
        | Mapping[Any, str | dict[str, Any] | None]
        | Sequence[str | dict[str, Any] | None]
        | None
    ) = None,
    keep_cols: list[str] | None = None,
    drop_text_col: bool = False,
    mode: str | None = None,
    speculative: bool | None = None,
    batch_rows: int | None = None,
    macro_window_rows: int = 4096,
    sort_by_length: bool = True,
    dedup_within_window: bool = True,
    gpu_batch_size: int | None = None,
    shard_rows: int = 4096,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
    max_whitespace_run: int | None = None,
    max_chars: int | None = None,
    max_pages: int = 0,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    skip_empty: bool = False,
    resume: bool = False,
    emit_extractconf: bool = True,
    extractconf_calibrator: ExtractConfCalibrator | None = None,
    mlx_cache_limit_mb: int | None = DEFAULT_MLX_CACHE_LIMIT_MB,
    use_wired_limit: bool = True,
    use_rolling: bool | str = "auto",
    lookahead_batches: int | None = None,
    memory_governor: LiveMemoryGovernor | bool | None = True,
    uma_ceiling_fraction: float = WIRED_MEMORY_CEILING_FRACTION,
    use_nocache: bool = False,
    max_retries: int = 0,
    retry_max_tokens: int | None = None,
    retry_on: Sequence[str] | str = DEFAULT_RETRY_REASONS,
    retry_risk_threshold: float | None = None,
    retry_temperature: float | None = None,
    retry_prompt_builder: Callable[[ExtractRetryRequest], str] | None = None,
    defer_retries: bool = False,
    deferred_retries: list[ExtractRetryRequest] | None = None,
    retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
    async_retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
    **extra_gen_kwargs: Any,
) -> ParquetExtractSummary:
    """Stream-extract structured JSON from Parquet input(s) with overlapped reader/GPU/writer threads."""
    in_files = resolve_input_files(input_path)
    out = Path(output_path)
    sharded = out.suffix.lower() != ".parquet"
    if sharded:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[Any] = set()
    resume_key_col: str | None = None
    if resume and sharded:
        status_by_id: dict[Any, bool] = {}
        for shard in sorted(out.glob("part-*.parquet")):
            pf_s = pq.ParquetFile(shard)
            names = pf_s.schema_arrow.names
            kcol = "id" if "id" in names else ("body_sha" if "body_sha" in names else None)
            if kcol is not None:
                resume_key_col = kcol
                read_s_cols = [kcol]
                has_vs = "valid_schema" in names
                has_fr = "finish_reason" in names
                if has_vs:
                    read_s_cols.append("valid_schema")
                if has_fr:
                    read_s_cols.append("finish_reason")
                tbl_s = pq.read_table(shard, columns=read_s_cols)
                keys_s = tbl_s.column(kcol).to_pylist()
                vs_s = tbl_s.column("valid_schema").to_pylist() if has_vs else None
                fr_s = tbl_s.column("finish_reason").to_pylist() if has_fr else None
                for i_s, key_val in enumerate(keys_s):
                    fr_val = fr_s[i_s] if fr_s is not None else None
                    if fr_val in ("skipped_empty", "empty"):
                        status_by_id[key_val] = True
                    else:
                        ok_vs = True if vs_s is None else bool(vs_s[i_s])
                        ok_fr = True if fr_s is None else (fr_val is None or fr_val == "stop")
                        status_by_id[key_val] = bool(ok_vs and ok_fr)
        done_ids = {k for k, ok in status_by_id.items() if ok}

    eff_ws = resolve_engine_whitespace_flexible(engine, whitespace_flexible)
    schema_resolver = _SchemaResolver(
        engine,
        json_schema,
        compact_optional_schema=compact_optional_schema,
        emit_extractconf=emit_extractconf,
    )
    eff_schema = schema_resolver.default_spec
    eff_schema_dict = schema_resolver.default_dict
    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
    if max_whitespace_run is not None:
        clean_extra.setdefault("max_whitespace_run", int(max_whitespace_run))
    if emit_extractconf and "collect_telemetry" not in clean_extra and _engine_accepts_explicit_kwarg(engine, "collect_telemetry"):
        clean_extra["collect_telemetry"] = True
    roll_kw = clean_extra.get("roll_slots")
    bs_kw = clean_extra.pop("batch_size", None)
    policy_b_slots = _resolve_default_b_slots(engine)
    roll_default = (
        int(roll_kw)
        if isinstance(roll_kw, int) and not isinstance(roll_kw, bool) and int(roll_kw) > 0
        else policy_b_slots
    )
    bs_extra = int(bs_kw) if isinstance(bs_kw, int) and not isinstance(bs_kw, bool) and int(bs_kw) > 0 else None

    eff_gpu_batch_size = (
        int(gpu_batch_size)
        if gpu_batch_size is not None
        else (bs_extra if bs_extra is not None else roll_default)
    )
    eff_lookahead = (
        max(1, int(lookahead_batches))
        if lookahead_batches is not None
        else (8 if (batch_rows is not None and int(batch_rows) >= 16) else 1)
    )
    if batch_rows is not None:
        if int(batch_rows) < 1:
            raise ValueError(f"Macro-window / batch_rows size must be >= 1, got {batch_rows}")
        if eff_lookahead > 1:
            eff_window_rows = min(max(int(macro_window_rows), eff_gpu_batch_size), int(batch_rows) * eff_lookahead)
        else:
            eff_window_rows = int(batch_rows)
    else:
        eff_window_rows = max(int(macro_window_rows), eff_gpu_batch_size)
    if eff_window_rows < 1:
        raise ValueError(f"Macro-window / batch_rows size must be >= 1, got {eff_window_rows}")

    governor = _resolve_memory_governor(memory_governor, uma_ceiling_fraction, engine=engine)
    eff_cache_mb = DEFAULT_MLX_CACHE_LIMIT_MB if mlx_cache_limit_mb is None else int(mlx_cache_limit_mb)
    engine_can_roll = _engine_supports_rolling(engine)
    eff_use_chat = bool(clean_extra.get("use_chat_template", True))
    eff_mode = str(mode or getattr(engine, "default_mode", "auto") or "auto").lower()
    accepts_per_doc = _engine_accepts_explicit_kwarg(engine, "per_doc_grammars")

    prep_queue: queue.Queue[tuple[pa.Table, list[str | None], list[Any] | None] | None | BaseException] = queue.Queue(maxsize=2)
    ready_queue: queue.Queue[dict[str, Any] | None | BaseException] = queue.Queue(maxsize=2)
    write_queue: queue.Queue[pa.Table | None] = queue.Queue(maxsize=2)
    reader_stop = threading.Event()
    tok_stop = threading.Event()
    writer_abort = threading.Event()
    writer_err: list[BaseException] = []

    def _put_prep(item: tuple[pa.Table, list[str | None], list[Any] | None] | None | BaseException) -> bool:
        while not reader_stop.is_set():
            try:
                prep_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _reader_worker() -> None:
        try:
            acc_tables: list[pa.Table] = []
            acc_prompts: list[str | None] = []
            acc_schemas: list[Any] | None = [] if schema_col is not None else None
            acc_rows = 0

            def _flush_window() -> bool:
                nonlocal acc_tables, acc_prompts, acc_schemas, acc_rows
                if not acc_tables:
                    return True
                merged = acc_tables[0] if len(acc_tables) == 1 else pa.concat_tables(acc_tables)
                prompts_out = acc_prompts
                schemas_out = acc_schemas
                acc_tables = []
                acc_prompts = []
                acc_schemas = [] if schema_col is not None else None
                acc_rows = 0
                return _put_prep((merged, prompts_out, schemas_out))

            read_batch_sz = int(batch_rows) if batch_rows is not None else eff_window_rows
            for fpath in in_files:
                if reader_stop.is_set():
                    return
                with ExitStack() as fstack:
                    if use_nocache:
                        fh = fstack.enter_context(nocache_open(fpath, "rb", buffering=0))
                        pf = pq.ParquetFile(fh)
                    else:
                        pf = pq.ParquetFile(fpath)
                    schema = pf.schema_arrow
                    tcol = resolve_llm_text_column(schema, text_col)
                    avail = set(schema.names)
                    if schema_col is not None and schema_col not in avail:
                        raise KeyError(f"Schema column {schema_col!r} not found in {fpath}; available={schema.names}")
                    meta_cols = (
                        [c for c in keep_cols if c in avail]
                        if keep_cols is not None
                        else [c for c in schema.names if not (drop_text_col and c == tcol)]
                    )
                    meta_only_cols = [c for c in meta_cols if c != "page_offsets"]
                    read_cols = list(dict.fromkeys(meta_cols + [tcol] + ([schema_col] if schema_col is not None else [])))
                    for extra in ("is_ocr", "ocr_pages", "page_offsets"):
                        if extra in avail and extra not in read_cols:
                            read_cols.append(extra)

                    for rb in _iter_filtered_batches(
                        pf,
                        read_cols=read_cols,
                        batch_size=read_batch_sz,
                        done_ids=done_ids,
                        resume_key_col=resume_key_col,
                        filter_ocr=filter_ocr,
                    ):
                        if reader_stop.is_set():
                            return

                        texts_raw = rb.column(tcol).to_pylist()
                        sch_col_vals = rb.column(schema_col).to_pylist() if schema_col is not None else None
                        offsets_col = rb.column("page_offsets").to_pylist() if "page_offsets" in rb.schema.names else None
                        ocr_pages_col = rb.column("ocr_pages").to_pylist() if "ocr_pages" in rb.schema.names else None

                        kept_indices: list[int] = []
                        prompts: list[str | None] = []
                        kept_schemas: list[Any] | None = [] if sch_col_vals is not None else None
                        for i in range(rb.num_rows):
                            t = texts_raw[i] if isinstance(texts_raw[i], str) else ""
                            if ocr_pages_only and offsets_col is not None and ocr_pages_col is not None:
                                t = slice_ocr_pages_utf8(t, offsets_col[i], ocr_pages_col[i])
                            elif max_pages > 0 and offsets_col is not None:
                                t = slice_first_pages_utf8(t, offsets_col[i], max_pages)
                            if max_chars is not None and int(max_chars) > 0 and len(t) > int(max_chars):
                                t = t[: int(max_chars)]
                            if not t.strip():
                                if skip_empty:
                                    continue
                                kept_indices.append(i)
                                prompts.append(None)
                                if kept_schemas is not None and sch_col_vals is not None:
                                    kept_schemas.append(sch_col_vals[i])
                            else:
                                kept_indices.append(i)
                                prompts.append(t)
                                if kept_schemas is not None and sch_col_vals is not None:
                                    kept_schemas.append(sch_col_vals[i])

                        if not kept_indices:
                            continue

                        meta_tbl = pa.Table.from_batches([rb.take(kept_indices)]).select(meta_only_cols)
                        cursor = 0
                        total_rb = meta_tbl.num_rows
                        while cursor < total_rb:
                            space = eff_window_rows - acc_rows
                            take_n = min(space, total_rb - cursor)
                            acc_tables.append(meta_tbl.slice(cursor, take_n))
                            acc_prompts.extend(prompts[cursor : cursor + take_n])
                            if acc_schemas is not None and kept_schemas is not None:
                                acc_schemas.extend(kept_schemas[cursor : cursor + take_n])
                            acc_rows += take_n
                            cursor += take_n
                            if acc_rows >= eff_window_rows:
                                if not _flush_window():
                                    return

            if acc_rows > 0:
                _flush_window()
        except BaseException as exc:
            _put_prep(exc)
        finally:
            _put_prep(None)

    shard_idx = next_shard_index(out) if sharded else 0

    def _writer_worker() -> None:
        nonlocal shard_idx
        writer: pq.ParquetWriter | None = None
        tmp_dst: Path | None = None
        final_dst: Path | None = None
        shard_acc = 0
        try:
            while True:
                if writer_abort.is_set():
                    break
                try:
                    out_table = write_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if out_table is None:
                    break
                if writer is None:
                    final_dst = (out / f"part-{shard_idx:05d}.parquet") if sharded else out
                    tmp_dst = final_dst.with_name(f".{final_dst.name}.tmp.{os.getpid()}")
                    writer = pq.ParquetWriter(tmp_dst, out_table.schema, compression="zstd")
                    shard_acc = 0

                writer.write_table(out_table)
                shard_acc += out_table.num_rows
                if sharded and shard_acc >= shard_rows:
                    writer.close()
                    writer = None
                    assert tmp_dst is not None and final_dst is not None
                    os.replace(tmp_dst, final_dst)
                    tmp_dst = None
                    final_dst = None
                    shard_idx += 1

            if writer is not None:
                if writer_abort.is_set():
                    try:
                        writer.close()
                    except Exception:
                        pass
                    writer = None
                    if tmp_dst is not None:
                        tmp_dst.unlink(missing_ok=True)
                else:
                    writer.close()
                    writer = None
                    assert tmp_dst is not None and final_dst is not None
                    os.replace(tmp_dst, final_dst)
                    tmp_dst = None
                    final_dst = None
        except BaseException as exc:
            writer_err.append(exc)
            writer_abort.set()
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                writer = None
            if tmp_dst is not None:
                tmp_dst.unlink(missing_ok=True)

    def _enqueue_write(tbl: pa.Table | None) -> None:
        while True:
            if writer_err:
                raise writer_err[0]
            try:
                write_queue.put(tbl, timeout=0.05)
                return
            except queue.Full:
                continue

    def _put_ready(item: dict[str, Any] | None | BaseException) -> bool:
        while not tok_stop.is_set():
            try:
                ready_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _resolve_active_schemas(
        meta_tbl: pa.Table,
        raw_schemas: list[Any] | None,
        active_indices: list[int],
        win_row_offset: int,
    ) -> tuple[list[str | dict[str, Any] | None], list[Mapping[str, Any] | None], list[str | None], bool]:
        specs: list[str | dict[str, Any] | None] = []
        dicts: list[Mapping[str, Any] | None] = []
        keys: list[str | None] = []
        is_callable_g = callable(per_doc_grammars)
        is_mapping_g = isinstance(per_doc_grammars, Mapping)
        is_seq_g = isinstance(per_doc_grammars, Sequence) and not isinstance(per_doc_grammars, (str, bytes))
        col_cache: dict[str, list[Any]] | None = None
        if is_callable_g and active_indices:
            col_cache = {name: meta_tbl.column(name).to_pylist() for name in meta_tbl.schema.names}

        for idx in active_indices:
            g_idx = win_row_offset + idx
            raw_spec: Any = None
            raw_col_val = raw_schemas[idx] if raw_schemas is not None else None
            if is_callable_g:
                assert col_cache is not None
                row_dict = {name: vals[idx] for name, vals in col_cache.items()}
                row_dict["_row_index"] = g_idx
                raw_spec = per_doc_grammars(row_dict)  # type: ignore[operator]
            elif is_mapping_g:
                if raw_col_val is not None and raw_col_val in per_doc_grammars:
                    raw_spec = per_doc_grammars[raw_col_val]
                elif g_idx in per_doc_grammars:
                    raw_spec = per_doc_grammars[g_idx]
                else:
                    raw_spec = raw_col_val
            elif is_seq_g:
                raw_spec = per_doc_grammars[g_idx] if g_idx < len(per_doc_grammars) else raw_col_val
            else:
                raw_spec = raw_col_val

            eff_s, eff_d, eff_k = schema_resolver.resolve(raw_spec)
            specs.append(eff_s)
            dicts.append(eff_d)
            keys.append(eff_k)

        distinct_keys = set(keys)
        has_mixed = len(distinct_keys) > 1 or (len(distinct_keys) == 1 and next(iter(distinct_keys)) != schema_resolver.default_key)
        return specs, dicts, keys, has_mixed

    def _tok_worker() -> None:
        try:
            win_row_offset = 0
            while not tok_stop.is_set():
                try:
                    raw_item = prep_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if raw_item is None:
                    break
                if isinstance(raw_item, BaseException):
                    _put_ready(raw_item)
                    return

                meta_tbl, raw_prompts, raw_schemas = raw_item
                n_win = len(raw_prompts)
                active_indices = [idx for idx, p in enumerate(raw_prompts) if p is not None]
                prepared: dict[str, Any] = {
                    "meta_tbl": meta_tbl,
                    "raw_prompts": raw_prompts,
                    "active_indices": active_indices,
                    "unique_entries": [],
                    "entry_schemas": [],
                    "entry_schema_dicts": [],
                    "pre_encoded_ids": None,
                    "use_roll": False,
                    "eff_thinking": False,
                }
                if active_indices:
                    act_specs, act_dicts, act_keys, has_mixed_schemas = _resolve_active_schemas(
                        meta_tbl, raw_schemas, active_indices, win_row_offset
                    )
                    can_dedup = (
                        bool(dedup_within_window)
                        and float(temperature) == 0.0
                        and len(active_indices) > 1
                    )
                    if can_dedup:
                        groups: dict[Any, tuple[int, str, str | dict[str, Any] | None, Mapping[str, Any] | None, list[int]]] = {}
                        for pos, idx in enumerate(active_indices):
                            p_str = str(raw_prompts[idx])
                            d_key = (act_keys[pos], p_str) if has_mixed_schemas else p_str
                            existing = groups.get(d_key)
                            if existing is None:
                                groups[d_key] = (idx, p_str, act_specs[pos], act_dicts[pos], [idx])
                            else:
                                existing[4].append(idx)
                        unique_items = list(groups.values())
                    else:
                        unique_items = [
                            (idx, str(raw_prompts[idx]), act_specs[pos], act_dicts[pos], [idx])
                            for pos, idx in enumerate(active_indices)
                        ]

                    unique_entries = [(it[0], it[1], it[4]) for it in unique_items]
                    entry_schemas = [it[2] for it in unique_items]
                    entry_schema_dicts = [it[3] for it in unique_items]
                    has_any_schema = any(s is not None for s in entry_schemas)
                    eff_thinking = _resolve_effective_thinking(
                        enable_thinking=bool(clean_extra.get("enable_thinking", False)),
                        has_schema=has_any_schema,
                        clean_extra=clean_extra,
                    )
                    use_roll = False
                    if engine_can_roll:
                        if use_rolling is True:
                            use_roll = True
                        elif str(use_rolling).strip().lower() == "auto":
                            use_roll = (
                                not eff_thinking
                                and (has_any_schema or len(unique_entries) > 4 or speculative is False)
                                and eff_mode != "hetero"
                            )

                    pre_encoded_ids: list[list[int]] | None = None
                    if use_roll:
                        pre_encoded_ids = [
                            list(ids)
                            for ids in engine.encode_prompts(
                                [e[1] for e in unique_entries],
                                system_prompt=system_prompt,
                                use_chat_template=eff_use_chat,
                                enable_thinking=eff_thinking,
                            )
                        ]
                    elif sort_by_length and len(unique_entries) > 1:
                        count_fn = getattr(engine, "count_tokens_batch", None)
                        tok_counts: list[int] | None = None
                        if callable(count_fn):
                            raw_counts = count_fn([entry[1] for entry in unique_entries])
                            if isinstance(raw_counts, Sequence) and len(raw_counts) == len(unique_entries):
                                tok_counts = [int(c) for c in raw_counts]
                        if tok_counts is not None:
                            order = sorted(
                                range(len(unique_entries)),
                                key=lambda i: (-tok_counts[i], unique_entries[i][0]),
                            )
                        else:
                            order = sorted(
                                range(len(unique_entries)),
                                key=lambda i: (-len(unique_entries[i][1].encode("utf-8")), unique_entries[i][0]),
                            )
                        unique_entries = [unique_entries[i] for i in order]
                        entry_schemas = [entry_schemas[i] for i in order]
                        entry_schema_dicts = [entry_schema_dicts[i] for i in order]

                    prepared["unique_entries"] = unique_entries
                    prepared["entry_schemas"] = entry_schemas
                    prepared["entry_schema_dicts"] = entry_schema_dicts
                    prepared["pre_encoded_ids"] = pre_encoded_ids
                    prepared["use_roll"] = use_roll
                    prepared["eff_thinking"] = eff_thinking
                    prepared["has_mixed_schemas"] = has_mixed_schemas

                win_row_offset += n_win
                if not _put_ready(prepared):
                    return
        except BaseException as exc:
            _put_ready(exc)
        finally:
            _put_ready(None)

    docs_processed = 0
    valid_json_tot = 0
    valid_schema_tot = 0
    prompt_toks_tot = 0
    comp_toks_tot = 0
    tot_dec_sec = 0.0
    retries_attempted_tot = 0
    retries_succeeded_tot = 0
    retry_reasons = _resolve_retry_reasons(retry_on)
    has_retry_hooks = (
        deferred_retries is not None
        or retry_hook is not None
        or async_retry_hook is not None
    )
    grammar_cache: dict[tuple[Any, ...], Any] = {}
    t0 = time.perf_counter()

    reader_thread = threading.Thread(target=_reader_worker, daemon=True)
    tok_thread = threading.Thread(target=_tok_worker, daemon=True)
    writer_thread = threading.Thread(target=_writer_worker, daemon=True)

    with ExitStack() as stack:
        _enter_mlx_limits(
            stack,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )
        _register_governor_shed_hooks(stack, governor, engine, grammar_cache)
        reader_thread.start()
        tok_thread.start()
        writer_thread.start()
        try:
            while True:
                if writer_err:
                    raise writer_err[0]
                item = ready_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item

                meta_tbl = item["meta_tbl"]
                raw_prompts = item["raw_prompts"]
                active_indices = item["active_indices"]
                n_win = len(raw_prompts)
                resp_texts: list[str] = [""] * n_win
                valid_jsons: list[bool] = [False] * n_win
                valid_schemas: list[bool] = [False] * n_win
                finish_reasons: list[str] = ["skipped_empty"] * n_win
                prompt_toks_col: list[int] = [0] * n_win
                comp_toks_col: list[int] = [0] * n_win
                risk_col: list[float] = [0.0] * n_win

                if active_indices:
                    unique_entries = item["unique_entries"]
                    entry_schemas = item["entry_schemas"]
                    entry_schema_dicts = item["entry_schema_dicts"]
                    pre_encoded_ids = item["pre_encoded_ids"]
                    use_roll = bool(item["use_roll"])
                    eff_thinking = bool(item["eff_thinking"])

                    eff_b_slots = eff_gpu_batch_size
                    clear_thresh_mb = 256
                    if governor is not None:
                        adaptation = governor.adapt(
                            base_b_slots=eff_gpu_batch_size,
                            base_cache_limit_mb=eff_cache_mb,
                            apply_mlx=True,
                        )
                        eff_b_slots = adaptation.b_slots
                        clear_thresh_mb = max(0, adaptation.cache_limit_mb)

                    if use_roll:
                        sorted_prompts = [entry[1] for entry in unique_entries]
                        res, order = _execute_rolling_batch(
                            engine,
                            sorted_prompts,
                            pre_encoded_ids=pre_encoded_ids,
                            system_prompt=system_prompt,
                            entry_schemas=entry_schemas,
                            grammar_cache=grammar_cache,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            whitespace_flexible=eff_ws,
                            max_whitespace_run=max_whitespace_run,
                            b_slots=eff_b_slots,
                            sort_by_length=sort_by_length,
                            clean_extra=clean_extra,
                            use_chat_template=eff_use_chat,
                            enable_thinking=eff_thinking,
                        )
                        sorted_entries = [unique_entries[i] for i in order]
                        sorted_schema_dicts = [entry_schema_dicts[i] for i in order]
                        sorted_schema_specs = [entry_schemas[i] for i in order]
                    else:
                        sorted_entries = unique_entries
                        sorted_schema_dicts = entry_schema_dicts
                        sorted_schema_specs = entry_schemas
                        sorted_prompts = [entry[1] for entry in sorted_entries]
                        base_gen_kw: dict[str, Any] = {
                            "system_prompt": system_prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "whitespace_flexible": eff_ws,
                            "batch_size": min(len(sorted_prompts), eff_b_slots),
                            **clean_extra,
                        }
                        if max_whitespace_run is not None and not _engine_accepts_explicit_kwarg(engine, "max_whitespace_run"):
                            base_gen_kw.pop("max_whitespace_run", None)
                        if mode is not None:
                            base_gen_kw["mode"] = mode
                        if speculative is not None:
                            base_gen_kw["speculative"] = speculative

                        res = _generate_batch_with_schema_fallback(
                            engine,
                            sorted_prompts,
                            entry_schemas=entry_schemas,
                            fallback_schema=eff_schema,
                            base_gen_kw=base_gen_kw,
                            accepts_per_doc=accepts_per_doc,
                            eff_b_slots=eff_b_slots,
                        )
                    clear_mlx_arena(clear_thresh_mb)

                    dedup_had_repeats = len(sorted_entries) < len(active_indices)
                    entry_states: list[dict[str, Any]] = []
                    for out_it, (_first_idx, p_str, orig_positions), item_schema_dict, item_schema_spec in zip(
                        res.items, sorted_entries, sorted_schema_dicts, sorted_schema_specs
                    ):
                        r_text = str(out_it.text)
                        v_json = bool(out_it.valid_json)
                        v_schema = bool(out_it.valid_schema)
                        f_reason = str(getattr(out_it, "finish_reason", "stop"))
                        p_toks = int(out_it.prompt_tokens)
                        c_toks = int(out_it.completion_tokens)
                        if emit_extractconf:
                            it_parsed = getattr(out_it, "parsed", None)
                            if it_parsed is None and v_json and r_text.strip():
                                try:
                                    it_parsed = json.loads(r_text)
                                except Exception:
                                    v_json = False
                            it_risk = _score_item_extractconf(
                                out_it,
                                parsed=it_parsed,
                                valid_json=v_json,
                                json_schema=item_schema_dict,
                                max_tokens=max_tokens,
                                calibrator=extractconf_calibrator,
                            )
                        else:
                            it_risk = 0.0

                        for orig_idx in orig_positions:
                            resp_texts[orig_idx] = r_text
                            valid_jsons[orig_idx] = v_json
                            valid_schemas[orig_idx] = v_schema
                            finish_reasons[orig_idx] = f_reason
                            prompt_toks_col[orig_idx] = p_toks
                            comp_toks_col[orig_idx] = c_toks
                            if emit_extractconf:
                                risk_col[orig_idx] = it_risk
                        entry_states.append(
                            {
                                "prompt": p_str,
                                "orig_positions": orig_positions,
                                "schema_spec": item_schema_spec,
                                "schema_dict": item_schema_dict,
                                "text": r_text,
                                "valid_json": v_json,
                                "valid_schema": v_schema,
                                "finish_reason": f_reason,
                                "risk": float(it_risk),
                            }
                        )

                    if dedup_had_repeats:
                        prompt_toks_tot += sum(prompt_toks_col[idx] for idx in active_indices)
                        comp_toks_tot += sum(comp_toks_col[idx] for idx in active_indices)
                    else:
                        prompt_toks_tot += int(
                            res.stats.get("prompt_tokens", sum(it.prompt_tokens for it in res.items))
                        )
                        comp_toks_tot += int(
                            res.stats.get("completion_tokens", sum(it.completion_tokens for it in res.items))
                        )
                    tot_dec_sec += float(res.stats.get("decode_sec", 0.0))

                    ran_inline_retry = False
                    if (defer_retries or (int(max_retries) <= 0 and has_retry_hooks)) and (
                        has_retry_hooks or int(max_retries) > 0
                    ):
                        deferred_reqs: list[ExtractRetryRequest] = []
                        for st in entry_states:
                            if _should_retry_row(
                                finish_reason=st["finish_reason"],
                                valid_json=st["valid_json"],
                                valid_schema=st["valid_schema"],
                                extractconf_risk=st["risk"],
                                retry_reasons=retry_reasons,
                                retry_risk_threshold=retry_risk_threshold,
                            ):
                                for orig_idx in st["orig_positions"]:
                                    deferred_reqs.append(
                                        ExtractRetryRequest(
                                            row_index=docs_processed + orig_idx,
                                            prompt=st["prompt"],
                                            system_prompt=system_prompt,
                                            json_schema=st["schema_spec"],
                                            json_schema_dict=st["schema_dict"],
                                            attempt=1,
                                            max_tokens=max_tokens,
                                            finish_reason=st["finish_reason"],
                                            valid_json=st["valid_json"],
                                            valid_schema=st["valid_schema"],
                                            extractconf_risk=st["risk"],
                                            response_text=st["text"],
                                        )
                                    )
                        if deferred_reqs:
                            _dispatch_retry_hooks(
                                deferred_reqs,
                                deferred_retries=deferred_retries,
                                retry_hook=retry_hook,
                                async_retry_hook=async_retry_hook,
                            )
                    elif int(max_retries) > 0:
                        ran_inline_retry = True
                        prev_budget = int(max_tokens)
                        for attempt in range(1, int(max_retries) + 1):
                            retry_state_indices: list[int] = []
                            round_reqs: list[ExtractRetryRequest] = []
                            attempt_budget = (
                                int(retry_max_tokens)
                                if retry_max_tokens is not None
                                else max(prev_budget * 2, 256)
                            )
                            prev_budget = attempt_budget
                            for s_idx, st in enumerate(entry_states):
                                if _should_retry_row(
                                    finish_reason=st["finish_reason"],
                                    valid_json=st["valid_json"],
                                    valid_schema=st["valid_schema"],
                                    extractconf_risk=st["risk"],
                                    retry_reasons=retry_reasons,
                                    retry_risk_threshold=retry_risk_threshold,
                                ):
                                    retry_state_indices.append(s_idx)
                                    round_reqs.append(
                                        ExtractRetryRequest(
                                            row_index=docs_processed + st["orig_positions"][0],
                                            prompt=st["prompt"],
                                            system_prompt=system_prompt,
                                            json_schema=st["schema_spec"],
                                            json_schema_dict=st["schema_dict"],
                                            attempt=attempt,
                                            max_tokens=attempt_budget,
                                            finish_reason=st["finish_reason"],
                                            valid_json=st["valid_json"],
                                            valid_schema=st["valid_schema"],
                                            extractconf_risk=st["risk"],
                                            response_text=st["text"],
                                        )
                                    )
                            if not round_reqs:
                                break
                            retry_results = run_deferred_retries(
                                engine,
                                round_reqs,
                                system_prompt=system_prompt,
                                retry_max_tokens=attempt_budget,
                                temperature=temperature if retry_temperature is None else retry_temperature,
                                whitespace_flexible=eff_ws,
                                max_whitespace_run=max_whitespace_run,
                                mode=mode,
                                speculative=speculative,
                                b_slots=eff_b_slots,
                                use_rolling=use_rolling,
                                emit_extractconf=emit_extractconf,
                                extractconf_calibrator=extractconf_calibrator,
                                retry_prompt_builder=retry_prompt_builder,
                                clear_cache_mb=clear_thresh_mb,
                                grammar_cache=grammar_cache,
                                **clean_extra,
                            )
                            for s_idx, cand in zip(retry_state_indices, retry_results):
                                st = entry_states[s_idx]
                                n_pos = len(st["orig_positions"])
                                retries_attempted_tot += n_pos
                                prompt_toks_tot += int(cand["prompt_tokens"]) * n_pos
                                comp_toks_tot += int(cand["completion_tokens"]) * n_pos
                                tot_dec_sec += float(cand["decode_sec"])
                                if cand["improved"]:
                                    retries_succeeded_tot += n_pos
                                    st["text"] = cand["text"]
                                    st["valid_json"] = bool(cand["valid_json"])
                                    st["valid_schema"] = bool(cand["valid_schema"])
                                    st["finish_reason"] = str(cand["finish_reason"])
                                    st["risk"] = float(cand["extractconf_risk"])
                                    for orig_idx in st["orig_positions"]:
                                        resp_texts[orig_idx] = st["text"]
                                        valid_jsons[orig_idx] = st["valid_json"]
                                        valid_schemas[orig_idx] = st["valid_schema"]
                                        finish_reasons[orig_idx] = st["finish_reason"]
                                        prompt_toks_col[orig_idx] = int(cand["prompt_tokens"])
                                        comp_toks_col[orig_idx] = int(cand["completion_tokens"])
                                        if emit_extractconf:
                                            risk_col[orig_idx] = st["risk"]

                    if dedup_had_repeats or ran_inline_retry:
                        valid_json_tot += sum(1 for idx in active_indices if valid_jsons[idx])
                        valid_schema_tot += sum(1 for idx in active_indices if valid_schemas[idx])
                    else:
                        valid_json_tot += int(
                            res.stats.get("valid_json_count", sum(1 for it in res.items if it.valid_json))
                        )
                        valid_schema_tot += int(
                            res.stats.get("valid_schema_count", sum(1 for it in res.items if it.valid_schema))
                        )

                out_table = (
                    meta_tbl.append_column("response_text", pa.array(resp_texts, type=pa.string()))
                    .append_column("valid_json", pa.array(valid_jsons, type=pa.bool_()))
                    .append_column("valid_schema", pa.array(valid_schemas, type=pa.bool_()))
                    .append_column("finish_reason", pa.array(finish_reasons, type=pa.string()))
                    .append_column("prompt_tokens", pa.array(prompt_toks_col, type=pa.int32()))
                    .append_column("completion_tokens", pa.array(comp_toks_col, type=pa.int32()))
                )
                if emit_extractconf:
                    out_table = out_table.append_column("extractconf_risk", pa.array(risk_col, type=pa.float32()))

                _enqueue_write(out_table)
                docs_processed += n_win

            _enqueue_write(None)
            writer_thread.join()
            if writer_err:
                raise writer_err[0]
        except BaseException:
            reader_stop.set()
            tok_stop.set()
            writer_abort.set()
            for q in (prep_queue, ready_queue):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            try:
                write_queue.put_nowait(None)
            except queue.Full:
                pass
            writer_thread.join(timeout=5.0)
            tok_thread.join(timeout=2.0)
            reader_thread.join(timeout=2.0)
            raise
        finally:
            reader_stop.set()
            tok_stop.set()
            for q in (prep_queue, ready_queue):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            tok_thread.join(timeout=2.0)
            reader_thread.join(timeout=2.0)
            writer_thread.join(timeout=2.0)
            clear_mlx_arena(256)

    elapsed = max(time.perf_counter() - t0, 1e-9)
    return ParquetExtractSummary(
        input_files=in_files,
        output_path=out,
        docs_processed=docs_processed,
        valid_json=valid_json_tot,
        valid_schema=valid_schema_tot,
        prompt_tokens=prompt_toks_tot,
        completion_tokens=comp_toks_tot,
        elapsed_sec=elapsed,
        docs_per_sec=docs_processed / elapsed,
        decode_tok_s=comp_toks_tot / max(tot_dec_sec, 1e-9),
        retries_attempted=retries_attempted_tot,
        retries_succeeded=retries_succeeded_tot,
    )


def extract_jsonl(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    system_prompt: str,
    json_schema: str | dict[str, Any] | None = None,
    compact_optional_schema: bool = False,
    text_key: str = "text",
    schema_key: str | None = None,
    per_doc_grammars: (
        Callable[[Mapping[str, Any]], str | dict[str, Any] | None]
        | Mapping[Any, str | dict[str, Any] | None]
        | Sequence[str | dict[str, Any] | None]
        | None
    ) = None,
    mode: str | None = None,
    speculative: bool | None = None,
    batch_size: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
    max_whitespace_run: int | None = None,
    response_text_key: str = "output_text",
    response_json_key: str = "parsed",
    emit_extractconf: bool = True,
    extractconf_calibrator: ExtractConfCalibrator | None = None,
    mlx_cache_limit_mb: int | None = DEFAULT_MLX_CACHE_LIMIT_MB,
    use_wired_limit: bool = True,
    use_rolling: bool | str = "auto",
    memory_governor: LiveMemoryGovernor | bool | None = True,
    uma_ceiling_fraction: float = WIRED_MEMORY_CEILING_FRACTION,
    max_retries: int = 0,
    retry_max_tokens: int | None = None,
    retry_on: Sequence[str] | str = DEFAULT_RETRY_REASONS,
    retry_risk_threshold: float | None = None,
    retry_temperature: float | None = None,
    retry_prompt_builder: Callable[[ExtractRetryRequest], str] | None = None,
    defer_retries: bool = False,
    deferred_retries: list[ExtractRetryRequest] | None = None,
    retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
    async_retry_hook: Callable[[ExtractRetryRequest], Any] | None = None,
    **extra_gen_kwargs: Any,
) -> dict[str, Any]:
    """Batch-extract structured JSON from a JSON or JSONL file into JSONL."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    raw_str = in_p.read_text(encoding="utf-8").strip()
    records: list[dict[str, Any]]
    if raw_str.startswith("["):
        loaded = json.loads(raw_str)
        records = [x if isinstance(x, dict) else {text_key: str(x)} for x in loaded]
    else:
        records = [json.loads(line) for line in raw_str.splitlines() if line.strip()]

    prompts = [str(r.get(text_key, "")) for r in records]
    eff_ws = resolve_engine_whitespace_flexible(engine, whitespace_flexible)
    schema_resolver = _SchemaResolver(
        engine,
        json_schema,
        compact_optional_schema=compact_optional_schema,
        emit_extractconf=emit_extractconf,
    )
    eff_schema = schema_resolver.default_spec
    is_callable_g = callable(per_doc_grammars)
    is_mapping_g = isinstance(per_doc_grammars, Mapping)
    is_seq_g = isinstance(per_doc_grammars, Sequence) and not isinstance(per_doc_grammars, (str, bytes))
    entry_schemas: list[str | dict[str, Any] | None] = []
    entry_schema_dicts: list[Mapping[str, Any] | None] = []
    entry_keys: list[str | None] = []
    for idx, rec in enumerate(records):
        raw_col = rec.get(schema_key) if schema_key is not None else None
        if is_callable_g:
            raw_spec = per_doc_grammars(rec)  # type: ignore[operator]
        elif is_mapping_g:
            if raw_col is not None and raw_col in per_doc_grammars:
                raw_spec = per_doc_grammars[raw_col]
            elif idx in per_doc_grammars:
                raw_spec = per_doc_grammars[idx]
            else:
                raw_spec = raw_col
        elif is_seq_g:
            raw_spec = per_doc_grammars[idx] if idx < len(per_doc_grammars) else raw_col
        else:
            raw_spec = raw_col
        es, ed, ek = schema_resolver.resolve(raw_spec)
        entry_schemas.append(es)
        entry_schema_dicts.append(ed)
        entry_keys.append(ek)

    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
    if max_whitespace_run is not None and _engine_accepts_explicit_kwarg(engine, "max_whitespace_run"):
        clean_extra.setdefault("max_whitespace_run", int(max_whitespace_run))
    if emit_extractconf and "collect_telemetry" not in clean_extra and _engine_accepts_explicit_kwarg(engine, "collect_telemetry"):
        clean_extra["collect_telemetry"] = True
    roll_kw = clean_extra.get("roll_slots")
    eff_bs = (
        int(batch_size)
        if batch_size is not None
        else (
            int(roll_kw)
            if isinstance(roll_kw, int) and not isinstance(roll_kw, bool) and int(roll_kw) > 0
            else _resolve_default_b_slots(engine)
        )
    )
    governor = _resolve_memory_governor(memory_governor, uma_ceiling_fraction, engine=engine)
    eff_cache_mb = DEFAULT_MLX_CACHE_LIMIT_MB if mlx_cache_limit_mb is None else int(mlx_cache_limit_mb)
    distinct_keys = set(entry_keys)
    has_mixed_schemas = len(distinct_keys) > 1 or (
        len(distinct_keys) == 1 and next(iter(distinct_keys)) != schema_resolver.default_key
    )
    has_any_schema = any(s is not None for s in entry_schemas)
    eff_mode = str(mode or getattr(engine, "default_mode", "auto") or "auto").lower()
    eff_thinking = _resolve_effective_thinking(
        enable_thinking=bool(clean_extra.get("enable_thinking", False)),
        has_schema=has_any_schema,
        clean_extra=clean_extra,
    )
    use_roll = False
    if _engine_supports_rolling(engine) and records:
        if use_rolling is True:
            use_roll = True
        elif str(use_rolling).strip().lower() == "auto":
            use_roll = (
                not eff_thinking
                and (has_any_schema or len(records) > 4 or speculative is False)
                and eff_mode != "hetero"
            )

    grammar_cache: dict[tuple[Any, ...], Any] = {}
    with ExitStack() as stack:
        _enter_mlx_limits(
            stack,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )
        _register_governor_shed_hooks(stack, governor, engine, grammar_cache)
        eff_b_slots = eff_bs
        clear_thresh_mb = 256
        if governor is not None:
            adaptation = governor.adapt(
                base_b_slots=eff_bs,
                base_cache_limit_mb=eff_cache_mb,
                apply_mlx=True,
            )
            eff_b_slots = adaptation.b_slots
            clear_thresh_mb = max(0, adaptation.cache_limit_mb)

        if use_roll:
            res_roll, order_roll = _execute_rolling_batch(
                engine,
                prompts,
                pre_encoded_ids=None,
                system_prompt=system_prompt,
                entry_schemas=entry_schemas,
                grammar_cache=grammar_cache,
                max_tokens=max_tokens,
                temperature=temperature,
                whitespace_flexible=eff_ws,
                max_whitespace_run=max_whitespace_run,
                b_slots=eff_b_slots,
                sort_by_length=True,
                clean_extra=clean_extra,
                use_chat_template=bool(clean_extra.get("use_chat_template", True)),
                enable_thinking=eff_thinking,
            )
            restored_items: list[Any] = [None] * len(records)
            for pos_r, orig_r in enumerate(order_roll):
                restored_items[orig_r] = res_roll.items[pos_r]
            res = SimpleNamespace(items=restored_items, stats=res_roll.stats)
        else:
            gen_kw: dict[str, Any] = {
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "whitespace_flexible": eff_ws,
                "batch_size": eff_b_slots,
                **clean_extra,
            }
            if mode is not None:
                gen_kw["mode"] = mode
            if speculative is not None:
                gen_kw["speculative"] = speculative
            res = _generate_batch_with_schema_fallback(
                engine,
                prompts,
                entry_schemas=entry_schemas,
                fallback_schema=eff_schema,
                base_gen_kw=gen_kw,
                accepts_per_doc=_engine_accepts_explicit_kwarg(engine, "per_doc_grammars"),
                eff_b_slots=eff_b_slots,
            )
        clear_mlx_arena(clear_thresh_mb)

    out_stats = dict(getattr(res, "stats", {}) or {})
    retry_reasons = _resolve_retry_reasons(retry_on)
    has_retry_hooks = (
        deferred_retries is not None
        or retry_hook is not None
        or async_retry_hook is not None
    )
    row_states: list[dict[str, Any]] = []
    for item, item_schema_dict in zip(res.items, entry_schema_dicts):
        r_text = str(item.text)
        v_json = bool(item.valid_json)
        v_schema = bool(item.valid_schema)
        f_reason = str(getattr(item, "finish_reason", "stop"))
        p_toks = int(item.prompt_tokens)
        c_toks = int(item.completion_tokens)
        it_parsed = getattr(item, "parsed", None)
        if it_parsed is None and v_json and r_text.strip():
            try:
                it_parsed = json.loads(r_text)
            except Exception:
                v_json = False
        it_risk = (
            _score_item_extractconf(
                item,
                parsed=it_parsed,
                valid_json=v_json,
                json_schema=item_schema_dict,
                max_tokens=max_tokens,
                calibrator=extractconf_calibrator,
            )
            if emit_extractconf
            else 0.0
        )
        row_states.append(
            {
                "text": r_text,
                "parsed": it_parsed,
                "valid_json": v_json,
                "valid_schema": v_schema,
                "finish_reason": f_reason,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "risk": float(it_risk),
            }
        )

    retries_attempted = 0
    retries_succeeded = 0
    if (defer_retries and has_retry_hooks) or (int(max_retries) <= 0 and has_retry_hooks):
        deferred_reqs = [
            ExtractRetryRequest(
                row_index=idx_r,
                prompt=prompts[idx_r],
                system_prompt=system_prompt,
                json_schema=entry_schemas[idx_r],
                json_schema_dict=entry_schema_dicts[idx_r],
                attempt=1,
                max_tokens=max_tokens,
                finish_reason=st["finish_reason"],
                valid_json=st["valid_json"],
                valid_schema=st["valid_schema"],
                extractconf_risk=st["risk"],
                response_text=st["text"],
            )
            for idx_r, st in enumerate(row_states)
            if _should_retry_row(
                finish_reason=st["finish_reason"],
                valid_json=st["valid_json"],
                valid_schema=st["valid_schema"],
                extractconf_risk=st["risk"],
                retry_reasons=retry_reasons,
                retry_risk_threshold=retry_risk_threshold,
            )
        ]
        if deferred_reqs:
            _dispatch_retry_hooks(
                deferred_reqs,
                deferred_retries=deferred_retries,
                retry_hook=retry_hook,
                async_retry_hook=async_retry_hook,
            )
    elif int(max_retries) > 0:
        prev_budget = int(max_tokens)
        g_cache_retry: dict[tuple[Any, ...], Any] = {}
        for attempt in range(1, int(max_retries) + 1):
            attempt_budget = (
                int(retry_max_tokens)
                if retry_max_tokens is not None
                else max(prev_budget * 2, 256)
            )
            prev_budget = attempt_budget
            round_reqs = [
                ExtractRetryRequest(
                    row_index=idx_r,
                    prompt=prompts[idx_r],
                    system_prompt=system_prompt,
                    json_schema=entry_schemas[idx_r],
                    json_schema_dict=entry_schema_dicts[idx_r],
                    attempt=attempt,
                    max_tokens=attempt_budget,
                    finish_reason=st["finish_reason"],
                    valid_json=st["valid_json"],
                    valid_schema=st["valid_schema"],
                    extractconf_risk=st["risk"],
                    response_text=st["text"],
                )
                for idx_r, st in enumerate(row_states)
                if _should_retry_row(
                    finish_reason=st["finish_reason"],
                    valid_json=st["valid_json"],
                    valid_schema=st["valid_schema"],
                    extractconf_risk=st["risk"],
                    retry_reasons=retry_reasons,
                    retry_risk_threshold=retry_risk_threshold,
                )
            ]
            if not round_reqs:
                break
            retry_results = run_deferred_retries(
                engine,
                round_reqs,
                system_prompt=system_prompt,
                retry_max_tokens=attempt_budget,
                temperature=temperature if retry_temperature is None else retry_temperature,
                whitespace_flexible=eff_ws,
                max_whitespace_run=max_whitespace_run,
                mode=mode,
                speculative=speculative,
                b_slots=eff_b_slots,
                use_rolling=use_rolling,
                emit_extractconf=emit_extractconf,
                extractconf_calibrator=extractconf_calibrator,
                retry_prompt_builder=retry_prompt_builder,
                clear_cache_mb=clear_thresh_mb,
                grammar_cache=g_cache_retry,
                **clean_extra,
            )
            for cand in retry_results:
                idx_r = int(cand["row_index"])
                retries_attempted += 1
                out_stats["prompt_tokens"] = int(out_stats.get("prompt_tokens", 0)) + int(cand["prompt_tokens"])
                out_stats["completion_tokens"] = int(out_stats.get("completion_tokens", 0)) + int(cand["completion_tokens"])
                out_stats["decode_sec"] = float(out_stats.get("decode_sec", 0.0)) + float(cand["decode_sec"])
                if cand["improved"]:
                    retries_succeeded += 1
                    row_states[idx_r] = {
                        "text": cand["text"],
                        "parsed": cand["parsed"],
                        "valid_json": bool(cand["valid_json"]),
                        "valid_schema": bool(cand["valid_schema"]),
                        "finish_reason": str(cand["finish_reason"]),
                        "prompt_tokens": int(cand["prompt_tokens"]),
                        "completion_tokens": int(cand["completion_tokens"]),
                        "risk": float(cand["extractconf_risk"]),
                    }
        out_stats["valid_json_count"] = sum(1 for st in row_states if st["valid_json"])
        out_stats["valid_schema_count"] = sum(1 for st in row_states if st["valid_schema"])
        out_stats["retries_attempted"] = retries_attempted
        out_stats["retries_succeeded"] = retries_succeeded

    tmp_out = out_p.with_name(f".{out_p.name}.tmp.{os.getpid()}")
    try:
        with tmp_out.open("w", encoding="utf-8") as fp:
            for rec, st in zip(records, row_states):
                out_row: dict[str, Any] = {
                    **rec,
                    response_text_key: st["text"],
                    response_json_key: st["parsed"],
                    "valid_json": st["valid_json"],
                    "valid_schema": st["valid_schema"],
                    "finish_reason": st["finish_reason"],
                    "prompt_tokens": st["prompt_tokens"],
                    "completion_tokens": st["completion_tokens"],
                }
                if emit_extractconf:
                    out_row["extractconf_risk"] = round(float(st["risk"]), 6)
                fp.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        os.replace(tmp_out, out_p)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    return out_stats


__all__ = [
    "DEFAULT_RETRY_REASONS",
    "DEFAULT_TEXT_COLUMNS",
    "ExtractRetryRequest",
    "ParquetExtractSummary",
    "extract_jsonl",
    "extract_parquet",
    "next_shard_index",
    "prune_optional_nullable_fields",
    "resolve_engine_whitespace_flexible",
    "resolve_input_files",
    "resolve_llm_text_column",
    "resolve_text_column",
    "run_deferred_retries",
    "run_deferred_retries_async",
    "slice_first_pages_utf8",
    "slice_ocr_pages_utf8",
]
