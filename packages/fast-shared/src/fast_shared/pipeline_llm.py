"""Streaming and resumable Parquet + JSONL structured extraction pipeline for generative engines."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
import copy
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

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
from .runtime_lessons import DEFAULT_MLX_CACHE_LIMIT_MB, DynamicSiliconPolicy


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
    eff_schema = _resolve_effective_schema(engine, json_schema, compact_optional_schema)
    eff_schema_dict: Mapping[str, Any] | None = (
        parse_json_schema_dict(eff_schema) if emit_extractconf else None
    )
    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
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
    eff_window_rows = (
        int(batch_rows)
        if batch_rows is not None
        else max(int(macro_window_rows), eff_gpu_batch_size)
    )
    if eff_window_rows < 1:
        raise ValueError(f"Macro-window / batch_rows size must be >= 1, got {eff_window_rows}")

    prep_queue: queue.Queue[tuple[pa.Table, list[str | None]] | None | BaseException] = queue.Queue(maxsize=2)
    write_queue: queue.Queue[pa.Table | None] = queue.Queue(maxsize=2)
    reader_stop = threading.Event()
    writer_abort = threading.Event()
    writer_err: list[BaseException] = []

    def _put_prep(item: tuple[pa.Table, list[str | None]] | None | BaseException) -> bool:
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
            acc_rows = 0

            def _flush_window() -> bool:
                nonlocal acc_tables, acc_prompts, acc_rows
                if not acc_tables:
                    return True
                merged = acc_tables[0] if len(acc_tables) == 1 else pa.concat_tables(acc_tables)
                prompts_out = acc_prompts
                acc_tables = []
                acc_prompts = []
                acc_rows = 0
                return _put_prep((merged, prompts_out))

            for fpath in in_files:
                if reader_stop.is_set():
                    return
                pf = pq.ParquetFile(fpath)
                schema = pf.schema_arrow
                tcol = resolve_llm_text_column(schema, text_col)
                avail = set(schema.names)
                meta_cols = (
                    [c for c in keep_cols if c in avail]
                    if keep_cols is not None
                    else [c for c in schema.names if not (drop_text_col and c == tcol)]
                )
                meta_only_cols = [c for c in meta_cols if c != "page_offsets"]
                read_cols = list(dict.fromkeys(meta_cols + [tcol]))
                for extra in ("is_ocr", "ocr_pages", "page_offsets"):
                    if extra in avail and extra not in read_cols:
                        read_cols.append(extra)

                for rb in _iter_filtered_batches(
                    pf,
                    read_cols=read_cols,
                    batch_size=eff_window_rows,
                    done_ids=done_ids,
                    resume_key_col=resume_key_col,
                    filter_ocr=filter_ocr,
                ):
                    if reader_stop.is_set():
                        return

                    texts_raw = rb.column(tcol).to_pylist()
                    offsets_col = rb.column("page_offsets").to_pylist() if "page_offsets" in rb.schema.names else None
                    ocr_pages_col = rb.column("ocr_pages").to_pylist() if "ocr_pages" in rb.schema.names else None

                    kept_indices: list[int] = []
                    prompts: list[str | None] = []
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
                        else:
                            kept_indices.append(i)
                            prompts.append(t)

                    if not kept_indices:
                        continue

                    meta_tbl = pa.Table.from_batches([rb.take(kept_indices)]).select(meta_only_cols)
                    # Emit bounded macro-windows of up to `eff_window_rows`
                    cursor = 0
                    total_rb = meta_tbl.num_rows
                    while cursor < total_rb:
                        space = eff_window_rows - acc_rows
                        take_n = min(space, total_rb - cursor)
                        acc_tables.append(meta_tbl.slice(cursor, take_n))
                        acc_prompts.extend(prompts[cursor : cursor + take_n])
                        acc_rows += take_n
                        cursor += take_n
                        if acc_rows >= eff_window_rows:
                            if not _flush_window():
                                return

            if acc_rows > 0:
                _flush_window()
        except BaseException as exc:
            try:
                prep_queue.put(exc, timeout=1.0)
            except queue.Full:
                pass
        finally:
            try:
                prep_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

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

    docs_processed = 0
    valid_json_tot = 0
    valid_schema_tot = 0
    prompt_toks_tot = 0
    comp_toks_tot = 0
    tot_dec_sec = 0.0
    t0 = time.perf_counter()

    reader_thread = threading.Thread(target=_reader_worker, daemon=True)
    writer_thread = threading.Thread(target=_writer_worker, daemon=True)

    with ExitStack() as stack:
        _enter_mlx_limits(
            stack,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )
        reader_thread.start()
        writer_thread.start()
        try:
            while True:
                if writer_err:
                    raise writer_err[0]
                item = prep_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item

                meta_tbl, raw_prompts = item
                n_win = len(raw_prompts)
                resp_texts: list[str] = [""] * n_win
                valid_jsons: list[bool] = [False] * n_win
                valid_schemas: list[bool] = [False] * n_win
                finish_reasons: list[str] = ["skipped_empty"] * n_win
                prompt_toks_col: list[int] = [0] * n_win
                comp_toks_col: list[int] = [0] * n_win
                risk_col: list[float] = [0.0] * n_win

                active_indices = [idx for idx, p in enumerate(raw_prompts) if p is not None]
                if active_indices:
                    can_dedup = (
                        bool(dedup_within_window)
                        and float(temperature) == 0.0
                        and len(active_indices) > 1
                    )
                    if can_dedup:
                        groups_by_prompt: dict[str, list[int]] = {}
                        for idx in active_indices:
                            p_str = str(raw_prompts[idx])
                            pos_list = groups_by_prompt.get(p_str)
                            if pos_list is None:
                                groups_by_prompt[p_str] = [idx]
                            else:
                                pos_list.append(idx)
                        unique_entries: list[tuple[int, str, list[int]]] = [
                            (pos_list[0], p_str, pos_list)
                            for p_str, pos_list in groups_by_prompt.items()
                        ]
                    else:
                        unique_entries = [(idx, str(raw_prompts[idx]), [idx]) for idx in active_indices]

                    if sort_by_length and len(unique_entries) > 1:
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
                        sorted_entries = [unique_entries[i] for i in order]
                    else:
                        sorted_entries = unique_entries
                    sorted_prompts = [entry[1] for entry in sorted_entries]

                    gen_kw: dict[str, Any] = {
                        "system_prompt": system_prompt,
                        "json_schema": eff_schema,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "whitespace_flexible": eff_ws,
                        "batch_size": min(len(sorted_prompts), eff_gpu_batch_size),
                        **clean_extra,
                    }
                    if mode is not None:
                        gen_kw["mode"] = mode
                    if speculative is not None:
                        gen_kw["speculative"] = speculative

                    res = engine.generate_batch(sorted_prompts, **gen_kw)
                    clear_mlx_arena(256)

                    dedup_had_repeats = len(sorted_entries) < len(active_indices)
                    for out_it, (_first_idx, _p_str, orig_positions) in zip(res.items, sorted_entries):
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
                                json_schema=eff_schema_dict,
                                max_tokens=max_tokens,
                                calibrator=extractconf_calibrator,
                            )
                        else:
                            it_risk = 0.0

                        for orig_idx in orig_positions:
                            resp_texts[orig_idx] = r_text
                            valid_jsons[orig_idx] = bool(out_it.valid_json)
                            valid_schemas[orig_idx] = v_schema
                            finish_reasons[orig_idx] = f_reason
                            prompt_toks_col[orig_idx] = p_toks
                            comp_toks_col[orig_idx] = c_toks
                            if emit_extractconf:
                                risk_col[orig_idx] = it_risk

                    if dedup_had_repeats:
                        valid_json_tot += sum(1 for idx in active_indices if valid_jsons[idx])
                        valid_schema_tot += sum(1 for idx in active_indices if valid_schemas[idx])
                        prompt_toks_tot += sum(prompt_toks_col[idx] for idx in active_indices)
                        comp_toks_tot += sum(comp_toks_col[idx] for idx in active_indices)
                    else:
                        valid_json_tot += int(
                            res.stats.get("valid_json_count", sum(1 for it in res.items if it.valid_json))
                        )
                        valid_schema_tot += int(
                            res.stats.get("valid_schema_count", sum(1 for it in res.items if it.valid_schema))
                        )
                        prompt_toks_tot += int(
                            res.stats.get("prompt_tokens", sum(it.prompt_tokens for it in res.items))
                        )
                        comp_toks_tot += int(
                            res.stats.get("completion_tokens", sum(it.completion_tokens for it in res.items))
                        )
                    tot_dec_sec += float(res.stats.get("decode_sec", 0.0))

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
            writer_abort.set()
            try:
                write_queue.put_nowait(None)
            except queue.Full:
                pass
            writer_thread.join(timeout=5.0)
            reader_thread.join(timeout=2.0)
            raise
        finally:
            reader_stop.set()
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
    mode: str | None = None,
    speculative: bool | None = None,
    batch_size: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool | None = None,
    response_text_key: str = "output_text",
    response_json_key: str = "parsed",
    emit_extractconf: bool = True,
    extractconf_calibrator: ExtractConfCalibrator | None = None,
    mlx_cache_limit_mb: int | None = DEFAULT_MLX_CACHE_LIMIT_MB,
    use_wired_limit: bool = True,
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
    eff_schema = _resolve_effective_schema(engine, json_schema, compact_optional_schema)
    eff_schema_dict: Mapping[str, Any] | None = (
        parse_json_schema_dict(eff_schema) if emit_extractconf else None
    )
    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
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
    gen_kw: dict[str, Any] = {
        "system_prompt": system_prompt,
        "json_schema": eff_schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "whitespace_flexible": eff_ws,
        "batch_size": eff_bs,
        **clean_extra,
    }
    if mode is not None:
        gen_kw["mode"] = mode
    if speculative is not None:
        gen_kw["speculative"] = speculative

    with ExitStack() as stack:
        _enter_mlx_limits(
            stack,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )
        res = engine.generate_batch(prompts, **gen_kw)
        clear_mlx_arena(256)

    tmp_out = out_p.with_name(f".{out_p.name}.tmp.{os.getpid()}")
    try:
        with tmp_out.open("w", encoding="utf-8") as fp:
            for rec, item in zip(records, res.items):
                out_row: dict[str, Any] = {
                    **rec,
                    response_text_key: item.text,
                    response_json_key: getattr(item, "parsed", None),
                    "valid_json": item.valid_json,
                    "valid_schema": item.valid_schema,
                    "finish_reason": str(getattr(item, "finish_reason", "stop")),
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                }
                if emit_extractconf:
                    out_row["extractconf_risk"] = round(
                        _score_item_extractconf(
                            item,
                            json_schema=eff_schema_dict,
                            max_tokens=max_tokens,
                            calibrator=extractconf_calibrator,
                        ),
                        6,
                    )
                fp.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        os.replace(tmp_out, out_p)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    return res.stats


__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "ParquetExtractSummary",
    "extract_jsonl",
    "extract_parquet",
    "next_shard_index",
    "prune_optional_nullable_fields",
    "resolve_engine_whitespace_flexible",
    "resolve_input_files",
    "resolve_llm_text_column",
    "resolve_text_column",
    "slice_first_pages_utf8",
    "slice_ocr_pages_utf8",
]
