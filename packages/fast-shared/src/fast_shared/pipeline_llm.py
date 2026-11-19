"""Streaming and resumable Parquet + JSONL structured extraction pipeline for generative engines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .parquet_pipeline import (
    DEFAULT_TEXT_COLUMNS,
    resolve_input_files,
    resolve_text_column,
    slice_ocr_pages_utf8,
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


def extract_parquet(
    engine: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    system_prompt: str,
    json_schema: str | dict[str, Any] | None = None,
    text_col: str | None = None,
    keep_cols: list[str] | None = None,
    drop_text_col: bool = False,
    mode: str | None = None,
    speculative: bool | None = None,
    batch_rows: int | None = None,
    gpu_batch_size: int | None = None,
    shard_rows: int = 4096,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool = False,
    max_chars: int = 8000,
    max_pages: int = 0,
    filter_ocr: bool = False,
    ocr_pages_only: bool = False,
    skip_empty: bool = True,
    resume: bool = False,
    **extra_gen_kwargs: Any,
) -> ParquetExtractSummary:
    """Stream-extract structured JSON from Parquet input(s) into a single file or sharded directory."""
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
                    ok_vs = True if vs_s is None else bool(vs_s[i_s])
                    ok_fr = True if fr_s is None else (fr_s[i_s] is None or fr_s[i_s] == "stop")
                    status_by_id[key_val] = bool(ok_vs and ok_fr)
        done_ids = {k for k, ok in status_by_id.items() if ok}

    writer: pq.ParquetWriter | None = None
    tmp_dst: Path | None = None
    final_dst: Path | None = None
    shard_idx = next_shard_index(out) if sharded else 0
    shard_acc = 0

    docs_processed = 0
    valid_json_tot = 0
    valid_schema_tot = 0
    prompt_toks_tot = 0
    comp_toks_tot = 0
    tot_dec_sec = 0.0
    t0 = time.perf_counter()

    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
    roll_kw = clean_extra.get("roll_slots")
    bs_kw = clean_extra.pop("batch_size", None)
    roll_default = int(roll_kw) if isinstance(roll_kw, int) and not isinstance(roll_kw, bool) and int(roll_kw) > 0 else 128
    bs_extra = int(bs_kw) if isinstance(bs_kw, int) and not isinstance(bs_kw, bool) and int(bs_kw) > 0 else None
    eff_batch_rows = int(batch_rows) if batch_rows is not None else max(roll_default, bs_extra or 0)
    eff_gpu_batch_size = (
        int(gpu_batch_size)
        if gpu_batch_size is not None
        else (bs_extra if bs_extra is not None else max(eff_batch_rows, roll_default))
    )

    try:
        for fpath in in_files:
            pf = pq.ParquetFile(fpath)
            schema = pf.schema_arrow
            tcol = resolve_llm_text_column(schema, text_col)
            avail = set(schema.names)
            meta_cols = [c for c in keep_cols if c in avail] if keep_cols is not None else [
                c for c in schema.names if not (drop_text_col and c == tcol)
            ]
            read_cols = list(dict.fromkeys(meta_cols + [tcol]))
            for extra in ("is_ocr", "ocr_pages", "page_offsets"):
                if extra in avail and extra not in read_cols:
                    read_cols.append(extra)

            for rb in pf.iter_batches(batch_size=eff_batch_rows, columns=read_cols):
                if filter_ocr:
                    mask_flags = [True] * rb.num_rows
                    has_iso = "is_ocr" in rb.schema.names
                    has_op = "ocr_pages" in rb.schema.names
                    iso_list = rb.column("is_ocr").to_pylist() if has_iso else None
                    op_list = rb.column("ocr_pages").to_pylist() if has_op else None
                    for i in range(rb.num_rows):
                        v_iso = bool(iso_list[i]) if iso_list is not None and iso_list[i] is not None else False
                        v_op = len(op_list[i]) > 0 if op_list is not None and op_list[i] is not None else False
                        if has_iso or has_op:
                            mask_flags[i] = v_iso or v_op
                    rb = rb.filter(pa.array(mask_flags, type=pa.bool_()))
                    if rb.num_rows == 0:
                        continue

                if done_ids:
                    kcol = resume_key_col if (resume_key_col and resume_key_col in rb.schema.names) else (
                        "id" if "id" in rb.schema.names else ("body_sha" if "body_sha" in rb.schema.names else None)
                    )
                    if kcol is not None:
                        ids_col = rb.column(kcol).to_pylist()
                        keep_flags = [x not in done_ids for x in ids_col]
                        rb = rb.filter(pa.array(keep_flags, type=pa.bool_()))
                        if rb.num_rows == 0:
                            continue

                texts_raw = rb.column(tcol).to_pylist()
                offsets_col = rb.column("page_offsets").to_pylist() if "page_offsets" in rb.schema.names else None
                ocr_pages_col = rb.column("ocr_pages").to_pylist() if "ocr_pages" in rb.schema.names else None

                kept_indices: list[int] = []
                prompts: list[str] = []
                for i in range(rb.num_rows):
                    t = texts_raw[i] if isinstance(texts_raw[i], str) else ""
                    if ocr_pages_only and offsets_col is not None and ocr_pages_col is not None:
                        t = slice_ocr_pages_utf8(t, offsets_col[i], ocr_pages_col[i])
                    elif max_pages > 0 and offsets_col is not None:
                        t = slice_first_pages_utf8(t, offsets_col[i], max_pages)
                    if max_chars > 0 and len(t) > max_chars:
                        t = t[:max_chars]
                    if skip_empty and not t.strip():
                        continue
                    kept_indices.append(i)
                    prompts.append(t)

                if not prompts:
                    continue

                gen_kw: dict[str, Any] = {
                    "system_prompt": system_prompt,
                    "json_schema": json_schema,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "whitespace_flexible": whitespace_flexible,
                    "batch_size": min(len(prompts), eff_gpu_batch_size),
                    **clean_extra,
                }
                if mode is not None:
                    gen_kw["mode"] = mode
                if speculative is not None:
                    gen_kw["speculative"] = speculative

                res = engine.generate_batch(prompts, **gen_kw)

                meta_only_cols = [c for c in meta_cols if c != "page_offsets"]
                out_table = pa.Table.from_batches([rb.take(kept_indices)]).select(meta_only_cols)
                out_table = (
                    out_table.append_column("response_text", pa.array([it.text for it in res.items], type=pa.string()))
                    .append_column("valid_json", pa.array([it.valid_json for it in res.items], type=pa.bool_()))
                    .append_column("valid_schema", pa.array([it.valid_schema for it in res.items], type=pa.bool_()))
                    .append_column(
                        "finish_reason",
                        pa.array([str(getattr(it, "finish_reason", "stop")) for it in res.items], type=pa.string()),
                    )
                    .append_column("prompt_tokens", pa.array([it.prompt_tokens for it in res.items], type=pa.int32()))
                    .append_column("completion_tokens", pa.array([it.completion_tokens for it in res.items], type=pa.int32()))
                )

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

                docs_processed += len(res.items)
                valid_json_tot += int(res.stats.get("valid_json_count", 0))
                valid_schema_tot += int(res.stats.get("valid_schema_count", 0))
                prompt_toks_tot += int(res.stats.get("prompt_tokens", 0))
                comp_toks_tot += int(res.stats.get("completion_tokens", 0))
                tot_dec_sec += float(res.stats.get("decode_sec", 0.0))
        if writer is not None:
            writer.close()
            writer = None
            assert tmp_dst is not None and final_dst is not None
            os.replace(tmp_dst, final_dst)
            tmp_dst = None
            final_dst = None
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if tmp_dst is not None:
            tmp_dst.unlink(missing_ok=True)
        raise

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
    text_key: str = "text",
    mode: str | None = None,
    speculative: bool | None = None,
    batch_size: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    whitespace_flexible: bool = False,
    response_text_key: str = "output_text",
    response_json_key: str = "parsed",
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
    clean_extra = {k: v for k, v in extra_gen_kwargs.items() if v is not None}
    roll_kw = clean_extra.get("roll_slots")
    eff_bs = (
        int(batch_size)
        if batch_size is not None
        else (int(roll_kw) if isinstance(roll_kw, int) and not isinstance(roll_kw, bool) and int(roll_kw) > 0 else 128)
    )
    gen_kw: dict[str, Any] = {
        "system_prompt": system_prompt,
        "json_schema": json_schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "whitespace_flexible": whitespace_flexible,
        "batch_size": eff_bs,
        **clean_extra,
    }
    if mode is not None:
        gen_kw["mode"] = mode
    if speculative is not None:
        gen_kw["speculative"] = speculative

    res = engine.generate_batch(prompts, **gen_kw)
    tmp_out = out_p.with_name(f".{out_p.name}.tmp.{os.getpid()}")
    try:
        with tmp_out.open("w", encoding="utf-8") as fp:
            for rec, item in zip(records, res.items):
                out_row = {
                    **rec,
                    response_text_key: item.text,
                    response_json_key: item.parsed,
                    "valid_json": item.valid_json,
                    "valid_schema": item.valid_schema,
                    "finish_reason": str(getattr(item, "finish_reason", "stop")),
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                }
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
    "resolve_input_files",
    "resolve_llm_text_column",
    "resolve_text_column",
    "slice_first_pages_utf8",
    "slice_ocr_pages_utf8",
]
