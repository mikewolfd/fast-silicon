"""CLI entrypoints (`gpt-oss setup-weights|generate|extract|parquet|serve|verify|bench`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from fast_shared.hf import copy_tokenizer_files

from .engine import FastGptOssEngine
from .paths import DEFAULT_HF_REPO, resolve_paths
from .pipeline import extract_jsonl, extract_parquet
from .server import FastGptOssServer
from .tokenizer import DEFAULT_KEPT_VOCAB_SIZE, O200kVocabRemapper


def _make_engine(args: argparse.Namespace) -> FastGptOssEngine:
    vocab_out_arg = getattr(args, "vocab_out", DEFAULT_KEPT_VOCAB_SIZE)
    vocab_out = int(vocab_out_arg) if vocab_out_arg and int(vocab_out_arg) > 0 else None
    return FastGptOssEngine(
        home=args.home,
        model_dir=getattr(args, "model_dir", None),
        vocab_out=vocab_out,
        default_reasoning_effort=getattr(args, "reasoning_effort", None) or "low",
        default_mode=getattr(args, "mode", "auto"),
        local_files_only=bool(getattr(args, "local_files_only", True)),
        warmup=not bool(getattr(args, "no_warmup", False)),
    )


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    model_dir = paths.resolve_model_dir(
        args.model_dir,
        repo_id=args.repo_id,
        local_files_only=args.local_files_only,
    )
    copy_tokenizer_files(model_dir, paths.tokenizer_dir)
    tok_json = paths.resolve_tokenizer_json(model_dir)
    target_v = int(args.vocab_out) if args.vocab_out and int(args.vocab_out) > 0 else DEFAULT_KEPT_VOCAB_SIZE
    remap = O200kVocabRemapper.load_or_build(
        tok_json,
        cache_npz_path=paths.vocab_remap_npz,
        target_size=target_v,
    )
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "tokenizer_dir": str(paths.tokenizer_dir),
                "vocab_remap_npz": str(paths.vocab_remap_npz),
                "kept_vocab_size": remap.vocab_size,
            },
            indent=2,
        )
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if getattr(args, "system_file", None) else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        prompt: object = json.loads(Path(args.messages_json).read_text(encoding="utf-8")) if args.messages_json else args.prompt
        item = eng.generate(
            prompt,  # type: ignore[arg-type]
            system_prompt=sys_txt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            mode=args.mode,
            whitespace_flexible=(not args.compact_json),
            enable_thinking=bool(args.enable_thinking),
            reasoning_effort=args.reasoning_effort,
            reasoning_budget=args.reasoning_budget,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "text": item.text,
                        "reasoning": item.reasoning,
                        "parsed": item.parsed,
                        "prompt_tokens": item.prompt_tokens,
                        "completion_tokens": item.completion_tokens,
                        "valid_json": item.valid_json,
                        "valid_schema": item.valid_schema,
                        "finish_reason": item.finish_reason,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            if item.reasoning and args.show_reasoning:
                print(f"[reasoning]\n{item.reasoning}\n[/reasoning]")
            print(item.text)
    finally:
        eng.close()
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        stats = extract_jsonl(
            eng,
            args.input,
            args.output,
            system_prompt=sys_txt,
            json_schema=schema_str,
            text_key=args.text_key,
            mode=args.mode,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            whitespace_flexible=args.whitespace_flexible,
            reasoning_effort=args.reasoning_effort,
            reasoning_budget=args.reasoning_budget,
        )
        print(json.dumps(stats, indent=2))
    finally:
        eng.close()
    return 0


def _cmd_parquet(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        summary = extract_parquet(
            eng,
            args.input,
            args.output,
            system_prompt=sys_txt,
            json_schema=schema_str,
            text_col=args.text_col,
            keep_cols=[c.strip() for c in args.keep_cols.split(",")] if args.keep_cols else None,
            drop_text_col=args.drop_text_col,
            mode=args.mode,
            batch_rows=args.batch_rows,
            shard_rows=args.shard_rows,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            whitespace_flexible=args.whitespace_flexible,
            max_chars=args.max_chars,
            max_pages=args.max_pages,
            filter_ocr=args.filter_ocr,
            ocr_pages_only=args.ocr_pages_only,
            resume=args.resume,
            reasoning_effort=args.reasoning_effort,
            reasoning_budget=args.reasoning_budget,
        )
        print(
            json.dumps(
                {
                    "docs_processed": summary.docs_processed,
                    "valid_json": summary.valid_json,
                    "valid_schema": summary.valid_schema,
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "elapsed_sec": round(summary.elapsed_sec, 3),
                    "docs_per_sec": round(summary.docs_per_sec, 2),
                    "decode_tok_s": round(summary.decode_tok_s, 1),
                    "output_path": str(summary.output_path),
                },
                indent=2,
            )
        )
    finally:
        eng.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    srv = FastGptOssServer(
        eng,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        coalesce_ms=args.coalesce_ms,
        max_batch_size=args.max_batch,
    )
    print(
        f"fast-gpt-oss listening on http://{srv.host}:{srv.port} "
        f"(model={srv.adapter.model_name}, vocab_out={eng.model.vocab_n}, mode={eng.default_mode}, coalesce_ms={args.coalesce_ms})"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        eng.close()
    return 0


def _load_fixture_payload(home: Path) -> tuple[str, dict[str, object], list[str]]:
    fixtures = home / "tests" / "fixtures"
    sys_msg = (fixtures / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    schema_obj = json.loads((fixtures / "extract_v1.schema.json").read_text(encoding="utf-8"))
    raw_docs = json.loads((fixtures / "sample_docs_8.json").read_text(encoding="utf-8"))
    prompts: list[str] = []
    for entry in raw_docs:
        if isinstance(entry, list) and len(entry) >= 4:
            _, fn, known, txt = entry[0], entry[1], entry[2], entry[3]
            prompts.append(f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}")
        elif isinstance(entry, dict):
            prompts.append(str(entry.get("text", "")))
        else:
            prompts.append(str(entry))
    return sys_msg, schema_obj, prompts


def _cmd_verify(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_msg, schema_obj, prompts = _load_fixture_payload(eng.paths.home)
        r_b1 = eng.generate_batch(
            prompts[:4],
            system_prompt=sys_msg,
            json_schema=schema_obj,
            max_tokens=512,
            mode="b1",
            compact=True,
            reasoning_effort="skip",
        )
        r_batched = eng.generate_batch(
            prompts[:8],
            system_prompt=sys_msg,
            json_schema=schema_obj,
            max_tokens=512,
            mode="batched",
            batch_size=8,
            compact=True,
            reasoning_effort="skip",
        )
        report = {
            "b1_valid": f"{r_b1.stats['valid_schema_count']}/{len(r_b1.items)}",
            "b1_docs_per_sec": round(float(r_b1.stats["docs_per_sec"]), 3),
            "b1_decode_tok_s": round(float(r_b1.stats["decode_tok_s"]), 1),
            "b8_valid": f"{r_batched.stats['valid_schema_count']}/{len(r_batched.items)}",
            "b8_docs_per_sec": round(float(r_batched.stats["docs_per_sec"]), 3),
            "b8_decode_tok_s": round(float(r_batched.stats["decode_tok_s"]), 1),
            "prefix16_token_matches": sum(
                1 for a, b in zip(r_b1.items, r_batched.items[:4]) if a.token_ids[:16] == b.token_ids[:16]
            ),
            "key_field_matches": sum(
                1
                for a, b in zip(r_b1.items, r_batched.items[:4])
                if a.parsed and b.parsed and all(a.parsed.get(k) == b.parsed.get(k) for k in ("status", "primary_kind", "domain"))
            ),
        }
        print(json.dumps(report, indent=2))
        if r_b1.stats["valid_schema_count"] != len(r_b1.items) or r_batched.stats["valid_schema_count"] != len(r_batched.items):
            return 1
    finally:
        eng.close()
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_msg, schema_obj, base_prompts = _load_fixture_payload(eng.paths.home)
        results: dict[str, object] = {}

        # Warm prefix cache
        _ = eng.generate_batch(
            base_prompts[:2],
            system_prompt=sys_msg,
            json_schema=schema_obj,
            max_tokens=args.max_tokens,
            mode="batched",
            batch_size=2,
            compact=True,
            reasoning_effort="skip",
        )

        # B=1 sequential
        t0 = time.perf_counter()
        r1 = eng.generate_batch(
            base_prompts,
            system_prompt=sys_msg,
            json_schema=schema_obj,
            max_tokens=args.max_tokens,
            mode="b1",
            compact=True,
            reasoning_effort="skip",
        )
        results["b1_seq"] = {
            "docs": len(base_prompts),
            "wall_sec": round(time.perf_counter() - t0, 3),
            "docs_per_sec": round(float(r1.stats["docs_per_sec"]), 3),
            "decode_tok_s": round(float(r1.stats["decode_tok_s"]), 1),
            "valid": f"{r1.stats['valid_schema_count']}/{len(r1.items)}",
            "peak_gb": round(float(r1.stats["peak_gb"]), 2),
        }

        for bs in [int(x) for x in args.batches.split(",") if x.strip()]:
            n = max(bs, len(base_prompts)) if bs >= len(base_prompts) else bs
            wave_prompts = [base_prompts[i % len(base_prompts)] for i in range(n)]
            rb = eng.generate_batch(
                wave_prompts,
                system_prompt=sys_msg,
                json_schema=schema_obj,
                max_tokens=args.max_tokens,
                mode="batched",
                batch_size=bs,
                compact=True,
                reasoning_effort="skip",
            )
            results[f"batched_B{bs}"] = {
                "docs": n,
                "elapsed_sec": round(float(rb.stats["elapsed_sec"]), 3),
                "prefill_sec": round(float(rb.stats["prefill_sec"]), 3),
                "decode_sec": round(float(rb.stats["decode_sec"]), 3),
                "docs_per_sec": round(float(rb.stats["docs_per_sec"]), 3),
                "decode_tok_s": round(float(rb.stats["decode_tok_s"]), 1),
                "valid": f"{rb.stats['valid_schema_count']}/{len(rb.items)}",
                "peak_gb": round(float(rb.stats["peak_gb"]), 2),
            }
        print(json.dumps(results, indent=2))
    finally:
        eng.close()
    return 0


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--home", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--vocab-out", type=int, default=DEFAULT_KEPT_VOCAB_SIZE)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpt-oss",
        description="Hardware-limit Apple Silicon openai/gpt-oss-20b inference CLI (Metal 4 MPP + MXFP4 + Hydragen v2)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Download/resolve weights, sync tokenizer, and build o200k vocab remap")
    _add_common_model_args(sw)
    sw.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    sw.set_defaults(func=_cmd_setup_weights)

    gn = sub.add_parser("generate", help="Run single prompt / conversation generation")
    _add_common_model_args(gn)
    gn.add_argument("prompt", nargs="?", default="Summarize the core idea of MoE routing cleanly in two sentences.")
    gn.add_argument("--messages-json", default=None)
    gn.add_argument("--system", default=None)
    gn.add_argument("--system-file", default=None)
    gn.add_argument("--schema", default=None)
    gn.add_argument("--max-tokens", type=int, default=512)
    gn.add_argument("--temperature", type=float, default=0.0)
    gn.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    gn.add_argument("--reasoning-effort", choices=("skip", "none", "low", "medium", "high"), default=None)
    gn.add_argument("--reasoning-budget", type=int, default=None)
    gn.add_argument("--enable-thinking", action="store_true")
    gn.add_argument("--show-reasoning", action="store_true")
    gn.add_argument("--compact-json", action="store_true")
    gn.add_argument("--no-warmup", action="store_true")
    gn.add_argument("--json", action="store_true")
    gn.set_defaults(func=_cmd_generate)

    ex = sub.add_parser("extract", help="Batch-extract structured JSON from JSON/JSONL into JSONL")
    _add_common_model_args(ex)
    ex.add_argument("input")
    ex.add_argument("output")
    ex.add_argument(
        "--system",
        default="Extract structured metadata as one compact minified JSON matching the schema. Output raw JSON only.",
    )
    ex.add_argument("--system-file", default=None)
    ex.add_argument("--schema", default=None)
    ex.add_argument("--text-key", default="text")
    ex.add_argument("--batch-size", type=int, default=64)
    ex.add_argument("--max-tokens", type=int, default=512)
    ex.add_argument("--temperature", type=float, default=0.0)
    ex.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    ex.add_argument("--reasoning-effort", choices=("skip", "none", "low", "medium", "high"), default="skip")
    ex.add_argument("--reasoning-budget", type=int, default=None)
    ex.add_argument("--whitespace-flexible", action="store_true")
    ex.add_argument("--no-warmup", action="store_true")
    ex.set_defaults(func=_cmd_extract)

    pq_p = sub.add_parser("parquet", help="Stream-extract structured JSON from Parquet file, directory, or glob")
    _add_common_model_args(pq_p)
    pq_p.add_argument("input")
    pq_p.add_argument("output")
    pq_p.add_argument(
        "--system",
        default="Extract structured metadata as one compact minified JSON matching the schema. Output raw JSON only.",
    )
    pq_p.add_argument("--system-file", default=None)
    pq_p.add_argument("--schema", default=None)
    pq_p.add_argument("--text-col", default=None)
    pq_p.add_argument("--keep-cols", default=None)
    pq_p.add_argument("--drop-text-col", action="store_true")
    pq_p.add_argument("--batch-rows", type=int, default=64)
    pq_p.add_argument("--shard-rows", type=int, default=4096)
    pq_p.add_argument("--max-tokens", type=int, default=512)
    pq_p.add_argument("--max-chars", type=int, default=8000)
    pq_p.add_argument("--max-pages", type=int, default=0)
    pq_p.add_argument("--temperature", type=float, default=0.0)
    pq_p.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    pq_p.add_argument("--reasoning-effort", choices=("skip", "none", "low", "medium", "high"), default="skip")
    pq_p.add_argument("--reasoning-budget", type=int, default=None)
    pq_p.add_argument("--whitespace-flexible", action="store_true")
    pq_p.add_argument("--filter-ocr", action="store_true")
    pq_p.add_argument("--ocr-pages-only", action="store_true")
    pq_p.add_argument("--resume", action="store_true")
    pq_p.add_argument("--no-warmup", action="store_true")
    pq_p.set_defaults(func=_cmd_parquet)

    sv = sub.add_parser("serve", help="Run OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions)")
    _add_common_model_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--model-name", default=None)
    sv.add_argument("--coalesce-ms", type=float, default=0.0)
    sv.add_argument("--max-batch", type=int, default=64)
    sv.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    sv.add_argument("--reasoning-effort", choices=("skip", "none", "low", "medium", "high"), default="low")
    sv.add_argument("--no-warmup", action="store_true")
    sv.set_defaults(func=_cmd_serve)

    vf = sub.add_parser("verify", help="Verify DocumentExtractV1 B=1 and B=8 parity + strict schema compliance")
    _add_common_model_args(vf)
    vf.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    vf.add_argument("--no-warmup", action="store_true")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark B=1 vs Batched (Hydragen v2 + Metal 4 Hybrid MoE)")
    _add_common_model_args(bn)
    bn.add_argument("--batches", default="8,64,128")
    bn.add_argument("--max-tokens", type=int, default=512)
    bn.add_argument("--mode", choices=("auto", "b1", "batched"), default="auto")
    bn.add_argument("--no-warmup", action="store_true")
    bn.set_defaults(func=_cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    from fast_shared.hw_check import warn_if_untested_hardware

    parser = build_parser()
    args = parser.parse_args(argv)
    warn_if_untested_hardware()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
