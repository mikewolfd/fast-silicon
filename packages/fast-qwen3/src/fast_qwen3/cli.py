"""CLI entrypoints (`fast-qwen3 setup-weights|generate|extract|parquet|serve|bench`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from fast_shared.hf import copy_tokenizer_files

from .engine import FastQwen3Engine
from .paths import DEFAULT_HF_REPO, MODEL_PRESETS, resolve_paths, resolve_preset_repo
from .pipeline import extract_jsonl, extract_parquet
from .server import FastQwen3Server


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--home", default=None)
    p.add_argument("--model-dir", default=None, help=f"Local path, HF repo, or preset ({', '.join(sorted(MODEL_PRESETS))})")
    p.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    p.add_argument("--lora", default=None, help="Optional PEFT / MLX LoRA adapter directory or .safetensors file")
    p.add_argument("--vocab-out", type=int, default=0, help="Optional pruned output vocab size for untied models")
    p.add_argument("--mode", default="auto", choices=("auto", "gpu"))
    p.add_argument("--allow-download", action="store_true", help="Allow pulling missing snapshot from Hugging Face")
    p.add_argument("--no-warmup", action="store_true")


def _make_engine(args: argparse.Namespace) -> FastQwen3Engine:
    vocab_out = int(args.vocab_out) if getattr(args, "vocab_out", 0) and int(args.vocab_out) > 0 else None
    return FastQwen3Engine(
        home=args.home,
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        lora_adapter=args.lora,
        vocab_out=vocab_out,
        default_mode=args.mode,
        local_files_only=not bool(args.allow_download),
        warmup=not bool(args.no_warmup),
    )


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    eff_repo = resolve_preset_repo(args.model_dir or args.repo_id)
    model_dir = paths.resolve_model_dir(
        args.model_dir,
        repo_id=eff_repo,
        local_files_only=not bool(args.allow_download),
    )
    copy_tokenizer_files(model_dir, paths.tokenizer_dir)
    print(
        json.dumps(
            {
                "repo_id": eff_repo,
                "model_dir": str(model_dir),
                "tokenizer_dir": str(paths.tokenizer_dir),
            },
            indent=2,
        )
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        eff_ws: bool | None = False if args.compact_json else None
        item = eng.generate(
            args.prompt,
            system_prompt=sys_txt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            speculative=None if args.speculative == "auto" else (args.speculative == "on"),
            whitespace_flexible=eff_ws,
            enable_thinking=args.enable_thinking,
            thinking_budget_tokens=args.thinking_budget,
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
            print(item.text)
    finally:
        eng.close()
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else (args.system or "")
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        stats = extract_jsonl(
            eng,
            args.input,
            args.output,
            system_prompt=sys_txt,
            json_schema=schema_str,
            text_key=args.text_key,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(json.dumps(stats, indent=2))
    finally:
        eng.close()
    return 0


def _cmd_parquet(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else (args.system or "")
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
            batch_rows=args.batch_rows,
            shard_rows=args.shard_rows,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_chars=args.max_chars,
            max_pages=args.max_pages,
            resume=args.resume,
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
    try:
        with FastQwen3Server(eng, host=args.host, port=args.port, max_batch_size=args.max_batch_size) as srv:
            print(f"Serving fast-qwen3 ({eng.model_name}) on http://{srv.host}:{srv.port}", flush=True)
            srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.close()
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        prompts = [f"Summarize key entities and dates in passage #{i}: Alpha Corp filed notice on 2026-03-14." for i in range(args.batch_size)]
        schema = {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["entity", "date"],
            "additionalProperties": False,
        }
        t0 = time.perf_counter()
        res = eng.generate_batch(
            prompts,
            system_prompt="Extract structured fields as JSON.",
            json_schema=schema if args.structured else None,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        dt = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "model": eng.model_name,
                    "batch_size": args.batch_size,
                    "structured": bool(args.structured),
                    "elapsed_sec": round(dt, 4),
                    "stats": res.stats,
                    "valid_schema": sum(1 for x in res.items if x.valid_schema),
                },
                indent=2,
            )
        )
    finally:
        eng.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fast-qwen3", description="Apple Silicon hybrid Qwen3.5 / Qwen3.6 engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup-weights")
    _add_common_model_args(p_setup)
    p_setup.set_defaults(func=_cmd_setup_weights)

    p_gen = sub.add_parser("generate")
    _add_common_model_args(p_gen)
    p_gen.add_argument("prompt")
    p_gen.add_argument("--system", default=None)
    p_gen.add_argument("--system-file", default=None)
    p_gen.add_argument("--schema", default=None)
    p_gen.add_argument("--max-tokens", type=int, default=512)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--speculative", choices=("auto", "on", "off"), default="auto")
    p_gen.add_argument("--compact-json", action="store_true")
    p_gen.add_argument("--enable-thinking", action="store_true")
    p_gen.add_argument("--thinking-budget", type=int, default=0)
    p_gen.add_argument("--json", action="store_true")
    p_gen.set_defaults(func=_cmd_generate)

    p_ext = sub.add_parser("extract")
    _add_common_model_args(p_ext)
    p_ext.add_argument("--input", required=True)
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--system", default="Extract structured JSON.")
    p_ext.add_argument("--system-file", default=None)
    p_ext.add_argument("--schema", default=None)
    p_ext.add_argument("--text-key", default="text")
    p_ext.add_argument("--batch-size", type=int, default=32)
    p_ext.add_argument("--max-tokens", type=int, default=512)
    p_ext.add_argument("--temperature", type=float, default=0.0)
    p_ext.set_defaults(func=_cmd_extract)

    p_pq = sub.add_parser("parquet")
    _add_common_model_args(p_pq)
    p_pq.add_argument("--input", required=True)
    p_pq.add_argument("--output", required=True)
    p_pq.add_argument("--system", default="Extract structured JSON.")
    p_pq.add_argument("--system-file", default=None)
    p_pq.add_argument("--schema", default=None)
    p_pq.add_argument("--text-col", default=None)
    p_pq.add_argument("--keep-cols", default=None)
    p_pq.add_argument("--drop-text-col", action="store_true")
    p_pq.add_argument("--batch-rows", type=int, default=64)
    p_pq.add_argument("--shard-rows", type=int, default=4096)
    p_pq.add_argument("--max-tokens", type=int, default=512)
    p_pq.add_argument("--temperature", type=float, default=0.0)
    p_pq.add_argument("--max-chars", type=int, default=16000)
    p_pq.add_argument("--max-pages", type=int, default=0)
    p_pq.add_argument("--resume", action="store_true")
    p_pq.set_defaults(func=_cmd_parquet)

    p_srv = sub.add_parser("serve")
    _add_common_model_args(p_srv)
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.add_argument("--max-batch-size", type=int, default=64)
    p_srv.set_defaults(func=_cmd_serve)

    p_bench = sub.add_parser("bench")
    _add_common_model_args(p_bench)
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--max-tokens", type=int, default=64)
    p_bench.add_argument("--structured", action="store_true")
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
