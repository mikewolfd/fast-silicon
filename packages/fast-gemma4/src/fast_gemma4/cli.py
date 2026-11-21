"""CLI entrypoints (`gemma4 setup-weights|setup-ane|generate|extract|parquet|serve|verify|bench`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir

from .engine import FastGemma4Engine
from .paths import DEFAULT_HF_REPO, resolve_paths
from .pipeline import extract_jsonl, extract_parquet
from .server import FastGemma4Server


def _make_engine(args: argparse.Namespace) -> FastGemma4Engine:
    ane_var = getattr(args, "ane_variant", "fp16")
    mtp_ref = getattr(args, "mtp_drafter_path", None) or (True if getattr(args, "mtp_drafter", False) else None)
    return FastGemma4Engine(
        home=args.home,
        model_dir=getattr(args, "model_dir", None),
        default_mode=getattr(args, "mode", "auto"),
        vocab_out=getattr(args, "vocab_out", None),
        ane_variant=ane_var,
        ane_ring_variant=ane_var,
        lm_head_mode=getattr(args, "lm_head_mode", "q4"),
        mlp_dtype=getattr(args, "mlp_dtype", "bfloat16"),
        kv_i8=not getattr(args, "no_kv_i8", False),
        mtp_drafter=mtp_ref,
        warmup=not getattr(args, "no_warmup", False),
        warmup_ane=bool(getattr(args, "mode", "auto") == "hetero"),
    )


def _resolve_script(paths: Any, name: str) -> Path:
    script = paths.home / "scripts" / name
    if script.exists():
        return script
    return Path(__file__).resolve().parents[2] / "scripts" / name


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    model_dir = resolve_hf_model_dir(args.model_dir, DEFAULT_HF_REPO, local_files_only=args.local_files_only)
    copy_tokenizer_files(model_dir, paths.tokenizer_dir)

    script_vocab = _resolve_script(paths, "build_vocab_remap.py")
    if script_vocab.exists():
        rc = subprocess.call([sys.executable, str(script_vocab), "--home", str(paths.home)])
        if rc != 0:
            return rc

    if not args.skip_early14:
        script_exp = _resolve_script(paths, "export_early14_f16.py")
        if script_exp.exists():
            cmd = [sys.executable, str(script_exp), "--home", str(paths.home), "--model-dir", str(model_dir)]
            return subprocess.call(cmd)
    return 0


def _cmd_setup_ane(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    script = _resolve_script(paths, "build_ane_models.py")
    if not script.exists():
        raise FileNotFoundError(f"Missing {script}")
    cmd = [sys.executable, str(script), "--home", str(paths.home), "--variant", args.variant]
    if getattr(args, "model_dir", None):
        cmd.extend(["--model-dir", str(args.model_dir)])
    return subprocess.call(cmd)


def _cmd_generate(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if getattr(args, "system_file", None) else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        prompt: object = json.loads(Path(args.messages_json).read_text(encoding="utf-8")) if args.messages_json else args.prompt
        spec = True if args.speculative else (False if args.no_speculative else None)
        item = eng.generate(
            prompt,  # type: ignore[arg-type]
            system_prompt=sys_txt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            mode=args.mode,
            speculative=spec,
            whitespace_flexible=args.whitespace_flexible,
            max_whitespace_run=getattr(args, "json_max_whitespace", None),
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "text": item.text,
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
        sys_txt = Path(args.system_file).read_text(encoding="utf-8") if args.system_file else args.system
        schema_str = Path(args.schema).read_text(encoding="utf-8") if args.schema else None
        spec = True if args.speculative else (False if args.no_speculative else None)
        stats = extract_jsonl(
            eng,
            args.input,
            args.output,
            system_prompt=sys_txt,
            json_schema=schema_str,
            text_key=args.text_key,
            mode=args.mode,
            speculative=spec,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            whitespace_flexible=args.whitespace_flexible,
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
        spec = True if args.speculative else (False if args.no_speculative else None)
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
            speculative=spec,
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
    srv = FastGemma4Server(
        eng,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        coalesce_ms=args.coalesce_ms,
        max_batch_size=args.max_batch,
    )
    print(
        f"fast-gemma4 listening on http://{srv.host}:{srv.port} "
        f"(model={srv.adapter.model_name}, ane={eng.ane_available}, mode={eng.default_mode}, coalesce_ms={args.coalesce_ms})"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        eng.close()
    return 0


def _load_fixture_dataset(paths) -> tuple[str, str, list[str]]:
    fix_dir = paths.home / "tests" / "fixtures"
    schema_p = fix_dir / "extract_v1.schema.json"
    docs_p = fix_dir / "sample_docs_8.json"
    if schema_p.exists() and docs_p.exists():
        schema_str = schema_p.read_text(encoding="utf-8")
        docs = json.loads(docs_p.read_text(encoding="utf-8"))
        sys_p = fix_dir / "sys_extract_v1.txt"
        if sys_p.exists():
            sys_prompt = sys_p.read_text(encoding="utf-8").strip()
        else:
            sys_prompt = (
                "Extract structured document metadata as strict JSON matching the schema. "
                "Use exact substrings for entity names when possible."
            )
        prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]
        return sys_prompt, schema_str, prompts

    schema_str = json.dumps(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "domain": {"type": "string", "enum": ["energy", "finance", "transportation", "other"]},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "domain", "entities"],
            "additionalProperties": False,
        }
    )
    sys_prompt = "Extract document metadata as one JSON object matching the schema. Output raw JSON only."
    prompts = [
        f"Filename: report_{i:02d}.txt\nText:\nPacific Transmission Consortium technical report PTR-2025-{i:02d} on regional substation inverter stability and grid protection upgrades."
        for i in range(8)
    ]
    return sys_prompt, schema_str, prompts


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    eng = FastGemma4Engine(
        home=args.home,
        model_dir=getattr(args, "model_dir", None),
        default_mode="gpu",
        vocab_out=args.vocab_out,
        lm_head_mode=getattr(args, "lm_head_mode", "q4"),
        mlp_dtype=getattr(args, "mlp_dtype", "bfloat16"),
        kv_i8=not getattr(args, "no_kv_i8", False),
        warmup=True,
        warmup_ane=False,
    )
    try:
        print(
            f"home={paths.home} vocab_out={eng.model.vocab_size} param_gb={eng.model.param_gb:.2f}GB "
            f"mpp={eng.model.use_mpp} ane_available={eng.ane_available} bundle={eng._ane_bundle}"
        )
        if paths.lib_ane_prefix.exists() and eng._ane_bundle is not None and paths.ane_cli.exists():
            out = subprocess.check_output(
                [str(paths.ane_cli), "--model", str(eng._ane_bundle), "--batch", "4", "--iters", "1", "--inspect", "--json"],
                text=True,
            ).strip()
            print(f"native_ane_inspect={out.splitlines()[-1]}")

        sys_prompt, schema_str, base_prompts = _load_fixture_dataset(paths)
        r_gpu = eng.generate_batch(
            base_prompts,
            system_prompt=sys_prompt,
            json_schema=schema_str,
            max_tokens=512,
            mode="gpu",
            speculative=False,
        )
        r_spec = eng.generate_batch(
            base_prompts,
            system_prompt=sys_prompt,
            json_schema=schema_str,
            max_tokens=512,
            mode="gpu",
            speculative=True,
        )
        spec_exact = sum(1 for a, b in zip(r_gpu.items, r_spec.items) if a.parsed and b.parsed and a.parsed.get("status") == b.parsed.get("status") and a.parsed.get("primary_kind") == b.parsed.get("primary_kind"))
        print(
            f"GPU 8-doc: valid_json={r_gpu.stats['valid_json_count']}/8 valid_schema={r_gpu.stats['valid_schema_count']}/8 | "
            f"PromptLookup semantic_match={spec_exact}/8 (spec_valid={r_spec.stats['valid_schema_count']}/8) | "
            f"decode_gpu={r_gpu.stats['decode_tok_s']:.0f} tok/s vs spec={r_spec.stats['decode_tok_s']:.0f} tok/s"
        )
        if eng.ane_available:
            # Pad to >=64 docs (min wave size) and check hetero ANE+GPU exact schema compliance
            pad_prompts = [base_prompts[i % len(base_prompts)] for i in range(64)]
            r_het = eng.generate_batch(
                pad_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=512,
                mode="hetero",
                batch_size=64,
            )
            print(
                f"Hetero 64-doc (1 wave ANE+GPU): valid_json={r_het.stats['valid_json_count']}/64 "
                f"valid_schema={r_het.stats['valid_schema_count']}/64 | total={r_het.stats['docs_per_sec']:.2f} docs/s "
                f"decode={r_het.stats['decode_tok_s']:.0f} tok/s"
            )
    finally:
        eng.close()
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    eng = FastGemma4Engine(
        home=args.home,
        model_dir=getattr(args, "model_dir", None),
        default_mode="auto",
        vocab_out=args.vocab_out,
        ane_variant=args.ane_variant,
        ane_ring_variant=args.ane_variant,
        lm_head_mode=getattr(args, "lm_head_mode", "q4"),
        mlp_dtype=getattr(args, "mlp_dtype", "bfloat16"),
        kv_i8=not getattr(args, "no_kv_i8", False),
        warmup=True,
        warmup_ane=(not args.skip_ane),
    )
    try:
        sys_prompt, schema_str, base_prompts = _load_fixture_dataset(paths)
        print(
            f"home={paths.home} vocab={eng.model.vocab_size} param_gb={eng.model.param_gb:.2f}GB "
            f"ane_available={eng.ane_available} ane_variant={args.ane_variant}"
        )

        # Warmup compile shapes
        eng.generate_batch(
            base_prompts[:2],
            system_prompt=sys_prompt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            mode="gpu",
            speculative=True,
        )

        # Track A: GPU-only Batched (B_gpu)
        b_gpu = args.batch_gpu
        p_gpu = [base_prompts[i % len(base_prompts)] for i in range(b_gpu)]
        res_gpu = eng.generate_batch(
            p_gpu,
            system_prompt=sys_prompt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            mode="gpu",
            speculative=False,
            batch_size=b_gpu,
        )
        print(
            f"[GPUOnly B={b_gpu}] JSON={res_gpu.stats['valid_json_count']}/{b_gpu} "
            f"Strict={res_gpu.stats['valid_schema_count']}/{b_gpu} | "
            f"pre={res_gpu.stats['prefill_sec']:.2f}s | dec={res_gpu.stats['decode_sec']:.2f}s "
            f"({res_gpu.stats['decode_tok_s']:.0f} tok/s) | TOTAL={res_gpu.stats['elapsed_sec']:.2f}s -> "
            f"{res_gpu.stats['docs_per_sec']:.2f} docs/s | peak={res_gpu.stats['peak_gb']:.2f}GB"
        )

        # Track B: PromptLookup L_q=4 (B_spec)
        b_spec = args.batch_spec
        p_spec = [base_prompts[i % len(base_prompts)] for i in range(b_spec)]
        res_spec = eng.generate_batch(
            p_spec,
            system_prompt=sys_prompt,
            json_schema=schema_str,
            max_tokens=args.max_tokens,
            mode="gpu",
            speculative=True,
            batch_size=b_spec,
        )
        print(
            f"[PromptLookup B={b_spec}] JSON={res_spec.stats['valid_json_count']}/{b_spec} "
            f"Strict={res_spec.stats['valid_schema_count']}/{b_spec} | "
            f"pre={res_spec.stats['prefill_sec']:.2f}s | dec={res_spec.stats['decode_sec']:.2f}s "
            f"({res_spec.stats['decode_tok_s']:.0f} eff tok/s) | TOTAL={res_spec.stats['elapsed_sec']:.2f}s -> "
            f"{res_spec.stats['docs_per_sec']:.2f} docs/s | peak={res_spec.stats['peak_gb']:.2f}GB"
        )

        # Track C: Concurrent ANE+GPU Pipelined Waves (waves x B_wave)
        if not args.skip_ane and eng.ane_available:
            b_wave = args.batch_gpu
            n_tot = args.waves * b_wave
            p_het = [base_prompts[i % len(base_prompts)] for i in range(n_tot)]
            res_het = eng.generate_batch(
                p_het,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=args.max_tokens,
                mode="hetero",
                speculative=False,
                batch_size=b_wave,
                use_ane_ring="auto",
            )
            print(
                f"[Concurrent ANE+GPU {args.waves}xB={b_wave}] JSON={res_het.stats['valid_json_count']}/{n_tot} "
                f"Strict={res_het.stats['valid_schema_count']}/{n_tot} | "
                f"pre+wait={res_het.stats['prefill_sec']:.2f}s | dec={res_het.stats['decode_sec']:.2f}s "
                f"({res_het.stats['decode_tok_s']:.0f} tok/s) | TOTAL={res_het.stats['elapsed_sec']:.2f}s -> "
                f"{res_het.stats['docs_per_sec']:.2f} docs/s | peak={res_het.stats['peak_gb']:.2f}GB"
            )
    finally:
        eng.close()
    return 0


def _add_json_layout_args(subparser: argparse.ArgumentParser) -> None:
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("--compact-json", dest="whitespace_flexible", action="store_false", help="Generate minified JSON with no separator spaces")
    group.add_argument("--whitespace-flexible", dest="whitespace_flexible", action="store_true", help="Allow JSON whitespace between tokens")
    subparser.set_defaults(whitespace_flexible=None)


def _add_model_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--vocab-out", type=int, help="Limit allowed output tokens, e.g. 150786 for the English-oriented profile (default: checkpoint vocabulary)")
    subparser.add_argument("--model-dir", default=None, help="Override local HuggingFace/fused checkpoint directory")
    subparser.add_argument("--no-kv-i8", action="store_true", help="Disable INT8 rolling KV cache compression")
    subparser.add_argument("--lm-head-mode", choices=("q4", "bf16_mpp"), default="q4")
    subparser.add_argument("--mlp-dtype", choices=("bfloat16", "quantized"), default="bfloat16")
    subparser.add_argument(
        "--mtp-drafter",
        action="store_true",
        help="Opt-in Gemma-4 E2B MTP speculative drafter (google/gemma-4-E2B-it-assistant; strictly gated to B=1..4, T=0)",
    )
    subparser.add_argument(
        "--mtp-drafter-path",
        default=None,
        help="Opt-in local path or HF repo for the Gemma-4 E2B MTP drafter (strictly gated to B=1..4, T=0)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma4", description="Apple Silicon (M4 / M5+) fast-gemma4 CLI (E2B / E4B)")
    p.add_argument("--home", default=None, help="Override FAST_GEMMA4_HOME directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Copy tokenizer, build vocab remap, and export Early14 FP16 weights")
    sw.add_argument("--model-dir", default=None)
    sw.add_argument("--local-files-only", action="store_true")
    sw.add_argument("--skip-early14", action="store_true")
    sw.set_defaults(func=_cmd_setup_weights)

    sa = sub.add_parser("setup-ane", help="Compile CoreML ANE Early14 prefix chunk bundles (.mlmodelc)")
    sa.add_argument("--model-dir", default=None)
    sa.add_argument("--variant", choices=("both", "fp16", "w8"), default="both")
    sa.set_defaults(func=_cmd_setup_ane)

    gn = sub.add_parser("generate", help="Generate structured JSON or text from a single prompt")
    _add_model_args(gn)
    gn.add_argument("prompt", nargs="?", default="Summarize this item as JSON.")
    gn.add_argument("--system", default=None)
    gn.add_argument("--system-file", default=None)
    gn.add_argument("--messages-json", default=None)
    gn.add_argument("--schema", default=None, help="Path to JSON Schema file")
    gn.add_argument("--max-tokens", type=int, default=512)
    gn.add_argument("--temperature", type=float, default=0.0)
    gn.add_argument("--mode", choices=("auto", "gpu", "hetero"), default="auto")
    gn.add_argument("--ane-variant", choices=("fp16", "w8"), default="fp16")
    gn.add_argument("--speculative", action="store_true")
    gn.add_argument("--no-speculative", action="store_true")
    _add_json_layout_args(gn)
    gn.add_argument(
        "--json-max-whitespace",
        type=int,
        default=32,
        help="Max consecutive whitespace chars between JSON tokens when flexible JSON is active (default: 32; 0 disables)",
    )
    gn.add_argument("--no-warmup", action="store_true")
    gn.add_argument("--json", action="store_true")
    gn.set_defaults(func=_cmd_generate)

    ex = sub.add_parser("extract", help="Batch-extract structured JSON from JSON/JSONL file into JSONL")
    _add_model_args(ex)
    ex.add_argument("input")
    ex.add_argument("output")
    ex.add_argument(
        "--system",
        default="Extract structured metadata as one JSON object matching the schema. Output raw JSON only.",
    )
    ex.add_argument("--system-file", default=None)
    ex.add_argument("--schema", default=None)
    ex.add_argument("--text-key", default="text")
    ex.add_argument("--batch-size", type=int, default=128)
    ex.add_argument("--max-tokens", type=int, default=512)
    ex.add_argument("--temperature", type=float, default=0.0)
    ex.add_argument("--mode", choices=("auto", "gpu", "hetero"), default="auto")
    ex.add_argument("--ane-variant", choices=("fp16", "w8"), default="fp16")
    ex.add_argument("--speculative", action="store_true")
    ex.add_argument("--no-speculative", action="store_true")
    _add_json_layout_args(ex)
    ex.add_argument("--no-warmup", action="store_true")
    ex.set_defaults(func=_cmd_extract)

    pq_p = sub.add_parser("parquet", help="Stream-extract structured JSON from Parquet file, directory, or glob")
    _add_model_args(pq_p)
    pq_p.add_argument("input")
    pq_p.add_argument("output")
    pq_p.add_argument(
        "--system",
        default="Extract structured metadata as one JSON object matching the schema. Output raw JSON only.",
    )
    pq_p.add_argument("--system-file", default=None)
    pq_p.add_argument("--schema", default=None)
    pq_p.add_argument("--text-col", default=None)
    pq_p.add_argument("--keep-cols", default=None)
    pq_p.add_argument("--drop-text-col", action="store_true")
    pq_p.add_argument("--batch-rows", type=int, default=128)
    pq_p.add_argument("--shard-rows", type=int, default=4096)
    pq_p.add_argument("--max-tokens", type=int, default=512)
    pq_p.add_argument("--max-chars", type=int, default=8000)
    pq_p.add_argument("--max-pages", type=int, default=0, help="If >0, slice each doc to its first N pages via page_offsets")
    pq_p.add_argument("--temperature", type=float, default=0.0)
    pq_p.add_argument("--mode", choices=("auto", "gpu", "hetero"), default="auto")
    pq_p.add_argument("--ane-variant", choices=("fp16", "w8"), default="fp16")
    pq_p.add_argument("--speculative", action="store_true")
    pq_p.add_argument("--no-speculative", action="store_true")
    _add_json_layout_args(pq_p)
    pq_p.add_argument("--filter-ocr", action="store_true")
    pq_p.add_argument("--ocr-pages-only", action="store_true")
    pq_p.add_argument("--resume", action="store_true")
    pq_p.add_argument("--no-warmup", action="store_true")
    pq_p.set_defaults(func=_cmd_parquet)

    sv = sub.add_parser("serve", help="Run OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions)")
    _add_model_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--model-name", default=None)
    sv.add_argument("--coalesce-ms", type=float, default=0.0, help="Concurrent chat batch coalesce wait in ms (default: 0)")
    sv.add_argument("--max-batch", type=int, default=128, help="Max coalesced concurrent chat batch size (default: 128)")
    sv.add_argument("--mode", choices=("auto", "gpu", "hetero"), default="auto")
    sv.add_argument("--ane-variant", choices=("fp16", "w8"), default="fp16")
    sv.add_argument("--no-warmup", action="store_true")
    sv.set_defaults(func=_cmd_serve)

    vf = sub.add_parser("verify", help="Verify native ANE, JSON schema, PromptLookup exactness, and Hetero parity")
    _add_model_args(vf)
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark GPUOnly vs PromptLookup vs Concurrent ANE+GPU end-to-end")
    _add_model_args(bn)
    bn.add_argument("--batch-gpu", type=int, default=128)
    bn.add_argument("--batch-spec", type=int, default=32)
    bn.add_argument("--waves", type=int, default=2)
    bn.add_argument("--max-tokens", type=int, default=512)
    bn.add_argument("--ane-variant", choices=("fp16", "w8"), default="fp16")
    bn.add_argument("--skip-ane", action="store_true")
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
