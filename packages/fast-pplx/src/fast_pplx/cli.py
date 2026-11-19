"""CLI entrypoints (`fast-pplx setup-weights|setup-ane|embed|bench|parquet|serve`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from fast_shared.cli_common import (
    add_embed_arguments,
    add_parquet_arguments,
    add_serve_arguments,
    run_embed_command,
    run_parquet_command,
    run_serve_command,
)
from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir
from fast_shared.hw_check import warn_if_untested_hardware

from .config import DEFAULT_MODEL_NAME
from .gpu import _find_local_hf_snapshot
from .parquet_pipeline import embed_contextual_parquet, resolve_text_column
from .paths import resolve_paths

_MODES: tuple[str, ...] = ("auto", "gpu", "ane", "hetero", "hetero2")


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default=None, help="Override FAST_PPLX_HOME directory")
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Model activation weight dtype (default: float16)",
    )
    parser.add_argument(
        "--no-mapped-embedding",
        action="store_true",
        help="Load embed_tokens onto Metal GPU instead of CPU safetensors mmap",
    )
    parser.add_argument(
        "--cache-limit-mb",
        type=int,
        default=512,
        help="MLX allocator cache limit in MiB (default: 512)",
    )


def _make_engine(args: argparse.Namespace) -> Any:
    from .engine import FastPplx

    return FastPplx(
        home=getattr(args, "home", None),
        mode=getattr(args, "mode", "auto"),
        dtype=getattr(args, "model_dtype", "float16"),
        use_mapped_embedding=not getattr(args, "no_mapped_embedding", False),
        cache_limit_mb=getattr(args, "cache_limit_mb", 512),
    )


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    script = paths.home / "scripts" / "export_weights.py"
    if script.exists():
        cmd = [sys.executable, str(script), "--home", str(paths.home)]
        if args.model_dir:
            cmd.extend(["--model-dir", str(args.model_dir)])
        if args.local_files_only:
            cmd.append("--local-files-only")
        return subprocess.call(cmd)

    model_dir: Path | None = None
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    elif args.local_files_only:
        model_dir = _find_local_hf_snapshot(DEFAULT_MODEL_NAME)
    if model_dir is None:
        model_dir = resolve_hf_model_dir(
            args.model_dir,
            DEFAULT_MODEL_NAME,
            local_files_only=args.local_files_only,
        )

    copy_tokenizer_files(model_dir, paths.tokenizer_dir)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, paths.model_dir / fname)
    for st_file in sorted(model_dir.glob("*.safetensors")):
        shutil.copy2(st_file, paths.model_dir / st_file.name)
    print(f"Prepared model assets at {paths.model_dir} and tokenizer at {paths.tokenizer_dir}")
    return 0


def _cmd_setup_ane(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    script = paths.home / "scripts" / "build_ane_models.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Missing ANE export script {script}. Install full package scripts or build ANE artifacts first."
        )
    cmd = [sys.executable, str(script), "--home", str(paths.home)]
    if getattr(args, "model_dir", None):
        cmd.extend(["--model-dir", str(args.model_dir)])
    if getattr(args, "variant", None):
        cmd.extend(["--variant", str(args.variant)])
    if getattr(args, "seq_lens", None):
        cmd.extend(["--seq-lens", *str(args.seq_lens).replace(",", " ").split()])
    if getattr(args, "precision", None):
        cmd.extend(["--precision", str(args.precision)])
    if getattr(args, "layers_per_stage", None) is not None:
        cmd.extend(["--layers-per-stage", str(args.layers_per_stage)])
    if getattr(args, "inv_mode", None):
        cmd.extend(["--inv-mode", str(args.inv_mode)])
    if getattr(args, "calib_stats", None):
        cmd.extend(["--calib-stats", str(args.calib_stats)])
    if getattr(args, "ftz_alpha", None) is not None:
        cmd.extend(["--ftz-alpha", str(args.ftz_alpha)])
    return subprocess.call(cmd)


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    try:
        if args.parquet:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(args.parquet)
            col = resolve_text_column(pf.schema_arrow, args.text_col)
            raw = pq.read_table(args.parquet, columns=[col]).column(col).to_pylist()[: args.n]
            texts = [t for t in raw if isinstance(t, str) and t.strip()]
        else:
            phrase = (
                "Ocean currents move thermal energy toward higher latitudes across basin gyres, "
                "moderating coastal climates over seasonal and decadal intervals. "
            )
            repeats = max(1, (args.seq_len + 15) // 24)
            base_text = phrase * repeats
            texts = [f"Document {i}. {base_text}" for i in range(args.n)]

        toks = [eng.tokenizer.encode_document(t, max_length=args.seq_len) for t in texts]
        total_toks = sum(len(row) for row in toks)

        if not args.no_warmup and texts:
            _ = eng.encode(
                texts[: min(4, len(texts))],
                batch_size=min(args.batch_size, 4),
                max_length=args.seq_len,
                dimensions=args.dimensions,
            )

        t0 = time.perf_counter()
        vecs = eng.encode(
            texts,
            batch_size=args.batch_size,
            max_length=args.seq_len,
            dimensions=args.dimensions,
        )
        dt = time.perf_counter() - t0
        print(
            f"mode={args.mode} n={len(texts)} shape={tuple(vecs.shape)} "
            f"total_tokens={total_toks} elapsed={dt:.3f}s "
            f"throughput={total_toks / max(dt, 1e-9):.1f} tok/s ({len(texts) / max(dt, 1e-9):.1f} texts/s)"
        )
        return 0
    finally:
        eng.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-pplx",
        description="Hardware-limit Apple Silicon contextual embedding engine for pplx-embed-v2-context-0.6b",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_weights = sub.add_parser("setup-weights", help="Prepare local weights and tokenizer assets")
    _add_common_engine_args(p_weights)
    p_weights.add_argument("--model-dir", default=None, help="Optional local HF snapshot directory")
    p_weights.add_argument("--local-files-only", action="store_true", help="Do not hit network")

    p_ane = sub.add_parser("setup-ane", help="Build CoreML ANE bundles via scripts/build_ane_models.py")
    _add_common_engine_args(p_ane)
    p_ane.add_argument("--model-dir", default=None)
    p_ane.add_argument("--variant", default=None)
    p_ane.add_argument("--seq-lens", default=None)
    p_ane.add_argument("--precision", choices=("w8a16", "w8a8"), default=None)
    p_ane.add_argument("--layers-per-stage", type=int, default=None)
    p_ane.add_argument("--inv-mode", choices=("block", "fwd_subst"), default=None)
    p_ane.add_argument("--calib-stats", default=None)
    p_ane.add_argument("--ftz-alpha", type=float, default=None)

    p_embed = sub.add_parser("embed", help="Embed input texts from arguments or stdin")
    _add_common_engine_args(p_embed)
    add_embed_arguments(
        p_embed,
        mode_choices=_MODES,
        default_mode="auto",
        allow_doc_prefix=True,
        allow_dimensions=True,
    )

    p_bench = sub.add_parser("bench", help="Benchmark throughput on synthetic or Parquet inputs")
    _add_common_engine_args(p_bench)
    p_bench.add_argument("--mode", choices=_MODES, default="auto")
    p_bench.add_argument("--n", type=int, default=64)
    p_bench.add_argument("--batch-size", type=int, default=16)
    p_bench.add_argument("--seq-len", type=int, default=512)
    p_bench.add_argument("--dimensions", type=int, default=None)
    p_bench.add_argument("--no-warmup", action="store_true")
    p_bench.add_argument("--parquet", default=None)
    p_bench.add_argument("--text-col", default=None)

    p_parquet = sub.add_parser("parquet", help="Embed Parquet columns into sharded output Parquet files")
    _add_common_engine_args(p_parquet)
    add_parquet_arguments(
        p_parquet,
        mode_choices=_MODES,
        default_mode="auto",
        default_max_body_tokens=508,
        allow_doc_prefix=True,
        allow_dimensions=True,
    )
    p_parquet.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="Optional two-knob contextual window budget (e.g. 2048 or 4096) grouping document chunks",
    )
    p_parquet.add_argument(
        "--chunk-tokens",
        type=int,
        default=256,
        help="Target chunk size in tokens when --context-tokens is active (default: 256)",
    )

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible HTTP embedding server")
    _add_common_engine_args(p_serve)
    add_serve_arguments(
        p_serve,
        mode_choices=_MODES,
        default_mode="auto",
        allow_dimensions=True,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_if_untested_hardware()

    if args.command == "setup-weights":
        return _cmd_setup_weights(args)
    if args.command == "setup-ane":
        return _cmd_setup_ane(args)
    if args.command == "embed":
        return run_embed_command(args, _make_engine)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "parquet":
        if getattr(args, "context_tokens", None):
            eng = _make_engine(args)
            try:
                summary = embed_contextual_parquet(
                    eng,
                    args.input,
                    args.output,
                    text_col=args.text_col,
                    keep_cols=args.keep_cols,
                    drop_text_col=args.drop_text_col,
                    chunk_tokens=args.chunk_tokens,
                    context_tokens=args.context_tokens,
                    overlap=args.overlap,
                    mode=args.mode,
                    dtype=args.dtype,
                    batch_rows=args.batch_rows,
                    filter_ocr=args.filter_ocr,
                    ocr_pages_only=args.ocr_pages_only,
                    dimensions=getattr(args, "dimensions", None),
                    compression=args.compression,
                    limit=args.limit,
                )
                print(
                    f"DONE (contextual): files={summary.input_files} rows_read={summary.rows_read:,} "
                    f"chunks_emitted={summary.rows_emitted:,} tokens={summary.total_tokens:,} "
                    f"elapsed={summary.elapsed_seconds:.2f}s -> {summary.output_path}"
                )
                return 0
            finally:
                eng.close()
        return run_parquet_command(args, _make_engine)
    if args.command == "serve":
        return run_serve_command(args, _make_engine, label="fast-pplx")
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
