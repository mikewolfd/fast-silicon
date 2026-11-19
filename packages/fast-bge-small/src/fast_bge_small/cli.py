"""CLI entrypoints for fast-bge-small (shared commands via `fast_shared.cli_common`)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import time

import numpy as np

from fast_shared.cli_common import (
    add_embed_arguments,
    add_parquet_arguments,
    add_serve_arguments,
    run_embed_command,
    run_parquet_command,
    run_serve_command,
)
from fast_shared.hf import copy_tokenizer_files, resolve_hf_model_dir
from fast_shared.parquet_pipeline import resolve_text_column

from .engine import FastBgeSmall
from .paths import resolve_paths

_MODES = ("hetero", "gpu", "ane")


def _make_engine(args: argparse.Namespace) -> FastBgeSmall:
    warmup = not getattr(args, "no_warmup", False)
    return FastBgeSmall(home=args.home, default_mode=getattr(args, "mode", "hetero"), warmup=warmup)


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    script = paths.home / "scripts" / "export_weights.py"
    if not script.exists():
        import mlx.core as mx

        model_dir = resolve_hf_model_dir(
            args.model_dir, "mlx-community/bge-small-en-v1.5-bf16", local_files_only=args.local_files_only
        )
        paths.weights_npz.parent.mkdir(parents=True, exist_ok=True)
        raw_mx = mx.load(str(model_dir / "model.safetensors"))
        np_f16 = {
            k: np.array(v.astype(mx.float16), dtype=np.float16)
            for k, v in sorted(raw_mx.items())
            if not k.startswith("pooler.") and "position_ids" not in k
        }
        np.savez(paths.weights_npz, **np_f16)
        copy_tokenizer_files(model_dir, paths.tokenizer_dir)
        print(f"Exported weights to {paths.weights_npz}")
        return 0

    cmd = [sys.executable, str(script), "--home", str(paths.home)]
    if args.model_dir:
        cmd.extend(["--model-dir", str(args.model_dir)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    return subprocess.call(cmd)


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    eng = FastBgeSmall(home=args.home, warmup=True)
    print(f"home={paths.home} ane_available={eng.ane_available} buckets={sorted(eng.buckets.keys())}")

    fix = paths.home / "tests" / "fixtures" / "ref_vecs_64.npz"
    if fix.exists():
        ref = np.load(fix)
        sub_lists = [ref["ids"][i, : int(ref["mask"][i].sum())].tolist() for i in range(64)]
        v_gpu = eng.encode_corpus(sub_lists, mode="gpu")
        v_het = eng.encode_corpus(sub_lists, mode="hetero")
        cos_f16 = np.sum(v_gpu * ref["v_f16"], axis=1)
        cos_bf16 = np.sum(v_gpu * ref["v_stock_bf16"], axis=1)
        cos_het = np.sum(v_het * ref["v_f16"], axis=1)
        print(
            f"64-doc ref parity: GPU_vs_F16 min={cos_f16.min():.6f} mean={cos_f16.mean():.6f} | "
            f"GPU_vs_BF16 min={cos_bf16.min():.6f} | Hetero_vs_F16 min={cos_het.min():.6f}"
        )
        q_gpu = eng.gpu.encode_single(sub_lists[0][:24])
        q_ane = eng.encode_query(sub_lists[0][:24], prefer_ane=True)
        print(f"Single-query B=1 S=24 ANE_vs_GPU cos={float(np.dot(q_ane, q_gpu)):.6f}")
    else:
        v1 = eng.encode_texts("renewable grid reliability technical report", mode="gpu")
        v2 = eng.encode_texts("renewable grid reliability technical report", mode="ane" if eng.ane_available else "gpu")
        print(f"Smoke parity cos={float(np.dot(v1, v2)):.6f}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = FastBgeSmall(home=args.home, warmup=True)

    if args.parquet:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(args.parquet)
        col = resolve_text_column(pf.schema_arrow, args.text_col)
        texts = pq.read_table(args.parquet, columns=[col]).column(col).to_pylist()[: args.n]
        tok_lists = eng.tokenizer.encode_batch([t or "" for t in texts], max_length=512)
    else:
        rng = np.random.default_rng(42)
        lens = np.clip(rng.integers(32, 465, size=args.n), 16, 512)
        tok_lists = [
            [101] + rng.integers(1000, 29000, size=int(l) - 2, dtype=np.int32).tolist() + [102]
            for l in lens
        ]

    tot_toks = sum(len(x) for x in tok_lists)
    q_ids = tok_lists[0][:24] if len(tok_lists[0]) >= 24 else [101] * 24

    for label, fn in [
        ("ANE_B1_S24", lambda: eng.encode_query(q_ids, prefer_ane=True)),
        ("GPU_B1_S24", lambda: eng.encode_query(q_ids, prefer_ane=False)),
    ]:
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"Query [{label}]: median={np.median(ts):.3f}ms min={min(ts):.3f}ms")

    modes = ("gpu", "ane", "hetero") if (args.mode == "all" and eng.ane_available) else ((args.mode,) if args.mode != "all" else ("gpu",))
    for m in modes:
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            eng.encode_corpus(tok_lists, mode=m)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        st = eng.last_stats or {}
        print(
            f"Corpus n={len(tok_lists):,} toks={tot_toks:,} [mode={m}]: "
            f"{med:.3f}s -> {tot_toks/med:,.0f} tok/s | split={st}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bge-small", description="Apple Silicon M4 Pro fast-bge-small CLI")
    p.add_argument("--home", default=None, help="Override FAST_BGE_SMALL_HOME directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Export bge_small_f16.npz + tokenizer into artifacts/")
    sw.add_argument("--model-dir", default=None)
    sw.add_argument("--local-files-only", action="store_true")
    sw.set_defaults(func=_cmd_setup_weights)

    em = sub.add_parser("embed", help="Embed text arguments or --stdin lines")
    add_embed_arguments(em, mode_choices=_MODES, default_mode="hetero", allow_doc_prefix=False)
    em.set_defaults(func=lambda a: run_embed_command(a, _make_engine))

    pq_p = sub.add_parser("parquet", help="Stream-embed a Parquet file, directory, or glob")
    add_parquet_arguments(pq_p, mode_choices=_MODES, default_mode="hetero", default_max_body_tokens=510, allow_doc_prefix=False)
    pq_p.set_defaults(func=lambda a: run_parquet_command(a, _make_engine))

    sv = sub.add_parser("serve", help="Run persistent OpenAI-compatible HTTP server (/v1/embeddings)")
    add_serve_arguments(sv, mode_choices=_MODES, default_mode="hetero")
    sv.set_defaults(func=lambda a: run_serve_command(a, _make_engine, label="fast-bge-small"))

    vf = sub.add_parser("verify", help="Verify ANE/GPU parity and installation")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark query latency and corpus throughput")
    bn.add_argument("--parquet", default=None)
    bn.add_argument("--text-col", default=None)
    bn.add_argument("--n", type=int, default=4000)
    bn.add_argument("--mode", choices=("all", "hetero", "gpu", "ane"), default="all")
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
