"""CLI entrypoints for fast-embeddinggemma-2 (shared commands via `fast_shared.cli_common`)."""

from __future__ import annotations

import argparse
import subprocess
import sys
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

from .engine import FastEmbeddingGemma2
from .paths import resolve_paths

_MODES = ("hetero2", "hetero", "gpu", "ane", "cpu")


def _make_engine(args: argparse.Namespace) -> FastEmbeddingGemma2:
    warmup = not getattr(args, "no_warmup", False)
    return FastEmbeddingGemma2(
        home=args.home,
        default_mode=getattr(args, "mode", "hetero2"),
        warmup=warmup,
        dimensions=getattr(args, "dimensions", None),
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

    model_dir = resolve_hf_model_dir(
        args.model_dir, "mlx-community/embeddinggemma-2-bf16", local_files_only=args.local_files_only
    )
    copy_tokenizer_files(model_dir, paths.tokenizer_dir)
    print(f"Tokenizer copied; run scripts/export_weights.py with {model_dir} to populate {paths.weights_npz}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    eng = FastEmbeddingGemma2(home=args.home, warmup=True)
    print(
        f"home={paths.home} ane_available={eng.ane_available} ane_buckets={sorted(eng.ane_handles.keys())} "
        f"cpu_sme_available={eng.cpu_sme_available} cpu_buckets={sorted(eng.cpu_handles.keys())}"
    )

    fix = paths.home / "tests" / "fixtures" / "ref_vecs_64.npz"
    if fix.exists():
        ref = np.load(fix)
        lens = ref["lens"]
        sub_lists = [ref["ids"][i, : int(lens[i])].tolist() for i in range(len(lens))]
        v_gpu = eng.encode_corpus(sub_lists, mode="gpu")
        v_het = eng.encode_corpus(sub_lists, mode="hetero")
        cos_f16 = np.sum(v_gpu * ref["v_f16"], axis=1)
        cos_bf16 = np.sum(v_gpu * ref["v_stock_bf16"], axis=1)
        cos_het = np.sum(v_het * ref["v_f16"], axis=1)
        print(
            f"{len(sub_lists)}-doc ref parity: GPU_vs_F16 min={cos_f16.min():.6f} mean={cos_f16.mean():.6f} | "
            f"GPU_vs_BF16 min={cos_bf16.min():.6f} | Hetero_vs_F16 min={cos_het.min():.6f}"
        )
        if eng.ane_available:
            q_gpu = eng.encode_query(sub_lists[0][:24], prefer_ane=False)
            q_ane = eng.encode_query(sub_lists[0][:24], prefer_ane=True)
            print(f"Single-query B=1 S=24 ANE_vs_GPU cos={float(np.dot(q_ane, q_gpu)):.6f}")
    else:
        v1 = eng.encode_texts("renewable grid reliability technical report", mode="gpu")
        v2 = eng.encode_texts("renewable grid reliability technical report", mode="ane" if eng.ane_available else "gpu")
        print(f"Smoke parity cos={float(np.dot(v1, v2)):.6f}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = FastEmbeddingGemma2(home=args.home, warmup=True, dimensions=getattr(args, "dimensions", None))

    if args.parquet:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(args.parquet)
        col = resolve_text_column(pf.schema_arrow, args.text_col)
        texts = pq.read_table(args.parquet, columns=[col]).column(col).to_pylist()[: args.n]
        t0 = time.perf_counter()
        tok_lists = eng.tokenizer.encode_batch([t or "" for t in texts], add_document_prefix=True, max_length=eng.max_seq_len)
        dt_tok = time.perf_counter() - t0
        print(f"Tokenized {len(tok_lists):,} docs in {dt_tok * 1e3:.1f}ms")
    else:
        rng = np.random.default_rng(42)
        lens = np.clip(rng.integers(48, 505, size=args.n), 16, 512)
        tok_lists = [
            [eng.tokenizer.bos_id]
            + rng.integers(100, 200000, size=int(l) - 2, dtype=np.int32).tolist()
            + [eng.tokenizer.eos_id]
            for l in lens
        ]

    tot_toks = sum(len(x) for x in tok_lists)
    q_ids = tok_lists[0][:24] if len(tok_lists[0]) >= 24 else ([eng.tokenizer.bos_id] * 24)

    for label, fn in [
        ("ANE_B1_S32", lambda: eng.encode_query(q_ids, prefer_ane=True)),
        ("GPU_B1_S64", lambda: eng.encode_query(q_ids, prefer_ane=False)),
    ]:
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"Query [{label}]: median={np.median(ts):.3f}ms min={min(ts):.3f}ms")

    if args.mode == "all":
        modes: tuple[str, ...] = ("gpu",)
        if eng.ane_available:
            modes = modes + ("ane", "hetero2")
        if eng.ane_available and eng.cpu_sme_available:
            modes = modes + ("hetero",)
    else:
        modes = (args.mode,)

    for m in modes:
        eng.encode_corpus(tok_lists, mode=m)  # compile & warm all batch shapes for this mode
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            eng.encode_corpus(tok_lists, mode=m)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        st = eng.last_stats or {}
        print(
            f"Corpus n={len(tok_lists):,} toks={tot_toks:,} [mode={m}]: "
            f"{med:.3f}s -> {tot_toks / med:,.0f} tok/s ({len(tok_lists) / med:.1f} doc/s) | split={st}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="embeddinggemma-2", description="Apple Silicon M4 Pro fast-embeddinggemma-2 CLI")
    p.add_argument("--home", default=None, help="Override FAST_EMBEDDINGGEMMA2_HOME directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Export embeddinggemma_2_f16.npz + tokenizer into artifacts/")
    sw.add_argument("--model-dir", default=None)
    sw.add_argument("--local-files-only", action="store_true")
    sw.set_defaults(func=_cmd_setup_weights)

    em = sub.add_parser("embed", help="Embed text arguments or --stdin lines")
    add_embed_arguments(em, mode_choices=_MODES, default_mode="hetero2", allow_doc_prefix=True, allow_dimensions=True)
    em.set_defaults(func=lambda a: run_embed_command(a, _make_engine))

    pq_p = sub.add_parser("parquet", help="Stream-embed a Parquet file, directory, or glob")
    add_parquet_arguments(
        pq_p,
        mode_choices=_MODES,
        default_mode="hetero2",
        default_max_body_tokens=None,
        allow_doc_prefix=True,
        allow_dimensions=True,
    )
    pq_p.set_defaults(func=lambda a: run_parquet_command(a, _make_engine))

    sv = sub.add_parser("serve", help="Run persistent OpenAI-compatible HTTP server (/v1/embeddings)")
    add_serve_arguments(sv, mode_choices=_MODES, default_mode="hetero2", allow_dimensions=True)
    sv.set_defaults(func=lambda a: run_serve_command(a, _make_engine, label="fast-embeddinggemma-2"))

    vf = sub.add_parser("verify", help="Verify ANE/CPU-SME/GPU parity and installation")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark query latency and corpus throughput")
    bn.add_argument("--parquet", default=None)
    bn.add_argument("--text-col", default=None)
    bn.add_argument("--n", type=int, default=2000)
    bn.add_argument("--runs", type=int, default=2)
    bn.add_argument("--dimensions", type=int, default=None)
    bn.add_argument("--mode", choices=("all", "hetero", "hetero2", "gpu", "ane", "cpu"), default="all")
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
