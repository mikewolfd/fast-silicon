"""CLI entrypoints for fast-bge-small (shared commands via `fast_shared.cli_common`)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np

from fast_shared.bench import (
    check_system_quiescence,
    clear_mlx_arena,
    interleaved_abba,
    peak_footprint_gb,
)
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
from .gpu import probe_metal_capabilities
from .paths import resolve_paths

_MODES = ("hetero", "hetero2", "gpu", "ane")


def _rank_metrics(vecs: np.ndarray, ref: np.ndarray, *, top_k: int = 5) -> dict[str, float]:
    """Compare retrieval rank ordering (Top-1, Top-K overlap, and mean Spearman rho) against reference."""
    n = int(vecs.shape[0])
    sim_v = vecs.astype(np.float64) @ vecs.astype(np.float64).T
    sim_r = ref.astype(np.float64) @ ref.astype(np.float64).T
    top1_hits = 0
    topk_hits = 0
    rhos: list[float] = []
    k = min(top_k, n)
    for i in range(n):
        ord_v = np.argsort(-sim_v[i], kind="mergesort")
        ord_r = np.argsort(-sim_r[i], kind="mergesort")
        if int(ord_v[0]) == int(ord_r[0]):
            top1_hits += 1
        topk_hits += len(set(ord_v[:k].tolist()) & set(ord_r[:k].tolist()))
        rv = np.empty(n, dtype=np.float64)
        rr = np.empty(n, dtype=np.float64)
        rv[ord_v] = np.arange(n, dtype=np.float64)
        rr[ord_r] = np.arange(n, dtype=np.float64)
        rv -= rv.mean()
        rr -= rr.mean()
        denom = float(np.linalg.norm(rv) * np.linalg.norm(rr))
        rhos.append(float(np.dot(rv, rr) / denom) if denom > 0.0 else 1.0)
    return {
        "top1_match": top1_hits / max(1, n),
        "topk_overlap": topk_hits / max(1, n * k),
        "spearman_mean": float(np.mean(rhos)) if rhos else 1.0,
        "spearman_min": float(np.min(rhos)) if rhos else 1.0,
    }


def _replay_hetero_outliers(
    eng: FastBgeSmall,
    sub_lists: list[list[int]],
    v_mixed: np.ndarray,
    v_gpu: np.ndarray,
    *,
    max_outliers: int = 8,
) -> dict[str, Any]:
    """Replay worst mixed-vs-GPU cosine outliers on isolated GPU and ANE to separate handoffs from math drift."""
    cos_mix = np.sum(v_mixed * v_gpu, axis=1)
    k = min(max_outliers, len(sub_lists))
    outlier_idxs = np.argsort(cos_mix)[:k].tolist()
    out_seqs = [sub_lists[i] for i in outlier_idxs]
    rep_gpu = eng.encode_corpus(out_seqs, mode="gpu")
    rep_ane = eng.encode_corpus(out_seqs, mode="ane") if eng.ane_available else None

    cos_rep_gpu = np.sum(rep_gpu * v_gpu[outlier_idxs], axis=1)
    cos_mix_vs_ane = np.sum(v_mixed[outlier_idxs] * rep_ane, axis=1) if rep_ane is not None else None
    cos_mix_vs_gpu = np.sum(v_mixed[outlier_idxs] * rep_gpu, axis=1)
    closest_counts = {"gpu": 0, "ane": 0}
    for j in range(k):
        if cos_mix_vs_ane is not None and float(cos_mix_vs_ane[j]) > float(cos_mix_vs_gpu[j]):
            closest_counts["ane"] += 1
        else:
            closest_counts["gpu"] += 1

    return {
        "outlier_idxs": outlier_idxs,
        "mixed_min_cos": float(cos_mix.min()),
        "replay_fixed_gpu_min_cos": float(cos_rep_gpu.min()),
        "replay_mixed_vs_ane_min_cos": float(cos_mix_vs_ane.min()) if cos_mix_vs_ane is not None else None,
        "closest_proc": closest_counts,
    }


def _format_telemetry(caps: dict[str, Any]) -> str:
    return (
        f"device={caps.get('device_name') or 'unknown'} arch={caps.get('architecture') or 'unknown'} "
        f"Apple9={caps.get('apple9')} Apple10={caps.get('apple10')} Metal4={caps.get('metal4')} "
        f"sparse={caps.get('supports_placement_sparse')} "
        f"macOS={caps.get('macos_version_str')}({caps.get('macos_build')}) "
        f"mlx={caps.get('mlx_version_str')} MLX_ENABLE_TF32={caps.get('mlx_enable_tf32')}"
    )


def _make_engine(args: argparse.Namespace) -> FastBgeSmall:
    warmup = not getattr(args, "no_warmup", False)
    return FastBgeSmall(home=args.home, default_mode=getattr(args, "mode", "hetero"), warmup=warmup)


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    from_source = bool(getattr(args, "from_source", False))
    if not from_source and not args.model_dir:
        from .paths import ensure_prebuilt_artifacts

        ensure_prebuilt_artifacts(
            args.home,
            local_files_only=args.local_files_only,
            include_ane=not bool(getattr(args, "skip_ane", False)),
        )
        print(f"Materialized prebuilt artifacts into {paths.home / 'artifacts'}")
        return 0

    script = paths.home / "scripts" / "export_weights.py"
    if not script.exists():
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
    caps = probe_metal_capabilities()
    print(f"[Telemetry] {_format_telemetry(caps)}")

    strict = bool(getattr(args, "strict", False))
    failures: list[str] = []

    with FastBgeSmall(home=args.home, warmup=True, verify_ane_plan=True) as eng:
        print(
            f"home={paths.home} ane_available={eng.ane_available} "
            f"buckets={sorted(eng.buckets.keys())} query_buckets={sorted(eng.query_buckets.keys())}"
        )
        if eng.ane_available:
            plans = eng.ane_plan_stats or eng.inspect_ane_plans(warn_on_cpu_fallback=False)
            for cap, st in sorted(plans.items()):
                print(f"  MLComputePlan S={cap}: ANE={st['ane_ops']} GPU={st['gpu_ops']} CPU={st['cpu_ops']}")
                if st["ane_ops"] == 0 or st["cpu_ops"] > 0:
                    failures.append(f"ANE plan S={cap} unexpected placement: {st}")

        fix = paths.home / "tests" / "fixtures" / "ref_vecs_64.npz"
        if fix.exists():
            ref = np.load(fix)
            sub_lists = [ref["ids"][i, : int(ref["mask"][i].sum())].tolist() for i in range(64)]
            v_gpu = eng.encode_corpus(sub_lists, mode="gpu")
            v_het = eng.encode_corpus(sub_lists, mode="hetero")
            v_ane = eng.encode_corpus(sub_lists, mode="ane") if eng.ane_available else None

            for label, arr in (("gpu", v_gpu), ("hetero", v_het), ("ane", v_ane)):
                if arr is None:
                    continue
                if not np.all(np.isfinite(arr)):
                    failures.append(f"{label} produced non-finite embeddings")
                if not np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-4):
                    failures.append(f"{label} embeddings not unit-normalized")

            cos_f16 = np.sum(v_gpu * ref["v_f16"], axis=1)
            cos_bf16 = np.sum(v_gpu * ref["v_stock_bf16"], axis=1)
            cos_het = np.sum(v_het * ref["v_f16"], axis=1)
            line = (
                f"64-doc ref parity: GPU_vs_F16 min={cos_f16.min():.6f} mean={cos_f16.mean():.6f} | "
                f"GPU_vs_BF16 min={cos_bf16.min():.6f} | Hetero_vs_F16 min={cos_het.min():.6f}"
            )
            if v_ane is not None:
                cos_ane_gpu = np.sum(v_ane * v_gpu, axis=1)
                cos_ane_f16 = np.sum(v_ane * ref["v_f16"], axis=1)
                line += f" | ANE_vs_GPU min={cos_ane_gpu.min():.6f} | ANE_vs_F16 min={cos_ane_f16.min():.6f}"
                if float(cos_ane_gpu.min()) < 0.9995:
                    failures.append(f"ANE_vs_GPU min cos {cos_ane_gpu.min():.6f} < 0.9995")
            print(line)

            if float(cos_f16.min()) < 0.9998:
                failures.append(f"GPU_vs_F16 min cos {cos_f16.min():.6f} < 0.9998")

            for label, arr in (("gpu", v_gpu), ("ane", v_ane), ("hetero", v_het)):
                if arr is None:
                    continue
                rm = _rank_metrics(arr, ref["v_f16"], top_k=5)
                print(
                    f"  Rank check [{label}]: top1={rm['top1_match']:.3f} top5={rm['topk_overlap']:.3f} "
                    f"spearman_mean={rm['spearman_mean']:.6f} spearman_min={rm['spearman_min']:.6f}"
                )
                if rm["top1_match"] < 1.0 or rm["topk_overlap"] < 0.95 or rm["spearman_mean"] < 0.995:
                    failures.append(f"{label} rank check failed: {rm}")

            replay = _replay_hetero_outliers(eng, sub_lists, v_het, v_gpu)
            print(
                f"  Outlier replay [hetero vs gpu]: mixed_min={replay['mixed_min_cos']:.6f} "
                f"fixed_gpu_min={replay['replay_fixed_gpu_min_cos']:.6f} "
                f"closest_proc={replay['closest_proc']}"
            )
            if replay["replay_fixed_gpu_min_cos"] < 0.9998:
                failures.append(f"Fixed-GPU outlier replay min cos {replay['replay_fixed_gpu_min_cos']:.6f} < 0.9998")

            q_gpu = eng.gpu.encode_single(sub_lists[0][:24])
            q_ane = eng.encode_query(sub_lists[0][:24], prefer_ane=True)
            print(f"Single-query B=1 S=24 ANE_vs_GPU cos={float(np.dot(q_ane, q_gpu)):.6f}")
        else:
            v1 = eng.encode_texts("renewable grid reliability technical report", mode="gpu")
            v2 = eng.encode_texts("renewable grid reliability technical report", mode="ane" if eng.ane_available else "gpu")
            print(f"Smoke parity cos={float(np.dot(v1, v2)):.6f}")

        print(f"Unified memory peak_footprint={peak_footprint_gb():.2f} GB")

    if failures:
        for msg in failures:
            print(f"[VERIFY FAIL] {msg}", file=sys.stderr)
        return 1 if strict or any("non-finite" in m for m in failures) else 0
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    caps = probe_metal_capabilities()
    print(f"[Telemetry] {_format_telemetry(caps)}")
    warns = check_system_quiescence()
    if warns:
        print(f"[Quiescence WARNING] {len(warns)} contender(s) detected:", file=sys.stderr)
        for w in warns[:5]:
            print(f"  {w}", file=sys.stderr)
        if getattr(args, "check_ps", False):
            return 2

    with FastBgeSmall(home=args.home, warmup=True) as eng:
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

        query_arms = [("GPU_B1_S24", lambda: eng.encode_query(q_ids, prefer_ane=False))]
        if eng.ane_available:
            query_arms.insert(0, ("ANE_B1_S24", lambda: eng.encode_query(q_ids, prefer_ane=True)))
        for label, fn in query_arms:
            for _ in range(3):
                fn()  # warm query bucket/compiled single_fn before timing
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e3)
            print(f"Query [{label}]: median={np.median(ts):.3f}ms min={min(ts):.3f}ms")

        runs = max(1, int(getattr(args, "runs", 2)))
        if args.mode == "all" and eng.ane_available:
            # Paired interleaved A/B/B/A between gpu (A) and hetero (B) with full warmup
            def _run_mode(m: str) -> dict[str, Any]:
                t0 = time.perf_counter()
                eng.encode_corpus(tok_lists, mode=m)
                mx.synchronize()
                dt = time.perf_counter() - t0
                return {"wall_sec": dt, "tok_s": tot_toks / max(dt, 1e-9), "doc_s": len(tok_lists) / max(dt, 1e-9)}

            abba = interleaved_abba(
                lambda: _run_mode("gpu"),
                lambda: _run_mode("hetero"),
                warmup=True,
                rounds=max(2, runs),
            )
            a_med = abba["A_median"]["wall_sec"]
            b_med = abba["B_median"]["wall_sec"]
            a_drift = abba["A_runs"][-1]["wall_sec"] / max(abba["A_runs"][0]["wall_sec"], 1e-9)
            b_drift = abba["B_runs"][-1]["wall_sec"] / max(abba["B_runs"][0]["wall_sec"], 1e-9)
            print(
                f"ABBA Corpus n={len(tok_lists):,} toks={tot_toks:,} [gpu vs hetero]: "
                f"gpu={a_med:.3f}s ({tot_toks / a_med:,.0f} tok/s, drift={a_drift:.3f}x) | "
                f"hetero={b_med:.3f}s ({tot_toks / b_med:,.0f} tok/s, drift={b_drift:.3f}x) | "
                f"speedup={a_med / max(b_med, 1e-9):.3f}x | peak_footprint={peak_footprint_gb():.2f}GB"
            )
            # Standalone ANE benchmark after warming ANE buckets
            eng.encode_corpus(tok_lists, mode="ane")
            ts_ane = []
            for _ in range(runs):
                t0 = time.perf_counter()
                eng.encode_corpus(tok_lists, mode="ane")
                ts_ane.append(time.perf_counter() - t0)
            med_ane = float(np.median(ts_ane))
            print(
                f"Corpus n={len(tok_lists):,} toks={tot_toks:,} [mode=ane]: "
                f"{med_ane:.3f}s -> {tot_toks / med_ane:,.0f} tok/s ({len(tok_lists) / med_ane:.1f} doc/s) "
                f"| split={eng.last_stats}"
            )
        else:
            m = "gpu" if args.mode == "all" else args.mode
            eng.encode_corpus(tok_lists, mode=m)  # warm all active shapes for this mode
            mx.synchronize()
            clear_mlx_arena(256)
            ts = []
            peak_mlx = 0.0
            for _ in range(runs):
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                eng.encode_corpus(tok_lists, mode=m)
                mx.synchronize()
                ts.append(time.perf_counter() - t0)
                peak_mlx = max(peak_mlx, mx.get_peak_memory() / (1024**3))
                clear_mlx_arena(256)
            med = float(np.median(ts))
            st = eng.last_stats or {}
            print(
                f"Corpus n={len(tok_lists):,} toks={tot_toks:,} [mode={m}]: "
                f"{med:.3f}s -> {tot_toks / med:,.0f} tok/s ({len(tok_lists) / med:.1f} doc/s) "
                f"| peak_mlx={peak_mlx:.2f}GB peak_footprint={peak_footprint_gb():.2f}GB | split={st}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bge-small", description="Apple Silicon fast-bge-small CLI")
    p.add_argument("--home", default=None, help="Override FAST_BGE_SMALL_HOME directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Download prebuilt ANE/FP16 artifacts (or export from source with --from-source)")
    sw.add_argument("--model-dir", default=None)
    sw.add_argument("--local-files-only", action="store_true")
    sw.add_argument("--from-source", action="store_true", help="Export weights locally from HF base checkpoint instead of pulling prebuilt bundles")
    sw.add_argument("--skip-ane", action="store_true", help="Pull only weights + tokenizer without Core ML ANE bundles")
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

    vf = sub.add_parser("verify", help="Verify ANE/GPU parity, rank ordering, MLComputePlan, and outlier replay")
    vf.add_argument("--strict", action="store_true", help="Exit non-zero if any parity, rank, or ANE-plan check fails")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark query latency and corpus throughput (with ABBA)")
    bn.add_argument("--parquet", default=None)
    bn.add_argument("--text-col", default=None)
    bn.add_argument("--n", type=int, default=4000)
    bn.add_argument("--runs", type=int, default=2)
    bn.add_argument("--check-ps", action="store_true", help="Fail if heavy GPU/CPU contenders are active")
    bn.add_argument("--mode", choices=("all", "hetero", "hetero2", "gpu", "ane"), default="all")
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
