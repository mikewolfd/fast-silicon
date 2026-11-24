"""CLI entrypoints for fast-embeddinggemma-2 (shared commands via `fast_shared.cli_common`)."""

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

from .engine import FastEmbeddingGemma2, probe_cpu_topology
from .gpu import probe_metal_capabilities
from .paths import resolve_paths

# Canonical mode naming aligned with fast-bge-small:
#   hetero (alias hetero2) = 2-way GPU + ANE; hetero3 = explicit opt-in 3-way GPU + ANE + CPU-SME.
_MODES = ("hetero", "hetero2", "hetero3", "gpu", "ane", "cpu")


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
    eng: FastEmbeddingGemma2,
    sub_lists: list[list[int]],
    v_mixed: np.ndarray,
    v_gpu: np.ndarray,
    *,
    max_outliers: int = 8,
) -> dict[str, Any]:
    """Replay worst mixed-vs-GPU cosine outliers on fixed GPU/ANE/CPU to separate scheduling handoffs from math drift."""
    cos_mix = np.sum(v_mixed * v_gpu, axis=1)
    k = min(max_outliers, len(sub_lists))
    outlier_idxs = np.argsort(cos_mix)[:k].tolist()
    out_seqs = [sub_lists[i] for i in outlier_idxs]
    rep_gpu = eng.encode_corpus(out_seqs, mode="gpu")

    # Only replay <=512-token outliers on ANE/CPU (CoreML bundles cap at 512 tokens; >512 runs GPU-only).
    le512_pos = [j for j, seq in enumerate(out_seqs) if len(seq) <= eng.ane_query_max_len]
    le512_seqs = [out_seqs[j] for j in le512_pos]
    rep_ane_map: dict[int, np.ndarray] = {}
    rep_cpu_map: dict[int, np.ndarray] = {}
    if le512_seqs and eng.ane_available:
        rep_ane = eng.encode_corpus(le512_seqs, mode="ane")
        for idx_pos, vec in zip(le512_pos, rep_ane):
            rep_ane_map[idx_pos] = vec
    if le512_seqs and eng.cpu_sme_available:
        rep_cpu = eng.encode_corpus(le512_seqs, mode="cpu")
        for idx_pos, vec in zip(le512_pos, rep_cpu):
            rep_cpu_map[idx_pos] = vec

    cos_rep_gpu = np.sum(rep_gpu * v_gpu[outlier_idxs], axis=1)
    cos_mix_vs_gpu = np.sum(v_mixed[outlier_idxs] * rep_gpu, axis=1)
    cos_ane_vals: list[float] = []

    closest_counts = {"gpu": 0, "ane": 0, "cpu": 0}
    for j in range(k):
        best_proc = "gpu"
        best_val = float(cos_mix_vs_gpu[j])
        if j in rep_ane_map:
            c_ane = float(np.dot(v_mixed[outlier_idxs[j]], rep_ane_map[j]))
            cos_ane_vals.append(c_ane)
            if c_ane > best_val:
                best_proc = "ane"
                best_val = c_ane
        if j in rep_cpu_map:
            c_cpu = float(np.dot(v_mixed[outlier_idxs[j]], rep_cpu_map[j]))
            if c_cpu > best_val:
                best_proc = "cpu"
        closest_counts[best_proc] += 1

    return {
        "outlier_idxs": outlier_idxs,
        "mixed_min_cos": float(cos_mix.min()),
        "replay_fixed_gpu_min_cos": float(cos_rep_gpu.min()),
        "replay_mixed_vs_ane_min_cos": min(cos_ane_vals) if cos_ane_vals else None,
        "closest_proc": closest_counts,
    }


def _format_telemetry(caps: dict[str, Any], topo: dict[str, Any]) -> str:
    p1 = (
        f"+{topo.get('perflevel1_cores')}{topo.get('perflevel1_name')}"
        if topo.get("perflevel1_cores")
        else ""
    )
    return (
        f"device={caps.get('device_name') or 'unknown'} arch={caps.get('architecture') or 'unknown'} "
        f"Apple9={caps.get('apple9')} Apple10={caps.get('apple10')} Metal4={caps.get('metal4')} "
        f"sparse={caps.get('supports_placement_sparse')} "
        f"cpu={topo.get('perflevel0_cores')}{topo.get('perflevel0_name')}{p1} "
        f"has_ecores={topo.get('has_efficiency_cores')} "
        f"macOS={caps.get('macos_version_str')}({caps.get('macos_build')}) "
        f"mlx={caps.get('mlx_version_str')} MLX_ENABLE_TF32={caps.get('mlx_enable_tf32')}"
    )


def _make_engine(args: argparse.Namespace) -> FastEmbeddingGemma2:
    warmup = not getattr(args, "no_warmup", False)
    mode = getattr(args, "mode", "hetero")
    return FastEmbeddingGemma2(
        home=args.home,
        default_mode=mode,
        warmup=warmup,
        enable_cpu_sme=(mode in ("cpu", "hetero3")),
        dimensions=getattr(args, "dimensions", None),
    )


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
    caps = probe_metal_capabilities()
    topo = probe_cpu_topology()
    print(f"[Telemetry] {_format_telemetry(caps, topo)}")

    strict = bool(getattr(args, "strict", False))
    failures: list[str] = []

    with FastEmbeddingGemma2(home=args.home, warmup=True, enable_cpu_sme=True, verify_ane_plan=True) as eng:
        print(
            f"home={paths.home} ane_available={eng.ane_available} ane_buckets={sorted(eng.ane_handles.keys())} "
            f"cpu_sme_available={eng.cpu_sme_available} cpu_buckets={sorted(eng.cpu_handles.keys())}"
        )
        if eng.ane_available:
            plans = eng.ane_plan_stats or eng.inspect_ane_plans(warn_on_cpu_fallback=False)
            for seq_len, st in sorted(plans.items()):
                print(f"  MLComputePlan S={seq_len}: ANE={st['ane_ops']} GPU={st['gpu_ops']} CPU={st['cpu_ops']}")
                if st["ane_ops"] == 0 or st["cpu_ops"] > 0:
                    failures.append(f"ANE plan S={seq_len} unexpected placement: {st}")

        fix = paths.home / "tests" / "fixtures" / "ref_vecs_64.npz"
        if fix.exists():
            ref = np.load(fix)
            lens = ref["lens"]
            sub_lists = [ref["ids"][i, : int(lens[i])].tolist() for i in range(len(lens))]
            le512_idxs = [i for i, seq in enumerate(sub_lists) if len(seq) <= eng.ane_query_max_len]
            sub_lists_le512 = [sub_lists[i] for i in le512_idxs]

            v_gpu = eng.encode_corpus(sub_lists, mode="gpu")
            v_het = eng.encode_corpus(sub_lists, mode="hetero")
            v_ane = eng.encode_corpus(sub_lists_le512, mode="ane") if (eng.ane_available and sub_lists_le512) else None
            v_cpu = eng.encode_corpus(sub_lists_le512, mode="cpu") if (eng.cpu_sme_available and sub_lists_le512) else None

            for label, arr in (("gpu", v_gpu), ("ane", v_ane), ("cpu", v_cpu), ("hetero", v_het)):
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
                f"{len(sub_lists)}-doc ref parity: GPU_vs_F16 min={cos_f16.min():.6f} mean={cos_f16.mean():.6f} | "
                f"GPU_vs_BF16 min={cos_bf16.min():.6f} | Hetero_vs_F16 min={cos_het.min():.6f}"
            )
            if v_ane is not None:
                cos_ane_gpu = np.sum(v_ane * v_gpu[le512_idxs], axis=1)
                line += f" | ANE_vs_GPU({len(le512_idxs)}<=512) min={cos_ane_gpu.min():.6f}"
                if float(cos_ane_gpu.min()) < 0.9990:
                    failures.append(f"ANE_vs_GPU min cos {cos_ane_gpu.min():.6f} < 0.9990")
            if v_cpu is not None:
                cos_cpu_gpu = np.sum(v_cpu * v_gpu[le512_idxs], axis=1)
                line += f" | CPU_vs_GPU({len(le512_idxs)}<=512) min={cos_cpu_gpu.min():.6f}"
                if float(cos_cpu_gpu.min()) < 0.9990:
                    failures.append(f"CPU_vs_GPU min cos {cos_cpu_gpu.min():.6f} < 0.9990")
            print(line)

            if float(cos_f16.min()) < 0.9998:
                failures.append(f"GPU_vs_F16 min cos {cos_f16.min():.6f} < 0.9998")

            for label, arr, ref_sub in (
                ("gpu", v_gpu, ref["v_f16"]),
                ("ane", v_ane, ref["v_f16"][le512_idxs]),
                ("cpu", v_cpu, ref["v_f16"][le512_idxs]),
                ("hetero", v_het, ref["v_f16"]),
            ):
                if arr is None:
                    continue
                rm = _rank_metrics(arr, ref_sub, top_k=5)
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

            if eng.ane_available:
                q_gpu = eng.encode_query(sub_lists[0][:24], prefer_ane=False)
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
    topo = probe_cpu_topology()
    print(f"[Telemetry] {_format_telemetry(caps, topo)}")
    warns = check_system_quiescence()
    if warns:
        print(f"[Quiescence WARNING] {len(warns)} contender(s) detected:", file=sys.stderr)
        for w in warns[:5]:
            print(f"  {w}", file=sys.stderr)
        if getattr(args, "check_ps", False):
            return 2

    want_cpu = args.mode in ("cpu", "hetero3")
    with FastEmbeddingGemma2(
        home=args.home,
        warmup=True,
        enable_cpu_sme=want_cpu,
        dimensions=getattr(args, "dimensions", None),
    ) as eng:
        if args.parquet:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(args.parquet)
            col = resolve_text_column(pf.schema_arrow, args.text_col)
            texts = pq.read_table(args.parquet, columns=[col]).column(col).to_pylist()[: args.n]
            t0 = time.perf_counter()
            tok_lists = eng.tokenizer.encode_batch(
                [t or "" for t in texts],
                add_document_prefix=bool(getattr(args, "document_prefix", False)),
                max_length=eng.max_seq_len,
            )
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

        query_arms = [("GPU_B1_S64", lambda: eng.encode_query(q_ids, prefer_ane=False))]
        if eng.ane_available:
            query_arms.insert(0, ("ANE_B1_S32", lambda: eng.encode_query(q_ids, prefer_ane=True)))
        for label, fn in query_arms:
            for _ in range(3):
                fn()
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e3)
            print(f"Query [{label}]: median={np.median(ts):.3f}ms min={min(ts):.3f}ms")

        runs = max(1, int(args.runs))
        if args.mode == "all" and eng.ane_available:
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
            eng.encode_corpus(tok_lists, mode=m)  # compile & warm all batch shapes for this mode
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


def _cmd_embed_images(args: argparse.Namespace) -> int:
    import json

    eng = _make_engine(args)
    try:
        vecs = eng.encode_images(
            args.images,
            max_soft_tokens=args.max_soft_tokens,
            batch_size=args.batch_size,
            dimensions=args.dimensions,
            vision_model_dir=args.vision_model_dir,
        )
        if args.json:
            print(json.dumps({"count": int(vecs.shape[0]), "dim": int(vecs.shape[1]), "embeddings": vecs.tolist()}))
        else:
            for path, row in zip(args.images, vecs):
                preview = " ".join(f"{float(v):.4f}" for v in row[:8])
                print(f"{path}\tdim={len(row)}\t[{preview} ...]")
        return 0
    finally:
        eng.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="embeddinggemma-2", description="Apple Silicon fast-embeddinggemma-2 CLI")
    p.add_argument("--home", default=None, help="Override FAST_EMBEDDINGGEMMA2_HOME directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("setup-weights", help="Download prebuilt ANE/FP16 artifacts (or export from source with --from-source)")
    sw.add_argument("--model-dir", default=None)
    sw.add_argument("--local-files-only", action="store_true")
    sw.add_argument("--from-source", action="store_true", help="Export weights locally from HF base checkpoint instead of pulling prebuilt bundles")
    sw.add_argument("--skip-ane", action="store_true", help="Pull only weights + tokenizer without Core ML ANE bundles")
    sw.set_defaults(func=_cmd_setup_weights)

    em = sub.add_parser("embed", help="Embed text arguments or --stdin lines")
    add_embed_arguments(em, mode_choices=_MODES, default_mode="hetero", allow_doc_prefix=True, allow_dimensions=True)
    em.set_defaults(func=lambda a: run_embed_command(a, _make_engine))

    ei = sub.add_parser("embed-images", help="Embed image files using lazy vision tower")
    ei.add_argument("images", nargs="+", help="Image file paths to encode")
    ei.add_argument("--vision-model-dir", default=None, help="Override HF/MLX directory with vision weights")
    ei.add_argument("--max-soft-tokens", type=int, default=None, help="Soft vision token budget per image (default: 140; supports 70/140/280/560/1120)")
    ei.add_argument("--batch-size", type=int, default=8, help="Batch size for ViT and trunk execution")
    ei.add_argument("--dimensions", type=int, default=None, help="Output MRL embedding dimension")
    ei.add_argument("--mode", choices=_MODES, default="gpu")
    ei.add_argument("--no-warmup", action="store_true", help="Skip text shape warmup before encoding images")
    ei.add_argument("--json", action="store_true", help="Emit full JSON output")
    ei.set_defaults(func=_cmd_embed_images)

    pq_p = sub.add_parser("parquet", help="Stream-embed a Parquet file, directory, or glob")
    add_parquet_arguments(
        pq_p,
        mode_choices=_MODES,
        default_mode="hetero",
        default_max_body_tokens=None,
        allow_doc_prefix=True,
        allow_dimensions=True,
    )
    pq_p.set_defaults(func=lambda a: run_parquet_command(a, _make_engine))

    sv = sub.add_parser("serve", help="Run persistent OpenAI-compatible HTTP server (/v1/embeddings)")
    add_serve_arguments(sv, mode_choices=_MODES, default_mode="hetero", allow_dimensions=True)
    sv.set_defaults(func=lambda a: run_serve_command(a, _make_engine, label="fast-embeddinggemma-2"))

    vf = sub.add_parser("verify", help="Verify ANE/CPU-SME/GPU parity, rank ordering, MLComputePlan, and outlier replay")
    vf.add_argument("--strict", action="store_true", help="Exit non-zero if any parity, rank, or ANE-plan check fails")
    vf.set_defaults(func=_cmd_verify)

    bn = sub.add_parser("bench", help="Benchmark query latency and corpus throughput (with ABBA)")
    bn.add_argument("--parquet", default=None)
    bn.add_argument("--text-col", default=None)
    bn.add_argument("--document-prefix", action="store_true", help="Prepend DOCUMENT_PREFIX when benchmarking --parquet")
    bn.add_argument("--n", type=int, default=2000)
    bn.add_argument("--runs", type=int, default=2)
    bn.add_argument("--dimensions", type=int, default=None)
    bn.add_argument("--check-ps", action="store_true", help="Fail if heavy GPU/CPU contenders are active")
    bn.add_argument("--mode", choices=("all", "hetero", "hetero2", "hetero3", "gpu", "ane", "cpu"), default="all")
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
