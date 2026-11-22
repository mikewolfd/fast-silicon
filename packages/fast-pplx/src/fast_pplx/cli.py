"""CLI entrypoints (`fast-pplx setup-weights|setup-ane|embed|verify|bench|parquet|serve`)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from fast_shared.bench import clear_mlx_arena, peak_footprint_gb
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
from .gpu import _find_local_hf_snapshot, get_mlx_runtime_telemetry
from .parquet_pipeline import embed_contextual_parquet, embed_parquet, resolve_text_column
from .paths import resolve_paths

_MODES: tuple[str, ...] = ("auto", "gpu", "ane", "hetero", "hetero2")
_BENCH_MODES: tuple[str, ...] = ("auto", "gpu", "ane", "hetero", "hetero2", "all")
_VERIFY_MODES: tuple[str, ...] = ("gpu", "ane", "hetero", "hetero2", "all")


def _add_common_engine_args(
    parser: argparse.ArgumentParser,
    *,
    include_ane_profile: bool = True,
) -> None:
    parser.add_argument("--home", default=None, help="Override FAST_PPLX_HOME directory")
    parser.add_argument(
        "--model-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Model activation weight dtype (default: float16)",
    )
    parser.add_argument(
        "--quantize-weights",
        choices=("none", "w8"),
        default="none",
        help="Optional Metal linear projection weight quantization: 'none' (default) or 'w8' (8-bit group_size=64)",
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
    if include_ane_profile:
        parser.add_argument(
            "--ane-profile",
            choices=("fast", "balanced", "parity", "custom"),
            default=None,
            help="Require staged ANE bundles matching this profile (default: accept any valid staged profile)",
        )


def _make_engine(args: argparse.Namespace, *, deterministic: bool = False) -> Any:
    from .engine import FastPplx

    req_mode = getattr(args, "mode", "auto")
    eff_mode = "auto" if req_mode == "all" else req_mode
    return FastPplx(
        home=getattr(args, "home", None),
        mode=eff_mode,
        dtype=getattr(args, "model_dtype", "float16"),
        quantize_weights=getattr(args, "quantize_weights", "none"),
        use_mapped_embedding=not getattr(args, "no_mapped_embedding", False),
        cache_limit_mb=getattr(args, "cache_limit_mb", 512),
        ane_profile=getattr(args, "ane_profile", None),
        deterministic=deterministic,
    )


def _cmd_setup_weights(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.home)
    script = paths.home / "scripts" / "export_weights.py"
    if not script.exists():
        pkg_script = Path(__file__).resolve().parents[2] / "scripts" / "export_weights.py"
        is_stub_dir = bool(
            args.model_dir
            and (Path(args.model_dir) / "model.safetensors").is_file()
            and (Path(args.model_dir) / "model.safetensors").stat().st_size < 64
        )
        if pkg_script.exists() and not is_stub_dir:
            script = pkg_script
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
        pkg_script = Path(__file__).resolve().parents[2] / "scripts" / "build_ane_models.py"
        if pkg_script.exists():
            script = pkg_script
        else:
            raise FileNotFoundError(
                f"Missing ANE export script {script}. Install full package scripts or build ANE artifacts first."
            )
    cmd = [sys.executable, str(script), "--home", str(paths.home)]
    if getattr(args, "model_dir", None):
        cmd.extend(["--model-dir", str(args.model_dir)])
    if getattr(args, "variant", None):
        cmd.extend(["--variant", str(args.variant)])
    if getattr(args, "backend", None):
        cmd.extend(["--backend", str(args.backend)])
    if getattr(args, "seq_lens", None):
        cmd.extend(["--seq-lens", *str(args.seq_lens).replace(",", " ").split()])
    if getattr(args, "precision", None):
        cmd.extend(["--precision", str(args.precision)])
    if getattr(args, "layers_per_stage", None) is not None:
        cmd.extend(["--layers-per-stage", str(args.layers_per_stage)])
    if getattr(args, "inv_mode", None):
        cmd.extend(["--inv-mode", str(args.inv_mode)])
    if getattr(args, "ane_profile", None):
        cmd.extend(["--ane-profile", str(args.ane_profile)])
    if getattr(args, "silu_mode", None):
        cmd.extend(["--silu-mode", str(args.silu_mode)])
    if getattr(args, "rmsnorm_mode", None):
        cmd.extend(["--rmsnorm-mode", str(args.rmsnorm_mode)])
    if getattr(args, "calib_stats", None):
        cmd.extend(["--calib-stats", str(args.calib_stats)])
    if getattr(args, "ftz_alpha", None) is not None:
        cmd.extend(["--ftz-alpha", str(args.ftz_alpha)])
    if getattr(args, "preserve_gdn_fp16", None) is False:
        cmd.append("--no-preserve-gdn-fp16")
    return subprocess.call(cmd)


def _cmd_embed(args: argparse.Namespace) -> int:
    out_dtype = getattr(args, "dtype", "float32")
    if out_dtype == "float32":
        return run_embed_command(args, _make_engine)

    texts = list(args.texts)
    if args.stdin:
        texts.extend(line.rstrip("\r\n") for line in sys.stdin if line.rstrip("\r\n"))
    if not texts:
        print("No input text provided. Pass text arguments or --stdin.", file=sys.stderr)
        return 2

    eng = _make_engine(args)
    try:
        kw: dict[str, Any] = {
            "is_query": args.query,
            "add_query_prefix": args.query_prefix,
            "add_document_prefix": bool(getattr(args, "document_prefix", False)),
            "prefix": getattr(args, "prefix", None),
            "prompt_name": getattr(args, "prompt_name", None),
            "dimensions": getattr(args, "dimensions", None),
            "output_dtype": out_dtype,
            "normalize": out_dtype != "int8",
            "mode": args.mode,
        }
        t0 = time.perf_counter()
        vecs = eng.encode_texts(texts, **kw)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if args.json:
            print(
                json.dumps(
                    {
                        "count": int(vecs.shape[0]),
                        "dim": int(vecs.shape[1]),
                        "mode": args.mode,
                        "dtype": out_dtype,
                        "latency_ms": round(dt_ms, 3),
                        "embeddings": vecs.tolist(),
                    }
                )
            )
        else:
            for i, row in enumerate(vecs):
                preview = row[:6].tolist() if out_dtype == "int8" else np.round(row[:6].astype(np.float32), 5).tolist()
                print(f"[{i}] dtype={out_dtype} vec[:6]={preview}")
            print(f"Encoded {vecs.shape[0]} text(s) in {dt_ms:.2f} ms (mode={args.mode}, dtype={out_dtype})")
        return 0
    finally:
        eng.close()


def _cos_row_min(got: np.ndarray, exp: np.ndarray) -> float:
    g = np.asarray(got, dtype=np.float64)
    e = np.asarray(exp, dtype=np.float64)
    vals: list[float] = []
    for rg, re_ in zip(g, e):
        if np.all(rg == 0.0) and np.all(re_ == 0.0):
            vals.append(1.0)
        else:
            denom = max(float(np.linalg.norm(rg) * np.linalg.norm(re_)), 1e-12)
            vals.append(float(np.dot(rg, re_) / denom))
    return min(vals) if vals else 1.0


def _ane_verify_min_cos_threshold(
    ane_manifest: dict[str, Any] | None,
    active_profile: str | None,
) -> float:
    """Return profile-aware ANE minimum cosine bar (`0.998` for parity/balanced, `0.980` for fast/custom)."""
    if active_profile in ("parity", "balanced"):
        return 0.998
    if (
        isinstance(ane_manifest, dict)
        and ane_manifest.get("rmsnorm_mode") == "max_abs"
        and ane_manifest.get("silu_mode") in ("tanh_gdn", "tanh_all")
    ):
        return 0.998
    return 0.980


def _cmd_verify(args: argparse.Namespace) -> int:
    """Verify deterministic GPU parity separately from ANE/hetero2 drift tolerance + MLComputePlan."""
    from .engine import _discover_ane_stage_specs, _load_coreml_lib, read_ane_manifest

    paths = resolve_paths(args.home)
    telemetry = get_mlx_runtime_telemetry()
    req_mode = args.mode
    req_ane_profile = getattr(args, "ane_profile", None)
    ane_manifest = read_ane_manifest(paths.ane_models_dir)

    # Check ANE MLComputePlan residency when ANE artifacts are present.
    ane_plans: dict[str, Any] = {}
    if paths.coreml_dylib.is_file() and paths.ane_models_dir.is_dir():
        stage_specs = _discover_ane_stage_specs(paths.ane_models_dir, expected_profile=req_ane_profile)
        if stage_specs:
            lib = _load_coreml_lib(paths.coreml_dylib)
            has_fn_inspect = hasattr(lib, "pplx_ane_inspect_plan_fn")
            buf = ctypes.create_string_buffer(512)
            for _cap, (stages, fn_name) in stage_specs.items():
                for st_path in stages:
                    plan_key = f"{st_path.name}:{fn_name}" if fn_name else st_path.name
                    if fn_name and has_fn_inspect:
                        rc = lib.pplx_ane_inspect_plan_fn(
                            str(st_path).encode("utf-8"), fn_name.encode("utf-8"), 0, buf, 512
                        )
                    else:
                        rc = lib.pplx_ane_inspect_plan(str(st_path).encode("utf-8"), 0, buf, 512)
                    ane_plans[plan_key] = (
                        json.loads(buf.value.decode("utf-8")) if rc > 0 else {"error": buf.value.decode("utf-8", errors="replace")}
                    )

    eng = _make_engine(args, deterministic=(req_mode == "gpu"))
    try:
        ane_init_broken = bool(
            (not eng.ane_available)
            and (req_ane_profile is not None or (ane_manifest is not None and bool(eng._ane_init_error)))
        )
        modes_to_run: list[str]
        if req_mode == "all":
            modes_to_run = ["gpu"]
            if eng.ane_available:
                modes_to_run.extend(["ane", "hetero", "hetero2"])
            elif ane_init_broken:
                modes_to_run.append("ane")
        else:
            modes_to_run = [req_mode]

        pkg_root = Path(__file__).resolve().parents[2]
        fix_candidates = [
            paths.home / "tests" / "fixtures",
            pkg_root / "tests" / "fixtures",
        ]
        fix_dir = next((d for d in fix_candidates if (d / "cases.json").is_file() and (d / "reference.npz").is_file()), None)

        ane_min_cos_bar = _ane_verify_min_cos_threshold(ane_manifest, eng.ane_profile)
        reports: list[dict[str, Any]] = []
        failed = False

        if fix_dir is not None:
            cases = json.loads((fix_dir / "cases.json").read_text(encoding="utf-8"))
            ref = np.load(fix_dir / "reference.npz")
            for m in modes_to_run:
                if m in ("ane", "hetero", "hetero2") and not eng.ane_available:
                    if ane_init_broken:
                        failed = True
                        reports.append(
                            {
                                "mode": m,
                                "parity_tier": "ane_drift_tolerant",
                                "status": "failed",
                                "reason": eng._ane_init_error or f"ANE profile {req_ane_profile!r} unavailable",
                            }
                        )
                    else:
                        reports.append({"mode": m, "status": "skipped", "reason": eng._ane_init_error or "ANE unavailable"})
                    continue
                is_det_gpu = m == "gpu"
                min_cos = 1.0
                max_i8_err = 0
                exact_fracs: list[float] = []
                evaluated = 0
                for case in cases:
                    cid = case["id"]
                    docs = case["documents"]
                    if m == "ane":
                        lens = (
                            [len(eng.tokenizer.encode_query(row[0])) for row in docs]
                            if case["task"] == "query"
                            else [len(eng.tokenizer.encode_contextual_document(d)[0]) for d in docs]
                        )
                        if any(ln > eng.ane_query_max_len for ln in lens):
                            continue
                    if case["task"] == "query":
                        flat_q = [row[0] for row in docs]
                        gp = eng.encode_queries(flat_q, output_dtype="float32", normalize=False, mode=m)
                        gi = eng.encode_queries(flat_q, output_dtype="int8", normalize=False, mode=m)
                        ep = ref[f"{cid}/projected"].reshape(-1, 1024)
                        ei = np.concatenate([ref[f"{cid}/encoded/{k}"] for k in range(len(flat_q))], axis=0)
                        min_cos = min(min_cos, _cos_row_min(gp, ep))
                        diff = np.abs(gi.astype(np.int32) - ei.astype(np.int32))
                        max_i8_err = max(max_i8_err, int(diff.max()))
                        exact_fracs.append(float(np.mean(diff == 0)))
                        evaluated += 1
                    else:
                        gp_list = eng.encode_contextual_chunks(docs, output_dtype="float32", normalize=False, mode=m)
                        gi_list = eng.encode_contextual_chunks(docs, output_dtype="int8", normalize=False, mode=m)
                        ep_all = ref[f"{cid}/projected"]
                        for di, doc in enumerate(docs):
                            non_empty = [ci for ci, c in enumerate(doc) if c]
                            if not non_empty:
                                continue
                            min_cos = min(min_cos, _cos_row_min(gp_list[di][non_empty], ep_all[di, non_empty]))
                            diff = np.abs(
                                gi_list[di][non_empty].astype(np.int32)
                                - ref[f"{cid}/encoded/{di}"][non_empty].astype(np.int32)
                            )
                            max_i8_err = max(max_i8_err, int(diff.max()))
                            exact_fracs.append(float(np.mean(diff == 0)))
                        evaluated += 1

                # Deterministic GPU uses strict parity bar; ANE/hetero/hetero2 uses profile-aware drift tolerance.
                if is_det_gpu:
                    strict_ok = min_cos >= 0.9998 and (max_i8_err <= 1 if args.model_dtype == "float16" else max_i8_err <= 3)
                    tier = "deterministic_gpu"
                    passed = strict_ok
                else:
                    strict_ok = min_cos >= 0.9998
                    tier = "ane_drift_tolerant"
                    passed = min_cos >= ane_min_cos_bar
                if not passed:
                    failed = True
                rep_entry: dict[str, Any] = {
                    "mode": m,
                    "parity_tier": tier,
                    "cases_evaluated": evaluated,
                    "proj_min_cos": round(min_cos, 6),
                    "int8_max_abs_err": max_i8_err,
                    "int8_exact_fraction": round(sum(exact_fracs) / max(len(exact_fracs), 1), 4),
                    "within_strict_parity": strict_ok,
                    "ane_drift_warning": (not is_det_gpu) and (not strict_ok),
                    "status": "ok" if passed else "failed",
                }
                if not is_det_gpu:
                    rep_entry["ane_min_cos_threshold"] = ane_min_cos_bar
                reports.append(rep_entry)
        else:
            probe_text = "Apple Silicon unified memory enables low-latency contextual embeddings."
            v_gpu = eng.encode_documents([probe_text], mode="gpu", normalize=True)[0]
            reports.append({"mode": "gpu", "parity_tier": "deterministic_gpu", "smoke_norm": round(float(np.linalg.norm(v_gpu)), 6), "status": "ok"})
            if req_mode != "gpu":
                if eng.ane_available:
                    v_ane = eng.encode_documents([probe_text], mode="ane", normalize=True)[0]
                    cos = float(np.dot(v_gpu, v_ane))
                    smoke_ok = cos >= ane_min_cos_bar
                    if not smoke_ok:
                        failed = True
                    reports.append(
                        {
                            "mode": "ane",
                            "parity_tier": "ane_drift_tolerant",
                            "smoke_cos_vs_gpu": round(cos, 6),
                            "ane_min_cos_threshold": ane_min_cos_bar,
                            "within_strict_parity": cos >= 0.9998,
                            "status": "ok" if smoke_ok else "failed",
                        }
                    )
                elif ane_init_broken:
                    failed = True
                    reports.append(
                        {
                            "mode": "ane",
                            "parity_tier": "ane_drift_tolerant",
                            "status": "failed",
                            "reason": eng._ane_init_error or f"ANE profile {req_ane_profile!r} unavailable",
                        }
                    )

        payload = {
            "home": str(paths.home),
            "ane_available": eng.ane_available,
            "ane_profile": eng.ane_profile,
            "staged_ane_profile": eng.staged_ane_profile,
            "ane_manifest": (
                {
                    "schema_version": ane_manifest.get("schema_version"),
                    "ane_profile": ane_manifest.get("ane_profile"),
                    "silu_mode": ane_manifest.get("silu_mode"),
                    "rmsnorm_mode": ane_manifest.get("rmsnorm_mode"),
                    "precision": ane_manifest.get("precision"),
                    "inv_mode": ane_manifest.get("inv_mode"),
                    "config_hash": ane_manifest.get("config_hash"),
                }
                if ane_manifest is not None
                else None
            ),
            "telemetry": telemetry,
            "ane_compute_plans": ane_plans,
            "reports": reports,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            prof_txt = f" ane_profile={eng.ane_profile}" if eng.ane_profile else ""
            print(
                f"home={paths.home} ane_available={eng.ane_available}{prof_txt} "
                f"MLX_ENABLE_TF32={telemetry['mlx_enable_tf32']} "
                f"chip={telemetry['device_name']} apple10={telemetry['is_apple10']} "
                f"sdpa_slice={telemetry['sdpa_slice_len']}"
            )
            for rep in reports:
                print(f"  [{rep['mode']}] ({rep.get('parity_tier', '-')}) {json.dumps(rep)}")
        return 1 if failed else 0
    finally:
        eng.close()


def _cmd_bench(args: argparse.Namespace) -> int:
    eng = _make_engine(args)
    telemetry = get_mlx_runtime_telemetry()
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

        t_tok0 = time.perf_counter()
        toks = [eng.tokenizer.encode_document(t, max_length=args.seq_len) for t in texts]
        tok_ms = (time.perf_counter() - t_tok0) * 1e3
        total_toks = sum(len(row) for row in toks)

        if args.mode == "all":
            modes: list[str] = ["gpu"]
            if eng.ane_available:
                if args.seq_len <= eng.ane_query_max_len:
                    modes.append("ane")
                modes.extend(["hetero", "hetero2"])
        else:
            modes = [args.mode]

        print(
            f"MLX_ENABLE_TF32={telemetry['mlx_enable_tf32']} chip={telemetry['device_name']} "
            f"apple10={telemetry['is_apple10']} sdpa_slice={telemetry['sdpa_slice_len']} "
            f"tokenized={len(texts)} ({total_toks:,} tok) in {tok_ms:.1f}ms"
        )

        runs = max(1, int(getattr(args, "runs", 3)))
        timing_scope = getattr(args, "timing_scope", "forward")
        for m in modes:
            if not args.no_warmup and texts:
                if timing_scope == "e2e":
                    _ = eng.encode(
                        texts[: min(4, len(texts))],
                        batch_size=min(args.batch_size, 4),
                        max_length=args.seq_len,
                        dimensions=args.dimensions,
                        mode=m,
                    )
                else:
                    _ = eng.encode_tokens(
                        toks[: min(4, len(toks))],
                        batch_size=min(args.batch_size, 4),
                        dimensions=args.dimensions,
                        mode=m,
                    )

            times: list[float] = []
            vecs: Any = None
            for _ in range(runs):
                t0 = time.perf_counter()
                if timing_scope == "e2e":
                    vecs = eng.encode(
                        texts,
                        batch_size=args.batch_size,
                        max_length=args.seq_len,
                        dimensions=args.dimensions,
                        mode=m,
                    )
                else:
                    vecs = eng.encode_tokens(
                        toks,
                        batch_size=args.batch_size,
                        dimensions=args.dimensions,
                        mode=m,
                    )
                times.append(time.perf_counter() - t0)
            dt = float(np.median(times))
            clear_mlx_arena(256)
            fp_gb = peak_footprint_gb()
            print(
                f"mode={m} scope={timing_scope} n={len(texts)} shape={tuple(vecs.shape)} "
                f"total_tokens={total_toks} median={dt:.3f}s "
                f"throughput={total_toks / max(dt, 1e-9):.1f} tok/s ({len(texts) / max(dt, 1e-9):.1f} texts/s) "
                f"peak_footprint_gb={fp_gb:.3f}"
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
    _add_common_engine_args(p_weights, include_ane_profile=False)
    p_weights.add_argument("--model-dir", default=None, help="Optional local HF snapshot directory")
    p_weights.add_argument("--local-files-only", action="store_true", help="Do not hit network")

    p_ane = sub.add_parser("setup-ane", help="Build CoreML ANE bundles via scripts/build_ane_models.py")
    _add_common_engine_args(p_ane, include_ane_profile=False)
    p_ane.add_argument("--model-dir", default=None)
    p_ane.add_argument("--variant", default=None)
    p_ane.add_argument("--backend", choices=("coreml", "coreai"), default="coreml")
    p_ane.add_argument("--seq-lens", default=None)
    p_ane.add_argument("--precision", choices=("fp16", "w8a16", "w8a8"), default=None)
    p_ane.add_argument("--layers-per-stage", type=int, default=None)
    p_ane.add_argument("--inv-mode", choices=("block", "fwd_subst"), default=None)
    p_ane.add_argument(
        "--ane-profile",
        choices=("fast", "balanced", "parity"),
        default=None,
        help="Preset ANE speed/parity profile: 'balanced' (tanh_gdn+max_abs, default), 'fast' (silu+prescale), "
        "or 'parity' (tanh_all+max_abs). 'parity'/'balanced' is recommended for --dtype int8 or --dimensions 128 "
        "(prescale 1/64 inflates 2^-14 to 0.25 unscaled eps). Explicit --silu-mode/--rmsnorm-mode override the preset.",
    )
    p_ane.add_argument("--silu-mode", choices=("silu", "tanh_gdn", "tanh_all"), default=None)
    p_ane.add_argument("--rmsnorm-mode", choices=("prescale", "max_abs"), default=None)
    p_ane.add_argument("--calib-stats", default=None)
    p_ane.add_argument("--ftz-alpha", type=float, default=None)
    p_ane.add_argument("--preserve-gdn-fp16", action=argparse.BooleanOptionalAction, default=True)

    p_embed = sub.add_parser("embed", help="Embed input texts from arguments or stdin")
    _add_common_engine_args(p_embed)
    add_embed_arguments(
        p_embed,
        mode_choices=_MODES,
        default_mode="auto",
        allow_doc_prefix=True,
        allow_dimensions=True,
    )
    p_embed.add_argument(
        "--dtype",
        choices=("float32", "float16", "int8"),
        default="float32",
        help="Output embedding dtype (default: float32)",
    )

    p_verify = sub.add_parser("verify", help="Verify deterministic GPU parity, ANE drift tolerance, and MLComputePlan")
    _add_common_engine_args(p_verify)
    p_verify.add_argument("--mode", choices=_VERIFY_MODES, default="all")
    p_verify.add_argument("--json", action="store_true")

    p_bench = sub.add_parser("bench", help="Benchmark throughput on synthetic or Parquet inputs")
    _add_common_engine_args(p_bench)
    p_bench.add_argument("--mode", choices=_BENCH_MODES, default="auto")
    p_bench.add_argument("--n", type=int, default=64)
    p_bench.add_argument("--batch-size", type=int, default=16)
    p_bench.add_argument("--seq-len", type=int, default=512)
    p_bench.add_argument("--runs", type=int, default=3)
    p_bench.add_argument("--timing-scope", choices=("forward", "e2e"), default="forward")
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
    # Extend shared --dtype choices locally to support signed int8 embeddings.
    for act in p_parquet._actions:
        if "--dtype" in act.option_strings:
            act.choices = ("float32", "float16", "int8")
            break
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
    p_parquet.add_argument(
        "--header-text",
        default=None,
        help="Optional document header/title anchored across contextual windows (--context-tokens)",
    )
    p_parquet.add_argument(
        "--header-col",
        default=None,
        help="Optional input Parquet column containing per-document header/title text",
    )
    p_parquet.add_argument(
        "--repeat-header-tokens",
        type=int,
        default=None,
        help="Lead tokens from chunk 0 anchored on continuation windows (default: auto 128 when --context-tokens >= 512 and no header is supplied; 0 disables)",
    )
    p_parquet.add_argument(
        "--window-overlap-chunks",
        type=int,
        default=0,
        help="Opt-in leading chunk overlap for cross-window context without duplicating emitted rows (default: 0)",
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
        return _cmd_embed(args)
    if args.command == "verify":
        return _cmd_verify(args)
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
                    header_text=getattr(args, "header_text", None),
                    header_col=getattr(args, "header_col", None),
                    repeat_header_tokens=getattr(args, "repeat_header_tokens", None),
                    window_overlap_chunks=int(getattr(args, "window_overlap_chunks", 0)),
                    mode=args.mode,
                    dtype=args.dtype,
                    batch_rows=args.batch_rows,
                    encode_batch_rows=args.encode_batch_rows,
                    shard_rows=args.shard_rows,
                    filter_ocr=args.filter_ocr,
                    ocr_pages_only=args.ocr_pages_only,
                    skip_empty=args.skip_empty,
                    include_chunk_text=True if not getattr(args, "drop_text_col", False) or getattr(args, "include_chunk_text", False) else True,
                    prefix=getattr(args, "prefix", None),
                    prompt_name=getattr(args, "prompt_name", None),
                    add_query_prefix=bool(getattr(args, "query_prefix", False)),
                    add_document_prefix=bool(getattr(args, "document_prefix", False)),
                    dimensions=getattr(args, "dimensions", None),
                    compression=args.compression,
                    resume=bool(getattr(args, "resume", False)),
                    limit=args.limit,
                    progress_every=int(getattr(args, "progress_every", 0)),
                )
                print(
                    f"DONE (contextual): files={summary.input_files} rows_read={summary.rows_read:,} "
                    f"chunks_emitted={summary.rows_emitted:,} tokens={summary.total_tokens:,} "
                    f"elapsed={summary.elapsed_seconds:.2f}s -> {summary.output_path}"
                )
                return 0
            finally:
                eng.close()
        if args.dtype == "int8":
            eng = _make_engine(args)
            try:
                summary = embed_parquet(
                    eng,
                    args.input,
                    args.output,
                    text_col=args.text_col,
                    keep_cols=args.keep_cols,
                    drop_text_col=args.drop_text_col,
                    mode=args.mode,
                    dtype=args.dtype,
                    batch_rows=args.batch_rows,
                    encode_batch_rows=args.encode_batch_rows,
                    shard_rows=args.shard_rows,
                    filter_ocr=args.filter_ocr,
                    ocr_pages_only=args.ocr_pages_only,
                    per_ocr_page=args.per_ocr_page,
                    chunk=args.chunk,
                    max_body_tokens=args.max_body_tokens,
                    overlap=args.overlap,
                    max_length=args.max_length,
                    max_chunk_tokens=args.max_chunk_tokens,
                    skip_empty=args.skip_empty,
                    include_chunk_text=args.include_chunk_text,
                    prefix=getattr(args, "prefix", None),
                    prompt_name=getattr(args, "prompt_name", None),
                    add_query_prefix=args.query_prefix,
                    add_document_prefix=bool(getattr(args, "document_prefix", False)),
                    dimensions=getattr(args, "dimensions", None),
                    compression=args.compression,
                    resume=args.resume,
                    limit=args.limit,
                    progress_every=args.progress_every,
                )
                print(
                    f"DONE: files={summary.input_files} rows_read={summary.rows_read:,} "
                    f"rows_emitted={summary.rows_emitted:,} tokens={summary.total_tokens:,} "
                    f"elapsed={summary.elapsed_seconds:.2f}s throughput={summary.tokens_per_sec:,.0f} tok/s "
                    f"({summary.rows_per_sec:,.0f} rows/s) -> {summary.output_path}"
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
