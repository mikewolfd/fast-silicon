#!/usr/bin/env python3
"""Reproducible Apple Silicon benchmark & evaluation harness for `fast-pplx`.

Captures speed (symmetric E2E + forward-only breakdown, interleaved ABBA thermal-drift
cancellation, `clear_mlx_arena(256)`, and `wired_limit_scope`), Metal active peak vs macOS
OS unified-memory footprint (`RUSAGE_INFO_V4`), 16-case PyTorch CPU reference parity across
`gpu`, `gpu-bf16`, `ane`, and `hetero2` plus `S=4096` streamed parity, streamed 32K context
scaling, contextual window retrieval sensitivity, and exact bit-identity regression checks —
all serialized cleanly under `/tmp/m4_gpu_validation.lock` with system quiescence checks and
explicit `MLX_ENABLE_TF32` telemetry.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parents[1] if (_PACKAGE_ROOT.parents[1] / "packages").is_dir() else _PACKAGE_ROOT
_FIXTURES_DIR = _PACKAGE_ROOT / "tests" / "fixtures"
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"

# Contextual disambiguation corpus used for offline retrieval-window probing (synthetic, zero downloads).
_SYNTHETIC_CONTEXT_DOCS: tuple[dict[str, Any], ...] = (
    {
        "doc_id": "marine-heat",
        "title": "North Atlantic Thermohaline Buoy Observations",
        "preamble": (
            "Station K4-North operates a deep moored thermistor array at 54N 36W measuring salinity "
            "anomalies and downward heat transport beneath the subpolar gyre across winter storms. "
            "Lead oceanographer Dr. Linnea Lindholm calibrated the optical backscatter sensors before deployment. "
        ),
        "filler": (
            "Routine satellite altimetry passes recorded moderate swell heights and steady zonal winds "
            "across neighboring weather buoys throughout the autumn cruise window without instrument drift. "
        ),
        "target": (
            "The primary thermistor mooring maintained uninterrupted telemetry at 1,400 meters depth "
            "for fourteen months while recording a 0.18 degree Celsius subsurface warming signal."
        ),
        "query": "Which mooring recorded a 0.18 C warming signal at 1,400 meters depth after Linnea Lindholm calibrated sensors?",
    },
    {
        "doc_id": "battery-cathode",
        "title": "Solid-State Electrolyte Sintering Trial",
        "preamble": (
            "Cell batch LFP-774 uses a tantalum-doped garnet ceramic separator co-sintered at 980 C "
            "in an argon-purged tube furnace overseen by materials engineer Mateo Alvarez. "
        ),
        "filler": (
            "Baseline pouch assemblies from standard slurry lines underwent routine visual inspection "
            "and tab-weld pull testing on adjacent benches during the same facility shift. "
        ),
        "target": (
            "The co-sintered ceramic batch cut interfacial charge-transfer resistance by 41 percent "
            "and completed 650 fast-charge cycles without measurable lithium dendrite penetration."
        ),
        "query": "Which co-sintered ceramic separator batch cut interfacial resistance by 41 percent over 650 cycles?",
    },
    {
        "doc_id": "alpine-glacier",
        "title": "Engadin Firn Core Isotope Record",
        "preamble": (
            "Core borehole EFC-19 on the Upper Engadin saddle extracted stratified firn lenses down "
            "to bedrock to reconstruct spring meltwater percolation since 1965 under glaciologist Hannah Berg. "
        ),
        "filler": (
            "Automatic weather masts along lower valley pastures logged daily barometric pressure, "
            "diffuse solar irradiance, and light nocturnal frost events across the field season. "
        ),
        "target": (
            "The upper firn core revealed three anomalous refrozen ice lenses matching Mediterranean "
            "dust-deposition horizons that accelerated summer albedo decay by 9 percent."
        ),
        "query": "Which upper firn core linked Mediterranean dust horizons to a 9 percent summer albedo decay?",
    },
    {
        "doc_id": "compiler-register",
        "title": "Static Single Assignment Coalescing Pass",
        "preamble": (
            "Backend pass RegCoalesce-V3 replaces redundant phi-web copies on tight SIMD reduction loops "
            "within the vector code generator authored by compiler engineer Kenji Takahashi. "
        ),
        "filler": (
            "Front-end lexical scanning, diagnostics formatting, and link-time symbol deduplication "
            "remained unchanged across nightly host test builds during this optimization cycle. "
        ),
        "target": (
            "Applying the aggressive copy-propagation pass eliminated 28 percent of register spills "
            "in 256-wide stencil kernels and improved pipeline issue width by 14 percent."
        ),
        "query": "Which pass authored by Kenji Takahashi eliminated 28 percent of spills in 256-wide stencil kernels?",
    },
    {
        "doc_id": "optical-interconnect",
        "title": "Silicon Nitride Ring Modulator Characterization",
        "preamble": (
            "Wafer reticle SNRM-12 integrates sixteen micro-ring resonators coupled to low-loss bus "
            "waveguides measured on a temperature-stabilized chuck by photonics researcher Priya Nair. "
        ),
        "filler": (
            "Fiber-array epoxy curing profiles and rack cooling airflow checks followed standard "
            "cleanroom operating checklists prior to optical wafer probing. "
        ),
        "target": (
            "The micro-ring modulator bank sustained 112 gigabits per second per lane at 1.4 picojoules "
            "per bit thermal tuning efficiency across the full C-band comb."
        ),
        "query": "Which silicon nitride micro-ring wafer sustained 112 Gbps per lane at 1.4 pJ/bit?",
    },
    {
        "doc_id": "seismic-array",
        "title": "Borehole Distributed Acoustic Sensing Deployment",
        "preamble": (
            "Optical interrogation unit DAS-408 sampled 10-meter gauge lengths along an abandoned "
            "geothermal well casing operated by geophysicist Nora Lindqvist. "
        ),
        "filler": (
            "Surface microbarograph loggers and solar charge controllers operated nominally along "
            "the perimeter fence line with weekly battery voltage telemetry checks. "
        ),
        "target": (
            "The downhole fiber transect resolved sub-millimeter micro-seismic rupture events at "
            "2.8 kilometers depth ten times more clearly than surface geophones."
        ),
        "query": "Which downhole fiber unit operated by Nora Lindqvist resolved 2.8 km deep micro-seismic events?",
    },
)


def _sys_quiescence() -> list[str]:
    from fast_shared.bench import check_system_quiescence

    return check_system_quiescence()


def _footprint_snapshot() -> dict[str, float | int]:
    from fast_shared.bench import get_unified_memory_footprint

    return get_unified_memory_footprint()


def _clear_arena(threshold_mib: int = 256) -> None:
    try:
        from fast_shared.bench import clear_mlx_arena

        clear_mlx_arena(threshold_mib)
    except Exception:
        pass


def _collect_telemetry() -> dict[str, Any]:
    try:
        from fast_pplx.gpu import get_mlx_runtime_telemetry

        return get_mlx_runtime_telemetry()
    except Exception:
        return {"mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "0")}


def _mlx_peak_gb() -> float:
    try:
        import mlx.core as mx
    except ImportError:
        return 0.0
    try:
        return round(float(mx.get_peak_memory()) / 1e9, 4)
    except Exception:
        return 0.0


def _reset_mlx_peak() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass


def _extract_last_json_dict(raw_stdout: str) -> dict[str, Any]:
    """Extract the final top-level JSON dict from stdout, ignoring C-runtime log noise."""
    stripped = raw_stdout.strip()
    if not stripped:
        raise ValueError("empty subprocess stdout")
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    last_dict: dict[str, Any] | None = None
    idx = 0
    n = len(stripped)
    while idx < n:
        brace = stripped.find("{", idx)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(stripped, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            last_dict = obj
        idx = end
    if last_dict is None:
        raise ValueError(f"no JSON object found in stdout ({stripped[:120]!r})")
    return last_dict


@contextmanager
def _wired_limit_ctx() -> Iterator[int | None]:
    try:
        from fast_pplx.gpu import wired_limit_scope

        with wired_limit_scope() as prev:
            yield prev
    except Exception:
        yield None


@contextmanager
def _hw_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    import fcntl

    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev = os.environ.get("M4_GPU_LOCK_HELD")
        os.environ["M4_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("M4_GPU_LOCK_HELD", None)
            else:
                os.environ["M4_GPU_LOCK_HELD"] = prev
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _preflight_quiescence(*, allow_contended: bool) -> list[str]:
    warnings = _sys_quiescence()
    if warnings:
        print(f"QUIESCENCE WARNING ({len(warnings)} contender(s) active):", file=sys.stderr)
        for line in warnings:
            print(f"  {line}", file=sys.stderr)
        if not allow_contended:
            raise RuntimeError(
                "System not quiescent for reliable Apple Silicon benchmarking; re-run when clear or pass --allow-contended."
            )
    return warnings


def _parse_int_csv(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _parse_str_csv(raw: str) -> list[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _resolve_model_dir(explicit: str | None, home: str | None) -> Path | None:
    from fast_pplx.config import DEFAULT_MODEL_NAME
    from fast_pplx.gpu import _find_local_hf_snapshot
    from fast_pplx.paths import resolve_paths

    if explicit:
        return Path(explicit).expanduser().resolve()
    paths = resolve_paths(home)
    if (paths.model_dir / "config.json").is_file():
        return paths.model_dir
    return _find_local_hf_snapshot(DEFAULT_MODEL_NAME)


def _resolve_engine_model_dir(explicit: str | None, home: str | None) -> tuple[bool, Path | None]:
    """Return `(ready, model_dir_arg)` for `FastPplx`, preferring staged `.npz` + mmap when present."""
    from fast_pplx.paths import resolve_paths

    if explicit:
        return True, Path(explicit).expanduser().resolve()
    paths = resolve_paths(home)
    if paths.weights_npz.is_file() and paths.embed_tokens_safetensors.is_file():
        return True, None
    hf_dir = _resolve_model_dir(None, home)
    return (hf_dir is not None), hf_dir


def _check_ane_ready(home: str | None) -> tuple[bool, str]:
    from fast_pplx.paths import resolve_paths

    paths = resolve_paths(home)
    if not (paths.coreml_dylib.is_file() and paths.io_dylib.is_file()):
        return False, f"native dylibs missing under {paths.build_dir} (run `make native`)"
    if not (paths.ane_models_dir.is_dir() and any(paths.ane_models_dir.glob("*.mlmodelc"))):
        return False, f"no compiled .mlmodelc bundles under {paths.ane_models_dir} (run `fast-pplx setup-ane`)"
    if not (paths.weights_npz.is_file() and paths.embed_tokens_safetensors.is_file()):
        return False, f"exported F16 weights missing under {paths.weights_dir} (run `fast-pplx setup-weights`)"
    return True, f"staged at {paths.ane_models_dir}"


def _build_synthetic_text(seq_tokens: int, seed_idx: int = 0) -> str:
    phrase = (
        "Ocean currents transport thermal energy across subtropical gyres, modulating coastal "
        "pressure gradients and seasonal precipitation patterns over decadal timescales. "
    )
    repeats = max(1, (int(seq_tokens) + 15) // 22)
    return f"Sample record {seed_idx}. " + (phrase * repeats)


def _cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    denom = max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-12)
    return float(np.dot(af, bf) / denom)


def compute_vector_parity(got: np.ndarray, exp: np.ndarray) -> dict[str, float]:
    """Compare predicted matrix `[N, D]` vs reference matrix `[N, D]`."""
    g = np.asarray(got, dtype=np.float64)
    e = np.asarray(exp, dtype=np.float64)
    if g.shape != e.shape:
        raise ValueError(f"Shape mismatch: got {g.shape} vs expected {e.shape}")
    if g.size == 0:
        return {"max_abs_err": 0.0, "min_cos": 1.0, "mean_cos": 1.0}
    max_abs = float(np.max(np.abs(g - e)))
    cos_vals: list[float] = []
    for row_g, row_e in zip(g, e):
        if np.all(row_e == 0.0) and np.all(row_g == 0.0):
            cos_vals.append(1.0)
        else:
            cos_vals.append(_cos_similarity(row_g, row_e))
    return {
        "max_abs_err": max_abs,
        "min_cos": float(min(cos_vals)),
        "mean_cos": float(sum(cos_vals) / len(cos_vals)),
    }


def compute_int8_parity(got: np.ndarray, exp: np.ndarray) -> dict[str, float | int]:
    """Compare quantized int8 matrices `[N, D]` on max abs diff, exact fraction, and cosine."""
    gi = np.asarray(got, dtype=np.int32)
    ei = np.asarray(exp, dtype=np.int32)
    if gi.shape != ei.shape:
        raise ValueError(f"Int8 shape mismatch: got {gi.shape} vs expected {ei.shape}")
    if gi.size == 0:
        return {"max_abs_err": 0, "exact_fraction": 1.0, "min_cos": 1.0}
    diff = np.abs(gi - ei)
    exact = float(np.mean(diff == 0))
    cos_vals: list[float] = []
    for row_g, row_e in zip(gi, ei):
        if np.all(row_e == 0) and np.all(row_g == 0):
            cos_vals.append(1.0)
        else:
            cos_vals.append(_cos_similarity(row_g, row_e))
    return {
        "max_abs_err": int(np.max(diff)),
        "exact_fraction": exact,
        "min_cos": float(min(cos_vals)),
    }


def score_retrieval_metrics(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_doc_ids: Sequence[str],
    query_doc_ids: Sequence[str],
) -> dict[str, float]:
    """Compute MRR, Recall@1, and nDCG@5 over normalized embedding matrices."""
    q = np.asarray(query_vecs, dtype=np.float64)
    c = np.asarray(chunk_vecs, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    c = c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)
    sims = q @ c.T

    rr_sum = 0.0
    hits_at_1 = 0
    ndcg5_sum = 0.0
    n_q = len(query_doc_ids)
    for qi, expected_doc in enumerate(query_doc_ids):
        ranked = np.argsort(-sims[qi], kind="stable")
        first_rank = None
        rels: list[int] = []
        for pos, ci in enumerate(ranked[:5]):
            rel = 1 if chunk_doc_ids[int(ci)] == expected_doc else 0
            rels.append(rel)
        for pos, ci in enumerate(ranked):
            if chunk_doc_ids[int(ci)] == expected_doc:
                first_rank = pos + 1
                break
        if first_rank is not None:
            rr_sum += 1.0 / first_rank
            if first_rank == 1:
                hits_at_1 += 1
        dcg = sum(r / math.log2(idx + 2) for idx, r in enumerate(rels))
        n_pos = sum(1 for did in chunk_doc_ids if did == expected_doc)
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(5, n_pos)))
        ndcg5_sum += (dcg / idcg) if idcg > 0.0 else 0.0

    denom = max(n_q, 1)
    return {
        "queries": n_q,
        "chunks": len(chunk_doc_ids),
        "mrr": round(rr_sum / denom, 6),
        "recall_at_1": round(hits_at_1 / denom, 6),
        "ndcg_at_5": round(ndcg5_sum / denom, 6),
    }


def annotate_speed_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate `speed` rows with speedups vs baselines (`gpu-bf16`, `gpu-no-mmap`, `torch-cpu`) & mem delta."""
    out = [dict(r) for r in rows]
    by_wl_mode: dict[tuple[str, str], dict[str, Any]] = {}
    for r in out:
        if r.get("status") == "ok" and r.get("workload"):
            by_wl_mode[(str(r["workload"]), str(r["mode"]))] = r

    for r in out:
        if r.get("status") != "ok":
            continue
        wl = str(r.get("workload", ""))
        mode = str(r.get("mode", ""))
        notes_parts = [str(r.get("notes", ""))] if r.get("notes") else []
        tok_s = float(r.get("tok_s", 0.0))

        base_bf16 = by_wl_mode.get((wl, "gpu-bf16"))
        if base_bf16 and mode not in ("gpu-bf16", "torch-cpu"):
            ratio = tok_s / max(float(base_bf16["tok_s"]), 1e-9)
            r["vs_gpu_bf16"] = round(ratio, 3)
            notes_parts.append(f"{ratio:.2f}x vs bf16")

        base_f16 = by_wl_mode.get((wl, "gpu"))
        if base_f16 and mode == "hetero2":
            ratio = tok_s / max(float(base_f16["tok_s"]), 1e-9)
            r["vs_gpu_f16"] = round(ratio, 3)
            notes_parts.append(f"{ratio:.2f}x vs gpu")

        base_torch = by_wl_mode.get((wl, "torch-cpu"))
        if base_torch and mode != "torch-cpu":
            ratio = tok_s / max(float(base_torch["tok_s"]), 1e-9)
            r["vs_torch_cpu"] = round(ratio, 3)
            notes_parts.append(f"{ratio:.2f}x vs torch-cpu")

        base_mps = by_wl_mode.get((wl, "torch-mps"))
        if base_mps and mode not in ("torch-cpu", "torch-mps"):
            ratio = tok_s / max(float(base_mps["tok_s"]), 1e-9)
            r["vs_torch_mps"] = round(ratio, 3)
            notes_parts.append(f"{ratio:.2f}x vs torch-mps")

        base_pub = by_wl_mode.get((wl, "pplx-context-mlx-f16"))
        if base_pub and mode in ("gpu", "ane", "hetero2"):
            ratio = tok_s / max(float(base_pub["tok_s"]), 1e-9)
            r["vs_pplx_context_mlx_f16"] = round(ratio, 3)
            notes_parts.append(f"{ratio:.2f}x vs pplx-mlx-f16")

        base_nommap = by_wl_mode.get((wl, "gpu-no-mmap"))
        if base_nommap and mode in ("gpu", "gpu-bf16", "hetero2"):
            delta_gb = float(r.get("peak_footprint_gb", 0.0)) - float(base_nommap.get("peak_footprint_gb", 0.0))
            r["footprint_delta_vs_no_mmap_gb"] = round(delta_gb, 4)
            notes_parts.append(f"dFootprint={delta_gb:+.3f}GB")

        r["baseline_summary"] = ", ".join(p for p in notes_parts if p)
    return out


def format_speed_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Format speed suite rows as a compact Markdown table with baseline comparisons."""
    lines = [
        "| Mode / Workload | Tok/s | Latency (ms) | Metal Peak (GB) | OS Footprint (GB) | RSS (GB) | Baseline Comparison / Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                f"| {r.get('label', r.get('mode', '?'))} | - | - | - | - | - | {r.get('reason', r.get('status'))} |"
            )
            continue
        comp = r.get("baseline_summary") or r.get("notes", "")
        lines.append(
            f"| {r['label']} | {r['tok_s']:.1f} | {r['latency_ms']:.2f} | "
            f"{r.get('peak_active_gb', 0.0):.3f} | {r.get('peak_footprint_gb', 0.0):.3f} | "
            f"{r.get('rss_gb', 0.0):.3f} | {comp} |"
        )
    return "\n".join(lines)


def annotate_context_limit_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each longer sequence length against the shortest baseline run (`min tokens`)."""
    out = [dict(r) for r in rows]
    ok_rows = [r for r in out if r.get("status") == "ok"]
    if not ok_rows:
        return out
    base = min(ok_rows, key=lambda r: int(r.get("actual_tokens", r.get("target_tokens", 0))))
    base_fp = float(base.get("peak_footprint_gb", 0.0))
    base_ts = max(float(base.get("tok_s", 0.0)), 1e-9)
    for r in out:
        if r.get("status") != "ok":
            continue
        r["footprint_delta_vs_base_gb"] = round(float(r.get("peak_footprint_gb", 0.0)) - base_fp, 4)
        r["tok_s_ratio_vs_base"] = round(float(r.get("tok_s", 0.0)) / base_ts, 3)
    return out


def format_context_limit_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Format context-limit scaling rows as a compact Markdown table."""
    lines = [
        "| Target Tokens | Actual Tokens | Elapsed (s) | Tok/s | vs Base Tok/s | Metal Peak (GB) | OS Footprint (GB) | Delta Footprint | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        ratio = r.get("tok_s_ratio_vs_base")
        delta = r.get("footprint_delta_vs_base_gb")
        ratio_s = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "-"
        delta_s = f"{delta:+.3f} GB" if isinstance(delta, (int, float)) else "-"
        lines.append(
            f"| {r.get('target_tokens', 0):,} | {r.get('actual_tokens', 0):,} | "
            f"{r.get('elapsed_s', 0.0):.3f} | {r.get('tok_s', 0.0):.1f} | {ratio_s} | "
            f"{r.get('peak_active_gb', 0.0):.3f} | {r.get('peak_footprint_gb', 0.0):.3f} | "
            f"{delta_s} | {r.get('status', 'unknown')} |"
        )
    return "\n".join(lines)


def annotate_context_window_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute retrieval deltas vs the non-contextual `isolated` baseline row."""
    out = [dict(r) for r in rows]
    base = next((r for r in out if str(r.get("window_budget")) == "isolated"), None)
    if base is None:
        return out
    base_mrr = float(base.get("mrr", 0.0))
    base_ndcg = float(base.get("ndcg_at_5", 0.0))
    for r in out:
        if str(r.get("window_budget")) == "isolated":
            r["delta_mrr_vs_isolated"] = 0.0
            r["delta_ndcg5_vs_isolated"] = 0.0
        else:
            r["delta_mrr_vs_isolated"] = round(float(r.get("mrr", 0.0)) - base_mrr, 6)
            r["delta_ndcg5_vs_isolated"] = round(float(r.get("ndcg_at_5", 0.0)) - base_ndcg, 6)
    return out


def format_context_windows_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Format contextual window vs isolated baseline rows cleanly."""
    lines = [
        "| Context Budget | Windows | Tok/s | MRR | Delta MRR | Recall@1 | nDCG@5 | Delta nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        dm = float(r.get("delta_mrr_vs_isolated", 0.0))
        dn = float(r.get("delta_ndcg5_vs_isolated", 0.0))
        lines.append(
            f"| {r['window_budget']} | {r['windows']} | {r['tok_s']:.1f} | "
            f"{r['mrr']:.4f} | {dm:+.4f} | {r['recall_at_1']:.4f} | {r['ndcg_at_5']:.4f} | {dn:+.4f} |"
        )
    return "\n".join(lines)


def dump_bit_identity(vectors_by_case: dict[str, np.ndarray], metadata: dict[str, Any], out_dir: Path) -> Path:
    """Persist unrounded float32 reference arrays + SHA-256 digests to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "vectors.npz"
    manifest_path = out_dir / "manifest.json"
    np.savez(npz_path, **{k: np.asarray(v, dtype=np.float32) for k, v in sorted(vectors_by_case.items())})
    digests = {}
    for k, arr in sorted(vectors_by_case.items()):
        c_arr = np.ascontiguousarray(arr, dtype=np.float32)
        digests[k] = {
            "shape": list(c_arr.shape),
            "dtype": str(c_arr.dtype),
            "sha256": hashlib.sha256(c_arr.tobytes()).hexdigest(),
        }
    manifest = {"metadata": metadata, "cases": digests}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def compare_bit_identity(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Compare two bit-identity dump directories (`vectors.npz` + `manifest.json`)."""
    base_npz = np.load(baseline_dir / "vectors.npz")
    cand_npz = np.load(candidate_dir / "vectors.npz")
    base_keys = set(base_npz.files)
    cand_keys = set(cand_npz.files)
    missing = sorted(base_keys - cand_keys)
    extra = sorted(cand_keys - base_keys)
    common = sorted(base_keys & cand_keys)

    per_case: list[dict[str, Any]] = []
    exact_matches = 0
    for key in common:
        a = np.asarray(base_npz[key])
        b = np.asarray(cand_npz[key])
        same_shape = a.shape == b.shape
        bit_eq = bool(same_shape and np.array_equal(a, b))
        if bit_eq:
            exact_matches += 1
        stats = (
            compute_vector_parity(b.reshape(-1, b.shape[-1]), a.reshape(-1, a.shape[-1]))
            if same_shape and a.size > 0
            else {"max_abs_err": float("inf"), "min_cos": 0.0, "mean_cos": 0.0}
        )
        per_case.append({"case": key, "same_shape": same_shape, "bit_identical": bit_eq, **stats})

    all_equal = not missing and not extra and exact_matches == len(common)
    return {
        "all_bit_identical": all_equal,
        "matched": exact_matches,
        "total_common": len(common),
        "missing": missing,
        "extra": extra,
        "cases": per_case,
    }


def _median_timed(fn: Any, *, warmup: int, repeats: int) -> tuple[float, Any]:
    for _ in range(max(0, warmup)):
        _ = fn()
    times: list[float] = []
    last_out: Any = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        last_out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), last_out


def _resolve_speed_mode_spec(mode: str, args: argparse.Namespace) -> tuple[str, str, bool]:
    """Map benchmark variant (`gpu`, `gpu-f16`, `gpu-bf16`, `gpu-no-mmap`, `ane`, `hetero`, `hetero2`) -> `(engine_mode, dtype, mapped)`."""
    if mode == "gpu-no-mmap":
        return "gpu", args.dtype, False
    if mode == "gpu-bf16":
        return "gpu", "bfloat16", not args.no_mapped_embedding
    if mode == "gpu-f16":
        return "gpu", "float16", not args.no_mapped_embedding
    return mode, args.dtype, not args.no_mapped_embedding


def run_abba_speed_pair(
    args: argparse.Namespace,
    mode_a: str,
    mode_b: str,
    *,
    rounds: int = 2,
) -> dict[str, Any]:
    """Compare two internal `fast-pplx` modes using `fast_shared.bench.interleaved_abba`."""
    from fast_pplx.engine import FastPplx
    from fast_shared.bench import interleaved_abba

    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"status": "skipped", "reason": "no local weights found"}

    spec_a = _resolve_speed_mode_spec(mode_a, args)
    spec_b = _resolve_speed_mode_spec(mode_b, args)
    doc_lengths = _parse_int_csv(args.doc_lengths)
    probe_len = 512 if 512 in doc_lengths else doc_lengths[0]
    b_doc = args.doc_batch if probe_len <= 1024 else max(1, min(args.doc_batch, 4))
    docs = [_build_synthetic_text(probe_len, i) for i in range(b_doc)]
    timing_scope = getattr(args, "timing_scope", "e2e")

    with _wired_limit_ctx():
        with FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=spec_a[0],
            dtype=spec_a[1],
            use_mapped_embedding=spec_a[2],
            local_files_only=True,
        ) as eng_a, FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=spec_b[0],
            dtype=spec_b[1],
            use_mapped_embedding=spec_b[2],
            local_files_only=True,
        ) as eng_b:
            toks_a = [eng_a.tokenizer.encode_document(d, max_length=probe_len) for d in docs]
            toks_b = [eng_b.tokenizer.encode_document(d, max_length=probe_len) for d in docs]
            total_toks = sum(len(t) for t in toks_a)

            def _run_a() -> dict[str, Any]:
                t0 = time.perf_counter()
                if timing_scope == "e2e":
                    _ = eng_a.encode_documents(docs, batch_size=b_doc, max_length=probe_len, dimensions=args.dimensions, mode=spec_a[0])
                else:
                    _ = eng_a.encode_tokens(toks_a, batch_size=b_doc, dimensions=args.dimensions, mode=spec_a[0])
                dt = time.perf_counter() - t0
                return {"wall_sec": round(dt, 4), "tok_s": round(total_toks / max(dt, 1e-9), 1)}

            def _run_b() -> dict[str, Any]:
                t0 = time.perf_counter()
                if timing_scope == "e2e":
                    _ = eng_b.encode_documents(docs, batch_size=b_doc, max_length=probe_len, dimensions=args.dimensions, mode=spec_b[0])
                else:
                    _ = eng_b.encode_tokens(toks_b, batch_size=b_doc, dimensions=args.dimensions, mode=spec_b[0])
                dt = time.perf_counter() - t0
                return {"wall_sec": round(dt, 4), "tok_s": round(total_toks / max(dt, 1e-9), 1)}

            abba = interleaved_abba(_run_a, _run_b, warmup=(args.warmup > 0), rounds=rounds)
            return {
                "status": "ok",
                "mode_a": mode_a,
                "mode_b": mode_b,
                "workload": f"doc_s{probe_len}",
                "timing_scope": timing_scope,
                **abba,
            }


_TORCH_CPU_SNIPPET = r"""
import ctypes, json, os, platform, resource, sys, time
import statistics
import numpy as np

try:
    import torch
    from transformers import AutoModel
except Exception as exc:
    print(json.dumps({"mode": "torch-cpu", "status": "skipped", "reason": f"torch/transformers unavailable ({exc})", "rows": []}))
    raise SystemExit(0)

cfg = json.loads(sys.argv[1])
doc_lengths = cfg["doc_lengths"]
threads_list = cfg["torch_threads"]
model_ref = cfg.get("model_dir") or "perplexity-ai/pplx-embed-v2-context-0.6b"
phrase = (
    "Ocean currents transport thermal energy across subtropical gyres, modulating coastal "
    "pressure gradients and seasonal precipitation patterns over decadal timescales. "
)

def build_text(seq_tokens, seed_idx=0):
    repeats = max(1, (int(seq_tokens) + 15) // 22)
    return f"Sample record {seed_idx}. " + (phrase * repeats)

def footprint_gb():
    if platform.system() == "Darwin":
        buf = ctypes.create_string_buffer(512)
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0:
            phys = int.from_bytes(buf.raw[240:248], "little") / 1e9
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
            return round(phys, 4), round(rss, 4)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    return round(rss, 4), round(rss, 4)

def median_t(fn, warmup, repeats):
    for _ in range(max(0, warmup)):
        _ = fn()
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = fn()
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))

t_load0 = time.perf_counter()
try:
    model = AutoModel.from_pretrained(model_ref, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True).eval()
except Exception as exc:
    print(json.dumps({"mode": "torch-cpu", "status": "skipped", "reason": f"torch model load skipped ({exc})", "rows": []}))
    raise SystemExit(0)
load_s = time.perf_counter() - t_load0

workloads = [
    ("query_b1", "torch-cpu query B=1", lambda: model.encode_queries([["What drives oceanic thermohaline heat transport across hemispheres?"]], batch_size=1), None, 1),
    ("query_b32", "torch-cpu query B=32", lambda: model.encode_queries([[f"Query {i}: summarize coastal pressure anomaly trends."] for i in range(32)], batch_size=32), None, 32),
]
for s_len in doc_lengths:
    b_doc = cfg["doc_batch"] if s_len <= 1024 else max(1, min(cfg["doc_batch"], 4))
    docs = [[build_text(s_len, i)] for i in range(b_doc)]
    workloads.append((f"doc_s{s_len}", f"torch-cpu doc B={b_doc} S={s_len}", lambda d=docs, b=b_doc: model.encode(d, batch_size=b), s_len, b_doc))

best = {}
max_t = max(int(t) for t in threads_list) if threads_list else 6
best_t = max_t
for wid, label, fn, s_len, b_n in workloads:
    if wid == "doc_s256" and len(threads_list) > 1:
        t_sweep = sorted((int(t) for t in threads_list), reverse=True)
        w_n, r_n = 0, 1
    elif wid in ("query_b1", "query_b32"):
        t_sweep = [best_t]
        w_n = 1 if cfg["warmup"] > 0 else 0
        r_n = 1
    else:
        t_sweep = [best_t]
        w_n, r_n = 0, 1
    if wid == "query_b1":
        toks = int(model.prepare_inputs([["What drives oceanic thermohaline heat transport across hemispheres?"]], "query")["input_ids"].shape[1])
    elif wid == "query_b32":
        toks = int(model.prepare_inputs([[f"Query {i}: summarize coastal pressure anomaly trends."] for i in range(32)], "query")["input_ids"].shape[1]) * 32
    else:
        docs_ref = [[build_text(s_len, i)] for i in range(b_n)]
        toks = int(model.prepare_inputs(docs_ref, "document")["attention_mask"].sum().item())
    for n_threads in t_sweep:
        torch.set_num_threads(n_threads)
        dt = median_t(fn, w_n, r_n)
        tok_s = toks / max(dt, 1e-9)
        if wid not in best or tok_s > best[wid]["tok_s"]:
            if wid == "doc_s256":
                best_t = n_threads
            fp_gb, rss_gb = footprint_gb()
            best[wid] = {
                "workload": wid,
                "label": label,
                "mode": "torch-cpu",
                "timing_scope": "e2e",
                "status": "ok",
                "tok_s": tok_s,
                "latency_ms": (dt * 1000.0) / max(b_n, 1),
                "peak_active_gb": 0.0,
                "peak_footprint_gb": fp_gb,
                "rss_gb": rss_gb,
                "notes": f"threads={n_threads},load={load_s:.2f}s",
            }

print(json.dumps({"mode": "torch-cpu", "status": "ok", "load_s": round(load_s, 3), "rows": list(best.values())}))
"""


_PPLX_CTX_MLX_SNIPPET = r"""
import ctypes, json, os, platform, resource, statistics, sys, types, time
import numpy as np

cfg = json.loads(sys.argv[1])
os.environ.setdefault("MLX_ENABLE_TF32", str(cfg.get("mlx_enable_tf32", "0")))
if cfg.get("repo_path"):
    sys.path.insert(0, cfg["repo_path"])

# Provide lightweight PIL stub if pillow is absent (text workloads never decode images).
try:
    from PIL import Image as _  # noqa: F401
except Exception:
    pil_mod = types.ModuleType("PIL")
    img_mod = types.ModuleType("PIL.Image")
    img_mod.Image = type("Image", (), {})
    pil_mod.Image = img_mod
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = img_mod

try:
    import mlx.core as mx
    from pplx_context_mlx import load_model
except Exception as exc:
    print(json.dumps({"mode": cfg["label_mode"], "status": "skipped", "reason": f"pplx-context-mlx unavailable ({exc})", "rows": []}))
    raise SystemExit(0)

def footprint_gb():
    if platform.system() == "Darwin":
        buf = ctypes.create_string_buffer(512)
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0:
            phys = int.from_bytes(buf.raw[240:248], "little") / 1e9
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
            return round(phys, 4), round(rss, 4)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    return round(rss, 4), round(rss, 4)

def median_t(fn, warmup, repeats):
    for _ in range(max(0, warmup)):
        _ = fn()
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = fn()
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))

phrase = (
    "Ocean currents transport thermal energy across subtropical gyres, modulating coastal "
    "pressure gradients and seasonal precipitation patterns over decadal timescales. "
)
def build_text(seq_tokens, seed_idx=0):
    repeats = max(1, (int(seq_tokens) + 15) // 22)
    return f"Sample record {seed_idx}. " + (phrase * repeats)

model_ref = cfg.get("model_dir") or "perplexity-ai/pplx-embed-v2-context-0.6b"
dtype = cfg["dtype"]
mode_lbl = cfg["label_mode"]
t_load0 = time.perf_counter()
try:
    eng = load_model(model_ref, dtype=dtype)
except Exception as exc:
    print(json.dumps({"mode": mode_lbl, "status": "skipped", "reason": f"load failed ({exc})", "rows": []}))
    raise SystemExit(0)
load_s = time.perf_counter() - t_load0

rows = []
q_short = [["What drives oceanic thermohaline heat transport across hemispheres?"]]
q1_toks = int(eng.tokenizer.encode_queries(q_short)["input_ids"].shape[1])
mx.reset_peak_memory()
dt_q1 = median_t(lambda: eng.encode_queries(q_short, batch_size=1), cfg["warmup"], cfg["repeats"])
fp_gb, rss_gb = footprint_gb()
rows.append({
    "workload": "query_b1",
    "label": f"{mode_lbl} query B=1 (S={q1_toks})",
    "mode": mode_lbl,
    "timing_scope": "e2e",
    "status": "ok",
    "tok_s": q1_toks / max(dt_q1, 1e-9),
    "latency_ms": dt_q1 * 1000.0,
    "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
    "peak_footprint_gb": fp_gb,
    "rss_gb": rss_gb,
    "notes": f"load={load_s:.2f}s,tf32={os.environ.get('MLX_ENABLE_TF32', '0')}",
})

q_batch = [[f"Query {i}: summarize coastal pressure anomaly trends."] for i in range(32)]
qb_toks = sum(int(row.shape[0]) for row in eng.tokenizer._encode(q_batch, is_query=True)[0]) if hasattr(eng.tokenizer, "_encode") else int(eng.tokenizer.encode_queries(q_batch)["input_ids"].size)
mx.reset_peak_memory()
dt_qb = median_t(lambda: eng.encode_queries(q_batch, batch_size=32), cfg["warmup"], cfg["repeats"])
fp_gb, rss_gb = footprint_gb()
rows.append({
    "workload": "query_b32",
    "label": f"{mode_lbl} query B=32",
    "mode": mode_lbl,
    "timing_scope": "e2e",
    "status": "ok",
    "tok_s": qb_toks / max(dt_qb, 1e-9),
    "latency_ms": (dt_qb * 1000.0) / 32.0,
    "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
    "peak_footprint_gb": fp_gb,
    "rss_gb": rss_gb,
    "notes": f"q/s={32.0 / max(dt_qb, 1e-9):.1f}",
})

for s_len in cfg["doc_lengths"]:
    b_doc = cfg["doc_batch"] if s_len <= 1024 else max(1, min(cfg["doc_batch"], 4))
    docs = [[build_text(s_len, i)] for i in range(b_doc)]
    enc_in = eng.tokenizer.encode_documents(docs)
    total_toks = int(np.asarray(enc_in["attention_mask"]).sum())
    mx.reset_peak_memory()
    dt_d = median_t(lambda d=docs, b=b_doc: eng.encode(d, batch_size=b), cfg["warmup"], cfg["repeats"])
    fp_gb, rss_gb = footprint_gb()
    rows.append({
        "workload": f"doc_s{s_len}",
        "label": f"{mode_lbl} doc B={b_doc} S={s_len}",
        "mode": mode_lbl,
        "timing_scope": "e2e",
        "status": "ok",
        "tok_s": total_toks / max(dt_d, 1e-9),
        "latency_ms": (dt_d * 1000.0) / b_doc,
        "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
        "peak_footprint_gb": fp_gb,
        "rss_gb": rss_gb,
        "notes": f"docs/s={b_doc / max(dt_d, 1e-9):.2f}",
    })

mixed = [[build_text(128, i)] for i in range(24)] + [[build_text(2048, 100 + i)] for i in range(4)]
m_in = eng.tokenizer.encode_documents(mixed)
m_toks = int(np.asarray(m_in["attention_mask"]).sum())
mx.reset_peak_memory()
dt_m = median_t(lambda: eng.encode(mixed, batch_size=8), cfg["warmup"], cfg["repeats"])
fp_gb, rss_gb = footprint_gb()
rows.append({
    "workload": "mixed_short_long",
    "label": f"{mode_lbl} mixed (24x~128 + 4x2048)",
    "mode": mode_lbl,
    "timing_scope": "e2e",
    "status": "ok",
    "tok_s": m_toks / max(dt_m, 1e-9),
    "latency_ms": dt_m * 1000.0,
    "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
    "peak_footprint_gb": fp_gb,
    "rss_gb": rss_gb,
    "notes": "public mlx GPU",
})

print(json.dumps({"mode": mode_lbl, "status": "ok", "load_s": round(load_s, 3), "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "0"), "rows": rows}))
"""


_TORCH_MPS_SNIPPET = r"""
import ctypes, json, os, platform, resource, statistics, sys, time
try:
    import torch
    from transformers import AutoModel
except Exception as exc:
    print(json.dumps({"mode": "torch-mps", "status": "skipped", "reason": f"torch/transformers unavailable ({exc})", "rows": []}))
    raise SystemExit(0)

if not torch.backends.mps.is_available():
    print(json.dumps({"mode": "torch-mps", "status": "skipped", "reason": "PyTorch MPS backend unavailable", "rows": []}))
    raise SystemExit(0)

cfg = json.loads(sys.argv[1])
model_ref = cfg.get("model_dir") or "perplexity-ai/pplx-embed-v2-context-0.6b"
phrase = (
    "Ocean currents transport thermal energy across subtropical gyres, modulating coastal "
    "pressure gradients and seasonal precipitation patterns over decadal timescales. "
)
def build_text(seq_tokens, seed_idx=0):
    repeats = max(1, (int(seq_tokens) + 15) // 22)
    return f"Sample record {seed_idx}. " + (phrase * repeats)

def footprint_gb():
    if platform.system() == "Darwin":
        buf = ctypes.create_string_buffer(512)
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0:
            phys = int.from_bytes(buf.raw[240:248], "little") / 1e9
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
            return round(phys, 4), round(rss, 4)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    return round(rss, 4), round(rss, 4)

def median_t(fn, warmup, repeats):
    for _ in range(max(0, warmup)):
        _ = fn(); torch.mps.synchronize()
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = fn(); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))

t_load0 = time.perf_counter()
try:
    model = AutoModel.from_pretrained(model_ref, trust_remote_code=True, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True).to("mps").eval()
except Exception as exc:
    print(json.dumps({"mode": "torch-mps", "status": "skipped", "reason": f"torch MPS load skipped ({exc})", "rows": []}))
    raise SystemExit(0)
load_s = time.perf_counter() - t_load0

rows = []
with torch.inference_mode():
    q1 = ["What drives oceanic thermohaline heat transport across hemispheres?"]
    q1_toks = int(model.prepare_inputs([q1], "query")["input_ids"].shape[1])
    dt_q1 = median_t(lambda: model.encode_queries([q1], batch_size=1), cfg["warmup"], cfg["repeats"])
    fp_gb, rss_gb = footprint_gb()
    rows.append({"workload": "query_b1", "label": f"torch-mps query B=1 (S={q1_toks})", "mode": "torch-mps", "timing_scope": "e2e", "status": "ok", "tok_s": q1_toks / max(dt_q1, 1e-9), "latency_ms": dt_q1 * 1000.0, "peak_active_gb": round(float(torch.mps.current_allocated_memory()) / 1e9, 4), "peak_footprint_gb": fp_gb, "rss_gb": rss_gb, "notes": f"load={load_s:.2f}s"})

    qb = [f"Query {i}: summarize coastal pressure anomaly trends." for i in range(32)]
    qb_toks = int(model.prepare_inputs([[x] for x in qb], "query")["input_ids"].shape[1]) * 32
    dt_qb = median_t(lambda: model.encode_queries([[x] for x in qb], batch_size=32), cfg["warmup"], cfg["repeats"])
    fp_gb, rss_gb = footprint_gb()
    rows.append({"workload": "query_b32", "label": "torch-mps query B=32", "mode": "torch-mps", "timing_scope": "e2e", "status": "ok", "tok_s": qb_toks / max(dt_qb, 1e-9), "latency_ms": (dt_qb * 1000.0) / 32.0, "peak_active_gb": round(float(torch.mps.current_allocated_memory()) / 1e9, 4), "peak_footprint_gb": fp_gb, "rss_gb": rss_gb, "notes": f"q/s={32.0 / max(dt_qb, 1e-9):.1f}"})

    for s_len in cfg["doc_lengths"]:
        b_doc = cfg["doc_batch"] if s_len <= 1024 else max(1, min(cfg["doc_batch"], 4))
        docs = [[build_text(s_len, i)] for i in range(b_doc)]
        toks = int(model.prepare_inputs(docs, "document")["attention_mask"].sum().item())
        dt_d = median_t(lambda d=docs, b=b_doc: model.encode(d, batch_size=b), cfg["warmup"], cfg["repeats"])
        fp_gb, rss_gb = footprint_gb()
        rows.append({"workload": f"doc_s{s_len}", "label": f"torch-mps doc B={b_doc} S={s_len}", "mode": "torch-mps", "timing_scope": "e2e", "status": "ok", "tok_s": toks / max(dt_d, 1e-9), "latency_ms": (dt_d * 1000.0) / b_doc, "peak_active_gb": round(float(torch.mps.current_allocated_memory()) / 1e9, 4), "peak_footprint_gb": fp_gb, "rss_gb": rss_gb, "notes": f"docs/s={b_doc / max(dt_d, 1e-9):.2f}"})

print(json.dumps({"mode": "torch-mps", "status": "ok", "load_s": round(load_s, 3), "rows": rows}))
"""


def _resolve_hf_cached_repo(repo_id: str) -> Path | None:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if snaps.is_dir():
        cands = sorted(snaps.iterdir())
        if cands:
            return cands[-1]
    return None


def _run_pplx_context_mlx_baseline(args: argparse.Namespace, *, dtype: str) -> dict[str, Any]:
    repo_path = getattr(args, "pplx_mlx_path", None) or os.environ.get("PPLX_CONTEXT_MLX_PATH")
    # Per pplx-context-mlx README, prefer pre-cast `zainmerchan/pplx-embed-v2-context-0.6b-mlx` for float16
    model_dir = (
        _resolve_hf_cached_repo("zainmerchan/pplx-embed-v2-context-0.6b-mlx")
        if (dtype == "float16" and getattr(args, "model_dir", None) is None)
        else None
    ) or _resolve_model_dir(args.model_dir, args.home)
    label = "pplx-context-mlx-f16" if dtype == "float16" else "pplx-context-mlx-f32"
    cfg = {
        "repo_path": str(Path(repo_path).expanduser().resolve()) if repo_path else None,
        "dtype": dtype,
        "label_mode": label,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "0"),
        "doc_lengths": _parse_int_csv(args.doc_lengths),
        "doc_batch": args.doc_batch,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "model_dir": str(model_dir) if model_dir is not None else None,
    }
    try:
        proc = subprocess.run([sys.executable, "-c", _PPLX_CTX_MLX_SNIPPET, json.dumps(cfg)], capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            reason = (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit={proc.returncode}")
            return {"mode": label, "status": "skipped", "reason": reason, "rows": []}
        return _extract_last_json_dict(proc.stdout)
    except Exception as exc:
        return {"mode": label, "status": "skipped", "reason": str(exc), "rows": []}


def _run_torch_mps_baseline(args: argparse.Namespace) -> dict[str, Any]:
    torch_py = getattr(args, "torch_python", None) or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    model_dir = _resolve_model_dir(args.model_dir, args.home)
    cfg = {
        "doc_lengths": _parse_int_csv(args.doc_lengths),
        "doc_batch": args.doc_batch,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "model_dir": str(model_dir) if model_dir is not None else None,
    }
    try:
        proc = subprocess.run([torch_py, "-c", _TORCH_MPS_SNIPPET, json.dumps(cfg)], capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            reason = (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit={proc.returncode}")
            return {"mode": "torch-mps", "status": "skipped", "reason": reason, "rows": []}
        return _extract_last_json_dict(proc.stdout)
    except Exception as exc:
        return {"mode": "torch-mps", "status": "skipped", "reason": str(exc), "rows": []}


def _run_torch_cpu_baseline(args: argparse.Namespace) -> dict[str, Any]:
    torch_py = getattr(args, "torch_python", None) or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    model_dir = _resolve_model_dir(args.model_dir, args.home)
    cfg = {
        "doc_lengths": _parse_int_csv(args.doc_lengths),
        "doc_batch": args.doc_batch,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch_threads": _parse_int_csv(getattr(args, "torch_threads", "1,2,4,6")),
        "model_dir": str(model_dir) if model_dir is not None else None,
    }
    try:
        proc = subprocess.run([torch_py, "-c", _TORCH_CPU_SNIPPET, json.dumps(cfg)], capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            reason = (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit={proc.returncode}")
            return {"mode": "torch-cpu", "status": "skipped", "reason": reason, "rows": []}
        return _extract_last_json_dict(proc.stdout)
    except Exception as exc:
        return {"mode": "torch-cpu", "status": "skipped", "reason": str(exc), "rows": []}


def _run_speed_single_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    if mode == "torch-cpu":
        return _run_torch_cpu_baseline(args)
    if mode == "torch-mps":
        return _run_torch_mps_baseline(args)
    if mode == "pplx-context-mlx-f16":
        return _run_pplx_context_mlx_baseline(args, dtype="float16")
    if mode == "pplx-context-mlx-f32":
        return _run_pplx_context_mlx_baseline(args, dtype="float32")

    from fast_pplx.engine import FastPplx

    engine_mode, eff_dtype, eff_mapped = _resolve_speed_mode_spec(mode, args)
    doc_lengths = _parse_int_csv(args.doc_lengths)
    timing_scope = getattr(args, "timing_scope", "e2e")
    if engine_mode in ("ane", "hetero", "hetero2"):
        ane_ok, ane_note = _check_ane_ready(args.home)
        if not ane_ok:
            return {"mode": mode, "status": "skipped", "reason": ane_note, "rows": []}

    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"mode": mode, "status": "skipped", "reason": "no local HF snapshot or staged weights found", "rows": []}

    rows: list[dict[str, Any]] = []
    t_load0 = time.perf_counter()
    with _wired_limit_ctx():
        with FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=engine_mode,
            dtype=eff_dtype,
            use_mapped_embedding=eff_mapped,
            local_files_only=True,
        ) as eng:
            load_s = time.perf_counter() - t_load0
            q_short = ["What drives oceanic thermohaline heat transport across hemispheres?"]
            q_batch = [f"Query {i}: summarize coastal pressure anomaly trends." for i in range(32)]

            def _measure_pair(
                *,
                total_toks: int,
                denom_items: int,
                fn_e2e: Any,
                fn_fwd: Any,
                tok_fn: Any,
            ) -> tuple[float, float, float, float]:
                t_tok0 = time.perf_counter()
                _ = tok_fn()
                tok_ms = (time.perf_counter() - t_tok0) * 1000.0
                _reset_mlx_peak()
                med_e2e, _ = _median_timed(fn_e2e, warmup=args.warmup, repeats=args.repeats)
                med_fwd, _ = _median_timed(fn_fwd, warmup=max(0, args.warmup), repeats=args.repeats)
                primary_dt = med_e2e if timing_scope == "e2e" else med_fwd
                return primary_dt, med_e2e, med_fwd, tok_ms

            # 1. B=1 Query (symmetric E2E + forward breakdown)
            q1_ids = [eng.tokenizer.encode_query(q_short[0])]
            q1_toks = len(q1_ids[0])
            dt_q1, e2e_q1, fwd_q1, tok_q1_ms = _measure_pair(
                total_toks=q1_toks,
                denom_items=1,
                fn_e2e=lambda: eng.encode_queries(q_short, batch_size=1, dimensions=args.dimensions, mode=engine_mode),
                fn_fwd=lambda: eng.encode_tokens(q1_ids, batch_size=1, dimensions=args.dimensions, mode=engine_mode),
                tok_fn=lambda: [eng.tokenizer.encode_query(q_short[0])],
            )
            fp = _footprint_snapshot()
            rows.append(
                {
                    "workload": "query_b1",
                    "label": f"{mode} query B=1 (S={q1_toks})",
                    "mode": mode,
                    "timing_scope": timing_scope,
                    "status": "ok",
                    "tok_s": q1_toks / max(dt_q1, 1e-9),
                    "e2e_tok_s": round(q1_toks / max(e2e_q1, 1e-9), 1),
                    "forward_tok_s": round(q1_toks / max(fwd_q1, 1e-9), 1),
                    "tokenize_ms": round(tok_q1_ms, 3),
                    "latency_ms": dt_q1 * 1000.0,
                    "peak_active_gb": _mlx_peak_gb(),
                    "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
                    "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
                    "notes": f"load={load_s:.2f}s,scope={timing_scope}",
                }
            )
            _clear_arena(256)

            # 2. B=32 Queries
            qb_ids = [eng.tokenizer.encode_query(t) for t in q_batch]
            qb_toks = sum(len(row) for row in qb_ids)
            dt_qb, e2e_qb, fwd_qb, tok_qb_ms = _measure_pair(
                total_toks=qb_toks,
                denom_items=32,
                fn_e2e=lambda: eng.encode_queries(q_batch, batch_size=32, dimensions=args.dimensions, mode=engine_mode),
                fn_fwd=lambda: eng.encode_tokens(qb_ids, batch_size=32, dimensions=args.dimensions, mode=engine_mode),
                tok_fn=lambda: [eng.tokenizer.encode_query(t) for t in q_batch],
            )
            fp = _footprint_snapshot()
            rows.append(
                {
                    "workload": "query_b32",
                    "label": f"{mode} query B=32 (~{qb_toks // 32} tok/q)",
                    "mode": mode,
                    "timing_scope": timing_scope,
                    "status": "ok",
                    "tok_s": qb_toks / max(dt_qb, 1e-9),
                    "e2e_tok_s": round(qb_toks / max(e2e_qb, 1e-9), 1),
                    "forward_tok_s": round(qb_toks / max(fwd_qb, 1e-9), 1),
                    "tokenize_ms": round(tok_qb_ms, 3),
                    "latency_ms": (dt_qb * 1000.0) / 32.0,
                    "peak_active_gb": _mlx_peak_gb(),
                    "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
                    "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
                    "notes": f"q/s={32.0 / max(dt_qb, 1e-9):.1f}",
                }
            )
            _clear_arena(256)

            # 3. Document length sweep (skip >512 for strict `ane` mode)
            max_ane = eng.ane_query_max_len if engine_mode == "ane" else 32768
            for s_len in doc_lengths:
                if s_len > max_ane:
                    rows.append(
                        {
                            "workload": f"doc_s{s_len}",
                            "label": f"{mode} doc S={s_len}",
                            "mode": mode,
                            "timing_scope": timing_scope,
                            "status": "skipped",
                            "reason": f"exceeds ANE bucket cap {max_ane}",
                        }
                    )
                    continue
                b_doc = args.doc_batch if s_len <= 1024 else max(1, min(args.doc_batch, 4))
                docs = [_build_synthetic_text(s_len, i) for i in range(b_doc)]
                tok_lists = [eng.tokenizer.encode_document(d, max_length=s_len) for d in docs]
                total_toks = sum(len(t) for t in tok_lists)
                dt_d, e2e_d, fwd_d, tok_d_ms = _measure_pair(
                    total_toks=total_toks,
                    denom_items=b_doc,
                    fn_e2e=lambda d=docs, b=b_doc, sl=s_len: eng.encode_documents(
                        d, batch_size=b, max_length=sl, dimensions=args.dimensions, mode=engine_mode
                    ),
                    fn_fwd=lambda tl=tok_lists, b=b_doc: eng.encode_tokens(
                        tl, batch_size=b, dimensions=args.dimensions, mode=engine_mode
                    ),
                    tok_fn=lambda d=docs, sl=s_len: [eng.tokenizer.encode_document(x, max_length=sl) for x in d],
                )
                fp = _footprint_snapshot()
                rows.append(
                    {
                        "workload": f"doc_s{s_len}",
                        "label": f"{mode} doc B={b_doc} S={s_len}",
                        "mode": mode,
                        "timing_scope": timing_scope,
                        "status": "ok",
                        "tok_s": total_toks / max(dt_d, 1e-9),
                        "e2e_tok_s": round(total_toks / max(e2e_d, 1e-9), 1),
                        "forward_tok_s": round(total_toks / max(fwd_d, 1e-9), 1),
                        "tokenize_ms": round(tok_d_ms, 3),
                        "latency_ms": (dt_d * 1000.0) / b_doc,
                        "peak_active_gb": _mlx_peak_gb(),
                        "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
                        "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
                        "notes": f"docs/s={b_doc / max(dt_d, 1e-9):.2f}",
                    }
                )
                _clear_arena(256)

            # 4. Mixed short+long workload highlighting hetero / hetero2 concurrent scheduling
            if engine_mode in ("gpu", "hetero", "hetero2"):
                mixed_docs = [_build_synthetic_text(128, i) for i in range(24)] + [
                    _build_synthetic_text(2048, 100 + i) for i in range(4)
                ]
                mixed_toks = [eng.tokenizer.encode_document(d, max_length=256) for d in mixed_docs[:24]] + [
                    eng.tokenizer.encode_document(d, max_length=2048) for d in mixed_docs[24:]
                ]
                m_total = sum(len(t) for t in mixed_toks)
                dt_m, e2e_m, fwd_m, tok_m_ms = _measure_pair(
                    total_toks=m_total,
                    denom_items=1,
                    fn_e2e=lambda: eng.encode_documents(
                        mixed_docs, batch_size=8, max_length=2048, dimensions=args.dimensions, mode=engine_mode
                    ),
                    fn_fwd=lambda: eng.encode_tokens(
                        mixed_toks, batch_size=8, dimensions=args.dimensions, mode=engine_mode
                    ),
                    tok_fn=lambda: [eng.tokenizer.encode_document(d, max_length=2048) for d in mixed_docs],
                )
                fp = _footprint_snapshot()
                st = dict(eng.last_stats)
                rows.append(
                    {
                        "workload": "mixed_short_long",
                        "label": f"{mode} mixed (24x~128 + 4x2048)",
                        "mode": mode,
                        "timing_scope": timing_scope,
                        "status": "ok",
                        "tok_s": m_total / max(dt_m, 1e-9),
                        "e2e_tok_s": round(m_total / max(e2e_m, 1e-9), 1),
                        "forward_tok_s": round(m_total / max(fwd_m, 1e-9), 1),
                        "tokenize_ms": round(tok_m_ms, 3),
                        "latency_ms": dt_m * 1000.0,
                        "peak_active_gb": _mlx_peak_gb(),
                        "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
                        "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
                        "notes": f"ane={st.get('ane_items', 0)},gpu={st.get('gpu_items', 0)}",
                    }
                )
                _clear_arena(256)

    return {"mode": mode, "status": "ok", "load_s": round(load_s, 3), "rows": rows}


def cmd_speed(args: argparse.Namespace) -> dict[str, Any]:
    modes = _parse_str_csv(args.modes)
    timing_scope = getattr(args, "timing_scope", "e2e")
    plan = {
        "suite": "speed",
        "modes": modes,
        "dtype": args.dtype,
        "timing_scope": timing_scope,
        "doc_lengths": _parse_int_csv(args.doc_lengths),
        "isolate": bool(args.isolate),
        "telemetry": _collect_telemetry(),
        "quiescence_warnings": _sys_quiescence(),
        "ane_status": _check_ane_ready(args.home),
        "model_dir": str(_resolve_model_dir(args.model_dir, args.home)),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    _preflight_quiescence(allow_contended=args.allow_contended)
    all_rows: list[dict[str, Any]] = []
    mode_results: list[dict[str, Any]] = []
    abba_report: dict[str, Any] | None = None
    with _hw_lock():
        for mode in modes:
            if args.isolate:
                child_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "speed",
                    "--modes",
                    mode,
                    "--dtype",
                    args.dtype,
                    "--timing-scope",
                    timing_scope,
                    "--doc-lengths",
                    args.doc_lengths,
                    "--doc-batch",
                    str(args.doc_batch),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--torch-threads",
                    getattr(args, "torch_threads", "1,2,4,6"),
                    "--no-isolate",
                    "--json-only",
                ]
                if args.allow_contended:
                    child_cmd.append("--allow-contended")
                if getattr(args, "enable_tf32", False):
                    child_cmd.append("--enable-tf32")
                if getattr(args, "torch_python", None):
                    child_cmd.extend(["--torch-python", str(args.torch_python)])
                if getattr(args, "pplx_mlx_path", None):
                    child_cmd.extend(["--pplx-mlx-path", str(args.pplx_mlx_path)])
                if args.home:
                    child_cmd.extend(["--home", str(args.home)])
                if args.model_dir:
                    child_cmd.extend(["--model-dir", str(args.model_dir)])
                if args.dimensions is not None:
                    child_cmd.extend(["--dimensions", str(args.dimensions)])
                if args.no_mapped_embedding:
                    child_cmd.append("--no-mapped-embedding")
                proc = subprocess.run(child_cmd, capture_output=True, text=True, check=True)
                parsed = _extract_last_json_dict(proc.stdout)
                mode_results.extend(parsed.get("mode_results", []))
                all_rows.extend(parsed.get("rows", []))
            else:
                res = _run_speed_single_mode(args, mode)
                mode_results.append(res)
                if res["rows"]:
                    all_rows.extend(res["rows"])
                elif res["status"] != "ok":
                    all_rows.append({"label": f"{mode}", **res})

        abba_pair = getattr(args, "abba_pair", None)
        if abba_pair:
            pair = _parse_str_csv(abba_pair)
            if len(pair) == 2:
                abba_report = run_abba_speed_pair(args, pair[0], pair[1], rounds=max(2, args.repeats))
        elif getattr(args, "abba", False) and not args.isolate:
            internal = [m for m in modes if m in ("gpu", "gpu-f16", "gpu-bf16", "gpu-no-mmap", "ane", "hetero", "hetero2")]
            if len(internal) >= 2:
                abba_report = run_abba_speed_pair(args, internal[0], internal[1], rounds=max(2, args.repeats))

    annotated = annotate_speed_comparisons(all_rows)
    out: dict[str, Any] = {
        "suite": "speed",
        "timing_scope": timing_scope,
        "telemetry": plan["telemetry"],
        "quiescence_warnings": plan["quiescence_warnings"],
        "mode_results": mode_results,
        "rows": annotated,
        "markdown": format_speed_markdown(annotated),
    }
    if abba_report is not None:
        out["abba"] = abba_report
    return out


def _eval_s4096_streamed_parity(eng: Any) -> dict[str, Any]:
    """Compare streamed (S > 2048) vs whole forward pass at S=4096 on the live GPU engine."""
    import fast_pplx.gpu as pplx_gpu_mod

    text = _build_synthetic_text(4200, seed_idx=4096)
    toks = eng.tokenizer.encode_document(text, max_length=4096)
    assert len(toks) == 4096 or len(toks) > 2048

    orig_thresh = pplx_gpu_mod._STREAMED_SEQ_THRESHOLD
    try:
        pplx_gpu_mod._STREAMED_SEQ_THRESHOLD = 10_000
        whole_p = eng.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False, mode="gpu", apply_tanh=False)
        whole_i = eng.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False, mode="gpu")

        pplx_gpu_mod._STREAMED_SEQ_THRESHOLD = 2048
        stream_p = eng.encode_tokens([toks], batch_size=1, output_dtype="float32", normalize=False, mode="gpu", apply_tanh=False)
        stream_i = eng.encode_tokens([toks], batch_size=1, output_dtype="int8", normalize=False, mode="gpu")
    finally:
        pplx_gpu_mod._STREAMED_SEQ_THRESHOLD = orig_thresh
        _clear_arena(256)

    vp = compute_vector_parity(stream_p, whole_p)
    ip = compute_int8_parity(stream_i, whole_i)
    return {
        "seq_len": len(toks),
        "proj_max_abs_err": round(float(vp["max_abs_err"]), 6),
        "proj_min_cos": round(float(vp["min_cos"]), 6),
        "int8_max_abs_err": int(ip["max_abs_err"]),
        "int8_exact_fraction": round(float(ip["exact_fraction"]), 4),
        "int8_min_cos": round(float(ip["min_cos"]), 6),
        "passed": bool(vp["min_cos"] >= 0.9999 and int(ip["max_abs_err"]) <= 1),
    }


def cmd_parity(args: argparse.Namespace) -> dict[str, Any]:
    cases_path = _FIXTURES_DIR / "cases.json"
    ref_path = _FIXTURES_DIR / "reference.npz"
    modes = _parse_str_csv(args.modes)
    plan = {
        "suite": "parity",
        "modes": modes,
        "dtype": args.dtype,
        "include_s4096": bool(getattr(args, "include_s4096", True)),
        "telemetry": _collect_telemetry(),
        "cases_file": str(cases_path),
        "reference_file": str(ref_path),
        "quiescence_warnings": _sys_quiescence(),
        "ane_status": _check_ane_ready(args.home),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    _preflight_quiescence(allow_contended=args.allow_contended)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    ref = np.load(ref_path)
    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"suite": "parity", "status": "skipped", "reason": "no local HF snapshot cached", "telemetry": plan["telemetry"]}

    from fast_pplx.engine import FastPplx

    reports: list[dict[str, Any]] = []
    with _hw_lock(), _wired_limit_ctx():
        for mode in modes:
            engine_mode, eff_dtype, eff_mapped = _resolve_speed_mode_spec(mode, args)
            if engine_mode in ("ane", "hetero", "hetero2"):
                ok_ane, reason = _check_ane_ready(args.home)
                if not ok_ane:
                    reports.append({"mode": mode, "status": "skipped", "reason": reason})
                    continue

            with FastPplx(
                home=args.home,
                model_dir=model_dir,
                mode=engine_mode,
                dtype=eff_dtype,
                use_mapped_embedding=eff_mapped,
                deterministic=(engine_mode == "gpu"),
                local_files_only=True,
            ) as eng:
                proj_errs: list[float] = []
                proj_cos: list[float] = []
                i8_errs: list[int] = []
                i8_exact: list[float] = []
                i8_cos: list[float] = []
                evaluated = 0
                skipped_len = 0

                for case in cases:
                    cid = case["id"]
                    docs = case["documents"]
                    if engine_mode == "ane":
                        if case["task"] == "query":
                            lens = [len(eng.tokenizer.encode_query(row[0])) for row in docs]
                        else:
                            lens = [len(eng.tokenizer.encode_contextual_document(d)[0]) for d in docs]
                        if any(ln > eng.ane_query_max_len for ln in lens):
                            skipped_len += 1
                            continue

                    if case["task"] == "query":
                        flat_q = [row[0] for row in docs]
                        got_p = eng.encode_queries(flat_q, output_dtype="float32", normalize=False, mode=engine_mode, apply_tanh=False)
                        got_i = eng.encode_queries(flat_q, output_dtype="int8", normalize=False, mode=engine_mode)
                        exp_p = ref[f"{cid}/projected"].reshape(-1, 1024)
                        exp_i = np.concatenate([ref[f"{cid}/encoded/{k}"] for k in range(len(flat_q))], axis=0)
                        vp = compute_vector_parity(got_p, exp_p)
                        ip = compute_int8_parity(got_i, exp_i)
                        proj_errs.append(float(vp["max_abs_err"]))
                        proj_cos.append(float(vp["min_cos"]))
                        i8_errs.append(int(ip["max_abs_err"]))
                        i8_exact.append(float(ip["exact_fraction"]))
                        i8_cos.append(float(ip["min_cos"]))
                        evaluated += 1
                    else:
                        got_p_list = eng.encode_contextual_chunks(docs, output_dtype="float32", normalize=False, mode=engine_mode, apply_tanh=False)
                        got_i_list = eng.encode_contextual_chunks(docs, output_dtype="int8", normalize=False, mode=engine_mode)
                        exp_p_all = ref[f"{cid}/projected"]
                        for di, doc in enumerate(docs):
                            if not doc:
                                continue
                            non_empty = [ci for ci, c in enumerate(doc) if c]
                            if not non_empty:
                                continue
                            rp = exp_p_all[di, non_empty]
                            gp = got_p_list[di][non_empty]
                            ri = ref[f"{cid}/encoded/{di}"][non_empty]
                            gi = got_i_list[di][non_empty]
                            vp = compute_vector_parity(gp, rp)
                            ip = compute_int8_parity(gi, ri)
                            proj_errs.append(float(vp["max_abs_err"]))
                            proj_cos.append(float(vp["min_cos"]))
                            i8_errs.append(int(ip["max_abs_err"]))
                            i8_exact.append(float(ip["exact_fraction"]))
                            i8_cos.append(float(ip["min_cos"]))
                        evaluated += 1

                min_proj_cos = round(min(proj_cos) if proj_cos else 1.0, 6)
                is_det_gpu = engine_mode == "gpu"
                rep: dict[str, Any] = {
                    "mode": mode,
                    "dtype": eff_dtype,
                    "parity_tier": "deterministic_gpu" if is_det_gpu else "ane_drift_tolerant",
                    "status": "ok",
                    "cases_evaluated": evaluated,
                    "cases_skipped_ane_cap": skipped_len,
                    "proj_max_abs_err": round(max(proj_errs) if proj_errs else 0.0, 6),
                    "proj_min_cos": min_proj_cos,
                    "int8_max_abs_err": max(i8_errs) if i8_errs else 0,
                    "int8_exact_fraction": round(sum(i8_exact) / max(len(i8_exact), 1), 4),
                    "int8_min_cos": round(min(i8_cos) if i8_cos else 1.0, 6),
                    "within_strict_parity": bool(min_proj_cos >= 0.9998),
                    "ane_drift_warning": bool((not is_det_gpu) and min_proj_cos < 0.9998),
                }
                if is_det_gpu and getattr(args, "include_s4096", True):
                    rep["s4096_streamed_parity"] = _eval_s4096_streamed_parity(eng)
                reports.append(rep)
                _clear_arena(256)

    return {"suite": "parity", "telemetry": plan["telemetry"], "reports": reports}


def _run_context_limit_worker(args: argparse.Namespace) -> dict[str, Any]:
    from fast_pplx.engine import FastPplx

    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"target_tokens": args.single_length, "status": "skipped", "reason": "missing weights"}

    target = int(args.single_length)
    with _wired_limit_ctx():
        with FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=args.mode,
            dtype=args.dtype,
            use_mapped_embedding=not args.no_mapped_embedding,
            local_files_only=True,
        ) as eng:
            raw_text = _build_synthetic_text(target + 256, 7)
            toks = eng.tokenizer.encode_document(raw_text, max_length=target)
            _ = eng.encode_tokens([toks[: min(128, len(toks))]], batch_size=1, mode=args.mode)
            _clear_arena(256)
            _reset_mlx_peak()
            t0 = time.perf_counter()
            vecs = eng.encode_tokens([toks], batch_size=1, dimensions=args.dimensions, mode=args.mode)
            dt = time.perf_counter() - t0
            fp = _footprint_snapshot()
            _clear_arena(256)
            return {
                "target_tokens": target,
                "actual_tokens": len(toks),
                "elapsed_s": round(dt, 4),
                "tok_s": round(len(toks) / max(dt, 1e-9), 1),
                "peak_active_gb": _mlx_peak_gb(),
                "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
                "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
                "finite": bool(np.all(np.isfinite(vecs))),
                "status": "ok" if np.all(np.isfinite(vecs)) else "non_finite",
            }


def cmd_context_limit(args: argparse.Namespace) -> dict[str, Any]:
    if args.single_length is not None:
        return _run_context_limit_worker(args)

    lengths = _parse_int_csv(args.lengths)
    plan = {
        "suite": "context-limit",
        "mode": args.mode,
        "dtype": args.dtype,
        "lengths": lengths,
        "telemetry": _collect_telemetry(),
        "quiescence_warnings": _sys_quiescence(),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    _preflight_quiescence(allow_contended=args.allow_contended)
    rows: list[dict[str, Any]] = []
    with _hw_lock():
        for length in lengths:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "context-limit",
                "--single-length",
                str(length),
                "--mode",
                args.mode,
                "--dtype",
                args.dtype,
                "--json-only",
            ]
            if args.allow_contended:
                cmd.append("--allow-contended")
            if getattr(args, "enable_tf32", False):
                cmd.append("--enable-tf32")
            if args.home:
                cmd.extend(["--home", str(args.home)])
            if args.model_dir:
                cmd.extend(["--model-dir", str(args.model_dir)])
            if args.dimensions is not None:
                cmd.extend(["--dimensions", str(args.dimensions)])
            if args.no_mapped_embedding:
                cmd.append("--no-mapped-embedding")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_sec, check=True)
                rows.append(_extract_last_json_dict(proc.stdout))
            except Exception as exc:
                rows.append({"target_tokens": length, "status": "error", "reason": str(exc)})
                break
    annotated_cl = annotate_context_limit_comparisons(rows)
    return {
        "suite": "context-limit",
        "telemetry": plan["telemetry"],
        "rows": annotated_cl,
        "markdown": format_context_limit_markdown(annotated_cl),
    }


def build_synthetic_context_corpus(filler_repeats: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build deterministic multi-chunk documents where only contextual windows resolve ambiguous targets."""
    docs: list[dict[str, Any]] = []
    queries: list[dict[str, str]] = []
    for entry in _SYNTHETIC_CONTEXT_DOCS:
        chunks = [entry["preamble"]]
        chunks.extend([entry["filler"]] * max(1, int(filler_repeats)))
        chunks.append(entry["target"])
        docs.append({"doc_id": entry["doc_id"], "chunks": chunks})
        queries.append({"doc_id": entry["doc_id"], "query": entry["query"]})
    return docs, queries


def cmd_context_windows(args: argparse.Namespace) -> dict[str, Any]:
    budgets = _parse_int_csv(args.context_budgets)
    rep_tok_arg = getattr(args, "repeat_header_tokens", 128)
    rep_tokens = 128 if rep_tok_arg is None else int(rep_tok_arg)
    ov_chunks = int(getattr(args, "window_overlap_chunks", 0))
    plan = {
        "suite": "context-windows",
        "mode": args.mode,
        "dtype": args.dtype,
        "context_budgets": budgets,
        "filler_repeats": args.filler_repeats,
        "repeat_header_tokens": rep_tokens,
        "window_overlap_chunks": ov_chunks,
        "telemetry": _collect_telemetry(),
        "quiescence_warnings": _sys_quiescence(),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    _preflight_quiescence(allow_contended=args.allow_contended)
    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"suite": "context-windows", "status": "skipped", "reason": "missing weights", "telemetry": plan["telemetry"]}

    from fast_pplx.engine import FastPplx

    docs, queries = build_synthetic_context_corpus(filler_repeats=args.filler_repeats)
    rows: list[dict[str, Any]] = []
    with _hw_lock(), _wired_limit_ctx():
        with FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=args.mode,
            dtype=args.dtype,
            use_mapped_embedding=not args.no_mapped_embedding,
            local_files_only=True,
        ) as eng:
            q_vecs = eng.encode_queries(
                [q["query"] for q in queries], dimensions=args.dimensions, normalize=True, mode=args.mode
            )
            q_ids = [q["doc_id"] for q in queries]

            # Baseline 0: isolated non-contextual chunks
            t0 = time.perf_counter()
            iso_texts: list[str] = []
            iso_doc_ids: list[str] = []
            for d in docs:
                for ch in d["chunks"]:
                    iso_texts.append(ch)
                    iso_doc_ids.append(d["doc_id"])
            iso_vecs = eng.encode_documents(iso_texts, batch_size=16, dimensions=args.dimensions, normalize=True, mode=args.mode)
            dt_iso = time.perf_counter() - t0
            iso_toks = sum(len(eng.tokenizer.encode_document(t)) for t in iso_texts)
            rows.append(
                {
                    "window_budget": "isolated",
                    "windows": len(iso_texts),
                    "tok_s": round(iso_toks / max(dt_iso, 1e-9), 1),
                    **score_retrieval_metrics(q_vecs, iso_vecs, iso_doc_ids, q_ids),
                }
            )
            _clear_arena(256)

            # Sweep contextual window token budgets
            for budget in budgets:
                t1 = time.perf_counter()
                windows_chunks: list[list[str]] = []
                win_emit_meta: list[tuple[str, int, int]] = []
                total_toks = 0
                for d in docs:
                    packed = eng.tokenizer.group_chunks_by_context_windows(
                        d["chunks"],
                        context_tokens=budget,
                        repeat_header_tokens=rep_tokens if rep_tokens > 0 else None,
                        window_overlap_chunks=ov_chunks,
                    )
                    body_cursor = 0
                    for w_idx, win in enumerate(packed):
                        windows_chunks.append(win)
                        has_anchor = (
                            rep_tokens > 0
                            and w_idx > 0
                            and body_cursor < len(d["chunks"])
                            and len(win) >= 2
                            and win[0] != d["chunks"][body_cursor]
                        )
                        min_prefix = 1 if has_anchor else 0
                        n_rem = len(d["chunks"]) - body_cursor
                        max_ov = min(ov_chunks, body_cursor) if w_idx > 0 else 0
                        emit_st = min_prefix
                        if max_ov > 0:
                            for cand_ov in range(max_ov, -1, -1):
                                st_k = min_prefix + cand_ov
                                cand_n = len(win) - st_k
                                if 1 <= cand_n <= n_rem and win[st_k:] == d["chunks"][body_cursor : body_cursor + cand_n]:
                                    emit_st = st_k
                                    break
                        n_body = len(win) - emit_st
                        win_emit_meta.append((d["doc_id"], emit_st, n_body))
                        body_cursor += n_body
                        total_toks += len(eng.tokenizer.encode_contextual_document(win)[0])
                vec_lists = eng.encode_contextual_chunks(
                    windows_chunks, batch_size=4, dimensions=args.dimensions, normalize=True, mode=args.mode
                )
                emitted_vecs: list[np.ndarray] = []
                chunk_doc_ids: list[str] = []
                for vec_mat, (doc_id, emit_st, n_body) in zip(vec_lists, win_emit_meta, strict=True):
                    arr = np.asarray(vec_mat)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    emitted_vecs.append(arr[emit_st : emit_st + n_body])
                    chunk_doc_ids.extend([doc_id] * n_body)
                c_vecs = np.concatenate(emitted_vecs, axis=0)
                dt_w = time.perf_counter() - t1
                rows.append(
                    {
                        "window_budget": str(budget),
                        "windows": len(windows_chunks),
                        "tok_s": round(total_toks / max(dt_w, 1e-9), 1),
                        **score_retrieval_metrics(q_vecs, c_vecs, chunk_doc_ids, q_ids),
                    }
                )
                _clear_arena(256)

    annotated_cw = annotate_context_window_comparisons(rows)
    return {
        "suite": "context-windows",
        "telemetry": plan["telemetry"],
        "rows": annotated_cw,
        "markdown": format_context_windows_markdown(annotated_cw),
    }


def cmd_bit_identity(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "compare":
        res = compare_bit_identity(Path(args.baseline_dir), Path(args.candidate_dir))
        return {"suite": "bit-identity", "action": "compare", **res}

    if args.dry_run:
        return {
            "dry_run": True,
            "suite": "bit-identity",
            "action": "dump",
            "mode": args.mode,
            "telemetry": _collect_telemetry(),
            "out_dir": str(args.out_dir),
        }

    _preflight_quiescence(allow_contended=args.allow_contended)
    ready, model_dir = _resolve_engine_model_dir(args.model_dir, args.home)
    if not ready:
        return {"suite": "bit-identity", "action": "dump", "status": "skipped", "reason": "missing weights"}

    cases = json.loads((_FIXTURES_DIR / "cases.json").read_text(encoding="utf-8"))
    from fast_pplx.engine import FastPplx

    vectors: dict[str, np.ndarray] = {}
    with _hw_lock(), _wired_limit_ctx():
        with FastPplx(
            home=args.home,
            model_dir=model_dir,
            mode=args.mode,
            dtype=args.dtype,
            use_mapped_embedding=not args.no_mapped_embedding,
            deterministic=(args.mode == "gpu"),
            local_files_only=True,
        ) as eng:
            for case in cases:
                cid = case["id"]
                if case["task"] == "query":
                    flat_q = [row[0] for row in case["documents"]]
                    vectors[f"{cid}/projected"] = eng.encode_queries(
                        flat_q, output_dtype="float32", normalize=False, mode=args.mode, apply_tanh=False
                    )
                else:
                    mats = eng.encode_contextual_chunks(
                        case["documents"], output_dtype="float32", normalize=False, mode=args.mode, apply_tanh=False
                    )
                    for di, mat in enumerate(mats):
                        vectors[f"{cid}/doc_{di}"] = mat
    telemetry = _collect_telemetry()
    manifest = dump_bit_identity(vectors, {"mode": args.mode, "dtype": args.dtype, "telemetry": telemetry}, Path(args.out_dir))
    return {"suite": "bit-identity", "action": "dump", "telemetry": telemetry, "manifest": str(manifest), "n_tensors": len(vectors)}


def format_ane_profiles_markdown(
    compile_sec: dict[str, float],
    parity: dict[str, dict[str, Any]],
    perf: dict[str, Any],
    stock_mlx: dict[str, Any] | None = None,
) -> str:
    lines = [
        "| Profile / Baseline | Compile (s) | S=128 tok/s | S=256 tok/s | S=256 f32 min cos | S=256 MRL-128 min cos | S=256 int8 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    s128_prof = (perf.get("S128_32docs") or {}).get("profiles") or {}
    s256_prof = (perf.get("S256_32docs") or {}).get("profiles") or {}
    for prof, p_metrics in parity.items():
        t128 = s128_prof.get(prof, {}).get("tok_s")
        t256 = s256_prof.get(prof, {}).get("tok_s")
        lines.append(
            f"| `ane:{prof}` | {compile_sec.get(prof, 0.0):.2f} | "
            f"{f'{t128:,.1f}' if t128 is not None else '-'} | "
            f"{f'{t256:,.1f}' if t256 is not None else '-'} | "
            f"{p_metrics.get('s256_f32_min_cos', 0.0):.6f} | "
            f"{p_metrics.get('s256_d128_min_cos', 0.0):.6f} | "
            f"{p_metrics.get('s256_int8_mae', 0.0):.4f} |"
        )
    if isinstance(stock_mlx, dict) and stock_mlx.get("status") == "ok":
        st128 = (stock_mlx.get("S128_32docs") or {}).get("tok_s")
        st256 = (stock_mlx.get("S256_32docs") or {}).get("tok_s")
        lines.append(
            f"| `{stock_mlx.get('harness', 'pplx-context-mlx-f16')}` | - | "
            f"{f'{st128:,.1f}' if st128 is not None else '-'} | "
            f"{f'{st256:,.1f}' if st256 is not None else '-'} | "
            "ref | ref | ref |"
        )
    return "\n".join(lines)


def cmd_ane_profiles(args: argparse.Namespace) -> dict[str, Any]:
    """Compile and ABBA-benchmark `fast`, `balanced`, and `parity` ANE profiles in isolated temp homes."""
    import statistics
    import tempfile

    profiles = _parse_str_csv(args.profiles)
    seq_lens = _parse_int_csv(args.seq_lens)
    telemetry = _collect_telemetry()
    quiescence = _sys_quiescence()

    if args.dry_run:
        return {
            "dry_run": True,
            "suite": "ane-profiles",
            "profiles": profiles,
            "seq_lens": seq_lens,
            "repeats": int(args.repeats),
            "telemetry": telemetry,
            "quiescence_warnings": quiescence,
            "ane_status": _check_ane_ready(args.home),
        }

    if quiescence and not args.allow_contended:
        raise RuntimeError(f"Contention detected ({len(quiescence)} processes); pass --allow-contended to override.")

    from fast_pplx.engine import FastPplx, read_ane_manifest
    from fast_pplx.paths import resolve_paths

    prod_paths = resolve_paths(args.home)
    build_script = _PACKAGE_ROOT / "scripts" / "build_ane_models.py"
    seq_csv = ",".join(str(x) for x in seq_lens)

    with _gpu_lock():
        with tempfile.TemporaryDirectory(prefix="pplx_ane_profiles_") as tmp_s:
            tmp_root = Path(tmp_s)
            homes: dict[str, Path] = {}
            compile_sec: dict[str, float] = {}
            manifests: dict[str, Any] = {}

            for prof in profiles:
                h = tmp_root / f"home_{prof}"
                (h / "artifacts").mkdir(parents=True, exist_ok=True)
                for sub in ("weights", "tokenizer"):
                    src_sub = prod_paths.home / "artifacts" / sub
                    if src_sub.exists():
                        (h / "artifacts" / sub).symlink_to(src_sub)
                if (prod_paths.home / "build").exists():
                    (h / "build").symlink_to(prod_paths.home / "build")
                t0 = time.perf_counter()
                subprocess.check_call(
                    [
                        sys.executable,
                        str(build_script),
                        "--home",
                        str(h),
                        "--seq-lens",
                        seq_csv,
                        "--ane-profile",
                        prof,
                    ]
                )
                compile_sec[prof] = round(time.perf_counter() - t0, 2)
                manifests[prof] = read_ane_manifest(h / "artifacts" / "ane_models")
                homes[prof] = h

            base_sent = (
                "Apple Silicon Neural Engine hybrid GatedDeltaNet + partial-RoPE contextual embedding "
                "evaluation under IEEE binary16 FTZ, Tanh-factored SiLU, and row-wise max_abs RMSNorm."
            )
            texts_128 = [
                f"Record {i:02d}: {base_sent} Clause {i * 3 + 1} governs indemnification and liability caps."
                for i in range(32)
            ]
            texts_256 = [
                f"Record {i:02d}: " + " ".join([base_sent] * 4) + f" Section {i:02d} specifies arbitration venue."
                for i in range(32)
            ]

            ref_home = homes[profiles[0]]
            eng_gpu = FastPplx(home=ref_home, model_dir=args.model_dir, mode="gpu", local_files_only=True)
            try:
                ref_128_f32 = eng_gpu.encode_texts(texts_128[:16], mode="gpu", output_dtype="float32", normalize=True)
                ref_256_f32 = eng_gpu.encode_texts(texts_256[:16], mode="gpu", output_dtype="float32", normalize=True)
                ref_256_d128 = eng_gpu.encode_texts(
                    texts_256[:16], mode="gpu", dimensions=128, output_dtype="float32", normalize=True
                )
                ref_256_i8 = eng_gpu.encode_texts(texts_256[:16], mode="gpu", output_dtype="int8", normalize=False)
            finally:
                eng_gpu.close()

            engines = {
                prof: FastPplx(
                    home=homes[prof], model_dir=args.model_dir, mode="ane", ane_profile=prof, local_files_only=True
                )
                for prof in profiles
            }
            parity: dict[str, dict[str, Any]] = {}
            perf: dict[str, Any] = {}
            try:
                for prof, eng in engines.items():
                    if not eng.ane_available:
                        raise RuntimeError(f"ANE unavailable for profile={prof}: {eng._ane_init_error}")
                    v128 = eng.encode_texts(texts_128[:16], mode="ane", output_dtype="float32", normalize=True)
                    v256 = eng.encode_texts(texts_256[:16], mode="ane", output_dtype="float32", normalize=True)
                    vd128 = eng.encode_texts(
                        texts_256[:16], mode="ane", dimensions=128, output_dtype="float32", normalize=True
                    )
                    vi8 = eng.encode_texts(texts_256[:16], mode="ane", output_dtype="int8", normalize=False)
                    p128 = compute_vector_parity(v128, ref_128_f32)
                    p256 = compute_vector_parity(v256, ref_256_f32)
                    pd128 = compute_vector_parity(vd128, ref_256_d128)
                    pi8 = compute_int8_parity(vi8, ref_256_i8)
                    parity[prof] = {
                        "s128_f32_min_cos": p128["min_cos"],
                        "s128_f32_mean_cos": p128["mean_cos"],
                        "s256_f32_min_cos": p256["min_cos"],
                        "s256_f32_mean_cos": p256["mean_cos"],
                        "s256_d128_min_cos": pd128["min_cos"],
                        "s256_int8_min_cos": pi8["min_cos"],
                        "s256_int8_mae": pi8["mean_abs_err"],
                    }

                for prof in profiles:
                    _ = engines[prof].encode_texts(texts_128[:4], mode="ane")
                    _ = engines[prof].encode_texts(texts_256[:4], mode="ane")

                base_prof = "fast" if "fast" in profiles else profiles[0]
                for label, corpus in (("S128_32docs", texts_128), ("S256_32docs", texts_256)):
                    tot_toks = sum(len(engines[base_prof].tokenizer.encode(t)) + 2 for t in corpus)
                    times: dict[str, list[float]] = {p: [] for p in profiles}
                    for r in range(max(1, int(args.repeats))):
                        order = profiles if r % 2 == 0 else list(reversed(profiles))
                        for p in order:
                            t0 = time.perf_counter()
                            _ = engines[p].encode_texts(corpus, mode="ane", output_dtype="float32", normalize=True)
                            times[p].append(time.perf_counter() - t0)
                    med_base = statistics.median(times[base_prof])
                    perf[label] = {
                        "total_tokens": tot_toks,
                        "profiles": {
                            p: {
                                "median_ms": round(statistics.median(times[p]) * 1000.0, 2),
                                "tok_s": round(tot_toks / max(statistics.median(times[p]), 1e-9), 1),
                                "docs_s": round(len(corpus) / max(statistics.median(times[p]), 1e-9), 2),
                                "rel_vs_base": round(med_base / max(statistics.median(times[p]), 1e-9), 4),
                            }
                            for p in profiles
                        },
                    }
            finally:
                for eng in engines.values():
                    eng.close()

            stock_mlx: dict[str, Any] | None = None
            pub_py = args.torch_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
            pplx_dir = (
                Path(args.pplx_mlx_path).expanduser()
                if args.pplx_mlx_path
                else Path(os.environ.get("PPLX_CONTEXT_MLX_PATH", str(Path.home() / "Work" / "forks" / "pplx-context-mlx")))
            )
            if Path(pub_py).is_file() and pplx_dir.is_dir():
                mlx_snippet = r"""
import json, os, statistics, sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, sys.argv[2])
import pplx_context_mlx

hub_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
snaps_z = sorted((hub_root / "models--zainmerchan--pplx-embed-v2-context-0.6b-mlx" / "snapshots").glob("*"))
snaps = snaps_z or sorted((hub_root / "models--perplexity-ai--pplx-embed-v2-context-0.6b" / "snapshots").glob("*"))
if not snaps:
    print(json.dumps({"status": "skipped", "reason": "no pplx snapshot"}))
    raise SystemExit(0)
eng = pplx_context_mlx.load_model(str(snaps[-1]), dtype="float16")
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = {
    "status": "ok",
    "harness": "pplx-context-mlx (zainmerchan-f16)" if snaps_z else "pplx-context-mlx (f16)",
}
for lbl in ("S128_32docs", "S256_32docs"):
    txts = data[lbl]
    docs_nested = [[t] for t in txts]
    tot_toks = sum(len(eng.tokenizer._tokenizer.encode(t, add_special_tokens=False)) + 2 for t in txts)
    _ = eng.encode(docs_nested[:4], batch_size=32, normalize_embeddings=True)
    mx.synchronize()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = eng.encode(docs_nested, batch_size=32, normalize_embeddings=True)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    out[lbl] = {
        "median_ms": round(med * 1000.0, 2),
        "tok_s": round(tot_toks / max(med, 1e-9), 1),
        "docs_s": round(len(txts) / max(med, 1e-9), 2),
    }
print(json.dumps(out))
"""
                corp_f = tmp_root / "pplx_corp.json"
                corp_f.write_text(json.dumps({"S128_32docs": texts_128, "S256_32docs": texts_256}), encoding="utf-8")
                proc_p = subprocess.run(
                    [str(pub_py), "-c", mlx_snippet, str(corp_f), str(pplx_dir)],
                    capture_output=True,
                    text=True,
                    timeout=240,
                    check=False,
                )
                if proc_p.returncode == 0 and proc_p.stdout.strip():
                    stock_mlx = json.loads(proc_p.stdout.strip().splitlines()[-1])

    md = format_ane_profiles_markdown(compile_sec, parity, perf, stock_mlx)
    return {
        "suite": "ane-profiles",
        "telemetry": telemetry,
        "profiles": profiles,
        "seq_lens": seq_lens,
        "compile_sec": compile_sec,
        "manifests": manifests,
        "parity_vs_gpu": parity,
        "throughput": perf,
        "stock_mlx_baseline": stock_mlx,
        "footprint": _footprint_snapshot(),
        "markdown": md,
    }


def cmd_all(args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "suite": "all",
        "modes": _parse_str_csv(args.modes),
        "parity_modes": _parse_str_csv(getattr(args, "parity_modes", "gpu,gpu-bf16,ane,hetero2")),
        "doc_lengths": _parse_int_csv(args.doc_lengths),
        "context_limits": _parse_int_csv(args.lengths),
        "context_budgets": _parse_int_csv(args.context_budgets),
        "telemetry": _collect_telemetry(),
        "quiescence_warnings": _sys_quiescence(),
        "ane_status": _check_ane_ready(args.home),
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    parity_args = argparse.Namespace(**vars(args))
    parity_args.modes = getattr(args, "parity_modes", "gpu,gpu-bf16,ane,hetero2")
    return {
        "suite": "all",
        "telemetry": plan["telemetry"],
        "speed": cmd_speed(args),
        "parity": cmd_parity(parity_args),
        "context_limit": cmd_context_limit(args),
        "context_windows": cmd_context_windows(args),
    }


def _add_shared_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--home", default=None, help="Override FAST_PPLX_HOME")
    p.add_argument("--model-dir", default=None, help="Override HF/staged model directory")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument(
        "--state-dtype",
        choices=("float32", "bfloat16", "int8"),
        default=None,
        help="Optional GDN recurrent state format override (sets FAST_PPLX_STATE_DTYPE)",
    )
    p.add_argument("--dimensions", type=int, default=None, help="Optional Matryoshka output width")
    p.add_argument("--no-mapped-embedding", action="store_true", help="Put embed_tokens on GPU instead of mmap")
    p.add_argument("--enable-tf32", action="store_true", help="Opt into MLX_ENABLE_TF32=1 (default: 0 for IEEE FP32 exactness)")
    p.add_argument("--allow-contended", action="store_true", help="Proceed even if quiescence check sees contenders")
    p.add_argument("--dry-run", action="store_true", help="Print plan and asset readiness without GPU/ANE work")
    p.add_argument("--json-out", default=None, help="Write full JSON output to this path")
    p.add_argument("--json-only", action="store_true", help="Emit only JSON to stdout (used by isolated workers)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-pplx-bench",
        description="Apple Silicon benchmark, parity, 32K footprint, contextual window, and bit-identity harness for fast-pplx.",
    )
    sub = parser.add_subparsers(dest="suite", required=True)

    p_speed = sub.add_parser("speed", help="Measure latency, tok/s, Metal peak, and OS footprint across modes + baselines")
    _add_shared_flags(p_speed)
    p_speed.add_argument(
        "--modes",
        default="gpu-no-mmap,gpu-bf16,gpu,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu",
        help="Comma-separated modes & public baselines",
    )
    p_speed.add_argument("--timing-scope", choices=("e2e", "forward"), default="e2e", help="Symmetric primary timing scope (default: e2e)")
    p_speed.add_argument("--doc-lengths", default="256,512,2048,4096", help="Comma-separated doc lengths")
    p_speed.add_argument("--doc-batch", type=int, default=8)
    p_speed.add_argument("--warmup", type=int, default=1)
    p_speed.add_argument("--repeats", type=int, default=3)
    p_speed.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True)
    p_speed.add_argument("--abba", action=argparse.BooleanOptionalAction, default=True, help="Run interleaved ABBA pairing when --no-isolate")
    p_speed.add_argument("--abba-pair", default=None, help="Optional comma-separated pair (e.g. gpu,hetero2) for interleaved ABBA run")
    p_speed.add_argument("--torch-python", default=None, help="Optional Python binary with torch/transformers installed")
    p_speed.add_argument("--pplx-mlx-path", default=None, help="Optional checkout path for public pplx-context-mlx")
    p_speed.add_argument("--torch-threads", default="1,2,4,6", help="Thread counts swept for torch-cpu baseline")

    p_parity = sub.add_parser("parity", help="Verify all 16 reference cases + S=4096 streamed parity across modes")
    _add_shared_flags(p_parity)
    p_parity.add_argument("--modes", default="gpu,gpu-bf16,ane,hetero2")
    p_parity.add_argument("--include-s4096", action=argparse.BooleanOptionalAction, default=True)

    p_clim = sub.add_parser("context-limit", help="Sweep 512..32K sequence lengths in isolated subprocesses")
    _add_shared_flags(p_clim)
    p_clim.add_argument("--mode", choices=("gpu", "hetero2"), default="gpu")
    p_clim.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    p_clim.add_argument("--single-length", type=int, default=None, help=argparse.SUPPRESS)
    p_clim.add_argument("--timeout-sec", type=int, default=180)

    p_cwin = sub.add_parser("context-windows", help="Probe MRR/nDCG@5 & tok/s across contextual window budgets vs isolated")
    _add_shared_flags(p_cwin)
    p_cwin.add_argument("--mode", choices=("auto", "gpu", "hetero", "hetero2"), default="gpu")
    p_cwin.add_argument("--context-budgets", default="512,1024,2048,4096")
    p_cwin.add_argument("--filler-repeats", type=int, default=8)
    p_cwin.add_argument("--repeat-header-tokens", type=int, default=128)
    p_cwin.add_argument("--window-overlap-chunks", type=int, default=0)

    p_bit = sub.add_parser("bit-identity", help="Dump or compare exact float32 projected arrays across commits")
    _add_shared_flags(p_bit)
    p_bit.add_argument("action", choices=("dump", "compare"))
    p_bit.add_argument("--mode", choices=("gpu", "ane", "hetero", "hetero2"), default="gpu")
    p_bit.add_argument("--out-dir", default="artifacts/bit_identity")
    p_bit.add_argument("--baseline-dir", default=None)
    p_bit.add_argument("--candidate-dir", default=None)

    p_aprof = sub.add_parser(
        "ane-profiles",
        help="Compile and ABBA-benchmark fast, balanced, and parity ANE profiles in isolated temp homes",
    )
    _add_shared_flags(p_aprof)
    p_aprof.add_argument("--profiles", default="fast,balanced,parity")
    p_aprof.add_argument("--seq-lens", default="128,256")
    p_aprof.add_argument("--repeats", type=int, default=3)
    p_aprof.add_argument("--torch-python", default=None, help="Optional Python binary with stock pplx-context-mlx")
    p_aprof.add_argument("--pplx-mlx-path", default=None, help="Optional checkout path for public pplx-context-mlx")

    p_all = sub.add_parser("all", help="Run speed + parity + context-limit + context-windows suites")
    _add_shared_flags(p_all)
    p_all.add_argument("--modes", default="gpu-no-mmap,gpu-bf16,gpu,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu")
    p_all.add_argument("--parity-modes", default="gpu,gpu-bf16,ane,hetero2")
    p_all.add_argument("--timing-scope", choices=("e2e", "forward"), default="e2e")
    p_all.add_argument("--mode", choices=("gpu", "hetero2"), default="gpu")
    p_all.add_argument("--doc-lengths", default="256,512,2048,4096")
    p_all.add_argument("--doc-batch", type=int, default=8)
    p_all.add_argument("--warmup", type=int, default=1)
    p_all.add_argument("--repeats", type=int, default=3)
    p_all.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True)
    p_all.add_argument("--abba", action=argparse.BooleanOptionalAction, default=True)
    p_all.add_argument("--abba-pair", default=None)
    p_all.add_argument("--include-s4096", action=argparse.BooleanOptionalAction, default=True)
    p_all.add_argument("--torch-python", default=None)
    p_all.add_argument("--pplx-mlx-path", default=None)
    p_all.add_argument("--torch-threads", default="1,2,4,6")
    p_all.add_argument("--lengths", default="512,2048,8192,16384,32768")
    p_all.add_argument("--single-length", type=int, default=None, help=argparse.SUPPRESS)
    p_all.add_argument("--timeout-sec", type=int, default=180)
    p_all.add_argument("--context-budgets", default="512,1024,2048,4096")
    p_all.add_argument("--filler-repeats", type=int, default=8)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "state_dtype", None):
        os.environ["FAST_PPLX_STATE_DTYPE"] = str(args.state_dtype)
    if getattr(args, "enable_tf32", False):
        os.environ["MLX_ENABLE_TF32"] = "1"
    else:
        os.environ.setdefault("MLX_ENABLE_TF32", "0")

    if args.suite == "bit-identity" and args.action == "compare":
        if not args.baseline_dir or not args.candidate_dir:
            parser.error("`bit-identity compare` requires --baseline-dir and --candidate-dir")

    handlers = {
        "speed": cmd_speed,
        "parity": cmd_parity,
        "context-limit": cmd_context_limit,
        "context-windows": cmd_context_windows,
        "bit-identity": cmd_bit_identity,
        "ane-profiles": cmd_ane_profiles,
        "all": cmd_all,
    }
    res = handlers[args.suite](args)

    if getattr(args, "json_out", None):
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    if getattr(args, "json_only", False):
        print(json.dumps(res))
    elif "markdown" in res:
        print(res["markdown"])
        print()
        print(json.dumps({k: v for k, v in res.items() if k != "markdown"}, indent=2))
    else:
        print(json.dumps(res, indent=2))

    if args.suite == "bit-identity" and args.action == "compare":
        return 0 if res.get("all_bit_identical") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
