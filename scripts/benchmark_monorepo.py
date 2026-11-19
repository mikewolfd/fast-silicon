#!/usr/bin/env python3
"""Monorepo Apple Silicon M4 Pro benchmark & memory footprint investigation runner.

Runs isolated per-package subprocess benchmarks across `fast-silicon` packages,
capturing true macOS unified-memory physical footprint (`RUSAGE_INFO_V4`), Metal
active peak (`mx.get_peak_memory`), throughput/latency metrics, and optional parity
checks — serialized cleanly under `/tmp/m4_gpu_validation.lock`.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"

# Child wrapper run inside an isolated fresh Python interpreter per package step.
# Measures both MLX active peak and macOS RUSAGE_INFO_V4 OS physical footprint right
# before process exit so parent sees exact hardware allocation per package.
_CHILD_HARNESS = r"""
import json, os, runpy, sys, time
from fast_shared.bench import get_unified_memory_footprint

mode = sys.argv[1]
target = sys.argv[2]
extra_args = json.loads(sys.argv[3])
sys.argv = [target, *extra_args]

t0 = time.perf_counter()
rc = 0
exc_msg = None
try:
    if mode == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(target, run_name="__main__")
except SystemExit as exc:
    code = exc.code
    if isinstance(code, int):
        rc = code
    elif code is not None:
        rc = 1
        exc_msg = str(code)
except Exception as exc:
    rc = 1
    exc_msg = f"{type(exc).__name__}: {exc}"

elapsed_s = time.perf_counter() - t0
fp = get_unified_memory_footprint()
try:
    import mlx.core as mx
    mlx_peak_gb = round(float(mx.get_peak_memory()) / 1e9, 4)
except Exception:
    mlx_peak_gb = 0.0

summary = {
    "rc": int(rc),
    "elapsed_s": round(elapsed_s, 3),
    "mlx_peak_gb": mlx_peak_gb,
    "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
    "phys_footprint_gb": round(float(fp["phys_footprint_gb"]), 4),
    "rss_gb": round(float(fp["rss_bytes"]) / 1e9, 4),
    "error": exc_msg,
}
print("__MONOREPO_METRICS__=" + json.dumps(summary), flush=True)
raise SystemExit(rc)
"""


@dataclass(frozen=True)
class PackageSpec:
    name: str
    role: str
    home_env_var: str
    runner_kind: str  # "module" or "path"
    runner_target: str
    bench_args: tuple[str, ...]
    verify_args: tuple[str, ...]
    baseline_runs: tuple[tuple[str, tuple[str, ...]], ...] = ()


DEFAULT_PACKAGE_SPECS: tuple[PackageSpec, ...] = (
    PackageSpec(
        name="fast-bge-small",
        role="33M BERT encoder (ANE + GPU)",
        home_env_var="FAST_BGE_SMALL_HOME",
        runner_kind="module",
        runner_target="fast_bge_small.cli",
        bench_args=("bench", "--mode", "all", "--n", "1000"),
        verify_args=("verify",),
    ),
    PackageSpec(
        name="fast-embeddinggemma-2",
        role="308M bidirectional encoder (ANE + GPU + SME)",
        home_env_var="FAST_EMBEDDINGGEMMA2_HOME",
        runner_kind="module",
        runner_target="fast_embeddinggemma_2.cli",
        bench_args=("bench", "--mode", "all", "--n", "2000"),
        verify_args=("verify",),
    ),
    PackageSpec(
        name="fast-pplx",
        role="0.6B contextual GatedDeltaNet + SDPA (GPU + ANE hetero2)",
        home_env_var="FAST_PPLX_HOME",
        runner_kind="path",
        runner_target=str(_REPO_ROOT / "packages" / "fast-pplx" / "scripts" / "benchmark.py"),
        bench_args=(
            "speed",
            "--modes",
            "gpu-no-mmap,gpu-bf16,gpu,ane,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu",
            "--isolate",
            "--allow-contended",
            "--json-only",
        ),
        verify_args=("parity", "--modes", "gpu,ane,hetero2", "--allow-contended", "--json-only"),
    ),
    PackageSpec(
        name="fast-gemma4",
        role="Gemma-4 VLM structured JSON (Early14 ANE + Metal GPU)",
        home_env_var="FAST_GEMMA4_HOME",
        runner_kind="module",
        runner_target="fast_gemma4.cli",
        bench_args=("bench", "--batch-gpu", "128", "--batch-spec", "32", "--waves", "2", "--ane-variant", "w8"),
        verify_args=("verify",),
        baseline_runs=(
            ("no_kv_i8", ("bench", "--batch-gpu", "128", "--batch-spec", "1", "--waves", "1", "--skip-ane", "--no-kv-i8")),
        ),
    ),
    PackageSpec(
        name="fast-gpt-oss",
        role="20B MXFP4 MoE (Metal 4 MPP + Hydragen v2)",
        home_env_var="FAST_GPT_OSS_HOME",
        runner_kind="module",
        runner_target="fast_gpt_oss.cli",
        bench_args=("bench", "--batches", "8,64,128", "--local-files-only"),
        verify_args=("verify", "--local-files-only"),
    ),
    PackageSpec(
        name="fast-decision-models",
        role="Typed decision scoring (ANE CoreML + 0.8B W8 MLX)",
        home_env_var="FAST_DECISION_MODELS_HOME",
        runner_kind="module",
        runner_target="fast_decision_models.cli",
        bench_args=("bench", "--backend", "all", "--n", "64", "--batch-docs", "8", "--kev-quant", "w8"),
        verify_args=("verify", "--backend", "all"),
        baseline_runs=(
            ("bf16_b1", ("bench", "--backend", "kev-0.8b", "--n", "24", "--batch-docs", "1", "--kev-quant", "bf16")),
            ("bf16_b8", ("bench", "--backend", "kev-0.8b", "--n", "64", "--batch-docs", "8", "--kev-quant", "bf16")),
        ),
    ),
)

_ARTIFACT_SUBDIRS = ("weights", "ane_models", "kev_08b", "laya_typed", "models", "tokenizer")


def _has_staged_artifacts(home_dir: Path) -> bool:
    art = home_dir / "artifacts"
    if not art.is_dir():
        return False
    return any((art / sub).exists() and any((art / sub).iterdir()) for sub in _ARTIFACT_SUBDIRS if (art / sub).is_dir())


def resolve_package_home(spec: PackageSpec, base_env: dict[str, str] | None = None) -> Path:
    """Resolve effective home directory for `spec`, preferring explicit env or staged artifacts."""
    env = os.environ if base_env is None else base_env
    explicit = env.get(spec.home_env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    mono_home = _REPO_ROOT / "packages" / spec.name
    sibling_home = _REPO_ROOT.parent / spec.name
    if not _has_staged_artifacts(mono_home) and _has_staged_artifacts(sibling_home):
        return sibling_home
    return mono_home

_SPEC_BY_NAME: dict[str, PackageSpec] = {spec.name: spec for spec in DEFAULT_PACKAGE_SPECS}


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


def select_specs(raw_packages: str) -> list[PackageSpec]:
    """Resolve CLI `--packages` filter (`all` or comma-separated package names)."""
    tok = raw_packages.strip()
    if not tok or tok == "all":
        return list(DEFAULT_PACKAGE_SPECS)
    selected: list[PackageSpec] = []
    for part in tok.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in _SPEC_BY_NAME:
            valid = ", ".join(_SPEC_BY_NAME)
            raise ValueError(f"Unknown package {name!r}; choose from: all, {valid}")
        selected.append(_SPEC_BY_NAME[name])
    return selected


def parse_child_metrics(stdout: str) -> tuple[dict[str, Any] | None, str]:
    """Split child harness telemetry JSON line from raw stdout."""
    marker = "__MONOREPO_METRICS__="
    metrics: dict[str, Any] | None = None
    cleaned: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(marker):
            metrics = json.loads(line[len(marker) :])
        else:
            cleaned.append(line)
    return metrics, "\n".join(cleaned)


def _try_parse_json_tail(clean_stdout: str) -> dict[str, Any] | None:
    stripped = clean_stdout.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    idx = stripped.rfind("{")
    if idx >= 0:
        try:
            obj = json.loads(stripped[idx:])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def summarize_throughput(pkg_name: str, clean_stdout: str) -> str:
    """Extract a concise human-readable throughput/latency highlight from package bench output."""
    parsed = _try_parse_json_tail(clean_stdout)
    if pkg_name == "fast-pplx" and parsed is not None:
        ok_rows = [r for r in parsed.get("rows", []) if r.get("status") == "ok"]
        if ok_rows:
            native_rows = [r for r in ok_rows if r.get("mode") in ("gpu", "hetero2", "ane")]
            peak_row = max(native_rows or ok_rows, key=lambda r: float(r.get("tok_s", 0.0)))
            parts = [f"{peak_row['label']}: {peak_row['tok_s']:,.1f} tok/s ({peak_row['latency_ms']:.2f} ms)"]
            q1_ane = next((r for r in ok_rows if r.get("workload") == "query_b1" and r.get("mode") == "ane"), None)
            q1_het = next((r for r in ok_rows if r.get("workload") == "query_b1" and r.get("mode") == "hetero2"), None)
            if q1_ane:
                parts.append(f"B=1 ane {q1_ane['latency_ms']:.2f}ms")
            elif q1_het:
                parts.append(f"B=1 het2 {q1_het['latency_ms']:.2f}ms")
            return "; ".join(parts)

    if pkg_name == "fast-gemma4":
        pat_g = re.search(r"\[GPUOnly[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s", clean_stdout)
        pat_s = re.search(r"\[PromptLookup[^\]]*\][^\n]*?\((\d+)\s+eff tok/s\)[^\n]*?([0-9\.]+)\s+docs/s", clean_stdout)
        pat_h = re.search(r"\[Concurrent ANE\+GPU[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s", clean_stdout)
        parts = []
        if pat_g:
            parts.append(f"GPU: {int(pat_g.group(1)):,} tok/s ({pat_g.group(2)} docs/s)")
        if pat_s:
            parts.append(f"Spec: {int(pat_s.group(1)):,} eff tok/s")
        if pat_h:
            parts.append(f"ANE+GPU: {int(pat_h.group(1)):,} tok/s ({pat_h.group(2)} docs/s)")
        if parts:
            return "; ".join(parts)

    if pkg_name == "fast-decision-models" and parsed is not None:
        parts = []
        for k, v in parsed.items():
            if isinstance(v, dict) and "q_per_sec" in v:
                parts.append(f"{k}: {v['q_per_sec']} q/s ({v.get('ms_per_question', 0)} ms/q)")
        if parts:
            return "; ".join(parts)

    if pkg_name == "fast-gpt-oss" and parsed is not None:
        parts = []
        for k, v in parsed.items():
            if isinstance(v, dict) and "decode_tok_s" in v:
                parts.append(f"{k}: {v['decode_tok_s']} dec tok/s ({v.get('docs_per_sec', 0)} docs/s)")
        if parts:
            return "; ".join(parts[-3:])

    patterns = (
        r"([0-9][0-9,\.]*\s*(?:tok/s|q/s|docs/s))",
        r"(throughput\s*=\s*[0-9,\.]+\s*\S+)",
        r"(e2e[^\n]{0,40}[0-9,\.]+\s*tok/s)",
    )
    matches: list[str] = []
    for pat in patterns:
        found = re.findall(pat, clean_stdout, flags=re.IGNORECASE)
        for m in found:
            s = m.strip()
            if s not in matches:
                matches.append(s)
            if len(matches) >= 3:
                return "; ".join(matches)
    if matches:
        return "; ".join(matches)
    for line in reversed(clean_stdout.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:90]
    return "completed"


def classify_failure(clean_stdout: str, stderr: str, error_msg: str | None) -> str:
    """Distinguish missing local weights/ANE artifacts from hard execution errors."""
    combined = "\n".join(part for part in (clean_stdout, stderr, error_msg or "") if part).lower()
    skip_tokens = (
        "filenotfounderror",
        "not found",
        "missing exported",
        "missing native",
        "not staged",
        "no local",
        "laya_dir",
        "skipped",
    )
    return "skipped" if any(tok in combined for tok in skip_tokens) else "failed"


def _run_isolated_step(
    python_exe: str,
    spec: PackageSpec,
    step_args: Sequence[str],
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    resolved_home = resolve_package_home(spec)
    cmd = [
        python_exe,
        "-c",
        _CHILD_HARNESS,
        spec.runner_kind,
        spec.runner_target,
        json.dumps(list(step_args)),
    ]
    env = dict(os.environ)
    env["M4_GPU_LOCK_HELD"] = "1"
    env.setdefault(spec.home_env_var, str(resolved_home))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "rc": 124,
            "status": "failed",
            "resolved_home": str(resolved_home),
            "elapsed_s": float(timeout_sec),
            "mlx_peak_gb": 0.0,
            "peak_footprint_gb": 0.0,
            "rss_gb": 0.0,
            "highlight": f"timed out after {timeout_sec}s",
            "error": "timeout",
        }

    metrics, clean_out = parse_child_metrics(proc.stdout)
    rc = proc.returncode if metrics is None else int(metrics["rc"])
    err = metrics.get("error") if metrics else None
    status = "ok" if rc == 0 else classify_failure(clean_out, proc.stderr, err)
    if rc == 0:
        highlight = summarize_throughput(spec.name, clean_out)
    else:
        reason = err or (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit={rc}")
        highlight = reason[:96]
    parsed = _try_parse_json_tail(clean_out) if rc == 0 else None
    mlx_peak_gb = metrics["mlx_peak_gb"] if metrics else 0.0
    peak_footprint_gb = metrics["peak_footprint_gb"] if metrics else 0.0
    phys_footprint_gb = metrics.get("phys_footprint_gb", 0.0) if metrics else 0.0
    rss_gb = metrics["rss_gb"] if metrics else 0.0
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list) and spec.name == "fast-pplx":
        fast_rows = [
            r
            for r in parsed["rows"]
            if r.get("status") == "ok" and r.get("mode") in ("gpu", "hetero2", "ane")
        ]
        if fast_rows:
            mlx_peak_gb = max(float(r.get("peak_active_gb", 0.0) or 0.0) for r in fast_rows)
            peak_footprint_gb = max(float(r.get("peak_footprint_gb", 0.0) or 0.0) for r in fast_rows)
            phys_footprint_gb = peak_footprint_gb
            rss_gb = max(float(r.get("rss_gb", 0.0) or 0.0) for r in fast_rows)
    return {
        "rc": rc,
        "status": status,
        "resolved_home": str(resolved_home),
        "elapsed_s": metrics["elapsed_s"] if metrics else 0.0,
        "mlx_peak_gb": mlx_peak_gb,
        "peak_footprint_gb": peak_footprint_gb,
        "phys_footprint_gb": phys_footprint_gb,
        "rss_gb": rss_gb,
        "highlight": highlight,
        "parsed_json": parsed,
        "error": err,
        "stdout_tail": "\n".join(clean_out.splitlines()[-12:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
        "stdout_full": clean_out if rc == 0 else "",
    }


def _extract_encoder_comparison(text: str, *, fast_mode: str, ane_q_tag: str, gpu_q_tag: str) -> str:
    q_ane = re.search(rf"Query\s+\[{re.escape(ane_q_tag)}\]:\s+median=([0-9\.]+)ms", text)
    q_gpu = re.search(rf"Query\s+\[{re.escape(gpu_q_tag)}\]:\s+median=([0-9\.]+)ms", text)
    c_gpu = re.search(r"Corpus[^\n]*\[mode=gpu\]:[^\n]*->\s*([0-9,]+)\s*tok/s", text)
    c_fast = re.search(rf"Corpus[^\n]*\[mode={re.escape(fast_mode)}\]:[^\n]*->\s*([0-9,]+)\s*tok/s", text)
    parts: list[str] = []
    if c_gpu and c_fast:
        g_val = float(c_gpu.group(1).replace(",", ""))
        f_val = float(c_fast.group(1).replace(",", ""))
        if g_val > 0:
            parts.append(f"{fast_mode} {f_val:,.0f} vs gpu {g_val:,.0f} tok/s ({f_val / g_val:.2f}x)")
    if q_ane and q_gpu:
        a_ms = float(q_ane.group(1))
        g_ms = float(q_gpu.group(1))
        if a_ms > 0:
            parts.append(f"B=1 ANE {a_ms:.2f}ms vs GPU {g_ms:.2f}ms ({g_ms / a_ms:.2f}x)")
    return "; ".join(parts) if parts else "-"


def extract_baseline_comparison(
    pkg_name: str,
    bench_res: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> str:
    """Compute human-readable fast-path vs baseline speedup & memory comparison."""
    if bench_res.get("status") != "ok":
        return "-"
    stdout_full = str(bench_res.get("stdout_full") or bench_res.get("stdout_tail") or "")
    parsed = bench_res.get("parsed_json") or {}

    if pkg_name == "fast-bge-small":
        return _extract_encoder_comparison(stdout_full, fast_mode="hetero", ane_q_tag="ANE_B1_S24", gpu_q_tag="GPU_B1_S24")

    if pkg_name == "fast-embeddinggemma-2":
        return _extract_encoder_comparison(stdout_full, fast_mode="hetero2", ane_q_tag="ANE_B1_S32", gpu_q_tag="GPU_B1_S64")

    if pkg_name == "fast-pplx":
        rows = parsed.get("rows", []) if isinstance(parsed, dict) else []
        by_wl_mode = {(r.get("workload"), r.get("mode")): r for r in rows if r.get("status") == "ok"}
        parts: list[str] = []
        # 1. Peak short/mixed hetero2 vs bf16 speedup
        for wl in ("mixed_short_long", "doc_s256", "doc_s512"):
            fast = by_wl_mode.get((wl, "hetero2")) or by_wl_mode.get((wl, "gpu"))
            bf16 = by_wl_mode.get((wl, "gpu-bf16")) if fast else None
            if fast and bf16:
                ratio = float(fast["tok_s"]) / max(float(bf16["tok_s"]), 1e-9)
                parts.append(f"{wl} {fast['mode']} vs bf16: {ratio:.2f}x")
                break
        # 2. Low-latency ANE B=1 query highlight
        q_ane = by_wl_mode.get(("query_b1", "ane"))
        q_gpu = by_wl_mode.get(("query_b1", "gpu"))
        if q_ane and q_gpu:
            parts.append(f"B=1 ane {q_ane['latency_ms']:.2f}ms vs gpu {q_gpu['latency_ms']:.2f}ms")
        # 3. 4K mmap unified-memory savings vs non-mmap
        g4k = by_wl_mode.get(("doc_s4096", "gpu")) or by_wl_mode.get(("doc_s2048", "gpu"))
        n4k = by_wl_mode.get(("doc_s4096", "gpu-no-mmap")) or by_wl_mode.get(("doc_s2048", "gpu-no-mmap"))
        if g4k and n4k:
            dfp = float(g4k.get("peak_footprint_gb", 0.0)) - float(n4k.get("peak_footprint_gb", 0.0))
            parts.append(f"mmap dFootprint={dfp:+.3f}GB")
        return "; ".join(parts) if parts else "-"

    if pkg_name == "fast-gemma4":
        pat_g = re.search(r"\[GPUOnly[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s[^\n]*?peak=([0-9\.]+)GB", stdout_full)
        pat_h = re.search(r"\[Concurrent ANE\+GPU[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s", stdout_full)
        pat_s = re.search(r"\[PromptLookup[^\]]*\][^\n]*?\((\d+)\s+eff tok/s\)[^\n]*?([0-9\.]+)\s+docs/s", stdout_full)
        parts = []
        if pat_g and pat_h:
            g_ds = float(pat_g.group(2))
            h_ds = float(pat_h.group(2))
            if g_ds > 0:
                if h_ds >= g_ds:
                    parts.append(f"Hetero vs GPUOnly: {h_ds / g_ds:.2f}x ({h_ds:.2f} vs {g_ds:.2f} docs/s)")
                else:
                    parts.append(f"ANE+GPU {h_ds:.2f} docs/s ({pat_h.group(1)} tok/s); GPUOnly {g_ds:.2f} docs/s")
        elif pat_g and pat_s:
            g_tok = float(pat_g.group(1))
            s_tok = float(pat_s.group(1))
            if g_tok > 0:
                parts.append(f"PromptLookup vs GPUOnly: {s_tok / g_tok:.2f}x")
        no_kvi8 = baselines.get("no_kv_i8", {})
        if pat_g and no_kvi8.get("status") == "ok":
            base_out = str(no_kvi8.get("stdout_full") or no_kvi8.get("stdout_tail") or "")
            pat_base = re.search(r"\[GPUOnly[^\]]*\][^\n]*?peak=([0-9\.]+)GB", base_out)
            if pat_base:
                d_mem = float(pat_g.group(3)) - float(pat_base.group(1))
                parts.append(f"kv_i8 mem={d_mem:+.2f}GB")
        return "; ".join(parts) if parts else "-"

    if pkg_name == "fast-gpt-oss" and isinstance(parsed, dict):
        b1 = parsed.get("b1_seq")
        batch_keys = [k for k in parsed if k.startswith("batched_B")]
        if isinstance(b1, dict) and batch_keys:
            top_k = max(batch_keys, key=lambda k: int(k.removeprefix("batched_B")))
            top = parsed[top_k]
            dec_ratio = float(top["decode_tok_s"]) / max(float(b1["decode_tok_s"]), 1e-9)
            doc_ratio = float(top["docs_per_sec"]) / max(float(b1["docs_per_sec"]), 1e-9)
            return f"{top_k} vs B=1: {dec_ratio:.2f}x tok/s ({doc_ratio:.2f}x docs/s)"
        return "-"

    if pkg_name == "fast-decision-models" and isinstance(parsed, dict):
        fast_kev = parsed.get("kev_08b")
        if not isinstance(fast_kev, dict):
            return "-"
        fq = float(fast_kev["q_per_sec"])
        parts = []
        for label, tag in (("bf16_b8", "bf16 B=8"), ("bf16_b1", "bf16 B=1")):
            bres = baselines.get(label, {})
            bparsed = bres.get("parsed_json") or {}
            bkev = bparsed.get("kev_08b") if isinstance(bparsed, dict) else None
            if isinstance(bkev, dict) and float(bkev.get("q_per_sec", 0.0)) > 0:
                bq = float(bkev["q_per_sec"])
                parts.append(f"w8 B=8 vs {tag}: {fq / bq:.2f}x")
                dfp = float(bench_res.get("peak_footprint_gb", 0.0)) - float(bres.get("peak_footprint_gb", 0.0))
                if label == "bf16_b8":
                    parts.append(f"dFootprint={dfp:+.3f}GB")
        return "; ".join(parts) if parts else "-"

    return "-"


def extract_public_comparison(
    pkg_name: str,
    bench_res: dict[str, Any],
    public_res: dict[str, Any] | None,
) -> str:
    """Compare fast-silicon primary path against the next-fastest public Apple Silicon runtime."""
    if bench_res.get("status") != "ok":
        return "-"
    stdout_full = str(bench_res.get("stdout_full") or bench_res.get("stdout_tail") or "")
    parsed = bench_res.get("parsed_json") or {}

    if pkg_name == "fast-pplx":
        rows = parsed.get("rows", []) if isinstance(parsed, dict) else []
        by_wl_mode = {(r.get("workload"), r.get("mode")): r for r in rows if r.get("status") == "ok"}
        for wl in ("doc_s256", "mixed_short_long", "doc_s2048", "doc_s512", "query_b32"):
            fast = by_wl_mode.get((wl, "hetero2")) or by_wl_mode.get((wl, "gpu"))
            if fast is None:
                continue
            parts: list[str] = []
            for pub_mode, pub_lbl in (("pplx-context-mlx-f16", "pplx-ctx-mlx"), ("torch-mps", "torch-mps"), ("torch-cpu", "torch-cpu")):
                pub = by_wl_mode.get((wl, pub_mode))
                if pub:
                    ratio = float(fast["tok_s"]) / max(float(pub["tok_s"]), 1e-9)
                    parts.append(f"vs {pub_lbl}: {ratio:.2f}x")
            if parts:
                g4k = by_wl_mode.get(("doc_s4096", "gpu"))
                m4k = by_wl_mode.get(("doc_s4096", "torch-mps"))
                ram_note = (
                    f"; S=4K {g4k['peak_footprint_gb']:.1f} vs {m4k['peak_footprint_gb']:.1f}GB"
                    if (g4k and m4k and g4k.get("peak_footprint_gb") and m4k.get("peak_footprint_gb"))
                    else ""
                )
                return f"{wl} ({'; '.join(parts)}{ram_note})"
        return "skipped (pass --pplx-mlx-path / --public-python)"

    if not public_res or public_res.get("status") != "ok":
        reason = (public_res or {}).get("reason") or (public_res or {}).get("status", "skipped")
        return f"skipped ({str(reason)[:48]})"

    rt = str(public_res.get("runtime", "public"))
    if pkg_name in ("fast-bge-small", "fast-embeddinggemma-2"):
        fast_mode = "hetero" if pkg_name == "fast-bge-small" else "hetero2"
        m_fast = re.search(rf"Corpus[^\n]*\[mode={fast_mode}\]:[^\n]*->\s*([0-9,]+)\s*tok/s", stdout_full)
        pub_tps = float(public_res.get("tok_s", 0.0))
        if m_fast and pub_tps > 0:
            f_tps = float(m_fast.group(1).replace(",", ""))
            return f"vs {rt}: {f_tps / pub_tps:.2f}x ({f_tps:,.0f} vs {pub_tps:,.0f} tok/s)"
        return str(public_res.get("highlight", rt))

    if pkg_name == "fast-gemma4":
        matches: list[tuple[float, float, float | None]] = []
        for pat in (
            re.finditer(r"\[GPUOnly[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s(?:[^\n]*?peak=([0-9\.]+)GB)?", stdout_full),
            re.finditer(r"\[Concurrent ANE\+GPU[^\]]*\][^\n]*?\((\d+)\s+tok/s\)[^\n]*?([0-9\.]+)\s+docs/s(?:[^\n]*?peak=([0-9\.]+)GB)?", stdout_full),
        ):
            for m in pat:
                matches.append((float(m.group(2)), float(m.group(1)), float(m.group(3)) if m.group(3) else None))
        pub_ds = float(public_res.get("docs_per_sec", 0.0))
        if matches and pub_ds > 0:
            f_ds, f_tok, f_peak = max(matches, key=lambda x: x[0])
            pub_tok = float(public_res.get("decode_tok_s", 0.0))
            pub_peak = float(public_res.get("peak_active_gb", 0.0))
            extra = []
            if pub_tok > 0:
                extra.append(f"{f_tok / pub_tok:.2f}x tok/s")
            if f_peak is not None and pub_peak > 0:
                extra.append(f"dPeak={f_peak - pub_peak:+.2f}GB")
            tail = ("; " + ", ".join(extra)) if extra else ""
            return f"vs {rt}: {f_ds / pub_ds:.2f}x ({f_ds:.2f} vs {pub_ds:.2f} docs/s{tail})"
        return str(public_res.get("highlight", rt))

    if pkg_name == "fast-gpt-oss" and isinstance(parsed, dict):
        batch_keys = [k for k in parsed if k.startswith("batched_B")]
        if batch_keys:
            top = parsed[max(batch_keys, key=lambda k: int(k.removeprefix("batched_B")))]
            pub_tps = float(public_res.get("decode_tok_s", 0.0))
            if pub_tps > 0:
                f_tps = float(top["decode_tok_s"])
                return f"vs {rt}: {f_tps / pub_tps:.2f}x ({f_tps:.1f} vs {pub_tps:.1f} tok/s)"
        return str(public_res.get("highlight", rt))

    if pkg_name == "fast-decision-models" and isinstance(parsed, dict):
        fast_kev = parsed.get("kev_08b")
        fast_laya = parsed.get("laya_ane")
        pub_qs = float(public_res.get("q_per_sec", 0.0))
        if isinstance(fast_kev, dict) and pub_qs > 0:
            fq = float(fast_kev["q_per_sec"])
            dfp = float(bench_res.get("peak_footprint_gb", 0.0)) - float(public_res.get("peak_footprint_gb", 0.0))
            laya_note = ""
            if isinstance(fast_laya, dict) and float(fast_laya.get("q_per_sec", 0.0)) > 0:
                lq = float(fast_laya["q_per_sec"])
                laya_note = f"; laya_ane {lq / pub_qs:.2f}x"
            return f"vs {rt}: {fq / pub_qs:.2f}x ({fq:.1f} vs {pub_qs:.1f} q/s, dFP={dfp:+.2f}GB{laya_note})"
        return str(public_res.get("highlight", rt))

    return str(public_res.get("highlight", rt))


def format_monorepo_markdown(results: Sequence[dict[str, Any]]) -> str:
    """Render concise `pplx-context-mlx`-style cross-package comparison table."""
    lines = [
        "| Package | Throughput / Highlight | Internal Baseline | Public Apples-to-Apples | Metal Peak (GB) | OS Footprint (GB) | Verify | Status |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for item in results:
        b = item.get("bench", {})
        v = item.get("verify")
        v_txt = v.get("status", "-") if isinstance(v, dict) else "-"
        metal_gb = f"{b.get('mlx_peak_gb', 0.0):.3f}" if b.get("status") == "ok" else "-"
        os_gb = f"{b.get('peak_footprint_gb', 0.0):.3f}" if b.get("status") == "ok" else "-"
        comp = item.get("baseline_comparison") or extract_baseline_comparison(
            item["package"], b, item.get("baselines", {})
        )
        pub_comp = item.get("public_comparison") or extract_public_comparison(
            item["package"], b, item.get("public_baseline")
        )
        lines.append(
            f"| `{item['package']}` | {b.get('highlight', '-')} | {comp} | {pub_comp} | "
            f"{metal_gb} | {os_gb} | {v_txt} | {b.get('status', 'unknown')} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-monorepo",
        description="Run serialized M4 Pro benchmarks & OS unified-memory footprint checks across fast-silicon packages.",
    )
    parser.add_argument(
        "--packages",
        default="all",
        help="Comma-separated package list or 'all' (default: all)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to invoke in isolated child workers (default: current interpreter)",
    )
    parser.add_argument(
        "--public-python",
        "--torch-python",
        dest="public_python",
        default=None,
        help="Python binary with public runtimes (`torch`, `transformers`, `mlx_vlm`, `mlx_lm`, `mlx_embeddings`)",
    )
    parser.add_argument(
        "--pplx-mlx-path",
        default=None,
        help="Optional repo checkout path for public `pplx-context-mlx` comparison",
    )
    parser.add_argument("--include-verify", action="store_true", help="Also run each package's verify command")
    parser.add_argument("--skip-extra-baselines", action="store_true", help="Skip secondary internal baseline runs")
    parser.add_argument("--skip-public-baselines", action="store_true", help="Skip external public Apple Silicon baseline runs")
    parser.add_argument("--allow-contended", action="store_true", help="Run even if contention warnings appear")
    parser.add_argument("--dry-run", action="store_true", help="Print selected specs & quiescence status cleanly")
    parser.add_argument("--timeout-sec", type=int, default=300, help="Per-subprocess timeout seconds (default: 300)")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path")
    parser.add_argument("--markdown-out", default=None, help="Optional Markdown summary path")
    return parser


def _effective_bench_args(spec: PackageSpec, args: argparse.Namespace) -> list[str]:
    cmd_args = list(spec.bench_args)
    if spec.name == "fast-pplx":
        pub_py = getattr(args, "public_python", None) or os.environ.get("FAST_SILICON_PUBLIC_PYTHON")
        pplx_path = getattr(args, "pplx_mlx_path", None) or os.environ.get("PPLX_CONTEXT_MLX_PATH")
        if getattr(args, "skip_public_baselines", False) and "--modes" in cmd_args:
            modes_idx = cmd_args.index("--modes") + 1
            if modes_idx < len(cmd_args):
                cmd_args[modes_idx] = "gpu-no-mmap,gpu-bf16,gpu,ane,hetero2"
        if pub_py:
            cmd_args.extend(["--torch-python", str(pub_py)])
        if pplx_path:
            cmd_args.extend(["--pplx-mlx-path", str(pplx_path)])
    return cmd_args


def main(argv: Sequence[str] | None = None) -> int:
    from fast_shared.bench import check_system_quiescence

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from public_baselines import run_public_baseline

    parser = build_parser()
    args = parser.parse_args(argv)
    specs = select_specs(args.packages)
    quiescence = check_system_quiescence()

    if args.dry_run:
        plan = {
            "dry_run": True,
            "python": args.python,
            "public_python": args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON"),
            "pplx_mlx_path": args.pplx_mlx_path or os.environ.get("PPLX_CONTEXT_MLX_PATH"),
            "include_verify": bool(args.include_verify),
            "include_extra_baselines": not bool(args.skip_extra_baselines),
            "include_public_baselines": not bool(args.skip_public_baselines),
            "quiescence_warnings": quiescence,
            "packages": [
                {
                    "name": s.name,
                    "role": s.role,
                    "resolved_home": str(resolve_package_home(s)),
                    "bench_args": _effective_bench_args(s, args),
                    "baseline_runs": {}
                    if args.skip_extra_baselines
                    else {lbl: list(b_args) for lbl, b_args in s.baseline_runs},
                    "verify_args": list(s.verify_args) if args.include_verify else [],
                }
                for s in specs
            ],
        }
        print(json.dumps(plan, indent=2))
        return 0

    if quiescence and not args.allow_contended:
        print(f"QUIESCENCE WARNING ({len(quiescence)} active contenders):", file=sys.stderr)
        for line in quiescence:
            print(f"  {line}", file=sys.stderr)
        print("Aborting: re-run when quiet or pass --allow-contended.", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    total_specs = len(specs)
    with _hw_lock():
        for idx, spec in enumerate(specs, 1):
            print(f"[{idx}/{total_specs}] Benchmarking {spec.name}...", file=sys.stderr, flush=True)
            resolved_home = resolve_package_home(spec)
            bench_res = _run_isolated_step(
                args.python, spec, _effective_bench_args(spec, args), timeout_sec=args.timeout_sec
            )
            baselines: dict[str, dict[str, Any]] = {}
            if not args.skip_extra_baselines:
                for lbl, b_args in spec.baseline_runs:
                    print(
                        f"[{idx}/{total_specs}]   baseline '{lbl}' for {spec.name}...",
                        file=sys.stderr,
                        flush=True,
                    )
                    baselines[lbl] = _run_isolated_step(args.python, spec, b_args, timeout_sec=args.timeout_sec)
            public_res: dict[str, Any] | None = None
            if not args.skip_public_baselines and spec.name != "fast-pplx":
                print(
                    f"[{idx}/{total_specs}]   public baseline for {spec.name}...",
                    file=sys.stderr,
                    flush=True,
                )
                public_res = run_public_baseline(
                    pkg_name=spec.name,
                    resolved_home=resolved_home,
                    public_python=args.public_python,
                    timeout_sec=args.timeout_sec,
                )
            entry: dict[str, Any] = {
                "package": spec.name,
                "role": spec.role,
                "bench": bench_res,
                "baselines": baselines,
                "baseline_comparison": extract_baseline_comparison(spec.name, bench_res, baselines),
                "public_baseline": public_res,
                "public_comparison": extract_public_comparison(spec.name, bench_res, public_res),
            }
            if args.include_verify:
                print(
                    f"[{idx}/{total_specs}]   verify for {spec.name}...",
                    file=sys.stderr,
                    flush=True,
                )
                entry["verify"] = _run_isolated_step(
                    args.python, spec, spec.verify_args, timeout_sec=args.timeout_sec
                )
            results.append(entry)

    md = format_monorepo_markdown(results)
    payload = {
        "timestamp_epoch": int(time.time()),
        "quiescence_warnings": quiescence,
        "results": results,
    }
    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_m = Path(args.markdown_out)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md + "\n", encoding="utf-8")

    print(md)
    failures = [
        r for r in results if r["bench"]["status"] == "failed" or (r.get("verify", {}).get("status") == "failed")
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
