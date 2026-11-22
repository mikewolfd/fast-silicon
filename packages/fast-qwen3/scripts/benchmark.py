#!/usr/bin/env python3
"""Reproducible `ABBA` parity, throughput, speculative MTP, and 4-bucket memory benchmark for `fast-qwen3`."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np

from fast_qwen3 import (
    FastQwen3Engine,
    FastQwen3Model,
    FastQwen3Tokenizer,
    Qwen3MtpDrafter,
    resolve_paths,
)

LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")

BENCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "organization": {"type": "string"},
        "fiscal_year": {"type": "integer"},
        "category": {"type": "string", "enum": ["research", "infrastructure", "operations", "security"]},
        "budget_musd": {"type": "number"},
        "approved": {"type": "boolean"},
    },
    "required": ["organization", "fiscal_year", "category", "budget_musd", "approved"],
    "additionalProperties": False,
}

BENCH_SYSTEM_PROMPT = (
    "You are an exact structured extraction engine. Extract the organization name, fiscal year, "
    "category (one of research, infrastructure, operations, security), budget in millions USD, "
    "and boolean approval status from the record into strict JSON."
)

BENCH_PROMPTS: tuple[str, ...] = (
    "Aurora Photonics finalized its FY 2026 research program at 42.5 million USD; board status: approved.",
    "Meridian Rail Systems submitted a 118.0 million USD infrastructure plan for 2025; approval remains pending (false).",
    "Cascade Cloud Corp approved a 19.75 million USD security modernization budget for fiscal year 2026.",
    "Vanguard Logistics Group filed its 2025 operations budget of 64.2 million USD with full executive approval.",
    "Helios Semiconductor proposed an 88.4 million USD research expansion for 2027; board approval: false.",
    "Summit Grid Utilities cleared a 150.0 million USD infrastructure upgrade for fiscal year 2026 (approved).",
    "Kestrel Cyber Defense locked in 27.3 million USD for 2026 security operations; approved by audit board.",
    "Pacific Marine Freight logged a 53.8 million USD operations overhaul for 2025; approval denied (false).",
)


def _host_summary() -> dict[str, Any]:
    chip = "unknown"
    mem_gb = 0.0
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        mem_gb = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / (1024**3), 2)
    except Exception:
        pass
    dev = dict(mx.device_info()) if hasattr(mx, "device_info") else {}
    return {
        "chip": chip,
        "mem_gb": mem_gb,
        "macos": platform.mac_ver()[0],
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device_name": dev.get("device_name", "unknown"),
        "architecture": dev.get("architecture", "unknown"),
    }


def _memory_snapshot() -> dict[str, float]:
    out: dict[str, float] = {
        "mlx_active_gb": round(float(mx.get_active_memory()) / 1e9, 4) if hasattr(mx, "get_active_memory") else 0.0,
        "mlx_peak_gb": round(float(mx.get_peak_memory()) / 1e9, 4) if hasattr(mx, "get_peak_memory") else 0.0,
    }
    try:
        from fast_shared.bench import get_unified_memory_footprint

        fp = get_unified_memory_footprint()
        if isinstance(fp, dict):
            for k in ("phys_footprint_gb", "peak_footprint_gb"):
                if fp.get(k) is not None:
                    out[k] = round(float(fp[k]), 4)
    except Exception:
        pass
    return out


def run_model_suite(
    preset: str,
    *,
    rounds: int = 2,
    max_tokens: int = 64,
) -> dict[str, Any]:
    paths = resolve_paths()
    snap = paths.resolve_model_dir(preset, local_files_only=True)
    tok = FastQwen3Tokenizer(snap)

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    # 1. Measure 0-ULP fusion parity between SpecBucketedLinearGroup and unfused projections
    t_load0 = time.perf_counter()
    m_fused, mtp_w = FastQwen3Model.from_pretrained(snap, enable_fusion=True)
    load_sec = time.perf_counter() - t_load0

    m_unfused, _ = FastQwen3Model.from_pretrained(snap, enable_fusion=False)
    probe_ids = tok.encode_chat(BENCH_PROMPTS[0], system_prompt=BENCH_SYSTEM_PROMPT, enable_thinking=False)
    inp = mx.array([probe_ids], dtype=mx.int32)
    logits_f = m_fused(inp, m_fused.make_cache(kv_step=128))
    logits_u = m_unfused(inp, m_unfused.make_cache(kv_step=128))
    mx.eval(logits_f, logits_u)
    fusion_max_abs_diff = float(mx.max(mx.abs(logits_f.astype(mx.float32) - logits_u.astype(mx.float32))).item())
    top1_match = bool(int(mx.argmax(logits_f[0, -1]).item()) == int(mx.argmax(logits_u[0, -1]).item()))

    # 2. Build engines for interleaved ABBA (`unfused_no_lcp` vs `fast_qwen3`)
    mtp_drafter = Qwen3MtpDrafter.from_mtp_weights(m_fused, mtp_w)
    eng_fast = FastQwen3Engine(
        model=m_fused,
        tokenizer=tok,
        mtp_drafter=mtp_drafter,
        max_prefix_cache_entries=8,
        warmup=True,
    )
    eng_base = FastQwen3Engine(
        model=m_unfused,
        tokenizer=tok,
        mtp_drafter=None,
        max_prefix_cache_entries=0,
        warmup=True,
    )

    def _run_b1(eng: FastQwen3Engine, *, use_lcp: bool) -> tuple[float, int, int]:
        if not use_lcp:
            eng._prefix_cache.clear()
        t0 = time.perf_counter()
        comp = 0
        valid = 0
        for p in BENCH_PROMPTS[:4]:
            if not use_lcp:
                eng._prefix_cache.clear()
            it = eng.generate(
                p,
                system_prompt=BENCH_SYSTEM_PROMPT,
                json_schema=BENCH_SCHEMA if use_lcp else None,
                max_tokens=max_tokens,
                speculative=False,
            )
            comp += it.completion_tokens
            if it.valid_schema or it.valid_json:
                valid += 1
        return time.perf_counter() - t0, comp, valid

    def _run_b8(eng: FastQwen3Engine, *, fast_arm: bool) -> tuple[float, int, int]:
        if not fast_arm:
            eng._prefix_cache.clear()
        t0 = time.perf_counter()
        res = eng.generate_batch(
            BENCH_PROMPTS,
            system_prompt=BENCH_SYSTEM_PROMPT,
            json_schema=BENCH_SCHEMA if fast_arm else None,
            max_tokens=max_tokens,
            batch_size=8 if fast_arm else 1,
            speculative=False,
        )
        valid = sum(1 for it in res.items if (it.valid_schema if fast_arm else it.valid_json))
        comp = sum(it.completion_tokens for it in res.items)
        return time.perf_counter() - t0, comp, valid

    # Warm LCP cache on fast arm before steady-state ABBA timing
    _ = eng_fast.generate(
        BENCH_PROMPTS[0],
        system_prompt=BENCH_SYSTEM_PROMPT,
        json_schema=BENCH_SCHEMA,
        max_tokens=16,
        speculative=False,
    )

    b1_base_sec: list[float] = []
    b1_fast_sec: list[float] = []
    b1_base_toks: list[int] = []
    b1_fast_toks: list[int] = []
    b8_base_sec: list[float] = []
    b8_fast_sec: list[float] = []
    b8_base_valid = 0
    b8_fast_valid = 0
    b8_fast_toks = 0
    b8_base_toks = 0

    for _ in range(max(1, int(rounds))):
        # ABBA interleaved order for B=1 (4 items)
        sb1, tb1, _ = _run_b1(eng_base, use_lcp=False)
        sf1, tf1, _ = _run_b1(eng_fast, use_lcp=True)
        sf2, tf2, _ = _run_b1(eng_fast, use_lcp=True)
        sb2, tb2, _ = _run_b1(eng_base, use_lcp=False)
        b1_base_sec.extend([sb1, sb2])
        b1_fast_sec.extend([sf1, sf2])
        b1_base_toks.extend([tb1, tb2])
        b1_fast_toks.extend([tf1, tf2])

        # ABBA interleaved order for 8-doc extraction wave
        wb1, wt1, wv1 = _run_b8(eng_base, fast_arm=False)
        wf1, ft1, fv1 = _run_b8(eng_fast, fast_arm=True)
        wf2, ft2, fv2 = _run_b8(eng_fast, fast_arm=True)
        wb2, wt2, wv2 = _run_b8(eng_base, fast_arm=False)
        b8_base_sec.extend([wb1, wb2])
        b8_fast_sec.extend([wf1, wf2])
        b8_base_valid = max(b8_base_valid, wv1, wv2)
        b8_fast_valid = max(b8_fast_valid, fv1, fv2)
        b8_fast_toks = ft2
        b8_base_toks = wt2

    # Release baseline engine before MTP speculative check to keep peak UMA minimal
    eng_base.close()
    del eng_base, m_unfused
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    # Measure single-engine steady-state MLX peak memory on `eng_fast` across 8-doc batch + speculative decode
    _ = _run_b8(eng_fast, fast_arm=True)

    # 3. Speculative B=1 greedy T=0 verification (MTP drafter when present, else PromptLookup)
    spec_prompt = (
        "Summarize the following technical specification in three concise bullet points: "
        "Aurora Photonics FY 2026 Silicon Photonics Interconnect (400G-DR4, 3.2 pJ/bit, 5nm CMOS, approved)."
    )
    _ = eng_fast.generate(spec_prompt, max_tokens=16, speculative=True)
    _ = eng_fast.generate(spec_prompt, max_tokens=16, speculative=False)

    t_nospec0 = time.perf_counter()
    res_nospec = eng_fast.generate(spec_prompt, max_tokens=64, speculative=False)
    nospec_sec = time.perf_counter() - t_nospec0

    t_spec0 = time.perf_counter()
    res_spec = eng_fast.generate(spec_prompt, max_tokens=64, speculative=True)
    spec_sec = time.perf_counter() - t_spec0

    mem = _memory_snapshot()
    eng_fast.close()
    del eng_fast, m_fused
    gc.collect()
    mx.clear_cache()

    med_b1_base = float(np.median(b1_base_sec))
    med_b1_fast = float(np.median(b1_fast_sec))
    med_b8_base = float(np.median(b8_base_sec))
    med_b8_fast = float(np.median(b8_fast_sec))

    return {
        "preset": preset,
        "snapshot": snap.name,
        "load_sec": round(load_sec, 3),
        "has_native_mtp_drafter": mtp_drafter is not None,
        "fusion_parity": {
            "max_abs_logit_diff": round(fusion_max_abs_diff, 6),
            "top1_match": top1_match,
        },
        "abba_b1_4docs": {
            "unfused_unconstrained_median_sec": round(med_b1_base, 4),
            "fast_qwen3_strict_lcp_median_sec": round(med_b1_fast, 4),
            "wall_speedup_x": round(med_b1_base / max(med_b1_fast, 1e-9), 3),
            "fast_docs_per_sec": round(4.0 / max(med_b1_fast, 1e-9), 3),
            "fast_tok_s": round(float(np.median(b1_fast_toks)) / max(med_b1_fast, 1e-9), 2),
        },
        "abba_wave_8docs": {
            "seq_unconstrained_median_sec": round(med_b8_base, 4),
            "seq_valid_8": b8_base_valid,
            "seq_tok_s": round(b8_base_toks / max(med_b8_base, 1e-9), 2),
            "fast_b8_strict_lcp_median_sec": round(med_b8_fast, 4),
            "fast_valid_8": b8_fast_valid,
            "fast_docs_per_sec": round(8.0 / max(med_b8_fast, 1e-9), 3),
            "fast_tok_s": round(b8_fast_toks / max(med_b8_fast, 1e-9), 2),
            "wall_speedup_x": round(med_b8_base / max(med_b8_fast, 1e-9), 3),
        },
        "speculative_b1_64tok": {
            "drafter": "Qwen3MtpDrafter" if mtp_drafter is not None else "PromptLookupDrafter",
            "exact_greedy_token_match": res_spec.token_ids == res_nospec.token_ids,
            "nospec_sec": round(nospec_sec, 4),
            "nospec_tok_s": round(res_nospec.completion_tokens / max(nospec_sec, 1e-9), 2),
            "spec_sec": round(spec_sec, 4),
            "spec_tok_s": round(res_spec.completion_tokens / max(spec_sec, 1e-9), 2),
            "spec_speedup_x": round(nospec_sec / max(spec_sec, 1e-9), 3),
        },
        "memory": mem,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run locked ABBA parity & throughput benchmarks for fast-qwen3")
    parser.add_argument("--presets", default="4b,0.8b-bf16", help="Comma-separated presets to benchmark")
    parser.add_argument("--rounds", type=int, default=2, help="Interleaved ABBA rounds per preset")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "presets": presets, "rounds": args.rounds}, indent=2))
        return 0

    mx.set_cache_limit(512 << 20)
    LOCK_PATH.touch(exist_ok=True)
    with open(LOCK_PATH, "r+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        results: list[dict[str, Any]] = []
        for preset in presets:
            try:
                results.append(run_model_suite(preset, rounds=args.rounds, max_tokens=args.max_tokens))
            except FileNotFoundError as exc:
                results.append({"preset": preset, "status": "skipped", "reason": str(exc)})

    payload = {
        "host": _host_summary(),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
