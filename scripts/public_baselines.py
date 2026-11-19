#!/usr/bin/env python3
"""Apples-to-apples public open-source Apple Silicon baseline runners for `fast-silicon`.

Runs next-fastest public MLX / PyTorch MPS implementations against identical
cached HuggingFace weights and matching batch/token workloads, returning
JSON metrics (`tok/s`, `docs/s`, `q/s`, `ms`, Metal/MPS peak, OS `RUSAGE_INFO_V4` footprint).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Child snippet executed cleanly in `--public-python` (or current Python).
_PUBLIC_RUNNER_SNIPPET = r"""
import ctypes, json, os, platform, resource, statistics, sys, time
from pathlib import Path
import numpy as np

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

def resolve_hf_snapshot(repo_id):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        rev = refs_main.read_text(encoding="utf-8").strip()
        cand = repo_dir / "snapshots" / rev
        if cand.is_dir():
            return str(cand)
    snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").is_dir() else []
    return str(snaps[-1]) if snaps else None

def median_t(fn, warmup=1, repeats=3, sync_fn=None):
    for _ in range(max(0, warmup)):
        _ = fn()
        if sync_fn:
            sync_fn()
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = fn()
        if sync_fn:
            sync_fn()
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))

cfg = json.loads(sys.argv[1])
pkg = cfg["package"]
resolved_home = Path(cfg["resolved_home"])

def skip(reason):
    print(json.dumps({"status": "skipped", "runtime": "unavailable", "highlight": "-", "reason": reason}), flush=True)
    raise SystemExit(0)

if pkg in ("fast-bge-small", "fast-embeddinggemma-2"):
    repo_id = "mlx-community/bge-small-en-v1.5-bf16" if pkg == "fast-bge-small" else "mlx-community/embeddinggemma-2-bf16"
    snap = resolve_hf_snapshot(repo_id) or resolve_hf_snapshot("BAAI/bge-small-en-v1.5" if pkg == "fast-bge-small" else "google/embeddinggemma-2")
    if snap is None:
        skip(f"no local HF snapshot for {repo_id}")
    n_docs = 1000 if pkg == "fast-bge-small" else 512
    b_corpus = 64 if pkg == "fast-bge-small" else 32
    q_seq = 24 if pkg == "fast-bge-small" else 32
    rng = np.random.default_rng(42)
    lens = np.clip(rng.integers(32, 465, size=n_docs), 16, 512)
    tot_toks = int(lens.sum())

    # Prefer public `mlx_embeddings` if installed; otherwise PyTorch MPS via `transformers`.
    try:
        import mlx.core as mx
        import mlx_embeddings
        t0 = time.perf_counter()
        model, tokenizer = mlx_embeddings.utils.load(snap)
        load_s = time.perf_counter() - t0
        q_txt = ["What drives oceanic heat transport?"]
        texts = [f"Record {i}. " + ("Ocean currents transfer heat across hemispheres. " * max(1, int(l) // 10)) for i, l in enumerate(lens)]
        mx.reset_peak_memory()
        b1_dt = median_t(lambda: mx.eval(mlx_embeddings.utils.generate(model, tokenizer, q_txt)), warmup=2, repeats=15)
        def _run_corpus():
            for i in range(0, len(texts), b_corpus):
                mx.eval(mlx_embeddings.utils.generate(model, tokenizer, texts[i : i + b_corpus]))
        c_dt = median_t(_run_corpus, warmup=1, repeats=2)
        fp_gb, rss_gb = footprint_gb()
        tok_s = tot_toks / max(c_dt, 1e-9)
        print(json.dumps({
            "status": "ok",
            "runtime": "mlx-embeddings",
            "b1_ms": round(b1_dt * 1000.0, 3),
            "tok_s": round(tok_s, 1),
            "docs_per_sec": round(n_docs / max(c_dt, 1e-9), 2),
            "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
            "peak_footprint_gb": fp_gb,
            "rss_gb": rss_gb,
            "load_s": round(load_s, 3),
            "highlight": f"mlx-embeddings: {tok_s:,.0f} tok/s (B=1 {b1_dt * 1000.0:.2f}ms)",
        }), flush=True)
        raise SystemExit(0)
    except ImportError:
        pass

    try:
        import torch
        from transformers import AutoModel
    except Exception as exc:
        skip(f"neither mlx_embeddings nor torch+transformers installed ({exc})")
    if not torch.backends.mps.is_available():
        skip("torch MPS unavailable")

    dt_torch = torch.float16 if pkg == "fast-bge-small" else torch.bfloat16
    t0 = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(snap, dtype=dt_torch, local_files_only=True).to("mps").eval()
    except Exception as exc:
        if pkg == "fast-embeddinggemma-2":
            skip("requires transformers>=5.18 (embedding_gemma2 unsupported)")
        skip(f"transformers AutoModel unavailable ({exc})")
    load_s = time.perf_counter() - t0
    vocab_lim = max(1000, int(getattr(model.config, "vocab_size", 30522)) - 8)
    with torch.inference_mode():
        q_ids = torch.randint(100, min(vocab_lim, 29000), (1, q_seq), device="mps", dtype=torch.long)
        q_mask = torch.ones_like(q_ids)
        b1_dt = median_t(lambda: model(input_ids=q_ids, attention_mask=q_mask), warmup=3, repeats=20, sync_fn=torch.mps.synchronize)
        # Bucketed padded batches over the same synthetic token lengths.
        batches = []
        sorted_lens = sorted((int(x) for x in lens), reverse=True)
        for i in range(0, len(sorted_lens), b_corpus):
            sub = sorted_lens[i : i + b_corpus]
            max_l = max(sub)
            ids = torch.full((len(sub), max_l), 1000, device="mps", dtype=torch.long)
            mask = torch.zeros((len(sub), max_l), device="mps", dtype=torch.long)
            for row_i, ln in enumerate(sub):
                mask[row_i, :ln] = 1
            batches.append((ids, mask))
        c_dt = median_t(lambda: [model(input_ids=bi, attention_mask=bm) for bi, bm in batches], warmup=1, repeats=2, sync_fn=torch.mps.synchronize)
    fp_gb, rss_gb = footprint_gb()
    tok_s = tot_toks / max(c_dt, 1e-9)
    print(json.dumps({
        "status": "ok",
        "runtime": f"transformers-mps-{str(dt_torch).split('.')[-1]}",
        "b1_ms": round(b1_dt * 1000.0, 3),
        "tok_s": round(tok_s, 1),
        "docs_per_sec": round(n_docs / max(c_dt, 1e-9), 2),
        "peak_active_gb": round(float(torch.mps.current_allocated_memory()) / 1e9, 4),
        "peak_footprint_gb": fp_gb,
        "rss_gb": rss_gb,
        "load_s": round(load_s, 3),
        "highlight": f"transformers-mps: {tok_s:,.0f} tok/s (B=1 {b1_dt * 1000.0:.2f}ms)",
    }), flush=True)

elif pkg == "fast-gemma4":
    try:
        import mlx.core as mx
        import mlx_vlm
    except Exception as exc:
        skip(f"mlx_vlm unavailable ({exc})")
    snap = resolve_hf_snapshot("lmstudio-community/gemma-4-E2B-it-MLX-4bit")
    if snap is None:
        skip("no local HF snapshot for lmstudio-community/gemma-4-E2B-it-MLX-4bit")

    fix_dir = resolved_home / "tests" / "fixtures"
    sys_p_path = fix_dir / "sys_extract_v1.txt"
    docs8_path = fix_dir / "sample_docs_8.json"
    if sys_p_path.is_file() and docs8_path.is_file():
        sys_p = sys_p_path.read_text(encoding="utf-8").strip()
        docs8 = json.loads(docs8_path.read_text(encoding="utf-8"))
        raw_prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs8]
    else:
        sys_p = (
            "Extract structured document metadata as compact JSON with status, text_quality, "
            "primary_kind, domain, title, organization, doc_date, year, summary, topics."
        )
        raw_prompts = [
            "Northwind Cloud reported quarterly revenue growth of 14 percent across enterprise accounts. " * 20
        ]

    t0 = time.perf_counter()
    model, processor = mlx_vlm.load(snap)
    load_s = time.perf_counter() - t0
    tok = getattr(processor, "tokenizer", processor)
    if hasattr(tok, "apply_chat_template"):
        fmts = [
            tok.apply_chat_template(
                [{"role": "system", "content": sys_p}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in raw_prompts
        ]
    else:
        fmts = [f"{sys_p}\n\n{p}" for p in raw_prompts]

    # Warmup
    _ = mlx_vlm.generate(model, processor, fmts[0], max_tokens=32, verbose=False)
    mx.reset_peak_memory()
    t1 = time.perf_counter()
    r_b1 = mlx_vlm.generate(model, processor, fmts[0], max_tokens=512, verbose=False)
    dt_b1 = time.perf_counter() - t1
    b1_out_toks = int(getattr(r_b1, "generation_tokens", 0) or max(1, len(tok.encode(getattr(r_b1, "text", str(r_b1))))))
    b1_e2e_tps = b1_out_toks / max(dt_b1, 1e-9)

    b_sz = 64
    batch_fmts = [fmts[i % len(fmts)] for i in range(b_sz)]
    mx.reset_peak_memory()
    t2 = time.perf_counter()
    r_batch = mlx_vlm.batch_generate(model, processor, prompts=batch_fmts, max_tokens=512, verbose=False)
    dt_batch = time.perf_counter() - t2
    batch_stats = getattr(r_batch, "stats", None)
    texts = list(getattr(r_batch, "texts", []) or [])
    tot_gen = int(getattr(batch_stats, "generation_tokens", 0) or 0)
    if tot_gen <= 0 and texts:
        tot_gen = sum(len(tok.encode(tx)) for tx in texts)
    valid_json = 0
    for tx in texts:
        s = tx.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            json.loads(s)
            valid_json += 1
        except Exception:
            pass
    batch_e2e_tps = tot_gen / max(dt_batch, 1e-9)
    docs_s = b_sz / max(dt_batch, 1e-9)
    dec_tps = float(getattr(batch_stats, "generation_tps", batch_e2e_tps) or batch_e2e_tps)
    peak_gb = round(float(mx.get_peak_memory()) / 1e9, 4)
    fp_gb, rss_gb = footprint_gb()
    print(json.dumps({
        "status": "ok",
        "runtime": "stock-mlx_vlm",
        "b1_e2e_out_tok_s": round(b1_e2e_tps, 1),
        "b1_docs_per_sec": round(1.0 / max(dt_b1, 1e-9), 2),
        "batch_size": b_sz,
        "e2e_out_tok_s": round(batch_e2e_tps, 1),
        "docs_per_sec": round(docs_s, 3),
        "decode_tok_s": round(dec_tps, 1),
        "valid_json_count": valid_json,
        "peak_active_gb": peak_gb,
        "peak_footprint_gb": fp_gb,
        "rss_gb": rss_gb,
        "load_s": round(load_s, 3),
        "highlight": (
            f"mlx_vlm B={b_sz}: {dec_tps:.1f} dec tok/s ({docs_s:.2f} docs/s, "
            f"json={valid_json}/{b_sz}, peak={peak_gb:.2f}GB; B=1 {b1_e2e_tps:.1f} tok/s)"
        ),
    }), flush=True)

elif pkg == "fast-gpt-oss":
    try:
        import mlx.core as mx
        import mlx_lm
    except Exception as exc:
        skip(f"mlx_lm unavailable ({exc})")
    snap = resolve_hf_snapshot("mlx-community/gpt-oss-20b-MXFP4-Q4")
    if snap is None:
        skip("no local HF snapshot for mlx-community/gpt-oss-20b-MXFP4-Q4")

    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(snap)
    load_s = time.perf_counter() - t0
    prompt = "Summarize sliding-window attention and grouped-query attention in JSON with two short keys."
    _ = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)
    mx.reset_peak_memory()
    t1 = time.perf_counter()
    text_out = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
    dt_1 = time.perf_counter() - t1
    out_toks = max(1, len(tokenizer.encode(text_out)))
    b1_tps = out_toks / max(dt_1, 1e-9)

    batch_tps = b1_tps
    docs_s = 1.0 / max(dt_1, 1e-9)
    b_label = "B=1"
    if hasattr(mlx_lm, "batch_generate"):
        p_ids = tokenizer.encode(prompt)
        mx.reset_peak_memory()
        t2 = time.perf_counter()
        bres = mlx_lm.batch_generate(model, tokenizer, prompts=[p_ids] * 8, max_tokens=128, verbose=False)
        dt_b = time.perf_counter() - t2
        bstats = getattr(bres, "stats", None)
        batch_tps = float(getattr(bstats, "generation_tps", 0.0) or ((8 * 128) / max(dt_b, 1e-9)))
        docs_s = 8.0 / max(dt_b, 1e-9)
        b_label = "B=8"
    fp_gb, rss_gb = footprint_gb()
    print(json.dumps({
        "status": "ok",
        "runtime": "stock-mlx_lm",
        "b1_tok_s": round(b1_tps, 1),
        "decode_tok_s": round(batch_tps, 1),
        "docs_per_sec": round(docs_s, 3),
        "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
        "peak_footprint_gb": fp_gb,
        "rss_gb": rss_gb,
        "load_s": round(load_s, 3),
        "highlight": f"mlx_lm {b_label}: {batch_tps:.1f} tok/s ({docs_s:.2f} docs/s; B=1 {b1_tps:.1f} tok/s)",
    }), flush=True)

elif pkg == "fast-decision-models":
    kev_dir = resolved_home / "artifacts" / "kev_08b"
    if not (kev_dir / "model.safetensors").is_file():
        skip(f"missing BF16 kev_08b weights at {kev_dir}")
    meta_path = kev_dir / "kev_meta.json"
    if not meta_path.is_file():
        skip(f"missing kev_meta.json at {kev_dir}")
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except Exception as exc:
        skip(f"torch+transformers Qwen3_5 unavailable ({exc})")
    if not torch.backends.mps.is_available():
        skip("torch MPS unavailable")

    t0 = time.perf_counter()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = Qwen3_5TextConfig(**meta["text_config"])
    sd = load_file(str(kev_dir / "model.safetensors"))
    clean_sd = {}
    for k, v in sd.items():
        if k.startswith("pointer_head."):
            continue
        if k.endswith("linear_attn.conv1d.weight") and v.ndim == 3 and v.shape[-1] == 1:
            v = v.transpose(1, 2).contiguous()
        clean_sd[k] = v
    full_model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16)
    full_model.load_state_dict(clean_sd, strict=False)
    model = full_model.model.to("mps").eval()
    del full_model, sd, clean_sd
    torch.mps.synchronize()
    load_s = time.perf_counter() - t0
    seq_len = 96
    with torch.inference_mode():
        x_b1 = torch.randint(10, 2000, (1, seq_len), device="mps", dtype=torch.long)
        m_b1 = torch.ones_like(x_b1)
        dt_b1 = median_t(
            lambda: model(input_ids=x_b1, attention_mask=m_b1, use_cache=False),
            warmup=1,
            repeats=4,
            sync_fn=torch.mps.synchronize,
        )
        x_b8 = torch.randint(10, 2000, (8, seq_len), device="mps", dtype=torch.long)
        m_b8 = torch.ones_like(x_b8)
        dt_b8 = median_t(
            lambda: model(input_ids=x_b8, attention_mask=m_b8, use_cache=False),
            warmup=1,
            repeats=4,
            sync_fn=torch.mps.synchronize,
        )
    b1_qs = 1.0 / max(dt_b1, 1e-9)
    b8_qs = 8.0 / max(dt_b8, 1e-9)
    fp_gb, rss_gb = footprint_gb()
    print(json.dumps({
        "status": "ok",
        "runtime": "transformers-mps-bf16",
        "b1_q_per_sec": round(b1_qs, 2),
        "q_per_sec": round(b8_qs, 2),
        "ms_per_question": round((dt_b8 * 1000.0) / 8.0, 2),
        "peak_active_gb": round(float(torch.mps.current_allocated_memory()) / 1e9, 4),
        "peak_footprint_gb": fp_gb,
        "rss_gb": rss_gb,
        "load_s": round(load_s, 3),
        "highlight": f"transformers-mps B=8: {b8_qs:.2f} q/s ({(dt_b8 * 1000.0) / 8.0:.2f} ms/q; B=1 {b1_qs:.2f} q/s)",
    }), flush=True)
else:
    skip(f"no public runner registered for {pkg}")
"""


def run_public_baseline(
    *,
    pkg_name: str,
    resolved_home: Path,
    public_python: str | None,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """Execute the public Apple Silicon baseline for `pkg_name` in an isolated subprocess."""
    if pkg_name == "fast-pplx":
        return {"status": "delegated", "reason": "reported inside fast-pplx speed suite (pplx-context-mlx-f16 / torch-mps / torch-cpu)"}

    py_exe = public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    payload = {"package": pkg_name, "resolved_home": str(resolved_home)}
    try:
        proc = subprocess.run(
            [py_exe, "-c", _PUBLIC_RUNNER_SNIPPET, json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "runtime": "timeout", "highlight": f"timed out after {timeout_sec}s"}
    except Exception as exc:
        return {"status": "skipped", "runtime": "unavailable", "reason": str(exc), "highlight": "-"}

    if not proc.stdout.strip():
        reason = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit={proc.returncode}"
        return {"status": "skipped", "runtime": "unavailable", "reason": reason[:120], "highlight": "-"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "status": "skipped",
            "runtime": "parse_error",
            "reason": proc.stdout.strip().splitlines()[-1][:120],
            "highlight": "-",
        }
