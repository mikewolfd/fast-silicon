#!/usr/bin/env python3
"""Apples-to-apples public open-source Apple Silicon baseline runners for `fast-silicon`.

Runs next-fastest public MLX / PyTorch MPS implementations against identical
cached HuggingFace weights and matching batch/token workloads using interleaved
`ABBA` timing, `fast_shared.bench.get_unified_memory_footprint`, explicit device
synchronization, wired-limit hygiene, and full JSON Schema validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FAST_SHARED_SRC = _REPO_ROOT / "packages" / "fast-shared" / "src"


def resolve_hf_snapshot(repo_id: str) -> Path | None:
    """Resolve a locally cached HuggingFace Hub snapshot directory without network access."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        rev = refs_main.read_text(encoding="utf-8").strip()
        cand = repo_dir / "snapshots" / rev
        if cand.is_dir():
            return cand
    snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def resolve_pplx_mlx_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve local checkout of `pplx-context-mlx` when available."""
    raw = explicit or os.environ.get("PPLX_CONTEXT_MLX_PATH")
    candidates = [Path(raw).expanduser()] if raw else [
        _REPO_ROOT.parent / "forks" / "pplx-context-mlx",
        Path.home() / "Work" / "forks" / "pplx-context-mlx",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def resolve_coreai_python(explicit: str | Path | None = None) -> Path | None:
    """Resolve Python interpreter with Apple `coreai.runtime` (`macOS 27.2+`) if installed."""
    raw = explicit or os.environ.get("FAST_SILICON_COREAI_PYTHON")
    candidates = [Path(raw).expanduser().absolute()] if raw else [
        Path.home() / "Work" / "coreai-bench" / "coreai-models" / ".venv" / "bin" / "python",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def resolve_coreai_bundle(subpath: str, explicit_root: str | Path | None = None) -> Path | None:
    """Resolve an exported Apple CoreAI `.aimodel` parent bundle directory if present."""
    raw = explicit_root or os.environ.get("FAST_SILICON_COREAI_EXPORTS")
    roots = [Path(raw).expanduser()] if raw else [Path.home() / "Work" / "coreai-bench" / "exports"]
    for root in roots:
        cand = root / subpath
        if cand.is_dir():
            return cand
    return None

# Child snippet executed cleanly in `--public-python` (or current Python).
_PUBLIC_RUNNER_SNIPPET = r"""
from contextlib import contextmanager
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np

cfg = json.loads(sys.argv[1])
fast_shared_src = cfg.get("fast_shared_src")
if fast_shared_src and fast_shared_src not in sys.path:
    sys.path.insert(0, fast_shared_src)

from fast_shared.bench import get_unified_memory_footprint, interleaved_abba

def footprint_gb():
    fp = get_unified_memory_footprint()
    return round(float(fp["peak_footprint_gb"]), 4), round(float(fp["rss_bytes"]) / 1e9, 4)

@contextmanager
def mlx_wired_scope(enabled=True):
    if not enabled:
        yield
        return
    prev = None
    mx_mod = None
    try:
        import mlx.core as mx
        mx_mod = mx
        if hasattr(mx, "set_wired_limit"):
            max_rec = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
            if max_rec > 0:
                prev = int(mx.set_wired_limit(max_rec))
    except Exception:
        mx_mod = None
    try:
        yield
    finally:
        if mx_mod is not None:
            try:
                if prev is not None and hasattr(mx_mod, "set_wired_limit"):
                    mx_mod.set_wired_limit(prev)
                if hasattr(mx_mod, "clear_cache"):
                    mx_mod.clear_cache()
            except Exception:
                pass

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

def run_abba_pair(fn_a, fn_b, *, rounds=2, cooldown_sec=0.0, sync_fn=None, use_mlx_abba=True):
    def _wrap(fn):
        def _inner():
            t0 = time.perf_counter()
            out = fn()
            if sync_fn is not None:
                sync_fn(out)
            dt = time.perf_counter() - t0
            return {"wall_sec": dt, "payload": out}
        return _inner

    if use_mlx_abba:
        try:
            res = interleaved_abba(
                _wrap(fn_a),
                _wrap(fn_b),
                warmup=True,
                rounds=rounds,
                cooldown_sec=cooldown_sec,
            )
            a_runs = res["A_runs"]
            b_runs = res["B_runs"]
            a_med = float(statistics.median([r["wall_sec"] for r in a_runs]))
            b_med = float(statistics.median([r["wall_sec"] for r in b_runs]))
            a_drift = round(a_runs[-1]["wall_sec"] / max(a_runs[0]["wall_sec"], 1e-9), 4) if len(a_runs) >= 2 else 1.0
            b_drift = round(b_runs[-1]["wall_sec"] / max(b_runs[0]["wall_sec"], 1e-9), 4) if len(b_runs) >= 2 else 1.0
            return {
                "order": res["order"],
                "a_median_s": a_med,
                "b_median_s": b_med,
                "a_runs": a_runs,
                "b_runs": b_runs,
                "a_drift": a_drift,
                "b_drift": b_drift,
            }
        except Exception:
            pass

    # Fallback matched A-B-B-A loop (e.g. PyTorch MPS venv without mlx installed).
    wa = _wrap(fn_a)
    wb = _wrap(fn_b)
    _ = wa()
    _ = wb()
    order = ["A", "B", "B", "A"] * max(1, rounds // 2)
    if rounds % 2 == 1:
        order.extend(["A", "B"])
    a_runs, b_runs = [], []
    for idx, tag in enumerate(order):
        if idx > 0 and cooldown_sec > 0.0:
            time.sleep(cooldown_sec)
        (a_runs if tag == "A" else b_runs).append(wa() if tag == "A" else wb())
    a_med = float(statistics.median([r["wall_sec"] for r in a_runs]))
    b_med = float(statistics.median([r["wall_sec"] for r in b_runs]))
    a_drift = round(a_runs[-1]["wall_sec"] / max(a_runs[0]["wall_sec"], 1e-9), 4) if len(a_runs) >= 2 else 1.0
    b_drift = round(b_runs[-1]["wall_sec"] / max(b_runs[0]["wall_sec"], 1e-9), 4) if len(b_runs) >= 2 else 1.0
    return {
        "order": order,
        "a_median_s": a_med,
        "b_median_s": b_med,
        "a_runs": a_runs,
        "b_runs": b_runs,
        "a_drift": a_drift,
        "b_drift": b_drift,
    }

def validate_extract_schema(obj, schema_obj):
    if not isinstance(obj, dict):
        return False
    if schema_obj is not None:
        try:
            import jsonschema
            jsonschema.validate(instance=obj, schema=schema_obj)
            return True
        except ImportError:
            pass
        except Exception:
            return False
        # Structural fallback enforcing required keys, enums, types, and additionalProperties=false
        props = schema_obj.get("properties") or {}
        defs = schema_obj.get("$defs") or {}
        req = schema_obj.get("required") or []
        for rk in req:
            if rk not in obj:
                return False
        if schema_obj.get("additionalProperties") is False:
            if any(k not in props for k in obj):
                return False
        for k, v in obj.items():
            ps = props.get(k)
            if not isinstance(ps, dict):
                continue
            if "$ref" in ps and ps["$ref"].startswith("#/$defs/"):
                ps = defs.get(ps["$ref"].split("/")[-1], ps)
            t = ps.get("type")
            if t == "string" and not isinstance(v, str):
                return False
            if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
                return False
            if t == "array" and not isinstance(v, list):
                return False
            if "enum" in ps and v not in ps["enum"]:
                return False
            if t == "integer":
                if "minimum" in ps and v < ps["minimum"]:
                    return False
                if "maximum" in ps and v > ps["maximum"]:
                    return False
            if t == "array":
                if "maxItems" in ps and len(v) > ps["maxItems"]:
                    return False
                item_s = ps.get("items") or {}
                if item_s.get("type") == "string" and not all(isinstance(x, str) for x in v):
                    return False
                if item_s.get("type") == "object":
                    ireq = item_s.get("required") or []
                    iprops = item_s.get("properties") or {}
                    for it in v:
                        if not isinstance(it, dict):
                            return False
                        if any(rk not in it for rk in ireq):
                            return False
                        if item_s.get("additionalProperties") is False and any(ik not in iprops for ik in it):
                            return False
        return True
    return all(isinstance(obj.get(k), str) and bool(obj.get(k)) for k in ("status", "text_quality", "primary_kind", "domain"))

pkg = cfg["package"]
resolved_home = Path(cfg["resolved_home"])
use_wired = bool(cfg.get("wired_limit", True))
cooldown_sec = float(cfg.get("cooldown_sec", 0.0))

def skip(reason):
    print(json.dumps({"status": "skipped", "runtime": "unavailable", "highlight": "-", "reason": reason}), flush=True)
    raise SystemExit(0)

if pkg in ("fast-bge-small", "fast-embeddinggemma-2"):
    if pkg == "fast-bge-small":
        snap = (
            resolve_hf_snapshot("mlx-community/bge-small-en-v1.5-8bit")
            or resolve_hf_snapshot("mlx-community/bge-small-en-v1.5-bf16")
            or resolve_hf_snapshot("BAAI/bge-small-en-v1.5")
        )
        repo_id = "mlx-community/bge-small-en-v1.5-8bit"
    else:
        snap = (
            resolve_hf_snapshot("mlx-community/embeddinggemma-2-8bit")
            or resolve_hf_snapshot("mlx-community/embeddinggemma-2-bf16")
            or resolve_hf_snapshot("google/embeddinggemma-2")
        )
        repo_id = "mlx-community/embeddinggemma-2-8bit"
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
        with mlx_wired_scope(use_wired):
            t0 = time.perf_counter()
            model, tokenizer = mlx_embeddings.utils.load(snap)
            load_s = time.perf_counter() - t0
            q_txt = ["What drives oceanic heat transport?"]
            texts = [f"Record {i}. " + ("Ocean currents transfer heat across hemispheres. " * max(1, int(l) // 10)) for i, l in enumerate(lens)]
            mx.reset_peak_memory()

            def _sync_mlx(val):
                if val is not None:
                    try:
                        mx.eval(getattr(val, "text_embeds", val))
                    except Exception:
                        pass
                if hasattr(mx, "synchronize"):
                    mx.synchronize()

            def _run_b1():
                te = None
                for _ in range(8):
                    out = mlx_embeddings.utils.generate(model, tokenizer, q_txt, max_length=512)
                    te = getattr(out, "text_embeds", out)
                    mx.eval(te)
                return te

            def _run_corpus():
                te = None
                for i in range(0, len(texts), b_corpus):
                    out = mlx_embeddings.utils.generate(model, tokenizer, texts[i : i + b_corpus], max_length=512)
                    te = getattr(out, "text_embeds", out)
                    mx.eval(te)
                return te

            abba = run_abba_pair(_run_b1, _run_corpus, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_mlx, use_mlx_abba=True)
            b1_dt = abba["a_median_s"] / 8.0
            c_dt = abba["b_median_s"]
            fp_gb, rss_gb = footprint_gb()
            tok_s = tot_toks / max(c_dt, 1e-9)
            print(json.dumps({
                "status": "ok",
                "runtime": "mlx-embeddings",
                "timing_order": abba["order"],
                "b1_drift": abba["a_drift"],
                "corpus_drift": abba["b_drift"],
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
    except (ImportError, ValueError):
        pass

    # For `fast-embeddinggemma-2` (`mlx-community/embeddinggemma-2-bf16`), stock MLX support lives in `mlx_vlm.embedding_loader`.
    if pkg == "fast-embeddinggemma-2":
        try:
            import mlx.core as mx
            from mlx_vlm.embedding_loader import load_embedding_model
            from transformers import AutoTokenizer

            with mlx_wired_scope(use_wired):
                t0 = time.perf_counter()
                model = load_embedding_model(Path(snap))
                tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
                load_s = time.perf_counter() - t0
                q_inp = tokenizer(["task: search result | query: What drives oceanic heat transport?"], padding=True, return_tensors="np")
                q_mx = {k: mx.array(v) for k, v in q_inp.items()}
                texts = [f"title: none | text: Record {i}. " + ("Ocean currents transfer heat across hemispheres. " * max(1, int(l) // 10)) for i, l in enumerate(lens)]
                pre_batches = []
                for i in range(0, len(texts), b_corpus):
                    bi = tokenizer(texts[i : i + b_corpus], padding=True, truncation=True, max_length=512, return_tensors="np")
                    pre_batches.append({k: mx.array(v) for k, v in bi.items()})
                mx.reset_peak_memory()

                def _sync_vlm(val):
                    if val is not None:
                        mx.eval(val)
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()

                def _run_b1():
                    out = None
                    for _ in range(8):
                        out = model(**q_mx).text_embeds
                        mx.eval(out)
                    return out

                def _run_corpus():
                    last = None
                    for b_mx in pre_batches:
                        last = model(**b_mx).text_embeds
                        mx.eval(last)
                    return last

                abba = run_abba_pair(_run_b1, _run_corpus, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_vlm, use_mlx_abba=True)
                b1_dt = abba["a_median_s"] / 8.0
                c_dt = abba["b_median_s"]
                fp_gb, rss_gb = footprint_gb()
                tok_s = tot_toks / max(c_dt, 1e-9)
                print(json.dumps({
                    "status": "ok",
                    "runtime": "stock-mlx_vlm-embedding",
                    "timing_order": abba["order"],
                    "b1_drift": abba["a_drift"],
                    "corpus_drift": abba["b_drift"],
                    "b1_ms": round(b1_dt * 1000.0, 3),
                    "tok_s": round(tok_s, 1),
                    "docs_per_sec": round(n_docs / max(c_dt, 1e-9), 2),
                    "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
                    "peak_footprint_gb": fp_gb,
                    "rss_gb": rss_gb,
                    "load_s": round(load_s, 3),
                    "highlight": f"mlx_vlm-embedding: {tok_s:,.0f} tok/s (B=1 {b1_dt * 1000.0:.2f}ms)",
                }), flush=True)
                raise SystemExit(0)
        except Exception:
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

        def _sync_mps(_):
            torch.mps.synchronize()

        def _run_q():
            for _ in range(10):
                _ = model(input_ids=q_ids, attention_mask=q_mask)

        def _run_c():
            for bi, bm in batches:
                _ = model(input_ids=bi, attention_mask=bm)

        abba = run_abba_pair(_run_q, _run_c, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_mps, use_mlx_abba=False)
        b1_dt = abba["a_median_s"] / 10.0
        c_dt = abba["b_median_s"]
    fp_gb, rss_gb = footprint_gb()
    tok_s = tot_toks / max(c_dt, 1e-9)
    print(json.dumps({
        "status": "ok",
        "runtime": f"transformers-mps-{str(dt_torch).split('.')[-1]}",
        "timing_order": abba["order"],
        "b1_drift": abba["a_drift"],
        "corpus_drift": abba["b_drift"],
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
    schema_path = fix_dir / "extract_v1.schema.json"
    schema_obj = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else None
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

    with mlx_wired_scope(use_wired):
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

        b_sz = 64
        batch_fmts = [fmts[i % len(fmts)] for i in range(b_sz)]
        batch4_fmts = [fmts[i % len(fmts)] for i in range(4)]

        def _sync_mlx(_):
            if hasattr(mx, "synchronize"):
                mx.synchronize()

        def _run_b1():
            return mlx_vlm.generate(model, processor, fmts[0], max_tokens=512, verbose=False)

        def _run_b4():
            return mlx_vlm.batch_generate(model, processor, prompts=batch4_fmts, max_tokens=512, verbose=False)

        def _run_b64():
            return mlx_vlm.batch_generate(model, processor, prompts=batch_fmts, max_tokens=512, verbose=False)

        abba = run_abba_pair(_run_b1, _run_b64, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_mlx, use_mlx_abba=True)
        abba_b4 = run_abba_pair(_run_b1, _run_b4, rounds=1, cooldown_sec=cooldown_sec, sync_fn=_sync_mlx, use_mlx_abba=True)
        dt_b1 = abba["a_median_s"]
        dt_b4 = abba_b4["b_median_s"]
        dt_batch = abba["b_median_s"]
        r_b1 = abba["a_runs"][-1]["payload"]
        r_b4 = abba_b4["b_runs"][-1]["payload"]
        r_batch = abba["b_runs"][-1]["payload"]

        b1_out_toks = int(getattr(r_b1, "generation_tokens", 0) or max(1, len(tok.encode(getattr(r_b1, "text", str(r_b1))))))
        b1_e2e_tps = b1_out_toks / max(dt_b1, 1e-9)

        b4_stats = getattr(r_b4, "stats", None)
        b4_texts = list(getattr(r_b4, "texts", []) or [])
        b4_tot_gen = int(getattr(b4_stats, "generation_tokens", 0) or sum(len(tok.encode(tx)) for tx in b4_texts))
        b4_dec_tps = float(getattr(b4_stats, "generation_tps", 0.0) or (b4_tot_gen / max(dt_b4, 1e-9)))
        b4_docs_s = 4.0 / max(dt_b4, 1e-9)

        batch_stats = getattr(r_batch, "stats", None)
        texts = list(getattr(r_batch, "texts", []) or [])
        tot_gen = int(getattr(batch_stats, "generation_tokens", 0) or 0)
        if tot_gen <= 0 and texts:
            tot_gen = sum(len(tok.encode(tx)) for tx in texts)

        valid_json = 0
        valid_schema = 0
        for tx in texts:
            s = tx.strip()
            if s.startswith("```"):
                s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                parsed_obj = json.loads(s)
                valid_json += 1
                if validate_extract_schema(parsed_obj, schema_obj):
                    valid_schema += 1
            except Exception:
                pass

        batch_e2e_tps = tot_gen / max(dt_batch, 1e-9)
        docs_s = b_sz / max(dt_batch, 1e-9)
        valid_docs_s = valid_schema / max(dt_batch, 1e-9)
        valid_schema_rate = valid_schema / max(len(texts), 1)
        dec_tps = float(getattr(batch_stats, "generation_tps", batch_e2e_tps) or batch_e2e_tps)
        peak_gb = round(float(mx.get_peak_memory()) / 1e9, 4)
        fp_gb, rss_gb = footprint_gb()
        print(json.dumps({
            "status": "ok",
            "runtime": "stock-mlx_vlm",
            "timing_order": abba["order"],
            "b1_drift": abba["a_drift"],
            "batch_drift": abba["b_drift"],
            "b1_e2e_out_tok_s": round(b1_e2e_tps, 1),
            "b1_docs_per_sec": round(1.0 / max(dt_b1, 1e-9), 2),
            "b4_decode_tok_s": round(b4_dec_tps, 1),
            "b4_docs_per_sec": round(b4_docs_s, 2),
            "batch_size": b_sz,
            "e2e_out_tok_s": round(batch_e2e_tps, 1),
            "docs_per_sec": round(docs_s, 3),
            "valid_docs_per_sec": round(valid_docs_s, 3),
            "decode_tok_s": round(dec_tps, 1),
            "valid_json_count": valid_json,
            "valid_schema_count": valid_schema,
            "valid_schema_rate": round(valid_schema_rate, 4),
            "peak_active_gb": peak_gb,
            "peak_footprint_gb": fp_gb,
            "rss_gb": rss_gb,
            "load_s": round(load_s, 3),
            "highlight": (
                f"mlx_vlm B={b_sz}: {dec_tps:.1f} dec tok/s ({valid_docs_s:.2f} valid docs/s, "
                f"schema={valid_schema}/{b_sz}, json={valid_json}/{b_sz}, peak={peak_gb:.2f}GB; B=1 {b1_e2e_tps:.1f} tok/s)"
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

    with mlx_wired_scope(use_wired):
        t0 = time.perf_counter()
        model, tokenizer = mlx_lm.load(snap)
        load_s = time.perf_counter() - t0
        prompt = "Summarize sliding-window attention and grouped-query attention in JSON with two short keys."
        p_ids = tokenizer.encode(prompt)
        has_bg = hasattr(mlx_lm, "batch_generate")

        def _sync_mlx(_):
            if hasattr(mx, "synchronize"):
                mx.synchronize()

        def _run_b1():
            return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)

        def _run_b8():
            if has_bg:
                return mlx_lm.batch_generate(model, tokenizer, prompts=[p_ids] * 8, max_tokens=128, verbose=False)
            return mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)

        abba = run_abba_pair(_run_b1, _run_b8, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_mlx, use_mlx_abba=True)
        dt_1 = abba["a_median_s"]
        text_out = abba["a_runs"][-1]["payload"]
        out_toks = max(1, len(tokenizer.encode(text_out)))
        b1_tps = out_toks / max(dt_1, 1e-9)

        if has_bg:
            dt_b = abba["b_median_s"]
            bres = abba["b_runs"][-1]["payload"]
            bstats = getattr(bres, "stats", None)
            batch_tps = float(getattr(bstats, "generation_tps", 0.0) or ((8 * 128) / max(dt_b, 1e-9)))
            docs_s = 8.0 / max(dt_b, 1e-9)
            b_label = "B=8"
        else:
            batch_tps = b1_tps
            docs_s = 1.0 / max(dt_1, 1e-9)
            b_label = "B=1"

        fp_gb, rss_gb = footprint_gb()
        print(json.dumps({
            "status": "ok",
            "runtime": "stock-mlx_lm",
            "timing_order": abba["order"],
            "b1_drift": abba["a_drift"],
            "batch_drift": abba["b_drift"],
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
    kev_8bit_snap = resolve_hf_snapshot("AbstractFramework/kev-0.8b-mlx-8bit")
    kev_dir = resolved_home / "artifacts" / "kev_08b"
    if kev_8bit_snap is None and not (kev_dir / "model.safetensors").is_file():
        skip(f"missing AbstractFramework/kev-0.8b-mlx-8bit and BF16 kev_08b weights at {kev_dir}")
    try:
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.models import qwen3_5

        with mlx_wired_scope(use_wired):
            t0 = time.perf_counter()
            if kev_8bit_snap is not None:
                mdl, _ = mlx_lm.load(str(kev_8bit_snap))
                rt_lbl = "stock-mlx_lm-kev-8bit"
            else:
                meta = json.loads((kev_dir / "kev_meta.json").read_text(encoding="utf-8"))
                raw = mx.load(str(kev_dir / "model.safetensors"))
                margs = qwen3_5.ModelArgs.from_dict({"text_config": meta["text_config"]})
                lm_w = {(k if k.startswith("language_model.") else f"language_model.{k}"): v for k, v in raw.items() if not k.startswith("pointer_head.")}
                mdl = qwen3_5.Model(margs)
                if hasattr(mdl, "sanitize"):
                    lm_w = mdl.sanitize(lm_w)
                mdl.load_weights(list(lm_w.items()), strict=False)
                mx.eval(mdl.parameters())
                mdl.eval()
                del raw, lm_w
                rt_lbl = "stock-mlx_lm-qwen3_5-bf16"
            load_s = time.perf_counter() - t0

            backbone = mdl.language_model.model if hasattr(mdl, "language_model") else mdl.model
            seq_len = 96
            x_b1 = mx.random.randint(10, 2000, (1, seq_len), dtype=mx.int32)
            x_b8 = mx.random.randint(10, 2000, (8, seq_len), dtype=mx.int32)
            mx.reset_peak_memory()

            def _sync_q(_):
                if hasattr(mx, "synchronize"):
                    mx.synchronize()

            def _run_b1():
                out = None
                for _ in range(4):
                    out = backbone(x_b1)
                    mx.eval(out)
                return out

            def _run_b8():
                out = None
                for _ in range(4):
                    out = backbone(x_b8)
                    mx.eval(out)
                return out

            abba = run_abba_pair(_run_b1, _run_b8, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_q, use_mlx_abba=True)
            dt_b1 = abba["a_median_s"] / 4.0
            dt_b8 = abba["b_median_s"] / 4.0
            b1_qs = 1.0 / max(dt_b1, 1e-9)
            b8_qs = 8.0 / max(dt_b8, 1e-9)
            fp_gb, rss_gb = footprint_gb()
            print(json.dumps({
                "status": "ok",
                "runtime": rt_lbl,
                "timing_order": abba["order"],
                "b1_drift": abba["a_drift"],
                "b8_drift": abba["b_drift"],
                "b1_q_per_sec": round(b1_qs, 2),
                "q_per_sec": round(b8_qs, 2),
                "ms_per_question": round((dt_b8 * 1000.0) / 8.0, 2),
                "peak_active_gb": round(float(mx.get_peak_memory()) / 1e9, 4),
                "peak_footprint_gb": fp_gb,
                "rss_gb": rss_gb,
                "load_s": round(load_s, 3),
                "highlight": f"{rt_lbl} B=8: {b8_qs:.2f} q/s ({(dt_b8 * 1000.0) / 8.0:.2f} ms/q; B=1 {b1_qs:.2f} q/s)",
            }), flush=True)
            raise SystemExit(0)
    except Exception:
        pass
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
        x_b8 = torch.randint(10, 2000, (8, seq_len), device="mps", dtype=torch.long)
        m_b8 = torch.ones_like(x_b8)

        def _sync_mps(_):
            torch.mps.synchronize()

        def _run_b1():
            for _ in range(4):
                _ = model(input_ids=x_b1, attention_mask=m_b1, use_cache=False)

        def _run_b8():
            for _ in range(4):
                _ = model(input_ids=x_b8, attention_mask=m_b8, use_cache=False)

        abba = run_abba_pair(_run_b1, _run_b8, rounds=2, cooldown_sec=cooldown_sec, sync_fn=_sync_mps, use_mlx_abba=False)
        dt_b1 = abba["a_median_s"] / 4.0
        dt_b8 = abba["b_median_s"] / 4.0
    b1_qs = 1.0 / max(dt_b1, 1e-9)
    b8_qs = 8.0 / max(dt_b8, 1e-9)
    fp_gb, rss_gb = footprint_gb()
    print(json.dumps({
        "status": "ok",
        "runtime": "transformers-mps-bf16",
        "timing_order": abba["order"],
        "b1_drift": abba["a_drift"],
        "b8_drift": abba["b_drift"],
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
    wired_limit: bool = True,
    cooldown_sec: float = 0.0,
) -> dict[str, Any]:
    """Execute the public Apple Silicon baseline for `pkg_name` in an isolated subprocess."""
    if pkg_name == "fast-pplx":
        return {"status": "delegated", "reason": "reported inside fast-pplx speed suite (pplx-context-mlx-f16 / torch-mps / torch-cpu)"}

    py_exe = public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    payload = {
        "package": pkg_name,
        "resolved_home": str(resolved_home),
        "fast_shared_src": str(_FAST_SHARED_SRC),
        "wired_limit": bool(wired_limit),
        "cooldown_sec": float(cooldown_sec),
    }
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="public-baselines",
        description="Run apples-to-apples public Apple Silicon baseline for one fast-silicon package.",
    )
    parser.add_argument(
        "--package",
        required=True,
        choices=("fast-bge-small", "fast-embeddinggemma-2", "fast-gemma4", "fast-gpt-oss", "fast-decision-models"),
        help="Target package to compare against its public open-source baseline",
    )
    parser.add_argument("--resolved-home", type=Path, default=None, help="Optional package artifact home directory")
    parser.add_argument("--public-python", default=None, help="Python executable holding public MLX/PyTorch runtimes")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--cooldown-sec", type=float, default=0.0)
    parser.add_argument("--no-wired-limit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    home = args.resolved_home or (_REPO_ROOT / "packages" / args.package)
    res = run_public_baseline(
        pkg_name=args.package,
        resolved_home=home,
        public_python=args.public_python,
        timeout_sec=args.timeout_sec,
        wired_limit=not args.no_wired_limit,
        cooldown_sec=args.cooldown_sec,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") in ("ok", "skipped", "delegated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
