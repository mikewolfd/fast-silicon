"""Stock Apple `CoreAI` and public `MLX` model adapters for `scripts/benchmark_outcomes.py`.

Provides thin, zero-duplication adapters conforming to the exact Category 1 (encoder),
Category 2 (typed decision scorer), and Category 3-4 (chat/structured LLM) call contracts
used by `scripts/benchmark_outcomes.py`, guaranteeing identical datasets, chunking, prompts,
4-bucket memory attribution, and Effective Outcome Throughput (`EOT`) calculation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from public_baselines import (
    resolve_coreai_bundle,
    resolve_coreai_python,
    resolve_hf_snapshot,
    resolve_pplx_mlx_path,
)

_FAST_SHARED_SRC = Path(__file__).resolve().parents[1] / "packages" / "fast-shared" / "src"

_COREAI_DAEMON_PREAMBLE = r"""
import ctypes, json, os, sys, time
from pathlib import Path


def _peak_gb_fallback():
    buf = ctypes.create_string_buffer(512)
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    return round(int.from_bytes(buf.raw[240:248], "little") / 1e9, 4) if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0 else 0.0


def _inspect_aimodel_asset(aimodel_path):
    if not aimodel_path:
        return None
    p = Path(str(aimodel_path))
    if not p.exists():
        return None
    try:
        from coreai.runtime import AIModelAsset
        raw = AIModelAsset(str(p)).summary(include_statistics=True)
        out = {"asset_valid": True, "path": str(p)}
        for k in ("storage_types", "compute_types", "operation_distribution"):
            val = raw.get(k) if isinstance(raw, dict) else getattr(raw, k, None)
            if val is not None:
                out[k] = dict(val) if isinstance(val, dict) else val
        return out
    except Exception:
        return None


class _DaemonTelemetry:
    def __init__(self, model_dirs=None):
        self.model_dirs = [str(Path(d)) for d in (model_dirs or []) if d]
        self.cold_load_ms = None
        self.cold_first_infer_ms = None
        self.warm_latencies_ms = []
        self.external_peak_gb = 0.0
        self.asset_summary = None
        self.profiler_available = False
        self._sent_asset = False
        self._probe = None
        self._collect_4b = None
        self._summarize_lat = None
        try:
            from fast_shared.bench import (
                FourBucketMemoryProbe,
                collect_daemon_four_bucket_metrics,
                summarize_cold_warm_latencies,
            )
            self._collect_4b = collect_daemon_four_bucket_metrics
            self._summarize_lat = summarize_cold_warm_latencies
            self._probe = FourBucketMemoryProbe(model_directories=self.model_dirs)
        except Exception:
            pass

    def mark_loaded(self, aimodel_path=None, load_ms=None):
        if load_ms is not None and self.cold_load_ms is None:
            self.cold_load_ms = round(float(load_ms), 3)
        try:
            import coreai.runtime as _cr
            self.profiler_available = hasattr(_cr, "Profiler")
        except Exception:
            self.profiler_available = False
        if self.asset_summary is None and aimodel_path is not None:
            self.asset_summary = _inspect_aimodel_asset(aimodel_path)

    def record_infer(self, elapsed_ms):
        ms = round(float(elapsed_ms), 3)
        if self.cold_first_infer_ms is None:
            self.cold_first_infer_ms = ms
        else:
            self.warm_latencies_ms.append(ms)

    def observe_external_peak_gb(self, pk):
        if pk is not None:
            self.external_peak_gb = max(self.external_peak_gb, float(pk))

    def payload(self, *, final=False):
        if self._collect_4b is not None:
            four = self._collect_4b(self._probe, model_directories=self.model_dirs, evict_mapped=False)
            fb_for_json = {k: v for k, v in four.items() if k not in ("four_bucket_deltas", "ane_daemons")}
        else:
            pk_fb = _peak_gb_fallback()
            fb_for_json = {
                "pid": os.getpid(),
                "in_process": True,
                "rusage_version": 4,
                "lifetime_max_phys_footprint_mb": round((pk_fb * 1e9) / (1024.0 * 1024.0), 2),
                "combined_with_model_mapped_gb": pk_fb,
            }
        if self._summarize_lat is not None:
            lat = self._summarize_lat(
                cold_load_ms=self.cold_load_ms,
                cold_first_infer_ms=self.cold_first_infer_ms,
                warm_latencies_ms=self.warm_latencies_ms,
            )
        else:
            lat = {
                "cold_load_ms": self.cold_load_ms,
                "cold_first_infer_ms": self.cold_first_infer_ms,
                "warm_count": len(self.warm_latencies_ms),
            }
        max_phys_gb = float(fb_for_json.get("lifetime_max_phys_footprint_mb", 0.0)) * 1048576.0 / 1e9
        comb_gb = float(fb_for_json.get("combined_with_model_mapped_gb", 0.0))
        pk = round(max(max_phys_gb, comb_gb, self.external_peak_gb), 4)
        out = {
            "peak_gb": pk,
            "four_bucket_metrics": fb_for_json,
            "cold_warm_latency": lat,
            "profiler_available": bool(self.profiler_available),
        }
        if self.asset_summary is not None and (final or not self._sent_asset):
            out["asset_summary"] = self.asset_summary
            self._sent_asset = True
        return out
""".lstrip()


def _build_coreai_daemon_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment ensuring `packages/fast-shared/src` is importable on `PYTHONPATH`."""
    env = dict(os.environ if base_env is None else base_env)
    src_str = str(_FAST_SHARED_SRC)
    cur = env.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p] if cur else []
    if src_str not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src_str, *parts])
    return env


class _CoreAiSubprocessClient:
    """Shared JSON-lines subprocess lifecycle & telemetry collector for Core AI daemons."""

    def __init__(
        self,
        *,
        cmd: Sequence[str | Path],
        daemon_script: str,
        script_name: str = "daemon.py",
        script_args: Sequence[str | Path] = (),
        prefix: str = "coreai_daemon_",
        env: Mapping[str, str] | None = None,
        popen_cls: Any = None,
        include_preamble: bool = True,
    ) -> None:
        self._tmp_obj = tempfile.TemporaryDirectory(prefix=prefix)
        self.tmp_dir = Path(self._tmp_obj.name)
        script_path = self.tmp_dir / script_name
        full_src = f"{_COREAI_DAEMON_PREAMBLE}\n{daemon_script}" if include_preamble else daemon_script
        script_path.write_text(full_src, encoding="utf-8")

        spawn = subprocess.Popen if popen_cls is None else popen_cls
        full_cmd = [*(str(c) for c in cmd), str(script_path), *(str(a) for a in script_args)]
        self._proc = spawn(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=_build_coreai_daemon_env(env),
        )
        ready = self._proc.stdout.readline().strip() if self._proc.stdout else ""
        if ready != "READY":
            self._proc.kill()
            self._tmp_obj.cleanup()
            raise RuntimeError(f"CoreAI daemon ({script_name}) failed to start: {ready}")
        self._closed = False
        self._call_n = 0
        self._telemetry: dict[str, Any] = {
            "peak_gb": 0.0,
            "four_bucket_metrics": {},
            "cold_warm_latency": {},
            "profiler_available": False,
        }

    def _merge_response(self, resp: Mapping[str, Any]) -> None:
        if "peak_gb" in resp and resp["peak_gb"] is not None:
            self._telemetry["peak_gb"] = round(max(float(self._telemetry.get("peak_gb", 0.0)), float(resp["peak_gb"])), 4)
        for key in ("four_bucket_metrics", "cold_warm_latency", "asset_summary"):
            val = resp.get(key)
            if isinstance(val, Mapping):
                self._telemetry[key] = dict(val)
        if "profiler_available" in resp:
            self._telemetry["profiler_available"] = bool(resp["profiler_available"])

    @property
    def peak_gb(self) -> float:
        return float(self._telemetry.get("peak_gb", 0.0))

    def next_temp_path(self, pattern: str) -> Path:
        self._call_n += 1
        return self.tmp_dir / pattern.format(n=self._call_n)

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("CoreAI subprocess client is already closed")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(dict(payload)) + "\n")
        self._proc.stdin.flush()
        raw_line = self._proc.stdout.readline()
        if not raw_line:
            raise RuntimeError("CoreAI daemon exited unexpectedly before responding")
        resp = json.loads(raw_line)
        self._merge_response(resp)
        return resp

    def telemetry(self) -> dict[str, Any]:
        return {
            k: (dict(v) if isinstance(v, Mapping) else v)
            for k, v in self._telemetry.items()
        }

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.telemetry()
        self._closed = True
        try:
            if getattr(self._proc, "stdin", None) and getattr(self._proc, "stdout", None):
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                raw_line = self._proc.stdout.readline()
                if raw_line:
                    self._merge_response(json.loads(raw_line))
                self._proc.wait(timeout=10)
        finally:
            self._tmp_obj.cleanup()
        return self.telemetry()


@dataclass
class Cat1Adapter:
    """Category 1 retrieval encoder adapter."""

    harness_label: str
    mrl_dims: tuple[int | str, ...]
    int8_mode: str
    nb_pre_normalized: bool
    enc_q_nb: Callable[[list[str]], np.ndarray]
    enc_d_nb: Callable[[list[str]], np.ndarray]
    enc_q: Callable[[list[str]], np.ndarray]
    enc_d: Callable[[list[str]], np.ndarray]
    enc_coarse_2k: Callable[[list[str]], np.ndarray] | None = None
    enc_ctx: Callable[[list[list[str]]], list[np.ndarray]] | None = None
    close_fn: Callable[[], float | None] | None = None
    telemetry_fn: Callable[[], dict[str, Any]] | None = None

    def telemetry(self) -> dict[str, Any] | None:
        return self.telemetry_fn() if self.telemetry_fn is not None else None

    def close(self) -> float | None:
        if self.close_fn is not None:
            return self.close_fn()
        return None


def build_public_cat1_adapter(
    pkg_name: str,
    *,
    tier: str,
    coreai_python: str | None = None,
    pplx_mlx_path: str | None = None,
) -> Cat1Adapter:
    """Instantiate the fastest verified stock baseline encoder for Category 1."""
    if pkg_name == "public-bge-small":
        import mlx.core as mx
        import mlx_embeddings

        snap8 = resolve_hf_snapshot("mlx-community/bge-small-en-v1.5-8bit")
        snap = snap8 or resolve_hf_snapshot("mlx-community/bge-small-en-v1.5-bf16")
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for mlx-community/bge-small-en-v1.5-8bit")
        mdl, tok = mlx_embeddings.utils.load(str(snap))
        q_prefix = "Represent this sentence for searching relevant passages: "
        lbl = "mlx-embeddings-8bit" if snap8 is not None else "mlx-embeddings-bf16"

        def _enc_bge(txts: list[str], *, prefix: str = "", bsz: int = 64) -> np.ndarray:
            vecs: list[np.ndarray] = []
            for i in range(0, len(txts), bsz):
                sub = [prefix + t for t in txts[i : i + bsz]]
                out = mlx_embeddings.utils.generate(mdl, tok, sub, max_length=512)
                te = out.text_embeds
                mx.eval(te)
                vecs.append(np.asarray(te.astype(mx.float32)))
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)

        return Cat1Adapter(
            harness_label=lbl,
            mrl_dims=(128, 256, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_bge(txts, prefix=q_prefix),
            enc_d_nb=lambda txts: _enc_bge(txts, prefix=""),
            enc_q=lambda txts: _enc_bge(txts, prefix=q_prefix),
            enc_d=lambda txts: _enc_bge(txts, prefix=""),
        )

    if pkg_name == "public-embeddinggemma-2":
        import mlx.core as mx
        from mlx_vlm.embedding_loader import load_embedding_model
        from transformers import AutoTokenizer

        snap8 = resolve_hf_snapshot("mlx-community/embeddinggemma-2-8bit")
        snap = (
            snap8
            or resolve_hf_snapshot("mlx-community/embeddinggemma-2-bf16")
            or resolve_hf_snapshot("google/embeddinggemma-2")
        )
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for mlx-community/embeddinggemma-2-8bit")
        mdl = load_embedding_model(Path(snap))
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        q_prefix = "task: search result | query: "
        d_prefix = "title: none | text: "
        lbl = "mlx_vlm-embedding-8bit" if snap8 is not None else "mlx_vlm-embedding-bf16"

        def _enc_emb2(txts: list[str], *, prefix: str, max_len: int, bsz: int) -> np.ndarray:
            out_vecs: list[np.ndarray] = []
            for i in range(0, len(txts), bsz):
                sub = [prefix + t for t in txts[i : i + bsz]]
                inp = tok(sub, padding=True, truncation=True, max_length=max_len, return_tensors="np")
                emb = mdl(**{k: mx.array(v) for k, v in inp.items()}).text_embeds.astype(mx.float32)
                emb = emb / mx.maximum(mx.linalg.norm(emb, axis=-1, keepdims=True), 1e-12)
                mx.eval(emb)
                out_vecs.append(np.asarray(emb))
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            return np.vstack(out_vecs) if out_vecs else np.zeros((0, 768), dtype=np.float32)

        return Cat1Adapter(
            harness_label=lbl,
            mrl_dims=(128, 256, 512, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512, bsz=32),
            enc_d_nb=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512, bsz=32),
            enc_q=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512, bsz=32),
            enc_d=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512, bsz=32),
            enc_coarse_2k=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=2048, bsz=8),
        )

    if pkg_name == "coreai-embeddinggemma-2":
        from tokenizers import Tokenizer

        ca_py = resolve_coreai_python(coreai_python)
        ca_bundle = resolve_coreai_bundle("emb2_bf16/embeddinggemma-2-text") or resolve_coreai_bundle("emb2_bf16")
        snap = (
            resolve_hf_snapshot("mlx-community/embeddinggemma-2-8bit")
            or resolve_hf_snapshot("mlx-community/embeddinggemma-2-bf16")
            or resolve_hf_snapshot("google/embeddinggemma-2")
        )
        if ca_py is None or ca_bundle is None or not (ca_bundle / "text_encoder.aimodel").exists() or snap is None:
            raise FileNotFoundError("CoreAI emb2_bf16/text_encoder.aimodel or coreai runtime not staged on this host")

        tok_fast = Tokenizer.from_file(str(snap / "tokenizer.json"))
        tok_fast.no_truncation()
        tok_fast.no_padding()
        q_prefix = "task: search result | query: "
        d_prefix = "title: none | text: "

        daemon_code = r"""
import asyncio
from coreai.runtime import AIModel, NDArray
import numpy as np

async def main():
    bdir = Path(sys.argv[1])
    aim_path = bdir / "text_encoder.aimodel"
    telem = _DaemonTelemetry([bdir])
    t_load = time.perf_counter()
    model = await AIModel.load(str(aim_path))
    telem.mark_loaded(aim_path, (time.perf_counter() - t_load) * 1000.0)
    fns = {
        (128, 32): model.load_function("len128_b32"),
        (128, 8): model.load_function("len128_b8"),
        (128, 1): model.load_function("len128_b1"),
        (512, 32): model.load_function("len512_b32"),
        (512, 8): model.load_function("len512_b8"),
        (512, 1): model.load_function("len512_b1"),
        (2048, 8): model.load_function("len2048_b8"),
        (2048, 1): model.load_function("len2048_b1"),
    }
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\n")
            sys.stdout.flush()
            break
        seqs = req["seqs"]
        max_cap = int(req["max_cap"])
        vecs = []
        idx = 0
        while idx < len(seqs):
            rem = len(seqs) - idx
            sub_peek = seqs[idx : idx + min(rem, 32 if max_cap <= 512 else 8)]
            ml = max(len(s) for s in sub_peek)
            L = 128 if ml <= 128 else (512 if ml <= 512 else 2048)
            B = (8 if rem >= 8 else 1) if L == 2048 else (32 if rem >= 32 else (8 if rem >= 8 else 1))
            chunk = seqs[idx : idx + B]
            ml_c = max(len(s) for s in chunk)
            if ml_c > L:
                L = 512 if ml_c <= 512 else 2048
                if L == 2048 and B > 8:
                    B = 8
                    chunk = seqs[idx : idx + B]
            ids = np.zeros((B, L), dtype=np.int32)
            mask = np.zeros((B, L), dtype=np.int32)
            for i, s in enumerate(chunk):
                s_tr = s[:L]
                ids[i, :len(s_tr)] = s_tr
                mask[i, :len(s_tr)] = 1
            t_inf = time.perf_counter()
            out = await fns[(L, B)]({"input_ids": NDArray(ids), "attention_mask": NDArray(mask)})
            telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
            arr = out["embedding"].numpy()[:len(chunk)].astype(np.float32)
            norms = np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)
            vecs.append(arr / norms)
            idx += len(chunk)
        mat = np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)
        np.save(req["out_npy"], mat)
        sys.stdout.write(json.dumps({"ok": True, **telem.payload()}) + "\n")
        sys.stdout.flush()

asyncio.run(main())
"""
        client = _CoreAiSubprocessClient(
            cmd=[ca_py],
            daemon_script=daemon_code,
            script_name="daemon.py",
            script_args=[ca_bundle],
            prefix="coreai_emb2_",
        )

        def _enc_ca(txts: list[str], *, prefix: str, max_len: int) -> np.ndarray:
            seqs = []
            for t in txts:
                ids = tok_fast.encode(prefix + t, add_special_tokens=True).ids
                if len(ids) > max_len:
                    ids = ids[: max_len - 1] + [1]
                seqs.append(ids)
            npy_p = client.next_temp_path("out_{n}.npy")
            client.request({"seqs": seqs, "max_cap": max_len, "out_npy": str(npy_p)})
            return np.load(npy_p)

        return Cat1Adapter(
            harness_label="coreai-emb2-bf16",
            mrl_dims=(128, 256, 512, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_ca(txts, prefix=q_prefix, max_len=512),
            enc_d_nb=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=512),
            enc_q=lambda txts: _enc_ca(txts, prefix=q_prefix, max_len=512),
            enc_d=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=512),
            enc_coarse_2k=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=2048),
            close_fn=lambda: float(client.close().get("peak_gb", 0.0)),
            telemetry_fn=client.telemetry,
        )

    if pkg_name == "public-pplx":
        pdir = resolve_pplx_mlx_path(pplx_mlx_path)
        if pdir is not None and str(pdir) not in sys.path:
            sys.path.insert(0, str(pdir))
        try:
            import mlx.core as mx
            import pplx_context_mlx
        except ImportError as exc:
            raise FileNotFoundError(f"pplx_context_mlx not installed or checkout not found ({exc})") from exc

        snap_z = resolve_hf_snapshot("zainmerchan/pplx-embed-v2-context-0.6b-mlx")
        snap = snap_z or resolve_hf_snapshot("perplexity-ai/pplx-embed-v2-context-0.6b")
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for zainmerchan/pplx-embed-v2-context-0.6b-mlx")
        eng = pplx_context_mlx.load_model(str(snap), dtype="float16")
        lbl = "pplx-context-mlx-zainmerchan-f16" if snap_z is not None else "pplx-context-mlx-f16"

        def _raw_proj(txts: list[str], *, is_q: bool, bsz: int = 32, norm: bool = False) -> np.ndarray:
            rows: list[np.ndarray] = []
            for i in range(0, len(txts), bsz):
                batch = [[t] for t in txts[i : i + bsz]]
                v = eng.project_queries(batch) if is_q else eng.project_documents(batch)
                v = v[:, 0, :].astype(mx.float32)
                if norm:
                    v = v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                mx.eval(v)
                rows.append(np.asarray(v))
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            return np.vstack(rows) if rows else np.zeros((0, 1024), dtype=np.float32)

        ctx_max_len = 4096 if tier == "tier1" else 16384

        def _enc_ctx_pplx(doc_groups: list[list[str]]) -> list[np.ndarray]:
            safe_budget = max(512, ctx_max_len - 64)
            flat_windows: list[list[str]] = []
            win_meta: list[tuple[int, bool]] = []
            for d_idx, chunks in enumerate(doc_groups):
                if not chunks:
                    continue
                lens = [len(eng.tokenizer._tokenizer.encode(c, add_special_tokens=False)) + 2 for c in chunks]
                if sum(lens) <= safe_budget or len(chunks) == 1:
                    flat_windows.append(list(chunks))
                    win_meta.append((d_idx, False))
                else:
                    anc = chunks[0][:2048]
                    anc_l = len(eng.tokenizer._tokenizer.encode(anc, add_special_tokens=False)) + 2
                    cur = [chunks[0]]
                    cur_l = lens[0]
                    has_anc = False
                    for c_txt, c_l in zip(chunks[1:], lens[1:]):
                        if cur and cur_l + c_l > safe_budget:
                            flat_windows.append(cur)
                            win_meta.append((d_idx, has_anc))
                            cur = [anc, c_txt]
                            cur_l = anc_l + c_l
                            has_anc = True
                        else:
                            cur.append(c_txt)
                            cur_l += c_l
                    if cur:
                        flat_windows.append(cur)
                        win_meta.append((d_idx, has_anc))

            per_doc: list[list[np.ndarray]] = [[] for _ in doc_groups]
            for (d_idx, has_anc), w_chunks in zip(win_meta, flat_windows):
                v = eng.project_documents([w_chunks])[0, : len(w_chunks), :].astype(mx.float32)
                v = v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                mx.eval(v)
                arr = np.asarray(v)
                per_doc[d_idx].append(arr[1:] if has_anc else arr)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            return [np.vstack(p) if p else np.zeros((0, 1024), dtype=np.float32) for p in per_doc]

        return Cat1Adapter(
            harness_label=lbl,
            mrl_dims=(128, 256, 512, "max"),
            int8_mode="native_tanh_int8",
            nb_pre_normalized=False,
            enc_q_nb=lambda txts: _raw_proj(txts, is_q=True, norm=False),
            enc_d_nb=lambda txts: _raw_proj(txts, is_q=False, norm=False),
            enc_q=lambda txts: _raw_proj(txts, is_q=True, norm=True),
            enc_d=lambda txts: _raw_proj(txts, is_q=False, norm=True),
            enc_ctx=_enc_ctx_pplx,
        )

    raise ValueError(f"Unsupported public Cat 1 package: {pkg_name}")


class PublicLayaMlxBackend:
    """Stock `mlx-embeddings` (`modernbert` 8-bit g64) Laya-Typed backend matching `LayaAneBackend` API."""

    def __init__(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_embeddings.models import modernbert
        from tokenizers import Tokenizer
        from fast_decision_models.laya_ane import LayaAneBackend

        snap = resolve_hf_snapshot("convaiinnovations/laya-typed-decisions")
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for convaiinnovations/laya-typed-decisions")
        enc_cfg = json.loads((snap / "encoder" / "config.json").read_text(encoding="utf-8"))
        rl_cfg = json.loads((snap / "rl_agent_config.json").read_text(encoding="utf-8"))
        raw_w = mx.load(str(snap / "model.safetensors"))

        self._mx = mx
        self._nn = nn
        self.encoder = modernbert.Model(modernbert.ModelArgs.from_dict(enc_cfg))
        enc_w = {k[len("encoder.") :]: v.astype(mx.bfloat16) for k, v in raw_w.items() if k.startswith("encoder.")}
        if hasattr(self.encoder, "sanitize"):
            enc_w = self.encoder.sanitize(enc_w)
        self.encoder.load_weights(list(enc_w.items()), strict=False)
        self.encoder.set_dtype(mx.bfloat16)
        nn.quantize(self.encoder, group_size=64, bits=8)

        self.H = int(enc_cfg["hidden_size"])
        self.nh = int(enc_cfg["num_attention_heads"])
        self.hd = self.H // self.nh
        self.scale = self.hd**-0.5
        self.type_emb_w = raw_w["type_emb.weight"].astype(mx.bfloat16)
        self.head_layers = []
        for hi in range(2):
            hp = f"head.layers.{hi}."
            self.head_layers.append(
                {
                    "n1_w": raw_w[hp + "norm1.weight"].astype(mx.float32),
                    "n1_b": raw_w[hp + "norm1.bias"].astype(mx.float32),
                    "in_w": raw_w[hp + "self_attn.in_proj_weight"].astype(mx.bfloat16),
                    "in_b": raw_w[hp + "self_attn.in_proj_bias"].astype(mx.bfloat16),
                    "out_w": raw_w[hp + "self_attn.out_proj.weight"].astype(mx.bfloat16),
                    "out_b": raw_w[hp + "self_attn.out_proj.bias"].astype(mx.bfloat16),
                    "n2_w": raw_w[hp + "norm2.weight"].astype(mx.float32),
                    "n2_b": raw_w[hp + "norm2.bias"].astype(mx.float32),
                    "l1_w": raw_w[hp + "linear1.weight"].astype(mx.bfloat16),
                    "l1_b": raw_w[hp + "linear1.bias"].astype(mx.bfloat16),
                    "l2_w": raw_w[hp + "linear2.weight"].astype(mx.bfloat16),
                    "l2_b": raw_w[hp + "linear2.bias"].astype(mx.bfloat16),
                }
            )
        self.sc_n_w = raw_w["scorer.0.weight"].astype(mx.float32)
        self.sc_n_b = raw_w["scorer.0.bias"].astype(mx.float32)
        self.sc_1_w = raw_w["scorer.1.weight"].astype(mx.float32)
        self.sc_1_b = raw_w["scorer.1.bias"].astype(mx.float32)
        self.sc_3_w = raw_w["scorer.3.weight"].astype(mx.float32)
        self.sc_3_b = raw_w["scorer.3.bias"].astype(mx.float32)
        mx.eval(self.encoder.parameters(), self.type_emb_w, self.sc_n_w, self.sc_1_w, self.sc_3_w)

        tok_helper = object.__new__(LayaAneBackend)
        tok_helper.laya_dir = snap
        tok_helper.temperature = [float(x) for x in rl_cfg.get("temperature", [1.0, 1.0, 1.0])]
        tok_helper.temperature_by_options = {
            str(k): float(v) for k, v in rl_cfg.get("temperature_by_options", {}).items()
        }
        tok_helper.max_len = 512
        tok_helper.head_max_len = LayaAneBackend._resolve_head_max_len(rl_cfg, 512)
        tok_helper.tok = Tokenizer.from_file(str(snap / "tokenizer" / "tokenizer.json"))
        tok_helper.tok.no_truncation()
        tok_helper.tok.no_padding()
        tok_helper.cls_id = int(tok_helper.tok.token_to_id("[CLS]") or 50281)
        tok_helper.sep_id = int(tok_helper.tok.token_to_id("[SEP]") or 50282)
        tok_helper.pad_id = int(tok_helper.tok.token_to_id("[PAD]") or 50283)
        tok_helper.mask_id = int(tok_helper.tok.token_to_id("[MASK]") or 50284)
        tok_helper.mask_tok = "[MASK]"
        self._tok_helper = tok_helper
        self._qtype_map = {"choice": 0, "score": 1, "noul": 2}

    def encode_item(self, state: Any, qdef: Mapping[str, Any], *, on_truncate: str = "ignore") -> dict[str, Any]:
        return self._tok_helper.encode_item(state, qdef, on_truncate=on_truncate)

    def _score_generic(self, encs: Sequence[Mapping[str, Any]], qtype_override: str | None = None, bsz: int = 16) -> list[list[float]]:
        mx = self._mx
        nn = self._nn
        out_probs: list[list[float]] = []
        for i in range(0, len(encs), bsz):
            batch = encs[i : i + bsz]
            B = len(batch)
            qt_idx = self._qtype_map[qtype_override] if qtype_override else int(batch[0].get("qtype", 0))
            tv = self.type_emb_w[qt_idx].reshape(1, 1, self.H)
            max_l = max(len(e["ids"]) for e in batch)
            ids_np = np.full((B, max_l), self._tok_helper.pad_id, dtype=np.int32)
            mask_np = np.zeros((B, max_l), dtype=np.int32)
            for r_i, e in enumerate(batch):
                L = len(e["ids"])
                ids_np[r_i, :L] = e["ids"]
                mask_np[r_i, :L] = 1
            ids_mx = mx.array(ids_np)
            mask_mx = mx.array(mask_np)
            h = self.encoder(ids_mx, attention_mask=mask_mx).last_hidden_state.astype(mx.bfloat16) + tv
            mask_bf16 = mx.where(
                mask_mx[:, None, None, :] == 1,
                mx.array(0.0, dtype=mx.bfloat16),
                mx.array(-1e4, dtype=mx.bfloat16),
            )
            for hl in self.head_layers:
                hn1 = mx.fast.layer_norm(h.astype(mx.float32), hl["n1_w"], hl["n1_b"], eps=1e-5).astype(mx.bfloat16)
                qkv = (hn1 @ hl["in_w"].T + hl["in_b"]).reshape(B, max_l, 3, self.nh, self.hd)
                q = qkv[:, :, 0].transpose(0, 2, 1, 3)
                k = qkv[:, :, 1].transpose(0, 2, 1, 3)
                v = qkv[:, :, 2].transpose(0, 2, 1, 3)
                att = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask_bf16)
                att = att.transpose(0, 2, 1, 3).reshape(B, max_l, self.H)
                h = h + (att @ hl["out_w"].T + hl["out_b"])
                hn2 = mx.fast.layer_norm(h.astype(mx.float32), hl["n2_w"], hl["n2_b"], eps=1e-5).astype(mx.bfloat16)
                ff = nn.relu(hn2 @ hl["l1_w"].T + hl["l1_b"]) @ hl["l2_w"].T + hl["l2_b"]
                h = h + ff
            sn = mx.fast.layer_norm(h.astype(mx.float32), self.sc_n_w, self.sc_n_b, eps=1e-5)
            pos_l = (nn.gelu(sn @ self.sc_1_w.T + self.sc_1_b) @ self.sc_3_w.T + self.sc_3_b).squeeze(-1)
            mx.eval(pos_l)
            pos_np = np.asarray(pos_l)
            for r_i, e in enumerate(batch):
                m_pos = e["markers"]
                e_qi = self._qtype_map[qtype_override] if qtype_override else int(e.get("qtype", qt_idx))
                temp = self._tok_helper._resolve_temperature(e_qi, len(m_pos))
                raw_l = np.array([pos_np[r_i, p] for p in m_pos], dtype=np.float64) / max(temp, 0.05)
                raw_l -= np.max(raw_l)
                ex = np.exp(raw_l)
                out_probs.append((ex / np.sum(ex)).tolist())
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        return out_probs

    def score_noul_batch(self, encs: Sequence[Mapping[str, Any]]) -> list[float]:
        probs = self._score_generic(encs, qtype_override="noul")
        return [float(p[1]) if len(p) >= 2 else float(p[0]) for p in probs]

    def score_choice_batch(self, encs: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        return self._score_generic(encs, qtype_override=None)

    def close(self) -> None:
        pass


class CoreAiLayaBackend:
    """Apple `CoreAI` (`ahmedandaloes/laya-typed-decisions-CoreAI` `decision.aimodel`) backend."""

    def __init__(self, coreai_python: str | None = None) -> None:
        from tokenizers import Tokenizer
        from fast_decision_models.laya_ane import LayaAneBackend

        snap_ca = resolve_hf_snapshot("ahmedandaloes/laya-typed-decisions-CoreAI")
        ca_py = resolve_coreai_python(coreai_python)
        if snap_ca is None or ca_py is None or not (snap_ca / "decision.aimodel").exists():
            raise FileNotFoundError("Missing ahmedandaloes/laya-typed-decisions-CoreAI or coreai runtime")

        manifest_ca = json.loads((snap_ca / "manifest.json").read_text(encoding="utf-8"))
        rl_cfg = json.loads((snap_ca / "rl_agent_config.json").read_text(encoding="utf-8"))
        seq_cap = int(manifest_ca["sequence_length"])
        m_cap = int(manifest_ca["marker_capacity"])
        out_name0 = manifest_ca["outputs"][0]

        tok_helper = object.__new__(LayaAneBackend)
        tok_helper.laya_dir = snap_ca
        tok_helper.temperature = [float(x) for x in rl_cfg.get("temperature", [1.0, 1.0, 1.0])]
        tok_helper.temperature_by_options = {
            str(k): float(v) for k, v in rl_cfg.get("temperature_by_options", {}).items()
        }
        tok_helper.max_len = min(512, seq_cap)
        tok_helper.head_max_len = min(LayaAneBackend._resolve_head_max_len(rl_cfg, 512), m_cap)
        tok_helper.tok = Tokenizer.from_file(str(snap_ca / "tokenizer" / "tokenizer.json"))
        tok_helper.tok.no_truncation()
        tok_helper.tok.no_padding()
        tok_helper.cls_id = int(tok_helper.tok.token_to_id("[CLS]") or 50281)
        tok_helper.sep_id = int(tok_helper.tok.token_to_id("[SEP]") or 50282)
        tok_helper.pad_id = int(tok_helper.tok.token_to_id("[PAD]") or 50283)
        tok_helper.mask_id = int(tok_helper.tok.token_to_id("[MASK]") or 50284)
        tok_helper.mask_tok = "[MASK]"
        self._tok_helper = tok_helper
        self._qtype_map = {"choice": 0, "score": 1, "noul": 2}
        self.peak_gb = 0.0

        daemon_code = f"""
import asyncio
from coreai.runtime import AIModel, NDArray
import numpy as np

async def main():
    aim_path = Path({str(snap_ca)!r}) / "decision.aimodel"
    telem = _DaemonTelemetry([aim_path.parent])
    t_load = time.perf_counter()
    m = await AIModel.load(str(aim_path))
    telem.mark_loaded(aim_path, (time.perf_counter() - t_load) * 1000.0)
    fn = m.load_function("main")
    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        out_logits = []
        for ids, mpos, qti in zip(req["ids_list"], req["mpos_list"], req["qt_list"]):
            qt = np.asarray([int(qti)], dtype=np.int32)
            n, k = len(ids), len(mpos)
            ii = np.full((1, {seq_cap}), {tok_helper.pad_id}, dtype=np.int32); ii[0, :n] = ids
            am = np.zeros((1, {seq_cap}), dtype=np.int32); am[0, :n] = 1
            mp = np.zeros((1, {m_cap}), dtype=np.int32); mp[0, :k] = mpos
            mm = np.zeros((1, {m_cap}), dtype=bool); mm[0, :k] = True
            t_inf = time.perf_counter()
            res = await fn({{"input_ids": NDArray(ii), "attention_mask": NDArray(am), "marker_pos": NDArray(mp), "marker_mask": NDArray(mm), "qtype": NDArray(qt)}})
            telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
            lg = np.asarray(res[{out_name0!r}].numpy(), dtype=np.float64)[0, :k].tolist()
            out_logits.append(lg)
        Path(req["out_json"]).write_text(json.dumps(out_logits), encoding="utf-8")
        sys.stdout.write(json.dumps({{"ok": True, **telem.payload()}}) + "\\n")
        sys.stdout.flush()

asyncio.run(main())
"""
        self._client = _CoreAiSubprocessClient(
            cmd=[ca_py],
            daemon_script=daemon_code,
            script_name="d.py",
            prefix="coreai_laya_",
        )

    def encode_item(self, state: Any, qdef: Mapping[str, Any], *, on_truncate: str = "ignore") -> dict[str, Any]:
        return self._tok_helper.encode_item(state, qdef, on_truncate=on_truncate)

    def _score_generic(self, encs: Sequence[Mapping[str, Any]], qtype_override: str | None = None) -> list[list[float]]:
        if not encs:
            return []
        oj = self._client.next_temp_path("o_{n}.json")
        qt_list = [
            self._qtype_map[qtype_override] if qtype_override else int(e.get("qtype", 0))
            for e in encs
        ]
        self._client.request(
            {
                "qt_list": qt_list,
                "ids_list": [e["ids"] for e in encs],
                "mpos_list": [e["markers"] for e in encs],
                "out_json": str(oj),
            }
        )
        self.peak_gb = max(self.peak_gb, self._client.peak_gb)
        raw_list = json.loads(oj.read_text(encoding="utf-8"))
        probs: list[list[float]] = []
        for lg, qti in zip(raw_list, qt_list):
            temp = self._tok_helper._resolve_temperature(qti, len(lg))
            arr = np.asarray(lg, dtype=np.float64) / max(temp, 0.05)
            arr -= np.max(arr)
            ex = np.exp(arr)
            probs.append((ex / np.sum(ex)).tolist())
        return probs

    def score_noul_batch(self, encs: Sequence[Mapping[str, Any]]) -> list[float]:
        probs = self._score_generic(encs, qtype_override="noul")
        return [float(p[1]) if len(p) >= 2 else float(p[0]) for p in probs]

    def score_choice_batch(self, encs: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        return self._score_generic(encs, qtype_override=None)

    def telemetry(self) -> dict[str, Any]:
        return self._client.telemetry()

    def close(self) -> float:
        telem = self._client.close()
        self.peak_gb = max(self.peak_gb, float(telem.get("peak_gb", 0.0)))
        return self.peak_gb


class PublicKev08bMlxBackend:
    """Stock `mlx-lm` (`AbstractFramework/kev-0.8b-mlx-8bit`) backend matching `Kev08bMlxBackend` API."""

    def __init__(self, kev_dir: Path, *, batch_docs: int = 8) -> None:
        import mlx.core as mx
        import mlx_lm
        from fast_decision_models.kev_mlx import Kev08bMlxBackend

        snap_k8 = resolve_hf_snapshot("AbstractFramework/kev-0.8b-mlx-8bit")
        if snap_k8 is None:
            raise FileNotFoundError("Missing cached HF snapshot for AbstractFramework/kev-0.8b-mlx-8bit")
        dec_cfg = json.loads((snap_k8 / "decision.json").read_text(encoding="utf-8"))
        hw = mx.load(str(snap_k8 / "head.safetensors"))
        self._mx = mx
        self.q_w = hw["q.weight"].astype(mx.float32)
        self.q_b = hw["q.bias"].astype(mx.float32)
        self.k_w = hw["k.weight"].astype(mx.float32)
        self.k_b = hw["k.bias"].astype(mx.float32)
        self.temp_s = float(dec_cfg.get("temperature", 2.3510958125672174))
        self.quant_mode = "mlx_8bit"
        self.batch_docs = max(1, int(batch_docs))

        mdl, _ = mlx_lm.load(str(snap_k8))
        self.backbone = mdl.language_model.model if hasattr(mdl, "language_model") else mdl.model
        mx.eval(mdl.parameters(), self.q_w, self.q_b, self.k_w, self.k_b)
        self._fmt = Kev08bMlxBackend(kev_dir, quant_mode="w8", batch_docs=1)

    def encode_questions(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        return self._fmt.encode_questions(state, questions)

    def score_multi_states(
        self,
        pairs: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        mx = self._mx
        bsz = max(1, int(batch_docs or self.batch_docs))
        # Group exact-length sequences into unpadded batches (<= bsz) so GDN recurrent conv1d states never see pad tokens.
        out_by_doc: list[list[tuple[list[float], list[float]]]] = [[] for _ in pairs]
        by_len: dict[int, list[tuple[int, int, list[int], int, list[int]]]] = defaultdict(list)
        for d_i, (st, qlist) in enumerate(pairs):
            s_ids, q_rows = self._fmt.encode_questions(st, qlist)
            out_by_doc[d_i] = [([], []) for _ in q_rows]
            q_off = len(s_ids)
            for q_i, qr in enumerate(q_rows):
                full_ids = list(s_ids) + list(qr["ids"])
                dec_pos = q_off + int(qr.get("decide", len(qr["ids"]) - 1))
                opt_pos = [q_off + int(p) for p in qr["opts"]]
                by_len[len(full_ids)].append((d_i, q_i, full_ids, dec_pos, opt_pos))

        scale = math.sqrt(int(self.q_w.shape[0])) * self.temp_s
        for _, grp in sorted(by_len.items()):
            for i in range(0, len(grp), bsz):
                chunk = grp[i : i + bsz]
                ids_mx = mx.array([x[2] for x in chunk], dtype=mx.int32)
                lh_batch = self.backbone(ids_mx).astype(mx.float32)
                mx.eval(lh_batch)
                for b_i, (d_i, q_i, _, dec_pos, opt_pos) in enumerate(chunk):
                    lh = lh_batch[b_i]
                    qv = lh[dec_pos] @ self.q_w.T + self.q_b
                    kv = lh[mx.array(opt_pos, dtype=mx.int32)] @ self.k_w.T + self.k_b
                    raw_l = (kv @ qv) / math.sqrt(int(self.q_w.shape[0]))
                    lgs = raw_l / self.temp_s
                    pr = mx.softmax(lgs, axis=-1)
                    mx.eval(pr, raw_l)
                    out_by_doc[d_i][q_i] = (pr.tolist(), raw_l.tolist())
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        return out_by_doc

    def score_multi_states_noul(
        self,
        pairs: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
    ) -> list[list[tuple[float, list[float]]]]:
        raw = self.score_multi_states(pairs, batch_docs=batch_docs)
        return [
            [(float(pr[1]) if len(pr) >= 2 else float(pr[0]), lg) for pr, lg in doc_res]
            for doc_res in raw
        ]

    def close(self) -> None:
        self._fmt.close()


@dataclass
class PublicLlmAdapter:
    """Category 3 & 4 stock LLM generator adapter."""

    default_label: str
    generate_fn: Callable[..., tuple[list[str], list[str], list[list[int]], dict[str, Any]]]
    close_fn: Callable[[], float | None] | None = None
    telemetry_fn: Callable[[], dict[str, Any]] | None = None

    def telemetry(self) -> dict[str, Any] | None:
        return self.telemetry_fn() if self.telemetry_fn is not None else None

    def close(self) -> float | None:
        if self.close_fn is not None:
            return self.close_fn()
        return None


def build_public_llm_adapter(
    pkg_name: str,
    *,
    batch_size: int = 16,
    host_mem_gb: float = 48.0,
    coreai_python: str | None = None,
    coreai_variant: str = "4bit",
) -> PublicLlmAdapter:
    """Instantiate the fastest verified stock LLM baseline for Categories 3 & 4."""
    if pkg_name == "public-gemma4":
        import mlx.core as mx
        import mlx_vlm

        snap = resolve_hf_snapshot("lmstudio-community/gemma-4-E2B-it-MLX-4bit")
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for lmstudio-community/gemma-4-E2B-it-MLX-4bit")
        model, processor = mlx_vlm.load(str(snap))
        tok = getattr(processor, "tokenizer", processor)
        eff_bsz = min(max(1, int(batch_size)), 32 if 0 < host_mem_gb <= 28.5 else 64)

        def _gen_g4(
            prompts: list[str],
            *,
            system_prompt: str | None = None,
            per_doc_schemas: list[dict[str, Any] | None] | None = None,
            max_tokens: int = 256,
            reasoning_effort: str | None = None,
        ) -> tuple[list[str], list[str], list[list[int]], dict[str, Any]]:
            del per_doc_schemas, reasoning_effort
            fmts: list[str] = []
            for p in prompts:
                msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": p}
                ]
                try:
                    fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                except TypeError:
                    fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            texts: list[str] = []
            reasons: list[str] = []
            ids_list: list[list[int]] = []
            tot_comp = 0
            for i in range(0, len(fmts), eff_bsz):
                sub = fmts[i : i + eff_bsz]
                if len(sub) > 1 and hasattr(mlx_vlm, "batch_generate"):
                    rb = mlx_vlm.batch_generate(model, processor, prompts=sub, max_tokens=max_tokens, verbose=False)
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                    sub_txts = list(getattr(rb, "texts", []) or [])
                else:
                    sub_txts = [
                        getattr(
                            mlx_vlm.generate(model, processor, sp, max_tokens=max_tokens, temp=0.0, verbose=False),
                            "text",
                            "",
                        )
                        for sp in sub
                    ]
                    if hasattr(mx, "synchronize"):
                        mx.synchronize()
                for tx in sub_txts:
                    tids = list(tok.encode(tx, add_special_tokens=False))
                    texts.append(tx)
                    ids_list.append(tids)
                    tot_comp += len(tids)
                    reasons.append("length" if len(tids) >= max_tokens - 1 else "stop")
            return texts, reasons, ids_list, {"completion_tokens": tot_comp, "mask_fallbacks": 0}

        return PublicLlmAdapter(default_label="mlx_vlm_4bit", generate_fn=_gen_g4)

    if pkg_name == "public-gpt-oss":
        import mlx.core as mx
        import mlx_lm

        snap = resolve_hf_snapshot("mlx-community/gpt-oss-20b-MXFP4-Q4")
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for mlx-community/gpt-oss-20b-MXFP4-Q4")
        model, tok = mlx_lm.load(str(snap))
        eff_bsz = min(max(1, int(batch_size)), 16)

        def _gen_oss(
            prompts: list[str],
            *,
            system_prompt: str | None = None,
            per_doc_schemas: list[dict[str, Any] | None] | None = None,
            max_tokens: int = 256,
            reasoning_effort: str | None = None,
        ) -> tuple[list[str], list[str], list[list[int]], dict[str, Any]]:
            del per_doc_schemas
            eff_re = "low" if reasoning_effort in (None, "skip") else reasoning_effort
            eff_max = max(64, int(max_tokens)) if int(max_tokens) <= 16 else int(max_tokens)
            texts: list[str] = []
            reasons: list[str] = []
            ids_list: list[list[int]] = []
            tot_comp = 0
            for i in range(0, len(prompts), eff_bsz):
                sub_p = prompts[i : i + eff_bsz]
                p_toks = []
                for p in sub_p:
                    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                        {"role": "user", "content": p}
                    ]
                    try:
                        raw = tok.apply_chat_template(
                            msgs, tokenize=True, add_generation_prompt=True, reasoning_effort=eff_re
                        )
                    except TypeError:
                        raw = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                    p_toks.append(list(raw if isinstance(raw, list) else getattr(raw, "input_ids", raw)))
                rb = mlx_lm.batch_generate(model, tok, prompts=p_toks, max_tokens=eff_max, verbose=False)
                if hasattr(mx, "synchronize"):
                    mx.synchronize()
                for tx in rb.texts:
                    clean_tx = (
                        tx.split("<|message|>")[-1].replace("<|return|>", "").strip()
                        if "<|message|>" in tx
                        else tx
                    )
                    tids = list(tok.encode(clean_tx, add_special_tokens=False))
                    texts.append(clean_tx)
                    ids_list.append(tids)
                    tot_comp += len(tids)
                    reasons.append("length" if len(tids) >= eff_max - 1 else "stop")
            return texts, reasons, ids_list, {"completion_tokens": tot_comp, "mask_fallbacks": 0}

        return PublicLlmAdapter(default_label="mlx_lm_mxfp4_q4", generate_fn=_gen_oss)

    if pkg_name == "coreai-qwen3.5-4b":
        ca_py = resolve_coreai_python(coreai_python)
        ca_bdir = resolve_coreai_bundle(f"qwen35/qwen35_4b_{coreai_variant}")
        tok_snap = (
            (ca_bdir / "tokenizer") if (ca_bdir is not None and (ca_bdir / "tokenizer").is_dir()) else None
        ) or resolve_hf_snapshot("Qwen/Qwen3.5-4B") or ca_bdir
        if ca_py is None or ca_bdir is None or tok_snap is None or not list(ca_bdir.glob("*.aimodel")):
            raise FileNotFoundError(f"CoreAI Qwen3.5-4B ({coreai_variant}) bundle or coreai runtime not staged")

        llm_run_cand = ca_py.parents[2] / ".build" / "out" / "Products" / "Release" / "llm-runner"
        llm_run_str = str(llm_run_cand) if llm_run_cand.is_file() else ""

        daemon_q = f"""
import asyncio, re, subprocess
import ml_dtypes, numpy as np
from transformers import AutoTokenizer
from coreai.runtime import AIModel, NDArray

def make_state(fn, kv_len):
    st = {{}}
    for sn in fn.desc.state_names:
        sd = fn.desc.state_descriptor(sn)
        shape = tuple(kv_len if int(d) < 0 else int(d) for d in sd.shape)
        ds = str(sd.dtype).lower()
        dt = ml_dtypes.bfloat16 if "bfloat16" in ds else (ml_dtypes.float8_e4m3fn if ("float8" in ds or "e4m3" in ds) else (np.float16 if "float16" in ds else np.float32))
        st[sn] = NDArray(np.zeros(shape, dtype=dt))
    return st

def parse_runner_out(stdout_s: str) -> tuple[str, int]:
    txt = stdout_s.split("Generating...\\n", 1)[1] if "Generating...\\n" in stdout_s else stdout_s
    summary = ""
    if "Performance Summary:" in txt:
        txt, summary = txt.split("Performance Summary:", 1)
        txt = re.sub(r"\\n+[^\\n]*$", "", txt).rstrip("\\n")
    m = re.search(r"Generation:\\s*[0-9.]+ms,\\s*(\\d+)\\s*tokens", summary)
    return txt, (int(m.group(1)) if m else 0)

async def main():
    bdir = Path({str(ca_bdir)!r})
    llm_run = {llm_run_str!r}
    tok = AutoTokenizer.from_pretrained({str(tok_snap)!r}, local_files_only=True)
    raw_eos = tok.eos_token_id
    eos_set = set([int(x) for x in ([raw_eos] if isinstance(raw_eos, int) else (raw_eos or []))])
    telem = _DaemonTelemetry([bdir])
    aim_path = next(bdir.glob("*.aimodel"), None)
    fn = None
    if not llm_run and aim_path is not None:
        t_load = time.perf_counter()
        model = await AIModel.load(str(aim_path))
        telem.mark_loaded(aim_path, (time.perf_counter() - t_load) * 1000.0)
        fn = model.load_function("main")
    else:
        telem.mark_loaded(aim_path, None)
    tok_f = Path(__file__).resolve().parent / "in_toks.json"
    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        msgs = req["messages"]
        try:
            raw = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            raw = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        p_ids = [int(x) for x in (raw if isinstance(raw, list) else getattr(raw, "input_ids", raw))]
        max_new = int(req["max_tokens"])
        t_req = time.perf_counter()
        if llm_run:
            tok_f.write_text(json.dumps({{"tokens": p_ids}}), encoding="utf-8")
            proc_r = subprocess.run(
                ["/usr/bin/time", "-l", llm_run, "--model", str(bdir), "--raw-tokens", str(tok_f), "--temperature", "0.0", "--max-tokens", str(max_new), "--warmup", "off"],
                capture_output=True, text=True, timeout=180,
            )
            telem.record_infer((time.perf_counter() - t_req) * 1000.0)
            m_fp = re.search(r"(\\d+)\\s+peak memory footprint", proc_r.stderr)
            if m_fp:
                telem.observe_external_peak_gb(round(int(m_fp.group(1)) / 1e9, 4))
            if proc_r.returncode != 0:
                sys.stdout.write(json.dumps({{"gen_ids": [], "text": "", "reason": "error", **telem.payload()}}) + "\\n")
                sys.stdout.flush()
                continue
            txt, gtok = parse_runner_out(proc_r.stdout)
            gen_ids = list(tok.encode(txt, add_special_tokens=False))
            reason = "length" if (gtok or len(gen_ids)) >= max_new else "stop"
            sys.stdout.write(json.dumps({{"gen_ids": gen_ids, "text": txt, "reason": reason, **telem.payload()}}) + "\\n")
            sys.stdout.flush()
            continue
        kv_len = max(512, len(p_ids) + max_new + 32)
        st = make_state(fn, kv_len)
        pa = np.array([p_ids], dtype=np.int32)
        pp = np.arange(len(p_ids), dtype=np.int32).reshape(1, len(p_ids))
        out = await fn({{"input_ids": NDArray(pa), "position_ids": NDArray(pp)}}, state=st)
        last_l = out["logits"].numpy()[0, -1, :].astype(np.float32)
        gen_ids = []
        cur_len = len(p_ids)
        reason = "length"
        for _ in range(max_new):
            if not np.all(np.isfinite(last_l)):
                reason = "nan"
                break
            nxt = int(np.argmax(last_l))
            if nxt in eos_set:
                reason = "stop"
                break
            gen_ids.append(nxt)
            cur_len += 1
            ta = np.array([[nxt]], dtype=np.int32)
            tp = np.array([[cur_len - 1]], dtype=np.int32)
            out = await fn({{"input_ids": NDArray(ta), "position_ids": NDArray(tp)}}, state=st)
            last_l = out["logits"].numpy()[0, -1, :].astype(np.float32)
        telem.record_infer((time.perf_counter() - t_req) * 1000.0)
        txt = tok.decode(gen_ids, skip_special_tokens=True)
        sys.stdout.write(json.dumps({{"gen_ids": gen_ids, "text": txt, "reason": reason, **telem.payload()}}) + "\\n")
        sys.stdout.flush()

asyncio.run(main())
"""
        client_q = _CoreAiSubprocessClient(
            cmd=[ca_py],
            daemon_script=daemon_q,
            script_name="dq.py",
            prefix="coreai_qwen35_",
        )

        def _gen_qwen_ca(
            prompts: list[str],
            *,
            system_prompt: str | None = None,
            per_doc_schemas: list[dict[str, Any] | None] | None = None,
            max_tokens: int = 256,
            reasoning_effort: str | None = None,
        ) -> tuple[list[str], list[str], list[list[int]], dict[str, Any]]:
            del per_doc_schemas, reasoning_effort
            texts: list[str] = []
            reasons: list[str] = []
            ids_list: list[list[int]] = []
            tot_comp = 0
            for p in prompts:
                msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": p}
                ]
                resp = client_q.request({"messages": msgs, "max_tokens": int(max_tokens)})
                g_ids = list(resp["gen_ids"])
                texts.append(str(resp["text"]))
                reasons.append(str(resp["reason"]))
                ids_list.append(g_ids)
                tot_comp += len(g_ids)
            return texts, reasons, ids_list, {"completion_tokens": tot_comp, "mask_fallbacks": 0}

        return PublicLlmAdapter(
            default_label=f"coreai_qwen35_4b_{coreai_variant}",
            generate_fn=_gen_qwen_ca,
            close_fn=lambda: float(client_q.close().get("peak_gb", 0.0)),
            telemetry_fn=client_q.telemetry,
        )

    raise ValueError(f"Unsupported public LLM package: {pkg_name}")
