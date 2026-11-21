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
from functools import partial
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from public_baselines import (
    COREML_REPO_IDS,
    resolve_coreai_bundle,
    resolve_coreai_python,
    resolve_coreml_compiled_model,
    resolve_hf_snapshot,
    resolve_pplx_mlx_path,
)

_FAST_SHARED_SRC = Path(__file__).resolve().parents[1] / "packages" / "fast-shared" / "src"
_FAST_DECISION_MODELS_SRC = Path(__file__).resolve().parents[1] / "packages" / "fast-decision-models" / "src"
_SCRIPTS_DIR = Path(__file__).resolve().parent


def _ensure_fast_decision_models_on_path() -> None:
    src = str(_FAST_DECISION_MODELS_SRC)
    if _FAST_DECISION_MODELS_SRC.is_dir() and src not in sys.path:
        sys.path.insert(0, src)

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
        try:
            from coreai.authoring import AIModelAsset

            asset = AIModelAsset.load(str(p))
            asset_valid = bool(asset.is_valid(p)) if hasattr(asset, "is_valid") else True
        except (ImportError, AttributeError):
            from coreai.runtime import AIModelAsset

            asset = AIModelAsset(str(p))
            asset_valid = True
        raw = asset.summary(include_statistics=True)
        out = {"asset_valid": asset_valid, "path": str(p)}
        for k in ("storage_types", "compute_types", "operation_distribution"):
            val = raw.get(k) if isinstance(raw, dict) else getattr(raw, k, None)
            if val is not None:
                out[k] = dict(val) if isinstance(val, dict) else list(val)
        return out
    except Exception:
        return None


class _DaemonTelemetry:
    def __init__(self, model_dirs=None):
        self.model_dirs = [str(Path(d)) for d in (model_dirs or []) if d]
        self.compile_sec = 0.0
        self.cold_load_ms = None
        self.cold_warmup_ms = None
        self.cold_first_infer_ms = None
        self.warm_latencies_ms = []
        self.external_peak_gb = 0.0
        self.asset_summary = None
        self.coreml_info = None
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

    def mark_compiled(self, compile_sec):
        if compile_sec is not None and float(compile_sec) > 0.0:
            self.compile_sec = round(self.compile_sec + float(compile_sec), 4)

    def mark_warmed(self, warmup_ms):
        if warmup_ms is not None:
            self.cold_warmup_ms = round(float(warmup_ms), 3)

    def set_coreml_info(self, info):
        if isinstance(info, dict):
            self.coreml_info = dict(info)

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
        if self.cold_warmup_ms is not None:
            lat = dict(lat)
            lat["cold_warmup_ms"] = self.cold_warmup_ms
        max_phys_gb = float(fb_for_json.get("lifetime_max_phys_footprint_mb", 0.0)) * 1048576.0 / 1e9
        comb_gb = float(fb_for_json.get("combined_with_model_mapped_gb", 0.0))
        pk = round(max(max_phys_gb, comb_gb, self.external_peak_gb), 4)
        out = {
            "peak_gb": pk,
            "four_bucket_metrics": fb_for_json,
            "cold_warm_latency": lat,
            "profiler_available": bool(self.profiler_available),
        }
        if self.external_peak_gb > 0.0:
            out["external_peak_gb"] = round(self.external_peak_gb, 4)
        if self.compile_sec > 0.0 or self.coreml_info is not None:
            out["compile_sec"] = round(self.compile_sec, 4)
        if self.coreml_info is not None:
            out["coreml_info"] = self.coreml_info
        if self.asset_summary is not None and (final or not self._sent_asset):
            out["asset_summary"] = self.asset_summary
            self._sent_asset = True
        return out
""".lstrip()


def _build_coreai_daemon_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment ensuring `packages/fast-shared/src` and `scripts` are on `PYTHONPATH`."""
    env = dict(os.environ if base_env is None else base_env)
    cur = env.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p] if cur else []
    for extra_p in (str(_SCRIPTS_DIR), str(_FAST_SHARED_SRC)):
        if extra_p not in parts:
            parts.insert(0, extra_p)
    env["PYTHONPATH"] = os.pathsep.join(parts)
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
            if hasattr(self._proc, "wait"):
                try:
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._tmp_obj.cleanup()
            raise RuntimeError(f"CoreAI daemon ({script_name}) failed to start: {ready}")
        self._closed = False
        self._close_timeout = 10.0
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
        if "external_peak_gb" in resp and resp["external_peak_gb"] is not None:
            self._telemetry["external_peak_gb"] = round(
                max(float(self._telemetry.get("external_peak_gb", 0.0)), float(resp["external_peak_gb"])), 4
            )
        if "compile_sec" in resp and resp["compile_sec"] is not None:
            self._telemetry["compile_sec"] = round(float(resp["compile_sec"]), 4)
        for key in ("four_bucket_metrics", "cold_warm_latency", "asset_summary", "coreml_info"):
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
        quit_msg = json.dumps({"cmd": "quit"}) + "\n"
        try:
            try:
                if hasattr(self._proc, "communicate"):
                    out_s, _ = self._proc.communicate(input=quit_msg, timeout=self._close_timeout)
                    for line in reversed((out_s or "").splitlines()):
                        line_s = line.strip()
                        if line_s:
                            self._merge_response(json.loads(line_s))
                            break
                else:
                    if getattr(self._proc, "stdin", None) and getattr(self._proc, "stdout", None):
                        self._proc.stdin.write(quit_msg)
                        self._proc.stdin.flush()
                        raw_line = self._proc.stdout.readline()
                        if raw_line:
                            self._merge_response(json.loads(raw_line))
                    if hasattr(self._proc, "wait"):
                        self._proc.wait(timeout=self._close_timeout)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                if hasattr(self._proc, "kill"):
                    try:
                        self._proc.kill()
                    except OSError:
                        pass
                if hasattr(self._proc, "wait"):
                    try:
                        self._proc.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        finally:
            for pipe_name in ("stdin", "stdout", "stderr"):
                pipe = getattr(self._proc, pipe_name, None)
                if pipe is not None and hasattr(pipe, "close"):
                    try:
                        pipe.close()
                    except OSError:
                        pass
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
    encode_token_ids: Callable[[list[list[int]], bool], tuple[np.ndarray, float]] | None = None
    coarse_max_tokens: int | None = None

    def telemetry(self) -> dict[str, Any] | None:
        return self.telemetry_fn() if self.telemetry_fn is not None else None

    def close(self) -> float | None:
        if self.close_fn is not None:
            return self.close_fn()
        return None


def _encode_coreml_token_ids(
    client: _CoreAiSubprocessClient, seqs: list[list[int]], use_pack_256: bool
) -> tuple[np.ndarray, float]:
    """Return the embeddings and synchronous encode time inside the model process."""
    path = client.next_temp_path("out_{n}.npy")
    response = client.request({"seqs": seqs, "use_pack_256": use_pack_256, "out_npy": str(path)})
    try:
        return np.load(path), float(response["encode_sec"])
    finally:
        path.unlink(missing_ok=True)


def pack_emb2_sequences_ffd(
    seq_lens: Sequence[int],
    *,
    max_tokens: int = 256,
    max_slots: int = 8,
    single_fallback_max: int = 128,
) -> tuple[list[list[int]], list[int]]:
    """First-Fit Decreasing packer for EmbeddingGemma-2 `pack_256` (`<=max_tokens`, `<=max_slots`).

    Returns `(packed_bins, single_indices)` where each bin holds original item indices whose lengths
    sum to `<= max_tokens` with at most `max_slots` items. Any bin ending with 1 item of length
    `<= single_fallback_max` is moved to `single_indices` so smaller `embed_S` buckets handle it.
    """
    short_pairs: list[tuple[int, int]] = []
    singles: list[int] = []
    for idx, ln in enumerate(seq_lens):
        n = int(ln)
        if 0 < n <= max_tokens:
            short_pairs.append((n, idx))
        else:
            singles.append(idx)
    short_pairs.sort(key=lambda x: (-x[0], x[1]))

    bins: list[list[int]] = []
    bin_used: list[int] = []
    for ln, idx in short_pairs:
        placed = False
        for bi, cur in enumerate(bins):
            if len(cur) < max_slots and (bin_used[bi] + ln) <= max_tokens:
                cur.append(idx)
                bin_used[bi] += ln
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_used.append(ln)

    packed: list[list[int]] = []
    for cur, used in zip(bins, bin_used):
        if len(cur) == 1 and used <= single_fallback_max:
            singles.append(cur[0])
        else:
            packed.append(cur)
    return packed, singles


def kev_lane_starts(lengths: Sequence[int], *, lane: int = 128) -> list[int]:
    """Compute packed start offsets matching `KevFastManager.laneStarts` (no branch crosses a lane)."""
    starts: list[int] = []
    start = 0
    for raw_n in lengths:
        n = int(raw_n)
        if n > 0 and (start // lane) != ((start + n - 1) // lane):
            start = ((start // lane) + 1) * lane
        starts.append(start)
        start += n
    return starts


def kev_pack_groups(
    lengths: Sequence[int],
    *,
    packed_len: int = 192,
    lane: int = 128,
    max_readouts: int = 16,
) -> list[list[int]]:
    """Greedy in-order grouping of question indices fitting `KevFused` (`packed_len`, `max_readouts`)."""
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in range(len(lengths)):
        cand = [*current, idx]
        starts = kev_lane_starts([lengths[i] for i in cand], lane=lane)
        extent = (starts[-1] + int(lengths[idx])) if starts else 0
        if current and (extent > packed_len or len(cand) > max_readouts):
            groups.append(current)
            current = [idx]
        else:
            current = cand
    if current:
        groups.append(current)
    return groups


def encode_kev_questions_pure(
    tok: Any,
    special_ids: Sequence[int],
    state: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    max_state: int = 1024,
    max_branch: int = 2048,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Thin backwards-compatible wrapper over `fast_decision_models.kev_mlx.encode_kev_questions`."""
    _ensure_fast_decision_models_on_path()
    from fast_decision_models.kev_mlx import encode_kev_questions

    return encode_kev_questions(
        tok,
        state,
        questions,
        special_ids=special_ids,
        max_state_tokens=max_state,
        max_branch_tokens=max_branch,
    )


def _build_coreml_emb2_adapter(variant: str, *, max_coarse_tokens: int | None = None) -> Cat1Adapter:
    """Build DRY Core ML ANE EmbeddingGemma-2 adapter (`fp16` FluidInference or `int8` OpenInfinity)."""
    from tokenizers import Tokenizer

    if variant == "fp16":
        repo_id = COREML_REPO_IDS["coreml-embeddinggemma-2-fp16"]
        rel_model = "EmbeddingGemma2Text.mlpackage"
        tok_rel = "tokenizer.json"
        table_rel = "embeddings.bf16"
        scale_rel = ""
        table_mode = "bf16"
        harness_label = "coreml-emb2-fp16-ane"
    elif variant == "int8":
        repo_id = COREML_REPO_IDS["coreml-embeddinggemma-2-int8"]
        rel_model = "text/EmbeddingGemma2Text.mlmodelc"
        tok_rel = "text/tokenizer.json"
        table_rel = "text/embeddings.int8"
        scale_rel = "text/embeddings.int8.scale.f16"
        table_mode = "int8"
        harness_label = "coreml-emb2-int8-ane"
    else:
        raise ValueError(f"Unsupported EmbeddingGemma-2 CoreML variant: {variant!r}")

    snap = resolve_hf_snapshot(repo_id)
    if snap is None or not (snap / tok_rel).is_file() or not (snap / table_rel).is_file():
        raise FileNotFoundError(f"Missing cached HF CoreML snapshot for {repo_id}")

    tok_fast = Tokenizer.from_file(str(snap / tok_rel))
    tok_fast.no_truncation()
    tok_fast.no_padding()
    q_prefix = "task: search result | query: "
    d_prefix = "title: none | text: "

    daemon_code = f"""
import math
import coremltools as ct
import numpy as np
from public_baselines import resolve_coreml_compiled_model, resolve_hf_snapshot
from _outcome_public_backends import pack_emb2_sequences_ffd

_SQRT_512 = float(math.sqrt(512.0))
_LENGTHS = (32, 48, 64, 128, 256, 512)

def main():
    repo_id = {repo_id!r}
    rel_model = {rel_model!r}
    table_mode = {table_mode!r}
    snap = resolve_hf_snapshot(repo_id)
    mc_path, compile_s = resolve_coreml_compiled_model(repo_id, rel_model)
    model_dirs = [snap.parents[1], snap.parents[1] / "blobs", mc_path.parent]
    telem = _DaemonTelemetry(model_dirs)
    telem.mark_compiled(compile_s)
    telem.set_coreml_info({{
        "repo_id": repo_id,
        "variant": {variant!r},
        "compute_units": "CPU_AND_NE",
        "max_seq_len": 512,
        "functions": [*(f"embed_{{s}}" for s in _LENGTHS), "pack_256"],
        "compiled_path": str(mc_path),
    }})

    if table_mode == "bf16":
        raw_u16 = np.memmap(str(snap / {table_rel!r}), dtype=np.uint16, mode="r").reshape(262144, 512)
        def lookup(ids_arr):
            u32 = raw_u16[ids_arr].astype(np.uint32) << np.uint32(16)
            return (u32.view(np.float32) * _SQRT_512).astype(np.float16)
    else:
        raw_i8 = np.memmap(str(snap / {table_rel!r}), dtype=np.int8, mode="r").reshape(262144, 512)
        scales = np.fromfile(str(snap / {scale_rel!r}), dtype=np.float16).astype(np.float32) * _SQRT_512
        def lookup(ids_arr):
            return (raw_i8[ids_arr].astype(np.float32) * scales[ids_arr, None]).astype(np.float16)

    pad_row = lookup(np.array([0], dtype=np.int32))[0]

    t_load = time.perf_counter()
    embed_models = {{
        s: ct.models.CompiledMLModel(str(mc_path), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"embed_{{s}}")
        for s in _LENGTHS
    }}
    pack_model = ct.models.CompiledMLModel(str(mc_path), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="pack_256")
    telem.mark_loaded(None, (time.perf_counter() - t_load) * 1000.0)

    # Pre-warm every function before READY so one-time ANE specialization never contaminates warm EOT.
    t_warm = time.perf_counter()
    for s in _LENGTHS:
        wx = np.broadcast_to(pad_row, (1, s, 512)).copy()
        wm = np.zeros((1, s), dtype=np.float16)
        wm[0, 0] = np.float16(1.0)
        t_first = time.perf_counter()
        _ = embed_models[s].predict({{"inputs_embeds": wx, "attention_mask": wm}})
        if telem.cold_first_infer_ms is None:
            telem.record_infer((time.perf_counter() - t_first) * 1000.0)
    wb = np.full((1, 1, 256, 256), -10000.0, dtype=np.float16)
    np.fill_diagonal(wb[0, 0], np.float16(0.0))
    wp = np.zeros((256, 1), dtype=np.float16)
    wpl = np.zeros((8, 256), dtype=np.float16)
    wpl[0, 0] = np.float16(1.0)
    _ = pack_model.predict({{
        "inputs_embeds": np.broadcast_to(pad_row, (1, 256, 512)).copy(),
        "attention_bias": wb,
        "positions": wp,
        "pool": wpl,
    }})
    telem.mark_warmed((time.perf_counter() - t_warm) * 1000.0)

    x_bufs = {{s: np.empty((1, s, 512), dtype=np.float16) for s in _LENGTHS}}
    m_bufs = {{s: np.zeros((1, s), dtype=np.float16) for s in _LENGTHS}}
    pk_x = np.empty((1, 256, 512), dtype=np.float16)
    pk_bias = np.empty((1, 1, 256, 256), dtype=np.float16)
    pk_pos = np.empty((256, 1), dtype=np.float16)
    pk_pool = np.empty((8, 256), dtype=np.float16)

    def _run_single(ids):
        n = min(len(ids), 512)
        s = next((b for b in _LENGTHS if n <= b), 512)
        xb = x_bufs[s]
        mb = m_bufs[s]
        xb[0, :n] = lookup(np.asarray(ids[:n], dtype=np.int32))
        if n < s:
            xb[0, n:] = pad_row
        mb.fill(np.float16(0.0))
        mb[0, :n] = np.float16(1.0)
        t_inf = time.perf_counter()
        out = embed_models[s].predict({{"inputs_embeds": xb, "attention_mask": mb}})
        telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
        vec = np.asarray(out["embedding"], dtype=np.float32).reshape(-1)[:768]
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def _run_pack(seq_list):
        pk_bias.fill(np.float16(-10000.0))
        pk_pos.fill(np.float16(0.0))
        pk_pool.fill(np.float16(0.0))
        off = 0
        for slot, ids in enumerate(seq_list):
            n = len(ids)
            pk_x[0, off : off + n] = lookup(np.asarray(ids, dtype=np.int32))
            pk_bias[0, 0, off : off + n, off : off + n] = np.float16(0.0)
            pk_pos[off : off + n, 0] = np.arange(n, dtype=np.float16)
            pk_pool[slot, off : off + n] = np.float16(1.0 / float(n))
            off += n
        if off < 256:
            pk_x[0, off:] = pad_row
            pad_idx = np.arange(off, 256)
            pk_bias[0, 0, pad_idx, pad_idx] = np.float16(0.0)
        t_inf = time.perf_counter()
        out = pack_model.predict({{
            "inputs_embeds": pk_x,
            "attention_bias": pk_bias,
            "positions": pk_pos,
            "pool": pk_pool,
        }})
        telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
        mat = np.asarray(out["embedding"], dtype=np.float32).reshape(-1, 768)[: len(seq_list)]
        norms = np.maximum(np.linalg.norm(mat, axis=-1, keepdims=True), 1e-12)
        return mat / norms

    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        seqs = req["seqs"]
        use_pack = bool(req.get("use_pack_256", True))
        n_items = len(seqs)
        t_batch = time.perf_counter()
        out_mat = np.zeros((n_items, 768), dtype=np.float32)
        if use_pack and n_items > 1:
            packed_bins, singles = pack_emb2_sequences_ffd([len(s) for s in seqs])
            for b_idxs in packed_bins:
                vecs = _run_pack([seqs[i] for i in b_idxs])
                for row_i, orig_i in enumerate(b_idxs):
                    out_mat[orig_i] = vecs[row_i]
            for orig_i in singles:
                out_mat[orig_i] = _run_single(seqs[orig_i])
        else:
            for orig_i, ids in enumerate(seqs):
                out_mat[orig_i] = _run_single(ids)
        encode_sec = time.perf_counter() - t_batch
        np.save(req["out_npy"], out_mat)
        sys.stdout.write(json.dumps({{"ok": True, "encode_sec": encode_sec}}) + "\\n")
        sys.stdout.flush()

main()
"""
    client = _CoreAiSubprocessClient(
        cmd=[sys.executable],
        daemon_script=daemon_code,
        script_name=f"coreml_emb2_{variant}.py",
        prefix=f"coreml_emb2_{variant}_",
    )

    encode_ids = partial(_encode_coreml_token_ids, client)

    def _enc_emb2(txts: list[str], *, prefix: str, max_len: int = 512) -> np.ndarray:
        if not txts:
            return np.zeros((0, 768), dtype=np.float32)
        cap = min(512, int(max_len))
        seqs: list[list[int]] = []
        for t in txts:
            ids = list(tok_fast.encode(prefix + t, add_special_tokens=True).ids)
            if not ids or ids[0] != 2:
                ids = [2] + ids
            if ids[-1] != 1:
                ids.append(1)
            if len(ids) > cap:
                ids = ids[: cap - 1] + [1]
            seqs.append(ids)
        return encode_ids(seqs, True)[0]

    coarse_cap = min(512, int(max_coarse_tokens)) if max_coarse_tokens is not None else 512
    return Cat1Adapter(
        harness_label=harness_label,
        mrl_dims=(128, 256, 512, "max"),
        int8_mode="sym_int8",
        nb_pre_normalized=True,
        enc_q_nb=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512),
        enc_d_nb=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512),
        enc_q=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512),
        enc_d=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512),
        enc_coarse_2k=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=coarse_cap),
        close_fn=lambda: float(client.close().get("peak_gb", 0.0)),
        telemetry_fn=client.telemetry,
        encode_token_ids=encode_ids,
        coarse_max_tokens=coarse_cap,
    )


def build_public_cat1_adapter(
    pkg_name: str,
    *,
    tier: str,
    coreai_python: str | None = None,
    pplx_mlx_path: str | None = None,
    max_coarse_tokens: int | None = None,
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

        coarse_cap = int(max_coarse_tokens) if max_coarse_tokens is not None else 2048
        return Cat1Adapter(
            harness_label=lbl,
            mrl_dims=(128, 256, 512, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512, bsz=32),
            enc_d_nb=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512, bsz=32),
            enc_q=lambda txts: _enc_emb2(txts, prefix=q_prefix, max_len=512, bsz=32),
            enc_d=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=512, bsz=32),
            enc_coarse_2k=lambda txts: _enc_emb2(txts, prefix=d_prefix, max_len=coarse_cap, bsz=8),
            coarse_max_tokens=coarse_cap,
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

        coarse_cap = min(2048, int(max_coarse_tokens)) if max_coarse_tokens is not None else 2048
        return Cat1Adapter(
            harness_label="coreai-emb2-bf16",
            mrl_dims=(128, 256, 512, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_ca(txts, prefix=q_prefix, max_len=512),
            enc_d_nb=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=512),
            enc_q=lambda txts: _enc_ca(txts, prefix=q_prefix, max_len=512),
            enc_d=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=512),
            enc_coarse_2k=lambda txts: _enc_ca(txts, prefix=d_prefix, max_len=coarse_cap),
            close_fn=lambda: float(client.close().get("peak_gb", 0.0)),
            telemetry_fn=client.telemetry,
            coarse_max_tokens=coarse_cap,
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
            from fast_shared.outcome_eval.retrieval import collect_pplx_window_vectors, plan_pplx_windows

            def count_tokens(text: str) -> int:
                return len(eng.tokenizer._tokenizer.encode(text, add_special_tokens=False))

            windows = plan_pplx_windows(doc_groups, ctx_max_len, count_tokens)

            def project_window(chunks: list[str]) -> np.ndarray:
                v = eng.project_documents([chunks])[0, :len(chunks), :].astype(mx.float32)
                v = v / mx.maximum(mx.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
                mx.eval(v)
                return np.asarray(v)

            result = collect_pplx_window_vectors(windows, (project_window(w.chunks) for w in windows), len(doc_groups))
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            return result

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

    if pkg_name == "coreml-bge-small":
        from tokenizers import Tokenizer

        repo_id = COREML_REPO_IDS["coreml-bge-small"]
        snap = resolve_hf_snapshot(repo_id)
        if snap is None or not (snap / "fp16" / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Missing cached HF CoreML snapshot for {repo_id}")

        tok_fast = Tokenizer.from_file(str(snap / "fp16" / "tokenizer.json"))
        tok_fast.no_truncation()
        tok_fast.no_padding()
        cls_id = int(tok_fast.token_to_id("[CLS]") or 101)
        sep_id = int(tok_fast.token_to_id("[SEP]") or 102)
        q_prefix = "Represent this sentence for searching relevant passages: "

        daemon_bge = f"""
import coremltools as ct
import numpy as np
from public_baselines import resolve_coreml_compiled_model, resolve_hf_snapshot

_BUCKETS = (128, 256, 512)

def main():
    repo_id = {repo_id!r}
    snap = resolve_hf_snapshot(repo_id)
    mc_paths = {{}}
    tot_compile = 0.0
    for s in _BUCKETS:
        p, c_s = resolve_coreml_compiled_model(repo_id, f"fp16/model_{{s}}.mlmodelc")
        mc_paths[s] = p
        tot_compile += c_s
    telem = _DaemonTelemetry([snap.parents[1], snap.parents[1] / "blobs", mc_paths[128].parent])
    telem.mark_compiled(tot_compile)
    telem.set_coreml_info({{
        "repo_id": repo_id,
        "compute_units": "CPU_AND_NE",
        "buckets": list(_BUCKETS),
        "compiled_paths": {{str(k): str(v) for k, v in mc_paths.items()}},
    }})

    t_load = time.perf_counter()
    models = {{
        s: ct.models.CompiledMLModel(str(mc_paths[s]), compute_units=ct.ComputeUnit.CPU_AND_NE)
        for s in _BUCKETS
    }}
    telem.mark_loaded(None, (time.perf_counter() - t_load) * 1000.0)

    t_warm = time.perf_counter()
    for s in _BUCKETS:
        ii = np.zeros((1, s), dtype=np.int32)
        am = np.zeros((1, s), dtype=np.int32)
        ii[0, 0] = {cls_id}; ii[0, 1] = {sep_id}; am[0, :2] = 1
        t_first = time.perf_counter()
        _ = models[s].predict({{"input_ids": ii, "attention_mask": am}})
        if telem.cold_first_infer_ms is None:
            telem.record_infer((time.perf_counter() - t_first) * 1000.0)
    telem.mark_warmed((time.perf_counter() - t_warm) * 1000.0)

    ii_bufs = {{s: np.zeros((1, s), dtype=np.int32) for s in _BUCKETS}}
    am_bufs = {{s: np.zeros((1, s), dtype=np.int32) for s in _BUCKETS}}

    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        seqs = req["seqs"]
        t_batch = time.perf_counter()
        out_mat = np.zeros((len(seqs), 384), dtype=np.float32)
        for idx, ids in enumerate(seqs):
            n = min(len(ids), 512)
            s = 128 if n <= 128 else (256 if n <= 256 else 512)
            ii = ii_bufs[s]; am = am_bufs[s]
            ii.fill(0); am.fill(0)
            ii[0, :n] = ids[:n]; am[0, :n] = 1
            t_inf = time.perf_counter()
            out = models[s].predict({{"input_ids": ii, "attention_mask": am}})
            telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
            vec = np.asarray(out["embedding"], dtype=np.float32).reshape(-1)[:384]
            out_mat[idx] = vec / max(float(np.linalg.norm(vec)), 1e-12)
        encode_sec = time.perf_counter() - t_batch
        np.save(req["out_npy"], out_mat)
        sys.stdout.write(json.dumps({{"ok": True, "encode_sec": encode_sec}}) + "\\n")
        sys.stdout.flush()

main()
"""
        client_bge = _CoreAiSubprocessClient(
            cmd=[sys.executable],
            daemon_script=daemon_bge,
            script_name="coreml_bge.py",
            prefix="coreml_bge_",
        )

        encode_ids = partial(_encode_coreml_token_ids, client_bge)

        def _enc_bge_cm(txts: list[str], *, prefix: str = "") -> np.ndarray:
            if not txts:
                return np.zeros((0, 384), dtype=np.float32)
            seqs: list[list[int]] = []
            for t in txts:
                ids = list(tok_fast.encode(prefix + t, add_special_tokens=True).ids)
                if len(ids) > 512:
                    ids = ids[:511] + [sep_id]
                seqs.append(ids)
            return encode_ids(seqs, False)[0]

        return Cat1Adapter(
            harness_label="coreml-bge-small-fp16-ane",
            mrl_dims=(128, 256, "max"),
            int8_mode="sym_int8",
            nb_pre_normalized=True,
            enc_q_nb=lambda txts: _enc_bge_cm(txts, prefix=q_prefix),
            enc_d_nb=lambda txts: _enc_bge_cm(txts, prefix=""),
            enc_q=lambda txts: _enc_bge_cm(txts, prefix=q_prefix),
            enc_d=lambda txts: _enc_bge_cm(txts, prefix=""),
            close_fn=lambda: float(client_bge.close().get("peak_gb", 0.0)),
            telemetry_fn=client_bge.telemetry,
            encode_token_ids=encode_ids,
        )

    if pkg_name == "coreml-embeddinggemma-2-fp16":
        return _build_coreml_emb2_adapter("fp16", max_coarse_tokens=max_coarse_tokens)

    if pkg_name == "coreml-embeddinggemma-2-int8":
        return _build_coreml_emb2_adapter("int8", max_coarse_tokens=max_coarse_tokens)

    raise ValueError(f"Unsupported public Cat 1 package: {pkg_name}")


_STOCK_DECISION_REPOS = {
    "public-laya": "convaiinnovations/laya-typed-decisions",
    "public-kev-0.8b": "AbstractFramework/kev-0.8b-mlx-8bit",
}

_STOCK_HF_REPOS: dict[str, tuple[str, ...]] = {
    "public-bge-small": (
        "mlx-community/bge-small-en-v1.5-8bit",
        "mlx-community/bge-small-en-v1.5-bf16",
    ),
    "public-embeddinggemma-2": (
        "mlx-community/embeddinggemma-2-8bit",
        "mlx-community/embeddinggemma-2-bf16",
        "google/embeddinggemma-2",
    ),
    "public-pplx": (
        "zainmerchan/pplx-embed-v2-context-0.6b-mlx",
        "perplexity-ai/pplx-embed-v2-context-0.6b",
    ),
    "public-laya": (_STOCK_DECISION_REPOS["public-laya"],),
    "public-kev-0.8b": (_STOCK_DECISION_REPOS["public-kev-0.8b"],),
    "public-gemma4": ("lmstudio-community/gemma-4-E2B-it-MLX-4bit",),
    "public-gpt-oss": ("mlx-community/gpt-oss-20b-MXFP4-Q4",),
}


def stock_hf_model_files(pkg_name: str) -> list[Path]:
    """Return the actual cached files the OS may report for a stock HF baseline's mappings."""
    for repo in _STOCK_HF_REPOS.get(pkg_name, ()):
        snapshot = resolve_hf_snapshot(repo)
        if snapshot is not None:
            # Hugging Face snapshots can contain symlinks into a sibling blobs directory.
            return [path.resolve() for path in snapshot.rglob("*") if path.is_file()]
    return []


def stock_decision_model_files(backend_name: str) -> list[Path]:
    """Return the actual cached files the OS may report for this model's mappings."""
    if backend_name not in _STOCK_DECISION_REPOS:
        return []
    return stock_hf_model_files(backend_name)


class PublicLayaMlxBackend:
    """Stock `mlx-embeddings` (`modernbert` 8-bit g64) Laya-Typed backend matching `LayaAneBackend` API."""

    def __init__(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_embeddings.models import modernbert

        _ensure_fast_decision_models_on_path()
        from fast_decision_models.laya_ane import LayaAneBackend

        snap = resolve_hf_snapshot(_STOCK_DECISION_REPOS["public-laya"])
        if snap is None:
            raise FileNotFoundError("Missing cached HF snapshot for convaiinnovations/laya-typed-decisions")
        enc_cfg = json.loads((snap / "encoder" / "config.json").read_text(encoding="utf-8"))
        rl_cfg = json.loads((snap / "rl_agent_config.json").read_text(encoding="utf-8"))
        raw_w = mx.load(str(snap / "model.safetensors"))

        self._mx = mx
        self._nn = nn
        # The checkpoint stores the inner ModernBERT, without a masked-language-model head.
        self.encoder = modernbert.ModernBertModel(modernbert.ModelArgs.from_dict(enc_cfg))
        enc_w = {k[len("encoder.") :]: v.astype(mx.bfloat16) for k, v in raw_w.items() if k.startswith("encoder.")}
        self.encoder.load_weights(list(enc_w.items()), strict=True)
        self.encoder.eval()
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
        mx.eval(
            self.encoder.parameters(),
            self.type_emb_w,
            self.head_layers,
            self.sc_n_w,
            self.sc_n_b,
            self.sc_1_w,
            self.sc_1_b,
            self.sc_3_w,
            self.sc_3_b,
        )

        self._tok_helper = LayaAneBackend.from_tokenizer_only(
            snap / "tokenizer" / "tokenizer.json",
            rl_agent_config=rl_cfg,
            laya_dir=snap,
            max_len=512,
        )
        self._qtype_map = {"choice": 0, "score": 1, "noul": 2}
        # The variable-length suite issues many allocation shapes on the 24 GB host.
        self._previous_cache_limit = mx.set_cache_limit(512 << 20)

    def encode_item(self, state: Any, qdef: Mapping[str, Any], *, on_truncate: str = "ignore") -> dict[str, Any]:
        return self._tok_helper.encode_item(state, qdef, on_truncate=on_truncate)

    def _score_generic(self, encs: Sequence[Mapping[str, Any]], qtype_override: str | None = None, bsz: int = 16) -> list[list[float]]:
        mx = self._mx
        nn = self._nn
        out_probs: list[list[float]] = []
        for i in range(0, len(encs), bsz):
            batch = encs[i : i + bsz]
            B = len(batch)
            qtypes = [self._qtype_map[qtype_override] if qtype_override else int(e["qtype"]) for e in batch]
            tv = self.type_emb_w[mx.array(qtypes)][:, None, :]
            max_l = max(len(e["ids"]) for e in batch)
            ids_np = np.full((B, max_l), self._tok_helper.pad_id, dtype=np.int32)
            mask_np = np.zeros((B, max_l), dtype=np.int32)
            for r_i, e in enumerate(batch):
                L = len(e["ids"])
                ids_np[r_i, :L] = e["ids"]
                mask_np[r_i, :L] = 1
            ids_mx = mx.array(ids_np)
            # Stock ModernBERT casts its additive padding and sliding-window masks to this dtype.
            mask_mx = mx.array(mask_np, dtype=mx.bfloat16)
            h = self.encoder(ids_mx, attention_mask=mask_mx)["last_hidden_state"] + tv
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
                temp = self._tok_helper.resolve_temperature(qtypes[r_i], len(m_pos))
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
        if self._previous_cache_limit is not None:
            self._mx.set_cache_limit(self._previous_cache_limit)
            self._previous_cache_limit = None


class CoreAiLayaBackend:
    """Apple `CoreAI` (`ahmedandaloes/laya-typed-decisions-CoreAI` `decision.aimodel`) backend."""

    def __init__(self, coreai_python: str | None = None) -> None:
        _ensure_fast_decision_models_on_path()
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

        tok_helper = LayaAneBackend.from_tokenizer_only(
            snap_ca / "tokenizer" / "tokenizer.json",
            rl_agent_config=rl_cfg,
            laya_dir=snap_ca,
            max_len=min(512, seq_cap),
            head_max_len=LayaAneBackend.resolve_head_max_len(rl_cfg, 512),
            head_max_len_cap=m_cap,
        )
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
            temp = self._tok_helper.resolve_temperature(qti, len(lg))
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


def load_kev_benchmark_tokenizer(kev_dir: Path | None, delimiters: Sequence[str]):
    from tokenizers import Tokenizer

    _ensure_fast_decision_models_on_path()
    from fast_decision_models.kev_mlx import resolve_kev_special_ids

    # The 8-bit repack changed Unicode pre-tokenization. Use the original Kev
    # adapter tokenizer for both scorers. Only tokenizer.json is required.
    source = kev_dir
    if source is None or not (source / "tokenizer.json").is_file():
        source = resolve_hf_snapshot("jaredpalmer/kev-0.8b")
    if source is None or not (source / "tokenizer.json").is_file():
        raise FileNotFoundError("Missing original jaredpalmer/kev-0.8b tokenizer.json for matched scoring")
    tok = Tokenizer.from_file(str(source / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    ids = resolve_kev_special_ids(tok, delimiters)
    return tok, ids


class PublicKev08bMlxBackend:
    """Stock `mlx-lm` (`AbstractFramework/kev-0.8b-mlx-8bit`) backend matching `Kev08bMlxBackend` API."""

    def __init__(self, kev_dir: Path | None = None, *, batch_docs: int = 8) -> None:
        import mlx.core as mx
        import mlx_lm

        snap_k8 = resolve_hf_snapshot(_STOCK_DECISION_REPOS["public-kev-0.8b"])
        if snap_k8 is None:
            raise FileNotFoundError("Missing cached HF snapshot for AbstractFramework/kev-0.8b-mlx-8bit")
        dec_cfg = json.loads((snap_k8 / "decision.json").read_text(encoding="utf-8"))
        self._tok, self._special_ids = load_kev_benchmark_tokenizer(kev_dir, dec_cfg["delimiters"])
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
        # Match the allocator budget used by the project scorer in both benchmark arms.
        self._previous_cache_limit = mx.set_cache_limit(512 << 20)

    def encode_questions(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        return encode_kev_questions_pure(self._tok, self._special_ids, state, questions)

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
            s_ids, q_rows = self.encode_questions(st, qlist)
            out_by_doc[d_i] = [([], []) for _ in q_rows]
            q_off = len(s_ids)
            for q_i, qr in enumerate(q_rows):
                full_ids = list(s_ids) + list(qr["ids"])
                dec_pos = q_off + int(qr.get("decide", len(qr["ids"]) - 1))
                opt_pos = [q_off + int(p) for p in qr["opts"]]
                by_len[len(full_ids)].append((d_i, q_i, full_ids, dec_pos, opt_pos))

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
        if self._previous_cache_limit is not None:
            self._mx.set_cache_limit(self._previous_cache_limit)
            self._previous_cache_limit = None


class CoreMlLayaBackend:
    """Open HF Core ML ANE (`NMTZ/laya-typed-decisions-421m-coreml-ane`, `L192`/`L384`/`L512`) backend."""

    def __init__(self) -> None:
        _ensure_fast_decision_models_on_path()
        from fast_decision_models.laya_ane import LayaAneBackend

        repo_id = COREML_REPO_IDS["coreml-laya"]
        snap_ane = resolve_hf_snapshot(repo_id)
        snap_src = resolve_hf_snapshot("convaiinnovations/laya-typed-decisions")
        if snap_ane is None or snap_src is None or not (snap_src / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Missing cached HF snapshot for {repo_id} or convaiinnovations/laya-typed-decisions"
            )

        avail_buckets = [b for b in (192, 384, 512) if (snap_ane / f"L{b}" / "model.mlpackage").is_dir()]
        if 512 not in avail_buckets:
            raise FileNotFoundError(f"Missing L512/model.mlpackage in {snap_ane}")

        enc_cfg = json.loads((snap_src / "encoder" / "config.json").read_text(encoding="utf-8"))
        rl_cfg = json.loads((snap_src / "rl_agent_config.json").read_text(encoding="utf-8"))
        local_half = int(enc_cfg.get("local_attention", 128)) // 2

        tok_helper = LayaAneBackend.from_tokenizer_only(
            snap_src / "tokenizer" / "tokenizer.json",
            rl_agent_config=rl_cfg,
            laya_dir=snap_src,
            max_len=512,
            head_max_len_cap=256,
        )
        self._tok_helper = tok_helper
        self._qtype_map = {"choice": 0, "score": 1, "noul": 2}
        self.peak_gb = 0.0
        self.last_encode_sec = 0.0

        daemon_laya = f"""
import coremltools as ct
import numpy as np
from safetensors import safe_open
from public_baselines import resolve_coreml_compiled_model, resolve_hf_snapshot

_BUCKETS = {tuple(avail_buckets)!r}
_PAD_ID = {int(tok_helper.pad_id)}
_LOCAL_HALF = {int(local_half)}

def main():
    repo_id = {repo_id!r}
    snap_ane = resolve_hf_snapshot(repo_id)
    snap_src = resolve_hf_snapshot("convaiinnovations/laya-typed-decisions")
    mc_paths = {{}}
    tot_compile = 0.0
    for b in _BUCKETS:
        p, c_s = resolve_coreml_compiled_model(repo_id, f"L{{b}}/model.mlpackage")
        mc_paths[b] = p
        tot_compile += c_s

    telem = _DaemonTelemetry([
        snap_ane.parents[1],
        snap_src.parents[1],
        snap_ane.parents[1] / "blobs",
        snap_src.parents[1] / "blobs",
        mc_paths[_BUCKETS[0]].parent,
    ])
    telem.mark_compiled(tot_compile)
    telem.set_coreml_info({{
        "repo_id": repo_id,
        "compute_units": "CPU_AND_NE",
        "buckets": list(_BUCKETS),
        "compiled_paths": {{str(k): str(v) for k, v in mc_paths.items()}},
    }})

    with safe_open(str(snap_src / "model.safetensors"), framework="numpy") as sf:
        word_f16 = np.ascontiguousarray(sf.get_tensor("encoder.embeddings.tok_embeddings.weight"), dtype=np.float16)
        type_f16 = np.ascontiguousarray(sf.get_tensor("type_emb.weight"), dtype=np.float16)

    t_load = time.perf_counter()
    models = {{
        b: ct.models.CompiledMLModel(str(mc_paths[b]), compute_units=ct.ComputeUnit.CPU_AND_NE)
        for b in _BUCKETS
    }}
    telem.mark_loaded(None, (time.perf_counter() - t_load) * 1000.0)

    windows = {{}}
    for b in _BUCKETS:
        pos = np.arange(b)
        windows[b] = (np.abs(pos[:, None] - pos[None, :]) <= _LOCAL_HALF)

    def _run_one(ids, mpos, qti):
        n = min(len(ids), _BUCKETS[-1])
        b = next((bk for bk in _BUCKETS if n <= bk), _BUCKETS[-1])
        k = min(len(mpos), 32)
        ii = np.full((b,), _PAD_ID, dtype=np.int32)
        ii[:n] = ids[:n]
        valid = np.zeros((b,), dtype=bool)
        valid[:n] = True
        emb = np.ascontiguousarray(word_f16[ii].T[None, :, None, :], dtype=np.float16)
        full_b = np.broadcast_to(valid[None, :], (b, b))
        local_b = (windows[b] | (~valid[:, None])) & full_b
        full_mask = np.ascontiguousarray(
            np.where(full_b.T[None, :, None, :], np.float16(0.0), np.float16(-10000.0)), dtype=np.float16
        )
        local_mask = np.ascontiguousarray(
            np.where(local_b.T[None, :, None, :], np.float16(0.0), np.float16(-10000.0)), dtype=np.float16
        )
        tv = np.ascontiguousarray(type_f16[int(qti)][None, :, None, None], dtype=np.float16)
        mm = np.zeros((1, b, 1, 32), dtype=np.float16)
        mp_arr = np.zeros((32,), dtype=np.int32)
        for j in range(k):
            mp_arr[j] = min(max(int(mpos[j]), 0), b - 1)
        mm[0, mp_arr, 0, np.arange(32)] = np.float16(1.0)
        t_inf = time.perf_counter()
        out = models[b].predict({{
            "embeddings": emb,
            "full_mask": full_mask,
            "local_mask": local_mask,
            "type_vectors": tv,
            "marker_map": mm,
        }})
        telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
        logits_arr = next(v for v in out.values() if tuple(np.asarray(v).shape)[1] == 1)
        return np.asarray(logits_arr, dtype=np.float64).reshape(32)[:k].tolist()

    t_warm = time.perf_counter()
    for b in _BUCKETS:
        _ = _run_one([_PAD_ID] * b, [0, 1], 0)
    telem.mark_warmed((time.perf_counter() - t_warm) * 1000.0)
    # Preserve true first cold inference latency from warmup; keep warm_latencies_ms post-READY only.
    telem.warm_latencies_ms = []

    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        t_batch = time.perf_counter()
        out_logits = [
            _run_one(ids, mpos, qti)
            for ids, mpos, qti in zip(req["ids_list"], req["mpos_list"], req["qt_list"])
        ]
        encode_sec = time.perf_counter() - t_batch
        Path(req["out_json"]).write_text(json.dumps(out_logits), encoding="utf-8")
        sys.stdout.write(json.dumps({{"ok": True, "encode_sec": encode_sec}}) + "\\n")
        sys.stdout.flush()

main()
"""
        self._client = _CoreAiSubprocessClient(
            cmd=[sys.executable],
            daemon_script=daemon_laya,
            script_name="coreml_laya.py",
            prefix="coreml_laya_",
        )

    def encode_item(self, state: Any, qdef: Mapping[str, Any], *, on_truncate: str = "ignore") -> dict[str, Any]:
        return self._tok_helper.encode_item(state, qdef, on_truncate=on_truncate)

    def _score_generic(self, encs: Sequence[Mapping[str, Any]], qtype_override: str | None = None) -> list[list[float]]:
        if not encs:
            self.last_encode_sec = 0.0
            return []
        oj = self._client.next_temp_path("o_{n}.json")
        qt_list = [
            self._qtype_map[qtype_override] if qtype_override else int(e.get("qtype", 0))
            for e in encs
        ]
        try:
            resp = self._client.request(
                {
                    "qt_list": qt_list,
                    "ids_list": [e["ids"] for e in encs],
                    "mpos_list": [e["markers"] for e in encs],
                    "out_json": str(oj),
                }
            )
            self.last_encode_sec = float(resp.get("encode_sec", 0.0))
            self.peak_gb = max(self.peak_gb, self._client.peak_gb)
            raw_list = json.loads(oj.read_text(encoding="utf-8"))
        finally:
            oj.unlink(missing_ok=True)
        probs: list[list[float]] = []
        for lg, qti in zip(raw_list, qt_list):
            temp = self._tok_helper.resolve_temperature(qti, len(lg))
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


class CoreMlKev08bBackend:
    """Open HF Core ML FP16 (`FluidInference/kev-0.8b-coreml`, `KevFused` + `KevRow` GPU) backend."""

    def __init__(self) -> None:
        from tokenizers import Tokenizer

        repo_id = COREML_REPO_IDS["coreml-kev-0.8b"]
        snap = resolve_hf_snapshot(repo_id)
        if snap is None or not (snap / "fused" / "config.json").is_file() or not (snap / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Missing cached HF CoreML snapshot for {repo_id}")

        f_cfg = json.loads((snap / "fused" / "config.json").read_text(encoding="utf-8"))
        sp = f_cfg["special_tokens"]
        self._special_ids = [
            int(sp["state"]),
            int(sp["q"]),
            int(sp["opt"]),
            int(sp["close_opt"]),
            int(sp["decide"]),
        ]
        self._tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self._tok.no_truncation()
        self._tok.no_padding()
        self.peak_gb = 0.0
        self.last_encode_sec = 0.0

        daemon_kev = f"""
import coremltools as ct
import numpy as np
from public_baselines import resolve_coreml_compiled_model, resolve_hf_snapshot
from _outcome_public_backends import kev_lane_starts, kev_pack_groups

_STATE_BUCKETS = (32, 64, 128, 192, 256, 384)
_PACKED_LEN = 192
_LANE = 128
_READOUTS = 16
_MAX_OPTS = 16

def _softmax_f64(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr - np.max(arr)
    ex = np.exp(arr)
    return (ex / np.sum(ex)).tolist()

def main():
    repo_id = {repo_id!r}
    snap = resolve_hf_snapshot(repo_id)
    f_cfg = json.loads((snap / "fused" / "config.json").read_text(encoding="utf-8"))
    hidden_sz = int(f_cfg["hidden_size"])
    rotary_dim = int(f_cfg["rotary_dim"])
    rope_theta = float(f_cfg["rope_theta"])
    conv_tail = int(f_cfg["conv_kernel"]) - 1
    vocab_sz = int(f_cfg["vocab_size"])
    pad_id = int(f_cfg["pad_id"])
    temperature = float(f_cfg.get("temperature", 2.3510958125672174))

    fused_mc, c0 = resolve_coreml_compiled_model(repo_id, "fused/KevFused.mlpackage")
    row512_mc, c1 = resolve_coreml_compiled_model(repo_id, "L512_K16/KevRow_fp16.mlpackage")
    row1024_mc, c2 = resolve_coreml_compiled_model(repo_id, "L1024_K80/KevRow_fp16.mlpackage")
    telem = _DaemonTelemetry([snap.parents[1], snap.parents[1] / "blobs", fused_mc.parent])
    telem.mark_compiled(c0 + c1 + c2)
    telem.set_coreml_info({{
        "repo_id": repo_id,
        "compute_units": "CPU_AND_GPU",
        "fused_state_buckets": list(_STATE_BUCKETS),
        "packed_len": _PACKED_LEN,
        "row_buckets": ["L512_K16", "L1024_K80"],
        "compiled_paths": {{
            "fused": str(fused_mc),
            "L512_K16": str(row512_mc),
            "L1024_K80": str(row1024_mc),
        }},
    }})

    emb_table = np.memmap(str(snap / "L512_K16" / "embeddings.f16"), dtype=np.float16, mode="r").reshape(vocab_sz, hidden_sz)

    half_rot = rotary_dim // 2
    pos_arr = np.arange(2048, dtype=np.float64)[:, None]
    inv_f = 1.0 / (rope_theta ** (2.0 * np.arange(half_rot, dtype=np.float64) / float(rotary_dim)))
    angles = pos_arr * inv_f[None, :]
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)
    cos_f32 = np.concatenate([cos_half, cos_half], axis=-1).astype(np.float32)
    sin_f32 = np.concatenate([sin_half, sin_half], axis=-1).astype(np.float32)
    cos_f16 = cos_f32.astype(np.float16)
    sin_f16 = sin_f32.astype(np.float16)

    t_load = time.perf_counter()
    fused_models = {{
        s: ct.models.CompiledMLModel(
            str(fused_mc),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            function_name=f"fused_S{{s}}_P{{_PACKED_LEN}}_B{{_READOUTS}}_K{{_MAX_OPTS}}",
        )
        for s in _STATE_BUCKETS
    }}
    row512_model = ct.models.CompiledMLModel(str(row512_mc), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    row1024_model = ct.models.CompiledMLModel(str(row1024_mc), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    telem.mark_loaded(None, (time.perf_counter() - t_load) * 1000.0)

    def _run_fused(state_ids, state_len, branches, starts):
        total = state_len + _PACKED_LEN
        n = len(state_ids)
        tokens = np.full((total,), pad_id, dtype=np.int32)
        tokens[:n] = state_ids
        positions = np.empty((total,), dtype=np.int32)
        positions[:state_len] = np.arange(state_len, dtype=np.int32)
        positions[state_len:] = n

        valid = np.zeros((state_len,), dtype=np.float16)
        valid[:n] = np.float16(1.0)
        tail = np.zeros((conv_tail, state_len), dtype=np.float16)
        for j in range(conv_tail):
            pt = n - conv_tail + j
            if pt >= 0:
                tail[j, pt] = np.float16(1.0)
        segment = np.eye(_PACKED_LEN, dtype=np.float16)
        lag_keep = np.zeros((conv_tail, _PACKED_LEN), dtype=np.float16)
        lag_tail = np.zeros((conv_tail, _PACKED_LEN, conv_tail), dtype=np.float16)
        decide = np.zeros((_READOUTS, _PACKED_LEN), dtype=np.float16)
        options = np.zeros((_READOUTS, _MAX_OPTS, _PACKED_LEN), dtype=np.float16)
        opt_mask = np.zeros((_READOUTS, _MAX_OPTS), dtype=np.float16)

        for b, (br, st) in enumerate(zip(branches, starts)):
            cnt = len(br["ids"])
            tokens[state_len + st : state_len + st + cnt] = br["ids"]
            positions[state_len + st : state_len + st + cnt] = np.arange(n, n + cnt, dtype=np.int32)
            segment[st : st + cnt, st : st + cnt] = np.tril(np.ones((cnt, cnt), dtype=np.float16))
            for s_lag in range(1, conv_tail + 1):
                if cnt > s_lag:
                    lag_keep[s_lag - 1, st + s_lag : st + cnt] = np.float16(1.0)
                for p_i in range(min(s_lag, cnt)):
                    lag_tail[s_lag - 1, st + p_i, conv_tail + p_i - s_lag] = np.float16(1.0)
            decide[b, st + int(br["decide"])] = np.float16(1.0)
            for slot, o_idx in enumerate(br["opts"]):
                options[b, slot, st + int(o_idx)] = np.float16(1.0)
                opt_mask[b, slot] = np.float16(1.0)

        hidden = np.ascontiguousarray(emb_table[tokens][None, :, :], dtype=np.float16)
        t_inf = time.perf_counter()
        out = fused_models[state_len].predict({{
            "hidden": hidden,
            "cos": np.ascontiguousarray(cos_f16[positions]),
            "sin": np.ascontiguousarray(sin_f16[positions]),
            "valid": valid,
            "tail_onehot": tail,
            "segment": segment,
            "lag_keep": lag_keep,
            "lag_tail": lag_tail,
            "decide_onehot": decide,
            "option_onehot": options,
            "option_mask": opt_mask,
        }})
        telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
        logits_mat = np.asarray(out["logits"], dtype=np.float64).reshape(_READOUTS, _MAX_OPTS)
        res = []
        for b, br in enumerate(branches):
            k = len(br["opts"])
            scaled_l = logits_mat[b, :k]
            res.append((_softmax_f64(scaled_l), (scaled_l * temperature).tolist()))
        return res

    def _run_row(state_ids, br):
        q_ids = br["ids"]
        k = len(br["opts"])
        if len(state_ids) + len(q_ids) > 1024:
            s_eff = state_ids[: max(1, 1024 - len(q_ids))]
        else:
            s_eff = state_ids
        full_ids = s_eff + q_ids
        q_off = len(s_eff)
        dec_pos = q_off + int(br["decide"])
        opt_pos = [q_off + int(p) for p in br["opts"]]
        if len(full_ids) <= 512 and k <= 16:
            L, max_k, mdl = 512, 16, row512_model
        else:
            L, max_k, mdl = 1024, 80, row1024_model
            if len(full_ids) > 1024:
                full_ids = full_ids[:1024]
                dec_pos = min(dec_pos, 1023)
                opt_pos = [min(p, 1023) for p in opt_pos[:80]]
                k = len(opt_pos)
        toks = np.full((L,), pad_id, dtype=np.int32)
        toks[: len(full_ids)] = full_ids
        hidden = np.ascontiguousarray(emb_table[toks][None, :, :], dtype=np.float32)
        dec_oh = np.zeros((1, L), dtype=np.float32)
        dec_oh[0, dec_pos] = 1.0
        opt_oh = np.zeros((max_k, L), dtype=np.float32)
        opt_m = np.zeros((max_k,), dtype=np.float32)
        for slot, op in enumerate(opt_pos):
            opt_oh[slot, op] = 1.0
            opt_m[slot] = 1.0
        t_inf = time.perf_counter()
        out = mdl.predict({{
            "hidden": hidden,
            "cos": np.ascontiguousarray(cos_f32[:L]),
            "sin": np.ascontiguousarray(sin_f32[:L]),
            "decide_onehot": dec_oh,
            "option_onehot": opt_oh,
            "option_mask": opt_m,
        }})
        telem.record_infer((time.perf_counter() - t_inf) * 1000.0)
        scaled_l = np.asarray(out["logits"], dtype=np.float64).reshape(-1)[:k]
        return (_softmax_f64(scaled_l), (scaled_l * temperature).tolist())

    t_warm = time.perf_counter()
    dummy_br = {{"ids": [pad_id, pad_id], "decide": 1, "opts": [0, 1], "k": 2}}
    for s in _STATE_BUCKETS:
        _ = _run_fused([pad_id], s, [dummy_br], [0])
    _ = _run_row([pad_id], dummy_br)
    # Also warm L1024_K80 via >512 dummy state
    _ = _run_row([pad_id] * 513, dummy_br)
    telem.mark_warmed((time.perf_counter() - t_warm) * 1000.0)
    # Preserve true first cold inference latency from warmup; keep warm_latencies_ms post-READY only.
    telem.warm_latencies_ms = []

    sys.stdout.write("READY\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            sys.stdout.write(json.dumps(telem.payload(final=True)) + "\\n")
            sys.stdout.flush()
            break
        t_batch = time.perf_counter()
        out_all = []
        for s_ids, q_rows in req["items"]:
            doc_out = [None] * len(q_rows)
            s_bucket = next((s for s in _STATE_BUCKETS if len(s_ids) <= s), None)
            if s_bucket is None:
                for qi, qr in enumerate(q_rows):
                    doc_out[qi] = _run_row(s_ids, qr)
                out_all.append(doc_out)
                continue
            fits = [qi for qi, qr in enumerate(q_rows) if len(qr["ids"]) <= _LANE and len(qr["opts"]) <= _MAX_OPTS]
            fit_set = set(fits)
            for qi, qr in enumerate(q_rows):
                if qi not in fit_set:
                    doc_out[qi] = _run_row(s_ids, qr)
            if fits:
                groups = kev_pack_groups([len(q_rows[qi]["ids"]) for qi in fits], packed_len=_PACKED_LEN, lane=_LANE, max_readouts=_READOUTS)
                for grp in groups:
                    orig_qis = [fits[g] for g in grp]
                    brs = [q_rows[qi] for qi in orig_qis]
                    sts = kev_lane_starts([len(b["ids"]) for b in brs], lane=_LANE)
                    res_grp = _run_fused(s_ids, s_bucket, brs, sts)
                    for qi, pair_val in zip(orig_qis, res_grp):
                        doc_out[qi] = pair_val
            out_all.append(doc_out)
        encode_sec = time.perf_counter() - t_batch
        Path(req["out_json"]).write_text(json.dumps(out_all), encoding="utf-8")
        sys.stdout.write(json.dumps({{"ok": True, "encode_sec": encode_sec}}) + "\\n")
        sys.stdout.flush()

main()
"""
        self._client = _CoreAiSubprocessClient(
            cmd=[sys.executable],
            daemon_script=daemon_kev,
            script_name="coreml_kev.py",
            prefix="coreml_kev_",
        )

    def encode_questions(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        return encode_kev_questions_pure(self._tok, self._special_ids, state, questions)

    def score_multi_states(
        self,
        pairs: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        del batch_docs
        if not pairs:
            self.last_encode_sec = 0.0
            return []
        encoded = [self.encode_questions(st, qs) for st, qs in pairs]
        oj = self._client.next_temp_path("kev_{n}.json")
        try:
            resp = self._client.request({"items": encoded, "out_json": str(oj)})
            self.last_encode_sec = float(resp.get("encode_sec", 0.0))
            self.peak_gb = max(self.peak_gb, self._client.peak_gb)
            raw = json.loads(oj.read_text(encoding="utf-8"))
        finally:
            oj.unlink(missing_ok=True)
        return [[(list(pr), list(lg)) for pr, lg in doc_rows] for doc_rows in raw]

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

    def telemetry(self) -> dict[str, Any]:
        return self._client.telemetry()

    def close(self) -> float:
        telem = self._client.close()
        self.peak_gb = max(self.peak_gb, float(telem.get("peak_gb", 0.0)))
        return self.peak_gb


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
