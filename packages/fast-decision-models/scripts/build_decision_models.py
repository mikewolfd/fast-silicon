#!/usr/bin/env python3
"""Convenience script entrypoint wrapping `fast_decision_models.builder` with parallel CoreML `--jobs`."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any, Sequence

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from fast_shared.parallel_coreml import (
        atomic_publish_dir,
        compile_buckets_parallel,
        default_compile_jobs,
    )
except ModuleNotFoundError:  # pragma: no cover - standalone invocation without editable install
    _shared_src = Path(__file__).resolve().parents[2] / "fast-shared" / "src"
    if _shared_src.is_dir() and str(_shared_src) not in sys.path:
        sys.path.insert(0, str(_shared_src))
    from fast_shared.parallel_coreml import (
        atomic_publish_dir,
        compile_buckets_parallel,
        default_compile_jobs,
    )

from fast_decision_models.builder import (  # noqa: E402
    _compile_laya_seq_bundle,
    _resolve_dir_or_hf,
    acquire_hw_lock,
    add_setup_weights_arguments,
    build_kev_mlx_artifacts,
    build_laya_ane_artifacts,
    compile_native_dylib,
    quantize_kev_w8_artifacts,
)
from fast_decision_models.paths import (  # noqa: E402
    DEFAULT_LAYA_REPO,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
)

DEFAULT_LAYA_SEQ_LENS: tuple[int, ...] = (256, 512)
_WORKER_SAFETENSORS_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def _load_safetensors_as_float32_np(sf_path: Path) -> dict[str, np.ndarray]:
    """Read a safetensors file into CPU float32 NumPy arrays without initializing MLX/Metal."""
    raw_bytes = sf_path.read_bytes()
    if len(raw_bytes) < 8:
        raise ValueError(f"Truncated safetensors file: {sf_path}")
    header_len = struct.unpack_from("<Q", raw_bytes, 0)[0]
    data_offset = 8 + header_len
    header = json.loads(raw_bytes[8:data_offset].decode("utf-8"))

    tensors: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_tag = str(info["dtype"]).upper()
        shape = tuple(int(d) for d in info["shape"])
        start, end = info["data_offsets"]
        buf = memoryview(raw_bytes)[data_offset + int(start) : data_offset + int(end)]
        if dtype_tag == "F32":
            arr = np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
        elif dtype_tag == "F16":
            arr = np.frombuffer(buf, dtype="<f2").reshape(shape).astype(np.float32)
        elif dtype_tag == "BF16":
            u16 = np.frombuffer(buf, dtype="<u2")
            arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
        elif dtype_tag == "F64":
            arr = np.frombuffer(buf, dtype="<f8").reshape(shape).astype(np.float32)
        elif dtype_tag == "I32":
            arr = np.frombuffer(buf, dtype="<i4").reshape(shape).copy()
        elif dtype_tag == "I64":
            arr = np.frombuffer(buf, dtype="<i8").reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype_tag!r} for {name}")
        tensors[name] = arr
    return tensors


def _compile_laya_bucket_task(task: tuple[int, str, str]) -> str:
    """Top-level `spawn`-safe worker compiling one `Laya-Typed-421M` CoreML sequence bucket."""
    seq_len, src_dir_str, out_dir_str = task
    src_dir = Path(src_dir_str)
    out_dir = Path(out_dir_str)
    sf_path = src_dir / "model.safetensors"
    cache_key = (str(sf_path), sf_path.stat().st_mtime_ns)
    w_np = _WORKER_SAFETENSORS_CACHE.get(cache_key)
    if w_np is None:
        w_np = _load_safetensors_as_float32_np(sf_path)
        _WORKER_SAFETENSORS_CACHE.clear()
        _WORKER_SAFETENSORS_CACHE[cache_key] = w_np

    enc_cfg = json.loads((src_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
    stage_dir = out_dir / f".stage_laya_s{seq_len}_{os.getpid()}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.perf_counter()
        built_mc = _compile_laya_seq_bundle(int(seq_len), w_np, enc_cfg, stage_dir)
        dst_mc = out_dir / built_mc.name
        atomic_publish_dir(built_mc, dst_mc, move=True)
        print(
            f"[LAYA-ANE] Built {dst_mc.name} in {time.perf_counter() - t0:.2f}s (pid={os.getpid()})",
            flush=True,
        )
        return str(dst_mc)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _parse_seq_lens(spec: str | Sequence[int]) -> tuple[int, ...]:
    if not isinstance(spec, str):
        vals = tuple(int(x) for x in spec)
    else:
        vals = tuple(int(tok.strip()) for tok in spec.split(",") if tok.strip())
    if not vals or any(v < 16 for v in vals):
        raise ValueError(f"Invalid --laya-seq-lens={spec!r}; expected positive sequence lengths (>= 16).")
    return vals


def build_laya_ane_artifacts_parallel(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    laya_dir: str | None = None,
    copy_compiled_from: str | None = None,
    jobs: int | None = None,
    seq_lens: Sequence[int] = DEFAULT_LAYA_SEQ_LENS,
) -> Path:
    """Build & stage `Laya-Typed-421M` Pure-ANE (`W8A16`) bundles across `spawn` workers."""
    if copy_compiled_from is not None:
        return build_laya_ane_artifacts(
            repo_root,
            artifacts_dir=artifacts_dir,
            laya_dir=laya_dir,
            copy_compiled_from=copy_compiled_from,
        )

    eff_seq_lens = _parse_seq_lens(seq_lens)
    eff_jobs = default_compile_jobs(jobs)
    src_dir = _resolve_dir_or_hf(laya_dir, DEFAULT_LAYA_REPO)
    out_root = artifacts_dir if artifacts_dir is not None else resolve_artifacts_dir(home=repo_root)
    out_dir = out_root / "laya_typed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_dir / "rl_agent_config.json", out_dir / "rl_agent_config.json")
    shutil.copy2(src_dir / "encoder" / "config.json", out_dir / "encoder_config.json")
    shutil.copy2(src_dir / "tokenizer" / "tokenizer.json", out_dir / "tokenizer" / "tokenizer.json")
    if (src_dir / "tokenizer" / "tokenizer_config.json").exists():
        shutil.copy2(src_dir / "tokenizer" / "tokenizer_config.json", out_dir / "tokenizer" / "tokenizer_config.json")

    raw_np = _load_safetensors_as_float32_np(src_dir / "model.safetensors")
    word_f16 = np.ascontiguousarray(raw_np["encoder.embeddings.tok_embeddings.weight"].astype(np.float16))
    type_f16 = np.ascontiguousarray(raw_np["type_emb.weight"].astype(np.float16))
    np.savez(
        out_dir / "laya_typed_emb_f16.npz",
        word_f16=word_f16,
        type_f16=type_f16,
    )
    del raw_np, word_f16, type_f16

    dylib_p = ensure_laya_dylib(repo_root)
    tasks = [(int(s_len), str(src_dir.resolve()), str(out_dir.resolve())) for s_len in eff_seq_lens]

    t0 = time.perf_counter()
    with acquire_hw_lock("build_laya_ane_models"):
        compile_buckets_parallel(
            tasks,
            _compile_laya_bucket_task,
            max_workers=eff_jobs,
            label_fn=lambda t: f"laya_typed_s{t[0]}_w8",
            extra_sys_paths=(Path(__file__).resolve().parent, SRC_DIR),
        )
    print(
        f"[LAYA-ANE] Compiled {len(eff_seq_lens)} bundle(s) with jobs={min(eff_jobs, len(eff_seq_lens))} "
        f"in {time.perf_counter() - t0:.2f}s -> {out_dir}",
        flush=True,
    )

    with acquire_hw_lock("verify_laya_ane_plan"):
        lib = ctypes.CDLL(str(dylib_p))
        for s_len in eff_seq_lens:
            mc = out_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
            counts = (ctypes.c_int * 3)()
            # 0 == LAYA_COMPUTE_CPU_AND_ANE (aligned with fast_shared::ComputeMode::kCpuAndNeuralEngine)
            lib.laya_ane_inspect_plan(str(mc).encode("utf-8"), 0, counts)
            print(f"[LAYA-ANE] {mc.name}: MLComputePlan ANE/GPU/CPU = {list(counts)}", flush=True)
            if int(counts[1]) != 0 or int(counts[2]) != 0 or int(counts[0]) < 1000:
                raise RuntimeError(f"Unexpected non-pure-ANE compute plan on {mc}: {list(counts)}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Laya-Typed ANE & Kev-0.8B MLX decision-model artifacts")
    add_setup_weights_arguments(parser)
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"Parallel CoreML worker processes for Laya ANE buckets (default: min(4, cpu_count)={default_compile_jobs()}).",
    )
    parser.add_argument(
        "--laya-seq-lens",
        default="256,512",
        help="Comma-separated Laya ANE sequence bucket lengths (default: 256,512).",
    )
    args = parser.parse_args(argv)

    repo_root = resolve_home(args.home)
    out_artifacts = resolve_artifacts_dir(args.artifacts_dir, home=repo_root)
    run_all = not (
        args.compile_native or args.build_laya_ane or args.build_kev_mlx or args.quantize_existing_kev_w8
    )

    if args.compile_native or run_all:
        dylib = compile_native_dylib(repo_root)
        print(f"[NATIVE] Compiled {dylib}")
    if args.build_laya_ane or run_all:
        build_laya_ane_artifacts_parallel(
            repo_root,
            artifacts_dir=out_artifacts,
            laya_dir=args.laya_dir,
            copy_compiled_from=args.copy_compiled_laya_from,
            jobs=args.jobs,
            seq_lens=_parse_seq_lens(args.laya_seq_lens),
        )
    if args.build_kev_mlx or run_all:
        build_kev_mlx_artifacts(
            repo_root,
            artifacts_dir=out_artifacts,
            kev_adapter_dir=args.kev_adapter_dir,
            kev_base_dir=args.kev_base_dir,
        )
    elif args.quantize_existing_kev_w8:
        quantize_kev_w8_artifacts(repo_root, artifacts_dir=out_artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
