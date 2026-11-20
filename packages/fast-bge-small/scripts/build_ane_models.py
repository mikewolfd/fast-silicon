#!/usr/bin/env python3
"""Build CoreML FP16 Neural Engine (.mlmodelc) sequence buckets from artifacts/weights/bge_small_f16.npz.

Requires Python 3.12 + coremltools>=8.0 (only needed if regenerating .mlmodelc bundles).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

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

# Smallest positive normal IEEE-754 binary16 value: 2^-14 ~= 6.103515625e-5.
# Clamping LayerNorm epsilon to >= MIN_FP16_LAYERNORM_EPS prevents np.float16(1e-12)
# from underflowing to exact 0.0 in FP16 CoreML/ANE graphs (rsqrt(var + 0.0) -> Inf/NaN).
MIN_FP16_LAYERNORM_EPS: float = float(np.finfo(np.float16).tiny)

# Production zero-GPU-contention bucket suite matching Native/Python engine routing.
PRODUCTION_BUCKETS = [
    (1, 24),
    (1, 64),
    (1, 128),
    (1, 192),
    (8, 256),
    (1, 320),
    (1, 384),
    (1, 448),
    (1, 512),
]

# Optional single-query S=256 bucket eliminating the 193..256 -> 320 query padding gap.
OPTIONAL_QUERY_BUCKETS = [
    (1, 256),
]


def safe_fp16_layernorm_epsilon(eps: float = 1e-12) -> np.float16:
    """Return an FP16-representable positive normal LayerNorm epsilon (>= 2^-14)."""
    clamped = max(float(eps), MIN_FP16_LAYERNORM_EPS)
    val = np.float16(clamped)
    if not np.isfinite(val) or float(val) < MIN_FP16_LAYERNORM_EPS:
        val = np.float16(MIN_FP16_LAYERNORM_EPS)
    return val


def repo_root() -> Path:
    return Path(os.environ.get("FAST_BGE_SMALL_HOME") or Path(__file__).resolve().parent.parent)


def build_bert_ane_mil(
    weights: dict[str, np.ndarray],
    batch_size: int,
    seq_len: int,
    *,
    layer_norm_eps: float = 1e-12,
):
    """Build channel-first [B, 384, 1, S] BGE-small body with Layer-11 [CLS] slice."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    scale_val = np.float16(1.0 / np.sqrt(32.0))
    ln_eps = safe_fp16_layernorm_epsilon(layer_norm_eps)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch_size, 384, 1, seq_len), dtype=types.fp16),
            mb.TensorSpec(shape=(batch_size, 1, 1, seq_len), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def bge_body(emb_sum, ext_mask):
        emb_ln_w = mb.const(val=weights["embeddings.LayerNorm.weight"].astype(np.float16), name="emb_ln_w")
        emb_ln_b = mb.const(val=weights["embeddings.LayerNorm.bias"].astype(np.float16), name="emb_ln_b")
        h = mb.layer_norm(x=emb_sum, axes=[1], gamma=emb_ln_w, beta=emb_ln_b, epsilon=ln_eps)

        for l in range(11):
            p = f"encoder.layer.{l}."
            wq = weights[p + "attention.self.query.weight"]
            wk = weights[p + "attention.self.key.weight"]
            wv = weights[p + "attention.self.value.weight"]
            bq = weights[p + "attention.self.query.bias"]
            bk = weights[p + "attention.self.key.bias"]
            bv = weights[p + "attention.self.value.bias"]

            w_qkv = np.concatenate([wq, wk, wv], axis=0)[:, :, None, None].astype(np.float16)
            b_qkv = np.concatenate([bq, bk, bv], axis=0).astype(np.float16)
            qkv = mb.conv(x=h, weight=mb.const(val=w_qkv, name=f"l{l}_wqkv"), bias=mb.const(val=b_qkv, name=f"l{l}_bqkv"))
            q, k, v = mb.split(x=qkv, num_splits=3, axis=1)

            q4 = mb.reshape(x=q, shape=[batch_size, 12, 32, seq_len])
            k4 = mb.reshape(x=k, shape=[batch_size, 12, 32, seq_len])
            v4 = mb.reshape(x=v, shape=[batch_size, 12, 32, seq_len])
            q_t = mb.transpose(x=q4, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q_t, y=k4)
            scores = mb.mul(x=scores, y=mb.const(val=scale_val, name=f"l{l}_sc"))
            scores = mb.add(x=scores, y=ext_mask)
            probs = mb.softmax(x=scores, axis=-1)
            v_t = mb.transpose(x=v4, perm=[0, 1, 3, 2])
            ctx = mb.matmul(x=probs, y=v_t)
            ctx = mb.transpose(x=ctx, perm=[0, 1, 3, 2])
            ctx = mb.reshape(x=ctx, shape=[batch_size, 384, 1, seq_len])

            wo = weights[p + "attention.output.dense.weight"][:, :, None, None].astype(np.float16)
            bo = weights[p + "attention.output.dense.bias"].astype(np.float16)
            ao = mb.conv(x=ctx, weight=mb.const(val=wo, name=f"l{l}_wo"), bias=mb.const(val=bo, name=f"l{l}_bo"))
            h = mb.add(x=ao, y=h)
            ln1w = mb.const(val=weights[p + "attention.output.LayerNorm.weight"].astype(np.float16), name=f"l{l}_ln1w")
            ln1b = mb.const(val=weights[p + "attention.output.LayerNorm.bias"].astype(np.float16), name=f"l{l}_ln1b")
            h = mb.layer_norm(x=h, axes=[1], gamma=ln1w, beta=ln1b, epsilon=ln_eps)

            wup = weights[p + "intermediate.dense.weight"][:, :, None, None].astype(np.float16)
            bup = weights[p + "intermediate.dense.bias"].astype(np.float16)
            up = mb.conv(x=h, weight=mb.const(val=wup, name=f"l{l}_wup"), bias=mb.const(val=bup, name=f"l{l}_bup"))
            act = mb.gelu(x=up, mode="EXACT")
            wdn = weights[p + "output.dense.weight"][:, :, None, None].astype(np.float16)
            bdn = weights[p + "output.dense.bias"].astype(np.float16)
            dn = mb.conv(x=act, weight=mb.const(val=wdn, name=f"l{l}_wdn"), bias=mb.const(val=bdn, name=f"l{l}_bdn"))
            h = mb.add(x=dn, y=h)
            ln2w = mb.const(val=weights[p + "output.LayerNorm.weight"].astype(np.float16), name=f"l{l}_ln2w")
            ln2b = mb.const(val=weights[p + "output.LayerNorm.bias"].astype(np.float16), name=f"l{l}_ln2b")
            h = mb.layer_norm(x=h, axes=[1], gamma=ln2w, beta=ln2b, epsilon=ln_eps)

        p = "encoder.layer.11."
        h_cls = mb.slice_by_index(x=h, begin=[0, 0, 0, 0], end=[batch_size, 384, 1, 1])
        wq = weights[p + "attention.self.query.weight"][:, :, None, None].astype(np.float16)
        bq = weights[p + "attention.self.query.bias"].astype(np.float16)
        q11 = mb.conv(x=h_cls, weight=mb.const(val=wq, name="l11_wq"), bias=mb.const(val=bq, name="l11_bq"))

        wk = weights[p + "attention.self.key.weight"]
        wv = weights[p + "attention.self.value.weight"]
        bk = weights[p + "attention.self.key.bias"]
        bv = weights[p + "attention.self.value.bias"]
        w_kv = np.concatenate([wk, wv], axis=0)[:, :, None, None].astype(np.float16)
        b_kv = np.concatenate([bk, bv], axis=0).astype(np.float16)
        kv11 = mb.conv(x=h, weight=mb.const(val=w_kv, name="l11_wkv"), bias=mb.const(val=b_kv, name="l11_bkv"))
        k11, v11 = mb.split(x=kv11, num_splits=2, axis=1)

        q4 = mb.reshape(x=q11, shape=[batch_size, 12, 32, 1])
        k4 = mb.reshape(x=k11, shape=[batch_size, 12, 32, seq_len])
        v4 = mb.reshape(x=v11, shape=[batch_size, 12, 32, seq_len])
        q_t = mb.transpose(x=q4, perm=[0, 1, 3, 2])
        scores = mb.matmul(x=q_t, y=k4)
        scores = mb.mul(x=scores, y=mb.const(val=scale_val, name="l11_sc"))
        scores = mb.add(x=scores, y=ext_mask)
        probs = mb.softmax(x=scores, axis=-1)
        v_t = mb.transpose(x=v4, perm=[0, 1, 3, 2])
        ctx11 = mb.matmul(x=probs, y=v_t)
        ctx11 = mb.transpose(x=ctx11, perm=[0, 1, 3, 2])
        ctx11 = mb.reshape(x=ctx11, shape=[batch_size, 384, 1, 1])

        wo11 = weights[p + "attention.output.dense.weight"][:, :, None, None].astype(np.float16)
        bo11 = weights[p + "attention.output.dense.bias"].astype(np.float16)
        ao11 = mb.conv(x=ctx11, weight=mb.const(val=wo11, name="l11_wo"), bias=mb.const(val=bo11, name="l11_bo"))
        hc = mb.add(x=ao11, y=h_cls)
        hc = mb.layer_norm(
            x=hc,
            axes=[1],
            gamma=mb.const(val=weights[p + "attention.output.LayerNorm.weight"].astype(np.float16), name="l11_ln1w"),
            beta=mb.const(val=weights[p + "attention.output.LayerNorm.bias"].astype(np.float16), name="l11_ln1b"),
            epsilon=ln_eps,
        )
        wup11 = weights[p + "intermediate.dense.weight"][:, :, None, None].astype(np.float16)
        bup11 = weights[p + "intermediate.dense.bias"].astype(np.float16)
        up11 = mb.conv(x=hc, weight=mb.const(val=wup11, name="l11_wup"), bias=mb.const(val=bup11, name="l11_bup"))
        act11 = mb.gelu(x=up11, mode="EXACT")
        wdn11 = weights[p + "output.dense.weight"][:, :, None, None].astype(np.float16)
        bdn11 = weights[p + "output.dense.bias"].astype(np.float16)
        dn11 = mb.conv(x=act11, weight=mb.const(val=wdn11, name="l11_wdn"), bias=mb.const(val=bdn11, name="l11_bdn"))
        hc = mb.add(x=dn11, y=hc)
        hc = mb.layer_norm(
            x=hc,
            axes=[1],
            gamma=mb.const(val=weights[p + "output.LayerNorm.weight"].astype(np.float16), name="l11_ln2w"),
            beta=mb.const(val=weights[p + "output.LayerNorm.bias"].astype(np.float16), name="l11_ln2b"),
            epsilon=ln_eps,
        )
        return mb.squeeze(x=hc, axes=[2, 3], name="cls_emb")

    return bge_body


_WORKER_NPZ_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def compile_bucket(
    weights: dict[str, np.ndarray],
    ane_dir: Path,
    batch_size: int,
    seq_len: int,
    *,
    layer_norm_eps: float = 1e-12,
) -> Path:
    import coremltools as ct
    import shutil
    import tempfile

    tag = f"bge_b{batch_size}_s{seq_len}_fp16"
    out_mlmodelc = ane_dir / f"{tag}.mlmodelc"

    t0 = time.perf_counter()
    prog = build_bert_ane_mil(weights, batch_size, seq_len, layer_norm_eps=layer_norm_eps)
    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    with tempfile.TemporaryDirectory(prefix=f"{tag}_") as tmp_dir:
        pkg_path = Path(tmp_dir) / f"{tag}.mlpackage"
        mlmodel.save(str(pkg_path))
        compiled_tmp = Path(ct.utils.compile_model(str(pkg_path)))
        try:
            atomic_publish_dir(compiled_tmp, out_mlmodelc, move=True)
        except Exception:
            shutil.rmtree(compiled_tmp, ignore_errors=True)
            raise
    print(f"Built {tag}.mlmodelc in {time.perf_counter() - t0:.2f}s -> {out_mlmodelc}", flush=True)
    return out_mlmodelc


def _compile_bge_bucket_task(task: tuple[str, str, int, int, float]) -> str:
    """Top-level `spawn`-safe worker task compiling one `(B, S)` BGE-small `.mlmodelc` bucket."""
    npz_path_str, ane_dir_str, batch_size, seq_len, layer_norm_eps = task
    npz_p = Path(npz_path_str)
    cache_key = (str(npz_p), npz_p.stat().st_mtime_ns)
    weights = _WORKER_NPZ_CACHE.get(cache_key)
    if weights is None:
        with np.load(npz_p) as npz_obj:
            weights = {k: npz_obj[k] for k in npz_obj.files}
        _WORKER_NPZ_CACHE.clear()
        _WORKER_NPZ_CACHE[cache_key] = weights
    out_path = compile_bucket(
        weights,
        Path(ane_dir_str),
        int(batch_size),
        int(seq_len),
        layer_norm_eps=float(layer_norm_eps),
    )
    return str(out_path)


def _parse_bucket_specs(spec: str) -> list[tuple[int, int]]:
    buckets: list[tuple[int, int]] = []
    for raw in spec.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"Invalid bucket spec {raw!r}; expected '<B>x<S>' (e.g. '1x64,8x256').")
        b_str, s_str = tok.split("x", 1)
        b_val, s_val = int(b_str), int(s_str)
        if b_val < 1 or s_val < 1:
            raise ValueError(f"Bucket dimensions must be >= 1, got {raw!r}")
        buckets.append((b_val, s_val))
    if not buckets:
        raise ValueError("At least one bucket specification is required.")
    return buckets


def build_buckets(
    npz_path: Path,
    ane_dir: Path,
    buckets: Sequence[tuple[int, int]],
    *,
    jobs: int | None = None,
    layer_norm_eps: float = 1e-12,
) -> list[Path]:
    """Compile BGE-small CoreML buckets across `spawn` workers and return paths in deterministic order."""
    ane_dir.mkdir(parents=True, exist_ok=True)
    eff_jobs = default_compile_jobs(jobs)
    tasks = [
        (str(npz_path.resolve()), str(ane_dir.resolve()), int(b), int(s), float(layer_norm_eps))
        for b, s in buckets
    ]
    results = compile_buckets_parallel(
        tasks,
        _compile_bge_bucket_task,
        max_workers=eff_jobs,
        label_fn=lambda t: f"bge_b{t[2]}_s{t[3]}_fp16",
        extra_sys_paths=(Path(__file__).resolve().parent,),
    )
    return [Path(res.value) for res in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile CoreML .mlmodelc ANE buckets.")
    parser.add_argument("--home", type=Path, default=repo_root())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory for compiled .mlmodelc bundles (default: <home>/artifacts/ane_models).",
    )
    parser.add_argument(
        "--include-s256-query",
        action="store_true",
        help="Also build optional bge_b1_s256_fp16.mlmodelc single-query bucket (fills 193..256 gap).",
    )
    parser.add_argument(
        "--buckets",
        default=None,
        help="Optional comma-separated BxS subset, e.g. '1x24,1x64,1x128,1x192'.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"Parallel CoreML worker processes (default: min(4, cpu_count)={default_compile_jobs()}).",
    )
    args = parser.parse_args()

    home = args.home.resolve()
    npz_path = home / "artifacts" / "weights" / "bge_small_f16.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run scripts/export_weights.py first to create {npz_path}")

    ane_dir = args.out_dir.resolve() if args.out_dir is not None else (home / "artifacts" / "ane_models")
    if args.buckets:
        buckets = _parse_bucket_specs(args.buckets)
    else:
        buckets = list(PRODUCTION_BUCKETS)
        if args.include_s256_query:
            buckets.extend(OPTIONAL_QUERY_BUCKETS)

    eff_jobs = default_compile_jobs(args.jobs)
    t0 = time.perf_counter()
    built = build_buckets(npz_path, ane_dir, buckets, jobs=eff_jobs)
    print(
        f"Compiled {len(built)} BGE-small ANE bucket(s) with jobs={min(eff_jobs, len(built))} "
        f"in {time.perf_counter() - t0:.2f}s -> {ane_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
