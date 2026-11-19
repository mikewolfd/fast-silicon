#!/usr/bin/env python3
"""Build CoreML FP16 Neural Engine (.mlmodelc) sequence buckets from artifacts/weights/bge_small_f16.npz.

Requires Python 3.12 + coremltools>=8.0 (only needed if regenerating .mlmodelc bundles).
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

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


def repo_root() -> Path:
    return Path(os.environ.get("FAST_BGE_SMALL_HOME") or Path(__file__).resolve().parent.parent)


def build_bert_ane_mil(weights: dict[str, np.ndarray], batch_size: int, seq_len: int):
    """Build channel-first [B, 384, 1, S] BGE-small body with Layer-11 [CLS] slice."""
    scale_val = np.float16(1.0 / np.sqrt(32.0))

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
        h = mb.layer_norm(x=emb_sum, axes=[1], gamma=emb_ln_w, beta=emb_ln_b, epsilon=np.float16(1e-12))

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
            h = mb.layer_norm(x=h, axes=[1], gamma=ln1w, beta=ln1b, epsilon=np.float16(1e-12))

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
            h = mb.layer_norm(x=h, axes=[1], gamma=ln2w, beta=ln2b, epsilon=np.float16(1e-12))

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
            epsilon=np.float16(1e-12),
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
            epsilon=np.float16(1e-12),
        )
        return mb.squeeze(x=hc, axes=[2, 3], name="cls_emb")

    return bge_body


def compile_bucket(weights: dict[str, np.ndarray], ane_dir: Path, batch_size: int, seq_len: int) -> Path:
    tag = f"bge_b{batch_size}_s{seq_len}_fp16"
    out_mlmodelc = ane_dir / f"{tag}.mlmodelc"
    if out_mlmodelc.exists():
        shutil.rmtree(out_mlmodelc)

    t0 = time.perf_counter()
    prog = build_bert_ane_mil(weights, batch_size, seq_len)
    mlmodel = ct.convert(
        prog,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    compiled_tmp = mlmodel.get_compiled_model_path()
    shutil.copytree(compiled_tmp, out_mlmodelc)
    print(f"Built {tag}.mlmodelc in {time.perf_counter() - t0:.2f}s -> {out_mlmodelc}")
    return out_mlmodelc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile CoreML .mlmodelc ANE buckets.")
    parser.add_argument("--home", type=Path, default=repo_root())
    args = parser.parse_args()

    home = args.home.resolve()
    npz_path = home / "artifacts" / "weights" / "bge_small_f16.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run scripts/export_weights.py first to create {npz_path}")

    ane_dir = home / "artifacts" / "ane_models"
    ane_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(np.load(npz_path))

    for b, s in PRODUCTION_BUCKETS:
        compile_bucket(weights, ane_dir, b, s)


if __name__ == "__main__":
    main()
