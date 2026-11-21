#!/usr/bin/env python3
"""Compile packed ANE (INT8-W8) and CPU-SME (mix-W8) CoreML `.mlmodelc` bundles.

Reads `artifacts/weights/embeddinggemma_2_f16.npz`. Requires `coremltools>=8.0`
(`pip install -e ".[ane-build]"`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
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

FULL_LAYERS = {5, 11, 17, 23}
DEFAULT_ANE_SEQS = (32, 64, 128, 192, 256, 320, 384, 448, 512)
DEFAULT_CPU_SEQS = (256, 384, 512)
CPU_FP32_SKIP_OPS = "reduce_mean,rsqrt,gelu,softmax,add,sub,mul,transpose,reshape,split,concat"
FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)  # 2^-14 = 6.103515625e-5 ANE normal floor
MULTIFUNCTION_BUNDLE_NAME = "emb2_ane_w8.mlmodelc"
_WORKER_NPZ_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def emb2_function_name(seq_len: int) -> str:
    """Canonical multifunction entry name for an EmbeddingGemma-2 ANE sequence bucket."""
    return f"emb2_pk_s{int(seq_len)}"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _compile_and_install(mlmodel, out_dir: Path, name: str) -> Path:
    import coremltools as ct

    out_mlmodelc = out_dir / f"{name}.mlmodelc"
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp_dir:
        pkg = Path(tmp_dir) / f"{name}.mlpackage"
        mlmodel.save(str(pkg))
        compiled_tmp = Path(ct.utils.compile_model(str(pkg)))
        try:
            atomic_publish_dir(compiled_tmp, out_mlmodelc, move=True)
        except Exception:
            shutil.rmtree(compiled_tmp, ignore_errors=True)
            raise
    return out_mlmodelc


def build_ane_packed_w8(S: int, w: dict[str, np.ndarray], out_dir: Path, *, allow_over_512: bool = False) -> Path:
    """Build conv1x1 + W8 ANE-optimized packed sequence bundle (`emb2_pk_s{S}_w8.mlmodelc`, S <= 512)."""
    if int(S) > 512 and not allow_over_512:
        raise ValueError(
            f"ANE EmbeddingGemma-2 buckets must stay <= 512 tokens (got S={S}); S>=1024 wires >2 GB neural memory."
        )
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.optimize.coreml as cto

    H = 512

    def _rmsnorm_ch(x, weight_1xCx1x1, prefix: str):
        eps = np.float16(FP16_MIN_NORMAL)
        msq = mb.reduce_mean(x=mb.mul(x=x, y=x, name=prefix + "_sq"), axes=[1], keep_dims=True, name=prefix + "_msq")
        rsq = mb.rsqrt(x=mb.add(x=msq, y=eps, name=prefix + "_ae"), name=prefix + "_rsq")
        return mb.mul(x=mb.mul(x=x, y=rsq, name=prefix + "_n"), y=weight_1xCx1x1, name=prefix + "_w")

    def _rmsnorm_last(x, weight_opt, prefix: str):
        eps = np.float16(FP16_MIN_NORMAL)
        msq = mb.reduce_mean(x=mb.mul(x=x, y=x, name=prefix + "_sq"), axes=[-1], keep_dims=True, name=prefix + "_msq")
        xn = mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=msq, y=eps, name=prefix + "_ae"), name=prefix + "_rsq"), name=prefix + "_n")
        return mb.mul(x=xn, y=weight_opt, name=prefix + "_w") if weight_opt is not None else xn

    def _rope(x, cos_t, sin_t, prefix: str):
        x0, x1 = mb.split(x=x, num_splits=2, axis=-1, name=prefix + "_sp")
        y0 = mb.sub(x=mb.mul(x=x0, y=cos_t, name=prefix + "_c0"), y=mb.mul(x=x1, y=sin_t, name=prefix + "_s0"), name=prefix + "_y0")
        y1 = mb.add(x=mb.mul(x=x1, y=cos_t, name=prefix + "_c1"), y=mb.mul(x=x0, y=sin_t, name=prefix + "_s1"), name=prefix + "_y1")
        return mb.concat(values=[y0, y1], axis=-1, name=prefix + "_cat")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, S), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, S, S), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 128), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 128), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=ct.converters.mil.mil.types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(h_in, attn_bias, cos_swa, sin_swa, cos_full, sin_full):
        pl = mb.mul(x=mb.conv(x=h_in, weight=w["ple_proj"], name="pl_proj"), y=np.float16(H ** -0.5), name="pl_sc")
        pl4 = mb.reshape(x=pl, shape=[1, 24, H, S], name="pl_r4")
        pl_msq = mb.reduce_mean(x=mb.mul(x=pl4, y=pl4, name="pln_sq"), axes=[2], keep_dims=True, name="pln_msq")
        pl_n = mb.mul(
            x=mb.mul(x=pl4, y=mb.rsqrt(x=mb.add(x=pl_msq, y=np.float16(FP16_MIN_NORMAL), name="pln_ae"), name="pln_rsq"), name="pln_n"),
            y=w["ple_norm"][None, None, :, None],
            name="pln_w",
        )
        pli_chunks = mb.split(x=mb.reshape(x=pl_n, shape=[1, 24 * H, 1, S], name="pln_flat"), num_splits=24, axis=1, name="pl_split")

        cur = h_in
        for i in range(24):
            q = f"l{i}_"
            is_full = i in FULL_LAYERS
            hd = 512 if is_full else 256
            n_kv = 1 if is_full else 2
            kv_rep = 4 if is_full else 2
            cos_t, sin_t = (cos_full, sin_full) if is_full else (cos_swa, sin_swa)

            xn = _rmsnorm_ch(cur, w[q + "in_ln"], q + "in")
            qkv = mb.conv(x=xn, weight=w[q + "qkv"], name=q + "qkv")
            qt, kt, vt = mb.split(x=qkv, split_sizes=[4 * hd, n_kv * hd, n_kv * hd], axis=1, name=q + "spqkv")
            qh = mb.transpose(x=mb.reshape(x=qt, shape=[1, 4, hd, S], name=q + "qr"), perm=[0, 1, 3, 2], name=q + "qt")
            kh = mb.transpose(x=mb.reshape(x=kt, shape=[1, n_kv, hd, S], name=q + "kr"), perm=[0, 1, 3, 2], name=q + "kt")
            vh = mb.transpose(x=mb.reshape(x=vt, shape=[1, n_kv, hd, S], name=q + "vr"), perm=[0, 1, 3, 2], name=q + "vt")

            qn = _rmsnorm_last(qh, w[q + "qn"], q + "qn")
            kn = _rmsnorm_last(kh, w[q + "kn"], q + "kn")
            vn = _rmsnorm_last(vh, None, q + "vn")

            qrp = _rope(qn, cos_t, sin_t, q + "qrp")
            krp = _rope(kn, cos_t, sin_t, q + "krp")
            if n_kv == 1:
                kr4 = mb.tile(x=krp, reps=[1, kv_rep, 1, 1], name=q + "k4")
                vn4 = mb.tile(x=vn, reps=[1, kv_rep, 1, 1], name=q + "v4")
            else:
                kp = mb.split(x=krp, num_splits=n_kv, axis=1, name=q + "ksp")
                vp = mb.split(x=vn, num_splits=n_kv, axis=1, name=q + "vsp")
                kr4 = mb.concat(values=[p for p in kp for _ in range(kv_rep)], axis=1, name=q + "k4")
                vn4 = mb.concat(values=[p for p in vp for _ in range(kv_rep)], axis=1, name=q + "v4")

            qk = mb.matmul(x=qrp, y=kr4, transpose_y=True, name=q + "qk")
            prob = mb.softmax(x=mb.add(x=qk, y=attn_bias, name=q + "qkb"), axis=-1, name=q + "sm")
            att = mb.matmul(x=prob, y=vn4, name=q + "att")
            ao = mb.conv(
                x=mb.reshape(x=mb.transpose(x=att, perm=[0, 1, 3, 2], name=q + "at"), shape=[1, 4 * hd, 1, S], name=q + "ar"),
                weight=w[q + "o"],
                name=q + "o",
            )
            cur = mb.add(x=cur, y=_rmsnorm_ch(ao, w[q + "pa_ln"], q + "pa"), name=q + "res1")

            pre_ff = _rmsnorm_ch(cur, w[q + "pff_ln"], q + "pff")
            g, u = mb.split(x=mb.conv(x=pre_ff, weight=w[q + "gu"], name=q + "gu"), num_splits=2, axis=1, name=q + "spgu")
            md = mb.conv(x=mb.mul(x=mb.gelu(x=g, mode="TANH_APPROXIMATION", name=q + "gelu"), y=u, name=q + "geglu"), weight=w[q + "dn"], name=q + "dn")
            cur = mb.add(x=cur, y=_rmsnorm_ch(md, w[q + "poff_ln"], q + "poff"), name=q + "res2")

            pg = mb.gelu(x=mb.conv(x=cur, weight=w[q + "pg"], name=q + "pg"), mode="TANH_APPROXIMATION", name=q + "pga")
            pp = _rmsnorm_ch(mb.conv(x=mb.mul(x=pg, y=pli_chunks[i], name=q + "pm"), weight=w[q + "pp"], name=q + "pp"), w[q + "pple_ln"], q + "pple")
            cur = mb.mul(x=mb.add(x=cur, y=pp, name=q + "res3"), y=w[q + "scalar"][0], name=q + "lsc")

        normed = _rmsnorm_ch(cur, w["final_norm"], "final_norm")
        # Keep NCHW [1, H, 1, S] so H17 CoreML never schedules a trailing [H,S]->[S,H] transpose onto CPU.
        return mb.identity(x=normed, name="out_h")

    ml = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="h_in", shape=(1, H, 1, S), dtype=np.float16),
            ct.TensorType(name="attn_bias", shape=(1, 1, S, S), dtype=np.float16),
            ct.TensorType(name="cos_swa", shape=(1, 1, S, 128), dtype=np.float16),
            ct.TensorType(name="sin_swa", shape=(1, 1, S, 128), dtype=np.float16),
            ct.TensorType(name="cos_full", shape=(1, 1, S, 256), dtype=np.float16),
            ct.TensorType(name="sin_full", shape=(1, 1, S, 256), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="out_h", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    m_w8 = cto.linear_quantize_weights(
        ml,
        config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        ),
    )
    if out_dir.suffix == ".mlpackage":
        m_w8.save(str(out_dir))
        return out_dir
    return _compile_and_install(m_w8, out_dir, f"emb2_pk_s{S}_w8")


def _load_worker_weights(npz_path_str: str) -> dict[str, np.ndarray]:
    npz_p = Path(npz_path_str)
    cache_key = (str(npz_p), npz_p.stat().st_mtime_ns)
    w = _WORKER_NPZ_CACHE.get(cache_key)
    if w is None:
        with np.load(npz_p) as npz_obj:
            w = {k: npz_obj[k] for k in npz_obj.files}
        _WORKER_NPZ_CACHE.clear()
        _WORKER_NPZ_CACHE[cache_key] = w
    return w


def _save_ane_mlpackage_task(task: tuple[str, str, int]) -> tuple[str, str]:
    npz_path_str, tmp_dir_str, seq_len = task
    w = _load_worker_weights(npz_path_str)
    fn_name = emb2_function_name(seq_len)
    pkg_path = Path(tmp_dir_str) / f"{fn_name}.mlpackage"
    build_ane_packed_w8(int(seq_len), w, pkg_path)
    return fn_name, str(pkg_path)


def _compile_ane_bucket_task(task: tuple[str, str, int]) -> str:
    npz_path_str, out_dir_str, seq_len = task
    w = _load_worker_weights(npz_path_str)
    return str(build_ane_packed_w8(int(seq_len), w, Path(out_dir_str)))


def build_ane_multifunction_w8(
    weights_npz: Path,
    out_dir: Path,
    seq_lens: Sequence[int] = DEFAULT_ANE_SEQS,
    *,
    jobs: int | None = None,
    bundle_name: str = MULTIFUNCTION_BUNDLE_NAME,
) -> Path:
    """Build all W8A16 ANE buckets in parallel and merge into one shared-weight multifunction `.mlmodelc` (~147 MB)."""
    if not seq_lens:
        raise ValueError("At least one sequence length is required.")
    if any(int(s) > 512 for s in seq_lens):
        raise ValueError(
            f"Multifunction ANE seq_lens must stay <= 512 to avoid wiring >2 GB neural memory, got {tuple(seq_lens)}"
        )
    import coremltools as ct
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_jobs = default_compile_jobs(jobs)
    out_mlmodelc = out_dir / bundle_name

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="emb2_mf_stage_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        tasks = [(str(weights_npz.resolve()), str(tmp_root), int(s)) for s in seq_lens]
        results = compile_buckets_parallel(
            tasks,
            _save_ane_mlpackage_task,
            max_workers=eff_jobs,
            label_fn=lambda t: emb2_function_name(t[2]),
            extra_sys_paths=(Path(__file__).resolve().parent,),
        )

        desc = ct.utils.MultiFunctionDescriptor()
        default_fn: str | None = None
        for res in results:
            fn_name, pkg_path_str = res.value
            desc.add_function(pkg_path_str, "main", fn_name)
            if default_fn is None:
                default_fn = fn_name
        assert default_fn is not None
        desc.default_function_name = default_fn

        merged_pkg = tmp_root / "emb2_merged.mlpackage"
        ct.utils.save_multifunction(desc, str(merged_pkg))
        compiled_tmp = Path(ct.utils.compile_model(str(merged_pkg)))
        try:
            atomic_publish_dir(compiled_tmp, out_mlmodelc, move=True)
        except Exception:
            shutil.rmtree(compiled_tmp, ignore_errors=True)
            raise

    for s in seq_lens:
        link_p = out_dir / f"emb2_pk_s{int(s)}_w8.mlmodelc"
        if link_p == out_mlmodelc:
            continue
        if link_p.exists() or link_p.is_symlink():
            if link_p.is_symlink() or link_p.is_file():
                link_p.unlink()
            else:
                shutil.rmtree(link_p)
        os.symlink(out_mlmodelc.name, link_p)

    print(
        f"Built multifunction {bundle_name} ({len(seq_lens)} functions) in {time.perf_counter() - t0:.2f}s -> {out_mlmodelc}",
        flush=True,
    )
    return out_mlmodelc


def build_cpu_mix_w8(S: int, w: dict[str, np.ndarray], out_dir: Path) -> Path:
    """Build BNNS/SME matmul + INT8-W8 mixed-precision bundle (`emb2_cpu_s{S}_mix_w8.mlmodelc`)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import coremltools.optimize.coreml as cto

    def _rmsnorm_mm32(x, w_arr, eps=1e-6, name="rn"):
        sq = mb.mul(x=x, y=x, name=f"{name}_sq")
        ms = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{name}_ms")
        rs = mb.rsqrt(x=mb.add(x=ms, y=np.float32(eps), name=f"{name}_ae"), name=f"{name}_rs")
        n = mb.mul(x=x, y=rs, name=f"{name}_n")
        return mb.mul(x=n, y=mb.const(val=w_arr.astype(np.float32), name=f"{name}_w"), name=f"{name}_nw") if w_arr is not None else n

    def _rope32(x4, c, s, name="rope"):
        x0, x1 = mb.split(x=x4, num_splits=2, axis=-1, name=f"{name}_sp")
        y0 = mb.sub(x=mb.mul(x=x0, y=c, name=f"{name}_xc0"), y=mb.mul(x=x1, y=s, name=f"{name}_xs0"), name=f"{name}_y0")
        y1 = mb.add(x=mb.mul(x=x1, y=c, name=f"{name}_xc1"), y=mb.mul(x=x0, y=s, name=f"{name}_xs1"), name=f"{name}_y1")
        return mb.concat(values=[y0, y1], axis=-1, name=f"{name}_cat")

    def _lin32(x, w4d, name: str):
        w2d = np.ascontiguousarray(w4d[:, :, 0, 0].T.astype(np.float32))
        return mb.matmul(x=x, y=mb.const(val=w2d, name=f"{name}_w"), name=name)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 512, 1, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 128), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 128), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(h_in, attn_bias, cos_swa, sin_swa, cos_full, sin_full):
        h = mb.cast(
            x=mb.transpose(x=mb.reshape(x=h_in, shape=[1, 512, S], name="in_r3"), perm=[0, 2, 1], name="in_t"),
            dtype="fp32",
            name="h32",
        )
        bias_32 = mb.cast(x=attn_bias, dtype="fp32", name="b32")
        cs_s, ss_s = mb.cast(x=cos_swa, dtype="fp32", name="cs32"), mb.cast(x=sin_swa, dtype="fp32", name="ss32")
        cf_s, sf_s = mb.cast(x=cos_full, dtype="fp32", name="cf32"), mb.cast(x=sin_full, dtype="fp32", name="sf32")

        ple_w = w["ple_proj"].astype(np.float32) * (512.0 ** -0.5)
        pl = _lin32(h, ple_w, "pl")
        pln = _rmsnorm_mm32(mb.reshape(x=pl, shape=[1, S, 24, 512], name="pl4"), w["ple_norm"].reshape(1, 1, 1, 512), name="pln")
        pl_layers = mb.split(x=pln, num_splits=24, axis=2, name="plsp")

        for i in range(24):
            p = f"l{i}"
            is_full = i in FULL_LAYERS
            hd = 512 if is_full else 256
            n_kv = 1 if is_full else 2
            q_dim, kv_dim = 4 * hd, n_kv * hd

            qkv = _lin32(_rmsnorm_mm32(h, w[f"{p}_in_ln"].reshape(1, 1, 512), name=f"{p}_in"), w[f"{p}_qkv"], f"{p}_qkv")
            q_r, k_r, v_r = mb.split(x=qkv, split_sizes=[q_dim, kv_dim, kv_dim], axis=-1, name=f"{p}_qkvsp")
            q4 = mb.transpose(x=mb.reshape(x=q_r, shape=[1, S, 4, hd], name=f"{p}_qsh"), perm=[0, 2, 1, 3], name=f"{p}_qt")
            k4 = mb.transpose(x=mb.reshape(x=k_r, shape=[1, S, n_kv, hd], name=f"{p}_ksh"), perm=[0, 2, 1, 3], name=f"{p}_kt")
            v4 = mb.transpose(x=mb.reshape(x=v_r, shape=[1, S, n_kv, hd], name=f"{p}_vsh"), perm=[0, 2, 1, 3], name=f"{p}_vt")
            q4 = _rmsnorm_mm32(q4, w[f"{p}_qn"], name=f"{p}_qn")
            k4 = _rmsnorm_mm32(k4, w[f"{p}_kn"], name=f"{p}_kn")
            v4 = _rmsnorm_mm32(v4, None, name=f"{p}_vn")
            c_cur, s_cur = (cf_s, sf_s) if is_full else (cs_s, ss_s)
            q4, k4 = _rope32(q4, c_cur, s_cur, name=f"{p}_qrp"), _rope32(k4, c_cur, s_cur, name=f"{p}_krp")

            if n_kv == 1:
                k_gqa = mb.concat(values=[k4, k4, k4, k4], axis=1, name=f"{p}_kg")
                v_gqa = mb.concat(values=[v4, v4, v4, v4], axis=1, name=f"{p}_vg")
            else:
                k0, k1 = mb.split(x=k4, num_splits=2, axis=1, name=f"{p}_ksp")
                v0, v1 = mb.split(x=v4, num_splits=2, axis=1, name=f"{p}_vsp")
                k_gqa = mb.concat(values=[k0, k0, k1, k1], axis=1, name=f"{p}_kg")
                v_gqa = mb.concat(values=[v0, v0, v1, v1], axis=1, name=f"{p}_vg")

            qk = mb.matmul(x=q4, y=k_gqa, transpose_x=False, transpose_y=True, name=f"{p}_qk")
            sm = mb.softmax(x=mb.add(x=qk, y=bias_32, name=f"{p}_qkb"), axis=-1, name=f"{p}_sm")
            att = mb.matmul(x=sm, y=v_gqa, name=f"{p}_att")
            ar = mb.reshape(x=mb.transpose(x=att, perm=[0, 2, 1, 3], name=f"{p}_at"), shape=[1, S, q_dim], name=f"{p}_ar")
            h = mb.add(x=h, y=_rmsnorm_mm32(_lin32(ar, w[f"{p}_o"], f"{p}_o"), w[f"{p}_pa_ln"].reshape(1, 1, 512), name=f"{p}_paln"), name=f"{p}_r1")

            gu = _lin32(_rmsnorm_mm32(h, w[f"{p}_pff_ln"].reshape(1, 1, 512), name=f"{p}_pff"), w[f"{p}_gu"], f"{p}_gu")
            gate, up = mb.split(x=gu, num_splits=2, axis=-1, name=f"{p}_gusp")
            dn = _lin32(mb.mul(x=mb.gelu(x=gate, mode="TANH_APPROXIMATION", name=f"{p}_g"), y=up, name=f"{p}_geglu"), w[f"{p}_dn"], f"{p}_dn")
            h = mb.add(x=h, y=_rmsnorm_mm32(dn, w[f"{p}_poff_ln"].reshape(1, 1, 512), name=f"{p}_poff"), name=f"{p}_r2")

            pg = _lin32(h, w[f"{p}_pg"], f"{p}_pg")
            pp = _lin32(mb.mul(x=mb.gelu(x=pg, mode="TANH_APPROXIMATION", name=f"{p}_pgop"), y=mb.reshape(x=pl_layers[i], shape=[1, S, 512], name=f"{p}_pli"), name=f"{p}_pm"), w[f"{p}_pp"], f"{p}_pp")
            h = mb.add(x=h, y=_rmsnorm_mm32(pp, w[f"{p}_pple_ln"].reshape(1, 1, 512), name=f"{p}_pple"), name=f"{p}_r3")
            h = mb.mul(x=h, y=mb.const(val=w[f"{p}_scalar"].astype(np.float32).reshape(1, 1, 1), name=f"{p}_sc"), name=f"{p}_lsc")

        fn = _rmsnorm_mm32(h, w["final_norm"].reshape(1, 1, 512), name="final_norm")
        return mb.cast(x=mb.reshape(x=fn, shape=[S, 512], name="out_r2"), dtype="fp16", name="out_h")

    pp = ct.PassPipeline.DEFAULT
    pp.set_options("common::add_fp16_cast", {"skip_ops_by_type": CPU_FP32_SKIP_OPS})
    mlm = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        pass_pipeline=pp,
        minimum_deployment_target=ct.target.macOS15,
    )
    qcfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    mlm = cto.linear_quantize_weights(mlm, qcfg)
    return _compile_and_install(mlm, out_dir, f"emb2_cpu_s{S}_mix_w8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EmbeddingGemma-2 W8A16 multifunction ANE CoreML bundle.")
    parser.add_argument("--home", type=Path, default=repo_root())
    parser.add_argument("--mode", choices=("ane", "all", "cpu"), default="ane")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None)
    parser.add_argument(
        "--separate-buckets",
        action="store_true",
        help="Emit individual emb2_pk_s{S}_w8.mlmodelc directories instead of unified emb2_ane_w8.mlmodelc.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"Parallel CoreML worker processes (default: min(4, cpu_count)={default_compile_jobs()}).",
    )
    args = parser.parse_args()

    home = args.home.resolve()
    weights_npz = home / "artifacts" / "weights" / "embeddinggemma_2_f16.npz"
    out_dir = home / "artifacts" / "ane_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_jobs = default_compile_jobs(args.jobs)

    if args.mode in ("all", "ane"):
        seqs = tuple(args.seq_lens) if args.seq_lens else DEFAULT_ANE_SEQS
        if args.separate_buckets:
            tasks = [(str(weights_npz.resolve()), str(out_dir.resolve()), int(s)) for s in seqs]
            results = compile_buckets_parallel(
                tasks,
                _compile_ane_bucket_task,
                max_workers=eff_jobs,
                label_fn=lambda t: f"emb2_pk_s{t[2]}_w8",
                extra_sys_paths=(Path(__file__).resolve().parent,),
            )
            for res in results:
                print(f"Built {res.value}")
        else:
            build_ane_multifunction_w8(weights_npz, out_dir, seqs, jobs=eff_jobs)

    if args.mode in ("all", "cpu"):
        w = dict(np.load(weights_npz))
        seqs = tuple(args.seq_lens) if args.seq_lens else DEFAULT_CPU_SEQS
        for s in seqs:
            dst = build_cpu_mix_w8(s, w, out_dir)
            print(f"Built {dst}")


if __name__ == "__main__":
    main()
