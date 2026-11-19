"""Compile CoreML ANE `gemma4_early{14,23}_prefix_S256_{fp16,w8}.mlmodelc` bundles.

Uses per-output-channel symmetric `int8` (`cto.coreml.linear_quantize_weights` with
`mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048`)
and optional closed-form `AWQ-MSE` activation-weighted channel clipping (`--awq-mse`, zero training).

Hardware compiler invariants (`H16ANE` / `macOS 15+`):
  - NEVER use `palettize_weights(mode="kmeans", nbits=8)` (global per-tensor 256-entry LUT crushes K/V SNR by ~5 dB).
  - NEVER use `granularity="per_block"` (`E5RT: ANECCompile() FAILED` + `MLComputePlan` XPC hang).
  - NEVER mix `S=1` spatial slices inside the static `S=256` MIL attention graph (`ANECCompile() FAILED`).
  - Cut strictly at `k = first_kv_shared` (`EarlyExit14` `k=15` on `E2B`, `EarlyExit23` `k=24` on `E4B`).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterator

import numpy as np

DEFAULT_ANE_CHUNK_S = 256
DEFAULT_ANE_P_SWA = 512
DEFAULT_ANE_P_FULL = 1920
AWQ_ALPHA_GRID: tuple[float, ...] = (0.86, 0.89, 0.92, 0.94, 0.96, 0.975, 0.99, 1.00)
GPU_LOCK_PATH = "/tmp/m4_gpu_validation.lock"


@contextmanager
def _gpu_validation_lock(lock_path: str = GPU_LOCK_PATH) -> Iterator[None]:
    with open(lock_path, "w", encoding="utf-8") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def infer_ane_arch_from_npz(npz: dict[str, np.ndarray]) -> dict[str, Any]:
    """Infer `(first_kv_shared, hidden_size, ple_dim, n_kv, layer_specs)` directly from exported `.npz` shapes."""
    k = 0
    while f"l{k}_w_in" in npz:
        k += 1
    if k < 2:
        raise ValueError(f"Expected >= 2 EarlyExit layers in weights npz, found k={k}")

    hidden_size = int(npz["l0_w_in"].shape[1])
    ple_dim = int(npz["per_layer_proj_norm_w"].shape[0])
    n_pli_layers = k - 1
    if int(npz["per_layer_model_proj_w"].shape[0]) != n_pli_layers * ple_dim:
        raise ValueError(
            f"Mismatched per_layer_model_proj_w shape {npz['per_layer_model_proj_w'].shape} "
            f"for n_pli_layers={n_pli_layers}, ple_dim={ple_dim}"
        )

    layers: list[dict[str, Any]] = []
    n_swa = 0
    n_full = 0
    n_kv_global: int | None = None
    for i in range(k):
        p = f"l{i}_"
        hd = int(npz[p + "w_kn"].shape[-1])
        is_full = hd == 512
        if hd not in (256, 512):
            raise ValueError(f"Unsupported head_dim={hd} at layer {i} (expected 256 SWA or 512 Full)")
        qdim = 8 * hd
        if i == k - 1:
            kv_w = npz[p + "kv"] if (p + "kv") in npz else npz[p + "qkv"][qdim:]
            kv_out = int(kv_w.shape[0])
        else:
            qkv_out = int(npz[p + "qkv"].shape[0])
            kv_out = qkv_out - qdim
        if kv_out <= 0 or kv_out % (2 * hd) != 0:
            raise ValueError(f"Invalid KV output channels={kv_out} for head_dim={hd} at layer {i}")
        n_kv = kv_out // (2 * hd)
        if n_kv_global is None:
            n_kv_global = n_kv
        elif n_kv != n_kv_global:
            raise ValueError(f"Heterogeneous n_kv ({n_kv} vs {n_kv_global}) unsupported in packed ANE KV tensor")
        if is_full:
            n_full += 1
        else:
            n_swa += 1
        layers.append({"idx": i, "is_full": is_full, "hd": hd, "qdim": qdim, "n_kv": n_kv})

    assert n_kv_global is not None
    if n_full < 1:
        raise ValueError("Expected at least 1 full_attention layer in EarlyExit trunk")
    return {
        "first_kv_shared": k,
        "exit_layer": k - 1,
        "hidden_size": hidden_size,
        "ple_dim": ple_dim,
        "n_pli_layers": n_pli_layers,
        "n_swa": n_swa,
        "n_full": n_full,
        "n_full_past": n_full - 1,
        "n_kv": n_kv_global,
        "layers": layers,
    }


def collect_awq_calibration_col_sq(
    model: Any,
    *,
    chunk_s: int = DEFAULT_ANE_CHUNK_S,
) -> dict[str, np.ndarray]:
    """Collect per-input-channel `E[x^2]` across 8 legislative calibration chunks in <0.2s on GPU (zero training)."""
    import mlx.core as mx

    from fast_gemma4.kernels import (
        fused_geglu_split,
        fused_norm_add_norm,
        fused_ple_geglu_indexed,
        fused_qkv_packed_norm_rope,
        fused_res_rms_norm,
    )
    from fast_gemma4.model import create_causal_mask
    from fast_gemma4.tokenizer import FastGemma4Tokenizer

    tok = FastGemma4Tokenizer(model.tokenizer_json)
    calib_texts = [
        f"Pacific Transmission Consortium Technical Report #{c_i}: Engineering assessment regarding "
        "substation thermal ratings, battery inverter stability, statutory compliance deadlines, and grid reliability. " * 8
        for c_i in range(8)
    ]
    sys_ids = [tok.bos_token_id] + (
        tok.encode("Extract JSON metadata for this technical document:\n\n", add_special_tokens=False) * 20
    )
    sys_ids = sys_ids[:600]
    pkv = model.prefill_prefix(sys_ids)

    sufs = np.zeros((len(calib_texts), chunk_s), dtype=np.int32)
    for idx, txt in enumerate(calib_texts):
        enc = tok.encode(txt, add_special_tokens=False)[:chunk_s]
        sufs[idx, : len(enc)] = enc

    tok_mx = mx.array(sufs, dtype=mx.int32)
    B, S = int(tok_mx.shape[0]), int(tok_mx.shape[1])
    l_sys = int(pkv["l_sys"])
    k_sys_l, v_sys_l = pkv["k_sys"], pkv["v_sys"]
    off = l_sys
    off_meta = mx.array([off, S], dtype=mx.int32)
    col_sq: dict[str, np.ndarray] = {}

    def _rec(name: str, x: mx.array) -> None:
        xf = x.astype(mx.float32).reshape(-1, x.shape[-1])
        msq = mx.mean(xf * xf, axis=0)
        mx.eval(msq)
        col_sq[name] = np.array(msq, copy=True)

    h, pli = model._compute_embed_and_pli_early14(tok_mx)
    _rec("per_layer_model_proj_w", h)
    x_in = mx.fast.rms_norm(h, model.layers[0]["w_in"], model.eps)
    masks: dict[tuple[str, int, int], Any] = {}
    for i in range(model.first_kv_shared):
        l = model.layers[i]
        p = f"l{i}_"
        D = l["head_dim"]
        if i == model.first_kv_shared - 1:
            _rec(p + "kv", x_in)
            break
        _rec(p + "qkv", x_in)
        qkv = l["qkv_proj"](x_in)
        q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
            qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, model.eps
        )
        if l["is_full"]:
            kf = mx.concatenate([mx.broadcast_to(k_sys_l[i][None, None], (B, 1, l_sys, D)), k_rot[:, None]], axis=2)
            vf = mx.concatenate([mx.broadcast_to(v_sys_l[i][None, None], (B, 1, l_sys, D)), v_norm[:, None]], axis=2)
            mk = ("full", off, S)
            if mk not in masks:
                masks[mk] = create_causal_mask(S, offset=off, window_size=None)
        else:
            t_start = max(0, off - model.window_size + 1)
            kf = mx.concatenate(
                [mx.broadcast_to(k_sys_l[i][t_start:l_sys][None, None], (B, 1, l_sys - t_start, D)), k_rot[:, None]],
                axis=2,
            )
            vf = mx.concatenate(
                [mx.broadcast_to(v_sys_l[i][t_start:l_sys][None, None], (B, 1, l_sys - t_start, D)), v_norm[:, None]],
                axis=2,
            )
            mk = ("swa", off - t_start, S)
            if mk not in masks:
                masks[mk] = create_causal_mask(S, offset=(off - t_start), window_size=model.window_size)
        att = (
            mx.fast.scaled_dot_product_attention(q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk])
            .transpose(0, 2, 1, 3)
            .reshape(B, S, 8 * D)
        )
        _rec(p + "wo", att)
        ao = l["o_proj"](att)
        h_mid, mlp_in = fused_norm_add_norm(h, ao, l["w_post_attn"], l["w_pre_ff"], model.eps, None)
        _rec(p + "gu", mlp_in)
        mlp_act = fused_geglu_split(l["gate_up_proj"](mlp_in))
        _rec(p + "dn", mlp_act)
        mlp_out = model._run_mlp(mlp_in, l)
        h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], model.eps)
        _rec(p + "w_pg", h_ff)
        ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, model._layer_idx_mx[i])
        _rec(p + "w_pp", ple_g)
        ple_out = l["ple_proj"](ple_g)
        h, x_in = fused_norm_add_norm(
            h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], model.eps, l["layer_scalar"]
        )
    return col_sq


def apply_awq_mse_clipping(
    npz: dict[str, np.ndarray],
    act_col_sq: dict[str, np.ndarray],
    *,
    alphas: tuple[float, ...] = AWQ_ALPHA_GRID,
) -> tuple[dict[str, np.ndarray], float]:
    """Closed-form per-output-channel symmetric int8 AWQ-MSE clipping (`0` gradient/training steps)."""
    clipped = dict(npz)
    alpha_means: list[float] = []
    for key, w_raw in npz.items():
        if key not in act_col_sq or w_raw.ndim != 4:
            continue
        w = w_raw[:, :, 0, 0].astype(np.float32)
        col_w = act_col_sq[key].astype(np.float32)[None, :]
        if col_w.shape[1] != w.shape[1]:
            continue
        amax = np.maximum(np.max(np.abs(w), axis=1, keepdims=True), 1e-8)
        best_alpha = np.ones_like(amax)
        best_err = np.full((w.shape[0], 1), np.inf, dtype=np.float32)
        for alpha in alphas:
            sc = (float(alpha) * amax) / 127.0
            wq = np.clip(np.round(w / sc), -127.0, 127.0) * sc
            diff = w - wq
            err = np.sum((diff * diff) * col_w, axis=1, keepdims=True)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_alpha = np.where(better, float(alpha), best_alpha)
        lim = best_alpha * amax
        w_clipped = np.clip(w, -lim, lim).astype(np.float16)
        clipped[key] = np.ascontiguousarray(w_clipped)[:, :, None, None]
        alpha_means.append(float(np.mean(best_alpha)))
    mean_alpha = float(np.mean(alpha_means)) if alpha_means else 1.0
    return clipped, mean_alpha


def build_early_prefix_mil_program(
    npz: dict[str, np.ndarray],
    *,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
) -> tuple[Any, dict[str, Any]]:
    """Build CoreML MIL `EarlyExit{k-1}` program (`k = first_kv_shared`: `15` on E2B, `24` on E4B)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if S <= 1:
        raise ValueError(f"H16ANE requires static chunk S >= 16 (got S={S}); S=1 slices fail ANECCompile().")

    arch = infer_ane_arch_from_npz(npz)
    k = arch["first_kv_shared"]
    H = arch["hidden_size"]
    ple_dim = arch["ple_dim"]
    n_pli = arch["n_pli_layers"]
    n_swa = arch["n_swa"]
    n_full_past = arch["n_full_past"]
    n_kv = arch["n_kv"]
    layers = arch["layers"]

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n_pli * ple_dim, 1, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(n_swa, n_kv, P_SWA, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(n_swa, n_kv, P_SWA, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(n_full_past, n_kv, P_FULL, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(n_full_past, n_kv, P_FULL, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, P_SWA + S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, P_FULL + S), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(
        h_in,
        pli_raw,
        cos_swa,
        sin_swa,
        cos_full,
        sin_full,
        k_past_swa,
        v_past_swa,
        k_past_full,
        v_past_full,
        mask_swa,
        mask_full,
    ):
        def _rmsnorm_ch(x, w1xc, prefix: str):
            msq = mb.reduce_mean(x=mb.mul(x=x, y=x, name=prefix + "_sq"), axes=[1], keep_dims=True, name=prefix + "_msq")
            rsq = mb.rsqrt(x=mb.add(x=msq, y=np.float16(1e-6), name=prefix + "_ae"), name=prefix + "_rsq")
            return mb.mul(x=mb.mul(x=x, y=rsq, name=prefix + "_n"), y=w1xc, name=prefix + "_w")

        def _rmsnorm_last(x, w, prefix: str):
            msq = mb.reduce_mean(x=mb.mul(x=x, y=x, name=prefix + "_sq"), axes=[-1], keep_dims=True, name=prefix + "_msq")
            rsq = mb.rsqrt(x=mb.add(x=msq, y=np.float16(1e-6), name=prefix + "_ae"), name=prefix + "_rsq")
            xn = mb.mul(x=x, y=rsq, name=prefix + "_n")
            return mb.mul(x=xn, y=w, name=prefix + "_w") if w is not None else xn

        def _rope(x, c, s, half_d: int, prefix: str):
            x1, x2 = mb.split(x=x, split_sizes=[half_d, half_d], axis=-1, name=prefix + "_sp")
            rot = mb.concat(
                values=[mb.mul(x=x2, y=np.float16(-1.0), name=prefix + "_neg"), x1], axis=-1, name=prefix + "_cat"
            )
            return mb.add(
                x=mb.mul(x=x, y=c, name=prefix + "_xc"),
                y=mb.mul(x=rot, y=s, name=prefix + "_rs"),
                name=prefix + "_out",
            )

        plm = mb.mul(
            x=mb.conv(x=h_in, weight=npz["per_layer_model_proj_w"], name="plm"),
            y=npz["per_layer_proj_scale"][0],
            name="plms",
        )
        plm_r = mb.reshape(x=plm, shape=[1, n_pli, ple_dim, S], name="plmr")
        pl_msq = mb.reduce_mean(x=mb.mul(x=plm_r, y=plm_r, name="plsq"), axes=[2], keep_dims=True, name="plmsq")
        pl_n = mb.mul(
            x=plm_r,
            y=mb.rsqrt(x=mb.add(x=pl_msq, y=np.float16(1e-6), name="plae"), name="plrsq"),
            name="pln",
        )
        pl_nw = mb.mul(x=pl_n, y=npz["per_layer_proj_norm_w"].reshape(1, 1, ple_dim, 1), name="plnw")
        pl_add = mb.add(x=mb.reshape(x=pl_nw, shape=[1, n_pli * ple_dim, 1, S], name="plnw_f"), y=pli_raw, name="pladd")
        pli_all = mb.mul(x=pl_add, y=npz["per_layer_input_scale"][0], name="plall")
        plis = mb.split(x=pli_all, split_sizes=[ple_dim] * n_pli, axis=1, name="plisp")

        k_past_swa_list = (
            [k_past_swa]
            if n_swa == 1
            else list(mb.split(x=k_past_swa, split_sizes=[1] * n_swa, axis=0, name="kps_sp"))
        )
        v_past_swa_list = (
            [v_past_swa]
            if n_swa == 1
            else list(mb.split(x=v_past_swa, split_sizes=[1] * n_swa, axis=0, name="vps_sp"))
        )
        k_past_full_list = (
            [k_past_full]
            if n_full_past == 1
            else list(mb.split(x=k_past_full, split_sizes=[1] * n_full_past, axis=0, name="kpf_sp"))
        )
        v_past_full_list = (
            [v_past_full]
            if n_full_past == 1
            else list(mb.split(x=v_past_full, split_sizes=[1] * n_full_past, axis=0, name="vpf_sp"))
        )

        cur = h_in
        k_swas, v_swas, k_fulls, v_fulls = [], [], [], []
        swa_idx = 0
        full_idx = 0

        for lspec in layers:
            i = lspec["idx"]
            p = f"l{i}_"
            is_full = lspec["is_full"]
            hd = lspec["hd"]
            qdim = lspec["qdim"]
            kv_dim = n_kv * hd
            in_n = _rmsnorm_ch(cur, npz[p + "w_in"], p + "in")

            if i == k - 1:
                kv_w = npz[p + "kv"] if (p + "kv") in npz else npz[p + "qkv"][qdim:]
                kv = mb.conv(x=in_n, weight=kv_w, name=p + "kv")
                k_t, v_t = mb.split(x=kv, split_sizes=[kv_dim, kv_dim], axis=1, name=p + "sp")
            else:
                qkv = mb.conv(x=in_n, weight=npz[p + "qkv"], name=p + "qkv")
                q, k_t, v_t = mb.split(x=qkv, split_sizes=[qdim, kv_dim, kv_dim], axis=1, name=p + "sp")

            kr = mb.transpose(
                x=mb.reshape(x=k_t, shape=[1, n_kv, hd, S], name=p + "kr"), perm=[0, 1, 3, 2], name=p + "kh"
            )
            vr = mb.transpose(
                x=mb.reshape(x=v_t, shape=[1, n_kv, hd, S], name=p + "vr"), perm=[0, 1, 3, 2], name=p + "vh"
            )
            kn = _rope(
                _rmsnorm_last(kr, npz[p + "w_kn"], p + "kn"),
                cos_full if is_full else cos_swa,
                sin_full if is_full else sin_swa,
                hd // 2,
                p + "rk",
            )
            vn = _rmsnorm_last(vr, None, p + "vn")
            if is_full:
                k_fulls.append(kn)
                v_fulls.append(vn)
            else:
                k_swas.append(kn)
                v_swas.append(vn)
            if i == k - 1:
                break

            if is_full:
                k_tot = mb.concat(values=[k_past_full_list[full_idx], kn], axis=2, name=p + "kcat")
                v_tot = mb.concat(values=[v_past_full_list[full_idx], vn], axis=2, name=p + "vcat")
                m_use = mask_full
                full_idx += 1
            else:
                k_tot = mb.concat(values=[k_past_swa_list[swa_idx], kn], axis=2, name=p + "kcat")
                v_tot = mb.concat(values=[v_past_swa_list[swa_idx], vn], axis=2, name=p + "vcat")
                m_use = mask_swa
                swa_idx += 1

            qr = mb.transpose(
                x=mb.reshape(
                    x=mb.transpose(x=q, perm=[0, 2, 3, 1], name=p + "qt"),
                    shape=[1, S, 8, hd],
                    name=p + "qr",
                ),
                perm=[0, 2, 1, 3],
                name=p + "qh",
            )
            qn = _rope(
                _rmsnorm_last(qr, npz[p + "w_qn"], p + "qn"),
                cos_full if is_full else cos_swa,
                sin_full if is_full else sin_swa,
                hd // 2,
                p + "rq",
            )
            if n_kv == 1:
                sc = mb.add(
                    x=mb.matmul(x=qn, y=mb.transpose(x=k_tot, perm=[0, 1, 3, 2], name=p + "kt"), name=p + "qk"),
                    y=m_use,
                    name=p + "m",
                )
                att = mb.matmul(x=mb.softmax(x=sc, axis=-1, name=p + "sm"), y=v_tot, name=p + "av")
            else:
                q_per_kv = 8 // n_kv
                q_groups = mb.split(x=qn, split_sizes=[q_per_kv] * n_kv, axis=1, name=p + "qg_sp")
                k_groups = mb.split(x=k_tot, split_sizes=[1] * n_kv, axis=1, name=p + "kg_sp")
                v_groups = mb.split(x=v_tot, split_sizes=[1] * n_kv, axis=1, name=p + "vg_sp")
                att_groups = []
                for g_i in range(n_kv):
                    gp = f"{p}g{g_i}_"
                    sc_g = mb.add(
                        x=mb.matmul(
                            x=q_groups[g_i],
                            y=mb.transpose(x=k_groups[g_i], perm=[0, 1, 3, 2], name=gp + "kt"),
                            name=gp + "qk",
                        ),
                        y=m_use,
                        name=gp + "m",
                    )
                    att_groups.append(
                        mb.matmul(x=mb.softmax(x=sc_g, axis=-1, name=gp + "sm"), y=v_groups[g_i], name=gp + "av")
                    )
                att = mb.concat(values=att_groups, axis=1, name=p + "av_cat")

            att_ch = mb.transpose(
                x=mb.reshape(
                    x=mb.transpose(x=att, perm=[0, 2, 1, 3], name=p + "at"),
                    shape=[1, 1, S, qdim],
                    name=p + "ar",
                ),
                perm=[0, 3, 1, 2],
                name=p + "ac",
            )
            ao_n = _rmsnorm_ch(mb.conv(x=att_ch, weight=npz[p + "wo"], name=p + "wo"), npz[p + "w_pa"], p + "pa")
            c1 = mb.add(x=cur, y=ao_n, name=p + "res1")
            pff = _rmsnorm_ch(c1, npz[p + "w_pff"], p + "pff")
            g, u = mb.split(x=mb.conv(x=pff, weight=npz[p + "gu"], name=p + "gu"), num_splits=2, axis=1, name=p + "spgu")
            md = mb.conv(
                x=mb.mul(x=mb.gelu(x=g, mode="TANH_APPROXIMATION", name=p + "gelu"), y=u, name=p + "geglu"),
                weight=npz[p + "dn"],
                name=p + "dn",
            )
            c2 = mb.add(x=c1, y=_rmsnorm_ch(md, npz[p + "w_poff"], p + "poff"), name=p + "res2")
            w_pg = npz[p + "w_pg"] if (p + "w_pg") in npz else npz[p + "pg"]
            w_pp = npz[p + "w_pp"] if (p + "w_pp") in npz else npz[p + "pp"]
            pg = mb.conv(x=c2, weight=w_pg, name=p + "pg")
            pge = mb.mul(x=mb.gelu(x=pg, mode="TANH_APPROXIMATION", name=p + "pgelu"), y=plis[i], name=p + "pmul")
            cur = mb.mul(
                x=mb.add(
                    x=c2,
                    y=_rmsnorm_ch(mb.conv(x=pge, weight=w_pp, name=p + "pp"), npz[p + "w_pple"], p + "pple"),
                    name=p + "res3",
                ),
                y=npz[p + "scalar"][0],
                name=p + "sc",
            )

        return (
            mb.identity(x=mb.concat(values=k_swas, axis=0, name="ks_c"), name="k_swa"),
            mb.identity(x=mb.concat(values=v_swas, axis=0, name="vs_c"), name="v_swa"),
            mb.identity(x=mb.concat(values=k_fulls, axis=0, name="kf_c"), name="k_full"),
            mb.identity(x=mb.concat(values=v_fulls, axis=0, name="vf_c"), name="v_full"),
        )

    return prog, arch


def quantize_mlmodel_linear_w8_per_channel(ml_fp16: Any, *, weight_threshold: int = 2048) -> Any:
    """Quantize CoreML MLProgram conv1x1 weights to per-output-channel symmetric `int8` (`100%` H16ANE compatible)."""
    import coremltools.optimize as cto

    return cto.coreml.linear_quantize_weights(
        ml_fp16,
        config=cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",
                weight_threshold=weight_threshold,
            )
        ),
    )


def _inspect_compiled_bundle(
    lib_path: Path,
    mc_path: Path,
    *,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
) -> dict[str, int] | None:
    if not lib_path.exists():
        return None
    lib = ctypes.CDLL(str(lib_path))
    if not hasattr(lib, "gemma4_px_inspect_plan_ex"):
        return None
    mc_bytes = str(mc_path).encode("utf-8")
    # Warm-load once so ANECompilerService populates the E5RT cache before the 8s MLComputePlan guard.
    if hasattr(lib, "gemma4_px_load") and hasattr(lib, "gemma4_px_free"):
        lib.gemma4_px_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.gemma4_px_load.restype = ctypes.c_void_p
        lib.gemma4_px_free.argtypes = [ctypes.c_void_p]
        lib.gemma4_px_free.restype = None
        h = lib.gemma4_px_load(mc_bytes, int(S), int(P_SWA), int(P_FULL), 1)
        if not h:
            raise RuntimeError(f"Failed to load compiled CoreML bundle on ANE: {mc_path}")
        lib.gemma4_px_free(h)

    i32p = ctypes.POINTER(ctypes.c_int)
    lib.gemma4_px_inspect_plan_ex.argtypes = [ctypes.c_char_p, ctypes.c_int, i32p, i32p, i32p, i32p]
    lib.gemma4_px_inspect_plan_ex.restype = ctypes.c_int
    ane = ctypes.c_int(0)
    cpu_rt = ctypes.c_int(0)
    cpu_ce = ctypes.c_int(0)
    gpu = ctypes.c_int(0)
    rc = -1
    for attempt in range(2):
        rc = lib.gemma4_px_inspect_plan_ex(
            mc_bytes,
            0,
            ctypes.byref(ane),
            ctypes.byref(cpu_rt),
            ctypes.byref(cpu_ce),
            ctypes.byref(gpu),
        )
        if rc >= 0:
            break
        if attempt == 0:
            time.sleep(2.0)
    if rc < 0:
        raise RuntimeError(f"MLComputePlan inspection failed or timed out for {mc_path}")
    return {
        "ane_ops": ane.value,
        "cpu_runtime_ops": cpu_rt.value,
        "cpu_constexpr_ops": cpu_ce.value,
        "gpu_ops": gpu.value,
    }


def compile_ane_bundles_from_npz(
    *,
    fp16_npz_path: Path,
    w8_npz_path: Path,
    out_ane_dir: Path,
    variant: str,
    keep_mlpackage: bool = False,
    lib_ane_prefix: Path | None = None,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
) -> list[Path]:
    import coremltools as ct

    out_ane_dir.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []

    def _compile(model_obj: Any, name_stem: str) -> Path:
        pkg = out_ane_dir / f"{name_stem}.mlpackage"
        mc = out_ane_dir / f"{name_stem}.mlmodelc"
        if pkg.exists():
            shutil.rmtree(pkg)
        if mc.exists():
            shutil.rmtree(mc)
        model_obj.save(str(pkg))
        subprocess.check_call(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_ane_dir)])
        if not keep_mlpackage and pkg.exists():
            shutil.rmtree(pkg)
        if lib_ane_prefix is not None:
            plan_info = _inspect_compiled_bundle(lib_ane_prefix, mc, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
            if plan_info is not None:
                print(f"[MLComputePlan] {mc.name}: {plan_info}")
                if plan_info["cpu_runtime_ops"] > 0:
                    raise RuntimeError(f"Unexpected CPU runtime fallback ops in {mc}: {plan_info}")
        return mc

    if variant in ("both", "fp16"):
        npz_f16 = dict(np.load(fp16_npz_path))
        prog_fp16, arch = build_early_prefix_mil_program(npz_f16, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
        stem_prefix = f"gemma4_early{arch['exit_layer']}_prefix_S{S}"
        ml_fp16 = ct.convert(
            prog_fp16,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        mc = _compile(ml_fp16, f"{stem_prefix}_fp16")
        print(f"Compiled {mc}")
        built.append(mc)

    if variant in ("both", "w8"):
        npz_w8 = dict(np.load(w8_npz_path))
        prog_w8, arch = build_early_prefix_mil_program(npz_w8, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
        stem_prefix = f"gemma4_early{arch['exit_layer']}_prefix_S{S}"
        ml_for_w8 = ct.convert(
            prog_w8,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        ml_w8 = quantize_mlmodel_linear_w8_per_channel(ml_for_w8, weight_threshold=2048)
        mc8 = _compile(ml_w8, f"{stem_prefix}_w8")
        print(f"Compiled {mc8}")
        built.append(mc8)

    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CoreML ANE Early14/Early23 prefix chunk bundles (.mlmodelc)")
    parser.add_argument("--home", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--variant", choices=("both", "fp16", "w8"), default="both")
    parser.add_argument(
        "--awq-mse",
        action="store_true",
        help="Apply closed-form activation-weighted MSE per-channel weight clipping before linear int8 quantization",
    )
    parser.add_argument("--reexport", action="store_true", help="Re-export FP16 EarlyExit weights npz from model")
    parser.add_argument("--keep-mlpackage", action="store_true")
    parser.add_argument("--compile-only-fp16-npz", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-w8-npz", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-out-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-lib", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.compile_only_out_dir is not None:
        fp16_p = Path(args.compile_only_fp16_npz)
        w8_p = Path(args.compile_only_w8_npz)
        lib_p = Path(args.compile_only_lib) if args.compile_only_lib else None
        compile_ane_bundles_from_npz(
            fp16_npz_path=fp16_p,
            w8_npz_path=w8_p,
            out_ane_dir=Path(args.compile_only_out_dir),
            variant=args.variant,
            keep_mlpackage=args.keep_mlpackage,
            lib_ane_prefix=lib_p,
        )
        return 0

    from fast_gemma4.ane_pipeline import ANE_CHUNK_S, ANE_P_FULL, ANE_P_SWA
    from fast_gemma4.paths import resolve_paths

    paths = resolve_paths(args.home)
    weights_npz = paths.resolve_early14_weights_npz(args.model_dir)
    act_col_sq: dict[str, np.ndarray] | None = None

    need_model = args.reexport or (not weights_npz.exists()) or (args.awq_mse and args.variant in ("both", "w8"))
    if need_model:
        import mlx.core as mx
        from fast_shared.hf import copy_tokenizer_files

        from fast_gemma4.model import FastGemma4Model

        with _gpu_validation_lock():
            t0 = time.perf_counter()
            model = FastGemma4Model(paths=paths, model_dir=args.model_dir, mlp_dtype="bfloat16", use_mpp=False)
            if not (paths.tokenizer_dir / "tokenizer.json").exists():
                copy_tokenizer_files(model.model_dir, paths.tokenizer_dir)
            if args.reexport or not weights_npz.exists():
                model.export_early14_f16(weights_npz)
                print(f"Exported {weights_npz} in {time.perf_counter() - t0:.2f}s")
            if args.awq_mse and args.variant in ("both", "w8"):
                tc0 = time.perf_counter()
                act_col_sq = collect_awq_calibration_col_sq(model, chunk_s=ANE_CHUNK_S)
                print(f"Collected AWQ-MSE calibration E[x^2] in {time.perf_counter() - tc0:.2f}s")
            del model
            mx.clear_cache()

    w8_npz_path = weights_npz
    tmp_awq_npz: Path | None = None
    if act_col_sq is not None:
        npz_raw = dict(np.load(weights_npz))
        clipped_npz, mean_alpha = apply_awq_mse_clipping(npz_raw, act_col_sq)
        print(f"Applied AWQ-MSE symmetric channel clipping (mean_alpha={mean_alpha:.4f})")
        tmp_awq_npz = weights_npz.with_name("gemma4_early_awq_tmp.npz")
        np.savez(tmp_awq_npz, **clipped_npz)
        w8_npz_path = tmp_awq_npz
        del npz_raw, clipped_npz

    out_ane_dir = (
        paths.ane_models_dir
        if paths._is_default_model_dir(args.model_dir)
        else paths.ane_models_dir / paths.checkpoint_ane_key(Path(args.model_dir))
    )

    try:
        try:
            import coremltools as _ct  # noqa: F401

            has_ct = True
        except ModuleNotFoundError:
            has_ct = False

        if has_ct:
            compile_ane_bundles_from_npz(
                fp16_npz_path=weights_npz,
                w8_npz_path=w8_npz_path,
                out_ane_dir=out_ane_dir,
                variant=args.variant,
                keep_mlpackage=args.keep_mlpackage,
                lib_ane_prefix=paths.lib_ane_prefix,
                S=ANE_CHUNK_S,
                P_SWA=ANE_P_SWA,
                P_FULL=ANE_P_FULL,
            )
        else:
            uv_bin = shutil.which("uv")
            if not uv_bin:
                raise ModuleNotFoundError(
                    "coremltools is required to compile ANE .mlmodelc bundles (install `fast-gemma4[ane-build]` or `uv`)."
                )
            cmd = [
                uv_bin,
                "run",
                "--no-project",
                "--with",
                "coremltools",
                "python3",
                str(Path(__file__).resolve()),
                "--variant",
                args.variant,
                "--compile-only-fp16-npz",
                str(weights_npz),
                "--compile-only-w8-npz",
                str(w8_npz_path),
                "--compile-only-out-dir",
                str(out_ane_dir),
                "--compile-only-lib",
                str(paths.lib_ane_prefix),
            ]
            if args.keep_mlpackage:
                cmd.append("--keep-mlpackage")
            subprocess.check_call(cmd)
    finally:
        if tmp_awq_npz is not None and tmp_awq_npz.exists():
            tmp_awq_npz.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
