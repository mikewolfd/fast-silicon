"""Native Objective-C++, CoreML Pure-ANE (`Laya-Typed-421M`), and MLX (`Kev-0.8B`) artifact builders."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import pickle
import shutil
import zipfile
from typing import Any

import numpy as np

from ._common import acquire_hw_lock
from .paths import (
    DEFAULT_KEV_ADAPTER_REPO,
    DEFAULT_KEV_BASE_REPO,
    DEFAULT_LAYA_REPO,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
)

KEV_SPECIAL_TOKENS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

# Smallest normal IEEE-754 binary16 value (2^-14 = 6.103515625e-5, exact in FP16).
# Clamping LayerNorm eps_u >= FP16_MIN_NORMAL prevents subnormal flush-to-zero on ANE.
FP16_MIN_NORMAL = float(np.ldexp(1.0, -14))


def _laya_layer_res_scale(li: int) -> float:
    """Cumulative residual branch scale factor entering ModernBERT encoder layer `li`."""
    if li <= 4:
        return 1.0
    if li <= 13:
        return 1.0 / 32.0
    if li <= 18:
        return 1.0 / 256.0
    return 1.0 / 1024.0


def _laya_ln_gain_and_eps(
    cum_r: float,
    *,
    base_inv_scale: float = 0.25,
    s_eff_floor: float = 1.0 / 256.0,
    max_ln_gain: float = 16.0,
    eps: float = 1e-5,
) -> tuple[np.float16, np.float16]:
    """Compute scale-invariant FP16 LayerNorm `(ln_gain, eps_u)` compensating for `cum_r`.

    Avoids double attenuation (`1/32..1/1024` residual scaling stacked with `0.25` LN down-scaling)
    that previously pushed late-layer variances to the `2^-24` FP16 subnormal floor and inflated
    effective `eps` to `10.0`. Keeps `u = xc * ln_gain` safe against ModernBERT `33,024` outliers
    (`u^2 <= 16,641 < 65,504` up to `|x_orig| <= 65,504`) and clamps `eps_u >= 2^-14` (normal FP16).
    """
    r = float(cum_r)
    if r <= 0.0:
        raise ValueError(f"cum_r must be positive, got {cum_r}")
    if r >= 1.0:
        ln_gain_f = float(base_inv_scale)
    else:
        # Never attenuate an already-scaled residual below 1.0; up-scale 1/1024 -> 4.0 (s_eff = 1/256).
        ln_gain_f = max(1.0, min(float(max_ln_gain), float(s_eff_floor) / r))
    s_eff = r * ln_gain_f
    eps_u_f = max(float(eps) * (s_eff * s_eff), FP16_MIN_NORMAL)
    return np.float16(ln_gain_f), np.float16(eps_u_f)


def _laya_layernorm_ch_np(
    x: np.ndarray,
    gamma_1d: np.ndarray,
    beta_1d: np.ndarray | None = None,
    *,
    cum_r: float = 1.0,
    base_inv_scale: float = 0.25,
    eps: float = 1e-5,
    flush_subnormals: bool = False,
) -> np.ndarray:
    """Pure-NumPy FP16 simulation of `_layernorm_ch` (`[1, H, 1, S]`) for PyTorch-free parity tests."""

    def _f16(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float16)
        if flush_subnormals:
            out = np.where(np.abs(out) < np.float16(FP16_MIN_NORMAL), np.float16(0.0), out)
        return out

    ln_gain, eps_u = _laya_ln_gain_and_eps(cum_r, base_inv_scale=base_inv_scale, eps=eps)
    x16 = _f16(x)
    # ANE reduce_mean accumulates in higher precision and stores FP16 output (unlike naive sequential FP16 sum).
    mean = _f16(np.mean(x16.astype(np.float32), axis=1, keepdims=True))
    xc = _f16(x16 - mean)
    u = xc if float(ln_gain) == 1.0 else _f16(xc * ln_gain)
    sq = _f16(u * u)
    var_u = _f16(np.mean(sq.astype(np.float32), axis=1, keepdims=True))
    ae = _f16(var_u + eps_u)
    rsq = _f16(1.0 / np.sqrt(ae.astype(np.float32)))
    gam4 = _f16(gamma_1d).reshape(1, -1, 1, 1)
    if beta_1d is None:
        xn = _f16(u * rsq)
        return _f16(xn * gam4)
    ug = _f16(u * gam4)
    xn = _f16(ug * rsq)
    bet4 = _f16(beta_1d).reshape(1, -1, 1, 1)
    return _f16(xn + bet4)


def _build_laya_quant_config(cto: Any, *, weight_threshold: int = 2048) -> Any:
    """Build CoreML weight-only INT8 config scoped strictly to `conv` linear projections.

    Keeps RoPE `cos`/`sin` tables (`mul`), SWA `-10000.0` bias masks (`add`), and LayerNorm
    constants exact `FP16` by leaving `global_config=None`.
    """
    return cto.OptimizationConfig(
        global_config=None,
        op_type_configs={
            "conv": cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",
                weight_threshold=int(weight_threshold),
            )
        },
    )


def compile_native_dylib(repo_root: Path) -> Path:
    """Compile `native/laya_ane_runner.mm` -> `build/liblaya_ane_runner.dylib` cleanly."""
    out_dylib = repo_root / "build" / "liblaya_ane_runner.dylib"
    if out_dylib.exists():
        out_dylib.unlink()
    return ensure_laya_dylib(repo_root)


def _resolve_dir_or_hf(local_dir: str | None, repo_id: str) -> Path:
    if local_dir is not None:
        p = Path(local_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Directory does not exist: {p}")
        return p
    from fast_shared.hf import resolve_hf_model_dir

    return resolve_hf_model_dir(None, repo_id, local_files_only=False)


def _precompute_rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angles = np.arange(seq_len, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float16)[None, None, :, :]
    sin = np.sin(angles).astype(np.float16)[None, None, :, :]
    return cos, sin


def _precompute_sliding_window_bias(seq_len: int, local_attention: int = 128) -> np.ndarray:
    idx = np.arange(seq_len)
    allow = np.abs(idx[:, None] - idx[None, :]) <= (local_attention // 2)
    bias = np.where(allow, 0.0, -10000.0).astype(np.float16)
    return bias[None, None, :, :]


def _compile_laya_seq_bundle(
    seq_len: int,
    w: dict[str, np.ndarray],
    enc_cfg: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Convert `Laya-Typed-421M` ModernBERT + 2 head layers to `W8A16` Pure-ANE `.mlmodelc`."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.optimize.coreml as cto

    H = int(enc_cfg["hidden_size"])
    n_layers = int(enc_cfg["num_hidden_layers"])
    n_heads = int(enc_cfg["num_attention_heads"])
    hd = H // n_heads
    attn_scale = 1.0 / math.sqrt(hd)
    layer_types = enc_cfg["layer_types"]
    local_window = int(enc_cfg.get("local_attention", 128))
    rope_params = enc_cfg.get("rope_parameters", {})
    theta_full = float(rope_params.get("full_attention", {}).get("rope_theta", 160000.0))
    theta_swa = float(rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0))

    cos_full_np, sin_full_np = _precompute_rope_cos_sin(seq_len, hd, theta_full)
    cos_swa_np, sin_swa_np = _precompute_rope_cos_sin(seq_len, hd, theta_swa)
    swa_bias_np = _precompute_sliding_window_bias(seq_len, local_window)

    layer_norm_eps = float(enc_cfg.get("norm_eps", enc_cfg.get("layer_norm_eps", 1e-5)))

    def _layernorm_ch(
        x: Any,
        gamma_1d: np.ndarray,
        beta_1d: np.ndarray | None,
        prefix: str,
        *,
        cum_r: float = 1.0,
        base_inv_scale: float = 0.25,
    ) -> Any:
        ln_gain, eps_u = _laya_ln_gain_and_eps(
            cum_r,
            base_inv_scale=base_inv_scale,
            eps=layer_norm_eps,
        )
        mean = mb.reduce_mean(x=x, axes=[1], keep_dims=True, name=prefix + "_mn")
        xc = mb.sub(x=x, y=mean, name=prefix + "_xc")
        u = xc if float(ln_gain) == 1.0 else mb.mul(x=xc, y=ln_gain, name=prefix + "_u")
        var_u = mb.reduce_mean(
            x=mb.mul(x=u, y=u, name=prefix + "_sq"),
            axes=[1],
            keep_dims=True,
            name=prefix + "_vu",
        )
        rsq = mb.rsqrt(x=mb.add(x=var_u, y=eps_u, name=prefix + "_ae"), name=prefix + "_rsq")
        gam4 = gamma_1d.astype(np.float16).reshape(1, H, 1, 1)
        if beta_1d is None:
            xn = mb.mul(x=u, y=rsq, name=prefix + "_xn")
            return mb.mul(x=xn, y=mb.const(val=gam4, name=prefix + "_gw"), name=prefix + "_gm")
        ug = mb.mul(x=u, y=mb.const(val=gam4, name=prefix + "_gw"), name=prefix + "_ug")
        xn = mb.mul(x=ug, y=rsq, name=prefix + "_xn")
        bet4 = beta_1d.astype(np.float16).reshape(1, H, 1, 1)
        return mb.add(x=xn, y=mb.const(val=bet4, name=prefix + "_bw"), name=prefix + "_bm")

    def _rope(x4: Any, cos_c: Any, sin_c: Any, prefix: str) -> Any:
        x0, x1 = mb.split(x=x4, num_splits=2, axis=-1, name=prefix + "_sp")
        y0 = mb.sub(
            x=mb.mul(x=x0, y=cos_c, name=prefix + "_c0"),
            y=mb.mul(x=x1, y=sin_c, name=prefix + "_s0"),
            name=prefix + "_y0",
        )
        y1 = mb.add(
            x=mb.mul(x=x1, y=cos_c, name=prefix + "_c1"),
            y=mb.mul(x=x0, y=sin_c, name=prefix + "_s1"),
            name=prefix + "_y1",
        )
        return mb.concat(values=[y0, y1], axis=-1, name=prefix + "_cat")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, seq_len), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, 1, 1, seq_len), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, H, 1, 1), dtype=ct.converters.mil.mil.types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(tok_emb_in, ext_mask, type_vec):
        cos_f = mb.const(val=cos_full_np, name="cos_full")
        sin_f = mb.const(val=sin_full_np, name="sin_full")
        cos_s = mb.const(val=cos_swa_np, name="cos_swa")
        sin_s = mb.const(val=sin_swa_np, name="sin_swa")
        swa_c = mb.const(val=swa_bias_np, name="swa_const")
        swa_mask = mb.add(x=swa_c, y=ext_mask, name="swa_mask")

        cur = _layernorm_ch(
            tok_emb_in, w["encoder.embeddings.norm.weight"], None, "emb_ln", cum_r=1.0, base_inv_scale=1.0
        )

        for i in range(n_layers):
            px = f"encoder.layers.{i}."
            r_scale = _laya_layer_res_scale(i)
            is_full = layer_types[i] == "full_attention"
            x_attn = (
                cur
                if i == 0
                else _layernorm_ch(cur, w[px + "attn_norm.weight"], None, f"l{i}_aln", cum_r=r_scale)
            )

            wqkv_f32 = w[px + "attn.Wqkv.weight"].copy()
            wqkv_f32[:H, :] *= attn_scale
            wqkv_4d = wqkv_f32[:, :, None, None].astype(np.float16)

            qkv = mb.conv(x=x_attn, weight=mb.const(val=wqkv_4d, name=f"l{i}_wqkv"), name=f"l{i}_qkv")
            qt, kt, vt = mb.split(x=qkv, num_splits=3, axis=1, name=f"l{i}_spqkv")
            qh = mb.transpose(
                x=mb.reshape(x=qt, shape=[1, n_heads, hd, seq_len], name=f"l{i}_qr"),
                perm=[0, 1, 3, 2],
                name=f"l{i}_qt",
            )
            kh = mb.transpose(
                x=mb.reshape(x=kt, shape=[1, n_heads, hd, seq_len], name=f"l{i}_kr"),
                perm=[0, 1, 3, 2],
                name=f"l{i}_kt",
            )
            vh = mb.transpose(
                x=mb.reshape(x=vt, shape=[1, n_heads, hd, seq_len], name=f"l{i}_vr"),
                perm=[0, 1, 3, 2],
                name=f"l{i}_vt",
            )

            c_t, s_t = (cos_f, sin_f) if is_full else (cos_s, sin_s)
            qrp = _rope(qh, c_t, s_t, f"l{i}_qrp")
            krp = _rope(kh, c_t, s_t, f"l{i}_krp")

            qk = mb.matmul(x=qrp, y=krp, transpose_y=True, name=f"l{i}_qk")  # guard: allow-coreml-matmul (dynamic Q@K^T activations; [1585,0,0] ANE)
            m_bias = ext_mask if is_full else swa_mask
            probs = mb.softmax(x=mb.add(x=qk, y=m_bias, name=f"l{i}_qkb"), axis=-1, name=f"l{i}_sm")
            att = mb.matmul(x=probs, y=vh, name=f"l{i}_att")  # guard: allow-coreml-matmul (dynamic P@V activations; weights use 4D mb.conv)
            att_4d = mb.reshape(
                x=mb.transpose(x=att, perm=[0, 1, 3, 2], name=f"l{i}_at"),
                shape=[1, H, 1, seq_len],
                name=f"l{i}_ar",
            )

            wo_4d = (w[px + "attn.Wo.weight"] * r_scale)[:, :, None, None].astype(np.float16)
            ao = mb.conv(x=att_4d, weight=mb.const(val=wo_4d, name=f"l{i}_wo"), name=f"l{i}_ao")
            cur = mb.add(x=cur, y=ao, name=f"l{i}_res1")

            x_mlp = _layernorm_ch(cur, w[px + "mlp_norm.weight"], None, f"l{i}_mln", cum_r=r_scale)
            wi_4d = w[px + "mlp.Wi.weight"][:, :, None, None].astype(np.float16)
            wi_out = mb.conv(x=x_mlp, weight=mb.const(val=wi_4d, name=f"l{i}_wi"), name=f"l{i}_wio")
            val_h, gate_h = mb.split(x=wi_out, num_splits=2, axis=1, name=f"l{i}_spwi")
            geglu = mb.mul(x=mb.gelu(x=val_h, mode="EXACT", name=f"l{i}_gelu"), y=gate_h, name=f"l{i}_geglu")
            wom_4d = (w[px + "mlp.Wo.weight"] * r_scale)[:, :, None, None].astype(np.float16)
            mo = mb.conv(x=geglu, weight=mb.const(val=wom_4d, name=f"l{i}_wom"), name=f"l{i}_mo")
            cur = mb.add(x=cur, y=mo, name=f"l{i}_res2")

            if i == 4:
                cur = mb.mul(x=cur, y=np.float16(1.0 / 32.0), name="recenter_l4")
            elif i == 13:
                cur = mb.mul(x=cur, y=np.float16(1.0 / 8.0), name="recenter_l13")
            elif i == 18:
                cur = mb.mul(x=cur, y=np.float16(1.0 / 4.0), name="recenter_l18")

        final_cum_r = _laya_layer_res_scale(n_layers - 1)
        h = _layernorm_ch(cur, w["encoder.final_norm.weight"], None, "enc_final_ln", cum_r=final_cum_r)
        h = mb.add(x=h, y=type_vec, name="add_qtype")

        for hi in range(2):
            hp = f"head.layers.{hi}."
            hn1 = _layernorm_ch(
                h, w[hp + "norm1.weight"], w[hp + "norm1.bias"], f"hl{hi}_n1", cum_r=1.0, base_inv_scale=0.25
            )
            in_w = w[hp + "self_attn.in_proj_weight"].copy()
            in_b = w[hp + "self_attn.in_proj_bias"].copy()
            in_w[:H, :] *= attn_scale
            in_b[:H] *= attn_scale

            hqkv = mb.conv(
                x=hn1,
                weight=mb.const(val=in_w[:, :, None, None].astype(np.float16), name=f"hl{hi}_wqkv"),
                bias=mb.const(val=in_b.astype(np.float16), name=f"hl{hi}_bqkv"),
                name=f"hl{hi}_qkv",
            )
            hqt, hkt, hvt = mb.split(x=hqkv, num_splits=3, axis=1, name=f"hl{hi}_sp")
            hqh = mb.transpose(
                x=mb.reshape(x=hqt, shape=[1, n_heads, hd, seq_len], name=f"hl{hi}_qr"),
                perm=[0, 1, 3, 2],
                name=f"hl{hi}_qt",
            )
            hkh = mb.transpose(
                x=mb.reshape(x=hkt, shape=[1, n_heads, hd, seq_len], name=f"hl{hi}_kr"),
                perm=[0, 1, 3, 2],
                name=f"hl{hi}_kt",
            )
            hvh = mb.transpose(
                x=mb.reshape(x=hvt, shape=[1, n_heads, hd, seq_len], name=f"hl{hi}_vr"),
                perm=[0, 1, 3, 2],
                name=f"hl{hi}_vt",
            )
            hqk = mb.matmul(x=hqh, y=hkh, transpose_y=True, name=f"hl{hi}_qk")  # guard: allow-coreml-matmul (dynamic head Q@K^T activations; [1585,0,0] ANE)
            hpr = mb.softmax(x=mb.add(x=hqk, y=ext_mask, name=f"hl{hi}_qkb"), axis=-1, name=f"hl{hi}_sm")
            hatt = mb.matmul(x=hpr, y=hvh, name=f"hl{hi}_att")  # guard: allow-coreml-matmul (dynamic head P@V activations; weights use 4D mb.conv)
            hatt_4d = mb.reshape(
                x=mb.transpose(x=hatt, perm=[0, 1, 3, 2], name=f"hl{hi}_at"),
                shape=[1, H, 1, seq_len],
                name=f"hl{hi}_ar",
            )
            hao = mb.conv(
                x=hatt_4d,
                weight=mb.const(val=w[hp + "self_attn.out_proj.weight"][:, :, None, None].astype(np.float16), name=f"hl{hi}_wo"),
                bias=mb.const(val=w[hp + "self_attn.out_proj.bias"].astype(np.float16), name=f"hl{hi}_bo"),
                name=f"hl{hi}_ao",
            )
            h = mb.add(x=h, y=hao, name=f"hl{hi}_res1")

            hn2 = _layernorm_ch(
                h, w[hp + "norm2.weight"], w[hp + "norm2.bias"], f"hl{hi}_n2", cum_r=1.0, base_inv_scale=0.25
            )
            hl1 = mb.conv(
                x=hn2,
                weight=mb.const(val=w[hp + "linear1.weight"][:, :, None, None].astype(np.float16), name=f"hl{hi}_w1"),
                bias=mb.const(val=w[hp + "linear1.bias"].astype(np.float16), name=f"hl{hi}_b1"),
                name=f"hl{hi}_l1",
            )
            hact = mb.relu(x=hl1, name=f"hl{hi}_relu")
            hl2 = mb.conv(
                x=hact,
                weight=mb.const(val=w[hp + "linear2.weight"][:, :, None, None].astype(np.float16), name=f"hl{hi}_w2"),
                bias=mb.const(val=w[hp + "linear2.bias"].astype(np.float16), name=f"hl{hi}_b2"),
                name=f"hl{hi}_l2",
            )
            h = mb.add(x=h, y=hl2, name=f"hl{hi}_res2")

        sn = _layernorm_ch(
            h, w["scorer.0.weight"], w["scorer.0.bias"], "sc_ln", cum_r=1.0, base_inv_scale=0.25
        )
        s1 = mb.conv(
            x=sn,
            weight=mb.const(val=w["scorer.1.weight"][:, :, None, None].astype(np.float16), name="sc_w1"),
            bias=mb.const(val=w["scorer.1.bias"].astype(np.float16), name="sc_b1"),
            name="sc_l1",
        )
        sg = mb.gelu(x=s1, mode="EXACT", name="sc_gelu")
        return mb.conv(
            x=sg,
            weight=mb.const(val=w["scorer.3.weight"][:, :, None, None].astype(np.float16), name="sc_w2"),
            bias=mb.const(val=w["scorer.3.bias"].astype(np.float16), name="sc_b2"),
            name="pos_logits",
        )

    pp = ct.PassPipeline.DEFAULT
    pp.remove_passes(["common::fuse_elementwise_to_batchnorm", "common::add_fp16_cast"])
    ml = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="tok_emb_in", shape=(1, H, 1, seq_len), dtype=np.float16),
            ct.TensorType(name="ext_mask", shape=(1, 1, 1, seq_len), dtype=np.float16),
            ct.TensorType(name="type_vec", shape=(1, H, 1, 1), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="pos_logits", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        pass_pipeline=pp,
        skip_model_load=True,
    )
    ml = cto.linear_quantize_weights(
        ml,
        config=_build_laya_quant_config(cto),
    )
    tag = f"laya_typed_s{seq_len}_w8"
    pkg = out_dir / f"_{tag}.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    ml.save(str(pkg))
    compiled_tmp = ct.utils.compile_model(str(pkg))
    dst = out_dir / f"{tag}.mlmodelc"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(compiled_tmp, str(dst))
    shutil.rmtree(pkg, ignore_errors=True)
    return dst


def build_laya_ane_artifacts(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    laya_dir: str | None = None,
    copy_compiled_from: str | None = None,
) -> Path:
    """Build & stage `Laya-Typed-421M` Pure-ANE (`W8A16`) `S=256` & `S=512` CoreML bundles."""
    import mlx.core as mx

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

    raw_mx = mx.load(str(src_dir / "model.safetensors"))
    word_f16 = np.asarray(raw_mx["encoder.embeddings.tok_embeddings.weight"].astype(mx.float16))
    type_f16 = np.asarray(raw_mx["type_emb.weight"].astype(mx.float16))
    np.savez(
        out_dir / "laya_typed_emb_f16.npz",
        word_f16=np.ascontiguousarray(word_f16),
        type_f16=np.ascontiguousarray(type_f16),
    )

    dylib_p = ensure_laya_dylib(repo_root)
    if copy_compiled_from is not None:
        pre_dir = Path(copy_compiled_from).expanduser().resolve()
        for s_len in (256, 512):
            name = f"laya_typed_s{s_len}_w8.mlmodelc"
            src_mc = pre_dir / name
            dst_mc = out_dir / name
            if not src_mc.exists():
                raise FileNotFoundError(f"Missing precompiled bundle: {src_mc}")
            if dst_mc.exists():
                shutil.rmtree(dst_mc)
            shutil.copytree(src_mc, dst_mc)
    else:
        enc_cfg = json.loads((src_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
        w_np = {k: np.asarray(v.astype(mx.float32)) for k, v in raw_mx.items()}
        with acquire_hw_lock("build_laya_ane_models"):
            for s_len in (256, 512):
                _compile_laya_seq_bundle(s_len, w_np, enc_cfg, out_dir)

    with acquire_hw_lock("verify_laya_ane_plan"):
        lib = ctypes.CDLL(str(dylib_p))
        for s_len in (256, 512):
            mc = out_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
            counts = (ctypes.c_int * 3)()
            # 0 == LAYA_COMPUTE_CPU_AND_ANE (aligned with fast_shared::ComputeMode::kCpuAndNeuralEngine)
            lib.laya_ane_inspect_plan(str(mc).encode("utf-8"), 0, counts)
            print(f"[LAYA-ANE] {mc.name}: MLComputePlan ANE/GPU/CPU = {list(counts)}")
            if int(counts[1]) != 0 or int(counts[2]) != 0 or int(counts[0]) < 1000:
                raise RuntimeError(f"Unexpected non-pure-ANE compute plan on {mc}: {list(counts)}")
    return out_dir


def _load_kev_head_tensors_stdlib_zip(head_pt_path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Load `head.pt` PyTorch zip archive via pure stdlib (`zipfile` + `pickle`) + `numpy`."""
    from collections import OrderedDict
    import io

    dtype_map: dict[str, tuple[np.dtype[Any], int]] = {
        "FloatStorage": (np.dtype("<f4"), 4),
        "HalfStorage": (np.dtype("<f2"), 2),
        "BFloat16Storage": (np.dtype("<u2"), 2),
        "DoubleStorage": (np.dtype("<f8"), 8),
    }

    with zipfile.ZipFile(head_pt_path, "r") as zf:
        pkl_name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
        if pkl_name is None:
            raise RuntimeError(f"Unsupported head checkpoint format in {head_pt_path} (missing data.pkl)")
        prefix = pkl_name[: -len("data.pkl")]

        class _StorageType:
            def __init__(self, name: str) -> None:
                self.name = name

        def _rebuild_tensor(storage: np.ndarray, storage_offset: int, size: tuple[int, ...], _stride: Any, *_rest: Any) -> np.ndarray:
            n_elem = int(np.prod(size)) if size else 1
            sub = storage[storage_offset : storage_offset + n_elem]
            return sub.reshape(size).copy()

        class _PtUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild_tensor
                if module == "torch" and name in dtype_map:
                    return _StorageType(name)
                if module == "builtins":
                    return super().find_class(module, name)
                return super().find_class(module, name)

            def persistent_load(self, pid: Any) -> np.ndarray:
                tag, st_type, key, _loc, count = pid[:5]
                if tag != "storage":
                    raise pickle.UnpicklingError(f"Unexpected persistent pid: {pid!r}")
                st_name = getattr(st_type, "name", str(st_type))
                dt, item_sz = dtype_map[st_name]
                raw = zf.read(f"{prefix}data/{key}")
                arr = np.frombuffer(raw, dtype=dt, count=int(count))
                if st_name == "BFloat16Storage":
                    u32 = arr.astype(np.uint32) << 16
                    return u32.view(np.float32)
                return arr.astype(np.float32)

        ckpt = _PtUnpickler(io.BytesIO(zf.read(pkl_name))).load()

    temp = float(ckpt.get("temperature", 1.0))
    hsd = ckpt["head"]
    tensors = {
        "q_weight": np.asarray(hsd["q.weight"], dtype=np.float32),
        "q_bias": np.asarray(hsd["q.bias"], dtype=np.float32),
        "k_weight": np.asarray(hsd["k.weight"], dtype=np.float32),
        "k_bias": np.asarray(hsd["k.bias"], dtype=np.float32),
    }
    return tensors, temp


def _load_kev_head_tensors(head_pt_path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Load `head.pt` via `torch` if installed, else pure stdlib PyTorch ZIP reader."""
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_kev_head_tensors_stdlib_zip(head_pt_path)

    ckpt = torch.load(head_pt_path, map_location="cpu", weights_only=False)
    temp = float(ckpt.get("temperature", 1.0))
    hsd = ckpt["head"]
    tensors = {
        "q_weight": hsd["q.weight"].float().numpy(),
        "q_bias": hsd["q.bias"].float().numpy(),
        "k_weight": hsd["k.weight"].float().numpy(),
        "k_bias": hsd["k.bias"].float().numpy(),
    }
    return tensors, temp


def build_kev_mlx_artifacts(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    kev_adapter_dir: str | None = None,
    kev_base_dir: str | None = None,
) -> Path:
    """Merge `jaredpalmer/kev-0.8b` LoRA + `PointerHead` into `Qwen/Qwen3.5-0.8B-Base` and quantize W8."""
    import mlx.core as mx
    from tokenizers import Tokenizer

    adapter_dir = _resolve_dir_or_hf(kev_adapter_dir, DEFAULT_KEV_ADAPTER_REPO)
    base_dir = _resolve_dir_or_hf(kev_base_dir, DEFAULT_KEV_BASE_REPO)
    out_root = artifacts_dir if artifacts_dir is not None else resolve_artifacts_dir(home=repo_root)
    out_dir = out_root / "kev_08b"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    text_cfg = dict(base_cfg.get("text_config", base_cfg))
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    alpha = float(acfg["lora_alpha"]) / (float(acfg["r"]) ** 0.5 if acfg.get("use_rslora") else float(acfg["r"]))

    shutil.copy2(adapter_dir / "tokenizer.json", out_dir / "tokenizer.json")
    if (adapter_dir / "tokenizer_config.json").exists():
        shutil.copy2(adapter_dir / "tokenizer_config.json", out_dir / "tokenizer_config.json")

    tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    special_ids = [int(tok.token_to_id(s)) for s in KEV_SPECIAL_TOKENS]
    pad_id = int(text_cfg.get("eos_token_id", 248044))

    head_np, native_temp = _load_kev_head_tensors(adapter_dir / "head.pt")

    with acquire_hw_lock("merge_and_pack_kev_08b"):
        with mx.stream(mx.cpu):
            raw_base: dict[str, mx.array] = {}
            for sf in sorted(base_dir.glob("model*.safetensors")):
                raw_base.update(mx.load(str(sf)))
            lora_w = mx.load(str(adapter_dir / "adapter_model.safetensors"))

            text_weights: dict[str, mx.array] = {}
            for k, v in raw_base.items():
                if k.startswith(("vision_tower", "model.visual")) or "mtp." in k or k == "lm_head.weight":
                    continue
                if k.startswith("model.language_model."):
                    nk = "model." + k[len("model.language_model.") :]
                elif k.startswith("language_model.model."):
                    nk = "model." + k[len("language_model.model.") :]
                elif k.startswith("model."):
                    nk = k
                else:
                    continue
                text_weights[nk] = v

            merged_count = 0
            for lk, la in lora_w.items():
                if not lk.endswith(".lora_A.weight"):
                    continue
                stem = lk[: -len(".lora_A.weight")]
                tgt = stem.replace("base_model.model.", "model.", 1) + ".weight"
                if tgt not in text_weights:
                    raise KeyError(f"Missing LoRA target {tgt} (sample keys: {list(text_weights)[:5]})")
                lb = lora_w[stem + ".lora_B.weight"]
                w0 = text_weights[tgt]
                delta = (lb.astype(mx.float32) @ la.astype(mx.float32)) * alpha
                text_weights[tgt] = (w0.astype(mx.float32) + delta).astype(mx.bfloat16)
                mx.eval(text_weights[tgt])
                merged_count += 1

            has_unsanitized_conv = any("conv1d.weight" in k and v.shape[-1] != 1 for k, v in text_weights.items())
            norm_sfx = (
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                "model.norm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
            )
            packed: dict[str, mx.array] = {}
            for k, v in text_weights.items():
                arr = v
                if "conv1d.weight" in k and arr.shape[-1] != 1:
                    arr = arr.moveaxis(2, 1)
                if has_unsanitized_conv and any(k.endswith(sfx) for sfx in norm_sfx) and arr.ndim == 1:
                    arr = (arr.astype(mx.float32) + 1.0).astype(mx.bfloat16)
                elif arr.dtype != mx.bfloat16:
                    arr = arr.astype(mx.bfloat16)
                mx.eval(arr)
                packed[k] = arr

            packed["pointer_head.q_weight"] = mx.array(head_np["q_weight"]).astype(mx.bfloat16)
            packed["pointer_head.q_bias"] = mx.array(head_np["q_bias"]).astype(mx.bfloat16)
            packed["pointer_head.k_weight"] = mx.array(head_np["k_weight"]).astype(mx.bfloat16)
            packed["pointer_head.k_bias"] = mx.array(head_np["k_bias"]).astype(mx.bfloat16)
            mx.eval(
                packed["pointer_head.q_weight"],
                packed["pointer_head.q_bias"],
                packed["pointer_head.k_weight"],
                packed["pointer_head.k_bias"],
            )

            mx.save_safetensors(str(out_dir / "model.safetensors"), packed)

    meta = {
        "model_name": "kev-0.8b-r15-merged-bf16",
        "native_temperature": native_temp,
        "special_tokens": KEV_SPECIAL_TOKENS,
        "special_ids": special_ids,
        "pad_id": pad_id,
        "merged_lora_tensors": merged_count,
        "sanitized_mlx": True,
        "text_config": text_cfg,
    }
    (out_dir / "kev_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sz_mb = (out_dir / "model.safetensors").stat().st_size / (1024 * 1024)
    print(f"[KEV-0.8B] Saved {out_dir / 'model.safetensors'} ({sz_mb:.2f} MB, {merged_count} LoRA tensors merged, T={native_temp:.4f})")
    quantize_kev_w8_artifacts(repo_root, artifacts_dir=out_root)
    return out_dir


def quantize_kev_w8_artifacts(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    group_size: int = 64,
    bits: int = 8,
) -> Path:
    """Quantize merged BF16 `model.safetensors` linear + embedding matrices into lossless 8-bit `model_w8.safetensors`."""
    import mlx.core as mx

    out_root = artifacts_dir if artifacts_dir is not None else resolve_artifacts_dir(home=repo_root)
    out_dir = out_root / "kev_08b"
    bf16_sf = out_dir / "model.safetensors"
    meta_p = out_dir / "kev_meta.json"
    if not (bf16_sf.exists() and meta_p.exists()):
        raise FileNotFoundError(f"Missing BF16 Kev-0.8B artifacts in {out_dir}")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    with acquire_hw_lock("quantize_kev_08b_w8"):
        with mx.stream(mx.cpu):
            src_w = mx.load(str(bf16_sf))
            packed_q: dict[str, mx.array] = {}
            quant_count = 0
            for k, v in src_w.items():
                if (
                    k.startswith("model.")
                    and k.endswith(".weight")
                    and v.ndim == 2
                    and int(v.shape[-1]) % group_size == 0
                ):
                    stem = k[: -len(".weight")]
                    wq, scales, biases = mx.quantize(v.astype(mx.bfloat16), group_size=group_size, bits=bits)
                    mx.eval(wq, scales, biases)
                    packed_q[stem + ".weight"] = wq
                    packed_q[stem + ".scales"] = scales
                    packed_q[stem + ".biases"] = biases
                    quant_count += 1
                else:
                    packed_q[k] = v
            out_w8 = out_dir / "model_w8.safetensors"
            mx.save_safetensors(str(out_w8), packed_q)

    meta_w8 = dict(meta)
    meta_w8["model_name"] = f"kev-0.8b-r15-merged-w{bits}-g{group_size}"
    meta_w8["quantization"] = {"bits": bits, "group_size": group_size, "quantized_matrices": quant_count}
    (out_dir / "kev_meta_w8.json").write_text(json.dumps(meta_w8, indent=2), encoding="utf-8")
    sz_mb = (out_dir / "model_w8.safetensors").stat().st_size / (1024 * 1024)
    print(f"[KEV-0.8B-W{bits}] Saved {out_dir / 'model_w8.safetensors'} ({sz_mb:.2f} MB, {quant_count} matrices quantized)")
    return out_dir / "model_w8.safetensors"


def run_build_args(args: argparse.Namespace) -> int:
    """Execute artifact build/setup steps parsed from CLI flags."""
    repo_root = resolve_home(args.home)
    out_artifacts = resolve_artifacts_dir(args.artifacts_dir, home=repo_root)
    run_all = not (
        args.compile_native or args.build_laya_ane or args.build_kev_mlx or args.quantize_existing_kev_w8
    )

    if args.compile_native or run_all:
        dylib = compile_native_dylib(repo_root)
        print(f"[NATIVE] Compiled {dylib}")
    if args.build_laya_ane or run_all:
        build_laya_ane_artifacts(
            repo_root,
            artifacts_dir=out_artifacts,
            laya_dir=args.laya_dir,
            copy_compiled_from=args.copy_compiled_laya_from,
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


def add_setup_weights_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default=None, help="Override fast-decision-models root directory")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory")
    parser.add_argument("--compile-native", action="store_true", help="Compile native/laya_ane_runner.mm -> build/liblaya_ane_runner.dylib")
    parser.add_argument("--build-laya-ane", action="store_true", help="Build & stage Laya-Typed-421M CoreML ANE W8A16 bundles")
    parser.add_argument("--laya-dir", default=None, help="Local Laya-Typed checkpoint dir (defaults to HF download)")
    parser.add_argument("--copy-compiled-laya-from", default=None, help="Dir containing precompiled laya_typed_s{256,512}_w8.mlmodelc")
    parser.add_argument("--build-kev-mlx", action="store_true", help="Merge & stage Kev-0.8B r15 BF16 + W8 safetensors")
    parser.add_argument("--quantize-existing-kev-w8", action="store_true", help="Quantize staged Kev-0.8B model.safetensors -> model_w8.safetensors")
    parser.add_argument("--kev-adapter-dir", default=None, help="Local Kev-0.8B adapter dir")
    parser.add_argument("--kev-base-dir", default=None, help="Local Qwen3.5-0.8B-Base dir")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Laya-Typed ANE & Kev-0.8B MLX decision-model artifacts")
    add_setup_weights_arguments(parser)
    return run_build_args(parser.parse_args(argv))


__all__ = [
    "FP16_MIN_NORMAL",
    "KEV_SPECIAL_TOKENS",
    "_build_laya_quant_config",
    "_compile_laya_seq_bundle",
    "_laya_layer_res_scale",
    "_laya_layernorm_ch_np",
    "_laya_ln_gain_and_eps",
    "_load_kev_head_tensors",
    "_load_kev_head_tensors_stdlib_zip",
    "acquire_hw_lock",
    "add_setup_weights_arguments",
    "build_kev_mlx_artifacts",
    "build_laya_ane_artifacts",
    "compile_native_dylib",
    "quantize_kev_w8_artifacts",
    "run_build_args",
]
