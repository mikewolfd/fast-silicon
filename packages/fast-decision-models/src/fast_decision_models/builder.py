"""Native Objective-C++, CoreML Pure-ANE (`Laya-Typed-421M`), and MLX (`Kev-0.8B` / `Kev-4B`) artifact builders."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import zipfile
from typing import Any

import numpy as np

from ._common import acquire_hw_lock
from .paths import (
    DEFAULT_KEV_4B_ADAPTER_REPO,
    DEFAULT_KEV_4B_BASE_REPO,
    DEFAULT_KEV_ADAPTER_REPO,
    DEFAULT_KEV_BASE_REPO,
    DEFAULT_LAYA_REPO,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
)

KEV_SPECIAL_TOKENS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
DEFAULT_LAYA_SEQ_LENS: tuple[int, ...] = (64, 128, 192, 256, 384, 512)
DEFAULT_LAYA_MAX_MARKERS: int = 32

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
    """Compute scale-invariant FP16 LayerNorm `(ln_gain, eps_u)` compensating for `cum_r`."""
    r = float(cum_r)
    if r <= 0.0:
        raise ValueError(f"cum_r must be positive, got {cum_r}")
    if r >= 1.0:
        ln_gain_f = float(base_inv_scale)
    else:
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
    """Build CoreML weight-only INT8 config scoped strictly to `conv` linear projections."""
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
    """Return `(1, 1, seq_len, head_dim // 2)` FP16 RoPE tables (retained for `coreai_export.py`)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angles = np.arange(seq_len, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float16)[None, None, :, :]
    sin = np.sin(angles).astype(np.float16)[None, None, :, :]
    return cos, sin


def _precompute_rope_cos_sin_nchw(seq_len: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return `(1, head_dim // 2, 1, seq_len)` FP16 RoPE tables for split-head `[1, hd, 1, S]` ANE layout."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    angles = inv_freq[:, None] * np.arange(seq_len, dtype=np.float64)[None, :]
    cos = np.cos(angles).astype(np.float16)[None, :, None, :]
    sin = np.sin(angles).astype(np.float16)[None, :, None, :]
    return cos, sin


def _precompute_sliding_window_bias(seq_len: int, local_attention: int = 128) -> np.ndarray:
    """Return `(1, 1, seq_len, seq_len)` FP16 sliding-window bias (retained for `coreai_export.py`)."""
    idx = np.arange(seq_len)
    allow = np.abs(idx[:, None] - idx[None, :]) <= (local_attention // 2)
    bias = np.where(allow, 0.0, -10000.0).astype(np.float16)
    return bias[None, None, :, :]


def _precompute_sliding_window_bias_bkhq(seq_len: int, local_attention: int = 128) -> np.ndarray:
    """Return `(1, seq_len, 1, seq_len)` FP16 sliding-window bias for `einsum('bkhd,bdhq->bkhq')` layout."""
    idx = np.arange(seq_len)
    allow = np.abs(idx[:, None] - idx[None, :]) <= (local_attention // 2)
    bias = np.where(allow, 0.0, -10000.0).astype(np.float16)
    return bias[None, :, None, :]


def _build_laya_seq_mlpackage(
    seq_len: int,
    w: Mapping[str, np.ndarray],
    enc_cfg: Mapping[str, Any],
    out_dir: Path,
    *,
    max_markers: int = DEFAULT_LAYA_MAX_MARKERS,
    guard_ir: bool = True,
) -> Path:
    """Build quantized `W8A16` `.mlpackage` for one `Laya-Typed-421M` sequence bucket (`S`).

    Architecture optimizations:
      - Exact `erf` GELU (`mode="EXACT"`) on ModernBERT GeGLU and Laya scorer
      - Native `mb.layer_norm` (`axes=[1]`) with `_laya_ln_gain_and_eps` `>= 2^-14` FTZ floor
      - Split-head `[1, hd, 1, S]` `mb.einsum` (`0` 4D reshape+matmul ops; `100%` Pure-ANE)
      - Pre-scorer `marker_map` `[1, S, 1, 32]` gather (`16x` less scorer compute at `S=512`)
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import coremltools.optimize.coreml as cto
    from fast_shared.ir_dtype_guard import check_coreml_model, validate_export_quant_dtypes

    out_dir.mkdir(parents=True, exist_ok=True)
    H = int(enc_cfg["hidden_size"])
    n_layers = int(enc_cfg["num_hidden_layers"])
    n_heads = int(enc_cfg["num_attention_heads"])
    hd = H // n_heads
    M = int(max_markers)
    attn_scale = 1.0 / math.sqrt(hd)
    layer_types = enc_cfg["layer_types"]
    local_window = int(enc_cfg.get("local_attention", 128))
    rope_params = enc_cfg.get("rope_parameters", {})
    theta_full = float(rope_params.get("full_attention", {}).get("rope_theta", 160000.0))
    theta_swa = float(rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0))

    cos_full_np, sin_full_np = _precompute_rope_cos_sin_nchw(seq_len, hd, theta_full)
    cos_swa_np, sin_swa_np = _precompute_rope_cos_sin_nchw(seq_len, hd, theta_swa)
    swa_bias_np = _precompute_sliding_window_bias_bkhq(seq_len, local_window)
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
        u = x if float(ln_gain) == 1.0 else mb.mul(x=x, y=ln_gain, name=prefix + "_u")
        xn = mb.layer_norm(x=u, axes=[1], epsilon=eps_u, name=prefix + "_ln")
        gam4 = np.asarray(gamma_1d, dtype=np.float16).reshape(1, H, 1, 1)
        out = mb.mul(x=xn, y=mb.const(val=gam4, name=prefix + "_gw"), name=prefix + "_gm")
        if beta_1d is None:
            return out
        bet4 = np.asarray(beta_1d, dtype=np.float16).reshape(1, H, 1, 1)
        return mb.add(x=out, y=mb.const(val=bet4, name=prefix + "_bw"), name=prefix + "_bm")

    def _rope_head(xh: Any, cos_c: Any, sin_c: Any, prefix: str) -> Any:
        x0, x1 = mb.split(x=xh, num_splits=2, axis=1, name=prefix + "_sp")
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
        return mb.concat(values=[y0, y1], axis=1, name=prefix + "_cat")

    def _split_head_einsum_attn(
        qt: Any,
        kt: Any,
        vt: Any,
        m_bias: Any,
        prefix: str,
        *,
        cos_c: Any | None = None,
        sin_c: Any | None = None,
    ) -> Any:
        q_heads = mb.split(x=qt, num_splits=n_heads, axis=1, name=prefix + "_spq")
        k_heads = mb.split(x=kt, num_splits=n_heads, axis=1, name=prefix + "_spk")
        v_heads = mb.split(x=vt, num_splits=n_heads, axis=1, name=prefix + "_spv")
        att_heads: list[Any] = []
        for hi_idx in range(n_heads):
            qh_i = q_heads[hi_idx]
            kh_i = k_heads[hi_idx]
            vh_i = v_heads[hi_idx]
            if cos_c is not None and sin_c is not None:
                qh_i = _rope_head(qh_i, cos_c, sin_c, f"{prefix}_qrp{hi_idx}")
                kh_i = _rope_head(kh_i, cos_c, sin_c, f"{prefix}_krp{hi_idx}")
            kt_i = mb.transpose(x=kh_i, perm=[0, 3, 2, 1], name=f"{prefix}_kt{hi_idx}")
            qk_i = mb.einsum(
                values=[kt_i, qh_i],
                equation="bkhd,bdhq->bkhq",
                name=f"{prefix}_qk{hi_idx}",
            )
            probs_i = mb.softmax(
                x=mb.add(x=qk_i, y=m_bias, name=f"{prefix}_qkb{hi_idx}"),
                axis=1,
                name=f"{prefix}_sm{hi_idx}",
            )
            att_i = mb.einsum(
                values=[vh_i, probs_i],
                equation="bdhk,bkhq->bdhq",
                name=f"{prefix}_av{hi_idx}",
            )
            att_heads.append(att_i)
        return mb.concat(values=att_heads, axis=1, name=prefix + "_cat")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, seq_len), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, seq_len, 1, 1), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, H, 1, 1), dtype=ct.converters.mil.mil.types.fp16),
            mb.TensorSpec(shape=(1, seq_len, 1, M), dtype=ct.converters.mil.mil.types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(tok_emb_in, ext_mask, type_vec, marker_map):
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

            wqkv_f32 = np.asarray(w[px + "attn.Wqkv.weight"], dtype=np.float32).copy()
            wqkv_f32[:H, :] *= attn_scale
            wqkv_4d = wqkv_f32[:, :, None, None].astype(np.float16)

            qkv = mb.conv(x=x_attn, weight=mb.const(val=wqkv_4d, name=f"l{i}_wqkv"), name=f"l{i}_qkv")
            qt, kt, vt = mb.split(x=qkv, num_splits=3, axis=1, name=f"l{i}_spqkv")
            c_t, s_t = (cos_f, sin_f) if is_full else (cos_s, sin_s)
            m_bias = ext_mask if is_full else swa_mask
            att_4d = _split_head_einsum_attn(qt, kt, vt, m_bias, f"l{i}_att", cos_c=c_t, sin_c=s_t)

            wo_4d = (np.asarray(w[px + "attn.Wo.weight"], dtype=np.float32) * r_scale)[:, :, None, None].astype(np.float16)
            ao = mb.conv(x=att_4d, weight=mb.const(val=wo_4d, name=f"l{i}_wo"), name=f"l{i}_ao")
            cur = mb.add(x=cur, y=ao, name=f"l{i}_res1")

            x_mlp = _layernorm_ch(cur, w[px + "mlp_norm.weight"], None, f"l{i}_mln", cum_r=r_scale)
            wi_4d = np.asarray(w[px + "mlp.Wi.weight"], dtype=np.float16)[:, :, None, None]
            wi_out = mb.conv(x=x_mlp, weight=mb.const(val=wi_4d, name=f"l{i}_wi"), name=f"l{i}_wio")
            val_h, gate_h = mb.split(x=wi_out, num_splits=2, axis=1, name=f"l{i}_spwi")
            geglu = mb.mul(x=mb.gelu(x=val_h, mode="EXACT", name=f"l{i}_gelu"), y=gate_h, name=f"l{i}_geglu")
            wom_4d = (np.asarray(w[px + "mlp.Wo.weight"], dtype=np.float32) * r_scale)[:, :, None, None].astype(np.float16)
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
            in_w = np.asarray(w[hp + "self_attn.in_proj_weight"], dtype=np.float32).copy()
            in_b = np.asarray(w[hp + "self_attn.in_proj_bias"], dtype=np.float32).copy()
            in_w[:H, :] *= attn_scale
            in_b[:H] *= attn_scale

            hqkv = mb.conv(
                x=hn1,
                weight=mb.const(val=in_w[:, :, None, None].astype(np.float16), name=f"hl{hi}_wqkv"),
                bias=mb.const(val=in_b.astype(np.float16), name=f"hl{hi}_bqkv"),
                name=f"hl{hi}_qkv",
            )
            hqt, hkt, hvt = mb.split(x=hqkv, num_splits=3, axis=1, name=f"hl{hi}_sp")
            hatt_4d = _split_head_einsum_attn(hqt, hkt, hvt, ext_mask, f"hl{hi}_att")
            hao = mb.conv(
                x=hatt_4d,
                weight=mb.const(
                    val=np.asarray(w[hp + "self_attn.out_proj.weight"], dtype=np.float16)[:, :, None, None],
                    name=f"hl{hi}_wo",
                ),
                bias=mb.const(val=np.asarray(w[hp + "self_attn.out_proj.bias"], dtype=np.float16), name=f"hl{hi}_bo"),
                name=f"hl{hi}_ao",
            )
            h = mb.add(x=h, y=hao, name=f"hl{hi}_res1")

            hn2 = _layernorm_ch(
                h, w[hp + "norm2.weight"], w[hp + "norm2.bias"], f"hl{hi}_n2", cum_r=1.0, base_inv_scale=0.25
            )
            hl1 = mb.conv(
                x=hn2,
                weight=mb.const(
                    val=np.asarray(w[hp + "linear1.weight"], dtype=np.float16)[:, :, None, None],
                    name=f"hl{hi}_w1",
                ),
                bias=mb.const(val=np.asarray(w[hp + "linear1.bias"], dtype=np.float16), name=f"hl{hi}_b1"),
                name=f"hl{hi}_l1",
            )
            hact = mb.relu(x=hl1, name=f"hl{hi}_relu")
            hl2 = mb.conv(
                x=hact,
                weight=mb.const(
                    val=np.asarray(w[hp + "linear2.weight"], dtype=np.float16)[:, :, None, None],
                    name=f"hl{hi}_w2",
                ),
                bias=mb.const(val=np.asarray(w[hp + "linear2.bias"], dtype=np.float16), name=f"hl{hi}_b2"),
                name=f"hl{hi}_l2",
            )
            h = mb.add(x=h, y=hl2, name=f"hl{hi}_res2")

        # Gather <=32 option markers BEFORE the 2-layer 1024->1024->1 scorer (`16x` less compute at S=512).
        h_markers = mb.einsum(
            values=[h, marker_map],
            equation="bchs,bshm->bchm",
            name="gather_markers",
        )
        sn = _layernorm_ch(
            h_markers, w["scorer.0.weight"], w["scorer.0.bias"], "sc_ln", cum_r=1.0, base_inv_scale=0.25
        )
        s1 = mb.conv(
            x=sn,
            weight=mb.const(val=np.asarray(w["scorer.1.weight"], dtype=np.float16)[:, :, None, None], name="sc_w1"),
            bias=mb.const(val=np.asarray(w["scorer.1.bias"], dtype=np.float16), name="sc_b1"),
            name="sc_l1",
        )
        sg = mb.gelu(x=s1, mode="EXACT", name="sc_gelu")
        return mb.conv(
            x=sg,
            weight=mb.const(val=np.asarray(w["scorer.3.weight"], dtype=np.float16)[:, :, None, None], name="sc_w2"),
            bias=mb.const(val=np.asarray(w["scorer.3.bias"], dtype=np.float16), name="sc_b2"),
            name="pos_logits",
        )

    pp = ct.PassPipeline.DEFAULT
    pp.remove_passes(["common::fuse_elementwise_to_batchnorm", "common::add_fp16_cast"])
    ml = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="tok_emb_in", shape=(1, H, 1, seq_len), dtype=np.float16),
            ct.TensorType(name="ext_mask", shape=(1, seq_len, 1, 1), dtype=np.float16),
            ct.TensorType(name="type_vec", shape=(1, H, 1, 1), dtype=np.float16),
            ct.TensorType(name="marker_map", shape=(1, seq_len, 1, M), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="pos_logits", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        pass_pipeline=pp,
        skip_model_load=True,
    )
    disallowed = validate_export_quant_dtypes(["int8"], export_backend="coreml")
    if disallowed:
        raise ValueError(f"Disallowed export quantization dtypes for coreml: {disallowed}")
    ml = cto.linear_quantize_weights(
        ml,
        config=_build_laya_quant_config(cto),
    )
    tag = f"laya_typed_s{seq_len}_w8"
    if guard_ir:
        rep = check_coreml_model(
            ml,
            model_name=tag,
            flag_add_fp16_cast=True,
            warn_matmul_ane=True,
        )
        if not rep.ok:
            errs = "; ".join(f.message for f in rep.errors[:5])
            raise RuntimeError(f"ir_dtype_guard failed on {tag}: {errs}")
    pkg = out_dir / f"_{tag}.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    ml.save(str(pkg))
    return pkg


def _save_laya_multifunction_mlpackage(
    seq_pkgs: Mapping[int, Path],
    dst_mlpackage: Path,
    *,
    default_seq_len: int = 512,
) -> Path:
    """Merge per-bucket `.mlpackage` graphs into a single weight-deduplicated multifunction `.mlpackage`."""
    import coremltools as ct

    if not seq_pkgs:
        raise ValueError("seq_pkgs must not be empty")
    desc = ct.utils.MultiFunctionDescriptor()
    sorted_lens = sorted(int(s) for s in seq_pkgs.keys())
    for s_len in sorted_lens:
        desc.add_function(str(seq_pkgs[s_len]), "main", f"s{s_len}")
    eff_default = default_seq_len if default_seq_len in seq_pkgs else sorted_lens[-1]
    desc.default_function_name = f"s{eff_default}"
    if dst_mlpackage.exists():
        shutil.rmtree(dst_mlpackage)
    dst_mlpackage.parent.mkdir(parents=True, exist_ok=True)
    ct.utils.save_multifunction(desc, str(dst_mlpackage))
    return dst_mlpackage


def _publish_multifunction_and_symlinks(
    mf_pkg: Path,
    out_dir: Path,
    seq_lens: Sequence[int],
) -> Path:
    """Compile multifunction `.mlpackage` -> `laya_typed_w8.mlmodelc` and publish bucket symlinks."""
    import coremltools as ct

    compiled_tmp = ct.utils.compile_model(str(mf_pkg))
    mf_dst = out_dir / "laya_typed_w8.mlmodelc"
    if mf_dst.exists() or mf_dst.is_symlink():
        if mf_dst.is_symlink() or mf_dst.is_file():
            mf_dst.unlink()
        else:
            shutil.rmtree(mf_dst)
    shutil.move(compiled_tmp, str(mf_dst))

    for s_len in seq_lens:
        link_p = out_dir / f"laya_typed_s{int(s_len)}_w8.mlmodelc"
        if link_p.exists() or link_p.is_symlink():
            if link_p.is_symlink() or link_p.is_file():
                link_p.unlink()
            else:
                shutil.rmtree(link_p)
        os.symlink("laya_typed_w8.mlmodelc", link_p)
    return mf_dst


def _compile_laya_seq_bundle(
    seq_len: int,
    w: Mapping[str, np.ndarray],
    enc_cfg: Mapping[str, Any],
    out_dir: Path,
    *,
    guard_ir: bool = True,
) -> Path:
    """Convert one `Laya-Typed-421M` bucket `S` directly into standalone `{tag}.mlmodelc`."""
    import coremltools as ct

    pkg = _build_laya_seq_mlpackage(seq_len, w, enc_cfg, out_dir, guard_ir=guard_ir)
    tag = f"laya_typed_s{seq_len}_w8"
    try:
        compiled_tmp = ct.utils.compile_model(str(pkg))
        dst = out_dir / f"{tag}.mlmodelc"
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.move(compiled_tmp, str(dst))
        return dst
    finally:
        shutil.rmtree(pkg, ignore_errors=True)


def build_laya_ane_artifacts(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    laya_dir: str | None = None,
    copy_compiled_from: str | None = None,
    backend: str = "coreml",
    coreai_preset: str | None = None,
    block_size: int = 32,
    calib_batches: int = 2,
    guard_ir: bool = True,
    inspect_placement: bool = False,
    seq_lens: Sequence[int] | None = None,
    multifunction: bool = True,
) -> Path:
    """Build & stage `Laya-Typed-421M` Pure-ANE (`W8A16` multifunction `.mlmodelc`) or Core AI (`.aimodel`) bundles."""
    import mlx.core as mx

    if backend not in ("coreml", "coreai"):
        raise ValueError(f"Unsupported backend={backend!r}; choose 'coreml' or 'coreai'")

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

    if backend == "coreai":
        from fast_shared.hw_check import probe_core_ai_support
        from .coreai_export import run_laya_coreai_export

        probe = probe_core_ai_support()
        if not probe.get("export_ready", probe.get("coreai_export_ready", False)):
            raise RuntimeError(
                f"Core AI (.aimodel) offline export requires coreai-torch >= 0.4.1 (status: {probe})"
            )
        eff_preset = (coreai_preset or "w4_per_block").strip().lower()
        enc_cfg = json.loads((src_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
        w_np = {k: np.asarray(v.astype(mx.float32)) for k, v in raw_mx.items()}
        eff_ca_lens = tuple(int(s) for s in seq_lens) if seq_lens is not None else (256, 512)
        stages_meta: list[dict[str, Any]] = []
        for s_len in eff_ca_lens:
            res = run_laya_coreai_export(
                out_dir=out_dir,
                weights=w_np,
                enc_cfg=enc_cfg,
                seq_len=s_len,
                export_backend="coreai",
                preset=eff_preset,
                block_size=block_size,
                calib_batches=calib_batches,
                stem=f"laya_typed_s{s_len}_{eff_preset}",
                inspect_placement=inspect_placement,
            )
            stages_meta.append(
                {
                    "seq_len": s_len,
                    "asset": res.asset_path.name,
                    "cache_key_sha256": res.cache_key_sha256,
                    "sensitivity": res.sensitivity_report,
                    "asset_summary": res.asset_summary,
                }
            )
        (out_dir / "coreai_manifest.json").write_text(
            json.dumps({"backend": "coreai", "preset": eff_preset, "probe": probe, "stages": stages_meta}, indent=2),
            encoding="utf-8",
        )
        return out_dir

    eff_seq_lens = tuple(int(s) for s in (seq_lens if seq_lens is not None else DEFAULT_LAYA_SEQ_LENS))
    dylib_p = ensure_laya_dylib(repo_root)
    if copy_compiled_from is not None:
        pre_dir = Path(copy_compiled_from).expanduser().resolve()
        mf_src = pre_dir / "laya_typed_w8.mlmodelc"
        if mf_src.is_dir():
            mf_dst = out_dir / "laya_typed_w8.mlmodelc"
            if mf_dst.exists() or mf_dst.is_symlink():
                shutil.rmtree(mf_dst, ignore_errors=True)
            shutil.copytree(mf_src, mf_dst)
            for s_len in eff_seq_lens:
                lp = out_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
                if lp.exists() or lp.is_symlink():
                    lp.unlink() if lp.is_symlink() else shutil.rmtree(lp)
                os.symlink("laya_typed_w8.mlmodelc", lp)
        else:
            copied_lens: list[int] = []
            for s_len in eff_seq_lens:
                name = f"laya_typed_s{s_len}_w8.mlmodelc"
                src_mc = pre_dir / name
                if not src_mc.exists():
                    continue
                dst_mc = out_dir / name
                if dst_mc.exists() or dst_mc.is_symlink():
                    dst_mc.unlink() if dst_mc.is_symlink() else shutil.rmtree(dst_mc)
                shutil.copytree(src_mc.resolve(), dst_mc)
                copied_lens.append(s_len)
            if not copied_lens:
                raise FileNotFoundError(f"Missing precompiled Laya bundles in {pre_dir}")
            eff_seq_lens = tuple(copied_lens)
    else:
        enc_cfg = json.loads((src_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
        w_np = {k: np.asarray(v.astype(mx.float32)) for k, v in raw_mx.items()}
        with acquire_hw_lock("build_laya_ane_models"):
            if multifunction and len(eff_seq_lens) > 1:
                pkg_stage = out_dir / f".stage_mf_{os.getpid()}"
                if pkg_stage.exists():
                    shutil.rmtree(pkg_stage)
                pkg_stage.mkdir(parents=True, exist_ok=True)
                try:
                    seq_pkgs = {
                        s_len: _build_laya_seq_mlpackage(s_len, w_np, enc_cfg, pkg_stage, guard_ir=guard_ir)
                        for s_len in eff_seq_lens
                    }
                    mf_pkg = _save_laya_multifunction_mlpackage(
                        seq_pkgs, pkg_stage / "laya_typed_w8.mlpackage", default_seq_len=512
                    )
                    _publish_multifunction_and_symlinks(mf_pkg, out_dir, eff_seq_lens)
                finally:
                    shutil.rmtree(pkg_stage, ignore_errors=True)
            else:
                for s_len in eff_seq_lens:
                    _compile_laya_seq_bundle(s_len, w_np, enc_cfg, out_dir, guard_ir=guard_ir)

    with acquire_hw_lock("verify_laya_ane_plan"):
        lib = ctypes.CDLL(str(dylib_p))
        lib.laya_ane_inspect_plan_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        mf_mc = out_dir / "laya_typed_w8.mlmodelc"
        for s_len in eff_seq_lens:
            if mf_mc.is_dir():
                mc = mf_mc
                fn_tag = f"s{s_len}".encode("utf-8")
                label = f"{mc.name}[s{s_len}]"
            else:
                mc = out_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
                fn_tag = b"main"
                label = mc.name
            counts = (ctypes.c_int * 3)()
            lib.laya_ane_inspect_plan_fn(str(mc.resolve()).encode("utf-8"), fn_tag, 0, counts)
            print(f"[LAYA-ANE] {label}: MLComputePlan ANE/GPU/CPU = {list(counts)}")
            if int(counts[1]) != 0 or int(counts[2]) != 0 or int(counts[0]) < 1000:
                raise RuntimeError(f"Unexpected non-pure-ANE compute plan on {label}: {list(counts)}")
            if inspect_placement and not mf_mc.is_dir():
                from fast_shared.coreai_pipeline import inspect_coreml_bundle_placement

                pl = inspect_coreml_bundle_placement(mc, target_device="ane", model_name=mc.stem)
                print(f"[LAYA-ANE] {mc.name}: coreai_pipeline placement = {pl.get('placement')}")
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
                dt, _item_sz = dtype_map[st_name]
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
    kev_variant: str = "0.8b",
) -> Path:
    """Merge `jaredpalmer/kev-{0.8b,4b}` LoRA + `PointerHead` into `Qwen3.5-{0.8B,4B}-Base` and quantize."""
    import mlx.core as mx
    from tokenizers import Tokenizer

    is_4b = "4b" in str(kev_variant).strip().lower()
    default_adapter = DEFAULT_KEV_4B_ADAPTER_REPO if is_4b else DEFAULT_KEV_ADAPTER_REPO
    default_base = DEFAULT_KEV_4B_BASE_REPO if is_4b else DEFAULT_KEV_BASE_REPO
    sub_name = "kev_4b" if is_4b else "kev_08b"

    adapter_dir = _resolve_dir_or_hf(kev_adapter_dir, default_adapter)
    base_dir = _resolve_dir_or_hf(kev_base_dir, default_base)
    out_root = artifacts_dir if artifacts_dir is not None else resolve_artifacts_dir(home=repo_root)
    out_dir = out_root / sub_name
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

    with acquire_hw_lock(f"merge_and_pack_{sub_name}"):
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

    tag_ver = "4b" if is_4b else "0.8b-r15"
    meta = {
        "model_name": f"kev-{tag_ver}-merged-bf16",
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
    print(f"[KEV-{tag_ver.upper()}] Saved {out_dir / 'model.safetensors'} ({sz_mb:.2f} MB, {merged_count} LoRA tensors merged, T={native_temp:.4f})")
    quantize_kev_w8_artifacts(
        repo_root,
        artifacts_dir=out_root,
        group_size=32 if is_4b else 64,
        bits=4 if is_4b else 8,
        kev_variant="4b" if is_4b else "0.8b",
    )
    return out_dir


def quantize_kev_w8_artifacts(
    repo_root: Path,
    *,
    artifacts_dir: Path | None = None,
    group_size: int = 64,
    bits: int = 8,
    kev_variant: str = "0.8b",
) -> Path:
    """Quantize merged BF16 Kev `model.safetensors` (`8-bit g64` on `0.8B`; `4-bit g32` + `8-bit g32` GDN on `4B`)."""
    import mlx.core as mx

    is_4b = "4b" in str(kev_variant).strip().lower()
    sub_name = "kev_4b" if is_4b else "kev_08b"
    out_root = artifacts_dir if artifacts_dir is not None else resolve_artifacts_dir(home=repo_root)
    out_dir = out_root / sub_name
    bf16_sf = out_dir / "model.safetensors"
    meta_p = out_dir / "kev_meta.json"
    if not (bf16_sf.exists() and meta_p.exists()):
        raise FileNotFoundError(f"Missing BF16 Kev artifacts in {out_dir}")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    with acquire_hw_lock(f"quantize_{sub_name}"):
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
                    # Never crush GDN projections to 4-bit: keep all 5 GDN linears at 8-bit (`kev-4b-mlx-4bit` rule).
                    eff_bits = 8 if (".linear_attn." in k and bits < 8) else int(bits)
                    wq, scales, biases = mx.quantize(v.astype(mx.bfloat16), group_size=group_size, bits=eff_bits)
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
    meta_w8["model_name"] = f"kev-{'4b' if is_4b else '0.8b-r15'}-merged-w{bits}-g{group_size}"
    meta_w8["quantization"] = {
        "bits": bits,
        "gdn_bits": max(8, int(bits)),
        "group_size": group_size,
        "quantized_matrices": quant_count,
    }
    (out_dir / "kev_meta_w8.json").write_text(json.dumps(meta_w8, indent=2), encoding="utf-8")
    sz_mb = (out_dir / "model_w8.safetensors").stat().st_size / (1024 * 1024)
    print(f"[KEV-W{bits}] Saved {out_dir / 'model_w8.safetensors'} ({sz_mb:.2f} MB, {quant_count} matrices quantized)")
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
        raw_lens = getattr(args, "laya_seq_lens", None)
        seq_lens = (
            tuple(int(x.strip()) for x in str(raw_lens).split(",") if x.strip())
            if raw_lens
            else None
        )
        build_laya_ane_artifacts(
            repo_root,
            artifacts_dir=out_artifacts,
            laya_dir=args.laya_dir,
            copy_compiled_from=args.copy_compiled_laya_from,
            backend=args.backend,
            coreai_preset=args.coreai_preset,
            block_size=args.block_size,
            calib_batches=args.calib_batches,
            guard_ir=args.guard_ir,
            inspect_placement=args.inspect_placement,
            seq_lens=seq_lens,
            multifunction=bool(getattr(args, "multifunction", True)),
        )
    kev_var = getattr(args, "kev_variant", "0.8b")
    if args.build_kev_mlx or run_all:
        build_kev_mlx_artifacts(
            repo_root,
            artifacts_dir=out_artifacts,
            kev_adapter_dir=args.kev_adapter_dir,
            kev_base_dir=args.kev_base_dir,
            kev_variant=kev_var,
        )
    elif args.quantize_existing_kev_w8:
        is_4b = "4b" in str(kev_var).lower()
        quantize_kev_w8_artifacts(
            repo_root,
            artifacts_dir=out_artifacts,
            group_size=32 if is_4b else 64,
            bits=4 if is_4b else 8,
            kev_variant=kev_var,
        )
    return 0


def add_setup_weights_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default=None, help="Override fast-decision-models root directory")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory")
    parser.add_argument("--compile-native", action="store_true", help="Compile native/laya_ane_runner.mm -> build/liblaya_ane_runner.dylib")
    parser.add_argument("--build-laya-ane", action="store_true", help="Build & stage Laya-Typed-421M CoreML ANE W8A16 bundles")
    parser.add_argument("--laya-dir", default=None, help="Local Laya-Typed checkpoint dir (defaults to HF download)")
    parser.add_argument("--copy-compiled-laya-from", default=None, help="Dir containing precompiled Laya .mlmodelc bundles")
    parser.add_argument("--multifunction", action=argparse.BooleanOptionalAction, default=True, help="Merge Laya ANE buckets into shared-weight multifunction .mlmodelc")
    parser.add_argument("--backend", choices=("coreml", "coreai"), default="coreml", help="Laya export backend (CoreML .mlmodelc or Core AI .aimodel)")
    parser.add_argument("--coreai-preset", default=None, help="CoreAI-Opt preset (fp16, w8, w4, w4_per_block, fp4_fp8)")
    parser.add_argument("--block-size", type=int, default=32, help="CoreAI-Opt block quantization size")
    parser.add_argument("--calib-batches", type=int, default=2, help="Calibration batch count for CoreAI-Opt")
    parser.add_argument("--guard-ir", action=argparse.BooleanOptionalAction, default=True, help="Run fast_shared.ir_dtype_guard checks")
    parser.add_argument("--inspect-placement", action="store_true", help="Inspect MLComputePlan / Core AI asset statistics")
    parser.add_argument("--build-kev-mlx", action="store_true", help="Merge & stage Kev BF16 + quantized safetensors")
    parser.add_argument("--quantize-existing-kev-w8", action="store_true", help="Quantize staged Kev model.safetensors -> model_w8.safetensors")
    parser.add_argument("--kev-variant", choices=("0.8b", "4b"), default="0.8b", help="Kev model variant (0.8b or 4b)")
    parser.add_argument("--kev-adapter-dir", default=None, help="Local Kev adapter dir")
    parser.add_argument("--kev-base-dir", default=None, help="Local Qwen3.5-Base dir")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Laya-Typed ANE & Kev MLX decision-model artifacts")
    add_setup_weights_arguments(parser)
    return run_build_args(parser.parse_args(argv))


__all__ = [
    "DEFAULT_LAYA_MAX_MARKERS",
    "DEFAULT_LAYA_SEQ_LENS",
    "FP16_MIN_NORMAL",
    "KEV_SPECIAL_TOKENS",
    "_build_laya_quant_config",
    "_build_laya_seq_mlpackage",
    "_compile_laya_seq_bundle",
    "_laya_layer_res_scale",
    "_laya_layernorm_ch_np",
    "_laya_ln_gain_and_eps",
    "_load_kev_head_tensors",
    "_load_kev_head_tensors_stdlib_zip",
    "_precompute_rope_cos_sin",
    "_precompute_rope_cos_sin_nchw",
    "_precompute_sliding_window_bias",
    "_precompute_sliding_window_bias_bkhq",
    "_save_laya_multifunction_mlpackage",
    "acquire_hw_lock",
    "add_setup_weights_arguments",
    "build_kev_mlx_artifacts",
    "build_laya_ane_artifacts",
    "compile_native_dylib",
    "quantize_kev_w8_artifacts",
    "run_build_args",
]
