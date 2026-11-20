"""Domain-specific Core AI (`.aimodel`) & CoreAI-Opt model builders for `fast-pplx` (`Qwen3.5`).

Builds exportable 4D-NCHW PyTorch stage modules (`PplxCoreAiStageBlock`) matching
`build_stage_mil_program` (`2^-14` RMSNorm epsilon floor, `ftz_alpha` 1-homogeneous GDN
scaling, ANE-safe `relu + log(1 + exp(-|x|))` softplus, additive FP16/float causal
attention masks) and delegates quantization, sensitivity sweeps, Pattern-7/IR guarding,
and `.aimodel` (default FP16 compute) / `.mlpackage` export to `fast_shared.coreai_pipeline`.
Note: hand-crafted MIL `build_stage_mil_program` (`2.24 ms` pure ANE) remains the primary
backend until an `.aimodel` candidate clears Two-Gate promotion against `hetero2`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fast_shared.coreai_pipeline import (
    CoreAiExportResult,
    DEFAULT_BLOCK_SIZE,
    FP16_MIN_NORMAL,
    VALID_COREAI_PRESETS,
    build_coreai_opt_config,
    clamp_fp16_eps,
    run_coreai_opt_calibration_and_export,
    run_coreai_opt_preset_sweep,
)

from .config import DEFAULT_PPLX_CONFIG, PplxConfig

DEFAULT_CHUNK_SIZE = 32
DEFAULT_FTZ_ALPHA = 128.0

# Projections whose calibration absmax >= 50 stay in FP16 (aligned with build_ane_models.py).
DEFAULT_PPLX_OUTLIER_KEYS: frozenset[str] = frozenset(
    {
        "layers.5.mlp.down_proj",
        "layers.7.mlp.down_proj",
        "layers.8.mlp.down_proj",
        "layers.9.mlp.down_proj",
        "layers.10.mlp.down_proj",
        "layers.11.mlp.down_proj",
        "layers.8.linear_attn.out_proj",
        "layers.10.linear_attn.out_proj",
    }
)

_HF_OUTLIER_TO_CONV_SUFFIX: dict[str, str] = {
    "mlp.down_proj": "mlp_dn",
    "mlp.gate_proj": "mlp_gu",
    "mlp.up_proj": "mlp_gu",
    "linear_attn.in_proj_qkv": "qkvzba",
    "linear_attn.out_proj": "out_proj",
    "self_attn.q_proj": "qkvgate",
    "self_attn.o_proj": "o_proj",
}

__all__ = [
    "CoreAiExportResult",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_FTZ_ALPHA",
    "DEFAULT_PPLX_OUTLIER_KEYS",
    "FP16_MIN_NORMAL",
    "VALID_COREAI_PRESETS",
    "build_coreai_opt_pplx_config",
    "build_pplx_torch_stage_module",
    "clamp_fp16_eps",
    "resolve_pplx_stage_excluded_modules",
    "run_pplx_coreai_export",
    "run_pplx_coreai_sensitivity_sweep",
]


def resolve_pplx_stage_excluded_modules(
    layer_start: int,
    layer_end: int,
    layer_types: Sequence[str],
    *,
    preserve_gdn_fp16: bool = True,
    preserve_outliers: bool = True,
    outlier_keys: Sequence[str] = (),
    extra_excluded: Sequence[str] = (),
) -> list[str]:
    """Return `PplxCoreAiStageBlock` child module names kept uncompressed (`FP16`)."""
    excluded: set[str] = set(extra_excluded)
    active_outliers = set(DEFAULT_PPLX_OUTLIER_KEYS) | set(outlier_keys)

    for li in range(layer_start, layer_end):
        p = f"blocks.l{li}"
        lt = layer_types[li]
        if lt == "linear_attention":
            # Depthwise 1x4 conv has C_in_per_group=1: always keep uncompressed (Pattern 7).
            excluded.add(f"{p}.conv1d")
            if preserve_gdn_fp16:
                excluded.add(f"{p}.qkvzba")
                excluded.add(f"{p}.out_proj")
        if preserve_outliers:
            for key in active_outliers:
                parts = key.split(".", 2)
                if len(parts) == 3 and parts[0] == "layers" and int(parts[1]) == li:
                    suf = _HF_OUTLIER_TO_CONV_SUFFIX.get(parts[2])
                    if suf is not None:
                        excluded.add(f"{p}.{suf}")
    return sorted(excluded)


def build_coreai_opt_pplx_config(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Delegate directly to `fast_shared.coreai_pipeline.build_coreai_opt_config`."""
    return build_coreai_opt_config(**kwargs)


def build_pplx_torch_stage_module(
    ane_weights: Mapping[str, np.ndarray] | None = None,
    *,
    cfg: PplxConfig = DEFAULT_PPLX_CONFIG,
    layer_start: int = 0,
    layer_end: int | None = None,
    apply_final_norm: bool | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    ftz_alpha: float = DEFAULT_FTZ_ALPHA,
    seed: int = 0,
):
    """Build an exportable 4D-NCHW PyTorch stage module matching `build_stage_mil_program`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    eff_end = cfg.num_hidden_layers if layer_end is None else int(layer_end)
    eff_final = (eff_end == cfg.num_hidden_layers) if apply_final_norm is None else bool(apply_final_norm)
    h_dim = int(cfg.hidden_size)
    mlp_inter = int(cfg.intermediate_size)
    gdn_heads = int(cfg.linear_num_value_heads)
    gdn_dim = int(cfg.linear_key_head_dim)
    gdn_proj_dim = gdn_heads * gdn_dim
    conv_k = int(cfg.linear_conv_kernel_dim)
    attn_q_heads = int(cfg.num_attention_heads)
    attn_kv_heads = int(cfg.num_key_value_heads)
    attn_head_dim = int(cfg.head_dim)
    rope_half = int(cfg.rope_dim) // 2
    gqa_rep = attn_q_heads // attn_kv_heads
    rms_eps = clamp_fp16_eps(cfg.rms_norm_eps)
    gnorm_eps = clamp_fp16_eps(float(cfg.rms_norm_eps) * (float(ftz_alpha) ** 2))

    def _rms_ch(x: torch.Tensor, w: torch.Tensor, axis: int, eps: float) -> torch.Tensor:
        var = (x * x).mean(dim=axis, keepdim=True)
        return x * torch.rsqrt(var + eps) * w

    class _LinearAttnLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("in_ln", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("post_ln", torch.ones(1, h_dim, 1, 1))
            self.qkvzba = nn.Conv2d(h_dim, 4 * gdn_proj_dim + 2 * gdn_heads, kernel_size=1, bias=False)
            self.conv1d = nn.Conv2d(
                3 * gdn_proj_dim,
                3 * gdn_proj_dim,
                kernel_size=(1, conv_k),
                groups=3 * gdn_proj_dim,
                bias=False,
            )
            self.register_buffer("neg_a", torch.full((1, gdn_heads, 1, 1), -1.0))
            self.register_buffer("dt_bias", torch.zeros(1, gdn_heads, 1, 1))
            self.register_buffer("gnorm_w", torch.ones(1, 1, gdn_dim, 1))
            self.out_proj = nn.Conv2d(gdn_proj_dim, h_dim, kernel_size=1, bias=False)
            self.mlp_gu = nn.Conv2d(h_dim, 2 * mlp_inter, kernel_size=1, bias=False)
            self.mlp_dn = nn.Conv2d(mlp_inter, h_dim, kernel_size=1, bias=False)

            tril = torch.tril(torch.ones((chunk_size, chunk_size)), diagonal=0).view(1, 1, chunk_size, chunk_size)
            strict_neg = -torch.tril(torch.ones((chunk_size, chunk_size)), diagonal=-1).view(
                1, 1, chunk_size, chunk_size
            )
            neg100_upper = -100.0 * torch.triu(torch.ones((chunk_size, chunk_size)), diagonal=1).view(
                1, 1, chunk_size, chunk_size
            )
            self.register_buffer("_tril", tril, persistent=False)
            self.register_buffer("_strict_neg", strict_neg, persistent=False)
            self.register_buffer("_neg100_upper", neg100_upper, persistent=False)

        def forward(self, h_in: torch.Tensor, tok_valid: torch.Tensor, gdn_carry: torch.Tensor) -> torch.Tensor:
            seq_len = int(h_in.shape[3])
            n_c = seq_len // chunk_size
            nc_h = n_c * gdn_heads
            qkv_len = 3 * gdn_proj_dim

            xn = _rms_ch(h_in, self.in_ln, axis=1, eps=rms_eps)
            qkvzba = self.qkvzba(xn)
            mqkv = qkvzba[:, :qkv_len, :, :] * tok_valid
            z = qkvzba[:, qkv_len : qkv_len + gdn_proj_dim, :, :]
            b_ch = qkvzba[:, qkv_len + gdn_proj_dim : qkv_len + gdn_proj_dim + gdn_heads, :, :]
            a_ch = qkvzba[:, qkv_len + gdn_proj_dim + gdn_heads :, :, :]

            conv_act = F.silu(self.conv1d(F.pad(mqkv, (conv_k - 1, 0, 0, 0))))
            q_ch = conv_act[:, :gdn_proj_dim, :, :]
            k_ch = conv_act[:, gdn_proj_dim : 2 * gdn_proj_dim, :, :]
            v_raw = conv_act[:, 2 * gdn_proj_dim :, :, :]
            v_ch = v_raw * float(ftz_alpha) if ftz_alpha != 1.0 else v_raw

            def _to_b1(t: torch.Tensor) -> torch.Tensor:
                return (
                    t.reshape(1, gdn_proj_dim, n_c, chunk_size)
                    .permute(0, 2, 1, 3)
                    .reshape(1, nc_h, gdn_dim, chunk_size)
                    .permute(0, 1, 3, 2)
                )

            q_4d, k_4d, v_4d = _to_b1(q_ch), _to_b1(k_ch), _to_b1(v_ch)
            q_norm = q_4d * torch.rsqrt((q_4d * q_4d).sum(dim=-1, keepdim=True) + FP16_MIN_NORMAL) * (gdn_dim**-0.5)
            k_norm = k_4d * torch.rsqrt((k_4d * k_4d).sum(dim=-1, keepdim=True) + FP16_MIN_NORMAL)

            beta = torch.sigmoid(b_ch).reshape(1, gdn_heads, n_c, chunk_size).permute(0, 2, 1, 3).reshape(
                1, nc_h, chunk_size, 1
            )
            # Avoid PyTorch `F.softplus` so TorchConverter/CoreML keep ANE-eligible primitive ops.
            g_in = a_ch + self.dt_bias
            g_raw = self.neg_a * (F.relu(g_in) + torch.log(1.0 + torch.exp(-torch.abs(g_in))))
            g_4d = g_raw.reshape(1, gdn_heads, n_c, chunk_size).permute(0, 2, 1, 3).reshape(1, nc_h, 1, chunk_size)

            cum_decay = torch.matmul(g_4d, self._tril.transpose(-1, -2))
            cum_col = cum_decay.permute(0, 1, 3, 2)
            pairwise = cum_col - cum_decay
            pw_exp = torch.exp(pairwise * self._tril + self._neg100_upper) * self._tril

            v_beta = v_4d * beta
            k_beta = k_norm * beta
            ut = torch.matmul(k_beta, k_norm.transpose(-1, -2)) * pw_exp
            intra = torch.matmul(q_norm, k_norm.transpose(-1, -2)) * pw_exp

            eye = torch.eye(chunk_size, dtype=ut.dtype, device=ut.device).view(1, 1, chunk_size, chunk_size)
            m_strict = ut * self._strict_neg
            t_mat = eye + m_strict
            m_pow = m_strict
            b = 2
            while b < chunk_size:
                m_pow = torch.matmul(m_pow, m_pow)
                t_mat = t_mat + torch.matmul(t_mat, m_pow)
                b *= 2

            exp_cum = torch.exp(cum_col)
            new_values = torch.matmul(t_mat, v_beta)
            k_cumdecay = torch.matmul(t_mat, k_beta * exp_cum)
            q_dec = q_norm * exp_cum
            cum_last = cum_col[:, :, chunk_size - 1 : chunk_size, :]
            k_dec = k_norm * torch.exp(cum_last - cum_col)
            chunk_dec = torch.exp(cum_last)

            outs = []
            state = None
            for i in range(n_c):
                h0, h1 = i * gdn_heads, (i + 1) * gdn_heads
                nv_i = new_values[:, h0:h1, :, :]
                intra_i = intra[:, h0:h1, :, :]
                q_dec_i = q_dec[:, h0:h1, :, :]
                k_dec_i = k_dec[:, h0:h1, :, :]
                c_dec_i = chunk_dec[:, h0:h1, :, :]
                if i == 0:
                    v_new = nv_i
                    out_i = torch.matmul(intra_i, v_new)
                    state = torch.matmul(k_dec_i.transpose(-1, -2), v_new)
                else:
                    assert state is not None
                    gated_state = state * gdn_carry[:, i : i + 1, :, :]
                    v_new = nv_i - torch.matmul(k_cumdecay[:, h0:h1, :, :], gated_state)
                    out_i = torch.matmul(q_dec_i, gated_state) + torch.matmul(intra_i, v_new)
                    state = gated_state * c_dec_i + torch.matmul(k_dec_i.transpose(-1, -2), v_new)
                outs.append(out_i)

            delta_nchw = (
                torch.cat(outs, dim=1)
                .permute(0, 1, 3, 2)
                .reshape(1, n_c, gdn_proj_dim, chunk_size)
                .permute(0, 2, 1, 3)
                .reshape(1, gdn_proj_dim, 1, seq_len)
            )
            normed_hd = _rms_ch(delta_nchw.reshape(1, gdn_heads, gdn_dim, seq_len), self.gnorm_w, axis=2, eps=gnorm_eps)
            gated_nchw = (normed_hd * F.silu(z.reshape(1, gdn_heads, gdn_dim, seq_len))).reshape(
                1, gdn_proj_dim, 1, seq_len
            )

            h_mid = h_in + self.out_proj(gated_nchw)
            gu = self.mlp_gu(_rms_ch(h_mid, self.post_ln, axis=1, eps=rms_eps))
            return h_mid + self.mlp_dn(F.silu(gu[:, :mlp_inter, :, :]) * gu[:, mlp_inter:, :, :])

    class _FullAttnLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("in_ln", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("post_ln", torch.ones(1, h_dim, 1, 1))
            q_total = attn_q_heads * attn_head_dim
            kv_total = attn_kv_heads * attn_head_dim
            self.qkvgate = nn.Conv2d(h_dim, 2 * q_total + 2 * kv_total, kernel_size=1, bias=False)
            self.register_buffer("qn", torch.ones(1, 1, 1, attn_head_dim))
            self.register_buffer("kn", torch.ones(1, 1, 1, attn_head_dim))
            self.o_proj = nn.Conv2d(q_total, h_dim, kernel_size=1, bias=False)
            self.mlp_gu = nn.Conv2d(h_dim, 2 * mlp_inter, kernel_size=1, bias=False)
            self.mlp_dn = nn.Conv2d(mlp_inter, h_dim, kernel_size=1, bias=False)

        def forward(
            self,
            h_in: torch.Tensor,
            attn_bias: torch.Tensor,
            cos32: torch.Tensor,
            sin32: torch.Tensor,
        ) -> torch.Tensor:
            seq_len = int(h_in.shape[3])
            q_total = attn_q_heads * attn_head_dim
            kv_total = attn_kv_heads * attn_head_dim
            qkvg = self.qkvgate(_rms_ch(h_in, self.in_ln, axis=1, eps=rms_eps))
            q_ch = qkvg[:, :q_total, :, :]
            k_ch = qkvg[:, q_total : q_total + kv_total, :, :]
            v_ch = qkvg[:, q_total + kv_total : q_total + 2 * kv_total, :, :]
            gate_ch = qkvg[:, q_total + 2 * kv_total :, :, :]

            q_n = _rms_ch(q_ch.reshape(1, attn_q_heads, attn_head_dim, seq_len).permute(0, 1, 3, 2), self.qn, axis=3, eps=rms_eps)
            k_n = _rms_ch(k_ch.reshape(1, attn_kv_heads, attn_head_dim, seq_len).permute(0, 1, 3, 2), self.kn, axis=3, eps=rms_eps)
            v_hsd = v_ch.reshape(1, attn_kv_heads, attn_head_dim, seq_len).permute(0, 1, 3, 2)

            def _rope(x: torch.Tensor) -> torch.Tensor:
                r1 = x[:, :, :, :rope_half]
                r2 = x[:, :, :, rope_half : 2 * rope_half]
                return torch.cat([r1 * cos32 - r2 * sin32, r2 * cos32 + r1 * sin32, x[:, :, :, 2 * rope_half :]], dim=3)

            k_rep = torch.repeat_interleave(_rope(k_n), repeats=gqa_rep, dim=1)
            v_rep = torch.repeat_interleave(v_hsd, repeats=gqa_rep, dim=1)
            probs = torch.softmax(torch.matmul(_rope(q_n) * (attn_head_dim**-0.5), k_rep.transpose(-1, -2)) + attn_bias, dim=3)
            attn_nchw = torch.matmul(probs, v_rep).permute(0, 1, 3, 2).reshape(1, q_total, 1, seq_len)
            h_mid = h_in + self.o_proj(attn_nchw * torch.sigmoid(gate_ch))
            gu = self.mlp_gu(_rms_ch(h_mid, self.post_ln, axis=1, eps=rms_eps))
            return h_mid + self.mlp_dn(F.silu(gu[:, :mlp_inter, :, :]) * gu[:, mlp_inter:, :, :])

    class PplxCoreAiStageBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleDict()
            self.layer_specs: list[tuple[str, str]] = []
            for li in range(layer_start, eff_end):
                lt = cfg.layer_types[li]
                key = f"l{li}"
                self.blocks[key] = _LinearAttnLayer() if lt == "linear_attention" else _FullAttnLayer()
                self.layer_specs.append((key, lt))
            self.apply_final_norm = eff_final
            if eff_final:
                self.register_buffer("final_norm", torch.ones(1, h_dim, 1, 1))

        def forward(
            self,
            h_in: torch.Tensor,
            attn_bias: torch.Tensor,
            cos32: torch.Tensor,
            sin32: torch.Tensor,
            tok_valid: torch.Tensor,
            gdn_carry: torch.Tensor,
        ) -> torch.Tensor:
            h = h_in
            for key, lt in self.layer_specs:
                blk = self.blocks[key]
                h = blk(h, tok_valid, gdn_carry) if lt == "linear_attention" else blk(h, attn_bias, cos32, sin32)
            return _rms_ch(h, self.final_norm, axis=1, eps=rms_eps) if self.apply_final_norm else h

    torch.manual_seed(int(seed))
    mod = PplxCoreAiStageBlock().eval()
    if ane_weights is not None:
        with torch.no_grad():
            for key, lt in mod.layer_specs:
                blk = mod.blocks[key]
                blk.in_ln.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_in_ln"])).float())
                blk.post_ln.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_post_ln"])).float())
                blk.mlp_gu.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_mlp_gu"])).float())
                blk.mlp_dn.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_mlp_dn"])).float())
                if lt == "linear_attention":
                    blk.qkvzba.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_qkvzba"])).float())
                    blk.conv1d.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_conv1d"])).float())
                    blk.neg_a.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_neg_A"])).float().reshape(1, gdn_heads, 1, 1))
                    blk.dt_bias.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_dt_bias"])).float().reshape(1, gdn_heads, 1, 1))
                    blk.gnorm_w.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_gnorm_w"])).float().reshape(1, 1, gdn_dim, 1))
                    blk.out_proj.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_out_proj"])).float())
                else:
                    blk.qkvgate.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_qkvgate"])).float())
                    blk.qn.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_qn"])).float())
                    blk.kn.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_kn"])).float())
                    blk.o_proj.weight.copy_(torch.from_numpy(np.asarray(ane_weights[f"{key}_o_proj"])).float())
            if eff_final:
                mod.final_norm.copy_(torch.from_numpy(np.asarray(ane_weights["final_norm"])).float())
    return mod


def _pplx_sample_factory(cfg: PplxConfig, *, seq_len: int, chunk_size: int):
    import torch

    rope_half = int(cfg.rope_dim) // 2
    n_c = int(seq_len) // int(chunk_size)

    def _make(seed: int) -> tuple[Any, ...]:
        g = torch.Generator().manual_seed(int(seed))
        causal = np.triu(np.full((seq_len, seq_len), -10000.0, dtype=np.float32), k=1)[None, None, :, :]
        angles = torch.linspace(0.0, 1.0, seq_len * rope_half).reshape(1, 1, seq_len, rope_half)
        return (
            torch.randn((1, int(cfg.hidden_size), 1, seq_len), generator=g) * 0.25,
            torch.from_numpy(causal),
            torch.cos(angles),
            torch.sin(angles),
            torch.ones((1, 1, 1, seq_len)),
            torch.ones((1, n_c, 1, 1)),
        )

    return _make


def _pplx_ml_specs(cfg: PplxConfig, *, seq_len: int, chunk_size: int) -> tuple[list[Any], list[Any]]:
    import coremltools as ct

    rope_half = int(cfg.rope_dim) // 2
    n_c = int(seq_len) // int(chunk_size)
    return (
        [
            ct.TensorType(name="h_in", shape=(1, int(cfg.hidden_size), 1, int(seq_len)), dtype=np.float16),
            ct.TensorType(name="attn_bias", shape=(1, 1, int(seq_len), int(seq_len)), dtype=np.float16),
            ct.TensorType(name="cos32", shape=(1, 1, int(seq_len), rope_half), dtype=np.float16),
            ct.TensorType(name="sin32", shape=(1, 1, int(seq_len), rope_half), dtype=np.float16),
            ct.TensorType(name="tok_valid", shape=(1, 1, 1, int(seq_len)), dtype=np.float16),
            ct.TensorType(name="gdn_carry", shape=(1, n_c, 1, 1), dtype=np.float16),
        ],
        [ct.TensorType(name="h_out", dtype=np.float16)],
    )


def run_pplx_coreai_export(
    *,
    out_dir: str | Path,
    ane_weights: Mapping[str, np.ndarray] | None = None,
    cfg: PplxConfig = DEFAULT_PPLX_CONFIG,
    layer_start: int = 0,
    layer_end: int | None = None,
    apply_final_norm: bool | None = None,
    seq_len: int = 64,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    export_backend: str = "coreai",
    preset: str = "w4_per_block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    preserve_gdn_fp16: bool = True,
    preserve_outliers: bool = True,
    outlier_keys: Sequence[str] = (),
    calib_batches: int = 2,
    ftz_alpha: float = DEFAULT_FTZ_ALPHA,
    stem: str = "pplx_stage",
    inspect_placement: bool = False,
    allow_experimental_coreml_w4_block: bool = False,
    compute_dtype: str = "auto",
) -> CoreAiExportResult:
    """Calibrate, quantize (CoreAI-Opt), export (`.aimodel`/`.mlpackage`), and guard a PPLX stage."""
    eff_end = cfg.num_hidden_layers if layer_end is None else int(layer_end)
    excluded = resolve_pplx_stage_excluded_modules(
        layer_start,
        eff_end,
        cfg.layer_types,
        preserve_gdn_fp16=preserve_gdn_fp16,
        preserve_outliers=preserve_outliers,
        outlier_keys=outlier_keys,
    )
    ml_in, ml_out = _pplx_ml_specs(cfg, seq_len=seq_len, chunk_size=chunk_size) if export_backend == "coreml" else (None, None)
    return run_coreai_opt_calibration_and_export(
        module_factory=lambda: build_pplx_torch_stage_module(
            ane_weights,
            cfg=cfg,
            layer_start=layer_start,
            layer_end=eff_end,
            apply_final_norm=apply_final_norm,
            chunk_size=chunk_size,
            ftz_alpha=ftz_alpha,
        ),
        sample_inputs_factory=_pplx_sample_factory(cfg, seq_len=seq_len, chunk_size=chunk_size),
        out_dir=out_dir,
        stem=stem,
        model_name="perplexity-ai/pplx-embed-v2-context-0.6b",
        input_names=["h_in", "attn_bias", "cos32", "sin32", "tok_valid", "gdn_carry"],
        output_names=["h_out"],
        ml_inputs=ml_in,
        ml_outputs=ml_out,
        export_backend=export_backend,
        preset=preset,
        block_size=block_size,
        block_axis=1,
        excluded_modules=excluded,
        calib_batches=calib_batches,
        spec_metadata={
            "seq_len": int(seq_len),
            "layer_start": int(layer_start),
            "layer_end": eff_end,
            "preserve_gdn_fp16": bool(preserve_gdn_fp16),
            "preserve_outliers": bool(preserve_outliers),
        },
        flag_add_fp16_cast=False,
        inspect_placement=inspect_placement,
        target_device="ane",
        allow_experimental_coreml_w4_block=allow_experimental_coreml_w4_block,
        compute_dtype=compute_dtype,
    )


def run_pplx_coreai_sensitivity_sweep(
    *,
    out_dir: str | Path,
    ane_weights: Mapping[str, np.ndarray] | None = None,
    cfg: PplxConfig = DEFAULT_PPLX_CONFIG,
    layer_start: int = 0,
    layer_end: int | None = None,
    apply_final_norm: bool | None = None,
    seq_len: int = 64,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    export_backend: str = "coreml",
    presets: Sequence[str] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    preserve_gdn_fp16: bool = True,
    preserve_outliers: bool = True,
    outlier_keys: Sequence[str] = (),
    calib_batches: int = 2,
    ftz_alpha: float = DEFAULT_FTZ_ALPHA,
    stem_prefix: str = "pplx_stage",
    allow_experimental_coreml_w4_block: bool = False,
    compute_dtype: str = "auto",
) -> list[dict[str, Any]]:
    """Run a calibration sensitivity & BPW sweep across `CoreAI-Opt` presets on a PPLX stage."""
    eff_end = cfg.num_hidden_layers if layer_end is None else int(layer_end)
    excluded = resolve_pplx_stage_excluded_modules(
        layer_start,
        eff_end,
        cfg.layer_types,
        preserve_gdn_fp16=preserve_gdn_fp16,
        preserve_outliers=preserve_outliers,
        outlier_keys=outlier_keys,
    )
    ml_in, ml_out = _pplx_ml_specs(cfg, seq_len=seq_len, chunk_size=chunk_size) if export_backend == "coreml" else (None, None)
    return run_coreai_opt_preset_sweep(
        module_factory=lambda: build_pplx_torch_stage_module(
            ane_weights,
            cfg=cfg,
            layer_start=layer_start,
            layer_end=eff_end,
            apply_final_norm=apply_final_norm,
            chunk_size=chunk_size,
            ftz_alpha=ftz_alpha,
        ),
        sample_inputs_factory=_pplx_sample_factory(cfg, seq_len=seq_len, chunk_size=chunk_size),
        out_dir=out_dir,
        stem_prefix=stem_prefix,
        model_name="perplexity-ai/pplx-embed-v2-context-0.6b",
        input_names=["h_in", "attn_bias", "cos32", "sin32", "tok_valid", "gdn_carry"],
        output_names=["h_out"],
        ml_inputs=ml_in,
        ml_outputs=ml_out,
        export_backend=export_backend,
        presets=presets,
        block_size=block_size,
        block_axis=1,
        excluded_modules=excluded,
        calib_batches=calib_batches,
        spec_metadata={"seq_len": int(seq_len), "layer_start": int(layer_start), "layer_end": eff_end},
        allow_experimental_coreml_w4_block=allow_experimental_coreml_w4_block,
        compute_dtype=compute_dtype,
    )
