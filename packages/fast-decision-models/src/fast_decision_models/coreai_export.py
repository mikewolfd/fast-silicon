"""Domain-specific Core AI (`.aimodel`) & CoreAI-Opt model builders for `fast-decision-models` (`Laya-Typed-421M`).

Builds an exportable 4D-NCHW PyTorch ModernBERT + typed scoring head (`LayaCoreAiModel`)
mirroring `_compile_laya_seq_bundle` (exact `_laya_ln_gain_and_eps` >= 2^-14 FTZ floor,
residual recentering, RoPE cos/sin buffers, additive FP16 SWA/external masks) and delegates
calibration, quantization, sensitivity sweeps, and `.aimodel`/`.mlpackage` export cleanly to
`fast_shared.coreai_pipeline`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
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

from .builder import (
    _laya_layer_res_scale,
    _laya_ln_gain_and_eps,
    _precompute_rope_cos_sin,
    _precompute_sliding_window_bias,
)

DEFAULT_LAYA_CONFIG: dict[str, Any] = {
    "hidden_size": 1024,
    "intermediate_size": 2624,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "local_attention": 128,
    "norm_eps": 1e-5,
    "layer_types": [
        "full_attention" if (i % 3 == 0) else "sliding_attention" for i in range(28)
    ],
    "rope_parameters": {
        "full_attention": {"rope_theta": 160000.0},
        "sliding_attention": {"rope_theta": 10000.0},
    },
}

DEFAULT_LAYA_EXCLUDED_MODULES: frozenset[str] = frozenset(
    {
        # Final 1-channel logit projection keeps output scoring head numerically crisp.
        "scorer_l2",
    }
)

__all__ = [
    "CoreAiExportResult",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_LAYA_CONFIG",
    "DEFAULT_LAYA_EXCLUDED_MODULES",
    "FP16_MIN_NORMAL",
    "VALID_COREAI_PRESETS",
    "build_coreai_opt_laya_config",
    "build_laya_torch_module",
    "resolve_laya_excluded_modules",
    "run_laya_coreai_export",
    "run_laya_coreai_sensitivity_sweep",
]


def resolve_laya_excluded_modules(
    *,
    preserve_scorer_fp16: bool = True,
    extra_excluded: Sequence[str] = (),
) -> list[str]:
    """Return `LayaCoreAiModel` child module names kept uncompressed (`FP16`).

    RoPE cos/sin tables, sliding-window additive masks, and LayerNorm gain/bias
    parameters are registered as buffers (never quantized by CoreAI-Opt).
    """
    excluded: set[str] = set(extra_excluded)
    if preserve_scorer_fp16:
        excluded.update(DEFAULT_LAYA_EXCLUDED_MODULES)
    return sorted(excluded)


def build_coreai_opt_laya_config(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Delegate directly to `fast_shared.coreai_pipeline.build_coreai_opt_config`."""
    return build_coreai_opt_config(**kwargs)


def build_laya_torch_module(
    weights: Mapping[str, np.ndarray] | None = None,
    *,
    enc_cfg: Mapping[str, Any] = DEFAULT_LAYA_CONFIG,
    seq_len: int = 256,
    use_recenter: bool = True,
    seed: int = 0,
):
    """Build an exportable 4D-NCHW PyTorch ModernBERT + typed scoring head module."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    h_dim = int(enc_cfg["hidden_size"])
    inter_dim = int(enc_cfg.get("intermediate_size", h_dim * 2))
    n_layers = int(enc_cfg["num_hidden_layers"])
    n_heads = int(enc_cfg["num_attention_heads"])
    hd = h_dim // n_heads
    attn_scale = 1.0 / math.sqrt(hd)
    layer_types = list(enc_cfg["layer_types"])
    local_window = int(enc_cfg.get("local_attention", 128))
    rope_params = enc_cfg.get("rope_parameters", {})
    theta_full = float(rope_params.get("full_attention", {}).get("rope_theta", 160000.0))
    theta_swa = float(rope_params.get("sliding_attention", {}).get("rope_theta", 10000.0))
    layer_norm_eps = clamp_fp16_eps(float(enc_cfg.get("norm_eps", enc_cfg.get("layer_norm_eps", 1e-5))))

    cos_full_np, sin_full_np = _precompute_rope_cos_sin(seq_len, hd, theta_full)
    cos_swa_np, sin_swa_np = _precompute_rope_cos_sin(seq_len, hd, theta_swa)
    swa_bias_np = _precompute_sliding_window_bias(seq_len, local_window)

    def _ln_ch(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None,
        *,
        gain: float,
        eps_u: float,
    ) -> torch.Tensor:
        xc = x - x.mean(dim=1, keepdim=True)
        u = xc if gain == 1.0 else (xc * gain)
        var_u = (u * u).mean(dim=1, keepdim=True)
        rsq = torch.rsqrt(var_u + eps_u)
        if beta is None:
            return u * rsq * gamma
        return (u * gamma) * rsq + beta

    def _apply_rope(x4: torch.Tensor, cos_c: torch.Tensor, sin_c: torch.Tensor) -> torch.Tensor:
        x0, x1 = torch.chunk(x4, 2, dim=-1)
        return torch.cat([x0 * cos_c - x1 * sin_c, x1 * cos_c + x0 * sin_c], dim=-1)

    class _EncoderLayer(nn.Module):
        def __init__(self, idx: int, is_full: bool) -> None:
            super().__init__()
            self.idx = idx
            self.is_full = is_full
            self.r_scale = _laya_layer_res_scale(idx) if use_recenter else 1.0
            gain, eps_u = _laya_ln_gain_and_eps(self.r_scale, base_inv_scale=0.25, eps=layer_norm_eps)
            self.attn_ln_gain = gain
            self.attn_ln_eps = clamp_fp16_eps(eps_u)
            self.register_buffer("attn_norm", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("mlp_norm", torch.ones(1, h_dim, 1, 1))
            self.wqkv = nn.Conv2d(h_dim, 3 * h_dim, kernel_size=1, bias=False)
            self.wo = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=False)
            self.wi = nn.Conv2d(h_dim, 2 * inter_dim, kernel_size=1, bias=False)
            self.wom = nn.Conv2d(inter_dim, h_dim, kernel_size=1, bias=False)
            self.recenter = (
                (1.0 / 32.0 if idx == 4 else (1.0 / 8.0 if idx == 13 else (1.0 / 4.0 if idx == 18 else 1.0)))
                if (use_recenter and n_layers >= 20)
                else 1.0
            )

        def forward(
            self,
            cur: torch.Tensor,
            m_bias: torch.Tensor,
            cos_c: torch.Tensor,
            sin_c: torch.Tensor,
        ) -> torch.Tensor:
            x_attn = (
                cur
                if self.idx == 0
                else _ln_ch(cur, self.attn_norm, None, gain=self.attn_ln_gain, eps_u=self.attn_ln_eps)
            )
            qkv = self.wqkv(x_attn)
            qt, kt, vt = torch.split(qkv, h_dim, dim=1)
            qh = qt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)
            kh = kt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)
            vh = vt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)

            qrp = _apply_rope(qh, cos_c, sin_c)
            krp = _apply_rope(kh, cos_c, sin_c)
            qk = torch.matmul(qrp, krp.transpose(-1, -2)) * attn_scale
            probs = torch.softmax(qk + m_bias, dim=-1)
            att = torch.matmul(probs, vh)
            att_4d = att.permute(0, 1, 3, 2).reshape(1, h_dim, 1, seq_len)
            cur = cur + self.wo(att_4d)

            x_mlp = _ln_ch(cur, self.mlp_norm, None, gain=self.attn_ln_gain, eps_u=self.attn_ln_eps)
            wi_out = self.wi(x_mlp)
            val_h, gate_h = torch.split(wi_out, inter_dim, dim=1)
            geglu = F.gelu(val_h, approximate="none") * gate_h
            cur = cur + self.wom(geglu)
            if self.recenter != 1.0:
                cur = cur * self.recenter
            return cur

    class _HeadLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            gain, eps_u = _laya_ln_gain_and_eps(1.0, base_inv_scale=0.25, eps=layer_norm_eps)
            self.ln_gain = gain
            self.ln_eps = clamp_fp16_eps(eps_u)
            self.register_buffer("norm1_w", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("norm1_b", torch.zeros(1, h_dim, 1, 1))
            self.register_buffer("norm2_w", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("norm2_b", torch.zeros(1, h_dim, 1, 1))
            self.qkv = nn.Conv2d(h_dim, 3 * h_dim, kernel_size=1, bias=True)
            self.out_proj = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=True)
            self.linear1 = nn.Conv2d(h_dim, inter_dim, kernel_size=1, bias=True)
            self.linear2 = nn.Conv2d(inter_dim, h_dim, kernel_size=1, bias=True)

        def forward(self, h: torch.Tensor, ext_mask: torch.Tensor) -> torch.Tensor:
            hn1 = _ln_ch(h, self.norm1_w, self.norm1_b, gain=self.ln_gain, eps_u=self.ln_eps)
            hqkv = self.qkv(hn1)
            hqt, hkt, hvt = torch.split(hqkv, h_dim, dim=1)
            hqh = hqt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)
            hkh = hkt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)
            hvh = hvt.reshape(1, n_heads, hd, seq_len).permute(0, 1, 3, 2)
            hqk = torch.matmul(hqh, hkh.transpose(-1, -2)) * attn_scale
            hpr = torch.softmax(hqk + ext_mask, dim=-1)
            hatt_4d = torch.matmul(hpr, hvh).permute(0, 1, 3, 2).reshape(1, h_dim, 1, seq_len)
            h = h + self.out_proj(hatt_4d)
            hn2 = _ln_ch(h, self.norm2_w, self.norm2_b, gain=self.ln_gain, eps_u=self.ln_eps)
            return h + self.linear2(F.relu(self.linear1(hn2)))

    class LayaCoreAiModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            emb_gain, emb_eps = _laya_ln_gain_and_eps(1.0, base_inv_scale=1.0, eps=layer_norm_eps)
            final_r = _laya_layer_res_scale(n_layers - 1) if use_recenter else 1.0
            fin_gain, fin_eps = _laya_ln_gain_and_eps(final_r, base_inv_scale=0.25, eps=layer_norm_eps)
            sc_gain, sc_eps = _laya_ln_gain_and_eps(1.0, base_inv_scale=0.25, eps=layer_norm_eps)
            self.emb_gain = emb_gain
            self.emb_eps = clamp_fp16_eps(emb_eps)
            self.fin_gain = fin_gain
            self.fin_eps = clamp_fp16_eps(fin_eps)
            self.sc_gain = sc_gain
            self.sc_eps = clamp_fp16_eps(sc_eps)

            self.register_buffer("cos_full", torch.from_numpy(cos_full_np.astype(np.float32)))
            self.register_buffer("sin_full", torch.from_numpy(sin_full_np.astype(np.float32)))
            self.register_buffer("cos_swa", torch.from_numpy(cos_swa_np.astype(np.float32)))
            self.register_buffer("sin_swa", torch.from_numpy(sin_swa_np.astype(np.float32)))
            self.register_buffer("swa_bias", torch.from_numpy(swa_bias_np.astype(np.float32)))

            self.register_buffer("emb_norm", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("final_norm", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("scorer_norm_w", torch.ones(1, h_dim, 1, 1))
            self.register_buffer("scorer_norm_b", torch.zeros(1, h_dim, 1, 1))

            self.layers = nn.ModuleList(
                [_EncoderLayer(i, layer_types[i] == "full_attention") for i in range(n_layers)]
            )
            self.head_layers = nn.ModuleList([_HeadLayer(), _HeadLayer()])
            self.scorer_l1 = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=True)
            self.scorer_l2 = nn.Conv2d(h_dim, 1, kernel_size=1, bias=True)

        def forward(
            self,
            tok_emb_in: torch.Tensor,
            ext_mask: torch.Tensor,
            type_vec: torch.Tensor,
        ) -> torch.Tensor:
            swa_mask = self.swa_bias + ext_mask
            cur = _ln_ch(tok_emb_in, self.emb_norm, None, gain=self.emb_gain, eps_u=self.emb_eps)
            for layer in self.layers:
                if layer.is_full:
                    cur = layer(cur, ext_mask, self.cos_full, self.sin_full)
                else:
                    cur = layer(cur, swa_mask, self.cos_swa, self.sin_swa)
            h = _ln_ch(cur, self.final_norm, None, gain=self.fin_gain, eps_u=self.fin_eps)
            h = h + type_vec
            for hlayer in self.head_layers:
                h = hlayer(h, ext_mask)
            sn = _ln_ch(h, self.scorer_norm_w, self.scorer_norm_b, gain=self.sc_gain, eps_u=self.sc_eps)
            return self.scorer_l2(F.gelu(self.scorer_l1(sn), approximate="none"))

    torch.manual_seed(int(seed))
    mod = LayaCoreAiModel().eval()
    if weights is not None:
        with torch.no_grad():
            mod.emb_norm.copy_(torch.from_numpy(np.asarray(weights["encoder.embeddings.norm.weight"])).float().reshape(1, h_dim, 1, 1))
            mod.final_norm.copy_(torch.from_numpy(np.asarray(weights["encoder.final_norm.weight"])).float().reshape(1, h_dim, 1, 1))
            mod.scorer_norm_w.copy_(torch.from_numpy(np.asarray(weights["scorer.0.weight"])).float().reshape(1, h_dim, 1, 1))
            mod.scorer_norm_b.copy_(torch.from_numpy(np.asarray(weights["scorer.0.bias"])).float().reshape(1, h_dim, 1, 1))
            mod.scorer_l1.weight.copy_(torch.from_numpy(np.asarray(weights["scorer.1.weight"])).float()[:, :, None, None])
            assert mod.scorer_l1.bias is not None and mod.scorer_l2.bias is not None
            mod.scorer_l1.bias.copy_(torch.from_numpy(np.asarray(weights["scorer.1.bias"])).float())
            mod.scorer_l2.weight.copy_(torch.from_numpy(np.asarray(weights["scorer.3.weight"])).float()[:, :, None, None])
            mod.scorer_l2.bias.copy_(torch.from_numpy(np.asarray(weights["scorer.3.bias"])).float())
            for i, layer in enumerate(mod.layers):
                px = f"encoder.layers.{i}."
                if i > 0:
                    layer.attn_norm.copy_(torch.from_numpy(np.asarray(weights[px + "attn_norm.weight"])).float().reshape(1, h_dim, 1, 1))
                layer.mlp_norm.copy_(torch.from_numpy(np.asarray(weights[px + "mlp_norm.weight"])).float().reshape(1, h_dim, 1, 1))
                layer.wqkv.weight.copy_(torch.from_numpy(np.asarray(weights[px + "attn.Wqkv.weight"])).float()[:, :, None, None])
                layer.wo.weight.copy_(
                    (torch.from_numpy(np.asarray(weights[px + "attn.Wo.weight"])).float() * layer.r_scale)[:, :, None, None]
                )
                layer.wi.weight.copy_(torch.from_numpy(np.asarray(weights[px + "mlp.Wi.weight"])).float()[:, :, None, None])
                layer.wom.weight.copy_(
                    (torch.from_numpy(np.asarray(weights[px + "mlp.Wo.weight"])).float() * layer.r_scale)[:, :, None, None]
                )
            for hi, hlayer in enumerate(mod.head_layers):
                hp = f"head.layers.{hi}."
                hlayer.norm1_w.copy_(torch.from_numpy(np.asarray(weights[hp + "norm1.weight"])).float().reshape(1, h_dim, 1, 1))
                hlayer.norm1_b.copy_(torch.from_numpy(np.asarray(weights[hp + "norm1.bias"])).float().reshape(1, h_dim, 1, 1))
                hlayer.norm2_w.copy_(torch.from_numpy(np.asarray(weights[hp + "norm2.weight"])).float().reshape(1, h_dim, 1, 1))
                hlayer.norm2_b.copy_(torch.from_numpy(np.asarray(weights[hp + "norm2.bias"])).float().reshape(1, h_dim, 1, 1))
                hlayer.qkv.weight.copy_(torch.from_numpy(np.asarray(weights[hp + "self_attn.in_proj_weight"])).float()[:, :, None, None])
                assert hlayer.qkv.bias is not None and hlayer.out_proj.bias is not None
                hlayer.qkv.bias.copy_(torch.from_numpy(np.asarray(weights[hp + "self_attn.in_proj_bias"])).float())
                hlayer.out_proj.weight.copy_(torch.from_numpy(np.asarray(weights[hp + "self_attn.out_proj.weight"])).float()[:, :, None, None])
                hlayer.out_proj.bias.copy_(torch.from_numpy(np.asarray(weights[hp + "self_attn.out_proj.bias"])).float())
                hlayer.linear1.weight.copy_(torch.from_numpy(np.asarray(weights[hp + "linear1.weight"])).float()[:, :, None, None])
                assert hlayer.linear1.bias is not None and hlayer.linear2.bias is not None
                hlayer.linear1.bias.copy_(torch.from_numpy(np.asarray(weights[hp + "linear1.bias"])).float())
                hlayer.linear2.weight.copy_(torch.from_numpy(np.asarray(weights[hp + "linear2.weight"])).float()[:, :, None, None])
                hlayer.linear2.bias.copy_(torch.from_numpy(np.asarray(weights[hp + "linear2.bias"])).float())
    return mod


def _laya_sample_factory(hidden_size: int, *, seq_len: int):
    import torch

    def _make(seed: int) -> tuple[Any, ...]:
        g = torch.Generator().manual_seed(int(seed))
        valid_len = max(4, min(seq_len, int(seq_len * 0.75)))
        mask = torch.zeros((1, 1, 1, seq_len), dtype=torch.float32)
        if valid_len < seq_len:
            mask[:, :, :, valid_len:] = -10000.0
        return (
            torch.randn((1, hidden_size, 1, seq_len), generator=g) * 0.25,
            mask,
            torch.randn((1, hidden_size, 1, 1), generator=g) * 0.1,
        )

    return _make


def _laya_ml_specs(hidden_size: int, *, seq_len: int) -> tuple[list[Any], list[Any]]:
    import coremltools as ct

    return (
        [
            ct.TensorType(name="tok_emb_in", shape=(1, int(hidden_size), 1, int(seq_len)), dtype=np.float16),
            ct.TensorType(name="ext_mask", shape=(1, 1, 1, int(seq_len)), dtype=np.float16),
            ct.TensorType(name="type_vec", shape=(1, int(hidden_size), 1, 1), dtype=np.float16),
        ],
        [ct.TensorType(name="pos_logits", dtype=np.float16)],
    )


def run_laya_coreai_export(
    *,
    out_dir: str | Path,
    weights: Mapping[str, np.ndarray] | None = None,
    enc_cfg: Mapping[str, Any] = DEFAULT_LAYA_CONFIG,
    seq_len: int = 256,
    export_backend: str = "coreai",
    preset: str = "w4_per_block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    preserve_scorer_fp16: bool = True,
    calib_batches: int = 2,
    use_recenter: bool = True,
    stem: str = "laya_typed",
    inspect_placement: bool = False,
) -> CoreAiExportResult:
    """Calibrate, quantize (CoreAI-Opt), export (`.aimodel`/`.mlpackage`), and guard Laya-Typed."""
    h_dim = int(enc_cfg["hidden_size"])
    excluded = resolve_laya_excluded_modules(preserve_scorer_fp16=preserve_scorer_fp16)
    ml_in, ml_out = _laya_ml_specs(h_dim, seq_len=seq_len) if export_backend == "coreml" else (None, None)
    return run_coreai_opt_calibration_and_export(
        module_factory=lambda: build_laya_torch_module(
            weights,
            enc_cfg=enc_cfg,
            seq_len=seq_len,
            use_recenter=use_recenter,
        ),
        sample_inputs_factory=_laya_sample_factory(h_dim, seq_len=seq_len),
        out_dir=out_dir,
        stem=stem,
        model_name="Laya-Typed-421M",
        input_names=["tok_emb_in", "ext_mask", "type_vec"],
        output_names=["pos_logits"],
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
            "num_hidden_layers": int(enc_cfg["num_hidden_layers"]),
            "preserve_scorer_fp16": bool(preserve_scorer_fp16),
        },
        flag_add_fp16_cast=False,
        inspect_placement=inspect_placement,
        target_device="ane",
    )


def run_laya_coreai_sensitivity_sweep(
    *,
    out_dir: str | Path,
    weights: Mapping[str, np.ndarray] | None = None,
    enc_cfg: Mapping[str, Any] = DEFAULT_LAYA_CONFIG,
    seq_len: int = 256,
    export_backend: str = "coreml",
    presets: Sequence[str] = ("fp16", "w8", "w4", "w4_per_block"),
    block_size: int = DEFAULT_BLOCK_SIZE,
    preserve_scorer_fp16: bool = True,
    calib_batches: int = 2,
    use_recenter: bool = True,
    stem_prefix: str = "laya_typed",
) -> list[dict[str, Any]]:
    """Run a calibration sensitivity & BPW sweep across `CoreAI-Opt` presets on Laya-Typed."""
    h_dim = int(enc_cfg["hidden_size"])
    excluded = resolve_laya_excluded_modules(preserve_scorer_fp16=preserve_scorer_fp16)
    ml_in, ml_out = _laya_ml_specs(h_dim, seq_len=seq_len) if export_backend == "coreml" else (None, None)
    return run_coreai_opt_preset_sweep(
        module_factory=lambda: build_laya_torch_module(
            weights,
            enc_cfg=enc_cfg,
            seq_len=seq_len,
            use_recenter=use_recenter,
        ),
        sample_inputs_factory=_laya_sample_factory(h_dim, seq_len=seq_len),
        out_dir=out_dir,
        stem_prefix=stem_prefix,
        model_name="Laya-Typed-421M",
        input_names=["tok_emb_in", "ext_mask", "type_vec"],
        output_names=["pos_logits"],
        ml_inputs=ml_in,
        ml_outputs=ml_out,
        export_backend=export_backend,
        presets=presets,
        block_size=block_size,
        block_axis=1,
        excluded_modules=excluded,
        calib_batches=calib_batches,
        spec_metadata={"seq_len": int(seq_len), "num_hidden_layers": int(enc_cfg["num_hidden_layers"])},
    )
