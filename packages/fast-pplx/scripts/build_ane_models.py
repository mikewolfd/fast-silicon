#!/usr/bin/env python3
"""Build pure-ANE staged CoreML bundles for pplx-embed-v2-context-0.6b.

Key hardware optimizations encoded from Apple Silicon microbenchmarks:
- Chunked B=1 4D WY GatedDeltaNet (C=32) with hierarchical block inversion (18 ops/layer
  at depth 4 instead of 31 sequential slice-concat steps) plus optional fwd_subst mode
- Pure FP16 mul+add masking, tril matmul cumsum, and relu+log(1+exp(-|x|)) softplus
  (0 bool masks, 0 casts, 0 non-ANE ops on [CPU_AND_NE])
- IEEE binary16 normal epsilon floor (`eps >= 2^-14 = 6.103515625e-5`) across all
  RMSNorm / L2-norm ops so ANE FTZ never flushes `var + eps` to 0 -> `rsqrt(0) = Inf/NaN`
- Extended `1/64` residual prescaling across `l5..l11` and `final_norm` (preventing FP16
  `x*x` overflow on layers 5..8 where residuals can exceed `|x| > 2048`)
- t_delta FP16 FTZ fix: scale v post-silu(conv1d) by ftz_alpha (default 128.0) and
  scale RMSNormGated eps by ftz_alpha^2 * 1e-6 = 0.016384 (exact 1-homogeneity)
- High-outlier & sensitive GDN linear projection exclusion from global INT8 weight
  quantization (preserving FP16 weights on `OUTLIER_LINEAR_KEYS_DEFAULT` and recurrent
  GDN projections to prevent INT8 collapse)
- Multi-doc bucket isolation via tok_valid [1,1,1,S] before conv1d, gdn_carry [1,S/32,1,1]
  between chunks, external per-doc cos32/sin32, and block-diagonal FP16 attn_bias
- Stage splitting (configurable layers-per-stage, default 3 -> 4x3L) keeping each stage
  well below E5RT compile limits with async ring chaining in pplx_coreml_runner
- Support for FP16, selective W8A16 (outlier + GDN FP16 preserved), and W8A8 QDQ with
  mul(mul(gate, up), sigmoid(gate)) SwiGLU-QDQ lowering, plus macOS 27.0+ Core AI probe
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Sequence

import numpy as np

from fast_pplx.config import DEFAULT_PPLX_CONFIG, PplxConfig
from fast_pplx.coreai_export import (
    DEFAULT_BLOCK_SIZE,
    VALID_COREAI_PRESETS,
    run_pplx_coreai_export,
)
from fast_pplx.paths import resolve_paths
from fast_shared.coreai_pipeline import inspect_coreml_bundle_placement
from fast_shared.hw_check import probe_core_ai_support
from fast_shared.ir_dtype_guard import check_coreml_model, validate_export_quant_dtypes
from fast_shared.parallel_coreml import compile_buckets_parallel

HIDDEN = 1024
MLP_INTER = 3584
GDN_HEADS = 16
GDN_DIM = 128
ATTN_Q_HEADS = 8
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 256
ROPE_HALF = 32
RMS_EPS = 1e-6
FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)  # 2^-14 = 6.103515625e-5
MAX_ABS_FLOOR = 1e-4  # > 2^-14 ANE clamp floor; prevents NaN/zero-div on all-zero rows
VALID_SILU_MODES = ("silu", "tanh_gdn", "tanh_all")
VALID_RMSNORM_MODES = ("prescale", "max_abs")
MANIFEST_SCHEMA_VERSION = 2
ANE_PROFILES: dict[str, tuple[str, str]] = {
    "fast": ("silu", "prescale"),
    "balanced": ("tanh_gdn", "max_abs"),
    "parity": ("tanh_all", "max_abs"),
}
VALID_ANE_PROFILES = ("fast", "balanced", "parity")
DEFAULT_ANE_PROFILE = "balanced"
CHUNK_SIZE = 32
DEFAULT_FTZ_ALPHA = 128.0
DEFAULT_SEQ_LENS: tuple[int, ...] = (32, 64, 128, 256, 384, 512)
DEFAULT_LAYERS_PER_STAGE = 3

# Projections whose calibration absmax >= 50 stay in FP16 (excluded from naive global INT8 quantization).
OUTLIER_LINEAR_KEYS_DEFAULT: frozenset[str] = frozenset(
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

_HF_TO_MIL_SUFFIX: dict[str, str] = {
    "mlp.down_proj": "mlp_dn",
    "mlp.gate_proj": "mlp_gu",
    "mlp.up_proj": "mlp_gu",
    "linear_attn.in_proj_qkv": "qkvzba",
    "linear_attn.out_proj": "out_proj",
    "linear_attn.conv1d": "conv1d",
    "self_attn.q_proj": "qkvgate",
    "self_attn.o_proj": "o_proj",
}

_PRESCALE_64_PREFIXES: tuple[str, ...] = (
    "l5_",
    "l6_",
    "l7_",
    "l8_",
    "l9_",
    "l10_",
    "l11_",
)


def clamp_fp16_eps(eps: float) -> float:
    """Clamp FP16 epsilon to >= 2^-14 (`6.103515625e-5`) so ANE FTZ never flushes var+eps to 0."""
    return float(max(float(eps), FP16_MIN_NORMAL))


def resolve_ane_profile(
    ane_profile: str | None = None,
    silu_mode: str | None = None,
    rmsnorm_mode: str | None = None,
) -> tuple[str, str, str]:
    """Resolve `(--ane-profile, --silu-mode, --rmsnorm-mode)` -> `(canonical_profile, silu_mode, rmsnorm_mode)`.

    `ane_profile="fast"` selects `("silu", "prescale")` (default peak speed);
    `ane_profile="balanced"` selects `("tanh_gdn", "max_abs")` (~5% throughput cost, `>=0.999` cos);
    `ane_profile="parity"` selects `("tanh_all", "max_abs")` (max ANE/GPU parity, recommended for
    `--dtype int8` or `--dimensions 128`).
    Explicit `silu_mode` / `rmsnorm_mode` flags override the preset and record `"custom"`
    when the combination does not match a named preset.
    """
    if ane_profile is not None and ane_profile not in VALID_ANE_PROFILES:
        raise ValueError(f"Unsupported ane_profile={ane_profile!r}; choose from {VALID_ANE_PROFILES}")
    base_silu, base_rms = ANE_PROFILES[ane_profile or DEFAULT_ANE_PROFILE]
    eff_silu = base_silu if silu_mode is None else str(silu_mode)
    eff_rms = base_rms if rmsnorm_mode is None else str(rmsnorm_mode)
    if eff_silu not in VALID_SILU_MODES:
        raise ValueError(f"Unsupported silu_mode={eff_silu!r}; choose from {VALID_SILU_MODES}")
    if eff_rms not in VALID_RMSNORM_MODES:
        raise ValueError(f"Unsupported rmsnorm_mode={eff_rms!r}; choose from {VALID_RMSNORM_MODES}")

    canonical = "custom"
    for pname, pair in ANE_PROFILES.items():
        if (eff_silu, eff_rms) == pair:
            canonical = pname
            break
    return canonical, eff_silu, eff_rms


def compute_ane_config_hash(
    *,
    backend: str,
    precision: str,
    inv_mode: str,
    silu_mode: str,
    rmsnorm_mode: str,
    ftz_alpha: float,
    preserve_gdn_fp16: bool,
    layers_per_stage: int,
    outlier_threshold: float,
    num_layers: int,
    calib_stats: dict[str, Any] | None = None,
) -> str:
    """Return a deterministic 12-hex-char fingerprint of all build-affecting ANE graph knobs."""
    calib_digest: str | None = None
    if calib_stats is not None:
        calib_digest = hashlib.sha256(
            json.dumps(calib_stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "backend": str(backend),
        "precision": str(precision),
        "inv_mode": str(inv_mode),
        "silu_mode": str(silu_mode),
        "rmsnorm_mode": str(rmsnorm_mode),
        "ftz_alpha": float(ftz_alpha),
        "eps_floor_fp16": FP16_MIN_NORMAL,
        "preserve_gdn_fp16": bool(preserve_gdn_fp16),
        "layers_per_stage": int(layers_per_stage),
        "outlier_threshold": float(outlier_threshold),
        "chunk_size": CHUNK_SIZE,
        "hidden_size": HIDDEN,
        "num_layers": int(num_layers),
        "calib_stats_sha256": calib_digest,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def pplx_stage_function_name(seq_len: int) -> str:
    """Canonical CoreML multifunction entry name for a sequence-length bucket."""
    return f"pplx_s{int(seq_len)}"


def make_ane_stage_stem(
    seq_len: int,
    stage_idx: int,
    precision: str,
    *,
    ane_profile: str,
    silu_mode: str,
    rmsnorm_mode: str,
    config_hash: str,
) -> str:
    """Build a mode- and config-distinguishing `.mlmodelc` filename stem for a compiled stage."""
    mode_tag = ane_profile if ane_profile in VALID_ANE_PROFILES else f"{silu_mode}_{rmsnorm_mode}"
    return f"pplx_s{int(seq_len)}_st{int(stage_idx)}_{precision}_{mode_tag}_{config_hash[:8]}"


def make_ane_multifunction_stage_stem(
    stage_idx: int,
    precision: str,
    *,
    ane_profile: str,
    silu_mode: str,
    rmsnorm_mode: str,
    config_hash: str,
) -> str:
    """Build a multifunction `.mlmodelc` filename stem sharing stage weights across all `seq_lens`."""
    mode_tag = ane_profile if ane_profile in VALID_ANE_PROFILES else f"{silu_mode}_{rmsnorm_mode}"
    return f"pplx_mf_st{int(stage_idx)}_{precision}_{mode_tag}_{config_hash[:8]}"


def _load_preserved_buckets_for_hash(
    out_dir: Path,
    *,
    config_hash: str,
    requested_seq_lens: Sequence[int],
    expected_num_stages: int,
) -> dict[str, dict[str, Any]]:
    """Preserve untouched buckets ONLY when existing `manifest.json` matches `config_hash` exactly."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(existing, dict):
        return {}
    if int(existing.get("schema_version", 1)) != MANIFEST_SCHEMA_VERSION:
        return {}
    if existing.get("config_hash") != config_hash:
        return {}
    if int(existing.get("num_stages", -1)) != int(expected_num_stages):
        return {}

    req_set = {int(s) for s in requested_seq_lens}
    preserved: dict[str, dict[str, Any]] = {}
    for k_str, entry in existing.get("buckets", {}).items():
        if not isinstance(entry, dict):
            continue
        cap = int(entry.get("seq_len", k_str))
        if cap in req_set:
            continue
        if entry.get("config_hash") != config_hash:
            continue
        stages = entry.get("stages", [])
        if len(stages) != int(expected_num_stages):
            continue
        if stages and all((out_dir / name).exists() for name in stages):
            preserved[str(cap)] = dict(entry)
    return preserved


def _prune_stale_ane_bundles(
    out_dir: Path,
    *,
    active_stage_names: set[str],
    keep_mlpackage: bool = False,
) -> list[str]:
    """Remove stale `pplx_s*_st*.mlmodelc` and `.mlpackage` dirs not referenced by the active manifest."""
    active_stems = {Path(name).stem for name in active_stage_names}
    removed: list[str] = []
    for pat in (
        "pplx_s*_st*.mlmodelc",
        "pplx_s*_st*.mlpackage",
        "pplx_s*_st*.aimodel",
        "pplx_mf_st*.mlmodelc",
        "pplx_mf_st*.mlpackage",
    ):
        for cand in sorted(out_dir.glob(pat)):
            if cand.name in active_stage_names:
                continue
            if keep_mlpackage and cand.suffix == ".mlpackage" and cand.stem in active_stems:
                continue
            if cand.is_dir():
                shutil.rmtree(cand, ignore_errors=True)
            else:
                cand.unlink(missing_ok=True)
            removed.append(cand.name)
    return removed


def resolve_rmsnorm_prescale(
    name: str,
    calib_stats: dict[str, Any] | None = None,
) -> float:
    """Return power-of-2 prescale factor avoiding FP16 `x*x > 65504` overflow across deep layers."""
    if calib_stats is not None:
        entries = calib_stats.get("norms", calib_stats.get("layers", calib_stats))
        info = entries.get(name) if isinstance(entries, dict) else None
        if isinstance(info, dict) and float(info.get("absmax", 0.0)) > 0.0:
            amax = float(info["absmax"])
            if amax > 2048.0:
                return 1.0 / 64.0
            if amax > 256.0:
                return 1.0 / 16.0
            return 1.0

    if name == "final_norm" or name.startswith(_PRESCALE_64_PREFIXES):
        return 1.0 / 64.0
    return 1.0


def _hf_key_to_mil_op(hf_key: str) -> str | None:
    if not hf_key.startswith("layers."):
        return None
    parts = hf_key.split(".", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    li = int(parts[1])
    mil_sfx = _HF_TO_MIL_SUFFIX.get(parts[2])
    if mil_sfx is None:
        return None
    return f"l{li}_{mil_sfx}"


def select_fp16_preserved_ops(
    layer_start: int,
    layer_end: int,
    layer_types: Sequence[str],
    *,
    calib_stats: dict[str, Any] | None = None,
    outlier_threshold: float = 15.0,
    preserve_gdn_fp16: bool = True,
) -> frozenset[str]:
    """Return MIL `conv` op names excluded from INT8 weight quantization to prevent parity collapse.

    Preserves:
      1. All `OUTLIER_LINEAR_KEYS_DEFAULT` projections (`l5/7/8/9/10/11_mlp_dn`, `l8/10_out_proj`).
      2. When `preserve_gdn_fp16=True`, all recurrent GDN projections (`l{i}_qkvzba`, `l{i}_conv1d`,
         `l{i}_out_proj`) whose quantization noise compounds across DeltaNet recurrence steps.
      3. Any projection whose calibrated `absmax >= outlier_threshold`.
    """
    preserved: set[str] = set()
    layer_range = range(int(layer_start), int(layer_end))

    for hf_key in OUTLIER_LINEAR_KEYS_DEFAULT:
        mil_op = _hf_key_to_mil_op(hf_key)
        if mil_op is not None:
            li = int(hf_key.split(".")[1])
            if li in layer_range:
                preserved.add(mil_op)

    for li in layer_range:
        lt = layer_types[li]
        if preserve_gdn_fp16 and lt == "linear_attention":
            preserved.add(f"l{li}_qkvzba")
            preserved.add(f"l{li}_conv1d")
            preserved.add(f"l{li}_out_proj")

    if calib_stats is not None:
        entries = calib_stats.get("layers", calib_stats)
        if isinstance(entries, dict):
            for hf_key, info in entries.items():
                if not isinstance(info, dict):
                    continue
                if float(info.get("absmax", 0.0)) >= float(outlier_threshold):
                    mil_op = _hf_key_to_mil_op(hf_key)
                    if mil_op is not None:
                        li = int(hf_key.split(".")[1])
                        if li in layer_range:
                            preserved.add(mil_op)

    return frozenset(preserved)


def build_weight_quant_config(
    preserved_ops: Sequence[str] | frozenset[str],
    *,
    weight_threshold: int = 512,
    cto_module: Any | None = None,
) -> Any:
    """Build CoreML `OptimizationConfig` quantizing non-excluded conv weights to per-channel INT8."""
    if cto_module is None:
        import coremltools.optimize.coreml as cto_module
    q_op_cfg = cto_module.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
        weight_threshold=int(weight_threshold),
    )
    op_name_configs = {str(op_name): None for op_name in sorted(set(preserved_ops))}
    return cto_module.OptimizationConfig(
        op_type_configs={"conv": q_op_cfg},
        op_name_configs=op_name_configs,
    )


def check_coreai_backend_status(
    *,
    macos_version: str | None = None,
    use_os_coreai: bool | None = None,
    pkg_versions: dict[str, str | None] | None = None,
    delegate_probe: Any = None,
) -> dict[str, Any]:
    """Report decoupled offline Core AI export readiness (`export_ready`) vs OS 27.0+ runtime (`runtime_ready`)."""
    hw_override = (
        {"system": "Darwin", "machine": "arm64", "macos_version": str(macos_version)}
        if macos_version is not None
        else None
    )
    probe = probe_core_ai_support(
        hw=hw_override,
        use_os_coreai=use_os_coreai,
        pkg_versions=pkg_versions,
        delegate_probe=delegate_probe,
    )
    ver_str = (
        str(macos_version)
        if macos_version is not None
        else (platform.mac_ver()[0] or ("27.0" if probe["os_runtime_eligible"] else ""))
    )
    flag = bool(probe["use_os_coreai"])
    has_coreai_torch = probe["coreai_torch_version"] is not None
    export_ready = bool(probe["export_ready"])
    runtime_ready = bool(probe["runtime_ready"])

    if not probe["os_runtime_eligible"]:
        runtime_reason = f"requires macOS 27.0+ (detected {ver_str or 'unknown'})"
    elif not flag:
        runtime_reason = "USE_OS_COREAI=1 not enabled"
    elif probe["coreai_runtime_version"] is None:
        runtime_reason = "coreai runtime package not installed"
    elif probe["delegate_supported"] is False:
        runtime_reason = "Core AI runtime delegate specialization unsupported"
    else:
        runtime_reason = "ready"

    export_reason = (
        "ready"
        if export_ready
        else (
            "coreai-torch package not installed"
            if not has_coreai_torch
            else f"requires coreai-torch >= 0.4.1 (detected {probe['coreai_torch_version']})"
        )
    )
    return {
        "eligible": runtime_ready,
        "runtime_ready": runtime_ready,
        "export_ready": export_ready,
        "macos_version": ver_str,
        "use_os_coreai": flag,
        "has_coreai_torch": has_coreai_torch,
        "coreai_torch_version": probe["coreai_torch_version"],
        "coreai_opt_version": probe["coreai_opt_version"],
        "coreai_core_version": probe["coreai_core_version"],
        "coreai_runtime_version": probe["coreai_runtime_version"],
        "coreai_torch_ge_041": probe["coreai_torch_ge_041"],
        "os_runtime_eligible": probe["os_runtime_eligible"],
        "fp4_fp8_preset_eligible": probe["fp4_fp8_preset_eligible"],
        "delegate_supported": probe["delegate_supported"],
        "export_reason": export_reason,
        "reason": runtime_reason,
    }


def plan_stage_ranges(num_layers: int = 12, layers_per_stage: int = DEFAULT_LAYERS_PER_STAGE) -> list[tuple[int, int]]:
    """Split layer indices [0, num_layers) into contiguous [start, end) stage ranges."""
    lps = max(1, int(layers_per_stage))
    return [(s, min(num_layers, s + lps)) for s in range(0, num_layers, lps)]


def compute_gnorm_eps_scaled(ftz_alpha: float, rms_eps: float = RMS_EPS) -> float:
    """Exact RMSNormGated epsilon matching pre-scaled v -> ftz_alpha * v, clamped >= FP16_MIN_NORMAL."""
    return clamp_fp16_eps(float(rms_eps) * (float(ftz_alpha) ** 2))


def select_qdq_scales(
    layer_idx: int,
    layer_type: str,
    calib_stats: dict[str, Any] | None,
    *,
    outlier_threshold: float = 15.0,
) -> dict[str, float | None]:
    """Compute per-projection QDQ absmax bounds, returning None to preserve FP16/W8A16 for outliers."""
    if calib_stats is None:
        return {}
    entries = calib_stats.get("layers", calib_stats)
    scales: dict[str, float | None] = {}

    def _pick(hf_key: str, mil_key: str) -> None:
        if hf_key in OUTLIER_LINEAR_KEYS_DEFAULT and hf_key not in entries:
            scales[mil_key] = None
            return
        info = entries.get(hf_key)
        if not info:
            scales[mil_key] = None
            return
        amax = float(info.get("absmax", 0.0))
        if amax <= 0.0 or amax >= outlier_threshold:
            scales[mil_key] = None
        else:
            scales[mil_key] = amax

    if layer_type == "linear_attention":
        in_info = entries.get(f"layers.{layer_idx}.linear_attn.in_proj_qkv")
        if in_info and 0.0 < float(in_info.get("absmax", 0.0)) < outlier_threshold:
            scales[f"l{layer_idx}_qkvzba"] = float(in_info["absmax"])
        else:
            scales[f"l{layer_idx}_qkvzba"] = None
        _pick(f"layers.{layer_idx}.linear_attn.out_proj", f"l{layer_idx}_out_proj")
    else:
        _pick(f"layers.{layer_idx}.self_attn.q_proj", f"l{layer_idx}_qkvgate")
        _pick(f"layers.{layer_idx}.self_attn.o_proj", f"l{layer_idx}_o_proj")

    _pick(f"layers.{layer_idx}.mlp.gate_proj", f"l{layer_idx}_mlp_gu")
    _pick(f"layers.{layer_idx}.mlp.down_proj", f"l{layer_idx}_mlp_dn")
    return scales


def convert_weights_to_ane_layout(
    npz_weights: dict[str, np.ndarray],
    cfg: PplxConfig = DEFAULT_PPLX_CONFIG,
) -> dict[str, np.ndarray]:
    """Convert Track-2 canonical NPZ weights into ANE 4D conv/norm FP16 arrays."""
    if "l0_in_ln" in npz_weights:
        return dict(npz_weights)

    norms_shifted = bool(int(np.asarray(npz_weights.get("__norms_shifted__", 0)).item()))
    out: dict[str, np.ndarray] = {}

    def _norm(arr: np.ndarray, *, already_shifted: bool) -> np.ndarray:
        f32 = np.asarray(arr, dtype=np.float32)
        return (f32 if already_shifted else (1.0 + f32)).astype(np.float16)

    for i, layer_type in enumerate(cfg.layer_types):
        mp = f"layers.{i}."
        out[f"l{i}_in_ln"] = _norm(npz_weights[f"{mp}input_layernorm.weight"], already_shifted=norms_shifted)[
            None, :, None, None
        ]
        out[f"l{i}_post_ln"] = _norm(
            npz_weights[f"{mp}post_attention_layernorm.weight"], already_shifted=norms_shifted
        )[None, :, None, None]

        if layer_type == "linear_attention":
            in_all_key = f"{mp}linear_attn.in_proj_all.weight"
            if in_all_key in npz_weights:
                out[f"l{i}_qkvzba"] = np.asarray(npz_weights[in_all_key], dtype=np.float16)[:, :, None, None]
            else:
                w_qkv = np.asarray(npz_weights[f"{mp}linear_attn.in_proj_qkv.weight"], dtype=np.float16)
                w_z = np.asarray(npz_weights[f"{mp}linear_attn.in_proj_z.weight"], dtype=np.float16)
                w_b = np.asarray(npz_weights[f"{mp}linear_attn.in_proj_b.weight"], dtype=np.float16)
                w_a = np.asarray(npz_weights[f"{mp}linear_attn.in_proj_a.weight"], dtype=np.float16)
                out[f"l{i}_qkvzba"] = np.concatenate([w_qkv, w_z, w_b, w_a], axis=0)[:, :, None, None]

            conv_raw = np.asarray(npz_weights[f"{mp}linear_attn.conv1d.weight"], dtype=np.float16)
            if conv_raw.ndim == 3 and conv_raw.shape[1] == 4 and conv_raw.shape[2] == 1:
                conv_4d = np.ascontiguousarray(np.swapaxes(conv_raw, 1, 2)[:, :, None, :])
            elif conv_raw.ndim == 3 and conv_raw.shape[1] == 1 and conv_raw.shape[2] == 4:
                conv_4d = np.ascontiguousarray(conv_raw[:, :, None, :])
            elif conv_raw.ndim == 4:
                conv_4d = np.ascontiguousarray(conv_raw)
            else:
                raise ValueError(f"Unexpected conv1d shape {conv_raw.shape} at layer {i}")
            out[f"l{i}_conv1d"] = conv_4d

            a_log = np.asarray(npz_weights[f"{mp}linear_attn.A_log"], dtype=np.float32)
            out[f"l{i}_neg_A"] = (-np.exp(a_log)).astype(np.float16)
            out[f"l{i}_dt_bias"] = np.asarray(npz_weights[f"{mp}linear_attn.dt_bias"], dtype=np.float16)
            out[f"l{i}_gnorm_w"] = np.asarray(npz_weights[f"{mp}linear_attn.norm.weight"], dtype=np.float16)
            out[f"l{i}_out_proj"] = np.asarray(npz_weights[f"{mp}linear_attn.out_proj.weight"], dtype=np.float16)[
                :, :, None, None
            ]
        elif layer_type == "full_attention":
            qkv_key = f"{mp}self_attn.qkv_proj.weight"
            q_dim_inter = cfg.num_attention_heads * 2 * cfg.head_dim
            kv_dim = cfg.num_key_value_heads * cfg.head_dim
            if qkv_key in npz_weights:
                qkv = np.asarray(npz_weights[qkv_key], dtype=np.float16)
                w_q_raw = qkv[:q_dim_inter]
                w_k = qkv[q_dim_inter : q_dim_inter + kv_dim]
                w_v = qkv[q_dim_inter + kv_dim :]
            else:
                w_q_raw = np.asarray(npz_weights[f"{mp}self_attn.q_proj.weight"], dtype=np.float16)
                w_k = np.asarray(npz_weights[f"{mp}self_attn.k_proj.weight"], dtype=np.float16)
                w_v = np.asarray(npz_weights[f"{mp}self_attn.v_proj.weight"], dtype=np.float16)

            wqg = w_q_raw.reshape(cfg.num_attention_heads, 2, cfg.head_dim, cfg.hidden_size)
            w_q = wqg[:, 0, :, :].reshape(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
            w_gate = wqg[:, 1, :, :].reshape(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
            out[f"l{i}_qkvgate"] = np.concatenate([w_q, w_k, w_v, w_gate], axis=0)[:, :, None, None]
            out[f"l{i}_qn"] = _norm(npz_weights[f"{mp}self_attn.q_norm.weight"], already_shifted=norms_shifted)[
                None, None, None, :
            ]
            out[f"l{i}_kn"] = _norm(npz_weights[f"{mp}self_attn.k_norm.weight"], already_shifted=norms_shifted)[
                None, None, None, :
            ]
            out[f"l{i}_o_proj"] = np.asarray(npz_weights[f"{mp}self_attn.o_proj.weight"], dtype=np.float16)[
                :, :, None, None
            ]
        else:
            raise ValueError(f"Unsupported layer_type {layer_type!r} at layer {i}")

        gu_key = f"{mp}mlp.gate_up_proj.weight"
        if gu_key in npz_weights:
            w_gu = np.asarray(npz_weights[gu_key], dtype=np.float16)
        else:
            w_g = np.asarray(npz_weights[f"{mp}mlp.gate_proj.weight"], dtype=np.float16)
            w_u = np.asarray(npz_weights[f"{mp}mlp.up_proj.weight"], dtype=np.float16)
            w_gu = np.concatenate([w_g, w_u], axis=0)
        out[f"l{i}_mlp_gu"] = w_gu[:, :, None, None]
        out[f"l{i}_mlp_dn"] = np.asarray(npz_weights[f"{mp}mlp.down_proj.weight"], dtype=np.float16)[:, :, None, None]

    out["final_norm"] = _norm(npz_weights["norm.weight"], already_shifted=norms_shifted)[None, :, None, None]
    return out


def build_stage_mil_program(
    ane_weights: dict[str, np.ndarray],
    *,
    seq_len: int,
    layer_start: int,
    layer_end: int,
    layer_types: tuple[str, ...],
    apply_final_norm: bool,
    chunk_size: int = CHUNK_SIZE,
    inv_mode: str = "block",
    ftz_alpha: float = DEFAULT_FTZ_ALPHA,
    qdq_scales: dict[str, float | None] | None = None,
    calib_stats: dict[str, Any] | None = None,
    swiglu_mode: str = "silu",
    silu_mode: str = "silu",
    rmsnorm_mode: str = "prescale",
):
    """Build a pure-FP16 pure-ANE CoreML MIL program for layers [layer_start, layer_end)."""
    import coremltools as ct  # noqa: F401
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if silu_mode not in VALID_SILU_MODES:
        raise ValueError(f"Unsupported silu_mode={silu_mode!r}; choose from {VALID_SILU_MODES}")
    if rmsnorm_mode not in VALID_RMSNORM_MODES:
        raise ValueError(f"Unsupported rmsnorm_mode={rmsnorm_mode!r}; choose from {VALID_RMSNORM_MODES}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")

    scales = qdq_scales or {}
    n_c = seq_len // chunk_size
    gnorm_eps = compute_gnorm_eps_scaled(ftz_alpha)

    def _f16(val: float):
        return mb.const(val=np.float16(val))

    def _const_f16(arr: np.ndarray, name: str):
        return mb.const(val=np.ascontiguousarray(arr, dtype=np.float16), name=name)

    def _conv_or_qdq(x, w_f16: np.ndarray, name: str):
        w_c = _const_f16(w_f16, name=f"{name}_w")
        amax = scales.get(name)
        if amax is not None and amax > 0.0:
            s_val = np.float16(clamp_fp16_eps(float(amax) / 127.0))
            s_c = mb.const(val=s_val, name=f"{name}_q_scale")
            xq = mb.quantize(input=x, scale=s_c, output_dtype="int8", name=f"{name}_aq")
            xdq = mb.dequantize(input=xq, scale=s_c, name=f"{name}_adq")
            return mb.conv(x=xdq, weight=w_c, name=name)
        return mb.conv(x=x, weight=w_c, name=name)

    def _tanh_silu(x, name: str):
        """High-precision ANE SiLU via tanh: x * (0.5 * tanh(0.5 * x) + 0.5)."""
        half = _f16(0.5)
        cx = mb.mul(x=x, y=half, name=f"{name}_half")
        th = mb.tanh(x=cx, name=f"{name}_tanh")
        sig = mb.add(x=mb.mul(x=th, y=half, name=f"{name}_htanh"), y=half, name=f"{name}_sig")
        return mb.mul(x=x, y=sig, name=name)

    def _silu_op(x, name: str, *, is_mlp: bool):
        use_tanh = (silu_mode == "tanh_all") or (silu_mode == "tanh_gdn" and not is_mlp)
        return _tanh_silu(x, name=name) if use_tanh else mb.silu(x=x, name=name)

    def _rmsnorm_ch(x, weight_1c11: np.ndarray, name: str, eps: float = RMS_EPS):
        if rmsnorm_mode == "max_abs":
            # Dynamic per-token max-abs scaling prevents FP16 x*x overflow (>255.94)
            # without compiler prescale folding or 2^-14 eps inflation on small rows.
            amax = mb.reduce_max(x=mb.abs(x=x, name=f"{name}_abs"), axes=[1], keep_dims=True, name=f"{name}_amax")
            m = mb.maximum(x=amax, y=_f16(MAX_ABS_FLOOR), name=f"{name}_mfloor")
            xn = mb.real_div(x=x, y=m, name=f"{name}_divm")
            e_ratio = mb.real_div(x=_f16(float(eps) ** 0.5), y=m, name=f"{name}_eratio")
            eps_term = mb.mul(x=e_ratio, y=e_ratio, name=f"{name}_eterm")
            msq = mb.reduce_mean(x=mb.mul(x=xn, y=xn, name=f"{name}_sq"), axes=[1], keep_dims=True, name=f"{name}_mean")
            inv_rms = mb.rsqrt(x=mb.add(x=msq, y=eps_term, name=f"{name}_eps"), name=f"{name}_rsqrt")
            normed = mb.mul(x=xn, y=inv_rms, name=f"{name}_norm")
            return mb.mul(x=normed, y=_const_f16(weight_1c11, name=f"{name}_w"), name=name)

        # Deep residual streams reach high absmax in layers 5..11, risking x*x FP16 overflow
        # (> sqrt(65504) = 255.94). Prescaling by 1/64 cancels out while eps is clamped to >= 2^-14
        # so ANE FTZ never flushes var + eps to zero.
        inv_scale = resolve_rmsnorm_prescale(name, calib_stats=calib_stats)
        if inv_scale != 1.0:
            x_in = mb.mul(x=x, y=_f16(inv_scale), name=f"{name}_prescale")
            eps_val = clamp_fp16_eps(eps * inv_scale * inv_scale)
        else:
            x_in = x
            eps_val = clamp_fp16_eps(eps)

        sq = mb.mul(x=x_in, y=x_in, name=f"{name}_sq")
        msq = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{name}_mean")
        inv_rms = mb.rsqrt(x=mb.add(x=msq, y=_f16(eps_val), name=f"{name}_eps"), name=f"{name}_rsqrt")
        normed = mb.mul(x=x_in, y=inv_rms, name=f"{name}_norm")
        return mb.mul(x=normed, y=_const_f16(weight_1c11, name=f"{name}_w"), name=name)

    def _rmsnorm_axis(x, weight_arr: np.ndarray, axis: int, name: str, eps: float = RMS_EPS):
        # qnorm/knorm/gnorm inputs stay well below sqrt(65504) (observed absmax <= 62.9,
        # q/k rms_min >= 0.387, gnorm_eps=0.016384 >> 2^-14). MaxAbsRMSNorm on axis=2 (gnorm)
        # triggers ANECCompile() failure (100% CPU fallback) and axis=3 (q/k) adds +1.0% latency
        # for <=3e-6 cosine delta, so _rmsnorm_axis intentionally stays on clamp_fp16_eps.
        eps_val = clamp_fp16_eps(eps)
        sq = mb.mul(x=x, y=x, name=f"{name}_sq")
        msq = mb.reduce_mean(x=sq, axes=[axis], keep_dims=True, name=f"{name}_mean")
        inv_rms = mb.rsqrt(x=mb.add(x=msq, y=_f16(eps_val), name=f"{name}_eps"), name=f"{name}_rsqrt")
        normed = mb.mul(x=x, y=inv_rms, name=f"{name}_norm")
        return mb.mul(x=normed, y=_const_f16(weight_arr, name=f"{name}_w"), name=name)

    def _l2norm_last(x, dim: int, extra_scale: float, name: str, eps: float = RMS_EPS):
        eps_val = clamp_fp16_eps(eps)
        sq = mb.mul(x=x, y=x, name=f"{name}_sq")
        msq = mb.reduce_mean(x=sq, axes=[3], keep_dims=True, name=f"{name}_mean")
        inv_rms = mb.rsqrt(x=mb.add(x=msq, y=_f16(eps_val), name=f"{name}_eps"), name=f"{name}_rsqrt")
        inv_norm_scaled = mb.mul(x=inv_rms, y=_f16((dim ** -0.5) * extra_scale), name=f"{name}_scale")
        return mb.mul(x=x, y=inv_norm_scaled, name=name)

    def _softplus_ane(x, name: str):
        r = mb.relu(x=x, name=f"{name}_relu")
        neg_abs = mb.mul(x=mb.abs(x=x, name=f"{name}_abs"), y=_f16(-1.0), name=f"{name}_nabs")
        exp_n = mb.exp(x=neg_abs, name=f"{name}_exp")
        log1p = mb.log(x=mb.add(x=exp_n, y=_f16(1.0), name=f"{name}_add1"), name=f"{name}_log")
        return mb.add(x=r, y=log1p, name=name)

    def _cumsum_tril(g_4d, prefix: str):
        triu_ones = np.triu(np.ones((chunk_size, chunk_size), dtype=np.float16), k=0)[None, None, :, :]
        return mb.matmul(x=g_4d, y=_const_f16(triu_ones, name=f"{prefix}_triu"), name=f"{prefix}_mm")

    def _invert_unit_lower_block(m_strict, prefix: str):
        """Exact hierarchical block inversion (2->4->8->16->32) solving (I - m_strict)^-1 in 18 ops."""
        eye_np = np.eye(chunk_size, dtype=np.float16)[None, None, :, :]
        m2_np = np.zeros((1, 1, chunk_size, chunk_size), dtype=np.float16)
        for k in range(0, chunk_size, 2):
            if k + 1 < chunk_size:
                m2_np[0, 0, k + 1, k] = np.float16(1.0)
        t_cur = mb.add(
            x=_const_f16(eye_np, name=f"{prefix}_eye"),
            y=mb.mul(x=m_strict, y=_const_f16(m2_np, name=f"{prefix}_m2"), name=f"{prefix}_t2"),
            name=f"{prefix}_init2",
        )
        b = 2
        while b < chunk_size:
            mask_np = np.zeros((1, 1, chunk_size, chunk_size), dtype=np.float16)
            for k in range(0, chunk_size, 2 * b):
                r1 = min(chunk_size, k + 2 * b)
                if k + b < r1:
                    mask_np[0, 0, k + b : r1, k : k + b] = np.float16(1.0)
            off = mb.mul(x=m_strict, y=_const_f16(mask_np, name=f"{prefix}_mb{b}"), name=f"{prefix}_off{b}")
            t_off = mb.matmul(x=t_cur, y=off, name=f"{prefix}_toff{b}")
            t21 = mb.matmul(x=t_off, y=t_cur, name=f"{prefix}_t21_{b}")
            t_cur = mb.add(x=t_cur, y=t21, name=f"{prefix}_merge{b}")
            b *= 2
        return t_cur

    def _invert_unit_lower_fwd_subst(m_strict, d0: int, d1: int, prefix: str):
        eye = np.eye(chunk_size, dtype=np.float16)
        row0 = np.broadcast_to(eye[0:1, :][None, None, :, :], (d0, d1, 1, chunk_size)).copy()
        rows = [_const_f16(row0, name=f"{prefix}_row0")]
        for i in range(1, chunk_size):
            prev = mb.concat(values=rows, axis=2, name=f"{prefix}_prev_{i}")
            m_row = mb.slice_by_index(
                x=m_strict, begin=[0, 0, i, 0], end=[d0, d1, i + 1, i], name=f"{prefix}_mrow_{i}"
            )
            if i == 1:
                t_prefix = m_row
            else:
                prev_prefix = mb.slice_by_index(
                    x=prev, begin=[0, 0, 0, 0], end=[d0, d1, i, i], name=f"{prefix}_pprev_{i}"
                )
                t_prefix = mb.matmul(x=m_row, y=prev_prefix, name=f"{prefix}_mm_{i}")
            tail_arr = np.broadcast_to(
                eye[i : i + 1, i:][None, None, :, :], (d0, d1, 1, chunk_size - i)
            ).copy()
            row_i = mb.concat(
                values=[t_prefix, _const_f16(tail_arr, name=f"{prefix}_tail_{i}")],
                axis=3,
                name=f"{prefix}_row_{i}",
            )
            rows.append(row_i)
        return mb.concat(values=rows, axis=2, name=f"{prefix}_out")

    def _gated_delta_chunked_b1(q_ch, k_ch, v_ch, b_ch, a_ch, neg_A: np.ndarray, dt_bias: np.ndarray, gdn_carry, prefix: str):
        nc_h = n_c * GDN_HEADS

        def _to_b1_chunks(x_ch, tag: str):
            r1 = mb.reshape(x=x_ch, shape=[1, GDN_HEADS * GDN_DIM, n_c, chunk_size], name=f"{prefix}_{tag}r1")
            t1 = mb.transpose(x=r1, perm=[0, 2, 1, 3], name=f"{prefix}_{tag}t1")
            r2 = mb.reshape(x=t1, shape=[1, nc_h, GDN_DIM, chunk_size], name=f"{prefix}_{tag}r2")
            return mb.transpose(x=r2, perm=[0, 1, 3, 2], name=f"{prefix}_{tag}t2")

        q_4d = _to_b1_chunks(q_ch, "q")
        k_4d = _to_b1_chunks(k_ch, "k")
        v_4d = _to_b1_chunks(v_ch, "v")

        # Keep q_norm unit-L2 inside the chunked WY recurrence (+3.5 bits FP16 mantissa headroom)
        # and apply the exact 1-homogeneous 1/sqrt(GDN_DIM) scaling post-chunk.
        q_norm = _l2norm_last(q_4d, dim=GDN_DIM, extra_scale=1.0, name=f"{prefix}_ql2")
        k_norm = _l2norm_last(k_4d, dim=GDN_DIM, extra_scale=1.0, name=f"{prefix}_kl2")

        b_sig = mb.sigmoid(x=b_ch, name=f"{prefix}_beta_sig")
        b_t = mb.transpose(
            x=mb.reshape(x=b_sig, shape=[1, GDN_HEADS, n_c, chunk_size], name=f"{prefix}_br1"),
            perm=[0, 2, 1, 3],
            name=f"{prefix}_bt1",
        )
        beta = mb.reshape(x=b_t, shape=[1, nc_h, chunk_size, 1], name=f"{prefix}_beta_4d")

        neg_a_c = _const_f16(neg_A.reshape(1, GDN_HEADS, 1, 1), name=f"{prefix}_negA")
        dt_b_c = _const_f16(dt_bias.reshape(1, GDN_HEADS, 1, 1), name=f"{prefix}_dt_bias")
        g_raw = mb.mul(
            x=neg_a_c,
            y=_softplus_ane(mb.add(x=a_ch, y=dt_b_c, name=f"{prefix}_adt"), name=f"{prefix}_softplus"),
            name=f"{prefix}_g_raw",
        )
        g_t = mb.transpose(
            x=mb.reshape(x=g_raw, shape=[1, GDN_HEADS, n_c, chunk_size], name=f"{prefix}_gr1"),
            perm=[0, 2, 1, 3],
            name=f"{prefix}_gt1",
        )
        g_4d = mb.reshape(x=g_t, shape=[1, nc_h, 1, chunk_size], name=f"{prefix}_g_4d")

        cum_decay = _cumsum_tril(g_4d, prefix=f"{prefix}_cumsum")
        cum_decay_col = mb.transpose(x=cum_decay, perm=[0, 1, 3, 2], name=f"{prefix}_cum_col")
        pairwise = mb.sub(x=cum_decay_col, y=cum_decay, name=f"{prefix}_pairwise")

        tril_np = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float16), k=0)[None, None, :, :]
        neg100_upper_np = (-100.0 * np.triu(np.ones((chunk_size, chunk_size), dtype=np.float16), k=1))[
            None, None, :, :
        ]
        tril_f16 = _const_f16(tril_np, name=f"{prefix}_tril_f16")
        neg100_upper_f16 = _const_f16(neg100_upper_np, name=f"{prefix}_neg100_upper")
        neg_strict_f16 = _const_f16(
            (-np.tril(np.ones((chunk_size, chunk_size), dtype=np.float16), k=-1))[None, None, :, :],
            name=f"{prefix}_neg_strict",
        )

        pw_masked = mb.add(
            x=mb.mul(x=pairwise, y=tril_f16, name=f"{prefix}_pw_tril"),
            y=neg100_upper_f16,
            name=f"{prefix}_pw_mask",
        )
        pw_exp = mb.mul(
            x=mb.exp(x=pw_masked, name=f"{prefix}_pw_exp"),
            y=tril_f16,
            name=f"{prefix}_pw_exp_zero",
        )

        v_beta = mb.mul(x=v_4d, y=beta, name=f"{prefix}_v_beta")
        k_beta = mb.mul(x=k_norm, y=beta, name=f"{prefix}_k_beta")

        ut = mb.mul(
            x=mb.matmul(x=k_beta, y=k_norm, transpose_x=False, transpose_y=True, name=f"{prefix}_kk"),
            y=pw_exp,
            name=f"{prefix}_ut",
        )
        intra = mb.mul(
            x=mb.matmul(x=q_norm, y=k_norm, transpose_x=False, transpose_y=True, name=f"{prefix}_qk"),
            y=pw_exp,
            name=f"{prefix}_intra",
        )

        m_strict = mb.mul(x=ut, y=neg_strict_f16, name=f"{prefix}_m_strict")
        if inv_mode == "block":
            t_mat = _invert_unit_lower_block(m_strict, prefix=f"{prefix}_binv")
        elif inv_mode == "fwd_subst":
            t_mat = _invert_unit_lower_fwd_subst(m_strict, d0=1, d1=nc_h, prefix=f"{prefix}_inv")
        else:
            raise ValueError(f"Unsupported inv_mode={inv_mode!r}")

        exp_cum = mb.exp(x=cum_decay_col, name=f"{prefix}_exp_cum")
        decayed_k_beta = mb.mul(x=k_beta, y=exp_cum, name=f"{prefix}_dec_kb")

        new_values = mb.matmul(x=t_mat, y=v_beta, name=f"{prefix}_new_values")
        k_cumdecay = mb.matmul(x=t_mat, y=decayed_k_beta, name=f"{prefix}_k_cumdecay")

        q_dec = mb.mul(x=q_norm, y=exp_cum, name=f"{prefix}_q_dec")
        cum_last = mb.slice_by_index(
            x=cum_decay_col,
            begin=[0, 0, chunk_size - 1, 0],
            end=[1, nc_h, chunk_size, 1],
            name=f"{prefix}_cum_last",
        )
        k_dec = mb.mul(
            x=k_norm,
            y=mb.exp(
                x=mb.sub(x=cum_last, y=cum_decay_col, name=f"{prefix}_k_dec_diff"),
                name=f"{prefix}_k_dec_exp",
            ),
            name=f"{prefix}_k_dec",
        )
        chunk_dec = mb.exp(x=cum_last, name=f"{prefix}_chunk_dec")

        outs = []
        state = None
        for i in range(n_c):
            h0, h1 = i * GDN_HEADS, (i + 1) * GDN_HEADS
            nv_i = mb.slice_by_index(
                x=new_values, begin=[0, h0, 0, 0], end=[1, h1, chunk_size, GDN_DIM], name=f"{prefix}_nv_{i}"
            )
            intra_i = mb.slice_by_index(
                x=intra, begin=[0, h0, 0, 0], end=[1, h1, chunk_size, chunk_size], name=f"{prefix}_in_{i}"
            )
            q_dec_i = mb.slice_by_index(
                x=q_dec, begin=[0, h0, 0, 0], end=[1, h1, chunk_size, GDN_DIM], name=f"{prefix}_qd_{i}"
            )
            k_dec_i = mb.slice_by_index(
                x=k_dec, begin=[0, h0, 0, 0], end=[1, h1, chunk_size, GDN_DIM], name=f"{prefix}_kd_{i}"
            )
            c_dec_i = mb.slice_by_index(
                x=chunk_dec, begin=[0, h0, 0, 0], end=[1, h1, 1, 1], name=f"{prefix}_cd_{i}"
            )

            if i == 0:
                v_new = nv_i
                out_i = mb.matmul(x=intra_i, y=v_new, name=f"{prefix}_rec_out_{i}")
                state = mb.matmul(
                    x=k_dec_i, y=v_new, transpose_x=True, transpose_y=False, name=f"{prefix}_rec_st_{i}"
                )
            else:
                carry_i = mb.slice_by_index(
                    x=gdn_carry, begin=[0, i, 0, 0], end=[1, i + 1, 1, 1], name=f"{prefix}_gcarry_{i}"
                )
                gated_state = mb.mul(x=state, y=carry_i, name=f"{prefix}_rec_carry_{i}")
                kc_i = mb.slice_by_index(
                    x=k_cumdecay, begin=[0, h0, 0, 0], end=[1, h1, chunk_size, GDN_DIM], name=f"{prefix}_kc_{i}"
                )
                k_state = mb.matmul(x=kc_i, y=gated_state, name=f"{prefix}_rec_kst_{i}")
                v_new = mb.sub(x=nv_i, y=k_state, name=f"{prefix}_rec_vnew_{i}")
                inter = mb.matmul(x=q_dec_i, y=gated_state, name=f"{prefix}_rec_inter_{i}")
                intra_out = mb.matmul(x=intra_i, y=v_new, name=f"{prefix}_rec_intra_{i}")
                out_i = mb.add(x=inter, y=intra_out, name=f"{prefix}_rec_out_{i}")
                state_dec = mb.mul(x=gated_state, y=c_dec_i, name=f"{prefix}_rec_st_dec_{i}")
                state_upd = mb.matmul(
                    x=k_dec_i, y=v_new, transpose_x=True, transpose_y=False, name=f"{prefix}_rec_st_upd_{i}"
                )
                state = mb.add(x=state_dec, y=state_upd, name=f"{prefix}_rec_st_{i}")
            outs.append(out_i)

        out_chunks = mb.concat(values=outs, axis=1, name=f"{prefix}_rec_concat")
        out_t1 = mb.transpose(x=out_chunks, perm=[0, 1, 3, 2], name=f"{prefix}_out_t1")
        out_r1 = mb.reshape(x=out_t1, shape=[1, n_c, GDN_HEADS * GDN_DIM, chunk_size], name=f"{prefix}_out_r1")
        out_t2 = mb.transpose(x=out_r1, perm=[0, 2, 1, 3], name=f"{prefix}_out_t2")
        out_nchw = mb.reshape(x=out_t2, shape=[1, GDN_HEADS * GDN_DIM, 1, seq_len], name=f"{prefix}_out_nchw")
        return mb.mul(x=out_nchw, y=_f16(GDN_DIM ** -0.5), name=f"{prefix}_post_qscale")

    def _mlp_swiglu(x, prefix: str):
        gu = _conv_or_qdq(x, ane_weights[f"{prefix}_mlp_gu"], name=f"{prefix}_mlp_gu")
        gate = mb.slice_by_index(x=gu, begin=[0, 0, 0, 0], end=[1, MLP_INTER, 1, seq_len], name=f"{prefix}_mlp_g")
        up = mb.slice_by_index(
            x=gu, begin=[0, MLP_INTER, 0, 0], end=[1, 2 * MLP_INTER, 1, seq_len], name=f"{prefix}_mlp_u"
        )
        if swiglu_mode == "mul_sigmoid_reordered":
            gu_prod = mb.mul(x=gate, y=up, name=f"{prefix}_mlp_gu_prod")
            act = mb.mul(x=gu_prod, y=mb.sigmoid(x=gate, name=f"{prefix}_mlp_gsig"), name=f"{prefix}_mlp_act")
        elif swiglu_mode == "silu":
            act = mb.mul(x=_silu_op(gate, name=f"{prefix}_mlp_gsilu", is_mlp=True), y=up, name=f"{prefix}_mlp_act")
        else:
            raise ValueError(f"Unsupported swiglu_mode={swiglu_mode!r}")
        return _conv_or_qdq(act, ane_weights[f"{prefix}_mlp_dn"], name=f"{prefix}_mlp_dn")

    def _apply_linear_layer(h_in, tok_valid, gdn_carry, li: int):
        p = f"l{li}"
        xn = _rmsnorm_ch(h_in, ane_weights[f"{p}_in_ln"], name=f"{p}_in_ln")
        qkvzba = _conv_or_qdq(xn, ane_weights[f"{p}_qkvzba"], name=f"{p}_qkvzba")

        mqkv_raw = mb.slice_by_index(x=qkvzba, begin=[0, 0, 0, 0], end=[1, 6144, 1, seq_len], name=f"{p}_sl_qkv")
        mqkv = mb.mul(x=mqkv_raw, y=tok_valid, name=f"{p}_mqkv_mask")
        z = mb.slice_by_index(x=qkvzba, begin=[0, 6144, 0, 0], end=[1, 8192, 1, seq_len], name=f"{p}_sl_z")
        b_ch = mb.slice_by_index(x=qkvzba, begin=[0, 8192, 0, 0], end=[1, 8208, 1, seq_len], name=f"{p}_sl_b")
        a_ch = mb.slice_by_index(x=qkvzba, begin=[0, 8208, 0, 0], end=[1, 8224, 1, seq_len], name=f"{p}_sl_a")

        w_conv = _const_f16(ane_weights[f"{p}_conv1d"], name=f"{p}_conv1d_w")
        conv_raw = mb.conv(
            x=mqkv,
            weight=w_conv,
            pad_type="custom",
            pad=[0, 0, 3, 0],
            groups=6144,
            name=f"{p}_conv1d",
        )
        conv_act = _silu_op(conv_raw, name=f"{p}_conv_silu", is_mlp=False)
        q_ch = mb.slice_by_index(x=conv_act, begin=[0, 0, 0, 0], end=[1, 2048, 1, seq_len], name=f"{p}_sl_q")
        k_ch = mb.slice_by_index(x=conv_act, begin=[0, 2048, 0, 0], end=[1, 4096, 1, seq_len], name=f"{p}_sl_k")
        v_raw = mb.slice_by_index(x=conv_act, begin=[0, 4096, 0, 0], end=[1, 6144, 1, seq_len], name=f"{p}_sl_v")
        v_ch = mb.mul(x=v_raw, y=_f16(ftz_alpha), name=f"{p}_v_ftz_scale") if ftz_alpha != 1.0 else v_raw

        delta_nchw = _gated_delta_chunked_b1(
            q_ch,
            k_ch,
            v_ch,
            b_ch,
            a_ch,
            neg_A=ane_weights[f"{p}_neg_A"],
            dt_bias=ane_weights[f"{p}_dt_bias"],
            gdn_carry=gdn_carry,
            prefix=f"{p}_gdn",
        )
        delta_hd = mb.reshape(x=delta_nchw, shape=[1, GDN_HEADS, GDN_DIM, seq_len], name=f"{p}_delta_hd")
        z_hd = mb.reshape(x=z, shape=[1, GDN_HEADS, GDN_DIM, seq_len], name=f"{p}_z_hd")
        gnorm_w = ane_weights[f"{p}_gnorm_w"].reshape(1, 1, GDN_DIM, 1)
        normed_hd = _rmsnorm_axis(delta_hd, gnorm_w, axis=2, name=f"{p}_gnorm", eps=gnorm_eps)
        gated_hd = mb.mul(x=normed_hd, y=_silu_op(z_hd, name=f"{p}_z_silu", is_mlp=False), name=f"{p}_gated_hd")
        gated_nchw = mb.reshape(x=gated_hd, shape=[1, 2048, 1, seq_len], name=f"{p}_gated_nchw")

        attn_out = _conv_or_qdq(gated_nchw, ane_weights[f"{p}_out_proj"], name=f"{p}_out_proj")
        h_mid = mb.add(x=h_in, y=attn_out, name=f"{p}_res1")
        xn_post = _rmsnorm_ch(h_mid, ane_weights[f"{p}_post_ln"], name=f"{p}_post_ln")
        mlp_out = _mlp_swiglu(xn_post, prefix=p)
        return mb.add(x=h_mid, y=mlp_out, name=f"{p}_res2")

    def _apply_full_layer(h_in, attn_bias, cos32, sin32, li: int):
        p = f"l{li}"
        xn = _rmsnorm_ch(h_in, ane_weights[f"{p}_in_ln"], name=f"{p}_in_ln")
        qkvgate = _conv_or_qdq(xn, ane_weights[f"{p}_qkvgate"], name=f"{p}_qkvgate")

        q_ch = mb.slice_by_index(x=qkvgate, begin=[0, 0, 0, 0], end=[1, 2048, 1, seq_len], name=f"{p}_sl_q")
        k_ch = mb.slice_by_index(x=qkvgate, begin=[0, 2048, 0, 0], end=[1, 2560, 1, seq_len], name=f"{p}_sl_k")
        v_ch = mb.slice_by_index(x=qkvgate, begin=[0, 2560, 0, 0], end=[1, 3072, 1, seq_len], name=f"{p}_sl_v")
        gate_ch = mb.slice_by_index(x=qkvgate, begin=[0, 3072, 0, 0], end=[1, 5120, 1, seq_len], name=f"{p}_sl_gate")

        q_hsd = mb.transpose(
            x=mb.reshape(x=q_ch, shape=[1, ATTN_Q_HEADS, ATTN_HEAD_DIM, seq_len], name=f"{p}_qr"),
            perm=[0, 1, 3, 2],
            name=f"{p}_qt",
        )
        k_hsd = mb.transpose(
            x=mb.reshape(x=k_ch, shape=[1, ATTN_KV_HEADS, ATTN_HEAD_DIM, seq_len], name=f"{p}_kr"),
            perm=[0, 1, 3, 2],
            name=f"{p}_kt",
        )
        v_hsd = mb.transpose(
            x=mb.reshape(x=v_ch, shape=[1, ATTN_KV_HEADS, ATTN_HEAD_DIM, seq_len], name=f"{p}_vr"),
            perm=[0, 1, 3, 2],
            name=f"{p}_vt",
        )

        q_n = _rmsnorm_axis(q_hsd, ane_weights[f"{p}_qn"], axis=3, name=f"{p}_qnorm")
        k_n = _rmsnorm_axis(k_hsd, ane_weights[f"{p}_kn"], axis=3, name=f"{p}_knorm")

        def apply_partial_rope(x_hsd, nh: int, rp: str):
            r1 = mb.slice_by_index(x=x_hsd, begin=[0, 0, 0, 0], end=[1, nh, seq_len, ROPE_HALF], name=f"{rp}_r1")
            r2 = mb.slice_by_index(
                x=x_hsd, begin=[0, 0, 0, ROPE_HALF], end=[1, nh, seq_len, 2 * ROPE_HALF], name=f"{rp}_r2"
            )
            pass_part = mb.slice_by_index(
                x=x_hsd, begin=[0, 0, 0, 2 * ROPE_HALF], end=[1, nh, seq_len, ATTN_HEAD_DIM], name=f"{rp}_pass"
            )
            o1 = mb.sub(
                x=mb.mul(x=r1, y=cos32, name=f"{rp}_c1"),
                y=mb.mul(x=r2, y=sin32, name=f"{rp}_s2"),
                name=f"{rp}_o1",
            )
            o2 = mb.add(
                x=mb.mul(x=r2, y=cos32, name=f"{rp}_c2"),
                y=mb.mul(x=r1, y=sin32, name=f"{rp}_s1"),
                name=f"{rp}_o2",
            )
            return mb.concat(values=[o1, o2, pass_part], axis=3, name=f"{rp}_out")

        q_rope = apply_partial_rope(q_n, ATTN_Q_HEADS, f"{p}_qrope")
        k_rope = apply_partial_rope(k_n, ATTN_KV_HEADS, f"{p}_krope")

        k0 = mb.slice_by_index(x=k_rope, begin=[0, 0, 0, 0], end=[1, 1, seq_len, ATTN_HEAD_DIM], name=f"{p}_k0")
        k1 = mb.slice_by_index(x=k_rope, begin=[0, 1, 0, 0], end=[1, 2, seq_len, ATTN_HEAD_DIM], name=f"{p}_k1")
        k_rep = mb.concat(values=[k0, k0, k0, k0, k1, k1, k1, k1], axis=1, name=f"{p}_krep")

        v0 = mb.slice_by_index(x=v_hsd, begin=[0, 0, 0, 0], end=[1, 1, seq_len, ATTN_HEAD_DIM], name=f"{p}_v0")
        v1 = mb.slice_by_index(x=v_hsd, begin=[0, 1, 0, 0], end=[1, 2, seq_len, ATTN_HEAD_DIM], name=f"{p}_v1")
        v_rep = mb.concat(values=[v0, v0, v0, v0, v1, v1, v1, v1], axis=1, name=f"{p}_vrep")

        q_scaled = mb.mul(x=q_rope, y=_f16(ATTN_HEAD_DIM ** -0.5), name=f"{p}_q_scale")
        scores_raw = mb.matmul(
            x=q_scaled, y=k_rep, transpose_x=False, transpose_y=True, name=f"{p}_qk_scores"
        )
        scores = mb.add(x=scores_raw, y=attn_bias, name=f"{p}_bias_add")
        probs = mb.softmax(x=scores, axis=3, name=f"{p}_attn_softmax")
        attn_hsd = mb.matmul(x=probs, y=v_rep, name=f"{p}_av_matmul")

        attn_nchw = mb.reshape(
            x=mb.transpose(x=attn_hsd, perm=[0, 1, 3, 2], name=f"{p}_attn_t"),
            shape=[1, 2048, 1, seq_len],
            name=f"{p}_attn_nchw",
        )
        gated = mb.mul(x=attn_nchw, y=mb.sigmoid(x=gate_ch, name=f"{p}_gate_sig"), name=f"{p}_gated")
        attn_out = _conv_or_qdq(gated, ane_weights[f"{p}_o_proj"], name=f"{p}_o_proj")
        h_mid = mb.add(x=h_in, y=attn_out, name=f"{p}_res1")

        xn_post = _rmsnorm_ch(h_mid, ane_weights[f"{p}_post_ln"], name=f"{p}_post_ln")
        mlp_out = _mlp_swiglu(xn_post, prefix=p)
        return mb.add(x=h_mid, y=mlp_out, name=f"{p}_res2")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, HIDDEN, 1, seq_len), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, seq_len, seq_len), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, seq_len, ROPE_HALF), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, seq_len, ROPE_HALF), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, 1, seq_len), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n_c, 1, 1), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(h_in, attn_bias, cos32, sin32, tok_valid, gdn_carry):
        h = h_in
        for li in range(layer_start, layer_end):
            lt = layer_types[li]
            if lt == "linear_attention":
                h = _apply_linear_layer(h, tok_valid, gdn_carry, li)
            elif lt == "full_attention":
                h = _apply_full_layer(h, attn_bias, cos32, sin32, li)
            else:
                raise ValueError(f"Unsupported layer type {lt!r} at {li}")
        if apply_final_norm:
            h = _rmsnorm_ch(h, ane_weights["final_norm"], name="final_norm")
        return mb.identity(x=h, name="h_out")

    return prog


def _compile_mlpackage(pkg_path: Path, out_dir: Path) -> Path:
    """Compile a saved `.mlpackage` into `out_dir/<stem>.mlmodelc` without Xcode's `coremlc` CLI."""
    import coremltools as ct

    out_dir.mkdir(parents=True, exist_ok=True)
    compiled_tmp = Path(ct.utils.compile_model(str(pkg_path)))
    dst = out_dir / (pkg_path.stem + ".mlmodelc")
    if compiled_tmp.resolve() == dst.resolve():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    try:
        shutil.move(str(compiled_tmp), str(dst))
    except Exception:
        shutil.rmtree(compiled_tmp, ignore_errors=True)
        raise
    return dst


def _parse_seq_lens(raw: Sequence[str | int]) -> tuple[int, ...]:
    out: list[int] = []
    for item in raw:
        for tok in str(item).split(","):
            tok = tok.strip()
            if tok:
                val = int(tok)
                if val <= 0 or val % CHUNK_SIZE != 0:
                    raise ValueError(f"seq_len must be a positive multiple of {CHUNK_SIZE}, got {val}")
                out.append(val)
    if not out:
        raise ValueError("Expected at least one seq_len")
    return tuple(out)


def _resolve_coreai_preset(precision: str, coreai_preset: str | None) -> str:
    if coreai_preset is not None:
        p = coreai_preset.strip().lower()
        if p not in VALID_COREAI_PRESETS:
            raise ValueError(f"Unsupported coreai_preset={coreai_preset!r}; choose from {VALID_COREAI_PRESETS}")
        return p
    return {"fp16": "fp16", "w8a16": "w8", "w8a8": "w8"}.get(precision, "w4_per_block")


def build_ane_bundle_suite(
    *,
    home: str | Path | None = None,
    model_dir: str | Path | None = None,
    variant: str | None = None,
    seq_lens: tuple[int, ...] = DEFAULT_SEQ_LENS,
    layers_per_stage: int = DEFAULT_LAYERS_PER_STAGE,
    precision: str = "w8a16",
    inv_mode: str = "block",
    ftz_alpha: float = DEFAULT_FTZ_ALPHA,
    calib_stats_path: str | Path | None = None,
    outlier_threshold: float = 15.0,
    preserve_gdn_fp16: bool = True,
    backend: str = "coreml",
    keep_mlpackage: bool = False,
    ane_profile: str | None = None,
    silu_mode: str | None = None,
    rmsnorm_mode: str | None = None,
    coreai_preset: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    calib_batches: int = 2,
    guard_ir: bool = False,
    inspect_placement: bool = False,
    multifunction: bool = True,
    max_compile_workers: int | None = None,
) -> dict[str, Any]:
    eff_profile, eff_silu_mode, eff_rmsnorm_mode = resolve_ane_profile(
        ane_profile=ane_profile,
        silu_mode=silu_mode,
        rmsnorm_mode=rmsnorm_mode,
    )
    if backend not in ("coreml", "coreai"):
        raise ValueError(f"Unsupported backend={backend!r}; choose 'coreml' or 'coreai'")

    coreai_status: dict[str, Any] | None = None
    eff_coreai_preset: str | None = None
    if backend == "coreai":
        coreai_status = check_coreai_backend_status()
        if not coreai_status["export_ready"]:
            raise RuntimeError(
                "Core AI (.aimodel) offline export requires coreai-torch >= 0.4.1 "
                f"(status: {coreai_status['export_reason']})."
            )
        eff_coreai_preset = _resolve_coreai_preset(precision, coreai_preset)
    else:
        import coremltools as ct
        import coremltools.optimize.coreml as cto

    if variant:
        v_low = variant.strip().lower()
        if v_low in ("fp16", "w8a16", "w8a8"):
            precision = v_low
        elif v_low in ("block", "fwd_subst"):
            inv_mode = v_low

    # Validate quantization dtype export compatibility early before heavy model tracing/conversion.
    req_dtypes = (
        ["int8"]
        if backend == "coreml" and precision != "fp16"
        else (["int4"] if eff_coreai_preset in ("w4", "w4_per_block") else ["float16"])
    )
    disallowed = validate_export_quant_dtypes(req_dtypes, export_backend=backend)
    if disallowed:
        raise ValueError(f"Disallowed export quantization dtypes for {backend}: {disallowed}")

    paths = resolve_paths(home)
    if not paths.weights_npz.exists():
        if model_dir is not None:
            from export_weights import export_model_weights

            export_model_weights(home=paths.home, model_dir=model_dir, local_files_only=True)
        else:
            raise FileNotFoundError(f"Missing weights NPZ at {paths.weights_npz}; run export_weights.py first.")

    cfg_file = paths.tokenizer_dir / "config.json"
    cfg = PplxConfig.from_json(cfg_file) if cfg_file.exists() else DEFAULT_PPLX_CONFIG

    npz = np.load(paths.weights_npz)
    ane_weights = convert_weights_to_ane_layout({k: npz[k] for k in npz.files}, cfg=cfg)

    calib_stats: dict[str, Any] | None = None
    if calib_stats_path is not None:
        calib_stats = json.loads(Path(calib_stats_path).read_text(encoding="utf-8"))
    if backend == "coreml":
        if precision == "w8a8":
            if calib_stats is None:
                raise ValueError("--calib-stats is required for --precision w8a8")
            swiglu_mode = "mul_sigmoid_reordered"
        elif precision in ("w8a16", "fp16"):
            swiglu_mode = "silu"
        else:
            raise ValueError(f"Unsupported precision={precision!r}; choose fp16, w8a16, or w8a8")
    else:
        swiglu_mode = "silu"

    stage_ranges = plan_stage_ranges(cfg.num_hidden_layers, layers_per_stage=layers_per_stage)
    out_dir = paths.ane_models_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eff_prec_tag = eff_coreai_preset if backend == "coreai" and eff_coreai_preset is not None else precision
    config_hash = compute_ane_config_hash(
        backend=backend,
        precision=eff_prec_tag,
        inv_mode=inv_mode,
        silu_mode=eff_silu_mode,
        rmsnorm_mode=eff_rmsnorm_mode,
        ftz_alpha=ftz_alpha,
        preserve_gdn_fp16=preserve_gdn_fp16,
        layers_per_stage=layers_per_stage,
        outlier_threshold=outlier_threshold,
        num_layers=cfg.num_hidden_layers,
        calib_stats=calib_stats,
    )
    preserved_buckets = _load_preserved_buckets_for_hash(
        out_dir,
        config_hash=config_hash,
        requested_seq_lens=seq_lens,
        expected_num_stages=len(stage_ranges),
    )

    all_preserved = sorted(
        select_fp16_preserved_ops(
            0,
            cfg.num_hidden_layers,
            cfg.layer_types,
            calib_stats=calib_stats,
            outlier_threshold=outlier_threshold,
            preserve_gdn_fp16=preserve_gdn_fp16,
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_name": "perplexity-ai/pplx-embed-v2-context-0.6b",
        "backend": backend,
        "coreai_preset": eff_coreai_preset,
        "coreai_status": coreai_status,
        "ane_profile": eff_profile,
        "config_hash": config_hash,
        "precision": eff_prec_tag,
        "inv_mode": inv_mode,
        "silu_mode": eff_silu_mode,
        "rmsnorm_mode": eff_rmsnorm_mode,
        "ftz_alpha": float(ftz_alpha),
        "eps_floor_fp16": FP16_MIN_NORMAL,
        "preserve_gdn_fp16": bool(preserve_gdn_fp16),
        "fp16_preserved_ops": all_preserved if precision != "fp16" else ["all"],
        "chunk_size": CHUNK_SIZE,
        "hidden_size": cfg.hidden_size,
        "rope_half": cfg.rope_dim // 2,
        "num_layers": cfg.num_hidden_layers,
        "layers_per_stage": layers_per_stage,
        "num_stages": len(stage_ranges),
        "buckets": dict(preserved_buckets),
    }

    use_multifunction = bool(
        backend == "coreml"
        and multifunction
        and len(seq_lens) > 1
        and hasattr(ct.utils, "MultiFunctionDescriptor")
        and hasattr(ct.utils, "save_multifunction")
    )
    if use_multifunction:
        manifest["multifunction"] = True
        # Build per-stage multifunction .mlpackage bundles sharing weights across all seq_lens (1.30 -> 0.33 GB).
        t_mf = time.perf_counter()
        staging_dir = out_dir / f".mf_staging_{os.getpid()}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        mf_pkg_paths: list[Path] = []
        mf_stage_names: list[str] = []
        try:
            for st_idx, (l0, l1) in enumerate(stage_ranges):
                mf_stem = make_ane_multifunction_stage_stem(
                    st_idx,
                    eff_prec_tag,
                    ane_profile=eff_profile,
                    silu_mode=eff_silu_mode,
                    rmsnorm_mode=eff_rmsnorm_mode,
                    config_hash=config_hash,
                )
                stage_scales: dict[str, float | None] = {}
                if precision == "w8a8" and calib_stats is not None:
                    for li in range(l0, l1):
                        stage_scales.update(
                            select_qdq_scales(
                                li, cfg.layer_types[li], calib_stats, outlier_threshold=outlier_threshold
                            )
                        )
                desc = ct.utils.MultiFunctionDescriptor()
                for seq_len in seq_lens:
                    n_c = seq_len // CHUNK_SIZE
                    fn_name = pplx_stage_function_name(seq_len)
                    prog = build_stage_mil_program(
                        ane_weights,
                        seq_len=seq_len,
                        layer_start=l0,
                        layer_end=l1,
                        layer_types=cfg.layer_types,
                        apply_final_norm=(st_idx == len(stage_ranges) - 1),
                        chunk_size=CHUNK_SIZE,
                        inv_mode=inv_mode,
                        ftz_alpha=ftz_alpha,
                        qdq_scales=stage_scales,
                        calib_stats=calib_stats,
                        swiglu_mode=swiglu_mode,
                        silu_mode=eff_silu_mode,
                        rmsnorm_mode=eff_rmsnorm_mode,
                    )
                    ml = ct.convert(
                        prog,
                        inputs=[
                            ct.TensorType(name="h_in", shape=(1, HIDDEN, 1, seq_len), dtype=np.float16),
                            ct.TensorType(name="attn_bias", shape=(1, 1, seq_len, seq_len), dtype=np.float16),
                            ct.TensorType(name="cos32", shape=(1, 1, seq_len, ROPE_HALF), dtype=np.float16),
                            ct.TensorType(name="sin32", shape=(1, 1, seq_len, ROPE_HALF), dtype=np.float16),
                            ct.TensorType(name="tok_valid", shape=(1, 1, 1, seq_len), dtype=np.float16),
                            ct.TensorType(name="gdn_carry", shape=(1, n_c, 1, 1), dtype=np.float16),
                        ],
                        outputs=[ct.TensorType(name="h_out", dtype=np.float16)],
                        minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16,
                        skip_model_load=True,
                    )
                    if precision == "fp16":
                        ml_out = ml
                    else:
                        stage_preserved = select_fp16_preserved_ops(
                            l0,
                            l1,
                            cfg.layer_types,
                            calib_stats=calib_stats,
                            outlier_threshold=outlier_threshold,
                            preserve_gdn_fp16=preserve_gdn_fp16,
                        )
                        q_cfg = build_weight_quant_config(stage_preserved, weight_threshold=512, cto_module=cto)
                        ml_out = cto.linear_quantize_weights(ml, config=q_cfg)
                    if guard_ir:
                        rep = check_coreml_model(ml_out, model_name=f"{mf_stem}:{fn_name}", warn_matmul_ane=True)
                        if not rep.ok:
                            errs = "; ".join(f.message for f in rep.errors[:5])
                            raise RuntimeError(f"ir_dtype_guard failed on {mf_stem}:{fn_name}: {errs}")
                    tmp_pkg = staging_dir / f"{mf_stem}_{fn_name}.mlpackage"
                    ml_out.save(str(tmp_pkg))
                    desc.add_function(str(tmp_pkg), "main", fn_name)
                desc.default_function_name = pplx_stage_function_name(seq_lens[0])
                mf_pkg = out_dir / f"{mf_stem}.mlpackage"
                if mf_pkg.exists():
                    shutil.rmtree(mf_pkg)
                ct.utils.save_multifunction(desc, str(mf_pkg))
                mf_pkg_paths.append(mf_pkg)
                mf_stage_names.append(f"{mf_stem}.mlmodelc")
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        compile_jobs = [
            (pkg_p.stem, lambda p=pkg_p: _compile_mlpackage(p, out_dir))
            for pkg_p in mf_pkg_paths
        ]
        compile_buckets_parallel(compile_jobs, max_workers=max_compile_workers)
        if not keep_mlpackage:
            for pkg_p in mf_pkg_paths:
                shutil.rmtree(pkg_p, ignore_errors=True)

        stage_placements_mf: list[dict[str, Any]] = []
        if inspect_placement and not use_multifunction:
            for mf_name in mf_stage_names:
                stage_placements_mf.append(
                    inspect_coreml_bundle_placement(out_dir / mf_name, target_device="ane", model_name=mf_name)
                )
        per_bkt_s = round((time.perf_counter() - t_mf) / max(1, len(seq_lens)), 2)
        for seq_len in seq_lens:
            bkt_entry: dict[str, Any] = {
                "seq_len": int(seq_len),
                "config_hash": config_hash,
                "stages": list(mf_stage_names),
                "function_name": pplx_stage_function_name(seq_len),
                "build_s": per_bkt_s,
            }
            if stage_placements_mf:
                bkt_entry["placements"] = stage_placements_mf
            manifest["buckets"][str(seq_len)] = bkt_entry

    for seq_len in (() if use_multifunction else seq_lens):
        n_c = seq_len // CHUNK_SIZE
        stage_entries: list[str] = []
        stage_placements: list[dict[str, Any]] = []
        stage_coreai_meta: list[dict[str, Any]] = []
        stage_pkg_paths: list[Path] = []
        t_bucket = time.perf_counter()

        for st_idx, (l0, l1) in enumerate(stage_ranges):
            stem = make_ane_stage_stem(
                seq_len,
                st_idx,
                eff_prec_tag,
                ane_profile=eff_profile,
                silu_mode=eff_silu_mode,
                rmsnorm_mode=eff_rmsnorm_mode,
                config_hash=config_hash,
            )
            if backend == "coreai":
                assert eff_coreai_preset is not None
                res = run_pplx_coreai_export(
                    out_dir=out_dir,
                    ane_weights=ane_weights,
                    cfg=cfg,
                    layer_start=l0,
                    layer_end=l1,
                    apply_final_norm=(st_idx == len(stage_ranges) - 1),
                    seq_len=seq_len,
                    chunk_size=CHUNK_SIZE,
                    export_backend="coreai",
                    preset=eff_coreai_preset,
                    block_size=block_size,
                    preserve_gdn_fp16=preserve_gdn_fp16,
                    calib_batches=calib_batches,
                    ftz_alpha=ftz_alpha,
                    stem=stem,
                )
                stage_entries.append(res.asset_path.name)
                stage_coreai_meta.append(
                    {
                        "stage": res.asset_path.name,
                        "cache_key_sha256": res.cache_key_sha256,
                        "sensitivity": res.sensitivity_report,
                        "asset_summary": res.asset_summary,
                    }
                )
                continue

            stage_scales = {}
            if precision == "w8a8" and calib_stats is not None:
                for li in range(l0, l1):
                    stage_scales.update(
                        select_qdq_scales(
                            li, cfg.layer_types[li], calib_stats, outlier_threshold=outlier_threshold
                        )
                    )

            prog = build_stage_mil_program(
                ane_weights,
                seq_len=seq_len,
                layer_start=l0,
                layer_end=l1,
                layer_types=cfg.layer_types,
                apply_final_norm=(st_idx == len(stage_ranges) - 1),
                chunk_size=CHUNK_SIZE,
                inv_mode=inv_mode,
                ftz_alpha=ftz_alpha,
                qdq_scales=stage_scales,
                calib_stats=calib_stats,
                swiglu_mode=swiglu_mode,
                silu_mode=eff_silu_mode,
                rmsnorm_mode=eff_rmsnorm_mode,
            )

            ml = ct.convert(
                prog,
                inputs=[
                    ct.TensorType(name="h_in", shape=(1, HIDDEN, 1, seq_len), dtype=np.float16),
                    ct.TensorType(name="attn_bias", shape=(1, 1, seq_len, seq_len), dtype=np.float16),
                    ct.TensorType(name="cos32", shape=(1, 1, seq_len, ROPE_HALF), dtype=np.float16),
                    ct.TensorType(name="sin32", shape=(1, 1, seq_len, ROPE_HALF), dtype=np.float16),
                    ct.TensorType(name="tok_valid", shape=(1, 1, 1, seq_len), dtype=np.float16),
                    ct.TensorType(name="gdn_carry", shape=(1, n_c, 1, 1), dtype=np.float16),
                ],
                outputs=[ct.TensorType(name="h_out", dtype=np.float16)],
                minimum_deployment_target=ct.target.macOS15,
                compute_precision=ct.precision.FLOAT16,
                skip_model_load=True,
            )

            if precision == "fp16":
                ml_out = ml
            else:
                stage_preserved = select_fp16_preserved_ops(
                    l0,
                    l1,
                    cfg.layer_types,
                    calib_stats=calib_stats,
                    outlier_threshold=outlier_threshold,
                    preserve_gdn_fp16=preserve_gdn_fp16,
                )
                q_cfg = build_weight_quant_config(stage_preserved, weight_threshold=512, cto_module=cto)
                ml_out = cto.linear_quantize_weights(ml, config=q_cfg)

            if guard_ir:
                rep = check_coreml_model(
                    ml_out,
                    model_name=stem,
                    warn_matmul_ane=True,
                )
                if not rep.ok:
                    errs = "; ".join(f.message for f in rep.errors[:5])
                    raise RuntimeError(f"ir_dtype_guard failed on {stem}: {errs}")

            pkg_path = out_dir / f"{stem}.mlpackage"
            if pkg_path.exists():
                shutil.rmtree(pkg_path)
            ml_out.save(str(pkg_path))
            stage_pkg_paths.append(pkg_path)
            stage_entries.append(f"{stem}.mlmodelc")

        if stage_pkg_paths:
            compile_jobs = [
                (pkg_p.stem, lambda p=pkg_p: _compile_mlpackage(p, out_dir))
                for pkg_p in stage_pkg_paths
            ]
            compile_buckets_parallel(compile_jobs, max_workers=max_compile_workers)
            for pkg_p in stage_pkg_paths:
                compiled_path = out_dir / f"{pkg_p.stem}.mlmodelc"
                if inspect_placement:
                    stage_placements.append(
                        inspect_coreml_bundle_placement(compiled_path, target_device="ane", model_name=pkg_p.stem)
                    )
                if not keep_mlpackage:
                    shutil.rmtree(pkg_p, ignore_errors=True)

        bucket_entry: dict[str, Any] = {
            "seq_len": int(seq_len),
            "config_hash": config_hash,
            "stages": stage_entries,
            "build_s": round(time.perf_counter() - t_bucket, 2),
        }
        if stage_placements:
            bucket_entry["placements"] = stage_placements
        if stage_coreai_meta:
            bucket_entry["coreai_stages"] = stage_coreai_meta
        manifest["buckets"][str(seq_len)] = bucket_entry

    # Sort buckets numerically for deterministic manifest.json output.
    manifest["buckets"] = dict(sorted(manifest["buckets"].items(), key=lambda kv: int(kv[0])))
    manifest_path = out_dir / "manifest.json"
    tmp_manifest = out_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)

    active_stage_names = {
        name
        for bkt in manifest["buckets"].values()
        for name in bkt.get("stages", [])
    }
    _prune_stale_ane_bundles(out_dir, active_stage_names=active_stage_names, keep_mlpackage=keep_mlpackage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build staged ANE CoreML bundles for fast-pplx")
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--variant", default=None, help="Precision/inverse shortcut (e.g. fp16, w8a16, w8a8, block, fwd_subst)")
    parser.add_argument("--backend", default="coreml", choices=["coreml", "coreai"])
    parser.add_argument("--seq-lens", nargs="+", default=[str(s) for s in DEFAULT_SEQ_LENS])
    parser.add_argument("--layers-per-stage", type=int, default=DEFAULT_LAYERS_PER_STAGE)
    parser.add_argument("--precision", default="w8a16", choices=["fp16", "w8a16", "w8a8"])
    parser.add_argument("--inv-mode", default="block", choices=["block", "fwd_subst"])
    parser.add_argument(
        "--ane-profile",
        default=None,
        choices=list(VALID_ANE_PROFILES),
        help="Preset ANE speed/parity profile: 'fast' (silu+prescale, default), 'balanced' (tanh_gdn+max_abs), "
        "or 'parity' (tanh_all+max_abs). 'parity'/'balanced' is recommended for --dtype int8 or --dimensions 128 "
        "(prescale 1/64 inflates 2^-14 to 0.25 unscaled eps). Explicit --silu-mode/--rmsnorm-mode override the preset.",
    )
    parser.add_argument(
        "--silu-mode",
        default=None,
        choices=list(VALID_SILU_MODES),
        help="ANE SiLU op variant: 'silu' (default speed), 'tanh_gdn', or 'tanh_all' (max parity).",
    )
    parser.add_argument(
        "--rmsnorm-mode",
        default=None,
        choices=list(VALID_RMSNORM_MODES),
        help="Channel RMSNorm scaling: 'prescale' (default) or 'max_abs' (overflow-safe dynamic scaling).",
    )
    parser.add_argument("--ftz-alpha", type=float, default=DEFAULT_FTZ_ALPHA)
    parser.add_argument("--calib-stats", type=Path, default=None)
    parser.add_argument("--outlier-threshold", type=float, default=15.0)
    parser.add_argument("--preserve-gdn-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-mlpackage", action="store_true")
    parser.add_argument("--coreai-preset", default=None, choices=list(VALID_COREAI_PRESETS))
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--calib-batches", type=int, default=2)
    parser.add_argument("--guard-ir", action="store_true", help="Run ir_dtype_guard static MIL check before saving")
    parser.add_argument("--inspect-placement", action="store_true", help="Capture MLComputePlan ANE placement summary")
    parser.add_argument(
        "--multifunction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge all seq_len functions per stage into shared-weight multifunction .mlmodelc bundles (default: True)",
    )
    parser.add_argument("--compile-workers", type=int, default=None, help="Max parallel CoreML compile workers")
    args = parser.parse_args()

    manifest = build_ane_bundle_suite(
        home=args.home,
        model_dir=args.model_dir,
        variant=args.variant,
        seq_lens=_parse_seq_lens(args.seq_lens),
        layers_per_stage=args.layers_per_stage,
        precision=args.precision,
        inv_mode=args.inv_mode,
        ftz_alpha=args.ftz_alpha,
        calib_stats_path=args.calib_stats,
        outlier_threshold=args.outlier_threshold,
        preserve_gdn_fp16=args.preserve_gdn_fp16,
        backend=args.backend,
        keep_mlpackage=args.keep_mlpackage,
        ane_profile=args.ane_profile,
        silu_mode=args.silu_mode,
        rmsnorm_mode=args.rmsnorm_mode,
        coreai_preset=args.coreai_preset,
        block_size=args.block_size,
        calib_batches=args.calib_batches,
        guard_ir=args.guard_ir,
        inspect_placement=args.inspect_placement,
        multifunction=args.multifunction,
        max_compile_workers=args.compile_workers,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
