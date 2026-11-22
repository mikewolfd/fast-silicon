"""Shared CoreAI-Opt calibration, export, IR guard, and placement pipeline.

Single source of truth across `fast-pplx` (`Qwen3.5` linear+full attention) and
`fast-decision-models` (`Laya-Typed-421M` ModernBERT) for:
  - Decoupled offline `coreai-torch >= 0.4.1` / `coreai-opt >= 0.3.0` config & guards
  - CoreAI Pattern 7 block/group divisibility pre-checks & `ExportBackend.Torch`-only format rejection
  - `Quantizer.prepare` -> `calibration_mode()` -> `Quantizer.finalize` + sensitivity report
  - `TorchConverter` -> `.aimodel` export + `AIModelAsset.summary` + specialization cache keys
  - `ct.convert` (`ExportBackend.CoreML`) -> `.mlpackage` + `check_coreml_model` + `MLComputePlan` placement
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from .hw_check import (
    coreai_specialization_cache_key_parts,
    extract_coreml_compute_plan_ops,
    probe_core_ai_support,
    summarize_accelerator_placement,
)
from .ir_dtype_guard import (
    GuardReport,
    check_coreml_model,
    check_divisibility,
    validate_export_quant_dtypes,
)

FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)  # 2^-14 = 6.103515625e-5
DEFAULT_BLOCK_SIZE = 32
VALID_COREAI_PRESETS: tuple[str, ...] = ("fp16", "w8", "w4", "w4_per_block", "fp4_fp8")
_QWEN35_SMALL_GDN_RE = re.compile(
    r"(?:qwen\s*3[._-]?5|kev)[._\s-]*(?:0[._]?8|2)\s*b(?![0-9a-z])",
    re.IGNORECASE,
)

_PRESET_NOMINAL_BITS: dict[str, float] = {
    "fp16": 16.0,
    "w8": 8.0,
    "w4": 4.0,
    "w4_per_block": 4.0,
    "fp4_fp8": 4.0,
}


def _sanitize_custom_metadata(value: Any) -> Any:
    """Strip None values and normalize non-finite floats for AIModelAssetMetadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_custom_metadata(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_sanitize_custom_metadata(v) for v in value if v is not None]
    return value


@dataclass(frozen=True)
class CoreAiExportResult:
    """Result metadata from a Core AI (`.aimodel`) or CoreML (`.mlpackage`) build/calibration run."""

    backend: str
    preset: str
    asset_path: Path
    cache_key_parts: dict[str, Any]
    cache_key_sha256: str
    asset_summary: dict[str, Any] | None = None
    sensitivity_report: dict[str, Any] | None = None
    guard_report: dict[str, Any] | None = None
    placement_summary: dict[str, Any] | None = None


def clamp_fp16_eps(eps: float) -> float:
    """Clamp epsilon >= 2^-14 so ANE/CoreAI FP16 FTZ never flushes var + eps to zero."""
    return float(max(float(eps), FP16_MIN_NORMAL))


def _flat_outputs(out: Any) -> np.ndarray:
    """Flatten tensor or tuple of tensors into a 1-D NumPy vector for cosine/MSE scoring."""
    if isinstance(out, (tuple, list)):
        parts = [np.asarray(t.detach().cpu().numpy(), dtype=np.float64).reshape(-1) for t in out]
        return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float64)
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


def _extract_batch_feature_vector(batch_inputs: Any) -> np.ndarray:
    """Flatten nested batch inputs (tensors/arrays/sequences/mappings) into one 1-D float64 vector."""
    parts: list[np.ndarray] = []

    def _visit(obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            parts.append(np.asarray(obj.detach().cpu().numpy(), dtype=np.float64).reshape(-1))
        elif isinstance(obj, np.ndarray):
            parts.append(np.asarray(obj, dtype=np.float64).reshape(-1))
        elif isinstance(obj, Mapping):
            for k in sorted(obj.keys(), key=str):
                _visit(obj[k])
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                _visit(item)

    _visit(batch_inputs)
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts)


def assess_calibration_batch_diversity(
    batches: Sequence[Any],
    *,
    min_l2_diff: float = 1e-6,
) -> dict[str, Any]:
    """Measure sample diversity and dynamic-range spread across calibration input batches."""
    n_batches = len(batches)
    vecs = [_extract_batch_feature_vector(b) for b in batches]
    rms_vals = [float(np.sqrt(np.mean(v * v))) if v.size else 0.0 for v in vecs]
    max_rms = max(rms_vals) if rms_vals else 0.0
    min_rms = min(rms_vals) if rms_vals else 0.0
    rms_scale_ratio = float(max_rms / max(min_rms, 1e-12)) if n_batches >= 2 and max_rms > 0.0 else 1.0

    if n_batches <= 1:
        return {
            "n_batches": n_batches,
            "unique_batches": n_batches,
            "all_batches_identical": False,
            "diverse": bool(n_batches == 1 and max_rms > 0.0),
            "mean_pairwise_l2_diff": 0.0,
            "min_pairwise_l2_diff": 0.0,
            "min_pairwise_cosine": 1.0,
            "max_rms": round(max_rms, 6),
            "min_rms": round(min_rms, 6),
            "rms_scale_ratio": round(rms_scale_ratio, 6),
        }

    pair_l2: list[float] = []
    pair_cos: list[float] = []
    reps: list[np.ndarray] = []
    for i, vi in enumerate(vecs):
        if not any(
            vi.shape == r.shape and float(np.linalg.norm(vi - r)) <= min_l2_diff
            for r in reps
        ):
            reps.append(vi)
        ni = float(np.linalg.norm(vi))
        for j in range(i + 1, n_batches):
            vj = vecs[j]
            if vi.shape != vj.shape:
                pair_l2.append(float("inf"))
                pair_cos.append(0.0)
                continue
            l2 = float(np.linalg.norm(vi - vj))
            pair_l2.append(l2)
            nj = float(np.linalg.norm(vj))
            denom = ni * nj
            pair_cos.append(float(np.dot(vi, vj) / denom) if denom > 0.0 else 0.0)

    finite_l2 = [x for x in pair_l2 if np.isfinite(x)]
    mean_l2 = float(np.mean(finite_l2)) if finite_l2 else (float(min_l2_diff) + 1.0 if pair_l2 else 0.0)
    min_l2 = min(pair_l2) if pair_l2 else 0.0
    min_cos = min(pair_cos) if pair_cos else 1.0
    unique_cnt = len(reps)
    all_identical = unique_cnt <= 1
    diverse = bool(unique_cnt >= 2 and max_rms > 1e-12 and (mean_l2 > min_l2_diff or rms_scale_ratio > 1.01))
    return {
        "n_batches": n_batches,
        "unique_batches": unique_cnt,
        "all_batches_identical": all_identical,
        "diverse": diverse,
        "mean_pairwise_l2_diff": round(mean_l2, 6),
        "min_pairwise_l2_diff": round(min_l2 if np.isfinite(min_l2) else mean_l2, 6),
        "min_pairwise_cosine": round(min_cos, 6),
        "max_rms": round(max_rms, 6),
        "min_rms": round(min_rms, 6),
        "rms_scale_ratio": round(rms_scale_ratio, 6),
    }


def _score_teacher_vs_quant(ref_vec: np.ndarray, q_vec: np.ndarray) -> tuple[float, float, bool]:
    """Return `(cosine, mse, all_finite)` comparing flattened teacher vs quantized tensors."""
    finite_ok = bool(np.all(np.isfinite(ref_vec)) and np.all(np.isfinite(q_vec)))
    if not finite_ok or ref_vec.size == 0 or q_vec.size == 0:
        return 0.0, float("inf") if not finite_ok else 0.0, finite_ok
    denom = float(np.linalg.norm(ref_vec) * np.linalg.norm(q_vec))
    cos = float(np.dot(ref_vec, q_vec) / denom) if denom > 0.0 else 0.0
    mse = float(np.mean((ref_vec - q_vec) ** 2))
    return cos, mse, True


def evaluate_multi_batch_calibration_gate(
    ref_outputs: Sequence[np.ndarray],
    quant_outputs: Sequence[np.ndarray],
    *,
    batch_diversity: Mapping[str, Any] | None = None,
    calib_batches: int = 1,
    min_calib_batches: int = 1,
    require_calib_diversity: bool = False,
    min_cosine: float | None = None,
    max_cosine_std: float | None = None,
    max_cosine_drop: float | None = None,
    max_mse: float | None = None,
    check_finite: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Score per-batch teacher-vs-quantized fidelity and enforce multi-batch calibration gates."""
    if len(ref_outputs) != len(quant_outputs) or not ref_outputs:
        raise ValueError("ref_outputs and quant_outputs must be non-empty sequences of matching length")

    cos_list: list[float] = []
    mse_list: list[float] = []
    all_finite = True
    for ref_v, q_v in zip(ref_outputs, quant_outputs):
        c, m, fin = _score_teacher_vs_quant(np.asarray(ref_v, dtype=np.float64), np.asarray(q_v, dtype=np.float64))
        cos_list.append(c)
        mse_list.append(m)
        all_finite = all_finite and fin

    mean_cos = float(np.mean(cos_list))
    min_cos = float(np.min(cos_list))
    max_cos = float(np.max(cos_list))
    cos_std = float(np.std(cos_list))
    max_drop = float(max_cos - min_cos)
    finite_mses = [x for x in mse_list if np.isfinite(x)]
    mean_mse = float(np.mean(finite_mses)) if finite_mses else float("inf")
    max_mse_val = float(np.max(mse_list))

    eff_min_batches = max(2, int(min_calib_batches)) if strict else max(1, int(min_calib_batches))
    eff_req_div = True if strict else bool(require_calib_diversity)
    eff_min_cos = 0.90 if (strict and min_cosine is None) else min_cosine
    eff_max_std = 0.05 if (strict and max_cosine_std is None) else max_cosine_std
    eff_max_drop = 0.08 if (strict and max_cosine_drop is None) else max_cosine_drop

    failures: list[str] = []
    if check_finite and not all_finite:
        failures.append("nonfinite_output")
    if int(calib_batches) < eff_min_batches:
        failures.append(f"calib_batches={int(calib_batches)} < min_calib_batches={eff_min_batches}")
    if eff_req_div and int(calib_batches) >= 2:
        if batch_diversity is None or not bool(batch_diversity.get("diverse", False)):
            failures.append("calib_batches_not_diverse")
    if eff_min_cos is not None and min_cos < float(eff_min_cos):
        failures.append(f"min_cosine={min_cos:.6f} < min_cosine={float(eff_min_cos):.6f}")
    if eff_max_std is not None and cos_std > float(eff_max_std):
        failures.append(f"cosine_std={cos_std:.6f} > max_cosine_std={float(eff_max_std):.6f}")
    if eff_max_drop is not None and max_drop > float(eff_max_drop):
        failures.append(f"max_cosine_drop={max_drop:.6f} > max_cosine_drop={float(eff_max_drop):.6f}")
    if max_mse is not None and (not np.isfinite(max_mse_val) or max_mse_val > float(max_mse)):
        failures.append(f"max_mse={max_mse_val:.8g} > max_mse={float(max_mse):.8g}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_calib_batches": int(calib_batches),
        "n_eval_batches": len(ref_outputs),
        "all_finite": all_finite,
        "per_batch_cosine": [round(c, 6) for c in cos_list],
        "per_batch_mse": [round(m, 8) if np.isfinite(m) else float("inf") for m in mse_list],
        "mean_cosine_vs_teacher": round(mean_cos, 6),
        "min_cosine_vs_teacher": round(min_cos, 6),
        "max_cosine_vs_teacher": round(max_cos, 6),
        "cosine_std": round(cos_std, 6),
        "max_cosine_drop": round(max_drop, 6),
        "mean_mse_vs_teacher": round(mean_mse, 8) if np.isfinite(mean_mse) else float("inf"),
        "max_mse_vs_teacher": round(max_mse_val, 8) if np.isfinite(max_mse_val) else float("inf"),
    }


def compute_analytical_bpw(module: Any, excluded_modules: set[str], bits_quant: float) -> float:
    """Estimate bits-per-weight analytically across leaf parameterized modules (safe in Graph mode)."""
    total_w = 0
    total_b = 0.0
    for name, sub in module.named_modules():
        w = getattr(sub, "weight", None)
        if w is None or not hasattr(w, "numel") or list(sub.children()):
            continue
        n = int(w.numel())
        total_w += n
        total_b += n * (16.0 if name in excluded_modules else float(bits_quant))
    return round(total_b / max(1, total_w), 4)


def _cast_floating_tensors(obj: Any, dtype: Any) -> Any:
    """Cast floating-point PyTorch tensors recursively while keeping integer/bool tensors untouched."""
    if hasattr(obj, "is_floating_point") and obj.is_floating_point():
        return obj.to(dtype=dtype)
    if isinstance(obj, tuple):
        return tuple(_cast_floating_tensors(x, dtype) for x in obj)
    if isinstance(obj, list):
        return [_cast_floating_tensors(x, dtype) for x in obj]
    if isinstance(obj, dict):
        return {k: _cast_floating_tensors(v, dtype) for k, v in obj.items()}
    return obj


def _resolve_coreai_compute_dtype(export_backend: str, preset: str, compute_dtype: str = "auto") -> str:
    """Resolve effective graph compute dtype; default `coreai` dense/int presets to FP16 for ANE eligibility."""
    norm = compute_dtype.strip().lower()
    if norm in ("auto", ""):
        if export_backend.strip().lower() == "coreai" and preset.strip().lower() in (
            "fp16",
            "w8",
            "w4",
            "w4_per_block",
        ):
            return "fp16"
        return "fp32"
    if norm in ("fp16", "float16"):
        return "fp16"
    if norm in ("fp32", "float32"):
        return "fp32"
    raise ValueError(f"Unsupported compute_dtype={compute_dtype!r}; choose 'auto', 'fp16', or 'fp32'")


def inspect_coreml_bundle_placement(
    bundle_path: str | Path,
    *,
    target_device: str = "ane",
    compute_units: Any = None,
    model_name: str | None = None,
    include_ops: bool = False,
) -> dict[str, Any]:
    """Load `MLComputePlan` for a `.mlpackage` or `.mlmodelc` and summarize accelerator placement (P4)."""
    import coremltools as ct

    p = Path(bundle_path)
    if compute_units is None:
        compute_units = ct.ComputeUnit.ALL
    compiled_tmp: Path | None = None
    try:
        load_path = p
        if p.suffix == ".mlpackage":
            compiled_tmp = Path(ct.utils.compile_model(str(p)))
            load_path = compiled_tmp
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(load_path), compute_units=compute_units)
    finally:
        if compiled_tmp is not None and compiled_tmp.exists():
            shutil.rmtree(compiled_tmp, ignore_errors=True)
    return summarize_accelerator_placement(
        extract_coreml_compute_plan_ops(plan),
        target_device=target_device,
        model_name=model_name or p.stem,
        include_ops=include_ops,
    )


def _is_qwen35_small_gdn_model(model_name: str | None) -> bool:
    """Return True when `model_name` identifies Qwen3.5-{0.8B, 2B} / Kev-0.8B GDN models."""
    if not model_name:
        return False
    return bool(_QWEN35_SMALL_GDN_RE.search(str(model_name)))


def _has_gdn_module_exclusion(excluded_modules: Sequence[str]) -> bool:
    for mod in excluded_modules:
        low = str(mod).lower()
        if "linear_attn" in low or "gated_delta" in low or "gdn" in low:
            return True
    return False


def build_coreai_opt_config(
    *,
    preset: str = "w4_per_block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    export_backend: str = "coreml",
    excluded_modules: Sequence[str] = (),
    requested_dtypes: Sequence[Any] = (),
    weight_shapes: Mapping[str, tuple[int, ...]] | None = None,
    block_axis: int = 1,
    os_version: str | None = None,
    model_name: str | None = None,
    allow_experimental_coreml_w4_block: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build a Pattern-7 & export-backend guarded `coreai_opt.quantization.QuantizerConfig`."""
    preset_norm = preset.strip().lower()
    if preset_norm not in VALID_COREAI_PRESETS:
        raise ValueError(f"Unsupported preset={preset!r}; choose from {VALID_COREAI_PRESETS}")

    backend_norm = export_backend.strip().lower()
    if backend_norm not in ("coreml", "coreai", "torch"):
        raise ValueError(f"Unsupported export_backend={export_backend!r}")

    probe = probe_core_ai_support(
        hw={"system": "Darwin", "machine": "arm64", "macos_version": os_version} if os_version else None
    )
    if preset_norm == "w4_per_block" and backend_norm == "coreml" and not allow_experimental_coreml_w4_block:
        raise ValueError(
            "preset='w4_per_block' on export_backend='coreml' produces corrupted CoreML logits and CPU "
            "fallback on ios18.conv (use preset='w8' for CoreML, preset='w4_per_block' with "
            "export_backend='coreai', or pass allow_experimental_coreml_w4_block=True)"
        )
    if preset_norm == "fp4_fp8" and backend_norm == "coreml":
        raise ValueError("preset='fp4_fp8' requires export_backend='coreai' (CoreML target does not support FP4/FP8)")
    if preset_norm == "fp4_fp8" and os_version is not None and not probe["fp4_fp8_preset_eligible"]:
        raise ValueError(f"preset='fp4_fp8' requires macOS 27.2+, detected {os_version!r}")
    if (
        preset_norm == "fp4_fp8"
        and _is_qwen35_small_gdn_model(model_name)
        and not _has_gdn_module_exclusion(excluded_modules)
    ):
        raise ValueError(
            f"Blanket preset='fp4_fp8' is blocked on {model_name!r} (Qwen3.5-{{0.8B, 2B}}) because "
            "per-tensor FP8 activation quantization overflows GatedDeltaNet (GDN) layers to 100% NaN; "
            "keep activations in BF16 or exclude linear_attn/GDN modules."
        )

    checked_dtypes = list(requested_dtypes)
    if not checked_dtypes:
        if preset_norm in ("w4", "w4_per_block"):
            checked_dtypes = ["int4"]
        elif preset_norm == "w8":
            checked_dtypes = ["int8"]
        elif preset_norm == "fp4_fp8":
            checked_dtypes = ["float4_e2m1fn_x2", "float8_e4m3fn"]
        else:
            checked_dtypes = ["float16"]

    disallowed = validate_export_quant_dtypes(checked_dtypes, export_backend=backend_norm)
    if disallowed:
        raise ValueError(
            f"Quantization dtype(s) {disallowed} are only supported on ExportBackend.Torch, "
            f"not on export_backend={backend_norm!r}."
        )

    import torch
    from coreai_opt import ExportBackend
    from coreai_opt.quantization import (
        ModuleQuantizerConfig,
        QuantizationSpec,
        QuantizerConfig,
    )

    if backend_norm == "coreml":
        eb = ExportBackend.CoreML
    elif backend_norm == "coreai":
        eb = ExportBackend.CoreAI
    else:
        eb = ExportBackend.Torch

    eff_excluded = set(excluded_modules)
    skipped_non_divisible: dict[str, tuple[int, int]] = {}
    if preset_norm == "w4_per_block" and weight_shapes is not None:
        candidate_shapes = {k: shp for k, shp in weight_shapes.items() if k not in eff_excluded}
        bad = check_divisibility(candidate_shapes, axis=block_axis, block_size=block_size, min_ndim=2)
        if bad:
            skipped_non_divisible = bad
            eff_excluded.update(bad.keys())

    eff_weight_dtype: str | None = None
    eff_activation_dtype: str | None = None
    if preset_norm == "fp16":
        cfg = QuantizerConfig()
        eff_weight_dtype = "float16"
    elif preset_norm == "w8":
        cfg = QuantizerConfig.presets.w8()
        eff_weight_dtype = "int8"
    elif preset_norm == "w4":
        cfg = QuantizerConfig.presets.w4()
        eff_weight_dtype = "int4"
    elif preset_norm == "w4_per_block":
        cfg = QuantizerConfig.presets.w4_per_block(block_size=int(block_size), axis=block_axis)
        eff_weight_dtype = "int4"
    else:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)

        w_spec: Any = None
        if fp4_dtype is not None:
            try:
                w_spec = QuantizationSpec(dtype=fp4_dtype)
                eff_weight_dtype = str(fp4_dtype)
            except (TypeError, ValueError):
                w_spec = None
        if w_spec is None and fp8_dtype is not None:
            w_spec = QuantizationSpec(dtype=fp8_dtype)
            eff_weight_dtype = str(fp8_dtype)

        act_spec: Any = None
        if fp8_dtype is not None:
            try:
                act_spec = QuantizationSpec(dtype=fp8_dtype)
                eff_activation_dtype = str(fp8_dtype)
            except (TypeError, ValueError):
                act_spec = None

        mod_kw: dict[str, Any] = {
            "op_state_spec": {"weight": w_spec},
            "op_input_spec": {"*": act_spec},
            "op_output_spec": {"*": act_spec},
        }
        try:
            mod_params = inspect.signature(ModuleQuantizerConfig).parameters
            if "kv_cache_spec" in mod_params and act_spec is not None:
                mod_kw["kv_cache_spec"] = act_spec
        except (TypeError, ValueError):
            pass

        cfg = QuantizerConfig().set_global(ModuleQuantizerConfig(**mod_kw))

    # Graph mode validates module_name_configs entries strictly as ModuleQuantizerConfig (None raises AttributeError).
    noop_mod_cfg = ModuleQuantizerConfig(
        op_state_spec={"*": None},
        op_input_spec={"*": None},
        op_output_spec={"*": None},
    )
    for mod_name in sorted(eff_excluded):
        cfg.set_module_name(mod_name, noop_mod_cfg)

    meta = {
        "preset": preset_norm,
        "export_backend": str(eb.value),
        "block_size": int(block_size) if preset_norm == "w4_per_block" else None,
        "excluded_modules": sorted(eff_excluded),
        "skipped_non_divisible": {k: list(v) for k, v in skipped_non_divisible.items()},
        "checked_dtypes": [str(x) for x in checked_dtypes],
        "weight_dtype": eff_weight_dtype,
        "activation_dtype": eff_activation_dtype,
    }
    return cfg, meta


def _materialize_eval_batches(
    eval_inputs: Sequence[tuple[Any, ...]] | Callable[[int], tuple[Any, ...]] | None,
    raw_calib_batches: Sequence[tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    """Resolve evaluation batches from explicit sequence/callable or cached calibration batches."""
    if eval_inputs is None:
        return list(raw_calib_batches)
    if callable(eval_inputs):
        return [eval_inputs(bi) for bi in range(len(raw_calib_batches))]
    batches = [tuple(b) for b in eval_inputs]
    return batches if batches else list(raw_calib_batches)


def run_coreai_opt_calibration_and_export(
    *,
    module_factory: Callable[[], Any],
    sample_inputs_factory: Callable[[int], tuple[Any, ...]],
    out_dir: str | Path,
    stem: str,
    model_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    ml_inputs: Sequence[Any] | None = None,
    ml_outputs: Sequence[Any] | None = None,
    export_backend: str = "coreai",
    preset: str = "w4_per_block",
    block_size: int = DEFAULT_BLOCK_SIZE,
    block_axis: int = 1,
    excluded_modules: Sequence[str] = (),
    requested_dtypes: Sequence[Any] = (),
    calib_batches: int = 2,
    eval_inputs: Sequence[tuple[Any, ...]] | Callable[[int], tuple[Any, ...]] | None = None,
    min_calib_batches: int = 1,
    require_calib_diversity: bool = False,
    strict_calibration_gate: bool = False,
    min_calib_cosine: float | None = None,
    max_calib_cosine_std: float | None = None,
    max_calib_cosine_drop: float | None = None,
    max_calib_mse: float | None = None,
    fail_on_nonfinite: bool = True,
    spec_metadata: Mapping[str, Any] | None = None,
    flag_add_fp16_cast: bool = False,
    inspect_placement: bool = False,
    target_device: str = "ane",
    allow_experimental_coreml_w4_block: bool = False,
    compute_dtype: str = "auto",
) -> CoreAiExportResult:
    """Calibrate, quantize (CoreAI-Opt), gate across batches, export, and guard any `nn.Module`."""
    import torch

    eff_calib_batches = max(1, int(calib_batches))
    eff_min_calib = max(2, int(min_calib_batches)) if strict_calibration_gate else max(1, int(min_calib_batches))
    eff_req_div = bool(require_calib_diversity or strict_calibration_gate)
    gate_enforced = bool(
        strict_calibration_gate
        or require_calib_diversity
        or int(min_calib_batches) > 1
        or min_calib_cosine is not None
        or max_calib_cosine_std is not None
        or max_calib_cosine_drop is not None
        or max_calib_mse is not None
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    eff_compute_dtype = _resolve_coreai_compute_dtype(export_backend, preset, compute_dtype)
    module = module_factory().eval()
    raw_calib_batches = [sample_inputs_factory(bi) for bi in range(eff_calib_batches)]
    raw_eval_batches = _materialize_eval_batches(eval_inputs, raw_calib_batches)
    batch_diversity = assess_calibration_batch_diversity(raw_calib_batches)
    if gate_enforced:
        if eff_calib_batches < eff_min_calib:
            raise ValueError(
                f"Multi-batch calibration gate failed: calib_batches={eff_calib_batches} < min_calib_batches={eff_min_calib}"
            )
        if eff_req_div and eff_calib_batches >= 2 and not batch_diversity["diverse"]:
            raise ValueError(
                f"Multi-batch calibration gate failed: duplicate/non-diverse calibration batches ({batch_diversity})"
            )

    raw_sample = raw_calib_batches[0]

    weight_shapes = {
        name: tuple(int(d) for d in sub.weight.shape)
        for name, sub in module.named_modules()
        if not list(sub.children()) and getattr(sub, "weight", None) is not None
    }
    qcfg, qmeta = build_coreai_opt_config(
        preset=preset,
        block_size=block_size,
        export_backend=export_backend,
        excluded_modules=excluded_modules,
        requested_dtypes=requested_dtypes,
        weight_shapes=weight_shapes,
        block_axis=block_axis,
        model_name=model_name,
        allow_experimental_coreml_w4_block=allow_experimental_coreml_w4_block,
    )

    with torch.no_grad():
        ref_vecs = [_flat_outputs(module(*b)) for b in raw_eval_batches]

    if eff_compute_dtype == "fp16":
        # Cast floating weights/buffers + sample inputs to FP16 so Core AI exports stay ANE/FP16-eligible.
        module = module.to(dtype=torch.float16)
        calib_batches_cast = [_cast_floating_tensors(b, torch.float16) for b in raw_calib_batches]
        eval_batches_cast = [_cast_floating_tensors(b, torch.float16) for b in raw_eval_batches]
    else:
        calib_batches_cast = list(raw_calib_batches)
        eval_batches_cast = list(raw_eval_batches)
    sample = calib_batches_cast[0]

    if preset == "fp16":
        finalized = module
        if eff_compute_dtype == "fp16":
            with torch.no_grad():
                q_vecs = [_flat_outputs(finalized(*b)) for b in eval_batches_cast]
        else:
            q_vecs = list(ref_vecs)
    else:
        from coreai_opt import ExportBackend
        from coreai_opt.quantization import Quantizer

        eb = ExportBackend.CoreML if export_backend == "coreml" else ExportBackend.CoreAI
        quantizer = Quantizer(module, qcfg)
        prepared = quantizer.prepare(sample)
        with quantizer.calibration_mode():
            with torch.no_grad():
                for cb in calib_batches_cast:
                    prepared(*cb)
        with torch.no_grad():
            q_vecs = [_flat_outputs(prepared(*eb_in)) for eb_in in eval_batches_cast]
        finalized = quantizer.finalize(backend=eb)

    gate_res = evaluate_multi_batch_calibration_gate(
        ref_vecs,
        q_vecs,
        batch_diversity=batch_diversity,
        calib_batches=eff_calib_batches,
        min_calib_batches=eff_min_calib,
        require_calib_diversity=eff_req_div,
        min_cosine=min_calib_cosine,
        max_cosine_std=max_calib_cosine_std,
        max_cosine_drop=max_calib_cosine_drop,
        max_mse=max_calib_mse,
        check_finite=bool(fail_on_nonfinite),
        strict=strict_calibration_gate,
    )
    if fail_on_nonfinite and not gate_res["all_finite"]:
        raise ValueError("Multi-batch calibration gate failed: nonfinite quantized output")
    if gate_enforced and not gate_res["passed"]:
        raise ValueError(f"Multi-batch calibration gate failed: {'; '.join(gate_res['failures'])}")

    cos_b0 = float(gate_res["per_batch_cosine"][0])
    mse_b0 = gate_res["per_batch_mse"][0]
    bits_q = _PRESET_NOMINAL_BITS[preset.strip().lower()]
    sensitivity = {
        "cosine_vs_teacher": cos_b0,
        "mse_vs_teacher": mse_b0,
        "mean_cosine_vs_teacher": gate_res["mean_cosine_vs_teacher"],
        "min_cosine_vs_teacher": gate_res["min_cosine_vs_teacher"],
        "max_cosine_vs_teacher": gate_res["max_cosine_vs_teacher"],
        "cosine_std": gate_res["cosine_std"],
        "max_cosine_drop": gate_res["max_cosine_drop"],
        "mean_mse_vs_teacher": gate_res["mean_mse_vs_teacher"],
        "max_mse_vs_teacher": gate_res["max_mse_vs_teacher"],
        "per_batch_cosine": gate_res["per_batch_cosine"],
        "per_batch_mse": gate_res["per_batch_mse"],
        "all_finite": gate_res["all_finite"],
        "n_calib_batches": eff_calib_batches,
        "n_eval_batches": gate_res["n_eval_batches"],
        "batch_diversity": batch_diversity,
        "calibration_gate": {
            "passed": gate_res["passed"],
            "enforced": gate_enforced,
            "failures": gate_res["failures"],
        },
        "analytical_bpw": compute_analytical_bpw(module, set(qmeta["excluded_modules"]), bits_q),
        "compute_dtype": eff_compute_dtype,
        "excluded_modules": qmeta["excluded_modules"],
        "skipped_non_divisible": qmeta["skipped_non_divisible"],
    }

    hash_payload = json.dumps(
        {
            "model_name": model_name,
            "preset": preset,
            "backend": export_backend,
            "compute_dtype": eff_compute_dtype,
            "excluded": qmeta["excluded_modules"],
            "extra": dict(spec_metadata or {}),
        },
        sort_keys=True,
    ).encode("utf-8")
    model_hash = hashlib.sha256(hash_payload).hexdigest()[:16]
    probe = probe_core_ai_support()
    spec_options: dict[str, Any] = {
        "backend": export_backend,
        "preset": preset,
        "compute_dtype": eff_compute_dtype,
        **_sanitize_custom_metadata(dict(spec_metadata or {})),
    }
    if preset == "w4_per_block":
        spec_options["block_size"] = int(block_size)
    cache_parts = coreai_specialization_cache_key_parts(
        model_hash=model_hash,
        coreai_toolchain_ver=probe.get("coreai_torch_version") or "unknown",
        min_os="27.0" if export_backend == "coreai" else "15.0",
        arch="arm64",
        spec_options=spec_options,
    )
    cache_sha = hashlib.sha256(
        json.dumps(cache_parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    guard_dict: dict[str, Any] | None = None
    asset_summary: dict[str, Any] | None = None
    placement_summary: dict[str, Any] | None = None

    if export_backend == "coreai":
        import coreai_torch
        from coreai.authoring import OSVersion
        from coreai.runtime import AIModelAssetMetadata

        prog = (
            coreai_torch.TorchConverter()
            .add_pytorch_module(
                finalized,
                export_fn=lambda m: torch.export.export(m, args=sample).run_decompositions(
                    coreai_torch.get_decomp_table()
                ),
                input_names=list(input_names),
                output_names=list(output_names),
                entrypoint_name="main",
            )
            .to_coreai()
        )
        asset_path = out_path / f"{stem}.aimodel"
        if asset_path.exists():
            shutil.rmtree(asset_path)
        md = AIModelAssetMetadata()
        md.model_description = f"{model_name} {stem} ({preset})"
        md.set_custom("cache_key_sha256", cache_sha)
        md.set_custom("cache_key_parts", _sanitize_custom_metadata(cache_parts))
        md.set_custom("sensitivity", _sanitize_custom_metadata(sensitivity))
        asset = prog.save_asset(asset_path, metadata=md, minimum_os=OSVersion.v27)
        summ = asset.summary(include_statistics=True)
        if summ is not None:
            asset_summary = {
                "function_names": list(summ.function_names),
                "storage_types": [(str(k), int(v)) for k, v in summ.storage_types],
                "compute_types": [str(x) for x in summ.compute_types],
                "operation_distribution": [(str(k), int(v)) for k, v in summ.operation_distribution],
                "asset_valid": bool(asset.is_valid(asset_path)),
            }
    elif export_backend == "coreml":
        import coremltools as ct

        ep_ml = torch.export.export(finalized, sample).run_decompositions({})
        mlmodel = ct.convert(
            ep_ml,
            inputs=list(ml_inputs) if ml_inputs is not None else None,
            outputs=list(ml_outputs) if ml_outputs is not None else None,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
        asset_path = out_path / f"{stem}.mlpackage"
        if asset_path.exists():
            shutil.rmtree(asset_path)
        mlmodel.save(str(asset_path))
        rep: GuardReport = check_coreml_model(
            mlmodel,
            model_name=stem,
            flag_add_fp16_cast=flag_add_fp16_cast,
            warn_matmul_ane=True,
            check_block_size=block_size if preset == "w4_per_block" else None,
            block_axis=block_axis,
        )
        guard_dict = {
            "ok": bool(rep.ok),
            "error_count": len(rep.errors),
            "warning_count": len(rep.warnings),
            "findings": [
                {"severity": f.severity, "rule_id": f.rule_id, "symbol": f.symbol, "message": f.message}
                for f in rep.findings[:25]
            ],
        }
        if inspect_placement:
            placement_summary = inspect_coreml_bundle_placement(
                asset_path,
                target_device=target_device,
                model_name=stem,
            )
    else:
        raise ValueError(f"Unsupported export_backend={export_backend!r}; choose 'coreai' or 'coreml'")

    return CoreAiExportResult(
        backend=export_backend,
        preset=preset,
        asset_path=asset_path,
        cache_key_parts=cache_parts,
        cache_key_sha256=cache_sha,
        asset_summary=asset_summary,
        sensitivity_report=sensitivity,
        guard_report=guard_dict,
        placement_summary=placement_summary,
    )


def run_coreai_opt_preset_sweep(
    *,
    module_factory: Callable[[], Any],
    sample_inputs_factory: Callable[[int], tuple[Any, ...]],
    out_dir: str | Path,
    stem_prefix: str,
    model_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    ml_inputs: Sequence[Any] | None = None,
    ml_outputs: Sequence[Any] | None = None,
    export_backend: str = "coreml",
    presets: Sequence[str] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    block_axis: int = 1,
    excluded_modules: Sequence[str] = (),
    calib_batches: int = 2,
    eval_inputs: Sequence[tuple[Any, ...]] | Callable[[int], tuple[Any, ...]] | None = None,
    min_calib_batches: int = 1,
    require_calib_diversity: bool = False,
    strict_calibration_gate: bool = False,
    min_calib_cosine: float | None = None,
    max_calib_cosine_std: float | None = None,
    max_calib_cosine_drop: float | None = None,
    max_calib_mse: float | None = None,
    fail_on_nonfinite: bool = True,
    spec_metadata: Mapping[str, Any] | None = None,
    allow_experimental_coreml_w4_block: bool = False,
    compute_dtype: str = "auto",
) -> list[dict[str, Any]]:
    """Run a sensitivity & BPW calibration sweep across CoreAI-Opt presets."""
    out_root = Path(out_dir)
    if presets is None:
        presets = (
            ("fp16", "w8", "w4")
            if export_backend.strip().lower() == "coreml" and not allow_experimental_coreml_w4_block
            else ("fp16", "w8", "w4", "w4_per_block")
        )
    gate_kw: dict[str, Any] = {}
    if eval_inputs is not None:
        gate_kw["eval_inputs"] = eval_inputs
    if int(min_calib_batches) != 1:
        gate_kw["min_calib_batches"] = min_calib_batches
    if require_calib_diversity:
        gate_kw["require_calib_diversity"] = True
    if strict_calibration_gate:
        gate_kw["strict_calibration_gate"] = True
    if min_calib_cosine is not None:
        gate_kw["min_calib_cosine"] = min_calib_cosine
    if max_calib_cosine_std is not None:
        gate_kw["max_calib_cosine_std"] = max_calib_cosine_std
    if max_calib_cosine_drop is not None:
        gate_kw["max_calib_cosine_drop"] = max_calib_cosine_drop
    if max_calib_mse is not None:
        gate_kw["max_calib_mse"] = max_calib_mse
    if not fail_on_nonfinite:
        gate_kw["fail_on_nonfinite"] = False
    rows: list[dict[str, Any]] = []
    for preset in presets:
        res = run_coreai_opt_calibration_and_export(
            module_factory=module_factory,
            sample_inputs_factory=sample_inputs_factory,
            out_dir=out_root / preset,
            stem=f"{stem_prefix}_{preset}",
            model_name=model_name,
            input_names=input_names,
            output_names=output_names,
            ml_inputs=ml_inputs,
            ml_outputs=ml_outputs,
            export_backend=export_backend,
            preset=preset,
            block_size=block_size,
            block_axis=block_axis,
            excluded_modules=excluded_modules,
            calib_batches=calib_batches,
            spec_metadata=spec_metadata,
            allow_experimental_coreml_w4_block=allow_experimental_coreml_w4_block,
            compute_dtype=compute_dtype,
            **gate_kw,
        )
        rows.append(
            {
                "preset": preset,
                "backend": res.backend,
                "asset_path": str(res.asset_path),
                "cache_key_sha256": res.cache_key_sha256,
                **(res.sensitivity_report or {}),
                "guard_ok": res.guard_report["ok"] if res.guard_report is not None else None,
            }
        )
    return rows
