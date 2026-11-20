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
import json
from pathlib import Path
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

_PRESET_NOMINAL_BITS: dict[str, float] = {
    "fp16": 16.0,
    "w8": 8.0,
    "w4": 4.0,
    "w4_per_block": 4.0,
    "fp4_fp8": 8.0,
}


def _sanitize_custom_metadata(value: Any) -> Any:
    """Strip None values recursively because AIModelAssetMetadata.set_custom rejects NoneType."""
    if isinstance(value, Path):
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
    allow_experimental_coreml_w4_block: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build a Pattern-7 & export-backend guarded `coreai_opt.quantization.QuantizerConfig`."""
    import torch
    from coreai_opt import ExportBackend
    from coreai_opt.quantization import (
        ModuleQuantizerConfig,
        QuantizationSpec,
        QuantizerConfig,
    )

    preset_norm = preset.strip().lower()
    if preset_norm not in VALID_COREAI_PRESETS:
        raise ValueError(f"Unsupported preset={preset!r}; choose from {VALID_COREAI_PRESETS}")

    backend_norm = export_backend.strip().lower()
    if backend_norm == "coreml":
        eb = ExportBackend.CoreML
    elif backend_norm == "coreai":
        eb = ExportBackend.CoreAI
    elif backend_norm == "torch":
        eb = ExportBackend.Torch
    else:
        raise ValueError(f"Unsupported export_backend={export_backend!r}")

    probe = probe_core_ai_support(
        hw={"system": "Darwin", "machine": "arm64", "macos_version": os_version} if os_version else None
    )
    if preset_norm == "w4_per_block" and eb == ExportBackend.CoreML and not allow_experimental_coreml_w4_block:
        raise ValueError(
            "preset='w4_per_block' on export_backend='coreml' produces corrupted CoreML logits and CPU "
            "fallback on ios18.conv (use preset='w8' for CoreML, preset='w4_per_block' with "
            "export_backend='coreai', or pass allow_experimental_coreml_w4_block=True)"
        )
    if preset_norm == "fp4_fp8" and eb == ExportBackend.CoreML:
        raise ValueError("preset='fp4_fp8' requires export_backend='coreai' (CoreML target does not support FP4/FP8)")
    if preset_norm == "fp4_fp8" and os_version is not None and not probe["fp4_fp8_preset_eligible"]:
        raise ValueError(f"preset='fp4_fp8' requires macOS 27.2+, detected {os_version!r}")

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

    eff_excluded = set(excluded_modules)
    skipped_non_divisible: dict[str, tuple[int, int]] = {}
    if preset_norm == "w4_per_block" and weight_shapes is not None:
        candidate_shapes = {k: shp for k, shp in weight_shapes.items() if k not in eff_excluded}
        bad = check_divisibility(candidate_shapes, axis=block_axis, block_size=block_size, min_ndim=2)
        if bad:
            skipped_non_divisible = bad
            eff_excluded.update(bad.keys())

    if preset_norm == "fp16":
        cfg = QuantizerConfig()
    elif preset_norm == "w8":
        cfg = QuantizerConfig.presets.w8()
    elif preset_norm == "w4":
        cfg = QuantizerConfig.presets.w4()
    elif preset_norm == "w4_per_block":
        cfg = QuantizerConfig.presets.w4_per_block(block_size=int(block_size), axis=block_axis)
    else:
        cfg = QuantizerConfig().set_global(
            ModuleQuantizerConfig(
                op_state_spec={"weight": QuantizationSpec(dtype=torch.float8_e4m3fn)},
                op_input_spec={"*": None},
                op_output_spec={"*": None},
            )
        )

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
    }
    return cfg, meta


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
    spec_metadata: Mapping[str, Any] | None = None,
    flag_add_fp16_cast: bool = False,
    inspect_placement: bool = False,
    target_device: str = "ane",
    allow_experimental_coreml_w4_block: bool = False,
    compute_dtype: str = "auto",
) -> CoreAiExportResult:
    """Calibrate, quantize (CoreAI-Opt), export (`.aimodel` / `.mlpackage`), and guard any `nn.Module`."""
    import torch

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    eff_compute_dtype = _resolve_coreai_compute_dtype(export_backend, preset, compute_dtype)
    module = module_factory().eval()
    raw_sample = sample_inputs_factory(0)

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
        allow_experimental_coreml_w4_block=allow_experimental_coreml_w4_block,
    )

    with torch.no_grad():
        ref_vec = _flat_outputs(module(*raw_sample))

    if eff_compute_dtype == "fp16":
        # Cast floating weights/buffers + sample inputs to FP16 so Core AI exports stay ANE/FP16-eligible.
        module = module.to(dtype=torch.float16)
        sample = _cast_floating_tensors(raw_sample, torch.float16)

        def _calib_inputs(bi: int) -> tuple[Any, ...]:
            return _cast_floating_tensors(sample_inputs_factory(bi), torch.float16)
    else:
        sample = raw_sample
        _calib_inputs = sample_inputs_factory

    if preset == "fp16":
        finalized = module
        if eff_compute_dtype == "fp16":
            with torch.no_grad():
                q_vec = _flat_outputs(finalized(*sample))
        else:
            q_vec = ref_vec
    else:
        from coreai_opt import ExportBackend
        from coreai_opt.quantization import Quantizer

        eb = ExportBackend.CoreML if export_backend == "coreml" else ExportBackend.CoreAI
        quantizer = Quantizer(module, qcfg)
        prepared = quantizer.prepare(sample)
        with quantizer.calibration_mode():
            with torch.no_grad():
                for bi in range(max(1, int(calib_batches))):
                    prepared(*_calib_inputs(bi))
        with torch.no_grad():
            q_vec = _flat_outputs(prepared(*sample))
        finalized = quantizer.finalize(backend=eb)

    denom = float(np.linalg.norm(ref_vec) * np.linalg.norm(q_vec))
    cos = float(np.dot(ref_vec, q_vec) / denom) if denom > 0.0 else 0.0
    mse = float(np.mean((ref_vec - q_vec) ** 2)) if ref_vec.size else 0.0
    bits_q = _PRESET_NOMINAL_BITS[preset.strip().lower()]
    sensitivity = {
        "cosine_vs_teacher": round(cos, 6),
        "mse_vs_teacher": round(mse, 8),
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
