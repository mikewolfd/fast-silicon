#!/usr/bin/env python3
"""External Objective Outcome Benchmark Suite CLI for ``fast-silicon``.

Subcommands:
  - ``setup-data``: Stage & verify immutable HF dataset shards into local cache (0 GPU work).
  - ``run``: Execute Tier-1 / Tier-2 objective outcome benchmarks in isolated child subprocesses
    serialized under ``/tmp/m4_gpu_validation.lock``, capturing ``FourBucketMemoryProbe``,
    ``DeviceMemoryGuard``, Effective Outcome Throughput (``EOT``), and optional ``ABBA`` comparison.
  - ``compare``: Offline Two-Gate + Memory Ceiling promotion evaluation between two JSON reports.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import urllib.request

import numpy as np

from fast_shared.bench import (
    DeviceMemoryGuard,
    EXIT_SKIP,
    FourBucketMemoryProbe,
    check_system_quiescence,
    collect_env_telemetry,
    run_isolated_json_worker,
    wired_limit,
    write_worker_json,
)
from fast_shared.embeddings import accepts_kwarg
from fast_shared.outcome_eval import (
    DEFAULT_LOSSLESS_TOLERANCE,
    CLINC150_DOMAINS,
    CLINC_STAGE1_DOMAINS,
    build_banking77_flat_qdef,
    build_bfcl_response_schema,
    build_clinc_stage1_qdef,
    build_clinc_stage2_qdef,
    build_ragtruth_noul_qdef,
    build_ragtruth_state,
    build_rewardbench2_score_qdef,
    check_embedding_validity,
    check_probability_validity,
    chunk_document_char_spans,
    compute_contextual_lift,
    compute_effective_item_scores,
    compute_outcome_metrics,
    evaluate_bfcl_v3,
    evaluate_clinc_two_stage,
    evaluate_cord_v2,
    evaluate_doclaynet_gov,
    evaluate_fintabnet_otsl,
    evaluate_instruction_following,
    evaluate_jsonschemabench_compiler,
    evaluate_legalbench_spans,
    evaluate_livebench_reasoning,
    evaluate_math_500,
    evaluate_memory_gate,
    evaluate_mmlu_redux,
    evaluate_mrl_dtype_grid,
    evaluate_multiclass_choice,
    evaluate_olmocr_bench,
    evaluate_omnidocbench,
    evaluate_promotion_gates,
    evaluate_puredocbench,
    evaluate_ragtruth_noul,
    evaluate_retrieval_rankings,
    retrieval_coverage_errors,
    evaluate_reward_bench_2,
    evaluate_structured_extraction,
    evaluate_us_gov_ocr,
    evaluate_vrdu,
    load_banking77,
    load_bfcl_v3,
    load_clinc_oos,
    load_cord_v2,
    load_doclaynet_gov,
    load_fintabnet_otsl,
    load_ifbench_test,
    load_ifeval,
    load_jsonschemabench,
    load_legalbench_rag,
    load_liquidai_extraction,
    load_livebench_reasoning,
    load_lockfile,
    load_longembed_subset,
    load_math_500,
    load_mmlu_redux,
    load_nanobeir_subset,
    load_olmocr_bench,
    load_omnidocbench_en,
    load_puredocbench_legal_gov,
    load_ragtruth,
    load_reward_bench_2,
    load_us_gov_ocr,
    load_vrdu,
    partition_banking77_clusters,
    resolve_cache_dir,
    stage_tier_datasets,
    stage_us_gov_ocr,
    stratified_sha256_sample,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"
_WORKER_MARKER = "__OUTCOME_WORKER_JSON__="
_ARTIFACT_SUBDIRS = ("weights", "ane_models", "kev_08b", "laya_typed", "models", "tokenizer", "vocab")


@dataclass(frozen=True)
class OutcomePackageSpec:
    name: str
    category: int
    home_env_var: str
    description: str


PACKAGE_SPECS: tuple[OutcomePackageSpec, ...] = (
    OutcomePackageSpec("fast-bge-small", 1, "FAST_BGE_SMALL_HOME", "33M BERT bi-encoder (Cat 1)"),
    OutcomePackageSpec("fast-embeddinggemma-2", 1, "FAST_EMBEDDINGGEMMA2_HOME", "308M MRL encoder <=2K (Cat 1)"),
    OutcomePackageSpec("fast-pplx", 1, "FAST_PPLX_HOME", "0.6B 32K contextual encoder (Cat 1)"),
    OutcomePackageSpec("fast-decision-models", 2, "FAST_DECISION_MODELS_HOME", "Laya-ANE & Kev-0.8B decisions (Cat 2)"),
    OutcomePackageSpec("fast-gemma4", 4, "FAST_GEMMA4_HOME", "Gemma-4 chat/IF/reasoning & structured JSON (Cat 3+4)"),
    OutcomePackageSpec("fast-qwen3", 4, "FAST_QWEN3_HOME", "Qwen3.5/3.6 hybrid GDN+GQA chat & structured JSON (Cat 3+4)"),
    OutcomePackageSpec("fast-gpt-oss", 4, "FAST_GPT_OSS_HOME", "GPT-OSS-20B chat/reasoning & structured JSON (Cat 3+4)"),
    OutcomePackageSpec("fast-ocr", 5, "FAST_OCR_HOME", "Apple Vision ANE + FastOvisOCREngine / NuExtract3 OCR (Cat 5)"),
)
PUBLIC_BASELINE_SPECS: tuple[OutcomePackageSpec, ...] = (
    OutcomePackageSpec("public-bge-small", 1, "FAST_BGE_SMALL_HOME", "Stock mlx-embeddings BGE-small 8-bit (Cat 1)"),
    OutcomePackageSpec("public-embeddinggemma-2", 1, "FAST_EMBEDDINGGEMMA2_HOME", "Stock mlx-vlm EmbeddingGemma-2 8-bit (Cat 1)"),
    OutcomePackageSpec("coreai-embeddinggemma-2", 1, "FAST_EMBEDDINGGEMMA2_HOME", "Apple CoreAI EmbeddingGemma-2 BF16 (Cat 1)"),
    OutcomePackageSpec("public-pplx", 1, "FAST_PPLX_HOME", "Stock pplx-context-mlx zainmerchan FP16 (Cat 1)"),
    OutcomePackageSpec("public-laya", 2, "FAST_DECISION_MODELS_HOME", "Stock mlx-embeddings Laya ModernBERT 8-bit (Cat 2)"),
    OutcomePackageSpec("coreai-laya", 2, "FAST_DECISION_MODELS_HOME", "Apple CoreAI Laya Typed Decisions (Cat 2)"),
    OutcomePackageSpec("public-kev-0.8b", 2, "FAST_DECISION_MODELS_HOME", "Stock mlx-lm Kev-0.8B 8-bit (Cat 2)"),
    OutcomePackageSpec("public-gemma4", 4, "FAST_GEMMA4_HOME", "Stock mlx-vlm Gemma-4 E2B 4-bit (Cat 3+4)"),
    OutcomePackageSpec("public-qwen3.5", 4, "FAST_QWEN3_HOME", "Stock mlx-lm Qwen3.5-4B OptiQ 4-bit (Cat 3+4)"),
    OutcomePackageSpec("public-gpt-oss", 4, "FAST_GPT_OSS_HOME", "Stock mlx-lm GPT-OSS-20B MXFP4 (Cat 3+4)"),
    OutcomePackageSpec("coreai-qwen3.5-4b", 4, "FAST_QWEN3_HOME", "Apple CoreAI Qwen3.5-4B 4bit/fp4_fp8 (Cat 3+4)"),
    OutcomePackageSpec("coreai-gemma4", 4, "FAST_GEMMA4_HOME", "Apple CoreAI Gemma-4 E2B QAT (Cat 3+4)"),
    OutcomePackageSpec("public-ovis-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm OvisOCR2-8bit full-canvas AR (Cat 5)"),
    OutcomePackageSpec("public-apple-vision", 5, "FAST_OCR_HOME", "Standalone Apple Vision ANE OCR (Cat 5)"),
    OutcomePackageSpec("public-pypdfium2", 5, "FAST_OCR_HOME", "Native pypdfium2 PDF text extraction (Cat 5)"),
    OutcomePackageSpec("public-glm-ocr-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm GLM-OCR BF16 (Cat 5)"),
    OutcomePackageSpec("public-lightonocr2-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm LightOnOCR-2-1B BF16 (Cat 5)"),
    OutcomePackageSpec("public-nuextract3-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm NuExtract3 visual JSON KIE (Cat 5)"),
    OutcomePackageSpec("public-deepseek-ocr2-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm DeepSeek-OCR-2 4-bit (Cat 5)"),
    OutcomePackageSpec("public-paddleocr-vl-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm PaddleOCR-VL 8-bit (Cat 5)"),
    OutcomePackageSpec("public-nemotron-parse-mlx-vlm", 5, "FAST_OCR_HOME", "Stock mlx-vlm NVIDIA Nemotron-Parse 8-bit (Cat 5)"),
    OutcomePackageSpec("public-rapidocr", 5, "FAST_OCR_HOME", "RapidOCR PP-OCRv4/v5 ONNX CPU pipeline (Cat 5)"),
    OutcomePackageSpec("public-tesseract", 5, "FAST_OCR_HOME", "Tesseract CLI CPU OCR baseline (Cat 5)"),
    OutcomePackageSpec("public-docling", 5, "FAST_OCR_HOME", "IBM Docling DocumentConverter pipeline (Cat 5)"),
    OutcomePackageSpec("public-marker", 5, "FAST_OCR_HOME", "Marker + Surya PDF/image Markdown pipeline (Cat 5)"),
)
COREML_BASELINE_SPECS: tuple[OutcomePackageSpec, ...] = (
    OutcomePackageSpec("coreml-bge-small", 1, "FAST_BGE_SMALL_HOME", "Open HF CoreML FP16 BGE-small ANE (sidekickd, Cat 1)"),
    OutcomePackageSpec("coreml-embeddinggemma-2-fp16", 1, "FAST_EMBEDDINGGEMMA2_HOME", "Open HF CoreML FP16 EmbeddingGemma-2 ANE + pack_256 (FluidInference, Cat 1)"),
    OutcomePackageSpec("coreml-embeddinggemma-2-int8", 1, "FAST_EMBEDDINGGEMMA2_HOME", "Open HF CoreML INT8 EmbeddingGemma-2 ANE + pack_256 (OpenInfinity, Cat 1)"),
    OutcomePackageSpec("coreml-laya", 2, "FAST_DECISION_MODELS_HOME", "Open HF CoreML ANE Laya-Typed-421M L192/L384/L512 (NMTZ, Cat 2)"),
    OutcomePackageSpec("coreml-kev-0.8b", 2, "FAST_DECISION_MODELS_HOME", "Open HF CoreML FP16 Kev-0.8B Fused+Row GPU (FluidInference, Cat 2)"),
)
OCR_FAMILY_PACKAGES: tuple[str, ...] = (
    "fast-ocr",
    "public-ovis-mlx-vlm",
    "public-glm-ocr-mlx-vlm",
    "public-lightonocr2-mlx-vlm",
    "public-deepseek-ocr2-mlx-vlm",
    "public-paddleocr-vl-mlx-vlm",
    "public-nemotron-parse-mlx-vlm",
    "public-nuextract3-mlx-vlm",
    "public-apple-vision",
    "public-rapidocr",
    "public-tesseract",
    "public-docling",
    "public-marker",
    "public-pypdfium2",
)
VISION_BENCHMARK_SPECS: tuple[OutcomePackageSpec, ...] = (
    OutcomePackageSpec(
        "fast-gemma4-vision",
        5,
        "FAST_GEMMA4_HOME",
        "Gemma-4 lazy multimodal VLM OCR & visual JSON KIE (Cat 5)",
    ),
    OutcomePackageSpec(
        "fast-embeddinggemma-2-vision",
        5,
        "FAST_EMBEDDINGGEMMA2_HOME",
        "EmbeddingGemma-2 lazy cross-modal page-image retrieval (Cat 5)",
    ),
    OutcomePackageSpec(
        "fast-pplx-vision",
        5,
        "FAST_PPLX_HOME",
        "PPLX Qwen3.5-ViT multi-band MaxSim page-image retrieval (Cat 5)",
    ),
)
VISION_BENCHMARK_PACKAGES: tuple[str, ...] = tuple(s.name for s in VISION_BENCHMARK_SPECS)
PUBLIC_FOR_FAST_PACKAGE: dict[str, tuple[str, ...]] = {
    "fast-bge-small": ("public-bge-small", "coreml-bge-small"),
    "fast-embeddinggemma-2": (
        "public-embeddinggemma-2", "coreai-embeddinggemma-2",
        "coreml-embeddinggemma-2-fp16", "coreml-embeddinggemma-2-int8",
    ),
    "fast-pplx": ("public-pplx",),
    "fast-decision-models": ("public-laya", "coreai-laya", "public-kev-0.8b", "coreml-laya", "coreml-kev-0.8b"),
    "fast-gemma4": ("public-gemma4", "coreai-gemma4"),
    "fast-qwen3": ("public-qwen3.5", "coreai-qwen3.5-4b"),
    "fast-gpt-oss": ("public-gpt-oss",),
    "fast-ocr": ("public-ovis-mlx-vlm", "public-apple-vision", "public-pypdfium2"),
    "fast-gemma4-vision": ("public-ovis-mlx-vlm", "public-apple-vision"),
}
_SPEC_BY_NAME: dict[str, OutcomePackageSpec] = {
    s.name: s for s in (*PACKAGE_SPECS, *PUBLIC_BASELINE_SPECS, *COREML_BASELINE_SPECS, *VISION_BENCHMARK_SPECS)
}
_CANONICAL_HOME_PKG: dict[str, str] = {
    "public-bge-small": "fast-bge-small",
    "coreml-bge-small": "fast-bge-small",
    "public-embeddinggemma-2": "fast-embeddinggemma-2",
    "coreai-embeddinggemma-2": "fast-embeddinggemma-2",
    "coreml-embeddinggemma-2-fp16": "fast-embeddinggemma-2",
    "coreml-embeddinggemma-2-int8": "fast-embeddinggemma-2",
    "public-pplx": "fast-pplx",
    "public-laya": "fast-decision-models",
    "coreai-laya": "fast-decision-models",
    "coreml-laya": "fast-decision-models",
    "public-kev-0.8b": "fast-decision-models",
    "coreml-kev-0.8b": "fast-decision-models",
    "public-gemma4": "fast-gemma4",
    "public-qwen3.5": "fast-qwen3",
    "public-gpt-oss": "fast-gpt-oss",
    "coreai-qwen3.5-4b": "fast-qwen3",
    "coreai-gemma4": "fast-gemma4",
    "fast-gemma4-vision": "fast-gemma4",
    "fast-embeddinggemma-2-vision": "fast-embeddinggemma-2",
    "fast-pplx-vision": "fast-pplx",
    "public-ovis-mlx-vlm": "fast-ocr",
    "public-apple-vision": "fast-ocr",
    "public-pypdfium2": "fast-ocr",
    "public-glm-ocr-mlx-vlm": "fast-ocr",
    "public-lightonocr2-mlx-vlm": "fast-ocr",
    "public-nuextract3-mlx-vlm": "fast-ocr",
    "public-deepseek-ocr2-mlx-vlm": "fast-ocr",
    "public-paddleocr-vl-mlx-vlm": "fast-ocr",
    "public-nemotron-parse-mlx-vlm": "fast-ocr",
    "public-rapidocr": "fast-ocr",
    "public-tesseract": "fast-ocr",
    "public-docling": "fast-ocr",
    "public-marker": "fast-ocr",
}


def _coreai_cache_dir() -> Path:
    return Path.home() / "Library" / "Caches" / "coreai-cache"


def _has_staged_artifacts(home_dir: Path) -> bool:
    art = home_dir / "artifacts"
    if not art.is_dir():
        return False
    # For fast-gemma4, weights/models/ane_models may live in sibling repo while monorepo only has vocab/.
    strong_dirs = ("weights", "ane_models", "kev_08b", "laya_typed", "models")
    if any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in strong_dirs):
        return True
    return any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in _ARTIFACT_SUBDIRS)


def _has_strong_artifacts(home_dir: Path) -> bool:
    art = home_dir / "artifacts"
    if not art.is_dir():
        return False
    strong_dirs = ("weights", "ane_models", "kev_08b", "laya_typed", "models")
    return any((art / sub).is_dir() and any((art / sub).iterdir()) for sub in strong_dirs)


def resolve_package_home(spec: OutcomePackageSpec, base_env: Mapping[str, str] | None = None) -> Path:
    """Resolve effective home dir for ``spec``, preferring explicit env or staged weights/models."""
    env = os.environ if base_env is None else base_env
    explicit = env.get(spec.home_env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    base_pkg = _CANONICAL_HOME_PKG.get(spec.name, spec.name)
    mono_home = _REPO_ROOT / "packages" / base_pkg
    sibling_home = _REPO_ROOT.parent / base_pkg
    if not _has_strong_artifacts(mono_home) and _has_strong_artifacts(sibling_home):
        return sibling_home
    if not _has_staged_artifacts(mono_home) and _has_staged_artifacts(sibling_home):
        return sibling_home
    return mono_home


@contextmanager
def _hw_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    import fcntl

    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev = os.environ.get("M4_GPU_LOCK_HELD")
        os.environ["M4_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("M4_GPU_LOCK_HELD", None)
            else:
                os.environ["M4_GPU_LOCK_HELD"] = prev
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _mx_sync() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "synchronize"):
            mx.synchronize()
    except Exception:
        pass


def _mx_peak_gb() -> float:
    try:
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
            return round(float(mx.get_peak_memory()) / 1e9, 4)
    except Exception:
        pass
    return 0.0


def _mx_reset_peak() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    except Exception:
        pass


def _memory_4bucket_dict(probe: FourBucketMemoryProbe) -> dict[str, float]:
    snap = probe.snapshot(evict_mapped=False)
    fp = snap["phys_footprint"]
    vm = snap["task_vm_info"]
    mf = snap["mapped_files"]
    return {
        "mlx_peak_gb": _mx_peak_gb(),
        "phys_footprint_gb": round(float(fp["phys_footprint_gb"]), 4),
        "peak_footprint_gb": round(float(fp["peak_footprint_gb"]), 4),
        "model_mapped_gb": round(float(mf["model_bytes"]) / 1e9, 4),
        "neural_nofootprint_gb": round(max(0, int(vm["ledger_tag_neural_nofootprint_bytes"])) / 1e9, 4),
        "combined_with_model_mapped_gb": round(float(snap["combined_with_model_mapped_gb"]), 4),
    }


def _merge_daemon_telemetry(
    mem4: dict[str, Any],
    diag: dict[str, Any],
    telemetry: Mapping[str, Any] | None,
    *,
    fallback_peak_gb: float | None = None,
) -> None:
    """Merge subprocess daemon 4-bucket `RUSAGE_INFO_V6`, asset summary, and cold/warm latency."""
    mib_to_gb = (1024.0 * 1024.0) / 1e9
    four = telemetry.get("four_bucket_metrics") if telemetry else None
    has_four = isinstance(four, Mapping) and bool(four)

    ext_pk: float | None = None
    if telemetry and telemetry.get("external_peak_gb") is not None and float(telemetry["external_peak_gb"]) > 0:
        ext_pk = float(telemetry["external_peak_gb"])
    elif telemetry and telemetry.get("peak_gb") is not None and float(telemetry["peak_gb"]) > 0:
        ext_pk = float(telemetry["peak_gb"])
    elif fallback_peak_gb is not None and float(fallback_peak_gb) > 0:
        ext_pk = float(fallback_peak_gb)

    if has_four:
        phys_mb = four.get("phys_footprint_mb")
        if phys_mb is not None and float(phys_mb) > 0:
            mem4["phys_footprint_gb"] = round(
                max(float(mem4.get("phys_footprint_gb", 0.0)), float(phys_mb) * mib_to_gb), 4
            )
        max_phys_mb = four.get("lifetime_max_phys_footprint_mb")
        if max_phys_mb is not None and float(max_phys_mb) > 0:
            mem4["peak_footprint_gb"] = round(
                max(float(mem4.get("peak_footprint_gb", 0.0)), float(max_phys_mb) * mib_to_gb), 4
            )
        elif ext_pk is not None:
            mem4["peak_footprint_gb"] = round(max(float(mem4.get("peak_footprint_gb", 0.0)), ext_pk), 4)

        comb_gb = 0.0
        if four.get("combined_with_model_mapped_gb") is not None and float(four["combined_with_model_mapped_gb"]) > 0:
            comb_gb = float(four["combined_with_model_mapped_gb"])
        elif four.get("combined_with_model_mapped_mb") is not None and float(four["combined_with_model_mapped_mb"]) > 0:
            comb_gb = float(four["combined_with_model_mapped_mb"]) * mib_to_gb
        elif ext_pk is not None:
            comb_gb = ext_pk
        if comb_gb > 0:
            mem4["combined_with_model_mapped_gb"] = round(
                max(float(mem4.get("combined_with_model_mapped_gb", 0.0)), comb_gb), 4
            )

        neural_gb = max(0.0, float(four.get("neural_nofootprint_mb", 0.0) or 0.0)) * mib_to_gb
        mapped_gb = max(0.0, float(four.get("mapped_model_mb", 0.0) or 0.0)) * mib_to_gb
        if neural_gb > 0:
            mem4["neural_nofootprint_gb"] = round(max(float(mem4.get("neural_nofootprint_gb", 0.0)), neural_gb), 4)
        if mapped_gb > 0:
            mem4["model_mapped_gb"] = round(max(float(mem4.get("model_mapped_gb", 0.0)), mapped_gb), 4)
        if "ane_daemon_footprint_delta_mb" in four:
            mem4["ane_daemon_footprint_delta_mb"] = round(
                float(four.get("ane_daemon_footprint_delta_mb", 0.0) or 0.0), 2
            )
        diag["daemon_four_bucket_metrics"] = dict(four)
    elif ext_pk is not None:
        mem4["peak_footprint_gb"] = round(max(float(mem4.get("peak_footprint_gb", 0.0)), ext_pk), 4)
        mem4["combined_with_model_mapped_gb"] = round(
            max(float(mem4.get("combined_with_model_mapped_gb", 0.0)), ext_pk), 4
        )

    if not telemetry:
        return

    lat = telemetry.get("cold_warm_latency")
    if isinstance(lat, Mapping) and lat:
        diag["cold_warm_latency"] = dict(lat)
    if telemetry.get("compile_sec") is not None:
        diag["compile_sec"] = round(float(telemetry["compile_sec"]), 4)
    if telemetry.get("external_peak_gb") is not None:
        diag["external_peak_gb"] = round(float(telemetry["external_peak_gb"]), 4)
    if telemetry.get("max_rss_gb") is not None:
        diag["max_rss_gb"] = round(float(telemetry["max_rss_gb"]), 4)
    if isinstance(telemetry.get("coreml_info"), Mapping):
        diag["coreml_info"] = dict(telemetry["coreml_info"])
    if isinstance(telemetry.get("asset_summary"), Mapping):
        diag["asset_summary"] = dict(telemetry["asset_summary"])
    if "profiler_available" in telemetry:
        diag["profiler_available"] = bool(telemetry["profiler_available"])


def _host_mem_gb(env_meta: Mapping[str, Any]) -> float:
    val = env_meta.get("mem_gb")
    return float(val) if val else 0.0


def _build_record(
    *,
    package: str,
    category: int,
    config_label: str,
    benchmark: str,
    valid_flags: Sequence[bool],
    item_scores: Sequence[float],
    wall_sec: float,
    diagnostics: Mapping[str, Any],
    memory_4bucket_gb: Mapping[str, float],
    sub_benchmarks: Mapping[str, Any],
    host_mem_gb: float,
) -> dict[str, Any]:
    summary = compute_outcome_metrics(valid_flags, item_scores, max(wall_sec, 1e-6))
    rec = summary.to_dict()

    eff_vec = compute_effective_item_scores(valid_flags, item_scores)
    mem_ok, _, _, mem_reasons = evaluate_memory_gate(
        memory_4bucket_gb,
        host_mem_gb=host_mem_gb,
        device_guard_tripped=bool(diagnostics.get("device_guard_tripped", False)),
    )
    coverage_errors = retrieval_coverage_errors({"category": category, **rec, "sub_benchmarks": sub_benchmarks})
    if category == 1:
        rec["retrieval_coverage_status"] = "INCOMPLETE" if coverage_errors else "COMPLETE"
        query_groups = [(name, sub["ordered_query_ids_sha256"]) for name, sub in sub_benchmarks.items()
                        if "ordered_query_ids_sha256" in sub]
        if query_groups:
            encoded = json.dumps(query_groups, separators=(",", ":")).encode("utf-8")
            rec["retrieval_query_order_sha256"] = hashlib.sha256(encoded).hexdigest()
    reasons = [*mem_reasons, *coverage_errors]
    if rec["n_valid"] <= 0:
        reasons.append("n_valid == 0 (all outputs failed validity checks)")
    rep_loops = int(diagnostics.get("repeat_loop_count", 0) or 0)
    is_schema_constrained = int(category) == 4 or bool(diagnostics.get("llguidance", False))
    if rep_loops > 0 and is_schema_constrained:
        reasons.append(f"repeat_loop_count={rep_loops} > 0")
    verdict = "PASS" if (mem_ok and not reasons) else "FAIL"
    return {
        "package": package,
        "category": int(category),
        "config_label": config_label,
        "benchmark": benchmark,
        **rec,
        "se_paired_vs_baseline": None,
        "diagnostics": dict(diagnostics),
        "memory_4bucket_gb": dict(memory_4bucket_gb),
        "sub_benchmarks": dict(sub_benchmarks),
        "per_item_effective_scores": [round(float(x), 6) for x in eff_vec.tolist()],
        "gate_verdict": verdict,
        "gate_reasons": reasons,
        "status": "ok",
    }


def _parse_ocr_batch_size(raw: str) -> int | str:
    tok = str(raw).strip().lower()
    if tok == "auto":
        return "auto"
    try:
        val = int(tok)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--ocr-batch-size must be >= 1 or 'auto', got {raw!r}") from exc
    if val < 1:
        raise argparse.ArgumentTypeError(f"--ocr-batch-size must be >= 1 or 'auto', got {raw!r}")
    return val


def _parse_extra_flags(extra_args: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode", default=None)
    p.add_argument("--ane-profile", default=None)
    p.add_argument("--ane-variant", default="fp16")
    p.add_argument("--kv-i8", dest="kv_i8", action="store_true", default=None)
    p.add_argument("--no-kv-i8", action="store_true", default=False)
    p.add_argument("--kv-mode", choices=("auto", "bf16", "q8", "q4_quarot"), default=None)
    p.add_argument(
        "--pplx-doc-pooling",
        choices=("single", "maxsim_1024", "maxsim_512", "top2_512"),
        default="single",
    )
    p.add_argument("--model-dir", default=None)
    p.add_argument("--lm-head-mode", choices=("q4", "bf16_mpp"), default="q4")
    p.add_argument("--mlp-dtype", choices=("auto", "bfloat16", "quantized"), default="auto")
    p.add_argument("--mtp-drafter", action="store_true")
    p.add_argument("--mtp-drafter-path", default=None)
    p.add_argument("--backend", choices=("all", "laya-ane", "laya-hetero", "kev-0.8b"), default="all")
    p.add_argument("--kev-quant", choices=("auto", "w8", "bf16"), default="w8")
    p.add_argument("--batch-docs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--categories", default="all")
    p.add_argument("--no-llguidance", action="store_true")
    p.add_argument("--mlx-cache-limit-mb", type=int, help="Cap the isolated MLX worker allocation cache, in MiB")
    p.add_argument("--gemma-vocab-out", type=int, help="Limit the project Gemma output vocabulary; the default uses the checkpoint head")
    p.add_argument("--gemma-json-layout", choices=("fixed", "compact", "flexible"), default="fixed")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--coreai-variant", choices=("4bit", "fp4_fp8", "both"), default="both")
    p.add_argument("--max-coarse-tokens", type=int, default=None)
    # Category 5 (OCR) worker flags
    p.add_argument("--ocr-backend", choices=("auto", "ovis", "apple-vision"), default="auto")
    p.add_argument("--ocr-suites", default="default")
    p.add_argument("--table-mode", choices=("gfm", "html"), default="gfm")
    p.add_argument("--output-format", choices=("markdown", "raw", "text"), default="markdown")
    p.add_argument("--ocr-model", default=None)
    p.add_argument("--ocr-render-only", action="store_true", default=False)
    p.add_argument("--ocr-dpi", type=int, default=200)
    p.add_argument("--triage-dpi", type=int, default=None)
    p.add_argument("--ocr-batch-size", type=_parse_ocr_batch_size, default="auto")
    p.add_argument("--ocr-kv-i8", default=None)
    p.add_argument("--ocr-gdn-state-mode", choices=("auto", "fp32", "bf16", "int8_blk32"), default=None)
    p.add_argument("--drop-white-patches", action="store_true", default=False)
    p.add_argument("--no-latin-only", action="store_true", default=False)
    p.add_argument("--speculative", choices=("auto", "off", "self", "vision+self"), default="auto")
    p.add_argument("--no-crop-margins", action="store_true", default=False)
    p.add_argument("--no-auto-orient", action="store_true", default=False)
    p.add_argument("--no-adaptive-resolution", action="store_true", default=False)
    p.add_argument("--deskew", nargs="?", const="on", default=None, choices=("auto", "on", "off"))
    p.add_argument("--no-deskew", action="store_const", dest="deskew", const="off")
    p.add_argument("--region-routing", action="store_true", default=False)
    p.add_argument("--zoom-recover", action="store_true", default=False)
    p.add_argument("--strict-markdown-grammar", action="store_true", default=False)
    p.add_argument("--ocr-max-tokens", type=int, default=4096)
    p.add_argument("--ocr-vocab-size", type=int, default=None)
    p.add_argument("--puredocbench-track", choices=("clean", "digital", "real"), default="clean")
    p.add_argument("--puredocbench-images-root", default=None)
    p.add_argument("--puredocbench-en-root", default=None)
    # Multimodal vision & cross-modal page-image retrieval flags
    p.add_argument("--max-soft-tokens", type=int, default=None)
    p.add_argument("--vision-model-dir", default=None)
    p.add_argument("--visual-retrieval-suites", default="default")
    p.add_argument("--visual-index-mode", choices=("fp16", "int8", "binary_int8", "auto"), default="fp16")
    p.add_argument("--visual-chunks", type=int, default=8)
    p.add_argument("--no-visual-crop-margins", action="store_true", default=False)
    p.add_argument("--grid-bucket-step", type=int, default=None)
    ns, _ = p.parse_known_args(list(extra_args))
    return ns


# ---------------------------------------------------------------------------
# Category 1 Workers: fast-bge-small, fast-embeddinggemma-2, fast-pplx
# ---------------------------------------------------------------------------


def _split_judged_query_ids(
    query_ids: Sequence[str], qrels: Mapping[str, Mapping[str, float]]
) -> tuple[list[str], list[str]]:
    """Split retrieval query IDs into `(judged_ids, unjudged_ids)` matching MTEB positive-qrel filtering."""
    judged: list[str] = []
    unjudged: list[str] = []
    for qid in query_ids:
        rel_map = qrels.get(qid)
        if rel_map and any(float(sc) > 0.0 for sc in rel_map.values()):
            judged.append(qid)
        else:
            unjudged.append(qid)
    return judged, unjudged


def _retrieval_query_outcomes(
    intended_ids: Sequence[str], result: Mapping[str, Any], score_key: str,
) -> tuple[list[bool], list[float], dict[str, Any]]:
    """Keep benchmark coverage on the selected queries even when a ranking metric skips an unjudged query."""
    scored_rows = zip(result["per_query_valid"], result[score_key], strict=True)
    by_id = dict(zip(result["per_query_ids"], scored_rows, strict=True))
    if len(by_id) != len(result["per_query_ids"]) or len(set(intended_ids)) != len(intended_ids):
        raise ValueError("Duplicate retrieval query IDs")
    if by_id.keys() - set(intended_ids):
        raise ValueError("Scored query outside the intended retrieval workload")
    valid, scores = [], []
    for qid in intended_ids:
        ok, score = by_id.get(qid, (False, 0.0))
        usable = bool(ok and np.isfinite(score))
        valid.append(usable)
        scores.append(float(score) if usable else 0.0)
    id_bytes = json.dumps(list(intended_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    details = {"n_queries": len(intended_ids), "n_valid_queries": sum(valid),
               "ordered_query_ids_sha256": hashlib.sha256(id_bytes).hexdigest()}
    omitted = [qid for qid in intended_ids if qid not in by_id]
    if omitted:
        details["unscored_query_ids"] = omitted
    return valid, scores, details


def _run_cat1_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
    coreai_python: str | None = None,
    pplx_mlx_path: str | None = None,
    max_coarse_tokens: int | None = None,
) -> list[dict[str, Any]]:
    model_dirs: list[Path] = [home / "artifacts"]
    if pkg_name.startswith("public-"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import stock_hf_model_files

        model_dirs.extend(stock_hf_model_files(pkg_name))
    elif pkg_name.startswith("coreai-"):
        model_dirs.append(_coreai_cache_dir())
    probe = FourBucketMemoryProbe(model_directories=model_dirs)
    guard = DeviceMemoryGuard()
    guard_tripped = False

    ds_nb = lock["datasets"]["nanobeir_en"]
    subsets = list(ds_nb["tier1_subsets"] if tier == "tier1" else ds_nb["tier2_subsets"])
    if smoke:
        subsets = subsets[:1]

    lb_max_docs = 6 if smoke else (min(12, max_items) if max_items else None)
    lb_max_q = 12 if smoke else max_items

    nb_loaded = {
        sub: load_nanobeir_subset(sub, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
        for sub in subsets
    }
    skipped_unjudged_by_sub: dict[str, list[str]] = {}
    if smoke or max_items:
        q_cap = 12 if smoke else int(max_items or 50)
        d_cap = 120 if smoke else None
        for sub, pack in nb_loaded.items():
            judged_all, unjudged_all = _split_judged_query_ids(sorted(pack["queries"].keys()), pack["qrels"])
            if unjudged_all:
                skipped_unjudged_by_sub[sub] = unjudged_all
            q_ids_keep = judged_all[:q_cap]
            pack["queries"] = {qid: pack["queries"][qid] for qid in q_ids_keep}
            pack["qrels"] = {qid: pack["qrels"][qid] for qid in q_ids_keep if qid in pack["qrels"]}
            if d_cap is not None:
                rel_docs = {cid for mp in pack["qrels"].values() for cid in mp}
                extra_docs = [cid for cid in sorted(pack["corpus"].keys()) if cid not in rel_docs][: max(0, d_cap - len(rel_docs))]
                keep_docs = rel_docs | set(extra_docs)
                pack["corpus"] = {cid: pack["corpus"][cid] for cid in sorted(keep_docs) if cid in pack["corpus"]}

    lb_pack = load_legalbench_rag(
        tier=tier,
        max_docs=lb_max_docs,
        max_queries=lb_max_q,
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Chunk LegalBench-RAG documents deterministically (~256 tokens ~= 1024 chars).
    lb_doc_ids = sorted(lb_pack["corpus_texts"].keys())
    lb_chunks_by_doc = {
        did: chunk_document_char_spans(did, lb_pack["corpus_texts"][did], chunk_chars=1024, overlap_chars=0)
        for did in lb_doc_ids
    }
    lb_flat_chunks = [ch for did in lb_doc_ids for ch in lb_chunks_by_doc[did]]
    lb_queries = lb_pack["queries"]
    lb_q_texts = [q["query"] for q in lb_queries]
    lb_q_ids = [q["id"] for q in lb_queries]
    if not lb_queries or not lb_flat_chunks:
        raise ValueError(f"LegalBench-RAG selected {len(lb_queries)} queries and {len(lb_flat_chunks)} chunks; both are required")

    mode = extra.mode
    nb_pre_normalized = True
    _enc_coarse_2k = None
    coarse_max_tokens: int | None = None
    if pkg_name == "fast-bge-small":
        from fast_bge_small.engine import FastBgeSmall

        eff_mode = mode or "hetero"
        eng = FastBgeSmall(home=home, default_mode=eff_mode, warmup=True)
        mrl_dims: tuple[int | str, ...] = (128, 256, "max")
        int8_mode = "sym_int8"

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=True, add_query_prefix=True, mode=eff_mode)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=False, add_query_prefix=False, mode=eff_mode)

        _enc_q_nb = _enc_q
        _enc_d_nb = _enc_d
        _enc_ctx = None
    elif pkg_name == "fast-embeddinggemma-2":
        from fast_embeddinggemma_2.engine import FastEmbeddingGemma2

        eff_mode = mode or "hetero"
        eng = FastEmbeddingGemma2(home=home, default_mode=eff_mode, warmup=True)
        mrl_dims = (128, 256, 512, "max")
        int8_mode = "sym_int8"
        coarse_max_tokens = int(max_coarse_tokens) if max_coarse_tokens is not None else 2048

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=True, add_query_prefix=True, mode=eff_mode)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_texts(txts, is_query=False, add_document_prefix=True, mode=eff_mode)

        def _enc_coarse_2k(txts: list[str]) -> np.ndarray:
            # Exercise native 2K-token context (or matched --max-coarse-tokens override) via allow_gpu_overflow=True.
            tok_lists = eng.tokenizer.encode_batch(txts, add_document_prefix=True, max_length=coarse_max_tokens)
            return eng.encode_corpus(tok_lists, mode=eff_mode, allow_gpu_overflow=True)

        _enc_q_nb = _enc_q
        _enc_d_nb = _enc_d
        _enc_ctx = None
    elif pkg_name == "fast-pplx":
        from fast_pplx.engine import FastPplx

        eff_mode = mode or "hetero2"
        eng = FastPplx(home=home, mode=eff_mode, ane_profile=extra.ane_profile, local_files_only=True)
        mrl_dims = (128, 256, 512, "max")
        int8_mode = "native_tanh_int8"
        nb_pre_normalized = False

        def _enc_q_nb(txts: list[str]) -> np.ndarray:
            # Raw unnormalized projections required for exact native clip(rint(tanh(x)*127)) int8 parity.
            return eng.encode_queries(txts, mode=eff_mode, output_dtype="float32", normalize=False)

        def _enc_d_nb(txts: list[str]) -> np.ndarray:
            return eng.encode_documents(txts, mode=eff_mode, output_dtype="float32", normalize=False)

        def _enc_q(txts: list[str]) -> np.ndarray:
            return eng.encode_queries(txts, mode=eff_mode, output_dtype="float32", normalize=True)

        def _enc_d(txts: list[str]) -> np.ndarray:
            return eng.encode_documents(txts, mode=eff_mode, output_dtype="float32", normalize=True)

        ctx_max_len = 2048 if tier == "tier1" else 16384

        def _enc_ctx(doc_groups: list[list[str]]) -> list[np.ndarray]:
            from fast_shared.outcome_eval.retrieval import collect_pplx_window_vectors, plan_pplx_windows

            windows = plan_pplx_windows(doc_groups, ctx_max_len, lambda text: len(eng.tokenizer.encode(text)))
            matrices = eng.encode_contextual_chunks(
                [w.chunks for w in windows], max_length=ctx_max_len, mode=eff_mode, output_dtype="float32", normalize=True
            )
            return collect_pplx_window_vectors(windows, matrices, len(doc_groups))

    elif pkg_name in (
        "public-bge-small",
        "coreml-bge-small",
        "public-embeddinggemma-2",
        "coreai-embeddinggemma-2",
        "coreml-embeddinggemma-2-fp16",
        "coreml-embeddinggemma-2-int8",
        "public-pplx",
    ):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import build_public_cat1_adapter

        pub_ad = build_public_cat1_adapter(
            pkg_name,
            tier=tier,
            coreai_python=coreai_python,
            pplx_mlx_path=pplx_mlx_path,
            max_coarse_tokens=max_coarse_tokens,
        )
        eng = pub_ad
        eff_mode = pub_ad.harness_label
        mrl_dims = pub_ad.mrl_dims
        int8_mode = pub_ad.int8_mode
        nb_pre_normalized = pub_ad.nb_pre_normalized
        _enc_q_nb = pub_ad.enc_q_nb
        _enc_d_nb = pub_ad.enc_d_nb
        _enc_q = pub_ad.enc_q
        _enc_d = pub_ad.enc_d
        _enc_coarse_2k = pub_ad.enc_coarse_2k
        _enc_ctx = pub_ad.enc_ctx
        coarse_max_tokens = pub_ad.coarse_max_tokens
    else:
        raise ValueError(f"Unsupported Cat 1 package: {pkg_name}")

    try:
        guard.arm()
        # Untimed warmup batch
        _ = _enc_q(["warmup query"])
        _ = _enc_d(["warmup document passage"])
        _mx_sync()
        _mx_reset_peak()

        all_valid: list[bool] = []
        all_scores: list[float] = []
        sub_reports: dict[str, Any] = {}
        mrl_deltas: list[float] = []
        nb_vec_cache: dict[str, tuple[list[str], Any, list[str], Any, dict[str, Any]]] = {}

        t0 = time.perf_counter()

        # 1A: NanoBEIR-en primary retrieval encoding & scoring
        nb_ndcgs: list[float] = []
        for sub in subsets:
            pack = nb_loaded[sub]
            q_ids, unjudged_nb = _split_judged_query_ids(sorted(pack["queries"].keys()), pack["qrels"])
            if unjudged_nb:
                skipped_unjudged_by_sub[sub] = sorted(set(skipped_unjudged_by_sub.get(sub, [])) | set(unjudged_nb))
            d_ids = sorted(pack["corpus"].keys())
            q_vecs = _enc_q_nb([pack["queries"][qid] for qid in q_ids])
            d_vecs = _enc_d_nb([pack["corpus"][did] for did in d_ids])
            _mx_sync()
            _, ref = guard.check()
            if ref is not None:
                guard_tripped = True

            r_main = evaluate_retrieval_rankings(
                q_ids, q_vecs, d_ids, d_vecs, pack["qrels"], k=10, expect_unit_norm=nb_pre_normalized
            )
            nb_ndcgs.append(float(r_main["ndcg@10"]))
            valid, scores, coverage = _retrieval_query_outcomes(q_ids, r_main, "per_query_ndcg")
            all_valid.extend(valid)
            all_scores.extend(scores)
            sub_entry: dict[str, Any] = {
                **coverage,
                "n_docs": r_main["n_docs"],
                "ndcg@10": round(float(r_main["ndcg@10"]), 6),
                "mrr@10": round(float(r_main["mrr@10"]), 6),
                "recall@10": round(float(r_main["recall@10"]), 6),
            }
            if sub in skipped_unjudged_by_sub:
                sub_entry["skipped_unjudged_query_ids"] = skipped_unjudged_by_sub[sub]
            sub_reports[sub] = sub_entry
            nb_vec_cache[sub] = (q_ids, q_vecs, d_ids, d_vecs, pack)

        # 1B: LegalBench-RAG isolated primary encoding & scoring (325 core queries total on Tier-1)
        r_iso: dict[str, Any] | None = None
        lb_q_vecs: Any = None
        if lb_queries and lb_flat_chunks:
            lb_q_vecs = _enc_q(lb_q_texts)
            lb_iso_vecs = _enc_d([ch.text for ch in lb_flat_chunks])
            _mx_sync()
            r_iso = evaluate_legalbench_spans(lb_queries, lb_flat_chunks, lb_q_vecs, lb_iso_vecs, k=10)
            iso_valid, iso_scores, coverage = _retrieval_query_outcomes(lb_q_ids, r_iso, "per_query_span_ndcg")
            sub_reports["legalbench_rag_isolated"] = {
                **coverage,
                "n_chunks": len(lb_flat_chunks),
                "chunk_chars": 1024,
                "span_ndcg@10": round(float(r_iso["span_ndcg@10"]), 6),
                "char_recall@10": round(float(r_iso["char_recall@10"]), 6),
            }
            all_valid.extend(iso_valid)
            all_scores.extend(iso_scores)

        # 1C: Tier-2 LongEmbed (fast-pplx / public-pplx 4K-32K)
        if tier == "tier2" and pkg_name in ("fast-pplx", "public-pplx") and not smoke:
            le_cfg = lock["datasets"]["longembed"]["tier2"]
            doc_pooling = str(getattr(extra, "pplx_doc_pooling", "single") or "single")
            for le_sub in le_cfg["subsets"]:
                ctx_lens = le_cfg.get("passkey_context_lengths") if le_sub == "passkey" else None
                le_max_q = int(le_cfg.get(f"max_queries_{le_sub}", 200))
                if max_items is not None:
                    le_max_q = min(le_max_q, int(max_items))
                le_pack = load_longembed_subset(
                    le_sub,
                    context_lengths=ctx_lens,
                    max_queries=le_max_q,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                q_ids, unjudged_le = _split_judged_query_ids(sorted(le_pack["queries"].keys()), le_pack["qrels"])
                if unjudged_le:
                    skipped_unjudged_by_sub[f"longembed_{le_sub}"] = unjudged_le
                d_ids = sorted(le_pack["corpus"].keys())
                q_vecs = _enc_q([le_pack["queries"][qid] for qid in q_ids])
                if pkg_name == "fast-pplx" and doc_pooling != "single":
                    from fast_shared.outcome_eval.retrieval import check_embedding_validity, dcg_at_k
                    from fast_shared.retrieval_index import score_document_windows

                    win_toks = 1024 if doc_pooling == "maxsim_1024" else 512
                    agg_mode = "top2_mean" if doc_pooling == "top2_512" else "maxsim"
                    flat_wins, doc_offsets = eng.encode_documents_multivector(
                        [le_pack["corpus"][did] for did in d_ids],
                        window_tokens=win_toks,
                        context_tokens=2048,
                        repeat_header_tokens=128,
                        mode=eff_mode,
                        output_dtype="float32",
                        normalize=True,
                    )
                    _mx_sync()
                    win_sims = np.asarray(q_vecs, dtype=np.float64) @ np.asarray(flat_wins, dtype=np.float64).T
                    doc_sims = score_document_windows(win_sims, doc_offsets, agg=agg_mode)
                    q_ok = check_embedding_validity(q_vecs, expect_unit_norm=True)
                    d_ok = bool(np.all(check_embedding_validity(flat_wins, expect_unit_norm=True)))
                    avail_dids = set(d_ids)
                    eff_k = min(10, len(d_ids))
                    per_v, per_ndcg, per_mrr, per_rec, eval_qids = [], [], [], [], []
                    for qi, qid in enumerate(q_ids):
                        pos = {did: float(sc) for did, sc in (le_pack["qrels"].get(qid) or {}).items() if float(sc) > 0.0}
                        if not pos:
                            continue
                        eval_qids.append(qid)
                        row_sim = doc_sims[qi]
                        if not (bool(q_ok[qi]) and d_ok and pos.keys() <= avail_dids and bool(np.all(np.isfinite(row_sim)))):
                            per_v.append(False)
                            per_ndcg.append(0.0)
                            per_mrr.append(0.0)
                            per_rec.append(0.0)
                            continue
                        order = np.argsort(-row_sim, kind="mergesort")[:eff_k]
                        rels = [pos.get(d_ids[int(di)], 0.0) for di in order]
                        idcg = dcg_at_k(sorted(pos.values(), reverse=True), 10)
                        per_v.append(True)
                        per_ndcg.append((dcg_at_k(rels, 10) / idcg) if idcg > 0.0 else 0.0)
                        per_mrr.append(next((1.0 / float(ri + 1) for ri, rv in enumerate(rels) if rv > 0.0), 0.0))
                        per_rec.append(float(sum(1 for rv in rels if rv > 0.0)) / float(len(pos)))
                    r_le = {
                        "n_docs": len(d_ids),
                        "ndcg@10": float(np.mean(per_ndcg)) if per_ndcg else 0.0,
                        "mrr@10": float(np.mean(per_mrr)) if per_mrr else 0.0,
                        "recall@10": float(np.mean(per_rec)) if per_rec else 0.0,
                        "per_query_ids": eval_qids,
                        "per_query_valid": per_v,
                        "per_query_ndcg": per_ndcg,
                    }
                else:
                    d_vecs = _enc_d([le_pack["corpus"][did] for did in d_ids])
                    _mx_sync()
                    r_le = evaluate_retrieval_rankings(q_ids, q_vecs, d_ids, d_vecs, le_pack["qrels"], k=10)
                valid, scores, coverage = _retrieval_query_outcomes(q_ids, r_le, "per_query_ndcg")
                all_valid.extend(valid)
                all_scores.extend(scores)
                le_entry: dict[str, Any] = {
                    **coverage,
                    "n_docs": r_le["n_docs"],
                    "doc_pooling": doc_pooling if pkg_name == "fast-pplx" else "single",
                    "ndcg@10": round(float(r_le["ndcg@10"]), 6),
                    "mrr@10": round(float(r_le["mrr@10"]), 6),
                    "recall@10": round(float(r_le["recall@10"]), 6),
                }
                if unjudged_le:
                    le_entry["skipped_unjudged_query_ids"] = unjudged_le
                sub_reports[f"longembed_{le_sub}"] = le_entry

        _mx_sync()
        wall_sec = time.perf_counter() - t0

        # Auxiliary ablations (MRL/dtype sweep, contextual lift, coarse 2K lift) excluded from primary wall_sec & 325-item n_items
        t_abl0 = time.perf_counter()
        for sub, (q_ids, q_vecs, d_ids, d_vecs, pack) in nb_vec_cache.items():
            grid = evaluate_mrl_dtype_grid(
                q_ids,
                q_vecs,
                d_ids,
                d_vecs,
                pack["qrels"],
                dims=mrl_dims,
                dtypes=("f32", "f16", "int8"),
                int8_mode=int8_mode,
                pre_normalized=nb_pre_normalized,
                k=10,
            )
            if grid.get("mrl_128_int8_delta_ndcg@10") is not None:
                mrl_deltas.append(float(grid["mrl_128_int8_delta_ndcg@10"]))
            sub_reports[sub]["mrl_grid"] = grid

        contextual_lift = None
        coarse_2k_lift = None
        if r_iso is not None and lb_q_vecs is not None:
            if _enc_ctx is not None:
                doc_chunk_texts = [[ch.text for ch in lb_chunks_by_doc[did]] for did in lb_doc_ids]
                ctx_mats = _enc_ctx(doc_chunk_texts)
                _mx_sync()
                if [len(mat) for mat in ctx_mats] != [len(doc) for doc in doc_chunk_texts]:
                    raise ValueError("Contextual embedding count differs from the intended chunks per document")
                lb_ctx_vecs = np.vstack(ctx_mats) if ctx_mats else np.zeros((0, lb_q_vecs.shape[1]), dtype=np.float32)
                r_ctx = evaluate_legalbench_spans(lb_queries, lb_flat_chunks, lb_q_vecs, lb_ctx_vecs, k=10)
                contextual_lift = compute_contextual_lift(r_iso, r_ctx, k=10)
                _, _, coverage = _retrieval_query_outcomes(lb_q_ids, r_ctx, "per_query_span_ndcg")
                sub_reports["legalbench_rag_contextual"] = {
                    **coverage,
                    "span_ndcg@10": round(float(r_ctx["span_ndcg@10"]), 6),
                    "char_recall@10": round(float(r_ctx["char_recall@10"]), 6),
                    **contextual_lift,
                }
            elif _enc_coarse_2k is not None:
                lb_coarse_chunks = [
                    ch
                    for did in lb_doc_ids
                    for ch in chunk_document_char_spans(did, lb_pack["corpus_texts"][did], chunk_chars=6144, overlap_chars=0)
                ]
                lb_coarse_vecs = _enc_coarse_2k([ch.text for ch in lb_coarse_chunks])
                _mx_sync()
                r_coarse = evaluate_legalbench_spans(lb_queries, lb_coarse_chunks, lb_q_vecs, lb_coarse_vecs, k=10)
                coarse_2k_lift = compute_contextual_lift(r_iso, r_coarse, k=10)
                _, _, coverage = _retrieval_query_outcomes(lb_q_ids, r_coarse, "per_query_span_ndcg")
                sub_reports["legalbench_rag_coarse_2k"] = {
                    **coverage,
                    "n_chunks": len(lb_coarse_chunks),
                    "chunk_chars": 6144,
                    "max_tokens": coarse_max_tokens,
                    "span_ndcg@10": round(float(r_coarse["span_ndcg@10"]), 6),
                    "char_recall@10": round(float(r_coarse["char_recall@10"]), 6),
                    "coarse_2k_delta_ndcg@10": coarse_2k_lift["contextual_delta_ndcg@10"],
                    "se_paired": coarse_2k_lift["se_paired"],
                }
        _mx_sync()
        ablation_wall_sec = time.perf_counter() - t_abl0

        _, ref_end = guard.check()
        if ref_end is not None:
            guard_tripped = True

        mem4 = _memory_4bucket_dict(probe)
        daemon_pk = eng.close() if pkg_name.startswith(("public-", "coreai-", "coreml-")) else None
        daemon_telem = eng.telemetry() if hasattr(eng, "telemetry") else None
        diag: dict[str, Any] = {
            "mode": eff_mode,
            "nanobeir_macro_ndcg10": round(float(np.mean(nb_ndcgs)) if nb_ndcgs else 0.0, 6),
            "mrl_128_int8_mean_delta_ndcg10": round(float(np.mean(mrl_deltas)) if mrl_deltas else 0.0, 6),
            "contextual_delta_ndcg10": contextual_lift["contextual_delta_ndcg@10"] if contextual_lift else None,
            "coarse_2k_delta_ndcg10": coarse_2k_lift["contextual_delta_ndcg@10"] if coarse_2k_lift else None,
            "coarse_max_tokens": coarse_max_tokens,
            "ablation_wall_sec": round(ablation_wall_sec, 4),
            "skipped_unjudged_query_ids": skipped_unjudged_by_sub,
            "device_guard_tripped": guard_tripped,
            "repeat_loop_count": 0,
        }
        _merge_daemon_telemetry(mem4, diag, daemon_telem, fallback_peak_gb=daemon_pk)
        bench_name = f"NanoBEIR-en({len(subsets)})+LegalBench-RAG({','.join(lb_pack['subsets'])})"
        return [
            _build_record(
                package=pkg_name,
                category=1,
                config_label=config_label or eff_mode,
                benchmark=bench_name,
                valid_flags=all_valid,
                item_scores=all_scores,
                wall_sec=wall_sec,
                diagnostics=diag,
                memory_4bucket_gb=mem4,
                sub_benchmarks=sub_reports,
                host_mem_gb=host_mem_gb,
            )
        ]
    finally:
        if hasattr(eng, "close"):
            eng.close()


# ---------------------------------------------------------------------------
# Category 2 Worker: fast-decision-models (Laya-ANE & Kev-0.8B)
# ---------------------------------------------------------------------------


def _group_shared_prefix_indices(
    items: Sequence[Any],
    key_fn: Callable[[Any], Any],
    *,
    max_group_size: int,
    length_fn: Callable[[Any], int] | None = None,
) -> tuple[list[int], list[int]]:
    """Group sibling items sharing a prompt/passage into contiguous <=max_group_size blocks and return inverse map."""
    n = len(items)
    if n == 0:
        return [], []
    cap = max(1, int(max_group_size))
    buckets: dict[Any, list[int]] = {}
    for idx, it in enumerate(items):
        buckets.setdefault(key_fn(it), []).append(idx)
    groups: list[list[int]] = []
    for members in buckets.values():
        for off in range(0, len(members), cap):
            groups.append(members[off : off + cap])
    if length_fn is not None:
        groups.sort(key=lambda grp: int(length_fn(items[grp[0]])))
    grouped_order = [idx for grp in groups for idx in grp]
    inv = [0] * n
    for pos, orig_idx in enumerate(grouped_order):
        inv[orig_idx] = pos
    return grouped_order, inv


def _run_cat2_single_backend(
    backend_name: str,
    *,
    pkg_name: str = "fast-decision-models",
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
    coreai_python: str | None = None,
) -> dict[str, Any]:
    from fast_decision_models.kev_mlx import Kev08bMlxBackend
    from fast_decision_models.laya_ane import LayaAneBackend
    from fast_decision_models.paths import ensure_laya_dylib, resolve_kev_dir, resolve_laya_dir
    from _outcome_public_backends import stock_decision_model_files

    art_dir = home / "artifacts"
    cat2_model_dirs = [art_dir, *stock_decision_model_files(backend_name)]
    if backend_name.startswith("coreai-") or pkg_name.startswith("coreai-"):
        cat2_model_dirs.append(_coreai_cache_dir())
    probe = FourBucketMemoryProbe(model_directories=cat2_model_dirs)
    guard = DeviceMemoryGuard()
    guard_tripped = False

    rt_cap = 16 if smoke else max_items
    cl_cap = 16 if smoke else max_items
    rb_cap = 10 if smoke else max_items

    rt_items = load_ragtruth(tier=tier, max_items=rt_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    cl_pack = load_clinc_oos(tier=tier, max_items=cl_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    cl_items = cl_pack["items"]
    rb_items = load_reward_bench_2(tier=tier, max_items=rb_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
    b77_pack = (
        load_banking77(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
        if (tier == "tier2" and not smoke)
        else {"items": [], "label_names": []}
    )

    rt_qdef = build_ragtruth_noul_qdef()
    cl_s1_qdef = build_clinc_stage1_qdef()
    cl_s2_qdefs = {dom: build_clinc_stage2_qdef(dom) for dom in CLINC150_DOMAINS}
    rb_qdef = build_rewardbench2_score_qdef()

    is_laya = backend_name in ("laya-ane", "laya-hetero", "public-laya", "coreai-laya", "coreml-laya")
    if backend_name == "laya-ane":
        laya_dir = resolve_laya_dir(None, artifacts_dir=art_dir)
        eng: Any = LayaAneBackend(laya_dir, ensure_laya_dylib(), pipe_depth=4)
        eff_label = config_label or "laya_ane"
    elif backend_name == "laya-hetero":
        from fast_decision_models.laya_hetero import LayaHeteroEngine

        laya_dir = resolve_laya_dir(None, artifacts_dir=art_dir)
        eng = LayaHeteroEngine(laya_dir, ensure_laya_dylib(), mode="hetero", pipe_depth=4)
        eff_label = config_label or "laya_hetero"
    elif backend_name == "public-laya":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import PublicLayaMlxBackend

        eng = PublicLayaMlxBackend()
        eff_label = config_label or "mlx_embeddings_modernbert_8bit"
    elif backend_name == "coreai-laya":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import CoreAiLayaBackend

        eng = CoreAiLayaBackend(coreai_python=coreai_python)
        eff_label = config_label or "coreai_laya_decision"
    elif backend_name == "coreml-laya":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import CoreMlLayaBackend

        eng = CoreMlLayaBackend()
        eff_label = config_label or "coreml_laya_ane"
    elif backend_name == "public-kev-0.8b":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import PublicKev08bMlxBackend

        kev_dir = resolve_kev_dir(None, artifacts_dir=art_dir)
        eng = PublicKev08bMlxBackend(kev_dir, batch_docs=int(extra.batch_docs))
        eff_label = config_label or "mlx_lm_kev_08b_8bit"
    elif backend_name == "coreml-kev-0.8b":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import CoreMlKev08bBackend

        eng = CoreMlKev08bBackend()
        eff_label = config_label or "coreml_kev_08b_fp16_gpu"
    else:
        kev_dir = resolve_kev_dir(None, artifacts_dir=art_dir)
        eng = Kev08bMlxBackend(kev_dir, quant_mode=extra.kev_quant, batch_docs=int(extra.batch_docs))
        eff_label = config_label or f"kev_08b_{eng.quant_mode}"

    try:
        guard.arm()
        # Untimed warmup
        if is_laya:
            w_enc = eng.encode_item("warmup", rt_qdef, on_truncate="ignore")
            _ = eng.score_noul_batch([w_enc])
        else:
            _ = eng.score_multi_states_noul([("warmup", [rt_qdef])], batch_docs=1)
        _mx_sync()
        _mx_reset_peak()

        # Precompute exact real tokenizer token counts for RAGTruth length slices (untimed so Kev is not double-charged).
        rt_pairs = [(build_ragtruth_state(it), [rt_qdef]) for it in rt_items]
        if is_laya:
            rt_header_toks = len(eng.encode_item("", rt_qdef, on_truncate="ignore")["ids"])
            kev_rt_tok_counts: list[int] = []
        else:
            rt_header_toks = 0
            kev_rt_tok_counts = []
            for st, qlist in rt_pairs:
                s_ids, q_rows = eng.encode_questions(st, qlist)
                kev_rt_tok_counts.append(len(s_ids) + (len(q_rows[0]["ids"]) if q_rows else 0))

        t0 = time.perf_counter()
        sub_reports: dict[str, Any] = {}
        all_valid: list[bool] = []
        all_scores: list[float] = []
        laya_opt_truncations = 0

        # 2A: RAGTruth (`noul`) — group <=6 sibling candidates per passage while preserving deterministic item order
        rt_p_true: list[float] = []
        rt_tok_counts: list[int] = []
        rt_order, rt_inv = _group_shared_prefix_indices(
            rt_items,
            key_fn=lambda it: (str(it.get("context") or it.get("source") or ""), str(it.get("query") or "")),
            max_group_size=6,
            length_fn=lambda it: len(str(it.get("context") or it.get("source") or "")) + len(str(it.get("query") or "")),
        )
        rt_pairs_grouped = [rt_pairs[idx] for idx in rt_order]
        if is_laya:
            rt_encs_g = [eng.encode_item(st, rt_qdef, on_truncate="ignore") for st, _ in rt_pairs_grouped]
            rt_tok_g = [rt_header_toks + int(e.get("original_state_tokens", 0)) for e in rt_encs_g]
            rt_p_g = [float(p) for p in eng.score_noul_batch(rt_encs_g)]
            rt_tok_counts = [rt_tok_g[rt_inv[i]] for i in range(len(rt_items))]
            rt_p_true = [rt_p_g[rt_inv[i]] for i in range(len(rt_items))]
        else:
            rt_tok_counts = kev_rt_tok_counts
            out_rt_g = eng.score_multi_states_noul(rt_pairs_grouped, batch_docs=int(extra.batch_docs))
            rt_p_g = [float(r[0][0]) for r in out_rt_g]
            rt_p_true = [rt_p_g[rt_inv[i]] for i in range(len(rt_items))]
        _mx_sync()
        r_rt = evaluate_ragtruth_noul(rt_items, rt_p_true, token_counts=rt_tok_counts)
        sub_reports["ragtruth_noul"] = {k: v for k, v in r_rt.items() if not k.startswith("per_item_")}
        all_valid.extend(r_rt["per_item_valid"])
        all_scores.extend(r_rt["per_item_scores"])

        # 2B: CLINC150 2-stage (`K=11 -> K=15`)
        if is_laya:
            s1_encs = [eng.encode_item(it["text"], cl_s1_qdef, on_truncate="ignore") for it in cl_items]
            laya_opt_truncations += sum(1 for e in s1_encs if e.get("truncated"))
            s1_probs = eng.score_choice_batch(s1_encs)
            s2_preds: list[tuple[str, Sequence[float]] | None] = [None] * len(cl_items)
            s2_indices: list[int] = []
            s2_doms: list[str] = []
            s2_encs: list[dict[str, Any]] = []
            for idx, probs in enumerate(s1_probs):
                if not probs:
                    continue
                pred_dom = CLINC_STAGE1_DOMAINS[int(np.argmax(np.asarray(probs, dtype=np.float64)))]
                if pred_dom != "out_of_scope" and pred_dom in cl_s2_qdefs:
                    enc2 = eng.encode_item(cl_items[idx]["text"], cl_s2_qdefs[pred_dom], on_truncate="ignore")
                    if enc2.get("truncated"):
                        laya_opt_truncations += 1
                    s2_indices.append(idx)
                    s2_doms.append(pred_dom)
                    s2_encs.append(enc2)
            if s2_encs:
                s2_probs_batch = eng.score_choice_batch(s2_encs)
                for idx, dom, p2 in zip(s2_indices, s2_doms, s2_probs_batch):
                    s2_preds[idx] = (dom, p2)
        else:
            s1_raw = eng.score_multi_states([(it["text"], [cl_s1_qdef]) for it in cl_items], batch_docs=int(extra.batch_docs))
            s1_probs = [r[0][0] for r in s1_raw]
            s2_preds = [None] * len(cl_items)
            s2_indices = []
            s2_doms = []
            s2_pairs: list[tuple[Any, list[Mapping[str, Any]]]] = []
            for idx, probs in enumerate(s1_probs):
                if not probs:
                    continue
                pred_dom = CLINC_STAGE1_DOMAINS[int(np.argmax(np.asarray(probs, dtype=np.float64)))]
                if pred_dom != "out_of_scope" and pred_dom in cl_s2_qdefs:
                    s2_indices.append(idx)
                    s2_doms.append(pred_dom)
                    s2_pairs.append((cl_items[idx]["text"], [cl_s2_qdefs[pred_dom]]))
            if s2_pairs:
                s2_raw = eng.score_multi_states(s2_pairs, batch_docs=int(extra.batch_docs))
                for idx, dom, r2 in zip(s2_indices, s2_doms, s2_raw):
                    s2_preds[idx] = (dom, r2[0][0])
        _mx_sync()
        r_cl = evaluate_clinc_two_stage(cl_items, s1_probs, s2_preds)
        sub_reports["clinc150_2stage"] = {k: v for k, v in r_cl.items() if not k.startswith("per_item_")}
        all_valid.extend(r_cl["per_item_valid"])
        all_scores.extend(r_cl["per_item_scores"])

        # 2C: RewardBench-2 Best-of-4 (`score`) — group 4 candidates per prompt sorted by prompt length
        rb_cand_probs: list[list[list[float]]] = []
        rb_cand_logits: list[list[list[float]] | None] = []
        flat_rb_entries = [
            (item_idx, {"prompt": it["prompt"], "response": cand})
            for item_idx, it in enumerate(rb_items)
            for cand in it["candidates"]
        ]
        rb_order, rb_inv = _group_shared_prefix_indices(
            flat_rb_entries,
            key_fn=lambda ent: (ent[0], str(ent[1]["prompt"])),
            max_group_size=4,
            length_fn=lambda ent: len(str(ent[1]["prompt"])),
        )
        grouped_rb_states = [flat_rb_entries[idx][1] for idx in rb_order]
        if is_laya:
            rb_encs_g = [eng.encode_item(st, rb_qdef, on_truncate="ignore") for st in grouped_rb_states]
            flat_p_g = eng.score_choice_batch(rb_encs_g)
            flat_p = [flat_p_g[rb_inv[i]] for i in range(len(flat_rb_entries))]
            for idx in range(len(rb_items)):
                rb_cand_probs.append(flat_p[4 * idx : 4 * idx + 4])
                rb_cand_logits.append(None)
        else:
            rb_raw_g = eng.score_multi_states([(st, [rb_qdef]) for st in grouped_rb_states], batch_docs=int(extra.batch_docs))
            rb_raw = [rb_raw_g[rb_inv[i]] for i in range(len(flat_rb_entries))]
            for idx in range(len(rb_items)):
                sl = rb_raw[4 * idx : 4 * idx + 4]
                rb_cand_probs.append([x[0][0] for x in sl])
                rb_cand_logits.append([x[0][1] for x in sl])
        _mx_sync()
        r_rb = evaluate_reward_bench_2(rb_items, rb_cand_probs, rb_cand_logits)
        sub_reports["reward_bench_2"] = {k: v for k, v in r_rb.items() if not k.startswith("per_item_")}
        all_valid.extend(r_rb["per_item_valid"])
        all_scores.extend(r_rb["per_item_scores"])

        # Optional Tier-2 Banking77
        if b77_pack["items"]:
            b77_items = b77_pack["items"]
            b77_names = b77_pack["label_names"]
            if not is_laya:
                q_b77 = build_banking77_flat_qdef(b77_names)
                raw_b77 = eng.score_multi_states([(it["text"], [q_b77]) for it in b77_items], batch_docs=int(extra.batch_docs))
                probs_b77 = [r[0][0] for r in raw_b77]
                r_b77 = evaluate_multiclass_choice([int(it["label_id"]) for it in b77_items], probs_b77)
                sub_reports["banking77_k77"] = {k: v for k, v in r_b77.items() if not k.startswith("per_item_")}
                all_valid.extend(r_b77["per_item_valid"])
                all_scores.extend(r_b77["per_item_scores"])
            else:
                # Batched 2-stage 8-cluster hierarchy for Laya parity (0% option truncation).
                clusters = partition_banking77_clusters(b77_names, n_clusters=8)
                c_qdef = {
                    "id": "bank77_c1",
                    "type": "choice",
                    "instructions": "Select the banking intent cluster.",
                    "criteria": [f"cluster_{ci}: {', '.join(cl[:3]).replace('_', ' ')}" for ci, cl in enumerate(clusters)],
                }
                c2_qdefs = {
                    ci: {
                        "id": f"bank77_c2_{ci}",
                        "type": "choice",
                        "instructions": "Select the exact banking intent.",
                        "criteria": [nm.replace("_", " ") for nm in cl],
                    }
                    for ci, cl in enumerate(clusters)
                }
                c_encs = [eng.encode_item(it["text"], c_qdef, on_truncate="ignore") for it in b77_items]
                laya_opt_truncations += sum(1 for e in c_encs if e.get("truncated"))
                c_probs = eng.score_choice_batch(c_encs)
                b77_s1_valid = check_probability_validity(c_probs)

                b77_s2_idxs: list[int] = []
                b77_s2_cis: list[int] = []
                b77_s2_encs: list[dict[str, Any]] = []
                for idx, it in enumerate(b77_items):
                    if not b77_s1_valid[idx]:
                        continue
                    ci = int(np.argmax(np.asarray(c_probs[idx], dtype=np.float64)))
                    enc2 = eng.encode_item(it["text"], c2_qdefs[ci], on_truncate="ignore")
                    if enc2.get("truncated"):
                        laya_opt_truncations += 1
                    b77_s2_idxs.append(idx)
                    b77_s2_cis.append(ci)
                    b77_s2_encs.append(enc2)

                b77_valid = [False] * len(b77_items)
                b77_scores = [0.0] * len(b77_items)
                if b77_s2_encs:
                    s2_probs_batch = eng.score_choice_batch(b77_s2_encs)
                    s2_valid_batch = check_probability_validity(s2_probs_batch)
                    for idx, ci, p2, s2_ok in zip(b77_s2_idxs, b77_s2_cis, s2_probs_batch, s2_valid_batch):
                        is_ok = bool(b77_s1_valid[idx] and s2_ok)
                        b77_valid[idx] = is_ok
                        if is_ok:
                            pred_name = clusters[ci][int(np.argmax(np.asarray(p2, dtype=np.float64)))]
                            b77_scores[idx] = 1.0 if pred_name == b77_items[idx]["label_name"] else 0.0
                sub_reports["banking77_2stage"] = {
                    "n_items": len(b77_items),
                    "n_valid": int(sum(1 for v in b77_valid if v)),
                    "top1_accuracy": round(float(np.mean(b77_scores)) if b77_scores else 0.0, 6),
                }
                all_valid.extend(b77_valid)
                all_scores.extend(b77_scores)

        _mx_sync()
        wall_sec = time.perf_counter() - t0
        _, ref = guard.check()
        if ref is not None:
            guard_tripped = True

        # Headline Cat 2 macro composite stored in diagnostics; EOT uses exact sum(v_i * s_i) / wall_sec.
        headline_s = float(np.mean([r_rt["auroc"], r_cl["end_to_end_151_acc"], r_rb["macro_best_of_4_acc"]]))
        mem4 = _memory_4bucket_dict(probe)
        daemon_pk: float | None = None
        daemon_telem: dict[str, Any] | None = None
        if hasattr(eng, "telemetry") or hasattr(eng, "peak_gb"):
            pk_ret = eng.close()
            daemon_pk = float(pk_ret) if isinstance(pk_ret, (int, float)) else float(getattr(eng, "peak_gb", 0.0))
            if hasattr(eng, "telemetry"):
                daemon_telem = eng.telemetry()
        diag = {
            "backend": backend_name,
            "composite_macro_score": round(headline_s, 6),
            "ragtruth_auroc": r_rt["auroc"],
            "ragtruth_macro_f1": r_rt["macro_f1"],
            "ragtruth_brier": r_rt["brier"],
            "ragtruth_ece_15": r_rt["ece_15"],
            "clinc150_151_acc": r_cl["end_to_end_151_acc"],
            "clinc150_oos_auroc": r_cl["oos_auroc"],
            "rewardbench2_macro_bo4": r_rb["macro_best_of_4_acc"],
            "laya_clinc_option_truncations": laya_opt_truncations if is_laya else 0,
            "device_guard_tripped": guard_tripped,
            "repeat_loop_count": 0,
        }
        _merge_daemon_telemetry(mem4, diag, daemon_telem, fallback_peak_gb=daemon_pk)
        return _build_record(
            package=pkg_name,
            category=2,
            config_label=eff_label,
            benchmark="RAGTruth+CLINC150+RewardBench2",
            valid_flags=all_valid,
            item_scores=all_scores,
            wall_sec=wall_sec,
            diagnostics=diag,
            memory_4bucket_gb=mem4,
            sub_benchmarks=sub_reports,
            host_mem_gb=host_mem_gb,
        )
    finally:
        if hasattr(eng, "close"):
            eng.close()


# ---------------------------------------------------------------------------
# Categories 3 & 4 Workers: fast-gemma4, fast-gpt-oss, optional OpenAI-compatible
# ---------------------------------------------------------------------------


def _format_mmlu_prompt(item: Mapping[str, Any]) -> str:
    letters = ("A", "B", "C", "D")
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    return (
        f"Question ({item['subject'].replace('_', ' ')}):\n{item['question']}\n\n"
        f"{opts}\n\nRespond with only 'Answer: <A|B|C|D>'."
    )


def _format_math_prompt(item: Mapping[str, Any]) -> str:
    return f"Solve the following math problem step by step and put your final answer inside \\boxed{{}}.\n\n{item['problem']}"


def _format_liquidai_prompt(item: Mapping[str, Any]) -> str:
    sch_str = json.dumps(item["schema"], ensure_ascii=False)
    return (
        f"Extract the requested fields from the document as a single valid JSON object matching this schema:\n"
        f"{sch_str}\n\nDocument:\n{item['document']}"
    )


def _format_bfcl_prompt(item: Mapping[str, Any]) -> str:
    q_turns = item.get("question") or []
    user_txt = ""
    if q_turns and isinstance(q_turns[0], list):
        user_txt = "\n".join(str(m.get("content") or "") for m in q_turns[0] if isinstance(m, Mapping))
    elif isinstance(q_turns, list):
        user_txt = "\n".join(str(m) for m in q_turns)
    tools_str = json.dumps(item.get("functions") or [], ensure_ascii=False)
    return (
        f"Available tools (JSON definitions):\n{tools_str}\n\n"
        f"User request:\n{user_txt}\n\n"
        f'Return a JSON object {{"tool_calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}. '
        f'If none of the tools match the user request, return {{"tool_calls": []}}.'
    )


def _compile_required_grammars(
    package: str,
    schemas: Sequence[dict[str, Any] | None],
    compile_schema: Callable[..., Any],
    *,
    whitespace_flexible: bool | None,
) -> list[Any]:
    grammars = []
    for index, schema in enumerate(schemas):
        if schema is None:
            raise ValueError(f"Missing required schema for {package} input {index}")
        try:
            grammars.append(compile_schema(schema, whitespace_flexible=whitespace_flexible))
        except Exception as exc:
            raise ValueError(f"Could not compile required schema for {package} input {index}: {exc}") from exc
    return grammars


def _run_llm_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    include_unconstrained: bool,
    home: Path,
    host_mem_gb: float,
    openai_url: str | None = None,
    openai_model: str | None = None,
    coreai_python: str | None = None,
    coreai_variant_override: str | None = None,
) -> list[dict[str, Any]]:
    want_cats = {3, 4} if extra.categories == "all" else {int(x.strip()) for x in extra.categories.split(",") if x.strip()}
    model_dirs: list[Path] = [home / "artifacts"] if home.exists() else []
    if pkg_name.startswith("public-"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import stock_hf_model_files

        model_dirs.extend(stock_hf_model_files(pkg_name))
    elif pkg_name.startswith("coreai-"):
        model_dirs.append(_coreai_cache_dir())
    probe = FourBucketMemoryProbe(model_directories=model_dirs or None)
    guard = DeviceMemoryGuard()
    guard_tripped = False

    eng: Any = None
    pub_llm: Any = None
    ltok: Any = None
    eff_label = config_label
    if extra.batch_size is not None:
        batch_sz = int(extra.batch_size)
    elif pkg_name in ("fast-gemma4", "public-gemma4"):
        batch_sz = 64 if host_mem_gb >= 36.0 else 32
    elif pkg_name in ("fast-qwen3", "public-qwen3.5", "coreai-qwen3.5-4b"):
        batch_sz = 16 if 0 < host_mem_gb < 36.0 else 32
    elif pkg_name in ("fast-gpt-oss", "public-gpt-oss"):
        batch_sz = 8 if 0 < host_mem_gb < 36.0 else 16
    else:
        batch_sz = 16
    if pkg_name in ("fast-gpt-oss", "public-gpt-oss") and 0 < host_mem_gb < 36.0:
        batch_sz = min(batch_sz, 8)
    use_mtp = bool(extra.mtp_drafter or extra.mtp_drafter_path)

    if pkg_name == "fast-gemma4":
        from fast_gemma4.engine import FastGemma4Engine

        eff_mode = extra.mode or "gpu"
        mtp_ref = extra.mtp_drafter_path or (True if extra.mtp_drafter else None)
        eff_kv_i8 = False if extra.no_kv_i8 else bool(extra.kv_i8)
        req_mlp_dtype = extra.mlp_dtype or "auto"
        g4_kw: dict[str, Any] = {
            "home": home,
            "model_dir": extra.model_dir,
            "vocab_out": extra.gemma_vocab_out,
            "default_mode": eff_mode,
            "ane_variant": extra.ane_variant,
            "ane_ring_variant": extra.ane_variant,
            "lm_head_mode": extra.lm_head_mode,
            "mlp_dtype": req_mlp_dtype,
            "kv_i8": eff_kv_i8,
            "mtp_drafter": mtp_ref,
            "local_files_only": True,
            "warmup": True,
            "warmup_ane": (eff_mode == "hetero"),
        }
        try:
            eng = FastGemma4Engine(**g4_kw)
        except ValueError as exc:
            if req_mlp_dtype == "auto" and "mlp_dtype" in str(exc):
                g4_kw["mlp_dtype"] = "quantized" if (0 < host_mem_gb < 36.0 or batch_sz <= 32) else "bfloat16"
                eng = FastGemma4Engine(**g4_kw)
            else:
                raise
        ltok = eng.tokenizer.llguidance_tokenizer()
        mtp_tag = "_mtp" if use_mtp else ""
        eff_label = config_label or f"{eff_mode}_{extra.ane_variant}_{'kvi8' if eff_kv_i8 else 'kvbf16'}{mtp_tag}"
    elif pkg_name == "fast-qwen3":
        from fast_qwen3.engine import FastQwen3Engine

        eff_mode = extra.mode or "auto"
        qwen_kv: bool | str = (
            extra.kv_mode
            if extra.kv_mode is not None
            else (False if extra.no_kv_i8 else (True if extra.kv_i8 else "auto"))
        )
        eng = FastQwen3Engine(
            home=home,
            model_dir=extra.model_dir,
            default_mode=eff_mode,
            default_enable_thinking=False,
            kv_i8=qwen_kv,
            local_files_only=True,
            warmup=True,
        )
        ltok = eng.tokenizer.llguidance_tokenizer()
        mtp_tag = "_mtp" if use_mtp else ""
        kv_tag = f"_{qwen_kv}" if isinstance(qwen_kv, str) and qwen_kv != "auto" else ("_kvi8" if qwen_kv is True else "")
        eff_label = config_label or f"qwen3_{eff_mode}{kv_tag}{mtp_tag}"
    elif pkg_name == "fast-gpt-oss":
        from fast_gpt_oss.engine import FastGptOssEngine

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from public_baselines import resolve_hf_snapshot

        eff_mode = extra.mode or "auto"
        oss_model_dir: str | Path | None = extra.model_dir
        if oss_model_dir is None and not (home / "artifacts" / "hf_model" / "config.json").is_file():
            q8_snap = resolve_hf_snapshot("mlx-community/gpt-oss-20b-MXFP4-Q8")
            if q8_snap is not None:
                oss_model_dir = q8_snap
        eng = FastGptOssEngine(
            home=home,
            model_dir=oss_model_dir,
            default_mode=eff_mode,
            default_reasoning_effort=extra.reasoning_effort or "low",
            local_files_only=True,
            warmup=True,
        )
        ltok = eng.tokenizer.llguidance_tokenizer()
        eff_label = config_label or f"gpt_oss_{eff_mode}"
    elif pkg_name in ("public-gemma4", "public-qwen3.5", "public-gpt-oss", "coreai-qwen3.5-4b", "coreai-gemma4"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _outcome_public_backends import build_public_llm_adapter

        ca_var = coreai_variant_override or ("4bit" if extra.coreai_variant == "both" else extra.coreai_variant)
        pub_llm = build_public_llm_adapter(
            pkg_name,
            batch_size=batch_sz,
            host_mem_gb=host_mem_gb,
            coreai_python=coreai_python,
            coreai_variant=ca_var,
        )
        eff_label = config_label or pub_llm.default_label
    elif pkg_name == "openai-chat":
        if not openai_url:
            raise ValueError("openai-chat package requires --openai-url")
        eff_label = config_label or (openai_model or "openai_chat")
    else:
        raise ValueError(f"Unsupported LLM package: {pkg_name}")

    model_settings = {}
    if pkg_name == "fast-gemma4":
        model_settings = {
            "model_checkpoint": eng.model.model_dir.name,
            "output_vocab_size": eng.model.vocab_size,
            "batch_size": batch_sz,
            "mlx_cache_limit_mb": extra.mlx_cache_limit_mb,
        }

    def _generate_batch(
        prompts: list[str],
        *,
        system_prompt: str | None = None,
        per_doc_schemas: list[dict[str, Any] | None] | None = None,
        max_tokens: int = 256,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
    ) -> tuple[list[str], list[str], list[list[int]], dict[str, Any]]:
        if not prompts:
            return [], [], [], {"completion_tokens": 0, "mask_fallbacks": 0}
        if pub_llm is not None:
            pub_kw: dict[str, Any] = {
                "system_prompt": system_prompt,
                "per_doc_schemas": per_doc_schemas,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            }
            if reasoning_budget is not None and accepts_kwarg(pub_llm.generate_fn, "reasoning_budget"):
                pub_kw["reasoning_budget"] = int(reasoning_budget)
            return pub_llm.generate_fn(prompts, **pub_kw)
        if pkg_name == "openai-chat":
            texts: list[str] = []
            reasons: list[str] = []
            tot_comp = 0
            for idx, pr in enumerate(prompts):
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": pr})
                body: dict[str, Any] = {
                    "model": openai_model or "default",
                    "messages": msgs,
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                }
                if per_doc_schemas and per_doc_schemas[idx] is not None:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "extraction", "schema": per_doc_schemas[idx]},
                    }
                req = urllib.request.Request(
                    f"{openai_url.rstrip('/')}/v1/chat/completions",  # type: ignore[union-attr]
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                ch = payload["choices"][0]
                texts.append(str(ch["message"].get("content") or ""))
                reasons.append(str(ch.get("finish_reason") or "stop"))
                tot_comp += int(payload.get("usage", {}).get("completion_tokens", 0))
            return texts, reasons, [[] for _ in texts], {"completion_tokens": tot_comp, "mask_fallbacks": 0}

        safe_grammars = None
        if per_doc_schemas is not None:
            is_gemma = pkg_name == "fast-gemma4"
            whitespace = {"fixed": None, "compact": False, "flexible": True}[extra.gemma_json_layout] if is_gemma else None
            safe_grammars = _compile_required_grammars(
                pkg_name, per_doc_schemas, eng.get_grammar if is_gemma else eng._get_grammar,
                whitespace_flexible=whitespace,
            )

        eff_max_tokens = int(max_tokens)
        if (
            pkg_name == "fast-gpt-oss"
            and per_doc_schemas is None
            and reasoning_budget is not None
            and int(reasoning_budget) > 0
            and eff_max_tokens <= int(reasoning_budget)
        ):
            eff_max_tokens = int(reasoning_budget) + eff_max_tokens
        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "per_doc_grammars": safe_grammars,
            "max_tokens": eff_max_tokens,
            "temperature": 0.0,
            "batch_size": min(batch_sz, len(prompts)),
        }
        if pkg_name == "fast-gpt-oss":
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            if reasoning_budget is not None:
                kwargs["reasoning_budget"] = int(reasoning_budget)
        elif pkg_name in ("fast-gemma4", "fast-qwen3"):
            kwargs["enable_thinking"] = False
            kwargs["speculative"] = True if use_mtp else False

        res = eng.generate_batch(prompts, **kwargs)
        _mx_sync()
        texts = [it.text for it in res.items]
        reasons = [it.finish_reason for it in res.items]
        tok_ids = [list(it.token_ids) for it in res.items]
        st = dict(res.stats)
        st["grammar_compile_fallbacks"] = 0
        return texts, reasons, tok_ids, st

    records: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    try:
        guard.arm()
        # Untimed warmup
        _ = _generate_batch(["Reply OK."], max_tokens=8, reasoning_effort="skip")
        _mx_sync()
        _mx_reset_peak()

        # ---------------- Category 3: Chat, IF & Reasoning ----------------
        if 3 in want_cats:
            if_cap = 6 if smoke else max_items
            mm_cap = 12 if smoke else max_items
            math_cap = 4 if smoke else max_items

            ifb_items = load_ifbench_test(tier=tier, max_items=if_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            mm_items = load_mmlu_redux(
                tier=tier,
                subjects=lock["datasets"]["mmlu_redux_2"]["subjects"][:4] if smoke else None,
                max_items=mm_cap,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            math_items = load_math_500(tier=tier, max_items=math_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            ife_items = (
                load_ifeval(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
                if (tier == "tier2" and not smoke)
                else []
            )
            lb_items = (
                load_livebench_reasoning(tier=tier, max_items=max_items, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
                if (tier == "tier2" and not smoke)
                else []
            )

            t0 = time.perf_counter()
            sub3: dict[str, Any] = {}
            all_v3: list[bool] = []
            all_s3: list[float] = []
            tot_comp_toks = 0
            tot_rep_loops = 0
            tot_fin_len = 0

            # 3A: IFBench (+ IFEval in Tier-2)
            ifb_txt, ifb_fr, ifb_ids, ifb_st = _generate_batch(
                [it["prompt"] for it in ifb_items],
                max_tokens=256 if smoke else 512,
                reasoning_effort="low",
                reasoning_budget=192,
            )
            r_ifb = evaluate_instruction_following(ifb_items, ifb_txt, finish_reasons=ifb_fr, token_ids_list=ifb_ids)
            sub3["ifbench_test"] = {k: v for k, v in r_ifb.items() if not k.startswith("per_item_")}
            all_v3.extend(r_ifb["per_item_valid"])
            all_s3.extend(r_ifb["per_item_scores"])
            tot_comp_toks += int(ifb_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_ifb["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in ifb_fr if fr == "length")

            if_scores_for_comp = [float(r_ifb["prompt_loose_acc"])]
            if ife_items:
                ife_txt, ife_fr, ife_ids, ife_st = _generate_batch(
                    [it["prompt"] for it in ife_items],
                    max_tokens=512,
                    reasoning_effort="low",
                    reasoning_budget=192,
                )
                r_ife = evaluate_instruction_following(ife_items, ife_txt, finish_reasons=ife_fr, token_ids_list=ife_ids)
                sub3["ifeval"] = {k: v for k, v in r_ife.items() if not k.startswith("per_item_")}
                all_v3.extend(r_ife["per_item_valid"])
                all_s3.extend(r_ife["per_item_scores"])
                tot_comp_toks += int(ife_st.get("completion_tokens", 0))
                tot_rep_loops += int(r_ife["repeat_loop_count"])
                tot_fin_len += sum(1 for fr in ife_fr if fr == "length")
                if_scores_for_comp.append(float(r_ife["prompt_loose_acc"]))

            # 3B: MMLU-Redux 2.0 (max_tokens=16, budgeted low reasoning for Harmony models)
            mm_txt, mm_fr, mm_ids, mm_st = _generate_batch(
                [_format_mmlu_prompt(it) for it in mm_items],
                max_tokens=16,
                reasoning_effort="low",
                reasoning_budget=64,
            )
            r_mm = evaluate_mmlu_redux(mm_items, mm_txt, finish_reasons=mm_fr, token_ids_list=mm_ids)
            sub3["mmlu_redux_2"] = {k: v for k, v in r_mm.items() if not k.startswith("per_item_")}
            all_v3.extend(r_mm["per_item_valid"])
            all_s3.extend(r_mm["per_item_scores"])
            tot_comp_toks += int(mm_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_mm["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in mm_fr if fr == "length")

            # 3C: MATH-500 (+ LiveBench-Reasoning in Tier-2)
            math_txt, math_fr, math_ids, math_st = _generate_batch(
                [_format_math_prompt(it) for it in math_items],
                max_tokens=512 if smoke else 1536,
                reasoning_effort="low",
                reasoning_budget=384,
            )
            r_math = evaluate_math_500(math_items, math_txt, finish_reasons=math_fr, token_ids_list=math_ids)
            sub3["math_500"] = {k: v for k, v in r_math.items() if not k.startswith("per_item_")}
            all_v3.extend(r_math["per_item_valid"])
            all_s3.extend(r_math["per_item_scores"])
            tot_comp_toks += int(math_st.get("completion_tokens", 0))
            tot_rep_loops += int(r_math["repeat_loop_count"])
            tot_fin_len += sum(1 for fr in math_fr if fr == "length")

            reason_scores = [float(r_math["verified_accuracy"])]
            if lb_items:
                lb_txt, lb_fr, lb_ids, lb_st = _generate_batch(
                    [it["prompt"] for it in lb_items],
                    max_tokens=1536,
                    reasoning_effort="low",
                    reasoning_budget=384,
                )
                r_lb = evaluate_livebench_reasoning(lb_items, lb_txt, finish_reasons=lb_fr, token_ids_list=lb_ids)
                sub3["livebench_reasoning"] = {k: v for k, v in r_lb.items() if not k.startswith("per_item_")}
                all_v3.extend(r_lb["per_item_valid"])
                all_s3.extend(r_lb["per_item_scores"])
                tot_comp_toks += int(lb_st.get("completion_tokens", 0))
                tot_rep_loops += int(r_lb["repeat_loop_count"])
                tot_fin_len += sum(1 for fr in lb_fr if fr == "length")
                reason_scores.append(float(r_lb["verified_accuracy"]))

            _mx_sync()
            wall3 = time.perf_counter() - t0
            _, ref3 = guard.check()
            if ref3 is not None:
                guard_tripped = True
            mem3 = _memory_4bucket_dict(probe)
            comp3 = float(np.mean([float(np.mean(if_scores_for_comp)), float(r_mm["exact_accuracy"]), float(np.mean(reason_scores))]))
            diag3 = {
                **model_settings,
                "composite_macro_score": round(comp3, 6),
                "if_prompt_loose_acc": round(float(np.mean(if_scores_for_comp)), 6),
                "mmlu_redux_exact_acc": r_mm["exact_accuracy"],
                "math_reasoning_acc": round(float(np.mean(reason_scores)), 6),
                "avg_completion_tokens": round(float(tot_comp_toks) / float(max(1, len(all_v3))), 2),
                "repeat_4gram_rate": round(float(r_ifb["repeat_4gram_rate"]), 6),
                "repeat_loop_count": tot_rep_loops,
                "finish_length_count": tot_fin_len,
                "device_guard_tripped": guard_tripped,
            }
            if pub_llm is not None:
                _merge_daemon_telemetry(mem3, diag3, pub_llm.telemetry())
            pending_records.append(
                {
                    "package": pkg_name,
                    "category": 3,
                    "config_label": f"{eff_label}_cat3",
                    "benchmark": "IFBench+MMLU-Redux+MATH-500",
                    "valid_flags": all_v3,
                    "item_scores": all_s3,
                    "wall_sec": wall3,
                    "diagnostics": diag3,
                    "memory_4bucket_gb": mem3,
                    "sub_benchmarks": sub3,
                    "host_mem_gb": host_mem_gb,
                }
            )

        # ---------------- Category 4: Structured JSON & Tool AST ----------------
        if 4 in want_cats:
            lq_cap = 8 if smoke else max_items
            bfcl_per = 2 if smoke else (max(1, int(max_items) // 5) if max_items else None)

            lq_items = load_liquidai_extraction(tier=tier, max_items=lq_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            bfcl_items = load_bfcl_v3(tier=tier, per_split=bfcl_per, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)
            jsb_items = load_jsonschemabench(tier="tier1" if smoke else tier, per_subset=10 if smoke else None, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only)

            # Auxiliary JSONSchemaBench llguidance compiler diagnostic (< 5s CPU)
            jsb_diag = evaluate_jsonschemabench_compiler(jsb_items, llguidance_tokenizer=ltok)

            if pub_llm is not None:
                llg_modes = [False]
            else:
                llg_modes = [not bool(extra.no_llguidance)]
                if include_unconstrained and True in llg_modes and False not in llg_modes:
                    llg_modes.append(False)

            for use_llg in llg_modes:
                t0 = time.perf_counter()
                sub4: dict[str, Any] = {"jsonschemabench_compiler": jsb_diag}
                all_v4: list[bool] = []
                all_s4: list[float] = []
                tot_comp_4 = 0
                tot_fb_4 = 0
                tot_rep_4 = 0
                tot_len_4 = 0

                # 4A: LiquidAI/structured-extraction
                lq_prompts = [_format_liquidai_prompt(it) for it in lq_items]
                lq_schemas = [it["schema"] for it in lq_items] if use_llg else None
                lq_txt, lq_fr, lq_ids, lq_st = _generate_batch(
                    lq_prompts,
                    system_prompt="Extract fields strictly from the document as compact JSON matching the schema.",
                    per_doc_schemas=lq_schemas,
                    max_tokens=256 if smoke else 512,
                    reasoning_effort="low",
                    reasoning_budget=64,
                )
                r_lq = evaluate_structured_extraction(lq_items, lq_txt, finish_reasons=lq_fr, token_ids_list=lq_ids)
                sub4["liquidai_extraction"] = {k: v for k, v in r_lq.items() if not k.startswith("per_item_")}
                all_v4.extend(r_lq["per_item_valid"])
                all_s4.extend(r_lq["per_item_scores"])
                tot_comp_4 += int(lq_st.get("completion_tokens", 0))
                tot_fb_4 += int(lq_st.get("mask_fallbacks", lq_st.get("mask_hits", 0)) or 0)
                tot_rep_4 += int(r_lq["repeat_loop_count"])
                tot_len_4 += sum(1 for fr in lq_fr if fr == "length")

                # 4B: BFCL v3 AST + Irrelevance
                bf_prompts = [_format_bfcl_prompt(it) for it in bfcl_items]
                bf_schemas = [build_bfcl_response_schema(it["functions"]) for it in bfcl_items] if use_llg else None
                bf_txt, bf_fr, bf_ids, bf_st = _generate_batch(
                    bf_prompts,
                    system_prompt="Select matching tool calls or return empty tool_calls if none apply. Output raw JSON only.",
                    per_doc_schemas=bf_schemas,
                    max_tokens=192 if smoke else 384,
                    reasoning_effort="low",
                    reasoning_budget=96,
                )
                r_bf = evaluate_bfcl_v3(bfcl_items, bf_txt, finish_reasons=bf_fr, token_ids_list=bf_ids)
                sub4["bfcl_v3"] = {k: v for k, v in r_bf.items() if not k.startswith("per_item_")}
                all_v4.extend(r_bf["per_item_valid"])
                all_s4.extend(r_bf["per_item_scores"])
                tot_comp_4 += int(bf_st.get("completion_tokens", 0))
                tot_fb_4 += int(bf_st.get("mask_fallbacks", bf_st.get("mask_hits", 0)) or 0)
                tot_rep_4 += int(r_bf["repeat_loop_count"])
                tot_len_4 += sum(1 for fr in bf_fr if fr == "length")

                _mx_sync()
                wall4 = time.perf_counter() - t0
                _, ref4 = guard.check()
                if ref4 is not None:
                    guard_tripped = True
                mem4 = _memory_4bucket_dict(probe)
                fb_rate = float(tot_fb_4) / float(max(1, tot_comp_4))
                diag4 = {
                    "llguidance": bool(use_llg),
                    "grounded_field_f1": r_lq["grounded_field_f1"],
                    "valid_schema_rate": r_lq["valid_schema_rate"],
                    "bfcl_ast_acc": r_bf["overall_ast_acc"],
                    "bfcl_abstention_acc": r_bf["abstention_acc"],
                    "jsb_compile_rate": jsb_diag["compile_success_rate"],
                    "jsb_gct_p50_ms": jsb_diag["gct_p50_ms"],
                    "avg_completion_tokens": round(float(tot_comp_4) / float(max(1, len(all_v4))), 2),
                    "mask_fallback_rate": round(fb_rate, 6),
                    "repeat_4gram_rate": round(0.5 * (float(r_lq["repeat_4gram_rate"]) + float(r_bf["repeat_4gram_rate"])), 6),
                    "repeat_loop_count": tot_rep_4,
                    "finish_length_count": tot_len_4,
                    "device_guard_tripped": guard_tripped,
                }
                if model_settings:
                    diag4.update(model_settings, json_layout=extra.gemma_json_layout if use_llg else None)
                if pub_llm is not None:
                    _merge_daemon_telemetry(mem4, diag4, pub_llm.telemetry())
                pending_records.append(
                    {
                        "package": pkg_name,
                        "category": 4,
                        "config_label": f"{eff_label}_{'llg_on' if use_llg else 'llg_off'}",
                        "benchmark": f"LiquidAI/structured-extraction({len(lq_items)})+BFCL_v3({len(bfcl_items)})",
                        "valid_flags": all_v4,
                        "item_scores": all_s4,
                        "wall_sec": wall4,
                        "diagnostics": diag4,
                        "memory_4bucket_gb": mem4,
                        "sub_benchmarks": sub4,
                        "host_mem_gb": host_mem_gb,
                    }
                )
        if pub_llm is not None:
            daemon_pk = pub_llm.close()
            final_telem = pub_llm.telemetry()
            for item in pending_records:
                _merge_daemon_telemetry(
                    item["memory_4bucket_gb"],
                    item["diagnostics"],
                    final_telem,
                    fallback_peak_gb=daemon_pk,
                )
        for item in pending_records:
            records.append(_build_record(**item))
    finally:
        if pub_llm is not None:
            pub_llm.close()
        if eng is not None and hasattr(eng, "close"):
            eng.close()
    return records


# ---------------------------------------------------------------------------
# Category 5 Workers: fast-ocr + cross-family MLX VLMs / ANE / ONNX / CLI / PDF pipelines
# ---------------------------------------------------------------------------

_CAT5_PACKAGES: frozenset[str] = frozenset(OCR_FAMILY_PACKAGES) | frozenset({"fast-gemma4-vision"})
_VISUAL_RETRIEVAL_PACKAGES: frozenset[str] = frozenset({"fast-embeddinggemma-2-vision", "fast-pplx-vision"})
_VISUAL_RETRIEVAL_ALL_SUITES: tuple[str, ...] = ("fintabnet_otsl", "omnidocbench_en", "cord_v2", "olmocr_bench")
_VISUAL_RETRIEVAL_DEFAULT_SUITES: tuple[str, ...] = ("fintabnet_otsl", "omnidocbench_en", "cord_v2")
_CAT5_ALL_SUITES: tuple[str, ...] = (
    "olmocr_bench",
    "fintabnet_otsl",
    "omnidocbench_en",
    "cord_v2",
    "vrdu_ad_buy",
    "vrdu_registration",
    "doclaynet_gov",
    "puredocbench_legal_gov",
    "puredocbench_en",
    "us_gov_ocr",
)
_CAT5_PDF_NATIVE_SUITES: frozenset[str] = frozenset({"olmocr_bench", "doclaynet_gov", "us_gov_ocr"})
_CAT5_AUTO_BATCH_SIZE: dict[str, int] = {
    "fintabnet_otsl": 16,
    "cord_v2": 16,
    "vrdu_ad_buy": 16,
    "vrdu_registration": 16,
}


def _resolve_cat5_suite_batch_size(suite_name: str, extra: argparse.Namespace) -> int:
    """Resolve effective Category 5 batch size: explicit `--ocr-batch-size` / `--batch-size`, else suite policy."""
    req = getattr(extra, "ocr_batch_size", "auto")
    if isinstance(req, int):
        return max(1, req)
    if getattr(extra, "batch_size", None) is not None:
        return max(1, int(extra.batch_size))
    if req is None or str(req).strip().lower() == "auto":
        return int(_CAT5_AUTO_BATCH_SIZE.get(suite_name, 1))
    return max(1, int(req))

_PUREDOCBENCH_EN_SUBDIR: dict[str, str] = {
    "clean": "clean",
    "digital": "digital_degraded",
    "real": "real_degraded",
}


def load_puredocbench_en(
    *,
    track: str = "clean",
    max_items: int | None = None,
    en_root: str | Path | None = None,
    images_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load locally staged full-corpus English `puredocbench_en` pages across `clean|digital|real` tracks."""
    import csv

    eff_track = str(track or "clean").strip().lower()
    if eff_track not in _PUREDOCBENCH_EN_SUBDIR:
        raise ValueError(f"Unsupported PureDocBench track={track!r}; expected 'clean', 'digital', or 'real'")

    base_dir = Path(en_root).expanduser() if en_root is not None else (resolve_cache_dir(cache_dir) / "puredocbench_en")
    manifest_p = base_dir / "release_manifest_en.csv"
    if not manifest_p.is_file():
        cand_manifest = base_dir / "release_manifest_candidate_1475.csv"
        if cand_manifest.is_file():
            manifest_p = cand_manifest
        else:
            raise FileNotFoundError(
                f"PureDocBench EN manifest not found under {base_dir}; sync `puredocbench_en` or pass `--puredocbench-en-root`."
            )
    gt_root = base_dir / "gt"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"PureDocBench EN ground-truth directory missing: {gt_root}")

    img_root = Path(images_root).expanduser() if images_root is not None else (base_dir / "images")
    track_subdir = _PUREDOCBENCH_EN_SUBDIR[eff_track]
    rel_col = f"{eff_track}_rel"

    items: list[dict[str, Any]] = []
    with manifest_p.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            gt_rel = str(row.get("gt_rel") or "").strip()
            if not gt_rel:
                continue
            gt_path = gt_root / gt_rel
            if not gt_path.is_file():
                continue
            img_rel = str(row.get(rel_col) or "").strip()
            img_path: Path | None = None
            if img_rel:
                for cand in (img_root / track_subdir / img_rel, img_root / img_rel):
                    if cand.is_file():
                        img_path = cand
                        break
            if img_path is None:
                continue

            page = json.loads(gt_path.read_text(encoding="utf-8"))
            pattr = ((page.get("page_info") or {}).get("page_attribute") or {})
            raw_dets = [d for d in (page.get("layout_dets") or []) if isinstance(d, Mapping) and not d.get("ignore")]
            raw_dets.sort(key=lambda d: (int(d["order"]) if d.get("order") is not None else 10**9, str(d.get("anno_id") or "")))

            text_blocks: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []
            equations: list[dict[str, Any]] = []
            for d in raw_dets:
                ctype = str(d.get("category_type") or "")
                order_val = d.get("order")
                if ctype in ("text_block", "title", "reference", "figure_caption", "table_caption", "equation_caption", "code_txt"):
                    txt = str(d.get("text") or "").strip()
                    if txt:
                        text_blocks.append({"category_type": ctype, "order": order_val, "text": txt})
                elif ctype == "table":
                    html_s = str(d.get("html") or d.get("text") or "").strip()
                    if html_s:
                        tables.append({"order": order_val, "html": html_s})
                elif ctype in ("equation_isolated", "equation_semantic"):
                    ltx = str(d.get("latex") or d.get("text") or "").strip()
                    if ltx:
                        equations.append({"category_type": ctype, "order": order_val, "latex": ltx})

            if not (text_blocks or tables or equations):
                continue
            cat = str(row.get("category") or (Path(gt_rel).parts[0] if Path(gt_rel).parts else "puredocbench_en"))
            subcat = str(row.get("subcategory") or (Path(gt_rel).parts[1] if len(Path(gt_rel).parts) > 1 else cat))
            items.append(
                {
                    "id": str(row.get("page_id") or gt_rel),
                    "domain": cat,
                    "subcategory": subcat,
                    "track": eff_track,
                    "image_rel": img_rel,
                    "image_path": str(img_path),
                    "data_source": str(pattr.get("data_source") or "puredocbench_en"),
                    "text_blocks": text_blocks,
                    "tables": tables,
                    "equations": equations,
                }
            )

    if not items:
        raise FileNotFoundError(
            f"No ready PureDocBench EN items with images found under {base_dir} (track={eff_track})."
        )
    return stratified_sha256_sample(
        items,
        dataset_key=f"puredocbench_en:{eff_track}",
        id_fn=lambda r: r["id"],
        stratum_fn=lambda r: r["domain"],
        total_n=max_items,
    )


def _resolve_cat5_suites(pkg_name: str, extra: argparse.Namespace) -> list[str]:
    raw = str(getattr(extra, "ocr_suites", "default") or "default").strip().lower()
    ocr_model_low = str(getattr(extra, "ocr_model", None) or "").strip().lower()
    if raw == "all":
        requested = list(_CAT5_ALL_SUITES)
    elif raw == "gov":
        if pkg_name in ("public-nuextract3-mlx-vlm", "fast-gemma4-vision") or "nuextract" in ocr_model_low:
            requested = ["vrdu_ad_buy", "vrdu_registration", "doclaynet_gov", "us_gov_ocr"]
        else:
            requested = ["doclaynet_gov", "us_gov_ocr"]
    elif raw == "gov-all":
        requested = ["vrdu_ad_buy", "vrdu_registration", "doclaynet_gov", "us_gov_ocr"]
    elif raw == "default":
        if pkg_name == "public-pypdfium2":
            requested = ["olmocr_bench"]
        elif pkg_name == "fast-gemma4-vision":
            requested = ["olmocr_bench", "fintabnet_otsl", "omnidocbench_en", "cord_v2"]
        elif pkg_name == "public-nuextract3-mlx-vlm" or "nuextract" in ocr_model_low:
            requested = ["cord_v2", "olmocr_bench", "fintabnet_otsl"]
        else:
            # Default Markdown/Table/Formula OCR families (`OvisOCR2`, `GLM-OCR`, `LightOnOCR-2`,
            # `DeepSeek-OCR-2`, `PaddleOCR-VL`, `Nemotron-Parse`, `Apple Vision`, `RapidOCR`,
            # `Tesseract`, `Docling`, `Marker`). Pass `--ocr-suites all|gov|gov-all` for JSON KIE / Gov suites.
            requested = ["olmocr_bench", "fintabnet_otsl", "omnidocbench_en"]
    else:
        requested = [tok.strip() for tok in raw.split(",") if tok.strip()]
        for tok in requested:
            if tok not in _CAT5_ALL_SUITES:
                raise ValueError(f"Unknown --ocr-suites entry {tok!r}; expected subset of {_CAT5_ALL_SUITES}")
    if pkg_name == "public-pypdfium2":
        requested = [s for s in requested if s in _CAT5_PDF_NATIVE_SUITES]
    return requested


def _render_pdf_page_pil(pdf_source: str | Path | bytes, *, dpi: int, page_index: int = 0) -> Any:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_source if isinstance(pdf_source, (bytes, bytearray)) else str(pdf_source))
    try:
        idx = max(0, min(int(page_index), len(doc) - 1))
        page = doc[idx]
        bitmap = page.render(scale=float(dpi) / 72.0)
        pil = bitmap.to_pil().convert("RGB")
        page.close()
        return pil
    finally:
        doc.close()


def _render_pdf_first_page_pil(pdf_path: str | Path, *, dpi: int) -> Any:
    return _render_pdf_page_pil(pdf_path, dpi=dpi, page_index=0)


def _bytes_to_pil(image_bytes: bytes) -> Any:
    import io
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        return im.convert("RGB")


def _build_cord_v2_prompt(schema: Mapping[str, Any]) -> str:
    compact = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    return (
        "Extract the receipt fields from this image as a single valid JSON object conforming strictly to "
        f"the following JSON Schema. Omit or set null for fields not visible on the receipt. Output ONLY JSON.\n"
        f"Schema: {compact}"
    )


def _build_vrdu_prompt(schema: Mapping[str, Any], corpus_label: str) -> str:
    compact = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    return (
        f"Extract the {corpus_label} fields visible on this document page as a single valid JSON object "
        f"conforming strictly to the following JSON Schema. Omit or set null for fields not visible on this page. "
        f"Output ONLY JSON.\nSchema: {compact}"
    )


def _resolve_fast_ocr_model_arg(ocr_model: str | None) -> str | Path | None:
    if not ocr_model:
        return None
    low = str(ocr_model).strip().lower()
    if low in ("nuextract", "nuextract3"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from public_baselines import resolve_hf_snapshot

        for cand in ("numind/NuExtract3-mlx-4bits", "mlx-community/NuExtract3-bf16"):
            snap = resolve_hf_snapshot(cand)
            if snap is not None and (snap / "config.json").is_file():
                sfiles = list(snap.glob("*.safetensors"))
                if sfiles and all(f.exists() and f.stat().st_size > 0 for f in sfiles):
                    return snap
    return ocr_model


def _summarize_cat5_rates(
    metas: Sequence[Mapping[str, Any]],
    *,
    n_pages: int,
    wall_sec: float,
    default_family: str,
) -> dict[str, Any]:
    """Aggregate page throughput and token generation rates for a Category 5 benchmark record."""
    n = max(1, int(n_pages))
    wall = max(float(wall_sec), 1e-6)
    gen_toks = [int(m.get("generated_tokens", 0) or 0) for m in metas]
    prompt_toks = [int(m.get("prompt_tokens", 0) or 0) for m in metas]
    vis_toks = [int(m.get("visual_tokens", 0) or 0) for m in metas]
    total_gen = int(sum(gen_toks))
    tokens_per_page = round(float(total_gen) / float(n), 2)
    decode_ms_sum = sum(float(m.get("decode_ms", 0.0) or 0.0) for m in metas)
    pos_tps = [float(m.get("decode_tps", 0.0) or 0.0) for m in metas if float(m.get("decode_tps", 0.0) or 0.0) > 0.0]
    if total_gen > 0 and decode_ms_sum > 0.0:
        decode_tok_s = round(float(total_gen) / (decode_ms_sum / 1000.0), 2)
    elif pos_tps:
        decode_tok_s = round(float(np.mean(pos_tps)), 2)
    else:
        decode_tok_s = 0.0
    e2e_tok_s = round(float(total_gen) / wall, 2) if total_gen > 0 else 0.0
    family = next((str(m.get("model_family")) for m in metas if m.get("model_family")), default_family)
    backends_used = sorted({str(m.get("backend_used") or "") for m in metas if m.get("backend_used")})
    return {
        "model_family": family,
        "backends_used": backends_used,
        "pages_per_sec": round(float(n_pages) / wall, 4),
        "sec_per_page": round(wall / float(n), 4),
        "total_generated_tokens": total_gen,
        "tokens_per_page": tokens_per_page,
        "avg_generated_tokens": tokens_per_page,
        "prompt_tokens_per_page": round(float(sum(prompt_toks)) / float(n), 2),
        "avg_visual_tokens": round(float(sum(vis_toks)) / float(n), 2),
        "decode_tok_s": decode_tok_s,
        "e2e_tok_s": e2e_tok_s,
    }


def _extract_cat5_rates(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Extract normalized Category 5 page/token rates from a saved or live outcome record."""
    diag = rec.get("diagnostics") or {}
    n_items = int(rec.get("n_items", 0) or 0)
    wall = float(rec.get("wall_sec", 0.0) or 0.0)
    pps = diag.get("pages_per_sec")
    if pps is None:
        raw_ips = rec.get("raw_items_per_sec")
        if raw_ips is not None:
            pps = float(raw_ips)
        elif n_items > 0 and wall > 0.0:
            pps = float(n_items) / wall
        else:
            pps = 0.0
    spp = diag.get("sec_per_page")
    if spp is None:
        spp = (wall / float(n_items)) if (n_items > 0 and wall > 0.0) else ((1.0 / float(pps)) if float(pps) > 0.0 else 0.0)
    tpp = diag.get("tokens_per_page")
    if tpp is None:
        tpp = float(diag.get("avg_generated_tokens", 0.0) or 0.0)
    tot_tok = diag.get("total_generated_tokens")
    if tot_tok is None:
        tot_tok = int(round(float(tpp) * n_items)) if (float(tpp) > 0.0 and n_items > 0) else 0
    dec_tps = float(diag.get("decode_tok_s", 0.0) or 0.0)
    e2e_tps = diag.get("e2e_tok_s")
    if e2e_tps is None:
        e2e_tps = (float(tot_tok) / wall) if (int(tot_tok) > 0 and wall > 0.0) else 0.0
    mem_gb = float((rec.get("memory_4bucket_gb") or {}).get("combined_with_model_mapped_gb", 0.0) or 0.0)
    return {
        "model_family": str(diag.get("model_family") or diag.get("backend") or rec.get("package") or "-"),
        "pages_per_sec": round(float(pps), 4),
        "sec_per_page": round(float(spp), 4),
        "total_generated_tokens": int(tot_tok),
        "tokens_per_page": round(float(tpp), 2),
        "decode_tok_s": round(float(dec_tps), 2),
        "e2e_tok_s": round(float(e2e_tps), 2),
        "mem_gb": round(mem_gb, 4),
    }


def _run_cat5_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
) -> list[dict[str, Any]]:
    from PIL import Image

    model_dirs: list[Path] = [home / "artifacts"]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _outcome_public_backends import build_public_cat5_adapter, stock_hf_model_files

    if pkg_name == "fast-ocr":
        from fast_ocr.ovis_engine import resolve_ocr_model_dir

        if extra.ocr_backend != "apple-vision":
            try:
                mdir = resolve_ocr_model_dir(_resolve_fast_ocr_model_arg(extra.ocr_model), local_files_only=True)
                model_dirs.extend([p.resolve() for p in mdir.rglob("*") if p.is_file()])
            except Exception:
                pass
    elif pkg_name == "fast-gemma4-vision":
        if extra.model_dir:
            mdir = Path(extra.model_dir).expanduser().resolve()
            if mdir.is_dir():
                model_dirs.append(mdir)
    else:
        model_dirs.extend(stock_hf_model_files(pkg_name))

    probe = FourBucketMemoryProbe(model_directories=model_dirs)
    guard = DeviceMemoryGuard()
    guard_tripped = False

    suites = _resolve_cat5_suites(pkg_name, extra)
    if not suites:
        return []
    raw_ocr_suites = str(getattr(extra, "ocr_suites", "default") or "default").strip().lower()
    optional_on_all = raw_ocr_suites == "all"

    olm_cap = 4 if smoke else max_items
    ftn_cap = 2 if smoke else max_items
    od_cap = 2 if smoke else max_items
    cord_cap = 2 if smoke else max_items
    gov_cap = 2 if smoke else max_items

    loaded: dict[str, list[dict[str, Any]]] = {}
    if "olmocr_bench" in suites:
        loaded["olmocr_bench"] = load_olmocr_bench(
            tier=tier,
            max_pages=olm_cap,
            download_pdfs=True,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if "fintabnet_otsl" in suites:
        loaded["fintabnet_otsl"] = load_fintabnet_otsl(
            tier=tier,
            max_items=ftn_cap,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if "omnidocbench_en" in suites:
        loaded["omnidocbench_en"] = load_omnidocbench_en(
            tier=tier,
            max_items=od_cap,
            download_images=True,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if "cord_v2" in suites:
        loaded["cord_v2"] = load_cord_v2(
            tier=tier,
            max_items=cord_cap,
            lock=lock,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    for vrdu_key in ("vrdu_ad_buy", "vrdu_registration"):
        if vrdu_key in suites:
            loaded[vrdu_key] = load_vrdu(
                vrdu_key,
                tier=tier,
                max_items=gov_cap,
                download_pdfs=True,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
    if "doclaynet_gov" in suites:
        try:
            dl_items = load_doclaynet_gov(
                tier="tier2" if tier == "tier1" else tier,
                max_items=gov_cap,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=True if (optional_on_all and tier == "tier1") else local_files_only,
            )
            if dl_items:
                loaded["doclaynet_gov"] = dl_items
        except FileNotFoundError as exc:
            if not (optional_on_all and tier == "tier1"):
                raise
            print(f"[fast-silicon] skipping optional doclaynet_gov on tier1: {exc}", file=sys.stderr)
    if "puredocbench_legal_gov" in suites:
        try:
            pdb_items = load_puredocbench_legal_gov(
                track=str(getattr(extra, "puredocbench_track", "clean") or "clean"),
                max_items=gov_cap,
                images_root=getattr(extra, "puredocbench_images_root", None),
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            ready_pdb = [it for it in pdb_items if it.get("image_path") and Path(str(it["image_path"])).is_file()]
            if not ready_pdb:
                raise FileNotFoundError(
                    "PureDocBench legal_gov page images are missing; extract pdb_full.tar and pass "
                    "`--puredocbench-images-root <dir>` or place images under `cache_dir/puredocbench_legal_gov/images`."
                )
            loaded["puredocbench_legal_gov"] = ready_pdb
        except FileNotFoundError as exc:
            if not optional_on_all:
                raise
            print(f"[fast-silicon] skipping optional puredocbench_legal_gov: {exc}", file=sys.stderr)
    if "puredocbench_en" in suites:
        try:
            loaded["puredocbench_en"] = load_puredocbench_en(
                track=str(getattr(extra, "puredocbench_track", "clean") or "clean"),
                max_items=gov_cap if smoke else max_items,
                en_root=getattr(extra, "puredocbench_en_root", None),
                images_root=getattr(extra, "puredocbench_images_root", None),
                cache_dir=cache_dir,
            )
        except FileNotFoundError as exc:
            if not optional_on_all:
                raise
            print(f"[fast-silicon] skipping optional puredocbench_en: {exc}", file=sys.stderr)
    if "us_gov_ocr" in suites:
        try:
            gov_items = load_us_gov_ocr(
                max_items=gov_cap,
                cache_dir=cache_dir,
            )
            if gov_items:
                loaded["us_gov_ocr"] = gov_items
        except FileNotFoundError as exc:
            if not optional_on_all:
                raise
            print(f"[fast-silicon] skipping optional us_gov_ocr: {exc}", file=sys.stderr)
    if not loaded:
        return []

    fast_eng: Any = None
    gemma4_eng: Any = None
    pub_ad: Any = None
    eff_backend: str
    eff_label: str
    default_family: str
    allow_html_tables = (extra.table_mode == "html") or (extra.output_format != "markdown")
    if pkg_name == "fast-ocr":
        from fast_ocr import FastOCR

        eff_backend = str(extra.ocr_backend)
        model_arg = _resolve_fast_ocr_model_arg(extra.ocr_model)
        eff_ocr_kv = (
            getattr(extra, "ocr_kv_i8", None)
            if getattr(extra, "ocr_kv_i8", None) is not None
            else getattr(extra, "kv_mode", None)
        )
        fast_kw: dict[str, Any] = {
            "backend": eff_backend,
            "output_format": extra.output_format,
            "table_mode": extra.table_mode,
            "strict_markdown_grammar": bool(extra.strict_markdown_grammar),
            "model_path": model_arg,
            "crop_margins": not bool(extra.no_crop_margins),
            "auto_orient": not bool(extra.no_auto_orient),
            "speculative": extra.speculative,
            "vocab_size": extra.ocr_vocab_size,
            "max_tokens": int(extra.ocr_max_tokens),
        }
        if bool(getattr(extra, "no_adaptive_resolution", False)):
            fast_kw["adaptive_resolution"] = False
        if getattr(extra, "deskew", None) is not None:
            fast_kw["deskew"] = str(extra.deskew)
        if bool(getattr(extra, "region_routing", False)):
            fast_kw["region_routing"] = True
        if bool(getattr(extra, "zoom_recover", False)):
            fast_kw["zoom_recover"] = True
        if eff_ocr_kv is not None:
            fast_kw["kv_i8"] = eff_ocr_kv
        if getattr(extra, "ocr_gdn_state_mode", None) is not None:
            fast_kw["gdn_state_mode"] = str(extra.ocr_gdn_state_mode)
        if bool(getattr(extra, "drop_white_patches", False)):
            fast_kw["drop_white_patches"] = True
        if bool(getattr(extra, "no_latin_only", False)):
            fast_kw["latin_only"] = False
        fast_eng = FastOCR(**fast_kw)
        mode_tag = f"{eff_backend}_{extra.table_mode}"
        if extra.ocr_render_only:
            mode_tag += "_render"
        if extra.ocr_model:
            mode_tag += f"_{Path(str(extra.ocr_model)).name}"
        eff_label = config_label or mode_tag
        if eff_backend == "apple-vision":
            default_family = "apple-vision-ane"
        elif "nuextract" in str(extra.ocr_model or "").lower():
            default_family = "qwen3.5-nuextract3"
        elif eff_backend == "auto":
            default_family = "qwen3.5-ovis+ane"
        else:
            default_family = "qwen3.5-ovis"
    elif pkg_name == "fast-gemma4-vision":
        from fast_gemma4.engine import FastGemma4Engine

        eff_mode = extra.mode or "gpu"
        g4_kw: dict[str, Any] = {
            "home": home,
            "model_dir": extra.model_dir,
            "default_mode": eff_mode,
            "local_files_only": True,
            "warmup": False,
            "warmup_ane": False,
        }
        if extra.gemma_vocab_out is not None:
            g4_kw["vocab_out"] = extra.gemma_vocab_out
        if extra.kv_i8 is not None or extra.no_kv_i8:
            g4_kw["kv_i8"] = False if extra.no_kv_i8 else bool(extra.kv_i8)
        gemma4_eng = FastGemma4Engine(**g4_kw)
        eff_backend = "fast-gemma4-vision"
        soft_tag = f"_s{int(extra.max_soft_tokens)}" if getattr(extra, "max_soft_tokens", None) else ""
        mdl_tag = f"_{Path(str(extra.model_dir)).name}" if extra.model_dir else ""
        eff_label = config_label or f"gemma4_vision_{extra.table_mode}{soft_tag}{mdl_tag}"
        default_family = "gemma4-vision"
    else:
        pub_ad = build_public_cat5_adapter(
            pkg_name,
            table_mode=extra.table_mode,
            output_format=extra.output_format,
            max_tokens=int(extra.ocr_max_tokens),
        )
        eff_backend = pkg_name
        eff_label = config_label or pub_ad.default_label
        default_family = pub_ad.model_family

    def _fast_result_meta(res: Any) -> dict[str, Any]:
        return {
            "backend_used": res.backend_used,
            "model_family": default_family,
            "generated_tokens": int(res.timings.generated_tokens),
            "prompt_tokens": int(res.timings.prompt_tokens),
            "visual_tokens": int(res.timings.visual_tokens),
            "decode_tps": float(res.timings.decode_tps),
            "decode_ms": float(res.timings.decode_ms),
            "vit_ms": float(res.timings.vit_ms),
            "prefill_ms": float(res.timings.prefill_ms),
        }

    def _gemma4_result_meta(item: Any, stats: Mapping[str, Any] | None) -> dict[str, Any]:
        st = dict(stats or {})
        return {
            "backend_used": "fast-gemma4-vision",
            "model_family": default_family,
            "generated_tokens": int(st.get("completion_tokens", getattr(item, "completion_tokens", 0)) or 0),
            "prompt_tokens": int(st.get("prompt_tokens", getattr(item, "prompt_tokens", 0)) or 0),
            "visual_tokens": int(st.get("vision_tokens", 0) or 0),
            "used_speculative": bool(st.get("used_speculative", False)),
            "speculative_backend": st.get("speculative_backend"),
            "decode_tps": float(st.get("decode_tok_s", 0.0) or 0.0),
            "decode_ms": float(st.get("decode_sec", 0.0) or 0.0) * 1000.0,
            "vit_ms": float(st.get("vision_sec", 0.0) or 0.0) * 1000.0,
            "prefill_ms": float(st.get("prefill_sec", 0.0) or 0.0) * 1000.0,
        }

    def _default_gemma4_ocr_prompt(task_kind: str) -> str:
        if task_kind == "table":
            if extra.table_mode == "html":
                return "Transcribe the table in this image as clean HTML <table>...</table> preserving every cell and number verbatim."
            return "Transcribe the table in this image as a GitHub-Flavored Markdown pipe table preserving every cell and number verbatim."
        if extra.table_mode == "html":
            return (
                "Extract all readable content from this page image in natural reading order as Markdown. "
                "Format tables as HTML <table>...</table> and formulas as LaTeX. Preserve all text and numbers verbatim."
            )
        return (
            "Extract all readable content from this page image in natural reading order as GitHub-Flavored Markdown. "
            "Format tables as GFM pipe tables and formulas as LaTeX. Preserve all text and numbers verbatim."
        )

    def _run_image_ocr(
        pil_img: Any,
        *,
        prompt: str | None = None,
        schema: Mapping[str, Any] | None = None,
        max_tokens_override: int | None = None,
        task_kind: str = "page",
    ) -> tuple[str, str, dict[str, Any]]:
        if fast_eng is not None:
            if schema is not None or prompt is not None:
                if eff_backend == "apple-vision":
                    res = fast_eng.ocr_image(pil_img)
                else:
                    ovis = fast_eng._get_ovis_engine()
                    res = ovis.ocr_image(
                        pil_img,
                        prompt=prompt,
                        schema=schema,
                        output_format="raw" if schema is not None else extra.output_format,
                        table_mode=extra.table_mode,
                        max_tokens=max_tokens_override or int(extra.ocr_max_tokens),
                    )
            else:
                res = fast_eng.ocr_image(pil_img)
            return res.text, res.finish_reason, _fast_result_meta(res)
        if gemma4_eng is not None:
            eff_prompt = prompt if prompt is not None else _default_gemma4_ocr_prompt(task_kind)
            eff_schema = None if (schema is None or bool(extra.no_llguidance)) else dict(schema)
            item = gemma4_eng.generate(
                eff_prompt,
                images=[pil_img],
                json_schema=eff_schema,
                max_tokens=max_tokens_override or int(extra.ocr_max_tokens),
                temperature=0.0,
                enable_thinking=False,
                max_soft_tokens=getattr(extra, "max_soft_tokens", None),
            )
            stats = getattr(gemma4_eng, "last_multimodal_stats", None)
            return item.text, item.finish_reason, _gemma4_result_meta(item, stats)
        assert pub_ad is not None
        return pub_ad.ocr_image_fn(
            pil_img,
            prompt=prompt,
            schema=schema,
            max_tokens_override=max_tokens_override,
            task_kind=task_kind,
        )

    def _run_image_list_ocr(
        pil_imgs: Sequence[Any],
        *,
        suite_name: str = "",
        prompts: Sequence[str | None] | None = None,
        schemas: Sequence[Mapping[str, Any] | None] | None = None,
        max_tokens_override: int | None = None,
        task_kind: str = "page",
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        suite_bs = _resolve_cat5_suite_batch_size(suite_name, extra)
        if fast_eng is not None and suite_bs > 1 and len(pil_imgs) > 1:
            if schemas is not None:
                if eff_backend != "apple-vision" and callable(getattr(fast_eng, "_get_ovis_engine", None)):
                    ovis = fast_eng._get_ovis_engine()
                    ovis_batch = getattr(ovis, "ocr_images", None)
                    if callable(ovis_batch) and accepts_kwarg(ovis_batch, "schemas"):
                        outs = ovis_batch(
                            pil_imgs,
                            prompts=list(prompts) if prompts is not None else None,
                            schemas=list(schemas),
                            output_format="raw",
                            table_mode=extra.table_mode,
                            max_tokens=max_tokens_override or int(extra.ocr_max_tokens),
                            batch_size=suite_bs,
                        )
                        return (
                            [r.text for r in outs],
                            [r.finish_reason for r in outs],
                            [_fast_result_meta(r) for r in outs],
                        )
            else:
                outs = fast_eng.ocr_images(pil_imgs, batch_size=suite_bs)
                return (
                    [r.text for r in outs],
                    [r.finish_reason for r in outs],
                    [_fast_result_meta(r) for r in outs],
                )
        preds_l: list[str] = []
        reasons_l: list[str] = []
        metas_l: list[dict[str, Any]] = []
        prompt_seq = list(prompts) if prompts is not None else [None] * len(pil_imgs)
        schema_seq = list(schemas) if schemas is not None else [None] * len(pil_imgs)
        for pil_img, pr_i, sc_i in zip(pil_imgs, prompt_seq, schema_seq):
            tx, fr, md = _run_image_ocr(
                pil_img,
                prompt=pr_i,
                schema=sc_i,
                max_tokens_override=max_tokens_override,
                task_kind=task_kind,
            )
            preds_l.append(tx)
            reasons_l.append(fr)
            metas_l.append(md)
        return preds_l, reasons_l, metas_l

    def _run_pdf_page_ocr(
        pdf_path: str | Path,
        *,
        page_index: int = 0,
        task_kind: str = "page",
    ) -> tuple[str, str, dict[str, Any]]:
        if fast_eng is not None:
            use_native = (not bool(extra.ocr_render_only)) and (eff_backend == "auto")
            if use_native:
                pdf_kw: dict[str, Any] = {
                    "dpi": int(extra.ocr_dpi),
                    "page_indices": [int(page_index)],
                    "use_native_text": True,
                }
                if getattr(extra, "triage_dpi", None) is not None:
                    pdf_kw["triage_dpi"] = int(extra.triage_dpi)
                outs = fast_eng.ocr_pdf(pdf_path, **pdf_kw)
                res = outs[0]
                return res.text, res.finish_reason, _fast_result_meta(res)
            pil_img = _render_pdf_page_pil(pdf_path, dpi=int(extra.ocr_dpi), page_index=page_index)
            return _run_image_ocr(pil_img, task_kind=task_kind)
        if gemma4_eng is not None:
            pil_img = _render_pdf_page_pil(pdf_path, dpi=int(extra.ocr_dpi), page_index=page_index)
            return _run_image_ocr(pil_img, task_kind=task_kind)
        assert pub_ad is not None
        if pub_ad.ocr_pdf_page_fn is not None and (not bool(extra.ocr_render_only) or not pub_ad.supports_images):
            return pub_ad.ocr_pdf_page_fn(pdf_path, page_index=int(page_index))
        pil_img = _render_pdf_page_pil(pdf_path, dpi=int(extra.ocr_dpi), page_index=page_index)
        return pub_ad.ocr_image_fn(pil_img, task_kind=task_kind)

    records: list[dict[str, Any]] = []
    try:
        guard.arm()
        # Untimed tiny warmup for VLM/ANE paths so model compilation & weights load outside EOT timing.
        if fast_eng is not None:
            tiny = Image.new("RGB", (64, 64), (255, 255, 255))
            if eff_backend == "ovis":
                _ = fast_eng._get_ovis_engine().ocr_image(
                    tiny, output_format="raw", crop_margins=False, auto_orient=False, speculative="off", max_tokens=2
                )
            elif eff_backend == "apple-vision":
                _ = fast_eng.ocr_image(tiny)
            else:
                _ = fast_eng._get_vision_ocr().recognize_image(tiny, output_format="raw")
                if any(s in suites for s in _CAT5_ALL_SUITES):
                    _ = fast_eng._get_ovis_engine().ocr_image(
                        tiny, output_format="raw", crop_margins=False, auto_orient=False, speculative="off", max_tokens=2
                    )
        elif gemma4_eng is not None:
            tiny = Image.new("RGB", (64, 64), (255, 255, 255))
            _ = gemma4_eng.generate(
                "Transcribe any text.",
                images=[tiny],
                max_tokens=2,
                temperature=0.0,
                enable_thinking=False,
                max_soft_tokens=getattr(extra, "max_soft_tokens", None),
            )
        elif pub_ad is not None and pub_ad.supports_images:
            tiny = Image.new("RGB", (64, 64), (255, 255, 255))
            _ = pub_ad.ocr_image_fn(tiny, max_tokens_override=2, task_kind="page")
        _mx_sync()
        _mx_reset_peak()

        # Pre-decode/rasterize inputs outside timed inference strictly when --ocr-render-only is set
        # or on image-only suites so PNG/Parquet decompression isn't conflated with inference latency.
        for suite_name in suites:
            items = loaded.get(suite_name) or []
            if not items:
                continue
            _mx_reset_peak()

            if suite_name == "olmocr_bench":
                pre_imgs: list[Any] | None = None
                if bool(extra.ocr_render_only) and (pub_ad is None or pub_ad.supports_images):
                    pre_imgs = [_render_pdf_first_page_pil(it["pdf_path"], dpi=int(extra.ocr_dpi)) for it in items]
                t0 = time.perf_counter()
                if pre_imgs is not None:
                    preds, reasons, metas = _run_image_list_ocr(pre_imgs, suite_name=suite_name, task_kind="page")
                    del pre_imgs
                    gc.collect()
                else:
                    preds, reasons, metas = [], [], []
                    for it in items:
                        tx, fr, md = _run_pdf_page_ocr(it["pdf_path"], task_kind="page")
                        preds.append(tx)
                        reasons.append(fr)
                        metas.append(md)
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_olmocr_bench(
                    items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "render_only": bool(extra.ocr_render_only),
                    "dpi": int(extra.ocr_dpi),
                    "speculative": (
                        next((str(m["speculative_backend"]) for m in metas if m.get("speculative_backend")), "mtp")
                        if any(m.get("used_speculative") for m in metas)
                        else "off"
                    )
                    if pkg_name == "fast-gemma4-vision"
                    else (extra.speculative if pkg_name == "fast-ocr" else "off"),
                    "crop_margins": (not bool(extra.no_crop_margins)) if pkg_name == "fast-ocr" else False,
                    "auto_orient": (not bool(extra.no_auto_orient)) if pkg_name == "fast-ocr" else False,
                    "overall_macro": ev["overall_macro"],
                    "overall_no_hf": ev["overall_no_hf"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"olmOCR-bench({len(items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks=ev["per_category"],
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name == "fintabnet_otsl":
                pils = [_bytes_to_pil(it["image_bytes"]) for it in items]
                t0 = time.perf_counter()
                preds, reasons, metas = _run_image_list_ocr(pils, suite_name=suite_name, task_kind="table")
                del pils
                gc.collect()
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_fintabnet_otsl(
                    items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "teds": ev["teds"],
                    "teds_s": ev["teds_s"],
                    "grits_top": ev["grits_top"],
                    "grits_con": ev["grits_con"],
                    "numeric_cell_f1": ev["numeric_cell_f1"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"FinTabNet-OTSL({len(items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            "table_metrics": {
                                "teds": ev["teds"],
                                "teds_s": ev["teds_s"],
                                "grits_top": ev["grits_top"],
                                "grits_con": ev["grits_con"],
                                "numeric_cell_f1": ev["numeric_cell_f1"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name == "omnidocbench_en":
                pils = []
                for it in items:
                    with Image.open(it["image_path"]) as im:
                        pils.append(im.convert("RGB"))
                t0 = time.perf_counter()
                preds, reasons, metas = _run_image_list_ocr(pils, suite_name=suite_name, task_kind="page")
                del pils
                gc.collect()
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_omnidocbench(
                    items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                vocab_desc = (
                    f"custom_{extra.ocr_vocab_size}"
                    if extra.ocr_vocab_size
                    else ("full_248k" if bool(getattr(extra, "no_latin_only", False)) else "ocr_latin_140k")
                )
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "vocab_mode": vocab_desc if (pkg_name == "fast-ocr" and eff_backend != "apple-vision") else "n/a",
                    "omnidoc_overall": ev["omnidoc_overall"],
                    "text_edit_distance": ev["text_edit_distance"],
                    "reading_order_sim": ev["reading_order_sim"],
                    "table_teds": ev["table_teds"],
                    "formula_norm_sim": ev["formula_norm_sim"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"OmniDocBench-EN({len(items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            "omnidoc_pillars": {
                                "omnidoc_overall": ev["omnidoc_overall"],
                                "text_similarity": ev["text_similarity"],
                                "text_edit_distance": ev["text_edit_distance"],
                                "reading_order_sim": ev["reading_order_sim"],
                                "table_teds": ev["table_teds"],
                                "formula_norm_sim": ev["formula_norm_sim"],
                                "n_table_pages": ev["n_table_pages"],
                                "n_formula_pages": ev["n_formula_pages"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name == "cord_v2":
                pils = [_bytes_to_pil(it["image_bytes"]) for it in items]
                prompts = [_build_cord_v2_prompt(it["schema"]) for it in items]
                schemas = [it["schema"] for it in items]
                t0 = time.perf_counter()
                preds, reasons, metas = _run_image_list_ocr(
                    pils,
                    suite_name=suite_name,
                    prompts=prompts,
                    schemas=schemas,
                    max_tokens_override=min(1024, int(extra.ocr_max_tokens)),
                    task_kind="page",
                )
                del pils
                gc.collect()
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_cord_v2(items, preds, finish_reasons=reasons)
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "llguidance": bool(
                        (pkg_name == "fast-ocr" and eff_backend != "apple-vision")
                        or (pkg_name == "fast-gemma4-vision" and not bool(extra.no_llguidance))
                    ),
                    "grounded_field_f1": ev["grounded_field_f1"],
                    "anls": ev["anls"],
                    "valid_json_rate": ev["valid_json_rate"],
                    "valid_schema_rate": ev["valid_schema_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"CORD-v2({len(items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            "cord_v2": {
                                "grounded_field_f1": ev["grounded_field_f1"],
                                "anls": ev["anls"],
                                "valid_json_rate": ev["valid_json_rate"],
                                "valid_schema_rate": ev["valid_schema_rate"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name in ("vrdu_ad_buy", "vrdu_registration"):
                corpus_label = "FCC political ad-buy invoice" if suite_name == "vrdu_ad_buy" else "DOJ FARA registration form"
                bench_label = f"VRDU-AdBuy({len(items)})" if suite_name == "vrdu_ad_buy" else f"VRDU-Registration({len(items)})"
                pils = [
                    _render_pdf_page_pil(it["pdf_path"], dpi=int(extra.ocr_dpi), page_index=int(it.get("page_index", 0)))
                    for it in items
                ]
                prompts = [_build_vrdu_prompt(it["schema"], corpus_label) for it in items]
                schemas = [it["schema"] for it in items]
                t0 = time.perf_counter()
                preds, reasons, metas = _run_image_list_ocr(
                    pils,
                    suite_name=suite_name,
                    prompts=prompts,
                    schemas=schemas,
                    max_tokens_override=min(1536, int(extra.ocr_max_tokens)),
                    task_kind="page",
                )
                del pils
                gc.collect()
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_vrdu(items, preds, finish_reasons=reasons)
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "llguidance": bool(
                        (pkg_name == "fast-ocr" and eff_backend != "apple-vision")
                        or (pkg_name == "fast-gemma4-vision" and not bool(extra.no_llguidance))
                    ),
                    "grounded_field_f1": ev["grounded_field_f1"],
                    "anls": ev["anls"],
                    "valid_json_rate": ev["valid_json_rate"],
                    "valid_schema_rate": ev["valid_schema_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=bench_label,
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            suite_name: {
                                "grounded_field_f1": ev["grounded_field_f1"],
                                "anls": ev["anls"],
                                "valid_json_rate": ev["valid_json_rate"],
                                "valid_schema_rate": ev["valid_schema_rate"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name == "doclaynet_gov":
                import tempfile

                pdf_only_ad = pub_ad is not None and not pub_ad.supports_images
                suite_items = [it for it in items if it.get("pdf_bytes")] if pdf_only_ad else list(items)
                if not suite_items:
                    continue
                use_native_pdf = pdf_only_ad or (
                    fast_eng is not None and (not bool(extra.ocr_render_only)) and (eff_backend == "auto")
                )
                pre_imgs = None
                if not use_native_pdf:
                    pre_imgs = [
                        _render_pdf_page_pil(it["pdf_bytes"], dpi=int(extra.ocr_dpi), page_index=0)
                        if (bool(extra.ocr_render_only) and it.get("pdf_bytes"))
                        else _bytes_to_pil(it["image_bytes"])
                        for it in suite_items
                    ]
                t0 = time.perf_counter()
                if pre_imgs is not None:
                    preds, reasons, metas = _run_image_list_ocr(pre_imgs, suite_name=suite_name, task_kind="page")
                    del pre_imgs
                    gc.collect()
                else:
                    preds, reasons, metas = [], [], []
                    with tempfile.TemporaryDirectory(prefix="doclaynet_gov_") as tmp_dir:
                        for idx_p, it in enumerate(suite_items):
                            if it.get("pdf_bytes"):
                                tmp_pdf = Path(tmp_dir) / f"page_{idx_p}.pdf"
                                tmp_pdf.write_bytes(it["pdf_bytes"])
                                tx, fr, md = _run_pdf_page_ocr(tmp_pdf, page_index=0, task_kind="page")
                            else:
                                tx, fr, md = _run_image_ocr(_bytes_to_pil(it["image_bytes"]), task_kind="page")
                            preds.append(tx)
                            reasons.append(fr)
                            metas.append(md)
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_doclaynet_gov(
                    suite_items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(suite_items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "render_only": bool(extra.ocr_render_only),
                    "doclaynet_overall": ev["doclaynet_overall"],
                    "text_edit_distance": ev["text_edit_distance"],
                    "reading_order_sim": ev["reading_order_sim"],
                    "header_footer_leak_rate": ev["header_footer_leak_rate"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"DocLayNet-Gov({len(suite_items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            "doclaynet_gov": {
                                "doclaynet_overall": ev["doclaynet_overall"],
                                "text_similarity": ev["text_similarity"],
                                "reading_order_sim": ev["reading_order_sim"],
                                "header_footer_leak_rate": ev["header_footer_leak_rate"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name in ("puredocbench_legal_gov", "puredocbench_en"):
                pils = []
                for it in items:
                    with Image.open(it["image_path"]) as im:
                        pils.append(im.convert("RGB"))
                t0 = time.perf_counter()
                preds, reasons, metas = _run_image_list_ocr(pils, suite_name=suite_name, task_kind="page")
                del pils
                gc.collect()
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_puredocbench(
                    items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "puredocbench_track": str(getattr(extra, "puredocbench_track", "clean") or "clean"),
                    "puredocbench_overall": ev["puredocbench_overall"],
                    "text_edit_distance": ev["text_edit_distance"],
                    "reading_order_sim": ev["reading_order_sim"],
                    "table_teds": ev["table_teds"],
                    "formula_norm_sim": ev["formula_norm_sim"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                bench_prefix = "PureDocBench-EN" if suite_name == "puredocbench_en" else "PureDocBench-LegalGov"
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"{bench_prefix}({len(items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            suite_name: {
                                "puredocbench_overall": ev["puredocbench_overall"],
                                "text_similarity": ev["text_similarity"],
                                "reading_order_sim": ev["reading_order_sim"],
                                "table_teds": ev["table_teds"],
                                "formula_norm_sim": ev["formula_norm_sim"],
                            }
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )

            elif suite_name == "us_gov_ocr":
                suite_items = (
                    [it for it in items if it.get("pdf_path")]
                    if (pub_ad is not None and not pub_ad.supports_images)
                    else list(items)
                )
                if not suite_items:
                    continue
                pre_imgs = None
                if bool(extra.ocr_render_only) and (pub_ad is None or pub_ad.supports_images):
                    pre_imgs = []
                    for it in suite_items:
                        if it.get("pdf_path"):
                            pre_imgs.append(_render_pdf_page_pil(it["pdf_path"], dpi=int(extra.ocr_dpi), page_index=int(it.get("page_index", 0))))
                        else:
                            with Image.open(it["image_path"]) as im:
                                pre_imgs.append(im.convert("RGB"))
                t0 = time.perf_counter()
                if pre_imgs is not None:
                    preds, reasons, metas = _run_image_list_ocr(pre_imgs, suite_name=suite_name, task_kind="page")
                    del pre_imgs
                else:
                    preds, reasons, metas = [], [], []
                    for it in suite_items:
                        if it.get("pdf_path"):
                            tx, fr, md = _run_pdf_page_ocr(
                                it["pdf_path"],
                                page_index=int(it.get("page_index", 0)),
                                task_kind="page",
                            )
                        else:
                            with Image.open(it["image_path"]) as im:
                                tx, fr, md = _run_image_ocr(im.convert("RGB"), task_kind="page")
                        preds.append(tx)
                        reasons.append(fr)
                        metas.append(md)
                _mx_sync()
                wall = time.perf_counter() - t0
                _, ref = guard.check()
                if ref is not None:
                    guard_tripped = True
                ev = evaluate_us_gov_ocr(
                    suite_items,
                    preds,
                    finish_reasons=reasons,
                    allow_html_tables=allow_html_tables or (pkg_name != "fast-ocr"),
                )
                mem4 = _memory_4bucket_dict(probe)
                rate_diag = _summarize_cat5_rates(metas, n_pages=len(suite_items), wall_sec=wall, default_family=default_family)
                diag = {
                    "backend": eff_backend,
                    **rate_diag,
                    "table_mode": extra.table_mode,
                    "output_format": extra.output_format,
                    "render_only": bool(extra.ocr_render_only),
                    "us_gov_overall": ev["us_gov_overall"],
                    "text_edit_distance": ev["text_edit_distance"],
                    "reading_order_sim": ev["reading_order_sim"],
                    "table_teds": ev["table_teds"],
                    "valid_markdown_rate": ev["valid_markdown_rate"],
                    "repeat_4gram_rate": ev["repeat_4gram_rate"],
                    "repeat_loop_count": ev["repeat_loop_count"],
                    "device_guard_tripped": guard_tripped,
                }
                records.append(
                    _build_record(
                        package=pkg_name,
                        category=5,
                        config_label=eff_label,
                        benchmark=f"US-Gov-OCR({len(suite_items)})",
                        valid_flags=ev["per_item_valid"],
                        item_scores=ev["per_item_scores"],
                        wall_sec=wall,
                        diagnostics=diag,
                        memory_4bucket_gb=mem4,
                        sub_benchmarks={
                            "by_branch": ev["per_branch"],
                            "by_source": ev["per_source"],
                        },
                        host_mem_gb=host_mem_gb,
                    )
                )
    finally:
        if pub_ad is not None:
            pub_ad.close()
        if gemma4_eng is not None and hasattr(gemma4_eng, "close"):
            gemma4_eng.close()
        if fast_eng is not None and hasattr(fast_eng, "close"):
            fast_eng.close()
    return records


# ---------------------------------------------------------------------------
# Category 5 Visual Page-Image Retrieval Workers (`fast-embeddinggemma-2-vision`, `fast-pplx-vision`)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def _resolve_visual_retrieval_suites(extra: argparse.Namespace) -> list[str]:
    raw = str(getattr(extra, "visual_retrieval_suites", "default") or "default").strip().lower()
    if raw == "default":
        return list(_VISUAL_RETRIEVAL_DEFAULT_SUITES)
    if raw == "all":
        return list(_VISUAL_RETRIEVAL_ALL_SUITES)
    requested = [tok.strip() for tok in raw.split(",") if tok.strip()]
    for tok in requested:
        if tok not in _VISUAL_RETRIEVAL_ALL_SUITES:
            raise ValueError(
                f"Unknown --visual-retrieval-suites entry {tok!r}; expected subset of {_VISUAL_RETRIEVAL_ALL_SUITES}"
            )
    return requested


def _flatten_json_leaf_values(obj: Any, out: list[str], *, limit: int = 16) -> None:
    if len(out) >= limit or obj is None:
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _flatten_json_leaf_values(v, out, limit=limit)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_json_leaf_values(v, out, limit=limit)
    else:
        s = _WS_COLLAPSE_RE.sub(" ", str(obj)).strip()
        if s and s not in out:
            out.append(s)


def _extract_visual_retrieval_query(suite: str, item: Mapping[str, Any]) -> str:
    """Deterministically build a short cross-modal text query strictly from pinned Category 5 GT."""
    if suite == "fintabnet_otsl":
        raw_html = str(item.get("gold_html") or item.get("html") or "")
        plain = html.unescape(_HTML_TAG_RE.sub(" ", raw_html))
        cleaned = _WS_COLLAPSE_RE.sub(" ", plain).strip()
        return f"Financial table: {cleaned[:220]}".strip() if cleaned else ""
    if suite == "omnidocbench_en":
        blocks = item.get("text_blocks") or []
        parts: list[str] = []
        for blk in blocks:
            txt = _WS_COLLAPSE_RE.sub(" ", str(blk.get("text") or "")).strip()
            if txt:
                parts.append(txt)
            if sum(len(p) for p in parts) >= 220:
                break
        if not parts:
            for tbl in item.get("tables") or []:
                plain = html.unescape(_HTML_TAG_RE.sub(" ", str(tbl.get("html") or "")))
                txt = _WS_COLLAPSE_RE.sub(" ", plain).strip()
                if txt:
                    parts.append(txt)
                    break
        return " ".join(parts)[:240].strip()
    if suite == "cord_v2":
        leaves: list[str] = []
        _flatten_json_leaf_values(item.get("gt_parse") if item.get("gt_parse") is not None else item.get("gt_json"), leaves, limit=14)
        cleaned = " ".join(leaves).strip()
        return f"Receipt: {cleaned[:220]}".strip() if cleaned else ""
    if suite == "olmocr_bench":
        snippets: list[str] = []
        for t in item.get("tests") or []:
            if not isinstance(t, Mapping):
                continue
            ttype = str(t.get("type") or "")
            if ttype == "present" and t.get("text"):
                snippets.append(str(t["text"]))
            elif ttype == "order" and t.get("before") and t.get("after"):
                snippets.append(f"{t['before']} {t['after']}")
            elif ttype == "table" and t.get("cell"):
                hdr = " ".join(str(x) for x in (t.get("top_heading"), t.get("left_heading"), t.get("cell")) if x)
                snippets.append(hdr)
            elif ttype == "math" and t.get("math"):
                snippets.append(str(t["math"]))
            if sum(len(s) for s in snippets) >= 220:
                break
        cleaned = _WS_COLLAPSE_RE.sub(" ", " ".join(snippets)).strip()
        return cleaned[:240]
    return ""


def _build_visual_retrieval_corpus(
    suites: Sequence[str],
    *,
    tier: str,
    max_items: int | None,
    smoke: bool,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    dpi: int = 150,
) -> list[dict[str, Any]]:
    """Build deterministic `(id, source, query, image)` pairs from pinned Category 5 datasets."""
    from PIL import Image

    per_src_cap = 2 if smoke else (max(1, int(max_items)) if max_items is not None else (6 if tier == "tier1" else 20))
    corpus: list[dict[str, Any]] = []
    for suite in suites:
        if suite == "fintabnet_otsl":
            raw_items = load_fintabnet_otsl(
                tier=tier, max_items=per_src_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
            )
            for it in raw_items:
                q = _extract_visual_retrieval_query(suite, it)
                if len(q) < 4:
                    continue
                corpus.append(
                    {
                        "id": f"{suite}:{it['id']}",
                        "source": suite,
                        "query": q,
                        "image": _bytes_to_pil(it["image_bytes"]),
                    }
                )
        elif suite == "omnidocbench_en":
            raw_items = load_omnidocbench_en(
                tier=tier,
                max_items=per_src_cap,
                download_images=True,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            for it in raw_items:
                q = _extract_visual_retrieval_query(suite, it)
                if len(q) < 4:
                    continue
                with Image.open(it["image_path"]) as im:
                    pil = im.convert("RGB")
                corpus.append({"id": f"{suite}:{it['id']}", "source": suite, "query": q, "image": pil})
        elif suite == "cord_v2":
            raw_items = load_cord_v2(
                tier=tier, max_items=per_src_cap, lock=lock, cache_dir=cache_dir, local_files_only=local_files_only
            )
            for it in raw_items:
                q = _extract_visual_retrieval_query(suite, it)
                if len(q) < 4:
                    continue
                corpus.append(
                    {
                        "id": f"{suite}:{it['id']}",
                        "source": suite,
                        "query": q,
                        "image": _bytes_to_pil(it["image_bytes"]),
                    }
                )
        elif suite == "olmocr_bench":
            raw_items = load_olmocr_bench(
                tier=tier,
                max_pages=per_src_cap,
                download_pdfs=True,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            for it in raw_items:
                q = _extract_visual_retrieval_query(suite, it)
                if len(q) < 4:
                    continue
                pil = _render_pdf_first_page_pil(it["pdf_path"], dpi=int(dpi))
                corpus.append({"id": f"{suite}:{it['id']}", "source": suite, "query": q, "image": pil})
    return corpus


def _score_visual_retrieval_matrix(
    sims: np.ndarray,
    corpus: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute exact per-query MRR@10, R@1, R@5, and source breakdowns from `[N, N]` similarity scores."""
    mat = np.asarray(sims, dtype=np.float32)
    n = len(corpus)
    if mat.shape != (n, n):
        raise ValueError(f"Expected ({n}, {n}) visual retrieval score matrix, got {mat.shape}")

    valid_flags: list[bool] = []
    rr_scores: list[float] = []
    hit1_scores: list[float] = []
    hit5_scores: list[float] = []
    ranks: list[int] = []
    by_src_rr: dict[str, list[float]] = {}
    by_src_r1: dict[str, list[float]] = {}

    for i in range(n):
        row = mat[i]
        ok = bool(np.all(np.isfinite(row)))
        valid_flags.append(ok)
        if not ok:
            rr, h1, h5, rk = 0.0, 0.0, 0.0, n + 1
        else:
            order = np.argsort(-row, kind="stable")
            pos = int(np.where(order == i)[0][0])
            rk = pos + 1
            rr = (1.0 / float(rk)) if rk <= 10 else 0.0
            h1 = 1.0 if rk == 1 else 0.0
            h5 = 1.0 if rk <= 5 else 0.0
        rr_scores.append(rr)
        hit1_scores.append(h1)
        hit5_scores.append(h5)
        ranks.append(rk)
        src = str(corpus[i].get("source") or "unknown")
        by_src_rr.setdefault(src, []).append(rr)
        by_src_r1.setdefault(src, []).append(h1)

    query_ids = [str(c["id"]) for c in corpus]
    id_bytes = json.dumps(query_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    per_source = {
        src: {
            "n": len(vals),
            "recall_at_1": round(float(np.mean(by_src_r1[src])), 4),
            "mrr_at_10": round(float(np.mean(vals)), 4),
        }
        for src, vals in sorted(by_src_rr.items())
    }
    return {
        "valid_flags": valid_flags,
        "per_item_rr": rr_scores,
        "ranks": ranks,
        "mrr_at_10": round(float(np.mean(rr_scores)) if rr_scores else 0.0, 4),
        "recall_at_1": round(float(np.mean(hit1_scores)) if hit1_scores else 0.0, 4),
        "recall_at_5": round(float(np.mean(hit5_scores)) if hit5_scores else 0.0, 4),
        "per_source": per_source,
        "coverage": {
            "n_queries": n,
            "n_valid_queries": sum(valid_flags),
            "ordered_query_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
        },
    }


def _run_visual_retrieval_worker(
    pkg_name: str,
    *,
    tier: str,
    config_label: str,
    extra: argparse.Namespace,
    lock: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
    max_items: int | None,
    smoke: bool,
    home: Path,
    host_mem_gb: float,
) -> list[dict[str, Any]]:
    """Run cross-modal text-to-page-image retrieval for `fast-embeddinggemma-2-vision` or `fast-pplx-vision`."""
    from PIL import Image

    suites = _resolve_visual_retrieval_suites(extra)
    corpus = _build_visual_retrieval_corpus(
        suites,
        tier=tier,
        max_items=max_items,
        smoke=smoke,
        lock=lock,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dpi=int(extra.ocr_dpi),
    )
    if not corpus:
        return []

    model_dirs: list[Path] = [home / "artifacts"]
    idx_mode = str(getattr(extra, "visual_index_mode", "fp16") or "fp16")
    eff_batch = max(1, int(extra.batch_size)) if getattr(extra, "batch_size", None) is not None else 8
    eng: Any = None
    eff_label: str
    extra_diag: dict[str, Any] = {}

    if pkg_name == "fast-embeddinggemma-2-vision":
        from fast_embeddinggemma_2.engine import FastEmbeddingGemma2

        eff_mode = extra.mode or "gpu"
        eng = FastEmbeddingGemma2(
            home=home,
            default_mode=eff_mode,
            enable_ane=(eff_mode != "gpu"),
            warmup=False,
        )
        try:
            vdir = eng._resolve_vision_model_dir(getattr(extra, "vision_model_dir", None))
            model_dirs.append(vdir)
        except Exception:
            pass
        soft_toks = getattr(extra, "max_soft_tokens", None)
        soft_tag = f"_s{int(soft_toks)}" if soft_toks is not None else ""
        eff_label = config_label or f"emb2_vision_{eff_mode}_{idx_mode}{soft_tag}"
        extra_diag = {
            "backend": "fast-embeddinggemma-2-vision",
            "mode": eff_mode,
            "index_mode": idx_mode,
            "max_soft_tokens": soft_toks,
            "batch_size": eff_batch,
        }
    elif pkg_name == "fast-pplx-vision":
        from fast_pplx.engine import FastPplx

        eff_mode = extra.mode or "gpu"
        eff_chunks = max(1, int(getattr(extra, "visual_chunks", 8) or 8))
        eff_crop = not bool(getattr(extra, "no_visual_crop_margins", False))
        eff_gbs = getattr(extra, "grid_bucket_step", None)
        eng = FastPplx(home=home, mode=eff_mode, ane_profile=extra.ane_profile, local_files_only=True)
        if getattr(eng, "model_dir", None):
            model_dirs.append(Path(eng.model_dir))
        eff_label = config_label or f"pplx_vision_c{eff_chunks}_{idx_mode}{'_crop' if eff_crop else '_full'}"
        extra_diag = {
            "backend": "fast-pplx-vision",
            "mode": eff_mode,
            "index_mode": idx_mode,
            "chunks": eff_chunks,
            "crop_margins": eff_crop,
            "grid_bucket_step": eff_gbs,
            "batch_size": eff_batch,
        }
    else:
        raise ValueError(f"Unsupported visual retrieval package: {pkg_name!r}")

    probe = FourBucketMemoryProbe(model_directories=model_dirs)
    guard = DeviceMemoryGuard()
    guard_tripped = False
    images = [c["image"] for c in corpus]
    queries = [c["query"] for c in corpus]

    try:
        guard.arm()
        # Untimed tiny warmup so lazy vision towers & text encoders load outside EOT timing.
        tiny = Image.new("RGB", (64, 64), (255, 255, 255))
        if pkg_name == "fast-embeddinggemma-2-vision":
            _ = eng.encode_images(
                [tiny],
                max_soft_tokens=getattr(extra, "max_soft_tokens", None),
                batch_size=1,
                vision_model_dir=getattr(extra, "vision_model_dir", None),
            )
            _ = eng.encode_texts(["warmup query"], is_query=True, add_query_prefix=True, mode=extra.mode or "gpu")
        else:
            _ = eng.encode_images([tiny], chunks=1, batch_size=1, crop_margins=False)
            _ = eng.encode_queries(["warmup query"], mode=extra.mode or "gpu", output_dtype="float32", normalize=True)
        _mx_sync()
        _mx_reset_peak()

        t0 = time.perf_counter()
        if pkg_name == "fast-embeddinggemma-2-vision":
            t_img0 = time.perf_counter()
            index = eng.build_image_index(
                images,
                max_soft_tokens=getattr(extra, "max_soft_tokens", None),
                batch_size=eff_batch,
                mode=idx_mode,
                vision_model_dir=getattr(extra, "vision_model_dir", None),
            )
            _mx_sync()
            img_sec = time.perf_counter() - t_img0
            t_q0 = time.perf_counter()
            q_vecs = eng.encode_texts(queries, is_query=True, add_query_prefix=True, mode=extra.mode or "gpu")
            _mx_sync()
            q_sec = time.perf_counter() - t_q0
        else:
            t_img0 = time.perf_counter()
            index = eng.build_image_index(
                images,
                chunks=max(1, int(getattr(extra, "visual_chunks", 8) or 8)),
                batch_size=eff_batch,
                crop_margins=not bool(getattr(extra, "no_visual_crop_margins", False)),
                grid_bucket_step=getattr(extra, "grid_bucket_step", None),
                mode=idx_mode,
                agg="maxsim",
            )
            _mx_sync()
            img_sec = time.perf_counter() - t_img0
            t_q0 = time.perf_counter()
            q_vecs = eng.encode_queries(queries, mode=extra.mode or "gpu", output_dtype="float32", normalize=True)
            _mx_sync()
            q_sec = time.perf_counter() - t_q0
        sims = index.score_all(q_vecs)
        wall = time.perf_counter() - t0
        _, ref = guard.check()
        if ref is not None:
            guard_tripped = True
        mem4 = _memory_4bucket_dict(probe)
        scored = _score_visual_retrieval_matrix(sims, corpus)
        diag = {
            **extra_diag,
            "n_images": len(images),
            "recall_at_1": scored["recall_at_1"],
            "recall_at_5": scored["recall_at_5"],
            "mrr_at_10": scored["mrr_at_10"],
            "img_per_sec": round(float(len(images)) / max(img_sec, 1e-6), 3),
            "encode_images_sec": round(img_sec, 4),
            "encode_queries_sec": round(q_sec, 4),
            "index_bytes": int(getattr(index, "nbytes", 0)),
            "compression_ratio_vs_fp32": round(float(getattr(index, "compression_ratio_vs_fp32", 1.0)), 3),
            "device_guard_tripped": guard_tripped,
        }
        return [
            _build_record(
                package=pkg_name,
                category=5,
                config_label=eff_label,
                benchmark=f"VisualRetrieval({len(corpus)})",
                valid_flags=scored["valid_flags"],
                item_scores=scored["per_item_rr"],
                wall_sec=wall,
                diagnostics=diag,
                memory_4bucket_gb=mem4,
                sub_benchmarks={
                    "by_source": scored["per_source"],
                    "visual_retrieval": scored["coverage"],
                },
                host_mem_gb=host_mem_gb,
            )
        ]
    finally:
        if eng is not None and hasattr(eng, "close"):
            eng.close()


def _run_worker_main(args: argparse.Namespace) -> int:
    """Isolated child subprocess entrypoint executing one package+config under wired_limit & 4-bucket probes."""
    lock, _ = load_lockfile(args.lockfile)
    cache_dir = resolve_cache_dir(args.cache_dir)
    env_meta = collect_env_telemetry()
    host_mem = _host_mem_gb(env_meta)
    pkg_name = str(args.worker_package)
    extra = _parse_extra_flags(shlex.split(args.worker_extra_flags or ""))

    if pkg_name == "openai-chat":
        home = _REPO_ROOT
    else:
        spec = _SPEC_BY_NAME[pkg_name]
        home = resolve_package_home(spec)

    if extra.mlx_cache_limit_mb is not None:
        if extra.mlx_cache_limit_mb < 0:
            raise ValueError("--mlx-cache-limit-mb must be nonnegative")
        import mlx.core as mx

        mx.set_cache_limit(extra.mlx_cache_limit_mb * 1024**2)
    max_coarse_tokens = (
        extra.max_coarse_tokens
        if extra.max_coarse_tokens is not None
        else getattr(args, "max_coarse_tokens", None)
    )
    if max_coarse_tokens is not None and int(max_coarse_tokens) <= 0:
        raise ValueError("--max-coarse-tokens must be positive")
    use_wl = not bool(args.no_wired_limit)
    ctx = wired_limit() if use_wl else _null_ctx()
    with ctx:
        if pkg_name in (
            "fast-bge-small",
            "fast-embeddinggemma-2",
            "fast-pplx",
            "public-bge-small",
            "coreml-bge-small",
            "public-embeddinggemma-2",
            "coreai-embeddinggemma-2",
            "coreml-embeddinggemma-2-fp16",
            "coreml-embeddinggemma-2-int8",
            "public-pplx",
        ):
            try:
                recs = _run_cat1_worker(
                    pkg_name,
                    tier=args.tier,
                    config_label=args.worker_label or "",
                    extra=extra,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=bool(args.local_files_only),
                    max_items=args.max_items,
                    smoke=bool(args.smoke),
                    home=home,
                    host_mem_gb=host_mem,
                    coreai_python=args.coreai_python,
                    pplx_mlx_path=args.pplx_mlx_path,
                    max_coarse_tokens=max_coarse_tokens,
                )
            except FileNotFoundError as exc:
                if pkg_name.startswith(("public-", "coreai-", "coreml-")):
                    recs = [
                        {
                            "package": pkg_name,
                            "category": 1,
                            "config_label": args.worker_label or pkg_name,
                            "benchmark": "NanoBEIR+Banking77+LongEmbed",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    ]
                else:
                    raise
        elif pkg_name in (
            "fast-decision-models",
            "public-laya",
            "coreai-laya",
            "coreml-laya",
            "public-kev-0.8b",
            "coreml-kev-0.8b",
        ):
            if pkg_name == "fast-decision-models":
                backends = ["laya-ane", "kev-0.8b"] if extra.backend == "all" else [extra.backend]
            else:
                backends = [pkg_name]
            recs = []
            for bname in backends:
                try:
                    recs.append(
                        _run_cat2_single_backend(
                            bname,
                            pkg_name=pkg_name,
                            tier=args.tier,
                            config_label=args.worker_label or "",
                            extra=extra,
                            lock=lock,
                            cache_dir=cache_dir,
                            local_files_only=bool(args.local_files_only),
                            max_items=args.max_items,
                            smoke=bool(args.smoke),
                            home=home,
                            host_mem_gb=host_mem,
                            coreai_python=args.coreai_python,
                        )
                    )
                except FileNotFoundError as exc:
                    recs.append(
                        {
                            "package": pkg_name,
                            "category": 2,
                            "config_label": bname,
                            "benchmark": "RAGTruth+CLINC150+RewardBench2",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    )
        elif pkg_name in _VISUAL_RETRIEVAL_PACKAGES:
            recs = _run_visual_retrieval_worker(
                pkg_name,
                tier=args.tier,
                config_label=args.worker_label or "",
                extra=extra,
                lock=lock,
                cache_dir=cache_dir,
                local_files_only=bool(args.local_files_only),
                max_items=args.max_items,
                smoke=bool(args.smoke),
                home=home,
                host_mem_gb=host_mem,
            )
        elif pkg_name in _CAT5_PACKAGES:
            try:
                recs = _run_cat5_worker(
                    pkg_name,
                    tier=args.tier,
                    config_label=args.worker_label or "",
                    extra=extra,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=bool(args.local_files_only),
                    max_items=args.max_items,
                    smoke=bool(args.smoke),
                    home=home,
                    host_mem_gb=host_mem,
                )
            except FileNotFoundError as exc:
                if pkg_name.startswith(("public-", "coreai-", "coreml-")):
                    recs = [
                        {
                            "package": pkg_name,
                            "category": 5,
                            "config_label": args.worker_label or pkg_name,
                            "benchmark": "olmOCR-bench+FinTabNet+OmniDocBench+CORD-v2",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    ]
                else:
                    raise
        elif pkg_name == "coreai-qwen3.5-4b" and extra.coreai_variant == "both":
            recs = []
            for ca_var in ("4bit", "fp4_fp8"):
                try:
                    recs.extend(
                        _run_llm_worker(
                            pkg_name,
                            tier=args.tier,
                            config_label=(f"{args.worker_label}_{ca_var}" if args.worker_label else f"coreai_qwen35_4b_{ca_var}"),
                            extra=extra,
                            lock=lock,
                            cache_dir=cache_dir,
                            local_files_only=bool(args.local_files_only),
                            max_items=args.max_items,
                            smoke=bool(args.smoke),
                            include_unconstrained=False,
                            home=home,
                            host_mem_gb=host_mem,
                            coreai_python=args.coreai_python,
                            coreai_variant_override=ca_var,
                        )
                    )
                except FileNotFoundError as exc:
                    recs.append(
                        {
                            "package": pkg_name,
                            "category": 3,
                            "config_label": f"coreai_qwen35_4b_{ca_var}",
                            "benchmark": "IFBench+MMLU-Redux+MATH-500",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    )
        else:
            try:
                recs = _run_llm_worker(
                    pkg_name,
                    tier=args.tier,
                    config_label=args.worker_label or "",
                    extra=extra,
                    lock=lock,
                    cache_dir=cache_dir,
                    local_files_only=bool(args.local_files_only),
                    max_items=args.max_items,
                    smoke=bool(args.smoke),
                    include_unconstrained=bool(args.include_unconstrained_baseline),
                    home=home,
                    host_mem_gb=host_mem,
                    openai_url=args.openai_url,
                    openai_model=args.openai_model,
                    coreai_python=args.coreai_python,
                )
            except FileNotFoundError as exc:
                if pkg_name.startswith(("public-", "coreai-")):
                    recs = [
                        {
                            "package": pkg_name,
                            "category": _SPEC_BY_NAME[pkg_name].category if pkg_name in _SPEC_BY_NAME else 3,
                            "config_label": args.worker_label or pkg_name,
                            "benchmark": "IFBench+MMLU-Redux+MATH-500",
                            "status": "skipped",
                            "error": str(exc),
                            "gate_verdict": "SKIP",
                        }
                    ]
                else:
                    raise

    out_file = getattr(args, "worker_out_file", None)
    write_worker_json({"records": recs}, worker_out_file=out_file, marker=_WORKER_MARKER)
    if recs and all(r.get("status") == "skipped" for r in recs):
        return EXIT_SKIP
    return 0


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield None


# ---------------------------------------------------------------------------
# Coordinator Subprocess Spawning, ABBA Interleaving & Formatting
# ---------------------------------------------------------------------------


def _spawn_isolated_worker(
    *,
    python_exe: str,
    pkg_name: str,
    label: str,
    extra_flags: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cmd = [
        python_exe,
        str(Path(__file__).resolve()),
        "_worker",
        "--worker-package",
        pkg_name,
        "--worker-label",
        label,
        "--worker-extra-flags",
        extra_flags,
        "--tier",
        args.tier,
    ]
    if args.lockfile:
        cmd.extend(["--lockfile", str(args.lockfile)])
    if args.cache_dir:
        cmd.extend(["--cache-dir", str(args.cache_dir)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    if args.smoke:
        cmd.append("--smoke")
    if args.include_unconstrained_baseline:
        cmd.append("--include-unconstrained-baseline")
    if args.no_wired_limit:
        cmd.append("--no-wired-limit")
    if args.openai_url:
        cmd.extend(["--openai-url", str(args.openai_url)])
    if args.openai_model:
        cmd.extend(["--openai-model", str(args.openai_model)])
    if getattr(args, "coreai_python", None):
        cmd.extend(["--coreai-python", str(args.coreai_python)])
    if getattr(args, "pplx_mlx_path", None):
        cmd.extend(["--pplx-mlx-path", str(args.pplx_mlx_path)])
    if getattr(args, "max_coarse_tokens", None) is not None:
        cmd.extend(["--max-coarse-tokens", str(int(args.max_coarse_tokens))])

    env = dict(os.environ)
    env["M4_GPU_LOCK_HELD"] = "1"
    py_paths = [
        str(_REPO_ROOT / "scripts"),
        str(_REPO_ROOT / "packages" / "fast-shared" / "src"),
        str(_REPO_ROOT / "packages" / "fast-decision-models" / "src"),
        str(_REPO_ROOT / "packages" / "fast-gemma4" / "src"),
        str(_REPO_ROOT / "packages" / "fast-qwen3" / "src"),
        str(_REPO_ROOT / "packages" / "fast-bge-small" / "src"),
        str(_REPO_ROOT / "packages" / "fast-embeddinggemma-2" / "src"),
        str(_REPO_ROOT / "packages" / "fast-pplx" / "src"),
        str(_REPO_ROOT / "packages" / "fast-gpt-oss" / "src"),
        str(_REPO_ROOT / "packages" / "fast-ocr" / "src"),
    ]
    if getattr(args, "pplx_mlx_path", None):
        py_paths.append(str(Path(args.pplx_mlx_path).expanduser()))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    if pkg_name in _SPEC_BY_NAME:
        spec = _SPEC_BY_NAME[pkg_name]
        env.setdefault(spec.home_env_var, str(resolve_package_home(spec)))

    res = run_isolated_json_worker(
        cmd,
        worker_out_arg="--worker-out-file",
        marker=_WORKER_MARKER,
        timeout_sec=int(args.timeout_sec),
        cwd=_REPO_ROOT,
        env=env,
        skip_exit_codes=(EXIT_SKIP,),
        stderr_tail_lines=20,
    )
    if res["timed_out"]:
        return [
            {
                "package": pkg_name,
                "category": _SPEC_BY_NAME[pkg_name].category if pkg_name in _SPEC_BY_NAME else 3,
                "config_label": label or "default",
                "status": "failed",
                "error": f"timeout after {args.timeout_sec}s",
                "stderr_tail": res["stderr_tail"],
                "gate_verdict": "FAIL",
            }
        ]
    payload = res.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return list(payload["records"])

    rc = int(res.get("returncode") or 0)
    err_s = str(res.get("stderr") or "")
    out_s = str(res.get("stdout") or "")
    err_tail = str(res.get("stderr_tail") or out_s or f"exit={rc}").strip().splitlines()
    reason = err_tail[-1] if err_tail else f"exit={rc}"
    low = (err_s + "\n" + out_s).lower()
    is_skip = (res["status"] == "skipped") or any(
        tok in low for tok in ("filenotfounderror", "not found", "missing", "not staged")
    )
    return [
        {
            "package": pkg_name,
            "category": _SPEC_BY_NAME[pkg_name].category if pkg_name in _SPEC_BY_NAME else 3,
            "config_label": label or "default",
            "status": "skipped" if is_skip else "failed",
            "error": reason[:200],
            "stderr_tail": res["stderr_tail"],
            "gate_verdict": "SKIP" if is_skip else "FAIL",
        }
    ]


def _aggregate_abba_records(runs: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate repeated ABBA worker outputs by median wall_sec & EOT while preserving item scores."""
    if not runs:
        return []
    if len(runs) == 1:
        return [dict(r) for r in runs[0]]

    n_slots = max(map(len, runs))
    aggregated: list[dict[str, Any]] = []
    for slot_i in range(n_slots):
        slot_records = [r[slot_i] if slot_i < len(r) else {} for r in runs]
        slot_runs = [r for r in slot_records if r.get("status") == "ok"]
        if not slot_runs:
            aggregated.append(dict(next(r for r in slot_records if r)))
            continue
        med_wall = float(statistics.median([float(r["wall_sec"]) for r in slot_runs]))
        rep = dict(min(slot_runs, key=lambda r: abs(float(r["wall_sec"]) - med_wall)))
        rep["wall_sec"] = round(med_wall, 4)
        rep["raw_items_per_sec"] = round(float(rep["n_items"]) / max(med_wall, 1e-6), 4)
        rep["valid_outputs_per_sec"] = round(float(rep["n_valid"]) / max(med_wall, 1e-6), 4)
        rep["effective_outcome_per_sec"] = round(
            float(rep["raw_items_per_sec"]) * float(rep["objective_score_e2e"]), 4
        )
        rep["abba_repeats"] = len(slot_runs)
        rep["wall_series_sec"] = [round(float(r["wall_sec"]), 4) for r in slot_runs]
        if rep.get("category") == 5 and isinstance(rep.get("diagnostics"), Mapping):
            diag5 = dict(rep["diagnostics"])
            n_it = max(1, int(rep.get("n_items", 1) or 1))
            diag5["pages_per_sec"] = round(float(rep["n_items"]) / max(med_wall, 1e-6), 4)
            diag5["sec_per_page"] = round(med_wall / float(n_it), 4)
            tot_tok = int(diag5.get("total_generated_tokens", 0) or 0)
            if tot_tok > 0:
                diag5["e2e_tok_s"] = round(float(tot_tok) / max(med_wall, 1e-6), 2)
            rep["diagnostics"] = diag5
        if rep.get("category") == 1:
            incomplete = [i for i, r in enumerate(slot_records, 1) if r.get("status") != "ok" or retrieval_coverage_errors(r)]
            if incomplete:
                rep["incomplete_retrieval_repeats"] = incomplete
                rep["retrieval_coverage_status"] = "INCOMPLETE"
                rep["gate_verdict"] = "FAIL"
                rep["gate_reasons"] = [*rep.get("gate_reasons", []), f"incomplete retrieval repeats: {incomplete}"]
        aggregated.append(rep)
    return aggregated


def parse_abba_arm(raw: str) -> tuple[str, str]:
    """Parse ``label:flags`` CLI spec for ``--abba-compare``."""
    if ":" not in raw:
        return raw.strip(), ""
    lbl, flags = raw.split(":", 1)
    return lbl.strip(), flags.strip()


def _fmt_opt_rate(val: float | None, digits: int = 1) -> str:
    if val is None or float(val) <= 0.0:
        return "-"
    return f"{float(val):.{digits}f}"


def _fmt_opt_ratio(val: float | None) -> str:
    if val is None or float(val) <= 0.0:
        return "-"
    return f"{float(val):.2f}x"


def _short_chip_label(chip: str | None, fallback: str) -> str:
    raw = str(chip or "").strip()
    if not raw:
        return fallback
    if raw.lower().startswith("apple "):
        return raw[6:].strip() or fallback
    return raw


def format_outcomes_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    """Render concise Markdown summary table of External Objective Outcome Benchmark results."""
    lines = [
        "| Package | Cat | Config | Benchmark | Valid / intended | `S_obj,e2e` | `EOT` (u/s) | Wall (s) | 4-Bucket Mem (GB) | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    cat5_ok: list[Mapping[str, Any]] = []
    for r in results:
        pkg = r.get("package", "-")
        cat = r.get("category", "-")
        cfg = r.get("config_label", "-")
        bench = r.get("benchmark", "-")
        if r.get("status") != "ok":
            lines.append(f"| `{pkg}` | {cat} | `{cfg}` | {bench} | - | - | - | - | - | `{r.get('gate_verdict', r.get('status', 'FAIL'))}` |")
            continue
        n_str = f"{r['n_valid']}/{r['n_items']}"
        s_val = f"{float(r['objective_score_e2e']):.4f}"
        eot_val = f"{float(r['effective_outcome_per_sec']):.3f}"
        wall = f"{float(r['wall_sec']):.2f}"
        mem = float((r.get("memory_4bucket_gb") or {}).get("combined_with_model_mapped_gb", 0.0))
        guard = "FAIL (retrieval coverage)" if retrieval_coverage_errors(r) else r.get("gate_verdict", "PASS")
        lines.append(f"| `{pkg}` | {cat} | `{cfg}` | {bench} | {n_str} | {s_val} | {eot_val} | {wall} | {mem:.3f} | `{guard}` |")
        if cat == 5:
            cat5_ok.append(r)

    if cat5_ok:
        lines.extend(
            [
                "",
                "### Category 5 OCR Cross-Family Token & Page Rates",
                "",
                "| Package | Family | Config | Benchmark | `S_obj,e2e` | `pp/s` | `s/page` | `tok/page` | Decode `tok/s` | E2E `tok/s` | `EOT` (u/s) | Mem (GB) |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for r in cat5_ok:
            rates = _extract_cat5_rates(r)
            lines.append(
                f"| `{r.get('package', '-')}` | `{rates['model_family']}` | `{r.get('config_label', '-')}` "
                f"| {r.get('benchmark', '-')} | {float(r['objective_score_e2e']):.4f} "
                f"| {rates['pages_per_sec']:.3f} | {rates['sec_per_page']:.2f} "
                f"| {_fmt_opt_rate(rates['tokens_per_page'], 1)} | {_fmt_opt_rate(rates['decode_tok_s'], 1)} "
                f"| {_fmt_opt_rate(rates['e2e_tok_s'], 1)} | {float(r['effective_outcome_per_sec']):.3f} "
                f"| {rates['mem_gb']:.3f} |"
            )
    return "\n".join(lines)


def format_compare_markdown(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    baseline_chip: str = "Baseline",
    candidate_chip: str = "Candidate",
) -> str:
    """Render side-by-side comparison Markdown, including cross-chip M4 vs M5 OCR token/page rates."""
    b_lbl = _short_chip_label(baseline_chip, "Base")
    c_lbl = _short_chip_label(candidate_chip, "Cand")
    lines = [
        f"### Outcome Comparison (`{baseline_chip}` vs `{candidate_chip}`)",
        "",
        f"| Package | Cat | Benchmark | {b_lbl} `S_obj` | {c_lbl} `S_obj` | `ΔS_obj` | {b_lbl} `EOT` | {c_lbl} `EOT` | `EOT` Ratio | {b_lbl} Mem | {c_lbl} Mem | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    cat5_rows: list[Mapping[str, Any]] = []
    for c in comparisons:
        pkg = c.get("package", "-")
        cat = c.get("category", "-")
        bench = c.get("benchmark", "-")
        b_sc = c.get("baseline_score_e2e")
        c_sc = c.get("candidate_score_e2e")
        d_sc = c.get("delta_objective_score_e2e")
        b_eot = c.get("baseline_eot")
        c_eot = c.get("candidate_eot")
        eot_r = c.get("eot_ratio")
        b_mem = c.get("baseline_mem_gb")
        c_mem = c.get("candidate_mem_gb")
        lines.append(
            f"| `{pkg}` | {cat} | {bench} "
            f"| {f'{float(b_sc):.4f}' if b_sc is not None else '-'} "
            f"| {f'{float(c_sc):.4f}' if c_sc is not None else '-'} "
            f"| {f'{float(d_sc):+.4f}' if d_sc is not None else '-'} "
            f"| {f'{float(b_eot):.3f}' if b_eot is not None else '-'} "
            f"| {f'{float(c_eot):.3f}' if c_eot is not None else '-'} "
            f"| {_fmt_opt_ratio(eot_r)} "
            f"| {f'{float(b_mem):.2f} GB' if b_mem is not None else '-'} "
            f"| {f'{float(c_mem):.2f} GB' if c_mem is not None else '-'} "
            f"| `{c.get('verdict', 'FAIL')}` |"
        )
        if cat == 5 and c.get("baseline_pages_per_sec") is not None:
            cat5_rows.append(c)

    if cat5_rows:
        lines.extend(
            [
                "",
                f"### Category 5 Cross-Silicon OCR Token & Page Rates (`{baseline_chip}` vs `{candidate_chip}`)",
                "",
                f"| Package | Family | Benchmark | {b_lbl} `S_obj` | {c_lbl} `S_obj` | {b_lbl} `pp/s` (`s/p`) | {c_lbl} `pp/s` (`s/p`) | `pp/s` Ratio | {b_lbl} `tok/p` | {c_lbl} `tok/p` | {b_lbl} Dec `tok/s` | {c_lbl} Dec `tok/s` | Dec Ratio | {b_lbl} E2E `tok/s` | {c_lbl} E2E `tok/s` | E2E Ratio | {b_lbl} Mem | {c_lbl} Mem | Verdict |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for c in cat5_rows:
            b_pps = float(c.get("baseline_pages_per_sec", 0.0) or 0.0)
            c_pps = float(c.get("candidate_pages_per_sec", 0.0) or 0.0)
            b_spp = float(c.get("baseline_sec_per_page", 0.0) or 0.0)
            c_spp = float(c.get("candidate_sec_per_page", 0.0) or 0.0)
            b_mem = c.get("baseline_mem_gb")
            c_mem = c.get("candidate_mem_gb")
            lines.append(
                f"| `{c.get('package', '-')}` | `{c.get('model_family', '-')}` | {c.get('benchmark', '-')} "
                f"| {float(c.get('baseline_score_e2e', 0.0) or 0.0):.4f} "
                f"| {float(c.get('candidate_score_e2e', 0.0) or 0.0):.4f} "
                f"| {b_pps:.3f} ({b_spp:.2f}s) | {c_pps:.3f} ({c_spp:.2f}s) | {_fmt_opt_ratio(c.get('pages_per_sec_ratio'))} "
                f"| {_fmt_opt_rate(c.get('baseline_tokens_per_page'), 1)} | {_fmt_opt_rate(c.get('candidate_tokens_per_page'), 1)} "
                f"| {_fmt_opt_rate(c.get('baseline_decode_tok_s'), 1)} | {_fmt_opt_rate(c.get('candidate_decode_tok_s'), 1)} | {_fmt_opt_ratio(c.get('decode_tok_s_ratio'))} "
                f"| {_fmt_opt_rate(c.get('baseline_e2e_tok_s'), 1)} | {_fmt_opt_rate(c.get('candidate_e2e_tok_s'), 1)} | {_fmt_opt_ratio(c.get('e2e_tok_s_ratio'))} "
                f"| {f'{float(b_mem):.2f} GB' if b_mem is not None else '-'} "
                f"| {f'{float(c_mem):.2f} GB' if c_mem is not None else '-'} "
                f"| `{c.get('verdict', 'FAIL')}` |"
            )
    return "\n".join(lines)


def _cmd_setup_data(args: argparse.Namespace) -> int:
    ds_list = None if args.datasets in (None, "", "all") else [x.strip() for x in args.datasets.split(",") if x.strip()]
    report = stage_tier_datasets(
        tier=args.tier,
        datasets=ds_list,
        lock_path=args.lockfile,
        cache_dir=args.cache_dir,
        verify_checksums=not bool(args.no_verify_checksums),
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_stage_gov(args: argparse.Namespace) -> int:
    branches = tuple(x.strip() for x in str(args.branches).split(",") if x.strip())
    raw_sources = str(getattr(args, "sources", "all") or "all").strip()
    sources = None if raw_sources.lower() == "all" else tuple(x.strip() for x in raw_sources.split(",") if x.strip())
    report = stage_us_gov_ocr(
        branches=branches,
        per_source=int(args.per_source),
        year=int(args.year),
        cache_dir=args.cache_dir,
        loc_btp_path=getattr(args, "loc_btp_path", None),
        auto_loc_btp=not bool(getattr(args, "no_loc_btp", False)),
        sources=sources,
    )
    print(json.dumps(report, indent=2))
    return 0


def _compare_record_key(rec: Mapping[str, Any]) -> tuple[Any, ...]:
    """Disambiguating comparison key preventing Cat 2 (Laya/Kev) and Cat 4 (llg_on/llg_off) collisions."""
    diag = rec.get("diagnostics") or {}
    return (
        rec.get("package"),
        rec.get("category"),
        rec.get("benchmark"),
        diag.get("backend"),
        diag.get("llguidance"),
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    base_doc = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand_doc = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    if (base_doc.get("dataset_lock_sha256") and cand_doc.get("dataset_lock_sha256")
            and base_doc["dataset_lock_sha256"] != cand_doc["dataset_lock_sha256"]):
        print(json.dumps({"comparisons": [], "verdict": "FAIL", "error": "Dataset lock fingerprints differ."}, indent=2))
        return 1
    base_env = base_doc.get("environment") or {}
    cand_env = cand_doc.get("environment") or {}
    base_chip = str(base_env.get("chip") or "Baseline")
    cand_chip = str(cand_env.get("chip") or "Candidate")
    host_mem = float(args.host_mem_gb) if args.host_mem_gb is not None else float(
        cand_env.get("mem_gb") or 0.0
    )

    base_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for br in base_doc.get("results", []):
        if br.get("status") == "skipped":
            continue
        base_by_key.setdefault(_compare_record_key(br), []).append(br)

    comparisons: list[dict[str, Any]] = []
    for cr in cand_doc.get("results", []):
        if cr.get("status") == "skipped":
            continue
        key = _compare_record_key(cr)
        bucket = base_by_key.get(key)
        if not bucket:
            continue
        if len(bucket) == 1:
            br = bucket[0]
        else:
            exact = [r for r in bucket if r.get("config_label") == cr.get("config_label")]
            if len(exact) != 1:
                comparisons.append(
                    {
                        "package": cr.get("package"),
                        "category": cr.get("category"),
                        "benchmark": cr.get("benchmark"),
                        "baseline_label": None,
                        "candidate_label": cr.get("config_label"),
                        "baseline_chip": base_chip,
                        "candidate_chip": cand_chip,
                        "verdict": "FAIL",
                        "reasons": [f"Ambiguous baseline match ({len(bucket)} records) for key {key}"],
                    }
                )
                continue
            br = exact[0]
        ev = evaluate_promotion_gates(
            br,
            cr,
            profile=args.gate_profile,
            lossless_tolerance=args.lossless_tolerance,
            min_eot_ratio=float(args.min_eot_ratio),
            host_mem_gb=host_mem if host_mem > 0 else None,
        )
        b_mem = float((br.get("memory_4bucket_gb") or {}).get("combined_with_model_mapped_gb", 0.0) or 0.0)
        c_mem = float((cr.get("memory_4bucket_gb") or {}).get("combined_with_model_mapped_gb", 0.0) or 0.0)
        comp_row: dict[str, Any] = {
            "package": cr.get("package"),
            "category": cr.get("category"),
            "benchmark": cr.get("benchmark"),
            "baseline_label": br.get("config_label"),
            "candidate_label": cr.get("config_label"),
            "baseline_chip": base_chip,
            "candidate_chip": cand_chip,
            "baseline_score_e2e": round(float(br.get("objective_score_e2e", 0.0) or 0.0), 4),
            "candidate_score_e2e": round(float(cr.get("objective_score_e2e", 0.0) or 0.0), 4),
            "baseline_eot": round(float(br.get("effective_outcome_per_sec", 0.0) or 0.0), 3),
            "candidate_eot": round(float(cr.get("effective_outcome_per_sec", 0.0) or 0.0), 3),
            "baseline_wall_sec": round(float(br.get("wall_sec", 0.0) or 0.0), 4),
            "candidate_wall_sec": round(float(cr.get("wall_sec", 0.0) or 0.0), 4),
            "baseline_mem_gb": round(b_mem, 4),
            "candidate_mem_gb": round(c_mem, 4),
            **ev.to_dict(),
        }
        if cr.get("category") == 5:
            b_rates = _extract_cat5_rates(br)
            c_rates = _extract_cat5_rates(cr)
            comp_row.update(
                {
                    "model_family": c_rates["model_family"] or b_rates["model_family"],
                    "baseline_pages_per_sec": b_rates["pages_per_sec"],
                    "candidate_pages_per_sec": c_rates["pages_per_sec"],
                    "pages_per_sec_ratio": round(c_rates["pages_per_sec"] / max(b_rates["pages_per_sec"], 1e-6), 3)
                    if b_rates["pages_per_sec"] > 0.0
                    else None,
                    "baseline_sec_per_page": b_rates["sec_per_page"],
                    "candidate_sec_per_page": c_rates["sec_per_page"],
                    "baseline_tokens_per_page": b_rates["tokens_per_page"],
                    "candidate_tokens_per_page": c_rates["tokens_per_page"],
                    "baseline_decode_tok_s": b_rates["decode_tok_s"],
                    "candidate_decode_tok_s": c_rates["decode_tok_s"],
                    "decode_tok_s_ratio": round(c_rates["decode_tok_s"] / max(b_rates["decode_tok_s"], 1e-6), 3)
                    if (b_rates["decode_tok_s"] > 0.0 and c_rates["decode_tok_s"] > 0.0)
                    else None,
                    "baseline_e2e_tok_s": b_rates["e2e_tok_s"],
                    "candidate_e2e_tok_s": c_rates["e2e_tok_s"],
                    "e2e_tok_s_ratio": round(c_rates["e2e_tok_s"] / max(b_rates["e2e_tok_s"], 1e-6), 3)
                    if (b_rates["e2e_tok_s"] > 0.0 and c_rates["e2e_tok_s"] > 0.0)
                    else None,
                }
            )
        comparisons.append(comp_row)
    if not comparisons:
        out_empty = {
            "comparisons": [],
            "verdict": "FAIL",
            "error": "No matching records found between baseline and candidate reports.",
        }
        print(json.dumps(out_empty, indent=2))
        return 1
    verdicts = {r["verdict"] for r in comparisons}
    verdict = "FAIL" if "FAIL" in verdicts else "UNRESOLVED" if "UNRESOLVED" in verdicts else "PASS"
    md_out = format_compare_markdown(comparisons, baseline_chip=base_chip, candidate_chip=cand_chip)
    if getattr(args, "markdown_out", None):
        out_m = Path(args.markdown_out)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md_out + "\n", encoding="utf-8")
    if getattr(args, "markdown", False):
        print(md_out)
    else:
        print(
            json.dumps(
                {
                    "baseline_environment": base_env,
                    "candidate_environment": cand_env,
                    "comparisons": comparisons,
                    "verdict": verdict,
                },
                indent=2,
            )
        )
    return 0 if verdict == "PASS" else 1


_DEFAULT_COARSE_MAX_TOKENS: dict[str, int] = {
    "fast-embeddinggemma-2": 2048,
    "public-embeddinggemma-2": 2048,
    "coreai-embeddinggemma-2": 2048,
    "coreml-embeddinggemma-2-fp16": 512,
    "coreml-embeddinggemma-2-int8": 512,
}


def compute_public_comparisons(
    results: Sequence[Mapping[str, Any]], *, lossless_tolerance: float = DEFAULT_LOSSLESS_TOLERANCE,
) -> list[dict[str, Any]]:
    """Pair completed `fast-*` outcome records against their matching stock `public-*` / `coreai-*` / `coreml-*` baselines."""
    ok_rows = [r for r in results if r.get("status") == "ok"]
    out: list[dict[str, Any]] = []
    pair_map: tuple[tuple[str, str | None, str], ...] = (
        ("fast-bge-small", None, "public-bge-small"),
        ("fast-bge-small", None, "coreml-bge-small"),
        ("fast-embeddinggemma-2", None, "public-embeddinggemma-2"),
        ("fast-embeddinggemma-2", None, "coreai-embeddinggemma-2"),
        ("fast-embeddinggemma-2", None, "coreml-embeddinggemma-2-fp16"),
        ("fast-embeddinggemma-2", None, "coreml-embeddinggemma-2-int8"),
        ("fast-pplx", None, "public-pplx"),
        ("fast-decision-models", "laya-ane", "public-laya"),
        ("fast-decision-models", "laya-ane", "coreai-laya"),
        ("fast-decision-models", "laya-ane", "coreml-laya"),
        ("fast-decision-models", "kev-0.8b", "public-kev-0.8b"),
        ("fast-decision-models", "kev-0.8b", "coreml-kev-0.8b"),
        ("fast-gemma4", None, "public-gemma4"),
        ("fast-gemma4", None, "coreai-gemma4"),
        ("fast-qwen3", None, "public-qwen3.5"),
        ("fast-qwen3", None, "coreai-qwen3.5-4b"),
        ("fast-gpt-oss", None, "public-gpt-oss"),
        ("fast-ocr", None, "public-ovis-mlx-vlm"),
        ("fast-ocr", None, "public-apple-vision"),
        ("fast-ocr", None, "public-pypdfium2"),
        ("fast-ocr", None, "public-glm-ocr-mlx-vlm"),
        ("fast-ocr", None, "public-lightonocr2-mlx-vlm"),
        ("fast-ocr", None, "public-deepseek-ocr2-mlx-vlm"),
        ("fast-ocr", None, "public-paddleocr-vl-mlx-vlm"),
        ("fast-ocr", None, "public-nemotron-parse-mlx-vlm"),
        ("fast-ocr", None, "public-nuextract3-mlx-vlm"),
        ("fast-ocr", None, "public-rapidocr"),
        ("fast-ocr", None, "public-tesseract"),
        ("fast-ocr", None, "public-docling"),
        ("fast-ocr", None, "public-marker"),
        ("fast-gemma4-vision", None, "public-ovis-mlx-vlm"),
        ("fast-gemma4-vision", None, "public-apple-vision"),
    )
    for fast_pkg, backend_filter, pub_pkg in pair_map:
        fast_cands = [
            r
            for r in ok_rows
            if r.get("package") == fast_pkg
            and (backend_filter is None or (r.get("diagnostics") or {}).get("backend") == backend_filter)
        ]
        pub_cands = [r for r in ok_rows if r.get("package") == pub_pkg]
        for fr in fast_cands:
            for pr in pub_cands:
                if any(fr.get(key) != pr.get(key) for key in ("category", "benchmark", "n_items")):
                    continue
                f_eot = float(fr.get("effective_outcome_per_sec", 0.0))
                p_eot = float(pr.get("effective_outcome_per_sec", 0.0))
                f_sc = float(fr.get("objective_score_e2e", 0.0))
                p_sc = float(pr.get("objective_score_e2e", 0.0))
                evidence = evaluate_promotion_gates(pr, fr, lossless_tolerance=lossless_tolerance).to_dict()
                f_diag = fr.get("diagnostics") or {}
                p_diag = pr.get("diagnostics") or {}
                f_cap = f_diag.get("coarse_max_tokens")
                if f_cap is None:
                    f_cap = _DEFAULT_COARSE_MAX_TOKENS.get(fast_pkg)
                p_cap = p_diag.get("coarse_max_tokens")
                if p_cap is None:
                    p_cap = _DEFAULT_COARSE_MAX_TOKENS.get(pub_pkg)
                row: dict[str, Any] = {
                    "fast_package": fast_pkg,
                    "fast_config": fr.get("config_label"),
                    "stock_package": pub_pkg,
                    "stock_config": pr.get("config_label"),
                    "category": fr.get("category"),
                    "benchmark": fr.get("benchmark"),
                    "fast_score_e2e": round(f_sc, 4),
                    "stock_score_e2e": round(p_sc, 4),
                    "delta_score_e2e": round(f_sc - p_sc, 4),
                    "se_paired": evidence["se_paired"],
                    "n_paired": evidence["n_paired"],
                    "ci95_low": evidence["ci95_low"],
                    "ci95_high": evidence["ci95_high"],
                    "lossless_quality_floor": evidence["quality_floor"],
                    "lossless_quality_status": evidence["quality_status"],
                    "retrieval_coverage_pass": evidence["retrieval_coverage_pass"],
                    "fast_eot": round(f_eot, 3),
                    "stock_eot": round(p_eot, 3),
                    "eot_speedup_vs_stock": round(f_eot / max(p_eot, 1e-6), 3),
                }
                if fr.get("category") == 5:
                    f_rates = _extract_cat5_rates(fr)
                    p_rates = _extract_cat5_rates(pr)
                    row.update(
                        {
                            "fast_family": f_rates["model_family"],
                            "stock_family": p_rates["model_family"],
                            "fast_pages_per_sec": f_rates["pages_per_sec"],
                            "stock_pages_per_sec": p_rates["pages_per_sec"],
                            "pages_per_sec_ratio": round(
                                f_rates["pages_per_sec"] / max(p_rates["pages_per_sec"], 1e-6), 3
                            )
                            if p_rates["pages_per_sec"] > 0.0
                            else None,
                            "fast_sec_per_page": f_rates["sec_per_page"],
                            "stock_sec_per_page": p_rates["sec_per_page"],
                            "fast_tokens_per_page": f_rates["tokens_per_page"],
                            "stock_tokens_per_page": p_rates["tokens_per_page"],
                            "fast_decode_tok_s": f_rates["decode_tok_s"],
                            "stock_decode_tok_s": p_rates["decode_tok_s"],
                            "decode_tok_s_ratio": round(
                                f_rates["decode_tok_s"] / max(p_rates["decode_tok_s"], 1e-6), 3
                            )
                            if (f_rates["decode_tok_s"] > 0.0 and p_rates["decode_tok_s"] > 0.0)
                            else None,
                            "fast_e2e_tok_s": f_rates["e2e_tok_s"],
                            "stock_e2e_tok_s": p_rates["e2e_tok_s"],
                            "e2e_tok_s_ratio": round(
                                f_rates["e2e_tok_s"] / max(p_rates["e2e_tok_s"], 1e-6), 3
                            )
                            if (f_rates["e2e_tok_s"] > 0.0 and p_rates["e2e_tok_s"] > 0.0)
                            else None,
                            "fast_mem_gb": f_rates["mem_gb"],
                            "stock_mem_gb": p_rates["mem_gb"],
                        }
                    )
                if f_cap is not None and p_cap is not None:
                    row["fast_coarse_max_tokens"] = int(f_cap)
                    row["stock_coarse_max_tokens"] = int(p_cap)
                    row["coarse_max_tokens_mismatch"] = int(f_cap) != int(p_cap)
                out.append(row)
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    lock, lock_sha = load_lockfile(args.lockfile)
    env_meta = collect_env_telemetry()
    host_mem = _host_mem_gb(env_meta)
    quiescence = check_system_quiescence()

    raw_tokens = [p.strip() for p in args.packages.split(",") if p.strip()]
    pkg_names: list[str] = []
    saw_group_token = False
    for tok_p in raw_tokens:
        if tok_p == "all":
            saw_group_token = True
            for s in PACKAGE_SPECS:
                if s.name not in pkg_names:
                    pkg_names.append(s.name)
        elif tok_p == "public":
            saw_group_token = True
            for s in PUBLIC_BASELINE_SPECS:
                if s.name not in pkg_names:
                    pkg_names.append(s.name)
        elif tok_p == "coreml":
            saw_group_token = True
            for s in COREML_BASELINE_SPECS:
                if s.name not in pkg_names:
                    pkg_names.append(s.name)
        elif tok_p == "ocr-families":
            saw_group_token = True
            for name in OCR_FAMILY_PACKAGES:
                if name not in pkg_names:
                    pkg_names.append(name)
        elif tok_p == "vision":
            saw_group_token = True
            for name in VISION_BENCHMARK_PACKAGES:
                if name not in pkg_names:
                    pkg_names.append(name)
        elif tok_p in _SPEC_BY_NAME or tok_p == "openai-chat":
            if tok_p not in pkg_names:
                pkg_names.append(tok_p)
        else:
            raise ValueError(
                f"Unknown package {tok_p!r}; choose from all, public, coreml, ocr-families, vision, {', '.join(_SPEC_BY_NAME)}, openai-chat"
            )

    if getattr(args, "include_public_baselines", False):
        expanded: list[str] = []
        for pn in pkg_names:
            if pn not in expanded:
                expanded.append(pn)
            for pub_p in PUBLIC_FOR_FAST_PACKAGE.get(pn, ()):
                if pub_p not in expanded:
                    expanded.append(pub_p)
        pkg_names = expanded

    if args.skip_gpt_oss or (0 < host_mem < 23.0 and saw_group_token):
        pkg_names = [p for p in pkg_names if p not in ("fast-gpt-oss", "public-gpt-oss")]

    abba_arms = [parse_abba_arm(x) for x in (args.abba_compare or [])]
    if args.dry_run:
        plan = {
            "dry_run": True,
            "tier": args.tier,
            "smoke": bool(args.smoke),
            "max_coarse_tokens": args.max_coarse_tokens,
            "packages": pkg_names,
            "abba_arms": [{"label": lbl, "flags": fl} for lbl, fl in abba_arms],
            "dataset_lock_sha256": lock_sha,
            "cache_dir": str(resolve_cache_dir(args.cache_dir)),
            "environment": env_meta,
            "quiescence_warnings": quiescence,
        }
        print(json.dumps(plan, indent=2))
        return 0

    if quiescence and not args.allow_contended:
        print(f"QUIESCENCE WARNING ({len(quiescence)} active contenders):", file=sys.stderr)
        for line in quiescence:
            print(f"  {line}", file=sys.stderr)
        print("Aborting: re-run when quiet or pass --allow-contended.", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    cooldown = max(0.0, float(args.cooldown_sec))
    pub_py = args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or args.python

    with _hw_lock():
        for idx, pkg in enumerate(pkg_names, 1):
            if idx > 1 and cooldown > 0.0:
                time.sleep(cooldown)
            print(f"[{idx}/{len(pkg_names)}] Running outcome benchmarks for {pkg} ({args.tier})...", file=sys.stderr, flush=True)
            worker_py = (
                pub_py
                if pkg.startswith(("public-", "coreai-"))
                and pkg not in ("public-apple-vision", "public-pypdfium2", "public-tesseract")
                else args.python
            )

            if len(abba_arms) >= 2:
                # Interleaved A-B-B-A execution in isolated child subprocesses.
                (lbl_a, fl_a), (lbl_b, fl_b) = abba_arms[0], abba_arms[1]
                rounds = max(2, int(args.abba_rounds or 2))
                schedule = ["A", "B", "B", "A"] * max(1, rounds // 2)
                if rounds % 2 == 1:
                    schedule.extend(["A", "B"])
                runs_a: list[list[dict[str, Any]]] = []
                runs_b: list[list[dict[str, Any]]] = []
                for step_i, tag in enumerate(schedule):
                    if step_i > 0 and cooldown > 0.0:
                        time.sleep(cooldown)
                    lbl, fl = (lbl_a, fl_a) if tag == "A" else (lbl_b, fl_b)
                    print(f"  ABBA step {step_i + 1}/{len(schedule)} [{tag}:{lbl}]...", file=sys.stderr, flush=True)
                    recs = _spawn_isolated_worker(
                        python_exe=worker_py, pkg_name=pkg, label=lbl, extra_flags=fl, args=args
                    )
                    (runs_a if tag == "A" else runs_b).append(recs)
                agg_a = _aggregate_abba_records(runs_a)
                agg_b = _aggregate_abba_records(runs_b)
                for ra, rb in zip(agg_a, agg_b):
                    if ra.get("status") == "ok" and rb.get("status") == "ok":
                        ev = evaluate_promotion_gates(
                            ra,
                            rb,
                            profile=args.gate_profile,
                            lossless_tolerance=args.lossless_tolerance,
                            min_eot_ratio=float(args.min_eot_ratio),
                            host_mem_gb=host_mem if host_mem > 0 else None,
                        )
                        rb["se_paired_vs_baseline"] = ev.to_dict()["se_paired"]
                        rb["gate_verdict"] = ev.verdict
                        rb["gate_reasons"] = list(ev.reasons)
                        comparisons.append(
                            {
                                "package": pkg,
                                "category": rb.get("category"),
                                "benchmark": rb.get("benchmark"),
                                "baseline_label": ra.get("config_label"),
                                "candidate_label": rb.get("config_label"),
                                "abba_order": schedule,
                                **ev.to_dict(),
                            }
                        )
                results.extend(agg_a)
                results.extend(agg_b)
            else:
                lbl, fl = abba_arms[0] if abba_arms else ("", args.extra_args or "")
                recs = _spawn_isolated_worker(
                    python_exe=worker_py, pkg_name=pkg, label=lbl, extra_flags=fl, args=args
                )
                results.extend(recs)

    public_comparisons = compute_public_comparisons(results, lossless_tolerance=args.lossless_tolerance)

    # Strip verbose per-item score vectors only when requested; keep compact by default for `compare`.
    if args.omit_item_scores:
        for r in results:
            r.pop("per_item_effective_scores", None)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_epoch": int(time.time()),
        "tier": args.tier,
        "smoke": bool(args.smoke),
        "environment": {
            "hw_model": env_meta.get("hw_model"),
            "chip": env_meta.get("brand"),
            "mem_gb": env_meta.get("mem_gb"),
            "macos_version": env_meta.get("macos_version"),
            "mlx_version": env_meta.get("mlx_version"),
            "python_version": platform.python_version(),
        },
        "dataset_lock_sha256": lock_sha,
        "results": results,
        "comparisons": comparisons,
        "public_comparisons": public_comparisons,
    }
    md = format_outcomes_markdown(results)

    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_m = Path(args.markdown_out)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md + "\n", encoding="utf-8")

    print(md)
    failed = [
        r for r in results
        if r.get("status") == "failed"
        or (r.get("category") == 1 and r.get("status") == "ok" and retrieval_coverage_errors(r))
    ]
    advisory = bool(getattr(args, "advisory_gates", False))
    gate_failed = [] if advisory else [r for r in results if r.get("gate_verdict") == "FAIL"]
    is_subset = bool(args.smoke or args.max_items is not None)
    comp_failed = [] if advisory else [
        c for c in comparisons
        if (c.get("verdict") == "FAIL" if is_subset else c.get("verdict") != "PASS")
    ]
    return 1 if (failed or gate_failed or comp_failed) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark-outcomes",
        description="External Public Objective Outcome Benchmark Suite for fast-silicon.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sd = sub.add_parser("setup-data", help="Stage & verify pinned HF dataset shards (0 GPU work)")
    sd.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    sd.add_argument("--datasets", default="all", help="Comma-separated dataset keys or 'all'")
    sd.add_argument("--cache-dir", default=None)
    sd.add_argument("--lockfile", default=None)
    sd.add_argument("--no-verify-checksums", action="store_true")
    sd.set_defaults(func=_cmd_setup_data)

    sg = sub.add_parser("stage-gov", help="Stage paired Public-Domain U.S. Gov PDF<->XML/HTML/XLSX/scan benchmarks (0 GPU work)")
    sg.add_argument("--branches", default="legislative,executive,judicial", help="Comma-separated branches")
    sg.add_argument("--sources", default="all", help="Comma-separated sources or 'all'")
    sg.add_argument("--per-source", type=int, default=3, help="Sampled items per gov source")
    sg.add_argument("--year", type=int, default=2025, help="Target sitemap year for legislative/judicial")
    sg.add_argument("--loc-btp-path", default=None, help="Optional LoC By the People (btp-dataset) CSV/Parquet path")
    sg.add_argument("--no-loc-btp", action="store_true", help="Skip default LoC By the People archival scan staging")
    sg.add_argument("--cache-dir", default=None)
    sg.set_defaults(func=_cmd_stage_gov)

    rn = sub.add_parser("run", help="Run isolated Tier-1 / Tier-2 objective outcome benchmarks under GPU lock")
    rn.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    rn.add_argument(
        "--packages",
        default="all",
        help="Comma-separated package names or groups (all, public, coreml, ocr-families, vision)",
    )
    rn.add_argument("--abba-compare", nargs="+", default=None, help="Two 'label:flags' arms for interleaved ABBA")
    rn.add_argument("--abba-rounds", type=int, default=2, help="Interleaved ABBA rounds when --abba-compare is set")
    rn.add_argument("--gate-profile", choices=("lossless", "quantized"), default="lossless")
    rn.add_argument("--lossless-tolerance", type=float, default=DEFAULT_LOSSLESS_TOLERANCE,
                    help="Fixed acceptable score loss for the paired 95%% interval (default: 0.003)")
    rn.add_argument("--min-eot-ratio", type=float, default=1.0)
    rn.add_argument("--include-unconstrained-baseline", action="store_true")
    rn.add_argument(
        "--include-public-baselines",
        action="store_true",
        help="Also run matching public MLX, Apple Core AI and community Core ML baselines",
    )
    rn.add_argument("--public-python", default=None, help="Python executable for stock MLX baseline workers")
    rn.add_argument("--coreai-python", default=None, help="Python executable for Apple CoreAI (macOS 27.2+) workers")
    rn.add_argument("--pplx-mlx-path", default=None, help="Optional repo path for public pplx-context-mlx")
    rn.add_argument("--extra-args", default="", help="Forwarded worker flags (e.g. '--mode hetero2')")
    rn.add_argument("--max-coarse-tokens", type=int, default=None, help="Optional token cap for LegalBench-RAG coarse passages (e.g. 512)")
    rn.add_argument("--max-items", type=int, default=None, help="Optional per-sub-benchmark item cap")
    rn.add_argument("--smoke", action="store_true", help="Fast end-to-end smoke slice across selected packages")
    rn.add_argument("--openai-url", default=None)
    rn.add_argument("--openai-model", default=None)
    rn.add_argument("--cache-dir", default=None)
    rn.add_argument("--lockfile", default=None)
    rn.add_argument("--local-files-only", action="store_true")
    rn.add_argument("--python", default=sys.executable)
    rn.add_argument("--cooldown-sec", type=float, default=1.0)
    rn.add_argument("--timeout-sec", type=int, default=900)
    rn.add_argument("--no-wired-limit", action="store_true")
    rn.add_argument("--allow-contended", action="store_true")
    rn.add_argument("--skip-gpt-oss", action="store_true")
    rn.add_argument("--omit-item-scores", action="store_true")
    rn.add_argument("--advisory-gates", action="store_true", help="Exit 0 even when gate_verdict == FAIL")
    rn.add_argument("--dry-run", action="store_true")
    rn.add_argument("--json-out", default=None)
    rn.add_argument("--markdown-out", default=None)
    rn.set_defaults(func=_cmd_run)

    cp = sub.add_parser("compare", help="Compare two saved outcomes_*.json reports via Two-Gate promotion rules")
    cp.add_argument("baseline", help="Path to baseline outcomes JSON")
    cp.add_argument("candidate", help="Path to candidate outcomes JSON")
    cp.add_argument("--gate-profile", choices=("lossless", "quantized"), default="lossless")
    cp.add_argument("--lossless-tolerance", type=float, default=DEFAULT_LOSSLESS_TOLERANCE,
                    help="Fixed acceptable score loss for the paired 95%% interval (default: 0.003)")
    cp.add_argument("--min-eot-ratio", type=float, default=1.0)
    cp.add_argument("--host-mem-gb", type=float, default=None)
    cp.add_argument("--markdown", action="store_true", help="Render cross-chip/cross-family Markdown tables instead of JSON")
    cp.add_argument("--markdown-out", default=None, help="Optional path to write Markdown comparison tables")
    cp.set_defaults(func=_cmd_compare)

    wk = sub.add_parser("_worker", help=argparse.SUPPRESS)
    wk.add_argument("--worker-package", required=True)
    wk.add_argument("--worker-label", default="")
    wk.add_argument("--worker-extra-flags", default="")
    wk.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    wk.add_argument("--lockfile", default=None)
    wk.add_argument("--cache-dir", default=None)
    wk.add_argument("--local-files-only", action="store_true")
    wk.add_argument("--max-items", type=int, default=None)
    wk.add_argument("--smoke", action="store_true")
    wk.add_argument("--include-unconstrained-baseline", action="store_true")
    wk.add_argument("--no-wired-limit", action="store_true")
    wk.add_argument("--openai-url", default=None)
    wk.add_argument("--openai-model", default=None)
    wk.add_argument("--coreai-python", default=None)
    wk.add_argument("--pplx-mlx-path", default=None)
    wk.add_argument("--max-coarse-tokens", type=int, default=None)
    wk.add_argument("--worker-out-file", default=None)
    wk.set_defaults(func=_run_worker_main)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
