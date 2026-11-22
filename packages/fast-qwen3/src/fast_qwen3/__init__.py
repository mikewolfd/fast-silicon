"""Hardware-limit Apple Silicon (`M4` / `M5+`) hybrid `GatedDeltaNet` + Attention engine for Qwen3.5 & Qwen3.6."""

from __future__ import annotations

from .engine import FastQwen3Engine, GenerationBatchResult, GenerationItem, split_lcp_ids
from .kernels import (
    QwenDoubleQuantError,
    SpecBucketedLinearGroup,
    dequantize_gdn_state_int8,
    merge_lora_weights_safe,
    permute_q_proj_rows_to_q_gate,
    quantize_gdn_state_int8,
)
from .model import (
    DeltaSlotCache,
    FastQwen3Model,
    GrowingKVCache,
    HybridPrefixSnapshot,
    Qwen3Config,
    restore_hybrid_cache,
    snapshot_hybrid_cache,
)
from .mtp_drafter import Qwen3MtpDrafter, verify_draft_sequence_b1
from .paths import DEFAULT_HF_REPO, MODEL_PRESETS, Qwen3Paths, resolve_paths, resolve_preset_repo
from .pipeline import ParquetExtractSummary, extract_jsonl, extract_parquet
from .server import FastQwen3OpenAIAdapter, FastQwen3Server
from .tokenizer import (
    MIN_RECOMMENDED_THINKING_BUDGET,
    FastQwen3Tokenizer,
    QwenVocabRemapper,
    inject_reasoning_prefix_property,
    validate_thinking_budget,
)

__all__ = [
    "DEFAULT_HF_REPO",
    "DeltaSlotCache",
    "FastQwen3Engine",
    "FastQwen3Model",
    "FastQwen3OpenAIAdapter",
    "FastQwen3Server",
    "FastQwen3Tokenizer",
    "GenerationBatchResult",
    "GenerationItem",
    "GrowingKVCache",
    "HybridPrefixSnapshot",
    "MIN_RECOMMENDED_THINKING_BUDGET",
    "MODEL_PRESETS",
    "ParquetExtractSummary",
    "Qwen3Config",
    "Qwen3MtpDrafter",
    "Qwen3Paths",
    "QwenDoubleQuantError",
    "QwenVocabRemapper",
    "SpecBucketedLinearGroup",
    "dequantize_gdn_state_int8",
    "extract_jsonl",
    "extract_parquet",
    "inject_reasoning_prefix_property",
    "merge_lora_weights_safe",
    "permute_q_proj_rows_to_q_gate",
    "quantize_gdn_state_int8",
    "resolve_paths",
    "resolve_preset_repo",
    "restore_hybrid_cache",
    "snapshot_hybrid_cache",
    "split_lcp_ids",
    "validate_thinking_budget",
    "verify_draft_sequence_b1",
]
