# fast-shared

Shared runtime, hardware dispatch, compact vector indexing, KV compression, Parquet pipelines, HTTP serving, and benchmark tooling for [`fast-silicon`](https://github.com/mikewolfd/fast-silicon).

`fast-shared` provides one tested foundation across all model packages in the workspace — so encoders, decision scorers, and generators share the same Apple Silicon policy engine (`DynamicSiliconPolicy`), memory measurement (`RUSAGE_INFO_V6`), Core ML / Core AI build helpers, `llguidance` grammar loops, `PrefixTree` caching, and compact retrieval indexes instead of re-implementing them per model. [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md)

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

fast-silicon-hw-check          # Probe Apple Silicon chip, RAM ceiling, and OS / Core AI support
fast-bench-check --check-ps    # Check for background CPU/GPU contenders before benchmarking
fast-ir-guard --strict-bool-mask --coreml path/to/src path/to/tests
```

```python
import os
from fast_shared.bench import check_system_quiescence, read_process_rusage_v6
from fast_shared.runtime_lessons import DynamicSiliconPolicy

warnings = check_system_quiescence()
policy = DynamicSiliconPolicy()  # Probes Apple9 (M4) vs Apple10+ NAX (M5), RAM ceiling, and OS version
plan = policy.resolve(workload="extraction", batch_size=8, seq_len=512)
mem = read_process_rusage_v6(os.getpid())  # Physical + Neural Engine footprint
```

## Core capabilities

| Capability | Module | What it provides |
|---|---|---|
| **Dynamic silicon dispatch** | `fast_shared.runtime_lessons` | Routes workloads by chip (`Apple9` M4 vs `Apple10+` M5 `NAX`), unified memory budget (`<= 80%` wired ceiling; `pack=1024, grad_accum=2` on 24 GB Macs), and batch regime, alongside first-fit-decreasing sequence packing. |
| **Compact retrieval & `MaxSim`** | `fast_shared.retrieval_index` | `CompactRetrievalIndex` compresses embeddings 4–6× into symmetric INT8 (with model-aware Matryoshka sizing) and adds an ARM64 NEON 1-bit Hamming prefilter (`rerank_top_r=256` default in `search()`) that searches 100,000 vectors in **0.50–0.66 ms/query** (`top64`, 3.0–4.4× faster than FP32 BLAS), plus multi-window `MaxSim` document scoring. |
| **Quantized KV & mixed-bit fusion** | `fast_shared.kv_quant` | `QuantizedKVCache` runs attention directly over packed 8-bit (`q8`) and rotated 4-bit (`q4_quarot`, 3.56× smaller KV, 2.13× faster than 8-bit KV at 16K context) caches with zero F1 loss vs BF16, alongside `SpecBucketedLinearGroup` mixed-bit projection fusion. |
| **Shared Qwen3.5 & Gemma-4 vision** | `fast_shared.qwen35_vision`, `gemma4_vision`, `vision_preprocess` | Ported `FastQwen35VisionModel` & `FastGemma4VisionModel`, integer/fractional 3D MRoPE helpers (`rope_offset_delta`), `crop_to_ink`, `preprocess_gemma4_image`, Apple10 NAX `head_dim=256` SDPA workaround (`detect_apple10_d256_force_fused`), lazy MLX imports, and Category 5 OCR evaluators (`outcome_eval.ocr`). |
| **Hybrid `PrefixTree` cache** | `fast_shared.prefix_tree` | Chained SHA-256 256-token prefix cache for both KV caches and block-32 INT8 recurrent `GatedDeltaNet` states (3.76× smaller), with bounded 2-hit admission (`admit_prefix(..., admission_min_hits=2)`) and a rolling hit-rate circuit breaker. |
| **Schema grammars & resumable ETL** | `fast_shared.grammar`, `pipeline_llm`, `parquet_pipeline` | Per-document `llguidance` JSON Schema enforcement, cycle/4-gram loop breaking, automatic retry on length-truncated outputs, header-anchored Parquet chunking (+33–46% legal retrieval lift), and resumable shard writers. |
| **Core ML & Core AI build guards** | `fast_shared.coreai_pipeline`, `parallel_coreml`, `ir_dtype_guard` | Crash-safe `spawn`-parallel Core ML bucket compilation (2.0–2.1× faster), FP16 `.aimodel` ANE lowering on macOS 27.2.1, multi-batch calibration checks, and static AST/MIL dtype linting (`fast-ir-guard`). |
| **4-bucket memory & outcome harness** | `fast_shared.bench`, `outcome_eval` | Tracks Metal allocator memory, process `phys_footprint`, `neural_footprint`, mapped model pages, and ANE daemon deltas (`aned`), paired with deterministic Category 1–5 task evaluators and interleaved ABBA gates. |

## Provided modules

| Module | Key exports / CLI |
|---|---|
| `fast_shared.runtime_lessons` | `DynamicSiliconPolicy`, `resolve_dynamic_execution_plan`, `recommend_silicon_budget`, `pack_sequences_ffd` |
| `fast_shared.retrieval_index` | `CompactRetrievalIndex`, `quantize_int8_symmetric`, `pack_binary_bits`, `score_document_windows` |
| `fast_shared.qwen35_vision` / `gemma4_vision` / `vision_preprocess` | `FastQwen35VisionModel`, `FastGemma4VisionModel`, `crop_to_ink`, `preprocess_gemma4_image`, `prepare_pplx_image_item` |
| `fast_shared.kv_quant` | `QuantizedKVCache`, `HadamardTransformBank`, `SpecBucketedLinearGroup`, `fused_quantized_sdpa` |
| `fast_shared.prefix_tree` | `PrefixTree`, `PrefixMatch`, `admit_prefix`, `quantize_gdn_state_block32`, `dequantize_gdn_state_block32` |
| `fast_shared.embeddings` | `MappedEmbedding`, `truncate_and_normalize`, `tile_aligned_slices`, `set_mlx_cache_limit` |
| `fast_shared.parquet_pipeline` | `embed_parquet`, `CorpusBackfillConfig`, `plan_header_anchored_windows` |
| `fast_shared.grammar` / `pipeline_llm` | `GrammarHelper`, `compute_loop_breaker_penalties`, `extract_parquet`, `run_deferred_retries` |
| `fast_shared.server` | `ManagedHTTPServer`, `UnifiedOpenAIServer`, `ChatMicroBatcher` |
| `fast_shared.coreai_pipeline` / `parallel_coreml` | `build_coreai_opt_config`, `compile_buckets_parallel`, `inspect_coreml_bundle_placement` |
| `fast_shared.bench` / `outcome_eval` | `fast-bench-check`, `FourBucketMemoryProbe`, `read_process_rusage_v6`, `evaluate_promotion_gates`, `outcome_eval.ocr` |
| `fast_shared.hw_check` / `ir_dtype_guard` / `roofline` | `fast-silicon-hw-check`, `fast-ir-guard`, `fast-silicon-probe`, `fast-silicon-audit` |
| `fast_shared.metal_mpp` / `metal_mpp_int8` / `ane_ffn` / `include/*` | `HEADER_MPP`, `mpp_linear_bf16`, `PACKS_META_ONLY` inline XOR `mpp_quantized_linear` (`2.34–4.2+ GB` saved), `AneWave`, C++/ObjC++ Core ML & Accelerate headers |

## Shared Parquet controls

The shared CLI and `embed_parquet` separate source reading (`--batch-rows`), encoder calls (`--encode-batch-rows`), and shard writes (`--shard-rows`):

| Option | Controls | Default |
|---|---|---|
| `--batch-rows` | Source rows read and prepared together (`32` for whole PDFs; `4096` for short rows) | `4096` |
| `--encode-batch-rows` | Maximum chunks sent per `encode_corpus` call | `4096` |
| `--shard-rows` | Output rows per stored shard (with `--resume`) | `65536` |
| `--max-chunk-tokens` | Final chunk limit including prefix and special tokens | `512` |
| `--overlap` | Repeated body tokens between adjacent chunks | `51` |
| `--header-col` / `--header-text` | Per-row column or static document title/header prepended to each row or chunk | `None` |

```bash
embeddinggemma-2 parquet \
  --input documents.parquet --output embeddings/ --text-col body \
  --chunk --document-prefix --max-chunk-tokens 512 --chunk-overlap 51 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --keep-cols document_id --include-chunk-text --resume
```

`--resume` assumes source inputs, model, and chunk settings have not changed; write to a new output directory when changing them.

## Model packages in this workspace

Effective outcome throughput (EOT) divides total task score on valid outputs by elapsed seconds. Compare models only on the same task and inputs. [Full benchmark tables and public Core ML comparisons](../../docs/BENCHMARKS.md)

| Package | Model | Base M5 (24 GB) | M4 Pro (48 GB) | vs. Stock baseline (EOT) |
|---|---|---|---|---|
| [`fast-bge-small`](../fast-bge-small) | `bge-small-en-v1.5` (`hetero`) | 13.06 EOT · 200,646 tok/s (0.79 ms query) | 12.63 EOT · 187,150 tok/s (0.49 ms query) | 11.26× (M5) / 9.58× (M4 Pro) stock MLX |
| [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) | `embeddinggemma-2` (`hetero`) | 4.71 EOT · 48,803 tok/s (2.58 ms query) | 3.58 EOT · 38,517 tok/s (2.11 ms query) | 4.33× (M5) / 4.80× (M4 Pro) stock MLX |
| [`fast-pplx`](../fast-pplx) | `pplx-embed-v2-context-0.6b` (`hetero2`) | 1.72 EOT · 19,443 tok/s mixed-64 | 1.41 EOT · 17,793 tok/s mixed-64 | 4.41× (M5) / 4.33× (M4 Pro) stock MLX |
| [`fast-decision-models`](../fast-decision-models) | `Laya-Typed-421M` (Pure-ANE) | 6.16 EOT · 44.4 q/s (1.00 GB, 0 GPU ops) | 5.83 EOT · 41.9 q/s (0.98 GB, 0 GPU ops) | 8.29× CoreAI · 1.08× M4 Pro / 0.65× M5 stock MLX |
| [`fast-decision-models`](../fast-decision-models) | `Kev-0.8B` W8 + `PrefixTree` | 4.40 EOT (1.78 GB) | 3.42 EOT (2.29 GB) | 1.77–2.84× faster than stock MLX on shared passages |
| [`fast-gemma4`](../fast-gemma4) | Schema-constrained Gemma 4 E2B | 1.84 EOT (300/300 valid, 0.946 F1) | 2.41 EOT *(historical 150K minified)* | 1.22× unconstrained stock `mlx-vlm` on M5 |
| [`fast-qwen3`](../fast-qwen3) | Schema-constrained Qwen 3.5 / 3.6 | 0.513 EOT (`4B`, 50/50 valid, 0.902 score) | 0.396 EOT (`4B`, 50/50 valid, 0.902 score) | 1.53× (M5) / 1.38× (M4 Pro) unconstrained stock `mlx-lm` |
| [`fast-gpt-oss`](../fast-gpt-oss) | Schema-constrained `gpt-oss-20b` | Skipped on 24 GB | 0.318 EOT (49/50 valid; 454 tok/s @ B=128) | 1.61× unconstrained stock `mlx-lm` on M4 Pro |
| [`fast-ocr`](../fast-ocr) | `OvisOCR2-8bit` + Apple Vision ANE | 0.090 EOT (`olmOCR-bench(4)`, `4/4`, 5.01 GB) | 0.082 EOT (`olmOCR-bench(4)`, `4/4`, 5.75 GB) | 1.53–2.05× (M5) / 1.22–1.51× (M4 Pro) stock `mlx_vlm` `OvisOCR2` |

## Optimizer skill plugin & tests

`fast-shared` ships [`skills/apple-silicon-ml-optimizer/`](skills/apple-silicon-ml-optimizer/SKILL.md) alongside a queryable [12,032-entry M4/M5 capability matrix](skills/apple-silicon-ml-optimizer/references/m4-m5-capability-matrix.md) and [Vision, Multimodal Embeddings & OCR Playbook](skills/apple-silicon-ml-optimizer/references/vision-and-ocr-playbook.md):

```bash
claude --plugin-dir /path/to/fast-shared
python3 skills/apple-silicon-ml-optimizer/scripts/query_capability_matrix.py --stats
.venv/bin/pytest -v
```

## Scope, Acknowledgments & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**.

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspired the `proc_pid_rusage` OS physical footprint helpers, memmap `.safetensors` embedding loader (`MappedEmbedding`), `tile_aligned_slices` boundary helper, MLX cache discipline, and scannable README layout.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
