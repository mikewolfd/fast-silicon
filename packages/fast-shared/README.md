# fast-shared

Shared Apple Silicon runtime, hardware dispatch policy, compact vector indexing, KV compression, Parquet ETL pipelines, HTTP serving, and memory measurement used across [`fast-silicon`](https://github.com/mikewolfd/fast-silicon).

Runs 100% locally with zero telemetry and zero runtime dependency on `torch` or `transformers`. See [full cross-silicon benchmarks](../../docs/BENCHMARKS.md).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 15+ for Core ML ANE and native modules (`macOS 14+` for GPU-only MLX), Python 3.12+, and `uv`. Metal GPU and Core ML ANE modules require Apple Silicon macOS, while CPU retrieval and BM25 (`fast_shared.cpu_kernels`, `CompactRetrievalIndex`, `BM25Index`, `HybridRetrievalPipeline`) run cross-platform on macOS and Linux (`x86_64` AVX-512/AVX2 and ARM64 NEON/portable C++17) via `cpu_backend` (`auto`, `avx512`, `avx2`, `neon`, `neon_i8mm`, `portable_cpp`, `numpy`) or `FAST_SILICON_CPU_BACKEND`.

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-shared --inexact

.venv/bin/fast-silicon-hw-check          # Probe chip, RAM ceiling, and OS / Core AI support
.venv/bin/fast-bench-check --check-ps    # Check for background GPU/CPU contenders before benchmarking
.venv/bin/fast-ir-guard --strict-bool-mask --coreml packages/fast-shared/src packages/fast-shared/tests
```

```python
import os
from fast_shared import BM25Index, CompactRetrievalIndex, detect_cpu_isa
from fast_shared.bench import check_system_quiescence, read_process_rusage_v6
from fast_shared.runtime_lessons import DynamicSiliconPolicy

warnings = check_system_quiescence()
policy = DynamicSiliconPolicy()  # Probes Apple9 (M4) vs Apple10+ NAX (M5), RAM ceiling, and OS version
plan = policy.resolve(workload="extraction", batch_size=8, seq_len=512)
isa = detect_cpu_isa()           # Cross-platform backend selection: neon / avx512 / avx2 / portable_cpp / numpy
mem = read_process_rusage_v6(os.getpid())  # Physical + Neural Engine footprint
```

## Core capabilities

| Capability | Module | What it provides |
| --- | --- | --- |
| **Dynamic silicon dispatch** | `fast_shared.runtime_lessons` | Routes workloads by chip (`Apple9` vs `Apple10+ NAX`), unified memory budget (`<= 80%` wired ceiling; auto batch/pack sizing), and batch size, alongside first-fit-decreasing sequence packing. |
| **Compact & hybrid retrieval (`MaxSim` + ANE/`GenRM` rerank)** | `fast_shared.cpu_kernels`, `retrieval_index`, `hybrid_retrieval` | `CompactRetrievalIndex` and `BM25Index` run multi-core C++17 SIMD kernels across Apple Silicon & non-Apple CPUs (`neon`, `avx512` VNNI+VPOPCNTDQ, `avx2`, `portable_cpp`, plus opt-in `neon_i8mm` and `numpy` fallback) via `cpu_backend`: `100K x 1024d` search in **0.30 ms/q** (`binary_int8`) / **0.75 ms/q** (`int8`), **16.43x** batched `Q=16` speedup, **14.86x** skewed `MaxSim` speedup, and **10.79x-13.16x** gated Block-Max WAND BM25 (`6.05x` stacked hybrid E2E), alongside contextual 256-char `MaxSim` (`0.7569` `LegalBench-RAG`), `GenRMRerankFusion` (`0.7803`), `rrf_fuse`, and `HybridRetrievalPipeline`. |
| **Quantized KV & mixed-bit fusion** | `fast_shared.kv_quant` | `QuantizedKVCache` runs attention over packed 8-bit (`q8`) and rotated 4-bit (`q4_quarot`, 3.56x smaller KV, 2.13x faster decode than 8-bit KV at 16K) caches, plus `SpecBucketedLinearGroup`. |
| **Shared vision & OCR primitives** | `fast_shared.qwen35_vision`, `gemma4_vision`, `vision_preprocess` | Ported Qwen3.5 & Gemma-4 vision towers, 3D MRoPE helpers, ink cropping, lazy MLX imports, and public OCR evaluators (`outcome_eval.ocr`). |
| **Hybrid `PrefixTree`, GPU argmax & PLD** | `fast_shared.prefix_tree`, `argmax`, `pld_drafter` | 256-token SHA-256 prefix cache for KV and block-32 INT8 GatedDeltaNet states (3.76x smaller), single-pass NaN/Inf-safe Metal `gpu_argmax_conf`/`gpu_argmax_nan` (2x less vocab DRAM traffic), and grammar-checked prompt-lookup drafting. |
| **Schema grammars & resumable ETL** | `fast_shared.grammar`, `pipeline_llm`, `parquet_pipeline` | Per-document `llguidance` JSON Schema enforcement, 4-gram loop detection, deferred retries on truncated outputs, header-anchored Parquet chunking (+33-46% legal retrieval lift), and resumable shards. |
| **GPU top-k sampling & self-consistency** | `fast_shared.grammar`, `self_consistency` | `gather_topk_stats_mlx` + `try_sample_topk_row` (`2.49x` on `Qwen3.5-4B` `MATH-500`, `1.59x` on `FastGemma4` sampled decode), CPU `argpartition` top-256 fast path, and engine-agnostic `generate_self_consistent` majority voting (`select_majority_answer`, `select_majority_json`, `extract_choice_letter`, auto-enabling `rolling=True` at `>= 4` samples, `adaptive=False` default). |
| **Core ML / Core AI & Metal 4 MPP** | `fast_shared.coreai_pipeline`, `parallel_coreml`, `ir_dtype_guard`, `mpp_linear` | Parallel Core ML bucket compilation (2.0-2.1x faster), FP16 `.aimodel` ANE lowering on macOS 27.2.1, static AST/MIL dtype lints (`fast-ir-guard`), and inline-XOR Apple10 NAX linear wrappers. |
| **4-bucket memory & outcome harness** | `fast_shared.bench`, `outcome_eval` | Tracks Metal allocator memory, process `phys_footprint`, `neural_footprint`, mapped model pages, and ANE daemon deltas (`aned`), paired with deterministic evaluators and interleaved regression gates. |

## Provided modules

| Module | Key exports / CLI |
| --- | --- |
| `fast_shared.runtime_lessons` | `DynamicSiliconPolicy`, `resolve_dynamic_execution_plan`, `recommend_silicon_budget`, `pack_sequences_ffd` |
| `fast_shared.cpu_kernels` / `retrieval_index` / `hybrid_retrieval` | `detect_cpu_isa`, `resolve_cpu_backend`, `CompactRetrievalIndex`, `BM25Index`, `HybridRetrievalPipeline`, `score_document_windows` |
| `fast_shared.qwen35_vision` / `gemma4_vision` / `vision_preprocess` | `FastQwen35VisionModel`, `FastGemma4VisionModel`, `crop_to_ink`, `preprocess_gemma4_image`, `prepare_pplx_image_item` |
| `fast_shared.kv_quant` | `QuantizedKVCache`, `HadamardTransformBank`, `SpecBucketedLinearGroup`, `fused_quantized_sdpa` |
| `fast_shared.prefix_tree` / `pld_drafter` | `PrefixTree`, `admit_prefix`, `quantize_gdn_state_block32`, `PromptLookupDrafter`, `filter_drafts_with_grammar` |
| `fast_shared.embeddings` | `MappedEmbedding`, `truncate_and_normalize`, `tile_aligned_slices`, `set_mlx_cache_limit` |
| `fast_shared.parquet_pipeline` | `embed_parquet`, `CorpusBackfillConfig`, `plan_header_anchored_windows` |
| `fast_shared.grammar` / `pipeline_llm` / `extractconf` | `GrammarHelper`, `gather_topk_stats_mlx`, `try_sample_topk_row`, `detect_catastrophic_token_loop`, `extract_parquet`, `run_deferred_retries`, `normalize_extraction_telemetry`, `ExtractConfCalibrator` |
| `fast_shared.self_consistency` | `generate_self_consistent`, `select_majority_answer`, `select_majority_json`, `extract_choice_letter`, `canonical_json_key` |
| `fast_shared.server` | `ManagedHTTPServer`, `UnifiedOpenAIServer`, `ChatMicroBatcher` (forwards `reasoning`, `thinking_budget_tokens`, `answer_max_tokens`) |
| `fast_shared.coreai_pipeline` / `parallel_coreml` | `build_coreai_opt_config`, `compile_buckets_parallel`, `inspect_coreml_bundle_placement` |
| `fast_shared.bench` / `outcome_eval` | `fast-bench-check`, `FourBucketMemoryProbe`, `read_process_rusage_v6`, `evaluate_promotion_gates`, `evaluate_legalbench_rankings` |
| `fast_shared.hw_check` / `ir_dtype_guard` / `roofline` | `fast-silicon-hw-check`, `supports_apple10_nax`, `detect_apple10_d256_force_fused`, `fast-ir-guard`, `fast-silicon-probe`, `fast-silicon-audit` |
| `fast_shared.argmax` | `gpu_argmax_conf`, `gpu_argmax_nan`, `stats_to_logprob_margin` |
| `fast_shared.metal_mpp` / `metal_mpp_int8` / `mpp_linear` / `ane_ffn` | `HEADER_MPP`, `mpp_linear_bf16`, `mpp_quantized_linear`, `MppQuantizedLinear`, `AneWave`, C++/ObjC++ headers |

## Shared Parquet controls

The shared CLI and `embed_parquet` separate source reading (`--batch-rows`), encoder calls (`--encode-batch-rows`), and shard writes (`--shard-rows`):

| Option | Controls | Default |
| --- | --- | --- |
| `--batch-rows` | Source rows read and prepared together (`32` for whole PDFs; `4096` for short rows) | `4096` |
| `--encode-batch-rows` | Maximum chunks sent per `encode_corpus` call | `4096` |
| `--shard-rows` | Output rows per stored shard (with `--resume`) | `65536` |
| `--max-chunk-tokens` | Final chunk limit including prefix and special tokens | `512` |
| `--overlap` | Repeated body tokens between adjacent chunks | `51` |
| `--header-col` / `--header-text` | Per-row column or static document title/header prepended to each row or chunk | `None` |

`--resume` assumes source inputs, model, and chunk settings have not changed; write to a new output directory when changing them.

For per-package benchmark tables across all 8 model packages, see the [`fast-silicon` README](../../README.md) and [cross-silicon benchmarks](../../docs/BENCHMARKS.md).

## Optimizer skill plugin & tests

`fast-shared` ships [`skills/apple-silicon-ml-optimizer/`](skills/apple-silicon-ml-optimizer/SKILL.md) alongside a [12,032-entry M4/M5 capability matrix](skills/apple-silicon-ml-optimizer/references/m4-m5-capability-matrix.md) and [Vision, Multimodal Embeddings & OCR Playbook](skills/apple-silicon-ml-optimizer/references/vision-and-ocr-playbook.md):

```bash
claude --plugin-dir ./packages/fast-shared
python3 packages/fast-shared/skills/apple-silicon-ml-optimizer/scripts/query_capability_matrix.py --stats
.venv/bin/pytest -v packages/fast-shared/tests   # Requires --extra dev
```

## Limitations, Acknowledgments & License

- **Scope:** Verified on Apple M4 Pro (48 GB, macOS 26.6) and Base Apple M5 (24 GB, macOS 27.2.1). `ManagedHTTPServer` batches requests into completed JSON responses rather than token-by-token streaming.
- **Acknowledgments:** [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) (Copyright 2026 Merchantry LLC, MIT License) inspired the `proc_pid_rusage` OS footprint helpers, memmap `.safetensors` loader (`MappedEmbedding`), `tile_aligned_slices`, and MLX cache discipline.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
