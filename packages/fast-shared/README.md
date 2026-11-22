# fast-shared

**The Apple Silicon runtime, dynamic policy engine, CoreML / Core AI tooling, and benchmark discipline toolkit powering [`fast-silicon`](https://github.com/mikewolfd/fast-silicon).**

`fast-shared` gives local ML engines one reusable foundation instead of re-implementing custom Metal 4 kernels, dynamic M4/M5 silicon dispatch, tokenizers, OpenAI HTTP/SSE servers, resumable Parquet pipelines, or memory-mapped weight loaders (`MappedEmbedding`). It also ships practical guardrails any MLX, CoreML, or Core AI developer can adopt today: `DynamicSiliconPolicy` cross-silicon runtime dispatch, `coreai_pipeline` calibration & ANE lowering guards, `RUSAGE_INFO_V6` 4-bucket daemon telemetry, `fast-ir-guard`, crash-safe `spawn`-parallel CoreML bucket compilation, deterministic external outcome scorers, and an installable Apple Silicon optimization skill plugin backed by a `12,032`-entry capability matrix.

> **Full Cross-Silicon Benchmarks & Promotion Gates:** See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete M4 Pro (`48 GB`) vs. Base M5 (`24 GB`) outcome (`EOT`), throughput, training, RLVR, and `CoreAI-Opt` / `macOS 27.2.1` tables.

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

fast-silicon-hw-check          # Probe Apple Silicon host (M4 Pro / Base M5, OS & Core AI gates)
fast-bench-check --check-ps    # Check for active GPU/CPU contenders before benchmarking
fast-ir-guard --strict-bool-mask --coreml path/to/src path/to/tests
```

```python
import os
from fast_shared.bench import check_system_quiescence, read_process_rusage_v6
from fast_shared.runtime_lessons import DynamicSiliconPolicy

warnings = check_system_quiescence()
policy = DynamicSiliconPolicy()  # Auto-probes Apple9 vs Apple10+ NAX, RAM tier, and macOS gates
plan = policy.resolve(workload="extraction", batch_size=8, seq_len=512)
mem = read_process_rusage_v6(os.getpid())  # RUSAGE_INFO_V6 physical + neural footprint
```

## Core Platform Highlights

| Capability | Module | What It Solves |
|---|---|---|
| **Dynamic Silicon Dispatch** | `fast_shared.runtime_lessons` | `DynamicSiliconPolicy` & `resolve_dynamic_execution_plan` automatically route per-host and per-call across batch regime (`B=1` vs. `B>=2` on `Apple10+` NAX / `B>=5` on `Apple9`), chipset (`Apple9` M4 vs. `Apple10+` M5 `NAX`), RAM budget (`<=16 GB` / `24 GB` `19.2 GB` cap / `>=36 GB`, strictly enforcing `<= 80%` wired RAM), and macOS gates (`26.2`, `26.4`, `27.2.1+`). Ships `pack_sequences_ffd`, `pack_encoder_batches`, `select_speculative_policy`, and `validate_grpo_ane_offload`. |
| **CoreAI-Opt & `.aimodel` Pipeline** | `fast_shared.coreai_pipeline` | Offline `CoreAI-Opt` calibration/sensitivity + `.aimodel` / `.mlpackage` export (`run_coreai_opt_calibration_and_export`). Defaults `coreai` dense/int exports to **`FP16` compute** so macOS `27.2.1` lowers to ANE (unlocking `1,652` ANE ops and `4.09x` faster latency on `Laya-Typed-421M` vs. `FP32` `MPSGraph`), enforces a **`CoreML` `w4_per_block` corruption guard** (blocking the macOS `26.6` / CoreML `9.0` logit-inversion bug unless `allow_experimental_coreml_w4_block=True`), clamps `eps >= 2^-14`, and inspects `MLModelStructureProgram` compute plans. |
| **`RUSAGE_INFO_V6` 4-Bucket Telemetry** | `fast_shared.bench` | `read_process_rusage_v6`, `FourBucketMemoryProbe`, `AneDaemonMemoryProbe`, `collect_daemon_four_bucket_metrics`, `DeviceMemoryGuard`, and `summarize_cold_warm_latencies` track active Metal/MLX, process `phys_footprint` + `neural_footprint` (`neural_nofootprint`), file-backed mapped model & Core AI cache pages (`mapped_model_mb` / `mapped_all_mb`), and out-of-process ANE daemons (`aned`, `ANECompilerService`). |
| **Static Dtype & IR Guard** | `fast_shared.ir_dtype_guard` | `fast-ir-guard --strict-bool-mask --coreml` statically catches silent `FP32` SDPA mask upcasts, mixed `bf16 x fp16`, `0D` scalars, loop `mx.eval`, CoreML MIL ANE traps, and Pattern 7 block/channel divisibility (`check_divisibility`, `validate_export_quant_dtypes`). |
| **Parallel Core ML bucket compiler** | `fast_shared.parallel_coreml` | Four workers built nine BGE shapes in 4.38 versus 9.37 s on M4 Pro and 2.24 versus 5.02 s on M5. Both arms skipped model loading during conversion. The resulting embeddings matched bit for bit on all nine shapes on each host. [Compiler measurements](../../docs/benchmarks/capability-corrections-20261121.json) |
| **Apple Silicon Capability Matrix** | `skills/apple-silicon-ml-optimizer/` | Queryable **`12,032`-entry** M4/M5 capability catalog (`4,682` Arm A64 encodings, `3,648` Metal SDK symbols, `1,928` Core ML / Core AI / MPSGraph APIs, `390` MLX APIs, `326` Metal GPU rows, `221` ISA diffs, `123` Darwin `hw.optional.arm.*` keys, and six-chip OS `15.0–27.2.1` version gates). |

## At a glance

| Question | Answer |
|---|---|
| What hardware is this verified on? | **Apple M4 Pro** (`Mac16,8`, `48 GB`, `14c` CPU `[10P+4E]`, `20c` `Apple9` GPU, `16c` ANE, `210.2 GB/s` empirical DRAM BW, `macOS 26.6`, `MLX 0.32.4`) and **Apple Base M5** (`Mac17,2`, `24 GB` / `19.07 GB` Metal cap, `10c` CPU `[4S+6E]`, `10c` `Apple10` GPU w/ `NAX` `12.25 TF` FP16 / `14.49 TF` Q4, `16c` `h17` ANE, `137.3 GB/s` empirical DRAM BW, `macOS 27.2.1`, `CoreAI 1.0.0b3` + `MLX 0.32.4`). |
| How does memory accounting work? | `FourBucketMemoryProbe` + `read_process_rusage_v6()` combine `proc_pid_rusage(..., RUSAGE_INFO_V6)` (`phys_footprint`, `neural_footprint`), Mach `task_vm_info` (`neural_nofootprint`), mapped model/Core AI cache pages (`mapped_model_mb` / `mapped_all_mb`), and ANE daemon deltas (`aned`, `ANECompilerService`). |
| How does runtime dispatch adapt across Macs? | `DynamicSiliconPolicy` enforces `<= 80%` wired RAM, scales batch/pack sizes (`pack=2048` on `>=36 GB` vs. `pack=1024, grad_accum=2` on `24 GB`), gates `MTP` speculative decoding (`B=1` on `Apple10+` NAX; `B=1..4` on `Apple9`; off at `B>=5`), enables Prefix-Tree reuse (`shared_prefix >= 32`), and gates Core AI runtime promotion (`27.2.1+`). |
| Can it inject faults & verify quiescence? | Yes: `check_system_quiescence()`, `interleaved_abba()`, `DeviceMemoryGuard`, plus CPU fault-injection contracts tested across the suite. |

## Provided Modules

| Module / Binary | Purpose | Key Exports / CLI |
|---|---|---|
| `fast_shared.runtime_lessons` | Cross-silicon M4/M5 dynamic policy, FFD packing, speculative & GRPO offload guards | `DynamicSiliconPolicy`, `resolve_dynamic_execution_plan`, `recommend_silicon_budget`, `pack_sequences_ffd`, `pack_encoder_batches`, `select_speculative_policy`, `validate_grpo_ane_offload` |
| `fast_shared.coreai_pipeline` | CoreAI-Opt calibration, `FP16` `.aimodel` ANE lowering, CoreML `w4_per_block` guard, placement | `build_coreai_opt_config`, `run_coreai_opt_calibration_and_export`, `run_coreai_opt_preset_sweep`, `inspect_coreml_bundle_placement`, `clamp_fp16_eps` |
| `fast_shared.parallel_coreml` | `spawn`-only parallel CoreML multi-bucket compiler + atomic bundle publish | `compile_buckets_parallel`, `atomic_publish_dir`, `default_compile_jobs`, `resolve_compile_workers` |
| `fast_shared.bench` | Quiescence, `RUSAGE_INFO_V6` 4-bucket telemetry, `DeviceMemoryGuard`, `ABBA` harness | `fast-bench-check`, `FourBucketMemoryProbe`, `read_process_rusage_v6`, `collect_daemon_four_bucket_metrics`, `summarize_cold_warm_latencies`, `interleaved_abba` |
| `fast_shared.outcome_eval` | Zero-LLM-judge deterministic 4-category outcome & `EOT` scorers + promotion gates | `evaluate_promotion_gates`, Category 1–4 task evaluators |
| `fast_shared.hw_check` | Host hardware, OS gate (`26.2`/`26.4`/`27.2.1`), Core AI & `MLComputePlan` detection | `fast-silicon-hw-check [--json \| --strict]`, `probe_host_hardware`, `probe_core_ai_support`, `extract_coreml_compute_plan_ops` |
| `fast_shared.ir_dtype_guard` | Static AST dtype/loop guard, CoreML MIL & Pattern 7 divisibility checks | `fast-ir-guard --strict-bool-mask --coreml`, `check_divisibility`, `validate_export_quant_dtypes` |
| `fast_shared.roofline` | Zero-GPU safetensors forensics, M4/M5 bandwidth & GEMM roofline probes | `fast-silicon-probe`, `fast-silicon-audit` |
| `fast_shared.metal_mpp` / `_bf16` | Metal 4 `MPP::matmul2d` builders (`FP16`/`BF16`, GeGLU, bias epilogues) | `HEADER_MPP`, `mpp_linear_bf16`, `mpp_fused_geglu_bf16` |
| `fast_shared.include/*` | Header-only C++/ObjC++ CoreML & Accelerate helpers | `fast_shared_apple.hpp`, `fast_shared_coreml.hpp` |
| `fast_shared.embeddings` | Safetensors memmap gather, Matryoshka slice+norm, tile slices, cache limits | `MappedEmbedding`, `truncate_and_normalize`, `tile_aligned_slices`, `set_mlx_cache_limit` |
| `fast_shared.tokenizer` | Rust `tokenizers` base + sliding character-offset chunking | `BaseFastTokenizer` |
| `fast_shared.parquet_pipeline` | Multi-file streaming/sharded/resumable Parquet + UTF-8 `page_offsets` | `embed_parquet` |
| `fast_shared.server` | Managed HTTP/SSE server + OpenAI router + coalescing micro-batcher | `ManagedHTTPServer`, `UnifiedOpenAIServer`, `ChatMicroBatcher` |
| `fast_shared.prefix_tree` | Chained SHA-256 `256`-token hybrid KV + block-32 `INT8` GDN prefix tree w/ rolling hit-rate circuit breaker | `PrefixTree`, `PrefixMatch`, `quantize_gdn_state_block32`, `dequantize_gdn_state_block32` |
| `fast_shared.grammar` / `pipeline_llm` | Optional `[llm]` `llguidance` lazy GPU argmax masking + `A4_v2` `p=1..8` cycle/4-gram loop breaker + rolling per-doc schemas | `GrammarHelper`, `compute_loop_breaker_penalties`, `extract_parquet` |

### v2.2 Cross-Silicon Runtime & Kernel Highlights

Structured numbers live in [`../fast-silicon/docs/benchmarks/v22-validation-20261121.json`](../../docs/benchmarks/v22-validation-20261121.json):

- **`LiveMemoryGovernor` & 24 GB QLoRA `pack=1024`:** Wave memory admission check runs in `1.924 us`/wave (`12.26 us` raw) on M4 Pro and `1.458 us`/wave (`9.15 us` raw) on Base M5. On 24 GB Base M5 (`19.2 GB` cap, `1.5 GB` cache), `qlora-packing` (`pack_tokens=1024, grad_accum=2, ckpt_tokens=512`) completes with zero OOMs (`4.848 GiB` unpacked at `87.67` sup tok/s, `4.849 GiB` packed at `80.52` sup tok/s, `6.759 GiB` shared-KV at `81.63` sup tok/s; `loss_diff=0.0, grad_cos=1.0`).
- **Rolling Per-Doc Grammar & Whitespace Guard:** Across `96` multi-schema LiquidAI tasks (`38` schemas), per-document grammar compilation speeds up M4 Pro **`1.925x`** (`90.026s -> 46.953s`, `2.04 docs/s`, `0.9271 -> 0.9248` F1) and Base M5 **`2.486x`** (`86.584s -> 34.790s`, `2.76 docs/s`, `0.9289 -> 0.9263` F1). Preserving `whitespace_flexible=None` by default avoids the `-0.164` F1 regression (`0.7647` M4 / `0.7646` M5) observed when forcing `whitespace_flexible=False`.
- **Embedding Prefetch & No-Cache Guard:** `lookahead_batches=4` (`N=1024`) accelerates `fast-bge-small` **`1.331x`** on M4 Pro (`280.4 -> 373.2 docs/s`, `min_cos=0.9999754`) and **`1.178x`** on Base M5 (`359.1 -> 423.0 docs/s`, `min_cos=0.9999746`), and speeds `fast-pplx` GPU **`1.022x`** on both hosts (`min_cos >= 0.9999965`). `use_nocache=False` remains the default after `use_nocache=True` regressed M4 Pro `embeddinggemma` runtime (`23.793s -> 27.122s`).
- **Metal 4 `MPP::matmul2d` Quantized Gating & ANE FFN:** `metal_mpp_int8` keeps `W4A8` active on `bits=4` Apple10 prefill shapes (`>=8M`), while `W8A8` defaults OFF (`FAST_SILICON_MPP_W8A8=1` / `allow_w8a8=True` opt-in) with FP32 `fma(128.0|8.0, ws, bq)` weight metadata packing. Post-norm Hadamard `W8A8` ANE FFN raises worst 35-layer FFN cosine from `0.996032` to `0.999740`, `AneWave` wins **`1.66x–2.40x`** over GPU on M4 Pro, and `p28` `MTLSharedEvent` saves `7.84 ms` (M4 Pro) / `8.64 ms` (Base M5) CPU sync overhead at `M=1024`.

### v2.3 Accuracy, Retry, Chunking & ANE FFN Highlights

Structured numbers live in [`../fast-silicon/docs/benchmarks/v23-validation-20261121.json`](../../docs/benchmarks/v23-validation-20261121.json):

- **Constrained Grammar `finish_reason` Fix & Single-Model Retry (`pipeline_llm.py`):** Fixed constrained rolling decode reporting `"stop"` instead of `"length"` when hitting `max_tokens` (`ctok=117` @ `max_tokens=112`), and added `ExtractRetryRequest`, `run_deferred_retries`, and `run_deferred_retries_async` for inline or post-sweep retry (`tight48`: `0/16 -> 16/16` valid, `F1 0.0 -> 0.900034`, `+4.1 MB` RSS; `async_deferred_112`: `12/12` improved `F1 0.235243 -> 0.900034`).
- **Gated Rolling SWA (`pipeline_llm.py`):** Unlocked `roll_swa=1024` for `n_docs > 4` when `eff_roll_slots <= 8`, delivering **`+2.68%`** faster throughput (`197.25 -> 202.55 tok/s`) without the `-9.4%` regression observed at `B_slots=16`.
- **Header-Anchored Embedding Windows (`parquet_pipeline.py`):** `plan_header_anchored_windows` + `has_header_anchor` lifts `LegalBench-RAG` (`n=175`) `span_nDCG@10` on **`fast-bge-small`** **`0.304649 -> 0.443886` (`+45.7%`, `char_R@10 0.631442 -> 0.709903`)** and **`fast-embeddinggemma-2`** **`0.38028 -> 0.505361` (`+32.9%`, `char_R@10 0.720889 -> 0.811396`)**; `snap_sentences` stays opt-in after scoring `0.280048` alone.
- **CoreAI Calibration Diversity Gate (`coreai_pipeline.py`):** Added `assess_calibration_batch_diversity`, `evaluate_multi_batch_calibration_gate`, and `fail_on_nonfinite=True` (rejects single-batch, duplicate-batch, and synthetic collapse while diverse 4-batch passes at `min_cos=0.999953`).
- **`ANE FFN` `W8A16` Default & `H > 2048` WHT Fix (`ane_ffn.py`, `fast_ane_handoff.mm`):** `AneWave` defaults to `quantize_a8=False` (`W8A16`, `cos=0.999868..0.999901` on E2B vs `0.999794..0.999832` `W8A8_Had` and `0.9937..0.9951` `W8A8_NoHad`), and `H > 2048` WHT-128 scale bypass fix achieves `cos=0.999890..0.999892` (`row_max_rel_err <= 0.00044`) on `H=2560` / `4096`.

## Shared Parquet Controls

The shared CLI and `embed_parquet` separate source reading (`--batch-rows`), encoder calls (`--encode-batch-rows`), and shard writes (`--shard-rows`):

| CLI Option / Parameter | Controls | Default |
|---|---|---|
| `--batch-rows` / `batch_rows` | Source rows read & prepared together (`32` for whole PDFs; `4096` for short rows) | `4096` |
| `--encode-batch-rows` / `encode_batch_rows` | Max chunks sent per `encode_corpus` call | `4096` |
| `--shard-rows` / `shard_rows` | Output rows per stored shard (with `--resume`) | `65536` |
| `--max-chunk-tokens` / `max_chunk_tokens` | Final chunk limit including prefix and special tokens | `512` |
| `--overlap` / `overlap` | Repeated original body tokens between adjacent chunks | `51` |
| `--max-body-tokens` / `max_body_tokens` | Optional extra cap on original body tokens per window | `None` |

Example chunked document ingestion (UTF-8 `page_offsets` supported via `--per-ocr-page` / `--ocr-pages-only`):

```bash
embeddinggemma-2 parquet \
  --input documents.parquet --output embeddings/ --text-col body \
  --chunk --document-prefix --max-chunk-tokens 512 --chunk-overlap 51 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --keep-cols document_id --include-chunk-text
```

`--resume` assumes source inputs, model, and chunk settings did not change; use a fresh output directory when changing them.

## Sibling Engines: Cross-Silicon Highlights (Base M5 `24 GB` vs. M4 Pro `48 GB`)

Effective outcome throughput (EOT) divides the total score earned on valid outputs by elapsed benchmark seconds. Compare engines on the same task and inputs. Token rates and single-query timings come from separate warm runs.

| Engine | Model & Path | Base M5 (`24 GB`, `Apple10 NAX`) | M4 Pro (`48 GB`, `Apple9`) | vs. Stock `MLX` / `CoreAI` (`EOT`) |
|---|---|---|---|---|
| [`fast-bge-small`](../fast-bge-small) | `BAAI/bge-small-en-v1.5` (`hetero`) | **`13.061 EOT`** · **`200,646 tok/s`** (`0.79 ms` `B=1`, `1.54 GB`) | **`12.634 EOT`** · **`187,150 tok/s`** (`0.49 ms` `B=1`, `1.55 GB`) | **`11.26x`** (M5) / **`9.58x`** (M4 Pro) vs. `mlx-embeddings` |
| [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) | `google/embeddinggemma-2` (`hetero`) | **`4.707 EOT`** · **`48,803 tok/s`** (`2.58 ms` `B=1`, `3.43 GB`) | **`3.583 EOT`** · **`38,517 tok/s`** (`2.11 ms` `B=1`, `2.49 GB`) | **`4.33x` MLX / `6.24x` CoreAI** (M5) · **`4.80x` MLX** (M4 Pro) |
| [`fast-pplx`](../fast-pplx) | `pplx-embed-v2-context-0.6b` (`hetero2`) | 1.724 EOT (2.34 GB), 16,796 mixed-length tokens/s | 1.407 EOT (1.97 GB), 12,624 mixed-length tokens/s | 4.41× (M5) and 4.33× (M4 Pro) the EOT of `pplx-context-mlx` |
| [`fast-decision-models`](../fast-decision-models) | `Laya-Typed-421M` Pure-ANE (`[1585,0,0]`) | **`6.158 EOT`** · **`44.38 q/s`** (`1.00 GB`, `0` GPU ops) | **`5.827 EOT`** · **`41.91 q/s`** (`0.98 GB`, `0` GPU ops) | **`8.29×` CoreAI** (M5) · `0.65×` stock MLX on M5 (`6.158` vs `9.496`, `+0.0036` quality) / `1.08×` on M4 Pro (`5.827` vs `5.380`, `+0.0029` quality) · `5.1×–6.2×` PyTorch MPS (`0` GPU contention) |
| [`fast-decision-models`](../fast-decision-models) | Kev-0.8B W8, automatic prefix cache | **4.395 EOT** (1.78 GB) · 86.84 questions/s in the earlier batch probe | **3.415 EOT** (2.29 GB) · 54.52 questions/s in the earlier batch probe | `0.87×` single-model stock MLX EOT on M5 and **`1.05×` on M4 Pro** (`3.415` vs `3.249`) |
| [`fast-gemma4`](../fast-gemma4) | Schema-constrained Gemma 4 | **`1.835 EOT`** (`144.35 s`, `300/300` valid, `150/150` valid extractions, `0.946` field F1, `0` loops; historical pre-repair Nov 20 minified: `2.717 EOT`, `0.691` F1) | `2.412 EOT` on 300 public tasks *(historical pre-repair Nov 20 minified)* | **`1.22×` stock MLX** (`1.835` vs `1.501` EOT on M5; `0.946` vs `0.923` field F1, `300/300` valid, `0` loops; historical pre-repair Nov 20 minified: `1.81×` M5 / `1.57×` M4 Pro) |
| [`fast-qwen3`](../fast-qwen3) | Hybrid `3:1` GDN+GQA Qwen3.5/3.6 (`OptiQ` + `PrefixTree` + `R2` `lora_accel`) | **`0.513 EOT`** (`50/50` valid, `0.9024` score, `5.26 GB`) · `4B`: **`171.5 tok/s`** (`B=8`, `8.33×`) · `0.8B`: **`573.4 tok/s`** (`11.66×`) | **`0.396 EOT`** (`50/50` valid, `0.9024` score, `5.18 GB`) · `4B`: **`191.0 tok/s`** (`B=8`, `6.07×`) · `0.8B`: **`657.3 tok/s`** (`9.25×`) | **`1.53×` (M5) / `1.38×` (M4 Pro) stock `mlx-lm` EOT** (**`+24.1..+28.0 pp`** higher score: `0.9024` vs `0.6619`/`0.6225`; **`3.35×`** M5 `CoreAI` `51.2 tok/s` `B=1` ceiling) |
| [`fast-gpt-oss`](../fast-gpt-oss) | `openai/gpt-oss-20b` | Excluded at 24 GB | 0.318 EOT, 49/50 valid. A separate batch-128 fixture test reached 454.1 decode tokens/s. | 1.61× the stock EOT on 50 public tasks. The stock capacity probe used a different prompt, batch and output cap. |

### Public Core ML comparisons

The public Core ML exports produced valid outputs for all 325 BGE retrieval cases, 500 EmbeddingGemma retrieval cases and 1,400 decisions per scorer. The same locked inputs show different tradeoffs across models.

| Public Core ML export | M4 Pro EOT | M5 EOT | Comparison with the corresponding `fast-*` package |
|---|---:|---:|---|
| sidekickd BGE FP16 | 2.382 | 2.567 | Average ranking scores differed by 0.0002. `fast-bge-small` delivered 5.30× and 5.09× the score per second on M4 Pro and M5. |
| FluidInference EmbeddingGemma FP16 | 1.637 | 1.666 | `fast-embeddinggemma-2` ranked long passages more accurately. The public export reads the first 512 tokens per passage. |
| OpenInfinity EmbeddingGemma INT8 | 1.649 | 1.740 | This export used 1.03 GB on M4 Pro and 0.84 GB on M5. Its mean ranking score was 0.5183 on both. |
| NMTZ Laya | 7.598 | 7.909 | The public option scorer delivered 1.30× and 1.28× the EOT of Laya in `fast-decision-models` on M4 Pro and M5. The project package scored 0.5050 and used about 1 GB. NMTZ’s export scored 0.4993 and used 2.73–2.94 GB. |
| FluidInference Kev FP16 | 1.674 | 2.269 | Kev in `fast-decision-models` delivered 2.04× and 1.94× the public export’s EOT on M4 Pro and M5 and scored 0.5250 on both. FluidInference’s export scored 0.5207 and 0.5214. |

The results include individual task scores, warmed token rates, load times, memory components and compiler device preferences. [Benchmark methods](../../docs/BENCHMARKS.md#public-core-ml-results) · [Measured results (JSON)](../../docs/benchmarks/coreml-20261120.json)

## Agent Plugin, Capability Matrix & Tests

`fast-shared` exposes [`skills/apple-silicon-ml-optimizer/`](skills/apple-silicon-ml-optimizer/SKILL.md) as a Claude Code / Codex plugin alongside the [`12,032`-entry M4/M5 capability matrix](skills/apple-silicon-ml-optimizer/references/m4-m5-capability-matrix.md):

```bash
claude --plugin-dir /path/to/fast-shared
python3 skills/apple-silicon-ml-optimizer/scripts/query_capability_matrix.py --stats
.venv/bin/pytest -v
```

**Scope & not tested:** Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Apple Base M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)**. Other Apple Silicon SKUs (`M1`–`M3`, `M4 Max`, `M5 Pro`/`Max`), Linux/Windows hosts, and concurrent unserialized Metal/ANE runs are untested.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspired the macOS `proc_pid_rusage` OS physical footprint utilities (`fast_shared.bench`), zero-GPU `.safetensors` memmap embedding loader (`MappedEmbedding`), `tile_aligned_slices` final-slice boundary merge helper, MLX buffer cache limit discipline (`set_mlx_cache_limit`), and scannable README structure.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
