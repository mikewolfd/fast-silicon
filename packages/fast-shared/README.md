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
| **Parallel CoreML Bucket Compiler** | `fast_shared.parallel_coreml` | Crash-safe `spawn`-only multi-process compiler (`compile_buckets_parallel`, `atomic_publish_dir`, `skip_model_load=True`) delivering **`2.01x–2.09x`** speedup at `jobs=4` vs. `jobs=1` (`9.18s -> 4.56s` on M4 Pro, `5.12s -> 2.45s` on Base M5; **`5.88x`** vs. `26.80s` legacy on 9 `BGE-small` buckets, `cos = 1.000000`, `9/9` ANE). |
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
| `fast_shared.grammar` / `pipeline_llm` | Optional `[llm]` `llguidance` lazy GPU argmax masking + shard resume | `GrammarHelper`, `extract_parquet` |

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

All engines below are measured under isolated subprocess runs (`EOT = valid accurate units/s` on public deterministic Tier-1 benchmarks; full tables in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md)):

| Engine | Model & Path | Base M5 (`24 GB`, `Apple10 NAX`) | M4 Pro (`48 GB`, `Apple9`) | vs. Stock `MLX` / `CoreAI` (`EOT`) |
|---|---|---|---|---|
| [`fast-bge-small`](../fast-bge-small) | `BAAI/bge-small-en-v1.5` (`hetero`) | **`13.061 EOT`** · **`200,646 tok/s`** (`0.79 ms` `B=1`, `1.54 GB`) | **`12.634 EOT`** · **`187,150 tok/s`** (`0.49 ms` `B=1`, `1.55 GB`) | **`11.26x`** (M5) / **`9.58x`** (M4 Pro) vs. `mlx-embeddings` |
| [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) | `google/embeddinggemma-2` (`hetero`) | **`4.707 EOT`** · **`48,803 tok/s`** (`2.58 ms` `B=1`, `3.43 GB`) | **`3.583 EOT`** · **`38,517 tok/s`** (`2.11 ms` `B=1`, `2.49 GB`) | **`4.33x` MLX / `6.24x` CoreAI** (M5) · **`4.80x` MLX** (M4 Pro) |
| [`fast-pplx`](../fast-pplx) | `pplx-embed-v2-context-0.6b` (`hetero2`) | **`1.724 EOT`** · **`16,796 tok/s`** (`5.57 ms` `B=1`, `2.34 GB`) | **`1.407 EOT`** · **`14,330 tok/s`** (`1.97 GB`) | **`4.41x`** (M5) / **`4.33x`** (M4 Pro) vs. `pplx-context-mlx` |
| [`fast-decision-models`](../fast-decision-models) | `Laya-Typed-421M` Pure-ANE (`[1585,0,0]`) | **`6.158 EOT`** · **`44.38 q/s`** (`1.00 GB`, `0` GPU ops) | **`5.827 EOT`** · **`41.91 q/s`** (`0.98 GB`, `0` GPU ops) | **`8.29x` CoreAI** (M5) · **`1.16x–1.92x` MLX** (`-0.23` MLX quality drop) |
| [`fast-decision-models`](../fast-decision-models) | `Kev-0.8B` (`W8 g64`, `B=8` + `PrefixTree`) | **`5.078 EOT`** · **`86.84 q/s`** (`1.77 GB`, `1.59x` M4 Pro `B=8`) | **`3.254 EOT`** · **`54.52 q/s`** (`1.84 GB`) | **`-42%–44%` RAM**; up to **`6.19x`** (M5) / **`9.89x`** (M4 Pro) w/ `PrefixTree` |
| [`fast-gemma4`](../fast-gemma4) | `google/gemma-4-E2B-it` (`kv_i8` + `llg`) | **`2.717 EOT`** (`Cat 4`) · **`10.46 valid docs/s`** (`1,479 tok/s`) | **`2.412 EOT`** (`Cat 4`) · **`8.78 valid docs/s`** (`ANE+GPU`) | **`1.81x`** (M5) / **`1.57x`** (M4 Pro) `Cat 4` `EOT`; **`5.18x`** valid `docs/s` |
| [`fast-gpt-oss`](../fast-gpt-oss) | `openai/gpt-oss-20b` (`MXFP4` MoE) | *Gated on `24 GB`* | **`0.318 EOT`** (`49/50` valid) · **`454.1 dec tok/s`** (`B=128`) | **`1.61x`** `Cat 4` `EOT` (`+53%` valid docs) · **`2.91x`** `mlx-lm` `B=8` |

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
