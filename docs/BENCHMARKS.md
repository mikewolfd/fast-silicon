# fast-silicon benchmarks

On the 24 GB M5, `fast-embeddinggemma-2` and OpenInfinity’s public export had nearly identical warm single-query latency. On the same 2,000 pretokenized documents, the project encoded 3.27 times as many tokens per second. The repaired Gemma extraction default scored 0.946 field F1 with every response schema-valid. The decision-model comparison now checks pretrained weights and uses one scorer per process. [Query and indexing runs](#warm-query-corpus-and-packing) · [Extraction results](#structured-generation) · [Decision models](#task-quality-and-elapsed-time)

The November 21 review and repair record covers model loading, token limits, quality thresholds, model-specific JSON formatting and corrected performance claims. Historical results retain their original settings and recorded verdicts. [Findings and repair status](benchmarks/review-20261121.md)

### Which implementation ran?

An engine is the software that runs a model. All four choices below run locally on Apple Silicon. A package can use more than one Apple framework.

| Label in this guide | What the benchmark runs |
| --- | --- |
| `fast-silicon` package (for example, `fast-embeddinggemma-2`) | This project’s code for batching inputs and choosing where to run the model. Depending on the package, it can use MLX on the GPU, Core ML on the Apple Neural Engine (ANE), or both. The saved Core ML comparison calls this result `native`. |
| Stock MLX | The public model and MLX library named in the row, such as `mlx-embeddings`, `mlx-vlm`, `mlx-lm` or `pplx-context-mlx`. |
| Public Core ML export | A publisher’s preconverted model, such as `OpenInfinity/embeddinggemma-2-coreml-int8`. The benchmark adapter prepares its inputs, calls Apple’s Core ML runtime and scores its outputs. |
| Apple CoreAI | A separate Apple runtime for `.aimodel` files, tested on the M5. `CoreAI-Opt` in the audit is a model conversion and optimization tool. |

Model format and input limits can differ within a model family. For example, FluidInference supplies a 16-bit floating-point (FP16) EmbeddingGemma export and OpenInfinity an 8-bit integer (INT8) export. Both accept up to 512 tokens per input. The project package accepts up to 2,048. [Model revisions and input audit](benchmarks/coreml-20261120.json)

## Table of Contents

1. [Hardware Testbeds & Empirical Roofline](#1-hardware-testbeds--empirical-roofline)
2. [How the benchmark scores work](#2-how-the-benchmark-scores-work)
3. [External Deterministic Objective Outcomes (`Tier-1`, Cat 1–4)](#3-external-deterministic-objective-outcomes-tier-1-cat-14)
4. [Unified Monorepo Throughput, Latency & Footprint](#4-unified-monorepo-throughput-latency--footprint)
5. [Local Training & Post-Training (`QLoRA`, `ModernBERT`, `GRPO`)](#5-local-training--post-training-qlora-modernbert-grpo)
6. [Speculative Decoding, `PrefixTree`, ANE Profiles & Parallel CoreML](#6-speculative-decoding-prefixtree-ane-profiles--parallel-coreml)
7. [CoreAI audit of 22 decisions](#7-coreai-audit-of-22-decisions)
8. [Reproducible CLI Invocations](#8-reproducible-cli-invocations)
9. [v2.2 Cross-Silicon Validation & Regression Gates (2026-11-21)](#9-v22-cross-silicon-validation--regression-gates-2026-11-21)
10. [v2.3 Cross-Silicon Accuracy, Precision & Contextual Validation (2026-11-22)](#10-v23-cross-silicon-accuracy-precision--contextual-validation-2026-11-22)
11. [`fast-qwen3` Hybrid GatedDeltaNet + OptiQ + LoRA Validation (2026-11-22)](#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)
12. [v2.5 Cross-Model R&D Winners & Negative Gates (2026-11-22)](#12-v25-cross-model-rd-winners--negative-gates-2026-11-22)
13. [Vision, OCR (`Category 5`) & Apple10 NAX Prefill Validation (2026-11-23)](#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23)
14. [v3.0 Vision Prefetch, NAX Meta-Only Packing, Multimodal MTP, and OCR Confidence (2026-11-23)](#14-v30-vision-prefetch-nax-meta-only-packing-multimodal-mtp-and-ocr-confidence-2026-11-23)
15. [Dual-Mac Full 4-Suite Verification and Post-Fix Reruns (2026-11-24)](#15-dual-mac-full-4-suite-verification-and-post-fix-reruns-2026-11-24)
16. [Validated Engine, Default & Laya FP16 LayerNorm Fixes (2026-11-24)](#16-validated-engine-default--laya-fp16-layernorm-fixes-2026-11-24)
17. [Wave-5 DRY Optimizations & Dual-Mac ABBA Gates (`fast-qwen3` & `fast-pplx`, 2026-11-24)](#17-wave-5-dry-optimizations--dual-mac-abba-gates-fast-qwen3--fast-pplx-2026-11-24)
18. [Quiescent Base M5 Full 4-Suite Headline Verification (`d9f65a3`, 2026-11-26)](#18-quiescent-base-m5-full-4-suite-headline-verification-d9f65a3-2026-11-26)
19. [Saved Benchmark Receipts Index](#19-saved-benchmark-receipts-index)

[Public Core ML results and model revisions](#public-core-ml-results)

---

## 1. Hardware Testbeds & Empirical Roofline

| Specification / Measured Ceiling | Apple M4 Pro (`Mac16,8`) | Base Apple M5 (`Mac17,2`) | Base M5 vs. M4 Pro |
| :--- | :---: | :---: | :---: |
| **Unified Memory (80% Wired Guard)** | `48 GB` (`38.4 GB` guard; `37.44 GiB` Metal WS) | `24 GB` (`19.2 GB` guard; `19.07 GB` Metal WS) | `0.50×` RAM |
| **CPU Topology** | `14` cores (`10P + 4E`, `2` SME2 units) | `10` cores (`4 Super + 6E`, SME2.1) | `0.71×` cores |
| **GPU Architecture & Cores** | `20c` `Apple9` (shader `simdgroup_matrix`) | `10c` `Apple10` + **per-core `NAX`** | `0.50×` cores (+ `NAX`) |
| **Apple Neural Engine (ANE)** | `16` cores (`h16g`) | `16` cores (`h17`, fixed from `2` on `27.2.1`) | `1.00×` cores |
| **OS & Runtime Stack** | `macOS 26.6` (`25G5057c`), `MLX 0.32.4`, `coremltools 9.0`, `coreai-opt 0.5.0` | `macOS 27.2.1` (`26B131`), `MLX 0.32.4`, `CoreAI 1.0.0b3`, Xcode CLT-only | OS `27.2.1` CoreAI |
| **Peak GPU Compute (`M=512..2048`)** | `7.61 TF` BF16 (`7.42 TF` probe; `6.40 TF` Q4) | **`12.25 TF` FP16** (`12.15` BF16, `13.48` Q8, **`14.49 TF` Q4**) | **`1.65×` FP16 / `2.26×` Q4** |
| **Empirical GPU DRAM Bandwidth** | **`210.2 GB/s`** (`273 GB/s` spec; `182.9 GB/s` copy) | `137.3 GB/s` (`153 GB/s` spec; `129.0 GB/s` copy) | `0.65×` (`1.53×` on M4 Pro) |
| **Empirical Roofline Ridge** | `35.3 FLOP/B` | `89.2 FLOP/B` | `2.53×` higher AI needed |
| **Single-Token Matmul (`M=1`)** | **`1.21 TF`** @ `0.028 ms` (`Q4`) / **`0.19 TF`** @ `0.173 ms` (`BF16`) | `0.46 TF` @ `0.072 ms` (`Q4`) / `0.12 TF` @ `0.283 ms` (`BF16`) | `2.63×` Q4 / `1.58×` BF16 faster on M4 Pro |
| **Connectivity** | Thunderbolt 5 (`TB5`) | Thunderbolt 4 (`TB4`) | — |

The base M5 sustained higher bulk encoder throughput on these runs. The M4 Pro had higher measured memory bandwidth and encoded one warm BGE query in 0.49 ms, compared with 0.79 ms on the M5. Its 48 GB also accommodated the 20B generator at batch 128. These comparisons describe two complete systems with different chips, memory capacities and operating systems. [Encoder workloads](#warm-query-corpus-and-packing) · [Batch capacity](#generation-capacity)

---

<a id="2-methodology-eot-two-gate-promotion--4-bucket-memory"></a>

## 2. How the benchmark scores work

### Effective Outcome Throughput (`EOT`)

Effective outcome throughput (EOT) is the sum of the scores earned on a workload divided by elapsed seconds. Each invalid result earns zero. A valid result can still earn partial credit when some fields or choices are wrong. `S_obj,e2e` in the tables is the mean score across **all** items, including invalid ones.

```text
EOT = (1 / T_e2e) * sum_i(valid_i * score_i)
    = (N_valid / T_e2e) * S_obj|valid
    = (N / T_e2e) * S_obj,e2e
```

The evaluator assigns `valid_i` a value of 0 or 1 and `score_i` a value between 0 and 1 using public datasets and deterministic scoring. `T_e2e` is elapsed time for the scored workload. On the historical minified M5 Gemma run, 300 tasks × 0.748675 mean score ÷ 82.6576 seconds = **2.717 score units/s**. The same run produced 299 valid responses, or **3.62 valid responses/s**. [Recorded task scores and elapsed time](benchmarks/document-workloads.json)

### Quality, memory and throughput checks

The outcome runner checks three conditions when deciding whether to adopt a candidate configuration over its baseline. [Outcome runner](../scripts/benchmark_outcomes.py)

1. **Gate 1 — Objective Quality Non-Inferiority:**
   - The `lossless` profile requires the lower end of the paired 95% confidence interval for candidate minus baseline to reach −0.003 or better by default. Set an application-specific margin with `--lossless-tolerance`. Missing paired evidence or an interval crossing the floor leaves quality unresolved. Repeat loops fail promotion.
   - `quantized` profile: `ΔS_obj,e2e >= -0.010` and bounded grammar mask fallback rate.
2. **Gate 2 — Unified-Memory Headroom (`<= 80%` UMA Ceiling):**
   - The memory probe records four readings. MLX’s active allocator peak (`mlx_peak_gb`) is reported separately. The process physical footprint (`phys_footprint_gb`) comes from the OS. Additional neural memory (`neural_nofootprint_gb`) comes from the in-process `task_vm_info` ledger. Model mappings (`model_mapped_gb`) track mapped files identified as model artifacts. `mapped_all_mb` includes other mappings too. [Memory probe](../packages/fast-shared/src/fast_shared/bench.py)
   - The limit is `combined_with_model_mapped_gb <= 0.80 * host_mem_gb` (38.4 GB on the M4 Pro, 19.2 GB on the M5). The device memory guard must also stay below 98% of the recommended working set. An older OS reading of 385.8 GB for CoreAI EmbeddingGemma on the 24 GB M5 was invalid. A separate five-text smoke run measured 3.04 GB using `RUSAGE_INFO_V6`. Full-run peak memory for that row remains unknown.
3. **Gate 3 — Outcome Throughput Ratio:**
   - `EOT_candidate / EOT_baseline >= 1.00` (`>= 1.03` when introducing custom kernel or stateful complexity).

### How to read a comparison

| Reader’s question | Measurement to use | What the timer covers |
| --- | --- | --- |
| How quickly can I finish this job, and how good is the result? | Public task scores, valid count, elapsed time and EOT | The retrieval timer covers encoding, ranking and scoring the locked dataset after loading and warmup. Decision timing covers scoring and evaluation after warmup. |
| How long does a single search request spend in the encoder? | Warm 24-token query latency | Ready token IDs through a returned vector. First load, tokenization, transport and client queueing are outside this timer. |
| How fast can I index a collection or keep a generator busy? | Corpus tokens/s or valid fixture documents/s | Encoders process a named corpus. Generation capacity tests repeat a small set of prompts at a stated batch size and output limit. |

The coordinator runs one model worker at a time. Runs on the same host with the same locked questions support comparisons of the observed scores and times. A separate experiment can alternate two configurations in A–B–B–A order. Its `comparisons` record contains that schedule and a verdict. The five public Core ML comparisons below use separately timed runs. A per-run `PASS` describes the checks recorded for that one run. [Outcome runner](../scripts/benchmark_outcomes.py)

Across 50 M4 Pro GPT-OSS tasks, the paired score difference was −0.0692 with a 95% interval of −0.2226 to +0.0841. The interval crosses the −0.003 floor. Quality is unresolved and two repeat loops fail promotion. Retrieval also requires full coverage of the intended queries. The original PPLX run served 150 of 325 and fails coverage on replay. The repaired run served 325/325. Its 4.41× observed score-throughput gain on M5 stands, while its quality interval [−0.00362, +0.00528] crosses the default floor. [Recorded inputs and current verdicts](benchmarks/document-workloads.json)

The generation servers send the completed response text in one content event. The reports leave time to first useful text and latency under concurrent HTTP requests unmeasured. [Server event construction](../packages/fast-shared/src/fast_shared/server.py)

### Deterministic Oracle First & PPI++ Subjective Judge Audit

1. **Deterministic checks remain the primary promotion oracle:** Retrieval (`Recall@K`, `span_nDCG@10`), extraction (`schema_valid`, `Field-F1`), OCR (`TEDS`, `NumF1`), and code/math suites are gated by executable objective verifiers in [`scripts/benchmark_outcomes.py`](../scripts/benchmark_outcomes.py) rather than LLM verdicts.
2. **Never rely on unanchored LLM voting or binary cross-provider consensus:** Studies show LLM judges carry systematic family/position biases that flip system rankings (`arXiv:2611.04090`), while naive binary AND-consensus (`Judge1 == Judge2`) cuts coverage without catching shared blind spots. Whenever subjective LLM judging (`GenRMScorer`) is used for comparisons or RLVR audit checks, anchor bulk calibrated scores with a small human-labeled slice (`n >= 30`) via [`compute_ppi_comparison()`](../packages/fast-shared/src/fast_shared/outcome_eval/metrics.py) (PPI++, `arXiv:2311.01453`, using `ci_method="bootstrap_t"` for `n <= 40`) to cancel bias, flag misaligned judges (`lambda_hat < 0.2`), and validate gains with valid 95% confidence intervals.

---

## 3. External Deterministic Objective Outcomes (`Tier-1`, Cat 1–4)

### Retrieval

`NanoBEIR-en` (`NanoSciFact`, `NanoArguAna`, `NanoHotpotQA`) + `LegalBench-RAG` (`cuad`, `contractnli`; `4K` contextual chunk windows for `fast-pplx` / `pplx-context-mlx`, `2K` for `EmbeddingGemma-2`).

| Workload (`N`) | Engine / Stock Harness | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **`BGE-small`** (`325`) | **`fast-bge-small` (`hetero`)** | `0.4737` | **`13.061`** (`11.8s`) | `0.99 / 1.13 / 0.41 /` **`1.54 GB`** | `0.4737` | **`12.634`** (`12.2s`) | `1.19 / 1.15 / 0.41 /` **`1.55 GB`** | **`11.26×`** / **`9.58×`** | **`1.03×`** |
| `BGE-small` Stock `MLX` | `mlx-community/bge-small-en-v1.5-8bit` | `0.4774` | `1.160` (`133.8s`) | `2.76 / 18.78 / 0.00 / 18.78 GB` | `0.4774` | `1.319` (`117.6s`) | `2.93 / 38.99 / 0.00 / 38.99 GB` | `1.00×` | `0.88×` |
| **`EmbeddingGemma-2`** (`500`) | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5376`** | **`4.707`** (`57.1s`) | `1.38 / 2.77 / 0.66 /` **`3.43 GB`** | **`0.5365`** | **`3.583`** (`74.9s`) | `1.46 / 2.38 / 0.11 /` **`2.49 GB`** | **`4.33×` MLX (`6.24×` CoreAI)** / **`4.80×`** | **`1.31×`** |
| `EmbGemma-2` Stock `MLX` | `mlx-community/embeddinggemma-2-8bit` (`mlx-vlm`) | `0.5364` | `1.086` (`246.9s`) | `2.58 / 19.41 / 0.00 / 19.41 GB` | `0.5353` | `0.746` (`358.6s`) | `2.84 / 39.03 / 0.00 / 39.03 GB` | `1.00×` | `1.46×` |
| `EmbGemma-2` Stock `CoreAI` | `coreai-emb2-bf16` (`text_encoder.aimodel`, `27.2+`) | `0.5360` | `0.754` (`355.5s`) | Full-run memory unavailable* | — *(26.6)* | — *(26.6)* | — | `0.69×` MLX | — |
| **`pplx-embed-0.6B`** (`325`) | **`fast-pplx` (`hetero2` ctx `4K`)** | **`0.7244`** | **`1.724`** (`136.5s`) | `2.48 / 1.86 / 0.23 /` **`2.34 GB`** | **`0.7241`** | **`1.407`** (`167.3s`) | `2.53 / 1.42 / 0.23 /` **`1.97 GB`** | **`4.41×`** / **`4.33×`** | **`1.23×`** |
| `pplx-embed` Stock `MLX` | `zainmerchan/pplx-embed-v2-context-0.6b-mlx` (`fp16`) | `0.7236` | `0.391` (`601.5s`) | `5.69 / 2.12 / 0.00 / 2.12 GB` | `0.7236` | `0.325` (`723.1s`) | `5.69 / 2.19 / 0.00 / 2.19 GB` | `1.00×` | `1.20×` |

*The CoreAI EmbeddingGemma smoke run measured 2.90 GB physical and 3.04 GB combined. The table gives full-run quality and throughput. For PPLX, combined memory includes 0.25–0.31 GB of mapped embedding weights.*

The M5 PPLX rerun split oversized legal documents into windows. That recovered all 175 legal queries and took the valid count from 150/325 to 325/325. For those 175 questions, encoding each passage with its neighbors raised its ranking score in the first ten results (span nDCG@10) from 0.359 to 0.703. Coverage of relevant characters in the first ten rose from 68.4% to 87.9%. Stock contextual PPLX also scored 0.703. The faster project run finished the full 325-query suite in 136.5 seconds versus 601.5 for stock. [Selected runs and source hashes](benchmarks/document-workloads.json)


A stock-only M4 Pro test isolated one reason for PPLX’s elapsed time. With 384 passages at the same batch size, sorting by token length reduced encoding time from **20.65 to 13.34 seconds** across the four samples. All 24 labeled queries returned the same top ten in the same order. On eight long legal windows, the shipped single-window loop took 4.39 s, while two windows per batch took 4.71 s in source order or 4.36 s in length order. Padding explains part of the passage-encoding gap. The full 325-query comparison includes other work. [Inputs, rankings, full-corpus padding and repeat timings](benchmarks/pplx-padding-20261121.json)

<a id="decision-models"></a>

### Decision scoring

All seven implementations answered the same 1,400 items, comprising 600 faithfulness judgments, 550 intents and 250 four-way preferences. The table gives accuracy, elapsed time and combined process memory. The stock MLX memory snapshots are lower bounds because they excluded clean pages mapped from the downloaded checkpoints. The lifetime process peaks for stock Laya were 2.58 GB on M5 and 2.63 GB on M4 Pro. For stock Kev the peaks were 3.01 and 3.39 GB. Each corrected stock run produced 1,400 usable answers. [Corrected MLX runs and comparison method](benchmarks/decisions-20261121.json) · [Project and public Core ML runs](benchmarks/coreml-20261120.json)

| Model and implementation | M5 accuracy | M5 seconds | M5 GB | M4 Pro accuracy | M4 Pro seconds | M4 Pro GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya · project, Neural Engine | 0.5050 | 114.8 | 1.00 | 0.5050 | 121.3 | 0.98 |
| Laya · stock MLX, corrected encoder | 0.5014 | **73.9** | 1.88 | 0.5021 | 130.7 | 2.00 |
| Laya · NMTZ Core ML | 0.4993 | 88.4 | 2.73 | 0.4993 | **92.0** | 2.94 |
| Laya · ahmedandaloes export in Apple CoreAI | 0.4993 | 940.7 | 0.33¹ | — | — | — |
| Kev · project, automatic prefix caching | 0.5250 | 167.2 | 1.78 | 0.5250 | **215.2** | 2.29 |
| Kev · stock MLX, one scorer | 0.5236 | **145.5** | 2.28 | 0.5229 | 225.3 | 2.10 |
| Kev · FluidInference Core ML | 0.5214 | 321.7 | 1.29 | 0.5207 | 435.4 | 9.04 |

¹ Apple CoreAI also reported 1.68 GB across all file mappings. The charged footprint and the sum of all mappings have different boundaries.

The former stock Laya baseline skipped all 170 pretrained encoder weights. Its historical 0.2757 and 0.2986 scores and associated speed and memory ratios are invalid. The repaired loader checks the complete encoder and matches reference probabilities and option order on short and 512-token inputs. Even with the repaired weights, stock Laya chose the preferred answer on only 79/250 four-way questions. Project Laya chose 73/250 and NMTZ chose 76/250. Random choice averages 25%. In a separate 48-document review pilot, adding Laya to rules surfaced seven of twelve known defects in the first ten reviews, up from four with rules alone. Automatic verification left held-out extraction quality unchanged and took longer. [Decision results](benchmarks/decisions-20261121.json) · [Extraction results and review budget](benchmarks/review-20261121.md#decision-scoring)

The old stock Kev harness loaded a second full model to format its input. The corrected run loads one scorer and matches all 7,650 option sequences and score positions to the original formatter. Stock Kev finished sooner on M5. Project Kev finished sooner on M4 Pro. The separate shared-passage timing measures what prefix reuse buys when several questions really share the same document. [Cache and shared-passage measurements](#kev-prefix-cache)

### Structured generation

**Gemma extraction with strict JSON.** The public 4-bit Gemma 4 E2B weights, documents, schemas and field scorer were the same for both engines. Both used the public checkpoint without a task-specific LoRA adapter. The November 21 project replay varied only the allowed output vocabulary and generated JSON separators at batch size 8 on the M5.

| Output vocabulary and JSON layout | Valid / 150 | Mean field F1 | English (58) | Other languages (92) |
| --- | ---: | ---: | ---: | ---: |
| Project, 150,786 tokens, minified | 149 | 0.695 | 0.786 | 0.637 |
| Project, 150,786 tokens, unrestricted | 143 | 0.802 | 0.946 | 0.712 |
| Project, full vocabulary, minified | 150 | 0.766 | 0.789 | 0.752 |
| Project, full vocabulary, unrestricted | 145 | 0.918 | 0.945 | 0.900 |
| **Project, full vocabulary, fixed space after each key’s colon** | **150** | **0.945** | **0.961** | **0.935** |
| Stock MLX, full vocabulary, unrestricted (archived batch 16) | 146 | 0.923 | 0.945 | 0.910 |

The minified matcher blocked Gemma’s usual token for a space and opening quote. The alternative token opened a string and placed a colon inside it. With the full vocabulary, 73 documents had a field equal to `": "`. Requiring a single space after the key delimiter admitted the usual token while keeping types, keys, enums and required fields constrained. The fixed layout produced zero such substitutions. On completed valid outputs, mean field F1 was 0.945 with the grammar and 0.949 without it. On 48 additional M4 Pro documents, both runs returned 48 valid results and scored 0.9258 and 0.9256. Their paired 95% interval for the difference was −0.0044 to +0.0048. [Corpus, token trace, method and paired results](benchmarks/gemma-grammar-20261121.json) · [Schema compiler and separators](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md#whitespace-handling)

The normal M5 benchmark submitted all 150 forms and then all 150 tool requests. The two current project runs used the same checkpoint, full vocabulary, prompts, batch size of eight, and 256 MiB MLX cache limit. The older stock run used batch 16. Field F1 gives zero credit to an invalid extraction. Tool accuracy checks both selected tools and argument values, including cases where the correct decision was to abstain.

| M5 configuration | Field F1 | Valid extractions / 150 | Correct tool decisions / 150 | Usable tool responses / 150 | `S_obj,e2e` | `EOT (u/s)` (vs. Stock) | Full-job seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Project, strict fixed separators** | **0.9459** | **150** | **123** | **150** | **0.8830** | **1.835** (**1.22×**) | **144.35** |
| Project, unconstrained | 0.9241 | 146 | 125 | 142 | 0.8787 | 1.281 (0.85×) | 205.81 |
| Stock MLX, unconstrained (archived) | 0.9234 | 146 | 120 | 140 | 0.8617 | 1.501 (1.00×) | 172.19 |

Both project runs finished without a repeat loop. Against unconstrained project decoding, strict decoding improved field F1 by 0.0218 and recovered twelve additional usable outputs across both workloads. It answered two fewer tool requests correctly. The paired 95% intervals were −0.0028 to +0.0464 for field F1 and −0.0623 to +0.0357 for tool accuracy. The strict run reported a 6.10 GB combined memory snapshot. The controlled extraction replay above split admissions into 19 smaller calls and scored 0.9451. [Runner settings, task splits and paired scores](benchmarks/gemma-grammar-20261121.json) · [Historical stock run](benchmarks/document-workloads.json)

Using the locked tool labels, a separate check constructed a correct answer for each of the 150 requests and verified that the actual tokenizer and strict matcher could consume it. Gemma can express every answer on this set. On a disjoint 50-request test, a more detailed prompt about function selection, optional arguments and abstention scored the same **41/50** as the original prompt. Both produced 50 usable outputs. One improved decision was offset by one regression. The original tool prompt remains the default. In a separate compiler diagnostic, 35/40 assorted JSONSchemaBench schemas compiled. Unsupported required schemas now stop the benchmark with an error. [Answer checks and compiler diagnostics](benchmarks/gemma-grammar-20261121.json)

For GPT-OSS the same 150 tool answers fit both its shipped minified grammar and a strict colon-space grammar using its actual 131,072-token output mapping. The existing M4 Pro model run got 9/25 tool decisions right, and stock got 14/25. Model accuracy under the alternate GPT-OSS layout remains unmeasured. The 48 GB M4 Pro had 16.4 GiB available against the rerun’s 19 GiB start requirement. [GPT-OSS task outcomes and CPU grammar check](benchmarks/document-workloads.json)

Category 3 evaluates reasoning on `IFBench`, `MMLU-Redux` and `MATH-500`. Category 4 evaluates extracted fields on `LiquidAI/structured-extraction` and tool calls on `BFCL v3`.

| Cat & Workload (`N`) | Engine / Stock Harness | Base M5 `S_obj,e2e` (Valid) | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Comb`) | M4 Pro `S_obj,e2e` (Valid) | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | :---: | ---: | :---: | :---: | ---: | :---: | :---: | :---: |
| **Cat 3: `Gemma-4 E2B`** (`435`) | **`fast-gemma4` (`gpu_fp16_kvi8`)** | `0.3517` (`280/435`) | **`0.843`** (`181.5s`) | `5.70 / 9.65 GB` | `0.3586` (`282/435`) | `0.892` (`174.8s`) | `5.84 / 8.74 GB` | **`1.03×`** / `0.94×` | `0.95×` |
| Cat 3: Stock `MLX` | `gemma-4-E2B-it-MLX-4bit` (`mlx-vlm`) | `0.3655` (`292/435`) | `0.818` (`194.4s`) | `5.54 / 5.41 GB` | `0.3655` (`292/435`) | **`0.951`** (`167.2s`) | `5.79 / 5.47 GB` | `1.00×` | `0.86×` |
| **Cat 3: `GPT-OSS-20B`** (`75`) | **`fast-gpt-oss` (`auto`)** *(48 GB only)* | *Gated (`24 GB`)* | *Gated* | — | `0.3067` (`40/75`) | **`0.174`** (`132.3s`) | `11.94 / 12.26 GB` | — / **`1.23×`** | *48 GB only* |
| Cat 3: Stock `MLX` | `gpt-oss-20b-MXFP4-Q4` (`mlx-lm`) | *Gated (`24 GB`)* | *Gated* | — | `0.3600` (`69/75`) | `0.141` (`191.0s`) | `12.97 / 12.07 GB` | `1.00×` | *48 GB only* |
| **Cat 3: `CoreAI Qwen3.5-4B`** (`48`) | CoreAI `4bit` / `fp4_fp8` | 0.3125 (29/48 valid) / 0.3125 (28/48 valid) | 0.041 / 0.048 (369 s / 311 s) | `0.00 / 1.65 -> 1.72 GB` | — *(26.6)* | — *(26.6)* | — | Both failed the repeat-loop check (1 / 2 loops) | — |
| **Cat 4: `Gemma-4 E2B`** (`300`, historical, pre-repair minified) | `fast-gemma4` (`llg_on` / `llg_off`, 150K vocab, minified) | `0.7487` (`299/300`) / `0.8241` (`284/300`) | `2.717` (`82.7s`) / `1.848` (`133.8s`) | `5.92 / 16.94 GB` | `0.7473` (`299/300`) / `0.8178` (`285/300`) | `2.412` (`92.9s`) / `1.702` (`144.1s`) | `5.93 / 18.37 GB` | `1.81×` / `1.57×` (`llg_on` `EOT`, historical) | `1.13×` |
| Cat 4: Stock `MLX` | `gemma-4-E2B-it-MLX-4bit` (`mlx-vlm`, unconstrained) | `0.8617` (`286/300`) | `1.501` (`172.2s`) | `6.11 / 5.58 GB` | `0.8658` (`287/300`) | `1.537` (`169.0s`) | `6.36 / 5.66 GB` | `1.00×` | `0.98×` |
| **Cat 4: `Qwen3.5-4B-OptiQ`** (`50`) | **`fast-qwen3` (`llg_on`, constrained)** | **`0.9024`** (**`50/50`**) | **`0.513`** (`88.0s`) | `3.38 / 5.26 GB` | **`0.9024`** (**`50/50`**) | **`0.396`** (`114.1s`) | `3.38 /` **`5.18 GB`** | **`1.53×`** / **`1.38×`** (`EOT`) | **`1.30×`** |
| Cat 4: Stock `MLX` | `Qwen3.5-4B-OptiQ-4bit` (`mlx-lm`, unconstrained) | `0.6619` (`37/50`) | `0.336` (`98.6s`) | `— / 3.79 GB` | `0.6225` (`35/50`) | `0.286` (`108.7s`) | `— / 8.02 GB` | `1.00×` | `1.17×` |
| **Cat 4: `Qwen3.5-9B-OptiQ`** (`50`) | **`fast-qwen3` (`--mtp-drafter`, `llg_on`)** | **`0.8857`** (**`50/50`**) | **`0.402`** (`110.2s`) | `7.47 / 8.39 GB` | **`0.8874`** (**`50/50`**) | **`0.277`** (`160.0s`) | `7.47 / 8.79 GB` | — | **`1.45×`** |
| **Cat 4: `GPT-OSS-20B`** (`50`) | **`fast-gpt-oss` (`llg_on`, constrained)** *(48 GB only)* | *Gated (`24 GB`)* | *Gated* | — | `0.5063` (**`49/50`**) | **`0.318`** (`79.7s`) | `12.59 /` **`12.77 GB`** | — / **`1.61×` `EOT` (`+53%` valid docs)** | *48 GB only* |
| Cat 4: Stock `MLX` | `gpt-oss-20b-MXFP4-Q4` (`mlx-lm`, unconstrained) | *Gated (`24 GB`)* | *Gated* | — | `0.5755` (`32/50`) | `0.197` (`146.2s`) | `14.48 / 14.30 GB` | `1.00×` | *48 GB only* |

On Category 4, the `fast-*` (`llg_on`) runs enforce the target JSON Schema during decoding via `llguidance`, whereas stock `mlx-lm` and `mlx-vlm` run unconstrained on the same checkpoints and receive `0` credit under `S_obj,e2e` whenever a response is invalid JSON (for paired constrained-vs-unconstrained comparisons within the same engine, see the Gemma table above and the `llg_on` / `llg_off` rows). The November 20 Gemma row in this table used the pre-repair English-oriented output vocabulary and minified key separators (`299/300` valid in 82.7 s, 1 repeat loop); the repaired strict colon-space default (`0.8830` `S_obj,e2e`, `150/150` valid on both tasks) is shown in the M5 table above. [Historical task scores and source hashes](benchmarks/document-workloads.json) · [Qwen3 Cat-4 receipts](benchmarks/fast-qwen3-20261122.json)

---

<a id="open-core-ml-results"></a>

## Public Core ML results

Five public Core ML exports ran the same locked questions as the corresponding `fast-silicon` packages on each Mac. Each export produced a valid output for every item. The comparison uses the November 20 `fast-decision-models` run with Kev’s automatic cache policy and the earlier full runs for the other project packages. The saved results include model revisions, per-task scores, memory and warm timings. [Machine-readable results](benchmarks/coreml-20261120.json)

### Task quality and elapsed time

For scores, times and memory, the **`fast-silicon` package comes first and the named public Core ML export comes second**. A rate ratio below 1 favors the public export. EOT measures score units per second. Memory is the measured combined footprint of each scored run. The 500 EmbeddingGemma items include 150 short retrieval queries and the same 175 legal questions evaluated with two passage sizes. The decision suite comprises 600 faithfulness examples, 550 intent questions and 250 best-of-four choices.

| Public Core ML export | Mac | Scored items | Mean task score | Score units/s | Elapsed seconds | Combined memory (GB) | Rate ratio (fast-silicon ÷ public export) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BGE-small · sidekickd FP16 | M4 Pro | 325 | 0.4737 / 0.4739 | 12.634 / 2.382 | 12.2 / 64.7 | 1.55 / 0.74 | 5.30× |
| BGE-small · sidekickd FP16 | M5 | 325 | 0.4737 / 0.4739 | 13.061 / 2.567 | 11.8 / 60.0 | 1.54 / 0.54 | 5.09× |
| EmbeddingGemma · FluidInference FP16 | M4 Pro | 500 | 0.5365 / 0.5186 | 3.583 / 1.637 | 74.9 / 158.4 | 2.49 / 1.53 | 2.19× |
| EmbeddingGemma · FluidInference FP16 | M5 | 500 | 0.5376 / 0.5186 | 4.707 / 1.666 | 57.1 / 155.7 | 3.43 / 1.38 | 2.83× |
| EmbeddingGemma · OpenInfinity INT8 | M4 Pro | 500 | 0.5365 / 0.5183 | 3.583 / 1.649 | 74.9 / 157.2 | 2.49 / 1.03 | 2.17× |
| EmbeddingGemma · OpenInfinity INT8 | M5 | 500 | 0.5376 / 0.5183 | 4.707 / 1.740 | 57.1 / 148.9 | 3.43 / 0.84 | 2.70× |
| Laya · NMTZ | M4 Pro | 1400 | 0.5050 / 0.4993 | 5.827 / 7.598 | 121.3 / 92.0 | 0.98 / 2.94 | 0.77× |
| Laya · NMTZ | M5 | 1400 | 0.5050 / 0.4993 | 6.158 / 7.909 | 114.8 / 88.4 | 1.00 / 2.73 | 0.78× |
| Kev · FluidInference FP16 | M4 Pro | 1400 | 0.5250 / 0.5207 | 3.415 / 1.674 | 215.2 / 435.4 | 2.29 / 9.04 | 2.04× |
| Kev · FluidInference FP16 | M5 | 1400 | 0.5250 / 0.5214 | 4.395 / 2.269 | 167.2 / 321.7 | 1.78 / 1.29 | 1.94× |

All 818 short legal passages fit the public EmbeddingGemma exports’ 512-token limit. In the long-passage test, 136 of 144 passages exceeded 512 tokens and one exceeded the project package’s 2,048-token limit. Out of 175,632 tokens, the public exports encoded 71,961 and `fast-embeddinggemma-2` encoded 175,585. The scorer gives credit for returning the **original passage**, so it can count a relevant clause beyond the prefix the model actually read.

On M5, long-passage ranking scores were 0.569 for the project, 0.516 for FluidInference and 0.518 for OpenInfinity. With short passages the scores were 0.272, 0.271 and 0.270. A matched M4 Pro rerun on the same 175 questions and 144 long passages separated the effect of the input limit from differences among implementations. All three tokenizers matched. The table shows ranking quality in the top ten and the share of labeled answer characters actually visible there. [Original lengths and scores](benchmarks/coreml-20261120.json) · [Matched passage and visible-answer results](benchmarks/embeddinggemma-legal-512-20261121.json)

| M4 Pro model and limit | Whole-passage ranking | Visible-text ranking | Gold characters recovered from visible text | Warm encoding time |
| --- | ---: | ---: | ---: | ---: |
| Project, 512 tokens | 0.5172 | 0.3014 | 33.6% | 2.39 s |
| FluidInference FP16, 512 | 0.5157 | 0.2978 | 33.6% | 4.99 s |
| OpenInfinity INT8, 512 | 0.5179 | 0.3016 | 33.6% | 4.84 s |
| Project, 2,048 | 0.5687 | 0.5687 | 90.3% | 14.58 s |

At 512 tokens, the input windows omitted every labeled answer character for 81 of the 175 questions. Each implementation nevertheless earned ordinary top-ten passage credit on 64 questions where its ten returned prefixes contained no labeled text. Extending the project’s input to 2,048 exposed labeled text for all 175. Its whole-passage ranking gain was +0.0514, with a paired 95% interval of +0.0253 to +0.0775. Whole-passage scoring answers whether a search result points to a useful passage. Visible-text scoring answers whether the encoder saw the supporting clause. [Paired comparisons and scoring method](benchmarks/embeddinggemma-legal-512-20261121.json)

The public BGE comparison uses sidekickd’s FP16 export. In five M4 Pro spot checks its lowest vector cosine similarity to `fast-bge-small` was 0.99996. A second export, `michaeljelly/bge-small-en-coreml-v1.5`, returned non-finite vectors on CPU/GPU and similarities of 0.52–0.62 on the Neural Engine. These results led to the selection of sidekickd for timing.

Laya measures the public typed option-scoring path, including host embedding lookup and probability conversion. Kev uses a shared-state fused function for inputs that fit and fixed row functions for larger inputs or option sets. The eight-question warm probe below reuses one short state. The public outcome suite exercises the row paths as well.

NMTZ’s public Laya export finished 1,400 decisions sooner on both Macs. On M5 it took 88.4 seconds and 2.73 GB, against 114.8 seconds and 1.00 GB for `fast-decision-models`. The project package scored 0.5050 and NMTZ scored 0.4993 on both hosts. The publisher’s separate action head, which can choose an action or abstain, falls outside this option-scoring test.

On the same 1,400 M5 items, the corrected single-model stock MLX Kev earned 5.039 score units/s and used 2.28 GB. The project earned 4.395 and used 1.78 GB. FluidInference earned 2.269 and used 1.29 GB. The project completed the M4 Pro run in 215.2 seconds, stock MLX in 225.3 and FluidInference in 435.4. An earlier project M5 run finished in 144.9 seconds with one more correct response than the 167.2-second run. Those separate runs leave the cause of the time difference open. [Corrected stock runs](benchmarks/decisions-20261121.json) · [Project and public Core ML runs](benchmarks/coreml-20261120.json)

### Warm query, corpus and packing

The project packages and public Core ML encoders received identical pseudo-random token IDs (seed 42). BGE processed 249,303 IDs across 1,000 documents and EmbeddingGemma processed 551,889 across 2,000. Each single query had 24 IDs. Timing starts with a loaded model and ends when it returns a vector. Tokenization, interprocess transfer and writes to a search index occur outside the timer. The public query figures are medians of 30 calls.

On M5, `fast-embeddinggemma-2` and OpenInfinity INT8 took 2.577 and 2.592 ms for one query. Across the corpus the project package processed 48,803 tokens/s against 14,945, a 3.27× ratio. In the separate M5 retrieval test, the project used 3.43 GB of combined memory and the INT8 export used 0.84 GB. For a worker that starts on demand, include model load time as well. [Memory readings](benchmarks/coreml-20261120.json) · [First-use measurements](#first-use-and-reproduction)

| Encoder | M4 Pro query (ms) | M4 Pro corpus (tokens/s) | M5 query (ms) | M5 corpus (tokens/s) |
| --- | --- | --- | --- | --- |
| `fast-bge-small` | 0.490 | 187,150 | 0.793 | 200,646 |
| BGE · sidekickd public Core ML FP16 | 1.696 | 31,608 | 1.608 | 34,318 |
| `fast-embeddinggemma-2` | 2.106 | 38,517 | 2.577 | 48,803 |
| EmbeddingGemma · FluidInference public Core ML FP16 | 2.696 | 14,251 | 4.334 | 14,187 |
| EmbeddingGemma · OpenInfinity public Core ML INT8 | 2.590 | 14,294 | 2.592 | 14,945 |
| EmbeddingGemma Apple CoreAI BF16 | — | — | 9.378 | 12,482 |

Both public EmbeddingGemma exports can combine up to eight short sequences in a 256-token call. Longer sequences use a single-input function. The mixed-length corpus was measured packed, single, single, packed. The cosine column reports the least similar pair of packed and single results over all 2,000 documents.

| Export | Mac | Packed tokens/s | Single tokens/s | Packed / single | Minimum cosine |
| --- | --- | --- | --- | --- | --- |
| EmbeddingGemma · FluidInference FP16 | M4 Pro | 14,251 | 13,702 | 1.040× | 0.999444 |
| EmbeddingGemma · FluidInference FP16 | M5 | 14,187 | 13,556 | 1.046× | 0.999444 |
| EmbeddingGemma · OpenInfinity INT8 | M4 Pro | 14,294 | 13,764 | 1.038× | 0.999725 |
| EmbeddingGemma · OpenInfinity INT8 | M5 | 14,945 | 14,438 | 1.035× | 0.999725 |

These figures use the **public Core ML exports**. Kev’s eight-question call reuses one 14-token passage. The 1,400-item mixed decision suite also exercises longer inputs and separate passages.

| Mac | Laya one short question, L192 (ms) | Laya one long question, L512 (ms) | Kev one question (q/s) | Kev eight on one state (q/s) |
| --- | --- | --- | --- | --- |
| M4 Pro | 16.40 | 51.14 | 16.48 | 130.69 |
| M5 | 16.17 | 48.34 | 20.72 | 187.30 |

### Public model memory and placement

Memory figures are decimal GB. Physical, neural memory outside the charged footprint, and model mappings come from the model process. Combined is the larger reported evaluator or model-process total. MLX allocator peak is a separate measure. On M5, Kev’s low charged footprint coexisted with a much larger resident size and more file mappings. The result file gives both and records where the OS limited access to separate Neural Engine services. A simultaneous whole-machine peak would require measuring the processes together.

| Export | Mac | MLX | Physical | Neural outside footprint | Model mappings | Combined | Preferred ops — ANE / GPU / CPU / unclassified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BGE-small · sidekickd FP16 | M4 Pro | 0.000 | 0.396 | 0.145 | 0.202 | 0.743 | 687 / 0 / 48 / 1,089 |
| BGE-small · sidekickd FP16 | M5 | 0.000 | 0.195 | 0.145 | 0.202 | 0.542 | 687 / 0 / 48 / 1,089 |
| EmbeddingGemma · FluidInference FP16 | M4 Pro | 0.000 | 0.340 | 0.945 | 0.249 | 1.534 | 3,862 / 0 / 0 / 45,223 |
| EmbeddingGemma · FluidInference FP16 | M5 | 0.000 | 0.167 | 0.945 | 0.268 | 1.381 | 3,862 / 0 / 0 / 45,223 |
| EmbeddingGemma · OpenInfinity INT8 | M4 Pro | 0.000 | 0.370 | 0.536 | 0.126 | 1.033 | 3,862 / 0 / 0 / 45,223 |
| EmbeddingGemma · OpenInfinity INT8 | M5 | 0.000 | 0.170 | 0.536 | 0.134 | 0.841 | 3,862 / 0 / 0 / 45,223 |
| Laya · NMTZ | M4 Pro | 0.000 | 0.439 | 2.496 | 0.000 | 2.936 | 31,782 / 0 / 0 / 17,823 |
| Laya · NMTZ | M5 | 0.000 | 0.229 | 2.496 | 0.000 | 2.725 | 31,782 / 0 / 0 / 17,823 |
| Kev · FluidInference FP16 | M4 Pro | 0.000 | 8.838 | 0.000 | 0.198 | 9.035 | 0 / 18,973 / 0 / 93,042 |
| Kev · FluidInference FP16 | M5 | 0.000 | 1.002 | 0.000 | 0.198 | 1.294 | 0 / 18,973 / 0 / 93,042 |

Core ML’s compute plan assigns preferred processors per operation. Counts span all recorded functions and length buckets. Unclassified operations have no recognized preferred device. For both public EmbeddingGemma exports on both hosts, the 3,862 Neural Engine preferences were in `embed_128`. Every `pack_256` operation was unclassified. The measured packing speedup establishes a timing gain. A runtime trace would be needed to assign that gain to a processor. Empty initial M5 plans were re-read from the same compiled bundles with a reader that understands the returned program structure.

### First use and reproduction

Core ML prepares source packages and loads or specializes the resulting bundles before the scored workload starts. The source compilation and warmup below come from the measured run, so a zero compile time can indicate a cache hit. In an earlier M4 Pro preparation, source compilation took 2.26 seconds for FluidInference EmbeddingGemma, 2.07 seconds for Laya and 4.45 seconds for Kev. BGE and OpenInfinity ship precompiled bundles.

| Export | M4 Pro compile / load / prewarm (s) | M5 compile / load / prewarm (s) |
| --- | --- | --- |
| BGE-small · sidekickd FP16 | 0.00 / 1.84 / 0.03 | 0.00 / 4.38 / 0.03 |
| EmbeddingGemma · FluidInference FP16 | 0.00 / 3.56 / 0.09 | 0.00 / 72.11 / 0.11 |
| EmbeddingGemma · OpenInfinity INT8 | 0.00 / 3.63 / 0.09 | 0.00 / 76.85 / 0.09 |
| Laya · NMTZ | 0.00 / 0.91 / 0.13 | 0.00 / 0.84 / 0.12 |
| Kev · FluidInference FP16 | 0.00 / 25.51 / 2.77 | 3.21 / 30.03 / 22.46 |

On M5, OpenInfinity’s EmbeddingGemma took 76.85 seconds to load before the scored run and 3.01 seconds in the later corpus probe. A first-response comparison for an on-demand worker still needs both implementations to start from the same cache state. [Recorded load and warm timings](benchmarks/coreml-20261120.json)

The M5 first prepared Laya during its warm probe, which measured 1.53 seconds for compilation and 88.96 seconds for first load. The subsequent successful outcome run reused those artifacts. Its initial outcome attempt failed because the environment lacked the host weight reader, now included in the build dependencies.

Install the workspace build dependencies with `uv sync --extra build --extra extract`. Use an empty, dedicated `HF_HOME` for these model snapshots so the runner selects exactly the measured revisions. The dataset setup uses its own cache. The model and file list includes the host embedding weights that the public Laya export requires.

```bash
export HF_HOME="$PWD/artifacts/coreml-models"
.venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

spec = json.loads(Path("docs/benchmarks/coreml-20261120.json").read_text())
for model in spec["models"].values():
    snapshot_download(**model)
PYTHON
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier1
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages coreml \
  --local-files-only --timeout-sec 1800 --json-out artifacts/coreml-outcomes.json
.venv/bin/python scripts/public_baselines.py --package coreml-embeddinggemma-2-fp16 \
  --timeout-sec 1800
```

For an individual outcome run, pass a single model key such as `coreml-embeddinggemma-2-int8` to `--packages`. The public probe accepts each of the five keys. Run it on a quiet host with exclusive use of its accelerators.

## 4. Unified Monorepo Throughput, Latency & Footprint

### Encoder and decision throughput

The BGE and EmbeddingGemma project probes start with token IDs. Historical stock MLX probes start with text and count the target lengths used to generate it. The table therefore gives the stock measurements in documents/s and the project measurements in actual tokens/s. For direct token-rate comparisons, the project and public Core ML runs used the same token IDs. [Matched encoder timings](#warm-query-corpus-and-packing) · [Throughput runner](../scripts/benchmark_monorepo.py) · [Stock MLX probes](../scripts/public_baselines.py)

| Workload | M4 Pro | Base M5 |
| --- | --- | --- |
| `fast-bge-small`, 1,000 pretokenized documents | 187,150 tokens/s, 2.09 GB footprint | 200,646 tokens/s, 1.85 GB footprint |
| BGE, stock `mlx-embeddings`, text inputs | 123.12 documents/s, 4.85 ms for one query | 106.96 documents/s, 4.42 ms for one query |
| `fast-embeddinggemma-2`, 2,000 pretokenized documents | 38,517 tokens/s, 3.26 GB footprint | 48,803 tokens/s, 2.76 GB footprint |
| EmbeddingGemma, stock `mlx-vlm`, text inputs | 46.05 documents/s, 8.39 ms for one query | 64.91 documents/s, 8.48 ms for one query |
| `fast-pplx`, 24 short and four long inputs | 12,624 tokens/s | 16,796 tokens/s (1.33× the M4 Pro rate) |
| Kev, 64 short fixtures in batches of eight, project / stock `mlx-lm` | 54.52 / 62.13 questions/s | 86.84 questions/s project (scored stock short-fixture probe not rerun on M5; 1,400-item decision suite: 4.395 vs 5.039 score units/s in [Decision scoring](#decision-models)) |
| Laya, `fast-decision-models` 6-bucket ANE / `laya-hetero` (`1x_s64..64x_mixed`) | `195.7` q/s (`1x_s64`), `62.2` / `51.4` q/s (`64x_mixed`) *(vs GPU `33.0` q/s, contended)* | `153.1` q/s (`1x_s64`), `63.5` / `110.6` q/s (`64x_mixed`) *(vs GPU `81.6` q/s, quiescent)* |

The PPLX package also reports fixed-length runs. For public Kev performance when eight questions share one short passage, use the shared-passage results. [PPLX workloads](../packages/fast-pplx/README.md) · [Public Kev warm timings](#warm-query-corpus-and-packing)

### Generation capacity

| Implementation and workload | Batch | M4 Pro | Base M5 |
| --- | --- | --- | --- |
| `fast-gemma4` GPU, eight repeated extraction fixtures | 128 | 6.79 and 6.87 valid documents/s | 10.46 valid documents/s |
| `fast-gemma4` Neural Engine + GPU, same fixtures | 2 × 128 | 8.78 and 5.91 valid documents/s | — |
| Gemma, stock `mlx-vlm`, schema validity checked after generation | 64 | 1.79 and 1.80 valid documents/s in the respective runs | 2.02 valid documents/s |
| `fast-gpt-oss`, eight extraction fixtures, schema, up to 512 output tokens | 128 | 454.1 decode tokens/s, 23.98 GB suite footprint | Excluded by the 24 GB memory policy |
| GPT-OSS, stock `mlx-lm`, repeated summary prompt, up to 128 output tokens | 8 | 156.1 decode tokens/s | Excluded by the same policy |

Both M4 Pro Gemma runs produced 256/256 schema-valid outputs on the Neural Engine and GPU. They took 29.16 and 43.29 seconds. Against the stock measurement from each respective run, the observed valid-document rate ratios were 4.90× and 3.29× across different batch sizes. For GPT-OSS, a direct engine-speed comparison still needs the same prompts, schema, batch and output cap. For quality on common public tasks, use the scored results. [Gemma runs and source hashes](benchmarks/document-workloads.json) · [GPT-OSS fixture workload](../packages/fast-gpt-oss/src/fast_gpt_oss/cli.py) · [Stock workload](../scripts/public_baselines.py) · [Public task results](#structured-generation)

---

## 5. Local Training & Post-Training (`QLoRA`, `ModernBERT`, `GRPO`)

These experiments time individual training or rollout steps. The encoder throughput experiment uses a randomly initialized model, generated token sequences and random labels. Its fastest row reuses cached frozen-layer outputs. The suite checks initial loss and gradients. Downstream quality after fine-tuning requires a separate evaluation. [Training harness](../scripts/benchmark_capabilities.py)

### 5.1 Dense Encoder Fine-Tuning (`ModernBERT-421M`, `encoder-finetune`, `freeze < 18`)

| Regime | Base M5 (`24 GB`, `10c NAX`) | Base M5 Peak Mem | Base M5 vs. Stock | M4 Pro (`48 GB`, `20c Apple9`) | M4 Pro Peak Mem | M4 Pro vs. Stock | M5 vs. M4 Pro |
| :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: |
| Stock Unpacked (`FP32` head) | `29.63 items/s` | `5.15 GB` | `1.00×` | `20.25 items/s` | `5.22 GB` | `1.00×` | `1.46×` |
| `BF16` + `@mx.compile` | `34.68 items/s` | `4.97 GB` | `1.17×` | `21.17 items/s` | `5.08 GB` | `1.05×` | `1.64×` |
| **`Packed-512` (FFD Block-Diag)** | **`42.26 items/s`** | **`4.90 GB`** | **`1.43×`** | **`26.63 items/s`** | **`5.06 GB`** | **`1.32×`** | **`1.59×`** |
| **`Packed-512` + Cached `L0..17`** | **`58.19 items/s`** | **`4.89 GB`** | **`1.96×`** | **`40.84 items/s`** | **`5.06 GB`** | **`2.02×`** | **`1.42×`** |

- **Parity:** `grad_cos = 0.999942` (`|Δloss| = 1.1e-4`) on Base M5; `grad_cos = 0.999946` (`|Δloss| = 2.2e-5`) on M4 Pro.
On generated inputs with random labels, packing reached 42.26 items/s on M5 versus 26.63 on M4 Pro. Reusing cached frozen-layer outputs raised those rates to 58.19 and 40.84 items/s. These runs measure step throughput and gradient agreement. Quality after training on labeled examples remains open.

### 5.2 Causal `QLoRA` + `FFD` Packing + Shared-System KV (`Gemma-4 E2B`, `qlora-packing`)

| Training configuration on M4 Pro | Fixture | Reported tokens/s | Peak Metal memory |
| --- | --- | ---: | ---: |
| Unpacked BF16 LoRA using the `mlx-lm` model and the benchmark’s training loop | 12 examples, one copy of the instructions | 579.1 | 16.09 GiB |
| 8-bit QLoRA, packed to 2,048 tokens | Same 12 examples | 545.8 | 25.06 GiB |
| 8-bit QLoRA with shared-prefix reuse | 8 examples, four copies of the instructions | 1,385.4 | 25.82 GiB |

The first two runs used 11,146 input tokens with 1,683 supervised answer tokens. The shared-prefix run counted 15,003 logical input tokens with 1,120 answer tokens, crediting each example with the 1,288-token prefix that the model reused. Replaying the M4 Pro 2,048-token batch preparation scheduled three optimizer updates for unpacked training, two for packed training and one for the shared-prefix arm. At 1,280 tokens the two matched arms scheduled three each. These counts come from the actual batch planner. The 2.39× ratio compares different workloads. Measuring the gain from prefix reuse requires timing the same examples and optimizer updates with reuse on and off. The initial gradient cosines were 0.999673 for packed versus unpacked BF16 and 0.994881 for packed 8-bit versus packed BF16. [Training results and fixture counts](benchmarks/capability-corrections-20261121.json)

- On the 24 GB M5, earlier 2,048- and 1,280-token QLoRA attempts exhausted Metal memory. Validating the sizing policy (`pack_tokens=1024, grad_accum=2, ckpt_tokens=512`, `19.2 GB` memory ceiling, `1.5 GB` cache limit) completed with zero OOMs (`pressure=['normal']`): unpacked `5.205 GB` (`4.848 GiB`) peak at `87.67` sup tok/s, packed `5.207 GB` (`4.849 GiB`) peak at `80.52` sup tok/s (`loss_diff=0.0, grad_cos=1.0, min_tensor_cos=1.0`), and shared-KV `7.257 GB` (`6.759 GiB`) peak at `81.63` sup tok/s. [Sizing policy](../packages/fast-shared/src/fast_shared/runtime_lessons.py) · [v2.2 validation JSON](benchmarks/v22-validation-20261121.json)
- **P2.8 ANE Frozen-Prefix Prefetch (`Gemma-4 E2B` `L0..11` ANE `[1650,0,0]` + `L12..34` GPU LoRA on M4 Pro):** Hides **`99.3%–99.8%`** of frozen prefix time (`84.9–374.0 ms -> 0.56–0.83 ms` exposed), yielding **`1.086×` (`B=1`) / `1.088×` (`B=2`)** end-to-end LoRA step speedup with `h_12` cosine `0.999124` and adapter gradient cosine `0.996968`.

### 5.3 Disaggregated `GPU Actor + ANE Critic` `GRPO` & Best-of-N (`G=4`, `grpo-ane-critic`)

The actor generates candidate answers and the critic scores them. The project configuration uses schema-constrained `fast-gemma4` on the GPU and runs the Laya critic concurrently on the Neural Engine. The stock configuration generates sequentially with `mlx-vlm` and scores with Kev on the GPU. On the M4 Pro, measured stage times give an **estimated 7.44× full-iteration speedup** at temperature 0.7 (81.48 versus 10.96 seconds). The estimate assumes the trailing critic work overlaps the GPU update. In the isolated overlap test, the same update took 2.65 seconds alone and 2.67 seconds with Laya scoring concurrently. The 7.44× estimate includes the changes to generation, scheduling and critic model. How closely Laya and Kev agree on rewards remains open. [Recorded stage times and estimates](benchmarks/document-workloads.json) · [Training harness](../scripts/benchmark_capabilities.py)

| Host | Mode | Rollout + Critic (`rollouts/s`) | Speedup vs. Stock | Exposed ANE Critic During GPU VJP | Peak Mem (GB) | Estimated full-iteration speedup |
| :--- | :--- | ---: | :---: | :---: | ---: | :---: |
| **Base M5 (`24 GB`)** | Stock `mlx-vlm` + `Kev-8bit` GPU | `0.50 rollouts/s` (`32.01s`) | `1.00×` | `1,080 ms` (serial GPU) | — | `1.00×` |
| **Base M5 (`24 GB`)** | **`fast-gemma4` (`T=0.7` GRPO) + `Laya` ANE** | **`2.61 rollouts/s`** | **`5.22×`** | **`0.03–0.35 ms`** (`99.96%+` hidden over `2,456 ms` VJP) | **`8.68 GB`** | — |
| **Base M5 (`24 GB`)** | **`fast-gemma4` (`T=0.0` BoN) + `Laya` ANE** | **`3.38 rollouts/s`** | **`6.77×`** | **`0.03–0.35 ms`** | **`8.68 GB`** | — |
| **M4 Pro (`48 GB`)** | Stock `mlx-vlm` + `Kev-8bit` GPU | `0.203 rollouts/s` (`78.83s`) | `1.00×` | `1,414 ms` (serial GPU) | `19.38 GB` | `1.00×` |
| **M4 Pro (`48 GB`)** | **`fast-gemma4` (`T=0.7` GRPO) + `Laya` ANE** | **`1.812 rollouts/s`** (`8.83s`) | **`8.93×`** | **`0.015 ms`** (`99.999%` hidden, `+0.84%` GPU overhead) | **`7.11 GB`** (`-63%`) | **`7.44×`** (`81.2%` vs `0%` schema valid) |
| **M4 Pro (`48 GB`)** | **`fast-gemma4` (`T=0.0` BoN) + `Laya` ANE** | **`3.149 rollouts/s`** (`5.08s`) | **`15.51×`** | **`0.015 ms`** | **`7.18 GB`** (`-63%`) | **`11.30×`** (`93.8%` vs `0%` schema valid) |

*(At `32` saturated rollouts `8×G=4` on M4 Pro, `fast-gemma4` reaches `1.981 rollouts/s` `T=0.7` and `4.179 rollouts/s` `T=0.0`. Across hosts, `10c` Base M5 generates constrained `GRPO` rollouts **`1.44×` faster** than `20c` M4 Pro.)*

---

## 6. Speculative Decoding, `PrefixTree`, ANE Profiles & Parallel CoreML

### 6.1 `Gemma-4` `MTP` Speculative Decoding + `DraftScan` (`~149.3 MiB` `4`-Layer Drafter, `6.1–10.3%` Draft Share)

| Workload | Ordinary decoding, M5 / M4 Pro | Fastest draft configuration, M5 / M4 Pro | Decode speedup, M5 / M4 Pro | Exact token match, M5 / M4 Pro |
| --- | --- | --- | --- | --- |
| **`B=1` free text** | 33.5 / 47.9 tok/s | 83.4 / 102.0 tok/s (`k4 kvi8`) | 2.49× / 2.13× | No / No |
| **`B=4` free text** | 134.9 / 186.4 tok/s | 212.1 / 250.9 tok/s (`k2`) | 1.57× / 1.35× | No / No |
| **`B=1` constrained JSON** | 32.8 / 46.8 tok/s | 97.1 / 121.4 tok/s (`k4`) | 2.96× / 2.60× | No / Yes |
| **`B=4` constrained JSON** | 112.0 / 149.8 tok/s | 221.0 / 272.6 tok/s (`k2`) | 1.97× / 1.82× | No / No |

The constrained four-document fixture completed at 1.62 versus 0.84 documents/s on M5 and 1.90 versus 1.09 on M4 Pro. The M4 Pro document rate rose 1.74×. The exact-match flag compares the full token sequence against ordinary decoding on the same host. These fixtures carry no answer-quality scores. [Draft results by host](benchmarks/capability-corrections-20261121.json)

- **Batch-Regime Saturation Guard (`B >= 5` -> `0.56×` at `B=8`):** At `B >= 5`, batched speculative verification saturates `Apple9`/`Apple10` matmul tiles (`B=8` regresses to **`0.56×`** vs plain autoregressive `ragged_b_slots`), and on `Apple10 NAX`, `M >= 8` native TensorOps already accelerate `B >= 2` by `1.5×–1.9×`. [`DynamicSiliconPolicy`](../packages/fast-shared/src/fast_shared/runtime_lessons.py) and `fast-gemma4` (`mtp_drafter="auto"`) therefore auto-gate `MTP` to **`B == 1` on `Apple10+ NAX`** and **`1 <= B <= 4` on `Apple9`**, paying **`0 MB` drafter RAM** on `B >= 8` batch jobs. For multi-candidate shared-prefix verification (`shared_prefix_tokens >= 32`, `num_candidates >= 2`), `select_speculative_policy` enables prefix-tree KV + tree-mask coalescing (up to **`2.59×`** at `N=4..8` candidates without duplicating prefix KV).

<a id="kev-prefix-cache"></a>

### 6.2 Kev prefix reuse on repeated passages

With eight actual RAGTruth responses about two source passages, the project's prefix cache cut the repeat call from **1,113 to 488 ms on M5** and **1,650 to 453 ms on M4 Pro**, using the explicitly uncached project scorer as the baseline. Stock MLX took 862 and 1,287 ms on the same repeats. Each scorer used identical token IDs, scored the question with the decision head, and returned every option probability and raw logit. [Inputs, scores and per-run measurements](benchmarks/decisions-20261121.json)

| Host | Scorer | First single / repeat (ms) | First eight / repeat (ms) | Peak process footprint (GB) |
| --- | --- | ---: | ---: | ---: |
| M5 | Project, cache disabled | 135.6 / 136.1 | 1,112.7 / 1,112.7 | 4.15 |
| M5 | Project, automatic cache | 203.6 / 62.4 | 728.3 / 487.6 | 1.83 |
| M5 | Stock MLX | 131.5 / 129.2 | 862.5 / 862.1 | 2.67 |
| M4 Pro | Project, cache disabled | 198.6 / 200.2 | 1,651.1 / 1,649.5 | 4.41 |
| M4 Pro | Project, automatic cache | 208.8 / 60.0 | 706.7 / 453.1 | 1.95 |
| M4 Pro | Stock MLX | 194.9 / 189.8 | 1,286.0 / 1,286.8 | 2.99 |

Each state contained 432–1,015 tokens and the question added 44. Each arm ran twice in its own fresh process in a symmetric six-run order. The table reports median elapsed time and the larger of the two peak process footprints. The first call used a warmed model and an empty passage cache. The cached scorer returned identical logits and probabilities on its immediate repeat. Comparing the cached and explicitly uncached eight-document scores, the largest probability changes were 0.00965 on M5 and 0.00745 on M4 Pro. Both selected the same answer on every item. [Cache probe](../scripts/_kev_benchmark.py) · [Complete results](benchmarks/decisions-20261121.json)

A separate M4 Pro stock probe completed **64 short decisions in 1.030 s** (62.1 questions/s) in batches of eight. The archived project scorer took 1.174 s (54.5 questions/s) on the same 64 decisions and batch size. Both runs repeated three CLI fixtures of 74, 82 and 85 tokens including the question. The earlier stock rates timed random-token backbone calls that omitted the decision head. An older project-only experiment measured 143.3 to 22.6 ms on M5 and 232.2 to 18.5 ms on M4 Pro for a repeated 1,024-token prefix. That fixture and the mixed 1,400-item workload answer different questions. [Complete short-fixture output](benchmarks/decisions-20261121.json) · [Earlier cache experiments](benchmarks/review-20261121.md#decision-scoring)

### 6.3 `fast-pplx` ANE Profiles (`fast` / `balanced` / `parity`, `packages/fast-pplx/scripts/benchmark.py ane-profiles`)

| Profile | `S=128` Base M5 (`tok/s`) | `S=128` M4 Pro (`tok/s`) | `S=256` Base M5 (`tok/s`) | `S=256` M4 Pro (`tok/s`) | `S=256` `f32 min_cos` vs. GPU | `S=256` `int8 MAE` vs. GPU | Takeaway |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`fast`** (`--ane-profile fast`) | **`13,542`** (`1.16×` stock) | **`13,464`** (`1.54×` stock) | `10,074` | **`10,311`** (`1.22×` stock) | `0.9909` *(corpus `0.9866`)* | `2.02` | Optional peak ANE speed (`1.00×`) |
| **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`, **default**) | `12,754` (`0.94×`) | `12,856` (`0.95×`) | `9,618` | `9,762` (`0.95×`) | **`0.9988`** *(corpus **`0.9964`**)* | **`0.79`** | Default: closes **`73%`** of ANE-vs-GPU cosine gap for `4.5%–5.8%` latency cost |
| **`parity`** | `12,679` (`0.94×`) | `12,496` (`0.93×`) | `9,650` | `9,519` (`0.92×`) | **`0.9989`** | **`0.72`** | Tightest ANE-to-GPU parity (`0.9987` int8 min cos) |
| Stock `pplx-context-mlx` (`fp16`) | `11,707` | `8,735` | `10,384` | `8,425` | `1.0000` (GPU) | — | Stock GPU reference (`zainmerchan` `fp16`) |

### 6.4 Parallel Multi-Bucket CoreML Compile (`parallel-coreml`, `spawn`-only + `skip_model_load=True`)

| Workload | Serial Legacy (`skip_model_load=False`) | Serial (`skip_model_load=True`, `jobs=1`) | Parallel (`jobs=4` / `jobs=2`) | Parallel vs. `jobs=1` | Total vs. Legacy | Output Parity |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **`fast-bge-small` (9 shapes, M4 Pro)** | 26.80 s | 9.18 s | 4.56 s (4 workers) | 2.01× | 5.88× | Archived build timing only |
| **`fast-bge-small` (9 shapes, M5)** | — | 5.12 s | 2.45 s (4 workers) | 2.09× | — | Archived embedding call failed |
| **`fast-gemma4` `Early14` (FP16 and W8, M4 Pro)** | 65.27 s | 34.94 s median | 26.84 s median (2 workers) | 1.30× | 2.43× | Both bundles and manifest present |

The archived BGE check printed a cosine of 1.0 after a failed embedding call and counted build directories. The repaired check ran 16 vectors through all nine shapes on each host. Every component matched between the final serial and parallel builds. Across two builds per arm, the medians were 9.37 and 4.38 s on M4 Pro, and 5.02 and 2.24 s on M5. For the older Gemma run, the first serial build took 43.51 s. The second runs took 26.37 s serial and 25.97 s parallel. [Build records and numerical checks](benchmarks/capability-corrections-20261121.json)

---

<a id="7-coreai-opt-track-a--os-coreai-track-b-promotion-audit-2222-gates"></a>

## 7. CoreAI audit of 22 decisions

The audit reproduced all 22 recorded decisions, including configurations that failed the quality or speed checks. Four decisions covered `CoreAI-Opt` on the M4 Pro and 18 covered Apple CoreAI on the M5. The table identifies which configurations passed, failed or still need a full test. The audit also recorded zero `fast-ir-guard` errors and confirmed the FP16 epsilon floor (`2^-14`) on all 33 Laya normalization layers.

### 7.1 Cross-Mac Promotion Gate Matrix

This archived matrix preserves the 22 decisions made under the earlier quality rule. The reported gate verdicts belong to those runs. The current comparison uses a fixed quality tolerance, a paired interval and an explicit coverage check. [Current decision rule](#quality-memory-and-throughput-checks)

| Track / Host | Candidate vs. Baseline | Profile(s) | `S_e2e` (`ΔS` / `SE`) | G1 | 4-Bucket Mem / Cap | G2 | `EOT` (`cand / base = ratio`) | G3 | Placement & Key Receipt | Verdict |
| :--- | :--- | :---: | :---: | :-: | :---: | :-: | :---: | :-: | :--- | :---: |
| **Track A** M4 Pro `26.6` | `fast-pplx` `CoreAI-Opt` CoreML (`fp16..w4_pb`, `s64`) vs. Prod MIL `W8A16` | pre-ABBA | `st2 real_cos=0.9994`, `st3=0.9989` (`w8`) | — | — | — | `9.29 / 2.24 ms` (`4.1×` slower) | **F** | `0/349` st2, `0/307` st3 ANE (`ANECCompile()` fail -> 100% CPU; prod `375/375` ANE) | **FAIL** *(disqualified)* |
| **Track A** M4 Pro `26.6` | `Laya CoreAI-Opt w8` (`[1440,0,0]` ANE) vs. Prod `W8A16` (`[1585,0,0]`, `N=1400`) | `lossless` & `quantized` (`2` gates) | `0.5014` vs `0.5050` (`-0.0036`, `SE=0.0065`) | **T** | `1.002 / 38.4 GB` | **T** | `5.756 / 5.826 = 0.9879` (`122.0s` vs `121.4s`) | **F** | `100%` Pure-ANE (`353.8 MB`); **`-14.3%` `ECE`** (`0.0970` vs `0.1131`), `Brier 0.2174` vs `0.2214`, `RAG NLL 0.6245` vs `0.6341`, `CLINC NLL 3.4689` vs `3.5704` | **FAIL** *(Gate 3 `-1.21%`)* |
| **Track A** M4 Pro `26.6` | `Laya CoreAI-Opt w4` (`[1440,0,0]` ANE) vs. Prod `W8A16` (`N=1400`) | `lossless` & `quantized` (`2` gates) | `0.4529` vs `0.5050` (`-0.0521`, `SE=0.0107`) | **F** | `0.825 / 38.4 GB` | **T** | `5.282 / 5.830 = 0.9061` (`120.0s` vs `121.3s`) | **F** | `100%` Pure-ANE (`177.7 MB`), `AUROC 0.6433` vs `0.6812` | **FAIL** *(G1 & G3)* |
| **Track A** M4 Pro `26.6` | `Laya CoreAI-Opt w4_per_block` (`blk=32`, CoreML `9.0`) | pre-ABBA | `fake_cos=+0.9627` vs **`live_synth=-0.9133` / `real=-0.1985`** | **F** | — | — | `65.39 / 24.28 ms` (`2.69×` slower) | **F** | `[827,0,613]`, `0.0%` `ALL` ANE (`121/121` quantized `ios18.conv` unsupported + **corrupted inverted logits**) | **FAIL** *(guard-blocked)* |
| **Track B** Base M5 `27.2.1` | `fast-pplx --backend coreai` (`w8`, FP32 compute) vs. live `hetero2` / `gpu` (`N=24`) | `lossless` & `quantized` (`6` gates) | `0.7033` vs `0.7156` live (`-0.0123`*, `SE=0.0139`) | **T/F*** | `2.593 / 19.2 GB` (`mapped_all=4.00 GB`) | **T** | `0.660 / 3.585 = 0.1842` (`0.281×` GPU) | **F** | `336 GPU / 0 ANE` (`MPSGraph`, `130.6 ms` warm; `S=128`: `32.3 ms`, `3,965 tok/s`); `st0 w8_f16` hits `3.00 ms` | **FAIL** *(G3; G1*)* |
| **Track B** Base M5 `27.2.1` | `fast-pplx --backend coreai` (`w4_per_block`, FP32) vs. live `hetero2` / `gpu` (`N=24`) | `lossless` & `quantized` (`6` gates) | `0.7003` vs `0.7156` (`-0.0153`, `SE=0.0140`) | **F** | `3.071 / 19.2 GB` | **T** | `0.643 / 3.585 = 0.1792` (`0.273×` GPU) | **F** | `336 GPU / 0 ANE`, worse than `w8` on speed (`138.5 ms`), RAM, & cosine (`0.981` vs `0.9997`) | **FAIL** *(G1 & G3)* |
| **Track B** Base M5 `27.2.1` | `LayaCoreAiModel w8` (FP32 compute) vs. stock `coreai-laya` (`N=42`) | `lossless` & `quantized` (`2` gates) | `0.3571` vs `0.3571` (`+0.0000`, `SE=0.0341`) | **T** | `0.569 / 19.2 GB` (`mapped_all=1.49 GB`) | **T** | **`1.053 / 0.486 = 2.1691`** (`14.2s` vs `30.9s`) | **T** | `1397 GPU / 0 ANE`, `355 MB` vs `1,608 MB` (`4.54×` smaller), `logit_cos=0.999839` (`w4_pb` also `PASS` `2.075×` vs stock) | **PASS** *(vs stock CoreAI)* |
| **Track B** Base M5 `27.2.1` | `LayaCoreAiModel w8` & `w4_pb` (FP32) vs. the project’s `LayaAneBackend` (`N=42`) | `quantized` (`2` gates + `2` `w4_pb` stock) | `0.3571` vs `0.3571` (`+0.0000`) | **T** | `0.569 / 19.2 GB` | **T** | `1.053 / 3.696 = 0.2849` (`0.273×` `w4_pb`) | **F** | FP32 compute forces GPU fallback (`94.2–102.8 ms` vs `22.64 ms` project Laya scorer) | **FAIL** *(vs. the project Laya scorer)* |
| **Track B** Base M5 `27.2.1` | **`LayaCoreAiModel w8_f16` + `neural_engine`** (`S=256` ablation) | ablation | `logit_cos = 0.999831`, `argmax = 1.0`, `0` NaN | — | **`0.444 / 19.2 GB`** (`phys=62.3 MB`, `neural=360.8 MB`) | **T** | **`23.00 ms` iso** (`43.5 q/s`; `27.22 ms` shared) vs `22.64 ms` project Laya scorer & `354.1 ms` stock | — | **1,652 ANE ops** in the FP16 export, `4.09×` faster than FP32 `.aimodel` and `15.4×` faster than stock `CoreAI` | **PROMISING** *(needs `S=512` full gate)* |

*\*Track B `fast-pplx` `ΔS = -0.0123` vs live `hetero2` stemmed from `11` LegalBench chunks `>512` tokens truncated to static bucket `S=512` (`NanoSciFact` `0.9156` beat `hetero2` `0.9122`, FP32 cosine `0.9997`), not quantization error.*

### 7.2 Root-Cause Findings & Production Guardrails

1. **`FP32 -> MPSGraph` vs. `FP16 -> 1,652 ANE Ops` (`.aimodel` on `macOS 27.2.1`):**
   - Exporting PyTorch modules with default `float32` compute I/O and buffers causes `ANECCompile` Code=22 (`MLIR MPS to ANEC conversion failed`) under `prefer_neural_engine`, forcing `100%` GPU `MPSGraph` fallback (`Laya w8`: `94.18–102.75 ms`, `0` ANE ops; `fast-pplx w8`: `130.62 ms`, `0` ANE ops).
   - Casting floating weights, buffers, and inputs to **`FP16`** unlocks **`1,652` ANE ops** on `Laya w8_f16` (`23.00 ms` isolated `s=256`, **`4.09×` faster**, **`62.3 MB` `phys_footprint`**, `360.83 MB` `neural_footprint`), and on `fast-pplx`, rewriting `F.softplus` to `F.relu(x) + log1p(exp(-|x|))` alongside `FP16` drops `st0 s128` to `2.997 ms`.
   - `packages/fast-shared/src/fast_shared/coreai_pipeline.py` now auto-casts `.aimodel` dense/int exports to `FP16`, and `scripts/_outcome_public_backends.py` loads assets via `coreai.authoring.AIModelAsset.load(...)` (not `coreai.runtime`).
2. **`CoreAI-Opt` `w4_per_block` CoreML Corruption Guard (`macOS 26.6`, `CoreML 9.0`):**
   - `w4_per_block` (`block_size=32, axis=1`) passed PyTorch fake-quant calibration (`+0.9627` cosine on `Laya`) yet produced inverted logits (`-0.9133` synth / `-0.1985` real cosine) and `121/121` unsupported `ios18.conv` CPU fallbacks (`65.39 ms`) when compiled to `.mlmodelc`. `build_coreai_opt_config()` now rejects `w4_per_block` on `ExportBackend.CoreML` by default unless `allow_experimental_coreml_w4_block=True`.
3. **`Qwen3.5` `fp4_fp8` `GatedDeltaNet` Analysis (Base M5 `27.2.1`):**
   - **Swift `llm-runner` & Parity (`512` prompt / `128` gen tokens, `25`-step NLL):**

| Model (`CoreAI .aimodel`, Base M5 `27.2.1`) | `4bit` (`BF16` act) Gen `tok/s` / Peak RAM | `fp4_fp8` (`NAX`) Gen `tok/s` / Peak RAM | `4bit` Parity (`PPL` / `NaN`) | `fp4_fp8` Parity (`PPL` / `NaN`) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.5-0.8B`** | `195.7 tok/s` / `1.03 GB` | `220.5 tok/s` / `0.85 GB` *(invalid)* | `443.3` / `0` NaN | **`100% NaN`** (`6,208,000` NaN; sampler `-1` after 4 tok) | **Broken under `fp4_fp8` — keep `4bit`** |
| **`Qwen3.5-2B`** | `97.3 tok/s` / `2.00 GB` | `109.9 tok/s` / `1.54 GB` *(invalid)* | `117.1` / `0` NaN | **`100% NaN`** (`6,208,000` NaN; sampler `-1` after 4 tok) | **Broken under `fp4_fp8` — keep `4bit`** |
| **`Qwen3.5-4B`** | 43.1 tok/s / 3.71 GB | 49.4 tok/s / 2.91 GB | 118.9 / 0 NaN | 93.4 / 0 NaN | Both had finite values in this probe and repeated output on the scored task. |

On 48 reasoning and instruction tasks, 4-bit completed 29 valid responses in 369.2 s with one repeat loop. FP4/FP8 completed 28 in 310.7 s with two loops. Both scored 0.3125 and failed the benchmark’s zero-loop requirement. [Qwen task results](benchmarks/capability-corrections-20261121.json)

   - **Root Cause:** Generic `fp4_weights_fp8_activations_fp8_kv_cache` quantizes `18` `GatedDeltaNet` recurrence/decay projections (`in_proj_ba`, `in_proj_qkvz`) to per-tensor `FP8 (e4m3fn)` activations via `15`-sample `MinMax` calibration, overflowing small `<= 2B` hybrid models. Keep `BF16` activations (`4bit` / `qwen3_5_cfg.yaml`) on `Qwen3.5-0.8B` and `Qwen3.5-2B`.
4. **Locked `DynamicSiliconPolicy` Backend Rule:** Default `preferred_ane_backend = "coreml_mil"` (`100%` Pure-ANE 4D `conv1x1`) on every macOS release; switch to `"coreai"` only when `macOS >= 27.2.1` (active CoreAI delegate) **and** `(model_name, batch_size)` explicitly clears all three promotion gates.

---

## 8. Reproducible CLI Invocations

All commands run from the `fast-silicon` repository root using `.venv/bin/python` (all GPU/ANE runs self-serialize via `/tmp/m4_gpu_validation.lock`).

### 1. External Objective Outcomes (`scripts/benchmark_outcomes.py`)

```bash
# Stage & verify pinned public dataset shards (0 GPU/ANE work)
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier1

# Dry-run preflight across all packages
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages all --dry-run

# Run Tier-1 objective outcomes (add --skip-gpt-oss on <= 24 GB Macs)
.venv/bin/python scripts/benchmark_outcomes.py run \
  --tier tier1 --packages all --json-out artifacts/outcomes_tier1.json

# Gemma structured tasks with the full checkpoint vocabulary and strict fixed separators
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages fast-gemma4 \
  --extra-args="--categories 4 --batch-size 8 --mlx-cache-limit-mb 256"

# Same tasks with the historical English-oriented vocabulary and byte-compact JSON
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages fast-gemma4 \
  --extra-args="--categories 4 --batch-size 8 --mlx-cache-limit-mb 256 --gemma-vocab-out 150786 --gemma-json-layout compact"

# Run the named public baselines on the same tasks as the project packages
.venv/bin/python scripts/benchmark_outcomes.py run \
  --tier tier1 --packages all --include-public-baselines \
  --public-python /path/to/stock-mlx-venv/bin/python \
  --coreai-python /path/to/coreai-venv/bin/python \
  --pplx-mlx-path /path/to/pplx-context-mlx \
  --json-out artifacts/outcomes_tier1_with_baselines.json

# Check quality, memory and throughput between two saved reports
.venv/bin/python scripts/benchmark_outcomes.py compare \
  --gate-profile lossless --min-eot-ratio 1.00 --markdown \
  artifacts/outcomes_base.json artifacts/outcomes_cand.json

# Stage Category 5 public OCR datasets and run cross-family visual OCR benchmarks
.venv/bin/python scripts/benchmark_outcomes.py setup-data \
  --tier tier1 --datasets olmocr_bench,fintabnet_otsl,omnidocbench_en,cord_v2,vrdu_ad_buy,vrdu_registration
.venv/bin/python scripts/benchmark_outcomes.py stage-gov \
  --branches legislative,executive,judicial --per-source 4
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --smoke --local-files-only \
  --packages ocr-families --public-python /path/to/stock-mlx-venv/bin/python \
  --extra-args "--ocr-backend ovis --ocr-render-only" --json-out artifacts/outcomes_cat5_smoke.json
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier2 --smoke --local-files-only \
  --packages fast-ocr,public-pypdfium2 \
  --extra-args "--ocr-backend ovis --ocr-render-only --ocr-suites gov"
```

See also the [Category 5 & Gov-Doc OCR benchmark guide](../packages/fast-ocr/docs/OCR_BENCHMARKS_AND_BASELINES.md).

### 2. Post-Training, Speculative, Prefix-Tree & Parallel Compile (`scripts/benchmark_capabilities.py`)

```bash
# Dry-run all 6 isolated ABBA capability suites
.venv/bin/python scripts/benchmark_capabilities.py --suites all --dry-run

# Run full capability suite (qlora-packing, encoder-finetune, grpo-ane-critic,
# mtp-speculative, prefix-tree, parallel-coreml) with stock MLX baselines
.venv/bin/python scripts/benchmark_capabilities.py \
  --suites all \
  --public-python /path/to/stock-mlx-venv/bin/python \
  --json-out artifacts/capabilities_abba.json \
  --markdown-out artifacts/capabilities_abba.md

# Single-suite example (24 GB safe)
.venv/bin/python scripts/benchmark_capabilities.py \
  --suites encoder-finetune,grpo-ane-critic,mtp-speculative,prefix-tree,parallel-coreml \
  --public-python /path/to/stock-mlx-venv/bin/python \
  --json-out artifacts/capabilities_24gb.json
```

### 3. Unified Monorepo Speed & Footprint Sweep (`scripts/benchmark_monorepo.py`)

```bash
# Safe dry-run (0 GPU/ANE work)
.venv/bin/python scripts/benchmark_monorepo.py --dry-run

# Full serialized sweep with internal & public baselines (defaults --gemma4-ane-variant fp16)
.venv/bin/python scripts/benchmark_monorepo.py \
  --include-verify \
  --public-python /path/to/stock-mlx-venv/bin/python \
  --pplx-mlx-path /path/to/pplx-context-mlx \
  --json-out artifacts/monorepo_bench.json \
  --markdown-out artifacts/monorepo_bench.md
```

### 4. `fast-pplx` Deep Speed, Parity, `32K` Context & ANE Profiles (`packages/fast-pplx/scripts/benchmark.py`)

```bash
# Dry-run full harness
.venv/bin/python packages/fast-pplx/scripts/benchmark.py all --dry-run

# Multi-mode speed comparison (including stock pplx-context-mlx & PyTorch MPS/CPU)
.venv/bin/python packages/fast-pplx/scripts/benchmark.py speed \
  --modes gpu-no-mmap,gpu-bf16,gpu,ane,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu \
  --pplx-mlx-path /path/to/pplx-context-mlx \
  --torch-python /path/to/stock-mlx-venv/bin/python \
  --json-out artifacts/pplx_speed.json

# Compile and ABBA-benchmark fast / balanced / parity ANE profiles in isolated temp homes
.venv/bin/python packages/fast-pplx/scripts/benchmark.py ane-profiles \
  --profiles fast,balanced,parity --seq-lens 128,256 --repeats 2 \
  --pplx-mlx-path /path/to/pplx-context-mlx \
  --torch-python /path/to/stock-mlx-venv/bin/python \
  --json-out artifacts/pplx_ane_profiles.json

# 16-case numerical parity, 32K context-length scaling, contextual window lift & bit-identity
.venv/bin/python packages/fast-pplx/scripts/benchmark.py parity --modes gpu,ane,hetero2
.venv/bin/python packages/fast-pplx/scripts/benchmark.py context-limit --lengths 512,1024,2048,4096,8192,16384,32768
.venv/bin/python packages/fast-pplx/scripts/benchmark.py context-windows --context-budgets 512,1024,2048,4096
.venv/bin/python packages/fast-pplx/scripts/benchmark.py bit-identity dump --out-dir artifacts/bit_base
```

### 5. `fast-qwen3` & `v2.5` Long-Context / Multi-Vector Benchmarks

```bash
# Interleaved ABBA speed & footprint sweep on Qwen3.5 presets
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16,9b-optiq --rounds 2

# Qwen3.5 Category-4 outcomes with QuaRot 4-bit KV or 9B native MTP drafter
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages fast-qwen3 \
  --max-items 25 --extra-args="--categories 4 --kv-mode q4_quarot"
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages fast-qwen3 \
  --max-items 25 --extra-args="--categories 4 --model-dir 9b-optiq --mtp-drafter"

# fast-pplx Tier-2 LongEmbed 4K-32K with multi-vector contextual MaxSim (1024d INT8)
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier2 --packages fast-pplx \
  --extra-args="--mode gpu --pplx-doc-pooling maxsim_1024"
```

---

## 9. v2.2 Cross-Silicon Validation & Regression Gates (2026-11-21)

Machine-readable record: [`docs/benchmarks/v22-validation-20261121.json`](benchmarks/v22-validation-20261121.json).

### Track 1: `fast-shared` (`LiveMemoryGovernor`, `24 GB` QLoRA, Rolling Per-Doc Grammar, Lookahead)

- **`LiveMemoryGovernor` admission overhead:** `1.924 us`/wave (`12.26 us` raw) on M4 Pro; `1.458 us`/wave (`9.15 us` raw) on Base M5.
- **`24 GB` Base M5 QLoRA packing (`pack_tokens=1024, grad_accum=2, ckpt_tokens=512`, `19.2 GB` cap):** Unpacked peaked at `5.205 GB` (`4.848 GiB`) @ `87.67` sup tok/s; packed peaked at `5.207 GB` (`4.849 GiB`) @ `80.52` sup tok/s (`loss_diff=0.0, grad_cos=1.0, min_tensor_cos=1.0`); shared-KV peaked at `7.257 GB` (`6.759 GiB`) @ `81.63` sup tok/s (`pressure=['normal']`).
- **Rolling per-doc grammar compilation & `whitespace_flexible` guard:** `Suite 2A` (`96` docs) sped up `50.468s -> 47.787s` (`1.058x`, `98.96% -> 100.0%` valid) on M4 Pro and `42.761s -> 40.200s` (`1.062x`, `100.0%` valid) on Base M5. `Suite 2B` (`96` LiquidAI docs, `38` schemas) sped up **`1.925x`** on M4 Pro (`90.026s -> 46.953s`, `1.06 -> 2.04 docs/s`, `0.9271 -> 0.9248` F1) and **`2.486x`** on Base M5 (`86.584s -> 34.790s`, `1.11 -> 2.76 docs/s`, `0.9289 -> 0.9263` F1). Forcing `whitespace_flexible=False` crashed F1 to `0.7647` (M4 Pro) and `0.7646` (Base M5), confirming the `whitespace_flexible=None` default guard.
- **Embedding prefetch (`lookahead_batches=4`, `N=1024`) & no-cache guard:** `fast-bge-small` sped up **`1.331x`** on M4 Pro (`280.4 -> 373.2 docs/s`, `min_cos=0.9999754`) and **`1.178x`** on Base M5 (`359.1 -> 423.0 docs/s`, `min_cos=0.9999746`); `fast-pplx` GPU sped up `1.022x` on both hosts (`24.306s -> 23.793s` M4, `min_cos >= 0.9999965`). `use_nocache=True` regressed M4 Pro `embeddinggemma` (`23.793s -> 27.122s`), so `use_nocache=False` remains default.

### Track 2: `fast-gemma4` & `fast-gpt-oss` (`MoE FP32`, `MPP`, Constrained `MTP`, Rejections)

| Gate / Mechanism | M4 Pro (`48 GB`) | Base M5 (`24 GB`, `Apple10 NAX`) | Promotion Verdict |
|---|---|---|---|
| **MoE FP32 router + combine (`gpt-oss-20b-MXFP4-Q8`, `3072` rows)** | Router set/ord mismatch `0.130%/0.488% -> 0.0%/0.0%`; `bf16_round_diff 22.66% -> 0.003%` | Router set/ord mismatch `0.065%/0.358% -> 0.0%/0.065%` | **PROMOTED** (default) |
| **`MPP::matmul2d` `W4A8` prefill (`>=8M`)** | Fallback (`Apple9`) | Active for `bits=4 >=8M`; `gpt-oss-20b` (`0.69x..0.76x`) & `<8M` excluded | **PROMOTED** (`bits=4` gated) |
| **`MPP::matmul2d` `W8A8` (`metal_mpp_int8`)** | Fallback (`Apple9`) | Synthetic normal microbench `1.16x..2.62x`, `e2b_qat4` TTFT `1.16x..1.56x`, but un-rotated `L13 down_proj` `662.2x` outliers drop L14 ANE cos to `0.997748 < 0.9983` (`3/8` greedy `sample_docs_8`, `5.066s -> 5.239s` `B=8` E2E) | **OPT-IN ONLY** (`FAST_SILICON_MPP_W8A8=1` / `allow_w8a8=True`; FP32 `fma` weight metadata packing active) |
| **Constrained `MTP`** | `1<=B<=2` (`B=1` `1.48x`, `B=2` `1.16x`; dense wins `B>=3`) | `B==1` only (`LiquidAI` **`2.026x`** wall / **`2.152x`** dec `39.06 -> 84.05 tok/s`, `8/8` T=0 match; `JSB` `1.138x` wall / `1.740x` dec `36.91 -> 64.23`) | **PROMOTED** (gated by host + `B`) |
| **`Q8Q` & `col_gemv` kernels** | `col_gemv` `M=1` `0.911x..1.009x` | `Q8Q` `0.408x..0.958x` (`cos=0.754..0.943`); `col_gemv` `0.781x..0.961x` | **REJECTED** (`NEGATIVE_NOT_PROMOTED`) |

### Track 3: `fast-pplx` (`GdnDecodeShared`, `state_codec`, `Tier-1` GPU)

- **`GdnDecodeShared` ragged throughput:** M4 Pro `7,918 -> 9,211 tok/s` (`~1.16x`); Base M5 `12,743 -> 17,412 tok/s` (`~1.37x`).
- **`state_codec`:** `state_dtype="float32"` default (`S=4096` stream vs whole `min_cos=1.0` on M4 Pro); opt-in `state_dtype="int8"` reduces carried state `3.76x` (`6,291,456 B -> 1,671,168 B`, `S=4096 min_cos=0.9999918`).
- **`Tier-1` GPU (`N=325`):** M4 Pro `0.739249` (`305.34s`, `0.7868 EOT`) -> `cand_f32` `0.738421` (`268.36s`, `0.8943 EOT`, `1.137x`) / `cand_i8` `0.738421` (`264.47s`, `0.9074 EOT`, `1.153x`, `1.278 GB`); Base M5 `0.739189` (`203.68s`, `1.1795 EOT`) -> `cand_f32` `0.739011` (`149.29s`, `1.6089 EOT`, `1.364x`) / `cand_i8` `0.739011` (`146.51s`, `1.6394 EOT`, `1.390x`).

### Track 4: Hybrid ANE FFN (`Hadamard W8A8`, `AneWave`, `MTLSharedEvent`)

- **Post-norm Hadamard `W8A8` FFN parity:** Raises 35-layer `v64a35` worst FFN cosine from `0.996032` to **`0.999740`** (vs FP16 unnormalized L11 `cos=0.743022`), with live ANE wave parity `0.999812..0.999924` on M4 Pro and Base M5.
- **`AneWave` dispatch gating:** M4 Pro `d_ff=6144` wins **`1.81x–2.40x`** vs GPU (`M=128..256`) and `d_ff=12288` wins **`1.66x–2.13x`** (`M=128..2048`), while Base M5 breaker cleanly skips negative GPU-faster regimes.
- **`p28` `MTLSharedEvent`:** At `M=1024`, M4 Pro drops `cpu_sync` (`~8.29 ms`) to `cpu_begin_return=0.45 ms` (saving `7.84 ms`), and Base M5 drops `8.75 ms` to `0.27 ms` (saving `8.64 ms`).

## 10. v2.3 Cross-Silicon Accuracy, Precision & Contextual Validation (2026-11-22)

Machine-readable validation receipt: [`benchmarks/v23-validation-20261121.json`](benchmarks/v23-validation-20261121.json). Only paired outcome/accuracy wins pass into defaults; negative experiments stay behind explicit opt-in flags or out of default code paths.

### Track B — `fast-shared` Finish-Reason, Deferred Retries, Chunking & Calibration Gates

| Fix / Gate | Baseline -> Promoted Outcome | Notes & Source |
|---|---|---|
| **Grammar `finish_reason="length"` + Async Retry** | `liquidai_112`: `5/16` valid (`0.3125`, F1 `0.2977`, `item 6` `ctok=117` mislabeled `stop`) -> **`16/16` (`1.000`, F1 `0.9114` / `0.9000`, `12/12` improved)**; `tight48`: `0/16` (`0.000` F1) -> **`16/16` (`0.9000` F1)**; `number_max8`: `4/4` emit `"length"` across `generate_batch` / `extract_rolling` / `extract_batch` | [`pipeline_llm.py`](../packages/fast-shared/src/fast_shared/pipeline_llm.py) |
| **Conditional `SWA=1024` Gating** | Unconditional `swa=1024` regressed `norm256_roll_n16` (`230.8 -> 199.9 tok/s`, `3` `ff_jump` steps); gated `roll_swa` keeps `swa=512` (`B8` `305.9–314.0`, `B16` `459.9–463.5 tok/s`) | [`pipeline_llm.py`](../packages/fast-shared/src/fast_shared/pipeline_llm.py) |
| **Header-Anchored & Sentence-Snapped Chunking** (`LegalBench-RAG`, `175` queries) | `fast-bge-small`: `0.3046 -> 0.4439` `span_ndcg@10` (`+0.1392`), `0.6314 -> 0.7099` `char_recall@10` (`auto_snap_header`, `136.6k tok/s`); `fast-embeddinggemma-2`: `0.3803 -> 0.5054` (`header_anchored`, `+0.1251`) / `0.4958` (`auto_snap_header`); cutoff `min_cos = 1.0` | [`parquet_pipeline.py`](../packages/fast-shared/src/fast_shared/parquet_pipeline.py) |
| **Multi-Batch CoreAI Calibration & Diversity Gate** | Passes diverse 4-batch (`rms_scale_ratio=709.37`, `min_cos=0.999953`); rejects `calib_batches < 2`, duplicates (`unique=1`), and batch-1 outlier collapse (`per_batch_cosine=[1.0, -0.0519]`) | [`coreai_pipeline.py`](../packages/fast-shared/src/fast_shared/coreai_pipeline.py) |

### Track C — `fast-gemma4` & `fast-gpt-oss` Precision, Loop Breakers & CoT Budgets

| Package & Experiment | Promoted Default / Verdict | Paired Evidence (M4 Pro & Base M5) |
|---|---|---|
| **`fast-gemma4` FP32 `fused_softcap_argmax_bf16` & `logit_softcap`** | **PROMOTED** | Eliminates `1/1024` BF16 greedy mismatches vs FP32 (`0/1024` mismatches). `precise::exp` SDPA (`0.8232` vs `0.8232` F1, `52.7s` vs `51.7s`): **`NEGATIVE_NOT_PROMOTED`**. |
| **`fast-gemma4` `A4_v2` 4-gram / Cycle Breaker + Lazy MTP Fallback** | **PROMOTED** (`window=96, min_prior_hits=4/5, penalty=8.0, p=1..8`) | Preserves `IFBench` `0.14` / `0.1607` (fixing `A4` v1 `0.12` regression; `rep_4gram 0.0221 -> 0.0142`), `MMLU` `0.4737`, `LiquidAI` `0.9444`, `BFCL` `0.8133`, lifting `MATH-500` `0.18 -> 0.20`. |
| **`fast-gemma4` Negative / Unchanged Defaults** | `whitespace_flexible=None`, `kv_i8=True`, `MPP W8A8=OFF` | `ws_true`: BFCL `-24%..-35%` tok/s (`mask_fb` `34->49` / `32->180`); `kv_bf16` on `>6k` (`32` docs): identical `0.7720` F1 / `1.0` valid at `6.31 GB` vs `5.06 GB`; Base M5 `FWHT-64 MPP W8A8`: `EM=5/8`, `LQ48 0.9359–0.9373` vs `nompp` `8/8`, `0.9411` (`FAST_SILICON_MPP_W8A8` opt-in). |
| **`fast-gpt-oss` Hybrid `f16` + `f32` Accumulators** | **VALIDATED OPTIMAL** (`bf16`/`fp32` **`NEGATIVE_NOT_PROMOTED`**) | Step-0 (`12` prompts) `min_cos=0.9963, max_diff=1.8109, 12/12 top1` vs `bf16_mlx` (`min_cos=0.9929, max_diff=3.5235`, `MidOut bf16` `7.5x` worse). Outcomes: `canon_low` MMLU `0.7263` (`137.2s`) beats `true_fp32` (`0.7228`, `458.9s`) and `bf16_inter` (`0.7018`, `421.9s`). |
| **`fast-gpt-oss` `canon_low` CoT Budgets** | **VALIDATED OPTIMAL** (`_EFFORT_DEFAULT_BUDGET["low"]=48`, 4-gram `window=64`) | **`IFBench 0.19` (`inst 0.2321`), `MMLU 0.7263` (`207/285`), `MATH-500 0.54`, `LiquidAI 0.8128`, `BFCL 0.8467`** vs `skip` (`MATH 0.06`, `BFCL 0.52`), `natural_low` (`MMLU 0.4421`), `tight_48` (`IF 0.16, MMLU 0.7018`), `med` (`IF 0.13, MMLU 0.6702, MATH 0.48, BFCL 0.8133, LQ 0.8307`), and `low_loop_v2` (`IF 0.15`). |

### Track D — `fast-pplx` Contextual Windows, `M5` Invariance & `LongEmbed 4K–32K`

- **Contextual header repeat (`repeat_header_tokens=128` default for `context_tokens >= 512`, `overlap=0` default):** Lifts `LegalBench-RAG` `span_ndcg@10` / `char_recall@10` from `0.4528 / 0.6827` -> **`0.7352 / 0.9154`** (Tier-1, `+0.2825`) and `0.5145 / 0.7333` -> **`0.6832 / 0.8448`** (Tier-2, `+0.1687`, `1650.4 MB` peak), plus `0.9920` on `LongEmbed 2wikimqa`; `overlap=1` alone regressed (`0.4177` Tier-1, `0.5067` Tier-2) so remains opt-in.
- **Base M5 `slice-0` `GdnDecodeShared` invariance & FP32 precision:** Passing explicit zero `rec_state` into the first streamed chunk plus FP32 `beta` `sigmoid` and `MappedEmbedding.gather(..., out_dtype="float32")` eliminated Base M5 `S=4096` stream-vs-whole drift (`3.01e-03 -> 0.0`).
- **Paired `Tier-1` (`N=325`) `GPU` vs `hetero2` (`balanced`/`fast`/`parity`) & `Tier-2 LongEmbed 4K–32K` (`N=500`):** M4 Pro `hetero2_balanced` hits `155.8s` (`1.538 EOT`, `0.7376` E2E) vs `gpu` `273.7s` (`0.878 EOT`, `0.7392`); Base M5 `hetero2_fast` hits `110.9s` (`2.163 EOT`, `0.7381`) vs `gpu` `148.8s` (`1.614 EOT`, `0.7390`). On `LongEmbed 4K–32K` (`N=500`), `float32` scores `0.7510` (M4 `2205.5s`, `1.693 GB` `phys_footprint`) / `0.7511` (M5 `1630.9s`, `1.665 GB`) vs opt-in `int8` `state_codec` `0.7519` (`1.634–1.638 GB`).

### Track E — ANE `W8A16`, `H>2048` Packing, `Early14` Hadamard & `6K` `kv_i8`

- **`ane_ffn` `W8A16` default (`quantize_a8=False`) & `H > 2048` WHT fix:** Fixes row-scale packing (`row_max_rel_err <= 0.00044`, `H=2560`/`4096` `cos=0.99989`) and protects outlier layers (`E4B_M128_outlier`: `W8A8_NoHad` `0.993737` -> `W8A8_Had` `0.999788` -> **`W8A16_Had` `0.999893`** at `2.126 ms` M4 / `2.347 ms` M5).
- **`build_ane_models.py --hadamard-mode auto`:** Applies `post_norm + grouped_conv` to `w8` (M4 Pro `cos_l14k 0.999031 -> 0.999363`, `rel_l14k 0.04298 -> 0.03552`, `40.41 -> 41.91 ms`; M5 `0.998742 -> 0.999076`, `rel_l14k 0.04783 -> 0.04104`, `37.94 -> 40.05 ms`, `2065` ANE / `0` CPU / `0` GPU ops) and `none` to `fp16` (`safe_gelu_clamp=False`, `normalize_mid=False`).
- **`6K` Hadamard `kv_i8` (opt-in):** Improves `full_D512` `V cos 0.999798 -> 0.999961`, teacher-forced `KL 0.000877 -> 0.000498` (M4) / `0.000799 -> 0.000344` (M5), and M5 `6K` `bf16` agreement `0.6667 -> 0.8333` (`46.36 -> 32.69 tok/s` M4, `66.90 -> 43.17 tok/s` M5), so standard `kv_i8` remains default (`0.558` F1 on both).

---

## 11. `fast-qwen3` Hybrid GatedDeltaNet + OptiQ + LoRA Validation (2026-11-22)

Machine-readable `ABBA` & Category-4 outcome receipt: [`benchmarks/fast-qwen3-20261122.json`](benchmarks/fast-qwen3-20261122.json).

### Category-4 Public Outcomes vs Stock `mlx-lm 0.32.2` (`LiquidAI 25 + BFCL v3 25 + JSONSchemaBench 40`)

| Host | Engine (`Qwen3.5-4B-OptiQ-4bit`) | Valid / 50 | `LiquidAI` Valid / Field F1 | `BFCL v3` AST | Combined `S_obj,e2e` | Wall (`s`) | **`EOT` (`u/s`)** | 4-Bucket Mem (`GB`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`fast-qwen3` `v0.2`** (`qwen3_auto_llg_on`) | **`50/50`** (`100%`) | **`25/25` · `0.9248`** | **`25/25` · `0.8800`** | **`0.9024`** (**`+24.05 pp`**) | **`88.01`** | **`0.513`** (**`1.53×`**) | `5.255` |
| **Base M5 (`24 GB`, `Apple10 NAX`)** | Stock `mlx-lm 0.32.2` (`public-qwen3.5`) | `37/50` (`74%`) | `12/25` · `0.4438` | `25/25` · `0.8800` | `0.6619` | `98.56` | `0.336` (`1.00×`) | `3.789` |
| **M4 Pro (`48 GB`, `Apple9`)** | **`fast-qwen3` `v0.2`** (`qwen3_auto_llg_on`) | **`50/50`** (`100%`) | **`25/25` · `0.9248`** | **`25/25` · `0.8800`** | **`0.9024`** (**`+27.99 pp`**) | `114.09` | **`0.396`** (**`1.38×`**) | **`5.179`** (**`-35.4%`**) |
| **M4 Pro (`48 GB`, `Apple9`)** | Stock `mlx-lm 0.32.2` (`public-qwen3.5`) | `35/50` (`70%`) | `10/25` · `0.3650` | `25/25` · `0.8800` | `0.6225` | `108.66` | `0.286` (`1.00×`) | `8.022` |

*(Apples-to-apples note: `fast-qwen3` runs with `llguidance` JSON Schema constraints (`qwen3_auto_llg_on`), whereas `public-qwen3.5` runs stock `mlx-lm 0.32.2` unconstrained on the identical `mlx-community/Qwen3.5-4B-OptiQ-4bit` checkpoint; both achieve `25/25` (`0.8800`) on `BFCL v3`, while schema enforcement eliminates invalid JSON formatting failures on `LiquidAI` (`25/25` vs `10–12/25`).)*

### Cross-Silicon Interleaved `ABBA` Throughput & Footprint (`M4 Pro 48 GB` vs `Base M5 24 GB`)

| Model Preset | Workload (`max_tokens=64`) | Base M5 (`24 GB`, `10c Apple10 NAX`) | M4 Pro (`48 GB`, `20c Apple9`) | Steady-State Memory (`MLX Active / MLX Peak / OS Phys`) |
| :--- | :--- | :---: | :---: | :---: |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`) | `B=1` strict JSON + LCP (`4` docs) | **`57.45 tok/s`** (`1.445 docs/s`, **`2.77×`** vs `7.66s` unfused) | **`82.67 tok/s`** (`2.080 docs/s`, **`2.59×`** vs `4.97s` unfused) | M5: `3.38 / 4.56 / 4.46 GB` · M4: `3.38 / 4.76 / 4.49 GB` |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`) | `B=8` strict JSON wave (`8` docs) | **`171.46 tok/s`** (`4.341 docs/s`, `8/8` valid vs `0/8` raw, **`8.33×`**) | **`191.04 tok/s`** (`4.836 docs/s`, `8/8` valid vs `1/8` raw, **`6.07×`**) | M5: `3.38 / 4.56 / 4.46 GB` · M4: `3.38 / 4.76 / 4.49 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=1` strict JSON + LCP (`4` docs) | **`138.09 tok/s`** (`3.389 docs/s`, **`2.80×`** vs `3.31s` unfused) | **`196.21 tok/s`** (`4.815 docs/s`, **`2.75×`** vs `2.29s` unfused) | M5: `1.67 / 2.31 / 2.99 GB` · M4: `1.67 / 2.46 / 2.86 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=8` strict JSON wave (`8` docs) | **`573.37 tok/s`** (`14.157 docs/s`, `8/8` valid, **`11.66×`**) | **`657.29 tok/s`** (`16.229 docs/s`, `8/8` valid, **`9.25×`**) | M5: `1.67 / 2.31 / 2.99 GB` · M4: `1.67 / 2.46 / 2.86 GB` |

### Validated Architectural, Outcome-Quality & Post-Training Gates

- **DRY `PrefixTree`, Right-Padded Post-LCP Invariance, Multi-Token FF & Active-Row `lm_head` (`v0.2`):** Hoisting `PrefixTree` (`256`-token chunks + block-32 `INT8` `3.76×` GDN state codec) and `compute_loop_breaker_penalties` (`A4_v2`) into `fast-shared`, replacing post-LCP left-padding with right-padded `valid_lengths` `conv1d` gathers (`4/4` parsed matches on ragged `LiquidAI` prompts), folding `upfront_ff` into prefill, coalescing `step_w >= 2` forced transitions, and slicing `lm_head` to active `M_samp` rows lifted `B=8` wave throughput `+22.5%..+33.6%` over `v0.1` (`4B-OptiQ`: `140.0 -> 171.5 tok/s` M5, `147.0 -> 191.0 tok/s` M4 Pro; `0.8B`: `460.7 -> 573.4 tok/s` M5, `491.9 -> 657.3 tok/s` M4 Pro).
- **JSON Whitespace BPE Separator Guard (`whitespace_flexible=None` Default):** Paired 16-item `LiquidAI` probe on `Qwen3.5-4B-OptiQ-4bit` confirmed `whitespace_flexible=None` (`separators=(",", ": ")`) achieves `0.9252` field F1 (`16/16` valid, `0.0` repeat 4-gram rate, `40.98 s`, `+21%` higher EOT than `flexible=True` `49.46 s`), whereas minified `whitespace_flexible=False` (`":"`) drops field F1 by **`-9.43 pts` (`0.8309`)** despite `16/16` schema validity.

- **General `OptiQ` Spec-Bucketing (`SpecBucketedLinearGroup`):** Handles non-contiguous `(4, 8, 4, 4)` GatedDeltaNet bit layouts (`9/24` GDN layers in `Qwen3.5-4B-OptiQ-4bit`) and heterogeneous `(4, 4, 8)` / `(8, 4, 4)` / `(8, 4, 8)` Self-Attention layers (`8/8` layers) with `0.0` ULP weight drift and `100%` top-1 agreement (`max_abs_logit_diff <= 1.0` across 32 `BF16` layers).
- **No-Think Default & `>= 384`-Token Thinking-Budget Floor:** On `Qwen3.5-4B-OptiQ-4bit`, `enable_thinking=False` (`no-think`) matches or beats `<think>` on structured JSON and tool AST (`BFCL-hard` `87.5%` vs `83.3%` at `3.2×` lower latency; `LiquidAI` `0.933` F1), whereas truncating `<think>` at `128` tokens on multi-step math triggers a `-12.5 pp` valley (`28.1% -> 15.6%` vs `78.1%` at `1024` tokens). `validate_thinking_budget` enforces `>= 384` tokens whenever `enable_thinking=True`.
- **Hybrid `GatedDeltaNet` LoRA Regime (`R2`) & Double-Quantization Guard:** Including `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` alongside `self_attn + mlp` in `BF16` (`R2`, keeping `conv1d/A_log/dt_bias` frozen) beats `self_attn + mlp` alone (`R1`) by **`+17.5 pp`** (`66.7%` vs `49.2%`, `p=0.00051`) and parameter-matched `R1` (`r=16`) by **`+21.7 pp`**, whereas unfreezing `conv1d/A_log/dt_bias` (`R4`) collapses `-18.3 pp` to `-30.8 pp` and 2-pass `4-bit -> LoRA -> 4-bit` requantization collapses exact match `64.2% -> 10.0%` (`blocked` by `QwenDoubleQuantError`).
- **Empirical Speculative `MTP` Auto-Gating (`>= 9B` Native MTP):** On `0.8B` (`H=1024`), the `248,320`-row `lm_head` (`508 MB`, `33%` of model bytes) makes `K=4` `Qwen3MtpDrafter` projections `0.78×–0.89×` slower than `B=1` single-step decode (`89.2 -> 69.4 tok/s` M4 Pro; `73.7 -> 65.3 tok/s` Base M5, `100%` greedy token match), and `PromptLookupDrafter` on `4B-OptiQ` wins `1.12×` on Base M5 (`33.9 -> 37.9 tok/s`) but `0.86×` on M4 Pro (`46.7 -> 40.3 tok/s`). `speculative=None` (`"auto"`) therefore stays **`OFF` below `9B` native-MTP models**, with explicit `speculative=True` opt-in.
- **`9B-OptiQ` Native `MTP` (`norm_add_one` + `draft_k=3`), `ByteLevel` `65K` Vocab Remap & `Cat 3/4` Receipts (`v2.4`, `2026-11-22`):** Auto-detecting `optiq/mtp.safetensors` zero-centered RMSNorm weights (`detect_mtp_norm_add_one`) plus setting `recommended_draft_k=3` eliminated the pre-fix `MTP` slowdown (`13.25 tok/s` / `0.441×` M4 Pro, `10.13 tok/s` / `0.523×` Base M5), and decoding `ByteLevel` BPE tokens in `QwenVocabRemapper` restored compact `65,536`-entry vocab speedups:

| Model & Vocab (`v2.4`) | Workload | Base M5 (`24 GB`, `Apple10 NAX`) | M4 Pro (`48 GB`, `Apple9`) | Peak Footprint (`M5` / `M4 Pro`) |
| :--- | :--- | :---: | :---: | :---: |
| **`9B-OptiQ`** (`248K` vocab, `k=3`) | `B=1` LCP · `MTP B=1` · `B=8` wave | `32.73` · **`32.44 tok/s`** (**`1.68×`** vs `19.34`) · **`101.23 tok/s`** | `49.88` · **`41.43 tok/s`** (**`1.28×`** vs `32.32`) · **`112.90 tok/s`** | `16.18 GB` (`< 19.2`) / `16.36 GB` |
| **`9B-OptiQ`** (`65K` decoded, `k=3`) | `B=1` LCP · `MTP B=1` · `B=8` wave | `37.17` · **`42.25 tok/s`** (**`1.92×`**, `k`-sweep `44.29`) · **`105.74 tok/s`** | `54.81` · **`50.17 tok/s`** (**`1.40×`**, `k`-sweep `58.40`) · **`115.93 tok/s`** | `16.68 GB` (`< 19.2`) / `16.87 GB` |
| **`0.8B`** (`248K -> 65K` decoded) | `B=1` LCP · `B=8` strict JSON wave | `139.6 ->` **`178.3 tok/s`** · `570.7 ->` **`658.5 tok/s`** (`16.3 docs/s`) | `187.3 ->` **`240.3 tok/s`** · `635.6 ->` **`767.5 tok/s`** (`19.0 docs/s`) | `4.61–4.81 GB` / `4.77–4.99 GB` |
| **`9B-OptiQ --mtp-drafter`** | `Cat 4` (`LiquidAI 25 + BFCL 25`) | **`0.8857`** (`50/50` valid, `110.2 s`, **`0.402 EOT`**) | **`0.8874`** (`50/50` valid, `160.0 s`, **`0.277 EOT`**) | `10.94 GB` / `12.41 GB` |
| **`4B` / `9B-OptiQ`** (`enable_thinking=False`) | `Cat 3` (`IFBench 25 + MMLU 25 + MATH 25`) | `0.5333` (`4B`) · `0.5467` (`9B`) *(held down by `IFBench 0.16`)* | `0.5467` (`4B`) · `0.5733` (`9B`) *(`MMLU 0.76–0.80, MATH 0.68–0.72`)* | `4.13–8.15 GB` / `4.32–8.32 GB` |

- **Real-Checkpoint Parities & Concurrent `GPU + Pure-ANE` (`v2.4`):** `9B-OptiQ` passed `S>256` (`288..290`) ragged `B=1` vs `B=4` (`4/4`), `PrefixTree` wave vs uncached `kv_i8`/`bf16`, and `S=2213` chunked vs whole first-token parity across both Macs; Base M5 concurrent `4B-OptiQ GPU` (`8×48` tokens) + `FastBgeSmall pure-ANE` (`3×32×384`) completed in `2.626 s` wall inside **`9.39 GB` peak** (`OS phys 5.32 GB`, well under the `19.2 GB` `24 GB`-class ceiling).

---

## 12. v2.5 Cross-Model R&D Winners & Negative Gates (2026-11-22)

Machine-readable validation receipt: [`benchmarks/v25-rd-winners-20261122.json`](benchmarks/v25-rd-winners-20261122.json). All cross-model mechanisms live once in `packages/fast-shared` (`retrieval_index.py`, `kv_quant.py`, `parquet_pipeline.py`) with thin adapters across model packages; negative candidates remain out of defaults.

### 12.1 Multi-Vector Contextual `MaxSim` Retrieval (`fast-pplx` + `fast_shared.retrieval_index`)

Evaluated on `LongEmbed` (`N=500`: `200` `passkey` across `4K/8K/16K/32K` + `300` `2wikimqa`):

| Pooling / Window Strategy | Pooled `N=500` `nDCG@10` | Per-Split `N=500` `nDCG@10` | `32K passkey` (`split` / `pooled`, `R@10`) | `2wikimqa` (`N=300`) | M4 Pro GPU Encode Wall / Metal Peak | Storage / Doc | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-vector whole-doc (`FP32` `1024d`) | `0.7510` | `0.7532` | `0.0470` / `0.0470` (`0.14`) | `0.8945` | `2205.5 s` / `4.27 GB` | `4.00 KiB` | baseline |
| **`encode_documents_multivector` (`win=1024, ctx=2048, rep=128`, `MaxSim`, `1024d INT8`)** | **`0.8452`** | **`0.9911`** | **`1.0000` / `0.6743` (`1.00`)** | **`0.9851`** | **`955.4 s` (`2.31×`)** / **`1.72 GB` (`2.48×` lower)** | `15.97 KiB` (`0.0000` loss vs `FP32`) | **PROMOTED** (`multivector=True`) |
| `encode_documents_multivector` (`win=512, ctx=2048, rep=128`, `MaxSim`, `1024d INT8`) | `0.8449` | `0.9923` | `1.0000` / `0.6695` (`1.00`) | `0.9871` (`0.9884` `top2_mean`) | `955.4 s` / `1.72 GB` | `31.11 KiB` | opt-in (`window_tokens=512`) |
| Hybrid `0.7*max + 0.3*mean` (`win=512`) | `0.7689` | `0.9901` | `0.9926` / `0.4710` (`0.82`) | `0.9838` | `955.4 s` / `1.72 GB` | `31.11 KiB` | **`NEGATIVE_DEFAULT`** (`1/L` dilution) |

*(Apples-to-apples note: official `LongEmbed` `passkey` splits evaluate 50 queries against 100 docs at a single context length and reach **`1.0000` `nDCG@10` (`1.00` `R@1`)** across all `4K`, `8K`, `16K`, and `32K` splits; `benchmark_outcomes.py` also pools `[4K, 8K, 16K, 32K]` into one `400`-doc corpus where identical passkey entities exist in all 4 splits, capping any length-invariant oracle near `0.640–0.680` at `Recall@10 = 1.000`.)*

### 12.2 Model-Aware `MRL + INT8` & ARM64 NEON Binary Rerank (`fast-bge-small`, `fast-embeddinggemma-2`, `fast-pplx`)

Evaluated across `NanoBEIR-en` (`150` queries) and `LegalBench-RAG` (`175` queries / `796` chunks), with interactive single-query (`Q=1`, median of 64 queries on M4 Pro) scan latencies measured directly on `100,000`-vector `CompactRetrievalIndex` instances against `FP32` BLAS (`2.23` / `2.82` / `1.46 ms/q`):

| Engine (`is_mrl_trained`) | Storage / Search Mode | Bytes / Vec (Compression) | `NanoBEIR` `ΔnDCG@10` (`95% CI`) | `LegalBench-RAG` `Δspan_nDCG@10` (`95% CI`) | Interactive `Q=1` Scan (`100K` Docs vs `FP32` BLAS) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`FastEmbeddingGemma2`** (`True`, `768d`) | **`512d MRL + sym_int8`** (`mode="auto"` `< 16K`) | **`514 B` (`5.98×`)** | `-0.0037` `[-0.0110, +0.0030]` | **`+0.0038`** `[-0.0079, +0.0159]` *(pooled `+0.0003`)* | **`1.42 ms/q`** (`1.57×` vs `2.23 ms/q`) | **PROMOTED** (`CorpusBackfillConfig.for_engine`) |
| **`FastEmbeddingGemma2`** (`True`, `768d`) | **`bin512 -> Top-64 -> 512d INT8`** (`mode="auto"` `>= 16K`) | `64 B` hot (`48×`) / `578 B` (`5.31×`) | `-0.0078` *(`-0.0037` Top-256)* | **`+0.0041`** `[-0.0080, +0.0161]` | **`0.50 ms/q`** (**`4.43×` faster**; `9.2×` batched) | **PROMOTED** (`>= 16K` docs / opt-in) |
| **`FastPplx`** (`False`, `1024d`) | **`1024d sym_int8`** (`mode="auto"` `< 16K`) | **`1,026 B` (`3.99×`)** | **`-0.0007`** `[-0.0020, +0.0007]` | **`-0.0012`** `[-0.0029, +0.0005]` | **`2.11 ms/q`** (`1.34×` vs `2.82 ms/q`) | **PROMOTED** (`mrl_dim=None` enforced) |
| **`FastPplx`** (`False`, `1024d`) | **`bin1024 -> Top-64 -> 1024d INT8`** (`mode="auto"` `>= 16K`) | `128 B` hot (`32×`) / `1,154 B` (`3.55×`) | **`+0.0006`** `[-0.0016, +0.0036]` | **`-0.0012`** `[-0.0029, +0.0005]` | **`0.66 ms/q`** (**`4.28×` faster**; `0.61 ms/q` batched) | **PROMOTED** (CI95-lossless) |
| `FastPplx` (`False`, `1024d`) | `512d MRL + sym_int8` | `514 B` (`7.97×`) | `-0.0110` (`-3.10 pp` `NanoSciFact`) | **`-0.0373`** (`-0.0377` `FP32`) | — | **`NEGATIVE_NOT_PROMOTED`** (non-MRL) |
| **`FastBgeSmall`** (`False`, `384d`) | **`384d sym_int8`** (`mode="auto"` `< 16K`) | **`386 B` (`3.98×`)** | **`-0.0013`** `[-0.0044, +0.0018]` | **`+0.0026`** `[-0.0017, +0.0070]` | **`1.16 ms/q`** (`1.26×` vs `1.46 ms/q`) | **PROMOTED** (`mrl_dim=None` enforced) |
| **`FastBgeSmall`** (`False`, `384d`) | **`bin384 -> Top-64 -> 384d INT8`** (`mode="auto"` `>= 16K`) | `48 B` hot (`32×`) / `434 B` (`3.54×`) | **`-0.0003`** `[-0.0065, +0.0060]` | **`+0.0026`** `[-0.0017, +0.0070]` | **`0.50 ms/q`** (**`2.95×` faster** than `1.46 ms/q`) | **PROMOTED** (CI95-lossless) |
| `FastBgeSmall` (`False`, `384d`) | `256d MRL + sym_int8` | `258 B` (`5.95×`) | **`-0.0184`** `[-0.0351, -0.0016]` | — | — | **`NEGATIVE_NOT_PROMOTED`** (non-MRL) |

### 12.3 Fused Quantized `KV` SDPA, `QuaRot` `4-Bit KV`, `fast-pplx` `FFD`/`hetero2` & `Kev-4B` `SpecBucket`

| Mechanism / Target | Compression / Speedup | Quality & Parity Evidence (`M4 Pro 48 GB` & `Base M5 24 GB`) | Promotion Status |
| :--- | :---: | :--- | :---: |
| **Online `(Q, K, V)` `FWHT-64` `QuaRot` (`kv_i8="q4_quarot"`, post-RoPE + pre-gate inverse)** | **`3.56×` KV compression** (`576 MB` vs `2,048 MB` `BF16` @ `16K B=4`) · **`91.71 tok/s` E2E (`2.13×` faster than legacy `kv_i8` `43.05 tok/s`; `110.90 tok/s` `BF16`)** @ `16K B=4` (`2.70 ms` 1-layer SDPA) | **`4K` (`N=9`):** `0.9602 == BF16` `LiquidAI` F1, **`27/27` (`100%`) PK**, `100%` exact greedy seq; **`8K` (`N=6`):** **`0.8667` vs `0.8644` `BF16` F1, `18/18` (`100%`) PK**; **`16K` (`N=6`):** **`0.9464 == BF16` F1** (vs **`0.8631` raw `q4_g64` = `-8.33 pp`**), **`18/18` (`100%`) PK**, `100%` top-1 agree, `83.3%` vs `33.3%` exact seq (`KL 0.0016` vs `0.0034`) | **PROMOTED** (`kv_i8="auto"` @ `>=16K` / `>=8K` `<=24GB` / `B>=16`) |
| **`fused_quantized_sdpa` (`kv_i8="q8"`, `mx.quantized_matmul` on packed KV)** | **`1.98×` KV compression** (`1,032 MB`) · **`4.72×` faster** 1-layer `B=4, T=16K` SDPA kernel (`2.53 ms` vs `11.94 ms` Python-dequant `kv_i8`; `1.61 ms` `BF16`) | Strided non-compact `qvm_split_k` parity `max_diff = 0.000732` (`T=4261, alloc=4864`); `4K` / `8K` / `16K`: `0.9602 / 0.8644 / 0.9464 == BF16` F1, **`100%` passkeys (`63/63`)**, `KL = 0.00012–0.00025`, `100%` exact greedy seq @ `16K` | **SUPPORTED** (`kv_i8=True` / `"q8"`) |
| **`fast-pplx` `FFD` `pack_ane_buckets` + Host-Aware `hetero2` + Async Parquet + Opt-In `W8` (`T4.1–T4.5`)** | **`26` vs `32` ANE bins (`-18.8%`)**, **`648` vs `1,638` pad toks (`-60.4%`)**; Mixed-64 **`19,443 tok/s`** M5 (`1.17×`) / **`17,793 tok/s`** M4 (**`1.84×`** vs GPU); `w8` **`-46.4%` weights** (`460.8 -> 247.0 MB`) | `binary_int8` `multivector=True` (`72` wins): **`1.70 ms` M5 / `6.43 ms` M4**; `w8` 16-case `min_f32_cos = 0.99883` (`0.95×` FP16 M5 speed, opt-in `default="none"`). `S=768/1024` ANE MIL: M4 GPU fallback (`all_on_target=False`) & M5 GPU `1.40×` faster (`0.69–0.72×`) | **PROMOTED** (`FFD` / `hetero2` / async; `w8` opt-in; `S<=512` kept) |
| **`SpecBucketedLinearGroup` + `compress_gdn_prefix_int8="auto"` in `fast-decision-models` (`Kev-0.8B` / `Kev-4B`)** | **`3.76×` smaller GDN `PrefixTree` (`1.57 MB -> 0.42 MB`/node on `Kev-4B`)** + `2` GEMMs vs `4` on mixed `4b`/`8b` layers | `0.0`-ULP mixed-bit `(4, 8, 4, 4)` projection parity and `0.0`-logit-delta cold-vs-warm `PrefixTree` round-trip (`9/9` CPU + `stdlib_isolated_ok`) | **PROMOTED** (`kev_mlx.py`) |
| **Base M5 `macOS 27.2.1` `CoreAI Qwen3.5-4B` (`.aimodel` `w4` / `fp4_fp8`)** | **`0` ANE ops** (`64/64` GPU ops, `0.21×` EOT, `1.5×` RAM) | `6.92–9.04 tok/s` (`0.12×–0.16×` `fast-qwen3` `57.45 tok/s`), `Cat-4` `0.838` (`258.5s`, `0.162 EOT`, `7.94 GB` peak) vs `fast-qwen3` **`0.9024` (`58.3s`, `0.774 EOT`, `5.26 GB`)**; Python `CoreAI` API lacks `GatedDeltaNet` recurrent state carry | **`NEGATIVE_NOT_PROMOTED`** (harness baseline only) |

---

## 13. Vision, OCR (`Category 5`) & Apple10 NAX Prefill Validation (2026-11-23)

All visual OCR runs in this section compare forced VLM/vision decoding on identical rendered page images (`--ocr-backend ovis --ocr-render-only`, disabling native PDF bypasses so visual recognizers are scored apples-to-apples) across **Apple M4 Pro (`48 GB`, `macOS 26.6`, `25G5057c`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`, `26B131`)**. Detailed architecture notes and per-stage probe receipts live in [`packages/fast-ocr/docs/OCR_RESEARCH_AND_ARCHITECTURE.md`](../packages/fast-ocr/docs/OCR_RESEARCH_AND_ARCHITECTURE.md) and [`packages/fast-ocr/docs/OCR_BENCHMARKS_AND_BASELINES.md`](../packages/fast-ocr/docs/OCR_BENCHMARKS_AND_BASELINES.md).

### 13.1 Category 5 Cross-Family Visual OCR (`--ocr-render-only`)

Post-fix live smoke runs passed **`12/12`** checks across M4 Pro and Base M5 (`fast-ocr` with `auto_loop_recover=True` and `AppleVisionOCR` tiny-crop `_prepare_vision_image` upscaling; default `--ocr-backend auto` additionally routes clean digital PDF pages via `pypdfium2` at `138.8–220.2 pp/s` and single-column scans via zero-MLX ANE at `1–7.5 pp/s`):

| Baseline (`mlx_vlm`) / Family | Benchmark | M4 Pro `pp/s` Ratio (`fast-ocr` vs. Baseline) | M4 Pro Dec `tok/s` Ratio | M4 Pro `EOT` Ratio | Base M5 `pp/s` Ratio (`fast-ocr` vs. Baseline) | Base M5 Dec `tok/s` Ratio | Base M5 `EOT` Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| **`OvisOCR2-8bit` (same weights)** | `olmOCR-bench(4)` | **1.43×** | **1.81×** | **1.44×** | **2.02×** | **2.29×** | **2.05×** |
| **`OvisOCR2-8bit` (same weights)** | `FinTabNet-OTSL(2)` | **1.22×** | **1.42×** | **1.22×** | **1.72×** | **1.85×** | **1.65×** |
| **`OvisOCR2-8bit` (same weights)** | `OmniDocBench-EN(2)` | **1.51×** | **1.69×** | **1.51×** | **1.53×** | **1.57×** | **1.53×** |
| **`GLM-OCR-bf16`** | `olmOCR-bench(4)` | **2.54×** | **2.56×** | **3.57×** | **3.47×** | **3.72×** | **5.00×** |
| **`GLM-OCR-bf16`** | `FinTabNet-OTSL(2)` | **1.46×** | **1.72×** | **1.87×** | **2.23×** | **2.44×** | **2.85×** |
| **`GLM-OCR-bf16`** | `OmniDocBench-EN(2)` | **2.19×** | **2.01×** | **2.10×** | **2.44×** | **2.47×** | **2.35×** |
| **`LightOnOCR-2-1B-bf16`** | `olmOCR-bench(4)` | **2.41×** | **2.99×** | **2.41×** | **3.66×** | **3.98×** | **3.75×** |
| **`LightOnOCR-2-1B-bf16`** | `FinTabNet-OTSL(2)` | **13.0×*** | **2.20×** | *(baseline `0.00`)* | **20.2×*** | **3.07×** | *(baseline `0.00`)* |
| **`LightOnOCR-2-1B-bf16`** | `OmniDocBench-EN(2)` | **1.97×** | **1.93×** | **1.91×** | **2.74×** | **2.66×** | **2.63×** |

*\*`LightOnOCR-2-1B-bf16` hit degenerate repetition loops on cropped `FinTabNet-OTSL` tables (`0/2` valid).*

- **`fast-ocr` scores & resource receipts:** On `olmOCR-bench(4)` (after `auto_loop_recover=True` uncapped retry on dense multi-column page `[2]`), `fast-ocr` improved from `3/4` (`0.5938`) to **`4/4` (`0.8438 PASS`)** on both hosts: **`0.082` EOT** (`323.3` dec / `210.5` E2E `tok/s`, `5.75 GB` vs `7.82 GB` pre-fix) on M4 Pro and **`0.090` EOT** (**`2.05×`** vs stock `0.044`, `1.11×` vs M4 Pro, `286.1` dec / `232.7` E2E `tok/s`, `5.01 GB` vs `6.61 GB`) on Base M5. On `FinTabNet-OTSL(2)` and `OmniDocBench-EN(2)`, `fast-ocr` passed at **`0.8687` TEDS** (`0.674` M4 Pro / `0.733` M5 `pp/s`) and **`0.9093`** (`0.117` M4 Pro / `0.124` M5 `pp/s`), whereas stock `mlx_vlm` slowed `0.64×–0.81×` on Base M5.
- **`public-apple-vision` & non-VLM local baselines:** After adding tiny-crop bicubic upscaling (`_prepare_vision_image`: `min_long_edge=960`, `min_short_edge=240`, `max_scale=3.0`; `1.0×` unchanged on full pages), `public-apple-vision` on `FinTabNet-OTSL(2)` lifted from `0.5161` to **`0.6121 PASS`** (`7.03 pp/s`) on M4 Pro (`macOS 26.6`, `rev=3`) and from `0/2` (`0.0000`) to **`2/2` (`0.5914 PASS`)** at **`25.16 pp/s`** (`0.04 s/page`, `14.88` EOT) on Base M5 (`macOS 27.2.1`, `rev=3`), with full-page ANE throughput holding at `4.97–7.55 pp/s`. Pure CPU baselines (`pypdfium2` `138.8` / `220.2 pp/s` native text; M4 Pro `Docling` `0.130–2.514 pp/s` `0.4397` OmniDoc / `0.4547` FinTabNet; `Tesseract` `0.589–12.305 pp/s` `0.2717` OmniDoc; `RapidOCR` `0.261–1.488 pp/s` `0.1458` OmniDoc) trail VLM accuracy on complex scans.

### 13.2 Grammar-Locked PLD, `zoom_recover`, Triage & PDF Hygiene (`fast-ocr`)

- **Ink cropping, ByteLevel vocab pruning & double buffering (`OvisOCR2-8bit`):** Foreground `crop_to_ink` (`-56.3%` to `-61.4%` ViT tokens) cut Page A ViT latency **`4.47×`** on M4 Pro (`7,726 -> 1,728 ms`) and **`5.68×`** on Base M5 (`6,542 -> 1,152 ms`); ByteLevel-decoded `build_ocr_vocab_subset(target_size=None)` (`248,320 -> 140,307`) sped decode `1.10–1.12×` at `100.0%` token parity; overlapping ANE `N+1` behind GPU `N` hid `100.0%` of ANE pre-pass work (`0.03 ms` wait). In Wave-3 (`[prose, Page A, Page B, Page C]`), `fast-ocr` finished in `29,422 ms` (`M4 Pro`, `441.3 tok/s` batch Page C) and **`28,219 ms`** (`Base M5`, `1.18×` faster Page B end-to-end) with `4/4 valid=True` and `45/45` Page C dollar amounts.
- **Grammar-locked `OCRPLDDrafter` (`DynamicTableRowGrammar.clone()` / `LLMatcher.deep_copy()`):** Keeping PLD active under `--strict-markdown-grammar` delivered **`1.85×` decode speedup** (`170.0 -> 315.2 tok/s`, `6/6` exact parity on `FinTabNet` `n=6` M4 Pro) and **`1.144×` wall speedup** (`4,154.7 -> 3,632.8 ms`, `2/2` exact parity, identical `TEDS=0.868746` on `FinTabNet-OTSL` `n=2` Base M5); JSON-schema PLD on `CORD-v2` (`n=4`) ran **`1.11×–1.80×` `tok/s`** with exact parity and zero-cost `metadata["field_boxes"]` provenance.
- **ANE-guided `--zoom-recover`, `--triage-dpi`, and PDF hygiene:** Hamming-1 filtered numeric zoom recovery (`_is_single_digit_substitution`) lifted `FinTabNet` (`n=6`) numeric recall **`0.994 -> 1.000`** without wasting crops on `FT[5]`; `triage_dpi=100` cut 10-page packet triage **`16.62×`** (`44,762 -> 2,694 ms`); PDF routing strips running typesetting stamps (`strip_running_pdf_boilerplate`), rejects any C0/C1/`\ufffd` control char (`count_control_garble_chars == 0`), checks `FPDF_PAGEOBJ_IMAGE` so vector-outlined pages still rasterize, and clamps oversized renders at `--max-pdf-pixels` (`20M` default, `min_dpi=36`).

### 13.3 `fast-pplx` Visual Embeddings & `fast-qwen3` Compiled Kernels / `Apple10` NAX Prefill

| Component / Mode | M4 Pro (`Apple9`) | Base M5 (`Apple10 NAX`) | Parity & Comparison Notes |
|---|---:|---:|---|
| **`fast-pplx` `encode_images` (`100 DPI`, `chunks=8`, `fp16`)** | **`159.12 ms/p`** (`6.28 pp/s`, `1.87 GB`) | **`109.24 ms/p`** (`9.15 pp/s`, `1.72 GB`) | **`3.60×`** (M4 Pro) / **`5.25×`** (M5) vs PyTorch MPS (`573.40 ms/p`); `mean_cos = 0.99970`, `0.0` MRoPE diff, `1.000000` `90/180/270` CW rotation parity, `build_image_index` `6/6` R@1 (`10/10` intra-packet) |
| **`fast-pplx` `encode_images` (`150 DPI`, `chunks=8`, `fp16`)** | **`688.48 ms/p`** (`1.45 pp/s`, `2.33 GB`) | **`424.87 ms/p`** (`2.35 pp/s`, `2.33 GB`) | **`2.95×`** (M4 Pro) / **`4.79×`** (M5) vs PyTorch MPS (`2,033.55 ms/p`); `mean_cos = 0.99983–0.99984` |
| **`fast-qwen3` `@mx.compile` GDN/RoPE kernels** | **`1.053×–1.303×`** (`0.0` `max_abs_diff`) | **`1.038×–1.440×`** (`0.0` `max_abs_diff`) | `RMSNormGated`, `normalize_gdn_qk` (`shapeless=True`), and `apply_partial_rope_trig` |
| **`fast-qwen3` NAX `W4A8` default (`_mpp_linear.py`, `M>=128`)** | Disabled (`Apple9`) | **`1.06×–1.10×`** prefill (`4B`/`9B` `L=512`) · **`+4.6%` EOT** (`9B` Cat 4, `0.4128 -> 0.4316`) | `50/50 valid`, **`0.8857` exact `S_obj,e2e` parity** (`delta_mean=0.0`), `107.28 -> 102.61 s` (`12.57 GB` vs `10.07 GB`), `28%` RAM pack cap + `.vjp` LoRA preserved |
| **`fast-qwen3` NAX `W8A8` opt-in (`FAST_SILICON_QWEN3_MPP_W8A8=1`)** | Disabled (`Apple9`) | **`1.14×–1.18×`** prefill (`4B`/`9B`) · **`+8.3%` EOT** (`9B` Cat 4, `0.4469`) · **`-7.1%` LM prefill** (`OvisOCR2`) | `50/50 valid`, `delta=-0.00153` (`se=0.00193`); `OvisOCR2-8bit` `1024×7168` / `3584×1024` wins `1.29×–2.48×` micro (`cos >= 0.999974`) and `909.8 -> 845.6 ms` LM prefill on `6` `cord_v2` receipts |

**Rejected Apple10 NAX probes (kept out of defaults):** Uneven ViT block-diagonal/padded SDPA (`0.21×–0.43×` slower than `mx.split`), ViT MPP MLP fusion (`~1.01×–1.10×`, high drift), wide `64×128` / `128×64` / `128×128` MPP tiles (`~5×` slower than `64×64s4`), `M <= 64` `W4A8`/`W8A8` (`0.33×–0.90×`), Gemma-4's `should_use_mpp_w4a8` heuristics on Qwen3.5 shapes, `FastVLM-0.5B` (`0.089` FinTabNet recall), `MobileCLIP2-S0` (`0.199` FP16 cos / `1/35` R@1), and macOS 26 `VNRecognizeDocumentsRequest` (`0` tables on `FT[3]`/`FT[5]`).

### 13.4 `B > 8` Vision Batching, Parallel ANE & Gov-Doc Category 5 Expansion (M4 Pro Verified)

- **Parallel ANE (`run_vision_batch_parallel`, `fast-ocr` `pipeline.py`):** Running `AppleVisionOCR` with `workers=4` (`min_batch=4, min_total_pixels=2_000_000`) accelerated `n=16` resized `FinTabNet` `1024×512` pages **`1.592×`** (`2.651 s` / `6.04 pp/s` serial → **`1.665 s` / `9.61 pp/s`**; `1.69 s` / `9.47 pp/s` through `FastOCR.ocr_images(..., batch_size=16)`) with **`16/16` exact text parity**.
- **Batched `OvisOCR2-8bit` Decode (`batch_size=16` via `batched_multimodal_generate`):** Using `pack_length_waves`, `compact_hybrid_cache`, `512`-token prefill chunking, and `kv_step <= 512` lifted `FinTabNet-OTSL(16)` (`gfm`) **`1.663×`** (`21.621 s` / `0.74 pp/s` / `281.6 tok/s` / `3.07 GB` phys at `B=1` → **`12.998 s` / `1.231 pp/s` / `468.0 tok/s`** / `4.362 GB` phys at `B=16`) with identical **`TEDS=0.8675` / `NumF1=0.8945`** (`15/16` exact text parity). Conversely, on long ANE-assisted prose (`olmOCR-bench(8)` @ `150 DPI`), `B=1` with `ANE+PLD` (`vision+self`) stayed faster (`119.143 s` / `73.0 tok/s`) than `B=8` AR batched waves (`134.531 s` / `64.7 tok/s`, `0.886×`, identical `0.6205` macro score), confirming `B=16` waves belong on tables/short outputs while `B=1` PLD leads long prose.
- **`FastQwen3Engine.generate_batch(..., images=..., compact_active=True)` (`CORD-v2`, `Qwen/Qwen3.5-0.8B` BF16):** `B=8` achieved **`1.11×`** speedup over `B=1` (`21.418 s` → `19.295 s` / `0.415 docs/s`, `4.898 GB` phys, `NOCMP == CMP 4/4`); `B=16` completed 4 compacted waves with `valid_schema_rate=1.0` (`5.249 GB` phys).
- **`FastPplx.encode_images(..., batch_size=16)` (`n=24` mixed `FinTabNet+CORD`):** Matched `B=4` (`3.7438 s` / `6.41 img/s` / `4.044 GB` phys vs `3.7265 s` / `6.44 img/s`, **`min_cos=1.00000`**), while `white_pad_to_grid_bucket(grid_bucket_step=4)` lowered small-crop similarity (`mean_cos=0.94123`, `min_cos=0.7572`) and remains opt-in (`--grid-bucket-step`).
- **Large-Batch GDN/KV State Compression, Region Routing & Auto Camera Deskew (`M4 Pro` Verified):**
  - *`GdnDecodeShared` Metal State (`INT8-blk32`/`BF16`/`FP32`) + `auto` `q4_quarot` KV + opt-in `drop_white_patches` (`fast-qwen3` / `fast-ocr`):* Quantizing the `18` `OvisOCR2` `float32` GDN states (`FP32` decay `g`, exact `mask` freeze `frozen_diff=0.0`, `3.76×` DRAM cut, `256`-step AR `y_cos=0.999988`/`0.999971`/`0.999808`) accelerated 18-layer M4 Pro decode **`1.422×` @ `B=1`** (`1.022 -> 0.719 ms`), **`1.664×` @ `B=8`** (`2.623 -> 1.576 ms`), **`2.508×` @ `B=16`** (`5.314 -> 2.119 ms`), **`1.890×` @ `B=32`** (`11.334 -> 5.996 ms`), and **`3.432×` @ `B=64`** (`31.753 -> 9.252 ms` `bf16` / `14.692 ms` `int8_blk32`). To preserve `B=1` PLD latency (`1.49 s`), `gdn_state_mode="auto"` keeps `fp32` for `B < 8` (`< 4` on `<= 28 GB`) and switches to `int8_blk32` at `B >= 8` (`>= 4`); paired with `resolve_ovis_kv_mode` *(initially `"q4_quarot"` at `B >= 8`, tuned in `6826e34`, and further aligned in `5cc9610` to pressure-only gating on both Apple10 NAX and Apple9: `ctx >= 8192/16384 or B*ctx >= 262_144/524_288` with no blind `B >= 32` trigger)* on `16` E2E M4 Pro pages (`8` FinTabNet + `8` OmniDocBench, `max_tokens=384`), `auto` defaults cut wall time **`-9.0%`** (`46.81 -> 42.59 s`, **`14/16` exact, `median_sim=1.0000`, `min_sim=0.9271`**, `16/16` valid MD). `drop_white_patches` defaults **`False` (opt-in)** because dropping interior white column gaps caused compact-table hallucinations (`FT_175`, `FT_240`: `exact=2/16`, `min_sim=0.3899`).
  - *Ordered PLD, `24px` Adaptive Resolution, `deskew="auto"` & `region_routing`:* See [Section 13.5](#135-lossless-ocr-resolution-windowed-cursor-pld-auto-camera-photo-deskew--region-routing-m4-pro-verified) for paired M4 Pro receipts (`0/106` non-photo `deskew="auto"` false warps, `ordered_pld` + `adaptive_24`, `window_relock=True`, opt-in `region_routing` with multi-column caveat), Latin-gated scientific math/footnote vocab symbols (`--no-latin-only` overrides), and CLI/`benchmark_outcomes.py` (`--ocr-batch-size`, `puredocbench_en`) wiring.
- **Gov-Doc Category 5 Expansion (`configs/outcome_benchmarks.lock.json` `24` Datasets + 8-Source `us_gov_ocr`):**
  - Added `vrdu_ad_buy`, `vrdu_registration`, `doclaynet_gov`, and opt-in `puredocbench_legal_gov` alongside keyless 3-branch `us_gov_ocr` staging (`scripts/benchmark_outcomes.py stage-gov` with `--sources`, `--loc-btp-path`, `--no-loc-btp`): `GovInfo` `PLAW` USLM, `BILLS` XML, `USCODE` chapter HTML↔PDF, `FR` `<GPOTABLE>`, `BUDGET-2027-PER` `.xlsx`↔PDF tables, `USCOURTS`, `incrediblecrab/us-scotus-opinions`, and auto-staged LoC `btp-data`.
  - **Base M5 Bulk Staging & Initial CPU Baseline (`0` GPU work, `427Gi` free):** Staged `23/24` Tier-2 lockfile datasets (`vrdu_*` `100` each, `doclaynet_gov` `90` pp), extracted all `453` PureDocBench `03_legal_gov` images (`72/72` EN verified across `clean`/`digital`/`real`), and staged `31` verified `us_gov_ocr` items across all `8` sources (`60M`); `public-pypdfium2` on `US-Gov-OCR(27)` scored `1.0` on single-column `uscourts`/`scotus` vs `0.2979` overall and `0.0` `table_teds` (`govinfo_budget=0.0104`, `govinfo_bills=0.2073`, `govinfo_fr=0.5135`, `govinfo_uscode=0.5761`). *(Base M5 GPU validation of `B > 8` and M4 Pro / Base M5 GPU/ANE Gov-Doc runs remain tracked under `mikewolfd/fast-silicon#1`.)*

### 13.5 Lossless OCR Resolution, Windowed Cursor PLD, Auto Camera-Photo Deskew & Region Routing (M4 Pro Verified)

- **Ordered PLD & Lossless `24 px/line` Adaptive Resolution (`ordered_pld` + `adaptive_resolution=True`, default on):** Combining XY-cut reading-order + native PDF prose corpus seeding (`k=6` with native PDF text, `k=4` Vision-only) with a line-height adaptive visual cap (`24 px/line` prose, `28 px/line` tables, `min_max_pixels=786_432`, `min_scale=0.25`) accelerated `olmOCR-bench(8)` **`1.057×`** at exact score parity (`0.6695 -> 0.6695`, `+0.0000`), sped `OmniDocBench-EN(8)` **`1.140×`** (`-24%` visual tokens, `0.8022 -> 0.8020`, `-0.0002`), and left `FinTabNet-OTSL(6)` untouched (`0.8791 -> 0.8791`, `0/6` shrunk).
- **Windowed 1-Gram Cursor Re-Lock (`OCRPLDDrafter(..., window_relock=True)`, default on):** Scanning a `56`-token forward window around the ANE/PDF cursor at fixed `k=4/6` kept **`10/10` bit-identical** greedy outputs in strictly interleaved `A-B-C-C-B-A` ABBA (`n=5×2`), delivering **`1.022×`** overall decode speedup (`275.4 -> 281.4 tok/s`, `4,756 -> 4,670` forward passes) and **`+11.1%`** on dense `OmniDocBench` prose (`263.8 -> 293.2 tok/s`).
- **Auto Camera-Photo Deskew Gate (`deskew="auto"`, default on):** Pure-NumPy `is_likely_camera_photo()` + `max_area_ratio=0.84` triggered **`0/106` (`0.0%`) false warps** across `olmOCR` digital (`0/30`), `old_scans` (`0/12`), `OmniDocBench` (`0/24`), and `FinTabNet` (`0/40`), while lifting tilted `CORD-v2(8)` receipt transcription (`OvisOCR2` Markdown) by **`+10.94%` leaf recall** (`0.8091 -> 0.9185`, rescuing a `0% -> 100%` receipt) and **`1.637×` wall speedup** (`14.67 -> 8.96 s`, `-31%` visual tokens).
- **Opt-In Prose+Table Region Routing (`--region-routing`, `allow_formula_regions=False`, `min_pixels=4096`):** Routing prose to ANE and aspect-bucketed table crops (capped at `batch_size`) to `OvisOCR2` accelerated `olmOCR table_tests(2)` **`1.84×`** (`15.21 -> 8.25 s`, mean `0.6667 -> 0.7222` with mixed per-page variance `+0.333` / `-0.222`) and sped 2-column `olmOCR` `00f6c2` **`1.83×`** (`11.54 -> 6.32 s`, yet multi-column ANE prose interleaved columns `order_pass=0/3`); remains opt-in (`False` default) and auto-skips formula/math pages *(multi-column & `us_gov_ocr` table regressions fixed alongside `3.468×` digital-PDF splicing in [Section 14.3](#143-ocr-zero-sync-confidence-and-guarded-digital-pdf-splicing))*.
- **Rejected OCR probes (kept out of defaults):** Dynamic AIMD `draft_k >= 8` (`aimd_8_relock` ran **`0.877×` slower**, `275.4 -> 241.4 tok/s`, in interleaved ABBA despite `-10.6%` fewer forward passes because `M=9` crosses MLX's small-`M` tiling knee at `~1.26×` slower/pass), single-page whitespace-valley `B=2..3` strip splitting (**`0.844×` & `-0.0588`** on `olmOCR(4)`, **`0.671×` & `-0.0773`** on `OmniDoc(4)`), `adaptive_20` (`20 px/line` floor dropped `olmOCR(8)` **`-0.0079`**), unfiltered flatbed `deskew=True` on `old_scans` (**`-0.0714`** border clipping around `area ~0.90`), formula pages in `region_routing` (**`-0.29`** `OmniDoc` / **`-0.59`** `arxiv_math`), and math-band zoom recovery (**`+11.5%`** wall time for `+0.0000` pass rate — pruned).

### 13.6 Optional Gemma-4 / EmbeddingGemma-2 Vision & Cross-Modal Retrieval (M4 Pro Verified)

- **Zero text-side overhead (`fast_shared.gemma4_vision`):** Neither `fast_shared.gemma4_vision` nor `PIL` is imported until image inputs are supplied (`_vision_model=None` on text-only calls): `FastEmbeddingGemma2` ran `3.907 ms` p50 query / `419.0 docs/s` (`B=256`, `1.068 GB` MLX / `2.315 GB` RSS) and `FastGemma4Engine` (`E2B-4bit`) ran `63.09 tok/s` (`1.0144 s` median, `3.327 GB` MLX / `4.257 GB` RSS) with zero regression.
- **Stock `mlx-vlm` visual parity & `16`-page Category-5 cross-modal text-to-page retrieval:**

| Engine / Config | Workload (`M4 Pro`, `48 GB`) | Speed / Throughput | Retrieval / Stock Cosine Fidelity | Memory (`MLX Peak` / `RSS`) |
|---|---|---|---|---|
| **`FastEmbeddingGemma2`** (`bf16`, `B=1` / `B=8`) | `8` Category-5 images vs stock `mlx-vlm` | `3.39` (`B=1`, `1.10×`) / `2.91 img/s` (`B=8`, `0.94×`) | `min_cos = 0.999819` (`mean = 0.999899`) | **`1.586 GB`** *(stock `2.241 GB`)* / `3.022 GB` |
| **`FastEmbeddingGemma2`** (`8bit`, `B=1` / `B=8`) | `8` Category-5 images vs stock `mlx-vlm` | `3.44` (`B=1`, `0.99×`) / **`3.51 img/s`** (`B=8`, `1.01×`) | `min_cos = 0.999701` (`mean = 0.999810`) | **`1.586 GB`** *(stock `2.003 GB`)* / `2.945 GB` |
| **`FastEmbeddingGemma2`** (`[vision]`, single-vec) | `16` Category-5 page image retrieval | **`4.36 img/s`** (`3.670 s` img, `84.0 ms` q; **`2.01×`**) | **`15/16` (`93.75%`) `R@1` · `0.950` `MRR`** | **`1.588 GB`** / **`2.982 GB`** |
| **`FastPplx`** (contextual visual multi-vec `MaxSim`) | Same `16` Category-5 pages | `2.17 img/s` (`7.360 s` img, `112.8 ms` q) | `14/16` (`87.50%`) `R@1` · `0.9031` `MRR` | `1.853 GB` / `3.513 GB` |

- **`Gemma-4` Category-5 quick OCR check (`FastGemma4Engine(..., images=...)` on M4 Pro):**

| Checkpoint & Budget | `olmOCR-bench(2)` (`arxiv_math` + `old_scans_math`) | `FinTabNet-OTSL` (`TEDS` / `NumF1`) | `CORD-v2(4)` (`Valid Schema` / `Field F1`) | `OmniDocBench-EN(2)` (`Overall` / `Text Sim`) | Peak Memory (`MLX` / `RSS`) |
|---|---|---|---|---|---|
| **`E2B-4bit` (`max_soft_tokens=280`)** | `0.1667` (`0.098 pp/s`, `60.1 tok/s`) | `0.7665` / `0.9034` (`n=4`, `0.170 pp/s`) | **`1.000`** / `0.2566` (`0.506 pp/s`, `95.9 tok/s`) | **`0.5128`** / `0.7180` (`0.364 pp/s`, `67.1 tok/s`) | `4.28–4.30 GB` / `5.426 GB` |
| **`E2B-4bit` (`max_soft_tokens=1120`)** | `0.1667` (`0.075 pp/s`, `60.8 tok/s`) | **`0.8293`** / **`0.9053`** (`n=2`, `0.096 pp/s`) | — | — | `4.24 GB` / `5.426 GB` |
| **`E4B-bf16` (`max_soft_tokens=280`)** | **`0.5000`** (`3/3` `old_scans_math`, `0.025 pp/s`, `18.7 tok/s`) | `0.3932` / `0.4811` (`n=4`, `0.040 pp/s`) | **`1.000`** / `0.1146` (`0.256 pp/s`, `29.0 tok/s`) | `0.4836` / `0.7172` (`0.098 pp/s`, `17.4 tok/s`) | `14.88–15.05 GB` / `17.047 GB` |

  *(Takeaway: keep `fast-ocr` (`OvisOCR2-8bit`) for dedicated document OCR; use `fast-embeddinggemma-2` `[vision]` for fast cross-modal page retrieval and `fast-gemma4` `[vision]` for lightweight multimodal extraction/VQA.)*

---

## 14. v3.0 Vision Prefetch, NAX Meta-Only Packing, Multimodal MTP, and OCR Confidence (2026-11-23)

Machine-readable validation receipt: [`benchmarks/v30-validation-20261123.json`](benchmarks/v30-validation-20261123.json).

### 14.1 Vision Retrieval, CPU Image Prefetch, and Multimodal MTP

| Engine / Optimization | Workload & Host | Before -> After | Accuracy / Parity |
|---|---|---|---|
| **`FastEmbeddingGemma2`** `DEFAULT_VISION_SOFT_TOKENS = 140` | `48` Category-5 images (`Base M5`) | `100.69 -> 44.87 ms` ViT; **`19.57 img/s`** pre-decoded GPU (**`2.18×`**) / **`15.14 img/s`** disk E2E (**`1.936×`**) | Identical **`R@1 = 0.8542`**, `MRR@10 = 0.8697` |
| **`FastEmbeddingGemma2`** bounded `depth=1` CPU image prefetch (`L-2`) | `48` disk images (`Base M5` / `M4 Pro`) | **Base M5:** `3.2278s (14.87) ->` **`2.5287s (18.98 img/s, 1.2764×)`**, `peak_phys -45.76 MB`<br>**M4 Pro:** `5.1297s (9.36) ->` **`4.4061s (10.89 img/s, 1.1642×)`**, `peak_phys -38.63 MB` | `min_cos = 0.9999999` (M5) / `0.9999993` (M4 Pro), exact `R@1 = 0.8542` |
| **`FastPplxGPU`** `max_pixels = 786_432` default | `164` multi-domain docs (`FinTabNet`, `CORD`, `olmOCR`, `OmniDoc`, `US-Gov`) | `53.617s (3.059) ->` **`25.770s (6.364 img/s, 2.081×)`** | **`0.0000` drop** (`R@1 = 0.9634`, `R@5 = 0.9817`, `MRR@10 = 0.9726`) |
| **`FastPplxGPU`** parallel pre-batch / `depth=1` image prefetch (`L-2`) | `24` disk images (`Base M5` / `M4 Pro`) | **Base M5:** `3.0061s (7.98) ->` **`2.6955s (8.90 img/s, 1.1152×)`**<br>**M4 Pro:** `4.4176s (5.43) ->` **`4.0023s (6.00 img/s, 1.1038×)`** | `min_cos = 0.9999998` (both hosts), preserves exact `grid_thw` grouping |
| **`FastGemma4Engine`** multimodal `MTP` (`run_decode_mtp` on `prefill_multimodal_b1`) | `FinTabNet(4)` / `CORD-v2(4)` JSON | **FinTabNet:** `34.053s (48.81 tok/s) ->` **`16.04s (103.62 tok/s, 2.123×)`**<br>**CORD-v2:** `26.02s ->` **`17.28s (1.505×)`** | **`4/4` exact token & text parity** on both suites; `max_tokens=1024` hits `1.00` valid JSON |

### 14.2 Apple10 NAX Meta-Only Inline XOR, Kev 2-Hit Admission, and OvisOCR2 Tuning

- **Zero-shadow-RAM `metal_mpp_int8` (`PACKS_META_ONLY=True`, `L-5`):** Double-buffered threadgroup inline XOR unpack (`^ 0x80808080u` W8A8 / `^ 0x88888888u` W4A8) keeps only compact `[K//64, N] uint32` `w_meta`, eliminating **`2.341 GB`** on `Qwen3.5-4B-OptiQ`, **`> 4.2 GB`** on `9B-OptiQ`, and `0.403 GB` on `OvisOCR2-8bit` at **`max_diff = 0.0`**, **`cos = 1.0`**, `0.9955×` median vs packed (`1.183×` median vs `MLX qmm`, `12/12` edge + `32/32` post-integration Base M5 matrix cases passing).
- **`Kev` `"auto"` bounded 2-hit `PrefixTree` admission (`admit_prefix(..., admission_min_hits=2)`):** Gates snapshot writes with `count_for_circuit_breaker=False` on filtered 1st-sighting misses, eliminating cold-burst regressions (**`1.0039×` cold**, `0 B` held, breaker closed) while accelerating warm shared passages **`2.394×`** (`0` answer flips).
- **`OvisOCR2` batch KV, wave padding, triage refresh, and tiny tables (`6826e34`):**
  - `resolve_ovis_kv_mode` keeps Apple10 NAX in `bf16` unless extreme pressure (`ctx >= 16384 or B*ctx >= 262_144` on `<= 28 GB`; `69.4 vs 133.7 ms` and `8.664 vs 10.393 GB` vs `q4_quarot`), and was aligned in `5cc9610` so Apple9 also uses pressure-only gating (`ctx >= 8192/16384 or B*ctx >= 262_144/524_288`, removing the blind `B >= 32` trigger).
  - Gating `max_pad_ratio` splitting in `batched_multimodal_generate` strictly to `> 1024`-token prompts keeps `FinTabNet-24` in one wave (**`71.862s` vs `98.643s` = `1.373×`** on Base M5 `0.7295` TEDS; post-build `21.338s` `0.7705` TEDS M4 Pro and `39.551s` `0.7649` TEDS Base M5).
  - Auto-upscaling unpinned `< 150K px` Markdown tables (`min_side >= 80`) toward `262_144` yields **`0.8512` TEDS / `0.8930` NumF1** (`16.059s` M4 Pro) and **`0.8421` TEDS / `0.8915` NumF1** (`27.969s` Base M5) on `FinTabNet(16)`.
  - Refreshing Apple Vision at full target `dpi` on low-DPI rotated/low-conf/`min_side < 160` escalations plus unrotated `aborted_loop` retry cuts worst-case triage recovery **`103.020s -> 27.357s` (`3.77×`)**.

### 14.3 OCR Zero-Sync Confidence and Guarded Digital-PDF Splicing

| Capability (`L-8`, M4 Pro `48 GB`) | Mode & Design | Measured Latency / Speedup | Accuracy / Calibration Receipt |
|---|---|---|---|
| **Zero-sync `logprob_confidence`** (`ovis_engine.py`) | `"auto"` active on `B=1` via async Accelerate `vDSP` + `SimpleQueue` (`B >= 2` opt-in `+14.34%` overhead) | `FinTabNet(30)` `46.81 ->` **`45.87 s` (`-2.002%`)**; `olmOCR(12)` `210.68 ->` **`202.99 s` (`-3.652%`)**; **`0 MB`** extra MLX | **`30/30`** & **`12/12`** exact match; `FinTabNet(30)` AUROC **`0.9135`** (`min_window_logprob`) / **`0.9038`** (`mean_logprob`); `olmOCR` cal trio **`0.7792`** / **`0.7403`** (`bot15=0.0833` vs `top85=0.7715`) |
| **Repaired `region_routing` + Digital-PDF Splicing** (`region_router.py`, `pipeline.py`) | Opt-in (`region_routing=False` default); XY-cut leaf row grouping + multi-col narrative guards + `analyze_digital_pdf_splice` (`min_pixels=4096` table crops into clean prose) | Mixed 3-page digital PDF: `2.7492s ->` **`0.7926s` (`3.468×`)** | Eliminates prior `olmOCR` `table_tests` (`0.8049 ->` **`0.8780`** baseline) & `us_gov_ocr` `table_teds` (`0.2757 ->` **`0.6054`**) regressions; zero 2-col interleaving; lifts `govinfo_uscourts` `+7.67 pp` to **`0.9707`** (`us_gov_overall 0.6008 -> 0.6065`) |

### 14.4 v3.0 Negative and Opt-In Gates

- **`L-3` Global Apple10 `W8A8` Auto-Default (`KILL` -> Opt-in `FAST_SILICON_QWEN3_MPP_W8A8=1`):** `4B-OptiQ` (`1.0414×` EOT, `delta=0.0`, `50/50`) and `OvisOCR2` `olmOCR(42)` (`1.0622×` EOT, `41/42 -> 42/42`) passed, but `Qwen3.5-9B-OptiQ` (`Cat-4`, `n=50`) failed Gate 1 (`delta = -0.004495 < -0.003`, `1.0083×` EOT).
- **`L-4` Native `FP8 e4m3` M5 ViT (`KILL`):** MSL `4.1` `metal::metal_fp8_e4m3_format` (`mx.uint8`) peaked at **`15.70 TFLOPS`** (`1.049×` vs `MLX BF16` `14.97 TFLOPS`), yielding only `0.981–1.058×` on `OvisOCR2` `12L` ViT at `cos 0.9815–0.9926`.
- **`L-5b` Fused Activation-Quantize GEMM Prologue (`KILL`):** Re-quantizing `[TM, 64]` per `N`-tile ran **`0.3055×` / `0.3035×`** at `M >= 128` vs separate `quantize_activations_i8_k64` + inline XOR load.
- **`L-6` Masked Bidirectional `D=512` SDPA (`KILL`):** Even replacing masked `D=512` SDPA with unmasked `mask=None` yielded `<= 1.039×` E2E across `S=1024..2048` on both Macs (`< 1.05×` gate); custom `MPP` (`0.54–1.008×`) and `simdgroup` (`0.05–0.34×`) lost to MLX.
- **`L-7` Multi-Grid CoreML ANE `16L SigLIP` ViT (`KILL`):** Achieved **`100.0%` ANE placement (`1,736/1,736` ops on `CPU_AND_NE`)** at `10.53 img/s`, **`0` `R@1` drop** (`46/48 = 0.9583`), `min_cos = 0.9953` vs `FP32` (`GPU BF16` itself drifts to `0.9926` vs `FP32` on L15 `1,328` residual), and **`1.518–1.608×`** concurrent GPU+ANE work-stealing — **killed on Gate 2 memory (`+690.8 MB` `2`-grid / `+633.6 MB` `3`-grid vs `< +500 MB`)** because CoreML allocates `~250 MB` independent ANE heaps & IOSurfaces per active function even inside a shared-weight `MultiFunctionDescriptor`.
- **`L-8` Auto-DPI Escalation Default (`KILL` -> Opt-in `confidence_escalate_dpi=None`):** Flagged exact `4/30` degraded `old_scans` pages (`mean_logprob < -0.15`), but `300/400 DPI` uncropped retries added `+162.1s..+202.1s` with `+0.0000` accuracy delta.

## 15. Dual-Mac Full 4-Suite Verification and Post-Fix Reruns (2026-11-24)

Machine-readable receipt combining all 4 baseline suites (`monorepo`, `outcomes_tier1`, `outcomes_vision`, `capabilities`) plus targeted post-fix reruns on both hosts: [`benchmarks/dual-mac-suite-20261124.json`](benchmarks/dual-mac-suite-20261124.json).

- **Quiescence & contention disclosures:**
  - The full 4-suite baseline ran quiescent (`0` warnings) on **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** and with `--allow-contended` (`7` warnings) on **Apple M4 Pro (`48 GB`, `macOS 26.6`)**.
  - Targeted post-fix reruns on both Macs executed while an independent background host workload was active on each machine; compare relative ratios (`MTP` vs `AR`, `cached` vs `plain`/`stock`) on reruns rather than raw contended vs quiescent throughput.

### 15.1 Dual-Mac Baseline `benchmark_monorepo.py` Throughput & Peak Memory

| Package | M4 Pro (`48 GB`) Highlight & Internal Ratio | M4 Pro Mem (`MLX / OS GB`) | Base M5 (`24 GB`) Highlight & Internal Ratio | Base M5 Mem (`MLX / OS GB`) |
|---|---|---:|---|---:|
| `fast-bge-small` | `121,730; 192,512; 59,304 tok/s`; B=1 ANE `0.48 ms` vs GPU `1.98 ms` (`4.10×`) | `1.048 / 1.881` | `152,937; 201,181; 63,850 tok/s`; B=1 ANE `0.79 ms` vs GPU `1.29 ms` (`1.62×`) | `0.954 / 1.663` |
| `fast-embeddinggemma-2` | `17,936; 36,809; 21,445 tok/s`; B=1 ANE `2.08 ms` vs GPU `7.72 ms` (`3.72×`) | `1.251 / 2.523` | `30,430; 49,106; 21,888 tok/s`; B=1 ANE `2.58 ms` vs GPU `5.02 ms` (`1.94×`) | `1.128 / 2.321` |
| `fast-pplx` (`hetero2` mixed) | `15,067.6 tok/s` (`805.83 ms`, `1.53×` vs bf16; B=1 ANE `7.53 ms`) | `1.964 / 2.743` | `22,100.1 tok/s` (`549.41 ms`, `1.35×` vs bf16; B=1 ANE `9.12 ms`) | `1.666 / 2.579` |
| `fast-gemma4` | GPU `1,258 tok/s` (`5.64 docs/s`); ANE+GPU `1,182 tok/s` (`6.11 docs/s`, `1.08×`) | `8.480 / 13.045` | GPU `1,289 tok/s` (`6.42 docs/s`); Spec `342 eff tok/s` (ANE disabled on 24 GB) | `4.760 / 5.856` |
| `fast-gpt-oss` | `B=8 190.7`, `B=64 385.8`, `B=128 444.1 dec tok/s` (`1.568 docs/s`) | `15.962 / 16.665` | `B=1 49.0`, `B=8 93.6`, `B=16 103.1 dec tok/s` (`0.382 docs/s`) | `13.422 / 14.265` |
| `fast-decision-models` | `laya_ane 41.96 q/s`, `laya_hetero 110.55 q/s`, `kev_08b 101.5 q/s` | `3.051 / 3.957` | `laya_ane 44.39 q/s`, `laya_hetero 175.79 q/s`, `kev_08b 163.88 q/s` | `2.878 / 3.905` |

### 15.2 Dual-Mac Tier-1 & Vision Outcome Suite Results (with M4 `fast-embeddinggemma-2` Reruns)

| Package (`Config`) | Suite | M4 Pro (`Valid`, `S_obj,e2e`, `EOT u/s`, `4B Mem`, `Gate`) | Base M5 (`Valid`, `S_obj,e2e`, `EOT u/s`, `4B Mem`, `Gate`) |
|---|---|---|---|
| `fast-bge-small` (`hetero`) | NanoBEIR+LegalBench(325) | `325/325`, `0.4737`, `12.059`, `1.564 GB`, `PASS` | `325/325`, `0.4737`, `13.064`, `1.484 GB`, `PASS` |
| `fast-embeddinggemma-2` (`hetero`) | NanoBEIR+LegalBench(500) | **`500/500`, `0.5359`, `1.143` (`234.5 s` contended rerun), `1.287 GB`, `PASS`** | `500/500`, `0.5371`, `4.658` (`57.7 s`), `2.081 GB`, `PASS` |
| `fast-pplx` (`hetero2`) | NanoBEIR+LegalBench(325) | `325/325`, `0.7370`, `1.515`, `2.129 GB`, `PASS` | `325/325`, `0.7377`, `2.059`, `1.734 GB`, `PASS` |
| `fast-decision-models` (`laya_ane`) | RAGTruth+CLINC+RB2(1400) | `1400/1400`, `0.5050`, `5.817`, `1.010 GB`, `PASS` | `1400/1400`, `0.5050`, `6.158`, `1.011 GB`, `PASS` |
| `fast-decision-models` (`kev_08b_w8`) | RAGTruth+CLINC+RB2(1400) | `1400/1400`, `0.5257`, `2.757`, `2.881 GB`, `PASS` | `1400/1400`, `0.5250`, `4.760`, `2.263 GB`, `PASS` |
| `fast-gemma4` Cat 3 (`gpu_fp16_kvbf16_cat3`) | IFBench+MMLU+MATH(435) | `303/435`, `0.3632`, `1.260`, `7.141 GB`, `FAIL` (repeat loops) | `303/435`, `0.3632`, `1.003`, `4.045 GB`, `FAIL` (repeat loops) |
| `fast-gemma4` Cat 4 (`llg_on`) | LiquidAI+BFCL(300) | `300/300`, `0.8802`, `3.639`, `8.153 GB`, `PASS` | `300/300`, `0.8836`, `4.179`, `4.978 GB`, `PASS` |
| `fast-qwen3` Cat 3 (`qwen3_auto_cat3`) | IFBench+MMLU+MATH(435) | `398/435`, `0.6621`, `0.601`, `4.109 GB`, `FAIL` (repeat loops) | `394/435`, `0.6460`, `0.391`, `4.212 GB`, `FAIL` (repeat loops) |
| `fast-qwen3` Cat 4 (`llg_on`) | LiquidAI+BFCL(300) | `299/300`, `0.8929`, `0.608`, `4.602 GB`, `FAIL` | `300/300`, `0.8930`, `0.804`, `4.753 GB`, `FAIL` |
| `fast-gpt-oss` Cat 3 / Cat 4 | Cat 3(435) / Cat 4(300) | Cat 3 `411/435`, `0.5885`, `0.409`, `12.041 GB`, `FAIL` / Cat 4 `300/300`, `0.8287`, `0.446`, `12.670 GB`, `PASS` | Excluded (`<= 24 GB` RAM policy) |
| `fast-ocr` (`auto_gfm`) | olmOCR(42) / FinTabNet(30) / OmniDoc(24) | `0.4790` `PASS` / `0.8378` `PASS` / `0.5805` `FAIL` (`5.61–5.95 GB`) | `0.3994` `PASS` / `0.8599` `PASS` / `0.5820` `FAIL` (`4.70–5.80 GB`) |
| `fast-embeddinggemma-2-vision` | VisualRetrieval(18) | **`18/18`, `0.9722`, `3.492` (`5.01 s` contended rerun), `4.656 GB`, `PASS`** | `18/18`, `1.0000`, `16.042` (`1.12 s`), `4.810 GB`, `PASS` |
| `fast-pplx-vision` | VisualRetrieval(18) | `18/18`, `1.0000`, `6.140` (`2.93 s`), `4.122 GB`, `PASS` | `18/18`, `1.0000`, `8.872` (`2.03 s`), `3.802 GB`, `PASS` |
| `fast-gemma4-vision` (`gemma4_vision_gfm`) | FinTabNet(30) / olmOCR(42) / OmniDoc(24) / CORD(30) | FinTabNet `0.5798` `PASS`; olmOCR `0.1736` / OmniDoc `0.3362` / CORD `0.5434` (`FAIL` loops) | FinTabNet `0.5800` `PASS`; olmOCR `0.1864` / OmniDoc `0.3387` / CORD `0.5446` (`FAIL` loops) |

### 15.3 Targeted Post-Fix Capability Reruns & Root-Cause Engineering Notes

1. **`fast-embeddinggemma-2` M4 Pro Core ML `e5bundlecache` repair & worker IPC hardening:** Clearing `30` incomplete `H16S.bundle` directories (missing `weights/weights.bin`) and `2` missing-`model.milhash` directories on M4 Pro restored all 9 `emb2_pk_s*_w8` buckets (`rc=0, ANE=2288, GPU=0, CPU=0`). Hardening `scripts/benchmark_outcomes.py` with atomic `--worker-out-file` IPC plus substring `_WORKER_MARKER` matching prevents non-newline native stdout/stderr text from discarding worker JSON records.
2. **`KevMlxBackend` `profile="interactive"` 1-hit `PrefixTree` admission (`prefix-tree` Base M5 rerun):** Setting `admission_min_hits=1` for `profile="interactive"` (retaining `2` for `"auto"`/`"backfill"`) and passing `profile="interactive"` in `scripts/_kev_benchmark.py` eliminated the 1-shot cold-to-warm `0.051327` diff. Base M5 rerun passed cleanly (`status="ok"`, **`0.0` `cold_vs_warm_max_logit_diff` & `max_prob_diff`**, warm `B=1 5.63 q/s` / `B=8 9.59 q/s`, **`3.039×` vs stock `mlx-lm 3.15 q/s`**, `1.537×` vs plain, `2.910 GB` peak). Baseline M5 `encoder-finetune` (`33.1 items/s`, `1.31×`), `grpo-ane-critic` (`2.24/3.15 rollouts/s`, `4.48×`), and `parallel-coreml` (`9.25s`, `1.71×`) already passed (`M4 Pro`: `24.2 items/s` `1.28×`, `2.26/3.49 rollouts/s`, `12.37s` `1.78×`).
3. **`fast-gemma4` `mtp-speculative` `B >= 2` singleton gate & declined-wave rolling retention:** Gating `_singleton_verify_linear` / `singleton=True` inside `forward_verify_mtp_slot` strictly to `B == 1` eliminated `B=4` Metal register spills, and keeping explicit-MTP engines on dense `run_decode_rolling` when MTP declines `B > max_b` prevented unconstrained `PromptLookup` fallback:
   - **M4 Pro (1-round contended probe):** `free_b1` **`1.617×`** vs AR, `free_b4` **`0.185× -> 1.130×`** vs AR, `json_b1` **`1.488×`** vs AR, `json_b4` `0.920×` (`9.532 GB` peak), **`10/10` exact AR/MTP token matches** (up from `4/10`).
   - **Base M5 (2-round contended rerun with stock `mlx-vlm`):** `free_b1` **`1.677×`** vs AR, `free_b4` **`0.560× -> 0.967×`** vs AR, `json_b1` **`1.648×`** vs AR (`33.5 vs 15.6 tok/s` unrestricted `mlx-vlm` = **`2.15×`**), `json_b4` **`0.861× -> 1.002×`** vs AR (`59.7 vs 45.5 tok/s` unrestricted `mlx-vlm`), `9.300 GB` peak, **`10/10` exact AR/MTP token matches** (up from `5/10`).
4. **`setup_all.py` offline `qwen3` readiness & Category 3/4 repeat-loop probe:** Updating `_plan_qwen3` in `scripts/setup_all.py` to check `Qwen3Paths.is_complete_snapshot` made `setup_all.py --dry-run` report **`[READY]` across all 7 packages (`0.00 GB` pending)** on both Macs. Probing the `fast-gemma4` `math_500` first 16 items found the flagged item (`rep4=0.194`) finished cleanly with `finish_reason="stop"` and `exact_3x_periods=[]` (algebraic phrase reuse over 1,330 valid tokens rather than a cyclic loop), and a batched `fast-qwen3` `bfcl_v3` probe (`150` items) had `0` `length` truncations and `0` `rep4 >= 0.15` flags, so `compute_loop_breaker_penalties` remains unchanged.

## 16. Validated Engine, Default & Laya FP16 LayerNorm Fixes (2026-11-24)

Machine-readable receipt covering the dual-Mac Wave 3 post-fix verification runs, 6-shape Laya matrix, and Laya Core ML FP16 LayerNorm root-cause repair: [`benchmarks/rca-and-laya-fixes-20261124.json`](benchmarks/rca-and-laya-fixes-20261124.json).

- **Quiescence & contention disclosures:**
  - All Wave 3 **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** runs executed **quiescent** (zero system quiescence warnings).
  - All Wave 3 **Apple M4 Pro (`48 GB`, `macOS 26.6`)** runs executed with `--allow-contended` alongside active desktop/compositor/VM background load; compare internal backend/baseline deltas rather than raw wall-clock speeds across contended vs quiescent hosts.

### 16.1 Laya-Typed-421M 6-Bucket Multifunction Core ML & Two-Pass FP16 LayerNorm Repair

Restoring explicit two-pass centered `LayerNorm` (`reduce_mean -> sub -> mul(ln_gain) -> sq -> reduce_mean -> add(eps_u) -> rsqrt -> mul`) across all 6 multifunction buckets (`s64, s128, s192, s256, s384, s512`) while fixing four live-activation numerical failure modes achieved **`[10538, 0, 0]` (`100%` `CPU_AND_NE`)** placement and **`711/1400` (`0.5079` `S_obj,e2e`, `PASS`)** on **both** Macs (up from `707/1400` / `0.5050` baseline, and recovering from the `697/1400` regression):

- **`tok_emb` (`base_inv_scale 1.0 -> 4.0`):** `var_min=0.0007` suffered `+8.7%` `FP16_MIN_NORMAL` (`6.1e-5`) variance inflation; scaling by `4.0` lifts `var(u) >= 0.0112` (`u^2 <= 58`).
- **`l14..l18` (`s_eff_floor 1/256 -> 1/128`, `ln_gain=2.0`):** Real `RewardBench-2` activations (`max|xc|=11,255.2`, `var_min=7.11..14.39`) dropped to `var(u)=1.08e-4` next to `eps_u=6.1e-5` (`+56%` variance inflation across 10 consecutive LNs, dropping `RB2` `75/250 -> 63/250`); `1/128` keeps `var(u) >= 4.34e-4` with `max u^2 = 7,733 << 65,504`.
- **`l19..l27` (`max_ln_gain=4.0`):** Caps `s_eff=1/256` on late layers (`max|xc|=31,851`, `u^2 <= 15,480 << 65,504`).
- **Head/scorer LNs (`hl0_n1`, `hl0_n2`, `hl1_n1`, `hl1_n2`, `sc_ln`: `base_inv_scale 0.25 -> 0.0625`):** Eliminated `u^2 = 67,379..113,232 > 65,504` FP16 `INF` overflow (`max u^2 = 7,077`).
- **Pass removal after `cto.linear_quantize_weights` + split-head `mb.einsum`:** Removed `common::fuse_elementwise_to_batchnorm`, `common::add_fp16_cast`, and `common::fuse_layernorm_or_instancenorm` after weight quantization so `0` `mb.layer_norm` ops remain in `model.mil`, and retained split-head `mb.einsum` (`1.44×` faster than `mb.matmul` on Base M5 `RB2`: `37.33 s` vs `53.41 s`, `75/250` `0.3000`, `max_diff=0.07494` vs `gpu_bf16`).
- **`Kev-0.8B` LCP threshold & length-sorted candidate tail chunking (`_MIN_LCP_GROUP_TOKENS = 128`):** Raising the shared-prefix threshold from `32` to `128` keeps short 49-token `RewardBench-2` prompt overlaps on the 1-pass path instead of right-padding unsorted candidate tails (`-30.4 s` on `RAGTruth + RB2`), and sorting longer tails deterministically before batching by `eff_b` lifts quiescent Base M5 Tier-1 from `150.67 s` (`4.885` EOT) to **`122.83 s` (`6.000` EOT, `11.40 q/s`, `0.5264`, `PASS` — `1.18×` faster than Stock `mlx-lm` `145.5 s` / `5.039` EOT)** and lifts `prefix-tree` `B=8` warm throughput to **`27.65 q/s` (`2.971×` vs. Stock `mlx-lm` `9.31 q/s`, `0.0` logit/prob diff)**.

| Model & Backend (`Tier-1`, `1,400` items) | M4 Pro (`--allow-contended`) `S_obj,e2e` / `EOT` / Wall / Mem | Base M5 (quiescent) `S_obj,e2e` / `EOT` / Wall / Mem | Notes |
|---|---|---|---|
| `fast-decision-models` (`laya-ane`, 6-bucket pre-warmed) | **`0.5079` (`711/1400`)**, `5.341`, `133.12 s`, `1.516 GB`, `PASS` | **`0.5079` (`711/1400`)**, **`9.266`**, **`76.73 s`**, `1.593 GB`, `PASS` | `1.50×` faster than Nov 20 M5 `laya-ane` (`114.8 s`) and **`1.15×` faster than NMTZ Core ML (`88.4 s`)** |
| `fast-decision-models` (`laya-hetero`, ANE + compiled MLX GPU) | **`0.5050` (`707/1400`)**, **`11.932`**, **`59.25 s`**, `3.295 GB`, `PASS` | `0.5029` (`704/1400`), **`18.586`**, **`37.88 s`**, `3.722 GB`, `PASS` | `compile_encode=True`, makespan split, `n=1` zero-thread fast path; **`3.03×` faster than Nov 20 M5 `laya-ane`** |
| `fast-decision-models` (`kev-0.8b`, `kev_mlx` LCP 128 + sorted tails) | **`0.5250`**, `3.008`, `244.34 s` (`5.73 q/s`), `3.265 GB`, `PASS` | **`0.5264` (`737/1400`)**, **`6.000`**, **`122.83 s` (`11.40 q/s`)**, `2.200 GB`, `PASS` | **Base M5 `1.18×` vs. Stock `mlx-lm` (`145.5 s`, `5.039` EOT)**; Wave 3 pre-fix was `150.67 s` (`4.885` EOT, `0.5257`); `prefix-tree` `27.65 q/s` (`2.971×` Stock) |

| Laya 6-Shape Matrix (`ms` / `q/s`) | M4 Pro ANE | M4 Pro GPU | M4 Pro Hetero | Base M5 ANE | Base M5 GPU | Base M5 Hetero |
|---|---:|---:|---:|---:|---:|---:|
| `1x_s64` | **`5.11` (`195.7`)** | `8.87` (`112.7`) | `5.33` (`187.7`) | **`6.53` (`153.1`)** | `10.30` (`97.1`) | `6.54` (`152.9`) |
| `1x_short` | **`5.09` (`196.6`)** | `12.80` (`78.2`) | `5.39` (`185.5`) | **`6.53` (`153.1`)** | `11.97` (`83.5`) | `6.53` (`153.2`) |
| `8x_512` | `377.09` (`21.2`) | `679.27` (`11.8`) | **`284.21` (`28.2`)** | `342.53` (`23.4`) | `263.37` (`30.4`) | **`172.71` (`46.3`)** |
| `64x_short` | **`325.34` (`196.7`)** | `487.51` (`131.3`) | `331.27` (`193.2`) | `411.94` (`155.4`) | `210.78` (`303.6`) | **`200.98` (`318.4`)** |
| `64x_mixed` | **`1028.97` (`62.2`)** | `1939.23` (`33.0`) | `1244.54` (`51.4`) | `1008.38` (`63.5`) | `783.95` (`81.6`) | **`578.90` (`110.6`)** |

### 16.2 Post-Fix `fast-qwen3`, `fast-ocr`, `fast-embeddinggemma-2` Vision & `fast-pplx` Results

| Package & Workload | M4 Pro (`--allow-contended`) | Base M5 (quiescent) | Validation Delta / Notes |
|---|---|---|---|
| `fast-qwen3` Cat 4 (`LiquidAI 150 + BFCL v3 150`) | `299/300`, `0.8930` (`F1=0.9526`, `BFCL=0.8333`), `0.676` EOT, `396.52 s`, `4.990 GB`, `2` loops | **`300/300`**, **`0.8955`** (`F1=0.9510`, `BFCL=0.8400`), **`0.881` EOT**, `305.08 s`, `4.760 GB`, **`0` loops (`PASS`)** | Removing blind `bs >= 32` `q4_quarot` eliminated `BFCL[116]` looping and flipped Base M5 Cat 4 from `FAIL` to `300/300 PASS` |
| `fast-qwen3` Cat 3 (`MMLU 285 + IFBench 100 + MATH-500 50`) | `396/435`, `0.6368`, `0.715` EOT, `387.28 s`, `4.262 GB`, `12` loops (`MMLU 285/285 0.7860 0 loops`, `IFBench 72/100 7 loops`, `MATH 39/50 5 loops`) | `394/435`, `0.6598`, **`0.562` EOT** (**`1.44×`** vs `0.391`), `510.44 s`, `4.206 GB`, `6` loops (`MMLU 285/285 0.7895 0 loops`, `MATH 40/50 0.7200 0 loops`, `IFBench 69/100 6 loops`) | Lazy GPU argmax + `make_cache(batch_size)` lifted Base M5 Cat 3 EOT `1.44×` and cleared `MATH-500` loops (`5 -> 0`) |
| `fast-ocr` `olmOCR-bench(42)` | **`42/42`**, **`0.6155`**, `0.0689` EOT, `375.43 s`, `5.624 GB`, **`PASS` (`0` loops)** | `41/42`, **`0.6128`**, `0.0626` EOT, `411.09 s`, `5.445 GB`, `1` loop | Triage + `_resolve_ovis_min_pixels` lifted `olmOCR` **`+13.65 pp`** (M4 Pro `0.4790 -> 0.6155`) and **`+21.34 pp`** (Base M5 `0.3994 -> 0.6128`) |
| `fast-ocr` `FinTabNet-OTSL(30)` | **`30/30`**, **`0.8935`** (`NumF1=0.8829`), **`1.0189` EOT**, `26.31 s`, `5.313 GB`, **`PASS`** | **`30/30`**, **`0.8867`** (`NumF1=0.8808`), `0.9303` EOT, `28.60 s`, `4.002 GB`, **`PASS`** | **`+5.57 pp`** M4 Pro (`0.8378 -> 0.8935`) and **`+2.68 pp`** Base M5 (`0.8599 -> 0.8867`) |
| `fast-ocr` `OmniDocBench-EN(24)` | **`22/24`**, **`0.6574`**, `0.0895` EOT, `176.35 s`, `5.735 GB`, **`PASS` (`0` loops)** | **`22/24`**, **`0.6872`**, `0.0899` EOT, `183.52 s`, `5.825 GB`, **`PASS` (`0` loops)** | Flipped **both** Macs from `FAIL` (`0.5805` / `0.5820`) to **`PASS`** (**`+7.69 pp`** M4 Pro / **`+10.52 pp`** Base M5); all 3 OCR suites `PASS` on M4 Pro |
| `fast-embeddinggemma-2` Vision `VisualRetrieval(18)` | `18/18`, `0.9722`, **`9.909` EOT**, `1.77 s` (`11.63 img/s`), `3.532 GB` (`neural=0.0`), **`PASS`** | **`18/18`**, **`1.0000`**, **`11.207` EOT**, `1.61 s` (`15.06 img/s`), `3.558 GB` (`neural=0.0`), **`PASS`** | `enable_ane=None` avoids loading ANE text buckets on vision runs (`0.0 GB` ANE footprint; `1.77 s` vs `5.01 s` M4 Pro contended) |
| `fast-pplx` `hetero2` vs `gpu` (`benchmark.py`) | `B=1` `6.50 ms` (`0.89×`), `B=32` `9,249 tok/s` (`1.34×`), `S=256` `21,345` (`1.81×`), `S=512` `19,060` (`1.62×`), `mixed` `15,212` (`1.47×`) | `B=1` **`5.80 ms` (`1.22×`)**, `B=32` `9,098` (`0.85×`), `S=256` **`27,234` (`1.41×`)**, `S=512` **`28,415` (`1.46×`)**, `mixed` **`22,273` (`1.34×`)** | Apple9 (`B=1` GPU default) + Base M5 `hetero2` (`5.80 ms` `B=1` / `55.5 docs/s` `S=512` / `545 ms` `mixed`) |

## 17. Wave-5 DRY Optimizations & Dual-Mac ABBA Gates (`fast-qwen3` & `fast-pplx`, 2026-11-24)

Shared DRY primitives extracted into `fast-shared`: `detect_catastrophic_token_loop` (`fast_shared/grammar.py`), cursor-tracked `PromptLookupDrafter` + `filter_drafts_with_grammar` (`fast_shared/pld_drafter.py`), and generic Apple10 NAX `W4A8`/`W8A8` wrappers (`fast_shared/mpp_linear.py`). (`fast-embeddinggemma-2` was audited and kept closed — already at hardware ceiling at `4.33–4.80×` vs. stock MLX.)

| Track & Package | Default Disposition | Base M5 (`24 GB`, `Apple10 NAX`) ABBA | M4 Pro (`48 GB`, `Apple9`) ABBA | Notes & Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **`P1` (`fast-pplx`)**: Apple10 inline GPU tiny query batching | **Promoted default** on Apple10 `hetero2` | `B=32, S=15`: **`0.0383 s`** (`gpu=32`) vs. `0.0876 s` ANE (**`2.289×`**, `min_cos = 0.999623`) | Stays `hetero2`: `0.0518 s` (`ane=15, gpu=17`, `min_cos = 0.999619`) | Routes non-contextual tiny batches (`max_len <= 32`, `total_tokens <= 1024`) inline on Metal GPU on Apple10 (`M5+`). |
| **`P2` (`fast-pplx`)**: Apple10 NAX `W8A8` (`metal_mpp_int8`) | **Opt-in under `quantize_weights="w8"`** (`"none"` default) | Beats `w8_nompp` **`1.067–1.078×`** (`S256_B8`: `0.1218` vs. `0.1313 s`; `S512_B8`: `0.2432` vs. `0.2620 s`; `S2048_B4`: `0.3208` vs. `0.3423 s`) and edges `fp16` (`0.1232 / 0.2466 / 0.3243 s`); `min_cos = 0.9984–0.9993` | Native `qmm` (`Apple9`) | `M >= 128`, `min >= 1024`, `max >= 3584` (`9.75 MB` packed metadata); `FAST_SILICON_PPLX_MPP=0` kill-switch. |
| **`Q1` (`fast-qwen3`)**: Hygiene bundle | **Promoted default** | `admission_min_hits=2` (`0` cold snapshots, `circuit_breaker_open=False`), Apple10 KV cutoffs `16384 / 32768`, early loop abort, `mx.async_eval` on pure-FF steps | Identical 2-hit admission (`8192 / 16384` KV cutoffs), early loop abort, `mx.async_eval` | Eliminates cold `PrefixTree` thrash and sync overhead on forced tokens. |
| **`Q2` (`fast-qwen3`)**: Grammar-checked `PromptLookupDrafter` | **Opt-in `speculative=True`** under `json_schema` | `100%` token-for-token parity; short JSON ran `0.843×` on `4B` (`5.88 s` vs. `4.95 s`) | `100%` parity; `0.588×` on `4B` (`7.64 s` vs. `4.49 s`), `0.781×` on `9B` `MTP` (`7.72 s` vs. `6.03 s`) | `upfront_ff` already jumps JSON keys; `speculative=None` (`"auto"`) retains the `matcher is None` guard. |
| **`Q3` (`fast-qwen3`)**: Rolling continuous batching (`HybridCache`) | **Opt-in `rolling=True`** (`_ENABLE_ROLLING_AUTO=False`) | `100%` parity (`32/32`); `N=32, B=16` vs. static `B=8` ran `0.809×` (`11.34 s` vs. `9.18 s`, `6.83` vs. `5.17 GB` MLX peak) | `100%` parity (`48/48`); `B=32` sweep peaks at `592.70 tok/s` (`1.751 docs/s`, `11.36 GB` peak phys vs. `B=64` `495.51 tok/s`, `18.11 GB`); `N=48, B=32` hits **`1.021×`** (`24.15 s` vs. `26.04/23.51 s`, **`-2.4 GB` MLX peak** `9.8` vs. `12.2 GB`) | `RollingHybridCache` + `refill_min = B_slots // 4` scheduler (`rolling.py`). |

## 18. Quiescent Base M5 Full 4-Suite Headline Verification (`d9f65a3`, 2026-11-26)

Machine-readable receipt for all 4 quiescent Base Apple M5 (`24 GB`, `macOS 27.2.1`) suites at `d9f65a3` (`monorepo` `8/8 ok`, `outcomes_tier1` `12/12 PASS`, `outcomes_vision` `6/6 PASS`, `capabilities` `6/6 ok`; `32/32` total with `rc=0` on every suite): [`benchmarks/m5-suite-20261126.json`](benchmarks/m5-suite-20261126.json).

- **Contention gate:** Every suite verified `QUIESCENT:` immediately prior to launch (`--allow-contended` was never passed). M4 Pro remained contended during this window (`rustc`, `Virtualization`, `mdworker_s`) and keeps its prior verified reference numbers in the cross-silicon comparison tables above.

### 18.1 Monorepo (`8/8 ok`) & Capabilities (`6/6 ok`)

| Package / Capability Suite | Quiescent Base M5 (`d9f65a3`) | Internal / Stock Baseline | Peak Mem (GB) | Status |
|---|---|---|---:|---|
| `fast-bge-small` | `172,935; 218,860; 63,871 tok/s`; `B=1` ANE `0.79 ms` vs GPU `1.26 ms` (`1.59x`) | `fast-bge-small` Tier-1 `12.17x` vs stock MLX (`10.99 s` vs `133.8 s`) | `0.954` MLX / `1.683` OS | `ok` |
| `fast-embeddinggemma-2` | `31,628; 51,495; 21,901 tok/s`; `B=1` ANE `2.58 ms` vs GPU `4.97 ms` (`1.92x`) | `4.33x` vs stock MLX (`500`q `57.1 s`) | `1.104` MLX / `2.268` OS | `ok` |
| `fast-pplx` | `hetero2` `B=8 S=512`: `29,905.1 tok/s` (`17.12 ms`); `B=1` ANE `9.11 ms` | Mixed `1.28x` vs `bf16`; `mmap` `-0.506 GB` | `1.737` MLX / `2.574` OS | `ok` |
| `fast-gemma4` | GPU `1,291 tok/s` (`6.44 docs/s`); Spec `342 eff tok/s` | `kv_i8` `-0.67 GB` | `5.026` MLX / `5.817` OS | `ok` |
| `fast-qwen3` | `B=16` structured `161.2 tok/s` (`2.184 s`) | `100%` valid Cat-4 | `4.724` MLX / `5.392` OS | `ok` |
| `fast-gpt-oss` | `B=1` `50.0`, `B=8` `129.8`, `B=16` `168.2 dec tok/s` (`0.742 docs/s`) | `3.35x` tok/s (`2.67x` docs/s) `B=16` vs `B=1` | `12.897` MLX / `13.779` OS | `ok` |
| `fast-ocr` | `verify 3/3 ok` (`41/42` olmOCR, `30/30` FinTab, `22/24` Omni) | `1.87x` vs stock forced VLM (4-page) | `0.000` / `0.064` OS | `ok` |
| `fast-decision-models` | `laya_ane 119.13 q/s` (`8.39 ms/q`), `laya_hetero 228.61 q/s` (`4.37 ms/q`), `kev_08b 94.8 q/s` | `2.0x` MLX / `25x` CoreAI | `2.893` MLX / `3.679` OS | `ok` |
| `qlora-packing` | Packed `QLoRA (1024)` `101.3` sup tok/s (`ckpt=1024`, `1.0` parity); shared-sys `1,097.9` vs `727.1` log tok/s (`1.51x`, `7.26 GB`) | Unpacked `109.8` sup tok/s (`0.92x` on `929`-tok singletons) | `9.638` | `ok` |
| `encoder-finetune` | `Packed-512` `33.3 items/s` (`42.4 items/s` cached `L0..17`; FP32 cos `0.9962`, BF16 cos `0.9734`) | Stock unpacked `25.3 items/s` (`1.32x`) | `5.640` | `ok` |
| `grpo-ane-critic` | `2.26 rollouts/s` GRPO / `3.53 rollouts/s` BoN | `mlx-vlm` + Kev-8bit `0.50 rollouts/s` (`4.50x`) | `5.628` | `ok` |
| `mtp-speculative` | Constrained JSON `B=1 117.8 tok/s` (`1.84x` AR), `B=4 196.7 tok/s`, `10/10` exact | Unrestricted `mlx-vlm` `B=1 50.6`, `B=4 160.7` (`2.33x`) | `5.098` | `ok` |
| `prefix-tree` | Warm `B=1 18.39 q/s`, `B=8 27.67 q/s` | Stock `mlx-lm` `9.30 q/s` (`2.98x`) | `2.909` | `ok` |
| `parallel-coreml` | `9` BGE-small buckets in `9.31 s` (`jobs=4`) | Sequential `coremltools` `15.96 s` (`1.71x`) | `0.154` | `ok` |

### 18.2 Tier-1 & Vision Outcome Suites (`18/18 PASS`)

| Package (`Config`) | Benchmark (`N`) | Valid / Intended | `S_obj,e2e` | `EOT` (`u/s`) | Wall (`s`) | 4B Mem (`GB`) | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `fast-bge-small` (`hetero`) | NanoBEIR+LegalBench (`325`) | `325/325` | `0.4737` | `14.012` | `10.99` | `1.063` | `PASS` |
| `fast-embeddinggemma-2` (`hetero`) | NanoBEIR+LegalBench (`325`) | `325/325` | `0.5182` | `3.567` | `47.21` | `2.739` | `PASS` |
| `fast-pplx` (`hetero2`) | NanoBEIR+LegalBench (`325`) | `325/325` | `0.5297` | `1.802` | `95.51` | `2.289` | `PASS` |
| `fast-decision-models` (`laya_ane`) | RAGTruth+CLINC+RB2 (`1400`) | `1400/1400` | `0.5079` | `9.235` | `76.99` | `1.580` | `PASS` |
| `fast-decision-models` (`kev_08b_w8`) | RAGTruth+CLINC+RB2 (`1400`) | `1400/1400` | `0.5264` | `5.952` | `123.82` | `2.561` | `PASS` |
| `fast-gemma4` Cat 3 (`gpu_fp16_kvbf16_cat3`) | IFBench+MMLU+MATH (`435`) | `310/435` | `0.3655` | `1.008` | `157.77` | `3.931` | `PASS` |
| `fast-gemma4` Cat 4 (`llg_on`) | LiquidAI+BFCL (`300`) | `300/300` | `0.8836` | `4.174` | `63.51` | `4.982` | `PASS` |
| `fast-qwen3` Cat 3 (`qwen3_auto_cat3`) | IFBench+MMLU+MATH (`435`) | `394/435` | `0.6414` | `0.552` | `505.13` | `4.211` | `PASS` |
| `fast-qwen3` Cat 4 (`llg_on`) | LiquidAI+BFCL (`300`) | `300/300` | `0.8925` | `0.886` | `302.20` | `4.722` | `PASS` |
| `fast-ocr` (`auto_gfm`) | olmOCR (`42`) | `41/42` | `0.6128` | `0.064` | `405.09` | `5.104` | `PASS` |
| `fast-ocr` (`auto_gfm`) | FinTabNet-OTSL (`30`) | `30/30` | `0.8784` | `0.849` | `31.05` | `3.280` | `PASS` |
| `fast-ocr` (`auto_gfm`) | OmniDocBench-EN (`24`) | `22/24` | `0.6870` | `0.093` | `176.57` | `5.337` | `PASS` |
| `fast-gemma4-vision` (`gemma4_vision_gfm`) | olmOCR (`42`) | `31/42` | `0.1864` | `0.028` | `277.19` | `6.712` | `PASS` |
| `fast-gemma4-vision` (`gemma4_vision_gfm`) | FinTabNet-OTSL (`30`) | `20/30` | `0.5800` | `0.299` | `58.25` | `6.709` | `PASS` |
| `fast-gemma4-vision` (`gemma4_vision_gfm`) | OmniDocBench-EN (`24`) | `21/24` | `0.3303` | `0.066` | `120.81` | `6.860` | `PASS` |
| `fast-gemma4-vision` (`gemma4_vision_gfm`) | CORD-v2 (`30`) | `30/30` | `0.5446` | `0.229` | `71.36` | `6.639` | `PASS` |
| `fast-embeddinggemma-2-vision` (`gpu_fp16`) | VisualRetrieval (`18`) | `18/18` | `1.0000` | `16.178` | `1.11` | `3.547` | `PASS` |
| `fast-pplx-vision` (`c8_fp16_crop`) | VisualRetrieval (`18`) | `18/18` | `1.0000` | `9.050` | `1.99` | `3.951` | `PASS` |

### 18.3 Quiescent Base M5 Accuracy Push (2026-11-26)

Machine-readable receipt combining all follow-up quiescent Base M5 (`24 GB`, `macOS 27.2.1`) paired accuracy runs on top of `9ac1551`: [`benchmarks/m5-accuracy-20261126.json`](benchmarks/m5-accuracy-20261126.json).

| Package / Track | Configuration | Measured Base M5 Accuracy Gain | Latency / Memory & Scope Notes |
|---|---|---|---|
| **`fast-decision-models`** | 5-fold OOF Platt/Temp + OOS threshold alignment + 2/3-tier `DecisionCascadeScorer` (`--backend laya-kev-cascade`, `--calibrate-oof`) | `RAGTruth` ECE `0.1188 ->` **`0.0701`** (`Brier 0.2251 -> 0.2121`); `CLINC150` Laya NLL `3.8444 ->` **`1.4229`** (`oos_f1 0.3151 -> 0.4437`); held-out `LayaHetero -> Kev-0.8B` (`tau=0.55`): **`oos_auroc 0.8370`**, **`oos_f1 0.6259`** (`+30.4 pp` vs raw Laya `0.322`) | `5.525 s` held-out cascade eval (`AbstractFramework/kev-0.8b-mlx-8bit`); `configs/outcome_benchmarks.lock.json` pins test splits only, so leakage-free calibration + cascade replaces fine-tuning. |
| **`fast-shared`** (`HybridRetrievalPipeline`) | `CompactRetrievalIndex ->` ANE `LayaRerankFusion` (`dense_rerank`, top-20) & calibrated BM25 (`bm25_weight=0.25` + `ENGLISH_STOPWORDS`) | `LegalBench-RAG(175)` `dense_rerank` `span_nDCG@10` `0.2665 ->` **`0.3715`** (**`+0.1049`**, `CI95 [+0.0768, +0.1331]`), `char_recall@10` `0.5468 ->` **`0.5847`**; `NanoArguAna` `R@10` `0.98 ->` **`1.00`** (`0.6357` vs `0.6390`), `NanoHotpotQA` `0.8829 ->` **`0.8872`** | `2.403 GB` peak (`1.249 GB` phys), `486.8 ms/q` ANE rerank; equal-weight `bm25_weight=1.0` hurt dense retrieval, and ANE `Laya` rerank hurt `NanoSciFact` (`0.890 -> 0.700`), so `Laya` reranking stays scoped to structured/legal clauses. |
| **`fast-ocr`** (`OvisOCR2-8bit` defaults) | Default Tier-2 `strip_running_headers_footers` + gated `normalize_faded_scan_image` | `olmOCR-bench(42)` `0.6128 ->` **`0.6402`** e2e / `0.6505 ->` **`0.6918`** macro (`headers_footers` `4/12 ->` **`7/12`**, `long_tiny_text` `64/77 ->` **`67/77`**); `OmniDocBench-EN(24)` `0.6870 ->` **`0.6910`**; `FinTabNet-OTSL(30)` exact parity (`0.8784` TEDS / `0.8775` NumF1) | `0.000` loops on `olmOCR`; remaining `old_scans` misses (`8/26`) are 1860s cursive handwriting. |
| **`fast-ocr`** (opt-in `v7final` + `--ocr-max-tokens 8192`) | `FAST_OCR_GFM_MODEL_DIR` / `--ocr-model v7final` | `OmniDocBench-EN(24)` `0.6870 ->` **`0.7472`** (`24/24` valid, `0` truncations, `table_teds 0.6822 ->` **`0.7893`**); `FinTabNet-OTSL(30)` `NumF1 0.8775 ->` **`0.8923`**; `olmOCR-bench(42)` `0.6418` (`old_scans 9/26`, `multi_column 22/26`) | **`1.254x` faster** OmniDoc (`140.76 s`), **`1.508x` faster** FinTabNet (`20.59 s`), `1.077x` faster olmOCR (`376.06 s`), `11.89 GB` peak; kept **opt-in** because `olmOCR` `table_tests` dipped `37/41 -> 33/41` (`repeat_loop_count=1`). |
| **`fast-qwen3`** (`Qwen3.5-9B-OptiQ-4bit`) | Full Tier-1 `Cat 3(435)` + `Cat 4(300)` on `24 GB` Base M5 | **Cat 3:** `0.6805` vs `4B 0.6414` (`+3.91 pp`; `IFBench 0.27` vs `0.20`, `MMLU 0.8211` vs `0.7895`, `MATH-500 0.70` vs `0.68`); **Cat 4:** `0.8915` (`BFCL 0.8400` vs `0.8333`, `abstention 0.9000` vs `0.8667`, `150/150` valid schemas) | Cat 3 `887.46 s` (`13.34 GB` peak, `8.17 GB` phys); Cat 4 `541.21 s` (`14.33 GB` peak, `8.68 GB` phys), `0` memory-guard trips under `19.2 GB` limit. |
| **`fast-gemma4`** (`MMLU-Redux(285)` diagnostic) | `enable_thinking=False` at `max_tokens={16, 32, 64}` | Identical `209/285` valid (`0.4667`) across `16 / 32 / 64` tokens (`0/76` recovered; `0` engine bugs, matches stock `mlx-vlm`) | `32/76` items spontaneously emit `<\|channel>thought\n` and `44/76` write `> 64` prose tokens before answering; uniform `--mmlu-max-tokens` raised from `16` to `256` across Cat 3. |

### 18.4 Quiescent Base M5 Sprint-2 & Multi-Wave Intelligence Push (`4ee9231+`, 2026-11-26)

Machine-readable receipt combining Sprint-2 paired probes and Goal Waves 1-4 on quiescent **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**: [`benchmarks/m5-sprint2-20261126.json`](benchmarks/m5-sprint2-20261126.json). Every promoted change improves measured accuracy/speed without brittle heuristics or per-benchmark rules; every neutral or regressive probe is recorded below as an honest `KILL`.

| Track / Hypothesis | Outcome | Before -> After (Quiescent Base M5) | Peak Mem & Engineering Disposition |
|---|---|---|---|
| **Shared GPU Top-128 Sampling Fast Path** (`fast-shared/grammar.py`, `fast-qwen3/engine.py`, `fast-gemma4/speculative.py`) | **`PROMOTE`** | `Qwen3.5-4B` `MATH-500(50) K=4`: `3,915.41 s ->` **`1,570.72 s` (`2.493x` faster)** at identical **`0.8600`** accuracy; `FastGemma4` `16x256` (`T=0.6, top_p=0.95`): `16.897 s ->` **`10.651 s` (`1.586x` faster)** | Replaces full-vocab `[B, 250K]` FP32 host copies + CPU `argsort` with `[B, top_k]` GPU gather (`gather_topk_stats_mlx`) and deterministic `np.lexsort` tie-broken nucleus sampling (`try_sample_topk_row`) + safe flat-tail/NaN fallback (`7.10 GB` peak). |
| **Test-Time Self-Consistency & Rolling Dispatch** (`fast-shared/self_consistency.py`, `fast-qwen3`) | **`PROMOTE`** | `Qwen3.5-4B` `MATH-500(50)`: `greedy_1536` `0.6600` (`37/50` valid) / `greedy_2560` `0.7600` -> **`0.8600`** static (`46/50` valid, L3 `1.00`, L4 `0.90`, `1,570.72 s`) and **`0.8200`** rolling (`48/50` valid, **`966.93 s` = `4.049x` faster than Phase-1**) | Lifted into `fast-shared` (`select_majority_answer`, `select_majority_json`, `extract_choice_letter`, `generate_self_consistent`), auto-enabling `rolling=True` on supporting engines for `>= 4` samples (`10.37 GB` peak; explicit `rolling=False` keeps `7.10 GB` peak); `adaptive=False` default (`initial_k=2` `KILL`ed at `0.8000` vs `0.8600`, `K=8` `KILL`ed at identical `0.8600` / `7,580.4 s`). |
| **`MMLU-Redux(285)` Self-Consistency (`W2`)** (`Qwen3.5-4B`, `rolling=True`) | **`KILL`** | `greedy 0.7825` (`31.89 s`) vs `sc_adaptive_k4 0.7825` (`71.84 s`, `0.0000` lift) | Confirms test-time SC helps multi-step reasoning (`MATH`), not short factual MCQA. |
| **`OvisOCR2` `< 0.35` Continuation Gate (`C1`)** & **`Gemma-4` `MMLU-256` (`C2`)** | **`KILL`** | `C1`: `22/24`, `0.6896` (`0` lift over baseline `0.6896`); `C2`: `209/285 0.4667 -> 211/285 0.4807` (`+1.40 pp` at `4.06x` wall `6.83 s -> 27.74 s`) | Reverted `ovis_engine.py` `< 0.35` tweak; kept default `C1` / `C2` untouched. |
| **`RAGTruth(600)` Sentence Decomposition (`B1`)** & **Truncation Cascade (`W1`)** | **`KILL`** | `B1` (`sent_min`): `0.6719 -> 0.6226` AUROC (`-4.93 pp`); `W1` (`Laya` `123` clean + `Kev-0.8B` `477` trunc): `0.6795 -> 0.5396` raw / `0.6783` OOF (`39.54 s` vs `12.41 s`, `gt_512` `0.6354 -> 0.6153`) | Isolated sentences lose coreference context, and NLI-tuned `Laya` outperforms zero-shot `Kev-0.8B` even on `gt_512` RAGTruth rows. |
| **`RewardBench-2(250)` Comparative Judging (`B2`)** | **`KILL`** | Pointwise `Kev-0.8B 0.3040` -> listwise `0.3280` (`61.3 s`), sym pairwise `0.3640` (`216.1 s`, `+6.0 pp`); `Laya` fell `0.300 -> 0.208 / 0.172` | Missed pre-registered `+10 pp` (`0.404`) gate. |
| **`ExtractConf`-Gated Extraction Retry (`W3`)** (`LiquidAI(150)` + `BFCL(150)`) | **`KILL`** | `Qwen4B` (`0.9526 / 0.8333`, `349.2 s`): `_reasoning` dropped `LiquidAI` to `0.9410`; `K=3` vote `0.9531 / 0.8333` (`+150.7 s`); `4B->9B` `0.9476 / 0.8400` (`+90.9 s`, `15.56 GB`). `Gemma4->4B`: `0.9470 -> 0.9550` (`+4.03 pp` tail, `BFCL 0.8267` unchanged) | Missed `+1.5 pp` overall / `+5.0 pp` tail gate; `Gemma-4` standalone already delivers `>99%` of `Qwen-4B` Cat-4 accuracy (`0.9470 / 0.8267` vs `0.9526 / 0.8333`) **`6.56x` faster** (`53.25 s` vs `349.21 s`). |
| **6-Domain Safe Retrieval Fusion (`W4`)** (`LegalBench-RAG` + 5 `NanoBEIR` subsets) | **`KILL`** | `dense` 6-domain mean `0.5877` (`SciFact 0.8824`, `LegalBench 0.2677`); `dense_rerank_laya600` mean `0.5730` (`LegalBench 0.3701`, `SciFact 0.7703`); `rrf_dense_bm25_laya04` mean `0.5923` (`FiQA 0.5256`, `HotpotQA 0.9055`, `SciFact 0.8321`, `LegalBench 0.2684`) | `Laya` (`+10.2 pp` `LegalBench`) and `BM25` (`+5.5 pp` `FiQA`) help domain-specifically, while `dense` leads `SciFact`; kept `dense` safe default and explicit opt-in modes. |

### 18.5 Quiescent Base M5 Gap 1-4 & Wave-5 Frontier Accuracy Push (`6359a91+`, 2026-11-27)

Machine-readable receipt combining all Gap 1-4 and Wave-5 Bet 1-4 paired evaluations on quiescent **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**: [`benchmarks/m5-wave5-20261127.json`](benchmarks/m5-wave5-20261127.json). Zero production files were modified by experimental probes prior to `PROMOTE`.

| Track / Hypothesis | Outcome | Before -> After (Quiescent Base M5) | Peak Mem & Mechanism Disposition |
|---|---|---|---|
| **Gap 2: Gemma-4 `<\|channel>...<channel\|>` Fix & Bounded Thinking** (`fast-gemma4`, `benchmark_outcomes.py`) | **`PROMOTE`** | `MMLU-Redux(285)`: `baseline_nothink_256` `0.4807` (`212/285` valid, `35.08 s`) / `nothink_1024` `0.6351` (`269/285` valid, `101.97 s`) -> **`think_b1024_a256` `0.7404`** (**`+25.96 pp`** vs baseline, `281/285` valid, `249.65 s`); `Qwen3.5-4B` `IFBench(100)`: `nothink_512` `0.22` -> **`nothink_1024` `0.26`** (`84/100` valid, `188.70 s`) | Fixed token-ID stripping of Gemma-4 `<\|channel>...<channel\|>` (`100..101`) prior to `skip_special_tokens=True` (`+1.40 pp` recovery), added `GenerationItem.reasoning` + two-pass bounded `thinking_budget_tokens`; `Qwen3.5-4B` `MATH-500(50)` `K=1` thinking `KILL`ed (`0.68` vs `0.70`/`0.76`/`0.86`) and `IFBench` `<think>` `KILL`ed (`99/100` forced closes, `3.0x` lower EOT). |
| **Gap 4: `fast-ocr` `--ocr-max-tokens 8192` Default** (`benchmark_outcomes.py`) | **`PROMOTE`** | Stock `OvisOCR2-8bit` on `OmniDocBench-EN(24)`: `4096` (`22/24` valid, `0.6910`, `table_teds 0.6822`, `176.39 s`) -> **`8192` (`24/24` valid, `0.7437` = `+5.27 pp`, `table_teds 0.7670`, `153.74 s`, `1.147x` faster)** | Recovers `93.6%` of `v7final` OmniDoc gain with `0` `table_tests` regressions; `v7final` default promotion **`KILL`**ed after tracing `37/41 -> 33/41` dip to irreducible separator/cell-merge weights on `f5e5d540` (`0/5` across all 8 ablations). |
| **Bet 2 (Part A): Contextual Sub-Chunk `MaxSim` + Margin `GenRM`** (`fast-shared/hybrid_retrieval.py`) | **`PROMOTE`** | `LegalBench-RAG(175)` `fast-pplx` 4K (`repeat_header_tokens=128`): `dense_ctx4k` `0.6852` (`R=0.8541`) -> **`maxsim_always` `0.7569`** (**`+7.17 pp`**, `R=0.9133`, `0.169 ms/q`, `14.31 s` enc) and **`margin75_ms_genrm0_w065` `0.7803`** (**`+9.51 pp`**); `NanoSciFact(50)`: `0.8139 ->` **`0.8206` (`+0.68 pp`)** | Escapes single-vector sign-rank limits (`LIMIT`) while preserving `4K` + header context at identical encode time (`2.64 GB` peak). |
| **Bet 2 (Part B): 1-Pass Prefill `GenRMScorer`** (`fast-decision-models/genrm.py`) | **`PROMOTE`** | `RAGTruth(600)`: `Laya` `0.6795` -> **`Qwen3.5-4B` `genrm_0shot` `0.8993` AUROC** (**`+21.98 pp`**, `macro_f1 0.8181`, `Brier 0.1253`, `364.14 s`); `RewardBench-2(250)`: `Laya` `0.3160` -> **`genrm_0shot_bo4` `0.5640`** (**`+24.80 pp`**, `263.26 s`) | 1-pass calibrated prefill logprob beats truncated `80`-tok `enable_thinking=False` CoT on both `RAGTruth` (`0.8491`) and `RB2` (`0.5360`) at `2.5x` lower latency (`4.68 GB` phys / `10.18 GB` peak). |
| **Gap 1: `Laya` `>512` GPU Overflow** | **`KILL`** | `RAGTruth(600)` `LayaMlxGpuBackend`: `512` (`0.6717`, `gt_512=0.6280`) -> `4096` (`0` trunc, `0.6900`, `gt_512=0.6603` `+3.23 pp`), `RB2` `0.3160 -> 0.2800`, `21.1 s -> 40.7 s`, `2.61 -> 4.13 GB` | Overall `+1.06 pp` missed `+2.5 pp` gate (CI95 crosses `0`); `Laya` was fine-tuned at `max_len=1024`. |
| **Gap 3: Symmetric `Laya` BEIR Rerank & `pplx` 4K Stack** | **`KILL`** | `NanoSciFact`: `0.8901 -> 0.7202` (`-16.99 pp` even with symmetric `qdef`); `fast-pplx` `4K` + `Laya`: `0.6852 -> 0.6673` (`-1.78 pp`) | Isolated 600-char snippets strip `4K` contract context from `fast-pplx`; symmetric `qdef` did lift isolated `EmbeddingGemma-2` `LegalBench` `0.3715 -> 0.3987`. |
| **Bet 1: CoT Inflation Audit, Entropy-Gated `K=4`, Difficulty Routing** | **`KILL`** | `Qwen3.5-4B` `MATH-500` `K=4` entropy-gated (`20.2%` fork): `0.8200 -> 0.8400` (`+2.0 pp`), `-2.22%` tokens (`970.5 -> 988.4 s`); `Gemma-4` `MMLU` routed (`48.1%` esc): `0.7368` (`98.65%` retention) at `79.2%` wall | Zero mid-trace solved-to-wrong flips observed on either model (`0`); wrong traces bloat because models genuinely get stuck (`1.94x` length on `Qwen` `MATH`), not from late flips or routine-token noise. |
| **Bet 3: 120-Step `IPT + Forking-Token QAT-RLVR` LoRA** (`Gemma-4-E2B-4bit`) | **`KILL`** | Held-out `IPT(32)` `0.3125 -> 0.6250`, mean Cat-3 thought tokens `618.1 -> 354.8` (`-42.6%`), `IFBench` `0.24 -> 0.26`, BUT `MMLU` `0.7404 -> 0.6491` (`-9.12 pp`), `MATH-500` `0.36 -> 0.26` (`-10.0 pp`), `Cat-4` `0.8836 -> 0.8697` (`-1.39 pp`); `8.25 GB` peak | Synthetic math/format RLVR shortened thinking traces too aggressively and over-specialized away broad general trivia (`MMLU`) and tool accuracy (`Cat-4`). |
| **Bet 4: Verifier-Filtered `OvisOCR2 v8` LoRA** | **`KILL`** | `38` leak-free tables (`143.7 s` train, `3.45 GB`): cured separator/colon defects (`f5e5d540` `0/5 -> 3/5`, `0` loops), but scored `36/41` `table_tests`, `0.7400` `OmniDoc`, `0.8708` `FinTabNet NumF1` | Missed all three promotion thresholds (`>=37/41`, `>=0.745`, `>=0.890`); keep stock `OvisOCR2-8bit` + `8192` default. |

### 18.6 Cross-Platform CPU Retrieval & Query Acceleration (`cpu-query-opt-20261127`)

Machine-readable receipt for all paired `ABBA` experiments on **M4 Pro (`14` cores: `10P + 4E`, `48 GB`, `FEAT_DotProd=1`, `FEAT_I8MM=1`, Apple Clang `21.0.0`)** plus cross-ISA `x86_64` verification (Rosetta 2 `AVX2` live execution + `AVX-512` `vpdpbusd`/`vpopcntq` disassembly): [`benchmarks/m4-cpu-query-opt-20261127.json`](benchmarks/m4-cpu-query-opt-20261127.json).

| Track / Hypothesis | Outcome | Before -> After (Paired ABBA Median) | Parity & Architecture Scope |
|---|---|---|---|
| **H1: Cross-Platform C++17 `CompactRetrievalIndex` (`neon`, `avx512`, `avx2`, `portable_cpp`)** (`fast-shared/cpu_kernels.py`, `retrieval_index.py`) | **`PROMOTE`** | `100K x 1024d` `neon`: `int8` `Q=1` `2.9126 -> 0.7542 ms` (**`3.862x`**), `Q=16` `55.5229 -> 3.3796 ms` (**`16.429x`**), `Q=64` `252.7806 -> 31.2459 ms` (**`8.090x`**); `binary_int8` `Q=1` `0.8505 -> 0.3024 ms` (**`2.812x`**), `Q=16` `14.3327 -> 2.0428 ms` (**`7.016x`**); `portable_cpp` vs NumPy `Q=1 int8` **`42.362x`**, `Q=16 bin` **`54.875x`**; `N=2000` **`2.052x`** | `992` parity cases (`max_abs_diff = 2.86e-6`), Rosetta `x86_64` `AVX2` parity OK, `AVX-512` `vpdpbusd`+`vpopcntq` emit verified; combined `100K Q=16` Hybrid (`binary_int8 + BM25`) `41.9093 -> 6.9244 ms` (**`6.052x`**). |
| **H1b: ARM64 `vmmlaq_s32` (`neon_i8mm`) as Auto Default** | **`KILL` (opt-in only)** | Vs new `neon` `vdotq_s32`: `Q=1` `1.2162 -> 0.9021 ms` (`1.348x`), **but** `Q=16` `2.8554 -> 5.0402 ms` (**`0.567x` slower**), `Q=64` `19.8890 -> 33.2174 ms` (**`0.599x` slower**) | `vtrn1q`/`vtrn2q` interleaving and register pressure regressed batched throughput; keep `"auto" -> "neon"`, expose `"neon_i8mm"` explicitly only. |
| **H2: Fused C++ Multi-Vector `MaxSim` & Zero-Copy Binary Doc Rerank** (`retrieval_index.py`) | **`PROMOTE`** | Skewed `20K` docs / `W=78,048` (`1..32` windows/doc): `int8 Q=1` `3.2334 -> 0.8741 ms` (**`3.699x`**), `Q=16` `42.9776 -> 2.8916 ms` (**`14.863x`**); `binary_int8 Q=1` `2.7852 -> 0.4482 ms` (**`6.214x`**), `Q=16` `51.7898 -> 4.6562 ms` (**`11.123x`**); `portable_cpp Q=16` **`9.266x`** | Exact index parity across all 4 window aggregations (`maxsim`, `hybrid`, `top2_mean`, `mean`) and **`64/64` needle R@10** on both `int8` and `binary_int8`. |
| **H3: C++ Sparse-Reset + Gated Selective-Anchor BMW `BM25Index`** (`hybrid_retrieval.py`) | **`PROMOTE`** | `N=100K` Zipf: `Q=1 pos` BMW `1.3460 -> 0.1247 ms` (**`10.791x`**) / selective gated `5.2684 -> 0.4102 ms` (**`12.843x`**), sparse `0.4491 ms` (**`3.115x`**); `Q=32 default` `32.9868 -> 3.2634 ms` (**`10.108x`**), `Q=32 pos` `40.2015 -> 3.0559 ms` (**`13.155x`**); `N=1K` **`1.144x`** | Selective-anchor gate (`n_terms >= 2 && eff_k <= 256 && min_term_postings <= max(1024, num_docs / 16) && total_postings <= 0.75 * num_docs`) cured the un-gated high-DF cliff (`0.217x` -> **`2.701x`** vs Python / **`1.380x`** vs sparse). |
| **H4a: `FastBgeSmall` `Q >= 8` Batched Query Routing & ANE Table-Bind Cache** (`fast-bge-small/engine.py`) | **`PROMOTE` (`Q >= 8`)** | `Q=2` `0.952x`, `Q=4` `0.840x` (stay sequential ANE); **`Q=8`** `6.9260 -> 5.6926 ms` (**`1.217x`**), **`Q=16` (`hetero`)** `14.8808 -> 9.8339 ms` (**`1.513x`**), **`Q=64` (`hetero`)** `49.3281 -> 35.9106 ms` (**`1.374x`**) / `gpu` packed `239.8901 -> 62.6733 ms` (**`3.828x`**) | `min_cos = 1.000000`; keeps `Q <= 4` sequentially on low-latency ANE `encode_query` and deduplicates repeated `_bind_ane_tables` ctypes calls. |
| **H4b: Custom W8A8 CPU `FastBgeSmall` Encoder vs Accelerate AMX `FP32`** | **`KILL`** | `S=21`: `libbge_cpu_query.dylib` (Accelerate AMX `cblas_sgemm`) **`4.3854 ms`** vs `bge_int8_cpu_query` **`10.6953 ms`** (**`0.410x`**, `2.44x` slower, `cos=0.9991` vs `0.999999`) | Accelerate AMX saturates tiny-`M` `384d` weights in SLC faster than scalar/NEON `W8A8`; never hand-roll INT8 CPU transformer matvec over Accelerate AMX for small encoders. |
| **H5: `FastEmbeddingGemma2` Multi-Query Batching + Zero-Thread Inline ANE** (`fast-embeddinggemma-2/engine.py`) | **`PROMOTE`** | `encode_texts(..., is_query=True)`: `hetero` `Q=2..64` `1.340x-2.721x` vs sequential (`2.099x` `Q=32`, `2.721x` `Q=64`); inline ANE (`1..3` short lanes w/ `S=32/64` downshift) adds **`1.314x`/`1.320x`/`1.133x`** at `Q=2/4/8`; `gpu` `Q=4/16/64` **`2.836x`/`5.028x`/`10.888x`** (`1.028x` mixed `>512`) | `min_cos = 0.999998-0.999999`; `Q=1` C++/NEON `emb2_fast_io.cpp` (`prep=0.026 ms`, `fin=0.013 ms`) is `<1.5%` of `3.96 ms` ANE exec, so eliminating the sequential Python loop unlocks full speedups (`Q >= 2` batched). |
| **H6: Lower-Bit `EmbeddingGemma-2` GGUF Quants on AVX2-Only x86_64 (`llama.cpp` Haswell)** (`benchmarks/x86-avx2-eg2-gguf-quant-20261127.json`) | **`KILL`** | `4 vCPU` x86_64 AVX2+FMA+F16C (no VNNI, `34` frozen retrieval queries, `512d`, live `10.87M`-passage index): `UD-Q6_K_XL` `0.902x` 4c CPU / `1.043x` clean 1c CPU (`min_cos512=0.99857`, `21/34` top-5); `UD-Q4_K_XL` `1.21x` 1c / `1.28x-1.44x` 4c CPU (`0.99142`, `8/34` top-5, passage recall **`8/34` vs `9/34`**); `FFN=Q4_K` `1.09x` 1c / `1.15x` 4c CPU (`0.99481`, `12/34` top-5, **`8/34`** pass); runtime flags (`--load-mode none`, `--no-repack`, `-fa off`) `0.78x-0.94x` | Only 4-bit families trigger Haswell `CPU_REPACK` (`61-70 MiB`), not `Q8_0`/`Q6_K`/`Q5_K`; `eg2` overflows `FP16`. Keep `eg2-q8.gguf` (`BF16` parent) with existing defaults on AVX2-only VMs. |

### 18.7 Calibrated Local Judging, Order-Flip Deferral & PPI++ Audit (`judge-eval-w12-20261127`)

Machine-readable receipt combining all Wave 1+2 judge calibration, abstention, cross-family fusion, and PPI++ validation experiments on Base Apple M5 (`24 GB`, `macOS 27.2.1`; `Qwen3.5-4B-OptiQ-4bit` `7.65 GB` peak, `Gemma-4-E2B-it-qat-4bit` `5.08 GB` peak) plus CPU validation over Han & Kim's 4,000 Chatbot Arena battle verdicts (`arXiv:2611.04090v1`): [`benchmarks/judge-eval-w12-20261127.json`](benchmarks/judge-eval-w12-20261127.json).

| Track / Hypothesis | Outcome | Before -> After / Key Measured Result | Scope & Engineering Disposition |
|---|---|---|---|
| **`T-1` + `E-1`: `GenRMScorer` Order-Flip Telemetry & Conformal Deferral** (`fast-decision-models/genrm.py`) | **`T-1` `PROMOTE`** / **`E-1` `KILL` extra scorer** | `T-1` adds `score_pairwise_swap_invariant`, `decision_margin`, `pairwise_order_disagreement`, and opt-in `abstain` at `0` extra forward passes (`12/12` CPU tests). `E-1` 5-fold OOF tested adding `order_disagreement` to calibrated swap-averaged `decision_margin` on `RewardBench-2(250)`: `Qwen3.5-4B` `P@80` stayed **`0.7150 -> 0.7150`** (`+0.00 pp`, AURC `0.1514 -> 0.1513`) and `Gemma-4-E2B` regressed **`0.5600 -> 0.5450/0.5500`** (`-1.0` to `-1.5 pp`). | Swap-averaging (`0.5*(p_fwd + (1 - p_rev))`) already collapses `decision_margin` toward `0` when A/B orders disagree. Keep zero-cost telemetry in `genrm.py` (`T-1`) + existing `calibration.py` + `croq.py` (which on `RAGTruth(600)` cut `ECE-15` `-59.6%` to `0.0530` and lifted `P@80` `0.8708 -> 0.8979` on `Qwen`, `0.6292 -> 0.7708` on `Gemma`). |
| **`T-2` + `E-3`: Pure-NumPy PPI++ (`compute_ppi_comparison`) & `arXiv:2611.04090v1` Audit** (`fast-shared/outcome_eval/metrics.py`) | **`PROMOTE`** / **`PASS`** | Validated across all `5` judges x `33` Arena pairs x `200` resamples: Sim B asymptotic 95% CI coverage hits **`92.70%-93.53%`** at `n=30-40` (`PASS >= 92%` gate) and **`bootstrap_t`** hits **`95.45%-96.48%`**. At `n=30` human audits/pair, PPI++ resolves **`59.3%-62.7%`** of primary-23 pairs at **`99.96%-100.0%` precision** (vs. `34.8%` coverage / `87.5%` precision for binary `Haiku + GPT-6 Luna` AND-consensus), neutralizing `GPT-6 Luna` via power-tuned `lambda_hat = 0.054-0.132`. | `40/40` CPU tests pass (`test_ppi_metrics_cpu.py` + `test_outcome_eval_cpu.py`); Acklam rational `norm_ppf` + pooled variance power-tuning + `lambda_hat < 0.2` misalignment alarm, zero SciPy dependency. Recommend `ci_method="bootstrap_t"` for `n <= 40` human audits. |
| **`E-2`: Two-Family (`Qwen3.5-4B + Gemma-4-E2B`) Continuous Logit Fusion vs. Binary AND-Voting** | **`KILL`** | Base M5 standalone: `Qwen3.5-4B` `RB2 0.6400` (`545.1 s`) / `RAGTruth 0.9012` AUROC (`353.2 s`); `Gemma-4-E2B` `RB2 0.5120` (`93.1 s`, listwise `0.2960 -> 0.5160` after OOF `'A'`-slot prior debiasing) / `RAGTruth 0.7891` AUROC (`56.5 s`). On `RB2(250)` (`21.2%` pairwise / `39.6%` refined disagreement), binary `Qwen == Gemma` consensus hit `70.86%` precision at `60.4%` (`151/250`) coverage vs. **`82.78%`** for single calibrated `Qwen` at the same `60.4%` coverage (`-11.92 pp`). 5-fold OOF continuous fusion put `w_q = 1.0` on `RB2` (`0.7150 -> 0.7150`, `+0.00 pp`, `1.22x` full / `1.09x` min wall) and **`0.8979 -> 0.8938`** on `RAGTruth` (`-0.42 pp`, `1.16x` wall). | Keep single calibrated `Qwen3.5-4B` (`+11.92 pp` precision above binary consensus at matched `60.4%` coverage on `RB2`, `-2.33 pp` gap even vs calibrated consensus on `RAGTruth`); do not add a dual-family runtime scorer into `packages/`. |
| **`E-4`: `n=30` RLVR / LoRA Reward-Hacking Audit Gate** (`audit_judge_panel()`) | **`PASS`** | Verified **`32/32`** gate checks (`200` resamples, `n=30` on `RB2` & `RAGTruth`): single-judge `+0.15` bias shift blocked `100.0%`; shared cross-family `+0.15` bias shift escaped cross-family-gap checking **`100%`** (`0.0%` detection) and fooled judge-only panels **`100%`**, whereas PPI++ + cross-family gating blocked **`98.5%-100.0%`** while keeping null false promotion `0.5%-2.0%` and improving clean `+0.20` gain power (`0.875` vs `0.595` `RB2`; `0.980` vs `0.560` `RAGTruth`, `N_eff/n = 1.53x-2.71x`). | Ready pattern combining cross-family check (`gap <= 0.10`) with human-anchored `compute_ppi_comparison()` (`ci_low > 0.50`, `lambda_hat >= 0.30`). |

## 19. Saved Benchmark Receipts Index

| Artifact | Scope |
|---|---|
| [`benchmarks/judge-eval-w12-20261127.json`](benchmarks/judge-eval-w12-20261127.json) | 2026-11-27 Base M5 `Qwen3.5-4B` + `Gemma-4-E2B` judge capture, `E-1` order-flip `KILL`, `E-2` two-family fusion `KILL`, `E-3` `arXiv:2611.04090v1` PPI++ validation (`PASS`), and `E-4` reward-hacking audit gate (`PASS`) (`Section 18.7`) |
| [`benchmarks/x86-avx2-eg2-gguf-quant-20261127.json`](benchmarks/x86-avx2-eg2-gguf-quant-20261127.json) | 2026-11-27 x86_64 AVX2 Haswell `EmbeddingGemma-2` GGUF 20-candidate screen & paired ABBA `KILL` receipt (`Section 18.6 H6`) |
| [`benchmarks/m4-cpu-query-opt-20261127.json`](benchmarks/m4-cpu-query-opt-20261127.json) | 2026-11-27 M4 Pro + cross-ISA CPU retrieval, BM25 BMW, & BGE query batching receipt (`Section 18.6`) |
| [`benchmarks/m5-wave5-20261127.json`](benchmarks/m5-wave5-20261127.json) | 2026-11-27 quiescent Base M5 Gap 1-4 & Wave-5 Bet 1-4 receipt (`Section 18.5`) |
| [`benchmarks/m5-sprint2-20261126.json`](benchmarks/m5-sprint2-20261126.json) | 2026-11-26 quiescent Base M5 Sprint-2 & Goal Waves 1-4 receipt (`Section 18.4`) |
| [`benchmarks/m5-accuracy-20261126.json`](benchmarks/m5-accuracy-20261126.json) | 2026-11-26 quiescent Base M5 accuracy push receipt (`Section 18.3`) |
| [`benchmarks/m5-suite-20261126.json`](benchmarks/m5-suite-20261126.json) | 2026-11-26 quiescent Base M5 (`d9f65a3`) full 4-suite headline receipt (`Section 18.1-18.2`) |
| [`benchmarks/review-20261121.md`](benchmarks/review-20261121.md) | November 21 cross-model review & repair ledger |
| [`benchmarks/capability-corrections-20261121.json`](benchmarks/capability-corrections-20261121.json) | Capability matrix & parallel Core ML compile corrections |
| [`benchmarks/coreml-20261120.json`](benchmarks/coreml-20261120.json) | Public Core ML export revisions, token limits & parity audit |
| [`benchmarks/decisions-20261121.json`](benchmarks/decisions-20261121.json) | `Laya-Typed-421M`, `Kev-0.8B`, and review-queue pilot runs |
| [`benchmarks/document-workloads.json`](benchmarks/document-workloads.json) | Cross-model retrieval, extraction, and generation outcome JSON |
| [`benchmarks/embeddinggemma-legal-512-20261121.json`](benchmarks/embeddinggemma-legal-512-20261121.json) | `EmbeddingGemma-2` 512- vs 2,048-token `LegalBench-RAG` audit |
| [`benchmarks/gemma-grammar-20261121.json`](benchmarks/gemma-grammar-20261121.json) | `Gemma-4` strict colon-space JSON vs compact/150K vocab tests |
| [`benchmarks/pplx-padding-20261121.json`](benchmarks/pplx-padding-20261121.json) | `fast-pplx` length-sorted batching vs stock padding check |
| [`benchmarks/fast-qwen3-20261122.json`](benchmarks/fast-qwen3-20261122.json) · [`runs/`](benchmarks/runs/) | `fast-qwen3` `v0.2`/`v2.4` `ABBA`, Category-4, and per-run receipts |
| [`benchmarks/v22-validation-20261121.json`](benchmarks/v22-validation-20261121.json) | `v2.2` cross-silicon validation & promotion gates (`Section 9`) |
| [`benchmarks/v23-validation-20261121.json`](benchmarks/v23-validation-20261121.json) | `v2.3` accuracy, precision & contextual gates (`Section 10`) |
| [`benchmarks/v25-rd-winners-20261122.json`](benchmarks/v25-rd-winners-20261122.json) | `v2.5` `MaxSim` `32K`, `MRL + INT8`, `QuaRot` KV & CoreAI gates (`Section 12`) |
| [`benchmarks/v30-validation-20261123.json`](benchmarks/v30-validation-20261123.json) | `v3.0` vision prefetch, NAX meta-only XOR, multimodal `MTP`, OCR confidence & negative gates (`Section 14`) |
| [`benchmarks/dual-mac-suite-20261124.json`](benchmarks/dual-mac-suite-20261124.json) | 2026-11-24 dual-Mac 4-suite baseline & targeted post-fix rerun receipts (`Section 15`) |
| [`benchmarks/rca-and-laya-fixes-20261124.json`](benchmarks/rca-and-laya-fixes-20261124.json) | 2026-11-24 Wave 3 dual-Mac post-fix outcomes, 6-shape Laya matrix & Laya FP16 LayerNorm RCA (`Section 16`) |
