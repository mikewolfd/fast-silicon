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
| **Cat 4: `GPT-OSS-20B`** (`50`) | **`fast-gpt-oss` (`llg_on`)** *(48 GB only)* | *Gated (`24 GB`)* | *Gated* | — | `0.5063` (**`49/50`**) | **`0.318`** (`79.7s`) | `12.59 /` **`12.77 GB`** | — / **`1.61×` `EOT` (`+53%` valid docs)** | *48 GB only* |
| Cat 4: Stock `MLX` | `gpt-oss-20b-MXFP4-Q4` (`mlx-lm`, unconstrained) | *Gated (`24 GB`)* | *Gated* | — | `0.5755` (`32/50`) | `0.197` (`146.2s`) | `14.48 / 14.30 GB` | `1.00×` | *48 GB only* |

The November 20 generation table used the English-oriented output vocabulary and minified key separators for Gemma. Across its 300 mixed tasks, that configuration returned 299 valid outputs in 82.7 seconds. One output looped. The reported 3.62 valid responses/s and the 10.46 fixture documents/s measure different workloads. [Historical task scores and source hashes](benchmarks/document-workloads.json)

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
| Laya, `fast-decision-models` Neural Engine scorer | 41.91 questions/s | 44.38 questions/s |

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
| **`fast`** (default) | **`13,542`** (`1.16×` stock) | **`13,464`** (`1.54×` stock) | `10,074` | **`10,311`** (`1.22×` stock) | `0.9909` *(corpus `0.9866`)* | `2.02` | Peak ANE speed (`1.00×`) |
| **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`) | `12,754` (`0.94×`) | `12,856` (`0.95×`) | `9,618` | `9,762` (`0.95×`) | **`0.9988`** *(corpus **`0.9964`**)* | **`0.79`** | Closes **`73%`** of ANE-vs-GPU cosine gap for `4.5%–5.8%` latency cost |
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
  --gate-profile lossless --min-eot-ratio 1.00 \
  artifacts/outcomes_base.json artifacts/outcomes_cand.json
```

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
- **`9B-OptiQ` Native `MTP` (`norm_add_one` + `draft_k=3`) & `ByteLevel` `65K` Vocab Remap (`v2.4`, `2026-11-22`):** Auto-detecting `optiq/mtp.safetensors` zero-centered RMSNorm weights (`detect_mtp_norm_add_one`) plus setting `recommended_draft_k=3` eliminated the pre-fix `MTP` slowdown (`13.25 tok/s` / `0.441×` M4 Pro, `10.13 tok/s` / `0.523×` Base M5), and decoding `ByteLevel` BPE tokens in `QwenVocabRemapper` restored compact `65,536`-entry vocab speedups:
  - **`9B-OptiQ` (`vocab=248,077`, `k=3`):** M4 Pro `B=1` **`49.88 tok/s`** (`1.255 docs/s`, `2.709×`), `B=8` **`112.90 tok/s`** (`2.858 docs/s`, `8/8` valid vs `0/8` raw, `5.763×`), `MTP B=1` `32.32 ->` **`41.43 tok/s`** (**`1.282×`**, `100%` greedy token match), peak `16.36 GB` (`MLX active 7.47 / MLX peak 8.49 / OS phys 8.31 GB`). Base M5 `B=1` **`32.73 tok/s`** (`0.823 docs/s`, `2.841×`), `B=8` **`101.23 tok/s`** (`2.563 docs/s`, `8/8` valid vs `0/8` raw, `8.438×`), `MTP B=1` `19.34 ->` **`32.44 tok/s`** (**`1.677×`**, `100%` greedy token match), peak `16.18 GB` (`< 19.2 GB` ceiling; `MLX active 7.47 / MLX peak 8.33 / OS phys 8.29 GB`).
  - **`9B-OptiQ` (`vocab=65,536`, `ByteLevel` decoded):** M4 Pro `B=1` **`54.81 tok/s`** (`1.396 docs/s`, `2.526×`), `B=8` **`115.93 tok/s`** (`2.954 docs/s`, `8/8` valid, `4.930×`), `MTP B=1` `35.90 ->` **`50.17 tok/s`** (**`1.397×`**; `k`-sweep peaks at `58.40 tok/s` / `1.620×` at `k=3` vs `51.84` at `k=4` and `52.54` at `k=2`), peak `16.87 GB`. Base M5 `B=1` **`37.17 tok/s`** (`0.947 docs/s`, `2.676×`), `B=8` **`105.74 tok/s`** (`2.694 docs/s`, `8/8` valid, `7.189×`), `MTP B=1` `21.99 ->` **`42.25 tok/s`** (**`1.921×`**; `k`-sweep peaks at `44.29 tok/s` / `2.003×` at `k=3` vs `40.21` at `k=4` and `36.98` at `k=2`), peak `16.68 GB` (`< 19.2 GB`).
  - **`0.8B` (`248,070` -> `65,536` vocab) & `4B-OptiQ` (`v2.4` Receipts):** `0.8B` M4 Pro `B=1` `187.32 ->` **`240.30 tok/s`** (`4.597 -> 5.970 docs/s`), `B=8` `635.61 ->` **`767.46 tok/s`** (`15.694 -> 18.950 docs/s`, `8/8` valid, `4.77 -> 4.99 GB`; `65K` `MTP` `134.98 -> 101.64 tok/s`, `0.753×`); Base M5 `B=1` `139.56 ->` **`178.27 tok/s`** (`3.425 -> 4.429 docs/s`), `B=8` `570.71 ->` **`658.54 tok/s`** (`14.092 -> 16.311 docs/s`, `8/8` valid, `4.61 -> 4.81 GB`; `65K` `MTP` `94.25 -> 98.91 tok/s`, `1.049×`). `4B-OptiQ` M4 Pro `B=1` `79.00` / `B=8` `183.10 tok/s` (`8.76 GB` peak); Base M5 `B=1` `57.38` / `B=8` `171.99 tok/s` (`8.54 GB` peak).
- **Cross-Silicon `Cat 3` & `Cat 4` Outcomes (`scripts/benchmark_outcomes.py --tier tier1 --max-items 25 --timeout-sec 1800`):** Unbounded full `tier1` timed out at the default `900 s` cap; at `n=25` per task under real-task evaluation:
  - `4B-OptiQ` `Cat 4` (`LiquidAI(25) + BFCL_v3(25)`): **PASS** on both Macs — Base M5 **`50/50 valid`**, `0.9024` (`LiquidAI 0.9248`, `BFCL 0.880`, `jsb 0.95`), `64.97 s`, `EOT 0.6945`, `OS phys 4.37 / peak 6.84 GB`; M4 Pro **`50/50 valid`**, `0.9021` (`LiquidAI 0.9243`, `BFCL 0.880`, `jsb 0.95`), `92.09 s`, `EOT 0.4898`, `OS phys 4.84 / peak 8.49 GB`.
  - `9B-OptiQ --mtp-drafter` `Cat 4`: **PASS** on both Macs — Base M5 **`50/50 valid`**, `0.8857` (`LiquidAI 0.9314`, `BFCL 0.840`, `jsb 0.95`), `110.16 s`, `EOT 0.402`, `OS phys 8.39 / peak 10.94 GB`; M4 Pro **`50/50 valid`**, `0.8874` (`LiquidAI 0.9347`, `BFCL 0.840`, `jsb 0.95`), `159.96 s`, `EOT 0.2774`, `OS phys 8.79 / peak 12.41 GB`.
  - `Cat 3` (`IFBench(25) + MMLU(25) + MATH-500(25)`, `enable_thinking=False`): `IFBench` zero-shot adherence drags `Cat 3` below threshold (`FAIL`) — `4B-OptiQ` Base M5 `59/75`, `0.5333` (`MMLU 0.80`, `MATH 0.64`, `IFBench 0.16`, `repeat_loop_count=3`, `376.09 s`, `4.13 GB`) and M4 Pro `60/75`, `0.5467` (`MMLU 0.80`, `MATH 0.68`, `IFBench 0.16`, `repeat_loop_count=1`, `262.79 s`, `4.32 GB`); `9B-OptiQ --mtp-drafter` Base M5 `60/75`, `0.5467` (`MMLU 0.76`, `MATH 0.72`, `IFBench 0.16`, `repeat_loop_count=2`, `582.74 s`, `8.15 GB`) and M4 Pro `58/75`, `0.5733` (`MMLU 0.76`, `MATH 0.72`, `IFBench 0.24`, `repeat_loop_count=2`, `422.16 s`, `8.32 GB`).
- **Real-Checkpoint Parities & Concurrent `GPU + Pure-ANE` (`v2.4`):**
  - `9B-OptiQ` (M4 Pro & Base M5): `S>256` (`288..290`) ragged `B=1` vs `B=4` exact (`4/4` on both Macs), `PrefixTree` wave vs uncached `kv_i8`/`bf16` (`True` both; `1536` reused tokens across `3` snapshots on M4 Pro), `kv_i8` vs `bf16` (`True` both), `S=2213` chunked vs whole first-token (`True` both). `4B-OptiQ`: `S>256` ragged `4/4` Base M5 (`3/4` M4 Pro on `bf16` tie `" extraction"` vs `" extracted"`), `PrefixTree` wave vs `kv_i8` (`4/4` M4 Pro, `3/4` Base M5 deterministic tie `" list"` vs `" repetitive list"`), `S=2213` chunked vs whole (`True` both).
  - Base M5 concurrent `4B-OptiQ GPU` (`8x48` tokens) + `FastBgeSmall pure-ANE` (`3x32x384` `[32, 384]`): `2.626 s` wall, `OS phys 5.32 GB`, peak **`9.39 GB`** (`MLX peak 8.92 GB`, well under the `19.2 GB` `24 GB`-class ceiling).
  - `fast-pplx` M4 Pro real-checkpoint parity (`--allow-contended`): `gpu fp16` `min_cos=0.999994` (`s4096 cos=1.0`), `gpu-bf16` `min_cos=0.999892` (`s4096 cos=1.0`), `ane`/`hetero2` `min_cos=0.982756` (`int8_min_cos=0.98261`).
