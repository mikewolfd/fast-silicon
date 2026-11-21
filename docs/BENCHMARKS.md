# fast-silicon benchmarks

On the 24 GB M5, `fast-embeddinggemma-2` and OpenInfinity’s public 8-bit Core ML export had nearly identical warm single-query latency. The project package produced vectors for the same 2,000 pretokenized documents at 3.27× the token rate. NMTZ’s Core ML Laya finished the mixed decision suite sooner than the project’s Laya on both Macs. Stock MLX recovered field values more accurately than constrained `fast-gemma4` on the M5. [Matched query and indexing runs](#warm-query-corpus-and-packing) · [Decision timings](#task-quality-and-elapsed-time) · [Extraction results](#structured-generation)

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

The evaluator assigns `valid_i` a value of 0 or 1 and `score_i` a value between 0 and 1 using public datasets and deterministic scoring. `T_e2e` is elapsed time for the scored workload. On the M5 Gemma run, 300 tasks × 0.748675 mean score ÷ 82.6576 seconds = **2.717 score units/s**. The same run produced 299 valid responses, or **3.62 valid responses/s**. [Recorded task scores and elapsed time](benchmarks/document-workloads.json)

### Quality, memory and throughput checks

The outcome runner checks three conditions when deciding whether to adopt a candidate configuration over its baseline. [Outcome runner](../scripts/benchmark_outcomes.py)

1. **Gate 1 — Objective Quality Non-Inferiority:**
   - `lossless` profile: `ΔS_obj,e2e >= -max(0.003, 1.96 * SE_paired)` and `repeat_loop_count == 0`.
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

The `lossless` quality rule allows a larger score drop when the sample is more uncertain. On the 50 M4 Pro GPT-OSS extraction and tool tasks, the project scored 0.5063 and stock MLX scored 0.5755. The observed change was −0.0692 with a paired standard error of 0.0782, so the rule’s quality allowance was 0.1533. The project run also recorded two repeat loops and failed its local check. An application with a fixed accuracy requirement can compare it directly with the observed scores and uncertainty. [Paired item scores](benchmarks/document-workloads.json) · [Quality rule](../packages/fast-shared/src/fast_shared/outcome_eval/metrics.py)

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


### Category 2 — Decision, Faithfulness & Reward (`RAGTruth` `600` + `CLINC150` `550` + `RewardBench-2` `250`, `N=1400`)

| Workload (`N=1400`) | Engine / Stock Harness | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **`Laya-Typed-421M`** | **`fast-decision-models` (`laya_ane`)** | **`0.5050`** | **`6.158`** (`114.8s`) | `0.00 / 0.60 / 0.40 /` **`1.00 GB`** | **`0.5050`** | **`5.827`** (`121.3s`) | `0.00 / 0.58 / 0.40 /` **`0.98 GB`** | **`8.29×` CoreAI (`1.16×` MLX\*)** / **`1.92×` MLX\*** | **`1.06×`** |
| `Laya` Stock `CoreAI` | `ahmedandaloes/laya-typed-decisions-CoreAI` (`27.2+`) | `0.4993` | `0.743` (`940.7s`) | `0.00 / 0.29 / 0.00 / 0.33 GB`† | — *(26.6)* | — *(26.6)* | — | `1.00×` *(accurate stock)* | — |
| `Laya` Stock `MLX`\* | `mlx-embeddings` `ModernBERT 8-bit` | `0.2757`\* | `5.319` (`72.6s`) | `1.34 / 18.70 / 0.00 / 18.70 GB` | `0.2986`\* | `3.032` (`137.9s`) | `1.54 / 34.02 / 0.00 / 34.02 GB` | *(`-0.23` quality loss, `19×–34×` RAM)* | `1.75×` |
| **`Kev-0.8B`** | **`fast-decision-models`** (automatic prefix cache) | **0.5250** | **4.395** (167.2 s) | 3.82 / 1.78 / 0.00 / **1.78 GB** | **0.5250** | **3.415** (215.2 s) | 4.31 / 2.29 / 0.00 / **2.29 GB** | **0.87×** / **1.12×** | **1.29×** |
| `Kev-0.8B` Stock `MLX` | `AbstractFramework/kev-0.8b-mlx-8bit` (`mlx-lm`) | `0.5236` | `5.055` (`145.0s`) | `2.65 / 3.06 / 0.00 / 3.06 GB` | `0.5229` | `3.062` (`239.1s`) | `2.92 / 3.26 / 0.00 / 3.26 GB` | `1.00×` | `1.65×` |

The stock MLX ModernBERT run scored 0.2757 on M5. Laya in `fast-decision-models` scored 0.5050. The `ahmedandaloes/laya-typed-decisions-CoreAI` export, running in Apple’s CoreAI runtime, scored 0.4993 and mapped 1.68 GB across all file mappings. The Kev row uses the November 20 automatic-cache run. The Core ML section includes NMTZ’s faster Laya export, the earlier Kev timing and the shared-passage probe. [Public Core ML and project results](#task-quality-and-elapsed-time)

### Structured generation

Category 3 evaluates reasoning on `IFBench`, `MMLU-Redux` and `MATH-500`. Category 4 evaluates extracted fields on `LiquidAI/structured-extraction` and tool calls on `BFCL v3`.

*(Tier-2 additionally uses `epfl-dlab/JSONSchemaBench` as a `<15s` `llguidance` compiler/jump diagnostic gate; semantic value accuracy is evaluated strictly on ground-truth targets in `LiquidAI/structured-extraction` and `BFCL v3`.)*

| Cat & Workload (`N`) | Engine / Stock Harness | Base M5 `S_obj,e2e` (Valid) | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Comb`) | M4 Pro `S_obj,e2e` (Valid) | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | :---: | ---: | :---: | :---: | ---: | :---: | :---: | :---: |
| **Cat 3: `Gemma-4 E2B`** (`435`) | **`fast-gemma4` (`gpu_fp16_kvi8`)** | `0.3517` (`280/435`) | **`0.843`** (`181.5s`) | `5.70 / 9.65 GB` | `0.3586` (`282/435`) | `0.892` (`174.8s`) | `5.84 / 8.74 GB` | **`1.03×`** / `0.94×` | `0.95×` |
| Cat 3: Stock `MLX` | `gemma-4-E2B-it-MLX-4bit` (`mlx-vlm`) | `0.3655` (`292/435`) | `0.818` (`194.4s`) | `5.54 / 5.41 GB` | `0.3655` (`292/435`) | **`0.951`** (`167.2s`) | `5.79 / 5.47 GB` | `1.00×` | `0.86×` |
| **Cat 3: `GPT-OSS-20B`** (`75`) | **`fast-gpt-oss` (`auto`)** *(48 GB only)* | *Gated (`24 GB`)* | *Gated* | — | `0.3067` (`40/75`) | **`0.174`** (`132.3s`) | `11.94 / 12.26 GB` | — / **`1.23×`** | *48 GB only* |
| Cat 3: Stock `MLX` | `gpt-oss-20b-MXFP4-Q4` (`mlx-lm`) | *Gated (`24 GB`)* | *Gated* | — | `0.3600` (`69/75`) | `0.141` (`191.0s`) | `12.97 / 12.07 GB` | `1.00×` | *48 GB only* |
| **Cat 3: `CoreAI Qwen3.5-4B`** (`48`) | `CoreAI` `4bit` vs **`fp4_fp8` (`NAX`)** | `0.3125` / **`0.3125`** | `0.041` / **`0.048`** (`369s -> 311s`) | `0.00 / 1.65 -> 1.72 GB` | — *(26.6)* | — *(26.6)* | — | **`fp4_fp8` = `+18.8%` `EOT`, `0.00` Δacc** | — |
| **Cat 4: `Gemma-4 E2B`** (`300`) | **`fast-gemma4` (`llg_on` / `llg_off`)** | `0.7487` (**`299/300`**) / `0.8241` (`284/300`) | **`2.717`** (`82.7s`) / `1.848` (`133.8s`) | `5.92 / 16.94 GB` | `0.7473` (**`299/300`**) / `0.8178` (`285/300`) | **`2.412`** (`92.9s`) / `1.702` (`144.1s`) | `5.93 / 18.37 GB` | **`1.81×`** / **`1.57×`** (`llg_on` `EOT`) | **`1.13×`** |
| Cat 4: Stock `MLX` | `gemma-4-E2B-it-MLX-4bit` (`mlx-vlm`, unconstrained) | `0.8617` (`286/300`) | `1.501` (`172.2s`) | `6.11 / 5.58 GB` | `0.8658` (`287/300`) | `1.537` (`169.0s`) | `6.36 / 5.66 GB` | `1.00×` | `0.98×` |
| **Cat 4: `GPT-OSS-20B`** (`50`) | **`fast-gpt-oss` (`llg_on`)** *(48 GB only)* | *Gated (`24 GB`)* | *Gated* | — | `0.5063` (**`49/50`**) | **`0.318`** (`79.7s`) | `12.59 /` **`12.77 GB`** | — / **`1.61×` `EOT` (`+53%` valid docs)** | *48 GB only* |
| Cat 4: Stock `MLX` | `gpt-oss-20b-MXFP4-Q4` (`mlx-lm`, unconstrained) | *Gated (`24 GB`)* | *Gated* | — | `0.5755` (`32/50`) | `0.197` (`146.2s`) | `14.48 / 14.30 GB` | `1.00×` | *48 GB only* |

On M5, `fast-gemma4` produced 299 valid responses in 82.7 seconds and used 16.94 GB. Stock `mlx-vlm` produced 286 in 172.2 seconds and used 5.58 GB. The mean field F1 score (credit for recovering the correct field values) was 0.691 versus 0.923 on 150 extraction tasks. Accuracy on the other 150 tasks, which checked tool calls, was 0.807 versus 0.800. One repeated-output loop failed the project run’s local check. Its rate of 3.62 valid responses/s and the 10.46 valid documents/s from eight repeated fixtures answer different questions. [Task scores and source hashes](benchmarks/document-workloads.json) · [Batch capacity](#generation-capacity)

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

On M5, long-passage ranking scores were 0.569 for `fast-embeddinggemma-2`, 0.516 for FluidInference FP16 and 0.518 for OpenInfinity INT8. On short passages they were 0.272, 0.271 and 0.270. A run that gives all three the same 512-token inputs would isolate how much the input cap contributes. [Input lengths and ranking results](benchmarks/coreml-20261120.json)

The public BGE comparison uses sidekickd’s FP16 export. In five M4 Pro spot checks its lowest vector cosine similarity to `fast-bge-small` was 0.99996. A second export, `michaeljelly/bge-small-en-coreml-v1.5`, returned non-finite vectors on CPU/GPU and similarities of 0.52–0.62 on the Neural Engine. These results led to the selection of sidekickd for timing.

Laya measures the public typed option-scoring path, including host embedding lookup and probability conversion. Kev uses a shared-state fused function for inputs that fit and fixed row functions for larger inputs or option sets. The eight-question warm probe below reuses one short state. The public outcome suite exercises the row paths as well.

NMTZ’s public Laya export finished 1,400 decisions sooner on both Macs. On M5 it took 88.4 seconds and 2.73 GB, against 114.8 seconds and 1.00 GB for `fast-decision-models`. The project package scored 0.5050 and NMTZ scored 0.4993 on both hosts. The publisher’s separate action head, which can choose an action or abstain, falls outside this option-scoring test.

On the same 1,400 M5 items, stock MLX Kev earned 5.055 score units/s, `fast-decision-models` with automatic caching earned 4.395, and FluidInference’s public export earned 2.269. The project and stock MLX runs used 1.78 and 3.06 GB, respectively. An earlier project run using the previous cache default finished in 144.9 seconds (736/1,400), compared with 167.2 seconds (735/1,400) on the rerun. Only one best-of-four answer changed. These separate timings leave the cause of the time difference open. [Runs and settings](benchmarks/coreml-20261120.json)

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
| Kev, eight distinct inputs per batch, project / stock `mlx-lm` | 54.52 / 50.6 questions/s | 86.84 / 81.7 questions/s |
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

| Regime (`M4 Pro 48 GB`) | Throughput (`tok/s`) | Peak Metal (GB) | Speedup vs. Stock `mlx-lm` | Gradient Parity |
| :--- | ---: | ---: | :---: | :--- |
| Stock `mlx-lm` BF16 LoRA (unpacked) | `579.1` | `16.09 GB` | `1.00×` | Reference |
| `8-bit g64` `QLoRA` + `pack=2048` (`FFD`) | `545.8` | `25.06 GB` | `0.94×` | `packed_grad_cos = 0.999673`, `qlora_8bit_grad_cos = 0.994881` |
| **`8-bit g64` `QLoRA` + `shared-sys-kv`** | **`1,385.4`** *(up to `1,581` on `1,288`-tok schemas)* | `25.82 GB` | **`2.39×`** *(up to **`2.88×`**)* | Frozen trunk cut `9.23 -> 5.00 GB`; `min_tensor_cos = 0.9991` |

- On the 24 GB M5, two QLoRA attempts exhausted Metal memory. The benchmark targeted 2,048 tokens and then 1,280. The reports leave the failed step unspecified. The sizing policy recommends `pack_len=1024, grad_accum=2, ckpt_tokens=512` with a 19.2 GB ceiling. A completed 24 GB run at those settings would establish throughput. [Sizing policy](../packages/fast-shared/src/fast_shared/runtime_lessons.py)
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

| Workload | Baseline AR (`plain_lq1`) M5 / M4 Pro | Best `MTP` (`fast-gemma4`) M5 / M4 Pro | `MTP` Speedup vs. AR (`M5` / `M4 Pro`) | Notes & Parity |
| :--- | :---: | :---: | :---: | :--- |
| **`B=1` Free-Text** | `33.5` / `47.9 tok/s` | **`83.4`** (`k4 kvi8`) / **`102.0 tok/s`** (`k4 kvi8`) | **`2.49×`** / **`2.13×`** | `draft_share = 6.1%` (M5) / `10.3%` (M4 Pro) |
| **`B=4` Free-Text** | `134.9` / `186.4 tok/s` | **`212.1`** (`k2`) / **`250.9 tok/s`** (`k2`) | **`1.57×`** / **`1.35×`** | `k3/k4` regress on M4 Pro `B=4` (`M >= 16` verify tile) |
| **`B=1` Constrained JSON** | `32.8` / `46.8 tok/s` | **`97.1`** (`0.616 d/s`, `k4`) / **`121.4 tok/s`** (`0.755 d/s`, `k4`) | **`2.96×`** / **`2.60×`** | `exact_match = True` (`1.76×` vs stock `mlx-vlm` `69.0 tok/s`) |
| **`B=4` Constrained JSON** | `112.0` (`0.84 d/s`) / `149.8 tok/s` (`1.09 d/s`) | **`221.0`** (`1.62 d/s`, `k2`) / **`272.6 tok/s`** (`1.90 d/s`, `k2`) | **`1.97×`** / **`1.74×`** | `k2` keeps M4 Pro verify `M = 12 < 16` |

- **Batch-Regime Saturation Guard (`B >= 5` -> `0.56×` at `B=8`):** At `B >= 5`, batched speculative verification saturates `Apple9`/`Apple10` matmul tiles (`B=8` regresses to **`0.56×`** vs plain autoregressive `ragged_b_slots`), and on `Apple10 NAX`, `M >= 8` native TensorOps already accelerate `B >= 2` by `1.5×–1.9×`. [`DynamicSiliconPolicy`](../packages/fast-shared/src/fast_shared/runtime_lessons.py) and `fast-gemma4` (`mtp_drafter="auto"`) therefore auto-gate `MTP` to **`B == 1` on `Apple10+ NAX`** and **`1 <= B <= 4` on `Apple9`**, paying **`0 MB` drafter RAM** on `B >= 8` batch jobs. For multi-candidate shared-prefix verification (`shared_prefix_tokens >= 32`, `num_candidates >= 2`), `select_speculative_policy` enables prefix-tree KV + tree-mask coalescing (up to **`2.59×`** at `N=4..8` candidates without duplicating prefix KV).

### 6.2 `Kev-0.8B` `256`-Token `GDN + KV` `PrefixTree` (`max |Δlogit| = 0.0`, `max |Δprob| = 0.0`)

| Scenario | Base M5 Cold -> Warm (`PrefixTree`) | Base M5 Speedup | M4 Pro Cold -> Warm (`PrefixTree`) | M4 Pro Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **`B=1`, `550`-tok (`256`-tok hit)** | `124.9 ->` **`45.9 ms`** (`21.8 q/s`) | **`2.72×`** (`2.42×` vs stock `mlx-lm` `111.2 ms` / `8.99 q/s`) | `201.6 ->` **`44.4 ms`** (`22.5 q/s`) | **`4.54×`** (`4.14×` vs stock `mlx-lm` `183.9 ms` / `5.44 q/s`) |
| **`B=1`, exact `512`-tok hit** | `83.9 ->` **`21.6 ms`** (`46.3 q/s`) | **`3.88×`** (**`5.15×`** vs stock `mlx-lm` `B=1` `8.99 q/s`) | `128.2 ->` **`18.6 ms`** (`53.8 q/s`) | **`6.90×`** (**`9.89×`** vs stock `mlx-lm` `B=1` `5.44 q/s`) |
| **`B=1`, `1024`-tok hit** | `143.3 ->` **`22.6 ms`** (`44.3 q/s`) | **`6.34×`** | `232.2 ->` **`18.5 ms`** (`53.9 q/s`) | **`12.52×`** |
| **`8` Queries, Same Doc (branch)** | `984.3 ->` **`148.0 ms`** (`54.1 q/s`) | **`6.65×`** vs plain (**`6.19×`** vs stock `mlx-lm` `B=8` `8.73 q/s`) | `1,594.1 ->` **`257.9 ms`** (`31.0 q/s`) | **`6.18×`** vs plain (**`5.86×`** vs stock `mlx-lm` `B=8` `5.29 q/s`) |
| **`B=8` Shared Preamble (warm)** | `920.2 ->` **`354.1 ms`** (`22.6 q/s`) | **`2.60×`** (**`2.59×`** vs stock `mlx-lm` `B=8` `916.8 ms` / `8.73 q/s`) | `1,528.1 ->` **`308.0 ms`** (`26.0 q/s`) | **`4.96×`** (**`4.91×`** vs stock `mlx-lm` `B=8` `1,512.1 ms` / `5.29 q/s`) |

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
| **`fast-bge-small` (`9` buckets, M4 Pro)** | `26.80s` | `9.18s` | **`4.56s`** (`jobs=4`) | **`2.01×`** | **`5.88×`** | `cos = 1.000000` (`9/9` ANE) |
| **`fast-bge-small` (`9` buckets, Base M5)** | — | `5.12s` | **`2.45s`** (`jobs=4`) | **`2.09×`** | — | `cos = 1.000000` (`9/9` ANE) |
| **`fast-gemma4` `Early14` (`fp16 + w8`, M4 Pro)** | `65.27s` | `34.94s` (`26.37s` warm) | **`26.84s`** (`jobs=2`, Amdahl-bound by `w8`) | **`1.30×`** | **`2.43×`** | `2/2` bundles + manifest |

---

<a id="7-coreai-opt-track-a--os-coreai-track-b-promotion-audit-2222-gates"></a>

## 7. CoreAI audit of 22 decisions

The audit reproduced all 22 recorded decisions, including configurations that failed the quality or speed checks. Four decisions covered `CoreAI-Opt` on the M4 Pro and 18 covered Apple CoreAI on the M5. The table identifies which configurations passed, failed or still need a full test. The audit also recorded zero `fast-ir-guard` errors and confirmed the FP16 epsilon floor (`2^-14`) on all 33 Laya normalization layers.

### 7.1 Cross-Mac Promotion Gate Matrix

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
| **`Qwen3.5-4B`** | `43.1 tok/s` / `3.71 GB` | **`49.4 tok/s` (`1.15×`) / `2.91 GB` (`-22%`)** | `118.9` / `0` NaN | **`93.4` / `0` NaN** (`1.36×` warm smoke `1050` vs `1424 ms`, **`+18.8%` `EOT`**) | **Valid & faster (`+18.8%` `EOT`)** |

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
