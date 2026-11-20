# `fast-silicon` Cross-Silicon Benchmark Reference

Complete, audited benchmark evidence comparing **`fast-*` engines** against stock **`MLX`** (`mlx-embeddings`, `mlx-lm`, `mlx-vlm`, `pplx-context-mlx`) and stock **Apple `CoreAI`** (`.aimodel` + Swift `llm-runner`) across **Apple M4 Pro (`48 GB`)** and **Base Apple M5 (`24 GB`)**.

All measurements were executed in isolated subprocesses serialized under `/tmp/m4_gpu_validation.lock` via the four canonical repository harnesses: [`scripts/benchmark_outcomes.py`](../scripts/benchmark_outcomes.py), [`scripts/benchmark_monorepo.py`](../scripts/benchmark_monorepo.py), [`scripts/benchmark_capabilities.py`](../scripts/benchmark_capabilities.py), and [`packages/fast-pplx/scripts/benchmark.py`](../packages/fast-pplx/scripts/benchmark.py).

## Table of Contents

1. [Hardware Testbeds & Empirical Roofline](#1-hardware-testbeds--empirical-roofline)
2. [Methodology: `EOT`, Two-Gate Promotion & 4-Bucket Memory](#2-methodology-eot-two-gate-promotion--4-bucket-memory)
3. [External Deterministic Objective Outcomes (`Tier-1`, Cat 1–4)](#3-external-deterministic-objective-outcomes-tier-1-cat-14)
4. [Unified Monorepo Throughput, Latency & Footprint](#4-unified-monorepo-throughput-latency--footprint)
5. [Local Training & Post-Training (`QLoRA`, `ModernBERT`, `GRPO`)](#5-local-training--post-training-qlora-modernbert-grpo)
6. [Speculative Decoding, `PrefixTree`, ANE Profiles & Parallel CoreML](#6-speculative-decoding-prefixtree-ane-profiles--parallel-coreml)
7. [`CoreAI-Opt` (Track A) & OS `CoreAI` (Track B) Promotion Audit (`22/22` Gates)](#7-coreai-opt-track-a--os-coreai-track-b-promotion-audit-2222-gates)
8. [Reproducible CLI Invocations](#8-reproducible-cli-invocations)

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

**Roofline Synthesis:**
- **Where Base M5 (`10c` `Apple10 NAX`) wins (`1.06×–1.59×` over `20c` M4 Pro):** Any compute-bound dense or batched workload — batch encoders (`fast-embeddinggemma-2`, `fast-pplx`, `fast-bge-small`), batched `Kev-0.8B` (`1.56×–1.59×`), batched structured JSON (`fast-gemma4` `1.13×–1.52×`), `ModernBERT-421M` fine-tuning (`1.42×–1.59×`), and `GRPO` rollouts (`1.44×`).
- **Where M4 Pro (`20c` `Apple9`, `48 GB`) wins:** Bandwidth-bound `M=1` autoregressive decode (`fast-gemma4` Cat-3 reasoning `0.95×` on M5), `B=1` single-query ANE dispatch latency (`0.49–0.51 ms` vs `0.79 ms` on `fast-bge-small`), and high-memory jobs exceeding Base M5's `19.07 GB` Metal working-set cap (`fast-gpt-oss` `20B` MoE and packed `pack=2048` `E2B QLoRA`).

---

## 2. Methodology: `EOT`, Two-Gate Promotion & 4-Bucket Memory

### Effective Outcome Throughput (`EOT`)

Raw token throughput can rise while end-to-end utility falls (for example, quantization-induced repeat loops or invalid JSON outputs). To combine accuracy, structural validity, and wall-clock speed without squaring validity rate, every objective benchmark optimizes **Effective Outcome Throughput** (`outcome_units/sec`):

```text
EOT = (1 / T_e2e) * sum_i(valid_i * score_i)
    = (N_valid / T_e2e) * S_obj|valid
    = (N / T_e2e) * S_obj,e2e
```

where `valid_i ∈ {0, 1}`, `score_i ∈ [0, 1]` is computed by a 100% deterministic, zero-LLM-judge evaluator over un-gated, permissively licensed (`Apache-2.0`, `MIT`, `CC-BY`, `ODC-BY`, `OpenMDW-1.1`) public datasets, and `T_e2e` is full wall-clock time.

### Three-Gate Promotion Contract (`evaluate_promotion_gates`)

A candidate engine or quantization preset promotes over its baseline in [`scripts/benchmark_outcomes.py`](../scripts/benchmark_outcomes.py) only when all three gates pass under interleaved `A-B-B-A` execution:

1. **Gate 1 — Objective Quality Non-Inferiority:**
   - `lossless` profile: `ΔS_obj,e2e >= -max(0.003, 1.96 * SE_paired)` and `repeat_loop_count == 0`.
   - `quantized` profile: `ΔS_obj,e2e >= -0.010` and bounded grammar mask fallback rate.
2. **Gate 2 — Unified-Memory Headroom (`<= 80%` UMA Ceiling):**
   - `FourBucketMemoryProbe` (`RUSAGE_INFO_V6` in [`packages/fast-shared/src/fast_shared/bench.py`](../packages/fast-shared/src/fast_shared/bench.py)) measures four orthogonal unified-memory buckets:
     1. `mlx_peak_gb` — active Metal GPU allocator peak (`mx.get_peak_memory()`),
     2. `phys_footprint_gb` — OS process physical footprint (`ri_phys_footprint`),
     3. `neural_nofootprint_gb` — ANE accelerator memory outside process footprint (`ri_neural_footprint` via `RUSAGE_INFO_V6`),
     4. `model_mapped_gb` / `mapped_all_mb` — clean mmap weight pages (including OS-specialized `~/Library/Caches/coreai-cache/<os_build>/.../model.aimodelx/` pages).
   - Enforces `combined_with_model_mapped_gb <= 0.80 * host_mem_gb` (`<= 38.4 GB` on M4 Pro, `<= 19.2 GB` on Base M5) and `DeviceMemoryGuard` (`0.98 * recommendedMaxWorkingSetSize`) untripped. Upgrading from `RUSAGE_INFO_V4` to `RUSAGE_INFO_V6` also eliminates a macOS `27.x` CoreAI accounting artifact that previously reported `385.8 GB` on `coreai-emb2-bf16` (true `RUSAGE_INFO_V6` footprint: `3.04 GB` combined, `2.38 GB` `mapped_all_mb`).
3. **Gate 3 — Outcome Throughput Ratio:**
   - `EOT_candidate / EOT_baseline >= 1.00` (`>= 1.03` when introducing custom kernel or stateful complexity).

### Serialization & Stock Baseline Contract

- Every benchmark runs in an isolated subprocess under `fcntl.flock` (`/tmp/m4_gpu_validation.lock`) with `check_system_quiescence()`, explicit `mx.set_wired_limit(max_recommended_working_set_size)`, and thermal cooldowns.
- Every `fast-*` package is paired on identical datasets and evaluators against its fastest native non-`fast-*` stock harness: stock Apple `CoreAI` (`.aimodel` on `macOS 27.2.1`) and stock HuggingFace `MLX` (`mlx-embeddings`, `mlx-vlm`, `mlx-lm`, and `pplx-context-mlx` `fp16`).

---

## 3. External Deterministic Objective Outcomes (`Tier-1`, Cat 1–4)

### Category 1 — Retrieval & Contextual Embeddings (`NanoBEIR-en` 3-Subset + `LegalBench-RAG` `cuad+contractnli`)

`NanoBEIR-en` (`NanoSciFact`, `NanoArguAna`, `NanoHotpotQA`) + `LegalBench-RAG` (`cuad`, `contractnli`; `4K` contextual chunk windows for `fast-pplx` / `pplx-context-mlx`, `2K` for `EmbeddingGemma-2`).

| Workload (`N`) | Engine / Stock Harness | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **`BGE-small`** (`325`) | **`fast-bge-small` (`hetero`)** | `0.4737` | **`13.061`** (`11.8s`) | `0.99 / 1.13 / 0.41 /` **`1.54 GB`** | `0.4737` | **`12.634`** (`12.2s`) | `1.19 / 1.15 / 0.41 /` **`1.55 GB`** | **`11.26×`** / **`9.58×`** | **`1.03×`** |
| `BGE-small` Stock `MLX` | `mlx-community/bge-small-en-v1.5-8bit` | `0.4774` | `1.160` (`133.8s`) | `2.76 / 18.78 / 0.00 / 18.78 GB` | `0.4774` | `1.319` (`117.6s`) | `2.93 / 38.99 / 0.00 / 38.99 GB` | `1.00×` | `0.88×` |
| **`EmbeddingGemma-2`** (`500`) | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5376`** | **`4.707`** (`57.1s`) | `1.38 / 2.77 / 0.66 /` **`3.43 GB`** | **`0.5365`** | **`3.583`** (`74.9s`) | `1.46 / 2.38 / 0.11 /` **`2.49 GB`** | **`4.33×` MLX (`6.24×` CoreAI)** / **`4.80×`** | **`1.31×`** |
| `EmbGemma-2` Stock `MLX` | `mlx-community/embeddinggemma-2-8bit` (`mlx-vlm`) | `0.5364` | `1.086` (`246.9s`) | `2.58 / 19.41 / 0.00 / 19.41 GB` | `0.5353` | `0.746` (`358.6s`) | `2.84 / 39.03 / 0.00 / 39.03 GB` | `1.00×` | `1.46×` |
| `EmbGemma-2` Stock `CoreAI` | `coreai-emb2-bf16` (`text_encoder.aimodel`, `27.2+`) | `0.5360` | `0.754` (`355.5s`) | `0.00 / 2.90 / 0.00 / 3.04 GB`* | — *(26.6)* | — *(26.6)* | — | `0.69×` MLX | — |
| **`pplx-embed-0.6B`** (`325`) | **`fast-pplx` (`hetero2` ctx `4K`)** | **`0.7244`** | **`1.724`** (`136.5s`) | `2.48 / 1.86 / 0.23 /` **`2.34 GB`** | **`0.7241`** | **`1.407`** (`167.3s`) | `2.53 / 1.42 / 0.23 /` **`1.97 GB`** | **`4.41×`** / **`4.33×`** | **`1.23×`** |
| `pplx-embed` Stock `MLX` | `zainmerchan/pplx-embed-v2-context-0.6b-mlx` (`fp16`) | `0.7236` | `0.391` (`601.5s`) | `5.69 / 2.12 / 0.00 / 2.12 GB` | `0.7236` | `0.325` (`723.1s`) | `5.69 / 2.19 / 0.00 / 2.19 GB` | `1.00×` | `1.20×` |

*\*`coreai-emb2-bf16` true `RUSAGE_INFO_V6` telemetry (`2.90 GB` phys / `3.04 GB` combined, `2.38 GB` `mapped_all_mb`) verified in Track B; `fast-pplx` `combined` includes `0.25–0.31 GB` `model_mapped_gb` for `MappedEmbedding` safetensors pages.*

### Category 2 — Decision, Faithfulness & Reward (`RAGTruth` `600` + `CLINC150` `550` + `RewardBench-2` `250`, `N=1400`)

| Workload (`N=1400`) | Engine / Stock Harness | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall) | Base M5 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall) | M4 Pro 4-Bucket Mem (`Metal / Phys / ANE / Comb`) | `fast-*` vs. Stock (`M5` / `M4 Pro`) | M5 vs. M4 Pro (`EOT`) |
| :--- | :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **`Laya-Typed-421M`** | **`fast-decision-models` (`laya_ane`)** | **`0.5050`** | **`6.158`** (`114.8s`) | `0.00 / 0.60 / 0.40 /` **`1.00 GB`** | **`0.5050`** | **`5.827`** (`121.3s`) | `0.00 / 0.58 / 0.40 /` **`0.98 GB`** | **`8.29×` CoreAI (`1.16×` MLX\*)** / **`1.92×` MLX\*** | **`1.06×`** |
| `Laya` Stock `CoreAI` | `ahmedandaloes/laya-typed-decisions-CoreAI` (`27.2+`) | `0.4993` | `0.743` (`940.7s`) | `0.00 / 0.29 / 0.00 / 0.33 GB`† | — *(26.6)* | — *(26.6)* | — | `1.00×` *(accurate stock)* | — |
| `Laya` Stock `MLX`\* | `mlx-embeddings` `ModernBERT 8-bit` | `0.2757`\* | `5.319` (`72.6s`) | `1.34 / 18.70 / 0.00 / 18.70 GB` | `0.2986`\* | `3.032` (`137.9s`) | `1.54 / 34.02 / 0.00 / 34.02 GB` | *(`-0.23` quality loss, `19×–34×` RAM)* | `1.75×` |
| **`Kev-0.8B`** (single-Q) | **`fast-decision-models` (`kev_08b_w8`)** | **`0.5257`** | **`5.078`** (`144.9s`) | `3.70 / 1.76 / 0.00 /` **`1.77 GB`** | **`0.5250`** | **`3.254`** (`225.9s`) | `3.78 / 1.84 / 0.00 /` **`1.84 GB`** | **`1.005×` (`-42%` RAM; `2.59×–6.19×` w/ `PrefixTree`)** / **`1.06×` (`-44%` RAM; `4.91×–9.89×` w/ `PrefixTree`)** | **`1.56×`** |
| `Kev-0.8B` Stock `MLX` | `AbstractFramework/kev-0.8b-mlx-8bit` (`mlx-lm`) | `0.5236` | `5.055` (`145.0s`) | `2.65 / 3.06 / 0.00 / 3.06 GB` | `0.5229` | `3.062` (`239.1s`) | `2.92 / 3.26 / 0.00 / 3.26 GB` | `1.00×` | `1.65×` |

*\*Stock `mlx-embeddings` `ModernBERT 8-bit` suffers `-0.23` objective quality collapse and `18.7–34.0 GB` RAM growth; `coreai-laya` is the only accurate stock baseline (`S=0.4993`). †`coreai-laya` additionally maps `1.68 GB` `mapped_all_mb` from `~/Library/Caches/coreai-cache/`.*

### Categories 3 & 4 — Reasoning (`IFBench` + `MMLU-Redux` + `MATH-500`) and Structured JSON & Tool Calling (`LiquidAI/structured-extraction` + `BFCL v3`)

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

---

## 4. Unified Monorepo Throughput, Latency & Footprint

Isolated subprocess `ABBA` sweeps from [`scripts/benchmark_monorepo.py`](../scripts/benchmark_monorepo.py) + [`scripts/public_baselines.py`](../scripts/public_baselines.py) (with materialized sync in `mlx-embeddings` and `RUSAGE_INFO_V4`/`V6` OS footprint).

| Package | Metric | Base M5 `fast-*` (OS Footprint) | Base M5 Stock Baseline | M5 Speedup | M4 Pro `fast-*` (OS Footprint) | M4 Pro Stock Baseline | M4 Pro Speedup | M5 vs. M4 Pro (`fast-*`) |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **[`fast-bge-small`](../packages/fast-bge-small/README.md)** | Corpus `tok/s` (`B=1` ANE ms) | **`200,646 tok/s`** (`0.79 ms`, `1.85 GB`) | `mlx-embeddings`: `26,667 tok/s` (`4.42 ms`) | **`7.52×`** (`5.59×` `B=1`) | **`187,150 tok/s`** (`0.49 ms`, `2.09 GB`) | `mlx-embeddings`: `30,694 tok/s` (`4.85 ms`) | **`6.10×`** (`9.90×` `B=1`) | **`1.07×`** |
| **[`fast-embeddinggemma-2`](../packages/fast-embeddinggemma-2/README.md)** | Corpus `tok/s` (`B=1` ANE ms) | **`48,803 tok/s`** (`2.58 ms`, `2.76 GB`) | `mlx_vlm`: `16,094 tok/s` (`8.48 ms`) / `CoreAI`: `12,482` (`9.38 ms`) | **`3.03×` MLX / `3.91×` CoreAI** | **`38,517 tok/s`** (`2.11 ms`, `3.26 GB`) | `mlx_vlm`: `11,419 tok/s` (`8.39 ms`) | **`3.37×`** (`3.98×` `B=1`) | **`1.27×`** |
| **[`fast-pplx`](../packages/fast-pplx/README.md)** | Mixed / `doc_s256` `tok/s` | **`16,796 tok/s`** mixed (`5.57 ms` `B=1`, `3.13 GB`) | `pplx-context-mlx` (`fp16`): `1.32×` slower on `doc_s256` | **`1.32×`** | **`14,330 tok/s`** `doc_s256` (`7.45 ms` ANE / `4.97 ms` GPU, `3.10 GB`) | `pplx-context-mlx` (`fp16`): `8,633 tok/s` | **`1.66×`** | **`1.17×`** |
| **[`fast-decision-models`](../packages/fast-decision-models/README.md)** | `kev_08b B=8` / `laya_ane` | **`86.84 q/s`** / **`44.38 q/s`** (`2.20 GB`) | `stock-mlx_lm-kev-8bit`: `81.7 q/s` (`-0.18 GB` OS) | **`1.06×`** | **`54.52 q/s`** / **`41.91 q/s`** (`2.46 GB`) | `stock-mlx_lm-kev-8bit`: `50.6 q/s` (`-0.25 GB` OS) | **`1.08×`** | **`1.59×`** (`Kev`) |
| **[`fast-gemma4`](../packages/fast-gemma4/README.md)** | Schema-valid `docs/s` (`dec tok/s`) | **`10.46 valid docs/s`** (`1,479 tok/s`, `15.54 GB`) | `mlx_vlm`: `4.04 raw` / **`2.02 valid docs/s`** (`50%` schema) | **`2.59×` raw / `5.18×` valid** | **`6.87 GPU` / `8.78 ANE+GPU valid docs/s`** (`1,144` / `1,131 tok/s`, `18.96 GB`) | `mlx_vlm`: `3.60 raw` / **`1.80 valid docs/s`** (`32/64` valid) | **`1.91×–2.44×` raw / `3.82×–4.88×` valid** | **`1.52×`** (`GPU`) / **`1.19×`** (`peak`) |
| **[`fast-gpt-oss`](../packages/fast-gpt-oss/README.md)** | `B=128` `dec tok/s` (`docs/s`) | *Gated (`24 GB`)* | *Gated (`24 GB`)* | — | **`454.1 dec tok/s`** (`1.483 docs/s`, `23.98 GB`) | `stock-mlx_lm B=8`: `156.1 tok/s` | **`2.91×`** | *48 GB only* |

---

## 5. Local Training & Post-Training (`QLoRA`, `ModernBERT`, `GRPO`)

All training suites ran in isolated subprocesses via [`scripts/benchmark_capabilities.py`](../scripts/benchmark_capabilities.py) with step-0 loss and gradient cosine parity verified against stock MLX baselines.

### 5.1 Dense Encoder Fine-Tuning (`ModernBERT-421M`, `encoder-finetune`, `freeze < 18`)

| Regime | Base M5 (`24 GB`, `10c NAX`) | Base M5 Peak Mem | Base M5 vs. Stock | M4 Pro (`48 GB`, `20c Apple9`) | M4 Pro Peak Mem | M4 Pro vs. Stock | M5 vs. M4 Pro |
| :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: |
| Stock Unpacked (`FP32` head) | `29.63 items/s` | `5.15 GB` | `1.00×` | `20.25 items/s` | `5.22 GB` | `1.00×` | `1.46×` |
| `BF16` + `@mx.compile` | `34.68 items/s` | `4.97 GB` | `1.17×` | `21.17 items/s` | `5.08 GB` | `1.05×` | `1.64×` |
| **`Packed-512` (FFD Block-Diag)** | **`42.26 items/s`** | **`4.90 GB`** | **`1.43×`** | **`26.63 items/s`** | **`5.06 GB`** | **`1.32×`** | **`1.59×`** |
| **`Packed-512` + Cached `L0..17`** | **`58.19 items/s`** | **`4.89 GB`** | **`1.96×`** | **`40.84 items/s`** | **`5.06 GB`** | **`2.02×`** | **`1.42×`** |

- **Parity:** `grad_cos = 0.999942` (`|Δloss| = 1.1e-4`) on Base M5; `grad_cos = 0.999946` (`|Δloss| = 2.2e-5`) on M4 Pro.
- **Takeaway:** `10c` Base M5 `NAX` trains `ModernBERT-421M` **`1.42×–1.59×` faster** than `20c` M4 Pro in `< 4.9 GB` unified memory.

### 5.2 Causal `QLoRA` + `FFD` Packing + Shared-System KV (`Gemma-4 E2B`, `qlora-packing`)

| Regime (`M4 Pro 48 GB`) | Throughput (`tok/s`) | Peak Metal (GB) | Speedup vs. Stock `mlx-lm` | Gradient Parity |
| :--- | ---: | ---: | :---: | :--- |
| Stock `mlx-lm` BF16 LoRA (unpacked) | `579.1` | `16.09 GB` | `1.00×` | Reference |
| `8-bit g64` `QLoRA` + `pack=2048` (`FFD`) | `545.8` | `25.06 GB` | `0.94×` | `packed_grad_cos = 0.999673`, `qlora_8bit_grad_cos = 0.994881` |
| **`8-bit g64` `QLoRA` + `shared-sys-kv`** | **`1,385.4`** *(up to `1,581` on `1,288`-tok schemas)* | `25.82 GB` | **`2.39×`** *(up to **`2.88×`**)* | Frozen trunk cut `9.23 -> 5.00 GB`; `min_tensor_cos = 0.9991` |

- **`24 GB` Base M5 Auto-Sizing (`recommend_silicon_budget`):** Unmodified `pack=2048` (`25.06 GB` peak Metal) exceeds Base M5's `19.07 GB` Metal working-set limit (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). [`DynamicSiliconPolicy`](../packages/fast-shared/src/fast_shared/runtime_lessons.py) automatically scales `<= 24 GB` hosts to **`pack_len=1024, grad_accum=2, ckpt_tokens=512, memory_limit_gb=19.2`** (`<= 80%` wired ceiling), while `>= 36 GB` hosts run `pack_len=2048, grad_accum=1, ckpt_tokens=4096, memory_limit_gb=24.0`.
- **P2.8 ANE Frozen-Prefix Prefetch (`Gemma-4 E2B` `L0..11` ANE `[1650,0,0]` + `L12..34` GPU LoRA on M4 Pro):** Hides **`99.3%–99.8%`** of frozen prefix time (`84.9–374.0 ms -> 0.56–0.83 ms` exposed), yielding **`1.086×` (`B=1`) / `1.088×` (`B=2`)** end-to-end LoRA step speedup with `h_12` cosine `0.999124` and adapter gradient cosine `0.996968`.

### 5.3 Disaggregated `GPU Actor + ANE Critic` `GRPO` & Best-of-N (`G=4`, `grpo-ane-critic`)

`fast-gemma4` rolling GPU actor (`llguidance` schema-constrained) overlapped with `fast-decision-models` `Laya-Typed-421M` Pure-ANE critic (`[1585,0,0]` ANE ops, **`0.0 GB` GPU critic VRAM**) vs. stock `mlx-vlm` sequential rollouts + GPU `Kev-0.8B 8-bit` critic.

| Host | Mode | Rollout + Critic (`rollouts/s`) | Speedup vs. Stock | Exposed ANE Critic During GPU VJP | Peak Mem (GB) | Full `GRPO` Step Speedup |
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

## 7. `CoreAI-Opt` (Track A) & OS `CoreAI` (Track B) Promotion Audit (`22/22` Gates)

Independent cross-stream audit across **Track A (M4 Pro `48 GB`, `macOS 26.6`, `CoreML 9.0`, `coreai-opt 0.5.0`)** and **Track B (Base M5 `24 GB`, `macOS 27.2.1`, `CoreAI 1.0.0b3`)** verified **`22/22` `evaluate_promotion_gates` verdicts** (`4` Track A + `18` Track B, `0` mismatches), `0` `fast-ir-guard` errors, `33/33` `Laya` LayerNorms clamped to `eps_floor_fp16 = 2^-14` (`6.103515625e-5`), and `0` NaNs across all promoted/candidate evaluations.

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
| **Track B** Base M5 `27.2.1` | `LayaCoreAiModel w8` & `w4_pb` (FP32) vs. custom `LayaAneBackend` (`N=42`) | `quantized` (`2` gates + `2` `w4_pb` stock) | `0.3571` vs `0.3571` (`+0.0000`) | **T** | `0.569 / 19.2 GB` | **T** | `1.053 / 3.696 = 0.2849` (`0.273×` `w4_pb`) | **F** | FP32 compute forces GPU fallback (`94.2–102.8 ms` vs `22.64 ms` custom ANE) | **FAIL** *(vs custom ANE)* |
| **Track B** Base M5 `27.2.1` | **`LayaCoreAiModel w8_f16` + `neural_engine`** (`S=256` ablation) | ablation | `logit_cos = 0.999831`, `argmax = 1.0`, `0` NaN | — | **`0.444 / 19.2 GB`** (`phys=62.3 MB`, `neural=360.8 MB`) | **T** | **`23.00 ms` iso** (`43.5 q/s`; `27.22 ms` shared) vs `22.64 ms` custom ANE & `354.1 ms` stock | — | **Breakthrough:** **`1,652` ANE ops** once exported in `FP16`; `4.09×` faster than FP32 `.aimodel` and `15.4×` faster than stock `CoreAI` | **PROMISING** *(needs `S=512` full gate)* |

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

# Run paired apples-to-apples stock MLX and macOS 27.2+ CoreAI baselines
.venv/bin/python scripts/benchmark_outcomes.py run \
  --tier tier1 --packages all --include-public-baselines \
  --public-python /path/to/stock-mlx-venv/bin/python \
  --coreai-python /path/to/coreai-venv/bin/python \
  --pplx-mlx-path /path/to/pplx-context-mlx \
  --json-out artifacts/outcomes_tier1_with_baselines.json

# Two-Gate + 80% wired-memory promotion check between two saved reports
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
