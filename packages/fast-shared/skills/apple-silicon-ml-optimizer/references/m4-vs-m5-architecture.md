# Apple M4 vs M5+ Silicon Architecture & Model Roofline Matrix

Use this reference to translate chip capabilities and model geometry into the right engine, batch size, and quantization layout.

---

## 1. Measured hardware and runtime support

### Verified M4 Pro (`48 GiB`, `20`-core Metal GPU, `16`-core ANE, `14`-core `10P+4E` CPU w/ 2 SME2 units)
Measured cleanly under quiesced conditions (`macOS 26.6`, `Xcode 27.2`, `clang++ 21.0`):

| Subsystem | Verified Empirical M4 Pro Measurement | Key Operational Constraint |
|---|---|---|
| **GPU DRAM BW** | **`192–215 GB/s`** streaming read (`metal_f4_read`: `196.6` @ `256 MB`, `214.8` @ `512 MB`; `mx.sum` `191.9–211.5`; `2x` copy `172.7–195.7 GB/s`) vs `273 GB/s` spec | GPU DRAM BW caps around **`~70%–79%`** of nominal spec (`273 GB/s` is shared across GPU + ANE + CPU). |
| **GPU Matrix (`Apple9`)** | MPP `matmul2d` reached 7.61 TFLOP/s (BF16), 7.59 (FP16), 7.41 (FP16 × INT4) and 7.35 (FP16 × INT8). MPSGraph BF16 reached 7.49, `steel_gemm` 7.34–7.40, and MXFP4 MoE 6.15 TFLOP/s. | M4 runs `MPP::matmul2d` using GPU shader instructions. Custom `64×64` kernels reached 5.92–6.32 TFLOP/s with half accumulators and 5.44–5.64 TFLOP/s with float accumulators. |
| **Empirical GPU Ridge** | **`~34–38.4 FLOP/B`** (`7.38 TF / 192–215 GB/s`), shifting `BF16` vs `Q4` crossover up to **`B = 48..64`** | Do **not** assume datasheet ridge `7.61 / 0.273 = 27.9 FLOP/B`. |
| **ANE (`16`-core)** | **`18.56 TF`** (`conv8_w8a8_M128`), **`16.74 TF`** (`conv8_w8_M64`), **`15.73 TF`** (`conv1_M128_fp16`), **`12.58 TF`** sustained multi-layer `conv1x1` | Must use 4D `conv1x1 NCHW [1, C, 1, S]` + static buckets + `pipe_depth=4`. MIL `matmul` falls back (`0.42 TF`), while dynamic `mb.matmul` hits `4.34 TF`. |
| **CPU SME2 (`2` units)** | **`3.22–3.27 TF`** (`cblas_sgemm FP32` / `BNNSMatMul BF16`), **`7.24 TOp/s` `INT8` (`2` workers)**, **`5.30 TF` `BNNSGraph W8`**, **`2.25 eff TF` `MXFP4->BF16`** | Plateaus at `2` workers (2 physical shared SME2 units). Fights GPU on `273 GB/s` DRAM during `B=1` decode (`2.46×` slowdown). |
| **3-Way Concurrency & Thermal** | Burst GEMMs (`7.46 + 12.35 + 3.08 = 22.89 TF`, `3.08×` solo GPU). Sustained `10–65s+`: GPU scales **`-25%` to `-33%`** (shared SoC TDP + heat soak), ANE `-25.4%`. | Multi-minute GPU saturation (`5–15 min`) heavily throttles GPU clocks; always cool or interleave `ABBA` runs. |

### Metal features by macOS version

Check the running OS, Metal GPU family and installed MLX version independently. M4 Pro reports `Apple9` and runs Metal Performance Primitives (MPP) on GPU shaders. M5 reports `Apple10` and adds a matrix accelerator to each GPU core. Core ML can also use a separate Apple Neural Engine (ANE) on both chips. [Apple GPU families](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) · [Apple’s TensorOps architecture and release history](https://developer.apple.com/videos/play/tech-talks/111432/)

| macOS | What changes for these kernels |
|---|---|
| 26.0 / 26.1 | Metal 4 introduced TensorOps for inline matrix operations. The 26.1 update added BF16 tensor operands. The M4 Pro’s packed embedding projections use FP16. [Metal 4 introduction](https://developer.apple.com/videos/play/wwdc2025/262/) · [TensorOps updates](https://developer.apple.com/videos/play/tech-talks/111432/) |
| 26.2 | MLX can dispatch supported matrix operations to the M5 GPU accelerators. On Apple9, MLX uses its M4 GPU paths. [Apple’s MLX research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [MLX 0.32.4 device check](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L978-L995) |
| 26.3 | Matrix multiply accepts cooperative tensors as inputs. To reuse a previous operation’s output directly, check its layout with `is_compatible_as_left_input` or the right-input equivalent. Use threadgroup memory for incompatible layouts. The M4 probe below verified direct reuse on 26.6. [TensorOps updates](https://developer.apple.com/videos/play/tech-talks/111432/) · [Apple’s layout check and fallback](https://developer.apple.com/videos/play/wwdc2026/330/?time=753) |
| 26.4 | TensorOps added four- and eight-bit integer tensor operands. M4 Pro can consume packed weights directly on its GPU shader path. Compare with MLX quantized matrix multiply at the encoder’s shapes and include the model’s block scales and offsets. [Apple’s TensorOps update](https://developer.apple.com/videos/play/tech-talks/111432/) |
| 27 | Metal adds FP4, FP8 and two-bit integer tensor formats. Check support on the deployment OS. An installed 27 SDK can target a Mac still running 26.6. [Apple’s quantized tensor guide](https://developer.apple.com/videos/play/wwdc2026/330/) |

### M4 Pro encoder opportunities

The November 19, 2026 checks ran on the 20-core M4 Pro GPU with macOS 26.6, an installed 27.2 SDK and MLX 0.32.4. A runtime query reported `Apple9=true`, `Apple10=false` and `Metal4=true`.

| Encoder | Existing path | What carried over from M5 |
|---|---|---|
| BGE-small, 384 dimensions | The packed GPU path already fuses bias, residuals and GELU with 64-column FP16 matrix tiles. | Widening these tiles to 128 output columns produced eight bit-identical vectors on eight real OCR passages with routing fixed to the GPU. |
| EmbeddingGemma 2, 512 dimensions | The GPU uses custom FP16 MPP for plain projections and fused activations. The parallel ANE path already uses eight-bit weights. | Replacing only the plain projections with `mx.matmul(x, w.T)` gave a minimum cosine of 0.999999 on the same eight passages. Keep the fused operations and existing scheduler in a throughput comparison. |

These are M4 correctness results. The additional elapsed-time savings for these two changes and their held-out M4 retrieval quality remain unmeasured. The earlier M4 Gemma experiment already compared GPU-only with GPU+ANE on 512 passages and found the mixed path 2.49× faster. For Perplexity, the M4 run already tested merging the two feed-forward projections. Adding the merge to the fused convolution took 17.43 s versus 17.27 s for the convolution path alone, with identical vectors on all 384 passages.

**Two GPU features are available to the M4 encoders today.** A 64×64 FP16 multiply with direct signed INT4 and INT8 weights matched the CPU exactly for each format. This checks packed integer arithmetic. A model test also needs its quantization scales, real weights and passage ranks. A separate two-multiply probe reused a compatible cooperative tensor directly and matched the CPU. The running compiler required **one SIMD group** (32 GPU threads) and a **compile-time inner dimension** for the cooperative input. It rejected four groups and, separately, a dynamic inner dimension. The encoders’ fused MPP projections use four or eight groups and a dynamic inner dimension. Try a fixed tile in packed attention if profiling shows the intermediate store matters.

The `spicy-stack` workspace retains the capability checks and encoder outputs in `output/m4-encoder-os-20261119/`, the earlier Gemma comparison in `output/fast-encoder-hypotheses-20261116/RESULTS.md`, and the Perplexity comparison in `output/ocr-chunk-research-20261118/pplx-metal/RESULTS.md`.

### M5 hardware and runtime support

Query `[device supportsFamily:MTLGPUFamilyApple10]` for the M5 hardware family, then check MLX dispatch for the specific shape and dtype. [MLX matrix dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L962-L967)

Apple rates the base M5 at 153 GB/s of memory bandwidth and the M4 Pro at 273 GB/s. The base M5 worker had 10 GPU cores, 24 GiB of unified memory, macOS 26.2 and MLX 0.32.4. Its quick probe measured **123.4 GB/s** streaming reads and **11.66 TFLOP/s** at best in the matrix sweep (`K=512`, `N=1024`). That gives a rough compute-to-bandwidth crossover of **94.5 FLOP/byte** (`1000 × TFLOP/s ÷ GB/s`). [M5 specifications](https://support.apple.com/en-us/125405) · [M4 Pro specifications](https://support.apple.com/en-us/121553)

Repeat the measurements at the model’s actual shapes and on each chip variant. For small batches, compare native MLX with Metal Performance Primitives (MPP). When offloading to the ANE, time both processors working together. [Matrix and bandwidth probe](../scripts/probe_silicon_ceilings.py) · [Concurrency probe](../scripts/probe_ane_and_hetero.mm)

### Base M5 OCR embeddings

These same-machine comparisons ran on November 19, 2026. Each timed file test ran baseline, candidate, candidate, baseline after warmup. Times are the means per configuration, from Parquet read through final vector write. The retrieval test used 7,974 passages from 210 documents, fixed query vectors and ranks deduplicated by source passage. Its 40 development and 24 held-out questions targeted disjoint documents.

| Encoder and candidate | Input | Baseline → candidate | Search or output check |
|---|---|---|---|
| BGE-small, 384 dimensions. Widen the fused FP16 matrix tiles from 64 to 128 output columns. | 7,974 passages | 9.54 → 8.53 s, **10.6% less time** | All 64 target passage ranks matched. Held-out hits were 15/24 first and 20/24 in the top five. |
| BGE-small, same tiles starting from OCR page text. Includes passage splitting, tokenization and vector writes. | 117 documents in one catalog row group | 6.39 → 5.87 s, **8.2% less time** | Both produced 3,705 vectors in the same passage ID order. Minimum paired cosine was 0.999985. |
| EmbeddingGemma 2, 512 dimensions. Use `mx.matmul(x, w.T)` for plain GPU projections and keep the fused activations and GPU/ANE scheduler. | 7,974 passages | 54.28 → 46.20 s, **14.9% less time** | Both configurations retrieved a target for the same queries at each cutoff. Held-out hits were 17/24 first and 22/24 in the top five. |
| Perplexity `pplx-embed-v2-context-0.6b`, 1,024 dimensions. Combine two FP32 feed-forward projections. | 384 passages | 11.96 → 12.05 s | All 384 output vectors matched exactly. The existing path was faster. |

For **BGE**, try `128×128` tiles with eight SIMD groups for the combined query, key and value projection. Try `64×128` tiles with four SIMD groups for attention output and the feed-forward up and down projections. Tile sizes are token rows × output columns. Sweep the shapes with the real weights and time the full encoder with its fused operations. [Apple’s MPP tuning guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

For **Gemma**, a faster GPU changes which passages reach the eight-bit ANE path. The lowest mixed-run cosine was 0.998168. Replaying the eight largest differences on a fixed GPU raised the minimum to 0.999998. For all eight, the closest fixed-processor match switched between GPU and ANE across the two mixed runs. Two runs of the baseline also put 370 vectors below 0.9998. Repeat the baseline, replay outliers on each processor, and check held-out passage ranks before attributing drift to GPU math.

For **Perplexity**, a separate eight-passage pilot of `mx.fast.gated_delta_update` took 279 ms versus 284 ms for the current update. It changed 242 of 8,192 integer coordinates by one. Keep the current update when exact codes matter. The native update’s retrieval quality remains unmeasured. [MLX gated-delta dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/gated_delta_update.cpp)

Two follow-up experiments remain open. Profile BGE’s packed attention with its 32-wide heads. MLX 0.32.4’s main M5 full-attention path lists widths 64, 96, 128, 256 and 512. If attention dominates, test a fused kernel that preserves passage boundaries. For Gemma, compare eight-bit GPU weights on a supporting runtime against the faster FP16 MLX path using the same held-out queries. [MLX attention dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L410-L425) · [Apple’s fused-attention design](https://developer.apple.com/videos/play/wwdc2026/330/)

The original report in the `spicy-stack` workspace is `output/m5-architecture-research-20261119/RESULTS.md`. The same directory holds the individual timings, passage ranks and benchmark scripts. For the separate M4 Pro versus base M5 comparison of the original encoders, see `output/m5-speed-20261119/results.json`.

---

## 2. Paradigm Flip: Why Model Architecture Dictates Optimization Sign

The following measurements come from the M4 Pro. Use them to choose candidates for the target model and chip.

| Lever | Dense Encoders (`33M–300M`, e.g. `bge-small`, `emb-gemma-2`) | Dense KV-Shared LLMs (`2B–4B`, e.g. `Gemma-4 E2B/E4B`) | Sparse MoE LLMs (`20B–35B`, e.g. `gpt-oss-20b`, `26B-A4B`) |
|---|---|---|---|
| **Dominant Regime** | Packed varlen prefill (compute-bound, `M<=8192` SLC fit) | `BF16/Q8` MLP DRAM (`2,970 MB/step`) + Full SDPA (`7× D=512`) + prefill pause | Prefill MoE compute (`51%+` E2E!) + low-`B` active expert DRAM reads (`B=1` fast!) |
| **ANE Offload** | **King (`2.75×–3.56×`)**: `W8 conv1x1 S<=512` FFD + work-stealing `hetero`/`hetero2` | Mixed: `Early14 FP16` wins static fixed weights (`3.47 d/s`), **loses `2.5×`** to rolling GPU on LoRA variable docs | **Anti-Target (~`4×` slower)**: static ANE evaluates all `32E`/`128E` (`1.57` / `0.79 TF` eff.) |
| **Hydragen 2-Pass Prefix Split** | N/A | **Loses `~2×`** (`MQA-1`, KV sharing; use 1-pass `bfloat4` ragged SDPA instead) | **Wins `1.71×` + `-4.35 GB`**: `64Q:8KV` (`8:1` GQA) packs `simdgroup_matrix` on SLC `k_sys` |
| **`KV_i8` Cache Quantization** | N/A | **Wins** (`hd=256/512`: `-1.21 GB`, neutral/faster speed once LoRA-aligned) | **Loses on `hd=64`** (`-3.4%` `B=1`, `-42%` `pl=32`: ALU dequant > tiny KV read saved) |
| **Speculative / `PromptLookup`** | N/A | **Wins on `B<=8` tails** (`+23.8%` `B=1`, `+37.9%` `B=8`, `+46.1%` `Skew16`) | **Loses on `B=1` MoE** (`K=3/4` `-10.5%` to `-36.4%`, `EAGLE-3` `0.29–0.68×`: coupon collector touches `~3×` experts!) |
| **Vocab Pruning** | Corpus-exact (`52K/262K` `-39%` weights, `0` miss) | Byte-safe tied (`150,786`, `-42%` embed/head/`PLE`) | Untied output-only (`131,072` `-34.8%` `lm_head`, keeps full `201K` input!) |
| **Activation Dtype** | `FP16` (protect L2-pool in `FP32`) | **Forced `BF16`** (L13 hits `215K > 65,504` `FP16` overflow!) | **`FP16` core + `BF16` residuals** (safe for `43K` residual outliers + `2×` `half` ALU) |
| **MLP Dispatch Pattern** | Packed `MPP` w/ bias+GELU/res fusion | `16<=M<=64`: `MPP_pad32 gate_up` + `MLX down`; Prefill `MPP` only when `M>=2048` | Single-launch **Hot-`MPP` + Cold-`QMV`** (power-law expert skew) + `gather_qmm(sorted_indices=True)` |
| **Scheduler Sweet Spot** | FFD bucketed packing (`S<=512`, `96%` fill) | Continuous Rolling `B_slots=128..256` (`swa_ring_cap=512`, `refill_min=32`) + `EarlyExit` `ff` drain | Amortize experts via real micro-batches (`B=8..64+`); watch sustained thermal + refill correctness |

---

## 3. M4 Pro unified memory capacity (48 GiB)

- `iogpu.wired_limit_mb` defaults low; call `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` safely inside long jobs, keeping active working set (`weights + KV + arenas`) comfortably `< ~32–38 GiB` to prevent macOS compressed-memory swapout thrashing.
- **Dense `33M–300M` Encoders (`< 0.6 GB`):** `GPU + ANE` zero-contention; Co-execute behind LLMs (`+0.7%–2.2%` GPU impact, `swapouts=0`).
- **Small Dense `2B–4B` (`5–13 GB`):** Highest high-batch `docs/s` king on `48 GiB`; holds multiple quantized/BF16 models + ANE encoders resident simultaneously (`15.85 GB` across 4 models, `0` reload stall).
- **Medium Sparse MoE `20B–35B` (`11–22 GB`):** Best `B=1..8` interactive/router (`70–108 tok/s`); at `B>=32`, `98–100%` of experts fire per step, turning prefill & large batches compute/weight-bound.
- **Large Dense `27B–31B` (`~16–19 GB Q4`):** Strictly ~`10–11 tok/s` DRAM/FLOP wall (`min`-scale batch prefill).
- **`>= 80B` Dense or `~60 GiB` MoE (`gpt-oss-120b MXFP4` = `58–59 GiB`):** Exceeds safe `48 GiB` wired envelope—cannot run un-compressed without severe swap/2-bit degradation; target smaller active/distilled models or API escalation.
