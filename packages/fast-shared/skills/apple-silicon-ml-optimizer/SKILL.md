---
name: apple-silicon-ml-optimizer
description: Optimize local ML inference, quantization, custom Metal/MLX kernels, CoreML/ANE pipelines, and training/distillation/QAT/RLVR workloads across Apple Silicon M4 (Apple9) and M5+ (Apple10+ / NAX) Macs. Use when profiling hardware bottlenecks, reducing RAM/DRAM traffic, writing Metal 4 MPP or CoreML/CPU-SME kernels, accelerating LoRA/QLoRA, teacher distillation, encoder/head fine-tuning, or GRPO rollouts, debugging dtype regressions, or maximizing end-to-end throughput and schema/task accuracy on dense encoders, dense LLMs, and sparse MoE models.
---

# Apple Silicon ML Roofline Optimizer (M4 & M5+)

Optimize complete ML workloads on Apple Silicon. M4 (`Apple9`) runs Metal tensor operations on its GPU shaders. M5 (`Apple10`) adds a neural accelerator to each GPU core. Both chips also have a separate Apple Neural Engine (ANE), available through Core ML.

## Core Governing Principles

1. **Optimize `valid_outputs/sec × task_accuracy` within `peak_gb` RAM.** Never ship an optimization that boosts raw `dec tok/s` while regressing end-to-end `docs/s` or blind accuracy (`longer JSON`, `repeat loops`, `extra fallbacks`).
2. **Measure the target Mac.** Record the chip variant, running macOS version, SDK and MLX version. Measure bandwidth and matrix throughput at the shapes you use. Check which tensor formats and operations the OS exposes on that GPU family. [Metal features by macOS version](references/m4-vs-m5-architecture.md#metal-features-by-macos-version) · [Six-chip capability matrix](references/m4-m5-capability-matrix.md)
3. **Eliminate tokens and calls before tuning shaders.** Zero-LLM deterministic bypasses, reasoning/CoT skipping, `llguidance` forced-token jumping + lazy GPU `argmax`, and workload-tested output vocabulary pruning routinely yield `1.5×–4×+` before writing a line of Metal.
4. **Separate micro-benchmarks from E2E execution.** Single-layer `1×` extrapolation misattributes launch overhead (`PLE`) and hides double-wide/heterogeneous layers. Measure full `N`-layer pipelined graphs inside `mx.eval`, then validate E2E under paired interleaved `A/B` runs.
5. **Separate CoreAI-Opt build-time compression from Core AI runtime execution.** `coreai-opt` (`ExportBackend.CoreML` / `CoreAI`) is an offline host compression tool (`26.x+`), whereas Core AI runtime (`.aimodel` / `.aimodelc`, `AIModel` / `InferenceFunction`) requires macOS `27.0+` (`27.2+` for `FP4/FP8` presets & Gemma-4/Qwen3.5), Xcode `27.0+` (`Metal Toolchain` for AOT), verified OS delegate support (`USE_OS_COREAI=1`), and explicit cache keys and GPU stream synchronization. Never treat package importability or format acceptance as hardware delegation or parity.

---

## Execution Workflow (Follow In Order)

### Phase 1: Silicon & Model Forensic Audit (First 10 Minutes)

Before modifying model code or writing custom kernels:

1. **Check host quiescence** so UI/background processes don't poison measurements (`Chrome` compositor/GPU tabs swing throughput `15–25%`):
   ```bash
   python3 scripts/ab_bench_harness.py --check-ps   # or: fast-bench-check --check-ps
   ```
2. **Probe real GPU DRAM bandwidth, GEMM TFLOPS, and `Apple9` vs `Apple10+` (`NAX`) support:**
   ```bash
   python3 scripts/probe_silicon_ceilings.py          # or: fast-silicon-probe
   ```
   On macOS 26.4 or later with a matching SDK, the native Metal probe measures FP16, BF16, INT8 and INT4. It builds those kernels in one library. On 26.2, use the MLX probe above for floating-point throughput and query `[device supportsFamily:MTLGPUFamilyApple10]` for the hardware family. Check Core AI runtime delegate eligibility (`USE_OS_COREAI=1` + macOS `27.0+`/`27.2+`) separately from offline `coreai-torch` / `coreai-opt` export availability, and record 4-bucket physical footprint (`RUSAGE_INFO_V6` + active Metal/MLX + mapped `.safetensors`/`.aimodel` weight files). For exact six-chip specs, macOS `15.0–27.2.1` gates, `MLX` version ceilings, Darwin `hw.optional.arm.*` keys, Metal/CoreML/CoreAI/BNNS SDK declarations, or A64 encodings without loading raw sheets into context, read [references/m4-m5-capability-matrix.md](references/m4-m5-capability-matrix.md) or query `python3 scripts/query_capability_matrix.py --search <term>`.
   ```bash
   xcrun clang++ -O3 -std=c++20 -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformancePrimitives scripts/probe_mpp_metal4.mm -o /tmp/probe_mpp_metal4 && /tmp/probe_mpp_metal4
   ```
3. **Audit exact on-disk weights & roofline without GPU allocation:**
   ```bash
   python3 scripts/audit_safetensors_model.py /path/to/model --gpu-tf <PEAK_TF> --dram-gbs <EFF_GBS>  # or: fast-silicon-audit
   ```
   Verify:
   - True per-tensor quantization bits (many `"4-bit"` releases keep MLPs in `8-bit` or `BF16`!).
   - Tied vs. untied `embed` / `lm_head` (determines whether output-only head pruning is lossless).
   - KV layout (`SWA` vs `Full`, GQA `n_q/n_kv`, `KV-shared` layers, `KiB/tok`) and activation outlier behavior (`FP16 < 65,504` vs `BF16` residual vs `w_pre_ff` variance).
4. **Profile a full `N`-layer graph** (`attn`, `mlp/moe`, `head`, `norms`, `ple/misc`) inside a single `mx.eval` (or Core AI placement explanation reports + runtime profiling on `27.0+`) to attribute true millisecond shares and actual compute-unit placement. If native MLX/CoreML already achieves **`>= 90%`** of your measured ceiling on the bottleneck op, do **not** write a custom kernel unless fusing epilogues/reducing DRAM traffic.

---

### Phase 2: Problem-Level Multipliers (Token & Transfer Cuts)

Apply these before custom Metal unless already present:
- **Deterministic bypass / prompt hygiene:** Filter trivial rows in pure Python/regex (`0` GPU bytes); suppress unnecessary reasoning/CoT tokens when extracting strict structured outputs; preserve semantic whitespace only where formatting matters. For `Qwen3.5` / hybrid reasoning checkpoints, keep `enable_thinking=False` (`<think>\n\n</think>\n\n`) by default; never use truncated thinking budgets (`1..383` tokens degrades accuracy below no-think) — when reasoning is warranted, use either a `>= 384`-token CoT budget or a schema-first `"_reasoning"` field. For `>6K`-token documents, prefer section-filtered slices or map-reduce extraction over single-pass truncation.
- **Grammar compilation (`llguidance` + lazy GPU `argmax`) & tokenizer-matched separators:** Match JSON separator layout to the model's tokenizer and SFT distribution: SentencePiece/`Gemma-4` uses `separators=(",", ": ")` + `whitespace_flexible=False`, whereas BPE models like `Qwen3.5` (which tokenize `": "` as a single token and emit compact/indented transitions) require `whitespace_flexible=True` or aligned compact formatting (forcing strict space-free `(",", ":")` without flexibility dropped stock `Qwen3.5` accuracy `-9.43 pp`). Pre-apply `upfront_ff` inside prefill, jump forced `ff_q` tokens without `lm_head` evaluation (**slice `h` to `M_samp` active non-`ff_q` rows, or skip `final_norm + lm_head` cleanly when `M_samp == 0`**), and copy full `[B, V]` logits to CPU **only** on `< 1%` mask misses with `safe_argmax_1d` NaN guards (never let `np.argmax` silently emit token index `0`).
- **Conformal Grammar Routing (`CROQ`) + Calibrated Escalation (`ExtractConf`) & Retrieval Selection:** Replace blind full-corpus Pass-2 cascades or redundant Stage-2 classification heads with conformal subset routing (`CROQ`, `alpha=0.05`, `aps_det`, locking schema unions when `|C(x)| <= 4` and falling back otherwise) plus OOF-calibrated confidence risk (`ExtractConf` AUROC `0.707–0.745`, ECE `~0.04`) so only ambiguous tails escalate. For document retrieval, use `BGE-small 384d` (`W8` ANE / `FP16` MPP) for fast bulk ingestion (`~340–442 q/s`), `pplx-embed-context-0.6B` with **chunked `MaxSim` (`512`-tok chunks, `512d`/`1024d` `int8` MRL)** for long-context passkey/evidence lookup (`100%` `32K` vs `25%` naive mean pooling), and `EmbeddingGemma-2` when paraphrase generalization dominates OCR speed.
- **Asymmetric / byte-safe vocab pruning:**
  - *Untied `lm_head` (`gpt-oss`-style):* Prune only `lm_head` output rows (`201K -> 131K` = `-34.8%` head BW) while keeping full input embeddings (`0%` prompt miss).
  - *Tied `embed + lm_head` (`Gemma-4`-style):* Keep the checkpoint output vocabulary for multilingual extraction. The English-oriented 150,786-token profile preserves byte fallback but reduced field accuracy on non-English documents. Measure field values as well as format and throughput before enabling it. [Gemma extraction evidence](../../../../docs/benchmarks/gemma-grammar-20261121.json)

---

### Phase 3: Engine, Precision & Kernel Routing

For dense encoders, compare native MLX for plain matrix multiplies and tune tile width for projections that fuse bias, residuals or activations. The wider BGE tiles and native Gemma projections passed a fixed-GPU output check on M4 Pro. The elapsed-time gains were measured on the base M5. Compare the full pipeline on the target Mac using the same passages and frozen query vectors. [M4 Pro encoder checks](references/m4-vs-m5-architecture.md#m4-pro-encoder-opportunities) · [Base M5 OCR timings](references/m4-vs-m5-architecture.md#base-m5-ocr-embeddings) · [Metal, Core ML and MLX recipes](references/kernel-and-dtype-playbook.md)

Route workload stages strictly by arithmetic intensity (`AI = FLOPs / Bytes`), shape stability, and chip generation:

1. **GPU (`MLX` + Metal 4 `MPP::matmul2d`):**
   - *Prefill & compute-bound saturated tiles (`AI > Empirical_Ridge`):* Re-benchmark `MPP::matmul2d` (`64x64`, `BM=64 BN=64`) vs `MLX steel_gemm` / `gather_qmm(sorted_indices=True)` at your exact `(M, K, N)` on the target Mac. Gate custom `MPP` strictly to saturated tiles (`M >= threshold`); small scattered groups lose to native MLX.
   - *Sparse MoE:* Amortize experts across real batch (`B >= 8..128`) with single-launch `Hot-MPP + Cold-QMV` (`gather_qmm` `sorted_indices=True` for unpatched paths, explicit `lhs_indices` + `fp32` `(uint(s)+14)<<23` `E8M0` scale unpacking for `MXFP4`).
   - *Batched Decode (`B >= 16..256`) & `Qwen3.5` Hybrids (`GatedDeltaNet` + Full Attn):* Keep persistent contiguous rolling slots `[B_slots, Cap, n_kv, hd]` (`scatter` in-place via `k[slot_idx, pos]`) to eliminate wave-tail stragglers; split `Full` attention using **Hydragen 2-pass** on high-GQA models (`8:1`), or **1-pass vectorized `bfloat4` ragged SDPA** on low-KV / KV-shared models. On `3/4 GDN + 1/4 Full` hybrids (`Qwen3.5`), bucket `OptiQ` mixed-bit projections (`4-bit`/`8-bit`) via `SpecBucketedLinearGroup`, reuse `256`-token `PrefixTree` KV + `INT8` block-32 recurrent GDN snapshots (captured right after chunk insert, dequantized to `mx.bfloat16`), enable native 1-layer `MTP` only on verified dense `>= 9B` or `optiq/mtp.safetensors` headers (never MoE), and auto-size `<= 80%` UMA (`19.2 GiB` / `cache_limit <= 256–512 MiB` / `batch <= 8` on `<36 GB` hosts for `4B+` LLMs).
   - *Dtype & Calibration Guard:* Hunt silent `FP32` promotions (`bool/int` attention masks, mixed `bfloat16 × float16` `gather_qmm`, transposed non-contiguous weights, CoreML `add_fp16_cast`). Keep `BF16` residuals wherever activations can exceed `65,504` (`FP16 NaN`), and keep `strict_calibration_gate=True` (`>= 2` diverse batches) on `CoreAI-Opt` exports.

2. **ANE (`conv1x1` 4D NCHW `[1, C, 1, S]`, `pipe_depth=4`):**
   - *High ROI targets:* Dense encoders (`S=24..512`, `W8A16` SRAM-fit, dynamic FFD backfill + work-stealing `hetero`/`hetero2`), static fixed-weight early prefix chunks (`FP16` for greedy LLMs; verify LoRA & dynamic weights don't negate concurrency), and `ViT` vision frontends.
   - *Hard anti-targets:* Dynamic sparse MoE (`32E`/`128E` pays full all-expert FLOPs on ANE), dynamically changing LoRA rolling weights, and uncalibrated `W8` LLM prefixes (repeat loops).

3. **CPU SME (`Accelerate` / `BNNSGraph`):**
   - Measure concurrency headroom via [scripts/probe_ane_and_hetero.mm](scripts/probe_ane_and_hetero.mm).
   - Deploy **only** on compute-bound pools via explicit FFD work-stealing (`skip_ops_by_type` keeping elementwise/norms `FP32`), never alongside DRAM-bound `B=1` GPU decode (shared bus contention causes `2×+` slowdown).

4. **Core AI (`.aimodel` / `.aimodelc` on macOS `27.0+`/`27.2+`) & `CoreAI-Opt` (`26.x+`):**
   - *Build-time `CoreAI-Opt` (`ExportBackend.CoreML` across `26.x` & `27.x`):* Run offline calibration/sensitivity selection (with `ignored_ops` and block/channel divisibility checks) inside existing `conv1x1` `.mlmodelc` builders instead of maintaining brittle manual outlier blocklists, preserving `2^-14` clamping, `-40000.0` masks, and `fast-ir-guard --strict-bool-mask --coreml`. Treat `MXINT8`/`INT6`/`UINT6` (`coreai-opt 0.4.0`) as `ExportBackend.Torch`-only, not deployable to CoreML/Core AI.
   - *High-fit Core AI runtime pilots (`27.2+`, `USE_OS_COREAI=1`):* Target pure-ANE scorers (`Laya-Typed-421M`), `Qwen3.5` hybrid GDN/linear-attention models (`fast-pplx`, `Kev-0.8B`), and low-batch (`B=1..4`) `Qwen3.5-4B` / `GPT-OSS` comparisons — promoting only after verified Two-Gate quality + 4-bucket RAM + EOT uplift (`>= 1.00×`, or `>= 1.03×` with custom `TorchMetalKernel` / stateful code). Note `.aimodel` specialization caches key on `(model_hash, toolchain_ver, min_os, arch, options)` and invalidate across OS updates.
   - *Hard Core AI anti-targets:* Never replace packed `hetero2` ANE+GPU OCR encoders (stock padded `.aimodel` `blk_mx` wastes `~73%` positions and drops concurrent `18.56 TF` ANE work) or ragged continuous-batching `B=16..128` decoders (`B_slots=128`, pruned vocab, `llguidance` forced-token jump, `Hot-MPP/Cold-QMV`) with stock single-stream `.aimodel` exports.

---

### Phase 4: Zero-Sync Control & Verification Protocol

1. **Overlap host & device:** Queue pure-FF / non-dependent decode steps with `mx.async_eval(...)`, run `decode + json.loads + schema_validate` on background P-core worker threads, and call `mx.clear_cache()` after large waves (`> 256 MiB` MLX arena). When testing Core AI `NDArray.MutableRawView(metalBuffer:...)` / `InferenceFunction.encode(..., to:)`, enforce explicit MTLBuffer ownership and command-buffer sync, and verify it actually removes multi-MiB copies along the hot stream.
2. **Check output parity before scaling up.** For a drop-in kernel, compare 4–8 representative inputs on the same processor. Check output count, shape and validity, embedding cosine (`>= 0.9998`), and exact integer codes where the model returns them. For GPU/ANE scheduling changes, repeat the baseline and re-encode the largest differences on each fixed processor. For retrieval, check source passage IDs and held-out passage ranks. Treat an intentional precision change as a separate quality experiment.
3. **Control for warmup, subprocess isolation, peak memory & thermal drift.** Isolate different model checkpoints into fresh subprocesses so prior Metal buffers and allocators (`bf16` vs `4bit`) cannot contaminate peak RAM or throughput. Separate cold first-load specialization from warm steady-state latency, reset peak memory (`mx.reset_peak_memory()`), and report **BOTH** peak physical footprint (`RUSAGE_INFO_V6` max + active Metal) **AND** end-of-wave footprint (end-only snapshots hide UMA swap cliffs like `19.2 GiB` spikes!). Alternate runs in `ABBA` order and compare end-to-end `docs/s × task_accuracy`. [Benchmark harness](scripts/ab_bench_harness.py)
4. **Audit [references/dead-ends-registry.md](references/dead-ends-registry.md)** before testing speculative decoding on MoE, `KV_i8` on `hd=64`, `SpinQuant` norm-folding, scalar `kv_sys_i8`, tight batch-1 custom heads, or replacing custom `hetero2`/ragged pipelines with stock `.aimodel` exports.

---

### Phase 5: Training, Distillation & Post-Training Acceleration

Training flips the roofline from `M=1` DRAM-bandwidth-bound decode to **`3×`-GEMM (`Y`, `dX`, `dW`) TFLOPS + activation-RAM-bound backprop** (where `M5+` `NAX` scales `1.5×–4×` over `M4` shader ALUs). Read [references/training-and-distillation-playbook.md](references/training-and-distillation-playbook.md) before writing or tuning any training, distillation, calibration, or RLVR loop:

1. **Autograd & memory hygiene:** Split frozen bottom trunks outside `nn.value_and_grad`, unwrap dead KV-shared `k_proj`/`v_proj` LoRAs, gather `lm_head` strictly on answer tokens with `256`-token sliced `FP32` CE (`5×–8×` fewer `262K`-head FLOPs), pre-cast frozen weights once at load, keep self-attention `BF16` (no `FP32` SDPA upcast), and stay below `75–80%` wired RAM to avoid `50×` swap cliffs.
2. **Tile saturation & checkpoint elimination:** Pack variable-length examples into fixed `M=2048..4096` bins (`position_ids` reset + block-diagonal mask) instead of `B=1` or padded batches (`1.5×–2.6×`), and freeze base trunks in `8-bit g64` / `MXFP4` (`QLoRA`) so `nn.utils.checkpoint` (`+30–35%` 2nd-forward tax) stays off (`1.25×–1.4×`).
3. **Silicon co-execution and kernel fusion.** Pipeline frozen prefixes on the ANE or cache their activations for repeated training. Reuse shared prompt keys and values within gradient accumulation, and use `@mx.compile` on the cross-entropy and optimizer steps. Compare teacher distillation and GRPO rollouts with the same candidates and measured scores. For export, four workers built nine BGE Core ML shapes about twice as fast as one worker on both test Macs. Larger Gemma builds had similar warm serial and parallel times.
