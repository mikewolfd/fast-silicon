---
name: apple-silicon-ml-optimizer
description: Optimize local ML inference, quantization, custom Metal/MLX kernels, CoreML/ANE pipelines, and LoRA training across Apple Silicon M4 (Apple9) and M5+ (Apple10+ / NAX) Macs. Use when profiling hardware bottlenecks, reducing RAM/DRAM traffic, writing Metal 4 MPP or CoreML/CPU-SME kernels, debugging dtype regressions, or maximizing end-to-end throughput and schema/task accuracy on dense encoders, dense LLMs, and sparse MoE models.
---

# Apple Silicon ML Roofline Optimizer (M4 & M5+)

Optimize complete ML workloads on Apple Silicon. M4 (`Apple9`) runs Metal tensor operations on its GPU shaders. M5 (`Apple10`) adds a neural accelerator to each GPU core. Both chips also have a separate Apple Neural Engine (ANE), available through Core ML.

## Core Governing Principles

1. **Optimize `valid_outputs/sec × task_accuracy` within `peak_gb` RAM.** Never ship an optimization that boosts raw `dec tok/s` while regressing end-to-end `docs/s` or blind accuracy (`longer JSON`, `repeat loops`, `extra fallbacks`).
2. **Measure the target Mac.** Record the chip variant, running macOS version, SDK and MLX version. Measure bandwidth and matrix throughput at the shapes you use. Check which tensor formats and operations the OS exposes on that GPU family. [Metal features by macOS version](references/m4-vs-m5-architecture.md#metal-features-by-macos-version)
3. **Eliminate tokens and calls before tuning shaders.** Zero-LLM deterministic bypasses, reasoning/CoT skipping, `llguidance` forced-token jumping + lazy GPU `argmax`, and lossless/asymmetric vocab pruning routinely yield `1.5×–4×+` before writing a line of Metal.
4. **Separate micro-benchmarks from E2E execution.** Single-layer `1×` extrapolation misattributes launch overhead (`PLE`) and hides double-wide/heterogeneous layers. Measure full `N`-layer pipelined graphs inside `mx.eval`, then validate E2E under paired interleaved `A/B` runs.

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
   On macOS 26.4 or later with a matching SDK, the native Metal probe measures FP16, BF16, INT8 and INT4. It builds those kernels in one library. On 26.2, use the MLX probe above for floating-point throughput and query `[device supportsFamily:MTLGPUFamilyApple10]` for the hardware family.
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
4. **Profile a full `N`-layer graph** (`attn`, `mlp/moe`, `head`, `norms`, `ple/misc`) inside a single `mx.eval` to attribute true millisecond shares. If native MLX/CoreML already achieves **`>= 90%`** of your measured ceiling on the bottleneck op, do **not** write a custom kernel unless fusing epilogues/reducing DRAM traffic.

---

### Phase 2: Problem-Level Multipliers (Token & Transfer Cuts)

Apply these before custom Metal unless already present:
- **Deterministic bypass / prompt hygiene:** Filter trivial rows in pure Python/regex (`0` GPU bytes); suppress unnecessary reasoning/CoT tokens when extracting strict structured outputs; preserve semantic whitespace only where formatting matters.
- **Grammar compilation (`llguidance` + lazy GPU `argmax`):** Compile strict schemas (`separators=(",", ":")`, `whitespace_flexible=False`), pre-apply `upfront_ff` inside prefill, jump forced `ff_q` tokens without `lm_head` evaluation (**slice `h` to `M_samp` active non-`ff_q` rows, or skip `final_norm + lm_head` cleanly when `M_samp == 0`**), and copy full `[B, V]` logits to CPU **only** on `< 1%` mask misses.
- **Asymmetric / byte-safe vocab pruning:**
  - *Untied `lm_head` (`gpt-oss`-style):* Prune only `lm_head` output rows (`201K -> 131K` = `-34.8%` head BW) while keeping full input embeddings (`0%` prompt miss).
  - *Tied `embed + lm_head` (`Gemma-4`-style):* Preserve byte fallback (`<0x00>..<0xFF>`) + task tokens (`150,786` safe on English/JSON); **never** chop below byte-safe coverage (`<131K` destroys tail recall).

---

### Phase 3: Engine, Precision & Kernel Routing

For dense encoders, compare native MLX for plain matrix multiplies and tune tile width for projections that fuse bias, residuals or activations. The wider BGE tiles and native Gemma projections passed a fixed-GPU output check on M4 Pro. The elapsed-time gains were measured on the base M5. Compare the full pipeline on the target Mac using the same passages and frozen query vectors. [M4 Pro encoder checks](references/m4-vs-m5-architecture.md#m4-pro-encoder-opportunities) · [Base M5 OCR timings](references/m4-vs-m5-architecture.md#base-m5-ocr-embeddings) · [Metal, Core ML and MLX recipes](references/kernel-and-dtype-playbook.md)

Route workload stages strictly by arithmetic intensity (`AI = FLOPs / Bytes`), shape stability, and chip generation:

1. **GPU (`MLX` + Metal 4 `MPP::matmul2d`):**
   - *Prefill & compute-bound saturated tiles (`AI > Empirical_Ridge`):* Re-benchmark `MPP::matmul2d` (`64x64`, `BM=64 BN=64`) vs `MLX steel_gemm` / `gather_qmm(sorted_indices=True)` at your exact `(M, K, N)` on the target Mac. Gate custom `MPP` strictly to saturated tiles (`M >= threshold`); small scattered groups lose to native MLX.
   - *Sparse MoE:* Amortize experts across real batch (`B >= 8..128`) with single-launch `Hot-MPP + Cold-QMV` (`gather_qmm` `sorted_indices=True` for unpatched paths, explicit `lhs_indices` + `fp32` `(uint(s)+14)<<23` `E8M0` scale unpacking for `MXFP4`).
   - *Batched Decode (`B >= 16..256`):* Keep persistent contiguous rolling slots `[B_slots, Cap, n_kv, hd]` (`scatter` in-place via `k[slot_idx, pos]`) to eliminate wave-tail stragglers; split `Full` attention using **Hydragen 2-pass** on high-GQA models (`8:1`), or **1-pass vectorized `bfloat4` ragged SDPA** on low-KV / KV-shared models.
   - *Dtype Guard:* Hunt silent `FP32` promotions (`bool/int` attention masks, mixed `bfloat16 × float16` `gather_qmm`, transposed non-contiguous weights, CoreML `add_fp16_cast`). Keep `BF16` residuals wherever activations can exceed `65,504` (`FP16 NaN`).

2. **ANE (`conv1x1` 4D NCHW `[1, C, 1, S]`, `pipe_depth=4`):**
   - *High ROI targets:* Dense encoders (`S=24..512`, `W8A16` SRAM-fit, dynamic FFD backfill + work-stealing `hetero`/`hetero2`), static fixed-weight early prefix chunks (`FP16` for greedy LLMs; verify LoRA & dynamic weights don't negate concurrency), and `ViT` vision frontends.
   - *Hard anti-targets:* Dynamic sparse MoE (`32E`/`128E` pays full all-expert FLOPs on ANE), dynamically changing LoRA rolling weights, and uncalibrated `W8` LLM prefixes (repeat loops).

3. **CPU SME (`Accelerate` / `BNNSGraph`):**
   - Measure concurrency headroom via [scripts/probe_ane_and_hetero.mm](scripts/probe_ane_and_hetero.mm).
   - Deploy **only** on compute-bound pools via explicit FFD work-stealing (`skip_ops_by_type` keeping elementwise/norms `FP32`), never alongside DRAM-bound `B=1` GPU decode (shared bus contention causes `2×+` slowdown).

---

### Phase 4: Zero-Sync Control & Verification Protocol

1. **Overlap host & device:** Queue pure-FF / non-dependent decode steps with `mx.async_eval(...)`, run `decode + json.loads + schema_validate` on background P-core worker threads, and call `mx.clear_cache()` after large waves (`> 256 MiB` MLX arena).
2. **Check output parity before scaling up.** For a drop-in kernel, compare 4–8 representative inputs on the same processor. Check output count, shape and validity, embedding cosine (`>= 0.9998`), and exact integer codes where the model returns them. For GPU/ANE scheduling changes, repeat the baseline and re-encode the largest differences on each fixed processor. For retrieval, check source passage IDs and held-out passage ranks. Treat an intentional precision change as a separate quality experiment.
3. **Control for warmup and thermal drift.** Warm both paths, reset peak memory with `mx.reset_peak_memory()`, and alternate runs in `ABBA` order. On sustained workloads, check whether repeated baseline times drift as the Mac heats up. Compare end-to-end time and task quality. [Benchmark harness](scripts/ab_bench_harness.py)
4. **Audit [references/dead-ends-registry.md](references/dead-ends-registry.md)** before testing speculative decoding on MoE, `KV_i8` on `hd=64`, `SpinQuant` norm-folding, scalar `kv_sys_i8`, or tight batch-1 custom heads.
