# Apple M4 vs M5+ Merged Hardware & ML Capability Matrix

Concise merged reference synthesizing verified host measurements, Darwin `hw.optional.arm.*` readings, Xcode 27.2 SDK declarations, LLVM/Clang target flags, and framework release gates across all six M4/M5 SKUs (`M4`, `M4 Pro`, `M4 Max`, `M5`, `M5 Pro`, `M5 Max`).

> **Progressive Disclosure:** Read this curated reference for architecture decisions, version floors, and hardware gotchas. For exact lookups across **12,025** normalized records (all **4,682** Arm A64 encodings, **221** ISA differences, **3,648** Metal SDK symbols, **1,928** deduplicated Core ML / Core AI / MPSGraph APIs, **390** MLX APIs, **326** Metal GPU rows, **42** BNNS graph APIs, and **123** Darwin keys), run:
> ```bash
> python3 scripts/query_capability_matrix.py --stats
> python3 scripts/query_capability_matrix.py --table mlx --search "TF32|NO_NAX|force_fused|D512|gated_delta"
> python3 scripts/query_capability_matrix.py --table metal_sdk --min-os 27.0 --search "Tensor"
> python3 scripts/query_capability_matrix.py --table instruction_differences --diff-only --search "SVE_B16B16"
> ```

---

## 1. Evidence Taxonomy & Verified Hosts

Never conflate hardware capability, OS visibility, compiler target flags, and framework execution:

| Evidence Class | Meaning |
|---|---|
| `Verified Host (M4 Pro 26.6)` | Directly executed live on `Mac16,8` (`2026-11-19`, macOS `26.6 25G5057c`, Xcode `27.2 27B52`, Apple Clang `21.0.0`, MLX `0.32.4`). |
| `Reported Host (Base M5 26.2 / 27.2.1)` | Captured on `14" MacBook Pro Base M5` (`24 GiB`, macOS `26.2 25C56`, later rechecked on `27.2.1 26B131`, MLX `0.32.4`). |
| `Apple Official Spec` | Published Apple hardware specs and Metal Feature Set Tables; Pro/Max family columns without host access retain hardware-spec scope. |
| `Compiler / SDK Decl` | LLVM `22..23` / Apple Clang `21.0.0` target flags or Xcode `27.2` SDK `API_AVAILABLE` declarations (can precede/exceed hardware). |
| `Measured Benchmark` | Paired timed run (`OCR timings`) with exact input shape, dtype, OS, and retrieval parity checks. |

**Missing != False Guardrail:** An absent Darwin `hw.optional.arm.*` key or missing Metal selector (`Selector unavailable`) means **unknown / unexposed on that OS build**, never guaranteed hardware absence (e.g., Base M5 executed `BFADD` on `26.2` before `FEAT_SVE_B16B16` appeared in `sysctl` on `27.2.1`, and `supportsPlacementSparse` arrived in macOS `26.4`).

---

## 2. Six-Chip Hardware, Topology & Memory Matrix

| Dimension | M4 | M4 Pro | M4 Max | M5 | M5 Pro | M5 Max |
|---|---|---|---|---|---|---|
| **Process / Package** | 2nd-gen 3 nm monolithic | 2nd-gen 3 nm monolithic | 2nd-gen 3 nm monolithic | 3rd-gen 3 nm monolithic | 3rd-gen 3 nm **2-die Fusion** | 3rd-gen 3 nm **2-die Fusion** |
| **CPU Core Bins (Full / Low)** | `10` (`4P+6E`) / `8` (`4P+4E`) | `14` (`10P+4E`) / `12` (`8P+4E`) | `16` (`12P+4E`) / `14` (`10P+4E`) | `10` (`4 Super + 6E`) | `18` (`6 Super + 12P`) / `15` (`5 Super + 10P`) | `18` (`6 Super + 12P`) |
| **GPU Cores** | `8` or `10` (`Apple9`) | `16` or `20` (`Apple9`) | `32` or `40` (`Apple9`) | `8` or `10` (`Apple10`) | `16` or `20` (`Apple10`) | `32` or `40` (`Apple10`) |
| **GPU Neural Accelerator** | Absent | Absent | Absent | **1 per GPU core** (`NAX`) | **1 per GPU core** (`NAX`) | **1 per GPU core** (`NAX`) |
| **Memory BW (Full / Low Max)** | `120 GB/s` | `273 GB/s` | `546 GB/s` (`410 GB/s` @ 32-GPU) | `153 GB/s` | `307 GB/s` | `614 GB/s` (`460 GB/s` @ 32-GPU) |
| **Unified Memory Capacity** | `16 / 24 / 32 GB` | `24 / 48 / 64 GB` | `36 / 48 / 64 / 128 GB` | `16 / 24 / 32 GB` | `24 / 48 / 64 GB` | `36 / 48 / 64 / 128 GB` |
| **Standalone ANE** | `16` cores (`38 TOPS` launch claim) | `16` cores | `16` cores | `16` cores | `16` cores | `16` cores |
| **Media Engines & AV1** | `1` dec / `1` enc / `1` ProRes + AV1 dec | `1` dec / `1` enc / `1` ProRes + AV1 dec | `1` dec / **`2` enc** / **`2` ProRes** + AV1 dec | `1` dec / `1` enc / `1` ProRes + AV1 dec | `1` dec / `1` enc / `1` ProRes + AV1 dec | `1` dec / **`2` enc** / **`2` ProRes** + AV1 dec |
| **Thunderbolt / Security** | `TB4` / No MIE | **`TB5`** / No MIE | **`TB5`** / No MIE | `TB4` / **MIE (EMTE)** | **`TB5`** / **MIE (EMTE)** | **`TB5`** / **MIE (EMTE)** |

> **Topology Warning:** Base `M5` uses `4 Super + 6 Efficiency` cores (`E-cluster L2 = 6 MiB` vs M4 Pro `4 MiB`), whereas `M5 Pro / M5 Max` have `6 Super + 12 Performance` cores (zero E-cores). **Never transfer base-M5 E-core models or E-cluster L2 assumptions to `M5 Pro / Max`.**

---

## 3. OS Evolution (`26.2 -> 26.4 -> 26.6 -> 27.2.1`) & Live Probe

The second Base `M5` host recheck on macOS **`27.2.1` (`26B131`)** resolved three critical `26.2` (`25C56`) ambiguities, corroborated live on `Mac16,8` (`M4 Pro`, `26.6 25G5057c`):

1. **`MLNeuralEngineComputeDevice.totalCoreCount` Bug Fixed:** Initial `26.2` readings on Base M5 returned `2` in two bindings (despite Apple's `16`-core ANE spec). On **`27.2.1` (`26B131`)** the same Base M5 returned **`16`**, matching M4 Pro `26.6` (`16` verified live). Never interpret `2` as `2` physical cores or clusters.
2. **`MTLDevice.supportsPlacementSparse` (`26.4+`):** Unavailable on Base M5 `26.2`; returns **`Yes`** on Base M5 `27.2.1` and **`Yes` (verified live)** on M4 Pro `26.6` (`API_AVAILABLE(macos(26.4))` in `MTLDevice.h:1252`).
3. **`hw.optional.arm.FEAT_SVE_B16B16`:** Key was absent on Base M5 `26.2` and flipped to **`Reported 1`** on `27.2.1` (`0` live on M4 Pro `26.6`). This moved **`34`** A64 instruction forms (`16` `FEAT_SVE_B16B16` + `18` `FEAT_SME2 && FEAT_SVE_B16B16`) into **`M5 only • reported gates` (`110 -> 144` across the `221` difference forms)**, plus `1` `FEAT_SME2p1 && FEAT_SME_LUTv2` form to `Neither • reported gates` (`1 -> 2`).
4. **`10` Newly Exposed Darwin `sysctl` Keys on M5 `27.2.1` (all `Reported 0`):** `FEAT_CPA`, `FEAT_CPA2`, `FEAT_FAMINMAX`, `FEAT_FP8`, `FEAT_FPMR`, `FEAT_LUT`, `FEAT_PAuth_LR`, `FEAT_SME_F8F16`, `FEAT_SME_F8F32`, `FEAT_SME_LUTv2` (`CPU features`: `89 -> 99`; `Darwin readings`: `113 -> 123`). All `10` remain `ABSENT` on M4 Pro `26.6`.
5. **Live `Mac16,8` (`26.6`, Xcode `27.2`) Native Verification (`2026-11-19`):** `MTLDevice` (`Apple M4 Pro`, `applegpu_g16s`) returned `Apple9=1`, `Apple10=0`, `Metal3=1`, `Metal4=1`, `supportsPlacementSparse=1`, `supportsShaderBarycentricCoordinates=1`, `maxThreadgroupMemoryLength=32768`, `recommendedMaxWorkingSetSize=40200896512`, `maxBufferLength=30150672384`, `totalCoreCount=16`, `P-cluster L2=16 MiB` (`5 CPUs/L2`), `E-cluster L2=4 MiB` (`4 CPUs/L2`), `L1I/D=192/128 KiB (P)`, `128/64 KiB (E)`.

---

## 4. CPU ISA, SME2 vs SME2.1, Streaming Mode & Compiler Traps

| Feature / Probe | M4 Family (`26.6`) | M5 Family (`26.2 / 27.2.1`) | Execution & Compiler Notes |
|---|---|---|---|
| **Shared `SME` / `SME2`** | `Reported 1`, `64B` streaming SVL | `Reported 1`, `64B` streaming SVL | Both support widening `FP16/BF16 -> FP32` (`FMOPA.S`, `BFMOPA.S`), `INT8/INT16 -> INT32`, `F64F64`, `I16I64`. |
| **`SME2.1`, `SME_F16F16`, `SME_B16B16`** | `0` (skipped by guard) | **`Reported 1`** (`FMOPA.H -> 0x3c00`, `BFMOPA.H -> 0x3f80`) | Non-widening `16-bit` accumulators verified on Base M5 `26.2` and `27.2.1`. |
| **`FEAT_CSSC` / `FEAT_HBC` / `FEAT_EBF16` / `FEAT_MTE4`** | `0` (verified live) | **`Reported 1`** (`CTZ(0x1000)=12`) | M5 adds EMTE/MIE (`MTE`, `MTE2`, `MTE4`; process policy gates execution), `FEAT_MTE3`/`FEAT_MTE_ASYNC` report `0`. |
| **Streaming-Mode Trap (`BFADD Z0.H`)** | Traps (`SIGILL`) both modes | **Executes `2.0` (`0x4000`) ONLY in streaming mode; traps `SIGILL` in ordinary mode** | Confirmed on both `26.2` (sysctl absent) and `27.2.1` (`sysctl=1`). Validates this exact unpredicated form, not arbitrary SVE. |
| **Apple Clang `21.0.0` (`Xcode 27.2`) `-mcpu=apple-m5` Delta** | Baseline (`apple-m4`) | Adds **`+cssc +hbc +mte +predres +sme-b16b16 +sme-f16f16 +sme2p1 +specres2 +sve-b16b16`** | **Traps:** `FEAT_SPECRES` / `FEAT_SPECRES2` report `0` at runtime; LLVM `23.1` notes `FEAT_MPAM`/`FEAT_SEL2` are inherited in model despite HW absence; LLVM `22 Host.cpp` defaulted `-mcpu=native` on M5 to `apple-m4` until `LLVM 23.1.0`. |

---

## 5. Metal 4, MPP, Cooperative Tensors & Tensor Formats

- **GPU Hardware Delta (`Apple9` -> `Apple10`):** M5 doubles peak FP16/complex-ALU & geometry rates (`up to 2×`), bumps max texture edge `16384 -> 32768`, MSAA `4 -> 8`, mesh grid `1,048,575 -> 4,194,303`, ray-tracing alignment `16 KB -> 1 KB`, adds Sampler `reductionMode`/`lodBias` and compute-write Universal Texture Compression, while sharing **`1024` threads/TG** & **`32 KB` TG memory**. **Neither GPU supports FP64.**
- **macOS Storage vs Arithmetic vs Scale-Plane Gates:**
  - `26.0`: Metal 4, `MTLTensor` (`BFloat16`, `Int8`, `UInt8`, `Float16/32`, `Int16/32` **storage APIs**).
  - `26.1`: **`BF16` MPP tensor operands** (`BFloat16` tensor *storage* existed in `26.0`, *operands* require `26.1+`).
  - `26.3`: Cooperative tensors as `matmul` inputs (`is_compatible_as_left_input` / right; **live Xcode `27.2`/M4 `26.6` probe required `1 SIMD group` (`32` threads) and compile-time `K`** — `4` SIMD groups or dynamic `K` failed compile!). Also `MPSGraph.planarTensor(tensor:name:)`.
  - `26.4`: `Int4` / `UInt4` storage & direct `INT4/INT8` integer TensorOps operands; `MTLDevice.supportsPlacementSparse`; `MPSGraphCompilationDescriptor.convertLayoutToNHWC()`.
  - `27.0`: `Int2` / `UInt2` (`512`-el stride, `32`-el row, `4`-el slice, `128 B` buf), `MetalFloat4E2M1` (`256`-el stride, `2`-el slice), `MetalFloat8E5M2` / `E4M3` (`128`-el stride, `1`-el slice), and **auxiliary scale-plane APIs (`MetalFloat8UE8M0` — scale plane only, not matrix payload)**.
- **SDK Declaration Correction:** `MTLDevice.supportsShaderBarycentricCoordinates` requires **`macOS 10.15`** (correcting `"No minimum"` in earlier metadata).

---

## 6. Framework & Version Gates (`MLX 0.32.4`, `Core ML`, `Core AI 27.2+`, `BNNS`)

| Stack | Milestone / Release | Critical Gate or Gotcha |
|---|---|---|
| **MLX Build & Device** | `0.0.2` / `0.30.0` / `0.32.0` | `Device`/`DeviceType` expose only `CPU` and `GPU` (**no ANE backend in `0.32.4`**). Building from source with `CMAKE_OSX_DEPLOYMENT_TARGET < 26.2` compiled **`MLX_METAL_NO_NAX`** (gate fixed `0.32.0`). `mx.compile` existed `0.1.0`; GPU fusion arrived `0.2.0`. |
| **MLX Precision (`TF32`)** | `0.30.0` / `0.32.4` | **`MLX_ENABLE_TF32=0`** (default `1`, cached on first call!) disables eligible **FP32 NAX** paths in Metal GEMM, quantized matmul, and SDPA without changing array dtype. Always set before launch & log in FP32 comparisons. |
| **MLX Kernels (`0.30.x–0.32.x`)** | `0.30.1` -> `0.32.4` | `0.30.1` NAX JIT & TB5 RDMA `JACCL` (`0.30.6` faster on `26.3+` — Base M5 has TB4 only); `0.30.3` `MXFP8`/`NVFP4` QMM & `to_fp8`; `0.30.4` NAX split-K GEMM; `0.31.1` M5 Pro/Max GEMM; `0.31.2` NAX `segmented_mm` & >32K SDPA `int16` index overflow fix; `0.32.0` `qmv_wide` & DLPack & math modes; `0.32.1` MSL `4.0` (`26.x`) vs `4.1` (`27.x`); `0.32.2` `force_fused` SDPA & `MXFP8` scale rounding; `0.32.3` `D256` NAX array-mask prefill & `gated_delta_update` (**Perplexity pilot flipped `242/8192` signed output coords by `1` — retrieval open**); `0.32.4` 1-bit quant, `gated_delta_update` VJP, & **pre-M5 `D512` causal attention** (`Q >= 1024`, causal, no array mask/sinks/LSE). |
| **Core ML / Core AI / BNNS** | `14.0`–`27.2+` | Core ML remains supported alongside Core AI (`MLComputeUnits` has `cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine` [allows CPU fallback], `all` — **no ANE-only**; required for macOS `26.x` ANE `conv1x1` & non-neural models). Core AI runtime (`AIModel` / `InferenceFunction`, `NDArray.MutableRawView` shared `MTLBuffer` + async `encode`, `USE_OS_COREAI=1` OS delegate check, AOT `.aimodelc` via Xcode 27 Metal Toolchain + residual on-device specialization) arrives in **`27.0`**. Toolchain/package gates: `apple/coreai-models` pins `coreai-torch==0.4.3`, `coreai-core==1.0.0b3`, `coreai-opt==0.4.0` (with `to_coreai()` already optimized in `0.4.3+`); `coreai-torch 0.4.4` (Nov 18) adds FP8 constants, `torch.cond`/`while_loop` kernels, and `ImageType` I/O; **`0.4.0` `.aimodel` exports fail OS 27 beta 2+ specialization (`>=0.4.1` required)**; `coreai-opt 0.4.0` `MXINT8`/`INT6`/`UINT6` are `ExportBackend.Torch`-only (not CoreML/Core AI); **`fp4_weights_fp8_activations_fp8_kv_cache` & `fp8_weights_fp8_activations_fp8_kv_cache` presets require `27.2+`**. `BNNSGraph` starts `15.0`; `BNNSGraphContextSetStreamingAdvanceCount` requires **`15.4`**. |

---

## 7. Reconciled Encoder / OCR Timings (`ABBA` Unrounded)

| Encoder & Candidate | Host & OS / MLX | Workload | Baseline -> Candidate | Delta / Speedup | Parity Check |
|---|---|---|---|---|---|
| **`BGE-small 384d`** (wider 128-col fused FP16 tiles) | Base M5 (`26.2`, `0.32.4`) | `7,974` passages (`210` docs) | `9.5412 s -> 8.5262 s` | **`-10.64%` (`1.119×`)** | `64/64` ranks match (`15/24 @1`, `20/24 @5`) |
| **`BGE-small 384d`** (raw OCR end-to-end) | Base M5 (`26.2`, `0.32.4`) | `117` docs (`3,705` vecs) | `6.3865 s -> 5.8657 s` | **`-8.15%` (`1.089×`)** | Exact passage order; min cosine `0.999985` |
| **`EmbeddingGemma 2 512d`** (`mx.matmul` plain proj + ANE) | Base M5 (`26.2`, `0.32.4`) | `7,974` passages (`210` docs) | `54.2806 s -> 46.1975 s` | **`-14.89%` (`1.175×`)** | `17/24 @1`, `22/24 @5` held-out identical |
| **`EmbeddingGemma 2 512d`** (`force_fused` D256 `384/512`) | Base M5 (`26.2`, `0.32.4`) | `7,974` passages (`FP16`) | `46.0875 s -> 45.0408 s` | **`-2.27%` (`1.023×`)** | Top-1/Top-5 held; lower ranks shifted |
| **`EmbeddingGemma 2`** (`GPU -> GPU + ANE`) | M4 Pro `48 GiB` | `2,000` docs (`559,737` tok) | `47.257 s -> 17.186 s` | **`-63.63%` (`2.750×`)** | Same token count; adding CPU slowed to `19.523 s` (`+13.60%`) |
| **`pplx-embed-v2-context-0.6b`** (combine 2 FP32 FF proj) | Base M5 / M4 Pro | `384` passages | `11.9618 -> 12.0491 s` (M5) | `+0.73%` slower (M5), `+0.93%` (M4) | All `384` integer vectors exact |

---

## 8. Merge Validation & Data Hygiene Summary

Merging both source workbooks (`27`-sheet latest-M5 capture + `26`-sheet curated-addition merge) resolved:
- **Zero boolean predicate logic errors** across all `4,682` Arm A64 encodings and `221` ISA diffs (`397` label-only diffs = `'No explicit extension guard'` vs `'No extension guard'`).
- **Stale `26.2` artifacts fixed:** Restored `10` new `27.2.1` Darwin `hw.optional.arm.*` keys (`89 -> 99` CPU features, `113 -> 123` Darwin readings), `FEAT_SVE_B16B16` flip (`110 -> 144` M5-only diffs), `supportsPlacementSparse=Yes` (`26.4+`), `totalCoreCount=16`, and unrounded `OCR timings` + raw OCR `117`-doc row.
- **Curated WB2 fixes retained:** `17` six-chip additions (`CPU ISA +3`, `Instruction forms +1`, `GPU and Metal +2`, `MLX +7`, `Core AI +1`, `Other frameworks +3`), `supportsShaderBarycentricCoordinates=10.15`, `MetalFloat8UE8M0` scale-plane role, `MLX_ENABLE_TF32=0`, and `MLX_METAL_NO_NAX`.
- **Integrity & leak cleanup:** Collapsed `2` exact duplicate `MLShapedArrayProtocol` rows in `Apple APIs` (`1,930 -> 1,928`), separated `750` appended spreadsheet cell-diff rows from `Sources and scope` (`784 -> 29`), fixed `40+` concatenated word-version typos (`Broad 27.0`, `in 0.32.0`, etc.), and scrubbed all local machine filesystem paths.
