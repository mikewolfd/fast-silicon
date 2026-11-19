# Metal 4, MLX, CoreML/ANE & CPU-SME Implementation Playbook

Concrete engineering patterns and compiler/dtype traps verified on Apple Silicon (`macOS 26.x`, `Metal 4.0`, `MLX`, `CoreML 9`, `Accelerate`).

---

## 1. Metal 4 `MPP::matmul2d` & Custom Kernel Recipes

### Tensor Descriptor & Dim Order Trap
In `<metal_tensor>` (`metal::extents<int32_t, D0, D1>`), **`D0` is the innermost stride-1 X dimension**, **`D1` is the outer Y dimension**:
- Row-major `A[M, K]`: `extents<int32_t, K, M>`
- Row-major transposed `B[N, K]` (`transpose_right = true`): `extents<int32_t, K, N>`
- Destination `C[M, N]` cooperative indices: `idx[0]` = column (`0..BN-1`), `idx[1]` = row (`0..BM-1`).

### Fused Epilogues & Cooperative Dequant (`bf16/f16 × int4/int8`)
- Use `mpp::tensor_ops::matmul2d_descriptor(BM, BN, BK_or_dyn, false, true, false, mode::multiply_accumulate)`.
- Fuse post-GEMM elementwise chains cleanly before storing to `C` (`bias`, `bias + residual`, `bias + GELU` using `exact_erf`, `GeGLU`, `SwiGLU`), cutting intermediate HBM round-trips.
- Compare direct signed INT4/INT8 tensor operands with staged dequantization on macOS 26.4 or later. The M4 Pro shader path accepted both formats on 26.6. Include group scales, offsets and final passage quality when testing model weights. [OS and hardware support](m4-vs-m5-architecture.md#metal-features-by-macos-version) · [Native Metal probe](../scripts/probe_mpp_metal4.mm)
- When chaining matrix operations on M4 Pro, test input and output layouts separately. On 26.6 a compatible FP16 tile passed directly to the next multiply with one SIMD group and a fixed inner dimension. Guard the reuse with `is_compatible_as_left_input` or `is_compatible_as_right_input` and stage through threadgroup memory when layouts differ. Existing cooperative-output epilogues can use a different execution scope. [M4 Pro runtime checks](m4-vs-m5-architecture.md#m4-pro-encoder-opportunities)
- Always check whether **native `MLX qmm` + `mx.argmax` already saturates DRAM/compute** before shipping custom argmax epilogues (e.g., on `gpt-oss` `131K` Q4 `lm_head`, MLX native `qmm + argmax` ran `1.066 ms` @ `B=1` & `7.995 ms` @ `B=64`, beating custom `lm_q4_argmax_mpp` `4.38` & `8.88 ms`).
- Wait on `simd_ballot`: Metal C++ `simd_vote::vote_t` portability varies—use `simd_max` + `uint lane_win = (v == rmax) ? col : 0xFFFFFFFFu; uint win = simd_min(lane_win);` for clean warp-level argmax tie-breaking!

### `MXFP4` (`E2M1` + `E8M0`) Clean Unpacking
- Never unpack `E8M0` scales in `float16`! Real LLM MoE `E8M0` scales (`115..136`, exponent `-12..+9`) overflow/underflow naive FP16 unpacking (`inf/NaN`).
- Always fold `E8M0` scales in `float32` using direct exponent bit-shift:
  `float sc = as_type<float>((uint(e8m0) + 14u) << 23);` combined with integer E2M1 subnormal unpacking (`e2m1_pair_subnormal`).

### In-Place Contiguous Rolling Slot Cache Layout
- **Critical MLX Layout Rule:** `(B_slots, n_kv, Cap, hd)` **cannot** scatter per-slot variable positions `k[slot_idx, :, pos_idx, :]` without `Reshape` errors or full-buffer copies!
- Store KV caches as **`(B_slots, Cap, n_kv, hd)`**:
  - `k_buf[slot_idx_i32, write_pos_i32] = k_new` (`shape = (b_act, n_kv, hd)`) mutates MLX arrays **100% in-place** (`~1.1 ms` total across 12 layers @ `B=64`, `0 MB` extra copy peak!) AND keeps each warp's `lane 0..31` reading contiguous `128B` cache lines per head.
- **FP-Associativity Parity Note:** If validating bit-for-bit against a 2-stage static `[0..InitLen) + [InitLen..ValidLen)` attention kernel (`hydragen_p2_varlen`), split the rolling scan into the exact same 2 loops (`0..InitLen` then `InitLen..ValidLen`)—one combined `0..ValidLen` loop causes `~6e-5` fp32 softmax accumulation differences that cascade across `300+` greedy decode steps.

---

## 2. CoreML, ANE (`conv1x1`) & CPU-SME (`BNNSGraph` / `Accelerate`) Recipes

1. **Force ANE Execution With 4D NCHW `conv1x1`:**
   - Replace 2D/3D linear projections (`mb.matmul` falls back cleanly or runs `0.42–4.34 TF`) with static `[1, C_in, 1, S]` `mb.conv` (`1x1` kernel, static bucket `S in {24, 32, 64, 128, 256, 512}`).
   - Feed async queues with **`pipe_depth = 4`** (`dispatch_semaphore_create(4)` + `predictionFromFeatures:completionHandler:`) to hit `12.6–18.6 TF` ANE saturation.
   - Verify ZERO silent CPU/GPU fallback using **`[MLComputePlan loadContentsOfURL:...]`** (`scripts/probe_ane_and_hetero.mm`), ensuring `ANE == total_ops`.
2. **Prevent CoreML `FP16` Precision Collapse (`cos = 0.956 -> 0.99983`):**
   - CoreML's default `ct.ComputeUnit.CPU_ONLY` (`BNNSGraph`) / `ALL` FP16 pass inserts casts that wreck `RMSNorm`, `Softmax`, and `GeLU`.
   - Use **`compute_precision=ct.precision.FLOAT32` OR `ct.transform.FP16ComputePrecision(op_selector=...)` (`skip_ops_by_type`)** keeping normalization, softmax, reduction, and activation ops in `FP32` while weights/convs stay `FP16` / `W8`.
3. **CPU SME (`Accelerate` / `BNNSMatMul`):**
   - Cap worker concurrency around `2` BLAS/SME threads when sharing the 2 SME2 units (`>2` threads plateaus and wastes synchronization). Do not spawn `dispatch_apply` work right before `BNNSMatMul`.
4. **Private Zero-Copy `_ANEInMemoryModel` Caveats:**
   - `IOSurface` <-> Metal `newBufferWithIOSurface` enables zero-copy `FP16` SwiGLU (`10.86 TF`) & dynamic `mb.matmul` (`4.34 TF` w/ `0 MB` disk weight), **but** avoid multi-procedure bank switching (`status=0x2`) and raw 2-input in-memory dynamic `matmul` IOSurface maps (`status=0x1D`).

---

## 3. MLX & PyTorch/LoRA Silent Performance Traps

Catch these immediately during profiling:

| Trap / Bug | Measured Consequence | Exact Fix |
|---|---|---|
| **Silent `FP32` Attention / GEMM Upcast** | `int` masks forced `FP32` SDPA (`-36%`); `bf16` acts × `fp16` `gather_qmm` scales forced `FP32` (`15.6 -> 14.2 ms`); default CoreML `FP16` dropped cos to `0.956`. | Cast attention masks to native `BF16`/`FP16` (`mx.where`/bool); emit `rmsnorm_bf16_to_f16` matching weight/scale dtype; verify compiled graph dtypes. |
| **Missing `lhs_indices` in `mx.gather_qmm`** | Broadcasted `4 × 4 = 16` expert products silently instead of `4` (`15.29 -> 9.84 ms`!). | Always pass explicit `lhs_indices` when batching `[E_picked, ...]` against `rhs_indices`. |
| **Sorted MoE Routing Omission (`B >= 2`)** | Unsorted `gather_qmm` lost `15–27%` (`41.85` vs `36.25 ms` @ `B=8`). | Pass `sorted_indices=True` with `mx.argsort` for all `M >= 2` `gather_qmm` paths. |
| **`0D` Scalar Arguments to `mx.fast.metal_kernel`** | `0`-dim MLX arrays compile as value scalars instead of device pointers (`const device T*` fails/garbage reads). | Reshape scalars via `.reshape(1)` or pass in a tiny `[N]` `Params` `uint32` array. |
| **Transposed weight layout** | Preparing contiguous weights reduced time from 145.98 to 126.69 ms in the measured kernel. | For kernels that benefit, prepare the required layout once at load time. For plain MLX projections, benchmark `mx.matmul(x, w.T)` with the existing weight layout before adding a copy. [Base M5 Gemma comparison](m4-vs-m5-architecture.md#base-m5-ocr-embeddings) |
| **Host/GPU Sync Inside Step Loop & Stale Arenas** | Per-step `mx.eval(scalar_off)` cost `97.4 -> 90.8 tok/s` (`B=1`); retaining `>256 MiB` MLX cache after waves slowed next calls. | Use `mx.async_eval(...)` on pure-FF/pipelined steps; `mx.clear_cache()` when `mx.get_cache_memory() > (256 << 20)`. |
| **Serial Post-Validation & Blocked `LoRA` Autograd** | Serial `decode+json+grammar` adds `~0.4–1.6s`; differentiating frozen prefix + unwrapped `k_proj/v_proj` + full-seq CE slows LoRA `2×+`. | Offload doc decode/JSON validation to P-core `ThreadPoolExecutor`; compute frozen prefix + slice CE to answer tail tokens outside `nn.value_and_grad`. |
