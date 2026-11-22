# Training, Distillation & Post-Training Playbook (M4 & M5+)

Hardware-grounded patterns for accelerating SFT, LoRA/QLoRA, teacher-student distillation, encoder/head fine-tuning, speculative draft heads (`MTP`), quantization calibration (`AWQ-MSE` / QAT), and RLVR (`GRPO` / `DPO`) on Apple Silicon (`Apple9` M4 and `Apple10+` M5 `NAX`).

---

## Roofline Shift: Why Training Behaves Differently From Decode

1. **Compute (`3×` GEMMs/token) + Activation-RAM Bound:** Autoregressive decode (`M=1`) is pure DRAM-bandwidth-bound. Training runs **three dense/quantized GEMMs per token** (`Y = XW`, `dX = dY W^T`, `dW = X^T dY`) at high `M` (`512..4096`), making forward+backward **matrix-TFLOPS-bound** on linear projections and **DRAM-bandwidth + wired-RAM-capacity-bound** on saved activations, norms, `GeGLU`/`SwiGLU` VJPs, and large-vocab cross-entropy.
2. **Why `M5+` (`Apple10` `NAX`) Scales Training Harder Than Decode:** On M4 Pro (`Apple9`), shader `simdgroup_matrix` caps `BF16` at `7.61 TF` (`20c`). On M5 (`Apple10`), per-core `NAX` matrix accelerators hit `11.66 TF` even on a 10-core Base M5 (`1.53×` 20c M4 Pro) and `2×–4×` higher on M5 Pro/Max (`MLX >= 0.32.0` built against `macOS >= 26.2`).
3. **Hard UMA Swap Cliff (`75–80%` Physical RAM Ceiling):** Keep total wired footprint (`weights + optimizer + saved activations + MLX arena`) strictly below `~18 GiB` on `24 GiB` Macs and `~38 GiB` on `48 GiB` Macs. Spilling autograd activations into macOS compressed memory / NVMe swap degrades step times `20×–60×` (`~0.9s -> 56s/step`).

---

## 1. Local Apple Silicon Training Workload Matrix

| Workload | Primary Bottleneck | Target Silicon Split |
|---|---|---|
| **Causal LLM LoRA / QLoRA SFT & Prompt Distillation** | `3×` high-`M` GEMMs, `nn.utils.checkpoint` recompute, `200K–262K` `lm_head` CE | **GPU (`MLX` `BF16` / `Q8` / `MXFP4`)** for trainable upper trunk; **ANE (`pipe_depth=4`)** optional for frozen lower prefix |
| **Teacher -> Student Soft-Logit Distillation** | Teacher forward passes (`8–15` candidates/doc) + frozen lower student trunk | **GPU Rolling / `PrefixTree` Engine** for teacher + **GPU (`MLX`)** for student upper layers + **NVMe/ANE** for frozen trunk |
| **Dense Encoder / Router Fine-Tuning & OOF Calibration** | Pad-token waste on `S=256..512`, unfused head VJPs, `FP32` SDPA traps | **GPU (`MLX` varlen packed `BF16`)** for fine-tuning + OOF linear probes; **parallel `coremltools`** for final `.mlmodelc` |
| **Speculative Draft-Head (`MTP` / Assistant) Training** | Recomputing frozen `35–42L` target model just to train a `4L` (`150 MiB`) head | **Stream precomputed `last_storing_layer` hidden/KV states** from NVMe (`~6 GB/s`) into **GPU (`MLX`)** |
| **PTQ Calibration (`AWQ-MSE` / `SmoothQuant`), QAT & Palettization** | Activation collection + per-channel clipping search + serial CoreML bucket builds | **GPU batched forward** for Hessian/MSE stats + **CPU `ProcessPoolExecutor`** for parallel `s64..s512` `.mlmodelc` compilation |
| **Verifier-Guided Preference / RLVR (`GRPO` / `DPO` / Best-of-N)** | `80–85%` wall-clock spent in `G=4..8` rollout generation + critic scoring | **`fast-silicon` Rolling GPU Engine (`B_slots=128`)** + **concurrent `100%`-ANE reward critics** (`0` GPU contention) |

---

## 2. Mandatory Baseline Hygiene (Verify Before Anything Else)

1. **Split Frozen Lower Prefix Outside `nn.value_and_grad`:**
   - When fine-tuning upper layers (`L_k..L_{N-1}`), execute `L_0..L_{k-1}` (and frozen per-layer auxiliary embeddings like `PLE`) in a plain helper *before* entering `nn.value_and_grad`, or apply `mx.stop_gradient` immediately at the boundary. Never build autograd tapes or retain forward activations across frozen bottom blocks.
2. **Align SFT JSON Formatting & Tokenizer Boundaries With Serving `llguidance` Grammar:**
   - Format SFT ground-truth targets identically to the serving grammar (`separators` + `whitespace_flexible`) and validate `Qwen3.5` BPE / `Gemma-4` SentencePiece token alignment on held-out examples so fine-tuned checkpoints do not fight runtime constrained decoding.
3. **Unwrap Dead KV-Shared LoRA Modules (and Guard Rectangular `GatedDeltaNet` LoRA Shapes):**
   - On KV-sharing models (`Gemma-4 E2B` layers `15..34`, `E4B` layers `24..41`), generic `linear_to_lora_layers` attaches `LoRALinear` wrappers to `k_proj`/`v_proj` even on layers that reuse earlier KV tensors. Unwrap dead `k_proj`/`v_proj` adapters before optimizer init (`40` modules saved on `E2B`) so `AdamW` does not allocate or step dead states. On `Qwen3.5` `in_proj_a`/`in_proj_b` (`out < r` possible), match loaded adapter matrices against module `(in_features, out_features)` instead of heuristic shape inequalities.
4. **Answer-Only `lm_head` Gathering + Sliced `FP32` Cross-Entropy:**
   - On prompt-heavy SFT (`1,000–2,000` prompt tokens, `150–300` answer tokens), gather hidden states `h_ans` strictly at supervised answer indices *before* multiplying by `[D, V]` (`V = 162K..262K`). This cuts `lm_head` forward+backward FLOPs `5×–8×`.
   - Slice `FP32` `tanh` logit softcapping + `logsumexp` + NLL into `256`-token chunks (`~2.0 GB` lower peak wired RAM than materializing `[S_full, V]` `FP32` logits).
5. **Eliminate Per-Step Weight Casts & `FP32` SDPA Upcasts (`fast-ir-guard`):**
   - Pre-cast frozen trunk weights to `mx.bfloat16` **once at load time**, never via `w.astype(mx.bfloat16)` inside `__call__` on every step.
   - Keep trainable transformer/head attention inputs and masks in `BF16`/`FP16` (`mx.where`/additive `BF16` bias); casting hidden states or masks to `mx.float32` inside self-attention triggers a **`-36%` register-spill slowdown** on both forward and backward passes.

---

## 3. High-ROI Apple Silicon Training Accelerations (Ranked)

### 3.1 Varlen Sequence Packing + RoPE Position Reset (`1.5×–2.6×`)
- **Why `B=1` and Padded Batches Underperform:** `batch_size=1` starves GPU matrix tiles on short examples (`M=300..900`) and recompiles MLX graphs across ragged lengths, while right-padded `B=2..8` wastes FLOPs and activation RAM on `[PAD]` tokens—crossing the `ckpt_tokens` activation-checkpointing threshold early.
- **Implementation:** Pack multiple `(prompt, answer)` sequences end-to-end into fixed **`M = 2,048` or `4,096` token bins**:
  - Reset RoPE `position_ids` per packed segment (`0..L_1-1, 0..L_2-1, ...`).
  - Apply a `BF16`/boolean block-diagonal segment mask (`seg_id[:, None] == seg_id[None, :]`) combined with causal masking so tokens never attend across document boundaries.
  - Gather supervised target positions across all packed answers into one compact `h_ans` slice.

### 3.2 QLoRA (`8-bit g64` / `MXFP4` Base + `BF16` Adapters) to Eliminate Checkpointing (`1.25×–1.4×`)
- **Checkpoint Tax:** `nn.utils.checkpoint` trades `+30%–35%` step time (a second forward pass per checkpointed layer during backward) to fit `BF16` weights + activations in RAM.
- **Implementation:** Freeze base linear weights in **`8-bit g64`** (`cos >= 0.9998`) while keeping LoRA `(A, B)`, norms, embeddings, and activations in `BF16`:
  - Cuts base weight wired RAM by `50%` (`~9.1 -> ~4.8 GB` on `E2B`, `~16 -> ~8.5 GB` on `E4B`) and halves weight-read DRAM traffic during `mx.quantized_matmul` VJP (`dX`).
  - Raises the safe `--ckpt-tokens` ceiling from `~2,200` to `4,096+`, avoiding activation recomputation on standard packed bins.
  - For **sparse MoE QLoRA (`gpt-oss-20b`)**, keep expert banks in `MXFP4`/`Q4` and enforce `sorted_indices=True` + explicit `lhs_indices` on all `M >= 2` `gather_qmm` forward and VJP paths.

### 3.3 ANE Co-Execution or NVMe Activation Caching for Frozen Trunks (`1.3×–2.2×`)
CoreML/ANE cannot execute backward passes, leaving the **16-core ANE `100%` idle** during GPU training unless explicitly scheduled:
- **Pattern A — Software-Pipelined ANE Prefix Prefetch:** When bottom layers are frozen (`L0..11` on `Gemma-4`, `L0..17` on `ModernBERT`), run batch `t+1`'s frozen prefix on static-bucket ANE (`pipe_depth=4`, zero-copy `IOSurface -> Metal` `newBufferWithIOSurface` handoff) concurrently with batch `t`'s GPU upper-layer forward+backward pass.
- **Pattern B — Activation Caching Across Multi-Epoch / Sweep Runs:** When training upper heads (`L18..27` validator heads, linear/MLP routers, or `4`-layer `MTP` draft heads) for multiple epochs or hyperparameter sweeps, materialize `h_{frozen}` once in `BF16` (or per-token symmetric `int8` + `fp16` scale) and stream from NVMe (`~6 GB/s`), eliminating `60%–90%` of forward FLOPs completely.

### 3.4 Reusing a training prefix within gradient accumulation

LoRA weights stay fixed across the micro-batches in one optimizer step. Compute the shared prefix keys and values once, let each document suffix attend to them, and accumulate prefix gradients before the update. The Gemma benchmark used 12 examples with 1,683 supervised answer tokens without reuse and eight examples with 1,120 answer tokens with reuse. Compare the same examples and updates before assigning a speedup to the cache.

### 3.5 `@mx.compile` Fused Chunked CE, Elementwise VJPs & `AdamW` (`15%–30%`)
- **Compiled Softcap + Cross-Entropy (`Liger`-style):** Wrap the `256`-token `(bf16_logits -> fp32 -> tanh_softcap -> logsumexp -> nll)` function in `@mx.compile` so forward and VJP fuse into registers without streaming `[256, V]` `FP32` intermediate tensors to/from DRAM (`~3×` less head memory traffic).
- **Compiled Elementwise VJPs & Optimizer Step:** Apply `@mx.compile` to pure-elementwise `RMSNorm` / `GeGLU` / `SwiGLU` subgraphs and the `AdamW` parameter update (`150+` LoRA moment pairs), collapsing hundreds of tiny DRAM-bound Metal launches per step.

### 3.6 `PrefixTree` & Rolling Engines for Distillation, `GRPO` & Post-Training Builds (`3×–6×`)
- **Multi-Candidate Teacher Distillation & Conformal/Confidence Calibration:** When a teacher (`Kev-0.8B`, `E4B`, `gpt-oss-20b`, `Qwen3.5`) scores `8–15` corruptions or candidate spans per document, snapshot the shared document state in a `PrefixTree` (`256`-token `GDN` block-32 `INT8` + `KV` blocks, captured right after chunk insert) so the teacher prefills each document **once** instead of `10×`. Use held-out `OOF` telemetry to calibrate `CROQ` conformal routing sets (`alpha=0.05`, `aps_det`) and `ExtractConf` escalation thresholds (`strict_calibration_gate=True`).
- **Online `GRPO` / Best-of-N Rollouts:** Drive rollout generation with `fast-silicon`'s persistent rolling slot engine (`B_slots=128`, `llguidance` + lazy `argmax`) while evaluating reward/schema critics concurrently on ANE (`0` GPU contention).
- **Parallel Core ML compilation.** Four workers cut the build time for nine BGE shapes from 9.37 to 4.38 s on M4 Pro and 5.02 to 2.24 s on M5 (with identical output vectors across all nine shapes on 16 inputs). For two larger Gemma bundles, the second M4 Pro builds took 26.37 s serial and 25.97 s parallel. Compare actual output vectors from the compiled bundles as well as elapsed time.

### 3.7 Multi-Mac LoRA Data-Parallel (`mx.distributed` Over Thunderbolt / TB5 RDMA)
- Because LoRA adapter gradients are compact (`~20–60 MB` in `BF16`), `mx.distributed.all_sum` once per `grad_accum` step takes `< 5 ms` over Thunderbolt / TB5 RDMA (`macOS >= 26.2`), delivering `~1.8×–1.9×` near-linear SFT and distillation scaling across heterogeneous Macs (weighing shard sizes by each Mac's measured `TFLOPS`).

---

## 4. Training Verification & Anti-Target Guardrails

1. **Never attempt backprop or per-step dynamic weight compilation on ANE:** CoreML `.mlmodelc` graphs bake static weights and take seconds to recompile. Keep gradients strictly on Metal (`MLX`).
2. **Prefer `MLX` over PyTorch `MPS` for Transformer/LoRA training:** `MPS` lacks several fused `BF16` backward kernels (incurring silent CPU fallbacks) and fragments unified memory earlier near the wired-memory limit.
3. **Verify loss & gradient parity before scaling:** On any packed-attention, `QLoRA`, ANE-prefix, or `@mx.compile` change, assert step-0 loss parity (`|Δloss| < 1e-3`), gradient cosine (`>= 0.999` on adapter tensors), peak wired GB (`mx.get_peak_memory()`), and post-SFT blind downstream `docs/s × accuracy` (`llguidance` enabled).
