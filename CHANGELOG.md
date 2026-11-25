# Changelog

All notable changes to `fast-silicon` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-11-25

### Added
- **`fast-shared` single-pass GPU argmax (`fast_shared.argmax`):** Consolidated `gpu_argmax_conf`, `gpu_argmax_nan`, and `stats_to_logprob_margin` with a single-pass online-softmax Metal kernel (2x less vocab DRAM traffic), vocab-scaled threadgroup widths, pure-MLX CPU fallbacks, and IEEE bit-pattern `NaN`/`+Inf` handling.
- **`fast-qwen3` zero-copy rolling decode kernels (`MS1` + `MS2` + `MS3a`):**
  - `gdn_delta_update_slot_indexed`: Zero-copy slot-indexed `GdnDecodeShared` kernel reading persistent `RollingDeltaSlotCache` pools directly (`4.80x-6.22x` on `Ovis_18L`, `5.25x-5.89x` on `Qwen4B_24L` isolated GDN steps; bit-exact parity across `int8_blk32`, `bf16`, `fp32`).
  - `attend_rolling_flash_decode`: Zero-copy ragged Flash-Decode SDPA (`1-pass` + dynamic `Split-K`) for `RollingKVCache` (`3.77x-13.10x` on `Ovis_6L`, `5.79x-19.77x` on `Qwen4B_8L`, validated across both Apple9 M4 Pro and Apple10 M5 NAX at `cos >= 0.999997`).
  - `fused_gdn_pre_recurrence`: Fused Metal `Conv1d(k=4) + SiLU + L2 QK-Norm + GDN gates + conv_state update` (`2.10x-2.91x` on `Ovis_18L`, `2.79x-4.97x` on `Qwen4B_24L`) with per-slot `valid_lengths` protection and CPU fallback.
- **`fast-bge-small` Apple10 NAX varlen FlashAttention (`H1a`):** Added Metal 4 `MPP::matmul2d` `32x64` (`4` simdgroup) `_FA32_MPP_32x64_KERNEL` gated to Metal 4 + Apple10 hosts (`1.454x` attention speedup, lifting Base M5 `gpu` E2E to `173.2k tok/s` (`1.140x`) and `hetero` to `213.5k tok/s` (`1.052x`) at zero `NanoSciFact` diff).
- **`fast-pplx` fused attention prologue (`H2a`):** Added `_PPLX_QKV_NORM_ROPE_BCAST` fusing per-head FP32 QK RMSNorm, partial RoPE (`64` dims), gate split, and `[B, H, S, D]` transpose on GPU (`3.190x` prologue speedup, `1.229x-1.243x` across `6x PplxAttention`, `-41` to `-65 MB` MLX peak RAM, `+7.4%` GPU E2E / `+5.3-6.6%` `hetero2` E2E on Base M5).
- **CPU fallback & edge-case tests (`test_argmax_cpu.py`, `test_ms123_guards_cpu.py`, OCR/PPLX/Emb2 updates):** Added deterministic CPU tests for `+Inf`/`-Inf`/`NaN`/ties across `V in {1, 2, 17, 513}`, `Dk=256` threadgroup memory bounds, non-contiguous `RollingDeltaSlotCache` pool preservation, `valid_lengths` state protection, `3.15M` ViT cap escapes, and court-caption delimiter filtering.

### Changed
- **`fast-gpt-oss` default full vocabulary & M5 MoE routing (`Track B H1 + H2`):**
  - Switched `FastGptOssModel` and `FastGptOssEngine` default from `vocab_out=131072` to `vocab_out=None` (full `201K` `o200k` vocabulary), lifting `LiquidAI` structured extraction F1 by `+34.50 pp` (`0.6068 -> 0.9518`) at `1.236x` faster wall time while retaining explicit `vocab_out` opt-in pruning.
  - Restored `Hot-MPP + Cold-QMV` (`hot_threshold=8`, chunked at `256` tokens) in `_moe_forward` on Apple10 NAX and clamped `prefill_batch_suffixes` waves to bound `24 GB` memory, delivering **`1.709x` E2E (`2.734x` prefill, `1.389x` decode)** at `B=8` and `1.443x` at `B=4` on Base M5 with `-0.34 GB` MLX peak RAM.
- **`fast-ocr` default `3.15M` ViT pixel ceiling (`T3A`) & court-caption filter (`T3B`):**
  - Capped standard `compute_adaptive_max_pixels` pages at `3,145,728` px (`1.873x` faster `olmOCR-bench(21)`, `-1.94 GB` peak RAM at `0.0000` accuracy delta), while preserving the `16.78M` px escape for `"tiny_font_scan"` and `"low_vision_confidence"`.
  - Excluded isolated delimiter tokens (`(`, `)`, `[`, `]`, `{`, `}`, `|`, bullets, dashes) from layout column/band clustering in `analyze_page_from_lines`, improving `us_gov_ocr` judicial score by `+3.64 pp` (`0.9104 -> 0.9468`) with zero `olmOCR` or `FinTabNet` regressions.
  - Added compact-table `AR` (`B=16`) rolling dispatch (`MS4b`: `1.235x` on `FinTabNet-OTSL(30)`) and capped ANE `ProcessPoolExecutor` workers to `2` on `<= 28 GB` hosts (`4` on `>= 36 GB`).
  - Validated `OvisOCR2` balanced self-distilled GFM LoRA offline 8-bit merge (`v7final`, 0 overlap across 7 splits): `1.302x` faster `FinTabNet(30)` (`18.89s`, `TEDS=0.9031`, `NumF1=0.9184`, `0` loops), `+0.0076` on `OmniDocBench(12)` (`0.8234`), and parity on `olmOCR(16)` (`0.6993`).
- **`fast-embeddinggemma-2` widened M5 SWA `force_fused` (`H4b`):** Extended `should_force_fused_sdpa` on Apple10 `FP16` SWA `D=256` layers from `{384, 512}` to `seq_l >= 128` (`1.728x-1.932x` faster SWA attention, `-193` to `-301 MB` MLX peak, `+6.3%-8.1%` 24-layer `exec_packed` at `L=256/320`).
- **`fast-qwen3` rolling scheduling & 24 GB RAM guards:** Added `shortest_first` queue sorting (`1.172x` on `OmniDocBench-EN(12)`), `long_page_single_threshold_vis_tok=8000` outlier isolation, and `<= 28 GB` unified memory `b_slots` clamping in `rolling_multimodal_generate`.
- **Unified hardware probes:** Consolidated duplicate Apple10 NAX, `force_fused` SDPA, and Metal 4 capability checks across packages onto `fast_shared.hw_check`.

### Performance
- **`fast-qwen3` `Qwen3.5-4B-OptiQ-4bit` (`MS1 + MS2 + MS3a` stacked rolling decode):** `1.55x-2.12x` uniform decode (`503.5 tok/s` @ `B=64`) and `2.30x-3.88x` skewed decode (`488.7 tok/s` @ `B=64`), `1.12x-1.21x` E2E throughput on M4 Pro; `MS1` and `MS2` also verified faster on Base M5 `Apple10 NAX` across `B >= 4` / `hd=256`.

### Fixed
- **SDPA `force_fused` probe on MLX `0.32.x`:** Replaced broken `inspect.signature(mx.fast.scaled_dot_product_attention)` checks (which returned `False` on nanobind builtins in MLX `0.32.3/0.32.4`) with direct runtime probing in `fast_shared.hw_check` and `fast-pplx`.
- **+INFINITY `NaN`-poisoning in `gpu_argmax_conf`:** Fixed `+INFINITY - (+INFINITY) = NaN` in logsumexp accumulation via IEEE bitmask classification.
- **`fast-ir-guard --strict-bool-mask` compliance & CPU fallbacks:** Replaced `fp32` partial attention outputs with normalized `bf16`/`fp16` `o_part` + `fp32` `ml_part` in `attend_rolling_flash_decode`, enforced `<= 28,672 B` threadgroup sizing in `_effective_gdn_tile_rows`, preserved pool shapes on non-contiguous `RollingDeltaSlotCache` writes, and switched Metal kernels to lazy compilation with CPU fallbacks.
- **`fast-gpt-oss` NaN-safe greedy argmax:** Replaced unguarded `mx.argmax` in `FastGptOssEngine._eval_logits_for_sampling` with `gpu_argmax_nan` + safe EOS fallback instead of silently emitting token `0`.

## [2026-11-25] Wave-1 (`de8ed54`)

### Added
- On-GPU Metal SIMD `gpu_argmax_nan` and `gpu_argmax_conf` for `B=1` speculative verification in `fast-qwen3` and `fast-ocr` (`0.914` AUROC on `FinTabNet-30`, `+6.5%` to `+12.8%` `B=1` decode).
- Continuous rolling multimodal batching (`multimodal_rolling.py`) with batched speculative PLD (`K=4`), `1.27x-1.49x` faster and `-13%` to `-33.5%` peak RAM on mixed OCR batches.

## [2026-11-24] Wave-5, Prebuilt Artifacts & Benchmark Release

### Added
- Prebuilt Core ML ANE `.mlmodelc` and 8-bit MLX repositories (`mikewolfd/*`) wired across `setup_all.py` and package `setup-weights` commands.
- Wave-5 DRY optimizations (`c9bf185`) for `fast-qwen3` and `fast-pplx`, `Laya` 6-bucket FP16 LayerNorm repair, M4 Pro vs. Base M5 cross-silicon benchmark suite (`docs/BENCHMARKS.md`), benchmark SVG charts, and `CONTRIBUTING.md`.
