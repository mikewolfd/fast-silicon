# fast-gpt-oss

Run `openai/gpt-oss-20b` (`MXFP4` 32-expert MoE) locally on 24-48 GB+ Apple Silicon Macs (with `B=8` auto-clamp on 16 GB) for structured JSON extraction, tool calling, and batched generation. Shares system-prompt KV caches across batch slots (`Hydragen v2`) and uses hybrid Metal 4 matrix-matmul and quantized vector kernels to fit a 20.9B model into **12.9-13.8 GB** peak RAM on a 24 GB Base M5 (**168.2 tok/s** at `B=16`) and scale to **444.1 tok/s** at `B=128` (`15.96 GB` MLX / `16.67 GB` peak) on a 48 GB M4 Pro.

Zero telemetry, no account, and runs 100% offline after weight download (~11-12 GB `MXFP4-Q8` default checkpoint). [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md)

## Requirements

- Apple Silicon Mac (`M1-M5`), verified on 24-48 GB+ unified memory (`16 GB` auto-clamps to `B=8`, unverified on 16 GB hardware), macOS 14+ (GPU-only MLX; `macOS 15+` for the full monorepo ANE stack; verified on macOS 26.6 and 27.2.1), Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported.

## Quickstart (~11-12 GB model download)

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-gpt-oss --inexact

.venv/bin/gpt-oss setup-weights --no-local-files-only
.venv/bin/gpt-oss verify
.venv/bin/gpt-oss generate "Explain Sliding Window Attention in two sentences."
```

## Hardware memory sizing

| Unified memory | Auto wave-batch cap | Measured footprint & expected usage |
| --- | --- | --- |
| **48 GB+ (`M4/M5 Pro/Max`)** | `B=64` default (up to `B=128`) | Verified `15.96 GB` MLX / `16.67 GB` peak @ `B=128` (**444.1 tok/s**, **1.568 docs/s** in the monorepo suite). |
| **24 GB (`Base M5`)** | `B=16` (`<= 26 GiB` clamp) | Verified `12.90 GB` Metal / `13.78 GB` OS (**50.0 tok/s** `B=1`, **129.8 tok/s** `B=8`, **168.2 tok/s** `B=16`). |
| **16 GB / 32 GB** | `B=8` (16 GB) / `B=32` (32 GB) | Automatically clamps batch waves from host RAM so ~11 GB resident weights stay out of swap (auto-clamp only; unverified on 16 GB hardware). |

## Benchmark highlights

### 1. Structured JSON, tool calls & reasoning (`M4 Pro 48 GB`, recorded `MXFP4-Q4` run)

The current default checkpoint is `MXFP4-Q8`, while the outcome comparison table below records the verified `MXFP4-Q4` run prior to the Q8 default promotion. On public extraction and tool tasks, `fast-gpt-oss` enforces JSON schemas via `llguidance`, whereas stock `mlx-lm` runs unconstrained and loses points on invalid JSON. Because both runs also used different reasoning/output settings, valid counts and quality are listed explicitly ([comparison notes](../../docs/benchmarks/review-20261121.md#generation-and-model-precision)):

| Workload | `fast-gpt-oss` | Stock `mlx-lm` | Comparison |
| --- | --- | --- | --- |
| **Structured extraction + tools (50 tasks)** | **0.318 score/s** (49/50 valid, 0.506 score, 79.7 s, 12.8 GB) | 0.197 score/s (32/50 valid, 0.576 score, 146.2 s, 14.3 GB) | **1.61x score/s**, 1.83x faster wall, -1.5 GB RAM |
| **Reasoning (`75` MMLU + MATH + IFBench)** | **0.174 score/s** (40/75 valid, 0.307 score, 132.3 s, 12.3 GB) | 0.141 score/s (69/75 valid, 0.360 score, 191.0 s, 12.1 GB) | **1.23x score/s**, 1.44x faster wall |

Split on the 50 tasks: `24/25` valid extraction (`0.653` F1) vs. stock `17/25` (`0.591` F1), and `9/25` correct tools vs. stock `14/25`. Default checkpoints keep the full `201K` `o200k` output vocabulary (`vocab_out=None`, lifting Base M5 `LiquidAI` extraction F1 by `+34.50 pp` to `0.9518` at `1.24x` faster wall time with NaN-safe GPU argmax, while keeping explicit `--vocab-out 131072` pruning opt-in). Restoring `Hot-MPP + Cold-QMV` (`hot_threshold=8`) on Apple10 NAX adds **`1.709x` E2E (`2.734x` prefill)** at `B=8` (`11.69 GB` MLX peak) and `1.443x` at `B=4` on Base M5.

### 2. Batch scaling (`benchmark_monorepo.py`, 512-token JSON schema fixtures)

| Host | Mode / Batch (docs) | Schema valid | Prefill | Decode / wall | Decode tok/s | Docs/s | Memory (active MLX / peak comb) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **M4 Pro (48 GB)** | Sequential `B=1` (8) | 8/8 | - | 16.10 s wall | 88.0 | 0.497 | 10.50 GB MLX |
| **M4 Pro (48 GB)** | Batched `B=8` (8) | 8/8 | 2.82 s | 6.22 s decode | 184.2-190.7 | 0.883 | 10.92 GB MLX |
| **M4 Pro (48 GB)** | Batched `B=64` (64) | 64/64 | 22.89 s | 24.40 s decode | 374.3-385.8 | 1.350 | 12.60 GB MLX |
| **M4 Pro (48 GB)** | **Batched `B=128` (128)** | **128/128** | **46.14 s** | **39.87 s decode** | **444.1** | **1.568** | **15.96 GB MLX / 16.67 GB peak** |
| **Base M5 (24 GB)** | `B=1` / `B=8` / **`B=16`** | - | - | - | `50.0 / 129.8` / **168.2** | **0.742** (`B=16`) | **12.90 GB Metal / 13.78 GB OS** |

*Baseline note: An earlier capacity suite run reached `454.1 dec tok/s` @ `B=128` with a `23.98 GB` suite footprint. Separately, on a non-matching `128`-token summary prompt (without a 512-token extraction schema), stock `mlx-lm` (`B=8`) measured `156.1 tok/s` (`1.188 docs/s`, `11.24 GB`).*

By default, `fast-gpt-oss` keeps MoE router and combine reductions in FP32 (preventing expert-selection drift on 8-bit checkpoints), uses hybrid FP16 intermediates with FP32 accumulators, and pairs `reasoning_effort="low"` (`budget=48`) with 4-gram loop breaking: scoring **0.726 on MMLU-Redux**, **0.54 on MATH-500**, and **0.847 on BFCL** at `3.3-4.4x` faster wall time than pure FP32. [Precision & CoT ablations](../../docs/BENCHMARKS.md#track-c--fast-gemma4--fast-gpt-oss-precision-loop-breakers--cot-budgets)

## What keeps it fast

| Component | How it works | Source |
| --- | --- | --- |
| **Hydragen v2 shared-prefix SDPA** | Computes shared system-prompt attention once across the whole batch with zero KV cloning | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Zero-sync hybrid MXFP4 MoE** | Routes all 24 MoE layers on GPU without CPU syncs, splitting dense expert tiles to Metal 4 `MPP::matmul2d` (on M5 `Apple10 NAX`) and sparse tiles to coalesced QMV | [`kernels.py`](src/fast_gpt_oss/kernels.py), [`model.py`](src/fast_gpt_oss/model.py) |
| **Circular SWA-128 & mixed precision** | Keeps residuals in BF16, layer intermediates in FP16 with FP32 accumulators, and sliding-window KV in a fixed 128-slot ring | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Harmony prompt, schema & vocab** | Trims trailing non-vocab checkpoint padding rows beyond `tokenizer.json` under `vocab_out=None` (skipping unused cold LM head rows, with optional `vocab_out=131072` pruning), uses NaN-safe GPU/CPU argmax, and compacts finished slots mid-wave | [`engine.py`](src/fast_gpt_oss/engine.py), [`model.py`](src/fast_gpt_oss/model.py), [`tokenizer.py`](src/fast_gpt_oss/tokenizer.py) |

## CLI, Parquet & HTTP server

```bash
.venv/bin/gpt-oss parquet input.parquet out_dir --schema packages/fast-gpt-oss/tests/fixtures/extract_v1.schema.json --drop-text-col
.venv/bin/gpt-oss extract input.jsonl out.jsonl --schema packages/fast-gpt-oss/tests/fixtures/extract_v1.schema.json
.venv/bin/gpt-oss serve --port 8080 --coalesce-ms 10 --max-batch 64
.venv/bin/gpt-oss bench --batches 8,64,128
.venv/bin/pytest -v packages/fast-gpt-oss/tests   # Requires --extra dev
```

## Limitations, Prior Art & License

- **Limitations:** LoRA fine-tuning and Neural Engine offload are not supported (sparse 32-expert MoE is GPU-only). While 24 GB Macs run `B=1..16` cleanly, large `B=64-128` batch sweeps want 32-48 GB+ RAM. On 25 tool-calling tasks, the existing run trailed unconstrained `mlx-lm` accuracy (`9/25` vs. `14/25`); evaluate on your target prompt set before relying on low-effort reasoning budgets. HTTP `serve` returns completed JSON payloads rather than token-by-token streams.
- **Prior art:** Built on `mlx`, `mlx-lm`, `llguidance`, and OpenAI's `gpt-oss-20b` (`MXFP4-Q8` / `MXFP4-Q4` MLX checkpoints).
- **License:** Package code is [MIT licensed](LICENSE); `openai/gpt-oss-20b` weights carry the Apache-2.0 license. Verified on 48 GB M4 Pro and 24 GB Base M5.
