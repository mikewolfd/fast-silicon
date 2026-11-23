# fast-gpt-oss

Run `openai/gpt-oss-20b` locally on 48 GB+ Apple Silicon Macs for structured extraction, tool calling, or batch generation. On 25 extraction tasks on the 48 GB M4 Pro, schema-constrained `fast-gpt-oss` returned 24 valid responses and scored 0.653 field F1; unconstrained stock `mlx-lm` returned 17 valid responses and scored 0.591. On 25 tool calls the project passed 9 and stock passed 14 (49/50 vs 32/50 valid responses overall, **1.61×** the score per second). The two runs used different reasoning and output settings. [Task scores and comparison limits](../../docs/benchmarks/review-20261121.md#generation-and-model-precision)

`fast-gpt-oss` shares system-prompt KV caches across batch requests (`Hydragen v2`) and combines Metal 4 matrix-matmul and quantized vector kernels for the 32-expert MXFP4 MoE layers. That keeps a 20.9B model within **10.5 GB** of active MLX memory for single requests (`88.0 decode tok/s`) and **14.8 GB** at batch 128 (`454.1 decode tok/s`, `1.48 docs/s`), while retaining all 201,088 `o200k` input tokens. [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md)

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gpt-oss setup-weights --no-local-files-only
.venv/bin/gpt-oss verify
.venv/bin/gpt-oss generate "Explain Sliding Window Attention in two sentences."
```

## CLI, Parquet & HTTP server

```bash
# Schema-constrained JSON extraction from Parquet or JSONL
.venv/bin/gpt-oss parquet input.parquet out_dir --schema tests/fixtures/extract_v1.schema.json --drop-text-col
.venv/bin/gpt-oss extract input.jsonl out.jsonl --schema tests/fixtures/extract_v1.schema.json

# OpenAI-compatible HTTP server (sends completed text in one content event)
.venv/bin/gpt-oss serve --port 8080 --coalesce-ms 10 --max-batch 64

# Batch scaling benchmark & test suite
.venv/bin/gpt-oss bench --batches 8,64,128
.venv/bin/pytest -v
```

## Benchmark highlights (`M4 Pro 48 GB`)

All runs used `mlx-community/gpt-oss-20b-MXFP4-Q4` on the 48 GB M4 Pro (`macOS 26.6`). On Category 4, `fast-gpt-oss` enforces JSON schemas with `llguidance`, whereas stock `mlx-lm` decodes unconstrained.

### 1. Public outcome benchmarks

| Workload | `fast-gpt-oss` | Stock `mlx-lm` | Comparison |
|---|---|---|---|
| **Structured extraction + tool calls (50 tasks)** | **0.318 score/s** (49/50 valid, 0.506 score, 79.7 s, 12.8 GB) | 0.197 score/s (32/50 valid, 0.576 score, 146.2 s, 14.3 GB) | **1.61× score/s** · 1.83× faster · −1.5 GB RAM |
| **Reasoning & instruction following (75 tasks)** | **0.174 score/s** (40/75 valid, 0.307 score, 132.3 s, 12.3 GB) | 0.141 score/s (69/75 valid, 0.360 score, 191.0 s, 12.1 GB) | **1.23× score/s** · 1.44× faster |

A separate check with the actual GPT-OSS tokenizer and 131,072-token output vocabulary verified that every reference answer across 150 tool requests is expressible under both minified and colon-space JSON layouts; model accuracy under colon-space formatting on GPT-OSS remains open. [Grammar check](../../docs/benchmarks/document-workloads.json)

### 2. Batch scaling (`benchmark_monorepo.py`)

The project rows run eight extraction fixtures under a JSON schema with up to 512 output tokens. The stock row runs a repeated summary prompt with up to 128 output tokens.

| Implementation | Batch (documents) | Schema valid | Prefill | Decode / wall | Decode tok/s | Docs/s | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fast-gpt-oss`, sequential | 1 (8) | 8/8 | — | 16.10 s wall | 88.0 | 0.497 | 10.50 GB |
| `fast-gpt-oss`, batched | 8 (8) | 8/8 | 2.82 s | 6.22 s decode | 184.2 | 0.883 | 10.92 GB |
| `fast-gpt-oss`, batched | 64 (64) | 64/64 | 22.89 s | 24.40 s decode | 374.3 | 1.350 | 12.60 GB |
| **`fast-gpt-oss`, batched** | **128 (128)** | **128/128** | **46.14 s** | **39.87 s decode** | **454.1** | **1.483** | **14.79 GB** |
| Stock `mlx-lm`, summary prompt | 8 (8) | — | — | — | 156.1 | 1.188 | 11.24 GB |

By default, `fast-gpt-oss` keeps MoE router and combine reductions in FP32 (eliminating expert-selection drift on 8-bit checkpoints), uses hybrid FP16 intermediates with FP32 accumulators, and pairs `reasoning_effort="low"` with a compact chain-of-thought budget — scoring **0.726 on MMLU-Redux**, **0.54 on MATH-500**, and **0.847 on BFCL** in full-set ablations while running 3.3–4.4× faster than pure FP32. [Precision and CoT ablations](../../docs/BENCHMARKS.md#track-c--fast-gemma4--fast-gpt-oss-precision-loop-breakers--cot-budgets)

## Hardware sizing

| Unified memory | Policy ceiling | Recommended usage |
|---|---|---|
| **48 GB+ (`M4 Pro/Max`, `M5 Pro/Max`)** | 38.4 GB (80% UMA) | **Target host for `fast-gpt-oss` (`B=1..128`)** — resident MXFP4 weights (~11–12 GB) + Hydragen KV fit comfortably without swap. |
| **24 GB (`Base M5`)** | 19.2 GB | **Use `fast-gemma4` or `fast-qwen3` (`4B` / `9B`) for batch extraction.** The benchmark skips 20B sweeps on 24 GB hosts; if invoked directly, the engine caps waves at 16 requests on 24 GB and 8 on 16 GB. |

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **Hydragen v2 shared-prefix SDPA** | Computes shared system-prompt attention once across the whole batch with zero KV cloning | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Zero-sync hybrid MXFP4 MoE** | Routes all 24 MoE layers on GPU without CPU syncs, sending busy expert tiles to Metal 4 `MPP::matmul2d` and sparse tiles to coalesced quantized GEMV | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Circular SWA-128 & mixed precision** | Keeps the 24-layer residual stream in BF16, layer intermediates in FP16 with FP32 accumulators, and sliding-window KV in a fixed 128-slot ring | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Harmony prompt, schema & vocab** | Preserves all 201,088 `o200k` input tokens, prunes `lm_head` losslessly to 131,072 output tokens, and compacts finished batch slots mid-wave | [`engine.py`](src/fast_gpt_oss/engine.py), [`tokenizer.py`](src/fast_gpt_oss/tokenizer.py) |

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** (full hardware + outcome suites) and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** (CPU, tokenizer, and sizing contracts). LoRA training and ANE offload are not supported (sparse 32-expert MoE is a GPU workload).

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
