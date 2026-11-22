# fast-gpt-oss

Run `openai/gpt-oss-20b` locally for structured extraction or generation. On 25 M4 Pro extraction tasks the project returned 24 valid responses and scored 0.653 field F1. Stock MLX returned 17 and scored 0.591. On 25 tool calls the project passed 9 and stock passed 14. Across both tasks they produced 49 and 32 valid outputs respectively. The two runs had different reasoning and output settings. [Task scores and comparison limits](../../docs/benchmarks/review-20261121.md#generation-and-model-precision)

`fast-gpt-oss` eliminates redundant shared-prompt KV cache copies (`Hydragen v2`) and pairs Metal 4 `Hot-MPP / Cold-QMV` quantized MoE routing with `llguidance` grammar constraints so a **20.9B MXFP4** model runs at **`12.6–15.9 GB`** active MLX RAM (`23.98 GB` OS footprint at `B=128`, **`1.483 docs/s`**) or **`10.50 GB`** single-stream (`88.0–97.4 dec tok/s`), keeping all `201,088` `o200k` input prompt tokens intact with zero misses. Ready out of the box with OpenAI-compatible HTTP serving (`serve`), resumable Parquet/JSONL batch extraction (`parquet` / `extract`), and CLI generation.

> **Full Benchmarks & Methodology:** For complete cross-silicon tables (`M4 Pro 48 GB` vs `Base M5 24 GB`), all 16 deterministic public outcome datasets, 4-bucket `RUSAGE_INFO_V6` footprints, and reproducible CLI commands, see the monorepo [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

A check with the actual GPT-OSS tokenizer and shipped output vocabulary found a correctly scored answer that both minified and fixed-colon-space grammars could express for each of 150 tool requests. Changing the separator has a measured effect on Gemma. Its effect on GPT-OSS prediction accuracy remains open. [Tool outcomes and grammar check](../../docs/benchmarks/document-workloads.json)

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gpt-oss setup-weights --no-local-files-only
.venv/bin/gpt-oss verify
.venv/bin/gpt-oss generate "Explain Sliding Window Attention in two sentences."
```

## Cross-Silicon Benchmark Highlights (`M4 Pro 48 GB`)

The benchmark ran on the 48 GB M4 Pro with MLX 0.32.4 and the `mlx-community/gpt-oss-20b-MXFP4-Q4` checkpoint. The public outcome runs used the same tasks. The batch-capacity probes used different prompts and output budgets. [Benchmark methods](../../docs/BENCHMARKS.md)

### 1. External Deterministic Objective Outcomes (`benchmark_outcomes.py`, Tier-1)

| Category & Dataset | `fast-gpt-oss` (`Hot-MPP/Cold-QMV` + Hydragen + `llg`) | Stock `mlx-lm` (`gpt-oss-20b-MXFP4-Q4`) | `fast-gpt-oss` vs Stock |
|---|---|---|---|
| **Cat 4: JSON & Tool AST** (`LiquidAI/structured-extraction + BFCL v3`, `N=50`) | **`0.318 EOT`** (**`49/50` valid = `98%`**, `79.7 s`, `S_obj=0.5063`, `12.59 GB` MLX / `12.77 GB` `phys_footprint`, `2` repeat loops) | `0.197 EOT` (`32/50` valid = `64%`, `146.2 s`, `S_obj=0.5755`, `14.48 GB` MLX / `14.30 GB` `phys_footprint`, `4` repeat loops) | **`1.61x` `EOT`** · **`+53%` valid docs** · **`1.83x` faster** · **`-1.53 GB` RAM** |
| **Cat 3: Reasoning & IF** (`IFBench + MMLU-Redux-2.0 + MATH-500`, `N=75`) | **`0.174 EOT`** (`132.3 s`, `S_obj=0.3067`, `11.94 GB` MLX / `12.26 GB` `phys_footprint`) | `0.141 EOT` (`191.0 s`, `S_obj=0.3600`, `12.97 GB` MLX / `12.07 GB` `phys_footprint`) | **`1.23x` `EOT`** · **`1.44x` faster** |

### 2. Batched Throughput Scaling (`benchmark_monorepo.py` & `gpt-oss bench`)

Monorepo showcase (`Mac16,8`, `48 GB` M4 Pro; suite Metal Peak **`15.88 GB`**, OS Footprint **`23.98 GB`**):

The project rows use eight extraction fixtures, a JSON schema and up to 512 output tokens. The stock row uses a repeated summary prompt and up to 128 output tokens. The first four rows show how the project engine scales as the batch grows.

| Implementation | Batch (documents) | Schema valid | Prefill | Decode / wall | Decode tokens/s | Documents/s | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fast-gpt-oss`, sequential | 1 (8) | 8/8 | — | 16.100 s wall | 88.0 | 0.497 | 10.50 GB |
| `fast-gpt-oss`, batched | 8 (8) | 8/8 | 2.823 s | 6.218 s decode | 184.2 | 0.883 | 10.92 GB |
| `fast-gpt-oss`, batched | 64 (64) | 64/64 | 22.887 s | 24.398 s decode | 374.3 | 1.350 | 12.60 GB |
| `fast-gpt-oss`, batched | 128 (128) | 128/128 | 46.141 s | 39.868 s decode | 454.1 | 1.483 | 14.79 GB |
| Stock `mlx-lm`, summary prompt | 8 (8) | — | — | — | 156.1 | 1.188 | 11.24 GB |

- **Standalone long-prefix suite (`gpt-oss bench --batches 8,64,128`, `L_sys=1,275`, `max_tokens=512`, `vocab_out=131,072`):** reaches **`462.8 dec tok/s`** (`0.710 docs/s`, `128/128` valid, `16.56 GB` peak MLX) at `B=128` vs `71.4 tok/s` (`0.200 docs/s`, `11.46 GB`) sustained `B=1` (**`6.5x`** decode scaling).
- **Parity (`gpt-oss verify`):** passes `b1_valid=4/4` (`89.0 dec tok/s`), `b8_valid=8/8` (`185.0 dec tok/s`), `prefix16_token_matches=4`, and `key_field_matches=4`.
- **v2.2 `MoE FP32` router/combine & kernel gates:** Keeping MoE routing & weighted combine in `float32` on `gpt-oss-20b-MXFP4-Q8` (`3072` rows) cuts router set/order mismatch from `0.130%/0.488%` to `0.0%/0.0%` on M4 Pro (`bf16_round_diff 22.66% -> 0.003%`) and `0.065%/0.358%` to `0.0%/0.065%` on Base M5. `MPP` `4-bit` (`0.69x..0.76x` on `gpt-oss-20b`) and `col_gemv` (`0.781x..0.961x` M5) stay gated OFF. [v2.2 validation](../../docs/benchmarks/v22-validation-20261121.json)
- **v2.3 Precision & `canon_low` CoT Budget Validation:** Paired step-0 and outcome sweeps confirmed existing hybrid `f16` 10-bit intermediates + `f32` accumulators (`0.999338` mean cos / `1.8109` max diff / `12/12` top-1 vs `true_fp32`, whereas `bf16` intermediates doubled drift to `3.5235` and `MidOut` `bf16` was `7.5x` worse). On outcome suites, **`canon_low`** (`reasoning_effort="low"`, `_EFFORT_DEFAULT_BUDGET["low"]=48`, benchmark budgets `192/64/384/64/96`, 4-gram `window=64`) achieved **`0.19` IFBench** (`inst 0.2321`), **`0.7263` MMLU-Redux** (`207/285`, `137.18s`), **`0.54` MATH-500** (`167.64s`), **`0.8128` LiquidAI F1** (`150` valid), and **`0.8467` BFCL AST** (`150` valid) — beating `skip`, `natural_low` (`budget=0`, `0.4421` MMLU), `tight_48`, `med` (`0.6702` MMLU, `0.48` MATH, `0.13` IFBench), `true_fp32` (`3.34x–4.42x` slower, `0.7228` MMLU), and `bf16_inter` (`3.08x–3.92x` slower, `0.7018` MMLU). [v2.3 validation](../../docs/benchmarks/v23-validation-20261121.json)

## Hardware Sizing Guidance

| Host Unified Memory | Wired / Metal Budget | Recommended Engine & Regime | Why |
|---|---|---|---|
| **`48 GB+` (`M4 Pro/Max`, `M5 Pro/Max`)** | **`38.4 GB`** (`80%` UMA ceiling) | **`fast-gpt-oss` (`B=1..128`)** — sweet spot for `20B` MoE | Resident `MXFP4` weights (`~10.7–12 GB`) + Hydragen KV + Metal arenas take `10.5–16.6 GB` active MLX (`12.3–24.0 GB` OS footprint at `B=128`), fitting cleanly inside `38.4 GB` with zero macOS swap thrashing. |
| **24 GB base M5** | 19.07 GB Metal working set, 19.2 GB policy ceiling | The suite directs batch extraction to `fast-gemma4` | The GPT-OSS benchmark skipped the 24 GB host. The engine’s sizing policy caps a wave at 16 requests on 24 GB and eight on 16 GB. [Sizing code](src/fast_gpt_oss/engine.py) |

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **Hydragen v2 Prefix SDPA** | `hydragen_p1_mma` + `hydragen_p2_varlen` computes shared `O_sys/LSE_sys` once (`0` prefix KV cloning across batch `B`), with `L_sys == 0` fallback | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Zero-Sync Hybrid MXFP4 MoE** | Single-warp `moe_build_tiles` (`0` GPU->CPU router syncs over 24 layers); hot `(expert, tile)` (`9..32`) -> Metal 4 `MPP::matmul2d` (`BM=32`), cold (`1..8`) -> coalesced QMV with register E2M1 + E8M0 unpack | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Circular SWA-128 & Mixed Prec** | `rmsnorm_bf16_to_f16` keeps 24-layer residual in `bfloat16`, activations in `float16`; YaRN RoPE + circular `128`-slot SWA ring | [`kernels.py`](src/fast_gpt_oss/kernels.py) |
| **Harmony, Schema & Vocab** | Full `201,088` `o200k` input embeddings (`0%` prompt miss), lossless `131,072` `lm_head` pruning, `reasoning_effort=skip\|none\|low\|medium\|high`, `llguidance` lazy 4-byte argmax + mid-wave compaction + `forward_b1` tail handoff | [`engine.py`](src/fast_gpt_oss/engine.py), [`tokenizer.py`](src/fast_gpt_oss/tokenizer.py) |
| **Parquet & SSE Serving** | Resumable Parquet/JSONL structured extraction and persistent OpenAI-compatible HTTP server (`ChatMicroBatcher` + SSE) | [`pipeline.py`](src/fast_gpt_oss/pipeline.py), [`server.py`](src/fast_gpt_oss/server.py) |

## CLI, Parquet, Server & Tests

```bash
# Constrained JSON extraction from Parquet or JSONL
.venv/bin/gpt-oss parquet input.parquet out_dir --schema tests/fixtures/extract_v1.schema.json --drop-text-col
.venv/bin/gpt-oss extract input.jsonl out.jsonl --schema tests/fixtures/extract_v1.schema.json

# Persistent HTTP server (chat sends the completed text in one content event)
.venv/bin/gpt-oss serve --port 8080 --coalesce-ms 10 --max-batch 64

# Package throughput sweep (see monorepo docs/BENCHMARKS.md for Tier-1 outcome benchmarks)
.venv/bin/gpt-oss bench --batches 8,64,128

# CPU + hardware tests (hardware runs require serialized /tmp/m4_gpu_validation.lock)
.venv/bin/pytest -v
```

## Scope & Not-tested

- **Verified hardware:** Full hardware inference, parity, and `Tier-1` outcome benchmarks verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** under `/tmp/m4_gpu_validation.lock`. On **Base Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`)**, CPU/tokenizer/contract tests pass (`0` failures) while full `20B` batch benchmarks are intentionally memory-gated (see [Hardware Sizing Guidance](#hardware-sizing-guidance)).
- **Not tested:** Training/LoRA fine-tuning, CoreML ANE offload (sparse `32`-expert MoE is an ANE anti-target; weights fit Metal MoE/QMV directly), or multi-node execution.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
