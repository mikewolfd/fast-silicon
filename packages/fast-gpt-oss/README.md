# fast-gpt-oss

**Serve `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) across `128` parallel documents on a `48 GB` Mac at `454.1–462.8 dec tok/s` — `2.91x` faster than stock `mlx_lm` (`B=8`) and `1.61x` higher `EOT` (`+53%` valid structured docs) on public JSON/tool benchmarks.**

`fast-gpt-oss` eliminates redundant shared-prompt KV cache copies (`Hydragen v2`) and pairs Metal 4 `Hot-MPP / Cold-QMV` quantized MoE routing with `llguidance` grammar constraints so a **20.9B MXFP4** model runs at **`12.6–15.9 GB`** active MLX RAM (`23.98 GB` OS footprint at `B=128`, **`1.483 docs/s`**) or **`10.50 GB`** single-stream (`88.0–97.4 dec tok/s`), keeping all `201,088` `o200k` input prompt tokens intact with zero misses. Ready out of the box with OpenAI-compatible SSE serving (`serve`), resumable Parquet/JSONL batch extraction (`parquet` / `extract`), and CLI generation.

> **Full Benchmarks & Methodology:** For complete cross-silicon tables (`M4 Pro 48 GB` vs `Base M5 24 GB`), all 16 deterministic public outcome datasets, 4-bucket `RUSAGE_INFO_V6` footprints, and reproducible CLI commands, see the monorepo [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gpt-oss setup-weights --no-local-files-only
.venv/bin/gpt-oss verify
.venv/bin/gpt-oss generate "Explain Sliding Window Attention in two sentences."
```

## Cross-Silicon Benchmark Highlights (`M4 Pro 48 GB`)

All runs execute in isolated subprocesses under `/tmp/m4_gpu_validation.lock` (`MLX 0.32.4`) on `mlx-community/gpt-oss-20b-MXFP4-Q4`, paired against stock `mlx-lm`. See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for full breakdowns.

### 1. External Deterministic Objective Outcomes (`benchmark_outcomes.py`, Tier-1)

| Category & Dataset | `fast-gpt-oss` (`Hot-MPP/Cold-QMV` + Hydragen + `llg`) | Stock `mlx-lm` (`gpt-oss-20b-MXFP4-Q4`) | `fast-gpt-oss` vs Stock |
|---|---|---|---|
| **Cat 4: JSON & Tool AST** (`LiquidAI/structured-extraction + BFCL v3`, `N=50`) | **`0.318 EOT`** (**`49/50` valid = `98%`**, `79.7 s`, `S_obj=0.5063`, `12.59 GB` MLX / `12.77 GB` `phys_footprint`, `2` repeat loops) | `0.197 EOT` (`32/50` valid = `64%`, `146.2 s`, `S_obj=0.5755`, `14.48 GB` MLX / `14.30 GB` `phys_footprint`, `4` repeat loops) | **`1.61x` `EOT`** · **`+53%` valid docs** · **`1.83x` faster** · **`-1.53 GB` RAM** |
| **Cat 3: Reasoning & IF** (`IFBench + MMLU-Redux-2.0 + MATH-500`, `N=75`) | **`0.174 EOT`** (`132.3 s`, `S_obj=0.3067`, `11.94 GB` MLX / `12.26 GB` `phys_footprint`) | `0.141 EOT` (`191.0 s`, `S_obj=0.3600`, `12.97 GB` MLX / `12.07 GB` `phys_footprint`) | **`1.23x` `EOT`** · **`1.44x` faster** |

### 2. Batched Throughput Scaling (`benchmark_monorepo.py` & `gpt-oss bench`)

Monorepo showcase (`Mac16,8`, `48 GB` M4 Pro; suite Metal Peak **`15.88 GB`**, OS Footprint **`23.98 GB`**):

| Engine / Mode | Batch (Docs) | Valid | Prefill | Decode / Wall | Decode tok/s | Docs/s | Peak MLX Mem | vs Stock `B=8` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b1_seq` | `1` (`8`) | `8/8` | — | `16.100 s` *(wall)* | `88.0` | `0.497` | **10.50 GB** | `0.56x` | **`1.09x`** vs `stock-mlx_lm B=1` (`80.7 tok/s`); cool async `forward_b1`: **`97.4 tok/s`** |
| `batched_B8` | `8` (`8`) | `8/8` | `2.823 s` | `6.218 s` | `184.2` | `0.883` | `10.92 GB` | `1.18x` | Same batch size as `stock-mlx_lm B=8` (`-0.32 GB` Metal) |
| `batched_B64` | `64` (`64`) | `64/64` | `22.887 s` | `24.398 s` | `374.3` | `1.350` | `12.60 GB` | `2.40x` | Hydragen v2 + Hybrid MXFP4 MoE |
| **`batched_B128`** | `128` (`128`) | **`128/128`** | `46.141 s` | `39.868 s` | **454.1** | **1.483** | `14.79 GB` | **`2.91x`** | **`5.16x`** vs `b1_seq` (`2.98x` `docs/s`); `23.98 GB` suite OS footprint |
| `stock-mlx_lm` (Public) | `8` | — | — | — | `156.1` | `1.188` | `11.24 GB` | `1.00x` | Public MLX baseline (`B=1`: `80.7 tok/s`) |

- **Standalone long-prefix suite (`gpt-oss bench --batches 8,64,128`, `L_sys=1,275`, `max_tokens=512`, `vocab_out=131,072`):** reaches **`462.8 dec tok/s`** (`0.710 docs/s`, `128/128` valid, `16.56 GB` peak MLX) at `B=128` vs `71.4 tok/s` (`0.200 docs/s`, `11.46 GB`) sustained `B=1` (**`+6.5x`** decode scaling).
- **Parity (`gpt-oss verify`):** passes `b1_valid=4/4` (`89.0 dec tok/s`), `b8_valid=8/8` (`185.0 dec tok/s`), `prefix16_token_matches=4`, and `key_field_matches=4`.

## Hardware Sizing Guidance

| Host Unified Memory | Wired / Metal Budget | Recommended Engine & Regime | Why |
|---|---|---|---|
| **`48 GB+` (`M4 Pro/Max`, `M5 Pro/Max`)** | **`38.4 GB`** (`80%` UMA ceiling) | **`fast-gpt-oss` (`B=1..128`)** — sweet spot for `20B` MoE | Resident `MXFP4` weights (`~10.7–12 GB`) + Hydragen KV + Metal arenas take `10.5–16.6 GB` active MLX (`12.3–24.0 GB` OS footprint at `B=128`), fitting cleanly inside `38.4 GB` with zero macOS swap thrashing. |
| **`<= 24 GB` (`Base M4 / M5`)** | **`19.07 GB`** Metal cap (`19.2 GB` `80%` ceiling) | **Route structured batch workloads to `fast-gemma4`** (`6.1–6.5 GB` active MLX) | `20B` MoE large batches push right against the `19.07 GB` Metal cap (`fast-gpt-oss` is gated in `24 GB` benchmark suites, and [`select_gpt_oss_plan`](src/fast_gpt_oss/engine.py#L165-L196) clamps wave batch sizes to `<= 16` on `24 GB` / `<= 8` on `16 GB`). On `24 GB` Base M5, `fast-gemma4` runs `B=128` inside `6.13 GB` MLX (`15.54 GB` OS) at **`2.717 EOT`** (`8.5x` higher Cat-4 `EOT`) and **`10.46 valid docs/s`** (`7.1x` `1.483 docs/s`). |

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

# Persistent OpenAI-compatible HTTP server (ChatMicroBatcher + SSE streaming)
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
