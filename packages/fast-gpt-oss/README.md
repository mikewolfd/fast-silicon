# fast-gpt-oss

**Serve `openai/gpt-oss-20b` across 128 parallel documents on one Mac at 431–463 tok/s — 2.73x faster than `mlx_lm` (`B=8`) with 100% schema-valid JSON.**

`fast-gpt-oss` eliminates redundant shared-prompt KV cache copies and pairs Metal 4 GPU kernels with hybrid quantized MoE routing so a **20.9B MXFP4** model runs at **14.79 GB** peak (`B=128`, **1.413 docs/s** under host VM/CPU contention; **462.8 tok/s** standalone long-prompt) or **10.50 GB** single-stream (**87.5–97.4 tok/s**), keeping all `201K` input prompt tokens intact with zero misses. Ready out of the box with OpenAI-compatible SSE serving (`serve`), Parquet/JSONL batch extraction (`parquet` / `extract`), and CLI generation.

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gpt-oss setup-weights --no-local-files-only
.venv/bin/gpt-oss verify
.venv/bin/gpt-oss generate "Explain Sliding Window Attention in two sentences."
```

## At a glance

| Question | Answer |
|---|---|
| Does it match float schema/prefix outputs? | Yes: `gpt-oss verify` passes `b1_valid=4/4`, `b8_valid=8/8`, `prefix16_token_matches=4`, `key_field_matches=4`. |
| Peak batched throughput (`B=128`)? | **`431.0 dec tok/s`** (**`1.413 docs/s`**, `128/128` valid JSON, `14.79 GB` peak in `monorepo_bench` under host VM/CPU contention) = **`2.73x`** decode tok/s vs `stock-mlx_lm B=8` (`158.0 tok/s`). Standalone long-prompt (`L_sys=1,275`): **`462.8 dec tok/s`** (`0.710 docs/s`, `16.56 GB`). |
| Isolated `B=1` latency & memory? | **`87.5 dec tok/s`** (`0.495 docs/s`, `10.50 GB` `monorepo_bench` vs `74.2 tok/s` `stock-mlx_lm B=1`, `1.18x`); cool async `forward_b1` standalone: **`97.4 tok/s`**. |
| Are input tokens pruned? | No: input `embed_tokens` keeps all `201,088` `o200k` tokens (`0%` prompt miss); only `lm_head` prunes to lossless `131,072`. |

## Parity & Benchmarks (`mlx-community/gpt-oss-20b-MXFP4-Q4`, M4 Pro `48 GB`)

### 1. Monorepo Showcase & Public Baseline (`benchmark_monorepo.py`)

Suite Metal Peak **`15.880 GB`**, OS Footprint **`23.831 GB`** (`RUSAGE_INFO_V4`; recorded with `4` background quiescence warnings including `mediaanalysisd` CPU & `5.1 GB` active VM):

| Engine / Mode | Batch (Docs) | Valid | Prefill | Decode / Wall | Decode tok/s | Docs/s | Peak Mem | vs Stock `B=8` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b1_seq` | `1` (`8`) | `8/8` | — | `16.157 s` *(wall)* | `87.5` | `0.495` | **10.50 GB** | `0.55x` | **`1.18x`** vs `stock-mlx_lm B=1` (`74.2 tok/s`) |
| `batched_B8` | `8` (`8`) | `8/8` | `3.266 s` | `6.551 s` | `174.8` | `0.813` | `10.92 GB` | `1.11x` | Same batch size as `stock-mlx_lm B=8` (`-0.299 GB` Metal) |
| `batched_B64` | `64` (`64`) | `64/64` | `23.849 s` | `25.161 s` | `363.0` | `1.303` | `12.63 GB` | `2.30x` | Hydragen v2 + Hybrid MoE |
| **`batched_B128`** | `128` (`128`) | **`128/128`** | `48.268 s` | `42.010 s` | **431.0** | **1.413** | `14.79 GB` | **`2.73x`** | **`4.93x`** vs `b1_seq`; `2.85x` `docs/s` |
| `stock-mlx_lm` (Public) | `8` | — | — | — | `158.0` | `1.168` | `11.219 GB` | `1.00x` | `12.133 GB` OS (`load=2.533 s`) |
| `stock-mlx_lm` (Public) | `1` | — | — | — | `74.2` | — | — | `0.47x` | Single-stream baseline |

Monorepo `verify` (`20.389 s`, `18.863 GB` OS): `b1_valid=4/4` (`89.0 dec tok/s`, `0.433 docs/s`), `b8_valid=8/8` (`185.0 dec tok/s`, `0.888 docs/s`), `prefix16_token_matches=4`, `key_field_matches=4`.

### 2. Standalone Long-Prefix Suite (`gpt-oss bench --batches 8,64,128`)

`max_tokens=512`, structured JSON fixtures, `L_sys=1,275`, `vocab_out=131,072`:

| Batch / Mode | Docs Valid | Decode tok/s | Docs/s | Peak MLX Mem | Notes |
|---|---:|---:|---:|---:|---|
| `B=1` cool async (`forward_b1`) | — | **`97.4`** | `0.289` | — | Isolated uncontended `B=1` |
| `B=1` sequential (sustained) | `8/8` | `71.4` | `0.200` | **11.46 GB** | Sustained back-to-back |
| `B=8` batched | `8/8` | `116.8` | `0.295` | `12.63 GB` | `4/4` `key_field_matches` |
| `B=64` batched | `64/64` | `363.9` | `0.603` | `14.46 GB` | Hydragen v2 + Hybrid MoE |
| **`B=128` batched** | **`128/128`** | **462.8** | **0.710** | `16.56 GB` | **`+6.5x`** decode over sustained `B=1` |

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **Hydragen v2 Prefix SDPA** | `hydragen_p1_mma` + `hydragen_p2_varlen` computes shared `O_sys/LSE_sys` once (`0` prefix KV cloning across batch `B`), with `L_sys == 0` fallback | [`kernels.py`](src/fast_gpt_oss/) |
| **Zero-Sync Hybrid MXFP4 MoE** | Single-warp `moe_build_tiles` (`0` GPU->CPU router syncs over 24 layers); hot `(expert, tile)` (`9..32`) -> Metal 4 `MPP::matmul2d` (`BM=32`), cold (`1..8`) -> coalesced QMV with register E2M1 + E8M0 unpack | [`kernels.py`](src/fast_gpt_oss/) |
| **Circular SWA-128 & Mixed Prec** | `rmsnorm_bf16_to_f16` keeps 24-layer residual in `bfloat16`, activations in `float16`; YaRN RoPE + circular `128`-slot SWA ring | [`kernels.py`](src/fast_gpt_oss/) |
| **Harmony, Schema & Vocab** | `131,072` `lm_head` pruning, `reasoning_effort=skip\|none\|low\|medium\|high`, `reasoning_budget`, `llguidance` lazy 4-byte argmax + mid-wave compaction + `forward_b1` tail handoff | [`engine.py`](src/fast_gpt_oss/), [`tokenizer.py`](src/fast_gpt_oss/) |

## CLI, Parquet, Server & Tests

```bash
# Constrained JSON extraction from Parquet or JSONL
.venv/bin/gpt-oss parquet input.parquet out_dir --schema tests/fixtures/extract_v1.schema.json --drop-text-col
.venv/bin/gpt-oss extract input.jsonl out.jsonl --schema tests/fixtures/extract_v1.schema.json

# Persistent OpenAI-compatible HTTP server (ChatMicroBatcher + SSE streaming)
.venv/bin/gpt-oss serve --port 8080 --coalesce-ms 10 --max-batch 64

# CPU + hardware tests (hardware runs require serialized /tmp/m4_gpu_validation.lock)
.venv/bin/pytest -v
```

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB` unified memory)**. Fits alongside `16.56 GB` peak MLX allocation at `B=128`; on `16–24 GB` hosts (`M4`/`M5`), `select_gpt_oss_plan` automatically clamps default wave batch sizes (`<= 8` on `16 GB`, `<= 16` on `24 GB`) and bounds wired memory to preserve OS headroom.
- **Not tested:** Training/LoRA fine-tuning, CoreML ANE offload (weights fit Metal MoE/QMV bandwidth directly), or multi-node execution.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
