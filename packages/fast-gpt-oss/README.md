# fast-gpt-oss

Hardware-limit Apple Silicon (`M4 Pro`, Metal 4 `MPP::matmul2d` + SIMD) inference engine for **`openai/gpt-oss-20b`** (`mlx-community/gpt-oss-20b-MXFP4-Q4`), built DRY atop [`fast-shared`](../fast-shared).

## Architecture

- **Custom Metal 4 + SIMD Kernels (`fast_gpt_oss.kernels`)** composing `fast_shared.metal_mpp.HEADER_MPP`:
  - `rmsnorm_bf16_to_f16`: keeps the 24-layer residual stream in `bfloat16` (`>65K`-safe) while writing post-norm activations in `float16` (`7.59 TF` ALU/cooperative throughput).
  - **Hydragen v2 Split-Prefix SDPA** (`hydragen_p1_mma` + `hydragen_p2_varlen`): computes `O_sys`/`LSE_sys` once against shared prefix `K_sys`/`V_sys` with `simdgroup_matrix` MMA (`0` prefix KV cloning across batch `B`), then combines variable-length private suffixes (`SufLens[b]`), decoded tokens, and attention `Sinks` cleanly (including `L_sys == 0` fallback when prompts have no common prefix).
  - **Batched YaRN RoPE + Circular SWA-128** (`yarn_rope_split_b`, `swa128_prefill_mma`, `swa128_decode`): per-slot positions, banded causal SWA prefill MMA, and circular `128`-slot ring decode handling both partial (`<128`) and full (`>=128`) windows.
  - **Zero-Sync Metal 4 Hybrid MoE** (`moe_build_tiles`, `moe_batch_gu_unified`, `moe_batch_dn_unified`, `moe_reduce4_resid_bf16`): single-warp GPU tile binning (`0` GPU->CPU router syncs across 24 layers), routing hot `(expert, tile)` groups (`9..32` items) to Metal 4 `MPP::matmul2d` (`BM=32`) and cold tails (`1..8` items) to coalesced multi-column QMV, using register E2M1 subnormal unpacking + `(scale + 14) << 23` E8M0 exponent folding with zero LUTs.
- **Asymmetric Output Vocab Pruning (`fast_gpt_oss.tokenizer`)**: keeps full `201,088` input embeddings (`0%` prompt miss) while pruning `lm_head` to `131,072` lossless ASCII/JSON/punctuation/byte/Harmony tokens.
- **Harmony & Constrained JSON (`fast_gpt_oss.engine`, `fast_shared.grammar`)**: supports `reasoning_effort=skip|none|low|medium|high`, optional `reasoning_budget` analysis->final transition injection, `llguidance` lazy 4-byte argmax speculative checks, upfront + per-step `compute_ff_tokens()` skipping, mid-wave batch compaction, and `B=1` tail handoff (`extract_b1_cache` -> `forward_b1`).
- **OpenAI Server & Parquet/JSONL Pipelines (`fast_gpt_oss.server`, `fast_gpt_oss.pipeline`)**: thin DRY adapters on `fast_shared.server` (`/v1/chat/completions`, `/v1/completions`, SSE streaming, `ChatMicroBatcher`, `UnifiedOpenAIServer` registration) and `fast_shared.pipeline_llm`.

## Quickstart

```bash
uv venv --python cpython-3.13.14
uv pip install -e ../fast-shared -e ".[dev]"

# Resolve/download mlx-community/gpt-oss-20b-MXFP4-Q4 & build cached o200k vocab remap
gpt-oss setup-weights --no-local-files-only

# Verify B=1 and B=8 parity + strict JSON schema compliance on included structured JSON fixtures
gpt-oss verify

# Single prompt / chat
gpt-oss generate "Explain Sliding Window Attention in two sentences."

# Constrained JSON extraction from Parquet or JSONL
gpt-oss parquet input.parquet out_dir --schema tests/fixtures/extract_v1.schema.json --drop-text-col

# OpenAI-compatible HTTP server
gpt-oss serve --port 8080 --coalesce-ms 10 --max-batch 64
```

## Tests

```bash
.venv/bin/pytest -v
```

## Measured Apple M4 Pro Results (`mlx-community/gpt-oss-20b-MXFP4-Q4`, `48 GiB`, `vocab_out=131,072`)

Verified via `gpt-oss verify` and `gpt-oss bench --batches 8,64,128` (`max_tokens=512`, structured JSON extraction fixtures with `L_sys=1,275`, sustained back-to-back run):

- **Parity (`gpt-oss verify`):** `b1_valid=4/4`, `b8_valid=8/8`, `prefix16_token_matches=4/4`, `key_field_matches=4/4`
- **`B=1` sequential (`8` docs):** `8/8` valid, `71.4` decode tok/s, `0.200 docs/s`, `11.46 GB` peak MLX memory *(cool single-stream async `forward_b1` peak: `97.4 tok/s` / `0.289 docs/s`)*
- **`B=8` batched (`8` docs):** `8/8` valid, `116.8` decode tok/s, `0.295 docs/s`, `12.63 GB` peak
- **`B=64` batched (`64` docs):** `64/64` valid, `363.9` decode tok/s, `0.603 docs/s`, `14.46 GB` peak
- **`B=128` batched (`128` docs):** `128/128` valid, `462.8` decode tok/s, `0.710 docs/s`, `16.56 GB` peak
