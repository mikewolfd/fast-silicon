# fast-qwen3

Run hybrid `3:1` `GatedDeltaNet` + Gated `GQA` Qwen3.5 and Qwen3.6 models (`0.8B`, `2B`, `4B`, `9B`, `27B` dense and `35B-A3B` sparse MoE) locally on Apple Silicon with strict JSON Schema grammars, `INT8` recurrent prefix caching, general mixed-bit `OptiQ` projection fusion, and hot-swappable `lora_accel`.

On 50 locked Category-4 public tasks (`LiquidAI` structured extraction + `BFCL v3` tool AST), `fast-qwen3` (`mlx-community/Qwen3.5-4B-OptiQ-4bit`) scored an identical **`0.9024` (`50/50` valid, `0.9248` field F1, `0.8800` AST, `0` repeat loops)** across both Macs — beating stock `mlx-lm 0.32.2` by **`+24.05 pp` and `1.53×` EOT (`0.513` vs `0.336`, `88.01 s` vs `98.56 s`)** on the 24 GB Base M5 and **`+27.99 pp` and `1.38×` EOT (`0.396` vs `0.286`, `5.18 GB` vs `8.02 GB` 4-bucket memory)** on the 48 GB M4 Pro. On an 8-record shared-prefix extraction wave, `4B-OptiQ` reached **`191.04 tok/s` (`4.84 docs/s`, `8/8` valid, `6.07×` wall speedup)** on M4 Pro and **`171.46 tok/s` (`4.34 docs/s`, `8/8` valid, `8.33×` wall speedup — `3.35×` Apple `CoreAI`'s `51.2 tok/s` `B=1` ceiling)** on Base M5 inside `4.46–4.49 GB` OS physical footprint; `Qwen/Qwen3.5-0.8B` (`BF16`) reached **`657.29 tok/s` (`16.23 docs/s`, `9.25×`)** on M4 Pro and **`573.37 tok/s` (`14.16 docs/s`, `11.66×`)** on Base M5 inside `2.86–2.99 GB`. [Cross-silicon benchmarks](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)

## Quick start

From the `fast-silicon` workspace root, install the package and stage weights (`mlx-community/Qwen3.5-4B-OptiQ-4bit` by default):

```bash
uv sync --python 3.13 --package fast-qwen3 --inexact
.venv/bin/fast-qwen3 setup-weights --preset 4b --allow-download
```

Extract schema-constrained JSON in Python:

```python
from contextlib import closing
from fast_qwen3 import FastQwen3Engine

schema = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["company", "year"],
    "additionalProperties": False,
}

with closing(FastQwen3Engine(model_dir="4b")) as engine:
    item = engine.generate(
        "Acme Robotics submitted its annual safety audit in 2025.",
        system_prompt="Extract the company name and filing year as strict JSON.",
        json_schema=schema,
        max_tokens=64,
    )
    if item.valid_schema:
        print(item.parsed)
```

## Cross-Silicon `ABBA` Highlights (`M4 Pro 48 GB` vs `Base M5 24 GB`)

Lock-serialized (`2`-round `ABBA`) measurements from [`docs/benchmarks/fast-qwen3-20261122.json`](../../docs/benchmarks/fast-qwen3-20261122.json):

| Model Preset | Workload | Base M5 (`24 GB`, `10c Apple10 NAX`) | M4 Pro (`48 GB`, `20c Apple9`) | Footprint (`M5` / `M4 Pro`) |
| :--- | :--- | :---: | :---: | :---: |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`) | Cat-4 Public Outcomes (`LiquidAI 25 + BFCL 25`) | **`0.513 EOT`** (`50/50` valid, **`0.9024`** vs stock `0.6619` / `0.336 EOT`, **`1.53×`**) | **`0.396 EOT`** (`50/50` valid, **`0.9024`** vs stock `0.6225` / `0.286 EOT`, **`1.38×`**) | `5.26 GB` / `5.18 GB` *(4-bucket)* |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`, default) | `B=1` strict JSON + LCP (`4` docs) | **`57.45 tok/s`** (`1.45 docs/s`, **`2.77×`**) | **`82.67 tok/s`** (`2.08 docs/s`, **`2.59×`**) | `4.46 GB` / `4.49 GB` |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`, default) | `B=8` strict JSON wave (`8` docs) | **`171.46 tok/s`** (`4.34 docs/s`, `8/8` valid, **`8.33×`**) | **`191.04 tok/s`** (`4.84 docs/s`, `8/8` valid, **`6.07×`**) | `4.46 GB` / `4.49 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=1` strict JSON + LCP (`4` docs) | **`138.09 tok/s`** (`3.39 docs/s`, **`2.80×`**) | **`196.21 tok/s`** (`4.82 docs/s`, **`2.75×`**) | `2.99 GB` / `2.86 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=8` strict JSON wave (`8` docs) | **`573.37 tok/s`** (`14.16 docs/s`, `8/8` valid, **`11.66×`**) | **`657.29 tok/s`** (`16.23 docs/s`, `8/8` valid, **`9.25×`**) | `2.99 GB` / `2.86 GB` |

## Architecture & Validated Design Choices

- **General Per-Layer `OptiQ` Spec-Bucketing ([`kernels.py`](src/fast_qwen3/kernels.py)):** Real `Qwen3.5-4B-OptiQ-4bit` checkpoints use heterogeneous per-tensor quantization (`(4, 8, 4, 4)` non-contiguous GatedDeltaNet projections, `(4, 4, 8)` / `(8, 4, 4)` / `(8, 4, 8)` Self-Attention projections, and mixed `gate_proj`/`up_proj` MLPs). `SpecBucketedLinearGroup` buckets compatible `(bits, group_size, has_biases)` sub-projections in arbitrary order with `0.0` ULP weight drift, plus `0.0`-ULP `[Q|Gate]` row deinterleaving (`split_linear_q_proj_to_q_gate`).
- **DRY `PrefixTree`, Right-Padded Post-LCP State Invariance & `kv_i8` ([`model.py`](src/fast_qwen3/model.py), [`engine.py`](src/fast_qwen3/engine.py)):** Shares `fast_shared.prefix_tree.PrefixTree` (`256`-token chunks + block-32 `INT8` `3.76×` GDN recurrent compression + rolling hit-rate circuit breaker), `LiveMemoryGovernor`, and auto-gated `INT8` KV storage (`kv_i8="auto"`). Right-padding post-LCP suffixes with exact `valid_lengths` `conv1d` window gathers + `(g=1, beta=0)` identity masking preserves exact `B=1` vs `B=4..64` parity (`4/4` parsed matches on ragged `LiquidAI` prompts).
- **Prefill-Folded `upfront_ff`, Multi-Token FF Coalescing, Active-Row `lm_head` & `A4_v2` Loop Breaker ([`engine.py`](src/fast_qwen3/engine.py)):** Folds deterministic schema prefixes into prefill, coalesces multi-token forced `llguidance` transitions (`step_w >= 2`) across wave slots, slices the `248,320`-row `lm_head` strictly to slots requiring sampling (`M_samp`), preserves `whitespace_flexible=None` (`": "` BPE separator: `0.9252` F1 vs `0.8309` minified), and applies `fast_shared.grammar.compute_loop_breaker_penalties` (`A4_v2` `p=1..8` cycles + `96`-tok 4-grams).
- **No-Think Default & Truncated-Thinking Valley Guard ([`tokenizer.py`](src/fast_qwen3/tokenizer.py)):** Defaults to `enable_thinking=False` (`<think>\n\n</think>\n\n`), which matches or beats `<think>` on structured JSON and tool calls at `3.2×` lower latency. When `enable_thinking=True` is requested for multi-step reasoning, `validate_thinking_budget` enforces a `>= 384`-token floor (`default 1024`) to prevent mid-thought `<think>` truncation collapse, or injects a bounded leading `reasoning_prefix_field` into strict JSON schemas.
- **Hot-Swappable `lora_accel` (`R2` Regime) & Double-Quantization Guard ([`kernels.py`](src/fast_qwen3/kernels.py), [`model.py`](src/fast_qwen3/model.py)):** Attaches PEFT/MLX LoRA adapters across `self_attn + mlp + linear_attn.{in_proj_qkv, in_proj_z, out_proj}` (`+17.5 pp` over `self_attn + mlp` alone while keeping `conv1d/A_log/dt_bias` frozen), and raises `QwenDoubleQuantError` if attempting destructive 2-pass `4-bit -> LoRA -> 4-bit` requantization.
- **1-Layer `Qwen3MtpDrafter` & Zero-GEMM GDN Rollback ([`mtp_drafter.py`](src/fast_qwen3/mtp_drafter.py)):** Loads native `mtp.*` / `optiq/mtp.safetensors` weights (`PromptLookupDrafter` fallback when absent) with exact zero-GEMM recurrent + 3-tap `conv1d` rollback (`verify_draft_sequence_b1`), auto-gated to `>= 9B` native-MTP models where backbone DRAM savings outweigh `248,320`-row `lm_head` draft projections (pass `speculative=True` to force on smaller presets).

## CLI & HTTP Server

```bash
.venv/bin/fast-qwen3 generate --model 4b --prompt "Summarize silicon photonics in JSON" \
  --schema '{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}'

.venv/bin/fast-qwen3 serve --model 4b --host 127.0.0.1 --port 8080
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16 --rounds 2
```
