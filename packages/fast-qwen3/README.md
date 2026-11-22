# fast-qwen3

Run hybrid `3:1` `GatedDeltaNet` + Gated `GQA` Qwen3.5 and Qwen3.6 models (`0.8B`, `2B`, `4B`, `9B`, `27B` dense and `35B-A3B` sparse MoE) locally on Apple Silicon with strict JSON Schema grammars, `INT8` recurrent prefix caching, general mixed-bit `OptiQ` projection fusion, and hot-swappable `lora_accel`.

On an 8-record structured extraction wave with a shared system prompt, `fast-qwen3` (`mlx-community/Qwen3.5-4B-OptiQ-4bit`) reached **`146.96 tok/s` (`3.72 docs/s`, `8/8` valid, `5.15×` wall speedup)** on the 48 GB M4 Pro and **`140.01 tok/s` (`3.54 docs/s`, `8/8` valid, `6.72×` wall speedup)** on the 24 GB Base M5 inside `4.44–4.49 GB` OS physical footprint — `2.73×` the `51.2 tok/s` `B=1` ceiling of Apple `CoreAI` (`fp4_fp8`) while retaining full logit access, strict `llguidance` schemas, and batched execution. On `Qwen/Qwen3.5-0.8B` (`BF16`), batched waves reached **`491.93 tok/s` (`12.15 docs/s`, `7.86×`)** on M4 Pro and **`460.73 tok/s` (`11.34 docs/s`, `9.30×`)** on Base M5 inside `2.92–2.98 GB` physical footprint. [Cross-silicon benchmarks](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)

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

| Model Preset | Workload | Base M5 (`24 GB`, `10c Apple10 NAX`) | M4 Pro (`48 GB`, `20c Apple9`) | Physical Footprint (`M5` / `M4 Pro`) |
| :--- | :--- | :---: | :---: | :---: |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`, default) | `B=1` strict JSON + LCP (`4` docs) | **`51.59 tok/s`** (`1.30 docs/s`, **`2.50×`**) | **`66.43 tok/s`** (`1.67 docs/s`, **`2.29×`**) | `4.44 GB` / `4.49 GB` |
| **`Qwen3.5-4B-OptiQ-4bit`** (`4b`, default) | `B=8` strict JSON wave (`8` docs) | **`140.01 tok/s`** (`3.54 docs/s`, `8/8` valid, **`6.72×`**) | **`146.96 tok/s`** (`3.72 docs/s`, `8/8` valid, **`5.15×`**) | `4.44 GB` / `4.49 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=1` strict JSON + LCP (`4` docs) | **`130.22 tok/s`** (`3.20 docs/s`, **`2.64×`**) | **`155.28 tok/s`** (`3.81 docs/s`, **`2.53×`**) | `2.98 GB` / `2.92 GB` |
| **`Qwen3.5-0.8B`** (`0.8b-bf16`) | `B=8` strict JSON wave (`8` docs) | **`460.73 tok/s`** (`11.34 docs/s`, `8/8` valid, **`9.30×`**) | **`491.93 tok/s`** (`12.15 docs/s`, `8/8` valid, **`7.86×`**) | `2.98 GB` / `2.92 GB` |

## Architecture & Validated Design Choices

- **General Per-Layer `OptiQ` Spec-Bucketing ([`kernels.py`](src/fast_qwen3/kernels.py)):** Real `Qwen3.5-4B-OptiQ-4bit` checkpoints use heterogeneous per-tensor quantization (`(4, 8, 4, 4)` non-contiguous GatedDeltaNet projections, `(4, 4, 8)` / `(8, 4, 4)` / `(8, 4, 8)` Self-Attention projections, and mixed `gate_proj`/`up_proj` MLPs). `SpecBucketedLinearGroup` buckets compatible `(bits, group_size, has_biases)` sub-projections in arbitrary order with `0.0` ULP weight drift, plus `0.0`-ULP `[Q|Gate]` row deinterleaving (`split_linear_q_proj_to_q_gate`).
- **`INT8` Hybrid Prefix Snapshots & Left-Pad Invariance ([`model.py`](src/fast_qwen3/model.py), [`engine.py`](src/fast_qwen3/engine.py)):** Compresses `FP32` `[N_v, D_k, D_v]` `GatedDeltaNet` recurrent states `4×` into symmetric per-head `INT8` + `FP16` scales alongside `KV` slices, and preserves `B=1` parity on ragged waves via `valid_token_scale` + true per-slot `25%` partial RoPE positions.
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
