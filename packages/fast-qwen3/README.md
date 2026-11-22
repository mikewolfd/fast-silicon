# fast-qwen3

Run hybrid `3:1` `GatedDeltaNet` + Gated `GQA` Qwen3.5 and Qwen3.6 models (`0.8B`, `2B`, `4B`, `9B`, `27B` dense and `35B-A3B` sparse MoE) locally on Apple Silicon with strict JSON Schema grammars, `INT8` recurrent prefix caching, general mixed-bit `OptiQ` projection fusion, and hot-swappable `lora_accel`.

On 50 locked Category-4 public tasks (`LiquidAI` structured extraction + `BFCL v3` tool AST), `fast-qwen3` (`mlx-community/Qwen3.5-4B-OptiQ-4bit`) with `llguidance` grammar constraints scored an identical **`0.9024` (`50/50` valid, `0.9248` field F1, `0.8800` AST, `0` repeat loops)** on both Macs. Unconstrained stock `mlx-lm 0.32.2` on the same checkpoint produced `37/50` valid responses on the 24 GB Base M5 (`0.6619` score, `0.336 EOT` vs `fast-qwen3` **`0.513 EOT`, `1.53×`**) and `35/50` valid on the 48 GB M4 Pro (`0.6225` score, `0.286 EOT` vs **`0.396 EOT`, `1.38×`**, `5.18 GB` vs `8.02 GB` 4-bucket memory). `9B-OptiQ --mtp-drafter` also cleared `50/50` valid (`0.8857–0.8874`), whereas zero-shot `Cat-3` with `enable_thinking=False` scored `0.533–0.573` due to `IFBench` adherence (`0.16–0.24`).

On shared-prefix `B=8` strict JSON waves, `4B-OptiQ` reached **`191.04 tok/s`** on M4 Pro and **`171.46 tok/s`** on Base M5 (`4.46–4.49 GB` OS phys); `9B-OptiQ` reached **`112.90–115.93 tok/s`** (`B=8`) and **`41.43` / `50.17 tok/s`** (`MTP` `k=3`, `248K` / `65K`) on M4 Pro vs **`101.23–105.74 tok/s`** and **`32.44` / `42.25 tok/s`** (`16.18–16.68 GB` peak `< 19.2 GB`) on Base M5; and `0.8B` reached **`767.46 tok/s`** (M4 Pro) / **`658.54 tok/s`** (Base M5) with decoded `65K` vocab remapping. At `16K` context (`B=4`), online `QuaRot` 4-bit KV (`kv_i8="q4_quarot"`) compresses KV memory **`3.56×`** (`576 MB` vs `2,048 MB` `BF16`) while decoding at **`91.71 tok/s`** (**`2.13×` faster** than legacy `kv_i8` `43.05 tok/s`) with `0.00 pp` quality loss (`0.9464 == BF16` `LiquidAI` F1, `18/18` passkeys). [Cross-silicon benchmarks](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22) · [v2.5 QuaRot KV](../../docs/BENCHMARKS.md#123-fused-quantized-kv-sdpa-quarot-4-bit-kv-fast-pplx-ffdhetero2--kev-4b-specbucket)

## Quick start

From the `fast-silicon` workspace root, install the package and stage weights (`mlx-community/Qwen3.5-4B-OptiQ-4bit` by default):

```bash
uv sync --python 3.13 --package fast-qwen3 --inexact
.venv/bin/fast-qwen3 setup-weights --preset 4b --allow-download
```

Extract schema-constrained JSON in Python:

```python
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

with FastQwen3Engine(model_dir="4b") as engine:
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
| **`Qwen3.5-0.8B`** (`0.8b-bf16`, `248K` -> `65K`) | `B=8` strict JSON wave (`8` docs) | **`573.37` -> `658.54 tok/s`** (`16.31 docs/s`, `8/8` valid) | **`657.29` -> `767.46 tok/s`** (`18.95 docs/s`, `8/8` valid) | `2.99–4.81 GB` / `2.86–4.99 GB` |
| **`Qwen3.5-9B-OptiQ-4bit`** (`9b-optiq`, `248K`, `k=3`) | `B=1` `MTP` / `B=8` strict JSON wave | **`32.44 tok/s`** (`1.68×` `MTP`) · **`101.23 tok/s`** (`B=8`, `8.44×`) | **`41.43 tok/s`** (`1.28×` `MTP`) · **`112.90 tok/s`** (`B=8`, `5.76×`) | `8.29 GB` (`16.18` peak) / `8.31 GB` (`16.36` peak) |
| **`Qwen3.5-9B-OptiQ-4bit`** (`9b-optiq`, `65K`, `k=3`) | `B=1` `MTP` / `B=8` strict JSON wave | **`42.25 tok/s`** (`1.92×`, `k`-sweep `44.29`) · **`105.74 tok/s`** (`B=8`) | **`50.17 tok/s`** (`1.40×`, `k`-sweep `58.40`) · **`115.93 tok/s`** (`B=8`) | `16.68 GB` / `16.87 GB` peak |
| **`Qwen3.5-9B-OptiQ-4bit`** (`9b-optiq --mtp-drafter`) | Cat-4 Public Outcomes (`LiquidAI 25 + BFCL 25`) | **`0.402 EOT`** (`50/50` valid, **`0.8857`**) | **`0.277 EOT`** (`50/50` valid, **`0.8874`**) | `8.39 GB` (`10.94` peak) / `8.79 GB` (`12.41` peak) |
| **`Qwen3.5-4B-OptiQ-4bit`** (`kv_i8="q4_quarot"`) | `16K` long-context decode (`B=1` / `B=4`, `LiquidAI` + `PK`) | `0.9464 == BF16` F1 (`18/18` passkeys) | **`42.09 tok/s`** (`B=1`) · **`91.71 tok/s`** (`B=4`, **`2.13×`** vs legacy `kv_i8` `43.05`) | **`576 MB` KV** vs `2,048 MB` `BF16` (**`3.56×` smaller**) |
| **`4B-OptiQ GPU` + `FastBgeSmall ANE`** | Concurrent M5 wave (`8×48` LLM + `3×32×384` ANE) | `2.626 s` wall (`< 19.2 GB` cap) | — | `5.32 GB` (`9.39 GB` peak) / — |

*(Apples-to-apples note: on Cat-4, `fast-qwen3` enforces JSON Schema constraints via `llguidance`, whereas stock `mlx-lm 0.32.2` decodes unconstrained without grammar enforcement and scores `0` on invalid JSON outputs. `16K` `QuaRot` KV receipts are in [`docs/benchmarks/v25-rd-winners-20261122.json`](../../docs/benchmarks/v25-rd-winners-20261122.json); `v2.4` receipts are in [`docs/benchmarks/runs/`](../../docs/benchmarks/runs/).)*

## Architecture & Validated Design Choices

- **General Per-Layer `OptiQ` Spec-Bucketing ([`kernels.py`](src/fast_qwen3/kernels.py), `fast_shared.kv_quant`):** Real `Qwen3.5-4B-OptiQ-4bit` checkpoints use heterogeneous per-tensor quantization (`(4, 8, 4, 4)` non-contiguous GatedDeltaNet projections, `(4, 4, 8)` / `(8, 4, 4)` / `(8, 4, 8)` Self-Attention projections, and mixed `gate_proj`/`up_proj` MLPs). `SpecBucketedLinearGroup` (shared in `fast_shared.kv_quant` across `fast-qwen3` and `fast-decision-models`) buckets compatible `(bits, group_size, has_biases)` sub-projections in arbitrary order with `0.0` ULP weight drift, plus `0.0`-ULP `[Q|Gate]` row deinterleaving (`split_linear_q_proj_to_q_gate`).
- **DRY `PrefixTree`, `QuaRot` `kv_i8="q4_quarot"` & Fused `qmm` `kv_i8="q8"` ([`model.py`](src/fast_qwen3/model.py), [`engine.py`](src/fast_qwen3/engine.py), `fast_shared.kv_quant`):** Shares `fast_shared.prefix_tree.PrefixTree` (`256`-token chunks + block-32 `INT8` `3.76×` GDN recurrent compression + rolling hit-rate circuit breaker), `LiveMemoryGovernor`, and `QuantizedKVCache` (`fused_quantized_sdpa` executing `mx.quantized_matmul` directly on packed KV). Online post-RoPE `[Q, K, V]` `FWHT-64` Metal `simd_shuffle_xor` (`kv_i8="q4_quarot"`, inverted before `fused_sigmoid_attn_gate`) delivers **`3.56×` KV compression** (`576 MB` vs `2,048 MB` `BF16` at `16K B=4`), runs **`2.13×` faster than legacy `kv_i8`** (`91.71` vs `43.05 tok/s` E2E at `16K B=4`; `110.90 tok/s` `BF16`), and matches or beats `BF16` across **`4K` (`0.9602` F1, `27/27` PK, `100%` exact seq), `8K` (`0.8667` vs `0.8644` F1, `18/18` PK), and `16K` (`0.9464 == BF16` F1, `18/18` PK, `100%` top-1 agree vs `0.8631` raw `q4`)** — so `kv_i8="auto"` promotes `"q4_quarot"` automatically at `>= 16K` (`>= 8K` on `<= 24 GB` Macs) or `B >= 16` (`kv_i8=True` / `"q8"` retains `KL <= 0.00025`). Right-padding post-LCP suffixes with exact `valid_lengths` `conv1d` window gathers + `(g=1, beta=0)` identity masking preserves exact `B=1` vs `B=4..64` parity (`4/4` parsed matches on ragged `LiquidAI` prompts).
- **Prefill-Folded `upfront_ff`, Multi-Token FF Coalescing, Active-Row `lm_head` & `A4_v2` Loop Breaker ([`engine.py`](src/fast_qwen3/engine.py)):** Folds deterministic schema prefixes into prefill, coalesces multi-token forced `llguidance` transitions (`step_w >= 2`) across wave slots, slices the `248,320`-row `lm_head` strictly to slots requiring sampling (`M_samp`), preserves `whitespace_flexible=None` (`": "` BPE separator: `0.9252` F1 vs `0.8309` minified), and applies `fast_shared.grammar.compute_loop_breaker_penalties` (`A4_v2` `p=1..8` cycles + `96`-tok 4-grams).
- **No-Think Default & Truncated-Thinking Valley Guard ([`tokenizer.py`](src/fast_qwen3/tokenizer.py)):** Defaults to `enable_thinking=False` (`<think>\n\n</think>\n\n`), which matches or beats `<think>` on structured JSON and tool calls at `3.2×` lower latency. When `enable_thinking=True` is requested for multi-step reasoning, `validate_thinking_budget` enforces a `>= 384`-token floor (`default 1024`) to prevent mid-thought `<think>` truncation collapse, or injects a bounded leading `reasoning_prefix_field` into strict JSON schemas.
- **Hot-Swappable `lora_accel` (`R2` Regime) & Double-Quantization Guard ([`kernels.py`](src/fast_qwen3/kernels.py), [`model.py`](src/fast_qwen3/model.py)):** Attaches PEFT/MLX LoRA adapters across `self_attn + mlp + linear_attn.{in_proj_qkv, in_proj_z, out_proj}` (`+17.5 pp` over `self_attn + mlp` alone while keeping `conv1d/A_log/dt_bias` frozen), and raises `QwenDoubleQuantError` if attempting destructive 2-pass `4-bit -> LoRA -> 4-bit` requantization.
- **1-Layer `Qwen3MtpDrafter`, `norm_add_one` Auto-Detection, `ByteLevel` Vocab Remap & `draft_k=3` ([`mtp_drafter.py`](src/fast_qwen3/mtp_drafter.py), [`tokenizer.py`](src/fast_qwen3/tokenizer.py)):** Loads native `mtp.*` / `optiq/mtp.safetensors` weights (auto-detecting zero-centered vs `1+weight` RMSNorm scaling via `detect_mtp_norm_add_one`, supporting `mtp_drafter=bool | Qwen3MtpDrafter | None`, defaulting `recommended_draft_k=3`, and falling back to `PromptLookupDrafter` when absent) with exact zero-GEMM recurrent + 3-tap `conv1d` rollback (`verify_draft_sequence_b1`), auto-gated to `>= 9B` native-MTP models; `QwenVocabRemapper` decodes `ByteLevel` tokens before Latin filtering (`65,536`-entry remap).

## CLI & HTTP Server

```bash
.venv/bin/fast-qwen3 generate --model 4b --prompt "Summarize silicon photonics in JSON" \
  --schema '{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}'

.venv/bin/fast-qwen3 serve --model 4b --host 127.0.0.1 --port 8080
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16 --rounds 2
```
