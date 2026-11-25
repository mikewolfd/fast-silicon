# fast-qwen3

High-throughput schema-constrained JSON extraction and tool calling on Apple Silicon with Qwen 3.5 and Qwen 3.6 (`0.8B` to `27B` dense and `35B-A3B` MoE). On 300 public JSON extraction and tool-calling tasks, schema-constrained `Qwen3.5-4B-OptiQ-4bit` returns **300/300 valid schemas** at **0.896 score** on Base M5 (**299/300**, **0.893** on M4 Pro) with zero repeat loops.

Zero telemetry, no account, and runs 100% offline after initial weight download (~3.1 GB for default 4B checkpoint). Combines Qwen's hybrid 3:1 `GatedDeltaNet` + GQA architecture with compiled Metal GDN/RoPE kernels, Apple10 NAX prefill packing (`2.34-4.2+ GB` RAM saved at zero logit drift), 4-bit rotated KV caching (`2.1x` faster 16K decode), and 256-token `PrefixTree` state reuse. [Cross-silicon benchmarks](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+, Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported. Checked with `mlx-community/Qwen3.5-4B-OptiQ-4bit` (`4b`), `0.8b-bf16`, and `9b-optiq`.

## Quickstart (~3.1 GB default 4B model)

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-qwen3 --inexact

.venv/bin/fast-qwen3 setup-weights --model-dir 4b --allow-download
```

```python
from fast_qwen3 import FastQwen3Engine

schema = {
    "type": "object",
    "properties": {"company": {"type": "string"}, "year": {"type": "integer"}},
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

## CLI & HTTP server

```bash
.venv/bin/fast-qwen3 generate --model-dir 4b "Summarize silicon photonics in JSON" \
  --schema '{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}'
.venv/bin/fast-qwen3 serve --model-dir 4b --host 127.0.0.1 --port 8080
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16 --rounds 2
```

## Benchmark highlights

Measured on 24 GB Base M5 and 48 GB M4 Pro. Against stock `mlx-lm 0.32.2` on the 50-task extraction + tool split, `fast-qwen3` enforces target JSON schemas via `llguidance` while stock decodes unconstrained (both score `25/25` `0.880` on BFCL tool calls, while schema constraints lift `LiquidAI` extraction validity from `10-12/25` to `25/25`); both raw wall time (`88.0 s` vs. `98.6 s` on M5) and valid-rate lift are reported ([JSON receipts](../../docs/benchmarks/fast-qwen3-20261122.json)).

| Model preset | Workload | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Peak memory (M5 / M4 Pro) |
| --- | --- | ---: | ---: | ---: |
| `4B-OptiQ` (`4b`, strict) | 300 JSON + tool tasks (`LiquidAI 150 + BFCL 150`) | **0.8955 (300/300 valid, 0 loops)** | 0.8930 (299/300 valid, F1 0.953) | 4.76 / 4.99 GB |
| `4B-OptiQ` (`4b`, strict) | 50-task extraction + tool subset | **0.902 score**, 50/50 valid (88.0 s) | **0.902 score**, 50/50 valid (114.1 s) | 5.26 / 5.18 GB |
| Stock `mlx-lm` (`4B-OptiQ`, unconstrained) | Same 50-task subset | 0.662 score, 37/50 valid (98.6 s) | 0.623 score, 35/50 valid (108.7 s) | 3.79 / 8.02 GB |
| `9B-OptiQ` (`mtp_drafter`, strict) | Same 50-task subset | 0.886 score, 50/50 valid (110.2 s) | 0.887 score, 50/50 valid (160.0 s) | 8.39 / 8.79 GB |
| `4B-OptiQ` (`4b`) | Shared-prefix JSON (`B=1` / `B=8` / `B=32` M4P) | 57.5 / 171.5 tok/s | 82.7 / 191.0 / **592.7 tok/s** (`B=32`) | 4.46-4.56 / 4.49-11.36 GB |
| `9B-OptiQ` (65K Latin vocab) | Speculative MTP (`B=1` / `B=8`) | 42.3 / 105.7 tok/s | 50.2 / 115.9 tok/s | 16.7 / 16.9 GB |
| `0.8B` (65K Latin vocab) | Shared-prefix JSON (`B=1` / `B=8`) | 178.3 / 658.5 tok/s (16.3 docs/s) | 240.3 / 767.5 tok/s (19.0 docs/s) | 4.81 / 4.99 GB |
| `4B-OptiQ` (`kv_i8="q4_quarot"`) | 16K context decode (`B=4`) | Matches BF16 F1 (0.946, 18/18 passkeys) | **91.7 tok/s** (**2.13x** 8-bit KV) | 576 MB KV vs 2,048 MB BF16 |

- **What keeps it fast:** `@mx.compile` GDN and RoPE kernels add `1.04-1.44x`; `GdnDecodeShared` (`int8_blk32` GDN state) cuts recurrent state RAM `3.76x` at `1.42-3.43x` decode speed; Apple10 NAX inline XOR packing skips `2.34-4.2+ GB` shadow buffers at zero logit diff (`1.18x` vs. `MLX qmm`); and `R2` GDN+Attention `BF16` LoRA adapters beat Attention+MLP alone by `+17.5 pp` (`66.7%` vs. `49.2%`). [Long-context KV & ABBA receipts](../../docs/BENCHMARKS.md#17-wave-5-dry-optimizations--dual-mac-abba-gates-fast-qwen3--fast-pplx-2026-11-24)

## Key options & defaults

| Option | Default | What it does |
| --- | --- | --- |
| `kv_i8` | `"auto"` | Keeps BF16 KV on short/medium calls and switches to rotated 4-bit KV (`"q4_quarot"`, 3.56x smaller) at `>=16K/32K` tokens on Apple10 NAX (`>=8K/16K` on Apple9 / `<=28 GB`) or high `batch*ctx` pressure (`262K/524K`). |
| `gdn_state_mode` | `"auto"` | Keeps GDN recurrent state in `fp32` below `B=8` (`<4` on `<=28 GB`) and compresses to `int8_blk32` (`3.76x` smaller) on larger batches. |
| `enable_mpp_linear` / `FAST_SILICON_QWEN3_MPP_W8A8` | `"auto"` / off | Enables Apple10 NAX `W4A8` prefill on large 4-bit layers at zero diff; opt-in `W8A8=1` adds `+4.1%` on 4B extraction (kept opt-in because 9B lost `-0.0045` on tool/JSON tasks). |
| `default_enable_thinking` | `False` | Skips `<think>` chains for deterministic extraction and tools (~3.2x faster at equal accuracy). When `True`, enforces a `>=384`-token budget so reasoning doesn't truncate. |
| `prefix_cache_mb` | `"auto"` | Reuses 256-token prompt prefixes across calls in `PrefixTree` with 2-hit admission (avoiding cold cache churn) and block-32 INT8 GDN state snapshots. |
| `mtp_drafter` / `speculative` | `None` (`"auto"`) | Auto-enables native MTP on 9B+ checkpoints (`1.28-1.92x` `B=1` speedup); keeps prompt-lookup speculation off under JSON schemas by default (pass `speculative=True` for verbatim copy-heavy extraction). |
| `rolling` (`generate_batch`) | `"auto"` (dense) | Uses dense static batching by default (`rolling=True` enables continuous rolling batching: `100%` exact parity, `-2.4 GB` MLX peak on M4 Pro `N=48, B=32`). |
| `whitespace_flexible` | `None` (`": "`) | Emits standard `": "` JSON spacing instead of minified `":"`, preserving token merges (`0.925` vs. `0.831` minified F1 on 4B extraction). |
| `drop_white_patches` (vision) | `False` | Keeps uniform white patches when `images=...` is passed because dropping white column gaps caused financial table hallucinations. |

## Limitations, Prior Art & License

- **Limitations:** `enable_thinking=False` scores `0.53-0.57` across zero-shot reasoning benchmarks (`0.76-0.80` MMLU, `0.64-0.72` MATH, pulled down by `0.16-0.24` strict formatting on `IFBench`); enable thinking (`>=384` tokens) for multi-step reasoning. Grammar-checked prompt lookup (`speculative=True`) slows short JSON extraction (`0.59-0.84x`) because deterministic syntax tokens are already skipped in parallel, and continuous `rolling=True` slowed Base M5 `B=16` (`0.81x` vs. static `B=8`). HTTP `serve` returns completed responses rather than token-by-token streams.
- **Prior art:** Built on `mlx`, `mlx-lm`, `llguidance`, and `mlx-community` Qwen 3.5/3.6 checkpoints (`OptiQ` mixed-bit). For higher batch throughput on short 2K JSON schemas, compare [`fast-gemma4`](../fast-gemma4).
- **License:** Package code is [MIT licensed](LICENSE); Qwen weights carry the Apache-2.0 license. Verified on 48 GB M4 Pro and 24 GB Base M5.
