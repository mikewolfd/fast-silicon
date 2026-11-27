# fast-qwen3

High-throughput schema-constrained JSON extraction and tool calling on Apple Silicon with Qwen 3.5 and Qwen 3.6 (`0.8B` to `27B` dense and `35B-A3B` MoE). On 300 public JSON extraction and tool-calling tasks, schema-constrained `Qwen3.5-4B-OptiQ-4bit` returns **300/300 valid schemas** at **0.896 score** on Base M5 (**299/300**, **0.893** on M4 Pro) with zero repeat loops.

Zero telemetry, no account, and runs 100% offline after initial weight download (~3.3 GB for default 4B checkpoint). Combines Qwen's hybrid 3:1 `GatedDeltaNet` + GQA architecture with compiled Metal GDN/RoPE kernels, M5 (`Apple10 NAX`) prefill packing (`2.34-4.2+ GB` RAM saved at zero logit drift), 4-bit rotated KV caching (`2.1x` faster 16K decode), and 256-token `PrefixTree` state reuse. [Cross-silicon benchmarks](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+ (GPU-only MLX; `macOS 15+` for the full monorepo ANE stack; verified on macOS 26.6 and 27.2.1), Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported. Checked with `mlx-community/Qwen3.5-4B-OptiQ-4bit` (`4b`), `0.8b-bf16`, and `9b-optiq`.

## Quickstart (~3.3 GB default 4B model)

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
| `4B-OptiQ` (`K=4`, `T=0.6, p=0.95`) | `MATH-500(50)` (`2048` tok, [receipts](../../docs/benchmarks/m5-sprint2-20261126.json)) | **0.86** static (`1,570.7 s`, **2.49x** GPU top-k) / **0.82** rolling (`966.9 s`, **4.05x**) vs `0.66` greedy (`1536`) / `0.76` (`2560`) | - | 7.10 / 10.37 GB (M5) |

- **What keeps it fast:** Zero-copy slot-indexed `GdnDecodeShared` (`MS1`: `5.25x-5.89x` on `Qwen4B_24L`, with Apple10 NAX native `b_act=1, t_len > 1` `GDN` crossover), ragged Split-K Flash-Decode SDPA (`MS2`: `5.79x-19.77x` on `Qwen4B_8L`, `s_q <= 8` gate + `refill_min=1` yielding **`1.20x` Cat-4 / `1.88x` Skew4096** on `9B-OptiQ` and **`1.44x` Skew4096** on `4B-OptiQ`), and fused `Conv1d(k=4)+SiLU+L2 QKNorm+Gates` (`MS3a`: `2.79x-4.97x`) stack for **`1.55x-2.12x` uniform (`503.5 tok/s` @ `B=64`) and `2.30x-3.88x` skewed (`488.7 tok/s` @ `B=64`)** rolling decode on `Qwen3.5-4B-OptiQ-4bit`; `int8_blk32` cuts recurrent GDN state RAM `3.76x`; M5 (`Apple10 NAX`) inline XOR packing skips `2.34-4.2+ GB` shadow buffers; `GDN+Attention BF16 LoRA` + selective `4-bit -> 8-bit` offline merge (`promote_4bit_to_8bit=True`) hits `0.99975` logit cosine and `+6.0%` decode over runtime adapters; shared GPU top-k nucleus sampling (`gather_topk_stats_mlx` + `try_sample_topk_row`) cuts `K=4` sampled decode `2.49x` at exact parity alongside `generate_self_consistent` (auto-enabling `rolling=True` at `>= 4` samples with `adaptive=False` default); and single-pass Metal `gpu_argmax_nan`/`gpu_argmax_conf` eliminates full-vocab CPU copies. [Long-context KV & regression gate receipts](../../docs/BENCHMARKS.md#17-wave-5-dry-optimizations--dual-mac-abba-gates-fast-qwen3--fast-pplx-2026-11-24)

## Key options & defaults

| Option | Default | What it does |
| --- | --- | --- |
| `kv_i8` | `"auto"` | Keeps BF16 KV on short/medium calls and switches to rotated 4-bit KV (`"q4_quarot"`, 3.56x smaller) at `>=16K/32K` tokens on M5 (`Apple10 NAX`, `>=8K/16K` on M4 `Apple9` / `<=28 GB`) or high `batch*ctx` pressure (`262K/524K`). |
| `gdn_state_mode` | `"auto"` | Keeps GDN recurrent state in `fp32` below `B=8` (`<4` on `<=28 GB`) and compresses to `int8_blk32` (`3.76x` smaller) on larger batches. |
| `enable_mpp_linear` / `FAST_SILICON_QWEN3_MPP_W8A8` | `"auto"` / off | Enables M5 (`Apple10 NAX`) `W4A8` prefill on large 4-bit layers at zero diff; opt-in `W8A8=1` adds `+4.1%` on 4B extraction (kept opt-in because 9B lost `-0.0045` on tool/JSON tasks). |
| `default_enable_thinking` | `False` | Skips `<think>` chains for deterministic extraction and tools (~3.2x faster at equal accuracy). When `True`, enforces a `>=384`-token budget so reasoning doesn't truncate. |
| `prefix_cache_mb` | `"auto"` | Reuses 256-token prompt prefixes across calls in `PrefixTree` with 2-hit admission (avoiding cold cache churn) and block-32 INT8 GDN state snapshots. |
| `mtp_drafter` / `speculative` | `None` (`"auto"`) | Auto-enables native MTP on 9B+ checkpoints (`1.28-1.92x` `B=1` speedup); keeps prompt-lookup speculation off under JSON schemas by default (pass `speculative=True` for verbatim copy-heavy extraction). |
| `rolling` (`generate_batch`) | `"auto"` (dense) | Uses dense static batching by default (`rolling=True` enables continuous rolling batching: `100%` exact parity, `-2.4 GB` MLX peak on M4 Pro `N=48, B=32`). |
| `whitespace_flexible` | `None` (`": "`) | Emits standard `": "` JSON spacing instead of minified `":"`, preserving token merges (`0.925` vs. `0.831` minified F1 on 4B extraction). |
| `drop_white_patches` (vision) | `False` | Keeps uniform white patches when `images=...` is passed because dropping white column gaps caused financial table hallucinations. |

## Limitations, Prior Art & License

- **Limitations:** `enable_thinking=False` scores `0.53-0.57` across zero-shot reasoning benchmarks (`0.76-0.80` MMLU, `0.64-0.72` MATH, pulled down by `0.16-0.24` strict formatting on `IFBench`); enable thinking (`>=384` tokens) for multi-step reasoning. On Base M5, `generate_self_consistent(k=4)` raises `4B-OptiQ` `MATH-500(50)` to `0.86` static / `0.82` rolling (`0.66` greedy), while `mmlu285` saw no gain (`0.7825` vs `0.7825`) and opt-in `adaptive=True` dropped `MATH-500(50)` `-6 pp` to `0.80`. Grammar-checked prompt lookup (`speculative=True`) slows short JSON extraction (`0.59-0.84x`) because deterministic syntax tokens are already skipped in parallel, and continuous `rolling=True` slowed Base M5 `B=16` (`0.81x` vs. static `B=8`). HTTP `serve` returns completed responses rather than token-by-token streams.
- **Prior art:** Built on `mlx`, `mlx-lm`, `llguidance`, and `mlx-community` Qwen 3.5/3.6 checkpoints (`OptiQ` mixed-bit). For higher batch throughput on short 2K JSON schemas, compare [`fast-gemma4`](../fast-gemma4).
- **License:** Package code is [MIT licensed](LICENSE); Qwen weights carry the Apache-2.0 license. Verified on 48 GB M4 Pro and 24 GB Base M5.
