# fast-qwen3

Extract schema-constrained JSON or call tools locally on Apple Silicon with Qwen 3.5 and Qwen 3.6 models (`0.8B` to `27B` dense and `35B-A3B` MoE). On 50 public extraction and tool tasks, schema-constrained `Qwen3.5-4B-OptiQ-4bit` returned **50/50 valid responses and scored 0.902** on both the 24 GB M5 and the 48 GB M4 Pro. Unconstrained stock `mlx-lm` on the same checkpoint returned 37/50 valid responses on M5 (0.662 score) and 35/50 on M4 Pro (0.623 score), giving `fast-qwen3` 1.53× and 1.38× the score per second. [Outcome tables and methodology](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)

The engine runs Qwen’s hybrid 3:1 `GatedDeltaNet` + GQA architecture on MLX with mixed-bit `OptiQ` weight fusion, recurrent prefix caching, speculative multi-token prediction (`MTP`) on 9B+ models, rotated 4-bit KV compression (`QuaRot`) for long contexts, and hot-swappable LoRA adapters. Thinking mode defaults off (`<think>\n\n</think>\n\n`), which matches or beats thinking on structured JSON and tool calls at roughly one-third the latency.

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

## CLI & HTTP server

```bash
# Schema-constrained CLI generation
.venv/bin/fast-qwen3 generate --model 4b --prompt "Summarize silicon photonics in JSON" \
  --schema '{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}'

# OpenAI-compatible HTTP server
.venv/bin/fast-qwen3 serve --model 4b --host 127.0.0.1 --port 8080

# Interleaved ABBA speed and footprint benchmark
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16 --rounds 2
```

## Benchmark highlights

Measurements compare the 24 GB base M5 and the 48 GB M4 Pro under lock-serialized ABBA runs. On the 50-task public outcome test (`LiquidAI` extraction + `BFCL v3` tool calls), `fast-qwen3` enforces the target JSON schema via `llguidance`, whereas stock `mlx-lm 0.32.2` decodes unconstrained and scores `0` when an output is invalid JSON. [JSON receipts](../../docs/benchmarks/fast-qwen3-20261122.json) · [Long-context KV receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

| Model preset | Workload | Base M5 (24 GB) | M4 Pro (48 GB) | Memory (M5 / M4 Pro) |
| :--- | :--- | ---: | ---: | ---: |
| `4B-OptiQ` (`4b`, constrained) | 50 public extraction + tool tasks | 0.902 score, 50/50 valid (0.513 score/s) | 0.902 score, 50/50 valid (0.396 score/s) | 5.26 / 5.18 GB |
| Stock `mlx-lm` (`4B-OptiQ`, unconstrained) | Same 50 public tasks | 0.662 score, 37/50 valid (0.336 score/s) | 0.623 score, 35/50 valid (0.286 score/s) | 3.79 / 8.02 GB |
| `9B-OptiQ` (`--mtp-drafter`, constrained) | Same 50 public tasks | 0.886 score, 50/50 valid (0.402 score/s) | 0.887 score, 50/50 valid (0.277 score/s) | 8.39 / 8.79 GB |
| `4B-OptiQ` (`4b`) | Shared-prefix strict JSON, batch 1 / batch 8 | 57.5 / 171.5 tok/s | 82.7 / 191.0 tok/s | 4.46 / 4.49 GB |
| `9B-OptiQ` (`9b-optiq`, 248K vocab) | Speculative MTP batch 1 / batch 8 | 32.4 / 101.2 tok/s | 41.4 / 112.9 tok/s | 8.29 / 8.31 GB (16.4 GB peak) |
| `9B-OptiQ` (`9b-optiq`, 65K Latin vocab) | Speculative MTP batch 1 / batch 8 | 42.3 / 105.7 tok/s | 50.2 / 115.9 tok/s | 16.7 / 16.9 GB peak |
| `0.8B` (`0.8b-bf16`, 65K Latin vocab) | Shared-prefix strict JSON, batch 1 / batch 8 | 178.3 / 658.5 tok/s | 240.3 / 767.5 tok/s | 4.81 / 4.99 GB peak |
| `4B-OptiQ` (`kv_i8="q4_quarot"`) | 16K context decode, batch 4 | Matches BF16 F1 (0.946, 18/18 passkeys) | 91.7 tok/s (2.13× 8-bit KV) | 576 MB KV vs 2,048 MB BF16 |

On zero-shot reasoning and instruction benchmarks (`IFBench`, `MMLU-Redux`, `MATH-500`) with `enable_thinking=False`, weak `IFBench` formatting adherence (0.16–0.24) held overall Category-3 averages to 0.533–0.573 despite 0.76–0.80 MMLU and 0.64–0.72 MATH scores. Enable thinking with a token budget of at least 384 tokens for multi-step reasoning tasks.

## Key options & defaults

| Option | Default | What it does |
| :--- | :--- | :--- |
| `kv_i8` | `"auto"` | Uses BF16 KV on short requests and automatically switches to rotated 4-bit KV (`"q4_quarot"`, 3.56× smaller) at `>= 16K` tokens (`>= 8K` on 24 GB Macs) or batch `>= 16`. Pass `"q8"` for 8-bit KV or `False` for BF16 everywhere. |
| `enable_thinking` | `False` | Skips `<think>` chains for fast, deterministic extraction and tool calling. When `True`, enforces a `>= 384`-token budget (1,024 default) so thoughts are not cut off mid-sentence. |
| `mtp_drafter` | `"auto"` (`None` `< 9B`) | Loads the native 1-layer multi-token drafter (`draft_k=3`) on 9B+ checkpoints where speculative decoding speeds up single-stream generation (1.28–1.92×); stays off on smaller models where the 248K output head outweighs draft savings. |
| `prefix_cache_mb` | `"auto"` | Shares 256-token prompt prefixes across calls in `PrefixTree`, compressing recurrent `GatedDeltaNet` states 3.76× in block-32 INT8 without breaking ragged batch parity. |
| `lora_path` | `None` | Attaches PEFT/MLX LoRA adapters across attention, MLP, and safe `GatedDeltaNet` linear projections while guarding against destructive double quantization (`4-bit -> LoRA -> 4-bit`). |

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** serialized under `/tmp/m4_gpu_validation.lock`. On 24 GB Macs, concurrent `4B-OptiQ` GPU generation alongside `FastBgeSmall` on the Neural Engine stayed within a 9.39 GB peak footprint.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](../../LICENSE).
