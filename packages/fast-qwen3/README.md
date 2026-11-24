# fast-qwen3

Extract schema-constrained JSON or call tools locally on Apple Silicon with Qwen 3.5 and Qwen 3.6 models (`0.8B` to `27B` dense and `35B-A3B` MoE). On 50 public extraction and tool tasks, schema-constrained `Qwen3.5-4B-OptiQ-4bit` returned **50/50 valid responses and scored 0.902** on both the 24 GB M5 and the 48 GB M4 Pro (`1.53×` / `1.38×` EOT vs. unconstrained stock `mlx-lm`). On full 300-item Category 4 (`LiquidAI 150 + BFCL v3 150`), `fast-qwen3` hits **`300/300` (`0.8955` `S_obj,e2e`, `0.881` EOT, `0` loops, `PASS`)** on quiescent Base M5 and **`299/300` (`0.8930`, `F1=0.9526`, `BFCL=0.8333`)** on contended M4 Pro. [Outcome tables and methodology](../../docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22) · [Wave 3 Category 3/4 results](../../docs/BENCHMARKS.md#162-post-fix-fast-qwen3-fast-ocr-fast-embeddinggemma-2-vision--fast-pplx-results)

The engine runs Qwen’s hybrid 3:1 `GatedDeltaNet` + GQA architecture on MLX with mixed-bit `OptiQ` weight fusion, Apple10+ NAX `W4A8` prefill acceleration, compiled GDN and RoPE kernels, and recurrent prefix caching (`PrefixTree`). Thinking mode defaults off (`<think>\n\n</think>\n\n`), which matches or beats thinking on structured JSON and tool calls at roughly one-third the latency.

Optional subsystems cover large-batch GDN recurrent-state compression (`int8_blk32`), rotated 4-bit KV caches (`QuaRot`) for long contexts, speculative multi-token prediction (`MTP`) on 9B+ checkpoints, lazy Qwen3.5 vision (`generate(..., images=...)`), and hot-swappable LoRA adapters. [Vision, OCR & NAX prefill validation](../../docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23)

## Quick start

From the `fast-silicon` workspace root, install the package and stage weights (`mlx-community/Qwen3.5-4B-OptiQ-4bit` by default):

```bash
uv sync --python 3.13 --package fast-qwen3 --inexact
.venv/bin/fast-qwen3 setup-weights --model-dir 4b --allow-download
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
.venv/bin/fast-qwen3 generate --model-dir 4b "Summarize silicon photonics in JSON" \
  --schema '{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false}'

# OpenAI-compatible HTTP server
.venv/bin/fast-qwen3 serve --model-dir 4b --host 127.0.0.1 --port 8080

# Interleaved ABBA speed and footprint benchmark
.venv/bin/python packages/fast-qwen3/scripts/benchmark.py --presets 4b,0.8b-bf16 --rounds 2
```

## Benchmark highlights

Measurements compare the 24 GB base M5 and the 48 GB M4 Pro under lock-serialized ABBA runs. On the 50-task public outcome test (`LiquidAI` extraction + `BFCL v3` tool calls), `fast-qwen3` enforces the target JSON schema via `llguidance`, whereas stock `mlx-lm 0.32.2` decodes unconstrained and scores `0` when an output is invalid JSON. [JSON receipts](../../docs/benchmarks/fast-qwen3-20261122.json) · [Long-context KV receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

| Model preset | Workload | Base M5 (24 GB) | M4 Pro (48 GB) | Memory (M5 / M4 Pro) |
| :--- | :--- | ---: | ---: | ---: |
| `4B-OptiQ` (`4b`, constrained, Wave 3) | Full 300 Cat 4 (`LiquidAI 150 + BFCL 150`) | **0.8955 (`300/300`, `0.881` EOT, `0` loops, `PASS`)** | 0.8930 (`299/300`, `0.676` EOT, `2` loops, contended) | 4.76 / 4.99 GB |
| `4B-OptiQ` (`4b`, constrained) | 50 public extraction + tool tasks | 0.902 score, 50/50 valid (0.513 score/s) | 0.902 score, 50/50 valid (0.396 score/s) | 5.26 / 5.18 GB |
| Stock `mlx-lm` (`4B-OptiQ`, unconstrained) | Same 50 public tasks | 0.662 score, 37/50 valid (0.336 score/s) | 0.623 score, 35/50 valid (0.286 score/s) | 3.79 / 8.02 GB |
| `9B-OptiQ` (`mtp_drafter`, constrained) | Same 50 public tasks | 0.886 score, 50/50 valid (0.402 score/s) | 0.887 score, 50/50 valid (0.277 score/s) | 8.39 / 8.79 GB |
| `4B-OptiQ` (`4b`) | Shared-prefix strict JSON, batch 1 / batch 8 | 57.5 / 171.5 tok/s | 82.7 / 191.0 tok/s | 4.46 / 4.49 GB |
| `9B-OptiQ` (`9b-optiq`, 248K vocab) | Speculative MTP batch 1 / batch 8 | 32.4 / 101.2 tok/s | 41.4 / 112.9 tok/s | 8.29 / 8.31 GB (16.4 GB peak) |
| `9B-OptiQ` (`9b-optiq`, 65K Latin vocab) | Speculative MTP batch 1 / batch 8 | 42.3 / 105.7 tok/s | 50.2 / 115.9 tok/s | 16.7 / 16.9 GB peak |
| `0.8B` (`0.8b-bf16`, 65K Latin vocab) | Shared-prefix strict JSON, batch 1 / batch 8 | 178.3 / 658.5 tok/s | 240.3 / 767.5 tok/s | 4.81 / 4.99 GB peak |
| `4B-OptiQ` (`kv_i8="q4_quarot"`) | 16K context decode, batch 4 | Matches BF16 F1 (0.946, 18/18 passkeys) | 91.7 tok/s (2.13× 8-bit KV) | 576 MB KV vs 2,048 MB BF16 |
| `OvisOCR2` `GdnDecodeShared` (`int8_blk32` / `bf16`) | 18-layer GDN decode (`B=1` / `16` / `64`) | *(M4 Pro Metal verified)* | **1.422×** (`0.719ms`) / **2.508×** (`2.119ms`) / **3.432×** (`9.252ms bf16`) | `3.76×` GDN state cut (`y_cos=0.999808` @ 256AR) |
| `9B-OptiQ` (`Apple10` NAX `W4A8` / opt-in `W8A8`) | 50 public extraction + tool tasks (`B=8`) | 0.886 / 0.884 score, 50/50 valid (`+4.6%` / `+8.3%` EOT) | Native `qmm` (`Apple9`) | 12.57 / 14.59 GB peak (`10.07 GB` base) |
| `4B` / `9B-OptiQ` (`L=512` prefill + 16 tok) | Single-pass prefill (`W4A8` default / `W8A8` opt-in) | **1.06–1.14×** (`4B`), **1.10–1.18×** (`9B`) | Native `qmm` (`Apple9`) | 5.82–6.98 GB (`4B`), 10.72–12.78 GB (`9B`) |

- **Reasoning and instruction following.** On zero-shot reasoning benchmarks (`IFBench`, `MMLU-Redux`, `MATH-500`) with `enable_thinking=False`, weak `IFBench` formatting adherence (0.16–0.24) held overall Category-3 averages to 0.533–0.573 despite 0.76–0.80 MMLU and 0.64–0.72 MATH scores. Enable thinking with a budget of at least 384 tokens for multi-step reasoning tasks.
- **Compiled Metal kernels.** Bit-exact `@mx.compile` GDN and RoPE kernels deliver `1.053–1.303×` (M4 Pro) and `1.038–1.440×` (Base M5) speedups.
- **Optional lazy vision.** Passing `images=...` to `generate` or `generate_batch` loads the shared Qwen3.5 ViT on demand with zero extra RAM on text-only calls. On `CORD-v2` with `Qwen3.5-0.8B` BF16, `generate_batch(..., compact_active=True)` runs `1.11×` faster at `B=8` and maintains `1.0` schema validity at `B=16`.

## Key options & defaults

| Option | Default | What it does |
| :--- | :--- | :--- |
| `kv_i8` | `"auto"` | Uses BF16 KV on short/medium requests and switches to rotated 4-bit KV (`"q4_quarot"`, 3.56× smaller) at `>= 16K / 32K` tokens on Apple10 NAX (`>= 8K / 16K` on Apple9 / `<= 28 GB` hosts) or high `batch*ctx` pressure (`>= 262_144` on `<= 28 GB`, `524_288` above). Pass `"q8"` for 8-bit KV or `False`/`"bf16"` for BF16 everywhere. |
| `gdn_state_mode` (cache / `FAST_QWEN3_GDN_STATE_MODE`) | `"auto"` | Stores recurrent GDN state in `fp32` below `B=8` (`< 4` on `<= 28 GB` hosts) to avoid kernel launch overhead, and switches to `int8_blk32` (`3.76×` DRAM cut with `FP32` decay `g`) at larger batches. |
| `enable_mpp_linear` / `mpp_w8a8` (model / `FAST_SILICON_QWEN3_MPP_W8A8`) | `"auto"` / `False` | Uses Apple10+ NAX `64×64` `W4A8` prefill on large 4-bit projections (`M >= 128`, `k, n >= 1536`) with inline XOR unpacking, saving `2.341 GB` on `4B-OptiQ` and `> 4.2 GB` on `9B-OptiQ` at zero diff (`1.183×` vs. `MLX qmm`). Set `FAST_SILICON_QWEN3_MPP_W8A8=1` for 8-bit NAX prefill (`4B-OptiQ` `+4.1%` EOT; kept opt-in because `9B-OptiQ` lost `-0.0045` on Category 4). |
| `drop_white_patches` (multimodal batch) | `False` | Keeps all bright background patches by default because dropping white column gaps caused hallucinations on compact financial tables; pass `True` only to opt in. (`max_pad_ratio` wave splitting also stays inactive up to `1,024` tokens, giving `1.373×` on `FinTabNet-24`.) |
| `default_enable_thinking` / `enable_thinking` | `False` | Skips `<think>` chains for fast, deterministic extraction and tool calling. When `True`, enforces at least a `384`-token budget (`1,024` default) so reasoning does not truncate mid-thought. |
| `mtp_drafter` | `None` | Automatically loads the bundled 1-layer multi-token drafter when checkpoint weights include MTP layers (such as `9B+`, `1.28–1.92×` faster single-stream decode) and leaves smaller checkpoints without MTP weights unchanged. Pass `False` to disable. |
| `prefix_cache_mb` | `"auto"` | Shares 256-token prompt prefixes across calls in `PrefixTree` with 2-hit admission (`admission_min_hits=2`, so 1st-sighting misses never hold cold snapshots or trip the `"auto"` circuit breaker), compressing recurrent `GatedDeltaNet` states 3.76× in block-32 INT8 without breaking ragged batch parity. |
| `speculative` | `None` (`"auto"`) | Keeps prompt-lookup speculation off by default under `json_schema` (`matcher is None` guard, since `upfront_ff` already jumps schema keys and short JSON slowed `0.59–0.84×`). Pass `speculative=True` to opt into cursor-tracked `PromptLookupDrafter` + `LLMatcher.deep_copy()` verification (`100%` token-for-token parity). |
| `rolling` (`generate_batch`) | `"auto"` (dense) | Uses dense static batching by default (`_ENABLE_ROLLING_AUTO=False`). Pass `rolling=True` to opt into continuous `RollingHybridCache` batching (`100%` parity; M4 Pro `4B-OptiQ` `N=48, B=32` runs **`1.021×` faster** at **`9.8 GB` vs `12.2 GB` MLX peak** [`-2.4 GB`]; `B=32` profile hits `592.7 tok/s` / `1.751 docs/s` in `11.36 GB` peak phys; Base M5 `B=16` ran `0.809×` vs static `B=8`). |
| `lora_adapter` (`load_lora_adapter`) | `None` | Attaches PEFT or MLX LoRA adapters across attention, MLP, and safe `GatedDeltaNet` linear projections while guarding against destructive double quantization (`4-bit` to `LoRA` to `4-bit`). |

Default hygiene also enables early catastrophic-loop abort (`detect_catastrophic_token_loop`) and `mx.async_eval` on pure-FF steps; see the [Wave-5 dual-Mac ABBA gates](../../docs/BENCHMARKS.md#17-wave-5-dry-optimizations--dual-mac-abba-gates-fast-qwen3--fast-pplx-2026-11-24).

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** serialized under `/tmp/m4_gpu_validation.lock`. On 24 GB Macs, concurrent `4B-OptiQ` GPU generation alongside `FastBgeSmall` on the Neural Engine stayed within a 9.39 GB peak footprint.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](../../LICENSE).
