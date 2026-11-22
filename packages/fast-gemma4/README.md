# fast-gemma4

Extract fields into schema-constrained JSON or call tools locally with Gemma 4 E2B. On the 24 GB M5, the strict default returned **150/150 valid extractions at 0.946 field F1** and made **123/150 correct tool decisions** in the normal benchmark. Stock `mlx-vlm` scored 0.923 on the same documents, with 146 valid extractions and 120 correct tool decisions. Both loaded the same public instruction-tuned 4-bit checkpoint. [Extraction, tool calls and timing](../../docs/BENCHMARKS.md#structured-generation)

The engine uses the vocabulary shipped with the checkpoint. The public model has 262,144 output tokens. The default grammar enforces the schema as tokens are generated and requires exactly one space after each key’s colon. Use `--compact-json` or `whitespace_flexible=False` to generate the old minified layout. Use `--whitespace-flexible` or `whitespace_flexible=True` to allow optional whitespace. The English-oriented `--vocab-out 150786` profile reduced non-English field accuracy. A checkpoint exported with a smaller output head uses its stored mapping. [Vocabulary and grammar experiments](../../docs/benchmarks/gemma-grammar-20261121.json)

> **Full Benchmarks & Methodology:** For complete cross-silicon tables (`M4 Pro 48 GB` vs `Base M5 24 GB`), all 16 deterministic public outcome datasets, 4-bucket `RUSAGE_INFO_V6` footprints, and reproducible CLI commands, see the monorepo [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## Quick start

From the `fast-silicon` workspace root, install the package and prepare the model. This example runs on the GPU.

```bash
uv sync --python 3.13 --package fast-gemma4 --inexact
.venv/bin/gemma4 setup-weights --skip-early14
```

Run this example with `.venv/bin/python`. Check whether the output matches the schema before using its fields.

```python
from contextlib import closing
from fast_gemma4 import FastGemma4Engine

with closing(FastGemma4Engine(default_mode="gpu", mtp_drafter=None)) as engine:
    item = engine.generate(
        "Extract the named entity and topic from this text: Ocean Lab studies coral.",
        json_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
            "required": ["entity", "topic"],
            "additionalProperties": False,
        },
        max_tokens=128,
    )
    if item.valid_schema:
        print(item.parsed)
    else:
        print(f"Schema check failed (finish reason {item.finish_reason})")
```

On the M4 Pro this printed `{'entity': 'Ocean Lab', 'topic': 'coral'}`.

For byte-compact output on disk or the wire, serialize the validated value with `json.dumps(item.parsed, ensure_ascii=False, separators=(",", ":"))`. This preserves the field contents. When rescoring a Parquet extraction with new settings, write to a new output directory. Resume skips earlier successful rows by ID.

## Cross-Silicon Benchmark Highlights (`M4 Pro 48 GB` vs `Base M5 24 GB`)

The historical throughput table below used 150,786 output tokens and minified JSON. The capacity runs repeated eight extraction fixtures. The current strict extraction results and historical stock implementation share the same public `lmstudio-community/gemma-4-E2B-it-MLX-4bit` checkpoint. [Quality, configurations and timers](../../docs/BENCHMARKS.md#structured-generation)

### 1. Outcome (`EOT`) & Schema-Valid Batch Throughput

| Benchmark / Workload | Base M5 (`24 GB`, `10c Apple10 NAX`) | M4 Pro (`48 GB`, `20c Apple9`) | vs Stock `mlx-vlm` (`M5` / `M4 Pro`) | M5 vs M4 Pro |
|---|---|---|---|---|
| **Cat 4: JSON & Tool AST** (`LiquidAI/structured-extraction + BFCL v3`, `N=300`, `llg_on`) | **`2.717 EOT`** (`299/300` valid, `S_obj=0.7487`; `llg_off`: `1.848 EOT`, `S_obj=0.8241`) | **`2.412 EOT`** (`299/300` valid, `S_obj=0.7473`; `llg_off`: `1.702 EOT`, `S_obj=0.8178`) | **`1.81x`** / **`1.57x`** `EOT` *(vs stock `1.501` / `1.537 EOT`, `286–287/300` valid)* | **`1.13x`** |
| **Cat 3: Reasoning & IF** (`IFBench + MMLU-Redux-2.0 + MATH-500`, `N=435`, `gpu_fp16_kvi8`) | **`0.843 EOT`** (`181.5 s`, `S_obj=0.3517`) | `0.892 EOT` (`174.8 s`, `S_obj=0.3586`) | **`1.03x`** / `0.94x` *(vs stock `0.818` / `0.951 EOT`, `S_obj=0.3655`)* | `0.95x` |
| Repeated extraction fixtures, `kv_i8` | GPU 10.46 valid documents/s, batch 128 | GPU 6.79 and 6.87 valid documents/s. ANE+GPU 8.78 and 5.91 at 2 × 128. | Stock used batch 64. The M4 Pro ANE+GPU rates were 4.90× and 3.29× their respective stock runs. | M5 GPU / later M4 Pro GPU 1.52× |

The M5 public-task rate was 3.62 valid responses/s, and one repeat loop failed the project run’s local check. Reported combined memory was 16.94 GB for the project and 5.58 GB for stock. The fixture rows use different batch sizes and track model allocator memory separately. [Run records](../../docs/benchmarks/document-workloads.json)

- **Memory discipline:** `fast-gemma4` holds `B_slots=128` inside **`6.13–6.88 GB` peak active MLX RAM** (`kv_i8` saves `-0.67 GB` to `-1.21 GB` vs `BF16` KV), whereas unchunked `B=64` in stock `mlx_vlm` hits **`34.4 GB` OOM** (`15.24 GB` even with `chunk=256`).
- **End-to-end Parquet pipeline (M4 Pro `48 GB`, Tuned LoRA, `p1-2`, `max_chars=8000`):** **`2.33 docs/s`** (`255/255` valid, `1,051.8 dec tok/s`, `5,443.2 total tok/s`, `6.88 GB` peak).

### 2. Speculative (`MTP`), Prefix-Tree & Local Post-Training (`QLoRA` / `GRPO`)

| Capability Suite | Base M5 (`24 GB`, `Apple10 NAX`) | M4 Pro (`48 GB`, `Apple9`) | Key Takeaway |
|---|---|---|---|
| **`MTP` draft assistant** | One constrained JSON fixture rose from 32.8 to 97.1 decode tokens/s. Its output changed. | The same fixture rose from 46.8 to 121.4 decode tokens/s with identical output tokens. | Across the fastest tested free-text and JSON settings at batches one and four, exact matches were 0/4 on M5 and 1/4 on M4 Pro. The automatic policy uses MTP at batch one on M5 and batches one to four on M4 Pro. [Results by host](../../docs/benchmarks/capability-corrections-20261121.json) |
| **Shared-Prefix & Prefix-Tree KV Reuse** | Zero system-KV cloning across batch slots | Zero system-KV cloning across batch slots | `select_speculative_policy` enables prefix-tree KV verification (`shared_prefix_tokens >= 32`, `num_candidates >= 2`; up to **`2.59x`** at `N=4..8` shared-prefix candidates; see [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md)). |
| **Local 8-bit `QLoRA` and shared-prefix reuse** | Earlier 2,048- and 1,280-token attempts exhausted memory on 24 GB; validating `pack_tokens=1024, grad_accum=2, ckpt_tokens=512` (`19.2 GB` cap) completed with zero OOMs (`4.848 GiB` unpacked at `87.67` sup tok/s, `4.849 GiB` packed at `80.52` sup tok/s, `6.759 GiB` shared-KV at `81.63` sup tok/s; `loss_diff=0.0, grad_cos=1.0`). | On the same 12 examples, unpacked BF16 reached 579.1 tokens/s and packed 8-bit reached 545.8. A separate eight-example run with repeated instructions reached 1,385.4 logical tokens/s with prefix reuse. | That run credits each document with the reused prefix. Measuring the cache’s contribution requires identical examples and supervised answer counts. [Fixture and gradient details](../../docs/benchmarks/capability-corrections-20261121.json) · [v2.2 validation](../../docs/benchmarks/v22-validation-20261121.json) |
| **Disaggregated `G=4` `GRPO` + Concurrent `Laya` ANE Critic** | **`2.61 rollouts/s`** (`T=0.7`) / **`3.38 rollouts/s`** (`T=0.0` BoN, `8.68 GB`) = **`5.22x` / `6.77x`** vs stock `0.50` (`99.96%` ANE critic hidden) | **`1.81 rollouts/s`** (`T=0.7`) / **`3.15 rollouts/s`** (`T=0.0`, `7.11 GB`, `-63%` RAM) = **`8.93x` / `15.51x`** rollouts (**7.44× / 11.30×** estimated full-iteration speedups from the measured stages) | `10c` Base M5 generates constrained `GRPO` rollouts **`1.44x` faster** than `20c` M4 Pro with `0 GB` GPU critic VRAM. |

## Dynamic Runtime Modes (Default `"auto"`)

`FastGemma4Engine()` integrates `fast_shared.runtime_lessons.DynamicSiliconPolicy` by default so zero-argument callers automatically receive hardware-, RAM-, and batch-aware dispatch without hand-tuning per Mac:

| Parameter | Default | `"auto"` Behavior & Overrides | Source |
|---|---|---|---|
| `mtp_drafter` (`FastGemma4Engine`) | **`"auto"`** | **Lazy chipset-aware `MTP` loading:** pays **`0 MB` RAM and `0 ms` init** on `B>=8` batch ETL, stochastic (`T > 0`), ANE waves, or when no local drafter is staged. On the first qualifying greedy GPU wave, loads `google/gemma-4-E2B-it-assistant` (`~150 MiB`) once — strictly for **`B == 1`** on `Apple10+ NAX` (M5, where NAX TensorOps accelerate `B >= 2` directly by `1.5x–1.9x`) and **`1 <= B <= 4`** (`K=3/4` at `B=1`, `K=2` at `2 <= B <= 4`) on `Apple9` (M4). Pass `mtp_drafter=None` or `False` to disable, or `True`/a path to load eagerly. | [`engine.py`](src/fast_gemma4/engine.py), [`mtp_drafter.py`](src/fast_gemma4/mtp_drafter.py) |
| `ane_ring_variant` (`FastGemma4Engine`) | **`"auto"`** | **RAM- & certification-aware ANE ring variant:** probes host RAM via `probe_host_hardware()` + `DynamicSiliconPolicy`; on `<= 24 GB` Macs selects **`"w8"`** (`504 MB`, `7,814.8 tok/s` ANE chunk prefill) **only when** a staged resident `w8` bundle exists and passes [`is_w8_bundle_calibrated`](src/fast_gemma4/ane_pipeline.py#L151-L175), falling back cleanly to **`"fp16"`** (`1.0 GB`, `5,799.2 tok/s`) if uncalibrated or missing. On `> 24 GB` Macs, stays on `"fp16"` unless `"w8"` is explicitly passed. | [`engine.py`](src/fast_gemma4/engine.py), [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py) |
| `use_ane_ring` (`generate_batch`) | **`"auto"`** | **Chip- & batch-aware ANE prefill ring:** automatically enables the async ANE prefill ring **only when** all gates pass: non-`Apple10 NAX` (`Apple9` M4, with 8.78 and 5.91 valid documents/s in two M4 Pro runs), `>= 32 GB` RAM (or `allow_ane_on_low_ram=True`), `kv_i8=True`, `mode != "gpu"`, `len(prompts) >= max(16, ane_cohort_size)` (`max_suf >= 64`), and a valid ANE-resident ring bundle is staged. Automatically stays pure GPU on `Apple10+ NAX` (where pure GPU hits `10.46 valid docs/s`), low-RAM hosts, `kv_i8=False`, or small batches without raising. Explicit `False` always disables; explicit `True` / `"require"` always requires the ring. | [`engine.py`](src/fast_gemma4/engine.py), [`speculative.py`](src/fast_gemma4/speculative.py) |

### ANE Bundle Variants: `fp16` (`>24 GB` / fallback) vs Calibrated `w8` (`<=24 GB` `"auto"`)

| Variant | Size (`E2B Early14`) | Weight Format | Raw ANE Chunk Speed (`S=256`, `pipe=4`) | When to Use |
|---|---:|---|---:|---|
| **`fp16`** | `1.0 GB` | IEEE `FP16` (`types.fp16`) | `5,799.2 tok/s` (`44.144 ms/chunk`) | **Default on `>24 GB` Macs and fallback everywhere** (`FastGemma4Engine(ane_ring_variant="auto")`, `gemma4` CLI, `train_lora.py`): safe out of the box on base and arbitrary LoRA weights (`8.78 docs/s` Concurrent ANE+GPU on M4 Pro vs `7.99 docs/s` uncalibrated `w8`). |
| **`w8` (opt-in / `"auto"` on `<=24 GB`)** | `504 MB` | Per-channel symmetric `int8` + `AWQ-MSE` | **`7,814.8 tok/s`** (`32.758 ms/chunk`, `+34.8%`, `1326/0/0` ANE/GPU/CPU) | **Domain-calibrated LoRA / QAT checkpoints** (`cos >= 0.9983`, `84/84` exact field parity, `2.723x` prefill / `1.158x` paired `ABBA` E2E speedup). |

- **Certification gate ([`is_w8_bundle_calibrated`](src/fast_gemma4/ane_pipeline.py#L151-L175)):** `ane_calibration_manifest.json` must record `qat_validated=True` or `awq_mse=True` with `>= 8` real corpus/fixture documents (`corpus`, `domain_corpus`, `file`, or `fixtures`). Without real activation calibration, `Early14` quantization drift across autoregressive GPU layers (`15..34`) can trigger repeat loops; explicit `--ane-variant w8` still runs for ablation with a `UserWarning`.
- **Overflow protection on both variants ([`fold_fp16_residual_scales`](scripts/build_ane_models.py#L85-L131)):** Gemma-4 Layer `10..14+` residuals reach `~215K` (`> 65,504` `FP16` max); power-of-2 residual-scale folding and scale-normalized RMSNorm keep both `fp16` and `w8` CoreML graphs inside `FP16` range with `0` GPU/CPU ops.

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **Rolling Metal GPU + `KV_i8`** | Continuous `B_slots=128` scheduler (`refill_min=32`, `group_size=16`), INT8 KV compression + `bfloat4` split-K reduce, `mpp_fused_geglu_bf16_v4`, `EarlyExit14`/`23`, NaN-safe softcap argmax | [`model.py`](src/fast_gemma4/model.py), [`kernels.py`](src/fast_gemma4/kernels.py) |
| **Output vocabulary** | Checkpoint vocabulary by default. The optional 150,786-token profile reduces output-head work and can change the selected text. | [Vocabulary selection](src/fast_gemma4/tokenizer.py) |
| **Structured JSON & Speculative** | `llguidance` zero-alloc bitmask, `forward_ff_jump_slot` causal slice jump, chipset-aware `MTP` + `DraftScan` (`B=1` on `Apple10`, `B=1..4` on `Apple9`), `PromptLookup` | [`grammar.py`](src/fast_gemma4/grammar.py), [`mtp_drafter.py`](src/fast_gemma4/mtp_drafter.py), [`speculative.py`](src/fast_gemma4/speculative.py) |
| **ANE Prefix Ring (`fp16` / calibrated `w8`)** | Residual-folded `FP16` (`1.0 GB`) or `AWQ-MSE` `Linear-W8` (`504 MB`), unlocked `CVPixelBuffer` -> `-newBufferWithIOSurface:` Metal scatter, parallel CoreML build | [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py), [`gemma4_ane_prefix.mm`](native/gemma4_ane_prefix.mm) |
| **Local `QLoRA` Acceleration** | `@mx.compile` chunked softcap+CE & `AdamW`, fixed-bin varlen `FFD` packing with RoPE reset, `8-bit g64` frozen trunk, differentiable `shared-sys-kv` VJP | [`lora_accel.py`](src/fast_gemma4/lora_accel.py), [`train_lora.py`](scripts/train_lora.py) |
| **Routing, Parquet & HTTP** | Auto rolling GPU on `--model-dir`, UTF-8 `page_offsets` (`--max-pages`, `--ocr-pages-only`), coalescing OpenAI SSE server | [`pipeline.py`](src/fast_gemma4/pipeline.py), [`server.py`](src/fast_gemma4/server.py) |

## CLI & Reproduce

```bash
# CoreML ANE bundle builds (`fp16` default, or domain-calibrated `w8` with manifest)
.venv/bin/gemma4 setup-ane --variant both
.venv/bin/python scripts/build_ane_models.py --variant w8 --awq-mse --calib-file docs.jsonl

# Structured generation, resumable Parquet/JSONL extraction, and OpenAI SSE server
.venv/bin/gemma4 generate --json 'Extract entity: Acme Cloud grew 12%.'
.venv/bin/gemma4 parquet docs.parquet out_shards/ --model-dir /ckpt \
  --schema schema.json --system-file sys.txt --max-pages 2 --max-chars 8000 --drop-text-col --resume
.venv/bin/gemma4 extract docs.jsonl out.jsonl --schema schema.json --system-file sys.txt
.venv/bin/gemma4 serve --host 127.0.0.1 --port 8080 --coalesce-ms 10

# Local 8-bit QLoRA + FFD packing + shared-system KV (`--pack-tokens 1024 --grad-accum 2` on 24 GB Macs)
.venv/bin/python scripts/train_lora.py --pairs pairs.json --adapter-dir out_lora --out-model-dir out_ckpt \
  --qlora-8bit --pack-tokens 2048 --shared-sys-kv

# Package benchmark & native CoreML CLI inspection (see monorepo docs/BENCHMARKS.md for outcome/capability suites)
.venv/bin/gemma4 bench --batch-gpu 128 --batch-spec 32 --waves 2
./build/gemma4-ane-cli --model artifacts/ane_models/gemma4_early14_prefix_S256_fp16.mlmodelc --batch 16 --iters 3 --inspect --json
```

## Verification, Scope & Not-tested

- **Verified hosts:** Serialized under `/tmp/m4_gpu_validation.lock` on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Base Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, `Apple10 NAX`)**. CPU & hardware suites pass (`0` `fast-ir-guard --strict-bool-mask` findings; `gemma4 verify` `64/64` valid JSON & schema; `gemma4-ane-cli` `ANE=1326, GPU=0, CPU=0`, `max_int8_diff == 0`, `cos = 0.9991–0.9998`; causal slice jumping bit-exact greedy parity).
- **Memory boundaries.** Batch-128 inference used 6.13 GB of active MLX memory and a 15.54 GB process footprint on the M5. Adapter training completed at `pack_tokens=2048` on the 48 GB M4 Pro and at `pack_tokens=1024, grad_accum=2, ckpt_tokens=512` (`19.2 GB` cap) on the 24 GB M5 (`4.848 GiB` unpacked / `4.849 GiB` packed at `80.52` sup tok/s, `6.759 GiB` shared-KV at `81.63` sup tok/s, `0` OOMs, `loss_diff=0.0, grad_cos=1.0`).
- **v2.2 Silicon & Precision Gates:** Constrained `MTP` runs at `B==1` on Base M5 (`LiquidAI` **`2.026x`** wall / **`2.152x`** decode `39.06 -> 84.05 tok/s`, `8/8` T=0 match; `JSB` `1.740x` decode) and `1<=B<=2` on M4 Pro (`1.48x` / `1.16x`). Metal 4 `MPP::matmul2d` `W4A8` is active on `bits=4 >=8M` Apple10 prefill while un-rotated `W8A8` defaults OFF (`FAST_SILICON_MPP_W8A8=1` / `allow_w8a8=True` opt-in; Base M5 `L13 down_proj` `662.2x` outliers shifted L14 ANE cos to `0.997748 < 0.9983` and `B=8` E2E `5.066s -> 5.239s`), keeping FP32 `fma(128.0|8.0, ws, bq)` metadata packing active. Post-norm Hadamard `W8A8` ANE FFN raises worst 35-layer FFN cos `0.996032 -> 0.999740`, and `p28` `MTLSharedEvent` saves `7.84 ms` (M4 Pro) / `8.64 ms` (M5) CPU wait at `M=1024`. [v2.2 validation](../../docs/benchmarks/v22-validation-20261121.json)
- **Not tested:** Concurrent unserialized Metal/ANE processes, `< 24 GB` Macs at `B_slots=128`, or non-Apple-Silicon targets.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
