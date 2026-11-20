# fast-gemma4

**Guaranteed schema-valid JSON extraction and local QLoRA / GRPO post-training for Gemma-4 E2B — up to `5.18x` faster schema-valid throughput (`10.46 valid docs/s` on Base M5 `24 GB`) and `1.81x` higher `EOT` than stock `mlx-vlm` in `6.1–6.9 GB` active MLX RAM instead of `34.4 GB` OOM.**

Built for high-volume local structured extraction on Apple Silicon (`Apple9` M4 and `Apple10+` M5 `NAX`), `fast-gemma4` combines continuous rolling batching (`B_slots=128`), `int8` KV cache compression, lossless output vocabulary pruning (`262,144 -> 150,786` with lazy full-vocab fallback), chipset-aware `MTP` speculative decoding (`DraftScan`), and an optional Neural Engine prefill ring. Run it from Python, extract directly from resumable PDF/Parquet shards (`--max-pages`, `--ocr-pages-only`) or JSONL, fine-tune locally with `8-bit` `QLoRA`, or serve an OpenAI-compatible SSE endpoint.

> **Full Benchmarks & Methodology:** For complete cross-silicon tables (`M4 Pro 48 GB` vs `Base M5 24 GB`), all 16 deterministic public outcome datasets, 4-bucket `RUSAGE_INFO_V6` footprints, and reproducible CLI commands, see the monorepo [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gemma4 setup-weights --local-files-only
.venv/bin/gemma4 verify
```

```python
from fast_gemma4 import FastGemma4Engine

with FastGemma4Engine(
    default_mode="auto",
    mtp_drafter="auto",       # Lazy-loads MTP only on qualifying low-B waves (0 MB RAM on B>=8)
    ane_ring_variant="auto",  # Calibrated w8 on <=24 GB Macs when certified; fp16 otherwise
) as eng:
    item = eng.generate(
        "Extract entity and topic as JSON.",
        json_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
            "required": ["entity", "topic"],
            "additionalProperties": False,
        },
        max_tokens=128,
    )
    print(item.parsed)
```

## Cross-Silicon Benchmark Highlights (`M4 Pro 48 GB` vs `Base M5 24 GB`)

All runs execute in isolated subprocesses under `/tmp/m4_gpu_validation.lock` (`MLX 0.32.4`), pairing `fast-gemma4` against stock `mlx-vlm` (`lmstudio-community/gemma-4-E2B-it-MLX-4bit`). See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for full tables.

### 1. Outcome (`EOT`) & Schema-Valid Batch Throughput

| Benchmark / Workload | Base M5 (`24 GB`, `10c Apple10 NAX`) | M4 Pro (`48 GB`, `20c Apple9`) | vs Stock `mlx-vlm` (`M5` / `M4 Pro`) | M5 vs M4 Pro |
|---|---|---|---|---|
| **Cat 4: JSON & Tool AST** (`LiquidAI/structured-extraction + BFCL v3`, `N=300`, `llg_on`) | **`2.717 EOT`** (`299/300` valid, `S_obj=0.7487`; `llg_off`: `1.848 EOT`, `S_obj=0.8241`) | **`2.412 EOT`** (`299/300` valid, `S_obj=0.7473`; `llg_off`: `1.702 EOT`, `S_obj=0.8178`) | **`1.81x`** / **`1.57x`** `EOT` *(vs stock `1.501` / `1.537 EOT`, `286–287/300` valid)* | **`1.13x`** |
| **Cat 3: Reasoning & IF** (`IFBench + MMLU-Redux-2.0 + MATH-500`, `N=435`, `gpu_fp16_kvi8`) | **`0.843 EOT`** (`181.5 s`, `S_obj=0.3517`) | `0.892 EOT` (`174.8 s`, `S_obj=0.3586`) | **`1.03x`** / `0.94x` *(vs stock `0.818` / `0.951 EOT`, `S_obj=0.3655`)* | `0.95x` |
| **Monorepo Batch Extraction** (`B=128`, `kv_i8`, `100%` strict schema) | **`10.46 valid docs/s`** (`1,479 dec tok/s`, `6.13 GB` MLX / `15.54 GB` OS) | **`6.87 GPU` / `8.78 ANE+GPU valid docs/s`** (`1,144` / `1,131 dec tok/s`, `6.28–6.33 GB` MLX / `18.96 GB` OS) | **`2.59x` raw / `5.18x` valid** (`M5`) · **`1.91x–2.44x` raw / `3.82x–4.88x` valid** (`M4 Pro`; stock `50%` schema valid) | **`1.52x`** (`GPU`) / **`1.19x`** (`peak`) |

- **Memory discipline:** `fast-gemma4` holds `B_slots=128` inside **`6.13–6.88 GB` peak active MLX RAM** (`kv_i8` saves `-0.67 GB` to `-1.21 GB` vs `BF16` KV), whereas unchunked `B=64` in stock `mlx_vlm` hits **`34.4 GB` OOM** (`15.24 GB` even with `chunk=256`).
- **End-to-end Parquet pipeline (M4 Pro `48 GB`, Tuned LoRA, `p1-2`, `max_chars=8000`):** **`2.33 docs/s`** (`255/255` valid, `1,051.8 dec tok/s`, `5,443.2 total tok/s`, `6.88 GB` peak).

### 2. Speculative (`MTP`), Prefix-Tree & Local Post-Training (`QLoRA` / `GRPO`)

| Capability Suite | Base M5 (`24 GB`, `Apple10 NAX`) | M4 Pro (`48 GB`, `Apple9`) | Key Takeaway |
|---|---|---|---|
| **`MTP` Speculative + `DraftScan`** (`~150 MiB` `4`-layer drafter, `6.1%` draft share) | • `B=1` free-text: `33.5 ->` **`83.4 tok/s`** (**`2.49x`**)<br>• `B=4` free-text: `134.9 ->` **`212.1 tok/s`** (**`1.57x`**) | • `B=1` JSON: **`121.4 dec tok/s`** (`exact_match=True`, vs `69.0` stock `mlx-vlm`)<br>• `B=4` JSON: `1.09 ->` **`1.90 docs/s`** (**`1.74x`**, `272.6 tok/s`) | Strong at low `B`; auto-disabled at `B>=5` (avoids `0.56x` `B=8` saturation) and at `B>=2` on `Apple10 NAX` via `mtp_drafter="auto"`. |
| **Shared-Prefix & Prefix-Tree KV Reuse** | Zero system-KV cloning across batch slots | Zero system-KV cloning across batch slots | `select_speculative_policy` enables prefix-tree KV verification (`shared_prefix_tokens >= 32`, `num_candidates >= 2`; up to **`2.59x`** at `N=4..8` shared-prefix candidates; see [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md)). |
| **Local `QLoRA` + `Shared-Sys-KV`** (`8-bit g64` + `FFD` packing) | `pack=2048` (`25.1 GB`) exceeds `19.07 GB` Metal cap -> `recommend_silicon_budget` auto-scales to **`pack=1024, grad_accum=2, ckpt_tokens=512`** (`<= 19.2 GB` UMA cap) | • Stock `mlx-lm` BF16 LoRA: `579.1 tok/s` (`16.1 GB`)<br>• **`8-bit` + `shared-sys-kv`:** **`1,385.4 tok/s` (`2.39x`)** *(up to `1,581 tok/s` / `2.88x` on `1,288`-tok schemas)* | `packed_grad_cos = 0.99967`, `qlora_8bit_grad_cos = 0.99488`, `|Δloss| = 0.0039`; cuts frozen trunk weights `9.23 -> 5.00 GB`. |
| **Disaggregated `G=4` `GRPO` + Concurrent `Laya` ANE Critic** | **`2.61 rollouts/s`** (`T=0.7`) / **`3.38 rollouts/s`** (`T=0.0` BoN, `8.68 GB`) = **`5.22x` / `6.77x`** vs stock `0.50` (`99.96%` ANE critic hidden) | **`1.81 rollouts/s`** (`T=0.7`) / **`3.15 rollouts/s`** (`T=0.0`, `7.11 GB`, `-63%` RAM) = **`8.93x` / `15.51x`** rollouts (**`7.44x` / `11.30x`** full step; `99.999%` hidden) | `10c` Base M5 generates constrained `GRPO` rollouts **`1.44x` faster** than `20c` M4 Pro with `0 GB` GPU critic VRAM. |

## Dynamic Runtime Modes (`094ae65`)

`FastGemma4Engine` integrates `fast_shared.runtime_lessons.DynamicSiliconPolicy` so callers can opt into hardware- and batch-aware dispatch without hand-tuning per Mac:

| Parameter | Default | `"auto"` Behavior | Source |
|---|---|---|---|
| `mtp_drafter` | `None` (off) | **Lazy chipset-aware `MTP` loading:** pays **`0 MB` RAM and `0 ms` init** on `B>=8` batch ETL, stochastic (`T > 0`), ANE waves, or when no local drafter is staged. On the first qualifying greedy GPU wave, loads `google/gemma-4-E2B-it-assistant` (`~150 MiB`) once — strictly for **`B == 1`** on `Apple10+ NAX` (M5, where NAX TensorOps accelerate `B >= 2` directly by `1.5x–1.9x`) and **`1 <= B <= 4`** (`K=3/4` at `B=1`, `K=2` at `2 <= B <= 4`) on `Apple9` (M4). | [`engine.py`](src/fast_gemma4/engine.py), [`mtp_drafter.py`](src/fast_gemma4/mtp_drafter.py) |
| `ane_ring_variant` | `"fp16"` | **RAM- & certification-aware ANE ring:** probes host RAM via `probe_host_hardware()` + `DynamicSiliconPolicy`; on `<= 24 GB` Macs selects **`"w8"`** (`504 MB`, `7,814.8 tok/s` ANE chunk prefill) **only when** a staged `w8` bundle exists and passes [`is_w8_bundle_calibrated`](src/fast_gemma4/ane_pipeline.py#L151-L175), falling back cleanly to **`"fp16"`** (`1.0 GB`, `5,799.2 tok/s`) if uncalibrated or missing. On `> 24 GB` Macs, stays on `"fp16"` unless `"w8"` is explicitly passed. | [`engine.py`](src/fast_gemma4/engine.py), [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py) |

### ANE Bundle Variants: `fp16` (Default) vs Calibrated `w8`

| Variant | Size (`E2B Early14`) | Weight Format | Raw ANE Chunk Speed (`S=256`, `pipe=4`) | When to Use |
|---|---:|---|---:|---|
| **`fp16` (default)** | `1.0 GB` | IEEE `FP16` (`types.fp16`) | `5,799.2 tok/s` (`44.144 ms/chunk`) | **Default everywhere** (`FastGemma4Engine`, `gemma4` CLI, `train_lora.py`): safe out of the box on base and arbitrary LoRA weights (`8.78 docs/s` Concurrent ANE+GPU on M4 Pro vs `7.99 docs/s` uncalibrated `w8`). |
| **`w8` (opt-in / `"auto"` on `<=24 GB`)** | `504 MB` | Per-channel symmetric `int8` + `AWQ-MSE` | **`7,814.8 tok/s`** (`32.758 ms/chunk`, `+34.8%`, `1326/0/0` ANE/GPU/CPU) | **Domain-calibrated LoRA / QAT checkpoints** (`cos >= 0.9983`, `84/84` exact field parity, `2.723x` prefill / `1.158x` paired `ABBA` E2E speedup). |

- **Certification gate ([`is_w8_bundle_calibrated`](src/fast_gemma4/ane_pipeline.py#L151-L175)):** `ane_calibration_manifest.json` must record `qat_validated=True` or `awq_mse=True` with `>= 8` real corpus/fixture documents (`corpus`, `domain_corpus`, `file`, or `fixtures`). Without real activation calibration, `Early14` quantization drift across autoregressive GPU layers (`15..34`) can trigger repeat loops; explicit `--ane-variant w8` still runs for ablation with a `UserWarning`.
- **Overflow protection on both variants ([`fold_fp16_residual_scales`](scripts/build_ane_models.py#L85-L131)):** Gemma-4 Layer `10..14+` residuals reach `~215K` (`> 65,504` `FP16` max); power-of-2 residual-scale folding and scale-normalized RMSNorm keep both `fp16` and `w8` CoreML graphs inside `FP16` range with `0` GPU/CPU ops.

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **Rolling Metal GPU + `KV_i8`** | Continuous `B_slots=128` scheduler (`refill_min=32`, `group_size=16`), INT8 KV compression + `bfloat4` split-K reduce, `mpp_fused_geglu_bf16_v4`, `EarlyExit14`/`23`, NaN-safe softcap argmax | [`model.py`](src/fast_gemma4/model.py), [`kernels.py`](src/fast_gemma4/kernels.py) |
| **Lossless Vocab Pruning** | `262,144 -> 150,786` pruned embeddings + PLE tables ([`build_vocab_remap.py`](scripts/build_vocab_remap.py)) with lazy `262K` fallback + byte tokens | [`tokenizer.py`](src/fast_gemma4/tokenizer.py) |
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
- **Memory boundaries:** `B_slots=128` inference fits cleanly on `24 GB` (`6.13 GB` MLX / `15.54 GB` OS peak) and `48 GB` hosts; full `pack=2048` `E2B` `QLoRA` peaks at `25.1 GB` (`>= 32 GB` UMA recommended, or `pack=1024, grad_accum=2, ckpt_tokens=512` on `24 GB`).
- **Not tested:** Concurrent unserialized Metal/ANE processes, `< 24 GB` Macs at `B_slots=128`, or non-Apple-Silicon targets.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
