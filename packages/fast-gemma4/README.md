# fast-gemma4

High-throughput local JSON extraction, tool calling, speculative decoding, and 8-bit QLoRA on Apple Silicon with `google/gemma-4-E2B-it`. Holds **128 concurrent rolling batch slots inside 6.13 GB active MLX memory** at **10.46 valid docs/s** (`1,479 tok/s` on Base M5: `5.2x` stock `mlx-vlm` valid doc rate and `9.8x` Apple CoreAI `B=1`), while returning **150/150 valid extractions at 0.946 field F1** on multilingual forms.

Zero telemetry, no account, and runs 100% offline after initial weight download (~10.4 GB full setup with Early14 Core ML ANE + MTP assistant; smaller when skipping ANE via `--skip-early14`). Prebuilt Early14 Core ML ANE artifact: [`mikewolfd/gemma-4-e2b-it-early14-coreml-ane`](https://huggingface.co/mikewolfd/gemma-4-e2b-it-early14-coreml-ane) · [Full benchmarks](../../docs/BENCHMARKS.md#structured-generation).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+, Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported. Checked with `lmstudio-community/gemma-4-E2B-it-MLX-4bit`.

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-gemma4 --inexact

.venv/bin/gemma4 setup-weights --skip-early14   # GPU-only weights (~2.5-3 GB); omit flag to add ~1 GB Early14 ANE
```

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
        print(item.parsed)  # {'entity': 'Ocean Lab', 'topic': 'coral'}
```

## Grammar formatting & baseline fairness

The engine uses the full 262,144-token vocabulary from the checkpoint. By default, `llguidance` enforces your JSON schema token-by-token and inserts one space after each colon (`": "`), which scored **0.946 F1** vs. **0.924 unconstrained** (minified `whitespace_flexible=False` breaks Gemma's normal `" \"` token merge and lowers multilingual accuracy; `--vocab-out 150786` speeds ASCII-only heads but hurts non-English fields). [Grammar & vocab experiments](../../docs/benchmarks/gemma-grammar-20261121.json)

When comparing against unconstrained stock `mlx-vlm` below, invalid JSON outputs receive 0 credit in end-to-end scoring; both validity lift (`150/150` vs. `146/150`) and raw elapsed time (`144.4 s` vs. `172.2 s`) are shown separately.

## Benchmark highlights

### 1. Structured extraction, tools & batch capacity

| Workload | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Baseline comparison |
| --- | --- | --- | --- |
| **300-task JSON + tools (batched, strict)** | **4.179 score/s** (300/300 valid, 0.884 score, 4.98 GB) | **3.639 score/s** (300/300 valid, 0.880 score, 8.15 GB) | Batched rolling scheduler (`max_b_slots=128` on 24 GB) |
| **300-task JSON + tools (`B=8` M5)** | **1.835 score/s** (150/150 extract 0.946 F1, 123/150 tools, 144.4 s) | - | Unconstrained `mlx-vlm`: `1.501` (146/150 extract 0.923 F1, 120/150 tools, 172.2 s) |
| **Reasoning (`435` MMLU + MATH + IFBench)** | **0.843 score/s** (0.352 score, 181.5 s) | **0.892 score/s** (0.359 score, 174.8 s) | Stock `mlx-vlm`: `0.818` M5 / `0.951` M4 Pro (0.366 score) |
| **Batch-128 extraction (`kv_i8`)** | **10.46 valid docs/s** (`1,479 tok/s`, 6.13 GB MLX) | **8.78 valid docs/s** (ANE+GPU) / **6.87** (GPU) | **5.2x** `mlx-vlm` valid rate (`B=128` fits in 6.1 GB; stock OOMs at 34.4 GB on `B=64`) |

### 2. Speculative decoding (`MTP`), prefix reuse & local QLoRA

| Capability | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Takeaway |
| --- | --- | --- | --- |
| **Speculative `MTP` (`B=1` / `B=4` vs AR)** | **1.65-1.68x** `B=1` (**2.15x** vs `mlx-vlm`), **1.00x** `json_b4`, `10/10` exact (9.30 GB) | **1.49-1.62x** `B=1`, **1.13x** `free_b4`, `10/10` exact (9.53 GB) | Keeps singleton verify on `B=1`, batched linears on `B>=2`, and dense rolling decode when oversaturated |
| **Shared-prefix `PrefixTree`** | Zero system-KV cloning across batch slots | Up to **2.59x** across 4-8 shared-prefix candidates | Automatically reuses shared prefixes `>= 32` tokens |
| **8-bit `QLoRA` (`shared-sys-kv`)** | **533-1,097 tok/s** in 5.2-7.3 GB (`pack=1024`, `accum=2`) | **546-1,385 tok/s** (`pack=2048`) | Auto-sizes pack length by RAM to avoid 24 GB OOM at matching loss and gradients |
| **Disaggregated `GRPO` + `Laya` ANE critic** | **2.61-3.38 rollouts/s** (8.68 GB peak) | **1.81-3.15 rollouts/s** (7.11 GB vs. 19.38 GB GPU) | Runs critic scoring on ANE concurrently with GPU rollouts and backward passes |

**Optional lazy vision & multimodal `MTP`:** Install `uv sync --python 3.13 --package fast-gemma4 --extra vision --inexact`. Text-only calls keep zero vision weights in RAM (`63.09 tok/s`, `3.33 GB` MLX peak). With images passed (`generate(..., images=...)`), multimodal `MTP` on Base M5 speeds `FinTabNet(4)` **2.12x** (`34.05 s -> 16.04 s`, `48.8 -> 103.6 tok/s`, 4/4 exact parity) and `CORD-v2(4)` JSON **1.51x** (`26.02 s -> 17.28 s`, 4/4 parity). For full-document OCR benchmarks at scale, use [`fast-ocr`](../fast-ocr). [Vision & MTP receipts](../../docs/BENCHMARKS.md#141-vision-retrieval-cpu-image-prefetch-and-multimodal-mtp)

## Automatic hardware dispatch (`"auto"` defaults)

`FastGemma4Engine()` probes Apple Silicon generation and unified RAM automatically:

| Option | `"auto"` behavior | Override |
| --- | --- | --- |
| `mtp_drafter` | Lazily loads the ~150 MiB draft assistant only for greedy single/low-batch GPU calls (`B=1` on M5, `B=1..2` on M4 Pro); 0 MB cost on batch ETL. | Pass `False`/`None` to disable, or `True` to preload |
| `use_ane_ring` (`generate_batch`) | Enables `Early14` Pure-ANE prefill on `>= 32 GB` M4 Macs (`B >= 16`); stays pure GPU on M5 `NAX` (where GPU reaches 10.46 docs/s) and 24 GB Macs. | Pass `False` (GPU) or `True` (force ANE) |
| `ane_ring_variant` | Uses `fp16` (1.0 GB) on `> 24 GB` hosts and calibrated `w8` (504 MB) on `<= 24 GB` hosts when present. | Pass `"fp16"` or `"w8"` |

## What keeps it fast

| Component | How it works | Source |
| --- | --- | --- |
| **Rolling Metal GPU + `kv_i8`** | Continuous 128-slot scheduler, INT8 KV compression, fused BF16 GeGLU, Apple10 NAX inline XOR packing without duplicate weight buffers (`max_diff=0.0`), and FP32 logit softcap | [`model.py`](src/fast_gemma4/model.py), [`kernels.py`](src/fast_gemma4/kernels.py) |
| **Schema grammar & loop guard** | Zero-allocation `llguidance` bitmasking, forced-prefix jumps, deferred length-truncation retries, and 4-gram loop breaking | [`grammar.py`](src/fast_gemma4/grammar.py), [`speculative.py`](src/fast_gemma4/speculative.py) |
| **Pure-ANE `Early14` prefill ring** | Residual-scale folding keeps Layer 10-14 activations inside FP16 (`1,326` ANE ops, `0` GPU/CPU) with zero-copy `IOSurface` handoff | [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py), [`native/gemma4_ane_prefix.mm`](native/gemma4_ane_prefix.mm) |
| **Local `QLoRA` acceleration** | Compiled chunked softcap cross-entropy + AdamW, FFD sequence packing, and shared system-prompt KV backward passes | [`lora_accel.py`](src/fast_gemma4/lora_accel.py), [`scripts/train_lora.py`](scripts/train_lora.py) |

## CLI, Parquet & HTTP server

```bash
.venv/bin/gemma4 generate --json 'Extract entity: Acme Cloud grew 12%.'
.venv/bin/gemma4 parquet docs.parquet out_shards/ --schema schema.json --system-file sys.txt --drop-text-col --resume
.venv/bin/gemma4 serve --host 127.0.0.1 --port 8080 --coalesce-ms 10
.venv/bin/python packages/fast-gemma4/scripts/train_lora.py --pairs pairs.json --adapter-dir out_lora --out-model-dir out_ckpt --qlora-8bit --pack-tokens 2048 --shared-sys-kv
```

## Limitations, Prior Art & License

- **Limitations:** HTTP `serve` batches completed responses rather than token-by-token streams. `Early14` ANE prefill improves M4 Pro throughput (`8.78` vs. `6.87` docs/s) while Base M5 `NAX` GPU is already faster without it. On free-form reasoning (no schema), `fast-gemma4` matches `mlx-vlm` speed (`0.94x` M4 Pro, `1.03x` M5) rather than outpacing it.
- **Prior art:** Built on `mlx`, `mlx-vlm`, `llguidance`, `coremltools`, and `lmstudio-community/gemma-4-E2B-it-MLX-4bit`.
- **License:** Package code is [MIT licensed](LICENSE); Gemma 4 weights are governed by Google's Gemma Terms of Use. Verified on 48 GB M4 Pro and 24 GB Base M5.
