# fast-gemma4

Extract fields into schema-constrained JSON or call tools locally with Gemma 4 E2B. On the 24 GB M5, the strict default returned **150/150 valid extractions at 0.946 field F1** and made **123/150 correct tool decisions** in the normal benchmark. Stock `mlx-vlm` scored 0.923 on the same documents, with 146 valid extractions and 120 correct tool decisions. Both loaded the same public instruction-tuned 4-bit checkpoint. [Extraction, tool calls and timing](../../docs/BENCHMARKS.md#structured-generation)

The engine uses the 262,144-token vocabulary shipped with the checkpoint. By default, `llguidance` enforces the JSON schema as tokens are generated and requires one space after each key’s colon (`": "`), which scored 0.946 F1 versus 0.924 unconstrained. Pass `--compact-json` (`whitespace_flexible=False`) for minified JSON or `--whitespace-flexible` for optional whitespace. The optional English-oriented `--vocab-out 150786` profile speeds up output-head projections on ASCII workloads but reduces multilingual field accuracy. [Vocabulary and grammar experiments](../../docs/benchmarks/gemma-grammar-20261121.json)

## Quick start

From the `fast-silicon` workspace root, install the package and prepare the model. This example runs on the GPU.

```bash
uv sync --python 3.13 --package fast-gemma4 --inexact
.venv/bin/gemma4 setup-weights --skip-early14
```

Run this example with `.venv/bin/python`:

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
```

On the M4 Pro this printed `{'entity': 'Ocean Lab', 'topic': 'coral'}`. For compact output on disk or over the wire, serialize `item.parsed` after validation with `json.dumps(item.parsed, ensure_ascii=False, separators=(",", ":"))`.

Optional lazy multimodal image support is available via the `[vision]` extra (`uv pip install -e ".[vision]"`): pass `images=[...]` and `max_soft_tokens=280` to `engine.generate(...)`. Text-only calls keep `_vision_model=None` and never import `PIL` or load vision weights (`0` text-side overhead).

## CLI, Parquet & HTTP server

```bash
# Structured generation, resumable Parquet/JSONL extraction, and OpenAI SSE server
.venv/bin/gemma4 generate --json 'Extract entity: Acme Cloud grew 12%.'
# Optional lazy multimodal image input (install extra: `uv pip install -e ".[vision]"`)
.venv/bin/gemma4 generate --image receipt.png --max-soft-tokens 280 'Transcribe totals as Markdown.'
.venv/bin/gemma4 parquet docs.parquet out_shards/ --model-dir /ckpt \
  --schema schema.json --system-file sys.txt --max-pages 2 --max-chars 8000 --drop-text-col --resume
.venv/bin/gemma4 extract docs.jsonl out.jsonl --schema schema.json --system-file sys.txt
.venv/bin/gemma4 serve --host 127.0.0.1 --port 8080 --coalesce-ms 10

# Optional Core ML ANE Early14 prefill ring (`fp16` default, or domain-calibrated `w8`)
.venv/bin/gemma4 setup-ane --variant both
.venv/bin/python scripts/build_ane_models.py --variant w8 --awq-mse --calib-file docs.jsonl

# Local 8-bit QLoRA adapter training (`--pack-tokens 1024 --grad-accum 2` on 24 GB Macs)
.venv/bin/python scripts/train_lora.py --pairs pairs.json --adapter-dir out_lora --out-model-dir out_ckpt \
  --qlora-8bit --pack-tokens 2048 --shared-sys-kv
```

## Benchmark highlights

### 1. Public outcomes & batch capacity

| Workload | Base M5 (24 GB) | M4 Pro (48 GB) | Comparison & notes |
|---|---|---|---|
| **Strict JSON + tool calls (M5, 300 tasks)** | **1.835 score/s** (300/300 valid, 0.946 field F1, 144.4 s) | — | **1.22×** unconstrained stock `mlx-vlm` (1.501 score/s, 286/300 valid, 0.923 F1, 172.2 s) on the same checkpoint. |
| **Historical minified 150K profile (300 tasks)** | 2.717 score/s (299/300 valid, 82.7 s) | 2.412 score/s (299/300 valid, 92.9 s) | Faster output head, lower multilingual field F1 (0.749 combined score vs 0.883 repaired strict). |
| **Reasoning & instruction following (435 tasks)** | **0.843 score/s** (0.352 score, 181.5 s) | 0.892 score/s (0.359 score, 174.8 s) | 1.03× (M5) and 0.94× (M4 Pro) vs stock `mlx-vlm` (0.366 score). |
| **Batch-128 extraction fixtures (`kv_i8`)** | **10.46 valid docs/s** (GPU, 6.13 GB MLX) | **8.78 valid docs/s** (ANE+GPU) · 6.87 (GPU) | Holds 128 rolling slots inside 6.1–6.9 GB active MLX RAM, whereas unchunked batch 64 in stock `mlx-vlm` OOMs at 34.4 GB. |

### 2. Speculative decoding, prefix reuse & local training

| Capability | Base M5 (24 GB) | M4 Pro (48 GB) | Takeaway |
|---|---|---|---|
| **Speculative `MTP` (`B=1` greedy JSON)** | **84.1 tok/s** (2.15× decode speedup on LiquidAI) | **121.4 tok/s** (up from 46.8 tok/s on fixture) | Enabled automatically at `B=1` on M5 and `B=1..2` on M4 Pro; dense GPU batching wins at larger batches. |
| **Shared-prefix `PrefixTree` reuse** | Zero system-KV cloning across batch slots | Up to **2.59×** faster across 4–8 shared-prefix candidates | Automatically reuses shared prompt prefixes (`>= 32` tokens). |
| **Local 8-bit `QLoRA` adapter training** | **80.5–87.7 sup tok/s** in 4.85–6.76 GiB (`pack=1024, accum=2`) | **545.8 tok/s** packed 8-bit (`pack=2048`) | Sizing to `pack_tokens=1024, grad_accum=2, ckpt_tokens=512` eliminates 24 GB OOMs with matching loss and gradients. |
| **Disaggregated `GRPO` + `Laya` ANE critic** | **2.61–3.38 rollouts/s** (8.68 GB peak) | **1.81–3.15 rollouts/s** (7.11 GB vs 19.38 GB GPU critic) | Runs critic scoring on the Neural Engine concurrently with GPU rollouts and backward passes. |

#### Optional lazy multimodal vision (M4 Pro quick check)

On M4 Pro (`48 GB`), optional `generate(..., images=...)` leaves text-only latency and RAM unchanged (`63.09 tok/s`, `3.33 GB` MLX peak, `_vision_model=None`), while `Gemma-4 E2B-4bit` (`4.24–4.30 GB` MLX / `5.43 GB` RSS) scored `0.7665` TEDS (`@280` soft tokens, `0.170 pp/s`) → `0.8293` TEDS (`@1120`, `0.9053` NumF1) on `FinTabNet`, `1.000` schema validity (`0.2566` F1, `0.506 pp/s`, `95.9 tok/s`) on `CORD-v2`, `0.5128` (`0.364 pp/s`) on `OmniDocBench-EN(2)`, and `0.1667` on `olmOCR-bench(2)`. `E4B-bf16` (`14.9–15.1 GB` MLX / `17.05 GB` RSS, `16–29 tok/s`) lifted `olmOCR-bench(2)` to `0.500` (`3/3` `old_scans_math`) while trailing E2B on `FinTabNet` (`0.3932`) and `CORD-v2` (`0.1146`). Use `fast-ocr` (`OvisOCR2-8bit`) for dedicated document OCR and `fast-gemma4` vision for lightweight multimodal extraction/VQA. [Vision & OCR benchmarks](../../docs/BENCHMARKS.md#136-optional-gemma-4--embeddinggemma-2-vision--cross-modal-retrieval-m4-pro-verified)

## Automatic hardware dispatch (`"auto"` defaults)

`FastGemma4Engine()` probes chip generation and RAM via `DynamicSiliconPolicy` so callers do not have to hand-tune flags per Mac:

| Parameter | `"auto"` behavior | Manual override |
|---|---|---|
| `mtp_drafter` | Lazily loads the ~150 MiB draft assistant only on greedy single-stream GPU calls (`B=1` on M5, `B=1..2` on M4 Pro); costs 0 MB on batch ETL. | Pass `False` / `None` to disable, or `True` to load eagerly. |
| `use_ane_ring` | Uses the `Early14` Pure-ANE prefill ring on `>= 32 GB` M4 Macs for batches `>= 16`; stays pure GPU on M5 `NAX` (where GPU reaches 10.46 docs/s) and on 24 GB hosts. | Pass `False` to force GPU or `True` to require the ANE ring. |
| `ane_ring_variant` | Uses `fp16` (`1.0 GB`, safe for arbitrary base or LoRA weights) on `> 24 GB` Macs; uses calibrated Hadamard `w8` (`504 MB`, `7,815 tok/s` chunk prefill) on `<= 24 GB` Macs only when a verified calibration manifest is present. | Pass `"fp16"` or `"w8"`. |

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **Rolling Metal GPU + `kv_i8`** | Continuous 128-slot scheduler, INT8 KV cache compression, fused BF16 GeGLU, and FP32 logit softcapping (`0/1,024` greedy mismatches vs FP32) | [`model.py`](src/fast_gemma4/model.py), [`kernels.py`](src/fast_gemma4/kernels.py) |
| **Schema grammar & loop guard** | Zero-allocation `llguidance` bitmasking, multi-token forced-prefix jumps, deferred length-truncation retries, and a cycle/4-gram loop breaker | [`grammar.py`](src/fast_gemma4/grammar.py), [`speculative.py`](src/fast_gemma4/speculative.py) |
| **Pure-ANE `Early14` prefill ring** | Power-of-2 residual-scale folding keeps Layer 10–14 activations inside FP16 range (`1,326` ANE ops, `0` GPU/CPU) with zero-copy `IOSurface` handoff to Metal | [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py), [`native/gemma4_ane_prefix.mm`](native/gemma4_ane_prefix.mm) |
| **Local `QLoRA` acceleration** | Compiled chunked softcap cross-entropy + AdamW, variable-length FFD sequence packing, and shared system-prompt KV backward passes | [`lora_accel.py`](src/fast_gemma4/lora_accel.py), [`scripts/train_lora.py`](scripts/train_lora.py) |

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)** under `/tmp/m4_gpu_validation.lock`. Detailed precision ablations and negative-gate records are in [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md).

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
