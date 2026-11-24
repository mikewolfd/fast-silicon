# fast-silicon

Search documents, score choices about a passage, or extract structured records locally on an Apple Silicon Mac. Each model package has a Python API and a command-line tool. The `fast-*` packages in this repository choose how to run each model on the GPU, the Apple Neural Engine (ANE), or both.

On the 24 GB M5, `fast-embeddinggemma-2` and OpenInfinity’s public 8-bit Core ML export took almost the same time to encode one warm search query. For bulk encoding, the project package produced document vectors at 3.27× the token rate. The public export used less memory in the scored retrieval test. [Query, bulk and memory comparison](#benchmark-highlights)

## Quick start

From the repository root, install BGE and its search weights. This example uses the GPU and works without compiling Neural Engine models.

```bash
uv sync --python 3.13 --package fast-bge-small --inexact
.venv/bin/bge-small setup-weights
```

Run this with `.venv/bin/python`. It prints the passage about ocean currents.

```python
from fast_bge_small import FastBgeSmall

passages = [
    "Ocean currents carry heat from the tropics toward the poles.",
    "A piano has 88 keys.",
]
engine = FastBgeSmall(default_mode="gpu", enable_ane=False)
query = engine.encode_texts("How does heat move through the ocean?", add_query_prefix=True)
vectors = engine.encode_texts(passages)
print(passages[int((vectors @ query).argmax())])
```

For the full developer workspace, use `uv sync --extra dev --extra extract --extra build`. Preview downloads and package builds with `.venv/bin/python scripts/setup_all.py --dry-run`, or choose packages with `make setup SETUP_ARGS="--packages bge-small,pplx"`. The full setup stages approximately 35 GB. [Setup options](scripts/setup_all.py)

## Choose a package

| What you want to do | Package | What matters for that choice |
| --- | --- | --- |
| Search short English passages | [BGE-small](packages/fast-bge-small) | Produces a compact 384-number vector. A warm 24-token query took 0.49 ms on the M4 Pro and 0.79 ms on the M5. |
| Index longer passages & page images | [EmbeddingGemma-2](packages/fast-embeddinggemma-2) | Reads up to 2,048 tokens per passage (128–768 dims). With optional `[vision]`, encodes page images at **18.98 img/s on M5** and **10.89 img/s on M4 Pro** (`93.75%` cross-modal R@1 on 16 pages) with zero text-side overhead. |
| Find a passage that depends on surrounding text | [PPLX](packages/fast-pplx) | On 175 legal questions, encoding neighboring text raised passage ranking from 0.359 to 0.703 and finished 325 M5 queries in 136.5 s vs. stock 601.5 s. Lazy `encode_images` reaches **8.90 img/s on M5** and **6.00 img/s on M4 Pro** at unchanged retrieval. |
| Prioritize documents for review or score choices about a passage | [Decision models](packages/fast-decision-models) | 6-bucket multifunction Laya Core ML (`[10538, 0, 0]` Pure-ANE) scores **`711/1400` (`0.5079`)** in **`76.7 s`** (`9.27` EOT) ANE / **`37.9 s`** (`18.59` EOT) `laya-hetero` on Base M5 (`133.1 s` / `59.3 s` on M4 Pro). Kev W8 hits **`122.8 s`** (`6.00` EOT, `1.18×` Stock `mlx-lm`) on Base M5 Tier-1 and **`2.97×`** on shared-passage `PrefixTree`. |
| Extract fields or call tools | [Gemma 4](packages/fast-gemma4) | Strict extraction recovered 0.946 F1 on 150 multilingual forms (`150/150` valid). `MTP` decoding runs **1.49–1.68×** faster on `B=1` text (`10/10` parity across `B=1` and `B=4`) and **2.12×** faster end to end on `FinTabNet(4)` vision. |
| Extract fields or call tools with Qwen 3.5 / 3.6 | [Qwen 3.5 / 3.6](packages/fast-qwen3) | On 300 Cat 4 tasks (`LiquidAI(150) + BFCL_v3(150)`), schema-constrained 4B hits **`300/300` (`0.8955`, `0` loops, `PASS`)** on Base M5 (`0.8930`, `299/300` on M4 Pro). Apple10 NAX inline XOR packing avoids `2.34–4.2+ GB` of duplicate shadow RAM at zero logit drift. |
| OCR PDFs, scans, and financial tables | [OCR](packages/fast-ocr) | Routes clean PDF text, parallel Apple Vision ANE (`9.61 pp/s`), and `OvisOCR2-8bit` / `NuExtract3`: verified Wave 3 Tier-1 hits **`0.6155` `olmOCR(42)`**, **`0.8935` `FinTabNet(30)`**, **`0.6574` `OmniDoc(24)`** (all 3 `PASS` on M4 Pro) and **`0.6128` / `0.8867` (`PASS`) / `0.6872` (`PASS`)** on Base M5. |
| Generate structured records in large batches | [GPT-OSS](packages/fast-gpt-oss) | The 20B model handled 128 fixture requests at once on the 48 GB M4 Pro. The benchmark’s memory policy excluded the 24 GB M5. |
| Build a resumable document job or serve a model over HTTP | [Shared runtime](packages/fast-shared) | Model setup, compact vector indexing, batch processing, HTTP serving and memory measurement for the model packages. |

## Benchmark highlights

All implementations run locally. **Stock MLX** means the named public model and MLX library used in that test. A **public Core ML export** is a publisher’s preconverted model run with Apple’s Core ML framework. The `fast-*` packages can also use MLX and Core ML. **Apple CoreAI** is the separate runtime for `.aimodel` files. [Implementations and model names](docs/BENCHMARKS.md#which-implementation-ran)

| EmbeddingGemma workload on the 24 GB M5 | `fast-embeddinggemma-2` | OpenInfinity INT8 in Apple Core ML |
| --- | ---: | ---: |
| One warm 24-token query | 2.58 ms | 2.59 ms |
| Bulk encoding, same 2,000 documents | 48,803 tokens/s | 14,945 tokens/s |
| Combined memory, separate 500-item retrieval test | 3.43 GB | 0.84 GB |

The first two rows time token IDs through the returned vector. Tokenization, network calls, and writing vectors to a search index fall outside those timings. For an app that starts the model on demand, comparable first-response times still need measurement. [Matched inputs and timers](docs/BENCHMARKS.md#warm-query-corpus-and-packing) · [Retrieval scores and memory](docs/BENCHMARKS.md#task-quality-and-elapsed-time) · [First use](docs/BENCHMARKS.md#first-use-and-reproduction)

- **Retrieval and compact indexing.** Across the M4 Pro and M5, BGE delivered 9.58–11.26× the ranking score per second of stock MLX, EmbeddingGemma 4.33–4.80×, and PPLX 4.33–4.41×. Prepending document headers when chunking Parquet files (`--header-col`) raised 175-query legal passage ranking from 0.305 to 0.444 on BGE, 0.380 to 0.505 on EmbeddingGemma, and 0.453 to 0.735 on contextual PPLX. All three encoders share `CompactRetrievalIndex`, which pairs a 1-bit Hamming prefilter (`rerank_top_r=256` default in `search()`) with INT8 reranking to search 100,000 vectors in 0.50–0.66 ms per query (`top64`, 2.95–4.43× faster than FP32 BLAS) without measurable ranking loss. [Retrieval scores, times, and memory](docs/BENCHMARKS.md#retrieval) · [Compact index modes](docs/BENCHMARKS.md#122-model-aware-mrl--int8--arm64-neon-binary-rerank-fast-bge-small-fast-embeddinggemma-2-fast-pplx)
- **Long-document search (4K–32K).** Splitting 32,768-token documents into 2,048-token contextual windows and scoring the best-matching window (`MaxSim`) ran 2.31× faster on the GPU and used 2.48× less peak Metal memory than a single 32K pass in `fast-pplx`. On `LongEmbed`, that raised 32K passkey ranking from 0.047 to 1.000 on the 100-document split (0.674 when all four length splits are pooled together) and `2wikimqa` from 0.895 to 0.985. [Multi-vector results](docs/BENCHMARKS.md#121-multi-vector-contextual-maxsim-retrieval-fast-pplx--fast_sharedretrieval_index)
- **Decision scoring.** Six-bucket multifunction `Laya-Typed-421M` (`s64..s512`, `[10538, 0, 0]` `CPU_AND_NE` with two-pass FP16 `LayerNorm` guards) runs single short questions in **`5.09 ms`** on contended M4 Pro (`196.6` q/s) and **`6.53 ms`** on Base M5 (`153.1` q/s), scoring **`711/1400` (`0.5079`)** on **both** Macs (`76.7 s` / `9.27` EOT / `1.59 GB` on quiescent Base M5 — `1.15×` faster than NMTZ Core ML — and `133.1 s` / `1.52 GB` on contended M4 Pro). Heterogeneous `laya-hetero` runs **`8x_512` in `172.7 ms`** (`46.3` q/s) and **`64x_mixed` in `578.9 ms`** (`110.6` q/s) on Base M5, cutting Tier-1 wall time to **`37.9 s` (`18.59` EOT)** on Base M5 and **`59.3 s` (`11.93` EOT, `0.5050`)** on M4 Pro. `Kev-0.8B` (`_MIN_LCP_GROUP_TOKENS=128` + sorted tails) finishes Base M5 Tier-1 in **`122.8 s` (`6.00` EOT, `0.5264` — `1.18×` Stock `mlx-lm`)** and hits **`27.65 q/s` (`2.971×` Stock)** on shared passages with `PrefixTree`. In a document-extraction pilot, adding Laya helped choose what a human reviewed; measure decision accuracy on the intended task. [Decision comparisons](docs/BENCHMARKS.md#decision-scoring) · [Wave 3 Laya & engine fixes](docs/BENCHMARKS.md#16-validated-engine-default--laya-fp16-layernorm-fixes-2026-11-24)
- **Structured extraction and long-context KV.** Using the same public Gemma 4 E2B checkpoint on M5, strict schema generation scored **0.946 field F1 with 150/150 valid**. Unconstrained project decoding scored 0.924 with 146 valid, and stock MLX scored 0.923 with 146 valid. On 150 tool requests, strict generation returned 150 usable responses (123 correct) versus 142 usable (125 correct) unconstrained. On full 300-item Category 4 (`LiquidAI 150 + BFCL v3 150`), schema-constrained `fast-qwen3` (4B) hits **`300/300` (`0.8955` `S_obj,e2e`, `0.881` EOT, `0` loops, `PASS`)** on Base M5 (`299/300`, `0.8930` on M4 Pro), and on M5 (`Apple10` NAX) `W4A8`/`W8A8` prefill lifts 9B EOT by `+4.6%` / `+8.3%`. At 16K context, rotated 4-bit KV caching (`QuaRot`) cuts KV memory 3.56× and decodes 2.13× faster than 8-bit KV at matching field F1. [Structured generation](docs/BENCHMARKS.md#structured-generation) · [Wave 3 validation](docs/BENCHMARKS.md#162-post-fix-fast-qwen3-fast-ocr-fast-embeddinggemma-2-vision--fast-pplx-results)
- **Document OCR.** With layout/low-conf triage and `OvisOCR2` pixel budgeting (`50K..150K` floor), verified Wave 3 Tier-1 runs achieve **`0.6155` on `olmOCR-bench(42)` (`42/42`, `+13.65 pp`)**, **`0.8935` TEDS on `FinTabNet-OTSL(30)` (`30/30`)**, and **`0.6574` on `OmniDocBench-EN(24)`** (all 3 `PASS`, `0` loops on M4 Pro), and **`0.6128` (`+21.34 pp`) / `0.8867` (`PASS`) / `0.6872` (`+10.52 pp`, `PASS`)** on Base M5. Async `B=1` confidence scoring delivers `0.90–0.91` `FinTabNet` error-detection AUROC at `<= 0%` overhead, and opt-in `region_routing` with digital-PDF table splicing speeds mixed prose+table PDFs **3.47×**. [OCR and NAX prefill validation](docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23) · [Wave 3 OCR results](docs/BENCHMARKS.md#162-post-fix-fast-qwen3-fast-ocr-fast-embeddinggemma-2-vision--fast-pplx-results)
- **Visual page retrieval.** With `140` soft tokens and `depth=1` CPU image prefetch, `fast-embeddinggemma-2` encodes 48 page images at **18.98 img/s on M5** and **10.89 img/s on M4 Pro** at exact `R@1 = 0.8542`, and hits **`18/18` (`1.0000` M5 / `0.9722` M4 Pro, `PASS`)** on `VisualRetrieval(18)` with **`0.0 GB` ANE footprint** (`enable_ane=None`). Under `max_pixels=786_432` and CPU prefetch, `fast-pplx` encodes visual pages at **8.90 img/s on M5** and **6.00 img/s on M4 Pro**, running **2.081×** faster across 164 multi-domain documents with zero retrieval loss. [Visual retrieval benchmarks](docs/BENCHMARKS.md#141-vision-retrieval-cpu-image-prefetch-and-multimodal-mtp)

## Hardware, training, and serving

The measurements cover an M4 Pro with 48 GB on macOS 26.6 and a base M5 with 24 GB on macOS 27.2.1. The base M5 encoded the matched corpora faster. The M4 Pro had the shorter warm BGE query time and enough memory for the 20B generator. See the full 2026-11-24 dual-Mac 4-suite baseline and post-fix rerun ledger for cross-silicon throughput, outcome gates, and contention notes. [Hardware details](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline) · [Dual-Mac 4-suite receipt](docs/BENCHMARKS.md#15-dual-mac-full-4-suite-verification-and-post-fix-reruns-2026-11-24)

ModernBERT’s training rate measures steps on generated inputs and random labels, with frozen-layer outputs already cached in the fastest run. For Gemma adapter training (QLoRA), the 48 GB M4 Pro completed a 2,048-token run. On the 24 GB M5, earlier 2,048- and 1,280-token attempts exhausted memory; packing 1,024 tokens with two gradient-accumulation steps completed cleanly within 4.85–6.76 GiB at matching loss and gradients. [Training measurements](docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)

The generation servers send the completed text in one content event. The published timings leave interactive time to first token and latency with concurrent HTTP clients open. [HTTP serving and timing boundaries](docs/BENCHMARKS.md#how-to-read-a-comparison)

## Developer commands

Preview the full setup as shown above. After staging weights and compiled models, run one GPU or Neural Engine suite at a time. The hardware benchmarks share `/tmp/m4_gpu_validation.lock`.

```bash
make test-hw           # Run the hardware suites in sequence
make sync              # Copy package sources from sibling checkouts
make sync-check        # Check the workspace copies against their sources
make sync-clean-check  # Also check the sibling worktrees are clean
make leak-check        # Check for private paths and identifiers
```

The benchmark guide includes commands for public tasks, batch throughput, training, and PPLX experiments. For hardware dispatch defaults, see `DynamicSiliconPolicy`. The optimizer plugin is available with `claude --plugin-dir ./packages/fast-shared`. [Benchmark commands](docs/BENCHMARKS.md#8-reproducible-cli-invocations) · [Hardware dispatch](packages/fast-shared/src/fast_shared/runtime_lessons.py)

## Prebuilt Hugging Face artifacts

Pre-compiled Core ML ANE bundles and pre-quantized MLX checkpoints eliminate local compilation and merge/quantization steps:

| Package | Prebuilt Hugging Face repo | Contents |
| --- | --- | --- |
| `fast-decision-models` | [`mikewolfd/laya-typed-decisions-421m-coreml-ane`](https://huggingface.co/mikewolfd/laya-typed-decisions-421m-coreml-ane) | 6-bucket multifunction `laya_typed_w8.mlmodelc` (`[10538, 0, 0]` Pure-ANE, FP16 LN guards) + FP16 weights + tokenizer |
| `fast-decision-models` | [`mikewolfd/kev-0.8b-mlx-w8`](https://huggingface.co/mikewolfd/kev-0.8b-mlx-w8) | Pre-merged 8-bit (`g64`) `model_w8.safetensors` (`801 MB`) + original Kev tokenizer |
| `fast-pplx` | [`mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane`](https://huggingface.co/mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane) | 4-stage `W8A16` balanced Core ML ANE bundles (`64..512`) + FP16 weights + mmap `embed_tokens` |
| `fast-embeddinggemma-2` | [`mikewolfd/embeddinggemma-2-coreml-ane`](https://huggingface.co/mikewolfd/embeddinggemma-2-coreml-ane) | 9 packed `W8A16` ANE bundles (`32..512`) + 3 CPU-SME bundles + FP16 weights + tokenizer |
| `fast-bge-small` | [`mikewolfd/bge-small-en-v1.5-coreml-ane`](https://huggingface.co/mikewolfd/bge-small-en-v1.5-coreml-ane) | 9 FP16 Pure-ANE buckets (`s16..s512`) + weights + tokenizer |

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License) inspired the unified-memory footprint methodology, the clean README layout across this workspace, and key design and evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity and fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
