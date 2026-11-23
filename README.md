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
| Index longer passages | [EmbeddingGemma-2](packages/fast-embeddinggemma-2) | Reads up to 2,048 tokens per passage and returns vectors with 128 to 768 dimensions. The two public Core ML exports tested read up to 512 tokens. |
| Find a passage that depends on surrounding text | [PPLX](packages/fast-pplx) | On 175 legal questions, encoding neighboring text raised the passage-ranking score from 0.359 to 0.703. Stock contextual MLX also reached 0.703. On the full 325-query M5 test, `fast-pplx` finished in 136.5 seconds and stock took 601.5. |
| Prioritize documents for review or score choices about a passage | [Decision models](packages/fast-decision-models) | In a 48-document pilot, adding Laya to rules surfaced 7 of 12 known defects in the first ten reviews. Rules alone found 4. Kev can reuse a passage across questions. |
| Extract fields or call tools | [Gemma 4](packages/fast-gemma4) | The strict extraction default recovered field values at 0.946 F1 on 150 multilingual forms. Every response passed its schema. |
| Document | Extract fields or call tools with Qwen 3.5 / 3.6 | [Qwen 3.5 / 3.6](packages/fast-qwen3) | On 50 extraction and tool tasks, schema-constrained 4B scored 0.902 with 50/50 valid responses; unconstrained stock `mlx-lm` scored 0.623–0.662 with 35–37 valid. Supports dense 0.8B–27B, 35B-A3B MoE, batched multimodal generation (`images=...`), and Apple10 NAX prefill. |
| OCR PDFs, scans, and financial tables | [OCR](packages/fast-ocr) | Routes clean PDF text (`139–220 pp/s`), zero-MLX parallel Apple Vision ANE (`workers=4`, `9.61 pp/s` = `1.592×` on 16 `FinTabNet` images), and `B=1..16` `OvisOCR2-8bit` / `NuExtract3` decoding (`1.22–1.66×` on M4 Pro and `1.53–2.02×` on M5 vs stock `mlx_vlm`, `2.23–3.66×` vs `GLM-OCR`/`LightOnOCR-2`). |
| Generate structured records in large batches | [GPT-OSS](packages/fast-gpt-oss) | The 20B model handled 128 fixture requests at once on the 48 GB M4 Pro. The benchmark’s memory policy excluded the 24 GB M5. |
| Build a resumable document job or serve a model over HTTP | [Shared runtime](packages/fast-shared) | Model setup, compact vector indexing, batch processing, HTTP serving and memory measurement for the model packages. |

## Benchmark highlights

All implementations run locally. **Stock MLX** means the named public model and MLX library used in that test. A **public Core ML export** is a publisher’s preconverted model run with Apple’s Core ML framework. The `fast-*` packages can also use MLX and Core ML. **Apple CoreAI** is the separate runtime for `.aimodel` files. [Implementations and model names](docs/BENCHMARKS.md#which-implementation-ran)

| EmbeddingGemma workload on the 24 GB M5 | `fast-embeddinggemma-2` | OpenInfinity INT8 in Apple Core ML |
| --- | ---: | ---: |
| One warm 24-token query | 2.58 ms | 2.59 ms |
| Bulk encoding, same 2,000 documents | 48,803 tokens/s | 14,945 tokens/s |
| Combined memory, separate 500-item retrieval test | 3.43 GB | 0.84 GB |

The first two rows time token IDs through the returned vector. Tokenization, network calls and writing vectors to a search index fall outside those timings. For an app that starts the model on demand, comparable first-response times still need measurement. [Matched inputs and timers](docs/BENCHMARKS.md#warm-query-corpus-and-packing) · [Retrieval scores and memory](docs/BENCHMARKS.md#task-quality-and-elapsed-time) · [First use](docs/BENCHMARKS.md#first-use-and-reproduction)

- **Retrieval and compact indexing.** Across the M4 Pro and M5, BGE delivered 9.58–11.26× the ranking score per second of stock MLX, EmbeddingGemma 4.33–4.80×, and PPLX 4.33–4.41×. Prepending document headers when chunking Parquet files (`--header-col`) raised 175-query legal passage ranking from 0.305 to 0.444 on BGE, 0.380 to 0.505 on EmbeddingGemma, and 0.453 to 0.735 on contextual PPLX. All three encoders share `CompactRetrievalIndex`, which pairs a 1-bit Hamming prefilter (`rerank_top_r=256` default in `search()`) with INT8 reranking to search 100,000 vectors in 0.50–0.66 ms per query (`top64`, 2.95–4.43× faster than FP32 BLAS) without measurable ranking loss. [Retrieval scores, times and memory](docs/BENCHMARKS.md#retrieval) · [Compact index modes](docs/BENCHMARKS.md#122-model-aware-mrl--int8--arm64-neon-binary-rerank-fast-bge-small-fast-embeddinggemma-2-fast-pplx)
- **Long-document search (4K–32K).** Splitting 32,768-token documents into 2,048-token contextual windows and scoring the best-matching window (`MaxSim`) ran 2.31× faster on the GPU and used 2.48× less peak Metal memory than a single 32K pass in `fast-pplx`. On `LongEmbed`, that raised 32K passkey ranking from 0.047 to 1.000 on the 100-document split (0.674 when all four length splits are pooled together) and `2wikimqa` from 0.895 to 0.985. [Multi-vector results](docs/BENCHMARKS.md#121-multi-vector-contextual-maxsim-retrieval-fast-pplx--fast_sharedretrieval_index)
- **Decision scoring.** In a document-extraction pilot, adding Laya helped choose what a human reviewed. Adding it as an automatic verifier took the M5 pipeline from 93.3 to 160.7 seconds at the same held-out quality. On 250 four-way choices, project Laya picked the preferred response 73 times. Measure decision accuracy on the intended task. [Decision comparisons](docs/BENCHMARKS.md#decision-models) · [Review pilot and verifier](docs/benchmarks/review-20261121.md#decision-scoring)
- **Structured extraction and long-context KV.** Using the same public Gemma 4 E2B checkpoint on M5, strict schema generation scored **0.946 field F1 with 150/150 valid**. Unconstrained project decoding scored 0.924 with 146 valid, and stock MLX scored 0.923 with 146 valid. On 150 tool requests, strict generation returned 150 usable responses (123 correct) versus 142 usable (125 correct) unconstrained. On 50 public extraction and tool tasks, schema-constrained `fast-qwen3` (4B) scored 0.902 with 50/50 valid responses on both Macs, delivering 1.38–1.53× the score per second of unconstrained stock `mlx-lm`. On M5 (`Apple10` NAX), `fast-qwen3`'s default `W4A8` prefill lifts 50-task 9B EOT by +4.6% with zero score change (+8.3% with opt-in `W8A8`). At 16K context, rotated 4-bit KV caching (`QuaRot`) cuts KV memory 3.56× and decodes 2.13× faster than 8-bit KV at matching field F1. [Structured generation](docs/BENCHMARKS.md#structured-generation) · [Qwen 3.5 and QuaRot KV](docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22)
- **Document OCR and visual page embeddings.** Forced to VLM-only mode (`--ocr-backend ovis --ocr-render-only`), `fast-ocr` scored **0.844 on `olmOCR-bench(4)` (4/4 valid)**, **0.869 TEDS on `FinTabNet-OTSL(2)`**, and **0.909 on `OmniDocBench-EN(2)`** across both Macs—running **1.22–1.51× faster on M4 Pro** and **1.53–2.02× faster on M5** than stock `mlx_vlm OvisOCR2-8bit`, and **2.23–3.66× faster** than stock `GLM-OCR` and `LightOnOCR-2` on M5 (`2.35–5.00×` EOT). Under strict GFM table grammar (`FinTabNet`), grammar-checked `OCRPLDDrafter` delivered **1.85×** decode speedup on M4 Pro (`n=6`) and **1.144×** wall speedup on M5 (`n=2`) with 100% text parity. Sharing the same Qwen3.5 ViT lazily, `fast-pplx` encodes PDF page images at `100 DPI` in **109.2 ms/page on M5** and **159.1 ms/page on M4 Pro** (3.60–5.25× faster than PyTorch MPS). [OCR and NAX prefill validation](docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23)

## Hardware, training and serving

The measurements cover an M4 Pro with 48 GB on macOS 26.6 and a base M5 with 24 GB on macOS 27.2.1. The base M5 encoded the matched corpora faster. The M4 Pro had the shorter warm BGE query time and enough memory for the 20B generator. [Hardware details](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline)

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

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspiring the unified-memory footprint methodology, the clean README layout across this workspace, and key design and evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity and fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
