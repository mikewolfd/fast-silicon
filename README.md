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
| Score choices about a passage | [Decision models](packages/fast-decision-models) | Laya can leave the GPU free for another model. Kev can reuse a passage when answering several questions about it. |
| Extract fields or call tools | [Gemma 4](packages/fast-gemma4) | Constrain output to a schema and inspect each response’s `valid_schema` and `finish_reason`. The field and tool-call results below show how often the values were right. |
| Generate structured records in large batches | [GPT-OSS](packages/fast-gpt-oss) | The 20B model handled 128 fixture requests at once on the 48 GB M4 Pro. The benchmark’s memory policy excluded the 24 GB M5. |
| Build a resumable document job or serve a model over HTTP | [Shared runtime](packages/fast-shared) | Model setup, batch processing, HTTP serving and memory measurement for the model packages. |

## Benchmark highlights

All implementations run locally. **Stock MLX** means the named public model and MLX library used in that test. A **public Core ML export** is a publisher’s preconverted model run with Apple’s Core ML framework. The `fast-*` packages can also use MLX and Core ML. **Apple CoreAI** is the separate runtime for `.aimodel` files. [Implementations and model names](docs/BENCHMARKS.md#which-implementation-ran)

| EmbeddingGemma workload on the 24 GB M5 | `fast-embeddinggemma-2` | OpenInfinity INT8 in Apple Core ML |
| --- | ---: | ---: |
| One warm 24-token query | 2.58 ms | 2.59 ms |
| Bulk encoding, same 2,000 documents | 48,803 tokens/s | 14,945 tokens/s |
| Combined memory, separate 500-item retrieval test | 3.43 GB | 0.84 GB |

The first two rows time token IDs through the returned vector. Tokenization, network calls and writing vectors to a search index fall outside those timings. For an app that starts the model on demand, comparable first-response times still need measurement. [Matched inputs and timers](docs/BENCHMARKS.md#warm-query-corpus-and-packing) · [Retrieval scores and memory](docs/BENCHMARKS.md#task-quality-and-elapsed-time) · [First use](docs/BENCHMARKS.md#first-use-and-reproduction)

- **Retrieval.** Across the M4 Pro and M5, BGE delivered 9.58–11.26× the ranking score per second of stock MLX, EmbeddingGemma 4.33–4.80×, and PPLX 4.33–4.41×. This measure divides the sum of query scores by the time to index, query and score that workload. [Retrieval scores, times and memory](docs/BENCHMARKS.md#retrieval)
- **Decision scoring.** On 1,400 mixed decisions on M5, NMTZ’s public Core ML Laya finished in 88.4 seconds and used 2.73 GB. Laya in `fast-decision-models` took 114.8 seconds, used 1.00 GB, and scored slightly higher (0.5050 versus 0.4993). For Kev on the same workload, stock MLX finished first at 145.0 seconds, followed by `fast-decision-models` at 167.2 and public Core ML at 321.7. [Decision scores and timings](docs/BENCHMARKS.md#task-quality-and-elapsed-time)
- **Structured extraction and tool calls.** On M5, `fast-gemma4` produced 299 valid responses in 82.7 seconds across 300 tasks. Stock MLX produced 286 in 172.2 seconds. Accuracy depended on the task. Field F1, which balances missed fields and incorrect values, was **0.691 versus 0.923** on 150 extractions. Tool-call accuracy was **0.807 versus 0.800** on 150 calls. The runs used 16.94 and 5.58 GB, respectively. [Quality, validity and capacity measurements](docs/BENCHMARKS.md#structured-generation)

## Hardware, training and serving

The measurements cover an M4 Pro with 48 GB on macOS 26.6 and a base M5 with 24 GB on macOS 27.2.1. The base M5 encoded the matched corpora faster. The M4 Pro had the shorter warm BGE query time and enough memory for the 20B generator. [Hardware details](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline)

ModernBERT’s training rate measures steps on generated inputs and random labels, with frozen-layer outputs already cached in the fastest run. For Gemma adapter training (QLoRA), the M4 Pro completed a 2,048-token run. The M5 exhausted memory at 2,048 and 1,280 tokens. Its sizing policy suggests 1,024 tokens, a setting that still needs a completed run. [Training measurements](docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)

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

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspiring the unified-memory footprint methodology, the clean README layout across this workspace, and key design & evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity / bit-identity / fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
