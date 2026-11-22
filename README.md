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
| Run hybrid `GatedDeltaNet` + `GQA` Qwen3.5/3.6 extraction or MoE | [Qwen 3.5 / 3.6](packages/fast-qwen3) | On 50 public extraction + tool tasks, `Qwen3.5-4B-OptiQ-4bit` scored **0.9024 (`50/50` valid)** vs stock `mlx-lm` `0.623–0.662` (`35–37/50` valid, **`1.38–1.53×` EOT**), and reached **191.0 tok/s** (`4.84 docs/s`) on M4 Pro and **171.5 tok/s** (`4.34 docs/s`) on the 24 GB M5 (`4.46 GB`). |
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
- **Decision scoring.** In a document-extraction pilot, adding Laya helped choose what a human reviewed. Adding it as an automatic verifier took the M5 pipeline from 93.3 to 160.7 seconds at the same held-out quality. On 250 four-way choices, project Laya picked the preferred response 73 times. Measure decision accuracy on the intended task. [Decision comparisons](docs/BENCHMARKS.md#decision-models) · [Review pilot and verifier](docs/benchmarks/review-20261121.md#decision-scoring)
- **Structured extraction.** Using the same public Gemma 4 E2B checkpoint on M5, strict generation scored **0.946 field F1 with 150/150 valid**. The unconstrained project scored 0.924 with 146 valid, and stock MLX scored 0.923 with 146 valid. On a disjoint 48-document M4 Pro sample both project settings scored 0.926. On 150 tool requests, strict generation made 123 correct decisions and returned 150 usable responses. The unconstrained project made 125 correct decisions and returned 142 usable responses. [Matched settings, scores and timings](docs/BENCHMARKS.md#structured-generation)
- **v2.3 Accuracy, Precision & Contextual Retrieval.** Header-anchored + sentence-snapped `parquet_pipeline` chunking lifted 175-query `LegalBench-RAG` `span_ndcg@10` from `0.3046 -> 0.4439` (`fast-bge-small`) and `0.3803 -> 0.5054` (`fast-embeddinggemma-2`, header-anchored); `fast-pplx` `repeat_header_tokens=128` lifted contextual `LegalBench-RAG` `0.4528 -> 0.7352` (Tier-1) / `0.5145 -> 0.6832` (Tier-2) and achieved `0.0` `slice-0` stream-vs-whole divergence on Base M5; `fast-shared` fixed truncated `finish_reason="length"` detection + async deferred retries (`LiquidAI` `0.3125 -> 1.0` schema valid, F1 `0.2352 -> 0.9114`), conditional `SWA=1024`, and multi-batch CoreAI calibration gates; `fast-gemma4` promoted FP32 softcap (`0/1024` mismatches), conservative 4-gram/cycle loop breaking, and `--hadamard-mode auto` (`w8` Early14 `cos_l14k 0.999031 -> 0.999363` M4 / `0.998742 -> 0.999076` M5, `CPU=0/GPU=0`, `ane_ffn` `W8A16` default); and `fast-gpt-oss` validated hybrid `f16+f32` precision and `canon_low` CoT budgets (`0.7263` MMLU, `0.54` MATH-500, `0.8467` BFCL) against `bf16`/`fp32`/`skip` alternatives. [Section 10 & negative gates](docs/BENCHMARKS.md#10-v23-cross-silicon-accuracy-precision--contextual-validation-2026-11-22) · [v2.3 validation JSON](docs/benchmarks/v23-validation-20261121.json)
- **`fast-qwen3` Hybrid `GatedDeltaNet` + `OptiQ` Engine (`v0.2`).** Adds monorepo-native `Qwen3.5-{0.8B..27B}` & `Qwen3.5/3.6-35B-A3B` support sharing `fast_shared.prefix_tree.PrefixTree`, `A4_v2` cycle/4-gram loop breaking, right-padded post-LCP ragged state invariance, prefill-folded `upfront_ff`, multi-token `llguidance` fast-forward coalescing, active-row `lm_head` slicing, and `R2` GDN `lora_accel`. On 50 Category-4 public tasks (`LiquidAI 25 + BFCL v3 25`), `fast-qwen3` achieved an identical **`0.9024` (`50/50` valid, `0.9248` field F1, `0.8800` AST, `0` loops)** on both Macs — beating stock `mlx-lm 0.32.2` by **`+24.1..+28.0 pp` and `1.38×–1.53×` EOT** — while reaching **`6.07×–8.33×`** `4B-OptiQ` (`171.5–191.0 tok/s`) and **`9.25×–11.66×`** `0.8B` (`573.4–657.3 tok/s`) `B=8` strict-JSON wave speedups. [Section 11](docs/BENCHMARKS.md#11-fast-qwen3-hybrid-gateddeltanet--optiq--lora-validation-2026-11-22) · [Qwen3 JSON](docs/benchmarks/fast-qwen3-20261122.json)

## Hardware, training and serving

The measurements cover an M4 Pro with 48 GB on macOS 26.6 and a base M5 with 24 GB on macOS 27.2.1. The base M5 encoded the matched corpora faster. The M4 Pro had the shorter warm BGE query time and enough memory for the 20B generator. [Hardware details](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline)

ModernBERT’s training rate measures steps on generated inputs and random labels, with frozen-layer outputs already cached in the fastest run. For Gemma adapter training (QLoRA), the M4 Pro completed a 2,048-token run. On the 24 GB Base M5, earlier 2,048- and 1,280-token attempts exhausted memory, whereas the sizing policy setting (`pack_tokens=1024, grad_accum=2, ckpt_tokens=512`, `19.2 GB` cap) completed with zero OOMs (`4.848 GiB` unpacked / `4.849 GiB` packed at `80.52` sup tok/s, `6.759 GiB` shared-KV at `81.63` sup tok/s; `loss_diff=0.0, grad_cos=1.0`). [Training measurements](docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo) · [v2.2 validation JSON](docs/benchmarks/v22-validation-20261121.json)

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
