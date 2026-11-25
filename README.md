# fast-silicon

Local Apple Silicon engines for bulk embeddings, document OCR, decision scoring, and schema-constrained JSON extraction. Built to replace cloud ML API bills at batch scale by running the Metal GPU and 16-core Apple Neural Engine (ANE) concurrently, with zero telemetry, no account, and 100% offline execution after weight download.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/throughput_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/throughput_light.svg">
  <img alt="fast-silicon batch throughput multiplier versus stock MLX on Base M5 and M4 Pro" src="docs/assets/throughput_light.svg">
</picture>

## Requirements

- **Supported:** Apple Silicon Macs (verified on 48 GB M4 Pro macOS 26.6 and 24 GB base M5 macOS 27.2.1), macOS 14+, Python 3.12+, and `uv`.
- **Unsupported:** Windows, Linux, and Intel Macs (ANE runners and Apple Vision require macOS Apple Silicon frameworks). Xcode Command Line Tools are needed only if rebuilding native `.dylib` runners; inference needs neither `torch` nor `transformers`.

## Quickstart (`fast-bge-small`, ~0.35 GB prebuilt download)

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-bge-small --inexact
.venv/bin/bge-small setup-weights
```

```python
from fast_bge_small import FastBgeSmall

passages = [
    "Ocean currents carry heat from the tropics toward the poles.",
    "A piano has 88 keys.",
]
engine = FastBgeSmall(default_mode="hetero")
query = engine.encode_texts("How does heat move through the ocean?", add_query_prefix=True)
vectors = engine.encode_texts(passages)
print(passages[int((vectors @ query).argmax())])
```

To install multiple packages or build tools, run `uv sync --extra dev --extra extract --extra build` and preview weight downloads (~35 GB across all 8 model packages) with `.venv/bin/python scripts/setup_all.py --dry-run`.

## How it works

Stock MLX uses the Metal GPU and leaves the 16-core ANE idle, while Core ML uses the ANE and leaves the GPU idle. Most local runners are also tuned for single-user `B=1` chat instead of offline batch ETL jobs that share the same system prompt and JSON schema across thousands of rows.

```text
  Variable-length batch (sorted by token count)
     ├── longest sequences ──▶ Metal GPU (packed varlen / custom kernels) ──┐
     │                                                                      ├──▶ merged outputs in input order
     └── shortest sequences ◀─ 16-core ANE (multifunction Core ML ring)  ───┘
```

Pair that dual-accelerator scheduler with zero-padding sequence packing, shared system-prompt KV caches (`PrefixTree` / `Hydragen v2`), INT8/binary vector indexes, and schema-enforced fast-forward decoding, and throughput scales cleanly inside unified memory.

## Packages

| Workload | Package | What you get | Prebuilt download |
| --- | --- | --- | --- |
| Search short English passages | [`fast-bge-small`](packages/fast-bge-small) | 384d vectors; **0.49 ms** warm query (M4 Pro), **200.6k tok/s** bulk (**11.3x** stock MLX). | ~0.35 GB |
| Index 2K passages & page images | [`fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | Up to 2,048 tokens (128-768d); **48.8k tok/s** (**4.3-4.8x** MLX, **3.3x** Core ML), **19.0 img/s** lazy vision. | ~0.29 GB |
| Contextual passage & 32K search | [`fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` + `MaxSim`; raises legal nDCG from 0.359 to 0.703 (**4.4x** faster than stock). | ~2.38 GB |
| Score decisions & rerank choices | [`fast-decision-models`](packages/fast-decision-models) | 100% Pure-ANE 421M `Laya` (**5.1 ms** query, 1,400 items in **37.9 s** hetero) + `Kev-0.8B/4B` with prefix cache. | ~0.95-4.90 GB |
| High-throughput JSON & tools (2B) | [`fast-gemma4`](packages/fast-gemma4) | Holds **128 rolling slots in 6.1 GB** MLX RAM at **10.5 valid docs/s** (**5.2x** `mlx-vlm` valid rate, **9.8x** CoreAI B=1). | ~10.40 GB |
| Reasoning, tools & 16K JSON (4B/9B) | [`fast-qwen3`](packages/fast-qwen3) | Hybrid GatedDeltaNet + 4-bit QuaRot KV (**2.1x** faster at 16K); **300/300** valid on 300 JSON/tool tasks. | ~3.10 GB |
| Extract Markdown & tables from PDFs | [`fast-ocr`](packages/fast-ocr) | Clean PDF bypass (**139-220 pp/s**), parallel Apple Vision ANE (**9.6 pp/s**), and `OvisOCR2-8bit` VLM (**1.5-3.5x**). | ~2.60 GB |
| Batch 20B MXFP4 MoE generation | [`fast-gpt-oss`](packages/fast-gpt-oss) | `gpt-oss-20b` auto-clamped across **16-48 GB+ Macs** (**103 tok/s** @ B=16 on 24 GB M5; **454 tok/s** @ B=128 on 48 GB). | ~12.10 GB |
| Shared runtime & compact index | [`fast-shared`](packages/fast-shared) | Dynamic chip dispatch, `CompactRetrievalIndex` (0.50 ms @ 100k vecs), `PrefixTree`, KV quant, and Parquet ETL. | - |

## Cross-silicon benchmarks (`M4 Pro 48 GB` vs. `Base M5 24 GB`)

All runs execute locally after warmup with deterministic accuracy verification on pinned public benchmarks. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for datasets, stock configs, hardware ceilings, and CLI reproducers.

| Workload (metric) | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Baseline comparison |
| --- | --- | --- | --- |
| **BGE-small retrieval (`325` queries)** | **200.6k tok/s** (11.8 s, 1.54 GB) | **187.2k tok/s** (12.2 s, 1.55 GB) | **11.3x** (M5) / **9.6x** (M4 Pro) stock MLX |
| **EmbeddingGemma-2 (`500` queries)** | **48.8k tok/s** (57.1 s, 3.43 GB) | **38.5k tok/s** (74.9 s, 2.49 GB) | **4.3-4.8x** MLX, **3.3x** Core ML, **6.2x** CoreAI |
| **Contextual PPLX (`325` queries)** | **0.7244 score** (136.5 s, 1.97 GB) | **0.7241 score** (167.3 s, 1.97 GB) | **4.4x** (M5) / **4.3x** (M4 Pro) `pplx-context-mlx` |
| **Laya-421M scoring (`1,400` items)** | **37.9 s** hetero / **76.7 s** ANE (1.59 GB) | **59.3 s** hetero / **133.1 s** ANE | **2.0x** MLX / **12-25x** CoreAI (`711/1400` ANE) |
| **Gemma 4 E2B extraction (`B=128`)** | **10.5 valid docs/s** (`1,479 tok/s`, 6.1 GB) | **8.8 valid docs/s** (ANE+GPU) | **5.2x** stock `mlx-vlm` valid rate (OOMs @ B=64) |
| **Qwen 3.5 4B JSON + tools (`300` tasks)** | **0.8955** (`300/300` valid, 4.76 GB) | **0.8930** (`299/300` valid, 4.99 GB) | **1.4-1.5x** scored throughput vs unconstrained MLX |
| **Document OCR (`96` pages, auto router)** | `0.613` olmOCR / `0.887` FinTab / `0.687` Omni | `0.616` olmOCR / `0.894` FinTab / `0.657` Omni | **1.5-2.0x** `OvisOCR2`; **3.5x** mixed PDF splice |
| **GPT-OSS-20B (`MXFP4`, batched)** | **103.1 tok/s** @ B=16 (`13.4-14.3 GB`) | **454.1 tok/s** @ B=128 (`14.8-16.7 GB`) | Auto-clamped waves fit 16-48 GB+ Macs cleanly |

- **Sub-millisecond vector search:** `CompactRetrievalIndex` pairs 1-bit ARM64 NEON Hamming prefiltering with symmetric INT8 reranking (`4-6x` smaller than FP32) to search 100,000 vectors in **0.50-0.66 ms/query** (`3.0-4.4x` faster than FP32 BLAS) with zero ranking loss ([index details](docs/BENCHMARKS.md#122-model-aware-mrl--int8--arm64-neon-binary-rerank-fast-bge-small-fast-embeddinggemma-2-fast-pplx)).
- **Local post-training:** Packed 8-bit `QLoRA` with shared system-prompt KV runs at `533-1,385 tok/s` on 24-48 GB Macs, and disaggregated `GRPO` runs `Gemma 4` rollouts on Metal GPU concurrently with `Laya` critic scoring on the ANE (`<0.4 ms` exposed critic time) ([training benchmarks](docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)).

## Prebuilt Hugging Face artifacts

Pre-compiled `.mlmodelc` Core ML bundles and pre-quantized MLX checkpoints skip local model compilation:

| Package | Prebuilt Hugging Face repo | Contents |
| --- | --- | --- |
| `fast-decision-models` | [`mikewolfd/laya-typed-decisions-421m-coreml-ane`](https://huggingface.co/mikewolfd/laya-typed-decisions-421m-coreml-ane) | 6-bucket multifunction `laya_typed_w8.mlmodelc` (100% Pure-ANE) + FP16 weights + tokenizer |
| `fast-decision-models` | [`mikewolfd/kev-0.8b-mlx-w8`](https://huggingface.co/mikewolfd/kev-0.8b-mlx-w8) | Pre-merged 8-bit (`g64`) `model_w8.safetensors` (`801 MB`) + original Kev tokenizer |
| `fast-pplx` | [`mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane`](https://huggingface.co/mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane) | 4-stage `W8A16` balanced ANE bundles (`64..512`) + FP16 weights + mmap `embed_tokens` |
| `fast-embeddinggemma-2` | [`mikewolfd/embeddinggemma-2-coreml-ane`](https://huggingface.co/mikewolfd/embeddinggemma-2-coreml-ane) | 9 packed `W8A16` ANE bundles (`32..512`) + 3 CPU-SME bundles + FP16 weights + tokenizer |
| `fast-bge-small` | [`mikewolfd/bge-small-en-v1.5-coreml-ane`](https://huggingface.co/mikewolfd/bge-small-en-v1.5-coreml-ane) | 9 FP16 Pure-ANE buckets (`s16..s512`) + weights + tokenizer |
| `fast-gemma4` | [`mikewolfd/gemma-4-e2b-it-early14-coreml-ane`](https://huggingface.co/mikewolfd/gemma-4-e2b-it-early14-coreml-ane) | `Early14` Core ML ANE prefill bundles (`fp16` + `w8`, `1,326` ANE ops) + `150K` vocab remap |

## Limitations & Non-Goals

- **Single-model RAM vs. standalone Core ML:** Running Metal GPU and ANE together consumes more unified memory than a standalone 8-bit Core ML export (for instance, `3.43 GB` vs. `0.84 GB` on EmbeddingGemma-2). Use `mode="ane"` for minimal RAM when latency allows.
- **Batch HTTP serving:** OpenAI-compatible HTTP servers collect microbatches and return completed responses rather than token-by-token SSE streams.
- **Large-batch 20B memory:** `fast-gpt-oss` runs cleanly on 24 GB Macs (`13.4-14.3 GB` peak, auto-clamped to `B=16`), while large `B=64-128` batch sweeps want 32-48 GB+ RAM.
- **Grammar lift vs. kernel speed:** On structured JSON benchmarks against unconstrained stock `mlx-lm`/`mlx-vlm`, `llguidance` grammar enforcement eliminates malformed JSON alongside kernel speedups; package READMEs report raw speed and validity separately.
- **Hardware coverage:** Verified continuously on M4 Pro and base M5; M1-M3 Macs work via the same MLX/Core ML paths but are not in our hardware CI matrix.

## Prior Art, Development & License

- **Compared to `mlx-lm` / `mlx-vlm` and `llama.cpp`:** Those excel at general interactive chat and broad GGUF/MLX model coverage; `fast-silicon` targets high-volume Apple Silicon batch pipelines with ANE offloading, shared-prefix KV reuse, and compact retrieval.
- **Compared to public Core ML exports & Apple CoreAI:** Standalone Core ML models save RAM on single models, while `fast-silicon` adds long-context GPU fallback (`2K-32K`), batch work-stealing, and unified Python/Parquet CLIs.
- **Acknowledgments:** [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) (MIT) inspired `fast-pplx` memmap embeddings, tile-aligned slicing, and parity testing. Built on `mlx`, `mlx-lm`, `coremltools`, `llguidance`, `pypdfium2`, and upstream weights from BAAI, Google, Qwen, Perplexity, OpenAI, and ConvAI.

Code is [MIT licensed](LICENSE) (downloaded upstream model weights remain subject to their original licenses). For contributor workflows, sibling repo syncing, and test suites, see [CONTRIBUTING.md](CONTRIBUTING.md).
