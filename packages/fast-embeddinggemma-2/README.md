# fast-embeddinggemma-2

Encode up to 2,048 tokens per passage and choose an embedding width from 128 to 768 dimensions. Bulk encoding reached 38,517 tokens/s on M4 Pro and 48,803 on M5. On the public retrieval benchmark, the package delivered 4.33–4.80× the score per second of stock MLX and 6.24× that of Apple CoreAI on M5.

`fast-embeddinggemma-2` runs the 300M-parameter `google/embeddinggemma-2` text encoder on the Metal GPU and Apple Neural Engine (ANE) together for bulk encoding. A 24-token query on the ANE took 2.11 ms on M4 Pro and 2.58 ms on M5. The package includes a Python API, OCR-aware Parquet ingestion, an OpenAI-compatible `/v1/embeddings` endpoint with `dimensions` support, and a standalone `emb2-ane-cli`. Inference uses MLX and Core ML. Prebuilt Core ML ANE artifact: [`mikewolfd/embeddinggemma-2-coreml-ane`](https://huggingface.co/mikewolfd/embeddinggemma-2-coreml-ane) · [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md)

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/embeddinggemma-2 setup-weights
.venv/bin/embeddinggemma-2 verify
```

```python
from fast_embeddinggemma_2 import FastEmbeddingGemma2

eng = FastEmbeddingGemma2()
q = eng.encode_texts("quarterly semiconductor capex", is_query=True, add_query_prefix=True, dimensions=512)
docs = eng.encode_documents(["First long passage...", "Second passage..."], titles=["2025 Capex Note", "Summary"], dimensions=512)

# Build a 6x-smaller 512d INT8 / binary+INT8 search index (uses rerank_top_r=256 when binary):
index = eng.build_index(["First long passage...", "Second passage..."], titles=["2025 Capex Note", "Summary"])
top_ids, top_scores = index.search(q, k=5)

# Optional lazy image encoding (defaults to max_soft_tokens=140; install extra: `uv pip install -e ".[vision]"`).
# Text-only calls never import Pillow or load vision_tower weights (0 text overhead).
img_index = eng.build_image_index(["page_1.png", "page_2.png"])
page_hits, page_scores = img_index.search(eng.encode_texts("revenue table", is_query=True), k=1)
```

```bash
./build/emb2-ane-cli --ids "2,3250,236787,7293,1109,1816,236787,1" --seq-len 32 --bench 30 --json
.venv/bin/embeddinggemma-2 embed --dimensions 256 --query --query-prefix "quarterly semiconductor capex"
```

## Benchmark highlights

### 1. Retrieval quality & elapsed time (`NanoBEIR-en + LegalBench-RAG`, 500 items)

The retrieval score averages 150 NanoBEIR queries and 175 LegalBench-RAG queries evaluated twice, with short and long passages. Effective outcome throughput (EOT) divides the total score by elapsed seconds after warmup.

| Host | Engine | Mean score | EOT (score/s) | Wall time | Measured memory | Speedup vs. Stock MLX |
|---|---|---:|---:|---:|---:|---|
| **Base M5 (24 GB)** | **`fast-embeddinggemma-2` (`hetero`)** | **0.5376** | **4.707** | **57.1 s** | **3.43 GB** | **4.33× MLX · 6.24× CoreAI** |
| Base M5 | Public Core ML (FluidInference FP16) | 0.5186 | 1.666 | 155.7 s | 1.38 GB | 1.53× MLX |
| Base M5 | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.740 | 148.9 s | 0.84 GB | 1.60× MLX |
| Base M5 | Stock `mlx-vlm` (8-bit) | 0.5364 | 1.086 | 246.9 s | 19.41 GB | 1.00× MLX |
| Base M5 | Apple CoreAI (`.aimodel` BF16) | 0.5360 | 0.754 | 355.5 s | 3.04 GB (probe) | 0.69× MLX |
| **M4 Pro (48 GB)** | **`fast-embeddinggemma-2` (`hetero`)** | **0.5365** | **3.583** | **74.9 s** | **2.49 GB** | **4.80× MLX** |
| M4 Pro | Public Core ML (FluidInference FP16) | 0.5186 | 1.637 | 158.4 s | 1.53 GB | 2.19× MLX |
| M4 Pro | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.649 | 157.2 s | 1.03 GB | 2.21× MLX |
| M4 Pro | Stock `mlx-vlm` (8-bit) | 0.5353 | 0.746 | 358.6 s | 39.03 GB | 1.00× MLX |

Both public Core ML exports accept up to 512 tokens per input, whereas 136 of the 144 long LegalBench passages exceed 512 tokens. At 512 tokens, all three implementations ranked legal passages equally (0.516–0.518 whole-passage `nDCG@10`); raising `fast-embeddinggemma-2` to 2,048 tokens brought all annotated answer spans into view and lifted whole-passage `nDCG@10` to **0.569** (and visible character recall@10 from 0.336 to **0.903**). On M5, OpenInfinity INT8 used less memory (0.84 GB vs 3.43 GB), while `fast-embeddinggemma-2` finished scoring 2.6× faster (57.1 s vs 148.9 s). [512- vs 2,048-token legal retrieval analysis](../../docs/benchmarks/embeddinggemma-legal-512-20261121.json) · [Public Core ML details](../../docs/BENCHMARKS.md#public-core-ml-results)

Prepending document headers when chunking Parquet windows raised 175-query legal passage ranking from 0.380 to 0.505 and character recall from 0.721 to 0.811. Because EmbeddingGemma-2 is Matryoshka-trained, `eng.build_index(corpus)` defaults to 512-dimensional INT8 vectors (514 B/chunk, **6× smaller** than 768d FP32) with +0.0038 higher LegalBench ranking; at 16,000 or more documents, adding its 64-byte binary Hamming prefilter cuts single-query 100,000-document search from 2.23 ms to **0.50 ms** (4.43× faster). [Compact index receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 2. Stock `mlx-vlm` GPU vs. `fast-embeddinggemma-2` (`gpu` / `ane` / `hetero`)

| Host | Workload | Stock `mlx-vlm` 8-bit GPU | `fast-embeddinggemma-2` `gpu` | `fast-embeddinggemma-2` `ane` | **`fast-embeddinggemma-2` `hetero`** | Public Core ML (`OpenInfinity INT8` / `Fluid FP16`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | Warm single 24-tok query | `8.48 ms` | `5.02 ms` (`1.7×` stock) | **`2.58 ms`** (`3.3×` stock) | ANE-routed | `2.59 ms` / `4.33 ms` (`CoreAI 9.38 ms`) |
| **Base M5 (24 GB)** | Bulk text throughput | `64.91 docs/s` (Tier-1 `1.09` EOT) | `30,430 tok/s` | `21,888 tok/s` | **`49,106 tok/s`** (`48,803` 2K corpus) | `14,945` / `14,187 tok/s` (`CoreAI 12,482`) |
| **M4 Pro (48 GB)** | Warm single 24-tok query | `8.39 ms` | `7.72 ms` (`1.1×` stock) | **`2.08–2.11 ms`** (`4.0×` stock) | ANE-routed | `2.59 ms` / `2.70 ms` |
| **M4 Pro (48 GB)** | Bulk text throughput *(2026-11-24 contended)* | `46.05 docs/s` (Tier-1 `0.75` EOT) | `17,936 tok/s` | `21,445 tok/s` | **`36,809 tok/s`** (`38,517` 2K quiet) | `14,294` / `14,251 tok/s` |
| **M4 Pro (48 GB)** | Document OCR (2,000 docs, `hetero2`) | — | `11,844 tok/s` (`47.3 s`) | — | **`32,569 tok/s`** (`17.2 s`, `2.75×` GPU) | — |

Across a 64-document verification fixture (`embeddinggemma-2 verify`), minimum cosine similarity against the FP16 reference is `0.999998` on GPU, `0.999728` in `hetero2`, and `0.999887` on a single 24-token ANE query. [Dual-Mac receipt](../../docs/benchmarks/dual-mac-suite-20261124.json)

### 3. Optional lazy image encoding, CPU prefetch & cross-modal retrieval

With the optional `[vision]` extra (`uv pip install -e ".[vision]"`), `encode_images()` and `build_image_index()` load the shared `gemma4_vision` tower only when images are passed. Text-only calls never import `PIL` or load vision weights.

- **140-token default and background image prefetch:** Defaulting to `140` soft vision tokens (`70`, `140`, `280`, `560`, or `1120` supported) cut 48-page ViT latency on Base M5 from `100.69` to `44.87 ms`, lifting pre-decoded GPU throughput **2.18×** (`19.57 img/s`) and end-to-end disk throughput **1.936×** (`15.14 img/s`) at identical `R@1 = 0.8542`. Adding bounded `depth=1` Pillow decode/resize prefetch lifted 48-page disk throughput to **`18.98 img/s` on Base M5** (`1.276×`, `45.76 MB` lower peak physical memory) and **`10.89 img/s` on M4 Pro** (`1.164×`, `38.63 MB` lower peak physical memory) with `cos >= 0.9999993` and exact `R@1 = 0.8542`. [Vision prefetch benchmarks](../../docs/BENCHMARKS.md#141-vision-retrieval-cpu-image-prefetch-and-multimodal-mtp)
- **Stock `mlx-vlm` parity (8 Category-5 pages, M4 Pro):** Reaches `min_cos = 0.999819` in `bf16` (`3.39 img/s` at `B=1`, `2.91 img/s` at `B=8`, `1.59 GB` MLX peak vs. stock `2.24 GB`) and `min_cos = 0.999701` in `8bit` (`3.51 img/s` at `B=8`, `1.59 GB` MLX peak vs. stock `2.00 GB`).
- **Cross-modal text-to-page retrieval (16 Category-5 pages, M4 Pro):** Reaches **`15/16` (`93.75%`) `Recall@1` and `0.950` `MRR` at `4.36 img/s`** (`1.59 GB` MLX / `2.98 GB` RSS) with compact single-vector embeddings, encoding images **2.01×** faster than `FastPplx` (`14/16` `87.5%` `R@1`, `0.903` `MRR`, `2.17 img/s`, `1.85 GB` MLX / `3.51 GB` RSS). [Vision & OCR benchmarks](../../docs/BENCHMARKS.md#136-optional-gemma-4--embeddinggemma-2-vision--cross-modal-retrieval-m4-pro-verified)

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **Metal GPU (`Apple9` / `Apple10`)** | Native `mx.matmul` projections, fused GeGLU + Per-Layer Embedding kernels, fused QKV norm + RoPE, and cost-aware variable-length batch packing up to 2,048 tokens | [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py) |
| **Pure-ANE Core ML (`W8A16`)** | Nine packed sequence buckets (`32..512`, `2,288` ANE ops, `0` CPU/GPU ops) with a 4-slot async ring, NEON untiling, and Accelerate `cblas_sgemm` projection | [`native/emb2_coreml_runner.mm`](native/emb2_coreml_runner.mm), [`native/emb2_fast_io.cpp`](native/emb2_fast_io.cpp) |
| **CPU-SME (opt-in `hetero3`)** | Mixed FP16 matmul / FP32 norm+softmax Core ML graphs (`256, 384, 512`) for optional 3-way scheduling | [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

## CLI, Parquet & HTTP server

```bash
# Chunked Parquet ingestion (--mode hetero uses GPU + ANE; --header-col formats native "title: {header} | text: " prefixes)
.venv/bin/embeddinggemma-2 parquet \
  --input ocr-text.parquet --output embeddings/ --text-col body --header-col title \
  --mode hetero --dimensions 512 --document-prefix \
  --chunk --max-chunk-tokens 512 --overlap 64 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --drop-text-col --include-chunk-text

# OpenAI-compatible HTTP server (/v1/embeddings with optional "dimensions")
.venv/bin/embeddinggemma-2 serve --host 127.0.0.1 --port 8080 --dimensions 256

# Optional lazy image embedding CLI (requires `[vision]` extra)
.venv/bin/embeddinggemma-2 embed-images page_1.png page_2.png --batch-size 8 --json

# Rebuild ANE bundles (optional [ane-build] extra) and run tests
python scripts/build_ane_models.py --mode all
.venv/bin/pytest -v
```

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**. Rebuilding `.mlmodelc` bundles requires optional `coremltools >= 8.0`. Sequences exceeding 2,048 tokens are truncated to the model’s context limit.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
