# fast-embeddinggemma-2

Encode up to 2,048 tokens per passage (128-768 Matryoshka dimensions) or embed document page images lazily with `google/embeddinggemma-2`. Splits bulk encoding across the Metal GPU and 16-core Apple Neural Engine (`48,803 tok/s` on M5: `4.3-4.8x` stock MLX, `3.3x` public Core ML, `6.2x` Apple CoreAI), with `2.1-2.6 ms` warm single queries.

Zero telemetry, no account, and runs 100% offline after initial ~0.30-0.65 GB prebuilt download (text + ANE). Prebuilt Core ML ANE artifact: [`mikewolfd/embeddinggemma-2-coreml-ane`](https://huggingface.co/mikewolfd/embeddinggemma-2-coreml-ane) · [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 15+ (required for Core ML ANE buckets and native runner; `macOS 14+` for GPU-only `default_mode="gpu"`), Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported. Sequences beyond 2,048 tokens are truncated. Xcode Command Line Tools are required to compile the native ANE `.dylib` runner (`make -C packages/fast-embeddinggemma-2 native`, optional when passing `default_mode="gpu"`).

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-embeddinggemma-2 --inexact
make -C packages/fast-embeddinggemma-2 native

.venv/bin/embeddinggemma-2 setup-weights   # Downloads ~0.30-0.65 GB prebuilt weights + Core ML ANE buckets
.venv/bin/embeddinggemma-2 verify
```

```python
from fast_embeddinggemma_2 import FastEmbeddingGemma2

eng = FastEmbeddingGemma2()
q = eng.encode_texts("quarterly semiconductor capex", is_query=True, add_query_prefix=True, dimensions=512)
docs = eng.encode_documents(["First long passage...", "Second passage..."], titles=["2025 Capex Note", "Summary"], dimensions=512)

# Build a 6x-smaller 512d INT8 / binary+INT8 search index (reranks top 256 when binary):
index = eng.build_index(["First long passage...", "Second passage..."], titles=["2025 Capex Note", "Summary"])
top_ids, top_scores = index.search(q, k=5)

# Optional lazy image encoding (140 soft tokens default; install `[vision]` extra; 0 text-side RAM overhead):
img_index = eng.build_image_index(["page_1.png", "page_2.png"])
page_hits, page_scores = img_index.search(eng.encode_texts("revenue table", is_query=True), k=1)
```

## Benchmark highlights

### 1. Retrieval quality & elapsed time (`NanoBEIR-en + LegalBench-RAG`, 500 items)

Mean score averages 150 NanoBEIR queries and 175 LegalBench-RAG queries across short and long splits; score/s divides total score by elapsed seconds after warmup.

| Host | Engine | Mean score | Score/s | Wall time | Measured memory | Speedup vs. Stock MLX |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Base M5 (24 GB)** | **`fast-embeddinggemma-2` (`hetero`)** | **0.5376** | **4.707** | **57.1 s** | **3.43 GB** | **4.33x MLX / 6.24x CoreAI** |
| Base M5 | Public Core ML (FluidInference FP16) | 0.5186 | 1.666 | 155.7 s | 1.38 GB | 1.53x MLX |
| Base M5 | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.740 | 148.9 s | 0.84 GB | 1.60x MLX |
| Base M5 | Stock `mlx-vlm` (8-bit) | 0.5364 | 1.086 | 246.9 s | 19.41 GB | 1.00x MLX |
| Base M5 | Apple CoreAI (`.aimodel` BF16) | 0.5360 | 0.754 | 355.5 s | 3.04 GB (probe) | 0.69x MLX |
| **M4 Pro (48 GB)** | **`fast-embeddinggemma-2` (`hetero`)** | **0.5365** | **3.583** | **74.9 s** | **2.49 GB** | **4.80x MLX** |
| M4 Pro | Public Core ML (FluidInference FP16) | 0.5186 | 1.637 | 158.4 s | 1.53 GB | 2.19x MLX |
| M4 Pro | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.649 | 157.2 s | 1.03 GB | 2.21x MLX |
| M4 Pro | Stock `mlx-vlm` (8-bit) | 0.5353 | 0.746 | 358.6 s | 39.03 GB | 1.00x MLX |

*Memory note: Stock `mlx-vlm` reports `19.41 GB` (M5) and `39.03 GB` (M4 Pro) because it leaves MLX's Metal buffer cache uncapped up to macOS's 80% wired ceiling, whereas `fast-embeddinggemma-2` sets `mlx_cache_limit_mb=512`.*

Both public Core ML exports clip at 512 tokens, whereas 136 of the 144 long LegalBench passages exceed 512 tokens. At 512 tokens all three score identically (`0.516-0.518` whole-passage `nDCG@10`); supporting 2,048 tokens brings every answer span into view and lifts `nDCG@10` to **0.569** (`0.336 -> 0.903` visible character recall@10). On M5, OpenInfinity INT8 uses less RAM (`0.84 GB` vs. `3.43 GB`), while `fast-embeddinggemma-2` finishes 2.6x faster (`57.1 s` vs. `148.9 s`). [Legal 512 vs 2048 analysis](../../docs/benchmarks/embeddinggemma-legal-512-20261121.json) · [Public Core ML details](../../docs/BENCHMARKS.md#public-core-ml-results)

Prepending document titles raises 175-query legal passage ranking from `0.380` to `0.505` (`0.721 -> 0.811` recall). `eng.build_index(corpus)` defaults to 512d INT8 (`514 B/chunk`, 6x smaller than 768d FP32) and adds a 64-byte binary Hamming prefilter at 16,000+ docs, searching 100,000 vectors in **0.50 ms** (4.43x faster than FP32 BLAS). [Compact index receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 2. Stock `mlx-vlm` GPU vs. `fast-embeddinggemma-2` (`gpu` / `ane` / `hetero`)

| Host | Workload | Stock `mlx-vlm` 8-bit | `fast-embeddinggemma-2` `gpu` | `fast-embeddinggemma-2` `ane` | **`fast-embeddinggemma-2` `hetero`** | Public Core ML (`OpenInfinity` / `Fluid`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | Warm single 24-tok query | `8.48 ms` | `5.02 ms` (1.7x) | **`2.58 ms`** (3.3x) | ANE-routed | `2.59 ms` / `4.33 ms` (`CoreAI 9.38 ms`) |
| **Base M5 (24 GB)** | Bulk 2K text corpus | `64.91 docs/s` (text input) | `30,430 tok/s` | `21,888 tok/s` | **`48,803 tok/s`** | `14,945` / `14,187 tok/s` (`CoreAI 12,482`) |
| **M4 Pro (48 GB)** | Warm single 24-tok query | `8.39 ms` | `7.72 ms` (1.1x) | **`2.08-2.11 ms`** (4.0x) | ANE-routed | `2.59 ms` / `2.70 ms` |
| **M4 Pro (48 GB)** | Bulk 2K text corpus | `46.05 docs/s` (text input) | `17,936 tok/s` | `21,445 tok/s` | **`38,517 tok/s`** | `14,294` / `14,251 tok/s` |
| **M4 Pro (48 GB)** | 2,000 OCR documents (`hetero2`) | - | `11,844 tok/s` (`47.3 s`) | - | **`32,569 tok/s`** (`17.2 s`, 2.75x GPU) | - |

*Measurement note: `fast-embeddinggemma-2` and public Core ML/CoreAI bulk probes ran on matched pretokenized IDs (`tok/s`), whereas the stock `mlx-vlm` bulk probe started from raw text (`docs/s`); for apples-to-apples timed runs on identical documents, see Table 1 above (`4.33x` M5, `4.80x` M4 Pro).*

Verification (`embeddinggemma-2 verify`, 64 docs) hits minimum cosine similarity `0.999998` (`gpu`), `0.999728` (`hetero2`), and `0.999887` (24-tok `ane`) vs. FP16. [Dual-Mac receipt](../../docs/benchmarks/dual-mac-suite-20261124.json)

### 3. Optional lazy image encoding & cross-modal retrieval

Install with `uv sync --python 3.13 --package fast-embeddinggemma-2 --extra vision --inexact`; `encode_images()` and `build_image_index()` load `gemma4_vision` only when images are passed (`0.0 GB` ANE or vision overhead on text calls).

| Workload | Base M5 (24 GB) | M4 Pro (48 GB) | Quality / parity notes |
| --- | --- | --- | --- |
| **48 scanned pages (`140` soft tok + async prefetch)** | **18.98 img/s** (1.28x unprefetched) | **10.89 img/s** (1.16x unprefetched) | Exact `R@1 = 0.8542`, `cos >= 0.9999993` |
| **18-page visual retrieval benchmark** | **15.06 img/s** (`18/18` = `1.000`, `3.56 GB`) | **11.63 img/s** (`18/18` = `0.972`, `3.53 GB`) | `0.0 GB` ANE footprint (`enable_ane=None`) |
| **8-page parity vs. `mlx-vlm` (`bf16` / `8-bit`)** | - | `3.39` (`B=1`) / `3.51 img/s` (`B=8`, `1.59 GB` MLX) | `min_cos = 0.9998` (`bf16`) / `0.9997` (`8-bit`), `-0.41-0.65 GB` MLX peak vs stock |
| **16-page text-to-page cross-modal** | - | **4.36 img/s** (**93.8% R@1**, `0.950 MRR`) | Single-vector embeddings; **2.01x** faster image encode than `FastPplx` |

## What keeps it fast

| Component | How it works | Source |
| --- | --- | --- |
| **Metal GPU** | Native `mx.matmul` projections, fused GeGLU + Per-Layer Embedding kernels, fused QKV norm + RoPE, Apple10 `force_fused` SWA (`1.73-1.93x` at `seq_l >= 128`, `-193` to `-301 MB` MLX), and variable-length packing to 2,048 tokens | [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py) |
| **Pure-ANE Core ML (`W8A16`)** | Nine packed sequence buckets (`32..512`, `2,288` ANE ops, `0` CPU/GPU ops), 4-slot async ring, NEON untiling, and Accelerate `cblas_sgemm` projection | [`native/emb2_coreml_runner.mm`](native/emb2_coreml_runner.mm), [`native/emb2_fast_io.cpp`](native/emb2_fast_io.cpp) |
| **CPU-SME (opt-in `hetero3`)** | Mixed FP16 matmul / FP32 norm+softmax Core ML graphs (`256, 384, 512`) for standalone CPU/3-way runs | [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

## CLI, Parquet & HTTP server

```bash
.venv/bin/embeddinggemma-2 parquet \
  --input ocr-text.parquet --output embeddings/ --text-col body --header-col title \
  --mode hetero --dimensions 512 --document-prefix \
  --chunk --max-chunk-tokens 512 --overlap 64 --drop-text-col --include-chunk-text
.venv/bin/embeddinggemma-2 serve --host 127.0.0.1 --port 8080 --dimensions 256
.venv/bin/embeddinggemma-2 embed-images page_1.png page_2.png --batch-size 8 --json
.venv/bin/pytest -v packages/fast-embeddinggemma-2/tests   # Requires --extra dev
```

## Limitations, Prior Art & License

- **Limitations:** Text sequences past 2,048 tokens are truncated. Dual GPU+ANE `hetero` trades higher unified RAM (`2.49-3.43 GB`) for bulk throughput vs. OpenInfinity's 512-tok Core ML export (`0.84-1.03 GB`); pass `default_mode="ane"` if RAM is constrained. Vision encoding uses GPU+CPU only (`[vision]` extra).
- **Prior art:** Compared directly against `mlx-vlm`, `OpenInfinity/embeddinggemma-2-coreml-int8`, `FluidInference`, and Apple CoreAI. Use [`fast-bge-small`](../fast-bge-small) for ultra-fast 384d English-only search or [`fast-pplx`](../fast-pplx) for contextual 4K-32K documents.
- **License:** Package code is [MIT licensed](LICENSE); `google/embeddinggemma-2` weights carry the Apache-2.0 license. Verified on 48 GB M4 Pro and 24 GB Base M5.
