# fast-pplx

Use `fast-pplx` to find a passage whose meaning depends on the surrounding text. On 175 legal questions, encoding each passage with its neighbors raised the score for ranking relevant passages in the first ten results from **0.359 to 0.703**. Stock contextual MLX also scored **0.703**. On the full 325-query base M5 workload, the project finished in **136.5 seconds** and stock took **601.5 seconds**. Both returned usable results for all 325 queries. [Scored retrieval runs](../../docs/benchmarks/document-workloads.json)

`fast-pplx` runs Perplexity’s `pplx-embed-v2-context-0.6b` model locally. Its mixed mode (`hetero2`) sends short inputs to the Apple Neural Engine (ANE) and long inputs to the GPU. The M4 Pro GPU processed a 32,768-token input with a measured process footprint of 2.69 GB. The package includes a Python API, an OpenAI-compatible `/v1/embeddings` server, resumable contextual Parquet ingestion, and a native ANE command-line tool. [Full benchmark tables](../../docs/BENCHMARKS.md)

## Quick start

```bash
make -C packages/fast-pplx native
fast-pplx setup-weights --local-files-only
# Optional: compile Pure-ANE Core ML stages (`balanced` profile by default)
fast-pplx setup-ane --precision w8a16
```

```python
from fast_pplx import FastPplx

with FastPplx(mode="auto", dimensions=512) as model:  # picks hetero2 if ANE stages exist, else gpu
    q = model.encode_queries(["How do ocean currents transport heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ], max_length=512)

    # For long 4K-32K documents, build a contextual multi-vector MaxSim index (1024d INT8):
    long_index = model.build_index(
        ["Long 32K document text...", "Second long document..."],
        multivector=True,
    )
    top_doc_ids, top_scores = long_index.search(model.encode_queries(["ocean currents"])[0], k=5)
```

`max_length` caps the whole document, including its prefix and chunk separators. `encode_contextual_chunks` returns one matrix per input document and one row per original chunk, in input order. A chunk crossing the cap uses the tokens that fit, and a chunk wholly beyond it gets a zero vector. To index every passage in a longer document automatically, use `fast-pplx parquet --context-tokens 2048` or `model.build_index(..., multivector=True)`.

## CLI, Parquet & HTTP server

```bash
# Single or batch embeddings (--mode auto|gpu|ane|hetero2, --dimensions 128..1024, --dtype int8|float16|float32)
fast-pplx embed "First passage" "Second passage" --dimensions 1024 --dtype int8 --json

# Contextual or flat chunked Parquet ingestion (--header-col applies to both modes; repeats up to 128 header tokens when --context-tokens >= 512)
fast-pplx parquet --input docs.parquet --output out_shards/ --text-col body --header-col title \
  --mode hetero2 --chunk-tokens 256 --context-tokens 2048 --dtype int8 --resume

# OpenAI-compatible HTTP server (/v1/embeddings; optional --quantize-weights w8 saves ~214 MB of Metal RAM)
fast-pplx serve --host 127.0.0.1 --port 8080 --mode hetero2 --quantize-weights w8

# Native Pure-ANE stage inspector and microbenchmark
./packages/fast-pplx/build/pplx-ane-cli --seq-len 128 --inspect --bench 20

# Direct visual page / PDF image embeddings & index search via shared Qwen3.5 ViT + fractional MRoPE
fast-pplx embed-images page.png document.pdf --chunks 8 --crop-margins --rotate-cw 0 --dtype int8 --output img_vecs.npz
fast-pplx embed-images page1.png page2.png --chunks 8 --crop-margins --build-index --query "search question"
```

## Choosing modes & index settings

| Setting | Recommendation | Why |
| :--- | :--- | :--- |
| `mode="auto"` (`hetero2`) | Default when ANE bundles are staged | Routes single short `B=1` queries directly to Metal GPU on M5+ (`Apple10` NAX) and ANE on M4 (`Apple9`); for batches, packs `<= 512` inputs into Pure-ANE buckets using first-fit-decreasing bin packing (`-60%` pad waste) while GPU handles longer windows (`19,443 tok/s` on M5, `17,793 tok/s` on M4 Pro). |
| `[D] ` prefix + `tanh` space | Default for all document paths | Prepends `[D] ` while excluding its tokens from pooling (matching upstream non-contextual & contextual behavior), uses `round(tanh(x) * 127)` output space by default (pass `apply_tanh=False` for raw linear projections), and defaults `--context-tokens` chunking to non-overlapping (`overlap=0`). |
| `--ane-profile balanced` | Default ANE build profile | Closes 73%+ of the ANE-to-GPU cosine gap (`>= 0.999` mean cosine) with negligible throughput cost versus `--ane-profile fast`. Across 325 retrieval queries, `hetero2` cuts wall time 1.71–1.75× on M4 Pro and 1.28–1.34× on M5 versus GPU alone. |
| `--context-tokens 2048` | Default for multi-chunk documents | Encodes 256-token passages inside 2,048-token windows and repeats up to 128 header tokens per window, raising `LegalBench-RAG` ranking from 0.453 to 0.735 (Tier-1) and 0.515 to 0.683 (Tier-2) with zero extra memory. |
| `dimensions=1024` + `INT8` | Default for chunk indexes | Unlike MRL-trained encoders, `pplx-embed-v2-context-0.6b` loses 3.7 points on legal retrieval when truncated to 512 dimensions. Keeping all 1,024 dimensions in symmetric INT8 (`1,026 B/vec`, 4× smaller) is lossless (`-0.0007` NanoBEIR, `-0.0012` LegalBench). |
| `multivector=True` | Long 4K–32K whole-document search | Replaces `O(32K^2)` single-pass attention with 2,048-token contextual windows scored via `MaxSim` (`15.97 KiB/doc` in 1024d INT8). |

## Benchmark highlights

### 1. Short & contextual passage retrieval (`NanoBEIR-en + LegalBench-RAG`, 325 queries)

| Implementation | Base M5 score | Base M5 score/s (wall, mem) | M4 Pro score | M4 Pro score/s (wall, mem) | vs. Stock MLX (M5 / M4 Pro) |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **`fast-pplx` (`hetero2`)** | **0.7244** | **1.724** (136.5 s, 2.34 GB) | **0.7241** | **1.407** (167.3 s, 1.97 GB) | **4.41×** / **4.33×** |
| Stock `pplx-context-mlx` (FP16) | 0.7236 | 0.391 (601.5 s, 2.12 GB) | 0.7236 | 0.325 (723.1 s, 2.19 GB) | 1.00× |

The selected M5 run split oversized documents into windows so all 325 queries had usable results. Its mean score exceeded stock MLX by 0.0008, with a paired 95% interval of −0.0036 to +0.0053. Batching accounts for part of the whole-job speedup: sorting 384 sampled passages by length cut stock MLX’s encoding time on M4 Pro from 20.65 to 13.34 seconds (1.55×) without changing any top-ten results. [Per-task results](../../docs/benchmarks/document-workloads.json) · [Stock padding probe](../../docs/benchmarks/pplx-padding-20261121.json) · [ANE profile & CoreAI tables](../../docs/BENCHMARKS.md#63-fast-pplx-ane-profiles-fast--balanced--parity-packagesfast-pplxscriptsbenchmarkpy-ane-profiles)

### 2. Long-document `MaxSim` (4K–32K `LongEmbed`) & 100K-document index search

| Indexing mode (M4 Pro) | Quality vs. FP32 single-vector | Index size / peak Metal RAM | Speedup & latency |
| :--- | :--- | :--- | :--- |
| Single-vector whole document (FP32, up to 32K) | Baseline: 0.047 on 32K passkey, 0.895 on `2wikimqa` | 4.00 KiB/doc · 4.27 GB Metal peak | 2,205.5 s GPU encode · 2.82 ms/query @ 100K docs |
| **`build_index(..., multivector=True)` (`MaxSim`, 1024d INT8)** | **1.000** on 32K passkey split (**0.674** pooled, `R@10=1.00`), **0.985** on `2wikimqa` | 15.97 KiB/doc · **1.72 GB Metal peak** (2.48× lower) | **955.4 s GPU encode (2.31× faster)** |
| `build_index(..., mode="int8")` (1024d INT8, `< 16K` default) | −0.0007 NanoBEIR, −0.0012 LegalBench (lossless) | 1,026 B/vec (3.99× smaller) | 2.11 ms/query single-query @ 100K docs |
| **`build_index(..., mode="binary_int8")` (`>= 16K` default)** | +0.0006 NanoBEIR, −0.0012 LegalBench at Top-64 (`rerank_top_r=256` default) | 128 B hot binary + 1,026 B INT8 (3.55× smaller) | **0.66 ms/query @ Top-64 on 100K docs (4.28× faster)** |

On official `LongEmbed` per-length splits (50 queries against 100 documents at one length), multi-vector `MaxSim` reaches **1.000 `nDCG@10`** at every length from 4K to 32K. When pooling all 400 passkey documents across `[4K, 8K, 16K, 32K]`, identical target entities exist in all four splits, which caps any length-invariant ranker near 0.640–0.680 while still achieving 1.000 `Recall@10`. [LongEmbed and compact index receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 3. Throughput & latency across sequence lengths

| Workload | M4 Pro (48 GB) | Base M5 (24 GB) | Comparison & notes |
| :--- | ---: | ---: | :--- |
| Single short query (`S=15`, ANE) | **4.18 ms** (0.25 GB) | **5.52 ms** (0.25 GB) | 2.16× faster than stock `pplx-context-mlx` (9.04 ms); 6.87× faster than PyTorch MPS |
| Short passages (`S=256`, batch 8) | **14,884 tok/s** | **15,235 tok/s** | 1.68× (M4 Pro) and 1.32× (M5) vs stock `pplx-context-mlx` |
| Long windows (`S=4096`, batch 4) | **7,531 tok/s** (3.06 GB) | **9,697 tok/s** | 1.26× faster than stock `pplx-context-mlx`; 4.16× less OS memory than PyTorch MPS (12.70 GB) |
| Mixed corpus (64 docs, `FFD hetero2`) | **17,793 tok/s** | **19,443 tok/s** | 1.84× faster than GPU-only on M4 Pro and 1.17× on M5 |
| Single 32,768-token document (streamed GPU) | **2,720 tok/s** (2.69 GB) | Verified (`<= 19.2 GB`) | `MappedEmbedding` keeps the 1.02 GB token table mmap-backed on CPU; streamed 1,024-token GDN + 512-token SDPA holds Metal peak to 1.78 GB |

### 4. Direct visual page embeddings (`encode_images`, `build_image_index`, Metal GPU)

`encode_images(..., batch_size=16)`, `build_image_index(...)`, and `fast-pplx embed-images` reuse `fast_shared.qwen35_vision` (`FastQwen35VisionModel`, exact fractional `<|chunk_sep|>` 3D MRoPE, optional `crop_margins`, `rotations_cw` / `--rotate-cw`, and opt-in `grid_bucket_step` / `--grid-bucket-step`) lazily — text-only workloads allocate zero ViT memory (`B=16` holds `min_cos=1.00000` at `6.41 img/s` on mixed crops; keep `--grid-bucket-step` opt-in since padding tiny crops shifts cosine to `0.7572` min). `build_image_index(..., crop_margins=True, agg="maxsim")` flattens multi-band page embeddings into a `CompactRetrievalIndex` for cross-modal text-to-page search (`--build-index` / `--query` on CLI). [Cross-silicon vision & OCR receipts](../../docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23)

| Mode (`float16`, `batch_size=4`) | PyTorch MPS (M4 Pro `bf16`) | `fast-pplx` M4 Pro (`Apple9`) | `fast-pplx` Base M5 (`Apple10 NAX`) | Fidelity vs. PyTorch Ref |
| :--- | ---: | ---: | ---: | :--- |
| `100 DPI`, `chunks=8` (`600K` max px) | `573.40 ms/p` (`1.74 pp/s`) | **`159.12 ms/p`** (`6.28 pp/s`, **3.60×**, `1.87 GB`) | **`109.24 ms/p`** (`9.15 pp/s`, **5.25×**, `1.72 GB`) | `mean_cos = 0.99970`, `0.0` MRoPE diff, `3/3` cross-modal Top-1 |
| `100 DPI`, `chunks=1` (`600K` max px) | `539.97 ms/p` (`1.85 pp/s`) | **`158.71 ms/p`** (`6.30 pp/s`, **3.40×**, `1.88 GB`) | **`108.45 ms/p`** (`9.22 pp/s`, **4.98×**, `1.72 GB`) | `mean_cos = 0.99981`, exact token IDs & spans |
| `150 DPI`, `chunks=8` (`1.8M` max px) | `2,033.55 ms/p` (`0.49 pp/s`) | **`688.48 ms/p`** (`1.45 pp/s`, **2.95×**, `2.33 GB`) | **`424.87 ms/p`** (`2.35 pp/s`, **4.79×**, `2.33 GB`) | `mean_cos = 0.99984` (`M4`) / `0.99983` (`M5`) |

To enable `PIL` (images) and optional `pypdfium2` (PDF rasterization) dependencies, install with the `vision` extra (`uv pip install -e "packages/fast-pplx[vision]"`).

## Reproducible benchmarks

All GPU and ANE benchmarks self-serialize under `/tmp/m4_gpu_validation.lock`:

```bash
python packages/fast-pplx/scripts/benchmark.py all --dry-run
python packages/fast-pplx/scripts/benchmark.py speed --modes gpu,ane,hetero2
python packages/fast-pplx/scripts/benchmark.py parity --modes gpu,ane,hetero2
python packages/fast-pplx/scripts/benchmark.py context-limit --lengths 512,2048,8192,32768
```

## Scope, Acknowledgments & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**. Rebuilding `.mlmodelc` bundles requires optional `coremltools >= 8.0`.

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Foundational patterns for running `pplx-embed-v2-context-0.6b` cleanly on Apple Silicon (`MappedEmbedding`, streamed GatedDeltaNet + tile-aligned SDPA slicing, RMSNorm parity analysis, 16-case reference parity, contextual window evaluation, and OS footprint measurement).
- **[`mlx-lm`](https://github.com/ml-explore/mlx-lm)** (Copyright © 2025 Apple Inc., MIT License) & **[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)**: Metal `gated_delta_update` recurrence kernels and upstream model weights.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
