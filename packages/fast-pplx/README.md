# fast-pplx

Contextual passage embeddings and 4K-32K long-document search on Apple Silicon using Perplexity's `pplx-embed-v2-context-0.6b`. Encoding passages with surrounding document context lifts 175-query legal passage ranking from **0.359 to 0.703** while finishing a 325-query benchmark in **136.5 s** vs. stock `pplx-context-mlx`'s **601.5 s** (**4.4x** faster at equal accuracy).

Runs short batches on the 16-core Apple Neural Engine (ANE) and long or tiny query batches on the Metal GPU (`hetero2` cuts mixed batch wall time **1.34-1.75x** vs. GPU alone), with a 32,768-token GPU path in **2.69 GB** process footprint. Zero telemetry, no account, and runs 100% offline after initial ~2.38 GB prebuilt download. Prebuilt Core ML ANE artifact: [`mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane`](https://huggingface.co/mikewolfd/pplx-embed-v2-context-0.6b-coreml-ane) · [Full benchmark tables](../../docs/BENCHMARKS.md).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+, Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported for Metal GPU / Core ML execution. Xcode Command Line Tools are required to build the native ANE runner (`make -C packages/fast-pplx native`).

## Quickstart (~2.38 GB prebuilt download)

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-pplx --inexact
make -C packages/fast-pplx native

.venv/bin/fast-pplx setup-weights          # Stages ~2.38 GB prebuilt weights + ANE stages
# Optional: compile Pure-ANE stages locally (`balanced` profile by default):
# .venv/bin/fast-pplx setup-ane --precision w8a16
```

```python
from fast_pplx import FastPplx

with FastPplx(mode="auto", dimensions=512) as model:  # hetero2 if ANE stages exist, else gpu
    q = model.encode_queries(["How do ocean currents transport heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ], max_length=512)

    # For 4K-32K documents, build a contextual multi-vector MaxSim index (1024d INT8):
    long_index = model.build_index(["Long 32K document text...", "Second long document..."], multivector=True)
    top_ids, top_scores = long_index.search(model.encode_queries(["ocean currents"])[0], k=5)
```

`max_length` caps the full document including prefix and chunk separators. Chunks past the cap return zero vectors; use `fast-pplx parquet --context-tokens 2048` or `build_index(..., multivector=True)` to index arbitrary-length documents automatically.

## CLI, Parquet & HTTP server

```bash
# Single or batch embeddings (--mode auto|gpu|ane|hetero2, --dimensions 128..1024, --dtype int8|float16|float32)
.venv/bin/fast-pplx embed "First passage" "Second passage" --dimensions 1024 --dtype int8 --json

# Contextual Parquet ingestion (--header-col repeats up to 128 header tokens per 2K window)
.venv/bin/fast-pplx parquet --input docs.parquet --output out_shards/ --text-col body --header-col title \
  --mode hetero2 --chunk-tokens 256 --context-tokens 2048 --dtype int8 --resume

# OpenAI-compatible HTTP server (/v1/embeddings; optional --quantize-weights w8 saves ~214 MB RAM)
.venv/bin/fast-pplx serve --host 127.0.0.1 --port 8080 --mode hetero2 --quantize-weights w8

# Direct visual page / PDF embeddings (defaults --mode gpu, 0 ANE overhead; requires `[vision]` extra)
.venv/bin/fast-pplx embed-images page.png document.pdf --chunks 8 --crop-margins --dtype int8 --output img_vecs.npz
.venv/bin/fast-pplx embed-images page1.png page2.png --chunks 8 --crop-margins --build-index --query "search question"
```

## Choosing modes & index settings

| Setting | Default | Behavior |
| --- | --- | --- |
| `mode="auto"` | `hetero2` (GPU + ANE) | Routes single `B=1` queries and M5 tiny batches (`<=32` tok/item, `<=1024` total) inline on Metal GPU (`2.29x` faster on M5 `B=32, S=15`); uses first-fit bin packing on `<=512` ANE chunks (`-60%` pad waste) and GPU above 512 tokens. |
| `quantize_weights="none"` | `fp16` (`"w8"` opt-in) | Keeps full FP16 precision by default. Opt-in `"w8"` saves ~214 MB Metal RAM and enables Apple10 NAX `W8A8` (`1.07-1.08x` faster than `w8` without NAX at `>=0.9984` cosine vs. FP16). |
| `--ane-profile balanced` | `balanced` | Closes 73%+ of the ANE-vs-GPU cosine gap (`>=0.999` mean cosine) at negligible throughput cost vs. `fast`. |
| `--context-tokens 2048` | `2048` | Encodes 256-tok chunks inside 2,048-tok windows with up to 128 repeated header tokens (`0.453 -> 0.735` LegalBench nDCG). |
| `dimensions=1024` + `INT8` | `1024` / `int8` | Preserves all 1,024 dims in 4x-smaller INT8 (`1,026 B/vec`, lossless vs. FP32); truncating PPLX to 512d drops legal ranking 3.7 pts. |
| `multivector=True` | `False` | Splits 4K-32K docs into 2,048-tok windows scored via `MaxSim` (`15.97 KiB/doc` in 1024d INT8). |

## Benchmark highlights

### 1. Short & contextual retrieval (`NanoBEIR-en + LegalBench-RAG`, 325 queries)

| Implementation | Base M5 score | M5 score/s (wall, mem) | M4 Pro score | M4 Pro score/s (wall, mem) | vs. Stock MLX (M5 / M4 Pro) |
| --- | ---: | ---: | ---: | ---: | --- |
| **`fast-pplx` (`hetero2`)** | **0.7244** | **1.724** (136.5 s, 2.34 GB) | **0.7241** | **1.407** (167.3 s, 1.97 GB) | **4.41x** / **4.33x** |
| Stock `pplx-context-mlx` (FP16) | 0.7236 | 0.391 (601.5 s, 2.12 GB) | 0.7236 | 0.325 (723.1 s, 2.19 GB) | 1.00x |

The paired score difference vs. stock is `+0.0008` (95% CI `-0.0036` to `+0.0053`), with all 325 queries served cleanly via windowing. Length-sorting alone accounts for ~1.55x on stock MLX (cutting a 384-passage sample from `20.65 s` to `13.34 s` on M4 Pro). [Task results](../../docs/benchmarks/document-workloads.json) · [Padding probe](../../docs/benchmarks/pplx-padding-20261121.json)

### 2. Long-document `MaxSim` (`LongEmbed` 4K-32K) & 100K vector search (`M4 Pro`)

| Indexing mode | Quality vs. FP32 whole-doc | Index size / peak Metal RAM | Speedup & query latency |
| --- | --- | --- | --- |
| Single-vector whole doc (up to 32K) | `0.047` on 32K passkey, `0.895` on `2wikimqa` | 4.00 KiB/doc / 4.27 GB Metal peak | 2,205.5 s GPU encode / 2.82 ms @ 100K docs |
| **`build_index(..., multivector=True)` (`MaxSim`)** | **1.000** on 32K split (**0.674** pooled), **0.985** `2wikimqa` | 15.97 KiB/doc / **1.72 GB Metal peak** (2.48x lower) | **955.4 s GPU encode (2.31x faster)** |
| `build_index(..., mode="int8")` (`<16K`) | `-0.0007` NanoBEIR, `-0.0012` LegalBench | 1,026 B/vec (3.99x smaller) | 2.11 ms/query @ 100K docs |
| **`build_index(..., mode="binary_int8")` (`>=16K`)** | `+0.0006` NanoBEIR, `-0.0012` LegalBench @ Top-64 | 128 B binary + 1,026 B INT8 (3.55x smaller) | **0.66 ms/query @ 100K docs (4.28x faster)** |

### 3. Stock `pplx-context-mlx` GPU vs. `fast-pplx` (`gpu` / `ane` / `hetero2`)

| Workload | Host | Stock `pplx-context-mlx` (`fp16`) | `fast-pplx` `gpu` | `fast-pplx` `ane` (`balanced`) | **`fast-pplx` `hetero2`** | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Single query (`B=1, S=15`) | Base M5 | `9.04 ms` | `7.09 ms` (2,117 tok/s) | `9.06 ms` (0.26 GB) | **`5.80 ms`** (2,586 tok/s) | **1.56x** stock (**1.22x** GPU, 0.91 GB) |
| Single query (`B=1, S=15`) | M4 Pro | - | **`5.76 ms`** (2,604 tok/s) | `4.18 ms` (quiet) | `6.50 ms` (2,309 tok/s) | `B=1` routes inline on Metal GPU |
| Query batch (`B=32, S=15`) | Base M5 / M4 Pro | - | **`0.0383 s`** (10,706 tok/s) / `6,906` | `0.0876 s` (`4,381`) / `4,457` | **`0.0383 s`** (inline GPU) / **`0.0518 s`** (`9,249`) | M5 routes tiny non-contextual batches inline on GPU (**2.29x** ANE, `cos=0.9996`); M4 Pro uses ANE+GPU |
| Passages (`S=256, B=8`) | Base M5 / M4 Pro | `10,384` / `8,425 tok/s` | `19,332` / `11,812` | `13,407` / `13,516` | **`27,234` / `21,345 tok/s`** | **2.62x (M5) / 2.53x (M4 Pro)** stock |
| Passages (`S=512, B=8`) | Base M5 / M4 Pro | - | `19,419` / `11,776` | `11,988` / `12,175` | **`28,415` / `19,060 tok/s`** | `55.5 docs/s` M5 / `37.2 docs/s` M4 Pro |
| Mixed short+long (`28` docs) | Base M5 / M4 Pro | - | `16,636` / `10,357` | - | **`22,273` / `15,212 tok/s`** | **1.34x (M5) / 1.47x (M4 Pro)** GPU |
| Long windows (`S=4096, B=4`) | Base M5 / M4 Pro | - | **`9,697` / `7,531 tok/s`** | GPU-routed | GPU-routed | **1.26x** stock on M4 Pro; 4.16x less RAM than PyTorch MPS |
| Single 32,768-tok doc | M4 Pro / Base M5 | - | **`2,720 tok/s`** (M4 Pro, 2.69 GB) | GPU-routed | GPU-routed | Memory-maps `1.02 GB` embed table on CPU (`1.78 GB` Metal peak) |

### 4. Lazy visual page embeddings (`encode_images`, `[vision]` extra)

Install with `uv sync --python 3.13 --package fast-pplx --extra vision --inexact`; `fast-pplx embed-images` defaults to `--mode gpu` so ANE text stages stay unloaded (`0` text-side ViT RAM overhead). Async CPU image prefetch with `max_pixels=786_432` reaches **8.90 img/s on M5** (`1.12x` unprefetched) and **6.00 img/s on M4 Pro** (`1.10x`), cutting 164-page wall time **2.08x** (`53.62 s -> 25.77 s`) at unchanged `R@1 = 0.9634` (`min_cos >= 0.9999998`), and running **3.4-5.2x** faster than PyTorch MPS `bf16`. [Vision & prefetch receipts](../../docs/BENCHMARKS.md#141-vision-retrieval-cpu-image-prefetch-and-multimodal-mtp)

## Limitations, Acknowledgments & License

- **Limitations:** Opt-in `--grid-bucket-step` image padding can lower minimum cosine similarity to `0.7572` on small crops (kept `None` by default). Truncating PPLX embeddings to 512d drops retrieval accuracy unless using MRL-trained [`fast-embeddinggemma-2`](../fast-embeddinggemma-2).
- **Acknowledgments:** [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) (Copyright 2026 Merchantry LLC, MIT License) established `MappedEmbedding`, tile-aligned SDPA slicing, RMSNorm parity tests, and OS footprint measurement; Metal GDN kernels draw from [`mlx-lm`](https://github.com/ml-explore/mlx-lm) (MIT).
- **License:** Package code is [MIT licensed](LICENSE); Perplexity's `pplx-embed-v2-context-0.6b` weights remain subject to their upstream license. Verified on 48 GB M4 Pro and 24 GB Base M5.
