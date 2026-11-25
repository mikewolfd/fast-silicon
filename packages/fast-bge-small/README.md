# fast-bge-small

Turn short English queries and passages into 384-dimensional vectors with `BAAI/bge-small-en-v1.5`. Runs short queries on the 16-core Apple Neural Engine (`0.49 ms` on M4 Pro, `0.79 ms` on M5) and splits bulk encoding across GPU and ANE (`200,646 tok/s`, `11.3x` stock MLX).

Zero telemetry, no account, and runs 100% offline after initial ~0.35 GB prebuilt weight download. Inference uses neither `torch` nor `transformers`. Prebuilt Core ML ANE artifact: [`mikewolfd/bge-small-en-v1.5-coreml-ane`](https://huggingface.co/mikewolfd/bge-small-en-v1.5-coreml-ane) · [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+, Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported. Xcode Command Line Tools are needed only to rebuild native ANE `.dylib` runners (`make native`).

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-bge-small --inexact

.venv/bin/bge-small setup-weights   # Downloads ~0.35 GB prebuilt weights + Core ML ANE buckets
.venv/bin/bge-small verify
```

```python
from fast_bge_small import FastBgeSmall

eng = FastBgeSmall()
q = eng.encode_texts("What is the budget deficit?", is_query=True, add_query_prefix=True)  # (384,)
vecs = eng.encode_texts(["passage one...", "passage two..."], mode="hetero")            # (2, 384)

# Build a compact INT8 / binary+INT8 search index automatically sized to the corpus:
index = eng.build_index(["passage one...", "passage two..."])
top_ids, top_scores = index.search(q, k=5)
```

Optional standalone native CLI (after `make -C packages/fast-bge-small native`):

```bash
./packages/fast-bge-small/build/bge-ane-cli --text "distributed cache invalidation strategy" --query-prefix --json
```

## Benchmark highlights

### 1. Retrieval quality & elapsed time (`NanoBEIR-en + LegalBench-RAG`, 325 queries)

Mean score averages ranking performance across 150 NanoBEIR queries and 175 LegalBench-RAG queries; score/s divides total score by elapsed seconds after model load and warmup.

| Host | Engine | Mean score | Score/s | Wall time | Measured memory | Speedup vs. Stock MLX |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | **`fast-bge-small` (`hetero`)** | 0.4737 | **13.061** | **11.8 s** | **1.54 GB** | **11.26x** |
| Base M5 | Public Core ML (`sidekickd` FP16) | 0.4739 | 2.567 | 60.0 s | 0.54 GB | 2.21x |
| Base M5 | Stock `mlx-embeddings` (8-bit) | 0.4774 | 1.160 | 133.8 s | 18.78 GB | 1.00x |
| **M4 Pro (48 GB)** | **`fast-bge-small` (`hetero`)** | 0.4737 | **12.634** | **12.2 s** | **1.55 GB** | **9.58x** |
| M4 Pro | Public Core ML (`sidekickd` FP16) | 0.4739 | 2.382 | 64.7 s | 0.74 GB | 1.81x |
| M4 Pro | Stock `mlx-embeddings` (8-bit) | 0.4774 | 1.319 | 117.6 s | 38.99 GB | 1.00x |

`fast-bge-small` runs 5.09x (M5) and 5.30x (M4 Pro) faster than the public `sidekickd` FP16 Core ML export at matching mean score (diff 0.0002, min cosine `>= 0.99996`), while `sidekickd` uses less unified memory (`0.54-0.74 GB` vs. `1.54-1.55 GB`). The alternative `michaeljelly/bge-small-en-coreml-v1.5` export returned non-finite first-token embeddings on CPU/GPU and low cosine similarity (`0.518-0.619`) on ANE. [Public Core ML details](../../docs/BENCHMARKS.md#public-core-ml-results)

Prepending document headers during Parquet chunking raises 175-query legal passage ranking from `0.305` to `0.444` (`0.631` -> `0.710` recall). `eng.build_index(corpus)` compresses embeddings 4x into symmetric INT8 (`386 B/chunk`) and, at 16,000+ documents, adds a 48-byte binary Hamming prefilter that searches 100,000 vectors in **0.50 ms/query** vs. `1.46 ms` in FP32 BLAS (`2.95x` faster). Unlike Matryoshka-trained models, truncating BGE-small to 256d loses `0.0184` on NanoBEIR, so defaults keep all 384 dimensions. [Chunking and compact index receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 2. Stock `mlx-embeddings` vs. `fast-bge-small` (`gpu` / `ane` / `hetero`)

| Host | Workload | Stock `mlx-embeddings` GPU | `fast-bge-small` `gpu` | `fast-bge-small` `ane` | **`fast-bge-small` `hetero`** | Public Core ML (`sidekickd` FP16) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | Warm single 24-tok query | `4.42 ms` | `1.29 ms` (3.4x) | **`0.79 ms`** (5.6x) | ANE-routed | `1.61 ms` |
| **Base M5 (24 GB)** | Bulk 1K-2K text corpus | `26,667 tok/s` | `152,937 tok/s` | `63,850 tok/s` | **`200,646 tok/s`** | `34,318 tok/s` |
| **M4 Pro (48 GB)** | Warm single 24-tok query | `4.85 ms` | `1.98 ms` (2.4x) | **`0.48-0.49 ms`** (10.0x) | ANE-routed | `1.70 ms` |
| **M4 Pro (48 GB)** | Bulk 1K-2K text corpus | `30,694 tok/s` | `121,730 tok/s` | `59,304 tok/s` | **`187,150 tok/s`** | `31,608 tok/s` |

Parallel 9-shape Core ML compile (`--jobs 4`) completes in `2.24 s` on M5 (vs. `5.02 s` serial) and `4.38 s` on M4 Pro (vs. `9.37 s` serial) with bit-identical outputs. [Build checks](../../docs/benchmarks/capability-corrections-20261121.json) · [Dual-Mac receipt](../../docs/benchmarks/dual-mac-suite-20261124.json)

### 3. Mode breakdown (`bge-small bench --n 4000`, M4 Pro 48 GB)

| Workload (tokens) | Mode | Elapsed | Throughput | GPU docs (tokens) | ANE docs (tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic (993,944) | **`hetero`** | **5.25 s** | **189,338 tok/s** | 1,504 (577,600) | 2,496 (416,344) |
| Synthetic (993,944) | `gpu` | 7.19 s | 138,335 tok/s | 4,000 (993,944) | 0 (0) |
| Synthetic (993,944) | `ane` | 16.77 s | 59,261 tok/s | 0 (0) | 4,000 (993,944) |
| Document OCR (981,829) | **`hetero`** | **4.57 s** | **215,022 tok/s** | 1,856 (632,487) | 2,144 (349,342) |

On the 64-doc verification fixture (`bge-small verify`), minimum cosine similarity vs. FP16 reference is `0.999998` (`gpu`), `0.999984` (`hetero`), and `0.999986` (24-tok `ane`).

## What keeps it fast

| Component | How it works | Source |
| --- | --- | --- |
| **16-core ANE runner** | Pure-ANE Core ML buckets with a 4-slot async ring, NEON FP16 transpose, and Layer-11 `[CLS]` slicing | [`native/bge_ane_runner.mm`](native/bge_ane_runner.mm) |
| **Metal GPU** | Packed variable-length batching with zero attention padding, bidirectional FlashAttention, and fused GELU + LayerNorm | [`src/fast_bge_small/gpu.py`](src/fast_bge_small/gpu.py) |
| **Work-stealing scheduler** | GPU pulls longest sequences forward while ANE pulls shortest backward, with 4-batch Parquet prefetch (1.18-1.33x faster) | [`src/fast_bge_small/`](src/fast_bge_small/) |

## CLI, Parquet & HTTP server

```bash
.venv/bin/bge-small embed "first passage" "second passage" --json
.venv/bin/bge-small parquet --input in.parquet --output out/ --text-col body \
  --dtype float32 --chunk --max-body-tokens 510 --overlap 51 --filter-ocr --resume
.venv/bin/bge-small serve --host 127.0.0.1 --port 8080
.venv/bin/bge-small bench --mode all --n 4000
.venv/bin/python packages/fast-bge-small/scripts/build_ane_models.py --jobs 4
.venv/bin/pytest -v packages/fast-bge-small/tests
```

## Limitations, Prior Art & License

- **Limitations:** `bge-small-en-v1.5` is English-only with a hard 512-token ceiling (longer inputs are truncated or chunked via `parquet`). `hetero` allocates more unified RAM (`1.55 GB`) than standalone Core ML (`0.54-0.74 GB`); pass `default_mode="ane"` when memory is tight.
- **Prior art:** Built on `mlx` and Apple Core ML; use [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) for 2,048-token multilingual text or [`fast-pplx`](../fast-pplx) for contextual 4K-32K documents.
- **License:** Package code is [MIT licensed](LICENSE) (BAAI's `bge-small-en-v1.5` weights carry the MIT license). Verified on 48 GB M4 Pro and 24 GB Base M5.
