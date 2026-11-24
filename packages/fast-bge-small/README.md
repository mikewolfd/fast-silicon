# fast-bge-small

Turn short English queries and passages into 384-number vectors with `BAAI/bge-small-en-v1.5`. One warm 24-token query took 0.49 ms on the M4 Pro and 0.79 ms on the base M5. Across 1,000 pretokenized documents the package processed 187,150 and 200,646 tokens/s on those machines.

`fast-bge-small` runs short queries on the Apple Neural Engine (ANE) and can split bulk encoding between the GPU and ANE. It includes a Python API, an OpenAI-compatible `/v1/embeddings` server, a resumable Parquet CLI, and the standalone `bge-ane-cli`. Inference uses neither `torch` nor `transformers`. Prebuilt Core ML ANE artifact: [`mikewolfd/bge-small-en-v1.5-coreml-ane`](https://huggingface.co/mikewolfd/bge-small-en-v1.5-coreml-ane) · [Full cross-silicon benchmarks](../../docs/BENCHMARKS.md)

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/bge-small setup-weights
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

```bash
./build/bge-ane-cli --text "distributed cache invalidation strategy" --query-prefix --json
```

## Benchmark highlights

### 1. Retrieval quality & elapsed time (`NanoBEIR-en + LegalBench-RAG`, 325 queries)

The retrieval score averages the ranking scores for 150 NanoBEIR queries and 175 LegalBench-RAG queries. Effective outcome throughput (EOT) divides their total score by elapsed seconds. Timing starts after model loading and warmup.

| Host | Engine | Mean score | EOT (score/s) | Wall time | Measured memory | Speedup vs. Stock MLX |
|---|---|---:|---:|---:|---:|---:|
| **Base M5 (24 GB)** | **`fast-bge-small` (`hetero`)** | 0.4737 | **13.061** | **11.8 s** | **1.54 GB** | **11.26×** |
| Base M5 | Public Core ML (sidekickd FP16) | 0.4739 | 2.567 | 60.0 s | 0.54 GB | 2.21× |
| Base M5 | Stock `mlx-embeddings` (8-bit) | 0.4774 | 1.160 | 133.8 s | 18.78 GB | 1.00× |
| **M4 Pro (48 GB)** | **`fast-bge-small` (`hetero`)** | 0.4737 | **12.634** | **12.2 s** | **1.55 GB** | **9.58×** |
| M4 Pro | Public Core ML (sidekickd FP16) | 0.4739 | 2.382 | 64.7 s | 0.74 GB | 1.81× |
| M4 Pro | Stock `mlx-embeddings` (8-bit) | 0.4774 | 1.319 | 117.6 s | 38.99 GB | 1.00× |

`fast-bge-small` delivered 5.09× the score per second of the sidekickd FP16 export on M5 and 5.30× on M4 Pro. The mean ranking scores differed by 0.0002, and the sidekickd export used less measured memory on both Macs. On M4 Pro its embeddings had at least 0.99996 cosine similarity to `fast-bge-small`; the alternative `michaeljelly/bge-small-en-coreml-v1.5` export returned non-finite first-token embeddings on CPU/GPU and low cosine similarity (0.518–0.619) on ANE. [Public Core ML details](../../docs/BENCHMARKS.md#public-core-ml-results)

Prepending document headers when chunking Parquet files raised 175-query legal passage ranking from 0.305 to 0.444 and character recall from 0.631 to 0.710. For vector search, `eng.build_index(corpus)` compresses embeddings 4× into symmetric INT8 (386 B/chunk) with negligible score change (−0.0013 on NanoBEIR, +0.0026 on LegalBench). At 16,000 or more documents, it adds a 48-byte binary Hamming prefilter that searches 100,000 vectors in **0.50 ms per query** versus 1.46 ms in FP32 BLAS (2.95× faster). Unlike Matryoshka-trained models, truncating BGE-small to 256 dimensions drops NanoBEIR ranking by 0.0184, so backfill defaults preserve all 384 dimensions. [Chunking and compact index receipts](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 2. Stock `mlx-embeddings` GPU vs. `fast-bge-small` (`gpu` / `ane` / `hetero`)

| Host | Workload | Stock `mlx-embeddings` GPU | `fast-bge-small` `gpu` | `fast-bge-small` `ane` | **`fast-bge-small` `hetero`** | Public Core ML (`sidekickd` FP16) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | Warm single 24-tok query | `4.42 ms` | `1.29 ms` (`3.4×` stock) | **`0.79 ms`** (`5.6×` stock) | ANE-routed | `1.61 ms` |
| **Base M5 (24 GB)** | Bulk text throughput | `106.96 docs/s` (Tier-1 `1.16` EOT) | `152,937 tok/s` | `63,850 tok/s` | **`201,181 tok/s`** (`200,646` 1K corpus) | `34,318 tok/s` |
| **M4 Pro (48 GB)** | Warm single 24-tok query | `4.85 ms` | `1.98 ms` (`2.4×` stock) | **`0.48–0.49 ms`** (`10.0×` stock) | ANE-routed | `1.70 ms` |
| **M4 Pro (48 GB)** | Bulk text throughput *(2026-11-24 contended)* | `123.12 docs/s` (Tier-1 `1.32` EOT) | `121,730 tok/s` | `59,304 tok/s` | **`192,512 tok/s`** (`187,150` 1K quiet) | `31,608 tok/s` |

Parallel 9-shape Core ML build (`--jobs 4`): **`2.24 s`** on Base M5 (vs `5.02 s` serial) and **`4.38 s`** on M4 Pro (vs `9.37 s` serial) with bit-identical outputs across all shapes. [Build checks](../../docs/benchmarks/capability-corrections-20261121.json) · [Dual-Mac receipt](../../docs/benchmarks/dual-mac-suite-20261124.json)

### 3. Mode breakdown (`bge-small bench --n 4000`, M4 Pro 48 GB)

| Workload (tokens) | Mode | Elapsed | Throughput | GPU docs (tokens) | ANE docs (tokens) |
|---|---|---:|---:|---:|---:|
| Synthetic (993,944) | **`hetero`** | **5.25 s** | **189,338 tok/s** | 1,504 (577,600) | 2,496 (416,344) |
| Synthetic (993,944) | `gpu` | 7.19 s | 138,335 tok/s | 4,000 (993,944) | 0 (0) |
| Synthetic (993,944) | `ane` | 16.77 s | 59,261 tok/s | 0 (0) | 4,000 (993,944) |
| Document OCR (981,829) | **`hetero`** | **4.57 s** | **215,022 tok/s** | 1,856 (632,487) | 2,144 (349,342) |

Across a 64-document verification fixture (`bge-small verify`), minimum cosine similarity against the FP16 reference is `0.999998` on GPU, `0.999984` in `hetero`, and `0.999986` on a single 24-token ANE query.

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **16-core ANE runner** | Pure-ANE Core ML buckets with a 4-slot async ring, NEON FP16 transpose, and Layer-11 `[CLS]` slicing | [`native/bge_ane_runner.mm`](native/bge_ane_runner.mm) |
| **Metal 4 GPU** | Packed variable-length batching with zero attention padding, bidirectional FlashAttention, and fused GELU + LayerNorm | [`src/fast_bge_small/gpu.py`](src/fast_bge_small/gpu.py) |
| **Work-stealing scheduler** | The GPU pulls the longest sequences forward while the ANE pulls the shortest backward, plus 4-batch Parquet prefetch (1.18–1.33× faster) | [`src/fast_bge_small/`](src/fast_bge_small/) |

## CLI, Parquet & HTTP server

```bash
.venv/bin/bge-small embed "first passage" "second passage" --json
.venv/bin/bge-small parquet --input in.parquet --output out/ --text-col body \
  --dtype float32 --chunk --max-body-tokens 510 --overlap 51 --filter-ocr --resume
.venv/bin/bge-small serve --host 127.0.0.1 --port 8080
.venv/bin/bge-small bench --mode all --n 4000
.venv/bin/python scripts/build_ane_models.py --jobs 4
.venv/bin/pytest -v
```

Parquet ingestion also supports `--ocr-pages-only`, `--per-ocr-page` (UTF-8 `page_offsets`), directory shard `--resume`, and `int8` or `float16` output vectors.

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**. Sequences beyond the model’s 512-token context window and non-Apple-Silicon targets are not supported.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
