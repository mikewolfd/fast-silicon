# fast-bge-small

Turn short English queries and passages into 384-number vectors with `BAAI/bge-small-en-v1.5`. One warm 24-token query took 0.49 ms on the M4 Pro and 0.79 ms on the base M5. Across 1,000 pretokenized documents the package processed 187,150 and 200,646 tokens/s on those machines.

`fast-bge-small` runs short queries on the Apple Neural Engine (ANE) and can split bulk encoding between the GPU and ANE. It includes a Python API, an OpenAI-compatible `/v1/embeddings` server, a resumable Parquet CLI and the standalone `bge-ane-cli`. Inference uses neither `torch` nor `transformers`.

> **Full Cross-Silicon Benchmarks:** See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for full Tier-1 retrieval breakdowns, 4-bucket memory telemetry, and cross-package comparisons.

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
```

```bash
./build/bge-ane-cli --text "distributed cache invalidation strategy" --query-prefix --json
```

## Cross-Silicon Outcome & Speed Highlights

### 1. Deterministic Retrieval Outcomes (`NanoBEIR-en + LegalBench-RAG`, `N=325`)

The retrieval score averages the ranking scores for 150 NanoBEIR queries and 175 LegalBench-RAG queries. Effective outcome throughput (EOT) divides their total score by elapsed seconds. Timing starts after model loading and warmup.

| Host | Engine | `S_obj,e2e` | `EOT (u/s)` | Wall Time | 4-Bucket Mem | Speedup vs. Stock `MLX` |
|---|---|---:|---:|---:|---:|---:|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`fast-bge-small` (`hetero`)** | `0.4737` | **`13.061`** | **`11.8 s`** | **`1.54 GB`** | **`11.26x`** (`1.03x` M4 Pro) |
| Base M5 | Public Core ML (sidekickd FP16) | 0.4739 | 2.567 | 60.0 s | 0.54 GB | 2.21× |
| Base M5 (`24 GB`) | Stock `mlx-embeddings` (`bge-small-en-v1.5-8bit`) | `0.4774` | `1.160` | `133.8 s` | `18.78 GB` | `1.00x` |
| **M4 Pro (`48 GB`, `Apple9`)** | **`fast-bge-small` (`hetero`)** | `0.4737` | **`12.634`** | **`12.2 s`** | **`1.55 GB`** | **`9.58x`** |
| M4 Pro | Public Core ML (sidekickd FP16) | 0.4739 | 2.382 | 64.7 s | 0.74 GB | 1.81× |
| M4 Pro (`48 GB`) | Stock `mlx-embeddings` (`bge-small-en-v1.5-8bit`) | `0.4774` | `1.319` | `117.6 s` | `38.99 GB` | `1.00x` |

`fast-bge-small` delivered 5.09× the score per second of the sidekickd FP16 export on M5 and 5.30× on M4 Pro. The mean ranking scores differed by 0.0002. The sidekickd export used less measured memory on both Macs. The harness takes the larger of its evaluator and model-process estimates. The runs used the same dataset lock and were timed separately.

The public Core ML comparison uses `sidekickd/bge-small-en-v1.5-coreml`. On M4 Pro its embeddings had at least 0.99996 cosine similarity to `fast-bge-small` across two queries and three documents. The alternative `michaeljelly/bge-small-en-coreml-v1.5` export returned non-finite first-token embeddings with `CPU_ONLY` and `CPU_AND_GPU`. With `CPU_AND_NE` it reached 0.518 cosine similarity on the test query and 0.619 on the test document.

- **v2.3 Header-Anchored & Sentence-Snapped Chunking (`parquet_pipeline.py`):** Prepending document headers (`plan_header_anchored_windows` + `auto_snap_header`) lifted 175-query `LegalBench-RAG` `span_ndcg@10` from **`0.3046 -> 0.4439` (`+45.7%`)** and `char_recall@10` from **`0.6314 -> 0.7099`** at `136,600 tok/s` (`sentence_snap` alone without header anchoring scored `0.2800` and remains opt-in). [v2.3 validation](../../docs/benchmarks/v23-validation-20261121.json)
- **v2.5 `CompactRetrievalIndex` & Model-Aware `CorpusBackfillConfig.for_engine`:** `eng.build_index(corpus, mode="auto")` builds a `384d` symmetric `INT8 + FP16 scale` index (`386 B/chunk`, `3.98x` compression, `ΔnDCG@10 = -0.0013` on `NanoBEIR` / `+0.0026` on `LegalBench-RAG`, `1.16 ms/q` @ `100K` docs) for `< 16K` corpora and automatically upgrades to `mode="binary_int8"` (`48 B` hot binary Hamming `Top-64` + `384d INT8` rerank, `434 B` total = `3.54x`, `Δ = -0.0003` on `NanoBEIR`, **`0.50 ms/q` vs `1.46 ms/q` `FP32` BLAS @ `100K` docs = `2.95x` faster**) for `>= 16K` docs, while `CorpusBackfillConfig.for_engine(eng)` keeps full `384d INT8` (`is_mrl_trained=False`, avoiding the `-0.0184` `NanoBEIR` drop at `256d`). [v2.5 R&D receipt](../../docs/benchmarks/v25-rd-winners-20261122.json)

### 2. Throughput, Latency & Parallel Multi-Bucket CoreML Build

| Benchmark (`benchmark_monorepo.py` / `benchmark_capabilities.py`) | Base M5 (`24 GB`, `macOS 27.2.1`) | M4 Pro (`48 GB`, `macOS 26.6`) | Takeaway |
|---|---|---|---|
| **Corpus throughput (`hetero`)** | **200,646 tokens/s** (1.85 GB) | **187,150 tokens/s** (2.09 GB) | Both runs encoded the same 249,303 tokens in 1,000 documents. |
| **Single-query latency (ANE)** | **0.79 ms** | **0.49 ms** | Median for one warm 24-token query. |
| sidekickd FP16 corpus | 34,318 tokens/s | 31,608 tokens/s | Same token IDs as `fast-bge-small`, with tokenization and interprocess transfer outside timing. |
| sidekickd FP16 single query | 1.608 ms | 1.696 ms | Median of 30 warm calls on the same 24-token input. |
| **Parallel Core ML build, 9 shapes** | 5.02 s serial, 2.24 s with four workers | 9.37 s serial, 4.38 s with four workers | Both builds produced bit-identical outputs for 16 vectors across all nine shapes on each host. Timings exclude prediction. [Build runs and per-shape checks](../../docs/benchmarks/capability-corrections-20261121.json) |

Core ML's `MLComputePlan` for each sidekickd input size (128, 256 and 512 tokens) preferred the Neural Engine for 229 operations and the CPU for 16. It left 363 unclassified on each Mac. The model ships precompiled. Loading it before scoring took 4.38 seconds on M5 and 1.84 seconds on M4 Pro. In its separate natural-text run, stock `mlx-embeddings` encoded 106.96 documents/s on M5 and 123.12 on M4 Pro. Its single text query took 4.42 ms and 4.85 ms, respectively.

The run records include the model revisions, individual timings and per-bucket compute plans. [Benchmark methods](../../docs/BENCHMARKS.md#public-core-ml-results) · [Measured results (JSON)](../../docs/benchmarks/coreml-20261120.json)

## At a glance

| Question | Answer |
|---|---|
| How close to float reference? | `0.999998` min cos (`GPU` vs. `F16` and `BF16`); `0.999984` (`Hetero` vs. `F16`); `0.999986` (`ANE` vs. `GPU` `B=1 S=24`). |
| How fast is one 24-token query? | The ANE median was **0.49 ms** on M4 Pro and **0.79 ms** on M5. The standalone `bge-ane-cli` took 0.502 ms on its M4 Pro fixture. |
| How fast is bulk encoding (`hetero`)? | **200,646 tokens/s** on M5 and **187,150 tokens/s** on M4 Pro. A separate 4,000-document M4 Pro run reached **215,022 tokens/s** on OCR text and **189,338 tokens/s** on synthetic inputs. |
| Requires `torch` or `transformers`? | No at runtime (`mlx`, `numpy`, `tokenizers`, `pyarrow`, `huggingface-hub` only; `coremltools >= 8.0` only to rebuild `.mlmodelc`). |

## Parity (`64`-doc fixture, `bge-small verify`)

| Comparison | Min Cosine | Mean Cosine / Notes |
|---|---:|---|
| `GPU` vs. `F16` reference | `0.999998` | Mean `0.999999` across `64` docs |
| `GPU` vs. `BF16` reference | `0.999998` | Matches BF16 reference |
| `Hetero` (`GPU + ANE`) vs. `F16` | `0.999984` | Two-ended bucket routing |
| `Single-query B=1 S=24 ANE` vs. `GPU` | `0.999986` | `build/bge-ane-cli`: `0.502 ms` (`tokens=16`, `iters=30`) |

## Standalone Mode Breakdown (`bge-small bench --n 4000`, M4 Pro `48 GB`)

| Workload (Tokens) | Mode | Elapsed | Throughput | GPU Docs (`Tokens`) | ANE Docs (`Tokens`) |
|---|---|---:|---:|---:|---:|
| Synthetic (`993,944`) | **`hetero`** | **5.250 s** | **189,338 tok/s** | `1,504` (`577,600`) | `2,496` (`416,344`) |
| Synthetic (`993,944`) | `gpu` | `7.185 s` | `138,335 tok/s` | `4,000` (`993,944`) | `0` (`0`) |
| Synthetic (`993,944`) | `ane` | `16.772 s` | `59,261 tok/s` | `0` (`0`) | `4,000` (`993,944`) |
| Document OCR (`981,829`) | **`hetero`** | **4.566 s** | **215,022 tok/s** | `1,856` (`632,487`) | `2,144` (`349,342`) |
| Document OCR (`981,829`) | `gpu` | `8.249 s` | `119,028 tok/s` | `4,000` (`981,829`) | `0` (`0`) |
| Document OCR (`981,829`) | `ane` | `16.272 s` | `60,339 tok/s` | `0` (`0`) | `4,000` (`981,829`) |

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **16-core ANE Runner** | `MLComputeUnitsCPUAndNeuralEngine`, NEON FP16 add, `8x8` register transpose, Layer-11 `[CLS]` slice, 4-slot async ring | [`native/bge_ane_runner.mm`](native/bge_ane_runner.mm) |
| **Metal 4 GPU (`Apple9` / `Apple10 NAX`)** | Packed varlen batching (0 attn pad), custom `D=32` bidirectional FlashAttention, 128-wide `MPP::matmul2d` fused GELU/residual, single-pass LN+L2 | [`src/fast_bge_small/gpu.py`](src/fast_bge_small/gpu.py) |
| **Work-Stealing Scheduler & Parallel Build** | GPU consumes longest sequences forward while ANE pulls shortest sequences backward with zero contention; `spawn`-parallel 9-bucket CoreML build (`--jobs 4`); `lookahead_batches=4` Parquet prefetch delivers **`1.331x`** M4 Pro (`373.2 docs/s`, `min_cos=0.9999754`) and **`1.178x`** Base M5 (`423.0 docs/s`, `min_cos=0.9999746`) speedup (`N=1024`) | [`src/fast_bge_small/`](src/fast_bge_small/), [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

## CLI, Parquet & Reproduce

```bash
.venv/bin/bge-small embed "first passage" "second passage" --json
.venv/bin/bge-small parquet --input in.parquet --output out/ --text-col body \
  --dtype float32 --chunk --max-body-tokens 510 --overlap 51 --filter-ocr --resume
.venv/bin/bge-small serve --host 127.0.0.1 --port 8080
.venv/bin/bge-small bench --mode all --n 4000
.venv/bin/python scripts/build_ane_models.py --jobs 4
.venv/bin/pytest -v   # native bge-ane-cli, parity, ANE buckets 16..500, Parquet OCR, HTTP
```

Parquet options also include `--ocr-pages-only`, `--per-ocr-page` (UTF-8 `page_offsets`), directory shard `--resume`, and `float16` output vectors.

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`, Xcode SDK `27.2`)** and **Apple Base M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)** producing `0`-warning native builds (`libbge_ane_runner.dylib`, `bge-ane-cli`, plus optional `libbge_cpu_query.dylib` via `make cpu-query`).
- **Not tested:** Other Apple Silicon generations/RAM tiers, Linux/x86/CUDA, or sequences exceeding the model's `512`-token context window.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
