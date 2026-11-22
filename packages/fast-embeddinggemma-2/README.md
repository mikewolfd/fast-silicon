# fast-embeddinggemma-2

**Encode up to 2,048 tokens per passage and choose an embedding width from 128 to 768 dimensions.** Bulk encoding reached 38,517 tokens/s on M4 Pro and 48,803 on M5. On the public retrieval benchmark, the package delivered 4.33–4.80 times the score per second of stock MLX and 6.24 times that of Apple CoreAI on M5.

`fast-embeddinggemma-2` runs the 300M-parameter `google/embeddinggemma-2` text encoder on the Metal GPU and Apple Neural Engine (ANE) together for bulk encoding. A 24-token query on the ANE took 2.11 ms on M4 Pro and 2.58 ms on M5. The package includes a Python API, OCR-aware Parquet ingestion, an OpenAI-compatible `/v1/embeddings` endpoint with `dimensions` support, and a standalone `emb2-ane-cli`. Inference uses MLX and Core ML.

> **Full Cross-Silicon Benchmarks:** See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete M4 Pro vs. Base M5 retrieval, throughput, and CoreAI `.aimodel` comparisons.

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/embeddinggemma-2 setup-weights
.venv/bin/embeddinggemma-2 verify
```

```bash
./build/emb2-ane-cli --ids "2,3250,236787,7293,1109,1816,236787,1" --seq-len 32 --bench 30 --json
.venv/bin/embeddinggemma-2 embed --dimensions 256 --query --query-prefix "quarterly semiconductor capex"
```

## Cross-Silicon Outcome & Speed Highlights

### 1. Deterministic Retrieval Outcomes (`NanoBEIR-en + LegalBench-RAG`, `N=500`)

The retrieval score averages 150 NanoBEIR queries and 175 LegalBench-RAG queries evaluated twice, with short and long passages. Effective outcome throughput (EOT) divides the total score by elapsed seconds after warmup.

| Host | Engine | `S_obj,e2e` | `EOT (u/s)` | Wall Time | 4-Bucket Mem | Speedup vs. Stock |
|---|---|---:|---:|---:|---:|---|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5376`** | **`4.707`** | **`57.1 s`** | **`3.43 GB`** | **`4.33x` MLX · `6.24x` CoreAI** (`1.31x` M4 Pro) |
| Base M5 | Public Core ML (FluidInference FP16) | 0.5186 | 1.666 | 155.7 s | 1.38 GB | 1.53× MLX |
| Base M5 | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.740 | 148.9 s | 0.84 GB | 1.60× MLX |
| Base M5 (`24 GB`) | Stock `mlx-vlm` (`embeddinggemma-2-8bit`) | `0.5364` | `1.086` | `246.9 s` | `19.41 GB` | `1.00x` MLX |
| Base M5 (`24 GB`, `macOS 27.2+`) | Stock Apple `CoreAI` (`coreai-emb2-bf16` `.aimodel`) | `0.5360` | `0.754` | `355.5 s` | Unknown for full run | `0.69x` MLX |
| **M4 Pro (`48 GB`, `Apple9`)** | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5365`** | **`3.583`** | **`74.9 s`** | **`2.49 GB`** | **`4.80x` MLX** |
| M4 Pro | Public Core ML (FluidInference FP16) | 0.5186 | 1.637 | 158.4 s | 1.53 GB | 2.19× MLX |
| M4 Pro | Public Core ML (OpenInfinity INT8) | 0.5183 | 1.649 | 157.2 s | 1.03 GB | 2.21× MLX |
| M4 Pro (`48 GB`) | Stock `mlx-vlm` (`embeddinggemma-2-8bit`) | `0.5353` | `0.746` | `358.6 s` | `39.03 GB` | `1.00x` MLX |

Both public Core ML exports accept up to 512 tokens per input. All 818 short LegalBench passages fit. Of the 144 long passages, 136 exceeded 512 tokens and one exceeded the package’s 2,048-token limit. The evaluator ranks the original passages using the portion each engine encoded. On M5, `fast-embeddinggemma-2`, FluidInference FP16 and OpenInfinity INT8 scored 0.272, 0.271 and 0.270 on short passages and 0.569, 0.516 and 0.518 on long passages. This locates most of the observed difference in the long-passage task.

On M5, the public INT8 export used 0.84 GB and completed scoring in 149 seconds. `fast-embeddinggemma-2` used 3.43 GB and finished scoring in 57 seconds. Memory reports the larger of the evaluator and model-process estimates. The project package, stock MLX and the public Core ML exports ran the same locked inputs in separate timed runs. Before scoring, loading the FP16 and INT8 graphs took 72.1 and 76.9 seconds on M5, and 3.6 seconds each on M4 Pro. The subsequent M5 speed test loaded each export in about 3 seconds.

For Apple CoreAI on the 24 GB M5, the full run returned an invalid 385.79 GB memory estimate. A separate five-text check measured 3.04 GB.

#### How much of a long legal passage did the encoder see?

At 512 tokens, the project, FluidInference FP16 and OpenInfinity INT8 ranked legal passages about equally. All three encoded identical token IDs for 175 questions and 144 passages from 20 original documents on M4 Pro. Raising the project’s limit to 2,048 improved its top-ten whole-passage ranking score from 0.517 to 0.569.

The standard score credits an answer anywhere in a returned passage. The visible-text columns count only annotated characters the encoder received. The ranking score (span nDCG) also gives more credit to higher-ranked passages. Every average includes all 175 questions.

| Encoder | Token limit | Whole-passage nDCG@10 | Visible nDCG@5 / @10 | Visible character recall@5 / @10 | Warm encoding |
|---|---:|---:|---:|---:|---:|
| Project | 512 | 0.517 | 0.293 / 0.301 | 0.323 / 0.336 | 2.39 s |
| FluidInference FP16 | 512 | 0.516 | 0.289 / 0.298 | 0.323 / 0.336 | 4.99 s |
| OpenInfinity INT8 | 512 | 0.518 | 0.293 / 0.302 | 0.323 / 0.336 | 4.84 s |
| Project | 2,048 | 0.569 | 0.519 / 0.569 | 0.757 / 0.903 | 14.58 s |

At 512 tokens, 81 questions had no annotated answer text in any visible passage. All three encoders still earned whole-passage credit on 64 questions whose top ten contained no visible answer text. For example, each encoder ranked the New York City data-use agreement’s first passage first for `contractnli:22`. Its 512 tokens ended at character 2,246. The annotated answer begins at character 3,180. At 2,048 tokens the project could see all annotated answers in the dataset.

Warm encoding covers all queries and passages from prepared token IDs. The 2,048-token row reuses the same queries and includes their encoding time. Loading and warmup took 112.8 s for the project, 94.7 s for FP16 and 95.3 s for INT8 in this run. [Matched scores, timing and source spans](../../docs/benchmarks/embeddinggemma-legal-512-20261121.json)

- **v2.3 Header-Anchored Chunking & Cutoff Parity (`parquet_pipeline.py`):** On 175-query `LegalBench-RAG` chunked windows, `header_anchored` chunking lifted `span_ndcg@10` from **`0.3803 -> 0.5054` (`+32.9%`)** (`char_recall@10` `0.7209 -> 0.8114`; `auto_snap_header` reached `0.4958`), with boundary cutoff `min_cos = 1.0`. Under v2.2 `use_nocache=False` stays default after `use_nocache=True` regressed M4 Pro runtime (`23.79 s -> 27.12 s`). [v2.3 validation](../../docs/benchmarks/v23-validation-20261121.json)

### 2. Corpus Speed, `B=1` Latency & Silicon Wins

| Benchmark (`benchmark_monorepo.py` / ABBA OCR) | Base M5 (`24 GB`, `macOS 27.2.1`) | M4 Pro (`48 GB`, `macOS 26.6`) | Key Architectural Takeaway |
|---|---|---|---|
| **Corpus throughput (`hetero`)** | **48,803 tokens/s** (2.76 GB) | **38,517 tokens/s** (3.26 GB) | Apple CoreAI BF16 encoded the same 551,889 tokens at 12,482 tokens/s on M5. |
| **Single-query latency (ANE)** | **2.58 ms** | **2.11 ms** | A 24-token query uses the 32-token ANE graph. The Apple CoreAI BF16 graph took 9.38 ms on M5. |
| **Input packing in `hetero2` and Apple CoreAI** | Dynamic FFD packing avoids **`+73%` pad position waste** (`2.25M -> 3.90M` positions in static `blk_mx` `128/512/2048` `.aimodel`) while co-executing GPU + `18.56 TF` peak 16-core ANE (`2.75x` vs. GPU-only on OCR). | | `ANE=2288, GPU=0, CPU=0` across all `9` buckets on both M4 Pro and M5 `h17` (NCHW `out_h` + NEON untile + `>= 2^-14` RMSNorm epsilon floor). |
| **Base M5 `Apple10` GPU Tuning** | Native `mx.matmul` plain projections + fused GeGLU/PLE cut `7,974`-passage wall time **`-14.89%`** (`54.28 -> 46.20 s`, `1.175x`, identical Top-1/Top-5 retrieval); `Apple10`-gated `force_fused` FP16 SWA `D256` SDPA (`S in {384, 512}`) trims an extra **`-2.27%`**. | | Automatically gated on `Apple10` hardware capabilities in [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py). |

In its separate natural-text run, stock `mlx-vlm` encoded 64.91 documents/s on M5 and 46.05 on M4 Pro. Its single query took 8.48 ms and 8.39 ms, respectively.

`FluidInference/embeddinggemma-2-coreml` and `OpenInfinity/embeddinggemma-2-coreml-int8` encoded the same 551,889 token IDs in 2,000 documents as `fast-embeddinggemma-2`. `pack_256` groups up to eight short inputs into 256 tokens. Longer inputs use single-input graphs. The test timed the modes in packed, single, single, packed order. Query latency is the median of 30 calls using the same 24 tokens.

| Host | Public Core ML export | Query latency | Packed tokens/s | Single-input tokens/s | Packed / single |
|---|---|---:|---:|---:|---:|
| Base M5 | FluidInference FP16 | 4.334 ms | 14,187 | 13,556 | 1.0465× |
| Base M5 | OpenInfinity INT8 | 2.592 ms | 14,945 | 14,438 | 1.0351× |
| M4 Pro | FluidInference FP16 | 2.696 ms | 14,251 | 13,702 | 1.0401× |
| M4 Pro | OpenInfinity INT8 | 2.590 ms | 14,294 | 13,764 | 1.0385× |

Packing increased corpus throughput by 3.5–4.7%. Across all 2,000 documents on each Mac, the lowest cosine similarity between packed and single-input embeddings was 0.99944 for FP16 and 0.99973 for INT8. Core ML inspected seven functions in each export. Only `embed_128` exposed a Neural Engine preference, for 3,862 operations. The remaining 45,223 operations had no classified device, including all operations in `pack_256`.

The run records include the publisher revisions, per-function compute plans and per-task scores. [Benchmark methods](../../docs/BENCHMARKS.md#public-core-ml-results) · [Measured results (JSON)](../../docs/benchmarks/coreml-20261120.json)

## At a glance

| Question | Answer |
|---|---|
| How close to float reference (`64`-doc fixture)? | `0.999998` min cos (`GPU` vs. `F16`), `0.999889` (`GPU` vs. `BF16`), `0.999728` (`Hetero` vs. `F16`), `0.999887` (`ANE B=1 S=24`). |
| How fast is one query in the 32-token ANE graph? | **2.11 ms** in Python on M4 Pro and **2.58 ms** on M5. The standalone `emb2-ane-cli` took 1.871 ms on its M4 Pro fixture. |
| How fast is bulk `hetero` throughput? | **`48,803 tok/s`** on Base M5 and **`38,517 tok/s`** on M4 Pro (`monorepo_bench`); standalone `n=2,000` M4 Pro reaches **`36,911 tok/s`** synthetic and **`32,569 tok/s`** Document OCR (**`2.75x`** GPU OCR). |
| Does Matryoshka (`dimensions`) work everywhere? | Yes: `128, 256, 512, 768` (or any `1..768`) in Python, `parquet --dimensions`, and `/v1/embeddings` `"dimensions"`, all L2 re-normalized. |

## Parity (`64`-doc fixture, `embeddinggemma-2 verify`)

| Comparison | Min Cosine | Notes |
|---|---:|---|
| `GPU` vs. `F16` reference | `0.999998` | Mean `0.999999` |
| `GPU` vs. `BF16` reference | `0.999889` | Full `768`-d normalized match |
| `Hetero` vs. `F16` reference | `0.999728` | Dynamic `GPU + ANE` (`hetero2`) split |
| `Single-query B=1 S=24 ANE` vs. `GPU` | `0.999887` | `emb2_pk_s32_w8.mlmodelc` (`emb2-ane-cli`: `1.871 ms`) |

## Standalone Mode Breakdown (`embeddinggemma-2 bench --n 2000`, M4 Pro `48 GB`)

| Workload (Tokens) | Mode | Elapsed | Tok/s | Docs/s | Doc Split (`GPU / ANE / CPU`) | Token Split (`GPU / ANE / CPU`) |
|---|---|---:|---:|---:|---|---|
| Synthetic (`551,889`) | **`hetero2` (`GPU+ANE`)** | **14.952 s** | **36,911** | **133.8** | `910 / 1,090 / 0` | `256,316 / 295,573 / 0` |
| Synthetic (`551,889`) | `hetero` (`GPU+ANE+CPU`) | `17.639 s` | `31,288` | `113.4` | `1,032 / 715 / 253` | `286,755 / 197,377 / 67,757` |
| Synthetic (`551,889`) | `ane` | `27.406 s` | `20,138` | `73.0` | `0 / 2,000 / 0` | `0 / 551,889 / 0` |
| Synthetic (`551,889`) | `gpu` | `33.376 s` | `16,535` | `59.9` | `2,000 / 0 / 0` | `551,889 / 0 / 0` |
| Document OCR (`559,737`) | **`hetero2` (`GPU+ANE`)** | **17.186 s** | **32,569** | **116.4** | `730 / 1,270 / 0` | `212,621 / 347,116 / 0` |
| Document OCR (`559,737`) | `hetero` (`GPU+ANE+CPU`) | `19.523 s` | `28,671` | `102.4` | `957 / 805 / 238` | `271,748 / 222,921 / 65,068` |
| Document OCR (`559,737`) | `ane` | `27.136 s` | `20,627` | `73.7` | `64 / 1,936 / 0` | `37,609 / 522,128 / 0` |
| Document OCR (`559,737`) | `gpu` | `47.257 s` | `11,844` | `42.3` | `2,000 / 0 / 0` | `559,737 / 0 / 0` |

## What keeps it fast

| Engine | Optimization | Source |
|---|---|---|
| **GPU (`Apple9` / `Apple10 NAX`)** | Native `mx.matmul` projections + fused GeGLU (`_MPP_GEGLU_64`) + PLE (`_MPP_PG_64`), fused `[B,H,L,D]` QKNorm+VNorm+RoPE, `Apple10`-gated `force_fused` FP16 SWA `D256` SDPA (`S in {384, 512}`), `fast_pack_cost_aware` + `ceil128` (`<= 2048`) | [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py) |
| **ANE (`W8A16`, `CPU=0`)** | Packed buckets `emb2_pk_s{32,64,128,192,256,320,384,448,512}_w8.mlmodelc` (`ANE=2288, GPU=0, CPU=0` on M4 & M5 `h17`, `pipe_depth=4`), NCHW `[1,512,1,S]` `out_h` + NEON untile, `>= 2^-14` RMSNorm floor, `cblas_sgemm` (`512->768`) | [`native/emb2_coreml_runner.mm`](native/emb2_coreml_runner.mm), [`native/emb2_fast_io.cpp`](native/emb2_fast_io.cpp), [`native/emb2_neon_transpose.hpp`](native/emb2_neon_transpose.hpp) |
| **CPU-SME (Opt-in)** | Mixed `FP16` matmul / `FP32` norm+softmax CoreML graphs (`s{256,384,512}`) for optional 3-way `hetero3` | [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

## CLI, Parquet & Reproduce

```bash
# Chunked Parquet ingestion into shards (`--mode hetero` = 2-way GPU+ANE [`hetero2` alias]; `--mode hetero3` opts into +CPU-SME; `--max-chunk-tokens 512` keeps ANE buckets active)
.venv/bin/embeddinggemma-2 parquet \
  --input ocr-text.parquet --output embeddings/ --text-col body \
  --mode hetero --dimensions 512 --document-prefix \
  --chunk --max-chunk-tokens 512 --overlap 64 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --drop-text-col --include-chunk-text

# HTTP OpenAI server (/v1/embeddings with optional "dimensions")
.venv/bin/embeddinggemma-2 serve --host 127.0.0.1 --port 8080 --dimensions 256

# Rebuild ANE bundles (optional [ane-build] extra) + run full suite
python scripts/build_ane_models.py --mode all
.venv/bin/pytest -v   # parity, 9 ANE buckets 32..512 (CPU=0), SME/CPU, emb2-ane-cli, Parquet, HTTP
```

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Apple Base M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only, `40/40` tests passing with `ANE=2288, CPU=0` across all `9` buckets)**. Rebuilding `.mlmodelc` bundles requires optional `coremltools>=8.0`.
- **Not tested:** Sequences `> 2,048` tokens (model limit), other Apple Silicon chips, or concurrent GPU/ANE contention across processes.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
