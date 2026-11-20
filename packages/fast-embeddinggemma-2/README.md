# fast-embeddinggemma-2

**2,048-context Matryoshka embeddings at `38.5K–48.8K tok/s` (`4.33x–4.80x` vs. stock `MLX`, `6.24x` vs. Apple `CoreAI` `.aimodel`) and `~2 ms` latency on Apple Silicon, with no PyTorch runtime required.**

`fast-embeddinggemma-2` executes `google/embeddinggemma-2` (`300M`, `128..768`-d MRL) concurrently across Metal GPU and the 16-core Apple Neural Engine (`hetero2` dynamic FFD packing) — cutting single-query latency to **`2.11 ms`** on M4 Pro (**`3.98x`** vs. `mlx-vlm`) / **`2.58 ms`** on Base M5 (**`3.29x`** vs. `mlx-vlm`, **`3.64x`** vs. `CoreAI`), and reaching **`48,803 tok/s`** on Base M5 (`1.27x` M4 Pro's `38,517 tok/s`) while matching BF16 retrieval quality (`S_obj,e2e = 0.5376`, `0.999998` GPU / `0.999728` hetero cosine parity). Ships with a Python API, chunked OCR-aware Parquet ingestion, an OpenAI `/v1/embeddings` server (`"dimensions"` support), and the zero-Python `emb2-ane-cli`.

> **Full Cross-Silicon Benchmarks:** See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete M4 Pro vs. Base M5 retrieval, throughput, and Core AI `.aimodel` comparisons.

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

`EOT = (1 / T_e2e) * sum(valid_i * score_i)` in valid accurate units/s (`benchmark_outcomes.py`, isolated subprocesses, 4-bucket `RUSAGE_INFO_V6` memory):

| Host | Engine | `S_obj,e2e` | `EOT (u/s)` | Wall Time | 4-Bucket Mem | Speedup vs. Stock |
|---|---|---:|---:|---:|---:|---|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5376`** | **`4.707`** | **`57.1 s`** | **`3.43 GB`** | **`4.33x` MLX · `6.24x` CoreAI** (`1.31x` M4 Pro) |
| Base M5 (`24 GB`) | Stock `mlx-vlm` (`embeddinggemma-2-8bit`) | `0.5364` | `1.086` | `246.9 s` | `19.41 GB` | `1.00x` MLX |
| Base M5 (`24 GB`, `macOS 27.2+`) | Stock Apple `CoreAI` (`coreai-emb2-bf16` `.aimodel`) | `0.5360` | `0.754` | `355.5 s` | `3.04 GB` *(V6)* | `0.69x` MLX (`14,521` GPU / `0` ANE) |
| **M4 Pro (`48 GB`, `Apple9`)** | **`fast-embeddinggemma-2` (`hetero`)** | **`0.5365`** | **`3.583`** | **`74.9 s`** | **`2.49 GB`** | **`4.80x` MLX** |
| M4 Pro (`48 GB`) | Stock `mlx-vlm` (`embeddinggemma-2-8bit`) | `0.5353` | `0.746` | `358.6 s` | `39.03 GB` | `1.00x` MLX |

### 2. Corpus Speed, `B=1` Latency & Silicon Wins

| Benchmark (`benchmark_monorepo.py` / ABBA OCR) | Base M5 (`24 GB`, `macOS 27.2.1`) | M4 Pro (`48 GB`, `macOS 26.6`) | Key Architectural Takeaway |
|---|---|---|---|
| **Corpus Throughput (`hetero`)** | **`48,803 tok/s`** (`2.76 GB`) vs. `16,094` `mlx_vlm` (**`3.03x`**) / `12,482` `CoreAI` (**`3.91x`**) | **`38,517 tok/s`** (`3.26 GB`) vs. `11,419` `mlx_vlm` (**`3.37x`**) | `10c` Base M5 `NAX` beats `20c` M4 Pro by **`1.27x`** (`1.31x` `EOT`). |
| **Single-Query Latency (`B=1` ANE)** | **`2.58 ms`** vs. `8.48 ms` `mlx_vlm` (**`3.29x`**) / `9.38 ms` `CoreAI` (**`3.64x`**) | **`2.11 ms`** vs. `8.39 ms` `mlx_vlm` (**`3.98x`**) | Pure-ANE `S=32` bucket (`emb2-ane-cli`: `1.871 ms` on M4 Pro). |
| **Why `hetero2` beats stock `.aimodel`** | Dynamic FFD packing avoids **`+73%` pad position waste** (`2.25M -> 3.90M` positions in static `blk_mx` `128/512/2048` `.aimodel`) while co-executing GPU + `18.56 TF` peak 16-core ANE (`2.75x` vs. GPU-only on OCR). | | `ANE=2288, GPU=0, CPU=0` across all `9` buckets on both M4 Pro and M5 `h17` (NCHW `out_h` + NEON untile + `>= 2^-14` RMSNorm epsilon floor). |
| **Base M5 `Apple10` GPU Tuning** | Native `mx.matmul` plain projections + fused GeGLU/PLE cut `7,974`-passage wall time **`-14.89%`** (`54.28 -> 46.20 s`, `1.175x`, identical Top-1/Top-5 retrieval); `Apple10`-gated `force_fused` FP16 SWA `D256` SDPA (`S in {384, 512}`) trims an extra **`-2.27%`**. | | Automatically gated on `Apple10` hardware capabilities in [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py). |

See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete tables and public baseline details.

## At a glance

| Question | Answer |
|---|---|
| How close to float reference (`64`-doc fixture)? | `0.999998` min cos (`GPU` vs. `F16`), `0.999889` (`GPU` vs. `BF16`), `0.999728` (`Hetero` vs. `F16`), `0.999887` (`ANE B=1 S=24`). |
| How fast is single query (`B=1 S=32`)? | **`1.871 ms`** standalone `emb2-ane-cli` / **`2.11 ms`** Python ANE on M4 Pro (**`3.98x`** vs. `mlx_vlm`); **`2.58 ms`** on Base M5 (**`3.29x`** vs. `mlx_vlm`, **`3.64x`** vs. `CoreAI`). |
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
