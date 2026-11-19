# fast-embeddinggemma-2

**2,048-context Matryoshka embeddings at 31K–37K tok/s and ~2 ms latency on Apple Silicon, with no PyTorch runtime required.**

`fast-embeddinggemma-2` executes `google/embeddinggemma-2` (`300M`, `128..768`-d MRL) across Apple M4 Pro's Metal GPU and 16-core Neural Engine — cutting single-query latency to **1.87–2.19 ms** (**4.36x** faster than GPU-only) and reaching **31,786 tok/s** (**36,911 tok/s** standalone) at **0.999997** GPU / **0.999705** hetero cosine parity. Ships with a Python API, chunked OCR-aware Parquet ingestion, an OpenAI `/v1/embeddings` server (`"dimensions"` support), and the zero-Python `emb2-ane-cli`.

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

## At a glance

| Question | Answer |
|---|---|
| How close to float reference (`64`-doc fixture)? | `0.999997` min cos (`GPU` vs `F16`), `0.999891` (`GPU` vs `BF16`), `0.999705` (`Hetero` vs `F16`), `0.999890` (`ANE B=1 S=24`). |
| How fast is single query (`B=1 S=32`)? | **`1.8713 ms`** standalone `emb2-ane-cli` (`1.899 / 1.868 ms` ANE min); **`2.188 ms`** ANE vs `9.540 ms` GPU (`S=64`) in `monorepo_bench` (**`4.36x`** lower latency). |
| How fast is bulk `hetero2` throughput (`n=2,000`)? | **`31,786 tok/s`** (`115.2 doc/s`, **`1.85x`** GPU, `2.224 GB` Metal / `10.820 GB` OS in `monorepo_bench`); standalone up to **`36,911 tok/s`** (`133.8 doc/s`) synth and **`32,569 tok/s`** (`116.4 doc/s`) Document OCR (**`3.1x`** GPU OCR). |
| Does Matryoshka (`dimensions`) work everywhere? | Yes: `128, 256, 512, 768` (or any `1..768`) in Python, `parquet --dimensions`, and `/v1/embeddings` `"dimensions"`, all L2 re-normalized. |

## Parity (`64`-doc fixture, `embeddinggemma-2 verify`)

| Comparison | Min Cosine | Notes |
|---|---:|---|
| `GPU` vs `F16` reference | `0.999997` | Mean `0.999999` |
| `GPU` vs `BF16` reference | `0.999891` | Full `768`-d normalized match |
| `Hetero` vs `F16` reference | `0.999705` | Dynamic `GPU + ANE + CPU-SME` split |
| `Single-query B=1 S=24 ANE` vs `GPU` | `0.999890` | `emb2_pk_s32_w8.mlmodelc` (`emb2-ane-cli`: `1.8713 ms`) |

## Speed (`embeddinggemma-2 bench`, M4 Pro `48 GB`)

### 1. Standalone (`embeddinggemma-2 bench --n 2000`)

#### Single-Query Latency (`B=1`)

| Workload | Backend / Tool | Seq | Median | Min |
|---|---|---:|---:|---:|
| Synthetic | `emb2-ane-cli` (zero-Python) | `32` | **`1.871 ms`** | — |
| Synthetic | `ane` (`ANE_B1_S32`) | `32` | **`2.095 ms`** | `1.899 ms` |
| Synthetic | `gpu` (`GPU_B1_S64`) | `64` | `9.026 ms` | — |
| Document OCR | `ane` (`ANE_B1_S32`) | `32` | **`2.066 ms`** | `1.868 ms` |
| Document OCR | `gpu` (`GPU_B1_S64`) | `64` | `8.895 ms` | — |

#### Corpus Throughput (`n = 2,000` docs)

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

### 2. Monorepo Showcase & Parity (`benchmark_monorepo.py`, `n = 2,000` / `551,889` toks)

Captured alongside `6` background quiescence warnings (`Metal Peak: 2.224 GB`, `OS Footprint: 10.820 GB` via `RUSAGE_INFO_V4`; public `transformers-mps` skipped because installed `transformers` lacked `embedding_gemma2` support):

#### Single-Query Latency (`B=1`)

| Mode | Seq | Median | Min | Speedup vs GPU |
|---|---:|---:|---:|---:|
| **`ane`** | `32` | **2.188 ms** | `1.943 ms` | **`4.36x`** |
| `gpu` | `64` | `9.540 ms` | `9.089 ms` | `1.00x` |

#### Corpus Throughput (`n = 2,000` docs, `551,889` toks)

| Mode | Elapsed | Tok/s | Docs/s | Doc Split (`GPU / ANE / CPU`) | Token Split (`GPU / ANE / CPU`) | Speedup vs GPU |
|---|---:|---:|---:|---|---|---:|
| **`hetero2` (`GPU+ANE`)** | **17.363 s** | **31,786** | **115.2** | `591 / 1,409 / 0` | `208,158 / 343,731 / 0` | **`1.85x`** |
| `hetero` (`GPU+ANE+CPU`) | `17.621 s` | `31,319` | `113.5` | `650 / 1,322 / 28` | `232,550 / 311,494 / 7,845` | `1.82x` |
| `ane` (16-core ANE) | `27.149 s` | `20,328` | `73.7` | `0 / 2,000 / 0` | `0 / 551,889 / 0` | `1.18x` |
| `gpu` (Metal 4 MPP) | `32.129 s` | `17,177` | `62.2` | `2,000 / 0 / 0` | `551,889 / 0 / 0` | `1.00x` |

Parity (`verify`): `64-doc` `GPU_vs_F16` `min=0.999998` (`mean=0.999999`), `GPU_vs_BF16` `min=0.999894`, `Hetero_vs_F16` `min=0.999724`, `Single-query B=1 S=24 ANE_vs_GPU` `cos=0.999887`.

## What keeps it fast

| Engine | Optimization | Source |
|---|---|---|
| **GPU (Metal 4)** | `MPP::matmul2d` (`tm=128/64`), fused GeGLU (`_MPP_GEGLU_64`) + PLE (`_MPP_PG_64`), fused `[B,H,L,D]` QKNorm+VNorm+RoPE (`_QKNORM_ROPE_{256,512}_BHLD`), `fast_pack_cost_aware` + `ceil128` (`<= 2048`) | [`src/fast_embeddinggemma_2/gpu.py`](src/fast_embeddinggemma_2/gpu.py) |
| **ANE (`W8A16`)** | Packed buckets `emb2_pk_s{32,64,128,192,256,320,384,448,512}_w8.mlmodelc` (`pipe_depth=4`), NEON/L1 gather, block-diagonal mask, `cblas_sgemm` (`512->768`) | [`native/emb2_coreml_runner.mm`](native/emb2_coreml_runner.mm), [`native/emb2_fast_io.cpp`](native/emb2_fast_io.cpp) |
| **CPU-SME** | Mixed `FP16` matmul / `FP32` norm+softmax CoreML graphs (`s{256,384,512}`) for optional 3-way `hetero` | [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

## CLI, Parquet & Reproduce

```bash
# Chunked Parquet ingestion into shards (`--max-chunk-tokens 512` keeps ANE buckets active; up to 2048 on GPU)
.venv/bin/embeddinggemma-2 parquet \
  --input ocr-text.parquet --output embeddings/ --text-col body \
  --mode hetero2 --dimensions 512 --document-prefix \
  --chunk --max-chunk-tokens 512 --overlap 64 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --drop-text-col --include-chunk-text

# HTTP OpenAI server (/v1/embeddings with optional "dimensions")
.venv/bin/embeddinggemma-2 serve --host 127.0.0.1 --port 8080 --dimensions 256

# Rebuild ANE bundles (optional [ane-build] extra) + run full suite
python scripts/build_ane_models.py --mode all
.venv/bin/pytest -v   # 32 passed in 26.93s (parity, buckets 32..512, SME/CPU, emb2-ane-cli, Parquet, HTTP)
```

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6)**. Rebuilding `.mlmodelc` bundles requires optional `coremltools>=8.0`.
- **Not tested:** Sequences `> 2,048` tokens (model limit), other Apple Silicon chips, or concurrent GPU/ANE contention across processes.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
