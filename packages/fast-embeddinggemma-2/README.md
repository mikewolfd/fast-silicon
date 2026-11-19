# fast-embeddinggemma-2

Hardware-limit inference engine for **google/embeddinggemma-2** (`300M`, `768`-d, `2048`-context, Matryoshka `128..768`) executing across Apple Silicon M4 Pro's **20-core Metal 4 GPU**, **16-core Neural Engine (`W8A16`)**, and **14-core ARMv9 CPU-SME**.
Zero runtime `transformers` or `torch` dependencies, with a standalone C++/CoreML CLI (`build/emb2-ane-cli`) for zero-Python ANE inference.

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

## Speed (`embeddinggemma-2 bench`, `n = 2,000` docs, M4 Pro `48 GB`)

Single-query latency: `ANE_B1_S32` median `2.095 ms` (min `1.899 ms` synth) / `2.066 ms` (min `1.868 ms` OCR) vs `GPU_B1_S64` median `9.026 ms` / `8.895 ms`.

| Workload | Mode | Elapsed (s) | Tok/s | Doc/s | Split (`gpu / ane / cpu` docs & toks) |
|---|---|---:|---:|---:|---|
| Synthetic (`551,889` toks) | `gpu` | 33.376 | 16,535 | 59.9 | 2,000 / 0 / 0 (`551,889` / 0 / 0) |
| Synthetic (`551,889` toks) | `ane` | 27.406 | 20,138 | 73.0 | 0 / 2,000 / 0 (0 / `551,889` / 0) |
| Synthetic (`551,889` toks) | **`hetero2` (`GPU+ANE`)** | **14.952** | **36,911** | **133.8** | 910 / 1,090 / 0 (`256,316` / `295,573` / 0) |
| Synthetic (`551,889` toks) | `hetero` (`GPU+ANE+CPU`) | 17.639 | 31,288 | 113.4 | 1,032 / 715 / 253 (`286,755` / `197,377` / `67,757`) |
| Document OCR (`559,737` toks) | `gpu` | 47.257 | 11,844 | 42.3 | 2,000 / 0 / 0 (`559,737` / 0 / 0) |
| Document OCR (`559,737` toks) | `ane` | 27.136 | 20,627 | 73.7 | 64 / 1,936 / 0 (`37,609` / `522,128` / 0) |
| Document OCR (`559,737` toks) | **`hetero2` (`GPU+ANE`)** | **17.186** | **32,569** | **116.4** | 730 / 1,270 / 0 (`212,621` / `347,116` / 0) |
| Document OCR (`559,737` toks) | `hetero` (`GPU+ANE+CPU`) | 19.523 | 28,671 | 102.4 | 957 / 805 / 238 (`271,748` / `222,921` / `65,068`) |

### Monorepo Showcase & Parity (`benchmark_monorepo.py`, M4 Pro `48 GB`)

Verified in the unified monorepo runner (`scripts/benchmark_monorepo.py --packages fast-embeddinggemma-2`, `n=2,000` docs / `551,889` tokens, Metal Peak `2.224 GB`, OS Footprint `10.820 GB` via `RUSAGE_INFO_V4`):

| Workload / Mode | Latency (`B=1`) | Time (`s`) | Throughput | Docs/s | Doc Split (`GPU / ANE / CPU`) |
|---|---|---:|---:|---:|---|
| **`hetero2` (`GPU+ANE`)** | ANE (`S=32`) **2.188 ms** (min `1.943`) | **17.363** | **31,786 tok/s** | **115.2** | `591 / 1,409 / 0` (**`1.85x`** `gpu`) |
| `hetero` (`GPU+ANE+CPU`) | — | 17.621 | 31,319 tok/s | 113.5 | `650 / 1,322 / 28` |
| `ane` (16-core ANE) | ANE (`S=32`) `2.188 ms` | 27.149 | 20,328 tok/s | 73.7 | `0 / 2,000 / 0` (**`4.36x`** lower `B=1` vs GPU) |
| `gpu` (Metal 4 MPP) | GPU (`S=64`) `9.540 ms` (min `9.089`) | 32.129 | 17,177 tok/s | 62.2 | `2,000 / 0 / 0` |

Parity (`verify`): `64-doc` `GPU_vs_F16` `min=0.999998` (`mean=0.999999`), `GPU_vs_BF16` `min=0.999894`, `Hetero_vs_F16` `min=0.999724`, `Single-query B=1 S=24 ANE_vs_GPU` `cos=0.999887`. Public `transformers-mps` comparison was skipped in this environment because installed `transformers` lacked `embedding_gemma2` support.

*Note: Standalone & monorepo runs were captured on Apple M4 Pro (`48 GB`) alongside light background sessions (`6` quiescence warnings recorded in the monorepo run manifest).*

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
