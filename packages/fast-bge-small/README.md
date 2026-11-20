# fast-bge-small

**Sub-millisecond text embeddings and 183K–215K tok/s corpus indexing on a Mac — with zero `torch` or `transformers` runtime dependencies.**

`fast-bge-small` runs `BAAI/bge-small-en-v1.5` (`33M`, `384`-d) by routing short queries onto the 16-core Apple Neural Engine (**0.49–0.56 ms** `B=1`) and splitting bulk workloads concurrently across Metal GPU and ANE (**183,015 tok/s** and **2.24x** faster than PyTorch MPS in `monorepo_bench`; **215,022 tok/s** standalone Document OCR) while matching float16 reference cosine similarity to **0.999998**. Use it from Python, an OpenAI-compatible `/v1/embeddings` server, a resumable OCR-aware Parquet CLI, or the standalone zero-Python `bge-ane-cli` binary.

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

## At a glance

| Question | Answer |
|---|---|
| How close to float reference? | `0.999998` min cos (`GPU` vs `F16` and `BF16`); `0.999984` (`Hetero` vs `F16`); `0.999986` (`ANE` vs `GPU` `B=1 S=24`). |
| How fast is single query (`B=1 S=24`)? | **`0.493 ms`** ANE median (synth) / **`0.560 ms`** ANE median (OCR) / **`0.506 ms`** (`monorepo_bench`, **`3.92x`** faster than `1.985 ms` GPU; `0.502 ms` in zero-Python `bge-ane-cli`). |
| How fast is bulk throughput (`hetero`)? | **`183,015 tok/s`** (`monorepo_bench n=1,000`, **`2.24x`** vs `transformers-mps-float16`, **`1.58x`** vs GPU, `1.186 GB` Metal / `2.081 GB` OS); standalone `n=4,000` up to **`215,022 tok/s`** OCR (`4.566 s`) and **`189,338 tok/s`** synth (`5.250 s`). |
| Requires `torch` or `transformers`? | No at runtime (`mlx`, `numpy`, `tokenizers`, `pyarrow`, `huggingface-hub` only; `coremltools >= 8.0` only to rebuild `.mlmodelc`). |

## Parity (`64`-doc fixture, `bge-small verify`)

| Comparison | Min Cosine | Mean Cosine / Notes |
|---|---:|---|
| `GPU` vs `F16` reference | `0.999998` | Mean `0.999999` across `64` docs |
| `GPU` vs `BF16` reference | `0.999998` | Matches BF16 reference |
| `Hetero` (`GPU + ANE`) vs `F16` | `0.999984` | Two-ended bucket routing |
| `Single-query B=1 S=24 ANE` vs `GPU` | `0.999986` | `build/bge-ane-cli`: `0.502 ms` (`tokens=16`, `iters=30`) |

## Speed (`bge-small bench`, M4 Pro `48 GB`)

### 1. Standalone (`bge-small bench --n 4000`)

#### Single-Query Latency (`B=1`, `S=24`)

| Workload | Backend | Median | Min | Notes |
|---|---|---:|---:|---|
| Synthetic | `bge-ane-cli` (zero-Python) | **`0.502 ms`** | — | `tokens=16`, `iters=30` |
| Synthetic | `ane` (`ANE_B1_S24`) | **`0.493 ms`** | `0.475 ms` | **`3.93x`** faster than GPU |
| Synthetic | `gpu` (`GPU_B1_S24`) | `1.937 ms` | `1.843 ms` | Metal 4 MPP |
| Document OCR | `ane` (`ANE_B1_S24`) | **`0.560 ms`** | `0.485 ms` | **`7.26x`** faster than GPU |
| Document OCR | `gpu` (`GPU_B1_S24`) | `4.067 ms` | `2.227 ms` | Metal 4 MPP |

#### Corpus Throughput (`n = 4,000` docs)

| Workload (Tokens) | Mode | Elapsed | Throughput | GPU Docs (`Tokens`) | ANE Docs (`Tokens`) |
|---|---|---:|---:|---:|---:|
| Synthetic (`993,944`) | **`hetero`** | **5.250 s** | **189,338 tok/s** | `1,504` (`577,600`) | `2,496` (`416,344`) |
| Synthetic (`993,944`) | `gpu` | `7.185 s` | `138,335 tok/s` | `4,000` (`993,944`) | `0` (`0`) |
| Synthetic (`993,944`) | `ane` | `16.772 s` | `59,261 tok/s` | `0` (`0`) | `4,000` (`993,944`) |
| Document OCR (`981,829`) | **`hetero`** | **4.566 s** | **215,022 tok/s** | `1,856` (`632,487`) | `2,144` (`349,342`) |
| Document OCR (`981,829`) | `gpu` | `8.249 s` | `119,028 tok/s` | `4,000` (`981,829`) | `0` (`0`) |
| Document OCR (`981,829`) | `ane` | `16.272 s` | `60,339 tok/s` | `0` (`0`) | `4,000` (`981,829`) |

### 2. Monorepo Showcase & Public Baseline (`benchmark_monorepo.py`, `n = 1,000` / `249,303` toks)

Recorded alongside `4` host background quiescence warnings (`RUSAGE_INFO_V4` OS footprint):

#### Single-Query Latency (`B=1`, `S=24`)

| Engine / Runtime | Mode | Median | Min | Speedup vs Public MPS |
|---|---|---:|---:|---:|
| **`fast-bge-small`** | **`ane`** | **0.506 ms** | `0.464 ms` | **`14.67x`** (`3.92x` vs GPU) |
| `fast-bge-small` | `gpu` | `1.985 ms` | `1.881 ms` | `3.74x` |
| `transformers-mps-float16` (Public) | PyTorch MPS | `7.425 ms` | — | `1.00x` (Baseline) |

#### Corpus Throughput (`n = 1,000` docs, `249,303` toks)

| Engine / Runtime | Mode | Elapsed | Throughput | Doc Split (`GPU / ANE`) | Token Split (`GPU / ANE`) | Metal Peak | OS Footprint | vs Public MPS | Notes |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| **`fast-bge-small`** | **`hetero`** | **1.362 s** | **183,015 tok/s** | `—` | `—` | `1.186 GB` *(suite)* | **2.081 GB** *(suite)* | **`2.24x`** | **`1.58x`** vs `gpu` (`1.583x`) |
| `fast-bge-small` | `gpu` | `2.156 s` | `115,632 tok/s` | `—` | `—` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | `1.42x` | Metal-only baseline |
| `fast-bge-small` | `ane` | `4.211 s` | `59,209 tok/s` | `0 / 1,000` | `0 / 249,303` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | `0.73x` | `237.5 doc/s`; ANE-only baseline |
| `transformers-mps-float16` (Public) | PyTorch MPS | — | `81,592.6 tok/s` | — | — | `0.071 GB` | `3.532 GB` | `1.00x` | `327.28 docs/s`; `load=1.424 s` |

Parity (`verify`, `1.897 s`, `1.949 GB` OS): `64-doc` `GPU_vs_F16` `min=0.999998` (`mean=0.999999`), `Hetero_vs_F16` `min=0.999984`, `Single-query B=1 S=24 ANE_vs_GPU` `cos=0.999986`.

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **16-core ANE Runner** | `MLComputeUnitsCPUAndNeuralEngine`, NEON FP16 add, `8x8` register transpose, Layer-11 `[CLS]` slice, 4-slot async ring | [`native/bge_ane_runner.mm`](native/bge_ane_runner.mm) |
| **20-core Metal 4 GPU** | Packed varlen batching (0 attn pad), custom `D=32` bidirectional FlashAttention, `MPP::matmul2d` fused GELU/residual, single-pass LN+L2 | [`src/fast_bge_small/gpu.py`](src/fast_bge_small/gpu.py) |
| **Work-Stealing Scheduler** | GPU consumes longest sequences forward while ANE pulls shortest sequences backward with zero contention | [`src/fast_bge_small/`](src/fast_bge_small/) |

## CLI, Parquet & Reproduce

```bash
.venv/bin/bge-small embed "first passage" "second passage" --json
.venv/bin/bge-small parquet --input in.parquet --output out/ --text-col body \
  --dtype float32 --chunk --max-body-tokens 510 --overlap 51 --filter-ocr --resume
.venv/bin/bge-small serve --host 127.0.0.1 --port 8080
.venv/bin/bge-small bench --mode all --n 4000
.venv/bin/pytest -v   # 7 passed (native bge-ane-cli, parity, ANE buckets 16..500, Parquet OCR, HTTP)
```

Parquet options also include `--ocr-pages-only`, `--per-ocr-page` (UTF-8 `page_offsets`), directory shard `--resume`, and `float16` output vectors.

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6, Xcode SDK 27.2, `Apple clang 21.0.0`)** producing `0`-warning native builds (`libbge_ane_runner.dylib`, `bge-ane-cli`, plus optional `libbge_cpu_query.dylib` via `make cpu-query`).
- **Not tested:** Other Apple Silicon generations/RAM tiers, Linux/x86/CUDA, or sequences exceeding the model's `512`-token context window.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
