# fast-bge-small

**Sub-millisecond text embeddings, `187K–201K tok/s` cross-silicon indexing (`9.58x–11.26x` higher `EOT` than stock `mlx-embeddings`), and zero `torch` or `transformers` runtime dependencies.**

`fast-bge-small` runs `BAAI/bge-small-en-v1.5` (`33M`, `384`-d) by routing short queries onto the 16-core Apple Neural Engine (**`0.49 ms`** on M4 Pro / **`0.79 ms`** on Base M5 `B=1`) and splitting bulk workloads concurrently across Metal GPU and ANE (**`200,646 tok/s`** on Base M5, **`187,150 tok/s`** on M4 Pro, and up to **`215,022 tok/s`** standalone Document OCR) while matching float16 reference cosine similarity to **`0.999998`**. Use it from Python, an OpenAI-compatible `/v1/embeddings` server, a resumable OCR-aware Parquet CLI, or the standalone zero-Python `bge-ane-cli` binary.

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

`EOT = (1 / T_e2e) * sum(valid_i * score_i)` in valid accurate units/s (`benchmark_outcomes.py`, isolated subprocesses, 4-bucket `RUSAGE_INFO_V6` memory):

| Host | Engine | `S_obj,e2e` | `EOT (u/s)` | Wall Time | 4-Bucket Mem | Speedup vs. Stock `MLX` |
|---|---|---:|---:|---:|---:|---:|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`fast-bge-small` (`hetero`)** | `0.4737` | **`13.061`** | **`11.8 s`** | **`1.54 GB`** | **`11.26x`** (`1.03x` M4 Pro) |
| Base M5 (`24 GB`) | Stock `mlx-embeddings` (`bge-small-en-v1.5-8bit`) | `0.4774` | `1.160` | `133.8 s` | `18.78 GB` | `1.00x` |
| **M4 Pro (`48 GB`, `Apple9`)** | **`fast-bge-small` (`hetero`)** | `0.4737` | **`12.634`** | **`12.2 s`** | **`1.55 GB`** | **`9.58x`** |
| M4 Pro (`48 GB`) | Stock `mlx-embeddings` (`bge-small-en-v1.5-8bit`) | `0.4774` | `1.319` | `117.6 s` | `38.99 GB` | `1.00x` |

### 2. Throughput, Latency & Parallel Multi-Bucket CoreML Build

| Benchmark (`benchmark_monorepo.py` / `benchmark_capabilities.py`) | Base M5 (`24 GB`, `macOS 27.2.1`) | M4 Pro (`48 GB`, `macOS 26.6`) | Takeaway |
|---|---|---|---|
| **Corpus Throughput (`hetero`)** | **`200,646 tok/s`** (`1.85 GB`) vs. `26,667` `mlx-embeddings` (**`7.52x`**) | **`187,150 tok/s`** (`2.09 GB`) vs. `30,694` `mlx-embeddings` (**`6.10x`**) | `10c` Base M5 `NAX` leads `20c` M4 Pro by **`1.07x`**; wider 128-col FP16 tiles cut Base M5 `7,974`-passage wall time `-10.64%` (`9.54s -> 8.53s`). |
| **Single-Query Latency (`B=1` ANE)** | **`0.79 ms`** vs. `4.42 ms` `mlx-embeddings` (**`5.59x`**) | **`0.49 ms`** vs. `4.85 ms` `mlx-embeddings` (**`9.90x`**) | M4 Pro's lower ANE dispatch overhead wins `B=1` (`0.49 ms` vs. `0.79 ms`). |
| **Parallel CoreML Build (`9` buckets)** | `jobs=1` `5.12 s` -> **`jobs=4` `2.45 s` (`2.09x`)** | Legacy `26.80 s` (`jobs=1` `9.18 s`) -> **`jobs=4` `4.56 s` (`2.01x` / `5.88x` vs. legacy)** | `spawn`-only pool + `skip_model_load=True` in [`scripts/build_ane_models.py`](scripts/build_ane_models.py), bit-identical (`cos = 1.000000`, `9/9` ANE). |

See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for full `gpu`/`ane`/`hetero` ablations and public baseline methodology.

## At a glance

| Question | Answer |
|---|---|
| How close to float reference? | `0.999998` min cos (`GPU` vs. `F16` and `BF16`); `0.999984` (`Hetero` vs. `F16`); `0.999986` (`ANE` vs. `GPU` `B=1 S=24`). |
| How fast is single query (`B=1 S=24`)? | **`0.49 ms`** ANE median on M4 Pro (**`9.90x`** vs. `mlx-embeddings`, `0.502 ms` in zero-Python `bge-ane-cli`); **`0.79 ms`** on Base M5 (**`5.59x`** vs. `mlx-embeddings`). |
| How fast is bulk throughput (`hetero`)? | **`200,646 tok/s`** on Base M5 (**`7.52x`** vs. `mlx-embeddings`) and **`187,150 tok/s`** on M4 Pro (**`6.10x`** vs. `mlx-embeddings`); standalone `n=4,000` M4 Pro reaches **`215,022 tok/s`** OCR (`4.566 s`) and **`189,338 tok/s`** synthetic (`5.250 s`). |
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
| **Work-Stealing Scheduler & Parallel Build** | GPU consumes longest sequences forward while ANE pulls shortest sequences backward with zero contention; `spawn`-parallel 9-bucket CoreML build (`--jobs 4`) | [`src/fast_bge_small/`](src/fast_bge_small/), [`scripts/build_ane_models.py`](scripts/build_ane_models.py) |

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
