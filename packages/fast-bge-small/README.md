# fast-bge-small

Hardware-limit inference engine for **BAAI/bge-small-en-v1.5** (`33M`, `384`-d) on Apple Silicon **M4 Pro** combining 20-core Metal 4 `MPP::matmul2d` GPU kernels and 16-core Apple Neural Engine (`CoreML` / `IOSurface`) concurrent execution.
No `transformers`, `torch`, or Python runtime overhead is required for inference, with a standalone C++/CoreML binary (`build/bge-ane-cli`) for zero-Python single-query ANE execution.

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
| How fast is single query (`B=1 S=24`)? | **`0.493 ms`** ANE median (synth) / **`0.560 ms`** ANE median (OCR) / **`0.526 ms`** (`monorepo_bench`, **`3.83x`** faster than `2.015 ms` GPU; `0.502 ms` in zero-Python `bge-ane-cli`). |
| How fast is bulk throughput (`hetero`)? | **`207,264 tok/s`** (`monorepo_bench n=1,000`, **`2.50x`** vs `transformers-mps-float16`, **`1.52x`** vs GPU, `1.173 GB` Metal / `4.813 GB` OS); standalone `n=4,000` up to **`215,022 tok/s`** OCR (`4.566 s`) and **`189,338 tok/s`** synth (`5.250 s`). |
| Requires `torch` or `transformers`? | No at runtime (`mlx`, `numpy`, `tokenizers`, `pyarrow`, `huggingface-hub` only; `coremltools >= 8.0` only to rebuild `.mlmodelc`). |

## Parity (`64`-doc fixture, `bge-small verify`)

| Comparison | Min Cosine | Mean Cosine / Notes |
|---|---:|---|
| `GPU` vs `F16` reference | `0.999998` | Mean `0.999999` across `64` docs |
| `GPU` vs `BF16` reference | `0.999998` | Matches BF16 reference |
| `Hetero` (`GPU + ANE`) vs `F16` | `0.999984` | Two-ended bucket routing |
| `Single-query B=1 S=24 ANE` vs `GPU` | `0.999986` | `build/bge-ane-cli`: `0.502 ms` (`tokens=16`, `iters=30`) |

## Speed (`bge-small bench`, `n = 4,000` docs, M4 Pro `48 GB`)

Single-query `B=1 S=24` (`ANE` vs `GPU`):
- Synthetic: `ANE_B1_S24` median **`0.493 ms`** (min `0.475 ms`) vs `GPU_B1_S24` median `1.937 ms` (min `1.843 ms`).
- Document OCR: `ANE_B1_S24` median **`0.560 ms`** (min `0.485 ms`) vs `GPU_B1_S24` median `4.067 ms` (min `2.227 ms`).

Corpus throughput:

| Workload | Mode | Elapsed (s) | Throughput (tok/s) | Split (`gpu_docs / ane_docs`, tokens) |
|---|---|---:|---:|---|
| Synthetic (`993,944` toks) | `gpu` | 7.185 | 138,335 | 4,000 / 0 (`993,944` / `0`) |
| Synthetic (`993,944` toks) | `ane` | 16.772 | 59,261 | 0 / 4,000 (`0` / `993,944`) |
| Synthetic (`993,944` toks) | `hetero` | **5.250** | **189,338** | 1,504 / 2,496 (`577,600` / `416,344`) |
| Document OCR (`981,829` toks) | `gpu` | 8.249 | 119,028 | 4,000 / 0 (`981,829` / `0`) |
| Document OCR (`981,829` toks) | `ane` | 16.272 | 60,339 | 0 / 4,000 (`0` / `981,829`) |
| Document OCR (`981,829` toks) | `hetero` | **4.566** | **215,022** | 1,856 / 2,144 (`632,487` / `349,342`) |

### Monorepo Showcase & Public Baseline (`benchmark_monorepo.py`, M4 Pro `48 GB`)

Verified in the unified monorepo runner (`scripts/benchmark_monorepo.py --packages fast-bge-small`, `n=1,000` docs / `249,303` tokens, Metal Peak `1.173 GB`, OS Footprint `4.813 GB` via `RUSAGE_INFO_V4`):

| Workload / Mode | Latency (`B=1 S=24`) | Time (`s`) | Throughput | Notes / Split |
|---|---:|---:|---:|---|
| **`hetero` (Metal 4 + ANE)** | ANE **0.526 ms** (min `0.467`) | **1.203** | **207,264 tok/s** | `424` GPU / `576` ANE docs (**`1.52x`** `gpu`; **`2.50x`** vs `transformers-mps`) |
| `gpu` (Metal 4 MPP) | GPU `2.015 ms` (min `1.847`) | 1.828 | 136,416 tok/s | `1,000` GPU docs (`ANE B=1` is **`3.83x`** lower latency) |
| `ane` (CoreML ANE) | ANE `0.526 ms` | 4.388 | 56,808 tok/s | `1,000` ANE docs |
| `transformers-mps-float16` (Public) | `8.446 ms` | — | `82,883.2 tok/s` (`332.46 docs/s`) | OS Footprint `3.520 GB` (`load=1.314 s`) |

Parity (`verify`): `64-doc` `GPU_vs_F16` `min=0.999998` (`mean=0.999999`), `Hetero_vs_F16` `min=0.999984`, `Single-query B=1 S=24 ANE_vs_GPU` `cos=0.999986`.

*Note: Monorepo capture ran alongside light background system activity (`6` quiescence warnings recorded in the run manifest).*

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

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6, Xcode SDK 27.2, `Apple clang 21.0.0`)** producing `0`-warning native builds (`libbge_ane_runner.dylib`, `libbge_cpu_query.dylib`, `bge-ane-cli`).
- **Not tested:** Other Apple Silicon generations/RAM tiers, Linux/x86/CUDA, or sequences exceeding the model's `512`-token context window.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
