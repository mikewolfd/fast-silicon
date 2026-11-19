# fast-bge-small

Hardware-limit **BAAI/bge-small-en-v1.5** inference engine for Apple Silicon (**M4 Pro / Metal 4 + MPP + 16-core Apple Neural Engine**).

Combines zero-contention concurrent execution across both compute engines:
- **16-core Apple Neural Engine (`MLComputeUnitsCPUAndNeuralEngine`)**: custom Objective-C++ CoreML runner ([`native/bge_ane_runner.mm`](native/bge_ane_runner.mm)) with NEON FP16 word+position addition, 8x8 register transpose, zero-copy `MLMultiArray` backing, Layer-11 `[CLS]` slicing, and 4-slot asynchronous prediction pipelining (`~0.49 ms` single query at `S=24`).
- **20-core Metal GPU (`FastBgeSmallGPU`)**: custom Metal kernels via `mx.fast.metal_kernel` ([`src/fast_bge_small/gpu.py`](src/fast_bge_small/gpu.py)) using packed variable-length batching (zero attention padding), custom bidirectional `D=32` FlashAttention, Metal 4 `MPP::matmul2d` fused bias+GELU/residual epilogues, single-pass LayerNorm + L2 normalization, and Layer-11 `[CLS]`-only attention/FFN slicing.
- **Two-Ended Work Stealing Scheduler**: GPU consumes longest sequences forward while ANE concurrently pulls shortest sequences backward with zero GPU contention.

## Requirements

- **macOS 15+ / Xcode 26–27 SDK** (uses `<metal_tensor>` and `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` `MPP::matmul2d`; verified on Apple M4 Pro, macOS 26.6, Xcode SDK 27.2).
- **Python >= 3.11** (`mlx`, `numpy`, `tokenizers`, `pyarrow`, `huggingface-hub`). No `transformers` or `torch` required at runtime.
- `coremltools >= 8.0` (Python 3.12) is only required if rebuilding `.mlmodelc` ANE bundles from scratch via `scripts/build_ane_models.py`.

## Build & Setup

```bash
# 1. Compile native Objective-C++ ANE/CPU libraries and standalone CLI binary
make native

# 2. Create isolated Python environment (links sibling ../fast-shared DRY utilities)
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"

# 3. Export FP16 weights & tokenizer assets into artifacts/ (from HF cache or download)
.venv/bin/bge-small setup-weights

# 4. Run parity & integration suite
.venv/bin/pytest -v
```

## Usage

### 1. Standalone Compiled C++/CoreML Binary (Zero Python)

`build/bge-ane-cli` embeds single queries directly on the Apple Neural Engine:

```bash
./build/bge-ane-cli --text "distributed cache invalidation strategy" --query-prefix --json
./build/bge-ane-cli --text "What is the budget deficit?" --bench 20
```

### 2. Python Library

```python
from fast_bge_small import FastBgeSmall

eng = FastBgeSmall()
q = eng.encode_texts("What is the budget deficit?", is_query=True, add_query_prefix=True)  # (384,)
vecs = eng.encode_texts(["passage one...", "passage two..."], mode="hetero")            # (2, 384)
```

### 3. CLI Commands

```bash
# Embed text from CLI or stdin
.venv/bin/bge-small embed "first passage" "second passage" --json

# Stream-embed a Parquet file, directory of .parquet shards, or glob
.venv/bin/bge-small parquet \
  --input path/to/input.parquet \
  --output path/to/output.parquet \
  --text-col body

# Persistent OpenAI-compatible HTTP daemon (/v1/embeddings, /health)
.venv/bin/bge-small serve --host 127.0.0.1 --port 8080

# Verify numerical parity against 64-doc fixture
.venv/bin/bge-small verify

# Run query latency and corpus throughput benchmarks
.venv/bin/bge-small bench
```

### 4. Streaming Parquet Pipeline Options

For document collections and multi-file Iceberg/Parquet tables:
- `--input`: `.parquet` file, directory, or glob pattern.
- `--output`: `.parquet` file (single streaming output) or directory (resumable `part-00000.parquet` shards with `--resume`).
- `--dtype float32|float16`: output vector precision (`fixed_size_list<float32|float16>[384]`).
- `--chunk --max-body-tokens 510 --overlap 51`: sliding-window token chunking preserving source metadata columns plus `chunk_index`, `char_start`, `char_end`, `token_count`.
- `--filter-ocr`: filters `is_ocr == true` or `len(ocr_pages) > 0` when present.
- `--ocr-pages-only` / `--per-ocr-page`: slices UTF-8 `page_offsets` using 1-based `ocr_pages` (`body_text` schema).

## Measured Verification & Benchmarks (M4 Pro)

- `make native`: compiles `build/libbge_ane_runner.dylib`, `build/libbge_cpu_query.dylib`, and standalone `build/bge-ane-cli` with 0 compiler warnings (`Apple clang 21.0.0`, macOS 26.6 / SDK 27.2).
- Standalone ANE binary (`./build/bge-ane-cli --text "What is the budget deficit?" --query-prefix --bench 30`): `tokens=16 bucket=S24 iters=30 latency_ms=0.502` (zero Python runtime).
- `bge-small verify` on 64-doc reference fixture:
  - `GPU_vs_F16` min cosine: `0.999998` (mean `0.999999`)
  - `GPU_vs_BF16` min cosine: `0.999998`
  - `Hetero_vs_F16` min cosine: `0.999984`
  - `Single-query B=1 S=24 ANE_vs_GPU` cosine: `0.999986`
- `pytest -v`: `7 passed` covering 64-doc multi-mode parity, ANE bucket routing across lengths `16..500`, chunking + Rust WordPiece tokenizer, native `bge-ane-cli`, UTF-8 `page_offsets` OCR slicing, streaming/sharded/resumable Parquet embedding, and HTTP `/v1/embeddings` server.

### Throughput & Query Latency (`bge-small bench`, measured during concurrent session GPU/CPU activity)

Single-query `B=1 S=24` latency:
- Synthetic run: `ANE_B1_S24` median `0.493 ms` (min `0.475 ms`) vs `GPU_B1_S24` median `1.937 ms` (min `1.843 ms`).
- OCR Parquet run: `ANE_B1_S24` median `0.560 ms` (min `0.485 ms`) vs `GPU_B1_S24` median `4.067 ms` (min `2.227 ms`).

Corpus throughput (`n = 4,000` documents):

| Workload | Mode | Elapsed (s) | Throughput (tok/s) | Split (`gpu_docs` / `ane_docs`, `gpu_toks` / `ane_toks`) |
| --- | --- | ---: | ---: | --- |
| Synthetic (`993,944` toks) | `gpu` | 7.185 | 138,335 | 4,000 / 0 (`993,944` / `0`) |
| Synthetic (`993,944` toks) | `ane` | 16.772 | 59,261 | 0 / 4,000 (`0` / `993,944`) |
| Synthetic (`993,944` toks) | `hetero` | **5.250** | **189,338** | 1,504 / 2,496 (`577,600` / `416,344`) |
| Document OCR (`981,829` toks) | `gpu` | 8.249 | 119,028 | 4,000 / 0 (`981,829` / `0`) |
| Document OCR (`981,829` toks) | `ane` | 16.272 | 60,339 | 0 / 4,000 (`0` / `981,829`) |
| Document OCR (`981,829` toks) | `hetero` | **4.566** | **215,022** | 1,856 / 2,144 (`632,487` / `349,342`) |

*Note: Measured on Apple M4 Pro (48 GiB) while background sessions were potentially sharing GPU/CPU resources; uncontended throughput may be higher.*
