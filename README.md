# fast-silicon

**Run local embeddings, guaranteed-valid JSON extraction, and 20B MoE models on Apple Silicon up to 2.5x–5.3x faster — and at a fraction of the memory.**

`fast-silicon` is a `uv` workspace of focused inference engines, CLI tools, OpenAI-compatible HTTP servers, and resumable Parquet pipelines built on Apple MLX, CoreML (16-core Neural Engine), and Metal 4. By scheduling work across GPU and ANE without heavy PyTorch runtime bloat, the suite embeds at **207K tok/s** (`<0.53 ms` single query), holds **32K**-context embeddings in **2.69 GB** (`1.71x–2.42x` vs `pplx-context-mlx-f16`), extracts strict JSON with **Gemma-4 E2B** (`2.54x` vs `mlx_vlm`), and serves **GPT-OSS-20B** at **555.5 tok/s** across `128` parallel documents (`4.78x` vs `mlx_lm`). It also ships a unified monorepo benchmark and `RUSAGE_INFO_V4` memory-footprint harness ([`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py)).

> [!NOTE]
> **Tested Hardware Baseline (`Mac16,8` — `48 GB` Apple M4 Pro):** Built and verified on a **14-inch MacBook Pro (`Mac16,8`)** with **Apple M4 Pro** (`48 GB` unified RAM, `14` CPU `[10P + 4E]`, `20` GPU, `16` ANE `h16g` cores, macOS 26.6 / Metal 4). Other Apple Silicon chips and memory sizes are untested and emit a non-fatal startup note (`FAST_SILICON_NO_HW_WARN=1` to mute).

## Quick start

```bash
uv sync --extra dev          # Create workspace .venv across all 7 packages
make hw-check                # Verify Mac16,8 48 GB M4 Pro host baseline
make native                  # Compile CoreML / Objective-C++ / NEON libraries across packages
make guard                   # Run fast-ir-guard --strict-bool-mask (`0 errors, 0 warnings`)
make test-cpu                # Run offline CPU + fault-injection + native parity suites
```

```bash
# Dry-run unified M4 Pro benchmark sweep (zero GPU/ANE work)
.venv/bin/python scripts/benchmark_monorepo.py --dry-run

# Serialized live M4 Pro sweep (captures Metal Peak + OS Footprint via RUSAGE_INFO_V4)
.venv/bin/python scripts/benchmark_monorepo.py --include-verify --json-out artifacts/monorepo_bench.json
```

## Workspace Packages (`48 GB` M4 Pro `Mac16,8`)

| Package | Model | Best For / Capabilities |
|---|---|---|
| [`packages/fast-shared`](packages/fast-shared) | Shared runtime & toolkit | Reusable Metal 4/CoreML/Accelerate kernels, `MappedEmbedding`, `RUSAGE_INFO_V4` memory tooling (`fast-bench-check`), `fast-ir-guard`, Parquet & HTTP SSE, Agent plugin |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Sub-ms (`0.53 ms`) search queries & `207K tok/s` batch indexing (ANE+GPU `hetero`, OpenAI HTTP, Parquet, zero-Python `bge-ane-cli`) |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | `2048`-context MRL (`128..768`-d) retrieval & OCR chunking (`31.8K–36.9K tok/s`, OpenAI HTTP, Parquet, `emb2-ane-cli`) |
| [`packages/fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | `32K` late-chunking contextual embeddings in `2.69 GB` OS (`15.4K tok/s`, `1.71x–2.42x` vs `pplx-context-mlx`, MRL `128..1024`, `--context-tokens` Parquet) |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Guaranteed schema-valid batch JSON extraction (`8.92 docs/s` at `6.26 GB` Metal, `int8` KV cache, PDF/Parquet CLI, SSE server) |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) | High-concurrency 20B structured generation (`555.5 tok/s` at `B=128`, shared-prefix KV reuse, OpenAI SSE server, Parquet CLI) |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Fast local binary, multi-choice, & scalar document scoring (`50.95 q/s` Metal or `40.53 q/s` pure ANE with `0` GPU ops) |

## Live Monorepo Showcase (`scripts/benchmark_monorepo.py`)

Full 6-package showcase on M4 Pro (`Mac16,8`, `48 GB`, recorded alongside `6` background quiescence warnings; `make guard` `0` errors / `0` warnings; `make sync-check` & `make leak-check` pass).

### Winning Configs vs Internal & Public Baselines

| Package | Winning Run | Primary Speed | Secondary Metric | vs Internal Baseline | vs Public Baseline | Suite Metal | Suite OS | Verify |
|---|---|---:|---:|---|---|---:|---:|---|
| [`fast-bge-small`](packages/fast-bge-small) | `hetero` (`n=1,000`) | **207,264 tok/s** | `0.526 ms` (`B=1` ANE) | **`1.52x`** vs `gpu` (`136,416 tok/s`) | **`2.50x`** vs `transformers-mps` (`82,883 tok/s`) | `1.173 GB` | `4.813 GB` | `ok` |
| [`fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `hetero2` (`n=2,000`) | **31,786 tok/s** | `2.188 ms` (`B=1` ANE) | **`1.85x`** vs `gpu` (`17,177 tok/s`) | *skipped* (`transformers` lacks arch) | `2.224 GB` | `10.820 GB` | `ok` |
| [`fast-pplx`](packages/fast-pplx) | `ane` / `hetero2` (`doc_s256`, `B=8`) | **15,384.2 tok/s** | `3.93 ms` (`query_b1` ANE) | **`1.75x`** vs `gpu-bf16` (`mixed`) | **`1.71x`** vs `pplx-context-mlx-f16` (`8,984.9`; **`2.42x`** `mixed`), **`2.96x`** `torch-mps` | `1.621 GB` | `2.522 GB` | `ok` |
| [`fast-gemma4`](packages/fast-gemma4) | `Concurrent ANE+GPU` (`2xB=128`) | **8.92 docs/s** | `1,192 dec tok/s` | **`1.21x`** vs `GPUOnly` (`7.36 docs/s`) | **`2.54x`** vs `stock-mlx_vlm B=64` (`3.51 docs/s`) | `6.718 GB` | `18.314 GB` | `ok` |
| [`fast-gpt-oss`](packages/fast-gpt-oss) | `batched_B128` (`128` docs) | **555.5 dec tok/s** | `1.874 docs/s` | **`7.42x`** vs `B=1` (`74.9 dec tok/s`) | **`4.78x`** vs `stock-mlx_lm B=8` (`116.3 tok/s`) | `15.901 GB` | `21.709 GB` | `ok` |
| [`fast-decision-models`](packages/fast-decision-models) | `kev_08b` (`w8 B=8`, `n=64`) | **50.95 q/s** | `19.63 ms/q` | **`1.86x`** vs `bf16 B=1` (`-0.553 GB` OS) | **`5.27x`** vs `transformers-mps B=8` (`9.67 q/s`) | `1.748 GB` | `3.166 GB` | `ok` |

### Run-by-Run Breakdown (`monorepo_bench.json`)

#### 1. Embeddings — Single-Query Latency (`B=1`)

| Package | Seq | Mode / Runtime | Median | Min | Metal Peak | OS Footprint | vs Public MPS | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `fast-bge-small` | `24` | **`ane`** | **0.526 ms** | `0.467 ms` | `1.173 GB` *(suite)* | `4.813 GB` *(suite)* | **`16.06x`** | **`3.83x`** vs `gpu` (`2.015 ms`) |
| `fast-bge-small` | `24` | `gpu` | `2.015 ms` | `1.847 ms` | `1.173 GB` *(suite)* | `4.813 GB` *(suite)* | `4.19x` | Internal Metal baseline |
| `fast-bge-small` | `24` | `transformers-mps-float16` | `8.446 ms` | — | `0.071 GB` | `3.520 GB` | `1.00x` | Public PyTorch MPS (`load=1.314 s`) |
| `fast-embeddinggemma-2` | `32` | **`ane`** | **2.188 ms** | `1.943 ms` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | — | **`4.36x`** vs `gpu` (`9.540 ms`) |
| `fast-embeddinggemma-2` | `64` | `gpu` | `9.540 ms` | `9.089 ms` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | — | Internal Metal baseline |
| `fast-pplx` | `15` | **`ane`** | **3.93 ms** | `3,817.2 tok/s` | `0.000 GB` | **0.243 GB** | **`11.95x`** | **`2.28x`** `pplx-context-mlx-f16`; **`224.4x`** `torch-cpu`; `2.47x` `gpu` |
| `fast-pplx` | `15` | `hetero2` | `4.32 ms` | `3,475.4 tok/s` | `0.000 GB` | `0.684 GB` | `10.88x` | **`2.07x`** `pplx-context-mlx-f16`; `2.25x` vs `gpu` |
| `fast-pplx` | `15` | `gpu-no-mmap` | `6.06 ms` | `2,477.2 tok/s` | `1.015 GB` | `1.302 GB` | `7.76x` | `1.48x` `pplx-context-mlx-f16`; `load=0.42 s` |
| `fast-pplx` | `15` | `gpu-bf16` | `7.21 ms` | `2,080.4 tok/s` | `0.506 GB` | `1.271 GB` | `6.52x` | `1.24x` `pplx-context-mlx-f16`; `load=0.33 s` |
| `fast-pplx` | `15` | `pplx-context-mlx-f16` (Public) | `8.95 ms` | `1,675.7 tok/s` | `0.489 GB` | `0.890 GB` | `5.25x` | Public MLX GPU baseline (`load=0.40 s`) |
| `fast-pplx` | `15` | `gpu` (`mmap` FP16) | `9.72 ms` | `1,543.6 tok/s` | `0.506 GB` | `0.796 GB` | `4.83x` | `-0.507 GB` OS vs `gpu-no-mmap` |
| `fast-pplx` | `15` | `torch-mps` (Public) | `46.98 ms` | `319.3 tok/s` | `1.193 GB` | `2.875 GB` | `1.00x` | Public PyTorch MPS (`load=1.24 s`) |
| `fast-pplx` | `15` | `torch-cpu` (Public) | `881.81 ms` | `17.0 tok/s` | `0.000 GB` | `0.644 GB` | `0.05x` | `threads=6`, `load=0.27 s` |

#### 2a. Batch Embeddings (`fast-bge-small` `n=1,000` & `fast-embeddinggemma-2` `n=2,000`)

| Package | Corpus (Toks) | Mode / Runtime | Elapsed | Throughput | Docs/s | Doc Split (`GPU/ANE/CPU`) | Metal Peak | OS Footprint | Speedup |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| `fast-bge-small` | `1,000` (`249,303`) | **`hetero`** | **1.203 s** | **207,264 tok/s** | — | `424 / 576 / 0` | `1.173 GB` *(suite)* | `4.813 GB` *(suite)* | **`2.50x`** MPS; **`1.52x`** `gpu` |
| `fast-bge-small` | `1,000` (`249,303`) | `gpu` | `1.828 s` | `136,416 tok/s` | — | `1,000 / 0 / 0` | `1.173 GB` *(suite)* | `4.813 GB` *(suite)* | `1.65x` vs MPS |
| `fast-bge-small` | `1,000` (`249,303`) | `ane` | `4.388 s` | `56,808 tok/s` | — | `0 / 1,000 / 0` | `1.173 GB` *(suite)* | `4.813 GB` *(suite)* | `0.69x` vs MPS |
| `fast-bge-small` | `1,000` (`249,303`) | `transformers-mps-float16` | — | `82,883.2 tok/s` | `332.46` | — | `0.071 GB` | `3.520 GB` | `1.00x` (`load=1.314 s`) |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | **`hetero2` (`GPU+ANE`)** | **17.363 s** | **31,786 tok/s** | **115.2** | `591 / 1,409 / 0` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | **`1.85x`** vs `gpu` |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | `hetero` (`+CPU`) | `17.621 s` | `31,319 tok/s` | `113.5` | `650 / 1,322 / 28` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | `1.82x` vs `gpu` |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | `ane` | `27.149 s` | `20,328 tok/s` | `73.7` | `0 / 2,000 / 0` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | `1.18x` vs `gpu` |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | `gpu` | `32.129 s` | `17,177 tok/s` | `62.2` | `2,000 / 0 / 0` | `2.224 GB` *(suite)* | `10.820 GB` *(suite)* | `1.00x` (Internal baseline) |

#### 2b. Long-Context Embeddings (`fast-pplx`, representative `doc_s256`, `doc_s4096`, `mixed`)

| Workload | Mode / Runtime | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | Comparison |
|---|---|---:|---:|---:|---:|---:|---|
| `doc_s256` (`B=8`) | **`ane`** | **15,384.2 tok/s** | **16.64 ms** | **60.09** | `0.000 GB` | **0.250 GB** | **`1.71x`** `pplx-context-mlx-f16`, **`2.96x`** `torch-mps`, **`11.20x`** `torch-cpu`, **`1.69x`** `gpu` |
| `doc_s256` (`B=8`) | **`hetero2`** | **15,361.9 tok/s** | `16.66 ms` | `60.01` | `0.000 GB` | `1.057 GB` | **`1.71x`** `pplx-context-mlx-f16`, **`2.95x`** `torch-mps`, **`11.18x`** `torch-cpu`, **`1.69x`** `gpu` |
| `doc_s256` (`B=8`) | `pplx-context-mlx-f16` (Public) | `8,984.9 tok/s` | `35.62 ms` | `28.08` | `1.057 GB` | `1.569 GB` | **`6.28x`** higher OS footprint than `ane` (`1.73x` `torch-mps`) |
| `doc_s256` (`B=8`) | `torch-mps` (Public) | `5,200.9 tok/s` | `61.53 ms` | `16.25` | `1.193 GB` | `3.483 GB` | `3.79x` `torch-cpu` |
| `doc_s256` (`B=8`) | `torch-cpu` (Public) | `1,373.5 tok/s` | `232.97 ms` | — | `0.000 GB` | `1.305 GB` | `threads=1` |
| `doc_s4096` (`B=4`) | **`hetero2`** | **7,857.7 tok/s** | **521.27 ms** | **1.92** | **`1.226 GB`** | **2.480 GB** | **`1.51x`** `pplx-context-mlx-f16`, **`2.52x`** `torch-mps`, **`10.93x`** `torch-cpu` (`-0.248 GB` OS vs `pplx-mlx-f16`) |
| `doc_s4096` (`B=4`) | `pplx-context-mlx-f16` (Public) | `5,190.5 tok/s` | `933.24 ms` | `1.07` | `1.857 GB` | `2.728 GB` | `1.66x` `torch-mps` (`+0.631 GB` Metal vs `hetero2`) |
| `doc_s4096` (`B=4`) | `torch-mps` (Public) | `3,118.0 tok/s` | `1,553.58 ms` | `0.64` | `1.193 GB` | `15.690 GB` | **`6.33x`** higher OS footprint than `hetero2` |
| `doc_s4096` (`B=4`) | `torch-cpu` (Public) | `719.2 tok/s` | `6,735.20 ms` | — | `0.000 GB` | `6.569 GB` | `threads=1` |
| `mixed` (`24x~128 + 4x2048`) | **`hetero2`** | **13,935.0 tok/s** | **871.33 ms** | — | `1.621 GB` | **2.522 GB** | `ane=24, gpu=4`; **`2.42x`** `pplx-context-mlx-f16`, **`1.75x`** `gpu-bf16` (`1.49x` `gpu`) |
| `mixed` (`24x~128 + 4x2048`) | `gpu` (`mmap` FP16) | `9,331.9 tok/s` | `1,301.12 ms` | — | `1.621 GB` | `2.497 GB` | `ane=0, gpu=28`; **`1.62x`** `pplx-context-mlx-f16`, `1.17x` `gpu-bf16` |
| `mixed` (`24x~128 + 4x2048`) | `pplx-context-mlx-f16` (Public) | `5,762.7 tok/s` | `2,370.75 ms` | — | `1.439 GB` | `2.728 GB` | Public MLX GPU (`+0.206 GB` OS vs `hetero2`) |

*(Full 51-row `fast-pplx` matrix across `query_b1`, `query_b32`, `doc_s256`, `doc_s512`, `doc_s2048`, `doc_s4096`, `mixed_short_long` including [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) `pplx-context-mlx-f16` is in [`packages/fast-pplx/README.md`](packages/fast-pplx/README.md#live-hardware-benchmarks-m4-pro-mac168-48-gb)).*

#### 3. Structured JSON Generation — `fast-gemma4` (`Gemma-4 E2B`)

| Mode / Runtime | KV Cache | Batch (Docs) | Valid | Prefill / Wait | Decode | Decode tok/s | E2E Out / Eff tok/s | Docs/s | Peak Mem | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`Concurrent ANE+GPU`** (`W8` ring) | `kv_i8` | `2x128` (`256`) | `256/256` | `3.39 s` | `24.82 s` | `1,192` | — | **8.92** | **6.26 GB** | **`2.54x`** `stock B=64`; **`1.21x`** `GPUOnly` |
| `GPUOnly` | `kv_i8` | `128` (`128`) | `128/128` | `5.61 s` | `11.13 s` | **1,265** | — | `7.36` | **6.26 GB** | Saves **`-0.67 GB`** vs `no_kv_i8` |
| `GPUOnly` (`no_kv_i8` baseline) | `BF16` | `128` (`128`) | `128/128` | `5.91 s` | `12.29 s` | `1,218` | — | `6.83` | `6.93 GB` | Internal `no_kv_i8` ablation |
| `PromptLookup` | `kv_i8` | `32` (`32`) | `32/32` | `1.42 s` | `7.23 s` | — | `516 eff` | `3.64` | `6.10 GB` | Speculative draft verify |
| `PromptLookup` (`no_kv_i8` baseline) | `BF16` | `1` (`1`) | `1/1` | `0.03 s` | `1.92 s` | — | `79 eff` | `0.48` | `4.92 GB` | Internal `B=1` baseline |
| `stock-mlx_vlm` (Public) | `BF16` | `64` (`64`) | `64/64` | — | — | `804.1` | `455.8` | `3.51` | `9.764 GB` | `13.404 GB` OS (`load=2.06 s`) |
| `stock-mlx_vlm` (Public) | `BF16` | `1` | — | — | — | — | `68.8` | `0.44` | — | Single-stream baseline |

#### 4. Structured JSON Generation — `fast-gpt-oss` (`GPT-OSS-20B MXFP4`)

| Mode / Runtime | Batch (Docs) | Valid | Prefill | Decode / Wall | Decode tok/s | Docs/s | Peak Mem | vs Stock `B=8` | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`batched_B128`** | `128` (`128`) | **`128/128`** | `34.024 s` | `33.182 s` | **555.5** | **1.874** | `14.81 GB` | **`4.78x`** | **`7.42x`** vs `b1_seq`; `15.901 GB` suite Metal |
| `batched_B64` | `64` (`64`) | `64/64` | `15.532 s` | `22.625 s` | `410.0` | `1.636` | `12.63 GB` | `3.53x` | Hydragen v2 + Hybrid MoE |
| `batched_B8` | `8` (`8`) | `8/8` | `2.030 s` | `6.960 s` | `162.1` | `0.888` | `11.46 GB` | `1.39x` | Same batch as `stock-mlx_lm B=8` |
| `b1_seq` | `1` (`8`) | `8/8` | — | `17.27 s` *(wall)* | `74.9` | `0.464` | **10.70 GB** | `0.64x` | **`1.18x`** vs `stock-mlx_lm B=1` (`63.5 tok/s`) |
| `stock-mlx_lm` (Public) | `8` | — | — | — | `116.3` | `0.871` | `11.384 GB` | `1.00x` | `12.139 GB` OS (`load=3.985 s`) |
| `stock-mlx_lm` (Public) | `1` | — | — | — | `63.5` | — | — | `0.55x` | Single-stream baseline |

#### 5. Decision Scoring — `fast-decision-models` (`Laya-Typed-421M` + `Kev-0.8B`)

| Engine / Baseline | Precision | Batch | Questions | Elapsed | Throughput | Latency | Metal Peak | OS Footprint | vs MPS `B=8` | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`kev_08b`** (Default) | `W8` (`g64`) | `8` | `64` | `1.2562 s` | **50.95 q/s** | **19.63 ms/q** | `1.748 GB` *(suite)* | **3.166 GB** *(suite)* | **`5.27x`** | **`-2.13 GB`** OS vs MPS; **`-0.553 GB`** vs `bf16 B=8` |
| `laya_ane` (`Laya-Typed-421M`) | `W8A16` ANE | `P=4` | `64` | `1.5789 s` | **40.53 q/s** | **24.67 ms/q** | `1.748 GB` *(suite)* | **3.166 GB** *(suite)* | **`4.19x`** | `[1585,0,0]` ANE/GPU/CPU (`0` GPU ops) |
| `kev_08b` (Internal baseline) | `BF16` | `8` | `64` | `1.2390 s` | `51.65 q/s` | `19.36 ms/q` | `2.439 GB` | `3.718 GB` | `5.34x` | `w8 B=8` runs at `0.99x` with lower memory |
| `kev_08b` (Internal baseline) | `BF16` | `1` | `24` | `0.8754 s` | `27.42 q/s` | `36.48 ms/q` | `1.642 GB` | `2.086 GB` | `2.84x` | `w8 B=8` is **`1.86x`** faster |
| `transformers-mps-bf16` (Public) | `BF16` | `8` | — | — | `9.67 q/s` | `103.38 ms/q` | `1.505 GB` | `5.295 GB` | `1.00x` | Public MPS (`load=11.554 s`) |
| `transformers-mps-bf16` (Public) | `BF16` | `1` | — | — | `2.98 q/s` | — | — | — | `0.31x` | Public single-stream baseline |

## Hardware Suites, Weight Setup & Serialization

Model weights and compiled `.mlmodelc` bundles live outside git. Stage artifacts per package (`<cli> setup-weights` / `setup-ane`), then run serialized hardware validation:

```bash
make test-hw    # Checks host Mac16,8, then runs each package's full pytest suite sequentially
```

> [!IMPORTANT]
> **Serialize GPU & ANE workloads:** Run one Metal GPU or CoreML ANE suite/benchmark at a time under `/tmp/m4_gpu_validation.lock`. [`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py) and [`packages/fast-pplx/scripts/benchmark.py`](packages/fast-pplx/scripts/benchmark.py) enforce `check_system_quiescence()` and `fcntl.flock` locking automatically.

## Benchmark & Investigation Harnesses

| Script | Scope | Captures |
|---|---|---|
| [`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py) | Cross-package monorepo sweep | Isolated per-package throughput + internal ablations (`gpu` vs `hetero`, `kv_i8` vs `--no-kv-i8`, `B=1` vs `B=128`, `w8` vs `bf16`) + next-fastest **public apples-to-apples** baselines ([`public_baselines.py`](scripts/public_baselines.py): `pplx-context-mlx-f16`, `mlx_vlm`, `mlx_lm`, `mlx_embeddings` / `transformers` MPS) via `--public-python` / `--pplx-mlx-path` + OS `RUSAGE_INFO_V4` footprint |
| [`packages/fast-pplx/scripts/benchmark.py`](packages/fast-pplx/scripts/benchmark.py) | `fast-pplx` deep investigation | `speed` (vs `pplx-context-mlx-f16`, `torch-mps`, `torch-cpu`, `gpu-bf16`, `gpu-no-mmap`), `parity` (16 refs), `context-limit` (`512..32K` vs `512` curve), `context-windows` (vs `isolated` MRR/nDCG@5), `bit-identity` dump/compare |

## Agent Plugin (`apple-silicon-ml-optimizer`) & Repo Sync

```bash
claude --plugin-dir ./packages/fast-shared
make sync         # Mirror clean files from ../fast-* -> packages/* (keeps workspace deps)
make sync-check   # Fail if synced packages/* differ from sibling ../fast-* checkouts
make leak-check   # Fail on private domain tokens or local absolute user paths
```

**Scope & not tested:** Verified on Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6). Untested on M1/M2/M3/M4 Max/M5 Macs, Linux, x86/CUDA, or concurrent unserialized hardware workloads.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspiring the unified-memory `RUSAGE_INFO_V4` footprint methodology, the clean README layout across this workspace, and key design & evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity / bit-identity / fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
