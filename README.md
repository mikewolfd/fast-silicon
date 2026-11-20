# fast-silicon

**Run local embeddings, guaranteed-valid JSON extraction, and 20B MoE models on Apple Silicon up to 2.2x–5.8x faster — and at a fraction of the memory.**

`fast-silicon` is a `uv` workspace of focused inference engines, CLI tools, OpenAI-compatible HTTP servers, and resumable Parquet pipelines built on Apple MLX, CoreML (16-core Neural Engine), and Metal 4. By scheduling work across GPU and ANE without heavy PyTorch runtime bloat, the suite embeds at **183K tok/s** (**0.51 ms** single query; **215K tok/s** standalone OCR), holds **32K**-context embeddings in **2.69 GB** (**1.61x–2.16x** vs `pplx-context-mlx-f16`), extracts strict JSON with **Gemma-4 E2B** (**2.45x** raw / **4.90x** schema-valid vs `mlx_vlm`), and serves **GPT-OSS-20B** at **431.0 tok/s** across `128` parallel documents (**2.73x** vs `mlx_lm`). It also ships a unified monorepo benchmark and `RUSAGE_INFO_V4` memory-footprint harness ([`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py)).

> [!NOTE]
> **Tested Hardware Baseline (`Mac16,8` — `48 GB` Apple M4 Pro):** Built and verified on a **14-inch MacBook Pro (`Mac16,8`)** with **Apple M4 Pro** (`48 GB` unified RAM, `14` CPU `[10P + 4E]`, `20` GPU, `16` ANE `h16g` cores, macOS 26.6 / Metal 4), plus a partial hardware cross-check on **Apple M5 (`Mac17,2`, `24 GB`, macOS 27.2.1)** ([comparison below](#cross-chip-comparison-apple-m5-mac172-24-gb-macos-2721)). Other Apple Silicon chips and memory sizes emit a non-fatal startup note (`FAST_SILICON_NO_HW_WARN=1` to mute).

## Quick start

```bash
uv sync --extra dev --extra extract  # Create workspace .venv (`extract` needed for R8/schema/Parquet pipelines)
make hw-check                        # Verify Mac16,8 48 GB M4 Pro host baseline
make native                          # Compile CoreML / Objective-C++ / NEON libraries across packages
make guard                           # Run fast-ir-guard --strict-bool-mask (`0` errors)
make test-cpu                        # Run offline CPU + fault-injection + native parity suites
```

```bash
# Dry-run unified M4 Pro benchmark sweep (zero GPU/ANE work)
.venv/bin/python scripts/benchmark_monorepo.py --dry-run

# Serialized live M4 Pro sweep (captures Metal Peak + OS Footprint via RUSAGE_INFO_V4; add `--allow-contended` if background macOS tasks trigger quiescence abort)
.venv/bin/python scripts/benchmark_monorepo.py --include-verify --json-out artifacts/monorepo_bench.json
```

## Workspace Packages (`48 GB` M4 Pro `Mac16,8`)

| Package | Model | Best For / Capabilities |
|---|---|---|
| [`packages/fast-shared`](packages/fast-shared) | Shared runtime & toolkit | Reusable Metal 4/CoreML/Accelerate kernels, `MappedEmbedding`, `RUSAGE_INFO_V4` memory tooling (`fast-bench-check`), `fast-ir-guard`, Parquet & HTTP SSE, Agent plugin |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Sub-ms (`0.51 ms`) search queries & `183K tok/s` batch indexing (`215K` standalone OCR; ANE+GPU `hetero`, OpenAI HTTP, Parquet, zero-Python `bge-ane-cli`) |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | `2048`-context MRL (`128..768`-d) retrieval & OCR chunking (`35.7K–36.9K tok/s`, OpenAI HTTP, Parquet, `emb2-ane-cli`) |
| [`packages/fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | `32K` late-chunking contextual embeddings in `2.69 GB` OS (`14.9K tok/s`, `1.61x–2.16x` vs `pplx-context-mlx`, MRL `128..1024`, `--context-tokens` Parquet) |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Guaranteed schema-valid batch JSON extraction (`8.78 docs/s` at `6.28 GB` MLX peak, `int8` KV cache, PDF/Parquet CLI, SSE server) |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) | High-concurrency 20B structured generation (`431.0–462.8 tok/s` at `B=128`, shared-prefix KV reuse, OpenAI SSE server, Parquet CLI) |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Fast local binary, multi-choice, & scalar document scoring (`54.82 q/s` Metal or `42.03 q/s` pure ANE with `0` GPU ops) |

## Live Monorepo Showcase (`scripts/benchmark_monorepo.py`)

Full 6-package showcase on M4 Pro (`Mac16,8`, `48 GB`, post-remediation sweep recorded alongside `4` host background quiescence warnings; `fast-pplx` captured in an isolated post-fix sweep with `2` monorepo / `3` speed quiescence warnings; `make guard` `0` errors; `make sync-check` & `make leak-check` pass).

### Winning Configs vs Internal & Public Baselines

| Package | Winning Run | Primary Speed | Secondary Metric | vs Internal Baseline | vs Public Baseline | Suite Metal | Suite OS | Verify |
|---|---|---:|---:|---|---|---:|---:|---|
| [`fast-bge-small`](packages/fast-bge-small) | `hetero` (`n=1,000`) | **183,015 tok/s** | `0.506 ms` (`B=1` ANE) | **`1.58x`** vs `gpu` (`115,632 tok/s`) | **`2.24x`** vs `transformers-mps` (`81,593 tok/s`) | `1.186 GB` | `2.081 GB` | `ok` |
| [`fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `hetero` (`GPU+ANE`, `n=2,000`) | **35,717 tok/s** | `2.171 ms` (`B=1` ANE) | **`2.09x`** vs `gpu` (`17,105 tok/s`) | *skipped* (`transformers` lacks arch) | `1.465 GB` | `3.318 GB` | `ok` |
| [`fast-pplx`](packages/fast-pplx) | `ane` / `hetero2` (`doc_s256`, `B=8`) | **14,883.7 tok/s** | `4.18 ms` (`query_b1` ANE) | **`1.52x`** vs `gpu` (`doc_s256`; **`1.49x`** `mixed` vs `gpu-bf16`) | **`1.68x`** vs `pplx-context-mlx-f16` (`8,842.8`; **`1.72x`** `mixed`, **`2.16x`** `query_b1`), **`4.90x`** `torch-mps` | `2.030 GB` | `3.068 GB` | `ok` |
| [`fast-gemma4`](packages/fast-gemma4) | `Concurrent ANE+GPU` (`fp16`, `2xB=128`) | **8.78 docs/s** | `1,131 dec tok/s` | **`1.29x`** vs `GPUOnly` (`6.79`; **`+9.9%`** vs uncalib `w8` `7.99`) | **`4.90x`** valid / **`2.45x`** raw vs `stock-mlx_vlm B=64` (`1.791` / `3.583`) | `6.744 GB` | `18.953 GB` | `ok` |
| [`fast-gpt-oss`](packages/fast-gpt-oss) | `batched_B128` (`128` docs) | **431.0 dec tok/s** | `1.413 docs/s` | **`4.93x`** vs `B=1` (`87.5 dec tok/s`) | **`2.73x`** vs `stock-mlx_lm B=8` (`158.0 tok/s`) | `15.880 GB` | `23.831 GB` | `ok` |
| [`fast-decision-models`](packages/fast-decision-models) | `kev_08b` (`w8 B=8`, `n=64`) | **54.82 q/s** | `18.24 ms/q` | **`1.87x`** vs `bf16 B=1` (`-0.605 GB` OS vs `bf16 B=8`) | **`5.82x`** vs `transformers-mps B=8` (`9.42 q/s`) | `1.556 GB` | `2.440 GB` | `ok` |

### Run-by-Run Breakdown (`monorepo_bench_post.json` & `fast_pplx_post.json`)

#### 1. Embeddings — Single-Query Latency (`B=1`)

| Package | Seq | Mode / Runtime | Median | Min | Metal Peak | OS Footprint | vs Public MPS | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `fast-bge-small` | `24` | **`ane`** | **0.506 ms** | `0.464 ms` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | **`14.67x`** | **`3.92x`** vs `gpu` (`1.985 ms`) |
| `fast-bge-small` | `24` | `gpu` | `1.985 ms` | `1.881 ms` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | `3.74x` | Internal Metal baseline |
| `fast-bge-small` | `24` | `transformers-mps-float16` | `7.425 ms` | — | `0.071 GB` | `3.532 GB` | `1.00x` | Public PyTorch MPS (`load=1.424 s`) |
| `fast-embeddinggemma-2` | `32` | **`ane`** | **2.171 ms** | `1.892 ms` | `1.465 GB` *(suite)* | `3.318 GB` *(suite)* | — | **`3.60x`** vs `gpu` (`7.806 ms`) |
| `fast-embeddinggemma-2` | `64` | `gpu` | `7.806 ms` | `7.381 ms` | `1.465 GB` *(suite)* | `3.318 GB` *(suite)* | — | Internal Metal baseline |
| `fast-pplx` | `15` | **`ane`** | **4.18 ms** | `3,584.6 tok/s` | `0.000 GB` | **0.248 GB** | **`6.87x`** | **`2.16x`** `pplx-context-mlx-f16`; **`190.86x`** `torch-cpu`; `1.84x` `gpu` |
| `fast-pplx` | `15` | `hetero2` | `4.32 ms` | `3,469.0 tok/s` | `0.000 GB` | `0.782 GB` | `6.65x` | **`2.09x`** `pplx-context-mlx-f16`; `1.78x` vs `gpu` |
| `fast-pplx` | `15` | `gpu-no-mmap` | `5.72 ms` | `2,620.7 tok/s` | `1.024 GB` | `1.311 GB` | `5.03x` | `1.58x` `pplx-context-mlx-f16`; `load=0.41 s` |
| `fast-pplx` | `15` | `gpu` (`mmap` FP16) | `7.70 ms` | `1,947.3 tok/s` | `0.517 GB` | `0.818 GB` | `3.73x` | `1.17x` `pplx-context-mlx-f16`; `-0.493 GB` OS vs `gpu-no-mmap` |
| `fast-pplx` | `15` | `pplx-context-mlx-f16` (Public) | `9.04 ms` | `1,660.1 tok/s` | `0.489 GB` | `0.877 GB` | `3.18x` | Public MLX GPU baseline (`load=0.34 s`) |
| `fast-pplx` | `15` | `gpu-bf16` | `9.91 ms` | `1,513.6 tok/s` | `0.517 GB` | `1.290 GB` | `2.90x` | `-0.021 GB` OS vs `gpu-no-mmap`; `load=0.28 s` |
| `fast-pplx` | `15` | `torch-mps` (Public) | `28.77 ms` | `521.4 tok/s` | `1.193 GB` | `2.977 GB` | `1.00x` | Public PyTorch MPS (`load=2.01 s`) |
| `fast-pplx` | `15` | `torch-cpu` (Public) | `798.69 ms` | `18.8 tok/s` | `0.000 GB` | `0.764 GB` | `0.04x` | `threads=6`, `load=1.27 s` |

#### 2a. Batch Embeddings (`fast-bge-small` `n=1,000` & `fast-embeddinggemma-2` `n=2,000`)

| Package | Corpus (Toks) | Mode / Runtime | Elapsed | Throughput | Docs/s | Doc Split (`GPU/ANE/CPU`) | Metal Peak | OS Footprint | Speedup |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| `fast-bge-small` | `1,000` (`249,303`) | **`hetero`** | **1.362 s** | **183,015 tok/s** | — | `—` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | **`2.24x`** MPS; **`1.58x`** `gpu` (`1.583x`) |
| `fast-bge-small` | `1,000` (`249,303`) | `gpu` | `2.156 s` | `115,632 tok/s` | — | `—` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | `1.42x` vs MPS |
| `fast-bge-small` | `1,000` (`249,303`) | `ane` | `4.211 s` | `59,209 tok/s` | `237.5` | `0 / 1,000 / 0` | `1.186 GB` *(suite)* | `2.081 GB` *(suite)* | `0.73x` vs MPS |
| `fast-bge-small` | `1,000` (`249,303`) | `transformers-mps-float16` | — | `81,592.6 tok/s` | `327.28` | — | `0.071 GB` | `3.532 GB` | `1.00x` (`load=1.424 s`) |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | **`hetero` (`GPU+ANE`)** | **15.452 s** | **35,717 tok/s** | **129.4** | `—` | `1.465 GB` *(suite)* | `3.318 GB` *(suite)* | **`2.09x`** vs `gpu` (`2.088x`) |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | `ane` | `25.885 s` | `21,321 tok/s` | `77.3` | `0 / 2,000 / 0` | `1.465 GB` *(suite)* | `3.318 GB` *(suite)* | `1.25x` vs `gpu` |
| `fast-embeddinggemma-2` | `2,000` (`551,889`) | `gpu` | `32.265 s` | `17,105 tok/s` | `62.0` | `2,000 / 0 / 0` | `1.465 GB` *(suite)* | `3.318 GB` *(suite)* | `1.00x` (Internal baseline) |

#### 2b. Long-Context Embeddings (`fast-pplx`, representative `doc_s256`, `doc_s4096`, `mixed`)

| Workload | Mode / Runtime | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | Comparison |
|---|---|---:|---:|---:|---:|---:|---|
| `doc_s256` (`B=8`) | **`ane`** | **14,883.7 tok/s** | **17.20 ms** | **58.14** | `0.000 GB` | **0.248 GB** | **`1.68x`** `pplx-context-mlx-f16`, **`4.90x`** `torch-mps`, **`9.23x`** `torch-cpu`, **`1.52x`** `gpu` |
| `doc_s256` (`B=8`) | **`hetero2`** | **14,620.6 tok/s** | `17.51 ms` | `57.11` | `0.000 GB` | `1.193 GB` | **`1.65x`** `pplx-context-mlx-f16`, **`4.82x`** `torch-mps`, **`9.07x`** `torch-cpu`, **`1.50x`** `gpu` |
| `doc_s256` (`B=8`) | `pplx-context-mlx-f16` (Public) | `8,842.8 tok/s` | `36.19 ms` | `27.63` | `1.057 GB` | `1.558 GB` | **`6.28x`** higher OS footprint than `ane` (`2.91x` `torch-mps`) |
| `doc_s256` (`B=8`) | `torch-mps` (Public) | `3,035.1 tok/s` | `105.43 ms` | `9.48` | `1.193 GB` | `3.395 GB` | `1.88x` `torch-cpu` |
| `doc_s256` (`B=8`) | `torch-cpu` (Public) | `1,612.5 tok/s` | `198.45 ms` | — | `0.000 GB` | `1.333 GB` | `threads=1` |
| `doc_s4096` (`B=4`) | **`hetero2`** | **7,530.8 tok/s** | **543.90 ms** | **1.84** | **`2.030 GB`** | `3.055 GB` | **`1.26x`** `pplx-context-mlx-f16`, **`2.93x`** `torch-mps`, **`10.70x`** `torch-cpu` (`-0.347 GB` vs `gpu-no-mmap`) |
| `doc_s4096` (`B=4`) | `pplx-context-mlx-f16` (Public) | `5,997.3 tok/s` | `807.69 ms` | `1.24` | `1.857 GB` | `2.736 GB` | `2.34x` `torch-mps` (`-0.173 GB` Metal vs `hetero2`) |
| `doc_s4096` (`B=4`) | `torch-mps` (Public) | `2,566.2 tok/s` | `1,887.61 ms` | `0.53` | `1.193 GB` | `12.699 GB` | **`4.16x`** higher OS footprint than `hetero2` |
| `doc_s4096` (`B=4`) | `torch-cpu` (Public) | `703.8 tok/s` | `6,882.38 ms` | — | `0.000 GB` | `14.294 GB` | `threads=1` |
| `mixed` (`24x~128 + 4x2048`) | **`hetero2`** | **12,624.1 tok/s** | **961.81 ms** | — | `1.776 GB` | `3.055 GB` | `ane=24, gpu=4`; **`1.72x`** `pplx-context-mlx-f16`, **`1.49x`** `gpu-bf16` (`1.43x` `gpu`) |
| `mixed` (`24x~128 + 4x2048`) | `gpu` (`mmap` FP16) | `8,857.0 tok/s` | `1,370.89 ms` | — | `1.776 GB` | **2.939 GB** | `ane=0, gpu=28`; **`1.20x`** `pplx-context-mlx-f16`, `1.05x` `gpu-bf16` |
| `mixed` (`24x~128 + 4x2048`) | `pplx-context-mlx-f16` (Public) | `7,360.3 tok/s` | `1,856.18 ms` | — | `1.439 GB` | `2.736 GB` | Public MLX GPU baseline |

*(Full 51-row `fast-pplx` matrix across `query_b1`, `query_b32`, `doc_s256`, `doc_s512`, `doc_s2048`, `doc_s4096`, `mixed_short_long` including [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) `pplx-context-mlx-f16` is in [`packages/fast-pplx/README.md`](packages/fast-pplx/README.md#live-hardware-benchmarks-m4-pro-mac168-48-gb)).*

#### 3. Structured JSON Generation — `fast-gemma4` (`Gemma-4 E2B`)

| Mode / Runtime | KV Cache | Batch (Docs) | Valid | Prefill / Wait | Decode | Decode tok/s | E2E Out / Eff tok/s | Docs/s | Peak Mem | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`Concurrent ANE+GPU`** (`fp16` default) | `kv_i8` | `2x128` (`256`) | `256/256` | `3.36 s` | `25.25 s` | `1,131` | — | **8.78** | **6.28 GB** | **`4.90x`** valid / **`2.45x`** raw vs `stock B=64`; **`1.29x`** vs `GPUOnly` (`+9.9%` vs uncalib `w8`) |
| `Concurrent ANE+GPU` (`--gemma4-ane-variant w8`) | `kv_i8` | `2x128` (`256`) | `256/256` | `3.47 s` | `28.13 s` | `1,050` | — | `7.99` | `6.28 GB` | Uncalibrated `w8` (`UserWarning`; `1.16x` vs paired `GPUOnly` `6.89 docs/s`) |
| `GPUOnly` | `kv_i8` | `128` (`128`) | `128/128` | `5.87 s` | `12.80 s` | **1,150** | — | `6.79` | `6.33 GB` | Saves **`-0.67 GB`** vs `no_kv_i8` (`7.00 GB`) |
| `GPUOnly` (`no_kv_i8` baseline) | `BF16` | `128` (`128`) | `128/128` | `5.54 s` | `12.95 s` | `1,157` | — | `6.88` | `7.00 GB` | Internal `no_kv_i8` ablation |
| `PromptLookup` | `kv_i8` | `32` (`32`) | `32/32` | `1.33 s` | `8.01 s` | — | `467 eff` | `3.40` | `6.10 GB` | Speculative draft verify |
| `PromptLookup` (`no_kv_i8` baseline) | `BF16` | `32` (`32`) | `32/32` | `1.32 s` | `7.99 s` | — | `468 eff` | `3.41` | `6.10 GB` | Internal `B=32` `no_kv_i8` baseline |
| `stock-mlx_vlm` (Public) | `BF16` | `64` (`64`) | `64/64` JSON / **`32/64`** schema | — | — | `809.4` | `465.3` | `3.583` raw (**`1.791`** valid) | `5.003 GB` active | `13.399 GB` OS (`load=2.00 s`); `fast` guarantees `100%` valid schemas |
| `stock-mlx_vlm` (Public) | `BF16` | `1` | — | — | — | — | `72.9` | `0.46` | — | Single-stream baseline |

> [!NOTE]
> **`fast-gemma4` ANE Variant Default (`fp16` vs `w8`):** `fast-gemma4` and `benchmark_monorepo.py --gemma4-ane-variant` now default to **`fp16`** (`1.0 GB`, `5,799 tok/s` ANE chunk prefill) rather than **`w8`** (`504 MB`, `7,815 tok/s`). Uncalibrated `W8` generative prefixes can drift into repeat loops (and in live `monorepo_bench`, `fp16` delivered `8.78 docs/s` / `1,131 tok/s` vs `7.99 docs/s` / `1,050 tok/s` on uncalibrated `w8`); opt into `--gemma4-ane-variant w8` only when the bundle is domain-corpus `AWQ-MSE` calibrated or QAT-validated via `ane_calibration_manifest.json` (see [`packages/fast-gemma4/README.md`](packages/fast-gemma4/README.md#ane-bundle-variants-fp16-default-vs-w8)).

#### 4. Structured JSON Generation — `fast-gpt-oss` (`GPT-OSS-20B MXFP4`)

| Mode / Runtime | Batch (Docs) | Valid | Prefill | Decode / Wall | Decode tok/s | Docs/s | Peak Mem | vs Stock `B=8` | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`batched_B128`** | `128` (`128`) | **`128/128`** | `48.268 s` | `42.010 s` | **431.0** | **1.413** | `14.79 GB` | **`2.73x`** | **`4.93x`** vs `b1_seq`; `15.880 GB` suite Metal (active VM/CPU load) |
| `batched_B64` | `64` (`64`) | `64/64` | `23.849 s` | `25.161 s` | `363.0` | `1.303` | `12.63 GB` | `2.30x` | Hydragen v2 + Hybrid MoE |
| `batched_B8` | `8` (`8`) | `8/8` | `3.266 s` | `6.551 s` | `174.8` | `0.813` | `10.92 GB` | `1.11x` | Same batch as `stock-mlx_lm B=8` (`-0.299 GB` Metal) |
| `b1_seq` | `1` (`8`) | `8/8` | — | `16.157 s` *(wall)* | `87.5` | `0.495` | **10.50 GB** | `0.55x` | **`1.18x`** vs `stock-mlx_lm B=1` (`74.2 tok/s`) |
| `stock-mlx_lm` (Public) | `8` | — | — | — | `158.0` | `1.168` | `11.219 GB` | `1.00x` | `12.133 GB` OS (`load=2.533 s`) |
| `stock-mlx_lm` (Public) | `1` | — | — | — | `74.2` | — | — | `0.47x` | Single-stream baseline |

#### 5. Decision Scoring — `fast-decision-models` (`Laya-Typed-421M` + `Kev-0.8B`)

| Engine / Baseline | Precision | Batch | Questions | Elapsed | Throughput | Latency | Metal Peak | OS Footprint | vs MPS `B=8` | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`kev_08b`** (Default) | `W8` (`g64`) | `8` | `64` | `1.1675 s` | **54.82 q/s** | **18.24 ms/q** | `1.556 GB` *(suite)* | **2.440 GB** *(suite)* | **`5.82x`** | **`-2.87 GB`** OS vs MPS; **`-0.605 GB`** vs `bf16 B=8` |
| `laya_ane` (`Laya-Typed-421M`) | `W8A16` ANE | `P=4` | `64` | `1.5229 s` | **42.03 q/s** | **23.80 ms/q** | `1.556 GB` *(suite)* | **2.440 GB** *(suite)* | **`4.46x`** | `[1585,0,0]` ANE/GPU/CPU (`0` GPU ops) |
| `kev_08b` (Internal baseline) | `BF16` | `8` | `64` | `1.1863 s` | `53.95 q/s` | `18.54 ms/q` | `2.283 GB` | `3.045 GB` | `5.73x` | `w8 B=8` runs at `1.02x` with **`-0.605 GB`** OS |
| `kev_08b` (Internal baseline) | `BF16` | `1` | `24` | `0.8198 s` | `29.27 q/s` | `34.16 ms/q` | `1.622 GB` | `2.061 GB` | `3.11x` | `w8 B=8` is **`1.87x`** faster |
| `transformers-mps-bf16` (Public) | `BF16` | `8` | — | — | `9.42 q/s` | `106.12 ms/q` | `1.505 GB` | `5.308 GB` | `1.00x` | Public MPS (`load=10.934 s`) |
| `transformers-mps-bf16` (Public) | `BF16` | `1` | — | — | `8.87 q/s` | — | — | — | `0.94x` | Public single-stream baseline |

### Cross-Chip Comparison: Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`)

We cross-checked the compute-bound NAX architecture on a base **Apple M5 (`Mac17,2`, `24 GB`, `10` GPU `Apple10` w/ NAX, `16` ANE `h17`, `macOS 27.2.1`, Xcode CLT-only)** against the M4 Pro baseline:

| `fast-silicon-probe` Metric | M4 Pro (`Mac16,8`, `48 GB`, `20` GPU) | Base M5 (`Mac17,2`, `24 GB`, `10` GPU NAX) | M5 vs M4 Pro |
|---|---:|---:|---|
| Peak GPU TFLOPS (`BF16`/`FP16`/`Q4` `M=512..2048`) | **7.42 TF** (`BF16 M=512 6.87`, `FP16 M=2048 6.76`, `Q4 M=512 6.40`) | **12.25 TF** `FP16 M=512` (`BF16 M=2048 12.15`, `Q8 13.48`, `Q4 14.49`) | **`1.65x`** peak FP16 (`2.26x` Q4 `M=512`; `mpp_bf M=128 11.24 TF`) |
| Effective DRAM Bandwidth | **210.2 GB/s** (`182.9 GB/s` copy) | **137.3 GB/s** (`129.0 GB/s` copy) | **`0.65x`** (`1.53x` higher on M4 Pro) |
| Empirical Roofline Ridge | **35.3 FLOP/B** | **89.2 FLOP/B** | **`2.53x`** higher arithmetic intensity needed on M5 |
| Single-Token Matmul (`M=1`) | **1.21 TF** @ `0.028 ms` (`Q4`) / `0.19 TF` @ `0.173 ms` (`BF16`) | `0.46 TF` @ `0.072 ms` (`Q4`) / `0.12 TF` @ `0.283 ms` (`BF16`) | **`2.63x`** `Q4` / **`1.58x`** `BF16` faster on M4 Pro (bandwidth-bound) |

- **Why each chip wins where it does:** M5's NAX matrix engines double GPU dense/batched throughput despite half the GPU cores (`+80%` `emb2` GPU, `+45%` `pplx doc_s256` GPU, `+35%` `doc_s4096` GPU, `+32%` `pplx mixed`), while M4 Pro's `1.53x` higher memory bandwidth and lower ANE dispatch overhead win single-token (`M=1`) matmuls and `B=1` single-query ANE latency. Because M5 `s4096` GPU reaches `9,652 tok/s` vs `ANE` scaling up with chunk parallelism, `gpu-bf16`/`gpu-no-mmap` lead pure `doc_s4096` on M5 while `hetero2` still leads mixed workloads (`1.331x` vs `gpu-bf16`).

| Package / Workload | M4 Pro (`Mac16,8`) | Base M5 (`Mac17,2`) | M5 Speedup / Notes |
|---|---:|---:|---|
| `fast-bge-small` (`hetero` / `gpu` / `ane`) | `183,015` / `115,632` / `59,209 tok/s` | **197,452** / `145,936` / `63,873 tok/s` | **`1.08x`** `hetero` (`1.26x` `gpu`, `1.08x` `ane`); `0.993 GB` Metal / `1.844 GB` OS |
| `fast-bge-small` `B=1` (`ane` / `gpu`) | **0.506 ms** / `1.985 ms` | `0.79 ms` / **1.62 ms** (`2.07x` ANE vs GPU) | M4 Pro **`1.56x`** lower `B=1` ANE latency; M5 GPU `1.23x` faster |
| `fast-embeddinggemma-2` (`hetero` / `gpu` / `ane`) | `35,717` / `17,105` / `21,321 tok/s` | **49,475** / `30,847` / `21,890 tok/s` | **`1.39x`** `hetero` (`1.80x` `gpu`, `1.03x` `ane`); `1.341 GB` Metal / `2.761 GB` OS |
| `fast-embeddinggemma-2` `B=1` (`ane` / `gpu`) | **2.171 ms** / `7.806 ms` | `2.58 ms` / **5.09 ms** (`1.97x` ANE vs GPU) | M4 Pro `1.19x` lower ANE `B=1`; M5 GPU `1.53x` faster |
| `fast-pplx` `query_b1` (`ane` / `gpu` / `gpu-bf16`) | **4.18** / `7.70` / `9.91 ms` | `5.52` (`0.245 GB`) / **5.57** (`0.762 GB`) / `6.10 ms` | M4 Pro `1.32x` lower ANE; M5 GPU `1.38x` faster (`-0.509 GB` `mmap`) |
| `fast-pplx` `doc_s256` (`hetero2` / `ane` / `gpu`) | `14,620.6` / `14,883.7` / `9,772.6 tok/s` | **15,234.7** (`16.80 ms`) / `15,101.0` / `14,144.6 tok/s` | **`1.45x`** `gpu`, `1.04x` `hetero2`; `0.245 GB` OS (`ane`) |
| `fast-pplx` `doc_s4096` (`gpu-no-mmap` / `gpu` / `hetero2`) | `~7,160–7,531 tok/s` | **9,697.4** (`422.38 ms`) / `9,652.0` (`3.073 GB`) / `9,637.1 tok/s` | M5 NAX **`+28%–35%`** at `4K` (`-0.465 GB` `mmap`; `gpu-bf16 9,673.0`) |
| `fast-pplx` `mixed` (`24x~128 + 4x2048`) | `12,624.1 tok/s` (`961.81 ms`) | **16,720.7 tok/s** (`726.16 ms`) | **`1.32x`** vs M4 Pro (`1.331x` vs M5 `gpu-bf16`) |
| `fast-decision-models` `laya_ane` (`W8A16` ANE `P=4`, `n=64`) | `42.03 q/s` (`23.80 ms/q`) | **44.4 q/s** (`22.52 ms/q`, `1.4413 s`) | **`1.06x`** pure ANE (`kev_08b` not staged on `24 GB` M5) |

> [!NOTE]
> **M5 Scope & `h17` CoreML Layout:** `kev_08b`, `fast-gemma4` E2B, `gpt-oss-20b`, and HuggingFace baseline tokenizers were not staged on the `24 GB` M5 test machine. On M5 `h17` (`macOS 27.2.1`), `emb2_pk_s{32,64,128}_w8` bundles fell back to `CPU=1` for a trailing `H↔S` transpose; returning ANE-native `[1, 512, 1, S]` NCHW `out_h` and untiling in NEON eliminates the CPU fallback (`ANE=2288, GPU=0, CPU=0` across all `9` buckets, `40/40` passing on M5 and M4 Pro).

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
| [`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py) | Cross-package monorepo sweep | Isolated per-package throughput + internal ablations (`gpu` vs `hetero`, `kv_i8` vs `--no-kv-i8`, `B=1` vs `B=128`, `w8` vs `bf16`, `--gemma4-ane-variant fp16|w8` defaulting to `fp16`) + next-fastest **public apples-to-apples** baselines ([`public_baselines.py`](scripts/public_baselines.py): `pplx-context-mlx-f16`, `mlx_vlm`, `mlx_lm`, `mlx_embeddings` / `transformers` MPS) via `--public-python` / `--pplx-mlx-path` + OS `RUSAGE_INFO_V4` footprint |
| [`packages/fast-pplx/scripts/benchmark.py`](packages/fast-pplx/scripts/benchmark.py) | `fast-pplx` deep investigation | `speed` (vs `pplx-context-mlx-f16`, `torch-mps`, `torch-cpu`, `gpu-bf16`, `gpu-no-mmap`), `parity` (16 refs), `context-limit` (`512..32K` vs `512` curve), `context-windows` (vs `isolated` MRR/nDCG@5), `bit-identity` dump/compare |

## Agent Plugin (`apple-silicon-ml-optimizer`) & Repo Sync

```bash
claude --plugin-dir ./packages/fast-shared
make sync              # Mirror clean files from ../fast-* -> packages/* (keeps workspace deps)
make sync-check        # Fail if synced packages/* differ from sibling ../fast-* checkouts
make sync-clean-check  # Also fail if any sibling ../fast-* checkout has uncommitted changes
make leak-check        # Fail on private domain tokens or local absolute user paths
```

**Scope & not tested:** Full suite verified on Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`). Partial hardware verification on Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only) covers `fast-shared`, `fast-bge-small`, `fast-embeddinggemma-2` (`40/40`, `CPU=0` across all `9` ANE buckets), `fast-pplx`, and `fast-decision-models` (`laya_ane`), with `kev_08b`, `fast-gemma4` E2B, `gpt-oss-20b`, and HF public baselines un-staged. Untested on M1/M2/M3/M4 Max/Ultra/M5 Pro/Max Macs, Linux, x86/CUDA, or concurrent unserialized hardware workloads.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspiring the unified-memory `RUSAGE_INFO_V4` footprint methodology, the clean README layout across this workspace, and key design & evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity / bit-identity / fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
