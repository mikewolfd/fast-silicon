# fast-silicon

Hardware-limit Apple Silicon ML inference and embedding workspace (`uv` monorepo) combining Metal 4 `MPP::matmul2d`, 16-core Apple Neural Engine (`CoreML` / `IOSurface` zero-copy), and CPU SME/AMX pipelines.
Includes a monorepo-wide benchmark & OS footprint runner ([`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py)) measuring MLX active peak alongside macOS `RUSAGE_INFO_V4` unified-memory footprint across all packages.

> [!WARNING]
> **Tested Hardware Baseline Only (`Mac16,8` — `48 GB` Apple M4 Pro)**
>
> Everything in this workspace was built and verified **only** on a **14-inch MacBook Pro (`Mac16,8`)** with **Apple M4 Pro** (`48 GB` RAM, `14` CPU `[10P + 4E]`, `20` GPU cores, `16` ANE `h16g` cores, macOS 26.6 / Metal 4). Running on other Apple Silicon chips or RAM sizes is **untested** and emits a startup check warning (`FAST_SILICON_NO_HW_WARN=1` to mute after reading).

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

| Package | Model | Engine Summary |
|---|---|---|
| [`packages/fast-shared`](packages/fast-shared) | Runtime, kernels & tooling | Metal 4 `MPP` (`FP16`/`BF16`), CoreML/Accelerate headers, `MappedEmbedding`, `fast-ir-guard`, `RUSAGE_INFO_V4` footprint (`fast-bench-check`), Parquet & HTTP SSE, Agent plugin |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Metal 4 MPP + 16-core ANE (`hetero`) + standalone zero-Python `bge-ane-cli` |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | Metal 4 MPP + 16-core ANE (`W8A16`) + CPU SME (`hetero2`/`hetero`) + standalone `emb2-ane-cli` |
| [`packages/fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | `Qwen3.5` hybrid GDN + SDPA up to `32K`, `hetero2` ANE (`w8a16`), `MappedEmbedding`, MRL `128..1024`, `--context-tokens` Parquet + `benchmark.py` |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Rolling `B_slots=128` (`kv_i8`) + Early14 Linear-W8 ANE ring (`1,326` ops) + `llguidance` + `forward_ff_jump_slot` + `PromptLookup` |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) | Unified Hot-MPP + Cold-QMV MoE, Hydragen v2 + SWA-128, `131K` pruned `lm_head`, `llguidance` |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Pure ANE (`[1585,0,0]`, `P=4` `IOSurface` + NEON) & `W8` MLX left-padded judge (stdlib `zipfile` `head.pt`) |

## Live Monorepo Showcase (`scripts/benchmark_monorepo.py`)

Full 6-package showcase on M4 Pro (`Mac16,8`, `48 GB`, recorded alongside `6` background quiescence warnings; `make guard` `0` errors / `0` warnings; `make sync-check` & `make leak-check` pass).

### Winning Configs vs Internal & Public Baselines

| Package | Winning Run | Primary Speed | Secondary Metric | vs Internal Baseline | vs Public Baseline | Suite Metal | Suite OS | Verify |
|---|---|---:|---:|---|---|---:|---:|---|
| [`fast-bge-small`](packages/fast-bge-small) | `hetero` (`n=1,000`) | **207,264 tok/s** | `0.526 ms` (`B=1` ANE) | **`1.52x`** vs `gpu` (`136,416 tok/s`) | **`2.50x`** vs `transformers-mps` (`82,883 tok/s`) | `1.173 GB` | `4.813 GB` | `ok` |
| [`fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `hetero2` (`n=2,000`) | **31,786 tok/s** | `2.188 ms` (`B=1` ANE) | **`1.85x`** vs `gpu` (`17,177 tok/s`) | *skipped* (`transformers` lacks arch) | `2.224 GB` | `10.820 GB` | `ok` |
| [`fast-pplx`](packages/fast-pplx) | `hetero2` (`doc_s256`, `B=8`) | **15,244.9 tok/s** | `4.46 ms` (`query_b1` ANE) | **`1.50x`** vs `gpu-bf16` (`mixed`) | **`5.50x`** vs `torch-mps` (`2,772.2 tok/s`) | `1.621 GB` | `2.591 GB` | `ok` |
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
| `fast-pplx` | `15` | **`ane`** | **4.46 ms** | — | `0.000 GB` | **0.245 GB** | **`7.86x`** | **`238.4x`** `torch-cpu`; `1.85x` `gpu` |
| `fast-pplx` | `15` | `hetero2` | `4.51 ms` | — | `0.000 GB` | `0.696 GB` | `7.78x` | `1.83x` vs `gpu` |
| `fast-pplx` | `15` | `gpu-no-mmap` | `6.65 ms` | — | `1.015 GB` | `1.306 GB` | `5.27x` | `load=0.46 s` |
| `fast-pplx` | `15` | `gpu-bf16` | `7.21 ms` | — | `0.506 GB` | `1.293 GB` | `4.86x` | `load=0.53 s` |
| `fast-pplx` | `15` | `gpu` (`mmap` FP16) | `8.26 ms` | — | `0.506 GB` | `0.798 GB` | `4.24x` | `-0.508 GB` OS vs `gpu-no-mmap` |
| `fast-pplx` | `15` | `torch-mps` (Public) | `35.07 ms` | — | `1.193 GB` | `2.972 GB` | `1.00x` | Public PyTorch MPS (`load=2.68 s`) |
| `fast-pplx` | `15` | `torch-cpu` (Public) | `1,063.43 ms` | — | `0.000 GB` | `0.767 GB` | `0.03x` | `threads=6`, `load=1.42 s` |

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
| `doc_s256` (`B=8`) | **`hetero2`** | **15,244.9 tok/s** | **16.79 ms** | **59.55** | `0.000 GB` | `1.069 GB` | **`5.50x`** `torch-mps`, **`10.06x`** `torch-cpu`, **`1.55x`** `gpu` |
| `doc_s256` (`B=8`) | `torch-mps` (Public) | `2,772.2 tok/s` | `115.43 ms` | `8.66` | `1.193 GB` | `3.457 GB` | `1.83x` `torch-cpu` |
| `doc_s256` (`B=8`) | `torch-cpu` (Public) | `1,515.2 tok/s` | `211.19 ms` | — | `0.000 GB` | `1.375 GB` | `threads=1` |
| `doc_s4096` (`B=4`) | **`hetero2`** | **6,425.3 tok/s** | **637.48 ms** | **1.57** | `1.226 GB` | `2.555 GB` | **`2.99x`** `torch-mps`, **`12.06x`** `torch-cpu` |
| `doc_s4096` (`B=4`) | `torch-mps` (Public) | `2,147.7 tok/s` | `2,255.45 ms` | `0.44` | `1.193 GB` | `12.492 GB` | **`4.89x`** higher OS footprint than `hetero2` |
| `doc_s4096` (`B=4`) | `torch-cpu` (Public) | `532.9 tok/s` | `9,089.62 ms` | — | `0.000 GB` | `13.113 GB` | `threads=1` |
| `mixed` (`24x~128 + 4x2048`) | **`hetero2`** | **11,411.6 tok/s** | **1,064.00 ms** | — | `1.621 GB` | `2.591 GB` | `ane=24, gpu=4`; **`1.50x`** `gpu-bf16` (`7,615.6 tok/s`) |
| `mixed` (`24x~128 + 4x2048`) | `gpu` (`mmap` FP16) | `8,502.7 tok/s` | `1,428.02 ms` | — | `1.621 GB` | `2.493 GB` | `ane=0, gpu=28`; `1.12x` `gpu-bf16` |

*(Full 47-row `fast-pplx` matrix across `query_b1`, `query_b32`, `doc_s256`, `doc_s512`, `doc_s2048`, `doc_s4096`, `mixed_short_long` is in [`packages/fast-pplx/README.md`](packages/fast-pplx/README.md#live-hardware-benchmarks-m4-pro-mac168-48-gb)).*

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
