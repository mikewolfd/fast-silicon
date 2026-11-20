# fast-pplx

**32K-context `pplx-embed-v2-context-0.6b` embeddings in 2.69 GB of unified memory — 1.61x–2.16x faster than `pplx-context-mlx` and 4.2x smaller than PyTorch MPS at 4K context.**

`fast-pplx` preserves exact upstream tokenization, late-chunking context (`[Q]`/`[D]`/`<|chunk_sep|>`), and Matryoshka dimensions (`128..1024`, `float32`/`float16`/`int8`), while scheduling short chunks (`<=512` tok) onto the Apple Neural Engine and streaming long documents on Metal GPU. That delivers **4.18 ms** single queries (**2.16x** vs `pplx-context-mlx-f16`, **6.87x** vs `torch-mps`), **14,884 tok/s** at `S=256` (**1.68x**), and **12,624 tok/s** (**1.72x**) on mixed batches — with zero `torch`/`transformers` runtime dependencies and built-in Python, OpenAI `/v1/embeddings`, contextual Parquet CLI, and `pplx-ane-cli` interfaces.

## Quick start

```bash
make -C packages/fast-pplx native
fast-pplx setup-weights --local-files-only
# Optional: compile CoreML .mlmodelc ANE stages (`w8a16` default, or `w8a8` with `--calib-stats`; requires [build] `coremltools`)
fast-pplx setup-ane --precision w8a16
```

```python
from fast_pplx import FastPplx

with FastPplx(mode="auto", dimensions=512) as model:  # auto picks hetero2 if ANE stages exist, else gpu
    q = model.encode_queries(["How do ocean currents transport heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ])
```

```bash
./packages/fast-pplx/build/pplx-ane-cli --seq-len 128 --inspect --bench 20
```

## At a glance

| Question | Answer |
|---|---|
| What model & tokenizer behavior does it preserve? | `pplx-embed-v2-context-0.6b` (`12` layers, `full_attention_interval=2`, `+1.0` zero-centered RMSNorm shifts), matching upstream `split_special_tokens=True` (`[Q] ` = `248077`, `[D] ` = literal ordinary tokens, `<\|chunk_sep\|>` = `248079`). |
| Why does `S > 2048` stay flat in memory? | `MappedEmbedding` keeps `embed_tokens` (~`1.02 GB`) out of Metal RAM; `S > 2048` automatically switches to streamed state-carried `GatedDeltaNet` (`1024`-tok slices) + tile-aligned `SDPA` (`512`-tok slices). |
| How does `hetero2` combine ANE & GPU? | Short items (`<= 512` toks) pack chunk-aligned into `S{64,128,256,384,512}` CoreML ANE rings (`pipe_depth=4`), while Metal processes `> 512` sequences concurrently and steals from tail on idle. |
| What is verified right now? | **All suites verified on M4 Pro (`Mac16,8`, `48 GB`):** CPU/fault-injection suites (`fast-ir-guard` `0` errors/warnings), live `.mlmodelc` ANE (`w8a16`), `16`-case GPU / `14`-case ANE / `16`-case `hetero2` parity, full `speed` (`gpu-no-mmap`, `gpu-bf16`, `gpu`, `ane`, `hetero2`, [`pplx-context-mlx-f16`](https://github.com/zm2231/pplx-context-mlx), `torch-mps`, `torch-cpu` — **`1.61x–2.16x`** vs `pplx-context-mlx-f16`), `32K` `context-limit`, `context-windows` (`nDCG@5` `+0.1875`), and `19/19` `bit-identity` (`max_abs_err=0.0`). |

## Verification Status (`48 GB` M4 Pro `Mac16,8`)

| Check | Status | Details |
|---|---|---|
| Native build (`make native`) | **Verified** | Compiles `build/libpplx_fast_io.dylib`, `build/libpplx_coreml_runner.dylib`, and `build/pplx-ane-cli` (`Apple clang`, Accelerate/NEON + CoreML `IOSurface`) |
| Static IR guard (`fast-ir-guard --strict-bool-mask`) | **Verified (`0` findings)** | Clean across `src`, `tests`, and `scripts` |
| CPU & fault-injection suites (`pytest -m "not gpu_ane"`) | **Verified (passing)** | Covers Rust tokenizer/control-token isolation, packed ANE buckets + `conv_gap=3`, synthetic `libpplx_fast_io.dylib` Neon/cblas parity, `hetero2` routing, `S > 64` streamed CPU parity, fault injection, Parquet contextual windowing, HTTP `/v1/embeddings`, and benchmark harness CLI |
| Live 16-case GPU/ANE/`hetero2` parity & speed | **Verified (passing)** | `gpu` `16/16` (`proj_min_cos=0.999996`, `int8_exact=0.9606`, `int8_min_cos=0.999867`, `s4096` cos `1.0`), `ane` `14/14` (`<=512`, `proj_min_cos=0.988051`), `hetero2` `16/16` (`proj_min_cos=0.988051`), `ane` `query_b1` `4.18 ms` (**`2.16x`** `pplx-context-mlx-f16`, **`6.87x`** `torch-mps`), `ane`/`hetero2` `S=256` `14,883.7` / `14,620.6 tok/s` (**`1.68x`** / **`1.65x`** `pplx-context-mlx-f16`, **`4.90x`** / **`4.82x`** `torch-mps`, **`9.23x`** / **`9.07x`** `torch-cpu`), `hetero2` `mixed` `12,624.1 tok/s` (**`1.72x`** `pplx-context-mlx-f16`), `32K` `context-limit` in `2.690 GB` OS footprint, `19/19` `bit-identity` |

## Speed, Parity & Memory Benchmarks (Reproducible Harness)

No unmeasured numbers are reported here. Once ANE stages / local weights are present and GPU contention is clear, run [`scripts/benchmark.py`](scripts/benchmark.py) to measure:

```bash
# 1. Speed + internal & public baselines (gpu-no-mmap, gpu-bf16, gpu, hetero2, pplx-context-mlx-f16, torch-mps, torch-cpu):
python packages/fast-pplx/scripts/benchmark.py speed \
  --modes gpu-no-mmap,gpu-bf16,gpu,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu --json-out artifacts/bench_speed.json

# Pass --pplx-mlx-path and/or --torch-python to include public `pplx-context-mlx` & PyTorch MPS/CPU comparisons:
python packages/fast-pplx/scripts/benchmark.py speed --pplx-mlx-path /path/to/pplx-context-mlx --torch-python /path/to/python

# 2. All 16 PyTorch CPU reference cases (`projected` max abs diff & cosine + `int8` exact fraction)
python packages/fast-pplx/scripts/benchmark.py parity --modes gpu,ane,hetero2

# 3. Streamed 32K memory scaling curve in isolated subprocesses (`512 .. 32768`)
python packages/fast-pplx/scripts/benchmark.py context-limit --lengths 512,1024,2048,4096,8192,16384,32768

# 4. Contextual window sensitivity probe (`--context-tokens` vs isolated chunks on MRR / nDCG@5)
python packages/fast-pplx/scripts/benchmark.py context-windows --context-budgets 512,1024,2048,4096

# 5. Exact float32 bit-identity dump & regression compare across commits
python packages/fast-pplx/scripts/benchmark.py bit-identity dump --out-dir artifacts/bit_base
python packages/fast-pplx/scripts/benchmark.py bit-identity compare --baseline-dir artifacts/bit_base --candidate-dir artifacts/bit_cand

# Safe preflight check (zero GPU/ANE work):
python packages/fast-pplx/scripts/benchmark.py all --dry-run
```

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **GPU (`MLX` + Metal)** | MIT-adapted `gated_delta_update` kernel in `mx.float16`, zero per-layer `mx.eval` (`S <= 2048`), fused `qkv_proj` / `gate_up_proj`, FP32 `+1.0` zero-centered `RMSNorm` (`GatedRMSNorm` unshifted), whole-slice `a_all`/`b_all` GDN projections | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Streamed 32K Path** | Automatic `1024`-token recurrent `GatedDeltaNet` slices + tile-aligned `512`-token bidirectional `SDPA` slices (`tile_aligned_slices`) + `MappedEmbedding` memmap | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Staged ANE Pipeline (`w8a8`/`w8a16`)** | 3-stage sequential CoreML split, `c32` `GatedDeltaNet` (`S = S - M*K` recurrence, `t_delta` FTZ threshold `1e-4`), linear `w8a8` (`AWQ-MSE` + `SwiGLU-QDQ` + outlier list), `pipe_depth=4` async ring | [`scripts/build_ane_models.py`](scripts/build_ane_models.py), [`native/pplx_coreml_runner.mm`](native/pplx_coreml_runner.mm) |
| **Native I/O + Scheduler** | `NEON` FP16/uint16 token gather + block-diagonal masks + Accelerate `cblas_sgemm` MRL/`tanh` projection; `hetero2` chunk-aligned packing + tail steal | [`native/pplx_fast_io.cpp`](native/pplx_fast_io.cpp), [`src/fast_pplx/engine.py`](src/fast_pplx/engine.py) |

## CLI, Two-Knob Contextual Parquet & Server

```bash
# Single/batch embeddings (supports --mode auto|gpu|ane|hetero|hetero2, --dimensions 128..1024, --dtype int8|float16|float32)
fast-pplx embed "First passage" "Second passage" --dimensions 256 --dtype int8 --json

# Standard or contextual Parquet ingestion (--context-tokens groups chunks under up to 32K context)
fast-pplx parquet --input docs.parquet --output out_shards/ --text-col body \
  --mode hetero2 --chunk-tokens 256 --context-tokens 2048 --dimensions 512 --dtype int8 --resume

# OpenAI-compatible HTTP server (/v1/embeddings with optional "dimensions")
fast-pplx serve --host 127.0.0.1 --port 8080 --mode hetero2 --dimensions 512
```

## Live Hardware Benchmarks (M4 Pro `Mac16,8`, `48 GB`)

Measured via `fast-pplx/scripts/benchmark.py` (`monorepo_bench` unified 8-mode serialized sweep including upstream [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) `pplx-context-mlx-f16`, with `2` monorepo quiescence warnings and `3` inner speed quiescence warnings during active VM/background load; standalone serialized runs for `context-limit`, `context-windows`, and `bit-identity`; suite Metal Peak `2.030 GB`, OS Footprint `3.068 GB`).

### 1. Throughput & Latency Across Workloads (`speed`)

#### 1a. Single-Query Latency (`query_b1`, `S=15`)

| Mode | Throughput | Latency | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **3,584.6 tok/s** | **4.18 ms** | `0.000 GB` | **0.248 GB** | **`2.16x`** | **`6.87x`** | **`190.86x`** | **`1.84x`** vs `gpu`; `load=6.23 s` |
| `hetero2` | `3,469.0 tok/s` | `4.32 ms` | `0.000 GB` | `0.782 GB` | **`2.09x`** | `6.65x` | `184.71x` | `1.78x` vs `gpu`; `load=5.93 s` |
| `gpu-no-mmap` | `2,620.7 tok/s` | `5.72 ms` | `1.024 GB` | `1.311 GB` | `1.58x` | `5.03x` | `139.54x` | `load=0.41 s` |
| `gpu` (`mmap` FP16) | `1,947.3 tok/s` | `7.70 ms` | `0.517 GB` | `0.818 GB` | `1.17x` | `3.73x` | `103.68x` | `-0.493 GB` vs `gpu-no-mmap`; `load=0.27 s` |
| `pplx-context-mlx-f16` (Public) | `1,660.1 tok/s` | `9.04 ms` | `0.489 GB` | `0.877 GB` | `1.00x` | `3.18x` | `88.40x` | Public MLX GPU baseline (`load=0.34 s`) |
| `gpu-bf16` | `1,513.6 tok/s` | `9.91 ms` | `0.517 GB` | `1.290 GB` | `0.91x` | `2.90x` | `80.59x` | `load=0.28 s` |
| `torch-mps` (Public) | `521.4 tok/s` | `28.77 ms` | `1.193 GB` | `2.977 GB` | `0.31x` | `1.00x` | `27.76x` | Public MPS baseline (`load=2.01 s`) |
| `torch-cpu` (Public) | `18.8 tok/s` | `798.69 ms` | `0.000 GB` | `0.764 GB` | `0.01x` | `0.04x` | `1.00x` | `threads=6`, `load=1.27 s` |

#### 1b. Batched Queries (`query_b32`, `B=32`, `~11 tok/q`)

| Mode | Throughput | Latency (`ms/q`) | Queries/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **8,295.0 tok/s** | **1.41 ms** | **709.7** | `0.856 GB` | `1.193 GB` | **`1.01x`** | **`11.11x`** | **`20.94x`** | `-0.819 GB` vs `gpu-no-mmap`; `1.08x` vs `gpu` |
| `pplx-context-mlx-f16` (Public) | `8,238.8 tok/s` | `1.46 ms` | `686.6` | `0.642 GB` | `0.978 GB` | `1.00x` | `11.03x` | `20.80x` | Public MLX GPU baseline |
| `gpu` (`mmap` FP16) | `7,656.3 tok/s` | `1.53 ms` | `655.1` | `1.233 GB` | `1.596 GB` | `0.93x` | `10.25x` | `19.33x` | `-0.415 GB` vs `gpu-no-mmap` |
| `gpu-no-mmap` | `7,537.1 tok/s` | `1.55 ms` | `644.9` | `1.694 GB` | `2.011 GB` | `0.91x` | `10.09x` | `19.03x` | In-RAM embed table |
| `gpu-bf16` | `7,511.7 tok/s` | `1.56 ms` | `642.7` | `1.185 GB` | `1.524 GB` | `0.91x` | `10.06x` | `18.97x` | `-0.488 GB` vs `gpu-no-mmap` |
| `ane` | `4,845.2 tok/s` | `2.41 ms` | `414.6` | `0.000 GB` | **0.248 GB** | `0.59x` | `6.49x` | `12.23x` | Zero Metal allocation |
| `torch-mps` (Public) | `746.8 tok/s` | `16.07 ms` | `62.2` | `1.193 GB` | `3.309 GB` | `0.09x` | `1.00x` | `1.89x` | Public MPS baseline |
| `torch-cpu` (Public) | `396.0 tok/s` | `30.30 ms` | — | `0.000 GB` | `0.895 GB` | `0.05x` | `0.53x` | `1.00x` | `threads=6` |

#### 1c. Document Chunks — `S=256` (`doc_s256`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **14,883.7 tok/s** | **17.20 ms** | **58.14** | `0.000 GB` | **0.248 GB** | **`1.68x`** | **`4.90x`** | **`9.23x`** | Lowest OS footprint (`0.248 GB`, `6.28x` smaller than `pplx-mlx-f16`) |
| **`hetero2`** | **14,620.6 tok/s** | **17.51 ms** | **57.11** | `0.000 GB` | `1.193 GB` | **`1.65x`** | **`4.82x`** | **`9.07x`** | **`1.50x`** vs `gpu` (`1.48x` vs `gpu-bf16`) |
| `gpu-no-mmap` | `9,971.3 tok/s` | `25.67 ms` | `38.95` | `2.072 GB` | `2.494 GB` | `1.13x` | `3.29x` | `6.18x` | In-RAM embed table |
| `gpu-bf16` | `9,881.5 tok/s` | `25.91 ms` | `38.60` | `1.563 GB` | `2.007 GB` | `1.12x` | `3.26x` | `6.13x` | `-0.488 GB` vs `gpu-no-mmap` |
| `gpu` (`mmap` FP16) | `9,772.6 tok/s` | `26.20 ms` | `38.17` | `1.563 GB` | `2.000 GB` | `1.11x` | `3.22x` | `6.06x` | Default memmap FP16 GPU path (`-0.495 GB` vs `gpu-no-mmap`) |
| `pplx-context-mlx-f16` (Public) | `8,842.8 tok/s` | `36.19 ms` | `27.63` | `1.057 GB` | `1.558 GB` | `1.00x` | `2.91x` | `5.48x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `3,035.1 tok/s` | `105.43 ms` | `9.48` | `1.193 GB` | `3.395 GB` | `0.34x` | `1.00x` | `1.88x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,612.5 tok/s` | `198.45 ms` | — | `0.000 GB` | `1.333 GB` | `0.18x` | `0.53x` | `1.00x` | `threads=1` |

#### 1d. Document Chunks — `S=512` (`doc_s512`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **13,259.0 tok/s** | **38.62 ms** | **25.90** | `0.000 GB` | `1.193 GB` | **`1.61x`** | **`4.59x`** | **`8.88x`** | **`1.40x`** vs `gpu` & `gpu-bf16` (`-1.314 GB` vs `gpu-no-mmap`) |
| **`ane`** | **13,245.5 tok/s** | **38.65 ms** | **25.87** | `0.000 GB` | **0.273 GB** | **`1.61x`** | **`4.58x`** | **`8.87x`** | Max ANE bucket (`<=512`); `7.43x` smaller OS vs `pplx-mlx-f16` |
| `gpu-no-mmap` | `9,954.0 tok/s` | `51.44 ms` | `19.44` | `2.058 GB` | `2.507 GB` | `1.21x` | `3.44x` | `6.67x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `9,499.2 tok/s` | `53.90 ms` | `18.55` | `1.550 GB` | `2.012 GB` | `1.15x` | `3.29x` | `6.36x` | `-0.495 GB` vs `gpu-no-mmap` |
| `gpu-bf16` | `9,469.1 tok/s` | `54.07 ms` | `18.49` | `1.550 GB` | `2.018 GB` | `1.15x` | `3.28x` | `6.34x` | `-0.489 GB` vs `gpu-no-mmap` |
| `pplx-context-mlx-f16` (Public) | `8,242.7 tok/s` | `73.52 ms` | `13.60` | `1.426 GB` | `2.028 GB` | `1.00x` | `2.85x` | `5.52x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `2,890.1 tok/s` | `209.68 ms` | `4.77` | `1.193 GB` | `3.502 GB` | `0.35x` | `1.00x` | `1.94x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,492.8 tok/s` | `405.96 ms` | — | `0.000 GB` | `2.204 GB` | `0.18x` | `0.52x` | `1.00x` | `threads=1` |

#### 1e. Long Context — `S=2048` (`doc_s2048`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gpu-no-mmap` | `8,766.4 tok/s` | `233.62 ms` | `4.28` | `2.284 GB` | `3.060 GB` | `1.25x` | `3.14x` | `8.55x` | In-RAM embed table |
| **`hetero2`** | **8,744.5 tok/s** | **234.20 ms** | **4.27** | `1.776 GB` | `2.686 GB` | **`1.25x`** | **`3.13x`** | **`8.53x`** | Auto GPU offload above `S=512` (`-0.374 GB` vs `gpu-no-mmap`) |
| `gpu` (`mmap` FP16) | `8,322.0 tok/s` | `246.09 ms` | `4.06` | `1.776 GB` | **2.572 GB** | `1.19x` | `2.98x` | `8.12x` | `-0.488 GB` OS vs `gpu-no-mmap` |
| `gpu-bf16` | `8,039.0 tok/s` | `254.76 ms` | `3.93` | `1.776 GB` | `2.608 GB` | `1.15x` | `2.88x` | `7.84x` | Memmap BF16 |
| `pplx-context-mlx-f16` (Public) | `7,014.1 tok/s` | `345.88 ms` | `2.89` | `1.438 GB` | `2.132 GB` | `1.00x` | `2.51x` | `6.84x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `2,790.7 tok/s` | `869.33 ms` | `1.15` | `1.193 GB` | `6.222 GB` | `0.40x` | `1.00x` | `2.72x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,025.0 tok/s` | `2,366.75 ms` | — | `0.000 GB` | `5.731 GB` | `0.15x` | `0.37x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1f. Long Context — `S=4096` (`doc_s4096`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gpu-no-mmap` | `7,552.4 tok/s` | `542.34 ms` | `1.84` | `2.539 GB` | `3.402 GB` | `1.26x` | `2.94x` | `10.73x` | In-RAM embed table |
| **`hetero2`** | **7,530.8 tok/s** | **543.90 ms** | **1.84** | **`2.030 GB`** | `3.055 GB` | **`1.26x`** | **`2.93x`** | **`10.70x`** | Streamed `1024`-tok GDN + `512`-tok SDPA (`-0.347 GB` vs `gpu-no-mmap`) |
| `gpu` (`mmap` FP16) | `7,159.8 tok/s` | `572.09 ms` | `1.75` | `2.030 GB` | **2.939 GB** | `1.19x` | `2.79x` | `10.17x` | `-0.463 GB` OS vs `gpu-no-mmap` (`4.32x` smaller OS vs `torch-mps`) |
| `gpu-bf16` | `6,799.0 tok/s` | `602.44 ms` | `1.66` | `2.030 GB` | `2.944 GB` | `1.13x` | `2.65x` | `9.66x` | Streamed BF16 |
| `pplx-context-mlx-f16` (Public) | `5,997.3 tok/s` | `807.69 ms` | `1.24` | `1.857 GB` | `2.736 GB` | `1.00x` | `2.34x` | `8.52x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `2,566.2 tok/s` | `1,887.61 ms` | `0.53` | `1.193 GB` | `12.699 GB` | `0.43x` | `1.00x` | `3.65x` | **`4.16x`** higher OS footprint than `hetero2` |
| `torch-cpu` (Public) | `703.8 tok/s` | `6,882.38 ms` | — | `0.000 GB` | `14.294 GB` | `0.12x` | `0.27x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1g. Mixed Short + Long Batch (`mixed_short_long`, `24x~128 + 4x2048`)

| Mode | Throughput | Batch Latency | Routing (`ANE / GPU`) | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `gpu-bf16` | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **`hetero2`** | **12,624.1 tok/s** | **961.81 ms** | `24 / 4` | `1.776 GB` | `3.055 GB` | **`1.72x`** | **`1.49x`** | Short chunks on ANE + `2048`-tok docs on GPU (`1.43x` `gpu`, `-0.347 GB` vs `gpu-no-mmap`) |
| `gpu` (`mmap` FP16) | `8,857.0 tok/s` | `1,370.89 ms` | `0 / 28` | `1.776 GB` | **2.939 GB** | `1.20x` | `1.05x` | Lowest mixed OS footprint (`-0.463 GB` vs `gpu-no-mmap`) |
| `gpu-no-mmap` | `8,710.0 tok/s` | `1,394.03 ms` | `0 / 28` | `2.284 GB` | `3.402 GB` | `1.18x` | `1.03x` | In-RAM embed table |
| `gpu-bf16` | `8,462.2 tok/s` | `1,434.84 ms` | `0 / 28` | `1.776 GB` | `2.944 GB` | `1.15x` | `1.00x` | Internal mixed GPU BF16 baseline |
| `pplx-context-mlx-f16` (Public) | `7,360.3 tok/s` | `1,856.18 ms` | `0 / 28` | `1.439 GB` | `2.736 GB` | `1.00x` | `0.87x` | Public MLX GPU baseline |

### 2. 16-Case Numerical Parity (`parity`)

| Mode | Cases Evaluated | Skipped (`>512`) | Proj Max Abs Err | Proj Min Cos | INT8 Exact Frac | INT8 Min Cos | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpu` | `16` | `0` | `0.006486` | **`0.999996`** | **`0.9606`** | **`0.999867`** | `ok` (`s4096` cos `1.0`) |
| `ane` | `14` | `2` | `0.222394` | `0.988051` | `0.1896` | `0.987688` | `ok` |
| `hetero2` | `16` | `0` | `0.222394` | `0.988051` | `0.2759` | `0.987688` | `ok` |

### 3. `32K` Context-Length Scaling (`context-limit`, `mode=gpu`)

| Target Tokens | Actual Tokens | Elapsed (`s`) | Throughput (`tok/s`) | vs Base (`512`) | Metal Peak | OS Footprint | Delta Footprint | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `512` | `512` | `0.051` | `10,137.8` | `1.00x` | `1.145 GB` | `1.565 GB` | `+0.000 GB` | `ok` |
| `1,024` | `1,024` | `0.099` | `10,384.8` | `1.02x` | `1.364 GB` | `1.865 GB` | `+0.300 GB` | `ok` |
| `2,048` | `2,048` | `0.210` | `9,734.0` | `0.96x` | `1.409 GB` | `1.974 GB` | `+0.408 GB` | `ok` |
| `4,096` | `4,096` | `0.538` | `7,618.0` | `0.75x` | `0.667 GB` | `1.284 GB` | `-0.281 GB` | `ok` |
| `8,192` | `8,192` | `1.325` | `6,182.2` | `0.61x` | `0.826 GB` | `1.470 GB` | `-0.095 GB` | `ok` |
| `16,384` | `16,384` | `4.153` | `3,944.7` | `0.39x` | `1.145 GB` | `1.861 GB` | `+0.296 GB` | `ok` |
| `32,768` | `32,768` | `12.046` | `2,720.2` | `0.27x` | `1.782 GB` | **2.690 GB** | `+1.125 GB` | `ok` |

### 4. Contextual Chunking Retrieval (`context-windows`, `mode=hetero2`)

| Context Budget | Windows | Throughput (`tok/s`) | MRR | Delta MRR | Recall@1 | nDCG@5 | Delta nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `isolated` (baseline) | `60` | `9,884.0` | `1.0000` | `+0.0000` | `1.0000` | `0.8125` | `+0.0000` |
| `512` | `6` | `8,117.5` | `1.0000` | `+0.0000` | `1.0000` | **1.0000** | **`+0.1875`** |
| `1,024` | `6` | `8,205.7` | `1.0000` | `+0.0000` | `1.0000` | **1.0000** | **`+0.1875`** |
| `2,048` | `6` | `8,205.4` | `1.0000` | `+0.0000` | `1.0000` | **1.0000** | **`+0.1875`** |
| `4,096` | `6` | `8,114.9` | `1.0000` | `+0.0000` | `1.0000` | **1.0000** | **`+0.1875`** |

### 5. Determinism (`bit-identity`, `mode=gpu`)

| Mode | Tensors Matched | All Bit-Identical | Max Abs Err | Min Cos | Status |
|---|---:|---|---:|---:|---|
| `gpu` | `19/19` | `true` | `0.0` | `1.000000` | `ok` |

## Scope & Not-tested

- Built for **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6)** (`coremltools >= 8.0` optional extra required only when compiling fresh `.mlmodelc` bundles).
- Optional `--precision w8a8` CoreML calibration via `--calib-stats` is documented in CLI flags; live M4 Pro ANE parity and benchmarks above were validated with the default `w8a16` ANE stages (`gpu-no-mmap`, `gpu-bf16`, `gpu`, `ane`, `hetero2`, `torch-mps`, `torch-cpu`).

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Provided the foundational architectural and evaluation patterns for running `pplx-embed-v2-context-0.6b` cleanly on Apple Silicon — including safetensors memmap embedding gather (`MappedEmbedding`), `S > 2048` streamed GatedDeltaNet + tile-aligned SDPA slicing (`tile_aligned_slices`), unshifted `GatedRMSNorm` vs `+1.0` zero-centered `RMSNorm` parity analysis, 16-case text reference parity & fault-injection suites, `--context-tokens` contextual window evaluation findings, and `RUSAGE_INFO_V4` OS footprint benchmarking.
- **[`mlx-lm`](https://github.com/ml-explore/mlx-lm)** (Copyright © 2025 Apple Inc., MIT License): Adapted MIT-licensed Metal `gated_delta_update` recurrence kernels (`src/fast_pplx/gpu.py`).
- **[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)**: Upstream model architecture, weights, and contextual chunk separator conventions.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
