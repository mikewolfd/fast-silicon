# fast-pplx

**32K-context `pplx-embed-v2-context-0.6b` embeddings in 2.69 GB of unified memory — 1.71x–2.42x faster than `pplx-context-mlx` and 6.3x smaller than PyTorch MPS.**

`fast-pplx` preserves exact upstream tokenization, late-chunking context (`[Q]`/`[D]`/`<|chunk_sep|>`), and Matryoshka dimensions (`128..1024`, `float32`/`float16`/`int8`), while scheduling short chunks (`<=512` tok) onto the Apple Neural Engine and streaming long documents on Metal GPU. That delivers **3.93 ms** single queries (**2.28x** vs `pplx-context-mlx-f16`, **11.95x** vs `torch-mps`), **15,384 tok/s** at `S=256` (**1.71x**), and **13,935 tok/s** (**2.42x**) on mixed batches — with zero `torch`/`transformers` runtime dependencies and built-in Python, OpenAI `/v1/embeddings`, contextual Parquet CLI, and `pplx-ane-cli` interfaces.

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
| What is verified right now? | **All suites verified on M4 Pro (`Mac16,8`, `48 GB`):** CPU/fault-injection suites (`fast-ir-guard` `0` errors/warnings), live `.mlmodelc` ANE (`w8a16`), `16`-case GPU / `14`-case ANE / `16`-case `hetero2` parity, full `speed` (`gpu-no-mmap`, `gpu-bf16`, `gpu`, `ane`, `hetero2`, [`pplx-context-mlx-f16`](https://github.com/zm2231/pplx-context-mlx), `torch-mps`, `torch-cpu` — **`1.71x–2.42x`** vs `pplx-context-mlx-f16`), `32K` `context-limit`, `context-windows` (`nDCG@5` `+0.1875`), and `19/19` `bit-identity` (`max_abs_err=0.0`). |

## Verification Status (`48 GB` M4 Pro `Mac16,8`)

| Check | Status | Details |
|---|---|---|
| Native build (`make native`) | **Verified** | Compiles `build/libpplx_fast_io.dylib`, `build/libpplx_coreml_runner.dylib`, and `build/pplx-ane-cli` (`Apple clang`, Accelerate/NEON + CoreML `IOSurface`) |
| Static IR guard (`fast-ir-guard --strict-bool-mask`) | **Verified (`0` findings)** | Clean across `src`, `tests`, and `scripts` |
| CPU & fault-injection suites (`pytest -m "not gpu_ane"`) | **Verified (passing)** | Covers Rust tokenizer/control-token isolation, packed ANE buckets + `conv_gap=3`, synthetic `libpplx_fast_io.dylib` Neon/cblas parity, `hetero2` routing, `S > 64` streamed CPU parity, fault injection, Parquet contextual windowing, HTTP `/v1/embeddings`, and benchmark harness CLI |
| Live 16-case GPU/ANE/`hetero2` parity & speed | **Verified (passing)** | `gpu` `16/16` (`proj_min_cos=0.999997`, `int8_exact=0.9556`, `int8_min_cos=0.999896`), `ane` `14/14` (`<=512`, `proj_min_cos=0.988051`), `hetero2` `16/16` (`proj_min_cos=0.988051`), `ane` `query_b1` `3.93 ms` (**`2.28x`** `pplx-context-mlx-f16`, **`11.95x`** `torch-mps`), `ane`/`hetero2` `S=256` `15,384.2` / `15,361.9 tok/s` (**`1.71x`** `pplx-context-mlx-f16`, **`2.95x`** `torch-mps`, **`11.18x`** `torch-cpu`), `hetero2` `mixed` `13,935.0 tok/s` (**`2.42x`** `pplx-context-mlx-f16`), `32K` `context-limit` in `2.690 GB` OS footprint, `19/19` `bit-identity` |

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

Measured via `fast-pplx/scripts/benchmark.py` (`monorepo_bench` unified 8-mode serialized sweep including upstream [`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx) `pplx-context-mlx-f16`, with `6` background quiescence warnings for `speed` and `parity`; standalone serialized runs for `context-limit`, `context-windows`, and `bit-identity`).

### 1. Throughput & Latency Across Workloads (`speed`)

#### 1a. Single-Query Latency (`query_b1`, `S=15`)

| Mode | Throughput | Latency | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **3,817.2 tok/s** | **3.93 ms** | `0.000 GB` | **0.243 GB** | **`2.28x`** | **`11.95x`** | **`224.4x`** | **`2.47x`** vs `gpu`; `load=12.91 s` |
| `hetero2` | `3,475.4 tok/s` | `4.32 ms` | `0.000 GB` | `0.684 GB` | **`2.07x`** | `10.88x` | `204.3x` | `2.25x` vs `gpu`; `load=0.61 s` |
| `gpu-no-mmap` | `2,477.2 tok/s` | `6.06 ms` | `1.015 GB` | `1.302 GB` | `1.48x` | `7.76x` | `145.6x` | `load=0.42 s` |
| `gpu-bf16` | `2,080.4 tok/s` | `7.21 ms` | `0.506 GB` | `1.271 GB` | `1.24x` | `6.52x` | `122.3x` | `load=0.33 s` |
| `pplx-context-mlx-f16` (Public) | `1,675.7 tok/s` | `8.95 ms` | `0.489 GB` | `0.890 GB` | `1.00x` | `5.25x` | `98.5x` | Public MLX GPU baseline (`load=0.40 s`) |
| `gpu` (`mmap` FP16) | `1,543.6 tok/s` | `9.72 ms` | `0.506 GB` | `0.796 GB` | `0.92x` | `4.83x` | `90.8x` | `-0.507 GB` vs `gpu-no-mmap`; `load=0.42 s` |
| `torch-mps` (Public) | `319.3 tok/s` | `46.98 ms` | `1.193 GB` | `2.875 GB` | `0.19x` | `1.00x` | `18.8x` | Public MPS baseline (`load=1.24 s`) |
| `torch-cpu` (Public) | `17.0 tok/s` | `881.81 ms` | `0.000 GB` | `0.644 GB` | `0.01x` | `0.05x` | `1.00x` | `threads=6`, `load=0.27 s` |

#### 1b. Batched Queries (`query_b32`, `B=32`, `~11 tok/q`)

| Mode | Throughput | Latency (`ms/q`) | Queries/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`gpu-bf16`** | **9,844.1 tok/s** | **1.19 ms** | **842.3** | `1.173 GB` | `1.888 GB` | **`1.17x`** | **`5.54x`** | **`26.56x`** | Peak batched GPU query throughput (`-0.101 GB` vs `gpu-no-mmap`) |
| `gpu-no-mmap` | `8,893.0 tok/s` | `1.31 ms` | `760.9` | `1.682 GB` | `1.989 GB` | `1.06x` | `5.01x` | `24.00x` | In-RAM embed table |
| `pplx-context-mlx-f16` (Public) | `8,420.8 tok/s` | `1.43 ms` | `701.7` | `0.642 GB` | `0.987 GB` | `1.00x` | `4.74x` | `22.72x` | Public MLX GPU baseline |
| `hetero2` | `8,419.7 tok/s` | `1.39 ms` | `720.4` | `0.743 GB` | `1.057 GB` | `1.00x` | `4.74x` | `22.72x` | `-0.932 GB` vs `gpu-no-mmap`; `1.03x` vs `gpu` |
| `gpu` (`mmap` FP16) | `8,207.4 tok/s` | `1.42 ms` | `702.2` | `1.173 GB` | `1.486 GB` | `0.97x` | `4.62x` | `22.15x` | `-0.504 GB` vs `gpu-no-mmap` |
| `ane` | `4,914.1 tok/s` | `2.38 ms` | `420.5` | `0.000 GB` | **0.243 GB** | `0.58x` | `2.77x` | `13.26x` | Zero Metal allocation |
| `torch-mps` (Public) | `1,775.9 tok/s` | `6.76 ms` | `148.0` | `1.193 GB` | `3.358 GB` | `0.21x` | `1.00x` | `4.79x` | Public MPS baseline |
| `torch-cpu` (Public) | `370.6 tok/s` | `32.38 ms` | — | `0.000 GB` | `0.927 GB` | `0.04x` | `0.21x` | `1.00x` | `threads=6` |

#### 1c. Document Chunks — `S=256` (`doc_s256`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **15,384.2 tok/s** | **16.64 ms** | **60.09** | `0.000 GB` | **0.250 GB** | **`1.71x`** | **`2.96x`** | **`11.20x`** | Lowest OS footprint (`0.250 GB`, `6.28x` smaller than `pplx-mlx-f16`) |
| **`hetero2`** | **15,361.9 tok/s** | **16.66 ms** | **60.01** | `0.000 GB` | `1.057 GB` | **`1.71x`** | **`2.95x`** | **`11.18x`** | **`1.69x`** vs `gpu` (`1.31x` vs `gpu-bf16`) |
| `gpu-bf16` | `11,717.3 tok/s` | `21.85 ms` | `45.77` | `1.524 GB` | `2.227 GB` | `1.30x` | `2.25x` | `8.53x` | `-0.518 GB` vs `gpu-no-mmap` |
| `gpu-no-mmap` | `10,280.8 tok/s` | `24.90 ms` | `40.16` | `2.032 GB` | `2.744 GB` | `1.14x` | `1.98x` | `7.48x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `9,101.2 tok/s` | `28.13 ms` | `35.55` | `1.524 GB` | `2.241 GB` | `1.01x` | `1.75x` | `6.63x` | Default memmap FP16 GPU path (`-0.503 GB` vs `gpu-no-mmap`) |
| `pplx-context-mlx-f16` (Public) | `8,984.9 tok/s` | `35.62 ms` | `28.08` | `1.057 GB` | `1.569 GB` | `1.00x` | `1.73x` | `6.54x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `5,200.9 tok/s` | `61.53 ms` | `16.25` | `1.193 GB` | `3.483 GB` | `0.58x` | `1.00x` | `3.79x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,373.5 tok/s` | `232.97 ms` | — | `0.000 GB` | `1.305 GB` | `0.15x` | `0.26x` | `1.00x` | `threads=1` |

#### 1d. Document Chunks — `S=512` (`doc_s512`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **13,677.4 tok/s** | **37.43 ms** | **26.71** | `0.000 GB` | **0.271 GB** | **`1.73x`** | **`2.66x`** | **`11.03x`** | Max ANE bucket (`<=512`); `7.30x` smaller OS vs `pplx-mlx-f16` |
| **`hetero2`** | **13,619.1 tok/s** | **37.59 ms** | **26.60** | `0.000 GB` | `1.062 GB` | **`1.73x`** | **`2.65x`** | **`10.98x`** | **`1.42x`** vs `gpu` (`1.21x` vs `gpu-bf16`) |
| `gpu-bf16` | `11,228.8 tok/s` | `45.60 ms` | `21.93` | `1.332 GB` | `2.227 GB` | `1.42x` | `2.18x` | `9.05x` | `-0.518 GB` vs `gpu-no-mmap` |
| `gpu-no-mmap` | `10,806.8 tok/s` | `47.38 ms` | `21.11` | `1.840 GB` | `2.744 GB` | `1.37x` | `2.10x` | `8.71x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `9,612.1 tok/s` | `53.27 ms` | `18.77` | `1.332 GB` | `2.241 GB` | `1.22x` | `1.87x` | `7.75x` | `-0.503 GB` vs `gpu-no-mmap` |
| `pplx-context-mlx-f16` (Public) | `7,893.3 tok/s` | `76.77 ms` | `13.03` | `1.426 GB` | `1.978 GB` | `1.00x` | `1.53x` | `6.36x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `5,143.0 tok/s` | `117.83 ms` | `8.49` | `1.193 GB` | `3.924 GB` | `0.65x` | `1.00x` | `4.15x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,240.3 tok/s` | `488.59 ms` | — | `0.000 GB` | `2.108 GB` | `0.16x` | `0.24x` | `1.00x` | `threads=1` |

#### 1e. Long Context — `S=2048` (`doc_s2048`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **9,963.5 tok/s** | **205.55 ms** | **4.86** | `1.621 GB` | `2.480 GB` | **`1.64x`** | **`2.70x`** | **`9.40x`** | Auto GPU offload above `S=512` (`-0.458 GB` vs `gpu-no-mmap`) |
| `gpu-bf16` | `9,731.9 tok/s` | `210.44 ms` | `4.75` | `1.621 GB` | `2.486 GB` | `1.60x` | `2.64x` | `9.18x` | Memmap BF16 |
| `gpu-no-mmap` | `9,026.3 tok/s` | `226.89 ms` | `4.41` | `2.129 GB` | `2.938 GB` | `1.48x` | `2.44x` | `8.52x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `8,954.9 tok/s` | `228.70 ms` | `4.37` | `1.621 GB` | **2.461 GB** | `1.47x` | `2.42x` | `8.45x` | `-0.477 GB` OS vs `gpu-no-mmap` |
| `pplx-context-mlx-f16` (Public) | `6,087.2 tok/s` | `398.54 ms` | `2.51` | `1.438 GB` | `2.047 GB` | `1.00x` | `1.65x` | `5.74x` | Public MLX GPU baseline |
| `torch-mps` (Public) | `3,693.0 tok/s` | `656.91 ms` | `1.52` | `1.193 GB` | `8.564 GB` | `0.61x` | `1.00x` | `3.48x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,059.9 tok/s` | `2,288.87 ms` | — | `0.000 GB` | `3.766 GB` | `0.17x` | `0.29x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1f. Long Context — `S=4096` (`doc_s4096`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **7,857.7 tok/s** | **521.27 ms** | **1.92** | **1.226 GB** | `2.480 GB` | **`1.51x`** | **`2.52x`** | **`10.93x`** | Streamed `1024`-tok GDN + `512`-tok SDPA (`-0.248 GB` OS vs `pplx-mlx-f16`) |
| `gpu-bf16` | `7,580.7 tok/s` | `540.32 ms` | `1.85` | `1.226 GB` | `2.486 GB` | `1.46x` | `2.43x` | `10.54x` | Streamed BF16 |
| `gpu` (`mmap` FP16) | `7,368.4 tok/s` | `555.88 ms` | `1.80` | `1.226 GB` | **2.461 GB** | `1.42x` | `2.36x` | `10.25x` | `-0.477 GB` OS vs `gpu-no-mmap` |
| `gpu-no-mmap` | `7,104.3 tok/s` | `576.55 ms` | `1.73` | `1.734 GB` | `2.938 GB` | `1.37x` | `2.28x` | `9.88x` | In-RAM embed table |
| `pplx-context-mlx-f16` (Public) | `5,190.5 tok/s` | `933.24 ms` | `1.07` | `1.857 GB` | `2.728 GB` | `1.00x` | `1.66x` | `7.22x` | Public MLX GPU baseline (`+0.631 GB` Metal vs `hetero2`) |
| `torch-mps` (Public) | `3,118.0 tok/s` | `1,553.58 ms` | `0.64` | `1.193 GB` | `15.690 GB` | `0.60x` | `1.00x` | `4.34x` | **`6.33x`** higher OS footprint than `hetero2` |
| `torch-cpu` (Public) | `719.2 tok/s` | `6,735.20 ms` | — | `0.000 GB` | `6.569 GB` | `0.14x` | `0.23x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1g. Mixed Short + Long Batch (`mixed_short_long`, `24x~128 + 4x2048`)

| Mode | Throughput | Batch Latency | Routing (`ANE / GPU`) | Metal Peak | OS Footprint | vs `pplx-mlx-f16` | vs `gpu-bf16` | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **`hetero2`** | **13,935.0 tok/s** | **871.33 ms** | `24 / 4` | `1.621 GB` | `2.522 GB` | **`2.42x`** | **`1.75x`** | Short chunks on ANE + `2048`-tok docs on GPU (`1.49x` `gpu`, `-0.206 GB` OS vs `pplx-mlx-f16`) |
| `gpu-no-mmap` | `10,028.4 tok/s` | `1,210.76 ms` | `0 / 28` | `2.130 GB` | `2.957 GB` | `1.74x` | `1.26x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `9,331.9 tok/s` | `1,301.12 ms` | `0 / 28` | `1.621 GB` | **2.497 GB** | `1.62x` | `1.17x` | Lowest mixed OS footprint (`-0.460 GB` vs `gpu-no-mmap`) |
| `gpu-bf16` | `7,952.9 tok/s` | `1,526.74 ms` | `0 / 28` | `1.621 GB` | `2.559 GB` | `1.38x` | `1.00x` | Internal mixed GPU BF16 baseline |
| `pplx-context-mlx-f16` (Public) | `5,762.7 tok/s` | `2,370.75 ms` | `0 / 28` | `1.439 GB` | `2.728 GB` | `1.00x` | `0.72x` | Public MLX GPU baseline |

### 2. 16-Case Numerical Parity (`parity`)

| Mode | Cases Evaluated | Skipped (`>512`) | Proj Max Abs Err | Proj Min Cos | INT8 Exact Frac | INT8 Min Cos | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpu` | `16` | `0` | `0.010290` | **`0.999997`** | **`0.9556`** | **`0.999896`** | `ok` |
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
