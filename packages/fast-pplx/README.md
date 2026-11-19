# fast-pplx

Hardware-limit Apple Silicon inference engine for **`perplexity-ai/pplx-embed-v2-context-0.6b`** — a 12-layer hybrid Qwen3.5 (`GatedDeltaNet` + bidirectional full self-attention) contextual embedding model supporting **32K context**, **Matryoshka truncation (`128..1024`)**, **`float32`/`float16`/`int8`** outputs, and concurrent **Metal GPU + 16-core ANE (`hetero2`)** execution.
Includes a pure-Rust `tokenizers` wrapper (zero runtime `torch`/`transformers`), a custom CoreML/ObjC++ ANE async pipeline, `MappedEmbedding` memmap gather (`~1.02 GB` off GPU RAM), a streamed `S > 2048` 32K GPU path, a two-knob contextual Parquet pipeline (`--context-tokens` / `--chunk-tokens`), and an OpenAI HTTP server.

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
| What is verified right now? | **All suites verified on M4 Pro (`Mac16,8`, `48 GB`):** CPU/fault-injection suites (`fast-ir-guard` `0` errors/warnings), live `.mlmodelc` ANE (`w8a16`), `16`-case GPU / `14`-case ANE / `16`-case `hetero2` parity, full `speed` (`gpu`, `ane`, `hetero2`, `torch-mps`, `torch-cpu`), `32K` `context-limit`, `context-windows` (`nDCG@5` `+0.1875`), and `19/19` `bit-identity` (`max_abs_err=0.0`). |

## Verification Status (`48 GB` M4 Pro `Mac16,8`)

| Check | Status | Details |
|---|---|---|
| Native build (`make native`) | **Verified** | Compiles `build/libpplx_fast_io.dylib`, `build/libpplx_coreml_runner.dylib`, and `build/pplx-ane-cli` (`Apple clang`, Accelerate/NEON + CoreML `IOSurface`) |
| Static IR guard (`fast-ir-guard --strict-bool-mask`) | **Verified (`0` findings)** | Clean across `src`, `tests`, and `scripts` |
| CPU & fault-injection suites (`pytest -m "not gpu_ane"`) | **Verified (passing)** | Covers Rust tokenizer/control-token isolation, packed ANE buckets + `conv_gap=3`, synthetic `libpplx_fast_io.dylib` Neon/cblas parity, `hetero2` routing, `S > 64` streamed CPU parity, fault injection, Parquet contextual windowing, HTTP `/v1/embeddings`, and benchmark harness CLI |
| Live 16-case GPU/ANE/`hetero2` parity & speed | **Verified (passing)** | `gpu` `16/16` (`proj_min_cos=0.999997`, `int8_exact=0.9556`, `int8_min_cos=0.999896`), `ane` `14/14` (`<=512`, `proj_min_cos=0.988051`), `hetero2` `16/16` (`proj_min_cos=0.988051`), `hetero2` `S=256` `15,244.9 tok/s` (`5.50x` `torch-mps`, `10.06x` `torch-cpu`), `32K` `context-limit` in `2.690 GB` OS footprint, `19/19` `bit-identity` |

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

Measured via `fast-pplx/scripts/benchmark.py` (`monorepo_bench` with `5` background quiescence warnings for `speed` and `parity`; standalone serialized runs for `context-limit`, `context-windows`, and `bit-identity`). Note: `pplx-context-mlx-f16` was skipped because `pplx_context_mlx` was not installed in the comparison venv.

### 1. Throughput & Latency Across Workloads (`speed`)

#### 1a. Single-Query Latency (`query_b1`, `S=15`)

| Mode | Throughput | Latency | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **`ane`** | **3,363.3 tok/s** | **4.46 ms** | `0.000 GB` | **0.245 GB** | **`7.86x`** | **`238.4x`** | **`1.85x`** vs `gpu`; `load=18.64 s` |
| `hetero2` | `3,328.2 tok/s` | `4.51 ms` | `0.000 GB` | `0.696 GB` | `7.78x` | `236.0x` | `1.83x` vs `gpu`; `load=0.72 s` |
| `gpu-no-mmap` | `2,255.7 tok/s` | `6.65 ms` | `1.015 GB` | `1.306 GB` | `5.27x` | `159.9x` | `load=0.46 s` |
| `gpu-bf16` | `2,080.0 tok/s` | `7.21 ms` | `0.506 GB` | `1.293 GB` | `4.86x` | `147.5x` | `load=0.53 s` |
| `gpu` (`mmap` FP16) | `1,815.0 tok/s` | `8.26 ms` | `0.506 GB` | `0.798 GB` | `4.24x` | `128.7x` | `-0.508 GB` vs `gpu-no-mmap`; `load=0.67 s` |
| `torch-mps` (Public) | `427.7 tok/s` | `35.07 ms` | `1.193 GB` | `2.972 GB` | `1.00x` | `30.3x` | Public MPS baseline (`load=2.68 s`) |
| `torch-cpu` (Public) | `14.1 tok/s` | `1,063.43 ms` | `0.000 GB` | `0.767 GB` | `0.03x` | `1.00x` | `threads=6`, `load=1.42 s` |

#### 1b. Batched Queries (`query_b32`, `B=32`, `~11 tok/q`)

| Mode | Throughput | Latency (`ms/q`) | Queries/s | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`gpu-no-mmap`** | **9,616.0 tok/s** | **1.22 ms** | **822.8** | `1.682 GB` | `1.995 GB` | **`13.31x`** | **`30.86x`** | Peak un-mmap GPU query throughput |
| `gpu-bf16` | `8,678.8 tok/s` | `1.35 ms` | `742.6` | `1.173 GB` | `1.839 GB` | `12.02x` | `27.85x` | `-0.156 GB` vs `gpu-no-mmap` |
| `gpu` (`mmap` FP16) | `8,557.2 tok/s` | `1.37 ms` | `732.2` | `1.173 GB` | `1.487 GB` | `11.85x` | `27.46x` | `-0.508 GB` vs `gpu-no-mmap` |
| `hetero2` | `8,214.3 tok/s` | `1.42 ms` | `702.8` | `0.743 GB` | **1.069 GB** | `11.37x` | `26.36x` | `-0.926 GB` vs `gpu-no-mmap` |
| `ane` | `4,855.5 tok/s` | `2.41 ms` | `415.4` | `0.000 GB` | **0.245 GB** | `6.72x` | `15.58x` | Zero Metal allocation |
| `torch-mps` (Public) | `722.3 tok/s` | `16.61 ms` | `60.2` | `1.193 GB` | `3.369 GB` | `1.00x` | `2.32x` | Public MPS baseline |
| `torch-cpu` (Public) | `311.6 tok/s` | `38.51 ms` | — | `0.000 GB` | `0.947 GB` | `0.43x` | `1.00x` | `threads=6` |

#### 1c. Document Chunks — `S=256` (`doc_s256`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **15,244.9 tok/s** | **16.79 ms** | **59.55** | `0.000 GB` | `1.069 GB` | **`5.50x`** | **`10.06x`** | **`1.55x`** vs `gpu` (`1.52x` vs `gpu-bf16`) |
| `ane` | `15,096.0 tok/s` | `16.96 ms` | `58.97` | `0.000 GB` | **0.251 GB** | `5.45x` | `9.96x` | Lowest OS footprint (`0.251 GB`) |
| `gpu-no-mmap` | `10,886.4 tok/s` | `23.52 ms` | `42.53` | `2.032 GB` | `2.793 GB` | `3.93x` | `7.19x` | In-RAM embed table |
| `gpu-bf16` | `10,064.5 tok/s` | `25.44 ms` | `39.31` | `1.524 GB` | `2.241 GB` | `3.63x` | `6.64x` | `-0.551 GB` vs `gpu-no-mmap` |
| `gpu` (`mmap` FP16) | `9,843.2 tok/s` | `26.01 ms` | `38.45` | `1.524 GB` | `2.286 GB` | `3.55x` | `6.50x` | Default memmap FP16 GPU path |
| `torch-mps` (Public) | `2,772.2 tok/s` | `115.43 ms` | `8.66` | `1.193 GB` | `3.457 GB` | `1.00x` | `1.83x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,515.2 tok/s` | `211.19 ms` | — | `0.000 GB` | `1.375 GB` | `0.55x` | `1.00x` | `threads=1` |

#### 1d. Document Chunks — `S=512` (`doc_s512`, `B=8`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **13,577.3 tok/s** | **37.71 ms** | **26.52** | `0.000 GB` | `1.069 GB` | **`5.23x`** | **`9.96x`** | **`1.56x`** vs `gpu` |
| `ane` | `13,528.2 tok/s` | `37.85 ms` | `26.42` | `0.000 GB` | **0.293 GB** | `5.22x` | `9.92x` | Max ANE bucket (`<=512`) |
| `gpu-no-mmap` | `10,571.1 tok/s` | `48.43 ms` | `20.65` | `1.840 GB` | `2.793 GB` | `4.08x` | `7.76x` | In-RAM embed table |
| `gpu` (`mmap` FP16) | `8,681.4 tok/s` | `58.98 ms` | `16.96` | `1.332 GB` | `2.286 GB` | `3.35x` | `6.37x` | `1.03x` vs `gpu-bf16` |
| `gpu-bf16` | `8,415.2 tok/s` | `60.84 ms` | `16.44` | `1.332 GB` | `2.241 GB` | `3.24x` | `6.17x` | `-0.551 GB` vs `gpu-no-mmap` |
| `torch-mps` (Public) | `2,594.0 tok/s` | `233.61 ms` | `4.28` | `1.193 GB` | `3.566 GB` | `1.00x` | `1.90x` | Public MPS baseline |
| `torch-cpu` (Public) | `1,363.2 tok/s` | `444.54 ms` | — | `0.000 GB` | `2.245 GB` | `0.53x` | `1.00x` | `threads=1` |

#### 1e. Long Context — `S=2048` (`doc_s2048`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`gpu`** (`mmap` FP16) | **7,750.5 tok/s** | **264.24 ms** | **3.78** | `1.621 GB` | **2.463 GB** | **`2.97x`** | **`9.54x`** | `-0.497 GB` OS vs `gpu-no-mmap` |
| `gpu-no-mmap` | `7,522.5 tok/s` | `272.25 ms` | `3.67` | `2.129 GB` | `2.960 GB` | `2.88x` | `9.26x` | In-RAM embed table |
| `gpu-bf16` | `7,424.0 tok/s` | `275.86 ms` | `3.62` | `1.621 GB` | `2.502 GB` | `2.84x` | `9.13x` | Memmap BF16 |
| `hetero2` | `6,815.7 tok/s` | `300.48 ms` | `3.33` | `1.621 GB` | `2.555 GB` | `2.61x` | `8.39x` | Auto GPU offload above `S=512` |
| `torch-mps` (Public) | `2,613.2 tok/s` | `928.37 ms` | `1.08` | `1.193 GB` | `6.408 GB` | `1.00x` | `3.22x` | Public MPS baseline |
| `torch-cpu` (Public) | `812.8 tok/s` | `2,984.58 ms` | — | `0.000 GB` | `5.729 GB` | `0.31x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1f. Long Context — `S=4096` (`doc_s4096`, `B=4`)

| Mode | Throughput | Batch Latency | Docs/s | Metal Peak | OS Footprint | vs `torch-mps` | vs `torch-cpu` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`hetero2`** | **6,425.3 tok/s** | **637.48 ms** | **1.57** | `1.226 GB` | `2.555 GB` | **`2.99x`** | **`12.06x`** | Streamed `1024`-tok GDN + `512`-tok SDPA |
| `gpu-bf16` | `6,401.4 tok/s` | `639.86 ms` | `1.56` | `1.226 GB` | `2.502 GB` | `2.98x` | `12.01x` | Streamed BF16 |
| `gpu` (`mmap` FP16) | `6,325.4 tok/s` | `647.55 ms` | `1.54` | `1.226 GB` | **2.463 GB** | `2.95x` | `11.87x` | `-0.497 GB` OS vs `gpu-no-mmap` |
| `gpu-no-mmap` | `6,087.8 tok/s` | `672.82 ms` | `1.49` | `1.734 GB` | `2.960 GB` | `2.84x` | `11.42x` | In-RAM embed table |
| `torch-mps` (Public) | `2,147.7 tok/s` | `2,255.45 ms` | `0.44` | `1.193 GB` | `12.492 GB` | `1.00x` | `4.03x` | **`4.89x`** higher OS footprint than `hetero2` |
| `torch-cpu` (Public) | `532.9 tok/s` | `9,089.62 ms` | — | `0.000 GB` | `13.113 GB` | `0.25x` | `1.00x` | `threads=1` (`ane` skipped `>512`) |

#### 1g. Mixed Short + Long Batch (`mixed_short_long`, `24x~128 + 4x2048`)

| Mode | Throughput | Batch Latency | Routing (`ANE / GPU`) | Metal Peak | OS Footprint | vs `gpu-bf16` | Notes |
|---|---:|---:|---|---:|---:|---:|---|
| **`hetero2`** | **11,411.6 tok/s** | **1,064.00 ms** | `24 / 4` | `1.621 GB` | `2.591 GB` | **`1.50x`** | Short chunks on ANE + `2048`-tok docs on GPU (`1.34x` `gpu`) |
| `gpu` (`mmap` FP16) | `8,502.7 tok/s` | `1,428.02 ms` | `0 / 28` | `1.621 GB` | **2.493 GB** | `1.12x` | Lowest mixed OS footprint (`-0.503 GB` vs `gpu-no-mmap`) |
| `gpu-bf16` | `7,615.6 tok/s` | `1,594.35 ms` | `0 / 28` | `1.621 GB` | `2.574 GB` | `1.00x` | Internal mixed GPU BF16 baseline |
| `gpu-no-mmap` | `6,884.4 tok/s` | `1,763.70 ms` | `0 / 28` | `2.129 GB` | `2.996 GB` | `0.90x` | In-RAM embed table |

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
