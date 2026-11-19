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
| What model & tokenizer behavior does it preserve? | `pplx-embed-v2-context-0.6b` (`12` layers, `full_attention_interval=2`, `+1.0` zero-centered RMSNorm shifts), matching upstream `split_special_tokens=True` (`[Q] ` = `248077`, `[D] ` = literal ordinary tokens, `<|chunk_sep|>` = `248079`). |
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

Measured via `fast-pplx/scripts/benchmark.py` (with `benchmark_monorepo.py --allow-contended` for `speed`/`parity` and standalone serialized HW runs for `context-limit`, `context-windows`, and `bit-identity`):

| Workload | `ane` | `hetero2` | `gpu` | `torch-mps` | `torch-cpu` | Notes |
|---|---|---|---|---|---|---|
| `query_b1` latency | **`4.46 ms`** (`3,363.3 tok/s`, `0.2454 GB` OS) | `4.51 ms` (`3,328.2 tok/s`, `0.6955 GB` OS) | `8.26 ms` (`0.7982 GB` OS) | — | — | `ane` `B=1` `0.000 GB` Metal peak |
| `doc_s256` (`B=8`) | `15,096.0 tok/s` (`16.96 ms`, `58.97 docs/s`, `0.2507 GB` OS) | **`15,244.9 tok/s`** (`16.79 ms`, `59.55 docs/s`, `1.069 GB` OS) | — | **`5.50x`** slower | **`10.06x`** slower | `hetero2` vs public Torch baselines |
| `doc_s512` (`B=8`) | `13,528.2 tok/s` (`37.85 ms`, `26.42 docs/s`, `0.2925 GB` OS) | **`13,577.3 tok/s`** (`37.71 ms`, `26.52 docs/s`, `1.069 GB` OS) | — | — | — | Packed ANE bucket boundary (`<=512`) |
| `doc_s2048` | `—` (`>512`) | **`6,815.7 tok/s`** (`300.48 ms`, `2.5545 GB` OS) | — | — | — | Automatic GPU offload above `512` |
| `doc_s4096` | `—` (`>512`) | **`6,425.3 tok/s`** (`637.48 ms`, `1.57 docs/s`, `2.5545 GB` OS) | `6,325.4 tok/s` (`2.4629 GB` OS) | `2,147.7 tok/s` (`12.4922 GB` OS) | `532.9 tok/s` (`13.1128 GB` OS) | `gpu-no-mmap`: `6,087.8 tok/s`, `2.9603 GB` (`-0.497 GB` mmap savings) |
| `mixed_short_long` (`24x~128 + 4x2048`) | — | **`11,411.6 tok/s`** (`1,064.0 ms`, `ane=24,gpu=4`, `2.5906 GB` OS) | `8,502.7 tok/s` (`1,428.0 ms`, `2.4933 GB` OS) | — | — | `gpu-bf16`: `7,615.6 tok/s` (`1,594.3 ms`, `2.5741 GB` OS) -> **`1.50x`** `hetero2` speedup |

- **16-Case Numerical Parity:** `gpu` `16/16` (`proj_min_cos=0.999997`, `int8_exact=0.9556`, `int8_min_cos=0.999896`), `ane` `14/14` (`2 skipped >512`, `proj_min_cos=0.988051`), `hetero2` `16/16` (`proj_min_cos=0.988051`).
- **`32K` Context Limit (`mode=gpu`):** `7/7` lengths pass: `512` (`10,137.8 tok/s`, `1.145 GB` Metal / `1.565 GB` OS) → `2,048` (`9,734.0 tok/s`, `1.409 / 1.974 GB`) → `4,096` (`7,618.0 tok/s`, `0.667 / 1.284 GB`) → `8,192` (`6,182.2 tok/s`, `0.826 / 1.470 GB`) → `16,384` (`3,944.7 tok/s`, `1.145 / 1.861 GB`) → `32,768` (`12.046 s`, `2,720.2 tok/s`, `1.782 GB` Metal / `2.690 GB` OS).
- **Contextual Chunking (`context-windows`, `mode=hetero2`):** `nDCG@5` improves from `0.8125` (`isolated`, `MRR=1.0`, `Recall@1=1.0`) to **`1.0000` (`+0.1875`)** across `512`/`1024`/`2048`/`4096` budgets at `8,114.9–8,205.7 tok/s`.
- **Determinism (`bit-identity`, `mode=gpu`):** `19/19` tensors matched, `all_bit_identical=true`, `max_abs_err=0.0`, `min_cos=1.0`.

## Scope & Not-tested

- Built for **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 26.6)** (`coremltools >= 8.0` optional extra required only when compiling fresh `.mlmodelc` bundles).
- Optional `--precision w8a8` CoreML calibration via `--calib-stats` is documented in CLI flags; live M4 Pro ANE parity and benchmarks above were validated with the default `w8a16` ANE stages (`gpu-no-mmap`, `gpu-bf16`, `gpu`, `ane`, `hetero2`, `torch-mps`, `torch-cpu`).

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Provided the foundational architectural and evaluation patterns for running `pplx-embed-v2-context-0.6b` cleanly on Apple Silicon — including safetensors memmap embedding gather (`MappedEmbedding`), `S > 2048` streamed GatedDeltaNet + tile-aligned SDPA slicing (`tile_aligned_slices`), unshifted `GatedRMSNorm` vs `+1.0` zero-centered `RMSNorm` parity analysis, 16-case text reference parity & fault-injection suites, `--context-tokens` contextual window evaluation findings, and `RUSAGE_INFO_V4` OS footprint benchmarking.
- **[`mlx-lm`](https://github.com/ml-explore/mlx-lm)** (Copyright © 2025 Apple Inc., MIT License): Adapted MIT-licensed Metal `gated_delta_update` recurrence kernels (`src/fast_pplx/gpu.py`).
- **[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)**: Upstream model architecture, weights, and contextual chunk separator conventions.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
