# fast-pplx

Use `fast-pplx` to find a passage whose meaning depends on the surrounding text. On 175 legal questions, encoding each passage with its neighbors raised the score for ranking relevant passages in the first ten results from **0.359 to 0.703**. Stock contextual MLX also scored **0.703**. On the full 325-query base M5 workload, the project finished in **136.5 seconds** and stock took **601.5 seconds**. Both returned usable results for all 325 queries. [Scored retrieval runs](../../docs/benchmarks/document-workloads.json)

`fast-pplx` runs Perplexity’s `pplx-embed-v2-context-0.6b` model locally. Its mixed mode sends short inputs to the Apple Neural Engine (ANE) and long inputs to the GPU. The M4 Pro GPU processed a 32,768-token input with a measured process footprint of 2.69 GB. The package supports Python, OpenAI-compatible `/v1/embeddings`, contextual Parquet processing and a native ANE command-line tool. [Measurements](../../docs/BENCHMARKS.md)

> [!TIP]
> **Full Cross-Silicon & CoreAI Reference:** See [**`../../docs/BENCHMARKS.md`**](../../docs/BENCHMARKS.md) for complete Category 1–4 outcome tables, 4-bucket `RUSAGE_INFO_V6` memory footprints, and 22 audited CoreAI decisions, including rejected configurations.

## Quick start

```bash
make -C packages/fast-pplx native
fast-pplx setup-weights --local-files-only
# Optional: compile 100% Pure-ANE CoreML .mlmodelc stages (`w8a16` default; `--ane-profile fast|balanced|parity`)
fast-pplx setup-ane --precision w8a16
```

```python
from fast_pplx import FastPplx

with FastPplx(mode="auto", dimensions=512) as model:  # auto picks hetero2 if ANE stages exist, else gpu
    q = model.encode_queries(["How do ocean currents transport heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ], max_length=512)
```

`max_length` caps the whole document, including its prefix and chunk separators. The result contains one matrix per input document and one row per original chunk, in input order. A chunk crossing the cap uses the tokens that fit. A chunk wholly beyond it gets a zero vector. To index every passage in a longer document, split it into context windows before encoding.

```bash
./packages/fast-pplx/build/pplx-ane-cli --seq-len 128 --inspect --bench 20
```

## At a glance

| Question | Answer |
| :--- | :--- |
| What model & tokenizer behavior does it preserve? | `pplx-embed-v2-context-0.6b` (`12` layers, `full_attention_interval=2`, `+1.0` zero-centered RMSNorm shifts), matching upstream `split_special_tokens=True` (`[Q] ` = `248077`, `[D] ` = literal ordinary tokens, `<\|chunk_sep\|>` = `248079`). |
| Why does `S > 2048` stay flat in memory? | `MappedEmbedding` keeps `embed_tokens` (~`1.02 GB`) out of Metal RAM; `S > 2048` automatically switches to streamed state-carried `GatedDeltaNet` (`1024`-tok slices) + tile-aligned `SDPA` (`512`-tok slices) — holding `32,768` tokens in **`2.690 GB`** OS footprint. |
| How does `hetero2` combine ANE & GPU? | Short items (`<= 512` toks) pack chunk-aligned into `S{64,128,256,384,512}` Pure-ANE CoreML rings (`pipe_depth=4`, `0` CPU/GPU ops), while Metal processes `> 512` sequences concurrently and steals from tail on idle. |
| Which ANE profile should I choose? | Default **`fast`** for peak throughput (`13.5K tok/s` at `S=128`); **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`) closes **`73%`** of the ANE-vs-GPU cosine gap (`0.9866 -> 0.9964` corpus; `0.9909 -> 0.9988` `S=256`, `int8 MAE` `2.02 -> 0.79`) for only `4.5%–5.8%` latency cost. |

## Benchmark Highlights (Base M5 `24 GB` vs. M4 Pro `48 GB`)

### 1. Tier-1 Retrieval Outcomes (`NanoBEIR-en` + `LegalBench-RAG` `4K` Contextual, `N=325`)

| Engine / Baseline | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall / Mem) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall / Mem) | vs. Stock `MLX` (`M5` / `M4 Pro`) | M5 vs. M4 Pro |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| **`fast-pplx` (`hetero2`)** | **`0.7244`** | **`1.724`** (`136.5s`, `2.34 GB`) | **`0.7241`** | **`1.407`** (`167.3s`, `1.97 GB`) | **`4.41×`** / **`4.33×`** | **`1.23×`** |
| Stock `pplx-context-mlx` (`fp16`) | `0.7236` | `0.391` (`601.5s`, `2.12 GB`) | `0.7236` | `0.325` (`723.1s`, `2.19 GB`) | `1.00×` | `1.20×` |

The original M5 run had usable results for 150 of 325 queries. The selected rerun split oversized documents into windows and covered all 325. Its mean score exceeded stock MLX by 0.000831. The paired 95% interval ran from −0.003619 to +0.005281, leaving equivalence within the runner’s default 0.003 score-loss tolerance unresolved. The reported 4.41× compares the score units earned per second on these two runs. [Per-task results and run records](../../docs/benchmarks/document-workloads.json)

Batching accounts for part of the whole-job difference. On 384 sampled passages on M4 Pro, sorting by length reduced stock MLX’s total encoding time across the four sample sets from **20.65 to 13.34 seconds** (1.55×, using each set’s median). All 24 labeled NanoBEIR queries kept the same ten passages in the same order. The long legal windows behaved differently. Stock originally processed them one at a time. On eight complete windows that took **4.39 seconds**. Processing two at a time took **4.71 seconds** in source order and **4.36 seconds** in length order. [Matched stock runs and full-corpus padding counts](../../docs/benchmarks/pplx-padding-20261121.json)

### 2. Cross-Silicon Throughput, Latency & `32K` Scaling

| Workload | M4 Pro (`48 GB`) `fast-pplx` | Base M5 (`24 GB`, `NAX`) `fast-pplx` | vs. Stock Baselines & Notes |
| :--- | ---: | ---: | :--- |
| **`query_b1` (`S=15`)** | **`4.18 ms`** (`ane`, `0.248 GB` OS) | **`5.52 ms`** (`ane`, `0.245 GB`) / `5.57 ms` (`gpu`) | **`2.16×`** vs `pplx-context-mlx-f16` (`9.04 ms`), **`6.87×`** vs `torch-mps`, **`190.9×`** vs `torch-cpu` |
| **`doc_s256` (`B=8`)** | **`14,884 tok/s`** (`ane`) / `14,621` (`hetero2`) | **`15,235 tok/s`** (`hetero2`, `16.80 ms`) | **`1.68×`** (M4 Pro) / **`1.32×`** (M5) vs `pplx-context-mlx-f16`; `4.90×` vs `torch-mps` |
| **`doc_s512` (`B=8`)** | **`13,259 tok/s`** (`hetero2`, `38.62 ms`) | — | **`1.61×`** vs `pplx-context-mlx-f16` (`8,243`), `4.59×` vs `torch-mps`; `0.273 GB` OS (`ane`) |
| **`doc_s4096` (`B=4`)** | **`7,531 tok/s`** (`hetero2`, `3.06 GB` OS) | **`9,697 tok/s`** (`gpu-no-mmap`) / **`9,637`** (`hetero2`) | **`1.26×`** vs `pplx-context-mlx-f16`, **`4.16×`** smaller OS than `torch-mps` (`12.70 GB`); M5 `NAX` **`+28–35%`** |
| **`mixed` (`24×~128 + 4×2048`)** | **`12,624 tok/s`** (`hetero2`, `961.8 ms`) | **`16,796 tok/s`** (`hetero2`, `722.9 ms`) | **`1.72×`** vs `pplx-context-mlx-f16`, **`1.49×`** vs `gpu-bf16`; Base M5 **`1.33×`** M4 Pro |
| **`32K` Context (`context-limit`)** | **`2,720 tok/s`** (`12.05s`, **`2.690 GB` OS**) | Verified (`<= 19.2 GB` cap) | Streamed `1024`-tok GDN + `512`-tok SDPA (`1.782 GB` Metal peak) |
| **Parity, Windows & Determinism** | `16/16` GPU `min_cos=0.999996`, `19/19` bit-exact (`0.0` err) | `0` failures, `0` skips (`make test-hw`) | Contextual `--context-tokens 512..4096` lifts `nDCG@5` **`+0.1875`** (`0.8125 -> 1.0000`) |

### 3. ANE Profiles (`fast` / `balanced` / `parity`) Tradeoffs

Measured via `packages/fast-pplx/scripts/benchmark.py ane-profiles` in isolated temp homes against stock `zainmerchan/pplx-embed-v2-context-0.6b-mlx` (`fp16`):

| Profile | `S=128` Base M5 (`tok/s`) | `S=128` M4 Pro (`tok/s`) | `S=256` M4 Pro (`tok/s`) | `S=256` `f32 min_cos` vs. GPU | `S=256` `int8 MAE` vs. GPU | Speed vs. `fast` |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`fast`** (default) | **`13,542`** (`1.16×` stock) | **`13,464`** (`1.54×` stock) | **`10,311`** (`1.22×` stock) | `0.9909` *(corpus `0.9866`)* | `2.02` | `1.00×` |
| **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`) | `12,754` | `12,856` | `9,762` | **`0.9988`** *(corpus **`0.9964`**)* | **`0.79`** | `0.94–0.95×` |
| **`parity`** | `12,679` | `12,496` | `9,519` | **`0.9989`** | **`0.72`** | `0.92–0.94×` |
| Stock `pplx-context-mlx` (`fp16`) | `11,707` | `8,735` | `8,425` | `1.0000` (GPU) | — | — |

These measurements establish vector agreement and processing speed. Choosing a precision profile for search quality still requires a comparison of the same labeled queries across all three profiles.

### 4. Direct MIL (`100%` Pure-ANE `2.24 ms`) vs. `CoreAI-Opt` / `.aimodel` Findings

Audited across both **M4 Pro `macOS 26.6`** (`Track A`) and **Base M5 `macOS 27.2.1`** (`Track B`) in [`../../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#7-coreai-audit-of-22-decisions):

| Export / Runtime Path | Placement (`ANE / GPU / CPU`) | Warm Latency (`s64 st2` / Cat-1) | Parity / Outcome | Promotion Verdict |
| :--- | :---: | :---: | :--- | :---: |
| **Production Direct MIL `W8A16`** (`build_stage_mil_program`) | **`100%` Pure-ANE** (`369/369`, `306/306`, `375/375`, `317/317`) | **`2.24 ms`** (`st2 s64`, `85.9 MB`); **`3.585 EOT`** (M5 `N=24` smoke) | `real_cos = 0.9994` (`st2`), `0` NaN | **PROMOTED** (default) |
| `CoreAI-Opt` `ExportBackend.CoreML` (`macOS 26.6`, `fp16..w4_pb`) | **`0/349` ANE** (`100%` CPU; `_ANECompiler: ANECCompile()` fail) | `9.29 ms` (`4.1×` slower) | `st3 w4_pb` drops to `0.891` cos | **FAIL** (disqualified) |
| `macOS 27.2.1` `.aimodel` `w8` / `w4_pb` (`FP32` compute) | **`0` ANE / `336` GPU** (`MPSGraph` fallback) | `130.6 ms` warm; `0.660 EOT` (**`0.184×`** live `hetero2`) | `w4_pb` worse on speed, RAM (`3.07 GB`) & cos | **FAIL** (Gate 3) |
| `macOS 27.2.1` `.aimodel` `st0 s128 w8_f16` (`FP16` + `softplus` rewrite) | `FP16` compute + `relu + log1p(exp(-|x|))` | `2.997 ms` (`2.8×` faster than FP32 `st0`) | `cos = 0.9999` vs teacher | *Ablation receipt* |

> Direct hand-crafted 4D NCHW `conv1x1` MIL (`build_stage_mil_program`) remains the default ANE backend because `torch.export` lowering of chunked `GatedDeltaNet` block-inversion graphs fails ANE compilation on `macOS 26.6` and requires `FP16` compute + `F.softplus` replacement to avoid `MPSGraph` fallback on `27.2.1`.

## What keeps it fast

| Component | Optimization | Source |
| :--- | :--- | :--- |
| **GPU (`MLX` + Metal)** | MIT-adapted `gated_delta_update` kernel in `mx.float16`, zero per-layer `mx.eval` (`S <= 2048`), fused `qkv_proj` / `gate_up_proj`, FP32 `+1.0` zero-centered `RMSNorm` (`GatedRMSNorm` unshifted), whole-slice `a_all`/`b_all` GDN projections | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Streamed `32K` Path** | Automatic `1024`-token recurrent `GatedDeltaNet` slices + tile-aligned `512`-token bidirectional `SDPA` slices (`tile_aligned_slices`) + `MappedEmbedding` memmap | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Staged Pure-ANE Pipeline** | Direct MIL 4D `conv1x1` stages (`369..375` ANE ops, `0` CPU/GPU), `c32` `GatedDeltaNet` (`t_delta` FTZ floor `1e-4`), `fast`/`balanced`/`parity` profiles (`MaxAbsRMSNorm` + `TanhSiLU`), `pipe_depth=4` async ring | [`scripts/build_ane_models.py`](scripts/build_ane_models.py), [`native/pplx_coreml_runner.mm`](native/pplx_coreml_runner.mm) |
| **Native I/O + Scheduler** | `NEON` FP16/uint16 token gather + block-diagonal masks + Accelerate `cblas_sgemm` MRL/`tanh` projection; `hetero2` chunk-aligned packing + tail steal | [`native/pplx_fast_io.cpp`](native/pplx_fast_io.cpp), [`src/fast_pplx/engine.py`](src/fast_pplx/engine.py) |

## CLI, Two-Knob Contextual Parquet & Server

```bash
# Single/batch embeddings (--mode auto|gpu|ane|hetero|hetero2, --dimensions 128..1024, --dtype int8|float16|float32)
fast-pplx embed "First passage" "Second passage" --dimensions 256 --dtype int8 --json

# Standard or contextual Parquet ingestion (--context-tokens groups chunks under up to 32K context)
fast-pplx parquet --input docs.parquet --output out_shards/ --text-col body \
  --mode hetero2 --chunk-tokens 256 --context-tokens 2048 --dimensions 512 --dtype int8 --resume

# OpenAI-compatible HTTP server (/v1/embeddings with optional "dimensions")
fast-pplx serve --host 127.0.0.1 --port 8080 --mode hetero2 --dimensions 512
```

## Reproducible Benchmark Harness

Run [`scripts/benchmark.py`](scripts/benchmark.py) (all GPU/ANE runs self-serialize under `/tmp/m4_gpu_validation.lock`):

```bash
# Safe preflight check (zero GPU/ANE work)
python packages/fast-pplx/scripts/benchmark.py all --dry-run

# 1. Speed + internal & public baselines
python packages/fast-pplx/scripts/benchmark.py speed \
  --modes gpu-no-mmap,gpu-bf16,gpu,ane,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu \
  --pplx-mlx-path /path/to/pplx-context-mlx --torch-python /path/to/python \
  --json-out artifacts/bench_speed.json

# 2. ANE profile ABBA sweep (fast / balanced / parity in isolated temp homes)
python packages/fast-pplx/scripts/benchmark.py ane-profiles \
  --profiles fast,balanced,parity --seq-lens 128,256 --repeats 2 \
  --pplx-mlx-path /path/to/pplx-context-mlx --torch-python /path/to/python

# 3. 16-case PyTorch CPU reference parity, 32K scaling, contextual windows & bit-identity
python packages/fast-pplx/scripts/benchmark.py parity --modes gpu,ane,hetero2
python packages/fast-pplx/scripts/benchmark.py context-limit --lengths 512,1024,2048,4096,8192,16384,32768
python packages/fast-pplx/scripts/benchmark.py context-windows --context-budgets 512,1024,2048,4096
python packages/fast-pplx/scripts/benchmark.py bit-identity dump --out-dir artifacts/bit_base
```

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Base Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)** (`coremltools >= 8.0` optional extra required only when compiling fresh `.mlmodelc` bundles).
- Default live ANE benchmarks use `w8a16` (`fast`/`balanced`/`parity`); optional `--precision w8a8` CoreML calibration via `--calib-stats` is supported via CLI flags.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Provided the foundational architectural and evaluation patterns for running `pplx-embed-v2-context-0.6b` cleanly on Apple Silicon — including safetensors memmap embedding gather (`MappedEmbedding`), `S > 2048` streamed GatedDeltaNet + tile-aligned SDPA slicing (`tile_aligned_slices`), unshifted `GatedRMSNorm` vs `+1.0` zero-centered `RMSNorm` parity analysis, 16-case text reference parity & fault-injection suites, `--context-tokens` contextual window evaluation findings, and unified-memory OS footprint benchmarking.
- **[`mlx-lm`](https://github.com/ml-explore/mlx-lm)** (Copyright © 2025 Apple Inc., MIT License): Adapted MIT-licensed Metal `gated_delta_update` recurrence kernels (`src/fast_pplx/gpu.py`).
- **[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)**: Upstream model architecture, weights, and contextual chunk separator conventions.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
