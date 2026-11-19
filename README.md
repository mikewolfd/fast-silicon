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

| Package | Model | Engine Summary | Verified Benchmark / Status |
|---|---|---|---|
| [`packages/fast-shared`](packages/fast-shared) | Runtime, kernels & tooling | Metal 4 `MPP` (`FP16`/`BF16`), CoreML/Accelerate headers, `MappedEmbedding`, `fast-ir-guard`, `RUSAGE_INFO_V4` footprint (`fast-bench-check`), `fast-silicon-probe`/`audit`/`hw-check`, Parquet & HTTP SSE, Agent plugin | CPU + `fast-ir-guard` verified (`0` findings); cross-generation M4/M5 optimization plugin |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Metal 4 MPP + 16-core ANE (`hetero`) + standalone `bge-ane-cli` | **`207,264 tok/s`** `hetero` (**`2.50x`** `transformers-mps`), `136,416 tok/s` `gpu`, `0.53 ms` ANE `B=1` (**`3.83x`** GPU) |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | Metal 4 MPP + 16-core ANE (`W8A16`) + CPU SME (`hetero2`/`hetero`) + `emb2-ane-cli` | **`31,786 tok/s`** `hetero2` (**`1.85x`** `gpu` `17,177 tok/s`), `2.19 ms` ANE `B=1` (**`4.36x`** GPU `9.54 ms`) |
| [`packages/fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | `Qwen3.5` hybrid GDN + SDPA up to `32K`, `hetero2` ANE (`w8a16`) runner, `MappedEmbedding`, MRL `128..1024`, `--context-tokens` Parquet + `benchmark.py` | **`15,244.9 tok/s`** `hetero2` `doc_s256` (**`5.50x`** `torch-mps`, **`10.06x`** `torch-cpu`), `11,411.6 tok/s` mixed (**`1.50x`** `bf16`), `32K` in `2.690 GB` OS; `16/16` parity & `19/19` bit-identical |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Rolling `B_slots=128` Metal 4 (`kv_i8`) + Early14 Linear-W8 ANE ring (`use_ane_ring="auto"` / `1,326` ops) + `llguidance` + `forward_ff_jump_slot` + PromptLookup | **`8.92 docs/s`** (`1,192 dec tok/s`, `6.718 GB` Metal) concurrent ANE+GPU (**`2.54x`** `stock-mlx_vlm`, **`1.21x`** GPUOnly `7.36 docs/s`), Spec `516 eff tok/s` |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) | Unified Hot-MPP + Cold-QMV MoE, Hydragen v2 + SWA-128, `131K` pruned `lm_head`, `llguidance` | **`555.5 dec tok/s`** / `1.874 docs/s` at `B=128` (**`4.78x`** `stock-mlx_lm` `116.3 tok/s`, **`7.42x`** vs `B=1` tok/s), `410.0 tok/s` `B=64`, `162.1 tok/s` `B=8` |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Pure ANE (`[1585,0,0]`, `P=4` `IOSurface` + NEON) & `W8` MLX left-padded judge (stdlib `zipfile` `head.pt`) | `kev_08b` (`w8`) **`50.95 q/s`** (`19.63 ms/q`, **`5.27x`** `transformers-mps-bf16` `9.67 q/s`); `laya_ane` **`40.53 q/s`** (`24.67 ms/q`, **`4.19x`** `transformers-mps-bf16`) |

## Live Monorepo Showcase (`scripts/benchmark_monorepo.py`)

Full 6-package `extract_v1` / embedding / decision showcase on M4 Pro (`Mac16,8`, `48 GB`, recorded under active background CPU/memory pressure warnings; `make guard` `0` errors / `0` warnings; `make sync-check` & `make leak-check` pass):

| Package | Throughput / Highlight | Internal Baseline | Public Apples-to-Apples | Metal Peak (GB) | OS Footprint (GB) | Verify | Status |
|---|---|---|---|---:|---:|---|---|
| `fast-bge-small` | `136,416 tok/s`; `56,808 tok/s`; `207,264 tok/s` | `hetero` `207,264` vs `gpu` `136,416 tok/s` (`1.52x`); `B=1` ANE `0.53ms` vs GPU `2.02ms` (`3.83x`) | vs `transformers-mps-float16`: **`2.50x`** (`207,264` vs `82,883 tok/s`) | `1.173` | `4.813` | `ok` | `ok` |
| `fast-embeddinggemma-2` | `17,177 tok/s`; `20,328 tok/s`; `31,786 tok/s` | `hetero2` `31,786` vs `gpu` `17,177 tok/s` (`1.85x`); `B=1` ANE `2.19ms` vs GPU `9.54ms` (`4.36x`) | skipped (`requires transformers>=5.18 (embedding_gemma2 unsupported)`) | `2.224` | `10.820` | `ok` | `ok` |
| `fast-pplx` | `hetero2` `doc` `B=8` `S=256`: `15,244.9 tok/s` (`16.79 ms`); `B=1` `ane` `4.46ms` | `mixed_short_long` `hetero2` vs `bf16`: `1.50x`; `B=1` `ane` `4.46ms` vs `gpu` `8.26ms`; mmap `dFootprint=-0.497GB` | `doc_s256` (vs `torch-mps`: **`5.50x`**; vs `torch-cpu`: **`10.06x`**; `S=4K` `2.5` vs `12.5GB`) | `1.621` | `2.591` | `ok` | `ok` |
| `fast-gemma4` | GPU: `1,265 tok/s` (`7.36 docs/s`); Spec: `516 eff tok/s`; ANE+GPU: `1,192 tok/s` (`8.92 docs/s`) | `Hetero` vs `GPUOnly`: `1.21x` (`8.92` vs `7.36 docs/s`); `kv_i8` `mem=-0.67GB` | vs `stock-mlx_vlm`: **`2.54x`** (`8.92` vs `3.51 docs/s`; `1.48x` `tok/s`, `dPeak=-3.50GB`) | `6.718` | `18.314` | `ok` | `ok` |
| `fast-gpt-oss` | `batched_B8`: `162.1 dec tok/s` (`0.888 docs/s`); `batched_B64`: `410.0 dec tok/s` (`1.636 docs/s`); `batched_B128`: `555.5 dec tok/s` (`1.874 docs/s`) | `batched_B128` vs `B=1`: `7.42x` `tok/s` (`4.04x` `docs/s`) | vs `stock-mlx_lm`: **`4.78x`** (`555.5` vs `116.3 tok/s`) | `15.901` | `21.709` | `ok` | `ok` |
| `fast-decision-models` | `laya_ane`: `40.53 q/s` (`24.67 ms/q`); `kev_08b`: `50.95 q/s` (`19.63 ms/q`) | `w8` `B=8` vs `bf16` `B=8`: `0.99x`; `dFootprint=-0.553GB`; `w8` `B=8` vs `bf16` `B=1`: `1.86x` | vs `transformers-mps-bf16`: **`5.27x`** (`51.0` vs `9.7 q/s`, `dFP=-2.13GB`; `laya_ane` **`4.19x`**) | `1.748` | `3.166` | `ok` | `ok` |

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
