# fast-silicon

Hardware-limit Apple Silicon ML inference, embedding, and typed-decision workspace (`uv` monorepo) combining Metal 4 `MPP::matmul2d`, 16-core Apple Neural Engine (`CoreML` / `IOSurface` zero-copy), and CPU SME/AMX pipelines.

> [!WARNING]
> **Tested Hardware Baseline Only — Please Read Before Running**
>
> Everything in this monorepo was **only built, tuned, and tested** on a single **14-inch MacBook Pro (`Mac16,8`)** with:
> - **Chip:** Apple M4 Pro (`arm64`, macOS / Metal 4)
> - **Memory:** 48 GB unified memory (`51,539,607,552` bytes)
> - **CPU:** 14 cores (10 Performance + 4 Efficiency)
> - **GPU:** 20 cores
> - **Neural Engine (ANE):** 16 cores (`h16g`)
>
> Running on other Apple Silicon Macs (different memory capacity, core counts, M1/M2/M3/M4 Max/M5) or non-macOS hosts is **untested** and may hit out-of-memory errors, default batch/slot exhaustion, CoreML E5RT compilation failures, or different throughput.
>
> This monorepo automatically checks your host hardware via `make hw-check` (and before `native` / `test-cpu` / `test-hw`, plus at CLI startup in every package), reports the detected Mac model and core counts, and emits a stderr warning whenever your machine does not match the tested 48 GB M4 Pro (`14 CPU [10P+4E]`, `20 GPU`, `16 ANE`) baseline. Set `FAST_SILICON_NO_HW_WARN=1` to mute runtime CLI stderr warnings after acknowledging them.

## Host Hardware Check

```bash
uv run fast-silicon-hw-check          # Human-readable host check (warns if != Mac16,8 48 GB M4 Pro)
uv run fast-silicon-hw-check --json   # Machine-readable JSON report (model, RAM, P/E CPU, GPU, ANE cores)
uv run fast-silicon-hw-check --strict # Exit 2 if host differs from tested baseline
make hw-check                         # Run directly from the monorepo root
```

On the tested machine, `fast-silicon-hw-check` reports:

```text
[fast-silicon] Host verified: Mac16,8 (Darwin/arm64) | Apple M4 Pro | 48 GB RAM | 14 CPU cores (10P + 4E) | 20 GPU cores | 16 ANE cores (matches tested baseline)
```

## Workspace Packages

| Package | Model / Scope | Key Capabilities & Engines | Verified Benchmark / Suite (`48 GB` M4 Pro) |
| --- | --- | --- | --- |
| [`packages/fast-shared`](packages/fast-shared) | Shared runtime & tools | Metal 4 `MPP` (`FP16`/`BF16`), CoreML/Accelerate headers, `fast-ir-guard`, `fast-bench-check` (`ABBA`), `fast-silicon-probe`, `fast-silicon-audit`, `fast-silicon-hw-check`, OpenAI HTTP/SSE servers, resumable Parquet pipelines, `apple-silicon-ml-optimizer` plugin | `52 passed, 3 skipped` (`llguidance`/`coremltools` extras in standalone venv), `0` guard warnings |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Metal 4 MPP GPU + 16-core ANE (`hetero`) + standalone `bge-ane-cli` | Up to **`215,022 tok/s`** OCR / **`189,338 tok/s`** synth (`0.493 ms` ANE `B=1`); `7 passed` |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | Metal 4 MPP GPU + 16-core ANE + CPU SME (`hetero2`/`hetero`) | Up to **`36,911 tok/s`** synth / **`32,569 tok/s`** OCR (`1.871 ms` ANE `B=1`); `32 passed` |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Rolling `B_slots=128` Metal 4 GPU (`kv_i8`) + Early14 Linear-W8 ANE ring (`1,326` ANE ops, zero-copy `IOSurface -> KV_i8`) + `llguidance` + `forward_ff_jump_slot` + PromptLookup | **`671.1–704.8` e2e out tok/s** (`878.9–1,051.8` dec tok/s, `2.33 docs/s`, `6.88 GB`), Linear-W8 ANE **`7,814.8 tok/s`**; `44 passed` |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B`/`3.6B` act. `MXFP4` MoE) | Unified Hot-MPP + Cold-QMV MoE, Hydragen v2 + SWA-128, `131K` pruned `lm_head`, `llguidance` + `skip_lm_head` + GPU `top128` nucleus | **`462.8 dec tok/s`** (`0.710 docs/s`, `16.56 GB` @ `B=128`), isolated **`97.4 tok/s`** `B=1`; `7 passed` |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M W8A16` + `Kev-0.8B r15` | Pure 16-core ANE (`[1585,0,0]`, `P=4` `IOSurface` + NEON transpose) & Metal MLX left-padded judge (`w8`/`bf16`, stdlib ZIP `head.pt`) | `laya_ane`: **`41.72 q/s`** (`23.97 ms/q`); `kev_08b` (`w8`): **`59.68 q/s`** (`16.76 ms/q`); `11 passed` (`10 passed, 1 skipped` without staged `laya_typed`) |
| [`packages/fast-pplx`](packages/fast-pplx) | `perplexity-ai/pplx-embed-v2-context-0.6b` | `Qwen3.5` hybrid GatedDeltaNet + Full Attention, MLX GPU + `hetero2` ANE runner + `MappedEmbedding` | Pending build/bench verification |

## Quickstart & Commands

```bash
# 1. Create workspace virtualenv & check host MacBook compatibility
uv sync --extra dev
make hw-check

# 2. Build CoreML / Objective-C++ native libraries across native packages
make native

# 3. Run static Metal / CoreML dtype guard & offline CPU tests
make guard
make test-cpu
```

### Hardware Suites & Weight Artifacts

Model weights and compiled `.mlmodelc` bundles are excluded from git. Run the per-package `setup-weights` / `setup-ane` CLIs (see individual package `README.md` files) or point the corresponding `*_HOME` / `FAST_DECISION_MODELS_LAYA_DIR` environment variables to staged artifacts before running the full serialized hardware suite:

```bash
make test-hw   # Checks host hardware, then runs each package's pytest suite sequentially
```

> [!IMPORTANT]
> Run GPU and ANE benchmarks/tests **one at a time** (serialized). Do not run concurrent Metal GPU or CoreML ANE workloads against `/tmp/m4_gpu_validation.lock`.

## Agent Plugin (`apple-silicon-ml-optimizer`)

This repository includes an Agent / Claude Code / Codex plugin marketplace manifest ([`.agents/plugins/marketplace.json`](.agents/plugins/marketplace.json)) exposing [`packages/fast-shared`](packages/fast-shared) (`apple-silicon-ml-optimizer` skill + M4/M5 roofline references and probe scripts):

```bash
claude --plugin-dir ./packages/fast-shared
```

## Syncing from Standalone Sibling Checkouts (`../fast-*`)

If you also maintain standalone sibling `../fast-*` repositories locally, mirror clean tracked/unignored files into `packages/*` and verify cleanliness with:

```bash
make sync         # Sync ../fast-* -> packages/* (rewrites workspace refs cleanly)
make sync-check   # Fail if packages/* has drifted from ../fast-*
make leak-check   # Verify zero private or local user-path strings in tracked files
```
