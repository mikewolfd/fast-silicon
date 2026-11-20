# fast-silicon

**Run local embeddings, decision scoring, guaranteed-valid JSON extraction, 20B MoE inference, and local post-training on Apple Silicon up to 4.3×–11.3× faster in effective outcome throughput — at a fraction of the unified memory.**

`fast-silicon` is a `uv` workspace of focused Apple Silicon inference and post-training engines, CLI tools, OpenAI-compatible HTTP servers, and resumable Parquet pipelines built on Apple MLX, CoreML (16-core Neural Engine), and Metal 4. By scheduling work across GPU (`Apple9` / `Apple10 NAX`) and ANE without heavy PyTorch runtime bloat — and gating every optimization on deterministic **Effective Outcome Throughput (`EOT`)** and **`<= 80%` unified-memory ceilings** — the workspace beats both stock **`MLX`** (`mlx-embeddings`, `mlx-vlm`, `mlx-lm`, `pplx-context-mlx`) and stock **Apple `CoreAI`** (`.aimodel` on `macOS 27.2.1`) across **Apple M4 Pro (`48 GB`)** and **Base Apple M5 (`24 GB`)**.

> [!TIP]
> **Full Technical Reference:** See [**`docs/BENCHMARKS.md`**](docs/BENCHMARKS.md) for complete Category 1–4 objective outcome tables, 4-bucket `RUSAGE_INFO_V6` memory breakdowns, training/speculative/ANE-profile sweeps, the `22/22` audited `CoreAI-Opt` & `macOS 27.2.1` `CoreAI` promotion-gate matrix, and exact reproducible CLI commands.

> [!NOTE]
> **Verified Hardware Baselines:** Built and validated on **Apple M4 Pro (`Mac16,8`, `48 GB`, `20c` `Apple9` GPU, `16c` ANE `h16g`, `macOS 26.6`)** and cross-validated across all 7 package test suites (`0` failures, `0` skips) plus full Tier-1 outcome, capability, and `CoreAI` sweeps on **Base Apple M5 (`Mac17,2`, `24 GB`, `10c` `Apple10` GPU + `NAX`, `16c` ANE `h17`, `macOS 27.2.1`, Xcode CLT-only)**. Other Apple Silicon chips emit a non-fatal startup note (`FAST_SILICON_NO_HW_WARN=1` to mute).

## Quick start

```bash
uv sync --extra dev --extra extract --extra build  # Create .venv (`extract`: Parquet/schema; `build`: Torch/CoreML ANE export)
make hw-check                                      # Probe Apple Silicon host capabilities (chip, NAX, RAM, OS gates)
make setup                                         # Build native libs, download HF repos, export weights, compile ANE bundles (~35 GB)
make guard                                         # Run fast-ir-guard --strict-bool-mask (`0` errors)
make test-cpu                                      # Run offline CPU + fault-injection + native parity suites
```

```bash
# Preview `make setup` plan (idempotent; zero GPU/ANE/network work)
.venv/bin/python scripts/setup_all.py --dry-run

# Useful `setup_all.py` / `make setup SETUP_ARGS="..."` options:
#   --skip-gpt-oss --skip-test-extras   Skip ~24.8 GB optional checkpoints (recommended on 24 GB Macs)
#   --skip-ane                          Skip CoreML .mlmodelc compilation
#   --packages bge-small,pplx           Stage only selected package(s)
#   --target auto|in-repo|cache         Choose layout (or `--artifacts-root <dir>`)

# Dry-run canonical benchmarks (zero GPU/ANE work)
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages all --dry-run
.venv/bin/python scripts/benchmark_monorepo.py --dry-run
.venv/bin/python scripts/benchmark_capabilities.py --suites all --dry-run
```

## Workspace Packages

| Package | Model | Best For / Capabilities |
| :--- | :--- | :--- |
| [`packages/fast-shared`](packages/fast-shared) | Shared runtime & toolkit | `DynamicSiliconPolicy`, `RUSAGE_INFO_V6` 4-bucket memory probe (`FourBucketMemoryProbe`), `fast-ir-guard`, `parallel_coreml` (`spawn`-only), `coreai_pipeline`, `MappedEmbedding`, `12,032`-entry capability matrix, Agent plugin |
| [`packages/fast-bge-small`](packages/fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Sub-ms (`0.49–0.79 ms`) search queries & `187K–201K tok/s` batch indexing (`11.26×` M5 / `9.58×` M4 Pro `EOT` vs `mlx-embeddings`, OpenAI HTTP, Parquet, `bge-ane-cli`) |
| [`packages/fast-embeddinggemma-2`](packages/fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | `2048`-context MRL (`128..768`-d) retrieval (`48.8K` M5 / `38.5K` M4 Pro `tok/s`; **`6.24×` `EOT` vs Apple `CoreAI`** and `4.33×–4.80×` vs `mlx-vlm`, OpenAI HTTP, Parquet, `emb2-ane-cli`) |
| [`packages/fast-pplx`](packages/fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | `32K` late-chunking contextual embeddings in `2.69 GB` OS (`4.41×` M5 / `4.33×` M4 Pro `EOT` vs `pplx-context-mlx`, `fast`/`balanced`/`parity` ANE profiles, MRL `128..1024`, `--context-tokens` Parquet) |
| [`packages/fast-decision-models`](packages/fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Pure-ANE document scoring (`Laya`: **`8.29×` `EOT` vs Apple `CoreAI`**, `0` GPU ops) + batched `Kev-0.8B` (`86.8 q/s` on M5, `PrefixTree` up to `6.19×` `B=8` / `9.89×` `B=1` vs `mlx-lm`) + `ModernBERT` fine-tuning (`58.2 items/s`) |
| [`packages/fast-gemma4`](packages/fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Guaranteed schema-valid batch JSON & tool calling (`10.46 valid docs/s` M5 / `8.78 ANE+GPU` M4 Pro = **`4.88×–5.18×` valid docs/s** vs `mlx-vlm`, `int8` KV, `MTP` `auto`, `QLoRA`, `GRPO` actor) |
| [`packages/fast-gpt-oss`](packages/fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B` `MXFP4` MoE) | High-concurrency 20B structured generation on `>= 48 GB` Macs (`454.1 dec tok/s` at `B=128` = `2.91×` vs `mlx-lm`, `49/50` vs `32/50` valid docs, Hydragen shared-prefix KV, OpenAI SSE, Parquet) |

## Benchmark Highlights

### 1. Headline Speedups vs. Stock `MLX` & Apple `CoreAI` (`Tier-1` `EOT` & Monorepo Speed)

`EOT = (N / T_e2e) * S_obj,e2e` across zero-LLM-judge public datasets (`NanoBEIR-en`, `LegalBench-RAG`, `RAGTruth`, `CLINC150`, `RewardBench-2`, `IFBench`, `MMLU-Redux-2.0`, `MATH-500`, `LiquidAI/structured-extraction`, `BFCL v3`). Full tables with 4-bucket memory in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#3-external-deterministic-objective-outcomes-tier-1-cat-14).

| Package & Workload | Base M5 (`24 GB`) `fast-*` | M4 Pro (`48 GB`) `fast-*` | Speedup vs. Stock `MLX` (`M5` / `M4 Pro`) | Speedup vs. Stock Apple `CoreAI` (`27.2.1` M5) |
| :--- | :--- | :--- | :---: | :---: |
| **`fast-bge-small`** (`Cat 1`, `hetero`) | **`13.06 EOT`** (`200.6K tok/s`, `1.54 GB`) | **`12.63 EOT`** (`187.2K tok/s`, `1.55 GB`) | **`11.26×`** / **`9.58×`** `EOT` (`7.52×` / `6.10×` `tok/s`) | — *(no CoreAI export)* |
| **`fast-embeddinggemma-2`** (`Cat 1`, `hetero`) | **`4.71 EOT`** (`48.8K tok/s`, `3.43 GB`) | **`3.58 EOT`** (`38.5K tok/s`, `2.49 GB`) | **`4.33×`** / **`4.80×`** `EOT` (`3.03×` / `3.37×` `tok/s`) | **`6.24×`** `EOT` (`3.91×` `tok/s`) |
| **`fast-pplx`** (`Cat 1`, `hetero2` ctx `4K`) | **`1.72 EOT`** (`16.8K tok/s` mixed, `2.34 GB`) | **`1.41 EOT`** (`14.3K tok/s` `s256`, `1.97 GB`) | **`4.41×`** / **`4.33×`** `EOT` (`1.32×` / `1.66×` `tok/s`) | **`5.43×`** `EOT` *(vs `CoreAI w8` pilot `0.184×`)* |
| **`fast-decision-models`** (`Laya` Pure-ANE, `Cat 2`) | **`6.16 EOT`** (`44.4 q/s`, `1.00 GB`) | **`5.83 EOT`** (`41.9 q/s`, `0.98 GB`) | **`1.16×`** / **`1.92×`** `EOT` *(stock MLX loses `-0.23` acc)* | **`8.29×`** `EOT` (`0.5050` vs `0.4993` acc) |
| **`fast-decision-models`** (`Kev-0.8B`, `Cat 2`) | **`5.08 EOT`** (`86.8 q/s` `B=8`, `1.77 GB`) | **`3.25 EOT`** (`54.5 q/s` `B=8`, `1.84 GB`) | **`-42%` RAM** (up to `6.19×` M5 / **`9.89×` M4 Pro** w/ `PrefixTree`) | — |
| **`fast-gemma4`** (`Cat 4` JSON/Tools, `llg_on`) | **`2.72 EOT`** (`299/300`; **`10.46 valid d/s`**) | **`2.41 EOT`** (`299/300`; **`8.78 valid d/s`**) | **`1.81×`** / **`1.57×`** `EOT` (**`5.18×`** / **`4.88×`** valid `d/s`) | — |
| **`fast-gpt-oss`** (`Cat 4` `20B` MoE, `llg_on`) | *Gated (`24 GB` cap)* | **`0.32 EOT`** (`49/50` valid; **`454.1 tok/s`**) | — / **`1.61×` `EOT`** (`+53%` valid; **`2.91×`** `B=128` `tok/s`) | — |

### 2. Base M5 (`24 GB`, `10c` `Apple10 NAX`) vs. M4 Pro (`48 GB`, `20c` `Apple9`) Scaling

Despite half the GPU cores and `35%` less DRAM bandwidth (`137.3` vs `210.2 GB/s`), Base M5's per-core `NAX` matrix engines (`12.25 TF` FP16 / `14.49 TF` Q4 vs `7.42–7.61 TF`) outpace M4 Pro on every compute-bound batch and training workload, while M4 Pro leads bandwidth-bound `M=1` decode, `B=1` ANE latency, and `>19 GB` models ([full roofline table](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline)):

| Regime / Workload | M4 Pro (`48 GB`, `20c`) | Base M5 (`24 GB`, `10c NAX`) | Winner & Ratio |
| :--- | ---: | ---: | :--- |
| **Peak GPU Matrix (`FP16` / `Q4`)** | `7.42` / `6.40 TF` | **`12.25`** / **`14.49 TF`** | **Base M5 `1.65×` FP16 / `2.26×` Q4** |
| **Empirical DRAM BW / `M=1` Q4 Matmul** | **`210.2 GB/s`** / **`1.21 TF`** | `137.3 GB/s` / `0.46 TF` | **M4 Pro `1.53×` BW / `2.63×` `M=1` Q4** |
| **Batch Encoders (`emb2` / `pplx` / `bge`)** | `3.58` / `1.41` / `12.63 EOT` | **`4.71`** / **`1.72`** / **`13.06 EOT`** | **Base M5 `1.31×` / `1.23×` / `1.03×`** |
| **Batched `Kev-0.8B` (`Cat 2` / `B=8` `q/s`)** | `3.25 EOT` / `54.5 q/s` | **`5.08 EOT`** / **`86.8 q/s`** | **Base M5 `1.56×` `EOT` / `1.59×` `q/s`** |
| **Structured JSON (`fast-gemma4` Cat 4 / GPU `d/s`)** | `2.41 EOT` / `6.87 GPU d/s` | **`2.72 EOT`** / **`10.46 GPU d/s`** | **Base M5 `1.13×` `EOT` / `1.52×` GPU `d/s`** |
| **Local Training (`ModernBERT` / `GRPO` rollouts)** | `40.8 items/s` / `1.81 r/s` | **`58.2 items/s`** / **`2.61 r/s`** | **Base M5 `1.42×` train / `1.44×` `GRPO`** |
| **`B=1` Low-Latency ANE (`bge` / `pplx`) & `20B` MoE** | **`0.49 ms`** / **`4.18 ms`** / **`454 tok/s`** | `0.79 ms` / `5.57 ms` / *24 GB gated* | **M4 Pro `1.32×–1.61×` `B=1` + `48 GB` headroom** |

### 3. Training, RLVR, Speculative & CoreML Build Highlights

| Capability (`scripts/benchmark_capabilities.py`) | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Speedup vs. Stock & Key Parity |
| :--- | :--- | :--- | :--- |
| **`ModernBERT-421M` Fine-Tuning** (`freeze<18`) | **`58.19 items/s`** cached (`42.26` packed, `4.89 GB`) | **`40.84 items/s`** cached (`26.63` packed, `5.06 GB`) | **`1.96×`** (M5) / **`2.02×`** (M4 Pro) vs stock MLX; `grad_cos = 0.99994` |
| **`Gemma-4 E2B` `QLoRA` + `Shared-Sys-KV`** | Auto-sized `pack=1024, grad_accum=2` (`<=19.2 GB`) | **`1,385.4 tok/s`** (`25.8 GB`; up to `1,581 tok/s`) | **`2.39×`** (up to **`2.88×`**) vs `mlx-lm` BF16 LoRA; `grad_cos = 0.9997` |
| **Disaggregated `GRPO` (`GPU Actor + ANE Critic`)** | **`2.61`** (`T=0.7`) / **`3.38`** (`BoN`) `r/s` (`8.68 GB`) | **`1.81`** (`T=0.7`) / **`3.15`** (`BoN`) `r/s` (`7.11 GB`) | **`5.22×–6.77×`** (M5) / **`8.93×–15.51×`** (M4 Pro); **`0.015–0.35 ms`** exposed ANE (`0 GB` GPU critic) |
| **`Gemma-4` `MTP` Speculative + `DraftScan`** | `B=1` **`83.4`** free / **`97.1 tok/s`** JSON (`2.49×–2.96×`) | `B=1` **`121.4 tok/s`** JSON (`2.60×`, `exact=True`); `B=4` `1.74×` | `~149 MiB` drafter; auto-disabled at `B >= 5` (`B=8` regresses to `0.56×`) |
| **`Kev-0.8B` `256`-Tok `GDN+KV` `PrefixTree`** | `B=1` `550`-tok **`2.72×`** (`6.34×` @ `1K`); `B=8` **`2.60×`** | `B=1` `550`-tok **`4.54×`** (`12.52×` @ `1K`); `B=8` **`4.96×`** | **`2.59×–6.19×`** (M5) / **`4.91×–9.89×`** (M4 Pro) vs stock `mlx-lm`; `max \|Δprob\| = 0.0` |
| **Parallel Multi-Bucket CoreML Build (`9` `BGE`)** | `5.12s ->` **`2.45s`** (`jobs=4`) | Legacy `26.80s` (`9.18s` `j=1`) -> **`4.56s`** (`jobs=4`) | **`2.01×–2.09×`** vs `jobs=1` (**`5.88×`** vs legacy); `spawn`-only, `cos = 1.000000` |

## Dynamic Silicon Dispatch (`DynamicSiliconPolicy`)

[`packages/fast-shared/src/fast_shared/runtime_lessons.py`](packages/fast-shared/src/fast_shared/runtime_lessons.py) codifies our cross-silicon findings into a zero-GPU-on-import policy layer that probes the host once at startup and resolves per-call plans in `O(1)` Python:

- **Strict `<= 80%` Unified-Memory Auto-Sizing (`recommend_silicon_budget`):** Clamps MLX working sets and adjusts batch/packing parameters automatically across `<= 16 GB` (`pack=512, grad_accum=4`), `24 GB` Base M5 (`effective_cap=19.2 GB`, `extraction B=128`, `qlora pack=1024, grad_accum=2`), and `>= 36 GB` M4 Pro/Max (`extraction B=256`, `qlora pack=2048, grad_accum=1`).
- **Chipset-Aware Speculation & Lazy Drafter (`select_speculative_policy` / `mtp_drafter="auto"`):** Enables `MTP` at `B == 1` on `Apple10+ NAX` and `1 <= B <= 4` on `Apple9` (`draft_k=2` for `B=2..4`), auto-disables `MTP` when batch tiles saturate (`B >= 5`), and lazy-loads the `~150 MiB` drafter only on the first qualifying call (**`0 MB` RAM** on `B >= 8` batch jobs).
- **RAM-Aware ANE Ring & Work-Stealing (`ane_ring_variant="auto"` / `hetero2`):** Defaults generative ANE prefixes to `fp16`, opting into `w8` on `<= 24 GB` hosts only when an `AWQ-MSE`/QAT calibration manifest is verified; shifts `hetero2` GPU share from `0.50` (`Apple9`) to `0.70` (`Apple10 NAX`) and enables `Apple10` fused SWA SDPA on `S ∈ {384, 512}`.
- **OS & Promotion-Gated CoreAI Dispatch:** Tracks macOS version gates (`< 26.2` disables NAX assumptions; `>= 26.4` unlocks Metal 4 integer tensor ops; `>= 27.2.1` unlocks the `16`-core ANE fix & CoreAI runtime) while keeping `preferred_ane_backend="coreml_mil"` (`100%` Pure-ANE) unless `(model_name, batch_size)` passes all three promotion gates.

## Plain-Language Takeaways

- **Custom `MLX + CoreML MIL` beats stock `MLX` and stock `.aimodel` on real workloads:** Packing variable-length inputs with `FFD`, keeping `100%` Pure-ANE `conv1x1` stages running concurrently with Metal GPU, and applying `llguidance` grammar constraints deliver `1.8×–11.3×` higher `EOT` than stock runtimes while slashing memory footprint.
- **A `24 GB` Base M5 is a remarkably fast local batch and training machine — within its `19.2 GB` ceiling:** Its `10`-core `NAX` GPU outruns a `20`-core M4 Pro on batch embeddings (`1.03×–1.31×`), `Kev-0.8B` (`1.56×–1.59×`), batched JSON extraction (`1.13×–1.52×`), `ModernBERT` fine-tuning (`1.42×–1.59×`), and `GRPO` rollouts (`1.44×`). Reserve `48 GB+` Macs for `20B` MoE (`fast-gpt-oss`), `pack=2048` `QLoRA`, or latency-critical `B=1` decode.
- **CoreAI & `CoreAI-Opt` require strict promotion gates (`22/22` audited):** Never trust fake-quant PyTorch cosines alone — on `macOS 26.6`, `CoreAI-Opt` `w4_per_block` on `ExportBackend.CoreML` inverts compiled `.mlmodelc` logits (`-0.913` cosine; now blocked by default). On `macOS 27.2.1`, `.aimodel` exports must cast compute to **`FP16`** (unlocking `1,652` ANE ops and `23.0 ms` latency on `Laya w8_f16`, vs `94–103 ms` `MPSGraph` fallback in `FP32`), and blanket `fp4_fp8` quantization collapses `Qwen3.5-0.8B/2B` to `100% NaN` while accelerating `Qwen3.5-4B` by `+18.8%` `EOT` (`1.36×` warm latency). Full receipts in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#7-coreai-opt-track-a--os-coreai-track-b-promotion-audit-2222-gates).

## Hardware Suites, Weight Setup & Serialization

Model weights and compiled `.mlmodelc` bundles live outside git. Stage the full workspace in one command (`make setup`, or `--skip-gpt-oss --skip-test-extras` on `24 GB` Macs — individual packages also support `<cli> setup-weights` / `setup-ane`), then run serialized hardware validation:

```bash
make test-hw    # Checks host baseline, then runs all 7 package suites + root setup tests sequentially
```

> [!IMPORTANT]
> **Serialize GPU & ANE workloads:** Run one Metal GPU or CoreML ANE suite/benchmark at a time under `/tmp/m4_gpu_validation.lock`. [`scripts/benchmark_outcomes.py`](scripts/benchmark_outcomes.py), [`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py), [`scripts/benchmark_capabilities.py`](scripts/benchmark_capabilities.py), and [`packages/fast-pplx/scripts/benchmark.py`](packages/fast-pplx/scripts/benchmark.py) enforce `check_system_quiescence()` and `fcntl.flock` locking automatically.

## Benchmark & Investigation Harnesses

| Script | Scope | Captures |
| :--- | :--- | :--- |
| [`scripts/benchmark_outcomes.py`](scripts/benchmark_outcomes.py) | `16`-dataset external objective outcomes (`Cat 1`–`Cat 4`) | Deterministic quality (`S_obj,e2e`), Effective Outcome Throughput (`EOT`), 4-bucket unified RAM (`RUSAGE_INFO_V6`), Two-Gate `ABBA`/`compare`, and paired stock `MLX`/`CoreAI` baselines (`--include-public-baselines`) |
| [`scripts/benchmark_monorepo.py`](scripts/benchmark_monorepo.py) | Cross-package throughput & footprint sweep | Isolated per-package throughput + internal ablations (`gpu` vs `hetero`, `kv_i8`, `B=1` vs `B=128`, `--gemma4-ane-variant fp16\|w8`) + apples-to-apples public baselines ([`scripts/public_baselines.py`](scripts/public_baselines.py)) |
| [`scripts/benchmark_capabilities.py`](scripts/benchmark_capabilities.py) | Post-training, co-execution, speculative & stateful `ABBA` | `qlora-packing`, `encoder-finetune`, `grpo-ane-critic`, `mtp-speculative`, `prefix-tree`, and `parallel-coreml` vs stock MLX |
| [`packages/fast-pplx/scripts/benchmark.py`](packages/fast-pplx/scripts/benchmark.py) | `fast-pplx` deep investigation | `speed`, `parity` (`16` refs), `ane-profiles` (`fast`/`balanced`/`parity`), `context-limit` (`512..32K`), `context-windows`, and `bit-identity` |

## Agent Plugin (`apple-silicon-ml-optimizer`) & Repo Sync

```bash
claude --plugin-dir ./packages/fast-shared
make sync              # Mirror clean files from ../fast-* -> packages/* (keeps workspace deps)
make sync-check        # Fail if synced packages/* differ from sibling ../fast-* checkouts
make sync-clean-check  # Also fail if any sibling ../fast-* checkout has uncommitted changes
make leak-check        # Fail on private domain tokens or local absolute user paths (includes docs/)
```

**Scope & not tested:** Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Base Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)** across all 7 package suites (`0` failures, `0` skips). `fast-gpt-oss` (`~12 GB` `20B` MoE checkpoint) and `pack=2048` `Gemma-4 E2B` `QLoRA` require `>= 36–48 GB` unified memory and are gated on `24 GB` hosts (`fast-gpt-oss` CPU/contract tests still pass on M5). Untested on M1/M2/M3/M4 Max/Ultra/M5 Pro/Max Macs, Linux, x86/CUDA, or concurrent unserialized hardware workloads.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspiring the unified-memory footprint methodology, the clean README layout across this workspace, and key design & evaluation patterns adopted in `fast-pplx` and `fast-shared` — including memmap checkpoint embedding loading (`MappedEmbedding`), tile-aligned streamed attention slicing, 16-case text parity / bit-identity / fault-injection verification, and contextual window retrieval sensitivity probing.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
