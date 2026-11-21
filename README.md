# fast-silicon

Search documents, score decisions and extract structured records locally on an Apple Silicon Mac. Each engine has a Python API and a command-line tool. The workspace also includes HTTP serving, resumable Parquet jobs and local training tools.

Retrieval has the clearest speed advantage. On an M4 Pro with 48 GB and a base M5 with 24 GB, three encoders processed 4.33–11.26 times as much quality-weighted retrieval work per second as their stock MLX counterparts. Extraction presents a different choice. On 300 public M5 tasks, constrained Gemma returned 299 valid responses in 83 seconds, versus 286 in 172 seconds for stock MLX. Its field accuracy was lower. [Quality, time and memory by workload](docs/BENCHMARKS.md#3-external-deterministic-objective-outcomes-tier-1-cat-14)

## Quick start

From the repository root, install BGE and its search weights. This example uses the GPU and works without compiling Neural Engine models.

```bash
uv sync --python 3.13 --package fast-bge-small --inexact
.venv/bin/bge-small setup-weights
```

Run this with `.venv/bin/python`. It prints the passage about ocean currents.

```python
from fast_bge_small import FastBgeSmall

passages = [
    "Ocean currents carry heat from the tropics toward the poles.",
    "A piano has 88 keys.",
]
engine = FastBgeSmall(default_mode="gpu", enable_ane=False)
query = engine.encode_texts("How does heat move through the ocean?", add_query_prefix=True)
vectors = engine.encode_texts(passages)
print(passages[int((vectors @ query).argmax())])
```

For the full developer workspace, use `uv sync --extra dev --extra extract --extra build`. Preview downloads and native builds with `.venv/bin/python scripts/setup_all.py --dry-run`, or choose packages with `make setup SETUP_ARGS="--packages bge-small,pplx"`. The full setup stages approximately 35 GB. [Setup options](scripts/setup_all.py)

## Choose a package

| Job | Package | What you get |
| --- | --- | --- |
| Search short English passages | [BGE-small](packages/fast-bge-small) | 384-number vectors. A warmed 24-token query took 0.49 ms on M4 Pro and 0.79 ms on M5 on the Neural Engine. |
| Search longer passages | [EmbeddingGemma-2](packages/fast-embeddinggemma-2) | Up to 2,048 tokens per input and a choice of 128–768 output dimensions. The open Core ML exports accept up to 512 tokens. |
| Find passages whose meaning depends on nearby text | [PPLX](packages/fast-pplx) | Contextual passage vectors. On 175 legal questions, encoding surrounding text raised the ranking score from 0.359 to 0.703. |
| Score choices about a passage | [Decision models](packages/fast-decision-models) | Laya can score on the Neural Engine while the GPU is available for other work. Kev supports several questions about the same text and reuse of repeated prompts. |
| Extract records and call tools | [Gemma 4](packages/fast-gemma4) | Schema-constrained generation with per-response validity and completion status. On the M5 public suite, field F1 was 0.691 versus 0.923 for stock. Tool-call accuracy was 0.807 versus 0.800. |
| Generate structured records in large batches | [GPT-OSS](packages/fast-gpt-oss) | A 20B model tested up to batch 128 on the 48 GB M4 Pro. The 24 GB M5 was excluded by the memory policy. |
| Build an engine or a resumable processing job | [Shared runtime](packages/fast-shared) | Model setup, hardware dispatch, memory measurement, batch pipelines and serving helpers. |

## Benchmark highlights

**Effective outcome throughput (EOT)** is the sum of task scores divided by elapsed seconds. Invalid results earn zero. For the M5 extraction example, 300 tasks × 0.7487 average score ÷ 82.7 seconds gives **2.72 score units/s**. The same run produced **3.62 valid responses/s**. The separate 10.46 valid documents/s capacity test repeatedly processes eight fixtures.

The public retrieval suites compare identical datasets and scoring rules. Against stock MLX, the M4 Pro and M5 measured 9.58× and 11.26× for BGE, 4.80× and 4.33× for EmbeddingGemma, and 4.33× and 4.41× for PPLX. The measured hardware and operating systems were M4 Pro on macOS 26.6 and M5 on macOS 27.2.1. [Detailed results and reproducible commands](docs/BENCHMARKS.md)

### Native engines and public Core ML exports

On the same 500 retrieval items, native EmbeddingGemma earned **2.19× the score units/s of FluidInference FP16 and 2.17× OpenInfinity INT8 on M4 Pro**. On M5 the ratios were **2.83× and 2.70×**. The public INT8 model nearly matched native on one warm M5 query (2.59 vs 2.58 ms). For indexing, native processed 48,803 tokens/s versus 14,945. Its ranking advantage concentrated on long legal passages, where the public exports can read 512 tokens and native can read 2,048.

Open Core ML also delivered the faster option for one workload. The NMTZ Laya export earned **30% more score units/s on M4 Pro and 28% more on M5** than native Laya. On both hosts it scored 0.4993 versus 0.5050 and used more memory. For ordinary M5 Kev scoring, the saved stock MLX run was fastest at 5.06 score units/s, followed by the current native default at 4.40 and public Core ML at 2.27. [Five public exports, model revisions, timings and memory](docs/BENCHMARKS.md#open-core-ml-results)

### Base M5 (`24 GB`, `10c` `Apple10 NAX`) vs. M4 Pro (`48 GB`, `20c` `Apple9`) Scaling

The base M5 led the measured bulk encoding and training workloads. The M4 Pro had higher measured memory bandwidth and completed several single-query tests faster. Its larger memory also accommodated the 20B generator. [Hardware measurements](docs/BENCHMARKS.md#1-hardware-testbeds--empirical-roofline)

| Regime / Workload | M4 Pro (`48 GB`, `20c`) | Base M5 (`24 GB`, `10c NAX`) | Winner & Ratio |
| :--- | ---: | ---: | :--- |
| **Peak GPU Matrix (`FP16` / `Q4`)** | `7.42` / `6.40 TF` | **`12.25`** / **`14.49 TF`** | **Base M5 `1.65×` FP16 / `2.26×` Q4** |
| **Empirical DRAM BW / `M=1` Q4 Matmul** | **`210.2 GB/s`** / **`1.21 TF`** | `137.3 GB/s` / `0.46 TF` | **M4 Pro `1.53×` BW / `2.63×` `M=1` Q4** |
| **Batch Encoders (`emb2` / `pplx` / `bge`)** | `3.58` / `1.41` / `12.63 EOT` | **`4.71`** / **`1.72`** / **`13.06 EOT`** | **Base M5 `1.31×` / `1.23×` / `1.03×`** |
| **Earlier `Kev-0.8B` run (`Cat 2` / `B=8` `q/s`)** | `3.25 EOT` / `54.5 q/s` | **`5.08 EOT`** / **`86.8 q/s`** | **Base M5 `1.56×` `EOT` / `1.59×` `q/s`** |
| **Structured JSON (`fast-gemma4` Cat 4 / GPU `d/s`)** | `2.41 EOT` / `6.87 GPU d/s` | **`2.72 EOT`** / **`10.46 GPU d/s`** | **Base M5 `1.13×` `EOT` / `1.52×` GPU `d/s`** |
| **Local Training (`ModernBERT` / `GRPO` rollouts)** | `40.8 items/s` / `1.81 r/s` | **`58.2 items/s`** / **`2.61 r/s`** | **Base M5 `1.42×` train / `1.44×` `GRPO`** |
| **`B=1` Low-Latency ANE (`bge` / `pplx`) & `20B` MoE** | **`0.49 ms`** / **`4.18 ms`** / **`454 tok/s`** | `0.79 ms` / `5.57 ms` / *24 GB gated* | **M4 Pro `1.32×–1.61×` `B=1` + `48 GB` headroom** |

### Training, speculative decoding and Core ML compilation

| Capability (`scripts/benchmark_capabilities.py`) | Base M5 (`24 GB`) | M4 Pro (`48 GB`) | Speedup vs. Stock & Key Parity |
| :--- | :--- | :--- | :--- |
| **`ModernBERT-421M` Fine-Tuning** (`freeze<18`) | **`58.19 items/s`** cached (`42.26` packed, `4.89 GB`) | **`40.84 items/s`** cached (`26.63` packed, `5.06 GB`) | **`1.96×`** (M5) / **`2.02×`** (M4 Pro) vs stock MLX; `grad_cos = 0.99994` |
| **`Gemma-4 E2B` `QLoRA` + `Shared-Sys-KV`** | The 2,048-token run exhausted memory. The policy recommends `pack=1024, grad_accum=2`. | **`1,385.4 tok/s`** (`25.8 GB`; up to `1,581 tok/s`) | **`2.39×`** (up to **`2.88×`**) vs `mlx-lm` BF16 LoRA; `grad_cos = 0.9997` |
| **Disaggregated `GRPO` (`GPU Actor + ANE Critic`)** | **`2.61`** (`T=0.7`) / **`3.38`** (`BoN`) `r/s` (`8.68 GB`) | **`1.81`** (`T=0.7`) / **`3.15`** (`BoN`) `r/s` (`7.11 GB`) | **`5.22×–6.77×`** (M5) / **`8.93×–15.51×`** (M4 Pro); **`0.015–0.35 ms`** exposed ANE (`0 GB` GPU critic) |
| **`Gemma-4` `MTP` Speculative + `DraftScan`** | `B=1` **`83.4`** free / **`97.1 tok/s`** JSON (`2.49×–2.96×`) | `B=1` **`121.4 tok/s`** JSON (`2.60×`, `exact=True`); `B=4` `1.74×` | `~149 MiB` drafter; auto-disabled at `B >= 5` (`B=8` regresses to `0.56×`) |
| **`Kev-0.8B` `256`-Tok `GDN+KV` `PrefixTree`** | `B=1` `550`-tok **`2.72×`** (`6.34×` @ `1K`); `B=8` **`2.60×`** | `B=1` `550`-tok **`4.54×`** (`12.52×` @ `1K`); `B=8` **`4.96×`** | **`2.59×–6.19×`** (M5) / **`4.91×–9.89×`** (M4 Pro) vs stock `mlx-lm`; `max \|Δprob\| = 0.0` |
| **Parallel Multi-Bucket CoreML Build (`9` `BGE`)** | `5.12s ->` **`2.45s`** (`jobs=4`) | Legacy `26.80s` (`9.18s` `j=1`) -> **`4.56s`** (`jobs=4`) | **`2.01×–2.09×`** vs `jobs=1` (**`5.88×`** vs legacy); `spawn`-only, `cos = 1.000000` |

## Dynamic Silicon Dispatch (`DynamicSiliconPolicy`)

[`packages/fast-shared/src/fast_shared/runtime_lessons.py`](packages/fast-shared/src/fast_shared/runtime_lessons.py) codifies our cross-silicon findings into a zero-GPU-on-import policy layer that probes the host once at startup and resolves per-call plans in `O(1)` Python:

- **Strict `<= 80%` Unified-Memory Auto-Sizing (`recommend_silicon_budget`):** Clamps MLX working sets and adjusts batch/packing parameters automatically across `<= 16 GB` (`pack=512, grad_accum=4`), `24 GB` Base M5 (`effective_cap=19.2 GB`, `extraction B=128`, `qlora pack=1024, grad_accum=2`), and `>= 36 GB` M4 Pro/Max (`extraction B=256`, `qlora pack=2048, grad_accum=1`).
- **Chipset-Aware Speculation & Lazy Drafter (`select_speculative_policy` / `mtp_drafter="auto"`):** Enables `MTP` at `B == 1` on `Apple10+ NAX` and `1 <= B <= 4` on `Apple9` (`draft_k=2` for `B=2..4`), auto-disables `MTP` when batch tiles saturate (`B >= 5`), and lazy-loads the `~150 MiB` drafter only on the first qualifying call (**`0 MB` RAM** on `B >= 8` batch jobs).
- **RAM- & Chip-Aware ANE Ring, Hit-Aware `PrefixTree` & Work-Stealing (`ane_ring_variant="auto"`, `use_ane_ring="auto"`, `prefix_cache_mb="auto"`):** Zero-arg defaults enable `Gemma-4` ANE prefill rings automatically on `>= 32 GB` `Apple9` batches (`B >= 16`) while keeping `Apple10 NAX` on pure GPU (`10.46 valid docs/s`), select calibrated `w8` ANE rings on `<= 24 GB` hosts (`fp16` otherwise), route `Kev-0.8B` through `PrefixTree` automatically on `B == 1`, cache hits, or shared `256`-token batch prefixes (keeping cold distinct `B >= 2` waves on `B=8` `NAX`), and shift `hetero2` GPU share from `0.50` (`Apple9`) to `0.70` (`Apple10 NAX`).
- **OS & Promotion-Gated CoreAI Dispatch:** Tracks macOS version gates (`< 26.2` disables NAX assumptions; `>= 26.4` unlocks Metal 4 integer tensor ops; `>= 27.2.1` unlocks the `16`-core ANE fix & CoreAI runtime) while keeping `preferred_ane_backend="coreml_mil"` (`100%` Pure-ANE) unless `(model_name, batch_size)` passes all three promotion gates.

## Plain-Language Takeaways

- **Custom `MLX + CoreML MIL` beats stock `MLX` and stock `.aimodel` on real workloads:** Packing variable-length inputs with `FFD`, keeping `100%` Pure-ANE `conv1x1` stages running concurrently with Metal GPU, and applying `llguidance` grammar constraints deliver `1.8×–11.3×` higher `EOT` than stock runtimes while slashing memory footprint.
- **A `24 GB` Base M5 is a remarkably fast local batch and training machine — within its `19.2 GB` ceiling:** Its `10`-core `NAX` GPU outruns a `20`-core M4 Pro on batch embeddings (`1.03×–1.31×`), `Kev-0.8B` (`1.56×–1.59×`), batched JSON extraction (`1.13×–1.52×`), `ModernBERT` fine-tuning (`1.42×–1.59×`), and `GRPO` rollouts (`1.44×`). Reserve `48 GB+` Macs for `20B` MoE (`fast-gpt-oss`), `pack=2048` `QLoRA`, or latency-critical `B=1` decode.
- **CoreAI & `CoreAI-Opt` require strict promotion gates (22 audited verdicts, including rejections):** Never trust fake-quant PyTorch cosines alone — on `macOS 26.6`, `CoreAI-Opt` `w4_per_block` on `ExportBackend.CoreML` inverts compiled `.mlmodelc` logits (`-0.913` cosine; now blocked by default). On `macOS 27.2.1`, `.aimodel` exports must cast compute to **`FP16`** (unlocking `1,652` ANE ops and `23.0 ms` latency on `Laya w8_f16`, vs `94–103 ms` `MPSGraph` fallback in `FP32`), and blanket `fp4_fp8` quantization collapses `Qwen3.5-0.8B/2B` to `100% NaN` while accelerating `Qwen3.5-4B` by `+18.8%` `EOT` (`1.36×` warm latency). Full receipts in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#7-coreai-opt-track-a--os-coreai-track-b-promotion-audit-2222-gates).

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
| [`scripts/benchmark_outcomes.py`](scripts/benchmark_outcomes.py) | `16`-dataset external objective outcomes (`Cat 1`–`Cat 4`) | Deterministic quality (`S_obj,e2e`), Effective Outcome Throughput (`EOT`), 4-bucket unified RAM (`RUSAGE_INFO_V6`), quality, memory and throughput checks, and public MLX, CoreAI and Core ML baselines (`--include-public-baselines`) |
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
