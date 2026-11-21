# fast-decision-models

Use typed Laya to help a person prioritize extracted records for review. In a 48-document pilot with 12 known defects, the first ten selected by the existing rules contained four defects. Laya alone found five, and rules plus Laya found seven. Making Laya a blocking check increased M5 extraction time from 93.31 to 160.71 seconds while held-out quality stayed at 89.38. Keep automatic acceptance on the pipeline’s validated field checks. [Review-queue pilot](../../docs/benchmarks/review-20261121.md#decision-scoring) · [Extraction results and source fingerprints](../../docs/benchmarks/decisions-20261121.json)

Laya scores choices on the Apple Neural Engine while other work uses the GPU. On 250 four-way choices, this implementation picked the preferred answer 73 times (29.2%), compared with 76 for NMTZ Core ML and 79 for stock MLX. Use training rewards that have passed a separate task-quality check. Kev scores choices on the GPU and can save work when several questions share a passage. Both models support Python and the `score` command. [Scored results](../../docs/benchmarks/decisions-20261121.json) · [Project and public Core ML results](../../docs/benchmarks/coreml-20261120.json)

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"

.venv/bin/decision-models setup-weights --build-kev-mlx
.venv/bin/decision-models verify --backend kev-0.8b
```

```python
from fast_decision_models import Kev08bMlxBackend, LayaAneBackend

with Kev08bMlxBackend(quant_mode="auto", batch_docs=8) as kev:  # hit-aware prefix_cache_mb="auto" by default (None/0 to opt out)
    res = kev.score_multi_states([(
        "Invoice #8812: Total $1,240.00. Status: PAID via ACH.",
        [{"id": "overdue", "type": "noul", "instructions": "Is the balance overdue and unpaid?",
          "criteria": {"false": "Paid.", "true": "Overdue and unpaid."}}],
    )])
    probs, logits = res[0][0]
```

## Cross-Silicon Benchmark Highlights (All 4 Areas)

### 1. `Laya-Typed-421M` Pure-ANE (`Cat 2`: `RAGTruth + CLINC150 + RewardBench-2`, `N=1,400`)

Both Laya and Kev score the same 600 RAGTruth examples, 550 CLINC150 utterances and 250 RewardBench-2 comparisons. The score averages all 1,400 per-example results. Effective outcome throughput (EOT) divides their total by the elapsed scoring and evaluation time after warmup.

| Host | Implementation | Correct / 1,400 | Time after warmup | Score units/s | Measured memory |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 | Stock MLX ModernBERT 8-bit | 702 | 73.9 s | 9.496 | 1.88 GB |
| M5 | NMTZ Core ML | 699 | 88.4 s | 7.909 | 2.73 GB |
| M5 | Project Laya, Neural Engine | 707 | 114.8 s | 6.158 | 1.00 GB |
| M5 | `ahmedandaloes` in Apple CoreAI | 699 | 940.7 s | 0.743 | 0.33 GB plus 1.68 GB across mapped files |
| M4 Pro | NMTZ Core ML | 699 | 92.0 s | 7.598 | 2.94 GB |
| M4 Pro | Project Laya, Neural Engine | 707 | 121.3 s | 5.827 | 0.98 GB |
| M4 Pro | Stock MLX ModernBERT 8-bit | 703 | 130.7 s | 5.380 | 2.00 GB |

The repaired stock adapter loads all 170 pretrained encoder tensors and uses a 512 MiB MLX buffer cache. On both Macs, five inputs of 50–512 tokens retained the PyTorch reference’s full option order, including a reversed list of choices. The old stock scores of 0.2757 on M5 and 0.2986 on M4 Pro came from an adapter that skipped the pretrained encoder. The results file preserves those runs and their status. [Selected and superseded measurements](../../docs/benchmarks/decisions-20261121.json) · [Reference fixtures](../../tests/fixtures/laya_stock_reference.json)

The stock MLX memory figures in the mixed-suite tables are post-scoring lower bounds. Their file-mapping scan covered the package artifacts and omitted any clean pages in the Hugging Face cache. The separately measured peak process footprints on M5 and M4 Pro were 2.58 and 2.63 GB for Laya, and 3.01 and 3.39 GB for Kev. [Memory measurements](../../docs/benchmarks/decisions-20261121.json)

On both Macs, `NMTZ/laya-typed-decisions-421m-coreml-ane` finished the full scoring workload faster than Laya in `fast-decision-models`. Its score was 0.0057 lower and its measured memory was higher. The public adapter reads token embeddings from the upstream checkpoint, runs the fixed-size Core ML graph, and uses its option logits to calculate the probabilities scored here. The publisher’s separate FP32 action head can choose an action or abstain. This test measures only option scoring.

The public short and long probes used 47 and 512 tokens including the question. The project’s single-call latency and 64-question throughput come from separate runs. On both Macs, each public Laya graph (192, 384 and 512 tokens) reported 10,594 Neural Engine preferences and 5,941 unclassified operations. The first M5 public speed test spent 1.53 seconds compiling the graphs and 88.96 seconds loading them. The later scoring run reused them and loaded in 0.84 seconds.

- **Opt-in `CoreAI-Opt` & `macOS 27.2.1` `.aimodel` options** ([`coreai_export.py`](src/fast_decision_models/coreai_export.py)):
  - **`CoreAI-Opt` calibrated `w8` (`ExportBackend.CoreML`, M4 Pro `26.6`):** cuts `RAGTruth` `ECE-15` by **`-14.3%`** (`0.0970` vs. `0.1131`), improves Brier (`0.2174` vs. `0.2214`) and NLL (`0.6245` vs. `0.6341`), at `5.756` vs. `5.826 EOT` (`0.988x`). (`w4_per_block` on `ExportBackend.CoreML` is blocked by default due to macOS `26.6` CoreML logit inversion.)
  - **`macOS 27.2.1` `FP16` `.aimodel` ANE (`w8_f16` + `prefer_neural_engine`, Base M5):** casting weights/inputs to `FP16` unlocks **`1,652` ANE ops** (`logit_cos = 0.999831`), cutting isolated `S=256` warm latency **`4.09x`** from `94.18 ms` (FP32 `MPSGraph`) to **`23.00 ms`** (`15.4x` faster than stock `CoreAI`'s `354.1 ms`) in **`0.44 GB`** combined (`62.3 MB` `phys_footprint`).

### 2. `Kev-0.8B` (`W8 g64`) Batched `B=8` & `256`-Token `GDN+KV` `PrefixTree` (`Cat 2`, `N=1,400`)

The Kev implementation in `fast-decision-models` enables automatic prefix caching. On the same 1,400 decisions, the project’s Kev scorer earned 0.5250 on M5 and finished in 167 seconds. The public FP16 export scored 0.5214 and finished in 322 seconds. Its row graph accepts 1,024 tokens for the state and question combined. The project package allows up to 1,024 state tokens and 2,048 combined tokens. The public graph’s shorter window clipped 164 of 600 RAGTruth examples and 63 of 1,000 RewardBench-2 candidates. The stock MLX comparison uses one scorer and the original Kev tokenizer for the same input IDs and option positions.

| Host | Engine | Mean score | EOT (score/s) | Time after warmup | Harness memory |
|---|---|---:|---:|---:|---:|
| Base M5 | `fast-decision-models` W8, automatic cache | 0.5250 | 4.395 | 167.2 s | 1.78 GB |
| Base M5 | FluidInference public Core ML FP16 | 0.5214 | 2.269 | 321.7 s | 1.29 GB |
| Base M5 | Stock `mlx-lm` 8-bit | 0.5236 | 5.039 | 145.5 s | 2.28 GB |
| M4 Pro | `fast-decision-models` W8, automatic cache | 0.5250 | 3.415 | 215.2 s | 2.29 GB |
| M4 Pro | FluidInference public Core ML FP16 | 0.5207 | 1.674 | 435.4 s | 9.04 GB |
| M4 Pro | Stock `mlx-lm` 8-bit | 0.5229 | 3.249 | 225.3 s | 2.10 GB |

The public export, `FluidInference/kev-0.8b-coreml`, scored one warm question at 20.72 questions/s on M5 and 16.48 on M4 Pro. Asking eight questions about the same 14-token state raised these rates to 187.30 and 130.69 questions/s. All eight branches fit in one call to the 32-token fused graph. The shared-passage comparison below uses eight actual candidate responses about two source passages.

The public Kev graphs allowed the CPU and GPU. Both Macs reported the same device preferences in `MLComputePlan`.

| Kev graph | GPU-preferred operations | Unclassified operations |
|---|---:|---:|
| Fused, six functions | 5,171 | 68,358 |
| 512-token row | 5,893 | 10,038 |
| 1,024-token row | 7,909 | 14,646 |

The fused graph’s GPU preferences all came from its 128-token function. All 10,616 operations in the 32-token function used for the speed test were unclassified.

The harness memory estimate includes the model process. On M5, macOS reported a 1.00 GB physical footprint for that process and 11.55 GB of resident pages. The corresponding M4 Pro physical footprint was 8.84 GB.

The matched cache probe scores eight RAGTruth responses against two source passages. The state is 432–1,015 tokens and the question adds 44 tokens. Each run starts a fresh scorer, warms the model, measures the first call on these inputs, then repeats it. The plain arm explicitly disables the prefix cache. The saved results contain all option probabilities and raw logits, measured token lengths, timings and memory for each arm. [Matched cache results](../../docs/benchmarks/decisions-20261121.json)

| Host | Project, cache disabled | Project, automatic cache | Stock MLX |
| --- | ---: | ---: | ---: |
| M5 | 1,113 ms | 488 ms | 862 ms |
| M4 Pro | 1,650 ms | 453 ms | 1,287 ms |

These times cover the repeat call for all eight questions. Both project modes and stock chose the same answer on each item. The cached scorer reproduced its own first-pass probabilities and logits exactly. Against the explicitly uncached project scores, the largest probability changes were 0.00965 on M5 and 0.00745 on M4 Pro. [First-call timings and process memory](../../docs/BENCHMARKS.md#kev-prefix-cache)

Kev enables `prefix_cache_mb="auto"` by default, with a 128 MiB budget on a 24 GB Mac and 512 MiB on a 48 GB Mac. Pass `prefix_cache_mb=0` to disable reuse. An earlier 32-question check found identical warm and cold scores with the same 256-token chunking. Against the single-pass computation, maximum probability differences were 0.023 on M5 and 0.020 on M4 Pro. The previous random-token stock timings measured the backbone alone and cannot establish decision throughput. [Cache implementation](src/fast_decision_models/prefix_tree.py) · [Earlier experiments](../../docs/benchmarks/review-20261121.md#decision-scoring)

### 3. ModernBERT-style training throughput (`benchmark_capabilities.py --suites encoder-finetune`, `freeze<18`)

This throughput test uses a randomly initialized encoder and 128 generated examples with random labels, and starts the timer after preparing the cache. It measures training steps and gradient agreement. Training quality on the released Laya weights requires labeled examples and evaluation. [Training benchmark](../../scripts/benchmark_capabilities.py)

| Host | Stock Unpacked MLX | `Packed-512` | `Cached L0..17` | Peak Mem | Parity & Cross-Silicon Scaling |
|---|---:|---:|---:|---:|---|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | `29.63 items/s` (`5.15 GB`) | **`42.26 items/s` (`1.43x`)** | **`58.19 items/s` (`1.96x`)** | **`4.89 GB`** | **`10c` Base M5 `NAX` trains `1.42x–1.59x` faster than `20c` M4 Pro** (`grad_cos = 0.99994`). |
| **M4 Pro (`48 GB`, `Apple9`)** | `20.25 items/s` (`5.22 GB`) | **`26.63 items/s` (`1.32x`)** | **`40.84 items/s` (`2.02x`)** | **`5.06 GB`** | Matches stock gradient direction (`grad_cos = 0.99994`) with `-0.16 GB` RAM. |

### 4. Disaggregated `GPU Actor + ANE Critic` (`GRPO` RLVR, `G=4`, `benchmark_capabilities.py --suites grpo-ane-critic`)

| Host | `fast-gemma4` GPU + `Laya` ANE Critic | Stock `mlx-vlm` + GPU `Kev-8bit` | Speedup | Peak Mem (`0 GB` GPU Critic VRAM) | Exposed ANE Latency |
|---|---|---|---|---:|---|
| **Base M5 (`24 GB`)** | **`2.61 rollouts/s`** (`T=0.7` GRPO) / **`3.38 rollouts/s`** (`T=0.0` BoN) | `0.50 rollouts/s` | **`5.22x` / `6.77x`** | **`8.68 GB`** | **`0.03–0.35 ms`** (`99.96%` hidden during GPU VJP) |
| **M4 Pro (`48 GB`)** | **`1.81 rollouts/s`** (`T=0.7` GRPO) / **`3.15 rollouts/s`** (`T=0.0` BoN) | `0.203 rollouts/s` | **`8.93x` / `15.51x`** (7.44× / 11.30× estimated full-iteration speedups from the measured stages) | **`7.11 GB`** (`-63%` vs. `19.38 GB`) | **`0.015 ms`** (`99.999%` hidden) |

These project rates measure rollout generation or concurrent scoring on repeated fixtures. The historical stock critic also loaded an extra Kev model to format inputs. The historical speedups and memory comparisons include that configuration. The stock critic now uses the same single-scorer adapter as the 1,400-item comparison. The reward quality required for training needs its own labeled task. [Rollout benchmark](../../scripts/benchmark_capabilities.py) · [Review of training evidence](../../docs/benchmarks/review-20261121.md#training-and-shared-tools)

## At a glance

| Question | Answer |
|---|---|
| `Laya-Typed-421M` ANE compute plan? | `[1585, 0, 0]` (`ANE / GPU / CPU` ops) on both `S=256` and `S=512` buckets (`0` GPU contention). |
| `Kev-0.8B` (`w8`) batch & `PrefixTree` parity? | `B=1` vs. `B=3` `max_diff = 0.005613`; `PrefixTree` warm vs. cold `max |Δlogit| = 0.0`, `max |Δprob| = 0.0`. |
| Measured `B=8` / Pure-ANE throughput? | Base M5: **`86.84 q/s`** (`kev_08b B=8`), **`44.38 q/s`** (`laya_ane`). M4 Pro: **`54.52 q/s`** (`kev_08b B=8`), **`41.91 q/s`** (`laya_ane`). |
| PyTorch needed at runtime? | No. Both `.mlpackage` conversion and `head.pt` extraction use stdlib `zipfile` + `numpy`. |

## What keeps it fast

| Model / Engine | Optimization | Source |
|---|---|---|
| **Laya-Typed-421M ANE** | Piecewise Scale-Invariant Residual Re-Centering (`1 -> 1/32 -> 1/256 -> 1/1024`) w/ compensated `>= 2^-14` FP16 LayerNorm, scalar FP16 LUT + NEON `8x8` transpose into `IOSurface`-backed `CVPixelBuffer`s, `P=4` async CoreML ring, `spawn`-parallel bucket builds | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm), [`scripts/build_decision_models.py`](scripts/build_decision_models.py) |
| **Laya CoreAI-Opt / `.aimodel` Export** | Calibrated `CoreAI-Opt` `w8` (`-14.3%` `ECE-15`) and `macOS 27.2.1` `FP16` `.aimodel` ANE (`1,652` ANE ops, `23.00 ms`) with `head_ffn_dim=4096` and `2^-14` guards | [`coreai_export.py`](src/fast_decision_models/coreai_export.py) |
| **Kev-0.8B Metal MLX + `PrefixTree`** | Lossless `W8 (g64)` + `BF16` merged weights, `@mx.compile` forward, left-padded `B=8` batching, `_GDN_PACKED_BTREE_SRC`, default hit- & share-aware `256`-token zero-drift `GDN+KV` `PrefixTree` (`prefix_cache_mb="auto"`), stdlib ZIP `head.pt` loader | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py), [`prefix_tree.py`](src/fast_decision_models/prefix_tree.py) |

## CLI, Setup & Verification

```bash
# Score typed questions (`noul`, `choice`, `score`)
.venv/bin/decision-models score --backend kev-0.8b --type choice \
  --question "Which subsystem caused the alert?" --criteria "network,storage,authentication" \
  --state "Primary DB volume reached 99% utilization and paused writes." --json

# Stage W8 Kev weights or rebuild Laya CoreML ANE bundles (`[build]` extra adds coremltools; parallelized via fast_shared.parallel_coreml)
.venv/bin/decision-models setup-weights --quantize-existing-kev-w8
.venv/bin/decision-models setup-weights --build-laya-ane

# Static guard + tests
fast-ir-guard --strict-bool-mask --coreml src tests                           # 0 errors, 0 warnings
FAST_DECISION_MODELS_LAYA_DIR=/path/to/laya_typed .venv/bin/pytest -v tests
```

Environment overrides: `FAST_DECISION_MODELS_HOME`, `FAST_DECISION_MODELS_ARTIFACTS_DIR`, `FAST_DECISION_MODELS_LAYA_DIR`, `FAST_DECISION_MODELS_KEV_DIR`.

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Apple Base M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)**. Running `test_decision_models_cpu.py` and `test_prefix_tree_cpu.py` offline needs zero external weights; hardware tests require staged `laya_typed` and/or `kev_08b`.
- **Not tested:** Concurrent unserialized Metal + ANE workloads across sibling processes, or non-Apple-Silicon GPUs.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
