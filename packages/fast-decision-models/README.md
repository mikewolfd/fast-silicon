# fast-decision-models

Use typed Laya to help a person prioritize extracted records for review. In a 48-document pilot with 12 known defects, the first ten selected by the existing rules contained four defects. Laya alone found five, and rules plus Laya found seven. Making Laya a blocking check increased M5 extraction time from 93.31 to 160.71 seconds while held-out quality stayed at 89.38. Keep automatic acceptance on the pipeline’s validated field checks. [Review-queue pilot](../../docs/benchmarks/review-20261121.md#decision-scoring) · [Extraction results](../../docs/benchmarks/decisions-20261121.json)

Laya scores choices on the Apple Neural Engine while other work uses the GPU. On 250 four-way choices, this implementation picked the preferred answer 73 times (29.2%), compared with 76 for NMTZ Core ML and 79 for stock MLX. Use training rewards that have passed a separate task-quality check. Kev scores choices on the GPU and reuses cached work when several questions share a passage. Both models support Python and the `decision-models score` CLI. [Scored results](../../docs/benchmarks/decisions-20261121.json) · [Public Core ML comparisons](../../docs/benchmarks/coreml-20261120.json)

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

with Kev08bMlxBackend(quant_mode="auto", batch_docs=8) as kev:  # prefix_cache_mb="auto" by default (0 to disable)
    res = kev.score_multi_states([(
        "Invoice #8812: Total $1,240.00. Status: PAID via ACH.",
        [{"id": "overdue", "type": "noul", "instructions": "Is the balance overdue and unpaid?",
          "criteria": {"false": "Paid.", "true": "Overdue and unpaid."}}],
    )])
    probs, logits = res[0][0]
```

## Benchmark highlights

### 1. `Laya-Typed-421M` 6-Bucket ANE & `laya-hetero` (`RAGTruth + CLINC150 + RewardBench-2`, 1,400 items)

Both Laya and Kev score the same 600 RAGTruth examples, 550 CLINC150 utterances, and 250 RewardBench-2 comparisons. Effective outcome throughput (EOT) divides total score by elapsed seconds after warmup.

| Host | Implementation | Correct / 1,400 | `S_obj,e2e` | Time after warmup | Score units/s | Measured memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **M5 (quiescent)** | **Project `laya-hetero` (ANE + compiled GPU)** | **704** | **0.5029** | **37.9 s** | **18.586** | **3.72 GB** |
| M5 | Stock MLX ModernBERT 8-bit | 702 | 0.5014 | 73.9 s | 9.496 | 1.88 GB |
| **M5 (quiescent)** | **Project `laya-ane`, 6-bucket pre-warmed** | **711** | **0.5079** | **76.7 s** | **9.266** | **1.59 GB** |
| M5 | NMTZ Core ML | 699 | 0.4993 | 88.4 s | 7.909 | 2.73 GB |
| M5 | `ahmedandaloes` in Apple CoreAI | 699 | 0.4993 | 940.7 s | 0.743 | 0.33 GB + 1.68 GB mapped |
| **M4 Pro (contended)** | **Project `laya-hetero` (ANE + compiled GPU)** | **707** | **0.5050** | **59.3 s** | **11.932** | **3.30 GB** |
| M4 Pro | NMTZ Core ML | 699 | 0.4993 | 92.0 s | 7.598 | 2.94 GB |
| M4 Pro | Stock MLX ModernBERT 8-bit | 703 | 0.5021 | 130.7 s | 5.380 | 2.00 GB |
| **M4 Pro (contended)** | **Project `laya-ane`, 6-bucket pre-warmed** | **711** | **0.5079** | **133.1 s** | **5.341** | **1.52 GB** |

All six multifunction Core ML buckets (`s64, s128, s192, s256, s384, s512`) in `laya_typed_w8.mlmodelc` place **100% of their `10,538` operations on the Neural Engine** (`[10538, 0, 0]`, `0` GPU/CPU ops). Explicit two-pass centered FP16 `LayerNorm` scaling (`tok_emb` `base_inv_scale=4.0`, `l14..l18` `s_eff_floor=1/128`, `l19..l27` `max_ln_gain=4.0`, head/scorer `base_inv_scale=0.0625`, post-quantization pass removal including `common::fuse_layernorm_or_instancenorm`, and split-head `mb.einsum`) lifts ANE accuracy to **`711/1400` (`0.5079`)** on both Macs while beating NMTZ Core ML on quiescent M5 wall time (`76.7 s` vs `88.4 s`) at `1.59 GB` memory (`2.73 GB` NMTZ). `laya-hetero` adds compiled MLX GPU execution with makespan splitting and a zero-thread `n=1` fast path (`37.9 s` / `18.586` EOT on M5; `59.3 s` / `11.932` EOT on contended M4 Pro). Optional `CoreAI-Opt` 8-bit export reduces `RAGTruth` calibration error (`ECE-15`) by 14.3%, and casting macOS 27.2.1 `.aimodel` exports to FP16 unlocks 1,652 ANE operations (`23.0 ms` at `S=256`, 15.4× faster than stock CoreAI). [CoreAI audit](../../docs/BENCHMARKS.md#7-coreai-audit-of-22-decisions) · [Wave 3 Laya & FP16 LN fixes](../../docs/benchmarks/rca-and-laya-fixes-20261124.json)

| Laya Shape (`ms` / `q/s`) | M4 Pro ANE *(contended)* | M4 Pro GPU *(contended)* | M4 Pro Hetero *(contended)* | Base M5 ANE *(quiescent)* | Base M5 GPU *(quiescent)* | Base M5 Hetero *(quiescent)* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1x_s64` | **5.11 (195.7)** | 8.87 (112.7) | 5.33 (187.7) | **6.53 (153.1)** | 10.30 (97.1) | 6.54 (152.9) |
| `1x_short` | **5.09 (196.6)** | 12.80 (78.2) | 5.39 (185.5) | **6.53 (153.1)** | 11.97 (83.5) | 6.53 (153.2) |
| `8x_512` | 377.09 (21.2) | 679.27 (11.8) | **284.21 (28.2)** | 342.53 (23.4) | 263.37 (30.4) | **172.71 (46.3)** |
| `64x_short` | **325.34 (196.7)** | 487.51 (131.3) | 331.27 (193.2) | 411.94 (155.4) | 210.78 (303.6) | **200.98 (318.4)** |
| `64x_mixed` | **1028.97 (62.2)** | 1939.23 (33.0) | 1244.54 (51.4) | 1008.38 (63.5) | 783.95 (81.6) | **578.90 (110.6)** |

### 2. `Kev-0.8B` / `Kev-4B` GPU scoring & shared-passage prefix cache (1,400 items)

| Host | Engine | Mean score | EOT (score/s) | Time after warmup | Measured memory |
|---|---|---:|---:|---:|---:|
| Base M5 (Wave 3) | `fast-decision-models` W8, NAX `bf16` guard + cache | **0.5257** | **4.885** | 150.7 s | 2.23 GB |
| Base M5 | `fast-decision-models` W8, automatic cache (baseline) | 0.5250 | 4.395 | 167.2 s | 1.78 GB |
| Base M5 | FluidInference public Core ML FP16 | 0.5214 | 2.269 | 321.7 s | 1.29 GB |
| Base M5 | Stock `mlx-lm` 8-bit | 0.5236 | **5.039** | 145.5 s | 2.28 GB |
| M4 Pro | `fast-decision-models` W8, automatic cache | **0.5257** | **3.415** | 215.2 s | 2.29 GB |
| M4 Pro | FluidInference public Core ML FP16 | 0.5207 | 1.674 | 435.4 s | 9.04 GB |
| M4 Pro | Stock `mlx-lm` 8-bit | 0.5229 | 3.249 | 225.3 s | 2.10 GB |

The public Core ML graph clips combined state and question inputs at 1,024 tokens (truncating 164 of 600 RAGTruth examples), whereas `fast-decision-models` accepts up to 1,024 state tokens and 2,048 combined tokens. On `Kev-4B`, cached recurrent `GatedDeltaNet` snapshots are compressed 3.76× in block-32 INT8 with zero logit drift between cold and warm hits.

When eight questions share two 432–1,015-token passages, automatic 256-token `PrefixTree` caching (`prefix_cache_mb="auto"`) requires two sightings before caching a snapshot under `"auto"` and `"backfill"` profiles. That avoids cold-burst slowdowns (**1.0039× cold**, `0 B` held, circuit breaker closed) while speeding warm shared passages **2.394×** with zero answer flips (**1,113 to 488 ms** on M5; **1,650 to 453 ms** on M4 Pro). Under `profile="interactive"` (overridable via `prefix_admission_min_hits`), it caches on the first sighting and passes the Base M5 `prefix-tree` suite with **`0.0` cold-vs-warm logit difference** (**`3.04×` vs. stock `mlx-lm`**, **`1.54×` vs. plain**, `2.91 GB` peak). [Prefix-cache probe](../../docs/BENCHMARKS.md#62-kev-prefix-reuse-on-repeated-passages) · [2-hit admission validation](../../docs/BENCHMARKS.md#142-apple10-nax-meta-only-inline-xor-kev-2-hit-admission-and-ovisocr2-tuning) · [Dual-Mac rerun receipt](../../docs/BENCHMARKS.md#153-targeted-post-fix-capability-reruns--root-cause-engineering-notes)

### 3. Encoder fine-tuning & disaggregated GPU + ANE RLVR (`GRPO`)

- **ModernBERT-style fine-tuning (`freeze < 18`, 128 synthetic examples):** Packing to 512 tokens and caching frozen layers reaches **58.2 items/s** on Base M5 (1.96× stock unpacked MLX at 29.6 items/s) and **40.8 items/s** on M4 Pro (2.02× stock at 20.3 items/s) with matching gradient direction (`grad_cos = 0.99994`).
- **Disaggregated `GRPO` rollouts (`fast-gemma4` GPU actor + `Laya` Pure-ANE critic):** Reaches **2.61–3.38 rollouts/s** on M5 (8.68 GB peak) and **1.81–3.15 rollouts/s** on M4 Pro (7.11 GB peak vs 19.38 GB with a GPU critic), hiding over 99.96% of critic scoring latency behind GPU backward passes. [Training benchmarks](../../docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **Laya-Typed-421M ANE & Hetero** | 6-bucket multifunction Core ML (`s64..s512`, `10,538` ANE ops, `0` GPU/CPU) with two-pass FP16 `LayerNorm` scaling, split-head `einsum`, `encode_state`, `lazy_load`, and `laya-hetero` (`compile_encode=True`, makespan split, `n=1` zero-thread) | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`laya_hetero.py`](src/fast_decision_models/laya_hetero.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm) |
| **Kev-0.8B & Kev-4B MLX** | Lossless 8-bit (`g64`) with M5 NAX `bf16` guard, mixed-bit `SpecBucketedLinearGroup` projection fusion, `@mx.compile` forward, left-padded batching, and hit-aware 256-token `PrefixTree` reuse (`prefix_cache_mb="auto"`) without PyTorch | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py), [`prefix_tree.py`](src/fast_decision_models/prefix_tree.py) |

## CLI & verification

```bash
# Score typed questions (`noul`, `choice`, `score`)
.venv/bin/decision-models score --backend kev-0.8b --type choice \
  --question "Which subsystem caused the alert?" --criteria "network,storage,authentication" \
  --state "Primary DB volume reached 99% utilization and paused writes." --json

# Stage W8 Kev weights or rebuild Laya Core ML ANE bundles
.venv/bin/decision-models setup-weights --quantize-existing-kev-w8
.venv/bin/decision-models setup-weights --build-laya-ane

# Run tests
.venv/bin/pytest -v tests
```

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**. Offline CPU tests (`test_decision_models_cpu.py`, `test_prefix_tree_cpu.py`) run without external weights.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
