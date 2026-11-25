# fast-decision-models

Local zero-shot classifiers and typed decision scorers for Apple Silicon (`Laya-Typed-421M`, `Kev-0.8B`, and `Kev-4B`). Score binary checks (`noul`), multi-way choices (`choice`), and numeric scales (`score`) in a single forward pass with calibrated probabilities, running `Laya` 100% on the 16-core Apple Neural Engine (`5.1 ms` single query) or in concurrent GPU+ANE `hetero` mode (`37.9 s` for 1,400 items, `2.0x` stock MLX, `25x` faster than Apple CoreAI).

Zero telemetry, no account, and runs 100% offline after initial weight download (~0.95 GB for Laya ANE + Kev-0.8B; ~4.90 GB for full suite including Kev-4B). Prebuilt Hugging Face artifacts: [`mikewolfd/laya-typed-decisions-421m-coreml-ane`](https://huggingface.co/mikewolfd/laya-typed-decisions-421m-coreml-ane) · [`mikewolfd/kev-0.8b-mlx-w8`](https://huggingface.co/mikewolfd/kev-0.8b-mlx-w8) · [Scored results](../../docs/benchmarks/decisions-20261121.json) · [Public Core ML comparisons](../../docs/benchmarks/coreml-20261120.json).

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 14+, Python 3.12+, and `uv`. Windows, Linux, and Intel Macs are unsupported for Core ML ANE and Metal GPU execution. Xcode Command Line Tools are required to compile the native `Laya` ANE `.dylib` runner (`make -C packages/fast-decision-models native`; GPU-only `Kev` needs no native build).

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-decision-models --inexact
make -C packages/fast-decision-models native

.venv/bin/decision-models setup-weights --build-kev-mlx   # Stages ~0.95 GB Laya ANE + Kev-0.8B W8
.venv/bin/decision-models verify --backend kev-0.8b
```

```python
from fast_decision_models import Kev08bMlxBackend

with Kev08bMlxBackend(quant_mode="auto", batch_docs=8) as kev:  # 256-tok prefix cache on by default
    res = kev.score_multi_states([(
        "Invoice #8812: Total $1,240.00. Status: PAID via ACH.",
        [{"id": "overdue", "type": "noul", "instructions": "Is the balance overdue and unpaid?",
          "criteria": {"false": "Paid.", "true": "Overdue and unpaid."}}],
    )])
    probs, logits = res[0][0]
```

## Benchmark highlights

### 1. `Laya-Typed-421M` Pure-ANE & GPU+ANE (`RAGTruth + CLINC150 + RewardBench-2`, 1,400 items)

All engines score the same 600 RAGTruth hallucination checks, 550 CLINC150 intents, and 250 RewardBench-2 preference comparisons.

| Host | Implementation | Correct / 1,400 | Accuracy | Wall time | Throughput (`q/s`) | Score units/s | Memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | **`fast-decision-models` `laya-hetero` (ANE + GPU)** | **704** | **0.5029** | **37.9 s** | **36.96** | **18.586** | **3.72 GB** |
| Base M5 | Stock MLX ModernBERT 8-bit | 702 | 0.5014 | 73.9 s | 18.94 | 9.496 | 1.88 GB |
| **Base M5 (24 GB)** | **`fast-decision-models` `laya-ane` (Pure-ANE)** | **711** | **0.5079** | **76.7 s** | **18.25** | **9.266** | **1.59 GB** |
| Base M5 | NMTZ Core ML | 699 | 0.4993 | 88.4 s | 15.84 | 7.909 | 2.73 GB |
| Base M5 | `ahmedandaloes` in Apple CoreAI | 699 | 0.4993 | 940.7 s | 1.49 | 0.743 | 0.33 GB (+1.68 GB mmap) |
| **M4 Pro (48 GB)** | **`fast-decision-models` `laya-hetero` (ANE + GPU)** | **707** | **0.5050** | **59.3 s** | **23.63** | **11.932** | **3.30 GB** |
| M4 Pro | NMTZ Core ML | 699 | 0.4993 | 92.0 s | 15.22 | 7.598 | 2.94 GB |
| M4 Pro | Stock MLX ModernBERT 8-bit | 703 | 0.5021 | 130.7 s | 10.71 | 5.380 | 2.00 GB |
| **M4 Pro (48 GB)** | **`fast-decision-models` `laya-ane` (Pure-ANE)** | **711** | **0.5079** | **133.1 s** | **10.52** | **5.341** | **1.52 GB** |

All six Core ML buckets (`64..512` tokens) in `laya_typed_w8.mlmodelc` place 100% of their 10,538 operations on the Neural Engine (`0` GPU/CPU fallback ops). Two-pass FP16 `LayerNorm` inverse-scaling prevents exponent overflow in late layers, raising ANE accuracy to **711/1400 (0.5079)** on both Macs while cutting memory to `1.59 GB` (vs. `2.73 GB` for NMTZ Core ML). Dual-chip `laya-hetero` splits batches between Core ML ANE and compiled MLX GPU (`37.9 s` on M5). [LayerNorm & CoreAI details](../../docs/benchmarks/rca-and-laya-fixes-20261124.json)

| Laya batch shape (`ms` / `q/s`) | M4 Pro ANE | M4 Pro GPU | M4 Pro Hetero | Base M5 ANE | Base M5 GPU | Base M5 Hetero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1x_s64` | **5.11 (195.7)** | 8.87 (112.7) | 5.33 (187.7) | **6.53 (153.1)** | 10.30 (97.1) | 6.54 (152.9) |
| `1x_short` | **5.09 (196.6)** | 12.80 (78.2) | 5.39 (185.5) | **6.53 (153.1)** | 11.97 (83.5) | 6.53 (153.2) |
| `8x_512` | 377.09 (21.2) | 679.27 (11.8) | **284.21 (28.2)** | 342.53 (23.4) | 263.37 (30.4) | **172.71 (46.3)** |
| `64x_short` | **325.34 (196.7)** | 487.51 (131.3) | 331.27 (193.2) | 411.94 (155.4) | 210.78 (303.6) | **200.98 (318.4)** |
| `64x_mixed` | **1028.97 (62.2)** | 1939.23 (33.0) | 1244.54 (51.4) | 1008.38 (63.5) | 783.95 (81.6) | **578.90 (110.6)** |

### 2. `Kev-0.8B` / `Kev-4B` GPU scoring & shared-passage prefix cache (1,400 items)

| Host | Engine | Mean score | `q/s` | Score units/s | Wall time | Memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Base M5 (24 GB)** | **`fast-decision-models` W8 (`LCP>=128`, sorted tails)** | **0.5264** | **11.40** | **6.000** | **122.8 s** | **2.20 GB** |
| Base M5 | Stock `mlx-lm` 8-bit GPU | 0.5236 | 9.62 | 5.039 | 145.5 s | 2.28 GB |
| Base M5 | FluidInference public Core ML FP16 | 0.5214 | 4.35 | 2.269 | 321.7 s | 1.29 GB |
| **M4 Pro (48 GB)** | **`fast-decision-models` W8** | **0.5250** | **6.51** | **3.415** | **215.2 s** | **2.29 GB** |
| M4 Pro | Stock `mlx-lm` 8-bit GPU | 0.5229 | 6.21 | 3.249 | 225.3 s | 2.10 GB |
| M4 Pro | FluidInference public Core ML FP16 | 0.5207 | 3.22 | 1.674 | 435.4 s | 9.04 GB |

FluidInference Core ML clips combined inputs at 1,024 tokens (truncating 164 of 600 RAGTruth items), whereas `fast-decision-models` supports up to 1,024 state tokens and 2,048 combined tokens. Raising the prefix-grouping threshold to 128 tokens and deterministically sorting candidate tails eliminates padding waste on 1-question/state batches (`1.18x` vs. stock `mlx-lm` on M5). On **shared passages** (multiple criteria scored over the same document), `PrefixTree` (`prefix_cache_mb="auto"` with 2-hit admission) speeds warm scoring **2.39x** (`1,113 -> 488 ms` M5; `1,650 -> 453 ms` M4 Pro) with zero cold-burst thrash and zero answer flips; with `profile="interactive"` (1-hit admission), warm `B=8` hits **27.65 q/s (`2.97x` stock `mlx-lm`)** at exact `0.0` logit diff. [Prefix-cache benchmarks](../../docs/BENCHMARKS.md#62-kev-prefix-reuse-on-repeated-passages)

### 3. Encoder fine-tuning & concurrent GPU + ANE reinforcement learning (`GRPO`)

- **ModernBERT fine-tuning (`freeze < 18`):** Packing to 512 tokens and caching frozen layers hits **58.2 items/s** on Base M5 (1.96x stock MLX) and **40.8 items/s** on M4 Pro (2.02x stock) at `grad_cos = 0.99994`.
- **Disaggregated `GRPO` rollouts (`fast-gemma4` GPU actor + `Laya` Pure-ANE critic):** Hits **2.61-3.38 rollouts/s** on M5 (8.68 GB peak) and **1.81-3.15 rollouts/s** on M4 Pro (7.11 GB vs. 19.38 GB GPU critic), hiding >99.96% of critic scoring behind GPU backward passes. [Training benchmarks](../../docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)

## What keeps it fast

| Component | How it works | Source |
| --- | --- | --- |
| **Laya-Typed-421M ANE & Hetero** | 6-bucket multifunction Core ML (`s64..s512`, `10,538` ANE ops, `0` GPU/CPU), FP16 LayerNorm scaling, split-head `einsum`, `lazy_load`, and `laya-hetero` makespan split | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`laya_hetero.py`](src/fast_decision_models/laya_hetero.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm) |
| **Kev-0.8B & Kev-4B MLX** | Lossless 8-bit (`g64`) with NAX `bf16` guard, mixed-bit linear projection fusion, `@mx.compile` forward, and hit-aware 256-tok `PrefixTree` (block-32 INT8 GDN state on `4B`) | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py), [`prefix_tree.py`](src/fast_decision_models/prefix_tree.py) |

## CLI & verification

```bash
.venv/bin/decision-models score --backend kev-0.8b --type choice \
  --question "Which subsystem caused the alert?" --criteria "network,storage,authentication" \
  --state "Primary DB volume reached 99% utilization and paused writes." --json
.venv/bin/decision-models setup-weights --quantize-existing-kev-w8
.venv/bin/decision-models setup-weights --build-laya-ane
.venv/bin/pytest -v packages/fast-decision-models/tests
```

## Limitations, Prior Art & License

- **Limitations:** Decision scorers help prioritize human review queues or score domain-validated criteria, not replace schema validation. In a 48-doc extraction pilot, using Laya as a review-queue ranker improved defect discovery (`5-7/10` vs. `4/10` rule-selected), whereas using Laya as a blocking inline gate increased wall time (`93.3 -> 160.7 s`) with unchanged held-out accuracy (`89.38`). On 250 four-way `RewardBench-2` choices, Laya picked the preferred answer 73/250 times (`29.2%`, vs. 76 NMTZ Core ML and 79 stock MLX). Verify calibration on your own reward/triage task before automating rejections.
- **Prior art:** Built on upstream `convaiinnovations/laya-typed-decisions` (`421M` ModernBERT), `jaredpalmer/kev-0.8b` / `kev-4b` (`Qwen3.5`), `mlx-lm`, and Core ML (`NMTZ`, `FluidInference`).
- **License:** Package code is [MIT licensed](LICENSE); base weights follow their upstream licenses (ConvAI, Qwen Apache-2.0). Verified on 48 GB M4 Pro and 24 GB Base M5.
