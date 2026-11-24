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

### 1. `Laya-Typed-421M` Pure-ANE (`RAGTruth + CLINC150 + RewardBench-2`, 1,400 items)

Both Laya and Kev score the same 600 RAGTruth examples, 550 CLINC150 utterances, and 250 RewardBench-2 comparisons. Effective outcome throughput (EOT) divides total score by elapsed seconds after warmup.

| Host | Implementation | Correct / 1,400 | Time after warmup | Score units/s | Measured memory |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 | Stock MLX ModernBERT 8-bit | 702 | 73.9 s | 9.496 | 1.88 GB |
| M5 | NMTZ Core ML | 699 | 88.4 s | 7.909 | 2.73 GB |
| **M5** | **Project Laya, Neural Engine** | **707** | **114.8 s** | **6.158** | **1.00 GB** |
| M5 | `ahmedandaloes` in Apple CoreAI | 699 | 940.7 s | 0.743 | 0.33 GB + 1.68 GB mapped |
| M4 Pro | NMTZ Core ML | 699 | 92.0 s | 7.598 | 2.94 GB |
| **M4 Pro** | **Project Laya, Neural Engine** | **707** | **121.3 s** | **5.827** | **0.98 GB** |
| M4 Pro | Stock MLX ModernBERT 8-bit | 703 | 130.7 s | 5.380 | 2.00 GB |

On both Macs, NMTZ’s Core ML export finished the standalone 1,400-item sweep faster than `LayaAneBackend`, while `LayaAneBackend` scored 0.0057 higher, used ~1.0 GB instead of 2.73–2.94 GB, and kept **100% of its 1,585 operations on the Neural Engine** (`0` GPU ops) so it can run concurrently beside GPU generation or training. Optional `CoreAI-Opt` 8-bit export reduces `RAGTruth` calibration error (`ECE-15`) by 14.3%, and casting macOS 27.2.1 `.aimodel` exports to FP16 unlocks 1,652 ANE operations (`23.0 ms` at `S=256`, 15.4× faster than stock CoreAI). [CoreAI audit](../../docs/BENCHMARKS.md#7-coreai-audit-of-22-decisions)

### 2. `Kev-0.8B` / `Kev-4B` GPU scoring & shared-passage prefix cache (1,400 items)

| Host | Engine | Mean score | EOT (score/s) | Time after warmup | Measured memory |
|---|---|---:|---:|---:|---:|
| Base M5 | `fast-decision-models` W8, automatic cache | **0.5250** | 4.395 | 167.2 s | 1.78 GB |
| Base M5 | FluidInference public Core ML FP16 | 0.5214 | 2.269 | 321.7 s | 1.29 GB |
| Base M5 | Stock `mlx-lm` 8-bit | 0.5236 | **5.039** | 145.5 s | 2.28 GB |
| M4 Pro | `fast-decision-models` W8, automatic cache | **0.5250** | **3.415** | 215.2 s | 2.29 GB |
| M4 Pro | FluidInference public Core ML FP16 | 0.5207 | 1.674 | 435.4 s | 9.04 GB |
| M4 Pro | Stock `mlx-lm` 8-bit | 0.5229 | 3.249 | 225.3 s | 2.10 GB |

The public Core ML graph clips combined state + question inputs at 1,024 tokens (truncating 164 of 600 RAGTruth examples), whereas `fast-decision-models` accepts up to 1,024 state tokens and 2,048 combined tokens. When eight questions share two 432–1,015-token passages, automatic 256-token `PrefixTree` caching (`prefix_cache_mb="auto"`) defaults to bounded 2-hit `admit_prefix(..., admission_min_hits=2)` under `"auto"`/`"backfill"` profiles — eliminating cold-burst regressions (**1.0039× cold**, `0 B` held, breaker closed) while accelerating warm shared passages **2.394×** (`0` answer flips; **1,113 -> 488 ms** on M5 and **1,650 -> 453 ms** on M4 Pro) — and switches automatically to `admission_min_hits=1` under `profile="interactive"` (with optional `prefix_admission_min_hits` override), passing the Base M5 `prefix-tree` capability suite at **`0.0` cold-vs-warm logit diff** (**`3.04×` vs stock `mlx-lm`**, **`1.54×` vs plain**, `2.91 GB` peak). On `Kev-4B`, cached recurrent `GatedDeltaNet` snapshots are compressed 3.76× in block-32 INT8 with zero logit drift between cold and warm hits. [Prefix-cache probe](../../docs/BENCHMARKS.md#kev-prefix-cache) · [2-hit admission validation](../../docs/BENCHMARKS.md#142-apple10-nax-meta-only-inline-xor-kev-2-hit-admission-and-ovisocr2-tuning) · [Dual-Mac rerun receipt](../../docs/BENCHMARKS.md#153-targeted-post-fix-capability-reruns--root-cause-engineering-notes)

### 3. Encoder fine-tuning & disaggregated GPU + ANE RLVR (`GRPO`)

- **ModernBERT-style fine-tuning (`freeze < 18`, 128 synthetic examples):** Packing to 512 tokens and caching frozen layers reaches **58.2 items/s** on Base M5 (1.96× stock unpacked MLX at 29.6 items/s) and **40.8 items/s** on M4 Pro (2.02× stock at 20.3 items/s) with matching gradient direction (`grad_cos = 0.99994`).
- **Disaggregated `GRPO` rollouts (`fast-gemma4` GPU actor + `Laya` Pure-ANE critic):** Reaches **2.61–3.38 rollouts/s** on M5 (8.68 GB peak) and **1.81–3.15 rollouts/s** on M4 Pro (7.11 GB peak vs 19.38 GB with a GPU critic), hiding over 99.96% of critic scoring latency behind GPU backward passes. [Training benchmarks](../../docs/BENCHMARKS.md#5-local-training--post-training-qlora-modernbert-grpo)

## What keeps it fast

| Component | How it works | Source |
|---|---|---|
| **Laya-Typed-421M ANE** | Piecewise residual re-centering (`1 -> 1/1024`) to stay within FP16 range, `IOSurface`-backed `CVPixelBuffer` I/O, 4-slot async Core ML ring (`1,585` ANE ops, `0` GPU/CPU), and parallel bucket compilation | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm) |
| **Kev-0.8B & Kev-4B MLX** | Lossless 8-bit (`g64`) and mixed-bit `SpecBucketedLinearGroup` projection fusion, `@mx.compile` forward, left-padded batching, and hit-aware 256-token `PrefixTree` reuse (`prefix_cache_mb="auto"`) without PyTorch at runtime | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py), [`prefix_tree.py`](src/fast_decision_models/prefix_tree.py) |

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
