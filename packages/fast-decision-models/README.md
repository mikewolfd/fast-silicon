# fast-decision-models

**Fast local document classification, scoring, fine-tuning, and zero-VRAM ANE RLVR critics on Apple Silicon — `8.29x` higher `EOT` than stock Apple `CoreAI`, up to `6.19x` (`B=8`) / `9.89x` (`B=1`) faster than `mlx-lm` with `PrefixTree`, and `5.22x–15.51x` faster GRPO rollouts.**

`fast-decision-models` runs typed binary (`noul`), multi-choice (`choice`), and scalar (`score`) judges for **Kev-0.8B** on Metal (**`86.84 q/s`** on Base M5 / **`54.52 q/s`** on M4 Pro at `B=8`, plus zero-drift `256`-token `GDN+KV` `PrefixTree` caching) and **Laya-Typed-421M** purely on the 16-core Apple Neural Engine (**`44.38 q/s`** on Base M5 / **`41.91 q/s`** on M4 Pro with **`0` GPU ops**, keeping your GPU completely free for LLM generation, fine-tuning, or RLVR). Importing the package is pure Python stdlib with zero runtime PyTorch dependency, backed by CLI (`score`), CoreAI-Opt calibration/export hooks, and Python APIs.

> **Full Cross-Silicon Benchmarks & Promotion Gates:** See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete Category 2 (`RAGTruth + CLINC150 + RewardBench-2`), fine-tuning, GRPO, `PrefixTree`, and `CoreAI-Opt` Track A / Track B tables.

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

with Kev08bMlxBackend(quant_mode="auto", batch_docs=8) as kev:
    res = kev.score_multi_states([(
        "Invoice #8812: Total $1,240.00. Status: PAID via ACH.",
        [{"id": "overdue", "type": "noul", "instructions": "Is the balance overdue and unpaid?",
          "criteria": {"false": "Paid.", "true": "Overdue and unpaid."}}],
    )])
    probs, logits = res[0][0]
```

## Cross-Silicon Benchmark Highlights (All 4 Areas)

### 1. `Laya-Typed-421M` Pure-ANE (`Cat 2`: `RAGTruth + CLINC150 + RewardBench-2`, `N=1,400`)

| Host & Engine | `S_obj,e2e` | `EOT (u/s)` | Warm `B=1` / Throughput | 4-Bucket Mem | Comparison |
|---|---:|---:|---|---:|---|
| **Base M5 (`24 GB`) · `laya_ane` (`W8A16`, `[1585,0,0]`)** | **`0.5050`** | **`6.158`** (`114.8 s`) | **`22.64 ms`** (`44.38 q/s`) | **`1.00 GB`** | **`8.29x` vs. Stock `CoreAI`** (`15.6x` lower `B=1` latency) · **`1.06x`** M4 Pro |
| Base M5 (`24 GB`) · Stock Apple `CoreAI` (`decision.aimodel`) | `0.4993` | `0.743` (`940.7 s`) | `354.1 ms` (`MPSGraph` FP32) | `0.33 GB` *(1.69 GB mapped)* | Accurate stock baseline (`0` ANE ops) |
| Base M5 (`24 GB`) · Stock `MLX` (`ModernBERT 8-bit`) | `0.2757`\* | `5.319` (`72.6 s`) | — | `18.70 GB` | `1.16x` `EOT`, avoids **`-0.23` MLX quality collapse** & **`18.7x`** RAM |
| **M4 Pro (`48 GB`) · `laya_ane` (`W8A16`, `[1585,0,0]`)** | **`0.5050`** | **`5.827`** (`121.3 s`) | **`24.28 ms`** (`41.91 q/s`) | **`0.98 GB`** | **`1.92x` vs. Stock `MLX`** (`0.2986` `S_obj`, `34.0 GB`) · **`4.46x`** MPS `B=8` |

- **Opt-in `CoreAI-Opt` & `macOS 27.2.1` `.aimodel` options** ([`coreai_export.py`](src/fast_decision_models/coreai_export.py)):
  - **`CoreAI-Opt` calibrated `w8` (`ExportBackend.CoreML`, M4 Pro `26.6`):** cuts `RAGTruth` `ECE-15` by **`-14.3%`** (`0.0970` vs. `0.1131`), improves Brier (`0.2174` vs. `0.2214`) and NLL (`0.6245` vs. `0.6341`), at `5.756` vs. `5.826 EOT` (`0.988x`). (`w4_per_block` on `ExportBackend.CoreML` is blocked by default due to macOS `26.6` CoreML logit inversion.)
  - **`macOS 27.2.1` `FP16` `.aimodel` ANE (`w8_f16` + `prefer_neural_engine`, Base M5):** casting weights/inputs to `FP16` unlocks **`1,652` ANE ops** (`logit_cos = 0.999831`), cutting isolated `S=256` warm latency **`4.09x`** from `94.18 ms` (FP32 `MPSGraph`) to **`23.00 ms`** (`15.4x` faster than stock `CoreAI`'s `354.1 ms`) in **`0.44 GB`** combined (`62.3 MB` `phys_footprint`).

### 2. `Kev-0.8B` (`W8 g64`) Batched `B=8` & `256`-Token `GDN+KV` `PrefixTree` (`Cat 2`, `N=1,400`)

| Host & Mode | `S_obj,e2e` | `EOT (u/s)` | Batched `B=8` / `PrefixTree` | 4-Bucket Mem | Takeaway |
|---|---:|---:|---|---:|---|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | **`0.5257`** | **`5.078`** (`144.9 s`) | **`86.84 q/s`** (`B=8`) · **`45.9 ms`** `PrefixTree` `B=1` | **`1.77 GB`** | **`1.59x` faster than `20c` M4 Pro** at `B=8` (`1.56x` `EOT`) via `NAX`; **`-42%` RAM** vs. `mlx-lm` (`3.06 GB`); `PrefixTree` vs. `mlx-lm`: **`2.59x`** `B=8` shared preamble (`22.6` vs. `8.73 q/s`), **`6.19x`** `8q` same-doc branch (`54.1 q/s`), **`5.15x`** `B=1` `512`-tok hit (`46.3` vs. `8.99 q/s`). |
| **M4 Pro (`48 GB`, `Apple9`)** | **`0.5250`** | **`3.254`** (`225.9 s`) | **`54.52 q/s`** (`B=8`) · **`44.4 ms`** `PrefixTree` `B=1` | **`1.84 GB`** | **`-44%` RAM** vs. `mlx-lm` (`3.26 GB`); `PrefixTree` vs. `mlx-lm`: **`4.91x`** `B=8` shared preamble (`26.0` vs. `5.29 q/s`), **`5.86x`** `8q` same-doc branch (`31.0 q/s`), **`9.89x`** `B=1` `512`-tok hit (`53.8` vs. `5.44 q/s`); **`5.82x`** vs. `transformers-mps B=8`. |

- `PrefixTree` ([`prefix_tree.py`](src/fast_decision_models/prefix_tree.py)) snapshots `256`-token `GDN+KV` state with **exact zero drift (`max |Δlogit| = 0.0`, `max |Δprob| = 0.0`)** — delivering `2.72x` at `550` tokens (`3.88x` @ `512`, `6.34x` @ `1024`) on Base M5 and `4.54x` at `550` tokens (`6.90x` @ `512`, `12.52x` @ `1024`) on M4 Pro.

### 3. Local `ModernBERT-421M` Fine-Tuning (`benchmark_capabilities.py --suites encoder-finetune`, `freeze<18`)

| Host | Stock Unpacked MLX | `Packed-512` | `Cached L0..17` | Peak Mem | Parity & Cross-Silicon Scaling |
|---|---:|---:|---:|---:|---|
| **Base M5 (`24 GB`, `Apple10 NAX`)** | `29.63 items/s` (`5.15 GB`) | **`42.26 items/s` (`1.43x`)** | **`58.19 items/s` (`1.96x`)** | **`4.89 GB`** | **`10c` Base M5 `NAX` trains `1.42x–1.59x` faster than `20c` M4 Pro** (`grad_cos = 0.99994`). |
| **M4 Pro (`48 GB`, `Apple9`)** | `20.25 items/s` (`5.22 GB`) | **`26.63 items/s` (`1.32x`)** | **`40.84 items/s` (`2.02x`)** | **`5.06 GB`** | Matches stock gradient direction (`grad_cos = 0.99994`) with `-0.16 GB` RAM. |

### 4. Disaggregated `GPU Actor + ANE Critic` (`GRPO` RLVR, `G=4`, `benchmark_capabilities.py --suites grpo-ane-critic`)

| Host | `fast-gemma4` GPU + `Laya` ANE Critic | Stock `mlx-vlm` + GPU `Kev-8bit` | Speedup | Peak Mem (`0 GB` GPU Critic VRAM) | Exposed ANE Latency |
|---|---|---|---|---:|---|
| **Base M5 (`24 GB`)** | **`2.61 rollouts/s`** (`T=0.7` GRPO) / **`3.38 rollouts/s`** (`T=0.0` BoN) | `0.50 rollouts/s` | **`5.22x` / `6.77x`** | **`8.68 GB`** | **`0.03–0.35 ms`** (`99.96%` hidden during GPU VJP) |
| **M4 Pro (`48 GB`)** | **`1.81 rollouts/s`** (`T=0.7` GRPO) / **`3.15 rollouts/s`** (`T=0.0` BoN) | `0.203 rollouts/s` | **`8.93x` / `15.51x`** (`7.44x` / `11.30x` full step) | **`7.11 GB`** (`-63%` vs. `19.38 GB`) | **`0.015 ms`** (`99.999%` hidden) |

Base M5 generates constrained `GRPO` rollouts **`1.44x` faster than M4 Pro** while ANE critic overhead is effectively zero on both Macs. See [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) for complete breakdowns.

## At a glance

| Question | Answer |
|---|---|
| `Laya-Typed-421M` ANE compute plan? | `[1585, 0, 0]` (`ANE / GPU / CPU` ops) on both `S=256` and `S=512` buckets (`0` GPU contention). |
| `Kev-0.8B` (`w8`) batch & `PrefixTree` parity? | `B=1` vs. `B=3` `max_diff = 0.005613`; `PrefixTree` warm vs. cold `max |Δlogit| = 0.0`, `max |Δprob| = 0.0`. |
| Measured `B=8` / Pure-ANE throughput? | Base M5: **`86.84 q/s`** (`kev_08b B=8`), **`44.38 q/s`** (`laya_ane`). M4 Pro: **`54.52 q/s`** (`kev_08b B=8`, **`5.82x`** vs. `transformers-mps B=8`), **`41.91 q/s`** (`laya_ane`, **`4.46x`** vs. MPS). |
| PyTorch needed at runtime? | No. Both `.mlpackage` conversion and `head.pt` extraction use stdlib `zipfile` + `numpy`. |

## What keeps it fast

| Model / Engine | Optimization | Source |
|---|---|---|
| **Laya-Typed-421M ANE** | Piecewise Scale-Invariant Residual Re-Centering (`1 -> 1/32 -> 1/256 -> 1/1024`) w/ compensated `>= 2^-14` FP16 LayerNorm, scalar FP16 LUT + NEON `8x8` transpose into `IOSurface`-backed `CVPixelBuffer`s, `P=4` async CoreML ring, `spawn`-parallel bucket builds | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm), [`scripts/build_decision_models.py`](scripts/build_decision_models.py) |
| **Laya CoreAI-Opt / `.aimodel` Export** | Calibrated `CoreAI-Opt` `w8` (`-14.3%` `ECE-15`) and `macOS 27.2.1` `FP16` `.aimodel` ANE (`1,652` ANE ops, `23.00 ms`) with `head_ffn_dim=4096` and `2^-14` guards | [`coreai_export.py`](src/fast_decision_models/coreai_export.py) |
| **Kev-0.8B Metal MLX + `PrefixTree`** | Lossless `W8 (g64)` + `BF16` merged weights, `@mx.compile` forward, left-padded `B=8` batching, `_GDN_PACKED_BTREE_SRC`, `256`-token zero-drift `GDN+KV` `PrefixTree` LRU cache, stdlib ZIP `head.pt` loader | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py), [`prefix_tree.py`](src/fast_decision_models/prefix_tree.py) |

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
