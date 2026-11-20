# fast-decision-models

**Fast local document classification, scoring, and validation on Apple Silicon — 4.5x–5.8x faster and 2.9 GB smaller than PyTorch MPS.**

`fast-decision-models` runs typed binary (`noul`), multi-choice (`choice`), and scalar (`score`) judges for **Kev-0.8B** on Metal (**54.82 queries/s**, `18.24 ms/q`; `59.68 q/s` standalone `n=24`) and **Laya-Typed-421M** purely on the 16-core Apple Neural Engine (**41.72–42.03 queries/s** with `0` GPU ops, keeping your GPU completely free for LLM or embedding work). Importing the package is pure Python stdlib with zero runtime PyTorch dependency, backed by CLI (`score`) and Python APIs.

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

## At a glance

| Question | Answer |
|---|---|
| `Laya-Typed-421M` ANE compute plan? | `[1585, 0, 0]` (`ANE / GPU / CPU` ops) on both `S=256` and `S=512` buckets (`0` GPU contention). |
| `Kev-0.8B` (`w8`) batch parity? | `B=1` vs `B=3` `max_diff = 0.005613` (`probs = [0.8747, 0.1441, 0.1485]`). |
| Measured throughput (`monorepo_bench`, `n=64`)? | **`54.82 q/s`** (`18.24 ms/q`, `kev_08b w8 B=8` = **`5.82x`** vs `transformers-mps-bf16` & **`-2.87 GB`** OS footprint); **`42.03 q/s`** (`23.80 ms/q`, `laya_ane` = **`4.46x`**). Standalone `n=24`: **`59.68`** / **`41.72 q/s`**. |
| PyTorch needed at runtime? | No. Both `.mlpackage` conversion and `head.pt` extraction use stdlib `zipfile` + `numpy`. |

## Parity & Benchmarks (`decision-models verify` / `bench`, M4 Pro `48 GB`)

### 1. Monorepo Showcase, `bf16` Ablation & Public Baseline (`benchmark_monorepo.py`, `n=64`)

Default suite (`kev_08b w8 B=8` + `laya_ane`) Metal Peak **`1.556 GB`**, OS Footprint **`2.440 GB`** (`RUSAGE_INFO_V4`; recorded with `4` background quiescence warnings):

| Engine / Baseline | Precision | Batch | Questions | Elapsed | Throughput | Latency | Metal Peak | OS Footprint | vs MPS `B=8` | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`kev_08b` (Default)** | `W8` (`g64`) | `8` | `64` | `1.1675 s` | **54.82 q/s** | **18.24 ms/q** | `1.556 GB` *(suite)* | **2.440 GB** *(suite)* | **`5.82x`** | **`-2.87 GB`** OS vs MPS; **`-0.605 GB`** vs `bf16 B=8` |
| `laya_ane` (`Laya-Typed-421M`) | `W8A16` ANE | `P=4` | `64` | `1.5229 s` | **42.03 q/s** | **23.80 ms/q** | `1.556 GB` *(suite)* | **2.440 GB** *(suite)* | **`4.46x`** | `[1585,0,0]` ANE/GPU/CPU ops |
| `kev_08b` (Internal baseline) | `BF16` | `8` | `64` | `1.1863 s` | `53.95 q/s` | `18.54 ms/q` | `2.283 GB` | `3.045 GB` | `5.73x` | `w8 B=8` runs at `1.02x` speed with **`-0.605 GB`** OS |
| `kev_08b` (Internal baseline) | `BF16` | `1` | `24` | `0.8198 s` | `29.27 q/s` | `34.16 ms/q` | `1.622 GB` | `2.061 GB` | `3.11x` | `w8 B=8` is **`1.87x`** faster |
| `transformers-mps-bf16` (Public) | `BF16` | `8` | — | — | `9.42 q/s` | `106.12 ms/q` | `1.505 GB` | `5.308 GB` | `1.00x` | Public MPS (`load=10.934 s`) |
| `transformers-mps-bf16` (Public) | `BF16` | `1` | — | — | `8.87 q/s` | — | — | — | `0.94x` | Public single-stream baseline |

Monorepo `verify` (`1.671 s`, `1.782 GB` OS): `laya-ane` `[1585, 0, 0]` (`s256` & `s512`, `probs=[0.7107, 0.2738, 0.3558]`) and `kev-0.8b (w8)` `B=1 vs B=3` `max_diff=0.005613` (`probs=[0.8747, 0.1441, 0.1485]`).

### 2. Standalone Quick Bench (`decision-models bench`, `n=24`, `batch_docs=8`)

| Backend | Compute Plan / Parity | Elapsed | Throughput | Latency |
|---|---|---:|---:|---:|
| `kev-08b` (`Kev-0.8B r15 w8 g64`) | `B=1 vs B=3` `max_diff=0.005613`; probs `[0.8747, 0.1441, 0.1485]` | `0.4022 s` | **59.68 q/s** | **16.76 ms/q** |
| `laya-ane` (`Laya-Typed-421M W8A16`) | `[1585, 0, 0]` (`s256` & `s512`); probs `[0.9326, 0.0119, 0.0998]` | `0.5753 s` | **41.72 q/s** | **23.97 ms/q** |

## What keeps it fast

| Model / Engine | Optimization | Source |
|---|---|---|
| **Laya-Typed-421M ANE** | Piecewise Scale-Invariant Residual Re-Centering (`1 -> 1/32 -> 1/256 -> 1/1024`) w/ compensated FP16 LayerNorm, scalar FP16 LUT + NEON `8x8` transpose into `IOSurface`-backed `CVPixelBuffer`s, `P=4` async CoreML ring | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm) |
| **Kev-0.8B Metal MLX** | Lossless `W8 (g64)` + `BF16` merged weights, `@mx.compile` forward, left-padded `B=8` batching, `_GDN_PACKED_BTREE_SRC`, stdlib ZIP `head.pt` loader | [`kev_mlx.py`](src/fast_decision_models/kev_mlx.py) |

## CLI, Setup & Verification

```bash
# Score typed questions (`noul`, `choice`, `score`)
.venv/bin/decision-models score --backend kev-0.8b --type choice \
  --question "Which subsystem caused the alert?" --criteria "network,storage,authentication" \
  --state "Primary DB volume reached 99% utilization and paused writes." --json

# Stage W8 Kev weights or rebuild Laya CoreML ANE bundles (`[build]` extra adds coremltools)
.venv/bin/decision-models setup-weights --quantize-existing-kev-w8
.venv/bin/decision-models setup-weights --build-laya-ane

# Static guard + tests
fast-ir-guard --strict-bool-mask src tests                                    # 0 errors, 0 warnings
FAST_DECISION_MODELS_LAYA_DIR=/path/to/laya_typed .venv/bin/pytest -v tests   # 11 passed (10 passed, 1 skipped without laya_typed)
```

Environment overrides: `FAST_DECISION_MODELS_HOME`, `FAST_DECISION_MODELS_ARTIFACTS_DIR`, `FAST_DECISION_MODELS_LAYA_DIR`, `FAST_DECISION_MODELS_KEV_DIR`.

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, macOS 15+/26.6)**. Running `test_decision_models_cpu.py` offline needs zero external weights; hardware tests require staged `laya_typed` and/or `kev_08b`.
- **Not tested:** Concurrent unserialized Metal + ANE workloads across sibling processes, or non-Apple-Silicon GPUs.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
