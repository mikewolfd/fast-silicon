# fast-decision-models

Apple Silicon hardware-limit runners for typed-question decision models (**`noul`**, **`choice`**, **`score`**) targeting **Laya-Typed-421M (`W8A16`)** on the 16-core Apple Neural Engine and **Kev-0.8B r15 (`W8`/`BF16`)** on the 20-core Metal GPU.
`import fast_decision_models` stays pure Python stdlib while lazy backends drive zero-GPU-contention CoreML ANE (`P=4` `IOSurface`) or left-padded batched MLX execution.

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
| Measured throughput (`monorepo_bench`, `n=64`)? | **`50.95 q/s`** (`19.63 ms/q`, `kev_08b w8 B=8` = **`5.27x`** vs `transformers-mps-bf16` & **`-2.13 GB`** OS footprint); **`40.53 q/s`** (`24.67 ms/q`, `laya_ane` = **`4.19x`**). Standalone `n=24`: **`59.68`** / **`41.72 q/s`**. |
| PyTorch needed at runtime? | No. Both `.mlpackage` conversion and `head.pt` extraction use stdlib `zipfile` + `numpy`. |

## Parity & Benchmarks (`decision-models verify` / `bench`, M4 Pro `48 GB`)

### Monorepo Showcase, `bf16` Ablation & Public Baseline (`benchmark_monorepo.py`, `n=64`)

Verified in the unified monorepo runner (`scripts/benchmark_monorepo.py --packages fast-decision-models`; `w8`+`laya` suite Metal Peak `1.748 GB`, OS Footprint `3.166 GB` via `RUSAGE_INFO_V4`):

| Engine / Baseline | Batch / Precision | Throughput | Latency | Time (`s`) | Footprint / vs Public |
|---|---|---:|---:|---:|---|
| **`kev_08b` (`w8 B=8`, Default)** | `w8`, `B=8` (`n=64`) | **50.95 q/s** | **19.63 ms/q** | 1.2562 | OS `3.166 GB` (**`5.27x`** vs `transformers-mps`, **`-2.13 GB`**; `1.86x` vs `bf16 B=1`) |
| `laya_ane` (`Laya-Typed-421M`) | `W8A16` ANE (`n=64`) | **40.53 q/s** | **24.67 ms/q** | 1.5789 | Plans `[1585,0,0]` (`s256` & `s512`); **`4.19x`** vs `transformers-mps B=8` |
| `kev_08b` (`bf16 B=8`, Internal) | `bf16`, `B=8` (`n=64`) | 51.65 q/s | 19.36 ms/q | 1.2390 | OS `3.718 GB` (`w8 B=8` is `0.99x` speed with **`-0.553 GB`** footprint) |
| `kev_08b` (`bf16 B=1`, Internal) | `bf16`, `B=1` (`n=24`) | 27.42 q/s | 36.48 ms/q | 0.8754 | OS `2.086 GB` |
| `transformers-mps-bf16` (Public) | `bf16`, `B=8` (`B=1`: `2.98 q/s`) | 9.67 q/s | 103.38 ms/q | — | OS `5.295 GB`, Metal `1.505 GB` (`load=11.554 s`) |

Monorepo `verify` (`6` host quiescence warnings): `laya-ane` `[1585, 0, 0]` (`s256` & `s512`, `probs=[0.7107, 0.2738, 0.3558]`) and `kev-0.8b (w8)` `B=1 vs B=3` `max_diff=0.005613` (`probs=[0.8747, 0.1441, 0.1485]`).

### Standalone Quick Bench (`decision-models bench`, `n=24`, `batch_docs=8`)

| Backend | Plan / Parity | Throughput | Latency | Time (`n=24`) |
|---|---|---:|---:|---:|
| `laya-ane` (`Laya-Typed-421M W8A16`) | `[1585, 0, 0]` (`s256` & `s512`); probs `[0.9326, 0.0119, 0.0998]` | **41.72 q/s** | **23.97 ms/q** | `0.5753 s` |
| `kev-08b` (`Kev-0.8B r15 w8 g64`) | `B=1` vs `B=3` `max_diff=0.005613`; probs `[0.8747, 0.1441, 0.1485]` | **59.68 q/s** | **16.76 ms/q** | `0.4022 s` |

## What keeps it fast

| Model / Engine | Optimization | Source |
|---|---|---|
| **Laya-Typed-421M ANE** | Piecewise Scale-Invariant Residual Re-Centering (`1 -> 1/32 -> 1/256 -> 1/1024`), Accelerate/NEON LUT + `8x8` transpose, `P=4` async CoreML ring | [`laya_ane.py`](src/fast_decision_models/laya_ane.py), [`native/laya_ane_runner.mm`](native/laya_ane_runner.mm) |
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
