# fast-decision-models

Apple Silicon hardware-limit inference runners and builders for typed-question decision models (**`noul`**, **`choice`**, and **`score`**):

1. **`Laya-Typed-421M` (`convaiinnovations/laya-typed-decisions`) on 16-core Apple Neural Engine (`src/fast_decision_models/laya_ane.py`, `native/laya_ane_runner.mm`)**
   - **`W8A16` Linear-Quantized CoreML ANE (`S=256` and `S=512` buckets)**: **`[1585, 0, 0]`** (`ANE / GPU / CPU`) operations via **Piecewise Scale-Invariant Residual Re-Centering** (`1 -> 1/32 -> 1/256 -> 1/1024`), eliminating FP16 norm overflow (`0` Metal GPU ops, `0` GPU memory contention).
   - Custom Objective-C++ CoreML runner (`build/liblaya_ane_runner.dylib`) with Accelerate/NEON FP16 LUT + 8×8 register transpose and `P=4` asynchronous prediction pipelining (`MLComputeUnitsCPUAndNeuralEngine`).
2. **`Kev-0.8B r15` (`jaredpalmer/kev-0.8b` + `Qwen/Qwen3.5-0.8B-Base`) on 20-core Metal GPU MLX (`src/fast_decision_models/kev_mlx.py`)**
   - **Lossless `W8` (`g64`) + `BF16` Merged Weights** with fused `@mx.compile` forward pass, left-padded `B=8` cross-state batching, zero-copy `GatedDeltaNet` (`_GDN_PACKED_BTREE_SRC`) + GQA SDPA (`bool` mask-free fast paths + scalar `inv_t` temperature scaling), and pure-stdlib ZIP/pickle `head.pt` extraction (`_load_kev_head_tensors_stdlib_zip`).

Built alongside sibling engines [`fast-bge-small`](../fast-bge-small), [`fast-embeddinggemma-2`](../fast-embeddinggemma-2), [`fast-gemma4`](../fast-gemma4), and [`fast-gpt-oss`](../fast-gpt-oss), sharing `fast_shared.bench`, `fast_shared.ir_dtype_guard`, and `fast_shared.include/fast_shared_apple.hpp` via [`fast-shared`](../fast-shared).

## Requirements

- **macOS 15+ / Apple Silicon** (Xcode toolchain for CoreML ANE Objective-C++ dylib compilation).
- **Python >= 3.11** (`fast-shared`, `mlx>=0.25`, `numpy>=1.26`, `tokenizers>=0.20`, `huggingface-hub>=0.25`).
- Optional `[build]` extra (`coremltools>=8.0`, `torch>=2.4` — note PyTorch is only needed if invoking `torch` utilities elsewhere; `head.pt` loading and CoreML `.mlpackage` conversion use stdlib `zipfile` + `numpy` + `coremltools`).

## Build & Setup

```bash
cd fast-decision-models
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
```

Stage `Kev-0.8B` (`BF16` + `W8`) and/or `Laya-Typed-421M` CoreML ANE bundles into `artifacts/` (from local HuggingFace cache or download):

```bash
# Stage Kev-0.8B r15 merged BF16 + W8-g64 weights
.venv/bin/decision-models setup-weights --build-kev-mlx

# Quantize an already-staged BF16 Kev-0.8B checkpoint -> model_w8.safetensors
.venv/bin/decision-models setup-weights --quantize-existing-kev-w8

# Stage & compile Laya-Typed-421M CoreML ANE bundles (requires coremltools env via [build])
.venv/bin/decision-models setup-weights --build-laya-ane
```

Environment overrides:
- `FAST_DECISION_MODELS_HOME`: override repo root for `artifacts/` and `build/liblaya_ane_runner.dylib`.
- `FAST_DECISION_MODELS_ARTIFACTS_DIR`: override base `artifacts/` directory.
- `FAST_DECISION_MODELS_LAYA_DIR`: override `laya_typed/` bundle directory.
- `FAST_DECISION_MODELS_KEV_DIR`: override `kev_08b/` weight directory.

## CLI & Python Usage

```bash
# Verify Kev-0.8B W8/BF16 parity and/or Laya-Typed ANE ComputePlans
.venv/bin/decision-models verify --backend kev-0.8b
.venv/bin/decision-models verify --backend all

# Score a single typed question (`noul`, `choice`, or `score`)
.venv/bin/decision-models score --backend kev-0.8b --type noul \
  --question "Was release v2.8.1 rolled back due to failing canary checks?" \
  --criteria '{"false":"Promoted cleanly.","true":"Rolled back due to failing canary checks."}' \
  --state "Deploy log: v2.8.1 rolled back after canary 5xx exceeded 5%." --json

# Score a multi-way choice question
.venv/bin/decision-models score --backend kev-0.8b --type choice \
  --question "Which subsystem caused the alert?" \
  --criteria "network,storage,authentication" \
  --state "Primary DB volume reached 99% utilization and paused writes." --json

# Benchmark throughput
.venv/bin/decision-models bench --backend all --n 24 --batch-docs 8
```

Python API (`import fast_decision_models` is 100% Python stdlib; native/MLX backends load lazily):

```python
from fast_decision_models import Kev08bMlxBackend, LayaAneBackend

with Kev08bMlxBackend(quant_mode="auto", batch_docs=8) as kev:
    results = kev.score_multi_states(
        [
            (
                "Invoice #8812: Total $1,240.00. Status: PAID via ACH.",
                [
                    {
                        "id": "overdue",
                        "type": "noul",
                        "instructions": "Is the balance overdue and unpaid?",
                        "criteria": {"false": "Paid.", "true": "Overdue and unpaid."},
                    }
                ],
            )
        ]
    )
    probs, logits = results[0][0]
```

## Verification & IR Guard (M4 Pro)

- `make guard` (`fast-ir-guard --strict-bool-mask src tests`): `0 errors, 0 warnings`.
- CPU + Hardware pytest (`FAST_DECISION_MODELS_LAYA_DIR=<path/to/laya_typed> .venv/bin/pytest -v tests`): **`11 passed in 2.59s`** (`8` CPU stdlib-isolation/ZIP-reader/path/dylib tests + `3` ANE/MLX hardware tests; without external `laya_typed`, `.venv/bin/pytest` runs **`10 passed, 1 skipped`**).
- `decision-models verify --backend all`: `[PASS] laya-ane plans={'s256': (1585, 0, 0), 's512': (1585, 0, 0)} probs=[0.9326, 0.0119, 0.0998]`; `[PASS] kev-0.8b (w8) B=1 vs B=3 max_diff=0.005613 probs=[0.8747, 0.1441, 0.1485]`.
- `decision-models bench --backend all --n 24 --batch-docs 8`: `laya_ane` **`41.72 q/s`** (`23.97 ms/q`, `0.5753 s`); `kev_08b` (`w8`) **`59.68 q/s`** (`16.76 ms/q`, `0.4022 s`).
