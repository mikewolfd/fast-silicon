# fast-gemma4

Hardware-limit batch structured JSON extraction and inference engine for **Gemma-4 E2B** (`lmstudio-community/gemma-4-E2B-it-MLX-4bit` / `google/gemma-4-E2B-it`) on Apple Silicon M4 Pro (**20-core Metal 4 GPU + 16-core ANE**).
Combines a continuous length-sorted rolling slot scheduler (`B_slots=128`), INT8 KV compression (`kv_i8`), Metal 4 `MPP::matmul2d` BF16 kernels, lossless `150,786` vocab pruning, `llguidance` causal slice jumping, and an optional async Linear-W8 ANE prefix ring.

## Quick start

```bash
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
.venv/bin/gemma4 setup-weights --local-files-only
.venv/bin/gemma4 verify
```

```python
from fast_gemma4 import FastGemma4Engine

with FastGemma4Engine(default_mode="auto") as eng:
    item = eng.generate(
        "Extract entity and topic as JSON.",
        json_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
            "required": ["entity", "topic"],
            "additionalProperties": False,
        },
        max_tokens=128,
    )
    print(item.parsed)
```

## At a glance

| Question | Answer |
|---|---|
| How fast vs stock `mlx_vlm`? | **`8.92 docs/s`** (`1,192 dec tok/s`, Concurrent ANE+GPU `2xB=128`) = **`2.54x`** `docs/s` (`1.48x` dec tok/s, **`-3.50 GB`** peak) vs `stock-mlx_vlm B=64` (`3.51 docs/s`, `804.1` dec tok/s). Standalone `B_slots=128`: **`669.3–704.8` e2e out tok/s** (`2.33 docs/s` Parquet). |
| Peak memory at `B_slots=128`? | **`6.26 GB`** (`kv_i8`, `monorepo_bench`; `6.718 GB` suite Metal peak, **`-0.67 GB`** vs `no_kv_i8` `6.93 GB`); standalone `B_slots=128` **`6.88 GB`** vs `34.4 GB` OOM unchunked in `mlx_vlm`. |
| How fast is Linear-W8 ANE prefix ring? | `gemma4-ane-cli`: **`7,814.8 tok/s`** (`32.758 ms/chunk`, `1,326` ANE ops, `0` GPU/CPU). Paired `ABBA` `AsyncHybridPrefillRing`: **`2.723x` prefill** / **`1.158x` E2E** (`84/84` exact field parity). |
| Does `llguidance` lose rare tokens? | No: `150,786` pruned vocab covers English+JSON (`0` misses) with lazy `262,144` full-vocab fallback + byte tokens. |

## Verification & Parity (`gemma4 verify` & `pytest`)

| Check | Result (`48 GB` M4 Pro) |
|---|---|
| `pytest -v` (full M4 Pro suite) | **`43 passed, 1 skipped`** (`44` total; `0` `fast-ir-guard --strict-bool-mask` errors/warnings; `test_latin_extended_vocab162304_covers_accented_names_cpu` skipped) across rolling kernels, `KV_i8`, ANE CLI, schema/PromptLookup, Parquet & HTTP SSE |
| `gemma4 verify` (`vocab_out=150786`, `6.57 GB`) | `monorepo verify` (`50.855 s`): `native_ane_inspect` `S=256` `5,273.7 tok/s`; `GPU 8-doc`: `8/8` (`250` vs `PromptLookup 303 tok/s`); `Hetero 64-doc`: `64/64` (`4.20 docs/s`, `928` dec tok/s). Standalone: `Hetero` `769` dec tok/s / `2.57 docs/s` |
| Native CoreML (`S=256`, `gemma4-ane-cli`) | `ANE=1326, GPU=0, CPU=0`, `zero_copy=1` `CVPixelBuffer -> KV_i8`, `max_int8_diff == 0`, `cos = 0.9991–0.9998` |
| Causal Slice Jumping (`B=1`, `enable_ff_jump=True`) | Bit-exact greedy parity, `177 -> 151` GPU passes (`1.047x` `ABBA` decode speedup) |

## Speed & Memory (`48 GB` M4 Pro `Mac16,8`)

### Monorepo Showcase vs Stock `mlx_vlm` (`benchmark_monorepo.py`)

Verified in the unified monorepo runner (`scripts/benchmark_monorepo.py --packages fast-gemma4`, suite Metal Peak `6.718 GB`, OS Footprint `18.314 GB` via `RUSAGE_INFO_V4`):

| Engine / Mode | Valid / Strict | Prefill (`s`) | Decode (`s`) / tok/s | Docs/s | Peak Mem | Notes |
|---|---|---:|---|---|---:|---|
| **`Concurrent ANE+GPU 2xB=128`** (`W8` ring) | **`256/256`** | `3.39` (`pre+wait`) | `24.82` (**1,192 dec tok/s**) | **8.92** | **6.26 GB** | **`1.21x`** vs `GPUOnly`; **`2.54x`** `docs/s` & **`1.48x`** dec tok/s vs `stock-mlx_vlm B=64` |
| `GPUOnly B=128` (`kv_i8=True`) | `128/128` | `5.61` | `11.13` (**1,265 tok/s**) | 7.36 | **6.26 GB** | vs `no_kv_i8`: `1,218 tok/s` / `6.83 docs/s` / `6.93 GB` (`kv_i8` saves **`-0.67 GB`**) |
| `PromptLookup B=32` (`kv_i8=True`) | `32/32` | `1.42` | `7.23` (**516 eff tok/s**) | 3.64 | **6.10 GB** | vs `no_kv_i8 B=1`: `79 eff tok/s` / `0.48 docs/s` (`4.92 GB`) |
| `stock-mlx_vlm` (`B=64`, Public) | `64/64` JSON | — | **804.1 dec tok/s** (`455.8` e2e out) | 3.51 | `9.764 GB` Metal (`13.404 GB` OS) | Public baseline (`B=1`: `68.8 tok/s`, `0.44 docs/s`; `fast-gemma4` saves **`-3.50 GB`** peak) |

*Note: Monorepo capture ran alongside light background system activity (`6` quiescence warnings recorded in the run manifest).*

### End-to-End Rolling GPU vs Stock `mlx_vlm`

| Engine / Checkpoint | Workload | KV Cache | Docs | Valid | Decode tok/s | E2E Out / Total tok/s | Docs/s | Peak Mem |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Stock `mlx_vlm` `B=1` (`chunk=None`) | `L_in=2015, L_out=64` | FP16/BF16 | 1 | — | 68.7 | **24.4** / 793.2 | — | 5.65 GB |
| Stock `mlx_vlm` `B=32` (`chunk=256`) | `L_in=1024, L_out=64` | FP16/BF16 | 32 | — | 555.1 | **66.9** / 1,137.6 | — | 9.82 GB |
| Stock `mlx_vlm` `B=64` (`chunk=256`) | `L_in=1024, L_out=64` | FP16/BF16 | 64 | — | 570.8 | **66.6** *(34.4 GB OOM unchunked)* | — | 15.24 GB |
| `fast-gemma4` `MLX-4bit` (`bench`) | Structured JSON (`B=64`) | `kv_i8=True` | 64 | 64/64 | 773.0 | — | 2.39 | **6.30 GB** |
| `fast-gemma4` `MLX-4bit` (Rolling) | Structured JSON (`56`-doc) | `kv_i8=True` | 256 | 208/256 | 878.9 | **704.8** / 3,734.3 | 1.648 | **6.88 GB** |
| `fast-gemma4` Tuned LoRA (Rolling) | Structured JSON (`56`-doc) | `kv_i8=False` | 256 | 251/256 | 894.6 | **665.5** / 4,440.6 | 2.054 | 8.09 GB |
| `fast-gemma4` Tuned LoRA (Rolling) | Structured JSON (`56`-doc) | `kv_i8=True` | 256 | 251/256 | 894.6 | **669.3** / 4,490.3 | 2.079 | **6.88 GB** |
| `fast-gemma4` Tuned LoRA (`parquet`) | Multi-page `p1-2` (`max_chars=8000`) | `kv_i8=True` | 255 | **255/255** | **1,051.8** | **671.1** / **5,443.2** | **2.33** | — |

### Native ANE Prefix Kernel (`S=256`, `pipe=4`) & Paired `ABBA` Hybrid Ring

| Bundle / Pipeline Variant | Batch / Slots | Compute Plan (`ANE / GPU / CPU`) | Time / Latency | Throughput |
|---|---:|---|---:|---:|
| `gemma4_early14_prefix_S256_fp16.mlmodelc` | `B=16, iters=3` | `1326 / 0 / 0` | `0.7063 s` (`44.144 ms/chunk`) | `5,799.2 tok/s` |
| **`gemma4_early14_prefix_S256_w8.mlmodelc` (`AWQ-MSE`)** | `B=16, iters=3` | `1326 / 0 / 0` (`zero_copy=1`) | **`0.5241 s` (`32.758 ms/chunk`)** | **`7,814.8 tok/s`** |
| Tuned LoRA GPU Rolling (`use_ane_ring=False`, `ABBA`) | `N=96, B_slots=48` | Metal GPU | `62.043 s` (`14.882 s` prefill) | `1.55 docs/s` (`84/96`) |
| **Tuned LoRA Hybrid Linear-W8 ANE Ring (`use_ane_ring="auto"`)** | `N=96, B_slots=48` | ANE + Zero-Copy `KV_i8` + GPU | **`53.566 s` (`1.158x`, `5.465 s` `2.723x` prefill)** | **`1.79 docs/s`** (`84/84` match) |

Zero-shot model note: `QAT-4bit` (`50.0%` exact / `58.8` date F1 on `48`-doc eval) outperforms `PTQ MLX-4bit` (`6.2%` / `9.0`) at identical compute throughput (`4-bit g64` + `105` `8-bit` MLP overrides), while Tuned LoRA lifts `docs/s` via `25%` shorter structured outputs (`322` vs `428` tok/doc) and `72x` fewer grammar fallbacks.

## What keeps it fast

| Component | Optimization | Source |
|---|---|---|
| **Rolling Metal GPU + `KV_i8`** | Continuous `B_slots=128` scheduler (`refill_min=32`, `group_size=16`), INT8 KV slot compression + `bfloat4` split-K reduce, `mpp_fused_geglu_bf16_v4`, `EarlyExit14`/`23`, NaN-safe softcap argmax | [`model.py`](src/fast_gemma4/model.py), [`kernels.py`](src/fast_gemma4/kernels.py) |
| **Lossless Vocab Pruning** | `262,144 -> 150,786` pruned embeddings + PLE tables (`build_vocab_remap.py`) with lazy `262K` fallback | [`tokenizer.py`](src/fast_gemma4/tokenizer.py) |
| **Structured JSON & Speculative** | `llguidance` zero-alloc bitmask, `forward_ff_jump_slot` causal slice jump, automatic `PromptLookup` (`Bw <= 4`) | [`grammar.py`](src/fast_gemma4/grammar.py), [`speculative.py`](src/fast_gemma4/speculative.py) |
| **Linear-W8 ANE Ring Bridge** | `AWQ-MSE` Linear `W8`, unlocked `CVPixelBuffer` -> `-newBufferWithIOSurface:` Metal `bridge_scatter_*`, background producer ring | [`ane_pipeline.py`](src/fast_gemma4/ane_pipeline.py), [`native/gemma4_ane_prefix.mm`](native/gemma4_ane_prefix.mm) |
| **Routing, Parquet & HTTP** | Auto rolling GPU on overridden `--model-dir`, UTF-8 `page_offsets` (`--max-pages`, `--ocr-pages-only`), coalescing SSE OpenAI server | [`parquet_pipeline.py`](src/fast_gemma4/), [`server.py`](src/fast_gemma4/) |

## CLI & Reproduce

```bash
.venv/bin/gemma4 setup-ane --variant both                          # Rebuild CoreML bundles ([ane-build])
.venv/bin/gemma4 generate --json 'Extract entity: Acme Cloud grew 12%.'
.venv/bin/gemma4 parquet docs.parquet out_shards/ --model-dir /ckpt \
  --schema schema.json --system-file sys.txt --max-pages 2 --max-chars 8000 --drop-text-col --resume
.venv/bin/gemma4 extract docs.jsonl out.jsonl --schema schema.json --system-file sys.txt
.venv/bin/gemma4 serve --host 127.0.0.1 --port 8080 --coalesce-ms 10
.venv/bin/gemma4 bench --batch-gpu 128 --batch-spec 32 --waves 2
./build/gemma4-ane-cli --model artifacts/ane_models/gemma4_early14_prefix_S256_w8.mlmodelc --batch 16 --iters 3 --inspect --json
```

## Scope & Not-tested

- Verified via paired `ABBA` trials under `/tmp/m4_gpu_validation.lock` on an uncontended **Apple M4 Pro (`Mac16,8`, `48 GB`)**.
- **Not tested:** Concurrent unserialized Metal/ANE processes, smaller unified-memory Macs (`< 24 GB` at `B_slots=128`), or non-Apple-Silicon targets.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
