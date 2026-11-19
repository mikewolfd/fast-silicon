# fast-gemma4

Standalone Apple Silicon (M4 Pro **20-core Metal GPU + 16-core Apple Neural Engine**) high-throughput batch structured extraction & inference engine for **Gemma-4 E2B** (`lmstudio-community/gemma-4-E2B-it-MLX-4bit` / `google/gemma-4-E2B-it`).

Built as a sister repository to [`fast-bge-small`](../fast-bge-small) and [`fast-embeddinggemma-2`](../fast-embeddinggemma-2), sharing Apple Silicon Metal MPP BF16 kernel primitives, C++ helpers, path resolution, and streaming Parquet utilities via [`fast-shared`](../fast-shared).

## Architecture & Capabilities

1. **Metal GPU Ragged + Rolling Slot Engine (`src/fast_gemma4/model.py`, `src/fast_gemma4/kernels.py`)**
   - Mixed-precision `BF16` promoted MLPs & projections paired with `Q4` pruned LM head (`6.57 GB` active parameters) and multi-shard `model*.safetensors` support (supports `4-bit`, `QAT-4bit`, `BF16`, and fused LoRA checkpoints cleanly).
   - **Continuous Length-Sorted Rolling Slot Scheduler (`run_decode_rolling`)**: `B_slots=128` persistent slot pool (`refill_min=32`, `group_size=16`, `Layer-15` prefix split `n_keep`, multi-token `EarlyExit14` jump-forward drain `max_ff_drain=8`, and single-launch **`ff_q` causal slice jumping** `forward_ff_jump_slot` / `keep_from_col` for low-batch constrained forced spans) eliminating static wave tail stalls.
   - **INT8 Per-Doc KV Compression (`kv_i8=True` default)**: fused symmetric INT8 KV quantizer (`g4_quantize_kv_pair_i8_d*`) + slot-indexed MQA-8 split-K SDPA (`g4_mqa8_prefix_slot_i8_d*`) with `bfloat4` vectorized split-KV reduction (`g4_mqa8_prefix_reduce_vec4_d*`), cutting `B=128` peak RAM to **`6.88 GB`**.
   - **Metal 4 `MPP::matmul2d` & Lossless Writer Safety**: `mpp_fused_geglu_bf16_v4` (`m32_cutoff=96`), `mpp_linear_bf16`, `g4_fused_qkv_ragged_d*`, `g4_fused_kv_only_ragged_d*`, `g4_fused_res_rms_norm_d*` (`m_rows == 0` empty-batch safe; `use_vec4_norm=False` default after paired `ABBA` showed `256`-thread `fused_norm_add_norm` wins `0.984–0.987x` at `B=1..16`), `g4_fused_ple_geglu_idx_d256`, NaN-safe `g4_fused_softcap_argmax_*` (`-1` sentinel + `FloatingPointError`), and lazy `262,144` full-vocab logit fallback (`compute_full_vocab_logits_1d`) for unpruned rare Unicode grammar literals.
   - **EarlyExit14 / EarlyExit23** with `k = first_kv_shared` sliced `pli_proj` (`EarlyExit14` on E2B, `EarlyExit23` on E4B).

2. **Lossless Pruned Vocabulary (`src/fast_gemma4/tokenizer.py`, `scripts/build_vocab_remap.py`)**
   - Shrinks `embed_tokens`, `lm_head`, and all 35 `embed_tokens_per_layer` (`PLE`) tables from `262,144 -> 150,786` tokens (`vocab_remap_150786.npz`), covering ASCII/Latin/punctuation/control/digits/special tokens with lossless byte-level fallback (`0` misses on English + JSON fixtures).

3. **Zero-Overhead Structured JSON & Speculative Decoding (`src/fast_gemma4/grammar.py`, `src/fast_gemma4/speculative.py`)**
   - Compiles JSON Schemas with `llguidance` onto pre-allocated `unsafe_compute_mask_ptr` bitmask buffers (`0` per-fallback buffer allocation), computing upfront & deterministic jump-forward tokens (`compute_ff_tokens`).
   - **Multi-Token Forced-Span Causal Slice Jumping (`enable_ff_jump=True`)**: evaluates deterministic `k >= 2` forced spans (e.g. `","correspondence_flow":"`) in 1 causal slice (`forward_ff_jump_slot` / `keep_from_col`, `M <= 15`) with bit-exact greedy parity at `B=1..4` (`1.047x` paired `ABBA` `B=1` decode speedup, `177 -> 151` GPU passes).
   - Routes `B <= 4` (or explicit `speculative=True`) to **Prompt-Lookup N-gram (`3`-gram/`2`-gram)** `L_q=4` batched verification (`run_decode_prompt_lookup`), while `B > 4` GPU batches automatically run the continuous `run_decode_rolling` scheduler.

4. **Linear-W8 ANE Reader, Zero-Copy `IOSurface -> Metal KV_i8` Scatter Bridge & `AsyncHybridPrefillRing` (`scripts/build_ane_models.py`, `native/gemma4_ane_prefix.mm`, `src/fast_gemma4/ane_pipeline.py`)**
   - **Per-Channel Symmetric Linear `int8` + Closed-Form `AWQ-MSE` (`--awq-mse`)**: compiles `1,326 / 1,326` CoreML operations (`0` CPU runtime ops, `86` `constexpr` ops, `8s` `MLComputePlan` timeout guard, never `per_block` or legacy k-means LUT) on the **16-core Apple Neural Engine** at **`32.76 ms/chunk` (`7,814.8 tok/s`)** with **`cos = 0.9991–0.9998`** vs exact GPU `EarlyExit14` (supports checkpoint-scoped custom `model_dir` bundles).
   - **Zero-Copy `IOSurface -> Metal KV_i8` Scatter Bridge**: unlocked `CVPixelBuffer` (`OneComponent16Half`) output ring (`0` `.dataPointer` locks/copies) bound via `-newBufferWithIOSurface:` to 2 Metal compute dispatches (`bridge_scatter_swa12` / `bridge_scatter_full3`, `max_int8_diff == 0`, `~0.10 ms/chunk`).
   - **`AsyncHybridPrefillRing` (`use_ane_ring=True | "auto"`)**: Wave-0 `50/50` `GPU + ANE` bootstrap + background ANE producer refilling freed rolling slots (`CPU SME2` kept parked for `S_suffix <= 1,200`), delivering **`2.723x` faster prefill** and **`1.158x` (`+15.8%`) E2E throughput** on structured JSON extraction (`N=96, B_slots=48`) with `84/84` exact field parity.

5. **CLI, Batch Pipelines, OpenAI HTTP Server, and Tuned Defaults**
   - `FastGemma4Engine`: prefix `KV_sys` LRU cache and tuned `mode="auto"` routing:
     - **Rolling GPU vs Explicit Hetero ANE+GPU**: `mode="auto"` uses continuous `run_decode_rolling` on GPU (`6.88 GB` peak RAM, `1.648–2.238 docs/s`), automatically disabling base-4bit ANE bundles whenever `model_dir` overrides the default weights; `mode="hetero"` explicitly runs overlapped `ANE+GPU` waves on matching base weights.
     - **PromptLookup Auto Threshold**: automatically enabled in the bandwidth-bound **`Bw <= 4`** regime at `temperature=0.0` (with explicit `speculative=True` / `--speculative` override or `--no-speculative` to force rolling).
     - **JSON Whitespace & Headroom Safety**: `FastGemma4Engine.generate()`, `gemma4 generate`, and `FastGemma4Server` default to `whitespace_flexible=True` for ad-hoc prompts; batch `generate_batch()` / `extract` / `parquet` default to compact minified prompts (`whitespace_flexible=False`, `--whitespace-flexible` available).
   - `extract_parquet` & `extract_jsonl`: streaming, sharded, `--resume`-capable (`id` or `body_sha` keyed) Parquet/JSONL pipelines with UTF-8 `page_offsets` slicing (`--max-pages N`, `--ocr-pages-only`, `--filter-ocr`), `max_chars` cap, and `body_text` auto-detection.
   - `FastGemma4Server` & `FastGemma4OpenAIAdapter` (backed by `fast_shared.server.ManagedHTTPServer`): OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models{,/:id}`, and `/health` with `response_format` JSON Schema enforcement, SSE streaming (`stream: true`), `extra_body` forwarding, leader-follower concurrent micro-batching (`--coalesce-ms`, `--max-batch`), and pluggable mounting into `fast_shared.server.UnifiedOpenAIServer`.

## Build & Setup

```bash
cd fast-gemma4
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
```

Set up tokenizer, lossless `150,786` vocabulary remap, and Early14 FP16 weights (from local HuggingFace cache):

```bash
.venv/bin/gemma4 setup-weights --local-files-only
# Optional: rebuild CoreML .mlmodelc bundles (`uv pip install -e ".[ane-build]"`)
.venv/bin/gemma4 setup-ane --variant both
```

## CLI & Python Usage

```bash
# Verify native ANE, JSON schema, PromptLookup, and Hetero ANE+GPU parity
.venv/bin/gemma4 verify

# Benchmark GPUOnly vs PromptLookup vs Concurrent ANE+GPU
.venv/bin/gemma4 bench --batch-gpu 128 --batch-spec 32 --waves 2

# Standalone zero-Python native ANE chunk benchmark
./build/gemma4-ane-cli --model artifacts/ane_models/gemma4_early14_prefix_S256_w8.mlmodelc --batch 16 --iters 3 --inspect --json

# Single generation / JSON extraction (supports --model-dir, --system-file, --no-kv-i8)
.venv/bin/gemma4 generate --json 'Extract company and event as JSON: Acme Cloud reported quarterly revenue growth on Tuesday.'

# Streaming sharded Parquet & JSONL extraction (e.g. first 2 pages of multi-page Parquet text shards)
.venv/bin/gemma4 parquet docs.parquet out_shards/ --model-dir /path/to/ckpt --schema schema.json --system-file sys.txt --max-pages 2 --max-chars 8000 --drop-text-col --resume
.venv/bin/gemma4 extract docs.jsonl out.jsonl --schema schema.json --system-file sys.txt

# OpenAI-compatible HTTP server
.venv/bin/gemma4 serve --host 127.0.0.1 --port 8080
```

Python API:

```python
from fast_gemma4 import FastGemma4Engine

eng = FastGemma4Engine(default_mode="auto")
try:
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
finally:
    eng.close()
```

## Measured Verification & Benchmarks (M4 Pro)

### Correctness & Unit Verification

- `make native && .venv/bin/python -m pytest -v`: **`44 passed in 108.47s`** (`0:01:48` on uncontended M4 Pro) across [`test_ane_hybrid_and_writer.py`](tests/test_ane_hybrid_and_writer.py), [`test_fast_gemma4.py`](tests/test_fast_gemma4.py), [`test_packager_cpu.py`](tests/test_packager_cpu.py), [`test_per_doc_grammar.py`](tests/test_per_doc_grammar.py), and [`test_tier1_cpu.py`](tests/test_tier1_cpu.py):
  - `test_native_gemma4_ane_cli`: compiles & inspects `gemma4_early14_prefix_S256` on CoreML (`ANE=1326, GPU=0, CPU=0`).
  - `test_tokenizer_chat_and_lossless_vocab_remap`: `vocab_size=150786`, `0` misses on English+JSON fixture, lossless rare Unicode/byte round-trip.
  - `test_prefix_kv_equivalence_and_single_prompt`: shared `KV_sys` split vs unsplit `L_sys=0` prefill (`cos > 0.999`, `max_diff < 1.0`, identical `argmax`).
  - `test_schema_and_prompt_lookup_exactness`: `L_q=1` vs `L_q=4` forward logit parity (`cos > 0.999`, `max_diff < 1.0`) + `100%` valid JSON & strict JSON schema across `B=1` and `B=4`.
  - `test_hetero_ane_gpu_prefix_pipeline`: `8/8` GPU reference and `64/64` concurrent `hetero_ane_gpu` (`lcp <= 1408`) `100%` valid JSON & strict schema with matching top-level semantic fields.
  - `test_parquet_and_jsonl_pipeline`: JSONL extraction + sharded `--resume` Parquet + UTF-8 `page_offsets` OCR & `--max-pages` slicing + `body_text` auto-detection.
  - `test_http_server_smoke`: `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions` structured `json_schema` responses.
  - `test_tuned_defaults_and_routing_cpu` (CPU-only): verifies `auto` GPU rolling routing, explicit `hetero` ANE+GPU flag, `Bw <= 4` PromptLookup cutoff, `whitespace_flexible` grammar validation, and CLI defaults.
  - `test_http_server_coalescing_sse_and_unified_cpu` (CPU-only): verifies concurrent HTTP micro-batching (`coalesce_ms` / `max_batch_size`), SSE streaming, `extra_body`, and `UnifiedOpenAIServer` mounting.
  - `test_v4_kernels_rolling_split_kv_i8`: verifies bit-for-bit `fused_softcap_argmax_bf16` parity (including strided/padded slices), `fused_res_rms_norm`, `fused_ple_geglu_indexed`, `KV_i8` slot cache parity (`cos > 0.998`), continuous rolling refill (`N=12, B_slots=4`), `Layer-15` split, `EarlyExit14` FF drain, unconstrained rolling, and sampled (`T > 0`) rolling.
  - `test_multishard_bf16_and_custom_model_dir_routing`: verifies multi-shard `mlx-community/gemma-4-e2b-it-bf16` loading, automatic ANE disable on overridden `model_dir`, and continuous rolling schema extraction.
- `.venv/bin/gemma4 verify`:
  - `vocab_out=150786`, `param_gb=6.57GB`, `mpp=True`, `ane_available=True`
  - `native_ane_inspect`: `44.548 ms/chunk` -> **`5,746.6 tok/s`** (`1326` ANE ops, `S=256`, `batch=4`)
  - `GPU 8-doc`: `valid_json=8/8`, `valid_schema=8/8`, `PromptLookup semantic_match=8/8` (`spec_valid=8/8`), `decode_gpu=207 tok/s` (`speculative=False`) vs `spec=240 tok/s` (`speculative=True`)
  - `Hetero 64-doc (1 wave ANE+GPU)`: `valid_json=64/64`, `valid_schema=64/64`, `total=2.57 docs/s`, `decode=769 tok/s`

### Rolling Engine, Stock `mlx_vlm` Baseline & Real Corpus Benchmarks (M4 Pro)

| Engine / Checkpoint | Workload | KV Cache | Docs | Valid / Strict | Decode tok/s | E2E Output / Total tok/s | Docs/s | Peak Mem |
| --- | --- | --- | ---: | :---: | ---: | ---: | ---: | ---: |
| Stock `mlx_vlm` `B=1` (`chunk=None`) | `L_in=2015, L_out=64` | FP16/BF16 | 1 | — | `68.7` | **`24.4`** out / `793.2` tot | — | `5.65 GB` |
| Stock `mlx_vlm` `B=32` (`chunk=256`) | `L_in=1024, L_out=64` | FP16/BF16 | 32 | — | `555.1` | **`66.9`** out / `1,137.6` tot | — | `9.82 GB` |
| Stock `mlx_vlm` `B=64` (`chunk=256`) | `L_in=1024, L_out=64` | FP16/BF16 | 64 | — | `570.8` | **`66.6`** out *(OOMs `34.4 GB` if unchunked)* | — | `15.24 GB` |
| `fast-gemma4` `E2B-it-MLX-4bit` (`B=64` bench) | Structured JSON (`sample_docs_8`) | `kv_i8=True` | 64 | `64/64` | `773.0` | — | `2.39` | **`6.30 GB`** |
| `fast-gemma4` `E2B-it-MLX-4bit` (Rolling) | Structured JSON (`56`-doc suite) | `kv_i8=True` | 256 | `208/256` | `878.9` | **`704.8`** out / `3,734.3` tot | `1.648` | **`6.88 GB`** |
| `fast-gemma4` Tuned LoRA (Rolling) | Structured JSON (`56`-doc suite) | `kv_i8=False` | 256 | `251/256` | `894.6` | **`665.5`** out / `4,440.6` tot | `2.054` | `8.09 GB` |
| `fast-gemma4` Tuned LoRA (Rolling) | Structured JSON (`56`-doc suite) | `kv_i8=True` | 256 | `251/256` | `894.6` | **`669.3`** out / `4,490.3` tot | `2.079` | **`6.88 GB`** |
| `fast-gemma4` Tuned LoRA (`gemma4 parquet`) | Multi-page `body_text` p1-2 (`max_chars=8000`) | `kv_i8=True` | 255 | **`255/255`** | **`1,051.8`** | **`671.1`** out / **`5,443.2`** tot | **`2.33`** | — |

Key takeaways:
- **vs. Stock `mlx_vlm`:** Continuous rolling scheduling + shared `KV_sys` + `EarlyExit14` + `kv_i8` + V4 Metal kernels lifts end-to-end output throughput from **`24.4–66.9 tok/s`** to **`669.3–704.8 tok/s`** (~**10.5×** batched at `B=32–64`, ~**27×** vs `B=1` at `L_in~2015`) while cutting `B_slots=128` memory to **`6.88 GB`**.
- **vs. Prior Static Waves (`fast-gemma4` pre-V4):** On `E2B-it-MLX-4bit` (`256` docs), throughput jumped **+70.6%** (`0.966 -> 1.648 docs/s`, `505.7 -> 878.9` decode tok/s) with **`2.25 GB`** lower peak RAM (`9.13 -> 6.88 GB`).
- **Base PTQ vs QAT vs Tuned LoRA:** `E2B-it-MLX-4bit` (PTQ) and `E2B-it-qat-4bit` (QAT) share identical quantization (`4-bit g64` + `105` `8-bit` MLP overrides) and compute speed (`878.9` vs `894.6` decode tok/s). Tuned LoRA reaches higher `docs/s` (`2.08–2.33`) because the LoRA fine-tune emits `25%` fewer tokens/doc (`322` vs `428`) with `72×` fewer grammar fallbacks (`62` vs `4,448`); for zero-shot structured extraction without LoRA, `QAT-4bit` (`50.0%` exact classification / `58.8` date F1 on `48`-doc eval) strongly outperforms PTQ `MLX-4bit` (`6.2%` / `9.0`).

### Native CoreML ANE Prefix Chunk Kernel (`./build/gemma4-ane-cli`, `S=256`, `pipe=4`) & Hybrid `ABBA` Benchmarks

| Bundle / Pipeline Variant | Batch / Slots | Compute Plan (`ANE / GPU / CPU_rt`) | Best / Median | Latency / Prefill | Throughput |
| --- | ---: | --- | ---: | ---: | ---: |
| `gemma4_early14_prefix_S256_fp16.mlmodelc` | `B=16` (`iters=3`) | `1326 / 0 / 0` | `0.7063 s` | `44.144 ms/chunk` | `5,799.2 tok/s` |
| `gemma4_early14_prefix_S256_w8.mlmodelc` (Linear-W8 + `AWQ-MSE`) | `B=16` (`iters=3`) | `1326 / 0 / 0` (`zero_copy=1`) | **`0.5241 s`** | **`32.758 ms/chunk`** | **`7,814.8 tok/s`** |
| Tuned LoRA GPU Rolling (`use_ane_ring=False`, `ABBA`) | `N=96, B_slots=48` | Metal GPU | `62.043 s` | `14.882 s` prefill | `1.55 docs/s` (`84/96` valid) |
| Tuned LoRA Hybrid Linear-W8 ANE Ring (`use_ane_ring="auto"`, `ABBA`) | `N=96, B_slots=48` | ANE + Zero-Copy `KV_i8` + GPU | **`53.566 s`** (**`1.158x`**) | **`5.465 s`** (**`2.723x`**) | **`1.79 docs/s`** (`84/84` field match) |

*Note: Measured with paired `ABBA` trials under `/tmp/m4_gpu_validation.lock` on an uncontended Apple M4 Pro (48 GiB, `Mac16,8`); full suite: `44 passed` (`pytest -v`), `0` `fast-ir-guard` violations.*
