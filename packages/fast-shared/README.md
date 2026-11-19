# fast-shared

Shared Apple Silicon runtime, Metal 4 / CoreML / Accelerate primitives, benchmark hygiene utilities, and the M4/M5+ optimization plugin powering [`fast-silicon`](../fast-silicon).
Every engine package (`fast-bge-small`, `fast-embeddinggemma-2`, `fast-pplx`, `fast-gemma4`, `fast-gpt-oss`, `fast-decision-models`) reuses this layer rather than duplicating Metal kernel builders, tokenizers, HTTP/SSE routers, or Parquet pipelines.

## Quick start

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

fast-silicon-hw-check          # Check Apple Silicon baseline (Mac16,8 48 GB M4 Pro)
fast-bench-check --check-ps    # Check for active GPU/CPU contenders before benchmarking
fast-ir-guard --strict-bool-mask path/to/src path/to/tests
```

```python
from fast_shared.bench import check_system_quiescence, get_unified_memory_footprint, interleaved_abba
from fast_shared.embeddings import MappedEmbedding, truncate_and_normalize

warnings = check_system_quiescence()
mem = get_unified_memory_footprint()  # RUSAGE_INFO_V4 OS physical footprint (Metal + ANE + CPU)
```

## At a glance

| Question | Answer |
|---|---|
| What hardware is this verified on? | Apple M4 Pro `Mac16,8` (`48 GB` unified memory, `14` CPU `[10P+4E]`, `20` GPU, `16` ANE). |
| How does memory measurement work? | `get_unified_memory_footprint()` reads macOS `RUSAGE_INFO_V4` via `proc_pid_rusage` (`phys_footprint_bytes`, `lifetime_max_phys_footprint_bytes`), capturing Metal `IOSurface`, mmap, and ANE buffers that `ru_maxrss` misses. |
| What prevents silent FP32 regressions? | `fast-ir-guard` (`ir_dtype_guard.py`) statically AST-checks MLX/CoreML scripts for `int`/`bool`/`float32` SDPA masks, mixed `bf16 x fp16`, `0D` scalars, and loop `mx.eval`. |
| Can it inject faults & verify quiescence? | Yes: `check_system_quiescence()`, `interleaved_abba()`, plus CPU fault-injection contracts tested across the suite. |

## Provided Modules

| Module / Binary | Purpose | Key Exports / CLI |
|---|---|---|
| `fast_shared.hw_check` | Host hardware detection & warnings (`Mac16,8`) | `fast-silicon-hw-check [--json \| --strict]` |
| `fast_shared.bench` | Quiescence check, `RUSAGE_INFO_V4` OS footprint, `ABBA` harness, cos/dtype helpers | `fast-bench-check`, `get_unified_memory_footprint`, `peak_footprint_gb`, `interleaved_abba` |
| `fast_shared.ir_dtype_guard` | Static AST dtype/loop guard for MLX + CoreML MIL | `fast-ir-guard --strict-bool-mask` |
| `fast_shared.roofline` | Zero-GPU safetensors forensics, M4/M5 bandwidth probes | `fast-silicon-probe`, `fast-silicon-audit` |
| `fast_shared.metal_mpp` / `_bf16` | Metal 4 `MPP::matmul2d` builders (`FP16`/`BF16`, GeGLU, bias epilogues) | `HEADER_MPP`, `mpp_linear_bf16`, `mpp_fused_geglu_bf16` |
| `fast_shared.include/*` | Header-only C++/ObjC++ CoreML & Accelerate helpers | `fast_shared_apple.hpp`, `fast_shared_coreml.hpp` |
| `fast_shared.embeddings` | Safetensors memmap gather, Matryoshka slice+norm, tile slices, cache limits | `MappedEmbedding`, `truncate_and_normalize`, `tile_aligned_slices`, `set_mlx_cache_limit` |
| `fast_shared.tokenizer` | Rust `tokenizers` base + sliding character-offset chunking | `BaseFastTokenizer` |
| `fast_shared.parquet_pipeline` | Multi-file streaming/sharded/resumable Parquet + UTF-8 `page_offsets` | `embed_parquet` |
| `fast_shared.server` | Managed HTTP/SSE server + OpenAI router + coalescing micro-batcher | `ManagedHTTPServer`, `UnifiedOpenAIServer`, `ChatMicroBatcher` |
| `fast_shared.grammar` / `pipeline_llm` | Optional `[llm]` `llguidance` lazy GPU argmax masking + shard resume | `GrammarHelper` |

## Shared Parquet Controls

The shared CLI and `embed_parquet` separate source reading (`--batch-rows`), encoder calls (`--encode-batch-rows`), and shard writes (`--shard-rows`):

| CLI Option / Parameter | Controls | Default |
|---|---|---|
| `--batch-rows` / `batch_rows` | Source rows read & prepared together (`32` for whole PDFs; `4096` for short rows) | `4096` |
| `--encode-batch-rows` / `encode_batch_rows` | Max chunks sent per `encode_corpus` call | `4096` |
| `--shard-rows` / `shard_rows` | Output rows per stored shard (with `--resume`) | `65536` |
| `--max-chunk-tokens` / `max_chunk_tokens` | Final chunk limit including prefix and special tokens | `512` |
| `--overlap` / `overlap` | Repeated original body tokens between adjacent chunks | `51` |
| `--max-body-tokens` / `max_body_tokens` | Optional extra cap on original body tokens per window | `None` |

Example chunked document ingestion (UTF-8 `page_offsets` supported via `--per-ocr-page` / `--ocr-pages-only`):

```bash
embeddinggemma-2 parquet \
  --input documents.parquet --output embeddings/ --text-col body \
  --chunk --document-prefix --max-chunk-tokens 512 --chunk-overlap 51 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --keep-cols document_id --include-chunk-text
```

`--resume` assumes source inputs, model, and chunk settings did not change; use a fresh output directory when changing them.

## Sibling Engines (`48 GB` M4 Pro `Mac16,8`)

| Engine | Model | Execution Path | Verified M4 Pro Result (`monorepo_bench` + Standalone) |
|---|---|---|---|
| [`fast-bge-small`](../fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Metal 4 MPP + 16-core ANE (`hetero`) | **`207,264 tok/s`** `hetero` (`0.526 ms` ANE `B=1`, **`2.50x`** vs `transformers-mps-float16`, `1.173 GB` Metal / `4.813 GB` OS); standalone up to **`215,022 tok/s`** OCR / `0.493 ms` |
| [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | Metal 4 MPP + 16-core ANE + SME (`hetero2`) | **`31,786 tok/s`** `hetero2` (`115.2 doc/s`, `2.188 ms` ANE `B=1` = **`4.36x`** GPU, `2.224 GB` Metal / `10.820 GB` OS); standalone up to **`36,911 tok/s`** synth / `1.871 ms` |
| [`fast-pplx`](../fast-pplx) | `pplx-embed-v2-context-0.6b` (`0.6B`) | Metal MLX GDN+SDPA + Staged ANE (`hetero2`) | **`15,244.9 tok/s`** `hetero2` (`B=8 S=256`, **`5.50x`** vs `torch-mps` / **`10.06x`** vs `torch-cpu`, `4.46 ms` ANE `B=1`, `S=4K` `2.5` vs `12.5 GB`; `1.621 GB` Metal / `2.591 GB` OS, `16/16` parity) |
| [`fast-gemma4`](../fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` act.) | Rolling `B_slots=128` (`kv_i8`) + Early14 ANE ring | **`8.92 docs/s`** Concurrent ANE+GPU `2xB=128` (`1,192 dec tok/s`, **`2.54x`** vs `stock-mlx_vlm B=64`, `-3.50 GB` peak, `6.718 GB` Metal / `18.314 GB` OS); standalone **`671.1–704.8` e2e out tok/s** (`6.88 GB`) |
| [`fast-gpt-oss`](../fast-gpt-oss) | `openai/gpt-oss-20b` (`MXFP4` MoE) | Unified Hot-MPP + Cold-QMV + Hydragen v2 | **`555.5 dec tok/s`** (`1.874 docs/s` @ `B=128`, **`4.78x`** vs `stock-mlx_lm B=8`, `14.81 GB` peak / `15.901 GB` suite Metal / `21.709 GB` OS); `B=1` `74.9`–`97.4 tok/s` |
| [`fast-decision-models`](../fast-decision-models) | `Laya-Typed-421M` + `Kev-0.8B` | Pure ANE `[1585,0,0]` (`P=4`) & `W8` MLX | `kev_08b` **`50.95 q/s`** (**`5.27x`** vs `transformers-mps-bf16`, `-2.13 GB` OS footprint) & `laya_ane` **`40.53 q/s`** (**`4.19x`**, `1.748 GB` Metal / `3.166 GB` OS); standalone **`59.68`** / **`41.72 q/s`** |

## Agent Plugin & Tests

`fast-shared` exposes [`skills/apple-silicon-ml-optimizer/`](skills/apple-silicon-ml-optimizer/SKILL.md) as a Claude Code / Codex plugin:

```bash
claude --plugin-dir /path/to/fast-shared
.venv/bin/pytest -v
```

**Scope & not tested:** Verified on Apple M4 Pro (`Mac16,8`, `48 GB`, `14 CPU`, `20 GPU`, `16 ANE`, macOS 26.6). Other Apple Silicon chips (`M1`–`M3`, `M4 Max`, `M5`), Linux/Windows hosts, and concurrent unserialized Metal/ANE runs are untested.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Inspired the macOS `proc_pid_rusage(..., RUSAGE_INFO_V4)` OS physical footprint utilities (`fast_shared.bench`), zero-GPU `.safetensors` memmap embedding loader (`MappedEmbedding`), `tile_aligned_slices` final-slice boundary merge helper, MLX buffer cache limit discipline (`set_mlx_cache_limit`), and scannable README structure.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
