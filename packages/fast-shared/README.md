# fast-shared

Shared DRY infrastructure and cross-generation M4/M5+ optimization plugin for Apple Silicon hardware-limit inference engines (`fast-bge-small`, `fast-embeddinggemma-2`, `fast-gemma4`, `fast-gpt-oss`, and `fast-decision-models`).

## Provides

- `fast_shared.include/fast_shared_apple.hpp`: Header-only C++ / Objective-C++ CoreML & Accelerate helpers (`fs_now_sec`, `fs_l2_normalize_f32`, `fs_resolve_compute_units`).
- `fast_shared.include/fast_shared_coreml.hpp`: Header-only Objective-C++ CoreML `MLMultiArray` float16 allocation and dense/strided copy helpers (`is_dense_4d`, `make_f16_array`, `copy_to_mla_u16`, `copy_from_mla_u16`).
- `fast_shared.metal_mpp`: Shared Metal 4 `MPP::matmul2d` FP16 kernel builder (`none`, `bias`, `bias_res`, `bias_gelu`, `geglu`, `gelu_mul_pli`) and Metal standard headers.
- `fast_shared.metal_mpp_bf16`: Shared Metal 4 `MPP::matmul2d` BF16 linear (`mpp_linear_bf16`), kernel builder (`make_mpp_bf16_geglu_kernel`), fused Gate+Up+GeGLU (`mpp_fused_geglu_bf16`), and fused GeGLU split (`fused_geglu_split`) kernels.
- `fast_shared.roofline` (`fast-silicon-probe`, `fast-silicon-audit`): Zero-GPU `.safetensors` quantization/layout forensics, `B=1..256` & prefill roofline modeling, empirical Metal `float4` DRAM bandwidth ceiling probes, and `BF16`/`FP16`/`Q8`/`Q4`/`MPP` GEMM sweeps across M4 (`Apple9`) and M5+ (`Apple10+` / `NAX`).
- `fast_shared.bench` (`fast-bench-check`): Host GPU/CPU contention checks (`ps`), `ABBA` thermally interleaved benchmark harness, MLX cache arena cleanup, cosine parity helpers, and `FP32` silent-upcast dtype guards.
- `fast_shared.hw_check` (`fast-silicon-hw-check`): Fast MacBook / Apple Silicon detector (`hw.model`, chip, unified RAM, CPU P/E cores, GPU cores, ANE cores) that warns when running outside the tested `48 GB` Apple M4 Pro (`Mac16,8`: `14` CPU `[10P + 4E]`, `20` GPU, `16` ANE) baseline.
- `fast_shared.grammar` & `fast_shared.pipeline_llm` *(optional `[llm]` extra)*: Shared `GrammarHelper` (`llguidance` strict schema compiler + lazy GPU argmax masking), schema validation, JSONL/Parquet LLM extraction loop, and shard resume helpers.
- `fast_shared.tokenizer`: Rust-backed `tokenizers` wrapper (`BaseFastTokenizer`) with sliding-window character-offset chunking and named prompt presets.
- `fast_shared.embeddings`: Shared Matryoshka dimension validation (`resolve_embedding_dim`) and slice + L2 re-normalization (`truncate_and_normalize`).
- `fast_shared.parquet_pipeline`: Multi-file streaming/sharded/resumable Parquet embedding pipeline (`float32` / `float16`, optional Matryoshka `dimensions`, UTF-8 `page_offsets` OCR slicing, sliding-window chunking).
- `fast_shared.server`: Shared HTTP/JSON/SSE server lifecycle (`ManagedHTTPServer`, `JsonHTTPHandler`), OpenAI embedding/chat/completions routers (`UnifiedOpenAIServer`, `ChatMicroBatcher`, `OpenAIChatAdapter`).
- `fast_shared.cli_common`, `fast_shared.paths`, `fast_shared.hf`: Shared CLI parser/runner scaffolding, repo root resolution, and HuggingFace snapshot helpers.

## Parquet embedding controls

The shared CLI and `embed_parquet` API separate preparation, model calls, and
storage:

| CLI option / API parameter | Controls | Default |
| --- | --- | --- |
| `--batch-rows` / `batch_rows` | Source rows read and prepared together | `4096` |
| `--encode-batch-rows` / `encode_batch_rows` | Maximum output chunks per `encode_corpus` call, after chunk expansion | `4096` |
| `--shard-rows` / `shard_rows` | Output rows per stored directory shard | `65536` |
| `--max-chunk-tokens` / `max_chunk_tokens` | Final chunk length, including prompt and special tokens | `512` |
| `--overlap` (`--chunk-overlap`) / `overlap` | Shared original body tokens between windows | `51` |
| `--max-body-tokens` / `max_body_tokens` | Optional additional limit on original body tokens per window | `None`; engine CLIs can supply a default |

For example, the EmbeddingGemma CLI can chunk complete documents while retaining
their identifiers, exact chunk text, and source character ranges:

```bash
embeddinggemma-2 parquet \
  --input documents.parquet --output embeddings/ --text-col body \
  --chunk --document-prefix --max-chunk-tokens 512 --chunk-overlap 51 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --keep-cols document_id --include-chunk-text
```

Use a small source batch, such as `32`, for whole long PDFs; `4096` is a starting
point for rows that already contain short passages. CPU preparation and Parquet
writes already overlap encoding through bounded queues. The encode limit caps
individual model calls in both file and directory output modes; it does not bound
total pipeline memory. Large source records, their expanded chunks, and pending
output shards still occupy memory. Tune these controls for the actual input.

Chunking tokenizes the original body to choose windows, then encodes each exact
source slice together with its prefix and the tokenizer's native special tokens.
That preserves the normal prompt tokenization at boundaries. Retokenization can
shorten a window to fit the final limit. Unicode characters remain whole, and
ranges retain leading/trailing whitespace and line breaks. Overlap counts original
body tokens; character boundaries can extend the source text shared by adjacent
windows. `max_body_tokens` is capped by the available final chunk space. Invalid
limits and overlap that prevents forward progress raise `ValueError` instead of
silently changing the requested stride. Chunk length and non-chunked `max_length`
must fit the engine's sequence limit. The default `512` keeps chunks within the
engines' optimized short-sequence buckets; larger limits are available when the
engine supports them.

For `--per-ocr-page`, character ranges refer to the original decoded page,
including its whitespace. `--ocr-pages-only` uses the existing joined, trimmed
selection of OCR pages, so its ranges refer to that selected text. Existing
directory-shard resume behavior and atomic Parquet writes remain in place.
`--resume` assumes the source, model and chunk settings are unchanged; use a
fresh output directory after changing them.

## Agent Plugin (`skills/apple-silicon-ml-optimizer`)

`fast-shared` doubles as a Claude Code (`.claude-plugin/plugin.json`), Codex (`.codex-plugin/plugin.json`), and Agent Skills plugin bundling [`skills/apple-silicon-ml-optimizer/`](skills/apple-silicon-ml-optimizer/SKILL.md) (references for M4 vs M5 architecture, Metal 4 / CoreML / MLX kernels, verified dead-ends registry, and Metal/ObjC++/Python probe scripts backed directly by `fast_shared.roofline` and `fast_shared.bench`).

```bash
claude --plugin-dir /path/to/fast-shared
```

## Sibling Engines (Apple M4 Pro Benchmarks)

| Repo | Model | Execution Engines | Verified Throughput / Latency |
| --- | --- | --- | --- |
| [`fast-bge-small`](../fast-bge-small) | `BAAI/bge-small-en-v1.5` (`33M`) | Metal 4 MPP GPU + 16-core ANE (`hetero`) | Up to **`215,022 tok/s`** OCR / **`189,338 tok/s`** synthetic (`0.49 ms` ANE `B=1`) |
| [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) | `google/embeddinggemma-2` (`300M`) | Metal 4 MPP GPU + 16-core ANE + CPU SME | Up to **`36,911 tok/s`** synthetic / **`32,569 tok/s`** OCR (`1.87 ms` ANE `B=1`) |
| [`fast-gemma4`](../fast-gemma4) | `google/gemma-4-E2B-it` (`~2B` active) | Rolling `B_slots=128` Metal 4 MPP GPU (`kv_i8`) + Early14 ANE + `llguidance` + PromptLookup | **`671.1–704.8` e2e out tok/s** (`878.9–1,051.8` decode tok/s, **`2.33 docs/s`** real corpus, `6.88 GB` vs `24.4–66.9` out tok/s stock `mlx_vlm`) |
| [`fast-gpt-oss`](../fast-gpt-oss) | `openai/gpt-oss-20b` (`20.9B`/`3.6B` act. `MXFP4` MoE) | Unified Hot-MPP + Cold-QMV MoE, Hydragen v2 + SWA-128, `131K` pruned `lm_head`, `llguidance` | **`462.8 dec tok/s`** (`0.710 docs/s`, `128/128` valid @ `16.56 GB`), isolated **`97.4 t/s`** `B=1` |
| [`fast-decision-models`](../fast-decision-models) | `Laya-Typed-421M` W8A16 + `Kev-0.8B` (`w8`) | Pure 16-core ANE (`P=4` `IOSurface` + NEON transpose) & Metal MLX left-padded judge | `laya_ane`: **`41.72 q/s`** (`23.97 ms/q`, `[1585,0,0]` ANE ops); `kev_08b` (`w8`): **`59.68 q/s`** (`16.76 ms/q`) |

## Tests

```bash
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -v
```
