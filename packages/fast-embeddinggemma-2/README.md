# fast-embeddinggemma-2

Standalone Apple Silicon (M4 Pro **20-core Metal GPU + 16-core Apple Neural Engine + 14-core ARMv9 CPU-SME**) inference engine for `google/embeddinggemma-2` (`768`-dim, `2048`-context, zero `transformers`/`torch`/`mlx_vlm` runtime dependencies).

Built as a sister repository to [`fast-bge-small`](../fast-bge-small), sharing Apple Silicon Metal MPP generators, C++ helpers, Rust tokenizer base, Parquet pipeline, and HTTP server via [`fast-shared`](../fast-shared).

## Architecture

1. **GPU Engine (`src/fast_embeddinggemma_2/gpu.py`)**
   - Uses `fast_shared.metal_mpp` **Metal 4 `MPP::matmul2d`** kernels (`tm=128/64`), **fused GeGLU** (`_MPP_GEGLU_64`), and **fused PLE GELU x PLI** (`_MPP_PG_64`).
   - Fused **QKNorm + VNorm + RoPE** Metal kernels (`_QKNORM_ROPE_{256,512}_BHLD`) writing directly in `[B, H, L, D]` layout for MLX `fast.scaled_dot_product_attention` across alternating sliding-window (`hd=256, n_kv=2, theta=1e4`) and full-attention (`hd=512, n_kv=1, theta=1e6`) layers.
   - Cost-aware multi-bucket sequence packing (`fast_pack_cost_aware`) + `ceil128` long-doc handling (`<= 2048` tokens).

2. **ANE Engine (`native/emb2_coreml_runner.mm` + `native/emb2_fast_io.cpp`)**
   - `INT8` linear-quantized (`W8A16`) `conv1x1` CoreML packed sequence bundles (`emb2_pk_s{32,64,128,192,256,320,384,448,512}_w8.mlmodelc`, `pipe_depth=4`).
   - Single-query `<32` tok bucket (`emb2_pk_s32_w8.mlmodelc`) runs at ~`1.88 ms` (`cos >= 0.9999` vs GPU).
   - NEON/L1-blocked token gather + block-diagonal attention mask + Accelerate `cblas_sgemm` projection (`512 -> 768`).

3. **CPU-SME Engine (`emb2_cpu_s{256,384,512}_mix_w8.mlmodelc`)**
   - Mixed-precision `FP16` matmul / `FP32` normalization & softmax CoreML graphs executing concurrently on ARMv9 SME/AMX alongside GPU and ANE (`mode="hetero"`).

## Build & Install

```bash
cd fast-embeddinggemma-2
make native
uv venv --python cpython-3.13.14 .venv
uv pip install --python .venv/bin/python -e ../fast-shared -e ".[dev]"
```

If setting up weights / CoreML models from scratch:

```bash
.venv/bin/embeddinggemma-2 setup-weights
# Rebuilding CoreML bundles requires coremltools>=8.0 (`uv pip install -e ".[ane-build]"`)
python scripts/build_ane_models.py --mode all
```

## CLI & Python Usage

```bash
.venv/bin/embeddinggemma-2 verify
.venv/bin/embeddinggemma-2 bench
.venv/bin/embeddinggemma-2 embed --query --query-prefix "quarterly semiconductor capital expenditures"
# Matryoshka truncation (768 -> 256 or 128 dims, re-normalized L2; 3x-6x smaller vectors)
.venv/bin/embeddinggemma-2 embed --dimensions 256 --query-prefix "quarterly semiconductor capital expenditures"
.venv/bin/embeddinggemma-2 parquet --input docs.parquet --output out_shards/ --dimensions 256 --dtype float16 --document-prefix --chunk
.venv/bin/embeddinggemma-2 serve --host 127.0.0.1 --port 8080 --dimensions 256
./build/emb2-ane-cli --ids "2,3250,236787,7293,1109,1816,236787,1" --seq-len 32 --json
```

Matryoshka dimension selection (`128`, `256`, `512`, `768`, or any `1 <= d <= 768`) is supported in Python (`FastEmbeddingGemma2(dimensions=256)` or per-call `dimensions=...` on `encode_query` / `encode_texts` / `encode_corpus`), via `--dimensions`, and per-request via OpenAI `/v1/embeddings` `"dimensions"` (all sliced and L2 re-normalized).

### Bulk OCR text with overlapping chunks

Use the native Parquet command to read, tokenize, embed and write the corpus. It
prepares input on a background thread and writes completed vectors on another,
while `hetero2` uses the GPU and Neural Engine. Existing source IDs and metadata
travel with each emitted chunk.

For whole document bodies, use a modest source batch and enable chunking:

```bash
.venv/bin/embeddinggemma-2 parquet \
  --input ocr-text.parquet --output embeddings/ --text-col body \
  --mode hetero2 --dimensions 512 --document-prefix \
  --chunk --max-chunk-tokens 512 --overlap 64 \
  --batch-rows 32 --encode-batch-rows 4096 --shard-rows 65536 \
  --drop-text-col --include-chunk-text
```

`--max-chunk-tokens` includes the document prefix and special tokens. Keeping it
at 512 allows the existing Neural Engine buckets to participate; larger chunks
can use the GPU up to the model's 2,048-token context. `--overlap` (also
`--chunk-overlap`) repeats body tokens between windows, preserving context at
boundaries. More overlap increases the vector count and encoding work. The
optional `--max-body-tokens` imposes an additional body-token limit.

Chunk output includes `chunk_index`, `char_start`, `char_end`, `token_count` and,
with `--include-chunk-text`, the exact source substring. Offsets are character
positions, not UTF-8 byte positions. Dropping the original body avoids copying
the whole document into every chunk; the source ID still links back to it.
When OCR page controls are used, offsets refer to the selected text unit.

The three size controls serve different purposes:

| Option | Controls |
| --- | --- |
| `--batch-rows` | Source rows read and prepared at a time; use smaller batches for long documents. |
| `--encode-batch-rows` | Maximum emitted chunks in one encoder call, after chunk expansion. |
| `--shard-rows` | Output rows per stored Parquet shard. |

Encoding batches stay capped for both file and directory output. Large storage
shards therefore do not force large model calls. Preparation still holds source
records and their expanded chunks, so the encoder limit is not a byte limit for
the entire pipeline.

For text already split into the desired chunks, leave `--chunk` off:

```bash
.venv/bin/embeddinggemma-2 parquet \
  --input passages.parquet --output embeddings/ --text-col body \
  --mode hetero2 --dimensions 512 --document-prefix \
  --batch-rows 4096 --encode-batch-rows 4096 --shard-rows 65536
```

Use chunking for long whole-document bodies; the unchunked path applies the
model's context limit. `--dtype float16` is available to halve stored vector
bytes; inference and normalization remain unchanged.
`--resume` assumes unchanged source, model and chunk settings. Use a fresh output
directory after changing them.

## Measured Verification & Benchmarks (M4 Pro)

- `make native`: builds `build/libemb2_coreml_runner.dylib`, `build/libemb2_fast_io.dylib`, and standalone `build/emb2-ane-cli`.
- Standalone ANE binary (`./build/emb2-ane-cli --ids "2,3250,236787,7293,1109,1816,236787,1" --seq-len 32 --bench 30 --json`): `tokens=8 bucket=32 iters=30 latency_ms=1.8713` (zero Python runtime).
- `embeddinggemma-2 verify` on 64-doc reference fixture:
  - `GPU_vs_F16` min cosine: `0.999997` (mean `0.999999`)
  - `GPU_vs_BF16` min cosine: `0.999891`
  - `Hetero_vs_F16` min cosine: `0.999705`
  - `Single-query B=1 S=24 ANE_vs_GPU` cosine: `0.999890`
- `pytest -v`: `32 passed in 26.93s` covering 64-doc multi-mode parity, ANE bucket routing (`32, 128, 256, 384, 512`) and edge cases, fused residual+RMSNorm CPU parity, tokenizer & chunker, native `emb2-ane-cli`, UTF-8 `page_offsets` OCR slicing, streaming/sharded/resumable Parquet pipeline, and HTTP server smoke test.

### Throughput & Query Latency (`embeddinggemma-2 bench`, measured during potential background session activity)

Single-query latency:
- Synthetic run: `ANE_B1_S32` median `2.095 ms` (min `1.899 ms`) vs `GPU_B1_S64` median `9.026 ms` (min `8.860 ms`).
- Document OCR run: `ANE_B1_S32` median `2.066 ms` (min `1.868 ms`) vs `GPU_B1_S64` median `8.895 ms` (min `8.737 ms`).

Corpus throughput (`n = 2,000` documents):

| Workload | Mode | Elapsed (s) | Throughput (tok/s) | Throughput (doc/s) | Split (`gpu_docs` / `ane_docs` / `cpu_docs`, tokens) |
| --- | --- | ---: | ---: | ---: | --- |
| Synthetic (`551,889` toks) | `gpu` | 33.376 | 16,535 | 59.9 | 2,000 / 0 / 0 (`551,889` / `0` / `0`) |
| Synthetic (`551,889` toks) | `ane` | 27.406 | 20,138 | 73.0 | 0 / 2,000 / 0 (`0` / `551,889` / `0`) |
| Synthetic (`551,889` toks) | `hetero2` (GPU+ANE) | **14.952** | **36,911** | **133.8** | 910 / 1,090 / 0 (`256,316` / `295,573` / `0`) |
| Synthetic (`551,889` toks) | `hetero` (GPU+ANE+CPU) | 17.639 | 31,288 | 113.4 | 1,032 / 715 / 253 (`286,755` / `197,377` / `67,757`) |
| Document OCR (`559,737` toks) | `gpu` | 47.257 | 11,844 | 42.3 | 2,000 / 0 / 0 (`559,737` / `0` / `0`) |
| Document OCR (`559,737` toks) | `ane` | 27.136 | 20,627 | 73.7 | 64 / 1,936 / 0 (`37,609` / `522,128` / `0`) |
| Document OCR (`559,737` toks) | `hetero2` (GPU+ANE) | **17.186** | **32,569** | **116.4** | 730 / 1,270 / 0 (`212,621` / `347,116` / `0`) |
| Document OCR (`559,737` toks) | `hetero` (GPU+ANE+CPU) | 19.523 | 28,671 | 102.4 | 957 / 805 / 238 (`271,748` / `222,921` / `65,068`) |

- Compared with [`fast-bge-small`](../fast-bge-small) (`33M` params, `384`-d, up to `215,022 tok/s` OCR / `189,338 tok/s` synthetic), EmbeddingGemma-2 (`300M` params, `768`-d, 24 layers vs 12 layers / ~`9.1x` parameters) reaches **`36,911 tok/s`** synthetic and **`32,569 tok/s`** OCR in `hetero2` (`GPU+ANE`, ~`5.1x–6.6x` lower tok/s than `bge-small` while hitting **`3.1x` GPU-only** on OCR) and **`1.871 ms`** single-query ANE latency.

*Note: Measured on Apple M4 Pro (48 GiB) while background sessions were potentially sharing GPU/CPU resources; uncontended throughput may be higher.*

