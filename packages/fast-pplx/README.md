# fast-pplx

Hardware-limit Apple Silicon inference engine for `perplexity-ai/pplx-embed-v2-context-0.6b` — a 12-layer hybrid Qwen3.5 contextual embedding model combining alternating linear GatedDeltaNet layers and bidirectional full self-attention with a 1024-to-1024 contextual projection head.

## Architecture Highlights

- **Pure-Rust Tokenizer (`fast_pplx.tokenizer`)**: Zero `torch`/`transformers` runtime dependency. Matches upstream `split_special_tokens=True` semantics (`[Q] ` uses reserved token `248077`, `[D] ` tokenizes as literal ordinary tokens so arbitrary text cannot inject control tokens, and `<|chunk_sep|>` `248079` marks inter-chunk boundaries with exact character-to-token span alignment).
- **Hybrid Metal GPU Engine (`fast_pplx.gpu`)**:
  - Custom MIT-adapted `gated_delta_update` Metal kernel running `mx.float16` `q, k, v` tensors end-to-end with zero per-layer `mx.eval` barriers on sequences `<= 2048`.
  - Fused `qkv_proj` (`q+gate, k, v`) and `gate_up_proj`, bidirectional `scaled_dot_product_attention` with `float16`/`bfloat16` additive padding masks (`full_attention_interval=2`), and `+1.0` float32 zero-centered `RMSNorm` shifts (leaving `GatedRMSNorm` unshifted and computed in `float32`).
  - Automatic streamed state-carried GatedDeltaNet (`1024`-token slices) + tile-aligned `SDPA` (`512`-token slices) up to `32K` tokens with configurable `set_mlx_cache_limit`.
  - `MappedEmbedding` memmaps the `~1.02 GB` `embed_tokens` table out of Metal RAM while keeping `use_mapped_embedding=False` optional.
- **CLI, Server & Parquet Pipeline**:
  - `fast-pplx setup-weights` / `setup-ane` / `embed` / `bench` / `parquet` / `serve`
  - OpenAI-compatible HTTP server (`fast_pplx.server`) and sharded Parquet pipeline (`fast_pplx.parquet_pipeline`).

## Quick Usage

```python
from fast_pplx import FastPplx

with FastPplx() as model:
    q = model.encode_queries(["How do ocean currents move heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ])
```

## Verification

```bash
fast-ir-guard --strict-bool-mask packages/fast-pplx/src packages/fast-pplx/tests
pytest packages/fast-pplx/tests/test_fast_pplx_cpu.py packages/fast-pplx/tests/test_fault_injection_cpu.py
```
