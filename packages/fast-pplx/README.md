# fast-pplx

Use `fast-pplx` to find a passage whose meaning depends on the surrounding text. On 175 legal questions, encoding each passage with its neighbors raised the score for ranking relevant passages in the first ten results from **0.359 to 0.703**. Stock contextual MLX also scored **0.703**. On the full 325-query base M5 workload, the project finished in **136.5 seconds** and stock took **601.5 seconds**. Both returned usable results for all 325 queries. [Scored retrieval runs](../../docs/benchmarks/document-workloads.json)

`fast-pplx` runs Perplexity’s `pplx-embed-v2-context-0.6b` model locally. Its mixed mode sends short inputs to the Apple Neural Engine (ANE) and long inputs to the GPU. The M4 Pro GPU processed a 32,768-token input with a measured process footprint of 2.69 GB. The package supports Python, OpenAI-compatible `/v1/embeddings`, contextual Parquet processing and a native ANE command-line tool. [Measurements](../../docs/BENCHMARKS.md)

> [!TIP]
> **Full Cross-Silicon & CoreAI Reference:** See [**`../../docs/BENCHMARKS.md`**](../../docs/BENCHMARKS.md) for complete Category 1–4 outcome tables, 4-bucket `RUSAGE_INFO_V6` memory footprints, and 22 audited CoreAI decisions, including rejected configurations.

## Quick start

```bash
make -C packages/fast-pplx native
fast-pplx setup-weights --local-files-only
# Optional: compile 100% Pure-ANE CoreML .mlmodelc stages (`w8a16` default; `--ane-profile fast|balanced|parity`)
fast-pplx setup-ane --precision w8a16
```

```python
from fast_pplx import FastPplx

with FastPplx(mode="auto", dimensions=512) as model:  # auto picks hetero2 if ANE stages exist, else gpu
    q = model.encode_queries(["How do ocean currents transport heat?"])
    docs = model.encode_contextual_chunks([
        ["Chunk A discussing heat transport.", "Chunk B expanding on polar gyres."],
    ], max_length=512)

    # For long 4K-32K documents, build a contextual multi-vector MaxSim index (1024d INT8):
    long_index = model.build_index(
        ["Long 32K document text...", "Second long document..."],
        multivector=True,
    )
    top_doc_ids, top_scores = long_index.search(model.encode_queries(["ocean currents"])[0], top_k=5)
```

`max_length` caps the whole document, including its prefix and chunk separators. The result contains one matrix per input document and one row per original chunk, in input order. A chunk crossing the cap uses the tokens that fit. A chunk wholly beyond it gets a zero vector. To index every passage in a longer document, split it into context windows before encoding.

```bash
./packages/fast-pplx/build/pplx-ane-cli --seq-len 128 --inspect --bench 20
```

## At a glance

| Question | Answer |
| :--- | :--- |
| What model & tokenizer behavior does it preserve? | `pplx-embed-v2-context-0.6b` (`12` layers, `full_attention_interval=2`, `+1.0` zero-centered RMSNorm shifts), matching upstream `split_special_tokens=True` (`[Q] ` = `248077`, `[D] ` = literal ordinary tokens, `<\|chunk_sep\|>` = `248079`). |
| Why does `S > 2048` stay flat in memory? | `MappedEmbedding` keeps `embed_tokens` (~`1.02 GB`) out of Metal RAM; `S > 2048` automatically switches to streamed state-carried `GatedDeltaNet` (`1024`-tok slices) + tile-aligned `SDPA` (`512`-tok slices) — holding `32,768` tokens in **`2.690 GB`** OS footprint. |
| How does `hetero2` combine ANE & GPU? | Short items (`<= 512` toks) pack chunk-aligned into `S{64,128,256,384,512}` Pure-ANE CoreML rings (`pipe_depth=4`, `0` CPU/GPU ops), while Metal processes `> 512` sequences concurrently and steals from tail on idle. |
| Which ANE profile should I choose? | Default **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`) closes **`73%`** of the ANE-vs-GPU cosine gap (`0.9866 -> 0.9964` corpus; `0.9909 -> 0.9988` `S=256`, `int8 MAE` `2.02 -> 0.79`) for only `4.5%–5.8%` latency cost; use **`fast`** (`--ane-profile fast`) for peak throughput (`13.5K tok/s` at `S=128`). |

## Benchmark Highlights (Base M5 `24 GB` vs. M4 Pro `48 GB`)

### 1. Tier-1 Retrieval Outcomes (`NanoBEIR-en` + `LegalBench-RAG` `4K` Contextual, `N=325`)

| Engine / Baseline | Base M5 `S_obj,e2e` | Base M5 `EOT (u/s)` (Wall / Mem) | M4 Pro `S_obj,e2e` | M4 Pro `EOT (u/s)` (Wall / Mem) | vs. Stock `MLX` (`M5` / `M4 Pro`) | M5 vs. M4 Pro |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| **`fast-pplx` (`hetero2`)** | **`0.7244`** | **`1.724`** (`136.5s`, `2.34 GB`) | **`0.7241`** | **`1.407`** (`167.3s`, `1.97 GB`) | **`4.41×`** / **`4.33×`** | **`1.23×`** |
| Stock `pplx-context-mlx` (`fp16`) | `0.7236` | `0.391` (`601.5s`, `2.12 GB`) | `0.7236` | `0.325` (`723.1s`, `2.19 GB`) | `1.00×` | `1.20×` |

The original M5 run had usable results for 150 of 325 queries. The selected rerun split oversized documents into windows and covered all 325. Its mean score exceeded stock MLX by 0.000831. The paired 95% interval ran from −0.003619 to +0.005281, leaving equivalence within the runner’s default 0.003 score-loss tolerance unresolved. The reported 4.41× compares the score units earned per second on these two runs. [Per-task results and run records](../../docs/benchmarks/document-workloads.json)

Batching accounts for part of the whole-job difference. On 384 sampled passages on M4 Pro, sorting by length reduced stock MLX’s total encoding time across the four sample sets from **20.65 to 13.34 seconds** (1.55×, using each set’s median). All 24 labeled NanoBEIR queries kept the same ten passages in the same order. The long legal windows behaved differently. Stock originally processed them one at a time. On eight complete windows that took **4.39 seconds**. Processing two at a time took **4.71 seconds** in source order and **4.36 seconds** in length order. [Matched stock runs and full-corpus padding counts](../../docs/benchmarks/pplx-padding-20261121.json)

### 2. Cross-Silicon Throughput, Latency & `32K` Scaling

| Workload | M4 Pro (`48 GB`) `fast-pplx` | Base M5 (`24 GB`, `NAX`) `fast-pplx` | vs. Stock Baselines & Notes |
| :--- | ---: | ---: | :--- |
| **`query_b1` (`S=15`)** | **`4.18 ms`** (`ane`, `0.248 GB` OS) | **`5.52 ms`** (`ane`, `0.245 GB`) / `5.57 ms` (`gpu`) | **`2.16×`** vs `pplx-context-mlx-f16` (`9.04 ms`), **`6.87×`** vs `torch-mps`, **`190.9×`** vs `torch-cpu` |
| **`doc_s256` (`B=8`)** | **`14,884 tok/s`** (`ane`) / `14,621` (`hetero2`) | **`15,235 tok/s`** (`hetero2`, `16.80 ms`) | **`1.68×`** (M4 Pro) / **`1.32×`** (M5) vs `pplx-context-mlx-f16`; `4.90×` vs `torch-mps` |
| **`doc_s512` (`B=8`)** | **`13,259 tok/s`** (`hetero2`, `38.62 ms`) | — | **`1.61×`** vs `pplx-context-mlx-f16` (`8,243`), `4.59×` vs `torch-mps`; `0.273 GB` OS (`ane`) |
| **`doc_s4096` (`B=4`)** | **`7,531 tok/s`** (`hetero2`, `3.06 GB` OS) | **`9,697 tok/s`** (`gpu-no-mmap`) / **`9,637`** (`hetero2`) | **`1.26×`** vs `pplx-context-mlx-f16`, **`4.16×`** smaller OS than `torch-mps` (`12.70 GB`); M5 `NAX` **`+28–35%`** |
| **`mixed` (`24×~128 + 4×2048`)** | **`12,624 tok/s`** (`hetero2`, `961.8 ms`) | **`16,796 tok/s`** (`hetero2`, `722.9 ms`) | **`1.72×`** vs `pplx-context-mlx-f16`, **`1.49×`** vs `gpu-bf16`; Base M5 **`1.33×`** M4 Pro |
| **`32K` Context (`context-limit`)** | **`2,720 tok/s`** (`12.05s`, **`2.690 GB` OS**) | Verified (`<= 19.2 GB` cap) | Streamed `1024`-tok GDN + `512`-tok SDPA (`1.782 GB` Metal peak) |
| **Parity, Windows & Determinism** | `16/16` GPU `min_cos=0.999996`, `19/19` bit-exact (`0.0` err) | `0` failures, `0` skips (`make test-hw`) | Contextual `--context-tokens 512..4096` lifts `nDCG@5` **`+0.1875`** (`0.8125 -> 1.0000`) |

### 3. ANE Profiles (`fast` / `balanced` / `parity`) Tradeoffs

Measured via `packages/fast-pplx/scripts/benchmark.py ane-profiles` in isolated temp homes against stock `zainmerchan/pplx-embed-v2-context-0.6b-mlx` (`fp16`):

| Profile | `S=128` Base M5 (`tok/s`) | `S=128` M4 Pro (`tok/s`) | `S=256` M4 Pro (`tok/s`) | `S=256` `f32 min_cos` vs. GPU | `S=256` `int8 MAE` vs. GPU | Speed vs. `fast` |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **`fast`** | **`13,542`** (`1.16×` stock) | **`13,464`** (`1.54×` stock) | **`10,311`** (`1.22×` stock) | `0.9909` *(corpus `0.9866`)* | `2.02` | `1.00×` |
| **`balanced`** (`MaxAbsRMSNorm` + `TanhSiLU`, default) | `12,754` | `12,856` | `9,762` | **`0.9988`** *(corpus **`0.9964`**)* | **`0.79`** | `0.94–0.95×` |
| **`parity`** | `12,679` | `12,496` | `9,519` | **`0.9989`** | **`0.72`** | `0.92–0.94×` |
| Stock `pplx-context-mlx` (`fp16`) | `11,707` | `8,735` | `8,425` | `1.0000` (GPU) | — | — |

**Paired Tier-1 Retrieval Outcomes (`N=325` queries, identical dataset lock & isolated profile homes):**

| Host | Arm | `S_obj,e2e` | `EOT (u/s)` | Wall (`s`) | `NanoBEIR` / `LB-Ctx` (`lift`) | Paired vs. `gpu` (`Δ` / `95% CI` / `EOT×`) | Mem (`OS phys` / `comb` / `peak`) |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- | :---: |
| **M4 Pro** (`48 GB`) | `gpu` | `0.7392` | `0.878` | `273.73s` | `0.7483` / `0.7314` (`+0.3786`) | baseline | `1.48` / `1.48` / `3.09 GB` |
| **M4 Pro** (`48 GB`) | **`hetero2 fast`** | **`0.7394`** | `1.512` | `158.91s` | `0.7479` / `0.7320` (`+0.3737`) | `+0.0002` `[-0.0050, +0.0054]` (`1.72×`) | `1.13` / `1.36` / `3.24 GB` |
| **M4 Pro** (`48 GB`) | **`hetero2 balanced`** | `0.7376` | **`1.538`** | **`155.83s`** | `0.7452` / `0.7311` (`+0.3803`) | `-0.0016` `[-0.0036, +0.0004]` (`1.75×`) | `1.48` / `1.82` / `3.59 GB` |
| **M4 Pro** (`48 GB`) | `hetero2 parity` | `0.7369` | `1.497` | `160.02s` | `0.7426` / `0.7320` (`+0.3815`) | `-0.0023` `[-0.0052, +0.0005]` (`1.71×`) | `1.61` / `1.95` / `3.36 GB` |
| **Base M5** (`24 GB`) | `gpu` | `0.7390` | `1.614` | `148.79s` | `0.7478` / `0.7314` (`+0.3787`) | baseline | `1.06` / `1.06` / `2.93 GB` |
| **Base M5** (`24 GB`) | **`hetero2 fast`** | **`0.7381`** | **`2.163`** | **`110.93s`** | `0.7460` / `0.7313` (`+0.3734`) | `-0.0009` `[-0.0052, +0.0034]` (`1.34×`) | `1.54` / `1.88` / `3.05 GB` |
| **Base M5** (`24 GB`) | **`hetero2 balanced`** | `0.7379` | `2.116` | `113.36s` | `0.7450` / `0.7318` (`+0.3811`) | `-0.0011` `[-0.0038, +0.0017]` (`1.31×`) | `1.58` / `1.93` / `3.09 GB` |
| **Base M5** (`24 GB`) | `hetero2 parity` | `0.7373` | `2.064` | `116.07s` | `0.7434` / `0.7321` (`+0.3817`) | `-0.0017` `[-0.0039, +0.0006]` (`1.28×`) | `1.20` / `1.54` / `3.08 GB` |

Across all three ANE profiles (`fast`, `balanced`, `parity`), `hetero2` cuts end-to-end Tier-1 wall time by **`1.71–1.75×`** on M4 Pro and **`1.28–1.34×`** on Base M5 while keeping `S_obj,e2e` within `0.0002–0.0023` of `gpu`. (`NanoBEIR` + `LegalBench` per-query variance spans the `-0.003` lossless lower bound on `N=325`, so formal equivalence under `lossless_tolerance=0.003` remains statistically unresolved even where point differences are `< 0.002`.)

### 4. Direct MIL (`100%` Pure-ANE `2.24 ms`) vs. `CoreAI-Opt` / `.aimodel` Findings

Audited across both **M4 Pro `macOS 26.6`** (`Track A`) and **Base M5 `macOS 27.2.1`** (`Track B`) in [`../../docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md#7-coreai-audit-of-22-decisions):

| Export / Runtime Path | Placement (`ANE / GPU / CPU`) | Warm Latency (`s64 st2` / Cat-1) | Parity / Outcome | Promotion Verdict |
| :--- | :---: | :---: | :--- | :---: |
| **Production Direct MIL `W8A16`** (`build_stage_mil_program`) | **`100%` Pure-ANE** (`369/369`, `306/306`, `375/375`, `317/317`) | **`2.24 ms`** (`st2 s64`, `85.9 MB`); **`3.585 EOT`** (M5 `N=24` smoke) | `real_cos = 0.9994` (`st2`), `0` NaN | **PROMOTED** (default) |
| `CoreAI-Opt` `ExportBackend.CoreML` (`macOS 26.6`, `fp16..w4_pb`) | **`0/349` ANE** (`100%` CPU; `_ANECompiler: ANECCompile()` fail) | `9.29 ms` (`4.1×` slower) | `st3 w4_pb` drops to `0.891` cos | **FAIL** (disqualified) |
| `macOS 27.2.1` `.aimodel` `w8` / `w4_pb` (`FP32` compute) | **`0` ANE / `336` GPU** (`MPSGraph` fallback) | `130.6 ms` warm; `0.660 EOT` (**`0.184×`** live `hetero2`) | `w4_pb` worse on speed, RAM (`3.07 GB`) & cos | **FAIL** (Gate 3) |
| `macOS 27.2.1` `.aimodel` `st0 s128 w8_f16` (`FP16` + `softplus` rewrite) | `FP16` compute + `relu + log1p(exp(-|x|))` | `2.997 ms` (`2.8×` faster than FP32 `st0`) | `cos = 0.9999` vs teacher | *Ablation receipt* |

> Direct hand-crafted 4D NCHW `conv1x1` MIL (`build_stage_mil_program`) remains the default ANE backend because `torch.export` lowering of chunked `GatedDeltaNet` block-inversion graphs fails ANE compilation on `macOS 26.6` and requires `FP16` compute + `F.softplus` replacement to avoid `MPSGraph` fallback on `27.2.1`.

### 5. `GdnDecodeShared`, Numerical Invariance & Paired `LongEmbed` `state_codec` (`4K–32K`)

- **`GdnDecodeShared` carried/segmented decode:** Ragged NanoBEIR throughput jumps from `7,918` to **`9,211 tok/s`** (`~1.16×`) on M4 Pro and from `12,743` to **`17,412 tok/s`** (`~1.37×`) on Base M5.
- **Precision & Apple10 (`Base M5`) streamed invariance:** Passing explicit zero `rec_state` into the first `S > 2048` streamed GatedDeltaNet slice guarantees identical `GdnDecodeShared` kernel routing on `slice-0`, eliminating the prior Base M5 `S=4096` stream-vs-whole divergence (`3.01e-03 -> 0.0` `max|stream - whole|` and `0.0` `slice-0` diff across all linear layers on both M4 Pro and Base M5). `PplxGatedDeltaNet` also evaluates `beta` `mx.sigmoid` in `float32` (`float16` path) and `MappedEmbedding.gather(..., out_dtype="float32")` preserves exact `F32`/`BF16` precision when loading into `bfloat16` activations.
- **Tier-1 GPU (`N=325`) verified runs:** M4 Pro improved from `305.34s` (`0.739249` score, `0.7868 EOT`) to `268.36s` (`0.738421`, `0.8943 EOT`, **`1.137×`**) in `f32` and `264.47s` (`0.738421`, `0.9074 EOT`, **`1.153×`**, `1.278 GB`) in `i8`; Base M5 improved from `203.68s` (`0.739189`, `1.1795 EOT`) to `149.29s` (`0.739011`, `1.6089 EOT`, **`1.364×`**) in `f32` and `146.51s` (`0.739011`, `1.6394 EOT`, **`1.390×`**) in `i8`. [v2.2 validation JSON](../../docs/benchmarks/v22-validation-20261121.json)

**Paired Tier-2 `LongEmbed 4K–32K` (`N=500`: `200` `passkey` `4K/8K/16K/32K` + `300` `2wikimqa`) — `float32` vs. opt-in `int8` `state_codec`:**

| Host | `state_codec` | Mean `nDCG@10` (`N=500`) | `passkey 16K–32K` (`n=100`, `R@10`) | `passkey 4K / 8K / 16K / 32K` | `2wikimqa` (`n=300`) | Eval Wall (`s`) / `EOT` | Mem (`mlx_peak` / `OS phys` / `peak_fp`) |
| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: |
| **M4 Pro** (`48 GB`) | `float32` (default) | `0.750958` | `0.2534` (`0.56`) | `1.000 / 0.636 / 0.460 / 0.047` | `0.894502` | `2205.5s` / `0.1702` | `4.27` / `1.69` / `5.60 GB` |
| **M4 Pro** (`48 GB`) | `int8` (opt-in) | **`0.751856`** | **`0.2579`** (`0.57`) | `1.000 / 0.636 / 0.462 / 0.053` | `0.894502` | `2265.8s` / `0.1659` | `4.25` / **`1.63`** / `5.59 GB` |
| **Base M5** (`24 GB`) | `float32` (default) | `0.751073` | `0.2541` (`0.56`) | `1.000 / 0.636 / 0.462 / 0.047` | `0.894462` | `1630.9s` / `0.2303` | `4.42` / `1.67` / `5.74 GB` |
| **Base M5** (`24 GB`) | `int8` (opt-in) | **`0.751856`** | **`0.2579`** (`0.57`) | `1.000 / 0.636 / 0.462 / 0.053` | `0.894502` | **`1628.0s`** / **`0.2309`** | `4.44` / **`1.64`** / `5.72 GB` |

- `int8` `state_codec` (`3.76×` smaller carried GDN recurrent state) passes paired equivalence cleanly against `float32` across `N=500` queries on both hosts (**M4 Pro** `Δ=+0.000898`, `95% CI = [-0.000310, +0.002105]`; **Base M5** `Δ=+0.000783`, `95% CI = [-0.000395, +0.001961]`) while holding `32K` retrieval inside `1.63–1.69 GB` live OS footprint. Base M5 (`NAX`) executes the full `4K–32K` sweep **`1.35×` faster** than M4 Pro (`1628s` vs `2205–2266s`).

### 6. Contextual Window Header Anchoring (`embed_contextual_parquet`, M4 Pro)

Validated on full-document `LegalBench-RAG` (`Tier-1`: `175` queries / `796` chunks; `Tier-2`: `568` queries / `2,700` chunks) and `LongEmbed 2wikimqa` (`300` queries):

| Windowing Config (`chunk_tokens=256`) | `LegalBench Tier-1` (`span_nDCG@10` / `char_R@10`) | `LegalBench Tier-2` (`span_nDCG@10` / `char_R@10`) | `2wikimqa` (`nDCG@10`) | Tier-2 Wall (`s`) / GPU Peak | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `isolated` (no surrounding context) | `0.3562` / `0.7225` | `0.3571` / `0.6248` | `1.0000` | `73.4s` / `1577.5 MB` | baseline |
| `context_tokens=512` (`rep=0`) | `0.3513` / `0.7185` | `0.3563` / `0.6220` | — | `73.1s` / `1577.2 MB` | collapsed |
| `context_tokens=2048` (`rep=0`) | `0.4528` / `0.6827` | `0.5145` / `0.7333` | `0.9923` | `76.6s` / `1650.1 MB` | prior default |
| `context_tokens=4096` (`rep=0`) | `0.5343` / `0.6912` | `0.5802` / `0.7265` | `0.9811` | `89.0s` / `2367.4 MB` | `+43%` GPU peak, hurts `2wikimqa` |
| `context_tokens=2048 + header_text` (`--header-col`) | `0.5446` / `0.7460` | `0.5952` / `0.7931` | — | `77.6s` / `1650.0 MB` | `+0.08–0.09` at `1%` cost |
| **`context_tokens=2048 + repeat_header_tokens=128`** | **`0.7352`** / **`0.9154`** | **`0.6832`** / **`0.8448`** | **`0.9920`** | `82.7s` / `1650.4 MB` | **PROMOTED** (`+0.28` T1 / `+0.17` T2, `0 MB` extra) |
| `context_tokens=2048 + overlap=1` (`rep=0`) | `0.4177` / `0.6634` | `0.5067` / `0.7131` | `0.9884` | `88.8s` / `1650.1 MB` | **REGRESSED** (`0` default) |
| `context_tokens=2048 + rep=128 + overlap=1` | `0.7326` / `0.8976` | `0.6887` / `0.8550` | `0.9926` | `96.3s` / `1650.1 MB` | opt-in only (`+16.5%` wall) |

`embed_contextual_parquet` and `fast-pplx parquet --context-tokens` default to `repeat_header_tokens=128` for `context_tokens >= 512` (disable via `--repeat-header-tokens 0`), support explicit `--header-text` / `--header-col` and opt-in `--window-overlap-chunks`, and slice off injected prefix/overlap rows before Parquet write so emitted rows stay strictly `1:1` with unique source chunks.

### 7. `v2.5` Multi-Vector `MaxSim`, `CompactRetrievalIndex` & `bin1024 -> 1024d INT8` Rerank

`FastPplx` sets `is_mrl_trained = False` (because `512d` truncation drops `-3.73 pp` on `LegalBench-Ctx` and `-3.10 pp` on `NanoSciFact` vs `+0.38 pp` on MRL-trained `FastEmbeddingGemma2`) and shares `fast_shared.retrieval_index.CompactRetrievalIndex` across `encode_documents_multivector(...)` and `build_index(...)`:

| Retrieval / Storage Mode (`M4 Pro`, `N=500` `LongEmbed` / `N=325` `Tier-1`) | Quality vs `FP32` Single-Vec | Storage / Footprint | Speedup & Notes |
| :--- | :---: | :---: | :--- |
| **`encode_documents_multivector` (`window_tokens=1024, context_tokens=2048, rep=128`, `MaxSim`)** | **`32K passkey`: `0.0470 -> 1.0000` split (`0.6743` pooled, `R@10=1.000`)** · **`2wikimqa`: `0.8945 -> 0.9851`** (`N=500`: `0.7510 -> 0.8452` pooled / `0.9911` split) | `15.97 KiB/doc` (`1024d INT8`, `0.0000` loss vs `FP32`); `1.72 GB` Metal peak (**`2.48×` lower** than `4.27 GB` `32K`) | **`2.31×` faster GPU encode** (`955.4s` vs `2205.5s`) by replacing `O(32768^2)` full attention with `2K` contextual windows |
| **Symmetric `1024d INT8` (`mode="auto"` `< 16K` / `"int8"`)** | **`-0.0007` `NanoBEIR`** `[-0.0020, +0.0007]`, **`-0.0012` `LB-Ctx`** `[-0.0029, +0.0005]` (**PASS** `<= 0.003`, `2.6–6.8×` lower error than `native_tanh_int8`) | `1,026 B/vec` (**`3.99×` smaller**) | **`2.11 ms/q`** `Q=1` @ `100K` docs (`1.34×` vs `2.82 ms/q` `FP32` BLAS); enforced automatically by `CorpusBackfillConfig.for_engine(engine)` (`mrl_dim=None`) |
| **NEON `bin1024 -> Top-64 -> 1024d INT8` (`mode="auto"` `>= 16K` / `"binary_int8"`)** | **`+0.0006` `NanoBEIR`** `[-0.0016, +0.0036]`, **`-0.0012` `LB-Ctx`** `[-0.0029, +0.0005]` (**PASS** `<= 0.003` lossless gate) | `128 B` hot binary scan (`32×` smaller) + `1,026 B` `INT8` (`3.55×` total) | **`0.66 ms/q` interactive `Q=1` @ `100K` docs** (**`4.28×` faster** than `2.82 ms/q` `FP32` BLAS; `0.61 ms/q` batched; `1.70 ms` M5 / `6.43 ms` M4 on `72`-window `multivector=True` `MaxSim`) |
| **FFD `pack_ane_buckets` + Host-Aware `hetero2` + Async Parquet (`T4.1–T4.3`)** | Exact deterministic row/vector parity (`resume=True` + `.tmp` cleanup) | **`26` vs `32` ANE bins (`-18.8%`)**, **`648` vs `1,638` pad toks (`-60.4%`)** | Mixed-64 **`19,443 tok/s`** M5 (`1.17×` vs GPU) / **`17,793 tok/s`** M4 (**`1.84×`** vs GPU); `S>512` (`768`/`1024`) ANE gated out (M4 GPU fallback, M5 GPU `1.40×` faster) |
| **Opt-In `quantize_weights="w8"` (`T4.4`, `--quantize-weights w8`)** | `16`-case `min_f32_cos = 0.99883` (`mean = 0.99970`) | **`460.8 -> 247.0 MB` (`-46.4%` active linear weights)** | `0.94–0.95×` FP16 speed on M5 (`15,628` vs `16,419 tok/s` @ `16×2048`), saving `~214 MB` for `24 GB` multi-model co-residency (`default="none"`) |

*(Apples-to-apples note on `LongEmbed` `passkey`: official per-length splits evaluate 50 queries against 100 docs at a single length and reach **`1.0000` `nDCG@10`** across all `4K–32K` splits; `benchmark_outcomes.py` also reports the pooled `400`-doc corpus across `[4K, 8K, 16K, 32K]`, where identical target entities appear in all 4 splits and cap any length-invariant oracle near `0.640–0.680` at `Recall@10 = 1.000`.)*

## What keeps it fast

| Component | Optimization | Source |
| :--- | :--- | :--- |
| **GPU (`MLX` + Metal)** | MIT-adapted `gated_delta_update` kernel in `mx.float16`, zero per-layer `mx.eval` (`S <= 2048`), fused `qkv_proj` / `gate_up_proj`, FP32 `+1.0` zero-centered `RMSNorm` (`GatedRMSNorm` unshifted), whole-slice `a_all`/`b_all` GDN projections | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Streamed `32K` Path** | Automatic `1024`-token recurrent `GatedDeltaNet` slices + tile-aligned `512`-token bidirectional `SDPA` slices (`tile_aligned_slices`) + `MappedEmbedding` memmap | [`src/fast_pplx/gpu.py`](src/fast_pplx/gpu.py) |
| **Staged Pure-ANE Pipeline** | Direct MIL 4D `conv1x1` stages (`369..375` ANE ops, `0` CPU/GPU), `c32` `GatedDeltaNet` (`t_delta` FTZ floor `1e-4`), `fast`/`balanced`/`parity` profiles (`MaxAbsRMSNorm` + `TanhSiLU`), `pipe_depth=4` async ring | [`scripts/build_ane_models.py`](scripts/build_ane_models.py), [`native/pplx_coreml_runner.mm`](native/pplx_coreml_runner.mm) |
| **Native I/O + Scheduler** | `NEON` FP16/uint16 token gather + block-diagonal masks + Accelerate `cblas_sgemm` MRL/`tanh` projection; `hetero2` chunk-aligned packing + tail steal | [`native/pplx_fast_io.cpp`](native/pplx_fast_io.cpp), [`src/fast_pplx/engine.py`](src/fast_pplx/engine.py) |

## CLI, Two-Knob Contextual Parquet & Server

```bash
# Single/batch embeddings (--mode auto|gpu|ane|hetero|hetero2, --dimensions 128..1024, --dtype int8|float16|float32)
fast-pplx embed "First passage" "Second passage" --dimensions 256 --dtype int8 --json

# Standard or contextual Parquet ingestion (--context-tokens defaults repeat_header_tokens=128 when >= 512)
fast-pplx parquet --input docs.parquet --output out_shards/ --text-col body --header-col title \
  --mode hetero2 --chunk-tokens 256 --context-tokens 2048 --dimensions 512 --dtype int8 --resume

# OpenAI-compatible HTTP server (/v1/embeddings with optional "dimensions" and opt-in 8-bit Metal weights)
fast-pplx serve --host 127.0.0.1 --port 8080 --mode hetero2 --dimensions 512 --quantize-weights w8
```

## Reproducible Benchmark Harness

Run [`scripts/benchmark.py`](scripts/benchmark.py) (all GPU/ANE runs self-serialize under `/tmp/m4_gpu_validation.lock`):

```bash
# Safe preflight check (zero GPU/ANE work)
python packages/fast-pplx/scripts/benchmark.py all --dry-run

# 1. Speed + internal & public baselines
python packages/fast-pplx/scripts/benchmark.py speed \
  --modes gpu-no-mmap,gpu-bf16,gpu,ane,hetero2,pplx-context-mlx-f16,torch-mps,torch-cpu \
  --pplx-mlx-path /path/to/pplx-context-mlx --torch-python /path/to/python \
  --json-out artifacts/bench_speed.json

# 2. ANE profile ABBA sweep (fast / balanced / parity in isolated temp homes)
python packages/fast-pplx/scripts/benchmark.py ane-profiles \
  --profiles fast,balanced,parity --seq-lens 128,256 --repeats 2 \
  --pplx-mlx-path /path/to/pplx-context-mlx --torch-python /path/to/python

# 3. 16-case PyTorch CPU reference parity, 32K scaling, contextual windows & bit-identity
python packages/fast-pplx/scripts/benchmark.py parity --modes gpu,ane,hetero2
python packages/fast-pplx/scripts/benchmark.py context-limit --lengths 512,1024,2048,4096,8192,16384,32768
python packages/fast-pplx/scripts/benchmark.py context-windows --context-budgets 512,1024,2048,4096
python packages/fast-pplx/scripts/benchmark.py bit-identity dump --out-dir artifacts/bit_base
```

## Scope & Not-tested

- Verified on **Apple M4 Pro (`Mac16,8`, `48 GB`, `macOS 26.6`)** and **Base Apple M5 (`Mac17,2`, `24 GB`, `macOS 27.2.1`, Xcode CLT-only)** (`coremltools >= 8.0` optional extra required only when compiling fresh `.mlmodelc` bundles).
- Default live ANE benchmarks use `w8a16` (`fast`/`balanced`/`parity`); optional `--precision w8a8` CoreML calibration via `--calib-stats` is supported via CLI flags.

## Acknowledgments

- **[`zm2231/pplx-context-mlx`](https://github.com/zm2231/pplx-context-mlx)** (Copyright © 2026 Merchantry LLC, MIT License): Provided the foundational architectural and evaluation patterns for running `pplx-embed-v2-context-0.6b` cleanly on Apple Silicon — including safetensors memmap embedding gather (`MappedEmbedding`), `S > 2048` streamed GatedDeltaNet + tile-aligned SDPA slicing (`tile_aligned_slices`), unshifted `GatedRMSNorm` vs `+1.0` zero-centered `RMSNorm` parity analysis, 16-case text reference parity & fault-injection suites, `--context-tokens` contextual window evaluation findings, and unified-memory OS footprint benchmarking.
- **[`mlx-lm`](https://github.com/ml-explore/mlx-lm)** (Copyright © 2025 Apple Inc., MIT License): Adapted MIT-licensed Metal `gated_delta_update` recurrence kernels (`src/fast_pplx/gpu.py`).
- **[`perplexity-ai/pplx-embed-v2-context-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b)**: Upstream model architecture, weights, and contextual chunk separator conventions.

## Author & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](LICENSE).
