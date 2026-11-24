# `fast-ocr`

Extract text, rectangular GFM Markdown tables, and structured fields from PDFs and document images locally on Apple Silicon. In forced VLM evaluation (`--ocr-backend ovis`), `fast-ocr` scored **0.844 on `olmOCR-bench(4)` (4/4 valid)**, **0.869 TEDS on `FinTabNet-OTSL(2)`**, and **0.909 on `OmniDocBench-EN(2)`**, running **1.22–1.51× faster on M4 Pro** and **1.53–2.05× faster EOT on Base M5** than stock `mlx_vlm OvisOCR2-8bit`. In full Wave 3 Tier-1 hybrid `auto` evaluation (`96` pages), `fast-ocr` achieves **`0.6155` `olmOCR(42)` (`42/42`)**, **`0.8935` `FinTabNet(30)` (`30/30`)**, and **`0.6574` `OmniDocBench-EN(24)`** (all three `PASS`, `0` loops on contended M4 Pro), and **`0.6128` / `0.8867` (`PASS`) / `0.6872` (`PASS`)** on quiescent Base M5.

The router (`FastOCR` / `--backend auto`) pairs zero-training extraction across three tiers: clean native PDF text bypass via `pypdfium2` (`139–220 pp/s`), isolated Apple Vision on the macOS Neural Engine (`AppleVisionOCR`), and lazy Qwen3.5 VLM escalation (`OvisOCR2-8bit` or `NuExtract3-mlx-4bits`) backed by [`fast-qwen3`](../fast-qwen3/README.md).

## Quick start

From the `fast-silicon` workspace root, install the package:

```bash
uv sync --python 3.13 --package fast-ocr --inexact
```

Extract Markdown in Python:

```python
from fast_ocr import FastOCR

with FastOCR(backend="auto") as ocr:
    page = ocr.ocr_image("table_scan.png")
    print(page.text)
    pages = ocr.ocr_pdf("packet.pdf", triage_dpi=100)
```

Run common CLI workflows:

```bash
# Standalone ANE Apple Vision -> rectangular GFM Markdown
.venv/bin/fast-ocr run document.pdf --backend apple-vision --format markdown

# Accelerated OvisOCR2-8bit with strict Markdown grammar and 2x numeric zoom recovery
.venv/bin/fast-ocr run table_scan.png --backend ovis --format markdown --strict-markdown-grammar --zoom-recover

# Explicit KV/GDN precision or full multilingual vocab override on batched directory runs
.venv/bin/fast-ocr run scans_dir/ --backend ovis --batch-size 16 --kv-i8 q4_quarot --gdn-state-mode int8_blk32 --no-latin-only

# Opt-in mixed prose+table region routing (keeps formula pages on full-page VLM)
.venv/bin/fast-ocr run mixed_page.png --backend auto --region-routing --format markdown

# Hybrid PDF routing with 100 DPI ANE triage
.venv/bin/fast-ocr run packet.pdf --backend auto --triage-dpi 100 --max-pdf-pixels 20000000 --json --out out_dir/

# Resumable sharded Parquet OCR with host-budget memory limits
.venv/bin/fast-ocr parquet inputs.parquet out_ocr/ --backend auto --shard-rows 64 --json

# Compare backends with per-stage timings and Markdown validation
.venv/bin/fast-ocr bench sample.png --backend all --warmup 1 --repeat 3 --json
```

## Routing and capabilities

| Tier | Trigger | Behavior |
|---|---|---|
| **Native PDF text** | Clean text pages with zero control garble | Emits `pypdfium2` text directly without rasterizing (`139–220 pp/s`), stripping running stamps while still rasterizing vector-outlined pages. |
| **Apple Vision ANE** | Single-column scanned or rasterized pages | Runs isolated macOS `VNRecognizeTextRequest` with multi-band deduplication (`bands=1|3`), 360° baseline orientation, tiny-crop upscaling, and sub-millisecond GFM Markdown synthesis. |
| **Qwen3.5 VLM (`Ovis`)** | Tables, dense ledgers, formulas, or low-confidence pages | Escalates asynchronously behind ANE pre-analysis in double-buffered `batch_size=16` waves (`FastOvisOCREngine`). |

- **VLM preprocessing and vocabulary.** Ink-tight margin cropping cuts ViT tokens by `56–61%` (`4.5–5.7×` faster ViT). Adaptive `24 px`/line resolution sizing preserves dense typography, camera-gated `--deskew` prevents false warps on flat scans (`0/106`), unpinned `<150K px` Markdown tables upscale toward `262_144` pixels, and lossless Latin/math/symbol pruning trims the `OvisOCR2` head to `140K` tokens (`--no-latin-only` disables).
- **Speculative decoding, confidence, and recovery.** Bit-identical prompt-lookup decoding (`vision+self`, `draft_k=4`) stays active under `--strict-markdown-grammar` and JSON schema locks (`1.85×` M4 Pro / `1.144×` M5 on GFM; `1.11–1.80×` on `CORD-v2`), records `metadata["field_boxes"]`, and recovers from rare degenerate loops or mis-rotations (`3.77×` faster recovery). Async Accelerate `vDSP` confidence (`B=1` default) reaches `0.90–0.91` `FinTabNet` error-detection AUROC at `<= 0%` overhead.
- **Memory, KV, and GDN state precision.** `--kv-i8 auto` keeps `bf16` on both Apple9 and Apple10 NAX unless context or `batch*ctx` memory pressure exceeds cutoffs (`ctx >= 8192/16384` or `B*ctx >= 262_144/524_288` on Apple9; `ctx >= 16384/32768` or `B*ctx >= 262_144/524_288` on Apple10 NAX). `--gdn-state-mode auto` keeps `fp32` below `B=8` (`< 4` on `<= 28 GB`) and switches to `int8_blk32` (`3.76×` DRAM cut) on larger batches. `FAST_SILICON_QWEN3_MPP_W8A8=1` unlocks opt-in Apple10 NAX prefill with zero-shadow-RAM inline XOR unpacking; `--drop-white-patches` stays opt-in (`False` default).
- **Opt-in routing, zoom recovery, triage, and Parquet.** `--zoom-recover` re-checks digits missed by the VLM against ANE text (`0.994` to `1.000` recall). `--region-routing` sends clean digital prose through `pypdfium2` and crops only table regions (`3.468×` on mixed digital PDFs; keeps formulas on full-page VLM). `--triage-dpi` screens packets at low resolution (`100` on `run`, `72` default on `parquet`, `11.5–16.6×` speedup) under `--max-pdf-pixels` (`20M` default), and `fast-ocr parquet` checkpoints resumable `zstd` shards with `manifest.json`.

## Benchmark highlights

Measured on Apple M4 Pro (`48 GB`, `macOS 26.6`) and Base Apple M5 (`24 GB`, `macOS 27.2.1`) in forced VLM mode (`--ocr-backend ovis --ocr-render-only`), standalone ANE, and hybrid `auto` paths. [Cross-silicon OCR & NAX tables](../../docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23) · [Wave 3 Tier-1 OCR outcomes](../../docs/BENCHMARKS.md#162-post-fix-fast-qwen3-fast-ocr-fast-embeddinggemma-2-vision--fast-pplx-results) · [v3.0 OCR confidence & PDF splicing](../../docs/BENCHMARKS.md#143-ocr-zero-sync-confidence-and-guarded-digital-pdf-splicing) · [Research & architecture spec](docs/OCR_RESEARCH_AND_ARCHITECTURE.md) · [Category 5 benchmark guide](docs/OCR_BENCHMARKS_AND_BASELINES.md)

| Suite / Workload | Engine | M4 Pro (48 GB) | Base M5 (24 GB) | Notes |
| :--- | :--- | ---: | ---: | :--- |
| **Wave 3 `olmOCR-bench(42)` (`auto_gfm`)** | **`fast-ocr` (`auto`)** | **`0.6155` (`42/42`, `0` loops, `PASS`)** | **`0.6128` (`41/42`, `1` loop)** | **`+13.65 pp`** (M4 Pro `0.4790 -> 0.6155`) / **`+21.34 pp`** (M5 `0.3994 -> 0.6128`) |
| **Wave 3 `FinTabNet-OTSL(30)` (`auto_gfm`)** | **`fast-ocr` (`auto`)** | **`0.8935` TEDS (`30/30`, `1.019` EOT, `PASS`)** | **`0.8867` TEDS (`30/30`, `0.930` EOT, `PASS`)** | **`+5.57 pp`** M4 Pro / **`+2.68 pp`** Base M5 over baseline |
| **Wave 3 `OmniDocBench-EN(24)` (`auto_gfm`)** | **`fast-ocr` (`auto`)** | **`0.6574` (`22/24`, `0` loops, `PASS`)** | **`0.6872` (`22/24`, `0` loops, `PASS`)** | Flipped both Macs from `FAIL` (`0.5805`/`0.5820`) to **`PASS`** |
| **`olmOCR-bench(4)` (forced VLM)** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.8438` (`4/4`) · `0.082` EOT (`5.75 GB`)** | **`0.8438` (`4/4`) · `0.090` EOT (`5.01 GB`)** | `1.44×` (M4 Pro) / `2.05×` (M5) EOT vs. stock `mlx_vlm OvisOCR2` |
| **`FinTabNet-OTSL(2)`** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.8687` TEDS (`0.674 pp/s`)** | **`0.8687` TEDS (`0.733 pp/s`)** | `1.22×` (M4 Pro) / `1.72×` (M5) `pp/s` vs. stock `OvisOCR2`; `LightOnOCR-2` looped (`0.00`) |
| **`OmniDocBench-EN(2)`** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.9093` (`0.117 pp/s`)** | **`0.9093` (`0.124 pp/s`)** | `1.51×` (M4 Pro) / `1.53×` (M5) `pp/s` vs. stock `OvisOCR2`; `2.44×–2.74×` M5 vs. `GLM-OCR` / `LightOnOCR-2` |
| **Strict GFM PLD (`--strict-markdown-grammar`)** | `fast-ocr` (`vision+self`, `K=4`) | **`1.85×` decode** (`315.2 tok/s`, `6/6` parity) | **`1.144×` wall** (`2/2` parity, `0.8687` TEDS) | Preserves PLD under column lock; `1.11×–1.80×` on `CORD-v2` JSON |
| **`B=16..24` Batched Table Waves** | `fast-ocr` (`OvisOCR2-8bit`) | `B=16`: **`1.231 pp/s`** (`0.8675` TEDS)<br>`B=24`: **`21.34 s`** (`0.7705` TEDS)<br>Tiny `n=16`: **`0.8512` TEDS** (`16.06 s`) | `B=24`: **`39.55 s`** (`0.7649` TEDS, `1.373×` wave gate)<br>Tiny `n=16`: **`0.8421` TEDS** (`27.97 s`) | Single-wave batching (`<= 1024` prompt tokens); auto-upscales `<150K px` tables |
| **`B=1` Logprob Confidence (`"auto"`)** | `fast-ocr` (`vDSP`) | **`0.9135` AUROC** (`FinTabNet-30`, `-2.00%` wall, `30/30` exact) | *(`0` extra MLX)* | `0.7792` AUROC on hard `olmOCR` trio (`-3.65%` wall) |
| **Opt-in `--region-routing` + Digital-PDF Splice** | `fast-ocr` | **`3.468×`** mixed digital PDF (`2.75 s` to `0.79 s`); `us_gov_ocr`: **`0.6065`** | *(`False` default)* | Avoids multi-column interleaving; lifts `uscourts` `+7.67 pp` to `0.9707` |
| **Parallel ANE (`FinTabNet` `n=16`, `1024×512`)** | `AppleVisionOCR` (`workers=4`) | **`9.61 pp/s`** (`1.665 s`, `1.592×`, `16/16` parity) | — | `FastOCR.ocr_images(..., batch_size=16)` runs `9.47 pp/s` |
| **Adaptive 24 px + Ordered/Relock PLD** | `fast-ocr` (`OvisOCR2-8bit`) | **`1.057×–1.140×` wall (`+11.1%` dense prose)** | — | Exact `olmOCR(8)`, `-0.0002` `OmniDoc(8)` (`-24%` visual tokens), `10/10` bit-identical relock |
| **Auto Camera-Photo Deskew (`CORD-v2(8)`)** | `fast-ocr` (`deskew="auto"`) | **`+10.94%` recall (`0.809` to `0.919`) · `1.637×` wall** | — | `14.67 s` to `8.96 s` (`-31%` visual tokens); `0/106` false warps on flat scans |
| **Tiny table crops (`FinTabNet-OTSL(2)`)** | `AppleVisionOCR` (`ANE`) | **`0.6121` (`7.03 pp/s`)** | **`0.5914` (`25.16 pp/s`, `14.88` EOT)** | Bicubic upscaling resolves macOS 27.2.1 tiny-crop drops (`0.0000` to `0.5914`) |

## Scope & License

Verified on **Apple M4 Pro (`48 GB`, `macOS 26.6`)** and **Base Apple M5 (`24 GB`, `macOS 27.2.1`)**. Non-macOS targets are unsupported because Apple Vision requires macOS frameworks.

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](../../LICENSE).
