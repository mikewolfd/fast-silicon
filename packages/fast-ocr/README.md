# fast-ocr

Convert PDFs, scanned pages, and dense financial tables into clean Markdown and rectangular GFM tables locally on Apple Silicon. Routes work across three tiers without training: native PDF text bypass via `pypdfium2` (`139-220 pp/s`), isolated Apple Vision on the 16-core Neural Engine (`5-9.6 pp/s`), and lazy Qwen3.5 VLM escalation (`OvisOCR2-8bit` or `NuExtract3`) backed by [`fast-qwen3`](../fast-qwen3/README.md).

Across a 96-page public benchmark suite (`olmOCR-bench`, `FinTabNet-OTSL`, `OmniDocBench-EN`), the `auto` router scores **0.616 on `olmOCR(42)` (`42/42`)**, **0.894 TEDS on `FinTabNet(30)` (`30/30`)**, and **0.657-0.687 on `OmniDocBench(24)`**, running **1.22-2.05x** faster than stock `mlx_vlm OvisOCR2-8bit` on full VLM pages and **3.47x** faster on mixed digital PDFs with region splicing. Zero telemetry, no account, and runs 100% offline.

## Example: Strict Markdown Table Locking

Raw multi-column table crops often drop alignment pipes or collapse adjacent cells into whitespace-separated runs under unconstrained OCR. Ink-tight cropping plus column-locked decoding emits clean rectangular GFM Markdown (illustrative example):

| Unconstrained OCR text (illustrative collapsed cells) | `fast-ocr --format markdown --strict-markdown-grammar` |
| --- | --- |
| `Segment 2024 2023`<br>`Cloud Revenue $1,420 $1,105`<br>`Operating Margin 28.4% 24.1%` | `\| Segment \| 2024 \| 2023 \|`<br>`\| --- \| ---: \| ---: \|`<br>`\| Cloud Revenue \| $1,420 \| $1,105 \|`<br>`\| Operating Margin \| 28.4% \| 24.1% \|` |

## Requirements

- Apple Silicon Mac (`M1-M5`), macOS 15+ recommended across the monorepo (Apple Vision and MLX support `macOS 14+`; verified on macOS 26.6 and 27.2.1), Python 3.12+, and `uv`. Non-macOS targets (Linux, Windows, Intel Macs) are unsupported because Apple Vision requires macOS frameworks.
- Clean digital PDFs and Apple Vision ANE OCR require zero Hugging Face weights; VLM escalation downloads `mlx-community/OvisOCR2-8bit` (~2.6 GB) once when first used.

## Quickstart

```bash
git clone https://github.com/mikewolfd/fast-silicon
cd fast-silicon
uv sync --python 3.13 --package fast-ocr --inexact
```

```python
from fast_ocr import FastOCR

with FastOCR(backend="auto") as ocr:
    page = ocr.ocr_image("table_scan.png")
    print(page.text)
    pages = ocr.ocr_pdf("packet.pdf", triage_dpi=100)
```

```bash
.venv/bin/fast-ocr run document.pdf --backend apple-vision --format markdown
.venv/bin/fast-ocr run table_scan.png --backend ovis --format markdown --strict-markdown-grammar --zoom-recover
.venv/bin/fast-ocr run mixed_page.png --backend auto --region-routing --format markdown
.venv/bin/fast-ocr run packet.pdf --backend auto --triage-dpi 100 --json --out out_dir/
.venv/bin/fast-ocr parquet inputs.parquet out_ocr/ --backend auto --shard-rows 64 --json
```

## Routing and key defaults

| Tier | Trigger | Behavior |
| --- | --- | --- |
| **Native PDF text** | Clean digital text pages (`0` control garble) | Extracts `pypdfium2` text directly without rasterizing (`139-220 pp/s`), stripping stamps while rasterizing vector-outlined pages. |
| **Apple Vision ANE** | Clean single-column scans / raster pages | Runs isolated macOS `VNRecognizeTextRequest` with multi-band dedup (`workers=4` reaches `9.61 pp/s` on resized table crops; `4.97-7.55 pp/s` full pages), 360-deg orientation, tiny-crop upscaling, and GFM synthesis. |
| **Qwen3.5 VLM (`Ovis`)** | Tables, ledgers, formulas, or low-conf pages | Escalates asynchronously behind ANE pre-analysis (`FastOvisOCREngine`) via continuous rolling `B=8` batched speculative PLD (`K=4`), falling back to static length waves for JSON schemas, sampling, or telemetry. |

- **Preprocessing & speculative decode:** Ink-tight cropping cuts ViT tokens `56-61%`; `compute_adaptive_max_pixels` caps standard pages at `3.15M` px (`1.873x` faster `olmOCR-bench(21)`, `-1.94 GB` peak RAM, preserving `16.78M` escape for tiny-font and low-confidence scans); `analyze_page_from_lines` excludes isolated court-caption/bullet delimiters (`+3.64 pp` on `us_gov_ocr` judicial); default Tier-2 `strip_running_headers_footers` + gated `normalize_faded_scan_image` lift Base M5 `olmOCR-bench(42)` from `0.6128` to `0.6402` (`headers_footers 4/12 -> 7/12`); camera-gated `--deskew auto` boosts `CORD-v2` recall `+10.9 pp` at `1.64x` speed; Latin/math head pruning trims Ovis to 140K tokens; zero-copy rolling `B=8` batched PLD (`K=4`) plus compact-table `B=16 AR` (`1.235x` on `FinTabNet-30`) and single-pass `gpu_argmax_conf` (`0.914` AUROC) speed VLM decoding without extra MLX RAM; and opt-in `v7final` (`--ocr-model v7final` / `FAST_OCR_GFM_MODEL_DIR`) lifts `OmniDocBench-EN(24)` to `0.7472` (`24/24` valid at `8192` tokens, `1.254x` faster) and `FinTabNet-OTSL(30)` `NumF1` to `0.8923` (`1.508x` faster) while leaving stock `OvisOCR2-8bit` as default.
- **Memory & opt-in flags:** `--kv-i8 auto` keeps BF16 until context exceeds `16K/32K` on M5 or `8K/16K` on M4; `--gdn-state-mode auto` switches GDN recurrent state to `int8_blk32` (`3.76x` smaller) on large batches; outlier `>=8K`-vis-tok pages isolate into `B=1` while standard pages sort `shortest_first` (`1.172x` on `OmniDocBench-EN(12)`); ANE pool workers auto-cap at `2` on `<= 28 GB` Macs (`4` on `>= 36 GB`); `--triage-dpi` defaults to `72` on `parquet`; `--region-routing` reaches `3.47x` on mixed digital PDFs; and `--drop-white-patches` stays off by default.

## Benchmark highlights

Measured on 48 GB M4 Pro (`macOS 26.6`) and 24 GB Base M5 (`macOS 27.2.1`) ([cross-silicon OCR tables](../../docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23), [full 96-page results](../../docs/BENCHMARKS.md#162-post-fix-fast-qwen3-fast-ocr-fast-embeddinggemma-2-vision--fast-pplx-results)):

| Suite / Workload | Engine | M4 Pro (48 GB) | Base M5 (24 GB) | Notes |
| --- | --- | ---: | ---: | --- |
| **`olmOCR-bench(42)` (`auto`)** | **`fast-ocr` (`auto`)** | **0.6155** (`42/42`, 0 loops) | **0.6128** (`41/42`, 1 loop) | `+13.7 pp` (M4P) / `+21.3 pp` (M5) vs pre-triage baseline |
| **`FinTabNet-OTSL(30)` (`auto`)** | **`fast-ocr` (`auto`)** | **0.8935** TEDS (`30/30`, 1.02 score/s) | **0.8867** TEDS (`30/30`, 0.93 score/s) | `+5.6 pp` (M4P) / `+2.7 pp` (M5) over baseline |
| **`OmniDocBench-EN(24)` (`auto`)** | **`fast-ocr` (`auto`)** | **0.6574** (`22/24`, 0 loops) | **0.6872** (`22/24`, 0 loops) | `+7.7 pp` (M4P) / `+10.5 pp` (M5) over baseline |
| **`olmOCR-bench(4)` (forced VLM)** | `fast-ocr` (`OvisOCR2-8bit`) | **0.8438** (`4/4`, 5.75 GB) | **0.8438** (`4/4`, 5.01 GB) | **1.44x** (M4P) / **2.05x** (M5) vs stock `mlx_vlm OvisOCR2` |
| **Mixed digital PDF splice** | `fast-ocr` (`--region-routing`) | **3.47x** (`2.75 s -> 0.79 s`) | Opt-in (`False` default) | Lifts `govinfo_uscourts` `+7.7 pp` to `0.9707` |
| **Parallel ANE (`n=16` crops)** | `AppleVisionOCR` (`workers=4`) | **9.61 pp/s** (`1.67 s`, 1.59x serial) | - | `FastOCR.ocr_images(..., batch_size=16)` runs `9.47 pp/s` |
| **`B=1` Logprob confidence** | `fast-ocr` (`gpu_argmax_conf`) | **0.9135** AUROC (`FinTabNet-30`) | `0` extra MLX RAM | On-GPU Metal SIMD top-1/top-2/logsumexp (`+6.5-12.8%` `B=1` decode) |
| **Rolling batched PLD (`B=16`)** | `fast-ocr` (`OvisOCR2-8bit`) | **1.49x** mixed / **1.27x** OmniDoc | **2.43-2.96 GB** | `Mixed(8+8)` **0.281** vs `0.189 pp/s`; `FinTabNet(30)` **-33.5%** RAM (`0.902` TEDS, `0` loops) |

*(On small 2-page forced-VLM checks against stock `mlx_vlm OvisOCR2-8bit`, `FinTabNet(2)` ran `1.22-1.72x` and `OmniDocBench(2)` ran `1.51-1.53x`, `2.44-2.74x` faster than `GLM-OCR`/`LightOnOCR-2`.)* [Architecture & OCR specs](docs/OCR_RESEARCH_AND_ARCHITECTURE.md)

## Limitations, Prior Art & License

- **Limitations:** macOS only (`VNRecognizeTextRequest` is unavailable on Linux/Windows). M4 Pro hosts on macOS `< 27.2` run slower on Apple Vision tiny-crop ANE upscaling than macOS 27.2.1+. Forced-VLM `latin_only=True` prunes non-Latin output heads to 140K tokens for speed; pass `--no-latin-only` for CJK or Cyrillic documents.
- **Prior art:** Combines Apple `Vision`, `pypdfium2`, `mlx_vlm`, `llguidance`, and `mlx-community/OvisOCR2-8bit` / `NuExtract3` checkpoints. For fast 1-vector page retrieval without full OCR text generation, use [`fast-embeddinggemma-2`](../fast-embeddinggemma-2) or [`fast-pplx`](../fast-pplx).
- **License:** Package code is [MIT licensed](LICENSE); upstream VLM weights keep their original licenses (Qwen / AIDC / Numind).
