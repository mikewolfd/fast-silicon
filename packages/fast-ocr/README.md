# `fast-ocr`

Zero-training high-speed Apple Silicon OCR suite built on macOS Neural Engine (`VNRecognizeTextRequest`) and [`fast-qwen3`](../fast-qwen3/README.md).

## Key Capabilities

1. **Independent ANE Backend (`--backend apple-vision` / `AppleVisionOCR`)**:
   - Zero MLX or VLM imports (`importlib` isolated, millisecond cold start, zero `CGDataProvider` leak) with macOS `SystemVersion.plist` + `VNRecognizeTextRequest.revision()` metadata provenance.
   - Multi-band strip deduplication (`bands=1|3`), `--no-language-correction`, tiny-crop bicubic upscaling (`_prepare_vision_image`, lifting `<600 px` table accuracy on macOS 26.6 and 27.2.1 with zero full-page slowdown), 360° baseline-vector orientation inference, and `ProcessPoolExecutor` ANE scaling.
   - `<1 ms` XY-Cut + typography heading + rectangular GFM table Markdown synthesis (`synthesize_markdown_from_lines`).
2. **Accelerated Local Qwen3.5 VLM Backend (`--backend ovis` / `FastOvisOCREngine`)**:
   - Supports `mlx-community/OvisOCR2-8bit` (default) and `numind/NuExtract3-mlx-4bits` via shared `fast_shared.qwen35_vision` + `fast-qwen3` (optional `FAST_SILICON_QWEN3_MPP_W8A8=1` Apple10 NAX prefill acceleration), automatic `q4_quarot` KV cache under batch/context pressure (`--kv-i8`: `auto` -> `q4_quarot` at `B >= 8` or `ctx >= 4096/8192`), and `GdnDecodeShared` recurrent-state compression (`--gdn-state-mode`: `auto` keeps `fp32` for `B < 8` / `< 4` on `<= 28 GB` hosts so `B=1` PLD pays zero kernel dispatch overhead, and switches to `int8_blk32` `3.76×` compression at `B >= 8` / `>= 4`). Strict bright-patch dropping (`--drop-white-patches` / `drop_white_patches=True`, `False` default) stays strictly opt-in because dropping interior white column gaps caused compact-table hallucinations (`FT_175`/`FT_240`), while `crop_margins=True` removes outer margins losslessly pre-ViT.
   - Exact interleaved 3D MRoPE (`mrope_section=[11, 11, 10]`, `rope_offset_delta` decode invariance).
   - Lossless OCR Latin/symbol/math vocabulary pruning (`build_ocr_vocab_subset`, `140K` subset with `" " + sym` and scientific math/footnote glyphs strictly under Latin gating; `0%` drift, `+10–12%` decode speed; disable via `--no-latin-only` / `latin_only=False`).
   - Stage-0 foreground ink-tight margin crop (`4.5×–5.7×` ViT speedup), lossless `24px`/line adaptive visual resolution floor (`--no-adaptive-resolution` disables), camera-gated ANE perspective deskew (`--deskew [auto|on|off]` / `--no-deskew`, catching `39/39` `CORD-v2` camera photos at `0%` false triggers on flat scans/tables), 360° auto-orientation, and `detect_catastrophic_token_loop()` + `auto_loop_recover=True` uncropped fallback.
   - Zero-training bit-identical speculative decoding (`OCRPLDDrafter`, `vision+self` with `K=4`, reading-order line ordering + windowed cursor relocking) delivering `1.94×–2.75×` decode speedup, including grammar-checked drafting under `--strict-markdown-grammar` (`1.85×` M4 Pro / `1.144×` Base M5) and JSON schema constraints (`1.11×–1.80×` `CORD-v2`).
   - Zero-VLM box-grounded provenance (`metadata["field_boxes"]` via `align_json_values_to_lines`) aligning extracted JSON leaf values back to oriented ANE `OCRLine` bounding boxes, plus automatic host-aware MLX memory budgeting (e.g. `19.2 GiB` default cap on `24 GiB` Macs, overridable via `--memory-limit-gb`).
   - 2D occupancy HTML→GFM canonicalizer (`colspan`/`rowspan` expansion, fence/math balancing), opt-in JIT `C`-column `llguidance` table grammar (`--strict-markdown-grammar`), opt-in mixed prose+table crop escalation (`--region-routing`, `REGION_CROP_MIN_PIXELS=4096`, `batch_size` capped; can interleave multi-column ANE prose and excludes formula pages), and opt-in Hamming-1-filtered ANE-vs-VLM missing-number 2× zoom crop recovery (`--zoom-recover` with per-item `prompt_seq[i]`, lifting `FinTabNet` `n=6` numeric recall from `0.994` to `1.000`).
3. **Hybrid Router & Resumable Parquet Worker (`--backend auto` / `AutoOCRPipeline` / `ocr_parquet`)**:
   - Skips rasterization on clean native PDF text (`pypdfium2`) after stripping running stamps/page numbers, rejecting C0/C1/`\ufffd` control garble, and requiring `eff_chars > 0` on zero-image pages so vector-outlined pages still rasterize.
   - Clamps oversized PDF canvases to `--max-pdf-pixels` (`20M` default, `min_dpi=36`), and supports 2-stage low-DPI ANE packet screening (`--triage-dpi 100` on `run`, default `72` on `parquet`; `11.5×–16.6×` packet triage speedup) before full-DPI re-render of escalated pages.
   - Keeps single-column prose on fast ANE Apple Vision (`bands=3` pre-pass) and escalates tables, dense ledgers, and rotated scans to `FastOvisOCREngine` with double-buffered ANE/GPU execution and `batch_size=16` (`--batch-size`).
   - Streams `pdf_path` / `image_path` / `image_bytes` Parquet rows via `ocr_parquet` (`fast-ocr parquet`) with atomic manifest checkpoints (`manifest.json`), shard resume, and `zstd` compression.

## CLI Quickstart

```bash
# Fast standalone ANE Apple Vision -> rectangular GFM Markdown
uv run fast-ocr run document.pdf --backend apple-vision --format markdown

# Accelerated OvisOCR2-8bit with auto-crop, auto-orient, auto-deskew & Vision+Self speculative decode
uv run fast-ocr run table_scan.png --backend ovis --format markdown --strict-markdown-grammar --zoom-recover

# Explicit KV/GDN precision or full multilingual vocab override on batched directory/Parquet runs
uv run fast-ocr run scans_dir/ --backend ovis --batch-size 16 --kv-i8 q4_quarot --gdn-state-mode int8_blk32 --no-latin-only

# Fast mixed prose+table page via opt-in region routing (keeps formula pages on full-page VLM)
uv run fast-ocr run mixed_page.png --backend auto --region-routing --format markdown

# Hybrid routing across mixed digital + scanned + financial-table PDFs with 100 DPI ANE triage
uv run fast-ocr run packet.pdf --backend auto --triage-dpi 100 --max-pdf-pixels 20000000 --json --out out_dir/

# Resumable sharded Parquet OCR with host-budget cap (19.2 GiB default on 24 GiB hosts)
uv run fast-ocr parquet inputs.parquet out_ocr/ --backend auto --shard-rows 64 --json

# Compare backends and print per-stage timings + Markdown validation
uv run fast-ocr bench sample.png --backend all --warmup 1 --repeat 3 --json
```

## Benchmark highlights

Measured on Apple M4 Pro (`48 GB`, `macOS 26.6`) and Base Apple M5 (`24 GB`, `macOS 27.2.1`) in forced VLM mode (`--ocr-backend ovis --ocr-render-only`) alongside `auto` and standalone ANE paths. [Cross-silicon OCR & NAX tables](../../docs/BENCHMARKS.md#13-vision-ocr-category-5--apple10-nax-prefill-validation-2026-11-23) · [Research & architecture spec](docs/OCR_RESEARCH_AND_ARCHITECTURE.md) · [Category 5 benchmark guide](docs/OCR_BENCHMARKS_AND_BASELINES.md)

| Suite / Workload | Engine | M4 Pro (48 GB) | Base M5 (24 GB) | Notes |
| :--- | :--- | ---: | ---: | :--- |
| **`olmOCR-bench(4)`** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.8438` (`4/4`) · `0.082` EOT (`5.75 GB`)** | **`0.8438` (`4/4`) · `0.090` EOT (`5.01 GB`)** | **`1.44×` (M4 Pro) / `2.05×` (M5) EOT vs stock `mlx_vlm OvisOCR2`** |
| **`FinTabNet-OTSL(2)`** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.8687` TEDS (`0.674 pp/s`)** | **`0.8687` TEDS (`0.733 pp/s`)** | **`1.22×` (M4 Pro) / `1.72×` (M5) `pp/s` vs stock `OvisOCR2`; `LightOnOCR-2` looped (`0.00`)** |
| **`OmniDocBench-EN(2)`** | **`fast-ocr` (`OvisOCR2-8bit`)** | **`0.9093` (`0.117 pp/s`)** | **`0.9093` (`0.124 pp/s`)** | **`1.51×` (M4 Pro) / `1.53×` (M5) `pp/s` vs stock `OvisOCR2`; `2.44×`–`2.74×` M5 vs `GLM-OCR`/`LightOnOCR-2`** |
| **Strict GFM PLD (`--strict-markdown-grammar`)** | `fast-ocr` (`vision+self`, `K=4`) | **`1.85×` decode** (`315.2 tok/s`, `6/6` parity) | **`1.144×` wall** (`2/2` parity, `0.8687` TEDS) | `DynamicTableRowGrammar.clone()` keeps PLD active under column lock; `1.11×–1.80×` on `CORD-v2` JSON |
| **`B=16` Table Batch (`FinTabNet-OTSL(16)`)** | `fast-ocr` (`OvisOCR2-8bit`) | **`1.231 pp/s` (`468.0 tok/s`, `1.663×` vs `B=1`)** | *(pending wake)* | Identical `0.8675` TEDS & `0.8945` NumF1 (`15/16` parity, `4.362 GB` phys); `B=1` PLD remains faster on long ANE prose (`olmOCR-bench`) |
| **`B=16` Mixed E2E (`auto` KV/GDN, `16` pp)** | `fast-ocr` (`FinTabNet(8)+OmniDoc(8)`) | **`42.59 s` (`0.38 pp/s`, `-9.0%` vs `46.81 s`)** | *(pending wake)* | `kv="auto"` (`q4_quarot`) + `gdn="auto"` (`int8_blk32`) + `drop_white_patches=False`: **`14/16` exact**, `median_sim=1.0000`, `min_sim=0.9271`, `16/16` valid MD |
| **Parallel ANE (`FinTabNet` `n=16`, `1024×512`)** | `AppleVisionOCR` (`workers=4`) | **`9.61 pp/s` (`1.665 s`, `1.592×`, `16/16` parity)** | *(pending wake)* | `FastOCR.ocr_images(..., batch_size=16)` runs `9.47 pp/s` with lazy `_NativeVisionRunner` reuse |
| **Adaptive 24 px + Ordered/Relock PLD** | `fast-ocr` (`OvisOCR2-8bit`) | **`1.057×–1.140×` wall (`+11.1%` dense prose)** | *(pending wake)* | `olmOCR(8)` `+0.0000`, `OmniDoc(8)` `-0.0002` (`-24%` vis_tok), `FinTabNet(6)` `0/6` shrunk; `window_relock` `10/10` bit-identical (`1.022×` interleaved ABBA) |
| **Auto Camera-Photo Deskew (`CORD-v2(8)`)** | `fast-ocr` (`deskew="auto"`) | **`+10.94%` recall (`0.809 -> 0.919`) · `1.637×` wall** | *(pending wake)* | `14.67 -> 8.96 s` (`-31%` vis_tok), rescues `0% -> 100%` tilted receipt; **`0/106` false warps** on flat scans/PDFs/tables |
| **Tiny table crops (`FinTabNet-OTSL(2)`)** | `AppleVisionOCR` (`ANE`) | **`0.6121` (`7.03 pp/s`)** | **`0.5914` (`25.16 pp/s`, `14.88` EOT)** | `_prepare_vision_image` upscaling fixes macOS 27.2.1 tiny-crop drops (`0.0000 -> 0.5914`) |

## Scope & License

**Michael Deeb** ([`@mikewolfd`](https://github.com/mikewolfd)) · **TealWolf Consulting LLC** · [MIT License](../../LICENSE).
