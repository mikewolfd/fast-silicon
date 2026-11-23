# Vision, Multimodal Embeddings & OCR Playbook (M4 & M5+)

Actionable engineering patterns, cross-silicon measurements (`M4 Pro Apple9` vs. `Base M5 Apple10 NAX`), and public evaluation gates for PDF/image OCR, Qwen3.5 VLM decoding, and multimodal visual embeddings on Apple Silicon.

---

## 1. Three-Tier Heterogeneous PDF & Image Routing

Never run a VLM on clean digital PDFs or clean single-column prose. Route every page to the cheapest tier that preserves downstream accuracy:

```text
PDF Page ─► Tier 0: Native Text & Image-Object Audit (`pypdfium2`, 0.008–0.023 s/p)
              ├─ clean & (eff_chars >= 80 or (eff_chars > 0 and image_count == 0)) ─► Native Markdown
              └─ otherwise ─► Adaptive DPI Raster (`max_pixels = 20M`, `min_dpi = 36`)
                                └─► Tier 1: Zero-MLX Apple Vision ANE (`triage_dpi = 100` optional)
                                      ├─ clean prose (`needs_vlm == False`) ─► XY-Cut GFM Markdown
                                      └─ tables / high digits / low conf / rotated ─► Full-DPI Re-Render + Tier 2 VLM
```

### Tier 0: Native PDF Hygiene (`pypdfium2`, `0.008–0.023 s/p`)
- **Boilerplate & control-garble filtering before length checks (`preprocess.py`, `pipeline.py`):**
  - Strip standalone running headers, footers, page numbers, divider rules, short serial codes, date lines, and typesetting stamps (`VerDate ... Frm ... Fmt ...`) via `strip_running_pdf_boilerplate` before computing `eff_chars` (`native_body_char_count`), so stamped blank/scanned pages do not masquerade as rich digital text.
  - Count non-whitespace C0/C1 control characters (`< 0x20` excl. `\t\n\r`, `0x7f..0x9f`) and `U+FFFD` replacement glyphs (`count_control_garble_chars`). Require **both** `native_control_char_count == 0` **and** `native_garbled_ratio < 0.02`.
  - Count embedded image objects via `pypdfium2.raw.FPDF_PAGEOBJ_IMAGE`.
- **Vector-outlined PDF trap:** Route natively only when `is_clean and (eff_chars >= 80 or (eff_chars > 0 and image_count == 0))`. **Never** bypass on `image_count == 0` without `eff_chars > 0` — vector-outlined text PDFs report `0` embedded images *and* `0` extractable text chars and must rasterize to OCR.
- **Adaptive DPI cap:** Clamp render DPI (`compute_adaptive_render_dpi`) to `DEFAULT_MAX_PDF_PIXELS = 20_000_000` (with `min_dpi = 36` floor) before `pypdfium2` rasterization so wide-format charts/maps do not spike raster or ViT memory.

### Tier 1: Zero-MLX Apple Vision ANE (`VNRecognizeTextRequest`)
- **Strict MLX isolation:** Keep `AppleVisionOCR` completely free of `mlx` / VLM imports (`"mlx" not in sys.modules`), starting in milliseconds at `0.82–1.0 GiB` RSS/worker and scaling across `ProcessPoolExecutor` workers (`1.99×` to `5.17 pp/s` at 4 workers on M4 Pro; `2.33×` to `12.60 pp/s` at 4 workers on Base M5; up to `17.0 pp/s` ANE ceiling @ `10–12` procs).
- **Zero-leak raw RGB CGImage path:** Render directly to raw RGB bytes (skipping PNG encode/decode saves `~52 ms/p`) and wrap via `NSData.dataWithBytes_length_` → `CGDataProviderCreateWithCFData` → `CGImageCreate` inside `objc.autorelease_pool()`. This eliminates PyObjC `CGDataProviderCreateWithData`'s `~10.2 MB/page` leak (`-> ~0.2 MB/page`).
- **Two-stage `triage_dpi=100` packet screening:** Screening PDFs at `triage_dpi=100` (`850×1100` letter) on ANE achieves `word_jaccard = 0.938–1.000` vs `200 DPI` with identical `needs_vlm` routing decisions, cutting `10`-page scanned packet latency from `44,762 ms` (blind `200 DPI` VLM) to **`2,694 ms` (`16.62×` faster)** (`11.47×–14.86×` across `10–11`-page packets: `2.67–3.37 s` vs `38.7–39.7 s`). Re-render at `200–300 DPI` **only** for pages escalated to Tier 2.
- **`bands` & `language_correction` split:**
  - *Standalone prose OCR (`backend="apple-vision"`):* Default `bands=1` + `language_correction=True` (`104.0 ms` M5 / `223.9 ms` M4 Pro on clean prose). When targeting acronym/citation-heavy scans where Vision's LM corrupts identifiers, `--no-language-correction` cuts 200-DPI runtime (`0.36 s -> 0.19 s`, `293.5 ms` vs `792.4 ms` Page A `b1`) without hurting digit accuracy.
  - *Auto escalation pre-pass (`backend="auto"`):* Default **`bands=3` (`2%` strip overlap deduplication) + `language_correction=False`**. Three overlapping strips recover dense tight lines where single-band Vision collapses: on `Page B`, `bands=3` raised `table_score` `0.3 -> 1.0` (catching escalation); on `Page C` (rotated ledger), `bands=3` lifted token recall `89.9% -> 98.6%` (`32/45 -> 38/45` Base M5; `44/45 -> 45/45` upright M4 Pro), while `b3_lc0` stayed `369–464 ms` on M4 Pro and `120.4 ms` on Base M5.
- **Unambiguous 360° orientation (`0 / 90 / 180 / 270` CW):** Extract pixel-scaled baseline vectors `(tr.x - tl.x) * W, (tr.y - tl.y) * H` from `topLeft` / `topRight` of `VNRecognizedTextObservation` rather than axis-aligned `h / w` ratios (which cannot distinguish `180°` or `270°` vs `90°`). Achieved `8/8` exact `0/90/180/270` detection (`Page C` `90° CW` lift: `44/45 -> 45/45` amounts).
- **OS, revision provenance & tiny-crop upscaling:** Always log `/System/Library/CoreServices/SystemVersion.plist` (`ProductVersion`, `ProductBuildVersion`) alongside `VNRecognizeTextRequest.revision()` in result metadata. Behavior shifts materially across OS builds: on `macOS 27.2.1` (`26B131`, `rev=3`), ANE Vision ran **`4.71×–5.18×` faster** than `macOS 26.6` (`25G5057c`, `rev=3`) and improved `OmniDocBench-EN` (`0.2666 -> 0.3169`), yet dropped/merged lines on un-upscaled tiny `<600 px` table crops (`258×61` and `339×106 px`: `26->13` and `17->7` garbled lines; `0/2` `FinTabNet-OTSL` tables `0.5161 -> 0.0000`). Apply **`AppleVisionOCR._prepare_vision_image()`** (`min_long_edge=960`, `min_short_edge=240`, `max_scale=3.0` bicubic upscale; full pages untouched at `1.0×`, preserving original `(w, h)` on `PageAnalysis`), which lifted `FinTabNet-OTSL(2)` to **`0.6121` (`7.03 pp/s`)** on M4 Pro `26.6` and **`0.0000 -> 0.5914 PASS` (`25.16 pp/s`, `14.88` EOT)** on Base M5 `27.2.1` with zero full-page slowdown (`4.97` / `7.55 pp/s`).
- **Zero-latency XY-Cut & HTML→GFM canonicalization (`<1–13.5 ms`):**
  - `synthesize_markdown_from_lines` handles multi-column XY-Cut projection splits (rejecting 2-column prose/bullet lists from false table wrapping), typography headings, de-hyphenation, and rectangular GFM tables (`8.11–13.46 ms` dense, `<1 ms` typical prose).
  - `canonicalize_markdown` (`0.17–1.57 ms`) expands HTML `<table>` `colspan`/`rowspan` via an exact 2D occupancy grid (placing `<td>` spans at origin only to avoid `7×` duplication, while merging multi-row `<th>` headers), recovers unclosed trailing `<table` tags and unclosed code/math fences, and trims bloated nested HTML exports by **`88.9%–93.6%`** (`25.9–56.8 KB -> 2.88–3.64 KB`) with `100%` numeric fidelity.

### Tier 2: Compact OCR VLM Selection & Double-Buffered ANE/GPU Pipeline
Why **`OvisOCR2-8bit` (`0.85B`)** and **`NuExtract3` (`4B`)** lead local Apple Silicon OCR:
- **`OvisOCR2-8bit` (`Qwen3.5-0.8B` `3:1` GDN+GQA, `12L 88M` ViT, `1.00 GB`):** Keeps native 300-DPI resolution (`16.78M px`, `32 px` stride — zero digit blur unlike `LightOnOCR-2`'s `1540 px` cap), uses a `4.6×` lighter ViT (`88M` vs `404M` in `GLM-OCR`/`LightOnOCR-2`), holds KV+GDN cache to **`138.6 MB` @ 10K tokens (`12 KB/tok` vs `64–112 KB/tok`)**, and runs on `fast-qwen3` (`60/60` multi-cfg, `46/46` hard tables, `44/45` sideways / **`45/45` auto-oriented** amounts, `2.52–2.92 GB` peak).
- **`NuExtract3-4bit` (`Qwen3.5-4B` `3:1` GDN+GQA, `24L 306M` ViT, `3.05 GB`):** Shares the exact same `fast-qwen3` backbone and ViT class for direct schema-guided image-to-JSON (`46/46` hard tables, `15/15` @ 4K tok). Pair with `llguidance` `json_matcher` to prevent `<td></td>` loops.
- **`DeepSeek-OCR-2-4bit` (`3B` MoE `~574M` active, `<=1,121` vis toks, `8K` ctx):** Fastest raw local baseline (`3.86 s/p` median) via fixed `768²` / `1024²` crops (suited to future fixed-shape ANE ViT), but `16×` spatial compression mutates fine digits (`33/40` exact anchors).
- **Double-buffered ANE (`N+1`) behind GPU (`N`):** Dispatching `AppleVisionOCR` on batch/page `N+1` in a background thread while `FastOvisOCREngine` runs GPU ViT+prefill+decode on `N` hides **`100.0%` of ANE latency** (`0.03 ms` total exposed wait across `1,750.5 ms` of ANE work on Pages B+C, saving `1.57 s` on a 3-page batch).
- **Memory guards (`24 GB` Base M5 vs `48 GB` M4 Pro):** Keep `FastOCR` working set well under the `80%` UMA cliff (`19.2 GB` on `24 GB` hosts) via `mx.clear_cache()` between waves, `mlx_cache_limit_mb <= 256`, and capping VLM batches (`batch <= 8` on `<= 32 GB` hosts).
- **`B > 8` Vision Batching, Parallel ANE & Routing Nuances (`M4 Pro` verified):**
  - *Parallel ANE (`run_vision_batch_parallel`, `pipeline.py`):* Running `AppleVisionOCR` with `workers=4` (`min_batch=4, min_total_pixels=2_000_000`) accelerated `n=16` `FinTabNet` `1024×512` pages **`1.592×`** (`2.651s` `6.04 pp/s -> 1.665s` `9.61 pp/s`; `1.69s` `9.47 pp/s` through `FastOCR.ocr_images`, `16/16` exact text parity). Keep `_NativeVisionRunner` eligible after lazy init (only skip non-`_NativeVisionRunner` mocks) so parallelism does not silently fall back to serial after call 1.
  - *Batched VLM (`OvisOCR2-8bit` `batch_size=16`, `batched_multimodal_generate`):* Using `pack_length_waves`, `compact_hybrid_cache`, `512`-tok prefill chunking, and `kv_step <= 512` lifted `FinTabNet-OTSL(16)` (`table_mode="gfm"`) **`1.663×`** (`21.621s` `0.74 pp/s` `281.6 tok/s` -> **`12.998s` `1.231 pp/s` `468.0 tok/s`**) with identical `TEDS=0.8675` & `NumF1=0.8945` (`15/16` parity, `4.362 GB` vs `3.07 GB` phys).
  - *Routing split vs `B=1` PLD (`olmOCR-bench(8)` @ `150 DPI`):* On long dense ANE-assisted prose pages, `B=1` with `ANE+PLD` (`vision+self`) remained faster (`119.143s` `73.0 tok/s`) than `B=8` AR batched waves (`134.531s` `64.7 tok/s`, `0.886×`, identical `0.6205` macro). Route tables/short outputs through `B=16` waves and long ANE prose through `B=1` PLD.
  - *Multimodal `FastQwen3Engine.generate_batch(..., images=..., compact_active=True)`:* Verified on `CORD-v2` (`Qwen3.5-0.8B` BF16): `B=8` sped `1.11×` over `B=1` (`21.418s -> 19.295s`, `0.415 docs/s`, `4.898 GB` phys, `NOCMP == CMP 4/4`); `B=16` hit `valid_schema_rate=1.0` (`5.249 GB` phys).
- **Large-Batch `OvisOCR2` GDN / KV Memory Parity & Region/PLD Refinements (M4 Pro `P1–P4` + ABBA Probes):**
  - *Live `FP32` GDN recurrence bottleneck & `INT8 blk32` codec:* All `18` `OvisOCR2` GDN layers allocate `rec_state` in `float32` (`37.75 MB/seq/step R+W`, `~87%` decode BW at `B=64`) while `MLX 0.32.4` `gated_delta_update` downcasts `g` to `BF16`. Quantizing `rec_state` `S` in `INT8 block-32` (`3.76×` DRAM cut) with `FP32` decay `g` kept `y_cos_dec512=0.999793` across `512` single-token decode roundtrips (`2048` prefill), and combined **`whitepatch_drop + q4_quarot KV + int8_blk32 GDN`** achieved **`160/160` exact greedy parity** (`kv=1.18 MB`, `3.57×` KV cut, `1.44 GB` peak) on `FinTabNet` and **100% OCR text parity** (`kv: 38.35 -> 10.79 MB`, `3.55×` cut; diffs only 1–2 px `<img>` bbox jitter) on `OmniDocBench PPT`.
  - *Safe white-patch token dropping:* Filter ViT patch blocks strictly with `(max_ch_std < 1.5) & (min_rgb >= 250) & (edge_grad_max_rgb <= 4)` (`0` false drops across `357K` blocks, `-22.2%` tokens on `FinTabNet` -> `-20–25%` tokens).
  - *Region cropping & PLD refinements:* Set `min_pixels=1024..4096` on region crops passed to `OvisOCR2` (stock `65,536` inflates `73%` of crops); `ordered_pld` (`1.026–1.038×` lossless), `adaptive_24` (`1.057–1.140×` lossless), opt-in `--deskew` (`+10.94%` recall & `1.637×` on `CORD-v2`), `region_routing` (`1.84×` on prose+table with `allow_formula_regions=False`), and add `" " + sym` (`' ✓'`, `' ✔'`, `' ™'`) + scientific math/footnote chars (`−`, `→`, `≤`, `≥`, `†`, `μ`, `Δ`) to `140K` OCR vocab strictly when Latin-gated (`+75.6%` inflation on ZH).

---

## 2. ViT & Vision Preprocessing Optimizations (`fast-shared`)

1. **Ink-Tight Margin Cropping (`crop_to_ink`, `pad_px=24`, `min_save_ratio=0.08`):**
   - Computing foreground bounding boxes on `2×` downsampled grayscale (`< 246`) eliminates surrounding margin whitespace *before* `smart_resize_qwen`.
   - **OCR impact & loop recovery:** Cuts visual tokens `-56.3%` to `-61.4%` (`8,240 -> 3,600` Page A, `8,240 -> 3,182` Page B), speeding ViT **`4.47×–5.16×` on M4 Pro** (`7,726 -> 1,728 ms`) and **`5.68×` on Base M5** (`6,542 -> 1,152 ms`), speeding prefill `2.39×–2.44×`, and dropping single-page peak RAM (`3.01 -> 2.71 GB` M4P; `2.86 -> 2.52 GB` M5) with **exact text parity** on standard documents. **Trap on dense multi-column pages (`olmOCR-bench` page `[2]`):** tight `crop_pad_px=24` strips framing margins and can induce token-0 greedy repeat loops (`"(2007 2007 ..."`, `rep4=1.0`, `score=0.0` to `4096` toks). Rather than penalizing normal `<td>`/`<tr>` table markup with `loop_breaker=True` globally (keep `loop_breaker=False` default), pair **`detect_catastrophic_token_loop()`** (`16+` consecutive `p=1..8` cycles, `40+` for table markup) with **`auto_loop_recover=True`** to abort degenerate loops at ~`20` tokens and automatically retry uncropped (`crop_margins=False`, escalating to `_force_loop_breaker=True` only if needed) — lifting `olmOCR-bench(4)` **`3/4 (0.5938) -> 4/4 (0.8438 PASS)`** on both chips (`5.75 GB` M4 Pro / `5.01 GB` Base M5; `2.05×` EOT vs stock on Base M5).
   - **Visual embedding impact (`fast-pplx`):** Cuts dense `200 DPI` letter visual tokens `1,692 -> 1,603` (`859 -> 805 ms/p`), holds `40/40 (100%)` R@1, and improved dense-page cross-modal query cosine (`[0.4299, 0.4416, 0.5918]` vs `[0.4351, 0.4192, 0.5830]`).
2. **Uniform-Frame SDPA Fast Path & `16`-Entry Grid Geometry Cache (`FastQwen35VisionModel`):**
   - Cache bilinear `pos_embed` interpolation indices/weights and 2D ViT rotary `(cos, sin)` tables computed purely in NumPy (`_GridGeometryCacheEntry`, LRU size `16`) to avoid per-frame CPU-GPU `.tolist()` syncs.
   - Compile 2D vision RoPE in FP32 (`@mx.compile _apply_vision_rope_qk`). When batched images share `(grid_h, grid_w)`, reshape into `[n_frames, H, uniform_frame_len, hd]` for single-launch batched `scaled_dot_product_attention`. Keep `mx.split` for variable-size batched frames and plain `BF16` `nn.Linear` for ViT MLPs — Base M5 (`Apple10 NAX`) probing rejected uneven ViT blockdiag/padded SDPA (`0.21×–0.43×` slower) and ViT MPP MLP fusion (`~1.01×–1.10×`, high drift).
3. **Base M5 `Apple10 NAX` `head_dim=256` Workaround (`detect_apple10_d256_force_fused`):**
   - On `Apple10` (`M5+`) + `macOS >= 26.2` + MLX supporting `force_fused`, pass `force_fused=True` to `mx.fast.scaled_dot_product_attention` when `head_dim == 256` (`Qwen3.5` full-attention layers). Never force on M4 Pro `Apple9` (`2.65 ms` forced vs `2.50 ms` default).
4. **3D Interleaved M-RoPE & `rope_offset_delta` Decode Invariance (`qwen35_vision.py`):**
   - Match `Qwen3_5Model` interleaved `mrope_section=[11, 11, 10]` (`partial_rotary_factor=0.25`, `rot_dims=64`, `rope_theta=1e7`).
   - Compute `rope_offset_delta = max_pos_id + 1 - len(prompt_ids)` alongside `(3, 1, L)` position IDs (`compute_mrope_positions`) and pass `offset = cache.offset + rope_offset_delta` during cached decode (`FastQwen3Engine.generate(..., images=...)` & `FastOvisOCREngine`). Verified **`1.0` same-pixel ViT cosine (`max_abs=0.0`)**, **`128/128` live token match**, and **`1,503/1,503` full-page text match** vs `mlx_vlm` at **`2.11×` faster prefill** (`1,962.6` vs `4,141.8 ms`).
   - Also keep `fast_shared/__init__.py` lazy (`__getattr__`) and `vision_preprocess.py` zero-`mlx` so ANE-only callers never pay MLX load cost.
5. **ViT Precision, Lazy Loading & Preprocessing Parity:**
   - Keep **OCR ViTs in `BF16`** (not `Q8`): `Q8` ViT measured slower on both M4 Pro (`0.9686×`: `7,976` vs `7,726 ms`) and Base M5 cropped (`0.9093×`: `1,267` vs `1,152 ms`) with `cos=0.9978`.
   - Lazy-load `FastQwen35VisionModel` inside `_ensure_vision_model()` so text-only callers pay `0.00 GB` ViT RAM.
   - Watch Pillow version invariance when auditing `<=1e-3` ViT diffs: Pillow `13.0.0` vs `12.3.0` `BICUBIC` resize differs on `2.44%` of edge pixels (`max_abs=0.227`, giving `vis_cos=0.99876`).

---

## 3. VLM Decoding, Speculative PLD & Structured Grammar (`fast-ocr` / `fast-qwen3`)

1. **ByteLevel-BPE Safe OCR Vocab Remapping (`build_ocr_vocab_subset`, `248,320 -> 140,307`):**
   - Decode every candidate token (`tokenizer.decode([tid], skip_special_tokens=False)`) before checking codepoints (`< 0x0250` + extended OCR punctuation/currency/math `—–“”‘’•§¶°±×÷€£¥¢½¼¾©®™✓✔✗☐☑`), plus Vision/ChatML special tokens.
   - **Keep `target_size=None` (`140,307` tokens) for lossless OCR:** Pruning tied/full `248,320` -> `140,307` yielded **`1.10–1.12×` decode speed** (`159.1 -> 174.8–178.1 tok/s`) with **`0` missing symbols and `100.0%` exact token/text parity**. Tighter `32K / 49K / 65K` caps dropped real high-ID English tokens (`'Thousands'`, `' LABOR'`) and degraded `Page C` (`45/45 -> 44/45`). Leave full vocab intact whenever `remap.vocab_size >= full_vocab_size`.
2. **Bit-Identical Prompt-Lookup Speculative Decoding (`OCRPLDDrafter`, `vision+self`, `draft_k=4`):**
   - Index ANE Vision's recognized tokens (`2..4`-gram suffix lookup) + online self-history (capturing repeating table cells/markup), proposing `draft_k=4` tokens verified in one GPU pass via `verify_draft_sequence_b1` and zero-GEMM `DeltaSlotCache.rollback_to_accepted`.
   - **100% bit-identical** to AR greedy across all `12` tested configs: M4 Pro hits **`360.5 tok/s` (`1.943×`)** on Page A and **`436.8 tok/s` (`2.465×`)** on Page C; Base M5 hits **`282.8 tok/s` (`2.070×`)** on Page A and **`349.2–355.2 tok/s` (`2.707×–2.753×`)** on Page C.
3. **Keeping PLD Active Under Strict Grammars (`DynamicTableRowGrammar.clone()` / `LLMatcher.deep_copy()`):**
   - Do **not** disable speculative decoding whenever a stateful grammar (`--strict-markdown-grammar` or `json_matcher`) is active:
     - Clone the grammar (`DynamicTableRowGrammar.clone()` via `LLMatcher.deep_copy()`).
     - Prefix-filter draft tokens (`_filter_table_safe_drafts`) so every drafted token is grammar-legal and halts *before* any boundary token that triggers `compute_ff_tokens()` (`| --- | ... |\n` separator inserts cleanly in `0` forward passes on the verification bonus token).
     - Advance the cloned `spec_gram` across verify steps, then commit accepted tokens to the live `table_gram`.
   - Verified **`1.85×` decode speedup** (`170.0 -> 315.2 tok/s`; `1.80×–2.24×` per table) with **`6/6` exact parity** on M4 Pro `FinTabNet (n=6)`, **`1.144×` wall speedup** (`4,154.7 -> 3,632.8 ms`) with **`2/2` exact parity** on Base M5 `FinTabNet (n=2)`, and **`1.11×–1.80×` `tok/s`** with exact parity on `CORD-v2 (n=4)` under JSON-schema `llguidance` PLD (`_filter_json_safe_drafts` + post-verify `ff` sync + zero-cost `metadata["field_boxes"]` via `align_json_values_to_lines`).
4. **Opt-In ANE-Guided Missing-Number Zoom Recovery (`--zoom-recover`, `numeric_recovery.py`):**
   - Compare ANE Vision numbers against VLM output (`find_filtered_missing_numbers`), filtering out single-digit tokens, `,` vs `.` decimal punctuation mismatches (`2,016` vs `2.016`), substrings merged into digit runs (`19.29` in `1929`), and same-length Edit-Distance-1 ANE blurry digit substitutions (`_is_single_digit_substitution`), then crop ranked rotation-aligned ANE band boxes (`2×` Bicubic if `<400 px`) for a single-crop `self`-PLD re-read.
   - Lifted `FinTabNet (n=6)` numeric recall **`0.994 -> 1.000`** (`FT[1] 0.990 -> 1.000`, `FT[4] 0.973 -> 1.000`), cutting naive-diff latency `-21.5%` (`6,651 -> 5,223 ms`); Edit-Distance-1 pruning (`_is_single_digit_substitution`) eliminated `FT[5]` wasted zoom (`0 ms`) and lifted `FT[3]` `0.709 -> 0.772`.
5. **`@mx.compile` & Apple10+ NAX `W4A8` / `W8A8` Prefill in `fast-qwen3` (`kernels.py`, `_mpp_linear.py`):**
   - `@partial(mx.compile, shapeless=True)` on `RMSNormGated` and `normalize_gdn_qk` + `@mx.compile` on `apply_partial_rope_trig` measured `0.0` `max_abs_diff` and bit-exact tokens at **`1.053×–1.303×` on M4 Pro** and **`1.038×–1.440×` on Base M5**.
   - On `Apple10` (`Base M5`), `_mpp_linear.py` gates `64×64s4` MPP prefill behind `should_use_qwen_mpp_linear` + shared `prequantized_x` (`MppBucketedLinearGroup`) + a `28%` RAM pack budget guard (falling back to native `qmm` if host available RAM drops below threshold, and preserving `.vjp` for LoRA `mx.value_and_grad`):
     - **`W4A8` (default on `Apple10`, `M >= 128`, `k,n >= 1536`, `k*n >= 8M`):** `1.06×–1.10×` (`L=512` prefill) and **`+4.6%` EOT** (`107.28 -> 102.61 s`, exact `0.8857` parity on `Qwen3.5-9B` Cat 4).
     - **`W8A8` (opt-in via `mpp_w8a8=True` / `FAST_SILICON_QWEN3_MPP_W8A8=1`, `M >= 128` for `OvisOCR2-8bit` `1024×7168` / `3584×1024`):** `1.29×–2.48×` micro (`cos >= 0.999974`), `1.14×–1.18×` `4B/9B` prefill (`+8.3%` EOT on `9B` Cat 4 within `-0.00153` SE parity), and reduced `OvisOCR2-8bit` `cord_v2 (n=6)` `LM sum_prefill` `909.8 -> 845.6 ms` (`w8_calls=594`).
6. **Plain-Text Post-OCR Polish Guardrail:**
   - Never run unconstrained fine-tuned text-only Markdown rewriters blindly (they silently omit entities/names and fail flattened multi-column vote/financial grids). If polishing clean prose, gate `Qwen3-VL-4B-Instruct-4bit` (`70.7 tok/s`) with a deterministic token-diff check allowing only whitespace, de-hyphenation, and common OCR character repairs; keep tables on image/coordinate extraction.

---

## 4. Multimodal Visual Embeddings (`fast-pplx` Visual Bridge)

`FastPplxGPU.encode_images` / `FastPplx.encode_images` / `FastPplx.build_image_index` (`fast-pplx embed-images` w/ `rotations_cw` / `--rotate-cw` and `--build-index` / `--query` / `--top-k`) shares `FastQwen35VisionModel` + `prepare_pplx_image_item` (fractional 3D MRoPE coordinates, `248078 [D]`, `248079` row-band separators, `chunk_token_spans`) and sorts batches by `(grid_h, grid_w, len(input_ids))` to eliminate padding across matched pages.

- **Exact HF PyTorch Parity:** `input_ids_exact=True`, `max_pos_abs_diff=0.0`, `spans_exact=True`, and **`1.000000` cosine parity** across `90/180/270` CW rotations (`normalize_rotations_cw`); across `n=6` (`24` bands vs HF MPS `bf16`): `min_band_cos=0.99912`, `mean_band_cos=0.99969`, `int8_cos_min=0.99908` (`MAE 0.29–0.95`), query `min_cos=0.99980`.
- **Cross-Silicon Throughput & Memory (`chunks=8`, `fp16`):**
  - `100 DPI`: **`159.12 ms/p`** (`6.28 pp/s`, `mean_cos=0.999698`, `1.87 GB` peak) on M4 Pro; **`109.24 ms/p`** (`9.15 pp/s`, `mean_cos=0.999699`, `1.72 GB` peak) on Base M5 (vs `573.40 ms/p` PyTorch MPS).
  - `150 DPI`: **`688.48 ms/p`** on M4 Pro; **`424.87 ms/p`** on Base M5 (`mean_cos=0.999825–0.999838`).
- **Retrieval & `crop_margins=True` Gains:** On `n=40` (`15` tables + `15` receipts + `10` `200 DPI` letters, `bf16`), cropped visual embedding ran `390.6 ms/p` (`3.16×` vs `1,232.8 ms/p` HF MPS) at **`40/40 (100.0%)` R@1** (beating `AppleVisionOCR + text` `39/40 = 97.5%` @ `203.3 ms/p`). On a `10`-page scanned packet at `100 DPI` cropped (`275.9 ms/p`), multi-band `MaxSim` via `CompactRetrievalIndex` (`build_image_index`) scored **`10/10` intra-packet R@1** (`6/6` R@1 in live smoke).
- **Scope Rule:** Never use visual embeddings for bulk text-corpus indexing (`fast-pplx` extracted text runs `69.5 ms/p` M4 Pro / `46.0 ms/p` Base M5 — `7.8×–27.8×` faster than MPS visual embedding — and supplies exact BM25 tokens and highlight spans). Reserve `encode_images` as a targeted supplement for low-text diagrams, charts, and scanned visual exhibits.

---

## 5. Cross-Silicon Receipts: M4 Pro (`Apple9`) vs. Base M5 (`Apple10 NAX`)

1. **Compute-Bound ViT/Prefill (`M5 NAX` wins `1.3×–1.65×`) vs. DRAM-Bound Decode (`M4 Pro` wins `1.22×–1.26×`):**
   - `FastOCR(backend="ovis")` Wave-3 defaults (`crop_margins=True`, `auto_orient=True`, `speculative="auto"`, `draft_k=4`, `140,307` vocab, `BF16` ViT):
     - **Page A (`3,600` vis toks, `3` GFM tables):** M4 Pro `7,350.3 ms` (`1,732.0` ViT / `808.3` pref / **`345.4 tok/s`**, `2.79 GB`) vs Base M5 **`7,249.9 ms`** (**`1,139.3` ViT `1.52×`** / **`541.4` pref `1.49×`** / `282.4 tok/s`) — vs stock `mlx_vlm` on M4 Pro (`7,617.8` ViT / `4,141.8` pref / `9.27 GB`).
     - **Page B (`3,182` vis toks, `1` table):** M4 Pro `5,961.6 ms` (`1,451.9` ViT / `715.5` pref / `287.1 tok/s`, `2.78 GB`) vs Base M5 **`5,040.5 ms`** (**`880.3` ViT `1.65×`** / **`469.6` pref `1.52×`** / `269.4 tok/s`, **`1.18×` E2E win**).
     - **Page C (`90° CW`, `6,528` vis toks, `45/45` amounts):** M4 Pro **`15,062.2 ms`** (`4,999.3` ViT / `1,511.8` pref / **`438.7 tok/s`**, `2.92 GB`) vs Base M5 `15,103.9 ms` (**`3,876.6` ViT `1.29×`** / **`956.8` pref `1.58×`** / `348.9 tok/s`).
     - **`auto` 4-Page Mix (`[prose, A, B, C]`):** M4 Pro `29,422.1 ms` (`prose` on ANE `169.8 ms`) vs Base M5 **`28,218.9 ms`** (`prose` on ANE `120.4 ms`), `4/4 valid=True`, `45/45` Page C on both chips.
2. **Category 5 Smoke Comparison (`--ocr-backend ovis --ocr-render-only`, `12/12 PASS` on Both Chips):**
   - **`fast-ocr` accelerates on Base M5 while cutting memory vs M4 Pro, whereas stock `mlx_vlm` slows down `0.64×–0.81×`:**
     - `olmOCR-bench(4)` (w/ `auto_loop_recover=True`): **`4/4 (0.8438 PASS)`** on both chips — M4 Pro `EOT=0.082` (`323.3 dec / 210.5 E2E tok/s`, `5.75 GB`, `1.44×` vs stock `0.057`) vs Base M5 **`EOT=0.090`** (`286.1 dec / 232.7 E2E tok/s`, **`5.01 GB`**, **`2.05×` vs stock `0.044`**, `1.11×` vs M4 Pro). *(Before loop recovery: `0.5938` `7.82 / 6.61 GB`.)*
     - `FinTabNet-OTSL(2)`: `0.674 -> 0.733 pp/s` (`1.09×`, `3.95 -> 3.80 GB`, `TEDS=0.8687 PASS`) vs stock `OvisOCR2` `0.628 -> 0.436 pp/s` (`0.69×`).
     - `OmniDocBench-EN(2)`: `0.117 -> 0.124 pp/s` (`1.06×`, `7.74 -> 6.75 GB`, `S=0.9093 PASS`) vs stock `OvisOCR2` `0.078 -> 0.081 pp/s`.
   - Net Base M5 speedup: `fast-ocr` leads stock `mlx_vlm OvisOCR2` by **`1.53×–2.05×`** (`pp/s`) and `GLM-OCR` / `LightOnOCR-2` by **`2.17×–3.71×`**.
   - **`macOS 27.2.1` ANE Vision divergence (`public-apple-vision`):** `4.71×–5.18×` faster (`0.965 -> 4.996 pp/s` `olmOCR`, `8.300 -> 39.103 pp/s` `FinTabNet`, `1.519 -> 7.431 pp/s` `OmniDocBench`), lifting `OmniDocBench-EN` (`0.2666 -> 0.3169`); `_prepare_vision_image()` tiny-crop upscaling fixed `FinTabNet-OTSL(2)` on **both** chips (`0.5161 -> 0.6121 PASS` @ `7.03 pp/s` M4 Pro; **`0.0000 -> 0.5914 PASS` @ `25.16 pp/s`, `14.88` EOT** Base M5) with zero full-page penalty (`4.97` / `7.55 pp/s`).

---

## 6. Public OCR Benchmark & Validation Protocol (`Category 5`)

All `8` pinned Category 5 datasets in `configs/outcome_benchmarks.lock.json` (`24` total datasets) and the paired `us_gov_ocr` corpus run pure Python + NumPy evaluators (`fast_shared/outcome_eval/ocr.py`, `gov_corpora.py`, and `scripts/benchmark_outcomes.py`, zero `datasets` / `torch` / `scipy` / `rapidfuzz` / TeX binary deps):

| Suite (`Repo / Source`) | License | Tier-1 / Tier-2 | Deterministic Metrics (`s_i ∈ [0, 1]`) & Validity (`v_i`) |
|---|---|---|---|
| **`olmocr_bench`** (`allenai/olmOCR-bench`) | `ODC-BY` | `42` / `210` pp | Binary pass rate across `present`, `absent`, `order`, `table`, canonicalized LaTeX `math`, and `baseline` + empty guard (`overall` & `overall_no_hf`). |
| **`fintabnet_otsl`** (`docling-project/FinTabNet_OTSL`) | `CDLA-Permissive-1.0` | `30` / `150` tbls | **TEDS** (Zhang–Shasha tree edit similarity, lifting GFM pipes → canonical HTML), **TEDS-S**, **GriTS-Top**, **GriTS-Con**, **Numeric Cell F1** (`($1,250) -> -1250`). |
| **`omnidocbench_en`** (`opendatalab/OmniDocBench`) | `Research-NonCommercial` (`code Apache-2.0`) | `24` / `120` EN pp | Composite `mean(1 - Text_NED, Reading_Order_Sim, Table_TEDS, Formula_NormSim)`; filtered to `language == "english"` for Latin-remapped vocab validation. |
| **`cord_v2`** (`naver-clova-ix/cord-v2`) + visual **`ExtractBench`** | `CC-BY-4.0` / `Apache-2.0` | `30` / `100` rcpts | `grounded_field_f1` (Hungarian array matching + Draft-2020-12 JSON Schema) + leaf **ANLS**. |
| **`vrdu_ad_buy`** / **`vrdu_registration`** (`google-research-datasets/vrdu`) | `CC-BY-4.0` | `20` / `100` each | Type-normalized `grounded_field_f1` + leaf **ANLS** on human KIE GT aligned to primary annotated page (`ocr` Cloud OCR tokens ignored). |
| **`doclaynet_gov`** (`docling-project/DocLayNet-v1.2`) | `CDLA-Permissive-1.0` | `0` / `90` pp | `doclaynet_overall` (`1 - Text_NED`, reading-order concordance, header/footer leak rate) across `laws_and_regulations`, `government_tenders`, `patents` (EN/Latin filtered). |
| **`puredocbench_legal_gov`** (`zhihengli-casia/puredocbench`) | `CC-BY-4.0` / `MIT` | `0` / `0` opt-in | `puredocbench_overall` (`03_legal_gov` English pages across `clean`, `digital`, `real` tracks with `--puredocbench-images-root`). |
| **`us_gov_ocr`** (GovInfo `PLAW`/`BILLS`/`USCOURTS`, `FR` API, optional LoC `btp-dataset`) | `U.S. Public Domain` / `CC0` | `stage-gov` | `us_gov_overall` (`OmniDocBench` metrics + per-branch & per-source breakdowns; auto-selects `<GPOTABLE>` page on `FR`). |

**Mandatory Validation Rules:**
1. **Isolate Visual OCR vs. PDF Routing:** Run `--ocr-render-only` (`use_native_text=False`) on identical rendered PIL images when checking lossless parity vs stock `mlx_vlm`, and run `use_native_text=True` separately to measure end-to-end Tier-0/1/2 PDF `EOT`.
2. **Enforce Structural & Loop Gates:** Validate every output with `canonicalize_markdown` + `validate_ocr_markdown` (`repeat_4gram_rate < 0.15`, rectangular GFM tables, balanced fences/math/HTML) so silent repetition loops (`repeat_loop_count > 0`) fail Promotion Gate 1 immediately.
