# OCR Research, Multi-Round Benchmarks & `fast-ocr` Architecture Specification

> **Document Scope:** Synthesis of 12 real-world document OCR evaluation campaigns (financial disbursement ledgers, regulatory forms, multi-column dense budgets, scanned vs. digital prose), exact architecture forensics of the Top 5 open-weights local OCR VLMs (+ honorable mentions and `PaddleOCR-VL-1.6` / `RapidOCR` analysis), our long-term theorized optimal OCR encoder-decoder architecture, and the zero-training `fast-ocr` production stack built on Apple Silicon (`M4 Pro` / `M5`).

---

## 1. Multi-Round Empirical OCR Benchmark Synthesis

Across 12 benchmark campaigns covering scanned financial ledgers, regulatory filings, multi-column hearings, and dense tables (both upright and 90°-rotated scans), open-weights local models separated clearly:

### 1A. Top 5 Open-Weights Local OCR VLMs

| Rank | Model & Tested MLX Precision | Key Measured Accuracy | Measured Speed & Memory (M4 Pro) | Verdict & Known Weakness |
|---:|---|---|---|---|
| **1** | **`OvisOCR2`** (`ATH-MaaS/OvisOCR2` / `mlx-community/OvisOCR2-8bit`, `1.00 GB`) | • **60/60** 4-pg multi-cfg suite<br>• **46/46** hard budget/hearing rows (byte-identical M4/L40/A100)<br>• **#1 local** on sideways financial ledger: **44/45** amounts (`+0` extra), **68/80** financial positions, **34/35** blanks, **243/244** literals | • `14.3 s/p` med conv (`22.1 s/p` warm hard)<br>• Sideways ledger: `39.0s` conv, **`133.0 tok/s`**, **`2.71 GB`** peak MLX *(L40 `9.07s` / A100 `8.59s`)* | **Undisputed #1 local OCR VLM.** Only miss on sideways ledger was omitting final grand total row and col-count mismatch between summary & detail rows. |
| **2** | **`GLM-OCR`** (`zai-org/GLM-OCR` / `mlx-community/GLM-OCR-bf16`, `2.21 GB`) | • **#1** in 25-model Docling sweep (`16/16`, **11/11** raw & export)<br>• **40/40** & **11/11** 10-pg suite<br>• **45/46** hard budget rows<br>• **INCOMPLETE** on 6.6M-px sideways ledger (`0/80` pos) | • `13.2 s/p` med sweep (`25.0s` hard)<br>• Sideways ledger: `241.3s` (`16K` cap loop), `76.2 tok/s`, `4.18 GB` | Strongest upright region/page table reader, but entered an infinite 6-row loop on oversized 90°-rotated input (exceeded 2.62M-px/region recipe) and lacks MTP in `mlx_vlm`. |
| **3** | **`LightOnOCR-2-1B`** (`lightonai/LightOnOCR-2-1B` / `mlx-community/LightOnOCR-2-1B-bf16`, `2.01 GB`) | • **#1 local** on regulatory filings (`4/8` fail vs `8/8` GLM/MinerU)<br>• **40/40** & **11/11** 10-pg suite; `8/11` Docling<br>• Sideways ledger: loop-free (`stop`), **`27/45` amounts**, `8/80` pos | • **`8.64 s/p`** med 10-pg (`9.99s` fixtures)<br>• Sideways ledger: `51.1s` conv, `94.3 tok/s`, **`6.57 GB`** | Fast, loop-free prose/form generalist (RLVR prevented loops), but hard `1540 px` longest-edge cap blurs small 7–8pt digits by `2.12×` and `28L` pure-GQA KV explodes peak RAM. |
| **4** | **`NuExtract3`** (`numind/NuExtract3-mlx-4bits`, `3.05 GB`) | • **Tied #1 (`46/46`)** hard tables (Markdown & native JSON)<br>• **15/15** at 4K tok (`54/60` overall)<br>• **TIMED OUT** sideways (`29/45` partial, empty `<td></td>` loop) | • `22.6 s/p` med; `87.2s` hard MD / `55.7s` JSON<br>• Sideways: `>254.9s` (`7.81 GB` phys) | Same `Qwen3.5` hybrid GDN architecture as `OvisOCR2` scaled to `4B`; high structured accuracy, ~3–4× slower without grammar constraints. |
| **5** | **`DeepSeek-OCR-2`** (`deepseek-ai/DeepSeek-OCR-2` / `mlx-community/DeepSeek-OCR-2-4bit`, `2.56 GB`) | • **40/40** complete, **11/11** relationships in all 4 configs<br>• `33/40` exact anchors (LaTeX `\(` delimiters + 1 unscored digit mutation) | • **`3.86 s/p`** baseline median *(fastest high-accuracy local VLM)* | Ultra-fast via strict `<=1,121` compressed visual tokens & shallow `12L` MoE, but aggressive `16×` compression + `8K` context drops fine character/number fidelity. |
| *#5B* | *`Nemotron-Parse-2.0`* *(honorable mention, `8-bit`)* | • **`11/11` raw relationships** in Docling sweep; export fell to `5/11` solely due to a Docling `\multicolumn`/`\multirow` parser bug | • `15.25 s/p` cold med (`15.0s` @ 300 DPI) | True `C-RADIOv2-H` + `10L` mBART Seq2Seq encoder-decoder; `1×4` horizontal neck preserves every text line cleanly. |

*(Frontier cloud APIs tested for ceiling reference on the 5-page disbursements suite: `gemini-4-argon` medium hit `535/535` amounts, `640/640` positions, `572/572` literals at `~$0.033/page`; `gpt-6.1-terra` medium hit `535/535`, `640/640`, `299/299` oriented at `~$0.017/page`; `gemini-3.1-pro-preview` low ran fastest among 100%-accurate APIs at `13.5 s/p`.)*

### 1B. What About `PaddleOCR-VL-1.6` and Classical `RapidOCR`?
Both were extensively benchmarked and **explicitly eliminated from the Top 5 VLM tier** (with one narrow role kept for classical `RapidOCR`):
- **`PaddleOCR-VL-1.6` (`jmbarrancoidener/PaddleOCR-VL-1.6-mlx-8bit`, `1.10 GB` + CPU pipeline):** Tested across 4 campaigns. While ultra-fast per generated token in MLX (**`151.7 tok/s`**, `5.1 s/p` conv at `2.07 GB` peak RAM), it suffered severe factual failures:
  - Regulatory filings (14 pp CPU): `64/87` Markdown (`80/87` layout), omitted monetary regions and invented years in blank fields (`562 s`).
  - 10-pg suite (CPU): `38/40` anchors, `10/11` rels (`61.7 s/p`).
  - 4-pg hard CT suite (MLX 8-bit): `41/60` relationships (`10–11/15`) vs `OvisOCR2`'s `60/60` — silently replaced real budget numbers (`22%`, FTE `9`) with `N/A`, and looped to the token cap at 300 DPI.
  - Sideways 6.6M-px ledger (MLX 8-bit): entered a `16,384`-token repetition loop (`123.6 s`), recovering only **`6/45` amounts** (`13%`) and **`0/80` financial positions**.
- **Classical `RapidOCR` (`PP-OCRv4` / `PP-OCRv6_small` ONNX):** Scored **`300/300` (`100%`) raw monetary amounts** across 4 oriented hard pages (`1.3 s/p` standalone) — strong as a zero-cost numeric cross-check on upright scans, but poor on table structure (`275/357` positions, `0/57` explicit blanks), drops footnotes under layout detectors (`0.904` vs native), and merges prose words (`2/40` exact anchors in 10-pg test).

### 1C. Other Local VLMs & Classical ANE/CPU Baselines
- **`Qwen3-VL-4B-Instruct` (`4-bit`, `3.09 GB`):** `35/46` hard tables (`63.7 s`, shifted 11 dense-budget rows 1 column left), `44/45` raw amounts sideways but hallucinated **`+32` extra repeated amounts** (`111 s`, `0/80` pos). Strong instead as a text-only post-OCR refiner (`70.7 tok/s`).
- **`MinerU2.5-Pro-1.2B`, `Granite-Docling-258M`, `SmolDocling-256M`, `olmOCR-2-7B`, `TeleOCR`, `dots.mocr`, `Falcon-OCR`:** All eliminated due to small-box gates dropping amounts (`MinerU`), severe under-capacity on dense tables (`3/11`–`4/11` rels on `258M`/`256M`), broken frontmatter parsers & fiscal misalignment (`olmOCR-2`), MPS/Transformers incompatibilities (`TeleOCR`, `dots.mocr`), or 100% token-cap stalls (`Falcon-OCR`).
- **Independent Classical ANE / CPU Baselines (`fast-ocr`'s fast path):**
  - **`Apple Vision` (`VNRecognizeTextRequest`, accurate, ANE):** `0.36–0.47 s/p` single worker, **`17.0 pp/s` ANE multi-process ceiling** (`0.82–1.0 GiB` RSS/worker); `0.983` agreement vs native PDF on 100-page archive set and **`0.966` word recall on 35 scans** (with 3-band strip deduplication recovering tightly-set digital lines where whole-page Vision dropped to `0.803`). Inadequate on dense multi-column tables (`5/11`–`0/80` cell positions).
  - **`PyMuPDF` / `pypdfium2` Native Text:** `0.008–0.023 s/p` (`100×` faster than OCR) on the ~70% clean digital pages.

---

## 2. Exact Architecture Forensics of the Top 5 (+ `#5B`)

| Spec | #1 `OvisOCR2` | #2 `GLM-OCR` | #3 `LightOnOCR-2-1B` | #4 `NuExtract3` | #5 `DeepSeek-OCR-2` | Bonus `#5B` `Nemotron-Parse-2.0` |
|---|---|---|---|---|---|---|
| **Base Class** | `Qwen3_5ForConditionalGeneration` (0.8B) | `GlmOcrForConditionalGeneration` (0.5B Dec) | `LightOnOCR` (`mistral3`: Pixtral + Qwen3-0.6B) | `Qwen3_5ForConditionalGeneration` (4B) | `DeepseekOCR2` (SAM-B + Qwen2-0.5B + DS-3B-MoE) | `NemotronParse` (C-RADIOv2-H + mBART) |
| **Total / Active Params** | **853.0M** / 853.0M | **1,107.4M** *(1,143M w/ HF MTP)* | **1,005.6M** / 1,005.6M | **4,539.3M** / 4,539.3M | **3,389.1M** / **~574.1M** decode | **903.1M** tied *(977M Q8 untied)* |
| **Param Split** *(Vis / Proj / Emb / Dec / Head)* | **88M** (10%) / 13M / **254M** (30% tied) / 498M | **404M** (37%) / 30M / 91M / 491M / 91M | **403M** (40%) / 6M / 156M (tied) / 440M | **306M** (7%) / 27M / 636M (tied) / 3,570M | **453M** (SAM 95+Q2 358) / 1M / 165M / 2,604M (409M act) / 165M | **652M** (72%) / 9M / 74M (tied) / **168M** (19%) |
| **Vision Encoder** | **12L** `h=768`, 12H `hd=64`, `p=16`, **full attn**, 2D-RoPE + `48²` abs | **24L** CogViT `h=1024`, 16H `hd=64`, `p=14`, **full attn**, QK-Norm, 2D-RoPE | **24L** Pixtral `h=1024`, 16H `hd=64`, `p=14`, **full attn**, 2D-RoPE | **24L** `h=1024`, 16H `hd=64`, `p=16`, **full attn**, 2D-RoPE + `48²` abs | **DeepEncoder V2**: 12L SAM-B (`p=16`, 8 win-14 + 4 global, 2× stride-2) → 24L Qwen2-0.5B Causal Flow | **32L** C-RADIOv2-H (`h=1280`, 16H **`hd=80`**, `p=16`, **full attn** over 13,320 toks) |
| **Connector** | `2×2` concat → LN → 2L GELU MLP | `2×2` Conv2d (`1024→1536`) → SwiGLU | RMSNorm → `2×2` linear → 2L GELU + row breaks | `2×2` concat → LN → 2L GELU MLP | Linear (`896→1280`) + `<|view_separator|>` | `1×1` Conv1d + **`1×4` horiz Conv2d** (`3,329` toks) |
| **Resolution Cap & 300-DPI Letter Tokens** | **`16.78M px`** (`32px` stride) → **8,240 toks** | Eff. **`4.82M px`** (`t_bar=2`, `28px`) → **6,052 toks** | **`longest_edge=1540 px`** (`28px`) → **2,420 toks** | **`16.78M px`** (`32px` stride) → **8,240 toks** | Global `1024²` (`256`) + `0..6` `768²` (`144`) → **cap `1,121` toks** | Fixed **`2048×1664`** padded canvas → **`3,329` enc toks** |
| **Decoder Architecture** | **24L 3:1 Hybrid**: 18 **GDN** + 6 GQA (`8Q:2KV`, `hd=256`, gate, QK-Norm) | **16L Dense GQA**: `16Q:8KV`, `hd=128`, 4× Sandwich RMSNorm, **no QK-Norm** | **28L Dense Qwen3 GQA**: `16Q:8KV`, `hd=128`, QK-Norm | **32L 3:1 Hybrid**: 24 **GDN** + 8 GQA (`16Q:4KV`, `hd=256`, gate, QK-Norm) | **12L MoE (`use_mla=False`)**: `10Q:10KV` MHA (`hd=128`), 64 routed (`k=6`) + 2 shared | **10L mBART Seq2Seq**: `16Q:16KV` MHA (`hd=64`), cross-attn, **NoPE** |
| **Pos / Ctx / Vocab** | Interleaved MRoPE `0.25`, `θ=1e7` / `262K` / **`248,320`** | Split MRoPE `1.0`, **`θ=10k`** / `131K` / `59,392` | 1D RoPE, `θ=1e6` / `16K` / `151,936` | Interleaved MRoPE `0.25`, `θ=1e7` / `262K` / **`248,320`** | 1D RoPE, `θ=10k` / **`8,192`** / `129,280` | **NoPE** / `9,000` / `72,256` |
| **BF16 Cache @ 10K** | **`139 MB`** (`12 KB/tok` + `19M` GDN) | `640 MB` (`64 KB/tok`) | **`1.12 GB`** (`112 KB/tok`) | `369 MB` (`32 KB/tok` + `49M` GDN) | `60 KB/tok` (`<=66M` prefill KV; 8K max) | `40 KB/tok` gen + `130 MB` static cross-KV |
| **MTP Head** | 1L config → stripped in MLX | `layers.16` (35M) → **stripped in MLX** | None | 1L config → stripped in MLX | None | 1 aux train head → unused |

### Key Architectural Explanations of Observed Benchmark Behavior
1. **Why `OvisOCR2` (`0.85B`) beat `4B–7B` models:** (a) native 300-DPI resolution (`16.78M px`, zero digit blur unlike `LightOnOCR-2`'s `1540 px` cap), (b) a `4.6×` lighter `12L 88M` ViT vs. `404M` in GLM/LightOn, (c) `3:1` GDN+GQA stability + `12 KB/tok` KV (`2.71 GB` peak RAM at `10.1K` tokens), and (d) on-policy distillation from a `4B` RL teacher with HTML table-alignment rewards.
2. **Why `GLM-OCR` looped on the 6.6M-px sideways ledger:** Trained behind a layout cropper (`PP-DocLayout-V3`), not rotated full pages; feeding `6,039` rotated tokens into a shallow `16L` decoder with short `rope_theta=10,000` and no QK-Norm collapsed into a repeating 6-row loop, while `mlx_vlm` stripped `layers.16` MTP (`76.2 tok/s` single-token BF16).
3. **Why `DeepSeek-OCR-2` ran at `3.86 s/p`:** Strict `<=1,121` visual tokens (fixed `768²` / `1024²` shapes -> ideal for ANE CoreML!) + `~574M` active 12L Q4 MoE decode (`~320 MB/token`), trading off fine-character accuracy (`33/40`) and `8,192` context limit.
4. **Why `OvisOCR2` & `NuExtract3` share the highest `fast-silicon` leverage:** Both run on `fast-qwen3`'s already-tuned 3:1 GDN decoder, and **`OvisOCR2` wastes `29.8%` (`254M` / `270 MB` Q8) of its parameters on a `248,320`-token multilingual vocab** — pruning losslessly to the `140,307`-token OCR Latin/symbol subset (`248,320 → 140,307`) via `QwenVocabRemapper` yields `1.10–1.12×` DRAM bandwidth gains with `100.0%` token/text parity and zero retraining (whereas tighter `49K–65K` caps dropped high-ID English tokens like `'Thousands'`).

---

## 3. Theorized Optimal OCR Architecture (Long-Term Horizon)

Combining every winning primitive while fixing every bottleneck above yields a ~`500–550M`-parameter architecture (`~100M` Encoder + `~400M` Decoder + `~40M` Vocab):

1. **Stage 0 — Micro-Orientation & Ink-Gated Static Tiles:** `2M` 4-way orientation CNN + pixel-variance blank-tile skipper (`40–60%` empty margins dropped before compute) over **fixed-shape tiles** (`256×1024` or `512×512`) so tile encoding lowers cleanly to **CoreML ANE `conv1x1` (`12.5–18.5 TF`)**.
2. **Stage 1–2 — Conv Glyph Stem + Window/Axial Encoder (`~90–110M`):** Three stride-2 `3×3` conv stages preserve thin decimal dots and rulings at full 300 DPI, followed by `6–7` wide-window (`4×16`) blocks and `2–3` **Row + Column Axial Attention layers** across active tiles (`71×` fewer pairs than global 2D attention, directly connecting row labels to right-aligned amounts and column headers to subtotals).
3. **Stage 3 — Anisotropic `1×4` Compression + Lightweight Causal Reorder (`~15M`):** Preserves full vertical row separation (`Nemotron-Parse`) while emitting `~128–256` global layout tokens + `~800–1,800` reading-order tokens (`DeepSeek-OCR-2`). *(Doubles as a zero-decoder sub-40ms retrieval encoder!)*
4. **Decoder — Decoupled Cross-Attention 3:1 GDN (`~400M` Dense) + `40K` Atomic-Digit Vocab + Native `K=6` MTP:**
   - Static visual `K_vis, V_vis` in 3–4 cross-attention layers (`~30 MB` total visual KV, zero FFN/GDN visual prefill!).
   - Dense `16L` (`12` GDN + `4` Full GQA, `hd=128`, QK-Norm) verifies `6–8` draft tokens in one DRAM pass (`~280 MB` Q4) with zero MoE expert-blowup penalty.
   - Native end-to-end `K=6` Eagle-3 / MTP head exploits OCR's deterministic visual copying (`75–90%` acceptance) to target `1,000–2,000+ tok/s`, backed by `llguidance` table grammar & RLVR exact-number/blank-cell rewards.

---

## 4. Zero-Training `fast-ocr` Production Architecture (Immediate Implementation)

To deliver maximum speed **immediately with \$0 training**, `packages/fast-ocr` implements three coordinated components on Apple Silicon:

1. **Independent `AppleVisionOCR` (`backend="apple-vision"` — zero MLX/VLM imports):**
   - PyObjC `VNRecognizeTextRequest` on ANE with leak-free `NSData` + `objc.autorelease_pool()` + optional 3-band strip deduplication (`14–17 pp/s` multi-worker ANE scaling) and `PageAnalysis` (orientation, tight ink box, digit/currency density, table complexity).
   - **Zero-Latency XY-Cut & Geometric GFM Synthesizer (`<0.5 ms/page`):** Uses Apple Vision's native line boxes to reconstruct multi-column XY-Cut reading order, `#`/`##`/`###` typography headings, lists/de-hyphenation, and strictly rectangular `| --- |` GFM tables (`---:` right-aligned numeric columns).
2. **`FastOvisOCREngine` (`backend="ovis"` — built on `fast-qwen3`):**
   - Ported 12L/24L Qwen3.5 ViT (`FastQwen35VisionModel`) + exact interleaved 3D MRoPE (`mrope_section=[11, 11, 10]`, `partial_rotary_factor=0.25`, `rope_offset_delta` decode offset invariance).
   - Lossless OCR-safe `QwenVocabRemapper` (`248,320 → 140,307`) slicing tied embeddings (`1.10–1.12×` decode speedup, `100.0%` token/text parity; uncapped because P2 showed `49K–65K` caps omit high-ID English tokens like `'Thousands'`).
   - Stage-0 ink-tight margin crop (`-35%..45%` whitespace -> up to `3×` fewer quadratic ViT ops) + auto-orientation.
   - **Zero-Training Speculative Decoding (`OCRPLDDrafter` + `verify_draft_sequence_b1`):** Uses ANE Apple Vision recognized text + online Self-PLD HTML/Markdown table structure to propose multi-token drafts verified in one GPU pass (mathematically bit-identical to greedy `OvisOCR2`, enabled strictly when probe measurements confirm wall-clock speedup).
   - **Validly-Structured Markdown Guarantees:**
     - **Default (`<0.5 ms/page`):** Exact 2D-occupancy HTML `<table>` parser (properly expanding `colspan`/`rowspan` and preserving explicit blanks) -> normalized GFM or rectangular HTML tables + fence/math balancing (`canonicalize_markdown` + `validate_markdown`).
     - **Opt-In Hack (`--strict-markdown-grammar`):** `DynamicTableRowGrammar` using `llguidance` Lark CFG JIT-compiles a $C$-column lock the moment a GFM header row completes, emitting the `| --- | ... |\n` separator via `compute_ff_tokens()` in `0` forward passes and locking body rows to exact $C$ cells.
3. **`AutoOCRPipeline` (`backend="auto"`):**
   - **Tier 0 (`pypdfium2` native text):** ~`0.01 s/p` on clean digital pages.
   - **Tier 1 (`AppleVisionOCR` on ANE):** Keeps clean prose directly in validly structured Markdown (`~0.15–0.45 s/p`).
   - **Tier 2 (`FastOvisOCREngine`, double-buffered ANE `N+1` + GPU `N`):** Escalates tables, high-digit ledgers, low-confidence scans, or rotated pages, reusing the already-computed ANE Vision lines for orientation, ink cropping, and speculative drafting.

---

## 5. Live Probe & Final Benchmark Receipts (M4 Pro vs. Base M5)

All hardware receipts below were measured directly on:
- **M4 Pro (`Mac16,8`):** 48 GB unified RAM, `Apple9` GPU (`210.2 GB/s` empirical BW), 16-core ANE, `macOS 26.6`, `MLX 0.32.4`
- **Base M5 (`Mac17,2`):** 24 GB unified RAM, `Apple10` GPU w/ NAX (`137.3 GB/s` empirical BW), 16-core `h17` ANE, `macOS 27.2.1`, `MLX 0.32.4`

### 5A. Focused Stage Probes (`P1`–`P7`)

| Probe | Target & Test | M4 Pro (`Apple9`) Receipt | Base M5 (`Apple10 NAX`) Receipt | Key Decision |
|---|---|---|---|---|
| **P1 Parity** | `FastOvisOCREngine` vs `mlx_vlm` `OvisOCR2-8bit` on full Page A (`8,240` vis toks) | Same-pixel ViT cos **`1.0`** (`max_abs=0.0`); `128/128` live token & `1,503/1,503` full-page text match; prefill `1,962.6` vs `4,141.8 ms` (**`2.11×`**); **`3.01 GB`** vs `9.27 GB` | Covered via P4/E2E (`2.86 GB` full Page A peak; **`45/45`** Page C exact) | Zero-regression MLX port + MRoPE `rope_offset_delta` verified |
| **P2 Vocab Remap** | `248,320` -> Latin/symbol subset (`build_ocr_vocab_subset`) | Uncapped OCR subset = **`140,307` tokens** (`0` missing symbols, `100.0%` token parity, **`1.10–1.12×`** decode speedup: `174.8` vs `159.1 tok/s`); `49K–65K` caps dropped real tokens (`'Thousands'`) | Same `140,307` (`140,312` padded) default active | Use `target_size=None` -> lossless **`140,307`** subset by default |
| **P3 ANE Vision** | Standalone `AppleVisionOCR` (`zero mlx`) + `workers=4` + `bands=1` vs `3` | `8/8` 360° orientation exact; clean prose **`223.9 ms`** (`b1`); `4`-worker scaling **`5.17 pp/s`** (`1.99×`); `b3` lifts Page B `table_score` `0.3->1.0` & Page C `45/45` | `zero mlx` confirmed; clean prose **`104.0 ms`** (`b1`) / **`120.4 ms`** (`b3` auto); `4`-worker scaling **`12.60 pp/s`** (`2.33×` vs `5.40 pp/s` seq); Page C `32/45` (`b1`) / `38/45` (`b3`) | Standalone defaults `bands=1` (`language_correction=True`); `auto` pre-pass uses `bands=3` (`False`) so complex tables always escalate cleanly |
| **P4 Crop + Orient** | `crop_to_ink(pad=24)` (Page A/B) & `90° CW` orient (Page C) (`spec="off"`) | Page A ViT `7,726 -> 1,728 ms` (**`4.47×`**), `-56.3%` vis toks (`8,240->3,600`), exact text; Page B ViT `7,747 -> 1,500 ms` (**`5.16×`**); Page C `44/45` -> **`45/45`** | Page A ViT `6,542.0 -> 1,152.0 ms` (**`5.68×`**), prefill `1,255.8 -> 524.6 ms` (`2.39×`), peak `2.86 -> 2.52 GB`, exact text; Page C **`45/45`** | Keep `crop_margins=True` & `auto_orient=True` on by default |
| **P5 Speculative PLD** | `OCRPLDDrafter` (`K=4..8`, `self` vs `vision+self`) vs `off` (AR) | **`100%` bit-identical** across all `12` configs; `vision+self K=4` hits **`360.5 tok/s`** Page A (`1.943×` vs `185.5`) and **`436.8 tok/s`** Page C (`2.465×` vs `177.2`) | **`100%` bit-identical**; `vision+self K=4` hits **`282.8 tok/s`** Page A (`2.070×` vs `136.6`); `K=4`/`K=6` hit **`349.2`** / **`355.2 tok/s`** Page C (`2.707×` / `2.753×` vs `129.0`) | Default **`vision+self` + `draft_k=4`** wins across both `Apple9` and `Apple10 NAX` |
| **P6 ViT Precision & Double Buffer** | `Q8` vs `BF16` ViT (`Page A`), ANE overlapped `N+1` behind GPU `N` | `Q8` ViT **`0.9686×`** (`7,976` vs `7,726 ms`, `cos=0.997832`); double-buffered ANE hidden **`100.0%`** (`0.03 ms` wait across B & C) | `Q8` ViT **`1.0154×`** raw (`6,735` vs `6,838 ms`, `cos=0.997817`) and **`0.9093×`** cropped (`1,267` vs `1,152 ms`, exact text) | Keep **`BF16` ViT** (faster on cropped active area on both chips) + async double-buffered `ocr_images` |
| **P7 Structured MD** | XY-Cut ANE MD, HTML->GFM canonicalizer, `DynamicTableRowGrammar` | XY-Cut `8.1–13.5 ms` (`8/8` valid); canonicalizer **`0.17–1.57 ms`** (`-89%..-94%` bytes vs heavy tree exports, `100%` amounts kept); Lark `C`-col lock **`0.54–4.10 ms`**, `\| --- \|` separator `0` steps | Verified identical CPU behavior (`11/11` `fast-ocr` tests pass on Base M5) | Default `<1 ms` `canonicalize_markdown` + `validate_markdown`; `--strict-markdown-grammar` opt-in |

### 5B. Wave-3 End-to-End `FastOCR` Suite (`[synth_prose, Page A, Page B, Page C]`)

Measured end-to-end with production defaults (`output_format="markdown"`, `table_mode="gfm"`, `crop_margins=True`, `auto_orient=True`, `speculative="auto"`, `draft_k=4`, `vocab_size=None` -> `140,307` active, `vit_quant_bits=None` BF16):

| Workload / Mode | M4 Pro (`Apple9`) | Base M5 (`Apple10 NAX`) | Accuracy, Validation & Observations |
|---|---|---|---|
| **Baseline `mlx_vlm` Page A** (`8,240` vis toks, AR) | ViT `7,617.8 ms` · Prefill `4,141.8 ms` · **`9.27 GB`** *(~`18.5 s` est. wall; `fast-ocr` full AR = `19,061.4 ms`, `3.01 GB`)* | — | `1,503` content tokens; `mlx_vlm` uses full-canvas uncropped ViT + `248K` vocab + single-token decode |
| **`FastOCR(backend="ovis")` — Page A** (`3,600` vis toks) | **`7,350.3 ms`** total (`1,732.0` ViT / `808.3` pref / **`345.4 tok/s`**), **`2.79 GB`** (**`2.59×`** vs full AR; **`4.40×`** ViT, **`5.12×`** prefill, **`3.32×`** lower RAM vs `mlx_vlm`) | **`7,249.9 ms`** total (**`1,139.3` ViT** / **`541.4` pref** / `282.4 tok/s`) | **`valid=True`** (`3` GFM tables); M5 `NAX` leads ViT by `1.52×`, M4 Pro leads DRAM-bound decode by `1.22×` |
| **`FastOCR(backend="ovis")` — Page B** (`3,182` vis toks) | **`5,961.6 ms`** total (`1,451.9` ViT / `715.5` pref / `287.1 tok/s`), `2.78 GB` | **`5,040.5 ms`** total (**`880.3` ViT** / **`469.6` pref** / `269.4 tok/s`) | **`valid=True`** (`1` table); M5 wins end-to-end by **`1.18×`** because Page B is ViT+prefill heavy |
| **`FastOCR(backend="ovis")` — Page C** (`90° CW`, `6,528` vis toks) | **`15,062.2 ms`** total (`4,999.3` ViT / `1,511.8` pref / **`438.7 tok/s`**), `2.92 GB` | **`15,103.9 ms`** total (**`3,876.6` ViT** / **`956.8` pref** / `348.9 tok/s`) | **`45/45` raw & `45/45` GFM** amounts (`0` omitted, `0` extra), `rot=90° CW`, **`valid=True`** on **both** chips |
| **`FastOCR(backend="ovis")` 3-Page Batch** (`[A, B, C]`, double-buffered) | **`27,189.0 ms`** (`14,413.6 ms` Page C @ **`441.3 tok/s`**) vs `28,455.4 ms` seq | **`27,092.0 ms`** (`14,850.3 ms` Page C @ `348.9 tok/s`) vs `27,435.0 ms` seq | **`45/45`** raw & GFM on both chips |
| **`FastOCR(backend="auto")` 4-Page Mix** (`[synth_prose, A, B, C]`) | **`29,422.1 ms`** all-4 wall: `synth_prose` stays on **`apple-vision` (`169.8 ms`)**; A/B/C escalate (`6,946` / `5,395` / `14,564 ms`) | **`28,218.9 ms`** all-4 wall: `synth_prose` stays on **`apple-vision` (`120.4 ms`)**; A/B/C escalate (`7,137` / `4,809` / `14,835 ms`) | **`4/4` `valid=True`**, **`45/45` Page C** on **both** M4 Pro and Base M5 |

---

## 6. Completed DRY Backport (`fast-shared` -> `fast-pplx` & `fast-qwen3`)

The shared vision stack has been promoted into `packages/fast-shared` (with `fast-ocr` retaining thin re-export shims) and verified across **M4 Pro (`Apple9`)** and **Base M5 (`Apple10 NAX`)**:

1. **Shared Modules in `fast-shared`:**
   - `vision_preprocess.py` (zero-`mlx`, lazy-`PIL` `smart_resize_qwen`, `crop_to_ink`, `build_qwen_vision_inputs`, and `prepare_pplx_image_item`) + lazy MLX module resolution in `fast_shared/__init__.py` (preserving zero-MLX startup for `AppleVisionOCR`).
   - `qwen35_vision.py` (`FastQwen35VisionModel`, uniform-frame SDPA fast path, `16`-entry grid geometry cache, integer `compute_mrope_positions`, fractional `precompute_mrope_trig`) + `detect_apple10_d256_force_fused` in `hw_check.py`.
2. **`fast-pplx` Visual Embeddings (`FastPplxGPU.encode_images` / `FastPplx.encode_images` / `fast-pplx embed-images`):**
   - Verified `input_ids_exact=True`, `max_pos_abs_diff=0.0`, and `spans_exact=True` vs upstream PyTorch `prepare_images`.
   - At `100 DPI`, `chunks=8` (`gpu=float16, vit=float16`): **`159.12 ms/p`** (`6.28 pp/s`, `mean_cos_vs_pt=0.999698`, `1.87 GB` peak) on M4 Pro and **`109.24 ms/p`** (`9.15 pp/s`, `mean_cos_vs_pt=0.999699`, `1.72 GB` peak) on Base M5 (versus `573.40 ms/p` PyTorch MPS). At `150 DPI`, `chunks=8`: **`688.48 ms/p`** (M4 Pro) and **`424.87 ms/p`** (Base M5, `mean_cos_vs_pt=0.999825`–`0.999838`).
   - Lazy ViT loading leaves text-only `FastPplx` callers with `0` extra ViT RAM; `crop_margins=True` improved dense-page cross-modal query similarity (`[0.4299, 0.4416, 0.5918]` vs `[0.4351, 0.4192, 0.5830]`).
3. **`fast-qwen3` Compiled Kernels & Lazy Multimodal `generate(..., images=...)`:**
   - Microbench + end-to-end Page A probes confirmed `0.0` `max_abs_diff` and bit-exact token output with speedup on both M4 Pro (`1.053–1.303x`) and Base M5 (`1.038–1.440x`) using `@partial(mx.compile, shapeless=True)` on `RMSNormGated` and `normalize_gdn_qk` and `@mx.compile` on `apply_partial_rope_trig`.
   - `FastQwen3Engine.generate(..., images=...)` lazily loads `FastQwen35VisionModel` only when image inputs are passed (and preserves prompt-injection sanitization while inserting trusted `<|vision_start|><|image_pad|><|vision_end|>` markers and `rope_offset_delta` decode invariance).
4. **What Remains `fast-ocr`-Specific:**
   - `OCRPLDDrafter`, `DynamicTableRowGrammar`, and `QwenVocabRemapper` (`140,307`-token English OCR subset) stay scoped to OCR decoding and do not load into `fast-pplx` (`MappedEmbedding` keeps all `248,320` token rows mmap-backed on CPU).

---

## 7. Cross-Silicon Apple Vision Hygiene, Grammar-Locked PLD & Base M5 Receipts

1. **Apple Vision Provenance & Zero-MLX Isolation (`apple_vision.py`):**
   - `AppleVisionOCR.recognize_image` records `"macos_version"`, `"macos_build"` (`/System/Library/CoreServices/SystemVersion.plist`), and `"request_revision"` (`VNRecognizeTextRequest.revision()`) in `OCRResult.metadata` while keeping `fast_ocr.apple_vision` strictly zero-`mlx`:
     - **M4 Pro (`Apple9`):** `macos_version="26.6"`, `macos_build="25G5057c"`, `request_revision=3`
     - **Base M5 (`Apple10 NAX`):** `macos_version="27.2.1"`, `macos_build="26B131"`, `request_revision=3`
   - CLI exposes `--no-language-correction` alongside `--bands`, `--level`, and `--triage-dpi`.
2. **PDF Native Hygiene & Adaptive DPI Cap (`preprocess.py`, `pipeline.py`, `cli.py`):**
   - `compute_adaptive_render_dpi` clamps oversized PDF canvases to `DEFAULT_MAX_PDF_PIXELS = 20_000_000` (with `min_dpi = 36` floor; exposed via `--max-pdf-pixels`, `<= 0` disables).
   - `extract_native_pdf_text` and `AutoOCRPipeline.ocr_pdf` strip standalone running headers/footers/typesetting stamps (`strip_running_pdf_boilerplate`), count embedded page images (`pypdfium2.raw.FPDF_PAGEOBJ_IMAGE`), reject any page with non-whitespace C0/C1 control characters or `U+FFFD` (`count_control_garble_chars == 0`), and route native pages when `is_clean and (eff_chars >= 80 or (eff_chars > 0 and image_count == 0))` (preserving raster OCR on vector-outlined zero-text pages).
3. **Cross-Silicon Verification Receipts (After All Edits):**
   - **M4 Pro (`Apple9`, macOS `26.6`, `25G5057c`):** `packages/fast-ocr/tests/{test_fast_ocr_cpu.py,test_fast_ocr_hw.py}` **`14 passed in 11.25s`**; `python -m fast_ocr.cli verify --json` **`"ok": true`**; `make sync-check guard leak-check` green (`0` errors, `0` warnings).
   - **Base M5 (`Apple10 NAX`, macOS `27.2.1`, `26B131`):** `packages/fast-ocr/tests/{test_fast_ocr_cpu.py,test_fast_ocr_hw.py}` **`14 passed`** (`test_apple_vision_independent_ane_orientation_and_markdown_hw` + `test_ovis_engine_speculative_bit_identity_and_structured_markdown_hw`); `python -m fast_ocr.cli verify --json` **`"ok": true`**.
   - **Base M5 `FinTabNet-OTSL` (`n=2`, `strict_markdown_grammar=True`, `table_mode="gfm"`):**
     - `speculative="off"` (AR baseline): `4,154.7 ms` wall (`2,469.9` / `1,637.0 ms` per-page), `TEDS=0.868746`, `TEDS-S=0.908472`, `Numeric Cell F1=0.929874`, `valid_markdown_rate=1.0`.
     - `speculative="vision+self"` (`draft_k=4`, clone-safe `DynamicTableRowGrammar`): **`2/2` exact text parity (`all_exact_parity=True`)**, **`3,632.8 ms` wall** (`2,117.6` / `1,082.2 ms` per-page, **`1.144×`** wall speedup), identical `TEDS=0.868746`, `TEDS-S=0.908472`, `Numeric Cell F1=0.929874`, `valid_markdown_rate=1.0`.
     - `speculative="vision+self"` + `zoom_recover=True`: `4,185.3 ms` wall (`0` zoom triggers needed on these two clean crops), identical `TEDS=0.868746` and `valid_markdown_rate=1.0`.

---

## 8. `B > 8` Batching, GDN/KV State Compression, Region Routing & Gov-Doc Expansion

Following Sections 5–7, additional landed runtime optimizations and Category 5 Gov-Doc benchmarks include:

1. **`B = 16` Vision Batching & Parallel ANE (`M4 Pro` verified):**
   - `run_vision_batch_parallel` (`workers=4`) accelerated `n=16` `FinTabNet` `1024×512` crops **`1.592×`** (`6.04 -> 9.61 pp/s`; `9.47 pp/s` through `FastOCR.ocr_images(..., batch_size=16)`) with `16/16` exact text parity.
   - Batched `OvisOCR2-8bit` decode (`batch_size=16`, `pack_length_waves`, `compact_hybrid_cache`, `512`-tok prefill chunking, `kv_step <= 512`) lifted `FinTabNet-OTSL(16)` **`1.663×`** (`21.621s` `0.74 pp/s` `281.6 tok/s` -> **`12.998s` `1.231 pp/s` `468.0 tok/s`**, identical `TEDS=0.8675` / `NumF1=0.8945`, `15/16` parity). Keep `B=1` `ANE+PLD` on long dense prose (`olmOCR-bench(8)` @ `150 DPI`: `73.0` vs `64.7 tok/s` `B=8` AR).
2. **`GdnDecodeShared` Metal State, `Auto` `q4_quarot` KV & Opt-In White-Patch Dropping (`fast-qwen3` / `fast-ocr`):**
   - `GdnDecodeShared` (`packages/fast-qwen3/src/fast_qwen3/gdn_kernels.py`, `FP32` decay `g`, exact `mask` freeze `frozen_diff=0.0`) delivers `y_cos >= 0.99998` @ `T=1` and `256`-step AR `0.999988` (`fp32`) / `0.999971` (`bf16`) / `0.999808` (`int8_blk32`, `3.76×` DRAM cut). Isolated 18-layer M4 Pro decode microbenchmarks sped up **`1.422×` @ `B=1`** (`1.022 -> 0.719 ms`), **`1.664×` @ `B=8`** (`2.623 -> 1.576 ms`), **`2.508×` @ `B=16`** (`5.314 -> 2.119 ms`), **`1.890×` @ `B=32`** (`11.334 -> 5.996 ms`), and **`3.432×` @ `B=64`** (`31.753 -> 9.252 ms` `bf16` / `14.692 ms` `int8_blk32`, `2.161×`).
   - `resolve_gdn_state_mode("auto")` uses native `fp32` for `B < 8` (`< 4` on `<= 28 GB` hosts) so single-stream PLD (`B=1`, `1.49 s`) avoids Metal dispatch overhead, and switches to `int8_blk32` at `B >= 8` (`>= 4` on `<= 28 GB`). Paired with `resolve_ovis_kv_mode` (`"auto"` -> `"q4_quarot"` at `B >= 8` or context `>= 4096/8192`) on `16` E2E M4 Pro pages (`8` FinTabNet + `8` OmniDocBench, `max_tokens=384`), `auto` defaults (`drop_white_patches=False`) ran **`42.59 s` (`0.38 pp/s`, `-9.0%` vs `46.81 s` `fp32/bf16`)** with **`14/16` exact matches, `median_sim=1.0000`, `min_sim=0.9271`**, and `16/16` valid Markdown.
   - `drop_white_patches=True` remains strictly **opt-in (`False` default)** and CLI-accessible via `--drop-white-patches`: while clean-margin/PPT probes showed high exact parity (`-22.2%` tokens), turning it on unconditionally across compact financial tables (`FinTabNet` `FT_175` `Employee -> Operating` / `$262 -> $263`, `FT_240` `Sacramento -> San Antonio`) collapsed 16-page E2E exact parity to `2/16` (`min_sim=0.3899`). Outer whitespace margins are already trimmed losslessly pre-ViT via `crop_to_ink`.
3. **Reading-Order PLD, `24px` Adaptive Resolution, Camera-Gated `--deskew auto`, `--region-routing`, & CLI/Benchmark Wiring:**
   - Reading-order PLD with windowed cursor relocking (`1.026×–1.038×` lossless), lossless `24px`/line `adaptive_resolution=True` (`1.057×–1.140×`), camera-photo gate `is_likely_camera_photo` (`deskew="auto"` default: catches `39/39` `CORD-v2` photos for `+10.94%` recall & `1.637×` while rejecting `0/12` archive scans, `0/24` `OmniDocBench` pages, and `0/40` cropped tables), Latin-gated scientific math/footnote glyphs in the `140K` OCR subset (`latin_only=True`, `--no-latin-only`), and full `AutoOCRPipeline`/`cli.py`/`ocr_parquet` passthrough (`--kv-i8`, `--gdn-state-mode`, `--drop-white-patches`, `--no-latin-only`, `prompt_seq[i]` zoom retry fix, `_dispatch_escalated_vlm` crop `batch_size` cap, and 2-stage PDF `triage_dpi`).
   - Opt-in `--region-routing` (`region_routing=False` default, `REGION_CROP_MIN_PIXELS=4096`, `allow_formula_regions=False`) achieved `1.84×` on mixed `olmOCR table_tests(2)` (`15.21 -> 8.25 s`) and `1.83×` (`11.54 -> 6.32 s`) on 2-column `olmOCR-bench` `00f6c2`, yet 2-column ANE prose strips interleaved columns (`order_pass=0/3`), reinforcing `region_routing=False` opt-in status.
4. **3-Branch Category 5 Gov-Doc Corpus (`configs/outcome_benchmarks.lock.json` `24` Datasets + `us_gov_ocr`):**
   - Integrated `vrdu_ad_buy`, `vrdu_registration`, `doclaynet_gov`, opt-in `puredocbench_legal_gov` (`453` `03_legal_gov` images staged on Base M5 across `clean`/`digital`/`real`; `72/72` EN resolved), and 8-source `us_gov_ocr` (`PLAW`, `BILLS`, `USCODE`, `FR` `<GPOTABLE>`, `BUDGET` `.xlsx`, `USCOURTS`, `scotus_opinions`, auto-staged LoC `btp-data`).
   - Initial Base M5 CPU baseline (`public-pypdfium2` on `US-Gov-OCR(27)`, filtering `4` raster `loc_btp` scans) scored `1.0` on single-column `uscourts`/`scotus` yet `0.2979` overall and **`0.0` `table_teds`** (`govinfo_budget=0.0104`, `govinfo_bills=0.2073`, `govinfo_fr=0.5135`, `govinfo_uscode=0.5761`), highlighting the need for multi-column/table layout escalation rather than blind Tier-0 plain-text bypass on digital government PDFs. See [`OCR_BENCHMARKS_AND_BASELINES.md`](OCR_BENCHMARKS_AND_BASELINES.md) for setup and CLI details.
