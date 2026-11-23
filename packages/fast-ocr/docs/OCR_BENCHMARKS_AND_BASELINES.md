# OCR Benchmarks, Baselines & `fast-silicon` Integration

This document catalogs the 2025–2026 public OCR benchmark ecosystem across output formats (`Markdown`, `HTML`/`GFM` tables, `LaTeX` equations, `JSON` schema extraction, and raw/reading-order text), standard published baselines, exact scoring definitions, and the **Category 5 (`fast-ocr`)** integration in `configs/outcome_benchmarks.lock.json`, `fast_shared/outcome_eval/ocr.py`, and `scripts/benchmark_outcomes.py`.

---

## 1. Benchmark Landscape by Output Type

### 1A. Full-Page Document → Markdown (Layout, Reading Order, Tables, Formulas)

| Benchmark | HF Dataset Repo | Scale & Structure | License | Core Metrics | Best Role & Notes |
|---|---|---|---|---|---|
| **olmOCR-Bench** | `allenai/olmOCR-bench` | 1,403 single-page PDFs + 7,010 binary unit tests across 7 categories (`arxiv_math`, `table_tests`, `multi_column`, `headers_footers`, `old_scans`, `old_scans_math`, `long_tiny_text`). | `ODC-BY` | Pass/fail unit-test accuracy across `present`, `absent`, `order`, `table`, `math`, `baseline`. | Zero-ambiguity English PDF-to-Markdown test; immune to HTML-vs-GFM styling differences. |
| **FinTabNet-OTSL** | `docling-project/FinTabNet_OTSL` | ~112K financial SEC tables (`test` Parquet shards with PIL images, HTML tokens, OTSL, restored HTML, `rows`, `cols`). | `CDLA-Permissive-1.0` | **TEDS**, **TEDS-S**, **GriTS-Top**, **GriTS-Con**, **Numeric Cell F1**. | Top permissive standalone financial table benchmark (colspans, rowspans, blank cells, aligned currency columns). |
| **OmniDocBench** | `opendatalab/OmniDocBench` | 1,355–1,651 pages across 9 document domains (`OmniDocBench.json` + `images/`). | `Research-NonCommercial (code Apache-2.0)` | `(1 - Text_Edit) + Table_TEDS + Formula_CDM`, plus Reading Order Edit Distance. | Gold standard for full-page VLM comparison; English split (`755` pages) evaluates compact Latin-vocab OCR models. Explicitly labeled `Research-NonCommercial` in our lockfile; `upstage/dp-bench` (`MIT`) is the commercial-permissive layout alternative. |
| **DP-Bench** | `upstage/dp-bench` | ~200 PDF pages + `reference.json` layout elements. | `MIT` | **NID** (Normalized Indel Distance), **TEDS**, **TEDS-S**. | Clean MIT alternative for pipeline benchmarks (`Docling`, `Marker`, `Upstage DP`, `LlamaParse`). |
| **Real5-OmniDocBench** | `PaddlePaddle/Real5-OmniDocBench` | 6,775 pages (5 physical degradations: scan, warp, screen photo, lighting, skew). | Research / base OmniDocBench GT | OmniDocBench suite across distortion tracks. | Stress-tests skew, camera warp, and physical illumination robustness. |
| **PureDocBench** | `zhihengli-casia/puredocbench` | 1,475 × 3 tracks (*Clean*, *Digital Degraded*, *Real Degraded*), 10 domains. | Open / Research | Source-traceable multi-track average ($\text{Avg}_3$). | Un-saturated 2026 benchmark (`~75–80` SOTA). |

### 1B. Mathematical Formulas, Structured JSON KIE & General/Handwriting OCR

| Category | Benchmark | HF Dataset Repo | License | Core Metrics | Notes |
|---|---|---|---|---|---|
| **Formulas / LaTeX** | **UniMER-Test** | `wanderkid/UniMER_Dataset` | Open / Research | **CDM**, **ExpRate@CDM**, **BLEU**, **NED**. | 23,757 formulas (`SPE`, `CPE`, `SCE`, `HWE`). Pure-Python tier uses canonicalized LaTeX `1 - NED`; TeXLive-rendered CDM is heavy/optional. |
| **Structured JSON KIE** | **CORD-v2** | `naver-clova-ix/cord-v2` | `CC-BY-4.0` | **`grounded_field_f1`**, JSON Schema compliance, leaf **ANLS**. | 100 test receipts with nested `menu[]` line items, subtotals, and totals. |
| **Structured JSON KIE** | **ExtractBench** | `llamaindex/ExtractBench` | `Apache-2.0` | **`grounded_field_f1`**, JSON Schema compliance. | Already pinned in `outcome_benchmarks.lock.json` (real multi-domain PDFs + JSON Schemas). |
| **General / Multilingual** | **CC-OCR** | `wulipc/CC-OCR` | `MIT` | Track F1, NED, TEDS. | Covers Multi-Scene, Multilingual (32+ languages), Doc, Table, KIE. |
| **LMM OCR Reasoning** | **OCRBench / v2** | `echo840/OCRBench` · `lmms-lab/OCRBench-v2` | Research | Exact/substring (`OCRBench` / 1000) · Macro (`OCRBench-v2`). | Broad VLM OCR QA benchmark. |
| **Degraded Scans / HTR** | `olmOCR-bench` (`old_scans`), `IAM` | `allenai/olmOCR-bench` · `gagan3012/IAM` | `ODC-BY` / Academic | Pass rate (`old_scans`), **CER / WER** (`IAM`). | `old_scans` gives clean ODC-BY coverage of historical degraded prints. |

---

## 2. Published Baselines Benchmarked in 2025–2026 Literature

- **Compact Specialized OCR VLMs (0.6B–4B):**
  - `OvisOCR2` (`0.85B`): `96.58` OmniDocBench v1.6, `75.06` PureDocBench $\text{Avg}_3$ (`fast-ocr` default VLM).
  - `PaddleOCR-VL-1.5 / 1.6` (`0.9B`): `94.50` (v1.5) / `96.34` (v1.6) OmniDocBench (`0.038` Text Edit, `91.67` TEDS, `96.89` CDM).
  - `GLM-OCR` (`0.9B–1.1B`): `94.62` (v1.5) / `95.22` (v1.6) OmniDocBench, `90.32` Real5-OmniDocBench.
  - `MinerU2.5 / MinerU2.5-Pro` (`1.2B`): `90.67`–`95.75` OmniDocBench, `75.2` olmOCR-Bench (`2.5.4`).
  - `DeepSeek-OCR / DeepSeek-OCR-2` (`3B` MoE): `91.10` OmniDocBench v1.5, `73.01` Real5-OmniDocBench, `75.7` olmOCR-Bench (`v1`).
  - `LightOnOCR-2-1B` (`1.0B`): `83.2 ± 0.9` olmOCR-Bench (RLVR leader at 1B scale).
  - `NuExtract3` (`4B` Qwen3.5): Apache-2.0 visual JSON KIE + Markdown (`fast-ocr` JSON preset).
- **Larger RLVR / Frontier VLMs & Pipelines:**
  - `olmOCR-2-7B` (`82.4` olmOCR-Bench), `Chandra-2` (`85.8–86.9` olmOCR-Bench), `Qwen2.5-VL-7B/72B` (`864/888` OCRBench), `Gemini Pro` (`~88–90+` OmniDocBench v1.5, `55.5` OCRBench v2 EN), `GPT-4o` (`736` OCRBench), `Docling` (`93.6–96.75%` PubTabNet TEDS), `Marker` (`76.1` olmOCR-Bench), `Apple Vision` (`VNRecognizeTextRequest` ANE), and stock `mlx_vlm`.

---

## 3. Metric Definitions in `fast_shared/outcome_eval/ocr.py`

All Category 5 metrics are implemented in pure Python + NumPy with zero `datasets`, `torch`, `scipy`, `rapidfuzz`, or TeX binaries:

- **Myers Bit-Parallel Levenshtein & Semi-Global Fuzzy Substring Matching:** Exact edit distance (`levenshtein_distance`, `normalized_edit_distance`) and semi-global matching (`fuzzy_find_substring`) with explicit empty-output guards and loop detection (`repeat_4gram_rate >= 0.15` fails validity).
- **Zhang–Shasha TEDS & TEDS-S (`compute_teds`, `compute_teds_structure`):** Evaluates both HTML `<table>` outputs and GFM pipe tables (canonicalized to HTML for apples-to-apples scoring) with `rowspan`/`colspan` grid expansion.
- **Factored 2D GriTS (`compute_grits`):** Computes `GriTS-Top` (2D topology) and `GriTS-Con` (normalized cell content F1).
- **Financial Numeric Cell F1 (`compute_numeric_cell_f1`):** Normalizes signed numbers, currency symbols, thousands separators, and accounting parentheses (`($1,250)` → `-1250`) to catch silent dropped rows/minus signs.
- **Page & KIE Evaluators:**
  - `evaluate_olmocr_bench`: `present`, `absent`, `order`, `table` (`up`/`down`/`left`/`right`/`top_heading`/`left_heading`), canonicalized LaTeX `math`, and `baseline`.
  - `evaluate_fintabnet_otsl`: `TEDS` primary score + `TEDS-S`, `GriTS-Top`, `GriTS-Con`, `Numeric Cell F1`.
  - `evaluate_omnidocbench`: `(1 - Text_NED)`, reading-order similarity, `Table_TEDS`, `Formula_NormSim`.
  - `evaluate_cord_v2`: `grounded_field_f1` (Hungarian array matching + JSON Schema validation) + recursive leaf `ANLS`.

---

## 4. `benchmark_outcomes.py` Workflows & Lockfile Integrity

All 4 Category 5 datasets are pinned in `configs/outcome_benchmarks.lock.json` (`20` total datasets) with 40-char commit SHAs, file sizes, and SHA-256 checksums:

```bash
# 1. Stage & SHA-256 verify Tier-1 OCR datasets (~450 MB total, 0 GPU work)
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier1 --datasets olmocr_bench,fintabnet_otsl,omnidocbench_en,cord_v2

# 2. Apples-to-apples visual OCR run (disables PDF text layer so native layers don't mask visual OCR)
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --smoke --local-files-only \
  --packages fast-ocr,public-apple-vision,public-pypdfium2 \
  --extra-args "--ocr-backend ovis --ocr-render-only" --json-out ocr_cat5_smoke.json

# 3. Structured receipt JSON KIE with NuExtract3
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --packages fast-ocr \
  --extra-args "--ocr-model nuextract3 --ocr-suites cord_v2"
```

---

## 5. Cross-Silicon Verification Receipts (`Apple M4 Pro` & `Base M5`)

| Host | Suite / Package (`smoke n=2`, `--ocr-backend ovis --ocr-render-only`) | Valid | `S_obj,e2e` | `EOT` (u/s) | Wall (s) | 4-Bucket Mem (GB) | Key Sub-Metrics / Gate |
|---|---|---:|---:|---:|---:|---:|---|
| **M4 Pro** (`macOS 26.6`) | `fast-ocr` · `FinTabNet-OTSL(2)` | `2/2` | **`0.8687`** | `0.635` | `2.74` | `3.243` | `TEDS=0.8687`, `TEDS-S=0.9085`, `GriTS-Top=0.9435`, `NumF1=0.9299` · **`PASS`** |
| **M4 Pro** (`macOS 26.6`) | `fast-ocr` · `OmniDocBench-EN(2)` | `2/2` | **`0.9093`** | `0.126` | `14.49` | `7.269` | `overall=0.9281`, `text=0.9219`, `order=1.0`, `table=0.9726`, `formula=0.8898` · **`PASS`** |
| **M4 Pro** (`macOS 26.6`) | `public-apple-vision` · `FinTabNet-OTSL(2)` | `2/2` | `0.5161` | `3.830` | `0.27` | `1.452` | `TEDS=0.5161`, `NumF1=0.6788` · `PASS` |
| **M4 Pro** (`macOS 26.6`) | `public-apple-vision` · `OmniDocBench-EN(2)` | `2/2` | `0.2666` | `0.479` | `1.11` | `1.173` | `overall=0.3445`, `formula=0.0` · `PASS` |
| **Base M5** (`macOS 27.2.1`) | `fast-ocr` · `FinTabNet-OTSL(2)` | `2/2` | **`0.8687`** | `0.607` | `2.86` | `3.038` | `TEDS=0.8687`, `TEDS-S=0.9085`, `GriTS-Top=0.9435`, `NumF1=0.9299` · **`PASS`** |
| **Base M5** (`macOS 27.2.1`) | `fast-ocr` · `OmniDocBench-EN(2)` | `2/2` | **`0.9093`** | `0.113` | `16.04` | `6.312` | `overall=0.9281`, `text=0.9219`, `order=1.0`, `table=0.9726`, `formula=0.8898` · **`PASS`** |
| **Base M5** (`macOS 27.2.1`) | `public-apple-vision` · `FinTabNet-OTSL(2)` | `0/2` | `0.0000` | `0.000` | `0.09` | `1.372` | Table formatting missed on crop (`0/2` valid) · `FAIL` |
| **Base M5** (`macOS 27.2.1`) | `public-apple-vision` · `OmniDocBench-EN(2)` | `2/2` | `0.3169` | `2.211` | `0.29` | `1.114` | `overall=0.4175`, `formula=0.0` · `PASS` |

Additional cross-silicon suite checks verified:
- **M4 Pro (`Mac16,8`):** `make sync-check` (`copied=0 removed=0 unchanged=85`), `make guard` (`Summary: 0 errors, 0 warnings`), `make leak-check` (clean), `make test-cpu` (`384 passed, 3 skipped`, `22 passed`, `59 passed, 2 skipped`), `test_outcome_eval_cpu.py` (`29 passed in 1.56s`), `test_fast_pplx_vision_hw.py` (`1 passed in 0.74s`). Note earlier `olmOCR-bench(4)` and `CORD-v2(2)` smoke checks are also supported in `scripts/benchmark_outcomes.py`.
- **Base M5 (`Mac17,2`):** `test_qwen35_vision_cpu.py + test_outcome_eval_cpu.py + test_fast_pplx_vision_cpu.py + test_fast_qwen3_cpu.py + test_fast_ocr_cpu.py` (`61 passed in 5.89s`), `test_fast_pplx_vision_hw.py` (`1 passed in 0.50s`).
