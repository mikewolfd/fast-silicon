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

### 1C. Government, Regulatory & Legal Suites (`Cat 5 Gov-Doc`)

| Suite Key | Source / Repo | Branch & Content | License | Core Metrics | Notes & Ground-Truth Hygiene |
|---|---|---|---|---|---|
| **`doclaynet_gov`** | `docling-project/DocLayNet-v1.2` (`test-00003`, `test-00004`) | Legislative/Executive/Procurement (`laws_and_regulations`, `government_tenders`, `patents`), English/Latin-filtered. | `CDLA-Permissive-1.0` | `doclaynet_overall` (`1 - Text_NED`, reading-order concordance, header/footer leak rate). | Pinned in `outcome_benchmarks.lock.json` (`tier1: n=0`, `tier2: n=90`). Scores against PDF digital `pdf_cells` (no pseudo-OCR GT). |
| **`vrdu_ad_buy`** / **`vrdu_registration`** | `google-research-datasets/vrdu` (GitHub commit `04a4df7`) | Executive regulatory filings: 641 FCC political ad-buy PDFs (nested `line_items`) + 1,915 DOJ FARA registration PDFs. | `CC-BY-4.0` / U.S. Pub Domain | **`grounded_field_f1`** + leaf **ANLS** with VRDU `PriceMatch` / `DateMatch` normalization. | Pinned in `outcome_benchmarks.lock.json`. Evaluates human KIE annotations visible on the primary annotated page; ignores `dataset.jsonl.gz` Google Cloud OCR tokens. |
| **`puredocbench_legal_gov`** | `zhihengli-casia/puredocbench` (`03_legal_gov`) | Legal & government pages (`72` English / `151` total across *Clean*, *Digital*, *Real* tracks). | `CC-BY-4.0` (data) / `MIT` (code) | `puredocbench_overall` (`Text_NED`, `Table_TEDS`, `Formula_NormSim`, reading order). | GT tarball pinned (`tier1/2: n=0`); requires extracting `pdb_full.tar` locally (`--puredocbench-images-root`). |
| **`us_gov_ocr`** | GovInfo (`PLAW` USLM, `BILLS`, `USCODE`, `FR`, `BUDGET` `.xlsx`, `USCOURTS`), Federal Register API, `incrediblecrab/us-scotus-opinions`, LoC `btp-data` | All 3 U.S. branches: `PLAW`/`BILLS`/`USCODE` XML/HTML↔PDF, 3-column `FR` `<GPOTABLE>`, single-page OMB `BUDGET` `.xlsx`↔PDF tables, `USCOURTS` + byte-for-byte `supremecourt.gov` slip PDFs, and LoC `btp-data` archival scans. | **U.S. Public Domain / CC0** (`17 U.S.C. § 105`) | `us_gov_overall` (`OmniDocBench` suite + per-branch & per-source breakdowns). | Built via `stage-gov` (`--sources`, `--loc-btp-path`, `--no-loc-btp`); auto-selects `<GPOTABLE>` / single-page OMB `.xlsx` table pages, checks `xcheck_equal` + `image_pages==0` on SCOTUS PDFs, forces `--ocr-render-only`, and filters raster-only items on `public-pypdfium2`. |

**Excluded Pseudo-GT & Non-Benchmark Corpora (Never Score Clean VLMs Against These):**
- *Automated OCR outputs masquerading as GT:* Harvard CAP `case.law` `casebody`/ALTO (raw ABBYY FineReader OCR; only metadata was human-reviewed), Chronicling America bulk ALTO XML, `aurelio-ai/jfk-files` / `HistoryLab/foiarchive`, `GovInfo USREPORTS` (lacks paired XML and exhibits `frst`/hyphen PDF ligature drops), and VRDU's token-level `ocr` field (Google Cloud OCR).
- *Text-only / non-OCR-text / restricted datasets:* `pile-of-law/pile-of-law`, `common-pile/usgpo`, `free-law/Caselaw_Access_Project`, `mauricedalton/irs-990` (XML/Parquet without paired PDFs), `cernis-intelligence/cernis-legal-ocr` (model weights), `CommonForms` (AcroForm widget bboxes only), `UnbrokenCocoon/Hansard-OCR-Dataset` (synthetic renders), `FUNSD` (non-commercial), `Kleister` (no LICENSE file in repo root), `LeakageBench` (privacy DUA), and `DUDE`/`DocVQA`/`ViDoRe` (QA/retrieval only).
- *Upstream endpoints replaced via public-domain equivalents:* Direct `supremecourt.gov/xmls/current/` (HTML/JS challenge) and `uscode.house.gov/download/download.shtml` (`404`) are bypassed via `incrediblecrab/us-scotus-opinions` and `GovInfo USCODE` (`chap*.htm`↔`.pdf`), while un-paired `BUDGET-2026-BUD.zip` is replaced by `BUDGET-2027-PER` single-page PDF↔`.xlsx` table alignment (`mikewolfd/fast-silicon#1`).

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

All Category 5 datasets are pinned in `configs/outcome_benchmarks.lock.json` (`24` total datasets) with 40-char commit SHAs, file sizes, and SHA-256 checksums (plus optional keyless 3-branch Public-Domain `us_gov_ocr` staging):

```bash
# 1. Stage & SHA-256 verify core Tier-1 OCR datasets (~450 MB total, 0 GPU work)
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier1 --datasets olmocr_bench,fintabnet_otsl,omnidocbench_en,cord_v2

# 2. Stage Tier-1 VRDU (DOJ FARA + FCC Ad-Buy) & Tier-2 DocLayNet-Gov shards, plus paired 3-branch U.S. Gov PDFs
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier1 --datasets vrdu_ad_buy,vrdu_registration
.venv/bin/python scripts/benchmark_outcomes.py setup-data --tier tier2 --datasets doclaynet_gov
.venv/bin/python scripts/benchmark_outcomes.py stage-gov --branches legislative,executive,judicial --per-source 4
# Optional: restrict sources or override / skip LoC By the People archival scans
.venv/bin/python scripts/benchmark_outcomes.py stage-gov --branches executive --sources fr,budget --per-source 3 --no-loc-btp
.venv/bin/python scripts/benchmark_outcomes.py stage-gov --branches executive --per-source 3 --loc-btp-path /path/to/btp.parquet

# 3. Apples-to-apples visual OCR run (disables PDF text layer so native layers don't mask visual OCR)
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --smoke --local-files-only \
  --packages fast-ocr,public-apple-vision,public-pypdfium2 \
  --extra-args "--ocr-backend ovis --ocr-render-only" --json-out ocr_cat5_smoke.json

# 4. Run Gov-Doc visual OCR suites (`doclaynet_gov`, `us_gov_ocr`), KIE (`vrdu_*`), or all available suites
# Note: `--ocr-suites all` safely skips un-staged optional suites (`doclaynet_gov` on tier1 unless cached, `us_gov_ocr` without manifest, `puredocbench_legal_gov` without extracted images).
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier2 --smoke --local-files-only \
  --packages fast-ocr,public-pypdfium2 --extra-args "--ocr-backend ovis --ocr-render-only --ocr-suites gov"
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier1 --smoke --local-files-only \
  --packages fast-ocr --extra-args "--ocr-model nuextract3 --ocr-suites vrdu_ad_buy,vrdu_registration"
.venv/bin/python scripts/benchmark_outcomes.py run --tier tier2 --smoke --local-files-only \
  --packages fast-ocr --extra-args "--ocr-backend ovis --ocr-render-only --ocr-suites puredocbench_legal_gov --puredocbench-track clean --puredocbench-images-root /path/to/pdb_images"

# 5. Structured receipt JSON KIE with NuExtract3
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
- **Gov-Doc CPU Harness & M5 Bulk Staging (`M4 Pro` + `Base M5`):** `test_outcome_eval_cpu.py` + `test_benchmark_suites_cpu.py` (`63 passed in 20.30s`), mirrored `fast-shared` `test_outcome_eval_cpu.py` (`33 passed in 0.93s`). On Base M5 (`423Gi` free remaining, 0 GPU work): staged Tier-1 VRDU + all 23 non-zero Tier-2 lockfile datasets (`doclaynet_gov` `90` pp, `vrdu_ad_buy`/`vrdu_registration` `100` each, `extractbench`, `fintabnet_otsl`, `omnidocbench_en`, `olmocr_bench`, `cord_v2`), extracted all `453` PureDocBench `03_legal_gov` images (`72/72` EN verified on `clean`/`digital`/`real`), and staged `us_gov_ocr` across all 8 sources (`31` verified items: `PLAW` 4, `BILLS` 4, `USCODE` 4, `FR` 4, `BUDGET` 3, `LoC btp-data` 4, `USCOURTS` 4, `SCOTUS` 4) with `public-pypdfium2` CPU baseline scoring `US-Gov-OCR(27)` (`govinfo_budget=0.0104`, `govinfo_bills=0.2073`, `govinfo_fr=0.5135`, `govinfo_uscode=0.5761`, `govinfo_plaw=0.6627`, `govinfo_uscourts=1.0`, `scotus_opinions=1.0`). M4 Pro / Base M5 GPU & ANE runs on the new Gov-Doc suites remain pending under active GPU validation locks (`mikewolfd/fast-silicon#1`).
- **Base M5 (`Mac17,2`):** `test_qwen35_vision_cpu.py + test_outcome_eval_cpu.py + test_fast_pplx_vision_cpu.py + test_fast_qwen3_cpu.py + test_fast_ocr_cpu.py` (`61 passed in 5.89s`), `test_fast_pplx_vision_hw.py` (`1 passed in 0.50s`).
