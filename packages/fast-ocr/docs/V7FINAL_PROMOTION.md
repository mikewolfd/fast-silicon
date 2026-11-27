# OvisOCR2 v7final Merged 8-bit GFM Promotion Checklist

Status: validated locally (`verdict: "PROMOTE"`), awaiting Hub publication / Base M5 verification to flip `DEFAULT_OVIS_REPO`.

## Validated Artifact Locations

- Merged 8-bit checkpoint directory (`ovisocr2_merged_gfm_v7final_8bit`, `group_size=64`, 187 Q8 LM stems + 153 BF16 visual tensors, `1.0012 GB`):
  referenced via `$FAST_OCR_GFM_MODEL_DIR`, `~/.cache/fast-silicon/ovisocr2_merged_gfm_v7final_8bit`, or HF Hub repo `mikewolfd/OvisOCR2-8bit-gfm` (`models--mikewolfd--OvisOCR2-8bit-gfm`).
- Source PEFT LoRA adapter (`lora_adapter_v7_final`, `rank=8`, `alpha=16.0`, `2,757,120` trainable params) merged offline into base `mlx-community/OvisOCR2-8bit` before 8-bit requantization.
- Merge equivalence receipt:
  `logits_cosine_sim=0.999622`, `top1_match=True`, `prefix_60c_match=True`, `peak_gb=2.0279`
- Gate results (`receipt.json` vs `de8ed54` baseline):
  - `FinTabNet-OTSL(30)`: `18.89s` (`1.302x` speedup), tokens `11,911 -> 8,149` (`-31.6%`), `TEDS 0.9020 -> 0.9031`, `NumericF1 0.8888 -> 0.9184`, `raw_html_tables=0`, `repeat_loop_count=0`
  - `OmniDocBench-EN(12)`: `0.8158 -> 0.8234` (`+0.0076`), `table_teds 0.8772 -> 0.8998`, `repeat_loop_count=0`
  - `olmOCR-bench(16)`: `0.6993 -> 0.6993` (`0` regression), `repeat_loop_count=0`
  - Peak inference RAM: `4.955 GB`

## Opt-In Paths Already Wired in `packages/fast-ocr`

1. Preset alias: `--ocr-model v7final` or `FastOCR(model_path="v7final")` / `FastOCR(model_path="ovis-gfm")` resolves via `paths.resolve_ocr_model_dir`.
2. Environment override: `FAST_OCR_GFM_MODEL_DIR=/path/to/ovisocr2_merged_gfm_v7final_8bit` + `FAST_OCR_PREFER_GFM=1` (cleanly falls back to `mlx-community/OvisOCR2-8bit` only when `FAST_OCR_PREFER_GFM=1` and `v7final` is absent locally).
3. Local cache stage path: `~/.cache/fast-silicon/ovisocr2_merged_gfm_v7final_8bit` or HF Hub snapshot `models--mikewolfd--OvisOCR2-8bit-gfm`.

## Final Default-Flip Checklist (When Published to Hub)

1. Upload `ovisocr2_merged_gfm_v7final_8bit/` to `mikewolfd/OvisOCR2-8bit-gfm` (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `preprocessor_config.json`, `video_preprocessor_config.json`, `model.safetensors`).
2. Verify Base M5 full-tier outcomes (`olmOCR-bench(42)`, `OmniDocBench-EN(24)`, `FinTabNet-OTSL(30)`) with `FAST_OCR_GFM_MODEL_DIR` and `local_files_only=True`.
3. Flip `DEFAULT_OVIS_REPO = DEFAULT_OVIS_GFM_REPO` in `packages/fast-ocr/src/fast_ocr/paths.py` and update `_OCR_HF_REPO` + `VETTED_PREBUILT_HF_REPOS["ocr"]` in `scripts/setup_all.py`.
