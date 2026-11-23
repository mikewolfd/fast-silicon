"""Command-line entrypoint (`fast-ocr run|bench|verify|parquet`) for Apple Silicon OCR."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .markdown_struct import (
    build_gfm_table_lock_lark,
    canonicalize_markdown,
    synthesize_markdown_from_lines,
    validate_markdown,
)
from .pipeline import FastOCR
from .preprocess import (
    DEFAULT_MAX_PDF_PIXELS,
    build_qwen_vision_inputs,
    detect_ink_bbox,
    rotate_box_normalized,
    smart_resize_qwen,
)
from .types import OCRBox, OCRLine, OCRResult

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})
_PDF_SUFFIXES = frozenset({".pdf"})
_SUPPORTED_SUFFIXES = _IMAGE_SUFFIXES | _PDF_SUFFIXES


def _resolve_input_paths(raw_inputs: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in raw_inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            children = sorted(
                c for c in p.iterdir() if c.is_file() and c.suffix.lower() in _SUPPORTED_SUFFIXES
            )
            if not children:
                raise FileNotFoundError(f"No supported image or PDF files found in directory: {p}")
            resolved.extend(children)
        elif p.is_file():
            resolved.append(p)
        else:
            raise FileNotFoundError(f"Input path does not exist: {p}")
    return resolved


def _add_shared_ocr_args(p: argparse.ArgumentParser, *, allow_all_backend: bool = False) -> None:
    backend_choices = ("apple-vision", "ovis", "auto", "all") if allow_all_backend else ("apple-vision", "ovis", "auto")
    p.add_argument(
        "--backend",
        choices=backend_choices,
        default="auto",
        help="OCR backend (`apple-vision`, `ovis`, `auto`" + (", or `all`" if allow_all_backend else "") + ")",
    )
    p.add_argument(
        "--format",
        dest="output_format",
        choices=("markdown", "raw", "text"),
        default="markdown",
        help="Output representation (`markdown`, `raw`, or `text`)",
    )
    p.add_argument(
        "--table-mode",
        choices=("gfm", "html"),
        default="gfm",
        help="Table serialization style (`gfm` or `html`)",
    )
    p.add_argument(
        "--strict-markdown-grammar",
        action="store_true",
        help="Enable opt-in JIT `llguidance` column-count lock during VLM table decoding",
    )
    p.add_argument(
        "--zoom-recover",
        action="store_true",
        help="Enable opt-in digit-stem filtered ANE-vs-VLM missing-number 2x zoom crop recovery",
    )
    p.add_argument("--model", default=None, help="Optional local path or Hugging Face preset/repo for Qwen3.5 VLM")
    p.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for scanned PDF pages (default: 300)")
    p.add_argument(
        "--triage-dpi",
        type=int,
        default=None,
        help="Optional low DPI (e.g. 100) for initial ANE screening on PDFs under `--backend auto`",
    )
    p.add_argument(
        "--bands",
        type=int,
        default=None,
        help="Horizontal strip count for Apple Vision (default: 1 standalone, 3 in auto router)",
    )
    p.add_argument("--level", choices=("accurate", "fast"), default="accurate", help="Apple Vision recognition level")
    p.add_argument(
        "--no-language-correction",
        action="store_true",
        help="Disable Apple Vision language correction even on standalone apple-vision backend",
    )
    p.add_argument(
        "--max-pdf-pixels",
        type=int,
        default=DEFAULT_MAX_PDF_PIXELS,
        help=f"Maximum raster pixels per PDF page before adaptive DPI downscale (<= 0 disables; default: {DEFAULT_MAX_PDF_PIXELS})",
    )
    p.add_argument("--no-crop", action="store_true", help="Disable tight foreground ink-bbox cropping before VLM ViT")
    p.add_argument("--no-orient", action="store_true", help="Disable automatic 90-degree orientation detection/rotation")
    p.add_argument(
        "--speculative",
        choices=("auto", "off", "self", "vision+self"),
        default="auto",
        help="Speculative decoding strategy for Qwen3.5 VLM",
    )
    p.add_argument("--draft-k", type=int, default=4, help="Maximum speculative draft lookahead tokens (default: 4)")
    p.add_argument("--max-tokens", type=int, default=8192, help="Maximum generated tokens per page for VLM backend")
    p.add_argument("--vocab-size", type=int, default=None, help="Optional pruned output vocab size for VLM")
    p.add_argument("--vit-quant-bits", type=int, default=None, help="Optional ViT linear quantization bit width")
    p.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="Optional MLX memory limit in GiB (default: auto host-budget cap, e.g. 19.2 GiB on 24 GiB Macs; <= 0 disables)",
    )
    p.add_argument(
        "--cache-limit-gb",
        type=float,
        default=None,
        help="Optional MLX buffer cache limit in GiB (default: auto host-budget cap)",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for multi-image/PDF double-buffered routing")


def _build_ocr_from_args(args: argparse.Namespace, *, backend_override: str | None = None) -> FastOCR:
    return FastOCR(
        backend=backend_override or args.backend,  # type: ignore[arg-type]
        output_format=args.output_format,
        table_mode=args.table_mode,
        strict_markdown_grammar=bool(args.strict_markdown_grammar),
        zoom_recover=bool(args.zoom_recover),
        model_path=args.model,
        bands=int(args.bands) if args.bands is not None else None,
        level=args.level,
        language_correction=(False if bool(getattr(args, "no_language_correction", False)) else None),
        crop_margins=not bool(args.no_crop),
        auto_orient=not bool(args.no_orient),
        speculative=args.speculative,
        vocab_size=args.vocab_size,
        vit_quant_bits=args.vit_quant_bits,
        draft_k=int(args.draft_k),
        max_tokens=int(args.max_tokens),
        memory_limit_gb=getattr(args, "memory_limit_gb", None),
        cache_limit_gb=getattr(args, "cache_limit_gb", None),
    )


def _run_inputs(
    ocr: FastOCR,
    paths: Sequence[Path],
    *,
    dpi: int,
    triage_dpi: int | None,
    batch_size: int,
    max_pdf_pixels: int | None = DEFAULT_MAX_PDF_PIXELS,
) -> list[tuple[Path, OCRResult]]:
    """Run `ocr` across mixed image and PDF paths in exact input order."""
    records: list[tuple[Path, OCRResult]] = []
    pending_imgs: list[Path] = []

    def _flush_images() -> None:
        if not pending_imgs:
            return
        outs = ocr.ocr_images(pending_imgs, batch_size=batch_size)
        for p, res in zip(pending_imgs, outs):
            records.append((p, res))
        pending_imgs.clear()

    for p in paths:
        if p.suffix.lower() in _PDF_SUFFIXES:
            _flush_images()
            eff_max_px = None if (max_pdf_pixels is not None and int(max_pdf_pixels) <= 0) else max_pdf_pixels
            pdf_kw: dict[str, Any] = {"dpi": dpi, "batch_size": batch_size, "max_pixels": eff_max_px}
            if triage_dpi is not None:
                pdf_kw["triage_dpi"] = int(triage_dpi)
            pdf_outs = ocr.ocr_pdf(p, **pdf_kw)
            for res in pdf_outs:
                records.append((p, res))
        else:
            pending_imgs.append(p)
    _flush_images()
    return records


def _serialize_result(path: Path, res: OCRResult) -> dict[str, Any]:
    val_dict = asdict(res.markdown_validation) if res.markdown_validation is not None else None
    analysis_dict = asdict(res.analysis) if res.analysis is not None else None
    return {
        "input": str(path),
        "page_index": res.page_index,
        "backend_used": res.backend_used,
        "output_format": res.output_format,
        "table_mode": res.table_mode,
        "rotation_applied_cw": res.rotation_applied_cw,
        "cropped_box_px": list(res.cropped_box_px) if res.cropped_box_px is not None else None,
        "finish_reason": res.finish_reason,
        "text": res.text,
        "raw_text": res.raw_text,
        "markdown_validation": val_dict,
        "analysis": analysis_dict,
        "timings": asdict(res.timings),
        "metadata": dict(res.metadata),
    }


def _write_run_output(
    out_target: str,
    records: Sequence[tuple[Path, OCRResult]],
    *,
    payload: dict[str, Any],
    output_format: str,
    as_json: bool,
) -> None:
    out_path = Path(out_target).expanduser()
    ext = ".md" if output_format == "markdown" else ".txt"
    is_dir_target = out_target.endswith(("/", "\\")) or out_path.is_dir()

    if is_dir_target:
        out_path.mkdir(parents=True, exist_ok=True)
        grouped: dict[Path, list[OCRResult]] = {}
        for src, res in records:
            grouped.setdefault(src, []).append(res)
        for src, items in grouped.items():
            target_file = out_path / f"{src.stem}{ext}"
            target_file.write_text("\n\n".join(item.text for item in items), encoding="utf-8")
        if as_json:
            (out_path / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_json or out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        out_path.write_text("\n\n".join(res.text for _, res in records), encoding="utf-8")


def _cmd_run(args: argparse.Namespace) -> int:
    paths = _resolve_input_paths(args.inputs)
    with _build_ocr_from_args(args) as ocr:
        records = _run_inputs(
            ocr,
            paths,
            dpi=int(args.dpi),
            triage_dpi=args.triage_dpi,
            batch_size=int(args.batch_size),
            max_pdf_pixels=args.max_pdf_pixels,
        )

    payload = {
        "backend": args.backend,
        "output_format": args.output_format,
        "table_mode": args.table_mode,
        "page_count": len(records),
        "results": [_serialize_result(p, r) for p, r in records],
    }

    if args.out is not None:
        _write_run_output(
            args.out,
            records,
            payload=payload,
            output_format=args.output_format,
            as_json=bool(args.json),
        )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.out is None:
        print("\n\n".join(res.text for _, res in records))
    return 0


def _summarize_benchmark(
    backend: str,
    records: Sequence[tuple[Path, OCRResult]],
    *,
    wall_ms: float,
    repeats: int,
) -> dict[str, Any]:
    n_pages = len(records)
    timings = [r.timings for _, r in records]
    stage_keys = (
        "preprocess_ms",
        "apple_vision_ms",
        "vit_ms",
        "prefill_ms",
        "decode_ms",
        "postprocess_ms",
        "total_ms",
    )
    stage_sum_ms = {k: round(sum(float(getattr(t, k)) for t in timings), 3) for k in stage_keys}
    stage_mean_ms = {k: round(stage_sum_ms[k] / max(1, n_pages), 3) for k in stage_keys}

    prompt_tokens = sum(int(t.prompt_tokens) for t in timings)
    visual_tokens = sum(int(t.visual_tokens) for t in timings)
    generated_tokens = sum(int(t.generated_tokens) for t in timings)
    drafted_tokens = sum(int(t.drafted_tokens) for t in timings)
    accepted_tokens = sum(int(t.accepted_draft_tokens) for t in timings)
    decode_sec = stage_sum_ms["decode_ms"] / 1000.0
    decode_tps = (generated_tokens / decode_sec) if decode_sec > 0.0 else 0.0
    peak_mem = max((float(t.peak_memory_gb) for t in timings), default=0.0)

    checked = sum(1 for _, r in records if r.markdown_validation is not None)
    valid = sum(1 for _, r in records if r.markdown_validation is not None and r.markdown_validation.valid)
    tables = sum(r.markdown_validation.table_count for _, r in records if r.markdown_validation is not None)
    backend_counts = dict(Counter(r.backend_used for _, r in records))

    return {
        "backend": backend,
        "repeats": repeats,
        "pages": n_pages,
        "wall_ms": round(wall_ms, 3),
        "pages_per_sec": round((n_pages / (wall_ms / 1000.0)) if wall_ms > 0.0 else 0.0, 3),
        "stage_sum_ms": stage_sum_ms,
        "stage_mean_ms": stage_mean_ms,
        "tokens": {
            "prompt": prompt_tokens,
            "visual": visual_tokens,
            "generated": generated_tokens,
            "drafted": drafted_tokens,
            "accepted_draft": accepted_tokens,
            "decode_tps": round(decode_tps, 2),
        },
        "peak_memory_gb": round(peak_mem, 3),
        "backends_used": backend_counts,
        "validation": {
            "checked": checked,
            "valid": valid,
            "invalid": checked - valid,
            "tables": tables,
        },
    }


def _cmd_bench(args: argparse.Namespace) -> int:
    paths = _resolve_input_paths(args.inputs)
    if args.backends:
        selected = list(dict.fromkeys(args.backends))
    elif args.backend == "all":
        selected = ["apple-vision", "ovis", "auto"]
    else:
        selected = [args.backend]

    repeats = max(1, int(args.repeat))
    warmup = max(0, int(args.warmup))
    summaries: list[dict[str, Any]] = []

    for be in selected:
        with _build_ocr_from_args(args, backend_override=be) as ocr:
            for _ in range(warmup):
                _ = _run_inputs(
                    ocr,
                    paths,
                    dpi=int(args.dpi),
                    triage_dpi=args.triage_dpi,
                    batch_size=int(args.batch_size),
                    max_pdf_pixels=args.max_pdf_pixels,
                )
            all_records: list[tuple[Path, OCRResult]] = []
            t0 = time.perf_counter()
            for _ in range(repeats):
                all_records.extend(
                    _run_inputs(
                        ocr,
                        paths,
                        dpi=int(args.dpi),
                        triage_dpi=args.triage_dpi,
                        batch_size=int(args.batch_size),
                        max_pdf_pixels=args.max_pdf_pixels,
                    )
                )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            summaries.append(_summarize_benchmark(be, all_records, wall_ms=wall_ms, repeats=repeats))

    if args.json:
        print(json.dumps({"benchmarks": summaries}, indent=2))
    else:
        for s in summaries:
            st = s["stage_mean_ms"]
            val = s["validation"]
            print(
                f"[{s['backend']}] pages={s['pages']} wall={s['wall_ms']:.2f}ms "
                f"throughput={s['pages_per_sec']:.2f} pages/s "
                f"(prep={st['preprocess_ms']:.2f}ms vision={st['apple_vision_ms']:.2f}ms "
                f"vit={st['vit_ms']:.2f}ms prefill={st['prefill_ms']:.2f}ms "
                f"decode={st['decode_ms']:.2f}ms post={st['postprocess_ms']:.2f}ms) "
                f"valid={val['valid']}/{val['checked']} tables={val['tables']} "
                f"routes={s['backends_used']}"
            )
    return 0


def _check_markdown_sanity() -> dict[str, Any]:
    lines = [
        OCRLine(text="ANNUAL SUMMARY", confidence=0.99, box=OCRBox(0.30, 0.05, 0.70, 0.09)),
        OCRLine(text="- Revenue grew across all segments.", confidence=0.98, box=OCRBox(0.10, 0.14, 0.75, 0.16)),
        OCRLine(text="Item", confidence=0.99, box=OCRBox(0.10, 0.22, 0.25, 0.24)),
        OCRLine(text="Amount", confidence=0.99, box=OCRBox(0.65, 0.22, 0.85, 0.24)),
        OCRLine(text="Widgets", confidence=0.99, box=OCRBox(0.10, 0.26, 0.28, 0.28)),
        OCRLine(text="$120.50", confidence=0.99, box=OCRBox(0.65, 0.26, 0.85, 0.28)),
        OCRLine(text="Services", confidence=0.99, box=OCRBox(0.10, 0.30, 0.30, 0.32)),
        OCRLine(text="$80.00", confidence=0.99, box=OCRBox(0.65, 0.30, 0.85, 0.32)),
    ]
    md = synthesize_markdown_from_lines(lines)
    v_synth = validate_markdown(md, allow_html_tables=False)
    html_raw = "<table><tr><th colspan='2'>Totals</th></tr><tr><td>A</td><td>$10.00</td></tr></table>\n```py\nx=1"
    canon = canonicalize_markdown(html_raw, table_mode="gfm")
    v_canon = validate_markdown(canon, allow_html_tables=False)
    v_bad = validate_markdown("| a | b |\n| --- | --- |\n| 1 |", allow_html_tables=False)
    ok = v_synth.valid and v_synth.gfm_table_count == 1 and v_canon.valid and (not v_bad.valid)
    return {
        "name": "markdown_validator",
        "ok": ok,
        "synthesized_tables": v_synth.gfm_table_count,
        "canonicalized_valid": v_canon.valid,
        "malformed_detected": not v_bad.valid,
    }


def _check_preprocess_sanity() -> dict[str, Any]:
    h, w = smart_resize_qwen(240, 310, factor=32, min_pixels=65536, max_pixels=16777216)
    arr = np.full((256, 320, 3), 255, dtype=np.uint8)
    arr[96:160, 112:208] = 10
    img = Image.fromarray(arr, mode="RGB")
    bbox = detect_ink_bbox(img)
    vin = build_qwen_vision_inputs(img)
    rbox = rotate_box_normalized((0.1, 0.2, 0.3, 0.4), 90)
    ok = (
        h % 32 == 0
        and w % 32 == 0
        and bbox is not None
        and vin.grid_thw == (1, 16, 20)
        and vin.pixel_values.shape == (320, 1536)
        and len(rbox) == 4
    )
    return {
        "name": "preprocess_sanity",
        "ok": ok,
        "grid_thw": list(vin.grid_thw),
        "pixel_shape": list(vin.pixel_values.shape),
        "ink_bbox": list(bbox) if bbox is not None else None,
    }


def _check_grammar_and_tokenizer_sanity() -> dict[str, Any]:
    lark = build_gfm_table_lock_lark(3)
    try:
        import llguidance
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as exc:
        return {
            "name": "grammar_tokenizer",
            "ok": True,
            "skipped": True,
            "reason": str(exc),
            "lark_lines": len(lark.splitlines()),
        }

    from .markdown_struct import DynamicTableRowGrammar

    vocab: dict[str, int] = {"<|endoftext|>": 0}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llguidance.LLTokenizer(bt.to_str(), eos_token=0)
    gram = DynamicTableRowGrammar(ltok)
    header = "| colA | colB | colC |\n"
    ff_ids: list[int] = []
    for tid in bt.encode(header).ids:
        ff_ids.extend(gram.advance_text_and_maybe_lock(bt.decode([tid]), tid))
    sep_text = bt.decode(ff_ids) if ff_ids else ""
    ok = gram.is_active and gram.locked_cols == 3 and sep_text.startswith("| --- | --- | --- |\n")
    return {
        "name": "grammar_tokenizer",
        "ok": ok,
        "skipped": False,
        "locked_cols": gram.locked_cols,
        "ff_tokens": len(ff_ids),
    }


def _cmd_verify(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = [
        _check_markdown_sanity(),
        _check_preprocess_sanity(),
        _check_grammar_and_tokenizer_sanity(),
    ]

    if args.model is not None:
        from fast_qwen3.tokenizer import FastQwen3Tokenizer

        model_dir = Path(args.model).expanduser()
        if model_dir.exists() and (model_dir / "tokenizer.json").exists():
            tok = FastQwen3Tokenizer(model_dir)
            checks.append(
                {
                    "name": "model_tokenizer",
                    "ok": tok.vocab_size > 0,
                    "model_path": str(model_dir),
                    "vocab_size": tok.vocab_size,
                }
            )
        else:
            checks.append(
                {
                    "name": "model_tokenizer",
                    "ok": False,
                    "model_path": str(model_dir),
                    "error": "Model directory or tokenizer.json not found",
                }
            )

    if args.image is not None:
        with FastOCR(
            backend=args.backend,
            model_path=args.model,
            strict_markdown_grammar=bool(args.strict_markdown_grammar),
        ) as ocr:
            res = ocr.ocr_image(args.image)
            val_ok = True if res.markdown_validation is None else res.markdown_validation.valid
            checks.append(
                {
                    "name": "image_ocr",
                    "ok": bool(val_ok),
                    "backend_used": res.backend_used,
                    "chars": len(res.text),
                    "valid_markdown": val_ok,
                }
            )

    all_ok = all(bool(c.get("ok")) for c in checks)
    payload = {"ok": all_ok, "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for c in checks:
            status = "PASS" if c.get("ok") else "FAIL"
            print(f"[{status}] {c['name']}: {json.dumps({k: v for k, v in c.items() if k not in ('name', 'ok')})}")
    return 0 if all_ok else 1


def _cmd_parquet(args: argparse.Namespace) -> int:
    schema_arg: str | dict[str, Any] | None = None
    if args.schema:
        cand = Path(str(args.schema)).expanduser()
        schema_arg = cand.read_text(encoding="utf-8") if cand.is_file() else str(args.schema)
    max_px = int(args.max_pdf_pixels) if int(args.max_pdf_pixels) > 0 else None
    with _build_ocr_from_args(args) as ocr:
        summary = ocr.ocr_parquet(
            args.input,
            args.output,
            input_col=args.input_col,
            shard_rows=int(args.shard_rows),
            limit=int(args.limit) if args.limit is not None else None,
            resume=not bool(args.no_resume),
            compression=str(args.compression),
            dpi=int(args.dpi),
            triage_dpi=int(args.triage_dpi) if args.triage_dpi is not None else 72,
            max_pixels=max_px,
            batch_size=int(args.batch_size),
            prompt=args.prompt,
            schema=schema_arg,
            skip_errors=bool(args.skip_errors),
        )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[parquet] output={summary['output_path']} total_rows={summary['total_rows']} "
            f"processed={summary['processed_rows']} skipped={summary['skipped_rows']} "
            f"valid={summary['valid_rows']} total_ms={summary['total_ms']:.1f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the `fast-ocr` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fast-ocr",
        description="Zero-training Apple Silicon OCR (`AppleVisionOCR` + `FastOvisOCREngine` + `AutoOCRPipeline`)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run OCR on image(s), PDF(s), or a directory")
    p_run.add_argument("inputs", nargs="+", help="Image file(s), PDF file(s), or directory")
    _add_shared_ocr_args(p_run, allow_all_backend=False)
    p_run.add_argument("--out", default=None, help="Output file or directory path")
    p_run.add_argument("--json", action="store_true", help="Emit structured JSON output")
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="Benchmark OCR backend throughput, stage timings, and validation")
    p_bench.add_argument("inputs", nargs="+", help="Image file(s), PDF file(s), or directory")
    _add_shared_ocr_args(p_bench, allow_all_backend=True)
    p_bench.add_argument(
        "--backends",
        nargs="+",
        choices=("apple-vision", "ovis", "auto"),
        default=None,
        help="Optional explicit list of backends to compare",
    )
    p_bench.add_argument("--warmup", type=int, default=0, help="Warmup iterations per backend")
    p_bench.add_argument("--repeat", type=int, default=1, help="Timed iterations per backend")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON benchmark summary")
    p_bench.set_defaults(func=_cmd_bench)

    p_verify = sub.add_parser("verify", help="Run lightweight CPU sanity checks and optional model/image checks")
    p_verify.add_argument("--model", default=None, help="Optional model directory to verify tokenizer files")
    p_verify.add_argument("--image", default=None, help="Optional image file to run end-to-end verification")
    p_verify.add_argument(
        "--backend",
        choices=("apple-vision", "ovis", "auto"),
        default="apple-vision",
        help="Backend to use when `--image` is provided",
    )
    p_verify.add_argument("--strict-markdown-grammar", action="store_true")
    p_verify.add_argument("--json", action="store_true", help="Emit JSON verification receipt")
    p_verify.set_defaults(func=_cmd_verify)

    p_pq = sub.add_parser("parquet", help="Stream OCR over Parquet document/image rows with sharded checkpoints")
    p_pq.add_argument("input", help="Input Parquet file, directory, or glob")
    p_pq.add_argument("output", help="Output Parquet directory (sharded) or coalesced .parquet file path")
    _add_shared_ocr_args(p_pq, allow_all_backend=False)
    p_pq.add_argument("--input-col", default=None, help="Explicit column name (`pdf_path`, `image_path`, `image_bytes`)")
    p_pq.add_argument("--shard-rows", type=int, default=64, help="Rows per checkpointed Parquet shard (default: 64)")
    p_pq.add_argument("--limit", type=int, default=None, help="Optional maximum input rows to process")
    p_pq.add_argument("--no-resume", action="store_true", help="Reprocess all shards instead of resuming from manifest")
    p_pq.add_argument("--compression", default="zstd", help="Output Parquet compression codec (default: zstd)")
    p_pq.add_argument("--prompt", default=None, help="Optional OCR instruction override")
    p_pq.add_argument("--schema", default=None, help="Optional JSON schema string or path to .json file")
    p_pq.add_argument("--skip-errors", action="store_true", help="Record per-row errors instead of aborting")
    p_pq.add_argument("--json", action="store_true", help="Emit structured JSON summary")
    p_pq.set_defaults(func=_cmd_parquet)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `fast-ocr`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
