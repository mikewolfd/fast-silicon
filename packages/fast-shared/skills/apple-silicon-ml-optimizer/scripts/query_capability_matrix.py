#!/usr/bin/env python3
"""Query the merged M4/M5 hardware, ISA, Metal, MLX, Core ML, Core AI, and SDK matrix."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SKILL_DIR / "references" / "data"
CURATED_PATH = DATA_DIR / "matrix_curated.json"

CATALOG_MAP: dict[str, str] = {
    "instruction_differences": "catalog_instruction_differences.jsonl",
    "a64_encodings": "catalog_a64_encodings.jsonl",
    "metal_gpu": "catalog_metal_gpu.jsonl",
    "mlx_api": "catalog_mlx_api.jsonl",
    "apple_apis": "catalog_apple_apis.jsonl",
    "metal_sdk": "catalog_metal_sdk.jsonl",
    "bnns_graph_api": "catalog_bnns_graph_api.jsonl",
}

CURATED_SIX_CHIP = {
    "hardware",
    "cpu_isa",
    "instruction_forms",
    "cpu_measurements",
    "gpu_and_metal",
    "mlx",
    "core_ml",
    "core_ai",
    "other_frameworks",
}

CURATED_TOP_LEVEL = {
    "observed_hosts",
    "host_os_checks",
    "version_gates",
    "native_tests",
    "live_metal",
    "tensor_formats",
    "cpu_features",
    "darwin_readings",
    "sources_and_scope",
}

_OS_VERSION_COLUMNS: tuple[str, ...] = (
    "min_macos",
    "macos_requirement",
    "macos_api_floor",
    "minimum_macos",
    "storage_api_since",
    "arithmetic_operands_since",
    "scale_plane_since",
    "macOS / runtime requirement",
    "Minimum macOS",
    "Minimum macOS for API",
    "macOS / API floor",
    "Earliest tagged declaration found",
    "macOS and build",
)

_SIX_CHIP_COLUMNS: tuple[str, ...] = (
    "M4",
    "M4 Pro",
    "M4 Max",
    "M5",
    "M5 Pro",
    "M5 Max",
)

_HOST_CAPTURE_COLUMN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("m4_pro_26_6", "base_m5_26_2", "base_m5_27_2_1", "m5_26_2"),
    ("m4_pro_runtime_26_6", "base_m5_runtime_26_2", "base_m5_runtime_27_2_1"),
    ("m4_pro_48gb_26_6", "base_m5_24gb_26_2", "base_m5_24gb_27_2_1"),
    ("M4 Pro • 26.6", "Base M5 • 26.2", "Base M5 • 27.2.1"),
    ("m4_pro_apple9", "m5_apple10"),
    ("m4_pro_0_32_4", "m5_0_32_4"),
    ("m4_pro_26_6_storage", "base_m5_26_2_storage", "base_m5_27_2_1_storage"),
    ("m4_pro_gate_26_6", "m5_gate_26_2", "m5_gate_27_2_1"),
    ("m4_pro_clang", "m5_clang"),
    ("m4_pro_clang_target", "m5_clang_target"),
)

# Match macOS version numbers (10.x..15.x or 26.x..99.x, including "27", "27.x", "27.2+", "27.2.1")
# while ignoring MSL 4.0/4.1, Metal 4, MLX 0.32.4, and GPU generation >=17.
_NON_OS_PREFIX_RE = re.compile(
    r"\b(?:MSL|Metal|MLX|Xcode|LLVM|Clang|iOS|iPadOS|tvOS|visionOS|watchOS|generation)\s*[≥>=]*\s*\d+(?:\.\d+)*",
    re.I,
)
_OS_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(1[0-5]|2[6-9]|[3-9]\d)(?:\.(\d+)(?:\.(\d+))?|\.[xX])?(?:\+)?(?![A-Za-z0-9_.])"
)


def parse_os_tuple(ver_str: str) -> tuple[int, int, int]:
    """Parse a user-supplied macOS version floor like '26.2', '27', '27.0', '27.2.1'."""
    cleaned = str(ver_str).strip().rstrip("+")
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", cleaned)
    if not m:
        raise ValueError(f"Invalid --min-os version string {ver_str!r}; expected e.g. '26.2', '27', '27.0', '27.2.1'")
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def extract_os_versions(text: str) -> list[tuple[int, int, int]]:
    """Extract all macOS semantic version tuples from an OS-requirement field string."""
    if not text:
        return []
    scrubbed = _NON_OS_PREFIX_RE.sub(" ", text)
    out: list[tuple[int, int, int]] = []
    for m in _OS_TOKEN_RE.finditer(scrubbed):
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) is not None else 0
        patch = int(m.group(3)) if m.group(3) is not None else 0
        # Guard against bogus large standalone numbers like 100000 or random two-digit counts
        if major >= 40 and m.group(2) is None:
            continue
        out.append((major, minor, patch))
    return out


def load_curated() -> dict[str, Any]:
    """Load the normalized curated matrix JSON payload."""
    return json.loads(CURATED_PATH.read_text(encoding="utf-8"))


def iter_table_rows(curated: dict[str, Any], table: str) -> list[dict[str, Any]]:
    """Yield rows from either a curated JSON section or one of the JSONL catalogs."""
    if table in CURATED_SIX_CHIP:
        return list(curated["six_chip_matrices"][table]["rows"])
    if table == "encoder_timings":
        enc = curated["encoder_timings"]
        return [
            {"_subtable": "ocr_timings_unrounded", **r} for r in enc["ocr_timings_unrounded"]
        ] + [
            {"_subtable": "encoder_results_summary", **r} for r in enc["encoder_results_summary"]
        ]
    if table in CURATED_TOP_LEVEL:
        return list(curated[table])
    if table in CATALOG_MAP:
        path = DATA_DIR / CATALOG_MAP[table]
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unknown table {table!r}")


def all_table_names() -> list[str]:
    """Return every supported curated and catalog table name in logical order."""
    return sorted(CURATED_SIX_CHIP) + sorted(CURATED_TOP_LEVEL) + ["encoder_timings"] + sorted(CATALOG_MAP)


def _row_has_chip_diff(row: dict[str, Any]) -> bool:
    """Return True if a row shows any difference across all 6 M4/M5 chips or host OS captures."""
    # 1. Six-chip matrix rows: compare across ALL 6 chips (M4, M4 Pro, M4 Max, M5, M5 Pro, M5 Max)
    six_vals = [
        str(row[col]).strip()
        for col in _SIX_CHIP_COLUMNS
        if col in row and row[col] not in ("", "—", None)
    ]
    if len(six_vals) >= 2:
        return len(set(six_vals)) > 1

    # 2. Explicit comparison annotations (instruction_differences, a64_encodings, live_metal, etc.)
    cmp_vals = [
        str(row[k]).strip().lower()
        for k in ("comparison_27_2_1", "comparison_26_2", "comparison_first_capture", "comparison")
        if k in row and row[k] not in ("", "—", None)
    ]

    # 3. Host/capture column groups (captures M4 Pro vs M5 AND 26.2 vs 27.2.1 OS evolution)
    group_saw_values = False
    for group in _HOST_CAPTURE_COLUMN_GROUPS:
        vals = [
            str(row[k]).strip()
            for k in group
            if k in row and row[k] not in ("", "—", None)
        ]
        if len(vals) >= 2:
            group_saw_values = True
            if len(set(vals)) > 1:
                return True

    if cmp_vals:
        joined = " ".join(cmp_vals)
        if any(tag in joined for tag in ("same", "both", "identical", "neither")):
            return False
        return True

    return not group_saw_values


def matches_filters(
    row: dict[str, Any],
    search_re: re.Pattern[str] | None,
    min_os: str | None,
    diff_only: bool,
) -> bool:
    """Apply optional regex search, semantic OS floor (`>=`) filter, and 6-chip difference filter."""
    blob = " | ".join(f"{k}={v}" for k, v in row.items() if v not in ("", "—", None))
    if search_re and not search_re.search(blob):
        return False
    if min_os:
        target_ver = parse_os_tuple(min_os)
        os_versions: list[tuple[int, int, int]] = []
        for k in _OS_VERSION_COLUMNS:
            val = row.get(k)
            if val not in ("", "—", None):
                os_versions.extend(extract_os_versions(str(val)))
        if not any(v >= target_ver for v in os_versions):
            return False
    if diff_only:
        if not _row_has_chip_diff(row):
            return False
    return True


def format_compact(row: dict[str, Any], chip: str | None = None) -> str:
    """Format a record into a compact readable summary line for CLI output."""
    if chip:
        chip_map = {
            "m4": ("M4", "m4"),
            "m4_pro": ("M4 Pro", "m4_pro_26_6", "m4_pro_48gb_26_6", "m4_pro_gate_26_6", "m4_pro_apple9", "m4_pro_runtime_26_6", "m4_pro_0_32_4"),
            "m4_max": ("M4 Max", "m4_max"),
            "m5": ("M5", "Base M5", "base_m5_26_2", "base_m5_27_2_1", "m5_gate_27_2_1", "m5_apple10", "base_m5_runtime_27_2_1", "m5_0_32_4"),
            "m5_pro": ("M5 Pro",),
            "m5_max": ("M5 Max",),
        }
        keys = chip_map.get(chip, ())
        subset = {k: v for k, v in row.items() if k in keys or k in ("Domain", "Capability / form", "feature", "symbol", "public_name", "mnemonic", "encoding_id", "arm_feature", "darwin_key", "min_macos", "macos_requirement", "function")}
        return json.dumps(subset or row, ensure_ascii=False)
    compact = {k: v for k, v in row.items() if v not in ("", "—", None)}
    return json.dumps(compact, ensure_ascii=False)


def print_stats(curated: dict[str, Any], as_json: bool) -> None:
    """Print dataset inventory, verified local host facts, and validation audit counts."""
    counts = {name: len(iter_table_rows(curated, name)) for name in all_table_names()}
    payload = {
        "metadata": curated["metadata"],
        "table_row_counts": counts,
        "total_records": sum(counts.values()),
    }
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    meta = curated["metadata"]
    host = meta["verified_local_host"]
    print(f"Merged Matrix Title : {meta['title']}")
    print(f"Verified Local Host : {host['model']} | {host['chip']} | macOS {host['macos']} | Xcode {host['xcode']} | MLX {host['mlx']}")
    print(f"Validation Audit    : A64={meta['validation_summary']['a64_all_encodings_rows']} (0 bool errors) | M5-only 26.2->27.2.1: {meta['validation_summary']['instruction_diff_26_2_m5_only']}->{meta['validation_summary']['instruction_diff_27_2_1_m5_only']} | Apple APIs dups collapsed={meta['validation_summary']['apple_apis_exact_duplicates_collapsed']}")
    print("\nTables:")
    for name, cnt in counts.items():
        print(f"  {name:26s} {cnt:6d}")
    print(f"  {'TOTAL':26s} {sum(counts.values()):6d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="Print table inventory and validation summary.")
    parser.add_argument(
        "--table",
        default="all",
        help="Target table name (or 'all' to search curated + catalog tables).",
    )
    parser.add_argument("--search", help="Case-insensitive regex or substring filter.")
    parser.add_argument(
        "--chip",
        choices=("m4", "m4_pro", "m4_max", "m5", "m5_pro", "m5_max"),
        help="Focus JSON output fields on a specific Apple Silicon SKU.",
    )
    parser.add_argument("--min-os", help="Filter rows requiring macOS >= VERSION (e.g. 26.2, 26.4, 27.0, 27.2).")
    parser.add_argument("--diff-only", action="store_true", help="Show records highlighting differences across all 6 M4/M5 chips or 26.2 vs 27.2.1.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum matching rows to display (default: 25, 0 = unlimited).")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    args = parser.parse_args(argv)

    if args.min_os:
        try:
            parse_os_tuple(args.min_os)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    curated = load_curated()
    if args.stats:
        print_stats(curated, args.json)
        return 0

    tables = all_table_names() if args.table == "all" else [args.table]
    for t in tables:
        if t not in all_table_names():
            print(f"Error: unknown table {t!r}. Valid tables: {', '.join(all_table_names())}", file=sys.stderr)
            return 2

    search_re = re.compile(args.search, re.I) if args.search else None
    results: list[dict[str, Any]] = []
    total_matches = 0
    for t in tables:
        for row in iter_table_rows(curated, t):
            if matches_filters(row, search_re, args.min_os, args.diff_only):
                total_matches += 1
                if args.limit == 0 or len(results) < args.limit:
                    results.append({"_table": t, **row})

    if args.json:
        print(json.dumps({"total_matches": total_matches, "shown": len(results), "results": results}, indent=2, ensure_ascii=False))
        return 0

    print(f"Matches: {total_matches} (showing {len(results)})")
    for item in results:
        tbl = item.pop("_table")
        print(f"[{tbl}] {format_compact(item, chip=args.chip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
