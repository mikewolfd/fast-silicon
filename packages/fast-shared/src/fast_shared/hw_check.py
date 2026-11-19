"""Host MacBook / Apple Silicon detection and tested-hardware warning helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import plistlib
import subprocess
import sys
from typing import Any

TESTED_REFERENCE_HARDWARE: dict[str, Any] = {
    "system": "Darwin",
    "machine": "arm64",
    "hw_model": "Mac16,8",
    "model_name": "14-inch MacBook Pro",
    "brand": "Apple M4 Pro",
    "mem_gb": 48,
    "cpu_cores": 14,
    "perf_cores": 10,
    "eff_cores": 4,
    "gpu_cores": 20,
    "ane_cores": 16,
}

_WARNED = False


def _sysctl(key: str) -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def _sysctl_int(key: str) -> int | None:
    raw = _sysctl(key)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _ioreg_first_plist(io_class: str) -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    try:
        raw = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", io_class, "-a"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        items = plistlib.loads(raw)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items[0]
    except Exception:
        pass
    return {}


def _probe_mem_bytes() -> int:
    mem = _sysctl_int("hw.memsize")
    if mem is not None and mem > 0:
        return mem
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_sz = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_sz > 0:
                return pages * page_sz
        except (ValueError, OSError):
            pass
    return 0


def probe_host_hardware() -> dict[str, Any]:
    """Probe local MacBook / host OS, chip, unified RAM, CPU P/E cores, GPU cores, and ANE cores."""
    sys_name = platform.system()
    machine = platform.machine()
    brand = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Unknown CPU"
    hw_model = _sysctl("hw.model") or "unknown"
    mem_bytes = _probe_mem_bytes()
    mem_gb = int(round(mem_bytes / (1024**3))) if mem_bytes > 0 else 0
    cpu_cores = _sysctl_int("hw.ncpu") or (os.cpu_count() or 0)
    perf_cores = _sysctl_int("hw.perflevel0.physicalcpu")
    eff_cores = _sysctl_int("hw.perflevel1.physicalcpu")

    agx = _ioreg_first_plist("AGXAccelerator")
    gpu_raw = agx.get("gpu-core-count")
    gpu_cores = int(gpu_raw) if isinstance(gpu_raw, int) else None

    ane = _ioreg_first_plist("H11ANEIn")
    ane_props = ane.get("DeviceProperties") if isinstance(ane.get("DeviceProperties"), dict) else {}
    ane_raw = ane_props.get("ANEDevicePropertyNumANECores")
    ane_cores = int(ane_raw) if isinstance(ane_raw, int) else None
    ane_arch = ane_props.get("ANEDevicePropertyTypeANEArchitectureTypeStr")

    return {
        "system": sys_name,
        "machine": machine,
        "hw_model": hw_model,
        "brand": brand,
        "mem_bytes": mem_bytes,
        "mem_gb": mem_gb,
        "cpu_cores": cpu_cores,
        "perf_cores": perf_cores,
        "eff_cores": eff_cores,
        "gpu_cores": gpu_cores,
        "ane_cores": ane_cores,
        "ane_arch": str(ane_arch) if ane_arch else None,
    }


def evaluate_hardware_support(
    hw: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Compare detected hardware against the tested 48 GB Apple M4 Pro MacBook Pro baseline."""
    info = hw if hw is not None else probe_host_hardware()
    ref = TESTED_REFERENCE_HARDWARE
    mismatches: list[str] = []

    if info.get("system") != ref["system"] or info.get("machine") != ref["machine"]:
        mismatches.append(f"platform={info.get('system')}/{info.get('machine')} (tested Darwin/arm64)")
    if info.get("brand") != ref["brand"]:
        mismatches.append(f"chip={info.get('brand')!r} (tested {ref['brand']!r})")
    if info.get("hw_model") != ref["hw_model"]:
        mismatches.append(f"model={info.get('hw_model')!r} (tested {ref['hw_model']!r})")
    if info.get("mem_gb") != ref["mem_gb"]:
        mismatches.append(f"memory={info.get('mem_gb')} GB (tested {ref['mem_gb']} GB)")
    if (
        info.get("cpu_cores") != ref["cpu_cores"]
        or info.get("perf_cores") != ref["perf_cores"]
        or info.get("eff_cores") != ref["eff_cores"]
    ):
        mismatches.append(
            f"cpu_cores={info.get('cpu_cores')} ({info.get('perf_cores')}P+{info.get('eff_cores')}E; "
            f"tested {ref['cpu_cores']} [{ref['perf_cores']}P+{ref['eff_cores']}E])"
        )
    if info.get("gpu_cores") != ref["gpu_cores"]:
        mismatches.append(f"gpu_cores={info.get('gpu_cores')} (tested {ref['gpu_cores']})")
    if info.get("ane_cores") != ref["ane_cores"]:
        mismatches.append(f"ane_cores={info.get('ane_cores')} (tested {ref['ane_cores']})")

    return (len(mismatches) == 0), mismatches, info


def format_hardware_summary(hw: dict[str, Any]) -> str:
    cpu_pe = (
        f"{hw['perf_cores']}P + {hw['eff_cores']}E"
        if hw.get("perf_cores") is not None and hw.get("eff_cores") is not None
        else "unknown P/E"
    )
    gpu_s = f"{hw['gpu_cores']} GPU cores" if hw.get("gpu_cores") is not None else "GPU cores unknown"
    ane_s = f"{hw['ane_cores']} ANE cores" if hw.get("ane_cores") is not None else "ANE cores unknown"
    return (
        f"{hw.get('hw_model')} ({hw.get('system')}/{hw.get('machine')}) | "
        f"{hw.get('brand')} | {hw.get('mem_gb')} GB RAM | "
        f"{hw.get('cpu_cores')} CPU cores ({cpu_pe}) | {gpu_s} | {ane_s}"
    )


def format_hardware_warning(hw: dict[str, Any], mismatches: list[str]) -> str:
    ref = TESTED_REFERENCE_HARDWARE
    tested_line = (
        f"{ref['hw_model']} {ref['model_name']} ({ref['system']}/{ref['machine']}) | "
        f"{ref['brand']} | {ref['mem_gb']} GB RAM | "
        f"{ref['cpu_cores']} CPU cores ({ref['perf_cores']}P + {ref['eff_cores']}E) | "
        f"{ref['gpu_cores']} GPU cores | {ref['ane_cores']} ANE cores"
    )
    return "\n".join(
        [
            "[fast-silicon] WARNING: Untested hardware configuration detected!",
            f"  Detected : {format_hardware_summary(hw)}",
            f"  Tested on: {tested_line}",
            f"  Differs  : {', '.join(mismatches)}",
            "  Note     : This monorepo was only tested on a 48 GB Apple M4 Pro MacBook Pro "
            f"({ref['hw_model']}) with {ref['cpu_cores']} CPU cores ({ref['perf_cores']}P + {ref['eff_cores']}E), "
            f"{ref['gpu_cores']} GPU cores, and {ref['ane_cores']} ANE cores. "
            "Other Macs or memory sizes may hit OOMs, kernel compile limits, or lower throughput "
            "(set FAST_SILICON_NO_HW_WARN=1 to silence runtime warnings).",
        ]
    )


def warn_if_untested_hardware(*, stream: Any = None, once: bool = True) -> bool:
    """Warn once on stderr when running outside the tested 48 GB M4 Pro (14C/20G/16N) baseline."""
    global _WARNED
    if os.environ.get("FAST_SILICON_NO_HW_WARN") == "1":
        matched, _, _ = evaluate_hardware_support()
        return matched
    if once and _WARNED:
        matched, _, _ = evaluate_hardware_support()
        return matched
    matched, mismatches, hw = evaluate_hardware_support()
    if not matched:
        _WARNED = True
        out = sys.stderr if stream is None else stream
        print(format_hardware_warning(hw, mismatches), file=out)
    return matched


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `fast-silicon-hw-check` (`python -m fast_shared.hw_check`)."""
    p = argparse.ArgumentParser(
        description="Check current MacBook/host specs against the tested 48 GB Apple M4 Pro baseline."
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON hardware report")
    p.add_argument("--strict", action="store_true", help="Exit with code 2 when host does not match tested baseline")
    p.add_argument("--quiet-ok", action="store_true", help="Suppress output when host matches tested baseline")
    args = p.parse_args(argv)

    matched, mismatches, hw = evaluate_hardware_support()
    if args.json:
        print(
            json.dumps(
                {
                    "matched_tested_baseline": matched,
                    "mismatches": mismatches,
                    "detected": hw,
                    "tested_reference": TESTED_REFERENCE_HARDWARE,
                },
                indent=2,
            )
        )
    elif not matched:
        print(format_hardware_warning(hw, mismatches), file=sys.stderr)
    elif not args.quiet_ok:
        print(f"[fast-silicon] Host verified: {format_hardware_summary(hw)} (matches tested baseline)")
    return 0 if (matched or not args.strict) else 2


if __name__ == "__main__":
    raise SystemExit(main())
