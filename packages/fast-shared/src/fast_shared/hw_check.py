"""Host MacBook / Apple Silicon detection, 6-chip M4/M5 capability matrix, and preflight checks."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import plistlib
import re
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

# Validated 6-chip M4 (Apple9) & M5 (Apple10 / NAX) capability matrix.
# Sources: references/m4-m5-capability-matrix.md & references/m4-vs-m5-architecture.md.
SUPPORTED_APPLE_SILICON_PROFILES: dict[str, dict[str, Any]] = {
    "Apple M4": {
        "chip_family": "M4",
        "gpu_family": "Apple9",
        "has_nax": False,
        "cpu_cores": (8, 9, 10),
        "gpu_cores": (8, 10),
        "ane_cores": 16,
        "mem_gb": (16, 24, 32),
        "thunderbolt": "TB4",
        "nominal_bw_gbs": 120.0,
        "empirical_gpu_tf": 3.7,
        "empirical_dram_gbs": 95.0,
        "has_eff_cores": True,
    },
    "Apple M4 Pro": {
        "chip_family": "M4 Pro",
        "gpu_family": "Apple9",
        "has_nax": False,
        "cpu_cores": (12, 14),
        "gpu_cores": (16, 20),
        "ane_cores": 16,
        "mem_gb": (24, 48, 64),
        "thunderbolt": "TB5",
        "nominal_bw_gbs": 273.0,
        "empirical_gpu_tf": 7.4,
        "empirical_dram_gbs": 200.0,
        "has_eff_cores": True,
    },
    "Apple M4 Max": {
        "chip_family": "M4 Max",
        "gpu_family": "Apple9",
        "has_nax": False,
        "cpu_cores": (14, 16),
        "gpu_cores": (32, 40),
        "ane_cores": 16,
        "mem_gb": (36, 48, 64, 128),
        "thunderbolt": "TB5",
        "nominal_bw_gbs": 546.0,
        "empirical_gpu_tf": 14.8,
        "empirical_dram_gbs": 420.0,
        "has_eff_cores": True,
    },
    "Apple M5": {
        "chip_family": "M5",
        "gpu_family": "Apple10",
        "has_nax": True,
        "cpu_cores": (10,),
        "gpu_cores": (8, 10),
        "ane_cores": 16,
        "mem_gb": (16, 24, 32),
        "thunderbolt": "TB4",
        "nominal_bw_gbs": 153.0,
        "empirical_gpu_tf": 11.66,
        "empirical_dram_gbs": 123.4,
        "has_eff_cores": True,
    },
    "Apple M5 Pro": {
        "chip_family": "M5 Pro",
        "gpu_family": "Apple10",
        "has_nax": True,
        "cpu_cores": (15, 18),
        "gpu_cores": (16, 20),
        "ane_cores": 16,
        "mem_gb": (24, 48, 64),
        "thunderbolt": "TB5",
        "nominal_bw_gbs": 307.0,
        "empirical_gpu_tf": 23.3,
        "empirical_dram_gbs": 245.0,
        "has_eff_cores": False,
    },
    "Apple M5 Max": {
        "chip_family": "M5 Max",
        "gpu_family": "Apple10",
        "has_nax": True,
        "cpu_cores": (18,),
        "gpu_cores": (32, 40),
        "ane_cores": 16,
        "mem_gb": (36, 48, 64, 128),
        "thunderbolt": "TB5",
        "nominal_bw_gbs": 614.0,
        "empirical_gpu_tf": 46.6,
        "empirical_dram_gbs": 490.0,
        "has_eff_cores": False,
    },
}

_ARM_OPTIONAL_FEATURE_KEYS: tuple[str, ...] = (
    "FEAT_BF16",
    "FEAT_EBF16",
    "FEAT_SME",
    "FEAT_SME2",
    "FEAT_SME2p1",
    "FEAT_SME_F16F16",
    "FEAT_SME_B16B16",
    "FEAT_SVE_B16B16",
    "FEAT_CSSC",
    "FEAT_HBC",
    "FEAT_MTE4",
    "FEAT_SPECRES",
    "FEAT_SPECRES2",
    "FEAT_FP8",
    "FEAT_SME_F8F16",
    "FEAT_SME_F8F32",
    "FEAT_SME_LUTv2",
)

# Metal GPU family constants (MTLDevice.h).
_MTL_GPU_FAMILY_APPLE9 = 1009
_MTL_GPU_FAMILY_APPLE10 = 1010
_MTL_GPU_FAMILY_METAL4 = 5002

_WARNED = False


def parse_version_tuple(version_str: str | None) -> tuple[int, ...]:
    """Parse a dotted numeric version string (e.g. '26.6', '0.32.4', '27.2.1') into an int tuple."""
    if not version_str:
        return ()
    m = re.search(r"(\d+(?:\.\d+)*)", str(version_str))
    if not m:
        return ()
    return tuple(int(part) for part in m.group(1).split("."))


def _version_ge(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    if not lhs or not rhs:
        return False
    n = max(len(lhs), len(rhs))
    a = lhs + (0,) * (n - len(lhs))
    b = rhs + (0,) * (n - len(rhs))
    return a >= b


def _run_cmd(argv: list[str], *, timeout: float = 2.0) -> str:
    try:
        return subprocess.check_output(
            argv,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).strip()
    except Exception:
        return ""


def _sysctl(key: str) -> str:
    if platform.system() != "Darwin":
        return ""
    return _run_cmd(["sysctl", "-n", key])


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


def _probe_metal_device_caps() -> dict[str, bool | None]:
    """Query MTLDevice GPU family and placement-sparse support via ObjC runtime.

    Missing selectors return ``None`` (preserving Missing != False).
    """
    out: dict[str, bool | None] = {
        "supports_apple9": None,
        "supports_apple10": None,
        "supports_metal4": None,
        "supports_placement_sparse": None,
    }
    if platform.system() != "Darwin":
        return out
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p

        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return out
        try:
            msg_bool_i = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
            )(("objc_msgSend", objc))
            msg_bool_p = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
            )(("objc_msgSend", objc))
            msg_bool = ctypes.CFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
            )(("objc_msgSend", objc))

            sel_sf = objc.sel_registerName(b"supportsFamily:")
            sel_resp = objc.sel_registerName(b"respondsToSelector:")
            sel_ps = objc.sel_registerName(b"supportsPlacementSparse")

            if msg_bool_p(dev, sel_resp, sel_sf):
                out["supports_apple9"] = bool(msg_bool_i(dev, sel_sf, _MTL_GPU_FAMILY_APPLE9))
                out["supports_apple10"] = bool(msg_bool_i(dev, sel_sf, _MTL_GPU_FAMILY_APPLE10))
                out["supports_metal4"] = bool(msg_bool_i(dev, sel_sf, _MTL_GPU_FAMILY_METAL4))
            if msg_bool_p(dev, sel_resp, sel_ps):
                out["supports_placement_sparse"] = bool(msg_bool(dev, sel_ps))
        finally:
            msg_void = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
            msg_void(dev, objc.sel_registerName(b"release"))
    except Exception:
        pass
    return out


def _probe_arm_optional_features() -> dict[str, bool | None]:
    """Read Darwin `hw.optional.arm.*` sysctls, returning `None` when absent (`Missing != False`)."""
    feats: dict[str, bool | None] = {}
    if platform.system() != "Darwin":
        return {k: None for k in _ARM_OPTIONAL_FEATURE_KEYS}
    try:
        raw = subprocess.check_output(
            ["sysctl", "hw.optional.arm"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        parsed: dict[str, int] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            if k.startswith("hw.optional.arm."):
                try:
                    parsed[k[len("hw.optional.arm.") :]] = int(v.strip())
                except ValueError:
                    pass
        for key in _ARM_OPTIONAL_FEATURE_KEYS:
            val = parsed.get(key)
            feats[key] = None if val is None else bool(val)
    except Exception:
        for key in _ARM_OPTIONAL_FEATURE_KEYS:
            val = _sysctl_int(f"hw.optional.arm.{key}")
            feats[key] = None if val is None else bool(val)
    return feats


def _probe_mlx_runtime() -> dict[str, Any]:
    tf32_env = os.environ.get("MLX_ENABLE_TF32")
    tf32_enabled = True if tf32_env is None else (tf32_env.strip().lower() not in ("0", "false", "off", "no"))
    no_nax_env = os.environ.get("MLX_METAL_NO_NAX")
    no_nax = False if no_nax_env is None else (no_nax_env.strip().lower() not in ("0", "false", "off", "no", ""))

    mlx_version: str | None = None
    mlx_arch: str | None = None
    try:
        import mlx.core as mx

        mlx_version = str(getattr(mx, "__version__", "")) or None
        dev_info = mx.device_info()
        if isinstance(dev_info, dict):
            arch = dev_info.get("architecture")
            if arch:
                mlx_arch = str(arch)
    except Exception:
        pass

    return {
        "mlx_version": mlx_version,
        "mlx_architecture": mlx_arch,
        "mlx_enable_tf32": tf32_enabled,
        "mlx_enable_tf32_env": tf32_env,
        "mlx_metal_no_nax": no_nax,
        "mlx_metal_no_nax_env": no_nax_env,
    }


def _identify_chip_profile(brand: str) -> dict[str, Any] | None:
    if brand in SUPPORTED_APPLE_SILICON_PROFILES:
        return SUPPORTED_APPLE_SILICON_PROFILES[brand]
    low = brand.lower()
    for name in ("Apple M5 Max", "Apple M5 Pro", "Apple M5", "Apple M4 Max", "Apple M4 Pro", "Apple M4"):
        if name.lower() in low:
            return SUPPORTED_APPLE_SILICON_PROFILES[name]
    return None


def _format_cpu_topology(
    perf_cores: int | None,
    eff_cores: int | None,
    perflevel0_name: str | None,
    perflevel1_name: str | None,
) -> str:
    if perf_cores is None and eff_cores is None:
        return "unknown P/E"
    l0 = (perflevel0_name or "").strip()
    l1 = (perflevel1_name or "").strip()
    if l0 or l1:
        parts = []
        if perf_cores is not None:
            parts.append(f"{perf_cores} {l0 or 'P'}")
        if eff_cores is not None:
            parts.append(f"{eff_cores} {l1 or 'E'}")
        return " + ".join(parts)
    return f"{perf_cores}P + {eff_cores}E"


def probe_host_hardware() -> dict[str, Any]:
    """Probe local MacBook / host OS, SDK, MLX, chip, unified RAM, CPU topology, GPU/Metal family, and ANE."""
    sys_name = platform.system()
    machine = platform.machine()
    brand = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Unknown CPU"
    hw_model = _sysctl("hw.model") or "unknown"
    mem_bytes = _probe_mem_bytes()
    mem_gb = int(round(mem_bytes / (1024**3))) if mem_bytes > 0 else 0
    cpu_cores = _sysctl_int("hw.ncpu") or (os.cpu_count() or 0)
    perf_cores = _sysctl_int("hw.perflevel0.physicalcpu")
    eff_cores = _sysctl_int("hw.perflevel1.physicalcpu")
    perflevel0_name = _sysctl("hw.perflevel0.name") or None
    perflevel1_name = _sysctl("hw.perflevel1.name") or None
    perflevel0_l2_bytes = _sysctl_int("hw.perflevel0.l2cachesize")
    perflevel1_l2_bytes = _sysctl_int("hw.perflevel1.l2cachesize")

    agx = _ioreg_first_plist("AGXAccelerator")
    gpu_raw = agx.get("gpu-core-count")
    gpu_cores = int(gpu_raw) if isinstance(gpu_raw, int) else None

    ane = _ioreg_first_plist("H11ANEIn")
    ane_props = ane.get("DeviceProperties") if isinstance(ane.get("DeviceProperties"), dict) else {}
    ane_raw = ane_props.get("ANEDevicePropertyNumANECores")
    ane_cores = int(ane_raw) if isinstance(ane_raw, int) else None
    ane_arch = ane_props.get("ANEDevicePropertyTypeANEArchitectureTypeStr")

    macos_version = _run_cmd(["sw_vers", "-productVersion"]) if sys_name == "Darwin" else ""
    if not macos_version and sys_name == "Darwin":
        macos_version = platform.mac_ver()[0] or ""
    macos_build = _run_cmd(["sw_vers", "-buildVersion"]) if sys_name == "Darwin" else ""
    xcode_sdk_version = _run_cmd(["xcrun", "--show-sdk-version"]) if sys_name == "Darwin" else ""

    metal_caps = _probe_metal_device_caps()
    mlx_info = _probe_mlx_runtime()
    arm_features = _probe_arm_optional_features()

    profile = _identify_chip_profile(brand)
    chip_family = profile["chip_family"] if profile else None
    thunderbolt = profile["thunderbolt"] if profile else None

    gpu_family: str | None = None
    if metal_caps["supports_apple10"] is True:
        gpu_family = "Apple10"
    elif metal_caps["supports_apple9"] is True:
        gpu_family = "Apple9"
    elif profile is not None:
        gpu_family = str(profile["gpu_family"])

    cpu_topology = _format_cpu_topology(perf_cores, eff_cores, perflevel0_name, perflevel1_name)

    return {
        "system": sys_name,
        "machine": machine,
        "hw_model": hw_model,
        "brand": brand,
        "chip_family": chip_family,
        "mem_bytes": mem_bytes,
        "mem_gb": mem_gb,
        "cpu_cores": cpu_cores,
        "perf_cores": perf_cores,
        "eff_cores": eff_cores,
        "perflevel0_name": perflevel0_name,
        "perflevel1_name": perflevel1_name,
        "perflevel0_l2_bytes": perflevel0_l2_bytes,
        "perflevel1_l2_bytes": perflevel1_l2_bytes,
        "cpu_topology": cpu_topology,
        "gpu_cores": gpu_cores,
        "gpu_family": gpu_family,
        "supports_apple9": metal_caps["supports_apple9"],
        "supports_apple10": metal_caps["supports_apple10"],
        "supports_metal4": metal_caps["supports_metal4"],
        "supports_placement_sparse": metal_caps["supports_placement_sparse"],
        "ane_cores": ane_cores,
        "ane_arch": str(ane_arch) if ane_arch else None,
        "thunderbolt": thunderbolt,
        "macos_version": macos_version or None,
        "macos_version_tuple": list(parse_version_tuple(macos_version)),
        "macos_build": macos_build or None,
        "xcode_sdk_version": xcode_sdk_version or None,
        "mlx_version": mlx_info["mlx_version"],
        "mlx_architecture": mlx_info["mlx_architecture"],
        "mlx_enable_tf32": mlx_info["mlx_enable_tf32"],
        "mlx_enable_tf32_env": mlx_info["mlx_enable_tf32_env"],
        "mlx_metal_no_nax": mlx_info["mlx_metal_no_nax"],
        "mlx_metal_no_nax_env": mlx_info["mlx_metal_no_nax_env"],
        "arm_features": arm_features,
    }


def supports_apple10_nax(hw: dict[str, Any] | None = None) -> bool:
    """Return True if the host supports Apple10 (M5+) GPU Neural Accelerators (NAX) in MLX/Metal."""
    info = hw if hw is not None else probe_host_hardware()
    if info.get("system") != "Darwin" or info.get("machine") != "arm64":
        return False
    if info.get("mlx_metal_no_nax"):
        return False
    a10 = info.get("supports_apple10")
    if a10 is False:
        return False
    if a10 is None:
        gpu_fam = str(info.get("gpu_family") or "")
        brand = str(info.get("brand") or "")
        arch = str(info.get("mlx_architecture") or "")
        if gpu_fam != "Apple10" and "M5" not in brand and not arch.startswith("applegpu_g17"):
            return False
    os_ver = parse_version_tuple(info.get("macos_version"))
    if os_ver and not _version_ge(os_ver, (26, 2)):
        return False
    return True


def supports_metal4_fp16(hw: dict[str, Any] | None = None) -> bool:
    """Return True if host supports Metal 4 MPP FP16 tensor operations (macOS >= 26.0, Apple9/Apple10)."""
    info = hw if hw is not None else probe_host_hardware()
    if info.get("system") != "Darwin" or info.get("machine") != "arm64":
        return False
    if info.get("supports_metal4") is False:
        return False
    os_ver = parse_version_tuple(info.get("macos_version"))
    if os_ver and not _version_ge(os_ver, (26, 0)):
        return False
    if info.get("supports_metal4") is True:
        return True
    return info.get("gpu_family") in ("Apple9", "Apple10") or _identify_chip_profile(str(info.get("brand") or "")) is not None


def supports_metal4_bf16(hw: dict[str, Any] | None = None) -> bool:
    """Return True if host supports Metal 4 MPP BF16 tensor operands (macOS >= 26.1, Apple9/Apple10)."""
    info = hw if hw is not None else probe_host_hardware()
    if not supports_metal4_fp16(info):
        return False
    os_ver = parse_version_tuple(info.get("macos_version"))
    if os_ver and not _version_ge(os_ver, (26, 1)):
        return False
    return True


def default_roofline_ceilings(hw: dict[str, Any] | None = None) -> tuple[float, float]:
    """Return empirical `(gpu_tf, dram_gbs)` roofline defaults for the detected M4/M5 chip."""
    info = hw if hw is not None else probe_host_hardware()
    profile = _identify_chip_profile(str(info.get("brand") or ""))
    if profile is not None:
        gpu_tf = float(profile["empirical_gpu_tf"])
        dram_gbs = float(profile["empirical_dram_gbs"])
        # Scale lower-bin Max SKUs (32 GPU cores vs 40 GPU cores, 3/4 memory bus)
        if profile["chip_family"] in ("M4 Max", "M5 Max") and info.get("gpu_cores") == 32:
            gpu_tf = round(gpu_tf * 0.8, 2)
            dram_gbs = round(dram_gbs * 0.75, 1)
        return gpu_tf, dram_gbs
    return 7.4, 200.0


def evaluate_hardware_support(
    hw: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate detected hardware against the 6-chip M4/M5 capability matrix (and primary M4 Pro baseline)."""
    info = hw if hw is not None else probe_host_hardware()
    ref = TESTED_REFERENCE_HARDWARE
    mismatches: list[str] = []

    is_darwin_arm = info.get("system") == "Darwin" and info.get("machine") == "arm64"
    profile = _identify_chip_profile(str(info.get("brand") or ""))

    if is_darwin_arm and profile is not None:
        if info.get("cpu_cores") is not None and int(info["cpu_cores"]) not in profile["cpu_cores"]:
            mismatches.append(f"cpu_cores={info.get('cpu_cores')} (expected {profile['cpu_cores']} for {profile['chip_family']})")
        if info.get("gpu_cores") is not None and int(info["gpu_cores"]) not in profile["gpu_cores"]:
            mismatches.append(f"gpu_cores={info.get('gpu_cores')} (expected {profile['gpu_cores']} for {profile['chip_family']})")
        if info.get("ane_cores") is not None and int(info["ane_cores"]) != profile["ane_cores"]:
            mismatches.append(f"ane_cores={info.get('ane_cores')} (expected {profile['ane_cores']})")
        if info.get("mem_gb") is not None and int(info["mem_gb"]) > 0 and int(info["mem_gb"]) not in profile["mem_gb"]:
            mismatches.append(f"memory={info.get('mem_gb')} GB (expected {profile['mem_gb']} GB for {profile['chip_family']})")
        return (len(mismatches) == 0), mismatches, info

    if not is_darwin_arm:
        mismatches.append(f"platform={info.get('system')}/{info.get('machine')} (tested Darwin/arm64)")
    if info.get("brand") != ref["brand"]:
        mismatches.append(f"chip={info.get('brand')!r} (supported: {', '.join(SUPPORTED_APPLE_SILICON_PROFILES)})")
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


def preflight_silicon_env(
    require_mlx_min: str = "0.32.4",
    check_tf32: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Run hardware, macOS/SDK, MLX floor, NAX, and TF32 environment preflight checks."""
    hw = probe_host_hardware()
    matched, hw_mismatches, _ = evaluate_hardware_support(hw)
    warnings: list[str] = []
    errors: list[str] = []

    if not matched:
        warnings.append(f"Unsupported/unvalidated hardware profile: {', '.join(hw_mismatches)}")

    mlx_ver = hw.get("mlx_version")
    min_tuple = parse_version_tuple(require_mlx_min)
    cur_tuple = parse_version_tuple(mlx_ver)
    mlx_meets_min = bool(mlx_ver and _version_ge(cur_tuple, min_tuple))
    if mlx_ver is None:
        msg = f"mlx is not installed (required >= {require_mlx_min})"
        (errors if strict else warnings).append(msg)
    elif not mlx_meets_min:
        msg = f"mlx=={mlx_ver} is below required minimum {require_mlx_min}"
        (errors if strict else warnings).append(msg)

    a10_nax = supports_apple10_nax(hw)
    m4_fp16 = supports_metal4_fp16(hw)
    m4_bf16 = supports_metal4_bf16(hw)

    if hw.get("mlx_metal_no_nax"):
        warnings.append("MLX_METAL_NO_NAX is set; Apple10/M5 NAX acceleration is disabled.")

    os_ver = parse_version_tuple(hw.get("macos_version"))
    if hw.get("system") == "Darwin" and os_ver and not _version_ge(os_ver, (26, 0)):
        warnings.append(f"macOS {hw.get('macos_version')} < 26.0; Metal 4 MPP kernels are unavailable.")
    elif hw.get("system") == "Darwin" and os_ver and not _version_ge(os_ver, (26, 1)):
        warnings.append(f"macOS {hw.get('macos_version')} < 26.1; Metal 4 MPP BF16 tensor operands are unavailable.")

    tf32_enabled = bool(hw.get("mlx_enable_tf32", True))
    if check_tf32:
        if hw.get("mlx_enable_tf32_env") is None and tf32_enabled:
            warnings.append(
                "MLX_ENABLE_TF32 is unset (defaults to 1, cached on first MLX call); "
                "set MLX_ENABLE_TF32=0 before process launch for strict FP32 parity on Apple10/NAX."
            )
        elif tf32_enabled:
            warnings.append(
                "MLX_ENABLE_TF32=1 enables TF32 on eligible FP32 NAX GEMM/SDPA paths; "
                "record this setting in FP32 benchmarks or set MLX_ENABLE_TF32=0 for strict FP32."
            )

    ok = len(errors) == 0 and (matched or not strict)
    report: dict[str, Any] = {
        "ok": ok,
        "matched_supported_silicon": matched,
        "warnings": warnings,
        "errors": errors,
        "hardware": hw,
        "mlx_version": mlx_ver,
        "require_mlx_min": require_mlx_min,
        "mlx_meets_min": mlx_meets_min,
        "supports_apple10_nax": a10_nax,
        "supports_metal4_fp16": m4_fp16,
        "supports_metal4_bf16": m4_bf16,
        "tf32_enabled": tf32_enabled,
    }
    if strict and not ok:
        details = "; ".join(errors or warnings)
        raise RuntimeError(f"Apple Silicon preflight failed: {details}")
    return report


def format_hardware_summary(hw: dict[str, Any]) -> str:
    cpu_topo = hw.get("cpu_topology") or _format_cpu_topology(
        hw.get("perf_cores"),
        hw.get("eff_cores"),
        hw.get("perflevel0_name"),
        hw.get("perflevel1_name"),
    )
    gpu_s = f"{hw['gpu_cores']} GPU cores" if hw.get("gpu_cores") is not None else "GPU cores unknown"
    if hw.get("gpu_family"):
        gpu_s = f"{gpu_s} ({hw['gpu_family']})"
    ane_s = f"{hw['ane_cores']} ANE cores" if hw.get("ane_cores") is not None else "ANE cores unknown"
    extras: list[str] = []
    if hw.get("macos_version"):
        os_s = f"macOS {hw['macos_version']}"
        if hw.get("macos_build"):
            os_s += f" ({hw['macos_build']})"
        extras.append(os_s)
    if hw.get("mlx_version"):
        extras.append(f"MLX {hw['mlx_version']}")
    if hw.get("thunderbolt"):
        extras.append(str(hw["thunderbolt"]))
    extra_str = f" | {' | '.join(extras)}" if extras else ""
    return (
        f"{hw.get('hw_model')} ({hw.get('system')}/{hw.get('machine')}) | "
        f"{hw.get('brand')} | {hw.get('mem_gb')} GB RAM | "
        f"{hw.get('cpu_cores')} CPU cores ({cpu_topo}) | {gpu_s} | {ane_s}{extra_str}"
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
            f"  Supported: {', '.join(SUPPORTED_APPLE_SILICON_PROFILES)}",
            f"  Differs  : {', '.join(mismatches)}",
            "  Note     : Primary reference baseline is a 48 GB Apple M4 Pro MacBook Pro "
            f"({ref['hw_model']}) with {ref['cpu_cores']} CPU cores ({ref['perf_cores']}P + {ref['eff_cores']}E), "
            f"{ref['gpu_cores']} GPU cores, and {ref['ane_cores']} ANE cores, with validated M4/M5 family profiles. "
            "Other Macs or memory sizes may hit OOMs, kernel compile limits, or lower throughput "
            "(set FAST_SILICON_NO_HW_WARN=1 to silence runtime warnings).",
        ]
    )


def warn_if_untested_hardware(*, stream: Any = None, once: bool = True) -> bool:
    """Warn once on stderr when running outside the validated M4/M5 Apple Silicon family matrix."""
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
        description="Check current MacBook/host specs against the validated M4/M5 Apple Silicon capability matrix."
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON hardware report")
    p.add_argument("--strict", action="store_true", help="Exit with code 2 when host is outside validated M4/M5 profiles")
    p.add_argument("--quiet-ok", action="store_true", help="Suppress output when host is validated")
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
                    "supported_profiles": list(SUPPORTED_APPLE_SILICON_PROFILES.keys()),
                    "supports_apple10_nax": supports_apple10_nax(hw),
                    "supports_metal4_fp16": supports_metal4_fp16(hw),
                    "supports_metal4_bf16": supports_metal4_bf16(hw),
                },
                indent=2,
            )
        )
    elif not matched:
        print(format_hardware_warning(hw, mismatches), file=sys.stderr)
    elif not args.quiet_ok:
        print(f"[fast-silicon] Host verified: {format_hardware_summary(hw)} (matches tested baseline)")
    return 0 if (matched or not args.strict) else 2


__all__ = [
    "SUPPORTED_APPLE_SILICON_PROFILES",
    "TESTED_REFERENCE_HARDWARE",
    "default_roofline_ceilings",
    "evaluate_hardware_support",
    "format_hardware_summary",
    "format_hardware_warning",
    "main",
    "parse_version_tuple",
    "preflight_silicon_env",
    "probe_host_hardware",
    "supports_apple10_nax",
    "supports_metal4_bf16",
    "supports_metal4_fp16",
    "warn_if_untested_hardware",
]


if __name__ == "__main__":
    raise SystemExit(main())
