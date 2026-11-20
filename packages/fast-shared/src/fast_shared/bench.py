"""Apple Silicon benchmark hygiene, contention checks, wired limits, ABBA harness, and MLX dtype guards."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gc
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .hw_check import probe_host_hardware, supports_apple10_nax, supports_metal4_bf16

_RUSAGE_INFO_V4 = 4
_RUSAGE_INFO_V4_SIZE = 296
_PHYS_FOOTPRINT_OFFSET = 72
_LIFETIME_MAX_FOOTPRINT_OFFSET = 240
_LIBPROC_CACHE: Any = None


def _load_libproc() -> Any:
    global _LIBPROC_CACHE
    if _LIBPROC_CACHE is None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        _LIBPROC_CACHE = lib
    return _LIBPROC_CACHE


def get_unified_memory_footprint(pid: int | None = None) -> dict[str, float | int]:
    """Read current & lifetime-peak OS physical footprint (Metal + ANE + CPU) on macOS.

    Uses ``proc_pid_rusage(..., RUSAGE_INFO_V4)`` from ``/usr/lib/libproc.dylib``
    so Metal IOSurface/buffer allocations and mmap pages are reflected alongside CPU RSS.
    Falls back cleanly to ``ru_maxrss`` on non-Darwin platforms.
    """
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    is_darwin = platform.system() == "Darwin"
    rss_bytes = int(ru_self.ru_maxrss) if is_darwin else int(ru_self.ru_maxrss) * 1024
    if not is_darwin:
        return {
            "phys_footprint_bytes": rss_bytes,
            "lifetime_max_phys_footprint_bytes": rss_bytes,
            "rss_bytes": rss_bytes,
            "phys_footprint_gb": rss_bytes / 1e9,
            "peak_footprint_gb": rss_bytes / 1e9,
        }

    buf = ctypes.create_string_buffer(512)
    target_pid = os.getpid() if pid is None else int(pid)
    rc = _load_libproc().proc_pid_rusage(target_pid, _RUSAGE_INFO_V4, ctypes.byref(buf))
    if rc != 0:
        raise OSError(f"proc_pid_rusage failed for pid={target_pid} (rc={rc})")

    phys_now = int.from_bytes(buf.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little")
    phys_peak = int.from_bytes(
        buf.raw[_LIFETIME_MAX_FOOTPRINT_OFFSET : _LIFETIME_MAX_FOOTPRINT_OFFSET + 8], "little"
    )
    return {
        "phys_footprint_bytes": phys_now,
        "lifetime_max_phys_footprint_bytes": phys_peak,
        "rss_bytes": rss_bytes,
        "phys_footprint_gb": phys_now / 1e9,
        "peak_footprint_gb": phys_peak / 1e9,
    }


def peak_footprint_gb(pid: int | None = None) -> float:
    """Return process lifetime-max unified-memory physical footprint in decimal GB."""
    return float(get_unified_memory_footprint(pid=pid)["peak_footprint_gb"])


@contextmanager
def wired_limit(limit_bytes: int | None = None) -> Iterator[int | None]:
    """Context manager applying `mx.set_wired_limit(...)` and restoring the previous limit on exit.

    When `limit_bytes` is `None`, uses `mx.device_info()["max_recommended_working_set_size"]`.
    Yields the applied limit in bytes, or `None` if MLX / `set_wired_limit` is unavailable.
    """
    try:
        import mlx.core as mx
    except ImportError:
        yield None
        return

    set_wired = getattr(mx, "set_wired_limit", None)
    if not callable(set_wired):
        yield None
        return

    if limit_bytes is None:
        try:
            dev = mx.device_info()
            target = int(dev.get("max_recommended_working_set_size", 0)) if isinstance(dev, dict) else 0
        except Exception:
            target = 0
    else:
        target = int(limit_bytes)

    if target <= 0:
        yield None
        return

    prev = set_wired(target)
    try:
        yield target
    finally:
        if prev is not None:
            try:
                set_wired(int(prev))
            except Exception:
                pass


def collect_env_telemetry() -> dict[str, Any]:
    """Collect host silicon, OS/build, Xcode SDK, MLX version, NAX, and TF32 benchmark telemetry."""
    hw = probe_host_hardware()
    return {
        "brand": hw.get("brand"),
        "chip_family": hw.get("chip_family"),
        "hw_model": hw.get("hw_model"),
        "macos_version": hw.get("macos_version"),
        "macos_build": hw.get("macos_build"),
        "xcode_sdk_version": hw.get("xcode_sdk_version"),
        "python_version": platform.python_version(),
        "mlx_version": hw.get("mlx_version"),
        "mlx_architecture": hw.get("mlx_architecture"),
        "gpu_family": hw.get("gpu_family"),
        "gpu_cores": hw.get("gpu_cores"),
        "ane_cores": hw.get("ane_cores"),
        "cpu_cores": hw.get("cpu_cores"),
        "cpu_topology": hw.get("cpu_topology"),
        "mem_gb": hw.get("mem_gb"),
        "thunderbolt": hw.get("thunderbolt"),
        "supports_apple10_nax": supports_apple10_nax(hw),
        "supports_metal4_bf16": supports_metal4_bf16(hw),
        "supports_placement_sparse": hw.get("supports_placement_sparse"),
        "mlx_enable_tf32": hw.get("mlx_enable_tf32"),
        "mlx_enable_tf32_env": hw.get("mlx_enable_tf32_env"),
        "mlx_metal_no_nax": hw.get("mlx_metal_no_nax"),
    }


def _classify_quiescence_line(
    line: str,
    *,
    ignore_pids: frozenset[int] | set[int] = frozenset(),
) -> str | None:
    """Return a formatted warning if a `ps aux` line indicates active GPU/CPU contention."""
    parts = line.split(None, 10)
    if len(parts) < 11:
        return None
    if ignore_pids:
        try:
            if int(parts[1]) in ignore_pids:
                return None
        except ValueError:
            pass
    try:
        cpu = float(parts[2])
    except ValueError:
        cpu = 0.0
    cmd = parts[10]
    if "ab_bench_harness.py" in cmd or "fast-bench-check" in cmd:
        return None
    low = line.lower()
    cmd_base = os.path.basename(cmd.split()[0]).lower() if cmd.split() else ""

    is_gpu_browser = (
        ("chrome" in low or "microsoft edge" in low or "msedge" in low or "/arc.app/" in low or cmd_base == "arc")
        and ("gpu" in low or "helper" in low or cpu >= 10.0)
        and cpu >= 0.5
    )
    is_other_browser = ("firefox" in low or "safari" in low) and cpu >= 15.0
    is_ml_runner = ("mlx" in low or "pytest" in low or "llama" in low or "coreml" in low) and cpu >= 10.0
    is_spotlight = cmd_base in ("mds", "mds_stores", "mdworker", "mdworker_shared") and cpu >= 5.0
    is_window_server = cmd_base == "windowserver" and cpu >= 15.0
    is_backup = cmd_base == "backupd" and cpu >= 5.0
    is_build_tool = cmd_base in ("xcodebuild", "clang", "clang++", "swift-frontend", "metal") and cpu >= 10.0

    if is_gpu_browser or is_other_browser or is_ml_runner or is_spotlight or is_window_server or is_backup or is_build_tool or cpu >= 40.0:
        return f"PID={parts[1]} CPU={cpu:.1f}% RSS_KB={parts[5]} :: {cmd[:120]}"
    return None


def check_system_quiescence() -> list[str]:
    """Return active GPU/CPU contention warning lines from `ps aux` (empty if clear)."""
    out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    self_pids = frozenset({os.getpid(), os.getppid()})
    warnings: list[str] = []
    for line in out.splitlines()[1:]:
        warn = _classify_quiescence_line(line, ignore_pids=self_pids)
        if warn is not None:
            warnings.append(warn)
    return warnings


def clear_mlx_arena(threshold_mib: int = 256) -> None:
    """Release cached MLX wave allocations once `mx.get_cache_memory()` exceeds `threshold_mib`."""
    try:
        import mlx.core as mx
    except ImportError:
        if threshold_mib <= 0:
            gc.collect()
        return

    get_cache = getattr(mx, "get_cache_memory", None)
    clear_cache = getattr(mx, "clear_cache", None)
    if threshold_mib <= 0:
        gc.collect()
        if callable(clear_cache):
            clear_cache()
        return
    if callable(get_cache) and get_cache() > (threshold_mib << 20):
        gc.collect()
        if callable(clear_cache):
            clear_cache()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute float64 cosine similarity between two tensors."""
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    denom = max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-12)
    return float(np.dot(af, bf) / denom)


def check_tensor_dtypes(
    named_arrays: Sequence[tuple[str, Any]],
    *,
    allow_fp32_substrings: Sequence[str] = ("sink", "rope", "inv_freq", "norm"),
    expected_dtype: str | None = None,
    check_masks: bool = True,
) -> list[str]:
    """Flag unexpected FP32 tensors, int/bool attention masks, and fp16/bf16 mismatches."""
    flags: list[str] = []
    low_float_seen: dict[str, list[str]] = {"float16": [], "bfloat16": []}

    for name, arr in named_arrays:
        dt_str = str(getattr(arr, "dtype", ""))
        dt_low = dt_str.lower()
        name_low = name.lower()
        shape = getattr(arr, "shape", None)

        if "float32" in dt_low and not any(tok in name_low for tok in allow_fp32_substrings):
            flags.append(f"{name}: {dt_str} shape={shape}")
            continue

        if check_masks and "mask" in name_low:
            if "bool" in dt_low or any(tok in dt_low for tok in ("int8", "int16", "int32", "int64", "uint")):
                flags.append(f"{name}: mask uses {dt_str} (prefer native fp16/bf16 additive mask) shape={shape}")
                continue

        if "bfloat16" in dt_low:
            low_float_seen["bfloat16"].append(name)
            if expected_dtype == "float16":
                flags.append(f"{name}: {dt_str} mismatches expected float16 shape={shape}")
        elif "float16" in dt_low:
            low_float_seen["float16"].append(name)
            if expected_dtype == "bfloat16":
                flags.append(f"{name}: {dt_str} mismatches expected bfloat16 shape={shape}")

    if expected_dtype is None and low_float_seen["float16"] and low_float_seen["bfloat16"]:
        flags.append(
            f"mixed fp16/bf16 tensors: float16={low_float_seen['float16']} vs bfloat16={low_float_seen['bfloat16']}"
        )
    return flags


def interleaved_abba(
    fn_a: Callable[[], dict[str, Any]],
    fn_b: Callable[[], dict[str, Any]],
    *,
    warmup: bool = True,
    rounds: int = 2,
    cooldown_sec: float = 0.0,
    use_wired_limit: bool = False,
    wired_limit_bytes: int | None = None,
    include_telemetry: bool = True,
) -> dict[str, Any]:
    """Run `fn_a` and `fn_b` in interleaved `A-B-B-A` order to cancel linear thermal drift."""
    import mlx.core as mx

    ctx = wired_limit(wired_limit_bytes) if use_wired_limit else _nullcontext()
    with ctx:
        if warmup:
            _ = fn_a()
            mx.synchronize()
            _ = fn_b()
            mx.synchronize()
            clear_mlx_arena(0)

        a_runs: list[dict[str, Any]] = []
        b_runs: list[dict[str, Any]] = []
        order = ["A", "B", "B", "A"] * max(1, rounds // 2)
        if rounds % 2 == 1:
            order.extend(["A", "B"])

        for idx, tag in enumerate(order):
            if idx > 0 and cooldown_sec > 0.0:
                time.sleep(cooldown_sec)
            mx.reset_peak_memory()
            mx.synchronize()
            t0 = time.perf_counter()
            res = dict(fn_a() if tag == "A" else fn_b())
            mx.synchronize()
            wall = round(time.perf_counter() - t0, 4)
            res.setdefault("wall_sec", wall)
            res.setdefault("peak_gb", round(mx.get_peak_memory() / (1024**3), 2))
            clear_mlx_arena(256)
            (a_runs if tag == "A" else b_runs).append(res)

    def _agg(runs: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return {k: round(float(np.median([r[k] for r in runs])), 4) for k in keys}

    def _drift(runs: list[dict[str, Any]]) -> dict[str, float | None]:
        if len(runs) < 2:
            return {"ratio": None, "delta_pct": None}
        first = float(runs[0].get("wall_sec", 0.0))
        last = float(runs[-1].get("wall_sec", 0.0))
        if first <= 0.0:
            return {"ratio": None, "delta_pct": None}
        ratio = last / first
        return {"ratio": round(ratio, 4), "delta_pct": round((ratio - 1.0) * 100.0, 2)}

    a_drift = _drift(a_runs)
    b_drift = _drift(b_runs)

    out: dict[str, Any] = {
        "order": order,
        "A_runs": a_runs,
        "B_runs": b_runs,
        "A_median": _agg(a_runs),
        "B_median": _agg(b_runs),
        "drift": {
            "A_last_over_first": a_drift["ratio"],
            "A_drift_pct": a_drift["delta_pct"],
            "B_last_over_first": b_drift["ratio"],
            "B_drift_pct": b_drift["delta_pct"],
        },
        "peak_footprint_gb": round(peak_footprint_gb(), 3),
    }
    if include_telemetry:
        out["env_telemetry"] = collect_env_telemetry()
    return out


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield None


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `fast-bench-check` / `ab_bench_harness.py`."""
    p = argparse.ArgumentParser(description="Benchmark quiescence check & ABBA harness utilities.")
    p.add_argument("--check-ps", action="store_true", help="Inspect running processes for GPU/CPU contention")
    p.add_argument("--telemetry", action="store_true", help="Print host/OS/SDK/MLX benchmark telemetry JSON")
    args = p.parse_args(argv)

    if args.telemetry:
        import json

        print(json.dumps(collect_env_telemetry(), indent=2))

    if args.check_ps or not args.telemetry:
        warns = check_system_quiescence()
        if not warns:
            print("QUIESCENT: no heavy GPU/CPU contenders detected.")
        else:
            print(f"CONTENTION WARNING ({len(warns)} process(es)):")
            for w in warns:
                print(f"  {w}")


__all__ = [
    "check_system_quiescence",
    "check_tensor_dtypes",
    "clear_mlx_arena",
    "collect_env_telemetry",
    "cos_sim",
    "get_unified_memory_footprint",
    "interleaved_abba",
    "main",
    "peak_footprint_gb",
    "wired_limit",
]


if __name__ == "__main__":
    main()
