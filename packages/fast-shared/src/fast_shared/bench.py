"""Apple Silicon benchmark hygiene, contention checks, ABBA harness, and MLX dtype guards."""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import platform
import resource
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np

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


def check_system_quiescence() -> list[str]:
    """Return active GPU/CPU contention warning lines from `ps aux` (empty if clear)."""
    out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    warnings: list[str] = []
    for line in out.splitlines()[1:]:
        low = line.lower()
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        try:
            cpu = float(parts[2])
        except ValueError:
            cpu = 0.0
        cmd = parts[10]
        is_hog = (
            ("chrome" in low and ("gpu" in low or cpu >= 10.0) and cpu >= 0.5)
            or ("firefox" in low and cpu >= 15.0)
            or ("safari" in low and cpu >= 15.0)
            or (("mlx" in low or "pytest" in low or "llama" in low or "coreml" in low) and cpu >= 10.0)
            or cpu >= 40.0
        )
        if is_hog and "ab_bench_harness.py" not in cmd and "fast-bench-check" not in cmd:
            warnings.append(f"PID={parts[1]} CPU={cpu:.1f}% RSS_KB={parts[5]} :: {cmd[:120]}")
    return warnings


def clear_mlx_arena(threshold_mib: int = 256) -> None:
    """Release cached MLX wave allocations once `mx.get_cache_memory()` exceeds `threshold_mib`."""
    import mlx.core as mx

    gc.collect()
    if mx.get_cache_memory() > (threshold_mib << 20):
        mx.clear_cache()


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
) -> list[str]:
    """Flag unexpected FP32 tensors that indicate silent dtype promotions in MLX/CoreML graphs."""
    flags: list[str] = []
    for name, arr in named_arrays:
        dt_str = str(getattr(arr, "dtype", ""))
        if "float32" in dt_str and not any(tok in name.lower() for tok in allow_fp32_substrings):
            flags.append(f"{name}: {dt_str} shape={getattr(arr, 'shape', None)}")
    return flags


def interleaved_abba(
    fn_a: Callable[[], dict[str, Any]],
    fn_b: Callable[[], dict[str, Any]],
    *,
    warmup: bool = True,
    rounds: int = 2,
    cooldown_sec: float = 0.0,
) -> dict[str, Any]:
    """Run `fn_a` and `fn_b` in interleaved `A-B-B-A` order to cancel linear thermal drift."""
    import mlx.core as mx

    if warmup:
        _ = fn_a()
        _ = fn_b()
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
        t0 = time.perf_counter()
        res = dict(fn_a() if tag == "A" else fn_b())
        res.setdefault("wall_sec", round(time.perf_counter() - t0, 4))
        res.setdefault("peak_gb", round(mx.get_peak_memory() / (1024**3), 2))
        clear_mlx_arena(256)
        (a_runs if tag == "A" else b_runs).append(res)

    def _agg(runs: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
        return {k: round(float(np.median([r[k] for r in runs])), 4) for k in keys}

    return {
        "order": order,
        "A_runs": a_runs,
        "B_runs": b_runs,
        "A_median": _agg(a_runs),
        "B_median": _agg(b_runs),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `fast-bench-check` / `ab_bench_harness.py`."""
    p = argparse.ArgumentParser(description="Benchmark quiescence check & ABBA harness utilities.")
    p.add_argument("--check-ps", action="store_true", help="Inspect running processes for GPU/CPU contention")
    args = p.parse_args(argv)

    if args.check_ps:
        warns = check_system_quiescence()
        if not warns:
            print("QUIESCENT: no heavy GPU/CPU contenders detected.")
        else:
            print(f"CONTENTION WARNING ({len(warns)} process(es)):")
            for w in warns:
                print(f"  {w}")


if __name__ == "__main__":
    main()
