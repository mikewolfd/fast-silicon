#!/usr/bin/env python3
"""Standalone entrypoint re-exporting `fast_shared.bench` and running `fast_shared.bench.main`."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from fast_shared.bench import (
        check_system_quiescence,
        check_tensor_dtypes,
        clear_mlx_arena,
        collect_env_telemetry,
        cos_sim,
        get_unified_memory_footprint,
        interleaved_abba,
        main,
        peak_footprint_gb,
        wired_limit,
    )
except ImportError:
    _SRC = Path(__file__).resolve().parents[3] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from fast_shared.bench import (
        check_system_quiescence,
        check_tensor_dtypes,
        clear_mlx_arena,
        collect_env_telemetry,
        cos_sim,
        get_unified_memory_footprint,
        interleaved_abba,
        main,
        peak_footprint_gb,
        wired_limit,
    )

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
