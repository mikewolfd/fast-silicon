#!/usr/bin/env python3
"""Standalone entrypoint delegating cleanly to `fast_shared.roofline.main_probe`."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from fast_shared.roofline import main_probe
except ImportError:
    _SRC = Path(__file__).resolve().parents[3] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from fast_shared.roofline import main_probe


if __name__ == "__main__":
    main_probe()
