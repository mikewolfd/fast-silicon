#!/usr/bin/env python3
"""Standalone entrypoint delegating cleanly to `fast_shared.roofline.main_audit`."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from fast_shared.roofline import main_audit
except ImportError:
    _SRC = Path(__file__).resolve().parents[3] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from fast_shared.roofline import main_audit


if __name__ == "__main__":
    main_audit()
