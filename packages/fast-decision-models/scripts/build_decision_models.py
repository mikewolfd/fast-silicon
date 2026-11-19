#!/usr/bin/env python3
"""Convenience script entrypoint wrapping `fast_decision_models.builder.main()`."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_decision_models.builder import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
