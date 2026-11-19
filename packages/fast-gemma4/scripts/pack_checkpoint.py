"""Pack and validate a Gemma-4 checkpoint for fast-gemma4 (Zero-QAT 8-bit + BF16 MLP + vocab remap)."""

from __future__ import annotations

import sys

from fast_gemma4.packager import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
