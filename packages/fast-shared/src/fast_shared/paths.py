"""Shared repository root and cache directory resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_home_dir(
    anchor_file: str | Path,
    *,
    env_var: str,
    cache_name: str,
    home: str | Path | None = None,
    depth_up: int = 3,
    marker_dirs: tuple[str, ...] = ("artifacts", "native"),
) -> Path:
    """Resolve the package data/runtime root (explicit > env > editable checkout > ~/.cache)."""
    if home is not None:
        return Path(home).expanduser().resolve()
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val).expanduser().resolve()

    cand = Path(anchor_file).resolve()
    for _ in range(depth_up):
        cand = cand.parent
    if any((cand / d).exists() for d in marker_dirs):
        return cand
    return (Path("~/.cache") / cache_name).expanduser().resolve()
