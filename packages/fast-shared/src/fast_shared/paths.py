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
    """Resolve the package data/runtime root.

    Precedence:
      1. Explicit `home` argument.
      2. Package-specific environment variable (`env_var`).
      3. Editable checkout root (`anchor_file` walked `depth_up` levels) when it contains
         `artifacts/` (or any custom marker when `"artifacts"` is not in `marker_dirs`).
      4. `FAST_ARTIFACTS_ROOT` environment variable (`<root>/<cache_name>`).
      5. Sibling checkout (`../<cache_name>` or monorepo sibling `<monorepo>/../<cache_name>`)
         containing `artifacts/`.
      6. Standalone checkout containing a non-`artifacts` marker (e.g. `native/`) ONLY when
         NOT inside a monorepo `packages/` directory (so `fast-silicon/packages/<pkg>/native`
         without `artifacts/` does not mask sibling artifacts or `~/.cache/<cache_name>`).
      7. Fallback `~/.cache/<cache_name>`.
    """
    if home is not None:
        return Path(home).expanduser().resolve()
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val).expanduser().resolve()

    cand = Path(anchor_file).resolve()
    for _ in range(depth_up):
        cand = cand.parent

    # 3. Direct artifacts checkout (or custom marker-only callers)
    if "artifacts" in marker_dirs:
        if (cand / "artifacts").exists():
            return cand
    elif any((cand / d).exists() for d in marker_dirs):
        return cand

    # 4. Shared FAST_ARTIFACTS_ROOT override
    art_root_env = os.environ.get("FAST_ARTIFACTS_ROOT")
    if art_root_env:
        art_root = Path(art_root_env).expanduser().resolve()
        pkg_root = art_root / cache_name
        if (pkg_root / "artifacts").exists() or not (art_root / "artifacts").exists():
            return pkg_root
        return art_root

    # 5. Sibling checkout containing `artifacts/`
    is_monorepo_pkg = cand.parent.name == "packages"
    sibling_candidates: list[Path] = []
    if is_monorepo_pkg:
        # e.g. /Work/fast-silicon/packages/<pkg> -> /Work/<cache_name>
        sibling_candidates.append(cand.parent.parent.parent / cache_name)
    if cand.name != cache_name:
        sibling_candidates.append(cand.parent / cache_name)
    for sib in sibling_candidates:
        if (sib / "artifacts").exists():
            return sib.resolve()

    # 6. Standalone checkout with non-artifacts marker (e.g. `native/`), never monorepo packages
    if not is_monorepo_pkg and any((cand / d).exists() for d in marker_dirs if d != "artifacts"):
        return cand

    # 7. Default user cache
    return (Path("~/.cache") / cache_name).expanduser().resolve()
