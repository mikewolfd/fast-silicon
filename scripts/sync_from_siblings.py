"""Sync clean package sources from sibling `../fast-*` checkouts into `packages/`."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGES: tuple[str, ...] = (
    "fast-shared",
    "fast-bge-small",
    "fast-embeddinggemma-2",
    "fast-gpt-oss",
    "fast-gemma4",
    "fast-decision-models",
)
MONOREPO_ONLY_PACKAGES: tuple[str, ...] = ("fast-pplx",)

IGNORED_DIR_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    ".claude",
}


def _is_clean_relpath(rel_path: str) -> bool:
    p = Path(rel_path)
    parts = p.parts
    if any(
        part in IGNORED_DIR_PARTS
        or part.startswith((".venv", "tmp_"))
        or part.endswith((".egg-info", ".mlpackage", ".mlmodelc"))
        for part in parts
    ):
        return False
    if p.name == ".DS_Store":
        return False
    if p.suffix in {".pyc", ".pyo", ".safetensors", ".parquet", ".npy"}:
        return False
    if p.suffix == ".npz" and "tests/fixtures/" not in rel_path:
        return False
    return True


def list_clean_files(src_pkg: Path) -> list[str]:
    """Return sorted relative file paths to include from a local `fast-*` checkout."""
    res = subprocess.run(
        ["git", "-C", str(src_pkg), "ls-files", "-co", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    rel_paths = [
        line.strip()
        for line in res.stdout.splitlines()
        if line.strip() and (src_pkg / line.strip()).is_file() and _is_clean_relpath(line.strip())
    ]
    return sorted(set(rel_paths))


def list_dest_tracked_candidates(dst_pkg: Path) -> list[str]:
    if not dst_pkg.exists():
        return []
    out: list[str] = []
    for path in dst_pkg.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(dst_pkg).as_posix()
        if _is_clean_relpath(rel):
            out.append(rel)
    return sorted(out)


def _transform_content(pkg: str, rel: str, raw: bytes) -> bytes:
    """Normalize package-local uv source references for the workspace monorepo."""
    if pkg != "fast-shared" and rel == "pyproject.toml":
        text = raw.decode("utf-8")
        text = text.replace(
            'fast-shared = { path = "../fast-shared", editable = true }',
            "fast-shared = { workspace = true }",
        )
        return text.encode("utf-8")
    return raw


def sync_package(src_pkg: Path, dst_pkg: Path, *, check: bool) -> tuple[int, int, int]:
    """Return `(copied, removed, unchanged)` counts for one package."""
    expected = set(list_clean_files(src_pkg))
    existing = set(list_dest_tracked_candidates(dst_pkg))
    to_remove = sorted(existing - expected)
    copied = 0
    unchanged = 0

    for rel in sorted(expected):
        src_file = src_pkg / rel
        dst_file = dst_pkg / rel
        payload = _transform_content(src_pkg.name, rel, src_file.read_bytes())
        if dst_file.exists() and dst_file.read_bytes() == payload:
            unchanged += 1
            continue
        copied += 1
        if not check:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_bytes(payload)
            shutil.copystat(src_file, dst_file)

    if not check:
        for rel in to_remove:
            target = dst_pkg / rel
            target.unlink(missing_ok=True)
            parent = target.parent
            while parent != dst_pkg and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

    return copied, len(to_remove), unchanged


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    all_packages = PACKAGES + MONOREPO_ONLY_PACKAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root.parent,
        help="Parent directory holding local fast-* sibling repos (default: %(default)s)",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        choices=all_packages,
        default=None,
        help="Subset of packages to sync or check (default: all known packages)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if packages/ differs from sibling fast-* checkouts without writing.",
    )
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    packages_dir = repo_root / "packages"
    if not args.check:
        packages_dir.mkdir(parents=True, exist_ok=True)

    selected = tuple(args.packages) if args.packages else all_packages
    drifted = False
    for pkg in selected:
        src_pkg = source_root / pkg
        dst_pkg = packages_dir / pkg
        if not src_pkg.is_dir():
            if pkg in MONOREPO_ONLY_PACKAGES:
                print(f"[SKIP] {pkg} (monorepo-native; no sibling checkout)")
            else:
                print(f"[SKIP] missing sibling repo: {src_pkg}", file=sys.stderr)
            continue
        copied, removed, unchanged = sync_package(src_pkg, dst_pkg, check=args.check)
        if copied or removed:
            drifted = True
        mode = "CHECK" if args.check else "SYNC"
        print(f"[{mode}] {pkg:24s} copied={copied} removed={removed} unchanged={unchanged}")

    return 1 if (args.check and drifted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
