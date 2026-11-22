"""Sync clean package sources from sibling `../fast-*` checkouts into `packages/`."""

from __future__ import annotations

import argparse
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
from typing import Any

PACKAGES: tuple[str, ...] = (
    "fast-shared",
    "fast-bge-small",
    "fast-embeddinggemma-2",
    "fast-gpt-oss",
    "fast-gemma4",
    "fast-decision-models",
)
MONOREPO_ONLY_PACKAGES: tuple[str, ...] = ("fast-pplx", "fast-qwen3")

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


def inspect_sibling_git_hygiene(src_pkg: Path) -> dict[str, Any]:
    """Inspect a sibling git repo for zero commits, uncommitted edits, or untracked non-ignored files."""
    head_res = subprocess.run(
        ["git", "-C", str(src_pkg), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    has_commits = head_res.returncode == 0
    head_sha = head_res.stdout.strip() if has_commits else None

    st_res = subprocess.run(
        ["git", "-C", str(src_pkg), "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        capture_output=True,
        text=True,
    )
    modified: list[str] = []
    untracked: list[str] = []
    if st_res.returncode == 0:
        for line in st_res.stdout.splitlines():
            if not line.strip():
                continue
            code = line[:2]
            rel = line[3:].strip()
            if code == "??":
                untracked.append(rel)
            else:
                modified.append(rel)

    reasons: list[str] = []
    if not has_commits:
        reasons.append("0 commits (unborn HEAD)")
    if modified:
        reasons.append(f"{len(modified)} uncommitted tracked change(s)")
    if untracked:
        reasons.append(f"{len(untracked)} untracked non-ignored file(s)")
    is_clean = has_commits and not modified and not untracked
    return {
        "repo": str(src_pkg),
        "has_commits": has_commits,
        "head_sha": head_sha,
        "modified": modified,
        "untracked": untracked,
        "is_clean": is_clean,
        "summary": ", ".join(reasons) if reasons else "clean",
    }


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


_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")


def _transform_markdown_links(text: str, *, sibling_prefix: str, package_dir: str) -> str:
    """Rewrite inline and reference-style sibling links outside fenced code blocks."""
    inline_re = re.compile(
        r"(?<=\]\()" + re.escape(sibling_prefix) + r"([^#)\s]*)(#[^)\s]*)?"
    )
    ref_re = re.compile(
        r"^([ ]{0,3}\[[^\]]+\]:[ \t]+<?)" + re.escape(sibling_prefix) + r"([^#>\s]*)(#[^>\s]*)?"
    )

    def _rewrite_target(path_part: str, anchor: str | None) -> str:
        anch = anchor or ""
        if not path_part:
            return f"{posixpath.relpath('.', package_dir)}/{anch}"
        rel_target = posixpath.relpath(path_part, package_dir)
        if path_part.endswith("/") and not rel_target.endswith("/"):
            rel_target += "/"
        return f"{rel_target}{anch}"

    out_lines: list[str] = []
    fence_char = ""
    fence_len = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if fence_char:
            out_lines.append(line)
            if re.match(rf"^[ ]{{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ ]*$", stripped):
                fence_char = ""
                fence_len = 0
            continue

        m_open = _FENCE_OPEN_RE.match(stripped)
        if m_open:
            marker, info = m_open.group(1), m_open.group(2)
            # CommonMark: backtick fence info strings cannot contain backticks.
            if marker[0] == "~" or "`" not in info:
                fence_char = marker[0]
                fence_len = len(marker)
                out_lines.append(line)
                continue

        if sibling_prefix in line:
            line = inline_re.sub(lambda m: _rewrite_target(m.group(1), m.group(2)), line)
            line = ref_re.sub(lambda m: f"{m.group(1)}{_rewrite_target(m.group(2), m.group(3))}", line)
        out_lines.append(line)
    return "".join(out_lines)


def _transform_content(pkg: str, rel: str, raw: bytes) -> bytes:
    """Resolve sibling package references in the monorepo's directory layout."""
    if rel.endswith(".md"):
        # Sibling READMEs link out to the adjacent fast-silicon checkout. The imported copy lives inside it.
        sibling_prefix = "../" * len(Path(rel).parts) + "fast-silicon/"
        if sibling_prefix.encode() in raw:
            package_dir = posixpath.dirname(f"packages/{pkg}/{rel}")
            text = _transform_markdown_links(
                raw.decode("utf-8"),
                sibling_prefix=sibling_prefix,
                package_dir=package_dir,
            )
            return text.encode("utf-8")
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
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help=(
            "Exit non-zero and refuse to sync if any sibling repo has zero commits, "
            "uncommitted tracked changes, or untracked non-ignored files."
        ),
    )
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    packages_dir = repo_root / "packages"
    selected = tuple(args.packages) if args.packages else all_packages

    # Inspect sibling git hygiene upfront so `--require-clean` fails before writing any files,
    # while default `make sync` prints clear visibility warnings for dirty / 0-commit trees.
    dirty_siblings: list[tuple[str, dict[str, Any]]] = []
    for pkg in selected:
        src_pkg = source_root / pkg
        if not src_pkg.is_dir():
            continue
        hygiene = inspect_sibling_git_hygiene(src_pkg)
        if not hygiene["is_clean"]:
            dirty_siblings.append((pkg, hygiene))
            sample_paths = (hygiene["modified"] + hygiene["untracked"])[:4]
            sample_txt = f" (e.g. {', '.join(sample_paths)})" if sample_paths else ""
            prefix = "ERROR" if args.require_clean else "WARNING"
            print(
                f"[{prefix}] sibling {pkg} is dirty: {hygiene['summary']}{sample_txt}",
                file=sys.stderr,
            )

    if args.require_clean and dirty_siblings:
        names = ", ".join(pkg for pkg, _ in dirty_siblings)
        print(
            f"[ABORT] --require-clean refused sync due to dirty/uncommitted/0-commit sibling repo(s): {names}. "
            "Commit or clean those sibling checkouts first.",
            file=sys.stderr,
        )
        return 2

    if not args.check:
        packages_dir.mkdir(parents=True, exist_ok=True)

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
