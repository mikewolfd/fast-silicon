"""Shared HuggingFace snapshot resolution and tokenizer asset copying."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Sequence

DEFAULT_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)


def is_hf_offline(local_files_only: bool = False) -> bool:
    """Return True when offline mode is requested explicitly or via `HF_HUB_OFFLINE`."""
    return bool(local_files_only) or (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    )


def resolve_hf_model_dir(
    model_dir: str | Path | None,
    repo_id: str,
    *,
    local_files_only: bool = False,
) -> Path:
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    from huggingface_hub import snapshot_download

    eff_offline = is_hf_offline(local_files_only)
    return Path(snapshot_download(repo_id=repo_id, local_files_only=eff_offline))


def _snapshot_has_payload(
    snap_dir: Path,
    subdirs: Sequence[str] | None = None,
) -> bool:
    """Return True when `snap_dir` contains non-metadata payload files (not just README/.gitattributes)."""
    if not snap_dir.is_dir():
        return False
    skip_names = {"README.md", ".gitattributes"}
    if subdirs is not None:
        if not subdirs:
            return False
        for sub in subdirs:
            sub_path = snap_dir / sub
            if sub_path.is_file():
                continue
            if not sub_path.is_dir():
                return False
            has_file = any(
                bool(filenames)
                for _cur, _dirnames, filenames in os.walk(sub_path, followlinks=True)
            )
            if not has_file:
                return False
        return True
    return any(p.name not in skip_names for p in snap_dir.iterdir())


def _resolve_cached_or_download(
    repo_id: str,
    *,
    local_files_only: bool = False,
    downloader: Callable[..., str | Path] | None = None,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    eff_offline = is_hf_offline(local_files_only)
    if downloader is not None:
        if eff_offline:
            return Path(downloader(repo_id=repo_id, local_files_only=True))
        try:
            cand = Path(downloader(repo_id=repo_id, local_files_only=True))
            if validator is None or validator(cand):
                return cand
        except Exception:
            pass
        return Path(downloader(repo_id=repo_id, local_files_only=False))
    if eff_offline:
        return resolve_hf_model_dir(None, repo_id, local_files_only=True)
    try:
        cand = resolve_hf_model_dir(None, repo_id, local_files_only=True)
        if validator is None or validator(cand):
            return cand
    except Exception:
        pass
    return resolve_hf_model_dir(None, repo_id, local_files_only=False)


def materialize_hf_snapshot(
    repo_id: str,
    dst_dir: Path | str,
    *,
    subdirs: Sequence[str] | None = None,
    local_files_only: bool = False,
    overwrite: bool = False,
    downloader: Callable[..., str | Path] | None = None,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    """Materialize a Hugging Face snapshot into `dst_dir` with symlink-free hardlinks/copies.

    Skips repository metadata files (`README.md`, `.gitattributes`) and optionally limits
    materialization to specific top-level `subdirs` that exist in the snapshot.
    """
    eff_validator = validator if validator is not None else (lambda p: _snapshot_has_payload(p, subdirs))
    snap_dir = _resolve_cached_or_download(
        repo_id,
        local_files_only=local_files_only,
        downloader=downloader,
        validator=eff_validator,
    ).resolve()
    dst_root = Path(dst_dir).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    skip_names = {"README.md", ".gitattributes"}
    if subdirs is None:
        roots: list[tuple[Path, Path]] = [(snap_dir, dst_root)]
    else:
        roots = [
            (snap_dir / sub, dst_root / sub)
            for sub in subdirs
            if (snap_dir / sub).exists()
        ]

    for src_base, dst_base in roots:
        for cur_dir, _dirnames, filenames in os.walk(src_base, followlinks=True):
            rel = Path(cur_dir).relative_to(src_base)
            target_dir = dst_base / rel if rel != Path(".") else dst_base
            target_dir.mkdir(parents=True, exist_ok=True)
            for fname in filenames:
                if fname in skip_names and rel == Path(".") and src_base == snap_dir:
                    continue
                src_f = (Path(cur_dir) / fname).resolve()
                dst_f = target_dir / fname
                if dst_f.exists() and not overwrite:
                    continue
                fd, tmp_name = tempfile.mkstemp(prefix=f".{fname}.", dir=str(target_dir))
                os.close(fd)
                tmp_p = Path(tmp_name)
                try:
                    tmp_p.unlink(missing_ok=True)
                    try:
                        os.link(src_f, tmp_p)
                    except OSError:
                        shutil.copy2(src_f, tmp_p)
                    os.replace(tmp_p, dst_f)
                finally:
                    if tmp_p.exists():
                        tmp_p.unlink(missing_ok=True)

    return dst_root


def copy_tokenizer_files(
    src_dir: Path,
    dst_dir: Path,
    filenames: Sequence[str] = DEFAULT_TOKENIZER_FILES,
) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for fname in filenames:
        src = src_dir / fname
        if src.exists():
            dst = dst_dir / fname
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied
