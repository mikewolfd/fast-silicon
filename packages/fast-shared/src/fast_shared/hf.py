"""Shared HuggingFace snapshot resolution and tokenizer asset copying."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

DEFAULT_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
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

    return Path(snapshot_download(repo_id=repo_id, local_files_only=local_files_only))


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
