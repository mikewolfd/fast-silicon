"""Filesystem and HuggingFace checkpoint path resolution for `fast-gpt-oss`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fast_shared.hf import resolve_hf_model_dir
from fast_shared.paths import resolve_home_dir

DEFAULT_HF_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q4"


@dataclass(frozen=True)
class GptOssPaths:
    home: Path
    tokenizer_dir: Path
    vocab_remap_npz: Path
    local_hf_dir: Path

    def resolve_model_dir(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str = DEFAULT_HF_REPO,
        local_files_only: bool = True,
    ) -> Path:
        if model_dir is not None:
            p = Path(model_dir).expanduser().resolve()
            if not (p / "config.json").exists():
                raise FileNotFoundError(f"Missing config.json in model_dir={p}")
            return p
        if (self.local_hf_dir / "config.json").exists() and any(self.local_hf_dir.glob("*.safetensors")):
            return self.local_hf_dir
        return resolve_hf_model_dir(None, repo_id, local_files_only=local_files_only)

    def resolve_tokenizer_json(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str = DEFAULT_HF_REPO,
        local_files_only: bool = True,
    ) -> Path:
        if model_dir is None:
            cand = self.tokenizer_dir / "tokenizer.json"
            if cand.exists():
                return cand
        mdir = self.resolve_model_dir(model_dir, repo_id=repo_id, local_files_only=local_files_only)
        tok = mdir / "tokenizer.json"
        if not tok.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.tokenizer_dir} or {mdir}")
        return tok


def resolve_paths(home: str | Path | None = None) -> GptOssPaths:
    root = resolve_home_dir(
        __file__,
        env_var="FAST_GPT_OSS_HOME",
        cache_name="fast-gpt-oss",
        home=home,
        marker_dirs=("artifacts",),
    )
    art = root / "artifacts"
    return GptOssPaths(
        home=root,
        tokenizer_dir=art / "tokenizer",
        vocab_remap_npz=art / "vocab" / "o200k_remap_131072.npz",
        local_hf_dir=art / "hf_model",
    )


__all__ = ["DEFAULT_HF_REPO", "GptOssPaths", "resolve_paths"]
