"""Filesystem and HuggingFace checkpoint path resolution for `fast-gpt-oss`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fast_shared.hf import resolve_hf_model_dir
from fast_shared.paths import resolve_home_dir

DEFAULT_HF_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q8"
FALLBACK_HF_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q4"


def _is_complete_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


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
            if _is_complete_model_dir(p):
                return p
            if local_files_only and (
                "gpt-oss-20b-MXFP4-Q8" in str(model_dir) or "gpt-oss-20b-MXFP4-Q8" in str(p)
            ):
                if _is_complete_model_dir(self.local_hf_dir):
                    return self.local_hf_dir
                try:
                    fb = resolve_hf_model_dir(None, FALLBACK_HF_REPO, local_files_only=True)
                    if _is_complete_model_dir(fb):
                        return fb
                except Exception:
                    pass
            if not (p / "config.json").exists():
                raise FileNotFoundError(f"Missing config.json in model_dir={p}")
            if not any(p.glob("*.safetensors")):
                raise FileNotFoundError(f"Missing *.safetensors in model_dir={p}")
            return p
        if _is_complete_model_dir(self.local_hf_dir):
            return self.local_hf_dir

        primary_err: Exception | None = None
        cand: Path | None = None
        try:
            cand = resolve_hf_model_dir(None, repo_id, local_files_only=local_files_only)
        except Exception as exc:
            primary_err = exc

        if cand is not None and (not local_files_only or _is_complete_model_dir(cand) or repo_id != DEFAULT_HF_REPO):
            return cand

        if local_files_only and repo_id == DEFAULT_HF_REPO:
            try:
                fb = resolve_hf_model_dir(None, FALLBACK_HF_REPO, local_files_only=True)
                if _is_complete_model_dir(fb):
                    return fb
            except Exception:
                pass

        if primary_err is not None:
            raise primary_err
        if cand is not None:
            raise FileNotFoundError(
                f"Incomplete local snapshot for {repo_id} at {cand} (missing config.json or *.safetensors) "
                f"and fallback {FALLBACK_HF_REPO} is unavailable."
            )
        raise FileNotFoundError(f"Unable to resolve model directory for repo_id={repo_id!r}")

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


__all__ = ["DEFAULT_HF_REPO", "FALLBACK_HF_REPO", "GptOssPaths", "resolve_paths"]
