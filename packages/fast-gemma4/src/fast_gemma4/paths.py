"""Filesystem and model artifact path resolution for `fast-gemma4`."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from fast_shared.hf import resolve_hf_model_dir
from fast_shared.paths import resolve_home_dir

DEFAULT_HF_REPO = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"


@dataclass(frozen=True)
class Gemma4Paths:
    home: Path
    tokenizer_dir: Path
    vocab_remap_npz: Path
    early14_weights_npz: Path
    ane_models_dir: Path
    ane_fp16_bundle: Path
    ane_w8_bundle: Path
    lib_ane_prefix: Path
    ane_cli: Path
    local_hf_dir: Path

    def resolve_model_dir(
        self,
        model_dir: str | Path | None = None,
        *,
        local_files_only: bool = True,
    ) -> Path:
        if model_dir is not None:
            p = Path(model_dir).expanduser().resolve()
            if not (p / "config.json").exists():
                raise FileNotFoundError(f"Missing config.json in model_dir={p}")
            return p
        if (self.local_hf_dir / "config.json").exists() and any(self.local_hf_dir.glob("model*.safetensors")):
            return self.local_hf_dir
        return resolve_hf_model_dir(None, DEFAULT_HF_REPO, local_files_only=local_files_only)

    def resolve_tokenizer_json(self, model_dir: str | Path | None = None, *, local_files_only: bool = True) -> Path:
        cand = self.tokenizer_dir / "tokenizer.json"
        if cand.exists():
            return cand
        mdir = self.resolve_model_dir(model_dir, local_files_only=local_files_only)
        tok = mdir / "tokenizer.json"
        if not tok.exists():
            raise FileNotFoundError(f"Missing tokenizer.json in {self.tokenizer_dir} or {mdir}")
        return tok

    @staticmethod
    def checkpoint_ane_key(model_dir: Path) -> str:
        resolved = model_dir.expanduser().resolve()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.name).strip("_") or "model"
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"{slug}-{digest}"

    def _is_default_model_dir(self, model_dir: str | Path | None, *, local_files_only: bool = True) -> bool:
        if model_dir is None:
            return True
        try:
            resolved = Path(model_dir).expanduser().resolve()
            default_dir = self.resolve_model_dir(None, local_files_only=local_files_only).expanduser().resolve()
            return resolved == default_dir
        except Exception:
            return False

    def resolve_early14_weights_npz(
        self,
        model_dir: str | Path | None = None,
        *,
        local_files_only: bool = True,
    ) -> Path:
        if self._is_default_model_dir(model_dir, local_files_only=local_files_only):
            return self.early14_weights_npz
        assert model_dir is not None
        key = self.checkpoint_ane_key(Path(model_dir))
        return self.early14_weights_npz.parent / key / "gemma4_early14_f16.npz"

    def resolve_ane_bundle(
        self,
        variant: str = "fp16",
        *,
        model_dir: str | Path | None = None,
        local_files_only: bool = True,
    ) -> Path | None:
        fname = self.ane_fp16_bundle.name if variant == "fp16" else self.ane_w8_bundle.name
        if self._is_default_model_dir(model_dir, local_files_only=local_files_only):
            p = self.ane_fp16_bundle if variant == "fp16" else self.ane_w8_bundle
            return p if p.exists() else None
        assert model_dir is not None
        resolved = Path(model_dir).expanduser().resolve()
        for cand in (
            resolved / "ane_models" / fname,
            self.ane_models_dir / self.checkpoint_ane_key(resolved) / fname,
        ):
            if cand.exists():
                return cand
        return None


def resolve_paths(home: str | Path | None = None) -> Gemma4Paths:
    root = resolve_home_dir(
        __file__,
        env_var="FAST_GEMMA4_HOME",
        cache_name="fast-gemma4",
        home=home,
    )
    art = root / "artifacts"
    ane_dir = art / "ane_models"
    return Gemma4Paths(
        home=root,
        tokenizer_dir=art / "tokenizer",
        vocab_remap_npz=art / "vocab" / "vocab_remap_150786.npz",
        early14_weights_npz=art / "weights" / "gemma4_early14_f16.npz",
        ane_models_dir=ane_dir,
        ane_fp16_bundle=ane_dir / "gemma4_early14_prefix_S256_fp16.mlmodelc",
        ane_w8_bundle=ane_dir / "gemma4_early14_prefix_S256_w8.mlmodelc",
        lib_ane_prefix=root / "build" / "libgemma4_ane_prefix.dylib",
        ane_cli=root / "build" / "gemma4-ane-cli",
        local_hf_dir=art / "hf_model",
    )
