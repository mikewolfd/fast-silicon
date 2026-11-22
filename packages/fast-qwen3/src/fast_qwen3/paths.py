"""Stdlib-only home, artifact directory, and Hugging Face preset resolution for `fast-qwen3`."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_HF_REPO: str = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

MODEL_PRESETS: dict[str, str] = {
    "0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
    "qwen3.5-0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
    "0.8b-bf16": "Qwen/Qwen3.5-0.8B",
    "2b": "mlx-community/Qwen3.5-2B-MLX-4bit",
    "qwen3.5-2b": "mlx-community/Qwen3.5-2B-MLX-4bit",
    "4b": DEFAULT_HF_REPO,
    "4b-optiq": DEFAULT_HF_REPO,
    "qwen3.5-4b": DEFAULT_HF_REPO,
    "4b-bf16": "Qwen/Qwen3.5-4B",
    "9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "9b-optiq": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "qwen3.5-9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "qwen3.5-9b-optiq": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "27b": "mlx-community/Qwen3.5-27B-4bit",
    "qwen3.5-27b": "mlx-community/Qwen3.5-27B-4bit",
    "35b-a3b": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "qwen3.6-35b-a3b": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "qwen3.5-35b-a3b": "mlx-community/Qwen3.5-35B-A3B-4bit",
}

SNAPSHOT_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.safetensors",
    "optiq/*.safetensors",
    "optiq/*.json",
)


def resolve_preset_repo(model_or_preset: str | None) -> str:
    """Map short preset alias (e.g. ``'4b'``, ``'0.8b'``, ``'35b-a3b'``) to a canonical HF repo ID."""
    if not model_or_preset:
        return DEFAULT_HF_REPO
    key = str(model_or_preset).strip()
    return MODEL_PRESETS.get(key.lower(), key)


def resolve_home(home: str | Path | None = None) -> Path:
    """Resolve `fast-qwen3` home directory (`explicit > FAST_QWEN3_HOME > FAST_ARTIFACTS_ROOT > cache`)."""
    if home is not None:
        return Path(home).expanduser().resolve()
    env_home = os.environ.get("FAST_QWEN3_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    cand = Path(__file__).resolve().parents[2]
    if (cand / "artifacts").is_dir():
        return cand
    env_art = os.environ.get("FAST_ARTIFACTS_ROOT")
    if env_art:
        art_root = Path(env_art).expanduser().resolve()
        return (art_root / "fast-qwen3").resolve()
    return (Path.home() / ".cache" / "fast-qwen3").resolve()


@dataclass(frozen=True)
class Qwen3Paths:
    """Resolved workspace and cache locations for `fast-qwen3`."""

    home: Path
    artifacts_dir: Path
    tokenizer_dir: Path
    vocab_dir: Path

    @property
    def vocab_remap_npz(self) -> Path:
        return self.vocab_dir / "qwen3_vocab_remap.npz"

    @staticmethod
    def is_complete_snapshot(snap: Path) -> bool:
        """Verify snapshot contains `config.json`, `tokenizer.json`, and at least one `.safetensors` shard."""
        if not (snap.is_dir() and (snap / "config.json").is_file() and (snap / "tokenizer.json").is_file()):
            return False
        return any(snap.glob("*.safetensors")) or any(snap.glob("optiq/*.safetensors"))

    def resolve_model_dir(
        self,
        model_dir: str | Path | None = None,
        *,
        repo_id: str | None = None,
        local_files_only: bool = True,
    ) -> Path:
        """Resolve local model directory, preset alias, or cached/downloaded Hugging Face snapshot."""
        raw = model_dir if model_dir is not None else os.environ.get("FAST_QWEN3_MODEL_DIR")
        if raw is not None:
            cand = Path(str(raw)).expanduser()
            if cand.is_dir() and (cand / "config.json").is_file():
                return cand.resolve()
            eff_repo = resolve_preset_repo(str(raw))
        else:
            eff_repo = resolve_preset_repo(repo_id)

        as_path = Path(eff_repo).expanduser()
        if as_path.is_dir() and (as_path / "config.json").is_file():
            return as_path.resolve()

        # Fast offline HF hub snapshot check before importing huggingface_hub
        slug = "models--" + eff_repo.replace("/", "--")
        hub_snaps = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots"
        if local_files_only and hub_snaps.is_dir():
            snaps = sorted(s for s in hub_snaps.iterdir() if s.is_dir())
            for snap in reversed(snaps):
                if self.is_complete_snapshot(snap):
                    return snap.resolve()

        from huggingface_hub import snapshot_download

        downloaded = Path(
            snapshot_download(
                repo_id=eff_repo,
                allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
                local_files_only=local_files_only,
            )
        ).resolve()
        if not self.is_complete_snapshot(downloaded):
            raise FileNotFoundError(
                f"Incomplete local snapshot for {eff_repo!r} at {downloaded} (missing weights or tokenizer.json)"
            )
        return downloaded

    def resolve_tokenizer_json(self, model_dir: str | Path | None = None) -> Path:
        if model_dir is not None:
            mdir = Path(model_dir).expanduser()
            if (mdir / "tokenizer.json").is_file():
                return (mdir / "tokenizer.json").resolve()
        if (self.tokenizer_dir / "tokenizer.json").is_file():
            return (self.tokenizer_dir / "tokenizer.json").resolve()
        raise FileNotFoundError(
            f"Could not find tokenizer.json in {model_dir!r} or {self.tokenizer_dir}"
        )


def resolve_paths(home: str | Path | None = None) -> Qwen3Paths:
    root = resolve_home(home)
    art = root / "artifacts"
    return Qwen3Paths(
        home=root,
        artifacts_dir=art,
        tokenizer_dir=art / "tokenizer",
        vocab_dir=art / "vocab",
    )
