"""Zero-MLX OCR model path resolution and v7final GFM preset discovery."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OVIS_REPO: str = "mlx-community/OvisOCR2-8bit"
DEFAULT_OVIS_GFM_REPO: str = "mikewolfd/OvisOCR2-8bit-gfm"

OCR_MODEL_PRESETS: dict[str, str] = {
    "ovis": DEFAULT_OVIS_REPO,
    "ovisocr2": DEFAULT_OVIS_REPO,
    "ovis2": DEFAULT_OVIS_REPO,
    "ovis-gfm": DEFAULT_OVIS_GFM_REPO,
    "ovisocr2-gfm": DEFAULT_OVIS_GFM_REPO,
    "v7final": DEFAULT_OVIS_GFM_REPO,
    "nuextract": "numind/NuExtract3-mlx-4bits",
    "nuextract3": "numind/NuExtract3-mlx-4bits",
}

OCR_PRESET_FALLBACKS: dict[str, tuple[str, ...]] = {
    "nuextract": ("mlx-community/NuExtract3-bf16",),
    "nuextract3": ("mlx-community/NuExtract3-bf16",),
}

_GFM_PRESETS: frozenset[str] = frozenset({"ovis-gfm", "ovisocr2-gfm", "v7final"})
_BASE_OVIS_PRESETS: frozenset[str] = frozenset({"", "ovis", "ovisocr2", "ovis2"})


def _hub_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        roots.append(Path(hub_cache).expanduser())
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            deduped.append(root)
    return tuple(deduped)


def _is_complete_ocr_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def _find_local_hf_snapshot(repo_id: str) -> Path | None:
    slug = "models--" + repo_id.replace("/", "--")
    for hub_root in _hub_roots():
        hub_snaps = hub_root / slug / "snapshots"
        if hub_snaps.is_dir():
            for snap in reversed(sorted(s for s in hub_snaps.iterdir() if s.is_dir())):
                if _is_complete_ocr_model_dir(snap):
                    return snap.resolve()
    return None


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_v7final_local_dir() -> Path | None:
    """Return a locally staged validated v7final merged 8-bit checkpoint if present on disk."""
    env_dir = os.environ.get("FAST_OCR_GFM_MODEL_DIR", "").strip()
    if env_dir:
        cand = Path(env_dir).expanduser()
        if _is_complete_ocr_model_dir(cand):
            return cand.resolve()
        raise FileNotFoundError(
            f"FAST_OCR_GFM_MODEL_DIR={env_dir!r} does not point to a complete OvisOCR2 snapshot."
        )
    local_gfm = _find_local_hf_snapshot(DEFAULT_OVIS_GFM_REPO)
    if local_gfm is not None:
        return local_gfm
    local_cache_cand = Path.home() / ".cache" / "fast-silicon" / "ovisocr2_merged_gfm_v7final_8bit"
    if _is_complete_ocr_model_dir(local_cache_cand):
        return local_cache_cand.resolve()
    return None


def resolve_ocr_model_dir(
    model_path_or_repo: str | Path | None = None,
    *,
    local_files_only: bool = True,
    prefer_gfm: bool | None = None,
) -> Path:
    """Resolve an OCR VLM snapshot path from local path, preset alias, or Hugging Face Hub cache."""
    raw = str(model_path_or_repo).strip() if model_path_or_repo is not None else os.environ.get("FAST_OCR_MODEL_DIR", "").strip()
    preset_key = raw.lower() if raw else ""

    want_gfm = (prefer_gfm if prefer_gfm is not None else _env_truthy("FAST_OCR_PREFER_GFM"))
    if preset_key in _GFM_PRESETS:
        gfm_dir = resolve_v7final_local_dir()
        if gfm_dir is not None:
            return gfm_dir
    elif want_gfm and preset_key in _BASE_OVIS_PRESETS:
        gfm_dir = resolve_v7final_local_dir()
        if gfm_dir is not None:
            return gfm_dir

    if raw:
        cand = Path(raw).expanduser()
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
        repo_id = OCR_MODEL_PRESETS.get(preset_key, raw)
    else:
        repo_id = DEFAULT_OVIS_REPO

    as_dir = Path(repo_id).expanduser()
    if as_dir.is_dir() and (as_dir / "config.json").is_file():
        return as_dir.resolve()

    local_snap = _find_local_hf_snapshot(repo_id)
    if local_snap is not None:
        return local_snap

    for fb_repo in OCR_PRESET_FALLBACKS.get(preset_key, ()):
        fb_snap = _find_local_hf_snapshot(fb_repo)
        if fb_snap is not None:
            return fb_snap

    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"],
        )
    ).resolve()
    if not (downloaded / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete OCR model snapshot for {repo_id!r} at {downloaded}")
    return downloaded
