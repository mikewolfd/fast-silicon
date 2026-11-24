"""Filesystem discovery for weights, tokenizer assets, ANE CoreML bundles, and native dylibs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

import numpy as np

from fast_shared.paths import resolve_home_dir

_PACKAGE_CACHE_NAME = "fast-bge-small"
_HOME_ENV_VAR = "FAST_BGE_SMALL_HOME"
DEFAULT_PREBUILT_HF_REPO = "mikewolfd/bge-small-en-v1.5-coreml-ane"


@dataclass(frozen=True)
class RepoPaths:
    home: Path
    weights_npz: Path
    tokenizer_dir: Path
    tokenizer_json: Path
    vocab_txt: Path
    ane_models_dir: Path
    build_dir: Path
    ane_dylib: Path
    cpu_dylib: Path
    ane_cli: Path


def resolve_home(home: str | Path | None = None, *, anchor_file: str | Path | None = None) -> Path:
    """Resolve the runtime data root (explicit > env > checkout with artifacts/ > FAST_ARTIFACTS_ROOT > sibling > $HOME/.cache)."""
    return resolve_home_dir(
        anchor_file if anchor_file is not None else __file__,
        env_var=_HOME_ENV_VAR,
        cache_name=_PACKAGE_CACHE_NAME,
        home=home,
        marker_dirs=("artifacts",),
    )


def resolve_paths(home: str | Path | None = None) -> RepoPaths:
    root = resolve_home(home)
    checkout_root = Path(__file__).resolve().parent.parent.parent

    artifacts = root / "artifacts"
    build_dir = root / "build"
    if (
        home is None
        and not os.environ.get(_HOME_ENV_VAR)
        and (checkout_root / "build").exists()
    ):
        build_dir = checkout_root / "build"
    elif not build_dir.exists() and (checkout_root / "build").exists():
        build_dir = checkout_root / "build"

    return RepoPaths(
        home=root,
        weights_npz=artifacts / "weights" / "bge_small_f16.npz",
        tokenizer_dir=artifacts / "tokenizer",
        tokenizer_json=artifacts / "tokenizer" / "tokenizer.json",
        vocab_txt=artifacts / "tokenizer" / "vocab.txt",
        ane_models_dir=artifacts / "ane_models",
        build_dir=build_dir,
        ane_dylib=build_dir / "libbge_ane_runner.dylib",
        cpu_dylib=build_dir / "libbge_cpu_query.dylib",
        ane_cli=build_dir / "bge-ane-cli",
    )


def require_runtime_paths(home: str | Path | None = None, require_ane: bool = False) -> RepoPaths:
    p = resolve_paths(home)
    if not p.weights_npz.exists():
        raise FileNotFoundError(
            f"Missing weights file: {p.weights_npz}. Run `bge-small setup-weights` or `scripts/export_weights.py`."
        )
    if not p.tokenizer_json.exists():
        raise FileNotFoundError(
            f"Missing tokenizer file: {p.tokenizer_json}. Run `bge-small setup-weights` or `scripts/export_weights.py`."
        )
    if require_ane:
        if not p.ane_dylib.exists():
            raise FileNotFoundError(f"Missing native ANE runner library: {p.ane_dylib}. Run `make native` in {p.home}.")
        if not p.ane_models_dir.exists():
            raise FileNotFoundError(f"Missing ANE models directory: {p.ane_models_dir}.")
    return p


def write_host_table_sidecars(
    weights_dir: Path,
    wt0_f16: np.ndarray,
    pos_f16: np.ndarray,
) -> tuple[Path, Path, Path]:
    """Atomically write raw little-endian float16 `wt0` and `pos` binary sidecars for clean read-only mmap."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    wt0_contig = np.ascontiguousarray(wt0_f16, dtype=np.dtype("<f2"))
    pos_contig = np.ascontiguousarray(pos_f16, dtype=np.dtype("<f2"))

    wt0_bin = weights_dir / "bge_wt0_f16.bin"
    pos_bin = weights_dir / "bge_pos_f16.bin"
    meta_json = weights_dir / "bge_tables_meta.json"
    token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"

    wt0_tmp = weights_dir / f".bge_wt0_f16.bin.{token}"
    pos_tmp = weights_dir / f".bge_pos_f16.bin.{token}"
    meta_tmp = weights_dir / f".bge_tables_meta.json.{token}"
    try:
        wt0_contig.tofile(wt0_tmp)
        pos_contig.tofile(pos_tmp)
        meta_payload = {
            "vocab_size": int(wt0_contig.shape[0]),
            "max_pos": int(pos_contig.shape[0]),
            "hidden_size": int(wt0_contig.shape[1]),
        }
        meta_tmp.write_text(json.dumps(meta_payload), encoding="utf-8")
        os.replace(wt0_tmp, wt0_bin)
        os.replace(pos_tmp, pos_bin)
        os.replace(meta_tmp, meta_json)
    finally:
        for tmp_f in (wt0_tmp, pos_tmp, meta_tmp):
            tmp_f.unlink(missing_ok=True)
    return wt0_bin, pos_bin, meta_json


def load_or_mmap_host_tables(weights_npz: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return read-only `(wt0_f16, pos_f16)` host embedding tables backed by clean OS file cache pages."""
    npz_path = Path(weights_npz)
    weights_dir = npz_path.parent
    wt0_bin = weights_dir / "bge_wt0_f16.bin"
    pos_bin = weights_dir / "bge_pos_f16.bin"
    meta_json = weights_dir / "bge_tables_meta.json"

    if not (wt0_bin.is_file() and pos_bin.is_file() and meta_json.is_file()):
        st = npz_path.stat()
        cache_dir = (
            Path.home()
            / ".cache"
            / "fast-bge-small"
            / "mmap_tables"
            / f"v1_{st.st_size}_{st.st_mtime_ns}"
        )
        wt0_bin = cache_dir / "bge_wt0_f16.bin"
        pos_bin = cache_dir / "bge_pos_f16.bin"
        meta_json = cache_dir / "bge_tables_meta.json"
        if not (wt0_bin.is_file() and pos_bin.is_file() and meta_json.is_file()):
            with np.load(npz_path) as raw:
                wt0 = np.ascontiguousarray(
                    raw["embeddings.word_embeddings.weight"].astype(np.float16)
                    + raw["embeddings.token_type_embeddings.weight"][0:1].astype(np.float16)
                )
                pos = np.ascontiguousarray(raw["embeddings.position_embeddings.weight"].astype(np.float16))
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                write_host_table_sidecars(cache_dir, wt0, pos)
            except OSError:
                wt0.setflags(write=False)
                pos.setflags(write=False)
                return wt0, pos

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab_size"])
    max_pos = int(meta["max_pos"])
    hidden_size = int(meta.get("hidden_size", 384))
    wt0_mm = np.memmap(wt0_bin, dtype=np.float16, mode="r", shape=(vocab_size, hidden_size))
    pos_mm = np.memmap(pos_bin, dtype=np.float16, mode="r", shape=(max_pos, hidden_size))
    return wt0_mm, pos_mm


def ensure_prebuilt_artifacts(
    home: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_PREBUILT_HF_REPO,
    local_files_only: bool = False,
    include_ane: bool = True,
    force: bool = False,
    downloader: object | None = None,
) -> RepoPaths:
    """Materialize prebuilt weights, tokenizer, and Core ML ANE buckets from Hugging Face."""
    from fast_shared.hf import materialize_hf_snapshot

    p = resolve_paths(home)
    subdirs = ("weights", "tokenizer", "ane_models") if include_ane else ("weights", "tokenizer")
    materialize_hf_snapshot(
        repo_id,
        p.home / "artifacts",
        subdirs=subdirs,
        local_files_only=local_files_only,
        overwrite=force,
        downloader=downloader,  # type: ignore[arg-type]
    )
    return p
