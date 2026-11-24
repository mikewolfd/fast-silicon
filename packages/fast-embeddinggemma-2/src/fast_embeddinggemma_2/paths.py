"""Filesystem discovery for weights, tokenizer assets, CoreML bundles, and native dylibs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from fast_shared.hf import resolve_hf_model_dir
from fast_shared.paths import resolve_home_dir

_PACKAGE_CACHE_NAME = "fast-embeddinggemma-2"
_HOME_ENV_VAR = "FAST_EMBEDDINGGEMMA2_HOME"
DEFAULT_PREBUILT_HF_REPO = "mikewolfd/embeddinggemma-2-coreml-ane"
OPENINFINITY_REPO_ID = "OpenInfinity/embeddinggemma-2-coreml-int8"
DEFAULT_VOCAB_SIZE = 262144
DEFAULT_HIDDEN_SIZE = 512


@dataclass(frozen=True)
class RepoPaths:
    home: Path
    weights_npz: Path
    tokenizer_dir: Path
    tokenizer_json: Path
    ane_models_dir: Path
    build_dir: Path
    coreml_dylib: Path
    io_dylib: Path
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
        weights_npz=artifacts / "weights" / "embeddinggemma_2_f16.npz",
        tokenizer_dir=artifacts / "tokenizer",
        tokenizer_json=artifacts / "tokenizer" / "tokenizer.json",
        ane_models_dir=artifacts / "ane_models",
        build_dir=build_dir,
        coreml_dylib=build_dir / "libemb2_coreml_runner.dylib",
        io_dylib=build_dir / "libemb2_fast_io.dylib",
        ane_cli=build_dir / "emb2-ane-cli",
    )


def require_runtime_paths(home: str | Path | None = None, require_ane: bool = False) -> RepoPaths:
    p = resolve_paths(home)
    if not p.weights_npz.exists():
        raise FileNotFoundError(
            f"Missing weights file: {p.weights_npz}. Run `embeddinggemma-2 setup-weights` or `scripts/export_weights.py`."
        )
    if not p.tokenizer_json.exists():
        raise FileNotFoundError(
            f"Missing tokenizer file: {p.tokenizer_json}. Run `embeddinggemma-2 setup-weights` or `scripts/export_weights.py`."
        )
    if require_ane:
        for req in (p.coreml_dylib, p.io_dylib):
            if not req.exists():
                raise FileNotFoundError(f"Missing native library: {req}. Run `make native` in {p.home}.")
        if not p.ane_models_dir.exists():
            raise FileNotFoundError(f"Missing CoreML models directory: {p.ane_models_dir}.")
    return p


def write_int8_token_table(
    target_dir: str | Path,
    embed_f16_or_f32: np.ndarray,
    *,
    pre_scale: float = float(DEFAULT_HIDDEN_SIZE**0.5),
) -> tuple[Path, Path, Path]:
    """Quantize `[V, H]` embeddings to per-row symmetric `int8` + `fp16` scale sidecars atomically."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_size, hidden_size = int(embed_f16_or_f32.shape[0]), int(embed_f16_or_f32.shape[1])
    i8_out = np.empty((vocab_size, hidden_size), dtype=np.int8)
    sc_out = np.empty((vocab_size,), dtype=np.dtype("<f2"))

    chunk_rows = 16384
    mult = np.float32(pre_scale)
    for r0 in range(0, vocab_size, chunk_rows):
        r1 = min(vocab_size, r0 + chunk_rows)
        blk = embed_f16_or_f32[r0:r1].astype(np.float32) * mult
        max_abs = np.maximum(np.max(np.abs(blk), axis=1), 1e-8)
        sc_f16 = (max_abs / 127.0).astype(np.float16)
        sc_safe = np.maximum(sc_f16.astype(np.float32), 1e-8)
        q_i8 = np.clip(np.round(blk / sc_safe[:, None]), -127.0, 127.0).astype(np.int8)
        i8_out[r0:r1] = q_i8
        sc_out[r0:r1] = sc_f16.astype(np.dtype("<f2"), copy=False)

    i8_path = out_dir / "embeddings.int8"
    sc_path = out_dir / "embeddings.int8.scale.f16"
    meta_path = out_dir / "embed_table_meta.json"
    token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"

    i8_tmp = out_dir / f".embeddings.int8.{token}"
    sc_tmp = out_dir / f".embeddings.int8.scale.f16.{token}"
    meta_tmp = out_dir / f".embed_table_meta.json.{token}"
    try:
        i8_out.tofile(i8_tmp)
        sc_out.tofile(sc_tmp)
        meta_tmp.write_text(
            json.dumps(
                {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "embed_scale": 1.0,
                    "pre_scaled": True,
                }
            ),
            encoding="utf-8",
        )
        os.replace(i8_tmp, i8_path)
        os.replace(sc_tmp, sc_path)
        os.replace(meta_tmp, meta_path)
    finally:
        for tmp_f in (i8_tmp, sc_tmp, meta_tmp):
            tmp_f.unlink(missing_ok=True)
    return i8_path, sc_path, meta_path


class MmapTokenTable:
    """Clean-page read-only `int8 + fp16 scale` token embedding table (`134.7 MB`, `0 B` dirty/wired copy)."""

    def __init__(
        self,
        emb_i8: np.ndarray,
        scale_f16: np.ndarray,
        *,
        embed_scale: float = 1.0,
        io_lib: Any = None,
    ) -> None:
        self.emb_i8 = emb_i8
        self.scale_f16 = scale_f16
        self.vocab_size = int(emb_i8.shape[0])
        self.hidden_size = int(emb_i8.shape[1])
        self.embed_scale = float(embed_scale)
        self._io_lib = io_lib

    def bind_io_lib(self, io_lib: Any) -> None:
        self._io_lib = io_lib

    @classmethod
    def from_int8_files(
        cls,
        i8_path: str | Path,
        scale_path: str | Path,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        embed_scale: float = 1.0,
    ) -> "MmapTokenTable":
        p_i8 = Path(i8_path)
        p_sc = Path(scale_path)
        if not p_i8.is_file():
            raise FileNotFoundError(f"Missing int8 token table: {p_i8}")
        if not p_sc.is_file():
            raise FileNotFoundError(f"Missing fp16 scale table: {p_sc}")
        mm_i8 = np.memmap(p_i8, dtype=np.int8, mode="r", shape=(int(vocab_size), int(hidden_size)))
        mm_sc = np.memmap(p_sc, dtype=np.float16, mode="r", shape=(int(vocab_size),))
        return cls(mm_i8, mm_sc, embed_scale=float(embed_scale))

    def gather_scaled_f16(self, ids: np.ndarray | list[int]) -> np.ndarray:
        """Gather scaled `float16` embeddings of shape `(*ids.shape, hidden_size)` (`ids < 0` -> zero row)."""
        arr = np.ascontiguousarray(ids, dtype=np.int32)
        out_shape = (*arr.shape, self.hidden_size)
        m_rows = int(arr.size)
        if m_rows == 0:
            return np.empty(out_shape, dtype=np.float16)

        io_lib = self._io_lib
        if io_lib is not None and hasattr(io_lib, "emb2_gather_scaled_i8") and self.hidden_size == 512:
            out = np.empty(out_shape, dtype=np.float16)
            io_lib.emb2_gather_scaled_i8(
                self.emb_i8.ctypes.data,
                self.scale_f16.ctypes.data,
                float(self.embed_scale),
                arr.ctypes.data,
                self.vocab_size,
                m_rows,
                out.ctypes.data,
            )
            return out

        flat = arr.reshape(-1)
        valid = (flat >= 0) & (flat < self.vocab_size)
        safe_ids = np.where(valid, flat, 0)
        q = np.asarray(self.emb_i8[safe_ids], dtype=np.float32)
        sc = np.asarray(self.scale_f16[safe_ids], dtype=np.float32) * np.float32(self.embed_scale)
        vals = (q * sc[:, None]).astype(np.float16)
        if not np.all(valid):
            vals[~valid] = 0.0
        return np.ascontiguousarray(vals.reshape(out_shape), dtype=np.float16)

    def close(self) -> None:
        for attr in ("emb_i8", "scale_f16"):
            mm = getattr(self, attr, None)
            if isinstance(mm, np.memmap):
                backing = getattr(mm, "_mmap", None)
                if backing is not None and not getattr(backing, "closed", False):
                    backing.close()


_DEFAULT_IO_LIB: Any = None


def _resolve_default_io_lib() -> Any:
    global _DEFAULT_IO_LIB
    if _DEFAULT_IO_LIB is not None:
        return _DEFAULT_IO_LIB
    try:
        dylib = resolve_paths().io_dylib
        if dylib.is_file():
            import ctypes

            lib = ctypes.CDLL(str(dylib))
            if hasattr(lib, "emb2_gather_scaled_i8"):
                lib.emb2_gather_scaled_i8.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.emb2_gather_scaled_i8.restype = None
                _DEFAULT_IO_LIB = lib
    except Exception:
        pass
    return _DEFAULT_IO_LIB


def load_or_mmap_token_table(weights_npz_or_dir: str | Path, *, io_lib: Any = None) -> MmapTokenTable:
    """Open or materialize a clean-page `int8 + fp16 scale` mmap token table without duplicating `embed_scaled`."""
    eff_io_lib = io_lib if io_lib is not None else _resolve_default_io_lib()
    target = Path(weights_npz_or_dir)
    base_dir = target if target.is_dir() else target.parent

    # 1. Direct sidecar or OpenInfinity text/ layout
    for sub_dir, default_sc in ((base_dir, 1.0), (base_dir / "text", float(DEFAULT_HIDDEN_SIZE**0.5))):
        i8_p = sub_dir / "embeddings.int8"
        sc_p = sub_dir / "embeddings.int8.scale.f16"
        if not sc_p.is_file():
            sc_p = sub_dir / "scale.f16"
        if i8_p.is_file() and sc_p.is_file():
            meta_p = sub_dir / "embed_table_meta.json"
            if meta_p.is_file():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                v_sz = int(meta.get("vocab_size", DEFAULT_VOCAB_SIZE))
                h_sz = int(meta.get("hidden_size", DEFAULT_HIDDEN_SIZE))
                e_sc = float(meta.get("embed_scale", 1.0))
            else:
                h_sz = DEFAULT_HIDDEN_SIZE
                v_sz = int(i8_p.stat().st_size // h_sz)
                # Unscaled OpenInfinity scales have median ~0.0011; pre-scaled have ~0.025
                mm_probe = np.memmap(sc_p, dtype=np.float16, mode="r", shape=(min(v_sz, 1024),))
                med = float(np.median(np.abs(np.asarray(mm_probe, dtype=np.float32))))
                del mm_probe
                e_sc = float(DEFAULT_HIDDEN_SIZE**0.5) if med < 0.005 else default_sc
            tbl = MmapTokenTable.from_int8_files(i8_p, sc_p, vocab_size=v_sz, hidden_size=h_sz, embed_scale=e_sc)
            tbl.bind_io_lib(eff_io_lib)
            return tbl

    # 2. Materialize into user cache (zero pollution of sibling/read-only source checkouts)
    npz_path = target if target.is_file() else (base_dir / "embeddinggemma_2_f16.npz")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing weights archive or int8 token table at {target}")
    st = npz_path.stat()
    cache_dir = Path.home() / ".cache" / _PACKAGE_CACHE_NAME / "mmap_tables" / f"v1_{st.st_size}_{st.st_mtime_ns}"
    i8_c = cache_dir / "embeddings.int8"
    sc_c = cache_dir / "embeddings.int8.scale.f16"
    meta_c = cache_dir / "embed_table_meta.json"
    if not (i8_c.is_file() and sc_c.is_file() and meta_c.is_file()):
        with np.load(npz_path) as raw:
            emb_raw = raw["embed_tokens"]
        write_int8_token_table(cache_dir, emb_raw, pre_scale=float(emb_raw.shape[1] ** 0.5))
        del emb_raw
    meta = json.loads(meta_c.read_text(encoding="utf-8"))
    tbl = MmapTokenTable.from_int8_files(
        i8_c,
        sc_c,
        vocab_size=int(meta["vocab_size"]),
        hidden_size=int(meta["hidden_size"]),
        embed_scale=float(meta.get("embed_scale", 1.0)),
    )
    tbl.bind_io_lib(eff_io_lib)
    return tbl


def resolve_openinfinity_dir(
    model_dir: str | Path | None = None,
    *,
    repo_id: str = OPENINFINITY_REPO_ID,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local or HF-cached `OpenInfinity/embeddinggemma-2-coreml-int8` root directory."""
    if model_dir is not None:
        cand = Path(model_dir).expanduser().resolve()
        if not cand.exists():
            raise FileNotFoundError(f"OpenInfinity directory does not exist: {cand}")
        return cand
    env_dir = os.environ.get("FAST_EMBEDDINGGEMMA2_OPENINFINITY_DIR")
    if env_dir:
        cand = Path(env_dir).expanduser().resolve()
        if cand.exists():
            return cand

    hub_roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        hub_roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve())
    if os.environ.get("HF_HOME"):
        hub_roots.append(Path(os.environ["HF_HOME"]).expanduser().resolve() / "hub")
    hub_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_cache_name = f"models--{repo_id.replace('/', '--')}"
    for hub_root in hub_roots:
        repo_root = hub_root / repo_cache_name
        snapshots_dir = repo_root / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        ref_main = repo_root / "refs" / "main"
        candidates: list[Path] = []
        if ref_main.is_file():
            rev = ref_main.read_text(encoding="utf-8").strip()
            if rev:
                candidates.append(snapshots_dir / rev)
        candidates.extend(sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), reverse=True))
        for snap in candidates:
            if (snap / "text" / "EmbeddingGemma2Text.mlmodelc").exists() or (snap / "EmbeddingGemma2Text.mlmodelc").exists():
                return snap.resolve()

    try:
        return resolve_hf_model_dir(None, repo_id, local_files_only=local_files_only)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve {repo_id!r} (local_files_only={local_files_only}): {exc}"
        ) from exc


def ensure_prebuilt_artifacts(
    home: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_PREBUILT_HF_REPO,
    local_files_only: bool = False,
    include_ane: bool = True,
    force: bool = False,
    downloader: Any | None = None,
) -> RepoPaths:
    """Materialize prebuilt weights, tokenizer, and Core ML ANE/CPU-SME bundles from Hugging Face."""
    from fast_shared.hf import materialize_hf_snapshot

    p = resolve_paths(home)
    subdirs = ("weights", "tokenizer", "ane_models") if include_ane else ("weights", "tokenizer")
    materialize_hf_snapshot(
        repo_id,
        p.home / "artifacts",
        subdirs=subdirs,
        local_files_only=local_files_only,
        overwrite=force,
        downloader=downloader,
    )
    return p
