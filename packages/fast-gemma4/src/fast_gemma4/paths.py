"""Filesystem and model artifact path resolution for `fast-gemma4`."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from fast_shared.hf import resolve_hf_model_dir
from fast_shared.paths import resolve_home_dir

DEFAULT_HF_REPO = "lmstudio-community/gemma-4-E2B-it-MLX-4bit"
QAT_E2B_REPO = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
QAT_E4B_REPO = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
QAT_MOBILE_E2B_REPO = "mlx-community/gemma-4-E2B-it-qat-mobile"
QAT_MOBILE_E4B_REPO = "mlx-community/gemma-4-E4B-it-qat-mobile"
MOE_26B_MXFP4_REPO = "mlx-community/gemma-4-26b-a4b-it-mxfp4"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_home(home: str | Path | None = None) -> Path:
    """Resolve the artifacts/data home directory for `fast-gemma4`.

    Priority:
      1. Explicit `home` argument.
      2. `FAST_GEMMA4_HOME` environment variable.
      3. Editable checkout root with staged artifacts (`ane_models`/`weights`/`models`/`tokenizer`/`hf_model`).
      4. `FAST_ARTIFACTS_ROOT` environment variable (`<root>/fast-gemma4` or `<root>`).
      5. Sibling checkout (`../fast-gemma4`) when imported from `fast-silicon/packages/fast-gemma4`.
      6. User cache directory (`$HOME/.cache/fast-gemma4`).
    """
    return resolve_home_dir(
        __file__,
        env_var="FAST_GEMMA4_HOME",
        cache_name="fast-gemma4",
        home=home,
        marker_dirs=(
            "artifacts/ane_models",
            "artifacts/weights",
            "artifacts/models",
            "artifacts/tokenizer",
            "artifacts/hf_model",
        ),
    )


def _model_dir_latest_weight_mtime_ns(resolved: Path) -> int:
    latest = 0
    cfg_p = resolved / "config.json"
    if cfg_p.is_file():
        latest = max(latest, int(cfg_p.stat().st_mtime_ns))
    for sf in sorted(resolved.glob("*.safetensors")):
        if sf.is_file():
            latest = max(latest, int(sf.stat().st_mtime_ns))
    return latest


def _is_local_ane_bundle_fresh(model_dir: Path, bundle_dir: Path, expected_key: str) -> bool:
    if not bundle_dir.exists():
        return False
    manifest_p = bundle_dir.parent / "ane_calibration_manifest.json"
    if manifest_p.is_file():
        try:
            meta = json.loads(manifest_p.read_text(encoding="utf-8"))
            bundles = meta.get("bundles")
            b_meta = bundles.get(bundle_dir.name) if isinstance(bundles, dict) else None
            rec_key = (
                b_meta.get("checkpoint_ane_key")
                if isinstance(b_meta, dict)
                else None
            ) or meta.get("checkpoint_ane_key")
            if rec_key and rec_key != expected_key:
                return False
        except Exception:
            pass
    latest_weight_ns = _model_dir_latest_weight_mtime_ns(model_dir)
    if latest_weight_ns <= 0:
        return True
    mil_p = bundle_dir / "model.mil"
    bundle_ns = int(mil_p.stat().st_mtime_ns) if mil_p.is_file() else int(bundle_dir.stat().st_mtime_ns)
    return bundle_ns >= latest_weight_ns


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
            cand = Path(model_dir).expanduser()
            if cand.exists():
                p = cand.resolve()
                if not (p / "config.json").exists():
                    raise FileNotFoundError(f"Missing config.json in model_dir={p}")
                return p
            if (
                isinstance(model_dir, str)
                and not cand.is_absolute()
                and not model_dir.startswith(("./", "../"))
                and model_dir.count("/") == 1
                and all(part.strip() for part in model_dir.split("/"))
            ):
                try:
                    return resolve_hf_model_dir(None, model_dir, local_files_only=local_files_only)
                except Exception:
                    pass
            p = cand.resolve()
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
        """Compute checkpoint-scoped ANE cache key incorporating path + weight/config fingerprint."""
        resolved = model_dir.expanduser().resolve()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.name).strip("_") or "model"
        h = hashlib.sha256()
        h.update(str(resolved).encode("utf-8"))
        # Include config.json & safetensors sizes + mtimes so in-place overwrites invalidate stale ANE bundles.
        cfg_p = resolved / "config.json"
        if cfg_p.is_file():
            st = cfg_p.stat()
            h.update(f"|config.json:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        for sf in sorted(resolved.glob("*.safetensors")):
            if sf.is_file():
                st = sf.stat()
                h.update(f"|{sf.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        prov_p = resolved / "provenance.json"
        if prov_p.is_file():
            try:
                prov = json.loads(prov_p.read_text(encoding="utf-8"))
                shas = prov.get("shard_sha256")
                if isinstance(shas, dict) and shas:
                    h.update(("|prov:" + json.dumps(shas, sort_keys=True)).encode("utf-8"))
            except Exception:
                st = prov_p.stat()
                h.update(f"|provenance.json:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        digest = h.hexdigest()[:12]
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
        exit_layer: int | None = None,
    ) -> Path:
        fname = f"gemma4_early{int(exit_layer)}_f16.npz" if exit_layer is not None else "gemma4_early14_f16.npz"
        if self._is_default_model_dir(model_dir, local_files_only=local_files_only):
            if exit_layer is None or int(exit_layer) == 14:
                return self.early14_weights_npz
            return self.early14_weights_npz.parent / fname
        assert model_dir is not None
        key = self.checkpoint_ane_key(Path(model_dir))
        return self.early14_weights_npz.parent / key / fname

    def resolve_ane_bundle(
        self,
        variant: str = "fp16",
        *,
        model_dir: str | Path | None = None,
        local_files_only: bool = True,
        exit_layer: int | None = None,
    ) -> Path | None:
        if exit_layer is not None and int(exit_layer) != 14:
            fname = f"gemma4_early{int(exit_layer)}_prefix_S256_{variant}.mlmodelc"
        else:
            fname = self.ane_fp16_bundle.name if variant == "fp16" else self.ane_w8_bundle.name
        if self._is_default_model_dir(model_dir, local_files_only=local_files_only):
            p = (self.ane_models_dir / fname) if exit_layer is not None else (
                self.ane_fp16_bundle if variant == "fp16" else self.ane_w8_bundle
            )
            return p if p.exists() else None
        assert model_dir is not None
        resolved = Path(model_dir).expanduser().resolve()
        ckpt_key = self.checkpoint_ane_key(resolved)
        local_cand = resolved / "ane_models" / fname
        if local_cand.exists() and _is_local_ane_bundle_fresh(resolved, local_cand, ckpt_key):
            return local_cand
        scoped_cand = self.ane_models_dir / ckpt_key / fname
        if scoped_cand.exists():
            return scoped_cand
        return None


def resolve_paths(home: str | Path | None = None) -> Gemma4Paths:
    data_root = resolve_home(home)
    pkg_root = _package_root()
    if home is not None or os.environ.get("FAST_GEMMA4_HOME"):
        build_root = data_root
        art = data_root / "artifacts"
    else:
        build_root = pkg_root if (pkg_root / "build").exists() or (pkg_root / "native").exists() else data_root
        has_direct_art = any(
            (data_root / d).is_dir() for d in ("ane_models", "weights", "models", "tokenizer", "hf_model")
        )
        art = (data_root / "artifacts") if ((data_root / "artifacts").is_dir() or data_root == pkg_root or not has_direct_art) else data_root
    ane_dir = art / "ane_models"
    return Gemma4Paths(
        home=build_root if (home is None and not os.environ.get("FAST_GEMMA4_HOME") and (build_root / "tests").is_dir() and data_root == pkg_root) else data_root,
        tokenizer_dir=art / "tokenizer",
        vocab_remap_npz=art / "vocab" / "vocab_remap_150786.npz",
        early14_weights_npz=art / "weights" / "gemma4_early14_f16.npz",
        ane_models_dir=ane_dir,
        ane_fp16_bundle=ane_dir / "gemma4_early14_prefix_S256_fp16.mlmodelc",
        ane_w8_bundle=ane_dir / "gemma4_early14_prefix_S256_w8.mlmodelc",
        lib_ane_prefix=build_root / "build" / "libgemma4_ane_prefix.dylib",
        ane_cli=build_root / "build" / "gemma4-ane-cli",
        local_hf_dir=art / "hf_model",
    )
