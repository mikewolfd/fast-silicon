"""Pure-ANE `W8A16` / zero-compile `NMTZ` typed-question scorer for `convaiinnovations/laya-typed-decisions`."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import struct
import threading
from typing import Any, Callable, Literal, Mapping, Sequence
import warnings

from ._common import softmax_2, softmax_vec
from .paths import (
    DEFAULT_LAYA_NMTZ_COREML_REPO,
    DEFAULT_LAYA_REPO,
    ensure_laya_dylib,
    resolve_laya_dir,
)

_QTYPES = {"choice": 0, "score": 1, "noul": 2}
_CANDIDATE_SEQ_LENS: tuple[int, ...] = (64, 128, 192, 256, 384, 512, 640, 1024)
_NMTZ_SEQ_LENS: tuple[int, ...] = (192, 384, 512, 640)

# Compute unit mode constants aligned 1:1 with `fast_shared::ComputeMode` & `laya_ane_runner.h`.
COMPUTE_MODE_CPU_AND_ANE = 0
COMPUTE_MODE_CPU_ONLY = 1
COMPUTE_MODE_ALL = 2


def _tok_id_or_default(tok: Any, piece: str, default: int) -> int:
    tid = tok.token_to_id(piece)
    return default if tid is None else int(tid)


def _discover_mf_metadata_buckets(mf_mc: Path) -> dict[int, tuple[Path, str]]:
    """Discover multifunction `s{seq}` entrypoints from `mf_mc/metadata.json` without CoreML plan probing."""
    meta_p = mf_mc / "metadata.json"
    if not meta_p.is_file():
        return {}
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cand_by_fn = {f"s{s_len}": s_len for s_len in _CANDIDATE_SEQ_LENS}
    found_fns: set[str] = set()

    def _walk(obj: Any, *, in_fn_scope: bool = False) -> None:
        if isinstance(obj, str):
            if obj in cand_by_fn and in_fn_scope:
                found_fns.add(obj)
            return
        if isinstance(obj, list):
            for elem in obj:
                _walk(elem, in_fn_scope=in_fn_scope)
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(key, str) and key in cand_by_fn and in_fn_scope:
                    found_fns.add(key)
                if key in ("functions", "function_names", "functionNames"):
                    _walk(val, in_fn_scope=True)
                elif key in ("name", "functionName", "function_name") and isinstance(val, str):
                    if val in cand_by_fn:
                        found_fns.add(val)
                elif isinstance(val, (dict, list)):
                    _walk(val, in_fn_scope=in_fn_scope)

    if isinstance(meta, (list, dict)):
        top_in_fn = isinstance(meta, list) and all(isinstance(x, str) for x in meta)
        _walk(meta, in_fn_scope=top_in_fn)
        if isinstance(meta, dict):
            for k in meta:
                if k in cand_by_fn:
                    found_fns.add(k)

    return {
        s_len: (mf_mc, f"s{s_len}")
        for s_len in _CANDIDATE_SEQ_LENS
        if f"s{s_len}" in found_fns
    }


def _read_safetensors_fp16_pair_cpu(sf_path: Path) -> tuple[Any, Any]:
    """Read `tok_embeddings.weight` & `type_emb.weight` as contiguous FP16 NumPy arrays with zero Metal/GPU alloc."""
    import numpy as np

    with open(sf_path, "rb") as fp:
        raw_h = fp.read(8)
        if len(raw_h) < 8:
            raise ValueError(f"Truncated safetensors file: {sf_path}")
        h_len = struct.unpack("<Q", raw_h)[0]
        hdr = json.loads(fp.read(h_len).decode("utf-8"))
        data_off = 8 + h_len

        def _read_f16(key: str) -> Any:
            info = hdr[key]
            dt_tag = str(info["dtype"]).upper()
            shape = tuple(int(d) for d in info["shape"])
            start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            fp.seek(data_off + start)
            buf = fp.read(end - start)
            if dt_tag == "F16":
                return np.ascontiguousarray(np.frombuffer(buf, dtype="<f2").reshape(shape).copy(), dtype=np.float16)
            if dt_tag == "F32":
                return np.ascontiguousarray(np.frombuffer(buf, dtype="<f4").reshape(shape).astype(np.float16))
            if dt_tag == "BF16":
                u16 = np.frombuffer(buf, dtype="<u2")
                f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
                return np.ascontiguousarray(f32.astype(np.float16))
            raise ValueError(f"Unsupported embedding dtype {dt_tag!r} in {sf_path}")

        return _read_f16("encoder.embeddings.tok_embeddings.weight"), _read_f16("type_emb.weight")


class LayaTokenizerHelper:
    """Zero-CoreML tokenizer, question encoder, and temperature resolver for `Laya-Typed-421M`."""

    def __init__(
        self,
        tokenizer_json: str | Path,
        *,
        rl_agent_config: Mapping[str, Any] | str | Path | None = None,
        laya_dir: str | Path | None = None,
        max_len: int = 512,
        head_max_len: int | None = None,
        head_max_len_cap: int | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        src = Path(tokenizer_json).expanduser().resolve()
        if src.is_dir():
            resolved_dir: Path | None = src
            cand_nested = src / "tokenizer" / "tokenizer.json"
            tok_path = cand_nested if cand_nested.is_file() else (src / "tokenizer.json")
        else:
            resolved_dir = Path(laya_dir).expanduser().resolve() if laya_dir is not None else None
            tok_path = src

        if not tok_path.is_file():
            raise FileNotFoundError(f"Missing Laya tokenizer file: {tok_path}")

        cfg: Mapping[str, Any]
        if isinstance(rl_agent_config, Mapping):
            cfg = rl_agent_config
        elif rl_agent_config is not None:
            cfg = json.loads(Path(rl_agent_config).expanduser().resolve().read_text(encoding="utf-8"))
        elif resolved_dir is not None and (resolved_dir / "rl_agent_config.json").is_file():
            cfg = json.loads((resolved_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
        else:
            cfg = {}

        self.laya_dir = resolved_dir
        self.temperature = [float(x) for x in cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {
            str(k): float(v) for k, v in cfg.get("temperature_by_options", {}).items()
        }
        self.max_len = int(max_len)
        resolved_head = (
            int(head_max_len)
            if head_max_len is not None
            else self.resolve_head_max_len(cfg, self.max_len)
        )
        if head_max_len_cap is not None:
            resolved_head = min(resolved_head, int(head_max_len_cap))
        self.head_max_len = resolved_head

        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.cls_id = _tok_id_or_default(self.tok, "[CLS]", 50281)
        self.sep_id = _tok_id_or_default(self.tok, "[SEP]", 50282)
        self.pad_id = _tok_id_or_default(self.tok, "[PAD]", 50283)
        self.mask_id = _tok_id_or_default(self.tok, "[MASK]", 50284)
        self.mask_tok = "[MASK]"

    @classmethod
    def from_laya_dir(
        cls,
        laya_dir: str | Path,
        *,
        max_len: int = 512,
        head_max_len: int | None = None,
        head_max_len_cap: int | None = None,
    ) -> "LayaTokenizerHelper":
        """Build a tokenizer-only helper from a Laya snapshot/artifact directory (zero CoreML load)."""
        base = Path(laya_dir).expanduser().resolve()
        tok_path = base / "tokenizer" / "tokenizer.json"
        if not tok_path.is_file():
            tok_path = base / "tokenizer.json"
        cfg_path = base / "rl_agent_config.json"
        return cls(
            tok_path,
            rl_agent_config=cfg_path if cfg_path.is_file() else None,
            laya_dir=base,
            max_len=max_len,
            head_max_len=head_max_len,
            head_max_len_cap=head_max_len_cap,
        )

    def _encode_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def encode_state(self, state: Any) -> list[int]:
        """Serialize and tokenize a state once for reuse across questions via `state_ids`."""
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return self._encode_ids(state_str.replace(self.mask_tok, " "))

    def resolve_temperature(self, qtype: int, k: int, temperature_override: float | None = None) -> float:
        if temperature_override is not None:
            return float(temperature_override)
        if qtype == 2:
            return float(self.temperature_by_options.get("noul:2", self.temperature[2]))
        prefix = "score" if qtype == 1 else "choice"
        default_t = self.temperature[1] if qtype == 1 else self.temperature[0]
        if k == 2:
            return float(self.temperature_by_options.get(f"{prefix}:2", default_t))
        if k <= 5:
            return float(self.temperature_by_options.get(f"{prefix}:3-5", default_t))
        if k <= 10:
            return float(self.temperature_by_options.get(f"{prefix}:6-10", default_t))
        return float(self.temperature_by_options.get(f"{prefix}:11+", 1.0))

    def _resolve_temperature(self, qtype: int, k: int, temperature_override: float | None = None) -> float:
        return self.resolve_temperature(qtype, k, temperature_override)

    @staticmethod
    def resolve_head_max_len(cfg: Mapping[str, Any], max_len: int) -> int:
        raw = int(cfg.get("head_max_len", 192))
        cap = min(256, max(128, int(max_len) - 128))
        return max(128, min(cap, raw))

    @staticmethod
    def _resolve_head_max_len(cfg: Mapping[str, Any], max_len: int) -> int:
        return LayaTokenizerHelper.resolve_head_max_len(cfg, max_len)

    def encode_item(
        self,
        state: Any,
        qdef: Mapping[str, Any],
        *,
        max_len: int | None = None,
        on_truncate: Literal["warn", "error", "ignore"] = "warn",
        state_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Encode one typed question (`noul`, `choice`, `score`) + state into token ids & option markers."""
        if on_truncate not in ("warn", "error", "ignore"):
            raise ValueError(f"Unsupported on_truncate={on_truncate!r}; expected 'warn', 'error', or 'ignore'.")
        eff_max = min(self.max_len, int(max_len)) if max_len is not None else self.max_len
        eff_head = min(self.head_max_len, max(64, eff_max - 128))
        t = str(qdef["type"])
        if t not in _QTYPES:
            raise ValueError(f"Unsupported question type={t!r}; expected one of {tuple(_QTYPES)}.")
        crit = qdef.get("criteria")
        if t == "choice":
            items = [(c, None) for c in crit] if isinstance(crit, list) else list((crit or {}).items())
            opts = [k if not v else f"{k}: {v}" for k, v in items]
        elif t == "score":
            opts = [f"level {i}: {c}" for i, c in enumerate(crit or ())]
        else:
            c_map = crit if isinstance(crit, Mapping) else {}
            opts = [
                "false: " + str(c_map.get("false") or "no, the statement does not hold"),
                "true: " + str(c_map.get("true") or "yes, the statement holds"),
            ]
        ins_raw = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
        ins = str(ins_raw).replace(self.mask_tok, " ")
        head_ids_raw = self._encode_ids(f"{t} question: {ins}")
        opt_cap = 32 if eff_max <= 256 else 48
        opt_ids: list[list[int]] = []
        truncated_parts: list[str] = []
        for opt_s in opts:
            raw_opt = self._encode_ids(" " + opt_s.replace(self.mask_tok, " "))
            if len(raw_opt) > opt_cap:
                truncated_parts.append(f"option({len(raw_opt)}>{opt_cap})")
            opt_ids.append([self.mask_id] + raw_opt[:opt_cap])
        opt_budget = eff_head - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (eff_head - 16) // max(1, len(opt_ids)))
            if any(len(o) > per for o in opt_ids):
                truncated_parts.append(f"options_budget(per={per})")
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = eff_head - sum(len(o) for o in opt_ids)
        head_lim = max(8, opt_budget)
        if len(head_ids_raw) > head_lim:
            truncated_parts.append(f"instructions({len(head_ids_raw)}>{head_lim})")
        head_ids = head_ids_raw[:head_lim]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers: list[int] = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, eff_max - len(ids) - 1)
        st_ids_raw = list(state_ids) if state_ids is not None else self.encode_state(state)
        if len(st_ids_raw) > room:
            truncated_parts.append(f"state({len(st_ids_raw)}>{room})")
        st_ids = st_ids_raw[:room]
        total_untruncated = len(ids) + len(st_ids_raw) + 1
        ids = (ids + st_ids + [self.sep_id])[:eff_max]
        truncated = bool(truncated_parts) or (total_untruncated > eff_max)
        if truncated and on_truncate != "ignore":
            msg = (
                f"LayaAneBackend.encode_item truncated input to eff_max={eff_max} tokens "
                f"(untruncated_total={total_untruncated}, details={', '.join(truncated_parts) or 'max_len'}). "
                f"Pass on_truncate='ignore' to silence or use Kev08bMlxBackend for >512 tokens."
            )
            if on_truncate == "error":
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        return {
            "ids": ids,
            "markers": markers,
            "qtype": _QTYPES[t],
            "k": len(opts),
            "truncated": truncated,
            "original_state_tokens": len(st_ids_raw),
        }

    def encode_items_for_state(
        self,
        state: Any,
        qdefs: Sequence[Mapping[str, Any]],
        *,
        max_len: int | None = None,
        on_truncate: Literal["warn", "error", "ignore"] = "warn",
    ) -> list[dict[str, Any]]:
        """Encode multiple questions against a shared `state` while tokenizing `state` only once."""
        if not qdefs:
            return []
        st_ids = self.encode_state(state)
        return [
            self.encode_item(
                state,
                qd,
                max_len=max_len,
                on_truncate=on_truncate,
                state_ids=st_ids,
            )
            for qd in qdefs
        ]


class LayaAneBackend(LayaTokenizerHelper):
    """Zero-GPU CoreML ANE runner supporting 6-bucket multifunction W8, legacy 256/512 W8, & prebuilt `NMTZ` CoreML."""

    @classmethod
    def from_tokenizer_only(
        cls,
        tokenizer_json: str | Path,
        *,
        rl_agent_config: Mapping[str, Any] | str | Path | None = None,
        laya_dir: str | Path | None = None,
        max_len: int = 512,
        head_max_len: int | None = None,
        head_max_len_cap: int | None = None,
    ) -> LayaTokenizerHelper:
        """Return a `LayaTokenizerHelper` with zero CoreML `.mlmodelc` or native dylib loading."""
        return LayaTokenizerHelper(
            tokenizer_json,
            rl_agent_config=rl_agent_config,
            laya_dir=laya_dir,
            max_len=max_len,
            head_max_len=head_max_len,
            head_max_len_cap=head_max_len_cap,
        )

    def _bind_native_lib(self) -> None:
        lib = ctypes.CDLL(str(self.dylib_path))
        lib.laya_ane_compile_mlpackage.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.laya_ane_compile_mlpackage.restype = ctypes.c_int
        lib.laya_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.laya_ane_inspect_plan_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.laya_ane_load_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.laya_ane_load_fn.restype = ctypes.c_void_p
        lib.laya_ane_max_markers.argtypes = [ctypes.c_void_p]
        lib.laya_ane_max_markers.restype = ctypes.c_int
        lib.laya_ane_exec_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.laya_ane_exec_batch.restype = ctypes.c_double
        lib.laya_ane_exec_markers_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.laya_ane_exec_markers_batch.restype = ctypes.c_double
        self._has_last_error = hasattr(lib, "laya_ane_last_error")
        if self._has_last_error:
            lib.laya_ane_last_error.argtypes = [ctypes.c_void_p]
            lib.laya_ane_last_error.restype = ctypes.c_char_p
        lib.laya_ane_free.argtypes = [ctypes.c_void_p]
        self._lib = lib

    def __init__(
        self,
        laya_dir: Path | str | None = None,
        dylib_path: Path | str | None = None,
        *,
        pipe_depth: int = 4,
        compute_mode: int = COMPUTE_MODE_CPU_AND_ANE,
        auto_prebuilt: bool = True,
        local_files_only: bool = False,
        nmtz_repo_id: str = DEFAULT_LAYA_NMTZ_COREML_REPO,
        laya_src_repo_id: str = DEFAULT_LAYA_REPO,
        max_len: int = 512,
        lazy_load: bool = True,
    ) -> None:
        import os
        import numpy as np

        self._np = np
        self._exec_lock = threading.Lock()
        self._handles: dict[int, Any] = {}
        self._bucket_sources: dict[int, tuple[Path, str]] = {}
        eff_offline = bool(local_files_only) or (
            os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        )
        resolved_dir = Path(laya_dir).expanduser().resolve() if laya_dir is not None else resolve_laya_dir()
        self.dylib_path = (
            Path(dylib_path).expanduser().resolve() if dylib_path is not None else ensure_laya_dylib()
        )
        self.pipe_depth = max(1, int(pipe_depth))
        self.compute_mode = int(compute_mode)
        self._bind_native_lib()

        has_local_mf = (resolved_dir / "laya_typed_w8.mlmodelc").is_dir()
        has_local_s512 = (resolved_dir / "laya_typed_s512_w8.mlmodelc").exists()
        use_nmtz = False

        if has_local_mf or has_local_s512:
            cfg = json.loads((resolved_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
            enc_cfg_p = resolved_dir / "encoder_config.json"
            enc_cfg = json.loads(enc_cfg_p.read_text(encoding="utf-8")) if enc_cfg_p.is_file() else {}
            local_half = int(enc_cfg.get("local_attention", 128)) // 2
            tok_path = resolved_dir / "tokenizer" / "tokenizer.json"
            super().__init__(tok_path, rl_agent_config=cfg, laya_dir=resolved_dir, max_len=max_len)

            emb = np.load(resolved_dir / "laya_typed_emb_f16.npz")
            self._word_f16 = np.ascontiguousarray(emb["word_f16"])
            self._type_f16 = np.ascontiguousarray(emb["type_f16"])
            self.artifact_source = "local_multifunction_w8" if has_local_mf else "local_per_bucket_w8"
        elif auto_prebuilt:
            import shutil
            from fast_shared.hf import resolve_hf_model_dir

            snap_nmtz: Path | None = None
            nmtz_hub_snaps = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / f"models--{nmtz_repo_id.replace('/', '--')}"
                / "snapshots"
            )
            if nmtz_hub_snaps.is_dir():
                for cand_s in sorted(nmtz_hub_snaps.iterdir()):
                    if (cand_s / "L512" / "model.mlpackage").is_dir():
                        snap_nmtz = cand_s
            if snap_nmtz is None:
                if eff_offline:
                    snap_nmtz = resolve_hf_model_dir(None, nmtz_repo_id, local_files_only=True)
                else:
                    try:
                        snap_nmtz = resolve_hf_model_dir(None, nmtz_repo_id, local_files_only=True)
                    except Exception:
                        snap_nmtz = resolve_hf_model_dir(None, nmtz_repo_id, local_files_only=False)

            if eff_offline:
                snap_src = resolve_hf_model_dir(None, laya_src_repo_id, local_files_only=True)
            else:
                try:
                    snap_src = resolve_hf_model_dir(None, laya_src_repo_id, local_files_only=True)
                except Exception:
                    snap_src = resolve_hf_model_dir(None, laya_src_repo_id, local_files_only=False)

            cfg = json.loads((snap_src / "rl_agent_config.json").read_text(encoding="utf-8"))
            enc_cfg = json.loads((snap_src / "encoder" / "config.json").read_text(encoding="utf-8"))
            local_half = int(enc_cfg.get("local_attention", 128)) // 2
            tok_path = snap_src / "tokenizer" / "tokenizer.json"
            super().__init__(tok_path, rl_agent_config=cfg, laya_dir=snap_src, max_len=max_len)

            self._word_f16, self._type_f16 = _read_safetensors_fp16_pair_cpu(snap_src / "model.safetensors")
            self.artifact_source = "nmtz_prebuilt_ane"
            use_nmtz = True
        else:
            raise FileNotFoundError(
                f"Missing ANE bundle {resolved_dir / 'laya_typed_s512_w8.mlmodelc'}. "
                "Run 'decision-models setup-weights --build-laya-ane'."
            )

        H = int(self._word_f16.shape[1])
        vocab_sz = int(self._word_f16.shape[0])
        self._load_dims: tuple[int, int, int] = (H, local_half, vocab_sz)

        try:
            if use_nmtz:
                import os
                import shutil

                repo_slug = nmtz_repo_id.replace("/", "--")
                shared_mc_dir = (
                    Path.home() / ".cache" / "fast-silicon" / "coreml_compiled" / repo_slug / snap_nmtz.name
                )
                cache_mc_dir = (
                    Path.home() / ".cache" / "fast-decision-models" / "nmtz_compiled" / snap_nmtz.name
                )
                cache_mc_dir.mkdir(parents=True, exist_ok=True)
                for s_len in _NMTZ_SEQ_LENS:
                    if s_len > max(512, self.max_len):
                        continue
                    pkg_p = snap_nmtz / f"L{s_len}" / "model.mlpackage"
                    if not pkg_p.is_dir():
                        continue
                    shared_mc = shared_mc_dir / f"L{s_len}__model.mlmodelc"
                    if shared_mc.is_dir() and (
                        (shared_mc / "model.mil").is_file() or (shared_mc / "coremldata.bin").is_file()
                    ):
                        dst_mc = shared_mc
                    else:
                        dst_mc = cache_mc_dir / f"L{s_len}.mlmodelc"
                    if not (dst_mc / "model.mil").is_file() and not (dst_mc / "coremldata.bin").is_file():
                        # Resolve HF Hub blob symlinks via hardlinks before CoreML compilation
                        mat_pkg = cache_mc_dir / f".stage_L{s_len}_{os.getpid()}.mlpackage"
                        if mat_pkg.exists():
                            shutil.rmtree(mat_pkg, ignore_errors=True)
                        try:
                            for root_d, _, files_d in os.walk(pkg_p):
                                rel_d = Path(root_d).relative_to(pkg_p)
                                (mat_pkg / rel_d).mkdir(parents=True, exist_ok=True)
                                for fn_d in files_d:
                                    s_f = (Path(root_d) / fn_d).resolve()
                                    d_f = mat_pkg / rel_d / fn_d
                                    try:
                                        os.link(s_f, d_f)
                                    except OSError:
                                        shutil.copy2(s_f, d_f)
                            rc = self._lib.laya_ane_compile_mlpackage(
                                str(mat_pkg).encode("utf-8"), str(dst_mc).encode("utf-8")
                            )
                        finally:
                            shutil.rmtree(mat_pkg, ignore_errors=True)
                        if rc != 0:
                            raise RuntimeError(f"Failed to compile NMTZ CoreML package {pkg_p}")
                    self._bucket_sources[s_len] = (dst_mc, "main")
            elif has_local_mf:
                mf_mc = (resolved_dir / "laya_typed_w8.mlmodelc").resolve()
                # Discover buckets lazily via symlinks first, then metadata.json, probing MLComputePlan only as fallback.
                for s_len in _CANDIDATE_SEQ_LENS:
                    link_p = resolved_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
                    if link_p.exists() and link_p.resolve() == mf_mc:
                        self._bucket_sources[s_len] = (mf_mc, f"s{s_len}")
                if not self._bucket_sources:
                    self._bucket_sources.update(_discover_mf_metadata_buckets(mf_mc))
                if not self._bucket_sources:
                    for s_len in _CANDIDATE_SEQ_LENS:
                        fn_str = f"s{s_len}"
                        counts = (ctypes.c_int * 3)()
                        self._lib.laya_ane_inspect_plan_fn(
                            str(mf_mc).encode("utf-8"), fn_str.encode("utf-8"), self.compute_mode, counts
                        )
                        if int(counts[0]) + int(counts[1]) + int(counts[2]) > 0:
                            self._bucket_sources[s_len] = (mf_mc, fn_str)
            else:
                for s_len in _CANDIDATE_SEQ_LENS:
                    raw_mc = resolved_dir / f"laya_typed_s{s_len}_w8.mlmodelc"
                    if not raw_mc.exists():
                        continue
                    mc_res = raw_mc.resolve()
                    fn_str = f"s{s_len}" if mc_res.name == "laya_typed_w8.mlmodelc" else "main"
                    self._bucket_sources[s_len] = (mc_res, fn_str)

            if not self._bucket_sources:
                raise RuntimeError(f"Failed to load any Laya ANE CoreML buckets from {resolved_dir}")

            self.seq_buckets: tuple[int, ...] = tuple(sorted(self._bucket_sources.keys()))
            if not lazy_load:
                for s_len in self.seq_buckets:
                    try:
                        self._ensure_bucket_handle(s_len)
                    except RuntimeError:
                        pass
                if not self._handles:
                    raise RuntimeError(f"Failed to load any Laya ANE CoreML buckets from {resolved_dir}")
                self.seq_buckets = tuple(sorted(self._handles.keys()))
                self._bucket_sources = {s: self._bucket_sources[s] for s in self.seq_buckets}
        except Exception:
            self.close()
            raise

    @property
    def _h256(self) -> Any:
        handles = getattr(self, "_handles", None)
        return handles.get(256) if handles else None

    @property
    def _h512(self) -> Any:
        handles = getattr(self, "_handles", None)
        return handles.get(512) if handles else None

    def _ensure_bucket_handle(self, s_len: int) -> Any:
        """Return cached native handle for bucket `s_len`, lazily loading it under `_exec_lock` on first use."""
        handle = self._handles.get(s_len)
        if handle:
            return handle
        with self._exec_lock:
            handle = self._handles.get(s_len)
            if handle:
                return handle
            src = self._bucket_sources.get(s_len)
            if src is None:
                raise RuntimeError(f"Missing Laya ANE CoreML bucket source for s{s_len}")
            mc_path, fn_str = src
            H, local_half, vocab_sz = self._load_dims
            h_ptr = self._lib.laya_ane_load_fn(
                str(mc_path).encode("utf-8"),
                fn_str.encode("utf-8"),
                H,
                int(s_len),
                local_half,
                self.pipe_depth,
                self.compute_mode,
                self._word_f16.ctypes.data,
                vocab_sz,
                self._type_f16.ctypes.data,
                int(self.pad_id),
            )
            if not h_ptr:
                raise RuntimeError(f"Failed to load Laya ANE CoreML bucket s{s_len} from {mc_path} ({fn_str})")
            self._handles[s_len] = h_ptr
            return h_ptr

    def __enter__(self) -> "LayaAneBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def inspect_compute_plans(self) -> dict[str, tuple[int, int, int]]:
        """Return `{bucket: (ane_ops, gpu_ops, cpu_ops)}` from `MLComputePlan` across all loaded buckets."""
        plans: dict[str, tuple[int, int, int]] = {}
        for s_len in self.seq_buckets:
            mc_path, fn_name = self._bucket_sources[s_len]
            if mc_path.exists():
                counts = (ctypes.c_int * 3)()
                self._lib.laya_ane_inspect_plan_fn(
                    str(mc_path).encode("utf-8"),
                    fn_name.encode("utf-8"),
                    self.compute_mode,
                    counts,
                )
                plans[f"s{s_len}"] = (int(counts[0]), int(counts[1]), int(counts[2]))
        return plans

    def _select_bucket(self, seq_len: int) -> int:
        for b in self.seq_buckets:
            if seq_len <= b:
                return b
        return self.seq_buckets[-1]

    def _exec_bucket_markers(
        self,
        bucket_items: list[dict[str, Any]],
        S: int,
        handle: Any,
        *,
        qtype_override: int | None = None,
    ) -> list[list[float]]:
        np = self._np
        N = len(bucket_items)
        max_m_hw = int(self._lib.laya_ane_max_markers(handle))
        chunk_m = max_m_hw if max_m_hw > 0 else max(32, max((len(it["markers"]) for it in bucket_items), default=1))

        ids_np = np.full((N, S), self.pad_id, dtype=np.int32)
        vl_np = np.zeros((N,), dtype=np.int32)
        qt_np = np.zeros((N,), dtype=np.int32)
        valid_markers_per_item: list[list[int]] = []
        for i, it in enumerate(bucket_items):
            ln = min(S, len(it["ids"]))
            ids_np[i, :ln] = it["ids"][:ln]
            vl_np[i] = ln
            qt_np[i] = int(qtype_override) if qtype_override is not None else int(it["qtype"])
            valid_markers_per_item.append([int(m) for m in it["markers"] if int(m) < S])

        max_k_all = max((len(vm) for vm in valid_markers_per_item), default=0)
        if max_k_all == 0:
            return [[] for _ in bucket_items]

        if max_k_all <= chunk_m:
            mp_np = np.zeros((N, chunk_m), dtype=np.int32)
            mc_np = np.zeros((N,), dtype=np.int32)
            for i, vm in enumerate(valid_markers_per_item):
                if vm:
                    mp_np[i, : len(vm)] = vm
                    mc_np[i] = len(vm)
            out_m = np.zeros((N, chunk_m), dtype=np.float32)
            with self._exec_lock:
                dt = self._lib.laya_ane_exec_markers_batch(
                    handle,
                    N,
                    ids_np.ctypes.data,
                    vl_np.ctypes.data,
                    qt_np.ctypes.data,
                    mp_np.ctypes.data,
                    mc_np.ctypes.data,
                    chunk_m,
                    out_m.ctypes.data,
                    1 if N > 1 else 0,
                )
                if dt < 0.0:
                    err_msg = ""
                    if self._has_last_error:
                        raw = self._lib.laya_ane_last_error(handle)
                        if raw:
                            err_msg = raw.decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Laya ANE CoreML prediction failed (bucket S={S}, N={N}): {err_msg or 'unknown error'}"
                    )
            return [[float(x) for x in out_m[i, : len(valid_markers_per_item[i])].tolist()] for i in range(N)]

        # Rare >32-option fallback: slice option markers into `chunk_m`-wide passes without dropping options
        acc_logits: list[list[float]] = [[] for _ in range(N)]
        for m0 in range(0, max_k_all, chunk_m):
            active = [i for i, vm in enumerate(valid_markers_per_item) if len(vm) > m0]
            n_act = len(active)
            sub_ids = np.ascontiguousarray(ids_np[active])
            sub_vl = np.ascontiguousarray(vl_np[active])
            sub_qt = np.ascontiguousarray(qt_np[active])
            mp_np = np.zeros((n_act, chunk_m), dtype=np.int32)
            mc_np = np.zeros((n_act,), dtype=np.int32)
            for a_i, orig_i in enumerate(active):
                sl = valid_markers_per_item[orig_i][m0 : m0 + chunk_m]
                mp_np[a_i, : len(sl)] = sl
                mc_np[a_i] = len(sl)
            out_m = np.zeros((n_act, chunk_m), dtype=np.float32)
            with self._exec_lock:
                dt = self._lib.laya_ane_exec_markers_batch(
                    handle,
                    n_act,
                    sub_ids.ctypes.data,
                    sub_vl.ctypes.data,
                    sub_qt.ctypes.data,
                    mp_np.ctypes.data,
                    mc_np.ctypes.data,
                    chunk_m,
                    out_m.ctypes.data,
                    1 if n_act > 1 else 0,
                )
                if dt < 0.0:
                    raise RuntimeError(f"Laya ANE multi-chunk marker pass failed (bucket S={S})")
            for a_i, orig_i in enumerate(active):
                cnt = int(mc_np[a_i])
                acc_logits[orig_i].extend(float(x) for x in out_m[a_i, :cnt].tolist())
        return acc_logits

    def _score_buckets(
        self,
        encoded_items: list[dict[str, Any]],
        row_fn: Callable[[list[float], dict[str, Any]], Any],
        *,
        default_factory: Callable[[], Any],
        qtype_override: int | None = None,
    ) -> list[Any]:
        if not encoded_items:
            return []
        results = [default_factory() for _ in encoded_items]
        by_bucket: dict[int, list[int]] = {s: [] for s in self.seq_buckets}
        for idx, it in enumerate(encoded_items):
            b_len = self._select_bucket(len(it["ids"]))
            by_bucket[b_len].append(idx)

        for s_lim in self.seq_buckets:
            indices = by_bucket[s_lim]
            if not indices:
                continue
            handle = self._ensure_bucket_handle(s_lim)
            raw_marker_logits = self._exec_bucket_markers(
                [encoded_items[i] for i in indices],
                s_lim,
                handle,
                qtype_override=qtype_override,
            )
            for b_i, orig_i in enumerate(indices):
                results[orig_i] = row_fn(raw_marker_logits[b_i], encoded_items[orig_i])
        return results

    def score_noul_batch(self, encoded_items: list[dict[str, Any]]) -> list[float]:
        """Execute ANE inference across tightest available buckets and return `P(true)` per item."""
        temp = self._resolve_temperature(2, 2)

        def _row_p_true(marker_logits: list[float], _item: dict[str, Any]) -> float:
            if len(marker_logits) < 2:
                return 0.5
            _, p1 = softmax_2(marker_logits[0], marker_logits[1], temp)
            return p1

        return self._score_buckets(
            encoded_items,
            _row_p_true,
            default_factory=lambda: 0.5,
            qtype_override=2,
        )

    def score_choice_batch(
        self,
        encoded_items: list[dict[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        """Execute ANE `Choice(K)` / `Score(K)` inference and return calibrated option probabilities."""

        def _row_probs(marker_logits: list[float], item: dict[str, Any]) -> list[float]:
            k = len(marker_logits)
            if k == 0:
                return []
            t_val = self._resolve_temperature(int(item["qtype"]), k, temperature_override)
            return softmax_vec(marker_logits, t_val, min_temp=0.05)

        return self._score_buckets(encoded_items, _row_probs, default_factory=list)

    def score_options_batch(
        self,
        encoded_items: list[dict[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        """Alias for `score_choice_batch` supporting `noul`, `choice`, and `score` uniformly."""
        return self.score_choice_batch(encoded_items, temperature_override=temperature_override)

    def close(self) -> None:
        """Release all native CoreML bucket handles."""
        handles = getattr(self, "_handles", None)
        if handles:
            for s_len in list(handles.keys()):
                h_ptr = handles.pop(s_len, None)
                if h_ptr:
                    self._lib.laya_ane_free(h_ptr)


__all__ = [
    "COMPUTE_MODE_ALL",
    "COMPUTE_MODE_CPU_AND_ANE",
    "COMPUTE_MODE_CPU_ONLY",
    "LayaAneBackend",
    "LayaTokenizerHelper",
]
