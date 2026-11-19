"""Pure-ANE `W8A16` typed-question scorer for `convaiinnovations/laya-typed-decisions`."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import threading
from typing import Any, Callable, Literal, Mapping
import warnings

from ._common import softmax_2, softmax_vec
from .paths import ensure_laya_dylib, resolve_laya_dir

_QTYPES = {"choice": 0, "score": 1, "noul": 2}

# Compute unit mode constants aligned 1:1 with `fast_shared::ComputeMode` & `laya_ane_runner.h`.
COMPUTE_MODE_CPU_AND_ANE = 0
COMPUTE_MODE_CPU_ONLY = 1
COMPUTE_MODE_ALL = 2


class LayaAneBackend:
    """Zero-GPU CoreML ANE `W8A16` runner (`S=256` & `S=512` buckets, `[1585, 0, 0]` ops on ANE)."""

    def __init__(
        self,
        laya_dir: Path | str | None = None,
        dylib_path: Path | str | None = None,
        *,
        pipe_depth: int = 4,
        compute_mode: int = COMPUTE_MODE_CPU_AND_ANE,
    ) -> None:
        import numpy as np
        from tokenizers import Tokenizer

        self._np = np
        self._exec_lock = threading.Lock()
        self.laya_dir = Path(laya_dir).expanduser().resolve() if laya_dir is not None else resolve_laya_dir()
        self.dylib_path = (
            Path(dylib_path).expanduser().resolve() if dylib_path is not None else ensure_laya_dylib()
        )
        self.pipe_depth = max(1, int(pipe_depth))
        self.compute_mode = int(compute_mode)

        cfg = json.loads((self.laya_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.temperature = [float(x) for x in cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {
            str(k): float(v) for k, v in cfg.get("temperature_by_options", {}).items()
        }
        self.max_len = 512
        self.head_max_len = 192

        tok_path = self.laya_dir / "tokenizer" / "tokenizer.json"
        self.tok = Tokenizer.from_file(str(tok_path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.cls_id = int(self.tok.token_to_id("[CLS]") or 50281)
        self.sep_id = int(self.tok.token_to_id("[SEP]") or 50282)
        self.pad_id = int(self.tok.token_to_id("[PAD]") or 50283)
        self.mask_id = int(self.tok.token_to_id("[MASK]") or 50284)
        self.mask_tok = "[MASK]"

        emb = np.load(self.laya_dir / "laya_typed_emb_f16.npz")
        self._word_f16 = np.ascontiguousarray(emb["word_f16"])
        self._type_f16 = np.ascontiguousarray(emb["type_f16"])
        H = int(self._word_f16.shape[1])
        vocab_sz = int(self._word_f16.shape[0])

        self._lib = ctypes.CDLL(str(self.dylib_path))
        self._lib.laya_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._lib.laya_ane_load.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.laya_ane_load.restype = ctypes.c_void_p
        self._lib.laya_ane_exec_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.laya_ane_exec_batch.restype = ctypes.c_double
        self._has_last_error = hasattr(self._lib, "laya_ane_last_error")
        if self._has_last_error:
            self._lib.laya_ane_last_error.argtypes = [ctypes.c_void_p]
            self._lib.laya_ane_last_error.restype = ctypes.c_char_p
        self._lib.laya_ane_free.argtypes = [ctypes.c_void_p]

        self._mc_s256 = self.laya_dir / "laya_typed_s256_w8.mlmodelc"
        self._mc_s512 = self.laya_dir / "laya_typed_s512_w8.mlmodelc"
        if not self._mc_s512.exists():
            raise FileNotFoundError(
                f"Missing ANE bundle {self._mc_s512}. Run 'decision-models setup-weights --build-laya-ane'."
            )

        self._h512 = self._lib.laya_ane_load(
            str(self._mc_s512).encode("utf-8"),
            H,
            512,
            self.pipe_depth,
            self.compute_mode,
            self._word_f16.ctypes.data,
            vocab_sz,
            self._type_f16.ctypes.data,
        )
        if not self._h512:
            raise RuntimeError(f"Failed to load ANE bundle {self._mc_s512}")

        self._h256 = None
        if self._mc_s256.exists():
            self._h256 = self._lib.laya_ane_load(
                str(self._mc_s256).encode("utf-8"),
                H,
                256,
                self.pipe_depth,
                self.compute_mode,
                self._word_f16.ctypes.data,
                vocab_sz,
                self._type_f16.ctypes.data,
            )

    def __enter__(self) -> "LayaAneBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def inspect_compute_plans(self) -> dict[str, tuple[int, int, int]]:
        """Return `{bucket: (ane_ops, gpu_ops, cpu_ops)}` from `MLComputePlan`."""
        plans: dict[str, tuple[int, int, int]] = {}
        for tag, mc in (("s256", self._mc_s256), ("s512", self._mc_s512)):
            if mc.exists():
                counts = (ctypes.c_int * 3)()
                self._lib.laya_ane_inspect_plan(str(mc).encode("utf-8"), self.compute_mode, counts)
                plans[tag] = (int(counts[0]), int(counts[1]), int(counts[2]))
        return plans

    def _encode_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def _resolve_temperature(self, qtype: int, k: int, temperature_override: float | None = None) -> float:
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

    def encode_item(
        self,
        state: Any,
        qdef: Mapping[str, Any],
        *,
        max_len: int | None = None,
        on_truncate: Literal["warn", "error", "ignore"] = "warn",
    ) -> dict[str, Any]:
        """Encode one typed question (`noul`, `choice`, `score`) + state into token ids & option markers.

        `Laya-Typed-421M` ANE buckets are fixed at `S <= 512` (`self.max_len`). When the encoded
        question + state exceeds `eff_max`, `on_truncate` controls behavior (`"warn"` emits a
        `UserWarning`, `"error"` raises `ValueError`, `"ignore"` truncates silently).
        """
        if on_truncate not in ("warn", "error", "ignore"):
            raise ValueError(f"Unsupported on_truncate={on_truncate!r}; expected 'warn', 'error', or 'ignore'.")
        eff_max = min(self.max_len, int(max_len)) if max_len is not None else self.max_len
        eff_head = min(self.head_max_len, 128) if eff_max <= 256 else self.head_max_len
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
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        st_ids_raw = self._encode_ids(state_str.replace(self.mask_tok, " "))
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

    def _exec_bucket(self, bucket_items: list[dict[str, Any]], S: int, handle: Any) -> Any:
        np = self._np
        N = len(bucket_items)
        ids_np = np.full((N, S), self.pad_id, dtype=np.int32)
        vl_np = np.zeros((N,), dtype=np.int32)
        qt_np = np.zeros((N,), dtype=np.int32)
        for i, it in enumerate(bucket_items):
            ln = min(S, len(it["ids"]))
            ids_np[i, :ln] = it["ids"][:ln]
            vl_np[i] = ln
            qt_np[i] = it["qtype"]
        out_pos = np.empty((N, S), dtype=np.float32)
        with self._exec_lock:
            dt = self._lib.laya_ane_exec_batch(
                handle,
                N,
                ids_np.ctypes.data,
                vl_np.ctypes.data,
                qt_np.ctypes.data,
                out_pos.ctypes.data,
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
        return out_pos

    def _score_buckets(
        self,
        encoded_items: list[dict[str, Any]],
        row_fn: Callable[[Any, dict[str, Any], int], Any],
        *,
        default_factory: Callable[[], Any],
    ) -> list[Any]:
        if not encoded_items:
            return []
        results = [default_factory() for _ in encoded_items]
        idx_256: list[int] = []
        idx_512: list[int] = []
        for idx, it in enumerate(encoded_items):
            if self._h256 is not None and len(it["ids"]) <= 256:
                idx_256.append(idx)
            else:
                idx_512.append(idx)
        for indices, S_lim, handle in ((idx_256, 256, self._h256), (idx_512, 512, self._h512)):
            if not indices or handle is None:
                continue
            out_mat = self._exec_bucket([encoded_items[i] for i in indices], S_lim, handle)
            for b_i, orig_i in enumerate(indices):
                results[orig_i] = row_fn(out_mat[b_i], encoded_items[orig_i], S_lim)
        return results

    def score_noul_batch(self, encoded_items: list[dict[str, Any]]) -> list[float]:
        """Execute ANE inference across `S=256` / `S=512` buckets and return `P(true)` per item."""
        temp = self._resolve_temperature(2, 2)

        def _row_p_true(row_arr: Any, item: dict[str, Any], _s_lim: int) -> float:
            mk = item["markers"]
            _, p1 = softmax_2(float(row_arr[mk[0]]), float(row_arr[mk[1]]), temp)
            return p1

        return self._score_buckets(encoded_items, _row_p_true, default_factory=lambda: 0.5)

    def score_choice_batch(
        self,
        encoded_items: list[dict[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        """Execute ANE `Choice(K)` / `Score(K)` inference and return calibrated option probabilities."""

        def _row_probs(row_arr: Any, item: dict[str, Any], S_lim: int) -> list[float]:
            valid_mk = [m for m in item["markers"] if m < S_lim]
            k = len(valid_mk)
            if k == 0:
                return []
            t_val = self._resolve_temperature(int(item["qtype"]), k, temperature_override)
            return softmax_vec([float(row_arr[m]) for m in valid_mk], t_val, min_temp=0.05)

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
        """Release native CoreML handles."""
        if getattr(self, "_h256", None):
            self._lib.laya_ane_free(self._h256)
            self._h256 = None
        if getattr(self, "_h512", None):
            self._lib.laya_ane_free(self._h512)
            self._h512 = None


__all__ = [
    "COMPUTE_MODE_ALL",
    "COMPUTE_MODE_CPU_AND_ANE",
    "COMPUTE_MODE_CPU_ONLY",
    "LayaAneBackend",
]
