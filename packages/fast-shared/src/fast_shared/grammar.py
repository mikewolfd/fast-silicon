"""Shared `llguidance` structured JSON grammar compiler, zero-alloc bitmask helper, and schema validator."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Sequence

import numpy as np


def normalize_json_schema(schema: str | dict[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema, ensure_ascii=False)


def validate_json_against_schema(
    val: Any,
    sch: dict[str, Any],
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> None:
    """Lightweight recursive JSON Schema subset validator (object/array/string/int/num/bool/enum/$ref)."""
    eff_root = root if root is not None else sch
    if "$ref" in sch:
        ref = sch["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise ValueError(f"{path}: unsupported $ref {ref!r}")
        sch = eff_root["$defs"][ref[len("#/$defs/") :]]
    t = sch.get("type")
    if t == "object":
        if not isinstance(val, dict):
            raise ValueError(f"{path}: expected object, got {type(val).__name__}")
        for req in sch.get("required", []):
            if req not in val:
                raise ValueError(f"{path}: missing required key {req!r}")
        props = sch.get("properties", {})
        if sch.get("additionalProperties") is False:
            extra = set(val.keys()) - set(props.keys())
            if extra:
                raise ValueError(f"{path}: unexpected keys {sorted(extra)}")
        for k, v in val.items():
            if k in props:
                validate_json_against_schema(v, props[k], eff_root, f"{path}.{k}")
    elif t == "array":
        if not isinstance(val, list):
            raise ValueError(f"{path}: expected array, got {type(val).__name__}")
        if "maxItems" in sch and len(val) > sch["maxItems"]:
            raise ValueError(f"{path}: len {len(val)} > maxItems {sch['maxItems']}")
        if "items" in sch:
            for idx, item in enumerate(val):
                validate_json_against_schema(item, sch["items"], eff_root, f"{path}[{idx}]")
    elif t == "string":
        if not isinstance(val, str):
            raise ValueError(f"{path}: expected string, got {type(val).__name__}")
    elif t == "integer":
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{path}: expected integer, got {type(val).__name__}")
        if "minimum" in sch and val < sch["minimum"]:
            raise ValueError(f"{path}: {val} < minimum {sch['minimum']}")
        if "maximum" in sch and val > sch["maximum"]:
            raise ValueError(f"{path}: {val} > maximum {sch['maximum']}")
    elif t == "number":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{path}: expected number, got {type(val).__name__}")
    elif t == "boolean":
        if not isinstance(val, bool):
            raise ValueError(f"{path}: expected boolean, got {type(val).__name__}")
    elif t == "null":
        if val is not None:
            raise ValueError(f"{path}: expected null, got {type(val).__name__}")
    if "enum" in sch and val not in sch["enum"]:
        raise ValueError(f"{path}: {val!r} not in enum {sch['enum']}")


def safe_argmax_1d(logits_1d: Any) -> int:
    """Return `int(np.argmax(logits_1d))`, or `-1` when empty or any `NaN` entry is present."""
    arr = np.asarray(logits_1d)
    if arr.size == 0 or bool(np.isnan(arr).any()):
        return -1
    return int(np.argmax(arr))


def _sample_from_logits(
    logits_1d: np.ndarray,
    new_to_old: np.ndarray,
    *,
    temperature: float,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature <= 0.0:
        best = safe_argmax_1d(logits_1d)
        return int(new_to_old[best]) if best >= 0 else -1
    gen = rng if rng is not None else np.random.default_rng()
    scaled = logits_1d.astype(np.float64) / float(temperature)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0.0:
        best = safe_argmax_1d(logits_1d)
        return int(new_to_old[best]) if best >= 0 else -1
    probs /= s
    if top_p < 1.0:
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cdf = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(cdf, top_p, side="right")) + 1
        keep = order[: max(1, cutoff)]
        sub_p = probs[keep]
        sub_p /= sub_p.sum()
        j = int(gen.choice(keep, p=sub_p))
        return int(new_to_old[j])
    j = int(gen.choice(len(probs), p=probs))
    return int(new_to_old[j])


def sample_unconstrained(
    logits_1d: np.ndarray,
    new_to_old: np.ndarray,
    argmax_idx: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature <= 0.0:
        if 0 <= argmax_idx < len(new_to_old):
            return int(new_to_old[argmax_idx])
        best = safe_argmax_1d(logits_1d)
        return int(new_to_old[best]) if best >= 0 else -1
    return _sample_from_logits(logits_1d, new_to_old, temperature=temperature, top_p=top_p, rng=rng)


def _import_llguidance():
    try:
        import llguidance
    except ImportError as exc:
        raise ImportError(
            "llguidance is required for GrammarHelper (install `fast-shared[llm]` or `llguidance>=1.0.0`)."
        ) from exc
    return llguidance


class GrammarHelper:
    """Thread-safe pruned-vocab `llguidance` matcher with zero-allocation bitmask mapping."""

    def __init__(
        self,
        tokenizer: Any,
        schema: str | dict[str, Any],
        new_to_old: np.ndarray | None = None,
        *,
        whitespace_flexible: bool | None = False,
        eos_token_id: int | None = None,
        separators: tuple[str, str] | None = None,
    ) -> None:
        llguidance = _import_llguidance()
        self.tokenizer = tokenizer
        self.schema_str = normalize_json_schema(schema)
        self.schema_obj: dict[str, Any] = json.loads(self.schema_str)
        self.ltok = tokenizer.llguidance_tokenizer() if hasattr(tokenizer, "llguidance_tokenizer") else tokenizer
        self.eos_token_id = int(
            eos_token_id
            if eos_token_id is not None
            else getattr(tokenizer, "eos_token_id", getattr(self.ltok, "eos_token", 0))
        )
        eff_separators = (
            separators
            if separators is not None
            else ((",", ": ") if whitespace_flexible is None else (",", ":"))
        )
        eff_whitespace_flexible = bool(whitespace_flexible)
        self.separators = eff_separators
        self.whitespace_flexible = eff_whitespace_flexible
        self.gram = llguidance.JsonCompiler(
            separators=eff_separators,
            whitespace_flexible=eff_whitespace_flexible,
        ).compile(self.schema_str)
        m = llguidance.LLMatcher(self.ltok, self.gram)
        self.upfront_ff: list[int] = list(m.compute_ff_tokens())
        if new_to_old is None:
            self.new_to_old = np.arange(int(self.ltok.vocab_size), dtype=np.int32)
        else:
            self.new_to_old = np.asarray(new_to_old, dtype=np.int32)
        self.k_word = self.new_to_old >> 5
        self.k_bit = self.new_to_old & 31
        n_words = (int(self.ltok.vocab_size) + 31) // 32
        self._mask_lock = threading.Lock()
        self._mask_buf = np.empty(n_words, dtype=np.uint32)
        self._mask_ptr = int(self._mask_buf.ctypes.data)
        self._mask_nbytes = int(self._mask_buf.nbytes)
        self._llguidance = llguidance

    def new_matcher(self):
        m = self._llguidance.LLMatcher(self.ltok, self.gram)
        if self.upfront_ff:
            m.consume_tokens(self.upfront_ff)
        return m

    def pick_valid_token(
        self,
        matcher: Any,
        logits_1d: np.ndarray | Callable[[], np.ndarray],
        argmax_idx: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        rng: np.random.Generator | None = None,
        fallback_to_full_vocab: bool = False,
        full_logits_1d: np.ndarray | Callable[[], np.ndarray] | None = None,
    ) -> tuple[int, bool]:
        vocab_sz = int(self.ltok.vocab_size)
        if temperature <= 0.0:
            if 0 <= argmax_idx < len(self.new_to_old):
                orig_id = int(self.new_to_old[argmax_idx])
                if matcher.validate_tokens([orig_id]):
                    return orig_id, False
            elif fallback_to_full_vocab and 0 <= argmax_idx < vocab_sz:
                if matcher.validate_tokens([int(argmax_idx)]):
                    return int(argmax_idx), False

        # Protect shared `_mask_buf` across concurrent threads; copy bit slices before releasing lock.
        with self._mask_lock:
            matcher.unsafe_compute_mask_ptr(self._mask_ptr, self._mask_nbytes)
            allowed = ((self._mask_buf[self.k_word] >> self.k_bit) & 1) != 0
            any_allowed = bool(np.any(allowed))
            valid_ids: np.ndarray | None = None
            argmax_full_allowed: bool | None = None
            if not any_allowed and fallback_to_full_vocab:
                bit_shifts = np.arange(32, dtype=np.uint32)
                valid_mask = ((self._mask_buf[:, None] >> bit_shifts) & 1).ravel()[:vocab_sz] != 0
                valid_ids = np.flatnonzero(valid_mask).astype(np.int32)
            elif any_allowed and not (0 <= argmax_idx < len(allowed)) and 0 <= argmax_idx < vocab_sz:
                argmax_full_allowed = bool((self._mask_buf[argmax_idx >> 5] >> (argmax_idx & 31)) & 1)

        if not any_allowed:
            if fallback_to_full_vocab:
                if valid_ids is not None and len(valid_ids) > 0:
                    full_arr: np.ndarray | None = None
                    if full_logits_1d is not None:
                        full_arr = np.asarray(
                            full_logits_1d() if callable(full_logits_1d) else full_logits_1d,
                            dtype=np.float32,
                        )
                    else:
                        logits_arr = np.asarray(
                            logits_1d() if callable(logits_1d) else logits_1d,
                            dtype=np.float32,
                        )
                        if len(logits_arr) >= vocab_sz:
                            full_arr = logits_arr
                    if full_arr is not None and len(full_arr) >= vocab_sz:
                        cand_logits = full_arr[valid_ids]
                        eos_allowed = int(self.eos_token_id) in {int(v) for v in valid_ids}
                        fallback_id = int(self.eos_token_id) if eos_allowed else int(valid_ids[0])
                        if temperature <= 0.0:
                            best_f = safe_argmax_1d(cand_logits)
                            return (int(valid_ids[best_f]) if best_f >= 0 else fallback_id), True
                        picked_f = _sample_from_logits(
                            cand_logits,
                            valid_ids,
                            temperature=temperature,
                            top_p=top_p,
                            rng=rng,
                        )
                        return (picked_f if picked_f >= 0 else fallback_id), True
                    if int(self.eos_token_id) in set(int(v) for v in valid_ids):
                        return self.eos_token_id, True
                    raise ValueError(
                        "Pruned vocabulary yielded no grammar-allowed tokens; fallback_to_full_vocab=True "
                        "requires `full_logits_1d` (or unpruned `logits_1d`) to score full-vocab candidates."
                    )
            return self.eos_token_id, True

        def _nan_safe_valid_fallback(mask: np.ndarray) -> int:
            eos_orig = int(self.eos_token_id)
            eos_new_matches = np.flatnonzero(self.new_to_old == eos_orig)
            if len(eos_new_matches) > 0:
                eos_new = int(eos_new_matches[0])
                if 0 <= eos_new < len(mask) and bool(mask[eos_new]):
                    return eos_orig
            valid_new = np.flatnonzero(mask)
            if len(valid_new) > 0:
                return int(self.new_to_old[int(valid_new[0])])
            return eos_orig

        logits_arr = logits_1d() if callable(logits_1d) else logits_1d
        pruned_logits = (
            logits_arr[self.new_to_old]
            if len(logits_arr) >= vocab_sz and len(self.new_to_old) < len(logits_arr)
            else logits_arr
        )
        if 0 <= argmax_idx < len(allowed):
            was_masked = not bool(allowed[argmax_idx])
        elif len(logits_arr) >= vocab_sz and argmax_full_allowed is not None:
            was_masked = not argmax_full_allowed
        else:
            raw_best = safe_argmax_1d(pruned_logits)
            was_masked = True if raw_best < 0 else (not bool(allowed[raw_best]))
        masked = np.where(allowed, pruned_logits, -1e9)
        if temperature <= 0.0:
            best_j = safe_argmax_1d(masked)
            if best_j < 0:
                return _nan_safe_valid_fallback(allowed), True
            return int(self.new_to_old[best_j]), bool(was_masked)
        picked = _sample_from_logits(masked, self.new_to_old, temperature=temperature, top_p=top_p, rng=rng)
        if picked < 0:
            return _nan_safe_valid_fallback(allowed), True
        return picked, bool(was_masked)

    def validate_token_sequence(self, full_ids: Sequence[int]) -> bool:
        """Verify that `full_ids` is strictly accepted from initial state and stops in an accepting state."""
        m = self._llguidance.LLMatcher(self.ltok, self.gram)
        idx = 0
        n = len(full_ids)
        while idx < n:
            ff = m.compute_ff_tokens()
            if ff and list(full_ids[idx : idx + len(ff)]) == ff:
                m.consume_tokens(ff)
                idx += len(ff)
                continue
            tid = int(full_ids[idx])
            if not m.validate_tokens([tid]):
                return False
            m.consume_token(tid)
            idx += 1
        return bool(m.is_stopped() and m.is_accepting())


def compute_loop_breaker_penalties(
    token_ids: Sequence[int] | None,
    *,
    window: int = 96,
    min_prior_hits: int = 4,
    base_penalty: float = 8.0,
    stop_token_ids: frozenset[int] = frozenset(),
) -> dict[int, float]:
    """Return `{next_token_id: penalty}` on cyclic (`p=1..8`) or dense 4-gram degenerate repeat loops (`A4_v2`)."""
    if not token_ids:
        return {}
    n = len(token_ids)
    out: dict[int, float] = {}

    # 1. Exact consecutive periodic suffix detector (`p=1..8`)
    if n >= 6:
        tail = [int(t) for t in token_ids[-min(n, 32) :]]
        nt = len(tail)
        if nt >= 6 and all(t == tail[-1] for t in tail[-6:-1]):
            d = tail[-1]
            if d not in stop_token_ids:
                out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)
        if nt >= 8 and tail[-2:] == tail[-4:-2] == tail[-6:-4] == tail[-8:-6]:
            d = tail[-2]
            if d not in stop_token_ids:
                out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)
        for p in range(3, 9):
            if nt >= 3 * p:
                pat = tail[-p:]
                if tail[-2 * p : -p] == pat and tail[-3 * p : -2 * p] == pat:
                    d = pat[0]
                    if d not in stop_token_ids:
                        out[d] = max(out.get(d, 0.0), float(base_penalty) * 2.0)

    # 2. Trailing-window 4-gram frequency check (fires on `(min_prior_hits + 1)`-th occurrence)
    if n >= (3 + min_prior_hits):
        start = max(0, n - int(window))
        a, b, c = int(token_ids[-3]), int(token_ids[-2]), int(token_ids[-1])
        if c in token_ids[start : n - 1]:
            hits: dict[int, int] = {}
            for i in range(start, n - 3):
                if int(token_ids[i]) == a and int(token_ids[i + 1]) == b and int(token_ids[i + 2]) == c:
                    d = int(token_ids[i + 3])
                    if d not in stop_token_ids:
                        hits[d] = hits.get(d, 0) + 1
            for d, cnt in hits.items():
                if cnt >= min_prior_hits:
                    pen = float(base_penalty) * float(min(cnt - min_prior_hits + 1, 3))
                    out[d] = max(out.get(d, 0.0), pen)
    return out


def apply_loop_breaker_penalties_1d(
    row_logits: np.ndarray,
    penalties: dict[int, float],
    old_to_new: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Copy a 1D logit vector, subtract `penalties` in active-vocab space, and return `(adjusted, argmax_idx)`."""
    adj = np.array(row_logits, copy=True)
    n_o2n = int(len(old_to_new)) if old_to_new is not None else -1
    v_len = int(adj.shape[0])
    for orig_id, pen in penalties.items():
        idx = int(old_to_new[orig_id]) if (old_to_new is not None and 0 <= orig_id < n_o2n) else int(orig_id)
        if 0 <= idx < v_len:
            adj[idx] = np.float32(float(adj[idx]) - float(pen))
    return adj, safe_argmax_1d(adj)


__all__ = [
    "GrammarHelper",
    "apply_loop_breaker_penalties_1d",
    "compute_loop_breaker_penalties",
    "normalize_json_schema",
    "safe_argmax_1d",
    "sample_unconstrained",
    "validate_json_against_schema",
]
