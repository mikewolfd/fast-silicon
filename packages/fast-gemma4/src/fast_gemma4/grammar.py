"""Gemma-4 `llguidance` grammar helper with zero-bloat runtime `WhitespaceGuard` for flexible JSON."""

from __future__ import annotations

from collections import OrderedDict
import ctypes
import threading
from typing import Any, Sequence

import numpy as np

from fast_shared.grammar import (
    GrammarHelper as _BaseGrammarHelper,
    _sample_from_logits,
    normalize_json_schema as _normalize_schema,
    sample_unconstrained,
    validate_json_against_schema,
)

DEFAULT_MAX_WHITESPACE_RUN: int = 32
_JSON_WS_BYTES = frozenset((0x20, 0x09, 0x0A, 0x0D))  # space, tab, LF, CR
_PROFILE_CACHE_LOCK = threading.Lock()
_PROFILE_CACHE: OrderedDict[tuple[int, int], "WhitespaceProfile"] = OrderedDict()
_MAX_PROFILE_CACHE = 4


class WhitespaceProfile:
    """Precomputed per-token JSON leading-whitespace lengths and lazy `uint32` exclusion bitmasks.

    Costs `~512 KiB` (`uint16[262144]`) plus `32 KiB` per distinct remaining-run bitmask lazily,
    preventing infinite indentation runaway in flexible JSON without expanding compiled grammars
    (contrast XGrammar `max_whitespace_cnt=32` inflating 262K grammars from `0.3 MiB` to `1.2 GiB`).
    """

    def __init__(self, ltok: Any) -> None:
        self.ltok = ltok
        vocab_size = int(ltok.vocab_size)
        self.vocab_size = vocab_size
        self.words = (vocab_size + 31) // 32
        used = vocab_size % 32
        self.last_word_mask = np.uint32(0xFFFFFFFF if used == 0 else ((1 << used) - 1))

        leading = np.zeros(vocab_size, dtype=np.uint16)
        decode_bytes = ltok.decode_bytes
        ws = _JSON_WS_BYTES
        for tid in range(vocab_size):
            raw = decode_bytes([tid])
            if raw and raw[0] in ws:
                cnt = 0
                for b in raw:
                    if b in ws:
                        cnt += 1
                    else:
                        break
                leading[tid] = min(cnt, 65535)
        self.leading_ws = leading
        self._mask_lock = threading.Lock()
        self._exceed_masks: dict[int, np.ndarray] = {}

    def token_bytes(self, token_id: int) -> bytes:
        if 0 <= token_id < self.vocab_size:
            return bytes(self.ltok.decode_bytes([int(token_id)]))
        return b""

    def exceeding_mask(self, remaining: int) -> np.ndarray:
        """Return `uint32[words]` bitmask with bit `tid` set iff `leading_ws[tid] > remaining`."""
        rem = max(0, int(remaining))
        with self._mask_lock:
            cached = self._exceed_masks.get(rem)
            if cached is not None:
                return cached
        bad_ids = np.flatnonzero(self.leading_ws > rem).astype(np.int64)
        mask = np.zeros(self.words, dtype=np.uint32)
        if bad_ids.size > 0:
            np.bitwise_or.at(mask, bad_ids >> 5, np.uint32(1) << (bad_ids & 31).astype(np.uint32))
        mask.setflags(write=False)
        with self._mask_lock:
            self._exceed_masks[rem] = mask
        return mask

    @property
    def cached_mask_bytes(self) -> int:
        with self._mask_lock:
            return int(self.leading_ws.nbytes) + sum(int(m.nbytes) for m in self._exceed_masks.values())


def get_whitespace_profile(ltok: Any) -> WhitespaceProfile:
    """Return process-cached `WhitespaceProfile` for `ltok`."""
    key = (id(ltok), int(ltok.vocab_size))
    with _PROFILE_CACHE_LOCK:
        prof = _PROFILE_CACHE.get(key)
        if prof is not None and prof.ltok is ltok:
            _PROFILE_CACHE.move_to_end(key)
            return prof
    prof = WhitespaceProfile(ltok)
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[key] = prof
        _PROFILE_CACHE.move_to_end(key)
        while len(_PROFILE_CACHE) > _MAX_PROFILE_CACHE:
            _PROFILE_CACHE.popitem(last=False)
    return prof


class WhitespaceGuard:
    """Tracks inter-token JSON whitespace runs (ignoring strings) and caps consecutive whitespace at `limit`."""

    _HISTORY_LIMIT = 256

    def __init__(self, limit: int, profile: WhitespaceProfile) -> None:
        if int(limit) <= 0:
            raise ValueError(f"WhitespaceGuard limit must be positive, got {limit}")
        self.limit = int(limit)
        self.profile = profile
        self.in_string = False
        self.escaped = False
        self.run = 0
        self._history: list[tuple[bool, bool, int]] = []

    def reset(self) -> None:
        self.in_string = False
        self.escaped = False
        self.run = 0
        self._history.clear()

    def _advance_bytes(
        self,
        raw: bytes,
        in_string: bool,
        escaped: bool,
        run: int,
    ) -> tuple[bool, bool, int]:
        ws = _JSON_WS_BYTES
        for b in raw:
            if in_string:
                if escaped:
                    escaped = False
                elif b == 0x5C:  # backslash
                    escaped = True
                elif b == 0x22:  # quote
                    in_string = False
                    run = 0
            elif b == 0x22:  # quote
                in_string = True
                run = 0
            elif b in ws:
                run += 1
            else:
                run = 0
        return in_string, escaped, run

    def accept(self, token_id: int) -> None:
        self._history.append((self.in_string, self.escaped, self.run))
        if len(self._history) > self._HISTORY_LIMIT:
            del self._history[: len(self._history) - self._HISTORY_LIMIT]
        raw = self.profile.token_bytes(int(token_id))
        self.in_string, self.escaped, self.run = self._advance_bytes(raw, self.in_string, self.escaped, self.run)

    def accept_many(self, token_ids: Sequence[int]) -> None:
        for tid in token_ids:
            self.accept(int(tid))

    def rollback(self, count: int) -> None:
        n = int(count)
        if n <= 0:
            return
        if n <= len(self._history):
            self.in_string, self.escaped, self.run = self._history[-n]
            del self._history[-n:]
        else:
            self.reset()

    def would_exceed(self, token_id: int) -> bool:
        if self.in_string:
            return False
        tid = int(token_id)
        if not (0 <= tid < self.profile.vocab_size):
            return False
        remaining = max(0, self.limit - self.run)
        return int(self.profile.leading_ws[tid]) > remaining

    def would_exceed_sequence(self, token_ids: Sequence[int]) -> bool:
        if not token_ids:
            return False
        if len(token_ids) == 1:
            return self.would_exceed(int(token_ids[0]))
        in_str, esc, run = self.in_string, self.escaped, self.run
        lim = self.limit
        lead = self.profile.leading_ws
        vsz = self.profile.vocab_size
        for tid_raw in token_ids:
            tid = int(tid_raw)
            if not in_str and 0 <= tid < vsz and int(lead[tid]) > max(0, lim - run):
                return True
            in_str, esc, run = self._advance_bytes(self.profile.token_bytes(tid), in_str, esc, run)
        return False

    def restrict_mask_u32(self, mask_u32: np.ndarray) -> bool:
        """Clear tokens that would exceed `limit` from `mask_u32` in place, unless that leaves zero survivors."""
        if self.in_string:
            return False
        remaining = max(0, self.limit - self.run)
        blocked = self.profile.exceeding_mask(remaining)
        n_words = min(len(mask_u32), len(blocked))
        if n_words == 0:
            return False
        survivors = mask_u32[:n_words] & (~blocked[:n_words])
        if n_words == self.profile.words:
            survivors[n_words - 1] &= self.profile.last_word_mask
        if not bool(np.any(survivors != 0)):
            return False
        mask_u32[:n_words] = survivors
        return True


class WhitespaceGuardedMatcher:
    """Drop-in `LLMatcher` wrapper applying `WhitespaceGuard` to token validation and bitmasks."""

    __slots__ = ("_inner", "_guard")

    def __init__(self, inner: Any, guard: WhitespaceGuard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def whitespace_guard(self) -> WhitespaceGuard:
        return self._guard

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        toks = list(tokens)
        if self._guard.would_exceed_sequence(toks):
            return False
        return bool(self._inner.validate_tokens(toks))

    def try_consume_tokens(self, tokens: Sequence[int]) -> int:
        toks = list(tokens)
        if not toks:
            return 0
        if len(toks) == 1:
            if self._guard.would_exceed(toks[0]):
                return 0
            n = int(self._inner.try_consume_tokens(toks))
            if n == 1:
                self._guard.accept(toks[0])
            return n
        # Multi-token: allow only longest prefix within whitespace cap.
        allowed_len = 0
        in_str, esc, run = self._guard.in_string, self._guard.escaped, self._guard.run
        lim = self._guard.limit
        lead = self._guard.profile.leading_ws
        vsz = self._guard.profile.vocab_size
        for tid_raw in toks:
            tid = int(tid_raw)
            if not in_str and 0 <= tid < vsz and int(lead[tid]) > max(0, lim - run):
                break
            in_str, esc, run = self._guard._advance_bytes(self._guard.profile.token_bytes(tid), in_str, esc, run)
            allowed_len += 1
        if allowed_len == 0:
            return 0
        n = int(self._inner.try_consume_tokens(toks[:allowed_len]))
        if n > 0:
            self._guard.accept_many(toks[:n])
        return n

    def consume_token(self, tok_id: int) -> bool:
        ok = bool(self._inner.consume_token(int(tok_id)))
        if ok:
            self._guard.accept(int(tok_id))
        return ok

    def consume_tokens(self, tokens: Sequence[int]) -> bool:
        toks = [int(t) for t in tokens]
        ok = bool(self._inner.consume_tokens(toks))
        if ok:
            self._guard.accept_many(toks)
        return ok

    def compute_ff_tokens(self) -> list[int]:
        return list(self._inner.compute_ff_tokens())

    def unsafe_compute_mask_ptr(self, ptr: int, nbytes: int) -> None:
        self._inner.unsafe_compute_mask_ptr(ptr, nbytes)
        n_words = int(nbytes) // 4
        if n_words > 0:
            c_arr = (ctypes.c_uint32 * n_words).from_address(int(ptr))
            mask_view = np.ctypeslib.as_array(c_arr)
            self._guard.restrict_mask_u32(mask_view)

    def is_stopped(self) -> bool:
        return bool(self._inner.is_stopped())

    def is_accepting(self) -> bool:
        return bool(self._inner.is_accepting())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class GrammarHelper(_BaseGrammarHelper):
    """Pruned-vocab `llguidance` helper with default `WhitespaceGuard` cap (`32`) in flexible JSON mode."""

    def __init__(
        self,
        tokenizer: Any,
        schema: str | dict[str, Any],
        new_to_old: np.ndarray | None = None,
        *,
        whitespace_flexible: bool = False,
        max_whitespace_run: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        super().__init__(
            tokenizer,
            schema,
            new_to_old,
            whitespace_flexible=whitespace_flexible,
            eos_token_id=eos_token_id,
        )
        self.whitespace_flexible = bool(whitespace_flexible)
        if not self.whitespace_flexible:
            self.max_whitespace_run = 0
        elif max_whitespace_run is None:
            self.max_whitespace_run = DEFAULT_MAX_WHITESPACE_RUN
        else:
            if int(max_whitespace_run) < 0:
                raise ValueError(f"max_whitespace_run must be >= 0 (0 disables), got {max_whitespace_run}")
            self.max_whitespace_run = int(max_whitespace_run)
        self._ws_profile: WhitespaceProfile | None = (
            get_whitespace_profile(self.ltok) if self.max_whitespace_run > 0 else None
        )

    def new_matcher(self):
        inner = self._llguidance.LLMatcher(self.ltok, self.gram)
        if self._ws_profile is not None and self.max_whitespace_run > 0:
            m: Any = WhitespaceGuardedMatcher(inner, WhitespaceGuard(self.max_whitespace_run, self._ws_profile))
            if self.upfront_ff:
                m.consume_tokens(self.upfront_ff)
            return m
        if self.upfront_ff:
            inner.consume_tokens(self.upfront_ff)
        return inner


__all__ = [
    "DEFAULT_MAX_WHITESPACE_RUN",
    "GrammarHelper",
    "WhitespaceGuard",
    "WhitespaceGuardedMatcher",
    "WhitespaceProfile",
    "_normalize_schema",
    "_sample_from_logits",
    "get_whitespace_profile",
    "sample_unconstrained",
    "validate_json_against_schema",
]
