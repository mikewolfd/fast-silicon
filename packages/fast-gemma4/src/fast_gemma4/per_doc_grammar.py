"""Per-document JSON grammars inside one rolling batch.

`run_rolling` asks its grammar helper for one matcher per admitted document, in admission order.
`PerDocGrammarHelper` answers with the standard schema's matcher for most documents and a narrower schema's
matcher for the ones the caller names. The caller supplies the narrower schemas: `doc_schema_keys` maps a
document index to a key, and `schema_for_key` turns a key into a JSON Schema string. What the keys mean is the
caller's business; this module knows nothing about any schema.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


GrammarCacheKey = str | tuple[tuple[str, str], str] | tuple[tuple[str, str], bool, str]


class SyncMatcher:
    """Keeps a document's own (narrower) `LLMatcher` in step with a shadow matcher of the batch's standard schema.

    The narrow grammar `kind_m` always decides validity. Fast-forward tokens follow the standard matcher
    `std_m` for as long as its output is a prefix of the narrow one, so every slot in a `run_rolling` batch
    sees the same upfront tokens and fast-forward lengths (no cross-slot prefill-length or FF-drain jitter).
    """

    __slots__ = ("kind_m", "std_m", "_synced")

    def __init__(self, kind_m: Any, std_m: Any):
        self.kind_m = kind_m
        self.std_m = std_m
        self._synced = True

    def try_consume_tokens(self, tokens: list[int]) -> int:
        n = self.kind_m.try_consume_tokens(tokens)
        if n == 0:
            self._synced = False
        elif self._synced:
            if self.std_m.try_consume_tokens(tokens[:n]) != n:
                self._synced = False
        return n

    def consume_token(self, tok_id: int) -> bool:
        ok = self.kind_m.consume_token(tok_id)
        if ok and self._synced:
            if not self.std_m.consume_token(tok_id):
                self._synced = False
        return ok

    def consume_tokens(self, tokens: list[int]) -> bool:
        ok = self.kind_m.consume_tokens(tokens)
        if ok and self._synced:
            if not self.std_m.consume_tokens(tokens):
                self._synced = False
        return ok

    def compute_ff_tokens(self) -> list[int]:
        k_ff = self.kind_m.compute_ff_tokens()
        if not self._synced or not k_ff:
            return k_ff
        s_ff = self.std_m.compute_ff_tokens()
        if len(s_ff) <= len(k_ff) and list(k_ff[: len(s_ff)]) == list(s_ff):
            return s_ff
        self._synced = False
        return k_ff

    def unsafe_compute_mask_ptr(self, ptr: int, nbytes: int) -> None:
        self.kind_m.unsafe_compute_mask_ptr(ptr, nbytes)

    def validate_tokens(self, tokens: Sequence[int]) -> bool:
        return bool(self.kind_m.validate_tokens(list(tokens)))

    def is_stopped(self) -> bool:
        return self.kind_m.is_stopped()

    def is_accepting(self) -> bool:
        return self.kind_m.is_accepting()


def compute_rolling_doc_order(
    suf_ids_list: Sequence[Sequence[int]],
    *,
    b_slots: int = 256,
    sort_by_length: bool = True,
) -> list[int]:
    """Compute exact admission order matching `fast_gemma4.speculative.run_decode_rolling`."""
    n_docs = len(suf_ids_list)
    if n_docs == 0:
        return []
    eff_b_slots = max(1, min(int(b_slots), n_docs))
    if sort_by_length and n_docs > eff_b_slots:
        return sorted(range(n_docs), key=lambda d: -len(suf_ids_list[d]))
    return list(range(n_docs))


class PerDocGrammarHelper:
    """Drop-in `GrammarHelper` for `run_rolling` that hands each document its own matcher."""

    def __init__(
        self,
        gh_std: Any,
        doc_order: Sequence[int] = (),
        *,
        doc_schema_keys: dict[int, str] | None = None,
        schema_for_key: Callable[[str], str],
        grammar_cache: dict[GrammarCacheKey, tuple[Any, list[int]]] | None = None,
        sync_ff_with_std: bool = False,
    ):
        import llguidance

        self._llguidance = llguidance
        self.tok = getattr(gh_std, "tok", getattr(gh_std, "tokenizer", None))
        self.tokenizer = self.tok
        self.ltok = gh_std.ltok
        self.upfront_ff = gh_std.upfront_ff
        self.new_to_old = gh_std.new_to_old
        self._gh_std = gh_std
        self._schema_for_key = schema_for_key
        self._separators = getattr(gh_std, "separators", (",", ":"))
        self._whitespace_flexible = bool(getattr(gh_std, "whitespace_flexible", False))
        self._max_whitespace_run = getattr(gh_std, "max_whitespace_run", 0)
        self._ws_profile = None
        if self._max_whitespace_run:
            from .grammar import get_whitespace_profile

            self._ws_profile = get_whitespace_profile(self.ltok)
        self._compiler = llguidance.JsonCompiler(
            separators=self._separators,
            whitespace_flexible=self._whitespace_flexible,
        )
        self._grammar_cache: dict[GrammarCacheKey, tuple[Any, list[int]]] = (
            grammar_cache if grammar_cache is not None else {}
        )
        self._doc_order = list(doc_order)
        self._doc_schema_keys: dict[int, str] = dict(doc_schema_keys or {})
        self._cursor = 0
        self.dispatched_doc_indices: list[int] = []

        # Prompt prefill and matcher initialization must agree even when the narrower schema reorders fields.
        self._use_standard_upfront: set[str] = set()
        for key in set(self._doc_schema_keys.values()):
            gram, up_ff = self.get_grammar_entry(key)
            if sync_ff_with_std and (
                up_ff[: len(self.upfront_ff)] == self.upfront_ff
                or llguidance.LLMatcher(self.ltok, gram).validate_tokens(self.upfront_ff) == len(self.upfront_ff)
            ):
                self._use_standard_upfront.add(key)

    def get_grammar_entry(self, key: str) -> tuple[Any, list[int]]:
        """`(compiled grammar, its upfront fast-forward tokens)` for a schema key, compiled once per cache."""
        cache_key: GrammarCacheKey
        if self._whitespace_flexible:
            cache_key = (self._separators, True, key)
        else:
            cache_key = key if self._separators == (",", ":") else (self._separators, key)
        entry = self._grammar_cache.get(cache_key)
        if entry is None:
            gram = self._compiler.compile(self._schema_for_key(key))
            probe_m = self._llguidance.LLMatcher(self.ltok, gram)
            up_ff = list(probe_m.compute_ff_tokens())
            entry = (gram, up_ff)
            self._grammar_cache[cache_key] = entry
        return entry

    def upfront_ff_for_doc(self, doc_idx: int) -> list[int]:
        key = self._doc_schema_keys.get(doc_idx)
        if key is None or key in self._use_standard_upfront:
            return list(self._gh_std.upfront_ff)
        return list(self.get_grammar_entry(key)[1])

    def new_matcher_for_doc(self, doc_idx: int) -> Any:
        d_i = int(doc_idx)
        self._cursor += 1
        self.dispatched_doc_indices.append(d_i)
        key = self._doc_schema_keys.get(d_i)
        if key is not None:
            gram, up_ff = self.get_grammar_entry(key)
            m = self._llguidance.LLMatcher(self.ltok, gram)
            if self._ws_profile is not None:
                from .grammar import WhitespaceGuard, WhitespaceGuardedMatcher

                m = WhitespaceGuardedMatcher(m, WhitespaceGuard(self._max_whitespace_run, self._ws_profile))
            if key in self._use_standard_upfront:
                if self.upfront_ff:
                    m.consume_tokens(self.upfront_ff)
                return SyncMatcher(m, self._gh_std.new_matcher())
            if up_ff:
                m.consume_tokens(up_ff)
            return m
        return self._gh_std.new_matcher()

    def new_matcher(self) -> Any:
        if self._cursor < len(self._doc_order):
            doc_idx = self._doc_order[self._cursor]
            return self.new_matcher_for_doc(doc_idx)
        self._cursor += 1
        return self._gh_std.new_matcher()

    def pick_valid_token(
        self,
        matcher: Any,
        logits_1d: Any,
        argmax_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[int, bool]:
        return self._gh_std.pick_valid_token(matcher, logits_1d, argmax_idx, *args, **kwargs)

    def validate_token_sequence(self, full_ids: Sequence[int]) -> bool:
        return bool(self._gh_std.validate_token_sequence(full_ids))


__all__ = ["PerDocGrammarHelper", "SyncMatcher", "compute_rolling_doc_order"]
