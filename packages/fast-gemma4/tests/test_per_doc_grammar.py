"""`fast_gemma4.per_doc_grammar` (per-document JSON grammars in one rolling batch) tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from fast_gemma4.grammar import GrammarHelper
from fast_gemma4.paths import resolve_paths
from fast_gemma4.per_doc_grammar import PerDocGrammarHelper, SyncMatcher, compute_rolling_doc_order
from fast_gemma4.tokenizer import FastGemma4Tokenizer, VocabRemapper

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _schema(kinds: list[str]) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": kinds}, "n": {"type": "integer"}},
            "required": ["kind", "n"],
            "additionalProperties": False,
        }
    )


def test_stdlib_only_modules_import_without_third_party_packages() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import fast_gemma4.per_doc_grammar\n"
        "for banned in ('mlx', 'numpy', 'pyarrow', 'tokenizers', 'llguidance', 'torch'):\n"
        "    assert banned not in sys.modules, f'Unexpected import: {banned}'\n"
    )
    proc = subprocess.run(["/usr/bin/python3", "-B", "-I", "-S", "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_rolling_doc_order_matches_admission() -> None:
    sufs = [[1, 2], [1, 2, 3, 4, 5, 6], [1, 2, 3]]
    assert compute_rolling_doc_order(sufs, b_slots=1, sort_by_length=True) == [1, 2, 0]
    assert compute_rolling_doc_order(sufs, b_slots=3, sort_by_length=True) == [0, 1, 2]  # all admitted at once
    assert compute_rolling_doc_order(sufs, b_slots=1, sort_by_length=False) == [0, 1, 2]
    assert compute_rolling_doc_order([], b_slots=4) == []


def test_helper_hands_each_document_its_own_matcher_in_admission_order() -> None:
    paths = resolve_paths()
    tok_json = paths.resolve_tokenizer_json()
    if not Path(tok_json).exists():
        pytest.skip(f"Missing {tok_json}")
    tok = FastGemma4Tokenizer(tok_json)
    rem = VocabRemapper.load_or_build(tok_json, paths.vocab_remap_npz)
    gh_std = GrammarHelper(tok, _schema(["alpha", "beta", "gamma"]), rem.kept_old_ids, whitespace_flexible=False)

    compiled: list[str] = []

    def schema_for_key(key: str) -> str:
        compiled.append(key)
        return _schema(key.split("|"))

    cache: dict[str, tuple] = {}
    order = compute_rolling_doc_order([[1, 2], [1, 2, 3, 4, 5, 6]], b_slots=1, sort_by_length=True)
    helper = PerDocGrammarHelper(
        gh_std, order, doc_schema_keys={0: "beta"}, schema_for_key=schema_for_key, grammar_cache=cache,
        sync_ff_with_std=True,
    )
    m_first = helper.new_matcher()  # document 1: the standard schema
    m_second = helper.new_matcher()  # document 0: the narrow schema, kept in step with the standard one
    assert not isinstance(m_first, SyncMatcher)
    assert isinstance(m_second, SyncMatcher)
    assert helper.dispatched_doc_indices == [1, 0]
    assert helper.upfront_ff_for_doc(0) == list(gh_std.upfront_ff)

    # The narrow matcher refuses a value the standard schema allows (matchers read the tokenizer's own ids).
    assert not m_second.validate_tokens(tok.encode('alpha"', add_special_tokens=False))
    assert m_second.validate_tokens(tok.encode('beta"', add_special_tokens=False))
    assert m_first.validate_tokens(tok.encode('alpha"', add_special_tokens=False))

    # One compile per key for every helper that shares the cache; a set key is the caller's own convention.
    again = PerDocGrammarHelper(
        gh_std, [0, 1], doc_schema_keys={0: "beta", 1: "alpha|gamma"}, schema_for_key=schema_for_key,
        grammar_cache=cache,
    )
    assert compiled == ["beta", "alpha|gamma"]
    assert again.upfront_ff_for_doc(1) == list(again.get_grammar_entry("alpha|gamma")[1])
    assert again.validate_token_sequence(gh_std.upfront_ff) in (True, False)
