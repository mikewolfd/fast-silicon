"""CPU unit tests for `fast_gemma4.grammar.WhitespaceGuard` & flexible-JSON runaway protection."""

from __future__ import annotations

import json
import numpy as np
import pytest

from fast_gemma4.grammar import (
    DEFAULT_MAX_WHITESPACE_RUN,
    GrammarHelper,
    WhitespaceGuard,
    WhitespaceGuardedMatcher,
    WhitespaceProfile,
)
from fast_gemma4.paths import resolve_paths
from fast_gemma4.tokenizer import FastGemma4Tokenizer


class _FakeLLTokenizer:
    """Minimal pure-CPU stand-in for `llguidance.LLTokenizer`."""

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = pieces
        self.vocab_size = len(pieces)
        self.eos_token = 0

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self._pieces[i] for i in ids)


def test_whitespace_guard_tracks_runs_ignores_strings_and_never_deadlocks_cpu() -> None:
    """Verify consecutive whitespace tracking, string escaping, rollback, and deadlock-free mask restriction."""
    pieces = [
        b"<eos>",      # 0
        b"{",          # 1
        b"}",          # 2
        b" ",          # 3 (lead=1)
        b"    ",       # 4 (lead=4)
        b"\n        ", # 5 (lead=9)
        b'"',          # 6 (quote)
        b"\\",         # 7 (backslash)
        b'\\"',        # 8 (escaped quote inside string)
        b'\\\\"',      # 9 (escaped backslash + closing quote)
        b"abc",        # 10
        b"  }",        # 11 (lead=2 then non-ws)
    ]
    prof = WhitespaceProfile(_FakeLLTokenizer(pieces))
    assert prof.leading_ws[0] == 0
    assert prof.leading_ws[3] == 1
    assert prof.leading_ws[4] == 4
    assert prof.leading_ws[5] == 9
    assert prof.leading_ws[11] == 2

    guard = WhitespaceGuard(limit=6, profile=prof)
    guard.accept(1)  # "{"
    assert not guard.in_string and guard.run == 0
    assert not guard.would_exceed(4)   # 4 <= 6
    assert guard.would_exceed(5)       # 9 > 6

    guard.accept(4)  # 4 spaces -> run=4, remaining=2
    assert guard.run == 4
    assert not guard.would_exceed(11)  # lead=2 <= 2
    assert guard.would_exceed(4)       # lead=4 > 2

    # Restrict a bitmask containing {2='}', 3=' ', 4='    ', 5='\n        ', 11='  }'}
    mask = np.array([(1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 11)], dtype=np.uint32)
    applied = guard.restrict_mask_u32(mask)
    assert applied is True
    # Tokens 4 (lead=4) and 5 (lead=9) must be cleared; 2, 3, 11 survive!
    assert int(mask[0]) == ((1 << 2) | (1 << 3) | (1 << 11))

    # Consume 2 more spaces -> run=6, remaining=0 -> all ws-leading tokens blocked, '}' survives.
    guard.accept(3)
    guard.accept(3)
    assert guard.run == 6
    assert guard.would_exceed(3) and guard.would_exceed(11) and not guard.would_exceed(2)

    # Deadlock safety: if ONLY whitespace tokens are allowed in mask, leave mask untouched!
    ws_only_mask = np.array([(1 << 3) | (1 << 4)], dtype=np.uint32)
    assert guard.restrict_mask_u32(ws_only_mask) is False
    assert int(ws_only_mask[0]) == ((1 << 3) | (1 << 4))

    # Rollback restores exact prior state.
    guard.rollback(2)
    assert guard.run == 4

    # Enter JSON string -> arbitrary whitespace inside strings is NEVER counted or blocked!
    guard.accept(6)  # '"'
    assert guard.in_string and guard.run == 0
    guard.accept(5)  # '\n        ' inside string
    guard.accept(5)
    assert guard.in_string and guard.run == 0
    assert not guard.would_exceed(5)
    mask_in_str = np.array([(1 << 5) | (1 << 2)], dtype=np.uint32)
    assert guard.restrict_mask_u32(mask_in_str) is False
    assert int(mask_in_str[0]) == ((1 << 5) | (1 << 2))

    # Escaped quote '\"' keeps in_string=True; '\\\\"' closes the string!
    guard.accept(8)
    assert guard.in_string is True
    guard.accept(9)
    assert guard.in_string is False and guard.run == 0


def test_grammar_helper_flexible_json_caps_runaway_whitespace_cpu() -> None:
    """Verify real Gemma-4 `GrammarHelper(whitespace_flexible=True)` stops runaway indentation with 0 grammar bloat."""
    paths = resolve_paths()
    try:
        tok_json = paths.resolve_tokenizer_json(paths.resolve_model_dir(None, local_files_only=True), local_files_only=True)
    except FileNotFoundError:
        pytest.skip("Local Gemma-4 tokenizer.json not staged")

    tok = FastGemma4Tokenizer(tok_json)
    schema = {
        "type": "object",
        "required": ["ok"],
        "additionalProperties": False,
        "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
    }

    # Compact JSON leaves WhitespaceGuard disabled (zero overhead).
    gh_compact = GrammarHelper(tok, schema, whitespace_flexible=False)
    assert gh_compact.max_whitespace_run == 0
    assert not isinstance(gh_compact.new_matcher(), WhitespaceGuardedMatcher)

    # Flexible JSON defaults to cap=32 and caps memory to < 2 MiB (vs 1.2 GiB XGrammar blowup).
    gh_flex = GrammarHelper(tok, schema, whitespace_flexible=True)
    assert gh_flex.max_whitespace_run == DEFAULT_MAX_WHITESPACE_RUN
    assert gh_flex._ws_profile is not None
    assert gh_flex._ws_profile.cached_mask_bytes < (2 << 20)

    # Explicit max_whitespace_run=0 disables guard cleanly.
    gh_unbounded = GrammarHelper(tok, schema, whitespace_flexible=True, max_whitespace_run=0)
    assert gh_unbounded.max_whitespace_run == 0
    assert not isinstance(gh_unbounded.new_matcher(), WhitespaceGuardedMatcher)

    # Simulate an adversarial model assigning highest logit (+100.0) to newline/space tokens forever!
    gh_cap8 = GrammarHelper(tok, schema, whitespace_flexible=True, max_whitespace_run=8)
    m = gh_cap8.new_matcher()
    assert isinstance(m, WhitespaceGuardedMatcher)

    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    sp_id = tok.encode(" ", add_special_tokens=False)[0]
    brace_l = tok.encode("{", add_special_tokens=False)[0]
    assert m.try_consume_tokens([brace_l]) == 1

    # Build adversarial logits strongly favoring '\n' and ' ', with moderate logit for '"ok":true}'
    vocab_n = len(gh_cap8.new_to_old)
    adv_logits = np.full((vocab_n,), -5.0, dtype=np.float32)
    adv_logits[nl_id] = 100.0
    adv_logits[sp_id] = 90.0

    # Consume whitespace tokens until the cap triggers; no more than 8 consecutive whitespace chars allowed!
    ws_emitted = 0
    forced_non_ws = False
    for _ in range(16):
        tid, _ = gh_cap8.pick_valid_token(m, adv_logits, int(nl_id))
        piece = tok.llguidance_tokenizer().decode_bytes([tid])
        if piece and all(b in (0x20, 0x09, 0x0A, 0x0D) for b in piece):
            ws_emitted += len(piece)
            assert m.consume_token(tid)
        else:
            forced_non_ws = True
            assert m.consume_token(tid)
            break

    assert forced_non_ws, "WhitespaceGuard failed to stop adversarial whitespace loop!"
    assert ws_emitted <= 8, f"Emitted {ws_emitted} whitespace chars, exceeding cap=8"
    assert m.whitespace_guard.run == 0

    # Also verify normal pretty-printed JSON well within the limit passes identically!
    pretty = json.dumps({"ok": True, "note": "a   b   c   d   e   f   g"}, indent=2)
    m_ok = gh_flex.new_matcher()
    for tid in tok.encode(pretty, add_special_tokens=False):
        assert m_ok.validate_tokens([tid])
        assert m_ok.consume_token(tid)
    assert m_ok.is_accepting()

    # Rollback on WhitespaceGuardedMatcher must roll back BOTH the inner matcher and the whitespace guard.
    m_rb = gh_cap8.new_matcher()
    assert isinstance(m_rb, WhitespaceGuardedMatcher)
    assert m_rb.try_consume_tokens([brace_l, sp_id, nl_id, sp_id]) == 4
    assert m_rb.whitespace_guard.run == 3
    m_rb.rollback(2)
    assert m_rb.whitespace_guard.run == 1
    m_rb.rollback(1)
    assert m_rb.whitespace_guard.run == 0

    # Strict compact mode (`whitespace_flexible=False`): rejects leading whitespace before `{`, strips via upfront_ff.
    assert gh_compact.upfront_ff
    decoded_up = tok.decode(gh_compact.upfront_ff)
    assert decoded_up.lstrip() == decoded_up and decoded_up.startswith("{")
    fresh_compact_m = gh_compact._llguidance.LLMatcher(gh_compact.ltok, gh_compact.gram)
    assert fresh_compact_m.validate_tokens([sp_id]) == 0
    assert fresh_compact_m.validate_tokens([nl_id]) == 0
    assert fresh_compact_m.validate_tokens([brace_l]) == 1
