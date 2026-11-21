"""CPU unit tests and GPU parity/gate smoke test for opt-in Gemma-4 E2B MTP drafter."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import fast_gemma4
from fast_gemma4.cli import build_parser
from fast_gemma4.mtp_drafter import (
    DEFAULT_MTP_DRAFTER_REPO,
    MAX_MTP_BATCH_SIZE,
    Gemma4MtpDrafter,
    draft_scan_top_k,
    recommended_mtp_draft_k,
    resolve_mtp_drafter_dir,
    run_decode_mtp,
    should_use_mtp_drafter,
)

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def test_should_use_mtp_drafter_strict_batch_gate() -> None:
    sentinel = object()
    assert MAX_MTP_BATCH_SIZE == 4
    for b in (1, 2, 3, 4):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False) is True
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=True, apple10_nax=False) is True
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=False, apple10_nax=False) is False
        assert should_use_mtp_drafter(sentinel, True, b, 0.0, apple10_nax=False) is False
        assert should_use_mtp_drafter(sentinel, False, b, 0.2, apple10_nax=False) is False
        assert should_use_mtp_drafter(None, False, b, 0.0, apple10_nax=False) is False

    assert should_use_mtp_drafter(sentinel, False, 1, 0.0, apple10_nax=True) is True
    for b in (2, 3, 4):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=True) is False

    for b in (0, -1, 5, 8, 16, 64, 128):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False) is False
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=True, apple10_nax=False) is False
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=True) is False

    # Constrained (has_grammar=True) Apple9 cap: strictly 1 <= B <= 2 (B>=3 rolling dense wins); Apple10: B == 1.
    for b in (1, 2):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False, has_grammar=True) is True
    for b in (3, 4, 8):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, apple10_nax=False, has_grammar=True) is False
    assert should_use_mtp_drafter(sentinel, False, 1, 0.0, apple10_nax=True, has_grammar=True) is True
    assert should_use_mtp_drafter(sentinel, False, 2, 0.0, apple10_nax=True, has_grammar=True) is False


def test_recommended_mtp_draft_k_and_hard_gate() -> None:
    assert recommended_mtp_draft_k(1, has_grammar=False, apple10_nax=False) == 4
    assert recommended_mtp_draft_k(1, has_grammar=True, apple10_nax=False) == 3
    assert recommended_mtp_draft_k(1, has_grammar=False, apple10_nax=True) == 4
    assert recommended_mtp_draft_k(1, has_grammar=True, apple10_nax=True) == 3
    for b in (2, 3, 4):
        assert recommended_mtp_draft_k(b, has_grammar=False, apple10_nax=False) == 2
        assert recommended_mtp_draft_k(b, has_grammar=True, apple10_nax=False) == 2
        # Keep M = B * (K + 1) < 16 to avoid the 16<=M<=64 MPP tile cliff.
        assert b * (recommended_mtp_draft_k(b, has_grammar=True, apple10_nax=False) + 1) < 16
        with pytest.raises(ValueError, match="strictly gated"):
            recommended_mtp_draft_k(b, apple10_nax=True)

    for bad_b in (0, 5, 8, 32, 128):
        with pytest.raises(ValueError, match="strictly gated"):
            recommended_mtp_draft_k(bad_b, apple10_nax=False)
        with pytest.raises(ValueError, match="strictly gated"):
            recommended_mtp_draft_k(bad_b, apple10_nax=True)


def test_mtp_drafter_rejects_e4b_and_invalid_top_k() -> None:
    e4b_stub = SimpleNamespace(hidden_size=2560, n_kv_heads=2)
    with pytest.raises(ValueError, match="Gemma-4 E2B target"):
        Gemma4MtpDrafter(e4b_stub, local_files_only=True)  # type: ignore[arg-type]

    e2b_stub = SimpleNamespace(hidden_size=1536, n_kv_heads=1)
    with pytest.raises(ValueError, match="top_k must be positive"):
        Gemma4MtpDrafter(e2b_stub, top_k=0, local_files_only=True)  # type: ignore[arg-type]


def test_run_decode_mtp_rejects_batch_above_4() -> None:
    for bad_b in (0, 5, 8, 16):
        with pytest.raises(ValueError, match="strictly gated"):
            run_decode_mtp(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                frozenset(),
                None,
                mx.zeros((1, 1, 1)),
                mx.zeros((1,), dtype=mx.int32),
                {},
                bad_b,
            )


def test_draft_scan_top_k_exact_and_clustered() -> None:
    rng = np.random.default_rng(42)
    n_c, per_c, dim = 16, 8, 32
    vocab = n_c * per_c

    centroids_np = rng.normal(size=(n_c, dim)).astype(np.float32)
    centroids_np /= np.linalg.norm(centroids_np, axis=-1, keepdims=True)
    # Build tightly clustered embeddings around each centroid.
    embeds_clustered = (
        centroids_np[:, None, :] + 0.05 * rng.normal(size=(n_c, per_c, dim)).astype(np.float32)
    )
    order_np = rng.permutation(vocab).astype(np.int32)
    order_2d_np = order_np.reshape(n_c, per_c)

    # Full unpermuted table indexed by original token ID for exact reference argmax.
    full_w_np = np.zeros((vocab, dim), dtype=np.float32)
    for c in range(n_c):
        for j in range(per_c):
            full_w_np[order_2d_np[c, j]] = embeds_clustered[c, j]

    with mx.stream(mx.cpu):
        centroids_mx = mx.array(centroids_np, dtype=mx.bfloat16)
        embed_perm_mx = mx.array(embeds_clustered, dtype=mx.bfloat16)
        order_2d_mx = mx.array(order_2d_np, dtype=mx.int32)
        full_w_mx = mx.array(full_w_np, dtype=mx.bfloat16)

        # Queries aligned near 4 distinct centroids -> top_k=4 and full scan must agree exactly.
        q_np = centroids_np[[1, 5, 9, 13]] + 0.02 * rng.normal(size=(4, dim)).astype(np.float32)
        q_mx = mx.array(q_np, dtype=mx.bfloat16)

        ref_ids = mx.argmax(q_mx @ full_w_mx.T, axis=-1).astype(mx.int32)
        full_scan_ids = draft_scan_top_k(q_mx, centroids_mx, embed_perm_mx, order_2d_mx, top_k=n_c)
        topk_scan_ids = draft_scan_top_k(q_mx, centroids_mx, embed_perm_mx, order_2d_mx, top_k=4)
        mx.eval(ref_ids, full_scan_ids, topk_scan_ids)

    assert full_scan_ids.tolist() == ref_ids.tolist()
    assert topk_scan_ids.tolist() == ref_ids.tolist()


def test_cli_mtp_drafter_flags_preserve_positional_prompt() -> None:
    parser = build_parser()

    args_default = parser.parse_args(["generate", "Hello world"])
    assert args_default.mtp_drafter is False
    assert args_default.mtp_drafter_path is None
    assert args_default.prompt == "Hello world"

    args_flag = parser.parse_args(["generate", "--mtp-drafter", "Hello world"])
    assert args_flag.mtp_drafter is True
    assert args_flag.mtp_drafter_path is None
    assert args_flag.prompt == "Hello world"

    args_path = parser.parse_args(["generate", "--mtp-drafter-path", "/tmp/my-drafter", "Hello world"])
    assert args_path.mtp_drafter is False
    assert args_path.mtp_drafter_path == "/tmp/my-drafter"
    assert args_path.prompt == "Hello world"


def test_lazy_exports_in_fast_gemma4_init() -> None:
    assert fast_gemma4.MAX_MTP_BATCH_SIZE == 4
    assert fast_gemma4.DEFAULT_MTP_DRAFTER_REPO == DEFAULT_MTP_DRAFTER_REPO
    assert fast_gemma4.Gemma4MtpDrafter is Gemma4MtpDrafter
    assert fast_gemma4.should_use_mtp_drafter is should_use_mtp_drafter
    assert fast_gemma4.recommended_mtp_draft_k is recommended_mtp_draft_k
    assert fast_gemma4.DEFAULT_MAX_WHITESPACE_RUN == 32
    assert fast_gemma4.WhitespaceGuard is not None


def test_engine_mtp_drafter_default_auto_vs_explicit_disable_and_eager_cpu(tmp_path: Path) -> None:
    """Verify FastGemma4Engine defaults `mtp_drafter='auto'`, disables on None/False, and eagerly loads True/path."""
    import inspect
    from unittest.mock import patch

    import fast_gemma4.engine as fg4_engine
    import fast_shared.hw_check as fs_hw

    sig = inspect.signature(fg4_engine.FastGemma4Engine.__init__).parameters
    assert sig["mtp_drafter"].default == "auto"
    assert sig["ane_ring_variant"].default == "auto"

    class _DummyPaths:
        lib_ane_prefix = tmp_path / "missing.dylib"

        def resolve_ane_bundle(self, *_a: object, **_kw: object) -> None:
            return None

    class _DummyModel:
        hidden_size = 1536
        num_layers = 35
        n_kv_heads = 1
        first_kv_shared = 15
        window_size = 512
        kv_i8 = True
        model_dir = tmp_path / "ckpt"
        tokenizer_json = tmp_path / "tokenizer.json"

    hw_m4 = {
        "system": "Darwin",
        "machine": "arm64",
        "brand": "Apple M4 Pro",
        "gpu_family": "Apple9",
        "supports_apple10": False,
        "mem_gb": 48,
        "macos_version": "26.6",
    }
    eager_calls: list[object] = []

    def _fake_drafter_ctor(_model: object, drafter_ref: object, **_kw: object) -> object:
        eager_calls.append(drafter_ref)
        return SimpleNamespace(ref=drafter_ref)

    def _build(**kwargs: object) -> fg4_engine.FastGemma4Engine:
        with (
            patch.object(fs_hw, "probe_host_hardware", return_value=hw_m4),
            patch.object(fg4_engine, "_host_ram_gib", return_value=48.0),
            patch.object(fg4_engine, "resolve_paths", return_value=_DummyPaths()),
            patch.object(fg4_engine, "FastGemma4Model", return_value=_DummyModel()),
            patch.object(fg4_engine, "FastGemma4Tokenizer", return_value=object()),
            patch.object(fg4_engine, "Gemma4MtpDrafter", side_effect=_fake_drafter_ctor),
            patch.object(mx, "set_wired_limit", return_value=0, create=True),
            patch.object(mx, "device_info", return_value={"max_recommended_working_set_size": 0}, create=True),
        ):
            return fg4_engine.FastGemma4Engine(warmup=False, **kwargs)  # type: ignore[arg-type]

    # 1. Default (zero-arg) sets lazy auto mode with 0 eager loads.
    eng_default = _build()
    assert eng_default._mtp_drafter_auto is True
    assert eng_default.mtp_drafter is None
    assert eager_calls == []

    # 2. Explicit None and False disable MTP completely.
    for disabled_val in (None, False):
        eng_off = _build(mtp_drafter=disabled_val)
        assert eng_off._mtp_drafter_auto is False
        assert eng_off.mtp_drafter is None
        assert eng_off._has_mtp_drafter_candidate() is False
        assert eng_off._maybe_ensure_mtp_drafter(use_ane=False, batch_w=1, temperature=0.0) is None
    assert eager_calls == []

    # 3. Explicit True and Path eagerly construct Gemma4MtpDrafter.
    eng_true = _build(mtp_drafter=True)
    assert eng_true._mtp_drafter_auto is False
    assert eng_true.mtp_drafter is not None
    assert eager_calls == [DEFAULT_MTP_DRAFTER_REPO]

    custom_dir = tmp_path / "custom_drafter"
    eng_path = _build(mtp_drafter=custom_dir)
    assert eng_path._mtp_drafter_auto is False
    assert eng_path.mtp_drafter is not None
    assert eager_calls == [DEFAULT_MTP_DRAFTER_REPO, custom_dir]


def test_constrained_mtp_walker_dfs_prompt_lookup_and_rollback_invariance_cpu() -> None:
    """Verify `_ConstrainedMtpWalker` mixes FF reach, prompt-lookup, top-M DFS candidates and restores matcher state."""
    from fast_gemma4.mtp_drafter import _ConstrainedMtpWalker
    from fast_gemma4.speculative import build_ngram_table

    class _SeqGrammarMatcher:
        def __init__(self, target_seq: list[int], ff_map: dict[int, list[int]]) -> None:
            self._target = target_seq
            self._ff_map = ff_map
            self.pos = 0

        def compute_ff_tokens(self) -> list[int]:
            return list(self._ff_map.get(self.pos, []))

        def try_consume_tokens(self, toks: list[int]) -> int:
            cnt = 0
            for t in toks:
                if self.pos < len(self._target) and int(t) == self._target[self.pos]:
                    self.pos += 1
                    cnt += 1
                else:
                    break
            return cnt

        def rollback(self, count: int) -> None:
            self.pos = max(0, self.pos - int(count))

        def is_stopped(self) -> bool:
            return self.pos >= len(self._target)

        def is_accepting(self) -> bool:
            return self.pos == len(self._target)

    # Target sequence: [10(draft cand[1]), 11(FF), 12(FF), 20(prompt lookup after 11,12), 21(draft), 22(closing accept)]
    prompt = [1, 2, 11, 12, 20, 99, 100]
    t3, t2 = build_ngram_table(prompt)
    matcher = _SeqGrammarMatcher([10, 11, 12, 20, 21, 22], ff_map={1: [11, 12]})
    walker = _ConstrainedMtpWalker(matcher, vocab_size=128, stop_token_ids=frozenset({0}), t3=t3, t2=t2)

    # Note step 0 has invalid top-1 (55) and valid candidate (10); FF [11,12] auto-reaches; step 1 prompt lookup
    # recovers 20 even when drafter candidate lists don't contain 20!
    cands = [
        [55, 10, 77],
        [88, 21, 66],
        [99, 22, 44],
    ]
    path = walker.build_draft(cands, ctx=[5, 6], max_draft=6)
    assert path == [10, 11, 12, 20, 21, 22]
    # Exact rollback invariance: matcher position must remain 0 after DFS probing!
    assert matcher.pos == 0


def test_engine_per_doc_grammars_mtp_dispatch_gate_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `generate_batch(..., per_doc_grammars=...)` routes `B==1` Apple10 / `B<=4` Apple9 via MTP and keeps `B>=5` dense."""
    import fast_gemma4.engine as eng_mod
    from fast_gemma4.engine import FastGemma4Engine

    class _DummyMatcher:
        def __init__(self, tok_seq: list[int]) -> None:
            self._toks = tok_seq
            self._idx = 0

        def validate_tokens(self, toks: list[int]) -> int:
            return 1 if (self._idx < len(self._toks) and toks[0] == self._toks[self._idx]) else 0

        def consume_token(self, tok: int) -> bool:
            self._idx += 1
            return True

        def consume_tokens(self, toks: list[int]) -> bool:
            self._idx += len(toks)
            return True

        def compute_ff_tokens(self) -> list[int]:
            return []

        def is_stopped(self) -> bool:
            return self._idx >= len(self._toks)

        def is_accepting(self) -> bool:
            return self._idx >= len(self._toks)

        def pick_mask(self, _logits: np.ndarray) -> int:
            return int(self._toks[self._idx])

    class _StubGrammar:
        def __init__(self, toks: list[int]) -> None:
            self.upfront_ff: list[int] = []
            self._toks = toks

        def new_matcher(self) -> _DummyMatcher:
            return _DummyMatcher(self._toks)

        def validate_token_sequence(self, _full_ids: list[int]) -> bool:
            return True

    class _StubTok:
        bos_token_id = 2
        stop_token_ids = frozenset({1})

        def decode(self, _ids: list[int], skip_special_tokens: bool = True) -> str:
            return '{"ok":true}'

    class _StubModel:
        hidden_size = 1536
        n_kv_heads = 1
        window_size = 512

        def make_slot_cache(self, _pkv: dict, **_kw: object) -> dict:
            return {"_last_x_in": mx.zeros((1, 1, 4), dtype=mx.float32)}

        def prefill_slots_lazy_lg(self, sufs: list[list[int]], _slots: list[int], _cache: dict, **_kw: object) -> tuple:
            b = len(sufs)
            return (
                mx.zeros((b, 1, 4), dtype=mx.float32),
                mx.zeros((b,), dtype=mx.int32),
                [len(s) for s in sufs],
            )

    mtp_calls: list[dict[str, object]] = []
    roll_calls: list[dict[str, object]] = []

    def _fake_mtp(model, drafter, stop_ids, grammar, init_lg, init_off, cache, B, **kw):
        mtp_calls.append({"B": B, "per_doc_grammars": kw.get("per_doc_grammars"), "doc_prompts": kw.get("doc_prompts")})
        return {
            "decode_sec": 0.01,
            "gen_toks": [[10, 11] for _ in range(B)],
            "valid_schemas": [True] * B,
            "tot_out_tok": 2 * B,
            "ff_tokens": 0,
            "speculative_backend": "mtp",
            "mtp_rounds": 1,
            "mtp_accepted": B,
            "mtp_proposed": B,
        }

    def _fake_rolling(model, stop_ids, grammar, pkv, sufs, **kw):
        roll_calls.append({"n": len(sufs), "per_doc_grammars": kw.get("per_doc_grammars")})
        n = len(sufs)
        return {
            "prefill_sec": 0.01,
            "decode_sec": 0.01,
            "gen_toks": [[10, 11] for _ in range(n)],
            "valid_schemas": [True] * n,
            "tot_out_tok": 2 * n,
            "ff_tokens": 0,
            "llm_steps": 1,
            "ff_early14_steps": 0,
            "mask_fallbacks": 0,
        }

    monkeypatch.setattr(eng_mod, "run_decode_mtp", _fake_mtp)
    monkeypatch.setattr(eng_mod, "run_decode_rolling", _fake_rolling)

    import threading

    def _make_engine(apple10: bool) -> FastGemma4Engine:
        e = FastGemma4Engine.__new__(FastGemma4Engine)
        e.model = _StubModel()  # type: ignore[assignment]
        e.tokenizer = _StubTok()  # type: ignore[assignment]
        e.mtp_drafter = object()  # type: ignore[assignment]
        e._mtp_drafter_auto = False
        e._mtp_drafter_auto_attempted = True
        e.default_mode = "gpu"
        e.profile = "auto"
        e.ane_available = False
        e.default_max_b_slots = 64
        e._host_hw = {"supports_apple10": apple10, "gpu_family": "Apple10" if apple10 else "Apple9"}
        e._host_ram_gib = 24.0 if apple10 else 48.0
        e.silicon_policy = None
        e._gpu_lock = threading.Lock()
        e.kv_i8 = False
        e.resolve_call_kv_i8 = lambda **_kw: False  # type: ignore[assignment]
        e.encode_prompts = lambda ps, **_kw: [[2, 3, 4] for _ in ps]  # type: ignore[assignment]
        e.get_prefix_kv = lambda _sys: {"l_sys": 1}  # type: ignore[assignment]
        e._maybe_get_ane_ring = lambda *_a, **_kw: None  # type: ignore[assignment]
        e.resolve_runtime_ane_ring_variant = lambda *_a, **_kw: "fp16"  # type: ignore[assignment]
        e._effective_ane_ring_variant = lambda: "fp16"  # type: ignore[assignment]
        return e

    g_list_4 = [_StubGrammar([10, 11]) for _ in range(4)]
    # Apple10: B=1 uses MTP with per_doc_grammars; B=2..4 stays dense (rolling)!
    e_a10 = _make_engine(apple10=True)
    res_a10_b1 = e_a10.generate_batch(["p0"], per_doc_grammars=[g_list_4[0]], max_tokens=8, temperature=0.0)
    assert len(mtp_calls) == 1 and mtp_calls[0]["B"] == 1
    assert mtp_calls[0]["per_doc_grammars"] is not None
    assert res_a10_b1.stats.get("valid_schema_count") == 1

    res_a10_b4 = e_a10.generate_batch(["p"] * 4, per_doc_grammars=g_list_4, max_tokens=8, temperature=0.0)
    assert len(mtp_calls) == 1 and len(roll_calls) == 1 and roll_calls[0]["n"] == 4
    assert res_a10_b4.stats.get("valid_schema_count") == 4

    # Apple9 constrained: B=2 uses MTP with per_doc_grammars; B=4 and B=5 stay dense (rolling)!
    e_a9 = _make_engine(apple10=False)
    res_a9_b2 = e_a9.generate_batch(["p"] * 2, per_doc_grammars=g_list_4[:2], max_tokens=8, temperature=0.0)
    assert len(mtp_calls) == 2 and mtp_calls[1]["B"] == 2
    assert res_a9_b2.stats.get("valid_schema_count") == 2

    res_a9_b4 = e_a9.generate_batch(["p"] * 4, per_doc_grammars=g_list_4, max_tokens=8, temperature=0.0)
    assert len(mtp_calls) == 2 and len(roll_calls) == 2 and roll_calls[1]["n"] == 4
    assert res_a9_b4.stats.get("valid_schema_count") == 4

    res_a9_b5 = e_a9.generate_batch(["p"] * 5, per_doc_grammars=g_list_4 + [None], max_tokens=8, temperature=0.0)
    assert len(mtp_calls) == 2 and len(roll_calls) == 3 and roll_calls[2]["n"] == 5
    assert res_a9_b5.stats.get("valid_schema_count") == 5


def test_mtp_drafter_b1_parity_and_b8_gate_gpu() -> None:
    """Serialized M4 GPU smoke test: B=1 MTP parity (free-text + JSON) & B=8 hard gate."""
    try:
        drafter_dir = resolve_mtp_drafter_dir(DEFAULT_MTP_DRAFTER_REPO, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Local HF cache missing {DEFAULT_MTP_DRAFTER_REPO}: {exc}")
    if not (drafter_dir / "model.safetensors").is_file():
        pytest.skip(f"Missing model.safetensors in {drafter_dir}")

    from fast_gemma4.engine import FastGemma4Engine

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    schema_str = (fixtures_dir / "extract_v1.schema.json").read_text(encoding="utf-8")
    sys_prompt = (fixtures_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    docs_raw = json.loads((fixtures_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    doc_prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs_raw]

    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HW_LOCK_PATH.open("a+") as lock_f:
        acquired_lock = False
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired_lock = True
        except BlockingIOError:
            import os
            import subprocess

            anc: set[int] = set()
            pid = os.getpid()
            while pid > 1 and pid not in anc:
                anc.add(pid)
                ppid_s = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
                pid = int(ppid_s) if ppid_s else 0
            holders = {
                int(x)
                for x in subprocess.check_output(["lsof", "-t", str(HW_LOCK_PATH)], text=True).split()
                if x.strip().isdigit()
            }
            if not ((anc - {os.getpid()}) & holders):
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                acquired_lock = True
        # Use kv_i8=False so plain rolling (`speculative=False`) and MTP (`make_batch_cache`) share identical BF16 KV.
        eng = FastGemma4Engine(default_mode="gpu", kv_i8=False, mtp_drafter=drafter_dir, warmup=False)
        try:
            # 1) B=1 free-text parity: MTP greedy vs plain greedy.
            ft_prompt = "Explain in three concise bullet points how ring-buffer sliding-window KV caching bounds memory."
            ref_ft = eng.generate_batch([ft_prompt], max_tokens=64, temperature=0.0, mode="gpu", speculative=False)
            mtp_ft = eng.generate_batch([ft_prompt], max_tokens=64, temperature=0.0, mode="gpu", speculative=None)
            assert mtp_ft.stats.get("speculative_backend") == "mtp"
            assert mtp_ft.stats.get("mtp_accept_rate", 0.0) > 0.25
            assert mtp_ft.items[0].token_ids == ref_ft.items[0].token_ids
            assert mtp_ft.items[0].text == ref_ft.items[0].text

            # 2) B=1 constrained JSON parity + schema validity.
            ref_json = eng.generate_batch(
                [doc_prompts[0]],
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=256,
                temperature=0.0,
                mode="gpu",
                speculative=False,
            )
            mtp_json = eng.generate_batch(
                [doc_prompts[0]],
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=256,
                temperature=0.0,
                mode="gpu",
                speculative=None,
            )
            assert mtp_json.stats.get("speculative_backend") == "mtp"
            assert mtp_json.stats.get("valid_json_count") == 1
            assert mtp_json.stats.get("valid_schema_count") == 1
            assert mtp_json.stats.get("mtp_accept_rate", 0.0) > 0.45
            assert mtp_json.items[0].token_ids == ref_json.items[0].token_ids
            assert mtp_json.items[0].parsed == ref_json.items[0].parsed

            # 3) Hard B>=8 gate: even with mtp_drafter attached and speculative=None or True, B=8 NEVER uses MTP.
            res_b8_auto = eng.generate_batch(
                doc_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=32,
                temperature=0.0,
                mode="gpu",
                speculative=None,
            )
            assert res_b8_auto.stats.get("speculative_backend") != "mtp"
            assert res_b8_auto.stats.get("rolling") is True

            res_b8_spec = eng.generate_batch(
                doc_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=32,
                temperature=0.0,
                mode="gpu",
                speculative=True,
            )
            assert res_b8_spec.stats.get("speculative_backend") != "mtp"
        finally:
            eng.close()
            mx.clear_cache()
            if acquired_lock:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
