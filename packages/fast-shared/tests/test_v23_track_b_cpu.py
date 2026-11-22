from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_shared.coreai_pipeline import (
    assess_calibration_batch_diversity,
    evaluate_multi_batch_calibration_gate,
    run_coreai_opt_calibration_and_export,
)
from fast_shared.parquet_pipeline import (
    CorpusBackfillConfig,
    build_backfill_cli_parser,
    embed_parquet,
    resolve_embedding_device_mode,
    run_corpus_backfill,
)
from fast_shared.pipeline_llm import (
    ExtractRetryRequest,
    _normalize_batch_finish_reasons,
    _resolve_rolling_finish_reason,
    extract_jsonl,
    extract_parquet,
    run_deferred_retries,
    run_deferred_retries_async,
)
from fast_shared.tokenizer import BaseFastTokenizer


class _FakeLLMTokenizer:
    def __init__(self, vocab: dict[str, int], *, stop_token_ids: set[int] | None = None) -> None:
        self._tok_to_id = dict(vocab)
        self._id_to_tok = {v: k for k, v in vocab.items()}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.stop_token_ids = frozenset(stop_token_ids or {2})

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for idx, ch in enumerate(text):
            key = f"ch:{ch}"
            if key not in self._tok_to_id:
                self._tok_to_id[key] = 100 + len(self._tok_to_id)
                self._id_to_tok[self._tok_to_id[key]] = ch
            out.append(self._tok_to_id[key])
            if idx > 4096:
                break
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        parts: list[str] = []
        for tid in ids:
            if skip_special_tokens and int(tid) in self.stop_token_ids:
                continue
            parts.append(self._id_to_tok.get(int(tid), ""))
        return "".join(parts)


def _build_word_fast_tokenizer(tmp_path: Path, words: list[str]) -> BaseFastTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab: dict[str, int] = {
        "[PAD]": 0,
        "[CLS]": 101,
        "[SEP]": 102,
        "[UNK]": 100,
    }
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok_json = tmp_path / "tokenizer.json"
    tok.save(str(tok_json))
    return BaseFastTokenizer(tok_json, default_max_length=128)


def test_rolling_finish_reason_under_constrained_grammar_cpu() -> None:
    """Constrained grammars hitting max_tokens without stop tokens must report length."""
    stop_ids = frozenset({2})
    assert (
        _resolve_rolling_finish_reason(
            [10, 11, 12, 13],
            max_tokens=4,
            stop_ids=stop_ids,
            valid_json=True,
            valid_schema=True,
        )
        == "length"
    )
    assert (
        _resolve_rolling_finish_reason(
            [10, 11, 12, 2],
            max_tokens=4,
            stop_ids=stop_ids,
            valid_json=True,
            valid_schema=True,
        )
        == "stop"
    )

    gh = SimpleNamespace(upfront_ff=[50, 51])
    eng = SimpleNamespace(
        tokenizer=SimpleNamespace(stop_token_ids={2}, eos_token_id=2),
        get_grammar=lambda _spec: gh,
    )
    raw_items = [
        SimpleNamespace(
            text='{"a":1}',
            parsed={"a": 1},
            token_ids=[50, 51, 10, 11, 12, 13],
            prompt_tokens=20,
            completion_tokens=6,
            valid_json=True,
            valid_schema=True,
            finish_reason="stop",
        ),
        SimpleNamespace(
            text='{"a":2}',
            parsed={"a": 2},
            token_ids=[50, 51, 10, 11, 2],
            prompt_tokens=20,
            completion_tokens=5,
            valid_json=True,
            valid_schema=True,
            finish_reason="stop",
        ),
        SimpleNamespace(
            text='{"a":3}',
            parsed={"a": 3},
            token_ids=None,
            prompt_tokens=20,
            completion_tokens=4,
            valid_json=True,
            valid_schema=True,
            finish_reason="stop",
        ),
    ]
    fixed = _normalize_batch_finish_reasons(
        eng,
        raw_items,
        max_tokens=4,
        entry_schemas=['{"type":"object"}'] * 3,
    )
    assert fixed[0].finish_reason == "length"
    assert fixed[1].finish_reason == "stop"
    assert fixed[2].finish_reason == "stop"


def test_extract_parquet_rolling_finish_reason_and_swa_gating_cpu(tmp_path: Path) -> None:
    """Verify rolling finish_reason='length' and B_slots<=8 SWA=1024 unlock for n_docs>4."""
    schema_obj = {
        "type": "object",
        "properties": {"tag": {"type": "string"}},
        "required": ["tag"],
        "additionalProperties": False,
    }
    schema_str = json.dumps(schema_obj, sort_keys=True)
    vocab = {
        "[PAD]": 0,
        "<s>": 1,
        "</s>": 2,
        "{": 10,
        '"tag":"': 11,
        "ok": 12,
        '"}': 13,
        "part": 14,
    }
    tok = _FakeLLMTokenizer(vocab)
    tok._id_to_tok = {
        10: "{",
        11: '"tag":"',
        12: "ok",
        13: '"}',
        14: "part",
    }

    class _FakeRollingEngine:
        def __init__(self) -> None:
            self.tokenizer = tok
            self.model = SimpleNamespace(window_size=512)
            self.roll_calls: list[dict[str, Any]] = []

        def encode_prompts(self, prompts: list[str], **_kw: Any) -> list[list[int]]:
            out = []
            for idx, _p in enumerate(prompts):
                out.append([1] * 20 + [1000 + idx] + [14] * 459)
            return out

        def get_grammar(self, schema_spec: Any, **_kw: Any) -> Any:
            return SimpleNamespace(
                upfront_ff=[10, 11],
                validate_token_sequence=lambda _ids: True,
            )

        def run_rolling(self, **kwargs: Any) -> dict[str, Any]:
            self.roll_calls.append(dict(kwargs))
            n = len(kwargs["suf_ids_list"])
            max_toks = int(kwargs["max_tokens"])
            gen_toks = []
            for i in range(n):
                if i % 2 == 0:
                    # Exactly max_tokens generated without stop token -> finish_reason="length"
                    gen_toks.append([12] * (max_toks - 1) + [13])
                else:
                    # Ended early on stop token -> finish_reason="stop"
                    gen_toks.append([12, 13, 2])
            return {
                "gen_toks": gen_toks,
                "valid_schemas": [True] * n,
                "tot_out_tok": sum(2 + len(g) for g in gen_toks),
                "decode_sec": 0.01,
            }

    in_6 = tmp_path / "in6.parquet"
    pq.write_table(
        pa.table({"id": list(range(6)), "body_text": [f"row {i}" for i in range(6)]}),
        in_6,
    )
    eng = _FakeRollingEngine()
    out_6 = tmp_path / "out6"
    extract_parquet(
        eng,
        in_6,
        out_6,
        system_prompt="Extract tag JSON.",
        json_schema=schema_str,
        batch_rows=6,
        shard_rows=6,
        max_tokens=80,
        use_rolling=True,
        roll_slots=6,
    )
    assert len(eng.roll_calls) == 1
    # n_docs=6 (>4) with eff_roll_slots=6 (<=8) unlocks swa_ring_cap=1024.
    assert eng.roll_calls[0]["swa_ring_cap"] == 1024
    tbl6 = pq.read_table(out_6 / "part-00000.parquet")
    assert tbl6.column("valid_schema").to_pylist() == [True] * 6
    assert tbl6.column("finish_reason").to_pylist() == ["length", "stop"] * 3

    # Larger active slot count (n_docs=12, roll_slots=12 > 8) stays at 512 unless explicitly overridden.
    in_12 = tmp_path / "in12.parquet"
    pq.write_table(
        pa.table({"id": list(range(12)), "body_text": [f"row {i}" for i in range(12)]}),
        in_12,
    )
    out_12 = tmp_path / "out12"
    extract_parquet(
        eng,
        in_12,
        out_12,
        system_prompt="Extract tag JSON.",
        json_schema=schema_str,
        batch_rows=12,
        shard_rows=12,
        max_tokens=80,
        use_rolling=True,
        roll_slots=12,
    )
    assert eng.roll_calls[1]["swa_ring_cap"] == 512

    out_12_explicit = tmp_path / "out12_exp"
    extract_parquet(
        eng,
        in_12,
        out_12_explicit,
        system_prompt="Extract tag JSON.",
        json_schema=schema_str,
        batch_rows=12,
        shard_rows=12,
        max_tokens=80,
        use_rolling=True,
        roll_slots=12,
        swa_ring_cap=1024,
    )
    assert eng.roll_calls[2]["swa_ring_cap"] == 1024


def test_single_model_inline_deferred_and_async_retry_hooks_cpu(tmp_path: Path) -> None:
    """Verify inline retries, deferred collection, and async single-model retry hooks."""
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"val": {"type": "string"}},
            "required": ["val"],
        }
    )
    in_pq = tmp_path / "docs.parquet"
    pq.write_table(
        pa.table({"id": [0, 1], "body_text": ["doc-alpha", "doc-beta"]}),
        in_pq,
    )

    class _RetryEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.tokenizer = SimpleNamespace(stop_token_ids={2}, eos_token_id=2)

        def generate_batch(self, prompts: list[str], **kwargs: Any) -> SimpleNamespace:
            self.calls.append({"prompts": list(prompts), **kwargs})
            max_toks = int(kwargs.get("max_tokens", 16))
            items = []
            for p in prompts:
                if "doc-alpha" in p and max_toks <= 16:
                    items.append(
                        SimpleNamespace(
                            text='{"val":"trunc',
                            parsed=None,
                            token_ids=list(range(10, 10 + max_toks)),
                            prompt_tokens=8,
                            completion_tokens=max_toks,
                            valid_json=False,
                            valid_schema=False,
                            finish_reason="length",
                        )
                    )
                else:
                    val = "alpha-retry" if "RETRY:" in p else ("alpha" if "doc-alpha" in p else "beta")
                    txt = json.dumps({"val": val})
                    items.append(
                        SimpleNamespace(
                            text=txt,
                            parsed={"val": val},
                            token_ids=[10, 11, 2],
                            prompt_tokens=8,
                            completion_tokens=3,
                            valid_json=True,
                            valid_schema=True,
                            finish_reason="stop",
                        )
                    )
            return SimpleNamespace(
                items=items,
                stats={
                    "docs": len(items),
                    "valid_json_count": sum(1 for it in items if it.valid_json),
                    "valid_schema_count": sum(1 for it in items if it.valid_schema),
                    "prompt_tokens": sum(it.prompt_tokens for it in items),
                    "completion_tokens": sum(it.completion_tokens for it in items),
                    "decode_sec": 0.01,
                },
            )

    eng = _RetryEngine()
    seen_sync_hooks: list[int] = []
    seen_async_hooks: list[int] = []

    async def _async_hook(req: ExtractRetryRequest) -> None:
        seen_async_hooks.append(req.row_index)

    out_inline = tmp_path / "inline_out"
    summary = extract_parquet(
        eng,
        in_pq,
        out_inline,
        system_prompt="Extract value JSON.",
        json_schema=schema,
        batch_rows=2,
        shard_rows=2,
        max_tokens=16,
        max_retries=1,
        retry_max_tokens=64,
        retry_prompt_builder=lambda req: f"RETRY:{req.prompt}",
    )
    assert summary.retries_attempted == 1
    assert summary.retries_succeeded == 1
    assert summary.valid_schema == 2
    tbl = pq.read_table(out_inline / "part-00000.parquet")
    assert tbl.column("finish_reason").to_pylist() == ["stop", "stop"]
    assert json.loads(tbl.column("response_text").to_pylist()[0])["val"] == "alpha-retry"

    # Deferred retry + sync/async per-row hooks + async execution across extract_parquet and extract_jsonl
    deferred_pq: list[ExtractRetryRequest] = []
    out_def = tmp_path / "deferred_out"
    sum_def = extract_parquet(
        eng,
        in_pq,
        out_def,
        system_prompt="Extract value JSON.",
        json_schema=schema,
        batch_rows=2,
        shard_rows=2,
        max_tokens=16,
        defer_retries=True,
        deferred_retries=deferred_pq,
        retry_hook=lambda req: seen_sync_hooks.append(req.row_index),
        async_retry_hook=_async_hook,
    )
    assert sum_def.retries_attempted == 0
    assert seen_sync_hooks == [0]
    assert seen_async_hooks == [0]
    assert len(deferred_pq) == 1
    assert deferred_pq[0].row_index == 0
    assert deferred_pq[0].finish_reason == "length"

    sync_res = run_deferred_retries(
        eng,
        deferred_pq,
        retry_max_tokens=64,
        retry_prompt_builder=lambda req: f"RETRY:{req.prompt}",
    )
    assert len(sync_res) == 1 and sync_res[0]["improved"] is True
    assert sync_res[0]["valid_schema"] is True

    in_jl = tmp_path / "docs.jsonl"
    in_jl.write_text(
        json.dumps({"id": 0, "text": "doc-alpha"}) + "\n"
        + json.dumps({"id": 1, "text": "doc-beta"}) + "\n",
        encoding="utf-8",
    )
    out_jl = tmp_path / "out.jsonl"
    deferred_jl: list[ExtractRetryRequest] = []
    extract_jsonl(
        eng,
        in_jl,
        out_jl,
        system_prompt="Extract value JSON.",
        json_schema=schema,
        batch_size=2,
        max_tokens=16,
        defer_retries=True,
        deferred_retries=deferred_jl,
    )
    assert len(deferred_jl) == 1
    hooked_async: list[int] = []

    async def _on_done(res: list[dict[str, Any]]) -> None:
        hooked_async.append(len(res))

    async_res = asyncio.run(
        run_deferred_retries_async(
            eng,
            deferred_jl,
            retry_max_tokens=64,
            retry_prompt_builder=lambda req: f"RETRY:{req.prompt}",
            async_complete_hook=_on_done,
        )
    )
    assert hooked_async == [1]
    assert async_res[0]["improved"] is True
    assert json.loads(async_res[0]["text"])["val"] == "alpha-retry"


def test_resolve_embedding_device_mode_pinned_cutoffs_cpu() -> None:
    """Verify non-hetero device cutoffs for mode='auto' and mode=None on ANE engines."""
    ane_eng = SimpleNamespace(ane_available=True, ane_query_max_len=128)
    gpu_only_eng = SimpleNamespace(ane_available=False, ane_query_max_len=128)
    dummy_eng = SimpleNamespace()

    short_small = [[1, 2, 3, 4] for _ in range(4)]  # B=4 (<12), T=16 (<1024), max_len=4 (<=128)
    long_small = [list(range(150)) for _ in range(2)]  # min_len=150 (>128)
    short_big_n = [[1, 2, 3, 4] for _ in range(12)]  # B=12 (>=12) but T=48 (<1024)
    big_batch = [list(range(100)) for _ in range(12)]  # B=12 (>=12) and T=1200 (>=1024)

    assert resolve_embedding_device_mode(ane_eng, short_small, mode="auto") == "ane"
    assert resolve_embedding_device_mode(ane_eng, short_small, mode=None) == "ane"
    assert resolve_embedding_device_mode(ane_eng, long_small, mode="auto") == "gpu"
    assert resolve_embedding_device_mode(gpu_only_eng, short_small, mode="auto") == "gpu"
    assert resolve_embedding_device_mode(ane_eng, short_big_n, mode="auto") == "ane"
    assert resolve_embedding_device_mode(ane_eng, big_batch, mode="auto") == "hetero"
    assert resolve_embedding_device_mode(ane_eng, short_small, mode="gpu") == "gpu"
    assert resolve_embedding_device_mode(ane_eng, big_batch, mode="ane") == "ane"
    assert resolve_embedding_device_mode(dummy_eng, short_small, mode=None) is None
    assert (
        resolve_embedding_device_mode(
            ane_eng,
            short_small,
            mode="hetero",
            pin_non_hetero_cutoffs=True,
        )
        == "ane"
    )


def test_embed_parquet_sentence_snapping_and_header_anchoring_cpu(tmp_path: Path) -> None:
    """Sentence-snapping and header anchoring preserve source spans while prepending headers."""
    header = "TITLE: Orbital Protocol Agreement"
    s1 = "First clause defines telemetry cadence and station handoff."
    s2 = "Second clause sets verification limits during orbit phase."
    s3 = "Third clause mandates redundant archive retention."
    body = f"{header}\n{s1} {s2} {s3}"
    words = sorted(set(body.replace(".", " .").replace(":", " :").split()))
    tok = _build_word_fast_tokenizer(tmp_path, words)

    encoded_calls: list[tuple[str, list[list[int]]]] = []

    class _ChunkEmbedEngine:
        def __init__(self, tokenizer: BaseFastTokenizer) -> None:
            self.tokenizer = tokenizer
            self.dim = 4
            self.ane_available = True
            self.ane_query_max_len = 128

        def encode_tokens(
            self,
            tok_lists: list[list[int]],
            *,
            mode: str | None = None,
            **_kw: Any,
        ) -> np.ndarray:
            encoded_calls.append((str(mode), [list(x) for x in tok_lists]))
            n = len(tok_lists)
            out = np.zeros((n, 4), dtype=np.float32)
            out[:, 0] = 1.0
            out[:, 1] = np.arange(n, dtype=np.float32)
            return out

    in_pq = tmp_path / "sections.parquet"
    pq.write_table(pa.table({"doc_id": ["doc0"], "body_text": [body]}), in_pq)
    out_pq = tmp_path / "embedded.parquet"

    eng = _ChunkEmbedEngine(tok)
    summary = embed_parquet(
        eng,
        in_pq,
        out_pq,
        text_col="body_text",
        chunk=True,
        max_chunk_tokens=18,
        max_body_tokens=16,
        overlap=2,
        include_chunk_text=True,
        snap_sentences=True,
        snap_min_fraction=0.45,
        header_anchored=True,
        anchor_max_chars=120,
        anchor_max_tokens=10,
        mode="auto",
        batch_rows=1,
    )
    assert summary.rows_emitted >= 2
    tbl = pq.read_table(out_pq)
    c_starts = tbl.column("char_start").to_pylist()
    c_ends = tbl.column("char_end").to_pylist()
    chunk_txts = tbl.column("chunk_text").to_pylist()
    has_anc = tbl.column("has_header_anchor").to_pylist()

    # Source spans slice body identically without shifting offsets for prepended headers.
    for c_txt, c0, c1 in zip(chunk_txts, c_starts, c_ends):
        assert c_txt == body[c0:c1]
    assert has_anc[0] is False
    assert any(has_anc[1:])
    # First snapped boundary should terminate cleanly on sentence punctuation '.'.
    assert chunk_txts[0].rstrip().endswith(".")
    # Header token "Orbital" must appear in continuation chunks sent to encoder.
    orbital_id = tok._tok.token_to_id("Orbital")
    assert any(orbital_id in toks for toks in encoded_calls[0][1][1:])
    assert encoded_calls[0][0] == "ane"

    # CLI parser + CorpusBackfillConfig forward Objective 2 knobs cleanly.
    parser = build_backfill_cli_parser()
    ns = parser.parse_args(
        [
            "--input",
            str(in_pq),
            "--output",
            str(out_pq),
            "--snap-sentences",
            "--header-anchored",
            "--anchor-max-chars",
            "512",
            "--pin-non-hetero-cutoffs",
            "--non-hetero-min-batch",
            "16",
            "--non-hetero-min-tokens",
            "2048",
        ]
    )
    assert ns.snap_sentences is True
    assert ns.header_anchored is True
    assert ns.anchor_max_chars == 512
    assert ns.pin_non_hetero_cutoffs is True
    assert ns.non_hetero_min_batch == 16
    assert ns.non_hetero_min_tokens == 2048

    out_cfg_pq = tmp_path / "cfg_embedded.parquet"
    cfg = CorpusBackfillConfig(
        snap_sentences=True,
        header_anchored=True,
        anchor_max_chars=120,
        pin_non_hetero_cutoffs=True,
        non_hetero_min_batch=12,
        non_hetero_min_tokens=1024,
        batch_rows=1,
    )
    sum_cfg = run_corpus_backfill(
        eng,
        in_pq,
        out_cfg_pq,
        config=cfg,
        text_col="body_text",
        chunk=True,
        max_chunk_tokens=18,
        max_body_tokens=16,
        overlap=2,
        mode="auto",
        progress_every=0,
    )
    assert sum_cfg.rows_emitted >= 2 and out_cfg_pq.is_file()


def test_coreai_multibatch_diversity_gate_and_nonfinite_guard_cpu(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Verify CoreAI multi-batch diversity check, collapse detection, and nonfinite failure."""
    b_a = (np.ones((2, 4), dtype=np.float32),)
    b_b = (np.full((2, 4), 2.5, dtype=np.float32) * np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32),)
    b_c = (np.full((2, 4), -3.0, dtype=np.float32),)

    div_ok = assess_calibration_batch_diversity([b_a, b_b, b_c])
    assert div_ok["diverse"] is True
    assert div_ok["unique_batches"] == 3
    assert div_ok["all_batches_identical"] is False

    div_dup = assess_calibration_batch_diversity([b_a, b_a, b_a])
    assert div_dup["diverse"] is False
    assert div_dup["unique_batches"] == 1
    assert div_dup["all_batches_identical"] is True

    refs = [np.array([1.0, 2.0, 3.0]), np.array([2.0, -1.0, 0.5])]
    quants_ok = [np.array([1.0, 2.0, 3.01]), np.array([2.01, -1.0, 0.5])]
    gate_ok = evaluate_multi_batch_calibration_gate(
        refs,
        quants_ok,
        batch_diversity=div_ok,
        calib_batches=3,
        strict=True,
    )
    assert gate_ok["passed"] is True
    assert gate_ok["all_finite"] is True

    # Single-batch collapse on batch 1 fails strict gate.
    quants_bad = [np.array([1.0, 2.0, 3.0]), np.array([-2.0, 1.0, -0.5])]
    gate_bad = evaluate_multi_batch_calibration_gate(
        refs,
        quants_bad,
        batch_diversity=div_ok,
        calib_batches=3,
        strict=True,
    )
    assert gate_bad["passed"] is False
    assert any("min_cosine" in f for f in gate_bad["failures"])
    assert any("cosine_std" in f for f in gate_bad["failures"])

    # Duplicate calibration batches fail strict gate.
    gate_dup = evaluate_multi_batch_calibration_gate(
        refs,
        quants_ok,
        batch_diversity=div_dup,
        calib_batches=3,
        strict=True,
    )
    assert gate_dup["passed"] is False
    assert "calib_batches_not_diverse" in gate_dup["failures"]

    # Export gate must raise on non-finite quantized output when fail_on_nonfinite=True.
    pytest.importorskip("coreai_opt")
    torch = pytest.importorskip("torch")
    import fast_shared.coreai_pipeline as cp

    class _TinyMod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(32, 32)

        def forward(self, x: Any) -> Any:
            return self.proj(x)

    monkeypatch.setattr(cp, "_score_teacher_vs_quant", lambda _r, _q: (0.0, float("inf"), False))
    with pytest.raises(ValueError, match="nonfinite quantized output"):
        run_coreai_opt_calibration_and_export(
            module_factory=lambda: _TinyMod().eval(),
            sample_inputs_factory=lambda bs: (torch.randn(bs, 32),),
            model_name="tiny_nf",
            input_names=["x"],
            output_names=["y"],
            eval_inputs=[
                (torch.ones(1, 32) * 0.5,),
                (torch.linspace(-1.0, 1.0, 32).unsqueeze(0),),
            ],
            out_dir=tmp_path / "nonfinite_export",
            stem="tiny_nf",
            preset="w8",
            export_backend="coreai",
            calib_batches=2,
            fail_on_nonfinite=True,
        )
