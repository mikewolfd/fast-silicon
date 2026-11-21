"""Regression checks for the stock decision adapters and scored timing probes."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages" / "fast-decision-models" / "src"))

import _outcome_public_backends as public
from _kev_benchmark import _timed_scores
from fast_decision_models.kev_mlx import Kev08bMlxBackend


def test_stock_model_mapping_attribution_follows_hub_weight_links(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots" / "revision"
    snap.mkdir(parents=True)
    weight = tmp_path / "blobs" / "hash"
    weight.parent.mkdir()
    weight.touch()
    (snap / "model.safetensors").symlink_to(weight)
    monkeypatch.setattr(public, "resolve_hf_snapshot", lambda repo: snap)
    assert public.stock_decision_model_files("public-laya") == [weight]
    assert public.stock_decision_model_files("public-kev-0.8b") == [weight]
    assert public.stock_decision_model_files("laya-ane") == []


def test_matched_kev_loads_only_the_original_tokenizer_file(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    delimiters = ["<state>", "<q>", "<opt>", "<close>", "<decide>"]
    vocab = {token: i for i, token in enumerate(["<unk>", *delimiters, "é(s)"])}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.save(str(tmp_path / "tokenizer.json"))
    loaded, ids = public.load_kev_benchmark_tokenizer(tmp_path, delimiters)
    assert ids == list(range(1, 6))
    assert loaded.encode("é(s)").ids == [6]
    with pytest.raises(ValueError, match="five decision delimiters"):
        public.load_kev_benchmark_tokenizer(tmp_path, delimiters[:-1] + ["missing"])


def test_monorepo_compares_only_complete_kevs_on_the_same_batch_and_count():
    from benchmark_monorepo import extract_public_comparison

    project = {"status": "ok", "parsed_json": {"kev_08b": {"q_per_sec": 2, "batch_docs": 8, "n_questions": 2}}}
    stock = {"status": "ok", "runtime": "stock", "q_per_sec": 1, "highlight": "stock timing"}
    def compare():
        return extract_public_comparison("fast-decision-models", project, stock)

    assert compare() == "Full stock decision scores unavailable"
    stock.update(n_questions=2, b8_scores=[{"probabilities": [0, 1], "logits": []}] * 2)
    assert compare() == "Full stock decision scores unavailable"
    stock.update(n_questions=2, b8_scores=[{"probabilities": [0, 1], "logits": [-1, 1]}] * 2)
    assert "2.00x" in compare()
    project["parsed_json"]["kev_08b"]["batch_docs"] = 1
    assert compare() == "stock timing"


def test_tokenizer_only_kev_matches_project_for_all_question_types_and_clipping():
    class ByteTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=list(text.encode()))

    project = object.__new__(Kev08bMlxBackend)
    project.tok = ByteTokenizer()
    project.special_ids = [1000, 1001, 1002, 1003, 1004]
    project.max_state, project.max_branch = 1024, 2048
    stock = object.__new__(public.PublicKev08bMlxBackend)
    stock._tok, stock._special_ids = project.tok, project.special_ids

    qs = [
        {"type": "noul", "instructions": "Paid?", "criteria": {"true": "Paid", "false": "Overdue"}},
        {"type": "choice", "instructions": ["Select"], "criteria": {"one": "", "two": ["choice"]}},
        {"type": "choice", "instructions": "order?", "criteria": ["last", "first"]},
        {"type": "score", "instructions": "Severity", "criteria": ["low", "high"]},
        {"type": "noul", "instructions": "x" * 3000},
    ]
    for state in ("the <|fim_prefix|> sign", {"notes": ["a", 2], "details": {"state": True}}, "ü" * 3000):
        assert stock.encode_questions(state, qs) == project.encode_questions(state, qs)


def test_public_decision_adapters_delegate_to_laya_and_kev_helpers(monkeypatch):
    import fast_decision_models.kev_mlx as kev_mlx

    seen = {}

    def _spy_encode(tok, state, questions, *, special_ids=None, max_state_tokens=1024, max_branch_tokens=2048):
        seen["args"] = (tok, state, questions, special_ids, max_state_tokens, max_branch_tokens)
        return [1], [{"ids": [2, 3], "decide": 1, "opts": [0], "k": 1}]

    monkeypatch.setattr(kev_mlx, "encode_kev_questions", _spy_encode)
    tok = object()
    res = public.encode_kev_questions_pure(tok, [1, 2, 3, 4, 5], "st", [{"type": "noul"}], max_state=64, max_branch=128)
    assert res == ([1], [{"ids": [2, 3], "decide": 1, "opts": [0], "k": 1}])
    assert seen["args"] == (tok, "st", [{"type": "noul"}], [1, 2, 3, 4, 5], 64, 128)

    src = (ROOT / "scripts" / "_outcome_public_backends.py").read_text(encoding="utf-8")
    assert "object.__new__(LayaAneBackend)" not in src
    assert src.count("LayaAneBackend.from_tokenizer_only(") == 3


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "wrong_shape"])
def test_stock_laya_strictly_checks_every_inner_encoder_weight(tmp_path, monkeypatch, mutation):
    mx = pytest.importorskip("mlx.core")
    modernbert = pytest.importorskip("mlx_embeddings.models.modernbert")
    from mlx.utils import tree_flatten

    config = dict(model_type="modernbert", vocab_size=64, hidden_size=64, num_hidden_layers=1,
                  intermediate_size=64, num_attention_heads=1)
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text(json.dumps(config))
    (tmp_path / "rl_agent_config.json").write_text("{}")
    monkeypatch.setattr(public, "resolve_hf_snapshot", lambda repo: tmp_path)
    with mx.stream(mx.cpu):
        inner = modernbert.ModernBertModel(modernbert.ModelArgs.from_dict(config))
        raw = {"encoder." + name: weight for name, weight in tree_flatten(inner.parameters())}
        if mutation == "missing":
            del raw["encoder.final_norm.weight"]
        elif mutation == "unexpected":
            raw["encoder.unknown.weight"] = mx.ones(64)
        else:
            raw["encoder.final_norm.weight"] = mx.ones(8)
        monkeypatch.setattr(mx, "load", lambda path: raw)
        with pytest.raises(ValueError, match="(?i)missing|not in model|shape|unexpected|not found"):
            public.PublicLayaMlxBackend()


@pytest.mark.parametrize("scores", [[], [[([0.5, 0.5], [1])]], [[([0.2, 0.4], [1, 2])]], [[([float("nan"), 0], [1, 2])]]])
def test_timed_kev_probe_rejects_missing_or_invalid_full_scores(scores):
    scorer = SimpleNamespace(score_multi_states=lambda *args, **kwargs: scores)
    with pytest.raises(ValueError):
        _timed_scores(scorer, [("invoice", [{"type": "noul"}])], 1)


@pytest.mark.skipif(os.environ.get("FAST_SILICON_TEST_LOADED_DECISIONS") != "1",
                    reason="requires the staged public Laya checkpoint and the stock MLX environment")
def test_loaded_laya_matches_pytorch_option_order_and_probabilities():
    from fast_decision_models._common import acquire_hw_lock

    fixture = json.loads((ROOT / "tests/fixtures/laya_stock_reference.json").read_text())
    assert public.resolve_hf_snapshot(fixture["checkpoint"]).name == fixture["revision"]
    with acquire_hw_lock("laya_stock_reference"):
        scorer = public.PublicLayaMlxBackend()
        try:
            enc = [scorer.encode_item(fixture["states"][c["state"]], c["question"]) for c in fixture["cases"]]
            outputs = scorer.score_choice_batch(enc)
        finally:
            scorer.close()
    for case, row, actual in zip(fixture["cases"], enc, outputs, strict=True):
        digest = hashlib.sha256(json.dumps(row["ids"], separators=(",", ":")).encode()).hexdigest()
        assert (len(row["ids"]), digest, row["markers"]) == (case["num_tokens"], case["tokens_sha256"], case["markers"])
        expected = case["reference_probabilities"]
        assert sorted(range(len(actual)), key=actual.__getitem__) == sorted(range(len(expected)), key=expected.__getitem__)
        assert actual == pytest.approx(expected, abs=fixture["max_abs_probability_error"])


def test_public_laya_length_sort_restores_exact_order_and_256mb_cache_limit(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    backend = object.__new__(public.PublicLayaMlxBackend)
    backend._mx = mx
    backend._nn = nn
    backend.H = 8
    backend.nh = 2
    backend.hd = 4
    backend.scale = 0.5
    backend.type_emb_w = mx.zeros((3, 8), dtype=mx.bfloat16)
    backend.head_layers = []
    backend.sc_n_w = mx.ones((8,), dtype=mx.float32)
    backend.sc_n_b = mx.zeros((8,), dtype=mx.float32)
    backend.sc_1_w = mx.eye(8, dtype=mx.float32)
    backend.sc_1_b = mx.zeros((8,), dtype=mx.float32)
    backend.sc_3_w = mx.array([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    backend.sc_3_b = mx.zeros((1,), dtype=mx.float32)
    backend._qtype_map = {"choice": 0, "score": 1, "noul": 2}
    backend._tok_helper = SimpleNamespace(pad_id=0, resolve_temperature=lambda _qt, _k: 1.0)

    seen_batch_shapes: list[tuple[int, int]] = []

    def _fake_enc(ids_mx, attention_mask=None):
        del attention_mask
        seen_batch_shapes.append(tuple(ids_mx.shape))
        # Direction varies monotonically with token id so LayerNorm preserves strict per-item ordering
        B, L = ids_mx.shape
        x = ids_mx.astype(mx.float32)[:, :, None] * 0.05
        vals = mx.concatenate([mx.ones((B, L, 1), dtype=mx.float32), x * x, mx.zeros((B, L, 6), dtype=mx.float32)], axis=-1)
        return {"last_hidden_state": vals.astype(mx.bfloat16)}

    backend.encoder = _fake_enc

    # Alternating long and short items; bsz=2 after length-sort packs short(4,6) -> width 6 and long(30,32) -> width 32
    encs = [
        {"ids": [10] + [1] * 31, "markers": [0, 1], "qtype": 0},  # len 32
        {"ids": [20] + [1] * 3, "markers": [0, 1], "qtype": 0},   # len 4
        {"ids": [30] + [1] * 29, "markers": [0, 1], "qtype": 0},  # len 30
        {"ids": [40] + [1] * 5, "markers": [0, 1], "qtype": 0},   # len 6
    ]
    with mx.stream(mx.cpu):
        probs = backend._score_generic(encs, bsz=2)
    assert seen_batch_shapes == [(2, 6), (2, 32)]
    assert len(probs) == 4
    # Exact original order restored ([10, 20, 30, 40], NOT length-sorted [20, 40, 30, 10])
    assert probs[3][0] > probs[2][0] > probs[1][0] > probs[0][0] > 0.5
    src = Path(public.__file__).read_text(encoding="utf-8")
    assert "set_cache_limit(256 * 1024 * 1024)" in src

