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
