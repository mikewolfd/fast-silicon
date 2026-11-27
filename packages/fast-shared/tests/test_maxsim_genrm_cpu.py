"""CPU unit tests for contextual sub-chunk MaxSim and GenRMRerankFusion in fast_shared."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fast_decision_models.genrm import GenRMScorer
from fast_shared import (
    DEFAULT_GENRM_RERANK_SYSTEM,
    BM25Index,
    GenRMRerankFusion,
    HybridRetrievalPipeline,
    build_contextual_subchunk_index,
    compute_margin_escalation_mask,
    split_subchunk_spans,
)
from fast_shared.retrieval_index import CompactRetrievalIndex


def _keyword_vec(text: str, *, dim: int = 8) -> np.ndarray:
    """Deterministic unit vector from keyword hashes for CPU retrieval tests."""
    vec = np.zeros((dim,), dtype=np.float32)
    for tok in text.lower().split():
        clean = "".join(ch for ch in tok if ch.isalnum())
        if not clean:
            continue
        bucket = sum((idx + 1) * ord(ch) for idx, ch in enumerate(clean)) % dim
        vec[bucket] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    else:
        vec[0] = 1.0
    return vec


class _DummyDocEncoder:
    def __init__(self, *, dim: int = 8) -> None:
        self.dim = dim
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def encode_documents(self, texts, *, mode=None):
        _ = mode
        batch = [str(t) for t in texts]
        self.doc_calls.append(batch)
        return np.vstack([_keyword_vec(t, dim=self.dim) for t in batch])

    def encode_queries(self, texts, *, mode=None):
        _ = mode
        batch = [str(t) for t in texts]
        self.query_calls.append(batch)
        return np.vstack([_keyword_vec(t, dim=self.dim) for t in batch])


class _FakeRawTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        _ = add_special_tokens
        return SimpleNamespace(ids=[idx + 1 for idx, _ in enumerate(str(text).split())])


class _FakeContextualTokenizer:
    def __init__(self) -> None:
        self._tokenizer = _FakeRawTokenizer()
        self.pack_calls: list[dict[str, object]] = []

    def _pack_context_windows(
        self,
        chunks,
        *,
        context_tokens: int = 4096,
        repeat_header_tokens: int | list[int] = 0,
    ):
        chunk_list = list(chunks)
        self.pack_calls.append(
            {
                "num_chunks": len(chunk_list),
                "context_tokens": context_tokens,
                "repeat_header_tokens": repeat_header_tokens,
            }
        )
        # Simulate prepending 1 repeated header window when repeat_header_tokens > 0.
        if repeat_header_tokens:
            all_chunks = ["[HEADER]"] + chunk_list
            emit_start = 1
        else:
            all_chunks = chunk_list
            emit_start = 0
        spans = [(i, i + 1) for i in range(len(chunk_list))]
        return [SimpleNamespace(chunks=all_chunks, emit_start=emit_start, spans=spans)]


class _DummyContextualEncoder(_DummyDocEncoder):
    def __init__(self, *, dim: int = 8) -> None:
        super().__init__(dim=dim)
        self.tokenizer = _FakeContextualTokenizer()
        self.contextual_calls: list[dict[str, object]] = []

    def encode_contextual_chunks(self, win_chunks, **kwargs):
        self.contextual_calls.append(
            {
                "win_chunks": [list(c) for c in win_chunks],
                "kwargs": dict(kwargs),
            }
        )
        out: list[np.ndarray] = []
        for chunks in win_chunks:
            out.append(np.vstack([_keyword_vec(c, dim=self.dim) for c in chunks]))
        return out


def test_split_subchunk_spans_basic_and_validation() -> None:
    long_doc = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon"
    )
    spans, offsets = split_subchunk_spans(["", "short note", long_doc], sub_chunk_chars=32)
    assert spans[0] == ""
    assert spans[1] == "short note"
    assert offsets[0] == 0 and offsets[1] == 1 and offsets[2] == 2
    assert offsets[3] == len(spans)
    assert offsets[3] >= 4  # long_doc split into >= 2 subchunks
    assert all(len(s) <= 36 for s in spans[2:])

    empty_spans, empty_offsets = split_subchunk_spans([], sub_chunk_chars=64)
    assert empty_spans == []
    assert empty_offsets.tolist() == [0]


def test_build_contextual_subchunk_index_generic_and_contextual() -> None:
    docs = [
        "Termination Clause. Landlord may terminate lease early on breach of covenant by written notice.",
        "Payment Schedule. Tenant owes base rent monthly plus quarterly tax adjustment and utility charges.",
    ]

    generic_enc = _DummyDocEncoder()
    generic_idx = build_contextual_subchunk_index(
        generic_enc,
        docs,
        doc_ids=["docA#0", "docB#0"],
        sub_chunk_chars=36,
        index_mode="fp32",
        use_neon=False,
    )
    assert isinstance(generic_idx, CompactRetrievalIndex)
    assert generic_idx.default_agg == "maxsim"
    assert generic_idx.num_docs == 2
    assert generic_idx.num_windows > 2
    assert len(generic_enc.doc_calls) == 1

    ctx_enc = _DummyContextualEncoder()
    ctx_idx = build_contextual_subchunk_index(
        ctx_enc,
        docs,
        doc_ids=["docA#0", "docA#1"],
        sub_chunk_chars=36,
        repeat_header_tokens=4,
        index_mode="fp32",
        use_neon=False,
    )
    assert ctx_idx.default_agg == "maxsim"
    assert ctx_idx.num_docs == 2
    assert ctx_idx.num_windows > 2
    assert len(ctx_enc.contextual_calls) == 1
    assert len(ctx_enc.tokenizer.pack_calls) == 1  # grouped under same doc prefix docA
    assert len(ctx_enc.tokenizer.pack_calls[0]["repeat_header_tokens"]) == 4


def test_compute_margin_escalation_mask_edge_cases() -> None:
    scores = np.array(
        [
            [0.95, 0.90, 0.80, 0.70, 0.50],      # rank0 - rank4 = 0.45 > 0.125 -> confident
            [0.91, 0.90, 0.89, 0.88, 0.87],      # rank0 - rank4 = 0.04 <= 0.125 -> escalate
            [0.875, 0.85, 0.82, 0.80, 0.75],     # rank0 - rank4 = 0.125 <= 0.125 -> escalate
        ],
        dtype=np.float64,
    )
    mask, margins = compute_margin_escalation_mask(scores, margin_threshold=0.125, margin_gap_rank=4)
    assert mask.tolist() == [False, True, True]
    assert np.allclose(margins, np.array([0.45, 0.04, 0.125], dtype=np.float64), atol=1e-9)

    single_doc = np.array([[0.42]], dtype=np.float32)
    mask1, margins1 = compute_margin_escalation_mask(single_doc, margin_threshold=0.05, margin_gap_rank=4)
    assert mask1.tolist() == [False]
    assert margins1.tolist() == [0.0]

    empty_mask, empty_margins = compute_margin_escalation_mask(np.zeros((0, 3), dtype=np.float32))
    assert empty_mask.tolist() == []
    assert empty_margins.tolist() == []


def test_genrm_rerank_fusion_context_prompt_blending_and_lifecycle() -> None:
    seen_prompts: list[str] = []
    seen_sys: list[str | None] = []
    seen_suffix: list[str] = []
    closed_count = 0

    def _fake_logit_scorer(prompts, system_prompt, verdict_suffix, option_variants):
        _ = option_variants
        seen_prompts.extend(prompts)
        seen_sys.append(system_prompt)
        seen_suffix.append(verdict_suffix)
        out = []
        for p in prompts:
            if "Contract Beta" in p:
                out.append([0.92, 0.08])
            elif "Contract Gamma" in p:
                out.append([0.55, 0.45])
            else:
                out.append([0.12, 0.88])
        return out

    class _TrackedScorer(GenRMScorer):
        def close(self) -> None:
            nonlocal closed_count
            closed_count += 1
            super().close()

    fusion = GenRMRerankFusion(
        scorer_factory=lambda: _TrackedScorer(logit_scorer=_fake_logit_scorer),
        max_rerank=3,
        prior_weight=0.20,
    )
    assert not fusion.is_initialized

    cards = [
        {
            "id": "DOC_A#0",
            "label": "Contract Alpha",
            "header": "SECTION 1: DEFINITIONS",
            "prev_text": "Preceding recital sentence.",
            "text": "Generic boilerplate definition of affiliate.",
            "next_text": "Following indemnity reference.",
        },
        {
            "id": "DOC_B#1",
            "label": "Contract Beta",
            "header": "SECTION 4: EXCLUSIVITY",
            "prev_text": "Vendor agrees to supply hardware.",
            "text": "Exclusivity runs for thirty-six months.",
            "next_text": "Renewal requires written notice.",
        },
        {
            "id": "DOC_C#2",
            "label": "Contract Gamma",
            "header": "SECTION 4: EXCLUSIVITY",
            "text": "Partial exclusivity discount tier.",
        },
        {"id": "DOC_D#3", "label": "Contract Delta", "text": "Unexamined tail document."},
    ]

    reranked = fusion.rerank(query="How long is exclusivity?", candidates=cards, limit=4)
    assert fusion.is_initialized
    assert seen_sys == [DEFAULT_GENRM_RERANK_SYSTEM]
    assert seen_suffix == ["Relevant (Yes or No):"]
    assert "Document Header Anchor:\nSECTION 4: EXCLUSIVITY" in seen_prompts[1]
    assert "Context Immediately Before Target:\nVendor agrees to supply hardware." in seen_prompts[1]
    assert "[TARGET PASSAGE]\nExclusivity runs for thirty-six months.\n[/TARGET PASSAGE]" in seen_prompts[1]
    assert "Context Immediately After Target:\nRenewal requires written notice." in seen_prompts[1]

    assert [c["id"] for c in reranked] == ["DOC_B#1", "DOC_C#2", "DOC_A#0", "DOC_D#3"]
    assert reranked[0]["rerankDecision"] == "high_confidence"
    assert reranked[1]["rerankDecision"] == "disambiguate"
    assert reranked[2]["rerankDecision"] == "low_confidence"
    assert "genrmScore" not in reranked[3]

    fusion.close()
    assert not fusion.is_initialized
    assert closed_count == 1


def test_genrm_rerank_fusion_fallback_on_error() -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("genrm failed")

    strict_fusion = GenRMRerankFusion(scorer=_boom, fallback_on_error=False)
    with pytest.raises(RuntimeError, match="genrm failed"):
        strict_fusion.rerank(query="q", candidates=[{"id": "d0", "text": "alpha"}])

    safe_fusion = GenRMRerankFusion(scorer=_boom, fallback_on_error=True)
    out = safe_fusion.rerank(query="q", candidates=[{"id": "d0", "text": "alpha"}], limit=1)
    assert [c["id"] for c in out] == ["d0"]


def test_hybrid_pipeline_margin_maxsim_and_margin_rerank_routing() -> None:
    docs = [
        "Contract A. Alpha preamble before termination fee clause.",
        "Contract A. Landlord early termination penalty equals two months rent.",
        "Contract A. Security deposit refund happens within thirty days.",
        "Contract B. Standard parking space allocation for tenant vehicles.",
        "Contract C. Zephyr quantum widget warranty covers five full years.",
    ]
    doc_ids = ["ca#0", "ca#1", "ca#2", "cb#0", "cc#0"]
    doc_headers = [
        "MASTER LEASE AGREEMENT (CA)",
        "MASTER LEASE AGREEMENT (CA)",
        "MASTER LEASE AGREEMENT (CA)",
        "PARKING ADDENDUM (CB)",
        "ZEPHYR HARDWARE WARRANTY (CC)",
    ]

    dense_vecs = np.array(
        [
            [0.70, 0.70, 0.0, 0.0],
            [0.70, 0.70, 0.0, 0.0],
            [0.69, 0.71, 0.0, 0.0],
            [0.68, 0.72, 0.0, 0.0],
            [0.00, 0.00, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sub_vecs = np.array(
        [
            [0.90, 0.435, 0.0, 0.0],  # doc0 (~0.945 cosine to query1)
            [0.435, 0.90, 0.0, 0.0],  # doc1 sub0 (~0.942)
            [0.71, 0.70, 0.0, 0.0],   # doc1 sub1 (exact 1.000 MaxSim for query1; margin vs rank2 ~0.055 <= 0.25)
            [0.90, 0.435, 0.0, 0.0],  # doc2 (~0.945)
            [0.435, 0.90, 0.0, 0.0],  # doc3 (~0.942)
            [0.00, 0.00, 1.0, 0.0],   # doc4
        ],
        dtype=np.float32,
    )
    sub_offsets = np.array([0, 1, 3, 4, 5, 6], dtype=np.int32)

    dense_idx = CompactRetrievalIndex(dense_vecs, mode="fp32", use_neon=False)
    maxsim_idx = CompactRetrievalIndex(
        sub_vecs,
        doc_offsets=sub_offsets,
        mode="fp32",
        agg="maxsim",
        use_neon=False,
    )

    maxsim_call_counts: list[int] = []
    orig_search = maxsim_idx.search

    def _spy_search(queries, **kwargs):
        maxsim_call_counts.append(int(np.asarray(queries).shape[0]))
        return orig_search(queries, **kwargs)

    maxsim_idx.search = _spy_search  # type: ignore[method-assign]

    seen_rerank_prompts: list[str] = []
    rerank_scorer_closed = False

    def _rerank_logit_scorer(prompts, system_prompt, verdict_suffix, option_variants):
        _ = system_prompt, verdict_suffix, option_variants
        seen_rerank_prompts.extend(prompts)
        out = []
        for p in prompts:
            if "Document: ca#1\n" in p:
                out.append([0.96, 0.04])
            else:
                out.append([0.10, 0.90])
        return out

    class _TrackedGenRMScorer(GenRMScorer):
        def close(self) -> None:
            nonlocal rerank_scorer_closed
            rerank_scorer_closed = True
            super().close()

    pipeline = HybridRetrievalPipeline(
        dense_index=dense_idx,
        maxsim_index=maxsim_idx,
        bm25_index=BM25Index(docs),
        doc_texts=docs,
        doc_ids=doc_ids,
        doc_headers=doc_headers,
        reranker_factory=lambda: GenRMRerankFusion(
            scorer_factory=lambda: _TrackedGenRMScorer(logit_scorer=_rerank_logit_scorer),
            max_rerank=4,
            prior_weight=0.25,
        ),
        margin_threshold=0.25,
        margin_gap_rank=2,
        rerank_margin_threshold=0.25,
        neighbor_context_chars=120,
    )

    q_vecs = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],   # confident match on doc4 (margin = 1.0)
            [0.71, 0.70, 0.0, 0.0], # ambiguous tie across docs 0..3
        ],
        dtype=np.float32,
    )
    queries = ["zephyr warranty", "early termination penalty"]

    res_margin_ms = pipeline.search_detailed(queries, query_vecs=q_vecs, k=3, mode="margin_maxsim")
    assert maxsim_call_counts == [1]
    assert res_margin_ms[0][0].doc_id == "cc#0"
    assert res_margin_ms[1][0].doc_id == "ca#1"

    maxsim_call_counts.clear()
    res_margin_rerank = pipeline.search_detailed(
        queries, query_vecs=q_vecs, k=3, mode="margin_maxsim_rerank"
    )
    assert maxsim_call_counts == [2]
    assert res_margin_rerank[0][0].doc_id == "cc#0"
    assert res_margin_rerank[0][0].rerank_decision == "margin_bypass"
    assert res_margin_rerank[1][0].doc_id == "ca#1"
    assert res_margin_rerank[1][0].rerank_decision == "high_confidence"

    # Only the ambiguous query reached GenRM, and its prompt preserved sibling prev/next context + header.
    assert len(seen_rerank_prompts) == 4
    target_ca1_prompts = [p for p in seen_rerank_prompts if "Document: ca#1\n" in p]
    assert len(target_ca1_prompts) == 1
    assert "two months rent" in target_ca1_prompts[0]
    assert "MASTER LEASE AGREEMENT (CA)" in target_ca1_prompts[0]
    assert "Alpha preamble before termination fee clause." in target_ca1_prompts[0]
    assert "Security deposit refund happens within thirty days." in target_ca1_prompts[0]

    # Legacy modes keep working cleanly.
    dense_ids, _dense_sc = pipeline.search(queries, query_vecs=q_vecs, k=2, mode="dense")
    assert doc_ids[int(dense_ids[0, 0])] == "cc#0"

    pipeline.close()
    assert rerank_scorer_closed


def test_from_corpus_builds_subchunk_maxsim_index() -> None:
    docs = [
        "Agreement A. Notice period is fifteen days prior to renewal date.",
        "Agreement B. Indemnification cap equals twelve months of fees paid.",
    ]
    enc = _DummyDocEncoder(dim=8)
    pipe = HybridRetrievalPipeline.from_corpus(
        encoder=enc,
        doc_texts=docs,
        doc_ids=["d0", "d1"],
        sub_chunk_chars=32,
        index_mode="fp32",
        use_neon=False,
    )
    assert pipe.maxsim_index is not None
    assert pipe.maxsim_index.default_agg == "maxsim"
    assert pipe.maxsim_index.num_docs == 2
    q_vec = enc.encode_queries(["fifteen days renewal"])
    top_idx, _top_sc = pipe.search(["fifteen days renewal"], query_vecs=q_vec, k=2, mode="maxsim")
    assert top_idx.shape == (1, 2)
    assert int(top_idx[0, 0]) == 0


def test_margin_mask_nan_sentinels_and_singleton_rerank_escalation() -> None:
    scores = np.array(
        [
            [float("nan"), 0.90, 0.10, 0.05, 0.01],     # NaN row -> force escalate
            [0.85, -1e30, -1e30, -1e30, -1e30],          # only 1 valid hit -> force escalate
            [0.95, 0.94, 0.50, -1e30, -1e30],            # 3 valid hits -> margin 0.95 - 0.50 = 0.45 > 0.10
        ],
        dtype=np.float64,
    )
    mask, margins = compute_margin_escalation_mask(scores, margin_threshold=0.10, margin_gap_rank=4)
    assert mask.tolist() == [True, True, False]
    assert margins.tolist() == [0.0, 0.0, pytest.approx(0.45)]

    # 1-doc corpus in margin_maxsim_rerank must escalate to GenRM rather than bypass.
    one_idx = CompactRetrievalIndex(
        np.array([[1.0, 0.0]], dtype=np.float32),
        doc_offsets=np.array([0, 1], dtype=np.int32),
        mode="fp32",
        agg="maxsim",
        use_neon=False,
    )
    scorer = GenRMScorer(logit_scorer=lambda p, *_a: [[0.88, 0.12] for _ in p])
    one_docs = ["Single contract clause about notice period."]
    pipe = HybridRetrievalPipeline(
        dense_index=one_idx,
        bm25_index=BM25Index(one_docs),
        maxsim_index=one_idx,
        doc_texts=one_docs,
        doc_ids=["doc0"],
        reranker=GenRMRerankFusion(scorer=scorer, max_rerank=1),
        rerank_margin_threshold=0.05,
    )
    hits = pipe.search_detailed(
        ["notice period"],
        query_vecs=np.array([[1.0, 0.0]], dtype=np.float32),
        k=1,
        mode="margin_maxsim_rerank",
    )
    assert len(hits[0]) == 1
    assert hits[0][0].rerank_decision == "high_confidence"


def test_genrm_rerank_fusion_deduplicate_labels_and_engine_guard() -> None:
    seen_prompts: list[str] = []

    def _scorer(prompts, *_a):
        seen_prompts.extend(prompts)
        return [[0.2, 0.8], [0.9, 0.1]]

    fusion = GenRMRerankFusion(
        scorer=GenRMScorer(logit_scorer=_scorer),
        deduplicate_labels=True,
        max_rerank=2,
    )
    cands = [
        {"id": "c0", "label": "Master Lease", "text": "Boilerplate recital."},
        {"id": "c1", "label": "Master Lease", "text": "Key exclusivity clause."},
    ]
    ranked = fusion.rerank(query="exclusivity?", candidates=cands, limit=2)
    assert "Document: Master Lease [#c0]" in seen_prompts[0]
    assert "Document: Master Lease [#c1]" in seen_prompts[1]
    assert ranked[0]["id"] == "c1"
    assert ranked[0]["label"] == "Master Lease"

    gemma_like = SimpleNamespace(tokenizer=SimpleNamespace(channel_open_id=100), model=object())
    with pytest.raises(TypeError, match="tokenizer.encode_chat|Qwen-compatible"):
        GenRMRerankFusion._wrap_if_engine(gemma_like)


def test_dense_rerank_preserves_small_stage1_pool_k() -> None:
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(6, 4)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = CompactRetrievalIndex(
        vecs,
        doc_offsets=np.arange(7, dtype=np.int32),
        mode="fp32",
        agg="mean",
        use_neon=False,
    )
    observed_k: list[int] = []
    orig_search = idx.search

    def _wrapped_search(q, *, k, **kw):
        observed_k.append(int(k))
        return orig_search(q, k=k, **kw)

    idx.search = _wrapped_search  # type: ignore[method-assign]
    docs = [f"doc text {i}" for i in range(6)]
    pipe = HybridRetrievalPipeline(
        dense_index=idx,
        bm25_index=BM25Index(docs),
        doc_texts=docs,
        doc_ids=[f"d{i}" for i in range(6)],
        reranker=GenRMRerankFusion(
            scorer=GenRMScorer(logit_scorer=lambda p, *_a: [[0.8, 0.2] for _ in p]),
            max_rerank=2,
        ),
        rerank_top_k=2,
        margin_gap_rank=4,
    )
    pipe.search(["query"], query_vecs=vecs[:1], k=2, mode="dense_rerank")
    assert observed_k == [2]


def test_evaluate_legalbench_rankings_and_engine_build_index_subchunk_cpu() -> None:
    from fast_bge_small.engine import FastBgeSmall
    from fast_embeddinggemma_2.engine import FastEmbeddingGemma2
    from fast_shared.outcome_eval.retrieval import (
        CharSpanChunk,
        evaluate_legalbench_rankings,
        evaluate_legalbench_spans,
    )

    chunks = [
        CharSpanChunk(doc_id="d0", chunk_index=0, start_char=0, end_char=50, text="Termination requires thirty days written notice."),
        CharSpanChunk(doc_id="d1", chunk_index=0, start_char=0, end_char=40, text="Payment due net fifteen days from invoice."),
    ]
    queries = [{"id": "q0", "query": "notice?", "gold_spans": [("d0", 0, 50)]}]
    q_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    c_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ref = evaluate_legalbench_spans(queries, chunks, q_vecs, c_vecs, k=2)
    ranked = evaluate_legalbench_rankings(queries, chunks, np.array([[0, 1]], dtype=np.int32), k=2)
    assert ranked["span_ndcg@2"] == pytest.approx(ref["span_ndcg@2"], abs=1e-12)
    assert ranked["char_recall@2"] == pytest.approx(ref["char_recall@2"], abs=1e-12)

    docs = ["Alpha clause with enough characters to split cleanly across subchunks for testing."]

    emb2 = FastEmbeddingGemma2.__new__(FastEmbeddingGemma2)
    emb2.is_mrl_trained = True
    emb2.encode_documents = lambda txts, **_kw: np.vstack([_keyword_vec(t, dim=8) for t in txts])  # type: ignore[method-assign]
    idx_e = emb2.build_index(docs, sub_chunk_chars=32, mode="fp32")
    assert idx_e.default_agg == "maxsim" and idx_e.num_windows >= 2

    bge = FastBgeSmall.__new__(FastBgeSmall)
    bge.is_mrl_trained = False
    bge.encode_documents = lambda txts, **_kw: np.vstack([_keyword_vec(t, dim=8) for t in txts])  # type: ignore[method-assign]
    idx_b = bge.build_index(docs, sub_chunk_chars=32, mode="fp32")
    assert idx_b.default_agg == "maxsim" and idx_b.num_windows >= 2
