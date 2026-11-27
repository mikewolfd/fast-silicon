"""Deterministic CPU tests for BM25Index, rrf_fuse, and HybridRetrievalPipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pytest

from fast_shared import (
    BM25Index,
    CompactRetrievalIndex,
    ENGLISH_STOPWORDS,
    HybridRetrievalPipeline,
    LayaRerankFusion,
    rrf_fuse,
    rrf_fuse_batch,
    tokenize_lexical,
)


class _FakeLayaBackend:
    """Deterministic test double matching LayaAneBackend / LayaHeteroEngine noul API."""

    def __init__(self, score_fn: Any) -> None:
        self._score_fn = score_fn
        self.calls: list[list[dict[str, Any]]] = []
        self.closed: bool = False

    def encode_items_for_state(
        self,
        state: Any,
        qdefs: Sequence[Mapping[str, Any]],
        *,
        max_len: int | None = None,
        on_truncate: str = "ignore",
    ) -> list[dict[str, Any]]:
        _ = (max_len, on_truncate)
        encoded: list[dict[str, Any]] = []
        for qd in qdefs:
            assert qd["type"] == "noul"
            assert isinstance(qd["instructions"], str) and qd["instructions"]
            assert isinstance(qd["criteria"], Mapping)
            encoded.append({"state": state, "qdef": dict(qd)})
        self.calls.append(encoded)
        return encoded

    def score_noul_batch(self, encoded_items: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(self._score_fn(it["state"], it["qdef"])) for it in encoded_items]

    def close(self) -> None:
        self.closed = True


@dataclass
class _DummyEncoder:
    dim: int = 64
    is_mrl_trained: bool = False

    def encode_documents(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(sum(ord(ch) for ch in text) + i + 7)
            vec = rng.standard_normal((self.dim,)).astype(np.float32)
            out[i] = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return out


def test_tokenize_lexical_and_bm25_idf_weighting() -> None:
    tokens = tokenize_lexical("Section 12.3(a): NDA_clause, foo_bar! Café?")
    assert tokens == ["section", "12", "3", "a", "nda", "clause", "foo", "bar", "café"]

    filtered = tokenize_lexical(
        "The NDA and 12a",
        stopwords=frozenset({"the", "and"}),
        min_token_len=3,
    )
    assert filtered == ["nda", "12a"]

    docs = [
        "contract termination notice period",
        "contract renewal notice",
        "contract liability cap indemnification clause",
        "contract governing law jurisdiction",
    ]
    bm25 = BM25Index(docs, k1=1.5, b=0.75)
    assert bm25.num_docs == 4
    assert bm25.num_windows == 4
    assert bm25.avgdl == pytest.approx(16.0 / 4.0)

    # Exact Lucene BM25 IDF check: log(1 + (N - df + 0.5) / (df + 0.5))
    expected_idf_contract = math.log1p((4.0 - 4.0 + 0.5) / (4.0 + 0.5))
    expected_idf_indem = math.log1p((4.0 - 1.0 + 0.5) / (1.0 + 0.5))
    assert bm25.get_idf("Contract") == pytest.approx(expected_idf_contract, rel=1e-5)
    assert bm25.get_idf("indemnification") == pytest.approx(expected_idf_indem, rel=1e-5)
    assert bm25.get_idf("unseen_token") == 0.0
    assert bm25.get_idf("indemnification") > bm25.get_idf("contract")

    # Hand-verify BM25 score for single rare term on doc 2 (len=5, avgdl=4.0, tf=1)
    dl_norm_2 = (1.0 - 0.75) + 0.75 * (5.0 / 4.0)
    expected_sc_2 = expected_idf_indem * ((1.0 * 2.5) / (1.0 + 1.5 * dl_norm_2))
    scores = bm25.score_all("indemnification")
    assert scores.shape == (4,)
    assert scores.dtype == np.float32
    assert float(scores[2]) == pytest.approx(expected_sc_2, rel=1e-5)
    assert float(scores[0]) == 0.0

    # Query term repetition scales proportionally
    scores_rep = bm25.score_all("indemnification indemnification")
    assert float(scores_rep[2]) == pytest.approx(2.0 * expected_sc_2, rel=1e-5)

    # Rare-term overlap beats common-term overlap
    idx, sc = bm25.search("contract indemnification", k=2)
    assert int(idx[0]) == 2
    assert float(sc[0]) > float(sc[1])


def test_bm25_multiwindow_aggregation_and_positive_only() -> None:
    windows = [
        "boilerplate preamble agreement parties",
        "force majeure excuse delay natural disaster",
        "boilerplate preamble agreement parties",
        "confidentiality survival five years post termination",
    ]
    bm25 = BM25Index(windows, doc_offsets=[0, 2, 4], agg="maxsim")
    assert bm25.num_windows == 4
    assert bm25.num_docs == 2

    idx_max, sc_max = bm25.search("confidentiality survival", k=2, positive_only=True)
    assert list(idx_max) == [1, -1]
    assert float(sc_max[0]) > 0.0
    assert float(sc_max[1]) < -1e20

    # Batch search shape & dtype
    b_idx, b_sc = bm25.search(["force majeure", "confidentiality"], k=2)
    assert b_idx.shape == (2, 2)
    assert b_sc.shape == (2, 2)
    assert b_idx.dtype == np.int32
    assert b_sc.dtype == np.float32
    assert list(b_idx[:, 0]) == [0, 1]


def test_rrf_fuse_rank_math_weights_and_edge_cases() -> None:
    fused = rrf_fuse(
        [
            [10, 20, 30, -1, 20],  # -1 sentinel ignored; duplicate 20 keeps rank 2
            [30, 10, 40],
        ],
        rrf_k=60.0,
        weights=[1.0, 2.0],
    )
    by_id = dict(fused)
    assert by_id[30] == pytest.approx(1.0 / 63.0 + 2.0 / 61.0)
    assert by_id[10] == pytest.approx(1.0 / 61.0 + 2.0 / 62.0)
    assert by_id[20] == pytest.approx(1.0 / 62.0)
    assert by_id[40] == pytest.approx(2.0 / 63.0)
    assert [doc for doc, _ in fused] == [30, 10, 40, 20]

    # Symmetric ties break deterministically by best_rank then doc_id
    tied = rrf_fuse([[5, 2], [2, 5]], rrf_k=60.0)
    assert [doc for doc, _ in tied] == [2, 5]
    assert tied[0][1] == pytest.approx(tied[1][1])

    # String IDs + top_k truncation + batch helper
    str_fused = rrf_fuse([["doc_b", "doc_a"], ["doc_c", "doc_b"]], top_k=2)
    assert [doc for doc, _ in str_fused] == ["doc_b", "doc_c"]
    batch = rrf_fuse_batch([[[1, 2], [3, 4]], [[2, 1], [4, 3]]], top_k=1)
    assert [row[0][0] for row in batch] == [1, 3]

    # Empty edge cases and validation errors
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], [-1, -2]]) == []
    assert rrf_fuse([[1, 2]], top_k=0) == []
    assert rrf_fuse_batch([]) == []

    with pytest.raises(ValueError, match="rrf_k"):
        rrf_fuse([[1]], rrf_k=0.0)
    with pytest.raises(ValueError, match="weights length"):
        rrf_fuse([[1], [2]], weights=[1.0])
    with pytest.raises(ValueError, match="weights\\[0\\]"):
        rrf_fuse([[1]], weights=[-1.0])


def test_hybrid_pipeline_closes_recall_ndcg_gap_with_laya_reranking() -> None:
    rng = np.random.default_rng(123)
    n_docs = 30
    dim = 64

    # Construct corpus where doc 7 is the exact lexical + semantic match for query 0,
    # while doc 0 is a dense semantic distractor placed closer in embedding space.
    doc_texts = [
        f"General agreement obligations and standard commercial boilerplate clause {i}."
        for i in range(n_docs)
    ]
    doc_texts[0] = "General commercial obligations regarding payment schedule and termination notice."
    doc_texts[7] = (
        "Section 14.2 Change of Control: Licensee may terminate upon thirty days written notice "
        "following any merger or acquisition of Licensor."
    )
    doc_texts[12] = "Standard warranty disclaimer and limitation of consequential damages."

    # Also test same-document chunk labels (e.g. LegalBench-RAG multiple chunks per file)
    doc_ids = [f"cuad/ContractA.pdf#chunk{i}" for i in range(n_docs)]
    doc_titles = ["cuad/ContractA.pdf" for _ in range(n_docs)]

    doc_vecs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    q_vec = np.zeros((dim,), dtype=np.float32)
    q_vec[0] = 1.0

    # Distractors rank 1..6 in dense, while true target doc 7 sits at dense rank 8.
    for r, d_idx in enumerate([0, 1, 2, 3, 4, 5, 6]):
        v = np.zeros((dim,), dtype=np.float32)
        v[0] = 0.95 - 0.02 * r
        v[1 + r] = math.sqrt(max(0.0, 1.0 - v[0] ** 2))
        doc_vecs[d_idx] = v

    v7 = np.zeros((dim,), dtype=np.float32)
    v7[0] = 0.78
    v7[10] = math.sqrt(1.0 - 0.78**2)
    doc_vecs[7] = v7

    # Ensure remaining docs have low dense similarity
    for d_idx in range(8, n_docs):
        doc_vecs[d_idx, 0] = -0.3

    dense_idx = CompactRetrievalIndex(doc_vecs, mode="int8", use_neon=False)
    bm25_idx = BM25Index(doc_texts)

    # Fake Laya backend gives highest noul probability to "Section 14.2 Change of Control"
    def _laya_score(state: Mapping[str, str], qdef: Mapping[str, Any]) -> float:
        ins = qdef["instructions"]
        if "Section 14.2 Change of Control" in ins:
            return 0.96
        if "payment schedule" in ins:
            return 0.35
        return 0.12

    fake_laya = _FakeLayaBackend(_laya_score)
    pipeline = HybridRetrievalPipeline(
        dense_index=dense_idx,
        bm25_index=bm25_idx,
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        doc_titles=doc_titles,
        reranker=fake_laya,
        dense_top_k=15,
        bm25_top_k=15,
        rerank_top_k=10,
        rerank_prior_weight=0.20,
    )

    query = "Section 14.2 Change of Control merger termination notice"

    # Dense alone puts distractor doc 0 at rank 1 and target doc 7 at rank 8
    d_ids, _ = pipeline.search(query, q_vec, k=10, mode="dense")
    assert int(d_ids[0]) == 0
    assert 7 in d_ids.tolist() and d_ids.tolist().index(7) > 5

    # BM25 + dense RRF promotes doc 7 sharply
    h_ids, h_sc = pipeline.search(query, q_vec, k=10, mode="hybrid")
    assert h_ids.tolist().index(7) < d_ids.tolist().index(7)
    assert np.all(np.diff(h_sc) <= 1e-6)

    # Top-K Laya rerank promotes doc 7 to rank 1, and k > rerank_top_k keeps tail monotonic
    r_ids, r_sc = pipeline.search(query, q_vec, k=15, mode="hybrid_rerank", rerank_top_k=10)
    assert int(r_ids[0]) == 7
    assert len(fake_laya.calls) == 1
    assert len(fake_laya.calls[0]) == 10
    assert np.all(np.diff(r_sc) <= 1e-6)
    # Sorting by -r_sc preserves exact returned index order
    assert np.array_equal(r_ids[np.argsort(-r_sc, kind="mergesort")], r_ids)

    # dense_rerank mode reranks dense top-K directly with Laya
    dr_ids, dr_sc = pipeline.search(query, q_vec, k=10, mode="dense_rerank", rerank_top_k=10)
    assert int(dr_ids[0]) == 7
    assert np.all(np.diff(dr_sc) <= 1e-6)

    # ENGLISH_STOPWORDS drops function words while keeping legal negation/modal polarity terms
    sw_tokens = tokenize_lexical(
        "The Licensee shall not assign without prior notice",
        stopwords=ENGLISH_STOPWORDS,
    )
    assert sw_tokens == ["licensee", "shall", "not", "assign", "without", "prior", "notice"]

    # Detailed provenance check
    hits = pipeline.search_detailed(query, q_vec, k=5, mode="auto")
    assert isinstance(hits, list) and len(hits) == 5
    assert hits[0].doc_idx == 7
    assert hits[0].doc_id == "cuad/ContractA.pdf#chunk7"
    assert hits[0].bm25_rank == 1
    assert hits[0].laya_score == pytest.approx(0.96, abs=1e-5)
    assert hits[0].rerank_decision == "high_confidence"

    # External LayaRerankFusion with deduplicate_labels=True must NOT drop same-title chunks
    ext_reranker = LayaRerankFusion(
        fake_laya,
        max_rerank=10,
        prior_weight=0.1,
        deduplicate_labels=True,
        attach_score=True,
    )
    with HybridRetrievalPipeline(
        dense_index=dense_idx,
        bm25_index=bm25_idx,
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        doc_titles=doc_titles,
        reranker=ext_reranker,
        rerank_top_k=10,
    ) as ext_pipe:
        ext_ids, _ = ext_pipe.search(query, q_vec, k=10, mode="hybrid_rerank")
        assert int(ext_ids[0]) == 7
        assert len(set(ext_ids.tolist())) == 10
    assert fake_laya.closed is False

    # Factory-owned reranker IS closed on pipeline.close()
    factory_be = _FakeLayaBackend(_laya_score)
    with HybridRetrievalPipeline.from_corpus(
        doc_texts,
        encoder=_DummyEncoder(dim=64),
        doc_ids=doc_ids,
        reranker_factory=lambda: factory_be,
        rerank_top_k=8,
    ) as fac_pipe:
        assert fac_pipe.has_reranker
        q_enc = _DummyEncoder(dim=64).encode_documents([query])
        fac_ids, _ = fac_pipe.search([query], q_enc, k=5, mode="hybrid_rerank")
        assert fac_ids.shape == (1, 5)
    assert factory_be.closed is True


def test_hybrid_empty_oov_and_tie_edge_cases() -> None:
    # Empty corpus
    empty_dense = CompactRetrievalIndex(np.empty((0, 32), dtype=np.float32), mode="int8", use_neon=False)
    empty_bm25 = BM25Index([])
    empty_pipe = HybridRetrievalPipeline(dense_index=empty_dense, bm25_index=empty_bm25, doc_texts=[])
    q_vec = np.ones((32,), dtype=np.float32)
    ei, es = empty_pipe.search("test query", q_vec, k=4, mode="hybrid")
    assert ei.shape == (4,) and np.all(ei == -1)
    assert es.shape == (4,) and np.all(es < -1e20)

    # Empty batch
    bi, bs = empty_pipe.search([], np.empty((0, 32), dtype=np.float32), k=3, mode="hybrid")
    assert bi.shape == (0, 3) and bs.shape == (0, 3)

    # OOV / empty query in hybrid mode: BM25 contributes zero noise, pure dense order preserved
    docs = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    vecs = np.eye(4, 16, dtype=np.float32)
    dense = CompactRetrievalIndex(vecs, mode="fp32")
    bm25 = BM25Index(docs)
    pipe = HybridRetrievalPipeline(dense_index=dense, bm25_index=bm25, doc_texts=docs)

    q_v = vecs[2].copy()
    hi_oov, _ = pipe.search("nonexistent_oov_token", q_v, k=4, mode="hybrid")
    di_oov, _ = pipe.search("nonexistent_oov_token", q_v, k=4, mode="dense")
    assert np.array_equal(hi_oov, di_oov)

    # k > num_docs pads cleanly with -1 / -1e30
    hi_pad, hs_pad = pipe.search("", q_v, k=6, mode="hybrid")
    assert hi_pad.shape == (6,)
    assert list(hi_pad[4:]) == [-1, -1]
    assert np.all(hs_pad[4:] < -1e20)

    # Missing reranker raises clearly when hybrid_rerank explicitly requested
    with pytest.raises(ValueError, match="reranker"):
        pipe.search("alpha", q_v, k=2, mode="hybrid_rerank")


def test_hybrid_retrieval_cpu_latency_budget() -> None:
    rng = np.random.default_rng(2026)
    n_docs = 2000
    dim = 384
    n_queries = 50

    vocab = [f"term{i}" for i in range(400)]
    doc_texts = [
        " ".join(vocab[idx] for idx in rng.integers(0, len(vocab), size=48))
        for _ in range(n_docs)
    ]
    doc_vecs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    q_vecs = rng.standard_normal((n_queries, dim)).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_texts = [
        " ".join(vocab[idx] for idx in rng.integers(0, len(vocab), size=6))
        for _ in range(n_queries)
    ]

    t_build0 = time.perf_counter()
    pipe = HybridRetrievalPipeline.from_corpus(
        doc_texts,
        vectors=doc_vecs,
        index_mode="int8",
        use_neon=False,
        dense_top_k=64,
        bm25_top_k=64,
    )
    build_sec = time.perf_counter() - t_build0

    # Warmup single query then time 50-query batch
    _ = pipe.search(q_texts[0], q_vecs[0], k=10, mode="hybrid")
    t_search0 = time.perf_counter()
    top_ids, top_sc = pipe.search(q_texts, q_vecs, k=10, mode="hybrid")
    search_sec = time.perf_counter() - t_search0

    assert top_ids.shape == (n_queries, 10)
    assert top_sc.shape == (n_queries, 10)
    assert np.all(top_ids >= 0)
    assert build_sec < 2.0, f"Hybrid index build took {build_sec:.3f}s (> 2.0s budget)"
    assert search_sec < 0.75, f"50-query hybrid search took {search_sec:.3f}s (> 0.75s budget)"


def test_bm25_cpp_sparse_and_gated_bmw_parity() -> None:
    from fast_shared import detect_cpu_isa

    isa = detect_cpu_isa()
    if not isa.get("has_cxx", False):
        pytest.skip("C++17 compiler required for C++ BM25 backend parity test")

    rng = np.random.default_rng(4242)
    n_docs = 2000
    vocab = [f"term{i}" for i in range(600)]
    probs = 1.0 / (np.arange(1, len(vocab) + 1, dtype=np.float64) ** 1.15)
    probs /= probs.sum()
    docs = [
        " ".join(vocab[t] for t in rng.choice(len(vocab), size=18, p=probs))
        for _ in range(n_docs - 4)
    ]
    docs.extend(["", "term0 term1 term2", "term0 term1 term2", "term250 term399"])

    bm_np = BM25Index(docs, cpu_backend="numpy")
    bm_port = BM25Index(docs, cpu_backend="portable_cpp", max_threads=2)
    bm_auto = BM25Index(docs, cpu_backend="auto")
    assert bm_port.resolved_cpu_backend == "portable_cpp"
    assert bm_port._term_blk_offsets is not None
    assert bm_auto._term_blk_offsets is not None

    queries = [
        "",
        "nonexistent_oov_xyz",
        "term399 term0 term1",           # selective anchor -> BMW path
        "term0 term1 term2",             # high-DF only -> gated sparse path
        "term5 term5 term250 term399",   # repeated terms
        "term4",                         # single term
    ]

    def _bm25_lex_oracle(bm_obj: BM25Index, q_list: list[str], *, k_val: int, pos_only: bool, agg: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        all_sc = bm_obj.score_all(q_list, agg=agg)  # type: ignore[arg-type]
        qn = len(q_list)
        oi = np.full((qn, k_val), -1, dtype=np.int32)
        os_ = np.full((qn, k_val), -1e30, dtype=np.float32)
        dids = np.arange(bm_obj.num_docs, dtype=np.int32)
        for qi in range(qn):
            row = all_sc[qi]
            if pos_only:
                m = row > np.float32(0.0)
                sub_id, sub_sc = dids[m], row[m]
            else:
                sub_id, sub_sc = dids, row
            if sub_id.size == 0:
                continue
            take = min(k_val, int(sub_id.size))
            ord_i = np.lexsort((sub_id, -sub_sc))[:take]
            oi[qi, :take] = sub_id[ord_i]
            os_[qi, :take] = sub_sc[ord_i]
        return oi, os_

    for pos_only in (False, True):
        for k_val in (1, 10, 2500):
            ri, rs = _bm25_lex_oracle(bm_np, queries, k_val=k_val, pos_only=pos_only)
            for bm_cand in (bm_port, bm_auto):
                ci, cs = bm_cand.search(queries, k=k_val, positive_only=pos_only)
                assert np.array_equal(ci, ri)
                valid = ci >= 0
                if np.any(valid):
                    np.testing.assert_allclose(cs[valid], rs[valid], atol=2e-5)

    # Skewed multi-window doc_offsets with an empty middle document across all 4 aggregations
    offs = [0, 5, 5, 25, 180, 600, len(docs)]
    bm_mv_np = BM25Index(docs, doc_offsets=offs, cpu_backend="numpy")
    bm_mv_cpp = BM25Index(docs, doc_offsets=offs, cpu_backend="portable_cpp", max_threads=2)
    for agg in ("maxsim", "hybrid", "top2_mean", "mean"):
        for pos_only in (False, True):
            ri, rs = _bm25_lex_oracle(bm_mv_np, queries, k_val=4, pos_only=pos_only, agg=agg)
            ci, cs = bm_mv_cpp.search(queries, k=4, positive_only=pos_only, agg=agg)  # type: ignore[arg-type]
            assert np.array_equal(ci, ri)
            valid = ci >= 0
            if np.any(valid):
                np.testing.assert_allclose(cs[valid], rs[valid], atol=2e-5)

    # HybridRetrievalPipeline.from_corpus forwards cpu_backend + max_threads cleanly
    vecs = rng.standard_normal((len(docs), 32)).astype(np.float32)
    q_vecs = rng.standard_normal((len(queries), 32)).astype(np.float32)
    pipe_cpp = HybridRetrievalPipeline.from_corpus(
        docs,
        vectors=vecs,
        index_mode="int8",
        cpu_backend="portable_cpp",
        max_threads=2,
    )
    assert pipe_cpp.dense_index.resolved_cpu_backend == "portable_cpp"
    assert pipe_cpp.bm25_index.resolved_cpu_backend == "portable_cpp"
    hi, hs = pipe_cpp.search(queries, q_vecs, k=5, mode="hybrid")
    assert hi.shape == (len(queries), 5) and hs.shape == (len(queries), 5)
    assert np.all(hi >= 0)
