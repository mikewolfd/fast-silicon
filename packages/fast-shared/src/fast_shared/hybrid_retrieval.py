"""Pure-NumPy BM25 lexical index, Reciprocal Rank Fusion (RRF), and hybrid retrieval + Laya reranking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any, Literal, TypeVar

import numpy as np

from .refspec_bridge import LayaRerankFusion
from .retrieval_index import (
    VALID_WINDOW_AGGS,
    CompactRetrievalIndex,
    IndexMode,
    WindowAgg,
    build_engine_index,
    score_document_windows,
)

RetrievalMode = Literal["auto", "dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
VALID_RETRIEVAL_MODES: frozenset[str] = frozenset(
    {"auto", "dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"}
)

# High-frequency function words only; deliberately retains negation and modal terms
# ("no", "not", "without", "unless", "shall", "must") that govern legal/argument polarity.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
    }
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_DocKey = TypeVar("_DocKey")


def tokenize_lexical(
    text: str,
    *,
    lowercase: bool = True,
    stopwords: frozenset[str] | None = None,
    min_token_len: int = 1,
) -> list[str]:
    """Tokenize text into alphanumeric terms for BM25 lexical matching."""
    raw = str(text or "")
    if not raw:
        return []
    prepared = raw.lower() if lowercase else raw
    tokens = _TOKEN_RE.findall(prepared)
    if stopwords is None and min_token_len <= 1:
        return tokens
    eff_min = max(1, int(min_token_len))
    if stopwords is None:
        return [tok for tok in tokens if len(tok) >= eff_min]
    return [tok for tok in tokens if len(tok) >= eff_min and tok not in stopwords]


def _validate_finite_nonneg(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return val


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _normalize_doc_key(raw: Any) -> Any:
    if isinstance(raw, np.integer) and not isinstance(raw, (bool, np.bool_)):
        return int(raw)
    if isinstance(raw, np.str_):
        return str(raw)
    return raw


class BM25Index:
    """Fast CSR-packed NumPy Okapi/Lucene BM25 lexical index with optional multi-window aggregation."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
        lowercase: bool = True,
        stopwords: Iterable[str] | None = None,
        min_token_len: int = 1,
    ) -> None:
        eff_k1 = _validate_finite_nonneg(k1, "k1")
        eff_b = _validate_finite_nonneg(b, "b")
        if eff_b > 1.0:
            raise ValueError(f"b must be in [0.0, 1.0], got {b!r}")
        if agg not in VALID_WINDOW_AGGS:
            raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")
        eff_alpha = float(hybrid_alpha)
        if not math.isfinite(eff_alpha):
            raise ValueError(f"hybrid_alpha must be finite, got {hybrid_alpha!r}")
        eff_min_len = _validate_positive_int(min_token_len, "min_token_len")

        self.k1: float = eff_k1
        self.b: float = eff_b
        self.default_agg: WindowAgg = agg
        self.default_hybrid_alpha: float = eff_alpha
        self.lowercase: bool = bool(lowercase)
        self.min_token_len: int = eff_min_len
        self.stopwords: frozenset[str] | None = (
            frozenset((str(w).lower() if self.lowercase else str(w)) for w in stopwords)
            if stopwords is not None
            else None
        )
        self._custom_tokenizer = tokenizer

        n_windows = len(documents)
        self.num_windows: int = n_windows

        if doc_offsets is not None:
            off_arr = np.ascontiguousarray(doc_offsets, dtype=np.int32)
            if (
                off_arr.ndim != 1
                or off_arr.size < 1
                or int(off_arr[0]) != 0
                or int(off_arr[-1]) != n_windows
                or bool(np.any(np.diff(off_arr) < 0))
            ):
                raise ValueError(
                    f"doc_offsets must be 1D, non-decreasing, start at 0, and end at num_windows={n_windows}"
                )
            self.doc_offsets: np.ndarray | None = off_arr
            self.num_docs: int = int(off_arr.size) - 1
        else:
            self.doc_offsets = None
            self.num_docs = n_windows

        doc_lengths = np.zeros((n_windows,), dtype=np.int32)
        term_to_id: dict[str, int] = {}
        postings_doc: list[list[int]] = []
        postings_tf: list[list[int]] = []

        for win_idx, text in enumerate(documents):
            tokens = self.tokenize(text)
            doc_lengths[win_idx] = len(tokens)
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            for term, tf in tf_counts.items():
                tid = term_to_id.get(term)
                if tid is None:
                    tid = len(term_to_id)
                    term_to_id[term] = tid
                    postings_doc.append([win_idx])
                    postings_tf.append([tf])
                else:
                    postings_doc[tid].append(win_idx)
                    postings_tf[tid].append(tf)

        self.doc_lengths: np.ndarray = doc_lengths
        self.avgdl: float = float(doc_lengths.mean()) if n_windows > 0 else 0.0
        self.vocab_size: int = len(term_to_id)
        self._term_to_id: dict[str, int] = term_to_id

        if self.vocab_size == 0 or n_windows == 0:
            self._term_offsets = np.zeros((self.vocab_size + 1,), dtype=np.int64)
            self._posting_doc_ids = np.empty((0,), dtype=np.int32)
            self._posting_scores = np.empty((0,), dtype=np.float32)
            self.idf = np.empty((0,), dtype=np.float32)
            return

        dfs = np.empty((self.vocab_size,), dtype=np.int32)
        term_offsets = np.zeros((self.vocab_size + 1,), dtype=np.int64)
        for tid in range(self.vocab_size):
            df = len(postings_doc[tid])
            dfs[tid] = df
            term_offsets[tid + 1] = term_offsets[tid] + df

        nnz = int(term_offsets[-1])
        flat_docs = np.empty((nnz,), dtype=np.int32)
        flat_tfs = np.empty((nnz,), dtype=np.float32)
        for tid in range(self.vocab_size):
            s0 = int(term_offsets[tid])
            s1 = int(term_offsets[tid + 1])
            flat_docs[s0:s1] = postings_doc[tid]
            flat_tfs[s0:s1] = postings_tf[tid]

        # Lucene BM25 non-negative IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
        dfs_f64 = dfs.astype(np.float64)
        idf = np.log1p((float(n_windows) - dfs_f64 + 0.5) / (dfs_f64 + 0.5)).astype(np.float32)

        if self.avgdl > 0.0:
            dl_norm = (
                np.float32(1.0 - eff_b)
                + np.float32(eff_b) * (doc_lengths.astype(np.float32) / np.float32(self.avgdl))
            ).astype(np.float32)
        else:
            dl_norm = np.ones((n_windows,), dtype=np.float32)

        denom = flat_tfs + np.float32(eff_k1) * dl_norm[flat_docs]
        tf_norm = np.where(
            denom > np.float32(0.0),
            (flat_tfs * np.float32(eff_k1 + 1.0)) / denom,
            np.float32(0.0),
        ).astype(np.float32)
        term_ids_per_posting = np.repeat(np.arange(self.vocab_size, dtype=np.int32), dfs)

        self._term_offsets = term_offsets
        self._posting_doc_ids = flat_docs
        self._posting_scores = (idf[term_ids_per_posting] * tf_norm).astype(np.float32)
        self.idf = idf

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single string using the configured tokenizer and filters."""
        if self._custom_tokenizer is None:
            return tokenize_lexical(
                text,
                lowercase=self.lowercase,
                stopwords=self.stopwords,
                min_token_len=self.min_token_len,
            )
        raw_tokens = [str(tok) for tok in self._custom_tokenizer(str(text or ""))]
        if not raw_tokens:
            return []
        if self.lowercase:
            raw_tokens = [tok.lower() for tok in raw_tokens]
        if self.stopwords is None and self.min_token_len <= 1:
            return raw_tokens
        if self.stopwords is None:
            return [tok for tok in raw_tokens if len(tok) >= self.min_token_len]
        return [
            tok
            for tok in raw_tokens
            if len(tok) >= self.min_token_len and tok not in self.stopwords
        ]

    def get_idf(self, term: str) -> float:
        """Return the BM25 IDF weight for ``term`` (0.0 if absent from vocabulary)."""
        key = str(term)
        tid = self._term_to_id.get(key)
        if tid is None and self.lowercase:
            tid = self._term_to_id.get(key.lower())
        if tid is None:
            return 0.0
        return float(self.idf[tid])

    def _score_windows_single(self, query: str) -> np.ndarray:
        scores = np.zeros((self.num_windows,), dtype=np.float32)
        if self.num_windows == 0 or self.vocab_size == 0:
            return scores
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return scores
        q_counts = Counter(q_tokens)
        for term, q_tf in q_counts.items():
            tid = self._term_to_id.get(term)
            if tid is None:
                continue
            s0 = int(self._term_offsets[tid])
            s1 = int(self._term_offsets[tid + 1])
            docs = self._posting_doc_ids[s0:s1]
            vals = self._posting_scores[s0:s1]
            if q_tf == 1:
                scores[docs] += vals
            else:
                scores[docs] += np.float32(q_tf) * vals
        return scores

    def score_all(
        self,
        queries: str | Sequence[str],
        *,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> np.ndarray:
        """Compute exact BM25 scores `[Q, N_docs]` (`[N_docs]` when ``queries`` is a single string)."""
        was_single = isinstance(queries, str)
        q_list = [queries] if was_single else list(queries)
        q_n = len(q_list)
        if q_n == 0 or self.num_docs == 0:
            empty = np.zeros((q_n, self.num_docs), dtype=np.float32)
            return empty[0] if was_single else empty

        win_scores = np.empty((q_n, self.num_windows), dtype=np.float32)
        for qi, q_str in enumerate(q_list):
            win_scores[qi] = self._score_windows_single(q_str)

        if self.doc_offsets is not None:
            doc_scores = score_document_windows(
                win_scores,
                self.doc_offsets,
                agg=agg or self.default_agg,
                hybrid_alpha=self.default_hybrid_alpha if hybrid_alpha is None else float(hybrid_alpha),
            )
        else:
            doc_scores = win_scores
        return doc_scores[0] if was_single else doc_scores

    def search(
        self,
        queries: str | Sequence[str],
        k: int = 10,
        *,
        positive_only: bool = False,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(top_indices, top_scores)`` of shape ``[Q, k]`` (``[k]`` for single string)."""
        eff_k_req = _validate_positive_int(k, "k")
        was_single = isinstance(queries, str)
        q_list = [queries] if was_single else list(queries)
        q_n = len(q_list)

        if q_n == 0 or self.num_docs == 0:
            empty_i = np.full((q_n, eff_k_req), -1, dtype=np.int32)
            empty_s = np.full((q_n, eff_k_req), -1e30, dtype=np.float32)
            return (empty_i[0], empty_s[0]) if was_single else (empty_i, empty_s)

        scores = self.score_all(q_list, agg=agg, hybrid_alpha=hybrid_alpha)
        out_idx = np.full((q_n, eff_k_req), -1, dtype=np.int32)
        out_sc = np.full((q_n, eff_k_req), -1e30, dtype=np.float32)
        doc_ids = np.arange(self.num_docs, dtype=np.int32)
        eff_k = min(eff_k_req, self.num_docs)

        for qi in range(q_n):
            row = scores[qi]
            if positive_only:
                pos_mask = row > np.float32(0.0)
                pos_ids = doc_ids[pos_mask]
                if pos_ids.size == 0:
                    continue
                pos_sc = row[pos_mask]
                take = min(eff_k, int(pos_ids.size))
                if take < int(pos_ids.size):
                    part = np.argpartition(-pos_sc, take - 1)[:take]
                    sub_sc = pos_sc[part]
                    sub_id = pos_ids[part]
                    order = np.lexsort((sub_id, -sub_sc))
                else:
                    sub_sc = pos_sc
                    sub_id = pos_ids
                    order = np.lexsort((sub_id, -sub_sc))
                out_idx[qi, :take] = sub_id[order]
                out_sc[qi, :take] = sub_sc[order]
                continue

            if eff_k < self.num_docs:
                part = np.argpartition(-row, eff_k - 1)[:eff_k]
                sub_sc = row[part]
                sub_id = doc_ids[part]
                order = np.lexsort((sub_id, -sub_sc))
                out_idx[qi, :eff_k] = sub_id[order]
                out_sc[qi, :eff_k] = sub_sc[order]
            else:
                order = np.lexsort((doc_ids, -row))[:eff_k]
                out_idx[qi, :eff_k] = doc_ids[order]
                out_sc[qi, :eff_k] = row[order]

        return (out_idx[0], out_sc[0]) if was_single else (out_idx, out_sc)


def rrf_fuse(
    rankings: Sequence[Sequence[_DocKey] | np.ndarray],
    *,
    rrf_k: float = 60.0,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[tuple[_DocKey, float]]:
    """Fuse multiple ranked candidate lists via weighted Reciprocal Rank Fusion.

    Each unique document in list ``r`` at 1-based position ``rank`` receives
    ``weights[r] / (rrf_k + rank)``. Negative integer sentinels (such as ``-1`` padding
    from ``CompactRetrievalIndex.search``) are ignored, duplicate entries inside a single
    ranking keep their first (best) rank, and ties break deterministically by
    ``(-fused_score, best_rank, doc_id)``.
    """
    eff_rrf_k = float(rrf_k)
    if not math.isfinite(eff_rrf_k) or eff_rrf_k <= 0.0:
        raise ValueError(f"rrf_k must be finite and > 0, got {rrf_k!r}")
    if top_k is not None:
        if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or int(top_k) < 0:
            raise ValueError(f"top_k must be a non-negative integer, got {top_k!r}")
        if int(top_k) == 0:
            return []
    if not rankings:
        return []

    if weights is not None:
        if len(weights) != len(rankings):
            raise ValueError(
                f"weights length ({len(weights)}) must match rankings length ({len(rankings)})"
            )
        eff_weights = [_validate_finite_nonneg(w, f"weights[{i}]") for i, w in enumerate(weights)]
    else:
        eff_weights = [1.0] * len(rankings)

    scores: dict[Any, float] = {}
    best_ranks: dict[Any, int] = {}
    first_seen: dict[Any, int] = {}

    for ranking, weight in zip(rankings, eff_weights, strict=True):
        seq = ranking.tolist() if isinstance(ranking, np.ndarray) else ranking
        seen_in_list: set[Any] = set()
        rank = 0
        for raw_id in seq:
            doc_id = _normalize_doc_key(raw_id)
            if isinstance(doc_id, int) and not isinstance(doc_id, bool) and doc_id < 0:
                continue
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            rank += 1
            contrib = weight / (eff_rrf_k + float(rank)) if weight > 0.0 else 0.0
            if doc_id not in scores:
                scores[doc_id] = contrib
                best_ranks[doc_id] = rank
                first_seen[doc_id] = len(first_seen)
            else:
                scores[doc_id] += contrib
                if rank < best_ranks[doc_id]:
                    best_ranks[doc_id] = rank

    if not scores:
        return []

    keys = list(scores.keys())
    all_ints = all(isinstance(k, int) and not isinstance(k, bool) for k in keys)
    all_strs = all(isinstance(k, str) for k in keys)
    if all_ints or all_strs:
        ordered = sorted(
            keys,
            key=lambda doc_id: (-scores[doc_id], best_ranks[doc_id], doc_id),
        )
    else:
        ordered = sorted(
            keys,
            key=lambda doc_id: (-scores[doc_id], best_ranks[doc_id], first_seen[doc_id]),
        )

    limit = len(ordered) if top_k is None else min(int(top_k), len(ordered))
    return [(doc_id, float(scores[doc_id])) for doc_id in ordered[:limit]]


def rrf_fuse_batch(
    rankings_per_retriever: Sequence[Sequence[Sequence[_DocKey] | np.ndarray] | np.ndarray],
    *,
    rrf_k: float = 60.0,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[list[tuple[_DocKey, float]]]:
    """Batch wrapper over :func:`rrf_fuse` across ``Q`` queries."""
    if not rankings_per_retriever:
        return []
    n_queries = len(rankings_per_retriever[0])
    for idx, batch in enumerate(rankings_per_retriever):
        if len(batch) != n_queries:
            raise ValueError(
                f"Retriever {idx} has {len(batch)} queries, expected {n_queries}"
            )
    return [
        rrf_fuse(
            [batch[qi] for batch in rankings_per_retriever],
            rrf_k=rrf_k,
            weights=weights,
            top_k=top_k,
        )
        for qi in range(n_queries)
    ]


@dataclass(frozen=True)
class HybridSearchHit:
    """Detailed per-hit provenance from :meth:`HybridRetrievalPipeline.search_detailed`."""

    doc_idx: int
    doc_id: str
    score: float
    rrf_score: float
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    laya_score: float | None = None
    rerank_decision: str | None = None


def _is_laya_reranker_instance(obj: Any) -> bool:
    return isinstance(obj, LayaRerankFusion) or (
        callable(getattr(obj, "rerank", None))
        and not callable(getattr(obj, "score_noul_batch", None))
        and not callable(getattr(obj, "score_choice_batch", None))
    )


class HybridRetrievalPipeline:
    """Two/three-stage retrieval pipeline: Dense (``CompactRetrievalIndex``) + BM25 + RRF + Laya ANE rerank."""

    def __init__(
        self,
        *,
        dense_index: CompactRetrievalIndex,
        bm25_index: BM25Index,
        doc_texts: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        doc_titles: Sequence[str] | None = None,
        reranker: Any | None = None,
        reranker_factory: Callable[[], Any] | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.25,
        dense_top_k: int = 100,
        bm25_top_k: int = 100,
        rerank_top_k: int = 20,
        dense_rerank_top_r: int = 256,
        rerank_prior_weight: float = 0.25,
        max_passage_chars: int = 600,
        rerank_fallback_on_error: bool = False,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> None:
        if bm25_index.num_docs != dense_index.num_docs:
            raise ValueError(
                f"bm25_index.num_docs ({bm25_index.num_docs}) != dense_index.num_docs ({dense_index.num_docs})"
            )
        if reranker is not None and reranker_factory is not None:
            raise ValueError("Provide at most one of `reranker` or `reranker_factory`")
        if reranker_factory is not None and not callable(reranker_factory):
            raise TypeError("`reranker_factory` must be callable")

        n_docs = dense_index.num_docs
        if doc_texts is not None and len(doc_texts) != n_docs:
            raise ValueError(f"doc_texts length ({len(doc_texts)}) != num_docs ({n_docs})")
        if doc_ids is not None and len(doc_ids) != n_docs:
            raise ValueError(f"doc_ids length ({len(doc_ids)}) != num_docs ({n_docs})")
        if doc_titles is not None and len(doc_titles) != n_docs:
            raise ValueError(f"doc_titles length ({len(doc_titles)}) != num_docs ({n_docs})")

        eff_rrf_k = float(rrf_k)
        if not math.isfinite(eff_rrf_k) or eff_rrf_k <= 0.0:
            raise ValueError(f"rrf_k must be finite and > 0, got {rrf_k!r}")
        eff_dw = _validate_finite_nonneg(dense_weight, "dense_weight")
        eff_bw = _validate_finite_nonneg(bm25_weight, "bm25_weight")
        eff_prior = _validate_finite_nonneg(rerank_prior_weight, "rerank_prior_weight")
        if eff_prior > 1.0:
            raise ValueError(f"rerank_prior_weight must be in [0.0, 1.0], got {rerank_prior_weight!r}")
        if agg is not None and agg not in VALID_WINDOW_AGGS:
            raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")

        self.dense_index: CompactRetrievalIndex = dense_index
        self.bm25_index: BM25Index = bm25_index
        self.doc_texts: tuple[str, ...] | None = (
            tuple(str(t) for t in doc_texts) if doc_texts is not None else None
        )
        self.doc_ids: tuple[str, ...] = (
            tuple(str(d) for d in doc_ids)
            if doc_ids is not None
            else tuple(str(i) for i in range(n_docs))
        )
        self.doc_titles: tuple[str, ...] | None = (
            tuple(str(t) for t in doc_titles) if doc_titles is not None else None
        )

        self.rrf_k: float = eff_rrf_k
        self.dense_weight: float = eff_dw
        self.bm25_weight: float = eff_bw
        self.dense_top_k: int = _validate_positive_int(dense_top_k, "dense_top_k")
        self.bm25_top_k: int = _validate_positive_int(bm25_top_k, "bm25_top_k")
        self.rerank_top_k: int = _validate_positive_int(rerank_top_k, "rerank_top_k")
        self.dense_rerank_top_r: int = _validate_positive_int(dense_rerank_top_r, "dense_rerank_top_r")
        self.rerank_prior_weight: float = eff_prior
        self.max_passage_chars: int = max(0, int(max_passage_chars))
        self.rerank_fallback_on_error: bool = bool(rerank_fallback_on_error)
        self.default_agg: WindowAgg | None = agg
        self.default_hybrid_alpha: float | None = (
            float(hybrid_alpha) if hybrid_alpha is not None else None
        )

        self._reranker: Any | None = None
        self._reranker_factory: Callable[[], Any] | None = reranker_factory
        self._owns_reranker: bool = False
        if reranker is not None:
            self._reranker = self._wrap_reranker(reranker)

    def _wrap_reranker(self, obj: Any, *, owns_backend: bool = False) -> Any:
        if _is_laya_reranker_instance(obj):
            return obj
        kwargs: dict[str, Any] = {
            "max_rerank": self.rerank_top_k,
            "prior_weight": self.rerank_prior_weight,
            "deduplicate_labels": False,
            "attach_score": True,
            "max_definition_chars": self.max_passage_chars,
            "max_broader": 0,
            "max_label_chars": 80,
            "fallback_on_error": self.rerank_fallback_on_error,
            "attach_confidence": True,
        }
        if owns_backend:
            return LayaRerankFusion(backend_factory=lambda: obj, **kwargs)
        return LayaRerankFusion(obj, **kwargs)

    def _get_reranker(self) -> Any | None:
        if self._reranker is not None:
            return self._reranker
        if self._reranker_factory is None:
            return None
        created = self._reranker_factory()
        wrapped = self._wrap_reranker(created, owns_backend=True)
        self._reranker = wrapped
        self._owns_reranker = True
        return wrapped

    @property
    def num_docs(self) -> int:
        return self.dense_index.num_docs

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None or self._reranker_factory is not None

    @classmethod
    def from_corpus(
        cls,
        doc_texts: Sequence[str],
        *,
        vectors: np.ndarray | CompactRetrievalIndex | None = None,
        encoder: Any | None = None,
        doc_ids: Sequence[str] | None = None,
        doc_titles: Sequence[str] | None = None,
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        window_texts: Sequence[str] | None = None,
        index_mode: IndexMode = "auto",
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg = "maxsim",
        hybrid_alpha: float = 0.7,
        use_neon: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
        stopwords: Iterable[str] | None = None,
        reranker: Any | None = None,
        reranker_factory: Callable[[], Any] | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.25,
        dense_top_k: int = 100,
        bm25_top_k: int = 100,
        rerank_top_k: int = 20,
        dense_rerank_top_r: int = 256,
        rerank_prior_weight: float = 0.25,
        max_passage_chars: int = 600,
        rerank_fallback_on_error: bool = False,
    ) -> HybridRetrievalPipeline:
        """Build dense ``CompactRetrievalIndex`` + ``BM25Index`` + optional Laya reranker from corpus texts."""
        if isinstance(vectors, CompactRetrievalIndex):
            dense_idx = vectors
        elif vectors is not None:
            dense_idx = CompactRetrievalIndex(
                vectors,
                doc_offsets=doc_offsets,
                mode=index_mode,
                is_mrl_trained=bool(getattr(encoder, "is_mrl_trained", False)) if encoder is not None else False,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                agg=agg,
                hybrid_alpha=hybrid_alpha,
                use_neon=use_neon,
            )
        elif encoder is not None:
            src_texts = window_texts if window_texts is not None else doc_texts
            dense_idx = build_engine_index(
                encoder,
                list(src_texts),
                doc_offsets=doc_offsets,
                mode=index_mode,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                agg=agg,
                hybrid_alpha=hybrid_alpha,
            )
        else:
            raise ValueError("Provide either `vectors` or `encoder` to HybridRetrievalPipeline.from_corpus")

        bm25_texts = window_texts if (window_texts is not None and doc_offsets is not None) else doc_texts
        bm25_offsets = doc_offsets if (window_texts is not None and doc_offsets is not None) else None
        bm25_idx = BM25Index(
            bm25_texts,
            k1=k1,
            b=b,
            doc_offsets=bm25_offsets,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            tokenizer=tokenizer,
            stopwords=stopwords,
        )
        return cls(
            dense_index=dense_idx,
            bm25_index=bm25_idx,
            doc_texts=doc_texts,
            doc_ids=doc_ids,
            doc_titles=doc_titles,
            reranker=reranker,
            reranker_factory=reranker_factory,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
            dense_rerank_top_r=dense_rerank_top_r,
            rerank_prior_weight=rerank_prior_weight,
            max_passage_chars=max_passage_chars,
            rerank_fallback_on_error=rerank_fallback_on_error,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
        )

    def _resolve_mode(
        self, mode: RetrievalMode
    ) -> Literal["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]:
        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_RETRIEVAL_MODES)}, got {mode!r}")
        if mode == "auto":
            if self.has_reranker and self.doc_texts is not None:
                return "hybrid_rerank"
            return "hybrid"
        if mode in ("hybrid_rerank", "dense_rerank"):
            if not self.has_reranker:
                raise ValueError(f"mode={mode!r} requires `reranker` or `reranker_factory`")
            if self.doc_texts is None:
                raise ValueError(f"mode={mode!r} requires `doc_texts` to format candidates")
        return mode

    def _build_laya_candidates(
        self,
        doc_indices: Sequence[int],
        *,
        deduplicate_labels: bool,
    ) -> list[dict[str, Any]]:
        assert self.doc_texts is not None
        candidates: list[dict[str, Any]] = []
        for doc_idx in doc_indices:
            doc_id = self.doc_ids[doc_idx]
            title = self.doc_titles[doc_idx] if self.doc_titles is not None else doc_id
            if deduplicate_labels:
                label = f"{title} [#{doc_idx}]"
            else:
                label = title
            candidates.append(
                {
                    "id": doc_id,
                    "doc_idx": int(doc_idx),
                    "label": label,
                    "definition": self.doc_texts[doc_idx],
                }
            )
        return candidates

    def _rerank_candidates(
        self,
        *,
        query_text: str,
        pool_scores: Sequence[tuple[int, float]],
        eff_k: int,
        eff_rr_k: int,
        reranker: Any,
        dedup_labels: bool,
        w_laya: float,
        w_prior: float,
        d_rank_map: Mapping[int, tuple[int, float]],
        b_rank_map: Mapping[int, tuple[int, float]],
        collect_details: bool,
    ) -> tuple[list[int], list[float], list[HybridSearchHit]]:
        rerank_pool = [int(di) for di, _ in pool_scores[:eff_rr_k]]
        tail_pool = [int(di) for di, _ in pool_scores[eff_rr_k:]]
        base_score_map = {int(di): float(sc) for di, sc in pool_scores}
        cands = self._build_laya_candidates(rerank_pool, deduplicate_labels=dedup_labels)
        pos_by_doc = {di: pos for pos, di in enumerate(rerank_pool)}
        m = len(rerank_pool)

        reranked_cards = reranker.rerank(
            query=query_text,
            candidates=cands,
            limit=m,
        )

        seen_docs: set[int] = set()
        reranked_head: list[tuple[int, float | None, str | None, float]] = []
        max_tail_sc = max((base_score_map[di] for di in tail_pool), default=0.0)
        head_offset = max_tail_sc + 1.0

        for card in reranked_cards:
            di = int(card["doc_idx"])
            if di in seen_docs:
                continue
            seen_docs.add(di)
            laya_sc_raw = card.get("layaScore")
            laya_sc = float(laya_sc_raw) if laya_sc_raw is not None else None
            decision = card.get("rerankDecision")
            if decision == "exact":
                mono_sc = head_offset + 2.0
            elif laya_sc is not None:
                orig_pos = pos_by_doc.get(di, m)
                prior = 1.0 - (float(orig_pos) / float(max(1, m)))
                blended = w_laya * laya_sc + w_prior * prior
                mono_sc = head_offset + blended
            else:
                mono_sc = head_offset + (1.0 / float(1 + len(reranked_head)))
            reranked_head.append((di, laya_sc, str(decision) if decision is not None else None, mono_sc))

        for di in rerank_pool:
            if di not in seen_docs:
                seen_docs.add(di)
                reranked_head.append((di, None, None, base_score_map[di]))

        combined: list[tuple[int, float | None, str | None, float]] = list(reranked_head)
        for di in tail_pool:
            if di not in seen_docs:
                seen_docs.add(di)
                combined.append((di, None, None, base_score_map[di]))

        take = min(eff_k, len(combined))
        out_ids: list[int] = []
        out_scs: list[float] = []
        hits: list[HybridSearchHit] = []
        prev_sc = float("inf")
        for pos in range(take):
            di, laya_sc, decision, mono_sc = combined[pos]
            safe_sc = min(prev_sc, mono_sc)
            prev_sc = safe_sc
            out_ids.append(di)
            out_scs.append(safe_sc)
            if collect_details:
                d_info = d_rank_map.get(di)
                b_info = b_rank_map.get(di)
                hits.append(
                    HybridSearchHit(
                        doc_idx=di,
                        doc_id=self.doc_ids[di],
                        score=float(safe_sc),
                        rrf_score=base_score_map[di],
                        dense_score=d_info[1] if d_info is not None else None,
                        dense_rank=d_info[0] if d_info is not None else None,
                        bm25_score=b_info[1] if b_info is not None else None,
                        bm25_rank=b_info[0] if b_info is not None else None,
                        laya_score=laya_sc,
                        rerank_decision=decision,
                    )
                )
        return out_ids, out_scs, hits

    def _run_batch(
        self,
        q_list: list[str],
        query_vecs: np.ndarray | None,
        eff_k: int,
        eff_mode: Literal["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"],
        *,
        dense_top_k: int | None,
        bm25_top_k: int | None,
        rerank_top_k: int | None,
        agg: WindowAgg | None,
        hybrid_alpha: float | None,
        collect_details: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[list[HybridSearchHit]]]:
        q_n = len(q_list)
        out_idx = np.full((q_n, eff_k), -1, dtype=np.int32)
        out_sc = np.full((q_n, eff_k), -1e30, dtype=np.float32)
        details: list[list[HybridSearchHit]] = [[] for _ in range(q_n)]

        if q_n == 0 or self.num_docs == 0:
            return out_idx, out_sc, details

        eff_agg = agg if agg is not None else self.default_agg
        eff_alpha = self.default_hybrid_alpha if hybrid_alpha is None else float(hybrid_alpha)

        if eff_mode in ("dense", "hybrid", "dense_rerank", "hybrid_rerank"):
            if query_vecs is None:
                raise ValueError(f"mode={eff_mode!r} requires `query_vecs`")
            q_arr = np.asarray(query_vecs)
            if q_arr.ndim == 1:
                q_arr = q_arr[None, :]
            if q_arr.ndim != 2 or int(q_arr.shape[0]) != q_n:
                raise ValueError(
                    f"query_vecs batch size ({q_arr.shape[0] if q_arr.ndim >= 1 else None}) != queries len ({q_n})"
                )
        else:
            q_arr = None

        if eff_mode == "dense":
            assert q_arr is not None
            d_idx, d_sc = self.dense_index.search(
                q_arr,
                k=eff_k,
                rerank_top_r=self.dense_rerank_top_r,
                agg=eff_agg,
                hybrid_alpha=eff_alpha,
            )
            if collect_details:
                for qi in range(q_n):
                    for r_pos in range(eff_k):
                        di = int(d_idx[qi, r_pos])
                        if di < 0:
                            continue
                        sc_v = float(d_sc[qi, r_pos])
                        details[qi].append(
                            HybridSearchHit(
                                doc_idx=di,
                                doc_id=self.doc_ids[di],
                                score=sc_v,
                                rrf_score=0.0,
                                dense_score=sc_v,
                                dense_rank=r_pos + 1,
                            )
                        )
            return d_idx, d_sc, details

        if eff_mode == "bm25":
            b_idx, b_sc = self.bm25_index.search(
                q_list,
                k=eff_k,
                positive_only=False,
                agg=eff_agg,
                hybrid_alpha=eff_alpha,
            )
            if collect_details:
                for qi in range(q_n):
                    for r_pos in range(eff_k):
                        di = int(b_idx[qi, r_pos])
                        if di < 0:
                            continue
                        sc_v = float(b_sc[qi, r_pos])
                        details[qi].append(
                            HybridSearchHit(
                                doc_idx=di,
                                doc_id=self.doc_ids[di],
                                score=sc_v,
                                rrf_score=0.0,
                                bm25_score=sc_v,
                                bm25_rank=r_pos + 1,
                            )
                        )
            return b_idx, b_sc, details

        assert q_arr is not None
        eff_rr_k = _validate_positive_int(
            self.rerank_top_k if rerank_top_k is None else rerank_top_k, "rerank_top_k"
        )
        uses_rerank = eff_mode in ("hybrid_rerank", "dense_rerank")
        base_pool = max(eff_k, eff_rr_k if uses_rerank else eff_k)

        reranker = self._get_reranker() if uses_rerank else None
        dedup_labels = bool(getattr(reranker, "deduplicate_labels", False)) if reranker is not None else False
        w_prior = float(getattr(reranker, "prior_weight", self.rerank_prior_weight)) if reranker is not None else 0.0
        w_laya = 1.0 - w_prior

        if eff_mode == "dense_rerank":
            assert reranker is not None
            eff_d_k = min(self.num_docs, base_pool)
            d_idx, d_sc = self.dense_index.search(
                q_arr,
                k=eff_d_k,
                rerank_top_r=max(self.dense_rerank_top_r, eff_d_k),
                agg=eff_agg,
                hybrid_alpha=eff_alpha,
            )
            for qi in range(q_n):
                pool_scores: list[tuple[int, float]] = []
                d_rank_map: dict[int, tuple[int, float]] = {}
                for r_pos, (di_raw, sc_raw) in enumerate(zip(d_idx[qi].tolist(), d_sc[qi].tolist(), strict=True), start=1):
                    if di_raw >= 0:
                        di_int = int(di_raw)
                        sc_f = float(sc_raw)
                        d_rank_map[di_int] = (r_pos, sc_f)
                        pool_scores.append((di_int, sc_f))
                if not pool_scores:
                    continue
                r_ids, r_scs, r_hits = self._rerank_candidates(
                    query_text=q_list[qi],
                    pool_scores=pool_scores,
                    eff_k=eff_k,
                    eff_rr_k=eff_rr_k,
                    reranker=reranker,
                    dedup_labels=dedup_labels,
                    w_laya=w_laya,
                    w_prior=w_prior,
                    d_rank_map=d_rank_map,
                    b_rank_map={},
                    collect_details=collect_details,
                )
                take = len(r_ids)
                out_idx[qi, :take] = r_ids
                out_sc[qi, :take] = np.asarray(r_scs, dtype=np.float32)
                if collect_details:
                    details[qi] = r_hits
            return out_idx, out_sc, details

        eff_d_k = min(
            self.num_docs,
            max(base_pool, _validate_positive_int(self.dense_top_k if dense_top_k is None else dense_top_k, "dense_top_k")),
        )
        eff_b_k = min(
            self.num_docs,
            max(base_pool, _validate_positive_int(self.bm25_top_k if bm25_top_k is None else bm25_top_k, "bm25_top_k")),
        )

        d_idx, d_sc = self.dense_index.search(
            q_arr,
            k=eff_d_k,
            rerank_top_r=max(self.dense_rerank_top_r, eff_d_k),
            agg=eff_agg,
            hybrid_alpha=eff_alpha,
        )
        b_idx, b_sc = self.bm25_index.search(
            q_list,
            k=eff_b_k,
            positive_only=True,
            agg=eff_agg,
            hybrid_alpha=eff_alpha,
        )

        for qi in range(q_n):
            d_row_ids = d_idx[qi]
            d_row_sc = d_sc[qi]
            b_row_ids = b_idx[qi]
            b_row_sc = b_sc[qi]

            fused = rrf_fuse(
                [d_row_ids, b_row_ids],
                rrf_k=self.rrf_k,
                weights=[self.dense_weight, self.bm25_weight],
                top_k=base_pool,
            )
            if not fused:
                continue

            d_rank_map = {}
            for r_pos, (di_raw, sc_raw) in enumerate(zip(d_row_ids.tolist(), d_row_sc.tolist(), strict=True), start=1):
                if di_raw >= 0:
                    d_rank_map[int(di_raw)] = (r_pos, float(sc_raw))

            b_rank_map: dict[int, tuple[int, float]] = {}
            for r_pos, (di_raw, sc_raw) in enumerate(zip(b_row_ids.tolist(), b_row_sc.tolist(), strict=True), start=1):
                if di_raw >= 0:
                    b_rank_map[int(di_raw)] = (r_pos, float(sc_raw))

            if reranker is None:
                take = min(eff_k, len(fused))
                for pos, (di, rrf_sc) in enumerate(fused[:take]):
                    out_idx[qi, pos] = int(di)
                    out_sc[qi, pos] = np.float32(rrf_sc)
                    if collect_details:
                        d_info = d_rank_map.get(int(di))
                        b_info = b_rank_map.get(int(di))
                        details[qi].append(
                            HybridSearchHit(
                                doc_idx=int(di),
                                doc_id=self.doc_ids[int(di)],
                                score=float(rrf_sc),
                                rrf_score=float(rrf_sc),
                                dense_score=d_info[1] if d_info is not None else None,
                                dense_rank=d_info[0] if d_info is not None else None,
                                bm25_score=b_info[1] if b_info is not None else None,
                                bm25_rank=b_info[0] if b_info is not None else None,
                            )
                        )
                continue

            r_ids, r_scs, r_hits = self._rerank_candidates(
                query_text=q_list[qi],
                pool_scores=fused,
                eff_k=eff_k,
                eff_rr_k=eff_rr_k,
                reranker=reranker,
                dedup_labels=dedup_labels,
                w_laya=w_laya,
                w_prior=w_prior,
                d_rank_map=d_rank_map,
                b_rank_map=b_rank_map,
                collect_details=collect_details,
            )
            take = len(r_ids)
            out_idx[qi, :take] = r_ids
            out_sc[qi, :take] = np.asarray(r_scs, dtype=np.float32)
            if collect_details:
                details[qi] = r_hits

        return out_idx, out_sc, details

    def search(
        self,
        queries: str | Sequence[str],
        query_vecs: np.ndarray | None = None,
        k: int = 10,
        *,
        mode: RetrievalMode = "auto",
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rerank_top_k: int | None = None,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search and return ``(top_indices, top_scores)`` of shape ``[Q, k]`` (``[k]`` for single string)."""
        eff_k = _validate_positive_int(k, "k")
        eff_mode = self._resolve_mode(mode)
        was_single = isinstance(queries, str)
        q_list = [queries] if was_single else list(queries)
        out_idx, out_sc, _ = self._run_batch(
            q_list,
            query_vecs,
            eff_k,
            eff_mode,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            collect_details=False,
        )
        return (out_idx[0], out_sc[0]) if was_single else (out_idx, out_sc)

    def search_detailed(
        self,
        queries: str | Sequence[str],
        query_vecs: np.ndarray | None = None,
        k: int = 10,
        *,
        mode: RetrievalMode = "auto",
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rerank_top_k: int | None = None,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
    ) -> list[HybridSearchHit] | list[list[HybridSearchHit]]:
        """Search and return rich :class:`HybridSearchHit` provenance records."""
        eff_k = _validate_positive_int(k, "k")
        eff_mode = self._resolve_mode(mode)
        was_single = isinstance(queries, str)
        q_list = [queries] if was_single else list(queries)
        _, _, details = self._run_batch(
            q_list,
            query_vecs,
            eff_k,
            eff_mode,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            collect_details=True,
        )
        return details[0] if was_single else details

    def close(self) -> None:
        """Close any lazily instantiated reranker owned by this pipeline."""
        reranker = self._reranker
        owns = self._owns_reranker
        self._reranker = None
        self._owns_reranker = False
        if owns and reranker is not None:
            close_fn = getattr(reranker, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> HybridRetrievalPipeline:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "BM25Index",
    "ENGLISH_STOPWORDS",
    "HybridRetrievalPipeline",
    "HybridSearchHit",
    "RetrievalMode",
    "VALID_RETRIEVAL_MODES",
    "rrf_fuse",
    "rrf_fuse_batch",
    "tokenize_lexical",
]
