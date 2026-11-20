"""Category 1 deterministic retrieval scorers: stable nDCG@10/MRR@10/Recall@10, MRLxDtype grid, LegalBench-RAG spans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import compute_paired_se


def check_embedding_validity(
    vecs: np.ndarray,
    *,
    expect_unit_norm: bool = True,
    norm_tol: float = 1e-3,
) -> np.ndarray:
    """Return boolean validity vector `[N]` checking finiteness, positive norm, and optional unit L2 norm."""
    arr = np.asarray(vecs)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix [N, D], got shape {arr.shape}")
    n = int(arr.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if arr.dtype == np.int8:
        norms = np.linalg.norm(arr.astype(np.float64), axis=1)
        return np.isfinite(norms) & (norms > 0.0)
    f64 = arr.astype(np.float64, copy=False)
    finite_ok = np.all(np.isfinite(f64), axis=1)
    norms = np.linalg.norm(f64, axis=1)
    if not expect_unit_norm:
        return finite_ok & np.isfinite(norms) & (norms > 0.0)
    return finite_ok & np.isfinite(norms) & (np.abs(norms - 1.0) < float(norm_tol))


def _l2_normalize_f64(vecs: np.ndarray) -> np.ndarray:
    f64 = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(f64, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    out = np.where(norms > 1e-12, f64 / safe, 0.0)
    return out


def project_mrl_dtype(
    vecs: np.ndarray,
    *,
    dim: int,
    dtype_mode: str = "f32",
    int8_mode: str = "sym_int8",
    pre_normalized: bool = True,
) -> np.ndarray:
    """Zero-recompute MRL truncation + dtype simulation returning float64 vectors ready for cosine scoring.

    Supported ``dtype_mode`` values:
      - ``f32`` / ``float32``: truncated float32 unit-normalized.
      - ``f16`` / ``float16``: truncated unit-normalized cast through float16.
      - ``int8`` / ``sym_int8`` / ``native_tanh_int8``: symmetric or ``tanh * 127`` int8 quantization.
        Note: ``native_tanh_int8`` requires unnormalized raw projections (``pre_normalized=False``)
        matching ``FastPplx(..., normalize=False, output_dtype="float32")``.
    """
    arr = np.asarray(vecs)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
    max_d = int(arr.shape[1])
    eff_d = max(1, min(int(dim), max_d))
    sub_raw = arr[:, :eff_d].astype(np.float64, copy=True)
    finite_rows = np.all(np.isfinite(sub_raw), axis=1)
    sub = np.where(finite_rows[:, None], sub_raw, 0.0)

    def _mask_nonfinite(out: np.ndarray) -> np.ndarray:
        if bool(np.all(finite_rows)):
            return out
        return np.where(finite_rows[:, None], out, np.nan)

    mode = str(dtype_mode).strip().lower()
    if mode in ("f32", "float32"):
        sub_norm = _l2_normalize_f64(sub.astype(np.float32))
        return _mask_nonfinite(sub_norm.astype(np.float32).astype(np.float64))

    if mode in ("f16", "float16"):
        sub_norm = _l2_normalize_f64(sub)
        return _mask_nonfinite(_l2_normalize_f64(sub_norm.astype(np.float16).astype(np.float64)))

    if mode in ("int8", "sym_int8", "native_tanh_int8"):
        eff_i8 = mode if mode != "int8" else str(int8_mode).strip().lower()
        if eff_i8 == "native_tanh_int8":
            if pre_normalized:
                raise ValueError(
                    "native_tanh_int8 requires unnormalized projections (pre_normalized=False); "
                    "quantizing L2-normalized vectors collapses tanh(x)*127 dynamic range."
                )
            # Matches fast-pplx native quantization: clip(rint(tanh(proj[:, :dim]) * 127), -128, 127).
            q = np.clip(np.rint(np.tanh(sub) * 127.0), -128.0, 127.0).astype(np.int8)
            return _mask_nonfinite(_l2_normalize_f64(q.astype(np.float64)))
        # Symmetric per-vector int8 quantization on L2-normalized slice.
        sub_norm = _l2_normalize_f64(sub)
        max_abs = np.max(np.abs(sub_norm), axis=1, keepdims=True)
        scale = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0)
        q = np.clip(np.rint(sub_norm / scale), -127.0, 127.0).astype(np.int8)
        return _mask_nonfinite(_l2_normalize_f64(q.astype(np.float64)))

    raise ValueError(f"Unsupported dtype_mode={dtype_mode!r}; expected f32, f16, int8, sym_int8, or native_tanh_int8")


def dcg_at_k(relevances: Sequence[float] | np.ndarray, k: int) -> float:
    """Standard TREC/BEIR ``DCG@k = sum_{r=1..k} (2^{rel_r} - 1) / log2(r + 1)``."""
    lim = min(int(k), len(relevances))
    if lim <= 0:
        return 0.0
    total = 0.0
    for r in range(lim):
        rel = float(relevances[r])
        if rel > 0.0:
            total += (math.pow(2.0, rel) - 1.0) / math.log2(float(r + 2))
    return total


def evaluate_retrieval_rankings(
    query_ids: Sequence[str],
    query_vecs: np.ndarray,
    doc_ids: Sequence[str],
    doc_vecs: np.ndarray,
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
    query_chunk_size: int = 256,
    expect_unit_norm: bool = True,
) -> dict[str, Any]:
    """Compute exact stable-tie-broken ``nDCG@k``, ``MRR@k``, and ``Recall@k`` per query and macro-averaged.

    Uses ``np.argsort(..., kind="mergesort")`` on float64 similarities so tied scores never jitter ranks.
    Queries with empty positive ``qrels`` are skipped per MTEB v2 specification.
    """
    q_arr = np.asarray(query_vecs)
    d_arr = np.asarray(doc_vecs)
    if len(query_ids) != int(q_arr.shape[0]):
        raise ValueError(f"query_ids len {len(query_ids)} != query_vecs rows {q_arr.shape[0]}")
    if len(doc_ids) != int(d_arr.shape[0]):
        raise ValueError(f"doc_ids len {len(doc_ids)} != doc_vecs rows {d_arr.shape[0]}")

    q_valid_all = check_embedding_validity(q_arr, expect_unit_norm=expect_unit_norm)
    d_valid_all = check_embedding_validity(d_arr, expect_unit_norm=expect_unit_norm)
    q_f64 = _l2_normalize_f64(q_arr)
    d_f64 = _l2_normalize_f64(d_arr)
    # Zero out invalid document vectors so NaNs never poison valid query dot products.
    if not bool(np.all(d_valid_all)):
        d_f64 = np.where(d_valid_all[:, None], d_f64, 0.0)

    n_docs = len(doc_ids)
    eff_k = min(int(k), n_docs)

    eval_indices: list[int] = []
    eval_qids: list[str] = []
    pos_qrels_list: list[dict[str, float]] = []
    for idx, qid in enumerate(query_ids):
        rel_map = qrels.get(qid)
        if not rel_map:
            continue
        pos = {did: float(sc) for did, sc in rel_map.items() if float(sc) > 0.0}
        if not pos:
            continue
        eval_indices.append(idx)
        eval_qids.append(qid)
        pos_qrels_list.append(pos)

    if not eval_indices or n_docs == 0:
        return {
            "k": int(k),
            "n_queries": 0,
            "n_valid_queries": 0,
            "n_docs": n_docs,
            "n_valid_docs": int(np.sum(d_valid_all)),
            f"ndcg@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"recall@{k}": 0.0,
            "per_query_ids": [],
            "per_query_valid": [],
            "per_query_ndcg": [],
            "per_query_mrr": [],
            "per_query_recall": [],
        }

    per_valid: list[bool] = []
    per_ndcg: list[float] = []
    per_mrr: list[float] = []
    per_recall: list[float] = []
    all_docs_valid = bool(np.all(d_valid_all))

    chunk = max(1, int(query_chunk_size))
    for start in range(0, len(eval_indices), chunk):
        sub_idxs = eval_indices[start : start + chunk]
        sub_q = q_f64[sub_idxs]
        sims = sub_q @ d_f64.T  # [C, n_docs] float64
        for loc_i, orig_i in enumerate(sub_idxs):
            q_pos = start + loc_i
            pos_map = pos_qrels_list[q_pos]
            row_sim = sims[loc_i]
            row_ok = bool(q_valid_all[orig_i]) and all_docs_valid and bool(np.all(np.isfinite(row_sim)))
            if not row_ok:
                per_valid.append(False)
                per_ndcg.append(0.0)
                per_mrr.append(0.0)
                per_recall.append(0.0)
                continue

            order = np.argsort(-row_sim, kind="mergesort")[:eff_k]
            rels = [pos_map.get(doc_ids[int(di)], 0.0) for di in order]

            ideal_rels = sorted(pos_map.values(), reverse=True)
            idcg = dcg_at_k(ideal_rels, k)
            ndcg_v = (dcg_at_k(rels, k) / idcg) if idcg > 0.0 else 0.0

            mrr_v = 0.0
            for rank_idx, r_val in enumerate(rels):
                if r_val > 0.0:
                    mrr_v = 1.0 / float(rank_idx + 1)
                    break

            hits = sum(1 for r_val in rels if r_val > 0.0)
            rec_v = float(hits) / float(len(pos_map))

            per_valid.append(True)
            per_ndcg.append(ndcg_v)
            per_mrr.append(mrr_v)
            per_recall.append(rec_v)

    return {
        "k": int(k),
        "n_queries": len(eval_indices),
        "n_valid_queries": int(sum(1 for v in per_valid if v)),
        "n_docs": n_docs,
        "n_valid_docs": int(np.sum(d_valid_all)),
        f"ndcg@{k}": float(np.mean(per_ndcg)) if per_ndcg else 0.0,
        f"mrr@{k}": float(np.mean(per_mrr)) if per_mrr else 0.0,
        f"recall@{k}": float(np.mean(per_recall)) if per_recall else 0.0,
        "per_query_ids": eval_qids,
        "per_query_valid": per_valid,
        "per_query_ndcg": per_ndcg,
        "per_query_mrr": per_mrr,
        "per_query_recall": per_recall,
    }


def evaluate_mrl_dtype_grid(
    query_ids: Sequence[str],
    query_vecs: np.ndarray,
    doc_ids: Sequence[str],
    doc_vecs: np.ndarray,
    qrels: Mapping[str, Mapping[str, float]],
    *,
    dims: Sequence[int | str] = (128, 256, 512, "max"),
    dtypes: Sequence[str] = ("f32", "f16", "int8"),
    int8_mode: str = "sym_int8",
    pre_normalized: bool = True,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate zero-recompute MRL ``dims x dtypes`` retrieval matrix from one full-dim float32 embedding pass."""
    q_arr = np.asarray(query_vecs)
    d_arr = np.asarray(doc_vecs)
    max_dim = int(q_arr.shape[1])

    resolved_dims: list[int] = []
    for d in dims:
        dv = max_dim if str(d).lower() == "max" else min(int(d), max_dim)
        if dv > 0 and dv not in resolved_dims:
            resolved_dims.append(dv)
    resolved_dims.sort()

    cells: dict[str, dict[str, Any]] = {}
    for dv in resolved_dims:
        for dt in dtypes:
            q_proj = project_mrl_dtype(
                q_arr, dim=dv, dtype_mode=dt, int8_mode=int8_mode, pre_normalized=pre_normalized
            )
            d_proj = project_mrl_dtype(
                d_arr, dim=dv, dtype_mode=dt, int8_mode=int8_mode, pre_normalized=pre_normalized
            )
            res = evaluate_retrieval_rankings(
                query_ids, q_proj, doc_ids, d_proj, qrels, k=k, expect_unit_norm=False
            )
            key = f"d{dv}_{dt}"
            cells[key] = {
                "dim": dv,
                "dtype": dt,
                f"ndcg@{k}": round(float(res[f"ndcg@{k}"]), 6),
                f"mrr@{k}": round(float(res[f"mrr@{k}"]), 6),
                f"recall@{k}": round(float(res[f"recall@{k}"]), 6),
            }

    ref_key = f"d{max_dim}_f32"
    ref_ndcg = float(cells[ref_key][f"ndcg@{k}"]) if ref_key in cells else 0.0
    mrl128_key = "d128_int8"
    delta_128_i8 = (
        round(float(cells[mrl128_key][f"ndcg@{k}"]) - ref_ndcg, 6)
        if (mrl128_key in cells and ref_key in cells)
        else None
    )
    return {
        "max_dim": max_dim,
        "dims": resolved_dims,
        "dtypes": list(dtypes),
        "int8_mode": int8_mode,
        "cells": cells,
        f"mrl_128_int8_delta_ndcg@{k}": delta_128_i8,
    }


# ---------------------------------------------------------------------------
# LegalBench-RAG Character-Span Chunker & Span-nDCG@10 / Contextual Lift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharSpanChunk:
    """One deterministic character-span chunk of a source document."""

    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def chunk_document_char_spans(
    doc_id: str,
    text: str,
    *,
    chunk_chars: int = 1024,
    overlap_chars: int = 0,
) -> list[CharSpanChunk]:
    """Split ``text`` deterministically into character-span chunks snapping cleanly to whitespace boundaries."""
    n = len(text)
    if n == 0:
        return []
    width = max(32, int(chunk_chars))
    ov = max(0, min(int(overlap_chars), width // 2))
    chunks: list[CharSpanChunk] = []
    pos = 0
    c_idx = 0
    while pos < n:
        end = min(n, pos + width)
        if end < n:
            # Snap backward to nearest whitespace within last 15% of window to avoid mid-word splits.
            floor_snap = pos + int(width * 0.85)
            ws = text.rfind(" ", floor_snap, end)
            nl = text.rfind("\n", floor_snap, end)
            snap = max(ws, nl)
            if snap > pos:
                end = snap + 1
        chunk_txt = text[pos:end]
        if chunk_txt:
            chunks.append(CharSpanChunk(doc_id=doc_id, chunk_index=c_idx, start_char=pos, end_char=end, text=chunk_txt))
            c_idx += 1
        if end >= n:
            break
        nxt = end - ov
        pos = nxt if nxt > pos else end
    return chunks


def merge_char_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent ``[start, end)`` half-open intervals."""
    valid = sorted((int(a), int(b)) for a, b in spans if int(b) > int(a))
    if not valid:
        return []
    merged: list[tuple[int, int]] = [valid[0]]
    for a, b in valid[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def span_overlap_length(
    chunk_doc_id: str,
    chunk_span: tuple[int, int],
    gold_spans_by_doc: Mapping[str, Sequence[tuple[int, int]]],
) -> int:
    """Exact character overlap between a retrieved chunk and merged gold spans in the same document."""
    g_list = gold_spans_by_doc.get(chunk_doc_id)
    if not g_list:
        return 0
    ca, cb = int(chunk_span[0]), int(chunk_span[1])
    if cb <= ca:
        return 0
    ov = 0
    for ga, gb in g_list:
        lo = max(ca, ga)
        hi = min(cb, gb)
        if hi > lo:
            ov += hi - lo
    return ov


def evaluate_legalbench_spans(
    queries: Sequence[Mapping[str, Any]],
    chunks: Sequence[CharSpanChunk],
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    *,
    k: int = 10,
    expect_unit_norm: bool = True,
) -> dict[str, Any]:
    """Compute LegalBench-RAG deterministic ``Span-nDCG@k`` and ``CharRecall@k`` with stable tie-breaking."""
    q_arr = np.asarray(query_vecs)
    c_arr = np.asarray(chunk_vecs)
    if len(queries) != int(q_arr.shape[0]):
        raise ValueError(f"queries len {len(queries)} != query_vecs rows {q_arr.shape[0]}")
    if len(chunks) != int(c_arr.shape[0]):
        raise ValueError(f"chunks len {len(chunks)} != chunk_vecs rows {c_arr.shape[0]}")

    q_valid_all = check_embedding_validity(q_arr, expect_unit_norm=expect_unit_norm)
    c_valid_all = check_embedding_validity(c_arr, expect_unit_norm=expect_unit_norm)
    q_f64 = _l2_normalize_f64(q_arr)
    c_f64 = _l2_normalize_f64(c_arr)
    if not bool(np.all(c_valid_all)):
        c_f64 = np.where(c_valid_all[:, None], c_f64, 0.0)

    n_chunks = len(chunks)
    eff_k = min(int(k), n_chunks)
    if not queries or n_chunks == 0:
        return {
            "k": int(k),
            "n_queries": 0,
            "n_valid_queries": 0,
            f"span_ndcg@{k}": 0.0,
            f"char_recall@{k}": 0.0,
            "per_query_ids": [],
            "per_query_valid": [],
            "per_query_span_ndcg": [],
            "per_query_char_recall": [],
        }

    # Index chunks by doc_id once so computing exact ideal Span-IDCG@k is O(chunks_in_gold_docs).
    doc_to_chunk_indices: dict[str, list[int]] = {}
    for ci, ch in enumerate(chunks):
        doc_to_chunk_indices.setdefault(ch.doc_id, []).append(ci)

    sims = q_f64 @ c_f64.T
    all_chunks_valid = bool(np.all(c_valid_all))
    per_ids: list[str] = []
    per_valid: list[bool] = []
    per_ndcg: list[float] = []
    per_recall: list[float] = []

    for qi, q_item in enumerate(queries):
        raw_spans = q_item.get("gold_spans") or []
        by_doc_raw: dict[str, list[tuple[int, int]]] = {}
        for doc_id, a, b in raw_spans:
            if int(b) > int(a):
                by_doc_raw.setdefault(str(doc_id), []).append((int(a), int(b)))
        gold_by_doc = {d: merge_char_spans(sp) for d, sp in by_doc_raw.items()}
        total_gold_chars = sum(b - a for sp_list in gold_by_doc.values() for a, b in sp_list)
        if total_gold_chars <= 0:
            continue

        per_ids.append(str(q_item.get("id") or qi))
        row_sim = sims[qi]
        is_ok = bool(q_valid_all[qi]) and all_chunks_valid and bool(np.all(np.isfinite(row_sim)))
        if not is_ok:
            per_valid.append(False)
            per_ndcg.append(0.0)
            per_recall.append(0.0)
            continue

        order = np.argsort(-row_sim, kind="mergesort")[:eff_k]

        # Compute exact graded overlap relevance in [0, 1] and exact merged CharRecall@k.
        rels: list[float] = []
        covered_by_doc: dict[str, list[tuple[int, int]]] = {}
        for ci_raw in order:
            ch = chunks[int(ci_raw)]
            ov = span_overlap_length(ch.doc_id, (ch.start_char, ch.end_char), gold_by_doc)
            rels.append(float(ov) / float(total_gold_chars))
            if ov > 0:
                for ga, gb in gold_by_doc.get(ch.doc_id, ()):
                    lo = max(ch.start_char, ga)
                    hi = min(ch.end_char, gb)
                    if hi > lo:
                        covered_by_doc.setdefault(ch.doc_id, []).append((lo, hi))

        covered_chars = sum(
            b - a for sp_list in covered_by_doc.values() for a, b in merge_char_spans(sp_list)
        )
        char_rec = min(1.0, float(covered_chars) / float(total_gold_chars))

        # Ideal relevance list over all chunks in gold documents.
        ideal_rels: list[float] = []
        for gd in gold_by_doc:
            for ci in doc_to_chunk_indices.get(gd, ()):
                ch = chunks[ci]
                ov = span_overlap_length(ch.doc_id, (ch.start_char, ch.end_char), gold_by_doc)
                if ov > 0:
                    ideal_rels.append(float(ov) / float(total_gold_chars))
        ideal_rels.sort(reverse=True)
        idcg = dcg_at_k(ideal_rels, k)
        sndcg = (dcg_at_k(rels, k) / idcg) if idcg > 0.0 else 0.0

        per_valid.append(True)
        per_ndcg.append(sndcg)
        per_recall.append(char_rec)

    return {
        "k": int(k),
        "n_queries": len(per_ids),
        "n_valid_queries": int(sum(1 for v in per_valid if v)),
        f"span_ndcg@{k}": float(np.mean(per_ndcg)) if per_ndcg else 0.0,
        f"char_recall@{k}": float(np.mean(per_recall)) if per_recall else 0.0,
        "per_query_ids": per_ids,
        "per_query_valid": per_valid,
        "per_query_span_ndcg": per_ndcg,
        "per_query_char_recall": per_recall,
    }


def compute_contextual_lift(
    isolated_result: Mapping[str, Any],
    contextual_result: Mapping[str, Any],
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Compute ``Contextual ΔnDCG@10 = span_ndcg(contextual) - span_ndcg(isolated)`` plus paired SE."""
    iso_s = isolated_result.get("per_query_span_ndcg", [])
    ctx_s = contextual_result.get("per_query_span_ndcg", [])
    iso_mean = float(isolated_result.get(f"span_ndcg@{k}", 0.0))
    ctx_mean = float(contextual_result.get(f"span_ndcg@{k}", 0.0))
    paired = (
        compute_paired_se(iso_s, ctx_s)
        if (len(iso_s) == len(ctx_s) and len(iso_s) > 0)
        else {"n_paired": 0, "delta_mean": ctx_mean - iso_mean, "se_paired": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    )
    return {
        f"isolated_span_ndcg@{k}": round(iso_mean, 6),
        f"contextual_span_ndcg@{k}": round(ctx_mean, 6),
        f"contextual_delta_ndcg@{k}": round(ctx_mean - iso_mean, 6),
        "se_paired": round(float(paired["se_paired"]), 6),
        "ci95_low": round(float(paired["ci95_low"]), 6),
        "ci95_high": round(float(paired["ci95_high"]), 6),
    }


__all__ = [
    "CharSpanChunk",
    "check_embedding_validity",
    "chunk_document_char_spans",
    "compute_contextual_lift",
    "dcg_at_k",
    "evaluate_legalbench_spans",
    "evaluate_mrl_dtype_grid",
    "evaluate_retrieval_rankings",
    "merge_char_spans",
    "project_mrl_dtype",
    "span_overlap_length",
]
