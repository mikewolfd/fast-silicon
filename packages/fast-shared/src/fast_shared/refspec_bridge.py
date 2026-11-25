"""Dependency-injection bridge connecting Apple Silicon ``fast-silicon`` engines to RefSpec Vocabulary Packs.

This module intentionally does NOT import ``refspec`` at runtime. Instead, its classes
structurally satisfy RefSpec's stdlib-only ``@runtime_checkable`` DI protocols
(``QueryEncoder``, ``SemanticIndex``, ``CandidateReranker``) via pure Python duck typing
while staying decoupled from accelerator packages until injected or lazily instantiated.

Components:
  * **T3.1 ``FastSiliconQueryEncoder``**: DI adapter wrapping ``FastBgeSmall``,
    ``FastEmbeddingGemma2``, ``FastPplx``, or any compatible encoder / lazy factory.
  * **T3.2 ``CompactRetrievalSemanticIndex``**: RefSpec ``SemanticIndex`` backed by
    ``CompactRetrievalIndex`` (ARM NEON ``vdotq_s32`` + ``vcntq_u8`` 1-bit binary prefilter
    + symmetric ``int8`` rerank) with both single-vector and multi-label ``MaxSim`` window
    support plus O(1) ``vector_for_rid`` reconstruction.
  * **T3.3 ``LayaRerankFusion``**: RefSpec ``CandidateReranker`` using Laya typed ``noul``
    binary relevance scoring in a single batched pass with strict Tier-0 exact-match pinning,
    configurable prior-rank fusion, and optional duplicate-label counterpart grouping.
  * **T3.4 ``export_pack_concept_texts`` / ``embed_sqlite_pack_to_index`` /
    ``embed_sqlite_pack_to_parquet``**: Bounded-memory streaming ETL from compiled RefSpec
    SQLite Vocabulary Packs into either an in-memory ``CompactRetrievalSemanticIndex`` or
    RefSpec-compatible ``input_sha256``-deduplicated Parquet (``resources.parquet`` +
    ``vectors/part-000.parquet``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import threading
import time
from typing import Any, Literal
from urllib.parse import quote

import numpy as np

from .embeddings import accepts_kwarg, truncate_and_normalize
from .retrieval_index import (
    VALID_INDEX_MODES,
    VALID_WINDOW_AGGS,
    CompactRetrievalIndex,
    IndexMode,
    WindowAgg,
    resolve_index_spec,
)

_SCHEME_URN_PREFIX = "urn:ref:atlas-resource-scheme:"
_EXACT_MATCH_REASONS: frozenset[str] = frozenset(
    {"exact_preferred", "exact_alternate", "exact_hidden"}
)
_VPACK_MAGIC = b"RFVP0001"
_VPACK_SCHEMA_VERSION = "2"
_MISSING_VIDX = 0xFFFFFFFF


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_pack_sqlite_metadata(pack_path: Path | str) -> tuple[Path, str, str, str, dict[str, str]]:
    """Read ``pack_meta`` and file SHA-256 digest from a compiled RefSpec SQLite pack."""
    resolved = Path(pack_path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Vocabulary pack file not found: {pack_path}")
    pack_sha256 = _sha256_file(resolved)
    uri = f"file:{quote(str(resolved))}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        meta = {str(k): str(v) for k, v in con.execute("SELECT key, value FROM pack_meta").fetchall()}
    finally:
        con.close()
    source_manifest_digest = meta.get("source_manifest_digest", "").strip()
    if not source_manifest_digest:
        raise ValueError(f"Vocabulary pack {resolved} lacks required pack_meta['source_manifest_digest']")
    pack_id = meta.get("pack_id", "").strip()
    return resolved, pack_sha256, pack_id, source_manifest_digest, meta


def _infer_encoder_model_id(encoder: Any) -> str | None:
    for attr in ("model_id", "model_name"):
        val = getattr(encoder, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _infer_encoder_dimensions(encoder: Any) -> int | None:
    for attr in ("dimensions", "dim", "max_dim"):
        val = getattr(encoder, attr, None)
        if isinstance(val, (int, np.integer)) and not isinstance(val, bool) and int(val) > 0:
            return int(val)
    return None


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return out


def _encode_text_batch(
    encoder: Any,
    texts: Sequence[str],
    *,
    dimensions: int | None = None,
    mode: str | None = None,
    is_query: bool = False,
    add_query_prefix: bool = False,
    add_document_prefix: bool = False,
    prefix: str | None = None,
    prompt_name: str | None = None,
    max_length: int | None = None,
) -> np.ndarray:
    """Encode a non-empty sequence of texts via any ``fast-silicon`` encoder into unit-L2 ``[N, D]`` ``float32``."""
    items = list(texts)
    if not items:
        dim = dimensions or _infer_encoder_dimensions(encoder) or 0
        return np.empty((0, dim), dtype=np.float32)

    encode_fn = getattr(encoder, "encode_texts", None)
    if not callable(encode_fn):
        raise TypeError(f"Injected encoder {type(encoder).__name__!r} does not implement callable encode_texts()")

    kwargs: dict[str, Any] = {}
    if accepts_kwarg(encode_fn, "is_query"):
        kwargs["is_query"] = bool(is_query)
    if accepts_kwarg(encode_fn, "add_query_prefix"):
        kwargs["add_query_prefix"] = bool(add_query_prefix)
    if accepts_kwarg(encode_fn, "add_document_prefix"):
        kwargs["add_document_prefix"] = bool(add_document_prefix)
    if mode is not None and accepts_kwarg(encode_fn, "mode"):
        kwargs["mode"] = mode
    if prefix is not None and accepts_kwarg(encode_fn, "prefix"):
        kwargs["prefix"] = prefix
    if prompt_name is not None and accepts_kwarg(encode_fn, "prompt_name"):
        kwargs["prompt_name"] = prompt_name
    if max_length is not None and accepts_kwarg(encode_fn, "max_length"):
        kwargs["max_length"] = int(max_length)
    if dimensions is not None and accepts_kwarg(encode_fn, "dimensions"):
        if getattr(encoder, "supports_dimensions", True) is not False:
            kwargs["dimensions"] = int(dimensions)

    raw = encode_fn(items, **kwargs)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1 and len(items) == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] != len(items):
        raise ValueError(
            f"Encoder returned unexpected embedding shape {arr.shape} for batch of {len(items)} texts"
        )
    eff_dim = int(dimensions) if dimensions is not None else int(arr.shape[1])
    if arr.shape[1] < eff_dim:
        raise ValueError(
            f"Encoder returned vectors of width {arr.shape[1]} < requested dimensions={eff_dim}"
        )
    return np.ascontiguousarray(truncate_and_normalize(arr, eff_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# T3.1: FastSiliconQueryEncoder (RefSpec QueryEncoder structural adapter)
# ---------------------------------------------------------------------------


class FastSiliconQueryEncoder:
    """Dependency-injected RefSpec ``QueryEncoder`` backed by a ``fast-silicon`` encoder.

    Structurally implements ``model_id``, ``dimensions``, and ``encode_query(text: str) -> np.ndarray``
    matching ``refspec.atlas.vocabulary_protocols.QueryEncoder`` without importing ``refspec``.

    Works with ``FastBgeSmall``, ``FastEmbeddingGemma2``, ``FastPplx``, test fakes, or a lazy
    ``encoder_factory`` (deferred until the first ``encode_query`` call when ``model_id`` and
    ``dimensions`` are provided upfront).
    """

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        encoder_factory: Callable[[], Any] | None = None,
        model_id: str | None = None,
        dimensions: int | None = None,
        add_query_prefix: bool = True,
        is_query: bool = True,
        mode: str | None = None,
        prefix: str | None = None,
        prompt_name: str | None = None,
        max_length: int | None = None,
    ) -> None:
        if (encoder is None) == (encoder_factory is None):
            raise ValueError("Provide exactly one of `encoder` or `encoder_factory`")
        if encoder_factory is not None and not callable(encoder_factory):
            raise TypeError("`encoder_factory` must be callable")

        self._encoder: Any | None = encoder
        self._encoder_factory: Callable[[], Any] | None = encoder_factory
        self._owns_encoder: bool = False
        self._lock = threading.Lock()

        if model_id is not None:
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("`model_id` must be a non-empty string")
            self._model_id: str | None = model_id.strip()
        else:
            self._model_id = _infer_encoder_model_id(encoder) if encoder is not None else None

        if dimensions is not None:
            self._dimensions: int | None = _validate_positive_int(dimensions, "dimensions")
        else:
            self._dimensions = _infer_encoder_dimensions(encoder) if encoder is not None else None

        if encoder is not None and self._dimensions is not None:
            max_dim = getattr(encoder, "max_dim", None)
            if isinstance(max_dim, (int, np.integer)) and not isinstance(max_dim, bool):
                if self._dimensions > int(max_dim):
                    raise ValueError(
                        f"Requested dimensions={self._dimensions} exceeds encoder max_dim={int(max_dim)}"
                    )

        self._add_query_prefix: bool = bool(add_query_prefix)
        self._is_query: bool = bool(is_query)
        self._mode: str | None = mode
        self._prefix: str | None = prefix
        self._prompt_name: str | None = prompt_name
        self._max_length: int | None = (
            _validate_positive_int(max_length, "max_length") if max_length is not None else None
        )

    def _get_encoder(self) -> Any:
        enc = self._encoder
        if enc is not None:
            return enc
        with self._lock:
            enc = self._encoder
            if enc is not None:
                return enc
            assert self._encoder_factory is not None
            enc = self._encoder_factory()
            self._encoder = enc
            self._owns_encoder = True
            if self._model_id is None:
                self._model_id = _infer_encoder_model_id(enc)
            if self._dimensions is None:
                self._dimensions = _infer_encoder_dimensions(enc)
            return enc

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            enc = self._get_encoder()
            inferred = _infer_encoder_model_id(enc)
            if not inferred:
                raise ValueError(
                    "Unable to infer `model_id` from injected encoder; pass `model_id` explicitly"
                )
            self._model_id = inferred
        return self._model_id

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            enc = self._get_encoder()
            inferred = _infer_encoder_dimensions(enc)
            if inferred is None:
                raise ValueError(
                    "Unable to infer `dimensions` from injected encoder; pass `dimensions` explicitly"
                )
            self._dimensions = inferred
        return self._dimensions

    @property
    def is_initialized(self) -> bool:
        """Whether the underlying encoder instance is already constructed."""
        return self._encoder is not None

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one query string into a unit-L2-normalized 1D ``np.float32`` vector of length ``dimensions``."""
        if not isinstance(text, str):
            raise TypeError(f"query text must be a str, got {type(text).__name__}")
        enc = self._get_encoder()
        encode_fn = getattr(enc, "encode_texts", None)

        if callable(encode_fn):
            kwargs: dict[str, Any] = {}
            if accepts_kwarg(encode_fn, "is_query"):
                kwargs["is_query"] = self._is_query
            if accepts_kwarg(encode_fn, "add_query_prefix"):
                kwargs["add_query_prefix"] = self._add_query_prefix
            if self._mode is not None and accepts_kwarg(encode_fn, "mode"):
                kwargs["mode"] = self._mode
            if self._prefix is not None and accepts_kwarg(encode_fn, "prefix"):
                kwargs["prefix"] = self._prefix
            if self._prompt_name is not None and accepts_kwarg(encode_fn, "prompt_name"):
                kwargs["prompt_name"] = self._prompt_name
            if self._max_length is not None and accepts_kwarg(encode_fn, "max_length"):
                kwargs["max_length"] = self._max_length
            if self._dimensions is not None and accepts_kwarg(encode_fn, "dimensions"):
                if getattr(enc, "supports_dimensions", True) is not False:
                    kwargs["dimensions"] = self._dimensions
            raw = encode_fn(text, **kwargs)
        else:
            fallback_fn = getattr(enc, "encode_query", None)
            if not callable(fallback_fn):
                raise TypeError(
                    f"Injected encoder {type(enc).__name__!r} must implement `encode_texts` or `encode_query`"
                )
            raw = fallback_fn(text)

        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1 or arr.shape[0] == 0:
            raise ValueError(f"Encoder returned invalid query vector shape {arr.shape}")

        eff_dim = self._dimensions if self._dimensions is not None else int(arr.shape[0])
        if arr.shape[0] < eff_dim:
            raise ValueError(
                f"Encoder returned query vector of length {arr.shape[0]} < dimensions={eff_dim}"
            )
        if self._dimensions is None:
            self._dimensions = eff_dim
        vec = truncate_and_normalize(arr, eff_dim)
        return np.ascontiguousarray(vec, dtype=np.float32)

    def close(self) -> None:
        """Close the underlying encoder only if it was lazily created by ``encoder_factory``."""
        if self._owns_encoder and self._encoder is not None:
            close_fn = getattr(self._encoder, "close", None)
            if callable(close_fn):
                close_fn()
            self._encoder = None
            self._owns_encoder = False

    def __enter__(self) -> FastSiliconQueryEncoder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# T3.2: CompactRetrievalSemanticIndex (RefSpec SemanticIndex structural adapter)
# ---------------------------------------------------------------------------


def _validate_binary_bits_vs_dim(
    *,
    mode: IndexMode,
    mrl_dim: int | None,
    binary_bits: int | None,
    raw_dim: int | None = None,
    is_mrl_trained: bool = False,
) -> None:
    if binary_bits is None:
        return
    b_bits = _validate_positive_int(binary_bits, "binary_bits")
    if mode in ("binary_int8", "auto") and mrl_dim is not None:
        target_dim = _validate_positive_int(mrl_dim, "mrl_dim")
        if b_bits > target_dim:
            raise ValueError(
                f"CompactRetrievalSemanticIndex does not support binary_bits={b_bits} > "
                f"mrl_dim={target_dim}: SemanticIndex.dimensions and vector_for_rid() operate "
                f"in {target_dim}-dimensional space so binary_bits cannot exceed dim "
                "(pass binary_bits <= mrl_dim or omit mrl_dim)."
            )
    if mode == "binary_int8" and raw_dim is not None:
        _, target_dim = resolve_index_spec(
            raw_dim=raw_dim,
            mode=mode,
            is_mrl_trained=is_mrl_trained,
            mrl_dim=mrl_dim,
            prefer_binary=True,
        )
        if b_bits > target_dim:
            raise ValueError(
                f"CompactRetrievalSemanticIndex does not support binary_bits={b_bits} > "
                f"dim={target_dim}: SemanticIndex.dimensions and vector_for_rid() operate "
                f"in {target_dim}-dimensional space so binary_bits cannot exceed dim "
                "(pass binary_bits <= mrl_dim or omit mrl_dim)."
            )


class CompactRetrievalSemanticIndex:
    """RefSpec ``SemanticIndex`` backed by ``CompactRetrievalIndex`` (NEON binary+INT8 & MaxSim).

    Supports both:
      1. Single-vector-per-concept (or deduplicated N-rids-per-vector) retrieval via
         ARM NEON ``vcntq_u8`` 1-bit binary prefilter + ``vdotq_s32`` symmetric ``int8`` rerank.
      2. Multi-label ``MaxSim`` concept windows (``doc_offsets`` spanning
         ``[pref_label, alt_labels..., definition]`` per concept), preserving O(1)
         ``vector_for_rid(rid)`` reconstruction of each concept's primary window vector.
    """

    def __init__(
        self,
        vectors: np.ndarray | CompactRetrievalIndex,
        rids_per_doc: Sequence[int | Sequence[int]],
        *,
        pack_sha256: str,
        source_manifest_digest: str,
        model_id: str,
        pack_id: str = "",
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        mode: IndexMode = "auto",
        is_mrl_trained: bool = False,
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
        rerank_top_r: int = 256,
        use_neon: bool = True,
    ) -> None:
        if not isinstance(pack_sha256, str) or not pack_sha256.strip():
            raise ValueError("`pack_sha256` must be a non-empty string")
        if not isinstance(source_manifest_digest, str) or not source_manifest_digest.strip():
            raise ValueError("`source_manifest_digest` must be a non-empty string")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("`model_id` must be a non-empty string")
        if mode not in VALID_INDEX_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_INDEX_MODES)}, got {mode!r}")
        if agg is not None and agg not in VALID_WINDOW_AGGS:
            raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")
        if hybrid_alpha is not None and not math.isfinite(float(hybrid_alpha)):
            raise ValueError(f"hybrid_alpha must be finite, got {hybrid_alpha!r}")

        if isinstance(vectors, CompactRetrievalIndex):
            if doc_offsets is not None:
                raise ValueError("Pass `doc_offsets` when constructing from arrays, not a prebuilt index")
            index = vectors
            eff_agg: WindowAgg = index.default_agg if agg is None else agg
            eff_hybrid_alpha: float = (
                float(index.default_hybrid_alpha) if hybrid_alpha is None else float(hybrid_alpha)
            )
        else:
            eff_agg = "maxsim" if agg is None else agg
            eff_hybrid_alpha = 0.7 if hybrid_alpha is None else float(hybrid_alpha)
            arr_view = np.asarray(vectors)
            raw_dim = int(arr_view.shape[1]) if arr_view.ndim == 2 else None
            _validate_binary_bits_vs_dim(
                mode=mode,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                raw_dim=raw_dim,
                is_mrl_trained=is_mrl_trained,
            )
            off_arg = (
                list(doc_offsets)
                if doc_offsets is not None and not isinstance(doc_offsets, np.ndarray)
                else doc_offsets
            )
            index = CompactRetrievalIndex(
                vectors,
                doc_offsets=off_arg,
                mode=mode,
                is_mrl_trained=is_mrl_trained,
                mrl_dim=mrl_dim,
                binary_bits=binary_bits,
                agg=eff_agg,
                hybrid_alpha=eff_hybrid_alpha,
                use_neon=use_neon,
            )

        idx_binary_bits = getattr(index, "binary_bits", None)
        if idx_binary_bits is not None and int(idx_binary_bits) > int(index.dim):
            raise ValueError(
                f"CompactRetrievalSemanticIndex does not support binary_bits={int(idx_binary_bits)} "
                f"> dim={int(index.dim)}: SemanticIndex.dimensions and vector_for_rid() operate "
                f"in {int(index.dim)}-dimensional space so binary_bits cannot exceed dim "
                "(pass binary_bits <= mrl_dim or omit mrl_dim)."
            )

        if len(rids_per_doc) != index.num_docs:
            raise ValueError(
                f"len(rids_per_doc)={len(rids_per_doc)} must match index.num_docs={index.num_docs}"
            )

        self._index: CompactRetrievalIndex = index
        self._pack_sha256: str = pack_sha256.strip()
        self._pack_id: str = pack_id.strip() if isinstance(pack_id, str) else ""
        self._source_manifest_digest: str = source_manifest_digest.strip()
        self._model_id: str = model_id.strip()
        self._agg: WindowAgg = eff_agg
        self._hybrid_alpha: float = eff_hybrid_alpha
        self._rerank_top_r: int = max(1, int(rerank_top_r))

        # Flatten doc_idx -> rids CSR table and build O(1) rid -> primary_window_idx lookup.
        rid_offsets = np.zeros(index.num_docs + 1, dtype=np.int64)
        flat_rids: list[int] = []
        rid_to_primary_window: dict[int, int] = {}
        empty_docs = 0

        for doc_idx, entry in enumerate(rids_per_doc):
            if isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
                doc_rids = (int(entry),)
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                doc_rids = tuple(int(r) for r in entry)
            else:
                raise TypeError(
                    f"rids_per_doc[{doc_idx}] must be an int or Sequence[int], got {type(entry).__name__}"
                )

            if not doc_rids:
                empty_docs += 1
            flat_rids.extend(doc_rids)
            rid_offsets[doc_idx + 1] = len(flat_rids)

            if index.doc_offsets is not None:
                w0 = int(index.doc_offsets[doc_idx])
                w1 = int(index.doc_offsets[doc_idx + 1])
                primary_win = w0 if w1 > w0 else -1
            else:
                primary_win = doc_idx

            if primary_win >= 0:
                for r in doc_rids:
                    rid_to_primary_window.setdefault(r, primary_win)

        self._rid_offsets: np.ndarray = rid_offsets
        self._rids: np.ndarray = np.asarray(flat_rids, dtype=np.int64)
        self._rid_to_primary_window: dict[int, int] | None = rid_to_primary_window
        self._rid_to_vidx_u32: np.ndarray | None = None
        self._empty_doc_count: int = empty_docs
        self._mm: mmap.mmap | None = None
        self._file: Any | None = None
        self._vpack_path: Path | None = None
        self._vpack_mode: str | None = None
        self._closed: bool = False

    @classmethod
    def from_vpack(
        cls,
        vpack_path: Path | str,
        *,
        trusted_sidecar_digest: str | None = None,
        rerank_top_r: int = 256,
        use_neon: bool = True,
    ) -> CompactRetrievalSemanticIndex:
        """Open a RefSpec v2 ``.vpack`` binary sidecar with zero-copy ``mmap`` binding.

        Supports all three ``.vpack`` storage modes directly without copying vectors or
        building Python ``rid`` dictionaries at open time:
          * ``binary_i8`` -> ``CompactRetrievalIndex.from_prequantized(mode="binary_int8", ...)``
            (ARM64 NEON ``vcntq_u8`` Hamming prefilter + ``vdotq_s32`` INT8 rerank on mmap pointers)
          * ``ivf_i8`` -> ``CompactRetrievalIndex.from_prequantized(mode="int8", ...)``
            (exact full ARM64 NEON ``vdotq_s32`` INT8 scan on mmap pointers, zero IVF cluster misses)
          * ``flat_f16`` -> ``CompactRetrievalIndex.from_prequantized(mode="fp16", ...)``
            (exact FP16 scan on mmap pointers)
        """
        eff_rerank_top_r = _validate_positive_int(rerank_top_r, "rerank_top_r")
        raw_path = Path(vpack_path).expanduser()
        if raw_path.is_symlink():
            raise ValueError(f"vpack path must not be a symlink: {vpack_path}")
        try:
            resolved = raw_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"vpack file not found: {vpack_path}") from exc
        if not resolved.is_file():
            raise ValueError(f"vpack path is not a regular file: {resolved}")

        if trusted_sidecar_digest is not None:
            if not isinstance(trusted_sidecar_digest, str) or not trusted_sidecar_digest.strip():
                raise ValueError("`trusted_sidecar_digest` must be a non-empty string when provided")
            norm_digest = trusted_sidecar_digest.strip().lower()
            if not norm_digest.startswith("sha256:"):
                norm_digest = f"sha256:{norm_digest}"
            actual_digest = _sha256_file(resolved)
            if actual_digest != norm_digest:
                raise ValueError(
                    f"vpack digest mismatch for {resolved}: expected {norm_digest}, got {actual_digest}"
                )

        try:
            file_obj = resolved.open("rb")
        except OSError as exc:
            raise ValueError(f"Unable to open vpack file {resolved}: {exc}") from exc

        mm: mmap.mmap | None = None
        rid_offsets_u32: np.ndarray | None = None
        rids_u32: np.ndarray | None = None
        rid_to_vidx_u32: np.ndarray | None = None
        vecs_view: np.ndarray | None = None
        scales_view: np.ndarray | None = None
        binary_view: np.ndarray | None = None
        index: CompactRetrievalIndex | None = None
        obj: CompactRetrievalSemanticIndex | None = None

        try:
            file_size = resolved.stat().st_size
            min_prefix = len(_VPACK_MAGIC) + 8
            if file_size < min_prefix:
                raise ValueError(
                    f"vpack file is empty or truncated ({file_size} bytes): {resolved}"
                )
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            mm_len = len(mm)
            if mm_len < min_prefix or mm[: len(_VPACK_MAGIC)] != _VPACK_MAGIC:
                raise ValueError(f"Invalid vpack magic header in {resolved}")

            header_len = struct.unpack("<Q", mm[len(_VPACK_MAGIC) : min_prefix])[0]
            h_start = min_prefix
            h_end = h_start + header_len
            if header_len == 0 or h_end > mm_len:
                raise ValueError(
                    f"Truncated or invalid vpack header length ({header_len}) in {resolved}"
                )

            try:
                header = json.loads(mm[h_start:h_end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed vpack JSON header in {resolved}: {exc}") from exc
            if not isinstance(header, dict):
                raise ValueError(f"vpack JSON header must be an object in {resolved}")

            schema_version = header.get("schemaVersion")
            if schema_version != _VPACK_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported vpack schemaVersion {schema_version!r} "
                    f"(expected {_VPACK_SCHEMA_VERSION!r}) in {resolved}"
                )

            vpack_mode = header.get("mode")
            if not isinstance(vpack_mode, str) or vpack_mode not in ("binary_i8", "ivf_i8", "flat_f16"):
                raise ValueError(f"Unsupported vpack mode {vpack_mode!r} in {resolved}")

            def _req_int(key: str, *, min_val: int = 0) -> int:
                val = header.get(key)
                if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
                    raise ValueError(
                        f"Invalid vpack header field {key!r}: expected int >= {min_val}, got {val!r}"
                    )
                return val

            def _req_str(key: str, *, allow_empty: bool = False) -> str:
                val = header.get(key)
                if not isinstance(val, str) or (not allow_empty and not val.strip()):
                    raise ValueError(f"Invalid vpack header field {key!r}: got {val!r}")
                return val.strip()

            dim = _req_int("dimensions", min_val=1)
            n_vecs = _req_int("uniqueVectorCount", min_val=0)
            mapped_count = _req_int("mappedResourceCount", min_val=0)
            max_rid = _req_int("maxRid", min_val=0)
            if "nClusters" in header:
                n_clusters = _req_int("nClusters", min_val=0)
            else:
                n_clusters = 0

            pack_sha256 = _req_str("packSha256")
            source_manifest_digest = _req_str("sourceManifestDigest")
            model_id = _req_str("modelId")
            pack_id = _req_str("packId", allow_empty=True) if "packId" in header else ""

            secs = header.get("sections")
            if not isinstance(secs, dict):
                raise ValueError(f"vpack header 'sections' must be a mapping in {resolved}")

            def _checked_section(name: str, expected_bytes: int, *, align: int = 1) -> int:
                meta = secs.get(name)
                if not isinstance(meta, dict):
                    raise ValueError(f"vpack missing or invalid section {name!r} in {resolved}")
                offset = meta.get("offset")
                nbytes = meta.get("nbytes")
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or isinstance(nbytes, bool)
                    or not isinstance(nbytes, int)
                ):
                    raise ValueError(
                        f"vpack section {name!r} offset/nbytes must be integers in {resolved}"
                    )
                if (
                    offset < h_end
                    or nbytes < expected_bytes
                    or offset + expected_bytes > mm_len
                    or offset + nbytes > mm_len
                ):
                    raise ValueError(
                        f"vpack section {name!r} bounds invalid "
                        f"(offset={offset}, nbytes={nbytes}, expected={expected_bytes}, "
                        f"header_end={h_end}, file_size={mm_len})"
                    )
                if align > 1 and offset % align != 0:
                    raise ValueError(
                        f"vpack section {name!r} offset {offset} is not {align}-byte aligned in {resolved}"
                    )
                return offset

            # Validate ALL section offsets and byte ranges BEFORE creating any np.frombuffer views!
            off_rid_offsets = _checked_section("rid_offsets_u32", (n_vecs + 1) * 4, align=4)
            off_rids = _checked_section("rids_u32", mapped_count * 4, align=4)
            off_rid_to_vidx = _checked_section("rid_to_vidx_u32", (max_rid + 1) * 4, align=4)

            off_vecs: int
            off_scales: int | None = None
            off_binary: int | None = None
            n_bytes_bin = (dim + 7) // 8

            if vpack_mode == "binary_i8":
                off_vecs = _checked_section("vectors", n_vecs * dim)
                off_scales = _checked_section("scales_f16", n_vecs * 2, align=2)
                off_binary = _checked_section("binary_u8", n_vecs * n_bytes_bin)
            elif vpack_mode == "ivf_i8":
                off_vecs = _checked_section("vectors", n_vecs * dim)
                off_scales = _checked_section("scales_f16", n_vecs * 2, align=2)
                if "centroids_f32" in secs:
                    _checked_section("centroids_f32", n_clusters * dim * 4, align=4)
                if "cluster_offsets_u32" in secs:
                    _checked_section("cluster_offsets_u32", (n_clusters + 1) * 4, align=4)
            else:
                off_vecs = _checked_section("vectors", n_vecs * dim * 2, align=2)

            # Bind ZERO-COPY mmap views.
            rid_offsets_u32 = np.frombuffer(
                mm, dtype=np.uint32, count=n_vecs + 1, offset=off_rid_offsets
            )
            rids_u32 = np.frombuffer(
                mm, dtype=np.uint32, count=mapped_count, offset=off_rids
            )
            rid_to_vidx_u32 = np.frombuffer(
                mm, dtype=np.uint32, count=max_rid + 1, offset=off_rid_to_vidx
            )
            if int(rid_offsets_u32[0]) != 0 or int(rid_offsets_u32[-1]) != mapped_count:
                raise ValueError(
                    f"vpack rid_offsets_u32 endpoints invalid "
                    f"(start={int(rid_offsets_u32[0])}, end={int(rid_offsets_u32[-1])}, "
                    f"mappedResourceCount={mapped_count})"
                )
            if n_vecs > 0 and bool(np.any(rid_offsets_u32[1:] < rid_offsets_u32[:-1])):
                raise ValueError(
                    f"vpack rid_offsets_u32 must be monotonically non-decreasing in {resolved}"
                )

            if vpack_mode == "binary_i8":
                assert off_scales is not None and off_binary is not None
                vecs_view = np.frombuffer(
                    mm, dtype=np.int8, count=n_vecs * dim, offset=off_vecs
                ).reshape(n_vecs, dim)
                scales_view = np.frombuffer(
                    mm, dtype=np.float16, count=n_vecs, offset=off_scales
                )
                binary_view = np.frombuffer(
                    mm, dtype=np.uint8, count=n_vecs * n_bytes_bin, offset=off_binary
                ).reshape(n_vecs, n_bytes_bin)
                index = CompactRetrievalIndex.from_prequantized(
                    mode="binary_int8",
                    dim=dim,
                    codes_i8=vecs_view,
                    scales_f16=scales_view,
                    binary_u8=binary_view,
                    binary_bits=dim,
                    use_neon=use_neon,
                )
            elif vpack_mode == "ivf_i8":
                assert off_scales is not None
                vecs_view = np.frombuffer(
                    mm, dtype=np.int8, count=n_vecs * dim, offset=off_vecs
                ).reshape(n_vecs, dim)
                scales_view = np.frombuffer(
                    mm, dtype=np.float16, count=n_vecs, offset=off_scales
                )
                index = CompactRetrievalIndex.from_prequantized(
                    mode="int8",
                    dim=dim,
                    codes_i8=vecs_view,
                    scales_f16=scales_view,
                    use_neon=use_neon,
                )
            else:
                vecs_view = np.frombuffer(
                    mm, dtype=np.float16, count=n_vecs * dim, offset=off_vecs
                ).reshape(n_vecs, dim)
                index = CompactRetrievalIndex.from_prequantized(
                    mode="fp16",
                    dim=dim,
                    dense_vecs=vecs_view,
                    use_neon=use_neon,
                )

            obj = cls.__new__(cls)
            obj._index = index
            obj._pack_sha256 = pack_sha256
            obj._pack_id = pack_id
            obj._source_manifest_digest = source_manifest_digest
            obj._model_id = model_id
            obj._agg = "maxsim"
            obj._hybrid_alpha = 0.7
            obj._rerank_top_r = eff_rerank_top_r
            obj._rid_offsets = rid_offsets_u32
            obj._rids = rids_u32
            obj._rid_to_primary_window = None
            obj._rid_to_vidx_u32 = rid_to_vidx_u32
            obj._empty_doc_count = 0
            obj._mm = mm
            obj._file = file_obj
            obj._vpack_path = resolved
            obj._vpack_mode = vpack_mode
            obj._closed = False
            return obj
        except BaseException:
            # Release every local mmap-backed buffer view BEFORE closing mm to prevent BufferError.
            if obj is not None:
                obj._rid_offsets = None  # type: ignore[assignment]
                obj._rids = None  # type: ignore[assignment]
                obj._rid_to_vidx_u32 = None
                obj._index = None  # type: ignore[assignment]
                obj._mm = None
                obj._file = None
                obj = None
            if index is not None:
                index.codes_i8 = None
                index.scales_f16 = None
                index.binary_u8 = None
                index.dense_vecs = None
                index.doc_offsets = None
                index.window_to_doc = None
            rid_offsets_u32 = None
            rids_u32 = None
            rid_to_vidx_u32 = None
            vecs_view = None
            scales_view = None
            binary_view = None
            if mm is not None:
                with contextlib.suppress(BufferError):
                    mm.close()
            file_obj.close()
            raise

    @classmethod
    def from_pack_metadata(
        cls,
        pack_path: Path | str,
        vectors: np.ndarray | CompactRetrievalIndex,
        rids_per_doc: Sequence[int | Sequence[int]],
        *,
        doc_offsets: np.ndarray | Sequence[int] | None = None,
        model_id: str,
        mode: IndexMode = "auto",
        is_mrl_trained: bool = False,
        mrl_dim: int | None = None,
        binary_bits: int | None = None,
        agg: WindowAgg | None = None,
        hybrid_alpha: float | None = None,
        rerank_top_r: int = 256,
        use_neon: bool = True,
    ) -> CompactRetrievalSemanticIndex:
        """Build a ``CompactRetrievalSemanticIndex`` pinned to a compiled RefSpec SQLite pack."""
        _, pack_sha256, pack_id, source_manifest_digest, _ = _read_pack_sqlite_metadata(pack_path)
        return cls(
            vectors,
            rids_per_doc,
            pack_sha256=pack_sha256,
            pack_id=pack_id,
            source_manifest_digest=source_manifest_digest,
            model_id=model_id,
            doc_offsets=doc_offsets,
            mode=mode,
            is_mrl_trained=is_mrl_trained,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            rerank_top_r=rerank_top_r,
            use_neon=use_neon,
        )

    @property
    def pack_sha256(self) -> str:
        return self._pack_sha256

    @property
    def pack_id(self) -> str:
        return self._pack_id

    @property
    def source_manifest_digest(self) -> str:
        return self._source_manifest_digest

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return int(self._index.dim)

    @property
    def index(self) -> CompactRetrievalIndex:
        return self._index

    @property
    def mode(self) -> str:
        return self._index.mode

    @property
    def num_docs(self) -> int:
        return int(self._index.num_docs)

    @property
    def num_windows(self) -> int:
        return int(self._index.num_windows)

    @property
    def is_multi_window(self) -> bool:
        return self._index.doc_offsets is not None

    @property
    def nbytes(self) -> int:
        if self._closed:
            return 0
        total = int(self._index.nbytes)
        for buf in (self._rid_offsets, self._rids, self._rid_to_vidx_u32):
            if buf is not None:
                total += int(buf.nbytes)
        return total

    @property
    def compression_ratio_vs_fp32(self) -> float:
        return float(self._index.compression_ratio_vs_fp32)

    @property
    def vpack_mode(self) -> str | None:
        return self._vpack_mode

    def close(self) -> None:
        """Release all mmap-backed buffer views and close mmap/file handles (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._mm is not None:
            # Release ALL mmap-backed NumPy buffer views on both self and self._index
            # before closing mmap (Python raises BufferError if any exported pointer survives).
            if self._index is not None:
                self._index.codes_i8 = None
                self._index.scales_f16 = None
                self._index.binary_u8 = None
                self._index.dense_vecs = None
                self._index.doc_offsets = None
                self._index.window_to_doc = None
            self._rid_offsets = None  # type: ignore[assignment]
            self._rids = None  # type: ignore[assignment]
            self._rid_to_vidx_u32 = None
            try:
                self._mm.close()
            finally:
                self._mm = None
                if self._file is not None:
                    self._file.close()
                    self._file = None
        elif self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> CompactRetrievalSemanticIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def search_rids(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 40,
        min_cosine: float = 0.55,
        nprobe: int = 24,
    ) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(rid, cosine_score)`` pairs sorted descending by cosine similarity."""
        if self._closed:
            raise ValueError("CompactRetrievalSemanticIndex is closed")
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] not in (self._index.dim, self._index.raw_dim):
            raise ValueError(
                f"query_vector must be 1D of length {self._index.dim} "
                f"(or raw_dim={self._index.raw_dim}), got shape {q.shape}"
            )
        if top_k <= 0 or self._index.num_docs == 0:
            return []
        norm = float(np.linalg.norm(q[: self._index.dim]))
        if not math.isfinite(norm) or norm <= 1e-12:
            return []

        eff_k = min(self._index.num_docs, int(top_k) + self._empty_doc_count)
        eff_rerank_r = max(self._rerank_top_r, max(1, int(nprobe)) * 16, eff_k * 4)
        top_doc_ids, top_scores = self._index.search(
            q,
            k=eff_k,
            rerank_top_r=eff_rerank_r,
            agg=self._agg,
            hybrid_alpha=self._hybrid_alpha,
        )

        threshold = float(min_cosine)
        out: list[tuple[int, float]] = []
        seen: set[int] = set()
        for doc_idx_raw, score_raw in zip(top_doc_ids.tolist(), top_scores.tolist()):
            doc_idx = int(doc_idx_raw)
            score = float(score_raw)
            if doc_idx < 0 or not math.isfinite(score) or score < threshold:
                continue
            r0 = int(self._rid_offsets[doc_idx])
            r1 = int(self._rid_offsets[doc_idx + 1])
            for rid_val in self._rids[r0:r1]:
                rid_int = int(rid_val)
                if rid_int not in seen:
                    seen.add(rid_int)
                    out.append((rid_int, score))
                    if len(out) >= top_k:
                        return out
        return out

    def vector_for_rid(self, rid: int) -> np.ndarray | None:
        """Reconstruct the unit-L2-normalized ``float32`` primary vector for ``rid`` in O(1), or ``None``."""
        if self._closed:
            raise ValueError("CompactRetrievalSemanticIndex is closed")
        if isinstance(rid, (bool, np.bool_)) or not isinstance(rid, (int, np.integer)):
            return None
        rid_int = int(rid)
        if self._rid_to_vidx_u32 is not None:
            if rid_int < 0 or rid_int >= self._rid_to_vidx_u32.size:
                return None
            vidx = int(self._rid_to_vidx_u32[rid_int])
            if vidx == _MISSING_VIDX or vidx >= self._index.num_windows:
                return None
            win_idx = vidx
        else:
            assert self._rid_to_primary_window is not None
            win_opt = self._rid_to_primary_window.get(rid_int)
            if win_opt is None:
                return None
            win_idx = win_opt
        if self._index.codes_i8 is not None and self._index.scales_f16 is not None:
            raw = self._index.codes_i8[win_idx].astype(np.float32) * np.float32(
                self._index.scales_f16[win_idx]
            )
            return np.ascontiguousarray(truncate_and_normalize(raw), dtype=np.float32)
        if self._index.dense_vecs is not None:
            raw = self._index.dense_vecs[win_idx].astype(np.float32)
            return np.ascontiguousarray(truncate_and_normalize(raw), dtype=np.float32)
        return None


# ---------------------------------------------------------------------------
# T3.3: LayaRerankFusion (RefSpec CandidateReranker structural adapter)
# ---------------------------------------------------------------------------


def _short_scheme_name(scheme: Any) -> str:
    raw = str(scheme or "").strip()
    if raw.startswith(_SCHEME_URN_PREFIX):
        return raw[len(_SCHEME_URN_PREFIX) :]
    return raw


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    prefix = text[: max_chars - 3].rstrip(" ,;:-") or text[: max_chars - 3].rstrip()
    return prefix + "..."


def format_candidate_concept_summary(
    candidate: Mapping[str, Any],
    *,
    max_definition_chars: int = 200,
    max_broader: int = 4,
    max_label_chars: int = 160,
    max_broader_chars: int = 120,
) -> str:
    """Format a hydrated RefSpec concept dict into a compact summary for Laya ``head_max_len=192``."""
    label = " ".join(str(candidate.get("label") or "").split())
    if label and max_label_chars > 0 and len(label) > max_label_chars:
        label = _truncate_with_ellipsis(label, int(max_label_chars))
    scheme = _short_scheme_name(candidate.get("scheme"))
    defn_raw = str(candidate.get("definition") or "").strip()
    broader_raw = candidate.get("broaderLabels")

    parts: list[str] = [f"Concept: {label}" if label else "Concept: (unlabeled)"]
    if scheme:
        parts.append(f"Scheme: {scheme}")

    if (
        max_broader > 0
        and max_broader_chars > 0
        and isinstance(broader_raw, Sequence)
        and not isinstance(broader_raw, (str, bytes))
    ):
        broader_items = [
            " ".join(str(b).split()) for b in broader_raw if str(b).strip()
        ][: int(max_broader)]
        if broader_items:
            broader_str = ", ".join(broader_items)
            if len(broader_str) > max_broader_chars:
                broader_str = _truncate_with_ellipsis(broader_str, int(max_broader_chars))
            parts.append("Broader: " + broader_str)

    if defn_raw and max_definition_chars > 0:
        clean_def = " ".join(defn_raw.split())
        if len(clean_def) > max_definition_chars:
            clean_def = clean_def[: max(1, max_definition_chars - 1)].rstrip() + "…"
        parts.append(f"Definition: {clean_def}")

    return " | ".join(parts)


def build_rerank_noul_qdef(
    candidate: Mapping[str, Any],
    *,
    index: int = 0,
    max_definition_chars: int = 200,
    max_broader: int = 4,
    max_label_chars: int = 160,
    max_broader_chars: int = 120,
) -> dict[str, Any]:
    """Build a valid Laya ``noul`` typed-question payload judging query-to-concept relevance.

    Sized deliberately to stay well below Laya's ``head_max_len=192`` token budget (0% truncation)
    and land in Laya's fastest ``s128``/``s192`` ANE sequence buckets.
    """
    summary = format_candidate_concept_summary(
        candidate,
        max_definition_chars=max_definition_chars,
        max_broader=max_broader,
        max_label_chars=max_label_chars,
        max_broader_chars=max_broader_chars,
    )
    return {
        "id": f"refspec_rerank_{int(index)}",
        "type": "noul",
        "instructions": (
            "Does this vocabulary concept directly and concisely match the user query? "
            + summary
        ),
        "criteria": {
            "false": "Unrelated, ambiguous, or an overly narrow subdivision.",
            "true": "Direct, concise, accurate vocabulary match for the query.",
        },
    }


def _is_exact_candidate(candidate: Mapping[str, Any], query_lower: str) -> bool:
    reason = candidate.get("matchReason") or candidate.get("_matchReason")
    if reason in _EXACT_MATCH_REASONS:
        return True
    if not query_lower:
        return False
    label_lower = str(candidate.get("label") or "").strip().lower()
    if label_lower == query_lower:
        return True
    labels = candidate.get("labels")
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        return any(
            isinstance(lbl, Mapping) and str(lbl.get("text") or "").strip().lower() == query_lower
            for lbl in labels
        )
    return False


def _copy_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    card = dict(candidate)
    if isinstance(card.get("counterparts"), list):
        card["counterparts"] = [dict(cp) if isinstance(cp, Mapping) else cp for cp in card["counterparts"]]
    return card


def _deduplicate_by_label(ordered: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for item in ordered:
        key = str(item.get("label") or "").strip().lower()
        existing = by_label.get(key)
        if existing is None:
            if len(grouped) >= limit:
                continue
            card = _copy_candidate(item)
            by_label[key] = card
            grouped.append(card)
        else:
            counterparts = existing.setdefault("counterparts", [])
            counterparts.append(
                {
                    "id": item.get("id"),
                    "scheme": item.get("scheme"),
                    "release": item.get("release"),
                    "label": item.get("label"),
                    "definition": item.get("definition"),
                    "sourceRecord": item.get("sourceRecord"),
                }
            )
    return grouped


def _validate_binary_profile(profile: Any, *, name: str = "binary_profile") -> Any:
    if profile is None:
        return None
    if callable(getattr(profile, "apply", None)):
        return profile
    if (
        isinstance(profile, Sequence)
        and not isinstance(profile, (str, bytes))
        and len(profile) == 2
    ):
        a_raw, b_raw = profile[0], profile[1]
        if isinstance(a_raw, (bool, np.bool_)) or isinstance(b_raw, (bool, np.bool_)):
            raise ValueError(f"{name} Platt tuple must contain floats, got {profile!r}")
        a, b = float(a_raw), float(b_raw)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"{name} Platt parameters must be finite, got {profile!r}")
        return (a, b)
    raise TypeError(
        f"`{name}` must implement `.apply(probs, from_probs=True)` or be a `(platt_a, platt_b)` pair"
    )


def _apply_binary_calibration(probs: Sequence[float], profile: Any | None) -> list[float]:
    if not probs:
        return []
    if profile is None:
        return [float(p) for p in probs]
    arr = np.asarray(probs, dtype=np.float64)
    apply_fn = getattr(profile, "apply", None)
    if callable(apply_fn):
        if accepts_kwarg(apply_fn, "from_probs"):
            cal = np.asarray(apply_fn(arr, from_probs=True), dtype=np.float64).reshape(-1)
        else:
            cal = np.asarray(apply_fn(arr), dtype=np.float64).reshape(-1)
    else:
        a, b = float(profile[0]), float(profile[1])
        p_clamped = np.clip(arr, 1e-7, 1.0 - 1e-7)
        logits = np.log(p_clamped / (1.0 - p_clamped))
        z = np.clip(a * logits + b, -30.0, 30.0)
        cal = 1.0 / (1.0 + np.exp(-z))
    if cal.size != len(probs):
        raise RuntimeError(
            f"binary_profile returned {cal.size} calibrated probabilities for {len(probs)} inputs"
        )
    out: list[float] = []
    for v in cal.tolist():
        fv = float(v)
        out.append(0.5 if not math.isfinite(fv) else max(0.0, min(1.0, fv)))
    return out


def _sanitize_prob_list(probs: Sequence[float], expected_len: int, backend_label: str) -> list[float]:
    if len(probs) != expected_len:
        raise RuntimeError(
            f"{backend_label} returned {len(probs)} scores for {expected_len} candidates"
        )
    clean: list[float] = []
    for p in probs:
        fp = float(p)
        clean.append(0.5 if not math.isfinite(fp) else max(0.0, min(1.0, fp)))
    return clean


class LayaRerankFusion:
    """RefSpec ``CandidateReranker`` fusing batched Laya ``noul`` scores with strict Tier-0 exact pinning.

    Accepts any injected Laya backend (``LayaAneBackend``, ``LayaHeteroEngine``,
    ``LayaMlxGpuBackend``, or a lightweight test fake implementing ``encode_items_for_state``
    or ``encode_state`` + ``encode_item`` along with ``score_noul_batch`` or ``score_choice_batch``),
    plus optional Platt probability calibration (``binary_profile``) and lazy confidence-cascade
    escalation to a stronger scorer (such as ``Kev08bMlxBackend`` via ``score_multi_states``,
    with optional ``escalation_binary_profile``).

    When ``attach_score=True``, each reranked candidate's ``layaScore`` field holds the final
    post-calibration/post-escalation rerank probability (matching ``rerankConfidence`` when
    ``attach_confidence=True``).
    """

    def __init__(
        self,
        backend: Any | None = None,
        *,
        backend_factory: Callable[[], Any] | None = None,
        max_rerank: int = 20,
        prior_weight: float = 0.25,
        deduplicate_labels: bool = True,
        attach_score: bool = True,
        max_definition_chars: int = 200,
        max_broader: int = 4,
        max_label_chars: int = 160,
        max_broader_chars: int = 120,
        max_len: int | None = None,
        on_truncate: Literal["warn", "error", "ignore"] = "ignore",
        fallback_on_error: bool = False,
        binary_profile: Any | None = None,
        escalation_binary_profile: Any | None = None,
        escalation_backend: Any | None = None,
        escalation_factory: Callable[[], Any] | None = None,
        uncertainty_band: tuple[float, float] = (0.30, 0.70),
        escalation_margin: float = 0.08,
        max_escalation_candidates: int = 8,
        kev_batch_docs: int = 8,
        high_confidence_threshold: float = 0.70,
        low_confidence_threshold: float = 0.35,
        attach_confidence: bool = False,
    ) -> None:
        if (backend is None) == (backend_factory is None):
            raise ValueError("Provide exactly one of `backend` or `backend_factory`")
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("`backend_factory` must be callable")
        if escalation_backend is not None and escalation_factory is not None:
            raise ValueError("Provide at most one of `escalation_backend` or `escalation_factory`")
        if escalation_factory is not None and not callable(escalation_factory):
            raise TypeError("`escalation_factory` must be callable")
        prior_w = _validate_finite_float(prior_weight, "prior_weight")
        if not (0.0 <= prior_w <= 1.0):
            raise ValueError(f"prior_weight must be in [0.0, 1.0], got {prior_weight!r}")
        if on_truncate not in ("warn", "error", "ignore"):
            raise ValueError(f"on_truncate must be 'warn', 'error', or 'ignore', got {on_truncate!r}")

        if (
            not isinstance(uncertainty_band, Sequence)
            or isinstance(uncertainty_band, (str, bytes))
            or len(uncertainty_band) != 2
        ):
            raise ValueError(f"uncertainty_band must be a (low, high) pair, got {uncertainty_band!r}")
        band_lo = _validate_finite_float(uncertainty_band[0], "uncertainty_band[0]")
        band_hi = _validate_finite_float(uncertainty_band[1], "uncertainty_band[1]")
        if not (0.0 <= band_lo <= band_hi <= 1.0):
            raise ValueError(
                f"uncertainty_band must satisfy 0.0 <= low <= high <= 1.0, got {uncertainty_band!r}"
            )

        margin_f = _validate_finite_float(escalation_margin, "escalation_margin")
        if margin_f < 0.0:
            raise ValueError(f"escalation_margin must be finite and >= 0.0, got {escalation_margin!r}")

        hi_thr = _validate_finite_float(high_confidence_threshold, "high_confidence_threshold")
        lo_thr = _validate_finite_float(low_confidence_threshold, "low_confidence_threshold")
        if not (0.0 <= lo_thr <= hi_thr <= 1.0):
            raise ValueError(
                "confidence thresholds must satisfy 0.0 <= low_confidence_threshold <= "
                f"high_confidence_threshold <= 1.0, got ({lo_thr!r}, {hi_thr!r})"
            )

        self._backend: Any | None = backend
        self._backend_factory: Callable[[], Any] | None = backend_factory
        self._owns_backend: bool = False
        self._escalation_backend: Any | None = escalation_backend
        self._escalation_factory: Callable[[], Any] | None = escalation_factory
        self._owns_escalation: bool = False
        self._lock = threading.Lock()

        self.max_rerank: int = _validate_positive_int(max_rerank, "max_rerank")
        self.prior_weight: float = prior_w
        self.deduplicate_labels: bool = bool(deduplicate_labels)
        self.attach_score: bool = bool(attach_score)
        self.max_definition_chars: int = max(0, int(max_definition_chars))
        self.max_broader: int = max(0, int(max_broader))
        self.max_label_chars: int = max(0, int(max_label_chars))
        self.max_broader_chars: int = max(0, int(max_broader_chars))
        self.max_len: int | None = (
            _validate_positive_int(max_len, "max_len") if max_len is not None else None
        )
        self.on_truncate: Literal["warn", "error", "ignore"] = on_truncate
        self.fallback_on_error: bool = bool(fallback_on_error)
        self.binary_profile: Any | None = _validate_binary_profile(binary_profile)
        self.escalation_binary_profile: Any | None = _validate_binary_profile(
            escalation_binary_profile, name="escalation_binary_profile"
        )
        self.uncertainty_band: tuple[float, float] = (band_lo, band_hi)
        self.escalation_margin: float = margin_f
        self.max_escalation_candidates: int = _validate_positive_int(
            max_escalation_candidates, "max_escalation_candidates"
        )
        self.kev_batch_docs: int = _validate_positive_int(kev_batch_docs, "kev_batch_docs")
        self.high_confidence_threshold: float = hi_thr
        self.low_confidence_threshold: float = lo_thr
        self.attach_confidence: bool = bool(attach_confidence)

    def _get_backend(self) -> Any:
        be = self._backend
        if be is not None:
            return be
        with self._lock:
            be = self._backend
            if be is not None:
                return be
            assert self._backend_factory is not None
            be = self._backend_factory()
            self._backend = be
            self._owns_backend = True
            return be

    def _get_escalation_backend(self) -> Any | None:
        be = self._escalation_backend
        if be is not None:
            return be
        if self._escalation_factory is None:
            return None
        with self._lock:
            be = self._escalation_backend
            if be is not None:
                return be
            be = self._escalation_factory()
            self._escalation_backend = be
            self._owns_escalation = True
            return be

    @property
    def is_initialized(self) -> bool:
        return self._backend is not None

    @property
    def is_escalation_initialized(self) -> bool:
        return self._escalation_backend is not None

    def _classify_decision(
        self, score: float
    ) -> Literal["high_confidence", "disambiguate", "low_confidence"]:
        if score >= self.high_confidence_threshold:
            return "high_confidence"
        if score < self.low_confidence_threshold:
            return "low_confidence"
        return "disambiguate"

    def _build_qdefs(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            build_rerank_noul_qdef(
                cand,
                index=i,
                max_definition_chars=self.max_definition_chars,
                max_broader=self.max_broader,
                max_label_chars=self.max_label_chars,
                max_broader_chars=self.max_broader_chars,
            )
            for i, cand in enumerate(candidates)
        ]

    def _score_with_laya_style_backend(
        self,
        backend: Any,
        state: dict[str, str],
        qdefs: Sequence[dict[str, Any]],
        *,
        backend_label: str = "Laya backend",
    ) -> list[float]:
        enc_batch_fn = getattr(backend, "encode_items_for_state", None)
        if callable(enc_batch_fn):
            enc_kwargs: dict[str, Any] = {}
            if self.max_len is not None and accepts_kwarg(enc_batch_fn, "max_len"):
                enc_kwargs["max_len"] = self.max_len
            if accepts_kwarg(enc_batch_fn, "on_truncate"):
                enc_kwargs["on_truncate"] = self.on_truncate
            encoded_items = enc_batch_fn(state, qdefs, **enc_kwargs)
        else:
            enc_item_fn = getattr(backend, "encode_item", None)
            if not callable(enc_item_fn):
                raise TypeError(
                    f"Injected {backend_label} {type(backend).__name__!r} must implement "
                    "`encode_items_for_state` or `encode_item`"
                )
            enc_state_fn = getattr(backend, "encode_state", None)
            st_ids = enc_state_fn(state) if callable(enc_state_fn) else None
            encoded_items = []
            for qd in qdefs:
                item_kwargs: dict[str, Any] = {}
                if self.max_len is not None and accepts_kwarg(enc_item_fn, "max_len"):
                    item_kwargs["max_len"] = self.max_len
                if accepts_kwarg(enc_item_fn, "on_truncate"):
                    item_kwargs["on_truncate"] = self.on_truncate
                if st_ids is not None and accepts_kwarg(enc_item_fn, "state_ids"):
                    item_kwargs["state_ids"] = st_ids
                encoded_items.append(enc_item_fn(state, qd, **item_kwargs))

        score_noul_fn = getattr(backend, "score_noul_batch", None)
        score_choice_fn = getattr(backend, "score_choice_batch", None)

        if callable(score_noul_fn):
            raw_probs = score_noul_fn(encoded_items)
            probs = [float(p) for p in raw_probs]
        elif callable(score_choice_fn):
            choice_probs = score_choice_fn(encoded_items)
            probs = [
                float(row[1]) if len(row) >= 2 else (float(row[0]) if row else 0.5)
                for row in choice_probs
            ]
        else:
            raise TypeError(
                f"Injected {backend_label} {type(backend).__name__!r} must implement "
                "`score_noul_batch` or `score_choice_batch`"
            )

        return _sanitize_prob_list(probs, len(qdefs), backend_label)

    def _score_candidates_laya(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        *,
        qdefs: Sequence[dict[str, Any]] | None = None,
    ) -> list[float]:
        backend = self._get_backend()
        state: dict[str, str] = {"query": query}
        eff_qdefs = list(qdefs) if qdefs is not None else self._build_qdefs(candidates)
        return self._score_with_laya_style_backend(
            backend, state, eff_qdefs, backend_label="Laya backend"
        )

    def _score_candidates_escalation(
        self,
        query: str,
        qdefs: Sequence[dict[str, Any]],
    ) -> list[float]:
        esc_backend = self._get_escalation_backend()
        if esc_backend is None:
            raise RuntimeError("Escalation requested without an escalation backend")
        state: dict[str, str] = {"query": query}
        eff_qdefs = list(qdefs)

        score_multi_fn = getattr(esc_backend, "score_multi_states", None)
        if callable(score_multi_fn):
            kw: dict[str, Any] = {}
            if accepts_kwarg(score_multi_fn, "batch_docs"):
                kw["batch_docs"] = self.kev_batch_docs
            raw = score_multi_fn([(state, eff_qdefs)], **kw)
            if (
                not isinstance(raw, (Sequence, np.ndarray))
                or isinstance(raw, (str, bytes))
                or len(raw) != 1
                or len(raw[0]) != len(eff_qdefs)
            ):
                raise RuntimeError(
                    f"Escalation backend returned unexpected shape for {len(eff_qdefs)} candidates"
                )
            probs: list[float] = []
            for q_res in raw[0]:
                # Kev08bMlxBackend returns (probs_list, raw_logits) per question where
                # `noul` options are ordered [no (false), yes (true)] -> P(true) = probs[1].
                if (
                    isinstance(q_res, (tuple, list))
                    and len(q_res) >= 1
                    and isinstance(q_res[0], (Sequence, np.ndarray))
                    and not isinstance(q_res[0], (str, bytes))
                ):
                    p_entry = q_res[0]
                else:
                    p_entry = q_res
                if (
                    isinstance(p_entry, (int, float, np.integer, np.floating))
                    and not isinstance(p_entry, (bool, np.bool_))
                ):
                    probs.append(float(p_entry))
                elif isinstance(p_entry, (Sequence, np.ndarray)) and not isinstance(
                    p_entry, (str, bytes)
                ):
                    p_arr = np.asarray(p_entry, dtype=np.float64).reshape(-1)
                    if p_arr.size >= 2:
                        probs.append(float(p_arr[1]))
                    elif p_arr.size == 1:
                        probs.append(float(p_arr[0]))
                    else:
                        probs.append(0.5)
                else:
                    probs.append(0.5)
            return _sanitize_prob_list(probs, len(eff_qdefs), "Escalation backend")

        score_multi_noul_fn = getattr(esc_backend, "score_multi_states_noul", None)
        if callable(score_multi_noul_fn):
            kw = {}
            if accepts_kwarg(score_multi_noul_fn, "batch_docs"):
                kw["batch_docs"] = self.kev_batch_docs
            raw = score_multi_noul_fn([(state, eff_qdefs)], **kw)
            if not isinstance(raw, Sequence) or len(raw) != 1 or len(raw[0]) != len(eff_qdefs):
                raise RuntimeError(
                    f"Escalation backend returned unexpected shape for {len(eff_qdefs)} candidates"
                )
            probs = [
                float(item[0]) if isinstance(item, tuple) and len(item) >= 1 else float(item)
                for item in raw[0]
            ]
            return _sanitize_prob_list(probs, len(eff_qdefs), "Escalation backend")

        if callable(getattr(esc_backend, "encode_items_for_state", None)) or callable(
            getattr(esc_backend, "encode_item", None)
        ):
            return self._score_with_laya_style_backend(
                esc_backend, state, eff_qdefs, backend_label="Escalation backend"
            )

        score_noul_fn = getattr(esc_backend, "score_noul_batch", None)
        if callable(score_noul_fn):
            raw_probs = score_noul_fn([{"state": state, "qdef": qd} for qd in eff_qdefs])
            return _sanitize_prob_list(
                [float(p) for p in raw_probs], len(eff_qdefs), "Escalation backend"
            )

        raise TypeError(
            f"Injected escalation backend {type(esc_backend).__name__!r} must implement "
            "`score_multi_states`, `score_multi_states_noul`, or `score_noul_batch`"
        )

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` reranked concept dicts with Tier-0 exact matches strictly pinned."""
        if limit <= 0 or not candidates:
            return []

        q_clean = str(query or "").strip()
        ql = q_clean.lower()

        exact_cards: list[dict[str, Any]] = []
        non_exact_cards: list[dict[str, Any]] = []
        for item in candidates:
            card = _copy_candidate(item)
            if _is_exact_candidate(card, ql):
                if self.attach_confidence:
                    card["rerankConfidence"] = 1.0
                    card["rerankEscalated"] = False
                    card["rerankDecision"] = "exact"
                exact_cards.append(card)
            else:
                non_exact_cards.append(card)

        # Skip Laya inference entirely if pinned Tier-0 exact matches already satisfy `limit`.
        exact_fills_limit = (
            len({str(c.get("label") or "").strip().lower() for c in exact_cards}) >= limit
            if self.deduplicate_labels
            else len(exact_cards) >= limit
        )

        reranked_non_exact: list[dict[str, Any]]
        if not non_exact_cards or exact_fills_limit or not q_clean:
            if self.attach_confidence and not exact_fills_limit:
                for card in non_exact_cards:
                    card["rerankConfidence"] = 0.0
                    card["rerankEscalated"] = False
                    card["rerankDecision"] = "low_confidence"
            reranked_non_exact = non_exact_cards
        else:
            head = non_exact_cards[: self.max_rerank]
            tail = non_exact_cards[self.max_rerank :]
            qdefs = self._build_qdefs(head)
            try:
                raw_laya_probs = self._score_candidates_laya(q_clean, head, qdefs=qdefs)
                cal_probs = _apply_binary_calibration(raw_laya_probs, self.binary_profile)
            except Exception:
                if not self.fallback_on_error:
                    raise
                if self.attach_confidence:
                    for card in non_exact_cards:
                        card["rerankConfidence"] = 0.0
                        card["rerankEscalated"] = False
                        card["rerankDecision"] = "low_confidence"
                reranked_non_exact = non_exact_cards
            else:
                m = len(head)
                w_prior = self.prior_weight
                w_laya = 1.0 - w_prior
                priors = [1.0 - (float(pos) / float(max(1, m))) for pos in range(m)]
                primary_blended = [
                    w_laya * p_cal + w_prior * priors[pos] for pos, p_cal in enumerate(cal_probs)
                ]
                primary_order = sorted(
                    range(m),
                    key=lambda idx: (-primary_blended[idx], -cal_probs[idx], idx),
                )

                final_probs = list(cal_probs)
                escalated = [False] * m
                has_escalation = (
                    self._escalation_backend is not None or self._escalation_factory is not None
                )
                if has_escalation and primary_order:
                    top_pool = primary_order[: self.max_escalation_candidates]
                    band_lo, band_hi = self.uncertainty_band
                    should_escalate = any(
                        band_lo <= cal_probs[idx] <= band_hi for idx in top_pool
                    )
                    if not should_escalate and len(top_pool) >= 2:
                        i0, i1 = top_pool[0], top_pool[1]
                        if (
                            min(
                                abs(primary_blended[i0] - primary_blended[i1]),
                                abs(cal_probs[i0] - cal_probs[i1]),
                            )
                            <= self.escalation_margin
                        ):
                            should_escalate = True

                    if should_escalate:
                        esc_qdefs = [qdefs[idx] for idx in top_pool]
                        try:
                            raw_esc_probs = self._score_candidates_escalation(q_clean, esc_qdefs)
                            esc_probs = _apply_binary_calibration(
                                raw_esc_probs, self.escalation_binary_profile
                            )
                        except Exception:
                            if not self.fallback_on_error:
                                raise
                        else:
                            for idx, p_esc in zip(top_pool, esc_probs, strict=True):
                                final_probs[idx] = p_esc
                                escalated[idx] = True

                scored: list[tuple[float, float, int, dict[str, Any]]] = []
                for pos, (card, p_final, was_esc) in enumerate(
                    zip(head, final_probs, escalated, strict=True)
                ):
                    blended = w_laya * p_final + w_prior * priors[pos]
                    if self.attach_score:
                        card["layaScore"] = round(p_final, 6)
                    if self.attach_confidence:
                        card["rerankConfidence"] = round(p_final, 6)
                        card["rerankEscalated"] = bool(was_esc)
                        card["rerankDecision"] = self._classify_decision(p_final)
                    scored.append((blended, p_final, pos, card))
                scored.sort(key=lambda tup: (-tup[0], -tup[1], tup[2]))
                if self.attach_confidence:
                    for card in tail:
                        card["rerankConfidence"] = 0.0
                        card["rerankEscalated"] = False
                        card["rerankDecision"] = "low_confidence"
                reranked_non_exact = [tup[3] for tup in scored] + tail

        ordered = exact_cards + reranked_non_exact
        if not self.deduplicate_labels:
            return ordered[:limit]
        return _deduplicate_by_label(ordered, limit)

    def close(self) -> None:
        with self._lock:
            backend, owns_backend = self._backend, self._owns_backend
            esc, owns_esc = self._escalation_backend, self._owns_escalation
            self._backend = None
            self._owns_backend = False
            self._escalation_backend = None
            self._owns_escalation = False
        try:
            if owns_backend and backend is not None:
                close_fn = getattr(backend, "close", None)
                if callable(close_fn):
                    close_fn()
        finally:
            if owns_esc and esc is not None:
                close_fn = getattr(esc, "close", None)
                if callable(close_fn):
                    close_fn()

    def __enter__(self) -> LayaRerankFusion:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# T3.4: Streaming SQLite Pack ETL -> CompactRetrievalSemanticIndex / Parquet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackConceptRecord:
    """One labeled concept streamed from a compiled RefSpec SQLite Vocabulary Pack."""

    rid: int
    resource_id: str
    scheme: str
    release: str
    pref_label: str
    alt_labels: tuple[str, ...]
    definition: str | None
    card_text: str
    windows: tuple[str, ...]
    input_sha256: str


@dataclass(frozen=True, slots=True)
class PackEmbedSummary:
    """Result metadata from ``embed_sqlite_pack_to_parquet``."""

    output_dir: Path
    resources_parquet: Path
    windows_parquet: Path | None
    vector_files: tuple[Path, ...]
    manifest_path: Path
    pack_sha256: str
    pack_id: str
    source_manifest_digest: str
    model_id: str
    dimensions: int
    multi_window: bool
    resource_count: int
    window_count: int
    unique_vector_count: int
    deduplicated_hits: int
    elapsed_seconds: float
    semantic_index: CompactRetrievalSemanticIndex | None = None


def _build_concept_texts(
    pref_label: str,
    alt_labels: Sequence[str],
    definition: str | None,
    *,
    multi_window: bool,
    include_definition: bool,
    max_alt_labels: int,
    max_definition_chars: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    pref_clean = pref_label.strip()
    seen_lower: set[str] = {pref_clean.lower()}
    dedup_alts: list[str] = []
    for alt in alt_labels:
        clean_alt = alt.strip()
        if not clean_alt:
            continue
        low = clean_alt.lower()
        if low in seen_lower:
            continue
        seen_lower.add(low)
        dedup_alts.append(clean_alt)
        if len(dedup_alts) >= max_alt_labels:
            break

    clean_def: str | None = None
    if include_definition and definition:
        compact_def = " ".join(definition.split())
        if compact_def:
            if max_definition_chars > 0 and len(compact_def) > max_definition_chars:
                compact_def = compact_def[: max(1, max_definition_chars - 1)].rstrip() + "…"
            clean_def = compact_def

    card_parts = [pref_clean]
    if dedup_alts:
        card_parts.append("; ".join(dedup_alts))
    if clean_def:
        card_parts.append(clean_def)
    card_text = " | ".join(card_parts)

    if not multi_window:
        return card_text, (card_text,), tuple(dedup_alts)

    windows: list[str] = [pref_clean]
    windows.extend(dedup_alts)
    if clean_def and clean_def.lower() not in seen_lower:
        windows.append(clean_def)
    return card_text, tuple(windows), tuple(dedup_alts)


def export_pack_concept_texts(
    pack_path: Path | str,
    *,
    batch_size: int = 4096,
    schemes: Iterable[str] | None = None,
    multi_window: bool = False,
    include_definition: bool = True,
    max_alt_labels: int = 16,
    max_definition_chars: int = 512,
) -> Iterator[list[PackConceptRecord]]:
    """Stream labeled concepts from a RefSpec SQLite pack in bounded keyset batches.

    Uses ``WHERE r.pref_label IS NOT NULL AND r.rid > ? ORDER BY r.rid LIMIT ?`` so peak
    memory stays strictly O(``batch_size``) even on multi-million-concept packs.
    """
    eff_batch = _validate_positive_int(batch_size, "batch_size")
    eff_max_alts = max(0, int(max_alt_labels))
    eff_max_def = max(0, int(max_definition_chars))
    resolved = Path(pack_path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Vocabulary pack file not found: {pack_path}")

    scheme_filter: set[str] | None = None
    if schemes is not None:
        if isinstance(schemes, (str, bytes)):
            raise ValueError("`schemes` must be an iterable of scheme URIs, not a bare string")
        scheme_filter = {str(s).strip() for s in schemes if str(s).strip()}
        if not scheme_filter:
            return

    uri = f"file:{quote(str(resolved))}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        scheme_by_sid = {int(sid): str(u) for sid, u in con.execute("SELECT sid, uri FROM schemes").fetchall()}
        release_by_id = {int(rid): str(u) for rid, u in con.execute("SELECT rel_id, uri FROM releases").fetchall()}

        allowed_sids: tuple[int, ...] | None = None
        if scheme_filter is not None:
            allowed_sids = tuple(sorted(sid for sid, u in scheme_by_sid.items() if u in scheme_filter))
            if not allowed_sids:
                return

        last_rid = -1
        while True:
            if allowed_sids is not None:
                ph = ",".join("?" for _ in allowed_sids)
                rows = con.execute(
                    f"""
                    SELECT rid, id, sid, rel_id, pref_label, definition
                    FROM resources
                    WHERE pref_label IS NOT NULL
                      AND sid IN ({ph})
                      AND rid > ?
                    ORDER BY rid
                    LIMIT ?
                    """,
                    (*allowed_sids, last_rid, eff_batch),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT rid, id, sid, rel_id, pref_label, definition
                    FROM resources
                    WHERE pref_label IS NOT NULL
                      AND rid > ?
                    ORDER BY rid
                    LIMIT ?
                    """,
                    (last_rid, eff_batch),
                ).fetchall()

            if not rows:
                break

            min_rid = int(rows[0][0])
            max_rid = int(rows[-1][0])
            last_rid = max_rid

            batch_rid_list = [int(row[0]) for row in rows]
            batch_rids = set(batch_rid_list)
            alt_by_rid: dict[int, list[str]] = {}
            if max_rid - min_rid > len(batch_rids) * 4:
                for chunk_start in range(0, len(batch_rid_list), 400):
                    rid_chunk = batch_rid_list[chunk_start : chunk_start + 400]
                    ph_rids = ",".join("?" for _ in rid_chunk)
                    for lbl_rid, value in con.execute(
                        f"""
                        SELECT rid, value
                        FROM labels
                        WHERE rid IN ({ph_rids})
                        ORDER BY rid, role_rank, value_lower, value
                        """,
                        rid_chunk,
                    ):
                        alt_by_rid.setdefault(int(lbl_rid), []).append(str(value))
            else:
                for lbl_rid, value in con.execute(
                    """
                    SELECT rid, value
                    FROM labels
                    WHERE rid BETWEEN ? AND ?
                    ORDER BY rid, role_rank, value_lower, value
                    """,
                    (min_rid, max_rid),
                ):
                    rid_int = int(lbl_rid)
                    if rid_int not in batch_rids:
                        continue
                    alt_by_rid.setdefault(rid_int, []).append(str(value))

            batch_records: list[PackConceptRecord] = []
            for rid_raw, res_id, sid, rel_id, pref_label, definition in rows:
                rid_int = int(rid_raw)
                pref_str = str(pref_label).strip()
                if not pref_str:
                    continue
                card_text, windows, dedup_alts = _build_concept_texts(
                    pref_str,
                    alt_by_rid.get(rid_int, ()),
                    str(definition) if definition is not None else None,
                    multi_window=multi_window,
                    include_definition=include_definition,
                    max_alt_labels=eff_max_alts,
                    max_definition_chars=eff_max_def,
                )
                primary_text = windows[0] if multi_window else card_text
                batch_records.append(
                    PackConceptRecord(
                        rid=rid_int,
                        resource_id=str(res_id),
                        scheme=scheme_by_sid[int(sid)],
                        release=release_by_id[int(rel_id)],
                        pref_label=pref_str,
                        alt_labels=dedup_alts,
                        definition=str(definition) if definition is not None else None,
                        card_text=card_text,
                        windows=windows,
                        input_sha256=_sha256_text(primary_text),
                    )
                )

            if batch_records:
                yield batch_records
    finally:
        con.close()


def embed_sqlite_pack_to_index(
    pack_path: Path | str,
    encoder: Any,
    *,
    model_id: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 4096,
    schemes: Iterable[str] | None = None,
    multi_window: bool = False,
    include_definition: bool = True,
    max_alt_labels: int = 16,
    max_definition_chars: int = 512,
    encode_mode: str | None = None,
    add_document_prefix: bool = False,
    index_mode: IndexMode = "auto",
    mrl_dim: int | None = None,
    binary_bits: int | None = None,
    agg: WindowAgg = "maxsim",
    hybrid_alpha: float = 0.7,
    rerank_top_r: int = 256,
    use_neon: bool = True,
) -> CompactRetrievalSemanticIndex:
    """Stream a RefSpec SQLite pack, encode concepts via ``encoder``, and return an in-memory index.

    * When ``multi_window=False``: deduplicates identical concept cards by ``input_sha256`` so
      repeated labels across schemes share one compact vector mapped to all matching ``rid``s.
    * When ``multi_window=True``: builds ragged ``doc_offsets`` over
      ``[pref_label, alt_labels..., definition]`` per concept for multi-label ``MaxSim`` search
      while still avoiding duplicate encoder calls for repeated window strings.
    """
    resolved, pack_sha256, pack_id, source_manifest_digest, _ = _read_pack_sqlite_metadata(pack_path)
    eff_model_id = (model_id.strip() if isinstance(model_id, str) and model_id.strip() else None) or _infer_encoder_model_id(encoder)
    if not eff_model_id:
        raise ValueError("Unable to infer `model_id` from encoder; pass `model_id` explicitly")
    eff_dim = _validate_positive_int(dimensions, "dimensions") if dimensions is not None else _infer_encoder_dimensions(encoder)
    is_mrl = bool(getattr(encoder, "is_mrl_trained", False))
    if index_mode not in VALID_INDEX_MODES:
        raise ValueError(f"index_mode must be one of {sorted(VALID_INDEX_MODES)}, got {index_mode!r}")
    if agg not in VALID_WINDOW_AGGS:
        raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")
    _validate_binary_bits_vs_dim(
        mode=index_mode,
        mrl_dim=mrl_dim,
        binary_bits=binary_bits,
        raw_dim=eff_dim,
        is_mrl_trained=is_mrl,
    )

    if not multi_window:
        sha_to_idx: dict[str, int] = {}
        rids_per_vec: list[list[int]] = []
        vec_chunks: list[np.ndarray] = []

        for batch in export_pack_concept_texts(
            resolved,
            batch_size=batch_size,
            schemes=schemes,
            multi_window=False,
            include_definition=include_definition,
            max_alt_labels=max_alt_labels,
            max_definition_chars=max_definition_chars,
        ):
            new_texts: list[str] = []
            for rec in batch:
                existing = sha_to_idx.get(rec.input_sha256)
                if existing is not None:
                    rids_per_vec[existing].append(rec.rid)
                else:
                    idx = len(rids_per_vec)
                    sha_to_idx[rec.input_sha256] = idx
                    rids_per_vec.append([rec.rid])
                    new_texts.append(rec.card_text)

            if new_texts:
                vecs = _encode_text_batch(
                    encoder,
                    new_texts,
                    dimensions=eff_dim,
                    mode=encode_mode,
                    add_document_prefix=add_document_prefix,
                )
                if eff_dim is None:
                    eff_dim = int(vecs.shape[1])
                vec_chunks.append(vecs)

        if not vec_chunks:
            raise ValueError(f"Vocabulary pack {resolved} yielded no labeled concepts to embed")
        all_vecs = vec_chunks[0] if len(vec_chunks) == 1 else np.concatenate(vec_chunks, axis=0)
        return CompactRetrievalSemanticIndex(
            all_vecs,
            rids_per_vec,
            pack_sha256=pack_sha256,
            pack_id=pack_id,
            source_manifest_digest=source_manifest_digest,
            model_id=eff_model_id,
            doc_offsets=None,
            mode=index_mode,
            is_mrl_trained=is_mrl,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            agg=agg,
            hybrid_alpha=hybrid_alpha,
            rerank_top_r=rerank_top_r,
            use_neon=use_neon,
        )

    # Multi-window MaxSim path: preserve contiguous per-concept window spans while caching repeated texts.
    text_sha_to_vec: dict[str, np.ndarray] = {}
    doc_rids: list[int] = []
    doc_offsets_list: list[int] = [0]
    ordered_window_shas: list[str] = []

    for batch in export_pack_concept_texts(
        resolved,
        batch_size=batch_size,
        schemes=schemes,
        multi_window=True,
        include_definition=include_definition,
        max_alt_labels=max_alt_labels,
        max_definition_chars=max_definition_chars,
    ):
        batch_new_shas: list[str] = []
        batch_new_texts: list[str] = []
        batch_seen: set[str] = set()
        for rec in batch:
            doc_rids.append(rec.rid)
            for win_text in rec.windows:
                w_sha = _sha256_text(win_text)
                ordered_window_shas.append(w_sha)
                if w_sha not in text_sha_to_vec and w_sha not in batch_seen:
                    batch_seen.add(w_sha)
                    batch_new_shas.append(w_sha)
                    batch_new_texts.append(win_text)
            doc_offsets_list.append(len(ordered_window_shas))

        if batch_new_texts:
            vecs = _encode_text_batch(
                encoder,
                batch_new_texts,
                dimensions=eff_dim,
                mode=encode_mode,
                add_document_prefix=add_document_prefix,
            )
            if eff_dim is None:
                eff_dim = int(vecs.shape[1])
            for w_sha, row_vec in zip(batch_new_shas, vecs, strict=True):
                text_sha_to_vec[w_sha] = row_vec

    if not ordered_window_shas or eff_dim is None:
        raise ValueError(f"Vocabulary pack {resolved} yielded no labeled concepts to embed")

    all_window_vecs = np.empty((len(ordered_window_shas), eff_dim), dtype=np.float32)
    for i, w_sha in enumerate(ordered_window_shas):
        all_window_vecs[i] = text_sha_to_vec[w_sha]

    return CompactRetrievalSemanticIndex(
        all_window_vecs,
        doc_rids,
        pack_sha256=pack_sha256,
        pack_id=pack_id,
        source_manifest_digest=source_manifest_digest,
        model_id=eff_model_id,
        doc_offsets=np.asarray(doc_offsets_list, dtype=np.int32),
        mode=index_mode,
        is_mrl_trained=is_mrl,
        mrl_dim=mrl_dim,
        binary_bits=binary_bits,
        agg=agg,
        hybrid_alpha=hybrid_alpha,
        rerank_top_r=rerank_top_r,
        use_neon=use_neon,
    )


def embed_sqlite_pack_to_parquet(
    pack_path: Path | str,
    encoder: Any,
    output_dir: Path | str,
    *,
    model_id: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 4096,
    shard_rows: int = 65536,
    schemes: Iterable[str] | None = None,
    multi_window: bool = False,
    include_definition: bool = True,
    max_alt_labels: int = 16,
    max_definition_chars: int = 512,
    encode_mode: str | None = None,
    add_document_prefix: bool = False,
    vector_dtype: Literal["float16", "float32"] = "float16",
    compression: str = "zstd",
    return_index: bool = False,
    index_mode: IndexMode = "auto",
    mrl_dim: int | None = None,
    binary_bits: int | None = None,
    agg: WindowAgg = "maxsim",
    hybrid_alpha: float = 0.7,
    rerank_top_r: int = 256,
    use_neon: bool = True,
) -> PackEmbedSummary:
    """Stream a RefSpec SQLite pack into ``input_sha256``-deduplicated Parquet ready for ``build_semantic_sidecar``.

    Writes:
      * ``<output_dir>/resources.parquet`` (``resource_id``, ``scheme``, ``release``, ``input_sha256``,
        ``rid``, ``pref_label``) — directly consumable by RefSpec's ``build_semantic_sidecar``.
      * ``<output_dir>/vectors/part-000.parquet`` (sharded by ``shard_rows``; unique ``input_sha256`` +
        fixed-size list ``embedding`` column in ``float16`` or ``float32``).
      * ``<output_dir>/windows.parquet`` (only when ``multi_window=True``, mapping each concept's
        ordered windows ``[pref_label, alt_labels..., definition]`` to ``input_sha256``).
      * ``<output_dir>/manifest.json`` summarizing digests, model identity, and deduplication counts.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "embed_sqlite_pack_to_parquet requires `pyarrow` to be installed."
        ) from exc

    if vector_dtype not in ("float16", "float32"):
        raise ValueError(f"vector_dtype must be 'float16' or 'float32', got {vector_dtype!r}")
    eff_shard_rows = _validate_positive_int(shard_rows, "shard_rows")

    t0 = time.perf_counter()
    resolved_pack, pack_sha256, pack_id, source_manifest_digest, _ = _read_pack_sqlite_metadata(pack_path)
    eff_model_id = (model_id.strip() if isinstance(model_id, str) and model_id.strip() else None) or _infer_encoder_model_id(encoder)
    if not eff_model_id:
        raise ValueError("Unable to infer `model_id` from encoder; pass `model_id` explicitly")
    eff_dim = _validate_positive_int(dimensions, "dimensions") if dimensions is not None else _infer_encoder_dimensions(encoder)
    is_mrl = bool(getattr(encoder, "is_mrl_trained", False))
    if return_index:
        if index_mode not in VALID_INDEX_MODES:
            raise ValueError(f"index_mode must be one of {sorted(VALID_INDEX_MODES)}, got {index_mode!r}")
        if agg not in VALID_WINDOW_AGGS:
            raise ValueError(f"agg must be one of {sorted(VALID_WINDOW_AGGS)}, got {agg!r}")
        _validate_binary_bits_vs_dim(
            mode=index_mode,
            mrl_dim=mrl_dim,
            binary_bits=binary_bits,
            raw_dim=eff_dim,
            is_mrl_trained=is_mrl,
        )

    out_root = Path(output_dir).expanduser().resolve()
    vec_dir = out_root / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    res_schema = pa.schema(
        [
            pa.field("resource_id", pa.string()),
            pa.field("scheme", pa.string()),
            pa.field("release", pa.string()),
            pa.field("input_sha256", pa.string()),
            pa.field("rid", pa.int64()),
            pa.field("pref_label", pa.string()),
        ]
    )
    win_schema = pa.schema(
        [
            pa.field("rid", pa.int64()),
            pa.field("resource_id", pa.string()),
            pa.field("scheme", pa.string()),
            pa.field("release", pa.string()),
            pa.field("window_index", pa.int32()),
            pa.field("input_sha256", pa.string()),
        ]
    )

    pa_elem_type = pa.float16() if vector_dtype == "float16" else pa.float32()
    np_out_dtype = np.float16 if vector_dtype == "float16" else np.float32

    res_final = out_root / "resources.parquet"
    res_tmp: Path | None = _mk_tmp_path(out_root, ".resources.parquet.")
    win_final: Path | None = (out_root / "windows.parquet") if multi_window else None
    win_tmp: Path | None = _mk_tmp_path(out_root, ".windows.parquet.") if multi_window else None

    res_writer: pq.ParquetWriter | None = None
    win_writer: pq.ParquetWriter | None = None
    vec_writer: pq.ParquetWriter | None = None
    vec_schema: pa.Schema | None = None
    vec_tmp: Path | None = None
    staged_vec_files: list[tuple[Path, Path]] = []
    vec_final_files: list[Path] = []
    shard_idx = 0
    rows_in_shard = 0

    seen_digests: set[bytes] = set()
    resource_count = 0
    window_count = 0
    unique_vector_count = 0
    deduplicated_hits = 0

    # Optional in-memory accumulator only populated when return_index=True.
    idx_sha_to_vec_idx: dict[str, int] = {}
    idx_rids_per_vec: list[list[int]] = []
    idx_vec_chunks: list[np.ndarray] = []
    idx_sha_to_vec: dict[str, np.ndarray] = {}
    idx_doc_rids: list[int] = []
    idx_doc_offsets: list[int] = [0]
    idx_ordered_shas: list[str] = []

    def _close_current_vec_shard() -> None:
        nonlocal vec_writer, vec_tmp, shard_idx, rows_in_shard
        if vec_writer is not None:
            vec_writer.close()
            vec_writer = None
            assert vec_tmp is not None
            final_part = vec_dir / f"part-{shard_idx:03d}.parquet"
            staged_vec_files.append((vec_tmp, final_part))
            vec_tmp = None
            shard_idx += 1
            rows_in_shard = 0

    def _write_vec_rows(shas: list[str], vecs_f32: np.ndarray) -> None:
        nonlocal vec_writer, vec_schema, vec_tmp, rows_in_shard
        dim = int(vecs_f32.shape[1])
        if vec_schema is None:
            vec_schema = pa.schema(
                [
                    pa.field("input_sha256", pa.string()),
                    pa.field("embedding", pa.list_(pa_elem_type, dim)),
                ]
            )
        cursor = 0
        total = len(shas)
        while cursor < total:
            if vec_writer is None:
                vec_tmp = _mk_tmp_path(
                    vec_dir,
                    f".part-{shard_idx:03d}.{os.getpid()}.{threading.get_ident()}.",
                    suffix=".tmp.parquet",
                )
                vec_writer = pq.ParquetWriter(vec_tmp, vec_schema, compression=compression)
                rows_in_shard = 0
            room = eff_shard_rows - rows_in_shard
            take = min(room, total - cursor)
            sub_shas = shas[cursor : cursor + take]
            sub_vecs = np.ascontiguousarray(vecs_f32[cursor : cursor + take], dtype=np_out_dtype)
            flat_arr = pa.array(sub_vecs.reshape(-1), type=pa_elem_type)
            emb_arr = pa.FixedSizeListArray.from_arrays(flat_arr, dim)
            tbl = pa.table(
                {
                    "input_sha256": pa.array(sub_shas, type=pa.string()),
                    "embedding": emb_arr,
                },
                schema=vec_schema,
            )
            vec_writer.write_table(tbl)
            rows_in_shard += take
            cursor += take
            if rows_in_shard >= eff_shard_rows:
                _close_current_vec_shard()

    semantic_index: CompactRetrievalSemanticIndex | None = None
    try:
        assert res_tmp is not None
        res_writer = pq.ParquetWriter(res_tmp, res_schema, compression=compression)
        if multi_window and win_tmp is not None:
            win_writer = pq.ParquetWriter(win_tmp, win_schema, compression=compression)

        for batch in export_pack_concept_texts(
            resolved_pack,
            batch_size=batch_size,
            schemes=schemes,
            multi_window=multi_window,
            include_definition=include_definition,
            max_alt_labels=max_alt_labels,
            max_definition_chars=max_definition_chars,
        ):
            res_ids: list[str] = []
            res_schemes: list[str] = []
            res_releases: list[str] = []
            res_shas: list[str] = []
            res_rids: list[int] = []
            res_prefs: list[str] = []

            win_rids: list[int] = []
            win_res_ids: list[str] = []
            win_schemes: list[str] = []
            win_releases: list[str] = []
            win_indices: list[int] = []
            win_shas: list[str] = []

            new_shas: list[str] = []
            new_texts: list[str] = []

            for rec in batch:
                resource_count += 1
                res_ids.append(rec.resource_id)
                res_schemes.append(rec.scheme)
                res_releases.append(rec.release)
                res_shas.append(rec.input_sha256)
                res_rids.append(rec.rid)
                res_prefs.append(rec.pref_label)

                if not multi_window:
                    window_count += 1
                    digest_bytes = bytes.fromhex(rec.input_sha256)
                    if digest_bytes in seen_digests:
                        deduplicated_hits += 1
                        if return_index:
                            idx_rids_per_vec[idx_sha_to_vec_idx[rec.input_sha256]].append(rec.rid)
                    else:
                        seen_digests.add(digest_bytes)
                        new_shas.append(rec.input_sha256)
                        new_texts.append(rec.card_text)
                        if return_index:
                            idx_sha_to_vec_idx[rec.input_sha256] = len(idx_rids_per_vec)
                            idx_rids_per_vec.append([rec.rid])
                else:
                    if return_index:
                        idx_doc_rids.append(rec.rid)
                    for w_idx, w_text in enumerate(rec.windows):
                        window_count += 1
                        w_sha = rec.input_sha256 if w_idx == 0 else _sha256_text(w_text)
                        win_rids.append(rec.rid)
                        win_res_ids.append(rec.resource_id)
                        win_schemes.append(rec.scheme)
                        win_releases.append(rec.release)
                        win_indices.append(w_idx)
                        win_shas.append(w_sha)
                        if return_index:
                            idx_ordered_shas.append(w_sha)
                        digest_bytes = bytes.fromhex(w_sha)
                        if digest_bytes in seen_digests:
                            deduplicated_hits += 1
                        else:
                            seen_digests.add(digest_bytes)
                            new_shas.append(w_sha)
                            new_texts.append(w_text)
                    if return_index:
                        idx_doc_offsets.append(len(idx_ordered_shas))

            res_writer.write_table(
                pa.table(
                    {
                        "resource_id": pa.array(res_ids, type=pa.string()),
                        "scheme": pa.array(res_schemes, type=pa.string()),
                        "release": pa.array(res_releases, type=pa.string()),
                        "input_sha256": pa.array(res_shas, type=pa.string()),
                        "rid": pa.array(res_rids, type=pa.int64()),
                        "pref_label": pa.array(res_prefs, type=pa.string()),
                    },
                    schema=res_schema,
                )
            )
            if win_writer is not None and win_rids:
                win_writer.write_table(
                    pa.table(
                        {
                            "rid": pa.array(win_rids, type=pa.int64()),
                            "resource_id": pa.array(win_res_ids, type=pa.string()),
                            "scheme": pa.array(win_schemes, type=pa.string()),
                            "release": pa.array(win_releases, type=pa.string()),
                            "window_index": pa.array(win_indices, type=pa.int32()),
                            "input_sha256": pa.array(win_shas, type=pa.string()),
                        },
                        schema=win_schema,
                    )
                )

            if new_texts:
                vecs_f32 = _encode_text_batch(
                    encoder,
                    new_texts,
                    dimensions=eff_dim,
                    mode=encode_mode,
                    add_document_prefix=add_document_prefix,
                )
                if eff_dim is None:
                    eff_dim = int(vecs_f32.shape[1])
                unique_vector_count += len(new_shas)
                _write_vec_rows(new_shas, vecs_f32)
                if return_index:
                    if not multi_window:
                        idx_vec_chunks.append(vecs_f32)
                    else:
                        for w_sha, row_vec in zip(new_shas, vecs_f32, strict=True):
                            idx_sha_to_vec[w_sha] = row_vec

        if resource_count == 0 or eff_dim is None:
            raise ValueError(f"Vocabulary pack {resolved_pack} yielded no labeled concepts to embed")

        res_writer.close()
        res_writer = None

        if win_writer is not None:
            win_writer.close()
            win_writer = None

        _close_current_vec_shard()

        if return_index:
            if not multi_window:
                all_vecs = (
                    idx_vec_chunks[0]
                    if len(idx_vec_chunks) == 1
                    else np.concatenate(idx_vec_chunks, axis=0)
                )
                semantic_index = CompactRetrievalSemanticIndex(
                    all_vecs,
                    idx_rids_per_vec,
                    pack_sha256=pack_sha256,
                    pack_id=pack_id,
                    source_manifest_digest=source_manifest_digest,
                    model_id=eff_model_id,
                    doc_offsets=None,
                    mode=index_mode,
                    is_mrl_trained=is_mrl,
                    mrl_dim=mrl_dim,
                    binary_bits=binary_bits,
                    agg=agg,
                    hybrid_alpha=hybrid_alpha,
                    rerank_top_r=rerank_top_r,
                    use_neon=use_neon,
                )
            else:
                all_win_vecs = np.empty((len(idx_ordered_shas), eff_dim), dtype=np.float32)
                for i, w_sha in enumerate(idx_ordered_shas):
                    all_win_vecs[i] = idx_sha_to_vec[w_sha]
                semantic_index = CompactRetrievalSemanticIndex(
                    all_win_vecs,
                    idx_doc_rids,
                    pack_sha256=pack_sha256,
                    pack_id=pack_id,
                    source_manifest_digest=source_manifest_digest,
                    model_id=eff_model_id,
                    doc_offsets=np.asarray(idx_doc_offsets, dtype=np.int32),
                    mode=index_mode,
                    is_mrl_trained=is_mrl,
                    mrl_dim=mrl_dim,
                    binary_bits=binary_bits,
                    agg=agg,
                    hybrid_alpha=hybrid_alpha,
                    rerank_top_r=rerank_top_r,
                    use_neon=use_neon,
                )

        os.replace(res_tmp, res_final)
        res_tmp = None

        if win_tmp is not None and win_final is not None:
            os.replace(win_tmp, win_final)
            win_tmp = None

        promoted_vec_paths: set[Path] = set()
        for tmp_part, final_part in staged_vec_files:
            os.replace(tmp_part, final_part)
            vec_final_files.append(final_part)
            promoted_vec_paths.add(final_part)

        promoted_vec_names = {p.name for p in promoted_vec_paths}
        for existing_part in vec_dir.glob("part-*.parquet"):
            if existing_part.name not in promoted_vec_names:
                existing_part.unlink(missing_ok=True)
    except BaseException:
        for writer in (res_writer, win_writer, vec_writer):
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
        for tmp_p in (res_tmp, win_tmp, vec_tmp):
            if tmp_p is not None:
                tmp_p.unlink(missing_ok=True)
        for staged_tmp, _ in staged_vec_files:
            staged_tmp.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - t0
    manifest_path = out_root / "manifest.json"
    manifest_payload = {
        "packSha256": pack_sha256,
        "packId": pack_id,
        "sourceManifestDigest": source_manifest_digest,
        "modelId": eff_model_id,
        "dimensions": int(eff_dim),
        "vectorDtype": vector_dtype,
        "multiWindow": bool(multi_window),
        "resourceCount": resource_count,
        "windowCount": window_count,
        "uniqueVectorCount": unique_vector_count,
        "deduplicatedHits": deduplicated_hits,
        "resourcesParquet": res_final.name,
        "windowsParquet": win_final.name if win_final is not None else None,
        "vectorShards": [f"vectors/{p.name}" for p in vec_final_files],
    }
    manifest_tmp = _mk_tmp_path(out_root, ".manifest.json.")
    try:
        manifest_tmp.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
    except BaseException:
        manifest_tmp.unlink(missing_ok=True)
        raise

    return PackEmbedSummary(
        output_dir=out_root,
        resources_parquet=res_final,
        windows_parquet=win_final,
        vector_files=tuple(vec_final_files),
        manifest_path=manifest_path,
        pack_sha256=pack_sha256,
        pack_id=pack_id,
        source_manifest_digest=source_manifest_digest,
        model_id=eff_model_id,
        dimensions=int(eff_dim),
        multi_window=bool(multi_window),
        resource_count=resource_count,
        window_count=window_count,
        unique_vector_count=unique_vector_count,
        deduplicated_hits=deduplicated_hits,
        elapsed_seconds=elapsed,
        semantic_index=semantic_index,
    )


def _mk_tmp_path(directory: Path, prefix: str, suffix: str = ".tmp") -> Path:
    fd, raw_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(directory))
    os.close(fd)
    return Path(raw_name)


__all__ = [
    "CompactRetrievalSemanticIndex",
    "FastSiliconQueryEncoder",
    "LayaRerankFusion",
    "PackConceptRecord",
    "PackEmbedSummary",
    "build_rerank_noul_qdef",
    "embed_sqlite_pack_to_index",
    "embed_sqlite_pack_to_parquet",
    "export_pack_concept_texts",
    "format_candidate_concept_summary",
]
