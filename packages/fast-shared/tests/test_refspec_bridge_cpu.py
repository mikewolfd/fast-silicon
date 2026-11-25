"""CPU unit tests for Track C Wave 3: Apple Silicon ``fast-silicon`` DI bridge for RefSpec."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import pytest

import fast_shared
import fast_shared.refspec_bridge as rb
from fast_shared.refspec_bridge import (
    CompactRetrievalSemanticIndex,
    FastSiliconQueryEncoder,
    LayaRerankFusion,
    PackConceptRecord,
    PackEmbedSummary,
    build_rerank_noul_qdef,
    embed_sqlite_pack_to_index,
    embed_sqlite_pack_to_parquet,
    export_pack_concept_texts,
    format_candidate_concept_summary,
)
from fast_shared.retrieval_index import CompactRetrievalIndex, pack_binary_bits


def _unit_vec(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _deterministic_text_vec(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return _unit_vec(seed, dim)


class _FakeFastBgeSmall:
    """Test double matching ``FastBgeSmall.encode_texts`` signature and attributes."""

    dim: int = 384
    max_dim: int = 384
    is_mrl_trained: bool = False
    model_name: str = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed: bool = False

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        mode: str | None = None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": texts,
                "is_query": is_query,
                "add_query_prefix": add_query_prefix,
                "mode": mode,
            }
        )
        if isinstance(texts, str):
            # Return intentionally unnormalized float32 vector to verify bridge normalization.
            return _deterministic_text_vec(texts, self.dim) * np.float32(3.5)
        items = list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([_deterministic_text_vec(t, self.dim) for t in items], axis=0)

    def close(self) -> None:
        self.closed = True


class _FakeFastEmbeddingGemma2:
    """Test double matching ``FastEmbeddingGemma2.encode_texts`` with MRL dimensions."""

    dim: int = 768
    max_dim: int = 768
    supports_dimensions: bool = True
    is_mrl_trained: bool = True
    model_name: str = "google/embeddinggemma-2"

    def __init__(self, *, honor_dimensions: bool = True) -> None:
        self.honor_dimensions = honor_dimensions
        self.calls: list[dict[str, Any]] = []
        self.closed: bool = False

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        prefix: str | None = None,
        prompt_name: str | None = None,
        mode: str | None = None,
        max_length: int | None = None,
        dimensions: int | None = None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": texts,
                "is_query": is_query,
                "add_query_prefix": add_query_prefix,
                "add_document_prefix": add_document_prefix,
                "prefix": prefix,
                "prompt_name": prompt_name,
                "mode": mode,
                "max_length": max_length,
                "dimensions": dimensions,
            }
        )
        out_d = int(dimensions) if (self.honor_dimensions and dimensions is not None) else self.dim
        if isinstance(texts, str):
            # Unnormalized float64 to verify MRL truncation + float32 unit normalization.
            raw = _deterministic_text_vec(texts, self.dim).astype(np.float64) * 5.0
            return raw[:out_d]
        items = list(texts)
        if not items:
            return np.empty((0, out_d), dtype=np.float32)
        full = np.stack([_deterministic_text_vec(t, self.dim) for t in items], axis=0)
        return full[:, :out_d]

    def close(self) -> None:
        self.closed = True


class _FakeLayaBackend:
    """Test double mimicking ``LayaAneBackend`` / ``LayaHeteroEngine`` batched ``noul`` API."""

    def __init__(self, score_fn: Any) -> None:
        self._score_fn = score_fn
        self.state_calls: list[Any] = []
        self.batch_encode_calls: list[tuple[Any, list[dict[str, Any]]]] = []
        self.score_batch_calls: list[list[dict[str, Any]]] = []
        self.closed: bool = False

    def encode_state(self, state: Any) -> list[int]:
        self.state_calls.append(state)
        return [101, len(str(state)), 102]

    def encode_items_for_state(
        self,
        state: Any,
        qdefs: Sequence[Mapping[str, Any]],
        *,
        max_len: int | None = None,
        on_truncate: str = "warn",
    ) -> list[dict[str, Any]]:
        _ = (max_len, on_truncate)
        st_ids = self.encode_state(state)
        qd_list = [dict(q) for q in qdefs]
        self.batch_encode_calls.append((state, qd_list))
        encoded: list[dict[str, Any]] = []
        for qd in qd_list:
            assert qd["type"] == "noul"
            assert isinstance(qd["instructions"], str) and qd["instructions"]
            assert isinstance(qd["criteria"], Mapping)
            assert "false" in qd["criteria"] and "true" in qd["criteria"]
            encoded.append(
                {
                    "ids": st_ids + [len(qd["instructions"])],
                    "markers": [1, 2],
                    "qtype": 2,
                    "k": 2,
                    "state": state,
                    "qdef": qd,
                }
            )
        return encoded

    def score_noul_batch(self, encoded_items: Sequence[Mapping[str, Any]]) -> list[float]:
        items = [dict(it) for it in encoded_items]
        self.score_batch_calls.append(items)
        return [float(self._score_fn(it["state"], it["qdef"])) for it in items]

    def close(self) -> None:
        self.closed = True


class _ChoiceOnlyFakeLayaBackend:
    """Minimal fake implementing only ``encode_item`` + ``score_choice_batch``."""

    def __init__(self, score_fn: Any) -> None:
        self._score_fn = score_fn
        self.calls = 0

    def encode_item(self, state: Any, qdef: Mapping[str, Any]) -> dict[str, Any]:
        return {"state": state, "qdef": dict(qdef)}

    def score_choice_batch(self, encoded_items: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for it in encoded_items:
            p_true = float(self._score_fn(it["state"], it["qdef"]))
            out.append([1.0 - p_true, p_true])
        return out


def _create_sqlite_pack_fixture(pack_path: Path) -> dict[str, Any]:
    """Create a realistic compiled RefSpec SQLite vocabulary pack fixture."""
    source_digest = "sha256:" + "a" * 64
    pack_id = "urn:ref:vocabulary-pack:" + "b" * 64
    schemes = [
        "urn:ref:atlas-resource-scheme:eurovoc",
        "urn:ref:atlas-resource-scheme:mesh-descriptors",
        "urn:ref:atlas-resource-scheme:federal-register-thesaurus-2025",
    ]
    releases = [
        "urn:ref:atlas-release:eurovoc-2025",
        "urn:ref:atlas-release:mesh-2025",
        "urn:ref:atlas-release:fr-2025",
    ]

    con = sqlite3.connect(str(pack_path))
    try:
        con.executescript(
            """
            CREATE TABLE pack_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE schemes (
                sid INTEGER PRIMARY KEY,
                uri TEXT NOT NULL UNIQUE
            );

            CREATE TABLE releases (
                rel_id INTEGER PRIMARY KEY,
                uri TEXT NOT NULL UNIQUE
            );

            CREATE TABLE resources (
                rid INTEGER PRIMARY KEY,
                id TEXT NOT NULL,
                sid INTEGER NOT NULL,
                rel_id INTEGER NOT NULL,
                definition TEXT,
                src_pfx INTEGER NOT NULL,
                src_suffix TEXT NOT NULL,
                notations_json TEXT,
                pref_label TEXT,
                label_count INTEGER NOT NULL,
                UNIQUE(id, sid, rel_id)
            );

            CREATE TABLE labels (
                rid INTEGER NOT NULL,
                role_rank INTEGER NOT NULL,
                value_lower TEXT NOT NULL,
                value TEXT NOT NULL,
                lbl_pfx INTEGER NOT NULL,
                lbl_suffix TEXT NOT NULL,
                role TEXT NOT NULL,
                language TEXT NOT NULL,
                PRIMARY KEY (rid, role_rank, value_lower, value, lbl_pfx, lbl_suffix)
            ) WITHOUT ROWID;
            """
        )

        con.executemany(
            "INSERT INTO pack_meta (key, value) VALUES (?, ?)",
            [
                ("pack_id", pack_id),
                ("source_manifest_digest", source_digest),
                ("schemes_json", json.dumps(schemes, separators=(",", ":"))),
                ("schema_version", "2"),
            ],
        )
        con.executemany(
            "INSERT INTO schemes (sid, uri) VALUES (?, ?)",
            [(i + 1, uri) for i, uri in enumerate(schemes)],
        )
        con.executemany(
            "INSERT INTO releases (rel_id, uri) VALUES (?, ?)",
            [(i + 1, uri) for i, uri in enumerate(releases)],
        )

        # Include:
        #   - rid=1: multi-synonym concept with definition
        #   - rid=2 & rid=3: identical label + no definition across 2 schemes -> input_sha256 dedup!
        #   - rid=4: unlabeled resource (pref_label IS NULL) -> MUST be skipped by streaming ETL!
        #   - rid=5: concept with multiple alt labels + definition
        #   - rid=6: third scheme concept
        resources_rows = [
            (
                1,
                "urn:ref:concept:mesh:D009203",
                2,
                2,
                "Necrosis of the myocardium caused by obstruction of coronary blood supply.",
                0,
                "src-1",
                None,
                "Myocardial Infarction",
                3,
            ),
            (
                2,
                "urn:ref:concept:eurovoc:water",
                1,
                1,
                None,
                0,
                "src-2",
                None,
                "Clean Water",
                1,
            ),
            (
                3,
                "urn:ref:concept:fr:water",
                3,
                3,
                None,
                0,
                "src-3",
                None,
                "Clean Water",
                1,
            ),
            (
                4,
                "urn:ref:concept:eurovoc:unlabeled",
                1,
                1,
                "Internal anchor without label",
                0,
                "src-4",
                None,
                None,
                0,
            ),
            (
                5,
                "urn:ref:concept:eurovoc:solar",
                1,
                1,
                "Conversion of sunlight into electricity using semiconductor cells.",
                0,
                "src-5",
                None,
                "Photovoltaic Power",
                3,
            ),
            (
                6,
                "urn:ref:concept:fr:veterans",
                3,
                3,
                "Federal health benefits provided to military veterans.",
                0,
                "src-6",
                None,
                "Veterans Health Care",
                2,
            ),
        ]
        con.executemany(
            """
            INSERT INTO resources (rid, id, sid, rel_id, definition, src_pfx, src_suffix, notations_json, pref_label, label_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            resources_rows,
        )

        label_rows = [
            (1, 0, "myocardial infarction", "Myocardial Infarction", 0, "l1", "prefLabel", "en"),
            (1, 1, "heart attack", "Heart Attack", 0, "l2", "altLabel", "en"),
            (1, 1, "cardiac arrest", "Cardiac Arrest", 0, "l3", "altLabel", "en"),
            (2, 0, "clean water", "Clean Water", 0, "l4", "prefLabel", "en"),
            (3, 0, "clean water", "Clean Water", 0, "l5", "prefLabel", "en"),
            (5, 0, "photovoltaic power", "Photovoltaic Power", 0, "l6", "prefLabel", "en"),
            (5, 1, "solar electricity", "Solar Electricity", 0, "l7", "altLabel", "en"),
            (5, 1, "pv energy", "PV Energy", 0, "l8", "altLabel", "en"),
            (6, 0, "veterans health care", "Veterans Health Care", 0, "l9", "prefLabel", "en"),
            (6, 1, "va medical benefits", "VA Medical Benefits", 0, "l10", "altLabel", "en"),
        ]
        con.executemany(
            """
            INSERT INTO labels (rid, role_rank, value_lower, value, lbl_pfx, lbl_suffix, role, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            label_rows,
        )
        con.commit()
    finally:
        con.close()

    digest = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    return {
        "pack_id": pack_id,
        "source_manifest_digest": source_digest,
        "pack_sha256": f"sha256:{digest}",
        "schemes": schemes,
    }


# ---------------------------------------------------------------------------
# Protocol & Packaging Hygiene Tests
# ---------------------------------------------------------------------------


def test_no_runtime_refspec_import_and_exports_match() -> None:
    src = Path(rb.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("refspec"), f"Forbidden refspec import: {alias.name}"
                assert not alias.name.startswith("fast_decision_models"), (
                    f"Forbidden fast_decision_models import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("refspec"), f"Forbidden refspec import: {mod}"
            assert not mod.startswith("fast_decision_models"), (
                f"Forbidden fast_decision_models import: {mod}"
            )

    for sym in rb.__all__:
        assert hasattr(fast_shared, sym), f"Missing {sym} from fast_shared"
        assert sym in fast_shared.__all__, f"Missing {sym} from fast_shared.__all__"


def test_structural_protocol_compatibility_with_refspec() -> None:
    bge = _FakeFastBgeSmall()
    qenc = FastSiliconQueryEncoder(bge)
    sidx = CompactRetrievalSemanticIndex(
        np.stack([_unit_vec(1, 32), _unit_vec(2, 32)], axis=0),
        [10, 20],
        pack_sha256="sha256:" + "1" * 64,
        pack_id="urn:ref:vocabulary-pack:" + "2" * 64,
        source_manifest_digest="sha256:" + "3" * 64,
        model_id="BAAI/bge-small-en-v1.5",
        mode="int8",
    )
    reranker = LayaRerankFusion(_FakeLayaBackend(lambda _s, _q: 0.5))

    # Direct signature checks (always run even if RefSpec is not installed).
    sig_search = inspect.signature(sidx.search_rids)
    assert list(sig_search.parameters) == ["query_vector", "top_k", "min_cosine", "nprobe"]
    assert sig_search.parameters["top_k"].default == 40
    assert sig_search.parameters["min_cosine"].default == 0.55
    assert sig_search.parameters["nprobe"].default == 24

    sig_rerank = inspect.signature(reranker.rerank)
    assert list(sig_rerank.parameters) == ["query", "candidates", "limit"]
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in sig_rerank.parameters.values())

    sig_enc = inspect.signature(qenc.encode_query)
    assert list(sig_enc.parameters) == ["text"]


def test_runtime_checkable_refspec_protocols(tmp_path: Path) -> None:
    proto_mod: Any = None
    mod_name = "_refspec_protocols_check"
    cleanup_mod = False
    try:
        import refspec.atlas.vocabulary_protocols as imported_protocols

        proto_mod = imported_protocols
    except ImportError:
        env_path = os.environ.get("REFSPEC_PROTOCOLS_PATH", "").strip()
        if not env_path or not Path(env_path).is_file():
            pytest.skip(
                "refspec.atlas.vocabulary_protocols is not importable and "
                "REFSPEC_PROTOCOLS_PATH is not set to an existing file"
            )
        spec = importlib.util.spec_from_file_location(mod_name, Path(env_path))
        assert spec is not None and spec.loader is not None
        proto_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = proto_mod
        cleanup_mod = True
        spec.loader.exec_module(proto_mod)

    try:
        qenc = FastSiliconQueryEncoder(_FakeFastBgeSmall())
        vecs = np.stack([_unit_vec(1, 32), _unit_vec(2, 32)], axis=0)
        sidx = CompactRetrievalSemanticIndex(
            vecs,
            [10, 20],
            pack_sha256="sha256:" + "1" * 64,
            pack_id="urn:ref:vocabulary-pack:" + "2" * 64,
            source_manifest_digest="sha256:" + "3" * 64,
            model_id="BAAI/bge-small-en-v1.5",
            mode="int8",
        )
        vpack_file = tmp_path / "proto_check.vpack"
        _write_synthetic_vpack(vpack_file, vecs, [[10], [20]], mode="binary_i8")
        reranker = LayaRerankFusion(
            _FakeLayaBackend(lambda _s, _q: 0.5),
            binary_profile=(1.0, 0.0),
            escalation_backend=_FakeKevBackend(lambda _s, _q: 0.8),
            attach_confidence=True,
        )
        with CompactRetrievalSemanticIndex.from_vpack(vpack_file) as vpack_sidx:
            assert isinstance(qenc, proto_mod.QueryEncoder)
            assert isinstance(sidx, proto_mod.SemanticIndex)
            assert isinstance(vpack_sidx, proto_mod.SemanticIndex)
            assert isinstance(reranker, proto_mod.CandidateReranker)
    finally:
        if cleanup_mod:
            sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# T3.1: FastSiliconQueryEncoder Tests
# ---------------------------------------------------------------------------


def test_query_encoder_fake_bge_small() -> None:
    bge = _FakeFastBgeSmall()
    enc = FastSiliconQueryEncoder(bge, mode="ane")
    assert enc.model_id == "BAAI/bge-small-en-v1.5"
    assert enc.dimensions == 384
    assert enc.is_initialized is True

    vec = enc.encode_query("clean water act")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (384,)
    assert np.isclose(float(np.linalg.norm(vec)), 1.0, atol=1e-5)

    assert len(bge.calls) == 1
    assert bge.calls[0] == {
        "texts": "clean water act",
        "is_query": True,
        "add_query_prefix": True,
        "mode": "ane",
    }

    # Closing wrapper must NOT close externally injected encoder.
    enc.close()
    assert bge.closed is False


def test_query_encoder_fake_embeddinggemma2_mrl_truncation_and_norm() -> None:
    # 1. Encoder honors dimensions=512 directly.
    gemma = _FakeFastEmbeddingGemma2(honor_dimensions=True)
    enc = FastSiliconQueryEncoder(gemma, dimensions=512, mode="hetero")
    assert enc.model_id == "google/embeddinggemma-2"
    assert enc.dimensions == 512

    v512 = enc.encode_query("solar energy incentives")
    assert v512.dtype == np.float32
    assert v512.shape == (512,)
    assert np.isclose(float(np.linalg.norm(v512)), 1.0, atol=1e-5)
    assert gemma.calls[0]["dimensions"] == 512
    assert gemma.calls[0]["is_query"] is True
    assert gemma.calls[0]["add_query_prefix"] is True
    assert gemma.calls[0]["mode"] == "hetero"

    # 2. Even if an encoder returns full 768d unnormalized float64, bridge truncates & re-normalizes.
    gemma_raw = _FakeFastEmbeddingGemma2(honor_dimensions=False)
    enc_trunc = FastSiliconQueryEncoder(gemma_raw, dimensions=256)
    v256 = enc_trunc.encode_query("solar energy incentives")
    assert v256.dtype == np.float32
    assert v256.shape == (256,)
    assert np.isclose(float(np.linalg.norm(v256)), 1.0, atol=1e-5)
    expected = _deterministic_text_vec("solar energy incentives", 768)[:256]
    expected = expected / np.linalg.norm(expected)
    assert float(np.dot(v256, expected)) > 0.99999


def test_query_encoder_lazy_factory_and_validation() -> None:
    created: list[_FakeFastBgeSmall] = []

    def _factory() -> _FakeFastBgeSmall:
        inst = _FakeFastBgeSmall()
        created.append(inst)
        return inst

    with FastSiliconQueryEncoder(
        encoder_factory=_factory,
        model_id="BAAI/bge-small-en-v1.5",
        dimensions=384,
    ) as enc:
        # Reading metadata must NOT eagerly construct the encoder!
        assert enc.model_id == "BAAI/bge-small-en-v1.5"
        assert enc.dimensions == 384
        assert enc.is_initialized is False
        assert len(created) == 0

        v1 = enc.encode_query("first query")
        v2 = enc.encode_query("second query")
        assert enc.is_initialized is True
        assert len(created) == 1
        assert v1.shape == (384,) and v2.shape == (384,)

    assert created[0].closed is True

    with pytest.raises(ValueError, match="exactly one"):
        FastSiliconQueryEncoder()
    with pytest.raises(ValueError, match="exceeds encoder max_dim"):
        FastSiliconQueryEncoder(_FakeFastBgeSmall(), dimensions=512)
    with pytest.raises(TypeError, match="query text must be a str"):
        FastSiliconQueryEncoder(_FakeFastBgeSmall()).encode_query(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T3.2: CompactRetrievalSemanticIndex Tests
# ---------------------------------------------------------------------------


def test_compact_semantic_index_binary_int8_search_vector_for_rid_and_abstention() -> None:
    dim = 128
    n_docs = 256
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((n_docs, dim)).astype(np.float32)
    vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    # Doc 0 maps to two deduplicated rids [1000, 2000]; remaining docs map 1:1.
    rids_per_doc: list[int | list[int]] = [[1000, 2000]] + [3000 + i for i in range(1, n_docs)]

    sidx = CompactRetrievalSemanticIndex(
        vecs,
        rids_per_doc,
        pack_sha256="sha256:" + "a" * 64,
        pack_id="urn:ref:vocabulary-pack:" + "b" * 64,
        source_manifest_digest="sha256:" + "c" * 64,
        model_id="test/embed-128",
        mode="binary_int8",
    )
    assert sidx.mode == "binary_int8"
    assert sidx.dimensions == dim
    assert sidx.num_docs == n_docs
    assert sidx.is_multi_window is False

    # Query near doc 0 -> must return both deduplicated rids 1000 & 2000 at top with ~1.0 cosine.
    hits = sidx.search_rids(vecs[0], top_k=5, min_cosine=0.55, nprobe=24)
    assert len(hits) >= 2
    assert {hits[0][0], hits[1][0]} == {1000, 2000}
    assert hits[0][1] > 0.99
    assert hits[1][1] > 0.99

    # Abstention: inverted query vector (-vecs[0]) has cosine ~ -1.0 < 0.85 -> empty.
    abs_hits = sidx.search_rids(-vecs[0], top_k=5, min_cosine=0.85)
    assert abs_hits == []

    # Zero query vector -> clean abstention.
    assert sidx.search_rids(np.zeros(dim, dtype=np.float32), top_k=5, min_cosine=0.0) == []

    # vector_for_rid O(1) reconstruction accuracy (>0.99 cosine vs original fp32).
    for rid, expected_idx in ((1000, 0), (2000, 0), (3042, 42)):
        reconstructed = sidx.vector_for_rid(rid)
        assert reconstructed is not None
        assert reconstructed.dtype == np.float32
        assert reconstructed.shape == (dim,)
        assert np.isclose(float(np.linalg.norm(reconstructed)), 1.0, atol=1e-5)
        cos = float(np.dot(reconstructed, vecs[expected_idx]))
        assert cos > 0.995, f"Expected cosine > 0.995 for rid={rid}, got {cos}"

    assert sidx.vector_for_rid(999999) is None

    # Dimension mismatch raises ValueError.
    with pytest.raises(ValueError, match="query_vector must be 1D"):
        sidx.search_rids(np.ones(dim // 2, dtype=np.float32))


def test_compact_semantic_index_multilabel_maxsim_beats_averaged_vector() -> None:
    """Prove multi-label MaxSim window scoring beats averaged concept vector on orthogonal synonyms."""
    dim = 128
    # Build 4 mutually orthogonal unit basis vectors via QR decomposition.
    rng = np.random.default_rng(1234)
    q_mat, _ = np.linalg.qr(rng.standard_normal((dim, 5)).astype(np.float32))
    v_pref = q_mat[:, 0]
    v_syn1 = q_mat[:, 1]
    v_syn2 = q_mat[:, 2]  # Query targets this alternate label!
    v_other = q_mat[:, 3]

    # Distractor concept aligned 0.72 with query synonym + 0.69 orthogonal noise.
    v_distractor = 0.72 * v_syn2 + math.sqrt(1.0 - 0.72**2) * v_other
    v_distractor = (v_distractor / np.linalg.norm(v_distractor)).astype(np.float32)

    # 1) Averaged single-vector concept representation: mean([v_pref, v_syn1, v_syn2]) has cos ~ 0.577 < 0.72!
    v_avg = v_pref + v_syn1 + v_syn2
    v_avg = (v_avg / np.linalg.norm(v_avg)).astype(np.float32)
    avg_index = CompactRetrievalSemanticIndex(
        np.stack([v_avg, v_distractor], axis=0),
        [10, 20],
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="test/embed",
        mode="int8",
    )
    avg_hits = avg_index.search_rids(v_syn2, top_k=2, min_cosine=0.1)
    assert avg_hits[0][0] == 20  # Distractor beats averaged concept!
    assert avg_hits[1][0] == 10
    assert avg_hits[1][1] < 0.62

    # 2) Multi-label MaxSim index over [pref_label, syn1, syn2] windows for rid=10:
    #    both int8 AND binary_int8 modes MUST rank target rid=10 #1 with score > 0.99!
    win_vecs = np.stack([v_pref, v_syn1, v_syn2, v_distractor], axis=0)
    doc_offsets = [0, 3, 4]
    for mode in ("int8", "binary_int8"):
        maxsim_index = CompactRetrievalSemanticIndex(
            win_vecs,
            [10, 20],
            pack_sha256="sha256:" + "a" * 64,
            source_manifest_digest="sha256:" + "b" * 64,
            model_id="test/embed",
            doc_offsets=doc_offsets,
            mode=mode,
            agg="maxsim",
        )
        assert maxsim_index.is_multi_window is True
        assert maxsim_index.num_docs == 2
        assert maxsim_index.num_windows == 4

        hits = maxsim_index.search_rids(v_syn2, top_k=2, min_cosine=0.55)
        assert hits[0][0] == 10, f"Expected MaxSim ({mode}) to lift target rid=10 to #1, got {hits}"
        assert hits[0][1] > 0.99
        assert hits[0][1] > avg_hits[1][1] + 0.35

        # vector_for_rid returns primary (window 0 = pref_label) vector with >0.995 cosine!
        primary_vec = maxsim_index.vector_for_rid(10)
        assert primary_vec is not None
        assert float(np.dot(primary_vec, v_pref)) > 0.995


def test_compact_semantic_index_from_pack_metadata(tmp_path: Path) -> None:
    pack_path = tmp_path / "vocab.sqlite"
    meta = _create_sqlite_pack_fixture(pack_path)
    vecs = np.stack([_unit_vec(i, 64) for i in range(5)], axis=0)
    rids = [1, 2, 3, 5, 6]

    sidx = CompactRetrievalSemanticIndex.from_pack_metadata(
        pack_path,
        vecs,
        rids,
        model_id="BAAI/bge-small-en-v1.5",
        mode="int8",
    )
    assert sidx.pack_sha256 == meta["pack_sha256"]
    assert sidx.pack_id == meta["pack_id"]
    assert sidx.source_manifest_digest == meta["source_manifest_digest"]
    assert sidx.model_id == "BAAI/bge-small-en-v1.5"
    assert sidx.dimensions == 64


# ---------------------------------------------------------------------------
# T3.3: LayaRerankFusion Tests
# ---------------------------------------------------------------------------


def test_laya_rerank_qdef_format_and_exact_pinning() -> None:
    exact_cand = {
        "id": "urn:ref:concept:mesh:D014612",
        "scheme": "urn:ref:atlas-resource-scheme:mesh-descriptors",
        "release": "urn:ref:atlas-release:mesh-2025",
        "label": "Vaccines",
        "definition": "Suspensions of killed or attenuated microorganisms administered to induce immunity.",
        "broaderLabels": ["Biological Products", "Immunologic Factors"],
        "matchReason": "exact_preferred",
    }
    alt_exact_cand = {
        "id": "urn:ref:concept:eurovoc:vac",
        "scheme": "urn:ref:atlas-resource-scheme:eurovoc",
        "release": "urn:ref:atlas-release:eurovoc-2025",
        "label": "Immunisation Agents",
        "definition": "Agents conferring active immunity.",
        "_matchReason": "exact_alternate",
    }
    distractor_cand = {
        "id": "urn:ref:concept:fast:sub1",
        "scheme": "urn:ref:atlas-resource-scheme:fast-topical",
        "release": "urn:ref:atlas-release:fast-2025",
        "label": "Vaccines--Law and legislation--United States--Accounting",
        "definition": "Accounting statutes for vaccine distribution.",
    }

    summary = format_candidate_concept_summary(exact_cand)
    assert "Concept: Vaccines" in summary
    assert "Scheme: mesh-descriptors" in summary
    assert "Broader: Biological Products, Immunologic Factors" in summary
    assert "Definition:" in summary

    qdef = build_rerank_noul_qdef(exact_cand, index=3)
    assert qdef["id"] == "refspec_rerank_3"
    assert qdef["type"] == "noul"
    assert "Vaccines" in qdef["instructions"]
    assert set(qdef["criteria"].keys()) == {"false", "true"}

    # Adversarial Laya scores distractor at 0.99 and would score exact matches at 0.01!
    def _adversarial_score(_state: Any, qd: Mapping[str, Any]) -> float:
        return 0.99 if "Accounting" in qd["instructions"] else 0.01

    backend = _FakeLayaBackend(_adversarial_score)
    reranker = LayaRerankFusion(backend, prior_weight=0.0, deduplicate_labels=True, attach_score=True)

    # Put distractor BEFORE exact matches in input list to prove exact pinning overrides order & Laya score!
    original_inputs = [distractor_cand, exact_cand, alt_exact_cand]
    out = reranker.rerank(query="vaccines", candidates=original_inputs, limit=3)

    assert [c["id"] for c in out] == [
        exact_cand["id"],
        alt_exact_cand["id"],
        distractor_cand["id"],
    ]
    # Exact matches were NOT sent to Laya; only the single non-exact distractor was scored in 1 batch!
    assert len(backend.score_batch_calls) == 1
    assert len(backend.score_batch_calls[0]) == 1
    assert "Accounting" in backend.score_batch_calls[0][0]["qdef"]["instructions"]
    assert out[2]["layaScore"] == pytest.approx(0.99, abs=1e-6)
    assert "layaScore" not in out[0]
    assert "layaScore" not in distractor_cand  # Never mutates input dicts!

    # Short-circuit: when pinned exact matches already satisfy limit, Laya is NOT called at all.
    backend_skip = _FakeLayaBackend(_adversarial_score)
    reranker_skip = LayaRerankFusion(backend_skip, deduplicate_labels=True)
    out_exact_only = reranker_skip.rerank(query="vaccines", candidates=original_inputs, limit=2)
    assert len(out_exact_only) == 2
    assert len(backend_skip.score_batch_calls) == 0


def test_laya_rerank_paraphrase_lift_over_fast_subdivision_and_dedup() -> None:
    candidates = [
        {
            "id": "urn:ref:concept:fast:1",
            "scheme": "urn:ref:atlas-resource-scheme:fast-topical",
            "release": "urn:ref:atlas-release:fast-2025",
            "label": "Veterans--Medical care--Accounting--Statistics",
            "definition": "Statistical accounting for veteran hospital budgets.",
            "sourceRecord": "urn:ref:atlas-source-record:fast-1",
        },
        {
            "id": "urn:ref:concept:fast:2",
            "scheme": "urn:ref:atlas-resource-scheme:fast-topical",
            "release": "urn:ref:atlas-release:fast-2025",
            "label": "Military hospitals--Administration--Handbooks",
            "definition": "Administrative manuals for armed forces hospitals.",
            "sourceRecord": "urn:ref:atlas-source-record:fast-2",
        },
        {
            "id": "urn:ref:concept:fr:vet-health",
            "scheme": "urn:ref:atlas-resource-scheme:federal-register-thesaurus-2025",
            "release": "urn:ref:atlas-release:fr-2025",
            "label": "Veterans Health Services",
            "definition": "Medical treatment and healthcare programs provided to former service members.",
            "broaderLabels": ["Veterans Benefits", "Health Care"],
            "sourceRecord": "urn:ref:atlas-source-record:fr-1",
        },
        {
            "id": "urn:ref:concept:mesh:vet-health-dup",
            "scheme": "urn:ref:atlas-resource-scheme:mesh-descriptors",
            "release": "urn:ref:atlas-release:mesh-2025",
            "label": "Veterans Health Services",
            "definition": "Healthcare services delivered to military veterans.",
            "sourceRecord": "urn:ref:atlas-source-record:mesh-1",
        },
    ]

    def _semantic_laya_score(state: Mapping[str, Any], qd: Mapping[str, Any]) -> float:
        assert state == {"query": "medical care for military veterans"}
        ins = qd["instructions"]
        if "Veterans Health Services" in ins:
            return 0.94 if "federal-register" in ins else 0.90
        if "Accounting" in ins:
            return 0.14
        return 0.20

    backend = _FakeLayaBackend(_semantic_laya_score)
    reranker = LayaRerankFusion(
        backend,
        max_rerank=20,
        prior_weight=0.25,
        deduplicate_labels=True,
        attach_score=True,
    )

    out = reranker.rerank(
        query="medical care for military veterans",
        candidates=candidates,
        limit=3,
    )
    # Single batched Laya call scored all 4 non-exact candidates at once!
    assert len(backend.score_batch_calls) == 1
    assert len(backend.score_batch_calls[0]) == 4

    # Concise target lifted from rank #2 to #1 above the FAST subdivision!
    assert out[0]["id"] == "urn:ref:concept:fr:vet-health"
    assert out[0]["label"] == "Veterans Health Services"
    assert out[0]["layaScore"] == pytest.approx(0.94, abs=1e-6)

    # Duplicate label from MeSH collapsed cleanly into `counterparts`!
    assert "counterparts" in out[0]
    assert len(out[0]["counterparts"]) == 1
    assert out[0]["counterparts"][0]["id"] == "urn:ref:concept:mesh:vet-health-dup"
    assert out[0]["counterparts"][0]["scheme"] == "urn:ref:atlas-resource-scheme:mesh-descriptors"
    assert len(out) == 3


def test_laya_rerank_choice_only_fallback_and_error_fallback() -> None:
    choice_be = _ChoiceOnlyFakeLayaBackend(
        lambda _s, qd: 0.92 if "Target Concept" in qd["instructions"] else 0.10
    )
    reranker = LayaRerankFusion(choice_be, prior_weight=0.1, deduplicate_labels=False)
    cands = [
        {"id": "c1", "label": "Subdivision--A--B", "scheme": "fast"},
        {"id": "c2", "label": "Target Concept", "scheme": "mesh"},
    ]
    out = reranker.rerank(query="target query", candidates=cands, limit=2)
    assert [c["id"] for c in out] == ["c2", "c1"]
    assert choice_be.calls == 1

    class _BrokenLaya:
        def encode_items_for_state(self, _s: Any, _q: Any) -> list[dict[str, Any]]:
            raise RuntimeError("ANE unavailable")

    reranker_fb = LayaRerankFusion(_BrokenLaya(), fallback_on_error=True, deduplicate_labels=False)
    fb_out = reranker_fb.rerank(query="target query", candidates=cands, limit=2)
    assert [c["id"] for c in fb_out] == ["c1", "c2"]


def test_laya_rerank_alt_label_pinning_max_rerank_cap_and_lazy_factory() -> None:
    created: list[_FakeLayaBackend] = []

    def _make_backend() -> _FakeLayaBackend:
        be = _FakeLayaBackend(lambda _s, qd: 0.95 if "Beta" in qd["instructions"] else 0.10)
        created.append(be)
        return be

    candidates = [
        {"id": "c1", "label": "Alpha Topic", "scheme": "fast"},
        {"id": "c2", "label": "Beta Topic", "scheme": "mesh"},
        {"id": "c3", "label": "Gamma Topic", "scheme": "eurovoc"},
        {
            "id": "c4",
            "label": "Myocardial Infarction",
            "scheme": "mesh",
            "labels": [{"text": "Heart Attack", "role": "altLabel"}],
        },
    ]

    with LayaRerankFusion(
        backend_factory=_make_backend,
        max_rerank=2,
        prior_weight=0.0,
        deduplicate_labels=False,
    ) as reranker:
        assert reranker.is_initialized is False
        out = reranker.rerank(query="heart attack", candidates=candidates, limit=4)
        assert reranker.is_initialized is True
        assert len(created) == 1
        # c4 pinned at #1 via exact altLabel in `labels`; among non-exact [c1, c2, c3],
        # only max_rerank=2 ([c1, c2]) are sent to Laya, lifting c2 above c1 while c3 stays tail!
        assert [c["id"] for c in out] == ["c4", "c2", "c1", "c3"]
        assert len(created[0].score_batch_calls) == 1
        assert len(created[0].score_batch_calls[0]) == 2
        assert "layaScore" in out[1] and "layaScore" not in out[3]

    assert created[0].closed is True


def test_compact_semantic_index_mrl_int8_and_fp16_modes() -> None:
    vecs = np.stack([_unit_vec(i + 10, 128) for i in range(8)], axis=0)
    rids = [100 + i for i in range(8)]

    mrl_idx = CompactRetrievalSemanticIndex(
        vecs,
        rids,
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="google/embeddinggemma-2",
        mode="mrl_int8",
        is_mrl_trained=True,
        mrl_dim=64,
    )
    assert mrl_idx.mode == "mrl_int8"
    assert mrl_idx.dimensions == 64
    # Full raw_dim=128 query auto-truncates cleanly against mrl_dim=64 index!
    hits = mrl_idx.search_rids(vecs[3], top_k=2, min_cosine=0.90)
    assert hits[0][0] == 103
    assert hits[0][1] > 0.99
    v3 = mrl_idx.vector_for_rid(103)
    assert v3 is not None and v3.shape == (64,)
    exp3 = vecs[3, :64] / np.linalg.norm(vecs[3, :64])
    assert float(np.dot(v3, exp3)) > 0.995

    fp16_idx = CompactRetrievalSemanticIndex(
        vecs,
        rids,
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="test/fp16",
        mode="fp16",
    )
    v5 = fp16_idx.vector_for_rid(105)
    assert v5 is not None and v5.dtype == np.float32
    assert float(np.dot(v5, vecs[5])) > 0.9995


# ---------------------------------------------------------------------------
# T3.4: Streaming SQLite ETL & Parquet Round-Trip Tests
# ---------------------------------------------------------------------------


def test_export_pack_concept_texts_streaming_and_windows(tmp_path: Path) -> None:
    pack_path = tmp_path / "vocab.sqlite"
    meta = _create_sqlite_pack_fixture(pack_path)

    # Stream in small batches (batch_size=2) across 5 labeled concepts -> 3 batches (2, 2, 1).
    batches = list(export_pack_concept_texts(pack_path, batch_size=2, multi_window=False))
    assert [len(b) for b in batches] == [2, 2, 1]
    records = [r for b in batches for r in b]
    assert all(isinstance(r, PackConceptRecord) for r in records)
    # rid=4 (pref_label IS NULL) MUST be skipped.
    assert [r.rid for r in records] == [1, 2, 3, 5, 6]

    # rid=2 and rid=3 share identical label & no definition -> identical input_sha256!
    assert records[1].card_text == "Clean Water"
    assert records[2].card_text == "Clean Water"
    assert records[1].input_sha256 == records[2].input_sha256

    # Multi-window mode produces [pref_label, alt_labels..., definition].
    mw_records = [
        r
        for b in export_pack_concept_texts(pack_path, batch_size=2, multi_window=True)
        for r in b
    ]
    mi = mw_records[0]
    assert mi.rid == 1
    assert mi.windows == (
        "Myocardial Infarction",
        "Cardiac Arrest",
        "Heart Attack",
        "Necrosis of the myocardium caused by obstruction of coronary blood supply.",
    )

    # Scheme filter restricts cleanly to MeSH only.
    mesh_only = [
        r
        for b in export_pack_concept_texts(
            pack_path,
            schemes=[meta["schemes"][1]],
            multi_window=False,
        )
        for r in b
    ]
    assert [r.rid for r in mesh_only] == [1]


def test_embed_sqlite_pack_to_parquet_roundtrip_dedup_and_sharding(tmp_path: Path) -> None:
    pack_path = tmp_path / "vocab.sqlite"
    meta = _create_sqlite_pack_fixture(pack_path)
    out_dir = tmp_path / "parquet_out"

    encoder = _FakeFastEmbeddingGemma2(honor_dimensions=True)
    summary = embed_sqlite_pack_to_parquet(
        pack_path,
        encoder,
        out_dir,
        dimensions=256,
        batch_size=2,
        shard_rows=2,
        multi_window=False,
        vector_dtype="float16",
        return_index=True,
        index_mode="binary_int8",
    )

    assert isinstance(summary, PackEmbedSummary)
    assert summary.pack_sha256 == meta["pack_sha256"]
    assert summary.pack_id == meta["pack_id"]
    assert summary.source_manifest_digest == meta["source_manifest_digest"]
    assert summary.model_id == "google/embeddinggemma-2"
    assert summary.dimensions == 256
    assert summary.resource_count == 5
    # rid=2 & rid=3 share exact same "Clean Water" card -> 4 unique vectors, 1 dedup hit!
    assert summary.unique_vector_count == 4
    assert summary.deduplicated_hits == 1
    assert summary.windows_parquet is None

    # Verify shards: 4 unique vectors with shard_rows=2 -> part-000.parquet & part-001.parquet.
    assert [p.name for p in summary.vector_files] == ["part-000.parquet", "part-001.parquet"]
    assert summary.resources_parquet == out_dir.resolve() / "resources.parquet"
    assert summary.manifest_path.is_file()

    # Verify Parquet schemas & exact RefSpec `build_semantic_sidecar` read path!
    res_tbl = pq.read_table(summary.resources_parquet)
    assert res_tbl.num_rows == 5
    assert {"resource_id", "scheme", "release", "input_sha256"}.issubset(set(res_tbl.column_names))

    vec_shas: list[str] = []
    vec_arrays: list[np.ndarray] = []
    for part_path in summary.vector_files:
        vtbl = pq.read_table(part_path)
        assert vtbl.column_names == ["input_sha256", "embedding"]
        vec_shas.extend(vtbl.column("input_sha256").to_pylist())
        emb_col = vtbl.column("embedding")
        for chunk in emb_col.chunks:
            flat = chunk.values.to_numpy(zero_copy_only=False)
            vec_arrays.append(np.asarray(flat, dtype=np.float16).reshape(len(chunk), -1))

    assert len(vec_shas) == 4
    assert len(set(vec_shas)) == 4
    all_vecs = np.concatenate(vec_arrays, axis=0)
    assert all_vecs.shape == (4, 256)

    # Every resource's input_sha256 exists in vectors/part-*.parquet!
    res_shas = set(res_tbl.column("input_sha256").to_pylist())
    assert res_shas == set(vec_shas)

    # Returned CompactRetrievalSemanticIndex works end-to-end with FastSiliconQueryEncoder!
    sidx = summary.semantic_index
    assert sidx is not None
    assert sidx.pack_sha256 == meta["pack_sha256"]
    assert sidx.dimensions == 256
    qenc = FastSiliconQueryEncoder(encoder, dimensions=256)
    qvec = qenc.encode_query("Clean Water")
    hits = sidx.search_rids(qvec, top_k=5, min_cosine=0.90)
    # Both deduplicated rids (2 and 3) match "Clean Water" at ~1.0 cosine!
    assert {hits[0][0], hits[1][0]} == {2, 3}
    assert hits[0][1] > 0.99


def test_embed_sqlite_pack_multilabel_maxsim_index_and_parquet(tmp_path: Path) -> None:
    pack_path = tmp_path / "vocab.sqlite"
    meta = _create_sqlite_pack_fixture(pack_path)
    out_dir = tmp_path / "parquet_mw"

    bge = _FakeFastBgeSmall()
    qenc = FastSiliconQueryEncoder(bge)

    # 1) Direct in-memory multi-window MaxSim index via embed_sqlite_pack_to_index
    mw_index = embed_sqlite_pack_to_index(
        pack_path,
        bge,
        batch_size=2,
        multi_window=True,
        index_mode="int8",
        agg="maxsim",
    )
    assert mw_index.pack_sha256 == meta["pack_sha256"]
    assert mw_index.is_multi_window is True
    assert mw_index.num_docs == 5

    # Querying alternate label "Heart Attack" directly hits rid=1 ("Myocardial Infarction") at >0.99!
    q_heart = qenc.encode_query("Heart Attack")
    hits = mw_index.search_rids(q_heart, top_k=3, min_cosine=0.90)
    assert hits[0][0] == 1
    assert hits[0][1] > 0.99

    # Querying alternate label "Solar Electricity" directly hits rid=5 ("Photovoltaic Power") at >0.99!
    q_solar = qenc.encode_query("Solar Electricity")
    hits_solar = mw_index.search_rids(q_solar, top_k=3, min_cosine=0.90)
    assert hits_solar[0][0] == 5
    assert hits_solar[0][1] > 0.99

    # vector_for_rid(1) returns primary pref_label ("Myocardial Infarction") vector!
    mi_primary = mw_index.vector_for_rid(1)
    assert mi_primary is not None
    expected_pref = qenc.encode_query("Myocardial Infarction")
    assert float(np.dot(mi_primary, expected_pref)) > 0.995

    # 2) Multi-window Parquet export also writes windows.parquet + deduplicated vectors/part-000.parquet.
    mw_summary = embed_sqlite_pack_to_parquet(
        pack_path,
        bge,
        out_dir,
        batch_size=2,
        multi_window=True,
        vector_dtype="float32",
        return_index=True,
    )
    assert mw_summary.windows_parquet is not None and mw_summary.windows_parquet.is_file()
    win_tbl = pq.read_table(mw_summary.windows_parquet)
    assert win_tbl.num_rows == mw_summary.window_count
    assert mw_summary.deduplicated_hits >= 1  # "Clean Water" shared across rid=2 & rid=3
    assert mw_summary.semantic_index is not None
    assert mw_summary.semantic_index.search_rids(q_heart, top_k=1, min_cosine=0.90)[0][0] == 1


# ---------------------------------------------------------------------------
# Code-Review Regression Tests (Findings 2, 3, 10, 11, 12, 13)
# ---------------------------------------------------------------------------


def test_compact_semantic_index_rejects_binary_bits_exceeding_dim(tmp_path: Path) -> None:
    vecs_128 = np.stack([_unit_vec(i + 20, 128) for i in range(6)], axis=0)
    rids = [10 + i for i in range(6)]

    # 1. Construction from arrays with binary_bits > mrl_dim fails fast.
    with pytest.raises(ValueError, match="binary_bits"):
        CompactRetrievalSemanticIndex(
            vecs_128,
            rids,
            pack_sha256="sha256:" + "a" * 64,
            source_manifest_digest="sha256:" + "b" * 64,
            model_id="test/embed",
            mode="binary_int8",
            mrl_dim=64,
            binary_bits=128,
        )

    # 2. Wrapping a prebuilt CompactRetrievalIndex with binary_bits > dim fails at construction time.
    prebuilt = CompactRetrievalIndex(
        vecs_128,
        mode="binary_int8",
        mrl_dim=64,
        binary_bits=128,
    )
    assert prebuilt.dim == 64 and prebuilt.binary_bits == 128
    with pytest.raises(ValueError, match="binary_bits"):
        CompactRetrievalSemanticIndex(
            prebuilt,
            rids,
            pack_sha256="sha256:" + "a" * 64,
            source_manifest_digest="sha256:" + "b" * 64,
            model_id="test/embed",
        )

    # 3. MRL-trained 768d auto-truncating to 512d rejects binary_bits=768.
    vecs_768 = np.stack([_unit_vec(i + 50, 768) for i in range(4)], axis=0)
    with pytest.raises(ValueError, match="binary_bits"):
        CompactRetrievalSemanticIndex(
            vecs_768,
            [1, 2, 3, 4],
            pack_sha256="sha256:" + "a" * 64,
            source_manifest_digest="sha256:" + "b" * 64,
            model_id="google/embeddinggemma-2",
            mode="binary_int8",
            is_mrl_trained=True,
            binary_bits=768,
        )

    # 4. Valid binary_bits <= mrl_dim works cleanly on 64d queries.
    valid_idx = CompactRetrievalSemanticIndex(
        vecs_128,
        rids,
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="test/embed",
        mode="binary_int8",
        mrl_dim=64,
        binary_bits=64,
    )
    q64 = vecs_128[2, :64] / np.linalg.norm(vecs_128[2, :64])
    hits = valid_idx.search_rids(q64, top_k=1, min_cosine=0.90)
    assert hits and hits[0][0] == 12

    # 5. embed_sqlite_pack_to_index & embed_sqlite_pack_to_parquet(..., return_index=True)
    #    fail fast BEFORE calling the encoder when binary_bits > mrl_dim.
    pack_path = tmp_path / "vocab.sqlite"
    _create_sqlite_pack_fixture(pack_path)

    enc_idx = _FakeFastEmbeddingGemma2()
    with pytest.raises(ValueError, match="binary_bits"):
        embed_sqlite_pack_to_index(
            pack_path,
            enc_idx,
            dimensions=256,
            index_mode="binary_int8",
            mrl_dim=64,
            binary_bits=128,
        )
    assert enc_idx.calls == []

    enc_pq = _FakeFastEmbeddingGemma2()
    with pytest.raises(ValueError, match="binary_bits"):
        embed_sqlite_pack_to_parquet(
            pack_path,
            enc_pq,
            tmp_path / "pq_fail_fast",
            dimensions=256,
            return_index=True,
            index_mode="binary_int8",
            mrl_dim=64,
            binary_bits=128,
        )
    assert enc_pq.calls == []


def test_embed_sqlite_pack_to_parquet_stale_pruning_and_failure_cleanup(tmp_path: Path) -> None:
    pack_path = tmp_path / "vocab.sqlite"
    _create_sqlite_pack_fixture(pack_path)
    out_dir = tmp_path / "parquet_reexport"
    encoder = _FakeFastEmbeddingGemma2(honor_dimensions=True)

    # Initial export with shard_rows=2 creates part-000.parquet and part-001.parquet.
    summary1 = embed_sqlite_pack_to_parquet(
        pack_path,
        encoder,
        out_dir,
        dimensions=64,
        batch_size=2,
        shard_rows=2,
    )
    assert [p.name for p in summary1.vector_files] == ["part-000.parquet", "part-001.parquet"]

    # Plant an extra stale shard (part-002.parquet) and re-export with shard_rows=2.
    vec_dir = out_dir / "vectors"
    stale_part = vec_dir / "part-002.parquet"
    stale_part.write_bytes(b"stale shard data")
    assert stale_part.is_file()

    summary2 = embed_sqlite_pack_to_parquet(
        pack_path,
        encoder,
        out_dir,
        dimensions=64,
        batch_size=2,
        shard_rows=2,
    )
    assert [p.name for p in summary2.vector_files] == ["part-000.parquet", "part-001.parquet"]
    assert sorted(p.name for p in vec_dir.glob("part-*.parquet")) == [
        "part-000.parquet",
        "part-001.parquet",
    ]
    assert not stale_part.exists()
    assert sorted(p.name for p in vec_dir.iterdir()) == ["part-000.parquet", "part-001.parquet"]

    # Mid-run encoder exception after shard 0 is closed must leave ZERO part-*.parquet or temp files!
    class _FailingSecondBatchEncoder(_FakeFastEmbeddingGemma2):
        def encode_texts(self, texts: str | Sequence[str], **kwargs: Any) -> np.ndarray:
            if len(self.calls) >= 1:
                raise RuntimeError("simulated encoder crash on batch 2")
            return super().encode_texts(texts, **kwargs)

    fail_dir = tmp_path / "parquet_mid_fail"
    with pytest.raises(RuntimeError, match="simulated encoder crash on batch 2"):
        embed_sqlite_pack_to_parquet(
            pack_path,
            _FailingSecondBatchEncoder(),
            fail_dir,
            dimensions=64,
            batch_size=2,
            shard_rows=2,
        )

    assert list((fail_dir / "vectors").glob("part-*.parquet")) == []
    assert list((fail_dir / "vectors").iterdir()) == []
    assert not (fail_dir / "resources.parquet").exists()
    assert not (fail_dir / "manifest.json").exists()


def test_encode_text_batch_forwards_explicit_false_add_document_prefix(tmp_path: Path) -> None:
    class _FakeFastPplx:
        """Fake matching FastPplx where omitted add_document_prefix (None) defaults to True."""

        dim: int = 64
        max_dim: int = 64
        model_name: str = "perplexity-ai/pplx-embed-v1-0.6b"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def encode_texts(
            self,
            texts: str | Sequence[str],
            *,
            is_query: bool = False,
            add_query_prefix: bool = False,
            add_document_prefix: bool | None = None,
        ) -> np.ndarray:
            effective_doc_prefix = True if add_document_prefix is None else bool(add_document_prefix)
            self.calls.append(
                {
                    "texts": list(texts) if not isinstance(texts, str) else [texts],
                    "is_query": is_query,
                    "add_query_prefix": add_query_prefix,
                    "add_document_prefix": add_document_prefix,
                    "effective_doc_prefix": effective_doc_prefix,
                }
            )
            items = [texts] if isinstance(texts, str) else list(texts)
            prefix = "doc: " if effective_doc_prefix else ""
            return np.stack([_deterministic_text_vec(prefix + t, self.dim) for t in items], axis=0)

    pplx = _FakeFastPplx()
    v_false = rb._encode_text_batch(pplx, ["Clean Water"], add_document_prefix=False)
    assert pplx.calls[-1]["add_document_prefix"] is False
    assert pplx.calls[-1]["effective_doc_prefix"] is False

    v_true = rb._encode_text_batch(pplx, ["Clean Water"], add_document_prefix=True)
    assert pplx.calls[-1]["add_document_prefix"] is True
    assert pplx.calls[-1]["effective_doc_prefix"] is True
    assert float(np.dot(v_false[0], v_true[0])) < 0.9

    # Also verify end-to-end through embed_sqlite_pack_to_index(add_document_prefix=False).
    pack_path = tmp_path / "vocab.sqlite"
    _create_sqlite_pack_fixture(pack_path)
    pplx.calls.clear()
    embed_sqlite_pack_to_index(pack_path, pplx, add_document_prefix=False)
    assert pplx.calls and all(c["add_document_prefix"] is False for c in pplx.calls)


def test_export_pack_concept_texts_multilingual_pref_labels_and_sparse_in_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Secondary role_rank == 0 multilingual preferred labels must be retained (primary deduped).
    pack_path = tmp_path / "multilingual.sqlite"
    _create_sqlite_pack_fixture(pack_path)
    con = sqlite3.connect(str(pack_path))
    try:
        con.execute(
            """
            INSERT INTO labels (rid, role_rank, value_lower, value, lbl_pfx, lbl_suffix, role, language)
            VALUES (1, 0, 'infarctus du myocarde', 'Infarctus du myocarde', 0, 'l1_fr', 'prefLabel', 'fr')
            """
        )
        con.commit()
    finally:
        con.close()

    mw_records = [
        r for b in export_pack_concept_texts(pack_path, multi_window=True) for r in b
    ]
    mi = next(r for r in mw_records if r.rid == 1)
    assert mi.pref_label == "Myocardial Infarction"
    assert "Infarctus du myocarde" in mi.alt_labels
    assert "Myocardial Infarction" not in mi.alt_labels
    assert mi.windows[0] == "Myocardial Infarction"
    assert "Infarctus du myocarde" in mi.windows
    assert mi.windows.count("Myocardial Infarction") == 1
    assert "Infarctus du myocarde" in mi.card_text

    # 2. Sparse rid gaps (max_rid - min_rid > len(batch_rids) * 4) use chunked WHERE rid IN (...).
    sparse_path = tmp_path / "sparse.sqlite"
    _create_sqlite_pack_fixture(sparse_path)
    con = sqlite3.connect(str(sparse_path))
    try:
        con.execute("DELETE FROM labels")
        con.execute("DELETE FROM resources")
        n_sparse = 450
        res_rows = [
            (
                i * 20 + 1,
                f"urn:ref:concept:eurovoc:c{i}",
                1,
                1,
                None,
                0,
                f"s{i}",
                None,
                f"Concept {i}",
                2,
            )
            for i in range(n_sparse)
        ]
        lbl_rows = []
        for i in range(n_sparse):
            rid = i * 20 + 1
            lbl_rows.append((rid, 0, f"concept {i}", f"Concept {i}", 0, f"p{i}", "prefLabel", "en"))
            lbl_rows.append((rid, 1, f"synonym {i}", f"Synonym {i}", 0, f"a{i}", "altLabel", "en"))
        con.executemany(
            """
            INSERT INTO resources (rid, id, sid, rel_id, definition, src_pfx, src_suffix, notations_json, pref_label, label_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            res_rows,
        )
        con.executemany(
            """
            INSERT INTO labels (rid, role_rank, value_lower, value, lbl_pfx, lbl_suffix, role, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            lbl_rows,
        )
        con.commit()
    finally:
        con.close()

    traced_sql: list[str] = []
    real_connect = sqlite3.connect

    def _traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(traced_sql.append)
        return conn

    monkeypatch.setattr(rb.sqlite3, "connect", _traced_connect)
    sparse_records = [
        r
        for b in export_pack_concept_texts(sparse_path, batch_size=500, multi_window=True)
        for r in b
    ]
    assert len(sparse_records) == 450
    assert sparse_records[0].alt_labels == ("Synonym 0",)
    assert sparse_records[-1].alt_labels == ("Synonym 449",)

    label_queries = [q for q in traced_sql if "FROM labels" in q]
    assert len(label_queries) == 2  # 400 + 50 chunks
    assert all("WHERE rid IN (" in q for q in label_queries)
    assert all("BETWEEN" not in q for q in label_queries)


def test_compact_semantic_index_prebuilt_defaults_bool_rid_and_nonfinite_query() -> None:
    dim = 64
    rng = np.random.default_rng(99)
    q_mat, _ = np.linalg.qr(rng.standard_normal((dim, 3)).astype(np.float32))
    v0 = q_mat[:, 0]
    v1 = q_mat[:, 1]  # Orthogonal to v0

    # Multi-window prebuilt index with default_agg="hybrid", default_hybrid_alpha=0.5.
    # For query=v0: window scores ~ [1.0, 0.0] -> maxsim=1.0, mean=0.5, hybrid(0.5)=0.75.
    prebuilt = CompactRetrievalIndex(
        np.stack([v0, v1, v1], axis=0),
        doc_offsets=[0, 2, 3],
        mode="int8",
        agg="hybrid",
        hybrid_alpha=0.5,
    )
    sidx = CompactRetrievalSemanticIndex(
        prebuilt,
        [1, 0],
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="test/embed",
    )
    assert sidx._agg == "hybrid"
    assert sidx._hybrid_alpha == pytest.approx(0.5)

    hits = sidx.search_rids(v0, top_k=2, min_cosine=0.1)
    assert hits and hits[0][0] == 1
    assert hits[0][1] == pytest.approx(0.75, abs=0.02)

    # Explicit override still works when caller passes agg="maxsim".
    sidx_override = CompactRetrievalSemanticIndex(
        prebuilt,
        [1, 0],
        pack_sha256="sha256:" + "a" * 64,
        source_manifest_digest="sha256:" + "b" * 64,
        model_id="test/embed",
        agg="maxsim",
    )
    hits_maxsim = sidx_override.search_rids(v0, top_k=2, min_cosine=0.1)
    assert hits_maxsim[0][1] > 0.98

    # vector_for_rid must reject bool (True == 1 and False == 0 in Python, both rids exist here!).
    assert sidx.vector_for_rid(1) is not None
    assert sidx.vector_for_rid(0) is not None
    assert sidx.vector_for_rid(True) is None
    assert sidx.vector_for_rid(False) is None
    assert sidx.vector_for_rid(np.bool_(True)) is None  # type: ignore[arg-type]

    # NaN / +Inf / -Inf queries cleanly abstain with [] instead of propagating NaN.
    for bad_val in (np.nan, np.inf, -np.inf):
        bad_q = v0.copy()
        bad_q[0] = bad_val
        assert sidx.search_rids(bad_q, top_k=5, min_cosine=-1.0) == []


def test_format_candidate_concept_summary_caps_long_label_and_broader() -> None:
    long_label = "Title 40 CFR Protection of Environment -- " + "Subchapter " * 30
    long_broader = [
        "Environmental Protection Regulatory Standards and Administrative Procedures",
        "Federal Clean Air Act Stationary Source Emissions Monitoring",
        "Interstate Air Pollution Control Compact Provisions",
        "Public Health Environmental Risk Assessments",
    ]
    definition = "National ambient air quality standards for particulate matter."
    cand = {
        "label": long_label,
        "scheme": "urn:ref:atlas-resource-scheme:cfr-titles",
        "broaderLabels": long_broader,
        "definition": definition,
    }

    summary = format_candidate_concept_summary(cand)
    segments = summary.split(" | ")
    concept_seg = next(s for s in segments if s.startswith("Concept: "))
    broader_seg = next(s for s in segments if s.startswith("Broader: "))
    def_seg = next(s for s in segments if s.startswith("Definition: "))

    label_part = concept_seg.removeprefix("Concept: ")
    broader_part = broader_seg.removeprefix("Broader: ")
    assert len(label_part) <= 160
    assert label_part.endswith("...")
    assert len(broader_part) <= 120
    assert broader_part.endswith("...")
    assert definition in def_seg

    # Custom tighter caps also flow through build_rerank_noul_qdef.
    qdef = build_rerank_noul_qdef(cand, max_label_chars=50, max_broader_chars=40)
    assert "Definition: " + definition in qdef["instructions"]
    custom_summary = format_candidate_concept_summary(
        cand, max_label_chars=50, max_broader_chars=40
    )
    c_parts = dict(seg.split(": ", 1) for seg in custom_summary.split(" | "))
    assert len(c_parts["Concept"]) <= 50 and c_parts["Concept"].endswith("...")
    assert len(c_parts["Broader"]) <= 40 and c_parts["Broader"].endswith("...")
    assert c_parts["Definition"] == definition


# ---------------------------------------------------------------------------
# Track 2 Item 1b: Zero-Copy .vpack v2 Reader Tests
# ---------------------------------------------------------------------------


def _write_synthetic_vpack(
    out_path: Path,
    vectors: np.ndarray,
    rids_per_vector: Sequence[Sequence[int]],
    *,
    mode: str,
    pack_sha256: str = "sha256:" + "a" * 64,
    pack_id: str = "urn:ref:vocabulary-pack:" + "b" * 64,
    source_manifest_digest: str = "sha256:" + "c" * 64,
    model_id: str = "BAAI/bge-m3",
    schema_version: str = "2",
) -> str:
    """Write a byte-exact RefSpec v2 ``.vpack`` binary container and return its ``sha256:...`` digest."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = np.where(norms > 1e-12, arr / np.maximum(norms, 1e-12), np.float32(0.0))
    n_vecs, dim = arr.shape

    def _align64(offset: int) -> int:
        rem = offset % 64
        return offset if rem == 0 else offset + (64 - rem)

    def _quant_i8(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_abs = np.max(np.abs(mat), axis=1)
        scales = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0).astype(np.float32)
        q = np.rint(mat / scales[:, None]).clip(-127, 127).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scales.astype(np.float16))

    sections: list[tuple[str, bytes]] = []
    n_clusters = 0
    if mode == "ivf_i8":
        n_clusters = min(4, n_vecs)
        centroids = np.ascontiguousarray(arr[:n_clusters], dtype=np.float32)
        c_offsets = np.zeros(n_clusters + 1, dtype=np.uint32)
        if n_clusters > 0:
            c_offsets[1:] = n_vecs
        q_i8, s_f16 = _quant_i8(arr)
        sections.append(("centroids_f32", centroids.tobytes()))
        sections.append(("cluster_offsets_u32", c_offsets.tobytes()))
        sections.append(("vectors", q_i8.tobytes()))
        sections.append(("scales_f16", s_f16.tobytes()))
    elif mode == "binary_i8":
        q_i8, s_f16 = _quant_i8(arr)
        bin_u8 = np.ascontiguousarray(np.packbits(arr > 0, axis=1))
        sections.append(("vectors", q_i8.tobytes()))
        sections.append(("scales_f16", s_f16.tobytes()))
        sections.append(("binary_u8", bin_u8.tobytes()))
    elif mode == "flat_f16":
        v_f16 = np.ascontiguousarray(arr.astype(np.float16))
        sections.append(("vectors", v_f16.tobytes()))
    else:
        raise ValueError(f"Unsupported test mode: {mode}")

    rid_offsets = np.zeros(n_vecs + 1, dtype=np.uint32)
    flat_rids: list[int] = []
    max_rid = 0
    for i, rids in enumerate(rids_per_vector):
        for r in rids:
            ri = int(r)
            flat_rids.append(ri)
            if ri > max_rid:
                max_rid = ri
        rid_offsets[i + 1] = len(flat_rids)
    rids_u32 = np.asarray(flat_rids, dtype=np.uint32)
    rid_to_vidx = np.full(max_rid + 1, 0xFFFFFFFF, dtype=np.uint32)
    for vidx, rids in enumerate(rids_per_vector):
        for r in rids:
            ri = int(r)
            if rid_to_vidx[ri] == 0xFFFFFFFF:
                rid_to_vidx[ri] = vidx

    sections.append(("rid_offsets_u32", rid_offsets.tobytes()))
    sections.append(("rids_u32", rids_u32.tobytes()))
    sections.append(("rid_to_vidx_u32", rid_to_vidx.tobytes()))

    magic = b"RFVP0001"
    base_header: dict[str, Any] = {
        "recordType": "AtlasSemanticSidecar",
        "schemaVersion": schema_version,
        "packId": pack_id,
        "packSha256": pack_sha256,
        "sourceManifestDigest": source_manifest_digest,
        "modelId": model_id,
        "mode": mode,
        "dimensions": int(dim),
        "uniqueVectorCount": int(n_vecs),
        "mappedResourceCount": int(rids_u32.size),
        "maxRid": int(max_rid),
        "nClusters": int(n_clusters),
        "builtAt": "2026-11-25T00:00:00Z",
        "sections": {},
    }
    sec_meta: dict[str, dict[str, int]] = {}
    header_bytes = b""
    for _ in range(6):
        cand = dict(base_header, sections=sec_meta)
        header_bytes = json.dumps(cand, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cursor = _align64(len(magic) + 8 + len(header_bytes))
        new_meta: dict[str, dict[str, int]] = {}
        for name, payload in sections:
            off = _align64(cursor)
            new_meta[name] = {"offset": off, "nbytes": len(payload)}
            cursor = off + len(payload)
        if new_meta == sec_meta:
            break
        sec_meta = new_meta

    with out_path.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for name, payload in sections:
            pad = sec_meta[name]["offset"] - handle.tell()
            if pad > 0:
                handle.write(b"\x00" * pad)
            handle.write(payload)

    return f"sha256:{hashlib.sha256(out_path.read_bytes()).hexdigest()}"


@pytest.mark.parametrize("dim", [70, 128, 384])
@pytest.mark.parametrize("vpack_mode,expected_idx_mode", [
    ("binary_i8", "binary_int8"),
    ("ivf_i8", "int8"),
    ("flat_f16", "fp16"),
])
def test_compact_semantic_index_from_vpack_all_modes_zero_copy_and_lifecycle(
    tmp_path: Path,
    dim: int,
    vpack_mode: str,
    expected_idx_mode: str,
) -> None:
    n_vecs = 64
    rng = np.random.default_rng(2026 + dim)
    raw = rng.standard_normal((n_vecs, dim)).astype(np.float32)
    # Plant exact zero coordinates in row 0 and an explicit exact all-zero row at the end!
    raw[0, 0] = 0.0
    raw[0, -1] = 0.0
    vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    vecs[-1] = 0.0

    # Vector 0 maps to 3 deduplicated rids [10, 20, 30]; vector 1 maps to [100] (leaving gaps!).
    rids_per_vec: list[list[int]] = [[10, 20, 30], [100]] + [[200 + i * 2] for i in range(2, n_vecs)]
    zero_row_rid = 200 + (n_vecs - 1) * 2
    vpack_path = tmp_path / f"sidecar_{vpack_mode}_{dim}.vpack"
    digest = _write_synthetic_vpack(vpack_path, vecs, rids_per_vec, mode=vpack_mode)

    # Also test bare hex digest pin (without "sha256:" prefix).
    raw_hex = digest.removeprefix("sha256:")
    with CompactRetrievalSemanticIndex.from_vpack(
        vpack_path,
        trusted_sidecar_digest=raw_hex,
        rerank_top_r=64,
        use_neon=True,
    ) as sidx:
        assert sidx.mode == expected_idx_mode
        assert sidx.vpack_mode == vpack_mode
        assert sidx.dimensions == dim
        assert sidx.num_docs == n_vecs
        assert sidx.model_id == "BAAI/bge-m3"
        assert sidx.pack_sha256 == "sha256:" + "a" * 64
        assert sidx.pack_id == "urn:ref:vocabulary-pack:" + "b" * 64
        assert sidx.source_manifest_digest == "sha256:" + "c" * 64
        # Must NOT materialize giant Python dict on mmap open!
        assert sidx._rid_to_primary_window is None
        assert sidx._rid_to_vidx_u32 is not None
        assert sidx.nbytes > 0

        # Prove 100% zero-copy: every buffer view shares memory directly with the mmap!
        assert sidx._mm is not None
        mm_u8 = np.frombuffer(sidx._mm, dtype=np.uint8)
        try:
            assert np.shares_memory(sidx._rid_offsets, mm_u8)
            assert np.shares_memory(sidx._rids, mm_u8)
            assert np.shares_memory(sidx._rid_to_vidx_u32, mm_u8)
            assert sidx._rid_offsets.flags.owndata is False
            if vpack_mode == "binary_i8":
                assert sidx.index.codes_i8 is not None and np.shares_memory(sidx.index.codes_i8, mm_u8)
                assert sidx.index.scales_f16 is not None and np.shares_memory(sidx.index.scales_f16, mm_u8)
                assert sidx.index.binary_u8 is not None and np.shares_memory(sidx.index.binary_u8, mm_u8)
                # Strict `arr > 0` parity with pack_binary_bits (including exact zeros & non-mult-8 tail!)
                expected_bits = pack_binary_bits(vecs, bits=dim)
                assert np.array_equal(sidx.index.binary_u8, expected_bits)
                assert np.all(sidx.index.binary_u8[-1] == 0)
            elif vpack_mode == "ivf_i8":
                assert sidx.index.codes_i8 is not None and np.shares_memory(sidx.index.codes_i8, mm_u8)
                assert sidx.index.scales_f16 is not None and np.shares_memory(sidx.index.scales_f16, mm_u8)
            else:
                assert sidx.index.dense_vecs is not None and np.shares_memory(sidx.index.dense_vecs, mm_u8)
        finally:
            del mm_u8

        # Search parity: vector 0 expands to all 3 deduplicated rids [10, 20, 30] with ~1.0 cosine!
        hits = sidx.search_rids(vecs[0], top_k=5, min_cosine=0.55)
        assert len(hits) >= 3
        assert {hits[0][0], hits[1][0], hits[2][0]} == {10, 20, 30}
        assert all(sc > 0.99 for _, sc in hits[:3])

        # Also verify single-rid vector 10 -> rid=200 + 10*2 = 220!
        hits10 = sidx.search_rids(vecs[10], top_k=2, min_cosine=0.80)
        assert hits10 and hits10[0][0] == 220 and hits10[0][1] > 0.99

        # vector_for_rid O(1) mmap lookup parity + edge cases (exact zero row, missing gap, OOB, neg, bool)
        for rid, expected_vidx in ((10, 0), (20, 0), (30, 0), (100, 1), (220, 10)):
            reconstructed = sidx.vector_for_rid(rid)
            assert reconstructed is not None
            assert reconstructed.dtype == np.float32
            assert reconstructed.shape == (dim,)
            assert np.isclose(float(np.linalg.norm(reconstructed)), 1.0, atol=1e-5)
            assert float(np.dot(reconstructed, vecs[expected_vidx])) > 0.995

        zero_vec = sidx.vector_for_rid(zero_row_rid)
        assert zero_vec is not None
        assert zero_vec.dtype == np.float32 and zero_vec.shape == (dim,)
        assert np.all(zero_vec == 0.0)

        assert sidx.vector_for_rid(55) is None  # 0xFFFFFFFF gap inside [0, max_rid]
        assert sidx.vector_for_rid(999999) is None  # out of bounds
        assert sidx.vector_for_rid(-1) is None  # negative
        assert sidx.vector_for_rid(True) is None  # bool rejected
        assert sidx.vector_for_rid(False) is None
        assert sidx.vector_for_rid(np.bool_(True)) is None  # type: ignore[arg-type]

        # Non-NEON parity check against the same .vpack file
        with CompactRetrievalSemanticIndex.from_vpack(
            vpack_path, trusted_sidecar_digest=digest, rerank_top_r=64, use_neon=False
        ) as sidx_np:
            hits_np = sidx_np.search_rids(vecs[10], top_k=3, min_cosine=0.50)
            assert hits_np[0][0] == hits10[0][0]
            assert hits_np[0][1] == pytest.approx(hits10[0][1], abs=1e-3)

    # Idempotent close() after context exit & closed-index guard
    sidx.close()
    assert sidx.nbytes == 0
    with pytest.raises(ValueError, match="closed"):
        sidx.search_rids(vecs[0], top_k=1)
    with pytest.raises(ValueError, match="closed"):
        sidx.vector_for_rid(10)


@pytest.mark.parametrize("vpack_mode", ["binary_i8", "ivf_i8", "flat_f16"])
def test_compact_semantic_index_from_vpack_empty_zero_vectors(
    tmp_path: Path,
    vpack_mode: str,
) -> None:
    dim = 70
    empty_vecs = np.empty((0, dim), dtype=np.float32)
    vpack_path = tmp_path / f"empty_n0_{vpack_mode}.vpack"
    digest = _write_synthetic_vpack(vpack_path, empty_vecs, [], mode=vpack_mode)

    with CompactRetrievalSemanticIndex.from_vpack(
        vpack_path, trusted_sidecar_digest=digest, use_neon=True
    ) as sidx:
        assert sidx.num_docs == 0
        assert sidx.dimensions == dim
        assert sidx.search_rids(np.ones(dim, dtype=np.float32), top_k=5, min_cosine=0.0) == []
        assert sidx.vector_for_rid(0) is None


def test_compact_semantic_index_from_vpack_rejects_corrupt_files_without_fd_leak(
    tmp_path: Path,
) -> None:
    vecs = np.stack([_unit_vec(i, 32) for i in range(4)], axis=0)
    valid_path = tmp_path / "valid.vpack"
    _write_synthetic_vpack(valid_path, vecs, [[1], [2], [3], [4]], mode="binary_i8")
    valid_bytes = valid_path.read_bytes()

    fd_dir = Path("/dev/fd")
    fds_before = len(list(fd_dir.iterdir())) if fd_dir.is_dir() else None

    # 1. Missing file -> ValueError
    with pytest.raises(ValueError, match="not found"):
        CompactRetrievalSemanticIndex.from_vpack(tmp_path / "does_not_exist.vpack")

    # 2. Digest mismatch -> ValueError
    with pytest.raises(ValueError, match="digest mismatch"):
        CompactRetrievalSemanticIndex.from_vpack(
            valid_path,
            trusted_sidecar_digest="sha256:" + "0" * 64,
        )

    # 3. Empty file -> ValueError
    empty_path = tmp_path / "empty.vpack"
    empty_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty or truncated"):
        CompactRetrievalSemanticIndex.from_vpack(empty_path)

    # 4. Bad magic -> ValueError
    bad_magic = tmp_path / "bad_magic.vpack"
    bad_magic.write_bytes(b"NOPE0001" + valid_bytes[8:])
    with pytest.raises(ValueError, match="magic"):
        CompactRetrievalSemanticIndex.from_vpack(bad_magic)

    # 5. Truncated payload -> ValueError (section bounds check before frombuffer!)
    truncated = tmp_path / "truncated.vpack"
    truncated.write_bytes(valid_bytes[:-32])
    with pytest.raises(ValueError, match="bounds invalid"):
        CompactRetrievalSemanticIndex.from_vpack(truncated)

    # 6. Unsupported schemaVersion == "1" -> ValueError
    v1_path = tmp_path / "v1.vpack"
    _write_synthetic_vpack(v1_path, vecs, [[1], [2], [3], [4]], mode="flat_f16", schema_version="1")
    with pytest.raises(ValueError, match="schemaVersion"):
        CompactRetrievalSemanticIndex.from_vpack(v1_path)

    # 7. Corrupt JSON header -> ValueError
    bad_json = tmp_path / "bad_json.vpack"
    bad_hdr = b"{not-json"
    bad_json.write_bytes(b"RFVP0001" + struct.pack("<Q", len(bad_hdr)) + bad_hdr + b"\x00" * 64)
    with pytest.raises(ValueError, match="Malformed vpack JSON header"):
        CompactRetrievalSemanticIndex.from_vpack(bad_json)

    # 8. Out-of-bounds rid_offsets_u32 endpoint mismatch -> ValueError (tests post-frombuffer cleanup!)
    corrupt_hdr_len = struct.unpack("<Q", valid_bytes[8:16])[0]
    hdr_dict = json.loads(valid_bytes[16 : 16 + corrupt_hdr_len].decode("utf-8"))
    off_ro = hdr_dict["sections"]["rid_offsets_u32"]["offset"]
    mut = bytearray(valid_bytes)
    struct.pack_into("<I", mut, off_ro, 99)  # rid_offsets[0] != 0
    bad_endpoints = tmp_path / "bad_endpoints.vpack"
    bad_endpoints.write_bytes(bytes(mut))
    with pytest.raises(ValueError, match="rid_offsets_u32 endpoints invalid"):
        CompactRetrievalSemanticIndex.from_vpack(bad_endpoints)

    # 9. Non-monotonic rid_offsets_u32 with valid endpoints [0, 3, 1, 3, 4] -> ValueError
    mut_mono = bytearray(valid_bytes)
    struct.pack_into("<IIIII", mut_mono, off_ro, 0, 3, 1, 3, 4)
    bad_mono = tmp_path / "bad_monotonic.vpack"
    bad_mono.write_bytes(bytes(mut_mono))
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        CompactRetrievalSemanticIndex.from_vpack(bad_mono)

    # 10. Unaligned section offset (rid_offsets_u32 offset % 4 != 0, scales_f16 offset % 2 != 0) -> ValueError
    def _rewrite_vpack_header(src_bytes: bytes, mutate_fn: Any) -> bytes:
        h_len = struct.unpack("<Q", src_bytes[8:16])[0]
        hdr = json.loads(src_bytes[16 : 16 + h_len].decode("utf-8"))
        mutate_fn(hdr)
        new_h = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode("utf-8")
        # Pad file tail with extra bytes so shifted offsets stay within file bounds.
        assert len(new_h) <= h_len + 64
        body_start = min(sec["offset"] for sec in hdr["sections"].values())
        pad_len = max(0, body_start - (16 + len(new_h)))
        # Keep original body aligned by reusing exact tail from original body_start when header fits.
        orig_body_start = min(
            sec["offset"]
            for sec in json.loads(src_bytes[16 : 16 + h_len].decode("utf-8"))["sections"].values()
        )
        assert 16 + len(new_h) <= orig_body_start
        return (
            b"RFVP0001"
            + struct.pack("<Q", len(new_h))
            + new_h
            + b"\x00" * (orig_body_start - (16 + len(new_h)))
            + src_bytes[orig_body_start:]
            + b"\x00" * max(16, pad_len)
        )

    unaligned_u32 = tmp_path / "unaligned_u32.vpack"
    unaligned_u32.write_bytes(
        _rewrite_vpack_header(
            valid_bytes,
            lambda h: h["sections"]["rid_offsets_u32"].update(
                {"offset": h["sections"]["rid_offsets_u32"]["offset"] + 2}
            ),
        )
    )
    with pytest.raises(ValueError, match="4-byte aligned"):
        CompactRetrievalSemanticIndex.from_vpack(unaligned_u32)

    unaligned_f16 = tmp_path / "unaligned_f16.vpack"
    unaligned_f16.write_bytes(
        _rewrite_vpack_header(
            valid_bytes,
            lambda h: h["sections"]["scales_f16"].update(
                {"offset": h["sections"]["scales_f16"]["offset"] + 1}
            ),
        )
    )
    with pytest.raises(ValueError, match="2-byte aligned"):
        CompactRetrievalSemanticIndex.from_vpack(unaligned_f16)

    flat_path = tmp_path / "valid_flat_f16.vpack"
    _write_synthetic_vpack(flat_path, vecs, [[1], [2], [3], [4]], mode="flat_f16")
    unaligned_flat = tmp_path / "unaligned_flat_f16.vpack"
    unaligned_flat.write_bytes(
        _rewrite_vpack_header(
            flat_path.read_bytes(),
            lambda h: h["sections"]["vectors"].update(
                {"offset": h["sections"]["vectors"]["offset"] + 1}
            ),
        )
    )
    with pytest.raises(ValueError, match="2-byte aligned"):
        CompactRetrievalSemanticIndex.from_vpack(unaligned_flat)

    # 11. Symlink rejection (both valid target and dangling symlink) -> ValueError
    sym_valid = tmp_path / "link_valid.vpack"
    sym_valid.symlink_to(valid_path)
    with pytest.raises(ValueError, match="symlink"):
        CompactRetrievalSemanticIndex.from_vpack(sym_valid)

    sym_dangling = tmp_path / "link_dangling.vpack"
    sym_dangling.symlink_to(tmp_path / "nonexistent_target.vpack")
    with pytest.raises(ValueError, match="symlink"):
        CompactRetrievalSemanticIndex.from_vpack(sym_dangling)

    # Verify zero leaked file descriptors across all 11 error paths!
    if fds_before is not None:
        fds_after = len(list(fd_dir.iterdir()))
        assert fds_after <= fds_before


# ---------------------------------------------------------------------------
# Track 2 Item 3a: Calibrated LayaRerankFusion + Kev Confidence Cascade Tests
# ---------------------------------------------------------------------------


class _FakeKevBackend:
    """Test double matching ``Kev08bMlxBackend.score_multi_states`` binary ``noul`` contract."""

    def __init__(self, score_fn: Any) -> None:
        self._score_fn = score_fn
        self.calls: list[dict[str, Any]] = []
        self.closed: bool = False

    def score_multi_states(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
        temperature: float | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        self.calls.append({"items": items, "batch_docs": batch_docs, "temperature": temperature})
        out: list[list[tuple[list[float], list[float]]]] = []
        for state, qdefs in items:
            doc_out: list[tuple[list[float], list[float]]] = []
            for qd in qdefs:
                assert qd["type"] == "noul"
                assert "false" in qd["criteria"] and "true" in qd["criteria"]
                p_true = float(self._score_fn(state, qd))
                # Exact Kev08bMlxBackend return shape: ([P(false), P(true)], [logit_false, logit_true])
                doc_out.append(([1.0 - p_true, p_true], [-1.0, 1.0]))
            out.append(doc_out)
        return out

    def close(self) -> None:
        self.closed = True


class _DuckCalibrationProfile:
    """Pure duck-typed Platt calibration profile matching ``CalibrationProfile.apply``."""

    def __init__(self, platt_a: float, platt_b: float) -> None:
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.calls = 0

    def apply(self, values: Any, *, from_probs: bool = True) -> np.ndarray:
        assert from_probs is True
        self.calls += 1
        arr = np.asarray(values, dtype=np.float64)
        p_c = np.clip(arr, 1e-7, 1.0 - 1e-7)
        logits = np.log(p_c / (1.0 - p_c))
        z = np.clip(self.platt_a * logits + self.platt_b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


def test_laya_rerank_platt_calibration_profile_and_tuple_parity() -> None:
    cands = [
        {"id": "c1", "label": "Alpha Concept", "scheme": "mesh"},
        {"id": "c2", "label": "Beta Concept", "scheme": "eurovoc"},
    ]
    raw_scores = {"Alpha": 0.80, "Beta": 0.25}
    laya = _FakeLayaBackend(lambda _s, qd: raw_scores["Alpha"] if "Alpha" in qd["instructions"] else raw_scores["Beta"])

    platt_a, platt_b = 0.65, -0.30
    profile = _DuckCalibrationProfile(platt_a, platt_b)

    r_obj = LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        binary_profile=profile,
        attach_confidence=True,
    )
    r_tup = LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        binary_profile=(platt_a, platt_b),
        attach_confidence=True,
    )

    out_obj = r_obj.rerank(query="alpha query", candidates=cands, limit=2)
    out_tup = r_tup.rerank(query="alpha query", candidates=cands, limit=2)

    assert profile.calls == 1
    assert [c["id"] for c in out_obj] == ["c1", "c2"]
    assert [c["id"] for c in out_tup] == ["c1", "c2"]
    for c_o, c_t in zip(out_obj, out_tup, strict=True):
        assert c_o["layaScore"] == pytest.approx(c_t["layaScore"], abs=1e-6)
        assert c_o["rerankConfidence"] == pytest.approx(c_t["rerankConfidence"], abs=1e-6)
        assert c_o["rerankEscalated"] is False

    # Verify exact numerical value against manual sigmoid(a * logit(p) + b)
    expected_alpha = 1.0 / (1.0 + math.exp(-(platt_a * math.log(0.80 / 0.20) + platt_b)))
    assert out_obj[0]["layaScore"] == pytest.approx(expected_alpha, abs=1e-5)
    assert out_obj[0]["rerankConfidence"] == pytest.approx(expected_alpha, abs=1e-5)
    # 0.646 lies between low=0.35 and high=0.70 -> "disambiguate"!
    assert out_obj[0]["rerankDecision"] == "disambiguate"
    assert out_obj[1]["rerankDecision"] == "low_confidence"
    # Input dicts were never mutated!
    assert all("rerankConfidence" not in c and "layaScore" not in c for c in cands)


def test_laya_rerank_lazy_escalation_skipped_when_primary_confident() -> None:
    created_kev: list[_FakeKevBackend] = []

    def _kev_factory() -> _FakeKevBackend:
        inst = _FakeKevBackend(lambda _s, _q: 0.5)
        created_kev.append(inst)
        return inst

    laya = _FakeLayaBackend(
        lambda _s, qd: 0.94 if "Clear Winner" in qd["instructions"] else 0.08
    )
    cands = [
        {"id": "c1", "label": "Clear Loser", "scheme": "fast"},
        {"id": "c2", "label": "Clear Winner", "scheme": "mesh"},
    ]

    with LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        escalation_factory=_kev_factory,
        uncertainty_band=(0.30, 0.70),
        escalation_margin=0.08,
        attach_confidence=True,
    ) as reranker:
        out = reranker.rerank(query="clear medical concept", candidates=cands, limit=2)
        # Confident scores (0.94 vs 0.08) outside (0.30, 0.70) and margin 0.86 > 0.08 -> ZERO escalation!
        assert reranker.is_escalation_initialized is False
        assert created_kev == []
        assert [c["id"] for c in out] == ["c2", "c1"]
        assert out[0]["rerankEscalated"] is False
        assert out[0]["rerankDecision"] == "high_confidence"
        assert out[1]["rerankEscalated"] is False
        assert out[1]["rerankDecision"] == "low_confidence"


def test_laya_rerank_lazy_escalation_uncertain_band_reorders_and_attaches_confidence() -> None:
    created_kev: list[_FakeKevBackend] = []

    def _kev_score(_state: Any, qd: Mapping[str, Any]) -> float:
        ins = qd["instructions"]
        if "True Match" in ins:
            return 0.93
        if "Ambiguous Distractor" in ins:
            return 0.18
        raise AssertionError(f"Unexpected candidate escalated to Kev: {ins}")

    def _kev_factory() -> _FakeKevBackend:
        inst = _FakeKevBackend(_kev_score)
        created_kev.append(inst)
        return inst

    def _laya_score(_state: Any, qd: Mapping[str, Any]) -> float:
        ins = qd["instructions"]
        if "Ambiguous Distractor" in ins:
            return 0.62  # inside uncertainty_band (0.30, 0.70)
        if "True Match" in ins:
            return 0.48  # inside uncertainty_band (0.30, 0.70)
        return 0.07  # confident reject -> MUST NOT be escalated

    cands = [
        {"id": "exact_1", "label": "Exact Match", "scheme": "mesh", "matchReason": "exact_preferred"},
        {"id": "ambig_1", "label": "Ambiguous Distractor", "scheme": "fast"},
        {"id": "true_1", "label": "True Match", "scheme": "eurovoc"},
        {"id": "reject_1", "label": "Confident Reject", "scheme": "fast"},
        {"id": "tail_1", "label": "Tail Beyond MaxRerank", "scheme": "fast"},
    ]
    cands_snapshot = [dict(c) for c in cands]

    with LayaRerankFusion(
        _FakeLayaBackend(_laya_score),
        max_rerank=3,
        prior_weight=0.0,
        deduplicate_labels=False,
        escalation_factory=_kev_factory,
        uncertainty_band=(0.30, 0.70),
        escalation_margin=0.08,
        max_escalation_candidates=2,
        kev_batch_docs=4,
        attach_confidence=True,
    ) as reranker:
        assert reranker.is_escalation_initialized is False
        out = reranker.rerank(query="exact match", candidates=cands, limit=5)

        assert reranker.is_escalation_initialized is True
        assert len(created_kev) == 1
        assert len(created_kev[0].calls) == 1
        call = created_kev[0].calls[0]
        assert call["batch_docs"] == 4
        # Single shared query state with the full top_pool (capped at max_escalation_candidates=2)!
        assert len(call["items"]) == 1
        state_sent, qdefs_sent = call["items"][0]
        assert state_sent == {"query": "exact match"}
        assert len(qdefs_sent) == 2

        # Exact stays pinned #1; Kev rescues True Match (0.93) over Ambiguous Distractor (0.18)!
        assert [c["id"] for c in out] == ["exact_1", "true_1", "ambig_1", "reject_1", "tail_1"]

        assert out[0]["rerankDecision"] == "exact"
        assert out[0]["rerankConfidence"] == 1.0
        assert out[0]["rerankEscalated"] is False

        assert out[1]["rerankDecision"] == "high_confidence"
        assert out[1]["rerankConfidence"] == pytest.approx(0.93, abs=1e-6)
        assert out[1]["rerankEscalated"] is True

        assert out[2]["rerankDecision"] == "low_confidence"
        assert out[2]["rerankConfidence"] == pytest.approx(0.18, abs=1e-6)
        assert out[2]["rerankEscalated"] is True

        assert out[3]["rerankDecision"] == "low_confidence"
        assert out[3]["rerankConfidence"] == pytest.approx(0.07, abs=1e-6)
        assert out[3]["rerankEscalated"] is False

        assert out[4]["rerankDecision"] == "low_confidence"
        assert out[4]["rerankConfidence"] == 0.0
        assert out[4]["rerankEscalated"] is False

    # Lazily created escalation backend MUST be closed on exit; input dicts untouched!
    assert created_kev[0].closed is True
    assert cands == cands_snapshot


def test_laya_rerank_close_margin_escalation_and_error_fallback() -> None:
    # Both top-2 candidates score > 0.70 (outside uncertainty_band), yet within escalation_margin=0.08!
    laya = _FakeLayaBackend(
        lambda _s, qd: 0.84 if "Runner Up" in qd["instructions"] else (0.80 if "Better Match" in qd["instructions"] else 0.10)
    )
    kev = _FakeKevBackend(
        lambda _s, qd: 0.96
        if "Better Match" in qd["instructions"]
        else (0.55 if "Runner Up" in qd["instructions"] else 0.09)
    )
    cands = [
        {"id": "c1", "label": "Runner Up", "scheme": "mesh"},
        {"id": "c2", "label": "Better Match", "scheme": "eurovoc"},
        {"id": "c3", "label": "Low Distractor", "scheme": "fast"},
    ]

    with LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        escalation_backend=kev,
        uncertainty_band=(0.30, 0.70),
        escalation_margin=0.08,
        attach_confidence=True,
    ) as reranker:
        out = reranker.rerank(query="close margin query", candidates=cands, limit=3)
        assert [c["id"] for c in out] == ["c2", "c1", "c3"]
        assert len(kev.calls) == 1
        # Entire top_pool (all 3 candidates <= max_escalation_candidates=8) escalated coherently in ONE call!
        assert len(kev.calls[0]["items"][0][1]) == 3
        assert out[0]["rerankEscalated"] is True and out[0]["rerankDecision"] == "high_confidence"
        assert out[1]["rerankEscalated"] is True and out[1]["rerankDecision"] == "disambiguate"
        assert out[2]["rerankEscalated"] is True and out[2]["rerankDecision"] == "low_confidence"

    # Externally injected escalation backend MUST NOT be closed by LayaRerankFusion.close()!
    assert kev.closed is False

    # fallback_on_error=True keeps primary Laya scores cleanly if escalation raises!
    class _BrokenKev:
        def score_multi_states(self, _items: Any, **_kw: Any) -> Any:
            raise RuntimeError("Metal OOM")

    reranker_fb = LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        escalation_backend=_BrokenKev(),
        fallback_on_error=True,
        attach_confidence=True,
    )
    fb_out = reranker_fb.rerank(query="close margin query", candidates=cands, limit=3)
    assert [c["id"] for c in fb_out] == ["c1", "c2", "c3"]
    assert all(c["rerankEscalated"] is False for c in fb_out)


# ---------------------------------------------------------------------------
# Track 2 Code-Review Regression Tests (Findings 1, 3, 5, 6, 9, 13)
# ---------------------------------------------------------------------------


def test_laya_rerank_full_top_pool_escalation_prevents_rank_inversion_and_applies_esc_profile() -> None:
    # Platt calibration on Laya compresses c_top (0.92 -> ~0.77, outside band) while c_mid
    # (0.60 -> ~0.55, inside band) triggers escalation. Both Laya and Kev prefer c_top > c_mid.
    # Partial escalation would leave c_top at 0.77 and upgrade c_mid to Kev's 0.85 -> RANK INVERSION!
    # Coherent full-top_pool escalation sends ALL 3 top_pool candidates in ONE call so c_top stays #1.
    def _laya_score(_s: Any, qd: Mapping[str, Any]) -> float:
        ins = qd["instructions"]
        if "Top Contender" in ins:
            return 0.92
        if "Mid Contender" in ins:
            return 0.60
        return 0.05

    def _kev_score(_s: Any, qd: Mapping[str, Any]) -> float:
        ins = qd["instructions"]
        if "Top Contender" in ins:
            return 0.95
        if "Mid Contender" in ins:
            return 0.85
        return 0.12

    cands = [
        {"id": "c_top", "label": "Top Contender", "scheme": "mesh"},
        {"id": "c_mid", "label": "Mid Contender", "scheme": "eurovoc"},
        {"id": "c_low", "label": "Low Distractor", "scheme": "fast"},
    ]

    kev_raw = _FakeKevBackend(_kev_score)
    reranker = LayaRerankFusion(
        _FakeLayaBackend(_laya_score),
        prior_weight=0.0,
        deduplicate_labels=False,
        binary_profile=(0.50, 0.0),
        escalation_backend=kev_raw,
        uncertainty_band=(0.30, 0.70),
        escalation_margin=0.05,
        attach_confidence=True,
    )
    out = reranker.rerank(query="cardiac surgery", candidates=cands, limit=3)

    assert len(kev_raw.calls) == 1
    assert len(kev_raw.calls[0]["items"][0][1]) == 3
    assert [c["id"] for c in out] == ["c_top", "c_mid", "c_low"]
    assert all(c["rerankEscalated"] is True for c in out)
    assert out[0]["layaScore"] == pytest.approx(0.95, abs=1e-6)
    assert out[1]["layaScore"] == pytest.approx(0.85, abs=1e-6)
    assert out[2]["layaScore"] == pytest.approx(0.12, abs=1e-6)

    # Optional escalation_binary_profile (both Platt tuple and duck-typed profile object) calibrates Kev output.
    esc_a, esc_b = 0.60, -0.25
    esc_profile = _DuckCalibrationProfile(esc_a, esc_b)
    reranker_esc_obj = LayaRerankFusion(
        _FakeLayaBackend(_laya_score),
        prior_weight=0.0,
        deduplicate_labels=False,
        binary_profile=(0.50, 0.0),
        escalation_binary_profile=esc_profile,
        escalation_backend=_FakeKevBackend(_kev_score),
        uncertainty_band=(0.30, 0.70),
        attach_confidence=True,
    )
    reranker_esc_tup = LayaRerankFusion(
        _FakeLayaBackend(_laya_score),
        prior_weight=0.0,
        deduplicate_labels=False,
        binary_profile=(0.50, 0.0),
        escalation_binary_profile=(esc_a, esc_b),
        escalation_backend=_FakeKevBackend(_kev_score),
        uncertainty_band=(0.30, 0.70),
        attach_confidence=True,
    )
    out_obj = reranker_esc_obj.rerank(query="cardiac surgery", candidates=cands, limit=3)
    out_tup = reranker_esc_tup.rerank(query="cardiac surgery", candidates=cands, limit=3)
    assert esc_profile.calls == 1
    expected_top = 1.0 / (1.0 + math.exp(-(esc_a * math.log(0.95 / 0.05) + esc_b)))
    assert out_obj[0]["layaScore"] == pytest.approx(expected_top, abs=1e-5)
    assert out_obj[0]["rerankConfidence"] == pytest.approx(expected_top, abs=1e-5)
    for c_o, c_t in zip(out_obj, out_tup, strict=True):
        assert c_o["layaScore"] == pytest.approx(c_t["layaScore"], abs=1e-6)
        assert c_o["rerankEscalated"] is True

    with pytest.raises(ValueError, match="escalation_binary_profile"):
        LayaRerankFusion(_FakeLayaBackend(_laya_score), escalation_binary_profile=(True, 0.0))
    with pytest.raises(TypeError, match="escalation_binary_profile"):
        LayaRerankFusion(_FakeLayaBackend(_laya_score), escalation_binary_profile="invalid")


def test_laya_rerank_close_margin_escalation_uses_primary_blended_under_default_prior() -> None:
    # Default prior_weight=0.25 with m=2:
    #   pos 0 (prior=1.0): cal_prob=0.72 -> primary_blended = 0.75*0.72 + 0.25*1.0 = 0.79
    #   pos 1 (prior=0.5): cal_prob=0.85 -> primary_blended = 0.75*0.85 + 0.25*0.5 = 0.7625
    # Both cal_probs > 0.70 (outside band) and |0.85 - 0.72| = 0.13 > 0.08, yet
    # |primary_blended[0] - primary_blended[1]| = 0.0275 <= 0.08 -> MUST trigger escalation!
    laya = _FakeLayaBackend(
        lambda _s, qd: 0.72 if "First Prior" in qd["instructions"] else 0.85
    )
    kev = _FakeKevBackend(
        lambda _s, qd: 0.40 if "First Prior" in qd["instructions"] else 0.97
    )
    cands = [
        {"id": "c1", "label": "First Prior", "scheme": "fast"},
        {"id": "c2", "label": "Second Better", "scheme": "mesh"},
    ]

    reranker = LayaRerankFusion(
        laya,
        deduplicate_labels=False,
        escalation_backend=kev,
        uncertainty_band=(0.30, 0.70),
        escalation_margin=0.08,
        attach_confidence=True,
    )
    assert reranker.prior_weight == pytest.approx(0.25)
    out = reranker.rerank(query="near tie blended", candidates=cands, limit=2)

    assert len(kev.calls) == 1
    assert len(kev.calls[0]["items"][0][1]) == 2
    assert [c["id"] for c in out] == ["c2", "c1"]
    assert all(c["rerankEscalated"] is True for c in out)
    assert out[0]["layaScore"] == pytest.approx(0.97, abs=1e-6)
    assert out[1]["layaScore"] == pytest.approx(0.40, abs=1e-6)


def test_laya_rerank_escalation_duck_typed_list_ndarray_and_bool_rejection() -> None:
    class _DuckArrayKev:
        def score_multi_states(
            self,
            items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
            **_kw: Any,
        ) -> list[list[Any]]:
            _, qdefs = items[0]
            assert len(qdefs) == 5
            return [
                [
                    [np.array([0.1, 0.9]), np.array([-1.0, 1.0])],  # list of ndarrays -> 0.9
                    np.array([0.15, 0.85]),  # bare 1D ndarray -> 0.85
                    np.array([0.78]),  # 1-element ndarray -> 0.78
                    0.66,  # plain float scalar -> 0.66
                    np.bool_(True),  # bool rejected in scalar branch -> 0.5
                ]
            ]

    laya = _FakeLayaBackend(lambda _s, _qd: 0.50)
    cands = [{"id": f"c{i}", "label": f"Concept {i}", "scheme": "mesh"} for i in range(5)]
    reranker = LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        escalation_backend=_DuckArrayKev(),
        attach_confidence=True,
    )
    out = reranker.rerank(query="duck typed kev", candidates=cands, limit=5)

    by_id = {c["id"]: c for c in out}
    assert by_id["c0"]["layaScore"] == pytest.approx(0.90, abs=1e-6)
    assert by_id["c1"]["layaScore"] == pytest.approx(0.85, abs=1e-6)
    assert by_id["c2"]["layaScore"] == pytest.approx(0.78, abs=1e-6)
    assert by_id["c3"]["layaScore"] == pytest.approx(0.66, abs=1e-6)
    assert by_id["c4"]["layaScore"] == pytest.approx(0.50, abs=1e-6)
    assert [c["id"] for c in out] == ["c0", "c1", "c2", "c3", "c4"]
    assert all(c["rerankEscalated"] is True for c in out)


def test_laya_rerank_exact_hidden_reason_is_pinned_as_exact() -> None:
    laya = _FakeLayaBackend(lambda _s, _qd: 0.99)
    cands = [
        {"id": "non_exact", "label": "Semantic Distractor", "scheme": "fast"},
        {
            "id": "hidden_1",
            "label": "Myocardial Infarction",
            "scheme": "mesh",
            "matchReason": "exact_hidden",
        },
        {
            "id": "hidden_2",
            "label": "Coronary Thrombosis",
            "scheme": "eurovoc",
            "_matchReason": "exact_hidden",
        },
    ]
    reranker = LayaRerankFusion(
        laya,
        prior_weight=0.0,
        deduplicate_labels=False,
        attach_confidence=True,
    )
    out = reranker.rerank(query="heart attack typo", candidates=cands, limit=3)

    assert [c["id"] for c in out] == ["hidden_1", "hidden_2", "non_exact"]
    assert len(laya.score_batch_calls) == 1
    assert len(laya.score_batch_calls[0]) == 1
    assert out[0]["rerankDecision"] == "exact"
    assert out[0]["rerankConfidence"] == 1.0
    assert out[0]["rerankEscalated"] is False
    assert out[1]["rerankDecision"] == "exact"
    assert out[1]["rerankConfidence"] == 1.0
    assert out[1]["rerankEscalated"] is False
    assert out[2]["rerankDecision"] == "high_confidence"


def test_laya_rerank_close_raising_primary_closes_escalation_idempotently_and_rejects_bool_thresholds() -> None:
    primary_close_calls = 0
    esc_close_calls = 0

    class _RaisingCloseLaya(_FakeLayaBackend):
        def close(self) -> None:
            nonlocal primary_close_calls
            primary_close_calls += 1
            raise RuntimeError("primary close boom")

    class _TrackingKev(_FakeKevBackend):
        def close(self) -> None:
            nonlocal esc_close_calls
            esc_close_calls += 1
            super().close()

    reranker = LayaRerankFusion(
        backend_factory=lambda: _RaisingCloseLaya(lambda _s, _qd: 0.50),
        escalation_factory=lambda: _TrackingKev(lambda _s, _qd: 0.90),
        deduplicate_labels=False,
    )
    reranker.rerank(
        query="close cleanup test",
        candidates=[{"id": "c1", "label": "Alpha", "scheme": "mesh"}],
        limit=1,
    )
    assert reranker.is_initialized is True
    assert reranker.is_escalation_initialized is True

    with pytest.raises(RuntimeError, match="primary close boom"):
        reranker.close()
    assert primary_close_calls == 1
    assert esc_close_calls == 1
    assert reranker.is_initialized is False
    assert reranker.is_escalation_initialized is False

    # Repeated close() MUST be completely idempotent (no second close call, no exception)!
    reranker.close()
    assert primary_close_calls == 1
    assert esc_close_calls == 1

    # Bool / non-finite thresholds MUST raise ValueError.
    laya = _FakeLayaBackend(lambda _s, _qd: 0.5)
    for bad_kw in (
        {"prior_weight": True},
        {"prior_weight": False},
        {"prior_weight": np.bool_(True)},
        {"prior_weight": float("nan")},
        {"prior_weight": float("inf")},
        {"uncertainty_band": (False, 0.7)},
        {"uncertainty_band": (0.3, True)},
        {"uncertainty_band": (0.3, float("nan"))},
        {"escalation_margin": True},
        {"escalation_margin": np.bool_(False)},
        {"escalation_margin": float("inf")},
        {"high_confidence_threshold": True},
        {"low_confidence_threshold": False},
        {"high_confidence_threshold": float("nan")},
        {"low_confidence_threshold": float("-inf")},
    ):
        with pytest.raises(ValueError):
            LayaRerankFusion(laya, **bad_kw)  # type: ignore[arg-type]
