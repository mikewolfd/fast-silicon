"""CPU unit tests for v2 `tokenizer.py` & `parquet_pipeline.py` (`H-P0.3` 10M-doc scale defaults)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_shared.parquet_pipeline import (
    CorpusBackfillConfig,
    build_backfill_cli_parser,
    embed_parquet,
    resolve_macro_window_tokens,
    run_corpus_backfill,
)
from fast_shared.tokenizer import (
    BaseFastTokenizer,
    HeaderAnchoredWindow,
    plan_header_anchored_windows,
)


def _make_bpe_tokenizer(tmp_path: Path, *, doc_prefix: str = "doc: ") -> BaseFastTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing

    vocab = {
        token: idx
        for idx, token in enumerate(["<pad>", "<eos>", "<bos>", "<unk>", *sorted(ByteLevel.alphabet())])
    }
    merges = [("a", "l"), ("al", "p"), ("alp", "h"), ("alph", "a"), ("b", "e"), ("be", "t"), ("bet", "a")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    raw_tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    raw_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 1)],
    )
    tok_path = tmp_path / "bpe_v2_tok.json"
    raw_tok.save(str(tok_path))
    return BaseFastTokenizer(
        tok_path,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        document_prefix=doc_prefix,
    )


class _LengthSensitiveEngine:
    max_dim = 32
    dim = 32
    supports_dimensions = True
    max_seq_len = 512
    model_name = "mock-mrl-emb"
    default_mode = "gpu"
    ane_available = False

    def __init__(self, tokenizer: BaseFastTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[list[list[int]]] = []

    def encode_corpus(self, tok_lists: list[list[int]], mode: str | None = None, dimensions: int | None = None) -> np.ndarray:
        del mode
        self.calls.append([list(x) for x in tok_lists])
        width = int(dimensions) if dimensions is not None else self.dim
        out = np.zeros((len(tok_lists), width), dtype=np.float32)
        for row_i, ids in enumerate(tok_lists):
            base = np.linspace(0.1, 1.0, width, dtype=np.float32)
            # Deterministic vector uniquely determined by exact token sequence
            weighted = np.cos(base * float(sum((idx + 1) * tok for idx, tok in enumerate(ids)) % 997) * 0.013)
            weighted[0] += float(len(ids)) * 0.05
            out[row_i] = weighted / max(float(np.linalg.norm(weighted)), 1e-12)
        return out


def test_chunk_texts_batch_single_pass_rust_slicing_and_unicode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tok = _make_bpe_tokenizer(tmp_path)
    texts = [
        "",
        "alpha beta alpha beta alpha beta",
        "\n é🧪漢字 alpha beta \n é🧪漢字 ",
        "   ",
    ]
    # Ensure chunk_texts_batch uses _tok.encode_batch once for body texts (not per-slice Python _tok.encode loops).
    encode_batch_calls: list[int] = []
    orig_encode_batch = tok._tok.encode_batch

    def _wrapped_encode_batch(batch_inputs, add_special_tokens=True):
        encode_batch_calls.append(len(batch_inputs))
        return orig_encode_batch(batch_inputs, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(tok._tok, "encode_batch", _wrapped_encode_batch)

    batched = tok.chunk_texts_batch(
        texts,
        max_chunk_tokens=16,
        max_body_tokens=6,
        overlap=2,
        add_document_prefix=True,
    )
    assert encode_batch_calls == [len(texts)]
    assert len(batched) == len(texts)

    for txt, win_list in zip(texts, batched, strict=True):
        assert len(win_list) >= 1
        assert win_list[0][1] == 0
        assert win_list[-1][2] == len(txt)
        for ids, c0, c1 in win_list:
            assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
            assert len(ids) <= 16
            assert 0 <= c0 <= c1 <= len(txt)
            # Slice boundaries must never split a multi-byte Python character
            _ = txt[c0:c1].encode("utf-8")

    # Zero-overlap chunks concatenate back to the exact original string
    no_ov = tok.chunk_texts_batch(texts, max_chunk_tokens=16, overlap=0)
    for txt, win_list in zip(texts, no_ov, strict=True):
        assert "".join(txt[c0:c1] for _, c0, c1 in win_list) == txt


def test_plan_header_anchored_windows_default_2048(tmp_path: Path) -> None:
    tok = _make_bpe_tokenizer(tmp_path, doc_prefix="")
    header = "Document Title: Quarterly Infrastructure Reliability Report\n" + ("meta " * 40)
    chunks = [header] + [f"Section {i}: " + ("alpha beta " * 45) for i in range(1, 8)]

    windows = plan_header_anchored_windows(
        chunks,
        max_window_tokens=256,
        header_max_chars=120,
        tokenizer=tok,
    )
    assert len(windows) >= 2
    assert isinstance(windows[0], HeaderAnchoredWindow)
    assert windows[0].has_anchor is False
    assert windows[0].chunks[0] == header
    for win in windows[1:]:
        assert win.has_anchor is True
        assert win.chunks[0] == header[:120]
        assert len(win.chunks) == len(win.chunk_indices) + 1

    # Reconstructed non-anchor chunk indices must equal exact input chunk order 0..len(chunks)-1
    flat_indices = [idx for win in windows for idx in win.chunk_indices]
    assert flat_indices == list(range(len(chunks)))

    # Default max_window_tokens=2048 keeps moderate documents in a single un-anchored window
    single = plan_header_anchored_windows(chunks[:3])
    assert len(single) == 1 and single[0].has_anchor is False


def test_macro_window_length_sort_and_exact_scatter_order(tmp_path: Path) -> None:
    assert resolve_macro_window_tokens(None, host_mem_gb=24.0) == 1_000_000
    assert resolve_macro_window_tokens(None, host_mem_gb=16.0) == 1_000_000
    assert resolve_macro_window_tokens(None, host_mem_gb=36.0) == 2_000_000
    assert resolve_macro_window_tokens(None, host_mem_gb=48.0) == 2_000_000
    assert resolve_macro_window_tokens(4096, host_mem_gb=48.0) == 4096

    tok = _make_bpe_tokenizer(tmp_path, doc_prefix="")
    # Alternating wildly different lengths to verify length-sort inside macro-window + exact scatter back
    lengths = [180, 12, 150, 8, 120, 24, 95, 6, 160, 15, 140, 10]
    rows = [
        {"doc_id": f"d-{i:02d}", "body": ("alpha " * ln).strip()}
        for i, ln in enumerate(lengths)
    ]
    in_pq = tmp_path / "unsorted.parquet"
    out_pq = tmp_path / "sorted_out.parquet"
    pq.write_table(pa.Table.from_pylist(rows), in_pq)

    eng_sorted = _LengthSensitiveEngine(tok)
    eng_unsorted = _LengthSensitiveEngine(tok)

    embed_parquet(
        eng_sorted,
        in_pq,
        out_pq,
        encode_batch_rows=4,
        sort_macro_window=True,
        macro_window_tokens=100_000,
        progress_every=0,
    )
    out_unsorted = tmp_path / "unsorted_out.parquet"
    embed_parquet(
        eng_unsorted,
        in_pq,
        out_unsorted,
        encode_batch_rows=4,
        sort_macro_window=False,
        progress_every=0,
    )

    # Every encode_corpus call under sort_macro_window=True must be monotonically non-decreasing in token length
    assert len(eng_sorted.calls) == 3
    all_sorted_lens = [len(seq) for call in eng_sorted.calls for seq in call]
    assert all_sorted_lens == sorted(all_sorted_lens)

    # Output rows and vectors must match exact input order bit-for-bit against unsorted execution
    tbl_s = pq.read_table(out_pq)
    tbl_u = pq.read_table(out_unsorted)
    assert tbl_s["doc_id"].to_pylist() == [r["doc_id"] for r in rows]
    vecs_s = np.asarray(tbl_s["vector"].to_pylist(), dtype=np.float32)
    vecs_u = np.asarray(tbl_u["vector"].to_pylist(), dtype=np.float32)
    assert np.allclose(vecs_s, vecs_u, atol=1e-6)


def test_int8_quantization_per_column_compression_and_wired_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_shared.parquet_pipeline as pp

    wired_entered: list[bool] = []
    orig_wired = pp.wired_limit

    def _spy_wired():
        wired_entered.append(True)
        return orig_wired()

    monkeypatch.setattr(pp, "wired_limit", _spy_wired)

    tok = _make_bpe_tokenizer(tmp_path)
    eng = _LengthSensitiveEngine(tok)
    rows = [{"doc_id": i, "body": f"alpha beta record {i} " * (3 + i)} for i in range(10)]
    in_pq = tmp_path / "in_i8.parquet"
    out_dir = tmp_path / "shards_i8"
    pq.write_table(pa.Table.from_pylist(rows), in_pq)

    summary = embed_parquet(
        eng,
        in_pq,
        out_dir,
        dtype="int8",
        mrl_dim=16,
        shard_rows=4,
        chunk=True,
        max_chunk_tokens=24,
        overlap=2,
        include_chunk_text=True,
        progress_every=0,
    )
    # Default use_wired_limit=True must have entered wired_limit()
    assert wired_entered == [True]
    assert summary.rows_emitted >= 10

    shards = sorted(out_dir.glob("part-*.parquet"))
    assert len(shards) >= 3
    pf0 = pq.ParquetFile(shards[0])
    # Vector column must be UNCOMPRESSED while metadata/text columns stay ZSTD compressed
    comp_by_col = {
        pf0.metadata.row_group(0).column(i).path_in_schema: pf0.metadata.row_group(0).column(i).compression
        for i in range(pf0.metadata.num_columns)
    }
    assert comp_by_col["vector.list.element"] == "UNCOMPRESSED"
    assert comp_by_col["chunk_text"] == "ZSTD"

    joined = pa.concat_tables([pq.read_table(p) for p in shards])
    assert joined.schema.field("vector").type == pa.list_(pa.int8(), 16)
    assert joined.schema.field("embedding_scale").type == pa.float16()

    q_vecs = np.asarray(joined["vector"].to_pylist(), dtype=np.int8)
    scales = np.asarray(joined["embedding_scale"].to_pylist(), dtype=np.float16)
    deq = q_vecs.astype(np.float32) * scales.astype(np.float32)[:, None]
    norms = np.linalg.norm(deq, axis=1)
    assert np.allclose(norms, 1.0, atol=2e-2)

    # Sharded resume with dtype="int8" must skip all already-written shards cleanly
    eng_resume = _LengthSensitiveEngine(tok)
    embed_parquet(
        eng_resume,
        in_pq,
        out_dir,
        dtype="int8",
        mrl_dim=16,
        shard_rows=4,
        chunk=True,
        max_chunk_tokens=24,
        overlap=2,
        include_chunk_text=True,
        resume=True,
        progress_every=0,
    )
    assert eng_resume.calls == []


def test_corpus_backfill_config_and_cli_defaults_d512_int8(tmp_path: Path) -> None:
    cfg = CorpusBackfillConfig()
    assert cfg.dtype == "int8"
    assert cfg.mrl_dim == 512
    assert cfg.use_wired_limit is True
    assert cfg.sort_macro_window is True

    parser = build_backfill_cli_parser()
    ns_default = parser.parse_args(["--input", "a.parquet", "--output", "b.parquet"])
    assert ns_default.dtype == "int8"
    assert ns_default.mrl_dim == 512

    ns_f32 = parser.parse_args(["--input", "a.parquet", "--output", "b.parquet", "--dtype", "float32"])
    assert ns_f32.dtype == "float32"

    # Clamps mrl_dim=512 safely down to engine max_dim (32) when running CorpusBackfillConfig on narrower models
    tok = _make_bpe_tokenizer(tmp_path)
    eng = _LengthSensitiveEngine(tok)
    in_pq = tmp_path / "bf_in.parquet"
    out_pq = tmp_path / "bf_out.parquet"
    pq.write_table(pa.table({"id": [1, 2], "body": ["alpha beta", "beta alpha"]}), in_pq)
    run_corpus_backfill(eng, in_pq, out_pq, progress_every=0)
    out_tbl = pq.read_table(out_pq)
    assert out_tbl.schema.field("vector").type == pa.list_(pa.int8(), 32)


def test_optional_lance_sink_graceful_and_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_shared.parquet_pipeline as pp

    tok = _make_bpe_tokenizer(tmp_path)
    eng = _LengthSensitiveEngine(tok)
    in_pq = tmp_path / "lance_in.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3, 4], "body": ["alpha", "beta", "alpha beta", "beta"]}), in_pq)

    # 1. When pylance is absent, requesting .lance raises a clear RuntimeError
    monkeypatch.setattr(pp, "_import_lance", lambda: None)
    with pytest.raises(RuntimeError, match="lance"):
        embed_parquet(eng, in_pq, tmp_path / "missing.lance", progress_every=0)

    # 2. When lance module is importable, streams batches via write_dataset (create -> append)
    written_calls: list[tuple[str, str, int]] = []
    written_tables: list[pa.Table] = []

    def _fake_write_dataset(data_obj, uri, *, mode="create", **kwargs):
        del kwargs
        tbl = data_obj if isinstance(data_obj, pa.Table) else pa.Table.from_batches(list(data_obj))
        written_calls.append((str(uri), str(mode), tbl.num_rows))
        written_tables.append(tbl)

    monkeypatch.setattr(pp, "_import_lance", lambda: SimpleNamespace(write_dataset=_fake_write_dataset))
    out_lance = tmp_path / "out_ds.lance"
    summary = embed_parquet(
        eng,
        in_pq,
        out_lance,
        dtype="int8",
        mrl_dim=16,
        batch_rows=2,
        progress_every=0,
    )
    assert summary.rows_emitted == 4
    assert [m for _, m, _ in written_calls] == ["create", "append"]
    joined = pa.concat_tables(written_tables)
    assert joined["id"].to_pylist() == [1, 2, 3, 4]
    assert joined.schema.field("vector").type == pa.list_(pa.int8(), 16)
