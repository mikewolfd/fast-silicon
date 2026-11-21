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


def test_extract_parquet_dedup_within_window_and_utf8_token_sort_cpu(tmp_path: Path) -> None:
    import json
    from types import SimpleNamespace
    import fast_shared.pipeline_llm as pllm

    schema = {
        "type": "object",
        "properties": {"tag": {"type": "string"}, "byte_len": {"type": "integer"}},
        "required": ["tag", "byte_len"],
    }

    class _RecordingLlmEngine:
        default_max_b_slots = 32

        def __init__(self, *, with_token_counter: bool = False) -> None:
            self.seen_batches: list[list[str]] = []
            self.with_token_counter = with_token_counter
            if with_token_counter:
                self.count_tokens_batch = self._count_tokens_batch  # type: ignore[attr-defined]

        def _count_tokens_batch(self, prompts: list[str]) -> list[int]:
            # Custom deterministic token count distinct from byte/char length
            return [1000 if p.startswith("TOK_FIRST:") else len(p) for p in prompts]

        def generate_batch(self, prompts: list[str], **_kwargs):
            self.seen_batches.append(list(prompts))
            items = []
            for p in prompts:
                b_len = len(p.encode("utf-8"))
                txt = json.dumps({"tag": p[:12], "byte_len": b_len})
                items.append(
                    SimpleNamespace(
                        text=txt,
                        parsed=None,
                        valid_json=True,
                        valid_schema=True,
                        finish_reason="stop",
                        prompt_tokens=max(1, b_len // 3),
                        completion_tokens=8,
                        telemetry={
                            "mean_h_pre": 0.25 if "dup_a" in p else 0.05,
                            "mean_neg_log_z_valid": 0.40 if "dup_a" in p else 0.01,
                            "mean_top2_margin": 1.2 if "dup_a" in p else 5.5,
                        },
                    )
                )
            return SimpleNamespace(
                items=items,
                stats={
                    "valid_json_count": len(items),
                    "valid_schema_count": len(items),
                    "prompt_tokens": sum(it.prompt_tokens for it in items),
                    "completion_tokens": sum(it.completion_tokens for it in items),
                    "decode_sec": 0.02,
                },
            )

    # 8 rows: 1 empty + 7 active with exact duplicates ("dup_a" x3, "漢字漢字漢字漢字" x2, "ascii_10ch" x1, "TOK_FIRST:z" x1)
    # Note: "漢字漢字漢字漢字" is 8 chars (24 UTF-8 bytes), longer in UTF-8 bytes than 10-char ASCII "ascii_10ch" (10 bytes)!
    rows = [
        {"id": 0, "body_text": "ascii_10ch"},
        {"id": 1, "body_text": "dup_a_record_payload"},
        {"id": 2, "body_text": "   "},
        {"id": 3, "body_text": "漢字漢字漢字漢字"},
        {"id": 4, "body_text": "dup_a_record_payload"},
        {"id": 5, "body_text": "TOK_FIRST:z"},
        {"id": 6, "body_text": "漢字漢字漢字漢字"},
        {"id": 7, "body_text": "dup_a_record_payload"},
    ]
    in_pq = tmp_path / "dedup_in.parquet"
    pq.write_table(pa.Table.from_pylist(rows), in_pq)

    eng_dedup = _RecordingLlmEngine(with_token_counter=False)
    out_dedup = tmp_path / "out_dedup.parquet"
    sum_dedup = pllm.extract_parquet(
        eng_dedup,
        in_pq,
        out_dedup,
        system_prompt="sys",
        json_schema=schema,
        temperature=0.0,
        dedup_within_window=True,
        emit_extractconf=True,
    )

    eng_nodedup = _RecordingLlmEngine(with_token_counter=False)
    out_nodedup = tmp_path / "out_nodedup.parquet"
    sum_nodedup = pllm.extract_parquet(
        eng_nodedup,
        in_pq,
        out_nodedup,
        system_prompt="sys",
        json_schema=schema,
        temperature=0.0,
        dedup_within_window=False,
        emit_extractconf=True,
    )

    # 1. Dedup runs only the 4 unique non-empty prompts, sorted by UTF-8 byte length descending ("漢字..." 24B > "dup_a..." 20B > "TOK_FIRST:z" 11B > "ascii_10ch" 10B)!
    assert len(eng_dedup.seen_batches) == 1
    sent_dedup = eng_dedup.seen_batches[0]
    assert len(sent_dedup) == 4
    assert [len(s.encode("utf-8")) for s in sent_dedup] == sorted(
        [len(s.encode("utf-8")) for s in sent_dedup], reverse=True
    )
    assert sent_dedup[0] == "漢字漢字漢字漢字"

    # 2. Non-dedup sends all 7 active prompts
    assert len(eng_nodedup.seen_batches[0]) == 7

    # 3. Exact 100% parity on row order, counts, metrics, response texts, and ExtractConf risk scores!
    assert sum_dedup.docs_processed == sum_nodedup.docs_processed == 8
    assert sum_dedup.valid_json == sum_nodedup.valid_json == 7
    assert sum_dedup.valid_schema == sum_nodedup.valid_schema == 7
    assert sum_dedup.prompt_tokens == sum_nodedup.prompt_tokens
    assert sum_dedup.completion_tokens == sum_nodedup.completion_tokens

    tbl_d = pq.read_table(out_dedup)
    tbl_n = pq.read_table(out_nodedup)
    for col in ("id", "response_text", "valid_json", "valid_schema", "finish_reason", "prompt_tokens", "completion_tokens"):
        assert tbl_d.column(col).to_pylist() == tbl_n.column(col).to_pylist()
    assert np.allclose(tbl_d.column("extractconf_risk").to_numpy(), tbl_n.column("extractconf_risk").to_numpy(), atol=1e-7)

    # 4. Does NOT dedup when temperature > 0.0!
    eng_temp = _RecordingLlmEngine(with_token_counter=False)
    pllm.extract_parquet(
        eng_temp,
        in_pq,
        tmp_path / "out_temp.parquet",
        system_prompt="sys",
        json_schema=schema,
        temperature=0.2,
        dedup_within_window=True,
    )
    assert len(eng_temp.seen_batches[0]) == 7

    # 5. Uses `engine.count_tokens_batch` when callable!
    eng_toksort = _RecordingLlmEngine(with_token_counter=True)
    pllm.extract_parquet(
        eng_toksort,
        in_pq,
        tmp_path / "out_toksort.parquet",
        system_prompt="sys",
        json_schema=schema,
        temperature=0.0,
        dedup_within_window=True,
    )
    assert eng_toksort.seen_batches[0][0] == "TOK_FIRST:z"


def test_multichunk_drop_text_col_preproject_and_dict_encode_decode_schema_parity_cpu(tmp_path: Path) -> None:
    import fast_shared.parquet_pipeline as pp

    tok = _make_bpe_tokenizer(tmp_path)
    eng = _LengthSensitiveEngine(tok)

    # Batch 1 expands (K > 1 chunks per row -> dictionary-encoded in RAM); batch 2 does NOT expand (1 chunk/row -> tests _align_dict_meta_parts!)
    long_body = ("alpha beta " * 120).strip()
    short_body = "alpha beta"
    in_schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("source_uri", pa.large_string()),
            pa.field("page_offsets", pa.list_(pa.int32())),
            pa.field("body", pa.string()),
        ]
    )
    in_tbl = pa.Table.from_arrays(
        [
            pa.array(["d-0", "d-1", "d-2", "d-3"], type=pa.string()),
            pa.array(["uri://long-0", "uri://long-1", "uri://short-2", "uri://short-3"], type=pa.large_string()),
            pa.array([[0, 50], [0, 50], [0], [0]], type=pa.list_(pa.int32())),
            pa.array([long_body, long_body, short_body, short_body], type=pa.string()),
        ],
        schema=in_schema,
    )
    in_pq = tmp_path / "multichunk_in.parquet"
    pq.write_table(in_tbl, in_pq, row_group_size=2)

    # 1. Direct _prepare_batch_rows inspection: drops `body` and `page_offsets` BEFORE .take, and dictionary-encodes strings in RAM!
    rb0 = next(pq.ParquetFile(in_pq).iter_batches(batch_size=2))
    prep_meta, prep_toks = pp._prepare_batch_rows(
        rb0,
        tokenizer=tok,
        text_col="body",
        keep_cols=None,
        drop_text_col=True,
        ocr_pages_only=False,
        per_ocr_page=False,
        chunk=True,
        max_body_tokens=None,
        overlap=2,
        max_encode_length=24,
        max_chunk_tokens=24,
        skip_empty=False,
        include_chunk_text=False,
        prefix=None,
        prompt_name=None,
        add_query_prefix=False,
        add_document_prefix=True,
    )
    assert len(prep_toks) > rb0.num_rows
    assert "body" not in prep_meta.schema.names
    assert "page_offsets" not in prep_meta.schema.names
    assert pa.types.is_dictionary(prep_meta.schema.field("doc_id").type)
    assert pa.types.is_dictionary(prep_meta.schema.field("source_uri").type)

    # 2. End-to-end single-file AND sharded outputs decode back to exact `pa.string()` / `pa.large_string()` with 0 leaked internal metadata!
    out_single = tmp_path / "multichunk_single.parquet"
    out_shards = tmp_path / "multichunk_shards"
    sum_single = embed_parquet(
        eng,
        in_pq,
        out_single,
        text_col="body",
        drop_text_col=True,
        chunk=True,
        max_chunk_tokens=24,
        overlap=2,
        batch_rows=2,
        dtype="int8",
        mrl_dim=16,
        progress_every=0,
    )
    sum_shards = embed_parquet(
        eng,
        in_pq,
        out_shards,
        text_col="body",
        drop_text_col=True,
        chunk=True,
        max_chunk_tokens=24,
        overlap=2,
        batch_rows=2,
        shard_rows=5,
        dtype="int8",
        mrl_dim=16,
        progress_every=0,
    )
    assert sum_single.rows_emitted == sum_shards.rows_emitted
    assert sum_single.rows_emitted > 4

    t_single = pq.read_table(out_single)
    t_shards = pa.concat_tables([pq.read_table(p) for p in sorted(out_shards.glob("part-*.parquet"))])
    for tbl_out in (t_single, t_shards):
        assert "body" not in tbl_out.schema.names
        assert "page_offsets" not in tbl_out.schema.names
        assert tbl_out.schema.field("doc_id").type == pa.string()
        assert tbl_out.schema.field("source_uri").type == pa.large_string()
        assert not (tbl_out.schema.field("doc_id").metadata and b"_fast_dict_orig" in tbl_out.schema.field("doc_id").metadata)
        assert not (tbl_out.schema.field("source_uri").metadata and b"_fast_dict_orig" in tbl_out.schema.field("source_uri").metadata)
    assert t_single.column("doc_id").to_pylist() == t_shards.column("doc_id").to_pylist()
    assert t_single.column("source_uri").to_pylist() == t_shards.column("source_uri").to_pylist()
    assert t_single.column("doc_id").to_pylist()[-2:] == ["d-2", "d-3"]


