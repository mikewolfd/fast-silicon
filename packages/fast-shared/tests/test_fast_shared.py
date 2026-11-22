from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_shared import (
    BaseFastTokenizer,
    EmbeddingHTTPServer,
    FaultInjectionCase,
    FaultInjectionResult,
    MappedEmbedding,
    UnifiedOpenAIServer,
    embed_parquet,
    fused_geglu_split,
    make_mpp_matmul_kernel,
    mpp_fused_geglu_bf16,
    mpp_linear_bf16,
    native_include_dir,
    resolve_embedding_dim,
    resolve_home_dir,
    run_fault_injection_suite,
    set_mlx_cache_limit,
    slice_ocr_pages_utf8,
    tile_aligned_slices,
    truncate_and_normalize,
)


class _DummyTokenizer:
    def resolve_prefix(self, **kwargs):
        return kwargs.get("prefix") or ""

    def encode_batch(self, texts, *, prefix="", max_length=512):
        return [[2, 10 + i, 1] for i, _ in enumerate(texts)]

    def encode(self, text, *, prefix="", max_length=512):
        return [2, 10, 1]

    def chunk_text(self, text, *, max_body_tokens=510, overlap=51, prefix="", max_chunk_tokens=512):
        mid = len(text) // 2 or len(text)
        return [([2, 10, 1], 0, mid)]


class _DummyEngine:
    dim = 16
    model_name = "dummy-emb"
    default_mode = "gpu"
    ane_available = False
    max_seq_len = 512
    ane_query_max_len = 512

    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def encode_query(self, toks):
        v = np.arange(1, 17, dtype=np.float32) + float(len(toks))
        return v / np.linalg.norm(v)

    def encode_corpus(self, tok_lists, mode=None):
        out = np.zeros((len(tok_lists), self.dim), dtype=np.float32)
        for i, toks in enumerate(tok_lists):
            v = np.arange(1, self.dim + 1, dtype=np.float32) + float(sum(toks))
            out[i] = v / np.linalg.norm(v)
        return out


def test_native_include_and_paths(tmp_path: Path, monkeypatch) -> None:
    hdr = native_include_dir() / "fast_shared_apple.hpp"
    coreml_hdr = native_include_dir() / "fast_shared_coreml.hpp"
    assert hdr.exists()
    assert coreml_hdr.exists()

    monkeypatch.setenv("TEST_SHARED_HOME", str(tmp_path))
    assert resolve_home_dir(__file__, env_var="TEST_SHARED_HOME", cache_name="x") == tmp_path.resolve()


def _gpu_validation_lock():
    from contextlib import contextmanager
    import fcntl
    import os

    @contextmanager
    def _ctx():
        if os.environ.get("FAST_SILICON_GPU_LOCK_HELD") == "1":
            yield
            return
        with open("/tmp/m4_gpu_validation.lock", "w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            os.environ["FAST_SILICON_GPU_LOCK_HELD"] = "1"
            try:
                yield
            finally:
                os.environ.pop("FAST_SILICON_GPU_LOCK_HELD", None)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    return _ctx()


def test_mpp_matmul_kernel_compile_and_run() -> None:
    with _gpu_validation_lock():
        mx.set_default_device(mx.gpu)
        k = make_mpp_matmul_kernel("shared_test_mpp64", epilogue="none", tm=64, tn=64, sgs=4)
        a = mx.ones((64, 64), dtype=mx.float16)
        b = mx.ones((64, 64), dtype=mx.float16)
        c = k(inputs=[a, b], grid=(128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(64, 64)], output_dtypes=[mx.float16])[0]
        mx.eval(c)
        assert np.allclose(np.array(c), 64.0)

        rng = np.random.default_rng(7)
        a_np = rng.normal(0.0, 0.35, size=(64, 64)).astype(np.float16)
        b_gu_np = rng.normal(0.0, 0.35, size=(128, 64)).astype(np.float16)
        b_pli_np = rng.normal(0.0, 0.35, size=(64, 64)).astype(np.float16)
        pli_np = rng.normal(0.0, 0.5, size=(64, 64)).astype(np.float16)

        a_f16 = mx.array(a_np)
        b_gu_f16 = mx.array(b_gu_np)
        b_pli_f16 = mx.array(b_pli_np)
        pli_f16 = mx.array(pli_np)

        k_geglu_tanh = make_mpp_matmul_kernel("shared_test_geglu_tanh", epilogue="geglu_tanh", tm=64, tn=64, sgs=4)
        k_geglu_erf = make_mpp_matmul_kernel("shared_test_geglu_erf", epilogue="geglu", tm=64, tn=64, sgs=4)
        k_pli_tanh = make_mpp_matmul_kernel("shared_test_pli_tanh", epilogue="gelu_tanh_mul_pli", tm=64, tn=64, sgs=4)

        out_geglu_tanh = k_geglu_tanh(
            inputs=[a_f16, b_gu_f16],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(64, 64)],
            output_dtypes=[mx.float16],
        )[0]
        out_geglu_erf = k_geglu_erf(
            inputs=[a_f16, b_gu_f16],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(64, 64)],
            output_dtypes=[mx.float16],
        )[0]
        out_pli_tanh = k_pli_tanh(
            inputs=[a_f16, b_pli_f16, pli_f16],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(64, 64)],
            output_dtypes=[mx.float16],
        )[0]
        mx.eval(out_geglu_tanh, out_geglu_erf, out_pli_tanh)

        # Reference float32 gelu_pytorch_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        g_ref = a_np.astype(np.float32) @ b_gu_np[:64].astype(np.float32).T
        u_ref = a_np.astype(np.float32) @ b_gu_np[64:].astype(np.float32).T
        sqrt_2_over_pi = np.float32(np.sqrt(2.0 / np.pi))
        geglu_tanh_ref = 0.5 * g_ref * (1.0 + np.tanh(sqrt_2_over_pi * (g_ref + 0.044715 * (g_ref**3)))) * u_ref

        g_pli_ref = a_np.astype(np.float32) @ b_pli_np.astype(np.float32).T
        pli_tanh_ref = (
            0.5
            * g_pli_ref
            * (1.0 + np.tanh(sqrt_2_over_pi * (g_pli_ref + 0.044715 * (g_pli_ref**3))))
            * pli_np.astype(np.float32)
        )

        assert np.allclose(np.array(out_geglu_tanh, dtype=np.float32), geglu_tanh_ref, atol=4e-3, rtol=4e-3)
        assert np.allclose(np.array(out_pli_tanh, dtype=np.float32), pli_tanh_ref, atol=4e-3, rtol=4e-3)
        # Verify geglu_tanh is closer to tanh reference than exact_erf geglu
        err_tanh = float(np.max(np.abs(np.array(out_geglu_tanh, dtype=np.float32) - geglu_tanh_ref)))
        err_erf = float(np.max(np.abs(np.array(out_geglu_erf, dtype=np.float32) - geglu_tanh_ref)))
        assert err_tanh < err_erf

        a_bf = mx.ones((128, 64), dtype=mx.bfloat16) * 0.5
        wg = mx.ones((64, 64), dtype=mx.bfloat16)
        wu = mx.ones((64, 64), dtype=mx.bfloat16)
        wd = mx.ones((32, 64), dtype=mx.bfloat16)
        ge = mpp_fused_geglu_bf16(a_bf, wg, wu)
        mm = mpp_linear_bf16(ge, wd, tm=32, tn=32)
        sp = fused_geglu_split(mx.concatenate([a_bf @ wg.T, a_bf @ wu.T], axis=-1))
        mx.eval(ge, mm, sp)
        assert ge.shape == (128, 64) and ge.dtype == mx.bfloat16
        assert mm.shape == (128, 32) and mm.dtype == mx.bfloat16
        assert np.allclose(np.array(ge.astype(mx.float32)), np.array(sp.astype(mx.float32)), atol=0.5, rtol=1e-2)


def test_slice_ocr_pages_utf8() -> None:
    p1 = "Intro — header page."
    p2 = "OCR scanned exhibit text."
    p3 = "Native appendix."
    full = p1 + p2 + p3
    offsets = [0, len(p1.encode("utf-8")), len((p1 + p2).encode("utf-8"))]
    assert slice_ocr_pages_utf8(full, offsets, [2]) == p2
    assert slice_ocr_pages_utf8(full, offsets, [1, 3]) == f"{p1}\n\n{p3}"


def test_base_fast_tokenizer_and_chunking(tmp_path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "q:": 4, "hello": 5, "world": 6, "foo": 7, "bar": 8}
    raw_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw_tok.pre_tokenizer = WhitespaceSplit()
    raw_tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 1)],
    )
    tok_path = tmp_path / "tokenizer.json"
    raw_tok.save(str(tok_path))

    ftok = BaseFastTokenizer(
        tok_path,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        query_prefix="q: ",
        prompts={"q": "q: "},
    )
    ids = ftok.encode("hello world", add_query_prefix=True)
    assert ids == [2, 4, 5, 6, 1]
    assert ftok.encode("hello world", prompt_name="q") == ids

    chunks = ftok.chunk_text("hello world foo bar hello world", max_body_tokens=3, overlap=1, prefix="q: ", max_chunk_tokens=6)
    assert len(chunks) >= 2
    for c_ids, c0, c1 in chunks:
        assert len(c_ids) <= 6
        assert c_ids[0] == 2 and c_ids[1] == 4 and c_ids[-1] == 1
        assert 0 <= c0 < c1


@pytest.fixture
def bpe_tokenizer(tmp_path: Path) -> BaseFastTokenizer:
    """A real byte-level BPE whose trailing prefix space merges with hello."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing

    vocab = {token: i for i, token in enumerate(["<pad>", "<eos>", "<bos>", "<unk>", *sorted(ByteLevel.alphabet())])}
    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"), ("Ġ", "hello")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    raw_tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    raw_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)]
    )
    path = tmp_path / "bpe-tokenizer.json"
    raw_tok.save(str(path))
    return BaseFastTokenizer(
        path, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", document_prefix="q: "
    )


@pytest.mark.parametrize("overlap", [0, 2])
@pytest.mark.parametrize("text", ["", "hello", "  hello\nhello  ", "\n\t  ", "\n é🧪漢字 hello\n é🧪漢字 \t"])
def test_chunk_text_matches_native_encoding_and_covers_source(bpe_tokenizer: BaseFastTokenizer, text: str, overlap: int) -> None:
    tok = bpe_tokenizer
    # This fixture detects the old separate-prefix/body concatenation bug.
    separate = [tok.bos_id, *tok._tok.encode("q: ", add_special_tokens=False).ids,
                *tok._tok.encode("hello", add_special_tokens=False).ids, tok.eos_id]
    assert separate != tok.encode("hello", add_document_prefix=True)
    chunks = tok.chunk_text(text, max_chunk_tokens=12, overlap=overlap, add_document_prefix=True)
    assert chunks[0][1] == 0
    assert chunks[-1][2] == len(text)
    previous_end = 0
    for ids, start, end in chunks:
        assert 0 <= start <= previous_end <= len(text)
        assert start <= end <= len(text)
        assert ids == tok.encode(text[start:end], add_document_prefix=True, max_length=4096)
        assert len(ids) <= 12
        previous_end = max(previous_end, end)
    if overlap == 0:
        assert "".join(text[start:end] for _, start, end in chunks) == text


def test_chunk_text_body_overlap_and_zero_overlap_coverage(bpe_tokenizer: BaseFastTokenizer) -> None:
    text = "abcdefghijklmno"
    chunks = bpe_tokenizer.chunk_text(text, max_chunk_tokens=8, overlap=2)
    assert [(start, end) for _, start, end in chunks] == [(0, 6), (4, 10), (8, 14), (12, 15)]
    text = "  hello \nabcdefghijklmno\n  "
    chunks = bpe_tokenizer.chunk_text(text, max_chunk_tokens=8, overlap=0, max_body_tokens=100)
    assert "".join(text[start:end] for _, start, end in chunks) == text
    assert all(ids == bpe_tokenizer.encode(text[start:end]) for ids, start, end in chunks)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"max_chunk_tokens": 0}, "max_chunk_tokens must be positive"),
    ({"max_body_tokens": 0}, "max_body_tokens must be positive"),
    ({"max_body_tokens": -2}, "max_body_tokens must be positive"),
    ({"overlap": -1}, "overlap must be nonnegative"),
    ({"max_body_tokens": 2, "overlap": 2}, "smaller than the usable body window"),
    ({"max_chunk_tokens": 6, "prefix": "abcdef", "overlap": 0}, "leave room for body tokens"),
])
def test_chunk_text_rejects_invalid_windows(bpe_tokenizer: BaseFastTokenizer, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bpe_tokenizer.chunk_text("hello world", **kwargs)


def test_chunk_text_refuses_unfittable_unicode_span(bpe_tokenizer: BaseFastTokenizer) -> None:
    with pytest.raises(ValueError, match="source character or token span cannot fit"):
        bpe_tokenizer.chunk_text("🧪", prefix="q: ", max_chunk_tokens=6, overlap=0)


def test_chunk_text_rejects_overlap_that_prevents_character_progress(bpe_tokenizer: BaseFastTokenizer) -> None:
    with pytest.raises(ValueError, match="overlap leaves no forward progress"):
        bpe_tokenizer.chunk_text("é🧪漢字", prefix="q: ", max_chunk_tokens=12, overlap=6)


@pytest.mark.parametrize(("batch_rows", "shard_rows", "single_file"), [(1, 4, True), (3, 8, True), (1, 8, False), (3, 4, False)])
def test_parquet_caps_expanded_chunks_and_preserves_values(
    tmp_path: Path, bpe_tokenizer: BaseFastTokenizer, batch_rows: int, shard_rows: int, single_file: bool
) -> None:
    class RecordingEngine(_DummyEngine):
        supports_dimensions = True

        def __init__(self):
            self.tokenizer = bpe_tokenizer
            self.calls: list[list[list[int]]] = []

        def encode_corpus(self, tok_lists, mode=None):
            self.calls.append(tok_lists)
            return super().encode_corpus(tok_lists, mode=mode)

    rows = [{"id": i, "tag": f"source-{i}", "body": f"  abcdefghijklmno {i}\n "} for i in range(3)]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    expected_rows = []
    expected_toks = []
    for row in rows:
        for index, (ids, start, end) in enumerate(bpe_tokenizer.chunk_text(
            row["body"], max_chunk_tokens=12, max_body_tokens=5, overlap=1, add_document_prefix=True
        )):
            expected_rows.append({"id": row["id"], "tag": row["tag"], "chunk_index": index,
                                  "char_start": start, "char_end": end, "token_count": len(ids),
                                  "chunk_text": row["body"][start:end]})
            expected_toks.append(ids)
    assert len(expected_rows) > 3
    destination = tmp_path / ("out.parquet" if single_file else "out")
    engine = RecordingEngine()
    options = dict(keep_cols=["id", "tag"], chunk=True, include_chunk_text=True,
                   max_chunk_tokens=12, max_body_tokens=5, overlap=1, add_document_prefix=True,
                   batch_rows=batch_rows, encode_batch_rows=3, shard_rows=shard_rows, dimensions=8, progress_every=0)
    summary = embed_parquet(engine, source, destination, **options)
    files = [destination] if single_file else sorted(destination.glob("part-*.parquet"))
    result = pa.concat_tables([pq.read_table(path) for path in files])
    assert result.drop(["vector"]).to_pylist() == expected_rows
    assert summary.rows_read == len(rows)
    assert summary.rows_emitted == len(expected_rows)
    assert summary.total_tokens == sum(map(len, expected_toks))
    assert max(map(len, engine.calls)) <= 3
    assert [ids for call in engine.calls for ids in call] == expected_toks
    vectors = np.asarray(result["vector"].to_pylist(), dtype=np.float32)
    assert result.schema.field("vector").type == pa.list_(pa.float32(), 8)
    assert np.allclose(vectors, truncate_and_normalize(_DummyEngine().encode_corpus(expected_toks), 8), atol=1e-6)
    if not single_file:
        assert [pq.ParquetFile(path).metadata.num_rows for path in files[:-1]] == [shard_rows] * (len(files) - 1)
        resumed = RecordingEngine()
        embed_parquet(resumed, source, destination, resume=True, **options)
        assert resumed.calls == []
        assert pa.concat_tables([pq.read_table(path) for path in files]).equals(result)


def test_parquet_per_ocr_page_preserves_original_page_offsets(tmp_path: Path, bpe_tokenizer: BaseFastTokenizer) -> None:
    pages = ["Native page", "\n é🧪 hello \n", " \n\t "]
    source = tmp_path / "pages.parquet"
    pq.write_table(pa.table({"id": [7], "body": ["".join(pages)],
                            "page_offsets": [[0, len(pages[0].encode()), len("".join(pages[:2]).encode())]],
                            "ocr_pages": [[2, 3]]}), source)
    engine = _DummyEngine()
    engine.tokenizer = bpe_tokenizer
    destination = tmp_path / "pages-out.parquet"
    embed_parquet(engine, source, destination, per_ocr_page=True, chunk=True,
                  include_chunk_text=True, skip_empty=True, max_chunk_tokens=12,
                  overlap=0, keep_cols=["id"], encode_batch_rows=2, progress_every=0)
    records = pq.read_table(destination).drop(["vector"]).to_pylist()
    assert all(record["ocr_page_number"] == 2 and record["id"] == 7 for record in records)
    assert records[0]["char_start"] == 0 and records[-1]["char_end"] == len(pages[1])
    assert "".join(record["chunk_text"] for record in records) == pages[1]
    for record in records:
        assert record["chunk_text"] == pages[1][record["char_start"]:record["char_end"]]
        assert record["token_count"] == len(bpe_tokenizer.encode(record["chunk_text"]))


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"batch_rows": 0}, "batch_rows must be positive"),
    ({"encode_batch_rows": -1}, "encode_batch_rows must be positive"),
    ({"shard_rows": 0}, "shard_rows must be positive"),
    ({"max_length": 0}, "max_length must be positive"),
    ({"max_length": 513}, "max_length must be positive"),
    ({"chunk": True, "max_chunk_tokens": 513}, "max_chunk_tokens must be positive"),
    ({"max_chunk_tokens": 0}, "max_chunk_tokens must be positive"),
])
def test_parquet_rejects_invalid_batch_and_sequence_limits(tmp_path: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        embed_parquet(_DummyEngine(), tmp_path / "missing.parquet", tmp_path / "out", **kwargs)


def test_parquet_cli_controls_and_forwarding(monkeypatch, tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace
    import fast_shared.cli_common as cli

    parser = argparse.ArgumentParser()
    cli.add_parquet_arguments(parser, mode_choices=["gpu"], default_mode="gpu")
    defaults = parser.parse_args(["--input", "in", "--output", "out"])
    assert (defaults.max_body_tokens, defaults.max_chunk_tokens, defaults.encode_batch_rows, defaults.overlap) == (None, 512, 4096, 51)
    args = parser.parse_args(["--input", "in", "--output", "out", "--chunk", "--batch-rows", "2",
                              "--encode-batch-rows", "3", "--shard-rows", "4", "--max-chunk-tokens", "128",
                              "--chunk-overlap", "7", "--max-body-tokens", "64"])
    captured = {}
    engine = object()

    def fake_embed(given_engine, source, destination, **kwargs):
        assert given_engine is engine and source == "in" and destination == "out"
        captured.update(kwargs)
        return SimpleNamespace(input_files=1, rows_read=2, rows_emitted=3, total_tokens=4,
                               elapsed_seconds=1, tokens_per_sec=4, rows_per_sec=3, output_path=tmp_path)

    monkeypatch.setattr(cli, "embed_parquet", fake_embed)
    assert cli.run_parquet_command(args, lambda _: engine) == 0
    assert {name: captured[name] for name in ("batch_rows", "encode_batch_rows", "shard_rows", "max_chunk_tokens", "overlap", "max_body_tokens")} == {
        "batch_rows": 2, "encode_batch_rows": 3, "shard_rows": 4, "max_chunk_tokens": 128, "overlap": 7, "max_body_tokens": 64
    }


def test_parquet_and_http_server_generic_dim(tmp_path: Path) -> None:
    eng = _DummyEngine()
    in_p = tmp_path / "in.parquet"
    rows = [{"id": i, "body": f"Doc number {i} content"} for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), in_p)

    out_p = tmp_path / "out.parquet"
    summary = embed_parquet(eng, in_p, out_p, text_col="body", dtype="float16", progress_every=0)
    assert summary.rows_emitted == 6
    tbl = pq.read_table(out_p)
    assert tbl.schema.field("vector").type == pa.list_(pa.float16(), 16)

    with EmbeddingHTTPServer(eng, host="127.0.0.1", port=0) as srv:
        base = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            health = json.loads(r.read().decode())
            assert health["dim"] == 16 and health["model"] == "dummy-emb"
        req = urllib.request.Request(
            f"{base}/v1/embeddings",
            data=json.dumps({"input": ["alpha", "beta"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            payload = json.loads(r.read().decode())
            assert len(payload["data"]) == 2
            assert len(payload["data"][0]["embedding"]) == 16
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            models = json.loads(r.read().decode())
            assert [m["id"] for m in models["data"]] == ["dummy-emb"]


class _DummyGenItem:
    def __init__(self, text: str, prompt_tokens: int = 5, completion_tokens: int = 3, finish_reason: str = "stop") -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason


class _DummyGenResult:
    def __init__(self, items, stats):
        self.items = items
        self.stats = stats


class _DummyChatEngine:
    model_name = "dummy-gemma4"
    ane_available = False
    default_mode = "gpu"

    def generate_batch(self, prompts, **kwargs):
        schema = kwargs.get("json_schema")
        items = []
        for p in prompts:
            label = p[-1]["content"] if isinstance(p, list) else str(p)
            txt = json.dumps({"echo": label, "has_schema": schema is not None})
            items.append(_DummyGenItem(txt))
        return _DummyGenResult(items, {"docs": len(prompts)})


def test_unified_openai_server_routes_and_sse() -> None:
    import urllib.error

    emb = _DummyEngine()
    chat = _DummyChatEngine()
    with UnifiedOpenAIServer(
        embedding_engines={"BAAI/bge-small-en-v1.5": emb},
        chat_engines={"google/gemma-4-E2B-it": chat},
        host="127.0.0.1",
        port=0,
    ) as srv:
        base = f"http://{srv.host}:{srv.port}"

        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            h = json.loads(r.read().decode())
            assert h["status"] == "ok"
            assert "BAAI/bge-small-en-v1.5" in h["embedding_models"]
            assert "google/gemma-4-E2B-it" in h["chat_models"]

        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            m = json.loads(r.read().decode())
            ids = {row["id"] for row in m["data"]}
            assert ids == {"BAAI/bge-small-en-v1.5", "google/gemma-4-E2B-it"}
        with urllib.request.urlopen(f"{base}/v1/models/google%2Fgemma-4-E2B-it", timeout=5) as r:
            one = json.loads(r.read().decode())
            assert one["id"] == "google/gemma-4-E2B-it"

        # Embedding route (float + base64)
        emb_req = urllib.request.Request(
            f"{base}/v1/embeddings",
            data=json.dumps({"model": "BAAI/bge-small-en-v1.5", "input": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(emb_req, timeout=5) as r:
            emb_res = json.loads(r.read().decode())
            assert len(emb_res["data"][0]["embedding"]) == 16

        import base64
        emb_b64_req = urllib.request.Request(
            f"{base}/v1/embeddings",
            data=json.dumps({"model": "BAAI/bge-small-en-v1.5", "input": "hello", "encoding_format": "base64"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(emb_b64_req, timeout=5) as r:
            b64_res = json.loads(r.read().decode())
            vec_b64 = np.frombuffer(base64.b64decode(b64_res["data"][0]["embedding"]), dtype="<f4")
            assert vec_b64.shape == (16,)

        # Chat JSON schema + extra_body route
        chat_req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({
                "model": "google/gemma-4-E2B-it",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(chat_req, timeout=5) as r:
            chat_res = json.loads(r.read().decode())
            parsed = json.loads(chat_res["choices"][0]["message"]["content"])
            assert parsed == {"echo": "hi", "has_schema": True}

        # Streaming SSE completion route
        sse_req = urllib.request.Request(
            f"{base}/v1/completions",
            data=json.dumps({
                "model": "google/gemma-4-E2B-it",
                "prompt": ["p1", "p2"],
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(sse_req, timeout=5) as r:
            raw_sse = r.read().decode()
            assert "data: [DONE]" in raw_sse
            first_line = raw_sse.splitlines()[0]
            ev = json.loads(first_line.removeprefix("data: "))
            assert len(ev["choices"]) == 2
            assert ev["usage"]["total_tokens"] == 16

        # Unknown model returns 404
        bad_req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"model": "missing", "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req, timeout=5)
            raise AssertionError("Expected 404 for unknown model")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


def test_matryoshka_dimensions_and_server_cpu(tmp_path: Path) -> None:
    import base64
    import urllib.error

    rng = np.random.default_rng(123)
    raw = rng.standard_normal((5, 16)).astype(np.float32)
    full_normed = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    via_full = truncate_and_normalize(full_normed, 8)
    direct = raw[:, :8] / np.linalg.norm(raw[:, :8], axis=1, keepdims=True)
    assert np.allclose(via_full, direct, atol=1e-6)
    assert np.allclose(np.linalg.norm(via_full, axis=1), 1.0, atol=1e-6)
    assert truncate_and_normalize(raw[0], 4).shape == (4,)
    assert truncate_and_normalize(np.empty((0, 16), dtype=np.float32), 8).shape == (0, 8)

    assert resolve_embedding_dim(None, max_dim=16, default_dim=8) == 8
    assert resolve_embedding_dim(12, max_dim=16) == 12
    for bad in (0, -4, 17, True, 8.5, "8"):
        with pytest.raises(ValueError):
            resolve_embedding_dim(bad, max_dim=16)
    with pytest.raises(ValueError):
        resolve_embedding_dim(8, max_dim=16, supports_dimensions=False)

    class _MrlDummyEngine(_DummyEngine):
        max_dim = 16
        supports_dimensions = True
        mrl_dimensions = (4, 8, 16)

    mrl_eng = _MrlDummyEngine()
    non_mrl_eng = _DummyEngine()

    in_p = tmp_path / "mrl_in.parquet"
    out_p = tmp_path / "mrl_out.parquet"
    pq.write_table(pa.Table.from_pylist([{"body": "one"}, {"body": "two"}]), in_p)
    embed_parquet(mrl_eng, in_p, out_p, text_col="body", dtype="float16", dimensions=8, progress_every=0)
    tbl = pq.read_table(out_p)
    assert tbl.schema.field("vector").type == pa.list_(pa.float16(), 8)
    arr = np.asarray(tbl["vector"].to_pylist(), dtype=np.float32)
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=5e-3)

    with UnifiedOpenAIServer(
        embedding_engines={"mrl": mrl_eng, "fixed": non_mrl_eng},
        host="127.0.0.1",
        port=0,
    ) as srv:
        base = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            h = json.loads(r.read().decode())
            assert h["embedding_models"]["mrl"]["max_dim"] == 16
            assert h["embedding_models"]["mrl"]["mrl_dimensions"] == [4, 8, 16]

        for fmt in ("float", "base64"):
            req = urllib.request.Request(
                f"{base}/v1/embeddings",
                data=json.dumps({"model": "mrl", "input": ["a", "b"], "dimensions": 8, "encoding_format": fmt}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                body = json.loads(r.read().decode())
                for row in body["data"]:
                    vec = (
                        np.frombuffer(base64.b64decode(row["embedding"]), dtype="<f4")
                        if fmt == "base64"
                        else np.asarray(row["embedding"], dtype=np.float32)
                    )
                    assert vec.shape == (8,)
                    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

        for bad_payload in (
            {"model": "fixed", "input": "a", "dimensions": 8},
            {"model": "mrl", "input": "a", "dimensions": 32},
            {"model": "mrl", "input": [], "dimensions": -1},
        ):
            bad_req = urllib.request.Request(
                f"{base}/v1/embeddings",
                data=json.dumps(bad_payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(bad_req, timeout=5)
            assert exc.value.code == 400


def test_promoted_llm_server_pipeline_and_grammar_cpu(tmp_path: Path) -> None:
    import concurrent.futures
    import threading
    import time
    from types import SimpleNamespace

    from fast_shared import (
        ManagedHTTPServer,
        OpenAIChatAdapter,
        extract_jsonl,
        extract_parquet,
        make_chat_handler,
        slice_first_pages_utf8,
        validate_json_against_schema,
    )

    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "n": {"type": "integer", "minimum": 0}},
        "required": ["q"],
        "additionalProperties": False,
    }
    validate_json_against_schema({"q": "ok", "n": 2}, schema)
    with pytest.raises(ValueError):
        validate_json_against_schema({"q": "ok", "n": -1}, schema)

    p1, p2 = "Page1\n", "Page2\n"
    assert slice_first_pages_utf8(p1 + p2, [0, len(p1.encode()), len((p1 + p2).encode())], 1) == p1

    class _CoalesceEngine:
        model_name = "dummy-llm"
        default_mode = "auto"
        ane_available = False

        def __init__(self) -> None:
            self.calls: list[int] = []
            self._lock = threading.Lock()

        def generate_batch(self, prompts, **kwargs):
            with self._lock:
                self.calls.append(len(prompts))
            time.sleep(0.02)
            items = [
                SimpleNamespace(
                    text=json.dumps({"q": p[-1]["content"] if isinstance(p, list) else str(p)}),
                    reasoning="brief-cot" if kwargs.get("reasoning_effort") == "low" else None,
                    parsed={"q": "x"},
                    prompt_tokens=6,
                    completion_tokens=4,
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                )
                for p in prompts
            ]
            return SimpleNamespace(
                items=items,
                stats={
                    "docs": len(prompts),
                    "valid_json_count": len(items),
                    "valid_schema_count": len(items),
                    "prompt_tokens": 6 * len(items),
                    "completion_tokens": 4 * len(items),
                    "decode_sec": 0.05,
                },
            )

    eng = _CoalesceEngine()
    adapter = OpenAIChatAdapter(eng, owned_by="openai", extra_key="fast_gpt_oss", coalesce_ms=20.0, max_batch_size=16)
    with ManagedHTTPServer(make_chat_handler(adapter), host="127.0.0.1", port=0) as srv:
        base = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            h = json.loads(r.read().decode())
            assert h["model"] == "dummy-llm" and h["coalesce_ms"] == 20.0

        def _hit(i: int) -> dict:
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({
                    "messages": [{"role": "user", "content": f"m-{i}"}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}},
                    "extra_body": {"reasoning_effort": "low"},
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read().decode())

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            out = list(pool.map(_hit, range(4)))
        assert len(out) == 4
        assert out[0]["choices"][0]["message"]["reasoning"] == "brief-cot"
        assert "fast_gpt_oss" in out[0]
        assert len(eng.calls) < 4

    in_jsonl = tmp_path / "in.jsonl"
    out_jsonl = tmp_path / "out.jsonl"
    in_jsonl.write_text('{"id":1,"text":"hello"}\n{"id":2,"text":"world"}\n', encoding="utf-8")
    extract_jsonl(eng, in_jsonl, out_jsonl, system_prompt="sys", json_schema=schema)
    assert len(out_jsonl.read_text(encoding="utf-8").splitlines()) == 2

    in_pq = tmp_path / "in.parquet"
    out_dir = tmp_path / "shards"
    pq.write_table(pa.table({"id": [1, 2], "body_text": ["alpha", "beta"]}), in_pq)
    summary = extract_parquet(eng, in_pq, out_dir, system_prompt="sys", json_schema=schema, drop_text_col=True)
    assert summary.docs_processed == 2 and summary.valid_schema == 2
    resumed = extract_parquet(eng, in_pq, out_dir, system_prompt="sys", json_schema=schema, resume=True)
    assert resumed.docs_processed == 0


def test_promoted_grammar_helper_cpu() -> None:
    # Lazy llguidance verification (skipped automatically if optional [llm] extra not installed)
    llg = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.grammar import GrammarHelper

    assert llg is not None
    vocab = {"<|endoftext|>": 0, "{": 1, '"': 2, "a": 3, ":": 4, "1": 5, "}": 6}
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llg.LLTokenizer(bt.to_str(), eos_token=0)
    gh = GrammarHelper(
        ltok,
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False},
        whitespace_flexible=False,
        eos_token_id=0,
    )
    assert gh.upfront_ff == [1, 2, 3, 2, 4]
    m = gh.new_matcher()
    called = [0]

    def _lazy():
        called[0] += 1
        return np.zeros(len(vocab), dtype=np.float32)

    tok_id, masked = gh.pick_valid_token(m, _lazy, 5)
    assert tok_id == 5 and not masked and called[0] == 0

    tok_id2, masked2 = gh.pick_valid_token(m, lambda: np.array([0, 0, 0, 0, 0, 10.0, 0], dtype=np.float32), 6)
    assert tok_id2 == 5 and masked2 is True


def test_roofline_audit_bench_and_plugin_cpu(tmp_path: Path) -> None:
    import struct
    from fast_shared import (
        audit_safetensors_weights,
        check_system_quiescence,
        check_tensor_dtypes,
        compute_model_roofline,
        cos_sim,
        read_safetensors_headers,
    )

    # Synthetic safetensors file exercising header parsing + category/bit accounting
    st_header = json.dumps({
        "__metadata__": {"format": "mlx"},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 64], "data_offsets": [0, 12800]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "U32", "shape": [64, 8], "data_offsets": [12800, 14848]},
        "model.layers.0.self_attn.q_proj.scales": {"dtype": "BF16", "shape": [64, 1], "data_offsets": [14848, 14976]},
        "model.layers.0.mlp.experts.0.gate_up_proj.weight": {"dtype": "U32", "shape": [64, 8], "data_offsets": [14976, 17024]},
        "model.layers.0.mlp.experts.0.gate_up_proj.scales": {"dtype": "U8", "shape": [64, 2], "data_offsets": [17024, 17152]},
    }).encode("utf-8")
    st_path = tmp_path / "model.safetensors"
    with st_path.open("wb") as f:
        f.write(struct.pack("<Q", len(st_header)))
        f.write(st_header)
        f.write(b"\x00" * 17152)

    tensors = read_safetensors_headers(tmp_path)
    assert len(tensors) == 5
    anatomy = audit_safetensors_weights(tensors)
    assert anatomy["total_bytes"] == 17152
    assert "mlp_or_moe" in anatomy["effective_precision"]

    moe_cfg = {
        "num_hidden_layers": 24,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 201088,
        "tie_word_embeddings": False,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"] * 12,
    }
    roof = compute_model_roofline(moe_cfg, 11_180_000_000, gpu_tf=7.4, dram_gbs=200.0, ctx=2048)
    assert roof["kv_kib_per_tok"] == 24.0
    assert roof["decode"][0]["B"] == 1 and roof["decode"][0]["regime"] == "MEM-BW"
    assert roof["decode"][-1]["B"] == 256 and roof["decode"][-1]["regime"] == "COMPUTE"

    # Bench & dtype guard helpers
    assert abs(cos_sim(np.array([1.0, 2.0]), np.array([2.0, 4.0])) - 1.0) < 1e-9
    flags = check_tensor_dtypes([
        ("sinks_f32", mx.zeros((4,), dtype=mx.float32)),
        ("k_cache", mx.zeros((4,), dtype=mx.float32)),
        ("q_proj", mx.zeros((4,), dtype=mx.float16)),
    ])
    assert len(flags) == 1 and "k_cache" in flags[0]
    assert isinstance(check_system_quiescence(), list)

    # Plugin manifests & skill directory structure
    repo_root = Path(__file__).resolve().parents[1]
    c_manifest = json.loads((repo_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    x_manifest = json.loads((repo_root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    m_manifest = json.loads((repo_root / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
    assert c_manifest["skills"] == "./skills/" and x_manifest["skills"] == "./skills/"
    assert m_manifest["plugins"][0]["name"] == "apple-silicon-ml-optimizer"
    skill_dir = repo_root / "skills" / "apple-silicon-ml-optimizer"
    assert (skill_dir / "SKILL.md").is_file()
    for rel in (
        "references/m4-vs-m5-architecture.md",
        "references/m4-m5-capability-matrix.md",
        "references/kernel-and-dtype-playbook.md",
        "references/dead-ends-registry.md",
        "references/data/matrix_curated.json",
        "references/data/catalog_instruction_differences.jsonl",
        "references/data/catalog_a64_encodings.jsonl",
        "references/data/catalog_metal_gpu.jsonl",
        "references/data/catalog_mlx_api.jsonl",
        "references/data/catalog_apple_apis.jsonl",
        "references/data/catalog_metal_sdk.jsonl",
        "references/data/catalog_bnns_graph_api.jsonl",
        "scripts/probe_silicon_ceilings.py",
        "scripts/probe_mpp_metal4.mm",
        "scripts/probe_ane_and_hetero.mm",
        "scripts/audit_safetensors_model.py",
        "scripts/ab_bench_harness.py",
        "scripts/query_capability_matrix.py",
    ):
        assert (skill_dir / rel).is_file()

    # Validate merged capability matrix dataset & CLI helper
    import importlib.util

    matrix_data = json.loads((skill_dir / "references/data/matrix_curated.json").read_text(encoding="utf-8"))
    assert matrix_data["metadata"]["validation_summary"]["a64_boolean_predicate_logic_errors"] == 0
    assert matrix_data["metadata"]["validation_summary"]["apple_apis_exact_duplicates_collapsed"] == 2
    assert len(matrix_data["six_chip_matrices"]["mlx"]["rows"]) == 57
    assert len(matrix_data["cpu_features"]) == 99

    q_spec = importlib.util.spec_from_file_location("query_capability_matrix", skill_dir / "scripts" / "query_capability_matrix.py")
    assert q_spec is not None and q_spec.loader is not None
    q_mod = importlib.util.module_from_spec(q_spec)
    q_spec.loader.exec_module(q_mod)
    assert q_mod.main(["--stats", "--json"]) == 0
    assert q_mod.main(["--table", "mlx", "--search", "MLX_ENABLE_TF32", "--limit", "5"]) == 0


def test_grammar_fallback_and_temperature_was_masked_cpu() -> None:
    llg = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.grammar import GrammarHelper

    vocab = {"<|endoftext|>": 0, "{": 1, '"': 2, "a": 3, ":": 4, "1": 5, "2": 6, "}": 7}
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llg.LLTokenizer(bt.to_str(), eos_token=0)
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }

    # Pruned helper drops both valid digit tokens `5` ("1") and `6` ("2").
    pruned_gh = GrammarHelper(
        ltok,
        schema,
        new_to_old=np.array([0, 1, 2, 3, 4, 7], dtype=np.int32),
        whitespace_flexible=False,
        eos_token_id=0,
    )
    m_pruned = pruned_gh.new_matcher()
    pruned_logits = np.full(6, -10.0, dtype=np.float32)
    full_logits = np.full(len(vocab), -10.0, dtype=np.float32)
    full_logits[5] = -2.0  # lowest valid bit
    full_logits[6] = 15.0  # higher-logit valid token ("2")

    # 1a. Full-vocab fallback must score valid candidates against unpruned logits (not lowest bit 5).
    tok_fb, masked_fb = pruned_gh.pick_valid_token(
        m_pruned,
        pruned_logits,
        0,
        fallback_to_full_vocab=True,
        full_logits_1d=full_logits,
    )
    assert tok_fb == 6 and masked_fb is True

    # 1b. Passing full-vocab array directly as logits_1d also ranks properly.
    m_pruned2 = pruned_gh.new_matcher()
    tok_fb2, masked_fb2 = pruned_gh.pick_valid_token(
        m_pruned2,
        full_logits,
        0,
        fallback_to_full_vocab=True,
    )
    assert tok_fb2 == 6 and masked_fb2 is True

    # 1c. Missing unpruned logits with no EOS allowed raises ValueError instead of blind lowest-bit pick.
    m_pruned3 = pruned_gh.new_matcher()
    with pytest.raises(ValueError, match="full_logits_1d"):
        pruned_gh.pick_valid_token(
            m_pruned3,
            pruned_logits,
            0,
            fallback_to_full_vocab=True,
        )

    # 2. Temperature > 0 returns was_masked=True when unconstrained argmax is disallowed, False when allowed.
    full_gh = GrammarHelper(ltok, schema, whitespace_flexible=False, eos_token_id=0)
    m_temp = full_gh.new_matcher()
    logits_bad_argmax = np.full(len(vocab), -50.0, dtype=np.float32)
    logits_bad_argmax[5] = 2.0
    logits_bad_argmax[6] = 3.0
    logits_bad_argmax[7] = 100.0  # '}' disallowed before integer digit
    tok_t1, masked_t1 = full_gh.pick_valid_token(
        m_temp,
        logits_bad_argmax,
        7,
        temperature=0.8,
        rng=np.random.default_rng(0),
    )
    assert tok_t1 in (5, 6) and masked_t1 is True

    logits_ok_argmax = np.full(len(vocab), -50.0, dtype=np.float32)
    logits_ok_argmax[5] = 100.0  # '1' allowed
    logits_ok_argmax[6] = 2.0
    tok_t2, masked_t2 = full_gh.pick_valid_token(
        m_temp,
        logits_ok_argmax,
        5,
        temperature=0.8,
        rng=np.random.default_rng(0),
    )
    assert tok_t2 in (5, 6) and masked_t2 is False


def test_pipeline_resume_atomic_and_roll_slots_cpu(tmp_path: Path) -> None:
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from fast_shared.pipeline_llm import extract_jsonl, extract_parquet

    schema = {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
    n_docs = 300
    in_pq = tmp_path / "in.parquet"
    pq.write_table(
        pa.table({"id": list(range(n_docs)), "body_text": [f"doc-{i}" for i in range(n_docs)]}),
        in_pq,
    )
    out_dir = tmp_path / "shards"

    class _Pass1Engine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_batch(self, prompts, **kwargs):
            self.calls.append({"n": len(prompts), **kwargs})
            items = []
            vs_cnt = 0
            for p in prompts:
                doc_idx = int(p.split("-")[1])
                if doc_idx == 0:
                    # Schema failure must be retried on --resume
                    items.append(SimpleNamespace(
                        text='{"bad":1}', parsed=None, valid_json=True, valid_schema=False,
                        finish_reason="stop", prompt_tokens=4, completion_tokens=3,
                    ))
                elif doc_idx == 1:
                    # Truncated length row must be retried on --resume even if valid_schema wasn't False
                    items.append(SimpleNamespace(
                        text='{"id":1', parsed=None, valid_json=False, valid_schema=True,
                        finish_reason="length", prompt_tokens=4, completion_tokens=8,
                    ))
                else:
                    vs_cnt += 1
                    items.append(SimpleNamespace(
                        text=f'{{"id":{doc_idx}}}', parsed={"id": doc_idx}, valid_json=True, valid_schema=True,
                        finish_reason="stop", prompt_tokens=4, completion_tokens=3,
                    ))
            return SimpleNamespace(
                items=items,
                stats={
                    "docs": len(prompts),
                    "valid_json_count": len(prompts) - 1,
                    "valid_schema_count": vs_cnt,
                    "prompt_tokens": 4 * len(prompts),
                    "completion_tokens": 3 * len(prompts),
                    "decode_sec": 0.05,
                },
            )

    eng1 = _Pass1Engine()
    sum1 = extract_parquet(
        eng1,
        in_pq,
        out_dir,
        system_prompt="sys",
        json_schema=schema,
        roll_slots=256,
        shard_rows=1000,
    )
    assert sum1.docs_processed == n_docs
    # roll_slots=256 must NOT be clamped to 128 in gen_kw["batch_size"], and macro_window_rows=4096 feeds all 300 docs
    assert eng1.calls[0]["n"] >= 256 and eng1.calls[0]["batch_size"] == 256
    shard0 = pq.read_table(out_dir / "part-00000.parquet")
    assert "finish_reason" in shard0.column_names
    assert "extractconf_risk" in shard0.column_names
    assert shard0.column("finish_reason").to_pylist()[1] == "length"

    class _Pass2Engine:
        def __init__(self) -> None:
            self.seen_prompts: list[str] = []

        def generate_batch(self, prompts, **kwargs):
            self.seen_prompts.extend(prompts)
            items = [
                SimpleNamespace(
                    text=f'{{"id":{int(p.split("-")[1])}}}',
                    parsed={"id": int(p.split("-")[1])},
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    prompt_tokens=4,
                    completion_tokens=3,
                )
                for p in prompts
            ]
            return SimpleNamespace(
                items=items,
                stats={
                    "docs": len(prompts),
                    "valid_json_count": len(prompts),
                    "valid_schema_count": len(prompts),
                    "prompt_tokens": 4 * len(prompts),
                    "completion_tokens": 3 * len(prompts),
                    "decode_sec": 0.02,
                },
            )

    eng2 = _Pass2Engine()
    sum2 = extract_parquet(
        eng2,
        in_pq,
        out_dir,
        system_prompt="sys",
        json_schema=schema,
        roll_slots=256,
        resume=True,
    )
    assert eng2.seen_prompts == ["doc-0", "doc-1"]
    assert sum2.docs_processed == 2

    # Next --resume pass skips everything cleanly once repaired
    eng3 = _Pass2Engine()
    sum3 = extract_parquet(eng3, in_pq, out_dir, system_prompt="sys", json_schema=schema, resume=True)
    assert sum3.docs_processed == 0 and eng3.seen_prompts == []

    # Crash during shard write must leave NO partial/final part-*.parquet or .tmp file
    crash_in = tmp_path / "crash.parquet"
    crash_out = tmp_path / "crash_shards"
    pq.write_table(pa.table({"id": [1, 2, 3, 4], "body_text": ["a", "b", "c", "d"]}), crash_in)

    class _CrashingEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate_batch(self, prompts, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            items = [
                SimpleNamespace(
                    text='{"id":1}', parsed={"id": 1}, valid_json=True, valid_schema=True,
                    finish_reason="stop", prompt_tokens=2, completion_tokens=2,
                )
                for _ in prompts
            ]
            return SimpleNamespace(items=items, stats={"valid_json_count": 2, "valid_schema_count": 2})

    with pytest.raises(RuntimeError, match="boom"):
        extract_parquet(
            _CrashingEngine(),
            crash_in,
            crash_out,
            system_prompt="sys",
            json_schema=schema,
            batch_rows=2,
            shard_rows=10,
        )
    assert list(crash_out.glob("*")) == []

    # extract_jsonl atomic write & finish_reason
    in_jl = tmp_path / "in.jsonl"
    out_jl = tmp_path / "out.jsonl"
    in_jl.write_text('{"id":0,"text":"doc-0"}\n', encoding="utf-8")
    extract_jsonl(eng2, in_jl, out_jl, system_prompt="sys", json_schema=schema)
    rec0 = json.loads(out_jl.read_text(encoding="utf-8").splitlines()[0])
    assert rec0["finish_reason"] == "stop"




def test_parquet_async_writer_overlaps_encode_and_propagates_errors(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time
    import fast_shared.parquet_pipeline as pp

    n_rows = 12
    tbl = pa.table({"id": list(range(n_rows)), "text": [f"doc {i}" for i in range(n_rows)]})
    in_p = tmp_path / "in.parquet"
    pq.write_table(tbl, in_p)

    # 1) Sharded async writer must overlap batch (k+1) encode with batch (k) Parquet write
    events: list[tuple[str, int, float, float]] = []
    ev_lock = threading.Lock()
    orig_write_table = pp.pq.write_table
    shard_call_idx = 0

    def slow_write_table(table, where, *args, **kwargs):
        nonlocal shard_call_idx
        with ev_lock:
            idx = shard_call_idx
            shard_call_idx += 1
        t0 = time.perf_counter()
        time.sleep(0.06)
        orig_write_table(table, where, *args, **kwargs)
        t1 = time.perf_counter()
        with ev_lock:
            events.append(("write", idx, t0, t1))

    class _SlowEngine(_DummyEngine):
        def __init__(self) -> None:
            super().__init__()
            self._enc_idx = 0

        def encode_corpus(self, tok_lists, mode=None):
            idx = self._enc_idx
            self._enc_idx += 1
            t0 = time.perf_counter()
            time.sleep(0.06)
            out = super().encode_corpus(tok_lists, mode=mode)
            t1 = time.perf_counter()
            with ev_lock:
                events.append(("encode", idx, t0, t1))
            return out

    monkeypatch.setattr(pp.pq, "write_table", slow_write_table)
    out_dir = tmp_path / "shards_async"
    eng = _SlowEngine()
    stats = embed_parquet(
        eng,
        input_path=in_p,
        output_path=out_dir,
        batch_rows=4,
        shard_rows=4,
        progress_every=0,
    )
    assert stats.rows_emitted == n_rows
    enc_ev = {idx: (t0, t1) for kind, idx, t0, t1 in events if kind == "encode"}
    wrt_ev = {idx: (t0, t1) for kind, idx, t0, t1 in events if kind == "write"}
    assert len(enc_ev) == 3 and len(wrt_ev) == 3
    assert enc_ev[1][0] < wrt_ev[0][1], (
        f"Expected async writer overlap (encode[1] start {enc_ev[1][0]:.4f} < write[0] end {wrt_ev[0][1]:.4f})"
    )
    assert not list(out_dir.glob("*.tmp")) and not list(out_dir.glob(".*.tmp"))

    # Reference parity against plain synchronous _DummyEngine values
    ref_eng = _DummyEngine()
    parts = sorted(out_dir.glob("part-*.parquet"))
    assert len(parts) == 3
    joined = pa.concat_tables([pq.read_table(p) for p in parts])
    got_vecs = np.asarray(joined.column("vector").to_pylist(), dtype=np.float32)
    txt_list = tbl.column("text").to_pylist()
    exp_vecs = np.concatenate(
        [ref_eng.encode_corpus(ref_eng.tokenizer.encode_batch(txt_list[k : k + 4])) for k in range(0, n_rows, 4)],
        axis=0,
    )
    assert np.allclose(got_vecs, exp_vecs, atol=1e-6)

    # 2) Single-file async writer overlap + parity + no leftover .tmp
    events.clear()
    orig_pw_write = pp.pq.ParquetWriter.write_table
    pw_idx = 0

    def slow_pw_write(self_pw, table, *args, **kwargs):
        nonlocal pw_idx
        with ev_lock:
            idx = pw_idx
            pw_idx += 1
        t0 = time.perf_counter()
        time.sleep(0.06)
        orig_pw_write(self_pw, table, *args, **kwargs)
        t1 = time.perf_counter()
        with ev_lock:
            events.append(("pw_write", idx, t0, t1))

    monkeypatch.setattr(pp.pq.ParquetWriter, "write_table", slow_pw_write)
    out_single = tmp_path / "single_async.parquet"
    stats_single = embed_parquet(
        _SlowEngine(),
        input_path=in_p,
        output_path=out_single,
        batch_rows=4,
        progress_every=0,
    )
    assert stats_single.rows_emitted == n_rows
    enc_s = {idx: (t0, t1) for kind, idx, t0, t1 in events if kind == "encode"}
    wrt_s = {idx: (t0, t1) for kind, idx, t0, t1 in events if kind == "pw_write"}
    assert enc_s[1][0] < wrt_s[0][1], "Expected single-file async writer overlap"
    assert out_single.exists() and not out_single.with_suffix(".parquet.tmp").exists()
    got_single = np.asarray(pq.read_table(out_single).column("vector").to_pylist(), dtype=np.float32)
    assert np.allclose(got_single, exp_vecs, atol=1e-6)

    # 3) Error propagation & .tmp cleanup in both sharded and single-file modes
    def fail_shard_write(table, where, *args, **kwargs):
        if "part-00001" in str(where):
            raise RuntimeError("simulated disk error")
        orig_write_table(table, where, *args, **kwargs)

    monkeypatch.setattr(pp.pq, "write_table", fail_shard_write)
    err_dir = tmp_path / "shards_err"
    with pytest.raises(RuntimeError, match="simulated disk error"):
        embed_parquet(_DummyEngine(), input_path=in_p, output_path=err_dir, batch_rows=4, shard_rows=4, progress_every=0)
    assert not list(err_dir.glob("*.tmp")) and not list(err_dir.glob(".*.tmp"))

    def fail_pw_write(self_pw, table, *args, **kwargs):
        raise RuntimeError("simulated single writer error")

    monkeypatch.setattr(pp.pq.ParquetWriter, "write_table", fail_pw_write)
    err_single = tmp_path / "err_single.parquet"
    with pytest.raises(RuntimeError, match="simulated single writer error"):
        embed_parquet(_DummyEngine(), input_path=in_p, output_path=err_single, batch_rows=4, progress_every=0)
    assert not err_single.exists() and not err_single.with_suffix(".parquet.tmp").exists()


def test_hw_check_baseline_and_warning_cpu(monkeypatch: Any, capsys: Any) -> None:
    import fast_shared.hw_check as hc

    ref = dict(hc.TESTED_REFERENCE_HARDWARE)
    ok, mismatches, _ = hc.evaluate_hardware_support(ref)
    assert ok is True and mismatches == []

    other_mac = {
        **ref,
        "hw_model": "Mac15,6",
        "brand": "Apple M3 Pro",
        "mem_gb": 36,
        "cpu_cores": 12,
        "perf_cores": 6,
        "eff_cores": 6,
        "gpu_cores": 18,
    }
    ok2, mismatches2, _ = hc.evaluate_hardware_support(other_mac)
    assert ok2 is False
    assert any("Apple M3 Pro" in m for m in mismatches2)
    assert any("36 GB" in m for m in mismatches2)
    assert any("gpu_cores=18" in m for m in mismatches2)

    monkeypatch.setattr(hc, "probe_host_hardware", lambda: other_mac)
    monkeypatch.setattr(hc, "_WARNED", False)
    monkeypatch.delenv("FAST_SILICON_NO_HW_WARN", raising=False)
    assert hc.warn_if_untested_hardware() is False
    err = capsys.readouterr().err
    assert "WARNING: Untested hardware configuration detected!" in err
    assert "48 GB Apple M4 Pro" in err
    assert "14 CPU cores (10P + 4E)" in err
    assert "20 GPU cores" in err

    # Second call with once=True does not spam stderr again
    assert hc.warn_if_untested_hardware() is False
    assert capsys.readouterr().err == ""

    # Opt-out env var silences stderr warning
    monkeypatch.setattr(hc, "_WARNED", False)
    monkeypatch.setenv("FAST_SILICON_NO_HW_WARN", "1")
    assert hc.warn_if_untested_hardware(once=False) is False
    assert capsys.readouterr().err == ""

    assert hc.main(["--strict"]) == 2
    capsys.readouterr()
    assert hc.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matched_tested_baseline"] is False
    assert payload["tested_reference"]["gpu_cores"] == 20

    # Matched host produces no stderr warning and exits 0 even with --strict
    monkeypatch.setattr(hc, "probe_host_hardware", lambda: ref)
    monkeypatch.setattr(hc, "_WARNED", False)
    monkeypatch.delenv("FAST_SILICON_NO_HW_WARN", raising=False)
    assert hc.warn_if_untested_hardware() is True
    assert hc.main(["--strict"]) == 0
    captured = capsys.readouterr()
    assert "matches tested baseline" in captured.out
    assert captured.err == ""


def test_set_mlx_cache_limit_cpu(monkeypatch: Any) -> None:
    import sys

    prev = mx.set_cache_limit(64 * 1024 * 1024)
    try:
        applied = set_mlx_cache_limit(128)
        assert applied == 128 * 1024 * 1024
        assert mx.set_cache_limit(prev) == 128 * 1024 * 1024

        assert set_mlx_cache_limit(0) == 0
        assert mx.set_cache_limit(prev) == 0

        assert set_mlx_cache_limit(-4) == 0
        assert mx.set_cache_limit(prev) == 0
    finally:
        mx.set_cache_limit(prev)

    for bad in (True, "128", 1.5):
        with pytest.raises(ValueError):
            set_mlx_cache_limit(bad)  # type: ignore[arg-type]

    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert set_mlx_cache_limit(512) is None


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    import struct

    header: dict[str, Any] = {"__metadata__": {"format": "test"}}
    offset = 0
    chunks: list[bytes] = []
    for name, (dtype_name, shape, payload) in tensors.items():
        end = offset + len(payload)
        header[name] = {
            "dtype": dtype_name,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        chunks.append(payload)
        offset = end
    hdr_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(hdr_bytes)))
        handle.write(hdr_bytes)
        for chunk in chunks:
            handle.write(chunk)


def test_mapped_embedding_safetensors_cpu(tmp_path: Path) -> None:
    # Exact BF16 bit patterns (lower 16 bits zeroed in float32)
    raw_f32 = np.array(
        [
            [0.0, 1.0, -2.5, 3.25],
            [-0.125, 4.5, 16.0, -32.0],
            [0.5, -0.75, 128.0, 6.5],
            [1.25, -8.0, 0.25, -1.5],
        ],
        dtype=np.float32,
    )
    bf16_u16 = (raw_f32.view(np.uint32) >> 16).astype("<u2")
    bf16_path = tmp_path / "bf16.safetensors"
    _write_safetensors(
        bf16_path,
        {
            "other.bias": ("F32", (4,), np.zeros(4, dtype="<f4").tobytes()),
            "model.embed_tokens.weight": ("BF16", raw_f32.shape, bf16_u16.tobytes()),
        },
    )

    with MappedEmbedding(bf16_path, tensor_name="model.embed_tokens.weight") as emb:
        assert emb.vocab_size == 4
        assert emb.hidden_size == 4
        assert emb.dtype == "BF16"

        out_f32 = emb.gather(np.array([2, 0, 3], dtype=np.int32), out_dtype="float32")
        assert out_f32.dtype == np.float32 and out_f32.shape == (3, 4)
        assert np.array_equal(out_f32, raw_f32[[2, 0, 3]])

        ids_2d = np.array([[1, 2], [0, 3]], dtype=np.int64)
        out_f16 = emb.gather(ids_2d, out_dtype="float16")
        assert out_f16.dtype == np.float16 and out_f16.shape == (2, 2, 4)
        assert np.allclose(out_f16, raw_f32[ids_2d].astype(np.float16), atol=1e-3)

        # 0D scalar and empty shape handling
        assert np.array_equal(emb.gather(np.array(1), out_dtype="float32"), raw_f32[1])
        empty_1d = emb.gather(np.empty((0,), dtype=np.int32), out_dtype="float16")
        assert empty_1d.shape == (0, 4) and empty_1d.dtype == np.float16
        empty_2d = emb.gather(np.empty((2, 0), dtype=np.int64), out_dtype="float32")
        assert empty_2d.shape == (2, 0, 4) and empty_2d.dtype == np.float32
        assert emb.gather([], out_dtype="float16").shape == (0, 4)

        with pytest.raises(IndexError):
            emb.gather(np.array([-1]))
        with pytest.raises(IndexError):
            emb.gather(np.array([4]))
        with pytest.raises(TypeError):
            emb.gather(np.array([0.5]))
        with pytest.raises(TypeError):
            emb.gather(np.array([True]))
        with pytest.raises(ValueError, match="Unsupported out_dtype"):
            emb.gather(np.array([0]), out_dtype="bfloat16")

    with pytest.raises(ValueError, match="closed MappedEmbedding"):
        emb.gather(np.array([0]))

    # F16 + F32 via single-file model.safetensors directory
    single_dir = tmp_path / "single_ckpt"
    single_dir.mkdir()
    f16_vals = raw_f32.astype("<f2")
    _write_safetensors(
        single_dir / "model.safetensors",
        {
            "model.embed_tokens.weight": ("F16", f16_vals.shape, f16_vals.tobytes()),
            "extra.f32": ("F32", raw_f32.shape, raw_f32.astype("<f4").tobytes()),
        },
    )
    with MappedEmbedding.from_checkpoint(single_dir) as emb_f16:
        assert emb_f16.dtype == "F16"
        assert np.array_equal(emb_f16.gather(np.array([0, 3]), out_dtype="float16"), f16_vals[[0, 3]])
    with MappedEmbedding.from_checkpoint(single_dir, tensor_name="extra.f32") as emb_f32:
        assert emb_f32.dtype == "F32"
        assert np.array_equal(emb_f32.gather(np.array([1, 2]), out_dtype="float32"), raw_f32[[1, 2]])

    # Sharded checkpoint directory via model.safetensors.index.json
    sharded_dir = tmp_path / "sharded_ckpt"
    sharded_dir.mkdir()
    _write_safetensors(
        sharded_dir / "model-00001-of-00002.safetensors",
        {"layer0.weight": ("F32", (2, 2), np.ones((2, 2), dtype="<f4").tobytes())},
    )
    _write_safetensors(
        sharded_dir / "model-00002-of-00002.safetensors",
        {"language_model.embed_tokens.weight": ("BF16", raw_f32.shape, bf16_u16.tobytes())},
    )
    (sharded_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "layer0.weight": "model-00001-of-00002.safetensors",
                    "language_model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    with MappedEmbedding.from_checkpoint(
        sharded_dir, tensor_name="language_model.embed_tokens.weight"
    ) as emb_sharded:
        assert emb_sharded.vocab_size == 4 and emb_sharded.hidden_size == 4
        assert np.array_equal(emb_sharded.gather(np.array([1, 3]), out_dtype="float32"), raw_f32[[1, 3]])

    with pytest.raises(KeyError):
        MappedEmbedding.from_checkpoint(sharded_dir, tensor_name="missing.weight")
    with pytest.raises(FileNotFoundError):
        MappedEmbedding.from_checkpoint(tmp_path / "does_not_exist")


def test_tile_aligned_slices_cpu() -> None:
    assert tile_aligned_slices(0) == []
    assert tile_aligned_slices(10, slice_len=512, min_tail=64) == [(0, 10)]
    assert tile_aligned_slices(512, slice_len=512, min_tail=64) == [(0, 512)]
    assert tile_aligned_slices(513, slice_len=512, min_tail=64) == [(0, 513)]
    assert tile_aligned_slices(575, slice_len=512, min_tail=64) == [(0, 575)]
    assert tile_aligned_slices(576, slice_len=512, min_tail=64) == [(0, 512), (512, 576)]
    assert tile_aligned_slices(1087, slice_len=512, min_tail=64) == [(0, 512), (512, 1087)]
    assert tile_aligned_slices(1088, slice_len=512, min_tail=64) == [(0, 512), (512, 1024), (1024, 1088)]
    assert tile_aligned_slices(130, slice_len=64, min_tail=0) == [(0, 64), (64, 128), (128, 130)]

    for bad_kwargs in (
        {"seq_len": -1},
        {"seq_len": 100, "slice_len": 0},
        {"seq_len": 100, "slice_len": -16},
        {"seq_len": 100, "min_tail": -1},
        {"seq_len": True},
    ):
        with pytest.raises(ValueError):
            tile_aligned_slices(**bad_kwargs)  # type: ignore[arg-type]


def test_fault_injection_suite_cpu() -> None:
    from contextlib import contextmanager

    state = {"scale": 1.0, "permute": False, "explode": False}
    calls = 0

    def baseline_fn() -> np.ndarray:
        nonlocal calls
        calls += 1
        if state["explode"]:
            raise RuntimeError("injected failure")
        vec = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32) * state["scale"]
        if state["permute"]:
            vec = vec[::-1].copy()
        return vec

    @contextmanager
    def _patch(key: str, val: Any):
        old = state[key]
        state[key] = val
        try:
            yield
        finally:
            state[key] = old

    cases = [
        FaultInjectionCase(
            name="scale_shift",
            apply_fault=lambda: _patch("scale", 0.5),
            min_delta=1.0,
            metric="max_abs_diff",
        ),
        FaultInjectionCase(
            name="permute_cos_drop",
            apply_fault=lambda: _patch("permute", True),
            min_delta=0.1,
            metric="cos_drop",
        ),
        FaultInjectionCase(
            name="noop_missed_fault",
            apply_fault=lambda: _patch("scale", 1.0),
            min_delta=1e-2,
            metric="max_abs_diff",
        ),
        FaultInjectionCase(
            name="expected_exception",
            apply_fault=lambda: _patch("explode", True),
            metric="raises",
        ),
        FaultInjectionCase(
            name="missing_expected_exception",
            apply_fault=lambda: _patch("scale", 1.0),
            metric="raises",
        ),
        FaultInjectionCase(
            name="unexpected_exception_in_delta_case",
            apply_fault=lambda: _patch("explode", True),
            min_delta=1e-2,
            metric="max_abs_diff",
        ),
    ]

    results = run_fault_injection_suite(baseline_fn, cases)
    assert calls == 1 + len(cases)
    assert state == {"scale": 1.0, "permute": False, "explode": False}
    assert all(isinstance(r, FaultInjectionResult) for r in results)

    by_name = {r.name: r for r in results}
    assert by_name["scale_shift"].passed is True
    assert np.isclose(by_name["scale_shift"].measured_delta, 1.5, atol=1e-6)
    assert by_name["scale_shift"].error is None

    assert by_name["permute_cos_drop"].passed is True
    assert by_name["permute_cos_drop"].measured_delta > 0.1
    assert by_name["permute_cos_drop"].error is None

    assert by_name["noop_missed_fault"].passed is False
    assert by_name["noop_missed_fault"].measured_delta == 0.0
    assert by_name["noop_missed_fault"].error is None

    assert by_name["expected_exception"].passed is True
    assert by_name["expected_exception"].error is None

    assert by_name["missing_expected_exception"].passed is False
    assert by_name["missing_expected_exception"].error is not None

    assert by_name["unexpected_exception_in_delta_case"].passed is False
    assert "RuntimeError" in (by_name["unexpected_exception_in_delta_case"].error or "")

    with pytest.raises(ValueError, match="metric"):
        FaultInjectionCase(name="bad_metric", apply_fault=lambda: _patch("scale", 1.0), metric="unknown")
    with pytest.raises(ValueError, match="min_delta"):
        FaultInjectionCase(name="neg_delta", apply_fault=lambda: _patch("scale", 1.0), min_delta=-0.1)


def test_unified_memory_footprint_darwin_cpu() -> None:
    from fast_shared.bench import get_unified_memory_footprint, peak_footprint_gb

    snap = get_unified_memory_footprint()
    assert snap["phys_footprint_bytes"] > 0
    assert snap["lifetime_max_phys_footprint_bytes"] >= snap["phys_footprint_bytes"]
    assert snap["rss_bytes"] > 0
    assert snap["peak_footprint_gb"] == pytest.approx(
        snap["lifetime_max_phys_footprint_bytes"] / 1e9, rel=1e-9
    )
    assert peak_footprint_gb() == snap["peak_footprint_gb"]


def test_hw_check_six_chip_matrix_and_preflight_cpu(monkeypatch: Any) -> None:
    import fast_shared.hw_check as hc

    # All 6 M4/M5 chips validate cleanly without spurious warnings
    six_chips = [
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac16,1", "brand": "Apple M4", "mem_gb": 16, "cpu_cores": 10, "perf_cores": 4, "eff_cores": 6, "gpu_cores": 10, "ane_cores": 16, "macos_version": "26.6"},
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac16,8", "brand": "Apple M4 Pro", "mem_gb": 48, "cpu_cores": 14, "perf_cores": 10, "eff_cores": 4, "gpu_cores": 20, "ane_cores": 16, "macos_version": "26.6"},
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac16,5", "brand": "Apple M4 Max", "mem_gb": 64, "cpu_cores": 16, "perf_cores": 12, "eff_cores": 4, "gpu_cores": 40, "ane_cores": 16, "macos_version": "26.6"},
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac17,2", "brand": "Apple M5", "mem_gb": 24, "cpu_cores": 10, "perf_cores": 4, "eff_cores": 6, "perflevel0_name": "Super", "perflevel1_name": "Efficiency", "gpu_cores": 10, "ane_cores": 16, "gpu_family": "Apple10", "supports_apple10": True, "supports_metal4": True, "macos_version": "26.2"},
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac17,6", "brand": "Apple M5 Pro", "mem_gb": 48, "cpu_cores": 18, "perf_cores": 6, "eff_cores": 12, "perflevel0_name": "Super", "perflevel1_name": "Performance", "gpu_cores": 20, "ane_cores": 16, "gpu_family": "Apple10", "supports_apple10": True, "supports_metal4": True, "macos_version": "27.2.1"},
        {"system": "Darwin", "machine": "arm64", "hw_model": "Mac17,8", "brand": "Apple M5 Max", "mem_gb": 128, "cpu_cores": 18, "perf_cores": 6, "eff_cores": 12, "perflevel0_name": "Super", "perflevel1_name": "Performance", "gpu_cores": 40, "ane_cores": 16, "gpu_family": "Apple10", "supports_apple10": True, "supports_metal4": True, "macos_version": "27.2.1"},
    ]
    for chip in six_chips:
        ok, mismatches, _ = hc.evaluate_hardware_support(chip)
        assert ok is True and mismatches == [], f"Unexpected mismatch for {chip['brand']}: {mismatches}"

    # supports_apple10_nax & supports_metal4_bf16 OS/flag gates
    m5_262 = dict(six_chips[3])
    assert hc.supports_apple10_nax(m5_262) is True
    assert hc.supports_apple10_nax({**m5_262, "mlx_metal_no_nax": True}) is False
    assert hc.supports_apple10_nax({**m5_262, "macos_version": "26.1"}) is False
    assert hc.supports_apple10_nax(six_chips[1]) is False

    assert hc.supports_metal4_fp16({**six_chips[1], "macos_version": "26.0", "supports_metal4": True}) is True
    assert hc.supports_metal4_bf16({**six_chips[1], "macos_version": "26.0", "supports_metal4": True}) is False
    assert hc.supports_metal4_bf16({**six_chips[1], "macos_version": "26.1", "supports_metal4": True}) is True

    # Live host probe checks Missing != False and rich telemetry fields
    live = hc.probe_host_hardware()
    assert live["macos_version"] is not None
    assert live["mlx_version"] is not None
    assert isinstance(live["arm_features"], dict)
    assert "FEAT_SME2" in live["arm_features"]
    assert "FEAT_SVE_B16B16" in live["arm_features"]
    # Missing arm feature keys must be None, not coerced to False
    for val in live["arm_features"].values():
        assert val in (True, False, None)

    # preflight_silicon_env
    pf = hc.preflight_silicon_env(require_mlx_min="0.32.4", check_tf32=True, strict=False)
    assert pf["ok"] is True
    assert pf["mlx_meets_min"] is True
    with pytest.raises(RuntimeError, match="preflight failed"):
        hc.preflight_silicon_env(require_mlx_min="99.0.0", strict=True)


def test_roofline_quant_audit_and_hybrid_moe_models_cpu() -> None:
    from fast_shared.roofline import audit_safetensors_weights, compute_model_roofline, probe_hardware_identity

    # 1. Exact bit-width vs group-size classification across 2/3/4/6/8-bit + MXFP4/NVFP4 .blocks/.scales_e8m0
    tensors = {
        # 4-bit g128 (previously misclassified as 8-bit!)
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "U32", "shape": [64, 16], "bytes": 4096, "elems": 1024},
        "model.layers.0.self_attn.q_proj.scales": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        "model.layers.0.self_attn.q_proj.biases": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        # 8-bit g32 (previously misclassified as 4-bit!)
        "model.layers.0.self_attn.k_proj.weight": {"dtype": "U32", "shape": [64, 8], "bytes": 2048, "elems": 512},
        "model.layers.0.self_attn.k_proj.scales": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        "model.layers.0.self_attn.k_proj.biases": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        # 2-bit g64 (previously misclassified as 4-bit!)
        "model.layers.0.self_attn.v_proj.weight": {"dtype": "U32", "shape": [64, 4], "bytes": 1024, "elems": 256},
        "model.layers.0.self_attn.v_proj.scales": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        "model.layers.0.self_attn.v_proj.biases": {"dtype": "BF16", "shape": [64, 1], "bytes": 128, "elems": 64},
        # MXFP4 .blocks + .scales_e8m0 U8 (16 bytes per 32-element group -> 4-bit)
        "model.layers.0.mlp.experts.0.gate_up_proj.blocks": {"dtype": "U8", "shape": [64, 4, 16], "bytes": 4096, "elems": 4096},
        "model.layers.0.mlp.experts.0.gate_up_proj.scales_e8m0": {"dtype": "U8", "shape": [64, 4], "bytes": 256, "elems": 256},
    }
    cfg_q = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "model.layers.0.self_attn.q_proj": {"bits": 4, "group_size": 128},
            "model.layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32},
            "model.layers.0.self_attn.v_proj": {"bits": 2, "group_size": 64},
        }
    }
    audit = audit_safetensors_weights(tensors, config=cfg_q)
    attn_prec = audit["effective_precision"]["attention"]
    # Logical params: q=64*128=8192, k=64*32=2048, v=64*64=4096 => 14336 params total
    assert attn_prec["params"] == 8192 + 2048 + 4096
    assert set(attn_prec["formats"]) == {"q4_g128", "q8_g32", "q2_g64"}
    moe_prec = audit["effective_precision"]["mlp_or_moe"]
    # MXFP4: 4096 U8 bytes = 8192 4-bit params; (4096+256)*8 / 8192 = 4.25 bits/elem
    assert moe_prec["params"] == 8192
    assert moe_prec["bits_per_elem"] == 4.25
    assert "mxfp4_g32" in moe_prec["formats"]

    # 2. Hybrid linear_attention (Qwen3.5/DeltaNet) must NOT enter swa_own
    hybrid_cfg = {
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 151936,
        "layer_types": ["linear_attention"] * 21 + ["full_attention"] * 7,
    }
    roof_f16 = compute_model_roofline(hybrid_cfg, 600_000_000, ctx=2048, kv_dtype_bytes=2.0)
    roof_i8 = compute_model_roofline(hybrid_cfg, 600_000_000, ctx=2048, kv_dtype_bytes=1.0)
    assert roof_f16["full_own"] == 7
    assert roof_f16["linear_own"] == 21 and roof_f16["n_gdn"] == 21
    assert roof_f16["swa_own"] == 0 and roof_f16["swa_fixed_mib"] == 0.0
    assert roof_i8["kv_kib_per_tok"] == pytest.approx(roof_f16["kv_kib_per_tok"] / 2.0, rel=1e-6)

    # 3. Calibrated MoE fixed vs routed weight partition
    moe_cfg = {
        "num_hidden_layers": 12,
        "hidden_size": 2048,
        "intermediate_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 65536,
        "num_local_experts": 32,
        "num_experts_per_tok": 2,
    }
    r_est = compute_model_roofline(moe_cfg, 10_000_000_000, gpu_tf=7.4, dram_gbs=200.0)
    r_exact = compute_model_roofline(moe_cfg, 10_000_000_000, gpu_tf=7.4, dram_gbs=200.0, non_expert_weight_bytes=1_000_000_000)
    assert r_exact["moe_fixed_weight_frac"] == 0.1
    assert r_est["moe_fixed_weight_frac"] != 0.25  # Structural ratio, not hardcoded 0.25

    hw_id = probe_hardware_identity()
    for key in ("macos_version", "xcode_sdk_version", "mlx_version", "mlx_enable_tf32", "mlx_metal_no_nax", "gpu_family", "supports_placement_sparse"):
        assert key in hw_id


def test_metal_mpp_guards_and_tail_tiles() -> None:
    import fast_shared.metal_mpp as mmpp
    import fast_shared.metal_mpp_bf16 as mmpp_bf

    assert "#include <metal_stdlib>" in mmpp.HEADER_SIMDGROUP
    assert "#include <metal_stdlib>" in mmpp.HEADER_MPP

    with _gpu_validation_lock():
        # Tail-tile safety: non-multiple of 64 (M=50, N=50) must not crash or read OOB
        mx.set_default_device(mx.gpu)
        k_none = mmpp.make_mpp_matmul_kernel("shared_test_tail_none", epilogue="none", tm=64, tn=64, sgs=4)
        a = mx.ones((50, 64), dtype=mx.float16)
        b = mx.ones((50, 64), dtype=mx.float16)
        c = k_none(inputs=[a, b], grid=(128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(50, 50)], output_dtypes=[mx.float16])[0]
        mx.eval(c)
        assert c.shape == (50, 50)
        assert np.allclose(np.array(c), 64.0)

        # geglu epilogue uses exact_erf and matches reference on tail tile
        k_geglu = mmpp.make_mpp_matmul_kernel("shared_test_tail_geglu", epilogue="geglu", tm=64, tn=64, sgs=4)
        b_gu = mx.ones((100, 64), dtype=mx.float16) * 0.05
        out_g = k_geglu(inputs=[a, b_gu], grid=(128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(50, 50)], output_dtypes=[mx.float16])[0]
        mx.eval(out_g)
        assert out_g.shape == (50, 50)
        assert np.isfinite(np.array(out_g)).all()

        # metal_mpp_bf16 dtype / K-alignment / OS / Apple10 threshold fallback guards
        x_fp16 = mx.ones((64, 64), dtype=mx.float16)
        w_fp16 = mx.ones((64, 64), dtype=mx.float16)
        out_fp16 = mmpp_bf.mpp_linear_bf16(x_fp16, w_fp16, min_m=16)
        mx.eval(out_fp16)
        assert out_fp16.dtype == mx.float16 and np.allclose(np.array(out_fp16), 64.0)

        # Unaligned K=40 falls back cleanly without failing MPP
        x_odd = mx.ones((32, 40), dtype=mx.bfloat16)
        w_odd = mx.ones((32, 40), dtype=mx.bfloat16)
        out_odd = mmpp_bf.mpp_linear_bf16(x_odd, w_odd, min_m=16)
        mx.eval(out_odd)
        assert out_odd.dtype == mx.bfloat16 and np.allclose(np.array(out_odd.astype(mx.float32)), 40.0)


def test_server_locking_coalescing_chunked_and_mode_forwarding_cpu() -> None:
    import concurrent.futures
    import http.client
    import threading
    import time
    from types import SimpleNamespace

    from fast_shared.server import (
        ChatMicroBatcher,
        EmbeddingHTTPServer,
        ManagedHTTPServer,
        OpenAIChatAdapter,
        make_chat_handler,
    )

    # 1. Per-engine lock serialization + mode forwarding in EmbeddingHTTPServer
    class _LockProbeEmbEngine(_DummyEngine):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_concurrent = 0
            self.seen_modes: list[str | None] = []
            self._probe_mu = threading.Lock()

        def encode_query(self, toks, mode=None):
            with self._probe_mu:
                self.active += 1
                self.max_concurrent = max(self.max_concurrent, self.active)
                self.seen_modes.append(mode)
            time.sleep(0.02)
            with self._probe_mu:
                self.active -= 1
            return super().encode_query(toks)

    emb = _LockProbeEmbEngine()
    with EmbeddingHTTPServer(emb, host="127.0.0.1", port=0, mode="gpu", max_body_bytes=512) as srv:
        base = f"http://{srv.host}:{srv.port}"

        def _post_emb(i: int) -> int:
            req = urllib.request.Request(
                f"{base}/v1/embeddings",
                data=json.dumps({"input": f"hello-{i}"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(_post_emb, range(4)))
        assert statuses == [200, 200, 200, 200]
        assert emb.max_concurrent == 1  # Strictly serialized by per-engine lock
        assert emb.seen_modes == ["gpu"] * 4

        # Chunked Transfer-Encoding request body works cleanly
        conn = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        body_bytes = json.dumps({"input": "chunked-ok"}).encode("utf-8")
        c1, c2 = body_bytes[:10], body_bytes[10:]
        chunked_wire = f"{len(c1):x}\r\n".encode() + c1 + b"\r\n" + f"{len(c2):x}\r\n".encode() + c2 + b"\r\n0\r\n\r\n"
        conn.putrequest("POST", "/v1/embeddings")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(chunked_wire)
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.close()

        # Oversized body (> 512 bytes) returns HTTP 413
        conn2 = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        big_body = json.dumps({"input": "x" * 1000}).encode("utf-8")
        conn2.request("POST", "/v1/embeddings", body=big_body, headers={"Content-Type": "application/json"})
        resp2 = conn2.getresponse()
        assert resp2.status == 413
        resp2.read()
        conn2.close()

    # 2. ChatMicroBatcher coalesces heterogeneous max_tokens using max(max_tokens) AND handles /v1/completions
    class _MaxTokEngine:
        model_name = "dummy-llm"
        default_mode = "gpu"
        ane_available = False

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.active = 0
            self.max_concurrent = 0
            self._mu = threading.Lock()

        def generate_batch(self, prompts, **kwargs):
            with self._mu:
                self.active += 1
                self.max_concurrent = max(self.max_concurrent, self.active)
                self.calls.append((len(prompts), int(kwargs["max_tokens"])))
            time.sleep(0.03)
            with self._mu:
                self.active -= 1
            items = [
                SimpleNamespace(
                    text=f"ok:{p}",
                    prompt_tokens=3,
                    completion_tokens=2,
                    finish_reason="stop",
                )
                for p in prompts
            ]
            return SimpleNamespace(items=items, stats={"docs": len(prompts)})

    llm = _MaxTokEngine()
    batcher = ChatMicroBatcher(llm, coalesce_ms=25.0, max_batch_size=16)

    def _sub(max_tok: int) -> tuple[Any, dict[str, Any]]:
        return batcher.submit(
            [{"role": "user", "content": f"t-{max_tok}"}],
            json_schema=None,
            max_tokens=max_tok,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        res_pair = list(pool.map(_sub, [32, 128]))
    assert len(res_pair) == 2
    assert llm.max_concurrent == 1
    assert llm.calls == [(2, 128)]  # Coalesced into 1 batch with max(32, 128) == 128!

    # /v1/completions routes through ChatMicroBatcher too
    llm2 = _MaxTokEngine()
    adapter = OpenAIChatAdapter(llm2, coalesce_ms=25.0, max_batch_size=16)
    with ManagedHTTPServer(make_chat_handler(adapter), host="127.0.0.1", port=0) as srv2:
        base2 = f"http://{srv2.host}:{srv2.port}"

        def _hit_cmpl(mt: int) -> dict[str, Any]:
            req = urllib.request.Request(
                f"{base2}/v1/completions",
                data=json.dumps({"prompt": f"p-{mt}", "max_tokens": mt}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read().decode())

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            cmpl_outs = list(pool.map(_hit_cmpl, [16, 96]))
        assert len(cmpl_outs) == 2
        assert llm2.max_concurrent == 1
        assert len(llm2.calls) == 1 and llm2.calls[0] == (2, 96)


def test_grammar_bounds_was_masked_and_thread_safety_cpu() -> None:
    import concurrent.futures
    llg = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.grammar import GrammarHelper

    vocab = {"<|endoftext|>": 0, "{": 1, '"': 2, "a": 3, ":": 4, "1": 5, "2": 6, "}": 7}
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llg.LLTokenizer(bt.to_str(), eos_token=0)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}

    # Pruned helper (len=6) receiving full-vocab argmax_idx=5 (valid digit "1", out of pruned index range if len<6)
    pruned = GrammarHelper(ltok, schema, new_to_old=np.array([0, 1, 2, 3, 5], dtype=np.int32), eos_token_id=0)
    m = pruned.new_matcher()
    full_logits = np.full(len(vocab), -10.0, dtype=np.float32)
    full_logits[5] = 20.0  # valid token 5, index 5 >= len(new_to_old)==5!
    # Previously raised IndexError at new_to_old[5] and hardcoded was_masked=True!
    tok_id, was_masked = pruned.pick_valid_token(m, full_logits, 5)
    assert tok_id == 5 and was_masked is False

    # Concurrent threads sharing one GrammarHelper
    full_gh = GrammarHelper(ltok, schema, eos_token_id=0)
    logits = np.array([-5, -5, -5, -5, -5, 1.0, 2.0, 10.0], dtype=np.float32)

    def _worker(_: int) -> tuple[int, bool]:
        mw = full_gh.new_matcher()
        return full_gh.pick_valid_token(mw, logits, 7)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(32)))
    assert all(tid == 6 and masked is True for tid, masked in results)


def test_bench_wired_limit_quiescence_dtypes_and_paths_cpu(tmp_path: Path, monkeypatch: Any) -> None:
    from fast_shared.bench import (
        _classify_quiescence_line,
        check_tensor_dtypes,
        collect_env_telemetry,
        interleaved_abba,
        wired_limit,
    )

    with wired_limit(64 * 1024 * 1024) as applied:
        assert applied == 64 * 1024 * 1024

    telem = collect_env_telemetry()
    for k in ("brand", "macos_version", "mlx_version", "mlx_enable_tf32", "mlx_metal_no_nax", "supports_apple10_nax"):
        assert k in telem

    # Widened quiescence classifier
    for ps_line in (
        "root 101 12.0 0.5 100 200 ?? R 10:00 0:10 /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Support/mds_stores",
        "_windowserver 102 22.0 1.0 100 200 ?? R 10:00 0:10 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer",
        "root 103 8.0 0.5 100 200 ?? R 10:00 0:10 /System/Library/CoreServices/backupd.bundle/Contents/Resources/backupd",
        "user 104 25.0 1.0 100 200 ?? R 10:00 0:10 /usr/bin/clang++ -O3 foo.cpp",
        "user 105 3.0 1.0 100 200 ?? R 10:00 0:10 /Applications/Arc.app/Contents/Frameworks/Arc Helper (GPU).app/Contents/MacOS/Arc Helper (GPU)",
        "user 106 3.0 1.0 100 200 ?? R 10:00 0:10 /Applications/Microsoft Edge.app/Contents/Frameworks/Microsoft Edge Helper (GPU)",
    ):
        assert _classify_quiescence_line(ps_line) is not None, f"Missed contention line: {ps_line}"

    # Extended check_tensor_dtypes catches bool/int masks & fp16/bf16 mismatches
    flags = check_tensor_dtypes([
        ("attn_mask", mx.ones((1, 4), dtype=mx.bool_)),
        ("q_proj", mx.ones((4,), dtype=mx.float16)),
        ("k_proj", mx.ones((4,), dtype=mx.bfloat16)),
    ])
    assert any("attn_mask" in f for f in flags)
    assert any("mixed fp16/bf16" in f for f in flags)

    abba = interleaved_abba(lambda: {"v": 1}, lambda: {"v": 2}, warmup=False, rounds=2)
    assert "drift" in abba and "env_telemetry" in abba and "A_last_over_first" in abba["drift"]

    # Monorepo paths.resolve_home_dir: packages/<pkg>/native without artifacts/ must NOT stop prematurely!
    mono_pkg = tmp_path / "fast-silicon" / "packages" / "fast-foo"
    anchor = mono_pkg / "src" / "fast_foo" / "paths.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("# anchor", encoding="utf-8")
    (mono_pkg / "native").mkdir()

    monkeypatch.delenv("TEST_FOO_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)
    # 1. No sibling artifacts & no FAST_ARTIFACTS_ROOT -> falls back to user .cache/fast-foo (not mono_pkg!)
    got_cache = resolve_home_dir(anchor, env_var="TEST_FOO_HOME", cache_name="fast-foo")
    assert got_cache == (Path.home() / ".cache" / "fast-foo").resolve()

    # 2. Sibling ../fast-foo with artifacts/ -> resolves to sibling!
    sib = tmp_path / "fast-foo"
    (sib / "artifacts").mkdir(parents=True)
    got_sib = resolve_home_dir(anchor, env_var="TEST_FOO_HOME", cache_name="fast-foo")
    assert got_sib == sib.resolve()

    # 3. FAST_ARTIFACTS_ROOT takes precedence over sibling fallback
    art_root = tmp_path / "shared_arts"
    monkeypatch.setenv("FAST_ARTIFACTS_ROOT", str(art_root))
    got_env_root = resolve_home_dir(anchor, env_var="TEST_FOO_HOME", cache_name="fast-foo")
    assert got_env_root == (art_root / "fast-foo").resolve()


def test_parquet_resume_manifest_and_skill_scripts_cpu(tmp_path: Path) -> None:
    import importlib.util

    # 1. embed_parquet resume counts only newly encoded rows/tokens and validates manifest dimensions
    rows = [{"id": i, "body": f"doc {i}"} for i in range(6)]
    src = tmp_path / "in.parquet"
    pq.write_table(pa.Table.from_pylist(rows), src)
    out_dir = tmp_path / "emb_shards"

    class _CountingMrlEngine(_DummyEngine):
        max_dim = 16
        supports_dimensions = True

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def encode_corpus(self, tok_lists, mode=None):
            self.calls += 1
            return super().encode_corpus(tok_lists, mode=mode)

    eng1 = _CountingMrlEngine()
    s1 = embed_parquet(eng1, src, out_dir, shard_rows=3, dimensions=16, progress_every=0)
    assert s1.rows_emitted == 6 and s1.total_tokens > 0 and eng1.calls == 2
    assert (out_dir / ".embed_manifest.json").is_file()

    eng2 = _CountingMrlEngine()
    s2 = embed_parquet(eng2, src, out_dir, shard_rows=3, dimensions=16, resume=True, progress_every=0)
    assert eng2.calls == 0
    assert s2.rows_emitted == 0 and s2.total_tokens == 0

    # Changing dimensions invalidates manifest match -> re-encodes!
    eng3 = _CountingMrlEngine()
    s3 = embed_parquet(eng3, src, out_dir, shard_rows=3, dimensions=8, resume=True, progress_every=0)
    assert eng3.calls == 2 and s3.rows_emitted == 6

    # 2. Skill scripts: query_capability_matrix semantic --min-os & 6-chip --diff-only + ab_bench_harness exports
    scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "apple-silicon-ml-optimizer" / "scripts"
    q_spec = importlib.util.spec_from_file_location("qcm_test", scripts_dir / "query_capability_matrix.py")
    assert q_spec and q_spec.loader
    qcm = importlib.util.module_from_spec(q_spec)
    q_spec.loader.exec_module(qcm)
    cur = qcm.load_curated()

    vg_27_0 = [r for r in qcm.iter_table_rows(cur, "version_gates") if qcm.matches_filters(r, None, "27.0", False)]
    vg_27 = [r for r in qcm.iter_table_rows(cur, "version_gates") if qcm.matches_filters(r, None, "27", False)]
    assert len(vg_27_0) == len(vg_27) and len(vg_27_0) >= 6
    core_ai_26_2 = [r for r in qcm.iter_table_rows(cur, "core_ai") if qcm.matches_filters(r, None, "26.2", False)]
    assert len(core_ai_26_2) >= 14
    gpu_26_2 = [r for r in qcm.iter_table_rows(cur, "gpu_and_metal") if qcm.matches_filters(r, None, "26.2", False)]
    assert len(gpu_26_2) > 0
    # Six-chip diff-only catches M4 vs M4 Pro (e.g. Thunderbolt 4 vs 5 or ProRes count) even when M4 Pro == M5!
    assert qcm._row_has_chip_diff({"M4": "TB4", "M4 Pro": "TB5", "M4 Max": "TB5", "M5": "TB4", "M5 Pro": "TB5", "M5 Max": "TB5"}) is True
    assert qcm._row_has_chip_diff({"M4": "16", "M4 Pro": "16", "M4 Max": "16", "M5": "16", "M5 Pro": "16", "M5 Max": "16"}) is False

    ab_spec = importlib.util.spec_from_file_location("ab_test", scripts_dir / "ab_bench_harness.py")
    assert ab_spec and ab_spec.loader
    ab = importlib.util.module_from_spec(ab_spec)
    ab_spec.loader.exec_module(ab)
    for sym in ("get_unified_memory_footprint", "peak_footprint_gb", "wired_limit", "collect_env_telemetry"):
        assert hasattr(ab, sym)


def test_no_clt_xcrun_gate_and_quiescence_self_pid_exclusion_cpu(monkeypatch: Any) -> None:
    import os
    import subprocess
    from fast_shared import bench as fs_bench
    from fast_shared import hw_check as fs_hw

    calls: list[list[str]] = []

    def _fake_run_cmd(argv: list[str], *, timeout: float = 2.0) -> str:
        calls.append(list(argv))
        if argv[:2] == ["xcode-select", "-p"]:
            return ""
        if argv[:2] == ["sw_vers", "-productVersion"]:
            return "27.2.1"
        if argv[:2] == ["sw_vers", "-buildVersion"]:
            return "26B131"
        return ""

    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(fs_hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(fs_hw, "_run_cmd", _fake_run_cmd)
    assert fs_hw._has_apple_developer_tools() is False
    host = fs_hw.probe_host_hardware()
    assert host["xcode_sdk_version"] is None
    assert not any(c and c[0] == "xcrun" for c in calls), f"Unexpected xcrun call without CLT: {calls}"

    monkeypatch.setenv("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    assert fs_hw._has_apple_developer_tools() is True

    self_pid = os.getpid()
    parent_pid = os.getppid()
    fake_ps = "\n".join([
        "USER PID %CPU %MEM VSZ RSS TT STAT STARTED TIME COMMAND",
        f"user {self_pid} 85.0 1.0 100 200 ?? R 10:00 0:10 /venv/bin/python -c import mlx.core as mx",
        f"user {parent_pid} 45.0 0.5 100 200 ?? R 10:00 0:10 /venv/bin/python scripts/benchmark_monorepo.py",
        "user 999999 42.0 1.0 100 200 ?? R 10:00 0:10 /venv/bin/python -c import mlx.core as mx",
    ])
    monkeypatch.setattr(fs_bench.subprocess, "check_output", lambda *a, **kw: fake_ps)
    warnings = fs_bench.check_system_quiescence()
    assert len(warnings) == 1 and "PID=999999" in warnings[0], f"Unexpected quiescence warnings: {warnings}"


def test_core_ai_preflight_decoupled_and_cache_key_cpu(monkeypatch: Any) -> None:
    from fast_shared import hw_check as fs_hw

    base_26 = {
        "system": "Darwin",
        "machine": "arm64",
        "hw_model": "Mac16,8",
        "brand": "Apple M4 Pro",
        "mem_gb": 48,
        "cpu_cores": 14,
        "gpu_cores": 20,
        "ane_cores": 16,
        "gpu_family": "Apple9",
        "supports_apple9": True,
        "supports_apple10": False,
        "supports_metal4": True,
        "macos_version": "26.6",
        "mlx_version": "0.32.4",
        "mlx_enable_tf32": True,
        "mlx_enable_tf32_env": None,
        "mlx_metal_no_nax": False,
    }
    base_27 = {**base_26, "hw_model": "Mac17,2", "brand": "Apple M5", "mem_gb": 24, "cpu_cores": 10, "gpu_cores": 10, "gpu_family": "Apple10", "supports_apple10": True, "macos_version": "27.2.1"}

    # 1. Offline exporter on macOS 26.6 (`coreai-torch 0.4.3`, `coreai-opt 0.4.0`): export ready, runtime ineligible
    ca_26 = fs_hw.probe_core_ai_support(
        base_26,
        use_os_coreai=True,
        pkg_versions={"coreai-torch": "0.4.3", "coreai-opt": "0.4.0", "coreai-core": "1.0.0b3", "coreai": "0.4.3"},
        delegate_probe=lambda: True,
        coreai_build_path="",
        framework_exists=False,
    )
    assert ca_26["os_runtime_eligible"] is False
    assert ca_26["fp4_fp8_preset_eligible"] is False
    assert ca_26["coreai_torch_ge_041"] is True
    assert ca_26["export_ready"] is True
    assert ca_26["runtime_ready"] is False
    assert ca_26["delegate_supported"] is False

    # 2. Broken 0.4.0 export floor on macOS 27.2.1 -> export_ready False even though OS >= 27.2
    ca_040 = fs_hw.probe_core_ai_support(
        base_27,
        use_os_coreai=True,
        pkg_versions={"coreai-torch": "0.4.0", "coreai-opt": "0.3.0", "coreai-core": None, "coreai": None},
        coreai_build_path="",
        framework_exists=True,
    )
    assert ca_040["os_runtime_eligible"] is True
    assert ca_040["fp4_fp8_preset_eligible"] is True
    assert ca_040["coreai_torch_ge_041"] is False
    assert ca_040["export_ready"] is False
    assert ca_040["delegate_supported"] is None  # Missing != False
    assert ca_040["runtime_ready"] is False

    # 3. USE_OS_COREAI gate & live delegate gate on macOS 27.2.1
    ca_no_flag = fs_hw.probe_core_ai_support(
        base_27,
        use_os_coreai=False,
        pkg_versions={"coreai-torch": "0.4.3", "coreai-opt": "0.4.0", "coreai-core": "1.0.0b3", "coreai": "0.4.3"},
        delegate_probe=lambda: True,
    )
    assert ca_no_flag["export_ready"] is True and ca_no_flag["runtime_ready"] is False

    ca_full = fs_hw.probe_core_ai_support(
        base_27,
        use_os_coreai=True,
        pkg_versions={"coreai-torch": "0.4.3", "coreai-opt": "0.4.0", "coreai-core": "1.0.0b3", "coreai": "0.4.3"},
        delegate_probe=lambda: True,
        coreai_build_path="/usr/bin/coreai-build",
        framework_exists=True,
    )
    assert ca_full["export_ready"] is True and ca_full["runtime_ready"] is True
    assert ca_full["coreai_build_available"] is True

    # 4. preflight_silicon_env strict gates for TF32=0 & Core AI export/runtime
    hw_full = {**base_27, "core_ai": ca_full}
    pf_tf32_bad = fs_hw.preflight_silicon_env(hw=hw_full, check_tf32=True, require_tf32_off=True, strict=False)
    assert pf_tf32_bad["ok"] is False
    assert any("MLX_ENABLE_TF32=0" in e for e in pf_tf32_bad["errors"])

    hw_ok = {**hw_full, "mlx_enable_tf32": False, "mlx_enable_tf32_env": "0"}
    pf_ok = fs_hw.preflight_silicon_env(
        hw=hw_ok,
        require_tf32_off=True,
        require_coreai_export=True,
        require_coreai_runtime=True,
        strict=True,
    )
    assert pf_ok["ok"] is True
    assert pf_ok["coreai_export_ready"] is True
    assert pf_ok["coreai_runtime_ready"] is True

    hw_040 = {**base_27, "mlx_enable_tf32": False, "core_ai": ca_040}
    with pytest.raises(RuntimeError, match="preflight failed"):
        fs_hw.preflight_silicon_env(hw=hw_040, require_coreai_export=True, strict=True)

    # 5. Cache key sensitivity across model_hash, toolchain, min_os, arch, and spec_options
    k0 = fs_hw.coreai_specialization_cache_key_parts(
        model_hash="abc123",
        coreai_toolchain_ver="0.4.3",
        min_os="27.2",
        arch="arm64e",
        spec_options={"compute_units": "ALL"},
    )
    k_os = fs_hw.coreai_specialization_cache_key_parts(
        model_hash="abc123",
        coreai_toolchain_ver="0.4.3",
        min_os="27.3",
        arch="arm64e",
        spec_options={"compute_units": "ALL"},
    )
    k_opt = fs_hw.coreai_specialization_cache_key_parts(
        model_hash="abc123",
        coreai_toolchain_ver="0.4.3",
        min_os="27.2",
        arch="arm64e",
        spec_options={"compute_units": "CPU_ONLY"},
    )
    assert len({k0["cache_key_sha256"], k_os["cache_key_sha256"], k_opt["cache_key_sha256"]}) == 3


def test_summarize_accelerator_placement_and_coreml_plan_cpu() -> None:
    import fast_shared.hw_check as fs_hw

    class MLNeuralEngineComputeDevice:
        pass

    class MLGPUComputeDevice:
        pass

    class MLCPUComputeDevice:
        pass

    class _FakeOut:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeOp:
        def __init__(self, op_type: str, out_name: str, blocks: list | None = None) -> None:
            self.operator_name = op_type
            self.outputs = [_FakeOut(out_name)]
            self.blocks = blocks or []

    class _FakeBlock:
        def __init__(self, ops: list[_FakeOp]) -> None:
            self.operations = ops

    class _FakeFn:
        def __init__(self, block: _FakeBlock) -> None:
            self.block = block

    class _FakeStruct:
        def __init__(self, program: dict[str, _FakeFn]) -> None:
            self.program = program

    class _FakeUsage:
        def __init__(self, pref: object, sup: list[object]) -> None:
            self.preferred_compute_device = pref
            self.supported_compute_devices = sup

    class _FakeCost:
        def __init__(self, weight: float) -> None:
            self.weight = weight

    op1 = _FakeOp("conv", "c1")
    op2 = _FakeOp("softmax", "s1")
    op3 = _FakeOp("cast", "k1")
    nested = _FakeOp("while_loop", "w1", blocks=[_FakeBlock([op3])])
    usages = {
        id(op1): (_FakeUsage(MLNeuralEngineComputeDevice(), [MLNeuralEngineComputeDevice(), MLCPUComputeDevice()]), 0.7),
        id(op2): (_FakeUsage(MLNeuralEngineComputeDevice(), [MLNeuralEngineComputeDevice()]), 0.2),
        id(nested): (_FakeUsage(MLGPUComputeDevice(), [MLGPUComputeDevice(), MLCPUComputeDevice()]), 0.05),
        id(op3): (_FakeUsage(MLCPUComputeDevice(), [MLCPUComputeDevice()]), 0.05),
    }

    class _FakePlan:
        model_structure = _FakeStruct({"main": _FakeFn(_FakeBlock([op1, op2, nested]))})

        def get_compute_device_usage_for_mlprogram_operation(self, op: object) -> object:
            return usages[id(op)][0]

        def get_estimated_cost_for_mlprogram_operation(self, op: object) -> object:
            return _FakeCost(usages[id(op)][1])

    extracted = fs_hw.extract_coreml_compute_plan_ops(_FakePlan())
    assert [r["op_type"] for r in extracted] == ["conv", "softmax", "while_loop", "cast"]
    assert [r["preferred_device"] for r in extracted] == ["ane", "ane", "gpu", "cpu"]

    summary = fs_hw.summarize_accelerator_placement(extracted, target_device="ane", model_name="demo.mlmodelc")
    assert summary["total_ops"] == 4
    assert summary["target_ops"] == 2
    assert summary["target_share_pct"] == 50.0
    assert summary["target_cost_share_pct"] == 90.0
    assert summary["all_on_target"] is False
    assert summary["fallback_op_count"] == 2
    assert summary["decline_reason_counts"] == {"placed_on_gpu": 1, "placed_on_cpu": 1}

    coreai_records = [
        {"op_name": "lin_0", "op_type": "linear", "preferred_device": "neuralEngine"},
        {
            "op_name": "top_0",
            "op_type": "topk",
            "preferred_device": "cpu",
            "declined_placements": {"ane": "unsupported dynamic k"},
        },
    ]
    coreai_summary = fs_hw.summarize_accelerator_placement(coreai_records, target_device="ane", include_ops=True)
    assert coreai_summary["all_on_target"] is False
    assert coreai_summary["decline_reason_counts"] == {"unsupported dynamic k": 1}
    assert len(coreai_summary["operations"]) == 2

    # Bug 3: coremltools 9 MLComputePlan exposes `.program.functions` on MLModelStructureProgram, not a raw dict.
    class _FakeProgramObj:
        def __init__(self, functions: dict[str, _FakeFn]) -> None:
            self.functions = functions

    class _FakeObjStruct:
        def __init__(self, prog: _FakeProgramObj) -> None:
            self.program = prog

    class _FakeObjPlan(_FakePlan):
        model_structure = _FakeObjStruct(_FakeProgramObj({"main": _FakeFn(_FakeBlock([op1, op2, nested]))}))

    extracted_obj = fs_hw.extract_coreml_compute_plan_ops(_FakeObjPlan())
    assert [r["op_type"] for r in extracted_obj] == ["conv", "softmax", "while_loop", "cast"]


def test_coreml_bundle_placement_defaults_to_all_compute_units_cpu(monkeypatch: Any, tmp_path: Path) -> None:
    import sys
    import types
    import fast_shared.coreai_pipeline as cap

    seen: dict[str, Any] = {}

    class _FakeCU:
        ALL = "CU_ALL"
        CPU_AND_NE = "CU_CPU_AND_NE"

    def _fake_load_from_path(path: str, compute_units: Any = None) -> object:
        seen["path"] = path
        seen["compute_units"] = compute_units
        return types.SimpleNamespace(model_structure=None)

    fake_ct = types.SimpleNamespace(
        ComputeUnit=_FakeCU,
        models=types.SimpleNamespace(
            compute_plan=types.SimpleNamespace(MLComputePlan=types.SimpleNamespace(load_from_path=_fake_load_from_path))
        ),
        utils=types.SimpleNamespace(compile_model=lambda p: p),
    )
    monkeypatch.setitem(sys.modules, "coremltools", fake_ct)
    bundle = tmp_path / "m.mlmodelc"
    bundle.mkdir()
    out = cap.inspect_coreml_bundle_placement(bundle, target_device="ane")
    assert seen["compute_units"] == "CU_ALL"
    assert out["total_ops"] == 0


def test_coreai_pipeline_coreml_w4_block_guard_and_fp16_cast_cpu(monkeypatch: Any, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import fast_shared.coreai_pipeline as cap

    # Bug 4: w4_per_block on coreml must require explicit allow_experimental_coreml_w4_block=True.
    with pytest.raises(ValueError, match="allow_experimental_coreml_w4_block"):
        cap.build_coreai_opt_config(preset="w4_per_block", export_backend="coreml")

    assert cap._resolve_coreai_compute_dtype("coreai", "w8") == "fp16"
    assert cap._resolve_coreai_compute_dtype("coreai", "fp16") == "fp16"
    assert cap._resolve_coreai_compute_dtype("coreai", "fp4_fp8") == "fp32"
    assert cap._resolve_coreai_compute_dtype("coreml", "w8") == "fp32"
    assert cap._resolve_coreai_compute_dtype("coreai", "w8", "fp32") == "fp32"

    pytest.importorskip("coreai_opt")
    _, meta_opt = cap.build_coreai_opt_config(
        preset="w4_per_block",
        export_backend="coreml",
        allow_experimental_coreml_w4_block=True,
    )
    assert meta_opt["preset"] == "w4_per_block"

    # Default preset sweep on coreml excludes w4_per_block unless explicitly opted in.
    seen_presets: list[str] = []

    def _fake_export(**kwargs: Any) -> cap.CoreAiExportResult:
        seen_presets.append(kwargs["preset"])
        return cap.CoreAiExportResult(
            backend=kwargs["export_backend"],
            preset=kwargs["preset"],
            asset_path=tmp_path / f"{kwargs['stem']}.mlpackage",
            cache_key_parts={},
            cache_key_sha256="deadbeef",
        )

    monkeypatch.setattr(cap, "run_coreai_opt_calibration_and_export", _fake_export)
    cap.run_coreai_opt_preset_sweep(
        module_factory=lambda: nn.Linear(4, 4),
        sample_inputs_factory=lambda _: (torch.randn(1, 4),),
        out_dir=tmp_path / "sweep",
        stem_prefix="demo",
        model_name="Demo",
        input_names=["x"],
        output_names=["y"],
        export_backend="coreml",
    )
    assert seen_presets == ["fp16", "w8", "w4"]

    # Bug 7: FP16 cast casts floating tensors/buffers while preserving integer/bool inputs & buffers.
    monkeypatch.undo()
    casted = cap._cast_floating_tensors(
        (torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.int64), {"b": torch.tensor([True])}),
        torch.float16,
    )
    assert casted[0].dtype == torch.float16
    assert casted[1].dtype == torch.int64
    assert casted[2]["b"].dtype == torch.bool

    pytest.importorskip("coreai_torch")

    class _ToyMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale_buf", torch.ones(1, 16, 1, 1))
            self.register_buffer("idx_buf", torch.arange(4, dtype=torch.int64))
            self.c = nn.Conv2d(16, 16, 1)

        def forward(self, x: torch.Tensor, tok_ids: torch.Tensor) -> torch.Tensor:
            return self.c(x * self.scale_buf) + float(self.idx_buf.shape[0]) + tok_ids.float().mean() * 0.0

    res_f16 = cap.run_coreai_opt_calibration_and_export(
        module_factory=_ToyMod,
        sample_inputs_factory=lambda _: (torch.randn(1, 16, 4, 4), torch.ones(1, 2, dtype=torch.int64)),
        out_dir=tmp_path / "coreai_f16",
        stem="toy_w8_f16",
        model_name="Toy",
        input_names=["x", "tok_ids"],
        output_names=["y"],
        export_backend="coreai",
        preset="w8",
    )
    assert res_f16.asset_summary is not None
    assert "Float16" in res_f16.asset_summary["compute_types"]
    assert res_f16.sensitivity_report is not None
    assert res_f16.sensitivity_report["compute_dtype"] == "fp16"

    res_f32 = cap.run_coreai_opt_calibration_and_export(
        module_factory=_ToyMod,
        sample_inputs_factory=lambda _: (torch.randn(1, 16, 4, 4), torch.ones(1, 2, dtype=torch.int64)),
        out_dir=tmp_path / "coreai_f32",
        stem="toy_w8_f32",
        model_name="Toy",
        input_names=["x", "tok_ids"],
        output_names=["y"],
        export_backend="coreai",
        preset="w8",
        compute_dtype="fp32",
    )
    assert res_f32.sensitivity_report is not None
    assert res_f32.sensitivity_report["compute_dtype"] == "fp32"
    assert res_f32.cache_key_sha256 != res_f16.cache_key_sha256


def test_grammar_helper_none_whitespace_and_explicit_separators_cpu() -> None:
    llg = pytest.importorskip("llguidance")
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.grammar import GrammarHelper

    assert llg is not None
    # ByteLevel maps ASCII space (0x20) to 'Ġ' (U+0120).
    vocab = {
        "<|endoftext|>": 0,
        "{": 1,
        '"': 2,
        "a": 3,
        ":": 4,
        "1": 5,
        "}": 6,
        "\u0120": 7,
        ",": 8,
        "b": 9,
        "2": 10,
    }
    bt = Tokenizer(BPE(vocab=vocab, merges=[]))
    bt.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bt.decoder = ByteLevelDecoder(add_prefix_space=False)
    ltok = llg.LLTokenizer(bt.to_str(), eos_token=0)
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }

    # whitespace_flexible=None selects strict (",", ": ") separators without raising TypeError.
    gh_none = GrammarHelper(ltok, schema, whitespace_flexible=None, eos_token_id=0)
    assert gh_none.separators == (",", ": ")
    assert gh_none.whitespace_flexible is False
    assert gh_none.upfront_ff == [1, 2, 3, 2, 4, 7]
    assert gh_none.validate_token_sequence(bt.encode('{"a": 1,"b": 2}').ids)
    assert not gh_none.validate_token_sequence(bt.encode('{"a":1,"b":2}').ids)

    # whitespace_flexible=False defaults to compact (",", ":") separators.
    gh_compact = GrammarHelper(ltok, schema, whitespace_flexible=False, eos_token_id=0)
    assert gh_compact.separators == (",", ":")
    assert gh_compact.whitespace_flexible is False
    assert gh_compact.validate_token_sequence(bt.encode('{"a":1,"b":2}').ids)
    assert not gh_compact.validate_token_sequence(bt.encode('{"a": 1,"b": 2}').ids)

    # Explicit separators override the default cleanly.
    gh_custom = GrammarHelper(ltok, schema, whitespace_flexible=False, separators=(", ", ": "), eos_token_id=0)
    assert gh_custom.separators == (", ", ": ")
    assert gh_custom.whitespace_flexible is False
    assert gh_custom.validate_token_sequence(bt.encode('{"a": 1, "b": 2}').ids)
    assert not gh_custom.validate_token_sequence(bt.encode('{"a": 1,"b": 2}').ids)


def test_server_schema_key_batcher_whitespace_compact_and_raw_dispatch_cpu() -> None:
    import concurrent.futures
    import threading
    import time
    from types import SimpleNamespace

    from fast_shared.server import (
        ChatMicroBatcher,
        OpenAIChatAdapter,
        dispatch_chat_completion,
        dispatch_completion,
        extract_request_param,
        schema_cache_key,
    )

    # 1. schema_cache_key preserves insertion key order (top-level and nested properties).
    s_ab = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    s_ba = {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string"}}}
    assert schema_cache_key(s_ab) != schema_cache_key(s_ba)
    assert schema_cache_key(None) is None
    assert schema_cache_key('{"b":1,"a":2}') == '{"b":1,"a":2}'

    # 2. extract_request_param cleanly distinguishes missing key vs explicit JSON null.
    sentinel = object()
    assert extract_request_param({}, "whitespace_flexible", sentinel, allow_none=True) is sentinel
    assert extract_request_param({"whitespace_flexible": None}, "whitespace_flexible", True, allow_none=True) is None
    assert (
        extract_request_param(
            {"extra_body": {"whitespace_flexible": None}},
            "whitespace_flexible",
            True,
            allow_none=True,
        )
        is None
    )
    assert extract_request_param({"max_tokens": None}, "max_tokens", 512) == 512

    # 3. ChatMicroBatcher does NOT coalesce across whitespace_flexible in (None, False, True),
    #    while coalescing identical whitespace_flexible=None requests.
    class _RecordingEngine:
        model_name = "rec-llm"
        ane_available = False
        default_mode = "gpu"

        def __init__(self, delay: float = 0.0) -> None:
            self.delay = delay
            self.calls: list[tuple[int, dict[str, object]]] = []
            self._lock = threading.Lock()

        def generate_batch(self, prompts, **kwargs):
            with self._lock:
                self.calls.append((len(prompts), dict(kwargs)))
            if self.delay > 0.0:
                time.sleep(self.delay)
            items = [
                SimpleNamespace(
                    text="{}",
                    prompt_tokens=4,
                    completion_tokens=2,
                    finish_reason="stop",
                )
                for _ in prompts
            ]
            return SimpleNamespace(items=items, stats={"docs": len(prompts)})

    eng_batch = _RecordingEngine(delay=0.02)
    batcher = ChatMicroBatcher(eng_batch, coalesce_ms=30.0, max_batch_size=16)

    def _sub(ws: bool | None, tag: str) -> None:
        batcher.submit(
            [{"role": "user", "content": tag}],
            json_schema=s_ab,
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            whitespace_flexible=ws,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(_sub, None, "n1"),
            pool.submit(_sub, None, "n2"),
            pool.submit(_sub, False, "f1"),
            pool.submit(_sub, True, "t1"),
        ]
        for fut in futs:
            fut.result()

    ws_seen = [kw["whitespace_flexible"] for _, kw in eng_batch.calls]
    assert set(ws_seen) == {None, False, True}
    # Every call must stay homogeneous; total prompts across calls == 4; False and True each have size 1.
    assert sum(n for n, _ in eng_batch.calls) == 4
    for n, kw in eng_batch.calls:
        if kw["whitespace_flexible"] in (False, True):
            assert n == 1
        assert "compact" not in kw

    # 4. OpenAIChatAdapter does NOT inject compact=True, respects explicit compact and whitespace_flexible=None.
    eng_adapt = _RecordingEngine()
    adapter = OpenAIChatAdapter(eng_adapt, default_whitespace_flexible=True)

    status, _, _ = adapter.handle_chat_completion({"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    assert "compact" not in eng_adapt.calls[-1][1]
    assert eng_adapt.calls[-1][1]["whitespace_flexible"] is True

    status, _, _ = adapter.handle_chat_completion(
        {"messages": [{"role": "user", "content": "hi"}], "whitespace_flexible": None}
    )
    assert status == 200
    assert "compact" not in eng_adapt.calls[-1][1]
    assert eng_adapt.calls[-1][1]["whitespace_flexible"] is None

    status, _, _ = adapter.handle_chat_completion(
        {"messages": [{"role": "user", "content": "hi"}], "extra_body": {"whitespace_flexible": None, "compact": True}}
    )
    assert status == 200
    assert eng_adapt.calls[-1][1]["whitespace_flexible"] is None
    assert eng_adapt.calls[-1][1]["compact"] is True

    status, _, _ = adapter.handle_completion({"prompt": "hi"})
    assert status == 200
    assert "compact" not in eng_adapt.calls[-1][1]
    assert eng_adapt.calls[-1][1]["whitespace_flexible"] is True

    status, _, _ = adapter.handle_completion({"prompt": "hi", "compact": False, "whitespace_flexible": None})
    assert status == 200
    assert eng_adapt.calls[-1][1]["compact"] is False
    assert eng_adapt.calls[-1][1]["whitespace_flexible"] is None

    # 5. Raw-engine dispatch_chat_completion / dispatch_completion omit whitespace_flexible unless set.
    eng_raw = _RecordingEngine()
    st_c, _, _ = dispatch_chat_completion(eng_raw, {"messages": [{"role": "user", "content": "a"}]})
    assert st_c == 200
    assert "whitespace_flexible" not in eng_raw.calls[-1][1]
    assert "compact" not in eng_raw.calls[-1][1]

    st_p, _, _ = dispatch_completion(eng_raw, {"prompt": "a"})
    assert st_p == 200
    assert "whitespace_flexible" not in eng_raw.calls[-1][1]
    assert "compact" not in eng_raw.calls[-1][1]

    st_null, _, _ = dispatch_chat_completion(
        eng_raw, {"messages": [{"role": "user", "content": "a"}], "whitespace_flexible": None}
    )
    assert st_null == 200
    assert "whitespace_flexible" in eng_raw.calls[-1][1]
    assert eng_raw.calls[-1][1]["whitespace_flexible"] is None

    st_p_false, _, _ = dispatch_completion(eng_raw, {"prompt": "a", "whitespace_flexible": False, "compact": True})
    assert st_p_false == 200
    assert eng_raw.calls[-1][1]["whitespace_flexible"] is False
    assert eng_raw.calls[-1][1]["compact"] is True

    # Raw engine declaring `default_whitespace_flexible = None` preserves that default when request omits it.
    class _GemmaLikeRawEngine(_RecordingEngine):
        default_whitespace_flexible = None

    eng_gemma_raw = _GemmaLikeRawEngine()
    dispatch_chat_completion(eng_gemma_raw, {"messages": [{"role": "user", "content": "g"}]})
    assert eng_gemma_raw.calls[-1][1]["whitespace_flexible"] is None
    dispatch_completion(eng_gemma_raw, {"prompt": "g"})
    assert eng_gemma_raw.calls[-1][1]["whitespace_flexible"] is None
    dispatch_chat_completion(
        eng_gemma_raw, {"messages": [{"role": "user", "content": "g"}], "whitespace_flexible": True}
    )
    assert eng_gemma_raw.calls[-1][1]["whitespace_flexible"] is True


def test_truncate_and_normalize_full_width_and_zero_safe_cpu() -> None:
    raw = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    raw_copy = raw.copy()

    # Full-width (eff_dim == arr.shape[-1]) and dim=None MUST L2-normalize without mutating input
    norm_eq = truncate_and_normalize(raw, 4)
    norm_none = truncate_and_normalize(raw, None)
    assert np.array_equal(raw, raw_copy)
    assert np.allclose(norm_eq[0], [0.6, 0.8, 0.0, 0.0], atol=1e-6)
    assert np.allclose(norm_none[0], [0.6, 0.8, 0.0, 0.0], atol=1e-6)
    assert np.allclose(norm_eq[1], 0.0, atol=1e-12)
    assert np.all(np.isfinite(norm_eq))

    # 1D vector at full width
    v1 = np.array([6.0, 8.0], dtype=np.float16)
    out1 = truncate_and_normalize(v1, 2)
    assert out1.dtype == np.float32 and np.allclose(out1, [0.6, 0.8], atol=1e-6)


def test_coreai_pipeline_fp4_fp8_specs_and_qwen35_small_gdn_ban_cpu(monkeypatch: Any) -> None:
    import sys
    import types
    import fast_shared.coreai_pipeline as cap

    # 1. Blanket fp4_fp8 on Qwen3.5-{0.8B, 2B} is hard-blocked with ValueError (even without torch installed)
    for bad_model in ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B-Instruct", "Qwen3.5-2B", "qwen3_5_2b", "kev-0.8b"):
        with pytest.raises(ValueError, match="fp4_fp8"):
            cap.build_coreai_opt_config(
                preset="fp4_fp8",
                export_backend="coreai",
                model_name=bad_model,
                os_version="27.2.1",
            )

    # 2. Mock torch & coreai_opt to verify float4_e2m1fn_x2 weights + float8_e4m3fn activation specs on Qwen3.5-4B
    class _FakeSpec:
        def __init__(self, *, dtype: Any) -> None:
            self.dtype = dtype

    class _FakeModCfg:
        def __init__(
            self,
            *,
            op_state_spec: dict[str, Any] | None = None,
            op_input_spec: dict[str, Any] | None = None,
            op_output_spec: dict[str, Any] | None = None,
        ) -> None:
            self.op_state_spec = op_state_spec or {}
            self.op_input_spec = op_input_spec or {}
            self.op_output_spec = op_output_spec or {}

    class _FakeQuantCfg:
        def __init__(self) -> None:
            self.global_cfg: Any = None
            self.module_cfgs: dict[str, Any] = {}

        def set_global(self, mcfg: Any) -> "_FakeQuantCfg":
            self.global_cfg = mcfg
            return self

        def set_module_name(self, name: str, mcfg: Any) -> "_FakeQuantCfg":
            self.module_cfgs[name] = mcfg
            return self

    class _FakeEB:
        class _Item:
            def __init__(self, value: str) -> None:
                self.value = value

        CoreML = _Item("coreml")
        CoreAI = _Item("coreai")
        Torch = _Item("torch")

    fake_torch = types.SimpleNamespace(
        float4_e2m1fn_x2="torch.float4_e2m1fn_x2",
        float8_e4m3fn="torch.float8_e4m3fn",
        float16="torch.float16",
    )
    fake_qmod = types.SimpleNamespace(
        ModuleQuantizerConfig=_FakeModCfg,
        QuantizationSpec=_FakeSpec,
        QuantizerConfig=_FakeQuantCfg,
    )
    fake_coreai_opt = types.SimpleNamespace(ExportBackend=_FakeEB, quantization=fake_qmod)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "coreai_opt", fake_coreai_opt)
    monkeypatch.setitem(sys.modules, "coreai_opt.quantization", fake_qmod)

    cfg, meta = cap.build_coreai_opt_config(
        preset="fp4_fp8",
        export_backend="coreai",
        model_name="Qwen/Qwen3.5-4B",
        os_version="27.2.1",
        excluded_modules=["lm_head"],
    )
    assert cfg.global_cfg is not None
    assert cfg.global_cfg.op_state_spec["weight"].dtype == "torch.float4_e2m1fn_x2"
    assert cfg.global_cfg.op_input_spec["*"].dtype == "torch.float8_e4m3fn"
    assert cfg.global_cfg.op_output_spec["*"].dtype == "torch.float8_e4m3fn"
    assert cfg.module_cfgs["lm_head"].op_state_spec == {"*": None}
    assert meta["weight_dtype"] == "torch.float4_e2m1fn_x2"
    assert meta["activation_dtype"] == "torch.float8_e4m3fn"


def test_pipeline_llm_scale_defaults_resume_skip_io_macro_sort_and_extractconf_cpu(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import inspect
    from types import SimpleNamespace
    import fast_shared.pipeline_llm as pllm

    # 1. Verify 10M-doc scale defaults on extract_parquet & extract_jsonl
    sig_pq = inspect.signature(pllm.extract_parquet)
    assert sig_pq.parameters["whitespace_flexible"].default is None
    assert sig_pq.parameters["use_wired_limit"].default is True
    assert sig_pq.parameters["skip_empty"].default is False
    assert sig_pq.parameters["max_chars"].default is None
    assert sig_pq.parameters["emit_extractconf"].default is True
    assert sig_pq.parameters["macro_window_rows"].default == 4096

    sig_jl = inspect.signature(pllm.extract_jsonl)
    assert sig_jl.parameters["whitespace_flexible"].default is None
    assert sig_jl.parameters["use_wired_limit"].default is True
    assert sig_jl.parameters["emit_extractconf"].default is True

    # 2. Checkpoint provenance.json compact wire detection vs default None (": ")
    ckpt_compact = tmp_path / "ckpt_compact"
    ckpt_compact.mkdir()
    (ckpt_compact / "provenance.json").write_text(
        json.dumps({"json_separators": [",", ":"]}),
        encoding="utf-8",
    )

    schema = {
        "type": "object",
        "properties": {"idx": {"type": "integer"}, "len": {"type": "integer"}},
        "required": ["idx", "len"],
    }

    class _MacroEngine:
        def __init__(self, model_dir: Path | None = None) -> None:
            self.model_dir = model_dir
            self.default_max_b_slots = 64
            self.seen_batches: list[list[str]] = []
            self.seen_kwargs: list[dict[str, Any]] = []

        def generate_batch(self, prompts: list[str], **kwargs: Any) -> Any:
            self.seen_batches.append(list(prompts))
            self.seen_kwargs.append(dict(kwargs))
            items = []
            for p in prompts:
                row_id = int(p.split("|", 1)[0])
                is_uncertain = row_id == 3
                items.append(
                    SimpleNamespace(
                        text=json.dumps({"idx": row_id, "len": len(p)}),
                        parsed={"idx": row_id, "len": len(p)},
                        valid_json=True,
                        valid_schema=True,
                        finish_reason="stop",
                        prompt_tokens=max(1, len(p) // 4),
                        completion_tokens=6,
                        telemetry={
                            "mean_h_pre": 2.6 if is_uncertain else 0.03,
                            "mean_neg_log_z_valid": 3.2 if is_uncertain else 0.0,
                            "mean_top2_margin": 0.05 if is_uncertain else 6.0,
                        },
                    )
                )
            return SimpleNamespace(
                items=items,
                stats={
                    "docs": len(items),
                    "valid_json_count": len(items),
                    "valid_schema_count": len(items),
                    "prompt_tokens": sum(it.prompt_tokens for it in items),
                    "completion_tokens": 6 * len(items),
                    "decode_sec": 0.01,
                },
            )

    # Include >8000 char row (verifies max_chars=None does not truncate), empty row (skip_empty=False), and mixed lengths
    long_body = "2|" + ("x" * 9500)
    rows = [
        {"id": 0, "body_text": "0|short"},
        {"id": 1, "body_text": "   \n\t  "},  # deterministic skipped_empty row, 0 LLM calls!
        {"id": 2, "body_text": long_body},
        {"id": 3, "body_text": "3|" + ("y" * 400)},
    ]
    in_pq = tmp_path / "macro_in.parquet"
    pq.write_table(pa.Table.from_pylist(rows), in_pq, row_group_size=2)

    out_dir = tmp_path / "macro_out"
    eng_default = _MacroEngine()
    summary = pllm.extract_parquet(
        eng_default,
        in_pq,
        out_dir,
        system_prompt="sys",
        json_schema=schema,
        shard_rows=100,
    )
    assert summary.docs_processed == 4
    # Only the 3 non-empty rows went to LLM, sorted by length (longest first) inside the macro-window!
    assert len(eng_default.seen_batches) == 1
    sent = eng_default.seen_batches[0]
    assert len(sent) == 3
    assert [len(s) for s in sent] == sorted([len(s) for s in sent], reverse=True)
    assert eng_default.seen_kwargs[0]["whitespace_flexible"] is None

    # Output table preserves exact 1:1 input row order [0, 1, 2, 3] and full >8000 length!
    out_tbl = pq.read_table(out_dir / "part-00000.parquet")
    assert out_tbl.column("id").to_pylist() == [0, 1, 2, 3]
    fr_list = out_tbl.column("finish_reason").to_pylist()
    assert fr_list == ["stop", "skipped_empty", "stop", "stop"]
    resp_list = out_tbl.column("response_text").to_pylist()
    assert resp_list[1] == ""
    assert json.loads(resp_list[2]) == {"idx": 2, "len": len(long_body)}
    risks = out_tbl.column("extractconf_risk").to_pylist()
    assert all(0.0 <= r <= 1.0 for r in risks)
    assert risks[3] > risks[0]  # High telemetry conflict on row 3 produces higher risk than clean row 0

    # 3. --resume skips reading/decompressing `body_text` on completed row groups!
    cols_requested: list[list[str] | None] = []
    orig_read_rg = pllm.pq.ParquetFile.read_row_group
    orig_iter_batches = pllm.pq.ParquetFile.iter_batches

    def _spy_read_rg(self_pf: Any, i: int, columns: Any = None, *a: Any, **kw: Any) -> Any:
        if Path(getattr(self_pf, "where", in_pq)) == in_pq or "body_text" in self_pf.schema_arrow.names:
            cols_requested.append(list(columns) if columns is not None else None)
        return orig_read_rg(self_pf, i, columns=columns, *a, **kw)

    def _spy_iter_batches(self_pf: Any, *a: Any, **kw: Any) -> Any:
        if "body_text" in self_pf.schema_arrow.names:
            cols = kw.get("columns")
            cols_requested.append(list(cols) if cols is not None else None)
        return orig_iter_batches(self_pf, *a, **kw)

    monkeypatch.setattr(pllm.pq.ParquetFile, "read_row_group", _spy_read_rg)
    monkeypatch.setattr(pllm.pq.ParquetFile, "iter_batches", _spy_iter_batches)

    eng_resume = _MacroEngine()
    resumed_sum = pllm.extract_parquet(
        eng_resume,
        in_pq,
        out_dir,
        system_prompt="sys",
        json_schema=schema,
        resume=True,
    )
    assert resumed_sum.docs_processed == 0
    assert eng_resume.seen_batches == []
    assert len(cols_requested) >= 1
    assert all(c is not None and "body_text" not in c for c in cols_requested)
    monkeypatch.undo()

    # 4. Compact provenance.json flips whitespace_flexible=None -> False
    eng_compact = _MacroEngine(model_dir=ckpt_compact)
    in_jl = tmp_path / "compact.jsonl"
    out_jl = tmp_path / "compact_out.jsonl"
    in_jl.write_text('{"id": 0, "text": "0|hi"}\n', encoding="utf-8")
    pllm.extract_jsonl(eng_compact, in_jl, out_jl, system_prompt="sys", json_schema=schema)
    assert eng_compact.seen_kwargs[0]["whitespace_flexible"] is False
    jl_row = json.loads(out_jl.read_text(encoding="utf-8").strip())
    assert "extractconf_risk" in jl_row


def test_runtime_lessons_profiles_hetero2_queue_thermal_and_sse_streaming_cpu(monkeypatch: Any) -> None:
    import http.client
    import threading
    import time
    from types import SimpleNamespace

    from fast_shared.runtime_lessons import (
        DynamicSiliconPolicy,
        Hetero2WorkStealingQueue,
        poll_macos_thermal_state,
    )
    from fast_shared.server import ManagedHTTPServer, OpenAIChatAdapter, iter_chat_sse_events, make_chat_handler

    hw_m5 = {
        "system": "Darwin",
        "machine": "arm64",
        "hw_model": "Mac17,2",
        "brand": "Apple M5",
        "gpu_family": "Apple10",
        "supports_apple9": True,
        "supports_apple10": True,
        "supports_metal4": True,
        "mem_gb": 24,
        "macos_version": "27.2.1",
        "mlx_enable_tf32": True,
        "mlx_metal_no_nax": False,
    }

    # 1. DynamicSiliconPolicy profile="backfill" vs profile="interactive" & mlx_cache_limit_mb=256
    pol_default = DynamicSiliconPolicy(hw=hw_m5)
    assert pol_default.profile == "backfill"
    assert pol_default.mlx_cache_limit_mb == 256
    assert pol_default.drafter_budget_mb == 0.0

    pol_bf = DynamicSiliconPolicy(hw=hw_m5, profile="backfill")
    plan_bf = pol_bf.resolve(batch_size=1)
    assert plan_bf.mtp_enabled is False
    assert plan_bf.should_load_mtp_drafter is False
    assert plan_bf.estimated_drafter_mem_mb == 0.0
    assert plan_bf.mlx_cache_limit_mb == 256

    pol_int = DynamicSiliconPolicy(hw=hw_m5, profile="interactive")
    assert pol_int.profile == "interactive"
    assert pol_int.drafter_budget_mb > 0.0
    plan_int = pol_int.resolve(batch_size=1)
    assert plan_int.mtp_enabled is True
    assert plan_int.should_load_mtp_drafter is True
    assert plan_int.estimated_drafter_mem_mb > 0.0

    with pytest.raises(ValueError, match="profile"):
        DynamicSiliconPolicy(hw=hw_m5, profile="invalid")  # type: ignore[arg-type]

    # 2. Persistent Hetero2WorkStealingQueue: GPU steals longest from head, ANE steals shortest <= 512 from tail
    wq = Hetero2WorkStealingQueue()
    wq.enqueue_batch(["s32", "s1024", "s64", "s768", "s256"], lengths=[32, 1024, 64, 768, 256])
    gpu_wave = wq.pop_gpu_batch(2)
    assert [it.payload for it in gpu_wave] == ["s1024", "s768"]
    ane_item = wq.pop_ane_item(max_seq_len=512)
    assert ane_item is not None and ane_item.payload == "s32"
    # Persistent across batches: remaining ["s256", "s64"] stay queued when a second batch arrives!
    wq.enqueue_batch(["s2048", "s16"], lengths=[2048, 16])
    assert wq.pop_gpu_batch(1)[0].payload == "s2048"
    assert wq.pop_ane_item(max_seq_len=512).payload == "s16"  # type: ignore[union-attr]

    # Order-preserving concurrent execution across GPU + ANE workers
    wq_exec = Hetero2WorkStealingQueue()
    inputs = [f"item-{i}" for i in range(12)]
    lens = [64 + (i * 73) % 900 for i in range(12)]
    results = wq_exec.run_concurrent(
        inputs,
        lengths=lens,
        gpu_fn=lambda batch: [f"G:{x.payload}" for x in batch],
        ane_fn=lambda batch: [f"A:{x.payload}" for x in batch],
        gpu_batch_size=3,
        ane_batch_size=1,
        ane_max_seq_len=512,
        ane_workers=2,
    )
    assert [r.split(":", 1)[1] for r in results] == inputs

    # 3. macOS thermalState polling helper (live + pmset fallback)
    live_therm = poll_macos_thermal_state()
    assert live_therm.level in ("nominal", "fair", "serious", "critical")
    assert 0 <= live_therm.raw_state <= 3

    import fast_shared.runtime_lessons as rl
    monkeypatch.setattr(rl, "_read_foundation_thermal_state", lambda: None)
    monkeypatch.setattr(
        rl,
        "_read_pmset_thermal_output",
        lambda: "Note: Thermal warning level has been recorded\n\tCPU_Speed_Limit \t= 65\n",
    )
    pmset_therm = poll_macos_thermal_state()
    assert pmset_therm.source == "pmset"
    assert pmset_therm.level in ("serious", "critical")
    assert pmset_therm.throttle_recommended is True
    monkeypatch.undo()

    # 4. Real incremental SSE token streaming via `generate_chat_stream` + `ttft_ms` recording
    second_tok_gate = threading.Event()

    class _StreamCapableEngine:
        model_name = "stream-llm"
        ane_available = False
        default_mode = "gpu"

        def generate_batch(self, prompts: list[Any], **kwargs: Any) -> Any:
            return SimpleNamespace(
                items=[SimpleNamespace(text="buffered", prompt_tokens=4, completion_tokens=1, finish_reason="stop")],
                stats={"docs": 1},
            )

        def generate_chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
            yield {"delta": "Hello", "prompt_tokens": 5}
            second_tok_gate.wait(timeout=2.0)
            yield {"delta": " world", "completion_tokens": 2, "finish_reason": "stop"}

    stream_eng = _StreamCapableEngine()
    stream_adapter = OpenAIChatAdapter(stream_eng)
    assert callable(getattr(stream_adapter, "generate_chat_stream", None))

    with ManagedHTTPServer(make_chat_handler(stream_adapter), host="127.0.0.1", port=0) as srv:
        conn = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": "stream-llm",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        # Read role chunk + first token chunk BEFORE releasing `second_tok_gate`!
        # If SSE were buffered, readline() here would block 2.0s until `second_tok_gate` timed out!
        t_first_start = time.perf_counter()
        seen_hello = False
        while True:
            line = resp.fp.readline().decode("utf-8")
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                payload = json.loads(line[len("data: ") :])
                delta = payload["choices"][0].get("delta", {})
                if delta.get("content") == "Hello":
                    seen_hello = True
                    break
        first_wait_sec = time.perf_counter() - t_first_start
        assert seen_hello is True
        assert first_wait_sec < 0.5  # Proves true incremental streaming before completion finishes!

        second_tok_gate.set()
        rest = resp.read().decode("utf-8")
        conn.close()
        assert " world" in rest
        assert "data: [DONE]" in rest
        assert stream_adapter.last_stream_stats is not None
        assert stream_adapter.last_stream_stats["ttft_ms"] >= 0.0

    # Direct `iter_chat_sse_events` fallback on completed item still returns valid OpenAI chunks + ttft_ms
    fb_stats: dict[str, Any] = {}
    fb_events = list(
        iter_chat_sse_events(
            "m",
            SimpleNamespace(text="ok", prompt_tokens=3, completion_tokens=1, finish_reason="stop"),
            include_usage=True,
            stats_out=fb_stats,
        )
    )
    assert len(fb_events) == 3
    assert fb_events[1]["choices"][0]["delta"]["content"] == "ok"
    assert fb_events[-1]["usage"]["total_tokens"] == 4
    assert "ttft_ms" in fb_stats


def test_prune_optional_nullable_fields_and_qat_provenance_hook_cpu(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from fast_shared import prune_optional_nullable_fields, validate_json_against_schema
    import fast_shared.pipeline_llm as pllm

    full_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "string"},
            "subtitle": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["source"],
            },
        },
        "required": ["id", "title", "meta"],
    }

    pruned = prune_optional_nullable_fields(full_schema)
    # Original schema must not be mutated
    assert set(full_schema["properties"].keys()) == {"id", "title", "subtitle", "tags", "meta"}
    # Optional top-level ("subtitle", "tags") and nested ("notes") keys are pruned; required keys preserved
    assert set(pruned["properties"].keys()) == {"id", "title", "meta"}
    assert pruned["additionalProperties"] is False
    assert set(pruned["properties"]["meta"]["properties"].keys()) == {"source"}
    assert pruned["properties"]["meta"]["additionalProperties"] is False

    # Instance matching pruned schema remains 100% valid against original unpruned schema!
    sample_out = {"id": 1, "title": "Clean Output", "meta": {"source": "sensor"}}
    validate_json_against_schema(sample_out, pruned)
    validate_json_against_schema(sample_out, full_schema)

    # Also supports OpenAI-style required nullable fields when prune_required_nullable=True
    strict_nullable_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "opt_note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["id", "opt_note"],
    }
    pruned_strict = prune_optional_nullable_fields(strict_nullable_schema, prune_required_nullable=True)
    assert set(pruned_strict["properties"].keys()) == {"id"}
    assert pruned_strict["required"] == ["id"]

    # Pipeline opt-in `compact_optional_schema=True` and `provenance.json` hook
    qat_ckpt = tmp_path / "qat_ckpt"
    qat_ckpt.mkdir()
    (qat_ckpt / "provenance.json").write_text(
        json.dumps({"compact_optional_schema": True}),
        encoding="utf-8",
    )

    class _SchemaSpyEngine:
        def __init__(self, model_dir: Path | None = None) -> None:
            self.model_dir = model_dir
            self.seen_schemas: list[Any] = []

        def generate_batch(self, prompts: list[str], **kwargs: Any) -> Any:
            self.seen_schemas.append(kwargs.get("json_schema"))
            items = [
                SimpleNamespace(
                    text=json.dumps(sample_out),
                    parsed=sample_out,
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    prompt_tokens=8,
                    completion_tokens=6,
                )
                for _ in prompts
            ]
            return SimpleNamespace(items=items, stats={"docs": len(items), "valid_json_count": len(items), "valid_schema_count": len(items)})

    in_pq = tmp_path / "qat_in.parquet"
    pq.write_table(pa.table({"id": [1], "body_text": ["hello"]}), in_pq)

    # Default (compact_optional_schema=False, no provenance hook): passes full_schema unchanged
    eng_plain = _SchemaSpyEngine()
    pllm.extract_parquet(eng_plain, in_pq, tmp_path / "out_plain.parquet", system_prompt="s", json_schema=full_schema)
    assert set(eng_plain.seen_schemas[0]["properties"].keys()) == {"id", "title", "subtitle", "tags", "meta"}

    # Opt-in flag compact_optional_schema=True: passes pruned schema
    eng_opt = _SchemaSpyEngine()
    pllm.extract_parquet(
        eng_opt,
        in_pq,
        tmp_path / "out_opt.parquet",
        system_prompt="s",
        json_schema=full_schema,
        compact_optional_schema=True,
    )
    assert set(eng_opt.seen_schemas[0]["properties"].keys()) == {"id", "title", "meta"}

    # Checkpoint provenance.json hook (`"compact_optional_schema": true`): auto-prunes schema on both parquet & jsonl
    eng_qat = _SchemaSpyEngine(model_dir=qat_ckpt)
    pllm.extract_parquet(eng_qat, in_pq, tmp_path / "out_qat.parquet", system_prompt="s", json_schema=full_schema)
    assert set(eng_qat.seen_schemas[0]["properties"].keys()) == {"id", "title", "meta"}

    in_jl = tmp_path / "qat.jsonl"
    in_jl.write_text('{"id": 1, "text": "hello"}\n', encoding="utf-8")
    pllm.extract_jsonl(eng_qat, in_jl, tmp_path / "out_qat.jsonl", system_prompt="s", json_schema=full_schema)
    assert set(eng_qat.seen_schemas[1]["properties"].keys()) == {"id", "title", "meta"}
