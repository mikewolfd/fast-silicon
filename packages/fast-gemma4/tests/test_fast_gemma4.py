"""Comprehensive tests for `fast-gemma4` (native ANE, tokenizer/vocab remap, KV_sys prefix, JSON schema, PromptLookup, hetero ANE+GPU, Parquet, HTTP server)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import urllib.request

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fast_gemma4 import (
    FastGemma4Engine,
    FastGemma4OpenAIAdapter,
    FastGemma4Server,
    FastGemma4Tokenizer,
    VocabRemapper,
    extract_jsonl,
    extract_parquet,
    format_gemma4_chat,
    resolve_paths,
)
from fast_gemma4.grammar import GrammarHelper
from fast_gemma4.paths import Gemma4Paths


@pytest.fixture(scope="module")
def paths() -> Gemma4Paths:
    return resolve_paths()


@pytest.fixture(scope="module")
def engine(paths: Gemma4Paths):
    eng = FastGemma4Engine(home=paths.home, default_mode="auto", warmup=True, warmup_ane=False)
    try:
        yield eng
    finally:
        eng.close()


def test_native_gemma4_ane_cli(paths: Gemma4Paths) -> None:
    assert paths.lib_ane_prefix.exists(), f"Missing {paths.lib_ane_prefix}"
    assert paths.ane_cli.exists(), f"Missing {paths.ane_cli}"
    bundle = paths.resolve_ane_bundle("w8") or paths.resolve_ane_bundle("fp16")
    if bundle is None:
        pytest.skip("No compiled CoreML .mlmodelc bundle in artifacts/ane_models/")
    raw = subprocess.check_output(
        [str(paths.ane_cli), "--model", str(bundle), "--batch", "4", "--iters", "1", "--inspect", "--json"],
        text=True,
    ).strip()
    info = json.loads(raw.splitlines()[-1])
    assert info["ane_ops"] == 1326
    assert info["batch"] == 4
    assert info["S"] == 256
    assert info["best_sec"] > 0.0
    assert info["tok_per_sec"] > 0.0


def test_tokenizer_chat_and_lossless_vocab_remap(paths: Gemma4Paths) -> None:
    tok_json = paths.resolve_tokenizer_json()
    tok = FastGemma4Tokenizer(tok_json)
    formatted = format_gemma4_chat([{"role": "user", "content": "Hello world"}], add_generation_prompt=True)
    assert formatted == "<bos><|turn>user\nHello world<turn|>\n<|turn>model\n"
    ids = tok.encode(formatted, add_special_tokens=False)
    assert ids[0] == tok.bos_token_id

    rem = VocabRemapper.load_or_build(tok_json, paths.vocab_remap_npz)
    assert rem.vocab_size == 150786
    english_json = '{"status":"ok","primary_kind":"technical_report","domain":"energy","entities":["Pacific Grid","IEEE-2800-2022 — §101"]}'
    enc = tok.encode(english_json, add_special_tokens=False)
    new_ids, misses = rem.encode_remap(enc)
    assert misses == 0
    assert tok.decode(rem.decode_to_old(new_ids), skip_special_tokens=False) == english_json

    # Verify byte-level lossless fallback on rare non-pruned scripts/emojis
    rare = "Legal note: café naïve 🏛️ 日本語"
    rare_ids = tok.encode(rare, add_special_tokens=False)
    rare_new, _ = rem.encode_remap(rare_ids)
    assert tok.decode(rem.decode_to_old(rare_new), skip_special_tokens=False) == rare


def test_prefix_kv_equivalence_and_single_prompt(engine: FastGemma4Engine) -> None:
    item = engine.generate("Respond with only the capital of France in one word.", max_tokens=16)
    assert "PARIS" in item.text.upper()

    # Verify shared prefix KV_sys split matches unsplit L_sys=0 prefill logits!
    prompt = "Pacific Transmission Consortium grid reliability assessment review:"
    ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
    lcp = max(4, len(ids) - 3)
    pkv_split = engine.model.prefill_prefix(ids[:lcp])
    c_split = engine.model.make_batch_cache(pkv_split, batch_size=1, doc_cap_full=64)
    lg_split, _ = engine.model.prefill_suffix_batch([ids[lcp:]], c_split, group_size=1)

    pkv_empty = engine.model.prefill_prefix([])
    c_full = engine.model.make_batch_cache(pkv_empty, batch_size=1, doc_cap_full=64)
    lg_full, _ = engine.model.prefill_suffix_batch([ids], c_full, group_size=1)

    a = np.array(lg_split[0, 0, :].astype(mx.float32))
    b = np.array(lg_full[0, 0, :].astype(mx.float32))
    max_diff = float(np.max(np.abs(a - b)))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.999
    assert max_diff < 1.0
    assert int(np.argmax(a)) == int(np.argmax(b))


def test_schema_and_prompt_lookup_exactness(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    fix_dir = paths.home / "tests" / "fixtures"
    schema_p = fix_dir / "extract_v1.schema.json"
    docs_p = fix_dir / "sample_docs_8.json"
    sys_p = fix_dir / "sys_extract_v1.txt"
    schema_str = schema_p.read_text(encoding="utf-8")
    docs = json.loads(docs_p.read_text(encoding="utf-8"))[:4]
    sys_prompt = sys_p.read_text(encoding="utf-8").strip()
    prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]

    # 1. Verify L_q=1 sequential vs L_q=4 parallel ragged forward parity within tight BF16 bounds
    probe_ids = engine.tokenizer.encode("Pacific Transmission Consortium", add_special_tokens=False)[:4]
    pkv0 = engine.model.prefill_prefix([])
    c_seq = engine.model.make_batch_cache(pkv0, batch_size=1, doc_cap_full=32)
    off_s = mx.array([0], dtype=mx.int32)
    for t in probe_ids:
        lg_s = engine.model.forward_step_ragged(mx.array([[t]], dtype=mx.int32), c_seq, off_s)
        off_s = off_s + 1
    mx.eval(lg_s)

    c_par = engine.model.make_batch_cache(pkv0, batch_size=1, doc_cap_full=32)
    lg_p = engine.model.forward_step_ragged(
        mx.array([probe_ids], dtype=mx.int32), c_par, mx.array([0], dtype=mx.int32)
    )
    mx.eval(lg_p)
    a_np = np.array(lg_s[0, -1, :].astype(mx.float32))
    b_np = np.array(lg_p[0, -1, :].astype(mx.float32))
    cos_lq = float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-12))
    assert cos_lq > 0.999
    assert float(np.max(np.abs(a_np - b_np))) < 1.0
    assert int(np.argmax(a_np)) == int(np.argmax(b_np))

    # 2. Single-document (B=1) PromptLookup + strict schema parity
    one_gpu = engine.generate_batch(
        prompts[:1], system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512, mode="gpu", speculative=False
    )
    one_spec = engine.generate_batch(
        prompts[:1], system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512, mode="gpu", speculative=True
    )
    assert one_gpu.items[0].valid_json and one_gpu.items[0].valid_schema
    assert one_spec.items[0].valid_json and one_spec.items[0].valid_schema
    assert one_gpu.items[0].parsed is not None and one_spec.items[0].parsed is not None
    assert one_gpu.items[0].parsed["status"] == one_spec.items[0].parsed["status"] == "usable"
    assert one_gpu.items[0].parsed["primary_kind"] == one_spec.items[0].parsed["primary_kind"]

    # 3. Multi-document batched (B=4) PromptLookup + strict schema validity
    res_gpu = engine.generate_batch(
        prompts,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
    )
    res_spec = engine.generate_batch(
        prompts,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=True,
    )
    assert res_gpu.stats["valid_json_count"] == 4
    assert res_gpu.stats["valid_schema_count"] == 4
    assert res_spec.stats["valid_json_count"] == 4
    assert res_spec.stats["valid_schema_count"] == 4
    for a, b in zip(res_gpu.items, res_spec.items):
        assert a.parsed is not None and b.parsed is not None
        assert "status" in a.parsed and "text_quality" in a.parsed
        assert a.parsed["status"] == b.parsed["status"]
        assert a.parsed["text_quality"] == b.parsed["text_quality"]
        assert a.parsed.get("primary_kind") == b.parsed.get("primary_kind")


def test_hetero_ane_gpu_prefix_pipeline(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    if not engine.ane_available:
        pytest.skip("ANE bundle unavailable")
    fix_dir = paths.home / "tests" / "fixtures"
    schema_p = fix_dir / "extract_v1.schema.json"
    docs_p = fix_dir / "sample_docs_8.json"
    sys_p = fix_dir / "sys_extract_v1.txt"
    schema_str = schema_p.read_text(encoding="utf-8")
    docs = json.loads(docs_p.read_text(encoding="utf-8"))
    sys_prompt = sys_p.read_text(encoding="utf-8").strip()
    prompts_8 = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]
    ref_gpu = engine.generate_batch(
        prompts_8,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
    )
    assert ref_gpu.stats["valid_json_count"] == 8
    assert ref_gpu.stats["valid_schema_count"] == 8
    assert sum(1 for it in ref_gpu.items if it.parsed and it.parsed.get("status") == "usable") >= 6

    prompts_64 = [prompts_8[i % 8] for i in range(64)]
    res_het = engine.generate_batch(
        prompts_64,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="hetero",
        batch_size=64,
    )
    assert res_het.stats["mode_used"] == "hetero_ane_gpu"
    assert res_het.stats["lcp"] <= 1408
    assert res_het.stats["valid_json_count"] == 64
    assert res_het.stats["valid_schema_count"] == 64
    for i in range(8):
        assert res_het.items[i].parsed is not None and ref_gpu.items[i].parsed is not None
        assert res_het.items[i].parsed["status"] == ref_gpu.items[i].parsed["status"]
        assert res_het.items[i].parsed["text_quality"] == ref_gpu.items[i].parsed["text_quality"]
        if "primary_kind" in ref_gpu.items[i].parsed and "primary_kind" in res_het.items[i].parsed:
            assert res_het.items[i].parsed["primary_kind"] == ref_gpu.items[i].parsed["primary_kind"]


def test_parquet_and_jsonl_pipeline(tmp_path: Path, engine: FastGemma4Engine) -> None:
    schema = {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "enum": ["energy", "finance", "other"]},
            "topic": {"type": "string"},
        },
        "required": ["domain", "topic"],
        "additionalProperties": False,
    }
    sys_prompt = "Classify document text into domain and concise topic JSON."

    # JSONL extraction
    jsonl_in = tmp_path / "docs.jsonl"
    jsonl_out = tmp_path / "out.jsonl"
    jsonl_in.write_text(
        json.dumps({"id": 1, "text": "Grid reliability engineering report on coastal substation solar inverters."}) + "\n"
        + json.dumps({"id": 2, "text": "Quarterly clearing risk filing on treasury repo margin liquidity."}) + "\n",
        encoding="utf-8",
    )
    extract_jsonl(engine, jsonl_in, jsonl_out, system_prompt=sys_prompt, json_schema=schema, max_tokens=64)
    rows = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(r["valid_json"] and r["valid_schema"] for r in rows)

    # Sharded + resumable Parquet extraction with UTF-8 OCR page slicing
    in_pq = tmp_path / "in.parquet"
    out_dir = tmp_path / "shards"
    p1 = "COVER\n"
    p2 = "Quarterly treasury risk filing on fixed-income liquidity controls.\n"
    t0 = p1 + p2
    po = [0, len(p1.encode("utf-8"))]
    tbl = pa.table(
        {
            "id": pa.array([10, 11], type=pa.int64()),
            "body_text": pa.array([t0, "Substation transformer thermal load review for coastal transmission lines."], type=pa.string()),
            "is_ocr": pa.array([True, True], type=pa.bool_()),
            "ocr_pages": pa.array([[2], [1]], type=pa.list_(pa.int32())),
            "page_offsets": pa.array([po, [0]], type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(tbl, in_pq)

    summary = extract_parquet(
        engine,
        in_pq,
        out_dir,
        system_prompt=sys_prompt,
        json_schema=schema,
        drop_text_col=True,
        filter_ocr=True,
        ocr_pages_only=True,
        batch_rows=2,
        shard_rows=10,
        max_tokens=64,
        mode="gpu",
    )
    assert summary.docs_processed == 2
    assert summary.valid_json == 2
    assert summary.valid_schema == 2

    resumed = extract_parquet(
        engine,
        in_pq,
        out_dir,
        system_prompt=sys_prompt,
        json_schema=schema,
        resume=True,
        max_tokens=64,
        mode="gpu",
    )
    assert resumed.docs_processed == 0

    # Verify max_pages slicing on multi-page page_offsets
    from fast_gemma4.pipeline import slice_first_pages_utf8

    assert slice_first_pages_utf8(t0, po, 1) == p1
    assert slice_first_pages_utf8(t0, po, 2) == t0


def test_http_server_smoke(engine: FastGemma4Engine) -> None:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    with FastGemma4Server(engine, host="127.0.0.1", port=0) as srv:
        base = f"http://{srv.host}:{srv.port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            assert h["status"] == "ok"
            assert h["vocab_size"] == 150786

        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            assert models["data"][0]["id"] == engine.model_name

        payload = {
            "model": engine.model_name,
            "messages": [{"role": "user", "content": 'Return ONLY JSON with key "city" set to the capital city of Japan.'}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
            "max_tokens": 48,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            assert "Tokyo" in parsed["city"]

        comp_payload = {
            "model": engine.model_name,
            "prompt": 'Return ONLY JSON with key "city" set to the capital city of South Korea.',
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
            "max_tokens": 48,
            "temperature": 0.0,
        }
        comp_req = urllib.request.Request(
            f"{base}/v1/completions",
            data=json.dumps(comp_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(comp_req, timeout=15) as resp:
            comp_body = json.loads(resp.read().decode("utf-8"))
            comp_parsed = json.loads(comp_body["choices"][0]["text"])
            assert "Seoul" in comp_parsed["city"]


def test_tuned_defaults_and_routing_cpu(paths: Gemma4Paths) -> None:
    """CPU-only check of tuned auto-routing, PromptLookup cutoff, whitespace flexibility, and CLI defaults."""
    from fast_gemma4.cli import build_parser
    from fast_gemma4.engine import select_execution_plan, should_use_speculative

    # 1. Auto routes to GPU rolling (faster and 3x lower RAM than static ANE waves); explicit `hetero` enables ANE+GPU
    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=1, batch_size=None) == (False, 1)
    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=64, batch_size=None) == (False, 64)
    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=256, batch_size=None) == (False, 256)
    assert select_execution_plan("auto", True, lcp=1024, max_suf=400, n_docs=384, batch_size=None) == (False, 256)
    assert select_execution_plan("hetero", True, lcp=1024, max_suf=400, n_docs=64, batch_size=64) == (True, 64)
    assert select_execution_plan("hetero", True, lcp=1500, max_suf=400, n_docs=512, batch_size=None) == (False, 256)

    # 2. PromptLookup auto cutoff: enabled only at B <= 4 bandwidth regime (or explicit speculative=True)
    assert should_use_speculative(None, False, 1, 0.0) is True
    assert should_use_speculative(None, False, 4, 0.0) is True
    assert should_use_speculative(None, False, 8, 0.0) is False
    assert should_use_speculative(None, False, 32, 0.0) is False
    assert should_use_speculative(True, False, 32, 0.0) is True
    assert should_use_speculative(True, False, 4, 0.7) is False

    # 3. CPU llguidance whitespace flexibility for ad-hoc prompts vs minified batch extraction
    tok = FastGemma4Tokenizer(paths.resolve_tokenizer_json())
    rem = VocabRemapper.load_or_build(paths.resolve_tokenizer_json(), paths.vocab_remap_npz)
    city_schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}
    gh_flex = GrammarHelper(tok, city_schema, rem.kept_old_ids, whitespace_flexible=True)
    m_flex = gh_flex.new_matcher()
    assert m_flex.validate_tokens(tok.encode('{"city": "Tokyo"}', add_special_tokens=False))
    assert m_flex.validate_tokens(tok.encode('{"city":"Tokyo"}', add_special_tokens=False))

    # 4. CLI defaults
    p = build_parser()
    a_gen = p.parse_args(["generate", "hi"])
    assert a_gen.max_tokens == 512 and a_gen.mode == "auto" and a_gen.ane_variant == "fp16" and a_gen.compact_json is False
    a_ext = p.parse_args(["extract", "in.jsonl", "out.jsonl"])
    assert a_ext.batch_size == 128 and a_ext.max_tokens == 512 and a_ext.whitespace_flexible is False
    assert "minified JSON" in a_ext.system
    a_pq = p.parse_args(["parquet", "in.parquet", "out/"])
    assert a_pq.batch_rows == 128 and a_pq.max_tokens == 512 and a_pq.whitespace_flexible is False
    assert "minified JSON" in a_pq.system
    a_srv = p.parse_args(["serve", "--coalesce-ms", "2.5", "--max-batch", "64", "--model-name", "gemma4-local"])
    assert a_srv.coalesce_ms == 2.5 and a_srv.max_batch == 64 and a_srv.model_name == "gemma4-local"


def test_http_server_coalescing_sse_and_unified_cpu() -> None:
    """CPU-only verification of concurrent HTTP micro-batching, SSE streaming, extra_body, and UnifiedOpenAIServer."""
    import base64
    import concurrent.futures
    import threading
    import time
    import urllib.error
    from types import SimpleNamespace

    from fast_shared.server import UnifiedOpenAIServer

    class _MockModel:
        vocab_size = 150786
        lm_head_mode = "q4"

    class _MockEngine:
        model_name = "google/gemma-4-E2B-it"
        default_mode = "auto"
        ane_available = True

        def __init__(self) -> None:
            self.model = _MockModel()
            self.calls: list[tuple[int, dict[str, object]]] = []
            self._lock = threading.Lock()

        def generate_batch(self, prompts, **kwargs):
            with self._lock:
                self.calls.append((len(prompts), dict(kwargs)))
            time.sleep(0.03)  # Holds leader slightly so concurrent arrivals queue and coalesce cleanly
            items = []
            for p in prompts:
                content = p[-1]["content"] if isinstance(p, list) else str(p)
                items.append(
                    SimpleNamespace(
                        text=json.dumps({
                            "q": content,
                            "thinking": bool(kwargs.get("enable_thinking", False)),
                            "sys": kwargs.get("system_prompt"),
                            "schema": kwargs.get("json_schema") is not None,
                        }),
                        prompt_tokens=10,
                        completion_tokens=6,
                        finish_reason="stop",
                    )
                )
            return SimpleNamespace(items=items, stats={"docs": len(prompts)})

    eng = _MockEngine()
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

    with FastGemma4Server(eng, host="127.0.0.1", port=0, coalesce_ms=25.0, max_batch_size=32) as srv:
        base = f"http://{srv.host}:{srv.port}"

        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            assert h["coalesce_ms"] == 25.0 and h["max_batch_size"] == 32 and h["vocab_size"] == 150786

        with urllib.request.urlopen(f"{base}/v1/models/google%2Fgemma-4-E2B-it", timeout=5) as resp:
            m = json.loads(resp.read().decode("utf-8"))
            assert m["id"] == "google/gemma-4-E2B-it" and m["object"] == "model"

        with pytest.raises(urllib.error.HTTPError) as missing_exc:
            urllib.request.urlopen(f"{base}/v1/models/does-not-exist", timeout=5)
        assert missing_exc.value.code == 404

        bad_req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": "not-an-int"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as bad_exc:
            urllib.request.urlopen(bad_req, timeout=5)
        assert bad_exc.value.code == 400
        bad_err = json.loads(bad_exc.value.read().decode("utf-8"))
        assert bad_err["error"]["type"] == "invalid_request_error"

        def _post_chat(i: int) -> dict:
            body = {
                "model": "google/gemma-4-E2B-it",
                "messages": [{"role": "user", "content": f"doc-{i}"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "doc", "schema": schema}},
                "max_tokens": 64,
                "temperature": 0.0,
                "seed": 100 + i,  # Greedy ignores seed, so requests still coalesce!
                "extra_body": {"system_prompt": "SYS", "enable_thinking": True, "whitespace_flexible": False},
            }
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(_post_chat, range(6)))

        assert len(results) == 6
        for i, res in enumerate(results):
            parsed = json.loads(res["choices"][0]["message"]["content"])
            assert parsed == {"q": f"doc-{i}", "thinking": True, "sys": "SYS", "schema": True}
        assert len(eng.calls) < 6, f"Expected concurrent requests to coalesce (<6 batches), got {eng.calls}"
        assert any(batch_len >= 2 for batch_len, _ in eng.calls)

        # SSE streaming check
        sse_req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "stream-me"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(sse_req, timeout=5) as resp:
            lines = [ln for ln in resp.read().decode("utf-8").splitlines() if ln.startswith("data: ")]
            assert lines[-1] == "data: [DONE]"
            chunks = [json.loads(ln.removeprefix("data: ")) for ln in lines[:-1]]
            assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
            assert json.loads(chunks[1]["choices"][0]["delta"]["content"])["q"] == "stream-me"
            assert chunks[2]["choices"][0]["finish_reason"] == "stop"
            assert chunks[2]["usage"]["total_tokens"] == 16

    # Verify FastGemma4OpenAIAdapter mounts cleanly into UnifiedOpenAIServer
    class _MockEmbTok:
        def resolve_prefix(self, **_kwargs):
            return ""

        def encode_batch(self, texts, **_kwargs):
            return [[2, 5, 1] for _ in texts]

    class _MockEmbEng:
        dim = 4
        model_name = "BAAI/bge-small-en-v1.5"
        ane_available = True
        default_mode = "hetero"
        tokenizer = _MockEmbTok()

        def encode_query(self, _toks):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def encode_corpus(self, tok_lists, mode=None):
            return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(tok_lists), 1))

    adapter = FastGemma4OpenAIAdapter(eng, model_name="gemma-4-e2b-it")
    with UnifiedOpenAIServer(
        embedding_engines={"BAAI/bge-small-en-v1.5": _MockEmbEng()},
        chat_engines={"gemma-4-e2b-it": adapter},
        host="127.0.0.1",
        port=0,
    ) as unified:
        ubase = f"http://{unified.host}:{unified.port}"
        with urllib.request.urlopen(f"{ubase}/v1/models", timeout=5) as resp:
            ids = [m["id"] for m in json.loads(resp.read().decode("utf-8"))["data"]]
            assert set(ids) == {"BAAI/bge-small-en-v1.5", "gemma-4-e2b-it"}

        comp_req = urllib.request.Request(
            f"{ubase}/v1/completions",
            data=json.dumps({
                "model": "gemma-4-e2b-it",
                "prompt": ["hello", "world"],
                "use_chat_template": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(comp_req, timeout=5) as resp:
            comp = json.loads(resp.read().decode("utf-8"))
            assert [json.loads(c["text"])["q"] for c in comp["choices"]] == ["hello", "world"]
            assert eng.calls[-1][1]["use_chat_template"] is False

        emb_req = urllib.request.Request(
            f"{ubase}/v1/embeddings",
            data=json.dumps({"model": "BAAI/bge-small-en-v1.5", "input": "hello", "encoding_format": "base64"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(emb_req, timeout=5) as resp:
            ebody = json.loads(resp.read().decode("utf-8"))
            vec = np.frombuffer(base64.b64decode(ebody["data"][0]["embedding"]), dtype="<f4")
            assert np.allclose(vec, [1.0, 0.0, 0.0, 0.0])


def test_v4_kernels_rolling_split_kv_i8(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify fused_softcap_argmax_bf16, slot KV_i8 vs BF16, Layer-15 split, EarlyExit14 drain, and rolling refill."""
    from fast_gemma4.kernels import (
        fused_ple_geglu_indexed,
        fused_res_rms_norm,
        fused_softcap_argmax_bf16,
    )
    from fast_gemma4.model import logit_softcap

    # 1. Exact bit-for-bit parity of fused_softcap_argmax_bf16 (including strided/padded V_pad slice)
    rng = np.random.default_rng(7)
    raw_np = (rng.standard_normal((8, 1, 150848)) * 12.0).astype(np.float32)
    raw_mx = mx.array(raw_np, dtype=mx.bfloat16)
    v_lim = 150786
    ref_sc = logit_softcap(30.0, raw_mx[:, 0, :v_lim])
    ref_am = mx.argmax(ref_sc, axis=-1)
    got_am = fused_softcap_argmax_bf16(raw_mx, 30.0, vocab_limit=v_lim)
    got_am_slice = fused_softcap_argmax_bf16(raw_mx[:, :, :v_lim], 30.0)
    mx.eval(ref_am, got_am, got_am_slice)
    assert got_am.tolist() == ref_am.tolist()
    assert got_am_slice.tolist() == ref_am.tolist()

    # 2. fused_res_rms_norm & fused_ple_geglu_indexed parity
    res = mx.array(rng.standard_normal((4, 1536)), dtype=mx.bfloat16)
    x = mx.array(rng.standard_normal((4, 1536)), dtype=mx.bfloat16)
    w = mx.array(rng.uniform(0.5, 1.5, size=(1536,)), dtype=mx.bfloat16)
    got_rn = fused_res_rms_norm(res, x, w, 1e-6)
    ref_rn = res + mx.fast.rms_norm(x, w, 1e-6)
    mx.eval(got_rn, ref_rn)
    assert float(mx.max(mx.abs(got_rn.astype(mx.float32) - ref_rn.astype(mx.float32)))) < 0.05

    gate = mx.array(rng.standard_normal((4, 2, 256)), dtype=mx.bfloat16)
    pli = mx.array(rng.standard_normal((4, 2, 14, 256)), dtype=mx.bfloat16)
    l_idx = mx.array([5], dtype=mx.int32)
    got_pg = fused_ple_geglu_indexed(gate, pli, l_idx)
    import mlx.nn as nn

    ref_pg = nn.gelu_approx(gate) * pli[:, :, 5, :]
    mx.eval(got_pg, ref_pg)
    assert float(mx.max(mx.abs(got_pg.astype(mx.float32) - ref_pg.astype(mx.float32)))) < 0.05

    # 3. Slot KV_i8 + prefill_slots_lazy_lg vs non-slot BF16 forward_step_ragged logits parity
    toks = engine.tokenizer.encode("Pacific Transmission Consortium grid reliability assessment.", add_special_tokens=False)
    pkv = engine.model.prefill_prefix(toks[:5])
    c_bf = engine.model.make_batch_cache(pkv, batch_size=1, doc_cap_full=64)
    lg_bf, _ = engine.model.prefill_suffix_batch([toks[5:]], c_bf, group_size=1)

    c_slot = engine.model.make_slot_cache(pkv, batch_size=2, doc_cap_full=64, kv_i8=True)
    x_s, am_s, next_off = engine.model.prefill_slots_lazy_lg([toks[5:], toks[5:]], [0, 1], c_slot, group_size=2)
    lg_s = engine.model._apply_lm_head(x_s, apply_softcap=True)
    mx.eval(lg_bf, lg_s, am_s)
    a_v = np.array(lg_bf[0, 0, :].astype(mx.float32))
    b_v = np.array(lg_s[0, 0, :].astype(mx.float32))
    cos = float(np.dot(a_v, b_v) / (np.linalg.norm(a_v) * np.linalg.norm(b_v) + 1e-12))
    assert cos > 0.998
    assert am_s.tolist() == [int(np.argmax(b_v)), int(np.argmax(b_v))]
    assert next_off == [len(toks), len(toks)]

    # 4. Continuous rolling refill (N=12 > batch_size=4) + Layer-15 split + EarlyExit14 FF drain + unconstrained & T>0
    fix_dir = paths.home / "tests" / "fixtures"
    schema_p = fix_dir / "extract_v1.schema.json"
    docs_p = fix_dir / "sample_docs_8.json"
    sys_p = fix_dir / "sys_extract_v1.txt"
    schema_str = schema_p.read_text(encoding="utf-8")
    docs = json.loads(docs_p.read_text(encoding="utf-8"))
    sys_prompt = sys_p.read_text(encoding="utf-8").strip()
    prompts_12 = [
        f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}"
        for _, fn, known, txt in (docs + docs[:4])
    ]
    r_roll = engine.generate_batch(
        prompts_12,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
        batch_size=4,
    )
    assert r_roll.stats["rolling"] is True
    assert r_roll.stats["kv_i8"] is True
    assert r_roll.stats["valid_json_count"] == 12
    assert r_roll.stats["valid_schema_count"] == 12
    for i in range(4):
        a, b = r_roll.items[i].parsed, r_roll.items[8 + i].parsed
        assert a is not None and b is not None
        for key in ("status", "text_quality", "domain", "primary_kind"):
            assert a.get(key) == b.get(key)

    r_roll_exact = engine.generate_batch(
        prompts_12,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
        batch_size=12,
    )
    assert r_roll_exact.stats["rolling"] is True
    for i in range(4):
        assert r_roll_exact.items[i].text == r_roll_exact.items[8 + i].text

    # Unconstrained rolling & sampled (T > 0) rolling
    r_unc = engine.generate_batch(
        ["Reply with the word ALPHA.", "Reply with the word BETA."] * 3,
        max_tokens=12,
        mode="gpu",
        speculative=False,
        batch_size=2,
    )
    assert r_unc.stats["rolling"] is True and len(r_unc.items) == 6
    r_temp = engine.generate_batch(
        ["Return JSON {ok:true}." for _ in range(5)],
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        max_tokens=16,
        temperature=0.3,
        mode="gpu",
        batch_size=3,
    )
    assert r_temp.stats["rolling"] is True
    assert r_temp.stats["valid_schema_count"] == 5


def test_multishard_bf16_and_custom_model_dir_routing(paths: Gemma4Paths) -> None:
    """Verify multi-shard e2b-it-bf16 loads cleanly, disables stale ANE weights, and runs rolling generation."""
    from fast_shared.hf import resolve_hf_model_dir

    try:
        bf16_dir = resolve_hf_model_dir(None, "mlx-community/gemma-4-e2b-it-bf16", local_files_only=True)
    except Exception:
        pytest.skip("mlx-community/gemma-4-e2b-it-bf16 not cached locally")

    eng_bf16 = FastGemma4Engine(home=paths.home, model_dir=bf16_dir, default_mode="auto", warmup=True, warmup_ane=False)
    try:
        assert eng_bf16.ane_available is False  # Never use base E2B-4bit ANE bundles with overridden model_dir
        assert len(list(bf16_dir.glob("model*.safetensors"))) >= 2
        res = eng_bf16.generate_batch(
            [
                "Return city Tokyo in JSON.",
                "Return city Paris in JSON.",
                "Return city Rome in JSON.",
                "Return city Ottawa in JSON.",
                "Return city Bern in JSON.",
            ],
            json_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            max_tokens=32,
            mode="auto",
            speculative=False,
            batch_size=3,
        )
        assert res.stats["mode_used"] == "gpu"
        assert res.stats["rolling"] is True
        assert res.stats["valid_schema_count"] == 5
        assert "Tokyo" in res.items[0].text
    finally:
        eng_bf16.close()
        import gc

        gc.collect()
        mx.clear_cache()
