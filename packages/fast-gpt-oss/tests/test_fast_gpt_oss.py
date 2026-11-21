"""CPU and Metal 4 GPU unit tests for `fast_gpt_oss`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fast_gpt_oss
from fast_shared.grammar import GrammarHelper, validate_json_against_schema
from fast_shared.server import UnifiedOpenAIServer
from fast_gpt_oss.cli import build_parser
from fast_gpt_oss.engine import (
    GenerationBatchResult,
    GenerationItem,
    resolve_reasoning_params,
    select_gpt_oss_plan,
    split_lcp_ids,
)
from fast_gpt_oss.kernels import KernelSpec, render_kernel_sources
from fast_gpt_oss.model import (
    FastGptOssModel,
    GptOssConfig,
    infer_affine_quant_spec,
    make_yarn_rope_np,
    pack_gpt_oss_layer_tensors,
    prune_lm_head_tensors,
)
from fast_gpt_oss.pipeline import extract_jsonl, extract_parquet
from fast_gpt_oss.server import FastGptOssServer
from fast_gpt_oss.tokenizer import (
    FastGptOssTokenizer,
    O200kVocabRemapper,
    build_o200k_kept_vocab,
    format_harmony_chat,
    parse_harmony_output,
)


def _bytes_to_unicode_map() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _write_synthetic_o200k_tokenizer(path: Path) -> Path:
    """Write a complete byte-level BPE + Harmony special token `tokenizer.json`."""
    b2u = _bytes_to_unicode_map()
    vocab: dict[str, int] = {}
    for b in range(256):
        vocab[b2u[b]] = b

    extra_pieces = [
        "hello",
        "world",
        "true",
        "false",
        "null",
        '{"',
        '":',
        '","',
        '":true',
        '":false',
        '"}',
        "[]",
        "bill",
        "hearing",
        "primary_kind",
        "summary",
        "domain",
        "alpha",
        "beta",
        "123",
        " ",
        "  ",
        "\\n",
        "\u0120world",
        "\u4e2d",  # CJK character outside ASCII-pruned set to test vocab pruning remap
        "caf\u00e9",
    ]
    next_id = 256
    merges: list[str] = []
    for p in extra_pieces:
        if p not in vocab:
            vocab[p] = next_id
            next_id += 1

    # Add simple BPE merges for `{"`, `":`, `"}`, `true`, `false`
    if "{" in vocab and '"' in vocab:
        merges.append('{ "')
    if '"' in vocab and ":" in vocab:
        merges.append('" :')
    if '"' in vocab and "}" in vocab:
        merges.append('" }')

    specials = [
        "<|startoftext|>",
        "<|endoftext|>",
        "<|return|>",
        "<|end|>",
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|constrain|>",
    ]
    base_special = next_id
    added_tokens = []
    for idx, content in enumerate(specials):
        added_tokens.append(
            {
                "id": base_special + idx,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    tj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    tok_path = path / "tokenizer.json"
    tok_path.write_text(json.dumps(tj), encoding="utf-8")
    return tok_path


def test_harmony_formatting_and_output_parsing_cpu(tmp_path: Path) -> None:
    assert hasattr(fast_gpt_oss, "FastGptOssEngine")
    msgs = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "<|channel|>analysis<|message|>secret scratch<|end|><|start|>assistant<|channel|>final<|message|>Hello!<|return|>",
        },
        {"role": "user", "content": "Next"},
    ]
    fmt_low = format_harmony_chat(msgs, reasoning_effort="low")
    assert fmt_low.startswith("<|start|>system<|message|>Be concise.<|end|><|start|>developer<|message|>Reasoning: low")
    assert "secret scratch" not in fmt_low
    assert "<|start|>assistant<|channel|>final<|message|>Hello!<|end|>" in fmt_low
    assert fmt_low.endswith("<|start|>assistant<|channel|>analysis<|message|>")

    fmt_skip = format_harmony_chat([{"role": "user", "content": "Go"}], reasoning_effort="skip")
    assert "Reasoning:" not in fmt_skip
    assert fmt_skip.endswith(
        "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    )

    final_only, r0 = parse_harmony_output("plain json{}<|return|>")
    assert final_only == "plain json{}"
    assert r0 is None

    final_1, r1 = parse_harmony_output(
        "<|channel|>analysis<|message|>check x<|end|><|start|>assistant<|channel|>final<|message|>{\"ok\":1}<|return|>"
    )
    assert final_1 == '{"ok":1}'
    assert r1 == "check x"

    final_2, r2 = parse_harmony_output("step 1 thinking<|end|><|start|>assistant<|channel|>final<|message|>ans")
    assert final_2 == "ans"
    assert r2 == "step 1 thinking"

    final_3, r3 = parse_harmony_output("<|channel|>analysis<|message|>incomplete thought")
    assert final_3 == ""
    assert r3 == "incomplete thought"

    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    assert tok.end_token_id not in tok.analysis_stop_token_ids
    assert tok.end_token_id in tok.final_stop_token_ids
    assert tok.return_token_id in tok.analysis_stop_token_ids
    assert tok.decode(tok.encode("{\"ok\":true}"), skip_special_tokens=True) == '{"ok":true}'


def test_o200k_vocab_pruner_and_llguidance_lazy_mask_cpu(tmp_path: Path) -> None:
    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    cache_npz = tmp_path / "cache" / "vocab_remap.npz"

    kept = build_o200k_kept_vocab(tok_path, target_size=276)
    assert len(kept) == 276
    assert tok.return_token_id in set(kept.tolist())
    assert tok.end_token_id in set(kept.tolist())
    assert len(set(kept.tolist())) == len(kept)

    remap = O200kVocabRemapper.load_or_build(tok_path, cache_npz_path=cache_npz, target_size=276, full_vocab_size=620)
    assert cache_npz.exists()
    remap2 = O200kVocabRemapper.load_or_build(tok_path, cache_npz_path=cache_npz, target_size=276, full_vocab_size=620)
    assert np.array_equal(remap.kept_old_ids, remap2.kept_old_ids)
    assert remap.old_to_new.shape == (620,)
    assert int(remap.old_to_new[tok.return_token_id]) >= 0

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    gh = GrammarHelper(
        tok,
        schema,
        new_to_old=remap.kept_old_ids,
        whitespace_flexible=False,
        eos_token_id=tok.return_token_id,
    )
    assert len(gh.upfront_ff) >= 1
    assert tok.decode(gh.upfront_ff, skip_special_tokens=True).startswith('{"')

    # Verify lazy logits getter is NOT called on speculative hit!
    matcher = gh.new_matcher()

    called = [0]

    def _never_called():
        called[0] += 1
        return np.zeros(len(remap.kept_old_ids), dtype=np.float32)

    colon_id = tok.encode('":')[0]
    colon_new = int(remap.old_to_new[colon_id])
    picked, was_masked = gh.pick_valid_token(matcher, _never_called, colon_new, fallback_to_full_vocab=True)
    assert picked == colon_id
    assert was_masked is False
    assert called[0] == 0

    # Verify lazy logits getter IS called on rejection
    matcher2 = gh.new_matcher()
    invalid_id = tok.encode("z")[0]
    invalid_new = int(remap.old_to_new[invalid_id])

    def _lazy_logits():
        called[0] += 1
        arr = np.full(len(remap.kept_old_ids), -50.0, dtype=np.float32)
        arr[int(remap.old_to_new[colon_id])] = 10.0
        return arr

    picked2, was_masked2 = gh.pick_valid_token(matcher2, _lazy_logits, invalid_new, fallback_to_full_vocab=True)
    assert was_masked2 is True
    assert called[0] == 1
    assert picked2 == colon_id

    # Generate valid schema object through matcher manually (llguidance stops right at `}`)
    valid_ids = tok.encode('{"ok":true}')
    assert gh.validate_token_sequence(valid_ids)
    validate_json_against_schema({"ok": True}, schema)


def test_model_packing_config_rope_and_kernel_specs_cpu() -> None:
    cfg = GptOssConfig.from_dict(
        {
            "hidden_size": 256,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "sliding_window": 16,
            "num_local_experts": 8,
        }
    )
    assert cfg.q_dim == 512
    assert cfg.kv_dim == 64
    assert cfg.qkv_dim == 640
    spec = cfg.to_kernel_spec()
    sources = render_kernel_sources(spec)
    assert "hydragen_p2_varlen" in sources
    assert "HasSys" in sources["hydragen_p2_varlen"]["source"]
    assert "SwaInitLens" in sources["swa128_decode"]["input_names"]
    assert "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>" in sources["moe_batch_gu_unified"]["header"]

    full_sources = render_kernel_sources(KernelSpec())
    assert "2880u" in full_sources["rmsnorm_bf16_to_f16"]["source"]

    freqs, mscale = make_yarn_rope_np()
    assert freqs.shape == (32,)
    assert freqs.dtype == np.float32
    assert mscale > 1.0

    assert infer_affine_quant_spec((5120, 360), (5120, 45), 2880) == (True, 4, 64)
    assert infer_affine_quant_spec((32, 720), (32, 45), 2880) == (True, 8, 64)
    assert infer_affine_quant_spec((512, 256), None, 256) == (False, 16, 0)

    # Pure CPU layer packing & lm_head pruning
    w = {
        "model.layers.0.input_layernorm.weight": np.ones(256, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones(256, dtype=np.float32),
        "model.layers.0.self_attn.sinks": np.zeros(8, dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.ones((512, 32), dtype=np.uint32),
        "model.layers.0.self_attn.q_proj.scales": np.ones((512, 4), dtype=np.float16),
        "model.layers.0.self_attn.q_proj.biases": np.zeros((512, 4), dtype=np.float16),
        "model.layers.0.self_attn.q_proj.bias": np.zeros(512, dtype=np.float16),
        "model.layers.0.self_attn.k_proj.weight": np.ones((64, 32), dtype=np.uint32),
        "model.layers.0.self_attn.k_proj.scales": np.ones((64, 4), dtype=np.float16),
        "model.layers.0.self_attn.k_proj.biases": np.zeros((64, 4), dtype=np.float16),
        "model.layers.0.self_attn.k_proj.bias": np.zeros(64, dtype=np.float16),
        "model.layers.0.self_attn.v_proj.weight": np.ones((64, 32), dtype=np.uint32),
        "model.layers.0.self_attn.v_proj.scales": np.ones((64, 4), dtype=np.float16),
        "model.layers.0.self_attn.v_proj.biases": np.zeros((64, 4), dtype=np.float16),
        "model.layers.0.self_attn.v_proj.bias": np.zeros(64, dtype=np.float16),
        "model.layers.0.self_attn.o_proj.weight": np.ones((256, 64), dtype=np.uint32),
        "model.layers.0.self_attn.o_proj.scales": np.ones((256, 8), dtype=np.float16),
        "model.layers.0.self_attn.o_proj.biases": np.zeros((256, 8), dtype=np.float16),
        "model.layers.0.self_attn.o_proj.bias": np.zeros(256, dtype=np.float16),
        "model.layers.0.mlp.router.weight": np.ones((8, 64), dtype=np.uint32),
        "model.layers.0.mlp.router.scales": np.ones((8, 4), dtype=np.float16),
        "model.layers.0.mlp.router.biases": np.zeros((8, 4), dtype=np.float16),
        "model.layers.0.mlp.router.bias": np.zeros(8, dtype=np.float16),
        "model.layers.0.mlp.experts.gate_proj.weight": np.ones((8, 256, 8, 4), dtype=np.uint32),
        "model.layers.0.mlp.experts.gate_proj.scales": np.full((8, 256, 8), 127, dtype=np.uint8),
        "model.layers.0.mlp.experts.gate_proj.bias": np.zeros((8, 256), dtype=np.float16),
        "model.layers.0.mlp.experts.up_proj.weight": np.ones((8, 256, 8, 4), dtype=np.uint32),
        "model.layers.0.mlp.experts.up_proj.scales": np.full((8, 256, 8), 127, dtype=np.uint8),
        "model.layers.0.mlp.experts.up_proj.bias": np.zeros((8, 256), dtype=np.float16),
        "model.layers.0.mlp.experts.down_proj.weight": np.ones((8, 256, 8, 4), dtype=np.uint32),
        "model.layers.0.mlp.experts.down_proj.scales": np.full((8, 256, 8), 127, dtype=np.uint8),
        "model.layers.0.mlp.experts.down_proj.bias": np.zeros((8, 256), dtype=np.float16),
    }
    packed = pack_gpt_oss_layer_tensors(w, 0, "sliding_attention", 256, 512)
    assert packed["qkv_w"].shape == (640, 32)
    assert packed["gu_w"].shape == (8, 512, 8, 4)
    assert packed["qkv_bits"] == 4 and packed["qkv_gs"] == 64
    assert packed["r_bits"] == 8 and packed["r_gs"] == 64

    pw, ps, pb, n2o, o2n = prune_lm_head_tensors(
        np.arange(100 * 8, dtype=np.uint32).reshape(100, 8),
        np.ones((100, 4), dtype=np.float16),
        np.zeros((100, 4), dtype=np.float16),
        np.array([2, 10, 55], dtype=np.int32),
    )
    assert pw.shape == (3, 8) and ps.shape == (3, 4)
    assert int(o2n[55]) == 2 and int(o2n[1]) == -1 and int(n2o[2]) == 55


def test_engine_planning_server_pipeline_and_cli_cpu(tmp_path: Path) -> None:
    assert split_lcp_ids([[1, 2, 3]]) == ([], [[1, 2, 3]])
    assert split_lcp_ids([[1, 2, 3, 4], [1, 2, 3]]) == ([1, 2], [[3, 4], [3]])
    assert split_lcp_ids([[1, 9], [2, 9]]) == ([], [[1, 9], [2, 9]])

    assert select_gpt_oss_plan("auto", 1, None) == ("b1", 1)
    assert select_gpt_oss_plan("auto", 80, None) == ("batched", 64)
    assert select_gpt_oss_plan("b1", 80, 64) == ("b1", 1)

    assert resolve_reasoning_params(has_schema=False, reasoning_effort=None, reasoning_budget=None, enable_thinking=False) == ("low", 0)
    assert resolve_reasoning_params(has_schema=True, reasoning_effort=None, reasoning_budget=None, enable_thinking=False) == ("skip", 0)
    assert resolve_reasoning_params(has_schema=True, reasoning_effort="medium", reasoning_budget=None, enable_thinking=True) == ("medium", 80)
    assert resolve_reasoning_params(has_schema=True, reasoning_effort=None, reasoning_budget=48, enable_thinking=False) == ("low", 48)

    # CLI parser smoke check
    parser = build_parser()
    ns = parser.parse_args(["serve", "--port", "9091", "--coalesce-ms", "5", "--reasoning-effort", "skip"])
    assert ns.cmd == "serve" and ns.port == 9091 and ns.reasoning_effort == "skip"

    # Mock engine for OpenAI server, coalescing, SSE, UnifiedOpenAIServer, and Parquet/JSONL pipelines
    calls_lock = threading.Lock()
    batch_sizes: list[int] = []

    class _MockGptOssEngine:
        model_name = "openai/gpt-oss-20b"
        default_mode = "auto"
        default_reasoning_effort = "low"
        model = SimpleNamespace(vocab_n=131072)

        def generate_batch(self, prompts, **kwargs):
            with calls_lock:
                batch_sizes.append(len(prompts))
            items = [
                GenerationItem(
                    text=f'{{"doc":{i},"eff":"{kwargs.get("reasoning_effort", "low")}"}}',
                    parsed={"doc": i},
                    token_ids=[1, 2, 3],
                    prompt_tokens=12,
                    completion_tokens=6,
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    reasoning="brief thought" if kwargs.get("reasoning_effort") != "skip" else None,
                )
                for i in range(len(prompts))
            ]
            return GenerationBatchResult(items=items, stats={"docs": len(prompts), "elapsed_sec": 0.01, "decode_sec": 0.005, "docs_per_sec": 100.0})

    eng = _MockGptOssEngine()
    with FastGptOssServer(eng, port=0, coalesce_ms=25.0, max_batch_size=8) as srv:
        base = f"http://127.0.0.1:{srv.port}"
        with urllib.request.urlopen(f"{base}/health") as r:
            hp = json.loads(r.read().decode("utf-8"))
        assert hp["status"] == "ok" and hp["vocab_out"] == 131072

        def _post(path: str, body: dict):
            req = urllib.request.Request(
                f"{base}{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                return resp.read().decode("utf-8")

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [
                pool.submit(
                    _post,
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": f"hi {k}"}], "reasoning_effort": "low"},
                )
                for k in range(3)
            ]
            outs = [json.loads(f.result()) for f in futs]
        assert len(outs) == 3
        assert outs[0]["choices"][0]["message"]["reasoning"] == "brief thought"
        assert max(batch_sizes) >= 2

        sse_body = _post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "s"}], "stream": True},
        )
        assert "chat.completion.chunk" in sse_body and "[DONE]" in sse_body

    with UnifiedOpenAIServer(chat_engines={"gpt-oss-20b": srv.adapter}, port=0) as usrv:
        req = urllib.request.Request(
            f"http://127.0.0.1:{usrv.port}/v1/chat/completions",
            data=json.dumps({"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "x"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            uresp = json.loads(r.read().decode("utf-8"))
        assert "fast_gpt_oss" in uresp

    # JSONL + Parquet extraction
    in_jl = tmp_path / "in.jsonl"
    out_jl = tmp_path / "out.jsonl"
    in_jl.write_text('{"id":1,"text":"doc a"}\n{"id":2,"text":"doc b"}\n', encoding="utf-8")
    jstats = extract_jsonl(eng, in_jl, out_jl, system_prompt="Extract json", reasoning_effort="skip")
    assert jstats["docs"] == 2 and out_jl.exists()

    in_pq = tmp_path / "in.parquet"
    out_pq_dir = tmp_path / "out_pq"
    pq.write_table(
        pa.table(
            {
                "doc_id": ["a", "b"],
                "body": ["p1\np2", "x1\nx2"],
                "is_ocr": [False, True],
            }
        ),
        in_pq,
    )
    summary = extract_parquet(
        eng,
        in_pq,
        out_pq_dir,
        system_prompt="Extract json",
        filter_ocr=True,
        drop_text_col=True,
        reasoning_effort="skip",
    )
    assert summary.docs_processed == 1
    out_tbl = pq.read_table(list(out_pq_dir.glob("*.parquet"))[0])
    assert out_tbl.num_rows == 1 and "body" not in out_tbl.column_names


def test_batched_wave_multitoken_ff_and_strict_schema_cpu(tmp_path: Path, monkeypatch) -> None:
    """Verify multi-token terminal FF drains all tokens in `_run_batched_wave` and strict token validation runs."""
    from collections import OrderedDict
    import sys
    from fast_gpt_oss.engine import FastGptOssEngine

    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    vocab_n = tok.vocab_size

    class _FakeArr:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr)

        @property
        def shape(self):
            return self._arr.shape

        def astype(self, dt):
            return self._arr.astype(dt)

        def tolist(self):
            return self._arr.tolist()

        def item(self):
            return self._arr.item()

        def __getitem__(self, key):
            return _FakeArr(self._arr[key])

    class _FakeMx:
        int32 = np.int32
        float32 = np.float32

        @staticmethod
        def array(val, dtype=None):
            return _FakeArr(np.asarray(val, dtype=dtype))

        @staticmethod
        def argmax(val, axis=-1):
            arr = val._arr if isinstance(val, _FakeArr) else np.asarray(val)
            return _FakeArr(np.argmax(arr, axis=axis))

        @staticmethod
        def eval(*_args):
            return None

        @staticmethod
        def clear_cache():
            return None

        @staticmethod
        def reset_peak_memory():
            return None

        @staticmethod
        def get_peak_memory():
            return 0

    import mlx
    monkeypatch.setattr(mlx, "core", _FakeMx)
    monkeypatch.setitem(sys.modules, "mlx.core", _FakeMx)

    colon_tid = tok.encode('":')[0]
    true_tid = tok._tok.token_to_id("true")
    t_byte_tid = ord("t")  # consuming single 't' triggers an even longer multi-token FF ("rue,\"tail\":\"xyz\"}")

    class _FakeModel:
        def __init__(self) -> None:
            self.new_to_old = np.arange(vocab_n, dtype=np.int32)
            self.vocab_n = vocab_n
            self.step_calls = 0

        def compute_shared_prefix(self, prefix_ids: list[int]):
            return {"L_sys": len(prefix_ids)}

        def alloc_batch_cache(self, sp, B: int, suf_max: int, max_dec: int):
            return {"B": B, "sp": sp, "suf_max": suf_max, "max_dec": max_dec}

        def prefill_batch_suffixes(self, suf_lists, bc, chunk_docs: int = 8):
            lg = np.full((len(suf_lists), vocab_n), -20.0, dtype=np.float32)
            lg[:, colon_tid] = 10.0
            return _FakeArr(lg)

        def decode_step_batch(self, tok_mx, bc, samp_indices=None, skip_lm_head: bool = False):
            self.step_calls += 1
            if skip_lm_head:
                return None
            arr = tok_mx._arr if isinstance(tok_mx, _FakeArr) else np.asarray(tok_mx)
            lg = np.full((len(arr), vocab_n), -20.0, dtype=np.float32)
            if self.step_calls == 1 and len(arr) >= 2:
                lg[0, true_tid] = 10.0      # slot 0 takes 12-token FF
                lg[1, t_byte_tid] = 10.0    # slot 1 takes 15-token FF -> finishes later via b1_tail_mode
            else:
                lg[:, true_tid] = 10.0
            if samp_indices is not None:
                sidx = samp_indices._arr if isinstance(samp_indices, _FakeArr) else np.asarray(samp_indices)
                lg = lg[sidx]
            return _FakeArr(lg)

        def compact_batch_cache(self, bc, keep_local):
            bc["B"] = len(keep_local)

        def extract_b1_cache(self, bc, local_idx: int, max_rem: int):
            return [{"slot": local_idx, "max_rem": max_rem}]

        def forward_b1(self, tok_mx, b1_cache, last_only: bool = True):
            lg = np.full((1, 1, vocab_n), -20.0, dtype=np.float32)
            lg[:, :, true_tid] = 10.0
            return _FakeArr(lg)

    eng = FastGptOssEngine.__new__(FastGptOssEngine)
    eng.tokenizer = tok
    eng.model = _FakeModel()
    eng.default_mode = "batched"
    eng.default_reasoning_effort = "skip"
    eng.min_prefix_cache_tokens = 1
    eng._max_prefix_cache = 8
    eng._max_grammar_cache = 8
    eng._prefix_cache = OrderedDict()
    eng._grammar_cache = OrderedDict()
    eng._lock = threading.RLock()
    eng._dummy_pad_id = 0

    # Schema: after picking `true` or `t`, llguidance forces >=12 tokens ending with `}` (`is_stopped()==True`!).
    schema = {
        "type": "object",
        "properties": {"a": {"type": "boolean"}, "tail": {"const": "xyz"}},
        "required": ["a", "tail"],
        "additionalProperties": False,
    }
    gh_check = GrammarHelper(tok, schema, whitespace_flexible=False, eos_token_id=tok.return_token_id)
    m_chk = gh_check.new_matcher()
    assert m_chk.consume_token(colon_tid)
    assert m_chk.consume_token(true_tid)
    assert len(m_chk.compute_ff_tokens()) >= 10  # Confirms multi-token terminal FF under test

    res = eng.generate_batch(
        ["doc one", "doc two"],
        system_prompt="Extract schema.",
        json_schema=schema,
        whitespace_flexible=False,
        reasoning_effort="skip",
        mode="batched",
        max_tokens=64,
    )
    assert len(res.items) == 2
    assert res.stats["ff_lm_head_skipped_steps"] >= 10
    for item in res.items:
        assert item.parsed == {"a": True, "tail": "xyz"}
        assert item.valid_json is True
        assert item.valid_schema is True
        assert item.finish_reason == "stop"

    # Strict token sequence check must still reject invalid token formatting even when json.loads succeeds.
    strict_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    bad_space_ids = tok.encode('{"ok": true}')  # invalid when whitespace_flexible=False, valid json.loads

    class _BadTokenEng(FastGptOssEngine):
        def _run_batched_wave(self, *args, **kwargs):
            return {
                "out_ids": [bad_space_ids, tok.encode('{"ok":true}')],
                "ff_skipped": [0, 0],
                "mask_hits": [0, 0],
                "finished": [True, True],
                "lcp": 0,
                "prefix_sec": 0.0,
                "prefix_hit": False,
                "prefill_sec": 0.0,
                "decode_sec": 0.01,
                "steps": 1,
                "compactions": 0,
            }

    eng2 = _BadTokenEng.__new__(_BadTokenEng)
    eng2.tokenizer = tok
    eng2.model = _FakeModel()
    eng2.default_mode = "batched"
    eng2.default_reasoning_effort = "skip"
    eng2.min_prefix_cache_tokens = 1
    eng2._max_prefix_cache = 8
    eng2._max_grammar_cache = 8
    eng2._prefix_cache = OrderedDict()
    eng2._grammar_cache = OrderedDict()
    eng2._lock = threading.RLock()
    eng2._dummy_pad_id = 0

    res2 = eng2.generate_batch(
        ["a", "b"],
        system_prompt="sys",
        json_schema=strict_schema,
        whitespace_flexible=False,
        reasoning_effort="skip",
        mode="batched",
        max_tokens=32,
    )
    assert res2.items[0].valid_json is True and res2.items[0].valid_schema is False
    assert res2.items[1].valid_json is True and res2.items[1].valid_schema is True


def test_http_adapter_whitespace_flexible_and_compact_contract_cpu(tmp_path: Path, monkeypatch) -> None:
    """Verify `FastGptOssOpenAIAdapter` does not inject `compact=True` and preserves `eff_ws`."""
    from collections import OrderedDict
    import sys
    from fast_gpt_oss.engine import FastGptOssEngine
    from fast_gpt_oss.server import FastGptOssOpenAIAdapter

    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    vocab_n = tok.vocab_size

    class _NoopMx:
        @staticmethod
        def reset_peak_memory():
            return None

        @staticmethod
        def get_peak_memory():
            return 0

        @staticmethod
        def clear_cache():
            return None

        @staticmethod
        def set_cache_limit(_limit):
            return 0

    import mlx
    monkeypatch.setattr(mlx, "core", _NoopMx)
    monkeypatch.setitem(sys.modules, "mlx.core", _NoopMx)

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    spaced_ids = tok.encode('{"ok": true}')
    compact_ids = tok.encode('{"ok":true}')

    class _SpyGptOssEngine(FastGptOssEngine):
        model_name = "openai/gpt-oss-20b"

        def __init__(self) -> None:
            self.tokenizer = tok
            self.model = SimpleNamespace(new_to_old=np.arange(vocab_n, dtype=np.int32), vocab_n=vocab_n)
            self.default_mode = "b1"
            self.default_reasoning_effort = "skip"
            self.min_prefix_cache_tokens = 1
            self._max_prefix_cache = 8
            self._max_grammar_cache = 8
            self._prefix_cache = OrderedDict()
            self._grammar_cache = OrderedDict()
            self._lock = threading.RLock()
            self._dummy_pad_id = 0
            self.received_kwargs: list[dict[str, object]] = []
            self.observed_eff_ws: list[tuple[bool, bool | None]] = []

        def generate_batch(self, prompts, **kwargs):
            self.received_kwargs.append(dict(kwargs))
            return super().generate_batch(prompts, **kwargs)

        def _run_b1_single(self, prompt_ids, *, grammar, whitespace_flexible, **kwargs):
            self.observed_eff_ws.append(
                (bool(whitespace_flexible), None if grammar is None else grammar.whitespace_flexible)
            )
            out_ids = list(spaced_ids if whitespace_flexible else compact_ids)
            return {
                "out_ids": out_ids,
                "ana_ids": [],
                "ff_skipped": 0,
                "mask_hits": 0,
                "finished": True,
                "prefill_sec": 0.0,
                "decode_sec": 0.001,
                "steps": 1,
            }

    eng = _SpyGptOssEngine()
    adapter = FastGptOssOpenAIAdapter(eng)
    assert adapter.default_whitespace_flexible is True

    with FastGptOssServer(eng, port=0) as srv:
        base = f"http://127.0.0.1:{srv.port}"

        def _post(path: str, payload: dict) -> dict:
            req = urllib.request.Request(
                f"{base}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))

        rf = {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}

        # 1. Default chat request: default_whitespace_flexible=True, compact NOT injected -> eff_ws=True.
        out_default = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": "d"}], "response_format": rf})
        assert out_default["choices"][0]["message"]["content"] == '{"ok": true}'
        assert "compact" not in eng.received_kwargs[-1]
        assert eng.received_kwargs[-1]["whitespace_flexible"] is True
        assert eng.observed_eff_ws[-1] == (True, True)
        assert out_default["fast_gpt_oss"]["valid_schema_count"] == 1

        # 2. Explicit whitespace_flexible=True: compact NOT injected -> eff_ws=True.
        out_ws_true = _post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "t"}], "response_format": rf, "whitespace_flexible": True},
        )
        assert out_ws_true["choices"][0]["message"]["content"] == '{"ok": true}'
        assert "compact" not in eng.received_kwargs[-1]
        assert eng.observed_eff_ws[-1] == (True, True)
        assert out_ws_true["fast_gpt_oss"]["valid_schema_count"] == 1

        # 3. Default /v1/completions also must NOT inject compact=True -> eff_ws=True.
        out_cmpl = _post("/v1/completions", {"prompt": "c", "response_format": rf})
        assert out_cmpl["choices"][0]["text"] == '{"ok": true}'
        assert "compact" not in eng.received_kwargs[-1]
        assert eng.observed_eff_ws[-1] == (True, True)

        # 4. Explicit compact=True overrides whitespace_flexible -> eff_ws=False (compact JSON).
        out_compact = _post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "k"}], "response_format": rf, "compact": True},
        )
        assert out_compact["choices"][0]["message"]["content"] == '{"ok":true}'
        assert eng.received_kwargs[-1]["compact"] is True
        assert eng.observed_eff_ws[-1] == (False, False)
        assert out_compact["fast_gpt_oss"]["valid_schema_count"] == 1

        # 5. Explicit whitespace_flexible=False selects compact JSON -> eff_ws=False.
        out_ws_false = _post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "f"}], "response_format": rf, "whitespace_flexible": False},
        )
        assert out_ws_false["choices"][0]["message"]["content"] == '{"ok":true}'
        assert "compact" not in eng.received_kwargs[-1]
        assert eng.received_kwargs[-1]["whitespace_flexible"] is False
        assert eng.observed_eff_ws[-1] == (False, False)
        assert out_ws_false["fast_gpt_oss"]["valid_schema_count"] == 1


# ---------------------------------------------------------------------------
# Live Metal 4 GPU Synthetic Model Test (M4 Pro hardware kernels end-to-end)
# ---------------------------------------------------------------------------


def _build_synthetic_mxfp4_model_dir(model_dir: Path, tok_dir: Path) -> Path:
    """Create a 2-layer `MXFP4 + Affine-Q4` `gpt-oss` checkpoint for real Metal kernel testing."""
    import mlx.core as mx

    rng = np.random.default_rng(7)
    H = 256
    D_FF = 256
    N_Q = 8
    N_KV = 1
    HD = 64
    SWA = 16
    E = 16
    V = 272  # fits cleanly around synthetic tokenizer IDs

    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_json = _write_synthetic_o200k_tokenizer(tok_dir)
    (model_dir / "tokenizer.json").write_text(tok_json.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = {
        "hidden_size": H,
        "intermediate_size": D_FF,
        "num_hidden_layers": 2,
        "num_attention_heads": N_Q,
        "num_key_value_heads": N_KV,
        "head_dim": HD,
        "sliding_window": SWA,
        "rms_norm_eps": 1e-5,
        "vocab_size": V,
        "num_local_experts": E,
        "num_experts_per_tok": 4,
        "swiglu_limit": 7.0,
        "rope_theta": 150000.0,
        "rope_scaling": {
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "layer_types": ["sliding_attention", "full_attention"],
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def q4(rows: int, cols: int, scale: float = 0.04):
        f = mx.array((rng.standard_normal((rows, cols)) * scale).astype(np.float16))
        wq, sq, bq = mx.quantize(f, group_size=64, bits=4, mode="affine")
        return wq, sq.astype(mx.float16), bq.astype(mx.float16)

    def q8(rows: int, cols: int, scale: float = 0.08):
        f = mx.array((rng.standard_normal((rows, cols)) * scale).astype(np.float16))
        wq, sq, bq = mx.quantize(f, group_size=64, bits=8, mode="affine")
        return wq, sq.astype(mx.float16), bq.astype(mx.float16)

    def mxfp4(e: int, rows: int, cols: int, scale: float = 0.04):
        f = mx.array((rng.standard_normal((e, rows, cols)) * scale).astype(np.float16))
        wq, sq = mx.quantize(f, group_size=32, bits=4, mode="mxfp4")
        return wq, sq

    tensors: dict[str, mx.array] = {}
    tensors["model.embed_tokens.weight"], tensors["model.embed_tokens.scales"], tensors["model.embed_tokens.biases"] = q4(V, H, 0.25)
    tensors["model.norm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
    tensors["lm_head.weight"], tensors["lm_head.scales"], tensors["lm_head.biases"] = q4(V, H, 0.05)

    for li in range(2):
        p = f"model.layers.{li}."
        tensors[p + "input_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[p + "post_attention_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        tensors[p + "self_attn.sinks"] = mx.array(rng.normal(0.0, 0.1, (N_Q,)).astype(np.float32))
        for name, out_d, in_d in [
            ("self_attn.q_proj", N_Q * HD, H),
            ("self_attn.k_proj", N_KV * HD, H),
            ("self_attn.v_proj", N_KV * HD, H),
            ("self_attn.o_proj", H, N_Q * HD),
        ]:
            tw, ts, tb = q4(out_d, in_d)
            tensors[f"{p}{name}.weight"] = tw
            tensors[f"{p}{name}.scales"] = ts
            tensors[f"{p}{name}.biases"] = tb
            tensors[f"{p}{name}.bias"] = mx.array(rng.normal(0.0, 0.01, (out_d,)).astype(np.float16))

        rw, rs, rb = q8(E, H, 0.25)
        tensors[p + "mlp.router.weight"] = rw
        tensors[p + "mlp.router.scales"] = rs
        tensors[p + "mlp.router.biases"] = rb
        tensors[p + "mlp.router.bias"] = mx.array(np.linspace(-0.4, 0.4, E, dtype=np.float16))

        for ename in ("gate_proj", "up_proj", "down_proj"):
            ew, es = mxfp4(E, D_FF if ename != "down_proj" else H, H if ename != "down_proj" else D_FF)
            tensors[f"{p}mlp.experts.{ename}.weight"] = ew
            tensors[f"{p}mlp.experts.{ename}.scales"] = es
            tensors[f"{p}mlp.experts.{ename}.bias"] = mx.zeros((E, D_FF if ename != "down_proj" else H), dtype=mx.float16)

    mx.save_safetensors(str(model_dir / "model.safetensors"), tensors)
    return model_dir


def test_metal4_kernels_and_batch_parity_gpu(tmp_path: Path) -> None:
    import mlx.core as mx

    mdir = _build_synthetic_mxfp4_model_dir(tmp_path / "ckpt", tmp_path / "tok")
    m = FastGptOssModel(mdir, vocab_out=None, compile_kernels=True)

    # 1. yarn_rope_split_b vs mx.fast.rope
    mx.random.seed(99)
    b_chk = 6
    qkv_t = mx.random.normal((b_chk, m.qkv_dim)).astype(mx.float16)
    pos_t = mx.array([3, 12, 19, 25, 40, 128], dtype=mx.int32)
    q_got, k_got, _ = m.kernels["yarn_rope_split_b"](
        inputs=[qkv_t, pos_t, m.rope_freqs, m.mscale_arr],
        grid=((m.n_q + m.n_kv) * 32, b_chk, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b_chk, m.n_q, m.hd), (b_chk, m.n_kv, 1, m.hd), (b_chk, m.n_kv, 1, m.hd)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
    )
    mx.eval(q_got, k_got)
    for bi in range(b_chk):
        qb, kb, _ = mx.split(qkv_t[bi : bi + 1], m.qkv_split_idx, axis=-1)
        qb = qb.reshape(1, 1, m.n_q, m.hd).transpose(0, 2, 1, 3)
        kb = kb.reshape(1, 1, m.n_kv, m.hd).transpose(0, 2, 1, 3)
        qr, kr = m._apply_rope(qb, kb, offset=int(pos_t[bi].item()))
        mx.eval(qr, kr)
        assert float(mx.max(mx.abs(qr[0, :, 0, :].astype(mx.float32) - q_got[bi].astype(mx.float32)))) < 5e-3
        assert float(mx.max(mx.abs(kr[0].astype(mx.float32) - k_got[bi].astype(mx.float32)))) < 5e-3

    # 2. Hybrid Metal 4 MPP + Cold QMV MoE vs MLX gather_qmm reference at B=32 (hits BOTH hot >=9 and cold <=8!)
    ld0 = m.layers[0]
    b_moe = 32
    h_in = (mx.random.normal((b_moe, m.H)) * 4.0).astype(mx.bfloat16)
    pn = m.rms_f16(h_in, ld0["pa_ln"])
    rl = m._linear(
        pn, ld0["r_w"], ld0["r_s"], ld0["r_b"], is_quant=True, bits=ld0["r_bits"], group_size=ld0["r_gs"], bias=ld0["r_bias"]
    )
    inds, scores = m._route_c(rl)
    n_assign = b_moe * 4
    max_hot = max(1, (n_assign // 9) + 1)
    tok_idx, slot_idx, rweights, hot_meta, cold_meta, meta_counts = m.kernels["moe_build_tiles"](
        inputs=[inds.astype(mx.uint32), scores.astype(mx.float16), mx.array([b_moe], dtype=mx.uint32)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_assign,), (n_assign,), (n_assign,), (max_hot * 3,), (32 * 3,), (2,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float16, mx.uint32, mx.uint32, mx.uint32],
    )
    mid = m.kernels["moe_batch_gu_unified"](
        inputs=[pn, tok_idx, ld0["gu_w"], ld0["gu_s"], ld0["gu_bias"], hot_meta, cold_meta, meta_counts],
        grid=((m.d_ff // 8) * 128, max_hot + m.num_local_experts, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_assign, m.d_ff)],
        output_dtypes=[mx.float16],
    )[0]
    ys = m.kernels["moe_batch_dn_unified"](
        inputs=[mid, slot_idx, rweights, ld0["dn_w"], ld0["dn_s"], ld0["dn_bias"], hot_meta, cold_meta, meta_counts],
        grid=((m.H // 16) * 128, max_hot + m.num_local_experts, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b_moe, 4, m.H)],
        output_dtypes=[mx.float16],
    )[0]
    h_got = m.kernels["moe_reduce4_resid_bf16"](
        inputs=[ys, h_in],
        grid=(m.H // 4, b_moe, 1),
        threadgroup=(16, 1, 1),
        output_shapes=[(b_moe, m.H)],
        output_dtypes=[mx.bfloat16],
    )[0]

    fl = inds.reshape(-1)
    ord_i = mx.argsort(fl)
    inv_i = mx.argsort(ord_i)
    inds_e = fl[ord_i]
    hx = pn.reshape(-1, 1, m.H)[ord_i // 4]
    gu_ref = mx.gather_qmm(hx, ld0["gu_w"], ld0["gu_s"], rhs_indices=inds_e, transpose=True, group_size=32, bits=4, mode="mxfp4", sorted_indices=True)  # guard: allow-lhs-indices
    act_ref = m._swiglu_c(gu_ref.reshape(-1, 2 * m.d_ff), ld0["gu_bias"][inds_e]).reshape(*gu_ref.shape[:-1], m.d_ff)
    y_ref = mx.gather_qmm(act_ref, ld0["dn_w"], ld0["dn_s"], rhs_indices=inds_e, transpose=True, group_size=32, bits=4, mode="mxfp4", sorted_indices=True)  # guard: allow-lhs-indices
    y_ref = (y_ref.reshape(-1, m.H) + ld0["dn_bias"][inds_e])[inv_i].reshape(b_moe, 4, m.H)
    h_ref = h_in + mx.sum(y_ref * scores[..., None].astype(mx.float32), axis=-2).astype(mx.bfloat16)
    mx.eval(h_got, h_ref, meta_counts)
    assert int(meta_counts[0].item()) >= 1 and int(meta_counts[1].item()) >= 1
    assert float(mx.max(mx.abs(h_got.astype(mx.float32) - h_ref.astype(mx.float32)))) < 0.08

    # 4. End-to-end FastGptOssEngine on the synthetic MXFP4 checkpoint (B=1, batched, schema, budgeted analysis!)
    from fast_gpt_oss.engine import FastGptOssEngine

    eng_home = tmp_path / "eng_home"
    with FastGptOssEngine(home=eng_home, model_dir=mdir, vocab_out=None, warmup=True) as eng:
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer", "enum": [1, 2]}},
            "required": ["ok", "n"],
            "additionalProperties": False,
        }
        g1 = eng.generate(
            "hello world",
            system_prompt="hello",
            json_schema=schema,
            max_tokens=24,
            mode="b1",
            compact=True,
        )
        assert g1.valid_json and g1.valid_schema and isinstance(g1.parsed["ok"], bool)

        gb = eng.generate_batch(
            ["hello alpha", "hello beta", "hello world", "hello true"],
            system_prompt="hello",
            json_schema=schema,
            max_tokens=24,
            mode="batched",
            batch_size=4,
            compact=True,
        )
        assert len(gb.items) == 4
        assert all(it.valid_json and it.valid_schema for it in gb.items)

        g_bud = eng.generate(
            "hello world",
            system_prompt="hello",
            json_schema=schema,
            max_tokens=24,
            reasoning_budget=3,
            compact=True,
        )
        assert g_bud.valid_json and g_bud.valid_schema

        g_chat = eng.generate("hello world", max_tokens=10, temperature=0.4, seed=11)
        assert g_chat.completion_tokens >= 1

    # 3. Full 2-layer prefill + decode parity across:
    #    - Case A: L_sys == 0 (no shared prefix), partial SWA (< 16)
    #    - Case B: L_sys == 20 (>= SWA=16), variable suffix lens + compaction + extract_b1_cache!
    for prefix_len in (0, 20):
        prefix = [10 + (i % 40) for i in range(prefix_len)]
        suf0 = [50, 51, 52, 53]
        suf1 = [60, 61, 62, 63, 64, 65, 66]
        seq_steps = [71, 72, 73, 74, 75]

        # B=1 reference for seq 0 & seq 1 separately
        ref_logits = []
        for s_ids in (prefix + suf0, prefix + suf1):
            c_ref = m.make_b1_cache(len(s_ids) + len(seq_steps) + 4)
            lg_r = m.forward_b1(mx.array(s_ids, dtype=mx.int32)[None, :], c_ref, last_only=True)
            steps_r = [lg_r[0, 0]]
            for dt in seq_steps:
                lg_s = m.forward_b1(mx.array([[dt]], dtype=mx.int32), c_ref, last_only=True)
                steps_r.append(lg_s[0, 0])
            mx.eval(*steps_r)
            ref_logits.append(steps_r)

        # Batched Hydragen v2 + Circular SWA + Hybrid MoE
        sp = m.compute_shared_prefix(prefix)
        bc = m.alloc_batch_cache(sp, B=2, suf_max=len(suf1), max_dec=len(seq_steps) + 2)
        lg_b = m.prefill_batch_suffixes([suf0, suf1], bc, chunk_docs=2)
        mx.eval(lg_b)
        for bi in (0, 1):
            diff = float(mx.max(mx.abs(lg_b[bi].astype(mx.float32) - ref_logits[bi][0].astype(mx.float32))))
            assert diff < 0.15

        # Run 3 batched decode steps across both slots
        for step_i in range(3):
            toks = mx.array([seq_steps[step_i], seq_steps[step_i]], dtype=mx.int32)
            lg_step = m.decode_step_batch(toks, bc)
            mx.eval(lg_step)
            for bi in (0, 1):
                diff = float(mx.max(mx.abs(lg_step[bi].astype(mx.float32) - ref_logits[bi][step_i + 1].astype(mx.float32))))
                assert diff < 0.15

        # Compact down to slot 1 -> extract_b1_cache -> verify final 2 decode steps in forward_b1!
        m.compact_batch_cache(bc, [1])
        c1_handoff = m.extract_b1_cache(bc, 0, max_rem=8)
        for step_i in range(3, 5):
            lg_ho = m.forward_b1(mx.array([[seq_steps[step_i]]], dtype=mx.int32), c1_handoff, last_only=True)[0, 0]
            mx.eval(lg_ho)
            diff = float(mx.max(mx.abs(lg_ho.astype(mx.float32) - ref_logits[1][step_i + 1].astype(mx.float32))))
            assert diff < 0.15

    # 5. Pass-2 optimizations on live Metal 4 checkpoint:
    #    (a) skip_lm_head (M_samp == 0) & samp_indices (0 < M_samp < B) bit-exactness + greedy parity
    #    (b) per_doc_grammars enforcing distinct per-row schemas in one generate_batch wave
    #    (c) temperature > 0, top_p = 0.95 GPU top-128 tail-mass bound fast path (0 full-row copies on unmasked steps)
    sp_p2 = m.compute_shared_prefix([12, 13, 14, 15, 16, 17, 18, 19])
    sufs_p2 = [[40, 41, 42], [50, 51, 52, 53], [60, 61]]
    bc_ref = m.alloc_batch_cache(sp_p2, B=3, suf_max=4, max_dec=8)
    bc_opt = m.alloc_batch_cache(sp_p2, B=3, suf_max=4, max_dec=8)
    lg_p0_ref = m.prefill_batch_suffixes(sufs_p2, bc_ref, chunk_docs=3)
    lg_p0_opt = m.prefill_batch_suffixes(sufs_p2, bc_opt, chunk_docs=3)
    mx.eval(lg_p0_ref, lg_p0_opt)

    t_step0 = mx.array([71, 72, 73], dtype=mx.int32)
    _ = m.decode_step_batch(t_step0, bc_ref)
    out_skip = m.decode_step_batch(t_step0, bc_opt, skip_lm_head=True)
    assert out_skip is None
    mx.eval(
        bc_ref["rope_pos_arr"],
        bc_opt["rope_pos_arr"],
        *(lc.get("k_suf", lc.get("k_swa")) for lc in bc_ref["layers"]),
        *(lc.get("v_suf", lc.get("v_swa")) for lc in bc_ref["layers"]),
        *(lc.get("k_suf", lc.get("k_swa")) for lc in bc_opt["layers"]),
        *(lc.get("v_suf", lc.get("v_swa")) for lc in bc_opt["layers"]),
    )

    t_step1 = mx.array([74, 75, 76], dtype=mx.int32)
    lg_s1_full = m.decode_step_batch(t_step1, bc_ref)
    samp_idx = mx.array([0, 2], dtype=mx.int32)
    lg_s1_sliced = m.decode_step_batch(t_step1, bc_opt, samp_indices=samp_idx)
    mx.eval(lg_s1_full, lg_s1_sliced)
    assert lg_s1_sliced.shape == (2, m.vocab_n)
    assert float(mx.max(mx.abs(lg_s1_sliced.astype(mx.float32) - lg_s1_full[samp_idx].astype(mx.float32)))) == 0.0

    with FastGptOssEngine(home=tmp_path / "eng_p2", model_dir=mdir, vocab_out=None, warmup=False) as eng_p2:
        sch_bill = {
            "type": "object",
            "properties": {
                "primary_kind": {"type": "string", "enum": ["bill"]},
                "ok": {"type": "boolean"},
                "summary": {"type": "string", "enum": ["hello"]},
            },
            "required": ["primary_kind", "ok", "summary"],
            "additionalProperties": False,
        }
        sch_hrg = {
            "type": "object",
            "properties": {
                "primary_kind": {"type": "string", "enum": ["hearing"]},
                "ok": {"type": "boolean"},
                "domain": {"type": "string", "enum": ["world"]},
            },
            "required": ["primary_kind", "ok", "domain"],
            "additionalProperties": False,
        }
        sch_open = {
            "type": "object",
            "properties": {
                "primary_kind": {"type": "string", "enum": ["bill", "hearing"]},
                "ok": {"type": "boolean"},
                "summary": {"type": "string", "enum": ["world"]},
            },
            "required": ["primary_kind", "ok", "summary"],
            "additionalProperties": False,
        }

        prompts_4 = ["hello world", "hello true", "hello false", "hello bill"]
        grammars_4 = [sch_bill, sch_hrg, None, sch_bill]

        with pytest.raises(ValueError, match="per_doc_grammars"):
            eng_p2.generate_batch(prompts_4, per_doc_grammars=[sch_bill])

        res_opt = eng_p2.generate_batch(
            prompts_4,
            system_prompt="hello",
            json_schema=sch_open,
            per_doc_grammars=grammars_4,
            max_tokens=96,
            temperature=0.0,
            mode="batched",
            batch_size=4,
            compact=True,
        )
        assert res_opt.stats["valid_schema_count"] == 4
        assert res_opt.stats["ff_lm_head_skipped_steps"] > 0
        assert res_opt.items[0].parsed["primary_kind"] == "bill"
        assert res_opt.items[0].parsed["summary"] == "hello"
        assert res_opt.items[1].parsed["primary_kind"] == "hearing"
        assert res_opt.items[1].parsed["domain"] == "world"
        assert res_opt.items[2].parsed["primary_kind"] in ("bill", "hearing")
        assert res_opt.items[2].parsed["summary"] == "world"
        assert res_opt.items[3].parsed["primary_kind"] == "bill"

        # Compare against unskipped/unsliced decode_step_batch (token-for-token parity!)
        orig_decode_step = eng_p2.model.decode_step_batch
        saw_sliced = [0]

        def _spy_decode_step(tok_1d, batch_cache, samp_indices=None, skip_lm_head: bool = False):
            if samp_indices is not None:
                saw_sliced[0] += 1
            return orig_decode_step(tok_1d, batch_cache, samp_indices=samp_indices, skip_lm_head=skip_lm_head)

        eng_p2.model.decode_step_batch = _spy_decode_step  # type: ignore[method-assign]
        res_spy = eng_p2.generate_batch(
            prompts_4,
            system_prompt="hello",
            json_schema=sch_open,
            per_doc_grammars=grammars_4,
            max_tokens=96,
            temperature=0.0,
            mode="batched",
            batch_size=4,
            compact=True,
        )
        assert saw_sliced[0] > 0  # Confirms 0 < M_samp < B sliced lm_head path executed!
        assert [it.token_ids for it in res_spy.items] == [it.token_ids for it in res_opt.items]

        # Also compare against B=1 greedy reference on per_doc_grammars
        res_b1_pd = eng_p2.generate_batch(
            prompts_4,
            system_prompt="hello",
            json_schema=sch_open,
            per_doc_grammars=grammars_4,
            max_tokens=96,
            temperature=0.0,
            mode="b1",
            compact=True,
        )
        assert res_b1_pd.stats["valid_schema_count"] == 4

        # Restore original decode_step_batch and verify against forced full-row decode_step_batch (exact bit/token parity!)
        def _forced_full_decode_step(tok_1d, batch_cache, samp_indices=None, skip_lm_head: bool = False):
            lg_all = orig_decode_step(tok_1d, batch_cache, samp_indices=None, skip_lm_head=False)
            if skip_lm_head:
                mx.eval(lg_all)
                return None
            if samp_indices is not None:
                return lg_all[samp_indices]
            return lg_all

        eng_p2.model.decode_step_batch = _forced_full_decode_step  # type: ignore[method-assign]
        res_ref = eng_p2.generate_batch(
            prompts_4,
            system_prompt="hello",
            json_schema=sch_open,
            per_doc_grammars=grammars_4,
            max_tokens=96,
            temperature=0.0,
            mode="batched",
            batch_size=4,
            compact=True,
        )
        eng_p2.model.decode_step_batch = orig_decode_step  # type: ignore[method-assign]
        assert [it.token_ids for it in res_ref.items] == [it.token_ids for it in res_opt.items]

        # (c) Fast GPU top-128 tail-mass bound sampling at temperature=0.55, top_p=0.95
        res_t_top128 = eng_p2.generate_batch(
            prompts_4,
            system_prompt="hello",
            json_schema=sch_open,
            per_doc_grammars=grammars_4,
            max_tokens=96,
            temperature=0.10,
            top_p=0.95,
            seed=303,
            mode="batched",
            batch_size=4,
            compact=True,
        )
        assert res_t_top128.stats["valid_schema_count"] == 4
        assert all(it.valid_json and it.valid_schema for it in res_t_top128.items)
        assert res_t_top128.stats["top128_fast_hits"] > 0

        # Also test exact T=0.55, top_p=0.95 on peaked GPU logits -> 0 full-vocab host copies!
        orig_lm_s, orig_lm_b = eng_p2.model.lm_s, eng_p2.model.lm_b
        eng_p2.model.lm_s = (orig_lm_s.astype(mx.float32) * 32.0).astype(mx.float16)
        eng_p2.model.lm_b = (orig_lm_b.astype(mx.float32) * 32.0).astype(mx.float16)
        mx.eval(eng_p2.model.lm_s, eng_p2.model.lm_b)
        try:
            res_t055_unmasked = eng_p2.generate_batch(
                ["hello world", "hello true"],
                system_prompt="hello",
                max_tokens=12,
                temperature=0.55,
                top_p=0.95,
                seed=404,
                mode="batched",
                batch_size=2,
            )
            az_enum = [chr(c) for c in range(ord("a"), ord("z") + 1)]
            res_t055_schema = eng_p2.generate_batch(
                ["hello world", "hello true"],
                system_prompt="hello",
                per_doc_grammars=[
                    {
                        "type": "object",
                        "properties": {"primary_kind": {"type": "string", "enum": ["bill"]}, "tag": {"type": "string", "enum": az_enum}},
                        "required": ["primary_kind", "tag"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"primary_kind": {"type": "string", "enum": ["hearing"]}, "tag": {"type": "string", "enum": az_enum}},
                        "required": ["primary_kind", "tag"],
                        "additionalProperties": False,
                    },
                ],
                max_tokens=96,
                temperature=0.55,
                top_p=0.95,
                seed=404,
                mode="batched",
                batch_size=2,
                compact=True,
            )
        finally:
            eng_p2.model.lm_s, eng_p2.model.lm_b = orig_lm_s, orig_lm_b
        assert res_t055_unmasked.stats["top128_fast_hits"] > 0
        assert res_t055_unmasked.stats["full_vocab_logit_copies"] == 0
        assert res_t055_schema.stats["valid_schema_count"] == 2
        assert all(it.valid_json and it.valid_schema for it in res_t055_schema.items)
        assert res_t055_schema.stats["top128_fast_hits"] > 0
        assert res_t055_schema.stats["unmasked_full_vocab_logit_copies"] == 0


def test_top128_tail_mass_bounds_exact_and_fallback_cpu(tmp_path: Path) -> None:
    """Verify exact mathematical tail-mass bound (`0` full-vocab host copies when peaked, lazy fallback when straddling)."""
    from collections import OrderedDict
    import mlx.core as mx
    from fast_gpt_oss.engine import FastGptOssEngine

    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    base_v = tok.vocab_size
    pad_v = 2048  # > 128 so top-128 path is active with >1900 tail entries

    eng = FastGptOssEngine.__new__(FastGptOssEngine)
    eng.tokenizer = tok
    eng.model = SimpleNamespace(
        new_to_old=np.array([i % base_v for i in range(pad_v)], dtype=np.int32),
        vocab_n=pad_v,
    )
    eng.default_mode = "batched"
    eng.default_reasoning_effort = "skip"
    eng.min_prefix_cache_tokens = 1
    eng._max_prefix_cache = 8
    eng._max_grammar_cache = 16
    eng._prefix_cache = OrderedDict()
    eng._grammar_cache = OrderedDict()
    eng._lock = threading.RLock()
    eng._dummy_pad_id = 0

    schema = {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["bill", "hearing"]}},
        "required": ["kind"],
        "additionalProperties": False,
    }
    gh = GrammarHelper(
        tok,
        schema,
        new_to_old=np.arange(base_v, dtype=np.int32),
        whitespace_flexible=False,
        eos_token_id=tok.return_token_id,
    )
    eng.model.new_to_old = np.arange(pad_v, dtype=np.int32)
    # Pad gh.new_to_old to 2048 with valid base token 0 for indices >= base_v (invalid at enum step)
    gh_pad_n2o = np.zeros(pad_v, dtype=np.int32)
    gh_pad_n2o[:base_v] = np.arange(base_v, dtype=np.int32)
    gh_2k = GrammarHelper(
        tok,
        schema,
        new_to_old=gh_pad_n2o,
        whitespace_flexible=False,
        eos_token_id=tok.return_token_id,
    )

    bill_tid = tok._tok.token_to_id("bill")
    hrg_tid = tok._tok.token_to_id("hearing")
    colon_q_ids = tok.encode('":"')

    def _enum_matcher():
        m = gh_2k.new_matcher()
        for tid in colon_q_ids:
            if not m.is_stopped() and m.validate_tokens([tid]):
                m.consume_token(tid)
        return m

    # Case 1: Unmasked peaked distribution at temperature=0.55, top_p=0.95 -> 0 full-row copies + exact RNG parity!
    lg_np = np.full((2, pad_v), -18.0, dtype=np.float32)
    lg_np[0, bill_tid] = 6.2
    lg_np[0, hrg_tid] = 5.4
    # Row 1: Rank-0 token is grammar-INVALID ("hello"), valid tokens at ranks #1 and #2 -> STILL 0 full-row copies!
    hello_tid = tok._tok.token_to_id("hello")
    lg_np[1, hello_tid] = 7.0
    lg_np[1, hrg_tid] = 5.9
    lg_np[1, bill_tid] = 5.1

    lg_mx = mx.array(lg_np, dtype=mx.float32)
    pack_fast = eng._eval_logits_for_sampling(lg_mx, temperature=0.55, top_p=0.95)
    assert pack_fast["mode"] == "top128"

    counters: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}
    m0_fast, m1_fast = _enum_matcher(), _enum_matcher()
    rng_fast = np.random.default_rng(777)
    t0_f, mask0_f = eng._sample_row_from_eval(
        pack_fast, 0, grammar=gh_2k, matcher=m0_fast, temperature=0.55, top_p=0.95, rng=rng_fast, stats_counters=counters
    )
    t1_f, mask1_f = eng._sample_row_from_eval(
        pack_fast, 1, grammar=gh_2k, matcher=m1_fast, temperature=0.55, top_p=0.95, rng=rng_fast, stats_counters=counters
    )
    assert counters["top128_fast_hits"] == 2
    assert counters["full_vocab_logit_copies"] == 0
    assert mask0_f is False
    assert mask1_f is True
    assert t0_f in (bill_tid, hrg_tid) and t1_f in (bill_tid, hrg_tid)

    # Compare against forced full-row ("top1") reference with identical seed=777
    pack_ref = {"mode": "top1", "lg_2d": lg_mx, "top1_cpu": [int(np.argmax(lg_np[0])), int(np.argmax(lg_np[1]))]}
    ref_counters: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}
    m0_ref, m1_ref = _enum_matcher(), _enum_matcher()
    rng_ref = np.random.default_rng(777)
    t0_r, mask0_r = eng._sample_row_from_eval(
        pack_ref, 0, grammar=gh_2k, matcher=m0_ref, temperature=0.55, top_p=0.95, rng=rng_ref, stats_counters=ref_counters
    )
    t1_r, mask1_r = eng._sample_row_from_eval(
        pack_ref, 1, grammar=gh_2k, matcher=m1_ref, temperature=0.55, top_p=0.95, rng=rng_ref, stats_counters=ref_counters
    )
    assert ref_counters["full_vocab_logit_copies"] == 2
    assert (t0_f, mask0_f, t1_f, mask1_f) == (t0_r, mask0_r, t1_r, mask1_r)

    # Case 2: Flat tail-straddling / buried-valid distribution -> lazy full-row fallback!
    lg_flat_np = np.full((1, pad_v), -12.0, dtype=np.float32)
    lg_flat_np[0, base_v : base_v + 200] = 0.0  # 200 invalid tokens occupy all of top-128
    lg_flat_np[0, bill_tid] = -3.0
    lg_flat_np[0, hrg_tid] = -3.1
    lg_flat_mx = mx.array(lg_flat_np, dtype=mx.float32)
    pack_flat = eng._eval_logits_for_sampling(lg_flat_mx, temperature=0.55, top_p=0.95)
    fb_counters: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}
    t_fb, mask_fb = eng._sample_row_from_eval(
        pack_flat,
        0,
        grammar=gh_2k,
        matcher=_enum_matcher(),
        temperature=0.55,
        top_p=0.95,
        rng=np.random.default_rng(99),
        stats_counters=fb_counters,
    )
    assert fb_counters["top128_fast_hits"] == 0
    assert fb_counters["full_vocab_logit_copies"] == 1
    assert mask_fb is True and t_fb in (bill_tid, hrg_tid)

    # Case 3: Valid tokens IN top-128, but massive tail mass outside top-128 straddles top_p=0.95 -> lazy fallback!
    lg_straddle_np = np.full((1, pad_v), 0.0, dtype=np.float32)  # ~1900 tail tokens at 0.0 create huge Z_tail
    lg_straddle_np[0, bill_tid] = 2.5
    lg_straddle_np[0, hrg_tid] = 1.0
    lg_straddle_mx = mx.array(lg_straddle_np, dtype=mx.float32)
    pack_straddle = eng._eval_logits_for_sampling(lg_straddle_mx, temperature=0.55, top_p=0.95)
    st_counters: dict[str, int] = {"top128_fast_hits": 0, "full_vocab_logit_copies": 0}
    t_st, _ = eng._sample_row_from_eval(
        pack_straddle,
        0,
        grammar=gh_2k,
        matcher=_enum_matcher(),
        temperature=0.55,
        top_p=0.95,
        rng=np.random.default_rng(101),
        stats_counters=st_counters,
    )
    assert st_counters["top128_fast_hits"] == 0
    assert st_counters["full_vocab_logit_copies"] == 1
    assert _enum_matcher().validate_tokens([t_st])


def test_dead_kernels_removed_and_norm_bf16_coercion_cpu(tmp_path: Path) -> None:
    """Verify dead Metal kernels were removed and RMSNorm weights are coerced to `bfloat16`."""
    import mlx.core as mx

    sources = render_kernel_sources(KernelSpec())
    for dead_name in (
        "aff_q4_qmv_f16",
        "aff_q4_qmv_resid_bf16",
        "moe_b1_gu_swiglu",
        "moe_b1_dn_resid_bf16",
    ):
        assert dead_name not in sources, f"Dead kernel {dead_name} still compiled"
    assert "unpack4_scaled" not in sources["moe_batch_gu_unified"]["header"]
    assert "e8m0_to_float" in sources["moe_batch_gu_unified"]["header"]

    # Build a synthetic checkpoint with float32 norm weights and verify FastGptOssModel coerces them to bfloat16!
    mdir = _build_synthetic_mxfp4_model_dir(tmp_path / "ckpt_f32ln", tmp_path / "tok_f32ln")
    sf_path = mdir / "model.safetensors"
    raw = mx.load(str(sf_path))
    raw["model.norm.weight"] = raw["model.norm.weight"].astype(mx.float32)
    raw["model.layers.0.input_layernorm.weight"] = raw["model.layers.0.input_layernorm.weight"].astype(mx.float32)
    raw["model.layers.0.post_attention_layernorm.weight"] = raw["model.layers.0.post_attention_layernorm.weight"].astype(mx.float16)
    mx.save_safetensors(str(sf_path), raw)

    m = FastGptOssModel(mdir, vocab_out=None, compile_kernels=False)
    assert m.final_norm.dtype == mx.bfloat16
    assert m.layers[0]["in_ln"].dtype == mx.bfloat16
    assert m.layers[0]["pa_ln"].dtype == mx.bfloat16

    # Verify vocab_out pruning warns cleanly if tokenizer.json & vocab_remap_npz are absent,
    # and loads from cached vocab_remap_npz even when tokenizer.json is absent!
    import warnings

    (mdir / "tokenizer.json").unlink()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m_warn = FastGptOssModel(mdir, vocab_out=128, compile_kernels=False)
    assert m_warn.vocab_n == m_warn.full_vocab_size
    assert any(issubclass(w.category, UserWarning) and "vocab pruning" in str(w.message) for w in rec)

    remap_npz = tmp_path / "remap_128.npz"
    np.savez(remap_npz, kept_old_ids=np.arange(128, dtype=np.int32))
    m_npz = FastGptOssModel(mdir, vocab_out=128, vocab_remap_npz=remap_npz, compile_kernels=False)
    assert m_npz.vocab_n == 128


def test_memory_clamp_wave_sorting_wired_restore_and_paths_cpu(tmp_path: Path, monkeypatch) -> None:
    """Verify <=24 GiB wave clamping, length-sorted waves, wired-limit restore, and `paths.py` artifacts lookup."""
    from collections import OrderedDict
    import warnings
    from fast_gpt_oss.engine import FastGptOssEngine, recommended_max_wave_batch
    import fast_gpt_oss.paths as gpt_paths

    # 1. Unified memory clamping
    assert recommended_max_wave_batch(16 << 30) == 8
    assert recommended_max_wave_batch(24 << 30) == 16
    assert recommended_max_wave_batch(48 << 30) == 64

    assert select_gpt_oss_plan("auto", 64, None, memory_bytes=16 << 30) == ("batched", 8)
    assert select_gpt_oss_plan("auto", 64, None, memory_bytes=24 << 30) == ("batched", 16)
    assert select_gpt_oss_plan("auto", 64, None, memory_bytes=48 << 30) == ("batched", 64)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert select_gpt_oss_plan("batched", 64, 32, memory_bytes=16 << 30) == ("batched", 8)
    assert any(issubclass(w.category, UserWarning) and "Clamping wave batch_size" in str(w.message) for w in rec)

    # 2. `paths.py` checks `artifacts/` instead of stopping on `pyproject.toml`
    fake_pkg = tmp_path / "monorepo" / "packages" / "fast-gpt-oss"
    fake_file = fake_pkg / "src" / "fast_gpt_oss" / "paths.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("", encoding="utf-8")
    (fake_pkg / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.delenv("FAST_GPT_OSS_HOME", raising=False)
    monkeypatch.setattr(gpt_paths, "__file__", str(fake_file))
    assert gpt_paths.resolve_paths().home == (Path("~/.cache") / "fast-gpt-oss").expanduser().resolve()
    (fake_pkg / "artifacts").mkdir()
    assert gpt_paths.resolve_paths().home == fake_pkg.resolve()

    # 3. Length-sorted multi-wave slicing preserves exact original input order & clears cache between waves
    tok_path = _write_synthetic_o200k_tokenizer(tmp_path)
    tok = FastGptOssTokenizer(tok_path)
    seen_wave_lens: list[list[int]] = []

    class _SortingSpyEng(FastGptOssEngine):
        def _run_batched_wave(self, prompt_ids_wave, **_kwargs):
            seen_wave_lens.append([len(p) for p in prompt_ids_wave])
            return {
                "out_ids": [tok.encode(f"len={len(p)}") for p in prompt_ids_wave],
                "ff_skipped": [0] * len(prompt_ids_wave),
                "mask_hits": [0] * len(prompt_ids_wave),
                "finished": [True] * len(prompt_ids_wave),
                "lcp": 0,
                "prefix_sec": 0.0,
                "prefix_hit": False,
                "prefill_sec": 0.0,
                "decode_sec": 0.01,
                "steps": 1,
                "compactions": 0,
            }

    eng = _SortingSpyEng.__new__(_SortingSpyEng)
    eng.tokenizer = tok
    eng.model = SimpleNamespace(vocab_n=tok.vocab_size, new_to_old=np.arange(tok.vocab_size, dtype=np.int32))
    eng.default_mode = "batched"
    eng.default_reasoning_effort = "skip"
    eng.min_prefix_cache_tokens = 1
    eng._max_prefix_cache = 8
    eng._max_grammar_cache = 8
    eng._prefix_cache = OrderedDict()
    eng._grammar_cache = OrderedDict()
    eng._lock = threading.RLock()
    eng._dummy_pad_id = 0
    eng._host_memory_bytes = 48 << 30
    eng._prev_wired_limit = None
    eng._wired_limit_restored = True

    prompts = ["hi", "hello world alpha beta hello world", "a", "hello world beta alpha hello world"]
    res = eng.generate_batch(prompts, mode="batched", batch_size=2, use_chat_template=False)
    assert len(seen_wave_lens) == 2
    assert max(seen_wave_lens[0]) <= min(seen_wave_lens[1]), f"Waves not sorted by length: {seen_wave_lens}"
    # Verify original order restored in `res.items`!
    for i, item in enumerate(res.items):
        assert item.text == f"len={item.prompt_tokens}"
    assert res.items[0].prompt_tokens < res.items[1].prompt_tokens
    assert res.items[2].prompt_tokens < res.items[3].prompt_tokens

    # 4. Wired limit restore in `close()`
    import mlx.core as mx

    wired_calls: list[int] = []
    orig_set_wired = getattr(mx, "set_wired_limit", None)
    if callable(orig_set_wired):
        monkeypatch.setattr(mx, "set_wired_limit", lambda lim: wired_calls.append(lim) or 0)
        eng._prev_wired_limit = 123456789
        eng._wired_limit_restored = False
        eng.close()
        assert wired_calls == [123456789]
        # Second close is idempotent
        eng.close()
        assert wired_calls == [123456789]


def test_hot_mpp_extreme_e8m0_scale_chunked_moe_and_swa8q1kv_gpu(tmp_path: Path) -> None:
    """Verify Hot-MPP has zero NaN/inf at E8M0=141 (6*2^14=98304 > FP16 max) & E8M0=100, chunked >256 MoE, and 8Q:1KV SWA."""
    import mlx.core as mx

    mdir = _build_synthetic_mxfp4_model_dir(tmp_path / "ckpt_hot", tmp_path / "tok_hot")
    m = FastGptOssModel(mdir, vocab_out=None, compile_kernels=True)

    # 1. Hot-MPP vs Cold-QMV vs gather_qmm at extreme E8M0=141 (previously half4 -> inf -> NaN with 0.0!)
    #    and E8M0=100 (previously half4 underflow -> 0.0!).
    ld = dict(m.layers[0])
    E, D_ff, H = m.num_local_experts, m.d_ff, m.H
    ng_h = H // 32
    ng_d = D_ff // 32

    # Pack E2M1 nibble 0x7 (+6.0) on block 0 with E8M0=141 (scale=2^14=16384 -> 6*2^14=98304 > 65504!)
    # and E8M0=100 (scale=2^-27 -> previously flushed to 0.0 in half4!) on block 1.
    gu_w = mx.full((E, 2 * D_ff, ng_h * 4), 0xF7F7F7F7, dtype=mx.uint32)
    dn_w = mx.full((E, H, ng_d * 4), 0xF7F7F7F7, dtype=mx.uint32)
    gu_w_np = np.full((E, 2 * D_ff, ng_h * 4), 0xF7F7F7F7, dtype=np.uint32)
    gu_w_np[0, :, :4] = 0x77777777  # +6.0 across block 0 -> exact non-zero product with E8M0=141!
    dn_w_np = np.full((E, H, ng_d * 4), 0xF7F7F7F7, dtype=np.uint32)
    dn_w_np[0, :, :4] = 0x11111111  # +0.5 across block 0
    gu_s_np = np.full((E, 2 * D_ff, ng_h), 120, dtype=np.uint8)
    gu_s_np[0, :, 0] = 141
    gu_s_np[0, :, 1] = 100
    dn_s_np = np.full((E, H, ng_d), 120, dtype=np.uint8)
    dn_s_np[0, :, 0] = 132
    dn_s_np[0, :, 1] = 100
    ld["gu_w"] = mx.array(gu_w_np, dtype=mx.uint32)
    ld["gu_s"] = mx.array(gu_s_np, dtype=mx.uint8)
    ld["dn_w"] = mx.array(dn_w_np, dtype=mx.uint32)
    ld["dn_s"] = mx.array(dn_s_np, dtype=mx.uint8)

    # 20 tokens all routing slot 0 to expert 0 (20 > 8 -> Hot-MPP with 12 zero-padded rows!) and slot 1..3 to cold experts
    b_ext = 20
    rng = np.random.default_rng(55)
    pn_np = (rng.standard_normal((b_ext, H)) * 1e-4).astype(np.float16)
    # Explicitly put exact 0.0f in half of block 0 (would trigger inf*0.0 -> NaN under old half4 unpack4_scaled!)
    # and positive values in the other half so Hot-MPP produces non-trivial finite outputs!
    pn_np[:, :16] = 0.0
    pn_np[:, 16:32] = np.float16(1e-4)
    pn_ext = mx.array(pn_np, dtype=mx.float16)
    h_ext = mx.zeros((b_ext, H), dtype=mx.bfloat16)

    inds_np = np.zeros((b_ext, 4), dtype=np.int32)
    inds_np[:, 0] = 0
    for i in range(b_ext):
        inds_np[i, 1] = 1 + (i % 5)
        inds_np[i, 2] = 6 + (i % 5)
        inds_np[i, 3] = 11 + (i % 5)
    inds_ext = mx.array(inds_np, dtype=mx.int32)
    scores_ext = mx.full((b_ext, 4), 0.25, dtype=mx.float32)

    h_mpp = m._moe_tile_mpp_chunk(pn_ext, h_ext, inds_ext, scores_ext, ld)
    # Use float32 input for exact reference (avoids MLX float16 gather_qmm's own FP16 dequant overflow at E8M0=141!)
    h_ref = m._moe_gather_qmm_nax(pn_ext.astype(mx.float32), h_ext, inds_ext, scores_ext, ld)
    # Also run B=1 (Cold-QMV only!) for token 0 to verify Hot-MPP == Cold-QMV directly!
    h_cold0 = m._moe_tile_mpp_chunk(pn_ext[:1], h_ext[:1], inds_ext[:1], scores_ext[:1], ld)
    mx.eval(h_mpp, h_ref, h_cold0)

    mpp_f32 = np.asarray(h_mpp.astype(mx.float32))
    ref_f32 = np.asarray(h_ref.astype(mx.float32))
    cold0_f32 = np.asarray(h_cold0.astype(mx.float32))
    assert np.all(np.isfinite(mpp_f32)), "Hot-MPP produced NaN/inf at E8M0=141!"
    assert np.all(np.isfinite(cold0_f32)), "Cold-QMV produced NaN/inf at E8M0=141!"
    assert float(np.max(np.abs(mpp_f32))) > 1.0, "Expected non-trivial signal from E8M0=141 block"
    np.testing.assert_allclose(mpp_f32[0], cold0_f32[0], rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(mpp_f32, ref_f32, rtol=3e-2, atol=3e-2)

    # 2. Chunked tile-table MPP for n_tok > 256 (e.g. n_tok=280) vs gather_qmm reference!
    ld0 = m.layers[0]
    n_big = 280
    h_big = (mx.random.normal((n_big, H)) * 2.0).astype(mx.bfloat16)
    pn_big = m.rms_f16(h_big, ld0["pa_ln"])
    rl_big = m._linear(
        pn_big, ld0["r_w"], ld0["r_s"], ld0["r_b"], is_quant=True, bits=ld0["r_bits"], group_size=ld0["r_gs"], bias=ld0["r_bias"]
    )
    inds_big, scores_big = m._route_c(rl_big)
    h_chunked = m._moe_forward(pn_big, h_big, inds_big, scores_big, ld0)
    h_big_ref = m._moe_gather_qmm_nax(pn_big, h_big, inds_big, scores_big, ld0)
    mx.eval(h_chunked, h_big_ref)
    assert float(mx.max(mx.abs(h_chunked.astype(mx.float32) - h_big_ref.astype(mx.float32)))) < 0.08

    # 3. 8Q:1KV packed `swa128_prefill_mma` on non-multiple-of-8 L_q=37 with Offset=11, Window=16 vs exact float32 ref!
    l_q, off_swa, win_swa = 37, 11, 16
    l_kv = off_swa + l_q
    q_t = (mx.random.normal((1, m.n_q, l_q, m.hd)) * 0.3).astype(mx.float16)
    k_t = (mx.random.normal((1, m.n_kv, l_kv, m.hd)) * 0.3).astype(mx.float16)
    v_t = (mx.random.normal((1, m.n_kv, l_kv, m.hd)) * 0.3).astype(mx.float16)
    sinks_t = ld0["sinks_f32"]
    p_arr = mx.array([l_q, l_kv, off_swa, win_swa, m.scale_u32], dtype=mx.uint32)
    att_got = m.kernels["swa128_prefill_mma"](
        inputs=[q_t, k_t, v_t, sinks_t, p_arr],
        grid=(((l_q + 7) // 8) * 256, m.n_kv, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, m.n_q, l_q, m.hd)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(att_got)

    q_np = np.asarray(q_t.astype(mx.float32))[0]
    k_np = np.asarray(k_t.astype(mx.float32))[0]
    v_np = np.asarray(v_t.astype(mx.float32))[0]
    sinks_np = np.asarray(sinks_t, dtype=np.float32)
    ref_att = np.zeros_like(q_np)
    for hq in range(m.n_q):
        hkv = hq // 8
        sc = (q_np[hq] @ k_np[hkv].T) * m.scale  # [l_q, l_kv]
        q_abs = np.arange(l_q)[:, None] + off_swa
        k_abs = np.arange(l_kv)[None, :]
        valid = (q_abs >= k_abs) & ((q_abs - k_abs) < win_swa)
        sc = np.where(valid, sc, -1e30)
        sink_col = np.full((l_q, 1), sinks_np[hq], dtype=np.float32)
        sc_ext = np.concatenate([sc, sink_col], axis=-1)
        m_max = sc_ext.max(axis=-1, keepdims=True)
        ex = np.exp(sc_ext - m_max)
        probs = ex[:, :-1] / ex.sum(axis=-1, keepdims=True)
        ref_att[hq] = probs @ v_np[hkv]
    got_np = np.asarray(att_got.astype(mx.float32))[0]
    assert float(np.max(np.abs(got_np - ref_att))) < 5e-3

