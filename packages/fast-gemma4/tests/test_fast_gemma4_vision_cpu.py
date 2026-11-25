"""CPU/MLX unit tests for lazy optional vision support in `fast-gemma4`."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fast_gemma4.cli import build_parser
from fast_gemma4.engine import FastGemma4Engine
from fast_gemma4.model import FastGemma4Model
from fast_gemma4.tokenizer import FastGemma4Tokenizer


def test_import_fast_gemma4_does_not_load_vision_or_pil() -> None:
    """Importing `fast_gemma4` must never eagerly import `fast_shared.gemma4_vision` or `PIL`."""
    code = (
        "import sys, fast_gemma4; "
        "assert 'fast_shared.gemma4_vision' not in sys.modules; "
        "assert 'PIL' not in sys.modules; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "OK" in proc.stdout


def _find_any_gemma4_tokenizer() -> Path | None:
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--gemma-4-E2B-it-MLX-4bit/snapshots/*/tokenizer.json"),
        os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-*/snapshots/*/tokenizer.json"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return Path(matches[0])
    return None


def test_expand_multimodal_prompt_ids_variants() -> None:
    tok_path = _find_any_gemma4_tokenizer()
    if tok_path is None:
        pytest.skip("No local Gemma-4 tokenizer.json cached")
    eng = object.__new__(FastGemma4Engine)
    eng.tokenizer = FastGemma4Tokenizer(tok_path)

    # 1. Explicit structured message with image part
    ids1, spans1 = eng._expand_multimodal_prompt_ids(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}],
        [6],
        system_prompt=None,
        use_chat_template=True,
        enable_thinking=False,
        boi_token_id=255999,
        image_token_id=258880,
        eoi_token_id=258882,
    )
    assert len(spans1) == 1
    s0, s1 = spans1[0]
    assert s1 - s0 == 6
    assert ids1[s0 - 1] == 255999
    assert ids1[s0:s1] == [258880] * 6
    assert ids1[s1] == 258882

    # 2. Implicit plain string prompt inserts images right after `<|turn>user\n`
    ids2, spans2 = eng._expand_multimodal_prompt_ids(
        "Read both pages.",
        [4, 5],
        system_prompt=None,
        use_chat_template=True,
        enable_thinking=False,
        boi_token_id=255999,
        image_token_id=258880,
        eoi_token_id=258882,
    )
    assert len(spans2) == 2
    assert ids2[spans2[0][0] : spans2[0][1]] == [258880] * 4
    assert ids2[spans2[1][0] : spans2[1][1]] == [258880] * 5
    user_hdr = eng.tokenizer.encode("<|turn>user\n", add_special_tokens=False)
    hdr_end = spans2[0][0] - 1
    assert ids2[hdr_end - len(user_hdr) : hdr_end] == user_hdr

    # 3. Mismatched explicit `<|image|>` count raises ValueError
    with pytest.raises(ValueError, match="Expected 2"):
        eng._expand_multimodal_prompt_ids(
            "Only one <|image|> marker",
            [3, 3],
            system_prompt=None,
            use_chat_template=True,
            enable_thinking=False,
            boi_token_id=255999,
            image_token_id=258880,
            eoi_token_id=258882,
        )


def test_compute_pli_from_embeds_and_vision_mask() -> None:
    m = object.__new__(FastGemma4Model)
    m.act_dtype = mx.bfloat16
    m.eps = 1e-6
    m.window_size = 4
    m.ple_dim = 4
    m.num_layers = 2
    m._n_early_pli_layers = 1
    m._ple_mmap_i8 = None
    m._ple_mmap_scales = None
    m._etpl_14_groups = 0
    m._etpl_14_wcols = 0
    m.embed_tokens_per_layer_scale = mx.array(2.0, dtype=mx.bfloat16)
    m.per_layer_projection_scale = mx.array(0.25, dtype=mx.bfloat16)
    m.per_layer_projection_norm_w = mx.ones((4,), dtype=mx.bfloat16)
    m.per_layer_input_scale = mx.array(0.707, dtype=mx.bfloat16)
    m.embed_tokens_per_layer = nn.Embedding(16, 8)
    m.per_layer_model_projection = nn.Linear(8, 8, bias=False)
    m.pli_proj_14 = nn.Linear(8, 4, bias=False)

    emb = mx.ones((1, 5, 8), dtype=mx.bfloat16)
    ple_toks = mx.array([[2, 0, 0, 3, 1]], dtype=mx.int32)
    pli_e14 = m._compute_pli_from_embeds(emb, ple_toks, early14=True)
    pli_full = m._compute_pli_from_embeds(emb, ple_toks, early14=False)
    assert pli_e14 is not None and pli_e14.shape == (1, 5, 1, 4)
    assert pli_full is not None and pli_full.shape == (1, 5, 2, 4)

    # Bidirectional mask within same image block (tokens 1..3) and causal across text/different images
    vis_group_ids = np.array([-1, 0, 0, 0, -1, 1, 1, -1], dtype=np.int32)
    mask_full = m._build_vision_sdpa_mask(0, 8, 0, 8, vis_group_ids, is_full=True)
    mx.eval(mask_full)
    assert mask_full.dtype == mx.bfloat16
    mask_np = np.asarray(mask_full.astype(mx.float32))
    # Token 1 inside image 0 sees future token 3 (same image), NOT text token 0's reverse? No, 1->0 allowed causal, 0->1 forbidden
    assert mask_np[0, 1] == float("-inf")
    assert mask_np[1, 3] == 0.0
    # Image 0 token 1 does NOT see image 1 token 5
    assert mask_np[1, 5] == float("-inf")
    # Text token 4 sees image 0 (0.0) but NOT future image 1 (-inf)
    assert mask_np[4, 2] == 0.0
    assert mask_np[4, 5] == float("-inf")

    # Sliding window enforces distance < 4 on cross-block pairs
    mask_swa = m._build_vision_sdpa_mask(0, 8, 0, 8, vis_group_ids, is_full=False)
    mx.eval(mask_swa)
    swa_np = np.asarray(mask_swa.astype(mx.float32))
    assert swa_np[7, 1] == float("-inf")
    assert swa_np[7, 5] == 0.0


def test_cli_generate_image_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--image",
            "/tmp/p1.png",
            "--image",
            "/tmp/p2.png",
            "--max-soft-tokens",
            "1120",
            "--max-tokens",
            "128",
            "Read pages.",
        ]
    )
    assert args.cmd == "generate"
    assert args.image == ["/tmp/p1.png", "/tmp/p2.png"]
    assert args.max_soft_tokens == 1120
    assert args.prompt == "Read pages."


def test_multimodal_generate_mtp_dispatch_and_fallbacks(monkeypatch) -> None:
    """Verify greedy multimodal generate dispatches to `run_decode_mtp` when available and falls back cleanly."""
    import threading
    from types import SimpleNamespace
    import fast_gemma4.engine as eng_mod

    eng = object.__new__(FastGemma4Engine)
    eng._gpu_lock = threading.Lock()
    eng.mtp_draft_k = 3
    eng._engine_apple10_nax = lambda: True  # type: ignore[method-assign]
    eng._prepare_multimodal_prefill = lambda prompt, images, **kw: (  # type: ignore[method-assign]
        [2, 10, 11],
        mx.zeros((1, 3, 8), dtype=mx.bfloat16),
        0.01,
        2,
    )
    fake_grammar = SimpleNamespace(
        upfront_ff=[101, 102],
        validate_token_sequence=lambda ids: list(ids) == [101, 102, 201, 202],
    )
    eng.get_grammar = lambda schema, **kw: fake_grammar  # type: ignore[method-assign]

    provide_last_x = True

    class _FakeModel:
        window_size = 512
        embed_scale = mx.array(1.0, dtype=mx.bfloat16)

        @staticmethod
        def embed_tokens(arr: mx.array) -> mx.array:
            return mx.zeros((int(arr.shape[0]), int(arr.shape[1]), 8), dtype=mx.bfloat16)

        @staticmethod
        def prefill_prefix(_xs: Any) -> dict[str, Any]:
            return {}

        @staticmethod
        def make_batch_cache(_pkv: Any, **_kw: Any) -> dict[str, Any]:
            return {"_last_x_in": mx.zeros((1, 1, 8), dtype=mx.bfloat16)} if provide_last_x else {}

        @staticmethod
        def prefill_multimodal_b1(pids: Any, _embeds: Any, _cache: Any, apply_softcap: bool = True):
            assert apply_softcap is True
            return mx.zeros((1, 1, 16)), mx.array([len(pids)], dtype=mx.int32)

    eng.model = _FakeModel()  # type: ignore[assignment]
    eng.tokenizer = SimpleNamespace(
        stop_token_ids=frozenset({1}),
        decode=lambda ids, skip_special_tokens=True: "{" + ",".join(str(x) for x in ids) + "}",
    )  # type: ignore[assignment]

    fake_drafter = object()
    eng._maybe_ensure_mtp_drafter = (  # type: ignore[method-assign]
        lambda *, use_ane, batch_w, temperature, speculative=None, has_grammar=False: (
            fake_drafter if (not use_ane and batch_w == 1 and temperature <= 0.0 and speculative is not False) else None
        )
    )

    calls: list[str] = []

    def _fake_mtp(model, drafter, stop_ids, gram, lg0, ob0, cache, B, **kw):
        calls.append("mtp")
        assert drafter is fake_drafter and B == 1
        assert kw["doc_prompts"] == [[2, 10, 11, 101, 102]]
        return {
            "decode_sec": 0.02,
            "gen_toks": [[201, 202]],
            "used_speculative": True,
            "speculative_backend": "mtp",
            "mtp_accept_rate": 0.75,
        }

    def _fake_batch(model, stop_ids, gram, lg0, ob0, cache, B, **kw):
        calls.append("batch")
        assert B == 1
        return {
            "decode_sec": 0.04,
            "gen_toks": [[201, 202]],
            "used_speculative": False,
            "speculative_backend": None,
        }

    monkeypatch.setattr(eng_mod, "run_decode_mtp", _fake_mtp)
    monkeypatch.setattr(eng_mod, "run_decode_batch", _fake_batch)

    # 1. Default greedy call routes to MTP and prepends upfront_ff cleanly
    it_mtp = eng.generate("Extract", images=[np.zeros((48, 48, 3), dtype=np.uint8)], json_schema={"type": "object"})
    assert calls == ["mtp"]
    assert it_mtp.token_ids == [101, 102, 201, 202]
    assert it_mtp.text == "{101,102,201,202}"
    assert it_mtp.valid_schema is True
    assert eng.last_multimodal_stats["used_speculative"] is True
    assert eng.last_multimodal_stats["speculative_backend"] == "mtp"
    assert eng.last_multimodal_stats["mtp_accept_rate"] == 0.75

    # 2. speculative=False falls back cleanly to run_decode_batch
    calls.clear()
    _ = eng.generate("Extract", images=[np.zeros((48, 48, 3), dtype=np.uint8)], speculative=False)
    assert calls == ["batch"]
    assert eng.last_multimodal_stats["used_speculative"] is False
    assert eng.last_multimodal_stats["speculative_backend"] is None

    # 3. Non-greedy temperature > 0.0 falls back cleanly to run_decode_batch
    calls.clear()
    _ = eng.generate("Extract", images=[np.zeros((48, 48, 3), dtype=np.uint8)], temperature=0.5)
    assert calls == ["batch"]

    # 4. Missing cache["_last_x_in"] falls back safely to run_decode_batch
    calls.clear()
    provide_last_x = False
    _ = eng.generate("Extract", images=[np.zeros((48, 48, 3), dtype=np.uint8)])
    assert calls == ["batch"]


def test_multimodal_and_batch_preserve_caller_peak_memory_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `FastGemma4Engine._generate_multimodal` and `generate_batch` never erase caller peak memory."""
    import threading
    from types import SimpleNamespace
    import fast_gemma4.engine as eng_mod

    reset_calls = 0

    def _spy_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(eng_mod.mx, "reset_peak_memory", _spy_reset)

    eng = object.__new__(FastGemma4Engine)
    eng._gpu_lock = threading.Lock()
    eng.default_mode = "gpu"
    eng.profile = "auto"
    eng.ane_available = False
    eng.default_max_b_slots = 8
    eng.mtp_drafter = None
    eng._engine_apple10_nax = lambda: False  # type: ignore[method-assign]
    eng._has_mtp_drafter_candidate = lambda: False  # type: ignore[method-assign]
    eng._maybe_ensure_mtp_drafter = lambda **_kw: None  # type: ignore[method-assign]
    eng._maybe_get_ane_ring = lambda *_a, **_kw: None  # type: ignore[method-assign]
    eng._effective_ane_ring_variant = lambda: "fp16"  # type: ignore[method-assign]
    eng.resolve_call_kv_i8 = lambda **_kw: False  # type: ignore[method-assign]
    eng.get_prefix_kv = lambda _sys_ids: {"l_sys": 0}  # type: ignore[method-assign]
    eng.encode_prompts = lambda prompts, **_kw: [[2, 10] for _ in prompts]  # type: ignore[method-assign]
    eng._prepare_multimodal_prefill = lambda *_a, **_kw: (  # type: ignore[method-assign]
        [2, 10, 11],
        mx.zeros((1, 3, 8), dtype=mx.bfloat16),
        0.0,
        1,
    )

    class _TinyModel:
        window_size = 512
        embed_scale = mx.array(1.0, dtype=mx.bfloat16)

        @staticmethod
        def embed_tokens(arr: mx.array) -> mx.array:
            return mx.zeros((int(arr.shape[0]), int(arr.shape[1]), 8), dtype=mx.bfloat16)

        @staticmethod
        def prefill_prefix(_xs: Any) -> dict[str, Any]:
            return {}

        @staticmethod
        def make_batch_cache(_pkv: Any, **_kw: Any) -> dict[str, Any]:
            return {}

        @staticmethod
        def prefill_multimodal_b1(_pids: Any, _embeds: Any, _cache: Any, apply_softcap: bool = True):
            return mx.zeros((1, 1, 8)), mx.array([3], dtype=mx.int32)

    eng.model = _TinyModel()  # type: ignore[assignment]
    eng.tokenizer = SimpleNamespace(
        bos_token_id=2,
        stop_token_ids=frozenset({1}),
        decode=lambda ids, skip_special_tokens=True: "ok",
    )  # type: ignore[assignment]

    monkeypatch.setattr(
        eng_mod,
        "run_decode_batch",
        lambda *_a, **_kw: {"decode_sec": 0.01, "gen_toks": [[5]], "used_speculative": False},
    )
    monkeypatch.setattr(
        eng_mod,
        "run_decode_rolling",
        lambda *_a, **_kw: {
            "gen_toks": [[5]],
            "prefill_sec": 0.0,
            "decode_sec": 0.01,
            "tot_out_tok": 1,
            "ff_tokens": 0,
            "llm_steps": 1,
            "ff_early14_steps": 0,
            "mask_fallbacks": 0,
            "used_speculative": False,
        },
    )

    _ = eng.generate("hi", images=[np.zeros((8, 8, 3), dtype=np.uint8)], speculative=False)
    _ = eng.generate_batch(["hi"], mode="gpu", speculative=False)
    assert reset_calls == 0


def test_multimodal_online_catastrophic_loop_abort_cpu() -> None:
    """Verify `_generate_multimodal` aborts runaway `p=1` & `p>8` loops online instead of spinning to `max_tokens=4096`."""
    import threading
    from types import SimpleNamespace

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        vocab_n = 64
        eng = object.__new__(FastGemma4Engine)
        eng._gpu_lock = threading.Lock()
        eng.mtp_draft_k = 3
        eng._engine_apple10_nax = lambda: False  # type: ignore[method-assign]
        eng._prepare_multimodal_prefill = lambda *_a, **_kw: (  # type: ignore[method-assign]
            [2, 3, 4],
            mx.zeros((1, 3, 8), dtype=mx.bfloat16),
            0.0,
            1,
        )

        # Tokens: 1=EOS, 10..21=p12 cycle, 30=<td>, 31=</td>, 32=<tr>, 33=</tr>, 34=0.00, 40..59=row labels
        tok_map = {30: "<td>", 31: "</td>", 32: "<tr>", 33: "</tr>", 34: "0.00"}
        eng.tokenizer = SimpleNamespace(
            stop_token_ids=frozenset({1}),
            decode=lambda ids, skip_special_tokens=True: "".join(tok_map.get(int(t), f"w{int(t)}") for t in ids),
        )  # type: ignore[assignment]

        dense_table_seq = [
            tok
            for row_i in range(16)
            for tok in (32, 30, 40 + (row_i % 20), 31, 30, 31, 30, 34, 31, 30, 31, 30, 34, 31, 33)
        ]  # 240 tokens of real dense HTML table markup followed by EOS
        p12_cycle = list(range(10, 22))
        mode_box = {"mode": "p12", "step": 0}

        def _next_logits_np() -> np.ndarray:
            arr = np.full((1, 1, vocab_n), -20.0, dtype=np.float32)
            idx = mode_box["step"]
            mode_box["step"] += 1
            if mode_box["mode"] == "p12":
                tid = p12_cycle[idx % len(p12_cycle)]
            elif mode_box["mode"] == "p1":
                tid = 13
            else:
                tid = dense_table_seq[idx] if idx < len(dense_table_seq) else 1
            # Huge margin ensures penalty alone does not break loop, exercising online abort!
            arr[0, 0, tid] = 40.0
            return arr

        class _LoopModel:
            window_size = 512
            swa_layers = [0]
            full_layers = [1]
            new_to_old = np.arange(vocab_n, dtype=np.int32)
            old_to_new = np.arange(vocab_n, dtype=np.int32)
            embed_scale = mx.array(1.0, dtype=mx.bfloat16)

            @staticmethod
            def embed_tokens(arr: mx.array) -> mx.array:
                return mx.zeros((int(arr.shape[0]), int(arr.shape[1]), 8), dtype=mx.bfloat16)

            @staticmethod
            def prefill_prefix(_xs: Any) -> dict[str, Any]:
                return {}

            @staticmethod
            def make_batch_cache(_pkv: Any, **_kw: Any) -> dict[str, Any]:
                return {"swa_cap": 512, "l_sys": 0, "_last_x_in": mx.zeros((1, 1, 8), dtype=mx.bfloat16)}

            @staticmethod
            def compact_cache(cache: dict[str, Any], _keep: Any) -> dict[str, Any]:
                return cache

            @staticmethod
            def prefill_multimodal_b1(_pids: Any, _embeds: Any, cache: dict[str, Any], apply_softcap: bool = True):
                return mx.array(_next_logits_np()), mx.array([3], dtype=mx.int32)

            @staticmethod
            def forward_step_ragged(toks: mx.array, cache: dict[str, Any], off_b: mx.array, apply_softcap: bool = True, **_kw: Any):
                _ = (cache, off_b, apply_softcap)
                return mx.array(_next_logits_np())

            @staticmethod
            def forward_verify_mtp_slot(verify_toks: mx.array, cache: dict[str, Any], off_cur: mx.array, slot_idx: Any = None, return_logits: bool = False):
                _ = (cache, off_cur, slot_idx, return_logits)
                b_cur, l_q = int(verify_toks.shape[0]), int(verify_toks.shape[1])
                x_step = mx.zeros((b_cur, l_q, 8), dtype=mx.bfloat16)
                cols = []
                for _j in range(l_q):
                    cols.append(int(np.argmax(_next_logits_np()[0, 0])))
                am_mx = mx.array([cols], dtype=mx.int32)
                return x_step, am_mx

            @staticmethod
            def _apply_lm_head(_x: Any, apply_softcap: bool = True) -> mx.array:
                arr = np.full((1, 1, vocab_n), -20.0, dtype=np.float32)
                arr[0, 0, 13] = 40.0
                return mx.array(arr)

        class _LoopDrafter:
            @staticmethod
            def draft_k(anchor_mx: mx.array, _seed: Any, _cache: Any, _off: Any, cur_k: int, slot_idx: Any = None):
                return mx.full((int(anchor_mx.shape[0]), int(cur_k)), 13, dtype=mx.int32)

        eng.model = _LoopModel()  # type: ignore[assignment]

        # 1. run_decode_batch path: p=12 cycle aborts early well below max_tokens=4096
        eng._maybe_ensure_mtp_drafter = lambda **_kw: None  # type: ignore[method-assign]
        mode_box["mode"] = "p12"
        mode_box["step"] = 0
        it_p12 = eng.generate("OCR", images=[np.zeros((16, 16, 3), dtype=np.uint8)], max_tokens=4096, speculative=False)
        assert it_p12.finish_reason == "length"
        assert eng.last_multimodal_stats["loop_aborted"] is True
        assert 144 <= it_p12.completion_tokens < 256

        # 2. run_decode_batch path: normal 240-token dense <td>/<tr> table is NOT falsely aborted
        mode_box["mode"] = "table"
        mode_box["step"] = 0
        it_tbl = eng.generate("OCR", images=[np.zeros((16, 16, 3), dtype=np.uint8)], max_tokens=4096, speculative=False)
        assert it_tbl.finish_reason == "stop"
        assert eng.last_multimodal_stats["loop_aborted"] is False
        assert it_tbl.completion_tokens == len(dense_table_seq)

        # 3. run_decode_mtp path: p=1 runaway cycle aborts early well below max_tokens=4096
        eng._maybe_ensure_mtp_drafter = lambda **_kw: _LoopDrafter()  # type: ignore[method-assign]
        mode_box["mode"] = "p1"
        mode_box["step"] = 0
        it_mtp_loop = eng.generate("OCR", images=[np.zeros((16, 16, 3), dtype=np.uint8)], max_tokens=4096)
        assert it_mtp_loop.finish_reason == "length"
        assert eng.last_multimodal_stats["loop_aborted"] is True
        assert it_mtp_loop.completion_tokens < 64
    finally:
        mx.set_default_device(prev_dev)


def test_generate_multimodal_batch_rolling_and_delegation_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Track A H2 `generate_multimodal_batch` (`n=1` fallback, `n>=2` rolling, `generate_batch` delegation)."""
    import threading
    from types import SimpleNamespace
    import fast_gemma4.engine as eng_mod
    from fast_gemma4.engine import GenerationItem

    eng = object.__new__(FastGemma4Engine)
    eng._gpu_lock = threading.Lock()
    eng._is_low_ram = True
    eng.default_max_b_slots = 64

    b1_calls: list[Any] = []
    eng._generate_multimodal = lambda p, imgs, **_kw: (  # type: ignore[method-assign]
        b1_calls.append((p, len(imgs)))
        or GenerationItem("{\"ok\":true}", {"ok": True}, [5], 10, 1, True, True, "stop")
    )
    eng.last_multimodal_stats = {"vision_sec": 0.01, "prefill_sec": 0.02, "decode_sec": 0.03}

    r1 = eng.generate_batch(
        ["doc0"],
        images_list=[[np.zeros((8, 8, 3), dtype=np.uint8)]],
        max_tokens=16,
    )
    assert len(b1_calls) == 1
    assert r1.stats["multimodal_rolling"] is False
    assert r1.items[0].parsed == {"ok": True}

    # Multi-doc rolling path (n=2): stub vision, tokenizer, prefix KV, embeds, and run_decode_rolling.
    vis_batches: list[int] = []

    class _FakeVis:
        cfg = SimpleNamespace(
            boi_token_id=200,
            image_token_id=201,
            eoi_token_id=202,
            default_output_length=4,
            patch_size=4,
            pooling_kernel_size=1,
        )

        def encode_images(self, vinputs, batch_size=8):
            vis_batches.append(len(vinputs))
            return [mx.zeros((vi.num_soft_tokens, 8), dtype=mx.bfloat16) for vi in vinputs]

    eng._ensure_vision_model = lambda: _FakeVis()  # type: ignore[method-assign]
    eng.tokenizer = SimpleNamespace(
        stop_token_ids=frozenset({1}),
        decode=lambda ids, skip_special_tokens=True: '{"ok":true}',
    )  # type: ignore[assignment]
    eng._expand_multimodal_prompt_ids = lambda p, cnts, **_kw: ([10, 11, 200, 201, 201, 202, 12], [(3, 5)])  # type: ignore[method-assign]
    eng.get_prefix_kv = lambda sys_ids: {"l_sys": len(sys_ids), "k_sys": [], "v_sys": []}  # type: ignore[method-assign]
    eng.model = SimpleNamespace(
        hidden_size=8,
        embed_scale=1.0,
        embed_tokens=lambda ids: mx.zeros((int(ids.shape[0]), int(ids.shape[1]), 8), dtype=mx.bfloat16),
        prefill_slots_lazy_lg=object(),
        forward_embeds_into_slots=lambda *_a, **_kw: None,
        forward_decode_step_slot=lambda toks, cache, off_b, slot_mx, **_kw: (
            mx.zeros((int(toks.shape[0]), 1, 8), dtype=mx.bfloat16),
            mx.zeros((int(toks.shape[0]),), dtype=mx.int32),
        ),
    )  # type: ignore[assignment]

    seen_roll_kw: dict[str, Any] = {}

    def _fake_roll(_m, _stops, _g, _pkv, tagged_sufs, **kw):
        seen_roll_kw.update(kw)
        assert callable(eng.model.prefill_slots_lazy_lg)
        return {
            "gen_toks": [[5, 6] for _ in tagged_sufs],
            "prefill_sec": 0.01,
            "decode_sec": 0.02,
            "tot_out_tok": 2 * len(tagged_sufs),
            "ff_tokens": 0,
        }

    monkeypatch.setattr(eng_mod, "run_decode_rolling", _fake_roll)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    r2 = eng.generate_batch(["doc0", "doc1"], images_list=[[img], [img]], max_tokens=32, batch_size=8)
    assert vis_batches == [2]
    assert seen_roll_kw["b_slots"] == 2
    assert seen_roll_kw["kv_i8"] is False  # short ctx <= 1024 defaults kv_i8=False
    assert seen_roll_kw["swa_ring_cap"] >= 1024
    assert r2.stats["multimodal_rolling"] is True
    assert r2.stats["valid_json_count"] == 2
    assert eng.model.prefill_slots_lazy_lg is not None

