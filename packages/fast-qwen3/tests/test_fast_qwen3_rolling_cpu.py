"""Pure-synthetic CPU/GPU parity tests for `fast_qwen3.rolling` continuous-batching slot decoder."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from fast_qwen3 import FastQwen3Engine, FastQwen3Model, FastQwen3Tokenizer, Qwen3Config
from fast_qwen3.rolling import (
    RollingDeltaSlotCache,
    RollingHybridCache,
    RollingKVCache,
    forward_hidden_rolling,
    run_qwen3_decode_rolling,
)


def _make_tiny_byte_tokenizer(tmp_path: Path) -> FastQwen3Tokenizer:
    vocab: dict[str, int] = {}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)

    tok = Tokenizer(BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder(add_prefix_space=False)
    specials = [
        AddedToken("<|endoftext|>", special=True),
        AddedToken("<|im_start|>", special=True),
        AddedToken("<|im_end|>", special=True),
        AddedToken("<think>", special=True),
        AddedToken("</think>", special=True),
    ]
    while len(vocab) + len(specials) < 320:
        specials.append(AddedToken(f"<|extra_{len(specials)}|>", special=True))
    tok.add_special_tokens(specials)

    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    return FastQwen3Tokenizer(tok_path, full_vocab_size=tok.get_vocab_size(with_added_tokens=True))


def _small_dense_cfg(vocab_size: int = 320, *, tie: bool = True) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=vocab_size,
        tie_word_embeddings=tie,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        mtp_num_hidden_layers=0,
    )


def _make_synthetic_qwen3_weights(cfg: Qwen3Config, *, seed: int = 11) -> dict[str, mx.array]:
    mx.random.seed(seed)
    h = cfg.hidden_size
    v = cfg.vocab_size
    inter = cfg.intermediate_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = 2 * k_dim + v_dim

    def r_w(out_d: int, in_d: int) -> mx.array:
        return (mx.random.normal((out_d, in_d)) * 0.04).astype(mx.bfloat16)

    def r_n(dim: int) -> mx.array:
        return (mx.full((dim,), 1.0, dtype=mx.float32) + mx.random.normal((dim,)) * 0.02).astype(mx.bfloat16)

    w: dict[str, mx.array] = {
        "model.embed_tokens.weight": r_w(v, h),
        "model.norm.weight": r_n(h),
    }
    if not cfg.tie_word_embeddings:
        w["lm_head.weight"] = r_w(v, h)

    for i in range(cfg.num_hidden_layers):
        px = f"model.layers.{i}"
        w[f"{px}.input_layernorm.weight"] = r_n(h)
        w[f"{px}.post_attention_layernorm.weight"] = r_n(h)
        if cfg.layer_types[i] == "linear_attention":
            gpx = f"{px}.linear_attn"
            w[f"{gpx}.in_proj_qkv.weight"] = r_w(conv_dim, h)
            w[f"{gpx}.in_proj_z.weight"] = r_w(v_dim, h)
            w[f"{gpx}.in_proj_b.weight"] = r_w(cfg.linear_num_value_heads, h)
            w[f"{gpx}.in_proj_a.weight"] = r_w(cfg.linear_num_value_heads, h)
            w[f"{gpx}.out_proj.weight"] = r_w(h, v_dim)
            w[f"{gpx}.conv1d.weight"] = (
                mx.random.normal((conv_dim, 1, cfg.linear_conv_kernel_dim)) * 0.05
            ).astype(mx.bfloat16)
            w[f"{gpx}.dt_bias"] = mx.ones((cfg.linear_num_value_heads,), dtype=mx.bfloat16) * 0.5
            w[f"{gpx}.A_log"] = mx.zeros((cfg.linear_num_value_heads,), dtype=mx.bfloat16)
            w[f"{gpx}.norm.weight"] = mx.ones((cfg.linear_value_head_dim,), dtype=mx.bfloat16)
        else:
            apx = f"{px}.self_attn"
            w[f"{apx}.q_proj.weight"] = r_w(2 * q_dim, h)
            w[f"{apx}.k_proj.weight"] = r_w(kv_dim, h)
            w[f"{apx}.v_proj.weight"] = r_w(kv_dim, h)
            w[f"{apx}.o_proj.weight"] = r_w(h, q_dim)
            w[f"{apx}.q_norm.weight"] = r_n(cfg.head_dim)
            w[f"{apx}.k_norm.weight"] = r_n(cfg.head_dim)
        mpx = f"{px}.mlp"
        w[f"{mpx}.gate_proj.weight"] = r_w(inter, h)
        w[f"{mpx}.up_proj.weight"] = r_w(inter, h)
        w[f"{mpx}.down_proj.weight"] = r_w(h, inter)

    mx.eval(*w.values())
    return w


def test_rolling_kv_cache_scatter_and_zero_cross_slot_leakage() -> None:
    """Verify `(B_slots, Cap, n_kv, hd)` in-place scatter, single-doc attend, and BF16-masked decode isolation."""
    b_slots, cap, n_q, n_kv, hd = 4, 16, 4, 2, 32
    kv = RollingKVCache(b_slots=b_slots, cap=cap, n_kv=n_kv, head_dim=hd)
    assert kv.k_buf.shape == (b_slots, cap, n_kv, hd)
    assert kv.v_buf.shape == (b_slots, cap, n_kv, hd)
    assert kv.k_buf.dtype == mx.bfloat16

    mx.random.seed(5)
    # Slot 0 has 3 valid tokens; Slot 2 has 5 valid tokens; Slot 1 & 3 are poisoned with huge sentinel values
    kv.k_buf[1, :] = mx.full((cap, n_kv, hd), 1e4, dtype=mx.bfloat16)
    kv.v_buf[1, :] = mx.full((cap, n_kv, hd), 1e4, dtype=mx.bfloat16)
    # Also poison unused tail positions of Slot 0 (pos 3..cap-1) to prove valid_lens masking prevents leakage!
    kv.k_buf[0, 3:] = mx.full((cap - 3, n_kv, hd), -1e4, dtype=mx.bfloat16)
    kv.v_buf[0, 3:] = mx.full((cap - 3, n_kv, hd), -1e4, dtype=mx.bfloat16)

    k0 = mx.random.normal((1, n_kv, 3, hd)).astype(mx.bfloat16)
    v0 = mx.random.normal((1, n_kv, 3, hd)).astype(mx.bfloat16)
    k2 = mx.random.normal((1, n_kv, 5, hd)).astype(mx.bfloat16)
    v2 = mx.random.normal((1, n_kv, 5, hd)).astype(mx.bfloat16)
    kv.scatter_prefill(0, k0, v0, start_pos=0)
    kv.scatter_prefill(2, k2, v2, start_pos=0)

    # Decode step writing pos=3 for slot 0 and pos=5 for slot 2
    k_step = mx.random.normal((2, n_kv, 1, hd)).astype(mx.bfloat16)
    v_step = mx.random.normal((2, n_kv, 1, hd)).astype(mx.bfloat16)
    kv.scatter_decode_step([0, 2], [3, 5], k_step, v_step)

    q_dec = mx.random.normal((2, n_q, 1, hd)).astype(mx.bfloat16)
    scale = hd**-0.5
    out_rolling = kv.attend_rolling_decode(q_dec, [0, 2], [4, 6], scale)

    # Reference single-slot SDPA on exact [0..4) and [0..6) without any neighbor or tail poison
    ref_0 = kv.attend_slot_single(0, q_dec[0:1], 4, scale)
    ref_2 = kv.attend_slot_single(2, q_dec[1:2], 6, scale)
    mx.eval(out_rolling, ref_0, ref_2)

    assert not bool(mx.any(mx.isnan(out_rolling)).item())
    diff0 = float(mx.max(mx.abs(out_rolling[0:1].astype(mx.float32) - ref_0.astype(mx.float32))).item())
    diff2 = float(mx.max(mx.abs(out_rolling[1:2].astype(mx.float32) - ref_2.astype(mx.float32))).item())
    assert diff0 < 2e-2
    assert diff2 < 2e-2
    # Slot 1 poison must be completely untouched
    assert float(mx.min(kv.k_buf[1]).item()) > 9000.0


def test_rolling_delta_slot_cache_int8_and_fp32_isolation() -> None:
    """Verify `RollingDeltaSlotCache` slice/scatter/reset isolation in both `int8_blk32` and `fp32`."""
    b_slots, conv_k, d_conv, hv, dv, dk = 4, 4, 16, 4, 32, 32
    for mode in ("int8_blk32", "fp32"):
        rc = RollingDeltaSlotCache(
            b_slots,
            conv_k,
            d_conv,
            hv,
            dv,
            dk,
            gdn_state_mode=mode,
        )
        assert rc.state_mode == mode
        sub = rc.slice_for_slots([1, 3])
        assert sub.state_mode == mode
        assert sub.conv_state is not None and sub.conv_state.shape == (2, conv_k - 1, d_conv)

        mx.random.seed(9)
        sub.conv_state = mx.random.normal((2, conv_k - 1, d_conv)).astype(mx.bfloat16)
        sub.rec_state = mx.random.normal((2, hv, dv, dk)).astype(mx.float32) * 0.5
        rc.scatter_from_subcache([1, 3], sub)

        # Slots 0 and 2 must remain strictly zero
        sub_unused = rc.slice_for_slots([0, 2])
        mx.eval(sub_unused.conv_state, sub_unused.rec_state)
        assert float(mx.max(mx.abs(sub_unused.conv_state.astype(mx.float32))).item()) == 0.0
        assert float(mx.max(mx.abs(sub_unused.rec_state.astype(mx.float32))).item()) == 0.0

        # Active slots 1 and 3 retain exact scattered values
        sub_check = rc.slice_for_slots([1, 3])
        mx.eval(sub_check.conv_state, sub_check.rec_state, sub.conv_state, sub.rec_state)
        assert float(mx.max(mx.abs(sub_check.conv_state.astype(mx.float32) - sub.conv_state.astype(mx.float32))).item()) == 0.0
        tol = 2e-2 if mode == "int8_blk32" else 0.0
        assert float(mx.max(mx.abs(sub_check.rec_state.astype(mx.float32) - sub.rec_state.astype(mx.float32))).item()) <= tol

        # Resetting slot 1 leaves slot 3 intact
        rc.reset_slots([1])
        s1 = rc.slice_for_slots([1])
        s3 = rc.slice_for_slots([3])
        mx.eval(s1.conv_state, s1.rec_state, s3.conv_state, s3.rec_state)
        assert float(mx.max(mx.abs(s1.rec_state.astype(mx.float32))).item()) == 0.0
        assert float(mx.max(mx.abs(s3.rec_state.astype(mx.float32))).item()) > 0.0


def test_rolling_decode_12_mixed_prompts_exact_b1_parity(tmp_path: Path) -> None:
    """Verify `run_qwen3_decode_rolling` across 12 mixed prompts (`B_slots=4`, `int8_blk32`, `upfront_ff`, mixed schemas) matches `B=1` token-for-token."""
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    w = _make_synthetic_qwen3_weights(cfg, seed=11)
    model = FastQwen3Model(cfg, w, enable_fusion=True)
    eng = FastQwen3Engine(
        model=model,
        tokenizer=tok,
        warmup=False,
        prefix_cache_mb=0,
        max_prefix_cache_entries=0,
    )
    try:
        # Schema A has a long deterministic upfront_ff prefix; Schema B has a different object schema; None is free-text
        schema_a = {
            "type": "object",
            "properties": {
                "record_classification_kind": {"type": "string", "enum": ["bill", "notice"]},
                "verified_status": {"type": "boolean"},
            },
            "required": ["record_classification_kind", "verified_status"],
            "additionalProperties": False,
        }
        schema_b = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "enum": [1, 2, 3]},
                "note": {"type": "string", "enum": ["ok", "skip"]},
            },
            "required": ["count", "note"],
            "additionalProperties": False,
        }
        schema_const = {
            "type": "object",
            "properties": {"fixed": {"type": "boolean", "const": True}},
            "required": ["fixed"],
            "additionalProperties": False,
        }

        prompts = [
            "Short A1",
            "Medium prompt B2 with extra tokens for uneven length distribution.",
            "Long C3 " + ("alpha beta gamma delta " * 14),
            "Tiny D4",
            "Free text E5 summary request.",
            "Schema const F6",
            "Medium G7 record header: invoice 777 regional warehouse audit.",
            "Long H8 " + ("statutory compliance filing payload " * 11),
            "Short I9",
            "Medium J10 classification item with trailing details.",
            "Free text K11 another unconstrained sequence.",
            "Long L12 " + ("procurement reconciliation item " * 13),
        ]
        per_schemas = [
            schema_a,
            schema_b,
            schema_a,
            schema_b,
            None,
            schema_const,
            schema_a,
            schema_b,
            schema_a,
            schema_b,
            None,
            schema_a,
        ]
        assert len(prompts) == 12

        # Sequential B=1 ground truth via FastQwen3Engine.generate(...)
        b1_items = [
            eng.generate(
                p,
                json_schema=sch,
                max_tokens=36,
                temperature=0.0,
                speculative=False,
            )
            for p, sch in zip(prompts, per_schemas)
        ]

        # Rolling continuous-batching decode (`B_slots=4`, multiple refills, `int8_blk32` GDN, `upfront_ff` folding)
        encoded = [eng.tokenizer.encode_chat(p) for p in prompts]
        grammars = [eng._get_grammar_helper(sch, None) if sch is not None else None for sch in per_schemas]
        schema_objs = [g.schema_obj if g is not None else None for g in grammars]
        assert grammars[0] is not None and len(grammars[0].upfront_ff) >= 4

        rolling_items, roll_stats = run_qwen3_decode_rolling(
            model,
            eng.tokenizer,
            encoded,
            grammars,
            schema_objs,
            eng._finalize_item,
            eng._select_tokens_from_hidden,
            B_slots=4,
            Cap=600,
            max_tokens=36,
            temperature=0.0,
            gdn_state_mode="int8_blk32",
            prefill_chunk_size=64,
        )

        assert len(rolling_items) == 12
        assert roll_stats["b_slots"] == 4
        assert roll_stats["gdn_state_mode"] == "int8_blk32"
        assert roll_stats["refills"] >= 2
        assert roll_stats["ff_tokens"] > 0

        for idx, (ref, got) in enumerate(zip(b1_items, rolling_items)):
            assert got.token_ids == ref.token_ids, (
                f"Prompt {idx} token parity mismatch:\n  got={got.token_ids}\n  ref={ref.token_ids}"
            )
            assert got.text == ref.text
            assert got.finish_reason == ref.finish_reason
            assert got.valid_schema == ref.valid_schema
            assert got.parsed == ref.parsed

        # Also verify fp32 GDN mode + auto Cap=None + single-step forward_hidden_rolling parity
        rolling_fp32, stats_fp32 = run_qwen3_decode_rolling(
            model,
            eng.tokenizer,
            encoded[:5],
            grammars[:5],
            schema_objs[:5],
            eng._finalize_item,
            eng._select_tokens_from_hidden,
            B_slots=2,
            Cap=None,
            max_tokens=36,
            temperature=0.0,
            gdn_state_mode="fp32",
            prefill_chunk_size=64,
        )
        assert stats_fp32["cap"] >= max(len(e) for e in encoded[:5]) + 36
        for idx in range(5):
            assert rolling_fp32[idx].token_ids == b1_items[idx].token_ids

        # Verify prefix_restore_fn + RollingHybridCache.restore_slot_from_hybrid_cache parity
        from fast_qwen3 import split_lcp_ids

        shared_pfx = "Shared prefix context block " * 8
        pfx_prompts = [shared_pfx + "Alpha", shared_pfx + "Beta extra words"]
        pfx_enc = [eng.tokenizer.encode_chat(p) for p in pfx_prompts]
        pfx_ids, _ = split_lcp_ids(pfx_enc, min_lcp=32)
        assert len(pfx_ids) >= 32
        c_pfx = model.make_cache(kv_step=256)
        _ = model.forward_hidden(mx.array([pfx_ids], dtype=mx.int32), c_pfx)
        mx.eval(
            *[x for c in c_pfx if hasattr(c, "conv_state") for x in (c.conv_state, c.rec_state)]
            + [x for c in c_pfx if hasattr(c, "keys") for x in (c.keys, c.values)]
        )
        pfx_refs = [
            eng.generate(p, json_schema=schema_a, max_tokens=36, temperature=0.0, speculative=False)
            for p in pfx_prompts
        ]
        pfx_grams = [eng._get_grammar_helper(schema_a, None)] * 2
        pfx_schs = [pfx_grams[0].schema_obj] * 2

        def _restore_pfx(full_ids, rcache: RollingHybridCache, slot_id: int) -> int:
            if len(full_ids) > len(pfx_ids) and list(full_ids[: len(pfx_ids)]) == pfx_ids:
                return rcache.restore_slot_from_hybrid_cache(slot_id, c_pfx)
            return 0

        pfx_rolling, _ = run_qwen3_decode_rolling(
            model,
            eng.tokenizer,
            pfx_enc,
            pfx_grams,
            pfx_schs,
            eng._finalize_item,
            eng._select_tokens_from_hidden,
            B_slots=2,
            Cap=400,
            max_tokens=36,
            temperature=0.0,
            gdn_state_mode="fp32",
            prefix_restore_fn=_restore_pfx,
        )
        for ref, got in zip(pfx_refs, pfx_rolling):
            assert got.token_ids == ref.token_ids
    finally:
        eng.close()


def test_rolling_loop_abort_and_direct_forward_hidden_rolling(tmp_path: Path) -> None:
    """Verify `detect_catastrophic_token_loop` aborts runaway repetition and `forward_hidden_rolling` matches step-by-step `B=1`."""
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    w = _make_synthetic_qwen3_weights(cfg, seed=11)
    model = FastQwen3Model(cfg, w, enable_fusion=True)
    eng = FastQwen3Engine(model=model, tokenizer=tok, warmup=False, prefix_cache_mb=0, max_prefix_cache_entries=0)
    try:
        # 1. Direct forward_hidden_rolling vs B=1 forward_hidden
        p = [10, 20, 30, 40, 50]
        c_b1 = model.make_cache(gdn_state_mode="fp32")
        h_pf_b1 = model.forward_hidden(mx.array([p], dtype=mx.int32), c_b1)
        h_s1_b1 = model.forward_hidden(mx.array([[61]], dtype=mx.int32), c_b1)

        rc = RollingHybridCache(model, b_slots=2, cap=32, gdn_state_mode="fp32")
        h_pf_rc = rc.prefill_slots([1], [p])
        h_s1_rc = forward_hidden_rolling(model, rc, mx.array([[61]], dtype=mx.int32), [1], [5], [6])
        mx.eval(h_pf_b1, h_s1_b1, h_pf_rc, h_s1_rc)
        assert float(mx.max(mx.abs(h_pf_rc[0].astype(mx.float32) - h_pf_b1[0, -1].astype(mx.float32))).item()) < 2e-2
        assert float(mx.max(mx.abs(h_s1_rc[0, 0].astype(mx.float32) - h_s1_b1[0, 0].astype(mx.float32))).item()) < 2e-2

        # 2. Catastrophic loop abort (constant token 77 forced by select_tokens_fn -> aborts before max_tokens=64)
        def _loop_select(_h, need_rows, _row_count, **_kw):
            return {r: 77 for r in need_rows}, {r: None for r in need_rows}, 0

        items, stats = run_qwen3_decode_rolling(
            model,
            eng.tokenizer,
            [[10, 11, 12], [13, 14, 15]],
            [None, None],
            [None, None],
            eng._finalize_item,
            _loop_select,
            B_slots=2,
            Cap=128,
            max_tokens=64,
        )
        assert stats["loop_aborts"] == 2
        assert all(16 <= len(it.token_ids) < 64 for it in items)

        # 3. Engine.generate_batch(..., rolling=True) integration parity vs sequential B=1
        batch_prompts = [
            f"Record {i}: tax year 2024 status active" + (" extra detail" * (i % 4))
            for i in range(6)
        ]
        batch_schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer", "enum": [1, 2]}},
            "required": ["ok", "n"],
            "additionalProperties": False,
        }
        seq_ref = [
            eng.generate(p, system_prompt="Extract status.", json_schema=batch_schema, max_tokens=24, temperature=0.0, speculative=False)
            for p in batch_prompts
        ]
        res_roll = eng.generate_batch(
            batch_prompts,
            system_prompt="Extract status.",
            json_schema=batch_schema,
            max_tokens=24,
            temperature=0.0,
            batch_size=3,
            rolling=True,
        )
        assert res_roll.stats.get("rolling") is True
        assert [it.token_ids for it in res_roll.items] == [it.token_ids for it in seq_ref]
    finally:
        eng.close()
