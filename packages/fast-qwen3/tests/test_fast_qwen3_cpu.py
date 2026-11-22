"""Pure-synthetic / zero-checkpoint tests for `fast-qwen3` kernels, fusion, MoE, LoRA, and GDN rollback."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from fast_qwen3 import (
    FastQwen3Engine,
    FastQwen3Model,
    FastQwen3Server,
    FastQwen3Tokenizer,
    Qwen3Config,
    Qwen3MtpDrafter,
    QwenDoubleQuantError,
    SpecBucketedLinearGroup,
    dequantize_gdn_state_int8,
    inject_reasoning_prefix_property,
    merge_lora_weights_safe,
    permute_q_proj_rows_to_q_gate,
    quantize_gdn_state_int8,
    restore_hybrid_cache,
    snapshot_hybrid_cache,
    split_lcp_ids,
    validate_thinking_budget,
    verify_draft_sequence_b1,
)
from fast_qwen3.kernels import detect_rmsnorm_add_one, split_linear_q_proj_to_q_gate
from fast_qwen3.model import inspect_safetensor_header_keys, normalize_qwen3_weight_key


def _make_tiny_byte_tokenizer(tmp_path: Path) -> FastQwen3Tokenizer:
    """Create a deterministic self-contained ByteLevel `BPE` `Tokenizer` (`0` network/cache deps)."""
    vocab: dict[str, int] = {}
    for ch in ByteLevel.alphabet():
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


def _make_synthetic_qwen3_weights(
    cfg: Qwen3Config,
    *,
    seed: int = 11,
    zero_centered_norms: bool = False,
    include_mtp: bool = True,
) -> dict[str, mx.array]:
    mx.random.seed(seed)
    h = cfg.hidden_size
    v = cfg.vocab_size
    inter = cfg.intermediate_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = 2 * k_dim + v_dim
    norm_base = 0.0 if zero_centered_norms else 1.0

    def r_w(out_d: int, in_d: int) -> mx.array:
        return (mx.random.normal((out_d, in_d)) * 0.04).astype(mx.bfloat16)

    def r_n(dim: int) -> mx.array:
        return (mx.full((dim,), norm_base, dtype=mx.float32) + mx.random.normal((dim,)) * 0.02).astype(mx.bfloat16)

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
            w[f"{gpx}.conv1d.weight"] = (mx.random.normal((conv_dim, 1, cfg.linear_conv_kernel_dim)) * 0.05).astype(
                mx.bfloat16
            )
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

        if cfg.is_moe:
            mpx = f"{px}.mlp"
            ne = cfg.num_experts
            mi = cfg.moe_intermediate_size
            si = cfg.shared_expert_intermediate_size
            w[f"{mpx}.gate.weight"] = r_w(ne, h)
            w[f"{mpx}.shared_expert_gate.weight"] = r_w(1, h)
            w[f"{mpx}.shared_expert.gate_proj.weight"] = r_w(si, h)
            w[f"{mpx}.shared_expert.up_proj.weight"] = r_w(si, h)
            w[f"{mpx}.shared_expert.down_proj.weight"] = r_w(h, si)
            w[f"{mpx}.experts.gate_up_proj"] = (mx.random.normal((ne, 2 * mi, h)) * 0.03).astype(mx.bfloat16)
            w[f"{mpx}.experts.down_proj"] = (mx.random.normal((ne, h, mi)) * 0.03).astype(mx.bfloat16)
        else:
            mpx = f"{px}.mlp"
            w[f"{mpx}.gate_proj.weight"] = r_w(inter, h)
            w[f"{mpx}.up_proj.weight"] = r_w(inter, h)
            w[f"{mpx}.down_proj.weight"] = r_w(h, inter)

    if include_mtp:
        w["mtp.pre_fc_norm_embedding.weight"] = r_n(h)
        w["mtp.pre_fc_norm_hidden.weight"] = r_n(h)
        w["mtp.fc.weight"] = r_w(h, 2 * h)
        w["mtp.norm.weight"] = r_n(h)
        mp = "mtp.layers.0"
        w[f"{mp}.input_layernorm.weight"] = r_n(h)
        w[f"{mp}.post_attention_layernorm.weight"] = r_n(h)
        w[f"{mp}.self_attn.q_proj.weight"] = r_w(2 * q_dim, h)
        w[f"{mp}.self_attn.k_proj.weight"] = r_w(kv_dim, h)
        w[f"{mp}.self_attn.v_proj.weight"] = r_w(kv_dim, h)
        w[f"{mp}.self_attn.o_proj.weight"] = r_w(h, q_dim)
        w[f"{mp}.self_attn.q_norm.weight"] = r_n(cfg.head_dim)
        w[f"{mp}.self_attn.k_norm.weight"] = r_n(cfg.head_dim)
        if cfg.is_moe:
            ne = cfg.num_experts
            mi = cfg.moe_intermediate_size
            si = cfg.shared_expert_intermediate_size
            w[f"{mp}.mlp.gate.weight"] = r_w(ne, h)
            w[f"{mp}.mlp.shared_expert_gate.weight"] = r_w(1, h)
            w[f"{mp}.mlp.shared_expert.gate_proj.weight"] = r_w(si, h)
            w[f"{mp}.mlp.shared_expert.up_proj.weight"] = r_w(si, h)
            w[f"{mp}.mlp.shared_expert.down_proj.weight"] = r_w(h, si)
            w[f"{mp}.mlp.experts.gate_up_proj"] = (mx.random.normal((ne, 2 * mi, h)) * 0.03).astype(mx.bfloat16)
            w[f"{mp}.mlp.experts.down_proj"] = (mx.random.normal((ne, h, mi)) * 0.03).astype(mx.bfloat16)
        else:
            w[f"{mp}.mlp.gate_proj.weight"] = r_w(inter, h)
            w[f"{mp}.mlp.up_proj.weight"] = r_w(inter, h)
            w[f"{mp}.mlp.down_proj.weight"] = r_w(h, inter)

    mx.eval(*w.values())
    return w


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
        mtp_num_hidden_layers=1,
    )


def test_thinking_budget_valley_guard_and_schema_prefix() -> None:
    assert validate_thinking_budget(0, enable_thinking=False) == 0
    assert validate_thinking_budget(None, enable_thinking=True) == 384
    assert validate_thinking_budget(512, enable_thinking=True) == 512
    with pytest.raises(ValueError, match="truncated-thinking valley"):
        validate_thinking_budget(128, enable_thinking=True)
    assert validate_thinking_budget(64, enable_thinking=True, allow_short_thinking_budget=True) == 64

    sch = {
        "type": "object",
        "properties": {"entity": {"type": "string"}},
        "required": ["entity"],
        "additionalProperties": False,
    }
    augmented = inject_reasoning_prefix_property(sch, field_name="_reasoning", max_length=256)
    assert list(augmented["properties"].keys()) == ["_reasoning", "entity"]
    assert augmented["required"] == ["_reasoning", "entity"]


def test_non_contiguous_optiq_spec_bucketing_zero_ulp_and_lora() -> None:
    """Verify `0.0`-ULP equivalence on non-contiguous `(4, 8, 4, 4)` GDN & `[Q|Gate]` split + dynamic LoRA."""
    mx.random.seed(17)
    in_dims = 128

    def q_lin(out_d: int, bits: int) -> nn.QuantizedLinear:
        w = (mx.random.normal((out_d, in_dims)) * 0.05).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
        lin = nn.QuantizedLinear(in_dims, out_d, bias=False, group_size=64, bits=bits)
        lin.weight, lin.scales, lin.biases = wq, sc, bi
        return lin

    # Non-contiguous (4, 8, 4, 4) bit pattern observed on real Qwen3.5-4B-OptiQ-4bit layers!
    l_qkv = q_lin(192, 4)
    l_z = q_lin(128, 8)
    l_b = q_lin(16, 4)
    l_a = q_lin(16, 4)

    fused = SpecBucketedLinearGroup([("qkv", l_qkv), ("z", l_z), ("b", l_b), ("a", l_a)], enable_fusion=True)
    unfused = SpecBucketedLinearGroup([("qkv", l_qkv), ("z", l_z), ("b", l_b), ("a", l_a)], enable_fusion=False)
    assert fused.num_gemms == 2  # 4-bit {qkv, b, a} + 8-bit {z}

    x = mx.random.normal((2, 3, in_dims)).astype(mx.bfloat16)
    yf = fused(x)
    yu = unfused(x)
    mx.eval(*yf.values(), *yu.values())
    for k in ("qkv", "z", "b", "a"):
        diff = float(mx.max(mx.abs(yf[k].astype(mx.float32) - yu[k].astype(mx.float32))).item())
        assert diff == 0.0, f"Expected 0.0 ULP on {k}, got {diff}"

    # Verify [Q|Gate] split + (4, 4, 8) Self-Attention spec-bucketing + dynamic LoRA (B_Q / B_Gate)
    n_heads, hd = 2, 64
    l_q_inter = q_lin(2 * n_heads * hd, 4)
    l_k = q_lin(hd, 4)
    l_v = q_lin(hd, 8)
    lq, lg = split_linear_q_proj_to_q_gate(l_q_inter, n_heads, hd)
    attn_grp = SpecBucketedLinearGroup([("q", lq), ("k", l_k), ("v", l_v), ("gate", lg)], enable_fusion=True)
    assert attn_grp.num_gemms == 2  # 4-bit {q, k, gate} + 8-bit {v}

    la_q = (mx.random.normal((8, in_dims)) * 0.05).astype(mx.bfloat16)
    lb_q = (mx.random.normal((2 * n_heads * hd, 8)) * 0.05).astype(mx.bfloat16)
    bq, bg = permute_q_proj_rows_to_q_gate(lb_q, n_heads, hd)
    attn_grp.set_lora_slices([("q", la_q, bq, 2.0), ("gate", la_q, bg, 2.0)])

    out_grp = attn_grp(x)
    raw_q_inter = l_q_inter(x) + (mx.matmul(mx.matmul(x, la_q.T), lb_q.T) * 2.0).astype(mx.bfloat16)
    rq_split = raw_q_inter.reshape(2, 3, n_heads, 2, hd)
    ref_q = rq_split[:, :, :, 0, :].reshape(2, 3, n_heads * hd)
    ref_g = rq_split[:, :, :, 1, :].reshape(2, 3, n_heads * hd)
    mx.eval(out_grp["q"], out_grp["gate"], ref_q, ref_g)
    assert float(mx.max(mx.abs(out_grp["q"].astype(mx.float32) - ref_q.astype(mx.float32))).item()) == 0.0
    assert float(mx.max(mx.abs(out_grp["gate"].astype(mx.float32) - ref_g.astype(mx.float32))).item()) == 0.0


def test_double_4bit_quant_guard_and_bf16_optiq_merge() -> None:
    mx.random.seed(23)
    w_bf16 = (mx.random.normal((64, 128)) * 0.05).astype(mx.bfloat16)
    wq4, sc4, bi4 = mx.quantize(w_bf16, group_size=64, bits=4)
    la = (mx.random.normal((8, 128)) * 0.02).astype(mx.bfloat16)
    lb = (mx.random.normal((64, 8)) * 0.02).astype(mx.bfloat16)

    # 2-pass dequant(4b) -> merge -> requant(4b) MUST raise QwenDoubleQuantError
    with pytest.raises(QwenDoubleQuantError, match="grid-snap"):
        merge_lora_weights_safe(
            {"l.weight": wq4, "l.scales": sc4, "l.biases": bi4},
            {"l": (la, lb, 2.0)},
        )

    # 1-pass BF16 -> OptiQ 8-bit / 4-bit merge succeeds cleanly
    merged = merge_lora_weights_safe(
        {"l.weight": w_bf16},
        {"l": (la, lb, 2.0)},
        target_bit_schedule={"l": 8},
        group_size=64,
    )
    assert "l.scales" in merged and int(merged["l.weight"].shape[-1]) == (128 * 8) // 32


def test_vlm_header_skip_and_rmsnorm_autodetect(tmp_path: Path) -> None:
    cfg = _small_dense_cfg()
    # Save a pure-vision shard + a VLM-wrapped text shard (`model.language_model.*`, zero-centered norms)
    raw_w = _make_synthetic_qwen3_weights(cfg, zero_centered_norms=True, include_mtp=True)
    vlm_w: dict[str, mx.array] = {}
    for k, v in raw_w.items():
        if k.startswith("model."):
            vlm_w["model.language_model." + k[len("model.") :]] = v
        else:
            vlm_w[k] = v

    vis_shard = tmp_path / "model-00001-of-00002.safetensors"
    txt_shard = tmp_path / "model-00002-of-00002.safetensors"
    mx.save_safetensors(str(vis_shard), {"model.visual.patch_embed.weight": mx.zeros((16, 16), dtype=mx.bfloat16)})
    mx.save_safetensors(str(txt_shard), vlm_w)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 16},
                "text_config": {
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "num_hidden_layers": cfg.num_hidden_layers,
                    "num_attention_heads": cfg.num_attention_heads,
                    "num_key_value_heads": cfg.num_key_value_heads,
                    "head_dim": cfg.head_dim,
                    "vocab_size": cfg.vocab_size,
                    "tie_word_embeddings": True,
                    "full_attention_interval": 4,
                    "linear_num_value_heads": cfg.linear_num_value_heads,
                    "linear_num_key_heads": cfg.linear_num_key_heads,
                    "linear_key_head_dim": cfg.linear_key_head_dim,
                    "linear_value_head_dim": cfg.linear_value_head_dim,
                    "linear_conv_kernel_dim": 4,
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    vis_keys = inspect_safetensor_header_keys(vis_shard)
    assert all(normalize_qwen3_weight_key(k) is None for k in vis_keys)
    assert detect_rmsnorm_add_one(["model.language_model.norm.weight"], mx.zeros((32,))) is True
    assert detect_rmsnorm_add_one(["language_model.model.norm.weight"], mx.ones((32,))) is False

    model, mtp_w = FastQwen3Model.from_pretrained(tmp_path, quantize_bf16_bits=4)
    assert model.norm_add_one is True
    drafter = Qwen3MtpDrafter.from_mtp_weights(model, mtp_w)
    assert drafter is not None


def test_sparse_moe_fp32_router_and_int8_prefix_snapshot() -> None:
    moe_cfg = Qwen3Config(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=320,
        tie_word_embeddings=False,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )
    w = _make_synthetic_qwen3_weights(moe_cfg, include_mtp=False)
    model = FastQwen3Model(moe_cfg, w, quantize_bf16_bits=4)
    cache = model.make_cache()
    logits = model(mx.array([[5, 6, 7, 8, 9]], dtype=mx.int32), cache)
    mx.eval(logits)
    assert logits.shape == (1, 5, 320)
    assert int(mx.sum(mx.isnan(logits)).item()) == 0

    # Verify Block-32 INT8 state_codec round-trip & HybridPrefixSnapshot restore
    st = mx.random.normal((1, 4, 32, 32)).astype(mx.float32) * 0.4
    codes, scales = quantize_gdn_state_int8(st)
    rec = dequantize_gdn_state_int8(codes, scales)
    mx.eval(codes, scales, rec)
    assert codes.dtype == mx.int8
    cos = float(
        (mx.sum(st * rec) / (mx.sqrt(mx.sum(st * st)) * mx.sqrt(mx.sum(rec * rec)))).item()
    )
    assert cos > 0.999

    snap = snapshot_hybrid_cache([5, 6, 7, 8, 9], cache, compress_gdn_int8=True)
    c_restored = restore_hybrid_cache(snap, batch_size=2)
    out2 = model(mx.array([[10, 11], [10, 11]], dtype=mx.int32), c_restored)
    mx.eval(out2)
    assert float(mx.max(mx.abs(out2[0].astype(mx.float32) - out2[1].astype(mx.float32))).item()) == 0.0


def test_exact_zero_gemm_gdn_speculative_rollback_all_positions() -> None:
    """Prove `verify_draft_sequence_b1` matches step-by-step `L_q=1` greedy decode at `j=0, 1, 2, K`."""
    cfg = _small_dense_cfg()
    w = _make_synthetic_qwen3_weights(cfg, include_mtp=False)
    model = FastQwen3Model(cfg, w)

    prompt = [4, 7, 11, 15, 19, 23]

    # Reference: pure L_q = 1 greedy autoregressive decode for 6 tokens
    ref_cache = model.make_cache()
    h0 = model.forward_hidden(mx.array([prompt], dtype=mx.int32), ref_cache)
    l0 = model.project_logits(h0[:, -1:, :])
    mx.eval(l0)
    ref_tokens = [int(np.argmax(np.asarray(l0[0, 0].astype(mx.float32))))]
    for _ in range(5):
        hi = model.forward_hidden(mx.array([[ref_tokens[-1]]], dtype=mx.int32), ref_cache)
        li = model.project_logits(hi)
        mx.eval(li)
        ref_tokens.append(int(np.argmax(np.asarray(li[0, 0].astype(mx.float32)))))

    # Test 4 rollback scenarios for draft length K = 3 starting after ref_tokens[0]:
    # j = 0 (first draft wrong), j = 1 (1 right, 2nd wrong), j = 2 (2 right, 3rd wrong), j = 3 (all 3 right!)
    gold_next_4 = ref_tokens[1:5]
    wrong_tok = (gold_next_4[0] + 17) % cfg.vocab_size
    scenarios = [
        ([wrong_tok, 99, 101], 0, 1),
        ([gold_next_4[0], (gold_next_4[1] + 13) % cfg.vocab_size, 101], 1, 2),
        ([gold_next_4[0], gold_next_4[1], (gold_next_4[2] + 19) % cfg.vocab_size], 2, 3),
        ([gold_next_4[0], gold_next_4[1], gold_next_4[2]], 3, 4),
    ]

    for drafts, expected_matched, expected_emitted_len in scenarios:
        cand_cache = model.make_cache()
        _ = model.forward_hidden(mx.array([prompt], dtype=mx.int32), cand_cache)
        emitted, _, n_matched = verify_draft_sequence_b1(
            model,
            cand_cache,
            ref_tokens[0],
            drafts,
            pick_token_fn=lambda _i, lg: int(np.argmax(lg)),
            stop_token_ids=frozenset({0, 2}),
        )
        assert n_matched == expected_matched
        assert len(emitted) == expected_emitted_len
        assert emitted == ref_tokens[1 : 1 + expected_emitted_len]

        # Advance 1 more step after rollback and verify logits match L_q = 1 reference cache!
        next_in = emitted[-1]
        h_cand = model.forward_hidden(mx.array([[next_in]], dtype=mx.int32), cand_cache)
        l_cand = model.project_logits(h_cand)
        mx.eval(l_cand)
        cand_after = int(np.argmax(np.asarray(l_cand[0, 0].astype(mx.float32))))
        assert cand_after == ref_tokens[1 + expected_emitted_len]


def test_engine_strict_json_left_pad_batch_and_http_server(tmp_path: Path) -> None:
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size)
    w = _make_synthetic_qwen3_weights(cfg, include_mtp=True)
    model = FastQwen3Model(cfg, w)
    drafter = Qwen3MtpDrafter(cfg, w, norm_add_one=False)
    eng = FastQwen3Engine(model=model, tokenizer=tok, mtp_drafter=drafter, warmup=False)

    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["bill", "notice"]},
            "ok": {"type": "boolean"},
        },
        "required": ["kind", "ok"],
        "additionalProperties": False,
    }

    # 1. Single-item strict JSON + finish_reason="length" when max_tokens=2 is too small
    short_item = eng.generate(
        "Doc A",
        system_prompt="Return JSON.",
        json_schema=schema,
        max_tokens=2,
    )
    assert short_item.finish_reason == "length"

    full_item = eng.generate(
        "Doc A",
        system_prompt="Return JSON.",
        json_schema=schema,
        max_tokens=48,
        collect_telemetry=True,
    )
    assert full_item.valid_json is True
    assert full_item.valid_schema is True
    assert full_item.finish_reason == "stop"
    assert full_item.parsed["kind"] in ("bill", "notice")
    assert isinstance(full_item.parsed["ok"], bool)
    assert full_item.telemetry is not None

    # 2. Batched wave with shared LCP system prompt + unequal suffix lengths (`left-pad` invariance!)
    long_sys = "You are a deterministic structured extraction assistant for formal records. Always output valid JSON."
    prompts = ["Record 1", "Record number two with a much longer suffix payload for left padding"]
    b1_ref = eng.generate_batch(
        prompts,
        system_prompt=long_sys,
        json_schema=schema,
        max_tokens=48,
        batch_size=1,
        speculative=False,
    )
    batched = eng.generate_batch(
        prompts,
        system_prompt=long_sys,
        json_schema=schema,
        max_tokens=48,
        batch_size=2,
        speculative=False,
    )
    assert batched.stats["lcp_tokens"] >= 32
    assert all(it.valid_schema for it in batched.items)
    assert [it.parsed for it in batched.items] == [it.parsed for it in b1_ref.items]

    # 3. Speculative MTP B=1 greedy token-exact equivalence vs non-speculative greedy
    spec_on = eng.generate("Speculative test input", max_tokens=10, speculative=True)
    spec_off = eng.generate("Speculative test input", max_tokens=10, speculative=False)
    assert spec_on.token_ids == spec_off.token_ids

    # 4. OpenAI-compatible HTTP server smoke check
    with FastQwen3Server(eng, host="127.0.0.1", port=0) as srv:
        req_body = json.dumps(
            {
                "model": eng.model_name,
                "messages": [{"role": "user", "content": "Classify item"}],
                "max_tokens": 48,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "doc", "schema": schema},
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://{srv.host}:{srv.port}/v1/chat/completions",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed["kind"] in ("bill", "notice")

    assert split_lcp_ids([[1, 2, 3], [1, 2, 4]], min_lcp=2) == ([1, 2], [[3], [4]])


def test_post_lcp_ragged_right_pad_conv_and_gdn_state_invariance() -> None:
    """Verify right-padded post-LCP suffixes (`len=1, 2, 24`) preserve exact `conv_state` + `rec_state` vs `B=1`."""
    cfg = _small_dense_cfg()
    w = _make_synthetic_qwen3_weights(cfg)
    model = FastQwen3Model(cfg, w)

    lcp = [10 + (i % 40) for i in range(36)]
    suf_list = [
        [77],  # len=1: conv_state must combine LCP[-2:] + suf[0]
        [88, 99],  # len=2: conv_state must combine LCP[-1:] + suf[:2]
        [20 + (i * 3) % 50 for i in range(24)],  # len=24: >= conv_k
    ]

    # Build LCP snapshot once (with raw GDN state to isolate padding math from INT8 snapshotting)
    c_lcp = model.make_cache(kv_step=128)
    _ = model.forward_hidden(mx.array([lcp], dtype=mx.int32), c_lcp)
    snap = snapshot_hybrid_cache(lcp, c_lcp, compress_gdn_int8=False)

    # Reference: 3 independent B=1 runs from restored LCP snapshot
    ref_caches = []
    ref_last_h = []
    for suf in suf_list:
        c1 = restore_hybrid_cache(snap, batch_size=1, kv_step=128)
        h1 = model.forward_hidden(mx.array([suf], dtype=mx.int32), c1)
        mx.eval(h1)
        ref_caches.append(c1)
        ref_last_h.append(np.asarray(h1[0, -1, :].astype(mx.float32)))

    # Batched right-padded run (`B=3`) with `valid_token_scale` + `valid_lengths`
    max_s = max(len(s) for s in suf_list)
    padded = [s + [0] * (max_s - len(s)) for s in suf_list]
    v2d = np.zeros((3, max_s, 1), dtype=np.float32)
    for idx, s in enumerate(suf_list):
        v2d[idx, : len(s), 0] = 1.0
    cb = restore_hybrid_cache(snap, batch_size=3, kv_step=128)
    hb = model.forward_hidden(
        mx.array(padded, dtype=mx.int32),
        cb,
        valid_token_scale=mx.array(v2d, dtype=mx.bfloat16),
        valid_lengths=mx.array([len(s) for s in suf_list], dtype=mx.int32),
        attn_mask="causal",
    )
    mx.eval(hb)

    for idx, s in enumerate(suf_list):
        got_h = np.asarray(hb[idx, len(s) - 1, :].astype(mx.float32))
        cos_h = float(np.dot(got_h, ref_last_h[idx]) / (np.linalg.norm(got_h) * np.linalg.norm(ref_last_h[idx]) + 1e-9))
        assert cos_h > 0.9999 and np.max(np.abs(got_h - ref_last_h[idx])) < 3e-2
        for l_i, (e_bat, e_ref) in enumerate(zip(cb, ref_caches[idx])):
            if hasattr(e_bat, "conv_state"):
                c_got = np.asarray(e_bat.conv_state[idx].astype(mx.float32))
                c_exp = np.asarray(e_ref.conv_state[0].astype(mx.float32))
                assert np.max(np.abs(c_got - c_exp)) < 2.5e-2, f"conv_state mismatch slot={idx} layer={l_i}"
                r_got = np.asarray(e_bat.rec_state[idx].astype(mx.float32))
                r_exp = np.asarray(e_ref.rec_state[0].astype(mx.float32))
                assert np.max(np.abs(r_got - r_exp)) < 2.5e-2, f"rec_state mismatch slot={idx} layer={l_i}"


def test_prefix_tree_chunk_reuse_kv_i8_and_a4_v2_loop_breaker(tmp_path: Path) -> None:
    """Verify `256`-token `PrefixTree` reuse across distinct schemas, `kv_i8`, and `A4_v2` cycle breaker."""
    from fast_shared.grammar import compute_loop_breaker_penalties

    # 1. `A4_v2` consecutive p=1..8 cycle detector + 4-gram repeat detector
    cyc = [42, 43] * 6  # p=2 cycle repeated 6x (>= min_prior_hits=5)
    pens = compute_loop_breaker_penalties(cyc, window=96, min_prior_hits=5, base_penalty=8.0)
    assert 42 in pens and pens[42] >= 8.0

    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size)
    w = _make_synthetic_qwen3_weights(cfg)
    model = FastQwen3Model(cfg, w)
    eng = FastQwen3Engine(
        model=model,
        tokenizer=tok,
        warmup=False,
        prefix_cache_mb=32,
        kv_i8=True,
    )
    assert eng.prefix_tree is not None

    schema_a = {
        "type": "object",
        "properties": {"tag": {"type": "string", "enum": ["alpha", "beta"]}},
        "required": ["tag"],
        "additionalProperties": False,
    }
    schema_b = {
        "type": "object",
        "properties": {"count": {"type": "integer", "enum": [1, 2]}},
        "required": ["count"],
        "additionalProperties": False,
    }

    # >300 byte tokens so `PrefixTree` (`chunk_tokens=256`) publishes 1 chunk on call 1 and reuses it on call 2!
    shared_doc = "Section 101: Statutory Extraction Payload. " * 9
    r1 = eng.generate(shared_doc + "Question A?", json_schema=schema_a, max_tokens=32)
    assert r1.valid_schema is True
    assert eng.prefix_tree.node_count >= 1
    assert eng.prefix_tree.snapshot_count >= 1

    r2 = eng.generate(shared_doc + "Question B?", json_schema=schema_b, max_tokens=32)
    assert r2.valid_schema is True
    assert eng.prefix_tree.hits >= 1
    assert eng.prefix_tree.reused_tokens >= 256

    # Heterogeneous per-doc schemas + multi-token FF resynchronization under `kv_i8=True`
    het = eng.generate_batch(
        [shared_doc + "Q1", shared_doc + "Longer Q2 suffix here", "Short Q3", "Another Q4 with extra words"],
        json_schemas=[schema_a, schema_b, schema_a, schema_b],
        max_tokens=32,
        batch_size=4,
    )
    assert len(het.items) == 4
    assert all(it.valid_schema for it in het.items)
    assert het.items[0].parsed["tag"] in ("alpha", "beta")
    assert het.items[1].parsed["count"] in (1, 2)
    eng.close()
