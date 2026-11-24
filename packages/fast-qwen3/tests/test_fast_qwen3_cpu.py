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


def test_tied_vocab_remap_fallback_lora_r_ge_out_and_optiq_mtp(tmp_path: Path) -> None:
    """Verify tied `vocab_remap` slicing, `R3` LoRA `r>=out` orientation, and `optiq/mtp.safetensors` load."""
    from fast_qwen3 import QwenVocabRemapper
    from fast_qwen3.mtp_drafter import should_use_speculative_qwen

    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    w = _make_synthetic_qwen3_weights(cfg, include_mtp=True)

    remap = QwenVocabRemapper(
        np.arange(0, 260, dtype=np.int32), full_vocab_size=tok.vocab_size
    )
    model = FastQwen3Model(cfg, w, vocab_remap=remap, enable_fusion=False)
    assert model.lm_head is not None and int(model.lm_head.weight.shape[0]) == remap.vocab_size
    assert model.has_full_vocab_fallback is True
    h = mx.random.normal((1, 2, cfg.hidden_size)).astype(mx.bfloat16)
    pruned_l = model.project_logits(h, full_vocab=False)
    full_l = model.project_logits(h, full_vocab=True)
    mx.eval(pruned_l, full_l)
    assert int(pruned_l.shape[-1]) == remap.vocab_size
    assert int(full_l.shape[-1]) == cfg.vocab_size

    # `R3` `in_proj_a`/`in_proj_b` where `r=8 >= out_features=4` in both PEFT and MLX conventions
    peft_path = tmp_path / "peft_r8.safetensors"
    mx.save_safetensors(
        str(peft_path),
        {
            "base_model.model.layers.0.linear_attn.in_proj_a.lora_A.weight": mx.random.normal((8, cfg.hidden_size)).astype(mx.bfloat16),
            "base_model.model.layers.0.linear_attn.in_proj_a.lora_B.weight": mx.random.normal((cfg.linear_num_value_heads, 8)).astype(mx.bfloat16),
        },
    )
    assert model.load_lora_adapter(peft_path) == 1

    mlx_path = tmp_path / "mlx_r8.safetensors"
    mx.save_safetensors(
        str(mlx_path),
        {
            "model.layers.0.linear_attn.in_proj_b.lora_a": mx.random.normal((cfg.hidden_size, 8)).astype(mx.bfloat16),
            "model.layers.0.linear_attn.in_proj_b.lora_b": mx.random.normal((8, cfg.linear_num_value_heads)).astype(mx.bfloat16),
        },
    )
    assert model.load_lora_adapter(mlx_path) == 1

    # On-disk `optiq/mtp.safetensors` auto-discovery & `enable_fusion` propagation + `is_moe` auto-gate
    mdir = tmp_path / "optiq_repo"
    (mdir / "optiq").mkdir(parents=True)
    tok.backend_tokenizer.save(str(mdir / "tokenizer.json"))
    (mdir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
                "tie_word_embeddings": True,
                "linear_num_value_heads": cfg.linear_num_value_heads,
                "linear_num_key_heads": cfg.linear_num_key_heads,
                "linear_key_head_dim": cfg.linear_key_head_dim,
                "linear_value_head_dim": cfg.linear_value_head_dim,
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "mtp_num_hidden_layers": 1,
                "mtp_file": "optiq/mtp.safetensors",
            }
        ),
        encoding="utf-8",
    )
    base_only = {k: v for k, v in w.items() if not k.startswith("mtp.")}
    mtp_only = {k[len("mtp."):]: v for k, v in w.items() if k.startswith("mtp.")}
    mx.save_safetensors(str(mdir / "model.safetensors"), base_only)
    mx.save_safetensors(str(mdir / "optiq" / "mtp.safetensors"), mtp_only)

    from fast_qwen3.mtp_drafter import recommended_draft_k

    loaded_m, loaded_mtp = FastQwen3Model.from_pretrained(mdir, enable_fusion=False)
    drafter = Qwen3MtpDrafter.from_mtp_weights(loaded_m, loaded_mtp)
    assert drafter is not None
    assert loaded_m.enable_fusion is False
    assert recommended_draft_k(1, has_grammar=False) == 3
    assert recommended_draft_k(1, has_grammar=True) == 3
    assert should_use_speculative_qwen(1, 0.0, speculative=None, has_native_mtp=True, hidden_size=4096, is_moe=True) is False
    assert should_use_speculative_qwen(1, 0.0, speculative=None, has_native_mtp=True, hidden_size=4096, is_moe=False) is True

    # Bool `mtp_drafter=True/False` on FastQwen3Engine works cleanly without AttributeError
    (mdir / "tokenizer.json").write_text(
        tok.backend_tokenizer.to_str() if hasattr(tok, "backend_tokenizer") else (tmp_path / "tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    eng_mtp_true = FastQwen3Engine(model_dir=mdir, mtp_drafter=True, warmup=False, prefix_cache_mb=0)
    assert eng_mtp_true.mtp_drafter is not None
    eng_mtp_true.close()
    eng_mtp_false = FastQwen3Engine(model_dir=mdir, mtp_drafter=False, warmup=False, prefix_cache_mb=0)
    assert eng_mtp_false.mtp_drafter is None
    eng_mtp_false.close()


def test_thinking_early_exit_mtp_long_parity_and_seed_determinism(tmp_path: Path) -> None:
    """Verify `<think>` `upfront_ff` stop exit, `>10`-tok MTP parity vs non-spec, and `seed=` determinism."""
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=False)
    w = _make_synthetic_qwen3_weights(cfg, seed=29, include_mtp=True)
    model = FastQwen3Model(cfg, w)
    mtp_w = {k: v for k, v in w.items() if k.startswith("mtp.")}
    drafter = Qwen3MtpDrafter.from_mtp_weights(model, mtp_w)
    assert drafter is not None

    eng = FastQwen3Engine(model=model, tokenizer=tok, mtp_drafter=drafter, warmup=False)
    # 1. Fixed constant schema where `upfront_ff` closes matcher immediately (`< max_tokens`) after thinking
    const_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean", "const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    res_think = eng.generate(
        "Emit fixed boolean.",
        json_schema=const_schema,
        max_tokens=64,
        enable_thinking=True,
        thinking_budget_tokens=4,
        allow_short_thinking_budget=True,
    )
    assert res_think.valid_schema is True
    assert res_think.parsed == {"ok": True}

    # 2. >10-token MTP decode parity against `speculative=False` (syncs `mtp_cache` via exact `acc_h`)
    r_nospec = eng.generate("Summarize technical notes.", max_tokens=20, speculative=False, temperature=0.0)
    r_spec = eng.generate("Summarize technical notes.", max_tokens=20, speculative=True, temperature=0.0)
    assert r_spec.token_ids == r_nospec.token_ids
    assert len(r_spec.token_ids) >= 12

    # 3. Deterministic `seed=` sampling parity (`generate` vs repeated `generate` / `generate_batch`)
    s1 = eng.generate("Pick sample tokens.", max_tokens=12, temperature=0.8, seed=1234, speculative=False)
    s2 = eng.generate("Pick sample tokens.", max_tokens=12, temperature=0.8, seed=1234, speculative=False)
    assert s1.token_ids == s2.token_ids
    b2 = eng.generate_batch(["Pick sample tokens.", "Other prompt."], max_tokens=12, temperature=0.8, seed=1234, batch_size=2)
    assert len(b2.items) == 2
    eng.close()


def test_prefix_tree_multi_chunk_wave_reuse_and_chunked_ragged_parity(tmp_path: Path) -> None:
    """Verify atomic multi-chunk `PrefixTree` (`kv_i8`, wave reuse) and `S > 256` chunked `B=1` vs `B=4` ragged parity."""
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    w = _make_synthetic_qwen3_weights(cfg, seed=37, include_mtp=False)
    model = FastQwen3Model(cfg, w)

    prev_lim = mx.set_cache_limit(128 << 20)
    eng = FastQwen3Engine(
        model=model,
        tokenizer=tok,
        warmup=False,
        prefix_cache_mb=32,
        max_prefix_cache_entries=0,
        kv_i8=True,
        mlx_cache_limit_mb=64,
    )
    try:
        # 2+ full `256`-token chunks (`~600` byte tokens) so chunk 1 + chunk 2 publish atomically with `kv_i8`
        p_long = ("Statutory record section header. " * 20)
        b_ref = [eng.generate(p_long + sfx, max_tokens=10, temperature=0.0, speculative=False) for sfx in ("A", "Longer B suffix", "C!!", "Extra D words")]
        assert eng.prefix_tree is not None and eng.prefix_tree.snapshot_count >= 2

        # Force wave (`batch_size=4`) with `max_prefix_cache_entries=0` so `PrefixTree` feeds wave LCP directly!
        b_wave = eng.generate_batch(
            [p_long + sfx for sfx in ("A", "Longer B suffix", "C!!", "Extra D words")],
            max_tokens=10,
            temperature=0.0,
            batch_size=4,
            speculative=False,
        )
        assert [it.token_ids for it in b_wave.items] == [it.token_ids for it in b_ref]

        # Test ragged `S > 256` chunked wave prefill without `PrefixTree` interference
        eng2 = FastQwen3Engine(model=model, tokenizer=tok, warmup=False, prefix_cache_mb=0, max_prefix_cache_entries=0, kv_i8=False)
        ragged_prompts = [
            "Alpha " * 55,
            "Beta words " * 40,
            "Gamma record " * 45,
            "Delta payload " * 60,
        ]
        r_single = [eng2.generate(p, max_tokens=8, temperature=0.0, speculative=False).token_ids for p in ragged_prompts]
        r_batch = [it.token_ids for it in eng2.generate_batch(ragged_prompts, max_tokens=8, temperature=0.0, batch_size=4).items]
        assert r_batch == r_single
        eng2.close()
    finally:
        eng.close()

        mx.set_cache_limit(prev_lim)


def test_mtp_independent_norm_autodetect_and_bytelevel_vocab_remap(tmp_path: Path) -> None:
    """Verify `detect_mtp_norm_add_one` handles raw MTP weights on folded OptiQ and `build_ascii_latin_subset` caps ByteLevel CJK."""
    from tokenizers import Tokenizer as HfTokenizer
    from fast_qwen3 import QwenVocabRemapper
    from fast_qwen3.mtp_drafter import detect_mtp_norm_add_one

    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    # Main model folded (+1), raw zero-centered MTP weights (as in `9B-OptiQ` `optiq/mtp.safetensors`)
    main_w = _make_synthetic_qwen3_weights(cfg, seed=19, zero_centered_norms=False, include_mtp=False)
    mtp_raw_w = {
        k: v for k, v in _make_synthetic_qwen3_weights(cfg, seed=23, zero_centered_norms=True, include_mtp=True).items() if k.startswith("mtp.")
    }
    # Emulate negative raw pre_fc norms seen in HF Qwen3.5 MTP weights
    mtp_raw_w["mtp.pre_fc_norm_embedding.weight"] = mx.full((cfg.hidden_size,), -0.45, dtype=mx.bfloat16)
    mtp_raw_w["mtp.pre_fc_norm_hidden.weight"] = mx.full((cfg.hidden_size,), -0.25, dtype=mx.bfloat16)
    mtp_raw_w["mtp.norm.weight"] = mx.full((cfg.hidden_size,), 1.4, dtype=mx.bfloat16)

    folded_model = FastQwen3Model(cfg, main_w, norm_add_one=False, enable_fusion=False)
    assert detect_mtp_norm_add_one(mtp_raw_w, fallback=folded_model.norm_add_one) is True
    drafter_raw = Qwen3MtpDrafter.from_mtp_weights(folded_model, mtp_raw_w)
    assert drafter_raw is not None
    assert abs(float(mx.mean(drafter_raw.pre_fc_norm_embedding.weight.astype(mx.float32)).item()) - 0.55) < 0.05

    mtp_folded_w = {
        k: v for k, v in _make_synthetic_qwen3_weights(cfg, seed=29, zero_centered_norms=False, include_mtp=True).items() if k.startswith("mtp.")
    }
    assert detect_mtp_norm_add_one(mtp_folded_w, fallback=False) is False
    drafter_folded = Qwen3MtpDrafter.from_mtp_weights(folded_model, mtp_folded_w)
    assert drafter_folded is not None
    assert abs(float(mx.mean(drafter_folded.pre_fc_norm_embedding.weight.astype(mx.float32)).item()) - 1.0) < 0.08

    # ByteLevel tokenizer encodes CJK UTF-8 bytes as Latin-1 chars (< 0x0250); decoded check must filter CJK and honor target_size
    vocab = {}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    tok_bl = HfTokenizer(BPE(vocab=vocab, merges=[]))
    tok_bl.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok_bl.decoder = ByteLevelDecoder(add_prefix_space=False)
    # Add >300 ASCII tokens after a small prefix so CJK (`tid >= 275`) is outside `must_prefix` (`<= 256` for target_size=600)
    tok_bl.add_tokens(["hello", "world"])
    tok_bl.add_tokens([f"padtok{i}" for i in range(40)])
    tok_bl.add_tokens(["çļĦ", "ä¸Ń"])  # ByteLevel encodings of CJK '的' and '中'
    tok_bl.add_tokens([f"posttok{i}" for i in range(1100)])
    cjk_de = tok_bl.token_to_id("çļĦ")
    cjk_zh = tok_bl.token_to_id("ä¸Ń")
    ascii_hello = tok_bl.token_to_id("hello")
    assert cjk_de > 290 and tok_bl.decode([cjk_de]) == "的"

    remap_uncapped = QwenVocabRemapper.build_ascii_latin_subset(
        tok_bl, full_vocab_size=tok_bl.get_vocab_size(), target_size=None
    )
    # Use target_size=500 so must_prefix = 256; cjk_de (~298 >= 256) is checked cleanly and excluded
    remap_capped = QwenVocabRemapper.build_ascii_latin_subset(
        tok_bl, full_vocab_size=tok_bl.get_vocab_size(), target_size=500
    )
    assert remap_capped.vocab_size == 500
    assert ascii_hello in set(remap_capped.new_to_old.tolist())
    assert cjk_de not in set(remap_capped.new_to_old.tolist())
    assert cjk_zh not in set(remap_capped.new_to_old.tolist())
    assert remap_uncapped.vocab_size > 500


def test_fused_kv_q8_and_quarot_q4_with_prefix_tree_cpu(tmp_path: Path) -> None:
    """Verify `kv_i8='q8'` (`fused_quantized_sdpa`), `kv_i8='q4_quarot'` (`FWHT-64`), cross-mode snapshots, and `PrefixTree`."""
    tok = _make_tiny_byte_tokenizer(tmp_path)
    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    weights = _make_synthetic_qwen3_weights(cfg, seed=37, include_mtp=False)
    model = FastQwen3Model(cfg, weights, enable_fusion=True)

    ids = mx.array([[12, 19, 23, 31, 41, 47, 53, 59, 61, 67]], dtype=mx.int32)
    c_bf16 = model.make_cache(kv_i8="bf16")
    c_q8 = model.make_cache(kv_i8="q8")
    c_q4 = model.make_cache(kv_i8="q4_quarot")

    y_bf16 = model(ids[:, :8], cache=c_bf16)[:, -1, :].astype(mx.float32)
    y_q8 = model(ids[:, :8], cache=c_q8)[:, -1, :].astype(mx.float32)
    y_q4 = model(ids[:, :8], cache=c_q4)[:, -1, :].astype(mx.float32)
    mx.eval(y_bf16, y_q8, y_q4)

    def _cos(a: mx.array, b: mx.array) -> float:
        an = a / mx.maximum(mx.linalg.norm(a), 1e-8)
        bn = b / mx.maximum(mx.linalg.norm(b), 1e-8)
        return float(mx.sum(an * bn).item())

    assert _cos(y_bf16, y_q8) > 0.999
    assert _cos(y_bf16, y_q4) > 0.990

    snap_q8_8 = snapshot_hybrid_cache(ids[0, :8].tolist(), c_q8, compress_gdn_int8=False)
    snap_q4_8 = snapshot_hybrid_cache(ids[0, :8].tolist(), c_q4, compress_gdn_int8=False)

    # Decode step (`L=1`) exercises `qvm_split_k` / `qmv` strided slice path on non-compact grown cache!
    d_bf16 = model(ids[:, 8:9], cache=c_bf16)[:, -1, :].astype(mx.float32)
    d_q8 = model(ids[:, 8:9], cache=c_q8)[:, -1, :].astype(mx.float32)
    d_q4 = model(ids[:, 8:9], cache=c_q4)[:, -1, :].astype(mx.float32)
    mx.eval(d_bf16, d_q8, d_q4)
    assert _cos(d_bf16, d_q8) > 0.999
    assert _cos(d_bf16, d_q4) > 0.990

    # Rewind `GrowingKVCache` (`rewind_to(8)`) + restore GDN state matches exact pre-rewind decode logits (`0.0` ULP)!
    for cache_i, snap_8, ref_d, mode_k in ((c_q8, snap_q8_8, d_q8, "q8"), (c_q4, snap_q4_8, d_q4, "q4_quarot")):
        restored_8 = restore_hybrid_cache(snap_8, kv_i8=mode_k)
        for idx_l, lc in enumerate(cache_i):
            if hasattr(lc, "rewind_to"):
                lc.rewind_to(8)
            else:
                cache_i[idx_l] = restored_8[idx_l]
        d_replay = model(ids[:, 8:9], cache=cache_i)[:, -1, :].astype(mx.float32)
        mx.eval(d_replay)
        assert float(mx.max(mx.abs(d_replay - ref_d)).item()) == 0.0

    # Cross-mode snapshot (`bf16 -> q4_quarot` and `q4_quarot -> q8`) preserves coherent forward step
    snap_bf16 = snapshot_hybrid_cache(ids[0, :9].tolist(), c_bf16)
    c_restored_q4 = restore_hybrid_cache(snap_bf16, kv_i8="q4_quarot")
    step_from_snap = model(ids[:, 9:10], cache=c_restored_q4)[:, -1, :].astype(mx.float32)
    step_bf16_ref = model(ids[:, 9:10], cache=c_bf16)[:, -1, :].astype(mx.float32)
    mx.eval(step_from_snap, step_bf16_ref)
    assert _cos(step_bf16_ref, step_from_snap) > 0.990

    # Full `FastQwen3Engine` + `PrefixTree` chunked storage under `kv_i8="q8"` and `kv_i8="q4_quarot"`
    shared_prefix = "Shared extraction context block " * 10  # 320 byte-level tokens -> >= 1 256-token PrefixTree chunk!
    tree_bytes: dict[str, int] = {}
    for kv_mode in ("q8", "q4_quarot"):
        eng = FastQwen3Engine(
            model=model,
            tokenizer=tok,
            warmup=False,
            prefix_cache_mb=64,
            kv_i8=kv_mode,
        )
        try:
            r1 = eng.generate(shared_prefix + "Question Alpha?", max_tokens=6, temperature=0.0, speculative=False)
            r2 = eng.generate(shared_prefix + "Question Alpha?", max_tokens=6, temperature=0.0, speculative=False)
            assert r1.token_ids == r2.token_ids
            assert eng.prefix_tree is not None and eng.prefix_tree.node_count >= 1 and eng.prefix_tree.bytes_held > 0
            tree_bytes[kv_mode] = eng.prefix_tree.bytes_held
        finally:
            eng.close()
    assert tree_bytes["q4_quarot"] < tree_bytes["q8"]


def test_fast_qwen3_lazy_multimodal_generate_cpu(tmp_path: Path) -> None:
    """Verify lazy multimodal `generate(..., images=...)`, MRoPE expansion, JSON schema, and zero-ViT text isolation."""
    from PIL import Image, ImageDraw
    from tokenizers import Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.qwen35_vision import FastQwen35VisionModel, Qwen35VisionConfig

    vocab: dict[str, int] = {}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    bpe = HfTokenizer(BPE(vocab=vocab, merges=[]))
    bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bpe.decoder = ByteLevelDecoder(add_prefix_space=False)
    specials = [
        AddedToken("<|endoftext|>", special=True),
        AddedToken("<|im_start|>", special=True),
        AddedToken("<|im_end|>", special=True),
        AddedToken("<think>", special=True),
        AddedToken("</think>", special=True),
        AddedToken("<|vision_start|>", special=True),
        AddedToken("<|image_pad|>", special=True),
        AddedToken("<|vision_end|>", special=True),
    ]
    bpe.add_special_tokens(specials)
    tok_p = tmp_path / "tok_vlm.json"
    bpe.save(str(tok_p))
    tok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))

    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    weights = _make_synthetic_qwen3_weights(cfg, seed=41, include_mtp=False)
    model = FastQwen3Model(cfg, weights, enable_fusion=True)

    # 1. Text-only model_dir without vision_config raises clear ValueError on images, while text generate leaves _vision_model None
    text_only_dir = tmp_path / "text_only_ckpt"
    text_only_dir.mkdir()
    (text_only_dir / "config.json").write_text(json.dumps({"hidden_size": 128}), encoding="utf-8")
    model.model_dir = text_only_dir

    im = Image.new("RGB", (64, 64), (240, 240, 240))
    ImageDraw.Draw(im).rectangle([8, 8, 56, 56], fill=(20, 20, 20))

    eng = FastQwen3Engine(model=model, tokenizer=tok, warmup=False, prefix_cache_mb=8)
    try:
        r_txt = eng.generate("Hello", max_tokens=4, temperature=0.0, speculative=False)
        assert r_txt.completion_tokens >= 1
        assert eng._vision_model is None
        with pytest.raises(ValueError, match="vision_config"):
            eng.generate("Describe", images=[im], max_tokens=4, temperature=0.0)

        # 2. Inject synthetic tiny ViT (`out_hidden_size=128`) and verify multimodal prefill + MRoPE + JSON schema decode
        vcfg = Qwen35VisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=128,
            num_position_embeddings=64,
        )
        tiny_vis = FastQwen35VisionModel(vcfg)
        mx.eval(tiny_vis.parameters())
        eng._vision_model = tiny_vis

        r_mm = eng.generate("Extract", images=[im], max_tokens=5, temperature=0.0, speculative=False)
        assert r_mm.prompt_tokens > r_txt.prompt_tokens
        assert r_mm.completion_tokens >= 1

        sch = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        r_json = eng.generate(
            "JSON",
            images=[im],
            json_schema=sch,
            max_tokens=16,
            temperature=0.0,
            speculative=False,
        )
        assert json.loads(r_json.text)["ok"] in (True, False)
    finally:
        eng.close()

        assert eng._vision_model is None


def test_qwen3_apple10_mpp_gating_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_qwen3 import should_use_qwen_mpp_linear

    monkeypatch.delenv("FAST_SILICON_QWEN3_MPP", raising=False)
    monkeypatch.delenv("FAST_SILICON_QWEN3_MPP_W8A8", raising=False)
    monkeypatch.delenv("FAST_SILICON_MPP_W8A8", raising=False)

    # Apple9 / M4 and earlier must stay disabled regardless of shape or batch length
    for bits in (4, 8):
        assert not should_use_qwen_mpp_linear(1024, 4096, 12288, bits=bits, apple10_nax=False, allow_w8a8=True)

    # Qwen3.5-4B/9B 4-bit shapes: reject M<128 (0.33x..0.90x slower on M5), accept M>=128; reject <8M MACs
    assert not should_use_qwen_mpp_linear(32, 2560, 9216, bits=4, apple10_nax=True)
    assert not should_use_qwen_mpp_linear(64, 4096, 12288, bits=4, apple10_nax=True)
    assert should_use_qwen_mpp_linear(128, 2560, 9216, bits=4, apple10_nax=True)
    assert should_use_qwen_mpp_linear(256, 9216, 2560, bits=4, apple10_nax=True)
    assert should_use_qwen_mpp_linear(128, 4096, 24576, bits=4, apple10_nax=True)
    assert not should_use_qwen_mpp_linear(512, 4096, 1024, bits=4, apple10_nax=True)

    # 8-bit defaults OFF without opt-in
    assert not should_use_qwen_mpp_linear(512, 12288, 4096, bits=8, apple10_nax=True)
    # 8-bit opt-in activates large >=8M @ M>=64 and medium OvisOCR2 (1024x7168, 3584x1024) @ M>=128
    assert should_use_qwen_mpp_linear(64, 12288, 4096, bits=8, apple10_nax=True, allow_w8a8=True)
    assert not should_use_qwen_mpp_linear(64, 1024, 7168, bits=8, apple10_nax=True, allow_w8a8=True)
    assert should_use_qwen_mpp_linear(128, 1024, 7168, bits=8, apple10_nax=True, allow_w8a8=True)
    assert not should_use_qwen_mpp_linear(64, 3584, 1024, bits=8, apple10_nax=True, allow_w8a8=True)
    assert should_use_qwen_mpp_linear(128, 3584, 1024, bits=8, apple10_nax=True, allow_w8a8=True)
    assert not should_use_qwen_mpp_linear(512, 1024, 1024, bits=8, apple10_nax=True, allow_w8a8=True)

    # Env overrides: FAST_SILICON_QWEN3_MPP_W8A8, "w4a8", and "0"
    monkeypatch.setenv("FAST_SILICON_QWEN3_MPP_W8A8", "1")
    assert should_use_qwen_mpp_linear(64, 12288, 4096, bits=8, apple10_nax=True)
    monkeypatch.setenv("FAST_SILICON_QWEN3_MPP", "w4a8")
    assert should_use_qwen_mpp_linear(128, 4096, 12288, bits=4, apple10_nax=True)
    assert not should_use_qwen_mpp_linear(128, 12288, 4096, bits=8, apple10_nax=True)
    monkeypatch.setenv("FAST_SILICON_QWEN3_MPP", "0")
    assert not should_use_qwen_mpp_linear(512, 4096, 12288, bits=4, apple10_nax=True)


def test_qwen3_mpp_estimate_pack_bytes_respects_packs_meta_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_qwen3._mpp_linear as mpp_mod
    from fast_qwen3._mpp_linear import QwenMppPackState

    k, n = 4096, 12288
    expected_meta = 4 * n * (k // 64)
    monkeypatch.setattr(mpp_mod._metal_mpp_int8, "PACKS_META_ONLY", True, raising=False)
    assert QwenMppPackState.estimate_pack_bytes(k, n, bits=8) == expected_meta
    assert QwenMppPackState.estimate_pack_bytes(k, n, bits=4) == expected_meta

    monkeypatch.setattr(mpp_mod._metal_mpp_int8, "PACKS_META_ONLY", False, raising=False)
    assert QwenMppPackState.estimate_pack_bytes(k, n, bits=8) == (k * n) + 2 * n * (k // 64)


def test_qwen3_mpp_shared_group_budget_shed_and_lora_grad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_qwen3._mpp_linear as mpp_mod

    # Deterministic host-independent fake MPP kernels executing exact affine qmm
    pack_calls = 0
    quant_calls = 0
    mpp_calls: list[tuple[int, bool]] = []

    def _fake_pack(wq: mx.array, sq: mx.array, bq: mx.array, *, bits: int, group_size: int = 64):
        nonlocal pack_calls
        pack_calls += 1
        return mx.zeros((wq.shape[0], int(sq.shape[-1]) * 64), dtype=mx.uint8), mx.zeros_like(sq)

    def _fake_quant(x: mx.array):
        nonlocal quant_calls
        quant_calls += 1
        m_i, k_i = int(x.shape[0]), int(x.shape[1])
        return mx.zeros((m_i, k_i), dtype=mx.int8), mx.ones((m_i, k_i // 64), dtype=mx.bfloat16)

    def _fake_mpp_linear(
        x: mx.array,
        wq: mx.array,
        sq: mx.array,
        bq: mx.array,
        *,
        bits: int,
        group_size: int = 64,
        mode: str = "affine",
        bias: mx.array | None = None,
        out_dtype: mx.Dtype | None = None,
        packed_weights=None,
        prequantized_x=None,
    ) -> mx.array:
        mpp_calls.append((int(bits), prequantized_x is not None))
        assert packed_weights is not None
        out = mx.quantized_matmul(
            x, wq, scales=sq, biases=bq, transpose=True, group_size=group_size, bits=bits, mode=mode
        )
        if bias is not None:
            out = out + bias
        return out.astype(out_dtype or x.dtype)

    monkeypatch.setattr(mpp_mod, "pack_affine_weights_for_mpp_i8", _fake_pack)
    monkeypatch.setattr(mpp_mod, "quantize_activations_i8_k64", _fake_quant)
    monkeypatch.setattr(mpp_mod, "mpp_quantized_linear", _fake_mpp_linear)

    # Build a 1-layer full-attention Qwen3 model sized so MLP meets >=8M MACs (1536 x 6144 = 9.44M)
    # with mixed 4-bit + 8-bit gate_up buckets -> tests shared prequantized_x + LoRA VJP!
    cfg = Qwen3Config(
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=1,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=64,
        layer_types=["full_attention"],
        tie_word_embeddings=True,
    )
    rng = np.random.default_rng(7)

    def r_w(rows: int, cols: int) -> mx.array:
        return mx.array((rng.normal(scale=0.02, size=(rows, cols))).astype(np.float32), dtype=mx.bfloat16)

    def q_tens(prefix: str, rows: int, cols: int, bits: int, out_w: dict[str, mx.array]) -> None:
        wq, sq, bq = mx.quantize(r_w(rows, cols), group_size=64, bits=bits, mode="affine")
        out_w[f"{prefix}.weight"] = wq
        out_w[f"{prefix}.scales"] = sq
        out_w[f"{prefix}.biases"] = bq

    h, inter, v = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    w: dict[str, mx.array] = {
        "model.embed_tokens.weight": r_w(v, h),
        "model.norm.weight": mx.ones((h,), dtype=mx.bfloat16),
        "model.layers.0.input_layernorm.weight": mx.ones((h,), dtype=mx.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": mx.ones((h,), dtype=mx.bfloat16),
    }
    q_tens("model.layers.0.self_attn.q_proj", 2 * 6 * 256, h, 4, w)
    q_tens("model.layers.0.self_attn.k_proj", 2 * 256, h, 8, w)
    q_tens("model.layers.0.self_attn.v_proj", 2 * 256, h, 8, w)
    q_tens("model.layers.0.self_attn.o_proj", h, 6 * 256, 4, w)
    w["model.layers.0.self_attn.q_norm.weight"] = mx.ones((256,), dtype=mx.bfloat16)
    w["model.layers.0.self_attn.k_norm.weight"] = mx.ones((256,), dtype=mx.bfloat16)
    q_tens("model.layers.0.mlp.gate_proj", inter, h, 4, w)
    q_tens("model.layers.0.mlp.up_proj", inter, h, 8, w)
    q_tens("model.layers.0.mlp.down_proj", h, inter, 4, w)
    mx.eval(*w.values())

    ref_model = FastQwen3Model(cfg, w, enable_fusion=True, enable_mpp_linear=False)
    mpp_model = FastQwen3Model(cfg, w, enable_fusion=True, enable_mpp_linear=True, mpp_w8a8=True, apple10_nax=True)
    # Tight enough to hold gate_proj (4b), up_proj (8b), and down_proj (4b) (~28.8 MB), tested below too
    assert len(mpp_model.layers[0].mlp.gate_up.buckets) == 2

    # 1. Decode / short sequence (L=32 < 64): zero MPP dispatches, zero pack calls
    x_short = mx.array([list(range(32))], dtype=mx.int32)
    y_short_mpp = mpp_model(x_short)
    y_short_ref = ref_model(x_short)
    mx.eval(y_short_mpp, y_short_ref)
    assert pack_calls == 0 and mpp_model.mpp_linear_stats()["w4_calls"] == 0

    # 2. Prefill L=128: triggers lazy pack, shared single quantize_activations_i8_k64 for both mixed gate_up buckets, exact parity
    x_long = mx.array([[i % 64 for i in range(128)]], dtype=mx.int32)
    y_long_mpp = mpp_model(x_long)
    y_long_ref = ref_model(x_long)
    mx.eval(y_long_mpp, y_long_ref)
    assert float(mx.max(mx.abs(y_long_mpp - y_long_ref)).item()) == 0.0
    st = mpp_model.mpp_linear_stats()
    assert st["packed_modules"] == 3  # gate_proj(4b) + up_proj(8b) + down_proj(4b)
    assert st["w4_calls"] == 2 and st["w8_calls"] == 1
    assert st["shared_quant_calls"] == 1 and quant_calls == 1
    assert any(b == 4 and shared for b, shared in mpp_calls)
    assert any(b == 8 and shared for b, shared in mpp_calls)

    # 3. Autograd + LoRA compatibility through MppQuantizedLinear & MppBucketedLinearGroup
    la = mx.ones((4, h), dtype=mx.bfloat16) * 0.01
    lb = mx.ones((inter, 4), dtype=mx.bfloat16) * 0.01
    mpp_model.apply_lora_tensors({"model.layers.0.mlp.gate_proj": (la, lb, 2.0)})
    embs = mpp_model.embed_tokens(x_long)

    def _loss(e: mx.array) -> mx.array:
        return mx.mean(mpp_model.forward_hidden(inputs_embeds=e) ** 2)

    loss_val, g_emb = mx.value_and_grad(_loss)(embs)
    mx.eval(loss_val, g_emb)
    assert float(mx.sum(mx.abs(g_emb)).item()) > 0.0

    # 4. Shed clears packed weights via engine._on_memory_shed & tight budget falls back cleanly
    tok = _make_tiny_byte_tokenizer(tmp_path)
    eng = FastQwen3Engine(model=mpp_model, tokenizer=tok, warmup=False, prefix_cache_mb=0)
    try:
        eng._on_memory_shed()
        assert mpp_model.mpp_linear_stats()["packed_modules"] == 0
        mpp_model._mpp_state.max_pack_bytes = 1024  # force budget skip
        y_fallback = mpp_model(x_long)
        mx.eval(y_fallback)
        assert mpp_model.mpp_linear_stats()["budget_skips"] >= 1
    finally:
        eng.close()


def test_compact_hybrid_cache_pack_waves_and_generate_batch_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `compact_hybrid_cache`, `pack_length_waves`, and batched `generate_batch(..., images=...)` parity."""
    from PIL import Image, ImageDraw
    from tokenizers import Tokenizer as HfTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from fast_shared.qwen35_vision import FastQwen35VisionModel, Qwen35VisionConfig
    from fast_qwen3.model import DeltaSlotCache, GrowingKVCache, compact_hybrid_cache
    from fast_qwen3.multimodal_batch import pack_length_waves

    # 1. Cache row compaction slices BF16 KV, q4_quarot packed KV, and GDN DeltaSlotCache rows identically
    kv_bf16 = GrowingKVCache(step=16, kv_i8=False)
    kv_bf16.update_and_fetch(
        mx.arange(4, dtype=mx.float32)[:, None, None, None] * mx.ones((4, 2, 3, 8), dtype=mx.bfloat16),
        mx.ones((4, 2, 3, 8), dtype=mx.bfloat16),
    )
    kv_q4 = GrowingKVCache(step=16, kv_i8="q4_quarot")
    kv_q4.update_and_fetch(
        (mx.arange(4, dtype=mx.float32)[:, None, None, None] + 1.0).astype(mx.bfloat16)
        * mx.ones((4, 2, 3, 64), dtype=mx.bfloat16),
        mx.ones((4, 2, 3, 64), dtype=mx.bfloat16),
    )
    gdn = DeltaSlotCache()
    gdn.conv_state = mx.arange(4, dtype=mx.float32)[:, None, None] * mx.ones((4, 3, 8), dtype=mx.bfloat16)
    gdn.rec_state = mx.arange(4, dtype=mx.float32)[:, None, None, None] * mx.ones((4, 2, 4, 4), dtype=mx.float32)

    cache_list = [gdn, kv_bf16, kv_q4]
    compact_hybrid_cache(cache_list, [1, 3])
    assert kv_q4._qcache is not None
    assert kv_q4._qcache.k_w is not None and kv_q4._qcache.k_s is not None and kv_q4._qcache.v_w is not None
    assert kv_bf16.keys is not None and kv_bf16.values is not None
    assert gdn.conv_state is not None and gdn.rec_state is not None
    mx.eval(gdn.conv_state, gdn.rec_state, kv_bf16.keys, kv_q4._qcache.k_w, kv_q4._qcache.k_s)
    assert kv_bf16.keys.shape[0] == 2 and kv_bf16.values.shape[0] == 2
    assert float(kv_bf16.keys[0, 0, 0, 0].item()) == 1.0 and float(kv_bf16.keys[1, 0, 0, 0].item()) == 3.0
    assert kv_q4._qcache.k_w.shape[0] == 2 and kv_q4._qcache.k_s.shape[0] == 2 and kv_q4._qcache.v_w.shape[0] == 2
    k_dq, _ = kv_q4._qcache.fetch_dequantized()
    mx.eval(k_dq)
    assert abs(float(k_dq[0, 0, 0, 0].item()) - 2.0) < 0.25 and abs(float(k_dq[1, 0, 0, 0].item()) - 4.0) < 0.25
    assert gdn.conv_state.shape[0] == 2 and float(gdn.rec_state[1, 0, 0, 0].item()) == 3.0

    # 2. Wave packing handles out-of-order lengths, max_pad_ratio, and max_wave_tokens bounds
    waves = pack_length_waves([100, 120, 260, 140], lambda x: x, 4, max_pad_ratio=1.5, max_wave_tokens=1000)
    assert waves == [[100, 120], [260], [140]]

    # 3. Synthetic VLM + tiny ViT: batched generate_batch(..., images=...) matches B=1 generate(..., images=...)
    vocab: dict[str, int] = {}
    for ch in sorted(ByteLevel.alphabet()):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    bpe = HfTokenizer(BPE(vocab=vocab, merges=[]))
    bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
    bpe.decoder = ByteLevelDecoder(add_prefix_space=False)
    bpe.add_special_tokens(
        [
            AddedToken("<|endoftext|>", special=True),
            AddedToken("<|im_start|>", special=True),
            AddedToken("<|im_end|>", special=True),
            AddedToken("<think>", special=True),
            AddedToken("</think>", special=True),
            AddedToken("<|vision_start|>", special=True),
            AddedToken("<|image_pad|>", special=True),
            AddedToken("<|vision_end|>", special=True),
        ]
    )
    tok_p = tmp_path / "tok_vlm_batch.json"
    bpe.save(str(tok_p))
    tok = FastQwen3Tokenizer(tok_p, full_vocab_size=bpe.get_vocab_size(with_added_tokens=True))

    cfg = _small_dense_cfg(vocab_size=tok.vocab_size, tie=True)
    weights = _make_synthetic_qwen3_weights(cfg, seed=43, include_mtp=False)
    model = FastQwen3Model(cfg, weights, enable_fusion=True)
    vcfg = Qwen35VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=128,
        num_position_embeddings=64,
    )
    tiny_vis = FastQwen35VisionModel(vcfg)
    mx.eval(tiny_vis.parameters())

    eng = FastQwen3Engine(model=model, tokenizer=tok, warmup=False, prefix_cache_mb=8)
    eng._vision_model = tiny_vis
    try:
        im_sq = Image.new("RGB", (64, 64), (240, 240, 240))
        ImageDraw.Draw(im_sq).rectangle([8, 8, 56, 56], fill=(20, 20, 20))
        im_tall = Image.new("RGB", (64, 128), (235, 235, 235))
        ImageDraw.Draw(im_tall).rectangle([10, 10, 54, 110], fill=(30, 60, 90))

        sch = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        prompts = ["Alpha item", "Beta tall image", "Gamma text-only", "Delta square"]
        imgs: list[Any] = [[im_sq], [im_tall], None, [im_sq]]
        ref = [
            eng.generate(
                p,
                images=(im_list if im_list else None),
                json_schema=sch,
                max_tokens=16,
                temperature=0.0,
                speculative=False,
            )
            for p, im_list in zip(prompts, imgs)
        ]
        bat = eng.generate_batch(
            prompts,
            images=imgs,
            json_schema=sch,
            max_tokens=16,
            temperature=0.0,
            batch_size=4,
            compact_active=True,
            max_pad_ratio=1.8,
        )
        assert len(bat.items) == 4
        assert "waves" in bat.stats and "compactions" in bat.stats
        for r_item, b_item in zip(ref, bat.items):
            assert b_item.text == r_item.text
            assert b_item.token_ids == r_item.token_ids
            assert b_item.valid_schema is True

        # 3b. Short multimodal waves (<=1024 tok) ignore max_pad_ratio=1.8 to avoid extra decode waves,
        #     while long waves (>1024 tok) still forward max_pad_ratio cleanly.
        from fast_qwen3 import multimodal_batch as mm_mod

        im_wide = Image.new("RGB", (256, 64), (225, 225, 225))
        bat_short = eng.generate_batch(
            ["Short", "Medium"],
            images=[[im_sq], [im_wide]],
            max_tokens=4,
            temperature=0.0,
            batch_size=4,
            max_pad_ratio=1.8,
        )
        assert bat_short.stats["waves"] == 1

        seen_ratios: list[float | None] = []
        real_pack = mm_mod.pack_length_waves

        def _spy_pack(*args: Any, **kwargs: Any):
            seen_ratios.append(kwargs.get("max_pad_ratio"))
            return real_pack(*args, **kwargs)

        monkeypatch.setattr(mm_mod, "pack_length_waves", _spy_pack)
        _ = eng.generate_batch(
            ["w " * 560, "tiny"],
            images=[[im_sq], [im_sq]],
            max_tokens=2,
            temperature=0.0,
            batch_size=4,
            max_pad_ratio=1.8,
        )
        assert seen_ratios[-1] == 1.8

        # 4. Text-only compact_active batched wave matches B=1 generate
        txt_prompts = ["Short?", "Somewhat longer prompt alpha beta gamma", "Med prompt", "Another prompt here"]
        txt_ref = [eng.generate(p, max_tokens=12, temperature=0.0, speculative=False) for p in txt_prompts]
        txt_bat = eng.generate_batch(txt_prompts, max_tokens=12, temperature=0.0, batch_size=4, compact_active=True)
        for r_item, b_item in zip(txt_ref, txt_bat.items):
            assert b_item.token_ids == r_item.token_ids
    finally:
        eng.close()


def test_engine_non_lcp_make_cache_batch_size_and_kv_mode_cutoff_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure-CPU test (mocked model/`mx`) for Fix 1 (`make_cache(..., batch_size=b_sz)`) & Fix 2 (`_resolve_kv_i8`)."""
    import types
    import fast_qwen3.engine as engine_mod
    import fast_shared.hw_check as hw_check_mod

    tok = _make_tiny_byte_tokenizer(tmp_path)
    hidden_sz = 16
    vocab_sz = tok.vocab_size

    class _PureArray:
        def __init__(self, data: np.ndarray, dtype: str = "float32") -> None:
            self.data = np.asarray(data)
            self._dtype = dtype

        @property
        def shape(self) -> tuple[int, ...]:
            return self.data.shape

        @property
        def size(self) -> int:
            return int(self.data.size)

        @property
        def ndim(self) -> int:
            return int(self.data.ndim)

        @property
        def dtype(self) -> str:
            return self._dtype

        def astype(self, dtype: str) -> "_PureArray":
            np_dt = np.int32 if dtype == "int32" else np.float32
            return _PureArray(self.data.astype(np_dt), dtype=dtype)

        def tolist(self) -> Any:
            return self.data.tolist()

        def item(self) -> Any:
            return self.data.item()

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
            arr = np.asarray(self.data, dtype=dtype)
            return arr.copy() if copy else arr

        def __getitem__(self, key: Any) -> "_PureArray":
            def _u(k: Any) -> Any:
                if isinstance(k, _PureArray):
                    return k.data
                if isinstance(k, tuple):
                    return tuple(_u(x) for x in k)
                return k

            return _PureArray(self.data[_u(key)], dtype=self._dtype)

        def __invert__(self) -> "_PureArray":
            return _PureArray(~self.data, dtype="bool")

        def __and__(self, other: Any) -> "_PureArray":
            rhs = other.data if isinstance(other, _PureArray) else other
            return _PureArray(self.data & rhs, dtype="bool")

        def __ge__(self, other: Any) -> "_PureArray":
            rhs = other.data if isinstance(other, _PureArray) else other
            return _PureArray(self.data >= rhs, dtype="bool")

        def __lt__(self, other: Any) -> "_PureArray":
            rhs = other.data if isinstance(other, _PureArray) else other
            return _PureArray(self.data < rhs, dtype="bool")

        def __le__(self, other: Any) -> "_PureArray":
            rhs = other.data if isinstance(other, _PureArray) else other
            return _PureArray(self.data <= rhs, dtype="bool")

    class _PureMx:
        bfloat16 = "bfloat16"
        float32 = "float32"
        int32 = "int32"

        def array(self, data: Any, dtype: str | None = None) -> _PureArray:
            arr = data.data if isinstance(data, _PureArray) else data
            np_dt = np.int32 if dtype == self.int32 else np.float32
            return _PureArray(np.array(arr, dtype=np_dt), dtype=dtype or self.float32)

        def full(self, shape: tuple[int, ...], val: Any, dtype: str | None = None) -> _PureArray:
            np_dt = np.int32 if dtype == self.int32 else np.float32
            return _PureArray(np.full(shape, val, dtype=np_dt), dtype=dtype or self.int32)

        def arange(self, n: int, dtype: str | None = None) -> _PureArray:
            return _PureArray(np.arange(n, dtype=np.int32), dtype=dtype or self.int32)

        def take(self, arr: _PureArray, idx: _PureArray, axis: int = 0) -> _PureArray:
            return _PureArray(np.take(arr.data, idx.data, axis=axis), dtype=arr.dtype)

        def take_along_axis(self, arr: _PureArray, idx: _PureArray, axis: int = 1) -> _PureArray:
            return _PureArray(np.take_along_axis(arr.data, idx.data, axis=axis), dtype=arr.dtype)

        def argmax(self, arr: _PureArray, axis: int = -1) -> _PureArray:
            return _PureArray(np.argmax(arr.data, axis=axis).astype(np.int32), dtype=self.int32)

        def isnan(self, arr: _PureArray) -> _PureArray:
            return _PureArray(np.isnan(arr.data), dtype="bool")

        def any(self, arr: _PureArray, axis: int | None = None) -> _PureArray:
            return _PureArray(np.asarray(np.any(arr.data, axis=axis)), dtype="bool")

        def where(self, cond: _PureArray, x: Any, y: Any) -> _PureArray:
            xv = x.data if isinstance(x, _PureArray) else x
            yv = y.data if isinstance(y, _PureArray) else y
            dt = x.dtype if isinstance(x, _PureArray) else (y.dtype if isinstance(y, _PureArray) else self.float32)
            return _PureArray(np.where(cond.data, xv, yv), dtype=dt)

        def eval(self, *args: Any) -> None:
            return None

        def clear_cache(self) -> None:
            return None

    fake_mx = _PureMx()
    monkeypatch.setattr(engine_mod, "mx", fake_mx)

    make_cache_calls: list[dict[str, Any]] = []
    eos_id = tok.eos_token_id

    class _StubModel:
        def __init__(self, *, head_dim: int = 128) -> None:
            self.cfg = types.SimpleNamespace(head_dim=head_dim, hidden_size=hidden_sz, is_moe=False)
            self.model_dir = None
            self.vocab_remap = None
            self.vocab_n = vocab_sz
            self.has_full_vocab_fallback = False

        def make_cache(self, *, kv_step: int = 256, kv_i8: bool | str = False, batch_size: int = 1) -> list[dict[str, Any]]:
            make_cache_calls.append({"kv_step": kv_step, "kv_i8": kv_i8, "batch_size": batch_size})
            return [{"batch_size": batch_size, "kv_i8": kv_i8}]

        def precompute_rope_trig(self, pos_mx: _PureArray) -> tuple[_PureArray, _PureArray]:
            sh = (pos_mx.shape[0], 1, pos_mx.shape[1], 4)
            return _PureArray(np.zeros(sh, dtype=np.float32), "bfloat16"), _PureArray(np.zeros(sh, dtype=np.float32), "bfloat16")

        def forward_hidden(self, input_ids: _PureArray, cache: Any, **_kw: Any) -> _PureArray:
            b, s = input_ids.shape
            return _PureArray(np.zeros((b, s, hidden_sz), dtype=np.float32), dtype="bfloat16")

        def project_logits(self, h: _PureArray, full_vocab: bool = False) -> _PureArray:
            b, s, _ = h.shape
            lg = np.full((b, s, vocab_sz), -10.0, dtype=np.float32)
            lg[:, :, eos_id] = 10.0
            return _PureArray(lg, dtype="bfloat16")

    # 1. Verify Fix 2: _resolve_kv_i8 across Apple9 vs Apple10 and low RAM (24 GB) vs high RAM (48 GB)
    monkeypatch.setattr(hw_check_mod, "supports_apple10_nax", lambda: False)
    eng = FastQwen3Engine(model=_StubModel(head_dim=128), tokenizer=tok, warmup=False, prefix_cache_mb=0)
    try:
        assert eng._is_apple10 is False
        # Apple9, high RAM (48 GB)
        eng._is_low_ram = False
        assert eng._resolve_kv_i8(1, 512) == "bf16"
        assert eng._resolve_kv_i8(16, 512) == "bf16"
        assert eng._resolve_kv_i8(16, 4096) == "bf16"
        assert eng._resolve_kv_i8(32, 512) == "bf16"
        assert eng._resolve_kv_i8(64, 4096) == "bf16"
        assert eng._resolve_kv_i8(64, 8192) == "q4_quarot"
        assert eng._resolve_kv_i8(1, 16384) == "q4_quarot"
        assert eng._resolve_kv_i8(16, 32768) == "q4_quarot"

        # Apple9, low RAM (24 GB)
        eng._is_low_ram = True
        assert eng._resolve_kv_i8(16, 512) == "bf16"
        assert eng._resolve_kv_i8(32, 512) == "bf16"
        assert eng._resolve_kv_i8(64, 4096) == "q4_quarot"
        assert eng._resolve_kv_i8(1, 8192) == "q4_quarot"

        # Apple10 NAX, low RAM (24 GB Base M5)
        eng._is_apple10 = True
        eng._is_low_ram = True
        assert eng._resolve_kv_i8(16, 512) == "bf16"
        assert eng._resolve_kv_i8(32, 512) == "bf16"
        assert eng._resolve_kv_i8(64, 2048) == "bf16"
        assert eng._resolve_kv_i8(64, 4096) == "q4_quarot"
        assert eng._resolve_kv_i8(1, 8192) == "q4_quarot"

        # Apple10 NAX, high RAM (48 GB M5 Pro)
        eng._is_low_ram = False
        assert eng._resolve_kv_i8(32, 512) == "bf16"
        assert eng._resolve_kv_i8(64, 4096) == "bf16"
        assert eng._resolve_kv_i8(64, 8192) == "q4_quarot"
        assert eng._resolve_kv_i8(1, 16384) == "q4_quarot"

        # Non-64-divisible head_dim falls back to q8 when pressure/ctx cutoff triggers
        eng.model.cfg.head_dim = 96
        assert eng._resolve_kv_i8(64, 8192) == "q8"
        assert eng._resolve_kv_i8(16, 512) == "bf16"

        # Explicit overrides remain intact
        eng.kv_i8_mode = False
        assert eng._resolve_kv_i8(64, 16384) == "bf16"
        eng.kv_i8_mode = True
        assert eng._resolve_kv_i8(1, 256) == "q8"
        eng.kv_i8_mode = "q4_quarot"
        assert eng._resolve_kv_i8(1, 256) == "q4_quarot"
        eng.kv_i8_mode = "auto"

        # 2. Verify Fix 1: non-LCP wave calls make_cache(..., batch_size=b_sz)
        eng.model.cfg.head_dim = 128
        eng._is_apple10 = False
        eng._is_low_ram = False
        res = eng.generate_batch(
            ["Alpha prompt one", "Beta prompt two", "Gamma prompt three", "Delta prompt four"],
            max_tokens=2,
            batch_size=4,
            temperature=0.0,
        )
        assert len(res.items) == 4
        assert make_cache_calls[-1]["batch_size"] == 4
        assert make_cache_calls[-1]["kv_i8"] == "bf16"
    finally:
        eng.close()

    # Verify supports_apple10_nax=True propagates into Engine.__init__
    monkeypatch.setattr(hw_check_mod, "supports_apple10_nax", lambda: True)
    eng_a10 = FastQwen3Engine(model=_StubModel(head_dim=128), tokenizer=tok, warmup=False, prefix_cache_mb=0)
    try:
        assert eng_a10._is_apple10 is True
    finally:
        eng_a10.close()


def test_engine_lazy_gpu_argmax_fallback_and_batch_stats_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure-CPU test (mocked model/`mx`) for Fix 3 (lazy GPU argmax + selective FP32 fallback) & Fix 4 (stats)."""
    import types
    import fast_qwen3.engine as engine_mod
    import fast_shared.hw_check as hw_check_mod

    monkeypatch.setattr(hw_check_mod, "supports_apple10_nax", lambda: False)
    tok = _make_tiny_byte_tokenizer(tmp_path)
    hidden_sz = 16
    real_vocab_sz = tok.vocab_size
    padded_vocab_sz = real_vocab_sz + 16  # Padded lm_head columns beyond tokenizer vocab

    f32_casts: list[tuple[tuple[int, ...], str]] = []
    argmax_shapes: list[tuple[int, ...]] = []
    eval_shapes: list[list[tuple[int, ...]]] = []
    take_calls: list[tuple[tuple[int, ...], list[int]]] = []

    class _SpyArray:
        def __init__(self, data: np.ndarray, dtype: str = "float32") -> None:
            self.data = np.asarray(data)
            self._dtype = dtype

        @property
        def shape(self) -> tuple[int, ...]:
            return self.data.shape

        @property
        def size(self) -> int:
            return int(self.data.size)

        @property
        def ndim(self) -> int:
            return int(self.data.ndim)

        @property
        def dtype(self) -> str:
            return self._dtype

        def astype(self, dtype: str) -> "_SpyArray":
            if dtype == "float32" and self._dtype != "float32":
                f32_casts.append((self.data.shape, self._dtype))
            np_dt = np.int32 if dtype == "int32" else np.float32
            return _SpyArray(self.data.astype(np_dt), dtype=dtype)

        def tolist(self) -> Any:
            return self.data.tolist()

        def item(self) -> Any:
            return self.data.item()

        def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
            arr = np.asarray(self.data, dtype=dtype)
            return arr.copy() if copy else arr

        def __getitem__(self, key: Any) -> "_SpyArray":
            def _u(k: Any) -> Any:
                if isinstance(k, _SpyArray):
                    return k.data
                if isinstance(k, tuple):
                    return tuple(_u(x) for x in k)
                return k

            return _SpyArray(self.data[_u(key)], dtype=self._dtype)

        def __invert__(self) -> "_SpyArray":
            return _SpyArray(~self.data, dtype="bool")

        def __and__(self, other: Any) -> "_SpyArray":
            rhs = other.data if isinstance(other, _SpyArray) else other
            return _SpyArray(self.data & rhs, dtype="bool")

        def __ge__(self, other: Any) -> "_SpyArray":
            rhs = other.data if isinstance(other, _SpyArray) else other
            return _SpyArray(self.data >= rhs, dtype="bool")

        def __lt__(self, other: Any) -> "_SpyArray":
            rhs = other.data if isinstance(other, _SpyArray) else other
            return _SpyArray(self.data < rhs, dtype="bool")

        def __le__(self, other: Any) -> "_SpyArray":
            rhs = other.data if isinstance(other, _SpyArray) else other
            return _SpyArray(self.data <= rhs, dtype="bool")

    class _SpyMx:
        bfloat16 = "bfloat16"
        float32 = "float32"
        int32 = "int32"

        def array(self, data: Any, dtype: str | None = None) -> _SpyArray:
            arr = data.data if isinstance(data, _SpyArray) else data
            np_dt = np.int32 if dtype == self.int32 else np.float32
            return _SpyArray(np.array(arr, dtype=np_dt), dtype=dtype or self.float32)

        def full(self, shape: tuple[int, ...], val: Any, dtype: str | None = None) -> _SpyArray:
            np_dt = np.int32 if dtype == self.int32 else np.float32
            return _SpyArray(np.full(shape, val, dtype=np_dt), dtype=dtype or self.int32)

        def arange(self, n: int, dtype: str | None = None) -> _SpyArray:
            return _SpyArray(np.arange(n, dtype=np.int32), dtype=dtype or self.int32)

        def take(self, arr: _SpyArray, idx: _SpyArray, axis: int = 0) -> _SpyArray:
            take_calls.append((arr.shape, [int(x) for x in idx.data.tolist()]))
            return _SpyArray(np.take(arr.data, idx.data, axis=axis), dtype=arr.dtype)

        def take_along_axis(self, arr: _SpyArray, idx: _SpyArray, axis: int = 1) -> _SpyArray:
            return _SpyArray(np.take_along_axis(arr.data, idx.data, axis=axis), dtype=arr.dtype)

        def argmax(self, arr: _SpyArray, axis: int = -1) -> _SpyArray:
            argmax_shapes.append(arr.shape)
            return _SpyArray(np.argmax(arr.data, axis=axis).astype(np.int32), dtype=self.int32)

        def isnan(self, arr: _SpyArray) -> _SpyArray:
            return _SpyArray(np.isnan(arr.data), dtype="bool")

        def any(self, arr: _SpyArray, axis: int | None = None) -> _SpyArray:
            return _SpyArray(np.asarray(np.any(arr.data, axis=axis)), dtype="bool")

        def where(self, cond: _SpyArray, x: Any, y: Any) -> _SpyArray:
            xv = x.data if isinstance(x, _SpyArray) else x
            yv = y.data if isinstance(y, _SpyArray) else y
            dt = x.dtype if isinstance(x, _SpyArray) else (y.dtype if isinstance(y, _SpyArray) else self.float32)
            return _SpyArray(np.where(cond.data, xv, yv), dtype=dt)

        def eval(self, *args: Any) -> None:
            eval_shapes.append([a.shape for a in args if hasattr(a, "shape")])

        def clear_cache(self) -> None:
            return None

    fake_mx = _SpyMx()
    monkeypatch.setattr(engine_mod, "mx", fake_mx)

    def _reset_spies() -> None:
        f32_casts.clear()
        argmax_shapes.clear()
        eval_shapes.clear()
        take_calls.clear()

    # Token IDs for boolean schema: {"ok": true|false}
    t_tok = tok.encode("true", add_special_tokens=False)[0]
    f_tok = tok.encode("false", add_special_tokens=False)[0]
    bad_tok = tok.encode("Z", add_special_tokens=False)[0]
    tok_a = tok.encode("a", add_special_tokens=False)[0]
    tok_b = tok.encode("b", add_special_tokens=False)[0]
    tok_c = tok.encode("c", add_special_tokens=False)[0]
    eos_id = tok.eos_token_id
    bool_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    class _ScriptedModel:
        def __init__(self, *, vocab_n: int = real_vocab_sz) -> None:
            self.cfg = types.SimpleNamespace(head_dim=128, hidden_size=hidden_sz, is_moe=False)
            self.model_dir = None
            self.vocab_remap = None
            self.vocab_n = vocab_n
            self.has_full_vocab_fallback = False
            self.step_logits_fn: Any = None
            self.project_calls = 0

        def make_cache(self, *, kv_step: int = 256, kv_i8: bool | str = False, batch_size: int = 1) -> list[dict[str, Any]]:
            return [{"step": 0, "b": batch_size}]

        def precompute_rope_trig(self, pos_mx: _SpyArray) -> tuple[_SpyArray, _SpyArray]:
            sh = (pos_mx.shape[0], 1, pos_mx.shape[1], 4)
            return _SpyArray(np.zeros(sh, dtype=np.float32), "bfloat16"), _SpyArray(np.zeros(sh, dtype=np.float32), "bfloat16")

        def forward_hidden(self, input_ids: _SpyArray, cache: list[dict[str, Any]], **_kw: Any) -> _SpyArray:
            b, s = input_ids.shape
            step_idx = int(cache[0]["step"])
            cache[0]["step"] = step_idx + 1
            h_arr = np.zeros((b, s, hidden_sz), dtype=np.float32)
            for row in range(b):
                h_arr[row, :, 0] = float(row)
                h_arr[row, :, 1] = float(step_idx)
            return _SpyArray(h_arr, dtype="bfloat16")

        def project_logits(self, h: _SpyArray, full_vocab: bool = False) -> _SpyArray:
            self.project_calls += 1
            b, s, _ = h.shape
            out = np.full((b, s, self.vocab_n), -20.0, dtype=np.float32)
            for row in range(b):
                slot_id = int(round(float(h.data[row, 0, 0])))
                step_idx = int(round(float(h.data[row, 0, 1])))
                out[row, 0] = self.step_logits_fn(slot_id, step_idx, self.vocab_n)
            return _SpyArray(out, dtype="bfloat16")

    model = _ScriptedModel(vocab_n=padded_vocab_sz)
    eng = FastQwen3Engine(model=model, tokenizer=tok, warmup=False, prefix_cache_mb=0)
    try:
        # 1. Empty batch stats
        empty_res = eng.generate_batch([])
        assert empty_res.items == []
        assert empty_res.stats["docs"] == 0
        assert empty_res.stats["prompt_tokens"] == 0
        assert empty_res.stats["completion_tokens"] == 0
        assert empty_res.stats["mask_fallbacks"] == 0

        # 2. Unconstrained greedy batch + padded vocab column (> real_vocab_sz) with huge/NaN logit:
        #    _project_needed_rows must slice to len(_new_to_old) BEFORE mx.argmax, and perform 0 float32 casts!
        def _logits_padded_unconstrained(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            row[real_vocab_sz + 3] = float("nan")  # Must be sliced away before argmax/NaN check!
            row[real_vocab_sz + 5] = 999.0
            if step_idx == 0:
                row[tok_a if slot_id == 0 else tok_b] = 5.0
            else:
                row[eos_id] = 6.0
            return row

        model.step_logits_fn = _logits_padded_unconstrained
        _reset_spies()
        res_uncon = eng.generate_batch(["Prompt zero", "Prompt one here"], max_tokens=4, batch_size=2, temperature=0.0)
        assert [it.token_ids for it in res_uncon.items] == [[tok_a], [tok_b]]
        assert f32_casts == [], f"Expected zero FP32 casts on greedy unconstrained wave, got {f32_casts}"
        assert all(sh[-1] == real_vocab_sz for sh in argmax_shapes)
        assert res_uncon.stats["mask_fallbacks"] == 0
        assert res_uncon.stats["prompt_tokens"] == sum(it.prompt_tokens for it in res_uncon.items)
        assert res_uncon.stats["completion_tokens"] == sum(it.completion_tokens for it in res_uncon.items)

        # 3. Grammar-constrained B=2 wave where ALL rows pass matcher.validate_tokens -> 0 FP32 casts & 0 fallbacks
        def _logits_grammar_hit(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            row[t_tok if slot_id == 0 else f_tok] = 7.0
            return row

        model.step_logits_fn = _logits_grammar_hit
        _reset_spies()
        res_ghit = eng.generate_batch(
            ["Doc zero", "Doc one with extra text"],
            json_schema=bool_schema,
            max_tokens=32,
            batch_size=2,
            temperature=0.0,
        )
        assert [it.parsed for it in res_ghit.items] == [{"ok": True}, {"ok": False}]
        assert all(it.valid_schema for it in res_ghit.items)
        assert f32_casts == [], f"Expected zero FP32 casts when GPU argmax passes matcher, got {f32_casts}"
        assert res_ghit.stats["mask_fallbacks"] == 0

        # 4. Grammar-constrained B=2 wave with MASK MISS on row 1 only:
        #    Row 0 greedy argmax (t_tok) passes validate_tokens; Row 1 greedy argmax (bad_tok) misses ->
        #    validate_tokens MUST NOT mutate matcher state, and ONLY row 1 is cast to float32!
        def _logits_one_row_miss(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            if slot_id == 0:
                row[t_tok] = 9.0
            else:
                row[bad_tok] = 12.0  # Disallowed after '{"ok":'
                row[f_tok] = 6.0     # Highest valid token after mask fallback
            return row

        model.step_logits_fn = _logits_one_row_miss
        _reset_spies()
        res_miss = eng.generate_batch(
            ["Prompt A", "Prompt B slightly longer"],
            json_schema=bool_schema,
            max_tokens=32,
            batch_size=2,
            temperature=0.0,
        )
        assert [it.parsed for it in res_miss.items] == [{"ok": True}, {"ok": False}]
        assert all(it.valid_schema for it in res_miss.items)
        assert res_miss.stats["mask_fallbacks"] == 1
        assert len(f32_casts) == 1 and f32_casts[0][0] == (1, real_vocab_sz)
        assert len(take_calls) == 1 and take_calls[0] == ((2, real_vocab_sz), [1])

        # Verify exact greedy parity with _select_next_token for row 1's missed step
        row1_logits = _logits_one_row_miss(1, 0, padded_vocab_sz)
        gh = eng._get_grammar_helper(bool_schema, None)
        ref_tid, _ = eng._select_next_token(
            row1_logits,
            full_hidden_1d=None,
            matcher=gh.new_matcher(),
            gram=gh,
            recent_tokens=[],
            temperature=0.0,
            repeat_penalty=True,
            collect_telemetry=False,
        )
        assert ref_tid == f_tok

        # 5. NaN in valid vocab on row 0 in B=2 grammar batch -> NaN guard (-1) triggers fallback for row 0 only
        def _logits_row0_nan(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            if slot_id == 0:
                row[tok_c] = float("nan")
                row[t_tok] = 5.0
            else:
                row[f_tok] = 8.0
            return row

        model.step_logits_fn = _logits_row0_nan
        _reset_spies()
        res_nan = eng.generate_batch(
            ["Doc NaN", "Doc OK longer"],
            json_schema=bool_schema,
            max_tokens=32,
            batch_size=2,
            temperature=0.0,
        )
        assert res_nan.items[0].valid_schema and res_nan.items[1].valid_schema
        assert res_nan.items[1].parsed == {"ok": False}
        assert res_nan.stats["mask_fallbacks"] == 1
        assert len(f32_casts) == 1 and f32_casts[0][0] == (1, real_vocab_sz)

        # 6. Repeat-penalty override in greedy B=2 wave (cycle p=2 repeated 4x triggers penalty at step 8)
        #    Keep "Short stop" shorter than "Cycle prompt longer payload" so sorted wave places Cycle at slot 1.
        def _logits_repeat_cycle(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            if slot_id == 0:
                row[eos_id] = 10.0
                return row
            if step_idx >= 9:
                row[eos_id] = 10.0
                return row
            # Slot 1 emits [tok_a, tok_b] * 4 (8 tokens) -> at step 8, tok_a is penalized (-16.0), so tok_c (2.0) wins!
            if step_idx % 2 == 0:
                row[tok_a] = 10.0
                row[tok_c] = 2.0
            else:
                row[tok_b] = 10.0
            return row

        model.step_logits_fn = _logits_repeat_cycle
        _reset_spies()
        res_rep = eng.generate_batch(
            ["Short stop", "Cycle prompt longer payload"],
            max_tokens=12,
            batch_size=2,
            temperature=0.0,
        )
        assert res_rep.items[0].token_ids == []
        assert res_rep.items[1].token_ids == [tok_a, tok_b, tok_a, tok_b, tok_a, tok_b, tok_a, tok_b, tok_c]
        assert res_rep.stats["mask_fallbacks"] == 0
        # Only the single step where repeat penalty hit tok_a materialized a [1, V] float32 row!
        assert len(f32_casts) == 1 and f32_casts[0][0] == (1, real_vocab_sz)

        # 7. Non-greedy (temperature > 0) & telemetry (collect_telemetry=True) + B=1 mask_fallbacks aggregation
        model.step_logits_fn = _logits_one_row_miss
        _reset_spies()
        res_temp = eng.generate_batch(
            ["Temp doc A", "Temp doc B"],
            json_schema=bool_schema,
            max_tokens=32,
            batch_size=2,
            temperature=0.5,
            seed=7,
            collect_telemetry=True,
        )
        assert all(it.valid_schema and it.telemetry is not None for it in res_temp.items)
        assert res_temp.stats["mask_fallbacks"] >= 1
        assert len(f32_casts) >= 1 and argmax_shapes == []

        # 8. Mixed wave (3 docs, batch_size=2 -> wave of 2 + B=1 _decode_single):
        #    Verify lazy argmax + mask_fallbacks sums cleanly across both _decode_wave_batched and _decode_single!
        def _logits_3docs(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            row[bad_tok] = 12.0
            row[t_tok] = 6.0
            return row

        model.step_logits_fn = _logits_3docs
        _reset_spies()
        res_3 = eng.generate_batch(
            ["Short 1", "Medium two", "Longest three payload"],
            json_schema=bool_schema,
            max_tokens=32,
            batch_size=2,
            temperature=0.0,
        )
        assert all(it.parsed == {"ok": True} for it in res_3.items)
        assert res_3.stats["mask_fallbacks"] == 3
        assert res_3.stats["prompt_tokens"] == sum(it.prompt_tokens for it in res_3.items)
        assert res_3.stats["completion_tokens"] == sum(it.completion_tokens for it in res_3.items)

        # 9. Batch compaction (B=4 -> 2 alive rows) preserves live_rows matcher/row alignment in lazy argmax + fallback
        compact_calls: list[list[int]] = []
        monkeypatch.setattr(
            engine_mod,
            "compact_hybrid_cache",
            lambda _cache, alive_r: compact_calls.append(list(alive_r)) or list(alive_r),
        )

        class _CompactStubModel(_ScriptedModel):
            def forward_hidden(self, input_ids: _SpyArray, cache: list[dict[str, Any]], **_kw: Any) -> _SpyArray:
                b, s = input_ids.shape
                step_idx = int(cache[0]["step"])
                cache[0]["step"] = step_idx + 1
                if step_idx == 0:
                    rows = list(range(b))
                elif b == 2:
                    # After compaction of [0, 1, 2, 3] -> [1, 3], rows track original slots 1 and 3!
                    rows = [1, 3]
                else:
                    rows = list(range(b))
                h_arr = np.zeros((b, s, hidden_sz), dtype=np.float32)
                for r_i, orig_slot in enumerate(rows):
                    h_arr[r_i, :, 0] = float(orig_slot)
                    h_arr[r_i, :, 1] = float(step_idx)
                return _SpyArray(h_arr, dtype="bfloat16")

        def _logits_compact(slot_id: int, step_idx: int, v_n: int) -> np.ndarray:
            row = np.full((v_n,), -20.0, dtype=np.float32)
            if slot_id in (0, 2):
                row[eos_id] = 10.0  # Finish immediately at step 0 -> triggers compaction of [1, 3]!
            elif step_idx == 0:
                row[tok_a if slot_id == 1 else tok_b] = 9.0
            elif step_idx == 1:
                row[tok_c if slot_id == 1 else bad_tok] = 9.0
            else:
                row[eos_id] = 9.0
            return row

        c_model = _CompactStubModel(vocab_n=real_vocab_sz)
        c_model.step_logits_fn = _logits_compact
        c_eng = FastQwen3Engine(model=c_model, tokenizer=tok, warmup=False, prefix_cache_mb=0)
        try:
            _reset_spies()
            res_cmp = c_eng.generate_batch(
                ["A0", "B1aa", "C2bbbb", "D3cccccc"],
                max_tokens=4,
                batch_size=4,
                temperature=0.0,
                compact_active=True,
            )
            assert compact_calls == [[1, 3]]
            assert [it.token_ids for it in res_cmp.items] == [[], [tok_a, tok_c], [], [tok_b, bad_tok]]
            assert f32_casts == []
            assert res_cmp.stats["mask_fallbacks"] == 0
        finally:
            c_eng.close()

        # 10. Pruned vocab_remap + full_vocab fallback triggers cleanly when lazy argmax misses grammar
        from fast_qwen3 import QwenVocabRemapper

        kept_ids = np.array([i for i in range(real_vocab_sz) if i not in (t_tok, f_tok)], dtype=np.int32)
        remap = QwenVocabRemapper(kept_ids, full_vocab_size=real_vocab_sz)
        pruned_bad_idx = int(remap.old_to_new[bad_tok])

        class _RemapStubModel(_ScriptedModel):
            def __init__(self) -> None:
                super().__init__(vocab_n=len(kept_ids))
                self.vocab_remap = remap
                self.has_full_vocab_fallback = True
                self.full_vocab_calls = 0

            def project_logits(self, h: _SpyArray, full_vocab: bool = False) -> _SpyArray:
                if full_vocab:
                    self.full_vocab_calls += 1
                    b, s, _ = h.shape
                    out = np.full((b, s, real_vocab_sz), -20.0, dtype=np.float32)
                    out[:, :, f_tok] = 15.0
                    return _SpyArray(out, dtype="bfloat16")
                b, s, _ = h.shape
                out = np.full((b, s, len(kept_ids)), -20.0, dtype=np.float32)
                out[:, :, pruned_bad_idx] = 10.0
                return _SpyArray(out, dtype="bfloat16")

        r_model = _RemapStubModel()
        r_eng = FastQwen3Engine(model=r_model, tokenizer=tok, warmup=False, prefix_cache_mb=0)
        try:
            _reset_spies()
            res_rm = r_eng.generate_batch(
                ["Remap doc 1", "Remap doc 2 longer"],
                json_schema=bool_schema,
                max_tokens=32,
                batch_size=2,
                temperature=0.0,
            )
            assert [it.parsed for it in res_rm.items] == [{"ok": False}, {"ok": False}]
            assert r_model.full_vocab_calls == 2
            assert res_rm.stats["mask_fallbacks"] == 2
        finally:
            r_eng.close()
    finally:
        eng.close()






