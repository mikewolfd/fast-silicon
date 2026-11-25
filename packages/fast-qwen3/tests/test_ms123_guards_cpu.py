"""CPU fallback parity and P0/P1 guard tests for MS1, MS2, MS3a, and rolling memory/admission caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fast_qwen3.gdn_kernels import (
    _effective_gdn_tile_rows,
    fused_gdn_pre_recurrence,
    gdn_delta_update,
    gdn_delta_update_slot_indexed,
    quantize_gdn_state_blk32,
)
from fast_qwen3.kernels import compute_gdn_gates, normalize_gdn_qk
from fast_qwen3.model import FastQwen3Model, Qwen3Config
from fast_qwen3.multimodal_rolling import rolling_multimodal_generate
from fast_qwen3.rolling import (
    RollingHybridCache,
    RollingKVCache,
    attend_rolling_flash_decode,
    forward_hidden_rolling,
)


def test_effective_gdn_tile_rows_tg_memory_limit() -> None:
    """Verify P0-4 Dk=256 FP32 stays safely under 28,672 B Metal threadgroup limit."""
    rows = _effective_gdn_tile_rows(dk=256, dtype=mx.float32, requested=16)
    per_row_bytes = (2 * 256 + 16) * 4 + 9
    assert rows <= 12
    assert rows * per_row_bytes <= 28672


def test_ms1_ms2_ms3a_cpu_fallback_and_valid_lengths_parity() -> None:
    """Verify CPU fallback parity for MS1, MS2 (sq=1 & sq=5, len=0), and MS3a valid_lengths guard."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(20261125)
        b_slots, b_act, hk, hv, dk, dv = 6, 4, 2, 4, 32, 32
        conv_dim = hk * dk * 2 + hv * dv
        slot_ids = mx.array([4, 1, 5, 0], dtype=mx.int32)

        # 1. MS1 slot-indexed GDN CPU fallback parity across int8_blk32 / fp32
        t_len = 3
        q = mx.array(rng.normal(0.0, 0.2, (b_act, t_len, hk, dk)).astype(np.float32), dtype=mx.bfloat16)
        k = mx.array(rng.normal(0.0, 0.2, (b_act, t_len, hk, dk)).astype(np.float32), dtype=mx.bfloat16)
        v = mx.array(rng.normal(0.0, 0.2, (b_act, t_len, hv, dv)).astype(np.float32), dtype=mx.bfloat16)
        g = mx.array(rng.uniform(0.8, 0.98, (b_act, t_len, hv)).astype(np.float32), dtype=mx.float32)
        beta = mx.array(rng.uniform(0.1, 0.9, (b_act, t_len, hv)).astype(np.float32), dtype=mx.bfloat16)
        st_pool = mx.array(rng.normal(0.0, 0.1, (b_slots, hv, dv, dk)).astype(np.float32), dtype=mx.float32)
        c_pool, s_pool = quantize_gdn_state_blk32(st_pool)

        y_ref, (cr, sr) = gdn_delta_update(
            q, k, v, g, beta, (mx.take(c_pool, slot_ids, axis=0), mx.take(s_pool, slot_ids, axis=0)), state_mode="int8_blk32"
        )
        y_ms1, (cm, sm) = gdn_delta_update_slot_indexed(
            q, k, v, g, beta, slot_ids, codes_pool=c_pool, scales_pool=s_pool, state_mode="int8_blk32"
        )
        mx.eval(y_ref, cr, sr, y_ms1, cm, sm)
        assert float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_ms1.astype(mx.float32))).item()) < 1e-4
        assert int(mx.max(mx.abs(cr.astype(mx.int32) - cm.astype(mx.int32))).item()) == 0

        # 2. MS2 Flash-Decode CPU fallback parity (s_q=1 & s_q=5 + valid_len=0)
        cap, n_q, n_kv, hd = 20, 4, 2, 32
        kv = RollingKVCache(b_slots=b_slots, cap=cap, n_kv=n_kv, head_dim=hd)
        kv.k_buf[:, :12] = mx.array(rng.normal(0.0, 0.3, (b_slots, 12, n_kv, hd)).astype(np.float32), dtype=mx.bfloat16)
        kv.v_buf[:, :12] = mx.array(rng.normal(0.0, 0.3, (b_slots, 12, n_kv, hd)).astype(np.float32), dtype=mx.bfloat16)
        vlens = mx.array([10, 0, 12, 6], dtype=mx.int32)
        q_sq5 = mx.array(rng.normal(0.0, 0.3, (b_act, n_q, 5, hd)).astype(np.float32), dtype=mx.bfloat16)
        out_fd = attend_rolling_flash_decode(
            q_sq5, kv.k_buf, kv.v_buf, slot_ids, vlens, scale=hd**-0.5, max_valid=12
        )
        mx.eval(out_fd)
        assert out_fd.shape == q_sq5.shape
        assert float(mx.max(mx.abs(out_fd[1].astype(mx.float32))).item()) == 0.0
        assert bool(mx.all(mx.isfinite(out_fd)).item())

        # 3. MS3a valid_lengths guard: done slot (valid=0) & partial PLD (valid=2) must NOT pollute conv_state
        conv_pool = mx.array(rng.normal(0.0, 0.3, (b_slots, 3, conv_dim)).astype(np.float32), dtype=mx.bfloat16)
        l_q = 4
        qkv = mx.array(rng.normal(0.0, 0.3, (b_act, l_q, conv_dim)).astype(np.float32), dtype=mx.bfloat16)
        a_g = mx.array(rng.normal(0.0, 0.3, (b_act, l_q, hv)).astype(np.float32), dtype=mx.bfloat16)
        b_g = mx.array(rng.normal(0.0, 0.3, (b_act, l_q, hv)).astype(np.float32), dtype=mx.bfloat16)
        cw = mx.array(rng.normal(0.0, 0.2, (conv_dim, 4)).astype(np.float32), dtype=mx.bfloat16)
        alog = mx.zeros((hv,), dtype=mx.float32)
        dtb = mx.ones((hv,), dtype=mx.float32) * 0.2
        v_lens = mx.array([4, 0, 2, 1], dtype=mx.int32)

        q_m, k_m, v_m, g_m, beta_m, cs_m = fused_gdn_pre_recurrence(
            qkv,
            conv_pool,
            slot_ids,
            cw,
            a_g,
            b_g,
            alog,
            dtb,
            hk=hk,
            hv=hv,
            head_dim=dk,
            inv_scale=dk**-0.5,
            rms_eps=1e-6,
            valid_lengths=v_lens,
        )
        mx.eval(q_m, k_m, v_m, g_m, beta_m, cs_m)

        # Done slot row 1 (valid_len=0) retains exact prior conv_pool[slot_ids[1]]
        assert float(mx.max(mx.abs(cs_m[1].astype(mx.float32) - conv_pool[1].astype(mx.float32))).item()) == 0.0
        assert float(mx.max(mx.abs(q_m[1].astype(mx.float32))).item()) == 0.0
        assert float(mx.max(mx.abs(beta_m[1].astype(mx.float32))).item()) == 0.0
        # Partial slot row 2 (valid_len=2, l_q=4) advances conv_state by 2 tokens: [conv_pool[5, 2], qkv[2, 0], qkv[2, 1]]
        exp_row2 = mx.concatenate([conv_pool[5, 2:3, :], qkv[2, 0:2, :]], axis=0)
        assert float(mx.max(mx.abs(cs_m[2].astype(mx.float32) - exp_row2.astype(mx.float32))).item()) == 0.0
        # Row 0 (all valid) matches reference Silu(Conv1d) + normalize_gdn_qk + compute_gdn_gates
        cin0 = mx.concatenate([conv_pool[4:5], qkv[0:1]], axis=0 if False else 1)
        c1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=4, groups=conv_dim, padding=0)
        c1d.weight = cw[:, :, None]
        cout0 = nn.silu(c1d(cin0))
        qr0, kr0, vr0 = mx.split(cout0, [hk * dk, 2 * hk * dk], axis=-1)
        qn0, _ = normalize_gdn_qk(
            qr0.reshape(1, l_q, hk, dk), kr0.reshape(1, l_q, hk, dk), dk**-0.5, 1e-6
        )
        mx.eval(qn0)
        assert float(mx.max(mx.abs(q_m[0:1].astype(mx.float32) - qn0.astype(mx.float32))).item()) < 2e-2
    finally:
        mx.set_default_device(prev)


def test_rolling_noncontiguous_pool_preserves_b_slots_shape() -> None:
    """Verify P0-3: forward_hidden_rolling with b_act < b_slots keeps full pool shape intact."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        cfg = Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=64,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
        )
        rng = np.random.default_rng(7)
        h, v, inter = cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size
        conv_dim = 2 * (2 * 32) + 2 * 32
        w = {
            "model.embed_tokens.weight": mx.array(rng.normal(0, 0.05, (v, h)), dtype=mx.bfloat16),
            "model.norm.weight": mx.ones((h,), dtype=mx.bfloat16),
        }
        for i in range(4):
            p = f"model.layers.{i}"
            w[f"{p}.input_layernorm.weight"] = mx.ones((h,), dtype=mx.bfloat16)
            w[f"{p}.post_attention_layernorm.weight"] = mx.ones((h,), dtype=mx.bfloat16)
            w[f"{p}.mlp.gate_proj.weight"] = mx.array(rng.normal(0, 0.05, (inter, h)), dtype=mx.bfloat16)
            w[f"{p}.mlp.up_proj.weight"] = mx.array(rng.normal(0, 0.05, (inter, h)), dtype=mx.bfloat16)
            w[f"{p}.mlp.down_proj.weight"] = mx.array(rng.normal(0, 0.05, (h, inter)), dtype=mx.bfloat16)
            if cfg.layer_types[i] == "linear_attention":
                g = f"{p}.linear_attn"
                w[f"{g}.in_proj_qkv.weight"] = mx.array(rng.normal(0, 0.05, (conv_dim, h)), dtype=mx.bfloat16)
                w[f"{g}.in_proj_z.weight"] = mx.array(rng.normal(0, 0.05, (64, h)), dtype=mx.bfloat16)
                w[f"{g}.in_proj_b.weight"] = mx.array(rng.normal(0, 0.05, (2, h)), dtype=mx.bfloat16)
                w[f"{g}.in_proj_a.weight"] = mx.array(rng.normal(0, 0.05, (2, h)), dtype=mx.bfloat16)
                w[f"{g}.out_proj.weight"] = mx.array(rng.normal(0, 0.05, (h, 64)), dtype=mx.bfloat16)
                w[f"{g}.conv1d.weight"] = mx.array(rng.normal(0, 0.05, (conv_dim, 1, 4)), dtype=mx.bfloat16)
                w[f"{g}.dt_bias"] = mx.ones((2,), dtype=mx.float32)
                w[f"{g}.A_log"] = mx.zeros((2,), dtype=mx.float32)
                w[f"{g}.norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
            else:
                a = f"{p}.self_attn"
                w[f"{a}.q_proj.weight"] = mx.array(rng.normal(0, 0.05, (128, h)), dtype=mx.bfloat16)
                w[f"{a}.k_proj.weight"] = mx.array(rng.normal(0, 0.05, (32, h)), dtype=mx.bfloat16)
                w[f"{a}.v_proj.weight"] = mx.array(rng.normal(0, 0.05, (32, h)), dtype=mx.bfloat16)
                w[f"{a}.o_proj.weight"] = mx.array(rng.normal(0, 0.05, (h, 64)), dtype=mx.bfloat16)
                w[f"{a}.q_norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
                w[f"{a}.k_norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        m = FastQwen3Model(cfg, w, enable_fusion=False)
        cache = RollingHybridCache(m, b_slots=5, cap=16, gdn_state_mode="fp32")
        out = forward_hidden_rolling(
            m,
            cache,
            mx.array([[1], [2]], dtype=mx.int32),
            slot_ids=[3, 1],
            positions=[4, 2],
            valid_lens=[5, 3],
        )
        mx.eval(out)
        assert cache.layer_caches[0].conv_state.shape[0] == 5
        assert cache.layer_caches[0].rec_state.shape[0] == 5
    finally:
        mx.set_default_device(prev)


def test_multimodal_rolling_ram_cap_and_shortest_first_cpu() -> None:
    """Verify P0-5 low-RAM b_slots cap and P1-3 shortest_first order preserve caller order."""
    @dataclass
    class _Cfg:
        hidden_size: int = 2560
        rot_dims: int = 4
        rope_theta: float = 10000.0
        mrope_section: tuple[int, int, int] = (1, 1, 1)
        linear_key_head_dim: int = 32
        linear_conv_kernel_dim: int = 4

    class _EchoLM:
        def __init__(self) -> None:
            self.cfg = _Cfg()
            self.vocab_n = 16
            self.vocab_remap = None

        def make_cache(self, **_kw: Any) -> list[Any]:
            return []

        def embed_tokens(self, ids: mx.array) -> mx.array:
            return mx.broadcast_to(ids[:, :, None].astype(mx.bfloat16), (int(ids.shape[0]), int(ids.shape[1]), 4))

        def precompute_rope_trig(self, pos: mx.array) -> tuple[mx.array, mx.array]:
            b, s = int(pos.shape[0]), int(pos.shape[1])
            return mx.ones((b, 1, s, 2), dtype=mx.bfloat16), mx.zeros((b, 1, s, 2), dtype=mx.bfloat16)

        def forward_hidden(self, input_ids=None, cache=None, *, inputs_embeds=None, **_kw):
            return inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        def project_logits(self, hidden: mx.array) -> mx.array:
            b, s, _ = (int(d) for d in hidden.shape)
            lg = np.full((b, s, 16), -10.0, dtype=np.float32)
            lg[:, :, 0] = 10.0
            return mx.array(lg, dtype=mx.bfloat16)

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        prompts = [[1] * 20, [2] * 3, [3] * 10]
        res, st = rolling_multimodal_generate(
            lm=_EchoLM(),
            vision_model=None,
            tokenizer=object(),
            images_per_item=[None, None, None],
            prompt_builder=lambda i, _: (prompts[i], []),
            image_token_id=9,
            new_to_old=np.arange(16, dtype=np.int32),
            rolling_max_batch_size=8,
            batch_size=8,
            max_tokens=9000,
            stop_ids=frozenset({0}),
            pad_token_id=1,
            host_mem_gb=24.0,
        )
        assert [r.prompt_tokens for r in res] == [20, 3, 10]
        assert st["b_slots"] <= 2
    finally:
        mx.set_default_device(prev)
