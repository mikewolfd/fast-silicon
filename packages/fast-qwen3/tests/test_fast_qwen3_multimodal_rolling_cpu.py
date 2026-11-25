"""CPU unit tests for A1 GPU/CPU argmax+confidence, B1 batched PLD primitives, and B2 rolling multimodal batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from fast_qwen3.gdn_kernels import gdn_delta_update, quantize_gdn_state_blk32
from fast_qwen3.kernels import gpu_argmax_conf, gpu_argmax_nan, stats_to_logprob_margin
from fast_qwen3.model import DeltaSlotCache, FastQwen3Model, GrowingKVCache, Qwen3Config
from fast_qwen3.mtp_drafter import verify_draft_sequence_b1
from fast_qwen3.multimodal_batch import batched_multimodal_generate
from fast_qwen3.multimodal_rolling import (
    build_ragged_attn_mask,
    rollback_gdn_batched,
    rolling_multimodal_generate,
    scatter_kv_and_rewind,
    scatter_single_item_cache_into_slot,
)
from fast_shared.grammar import safe_argmax_1d


def _np_logprob_margin(row_f32: np.ndarray, idx: int) -> tuple[float, float]:
    chosen = float(row_f32[idx])
    m = float(np.max(row_f32))
    lse = m + float(np.log(max(float(np.sum(np.exp(row_f32 - m))), 1e-30)))
    lp = min(0.0, chosen - lse)
    if row_f32.shape[0] <= 1:
        return lp, 0.0
    other = -1e30
    if idx > 0:
        other = max(other, float(np.max(row_f32[:idx])))
    if idx + 1 < row_f32.shape[0]:
        other = max(other, float(np.max(row_f32[idx + 1 :])))
    return lp, (chosen - other if other > -1e29 else 0.0)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_gpu_argmax_nan_and_conf_parity_across_dtypes_cpu(dtype: mx.Dtype) -> None:
    rng = np.random.default_rng(20261125)
    raw = rng.normal(0.0, 2.5, size=(4, 137)).astype(np.float32)
    # Row 1: exact tie between indices 5 and 80 (must pick lowest index 5)
    raw[1, :] = -3.0
    raw[1, 5] = 4.5
    raw[1, 80] = 4.5
    # Row 2: NaN element (must return -1)
    raw[2, 42] = np.nan

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x = mx.array(raw).astype(dtype)
        x_np = np.asarray(x.astype(mx.float32))

        idx_only = gpu_argmax_nan(x)
        idx_conf, stats = gpu_argmax_conf(x)
        mx.eval(idx_only, idx_conf, stats)

        idx_list = idx_only.tolist()
        assert idx_list == idx_conf.tolist()
        assert idx_list[1] == 5
        assert idx_list[2] == -1

        stats_np = np.asarray(stats)
        for r in (0, 1, 3):
            expected_idx = safe_argmax_1d(x_np[r])
            assert idx_list[r] == expected_idx
            lp, margin = stats_to_logprob_margin(
                float(stats_np[r, 0]),
                float(stats_np[r, 1]),
                float(stats_np[r, 2]),
            )
            ref_lp, ref_margin = _np_logprob_margin(x_np[r], expected_idx)
            assert lp == pytest.approx(ref_lp, abs=1e-4)
            assert margin == pytest.approx(ref_margin, abs=1e-4)

        # Edge cases: empty rows, empty vocab, V=1
        assert gpu_argmax_nan(mx.zeros((0, 8), dtype=dtype)).shape == (0,)
        assert gpu_argmax_nan(mx.zeros((2, 0), dtype=dtype)).tolist() == [-1, -1]
        v1_idx, v1_stats = gpu_argmax_conf(mx.array([[1.25]], dtype=dtype))
        mx.eval(v1_idx, v1_stats)
        assert v1_idx.tolist() == [0]
        lp1, mar1 = stats_to_logprob_margin(
            float(v1_stats[0, 0].item()),
            float(v1_stats[0, 1].item()),
            float(v1_stats[0, 2].item()),
        )
        assert lp1 == pytest.approx(0.0, abs=1e-5)
        assert mar1 == 0.0
    finally:
        mx.set_default_device(prev_dev)


def test_verify_draft_sequence_b1_gpu_fn_matches_cpu_fn() -> None:
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=24,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.5,
    )
    rng = np.random.default_rng(77)

    def _w(shape: tuple[int, ...]) -> mx.array:
        return mx.array(rng.normal(0.0, 0.08, size=shape).astype(np.float32), dtype=mx.bfloat16)

    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": _w((cfg.vocab_size, cfg.hidden_size)),
        "model.norm.weight": mx.ones((cfg.hidden_size,), dtype=mx.bfloat16),
    }
    for li in range(cfg.num_hidden_layers):
        p = f"model.layers.{li}"
        weights[f"{p}.input_layernorm.weight"] = mx.ones((cfg.hidden_size,), dtype=mx.bfloat16)
        weights[f"{p}.post_attention_layernorm.weight"] = mx.ones((cfg.hidden_size,), dtype=mx.bfloat16)
        weights[f"{p}.mlp.gate_proj.weight"] = _w((cfg.intermediate_size, cfg.hidden_size))
        weights[f"{p}.mlp.up_proj.weight"] = _w((cfg.intermediate_size, cfg.hidden_size))
        weights[f"{p}.mlp.down_proj.weight"] = _w((cfg.hidden_size, cfg.intermediate_size))
        if cfg.layer_types[li] == "full_attention":
            weights[f"{p}.self_attn.q_proj.weight"] = _w((cfg.num_attention_heads * 2 * cfg.head_dim, cfg.hidden_size))
            weights[f"{p}.self_attn.k_proj.weight"] = _w((cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size))
            weights[f"{p}.self_attn.v_proj.weight"] = _w((cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size))
            weights[f"{p}.self_attn.o_proj.weight"] = _w((cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim))
            weights[f"{p}.self_attn.q_norm.weight"] = mx.ones((cfg.head_dim,), dtype=mx.bfloat16)
            weights[f"{p}.self_attn.k_norm.weight"] = mx.ones((cfg.head_dim,), dtype=mx.bfloat16)
        else:
            qkv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 + cfg.linear_num_value_heads * cfg.linear_value_head_dim
            z_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            weights[f"{p}.linear_attn.in_proj_qkv.weight"] = _w((qkv_dim, cfg.hidden_size))
            weights[f"{p}.linear_attn.in_proj_z.weight"] = _w((z_dim, cfg.hidden_size))
            weights[f"{p}.linear_attn.in_proj_b.weight"] = _w((cfg.linear_num_value_heads, cfg.hidden_size))
            weights[f"{p}.linear_attn.in_proj_a.weight"] = _w((cfg.linear_num_value_heads, cfg.hidden_size))
            weights[f"{p}.linear_attn.conv1d.weight"] = _w((qkv_dim, 1, cfg.linear_conv_kernel_dim))
            weights[f"{p}.linear_attn.A_log"] = mx.zeros((cfg.linear_num_value_heads,), dtype=mx.float32)
            weights[f"{p}.linear_attn.dt_bias"] = mx.ones((cfg.linear_num_value_heads,), dtype=mx.bfloat16)
            weights[f"{p}.linear_attn.norm.weight"] = mx.ones((cfg.linear_value_head_dim,), dtype=mx.bfloat16)
            weights[f"{p}.linear_attn.out_proj.weight"] = _w((cfg.hidden_size, z_dim))

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = FastQwen3Model(cfg, weights, enable_fusion=False)
        prompt = mx.array([[1, 5, 9, 13]], dtype=mx.int32)
        c_cpu = model.make_cache()
        c_gpu = model.make_cache()
        mx.eval(model.forward_hidden(prompt, c_cpu), model.forward_hidden(prompt, c_gpu))

        drafts = [2, 4, 6]
        seen_cpu_stats: list[tuple[float, float]] = []
        seen_gpu_stats: list[tuple[float, float]] = []

        def _pick_cpu(_i: int, row_np: np.ndarray) -> int:
            idx = safe_argmax_1d(row_np)
            seen_cpu_stats.append(_np_logprob_margin(row_np, idx))
            return idx

        def _pick_gpu(
            _i: int,
            top1_idx: int,
            stats: tuple[float, float, float] | None,
            _lazy_fn: Any,
        ) -> int:
            assert stats is not None and top1_idx >= 0
            seen_gpu_stats.append(stats_to_logprob_margin(stats[0], stats[1], stats[2]))
            return int(top1_idx)

        em_cpu, _, m_cpu = verify_draft_sequence_b1(
            model, c_cpu, 3, drafts, pick_token_fn=_pick_cpu, stop_token_ids=frozenset({0})
        )
        em_gpu, _, m_gpu = verify_draft_sequence_b1(
            model,
            c_gpu,
            3,
            drafts,
            pick_token_gpu_fn=_pick_gpu,
            need_conf=True,
            stop_token_ids=frozenset({0}),
        )
        assert em_cpu == em_gpu
        assert m_cpu == m_gpu
        assert len(seen_cpu_stats) == len(seen_gpu_stats)
        for (lp_c, ma_c), (lp_g, ma_g) in zip(seen_cpu_stats, seen_gpu_stats):
            assert lp_g == pytest.approx(lp_c, abs=1e-4)
            assert ma_g == pytest.approx(ma_c, abs=1e-4)

        # Compare next-step hidden state parity
        nxt = mx.array([[em_cpu[-1]]], dtype=mx.int32)
        h_cpu = model.forward_hidden(nxt, c_cpu)
        h_gpu = model.forward_hidden(nxt, c_gpu)
        mx.eval(h_cpu, h_gpu)
        assert np.allclose(np.asarray(h_cpu.astype(mx.float32)), np.asarray(h_gpu.astype(mx.float32)), atol=1e-4)
    finally:
        mx.set_default_device(prev_dev)


@pytest.mark.parametrize("kv_mode", ["bf16", "q8", "q4_quarot"])
def test_scatter_kv_and_slot_admission_all_modes_cpu(kv_mode: str) -> None:
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        b_slots = 3
        h_kv = 2
        d_head = 64
        rng = np.random.default_rng(123)

        # Build B=1 prefilled KV + GDN item_cache and scatter into slot 1
        ac_kv = GrowingKVCache(step=16, kv_i8=kv_mode)
        ic_kv = GrowingKVCache(step=16, kv_i8=kv_mode)
        k_item = mx.array(rng.normal(0.0, 0.5, size=(1, h_kv, 4, d_head)).astype(np.float32), dtype=mx.bfloat16)
        v_item = mx.array(rng.normal(0.0, 0.5, size=(1, h_kv, 4, d_head)).astype(np.float32), dtype=mx.bfloat16)
        ic_kv.update_and_fetch(k_item, v_item)

        # Fill slots 0 and 2 via B=3 batch write of length 3
        k_b3 = mx.array(rng.normal(0.0, 0.5, size=(b_slots, h_kv, 3, d_head)).astype(np.float32), dtype=mx.bfloat16)
        v_b3 = mx.array(rng.normal(0.0, 0.5, size=(b_slots, h_kv, 3, d_head)).astype(np.float32), dtype=mx.bfloat16)
        ac_kv.update_and_fetch(k_b3, v_b3)

        slot_kv_len = [3, 0, 3]
        ac_gdn = DeltaSlotCache(state_mode="fp32")
        ic_gdn = DeltaSlotCache(
            conv_state=mx.array(rng.normal(size=(1, 3, 16)).astype(np.float32), dtype=mx.bfloat16),
            rec_state=mx.array(rng.normal(size=(1, 2, 8, 32)).astype(np.float32), dtype=mx.float32),
            state_mode="fp32",
        )
        scatter_single_item_cache_into_slot(
            [ac_gdn, ac_kv],
            [ic_gdn, ic_kv],
            slot_r=1,
            b_slots=b_slots,
            slot_kv_len=slot_kv_len,
            new_len=4,
            kv_step=16,
        )
        assert slot_kv_len == [3, 4, 3]
        assert ac_kv.offset == 4

        # Verify ragged decode step + scatter_kv_and_rewind with asymmetric accepted counts [2, 1, 3]
        kv_prev = max(slot_kv_len)
        l_q = 3
        k_step = mx.array(rng.normal(0.0, 0.5, size=(b_slots, h_kv, l_q, d_head)).astype(np.float32), dtype=mx.bfloat16)
        v_step = mx.array(rng.normal(0.0, 0.5, size=(b_slots, h_kv, l_q, d_head)).astype(np.float32), dtype=mx.bfloat16)
        ac_kv.update_and_fetch(k_step, v_step)
        updated_lens = scatter_kv_and_rewind([ac_kv], slot_kv_len, [2, 1, 3], kv_prev, l_q, b_slots)
        assert updated_lens == [5, 5, 6]
        assert ac_kv.offset == 6

        # Mask must remain additive BF16 (never bool)
        mask = build_ragged_attn_mask(updated_lens, max(updated_lens), 2)
        assert isinstance(mask, mx.array) and mask.dtype == mx.bfloat16
    finally:
        mx.set_default_device(prev_dev)


@pytest.mark.parametrize("state_mode", ["fp32", "int8_blk32"])
def test_rollback_gdn_batched_parity_cpu(state_mode: str) -> None:
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        b, conv_k, l_q = 2, 4, 4
        hk, hv, dk, dv = 2, 4, 32, 8
        rng = np.random.default_rng(99)

        conv_in = mx.array(rng.normal(0.0, 0.2, size=(b, (conv_k - 1) + l_q, hk * dk * 2 + hv * dv)).astype(np.float32), dtype=mx.bfloat16)
        q = mx.array(rng.normal(0.0, 0.2, size=(b, l_q, hk, dk)).astype(np.float32), dtype=mx.bfloat16)
        k = mx.array(rng.normal(0.0, 0.2, size=(b, l_q, hk, dk)).astype(np.float32), dtype=mx.bfloat16)
        v = mx.array(rng.normal(0.0, 0.2, size=(b, l_q, hv, dv)).astype(np.float32), dtype=mx.bfloat16)
        g = mx.array(-rng.uniform(0.01, 0.2, size=(b, l_q, hv)).astype(np.float32), dtype=mx.float32)
        beta = mx.array(rng.uniform(0.1, 0.9, size=(b, l_q, hv)).astype(np.float32), dtype=mx.bfloat16)
        init_rec = mx.array(rng.normal(0.0, 0.1, size=(b, hv, dv, dk)).astype(np.float32), dtype=mx.float32)
        prev_rec: Any = quantize_gdn_state_blk32(init_rec) if state_mode == "int8_blk32" else init_rec

        entry = DeltaSlotCache(state_mode=state_mode)
        entry.record_verify_trace = True
        entry._verify_trace = (conv_in, q, k, v, g, beta, prev_rec)

        n_acc = [1, 3]
        rollback_gdn_batched([entry], n_acc, l_q, conv_k, b)
        assert entry._verify_trace is None and entry.record_verify_trace is False
        assert entry.conv_state is not None

        for r, na in enumerate(n_acc):
            ref_conv = np.asarray(conv_in[r : r + 1, na : na + conv_k - 1, :].astype(mx.float32))
            got_conv = np.asarray(entry.conv_state[r : r + 1].astype(mx.float32))
            assert np.allclose(got_conv, ref_conv, atol=1e-4)

            p_r = (prev_rec[0][r : r + 1], prev_rec[1][r : r + 1]) if isinstance(prev_rec, tuple) else prev_rec[r : r + 1]
            _, ref_rec_r = gdn_delta_update(
                q[r : r + 1, :na],
                k[r : r + 1, :na],
                v[r : r + 1, :na],
                g[r : r + 1, :na],
                beta[r : r + 1, :na],
                p_r,
                state_mode=state_mode,
            )
            got_rec = entry.rec_state
            assert got_rec is not None
            ref_rec_full = ref_rec_r[0] if not isinstance(ref_rec_r, tuple) else ref_rec_r
            if isinstance(ref_rec_full, tuple):
                from fast_qwen3.gdn_kernels import dequantize_gdn_state_int8

                ref_rec_dense = dequantize_gdn_state_int8(ref_rec_full[0], ref_rec_full[1], dtype=mx.float32)
            else:
                ref_rec_dense = ref_rec_full
            assert np.allclose(
                np.asarray(got_rec[r : r + 1].astype(mx.float32)),
                np.asarray(ref_rec_dense.astype(mx.float32)),
                atol=1e-4,
            )
    finally:
        mx.set_default_device(prev_dev)


def test_rolling_multimodal_generate_pld_and_ar_parity_cpu() -> None:
    """Verify rolling B_slots=3 (AR & PLD) matches independent B=1 runs on a deterministic stateful mock LM."""
    vocab_sz = 32
    eos_id = 0
    pad_id = 1
    img_tok = 2
    # Item 0 repeats [10, 11, 12] -> high PLD acceptance
    # Item 1 finishes quickly in 2 tokens -> triggers slot replacement
    # Items 2..4 vary prompt and output lengths -> triggers tail compaction
    prompt_marker_to_seq: dict[int, list[int]] = {
        20: [10, 11, 12, 10, 11, 12, 10, 11, eos_id],
        21: [14, eos_id],
        22: [15, 16, 17, 18, eos_id],
        23: [10, 11, 12, 10, 11, 12, eos_id],
        24: [19, 25, eos_id],
    }

    @dataclass
    class _Cfg:
        hidden_size: int = 8
        rot_dims: int = 4
        rope_theta: float = 10000.0
        mrope_section: tuple[int, int, int] = (1, 1, 1)
        linear_key_head_dim: int = 32
        linear_conv_kernel_dim: int = 4

    class _MockStatefulLM:
        def __init__(self) -> None:
            self.cfg = _Cfg()
            self.vocab_n = vocab_sz
            self.vocab_remap = None

        def make_cache(self, *, kv_step: int = 64, kv_i8: Any = "bf16", gdn_state_mode: str = "fp32", batch_size: int = 1) -> list[Any]:
            return [DeltaSlotCache(state_mode="fp32"), GrowingKVCache(step=kv_step, kv_i8="bf16")]

        def embed_tokens(self, ids: mx.array) -> mx.array:
            b, s = int(ids.shape[0]), int(ids.shape[1])
            out = np.zeros((b, s, self.cfg.hidden_size), dtype=np.float32)
            ids_np = np.asarray(ids)
            out[:, :, 0] = ids_np.astype(np.float32)
            return mx.array(out, dtype=mx.bfloat16)

        def precompute_rope_trig(self, pos: mx.array) -> tuple[mx.array, mx.array]:
            b, s = int(pos.shape[0]), int(pos.shape[1])
            return mx.ones((b, 1, s, 2), dtype=mx.bfloat16), mx.zeros((b, 1, s, 2), dtype=mx.bfloat16)

        def forward_hidden(
            self,
            input_ids: mx.array | None,
            cache: list[Any],
            *,
            inputs_embeds: mx.array | None = None,
            valid_token_scale: mx.array | None = None,
            attn_mask: Any = None,
            rope_trig: Any = None,
            **_kw: Any,
        ) -> mx.array:
            emb = self.embed_tokens(input_ids) if input_ids is not None else inputs_embeds
            assert emb is not None
            b, s, h = int(emb.shape[0]), int(emb.shape[1]), int(emb.shape[2])
            emb_np = np.asarray(emb.astype(mx.float32))
            gdn, kv = cache[0], cache[1]

            if gdn._rec_state is None:
                prev_marker = np.zeros((b,), dtype=np.float32)
                prev_step = np.zeros((b,), dtype=np.float32)
            else:
                st_np = np.asarray(gdn._rec_state.astype(mx.float32))
                prev_marker = st_np[:, 0, 0, 0].copy()
                prev_step = st_np[:, 0, 1, 0].copy()

            out = np.zeros((b, s, h), dtype=np.float32)
            v_trace_np = np.zeros((b, s, 1, 4), dtype=np.float32)
            for r in range(b):
                for t in range(s):
                    is_valid = True if valid_token_scale is None else (float(valid_token_scale[r, t, 0].item()) > 0.5)
                    if is_valid:
                        tok_v = int(round(float(emb_np[r, t, 0])))
                        if tok_v in prompt_marker_to_seq:
                            prev_marker[r] = float(tok_v)
                            prev_step[r] = 0.0
                        else:
                            prev_step[r] += 1.0
                    out[r, t, 0] = prev_marker[r]
                    out[r, t, 1] = prev_step[r]
                    v_trace_np[r, t, 0, 0] = prev_marker[r]
                    v_trace_np[r, t, 0, 1] = prev_step[r]

            gdn.conv_state = mx.zeros((b, 3, 4), dtype=mx.bfloat16)
            if gdn.record_verify_trace:
                conv_in = mx.zeros((b, 3 + s, 4), dtype=mx.bfloat16)
                k_np = np.zeros((b, s, 1, 32), dtype=np.float32)
                k_np[:, :, 0, 0] = 1.0
                q_t = mx.zeros((b, s, 1, 32), dtype=mx.bfloat16)
                k_t = mx.array(k_np, dtype=mx.bfloat16)
                v_t = mx.array(v_trace_np, dtype=mx.bfloat16)
                g_t = mx.ones((b, s, 1), dtype=mx.float32)
                b_t = mx.ones((b, s, 1), dtype=mx.bfloat16)
                gdn._verify_trace = (conv_in, q_t, k_t, v_t, g_t, b_t, gdn._raw_rec_snapshot())

            new_rec = np.zeros((b, 1, 4, 32), dtype=np.float32)
            new_rec[:, 0, 0, 0] = prev_marker
            new_rec[:, 0, 1, 0] = prev_step
            gdn.rec_state = mx.array(new_rec, dtype=mx.float32)
            kv.update_and_fetch(mx.zeros((b, 1, s, 32), dtype=mx.bfloat16), mx.zeros((b, 1, s, 32), dtype=mx.bfloat16))
            return mx.array(out, dtype=mx.bfloat16)

        def project_logits(self, hidden: mx.array) -> mx.array:
            b, s, _ = int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2])
            h_np = np.asarray(hidden.astype(mx.float32))
            logits = np.full((b, s, vocab_sz), -20.0, dtype=np.float32)
            for r in range(b):
                for t in range(s):
                    marker = int(round(float(h_np[r, t, 0])))
                    step = int(round(float(h_np[r, t, 1])))
                    seq = prompt_marker_to_seq.get(marker, [eos_id])
                    nxt = seq[step] if 0 <= step < len(seq) else eos_id
                    logits[r, t, nxt] = 20.0
            return mx.array(logits, dtype=mx.bfloat16)

    class _SimpleSelfDrafter:
        def draft(self, history_ids: list[int], *, mode: str = "self", max_k: int = 3) -> list[int]:
            n = len(history_ids)
            if n < 3:
                return []
            needle = tuple(history_ids[-2:])
            for pos in range(n - 3, -1, -1):
                if tuple(history_ids[pos : pos + 2]) == needle:
                    return list(history_ids[pos + 2 : pos + 2 + max_k])
            return []

    class _MockTok:
        vocab_size = vocab_sz

    prompts = [
        [3, 4, 20],
        [21],
        [5, 6, 7, 22],
        [8, 23],
        [9, 10, 24],
    ]

    def _pb(idx: int, _vc: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        return prompts[idx], []

    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        lm = _MockStatefulLM()
        # 1. Rolling AR (use_pld=False)
        res_ar, st_ar = rolling_multimodal_generate(
            lm=lm,
            vision_model=None,
            tokenizer=_MockTok(),
            images_per_item=[None] * len(prompts),
            prompt_builder=_pb,
            image_token_id=img_tok,
            new_to_old=np.arange(vocab_sz, dtype=np.int32),
            drafters=None,
            use_pld=False,
            max_tokens=32,
            batch_size=8,
            rolling_max_batch_size=3,
            kv_step=32,
            kv_i8_resolver="bf16",
            gdn_state_mode="fp32",
            stop_ids=frozenset({eos_id}),
            pad_token_id=pad_id,
        )
        # 2. Rolling PLD via batched_multimodal_generate(..., rolling="auto")
        res_pld, st_pld = batched_multimodal_generate(
            lm=lm,
            vision_model=None,
            tokenizer=_MockTok(),
            images_per_item=[None] * len(prompts),
            prompt_builder=_pb,
            image_token_id=img_tok,
            new_to_old=np.arange(vocab_sz, dtype=np.int32),
            old_to_new=None,
            grammars=[None] * len(prompts),
            drafters=[_SimpleSelfDrafter() for _ in prompts],
            spec_modes=["self"] * len(prompts),
            draft_k=3,
            rolling="auto",
            rolling_max_batch_size=3,
            max_tokens=32,
            batch_size=8,
            kv_step=32,
            stop_ids=frozenset({eos_id}),
            pad_token_id=pad_id,
        )

        expected = [[t for t in prompt_marker_to_seq[m] if t != eos_id] for m in (20, 21, 22, 23, 24)]
        assert [r.generated_ids for r in res_ar] == expected
        assert [r.generated_ids for r in res_pld] == expected
        assert st_ar["rolling"] is True and st_ar["use_pld"] is False and st_ar["replacements"] == 2
        assert st_pld["rolling"] is True and st_pld["use_pld"] is True and st_pld["replacements"] == 2
        assert st_pld["accepted_draft_tokens"] > 0
        assert st_pld["total_decode_steps"] < st_ar["total_decode_steps"]
    finally:
        mx.set_default_device(prev_dev)
