"""CPU unit tests for `GdnDecodeShared`, live `BF16`/`INT8 block-32` `DeltaSlotCache`, and cache roundtrips."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

import fast_qwen3.gdn_kernels as gdn_kernels
from fast_qwen3.gdn_kernels import (
    gated_delta_update_fallback,
    gdn_delta_update,
    normalize_gdn_state_mode,
    quantize_gdn_state_blk32,
    resolve_gdn_state_mode,
)
from fast_qwen3.kernels import dequantize_gdn_state_int8
from fast_qwen3.model import (
    DeltaSlotCache,
    FastQwen3Model,
    GrowingKVCache,
    Qwen3Config,
    compact_hybrid_cache,
    restore_hybrid_cache,
    snapshot_hybrid_cache,
)
from fast_qwen3.mtp_drafter import verify_draft_sequence_b1


def _tiny_qwen3_cfg(*, dk: int = 32, dv: int = 32) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=320,
        tie_word_embeddings=True,
        full_attention_interval=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=dk,
        linear_value_head_dim=dv,
        linear_conv_kernel_dim=4,
    )


def _tiny_qwen3_weights(cfg: Qwen3Config, *, seed: int = 13) -> dict[str, mx.array]:
    mx.random.seed(seed)
    h = cfg.hidden_size
    v = cfg.vocab_size
    inter = cfg.intermediate_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    conv_dim = 2 * k_dim + v_dim

    def rw(out_d: int, in_d: int) -> mx.array:
        return (mx.random.normal((out_d, in_d)) * 0.04).astype(mx.bfloat16)

    def rn(dim: int) -> mx.array:
        return (mx.ones((dim,), dtype=mx.float32) + mx.random.normal((dim,)) * 0.02).astype(mx.bfloat16)

    w: dict[str, mx.array] = {
        "model.embed_tokens.weight": rw(v, h),
        "model.norm.weight": rn(h),
    }
    for i in range(cfg.num_hidden_layers):
        px = f"model.layers.{i}"
        w[f"{px}.input_layernorm.weight"] = rn(h)
        w[f"{px}.post_attention_layernorm.weight"] = rn(h)
        if cfg.layer_types[i] == "linear_attention":
            gpx = f"{px}.linear_attn"
            w[f"{gpx}.in_proj_qkv.weight"] = rw(conv_dim, h)
            w[f"{gpx}.in_proj_z.weight"] = rw(v_dim, h)
            w[f"{gpx}.in_proj_b.weight"] = rw(cfg.linear_num_value_heads, h)
            w[f"{gpx}.in_proj_a.weight"] = rw(cfg.linear_num_value_heads, h)
            w[f"{gpx}.out_proj.weight"] = rw(h, v_dim)
            w[f"{gpx}.conv1d.weight"] = (
                mx.random.normal((conv_dim, 1, cfg.linear_conv_kernel_dim)) * 0.05
            ).astype(mx.bfloat16)
            w[f"{gpx}.dt_bias"] = mx.ones((cfg.linear_num_value_heads,), dtype=mx.bfloat16) * 0.5
            w[f"{gpx}.A_log"] = mx.zeros((cfg.linear_num_value_heads,), dtype=mx.bfloat16)
            w[f"{gpx}.norm.weight"] = mx.ones((cfg.linear_value_head_dim,), dtype=mx.bfloat16)
        else:
            apx = f"{px}.self_attn"
            w[f"{apx}.q_proj.weight"] = rw(2 * q_dim, h)
            w[f"{apx}.k_proj.weight"] = rw(kv_dim, h)
            w[f"{apx}.v_proj.weight"] = rw(kv_dim, h)
            w[f"{apx}.o_proj.weight"] = rw(h, q_dim)
            w[f"{apx}.q_norm.weight"] = rn(cfg.head_dim)
            w[f"{apx}.k_norm.weight"] = rn(cfg.head_dim)
        mpx = f"{px}.mlp"
        w[f"{mpx}.gate_proj.weight"] = rw(inter, h)
        w[f"{mpx}.up_proj.weight"] = rw(inter, h)
        w[f"{mpx}.down_proj.weight"] = rw(h, inter)
    mx.eval(*w.values())
    return w


def test_resolve_gdn_state_mode_matrix_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAST_QWEN3_GDN_STATE_MODE", raising=False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)

    # 48+ GB host: batch >= 8 -> int8_blk32, batch < 8 -> fp32
    monkeypatch.setattr(gdn_kernels, "_host_phys_memory_gb", lambda: 48.0)
    assert resolve_gdn_state_mode("auto", batch_size=1, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=4, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=7, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=8, head_k_dim=128) == "int8_blk32"

    # <= 28 GB host (24 GB MacBook): batch >= 4 -> int8_blk32, batch < 4 -> fp32
    monkeypatch.setattr(gdn_kernels, "_host_phys_memory_gb", lambda: 24.0)
    assert resolve_gdn_state_mode("auto", batch_size=1, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=3, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=4, head_k_dim=128) == "int8_blk32"
    assert resolve_gdn_state_mode("auto", batch_size=8, head_k_dim=128) == "int8_blk32"

    # Non-32 head_k_dim cannot use int8_blk32 -> auto returns fp32, explicit int8_blk32 downgrades cleanly
    assert resolve_gdn_state_mode("auto", batch_size=16, head_k_dim=48) == "fp32"
    assert resolve_gdn_state_mode("int8_blk32", batch_size=16, head_k_dim=48) == "bf16"

    # Default os.sysconf path executes without error when host_gb is omitted
    monkeypatch.setattr(os, "sysconf", lambda name: 16384 if name == "SC_PAGE_SIZE" else 1572864)
    assert resolve_gdn_state_mode("auto", batch_size=4, head_k_dim=128) == "int8_blk32"

    # Metal unavailable -> auto falls back to fp32
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert resolve_gdn_state_mode("auto", batch_size=1, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("auto", batch_size=16, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("int8_blk32", batch_size=16, head_k_dim=48) == "fp32"

    # Env override applies to "auto" / None, while explicit non-auto mode takes precedence
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("FAST_QWEN3_GDN_STATE_MODE", "int8_blk32")
    assert resolve_gdn_state_mode("auto", batch_size=1, head_k_dim=128) == "int8_blk32"
    assert resolve_gdn_state_mode(None, batch_size=1, head_k_dim=128) == "int8_blk32"
    assert resolve_gdn_state_mode("fp32", batch_size=16, head_k_dim=128) == "fp32"
    assert resolve_gdn_state_mode("bf16", batch_size=16, head_k_dim=128) == "bf16"

    monkeypatch.setenv("FAST_QWEN3_GDN_STATE_MODE", "fp32")
    assert resolve_gdn_state_mode("auto", batch_size=16, head_k_dim=128) == "fp32"

    assert normalize_gdn_state_mode("FP32") == "fp32"
    assert normalize_gdn_state_mode("bf16") == "bf16"
    assert normalize_gdn_state_mode("int8_blk32") == "int8_blk32"
    with pytest.raises(ValueError, match="Unsupported GDN state mode"):
        resolve_gdn_state_mode("fp8_e4m3")


def test_gdn_fallback_and_kernel_fp32_g_int8_blk32_and_nan_guard() -> None:
    rng = np.random.default_rng(2026)
    b_sz, t_len, hk, hv, dk, dv = 2, 19, 2, 4, 64, 32
    q_np = rng.normal(0.0, 0.15, (b_sz, t_len, hk, dk)).astype(np.float32)
    k_np = rng.normal(0.0, 0.15, (b_sz, t_len, hk, dk)).astype(np.float32)
    v_np = rng.normal(0.0, 0.25, (b_sz, t_len, hv, dv)).astype(np.float32)
    # Near-1.0 decay values specifically distinguish FP32 g from BF16-truncated g
    g_np = np.clip(0.996 + rng.normal(0.0, 0.0015, (b_sz, t_len, hv)), 0.85, 0.9999).astype(np.float32)
    beta_np = (1.0 / (1.0 + np.exp(-rng.normal(-0.2, 0.3, (b_sz, t_len, hv))))).astype(np.float32)

    # 1. Pure-CPU fallback (mx.cpu) + non-32 Dk fallback
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        q_bf = mx.array(q_np, dtype=mx.bfloat16)
        k_bf = mx.array(k_np, dtype=mx.bfloat16)
        v_bf = mx.array(v_np, dtype=mx.bfloat16)
        g_f32 = mx.array(g_np, dtype=mx.float32)
        beta_bf = mx.array(beta_np, dtype=mx.bfloat16)

        y_f32, st_f32 = gdn_delta_update(q_bf, k_bf, v_bf, g_f32, beta_bf, None, state_mode="fp32")
        y_bf, st_bf = gdn_delta_update(q_bf, k_bf, v_bf, g_f32, beta_bf, None, state_mode="bf16")
        y_i8_a, st_i8_a = gdn_delta_update(
            q_bf[:, :10], k_bf[:, :10], v_bf[:, :10], g_f32[:, :10], beta_bf[:, :10], None, state_mode="int8_blk32"
        )
        y_i8_b, st_i8_b = gdn_delta_update(
            q_bf[:, 10:], k_bf[:, 10:], v_bf[:, 10:], g_f32[:, 10:], beta_bf[:, 10:], st_i8_a, state_mode="int8_blk32"
        )
        y_i8 = mx.concatenate([y_i8_a, y_i8_b], axis=1)
        mx.eval(y_f32, st_f32, y_bf, st_bf, y_i8, st_i8_b[0], st_i8_b[1])

        assert isinstance(st_f32, mx.array) and st_f32.dtype == mx.float32
        assert isinstance(st_bf, mx.array) and st_bf.dtype == mx.bfloat16
        assert isinstance(st_i8_b, tuple)
        assert st_i8_b[0].dtype == mx.int8 and st_i8_b[1].dtype == mx.bfloat16
        assert st_i8_b[0].shape == (b_sz, hv, dv, dk)
        assert st_i8_b[1].shape == (b_sz, hv, dv, dk // 32)
        # Verify 3.76x DRAM compression vs FP32
        assert (st_i8_b[0].nbytes + st_i8_b[1].nbytes) * 3.7 < st_f32.nbytes

        yf_r = np.asarray(y_f32.astype(mx.float32)).ravel()
        yi_r = np.asarray(y_i8.astype(mx.float32)).ravel()
        yb_r = np.asarray(y_bf.astype(mx.float32)).ravel()
        assert float(np.dot(yf_r, yi_r) / (np.linalg.norm(yf_r) * np.linalg.norm(yi_r))) >= 0.9995
        assert float(np.dot(yf_r, yb_r) / (np.linalg.norm(yf_r) * np.linalg.norm(yb_r))) >= 0.9995

        # Non-32 Dk exercises pure-MLX fallback cleanly
        q24 = mx.array(q_np[:, :4, :, :24], dtype=mx.bfloat16)
        k24 = mx.array(k_np[:, :4, :, :24], dtype=mx.bfloat16)
        y24, s24 = gdn_delta_update(q24, k24, v_bf[:, :4], g_f32[:, :4], beta_bf[:, :4], None, state_mode="bf16")
        mx.eval(y24, s24)
        assert y24.shape == (b_sz, 4, hv, dv) and isinstance(s24, mx.array) and s24.shape == (b_sz, hv, dv, 24)

        # Empty T=0 cleanly zero-initializes both dense & int8_blk32 states
        y0, s0_i8 = gdn_delta_update(
            q_bf[:, :0], k_bf[:, :0], v_bf[:, :0], g_f32[:, :0], beta_bf[:, :0], None, state_mode="int8_blk32"
        )
        mx.eval(y0, s0_i8[0], s0_i8[1])
        assert y0.shape == (b_sz, 0, hv, dv)
        assert isinstance(s0_i8, tuple) and s0_i8[0].shape == (b_sz, hv, dv, dk) and s0_i8[1].shape == (b_sz, hv, dv, dk // 32)
    finally:
        mx.set_default_device(prev_dev)

    # 2. Metal GdnDecodeShared path (TILE_ROWS=16 for T>1 and TILE_ROWS=1 for T=1) + 0x7f80u NaN guard
    if mx.metal.is_available() and mx.default_device().type == mx.DeviceType.gpu:
        q_bf = mx.array(q_np, dtype=mx.bfloat16)
        k_bf = mx.array(k_np, dtype=mx.bfloat16)
        v_bf = mx.array(v_np, dtype=mx.bfloat16)
        g_f32 = mx.array(g_np, dtype=mx.float32)
        beta_bf = mx.array(beta_np, dtype=mx.bfloat16)

        y_ref, st_ref = gated_delta_update_fallback(
            q_bf, k_bf, v_bf, g_f32, beta_bf, None, state_mode="fp32"
        )
        # Prefill chunk T>1 (TILE_ROWS=16) + 1-token decode step T=1 (TILE_ROWS=1) in bf16 and int8_blk32
        y_m_p, st_m_p = gdn_delta_update(
            q_bf[:, :-1], k_bf[:, :-1], v_bf[:, :-1], g_f32[:, :-1], beta_bf[:, :-1], None, state_mode="bf16"
        )
        y_m_d, st_m_d = gdn_delta_update(
            q_bf[:, -1:], k_bf[:, -1:], v_bf[:, -1:], g_f32[:, -1:], beta_bf[:, -1:], st_m_p, state_mode="bf16"
        )
        y_m = mx.concatenate([y_m_p, y_m_d], axis=1)

        y_i_p, st_i_p = gdn_delta_update(
            q_bf[:, :-1], k_bf[:, :-1], v_bf[:, :-1], g_f32[:, :-1], beta_bf[:, :-1], None, state_mode="int8_blk32"
        )
        y_i_d, st_i_d = gdn_delta_update(
            q_bf[:, -1:], k_bf[:, -1:], v_bf[:, -1:], g_f32[:, -1:], beta_bf[:, -1:], st_i_p, state_mode="int8_blk32"
        )
        y_i = mx.concatenate([y_i_p, y_i_d], axis=1)

        # Verify FP32 g is NOT truncated to bfloat16 inside Metal kernel
        y_trunc, _ = gated_delta_update_fallback(
            q_bf, k_bf, v_bf, g_f32.astype(mx.bfloat16), beta_bf, None, state_mode="fp32"
        )
        y_m_fp32, st_m_fp32 = gdn_delta_update(
            q_bf, k_bf, v_bf, g_f32, beta_bf, None, state_mode="fp32"
        )
        mx.eval(y_ref, st_ref, y_m, st_m_d, y_i, st_i_d[0], st_i_d[1], y_trunc, y_m_fp32, st_m_fp32)

        yr = np.asarray(y_ref.astype(mx.float32)).ravel()
        ym_f = np.asarray(y_m_fp32.astype(mx.float32)).ravel()
        yt = np.asarray(y_trunc.astype(mx.float32)).ravel()
        assert float(np.max(np.abs(ym_f - yr))) < float(np.max(np.abs(yt - yr)))
        assert float(np.dot(ym_f, yr) / (np.linalg.norm(ym_f) * np.linalg.norm(yr))) >= 0.99999
        assert float(np.dot(np.asarray(y_m.astype(mx.float32)).ravel(), yr) / (
            np.linalg.norm(np.asarray(y_m.astype(mx.float32)).ravel()) * np.linalg.norm(yr)
        )) >= 0.9995
        assert float(np.dot(np.asarray(y_i.astype(mx.float32)).ravel(), yr) / (
            np.linalg.norm(np.asarray(y_i.astype(mx.float32)).ravel()) * np.linalg.norm(yr)
        )) >= 0.9995

        # 0x7f80u bitwise NaN/Inf isolation in state_i8 Metal kernel
        bad_st = np.zeros((1, hv, dv, dk), dtype=np.float32)
        bad_st[0, 1, 3, 40] = np.nan
        bad_codes, bad_scales = quantize_gdn_state_blk32(mx.array(bad_st))
        y_bad, st_bad = gdn_delta_update(
            q_bf[:1, :1], k_bf[:1, :1], v_bf[:1, :1], g_f32[:1, :1], beta_bf[:1, :1],
            (bad_codes, bad_scales),
            state_mode="int8_blk32",
        )
        mx.eval(y_bad, st_bad[0], st_bad[1])
        sc_u16 = np.asarray(st_bad[1].view(mx.uint16), dtype=np.uint16)
        assert sc_u16[0, 1, 3, 1] == 0x7FC0
        assert np.all(np.isfinite(np.asarray(st_bad[1][:, 0, :, :].astype(mx.float32))))


def test_delta_slot_cache_int8_blk32_compact_and_zero_double_quant_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    st_np = rng.normal(0.0, 0.3, (4, 2, 8, 64)).astype(np.float32)

    # 1. Raw DeltaSlotCache() preserves default state_mode="fp32"
    raw_cache = DeltaSlotCache()
    assert raw_cache.state_mode == "fp32"
    assert raw_cache.rec_state is None

    # 2. DeltaSlotCache(state_mode="int8_blk32") stores rec_codes/rec_scales, lazily dequantizes ONLY on .rec_state read
    i8_cache = DeltaSlotCache(state_mode="int8_blk32")
    i8_cache.conv_state = mx.arange(4, dtype=mx.float32)[:, None, None] * mx.ones((4, 3, 16), dtype=mx.bfloat16)
    i8_cache.rec_state = mx.array(st_np, dtype=mx.float32)
    assert i8_cache._rec_state is None
    assert i8_cache.rec_codes is not None and i8_cache.rec_scales is not None
    assert i8_cache.rec_codes.dtype == mx.int8
    assert i8_cache.rec_scales.dtype == mx.bfloat16
    orig_codes = mx.contiguous(i8_cache.rec_codes)
    orig_scales = mx.contiguous(i8_cache.rec_scales)
    mx.eval(orig_codes, orig_scales)

    # Reading .rec_state lazily dequantizes via dequantize_gdn_state_int8(..., dtype=mx.float32)
    dq_view = i8_cache.rec_state
    dq_ref = dequantize_gdn_state_int8(orig_codes, orig_scales, dtype=mx.float32)
    mx.eval(dq_view, dq_ref)
    assert dq_view is not None and dq_view.dtype == mx.float32
    assert np.array_equal(np.asarray(dq_view), np.asarray(dq_ref))
    assert i8_cache._rec_state is None

    # 3. compact_hybrid_cache slices rec_codes & rec_scales in-place with ZERO dequant/re-quant
    def _forbid(*_a: object, **_kw: object) -> object:
        raise AssertionError("compact/snapshot/restore must NOT quantize or dequantize int8_blk32 state")

    monkeypatch.setattr(gdn_kernels, "quantize_gdn_state_blk32", _forbid)
    kv = GrowingKVCache(step=16, kv_i8="q8")
    kv.update_and_fetch(
        mx.ones((4, 2, 3, 64), dtype=mx.bfloat16),
        mx.ones((4, 2, 3, 64), dtype=mx.bfloat16),
    )
    compact_hybrid_cache([i8_cache, kv], [1, 3])
    assert i8_cache.state_mode == "int8_blk32"
    assert i8_cache._rec_state is None
    assert i8_cache.rec_codes is not None and i8_cache.rec_scales is not None
    assert i8_cache.rec_codes.shape[0] == 2 and i8_cache.rec_scales.shape[0] == 2
    assert np.array_equal(
        np.asarray(i8_cache.rec_codes),
        np.asarray(mx.take(orig_codes, mx.array([1, 3], dtype=mx.int32), axis=0)),
    )
    assert np.array_equal(
        np.asarray(i8_cache.rec_scales.view(mx.uint16)),
        np.asarray(mx.take(orig_scales, mx.array([1, 3], dtype=mx.int32), axis=0).view(mx.uint16)),
    )

    # 4. snapshot_hybrid_cache + restore_hybrid_cache zero-double-quant bit-exact roundtrip
    import fast_qwen3.model as model_mod

    monkeypatch.setattr(model_mod, "quantize_gdn_state_int8", _forbid)
    monkeypatch.setattr(model_mod, "dequantize_gdn_state_int8", _forbid)
    b1_cache = DeltaSlotCache(
        conv_state=mx.ones((1, 3, 16), dtype=mx.bfloat16),
        rec_codes=orig_codes[:1],
        rec_scales=orig_scales[:1],
        state_mode="int8_blk32",
    )
    snap = snapshot_hybrid_cache([10, 11, 12], [b1_cache], compress_gdn_int8=True)
    assert snap.layer_states[0][0] == "gdn_i8"
    assert np.array_equal(np.asarray(snap.layer_states[0][2]), np.asarray(orig_codes[:1]))
    assert np.array_equal(
        np.asarray(snap.layer_states[0][3].view(mx.uint16)),
        np.asarray(orig_scales[:1].view(mx.uint16)),
    )

    restored = restore_hybrid_cache(snap, batch_size=8, gdn_state_mode="auto")
    assert len(restored) == 1
    r_gdn = restored[0]
    assert isinstance(r_gdn, DeltaSlotCache)
    assert r_gdn.state_mode == "int8_blk32"
    assert r_gdn._rec_state is None
    assert r_gdn.rec_codes is not None and r_gdn.rec_scales is not None
    assert r_gdn.rec_codes.shape[0] == 8 and r_gdn.rec_scales.shape[0] == 8
    mx.eval(r_gdn.rec_codes, r_gdn.rec_scales)
    for slot in range(8):
        assert np.array_equal(np.asarray(r_gdn.rec_codes[slot]), np.asarray(orig_codes[0]))
        assert np.array_equal(
            np.asarray(r_gdn.rec_scales[slot].view(mx.uint16)),
            np.asarray(orig_scales[0].view(mx.uint16)),
        )

    # Restoring gdn_i8 into dense bf16 dequantizes cleanly once
    monkeypatch.undo()
    restored_bf = restore_hybrid_cache(snap, batch_size=2, gdn_state_mode="bf16")
    rb_gdn = restored_bf[0]
    assert isinstance(rb_gdn, DeltaSlotCache)
    assert rb_gdn.state_mode == "bf16"
    assert rb_gdn._rec_state is not None and rb_gdn._rec_state.dtype == mx.bfloat16
    assert rb_gdn.rec_codes is None and rb_gdn.rec_scales is None

    # Setting .rec_state = None clears both dense and quantized storage
    i8_cache.rec_state = None
    assert i8_cache.rec_codes is None and i8_cache.rec_scales is None and i8_cache.rec_state is None


def test_make_cache_api_compat_and_gdn_modes_with_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAST_QWEN3_GDN_STATE_MODE", raising=False)
    monkeypatch.delenv("FAST_QWEN3_USE_GDN_METAL", raising=False)

    cfg = _tiny_qwen3_cfg(dk=32, dv=32)
    weights = _tiny_qwen3_weights(cfg)
    model = FastQwen3Model(cfg, weights)

    # Backward-compatible call signatures without gdn_state_mode / batch_size -> fp32 at B=1
    c_default = model.make_cache()
    c_legacy = model.make_cache(kv_step=128, kv_i8="q8")
    expected_b1 = "fp32"
    assert all(e.state_mode == expected_b1 for e in c_default if isinstance(e, DeltaSlotCache))
    assert all(e.state_mode == expected_b1 for e in c_legacy if isinstance(e, DeltaSlotCache))

    # Explicit gdn_state_mode & auto batch_size=8 -> int8_blk32
    c_fp32 = model.make_cache(gdn_state_mode="fp32")
    c_bf16 = model.make_cache(gdn_state_mode="bf16")
    c_i8 = model.make_cache(gdn_state_mode="int8_blk32")
    c_auto8 = model.make_cache(gdn_state_mode="auto", batch_size=8)
    assert all(e.state_mode == "fp32" for e in c_fp32 if isinstance(e, DeltaSlotCache))
    assert all(e.state_mode == "bf16" for e in c_bf16 if isinstance(e, DeltaSlotCache))
    assert all(e.state_mode == "int8_blk32" for e in c_i8 if isinstance(e, DeltaSlotCache))
    if mx.metal.is_available():
        assert all(e.state_mode == "int8_blk32" for e in c_auto8 if isinstance(e, DeltaSlotCache))

    # End-to-end forward + speculative rollback under fp32, bf16, and int8_blk32
    prompt = mx.array([[3, 7, 11, 15, 19, 23]], dtype=mx.int32)
    logits_fp32 = model(prompt, c_fp32)
    logits_bf16 = model(prompt, c_bf16)
    logits_i8 = model(prompt, c_i8)
    mx.eval(logits_fp32, logits_bf16, logits_i8)

    for e in c_i8:
        if isinstance(e, DeltaSlotCache):
            assert e._rec_state is None
            assert e.rec_codes is not None and e.rec_scales is not None
            assert e.rec_codes.dtype == mx.int8 and e.rec_scales.dtype == mx.bfloat16

    def _cos(a: mx.array, b: mx.array) -> float:
        af = a[0, -1].astype(mx.float32)
        bf = b[0, -1].astype(mx.float32)
        return float((mx.sum(af * bf) / (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf)) + 1e-12)).item())

    assert _cos(logits_fp32, logits_bf16) >= 0.999
    assert _cos(logits_fp32, logits_i8) >= 0.999

    # Verify rollback_to_accepted preserves int8_blk32 state representation across partial match + zero-accept
    l0 = np.asarray(logits_i8[0, -1].astype(mx.float32))
    t0 = int(np.argmax(l0))
    l1 = model(mx.array([[t0]], dtype=mx.int32), model.make_cache(gdn_state_mode="int8_blk32"))
    del l1
    emitted, _, n_matched = verify_draft_sequence_b1(
        model,
        c_i8,
        t0,
        [(t0 + 17) % cfg.vocab_size, 99, 101],
        pick_token_fn=lambda _i, lg: int(np.argmax(lg)),
        stop_token_ids=frozenset({0, 2}),
    )
    assert n_matched == 0 and len(emitted) == 1
    t1 = emitted[0]
    # Also test partial draft match (n_acc = 2) and FAST_QWEN3_USE_GDN_METAL=1 on fp32
    emitted2, _, n_matched2 = verify_draft_sequence_b1(
        model,
        c_i8,
        t1,
        [int(np.argmax(np.asarray(model(mx.array([[t1]], dtype=mx.int32), restore_hybrid_cache(snapshot_hybrid_cache([0], c_i8), batch_size=1, gdn_state_mode="int8_blk32"))[0, -1].astype(mx.float32)))), (t1 + 29) % cfg.vocab_size, 99],
        pick_token_fn=lambda _i, lg: int(np.argmax(lg)),
        stop_token_ids=frozenset({0, 2}),
    )
    assert n_matched2 == 1 and len(emitted2) == 2
    for e in c_i8:
        if isinstance(e, DeltaSlotCache):
            assert e.state_mode == "int8_blk32"
            assert e._rec_state is None
            assert e.rec_codes is not None and e.rec_scales is not None
            assert e.rec_codes.dtype == mx.int8 and e.rec_scales.dtype == mx.bfloat16
            # Test n_acc <= 0 rollback directly preserves exact tuple
            e.record_verify_trace = True
            e._verify_trace = (
                mx.zeros((1, 4, cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 + cfg.linear_num_value_heads * cfg.linear_value_head_dim), dtype=mx.bfloat16),
                mx.zeros((1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim), dtype=mx.bfloat16),
                mx.zeros((1, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim), dtype=mx.bfloat16),
                mx.zeros((1, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim), dtype=mx.bfloat16),
                mx.ones((1, 1, cfg.linear_num_value_heads), dtype=mx.float32),
                mx.zeros((1, 1, cfg.linear_num_value_heads), dtype=mx.bfloat16),
                (e.rec_codes, e.rec_scales),
            )
            saved_codes = e.rec_codes
            e.rollback_to_accepted(0, cfg.linear_conv_kernel_dim)
            assert e.rec_codes is saved_codes

    monkeypatch.setenv("FAST_QWEN3_USE_GDN_METAL", "1")
    c_fp32_metal = model.make_cache(gdn_state_mode="fp32")
    logits_fp32_metal = model(prompt, c_fp32_metal)
    mx.eval(logits_fp32_metal)
    assert _cos(logits_fp32, logits_fp32_metal) >= 0.9999
