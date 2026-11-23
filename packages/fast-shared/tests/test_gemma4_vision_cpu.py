"""CPU/MLX unit tests for `fast_shared.gemma4_vision` (lazy import, preprocessing, parity)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fast_shared.gemma4_vision import (
    FastGemma4VisionModel,
    Gemma4VisionConfig,
    get_aspect_ratio_preserving_size,
    preprocess_gemma4_image,
)


def test_import_fast_shared_does_not_load_gemma4_vision_or_pil() -> None:
    """Importing `fast_shared` must never eagerly import `gemma4_vision` or `PIL`."""
    code = (
        "import sys, fast_shared; "
        "assert 'fast_shared.gemma4_vision' not in sys.modules; "
        "assert 'PIL' not in sys.modules; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "OK" in proc.stdout


def test_aspect_ratio_preserving_size_variants() -> None:
    # Standard landscape (e.g. 420x280 -> side_mult=48, max_patches=280*9=2520)
    h, w = get_aspect_ratio_preserving_size(280, 420, patch_size=16, max_patches=2520, pooling_kernel_size=3)
    assert h % 48 == 0 and w % 48 == 0
    assert (h // 16) * (w // 16) <= 2520
    assert (h // 48) * (w // 48) <= 280

    # High-res 1120 soft tokens
    h2, w2 = get_aspect_ratio_preserving_size(1400, 1000, patch_size=16, max_patches=1120 * 9, pooling_kernel_size=3)
    assert h2 % 48 == 0 and w2 % 48 == 0
    assert (h2 // 48) * (w2 // 48) <= 1120

    # Extreme aspect ratio clamping edge cases
    h_ext, w_ext = get_aspect_ratio_preserving_size(10, 5000, patch_size=16, max_patches=2520, pooling_kernel_size=3)
    assert h_ext == 48 and w_ext >= 48 and w_ext % 48 == 0

    h_tall, w_tall = get_aspect_ratio_preserving_size(5000, 10, patch_size=16, max_patches=2520, pooling_kernel_size=3)
    assert w_tall == 48 and h_tall >= 48 and h_tall % 48 == 0


def test_preprocess_gemma4_image_range_and_grid() -> None:
    arr = np.full((192, 288, 3), 128, dtype=np.uint8)
    arr[:20, :20, :] = 255
    vi = preprocess_gemma4_image(arr, patch_size=16, pooling_kernel_size=3, max_soft_tokens=280)
    assert vi.pixel_values.dtype == np.float32
    assert vi.pixel_values.shape[0] == 3
    assert vi.pixel_values.shape[1] == vi.resized_height
    assert vi.pixel_values.shape[2] == vi.resized_width
    assert vi.resized_height % 48 == 0 and vi.resized_width % 48 == 0
    assert vi.grid_hw == (vi.resized_height // 16, vi.resized_width // 16)
    assert vi.num_soft_tokens == (vi.grid_hw[0] // 3) * (vi.grid_hw[1] // 3)
    assert 0.0 <= float(vi.pixel_values.min()) <= float(vi.pixel_values.max()) <= 1.0


def _bruteforce_ref_forward(
    vis: FastGemma4VisionModel,
    px: mx.array,
    grid_hw: tuple[int, int],
) -> mx.array:
    """Reference path using explicit 10,240-wide one-hot position matmul & one-hot pooler."""
    cfg = vis.cfg
    p = cfg.patch_size
    k = cfg.pooling_kernel_size
    ph, pw = grid_hw
    b_sz = int(px.shape[0])

    # Explicit one-hot position embedding
    gx, gy = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    pos_np = np.stack([gx.flatten(), gy.flatten()], axis=-1).astype(np.int32)
    pos_mx = mx.array(np.tile(pos_np[None], (b_sz, 1, 1)))
    oh = (mx.expand_dims(pos_mx, -1) == mx.arange(cfg.position_embedding_size)).astype(mx.float32)
    oh_t = oh.transpose(0, 2, 1, 3).astype(vis.vision_tower.patch_embedder.position_embedding_table.dtype)
    pos_emb = (oh_t @ vis.vision_tower.patch_embedder.position_embedding_table).sum(axis=1)

    patches = px.reshape(b_sz, 3, ph, p, pw, p).transpose(0, 2, 4, 3, 5, 1).reshape(b_sz, ph * pw, 3 * p * p)
    patches = 2.0 * (patches - 0.5)
    h = vis.vision_tower.patch_embedder.input_proj(patches.astype(mx.bfloat16)) + pos_emb

    # Explicit per-layer loop without fuse/compile + explicit 2D RoPE
    def _rot_half(t: mx.array) -> mx.array:
        m = t.shape[-1] // 2
        return mx.concatenate([-t[..., m:], t[..., :m]], axis=-1)

    def _rope2d(t: mx.array) -> mx.array:
        c_dim = 2 * (cfg.head_dim // 4)
        half = c_dim // 2
        parts = []
        for d in range(2):
            xp = t[..., d * c_dim : (d + 1) * c_dim]
            freq_exp = (2.0 / c_dim) * mx.arange(0, half).astype(mx.float32)
            ts = mx.power(cfg.rope_theta, freq_exp)
            sin_inp = pos_mx[..., d : d + 1].astype(mx.float32) / ts
            cd = mx.concatenate([mx.cos(sin_inp), mx.cos(sin_inp)], axis=-1).astype(t.dtype)[:, :, None, :]
            sd = mx.concatenate([mx.sin(sin_inp), mx.sin(sin_inp)], axis=-1).astype(t.dtype)[:, :, None, :]
            parts.append(xp * cd + _rot_half(xp) * sd)
        return mx.concatenate(parts, axis=-1)

    for layer in vis.vision_tower.encoder.layers:
        xn = mx.fast.rms_norm(h, layer.input_layernorm.weight, cfg.rms_norm_eps)
        sa = layer.self_attn
        q = sa.q_norm(sa.q_proj(xn).reshape(b_sz, ph * pw, cfg.num_attention_heads, cfg.head_dim))
        kv_k = sa.k_norm(sa.k_proj(xn).reshape(b_sz, ph * pw, cfg.num_key_value_heads, cfg.head_dim))
        v = mx.fast.rms_norm(sa.v_proj(xn).reshape(b_sz, ph * pw, cfg.num_key_value_heads, cfg.head_dim), None, cfg.rms_norm_eps)
        q = _rope2d(q).transpose(0, 2, 1, 3)
        kv_k = _rope2d(kv_k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        att = mx.fast.scaled_dot_product_attention(q, kv_k, v, scale=1.0).transpose(0, 2, 1, 3).reshape(b_sz, ph * pw, -1)
        h = h + mx.fast.rms_norm(sa.o_proj(att), layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        hin = mx.fast.rms_norm(h, layer.pre_feedforward_layernorm.weight, cfg.rms_norm_eps)
        mlp = layer.mlp
        ff = mlp.down_proj(nn.gelu_approx(mlp.gate_proj(hin)) * mlp.up_proj(hin))
        h = h + mx.fast.rms_norm(ff, layer.post_feedforward_layernorm.weight, cfg.rms_norm_eps)

    # Brute-force one-hot pooler
    n_soft = (ph // k) * (pw // k)
    k_idx = (pos_mx[..., 0] // k) + (pw // k) * (pos_mx[..., 1] // k)
    w_pool = ((mx.expand_dims(k_idx, -1) == mx.arange(n_soft)).astype(mx.float32)) / float(k * k)
    pooled = mx.einsum("bLl,bLd->bld", w_pool, h).astype(h.dtype) * (cfg.hidden_size**0.5)
    if cfg.standardize:
        pooled = (pooled - vis.vision_tower.std_bias) * vis.vision_tower.std_scale
    return vis.embed_vision(pooled)


@pytest.mark.parametrize("use_clip", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_tiny_gemma4_vision_parity_with_bruteforce(use_clip: bool, standardize: bool) -> None:
    mx.random.seed(7)
    cfg = Gemma4VisionConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=256,
        patch_size=16,
        pooling_kernel_size=3,
        default_output_length=12,
        position_embedding_size=64,
        use_clipped_linears=use_clip,
        standardize=standardize,
        text_hidden_size=96,
    )
    vis = FastGemma4VisionModel(cfg, dtype=mx.bfloat16)
    vis.vision_tower.patch_embedder.position_embedding_table = (
        mx.random.normal((2, 64, 128)) * 0.05
    ).astype(mx.bfloat16)
    if standardize:
        vis.vision_tower.std_bias = (mx.random.normal((128,)) * 0.1).astype(mx.bfloat16)
        vis.vision_tower.std_scale = (mx.ones((128,)) + mx.random.normal((128,)) * 0.05).astype(mx.bfloat16)
    for layer in vis.vision_tower.encoder.layers:
        for proj in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            proj.linear.weight = (mx.random.normal(proj.linear.weight.shape) * 0.05).astype(mx.bfloat16)
            if use_clip:
                proj.input_min = mx.array(-2.0, dtype=mx.bfloat16)
                proj.input_max = mx.array(2.0, dtype=mx.bfloat16)
                proj.output_min = mx.array(-2.0, dtype=mx.bfloat16)
                proj.output_max = mx.array(2.0, dtype=mx.bfloat16)
    vis.vision_tower.patch_embedder.input_proj.weight = (
        mx.random.normal(vis.vision_tower.patch_embedder.input_proj.weight.shape) * 0.05
    ).astype(mx.bfloat16)
    vis.embed_vision.embedding_projection.weight = (
        mx.random.normal(vis.embed_vision.embedding_projection.weight.shape) * 0.05
    ).astype(mx.bfloat16)
    mx.eval(vis.parameters())

    px = mx.random.uniform(shape=(2, 3, 96, 144), dtype=mx.float32)  # pH=6, pW=9 -> n_soft=6
    ref = _bruteforce_ref_forward(vis, px, (6, 9))
    vis.finalize_weights()
    got = vis.encode_batch(px, (6, 9))
    mx.eval(ref, got)
    assert got.shape == (2, 6, 96)
    max_diff = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())
    assert max_diff < 5e-2, f"max_diff={max_diff}"


def test_from_pretrained_quantized_embed_vision_and_missing_vision_error(tmp_path: Path) -> None:
    cfg_dict = {
        "model_type": "gemma4",
        "text_config": {"hidden_size": 128},
        "vision_config": {
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "intermediate_size": 128,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "default_output_length": 6,
            "position_embedding_size": 32,
            "use_clipped_linears": True,
        },
        "quantization": {"bits": 4, "group_size": 32},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")

    # Save vision + 4bit embed_vision + unrelated language_model tensor
    ql = nn.QuantizedLinear(64, 128, bias=False, group_size=32, bits=4)
    w_map: dict[str, mx.array] = {
        "language_model.model.embed_tokens.weight": mx.zeros((8, 128), dtype=mx.bfloat16),
        "vision_tower.patch_embedder.input_proj.weight": mx.ones((64, 3 * 16 * 16), dtype=mx.bfloat16) * 0.01,
        "vision_tower.patch_embedder.position_embedding_table": mx.zeros((2, 32, 64), dtype=mx.bfloat16),
        "embed_vision.embedding_projection.weight": ql.weight,
        "embed_vision.embedding_projection.scales": ql.scales,
        "embed_vision.embedding_projection.biases": ql.biases,
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), w_map)

    vis = FastGemma4VisionModel.from_pretrained(tmp_path, dtype=mx.bfloat16)
    assert isinstance(vis.embed_vision.embedding_projection, nn.QuantizedLinear)
    out = vis(mx.zeros((1, 3, 48, 48), dtype=mx.float32))
    mx.eval(out)
    assert out.shape == (1, 1, 128)

    # Text-only stripped checkpoint without vision_tower weights should raise ValueError
    stripped = tmp_path / "stripped"
    stripped.mkdir()
    (stripped / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    mx.save_safetensors(
        str(stripped / "model.safetensors"),
        {"language_model.model.embed_tokens.weight": mx.zeros((8, 128), dtype=mx.bfloat16)},
    )
    with pytest.raises(ValueError, match="vision_tower"):
        FastGemma4VisionModel.from_pretrained(stripped)
