"""Pure CPU tests for shared `fast_shared.qwen35_vision` and `detect_apple10_d256_force_fused`."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw
from safetensors.numpy import save_file

from fast_shared.hw_check import detect_apple10_d256_force_fused
from fast_shared.qwen35_vision import (
    FastQwen35VisionModel,
    Qwen35VisionConfig,
    compute_mrope_positions,
    precompute_mrope_trig,
    prepare_pplx_image_item,
)


def _tiny_vision_cfg() -> Qwen35VisionConfig:
    return Qwen35VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        out_hidden_size=48,
    )


def test_fast_qwen35_vision_single_uniform_hetero_and_lru_cache() -> None:
    mx.random.seed(123)
    cfg = _tiny_vision_cfg()
    vis = FastQwen35VisionModel(cfg)
    mx.eval(vis.parameters())

    img_a = mx.random.normal((64, 3 * 2 * 16 * 16), dtype=mx.float32)  # (1, 8, 8) -> 16 merged tokens
    img_b = mx.random.normal((64, 3 * 2 * 16 * 16), dtype=mx.float32)  # (1, 8, 8)
    img_c = mx.random.normal((60, 3 * 2 * 16 * 16), dtype=mx.float32)  # (1, 6, 10) -> 15 merged tokens

    out_a = vis(img_a, [(1, 8, 8)])
    out_b = vis(img_b, [(1, 8, 8)])
    out_c = vis(img_c, [(1, 6, 10)])
    out_uniform = vis(mx.concatenate([img_a, img_b], axis=0), [(1, 8, 8), (1, 8, 8)])
    out_hetero = vis(mx.concatenate([img_a, img_c], axis=0), [(1, 8, 8), (1, 6, 10)])
    mx.eval(out_a, out_b, out_c, out_uniform, out_hetero)

    assert out_a.shape == (16, 48)
    assert out_uniform.shape == (32, 48)
    assert out_hetero.shape == (31, 48)
    assert float(mx.max(mx.abs(out_uniform[:16] - out_a)).item()) < 1e-4
    assert float(mx.max(mx.abs(out_uniform[16:] - out_b)).item()) < 1e-4
    assert float(mx.max(mx.abs(out_hetero[:16] - out_a)).item()) < 1e-4
    assert float(mx.max(mx.abs(out_hetero[16:] - out_c)).item()) < 1e-4

    # Grid geometry LRU cache hits and evicts above 16 entries cleanly
    first_entry = vis._get_grid_geometry([(1, 8, 8)], mx.float32)
    second_entry = vis._get_grid_geometry([(1, 8, 8)], mx.float32)
    assert first_entry is second_entry
    assert vis._get_grid_geometry([(1, 8, 8), (1, 8, 8)], mx.float32).uniform_frame_len == 64
    assert vis._get_grid_geometry([(1, 8, 8), (1, 6, 10)], mx.float32).uniform_frame_len == 0

    for step in range(2, 24, 2):
        _ = vis._get_grid_geometry([(1, 4, step)], mx.float32)
    assert len(vis._grid_cache) <= 16


def test_fast_qwen35_vision_from_pretrained_casts_fp32_to_bf16(tmp_path: Path) -> None:
    cfg = _tiny_vision_cfg()
    model = FastQwen35VisionModel(cfg)
    mx.eval(model.parameters())

    def _flatten(prefix: str, obj: object, out: dict[str, np.ndarray]) -> None:
        if isinstance(obj, mx.array):
            out[prefix] = np.asarray(obj.astype(mx.float32))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _flatten(f"{prefix}.{idx}", v, out)

    raw_np: dict[str, np.ndarray] = {}
    _flatten("visual", model.parameters(), raw_np)
    save_file(raw_np, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vision_config": {
                    "depth": cfg.depth,
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "num_heads": cfg.num_heads,
                    "in_channels": cfg.in_channels,
                    "patch_size": cfg.patch_size,
                    "temporal_patch_size": cfg.temporal_patch_size,
                    "spatial_merge_size": cfg.spatial_merge_size,
                    "num_position_embeddings": cfg.num_position_embeddings,
                    "out_hidden_size": cfg.out_hidden_size,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = FastQwen35VisionModel.from_pretrained(tmp_path, dtype=mx.bfloat16)
    assert loaded.patch_embed.proj.weight.dtype == mx.bfloat16
    assert loaded.blocks[0].attn.qkv.weight.dtype == mx.bfloat16


def test_prepare_pplx_image_item_exact_mrope_and_lazy_pil() -> None:
    # Build an image with huge blank border -> verify crop_margins=True shrinks grid tokens
    bordered = Image.new("RGB", (640, 640), (255, 255, 255))
    draw = ImageDraw.Draw(bordered)
    draw.rectangle([240, 240, 400, 400], fill=(10, 10, 10))

    full_item = prepare_pplx_image_item(bordered, chunks=4, min_pixels=16384, max_pixels=409600, crop_margins=False)
    cropped_item = prepare_pplx_image_item(bordered, chunks=4, min_pixels=16384, max_pixels=409600, crop_margins=True)

    assert full_item.vision_inputs.num_visual_tokens == 400  # 640x640 -> (1, 40, 40) -> 20x20 merged
    assert cropped_item.vision_inputs.num_visual_tokens < full_item.vision_inputs.num_visual_tokens
    assert cropped_item.cropped_box_px is not None

    # 20x20 merged grid split into 4 bands of 5 rows -> 3 separators between bands (rows 4|5, 9|10, 14|15)
    assert len(full_item.chunk_token_spans) == 4
    ids = full_item.input_ids.tolist()
    pos = full_item.position_ids
    assert ids[0] == 248078 and ids[1] == 248053 and ids[-1] == 248054
    assert np.allclose(pos[:, 0], [0.0, 0.0, 0.0])
    assert np.allclose(pos[:, 1], [1.0, 1.0, 1.0])
    assert np.allclose(pos[:, -1], [22.0, 22.0, 22.0])  # start(2) + max(20, 20)

    sep_indices = [i for i, tok in enumerate(ids) if tok == 248079]
    assert len(sep_indices) == 3
    # First separator sits after 5 rows (rows 0..4 -> r=4+2=6, next row 5 -> 7; midpoint = 6.5, width_mid = 2 + 19/2 = 11.5)
    assert np.allclose(pos[:, sep_indices[0]], [2.0, 6.5, 11.5])
    assert np.allclose(pos[:, sep_indices[1]], [2.0, 11.5, 11.5])
    assert np.allclose(pos[:, sep_indices[2]], [2.0, 16.5, 11.5])

    # Every chunk span covers only image_pad tokens with contiguous indices
    for s0, s1 in full_item.chunk_token_spans:
        assert s1 - s0 == 100
        assert all(tok == 248056 for tok in ids[s0:s1])

    # Precompute trig handles both float32 pplx positions and int32 generative positions cleanly
    cos_f, sin_f = precompute_mrope_trig(mx.array(pos[:, None, :]), rot_dims=64, dtype=mx.float16)
    assert cos_f.shape == (1, 1, len(ids), 32) and sin_f.shape == (1, 1, len(ids), 32)
    assert not bool(mx.any(mx.isnan(cos_f)).item())

    gen_pos, delta = compute_mrope_positions(
        [10, 248053, *([248056] * 4), 248054, 11],
        image_grid_thw=[(1, 4, 4)],
        image_token_id=248056,
    )
    assert delta == -2
    cos_g, _ = precompute_mrope_trig(gen_pos, rot_dims=64, dtype=mx.bfloat16)
    assert cos_g.shape == (1, 1, 8, 32)

    # Importing `fast_shared.qwen35_vision` alone must not eagerly import `PIL`
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {src_dir!r}); import fast_shared.qwen35_vision; assert 'PIL' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_detect_apple10_d256_force_fused(monkeypatch) -> None:
    monkeypatch.setenv("MLX_METAL_NO_NAX", "0")
    assert detect_apple10_d256_force_fused({"device_name": "Apple M4 Pro", "architecture": "applegpu_g16"}, "26.6") is False
    assert detect_apple10_d256_force_fused({"device_name": "Apple M5", "architecture": "applegpu_g17"}, "26.1") is False

    sig_has_force = "force_fused" in inspect.signature(mx.fast.scaled_dot_product_attention).parameters
    assert detect_apple10_d256_force_fused({"device_name": "Apple M5", "architecture": "applegpu_g17"}, "27.2") is sig_has_force

    monkeypatch.setenv("MLX_METAL_NO_NAX", "1")
    assert detect_apple10_d256_force_fused({"device_name": "Apple M5", "architecture": "applegpu_g17"}, "27.2") is False
