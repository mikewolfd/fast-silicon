from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from fast_shared.metal_mpp_int8 import (
    PACKS_META_ONLY,
    pack_affine_weights_for_mpp_i8,
    quantize_activations_i8_k64,
    should_use_mpp_int8_linear,
    should_use_mpp_w4a8,
    should_use_mpp_w8a8,
)


def test_pack_affine_weights_meta_only_and_explicit_packed_cpu() -> None:
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert PACKS_META_ONLY is True
        rng = np.random.default_rng(17)
        for bits in (4, 8):
            w = mx.array(rng.normal(scale=0.05, size=(128, 192)).astype(np.float32), dtype=mx.bfloat16)
            wq, sq, bq = mx.quantize(w, group_size=64, bits=bits, mode="affine")
            mx.eval(wq, sq, bq)

            wp_def, wm_def = pack_affine_weights_for_mpp_i8(wq, sq, bq, bits=bits, group_size=64)
            mx.eval(wp_def, wm_def)
            assert int(wp_def.size) == 0
            assert wp_def.dtype == mx.uint32
            assert wm_def.shape == (192 // 64, 128)
            assert wm_def.dtype == mx.uint32

            wp_full, wm_full = pack_affine_weights_for_mpp_i8(
                wq, sq, bq, bits=bits, group_size=64, include_packed=True
            )
            mx.eval(wp_full, wm_full)
            assert int(wp_full.size) > 0
            assert wp_full.dtype == (mx.int8 if bits == 8 else mx.uint32)
            assert np.array_equal(np.asarray(wm_full), np.asarray(wm_def))

        with pytest.raises(ValueError, match="group_size=64"):
            pack_affine_weights_for_mpp_i8(
                mx.zeros((64, 16), dtype=mx.uint32),
                mx.ones((64, 2), dtype=mx.bfloat16),
                mx.zeros((64, 2), dtype=mx.bfloat16),
                bits=8,
                group_size=32,
            )
    finally:
        mx.set_default_device(prev_dev)


def test_quantize_activations_i8_k64_validation_cpu() -> None:
    with pytest.raises(ValueError, match="Expected 2D"):
        quantize_activations_i8_k64(mx.zeros((2, 4, 64), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="multiple of 64"):
        quantize_activations_i8_k64(mx.zeros((4, 96), dtype=mx.bfloat16))


def test_should_use_mpp_int8_gating_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAST_SILICON_MPP_W8A8", raising=False)
    assert not should_use_mpp_int8_linear(512, 1536, 12288, bits=4, apple10_nax=False)
    assert not should_use_mpp_int8_linear(512, 1536, 12288, bits=8, apple10_nax=False)
    assert should_use_mpp_w4a8(32, 1536, 12288, apple10_nax=True)
    assert not should_use_mpp_w8a8(512, 1536, 12288, apple10_nax=True)
    monkeypatch.setenv("FAST_SILICON_MPP_W8A8", "1")
    assert should_use_mpp_w8a8(512, 1536, 12288, apple10_nax=True)
    assert not should_use_mpp_int8_linear(512, 1536, 12288, bits=6, apple10_nax=True)
