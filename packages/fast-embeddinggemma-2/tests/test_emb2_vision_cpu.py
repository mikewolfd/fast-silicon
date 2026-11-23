"""CPU unit tests for lazy optional vision support in `fast-embeddinggemma-2`."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from fast_embeddinggemma_2.cli import build_parser
from fast_embeddinggemma_2.engine import EMBED_DIM, HIDDEN_SIZE, FastEmbeddingGemma2
from fast_shared.gemma4_vision import Gemma4VisionConfig


def test_import_fast_embeddinggemma2_does_not_load_vision_or_pil() -> None:
    """Text-only imports must never load `fast_shared.gemma4_vision` or `PIL`."""
    code = (
        "import sys, fast_embeddinggemma_2; "
        "assert 'fast_shared.gemma4_vision' not in sys.modules; "
        "assert 'PIL' not in sys.modules; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "OK" in proc.stdout


def test_encode_images_token_sequence_scatter_and_mrl_dim(tmp_path) -> None:
    captured: list[dict] = []

    class _FakeTokenTable:
        def gather_scaled_f16(self, ids: np.ndarray) -> np.ndarray:
            arr = np.asarray(ids, dtype=np.int32)
            captured.append({"gathered_ids": arr.copy()})
            out = np.zeros((arr.size, HIDDEN_SIZE), dtype=np.float16)
            # Mark each token row uniquely by token_id so we verify scatter overwrites only image positions
            out[:, 0] = np.where(arr >= 0, arr.astype(np.float32) * 0.01, -99.0).astype(np.float16)
            return out

    class _FakeGPU:
        def __init__(self) -> None:
            self.token_table = _FakeTokenTable()

        def encode_embedded_batch(
            self,
            h_mx: mx.array,
            msk_mx: mx.array,
            *,
            dimensions: int | None = None,
        ) -> np.ndarray:
            b_sz, l_pad = int(msk_mx.shape[0]), int(msk_mx.shape[1])
            h_np = np.asarray(h_mx, dtype=np.float32).reshape(b_sz, l_pad, HIDDEN_SIZE)
            m_np = np.asarray(msk_mx, dtype=bool)
            captured.append({"h_np": h_np.copy(), "msk_np": m_np.copy(), "dim": dimensions})
            eff_dim = EMBED_DIM if dimensions is None else int(dimensions)
            rng = np.random.default_rng(123)
            raw = rng.normal(size=(b_sz, eff_dim)).astype(np.float32)
            raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
            return raw

    class _FakeVisionModel:
        def __init__(self) -> None:
            self.cfg = Gemma4VisionConfig(
                hidden_size=768,
                text_hidden_size=HIDDEN_SIZE,
                patch_size=16,
                pooling_kernel_size=3,
                default_output_length=280,
                boi_token_id=255999,
                image_token_id=258880,
                eoi_token_id=258882,
            )

        def encode_images(self, inputs, *, batch_size: int = 8):
            outs = []
            for idx, item in enumerate(inputs):
                n_s = int(item.num_soft_tokens)
                arr = mx.full((n_s, HIDDEN_SIZE), float(idx + 7), dtype=mx.bfloat16)
                outs.append(arr)
            return outs

    class _FakeTokenizer:
        bos_id = 2
        eos_id = 1
        pad_id = 0

        def encode_batch(self, texts):
            return [[2, 1001, 1002, 1] for _ in texts]

    eng = object.__new__(FastEmbeddingGemma2)
    eng.dim = EMBED_DIM
    eng.tokenizer = _FakeTokenizer()
    eng._vision_model_dir = tmp_path
    eng._vision_model = _FakeVisionModel()
    import threading

    eng._vision_init_lock = threading.Lock()
    eng._gpu_lock = threading.Lock()
    eng._gpu = _FakeGPU()

    img1 = np.zeros((48, 48, 3), dtype=np.uint8)  # 48x48 -> 3x3 patches -> 1 soft token
    img2 = np.zeros((96, 48, 3), dtype=np.uint8)  # 96x48 -> 6x3 patches -> 2 soft tokens
    # Note preprocess scales to fit max_soft_tokens=280 (~16x more patches), test exact math cleanly:
    vecs = eng.encode_images([img1, img2], texts=[None, "receipt total"], dimensions=256, batch_size=2)
    assert vecs.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vecs, axis=-1), 1.0, atol=1e-5)

    assert len(captured) == 2
    g_ids = captured[0]["gathered_ids"].reshape(2, -1)
    assert g_ids.shape[1] % 64 == 0
    # Verify exact structure [bos=2, boi=255999, 0*n_vis, eoi=258882, ..., eos=1, -1...]
    row0 = g_ids[0]
    row1 = g_ids[1]
    assert list(row0[:2]) == [2, 255999]
    eoi_pos0 = list(row0).index(258882)
    assert eoi_pos0 > 2
    assert list(row0[eoi_pos0 : eoi_pos0 + 3]) == [258882, 1, -1]
    assert all(x == 0 for x in row0[2:eoi_pos0])

    # Second item includes optional text [1001, 1002] between eoi(258882) and eos(1)
    eoi_pos1 = list(row1).index(258882)
    assert list(row1[eoi_pos1 : eoi_pos1 + 5]) == [258882, 1001, 1002, 1, -1]

    # Verify visual scatter overwrote pad tokens with fill value 7.0 (row 0) and 8.0 (row 1)
    h_np = captured[1]["h_np"]
    assert np.allclose(h_np[0, 2:eoi_pos0, :], 7.0, atol=1e-2)
    assert np.allclose(h_np[1, 2:eoi_pos1, :], 8.0, atol=1e-2)
    assert captured[1]["dim"] == 256
    # Default budget is capped at 140 soft tokens (`4 + n_vis` <= 144 -> `eoi_pos0 - 2` <= 140)
    assert (eoi_pos0 - 2) <= 140

    # Explicit max_soft_tokens=280 overrides the default 140 cap cleanly
    captured.clear()
    _ = eng.encode_images([img1], max_soft_tokens=280, dimensions=256, batch_size=1)
    eoi_280 = list(captured[0]["gathered_ids"]).index(258882)
    assert (eoi_280 - 2) > 140


def test_cli_embed_images_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "embed-images",
            "/tmp/a.png",
            "/tmp/b.png",
            "--max-soft-tokens",
            "1120",
            "--batch-size",
            "4",
            "--dimensions",
            "256",
            "--mode",
            "gpu",
            "--no-warmup",
            "--json",
        ]
    )
    assert args.cmd == "embed-images"
    assert args.images == ["/tmp/a.png", "/tmp/b.png"]
    assert args.max_soft_tokens == 1120
    assert args.batch_size == 4
    assert args.dimensions == 256
    assert args.mode == "gpu"
    assert args.no_warmup is True
    assert args.json is True
