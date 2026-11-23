"""CPU/MLX unit tests for lazy optional vision support in `fast-gemma4`."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from fast_gemma4.cli import build_parser
from fast_gemma4.engine import FastGemma4Engine
from fast_gemma4.model import FastGemma4Model
from fast_gemma4.tokenizer import FastGemma4Tokenizer


def test_import_fast_gemma4_does_not_load_vision_or_pil() -> None:
    """Importing `fast_gemma4` must never eagerly import `fast_shared.gemma4_vision` or `PIL`."""
    code = (
        "import sys, fast_gemma4; "
        "assert 'fast_shared.gemma4_vision' not in sys.modules; "
        "assert 'PIL' not in sys.modules; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "OK" in proc.stdout


def _find_any_gemma4_tokenizer() -> Path | None:
    patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--gemma-4-E2B-it-MLX-4bit/snapshots/*/tokenizer.json"),
        os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-*/snapshots/*/tokenizer.json"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return Path(matches[0])
    return None


def test_expand_multimodal_prompt_ids_variants() -> None:
    tok_path = _find_any_gemma4_tokenizer()
    if tok_path is None:
        pytest.skip("No local Gemma-4 tokenizer.json cached")
    eng = object.__new__(FastGemma4Engine)
    eng.tokenizer = FastGemma4Tokenizer(tok_path)

    # 1. Explicit structured message with image part
    ids1, spans1 = eng._expand_multimodal_prompt_ids(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}],
        [6],
        system_prompt=None,
        use_chat_template=True,
        enable_thinking=False,
        boi_token_id=255999,
        image_token_id=258880,
        eoi_token_id=258882,
    )
    assert len(spans1) == 1
    s0, s1 = spans1[0]
    assert s1 - s0 == 6
    assert ids1[s0 - 1] == 255999
    assert ids1[s0:s1] == [258880] * 6
    assert ids1[s1] == 258882

    # 2. Implicit plain string prompt inserts images right after `<|turn>user\n`
    ids2, spans2 = eng._expand_multimodal_prompt_ids(
        "Read both pages.",
        [4, 5],
        system_prompt=None,
        use_chat_template=True,
        enable_thinking=False,
        boi_token_id=255999,
        image_token_id=258880,
        eoi_token_id=258882,
    )
    assert len(spans2) == 2
    assert ids2[spans2[0][0] : spans2[0][1]] == [258880] * 4
    assert ids2[spans2[1][0] : spans2[1][1]] == [258880] * 5
    user_hdr = eng.tokenizer.encode("<|turn>user\n", add_special_tokens=False)
    hdr_end = spans2[0][0] - 1
    assert ids2[hdr_end - len(user_hdr) : hdr_end] == user_hdr

    # 3. Mismatched explicit `<|image|>` count raises ValueError
    with pytest.raises(ValueError, match="Expected 2"):
        eng._expand_multimodal_prompt_ids(
            "Only one <|image|> marker",
            [3, 3],
            system_prompt=None,
            use_chat_template=True,
            enable_thinking=False,
            boi_token_id=255999,
            image_token_id=258880,
            eoi_token_id=258882,
        )


def test_compute_pli_from_embeds_and_vision_mask() -> None:
    m = object.__new__(FastGemma4Model)
    m.act_dtype = mx.bfloat16
    m.eps = 1e-6
    m.window_size = 4
    m.ple_dim = 4
    m.num_layers = 2
    m._n_early_pli_layers = 1
    m._ple_mmap_i8 = None
    m._ple_mmap_scales = None
    m._etpl_14_groups = 0
    m._etpl_14_wcols = 0
    m.embed_tokens_per_layer_scale = mx.array(2.0, dtype=mx.bfloat16)
    m.per_layer_projection_scale = mx.array(0.25, dtype=mx.bfloat16)
    m.per_layer_projection_norm_w = mx.ones((4,), dtype=mx.bfloat16)
    m.per_layer_input_scale = mx.array(0.707, dtype=mx.bfloat16)
    m.embed_tokens_per_layer = nn.Embedding(16, 8)
    m.per_layer_model_projection = nn.Linear(8, 8, bias=False)
    m.pli_proj_14 = nn.Linear(8, 4, bias=False)

    emb = mx.ones((1, 5, 8), dtype=mx.bfloat16)
    ple_toks = mx.array([[2, 0, 0, 3, 1]], dtype=mx.int32)
    pli_e14 = m._compute_pli_from_embeds(emb, ple_toks, early14=True)
    pli_full = m._compute_pli_from_embeds(emb, ple_toks, early14=False)
    assert pli_e14 is not None and pli_e14.shape == (1, 5, 1, 4)
    assert pli_full is not None and pli_full.shape == (1, 5, 2, 4)

    # Bidirectional mask within same image block (tokens 1..3) and causal across text/different images
    vis_group_ids = np.array([-1, 0, 0, 0, -1, 1, 1, -1], dtype=np.int32)
    mask_full = m._build_vision_sdpa_mask(0, 8, 0, 8, vis_group_ids, is_full=True)
    mx.eval(mask_full)
    assert mask_full.dtype == mx.bfloat16
    mask_np = np.asarray(mask_full.astype(mx.float32))
    # Token 1 inside image 0 sees future token 3 (same image), NOT text token 0's reverse? No, 1->0 allowed causal, 0->1 forbidden
    assert mask_np[0, 1] == float("-inf")
    assert mask_np[1, 3] == 0.0
    # Image 0 token 1 does NOT see image 1 token 5
    assert mask_np[1, 5] == float("-inf")
    # Text token 4 sees image 0 (0.0) but NOT future image 1 (-inf)
    assert mask_np[4, 2] == 0.0
    assert mask_np[4, 5] == float("-inf")

    # Sliding window enforces distance < 4 on cross-block pairs
    mask_swa = m._build_vision_sdpa_mask(0, 8, 0, 8, vis_group_ids, is_full=False)
    mx.eval(mask_swa)
    swa_np = np.asarray(mask_swa.astype(mx.float32))
    assert swa_np[7, 1] == float("-inf")
    assert swa_np[7, 5] == 0.0


def test_cli_generate_image_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--image",
            "/tmp/p1.png",
            "--image",
            "/tmp/p2.png",
            "--max-soft-tokens",
            "1120",
            "--max-tokens",
            "128",
            "Read pages.",
        ]
    )
    assert args.cmd == "generate"
    assert args.image == ["/tmp/p1.png", "/tmp/p2.png"]
    assert args.max_soft_tokens == 1120
    assert args.prompt == "Read pages."
