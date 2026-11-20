"""CPU unit tests and GPU parity/gate smoke test for opt-in Gemma-4 E2B MTP drafter."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import fast_gemma4
from fast_gemma4.cli import build_parser
from fast_gemma4.mtp_drafter import (
    DEFAULT_MTP_DRAFTER_REPO,
    MAX_MTP_BATCH_SIZE,
    Gemma4MtpDrafter,
    draft_scan_top_k,
    recommended_mtp_draft_k,
    resolve_mtp_drafter_dir,
    run_decode_mtp,
    should_use_mtp_drafter,
)

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def test_should_use_mtp_drafter_strict_batch_gate() -> None:
    sentinel = object()
    assert MAX_MTP_BATCH_SIZE == 4
    for b in (1, 2, 3, 4):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0) is True
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=True) is True
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=False) is False
        assert should_use_mtp_drafter(sentinel, True, b, 0.0) is False
        assert should_use_mtp_drafter(sentinel, False, b, 0.2) is False
        assert should_use_mtp_drafter(None, False, b, 0.0) is False

    for b in (0, -1, 5, 8, 16, 64, 128):
        assert should_use_mtp_drafter(sentinel, False, b, 0.0) is False
        assert should_use_mtp_drafter(sentinel, False, b, 0.0, speculative=True) is False


def test_recommended_mtp_draft_k_and_hard_gate() -> None:
    assert recommended_mtp_draft_k(1, has_grammar=False) == 4
    assert recommended_mtp_draft_k(1, has_grammar=True) == 3
    for b in (2, 3, 4):
        assert recommended_mtp_draft_k(b, has_grammar=False) == 2
        assert recommended_mtp_draft_k(b, has_grammar=True) == 2
        # Keep M = B * (K + 1) < 16 to avoid the 16<=M<=64 MPP tile cliff.
        assert b * (recommended_mtp_draft_k(b, has_grammar=True) + 1) < 16

    for bad_b in (0, 5, 8, 32, 128):
        with pytest.raises(ValueError, match="strictly gated"):
            recommended_mtp_draft_k(bad_b)


def test_mtp_drafter_rejects_e4b_and_invalid_top_k() -> None:
    e4b_stub = SimpleNamespace(hidden_size=2560, n_kv_heads=2)
    with pytest.raises(ValueError, match="Gemma-4 E2B target"):
        Gemma4MtpDrafter(e4b_stub, local_files_only=True)  # type: ignore[arg-type]

    e2b_stub = SimpleNamespace(hidden_size=1536, n_kv_heads=1)
    with pytest.raises(ValueError, match="top_k must be positive"):
        Gemma4MtpDrafter(e2b_stub, top_k=0, local_files_only=True)  # type: ignore[arg-type]


def test_run_decode_mtp_rejects_batch_above_4() -> None:
    for bad_b in (0, 5, 8, 16):
        with pytest.raises(ValueError, match="strictly gated"):
            run_decode_mtp(
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                frozenset(),
                None,
                mx.zeros((1, 1, 1)),
                mx.zeros((1,), dtype=mx.int32),
                {},
                bad_b,
            )


def test_draft_scan_top_k_exact_and_clustered() -> None:
    rng = np.random.default_rng(42)
    n_c, per_c, dim = 16, 8, 32
    vocab = n_c * per_c

    centroids_np = rng.normal(size=(n_c, dim)).astype(np.float32)
    centroids_np /= np.linalg.norm(centroids_np, axis=-1, keepdims=True)
    # Build tightly clustered embeddings around each centroid.
    embeds_clustered = (
        centroids_np[:, None, :] + 0.05 * rng.normal(size=(n_c, per_c, dim)).astype(np.float32)
    )
    order_np = rng.permutation(vocab).astype(np.int32)
    order_2d_np = order_np.reshape(n_c, per_c)

    # Full unpermuted table indexed by original token ID for exact reference argmax.
    full_w_np = np.zeros((vocab, dim), dtype=np.float32)
    for c in range(n_c):
        for j in range(per_c):
            full_w_np[order_2d_np[c, j]] = embeds_clustered[c, j]

    with mx.stream(mx.cpu):
        centroids_mx = mx.array(centroids_np, dtype=mx.bfloat16)
        embed_perm_mx = mx.array(embeds_clustered, dtype=mx.bfloat16)
        order_2d_mx = mx.array(order_2d_np, dtype=mx.int32)
        full_w_mx = mx.array(full_w_np, dtype=mx.bfloat16)

        # Queries aligned near 4 distinct centroids -> top_k=4 and full scan must agree exactly.
        q_np = centroids_np[[1, 5, 9, 13]] + 0.02 * rng.normal(size=(4, dim)).astype(np.float32)
        q_mx = mx.array(q_np, dtype=mx.bfloat16)

        ref_ids = mx.argmax(q_mx @ full_w_mx.T, axis=-1).astype(mx.int32)
        full_scan_ids = draft_scan_top_k(q_mx, centroids_mx, embed_perm_mx, order_2d_mx, top_k=n_c)
        topk_scan_ids = draft_scan_top_k(q_mx, centroids_mx, embed_perm_mx, order_2d_mx, top_k=4)
        mx.eval(ref_ids, full_scan_ids, topk_scan_ids)

    assert full_scan_ids.tolist() == ref_ids.tolist()
    assert topk_scan_ids.tolist() == ref_ids.tolist()


def test_cli_mtp_drafter_flags_preserve_positional_prompt() -> None:
    parser = build_parser()

    args_default = parser.parse_args(["generate", "Hello world"])
    assert args_default.mtp_drafter is False
    assert args_default.mtp_drafter_path is None
    assert args_default.prompt == "Hello world"

    args_flag = parser.parse_args(["generate", "--mtp-drafter", "Hello world"])
    assert args_flag.mtp_drafter is True
    assert args_flag.mtp_drafter_path is None
    assert args_flag.prompt == "Hello world"

    args_path = parser.parse_args(["generate", "--mtp-drafter-path", "/tmp/my-drafter", "Hello world"])
    assert args_path.mtp_drafter is False
    assert args_path.mtp_drafter_path == "/tmp/my-drafter"
    assert args_path.prompt == "Hello world"


def test_lazy_exports_in_fast_gemma4_init() -> None:
    assert fast_gemma4.MAX_MTP_BATCH_SIZE == 4
    assert fast_gemma4.DEFAULT_MTP_DRAFTER_REPO == DEFAULT_MTP_DRAFTER_REPO
    assert fast_gemma4.Gemma4MtpDrafter is Gemma4MtpDrafter
    assert fast_gemma4.should_use_mtp_drafter is should_use_mtp_drafter
    assert fast_gemma4.recommended_mtp_draft_k is recommended_mtp_draft_k
    assert fast_gemma4.DEFAULT_MAX_WHITESPACE_RUN == 32
    assert fast_gemma4.WhitespaceGuard is not None


def test_mtp_drafter_b1_parity_and_b8_gate_gpu() -> None:
    """Serialized M4 GPU smoke test: B=1 MTP parity (free-text + JSON) & B=8 hard gate."""
    try:
        drafter_dir = resolve_mtp_drafter_dir(DEFAULT_MTP_DRAFTER_REPO, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Local HF cache missing {DEFAULT_MTP_DRAFTER_REPO}: {exc}")
    if not (drafter_dir / "model.safetensors").is_file():
        pytest.skip(f"Missing model.safetensors in {drafter_dir}")

    from fast_gemma4.engine import FastGemma4Engine

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    schema_str = (fixtures_dir / "extract_v1.schema.json").read_text(encoding="utf-8")
    sys_prompt = (fixtures_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    docs_raw = json.loads((fixtures_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    doc_prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs_raw]

    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HW_LOCK_PATH.open("a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        # Use kv_i8=False so plain rolling (`speculative=False`) and MTP (`make_batch_cache`) share identical BF16 KV.
        eng = FastGemma4Engine(default_mode="gpu", kv_i8=False, mtp_drafter=drafter_dir, warmup=False)
        try:
            # 1) B=1 free-text parity: MTP greedy vs plain greedy.
            ft_prompt = "Explain in three concise bullet points how ring-buffer sliding-window KV caching bounds memory."
            ref_ft = eng.generate_batch([ft_prompt], max_tokens=64, temperature=0.0, mode="gpu", speculative=False)
            mtp_ft = eng.generate_batch([ft_prompt], max_tokens=64, temperature=0.0, mode="gpu", speculative=None)
            assert mtp_ft.stats.get("speculative_backend") == "mtp"
            assert mtp_ft.stats.get("mtp_accept_rate", 0.0) > 0.25
            assert mtp_ft.items[0].token_ids == ref_ft.items[0].token_ids
            assert mtp_ft.items[0].text == ref_ft.items[0].text

            # 2) B=1 constrained JSON parity + schema validity.
            ref_json = eng.generate_batch(
                [doc_prompts[0]],
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=256,
                temperature=0.0,
                mode="gpu",
                speculative=False,
            )
            mtp_json = eng.generate_batch(
                [doc_prompts[0]],
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=256,
                temperature=0.0,
                mode="gpu",
                speculative=None,
            )
            assert mtp_json.stats.get("speculative_backend") == "mtp"
            assert mtp_json.stats.get("valid_json_count") == 1
            assert mtp_json.stats.get("valid_schema_count") == 1
            assert mtp_json.stats.get("mtp_accept_rate", 0.0) > 0.45
            assert mtp_json.items[0].token_ids == ref_json.items[0].token_ids
            assert mtp_json.items[0].parsed == ref_json.items[0].parsed

            # 3) Hard B>=8 gate: even with mtp_drafter attached and speculative=None or True, B=8 NEVER uses MTP.
            res_b8_auto = eng.generate_batch(
                doc_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=32,
                temperature=0.0,
                mode="gpu",
                speculative=None,
            )
            assert res_b8_auto.stats.get("speculative_backend") != "mtp"
            assert res_b8_auto.stats.get("rolling") is True

            res_b8_spec = eng.generate_batch(
                doc_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=32,
                temperature=0.0,
                mode="gpu",
                speculative=True,
            )
            assert res_b8_spec.stats.get("speculative_backend") != "mtp"
        finally:
            eng.close()
            mx.clear_cache()
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
