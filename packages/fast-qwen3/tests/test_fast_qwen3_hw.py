"""Real-checkpoint hardware tests (`@pytest.mark.gpu_ane`) for `fast-qwen3` on Apple Silicon."""

from __future__ import annotations

import fcntl
import gc
import os
from pathlib import Path

import mlx.core as mx
import pytest

from fast_qwen3 import (
    FastQwen3Engine,
    FastQwen3Model,
    FastQwen3Tokenizer,
    Qwen3MtpDrafter,
    resolve_paths,
)

LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


def _find_cached_qwen35_checkpoints() -> list[tuple[str, Path]]:
    paths = resolve_paths()
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for preset in ("4b", "0.8b", "0.8b-bf16", "2b"):
        try:
            snap = paths.resolve_model_dir(preset, local_files_only=True)
            if snap not in seen:
                seen.add(snap)
                found.append((preset, snap))
        except Exception:
            continue
    return found


@pytest.mark.gpu_ane
def test_real_qwen35_strict_json_fusion_ulp_and_speculative() -> None:
    ckpts = _find_cached_qwen35_checkpoints()
    if not ckpts and os.environ.get("FAST_QWEN3_ALLOW_DOWNLOAD") != "1":
        pytest.skip("No cached Qwen3.5 checkpoint found locally (run `fast-qwen3 setup-weights --allow-download`)")

    preset, snap = ckpts[0] if ckpts else ("4b", resolve_paths().resolve_model_dir("4b", local_files_only=False))
    mx.set_cache_limit(512 << 20)
    LOCK_PATH.touch(exist_ok=True)
    with open(LOCK_PATH, "r+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        tok = FastQwen3Tokenizer(snap)

        # 1. Verify 0.0-ULP fusion parity between SpecBucketedLinearGroup and unfused projections
        m_unfused, _ = FastQwen3Model.from_pretrained(snap, enable_fusion=False)
        probe_ids = tok.encode_chat(
            "Acme Robotics submitted its annual safety audit in 2025.",
            system_prompt="Extract company and year as JSON.",
            enable_thinking=False,
        )
        inp = mx.array([probe_ids], dtype=mx.int32)
        logits_u = m_unfused(inp, m_unfused.make_cache(kv_step=128))
        mx.eval(logits_u)
        del m_unfused
        gc.collect()
        mx.clear_cache()

        m_fused, mtp_w = FastQwen3Model.from_pretrained(snap, enable_fusion=True)
        logits_f = m_fused(inp, m_fused.make_cache(kv_step=128))
        mx.eval(logits_f)
        lf_last = logits_f[0, -1].astype(mx.float32)
        lu_last = logits_u[0, -1].astype(mx.float32)
        cos_sim = float(
            (mx.sum(lf_last * lu_last) / (mx.sqrt(mx.sum(lf_last * lf_last)) * mx.sqrt(mx.sum(lu_last * lu_last)) + 1e-12)).item()
        )
        max_diff = float(mx.max(mx.abs(lf_last - lu_last)).item())
        assert cos_sim >= 0.999, f"Expected >=0.999 32-layer BF16 cosine parity on {preset}, got {cos_sim}"
        assert max_diff <= 1.5, f"Expected <=1.5 BF16 logit diff on {preset}, got {max_diff}"
        assert int(mx.argmax(lf_last).item()) == int(mx.argmax(lu_last).item())
        del logits_u, logits_f

        # 2. Verify strict JSON extraction + LCP prefix cache + left-pad B=2 batching + speculative greedy match
        drafter = Qwen3MtpDrafter.from_mtp_weights(m_fused, mtp_w)
        eng = FastQwen3Engine(model=m_fused, tokenizer=tok, mtp_drafter=drafter)
        try:
            schema = {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "year": {"type": "integer"},
                },
                "required": ["company", "year"],
                "additionalProperties": False,
            }
            sys_p = (
                "You are a deterministic structured extraction assistant. "
                "Extract the exact company name and integer filing year as strict JSON."
            )
            batch = eng.generate_batch(
                [
                    "Acme Robotics submitted its annual safety audit in 2025.",
                    "Northstar BioLabs published clinical phase-3 results in 2026.",
                ],
                system_prompt=sys_p,
                json_schema=schema,
                max_tokens=64,
                batch_size=2,
            )
            assert len(batch.items) == 2
            for item in batch.items:
                assert item.valid_json is True
                assert item.valid_schema is True
                assert isinstance(item.parsed["year"], int)

            # Speculative greedy T=0 factual coherence & prefix agreement (allowing qmv vs qmm BF16 1-ULP tie-breaks)
            sp_on = eng.generate("Answer in one word: what is the capital of France?", max_tokens=16, speculative=True)
            sp_off = eng.generate("Answer in one word: what is the capital of France?", max_tokens=16, speculative=False)
            assert "Paris" in sp_on.text and "Paris" in sp_off.text
            assert sp_on.token_ids[:2] == sp_off.token_ids[:2]
        finally:
            eng.close()
            del m_fused
            gc.collect()
            mx.clear_cache()
