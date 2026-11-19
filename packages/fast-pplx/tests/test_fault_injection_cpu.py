"""CPU fault-injection tests proving parity sensitivity to architectural regressions."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from fast_pplx.config import DEFAULT_PPLX_CONFIG, PplxConfig
from fast_pplx.gpu import PplxModel, _build_sdpa_mask
from fast_pplx.tokenizer import ContextualTokenizer
from fast_shared.fault_injection import FaultInjectionCase, run_fault_injection_suite


def _build_synthetic_cpu_model() -> PplxModel:
    cfg = PplxConfig(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        vocab_size=64,
        embedding_dim=32,
        layer_types=("linear_attention", "full_attention"),
    )
    rng = np.random.default_rng(42)
    with mx.stream(mx.cpu):
        model = PplxModel(cfg, has_embed_module=True)
        weights: list[tuple[str, mx.array]] = [
            ("embed_tokens.weight", mx.array(rng.normal(0.0, 0.25, (cfg.vocab_size, 32)).astype(np.float16))),
            ("norm.weight", mx.array(np.ones((32,), dtype=np.float16) + rng.normal(0.0, 0.05, (32,)).astype(np.float16))),
            ("contextual_projection.weight", mx.array(rng.normal(0.0, 0.25, (32, 32)).astype(np.float32))),
        ]
        for idx, layer in enumerate(model.layers):
            weights.append((f"layers.{idx}.input_layernorm.weight", mx.ones((32,), dtype=mx.float16)))
            weights.append((f"layers.{idx}.post_attention_layernorm.weight", mx.ones((32,), dtype=mx.float16)))
            weights.append((f"layers.{idx}.mlp.gate_up_proj.weight", mx.array(rng.normal(0.0, 0.15, (128, 32)).astype(np.float16))))
            weights.append((f"layers.{idx}.mlp.down_proj.weight", mx.array(rng.normal(0.0, 0.15, (32, 64)).astype(np.float16))))
            if layer.is_linear:
                weights.extend(
                    [
                        (f"layers.{idx}.linear_attn.in_proj_qkv.weight", mx.array(rng.normal(0.0, 0.15, (96, 32)).astype(np.float16))),
                        (f"layers.{idx}.linear_attn.in_proj_z.weight", mx.array(rng.normal(0.0, 0.15, (32, 32)).astype(np.float16))),
                        (f"layers.{idx}.linear_attn.in_proj_b.weight", mx.array(rng.normal(0.0, 0.15, (2, 32)).astype(np.float16))),
                        (f"layers.{idx}.linear_attn.in_proj_a.weight", mx.array(rng.normal(0.0, 0.15, (2, 32)).astype(np.float16))),
                        (f"layers.{idx}.linear_attn.conv1d.weight", mx.array(rng.normal(0.0, 0.15, (96, 4, 1)).astype(np.float16))),
                        (f"layers.{idx}.linear_attn.A_log", mx.zeros((2,), dtype=mx.float32)),
                        (f"layers.{idx}.linear_attn.dt_bias", mx.zeros((2,), dtype=mx.float32)),
                        (f"layers.{idx}.linear_attn.norm.weight", mx.ones((16,), dtype=mx.float32)),
                        (f"layers.{idx}.linear_attn.out_proj.weight", mx.array(rng.normal(0.0, 0.15, (32, 32)).astype(np.float16))),
                    ]
                )
            else:
                weights.extend(
                    [
                        (f"layers.{idx}.self_attn.qkv_proj.weight", mx.array(rng.normal(0.0, 0.15, (96, 32)).astype(np.float16))),
                        (f"layers.{idx}.self_attn.o_proj.weight", mx.array(rng.normal(0.0, 0.15, (32, 32)).astype(np.float16))),
                        (f"layers.{idx}.self_attn.q_norm.weight", mx.ones((16,), dtype=mx.float16)),
                        (f"layers.{idx}.self_attn.k_norm.weight", mx.ones((16,), dtype=mx.float16)),
                    ]
                )
        model.load_weights(weights, strict=True)
        mx.eval(model.parameters())
        for layer in model.layers:
            if layer.is_linear:
                layer.linear_attn.use_kernel = False
        return model


def _make_cpu_tokenizer(tmp_path: Path) -> ContextualTokenizer:
    vocab = {
        "<unk>": 0,
        "[D": 1,
        "]": 2,
        "alpha": 3,
        "beta": 4,
        "gamma": 5,
        "delta": 6,
        "<|endoftext|>": 7,
        "[Q] ": 8,
        "[D] ": 9,
        "<|chunk_sep|>": 10,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(["<|endoftext|>", "[Q] ", "[D] "])
    tok.add_tokens(["<|chunk_sep|>"])
    tok_path = tmp_path / "fault_tok.json"
    tok.save(str(tok_path))
    cfg = PplxConfig(
        vocab_size=64,
        eos_token_id=7,
        pad_token_id=7,
        query_prefix_token_id=8,
        document_prefix_token_id=9,
        chunk_sep_token_id=10,
        layer_types=DEFAULT_PPLX_CONFIG.layer_types,
    )
    return ContextualTokenizer(tok_path, cfg)


def test_fault_injection_breaks_model_parity_on_cpu() -> None:
    model = _build_synthetic_cpu_model()
    input_ids_np = np.array([[1, 2, 3, 4, 5, 0, 0], [1, 2, 6, 7, 8, 9, 10]], dtype=np.int32)
    valid_mask_np = np.array([[True, True, True, True, True, False, False], [True] * 7], dtype=np.bool_)

    fault_state = {
        "drop_sdpa_mask": False,
        "causal_sdpa": False,
        "unshifted_norm": False,
        "drop_sep_token": False,
    }

    def run_baseline() -> np.ndarray:
        ids = input_ids_np.copy()
        if fault_state["drop_sep_token"]:
            # Corrupt chunk separator position at token index 4 with ordinary token.
            ids[0, 3] = 20
        with mx.stream(mx.cpu):
            assert model.embed_tokens is not None
            h = model.embed_tokens(mx.array(ids, dtype=mx.int32))
            if fault_state["unshifted_norm"]:
                # Subtract the +1.0 zero-centered offset from RMSNorm weights temporarily.
                saved_norm = model.norm.weight
                model.norm.weight = saved_norm - mx.array(1.0, dtype=mx.float16)
            else:
                saved_norm = None

            try:
                ssm_mask = mx.array(valid_mask_np, dtype=mx.bool_)
                if fault_state["drop_sdpa_mask"]:
                    sdpa_mask = None
                elif fault_state["causal_sdpa"]:
                    B, S = ids.shape
                    causal_bool = np.tril(np.ones((S, S), dtype=np.bool_))[None, None, :, :] & valid_mask_np[:, None, None, :]
                    causal_arr = np.where(causal_bool, np.float16(0.0), np.float16(-65504.0))
                    sdpa_mask = mx.array(causal_arr, dtype=mx.float16)
                else:
                    sdpa_mask = _build_sdpa_mask(valid_mask_np, dtype_name="float16")

                out_h = model.forward_hidden(h, ssm_mask=ssm_mask, sdpa_mask=sdpa_mask)
                proj = model.contextual_projection(out_h[:, :5, :].mean(axis=1).astype(mx.float32))
                mx.eval(proj)
                return np.array(proj, dtype=np.float32)
            finally:
                if saved_norm is not None:
                    model.norm.weight = saved_norm

    @contextmanager
    def _flip(flag: str) -> Iterator[None]:
        fault_state[flag] = True
        try:
            yield
        finally:
            fault_state[flag] = False

    cases = [
        FaultInjectionCase("missing_sdpa_padding_mask", lambda: _flip("drop_sdpa_mask"), min_delta=1e-2, metric="max_abs_diff"),
        FaultInjectionCase("causal_sdpa_instead_of_bidirectional", lambda: _flip("causal_sdpa"), min_delta=1e-2, metric="max_abs_diff"),
        FaultInjectionCase("wrong_zero_centered_norm_shift", lambda: _flip("unshifted_norm"), min_delta=5e-2, metric="max_abs_diff"),
        FaultInjectionCase("dropped_chunk_sep_token", lambda: _flip("drop_sep_token"), min_delta=1e-2, metric="max_abs_diff"),
    ]
    results = run_fault_injection_suite(run_baseline, cases)
    assert len(results) == len(cases)
    for res in results:
        assert res.passed, f"Fault injection {res.name!r} failed: error={res.error}, delta={res.measured_delta}"


def test_fault_injection_tokenizer_special_tokens_and_dropped_sep(tmp_path: Path) -> None:
    tok = _make_cpu_tokenizer(tmp_path)

    fault_state = {"encode_special": True, "drop_marker": False}

    def baseline_ids() -> np.ndarray:
        tok._tokenizer.encode_special_tokens = fault_state["encode_special"]
        chunks = ["alpha beta", "gamma <|endoftext|> delta"]
        if fault_state["drop_marker"]:
            ids = tok.encode_document(" ".join(chunks))
        else:
            ids, _ = tok.encode_contextual_document(chunks, return_spans=True)
        # Pad/fix length for suite comparison
        arr = np.full((12,), tok.pad_id, dtype=np.int32)
        arr[: min(12, len(ids))] = ids[:12]
        return arr

    @contextmanager
    def _set_fault(key: str, val: object) -> Iterator[None]:
        old = fault_state[key]
        fault_state[key] = val
        try:
            yield
        finally:
            fault_state[key] = old

    cases = [
        FaultInjectionCase("special_tokens_injection_allowed", lambda: _set_fault("encode_special", False), min_delta=1.0, metric="max_abs_diff"),
        FaultInjectionCase("dropped_chunk_sep_marker", lambda: _set_fault("drop_marker", True), min_delta=1.0, metric="max_abs_diff"),
    ]
    results = run_fault_injection_suite(baseline_ids, cases)
    for res in results:
        assert res.passed, f"Tokenizer fault {res.name!r} failed: error={res.error}, delta={res.measured_delta}"
