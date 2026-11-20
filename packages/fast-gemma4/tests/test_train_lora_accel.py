"""Unit and parity tests for Gemma-4 LoRA/QLoRA training accelerations (P0.3, P1.5, P1.6, P1.7).

Runs self-contained on synthetic mini-Gemma4 models (SWA + Full + ProportionalRoPE + PLE +
KV-sharing + logit softcap + MoE) with zero dependency on external checkpoints or ``mlx_lm``.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import pytest

from fast_gemma4.lora_accel import (
    CompiledOptimizerStep,
    QuantizedMoELoRALinear,
    apply_rope_with_table,
    build_packed_groups,
    build_rope_table,
    build_shared_prefix_packed_groups,
    cast_lora_adapters_dtype,
    chunked_softcap_ce_loss,
    compute_effective_ckpt_tokens,
    compute_packed_tail_ce_loss,
    find_shared_prefix_len,
    forward_gemma4_layer,
    grad_tree_cosine,
    pack_examples_to_bin,
    pack_suffixes_for_shared_prefix,
    quantize_frozen_trunk_8bit,
    run_frozen_prefix_packed,
    run_shared_prefix_grad_accum_step,
    unwrap_kv_shared_lora_modules,
)


# ---------------------------------------------------------------------------
# Self-Contained Mini-Gemma4 Fixture (SWA + Full + PLE + KV-Sharing + LoRA)
# ---------------------------------------------------------------------------


class _RMSNormNoScale(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class _ProportionalRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        *,
        traditional: bool = False,
        base: float = 1000000.0,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / float(dims)
        self._freqs = mx.concatenate(
            [
                base**exponents,
                mx.full(((dims - rotated_dims) // 2,), mx.inf, dtype=mx.float32),
            ]
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class _MiniLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int = 4,
        scale: float = 2.0,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.linear = base
        out_dim, in_dim = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            in_dim = in_dim * 32 // base.bits
        self.scale = float(scale)
        init_s = 1.0 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(
            low=-init_s, high=init_s, shape=(in_dim, rank), dtype=dtype
        )
        self.lora_b = mx.zeros((rank, out_dim), dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + mx.array(self.scale, dtype=x.dtype) * z.astype(x.dtype)


class _MiniMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class _MiniAttention(nn.Module):
    def __init__(self, hidden_size: int, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.is_sliding = layer_type == "sliding_attention"
        self.n_heads = 2
        self.n_kv_heads = 1
        self.head_dim = 64 if self.is_sliding else 128
        self.scale = 1.0
        self.use_k_eq_v = False

        self.q_proj = nn.Linear(hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = _RMSNormNoScale(eps=1e-6)
        if self.is_sliding:
            self.rope: nn.Module = nn.RoPE(self.head_dim, traditional=False, base=10000.0)
        else:
            self.rope = _ProportionalRoPE(
                dims=self.head_dim, rotated_dims=32, traditional=False, base=1000000.0
            )


class _MiniDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ple_dim: int,
        layer_type: str,
        layer_idx: int,
        is_kv_shared: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.enable_moe = False
        self.self_attn = _MiniAttention(hidden_size, layer_type)
        mlp_dim = intermediate_size * (2 if is_kv_shared else 1)
        self.mlp = _MiniMLP(hidden_size, mlp_dim)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.pre_feedforward_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_feedforward_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)

        self.per_layer_input_gate = nn.Linear(hidden_size, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, hidden_size, bias=False)
        self.post_per_layer_input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.layer_scalar = mx.array([0.9 + 0.03 * layer_idx], dtype=mx.bfloat16)


class _MiniGemma4TextModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 512,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        ple_dim: int = 64,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = ple_dim
        self.embed_scale = hidden_size**0.5
        self.layer_types = [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.num_hidden_layers = len(self.layer_types)
        first_kv_shared = 5  # L0..4 store KV (L3=last sliding, L4=last full); L5..7 share KV.
        self.previous_kvs = [0, 1, 2, 3, 4, 3, 3, 4]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens_per_layer = nn.Embedding(
            vocab_size, self.num_hidden_layers * ple_dim
        )
        self.embed_tokens_per_layer_scale = ple_dim**0.5
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_projection_scale = hidden_size**-0.5
        self.per_layer_model_projection = nn.Linear(
            hidden_size, self.num_hidden_layers * ple_dim, bias=False
        )
        self.per_layer_projection_norm = nn.RMSNorm(ple_dim, eps=1e-6)
        self.layers = [
            _MiniDecoderLayer(
                hidden_size,
                intermediate_size,
                ple_dim,
                ltype,
                idx,
                is_kv_shared=(idx >= first_kv_shared),
            )
            for idx, ltype in enumerate(self.layer_types)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def _get_per_layer_inputs(
        self, input_ids: mx.array, _input_embeddings: mx.array | None = None
    ) -> mx.array:
        res = self.embed_tokens_per_layer(input_ids) * mx.array(
            self.embed_tokens_per_layer_scale, dtype=mx.bfloat16
        )
        return mx.unflatten(
            res, -1, (self.num_hidden_layers, self.hidden_size_per_layer_input)
        )

    def _project_per_layer_inputs(
        self, h: mx.array, per_layer_inputs: mx.array
    ) -> mx.array:
        proj = self.per_layer_model_projection(h) * mx.array(
            self.per_layer_projection_scale, dtype=h.dtype
        )
        proj = mx.unflatten(
            proj, -1, (self.num_hidden_layers, self.hidden_size_per_layer_input)
        )
        proj = self.per_layer_projection_norm(proj)
        return (proj + per_layer_inputs) * mx.array(
            self.per_layer_input_scale, dtype=h.dtype
        )


class _MiniLanguageModel(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = _MiniGemma4TextModel(**kwargs)
        self.tie_word_embeddings = True
        self.final_logit_softcapping = 30.0


class _MiniGemma4Wrapper(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.language_model = _MiniLanguageModel(**kwargs)

    @property
    def layers(self) -> list[_MiniDecoderLayer]:
        return self.language_model.model.layers


def _cast_module_bf16(module: nn.Module) -> None:
    flat = tree_flatten(module.parameters())
    casted = [
        (k, v.astype(mx.bfloat16) if v.dtype == mx.float32 else v) for k, v in flat
    ]
    module.update(tree_unflatten(casted))


def _attach_mini_lora(
    model: _MiniGemma4Wrapper,
    *,
    first_lora_layer: int = 2,
    rank: int = 4,
    scale: float = 2.0,
    nonzero_b: bool = False,
) -> int:
    model.freeze()
    for layer in model.layers[first_lora_layer:]:
        for name, sub in list(layer.named_modules()):
            if isinstance(sub, (nn.Linear, nn.QuantizedLinear)):
                parts = name.split(".")
                parent: Any = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                lora_mod = _MiniLoRALinear(sub, rank=rank, scale=scale, dtype=mx.bfloat16)
                if nonzero_b:
                    lora_mod.lora_b = (
                        mx.random.normal(lora_mod.lora_b.shape) * 0.02
                    ).astype(mx.bfloat16)
                setattr(parent, parts[-1], lora_mod)
    return unwrap_kv_shared_lora_modules(model)


def _make_mini_model(
    *,
    seed: int = 123,
    first_lora_layer: int = 2,
    qlora_8bit: bool = False,
    nonzero_b: bool = False,
) -> _MiniGemma4Wrapper:
    mx.random.seed(seed)
    model = _MiniGemma4Wrapper()
    tm = model.language_model.model
    # Match Gemma-4 weight scale (initializer_range=0.02, bounded q/k norm & residual post-norms)
    # so synthetic tests mirror real transformer conditioning instead of saturated random softmax.
    tm.embed_tokens.weight = (mx.random.normal(tm.embed_tokens.weight.shape) * 0.02).astype(
        mx.bfloat16
    )
    tm.embed_tokens_per_layer.weight = (
        mx.random.normal(tm.embed_tokens_per_layer.weight.shape) * 0.02
    ).astype(mx.bfloat16)
    tm.per_layer_model_projection.weight = (
        mx.random.normal(tm.per_layer_model_projection.weight.shape) * 0.02
    ).astype(mx.bfloat16)
    for layer in tm.layers:
        layer.self_attn.q_norm.weight = mx.full(
            layer.self_attn.q_norm.weight.shape, 0.35, dtype=mx.bfloat16
        )
        layer.self_attn.k_norm.weight = mx.full(
            layer.self_attn.k_norm.weight.shape, 0.35, dtype=mx.bfloat16
        )
        layer.post_attention_layernorm.weight = mx.full(
            layer.post_attention_layernorm.weight.shape, 0.25, dtype=mx.bfloat16
        )
        layer.post_feedforward_layernorm.weight = mx.full(
            layer.post_feedforward_layernorm.weight.shape, 0.25, dtype=mx.bfloat16
        )
        layer.post_per_layer_input_norm.weight = mx.full(
            layer.post_per_layer_input_norm.weight.shape, 0.25, dtype=mx.bfloat16
        )
        for _, sub in layer.named_modules():
            if isinstance(sub, nn.Linear):
                sub.weight = (mx.random.normal(sub.weight.shape) * 0.02).astype(
                    mx.bfloat16
                )
    _cast_module_bf16(model)
    mx.eval(model.parameters())
    if qlora_8bit:
        quantize_frozen_trunk_8bit(
            model, group_size=64, bits=8, quantize_ple_embedding=True
        )
        mx.eval(model.parameters())
    mx.random.seed(seed + 1000)
    _attach_mini_lora(
        model, first_lora_layer=first_lora_layer, rank=4, scale=2.0, nonzero_b=nonzero_b
    )
    mx.eval(model.parameters())
    return model


def _unpacked_single_doc_loss(
    model: _MiniGemma4Wrapper,
    full_ids: list[int],
    prompt_len: int,
    first_lora_layer: int,
) -> tuple[mx.array, Any]:
    """Reference unpacked B=1 forward+backward using native ``attn.rope(offset=0)``."""
    tm = model.language_model.model
    lm = model.language_model
    inp = mx.array([full_ids[:-1]], dtype=mx.int32)
    y_flat = mx.array(full_ids[prompt_len:], dtype=mx.int32)
    seq_len = int(inp.shape[1])

    input_embeddings = tm.embed_tokens(inp)
    h0 = input_embeddings * mx.array(tm.embed_scale, dtype=input_embeddings.dtype)
    pli = tm._get_per_layer_inputs(inp, input_embeddings)
    pli = tm._project_per_layer_inputs(h0, pli)
    pli_list = [pli[:, :, i, :] for i in range(len(tm.layers))]

    q_idx = mx.arange(seq_len, dtype=mx.int32)[:, None]
    k_idx = mx.arange(seq_len, dtype=mx.int32)[None, :]
    zero = mx.array(0.0, dtype=h0.dtype)
    neg = mx.array(-65504.0, dtype=h0.dtype)
    m_full = mx.where(q_idx >= k_idx, zero, neg)[None, None, :, :]
    m_swa = mx.where(
        (q_idx >= k_idx) & (q_idx < k_idx + tm.window_size), zero, neg
    )[None, None, :, :]
    masks = [
        m_swa if layer.layer_type == "sliding_attention" else m_full
        for layer in tm.layers
    ]

    inter: list[tuple[Any, Any]] = [(None, None)] * len(tm.layers)
    h = h0
    for idx in range(first_lora_layer):
        kvs, off = inter[tm.previous_kvs[idx]]
        h, kvs, off = forward_gemma4_layer(
            tm.layers[idx],
            h,
            masks[idx],
            per_layer_input=pli_list[idx],
            shared_kv=kvs,
            offset=off or 0,
        )
        inter[idx] = (kvs, off)

    def _upper_loss(m: _MiniGemma4Wrapper) -> mx.array:
        local_inter = list(inter)
        hx = h
        for idx in range(first_lora_layer, len(tm.layers)):
            kvs, off = local_inter[tm.previous_kvs[idx]]
            hx, kvs, off = forward_gemma4_layer(
                m.language_model.model.layers[idx],
                hx,
                masks[idx],
                per_layer_input=pli_list[idx],
                shared_kv=kvs,
                offset=off or 0,
            )
            local_inter[idx] = (kvs, off)
        h_norm = m.language_model.model.norm(hx)
        h_ans = h_norm[0, prompt_len - 1 : seq_len, :]
        return chunked_softcap_ce_loss(
            m.language_model.model.embed_tokens.as_linear,
            h_ans,
            y_flat,
            softcap=lm.final_logit_softcapping,
            compiled=False,
        )

    loss, grads = nn.value_and_grad(model, _upper_loss)(model)
    mx.eval(loss, grads)
    return loss, grads


def _unpacked_batched_loss(
    model: _MiniGemma4Wrapper,
    examples: list[tuple[list[int], int]],
    first_lora_layer: int,
) -> tuple[mx.array, Any]:
    """Reference right-padded B=K batched forward+backward (token-mean loss)."""
    tm = model.language_model.model
    lm = model.language_model
    max_in = max(len(ids) - 1 for ids, _ in examples)
    in_rows = [ids[:-1] + [0] * (max_in - (len(ids) - 1)) for ids, _ in examples]
    spans = [(p_len - 1, len(ids) - 1) for ids, p_len in examples]
    y_flat = mx.array([tok for ids, p_len in examples for tok in ids[p_len:]], dtype=mx.int32)
    inp = mx.array(in_rows, dtype=mx.int32)

    input_embeddings = tm.embed_tokens(inp)
    h0 = input_embeddings * mx.array(tm.embed_scale, dtype=input_embeddings.dtype)
    pli = tm._get_per_layer_inputs(inp, input_embeddings)
    pli = tm._project_per_layer_inputs(h0, pli)
    pli_list = [pli[:, :, i, :] for i in range(len(tm.layers))]

    q_idx = mx.arange(max_in, dtype=mx.int32)[:, None]
    k_idx = mx.arange(max_in, dtype=mx.int32)[None, :]
    zero = mx.array(0.0, dtype=h0.dtype)
    neg = mx.array(-65504.0, dtype=h0.dtype)
    m_full = mx.where(q_idx >= k_idx, zero, neg)[None, None, :, :]
    m_swa = mx.where(
        (q_idx >= k_idx) & (q_idx < k_idx + tm.window_size), zero, neg
    )[None, None, :, :]
    masks = [
        m_swa if layer.layer_type == "sliding_attention" else m_full
        for layer in tm.layers
    ]

    inter: list[tuple[Any, Any]] = [(None, None)] * len(tm.layers)
    h = h0
    for idx in range(first_lora_layer):
        kvs, off = inter[tm.previous_kvs[idx]]
        h, kvs, off = forward_gemma4_layer(
            tm.layers[idx],
            h,
            masks[idx],
            per_layer_input=pli_list[idx],
            shared_kv=kvs,
            offset=off or 0,
        )
        inter[idx] = (kvs, off)

    def _upper_loss(m: _MiniGemma4Wrapper) -> mx.array:
        local_inter = list(inter)
        hx = h
        for idx in range(first_lora_layer, len(tm.layers)):
            kvs, off = local_inter[tm.previous_kvs[idx]]
            hx, kvs, off = forward_gemma4_layer(
                m.language_model.model.layers[idx],
                hx,
                masks[idx],
                per_layer_input=pli_list[idx],
                shared_kv=kvs,
                offset=off or 0,
            )
            local_inter[idx] = (kvs, off)
        h_norm = m.language_model.model.norm(hx)
        h_ans = mx.concatenate([h_norm[bi, s0:s1, :] for bi, (s0, s1) in enumerate(spans)], axis=0)
        return chunked_softcap_ce_loss(
            m.language_model.model.embed_tokens.as_linear,
            h_ans,
            y_flat,
            softcap=lm.final_logit_softcapping,
            compiled=False,
        )

    loss, grads = nn.value_and_grad(model, _upper_loss)(model)
    mx.eval(loss, grads)
    return loss, grads


# ---------------------------------------------------------------------------
# P0.3 Tests: Compiled Softcap + Chunked CE & Compiled Optimizer Step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("softcap", [30.0, None])
@pytest.mark.parametrize("weighted", [False, True])
def test_compiled_softcap_ce_parity_and_gradients(
    softcap: float | None, weighted: bool
) -> None:
    mx.random.seed(11)
    n_ans, dim, vocab = 600, 64, 512  # Spans 3 chunks (256 + 256 + 88)
    w = (mx.random.normal((vocab, dim)) * 0.08).astype(mx.bfloat16)
    h = (mx.random.normal((n_ans, dim)) * 0.5).astype(mx.bfloat16)
    y = mx.random.randint(0, vocab, (n_ans,))
    weights = (
        mx.full((n_ans,), 1.0 / n_ans, dtype=mx.float32) * mx.linspace(0.5, 1.5, n_ans)
        if weighted
        else None
    )
    if weights is not None:
        weights = weights / weights.sum()

    def _head(x: mx.array) -> mx.array:
        return x @ w.T

    v_ref, g_ref = mx.value_and_grad(
        lambda h_: chunked_softcap_ce_loss(
            _head, h_, y, softcap=softcap, token_weights=weights, chunk_size=256, compiled=False
        )
    )(h)
    v_cmp, g_cmp = mx.value_and_grad(
        lambda h_: chunked_softcap_ce_loss(
            _head, h_, y, softcap=softcap, token_weights=weights, chunk_size=256, compiled=True
        )
    )(h)
    mx.eval(v_ref, g_ref, v_cmp, g_cmp)

    assert abs(float(v_ref.item()) - float(v_cmp.item())) < 1e-3
    gcos, _, _ = grad_tree_cosine({"h": g_ref}, {"h": g_cmp})
    assert gcos >= 0.9999


def test_compiled_optimizer_step_parity_across_lr_schedule() -> None:
    mx.random.seed(21)
    m_ref = nn.Linear(64, 32, bias=False)
    m_ref.weight = m_ref.weight.astype(mx.bfloat16)
    m_cmp = nn.Linear(64, 32, bias=False)
    m_cmp.weight = mx.array(m_ref.weight)

    opt_ref = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    opt_cmp = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    stepper = CompiledOptimizerStep(opt_cmp, m_cmp, max_grad_norm=1.0, compiled=True)

    lrs = [1e-4, 4e-4, 8e-4, 5e-4, 2e-4, 5e-5]
    for step_i, lr in enumerate(lrs):
        g1 = {"weight": (mx.random.normal((32, 64)) * (0.15 + 0.05 * step_i)).astype(mx.bfloat16)}
        g2 = {"weight": (mx.random.normal((32, 64)) * (0.15 + 0.05 * step_i)).astype(mx.bfloat16)}

        acc_ref = {"weight": (g1["weight"] + g2["weight"]) * mx.array(0.5, dtype=mx.bfloat16)}
        acc_ref, _ = optim.clip_grad_norm(acc_ref, max_norm=1.0)
        opt_ref.learning_rate = lr
        opt_ref.update(m_ref, acc_ref)

        acc_cmp = stepper.accumulate(None, g1)
        acc_cmp = stepper.accumulate(acc_cmp, g2)
        stepper.step(m_cmp, acc_cmp, lr=lr, accum_divisor=2)
        mx.eval(m_ref.parameters(), opt_ref.state, m_cmp.parameters(), opt_cmp.state)

    max_diff = float(
        mx.max(mx.abs(m_ref.weight.astype(mx.float32) - m_cmp.weight.astype(mx.float32))).item()
    )
    assert max_diff < 1e-4


# ---------------------------------------------------------------------------
# P1.5 Tests: Varlen Sequence Packing, RoPE Reset, SWA/Full Masks & PLE
# ---------------------------------------------------------------------------


def test_rope_table_exact_match_default_and_proportional() -> None:
    mx.random.seed(31)
    pos_single = mx.arange(20, dtype=mx.int32)[None, :]

    # Sliding nn.RoPE
    rope_s = nn.RoPE(64, traditional=False, base=10000.0)
    xs = mx.random.normal((1, 2, 20, 64)).astype(mx.bfloat16)
    ys_ref = rope_s(xs, offset=0)
    ys_tbl = apply_rope_with_table(xs, build_rope_table(rope_s, pos_single, mx.bfloat16))

    # Full ProportionalRoPE (inf unrotated tail)
    rope_f = _ProportionalRoPE(dims=128, rotated_dims=32, traditional=False, base=1000000.0)
    xf = mx.random.normal((1, 2, 20, 128)).astype(mx.bfloat16)
    yf_ref = rope_f(xf, offset=0)
    yf_tbl = apply_rope_with_table(xf, build_rope_table(rope_f, pos_single, mx.bfloat16))
    mx.eval(ys_ref, ys_tbl, yf_ref, yf_tbl)

    assert float(mx.max(mx.abs(ys_ref.astype(mx.float32) - ys_tbl.astype(mx.float32))).item()) == 0.0
    assert float(mx.max(mx.abs(yf_ref.astype(mx.float32) - yf_tbl.astype(mx.float32))).item()) == 0.0


def test_packed_mask_zero_cross_segment_leakage_full_and_swa() -> None:
    """Perturbing doc 0 and trailing pad tokens must produce 0.0 change on docs 1 & 2."""
    model = _make_mini_model(seed=41, first_lora_layer=2, nonzero_b=True)
    # Lengths 28, 22, 24 all exceed window_size=16 so both Full and SWA masks are active.
    doc0 = (list(range(10, 39)), 20)
    doc1 = (list(range(60, 83)), 15)
    doc2 = (list(range(120, 145)), 16)
    doc0_perturbed = ([300 + (i % 50) for i in range(29)], 20)

    bin_a = pack_examples_to_bin([doc0, doc1, doc2], pack_tokens=112, pad_to_fixed=True, pad_token_id=0)
    bin_b = pack_examples_to_bin(
        [doc0_perturbed, doc1, doc2], pack_tokens=112, pad_to_fixed=True, pad_token_id=7
    )

    def _run_hidden(pbin: Any) -> mx.array:
        h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(model, pbin, first_lora_layer=2)
        tm = model.language_model.model
        local_inter = list(inter)
        for idx in range(2, len(tm.layers)):
            layer = tm.layers[idx]
            kvs, _ = local_inter[tm.previous_kvs[idx]]
            h, kvs, off = forward_gemma4_layer(
                layer,
                h,
                masks[idx],
                per_layer_input=pli_list[idx],
                shared_kv=kvs,
                rope_table=rope_tables[layer.layer_type],
            )
            local_inter[idx] = (kvs, off)
        return tm.norm(h)

    ha = _run_hidden(bin_a)
    hb = _run_hidden(bin_b)
    mx.eval(ha, hb)

    # Doc 0 spans [0..28); Docs 1 & 2 span [28..74).
    diff_doc0 = float(mx.max(mx.abs(ha[0, :28].astype(mx.float32) - hb[0, :28].astype(mx.float32))).item())
    diff_docs12 = float(mx.max(mx.abs(ha[0, 28:74].astype(mx.float32) - hb[0, 28:74].astype(mx.float32))).item())
    assert diff_doc0 > 0.1
    assert diff_docs12 == 0.0


@pytest.mark.parametrize("nonzero_b", [False, True])
def test_packed_vs_unpacked_loss_and_grad_parity_token_norm(nonzero_b: bool) -> None:
    model = _make_mini_model(seed=51, first_lora_layer=2, nonzero_b=nonzero_b)
    examples = [
        ([((i * 17 + 3) % 480) + 1 for i in range(30)], 19),  # len_in=29 > SWA=16, ans=11
        ([((i * 23 + 7) % 480) + 1 for i in range(25)], 18),  # len_in=24 > SWA=16, ans=7
        ([((i * 31 + 11) % 480) + 1 for i in range(21)], 12), # len_in=20 > SWA=16, ans=9
    ]
    ref_loss, ref_grads = _unpacked_batched_loss(model, examples, first_lora_layer=2)

    pbin = pack_examples_to_bin(examples, pack_tokens=96, pad_to_fixed=True, loss_norm="token")
    assert pbin.bin_tokens == 96
    assert pbin.real_tokens == 29 + 24 + 20
    h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(model, pbin, first_lora_layer=2)
    vg = nn.value_and_grad(model, compute_packed_tail_ce_loss)
    p_loss, p_grads = vg(
        model,
        h,
        masks,
        inter,
        pli_list,
        rope_tables,
        pbin.ans_indices,
        pbin.y_flat,
        pbin.token_weights,
        2,
    )
    mx.eval(p_loss, p_grads)

    d_loss = abs(float(ref_loss.item()) - float(p_loss.item()))
    gcos, min_cos, n_cmp = grad_tree_cosine(ref_grads, p_grads)
    assert d_loss < 1e-3, f"d_loss={d_loss}"
    assert gcos >= 0.999, f"global grad cos={gcos}"
    assert min_cos >= 0.999, f"min tensor grad cos={min_cos}"
    assert n_cmp >= (30 if not nonzero_b else 60)


def test_packed_example_norm_matches_grad_accum_b1_no_lr_inflation() -> None:
    """Packing K docs with unequal answer lengths under loss_norm='example' must match 1/K * sum(B=1 grads)."""
    model = _make_mini_model(seed=61, first_lora_layer=2, nonzero_b=True)
    examples = [
        ([((i * 13 + 5) % 480) + 1 for i in range(26)], 22),  # 4 answer tokens
        ([((i * 19 + 9) % 480) + 1 for i in range(32)], 20),  # 12 answer tokens
        ([((i * 29 + 2) % 480) + 1 for i in range(38)], 14),  # 24 answer tokens
    ]
    k_docs = len(examples)
    acc_ref_grads: Any = None
    ref_losses: list[float] = []
    for ids, p_len in examples:
        l_i, g_i = _unpacked_single_doc_loss(model, ids, p_len, first_lora_layer=2)
        ref_losses.append(float(l_i.item()))
        acc_ref_grads = (
            g_i if acc_ref_grads is None else tree_map(mx.add, acc_ref_grads, g_i)
        )
    inv_k = mx.array(1.0 / float(k_docs), dtype=mx.float32)
    ref_grads = tree_map(lambda g: g * inv_k.astype(g.dtype), acc_ref_grads)
    ref_loss_val = sum(ref_losses) / float(k_docs)
    mx.eval(ref_grads)

    pbin = pack_examples_to_bin(examples, pack_tokens=128, pad_to_fixed=True, loss_norm="example")
    assert pbin.token_weights is not None
    assert abs(float(pbin.token_weights.sum().item()) - 1.0) < 1e-6
    h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(model, pbin, first_lora_layer=2)
    p_loss, p_grads = nn.value_and_grad(model, compute_packed_tail_ce_loss)(
        model,
        h,
        masks,
        inter,
        pli_list,
        rope_tables,
        pbin.ans_indices,
        pbin.y_flat,
        pbin.token_weights,
        2,
    )
    mx.eval(p_loss, p_grads)

    assert abs(ref_loss_val - float(p_loss.item())) < 1e-3
    gcos, min_cos, _ = grad_tree_cosine(ref_grads, p_grads)
    assert gcos >= 0.999
    assert min_cos >= 0.998

    # Verify gradient norm is NOT inflated by K:
    ref_norm = math.sqrt(
        sum(float((v.astype(mx.float32) ** 2).sum().item()) for _, v in tree_flatten(ref_grads))
    )
    p_norm = math.sqrt(
        sum(float((v.astype(mx.float32) ** 2).sum().item()) for _, v in tree_flatten(p_grads))
    )
    assert 0.99 <= (p_norm / max(ref_norm, 1e-12)) <= 1.01


def test_build_packed_groups_handles_oversized_and_preserves_all_items() -> None:
    items = [([i] * (40 + (i % 7) * 15), 20) for i in range(25)]
    items.append(([999] * 300, 150))  # Oversized > pack_tokens=128
    groups = build_packed_groups(items, pack_tokens=128, seed=7)
    flat = [it for g in groups for it in g]
    assert len(flat) == len(items)
    oversized_bins = [g for g in groups if any(len(ids) == 300 for ids, _ in g)]
    assert len(oversized_bins) == 1 and len(oversized_bins[0]) == 1
    assert len(groups) < len(items)  # Actually packed multiple docs per bin


# ---------------------------------------------------------------------------
# P1.6 Tests: 8-bit g64 QLoRA Trunk, Auto ckpt-tokens & MoE gather_qmm QLoRA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nonzero_b", [False, True])
def test_qlora_8bit_step0_and_nonzero_parity_and_invariants(nonzero_b: bool) -> None:
    m_bf16 = _make_mini_model(seed=71, first_lora_layer=2, qlora_8bit=False, nonzero_b=nonzero_b)
    m_q8 = _make_mini_model(seed=71, first_lora_layer=2, qlora_8bit=True, nonzero_b=nonzero_b)

    # Invariant checks: base linears + PLE embedding are quantized & frozen; LoRA (A, B) stay BF16 & trainable;
    # KV-shared layers 5..7 have unwrapped QuantizedLinear k_proj/v_proj (no dead LoRA adapters).
    tm_q8 = m_q8.language_model.model
    assert isinstance(tm_q8.embed_tokens_per_layer, nn.QuantizedEmbedding)
    for idx in (5, 6, 7):
        assert isinstance(tm_q8.layers[idx].self_attn.k_proj, nn.QuantizedLinear)
        assert isinstance(tm_q8.layers[idx].self_attn.v_proj, nn.QuantizedLinear)
    for k, v in tree_flatten(m_q8.trainable_parameters()):
        assert "lora_a" in k or "lora_b" in k, f"Non-LoRA trainable param: {k}"
        assert v.dtype == mx.bfloat16, f"Expected BF16 adapter {k}, got {v.dtype}"

    examples = [
        ([((i * 11 + 2) % 480) + 1 for i in range(28)], 18),
        ([((i * 19 + 4) % 480) + 1 for i in range(24)], 15),
    ]
    pbin = pack_examples_to_bin(examples, pack_tokens=64, pad_to_fixed=True, loss_norm="token")

    def _run_packed(m: _MiniGemma4Wrapper) -> tuple[mx.array, Any]:
        h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(m, pbin, first_lora_layer=2)
        loss, grads = nn.value_and_grad(m, compute_packed_tail_ce_loss)(
            m, h, masks, inter, pli_list, rope_tables, pbin.ans_indices, pbin.y_flat, None, 2
        )
        mx.eval(loss, grads)
        return loss, grads

    l_bf16, g_bf16 = _run_packed(m_bf16)
    l_q8, g_q8 = _run_packed(m_q8)
    d_loss = abs(float(l_bf16.item()) - float(l_q8.item()))
    gcos, _, n_cmp = grad_tree_cosine(g_bf16, g_q8)
    assert d_loss < 5e-3, f"8-bit QLoRA loss drift={d_loss}"
    assert gcos >= 0.995, f"8-bit QLoRA grad cosine={gcos}"
    assert n_cmp >= (30 if not nonzero_b else 60)

    # Auto --ckpt-tokens logic
    assert compute_effective_ckpt_tokens(None, qlora_8bit=False) == 2200
    assert compute_effective_ckpt_tokens(None, qlora_8bit=True) == 4096
    assert compute_effective_ckpt_tokens(3072, qlora_8bit=True) == 3072


@pytest.mark.parametrize("nonzero_b", [False, True])
def test_moe_gather_qmm_qlora_parity_step0_and_nonzero(nonzero_b: bool) -> None:
    mx.random.seed(81)
    num_experts, d_out, d_in, rank = 8, 128, 256, 8
    bsz, seq_len, k_top = 4, 8, 2
    w_dense = (mx.random.normal((num_experts, d_out, d_in)) * 0.05).astype(mx.bfloat16)
    qmoe = QuantizedMoELoRALinear.from_dense(
        w_dense, rank=rank, scale=2.0, dropout=0.0, group_size=64, bits=8
    )
    if nonzero_b:
        qmoe.lora_b = (mx.random.normal(qmoe.lora_b.shape) * 0.02).astype(mx.bfloat16)
    mx.eval(qmoe.parameters())

    # Only lora_a and lora_b should be trainable
    train_keys = {k for k, _ in tree_flatten(qmoe.trainable_parameters())}
    assert train_keys == {"lora_a", "lora_b"}

    x = mx.random.normal((bsz, seq_len, d_in)).astype(mx.bfloat16)
    expert_idx = mx.random.randint(0, num_experts, (bsz, seq_len, k_top))

    def _qmoe_loss(m: QuantizedMoELoRALinear, x_in: mx.array) -> mx.array:
        return (m(x_in, expert_idx).astype(mx.float32) ** 2).mean()

    def _dense_loss(params: dict[str, mx.array], x_in: mx.array) -> mx.array:
        la, lb = params["lora_a"], params["lora_b"]
        m_tok = bsz * seq_len
        x_f = x_in.reshape((m_tok, d_in))
        idx_f = expert_idx.reshape((m_tok, k_top))
        rows = []
        for t in range(m_tok):
            ks = []
            for k_i in range(k_top):
                e = int(idx_f[t, k_i].item())
                xt = x_f[t : t + 1]
                yb = xt @ w_dense[e].T
                zl = (xt @ la[e].T) @ lb[e].T
                ks.append(yb + mx.array(2.0, dtype=mx.bfloat16) * zl)
            rows.append(mx.stack(ks, axis=1))
        y = mx.concatenate(rows, axis=0).reshape((bsz, seq_len, k_top, d_out))
        return (y.astype(mx.float32) ** 2).mean()

    vq, gq = nn.value_and_grad(qmoe, _qmoe_loss)(qmoe, x)
    vd, gd = mx.value_and_grad(_dense_loss)(
        {"lora_a": mx.array(qmoe.lora_a), "lora_b": mx.array(qmoe.lora_b)}, x
    )
    mx.eval(vq, gq, vd, gd)

    assert abs(float(vq.item()) - float(vd.item())) < 1e-3
    gcos, min_cos, n_cmp = grad_tree_cosine(gq, gd)
    assert gcos >= 0.995
    assert min_cos >= 0.995
    assert n_cmp == (2 if nonzero_b else 1)


# ---------------------------------------------------------------------------
# P1.7 Tests: Differentiable Shared-Prefix KV Broadcast Across grad_accum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nonzero_b", [False, True])
def test_shared_sys_kv_exact_parity_across_grad_accum_swa_and_full(
    nonzero_b: bool,
) -> None:
    """Two-stage VJP and single-bin packed-suffix shared-sys-kv must match independent micro-batches."""
    model = _make_mini_model(seed=91, first_lora_layer=2, nonzero_b=nonzero_b)
    # sys_len=24 > window_size=16 so SWA prefix tail slicing is strictly exercised!
    sys_prefix = [((i * 7 + 13) % 450) + 1 for i in range(24)]
    examples = [
        (sys_prefix + [((i * 11 + 5) % 450) + 1 for i in range(20)], 34),  # suf_prompt=10, ans=10
        (sys_prefix + [((i * 17 + 9) % 450) + 1 for i in range(26)], 38),  # suf_prompt=14, ans=12
        (sys_prefix + [((i * 23 + 3) % 450) + 1 for i in range(18)], 31),  # suf_prompt=7,  ans=11
    ]
    detected_sys = find_shared_prefix_len(examples, min_len=8)
    assert detected_sys == 24

    # Independent B=1 reference averaged over K=3 micro-batches
    ref_losses: list[float] = []
    acc_ref: Any = None
    for ids, p_len in examples:
        l_i, g_i = _unpacked_single_doc_loss(model, ids, p_len, first_lora_layer=2)
        ref_losses.append(float(l_i.item()))
        acc_ref = g_i if acc_ref is None else tree_map(mx.add, acc_ref, g_i)
    inv_k = mx.array(1.0 / len(examples), dtype=mx.float32)
    ref_grads = tree_map(lambda g: g * inv_k.astype(g.dtype), acc_ref)
    ref_loss = sum(ref_losses) / len(examples)
    mx.eval(ref_grads)

    # 1. Multi-microbatch Two-Stage VJP (K=3 separate suffix micro-batches sharing sys_ids)
    suf_bins_k3 = [
        pack_suffixes_for_shared_prefix([ex], sys_len=detected_sys, loss_norm="token")
        for ex in examples
    ]
    sh_loss, sh_grads, per_l = run_shared_prefix_grad_accum_step(
        model, sys_prefix, suf_bins_k3, first_lora_layer=2, ckpt_tokens=4096
    )
    assert len(per_l) == 3
    assert abs(ref_loss - sh_loss) < 1e-3
    gcos, min_cos, n_cmp = grad_tree_cosine(ref_grads, sh_grads)
    assert gcos >= 0.999, f"Two-stage shared-sys-kv grad_cos={gcos}"
    assert min_cos >= 0.999, f"Two-stage shared-sys-kv min_cos={min_cos}"
    assert n_cmp >= (30 if not nonzero_b else 60)

    # 2. Single packed-suffix bin over shared prefix (P1.5 + P1.7 composed, example-norm)
    packed_suf = pack_suffixes_for_shared_prefix(
        examples, sys_len=detected_sys, pack_tokens=96, pad_to_fixed=True, loss_norm="example"
    )
    sp_loss, sp_grads, _ = run_shared_prefix_grad_accum_step(
        model, sys_prefix, [packed_suf], first_lora_layer=2, ckpt_tokens=4096
    )
    assert abs(ref_loss - sp_loss) < 1e-3
    gcos_p, min_cos_p, _ = grad_tree_cosine(ref_grads, sp_grads)
    assert gcos_p >= 0.999
    assert min_cos_p >= 0.998


def test_shared_sys_kv_when_all_storing_layers_frozen() -> None:
    """When first_lora_layer > last_storing_layer (storing_indices == []), gradients remain exact."""
    model = _make_mini_model(seed=101, first_lora_layer=5, nonzero_b=True)
    sys_prefix = [((i * 5 + 19) % 450) + 1 for i in range(20)]
    examples = [
        (sys_prefix + [((i * 13 + 7) % 450) + 1 for i in range(18)], 28),
        (sys_prefix + [((i * 19 + 3) % 450) + 1 for i in range(22)], 31),
    ]
    ref_l1, ref_g1 = _unpacked_single_doc_loss(model, examples[0][0], examples[0][1], first_lora_layer=5)
    ref_l2, ref_g2 = _unpacked_single_doc_loss(model, examples[1][0], examples[1][1], first_lora_layer=5)
    half = mx.array(0.5, dtype=mx.float32)
    ref_grads = tree_map(lambda a, b: (a + b) * half.astype(a.dtype), ref_g1, ref_g2)
    ref_loss = 0.5 * (float(ref_l1.item()) + float(ref_l2.item()))
    mx.eval(ref_grads)

    suf_bins = [
        pack_suffixes_for_shared_prefix([ex], sys_len=20, loss_norm="token")
        for ex in examples
    ]
    sh_loss, sh_grads, _ = run_shared_prefix_grad_accum_step(
        model, sys_prefix, suf_bins, first_lora_layer=5, ckpt_tokens=4096
    )
    assert abs(ref_loss - sh_loss) < 1e-3
    gcos, min_cos, _ = grad_tree_cosine(ref_grads, sh_grads)
    assert gcos >= 0.999
    assert min_cos >= 0.999


def test_find_shared_prefix_len_and_grouping_boundaries() -> None:
    # Identical sequences must cap sys_len at min(p_len) - 1 so answer predictor stays in suffix!
    same = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert find_shared_prefix_len([(same, 6), (same, 8)], min_len=4) == 5
    # Disjoint prefixes fall back cleanly to 0
    assert find_shared_prefix_len([([1] * 40, 30), ([2] * 40, 30)], min_len=8) == 0

    # Shared-prefix grouping packs all items without losing any example
    sys_a = [100 + i for i in range(40)]
    sys_b = [200 + i for i in range(40)]
    items = [(sys_a + [i] * 30, 55) for i in range(6)] + [
        (sys_b + [i] * 25, 50) for i in range(6)
    ]
    groups = build_shared_prefix_packed_groups(
        items, pack_tokens=120, min_shared_prefix=16, grad_accum=2, seed=5
    )
    assert sum(len(g) for g in groups) == len(items)


def test_train_lora_cli_exposes_accel_flags() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    out = subprocess.check_output([sys.executable, str(script), "--help"], text=True)
    for flag in (
        "--pack-tokens",
        "--loss-norm",
        "--qlora-8bit",
        "--shared-sys-kv",
        "--no-compile",
    ):
        assert flag in out, f"Missing CLI flag {flag} in train_lora.py --help"
