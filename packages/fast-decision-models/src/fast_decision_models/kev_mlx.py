"""Metal MLX typed-question scorer for `jaredpalmer/kev-0.8b` (`Qwen3.5-0.8B` + `PointerHead`)."""

from __future__ import annotations

from functools import partial
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

from ._common import softmax_vec
from .paths import resolve_kev_dir

_KEV_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# Metal kernel source for Qwen3.5 `GatedDeltaNet` (`Dk=128`, `Dv % 8 == 0` packed butterfly).
# Uses `acc_f32` output name for the FP32 recurrent state accumulator per `fast-ir-guard` contract.
_GDN_PACKED_BTREE_SRC = r"""
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = Dk / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;

    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);

    auto lane = thread_index_in_simdgroup;
    auto row_in_simdgroup = lane / lanes_per_row;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto row_group = thread_position_in_grid.y;
    auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

    auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
    y += (b_idx * T * Hv + hv_idx) * Dv;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    auto o_state = acc_f32 + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) {
        state[i] = static_cast<float>(i_state[i]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
        float gt = static_cast<float>(g_[hv_idx]);
        float part[partials_per_lane];
        for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] * gt;
                acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
        }
        float kv_mem =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);

        auto delta =
            (static_cast<float>(v_[dv_idx]) - kv_mem) *
            static_cast<float>(beta_[hv_idx]);

        for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int e = pb * 4 + i;
                state[e] = state[e] + static_cast<float>(k_[e]) * delta;
                acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
        }
        float out =
            ((part[0] + part[1]) + (part[2] + part[3])) +
            ((part[4] + part[5]) + (part[6] + part[7]));
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
        }

        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    for (int i = 0; i < values_per_lane; ++i) {
        o_state[i] = static_cast<StT>(state[i]);
    }
"""


def _kev_render(v: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {_kev_render(x, indent + 1).lstrip()}" for x in v)
    if isinstance(v, Mapping):
        return "\n".join(
            f"{pad}{k}:\n{_kev_render(x, indent + 1)}"
            if isinstance(x, (Mapping, list))
            else f"{pad}{k}: {_kev_render(x)}"
            for k, x in v.items()
        )
    return str(v)


def _build_qwen35_backbone(
    text_cfg: Mapping[str, Any],
    weights: dict[str, Any],
    *,
    quant_cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Build self-contained MLX `Qwen3_5TextModel` (`W8` quantized or `BF16`, batched left-pad ready)."""
    import mlx.core as mx
    import mlx.nn as nn

    gdn_packed_kernel = mx.fast.metal_kernel(
        name="kev_gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "acc_f32"],
        source=_GDN_PACKED_BTREE_SRC,
    )

    @partial(mx.compile, shapeless=True)
    def _fused_swiglu(gate: Any, up: Any) -> Any:
        return nn.silu(gate) * up

    @partial(mx.compile, shapeless=True)
    def _fused_precise_gate(xn: Any, gate: Any) -> Any:
        gf = nn.silu(gate.astype(mx.float32))
        xf = xn.astype(mx.float32)
        return (gf * xf).astype(xn.dtype)

    @partial(mx.compile, shapeless=True)
    def _fused_compute_g(A_log: Any, a: Any, dt_bias: Any) -> Any:
        return mx.exp(
            -mx.exp(A_log.astype(mx.float32))
            * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
        )

    hidden_size = int(text_cfg["hidden_size"])
    intermediate_size = int(text_cfg["intermediate_size"])
    num_layers = int(text_cfg["num_hidden_layers"])
    num_q_heads = int(text_cfg["num_attention_heads"])
    num_kv_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg.get("head_dim") or (hidden_size // num_q_heads))
    rms_eps = float(text_cfg.get("rms_norm_eps", 1e-6))
    vocab_size = int(text_cfg["vocab_size"])
    fa_interval = int(text_cfg.get("full_attention_interval", 4))
    num_v_heads = int(text_cfg["linear_num_value_heads"])
    num_k_heads = int(text_cfg["linear_num_key_heads"])
    head_k_dim = int(text_cfg["linear_key_head_dim"])
    head_v_dim = int(text_cfg["linear_value_head_dim"])
    conv_k = int(text_cfg.get("linear_conv_kernel_dim", 4))
    rope_p = text_cfg.get("rope_parameters") or {}
    rope_theta = float(rope_p.get("rope_theta", 10000000.0))
    partial_rotary = float(rope_p.get("partial_rotary_factor", 0.25))
    rot_dims = int(head_dim * partial_rotary)
    half_rot = rot_dims // 2
    inv_freq = mx.array(
        [1.0 / (rope_theta ** (float(i) / float(rot_dims))) for i in range(0, rot_dims, 2)],
        dtype=mx.float32,
    )

    def _apply_partial_rope(x: Any, pos_2d: Any) -> Any:
        angles = (pos_2d.astype(mx.float32)[:, None, :, None]) * inv_freq[None, None, None, :]
        cos_a = mx.cos(angles).astype(x.dtype)
        sin_a = mx.sin(angles).astype(x.dtype)
        x_rot, x_pass = mx.split(x, [rot_dims], axis=-1)
        x0, x1 = mx.split(x_rot, [half_rot], axis=-1)
        y0 = x0 * cos_a - x1 * sin_a
        y1 = x1 * cos_a + x0 * sin_a
        return mx.concatenate([y0, y1, x_pass], axis=-1)

    class _DeltaCache:
        def __init__(self, conv_state: Any = None, rec_state: Any = None) -> None:
            self.conv_state = conv_state
            self.rec_state = rec_state

        def select_rows(self, row_indices: Any) -> "_DeltaCache":
            cs = mx.take(self.conv_state, row_indices, axis=0) if self.conv_state is not None else None
            rs = mx.take(self.rec_state, row_indices, axis=0) if self.rec_state is not None else None
            return _DeltaCache(cs, rs)

    class _FullAttnCache:
        def __init__(self, keys: Any = None, values: Any = None, offset: int = 0) -> None:
            self.keys = keys
            self.values = values
            self.offset = offset

        def update_and_fetch(self, k_new: Any, v_new: Any) -> tuple[Any, Any]:
            if self.keys is None:
                self.keys = k_new
                self.values = v_new
            else:
                self.keys = mx.concatenate([self.keys, k_new], axis=2)
                self.values = mx.concatenate([self.values, v_new], axis=2)
            self.offset += int(k_new.shape[2])
            return self.keys, self.values

        def select_rows(self, row_indices: Any) -> "_FullAttnCache":
            ks = mx.take(self.keys, row_indices, axis=0) if self.keys is not None else None
            vs = mx.take(self.values, row_indices, axis=0) if self.values is not None else None
            return _FullAttnCache(ks, vs, self.offset)

    class _RMSNormGated(nn.Module):
        def __init__(self, d: int, eps: float) -> None:
            super().__init__()
            self.eps = eps
            self.weight = mx.ones((d,), dtype=mx.bfloat16)

        def __call__(self, x: Any, gate: Any) -> Any:
            xn = mx.fast.rms_norm(x, self.weight, self.eps)
            return _fused_precise_gate(xn, gate)

    class _GatedDeltaLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.key_dim = head_k_dim * num_k_heads
            self.value_dim = head_v_dim * num_v_heads
            self.conv_dim = self.key_dim * 2 + self.value_dim
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=False,
                kernel_size=conv_k,
                groups=self.conv_dim,
                padding=0,
            )
            self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
            self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
            self.dt_bias = mx.ones((num_v_heads,), dtype=mx.bfloat16)
            self.A_log = mx.zeros((num_v_heads,), dtype=mx.bfloat16)
            self.norm = _RMSNormGated(head_v_dim, rms_eps)
            self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
            self._inv_scale = head_k_dim**-0.5
            self._rms_eps = 1e-6 * (self._inv_scale**2)

        def __call__(self, x: Any, cache: _DeltaCache, valid_token_scale: Any | None = None) -> Any:
            B, S, _ = x.shape
            qkv = self.in_proj_qkv(x)
            if valid_token_scale is not None:
                qkv = qkv * valid_token_scale
            z = self.in_proj_z(x).reshape(B, S, num_v_heads, head_v_dim)
            b = self.in_proj_b(x)
            a = self.in_proj_a(x)

            conv_state = (
                cache.conv_state
                if cache.conv_state is not None
                else mx.zeros((B, conv_k - 1, self.conv_dim), dtype=x.dtype)
            )
            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            cache.conv_state = mx.contiguous(conv_in[:, -(conv_k - 1) :, :])
            conv_out = nn.silu(self.conv1d(conv_in))

            q_raw, k_raw, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
            q = q_raw.reshape(B, S, num_k_heads, head_k_dim)
            k = k_raw.reshape(B, S, num_k_heads, head_k_dim)
            v = v.reshape(B, S, num_v_heads, head_v_dim)

            s_q = mx.array(self._inv_scale**2, dtype=q.dtype)
            s_k = mx.array(self._inv_scale, dtype=k.dtype)
            q = mx.fast.rms_norm(q, None, self._rms_eps) * s_q
            k = mx.fast.rms_norm(k, None, self._rms_eps) * s_k

            beta = mx.sigmoid(b)
            if valid_token_scale is not None:
                # Zeroing beta on left-pad slots keeps initial rec_state == 0.0 bit-identical to unpadded B=1.
                beta = beta * valid_token_scale
            g = _fused_compute_g(self.A_log, a, self.dt_bias)
            rec_state = (
                cache.rec_state
                if cache.rec_state is not None
                else mx.zeros((B, num_v_heads, head_v_dim, head_k_dim), dtype=mx.float32)
            )
            T = int(k.shape[1])
            y, new_state = gdn_packed_kernel(
                inputs=[q, k, v, g, beta, rec_state, T],
                template=[
                    ("InT", q.dtype),
                    ("StT", mx.float32),
                    ("Dk", head_k_dim),
                    ("Dv", head_v_dim),
                    ("Hk", num_k_heads),
                    ("Hv", num_v_heads),
                ],
                grid=(32, head_v_dim // 8, B * num_v_heads),
                threadgroup=(32, 2, 1),
                output_shapes=[(B, T, num_v_heads, head_v_dim), rec_state.shape],
                output_dtypes=[q.dtype, mx.float32],
            )
            cache.rec_state = new_state
            out = self.norm(y, z)
            return self.out_proj(out.reshape(B, S, -1))

    class _GatedFullAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = head_dim**-0.5
            self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim * 2, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
            self.q_norm = nn.RMSNorm(head_dim, eps=rms_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=rms_eps)
            self.rope = nn.RoPE(rot_dims, traditional=False, base=rope_theta)

        def __call__(
            self,
            x: Any,
            cache: _FullAttnCache,
            attn_mask: Any = "causal",
            pos_2d: Any | None = None,
        ) -> Any:
            B, L, _ = x.shape
            qp = self.q_proj(x)
            queries, gate = mx.split(qp.reshape(B, L, num_q_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, L, -1)
            keys = self.k_proj(x).reshape(B, L, num_kv_heads, head_dim)
            values = self.v_proj(x).reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)

            if pos_2d is not None:
                queries = _apply_partial_rope(queries, pos_2d)
                keys = _apply_partial_rope(keys, pos_2d)
            else:
                off = cache.offset
                queries = self.rope(queries, offset=off)
                keys = self.rope(keys, offset=off)
            keys, values = cache.update_and_fetch(keys, values)

            eff_mask = (None if L == 1 else "causal") if attn_mask == "causal" else attn_mask
            out = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=eff_mask,
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(out * mx.sigmoid(gate))

    class _SwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        def __call__(self, x: Any) -> Any:
            return self.down_proj(_fused_swiglu(self.gate_proj(x), self.up_proj(x)))

    class _DecoderBlock(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.is_linear = (idx + 1) % fa_interval != 0
            if self.is_linear:
                self.linear_attn = _GatedDeltaLayer()
            else:
                self.self_attn = _GatedFullAttention()
            self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.mlp = _SwiGLU()

        def __call__(
            self,
            x: Any,
            cache: Any,
            *,
            valid_token_scale: Any | None = None,
            attn_mask: Any = "causal",
            pos_2d: Any | None = None,
        ) -> Any:
            if self.is_linear:
                h = x + self.linear_attn(self.input_layernorm(x), cache, valid_token_scale=valid_token_scale)
            else:
                h = x + self.self_attn(self.input_layernorm(x), cache, attn_mask=attn_mask, pos_2d=pos_2d)
            return h + self.mlp(self.post_attention_layernorm(h))

    class _Qwen35Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = [_DecoderBlock(i) for i in range(num_layers)]
            self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)

        def make_cache(self) -> list[Any]:
            return [_DeltaCache() if blk.is_linear else _FullAttnCache() for blk in self.layers]

        @staticmethod
        def select_cache_rows(cache: list[Any], row_indices: Any) -> list[Any]:
            return [c.select_rows(row_indices) for c in cache]

        def __call__(
            self,
            input_ids: Any,
            cache: list[Any],
            *,
            valid_token_scale: Any | None = None,
            attn_mask: Any = "causal",
            pos_2d: Any | None = None,
        ) -> Any:
            h = self.embed_tokens(input_ids)
            for idx in range(len(self.layers)):
                h = self.layers[idx](
                    h,
                    cache[idx],
                    valid_token_scale=valid_token_scale,
                    attn_mask=attn_mask,
                    pos_2d=pos_2d,
                )
            return self.norm(h)

    backbone = _Qwen35Backbone()
    if quant_cfg is not None:
        nn.quantize(
            backbone,
            group_size=int(quant_cfg.get("group_size", 64)),
            bits=int(quant_cfg.get("bits", 8)),
        )
    model_items = [(k[len("model.") :], v) for k, v in weights.items() if k.startswith("model.")]
    backbone.load_weights(model_items, strict=True)
    backbone.eval()
    mx.eval(backbone.parameters())
    return backbone


class Kev08bMlxBackend:
    """Metal MLX `Kev-0.8B r15` runner with `W8` quantization + left-padded `B=8` cross-state batching."""

    def __init__(
        self,
        kev_dir: Path | str | None = None,
        *,
        quant_mode: Literal["auto", "w8", "bf16"] = "auto",
        batch_docs: int = 8,
        max_state: int = 1024,
        max_branch: int = 2048,
    ) -> None:
        import mlx.core as mx
        import numpy as np
        from tokenizers import Tokenizer

        self._mx = mx
        self._np = np
        self.kev_dir = Path(kev_dir).expanduser().resolve() if kev_dir is not None else resolve_kev_dir()
        self.batch_docs = max(1, int(batch_docs))
        self.max_state = int(max_state)
        self.max_branch = int(max_branch)

        w8_sf = self.kev_dir / "model_w8.safetensors"
        w8_meta = self.kev_dir / "kev_meta_w8.json"
        bf16_sf = self.kev_dir / "model.safetensors"
        bf16_meta = self.kev_dir / "kev_meta.json"
        tok_p = self.kev_dir / "tokenizer.json"

        use_w8 = (quant_mode == "w8") or (quant_mode == "auto" and w8_sf.exists() and w8_meta.exists())
        self.quant_mode: Literal["w8", "bf16"] = "w8" if use_w8 else "bf16"
        sf_p = w8_sf if use_w8 else bf16_sf
        meta_p = w8_meta if use_w8 else bf16_meta
        if not (meta_p.exists() and sf_p.exists() and tok_p.exists()):
            raise FileNotFoundError(
                f"Missing Kev-0.8B artifacts in {self.kev_dir}. Run 'decision-models setup-weights --build-kev-mlx'."
            )

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        self.quantization = meta.get("quantization")
        self.native_temperature = float(meta["native_temperature"])
        self.special_ids = [int(x) for x in meta["special_ids"]]
        self.pad_id = int(meta["pad_id"])

        self.tok = Tokenizer.from_file(str(tok_p))
        self.tok.no_truncation()
        self.tok.no_padding()

        mx.set_cache_limit(1 << 30)
        weights = mx.load(str(sf_p))
        self.q_w = np.asarray(weights["pointer_head.q_weight"].astype(mx.float32))
        self.q_b = np.asarray(weights["pointer_head.q_bias"].astype(mx.float32))
        self.k_w = np.asarray(weights["pointer_head.k_weight"].astype(mx.float32))
        self.k_b = np.asarray(weights["pointer_head.k_bias"].astype(mx.float32))
        self.ptr_scale = 1.0 / math.sqrt(float(self.q_w.shape[0]))

        self.backbone = _build_qwen35_backbone(meta["text_config"], weights, quant_cfg=self.quantization)
        del weights
        mx.clear_cache()

    def __enter__(self) -> "Kev08bMlxBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _user_tokens(self, text: str) -> list[int]:
        return list(self.tok.encode(_KEV_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids)

    def encode_questions(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        """Encode one shared state prefix plus all typed branch questions (`noul`, `choice`, `score`)."""
        s_toks = self._user_tokens(_kev_render(state))
        state_ids = [self.special_ids[0]] + s_toks[: self.max_state - 1]
        q_id, o_id, c_id, d_id = self.special_ids[1:]

        rows: list[dict[str, Any]] = []
        for qdef in questions:
            t = str(qdef["type"])
            crit = qdef.get("criteria")
            if t == "noul":
                cm = crit if isinstance(crit, Mapping) else {}
                f_desc = cm.get("false")
                t_desc = cm.get("true")
                opts = [
                    "no" if not f_desc else f"no: {_kev_render(f_desc)}",
                    "yes" if not t_desc else f"yes: {_kev_render(t_desc)}",
                ]
            elif t == "choice":
                if isinstance(crit, list):
                    opts = [str(x) for x in crit]
                else:
                    opts = [k if not v else f"{k}: {_kev_render(v)}" for k, v in (crit or {}).items()]
            else:
                opts = [_kev_render(x) for x in (crit or ())]

            instr = [q_id] + self._user_tokens(_kev_render(qdef.get("instructions")))
            spans = [[o_id] + self._user_tokens(o) + [c_id] for o in opts]
            br = instr + [tid for sp in spans for tid in sp] + [d_id]
            if len(state_ids) + len(br) > self.max_branch:
                br = br[: self.max_branch - len(state_ids) - 1] + [d_id]
            ends: list[int] = []
            cursor = len(instr)
            for sp in spans:
                cursor += len(sp)
                ends.append(min(cursor - 1, len(br) - 2))
            rows.append({"ids": br, "decide": len(br) - 1, "opts": ends, "k": len(opts)})
        return state_ids, rows

    def _pointer_logits_and_probs(
        self,
        h_row_f32: Any,
        decide_idx: int,
        opt_indices: list[int],
        *,
        temperature: float | None = None,
    ) -> tuple[list[float], list[float]]:
        h_dec = h_row_f32[decide_idx]
        h_opt = h_row_f32[opt_indices]
        q_vec = h_dec @ self.q_w.T + self.q_b
        k_mat = h_opt @ self.k_w.T + self.k_b
        raw_z = [float(x) for x in ((k_mat @ q_vec) * self.ptr_scale).tolist()]
        temp = self.native_temperature if temperature is None else float(temperature)
        probs = softmax_vec(raw_z, temp)
        return probs, raw_z

    def _run_batched_pointer(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
        temperature: float | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        """Batch `B` document states via left-padded hybrid prefill + parallel branch pass."""
        if not items:
            return []
        mx = self._mx
        np = self._np
        eff_b = max(1, int(batch_docs or self.batch_docs))

        encoded_all = [self.encode_questions(st, qs) for st, qs in items]
        order = sorted(range(len(items)), key=lambda idx: len(encoded_all[idx][0]))
        out_by_orig: list[list[tuple[list[float], list[float]]]] = [[] for _ in items]

        for wave_start in range(0, len(order), eff_b):
            wave_orig = order[wave_start : wave_start + eff_b]
            B = len(wave_orig)

            if B == 1:
                orig_0 = wave_orig[0]
                state_ids, rows = encoded_all[orig_0]
                if not rows:
                    out_by_orig[orig_0] = []
                    continue
                cache = self.backbone.make_cache()
                h_st = self.backbone(mx.array([state_ids], dtype=mx.int32), cache)
                mx.eval(h_st)

                br_cache = (
                    cache
                    if len(rows) == 1
                    else self.backbone.select_cache_rows(cache, mx.zeros((len(rows),), dtype=mx.int32))
                )
                max_l = max(len(r["ids"]) for r in rows)
                ids_mx = mx.array(
                    [r["ids"] + [self.pad_id] * (max_l - len(r["ids"])) for r in rows],
                    dtype=mx.int32,
                )
                h_mx = self.backbone(ids_mx, br_cache)
                mx.eval(h_mx)
                h_np = np.asarray(h_mx.astype(mx.float32))
                out_by_orig[orig_0] = [
                    self._pointer_logits_and_probs(h_np[ri], r["decide"], r["opts"], temperature=temperature)
                    for ri, r in enumerate(rows)
                ]
                continue

            state_lists = [encoded_all[idx][0] for idx in wave_orig]
            s_lens = [len(s) for s in state_lists]
            s_max = max(s_lens)
            pad_lens = [s_max - ln for ln in s_lens]

            st_np = np.full((B, s_max), self.pad_id, dtype=np.int32)
            valid_np = np.zeros((B, s_max, 1), dtype=np.float32)
            pos_st_np = np.zeros((B, s_max), dtype=np.int32)
            mask_st_np = np.full((B, 1, s_max, s_max), -1e9, dtype=np.float32)

            for b_i in range(B):
                pl = pad_lens[b_i]
                ln = s_lens[b_i]
                st_np[b_i, pl:] = state_lists[b_i]
                valid_np[b_i, pl:, 0] = 1.0
                pos_st_np[b_i, pl:] = np.arange(ln, dtype=np.int32)
                for q_t in range(pl, s_max):
                    mask_st_np[b_i, 0, q_t, pl : q_t + 1] = 0.0

            cache = self.backbone.make_cache()
            h_st = self.backbone(
                mx.array(st_np, dtype=mx.int32),
                cache,
                valid_token_scale=mx.array(valid_np).astype(mx.bfloat16),
                attn_mask=mx.array(mask_st_np).astype(mx.bfloat16),
                pos_2d=mx.array(pos_st_np, dtype=mx.int32),
            )
            mx.eval(h_st)

            flat_rows: list[dict[str, Any]] = []
            owner_b: list[int] = []
            doc_slices: list[tuple[int, int]] = []
            for b_i, orig_i in enumerate(wave_orig):
                _, rows = encoded_all[orig_i]
                st_idx = len(flat_rows)
                flat_rows.extend(rows)
                owner_b.extend([b_i] * len(rows))
                doc_slices.append((st_idx, len(flat_rows)))

            if not flat_rows:
                continue

            N_br = len(flat_rows)
            max_br = max(len(r["ids"]) for r in flat_rows)
            br_np = np.full((N_br, max_br), self.pad_id, dtype=np.int32)
            pos_br_np = np.zeros((N_br, max_br), dtype=np.int32)
            mask_br_np = np.full((N_br, 1, max_br, s_max + max_br), -1e9, dtype=np.float32)

            for r_i, r in enumerate(flat_rows):
                b_i = owner_b[r_i]
                pl = pad_lens[b_i]
                sl = s_lens[b_i]
                bl = len(r["ids"])
                br_np[r_i, :bl] = r["ids"]
                pos_br_np[r_i] = sl + np.arange(max_br, dtype=np.int32)
                for q_t in range(max_br):
                    mask_br_np[r_i, 0, q_t, pl : s_max + q_t + 1] = 0.0

            br_cache = self.backbone.select_cache_rows(cache, mx.array(owner_b, dtype=mx.int32))
            h_br = self.backbone(
                mx.array(br_np, dtype=mx.int32),
                br_cache,
                valid_token_scale=None,
                attn_mask=mx.array(mask_br_np).astype(mx.bfloat16),
                pos_2d=mx.array(pos_br_np, dtype=mx.int32),
            )
            mx.eval(h_br)
            h_br_np = np.asarray(h_br.astype(mx.float32))

            flat_outs = [
                self._pointer_logits_and_probs(h_br_np[ri], r["decide"], r["opts"], temperature=temperature)
                for ri, r in enumerate(flat_rows)
            ]
            for b_i, orig_i in enumerate(wave_orig):
                lo, hi = doc_slices[b_i]
                out_by_orig[orig_i] = flat_outs[lo:hi]

        return out_by_orig

    def score_multi_states(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
        temperature: float | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        """Score typed questions (`noul`, `choice`, `score`) and return `(probs, raw_logits)` per question."""
        return self._run_batched_pointer(items, batch_docs=batch_docs, temperature=temperature)

    def score_state_noul(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> list[tuple[float, list[float]]]:
        """Single-state binary (`noul`) fast path returning `(P(true), raw_logits)`."""
        return self.score_multi_states_noul([(state, questions)])[0]

    def score_multi_states_noul(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
    ) -> list[list[tuple[float, list[float]]]]:
        """Multi-state binary (`noul`) batch path returning `(P(true), raw_logits)` per question."""
        raw = self._run_batched_pointer(items, batch_docs=batch_docs)
        return [
            [(probs[1] if len(probs) >= 2 else probs[0], logits) for probs, logits in doc_res]
            for doc_res in raw
        ]

    def close(self) -> None:
        """Release model weights and clear the MLX buffer cache."""
        self.backbone = None
        self._mx.clear_cache()


__all__ = ["Kev08bMlxBackend"]
