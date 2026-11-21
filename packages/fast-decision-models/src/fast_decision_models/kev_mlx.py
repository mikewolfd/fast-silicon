"""Metal MLX typed-question scorer for `jaredpalmer/kev-0.8b` (`Qwen3.5-0.8B` + `PointerHead`)."""

from __future__ import annotations

from collections import Counter
from functools import partial
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Literal, Mapping, Sequence

from ._common import softmax_vec
from .paths import resolve_kev_dir
from .prefix_tree import (
    DEFAULT_CHUNK_TOKENS,
    PrefixTree,
    recommended_prefix_budget_bytes,
    snapshot_boundaries,
)

_KEV_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
_KEV_CACHE_LIMIT_BYTES = 512 << 20
_KEV_CLEAR_CACHE_THRESHOLD_BYTES = 256 << 20
KEV_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)

# Metal kernel source for Qwen3.5 `GatedDeltaNet` (`Dk=128`, `Dv % 8 == 0` packed butterfly).
# Reads prefix `state_in` indirectly via `owner_idx[b_idx]` (zero `N_br`-way FP32 GDN state cloning!)
# and skips allocating/writing `acc_f32` during branch scoring when `WriteState == false`.
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

    auto src_b = static_cast<uint>(owner_idx[b_idx]);
    auto src_n = src_b * Hv + hv_idx;
    auto i_state = state_in + (src_n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

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

    if constexpr (WriteState) {
        auto o_state = acc_f32 + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        for (int i = 0; i < values_per_lane; ++i) {
            o_state[i] = static_cast<StT>(state[i]);
        }
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


def resolve_kev_special_ids(
    tok: Any,
    delimiters: Sequence[str] = KEV_SPECIAL_TOKENS,
) -> list[int]:
    """Resolve the 5 Kev delimiter token IDs from a Rust `Tokenizer`."""
    ids = [tok.token_to_id(t) for t in delimiters]
    if len(ids) != 5 or any(tid is None for tid in ids):
        raise ValueError(f"Kev requires all five decision delimiters in its tokenizer: {list(delimiters)!r}")
    return [int(tid) for tid in ids]


def _encode_kev_user_tokens(tok: Any, text: str) -> list[int]:
    return list(tok.encode(_KEV_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids)


def encode_kev_questions(
    tok: Any,
    state: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    special_ids: Sequence[int] | None = None,
    max_state_tokens: int = 1024,
    max_branch_tokens: int = 2048,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Pure-Python Kev-0.8B question encoder (zero MLX weight load; single source of truth).

    Encodes one shared state prefix plus all typed branch questions (`noul`, `choice`, `score`).
    When ``special_ids`` is omitted, resolves ``KEV_SPECIAL_TOKENS`` directly from ``tok``.
    """
    if special_ids is None:
        s_ids = resolve_kev_special_ids(tok)
    else:
        s_ids = [int(x) for x in special_ids]
        if len(s_ids) != 5:
            raise ValueError(f"Expected 5 Kev special_ids, got {len(s_ids)}")
    max_state = int(max_state_tokens)
    max_branch = int(max_branch_tokens)
    if max_state < 1:
        raise ValueError(f"max_state_tokens must be >= 1, got {max_state_tokens!r}")
    if max_branch < 3:
        raise ValueError(f"max_branch_tokens must be >= 3, got {max_branch_tokens!r}")

    s_toks = _encode_kev_user_tokens(tok, _kev_render(state))
    state_ids = [s_ids[0]] + s_toks[: max_state - 1]
    q_id, o_id, c_id, d_id = s_ids[1:]

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

        instr = [q_id] + _encode_kev_user_tokens(tok, _kev_render(qdef.get("instructions")))
        spans = [[o_id] + _encode_kev_user_tokens(tok, o) + [c_id] for o in opts]
        br = instr + [tid for sp in spans for tid in sp] + [d_id]
        if len(state_ids) + len(br) > max_branch:
            br = br[: max(2, max_branch - len(state_ids) - 1)] + [d_id]
        ends: list[int] = []
        cursor = len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(min(cursor - 1, len(br) - 2))
        rows.append({"ids": br, "decide": len(br) - 1, "opts": ends, "k": len(opts)})
    return state_ids, rows


def _parse_ver_tuple(ver_str: str) -> tuple[int, ...]:
    parts: list[int] = []
    for tok in str(ver_str).split("."):
        m = re.match(r"(\d+)", tok)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _detect_apple10_d256_force_fused(mx: Any) -> bool:
    """Return True only on Apple10 (M5+) + macOS >= 26.2 + MLX >= 0.32.2 supporting `force_fused` D256 SDPA."""
    if os.environ.get("MLX_METAL_NO_NAX") == "1":
        return False
    try:
        dev = mx.device_info()
    except Exception:
        return False
    dev_name = str(dev.get("device_name", "")).lower()
    arch = str(dev.get("architecture", "")).lower()
    is_m5 = ("m5" in dev_name) or ("apple10" in arch) or ("g17" in arch)
    if not is_m5:
        return False
    mac_ver = _parse_ver_tuple(platform.mac_ver()[0])
    if mac_ver and mac_ver < (26, 2):
        return False
    mlx_ver = _parse_ver_tuple(getattr(mx, "__version__", "0.0.0"))
    if mlx_ver < (0, 32, 2):
        return False
    doc = getattr(mx.fast.scaled_dot_product_attention, "__doc__", "") or ""
    return "force_fused" in doc


def _build_state_attn_mask(mx: Any, pad_lens: Sequence[int], s_max: int, act_dtype: Any) -> Any:
    """Vectorized additive `(B, 1, s_max, s_max)` causal left-pad mask directly in `act_dtype` (zero bool SDPA mask)."""
    if all(pl == 0 for pl in pad_lens):
        return "causal"
    pl = mx.array(list(pad_lens), dtype=mx.int32)[:, None, None, None]
    q_pos = mx.arange(s_max, dtype=mx.int32)[None, None, :, None]
    k_pos = mx.arange(s_max, dtype=mx.int32)[None, None, None, :]
    allow = (q_pos >= pl) & (k_pos >= pl) & (k_pos <= q_pos)
    return mx.where(allow, mx.array(0.0, dtype=act_dtype), mx.array(-1e9, dtype=act_dtype))


def _build_branch_attn_mask(mx: Any, owner_pads: Sequence[int], s_max: int, max_br: int, act_dtype: Any) -> Any:
    """Vectorized additive `(N_br, 1, max_br, s_max + max_br)` branch mask directly in `act_dtype`."""
    if all(pl == 0 for pl in owner_pads):
        return "causal"
    pl = mx.array(list(owner_pads), dtype=mx.int32)[:, None, None, None]
    q_pos = mx.arange(max_br, dtype=mx.int32)[None, None, :, None] + s_max
    k_pos = mx.arange(s_max + max_br, dtype=mx.int32)[None, None, None, :]
    allow = (k_pos >= pl) & (k_pos <= q_pos)
    return mx.where(allow, mx.array(0.0, dtype=act_dtype), mx.array(-1e9, dtype=act_dtype))


def _build_qwen35_backbone(
    text_cfg: Mapping[str, Any],
    weights: dict[str, Any],
    *,
    quant_cfg: Mapping[str, Any] | None = None,
    use_d256_force_fused: bool | None = None,
) -> Any:
    """Build self-contained MLX `Qwen3_5TextModel` (`W8` quantized or `BF16`, batched left-pad ready)."""
    import mlx.core as mx
    import mlx.nn as nn

    gdn_packed_kernel = mx.fast.metal_kernel(
        name="kev_gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "owner_idx", "T"],
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
    allow_d256_force_fused = (
        _detect_apple10_d256_force_fused(mx) if use_d256_force_fused is None else bool(use_d256_force_fused)
    )

    @partial(mx.compile, shapeless=True)
    def _precompute_partial_rope_trig(pos_2d: Any) -> tuple[Any, Any]:
        angles = (pos_2d.astype(mx.float32)[:, None, :, None]) * inv_freq[None, None, None, :]
        return mx.cos(angles).astype(mx.bfloat16), mx.sin(angles).astype(mx.bfloat16)

    @partial(mx.compile, shapeless=True)
    def _fused_rot_pair(x0: Any, x1: Any, cos_a: Any, sin_a: Any) -> tuple[Any, Any]:
        c = cos_a.astype(x0.dtype)
        s = sin_a.astype(x0.dtype)
        return x0 * c - x1 * s, x1 * c + x0 * s

    def _apply_partial_rope_trig(x: Any, cos_a: Any, sin_a: Any) -> Any:
        x_rot, x_pass = mx.split(x, [rot_dims], axis=-1)
        x0, x1 = mx.split(x_rot, [half_rot], axis=-1)
        y0, y1 = _fused_rot_pair(x0, x1, cos_a, sin_a)
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

        def update_and_fetch(
            self,
            k_new: Any,
            v_new: Any,
            *,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> tuple[Any, Any]:
            if self.keys is None:
                if write_state:
                    self.keys = k_new
                    self.values = v_new
                    self.offset += int(k_new.shape[2])
                return k_new, v_new

            if owner_idx is None or owner_is_identity:
                k_pre = self.keys
                v_pre = self.values
            elif int(self.keys.shape[0]) == 1:
                b_cur = int(k_new.shape[0])
                k_pre = mx.broadcast_to(self.keys, (b_cur, *self.keys.shape[1:]))
                v_pre = mx.broadcast_to(self.values, (b_cur, *self.values.shape[1:]))
            else:
                k_pre = mx.take(self.keys, owner_idx, axis=0)
                v_pre = mx.take(self.values, owner_idx, axis=0)

            k_cat = mx.concatenate([k_pre, k_new], axis=2)
            v_cat = mx.concatenate([v_pre, v_new], axis=2)
            if write_state:
                self.keys = k_cat
                self.values = v_cat
                self.offset += int(k_new.shape[2])
            return k_cat, v_cat

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

        def __call__(
            self,
            x: Any,
            cache: _DeltaCache,
            valid_token_scale: Any | None = None,
            *,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            B, S, _ = x.shape
            qkv = self.in_proj_qkv(x)
            if valid_token_scale is not None:
                qkv = qkv * valid_token_scale
            z = self.in_proj_z(x).reshape(B, S, num_v_heads, head_v_dim)
            b = self.in_proj_b(x)
            a = self.in_proj_a(x)

            if cache.conv_state is None:
                conv_state = mx.zeros((B, conv_k - 1, self.conv_dim), dtype=x.dtype)
            elif owner_idx is None or owner_is_identity:
                conv_state = cache.conv_state
            elif int(cache.conv_state.shape[0]) == 1:
                conv_state = mx.broadcast_to(cache.conv_state, (B, *cache.conv_state.shape[1:]))
            else:
                conv_state = mx.take(cache.conv_state, owner_idx, axis=0)

            conv_in = mx.concatenate([conv_state, qkv], axis=1)
            if write_state:
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
            if cache.rec_state is not None:
                rec_state = cache.rec_state
                eff_owner = owner_idx if owner_idx is not None else mx.arange(B, dtype=mx.int32)
            else:
                rec_state = mx.zeros((B, num_v_heads, head_v_dim, head_k_dim), dtype=mx.float32)
                eff_owner = mx.arange(B, dtype=mx.int32)
            T = int(k.shape[1])
            state_out_shape = (B, num_v_heads, head_v_dim, head_k_dim) if write_state else (1,)
            y, new_state = gdn_packed_kernel(
                inputs=[q, k, v, g, beta, rec_state, eff_owner, T],
                template=[
                    ("InT", q.dtype),
                    ("StT", mx.float32),
                    ("Dk", head_k_dim),
                    ("Dv", head_v_dim),
                    ("Hk", num_k_heads),
                    ("Hv", num_v_heads),
                    ("WriteState", bool(write_state)),
                ],
                grid=(32, head_v_dim // 8, B * num_v_heads),
                threadgroup=(32, 2, 1),
                output_shapes=[(B, T, num_v_heads, head_v_dim), state_out_shape],
                output_dtypes=[q.dtype, mx.float32],
            )
            if write_state:
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
            rope_trig: tuple[Any, Any] | None = None,
            *,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            B, L, _ = x.shape
            qp = self.q_proj(x)
            queries, gate = mx.split(qp.reshape(B, L, num_q_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, L, -1)
            keys = self.k_proj(x).reshape(B, L, num_kv_heads, head_dim)
            values = self.v_proj(x).reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)

            if rope_trig is not None:
                cos_a, sin_a = rope_trig
                queries = _apply_partial_rope_trig(queries, cos_a, sin_a)
                keys = _apply_partial_rope_trig(keys, cos_a, sin_a)
            else:
                off = cache.offset
                queries = self.rope(queries, offset=off)
                keys = self.rope(keys, offset=off)
            keys, values = cache.update_and_fetch(
                keys,
                values,
                owner_idx=owner_idx,
                owner_is_identity=owner_is_identity,
                write_state=write_state,
            )

            eff_mask = (None if L == 1 else "causal") if attn_mask == "causal" else attn_mask
            if allow_d256_force_fused and head_dim == 256 and 256 <= L < 1024:
                try:
                    out = mx.fast.scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        scale=self.scale,
                        mask=eff_mask,
                        force_fused=True,
                    )
                except Exception:
                    out = mx.fast.scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        scale=self.scale,
                        mask=eff_mask,
                    )
            else:
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
            rope_trig: tuple[Any, Any] | None = None,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            if self.is_linear:
                h = x + self.linear_attn(
                    self.input_layernorm(x),
                    cache,
                    valid_token_scale=valid_token_scale,
                    owner_idx=owner_idx,
                    owner_is_identity=owner_is_identity,
                    write_state=write_state,
                )
            else:
                h = x + self.self_attn(
                    self.input_layernorm(x),
                    cache,
                    attn_mask=attn_mask,
                    rope_trig=rope_trig,
                    owner_idx=owner_idx,
                    owner_is_identity=owner_is_identity,
                    write_state=write_state,
                )
            return h + self.mlp(self.post_attention_layernorm(h))

    class _Qwen35Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = [_DecoderBlock(i) for i in range(num_layers)]
            self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.use_d256_force_fused = allow_d256_force_fused

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
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            h = self.embed_tokens(input_ids)
            rope_trig = _precompute_partial_rope_trig(pos_2d) if pos_2d is not None else None
            for idx in range(len(self.layers)):
                h = self.layers[idx](
                    h,
                    cache[idx],
                    valid_token_scale=valid_token_scale,
                    attn_mask=attn_mask,
                    rope_trig=rope_trig,
                    owner_idx=owner_idx,
                    owner_is_identity=owner_is_identity,
                    write_state=write_state,
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
    """Metal MLX `Kev-0.8B r15` runner with `W8` quantization, left-padded `B=8` batching, and hit-aware `PrefixTree`."""

    @staticmethod
    def _resolve_prefix_tree(
        *,
        prefix_tree: PrefixTree | None,
        prefix_cache_mb: int | Literal["auto"] | None,
        prefix_chunk_tokens: int,
        host_mem_bytes: int,
    ) -> tuple[PrefixTree | None, bool]:
        if prefix_tree is not None:
            return prefix_tree, False
        if prefix_cache_mb == "auto":
            return (
                PrefixTree(
                    chunk_tokens=int(prefix_chunk_tokens),
                    byte_budget=recommended_prefix_budget_bytes(host_mem_bytes),
                ),
                True,
            )
        if prefix_cache_mb is not None and not isinstance(prefix_cache_mb, bool) and int(prefix_cache_mb) > 0:
            return (
                PrefixTree(
                    chunk_tokens=int(prefix_chunk_tokens),
                    byte_budget=int(prefix_cache_mb) << 20,
                ),
                True,
            )
        return None, False

    def __init__(
        self,
        kev_dir: Path | str | None = None,
        *,
        quant_mode: Literal["auto", "w8", "bf16"] = "auto",
        batch_docs: int = 8,
        max_state: int = 1024,
        max_branch: int = 2048,
        max_branch_batch: int | None = None,
        prefix_tree: PrefixTree | None = None,
        prefix_cache_mb: int | Literal["auto"] | None = "auto",
        prefix_chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        prefix_snapshot_stride: int = 1,
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
        self.mlx_enable_tf32 = os.environ.get("MLX_ENABLE_TF32", "1")
        self._prev_cache_limit: int | None = None
        self._prev_wired_limit: int | None = None
        self._limits_restored = False

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

        host_mem_bytes = 0
        try:
            dev_info = mx.device_info()
            host_mem_bytes = int(dev_info.get("memory_size", 0))
            max_rec = int(dev_info.get("max_recommended_working_set_size", 0))
            set_wired = getattr(mx, "set_wired_limit", None)
            if callable(set_wired) and max_rec > 0:
                wired_target = min(max_rec, int(host_mem_bytes * 0.72)) if 0 < host_mem_bytes <= (24 << 30) else max_rec
                self._prev_wired_limit = int(set_wired(wired_target))
        except Exception:
            self._prev_wired_limit = None

        set_cache = getattr(mx, "set_cache_limit", None)
        if callable(set_cache):
            try:
                self._prev_cache_limit = int(set_cache(_KEV_CACHE_LIMIT_BYTES))
            except Exception:
                self._prev_cache_limit = None

        default_max_br = 16 if (0 < host_mem_bytes <= (24 << 30)) else 32
        self.max_branch_batch = max(1, int(max_branch_batch or default_max_br))
        self.prefix_snapshot_stride = max(1, int(prefix_snapshot_stride))
        self.prefix_tree: PrefixTree | None
        self.prefix_tree, self._owns_prefix_tree = self._resolve_prefix_tree(
            prefix_tree=prefix_tree,
            prefix_cache_mb=prefix_cache_mb,
            prefix_chunk_tokens=prefix_chunk_tokens,
            host_mem_bytes=host_mem_bytes,
        )

        try:
            weights = mx.load(str(sf_p))
            self.q_w = np.asarray(weights["pointer_head.q_weight"].astype(mx.float32))
            self.q_b = np.asarray(weights["pointer_head.q_bias"].astype(mx.float32))
            self.k_w = np.asarray(weights["pointer_head.k_weight"].astype(mx.float32))
            self.k_b = np.asarray(weights["pointer_head.k_bias"].astype(mx.float32))
            self.ptr_scale = 1.0 / math.sqrt(float(self.q_w.shape[0]))

            self.backbone = _build_qwen35_backbone(meta["text_config"], weights, quant_cfg=self.quantization)
            self._full_layer_indices = [i for i, blk in enumerate(self.backbone.layers) if not blk.is_linear]
            self._linear_layer_indices = [i for i, blk in enumerate(self.backbone.layers) if blk.is_linear]
            del weights
            mx.clear_cache()
        except Exception:
            self._restore_mlx_limits()
            raise

    def _restore_mlx_limits(self) -> None:
        if getattr(self, "_limits_restored", True):
            return
        self._limits_restored = True
        mx = self._mx
        if self._prev_wired_limit is not None:
            set_wired = getattr(mx, "set_wired_limit", None)
            if callable(set_wired):
                try:
                    set_wired(self._prev_wired_limit)
                except Exception:
                    pass
            self._prev_wired_limit = None
        if self._prev_cache_limit is not None:
            set_cache = getattr(mx, "set_cache_limit", None)
            if callable(set_cache):
                try:
                    set_cache(self._prev_cache_limit)
                except Exception:
                    pass
            self._prev_cache_limit = None

    def __enter__(self) -> "Kev08bMlxBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _user_tokens(self, text: str) -> list[int]:
        return _encode_kev_user_tokens(self.tok, text)

    def encode_questions(self, state: Any, questions: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
        """Encode one shared state prefix plus all typed branch questions (`noul`, `choice`, `score`)."""
        return encode_kev_questions(
            self.tok,
            state,
            questions,
            special_ids=self.special_ids,
            max_state_tokens=self.max_state,
            max_branch_tokens=self.max_branch,
        )

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

    def _maybe_clear_wave_cache(self) -> None:
        mx = self._mx
        get_cache_mem = getattr(mx, "get_cache_memory", None)
        if callable(get_cache_mem):
            try:
                if int(get_cache_mem()) > _KEV_CLEAR_CACHE_THRESHOLD_BYTES:
                    mx.clear_cache()
            except Exception:
                pass

    def shed_prefix_cache(self) -> int:
        """Drop all cached prefix chunks/snapshots from `prefix_tree` and clear MLX allocator cache."""
        if self.prefix_tree is None:
            return 0
        freed = self.prefix_tree.clear()
        self._mx.clear_cache()
        return freed

    def _detach_kv_chunk(self, arr: Any, start: int, end: int) -> Any:
        mx = self._mx
        if start == 0 and int(arr.shape[2]) == end:
            return mx.contiguous(arr)
        sl = arr[:, :, start:end, :]
        if int(arr.shape[1]) > 1:
            return mx.contiguous(sl)
        return mx.concatenate([sl], axis=2)

    def _prefill_state_with_prefix_tree(
        self,
        state_ids: Sequence[int],
        *,
        eval_restored_concat: bool = False,
    ) -> list[Any]:
        """Unpadded `256`-token chunked prefill restoring/publishing KV + hybrid GDN states in `PrefixTree`."""
        mx = self._mx
        tree = self.prefix_tree
        assert tree is not None
        s_len = len(state_ids)
        chunk_sz = tree.chunk_tokens
        cache = self.backbone.make_cache()

        if tree.byte_budget <= 0 or s_len < chunk_sz:
            h_st = self.backbone(mx.array([state_ids], dtype=mx.int32), cache, write_state=True)
            mx.eval(h_st)
            return cache

        match = tree.match(state_ids, limit=s_len, needs_snapshot=True)
        pos = match.reused_tokens
        parent_key = match.keys[-1] if match.keys else None
        depth = len(match.keys)

        if pos > 0:
            n_chunks = len(match.chunks)
            for fa_i, l_idx in enumerate(self._full_layer_indices):
                if n_chunks == 1:
                    cache[l_idx].keys = match.chunks[0][0][fa_i]
                    cache[l_idx].values = match.chunks[0][1][fa_i]
                else:
                    cache[l_idx].keys = mx.concatenate([c[0][fa_i] for c in match.chunks], axis=2)
                    cache[l_idx].values = mx.concatenate([c[1][fa_i] for c in match.chunks], axis=2)
                cache[l_idx].offset = pos
            conv_snaps, rec_snaps = match.snapshot
            for gdn_i, l_idx in enumerate(self._linear_layer_indices):
                cache[l_idx].conv_state = conv_snaps[gdn_i]
                cache[l_idx].rec_state = rec_snaps[gdn_i]
            tree.record(match, restored=True)
        else:
            tree.record(match, restored=False)

        full_end = (s_len // chunk_sz) * chunk_sz
        if pos < full_end:
            snap_positions = set(
                snapshot_boundaries(
                    pos,
                    s_len,
                    chunk_tokens=chunk_sz,
                    snapshot_stride=self.prefix_snapshot_stride,
                    include_deepest=True,
                )
            )
            for c_start in range(pos, full_end, chunk_sz):
                c_end = c_start + chunk_sz
                chunk_toks = state_ids[c_start:c_end]
                h_c = self.backbone(mx.array([chunk_toks], dtype=mx.int32), cache, write_state=True)
                k_chunk = tuple(
                    self._detach_kv_chunk(cache[l_idx].keys, c_start, c_end) for l_idx in self._full_layer_indices
                )
                v_chunk = tuple(
                    self._detach_kv_chunk(cache[l_idx].values, c_start, c_end) for l_idx in self._full_layer_indices
                )
                want_snap = c_end in snap_positions
                if want_snap:
                    conv_snap = tuple(mx.contiguous(cache[l_idx].conv_state) for l_idx in self._linear_layer_indices)
                    rec_snap = tuple(mx.contiguous(cache[l_idx].rec_state) for l_idx in self._linear_layer_indices)
                    mx.eval(h_c, *k_chunk, *v_chunk, *conv_snap, *rec_snap)
                else:
                    conv_snap = ()
                    rec_snap = ()
                    mx.eval(h_c, *k_chunk, *v_chunk)

                new_key = tree.insert(
                    parent=parent_key,
                    tokens=chunk_toks,
                    depth=depth + 1,
                    chunk=(k_chunk, v_chunk),
                )
                pos = c_end
                if new_key is None:
                    break
                parent_key = new_key
                depth += 1
                if want_snap and not tree.has_snapshot(new_key):
                    tree.attach_snapshot(new_key, (conv_snap, rec_snap), keeping_chain_to=new_key)

        if pos < s_len:
            h_tail = self.backbone(mx.array([state_ids[pos:]], dtype=mx.int32), cache, write_state=True)
            mx.eval(h_tail)
        elif eval_restored_concat and pos > 0 and len(match.chunks) > 1:
            mx.eval(
                *[cache[i].keys for i in self._full_layer_indices],
                *[cache[i].values for i in self._full_layer_indices],
            )
        return cache

    def _score_single_cache_branches(
        self,
        cache: list[Any],
        rows: Sequence[Mapping[str, Any]],
        br_chunk_lim: int,
        *,
        temperature: float | None = None,
    ) -> list[tuple[list[float], list[float]]]:
        mx = self._mx
        np = self._np
        doc_outs: list[tuple[list[float], list[float]]] = []
        for c0 in range(0, len(rows), br_chunk_lim):
            sub_rows = rows[c0 : c0 + br_chunk_lim]
            n_sub = len(sub_rows)
            max_l = max(len(r["ids"]) for r in sub_rows)
            ids_mx = mx.array(
                [r["ids"] + [self.pad_id] * (max_l - len(r["ids"])) for r in sub_rows],
                dtype=mx.int32,
            )
            owner_mx = mx.zeros((n_sub,), dtype=mx.int32)
            h_mx = self.backbone(
                ids_mx,
                cache,
                owner_idx=owner_mx,
                owner_is_identity=(n_sub == 1),
                write_state=False,
            )
            mx.eval(h_mx)
            h_np = np.asarray(h_mx.astype(mx.float32))
            for ri, r in enumerate(sub_rows):
                doc_outs.append(
                    self._pointer_logits_and_probs(h_np[ri], r["decide"], r["opts"], temperature=temperature)
                )
        return doc_outs

    def _run_batched_pointer(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
        temperature: float | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        """Batch `B` document states via left-padded hybrid prefill + zero-GDN-clone branch pass."""
        if not items:
            return []
        mx = self._mx
        np = self._np
        eff_b = max(1, int(batch_docs or self.batch_docs))
        br_chunk_lim = max(eff_b, self.max_branch_batch)

        encoded_all = [self.encode_questions(st, qs) for st, qs in items]
        order = sorted(range(len(items)), key=lambda idx: len(encoded_all[idx][0]))
        out_by_orig: list[list[tuple[list[float], list[float]]]] = [[] for _ in items]

        if self.prefix_tree is not None and self.prefix_tree.byte_budget > 0:
            chunk_sz = self.prefix_tree.chunk_tokens
            n_nonempty_items = sum(1 for i in order if encoded_all[i][1])
            prefix_counts = Counter(
                tuple(encoded_all[i][0][:chunk_sz])
                for i in order
                if encoded_all[i][1] and len(encoded_all[i][0]) >= chunk_sz
            )
            leftover_order: list[int] = []
            for orig_i in order:
                state_ids, rows = encoded_all[orig_i]
                if not rows:
                    out_by_orig[orig_i] = []
                    continue
                if len(state_ids) < chunk_sz:
                    leftover_order.append(orig_i)
                    continue
                first_chunk = tuple(state_ids[:chunk_sz])
                use_prefix_tree = (
                    n_nonempty_items == 1
                    or prefix_counts[first_chunk] >= 2
                    or self.prefix_tree.match(state_ids, needs_snapshot=True).reused_tokens >= chunk_sz
                )
                if not use_prefix_tree:
                    leftover_order.append(orig_i)
                    continue
                cache = self._prefill_state_with_prefix_tree(
                    state_ids,
                    eval_restored_concat=(len(rows) > br_chunk_lim),
                )
                out_by_orig[orig_i] = self._score_single_cache_branches(
                    cache, rows, br_chunk_lim, temperature=temperature
                )
                del cache
                self._maybe_clear_wave_cache()
            order = leftover_order

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
                h_st = self.backbone(mx.array([state_ids], dtype=mx.int32), cache, write_state=True)
                mx.eval(h_st)
                out_by_orig[orig_0] = self._score_single_cache_branches(
                    cache, rows, br_chunk_lim, temperature=temperature
                )
                del cache, h_st
                self._maybe_clear_wave_cache()
                continue

            state_lists = [encoded_all[idx][0] for idx in wave_orig]
            s_lens = [len(s) for s in state_lists]
            s_max = max(s_lens)
            pad_lens = [s_max - ln for ln in s_lens]
            all_unpadded = all(pl == 0 for pl in pad_lens)

            st_np = np.full((B, s_max), self.pad_id, dtype=np.int32)
            for b_i in range(B):
                pl = pad_lens[b_i]
                st_np[b_i, pl:] = state_lists[b_i]

            if all_unpadded:
                valid_scale_mx = None
                pos_st_mx = None
                mask_st_mx: Any = "causal"
            else:
                pl_mx = mx.array(pad_lens, dtype=mx.int32)[:, None]
                seq_pos = mx.arange(s_max, dtype=mx.int32)[None, :]
                valid_scale_mx = mx.where(
                    seq_pos >= pl_mx,
                    mx.array(1.0, dtype=mx.bfloat16),
                    mx.array(0.0, dtype=mx.bfloat16),
                )[:, :, None]
                pos_st_mx = mx.maximum(seq_pos - pl_mx, 0)
                mask_st_mx = _build_state_attn_mask(mx, pad_lens, s_max, mx.bfloat16)

            cache = self.backbone.make_cache()
            h_st = self.backbone(
                mx.array(st_np, dtype=mx.int32),
                cache,
                valid_token_scale=valid_scale_mx,
                attn_mask=mask_st_mx,
                pos_2d=pos_st_mx,
                write_state=True,
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
                del cache, h_st
                self._maybe_clear_wave_cache()
                continue

            N_br = len(flat_rows)
            flat_outs: list[tuple[list[float], list[float]]] = []
            for c0 in range(0, N_br, br_chunk_lim):
                c1 = min(N_br, c0 + br_chunk_lim)
                sub_rows = flat_rows[c0:c1]
                sub_owner = owner_b[c0:c1]
                n_sub = len(sub_rows)
                max_br = max(len(r["ids"]) for r in sub_rows)
                br_np = np.full((n_sub, max_br), self.pad_id, dtype=np.int32)
                for r_i, r in enumerate(sub_rows):
                    bl = len(r["ids"])
                    br_np[r_i, :bl] = r["ids"]

                sub_pads = [pad_lens[b_i] for b_i in sub_owner]
                sub_slens = [s_lens[b_i] for b_i in sub_owner]
                sub_unpadded = all(pl == 0 for pl in sub_pads)
                if sub_unpadded:
                    pos_br_mx = None
                    mask_br_mx: Any = "causal"
                else:
                    pos_br_mx = (
                        mx.array(sub_slens, dtype=mx.int32)[:, None]
                        + mx.arange(max_br, dtype=mx.int32)[None, :]
                    )
                    mask_br_mx = _build_branch_attn_mask(mx, sub_pads, s_max, max_br, mx.bfloat16)

                owner_is_identity = (n_sub == B) and (sub_owner == list(range(B)))
                owner_mx = mx.array(sub_owner, dtype=mx.int32)
                h_br = self.backbone(
                    mx.array(br_np, dtype=mx.int32),
                    cache,
                    valid_token_scale=None,
                    attn_mask=mask_br_mx,
                    pos_2d=pos_br_mx,
                    owner_idx=owner_mx,
                    owner_is_identity=owner_is_identity,
                    write_state=False,
                )
                mx.eval(h_br)
                h_br_np = np.asarray(h_br.astype(mx.float32))
                for ri, r in enumerate(sub_rows):
                    flat_outs.append(
                        self._pointer_logits_and_probs(h_br_np[ri], r["decide"], r["opts"], temperature=temperature)
                    )

            for b_i, orig_i in enumerate(wave_orig):
                lo, hi = doc_slices[b_i]
                out_by_orig[orig_i] = flat_outs[lo:hi]

            del cache, h_st
            self._maybe_clear_wave_cache()

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
        """Release model weights, owned prefix cache, restore MLX memory/cache limits, and clear MLX cache."""
        if getattr(self, "_owns_prefix_tree", False) and getattr(self, "prefix_tree", None) is not None:
            self.prefix_tree.clear()
        self.backbone = None
        self._restore_mlx_limits()
        self._mx.clear_cache()


__all__ = [
    "KEV_SPECIAL_TOKENS",
    "Kev08bMlxBackend",
    "_build_branch_attn_mask",
    "_build_state_attn_mask",
    "_detect_apple10_d256_force_fused",
    "encode_kev_questions",
    "resolve_kev_special_ids",
]
