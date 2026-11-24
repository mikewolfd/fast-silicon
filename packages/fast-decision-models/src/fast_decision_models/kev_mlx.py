"""Metal MLX typed-question scorer for `Kev-0.8B` & `Kev-4B` (`Qwen3.5` + `PointerHead`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import partial
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Literal, Mapping, Sequence

from ._common import softmax_vec
from .paths import (
    DEFAULT_KEV_08B_MLX_REPO,
    DEFAULT_KEV_4B_ADAPTER_REPO,
    DEFAULT_KEV_4B_MLX_REPO,
    DEFAULT_KEV_ADAPTER_REPO,
    resolve_kev_4b_dir,
    resolve_kev_dir,
)
from .prefix_tree import (
    DEFAULT_CHUNK_TOKENS,
    PrefixTree,
    recommended_prefix_budget_bytes,
    snapshot_boundaries,
)

_KEV_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
_KEV_CACHE_LIMIT_BYTES = 512 << 20
_KEV_CLEAR_CACHE_THRESHOLD_BYTES = 256 << 20
_MIN_LCP_GROUP_TOKENS = 128
_TIGHT_WAVE_PAD_BUDGET = 16

KEV_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)

# Metal kernel source for Qwen3.5 `GatedDeltaNet` (`Dk=128`, `Dv % 8 == 0` packed butterfly).
# Retained for short multi-branch fan-out (`T <= 64`, `owner_idx` indirect prefix read, `WriteState=False`
# with zero FP32 GDN state cloning), while 2D prefill/unshared/long passes dispatch MLX 0.32.4's native
# chunked `mx.fast.gated_delta_update` (`Apple10 NAXTile C=16` on M5, `C=8` on M4).
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
    """Pure-Python Kev question encoder (zero MLX weight load; single source of truth)."""
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
    allow = ((q_pos >= pl) & (k_pos >= pl) & (k_pos <= q_pos)) | ((q_pos < pl) & (k_pos == q_pos))
    neg_fill = -65504.0 if act_dtype == getattr(mx, "float16", None) else -1e9
    return mx.where(allow, mx.array(0.0, dtype=act_dtype), mx.array(neg_fill, dtype=act_dtype))


def _build_branch_attn_mask(mx: Any, owner_pads: Sequence[int], s_max: int, max_br: int, act_dtype: Any) -> Any:
    """Vectorized additive `(N_br, 1, max_br, s_max + max_br)` branch mask directly in `act_dtype`."""
    if all(pl == 0 for pl in owner_pads):
        return "causal"
    pl = mx.array(list(owner_pads), dtype=mx.int32)[:, None, None, None]
    q_pos = mx.arange(max_br, dtype=mx.int32)[None, None, :, None] + s_max
    k_pos = mx.arange(s_max + max_br, dtype=mx.int32)[None, None, None, :]
    allow = (k_pos >= pl) & (k_pos <= q_pos)
    neg_fill = -65504.0 if act_dtype == getattr(mx, "float16", None) else -1e9
    return mx.where(allow, mx.array(0.0, dtype=act_dtype), mx.array(neg_fill, dtype=act_dtype))


def _normalize_kev_weight_key(k: str) -> str | None:
    if k.startswith(("vision_tower", "model.visual")) or "mtp." in k or k == "lm_head.weight":
        return None
    if k.startswith("language_model.model."):
        return "model." + k[len("language_model.model.") :]
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model.") :]
    if k.startswith("model.") or k.startswith("pointer_head."):
        return k
    return None


def _lcp_len(a: Sequence[int], b: Sequence[int], cap: int | None = None) -> int:
    lim = min(len(a), len(b))
    if cap is not None:
        lim = min(lim, cap)
    idx = 0
    while idx < lim and a[idx] == b[idx]:
        idx += 1
    return idx


def _partition_exact_lcp_groups(
    indices: Sequence[int],
    encoded_all: Sequence[tuple[list[int], list[dict[str, Any]]]],
    *,
    min_lcp: int = _MIN_LCP_GROUP_TOKENS,
) -> tuple[list[tuple[list[int], int]], list[int]]:
    """Group incoming item indices by `(prefix_min_lcp, exact_lcp >= min_lcp)` before length sorting.

    Returns `(groups, singletons)` where each entry in `groups` is `(member_indices, lcp_tokens)`.
    """
    by_p32: dict[tuple[int, ...], list[int]] = defaultdict(list)
    singletons: list[int] = []
    for idx in indices:
        s_ids, rows = encoded_all[idx]
        if not rows:
            continue
        if len(s_ids) < min_lcp:
            singletons.append(idx)
            continue
        by_p32[tuple(s_ids[:min_lcp])].append(idx)

    groups: list[tuple[list[int], int]] = []
    for bucket in by_p32.values():
        if len(bucket) < 2:
            singletons.extend(bucket)
            continue
        # Lexicographical sort places items with the longest shared prefixes adjacent while keeping
        # identical-prefix siblings ordered by total state length.
        lex = sorted(bucket, key=lambda i: encoded_all[i][0])
        cur_grp = [lex[0]]
        cur_lcp = len(encoded_all[lex[0]][0])
        for nxt in lex[1:]:
            pair_lcp = _lcp_len(encoded_all[cur_grp[0]][0], encoded_all[nxt][0])
            if pair_lcp >= min_lcp:
                cur_grp.append(nxt)
                cur_lcp = min(cur_lcp, pair_lcp)
            else:
                if len(cur_grp) >= 2:
                    groups.append((cur_grp, cur_lcp))
                else:
                    singletons.extend(cur_grp)
                cur_grp = [nxt]
                cur_lcp = len(encoded_all[nxt][0])
        if len(cur_grp) >= 2:
            groups.append((cur_grp, cur_lcp))
        else:
            singletons.extend(cur_grp)
    return groups, singletons


def _pack_tight_length_waves(
    item_entries: list[tuple[int, list[int], int, list[int]]],
    eff_b: int,
    *,
    tight_pad_budget: int = _TIGHT_WAVE_PAD_BUDGET,
) -> list[list[tuple[int, list[int], int, list[int]]]]:
    """Form length-sorted waves favoring exact-length or `<= 16`-pad batches without creating `B=1` stragglers."""
    if not item_entries:
        return []
    sorted_entries = sorted(item_entries, key=lambda x: len(x[1]))
    if eff_b <= 1:
        return [[e] for e in sorted_entries]

    waves: list[list[tuple[int, list[int], int, list[int]]]] = []
    n_total = len(sorted_entries)
    idx = 0
    min_flush_b = max(2, eff_b // 2)
    while idx < n_total:
        cur: list[tuple[int, list[int], int, list[int]]] = [sorted_entries[idx]]
        min_l = len(sorted_entries[idx][1])
        idx += 1
        while idx < n_total and len(cur) < eff_b:
            nxt_l = len(sorted_entries[idx][1])
            rem_after = n_total - idx
            if len(cur) >= min_flush_b and (nxt_l - min_l) > tight_pad_budget and rem_after >= 2:
                break
            cur.append(sorted_entries[idx])
            idx += 1
        waves.append(cur)
    return waves


def _build_qwen35_backbone(
    text_cfg: Mapping[str, Any],
    weights: dict[str, Any],
    *,
    quant_cfg: Mapping[str, Any] | None = None,
    use_d256_force_fused: bool | None = None,
) -> Any:
    """Build self-contained MLX `Qwen3_5TextModel` with horizontal linear fusion (`9 -> 3`), mixed `4b/8b` support, and `mx.fast.gated_delta_update`."""
    import mlx.core as mx
    import mlx.nn as nn

    has_fast_gdn = hasattr(mx.fast, "gated_delta_update")
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
    def _fused_gdn_gates(A_log: Any, a: Any, dt_bias: Any, b: Any) -> tuple[Any, Any]:
        g = mx.exp(
            -mx.exp(A_log.astype(mx.float32))
            * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
        )
        beta = mx.sigmoid(b)
        return g, beta

    norm_weights: dict[str, Any] = {}
    for raw_k, val in weights.items():
        nk = _normalize_kev_weight_key(raw_k)
        if nk is not None and nk.startswith("model."):
            norm_weights[nk[len("model.") :]] = val

    def _infer_quant_spec(stem: str, in_dims: int) -> tuple[int, int, bool] | None:
        sc = norm_weights.get(stem + ".scales")
        if sc is None:
            return None
        wq = norm_weights[stem + ".weight"]
        group_size = in_dims // int(sc.shape[-1])
        bits = (int(wq.shape[-1]) * 32) // in_dims
        has_biases = (stem + ".biases") in norm_weights
        return bits, group_size, has_biases

    def _make_single_linear(stem: str, in_dims: int, out_dims: int) -> Any:
        qspec = _infer_quant_spec(stem, in_dims)
        if qspec is not None:
            bits, group_size, has_biases = qspec
            lin = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=group_size, bits=bits)
            lin.weight = norm_weights[stem + ".weight"]
            lin.scales = norm_weights[stem + ".scales"].astype(mx.bfloat16)
            if has_biases:
                lin.biases = norm_weights[stem + ".biases"].astype(mx.bfloat16)
            return lin
        w_raw = norm_weights[stem + ".weight"].astype(mx.bfloat16)
        if quant_cfg is not None and int(w_raw.shape[-1]) % int(quant_cfg.get("group_size", 64)) == 0:
            gs = int(quant_cfg.get("group_size", 64))
            bt = int(quant_cfg.get("bits", 8))
            lin = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=gs, bits=bt)
            wq, sc, bi = mx.quantize(w_raw, group_size=gs, bits=bt)
            lin.weight = wq
            lin.scales = sc
            lin.biases = bi
            return lin
        lin_bf16 = nn.Linear(in_dims, out_dims, bias=False)
        lin_bf16.weight = w_raw
        return lin_bf16

    def _make_fused_linear(stems: Sequence[str], in_dims: int, out_dims_list: Sequence[int]) -> Any | None:
        """Horizontally concatenate quantized or BF16 linear projections sharing input `x` (`0` rounding drift)."""
        specs = [_infer_quant_spec(s, in_dims) for s in stems]
        total_out = sum(int(d) for d in out_dims_list)
        if all(sp is not None for sp in specs):
            first = specs[0]
            assert first is not None
            if any(sp != first for sp in specs[1:]):
                return None
            bits, group_size, has_biases = first
            lin = nn.QuantizedLinear(in_dims, total_out, bias=False, group_size=group_size, bits=bits)
            lin.weight = mx.concatenate([norm_weights[s + ".weight"] for s in stems], axis=0)
            lin.scales = mx.concatenate([norm_weights[s + ".scales"].astype(mx.bfloat16) for s in stems], axis=0)
            if has_biases:
                lin.biases = mx.concatenate([norm_weights[s + ".biases"].astype(mx.bfloat16) for s in stems], axis=0)
            return lin
        if all(sp is None for sp in specs):
            w_cat = mx.concatenate([norm_weights[s + ".weight"].astype(mx.bfloat16) for s in stems], axis=0)
            if quant_cfg is not None and in_dims % int(quant_cfg.get("group_size", 64)) == 0:
                gs = int(quant_cfg.get("group_size", 64))
                bt = int(quant_cfg.get("bits", 8))
                lin = nn.QuantizedLinear(in_dims, total_out, bias=False, group_size=gs, bits=bt)
                wq, sc, bi = mx.quantize(w_cat, group_size=gs, bits=bt)
                lin.weight = wq
                lin.scales = sc
                lin.biases = bi
                return lin
            lin_bf16 = nn.Linear(in_dims, total_out, bias=False)
            lin_bf16.weight = w_cat
            return lin_bf16
        return None

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
        def __init__(self, d: int, eps: float, w_arr: Any) -> None:
            super().__init__()
            self.eps = eps
            self.weight = w_arr.astype(mx.bfloat16)

        def __call__(self, x: Any, gate: Any) -> Any:
            xn = mx.fast.rms_norm(x, self.weight, self.eps)
            return _fused_precise_gate(xn, gate)

    class _GatedDeltaLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            px = f"layers.{layer_idx}.linear_attn"
            self.key_dim = head_k_dim * num_k_heads
            self.value_dim = head_v_dim * num_v_heads
            self.conv_dim = self.key_dim * 2 + self.value_dim
            self._split_stops = [
                self.conv_dim,
                self.conv_dim + self.value_dim,
                self.conv_dim + self.value_dim + num_v_heads,
            ]
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=False,
                kernel_size=conv_k,
                groups=self.conv_dim,
                padding=0,
            )
            cw = norm_weights[px + ".conv1d.weight"]
            if cw.shape[-1] != 1:
                cw = cw.moveaxis(2, 1)
            self.conv1d.weight = cw.astype(mx.bfloat16)

            fused_in = _make_fused_linear(
                [px + ".in_proj_qkv", px + ".in_proj_z", px + ".in_proj_b", px + ".in_proj_a"],
                hidden_size,
                [self.conv_dim, self.value_dim, num_v_heads, num_v_heads],
            )
            if fused_in is not None:
                self.in_proj_all = fused_in
            else:
                from fast_shared.kv_quant import SpecBucketedLinearGroup

                self.in_proj_group = SpecBucketedLinearGroup(
                    [
                        ("qkv", _make_single_linear(px + ".in_proj_qkv", hidden_size, self.conv_dim)),
                        ("z", _make_single_linear(px + ".in_proj_z", hidden_size, self.value_dim)),
                        ("b", _make_single_linear(px + ".in_proj_b", hidden_size, num_v_heads)),
                        ("a", _make_single_linear(px + ".in_proj_a", hidden_size, num_v_heads)),
                    ]
                )

            self.dt_bias = norm_weights[px + ".dt_bias"].astype(mx.bfloat16)
            self.A_log = norm_weights[px + ".A_log"].astype(mx.bfloat16)
            self.norm = _RMSNormGated(head_v_dim, rms_eps, norm_weights[px + ".norm.weight"])
            self.out_proj = _make_single_linear(px + ".out_proj", self.value_dim, hidden_size)
            self._inv_scale = head_k_dim**-0.5
            self._rms_eps = 1e-6 * (self._inv_scale**2)

        def __call__(
            self,
            x: Any,
            cache: _DeltaCache | None,
            valid_token_scale: Any | None = None,
            *,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            B, S, _ = x.shape
            if hasattr(self, "in_proj_all"):
                proj = self.in_proj_all(x)
                qkv, z_flat, b, a = mx.split(proj, self._split_stops, axis=-1)
                z = z_flat.reshape(B, S, num_v_heads, head_v_dim)
            else:
                parts = self.in_proj_group(x)
                qkv = parts["qkv"]
                z = parts["z"].reshape(B, S, num_v_heads, head_v_dim)
                b = parts["b"]
                a = parts["a"]

            if valid_token_scale is not None:
                qkv = qkv * valid_token_scale

            if cache is None or cache.conv_state is None:
                conv_in = mx.pad(qkv, [(0, 0), (conv_k - 1, 0), (0, 0)])
            else:
                if owner_idx is None or owner_is_identity:
                    conv_state = cache.conv_state
                elif int(cache.conv_state.shape[0]) == 1:
                    conv_state = mx.broadcast_to(cache.conv_state, (B, *cache.conv_state.shape[1:]))
                else:
                    conv_state = mx.take(cache.conv_state, owner_idx, axis=0)
                conv_in = mx.concatenate([conv_state, qkv], axis=1)

            if write_state and cache is not None:
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

            g, beta = _fused_gdn_gates(self.A_log, a, self.dt_bias, b)
            if valid_token_scale is not None:
                beta = beta * valid_token_scale
                vm_f32 = valid_token_scale.astype(mx.float32)
                g = g * vm_f32 + (mx.array(1.0, dtype=mx.float32) - vm_f32)

            T = int(k.shape[1])
            has_prior_state = cache is not None and cache.rec_state is not None
            non_identity_fanout = has_prior_state and (owner_idx is not None and not owner_is_identity)

            if has_fast_gdn and (not non_identity_fanout or T > 64):
                if not has_prior_state:
                    init_st = None
                elif not non_identity_fanout:
                    init_st = cache.rec_state
                elif int(cache.rec_state.shape[0]) == 1:
                    init_st = mx.broadcast_to(cache.rec_state, (B, *cache.rec_state.shape[1:]))
                else:
                    init_st = mx.take(cache.rec_state, owner_idx, axis=0)
                y, new_state = mx.fast.gated_delta_update(q, k, v, g, beta, init_st)
                if write_state and cache is not None:
                    cache.rec_state = new_state
            else:
                if has_prior_state:
                    assert cache is not None
                    rec_state = cache.rec_state
                    eff_owner = owner_idx if owner_idx is not None else mx.arange(B, dtype=mx.int32)
                else:
                    rec_state = mx.zeros((B, num_v_heads, head_v_dim, head_k_dim), dtype=mx.float32)
                    eff_owner = mx.arange(B, dtype=mx.int32)
                do_write = bool(write_state and cache is not None)
                state_out_shape = (B, num_v_heads, head_v_dim, head_k_dim) if do_write else (1,)
                y, new_state = gdn_packed_kernel(
                    inputs=[q, k, v, g, beta, rec_state, eff_owner, T],
                    template=[
                        ("InT", q.dtype),
                        ("StT", mx.float32),
                        ("Dk", head_k_dim),
                        ("Dv", head_v_dim),
                        ("Hk", num_k_heads),
                        ("Hv", num_v_heads),
                        ("WriteState", do_write),
                    ],
                    grid=(32, head_v_dim // 8, B * num_v_heads),
                    threadgroup=(32, 2, 1),
                    output_shapes=[(B, T, num_v_heads, head_v_dim), state_out_shape],
                    output_dtypes=[q.dtype, mx.float32],
                )
                if do_write and cache is not None:
                    cache.rec_state = new_state

            out = self.norm(y, z)
            return self.out_proj(out.reshape(B, S, -1))

    class _GatedFullAttention(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            px = f"layers.{layer_idx}.self_attn"
            self.scale = head_dim**-0.5
            q_out = num_q_heads * head_dim * 2
            kv_out = num_kv_heads * head_dim
            self._qkv_stops = [q_out, q_out + kv_out]

            fused_qkv = _make_fused_linear(
                [px + ".q_proj", px + ".k_proj", px + ".v_proj"],
                hidden_size,
                [q_out, kv_out, kv_out],
            )
            if fused_qkv is not None:
                self.qkv_proj = fused_qkv
            else:
                from fast_shared.kv_quant import SpecBucketedLinearGroup

                self.qkv_group = SpecBucketedLinearGroup(
                    [
                        ("q", _make_single_linear(px + ".q_proj", hidden_size, q_out)),
                        ("k", _make_single_linear(px + ".k_proj", hidden_size, kv_out)),
                        ("v", _make_single_linear(px + ".v_proj", hidden_size, kv_out)),
                    ]
                )

            self.o_proj = _make_single_linear(px + ".o_proj", num_q_heads * head_dim, hidden_size)
            self.q_norm = nn.RMSNorm(head_dim, eps=rms_eps)
            self.q_norm.weight = norm_weights[px + ".q_norm.weight"].astype(mx.bfloat16)
            self.k_norm = nn.RMSNorm(head_dim, eps=rms_eps)
            self.k_norm.weight = norm_weights[px + ".k_norm.weight"].astype(mx.bfloat16)
            self.rope = nn.RoPE(rot_dims, traditional=False, base=rope_theta)

        def __call__(
            self,
            x: Any,
            cache: _FullAttnCache | None,
            attn_mask: Any = "causal",
            rope_trig: tuple[Any, Any] | None = None,
            *,
            owner_idx: Any | None = None,
            owner_is_identity: bool = False,
            write_state: bool = True,
        ) -> Any:
            B, L, _ = x.shape
            if hasattr(self, "qkv_proj"):
                qkv_out = self.qkv_proj(x)
                qp, kp, vp = mx.split(qkv_out, self._qkv_stops, axis=-1)
            else:
                parts_qkv = self.qkv_group(x)
                qp = parts_qkv["q"]
                kp = parts_qkv["k"]
                vp = parts_qkv["v"]

            queries, gate = mx.split(qp.reshape(B, L, num_q_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, L, -1)
            keys = kp.reshape(B, L, num_kv_heads, head_dim)
            values = vp.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)

            # Concatenate Q and K along head axis so partial RoPE dispatches once per FullAttn layer.
            qk_cat = mx.concatenate([queries, keys], axis=1)
            if rope_trig is not None:
                cos_a, sin_a = rope_trig
                qk_cat = _apply_partial_rope_trig(qk_cat, cos_a, sin_a)
            else:
                off = 0 if cache is None else cache.offset
                qk_cat = self.rope(qk_cat, offset=off)
            queries, keys = mx.split(qk_cat, [num_q_heads], axis=1)

            if cache is not None:
                keys, values = cache.update_and_fetch(
                    keys,
                    values,
                    owner_idx=owner_idx,
                    owner_is_identity=owner_is_identity,
                    write_state=write_state,
                )

            eff_mask = (None if L == 1 else "causal") if attn_mask == "causal" else attn_mask
            if allow_d256_force_fused and queries.dtype in (mx.float16, mx.bfloat16) and head_dim == 256 and 256 <= L < 1024:
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
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            px = f"layers.{layer_idx}.mlp"
            fused_gu = _make_fused_linear(
                [px + ".gate_proj", px + ".up_proj"],
                hidden_size,
                [intermediate_size, intermediate_size],
            )
            if fused_gu is not None:
                self.gate_up_proj = fused_gu
            else:
                self.gate_proj = _make_single_linear(px + ".gate_proj", hidden_size, intermediate_size)
                self.up_proj = _make_single_linear(px + ".up_proj", hidden_size, intermediate_size)
            self.down_proj = _make_single_linear(px + ".down_proj", intermediate_size, hidden_size)

        def __call__(self, x: Any) -> Any:
            if hasattr(self, "gate_up_proj"):
                gate, up = mx.split(self.gate_up_proj(x), [intermediate_size], axis=-1)
                return self.down_proj(_fused_swiglu(gate, up))
            return self.down_proj(_fused_swiglu(self.gate_proj(x), self.up_proj(x)))

    class _DecoderBlock(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            px = f"layers.{idx}"
            self.is_linear = (idx + 1) % fa_interval != 0
            if self.is_linear:
                self.linear_attn = _GatedDeltaLayer(idx)
            else:
                self.self_attn = _GatedFullAttention(idx)
            self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.input_layernorm.weight = norm_weights[px + ".input_layernorm.weight"].astype(mx.bfloat16)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.post_attention_layernorm.weight = norm_weights[px + ".post_attention_layernorm.weight"].astype(mx.bfloat16)
            self.mlp = _SwiGLU(idx)

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
        _supports_stateless_fused_pass = True

        def __init__(self) -> None:
            super().__init__()
            emb_spec = _infer_quant_spec("embed_tokens", hidden_size)
            if emb_spec is not None:
                eb_bits, eb_gs, eb_has_biases = emb_spec
                self.embed_tokens: Any = nn.QuantizedEmbedding(
                    vocab_size, hidden_size, group_size=eb_gs, bits=eb_bits
                )
                self.embed_tokens.weight = norm_weights["embed_tokens.weight"]
                self.embed_tokens.scales = norm_weights["embed_tokens.scales"].astype(mx.bfloat16)
                if eb_has_biases:
                    self.embed_tokens.biases = norm_weights["embed_tokens.biases"].astype(mx.bfloat16)
            elif quant_cfg is not None and hidden_size % int(quant_cfg.get("group_size", 64)) == 0:
                eb_gs = int(quant_cfg.get("group_size", 64))
                eb_bits = int(quant_cfg.get("bits", 8))
                self.embed_tokens = nn.QuantizedEmbedding(vocab_size, hidden_size, group_size=eb_gs, bits=eb_bits)
                wq, sc, bi = mx.quantize(
                    norm_weights["embed_tokens.weight"].astype(mx.bfloat16), group_size=eb_gs, bits=eb_bits
                )
                self.embed_tokens.weight = wq
                self.embed_tokens.scales = sc
                self.embed_tokens.biases = bi
            else:
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.embed_tokens.weight = norm_weights["embed_tokens.weight"].astype(mx.bfloat16)

            self.layers = [_DecoderBlock(i) for i in range(num_layers)]
            self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.norm.weight = norm_weights["norm.weight"].astype(mx.bfloat16)
            self.use_d256_force_fused = allow_d256_force_fused

        def make_cache(self) -> list[Any]:
            return [_DeltaCache() if blk.is_linear else _FullAttnCache() for blk in self.layers]

        @staticmethod
        def select_cache_rows(cache: list[Any], row_indices: Any) -> list[Any]:
            return [c.select_rows(row_indices) for c in cache]

        def __call__(
            self,
            input_ids: Any,
            cache: list[Any] | None = None,
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
                layer_cache = None if cache is None else cache[idx]
                h = self.layers[idx](
                    h,
                    layer_cache,
                    valid_token_scale=valid_token_scale,
                    attn_mask=attn_mask,
                    rope_trig=rope_trig,
                    owner_idx=owner_idx,
                    owner_is_identity=owner_is_identity,
                    write_state=(write_state and cache is not None),
                )
            return self.norm(h)

    backbone = _Qwen35Backbone()
    backbone.eval()
    mx.eval(backbone.parameters())
    return backbone


def _resolve_original_kev_tokenizer_file(
    snap_dir: Path,
    *,
    model_variant: str,
    explicit_tokenizer: Path | str | None = None,
    vocab_size: int | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve original Kev `tokenizer.json` (preserving exact pre-tokenization across MLX repacks)."""
    if explicit_tokenizer is not None:
        tp = Path(explicit_tokenizer).expanduser().resolve()
        if tp.is_file():
            return tp
        raise FileNotFoundError(f"Missing explicit Kev tokenizer file: {tp}")

    # Custom/synthetic smaller-vocab checkpoints or local staged kev_08b/kev_4b dirs use their own tokenizer.json
    if (
        (vocab_size is not None and int(vocab_size) < 200000 and (snap_dir / "tokenizer.json").is_file())
        or (snap_dir / "kev_meta.json").is_file()
        or (snap_dir / "kev_meta_w8.json").is_file()
    ):
        local_tok = snap_dir / "tokenizer.json"
        if local_tok.is_file():
            return local_tok

    # Check local staged kev_08b / kev_4b artifacts for the original Kev tokenizer.json first (zero network)
    for v_tag in ((model_variant, "0.8b") if model_variant == "4b" else ("0.8b", "4b")):
        cand_dir = resolve_kev_dir(variant=v_tag)
        cand_tok = cand_dir / "tokenizer.json"
        if cand_tok.is_file():
            return cand_tok

    # Check cached HuggingFace adapter repo (`jaredpalmer/kev-4b` or `jaredpalmer/kev-0.8b`)
    adapter_repos = (
        (DEFAULT_KEV_4B_ADAPTER_REPO, DEFAULT_KEV_ADAPTER_REPO)
        if model_variant == "4b"
        else (DEFAULT_KEV_ADAPTER_REPO, DEFAULT_KEV_4B_ADAPTER_REPO)
    )
    try:
        from huggingface_hub import hf_hub_download

        for repo in adapter_repos:
            try:
                cached = Path(hf_hub_download(repo_id=repo, filename="tokenizer.json", local_files_only=True))
                if cached.is_file():
                    return cached
            except Exception:
                continue
    except Exception:
        pass

    fallback = snap_dir / "tokenizer.json"
    if fallback.is_file():
        return fallback

    if not local_files_only:
        try:
            from huggingface_hub import hf_hub_download

            primary = DEFAULT_KEV_4B_ADAPTER_REPO if model_variant == "4b" else DEFAULT_KEV_ADAPTER_REPO
            dl = Path(hf_hub_download(repo_id=primary, filename="tokenizer.json", local_files_only=False))
            if dl.is_file():
                return dl
        except Exception:
            pass

    raise FileNotFoundError(f"Missing tokenizer.json for Kev ({model_variant}) in {snap_dir}")


class KevMlxBackend:
    """Unified Metal MLX runner for `Kev-0.8B` & `Kev-4B` (prebuilt `AbstractFramework/*` + local fused checkpoints)."""

    _DEFAULT_VARIANT: str = "0.8b"

    @staticmethod
    def _resolve_prefix_tree(
        *,
        prefix_tree: PrefixTree | None,
        prefix_cache_mb: int | Literal["auto"] | None,
        prefix_chunk_tokens: int,
        host_mem_bytes: int,
        profile: Literal["auto", "backfill", "interactive"] | str = "auto",
        prefix_admission_min_hits: int | None = None,
    ) -> tuple[PrefixTree | None, bool]:
        if prefix_tree is not None:
            return prefix_tree, False
        eff_prof = str(profile).strip().lower()
        eff_min_hits = (
            max(1, int(prefix_admission_min_hits))
            if prefix_admission_min_hits is not None
            else (1 if eff_prof == "interactive" else 2)
        )
        if prefix_cache_mb == "auto":
            return (
                PrefixTree(
                    chunk_tokens=int(prefix_chunk_tokens),
                    byte_budget=recommended_prefix_budget_bytes(host_mem_bytes),
                    admission_min_hits=eff_min_hits,
                ),
                True,
            )
        if prefix_cache_mb is not None and not isinstance(prefix_cache_mb, bool) and int(prefix_cache_mb) > 0:
            explicit_min_hits = (
                max(1, int(prefix_admission_min_hits))
                if prefix_admission_min_hits is not None
                else 1
            )
            return (
                PrefixTree(
                    chunk_tokens=int(prefix_chunk_tokens),
                    byte_budget=int(prefix_cache_mb) << 20,
                    admission_min_hits=explicit_min_hits,
                ),
                True,
            )
        return None, False

    def __init__(
        self,
        kev_dir: Path | str | None = None,
        *,
        model_variant: Literal["0.8b", "4b"] | None = None,
        quant_mode: Literal["auto", "w8", "w4", "bf16"] = "auto",
        batch_docs: int = 8,
        max_state: int = 1024,
        max_branch: int = 2048,
        max_branch_batch: int | None = None,
        prefix_tree: PrefixTree | None = None,
        prefix_cache_mb: int | Literal["auto"] | None = "auto",
        prefix_chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        prefix_snapshot_stride: int = 1,
        prefix_admission_min_hits: int | None = None,
        compress_gdn_prefix_int8: bool | Literal["auto"] = "auto",
        profile: Literal["auto", "backfill", "interactive"] = "auto",
        tokenizer_path: Path | str | None = None,
        auto_prebuilt: bool = True,
        local_files_only: bool = False,
    ) -> None:
        import mlx.core as mx
        import numpy as np
        from tokenizers import Tokenizer

        self._mx = mx
        self._np = np
        eff_variant = str(model_variant or self._DEFAULT_VARIANT).strip().lower()
        self.model_variant = "4b" if "4b" in eff_variant else "0.8b"
        self.profile = str(profile).strip().lower()
        self._batched_lcp_enabled = True
        self.batch_docs = max(1, int(batch_docs))
        self.max_state = int(max_state)
        self.max_branch = int(max_branch)
        self.mlx_enable_tf32 = os.environ.get("MLX_ENABLE_TF32", "1")
        self._prev_cache_limit: int | None = None
        self._prev_wired_limit: int | None = None
        self._limits_restored = False
        eff_offline = bool(local_files_only) or (
            os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        )

        def _resolve_kev_hf_repo(repo_id: str) -> Path:
            from fast_shared.hf import resolve_hf_model_dir

            if eff_offline:
                return resolve_hf_model_dir(None, repo_id, local_files_only=True)
            try:
                return resolve_hf_model_dir(None, repo_id, local_files_only=True)
            except Exception:
                return resolve_hf_model_dir(None, repo_id, local_files_only=False)

        # Resolve directory or HF repo ID
        resolved_dir: Path | None = None
        is_repo_id = (
            isinstance(kev_dir, str)
            and kev_dir.count("/") == 1
            and not kev_dir.startswith(("/", ".", "~"))
            and not Path(kev_dir).expanduser().exists()
        )
        if is_repo_id:
            assert isinstance(kev_dir, str)
            if "4b" in kev_dir.lower():
                self.model_variant = "4b"
            resolved_dir = _resolve_kev_hf_repo(kev_dir)
        elif kev_dir is not None:
            cand_exp = Path(kev_dir).expanduser().resolve()
            if (
                not cand_exp.exists()
                and auto_prebuilt
                and cand_exp.name in ("kev_08b", "kev_4b")
                and cand_exp.parent.name == "artifacts"
            ):
                default_repo = DEFAULT_KEV_4B_MLX_REPO if self.model_variant == "4b" else DEFAULT_KEV_08B_MLX_REPO
                resolved_dir = _resolve_kev_hf_repo(default_repo)
            else:
                resolved_dir = cand_exp
        else:
            cand_local = resolve_kev_4b_dir() if self.model_variant == "4b" else resolve_kev_dir()
            if cand_local.is_dir() and (
                (cand_local / "kev_meta.json").is_file()
                or (cand_local / "kev_meta_w8.json").is_file()
                or (cand_local / "config.json").is_file()
            ):
                resolved_dir = cand_local
            elif auto_prebuilt:
                default_repo = DEFAULT_KEV_4B_MLX_REPO if self.model_variant == "4b" else DEFAULT_KEV_08B_MLX_REPO
                resolved_dir = _resolve_kev_hf_repo(default_repo)
            else:
                resolved_dir = cand_local

        assert resolved_dir is not None
        self.kev_dir = resolved_dir

        w8_sf = self.kev_dir / "model_w8.safetensors"
        w8_meta = self.kev_dir / "kev_meta_w8.json"
        bf16_sf = self.kev_dir / "model.safetensors"
        bf16_meta = self.kev_dir / "kev_meta.json"
        hf_cfg_p = self.kev_dir / "config.json"
        hf_dec_p = self.kev_dir / "decision.json"

        is_local_staged = w8_meta.is_file() or bf16_meta.is_file()
        is_prebuilt_mlx = hf_cfg_p.is_file() and (
            (self.kev_dir / "head.safetensors").is_file() or hf_dec_p.is_file()
        )

        if not is_local_staged and not is_prebuilt_mlx:
            raise FileNotFoundError(
                f"Missing Kev ({self.model_variant}) artifacts in {self.kev_dir}. "
                "Run 'decision-models setup-weights --build-kev-mlx' or enable prebuilt download."
            )

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
            profile=self.profile,
            prefix_admission_min_hits=prefix_admission_min_hits,
        )

        try:
            if is_local_staged:
                use_w8 = (quant_mode in ("w8", "w4")) or (
                    quant_mode == "auto" and w8_sf.is_file() and w8_meta.is_file()
                )
                sf_p = w8_sf if use_w8 else bf16_sf
                meta_p = w8_meta if use_w8 else bf16_meta
                if not (meta_p.is_file() and sf_p.is_file()):
                    raise FileNotFoundError(f"Missing Kev artifacts ({meta_p.name} / {sf_p.name}) in {self.kev_dir}")
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                self.quantization = meta.get("quantization")
                q_bits = int(self.quantization.get("bits", 8)) if isinstance(self.quantization, Mapping) else 16
                self.quant_mode: str = f"w{q_bits}" if use_w8 else "bf16"
                text_cfg = meta["text_config"]
                tok_p = _resolve_original_kev_tokenizer_file(
                    self.kev_dir,
                    model_variant=self.model_variant,
                    explicit_tokenizer=tokenizer_path,
                    vocab_size=int(text_cfg.get("vocab_size", 248320)),
                    local_files_only=eff_offline,
                )
                if int(text_cfg.get("hidden_size", 1024)) >= 2048:
                    self.model_variant = "4b"
                self.native_temperature = float(meta["native_temperature"])
                delimiters = meta.get("special_tokens", KEV_SPECIAL_TOKENS)
                self.pad_id = int(meta["pad_id"])

                self.tok = Tokenizer.from_file(str(tok_p))
                self.tok.no_truncation()
                self.tok.no_padding()
                self.special_ids = (
                    [int(x) for x in meta["special_ids"]]
                    if "special_ids" in meta
                    else resolve_kev_special_ids(self.tok, delimiters)
                )

                weights = mx.load(str(sf_p))
                self.q_w = np.asarray(weights["pointer_head.q_weight"].astype(mx.float32))
                self.q_b = np.asarray(weights["pointer_head.q_bias"].astype(mx.float32))
                self.k_w = np.asarray(weights["pointer_head.k_weight"].astype(mx.float32))
                self.k_b = np.asarray(weights["pointer_head.k_bias"].astype(mx.float32))
            else:
                raw_cfg = json.loads(hf_cfg_p.read_text(encoding="utf-8"))
                text_cfg = dict(raw_cfg.get("text_config", raw_cfg))
                if int(text_cfg.get("hidden_size", 1024)) >= 2048:
                    self.model_variant = "4b"
                dec_cfg = json.loads(hf_dec_p.read_text(encoding="utf-8")) if hf_dec_p.is_file() else {}
                self.quantization = raw_cfg.get("quantization")
                q_bits = int(self.quantization.get("bits", 8)) if isinstance(self.quantization, Mapping) else 16
                self.quant_mode = f"w{q_bits}" if isinstance(self.quantization, Mapping) else "bf16"
                self.native_temperature = float(dec_cfg.get("temperature", 2.3510958125672174))
                delimiters = dec_cfg.get("delimiters", KEV_SPECIAL_TOKENS)
                self.pad_id = int(text_cfg.get("eos_token_id", 248044))

                tok_p = _resolve_original_kev_tokenizer_file(
                    self.kev_dir,
                    model_variant=self.model_variant,
                    explicit_tokenizer=tokenizer_path,
                    vocab_size=int(text_cfg.get("vocab_size", 248320)),
                    local_files_only=eff_offline,
                )
                self.tok = Tokenizer.from_file(str(tok_p))
                self.tok.no_truncation()
                self.tok.no_padding()
                self.special_ids = resolve_kev_special_ids(self.tok, delimiters)

                head_file = str(dec_cfg.get("head_file", "head.safetensors"))
                head_p = self.kev_dir / head_file
                if not head_p.is_file():
                    raise FileNotFoundError(f"Missing PointerHead safetensors {head_p}")
                hw = mx.load(str(head_p))
                self.q_w = np.asarray(hw["q.weight"].astype(mx.float32))
                self.q_b = np.asarray(hw["q.bias"].astype(mx.float32))
                self.k_w = np.asarray(hw["k.weight"].astype(mx.float32))
                self.k_b = np.asarray(hw["k.bias"].astype(mx.float32))
                del hw

                model_shards = [
                    p
                    for p in sorted(self.kev_dir.glob("*.safetensors"))
                    if p.name != head_p.name and not p.name.startswith("head")
                ]
                if not model_shards:
                    raise FileNotFoundError(f"Missing model*.safetensors shards in {self.kev_dir}")
                weights = {}
                for sh in model_shards:
                    sh_w = mx.load(str(sh))
                    weights.update(sh_w)
                    del sh_w

            self.ptr_scale = 1.0 / math.sqrt(float(self.q_w.shape[0]))
            self.q_w_mx = mx.array(self.q_w, dtype=mx.float32)
            self.q_b_mx = mx.array(self.q_b, dtype=mx.float32)
            self.k_w_mx = mx.array(self.k_w, dtype=mx.float32)
            self.k_b_mx = mx.array(self.k_b, dtype=mx.float32)
            mx.eval(self.q_w_mx, self.q_b_mx, self.k_w_mx, self.k_b_mx)

            self.backbone = _build_qwen35_backbone(text_cfg, weights, quant_cfg=self.quantization)
            self._full_layer_indices = [i for i, blk in enumerate(self.backbone.layers) if not blk.is_linear]
            self._linear_layer_indices = [i for i, blk in enumerate(self.backbone.layers) if blk.is_linear]
            if compress_gdn_prefix_int8 == "auto":
                self.compress_gdn_prefix_int8 = self.model_variant == "4b"
            else:
                self.compress_gdn_prefix_int8 = bool(compress_gdn_prefix_int8)
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

    def __enter__(self) -> "KevMlxBackend":
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

    def _score_pointer_fp32(
        self,
        h_mx: Any,
        decide_indices: Sequence[int],
        opts_list: Sequence[list[int]],
        *,
        temperature: float | None = None,
    ) -> list[tuple[list[float], list[float]]]:
        """Score `[decide, *opts]` on GPU in FP32, copying only `B * K_max` scalars to CPU."""
        mx = self._mx
        np = self._np
        q_w_mx = getattr(self, "q_w_mx", None)
        if q_w_mx is None:
            mx.eval(h_mx)
            h_np = np.asarray(h_mx.astype(mx.float32))
            return [
                self._pointer_logits_and_probs(h_np[ri], int(decide_indices[ri]), list(opts_list[ri]), temperature=temperature)
                for ri in range(len(decide_indices))
            ]

        B = len(decide_indices)
        k_max = max((len(opts) for opts in opts_list), default=0)
        if B == 0 or k_max == 0:
            return [([], []) for _ in range(B)]

        gather_np = np.zeros((B, 1 + k_max), dtype=np.int32)
        for ri in range(B):
            gather_np[ri, 0] = int(decide_indices[ri])
            opts = opts_list[ri]
            if opts:
                gather_np[ri, 1 : 1 + len(opts)] = opts

        row_idx = mx.arange(B, dtype=mx.int32)[:, None]
        pos_idx = mx.array(gather_np, dtype=mx.int32)
        h_sel = h_mx[row_idx, pos_idx].astype(mx.float32)
        h_dec = h_sel[:, 0, :]
        h_opt = h_sel[:, 1:, :]
        q_vec = (h_dec @ q_w_mx.T + self.q_b_mx)[:, :, None]
        k_mat = h_opt @ self.k_w_mx.T + self.k_b_mx
        raw_logits_mx = mx.matmul(k_mat, q_vec).squeeze(-1) * float(self.ptr_scale)
        mx.eval(raw_logits_mx)
        raw_np = np.asarray(raw_logits_mx)

        temp = self.native_temperature if temperature is None else float(temperature)
        outs: list[tuple[list[float], list[float]]] = []
        for ri in range(B):
            k_i = len(opts_list[ri])
            raw_z = [float(x) for x in raw_np[ri, :k_i].tolist()]
            outs.append((softmax_vec(raw_z, temp), raw_z))
        return outs

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

    def _restore_single_cache_from_match(self, match: Any) -> list[Any]:
        mx = self._mx
        cache = self.backbone.make_cache()
        pos = int(match.reused_tokens)
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
            rs_entry = rec_snaps[gdn_i]
            if isinstance(rs_entry, tuple) and len(rs_entry) == 3:
                from fast_shared.prefix_tree import dequantize_gdn_state_block32

                codes, scales, orig_dt = rs_entry
                cache[l_idx].rec_state = dequantize_gdn_state_block32(codes, scales, out_dtype=orig_dt)
            else:
                cache[l_idx].rec_state = rs_entry
        return cache

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
            cache = self._restore_single_cache_from_match(match)
            tree.record(match, restored=True)
        else:
            tree.record(match, restored=False)

        eff_prof = getattr(self, "profile", "auto")
        if pos == 0 and not tree.should_write_snapshots(eff_prof):
            h_full = self.backbone(mx.array([state_ids], dtype=mx.int32), cache, write_state=True)
            mx.eval(h_full)
            return cache

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
                    use_i8_gdn = bool(getattr(self, "compress_gdn_prefix_int8", False))
                    rec_list: list[Any] = []
                    eval_tensors: list[Any] = [h_c, *k_chunk, *v_chunk, *conv_snap]
                    for l_idx in self._linear_layer_indices:
                        rs = cache[l_idx].rec_state
                        if (
                            use_i8_gdn
                            and isinstance(getattr(mx, "array", None), type)
                            and isinstance(rs, mx.array)
                            and rs.ndim >= 1
                            and int(rs.shape[-1]) >= 32
                            and int(rs.shape[-1]) % 32 == 0
                        ):
                            from fast_shared.prefix_tree import (
                                dequantize_gdn_state_block32,
                                quantize_gdn_state_block32,
                            )

                            q_c, s_c = quantize_gdn_state_block32(rs)
                            dq_rs = dequantize_gdn_state_block32(q_c, s_c, out_dtype=rs.dtype)
                            cache[l_idx].rec_state = dq_rs
                            rec_list.append((q_c, s_c, rs.dtype))
                            eval_tensors.extend((q_c, s_c, dq_rs))
                        else:
                            rc = mx.contiguous(rs)
                            rec_list.append(rc)
                            eval_tensors.append(rc)
                    rec_snap = tuple(rec_list)
                    mx.eval(*eval_tensors)
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
            doc_outs.extend(
                self._score_pointer_fp32(
                    h_mx,
                    [int(r["decide"]) for r in sub_rows],
                    [list(r["opts"]) for r in sub_rows],
                    temperature=temperature,
                )
            )
        return doc_outs

    def _run_one_pass_single_question_waves(
        self,
        single_q_indices: Sequence[int],
        encoded_all: Sequence[tuple[list[int], list[dict[str, Any]]]],
        out_by_orig: list[list[tuple[list[float], list[float]]]],
        *,
        eff_b: int,
        temperature: float | None = None,
    ) -> None:
        """Execute 1-pass `cache=None` scoring (`state_ids + branch_ids`) + GPU `PointerHead` on single-question items."""
        if not single_q_indices:
            return
        mx = self._mx
        np = self._np

        entries: list[tuple[int, list[int], int, list[int]]] = []
        for orig_i in single_q_indices:
            state_ids, rows = encoded_all[orig_i]
            r0 = rows[0]
            q_off = len(state_ids)
            full_ids = state_ids + r0["ids"]
            dec_pos = q_off + int(r0["decide"])
            opt_pos = [q_off + int(p) for p in r0["opts"]]
            entries.append((orig_i, full_ids, dec_pos, opt_pos))

        waves = _pack_tight_length_waves(entries, eff_b)
        for wave in waves:
            B = len(wave)
            f_lens = [len(e[1]) for e in wave]
            s_max = max(f_lens)
            pad_lens = [s_max - ln for ln in f_lens]
            all_unpadded = all(pl == 0 for pl in pad_lens)

            if all_unpadded:
                ids_mx = mx.array([e[1] for e in wave], dtype=mx.int32)
                valid_scale_mx = None
                pos_mx = None
                mask_mx: Any = "causal"
            else:
                st_np = np.full((B, s_max), self.pad_id, dtype=np.int32)
                for b_i, e in enumerate(wave):
                    pl = pad_lens[b_i]
                    st_np[b_i, pl:] = e[1]
                ids_mx = mx.array(st_np, dtype=mx.int32)
                pl_mx = mx.array(pad_lens, dtype=mx.int32)[:, None]
                seq_pos = mx.arange(s_max, dtype=mx.int32)[None, :]
                valid_scale_mx = mx.where(
                    seq_pos >= pl_mx,
                    mx.array(1.0, dtype=mx.bfloat16),
                    mx.array(0.0, dtype=mx.bfloat16),
                )[:, :, None]
                pos_mx = mx.maximum(seq_pos - pl_mx, 0)
                mask_mx = _build_state_attn_mask(mx, pad_lens, s_max, mx.bfloat16)

            h_mx = self.backbone(
                ids_mx,
                None,
                valid_token_scale=valid_scale_mx,
                attn_mask=mask_mx,
                pos_2d=pos_mx,
                write_state=False,
            )
            dec_shifted = [pad_lens[b_i] + wave[b_i][2] for b_i in range(B)]
            opts_shifted = [[pad_lens[b_i] + p for p in wave[b_i][3]] for b_i in range(B)]
            scored = self._score_pointer_fp32(h_mx, dec_shifted, opts_shifted, temperature=temperature)
            for b_i, (orig_i, *_rest) in enumerate(wave):
                out_by_orig[orig_i] = [scored[b_i]]
            del h_mx
            self._maybe_clear_wave_cache()

    def _score_candidates_against_prefix_cache(
        self,
        prefix_cache: list[Any],
        prefix_lens: Sequence[int],
        prefix_pads: Sequence[int],
        s_max_prefix: int,
        group_members: Sequence[list[int]],
        encoded_all: Sequence[tuple[list[int], list[dict[str, Any]]]],
        out_by_orig: list[list[tuple[list[float], list[float]]]],
        *,
        br_chunk_lim: int,
        eff_b: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Evaluate all sibling candidate `tail + branch` rows in batched `B=8..16` waves against `prefix_cache`."""
        mx = self._mx
        np = self._np
        supports_fused = bool(getattr(self.backbone, "_supports_stateless_fused_pass", False))

        if not supports_fused:
            # Compatibility path for two-stage stateful mocks (`_FakeBackbone` in unit tests)
            cand_items: list[tuple[int, int, list[int], list[dict[str, Any]]]] = []
            for g_i, members in enumerate(group_members):
                p_len = prefix_lens[g_i]
                for orig_i in members:
                    st_ids, rows = encoded_all[orig_i]
                    cand_items.append((orig_i, g_i, st_ids[p_len:], rows))

            for c0 in range(0, len(cand_items), br_chunk_lim):
                sub = cand_items[c0 : c0 + br_chunk_lim]
                n_sub = len(sub)
                owners = [x[1] for x in sub]
                cand_cache = self.backbone.make_cache()
                for l_idx in self._linear_layer_indices:
                    cand_cache[l_idx].rec_state = prefix_cache[l_idx].rec_state[owners]
                    cand_cache[l_idx].conv_state = prefix_cache[l_idx].conv_state[owners]
                for l_idx in self._full_layer_indices:
                    cand_cache[l_idx].keys = prefix_cache[l_idx].keys[owners]
                    cand_cache[l_idx].values = prefix_cache[l_idx].values[owners]
                    cand_cache[l_idx].offset = s_max_prefix

                tails = [x[2] for x in sub]
                t_max = max(len(t) for t in tails)
                if t_max > 0:
                    t_pads = [t_max - len(t) for t in tails]
                    t_np = np.full((n_sub, t_max), self.pad_id, dtype=np.int32)
                    for r_i, t in enumerate(tails):
                        if t:
                            t_np[r_i, t_pads[r_i] :] = t
                    pl_mx = mx.array(t_pads, dtype=mx.int32)[:, None]
                    seq_pos = mx.arange(t_max, dtype=mx.int32)[None, :]
                    vscale = mx.where(
                        seq_pos >= pl_mx,
                        mx.array(1.0, dtype=mx.bfloat16),
                        mx.array(0.0, dtype=mx.bfloat16),
                    )[:, :, None]
                    _ = self.backbone(mx.array(t_np, dtype=mx.int32), cand_cache, valid_token_scale=vscale, write_state=True)

                flat_q: list[tuple[int, int, dict[str, Any]]] = []
                for r_i, (orig_i, _gi, _tl, rows) in enumerate(sub):
                    out_by_orig[orig_i] = [([], []) for _ in rows]
                    for q_i, r in enumerate(rows):
                        flat_q.append((r_i, q_i, r))
                if flat_q:
                    max_b = max(len(x[2]["ids"]) for x in flat_q)
                    b_np = np.full((len(flat_q), max_b), self.pad_id, dtype=np.int32)
                    for qi_idx, (_, _, r) in enumerate(flat_q):
                        b_np[qi_idx, : len(r["ids"])] = r["ids"]
                    q_owners = mx.array([x[0] for x in flat_q], dtype=mx.int32)
                    h_br = self.backbone(mx.array(b_np, dtype=mx.int32), cand_cache, owner_idx=q_owners, write_state=False)
                    scored = self._score_pointer_fp32(
                        h_br,
                        [int(x[2]["decide"]) for x in flat_q],
                        [list(x[2]["opts"]) for x in flat_q],
                        temperature=temperature,
                    )
                    for qi_idx, (r_i, q_i, _) in enumerate(flat_q):
                        orig_i = sub[r_i][0]
                        out_by_orig[orig_i][q_i] = scored[qi_idx]
            return

        flat_cands: list[tuple[int, int, int, list[int], int, list[int]]] = []
        for g_i, members in enumerate(group_members):
            p_len = prefix_lens[g_i]
            for orig_i in members:
                st_ids, rows = encoded_all[orig_i]
                out_by_orig[orig_i] = [([], []) for _ in rows]
                tail_ids = st_ids[p_len:]
                t_off = len(tail_ids)
                for q_i, r in enumerate(rows):
                    seq_ids = tail_ids + r["ids"]
                    dec_pos = t_off + int(r["decide"])
                    opt_pos = [t_off + int(p) for p in r["opts"]]
                    flat_cands.append((orig_i, q_i, g_i, seq_ids, dec_pos, opt_pos))

        if not flat_cands:
            return

        # Sort tails deterministically by length so multi-candidate groups don't right-pad short tails
        # out to the longest tail in unsorted batches; cap wave batch by `eff_b` when long tails exist.
        flat_cands.sort(key=lambda x: (len(x[3]), x[0], x[1]))
        max_tail = max(len(x[3]) for x in flat_cands)
        tail_step = min(br_chunk_lim, eff_b) if (eff_b is not None and max_tail > 64) else br_chunk_lim

        N_tot = len(flat_cands)
        n_groups = len(group_members)
        for c0 in range(0, N_tot, tail_step):
            sub = flat_cands[c0 : c0 + tail_step]
            n_sub = len(sub)
            max_c = max(len(x[3]) for x in sub)
            c_np = np.full((n_sub, max_c), self.pad_id, dtype=np.int32)
            for r_i, x in enumerate(sub):
                seq = x[3]
                c_np[r_i, : len(seq)] = seq

            sub_owners = [x[2] for x in sub]
            sub_pads = [prefix_pads[g_i] for g_i in sub_owners]
            sub_plens = [prefix_lens[g_i] for g_i in sub_owners]
            sub_unpadded = all(pl == 0 for pl in sub_pads)
            if sub_unpadded:
                pos_c_mx = None
                mask_c_mx: Any = "causal"
            else:
                pos_c_mx = (
                    mx.array(sub_plens, dtype=mx.int32)[:, None]
                    + mx.arange(max_c, dtype=mx.int32)[None, :]
                )
                mask_c_mx = _build_branch_attn_mask(mx, sub_pads, s_max_prefix, max_c, mx.bfloat16)

            owner_is_id = (n_sub == n_groups) and (sub_owners == list(range(n_groups)))
            owner_mx = mx.array(sub_owners, dtype=mx.int32)
            h_c = self.backbone(
                mx.array(c_np, dtype=mx.int32),
                prefix_cache,
                valid_token_scale=None,
                attn_mask=mask_c_mx,
                pos_2d=pos_c_mx,
                owner_idx=owner_mx,
                owner_is_identity=owner_is_id,
                write_state=False,
            )
            scored = self._score_pointer_fp32(
                h_c,
                [x[4] for x in sub],
                [x[5] for x in sub],
                temperature=temperature,
            )
            for r_i, (orig_i, q_i, *_rest) in enumerate(sub):
                out_by_orig[orig_i][q_i] = scored[r_i]

    def _run_batched_shared_prefix_groups(
        self,
        groups: Sequence[tuple[list[int], int]],
        encoded_all: Sequence[tuple[list[int], list[dict[str, Any]]]],
        out_by_orig: list[list[tuple[list[float], list[float]]]],
        *,
        eff_b: int,
        br_chunk_lim: int,
        temperature: float | None = None,
    ) -> None:
        """Prefill unique shared prefixes in `B=eff_b` waves and evaluate candidate tails batched at `B=8..16`."""
        if not groups:
            return
        mx = self._mx
        np = self._np
        tree = self.prefix_tree
        has_tree = tree is not None and tree.byte_budget > 0
        chunk_sz = tree.chunk_tokens if tree is not None else DEFAULT_CHUNK_TOKENS
        eff_prof = getattr(self, "profile", "auto")

        warm_groups: list[tuple[list[int], int, Any]] = []
        cold_groups: list[tuple[list[int], int, bool]] = []

        for members, lcp_len in groups:
            rep_state = encoded_all[members[0]][0]
            if has_tree and lcp_len >= chunk_sz:
                assert tree is not None
                m = tree.match(rep_state, limit=lcp_len, needs_snapshot=True)
                if m.reused_tokens >= chunk_sz:
                    for _ in members:
                        tree.record(m, restored=True)
                    warm_groups.append((members, int(m.reused_tokens), m))
                    continue
                admit_fn = getattr(tree, "admit_prefix", None)
                admitted = (
                    bool(admit_fn(rep_state, profile=eff_prof, sightings=len(members)))
                    if callable(admit_fn)
                    else bool(tree.should_write_snapshots(eff_prof))
                )
                if admitted:
                    aligned_p = (lcp_len // chunk_sz) * chunk_sz
                    cold_groups.append((members, aligned_p, True))
                else:
                    tree.record(
                        m,
                        restored=False,
                        count_for_circuit_breaker=(
                            tree.circuit_breaker_open or getattr(tree, "admission_min_hits", 1) <= 1
                        ),
                    )
                    cold_groups.append((members, lcp_len, False))
            else:
                cold_groups.append((members, lcp_len, False))

        # 1. Warm PrefixTree hit groups: group by restored prefix length so KV concat aligns without padding,
        #    evaluating all sibling tails + branches in batched B=8..16 waves.
        by_warm_len: dict[int, list[tuple[list[int], Any]]] = defaultdict(list)
        for members, p_len, match in warm_groups:
            by_warm_len[p_len].append((members, match))

        for p_len, w_list in sorted(by_warm_len.items()):
            for g0 in range(0, len(w_list), eff_b):
                w_sub = w_list[g0 : g0 + eff_b]
                single_caches = [self._restore_single_cache_from_match(m) for _, m in w_sub]
                if len(single_caches) == 1:
                    merged_cache = single_caches[0]
                else:
                    merged_cache = self.backbone.make_cache()
                    for l_idx in self._full_layer_indices:
                        merged_cache[l_idx].keys = mx.concatenate([sc[l_idx].keys for sc in single_caches], axis=0)
                        merged_cache[l_idx].values = mx.concatenate([sc[l_idx].values for sc in single_caches], axis=0)
                        merged_cache[l_idx].offset = p_len
                    for l_idx in self._linear_layer_indices:
                        merged_cache[l_idx].conv_state = mx.concatenate([sc[l_idx].conv_state for sc in single_caches], axis=0)
                        merged_cache[l_idx].rec_state = mx.concatenate([sc[l_idx].rec_state for sc in single_caches], axis=0)

                self._score_candidates_against_prefix_cache(
                    merged_cache,
                    [p_len] * len(w_sub),
                    [0] * len(w_sub),
                    p_len,
                    [members for members, _ in w_sub],
                    encoded_all,
                    out_by_orig,
                    br_chunk_lim=br_chunk_lim,
                    eff_b=eff_b,
                    temperature=temperature,
                )
                del merged_cache, single_caches
                self._maybe_clear_wave_cache()

        # 2. Cold PrefixTree-published groups: partition by `p_len` (`by_cold_len`, matching `by_warm_len`)
        #    so `cold` and `warm` execute identical unpadded `(B_g, p_len)` and `(n_sub, max_c)` shapes (`0.0` ULP diff!).
        publish_by_len: dict[int, list[list[int]]] = defaultdict(list)
        ephemeral_cold: list[tuple[list[int], int]] = []
        for members, p_len, publish in cold_groups:
            if publish and tree is not None and p_len >= chunk_sz:
                publish_by_len[p_len].append(members)
            else:
                ephemeral_cold.append((members, p_len))

        for p_len, member_lists in sorted(publish_by_len.items()):
            for g0 in range(0, len(member_lists), eff_b):
                c_members = member_lists[g0 : g0 + eff_b]
                B_g = len(c_members)
                pref_np = np.full((B_g, p_len), self.pad_id, dtype=np.int32)
                for b_i, members in enumerate(c_members):
                    pref_np[b_i, :] = encoded_all[members[0]][0][:p_len]

                p_cache = self.backbone.make_cache()
                h_pref = self.backbone(
                    mx.array(pref_np, dtype=mx.int32),
                    p_cache,
                    valid_token_scale=None,
                    attn_mask="causal",
                    pos_2d=None,
                    write_state=True,
                )
                mx.eval(h_pref)

                assert tree is not None
                for b_i, members in enumerate(c_members):
                    rep_ids = encoded_all[members[0]][0][:p_len]
                    m_cold = tree.match(rep_ids, limit=p_len, needs_snapshot=True)
                    tree.record(m_cold, restored=False)
                    parent_k = m_cold.keys[-1] if m_cold.keys else None
                    start_d = len(m_cold.keys)
                    n_chunks = p_len // chunk_sz
                    for d_idx in range(start_d, n_chunks):
                        c_lo = d_idx * chunk_sz
                        c_hi = c_lo + chunk_sz
                        k_ch = tuple(
                            self._detach_kv_chunk(p_cache[l_idx].keys[b_i : b_i + 1], c_lo, c_hi)
                            for l_idx in self._full_layer_indices
                        )
                        v_ch = tuple(
                            self._detach_kv_chunk(p_cache[l_idx].values[b_i : b_i + 1], c_lo, c_hi)
                            for l_idx in self._full_layer_indices
                        )
                        is_last = d_idx == (n_chunks - 1)
                        if is_last:
                            c_snap = tuple(
                                mx.contiguous(p_cache[l_idx].conv_state[b_i : b_i + 1])
                                for l_idx in self._linear_layer_indices
                            )
                            r_snap = tuple(
                                mx.contiguous(p_cache[l_idx].rec_state[b_i : b_i + 1])
                                for l_idx in self._linear_layer_indices
                            )
                            mx.eval(*k_ch, *v_ch, *c_snap, *r_snap)
                        else:
                            c_snap = ()
                            r_snap = ()
                            mx.eval(*k_ch, *v_ch)
                        new_k = tree.insert(
                            parent=parent_k,
                            tokens=rep_ids[d_idx * chunk_sz : (d_idx + 1) * chunk_sz],
                            depth=d_idx + 1,
                            chunk=(k_ch, v_ch),
                        )
                        if new_k is None:
                            parent_k = None
                            break
                        parent_k = new_k
                        if is_last and not tree.has_snapshot(new_k):
                            tree.attach_snapshot(new_k, (c_snap, r_snap), keeping_chain_to=new_k)
                    if parent_k is not None and len(members) > 1:
                        m_hit = tree.match(rep_ids, limit=p_len, needs_snapshot=True)
                        if m_hit.reused_tokens >= chunk_sz:
                            for _ in members[1:]:
                                tree.record(m_hit, restored=True)

                for l_idx in self._full_layer_indices:
                    p_cache[l_idx].keys = mx.contiguous(p_cache[l_idx].keys)
                    p_cache[l_idx].values = mx.contiguous(p_cache[l_idx].values)
                for l_idx in self._linear_layer_indices:
                    p_cache[l_idx].conv_state = mx.contiguous(p_cache[l_idx].conv_state)
                    p_cache[l_idx].rec_state = mx.contiguous(p_cache[l_idx].rec_state)

                self._score_candidates_against_prefix_cache(
                    p_cache,
                    [p_len] * B_g,
                    [0] * B_g,
                    p_len,
                    c_members,
                    encoded_all,
                    out_by_orig,
                    br_chunk_lim=br_chunk_lim,
                    eff_b=eff_b,
                    temperature=temperature,
                )
                del p_cache, h_pref
                self._maybe_clear_wave_cache()

        # 3. Ephemeral cold shared-prefix groups (`publish=False`): sort by `P_g` and prefill in `B=eff_b` waves!
        cold_sorted = sorted(ephemeral_cold, key=lambda x: x[1])
        for g0 in range(0, len(cold_sorted), eff_b):
            c_sub = cold_sorted[g0 : g0 + eff_b]
            B_g = len(c_sub)
            p_lens = [x[1] for x in c_sub]
            s_max = max(p_lens)
            p_pads = [s_max - ln for ln in p_lens]
            all_unpadded = all(pl == 0 for pl in p_pads)

            pref_np = np.full((B_g, s_max), self.pad_id, dtype=np.int32)
            for b_i, (members, p_len) in enumerate(c_sub):
                pl = p_pads[b_i]
                pref_np[b_i, pl:] = encoded_all[members[0]][0][:p_len]

            if all_unpadded:
                valid_scale_mx = None
                pos_mx = None
                mask_mx: Any = "causal"
            else:
                pl_mx = mx.array(p_pads, dtype=mx.int32)[:, None]
                seq_pos = mx.arange(s_max, dtype=mx.int32)[None, :]
                valid_scale_mx = mx.where(
                    seq_pos >= pl_mx,
                    mx.array(1.0, dtype=mx.bfloat16),
                    mx.array(0.0, dtype=mx.bfloat16),
                )[:, :, None]
                pos_mx = mx.maximum(seq_pos - pl_mx, 0)
                mask_mx = _build_state_attn_mask(mx, p_pads, s_max, mx.bfloat16)

            p_cache = self.backbone.make_cache()
            h_pref = self.backbone(
                mx.array(pref_np, dtype=mx.int32),
                p_cache,
                valid_token_scale=valid_scale_mx,
                attn_mask=mask_mx,
                pos_2d=pos_mx,
                write_state=True,
            )
            mx.eval(h_pref)

            self._score_candidates_against_prefix_cache(
                p_cache,
                p_lens,
                p_pads,
                s_max,
                [x[0] for x in c_sub],
                encoded_all,
                out_by_orig,
                br_chunk_lim=br_chunk_lim,
                eff_b=eff_b,
                temperature=temperature,
            )
            del p_cache, h_pref
            self._maybe_clear_wave_cache()

    def _run_batched_pointer(
        self,
        items: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]],
        *,
        batch_docs: int | None = None,
        temperature: float | None = None,
    ) -> list[list[tuple[list[float], list[float]]]]:
        """Batch document states via exact-LCP grouping, 1-pass single-question fast path, and batched `PrefixTree`."""
        if not items:
            return []
        mx = self._mx
        np = self._np
        eff_b = max(1, int(batch_docs or self.batch_docs))
        br_chunk_lim = max(eff_b, self.max_branch_batch)

        encoded_all = [self.encode_questions(st, qs) for st, qs in items]
        out_by_orig: list[list[tuple[list[float], list[float]]]] = [[] for _ in items]
        batched_lcp = bool(getattr(self, "_batched_lcp_enabled", False))
        supports_fused_1pass = bool(getattr(self.backbone, "_supports_stateless_fused_pass", False))

        if not batched_lcp:
            # Legacy compatibility path for CPU unit mocks constructed via `__new__` without `_batched_lcp_enabled`
            order = sorted(range(len(items)), key=lambda idx: len(encoded_all[idx][0]))
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
                    m_leg = self.prefix_tree.match(state_ids, needs_snapshot=True)
                    admit_fn = getattr(self.prefix_tree, "admit_prefix", None)
                    eff_prof_leg = getattr(self, "profile", "auto")
                    use_prefix_tree = (
                        m_leg.reused_tokens >= chunk_sz
                        or prefix_counts[first_chunk] >= 2
                        or (
                            n_nonempty_items == 1
                            and (
                                bool(admit_fn(state_ids, profile=eff_prof_leg, sightings=1))
                                if callable(admit_fn)
                                else self.prefix_tree.should_write_snapshots(eff_prof_leg)
                            )
                        )
                    )
                    if not use_prefix_tree:
                        if n_nonempty_items == 1 and getattr(self.prefix_tree, "admission_min_hits", 1) > 1:
                            self.prefix_tree.record(
                                m_leg,
                                restored=False,
                                count_for_circuit_breaker=self.prefix_tree.circuit_breaker_open,
                            )
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
        else:
            nonempty_indices = [i for i in range(len(items)) if encoded_all[i][1]]
            if not nonempty_indices:
                return out_by_orig

            has_tree = self.prefix_tree is not None and self.prefix_tree.byte_budget > 0
            chunk_sz = self.prefix_tree.chunk_tokens if self.prefix_tree is not None else DEFAULT_CHUNK_TOKENS
            eff_prof = getattr(self, "profile", "auto")

            # B=1 interactive / single-item path: keep warm `PrefixTree` (`44-46 ms`) when >= chunk_sz,
            # or use 1-pass `cache=None` fast path when `< chunk_sz`, `prefix_tree` is off, or circuit-broken.
            if len(nonempty_indices) == 1:
                only_i = nonempty_indices[0]
                state_ids, rows = encoded_all[only_i]
                if has_tree and len(state_ids) >= chunk_sz:
                    assert self.prefix_tree is not None
                    m_one = self.prefix_tree.match(state_ids, limit=len(state_ids), needs_snapshot=True)
                    admit_fn = getattr(self.prefix_tree, "admit_prefix", None)
                    should_write = (
                        bool(admit_fn(state_ids, profile=eff_prof, sightings=1))
                        if callable(admit_fn)
                        else bool(self.prefix_tree.should_write_snapshots(eff_prof))
                    )
                    if m_one.reused_tokens >= chunk_sz or should_write:
                        cache = self._prefill_state_with_prefix_tree(
                            state_ids,
                            eval_restored_concat=(len(rows) > br_chunk_lim),
                        )
                        out_by_orig[only_i] = self._score_single_cache_branches(
                            cache, rows, br_chunk_lim, temperature=temperature
                        )
                        del cache
                        self._maybe_clear_wave_cache()
                        return out_by_orig
                    self.prefix_tree.record(
                        m_one,
                        restored=False,
                        count_for_circuit_breaker=(
                            self.prefix_tree.circuit_breaker_open
                            or getattr(self.prefix_tree, "admission_min_hits", 1) <= 1
                        ),
                    )

                if supports_fused_1pass and len(rows) == 1:
                    self._run_one_pass_single_question_waves(
                        [only_i], encoded_all, out_by_orig, eff_b=1, temperature=temperature
                    )
                    return out_by_orig
                order = [only_i]
            else:
                min_lcp = min(_MIN_LCP_GROUP_TOKENS, chunk_sz) if has_tree else _MIN_LCP_GROUP_TOKENS
                lcp_groups, singletons = _partition_exact_lcp_groups(
                    nonempty_indices, encoded_all, min_lcp=min_lcp
                )
                # Check singletons for warm PrefixTree hits as 1-member warm groups
                remaining_singletons: list[int] = []
                if has_tree:
                    assert self.prefix_tree is not None
                    for s_i in singletons:
                        s_ids = encoded_all[s_i][0]
                        if len(s_ids) >= chunk_sz:
                            m_s = self.prefix_tree.match(s_ids, limit=len(s_ids), needs_snapshot=True)
                            if m_s.reused_tokens >= chunk_sz:
                                lcp_groups.append(([s_i], int(m_s.reused_tokens)))
                                continue
                        remaining_singletons.append(s_i)
                else:
                    remaining_singletons = singletons

                if lcp_groups:
                    self._run_batched_shared_prefix_groups(
                        lcp_groups,
                        encoded_all,
                        out_by_orig,
                        eff_b=eff_b,
                        br_chunk_lim=br_chunk_lim,
                        temperature=temperature,
                    )

                if supports_fused_1pass:
                    one_q = [i for i in remaining_singletons if len(encoded_all[i][1]) == 1]
                    multi_q = [i for i in remaining_singletons if len(encoded_all[i][1]) > 1]
                    if one_q:
                        self._run_one_pass_single_question_waves(
                            one_q, encoded_all, out_by_orig, eff_b=eff_b, temperature=temperature
                        )
                    order = sorted(multi_q, key=lambda idx: len(encoded_all[idx][0]))
                else:
                    order = sorted(remaining_singletons, key=lambda idx: len(encoded_all[idx][0]))

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
                flat_outs.extend(
                    self._score_pointer_fp32(
                        h_br,
                        [int(r["decide"]) for r in sub_rows],
                        [list(r["opts"]) for r in sub_rows],
                        temperature=temperature,
                    )
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
        self.q_w_mx = None
        self.q_b_mx = None
        self.k_w_mx = None
        self.k_b_mx = None
        self._restore_mlx_limits()
        self._mx.clear_cache()


class Kev08bMlxBackend(KevMlxBackend):
    """Metal MLX `Kev-0.8B` runner (`W8` / prebuilt `AbstractFramework/kev-0.8b-mlx-8bit`, `PrefixTree` & 1-pass fast path)."""

    _DEFAULT_VARIANT = "0.8b"


class Kev4bMlxBackend(KevMlxBackend):
    """Metal MLX `Kev-4B` runner (`4-bit g32` MLP/Attn + `8-bit g32` GDN, `AbstractFramework/kev-4b-mlx-4bit`)."""

    _DEFAULT_VARIANT = "4b"


__all__ = [
    "KEV_SPECIAL_TOKENS",
    "Kev08bMlxBackend",
    "Kev4bMlxBackend",
    "KevMlxBackend",
    "_build_branch_attn_mask",
    "_build_state_attn_mask",
    "_detect_apple10_d256_force_fused",
    "encode_kev_questions",
    "resolve_kev_special_ids",
]
