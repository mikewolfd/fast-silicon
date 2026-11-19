from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from fast_shared.metal_mpp import (
    HEADER_SIMDGROUP as _HEADER_SG,
    HEADER_STD as _HEADER_STD,
    make_mpp_matmul_kernel,
)

from .paths import resolve_paths

# ---------------------------------------------------------------------------
# 1. Packed Embedding + LayerNorm (1 simdgroup / row, 32 threads x 12 halves)
# ---------------------------------------------------------------------------
_EMBED_LN_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)tok_ids_shape[0]) return;
    int tid_tok = tok_ids[row];
    int pid_tok = pos_ids[row];
    const device half* wt = W_wt0 + tid_tok * 384;
    const device half* wp = W_pos + pid_tok * 384;

    float4 f0 = float4(*(const device half4*)(wt + lane * 4))       + float4(*(const device half4*)(wp + lane * 4));
    float4 f1 = float4(*(const device half4*)(wt + 128 + lane * 4)) + float4(*(const device half4*)(wp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(wt + 256 + lane * 4)) + float4(*(const device half4*)(wp + 256 + lane * 4));

    float psum = (f0.x + f0.y + f0.z + f0.w) + (f1.x + f1.y + f1.z + f1.w) + (f2.x + f2.y + f2.z + f2.w);
    float mean = simd_sum(psum) * (1.0f / 384.0f);
    f0 -= mean; f1 -= mean; f2 -= mean;
    float pvar = dot(f0, f0) + dot(f1, f1) + dot(f2, f2);
    float inv_std = rsqrt(simd_sum(pvar) * (1.0f / 384.0f) + 1e-12f);

    float4 w0 = float4(*(const device half4*)(W_ln + lane * 4));
    float4 w1 = float4(*(const device half4*)(W_ln + 128 + lane * 4));
    float4 w2 = float4(*(const device half4*)(W_ln + 256 + lane * 4));
    float4 b0 = float4(*(const device half4*)(B_ln + lane * 4));
    float4 b1 = float4(*(const device half4*)(B_ln + 128 + lane * 4));
    float4 b2 = float4(*(const device half4*)(B_ln + 256 + lane * 4));

    device half* yp = Y + row * 384;
    *(device half4*)(yp + lane * 4)       = half4(f0 * inv_std * w0 + b0);
    *(device half4*)(yp + 128 + lane * 4) = half4(f1 * inv_std * w1 + b1);
    *(device half4*)(yp + 256 + lane * 4) = half4(f2 * inv_std * w2 + b2);
"""
_EMBED_LN_KERNEL = mx.fast.metal_kernel(
    name="bge_embed_packed_ln_f16",
    input_names=["tok_ids", "pos_ids", "W_wt0", "W_pos", "W_ln", "B_ln"],
    output_names=["Y"],
    header=_HEADER_STD,
    source=_EMBED_LN_SRC,
)

# ---------------------------------------------------------------------------
# 2. Fast LayerNorm [M, 384] half -> [M, 384] half (zero TG memory, 1 sg/row)
# ---------------------------------------------------------------------------
_LN_SRC = """
    uint sg = thread_position_in_threadgroup.x >> 5;
    uint lane = thread_position_in_threadgroup.x & 31;
    uint row = threadgroup_position_in_grid.x * 8 + sg;
    if (row >= (uint)X_shape[0]) return;
    const device half* xp = X + row * 384;
    float4 f0 = float4(*(const device half4*)(xp + lane * 4));
    float4 f1 = float4(*(const device half4*)(xp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(xp + 256 + lane * 4));

    float psum = (f0.x + f0.y + f0.z + f0.w) + (f1.x + f1.y + f1.z + f1.w) + (f2.x + f2.y + f2.z + f2.w);
    float mean = simd_sum(psum) * (1.0f / 384.0f);
    f0 -= mean; f1 -= mean; f2 -= mean;
    float pvar = dot(f0, f0) + dot(f1, f1) + dot(f2, f2);
    float inv_std = rsqrt(simd_sum(pvar) * (1.0f / 384.0f) + 1e-12f);

    float4 w0 = float4(*(const device half4*)(W + lane * 4));
    float4 w1 = float4(*(const device half4*)(W + 128 + lane * 4));
    float4 w2 = float4(*(const device half4*)(W + 256 + lane * 4));
    float4 b0 = float4(*(const device half4*)(B + lane * 4));
    float4 b1 = float4(*(const device half4*)(B + 128 + lane * 4));
    float4 b2 = float4(*(const device half4*)(B + 256 + lane * 4));

    device half* yp = Y + row * 384;
    *(device half4*)(yp + lane * 4)       = half4(f0 * inv_std * w0 + b0);
    *(device half4*)(yp + 128 + lane * 4) = half4(f1 * inv_std * w1 + b1);
    *(device half4*)(yp + 256 + lane * 4) = half4(f2 * inv_std * w2 + b2);
"""
_LN_KERNEL = mx.fast.metal_kernel(
    name="bge_fast_ln_384_f16",
    input_names=["X", "W", "B"],
    output_names=["Y"],
    header=_HEADER_STD,
    source=_LN_SRC,
)

# ---------------------------------------------------------------------------
# 3. Layer 11 Gather+LayerNorm+L2Normalize fused final output [N_docs, 384] f32
# ---------------------------------------------------------------------------
_FINAL_LN_L2_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x; // 0..31
    const device half* xp = X + row * 384;
    float4 f0 = float4(*(const device half4*)(xp + lane * 4));
    float4 f1 = float4(*(const device half4*)(xp + 128 + lane * 4));
    float4 f2 = float4(*(const device half4*)(xp + 256 + lane * 4));

    float psum = (f0.x + f0.y + f0.z + f0.w) + (f1.x + f1.y + f1.z + f1.w) + (f2.x + f2.y + f2.z + f2.w);
    float mean = simd_sum(psum) * (1.0f / 384.0f);
    f0 -= mean; f1 -= mean; f2 -= mean;
    float pvar = dot(f0, f0) + dot(f1, f1) + dot(f2, f2);
    float inv_std = rsqrt(simd_sum(pvar) * (1.0f / 384.0f) + 1e-12f);

    float4 w0 = float4(*(const device half4*)(W + lane * 4));
    float4 w1 = float4(*(const device half4*)(W + 128 + lane * 4));
    float4 w2 = float4(*(const device half4*)(W + 256 + lane * 4));
    float4 b0 = float4(*(const device half4*)(B + lane * 4));
    float4 b1 = float4(*(const device half4*)(B + 128 + lane * 4));
    float4 b2 = float4(*(const device half4*)(B + 256 + lane * 4));

    float4 y0 = f0 * inv_std * w0 + b0;
    float4 y1 = f1 * inv_std * w1 + b1;
    float4 y2 = f2 * inv_std * w2 + b2;
    float n2 = dot(y0, y0) + dot(y1, y1) + dot(y2, y2);
    float inv_n = rsqrt(simd_sum(n2) + 1e-24f);

    device float* yp = Y + row * 384;
    *(device float4*)(yp + lane * 4)       = y0 * inv_n;
    *(device float4*)(yp + 128 + lane * 4) = y1 * inv_n;
    *(device float4*)(yp + 256 + lane * 4) = y2 * inv_n;
"""
_FINAL_LN_L2_KERNEL = mx.fast.metal_kernel(
    name="bge_final_ln_l2_f32",
    input_names=["X", "W", "B"],
    output_names=["Y"],
    header=_HEADER_STD,
    source=_FINAL_LN_L2_SRC,
)

# ---------------------------------------------------------------------------
# 4. Varlen Bidirectional FlashAttention D=32 (32x32, aliased Q/P, S stride 36)
# ---------------------------------------------------------------------------
_FA32_SRC = """
    uint tid = thread_position_in_threadgroup.x; // 0..127
    uint sg = tid >> 5;                          // 0..3
    uint lane = tid & 31;                        // 0..31
    uint tile_idx = threadgroup_position_in_grid.y;
    uint h = threadgroup_position_in_grid.z;     // 0..11

    uint base_t = tile_idx * 4;
    uint q_start = (uint)tiles[base_t + 0];
    uint q_end   = (uint)tiles[base_t + 1];
    uint kv_start = (uint)tiles[base_t + 2];
    uint kv_end   = (uint)tiles[base_t + 3];
    uint q_cnt = q_end - q_start;

    threadgroup half QP_smem[32 * 36]; // Q loaded once before loop, then aliased for P!
    threadgroup half K_smem[32 * 36];
    threadgroup half V_smem[32 * 36];
    threadgroup float S_smem[32 * 36];

    uint h_off = h * 32;
    uint lr = tid >> 2;             // 0..31
    uint lc = (tid & 3) * 8;        // 0, 8, 16, 24
    if (lr < q_cnt) {
        const device half* qp = QKV + (q_start + lr) * 1152 + h_off + lc;
        *(threadgroup half4*)(QP_smem + lr * 36 + lc) = *(const device half4*)(qp);
        *(threadgroup half4*)(QP_smem + lr * 36 + lc + 4) = *(const device half4*)(qp + 4);
    } else {
        *(threadgroup half4*)(QP_smem + lr * 36 + lc) = half4(0.0h);
        *(threadgroup half4*)(QP_smem + lr * 36 + lc + 4) = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 q_mat[4];
    #pragma unroll
    for (uint db = 0; db < 4; ++db) {
        simdgroup_load(q_mat[db], QP_smem + (sg * 8) * 36 + db * 8, 36);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_reg[8];
    float l_reg[8];
    float o_reg[8];
    #pragma unroll
    for (uint rr = 0; rr < 8; ++rr) {
        m_reg[rr] = -1e30f;
        l_reg[rr] = 0.0f;
        o_reg[rr] = 0.0f;
    }

    const float scale_log2e = 0.17677669529663687f * 1.4426950408889634f;

    for (uint kb = kv_start; kb < kv_end; kb += 32) {
        uint k_row = kb + lr;
        if (k_row < kv_end) {
            const device half* kp = QKV + k_row * 1152 + 384 + h_off + lc;
            const device half* vp = QKV + k_row * 1152 + 768 + h_off + lc;
            *(threadgroup half4*)(K_smem + lr * 36 + lc) = *(const device half4*)(kp);
            *(threadgroup half4*)(K_smem + lr * 36 + lc + 4) = *(const device half4*)(kp + 4);
            *(threadgroup half4*)(V_smem + lr * 36 + lc) = *(const device half4*)(vp);
            *(threadgroup half4*)(V_smem + lr * 36 + lc + 4) = *(const device half4*)(vp + 4);
        } else {
            *(threadgroup half4*)(K_smem + lr * 36 + lc) = half4(0.0h);
            *(threadgroup half4*)(K_smem + lr * 36 + lc + 4) = half4(0.0h);
            *(threadgroup half4*)(V_smem + lr * 36 + lc) = half4(0.0h);
            *(threadgroup half4*)(V_smem + lr * 36 + lc + 4) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint nb = 0; nb < 4; ++nb) {
            simdgroup_float8x8 s_mat = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint db = 0; db < 4; ++db) {
                simdgroup_half8x8 k_mat;
                simdgroup_load(k_mat, K_smem + (nb * 8) * 36 + db * 8, 36, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_mat, q_mat[db], k_mat, s_mat);
            }
            simdgroup_store(s_mat, S_smem + (sg * 8) * 36 + nb * 8, 36);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float alpha_reg[8];
        bool k_valid = (kb + lane) < kv_end;
        #pragma unroll
        for (uint rr = 0; rr < 8; ++rr) {
            uint qr = sg * 8 + rr;
            float s = k_valid ? (S_smem[qr * 36 + lane] * scale_log2e) : -1e30f;
            float rmax = simd_max(s);
            float m_new = max(m_reg[rr], rmax);
            float alpha = exp2(m_reg[rr] - m_new);
            float p = k_valid ? exp2(s - m_new) : 0.0f;
            float rsum = simd_sum(p);
            m_reg[rr] = m_new;
            l_reg[rr] = l_reg[rr] * alpha + rsum;
            alpha_reg[rr] = alpha;
            QP_smem[qr * 36 + lane] = (half)p;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint db = 0; db < 4; ++db) {
            simdgroup_float8x8 o_mat = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint kbi = 0; kbi < 4; ++kbi) {
                simdgroup_half8x8 p_mat, v_mat;
                simdgroup_load(p_mat, QP_smem + (sg * 8) * 36 + kbi * 8, 36);
                simdgroup_load(v_mat, V_smem + (kbi * 8) * 36 + db * 8, 36);
                simdgroup_multiply_accumulate(o_mat, p_mat, v_mat, o_mat);
            }
            simdgroup_store(o_mat, S_smem + (sg * 8) * 36 + db * 8, 36);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint rr = 0; rr < 8; ++rr) {
            o_reg[rr] = o_reg[rr] * alpha_reg[rr] + S_smem[(sg * 8 + rr) * 36 + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint rr = 0; rr < 8; ++rr) {
        uint r_local = sg * 8 + rr;
        if (r_local < q_cnt) {
            float inv_l = 1.0f / l_reg[rr];
            Out[(q_start + r_local) * 384 + h_off + lane] = (half)(o_reg[rr] * inv_l);
        }
    }
"""
_FA32_KERNEL = mx.fast.metal_kernel(
    name="bge_varlen_fa32_d32_opt",
    input_names=["QKV", "tiles"],
    output_names=["Out"],
    header=_HEADER_SG,
    source=_FA32_SRC,
)

# ---------------------------------------------------------------------------
# 5. Layer 11 CLS-only Attention (Q_cls: [N_docs, 384], KV: [M_tok, 768])
# ---------------------------------------------------------------------------
_CLS_ATTN_SRC = """
    uint tid = thread_position_in_threadgroup.x; // 0..127
    uint sg = tid >> 5;                          // 0..3
    uint lane = tid & 31;                        // 0..31
    uint doc = threadgroup_position_in_grid.y;
    uint h = threadgroup_position_in_grid.z;     // 0..11

    uint ds = (uint)doc_bounds[doc * 2 + 0];
    uint de = (uint)doc_bounds[doc * 2 + 1];
    uint h_off = h * 32;

    const float scale_log2e = 0.17677669529663687f * 1.4426950408889634f;
    float q_val = (float)Q_cls[doc * 384 + h_off + lane] * scale_log2e;

    float m_loc = -1e30f;
    float l_loc = 0.0f;
    float o_loc = 0.0f;

    for (uint k = ds + sg; k < de; k += 4) {
        const device half* kvp = KV + k * 768 + h_off + lane;
        float k_val = (float)kvp[0];
        float v_val = (float)kvp[384];
        float score = simd_sum(q_val * k_val);
        if (l_loc == 0.0f) {
            m_loc = score;
            l_loc = 1.0f;
            o_loc = v_val;
        } else {
            float m_new = max(m_loc, score);
            float alpha = exp2(m_loc - m_new);
            float p = exp2(score - m_new);
            m_loc = m_new;
            l_loc = l_loc * alpha + p;
            o_loc = o_loc * alpha + p * v_val;
        }
    }

    threadgroup float sm_m[4];
    threadgroup float sm_l[4];
    threadgroup float sm_o[4 * 32];
    if (lane == 0) {
        sm_m[sg] = m_loc;
        sm_l[sg] = l_loc;
    }
    sm_o[sg * 32 + lane] = o_loc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (uint s = 0; s < 4; ++s) {
            if (sm_l[s] > 0.0f) m_g = max(m_g, sm_m[s]);
        }
        float l_g = 0.0f;
        float o_g = 0.0f;
        #pragma unroll
        for (uint s = 0; s < 4; ++s) {
            if (sm_l[s] > 0.0f) {
                float w = exp2(sm_m[s] - m_g);
                l_g += sm_l[s] * w;
                o_g += sm_o[s * 32 + lane] * w;
            }
        }
        Out[doc * 384 + h_off + lane] = (half)(o_g / l_g);
    }
"""
_CLS_ATTN_KERNEL = mx.fast.metal_kernel(
    name="bge_cls_attn_d32_f16",
    input_names=["Q_cls", "KV", "doc_bounds"],
    output_names=["Out"],
    header=_HEADER_STD,
    source=_CLS_ATTN_SRC,
)


_MPP_QKV = make_mpp_matmul_kernel("bge_mpp_bias_128x64_8", epilogue="bias", tm=128, tn=64, sgs=8)
_MPP_BIAS_RES = make_mpp_matmul_kernel("bge_mpp_bias_res_64x64_4", epilogue="bias_res", tm=64, tn=64, sgs=4)
_MPP_BIAS_GELU = make_mpp_matmul_kernel("bge_mpp_bias_gelu_64x64_4", epilogue="bias_gelu", tm=64, tn=64, sgs=4)


def pack_sequences(token_lists, max_tokens=8192):
    """Greedy pack variable-length token ID lists into >=128-aligned batches without padding tokens inside attention."""
    if not token_lists:
        return []
    batches = []
    cur_seqs = []
    cur_tok = 0
    for idx, ids in enumerate(token_lists):
        sl = len(ids)
        if sl == 0:
            ids = [101, 102]
            sl = 2
        elif sl > 512:
            ids = ids[:512]
            sl = 512
        if cur_seqs and cur_tok + sl > max_tokens:
            batches.append(cur_seqs)
            cur_seqs = [(idx, ids)]
            cur_tok = sl
        else:
            cur_seqs.append((idx, ids))
            cur_tok += sl
    if cur_seqs:
        batches.append(cur_seqs)

    packed = []
    for seqs in batches:
        M_tok = sum(len(x[1]) for x in seqs)
        M_pad = ((M_tok + 127) // 128) * 128
        tok_arr = np.zeros(M_pad, dtype=np.int32)
        pos_arr = np.zeros(M_pad, dtype=np.int32)
        orig_idx = np.empty(len(seqs), dtype=np.int32)
        doc_bounds = np.empty((len(seqs), 2), dtype=np.int32)
        cls_indices = np.empty(len(seqs), dtype=np.int32)
        tiles = []
        pos = 0
        for j, (oi, ids) in enumerate(seqs):
            sl = len(ids)
            orig_idx[j] = oi
            cls_indices[j] = pos
            doc_bounds[j, 0] = pos
            doc_bounds[j, 1] = pos + sl
            tok_arr[pos : pos + sl] = ids
            pos_arr[pos : pos + sl] = np.arange(sl, dtype=np.int32)
            for qs in range(pos, pos + sl, 32):
                qe = min(pos + sl, qs + 32)
                tiles.append((qs, qe, pos, pos + sl))
            pos += sl
        packed.append({
            "orig_idx": orig_idx,
            "M_tok": M_tok,
            "M_pad": M_pad,
            "tok_mx": mx.array(tok_arr, dtype=mx.int32),
            "pos_mx": mx.array(pos_arr, dtype=mx.int32),
            "tiles_mx": mx.array(np.asarray(tiles, dtype=np.int32), dtype=mx.int32),
            "bounds_mx": mx.array(doc_bounds, dtype=mx.int32),
            "cls_mx": mx.array(cls_indices, dtype=mx.int32),
        })
    return packed


class FastBgeSmallGPU:
    def __init__(self, weights_npz: str | Path | None = None, home: str | Path | None = None):
        mx.set_default_device(mx.gpu)
        if weights_npz is None:
            weights_npz = resolve_paths(home).weights_npz
        raw = dict(np.load(weights_npz))
        self.w_wt0 = mx.array(raw["embeddings.word_embeddings.weight"] + raw["embeddings.token_type_embeddings.weight"][0:1], dtype=mx.float16)
        self.w_pos = mx.array(raw["embeddings.position_embeddings.weight"], dtype=mx.float16)
        self.emb_ln_w = mx.array(raw["embeddings.LayerNorm.weight"], dtype=mx.float16)
        self.emb_ln_b = mx.array(raw["embeddings.LayerNorm.bias"], dtype=mx.float16)

        self.layers = []
        for l in range(12):
            p = f"encoder.layer.{l}."
            wq = raw[p + "attention.self.query.weight"]
            wk = raw[p + "attention.self.key.weight"]
            wv = raw[p + "attention.self.value.weight"]
            bq = raw[p + "attention.self.query.bias"]
            bk = raw[p + "attention.self.key.bias"]
            bv = raw[p + "attention.self.value.bias"]
            layer_d = {
                "w_o": mx.array(raw[p + "attention.output.dense.weight"], dtype=mx.float16),
                "b_o": mx.array(raw[p + "attention.output.dense.bias"], dtype=mx.float16),
                "ln1_w": mx.array(raw[p + "attention.output.LayerNorm.weight"], dtype=mx.float16),
                "ln1_b": mx.array(raw[p + "attention.output.LayerNorm.bias"], dtype=mx.float16),
                "w_up": mx.array(raw[p + "intermediate.dense.weight"], dtype=mx.float16),
                "b_up": mx.array(raw[p + "intermediate.dense.bias"], dtype=mx.float16),
                "w_dn": mx.array(raw[p + "output.dense.weight"], dtype=mx.float16),
                "b_dn": mx.array(raw[p + "output.dense.bias"], dtype=mx.float16),
                "ln2_w": mx.array(raw[p + "output.LayerNorm.weight"], dtype=mx.float16),
                "ln2_b": mx.array(raw[p + "output.LayerNorm.bias"], dtype=mx.float16),
            }
            if l < 11:
                layer_d["w_qkv"] = mx.array(np.concatenate([wq, wk, wv], axis=0), dtype=mx.float16)
                layer_d["b_qkv"] = mx.array(np.concatenate([bq, bk, bv], axis=0), dtype=mx.float16)
            else:
                layer_d["w_q"] = mx.array(wq, dtype=mx.float16)
                layer_d["b_q"] = mx.array(bq, dtype=mx.float16)
                layer_d["w_kv"] = mx.array(np.concatenate([wk, wv], axis=0), dtype=mx.float16)
                layer_d["b_kv"] = mx.array(np.concatenate([bk, bv], axis=0), dtype=mx.float16)
            self.layers.append(layer_d)

        # Single-query compiled FP16 path + transpose cache for fast small-batch addmm
        self._single_layers = []
        for l in range(12):
            ld = self.layers[l]
            d = {
                "w_oT": ld["w_o"].T, "b_o": ld["b_o"], "ln1_w": ld["ln1_w"], "ln1_b": ld["ln1_b"],
                "w_upT": ld["w_up"].T, "b_up": ld["b_up"], "w_dnT": ld["w_dn"].T, "b_dn": ld["b_dn"],
                "ln2_w": ld["ln2_w"], "ln2_b": ld["ln2_b"],
            }
            if l < 11:
                d["w_qkvT"] = ld["w_qkv"].T
                d["b_qkv"] = ld["b_qkv"]
            else:
                d["w_qT"] = ld["w_q"].T
                d["b_q"] = ld["b_q"]
                d["w_kvT"] = ld["w_kv"].T
                d["b_kv"] = ld["b_kv"]
            self._single_layers.append(d)

        @mx.compile
        def _single_fn(ids_1d):
            S = ids_1d.shape[0]
            h = self.w_wt0[ids_1d] + self.w_pos[:S]
            h = mx.fast.layer_norm(h, self.emb_ln_w, self.emb_ln_b, 1e-12)
            inv_s2 = mx.array(math.sqrt(0.5), dtype=mx.float16)
            scale = math.sqrt(1.0 / 32.0)
            for l in range(11):
                ld = self._single_layers[l]
                qkv = mx.addmm(ld["b_qkv"], h, ld["w_qkvT"])
                q = qkv[:, :384].reshape(1, S, 12, 32).transpose(0, 2, 1, 3)
                k = qkv[:, 384:768].reshape(1, S, 12, 32).transpose(0, 2, 1, 3)
                v = qkv[:, 768:].reshape(1, S, 12, 32).transpose(0, 2, 1, 3)
                ctx = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)[0].transpose(1, 0, 2).reshape(S, 384)
                h = mx.fast.layer_norm(mx.addmm(ld["b_o"], ctx, ld["w_oT"]) + h, ld["ln1_w"], ld["ln1_b"], 1e-12)
                up = mx.addmm(ld["b_up"], h, ld["w_upT"])
                act = up * (mx.array(0.5, dtype=mx.float16) * (mx.array(1.0, dtype=mx.float16) + mx.erf(up * inv_s2)))
                h = mx.fast.layer_norm(mx.addmm(ld["b_dn"], act, ld["w_dnT"]) + h, ld["ln2_w"], ld["ln2_b"], 1e-12)
            ld = self._single_layers[11]
            h_cls = h[:1]
            q11 = mx.addmm(ld["b_q"], h_cls, ld["w_qT"]).reshape(1, 1, 12, 32).transpose(0, 2, 1, 3)
            kv11 = mx.addmm(ld["b_kv"], h, ld["w_kvT"])
            k11 = kv11[:, :384].reshape(1, S, 12, 32).transpose(0, 2, 1, 3)
            v11 = kv11[:, 384:].reshape(1, S, 12, 32).transpose(0, 2, 1, 3)
            ctx11 = mx.fast.scaled_dot_product_attention(q11, k11, v11, scale=scale)[0].transpose(1, 0, 2).reshape(1, 384)
            hc = mx.fast.layer_norm(mx.addmm(ld["b_o"], ctx11, ld["w_oT"]) + h_cls, ld["ln1_w"], ld["ln1_b"], 1e-12)
            up11 = mx.addmm(ld["b_up"], hc, ld["w_upT"])
            act11 = up11 * (mx.array(0.5, dtype=mx.float16) * (mx.array(1.0, dtype=mx.float16) + mx.erf(up11 * inv_s2)))
            hc = mx.fast.layer_norm(mx.addmm(ld["b_dn"], act11, ld["w_dnT"]) + hc, ld["ln2_w"], ld["ln2_b"], 1e-12)
            out = hc.astype(mx.float32)
            return out / mx.linalg.norm(out, axis=-1, keepdims=True)

        self._single_fn = _single_fn
        mx.eval(self.w_wt0, self.w_pos, self.emb_ln_w, self.emb_ln_b)
        for ld in self.layers:
            mx.eval(*ld.values())
        for ld in self._single_layers:
            mx.eval(*ld.values())

    def encode_single(self, ids_1d):
        if len(ids_1d) == 0:
            ids_1d = [101, 102]
        elif len(ids_1d) > 512:
            ids_1d = ids_1d[:512]
        if not isinstance(ids_1d, mx.array):
            ids_1d = mx.array(ids_1d, dtype=mx.int32)
        out = self._single_fn(ids_1d)
        mx.eval(out)
        return np.array(out[0], dtype=np.float32)

    def forward_packed(self, tok_mx, pos_mx, tiles_mx, bounds_mx, cls_mx, M_tok, M_pad):
        nt = tiles_mx.shape[0]
        nd = bounds_mx.shape[0]

        h = _EMBED_LN_KERNEL(
            inputs=[tok_mx, pos_mx, self.w_wt0, self.w_pos, self.emb_ln_w, self.emb_ln_b],
            grid=(M_pad * 32, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(M_pad, 384)],
            output_dtypes=[mx.float16],
        )[0]

        gy128 = M_pad // 128
        gy64 = M_pad // 64
        gx_1152_64 = (1152 // 64) * 256  # 8 simdgroups = 256 threads
        gx_768_64 = (768 // 64) * 256
        gx_384_64 = (384 // 64) * 128    # 4 simdgroups = 128 threads
        gx_1536_64 = (1536 // 64) * 128

        for l in range(11):
            ld = self.layers[l]
            qkv = _MPP_QKV(
                inputs=[h, ld["w_qkv"], ld["b_qkv"]],
                grid=(gx_1152_64, gy128, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(M_pad, 1152)],
                output_dtypes=[mx.float16],
            )[0]
            attn_out = _FA32_KERNEL(
                inputs=[qkv, tiles_mx],
                grid=(128, nt, 12),
                threadgroup=(128, 1, 1),
                output_shapes=[(M_pad, 384)],
                output_dtypes=[mx.float16],
            )[0]
            h_res = _MPP_BIAS_RES(
                inputs=[attn_out, ld["w_o"], ld["b_o"], h],
                grid=(gx_384_64, gy64, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(M_pad, 384)],
                output_dtypes=[mx.float16],
            )[0]
            h = _LN_KERNEL(
                inputs=[h_res, ld["ln1_w"], ld["ln1_b"]],
                grid=(M_pad * 32, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(M_pad, 384)],
                output_dtypes=[mx.float16],
            )[0]
            act = _MPP_BIAS_GELU(
                inputs=[h, ld["w_up"], ld["b_up"]],
                grid=(gx_1536_64, gy64, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(M_pad, 1536)],
                output_dtypes=[mx.float16],
            )[0]
            h_res2 = _MPP_BIAS_RES(
                inputs=[act, ld["w_dn"], ld["b_dn"], h],
                grid=(gx_384_64, gy64, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(M_pad, 384)],
                output_dtypes=[mx.float16],
            )[0]
            h = _LN_KERNEL(
                inputs=[h_res2, ld["ln2_w"], ld["ln2_b"]],
                grid=(M_pad * 32, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(M_pad, 384)],
                output_dtypes=[mx.float16],
            )[0]

        # Layer 11: CLS-only optimization!
        ld = self.layers[11]
        kv11 = _MPP_QKV(
            inputs=[h, ld["w_kv"], ld["b_kv"]],
            grid=(gx_768_64, gy128, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(M_pad, 768)],
            output_dtypes=[mx.float16],
        )[0]
        h_cls = h[cls_mx]
        q11 = mx.addmm(ld["b_q"], h_cls, ld["w_q"].T)
        attn11 = _CLS_ATTN_KERNEL(
            inputs=[q11, kv11, bounds_mx],
            grid=(128, nd, 12),
            threadgroup=(128, 1, 1),
            output_shapes=[(nd, 384)],
            output_dtypes=[mx.float16],
        )[0]
        hc_mid = _LN_KERNEL(
            inputs=[mx.addmm(ld["b_o"], attn11, ld["w_o"].T) + h_cls, ld["ln1_w"], ld["ln1_b"]],
            grid=(((nd + 7) // 8) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(nd, 384)],
            output_dtypes=[mx.float16],
        )[0]
        up11 = mx.addmm(ld["b_up"], hc_mid, ld["w_up"].T)
        act11 = up11 * (mx.array(0.5, dtype=mx.float16) * (mx.array(1.0, dtype=mx.float16) + mx.erf(up11 * mx.array(math.sqrt(0.5), dtype=mx.float16))))
        hc_fin = mx.addmm(ld["b_dn"], act11, ld["w_dn"].T) + hc_mid
        out = _FINAL_LN_L2_KERNEL(
            inputs=[hc_fin, ld["ln2_w"], ld["ln2_b"]],
            grid=(nd * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(nd, 384)],
            output_dtypes=[mx.float32],
        )[0]
        return out

    def encode_corpus(self, token_lists, max_tokens=8192):
        if len(token_lists) == 0:
            return np.empty((0, 384), dtype=np.float32)
        packed = pack_sequences(token_lists, max_tokens=max_tokens)
        out = np.empty((len(token_lists), 384), dtype=np.float32)
        pending = None
        for b in packed:
            vecs = self.forward_packed(
                b["tok_mx"], b["pos_mx"], b["tiles_mx"], b["bounds_mx"], b["cls_mx"], b["M_tok"], b["M_pad"]
            )
            mx.async_eval(vecs)
            if pending is not None:
                prev_vecs, prev_idx = pending
                mx.eval(prev_vecs)
                out[prev_idx] = np.array(prev_vecs, dtype=np.float32)
            pending = (vecs, b["orig_idx"])
        if pending is not None:
            prev_vecs, prev_idx = pending
            mx.eval(prev_vecs)
            out[prev_idx] = np.array(prev_vecs, dtype=np.float32)
        return out
