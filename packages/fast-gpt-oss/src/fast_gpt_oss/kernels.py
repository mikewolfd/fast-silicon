"""Bare-Metal Metal 4 MPP & SIMD kernels for `openai/gpt-oss-20b` (`MXFP4` + Affine-Q4).

Composes `fast_shared.metal_mpp.HEADER_MPP` with exact E2M1 LUT unpacking and float32
E8M0 per-block scaling (avoiding half4 threadgroup overflow/underflow), Hydragen v2
split-prefix SDPA, 8Q:1KV packed SWA-128 prefill MMA, circular SWA-128 decode,
Batched YaRN RoPE, Zero-Sync Metal 4 Hybrid MoE (`MPP::matmul2d` hot tiles +
multi-column QMV cold tails), and fused `BF16` residual normalization.

All `mx.fast.metal_kernel(...)` compilations happen lazily inside `build_kernels()`
so importing this module on CPU or under unit tests executes zero GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_shared.metal_mpp import HEADER_MPP

_E2M1_HELPERS = r"""
constant half kE2M1_LUT[16] = {
  0.0h,  0.5h,  1.0h,  1.5h,  2.0h,  3.0h,  4.0h,  6.0h,
 -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

inline half4 unpack4_e2m1(uint packed16) {
  return half4(
    kE2M1_LUT[(packed16      ) & 0xFu],
    kE2M1_LUT[(packed16 >>  4) & 0xFu],
    kE2M1_LUT[(packed16 >>  8) & 0xFu],
    kE2M1_LUT[(packed16 >> 12) & 0xFu]
  );
}

inline float e8m0_to_float(uchar s) {
  // Exact IEEE-754 float32 for OCP MXFP4 E8M0 scale 2^(int(s) - 127) across s in [0..254].
  return (s == 0u) ? as_type<float>(0x00400000u) : as_type<float>(uint(s) << 23);
}
"""


def build_metal_header() -> str:
    """Build the shared Metal 4 + simdgroup_matrix header composing `fast_shared.metal_mpp.HEADER_MPP`."""
    return (
        "#include <metal_stdlib>\n"
        "#include <metal_simdgroup_matrix>\n"
        + HEADER_MPP
        + _E2M1_HELPERS
    )


@dataclass(frozen=True)
class KernelSpec:
    """Geometry constants baked into compiled `gpt-oss` Metal kernels."""

    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    num_local_experts: int = 32
    sliding_window: int = 128
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    moe_hot_threshold: int = 8

    def validate(self) -> None:
        if self.hidden_size % 64 != 0 or self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive multiple of 64, got {self.hidden_size}")
        if self.intermediate_size % 32 != 0 or self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be a positive multiple of 32, got {self.intermediate_size}"
            )
        if self.head_dim != 64:
            raise ValueError(f"Batched Metal MMA attention kernels require head_dim=64, got {self.head_dim}")
        if self.num_key_value_heads <= 0 or self.num_attention_heads != self.num_key_value_heads * 8:
            raise ValueError(
                f"Batched MMA attention kernels require GQA ratio 8 (n_q=8*n_kv), got n_q={self.num_attention_heads}, n_kv={self.num_key_value_heads}"
            )
        if not (1 <= self.num_local_experts <= 32):
            raise ValueError(f"num_local_experts must be in 1..32, got {self.num_local_experts}")
        if self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
        if not (1 <= self.moe_hot_threshold <= 16):
            raise ValueError(f"moe_hot_threshold must be in 1..16, got {self.moe_hot_threshold}")


def render_kernel_sources(spec: KernelSpec) -> dict[str, dict[str, Any]]:
    """Render Metal kernel names, I/O signatures, header, and MSL sources for `spec`."""
    spec.validate()
    H = spec.hidden_size
    D_FF = spec.intermediate_size
    N_Q = spec.num_attention_heads
    N_KV = spec.num_key_value_heads
    HD = spec.head_dim
    Q_DIM = N_Q * HD
    KV_DIM = N_KV * HD
    QKV_DIM = Q_DIM + 2 * KV_DIM
    NG_H = H // 32
    NG_D = D_FF // 32
    D_FF_2 = D_FF * 2
    N_EXP = spec.num_local_experts
    HOT_THRESH = spec.moe_hot_threshold
    TG_HALF_ELEMS = max(3840, HOT_THRESH * 480)
    LIMIT = f"{float(spec.swiglu_limit):.6f}f"
    ALPHA = f"{float(spec.swiglu_alpha):.6f}f"
    HOT_X_GU = D_FF // 32
    HOT_X_DN = H // 64

    hdr = build_metal_header()

    return {
        "rmsnorm_bf16_to_f16": {
            "input_names": ["X", "W", "Params"],
            "output_names": ["Out"],
            "header": hdr,
            "source": f"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint sg  = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat16_t* xp = X + (size_t)row * {H}u;
float ss = 0.0f;
{{
#pragma clang fp reassociate(off)
for (uint i = tid; i < {H}u; i += 128u) {{
  float v = float(xp[i]);
  ss += v * v;
}}
}}
float ws = simd_sum(ss);
threadgroup float tg_s[4];
if (lane == 0u) tg_s[sg] = ws;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = tg_s[0] + tg_s[1] + tg_s[2] + tg_s[3];
float inv = rsqrt(tot * (1.0f / {float(H):.1f}f) + Params[0]);

device half* op = Out + (size_t)row * {H}u;
for (uint i = tid; i < {H}u; i += 128u) {{
  op[i] = half(float(xp[i]) * inv * float(W[i]));
}}
""",
        },
        "rmsnorm_bf16_to_f16_f32": {
            "input_names": ["X", "W", "Params"],
            "output_names": ["OutF16", "OutF32"],
            "header": hdr,
            "source": f"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint sg  = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

const device bfloat16_t* xp = X + (size_t)row * {H}u;
float ss = 0.0f;
{{
#pragma clang fp reassociate(off)
for (uint i = tid; i < {H}u; i += 128u) {{
  float v = float(xp[i]);
  ss += v * v;
}}
}}
float ws = simd_sum(ss);
threadgroup float tg_s[4];
if (lane == 0u) tg_s[sg] = ws;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = tg_s[0] + tg_s[1] + tg_s[2] + tg_s[3];
float inv = rsqrt(tot * (1.0f / {float(H):.1f}f) + Params[0]);

device half* op16 = OutF16 + (size_t)row * {H}u;
device float* op32 = OutF32 + (size_t)row * {H}u;
for (uint i = tid; i < {H}u; i += 128u) {{
  float val = float(xp[i]) * inv * float(W[i]);
  op16[i] = half(val);
  op32[i] = val;
}}
""",
        },
        "yarn_rope_split_b": {
            "input_names": ["QKV", "Pos", "Freqs", "MScale"],
            "output_names": ["Q", "K", "V"],
            "header": hdr,
            "source": f"""
const uint h = threadgroup_position_in_grid.x;
const uint b = threadgroup_position_in_grid.y;
const uint d = thread_index_in_simdgroup;

const float mscale = MScale[0];
const float theta = float(Pos[b]) / Freqs[d];
float s, c;
s = sincos(theta, c);

const device half* base_qkv = QKV + (size_t)b * {QKV_DIM}u;
if (h < {N_Q}u) {{
  const device half* qp = base_qkv + h * 64u;
  float x0 = float(qp[d]) * mscale;
  float x1 = float(qp[d + 32u]) * mscale;
  device half* qout = Q + ((size_t)b * {N_Q}u + h) * 64u;
  qout[d]       = half(x0 * c - x1 * s);
  qout[d + 32u] = half(x0 * s + x1 * c);
}} else {{
  uint hk = h - {N_Q}u;
  const device half* kp = base_qkv + {Q_DIM}u + hk * 64u;
  float x0 = float(kp[d]) * mscale;
  float x1 = float(kp[d + 32u]) * mscale;
  device half* kout = K + ((size_t)b * {N_KV}u + hk) * 64u;
  kout[d]       = half(x0 * c - x1 * s);
  kout[d + 32u] = half(x0 * s + x1 * c);

  const device half* vp = base_qkv + {Q_DIM + KV_DIM}u + hk * 64u;
  device half* vout = V + ((size_t)b * {N_KV}u + hk) * 64u;
  vout[d]       = vp[d];
  vout[d + 32u] = vp[d + 32u];
}}
""",
        },
        "hydragen_p1_mma": {
            "input_names": ["Q", "K_sys", "V_sys", "Params"],
            "output_names": ["O_sys", "LSE_sys"],
            "header": hdr,
            "source": f"""
const uint B     = Params[0];
const uint L_sys = Params[1];
const float scale = as_type<float>(Params[2]);
const uint hkv   = threadgroup_position_in_grid.y;
const uint b     = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
const uint tid_tg = thread_index_in_threadgroup;
const uint sg_id = simdgroup_index_in_threadgroup;
const uint lane  = thread_index_in_simdgroup;
const bool b_valid = (b < B);

constexpr uint LD_KV = 72u;
constexpr uint LD_S  = 40u;
threadgroup half  Ks[32u * LD_KV];
threadgroup half  Vs[32u * LD_KV];
threadgroup float Sw[4u][8u * LD_S];
threadgroup half  Pw[4u][8u * LD_S];

threadgroup half* Qs_all = (threadgroup half*)Sw;
threadgroup half* Qw = Qs_all + sg_id * (8u * LD_KV);
if (b_valid) {{
  const device half* q_ptr = Q + ((size_t)b * {N_Q}u + hkv * 8u) * 64u;
  for (uint i = lane; i < 64u; i += 32u) {{
    uint r = i >> 3;
    uint c = (i & 7u) << 3;
    *(threadgroup uint4*)(Qw + r * LD_KV + c) = *(const device uint4*)(q_ptr + r * 64u + c);
  }}
}}
simdgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_half8x8 qMat[8];
#pragma unroll
for (int k = 0; k < 8; ++k) {{
  simdgroup_load(qMat[k], Qw + k * 8u, LD_KV, ulong2(0, 0), false);
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 oMat[8];
#pragma unroll
for (int k = 0; k < 8; ++k) oMat[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

float row_m[8], row_l[8];
#pragma unroll
for (int r = 0; r < 8; ++r) {{ row_m[r] = -1e30f; row_l[r] = 0.0f; }}

const size_t kv_base = (size_t)hkv * L_sys * 64u;

for (uint t0 = 0; t0 < L_sys; t0 += 32u) {{
  uint valid_t = min(32u, L_sys - t0);
  for (uint idx = tid_tg; idx < 256u; idx += 128u) {{
    uint r = idx >> 3;
    uint c = (idx & 7u) << 3;
    if (r < valid_t) {{
      size_t off = kv_base + (size_t)(t0 + r) * 64u + c;
      *(threadgroup uint4*)(Ks + r * LD_KV + c) = *(const device uint4*)(K_sys + off);
      *(threadgroup uint4*)(Vs + r * LD_KV + c) = *(const device uint4*)(V_sys + off);
    }} else {{
      *(threadgroup uint4*)(Ks + r * LD_KV + c) = uint4(0u);
      *(threadgroup uint4*)(Vs + r * LD_KV + c) = uint4(0u);
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* sw = Sw[sg_id];
  threadgroup half*  pw = Pw[sg_id];
  #pragma unroll
  for (int jb = 0; jb < 4; ++jb) {{
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {{
      simdgroup_half8x8 kMat;
      simdgroup_load(kMat, Ks + (jb * 8u) * LD_KV + k * 8u, LD_KV, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc, qMat[k], kMat, acc);
    }}
    simdgroup_store(acc, sw + jb * 8u, LD_S, ulong2(0, 0), false);
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);

  bool col_valid = (lane < valid_t);
  #pragma unroll
  for (int r = 0; r < 8; ++r) {{
    float v = col_valid ? (sw[r * LD_S + lane] * scale) : -1e30f;
    float blk_m = simd_max(v);
    float new_m = max(row_m[r], blk_m);
    float alpha = (row_l[r] > 0.0f) ? exp(row_m[r] - new_m) : 1.0f;
    float p = col_valid ? exp(v - new_m) : 0.0f;
    float blk_l = simd_sum(p);
    pw[r * LD_S + lane] = half(p);
    row_l[r] = row_l[r] * alpha + blk_l;
    sw[r * LD_S + 32u] = alpha;
    row_m[r] = new_m;
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);

  {{
    float a_row[8];
    if (lane < 8u) {{
      #pragma unroll
      for (int r = 0; r < 8; ++r) a_row[r] = sw[r * LD_S + 32u];
    }}
    float my_alpha = (lane < 8u) ? a_row[lane] : 1.0f;
    float alphas[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) alphas[r] = simd_shuffle(my_alpha, r);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {{
      #pragma unroll
      for (int kb = 0; kb < 4; ++kb) {{
        simdgroup_store(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
      }}
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma unroll
      for (int r = 0; r < 8; ++r) sw[r * LD_S + lane] *= alphas[r];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma unroll
      for (int kb = 0; kb < 4; ++kb) {{
        simdgroup_load(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
      }}
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
  }}

  #pragma unroll
  for (int jb = 0; jb < 4; ++jb) {{
    simdgroup_half8x8 pMat;
    simdgroup_load(pMat, pw + jb * 8u, LD_S, ulong2(0, 0), false);
    #pragma unroll
    for (int kc = 0; kc < 8; ++kc) {{
      simdgroup_half8x8 vMat;
      simdgroup_load(vMat, Vs + (jb * 8u) * LD_KV + kc * 8u, LD_KV, ulong2(0, 0), false);
      simdgroup_multiply_accumulate(oMat[kc], pMat, vMat, oMat[kc]);
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

threadgroup float* sw = Sw[sg_id];
if (b_valid) {{
  device float* lse_ptr = LSE_sys + (size_t)b * {N_Q}u + hkv * 8u;
  if (lane < 8u) lse_ptr[lane] = row_m[lane] + log(max(row_l[lane], 1e-30f));
}}
#pragma unroll
for (int half_idx = 0; half_idx < 2; ++half_idx) {{
  #pragma unroll
  for (int kb = 0; kb < 4; ++kb) {{
    simdgroup_store(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (b_valid) {{
    device half* o_ptr = O_sys + ((size_t)b * {N_Q}u + hkv * 8u) * 64u;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {{
      float inv_l = (row_l[r] > 0.0f) ? (1.0f / row_l[r]) : 0.0f;
      o_ptr[r * 64u + half_idx * 32u + lane] = half(sw[r * LD_S + lane] * inv_l);
    }}
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);
}}
""",
        },
        "hydragen_p2_varlen": {
            "input_names": ["Q", "K_suf", "V_suf", "O_sys", "LSE_sys", "Sinks", "SufLens", "Params"],
            # Params: [Cap, SufMax, DecCnt, float_as_uint(scale), HasSys]
            "output_names": ["Out"],
            "header": hdr,
            "source": f"""
const uint b      = threadgroup_position_in_grid.x;
const uint hkv    = threadgroup_position_in_grid.y;
const uint sg_id  = simdgroup_index_in_threadgroup;
const uint lane   = thread_index_in_simdgroup;
const uint Cap    = Params[0];
const uint SufMax = Params[1];
const uint DecCnt = Params[2];
const float scale = as_type<float>(Params[3]);
const uint HasSys = Params[4];
const uint suf_len = SufLens[b];
const uint qh_base = hkv * 8u;

const device half* q_ptr = Q + ((size_t)b * {N_Q}u + qh_base) * 64u + lane * 2u;
float2 q_reg[8];
#pragma unroll
for (int h = 0; h < 8; ++h) {{
  q_reg[h] = float2(*(const device half2*)(q_ptr + h * 64u)) * scale;
}}

float  m_sg[8], l_sg[8];
float2 o_sg[8];
#pragma unroll
for (int h = 0; h < 8; ++h) {{ m_sg[h] = -1e30f; l_sg[h] = 0.0f; o_sg[h] = float2(0.0f); }}

const size_t kv_base = ((size_t)b * {N_KV}u + hkv) * Cap * 64u + lane * 2u;
const device half* Kp = K_suf + kv_base;
const device half* Vp = V_suf + kv_base;

for (uint t = sg_id; t < suf_len; t += 4u) {{
  float2 kf = float2(*(const device half2*)(Kp + (size_t)t * 64u));
  float2 vf = float2(*(const device half2*)(Vp + (size_t)t * 64u));
  float sc[8];
  #pragma unroll
  for (int h = 0; h < 8; ++h) sc[h] = simd_sum(dot(q_reg[h], kf));
  #pragma unroll
  for (int h = 0; h < 8; ++h) {{
    float s = sc[h];
    float nm = max(m_sg[h], s);
    float al = (l_sg[h] > 0.0f) ? exp(m_sg[h] - nm) : 1.0f;
    float w = exp(s - nm);
    m_sg[h] = nm;
    l_sg[h] = l_sg[h] * al + w;
    o_sg[h] = o_sg[h] * al + w * vf;
  }}
}}
for (uint dt = sg_id; dt < DecCnt; dt += 4u) {{
  uint t = SufMax + dt;
  float2 kf = float2(*(const device half2*)(Kp + (size_t)t * 64u));
  float2 vf = float2(*(const device half2*)(Vp + (size_t)t * 64u));
  float sc[8];
  #pragma unroll
  for (int h = 0; h < 8; ++h) sc[h] = simd_sum(dot(q_reg[h], kf));
  #pragma unroll
  for (int h = 0; h < 8; ++h) {{
    float s = sc[h];
    float nm = max(m_sg[h], s);
    float al = (l_sg[h] > 0.0f) ? exp(m_sg[h] - nm) : 1.0f;
    float w = exp(s - nm);
    m_sg[h] = nm;
    l_sg[h] = l_sg[h] * al + w;
    o_sg[h] = o_sg[h] * al + w * vf;
  }}
}}

threadgroup float  M_tg[4u][8u];
threadgroup float  L_tg[4u][8u];
threadgroup float2 O_tg[4u][8u * 32u];
if (lane < 8u) {{ M_tg[sg_id][lane] = m_sg[lane]; L_tg[sg_id][lane] = l_sg[lane]; }}
#pragma unroll
for (int h = 0; h < 8; ++h) O_tg[sg_id][h * 32u + lane] = o_sg[h];
threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
for (int dh = 0; dh < 2; ++dh) {{
  uint h = sg_id * 2u + dh;
  float lse_sys = (HasSys != 0u) ? LSE_sys[(size_t)b * {N_Q}u + qh_base + h] : -1e30f;
  float sink_v  = Sinks[qh_base + h];

  float m0 = M_tg[0][h], l0 = L_tg[0][h];
  float m1 = M_tg[1][h], l1 = L_tg[1][h];
  float m2 = M_tg[2][h], l2 = L_tg[2][h];
  float m3 = M_tg[3][h], l3 = L_tg[3][h];

  float m_max = max(max(max(m0, m1), max(m2, m3)), max(lse_sys, sink_v));
  float w0 = (l0 > 0.0f) ? exp(m0 - m_max) : 0.0f;
  float w1 = (l1 > 0.0f) ? exp(m1 - m_max) : 0.0f;
  float w2 = (l2 > 0.0f) ? exp(m2 - m_max) : 0.0f;
  float w3 = (l3 > 0.0f) ? exp(m3 - m_max) : 0.0f;
  float ws = (HasSys != 0u) ? exp(lse_sys - m_max) : 0.0f;
  float wk = exp(sink_v  - m_max);

  float inv_l = 1.0f / (l0 * w0 + l1 * w1 + l2 * w2 + l3 * w3 + ws + wk);
  float2 acc = O_tg[0][h * 32u + lane] * w0 + O_tg[1][h * 32u + lane] * w1 +
               O_tg[2][h * 32u + lane] * w2 + O_tg[3][h * 32u + lane] * w3;
  if (HasSys != 0u) {{
    half2 osys_h = *(const device half2*)(O_sys + ((size_t)b * {N_Q}u + qh_base + h) * 64u + lane * 2u);
    acc += float2(osys_h) * ws;
  }}
  *(device half2*)(Out + ((size_t)b * {N_Q}u + qh_base + h) * 64u + lane * 2u) = half2(acc * inv_l);
}}
""",
        },
        "swa128_decode": {
            "input_names": ["Q", "K_swa", "V_swa", "Sinks", "SwaInitLens", "Params"],
            # Params: [Cap(swa), EndMod((swa_write+1)%Cap), DecCnt, float_as_uint(scale)]
            "output_names": ["Out"],
            "header": hdr,
            "source": f"""
const uint b      = threadgroup_position_in_grid.x;
const uint hkv    = threadgroup_position_in_grid.y;
const uint sg_id  = simdgroup_index_in_threadgroup;
const uint lane   = thread_index_in_simdgroup;
const uint Cap    = Params[0];
const uint EndMod = Params[1];
const uint DecCnt = Params[2];
const float scale = as_type<float>(Params[3]);
const uint VLen   = min(Cap, SwaInitLens[b] + DecCnt + 1u);
const uint qh_base = hkv * 8u;

const device half* q_ptr = Q + ((size_t)b * {N_Q}u + qh_base) * 64u + lane * 2u;
float2 q_reg[8];
#pragma unroll
for (int h = 0; h < 8; ++h) q_reg[h] = float2(*(const device half2*)(q_ptr + h * 64u)) * scale;

float m_sg[8], l_sg[8];
float2 o_sg[8];
#pragma unroll
for (int h = 0; h < 8; ++h) {{ m_sg[h] = -1e30f; l_sg[h] = 0.0f; o_sg[h] = float2(0.0f); }}

const size_t kv_base = ((size_t)b * {N_KV}u + hkv) * Cap * 64u + lane * 2u;
const device half* Kp = K_swa + kv_base;
const device half* Vp = V_swa + kv_base;

if (VLen == Cap) {{
  for (uint t = sg_id; t < Cap; t += 4u) {{
    float2 kf = float2(*(const device half2*)(Kp + (size_t)t * 64u));
    float2 vf = float2(*(const device half2*)(Vp + (size_t)t * 64u));
    float sc[8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) sc[h] = simd_sum(dot(q_reg[h], kf));
    #pragma unroll
    for (int h = 0; h < 8; ++h) {{
      float s = sc[h];
      float nm = max(m_sg[h], s);
      float al = (l_sg[h] > 0.0f) ? exp(m_sg[h] - nm) : 1.0f;
      float w = exp(s - nm);
      m_sg[h] = nm;
      l_sg[h] = l_sg[h] * al + w;
      o_sg[h] = o_sg[h] * al + w * vf;
    }}
  }}
}} else {{
  uint start = (EndMod >= VLen) ? (EndMod - VLen) : (Cap + EndMod - VLen);
  for (uint t = sg_id; t < VLen; t += 4u) {{
    uint slot = start + t;
    if (slot >= Cap) slot -= Cap;
    float2 kf = float2(*(const device half2*)(Kp + (size_t)slot * 64u));
    float2 vf = float2(*(const device half2*)(Vp + (size_t)slot * 64u));
    float sc[8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) sc[h] = simd_sum(dot(q_reg[h], kf));
    #pragma unroll
    for (int h = 0; h < 8; ++h) {{
      float s = sc[h];
      float nm = max(m_sg[h], s);
      float al = (l_sg[h] > 0.0f) ? exp(m_sg[h] - nm) : 1.0f;
      float w = exp(s - nm);
      m_sg[h] = nm;
      l_sg[h] = l_sg[h] * al + w;
      o_sg[h] = o_sg[h] * al + w * vf;
    }}
  }}
}}

threadgroup float  M_tg[4u][8u];
threadgroup float  L_tg[4u][8u];
threadgroup float2 O_tg[4u][8u * 32u];
if (lane < 8u) {{ M_tg[sg_id][lane] = m_sg[lane]; L_tg[sg_id][lane] = l_sg[lane]; }}
#pragma unroll
for (int h = 0; h < 8; ++h) O_tg[sg_id][h * 32u + lane] = o_sg[h];
threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
for (int dh = 0; dh < 2; ++dh) {{
  uint h = sg_id * 2u + dh;
  float sink_v = Sinks[qh_base + h];
  float m0 = M_tg[0][h], l0 = L_tg[0][h];
  float m1 = M_tg[1][h], l1 = L_tg[1][h];
  float m2 = M_tg[2][h], l2 = L_tg[2][h];
  float m3 = M_tg[3][h], l3 = L_tg[3][h];

  float m_max = max(max(max(m0, m1), max(m2, m3)), sink_v);
  float w0 = (l0 > 0.0f) ? exp(m0 - m_max) : 0.0f;
  float w1 = (l1 > 0.0f) ? exp(m1 - m_max) : 0.0f;
  float w2 = (l2 > 0.0f) ? exp(m2 - m_max) : 0.0f;
  float w3 = (l3 > 0.0f) ? exp(m3 - m_max) : 0.0f;
  float wk = exp(sink_v - m_max);
  float inv_l = 1.0f / (l0 * w0 + l1 * w1 + l2 * w2 + l3 * w3 + wk);

  float2 acc = O_tg[0][h * 32u + lane] * w0 + O_tg[1][h * 32u + lane] * w1 +
               O_tg[2][h * 32u + lane] * w2 + O_tg[3][h * 32u + lane] * w3;
  *(device half2*)(Out + ((size_t)b * {N_Q}u + qh_base + h) * 64u + lane * 2u) = half2(acc * inv_l);
}}
""",
        },
        "swa128_prefill_mma": {
            "input_names": ["Q", "K", "V", "Sinks", "Params"],
            "output_names": ["Out"],
            "header": hdr,
            "source": f"""
// 8Q:1KV GQA-packed SWA prefill MMA: 8 simdgroups (256 threads) share one Ks/Vs tile per hkv
// across all 8 query heads (`hq = hkv * 8 + sg_id`) for `8` query tokens (`q_blk .. q_blk + 7`).
const uint L_q    = Params[0];
const uint L_kv   = Params[1];
const uint Offset = Params[2];
const uint Window = Params[3];
const float Scale = as_type<float>(Params[4]);

const uint q_blk  = threadgroup_position_in_grid.x * 8u;
const uint hkv    = threadgroup_position_in_grid.y;
const uint b      = threadgroup_position_in_grid.z;
const uint tid_tg = thread_index_in_threadgroup;
const uint sg_id  = simdgroup_index_in_threadgroup;
const uint lane   = thread_index_in_simdgroup;
const uint hq     = hkv * 8u + sg_id;

constexpr uint LD_KV = 72u;
constexpr uint LD_S  = 40u;
threadgroup half  Ks[32u * LD_KV];
threadgroup half  Vs[32u * LD_KV];
threadgroup float Sw[8u][8u * LD_S];
threadgroup half  Pw[8u][8u * LD_S];

threadgroup half* Qs_all = (threadgroup half*)Sw;
threadgroup half* Qw = Qs_all + sg_id * (8u * LD_KV);

const device half* q_ptr = Q + (((size_t)b * {N_Q}u + hq) * L_q) * 64u;
for (uint i = lane; i < 64u; i += 32u) {{
  uint r = i >> 3;
  uint c = (i & 7u) << 3;
  uint qi = q_blk + r;
  if (qi < L_q) *(threadgroup uint4*)(Qw + r * LD_KV + c) = *(const device uint4*)(q_ptr + (size_t)qi * 64u + c);
  else *(threadgroup uint4*)(Qw + r * LD_KV + c) = uint4(0u);
}}
simdgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_half8x8 qMat[8];
#pragma unroll
for (int k = 0; k < 8; ++k) simdgroup_load(qMat[k], Qw + k * 8u, LD_KV, ulong2(0, 0), false);
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 oMat[8];
#pragma unroll
for (int k = 0; k < 8; ++k) oMat[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

float row_m[8], row_l[8];
#pragma unroll
for (int r = 0; r < 8; ++r) {{ row_m[r] = -1e30f; row_l[r] = 0.0f; }}

int min_q_abs = (int)(Offset + q_blk);
int max_q_abs = (int)(Offset + min(L_q, q_blk + 8u) - 1u);
int t_min = max(0, min_q_abs - (int)(Window - 1u));
int t_max = min((int)L_kv - 1, max_q_abs);
uint t_start = (uint)(t_min & ~31);
uint t_end   = (uint)(t_max + 1);

const size_t kv_base = ((size_t)b * {N_KV}u + hkv) * L_kv * 64u;
const uint kv_r = tid_tg >> 3;
const uint kv_c = (tid_tg & 7u) << 3;

for (uint t0 = t_start; t0 < t_end; t0 += 32u) {{
  uint valid_t = min(32u, L_kv - t0);
  if (kv_r < valid_t) {{
    size_t off = kv_base + (size_t)(t0 + kv_r) * 64u + kv_c;
    *(threadgroup uint4*)(Ks + kv_r * LD_KV + kv_c) = *(const device uint4*)(K + off);
    *(threadgroup uint4*)(Vs + kv_r * LD_KV + kv_c) = *(const device uint4*)(V + off);
  }} else {{
    *(threadgroup uint4*)(Ks + kv_r * LD_KV + kv_c) = uint4(0u);
    *(threadgroup uint4*)(Vs + kv_r * LD_KV + kv_c) = uint4(0u);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* sw = Sw[sg_id];
  threadgroup half*  pw = Pw[sg_id];
  #pragma unroll
  for (int jb = 0; jb < 4; ++jb) {{
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {{
      simdgroup_half8x8 kMat;
      simdgroup_load(kMat, Ks + (jb * 8u) * LD_KV + k * 8u, LD_KV, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc, qMat[k], kMat, acc);
    }}
    simdgroup_store(acc, sw + jb * 8u, LD_S, ulong2(0, 0), false);
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);

  uint key_pos = t0 + lane;
  #pragma unroll
  for (int r = 0; r < 8; ++r) {{
    uint qi = q_blk + (uint)r;
    int q_abs = (int)(Offset + qi);
    int delta = q_abs - (int)key_pos;
    bool valid = (qi < L_q) && (lane < valid_t) && (delta >= 0) && (delta < (int)Window);
    float v = valid ? (sw[r * LD_S + lane] * Scale) : -1e30f;
    float blk_m = simd_max(v);
    float new_m = max(row_m[r], blk_m);
    float alpha = (row_l[r] > 0.0f) ? exp(row_m[r] - new_m) : 1.0f;
    float p = valid ? exp(v - new_m) : 0.0f;
    float blk_l = simd_sum(p);
    pw[r * LD_S + lane] = half(p);
    row_l[r] = row_l[r] * alpha + blk_l;
    sw[r * LD_S + 32u] = alpha;
    row_m[r] = new_m;
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);

  {{
    float a_row[8];
    if (lane < 8u) {{
      #pragma unroll
      for (int r = 0; r < 8; ++r) a_row[r] = sw[r * LD_S + 32u];
    }}
    float my_alpha = (lane < 8u) ? a_row[lane] : 1.0f;
    float alphas[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) alphas[r] = simd_shuffle(my_alpha, r);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {{
      #pragma unroll
      for (int kb = 0; kb < 4; ++kb) simdgroup_store(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma unroll
      for (int r = 0; r < 8; ++r) sw[r * LD_S + lane] *= alphas[r];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      #pragma unroll
      for (int kb = 0; kb < 4; ++kb) simdgroup_load(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
  }}

  #pragma unroll
  for (int jb = 0; jb < 4; ++jb) {{
    simdgroup_half8x8 pMat;
    simdgroup_load(pMat, pw + jb * 8u, LD_S, ulong2(0, 0), false);
    #pragma unroll
    for (int kc = 0; kc < 8; ++kc) {{
      simdgroup_half8x8 vMat;
      simdgroup_load(vMat, Vs + (jb * 8u) * LD_KV + kc * 8u, LD_KV, ulong2(0, 0), false);
      simdgroup_multiply_accumulate(oMat[kc], pMat, vMat, oMat[kc]);
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

const float sink_v = Sinks[hq];
float inv_l[8], sink_scale[8];
#pragma unroll
for (int r = 0; r < 8; ++r) {{
  float nm = max(row_m[r], sink_v);
  float al = (row_l[r] > 0.0f) ? exp(row_m[r] - nm) : 0.0f;
  float wk = exp(sink_v - nm);
  sink_scale[r] = al;
  inv_l[r] = 1.0f / (row_l[r] * al + wk);
}}

threadgroup float* sw = Sw[sg_id];
device half* out_ptr = Out + (((size_t)b * {N_Q}u + hq) * L_q) * 64u;
#pragma unroll
for (int half_idx = 0; half_idx < 2; ++half_idx) {{
  #pragma unroll
  for (int kb = 0; kb < 4; ++kb) simdgroup_store(oMat[half_idx * 4 + kb], sw + kb * 8u, LD_S, ulong2(0, 0), false);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (int r = 0; r < 8; ++r) {{
    uint qi = q_blk + (uint)r;
    if (qi < L_q) out_ptr[(size_t)qi * 64u + half_idx * 32u + lane] = half(sw[r * LD_S + lane] * sink_scale[r] * inv_l[r]);
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);
}}
""",
        },
        "moe_build_tiles": {
            "input_names": ["Inds", "Scores", "NTokArr"],
            "output_names": ["TokIdx", "SlotIdx", "RWeights", "HotMeta", "ColdMeta", "MetaCounts"],
            "header": hdr,
            "source": f"""
uint lane = thread_index_in_simdgroup;
uint M = NTokArr[0];
uint N = M * 4u;

threadgroup uint cnts[32];
threadgroup uint offs[32];
cnts[lane] = 0u;
simdgroup_barrier(mem_flags::mem_threadgroup);

threadgroup atomic_uint* acnts = (threadgroup atomic_uint*)cnts;
for (uint i = lane; i < N; i += 32u) {{
  uint e = Inds[i];
  atomic_fetch_add_explicit(&acnts[e], 1u, memory_order_relaxed);
}}
simdgroup_barrier(mem_flags::mem_threadgroup);

uint c = cnts[lane];
uint off = simd_prefix_exclusive_sum(c);
offs[lane] = off;
cnts[lane] = off;
simdgroup_barrier(mem_flags::mem_threadgroup);

for (uint i = lane; i < N; i += 32u) {{
  uint e = Inds[i];
  uint pos = atomic_fetch_add_explicit(&acnts[e], 1u, memory_order_relaxed);
  TokIdx[pos]   = i >> 2;
  SlotIdx[pos]  = i;
  RWeights[pos] = float(Scores[i]);
}}
simdgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0u) {{
  uint nh = 0u, nc = 0u;
  for (uint e = 0u; e < {N_EXP}u; ++e) {{
    uint start = offs[e];
    uint tot   = cnts[e] - start;
    uint cur   = 0u;
    while (tot - cur > {HOT_THRESH}u) {{
      uint chunk = min(32u, tot - cur);
      HotMeta[nh * 3u + 0u] = e;
      HotMeta[nh * 3u + 1u] = start + cur;
      HotMeta[nh * 3u + 2u] = chunk;
      nh++;
      cur += chunk;
    }}
    if (cur < tot) {{
      ColdMeta[nc * 3u + 0u] = e;
      ColdMeta[nc * 3u + 1u] = start + cur;
      ColdMeta[nc * 3u + 2u] = tot - cur;
      nc++;
    }}
  }}
  MetaCounts[0] = nh;
  MetaCounts[1] = nc;
}}
""",
        },
        "moe_batch_gu_unified": {
            "input_names": ["X", "TokIdx", "Wgu", "Sgu", "Bgu", "HotMeta", "ColdMeta", "MetaCounts"],
            "output_names": ["MidOut"],
            "header": hdr,
            "source": f"""
const uint nh = MetaCounts[0];
const uint nc = MetaCounts[1];
const uint ty = threadgroup_position_in_grid.y;
if (ty >= nh + nc) return;

const uint tid_tg = thread_index_in_threadgroup;
const uint sg_id  = simdgroup_index_in_threadgroup;
const uint lane   = thread_index_in_simdgroup;

threadgroup half TG[{TG_HALF_ELEMS}];
threadgroup float fg_tg[32];
threadgroup float fu_tg[32];

if (ty < nh) {{
  if (threadgroup_position_in_grid.x >= {HOT_X_GU}u) return;
  const uint ti = ty * 3u;
  const uint e       = HotMeta[ti + 0u];
  const uint m_start = HotMeta[ti + 1u];
  const uint m_cnt   = HotMeta[ti + 2u];
  const uint col0    = threadgroup_position_in_grid.x * 32u;

  constexpr auto descM = mpp::tensor_ops::matmul2d_descriptor(
      32, 32, 32, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<descM, metal::execution_simdgroups<4>> mm;

  threadgroup half* Xs = TG;
  threadgroup half* Gs = TG + 1024u;
  threadgroup half* Us = TG + 2048u;

  auto tX = metal::tensor<threadgroup half, metal::extents<int32_t, 32, 32>, metal::tensor_inline>(Xs, metal::extents<int32_t, 32, 32>());
  auto tG = metal::tensor<threadgroup half, metal::extents<int32_t, 32, 32>, metal::tensor_inline>(Gs, metal::extents<int32_t, 32, 32>());
  auto tU = metal::tensor<threadgroup half, metal::extents<int32_t, 32, 32>, metal::tensor_inline>(Us, metal::extents<int32_t, 32, 32>());

  auto cG = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tG), float>();
  auto cU = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tU), float>();
  auto bG = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tG), float>();
  auto bU = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tU), float>();
  #pragma unroll
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{ cG[i] = 0.0f; cU[i] = 0.0f; }}

  const device uint* Wguu = (const device uint*)Wgu;

  for (uint g = 0u; g < {NG_H}u; ++g) {{
    {{
      uint r = tid_tg >> 2;
      uint c = (tid_tg & 3u) << 3;
      if (r < m_cnt) {{
        uint tok = TokIdx[m_start + r];
        *(threadgroup uint4*)(Xs + r * 32u + c) = *(const device uint4*)(X + (size_t)tok * {H}u + (g << 5) + c);
      }} else {{
        *(threadgroup uint4*)(Xs + r * 32u + c) = uint4(0u);
      }}
    }}
    {{
      uint row_loc = tid_tg >> 2;
      uint w_idx   = tid_tg & 3u;
      uint n_idx   = col0 + row_loc;
      size_t gidx  = ((size_t)e * {D_FF_2}u + n_idx) * {NG_H}u + g;
      size_t uidx  = gidx + (size_t){D_FF}u * {NG_H}u;
      uint vg = Wguu[gidx * 4u + w_idx];
      uint vu = Wguu[uidx * 4u + w_idx];
      if (w_idx == 0u) {{
        fg_tg[row_loc] = e8m0_to_float(Sgu[gidx]);
        fu_tg[row_loc] = e8m0_to_float(Sgu[uidx]);
      }}
      // Stage unscaled exact E2M1 half values [-6, +6] in TG; apply E8M0 scale in float32 below.
      *(threadgroup half4*)(Gs + row_loc * 32u + w_idx * 8u)      = unpack4_e2m1(vg & 0xFFFFu);
      *(threadgroup half4*)(Gs + row_loc * 32u + w_idx * 8u + 4u) = unpack4_e2m1(vg >> 16);
      *(threadgroup half4*)(Us + row_loc * 32u + w_idx * 8u)      = unpack4_e2m1(vu & 0xFFFFu);
      *(threadgroup half4*)(Us + row_loc * 32u + w_idx * 8u + 4u) = unpack4_e2m1(vu >> 16);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mm.run(tX, tG, bG);
    mm.run(tX, tU, bU);
    #pragma unroll
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
      if (cG.is_valid_element(i)) {{
        uint c = uint(cG.get_multidimensional_index(i)[0]);
        cG[i] += bG[i] * fg_tg[c];
        cU[i] += bU[i] * fu_tg[c];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  const size_t b_base = (size_t)e * {D_FF_2}u + col0;
  #pragma unroll
  for (uint16_t i = 0; i < cG.get_capacity(); ++i) {{
    if (cG.is_valid_element(i)) {{
      auto idx = cG.get_multidimensional_index(i);
      uint c = uint(idx[0]);
      uint r = uint(idx[1]);
      if (r < m_cnt) {{
        float gv = min(cG[i] + float(Bgu[b_base + c]), {LIMIT});
        float uv = clamp(cU[i] + float(Bgu[b_base + {D_FF}u + c]), -{LIMIT}, {LIMIT});
        MidOut[(size_t)(m_start + r) * {D_FF}u + col0 + c] = half((uv + 1.0f) * (gv / (1.0f + exp(-gv * {ALPHA}))));
      }}
    }}
  }}
}} else {{
  const uint ti = (ty - nh) * 3u;
  const uint e       = ColdMeta[ti + 0u];
  const uint m_start = ColdMeta[ti + 1u];
  const uint m_cnt   = ColdMeta[ti + 2u];
  const uint col0    = threadgroup_position_in_grid.x * 8u + sg_id * 2u;

  threadgroup half* Xs = TG;
  float acc_g[2][{HOT_THRESH}], acc_u[2][{HOT_THRESH}];
  #pragma unroll
  for (int c = 0; c < 2; ++c) {{
    #pragma unroll
    for (int r = 0; r < {HOT_THRESH}; ++r) {{ acc_g[c][r] = 0.0f; acc_u[c][r] = 0.0f; }}
  }}

  const uint g_loc = lane >> 1;
  const uint h_loc = lane & 1u;
  const device uint4* Wgu4 = (const device uint4*)Wgu;

  for (uint g0 = 0u; g0 < {NG_H}u; g0 += 15u) {{
    uint g_cnt = min(15u, {NG_H}u - g0);
    uint k_cnt = g_cnt * 32u;
    uint k_off = g0 * 32u;
    for (uint r = 0u; r < m_cnt; ++r) {{
      uint tok = TokIdx[m_start + r];
      for (uint i = tid_tg * 8u; i < k_cnt; i += 128u * 8u) {{
        *(threadgroup uint4*)(Xs + r * 480u + i) = *(const device uint4*)(X + (size_t)tok * {H}u + k_off + i);
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < g_cnt * 2u && col0 < {D_FF}u) {{
      uint g = g0 + g_loc;
      uint k_loc = g_loc * 32u + h_loc * 16u;
      #pragma unroll
      for (int c = 0; c < 2; ++c) {{
        uint col = col0 + (uint)c;
        if (col < {D_FF}u) {{
          size_t gwi = ((size_t)e * {D_FF_2}u + col) * {NG_H}u + g;
          size_t uwi = gwi + (size_t){D_FF}u * {NG_H}u;
          uint4 wg4 = Wgu4[gwi];
          uint4 wu4 = Wgu4[uwi];
          uint2 wg2 = (h_loc == 0u) ? uint2(wg4.x, wg4.y) : uint2(wg4.z, wg4.w);
          uint2 wu2 = (h_loc == 0u) ? uint2(wu4.x, wu4.y) : uint2(wu4.z, wu4.w);
          float fg = e8m0_to_float(Sgu[gwi]);
          float fu = e8m0_to_float(Sgu[uwi]);
          half4 wg_h0 = unpack4_e2m1(wg2.x & 0xFFFFu);
          half4 wg_h1 = unpack4_e2m1(wg2.x >> 16);
          half4 wg_h2 = unpack4_e2m1(wg2.y & 0xFFFFu);
          half4 wg_h3 = unpack4_e2m1(wg2.y >> 16);
          half4 wu_h0 = unpack4_e2m1(wu2.x & 0xFFFFu);
          half4 wu_h1 = unpack4_e2m1(wu2.x >> 16);
          half4 wu_h2 = unpack4_e2m1(wu2.y & 0xFFFFu);
          half4 wu_h3 = unpack4_e2m1(wu2.y >> 16);
          for (uint r = 0u; r < m_cnt; ++r) {{
            const threadgroup half4* xp = (const threadgroup half4*)(Xs + r * 480u + k_loc);
            float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
            acc_g[c][r] += (dot(x0, float4(wg_h0)) + dot(x1, float4(wg_h1)) + dot(x2, float4(wg_h2)) + dot(x3, float4(wg_h3))) * fg;
            acc_u[c][r] += (dot(x0, float4(wu_h0)) + dot(x1, float4(wu_h1)) + dot(x2, float4(wu_h2)) + dot(x3, float4(wu_h3))) * fu;
          }}
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  #pragma unroll
  for (int c = 0; c < 2; ++c) {{
    uint col = col0 + (uint)c;
    if (col < {D_FF}u) {{
      float bgv = (lane == 0u) ? float(Bgu[(size_t)e * {D_FF_2}u + col]) : 0.0f;
      float buv = (lane == 0u) ? float(Bgu[(size_t)e * {D_FF_2}u + {D_FF}u + col]) : 0.0f;
      for (uint r = 0u; r < m_cnt; ++r) {{
        float gs = simd_sum(acc_g[c][r]);
        float us = simd_sum(acc_u[c][r]);
        if (lane == 0u) {{
          float gv = min(gs + bgv, {LIMIT});
          float uv = clamp(us + buv, -{LIMIT}, {LIMIT});
          MidOut[(size_t)(m_start + r) * {D_FF}u + col] = half((uv + 1.0f) * (gv / (1.0f + exp(-gv * {ALPHA}))));
        }}
      }}
    }}
  }}
}}
""",
        },
        "moe_batch_dn_unified": {
            "input_names": ["MidOut", "SlotIdx", "RWeights", "Wd", "Sd", "Bd", "HotMeta", "ColdMeta", "MetaCounts"],
            "output_names": ["YSlots"],
            "header": hdr,
            "source": f"""
const uint nh = MetaCounts[0];
const uint nc = MetaCounts[1];
const uint ty = threadgroup_position_in_grid.y;
if (ty >= nh + nc) return;

const uint tid_tg = thread_index_in_threadgroup;
const uint sg_id  = simdgroup_index_in_threadgroup;
const uint lane   = thread_index_in_simdgroup;

threadgroup half TG[{TG_HALF_ELEMS}];
threadgroup float fd_tg[64];

if (ty < nh) {{
  if (threadgroup_position_in_grid.x >= {HOT_X_DN}u) return;
  const uint ti = ty * 3u;
  const uint e       = HotMeta[ti + 0u];
  const uint m_start = HotMeta[ti + 1u];
  const uint m_cnt   = HotMeta[ti + 2u];
  const uint col0    = threadgroup_position_in_grid.x * 64u;

  constexpr auto descD = mpp::tensor_ops::matmul2d_descriptor(
      32, 64, 32, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<descD, metal::execution_simdgroups<4>> mm;

  threadgroup half* Xs = TG;
  threadgroup half* Ws = TG + 1024u;

  auto tX = metal::tensor<threadgroup half, metal::extents<int32_t, 32, 32>, metal::tensor_inline>(Xs, metal::extents<int32_t, 32, 32>());
  auto tW = metal::tensor<threadgroup half, metal::extents<int32_t, 32, 64>, metal::tensor_inline>(Ws, metal::extents<int32_t, 32, 64>());

  auto cD = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  auto bD = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
  #pragma unroll
  for (uint16_t i = 0; i < cD.get_capacity(); ++i) cD[i] = 0.0f;

  const device uint* Wdu = (const device uint*)Wd;

  for (uint g = 0u; g < {NG_D}u; ++g) {{
    {{
      uint r = tid_tg >> 2;
      uint c = (tid_tg & 3u) << 3;
      if (r < m_cnt) *(threadgroup uint4*)(Xs + r * 32u + c) = *(const device uint4*)(MidOut + (size_t)(m_start + r) * {D_FF}u + (g << 5) + c);
      else *(threadgroup uint4*)(Xs + r * 32u + c) = uint4(0u);
    }}
    #pragma unroll
    for (uint rep = 0; rep < 2u; ++rep) {{
      uint idx     = tid_tg + rep * 128u;
      uint row_loc = idx >> 2;
      uint w_idx   = idx & 3u;
      uint n_idx   = col0 + row_loc;
      size_t gidx  = ((size_t)e * {H}u + n_idx) * {NG_D}u + g;
      uint vd = Wdu[gidx * 4u + w_idx];
      if (w_idx == 0u) {{
        fd_tg[row_loc] = e8m0_to_float(Sd[gidx]);
      }}
      // Stage unscaled exact E2M1 half values [-6, +6] in TG; apply E8M0 scale in float32 below.
      *(threadgroup half4*)(Ws + row_loc * 32u + w_idx * 8u)      = unpack4_e2m1(vd & 0xFFFFu);
      *(threadgroup half4*)(Ws + row_loc * 32u + w_idx * 8u + 4u) = unpack4_e2m1(vd >> 16);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mm.run(tX, tW, bD);
    #pragma unroll
    for (uint16_t i = 0; i < cD.get_capacity(); ++i) {{
      if (cD.is_valid_element(i)) {{
        uint c = uint(cD.get_multidimensional_index(i)[0]);
        cD[i] += bD[i] * fd_tg[c];
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  const size_t b_base = (size_t)e * {H}u + col0;
  #pragma unroll
  for (uint16_t i = 0; i < cD.get_capacity(); ++i) {{
    if (cD.is_valid_element(i)) {{
      auto idx = cD.get_multidimensional_index(i);
      uint c = uint(idx[0]);
      uint r = uint(idx[1]);
      if (r < m_cnt) {{
        float val = (cD[i] + float(Bd[b_base + c])) * float(RWeights[m_start + r]);
        uint slot = SlotIdx[m_start + r];
        YSlots[(size_t)slot * {H}u + col0 + c] = val;
      }}
    }}
  }}
}} else {{
  const uint ti = (ty - nh) * 3u;
  const uint e       = ColdMeta[ti + 0u];
  const uint m_start = ColdMeta[ti + 1u];
  const uint m_cnt   = ColdMeta[ti + 2u];
  const uint col0    = threadgroup_position_in_grid.x * 16u + sg_id * 4u;

  threadgroup half* Xs = TG;
  float acc[4][{HOT_THRESH}];
  #pragma unroll
  for (int c = 0; c < 4; ++c) {{
    #pragma unroll
    for (int r = 0; r < {HOT_THRESH}; ++r) acc[c][r] = 0.0f;
  }}

  const uint g_loc = lane >> 1;
  const uint h_loc = lane & 1u;
  const device uint4* Wd4 = (const device uint4*)Wd;

  for (uint g0 = 0u; g0 < {NG_D}u; g0 += 15u) {{
    uint g_cnt = min(15u, {NG_D}u - g0);
    uint k_cnt = g_cnt * 32u;
    uint k_off = g0 * 32u;
    for (uint r = 0u; r < m_cnt; ++r) {{
      for (uint i = tid_tg * 8u; i < k_cnt; i += 128u * 8u) {{
        *(threadgroup uint4*)(Xs + r * 480u + i) = *(const device uint4*)(MidOut + (size_t)(m_start + r) * {D_FF}u + k_off + i);
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < g_cnt * 2u && col0 < {H}u) {{
      uint g = g0 + g_loc;
      uint k_loc = g_loc * 32u + h_loc * 16u;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {{
        uint col = col0 + (uint)c;
        if (col < {H}u) {{
          size_t wi = ((size_t)e * {H}u + col) * {NG_D}u + g;
          uint4 wd4 = Wd4[wi];
          uint2 wd2 = (h_loc == 0u) ? uint2(wd4.x, wd4.y) : uint2(wd4.z, wd4.w);
          float fd = e8m0_to_float(Sd[wi]);
          half4 wd_h0 = unpack4_e2m1(wd2.x & 0xFFFFu);
          half4 wd_h1 = unpack4_e2m1(wd2.x >> 16);
          half4 wd_h2 = unpack4_e2m1(wd2.y & 0xFFFFu);
          half4 wd_h3 = unpack4_e2m1(wd2.y >> 16);
          for (uint r = 0u; r < m_cnt; ++r) {{
            const threadgroup half4* xp = (const threadgroup half4*)(Xs + r * 480u + k_loc);
            float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
            acc[c][r] += (dot(x0, float4(wd_h0)) + dot(x1, float4(wd_h1)) + dot(x2, float4(wd_h2)) + dot(x3, float4(wd_h3))) * fd;
          }}
        }}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}

  #pragma unroll
  for (int c = 0; c < 4; ++c) {{
    uint col = col0 + (uint)c;
    if (col < {H}u) {{
      float bdv = (lane == 0u) ? float(Bd[(size_t)e * {H}u + col]) : 0.0f;
      for (uint r = 0u; r < m_cnt; ++r) {{
        float ds = simd_sum(acc[c][r]);
        if (lane == 0u) {{
          float val = (ds + bdv) * float(RWeights[m_start + r]);
          uint slot = SlotIdx[m_start + r];
          YSlots[(size_t)slot * {H}u + col] = val;
        }}
      }}
    }}
  }}
}}
""",
        },
        "moe_reduce4_resid_bf16": {
            "input_names": ["YSlots", "H"],
            "output_names": ["HOut"],
            "header": hdr,
            "source": f"""
uint col4 = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
if (col4 >= {H // 4}u) return;
const device float4* Y4 = (const device float4*)YSlots;
size_t s0 = ((size_t)b * 4u) * {H // 4}u + col4;
float4 acc = Y4[s0] + Y4[s0 + {H // 4}u] + Y4[s0 + {2 * (H // 4)}u] + Y4[s0 + {3 * (H // 4)}u];
size_t hb = (size_t)b * {H}u + col4 * 4u;
HOut[hb + 0u] = bfloat16_t(float(H[hb + 0u]) + acc.x);
HOut[hb + 1u] = bfloat16_t(float(H[hb + 1u]) + acc.y);
HOut[hb + 2u] = bfloat16_t(float(H[hb + 2u]) + acc.z);
HOut[hb + 3u] = bfloat16_t(float(H[hb + 3u]) + acc.w);
""",
        },
    }


def build_kernels(spec: KernelSpec | None = None) -> dict[str, Any]:
    """Compile and return all `mx.fast.metal_kernel` handles for `spec`."""
    import mlx.core as mx

    eff = spec or KernelSpec()
    specs = render_kernel_sources(eff)
    return {
        name: mx.fast.metal_kernel(
            name=name,
            input_names=cfg["input_names"],
            output_names=cfg["output_names"],
            header=cfg["header"],
            source=cfg["source"],
        )
        for name, cfg in specs.items()
    }


__all__ = ["KernelSpec", "build_kernels", "build_metal_header", "render_kernel_sources"]
