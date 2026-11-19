"""Apple Silicon (M4 / M5+) safetensors forensics, silicon ceiling probes, and roofline modeling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import time
from typing import Any

from .hw_check import probe_host_hardware, warn_if_untested_hardware

DTYPE_BYTES = {
    "BOOL": 1.0,
    "U8": 1.0,
    "I8": 1.0,
    "F8_E4M3": 1.0,
    "F8_E5M2": 1.0,
    "U16": 2.0,
    "I16": 2.0,
    "F16": 2.0,
    "BF16": 2.0,
    "U32": 4.0,
    "I32": 4.0,
    "F32": 4.0,
    "U64": 8.0,
    "I64": 8.0,
    "F64": 8.0,
}


def read_safetensors_headers(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse `*.safetensors` JSON headers on CPU with zero weight allocation."""
    root = Path(model_dir)
    tensors: dict[str, dict[str, Any]] = {}
    st_files = sorted(root.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No *.safetensors files found in {root}")
    for path in st_files:
        with path.open("rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            off0, off1 = meta["data_offsets"]
            shape = meta["shape"]
            tensors[k] = {
                "dtype": meta["dtype"],
                "shape": shape,
                "bytes": off1 - off0,
                "elems": math.prod(shape) if shape else 1,
            }
    return tensors


def _categorize_tensor(name: str) -> str:
    low = name.lower()
    if any(t in low for t in ("vision", "visual", "audio", "mm_projector", "multi_modal")):
        return "multimodal_unused"
    if "embed_tokens_per_layer" in low or "per_layer_model_projection" in low or "ple" in low:
        return "ple_per_layer"
    if "lm_head" in low:
        return "lm_head"
    if "embed" in low:
        return "embed_tokens"
    if "router" in low or (".gate." in low and "gate_proj" not in low and "gate_up" not in low):
        return "router"
    if any(t in low for t in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "sinks", "q_norm", "k_norm", "v_norm")):
        return "attention"
    if any(t in low for t in ("mlp", "experts", "gate_proj", "up_proj", "down_proj", "gate_up", "gate_up_proj")):
        return "mlp_or_moe"
    if "norm" in low or "layer_scalar" in low:
        return "norms_scalars"
    return "other"


def audit_safetensors_weights(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize stored bytes and effective bits/weight per component across a model's tensors."""
    cat_bytes: dict[str, int] = defaultdict(int)
    cat_cnt: dict[str, int] = defaultdict(int)
    base_weights: dict[str, dict[str, Any]] = {}
    aux_bytes_by_prefix: dict[str, int] = defaultdict(int)

    for k, meta in tensors.items():
        cat = _categorize_tensor(k)
        cat_bytes[cat] += meta["bytes"]
        cat_cnt[cat] += 1
        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            base_weights[prefix] = {**meta, "cat": cat, "name": k}
        elif k.endswith(".scales") or k.endswith(".biases"):
            prefix = k.rsplit(".", 1)[0]
            aux_bytes_by_prefix[prefix] += meta["bytes"]

    cat_logical_elems: dict[str, int] = defaultdict(int)
    cat_quant_bytes: dict[str, int] = defaultdict(int)
    for pref, w in base_weights.items():
        cat = w["cat"]
        aux = aux_bytes_by_prefix.get(pref, 0)
        raw_b = w["bytes"]
        if w["dtype"] in ("F16", "BF16"):
            elems = w["elems"]
        elif w["dtype"] == "U32" and aux > 0:
            sc_meta = tensors.get(f"{pref}.scales")
            if sc_meta is not None and sc_meta["elems"] > 0:
                ratio_bytes_per_grp_u32 = (w["elems"] * 4) / sc_meta["elems"]
                bits_est = 4 if ratio_bytes_per_grp_u32 <= 34 else 8
                elems = w["elems"] * (32 // bits_est)
            else:
                elems = w["elems"] * 8
        else:
            elems = w["elems"]
        cat_logical_elems[cat] += elems
        cat_quant_bytes[cat] += raw_b + aux

    eff_bits: dict[str, dict[str, Any]] = {}
    for cat, q_b in cat_quant_bytes.items():
        elems = cat_logical_elems[cat]
        if elems > 0:
            eff_bits[cat] = {
                "params": elems,
                "bytes": q_b,
                "bits_per_elem": round((8.0 * q_b) / elems, 2),
            }
    return {
        "tensor_count": len(tensors),
        "total_bytes": sum(cat_bytes.values()),
        "categories": {
            cat: {"count": cat_cnt[cat], "bytes": b}
            for cat, b in sorted(cat_bytes.items(), key=lambda x: -x[1])
        },
        "effective_precision": eff_bits,
    }


def compute_model_roofline(
    config: dict[str, Any],
    total_weight_bytes: int,
    *,
    gpu_tf: float = 7.4,
    dram_gbs: float = 200.0,
    ctx: int = 2048,
) -> dict[str, Any]:
    """Compute pure first-principles B=1..256 decode and prefill rooflines from config metadata."""
    tc = config.get("text_config", config)
    L = int(tc.get("num_hidden_layers", 24))
    H = int(tc.get("hidden_size", 2880))
    d_ff = int(tc.get("intermediate_size", H))
    n_q = int(tc.get("num_attention_heads", 32))
    n_kv = int(tc.get("num_key_value_heads", n_q))
    hd = int(tc.get("head_dim", H // max(1, n_q)))
    V = int(tc.get("vocab_size", 128000))
    tied = bool(tc.get("tie_word_embeddings", True))

    num_exp = int(tc.get("num_local_experts", tc.get("num_experts", 1)))
    k_top = int(tc.get("num_experts_per_tok", 1))
    n_kv_shared = int(tc.get("num_kv_shared_layers", 0))
    n_kv_own = max(1, L - n_kv_shared)

    ltypes = tc.get("layer_types") or []
    swa_win = int(tc.get("sliding_window", 0) or 0)
    if ltypes:
        n_full_own = sum(1 for lt in ltypes[:n_kv_own] if "full" in str(lt).lower())
        n_swa_own = n_kv_own - n_full_own
    elif swa_win > 0:
        n_full_own = max(1, n_kv_own // 2)
        n_swa_own = n_kv_own - n_full_own
    else:
        n_full_own = n_kv_own
        n_swa_own = 0

    kv_b_per_tok_layer = 2 * n_kv * hd * 2
    kv_kib_per_tok = (n_full_own * kv_b_per_tok_layer) / 1024.0
    swa_fixed_mib = (n_swa_own * min(ctx, swa_win or ctx) * kv_b_per_tok_layer) / (1024.0**2)
    kv_bytes_per_seq_ctx = (n_full_own * ctx + n_swa_own * min(ctx, swa_win or ctx)) * kv_b_per_tok_layer

    gqa_ratio = n_q / max(1, n_kv)
    attn_macs_tok = L * H * (n_q * hd + 2 * n_kv * hd + n_q * hd)
    sdpa_macs_tok = (n_full_own + (L - n_kv_own)) * n_q * ctx * hd + n_swa_own * n_q * min(ctx, swa_win or ctx) * hd
    mlp_factor = k_top if num_exp > 1 else 1
    double_wide = bool(tc.get("use_double_wide_mlp", False))
    avg_dff = d_ff * (1.0 + 0.5 * (n_kv_shared / max(1, L))) if double_wide else d_ff
    mlp_macs_tok = L * mlp_factor * 3 * H * avg_dff
    head_macs_tok = V * H
    flops_per_tok = 2.0 * (attn_macs_tok + sdpa_macs_tok + mlp_macs_tok + head_macs_tok)
    ridge = (gpu_tf * 1000.0) / max(1.0, dram_gbs)

    decode_rows = []
    for B in (1, 4, 8, 16, 32, 64, 128, 256):
        if num_exp > 1:
            p_miss = (num_exp - k_top) / num_exp
            u_frac = 1.0 - (p_miss**B)
            act_w = total_weight_bytes * (0.25 + 0.75 * u_frac)
        else:
            act_w = float(total_weight_bytes)
        kv_b = B * kv_bytes_per_seq_ctx
        tot_b = act_w + kv_b
        step_f = B * flops_per_tok
        ai = step_f / max(1.0, tot_b)
        bw_tps = B / (tot_b / (dram_gbs * 1e9))
        cmp_tps = B / (step_f / (gpu_tf * 1e12))
        decode_rows.append({
            "B": B,
            "act_wt_gb": round(act_w / 1e9, 2),
            "kv_gb": round(kv_b / 1e9, 2),
            "ai": round(ai, 2),
            "bw_ceil_tps": round(bw_tps, 1),
            "cmp_ceil_tps": round(cmp_tps, 1),
            "regime": "COMPUTE" if ai > ridge else "MEM-BW",
        })

    prefill_rows = []
    for m_pref in (512, 4096):
        pf_flops = m_pref * (2.0 * (attn_macs_tok + mlp_macs_tok))
        pf_ms = max((pf_flops / (gpu_tf * 1e12)) * 1000.0, (total_weight_bytes / (dram_gbs * 1e9)) * 1000.0)
        prefill_rows.append({
            "M": m_pref,
            "gflops": round(pf_flops / 1e9, 1),
            "floor_ms": round(pf_ms, 1),
            "tok_per_sec": round(m_pref / (pf_ms / 1000.0), 1),
        })

    return {
        "layers": L,
        "own_kv": n_kv_own,
        "full_own": n_full_own,
        "swa_own": n_swa_own,
        "swa_win": swa_win,
        "kv_shared": n_kv_shared,
        "H": H,
        "d_ff": d_ff,
        "double_wide": double_wide,
        "n_q": n_q,
        "n_kv": n_kv,
        "gqa_ratio": round(gqa_ratio, 1),
        "hd": hd,
        "tied": tied,
        "vocab": V,
        "num_experts": num_exp,
        "top_k": k_top,
        "kv_kib_per_tok": round(kv_kib_per_tok, 1),
        "swa_fixed_mib": round(swa_fixed_mib, 2),
        "kv_mib_per_seq_ctx": round(kv_bytes_per_seq_ctx / (1024**2), 2),
        "gflops_per_tok": round(flops_per_tok / 1e9, 2),
        "ridge_flop_per_byte": round(ridge, 1),
        "decode": decode_rows,
        "prefill": prefill_rows,
    }


def probe_hardware_identity() -> dict[str, Any]:
    """Return Apple Silicon chip brand, core counts, Metal device name, and UMA limits."""
    import mlx.core as mx

    host = probe_host_hardware()
    dev = mx.device_info()
    return {
        "brand": str(host["brand"]),
        "hw_model": str(host["hw_model"]),
        "machine": str(host["machine"]),
        "ncpu": str(host["cpu_cores"]),
        "perf_cores": "" if host["perf_cores"] is None else str(host["perf_cores"]),
        "eff_cores": "" if host["eff_cores"] is None else str(host["eff_cores"]),
        "gpu_cores": host["gpu_cores"],
        "ane_cores": host["ane_cores"],
        "mem_gib": round(int(dev.get("memory_size", 0)) / (1024**3), 2),
        "device_name": str(dev.get("device_name", "")),
        "architecture": str(dev.get("architecture", "")),
        "max_recommended_working_set_bytes": int(dev.get("max_recommended_working_set_size", 0)),
    }


def _bench_ms(fn: Any, *, warmup: int = 5, iters: int = 20) -> float:
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def _bench_pipe_ms(fn: Any, *, warmup: int = 5, iters: int = 25) -> float:
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def measure_dram_bw(sizes_mb: list[int]) -> dict[str, Any]:
    """Measure empirical GPU streaming read & copy DRAM bandwidth via bare-metal Metal float4."""
    import mlx.core as mx

    stream_read = mx.fast.metal_kernel(
        name="fs_probe_stream_read_f4",
        input_names=["src"],
        output_names=["dst"],
        source="""
            const device float4* s4 = (const device float4*)src;
            uint gid = thread_position_in_grid.x;
            float4 v0 = s4[gid * 4 + 0];
            float4 v1 = s4[gid * 4 + 1];
            float4 v2 = s4[gid * 4 + 2];
            float4 v3 = s4[gid * 4 + 3];
            float s = (v0.x + v0.y + v0.z + v0.w) + (v1.x + v1.y + v1.z + v1.w)
                    + (v2.x + v2.y + v2.z + v2.w) + (v3.x + v3.y + v3.z + v3.w);
            if (s > 1e30f) dst[0] = s;
        """,
    )
    stream_copy = mx.fast.metal_kernel(
        name="fs_probe_stream_copy_f4",
        input_names=["src"],
        output_names=["dst"],
        source="""
            const device float4* s4 = (const device float4*)src;
            device float4* d4 = (device float4*)dst;
            uint gid = thread_position_in_grid.x;
            d4[gid] = s4[gid];
        """,
    )

    rows = []
    peak_read_gbs = 0.0
    peak_copy_gbs = 0.0
    for mb in sizes_mb:
        n_f32 = (mb * 1024 * 1024) // 4
        n_f4 = n_f32 // 4
        n_threads_r = n_f4 // 4
        a = mx.ones((n_f32,), dtype=mx.float32)
        a_f4 = mx.ones((n_f4, 4), dtype=mx.float32)
        mx.eval(a, a_f4)

        ms_sum = _bench_ms(lambda: mx.sum(a))
        bw_sum = (mb * 1048576.0 / 1e9) / (ms_sum / 1000.0)
        ms_read = _bench_ms(
            lambda: stream_read(
                inputs=[a_f4],
                output_shapes=[(1,)],
                output_dtypes=[mx.float32],
                grid=(n_threads_r, 1, 1),
                threadgroup=(256, 1, 1),
            )[0]
        )
        bw_read = (mb * 1048576.0 / 1e9) / (ms_read / 1000.0)
        ms_copy = _bench_ms(
            lambda: stream_copy(
                inputs=[a_f4],
                output_shapes=[(n_f4, 4)],
                output_dtypes=[mx.float32],
                grid=(n_f4, 1, 1),
                threadgroup=(256, 1, 1),
            )[0]
        )
        bw_copy = (2.0 * mb * 1048576.0 / 1e9) / (ms_copy / 1000.0)
        if mb >= 128:
            peak_read_gbs = max(peak_read_gbs, bw_read)
            peak_copy_gbs = max(peak_copy_gbs, bw_copy)
        rows.append({
            "mb": mb,
            "mx_sum_gbs": round(bw_sum, 1),
            "metal_f4_read_gbs": round(bw_read, 1),
            "metal_f4_copy_gbs": round(bw_copy, 1),
        })
        del a, a_f4
        gc.collect()
        mx.clear_cache()

    if peak_read_gbs == 0.0 and rows:
        peak_read_gbs = max(r["metal_f4_read_gbs"] for r in rows)
        peak_copy_gbs = max(r["metal_f4_copy_gbs"] for r in rows)
    return {"rows": rows, "peak_read_gbs": round(peak_read_gbs, 1), "peak_copy_gbs": round(peak_copy_gbs, 1)}


def _build_mpp_64x64(dtype_str: str) -> Any:
    import mlx.core as mx

    hdr = """#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""
    tname = "bfloat" if dtype_str == "bfloat16" else "half"
    src = f"""
const uint M = Dims[0], K = Dims[1], N = Dims[2];
const uint n_off = threadgroup_position_in_grid.x * 64u;
const uint m_off = threadgroup_position_in_grid.y * 64u;
if (m_off >= M || n_off >= N) return;
auto tA = tensor<device {tname}, dextents<int32_t, 2>, tensor_inline>(const_cast<device {tname}*>(A), dextents<int32_t, 2>(K, M));
auto tB = tensor<device {tname}, dextents<int32_t, 2>, tensor_inline>(const_cast<device {tname}*>(B), dextents<int32_t, 2>(K, N));
auto tC = tensor<device {tname}, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M));
auto mA = tA.slice(0, int(m_off));
auto mB = tB.slice(0, int(n_off));
auto mC = tC.slice(int(n_off), int(m_off));
constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false);
matmul2d<desc, execution_simdgroups<4>> op;
op.run(mA, mB, mC);
"""
    return mx.fast.metal_kernel(
        name=f"fs_probe_mpp_64x64_{dtype_str}",
        input_names=["A", "B", "Dims"],
        output_names=["C"],
        header=hdr,
        source=src,
    )


def measure_gemm_ceilings(ms: list[int], k_dim: int = 2048, n_dim: int = 8192) -> dict[str, Any]:
    """Measure MLX `matmul`/`quantized_matmul` and Metal 4 `MPP::matmul2d` TFLOPS across M."""
    import mlx.core as mx

    mpp_bf = mpp_fp = None
    mpp_status = "ok"
    try:
        mpp_bf = _build_mpp_64x64("bfloat16")
        mpp_fp = _build_mpp_64x64("float16")
    except Exception as exc:
        mpp_status = f"unavailable ({type(exc).__name__}: {exc})"

    w_bf = mx.random.normal((n_dim, k_dim)).astype(mx.bfloat16)
    w_fp = w_bf.astype(mx.float16)
    wq8, sc8, bi8 = mx.quantize(w_bf, group_size=64, bits=8)
    wq4, sc4, bi4 = mx.quantize(w_bf, group_size=64, bits=4)
    mx.eval(w_bf, w_fp, wq8, sc8, bi8, wq4, sc4, bi4)

    rows = []
    peak_tf = 0.0
    for M in ms:
        flops = 2.0 * M * k_dim * n_dim
        x_bf = mx.random.normal((M, k_dim)).astype(mx.bfloat16)
        x_fp = x_bf.astype(mx.float16)
        dims = mx.array([M, k_dim, n_dim], dtype=mx.uint32)
        mx.eval(x_bf, x_fp, dims)

        t_bf = _bench_pipe_ms(lambda: mx.matmul(x_bf, w_bf.T)) / 1000.0
        t_fp = _bench_pipe_ms(lambda: mx.matmul(x_fp, w_fp.T)) / 1000.0
        t_q8 = _bench_pipe_ms(lambda: mx.quantized_matmul(x_bf, wq8, sc8, bi8, transpose=True, group_size=64, bits=8)) / 1000.0
        t_q4 = _bench_pipe_ms(lambda: mx.quantized_matmul(x_fp, wq4, sc4, bi4, transpose=True, group_size=64, bits=4)) / 1000.0

        row: dict[str, Any] = {
            "M": M,
            "mlx_bf16_tf": round(flops / t_bf / 1e12, 2),
            "mlx_fp16_tf": round(flops / t_fp / 1e12, 2),
            "mlx_q8_tf": round(flops / t_q8 / 1e12, 2),
            "mlx_q4_tf": round(flops / t_q4 / 1e12, 2),
            "ms_bf16": round(t_bf * 1000.0, 3),
            "ms_q4": round(t_q4 * 1000.0, 3),
        }
        if mpp_bf is not None and M >= 64 and M % 64 == 0 and k_dim % 32 == 0 and n_dim % 64 == 0:
            try:
                gx = ((n_dim + 63) // 64) * 128
                gy = (M + 63) // 64
                t_mpp_bf = _bench_pipe_ms(
                    lambda: mpp_bf(
                        inputs=[x_bf, w_bf, dims],
                        grid=(gx, gy, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(M, n_dim)],
                        output_dtypes=[mx.bfloat16],
                    )[0]
                ) / 1000.0
                t_mpp_fp = _bench_pipe_ms(
                    lambda: mpp_fp(
                        inputs=[x_fp, w_fp, dims],
                        grid=(gx, gy, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(M, n_dim)],
                        output_dtypes=[mx.float16],
                    )[0]
                ) / 1000.0
                row["mpp_bf16_tf"] = round(flops / t_mpp_bf / 1e12, 2)
                row["mpp_fp16_tf"] = round(flops / t_mpp_fp / 1e12, 2)
                peak_tf = max(peak_tf, row["mpp_bf16_tf"], row["mpp_fp16_tf"])
            except Exception as exc:
                mpp_status = f"runtime_err ({type(exc).__name__}: {exc})"
        peak_tf = max(peak_tf, row["mlx_bf16_tf"], row["mlx_fp16_tf"])
        rows.append(row)
        del x_bf, x_fp
        gc.collect()
    mx.clear_cache()
    return {"K": k_dim, "N": n_dim, "mpp_status": mpp_status, "peak_tf": round(peak_tf, 2), "rows": rows}


def main_probe(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `fast-silicon-probe` / `probe_silicon_ceilings.py`."""
    import mlx.core as mx

    p = argparse.ArgumentParser(description="Probe Apple Silicon M4/M5+ empirical DRAM BW, GEMM ceilings, and ridge.")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--quick", action="store_true", help="Run minimal fast smoke sweep")
    p.add_argument("--json-out", type=str, default="", help="Optional path to save JSON results")
    args = p.parse_args(argv)
    warn_if_untested_hardware()

    hw = probe_hardware_identity()
    if hw["max_recommended_working_set_bytes"] > 0:
        mx.set_wired_limit(hw["max_recommended_working_set_bytes"])

    sizes = [64, 256] if args.quick else [64, 256, 512]
    ms = [1, 16, 32, 64, 512] if args.quick else [1, 8, 16, 24, 32, 48, 64, 128, 512, 2048]
    bw = measure_dram_bw(sizes)
    gemm = measure_gemm_ceilings(ms, args.k, args.n)
    ridge = round((gemm["peak_tf"] * 1000.0) / max(bw["peak_read_gbs"], 1.0), 1)
    out = {"hardware": hw, "dram_bw": bw, "gemm": gemm, "empirical_ridge_flop_per_byte": ridge}

    print(f"Hardware: {hw['brand']} | {hw['architecture']} | {hw['mem_gib']} GiB | P={hw['perf_cores']} E={hw['eff_cores']}")
    print(f"Empirical GPU DRAM Ceiling: read={bw['peak_read_gbs']} GB/s | copy(2x)={bw['peak_copy_gbs']} GB/s")
    for r in bw["rows"]:
        print(f"  {r['mb']:3d} MB | mx.sum={r['mx_sum_gbs']:5.1f} GB/s | metal_f4_read={r['metal_f4_read_gbs']:5.1f} GB/s | metal_f4_copy={r['metal_f4_copy_gbs']:5.1f} GB/s")
    print(f"\nGEMM Ceilings (K={args.k}, N={args.n}, MPP={gemm['mpp_status']}):")
    for r in gemm["rows"]:
        mpp_s = f" | mpp_bf={r.get('mpp_bf16_tf', '-')} mpp_fp={r.get('mpp_fp16_tf', '-')}" if "mpp_bf16_tf" in r else ""
        print(f"  M={r['M']:4d} | BF16={r['mlx_bf16_tf']:5.2f} TF ({r['ms_bf16']:6.3f}ms) | FP16={r['mlx_fp16_tf']:5.2f} | Q8={r['mlx_q8_tf']:5.2f} | Q4={r['mlx_q4_tf']:5.2f} ({r['ms_q4']:6.3f}ms){mpp_s}")
    print(f"\nPeak GPU TFLOPS={gemm['peak_tf']} TF | Effective DRAM={bw['peak_read_gbs']} GB/s => Empirical Ridge = {ridge} FLOP/B")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


def main_audit(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `fast-silicon-audit` / `audit_safetensors_model.py`."""
    p = argparse.ArgumentParser(description="Audit safetensors weights & compute M4/M5 roofline on CPU.")
    p.add_argument("model_dir", type=Path, help="Path to local HF/MLX model snapshot directory")
    p.add_argument("--gpu-tf", type=float, default=7.4, help="Empirical peak GPU TFLOPS (default 7.4)")
    p.add_argument("--dram-gbs", type=float, default=200.0, help="Empirical GPU DRAM BW GB/s (default 200.0)")
    p.add_argument("--ctx", type=int, default=2048, help="Effective Full-attention context length")
    args = p.parse_args(argv)
    warn_if_untested_hardware()

    cfg_path = args.model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {args.model_dir}")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tensors = read_safetensors_headers(args.model_dir)
    anatomy = audit_safetensors_weights(tensors)
    roof = compute_model_roofline(raw_cfg, anatomy["total_bytes"], gpu_tf=args.gpu_tf, dram_gbs=args.dram_gbs, ctx=args.ctx)

    tot_b = anatomy["total_bytes"]
    print(f"=== SAFETENSORS ANATOMY ({anatomy['tensor_count']} tensors, {tot_b / (1024**3):.3f} GiB / {tot_b / 1e9:.3f} GB) ===")
    for cat, info in anatomy["categories"].items():
        b = info["bytes"]
        print(f"  {cat:20s}: {info['count']:4d} tensors | {b / (1024**3):6.3f} GiB ({100.0 * b / max(1, tot_b):5.1f}%)")
    print("\nEstimated Effective Storage Bits / Weight (incl. scales + biases):")
    for cat, info in sorted(anatomy["effective_precision"].items(), key=lambda x: -x[1]["bytes"]):
        print(f"  {cat:20s}: ~{info['params'] / 1e6:8.1f}M params | {info['bytes'] / 1e6:8.1f} MB | ~{info['bits_per_elem']:4.2f} bits/elem")

    print("\n=== ARCHITECTURE & ROOFLINE PROFILE ===")
    print(f"Layers={roof['layers']} (own_kv={roof['own_kv']}: full={roof['full_own']}, swa={roof['swa_own']} win={roof['swa_win']}, kv_shared={roof['kv_shared']})")
    print(f"H={roof['H']}, d_ff={roof['d_ff']} (double_wide={roof['double_wide']}), n_q={roof['n_q']}, n_kv={roof['n_kv']} (GQA {roof['gqa_ratio']}:1), hd={roof['hd']}, tied={roof['tied']}, vocab={roof['vocab']}")
    if roof["num_experts"] > 1:
        print(f"Sparse MoE: {roof['num_experts']} experts, top-{roof['top_k']}")
    print(f"KV (FP16/BF16 @ ctx={args.ctx}): {roof['kv_kib_per_tok']} KiB/tok full + {roof['swa_fixed_mib']} MiB/seq SWA = {roof['kv_mib_per_seq_ctx']} MiB/seq")
    print(f"Compute/tok @ ctx={args.ctx}: {roof['gflops_per_tok']} GFLOPs | Target: {args.gpu_tf} TF, {args.dram_gbs} GB/s (Ridge={roof['ridge_flop_per_byte']} FLOP/B)")
    print(f"\n{'B':>4s} | {'ActWt(GB)':>9s} | {'KV(GB)':>7s} | {'AI(F/B)':>7s} | {'BW_Ceil(t/s)':>12s} | {'Cmp_Ceil(t/s)':>13s} | {'Regime':>9s}")
    for r in roof["decode"]:
        print(f"{r['B']:4d} | {r['act_wt_gb']:9.2f} | {r['kv_gb']:7.2f} | {r['ai']:7.2f} | {r['bw_ceil_tps']:12.1f} | {r['cmp_ceil_tps']:13.1f} | {r['regime']:>9s}")
    for r in roof["prefill"]:
        print(f"Prefill M={r['M']:4d}: {r['gflops']:7.1f} GF => Roofline floor = {r['floor_ms']:6.1f} ms ({r['tok_per_sec']:7.1f} tok/s)")
