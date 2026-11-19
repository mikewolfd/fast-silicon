"""CPU-only verification for fused Gemma-4 residual + RMSNorm Metal kernels in `gpu.py`.

Zero Metal/GPU execution: uses `py_compile`, AST inspection, kernel binding checks,
and pure-NumPy FP16/FP32 reference equivalence tests.
"""

from __future__ import annotations

import ast
from pathlib import Path
import py_compile
import subprocess

import numpy as np
import pytest

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
GPU_PY = WORKTREE_ROOT / "src" / "fast_embeddinggemma_2" / "gpu.py"


def _extract_module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    consts: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if isinstance(node.value, ast.Constant):
                consts[name] = node.value.value
    return consts


def _extract_metal_kernels(path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    consts = _extract_module_constants(path)
    kernels: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            call = node.value
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "metal_kernel"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "fast"
            ):
                kw = {k.arg: k.value for k in call.keywords if k.arg}
                info: dict[str, object] = {}
                for field in ("name", "input_names", "output_names", "header", "source"):
                    val_node = kw.get(field)
                    if isinstance(val_node, ast.Constant):
                        info[field] = val_node.value
                    elif isinstance(val_node, ast.List):
                        info[field] = [
                            elt.value for elt in val_node.elts if isinstance(elt, ast.Constant)
                        ]
                    elif isinstance(val_node, ast.Name):
                        info[field] = consts.get(val_node.id, val_node.id)
                kernels[node.targets[0].id] = info
    return kernels


def _numpy_rms_norm_f32(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32)
    w32 = w.astype(np.float32)
    msq = np.mean(x32 * x32, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(msq + np.float32(eps))
    return x32 * inv * w32


def _numpy_simulate_fused_res_rms_fp16(
    h_f16: np.ndarray,
    sub_f16: np.ndarray,
    w_post_f16: np.ndarray,
    w_next_f16: np.ndarray | None = None,
    scalar_f16: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Exact lane-level NumPy model of the Metal `half4 -> float4 -> half4` fused kernels."""
    assert h_f16.dtype == np.float16 and sub_f16.dtype == np.float16
    assert h_f16.shape[-1] == 512 and sub_f16.shape == h_f16.shape

    s32 = sub_f16.astype(np.float32)
    h32 = h_f16.astype(np.float32)
    wp32 = w_post_f16.astype(np.float32)

    sq1 = np.mean(s32 * s32, axis=-1, keepdims=True, dtype=np.float32)
    inv1 = np.float32(1.0) / np.sqrt(sq1 + np.float32(eps))
    r32 = h32 + s32 * inv1 * wp32
    if scalar_f16 is not None:
        sc32 = np.float32(scalar_f16.reshape(-1)[0])
        r32 = r32 * sc32

    y_res = r32.astype(np.float16)
    if w_next_f16 is None:
        return y_res, None

    wn32 = w_next_f16.astype(np.float32)
    sq2 = np.mean(r32 * r32, axis=-1, keepdims=True, dtype=np.float32)
    inv2 = np.float32(1.0) / np.sqrt(sq2 + np.float32(eps))
    y_norm = (r32 * inv2 * wn32).astype(np.float16)
    return y_res, y_norm


def test_gpu_py_compile_and_ast_fusion_invariants() -> None:
    py_compile.compile(str(GPU_PY), doraise=True)

    src = GPU_PY.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(GPU_PY))
    consts = _extract_module_constants(GPU_PY)
    assert consts["HIDDEN_SIZE"] == 512
    assert consts["NUM_LAYERS"] == 24
    assert "_FUSED_GEMMA_RES_RMS_SRC" in consts

    # Verify fused_gemma_res_rms signature, D=HIDDEN_SIZE guards, and FP16 output dtypes.
    fn_nodes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "fused_gemma_res_rms" in fn_nodes
    fn = fn_nodes["fused_gemma_res_rms"]
    arg_names = [a.arg for a in fn.args.args]
    kw_names = [a.arg for a in fn.args.kwonlyargs]
    assert arg_names == ["h", "sub", "w_post", "w_next"]
    assert kw_names == ["scalar"]

    fn_src = ast.get_source_segment(src, fn) or ""
    assert "HIDDEN_SIZE" in fn_src
    assert "mx.float16" in fn_src
    assert "mx.float32" not in fn_src
    assert "scalar.ndim != 1" in fn_src
    assert "if m_rows == 0:" in fn_src
    assert "_FUSED_RES_ADD_SC_KERN" in fn_src

    # Verify _forward_trunk replaces the 3 unfused `h + mx.fast.rms_norm(...)` patterns.
    trunk = fn_nodes["_forward_trunk"]
    trunk_src = ast.get_source_segment(src, trunk) or ""
    assert trunk_src.count("fused_gemma_res_rms(") == 3
    for node in ast.walk(trunk):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            bin_src = ast.get_source_segment(src, node) or ""
            assert "rms_norm" not in bin_src, f"Unfused residual+rms_norm remains in _forward_trunk: {bin_src}"

    # Verify layer scalar is stored as 1D (1,), never 0D () scalar in Metal kernels.
    assert 'raw[q + "scalar"].reshape((1,)).astype(mx.float16)' in src
    assert 'raw[q + "scalar"].reshape(()).astype(mx.float16)' not in src


def test_metal_kernel_sources_and_grid_threadgroup_invariants() -> None:
    kernels = _extract_metal_kernels(GPU_PY)
    for var_name, expected_in, expected_out in (
        ("_FUSED_RES_RMS_KERN", ["H", "Sub", "W_post", "W_next"], ["Y_res", "Y_norm"]),
        ("_FUSED_RES_ADD_KERN", ["H", "Sub", "W_post"], ["Y_res"]),
        ("_FUSED_RES_RMS_SC_KERN", ["H", "Sub", "W_post", "Scalar", "W_next"], ["Y_res", "Y_norm"]),
        ("_FUSED_RES_ADD_SC_KERN", ["H", "Sub", "W_post", "Scalar"], ["Y_res"]),
    ):
        assert var_name in kernels, f"Missing Metal kernel {var_name}"
        k = kernels[var_name]
        assert k["input_names"] == expected_in
        assert k["output_names"] == expected_out
        assert k["header"] == "HEADER_STD"
        msrc = str(k["source"])
        assert "if (row >= (uint)H_shape[0]) return;" in msrc
        assert "simd_sum(" in msrc
        assert "1.0f / 512.0f" in msrc
        assert "1e-6f" in msrc
        assert "half4(" in msrc
        assert "float4(" in msrc
        assert "device float" not in msrc
        for off in ("lane * 4", "128 + lane * 4", "256 + lane * 4", "384 + lane * 4"):
            assert off in msrc

    # Verify threadgroup=(256,1,1) / grid=(((m_rows+7)//8)*256,1,1) covers every row exactly once.
    for m_rows in (1, 7, 8, 9, 64, 128, 320, 512, 4096):
        tg_x = 256
        grid_x = ((m_rows + 7) // 8) * 256
        assert grid_x % tg_x == 0
        n_tgs = grid_x // tg_x
        covered_rows: list[int] = []
        for tg_idx in range(n_tgs):
            for tid in range(tg_x):
                sg = tid >> 5
                lane = tid & 31
                row = tg_idx * 8 + sg
                if row < m_rows and lane == 0:
                    covered_rows.append(row)
        assert covered_rows == list(range(m_rows))

    # Verify wrapper passes threadgroup=(256, 1, 1) and FP16 output_dtypes only.
    src = GPU_PY.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(GPU_PY))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fused_gemma_res_rms")
    fn_src = ast.get_source_segment(src, fn) or ""
    assert "tg = (256, 1, 1)" in fn_src
    assert "grid = (((m_rows + 7) // 8) * 256, 1, 1)" in fn_src
    assert "output_dtypes=[mx.float16, mx.float16]" in fn_src
    assert "output_dtypes=[mx.float16]" in fn_src


@pytest.mark.parametrize("rows", [1, 7, 64, 128])
@pytest.mark.parametrize("scalar_val", [None, 0.875, 1.0, -0.5])
def test_numpy_fp16_fp32_equivalence_single_and_multilayer(rows: int, scalar_val: float | None) -> None:
    rng = np.random.default_rng(100 + rows)
    h = rng.normal(0.0, 1.5, size=(rows, 512)).astype(np.float16)
    sub = rng.normal(0.0, 1.2, size=(rows, 512)).astype(np.float16)
    w_post = (1.0 + rng.normal(0.0, 0.2, size=(512,))).astype(np.float16)
    w_next = (1.0 + rng.normal(0.0, 0.2, size=(512,))).astype(np.float16)
    sc_arr = None if scalar_val is None else np.array([scalar_val], dtype=np.float16)

    # Fused kernel math model
    yr_fused, yn_fused = _numpy_simulate_fused_res_rms_fp16(h, sub, w_post, w_next, scalar_f16=sc_arr)
    assert yr_fused.dtype == np.float16 and yr_fused.shape == (rows, 512)
    assert yn_fused is not None and yn_fused.dtype == np.float16 and yn_fused.shape == (rows, 512)

    # Unfused FP32 reference: (h + rms_norm(sub, w_post)) * scalar -> rms_norm(..., w_next)
    yr_ref32 = h.astype(np.float32) + _numpy_rms_norm_f32(sub, w_post)
    if scalar_val is not None:
        yr_ref32 = yr_ref32 * np.float32(scalar_val)
    yn_ref32 = _numpy_rms_norm_f32(yr_ref32, w_next)

    # Unfused FP16 reference (exact behavior of the 3 unfused MLX steps before fusion)
    yr_unfused_f16 = (h + _numpy_rms_norm_f32(sub, w_post).astype(np.float16)).astype(np.float16)
    if sc_arr is not None:
        yr_unfused_f16 = (yr_unfused_f16 * sc_arr[0]).astype(np.float16)
    yn_unfused_f16 = _numpy_rms_norm_f32(yr_unfused_f16, w_next).astype(np.float16)

    def _min_cos(a: np.ndarray, b: np.ndarray) -> float:
        a32 = a.astype(np.float64)
        b32 = b.astype(np.float64)
        dots = np.sum(a32 * b32, axis=-1)
        norms = np.linalg.norm(a32, axis=-1) * np.linalg.norm(b32, axis=-1) + 1e-12
        return float(np.min(dots / norms))

    assert _min_cos(yr_fused, yr_ref32) > 0.99999
    assert _min_cos(yn_fused, yn_ref32) > 0.99999
    assert _min_cos(yr_fused, yr_unfused_f16) > 0.99999
    assert _min_cos(yn_fused, yn_unfused_f16) > 0.99999
    assert float(np.max(np.abs(yr_fused.astype(np.float32) - yr_ref32))) < 5e-3
    assert float(np.max(np.abs(yn_fused.astype(np.float32) - yn_ref32))) < 5e-3

    # Verify single-output res_add mode matches as well
    yr_only, yn_none = _numpy_simulate_fused_res_rms_fp16(h, sub, w_post, w_next_f16=None, scalar_f16=None)
    assert yn_none is None
    assert _min_cos(yr_only, h.astype(np.float32) + _numpy_rms_norm_f32(sub, w_post)) > 0.99999


def test_full_24_layer_res_rms_chain_equivalence() -> None:
    rng = np.random.default_rng(2026)
    rows = 16
    h_init = rng.normal(0.0, 1.0, size=(rows, 512)).astype(np.float16)

    # Unfused FP32 chain
    h_ref = h_init.astype(np.float32)
    h_fus = h_init.copy()
    in_lns = [(1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16) for _ in range(24)]
    final_ln = (1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16)

    xn_fus = _numpy_rms_norm_f32(h_fus, in_lns[0]).astype(np.float16)
    for li in range(24):
        ao = rng.normal(0.0, 0.8, size=(rows, 512)).astype(np.float16)
        dn = rng.normal(0.0, 0.8, size=(rows, 512)).astype(np.float16)
        pp = rng.normal(0.0, 0.8, size=(rows, 512)).astype(np.float16)
        pa_ln = (1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16)
        pff_ln = (1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16)
        poff_ln = (1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16)
        pple_ln = (1.0 + rng.normal(0.0, 0.1, size=(512,))).astype(np.float16)
        sc = np.array([float(rng.uniform(0.6, 1.1))], dtype=np.float16)

        # Reference unfused step
        xn_ref = _numpy_rms_norm_f32(h_ref, in_lns[li])
        assert float(np.min(np.sum(xn_fus.astype(np.float64) * xn_ref.astype(np.float64), axis=-1) / (
            np.linalg.norm(xn_fus.astype(np.float64), axis=-1) * np.linalg.norm(xn_ref.astype(np.float64), axis=-1) + 1e-12
        ))) > 0.9999
        h1_ref = h_ref + _numpy_rms_norm_f32(ao, pa_ln)
        m_in_ref = _numpy_rms_norm_f32(h1_ref, pff_ln)
        h2_ref = h1_ref + _numpy_rms_norm_f32(dn, poff_ln)
        h_ref = (h2_ref + _numpy_rms_norm_f32(pp, pple_ln)) * float(sc[0])

        # Fused 3-step chain
        h1_fus, m_in_fus = _numpy_simulate_fused_res_rms_fp16(h_fus, ao, pa_ln, pff_ln)
        assert m_in_fus is not None
        assert float(np.max(np.abs(m_in_fus.astype(np.float32) - m_in_ref))) < 1e-2
        h2_fus, _ = _numpy_simulate_fused_res_rms_fp16(h1_fus, dn, poff_ln, None)
        next_ln = in_lns[li + 1] if li + 1 < 24 else final_ln
        h_fus, xn_fus = _numpy_simulate_fused_res_rms_fp16(h2_fus, pp, pple_ln, next_ln, scalar_f16=sc)
        assert xn_fus is not None

    out_ref = _numpy_rms_norm_f32(h_ref, final_ln)
    cos = np.sum(xn_fus.astype(np.float64) * out_ref.astype(np.float64), axis=-1) / (
        np.linalg.norm(xn_fus.astype(np.float64), axis=-1) * np.linalg.norm(out_ref.astype(np.float64), axis=-1) + 1e-12
    )
    assert float(np.min(cos)) > 0.99999
