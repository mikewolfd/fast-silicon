"""CPU-only unit tests for `ir_dtype_guard` (MLX AST/Metal checks + CoreML 9.0 MIL IR checks)."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import py_compile
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT.parent

try:
    import fast_shared.ir_dtype_guard as ir_dtype_guard
except ModuleNotFoundError:
    _guard_path = REPO_ROOT / "src" / "fast_shared" / "ir_dtype_guard.py"
    _spec = importlib.util.spec_from_file_location("fast_shared_ir_dtype_guard", _guard_path)
    assert _spec is not None and _spec.loader is not None
    ir_dtype_guard = importlib.util.module_from_spec(_spec)
    sys.modules["fast_shared_ir_dtype_guard"] = ir_dtype_guard
    _spec.loader.exec_module(ir_dtype_guard)

check_coreml_mil_program = ir_dtype_guard.check_coreml_mil_program
check_coreml_model = ir_dtype_guard.check_coreml_model
check_mlx_paths = ir_dtype_guard.check_mlx_paths
check_mlx_source = ir_dtype_guard.check_mlx_source


def test_guard_py_compiles() -> None:
    py_compile.compile(str(Path(ir_dtype_guard.__file__)), doraise=True)


def test_detects_integer_fp32_and_bool_sdpa_masks() -> None:
    bad_src = """
import mlx.core as mx

def run_bad_sdpa(q, k, v, B, L):
    int_mask = mx.ones((B, 1, 1, L), dtype=mx.int32)
    a1 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=int_mask)

    f32_mask = mx.where(mx.arange(L)[None, :] <= mx.arange(L)[:, None], 0.0, -65504.0)
    a2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=f32_mask)

    bool_mask = (mx.arange(L)[None, :] <= mx.arange(L)[:, None])
    a3 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=bool_mask)
    return a1, a2, a3
"""
    rep = check_mlx_source(bad_src, flag_bool_sdpa_as_error=False)
    rule_ids = {f.rule_id for f in rep.findings}
    assert "MLX_SDPA_INT_MASK" in rule_ids
    assert "MLX_SDPA_FP32_MASK" in rule_ids
    assert "MLX_SDPA_BOOL_MASK" in rule_ids
    assert len(rep.errors) == 2
    assert len(rep.warnings) == 1

    rep_strict = check_mlx_source(bad_src, flag_bool_sdpa_as_error=True)
    assert len(rep_strict.errors) == 3

    clean_src = """
import mlx.core as mx

def run_clean_sdpa(q, k, v, L):
    cond = mx.arange(L)[None, :] <= mx.arange(L)[:, None]
    fp16_mask = mx.where(cond, mx.array(0.0, dtype=mx.float16), mx.array(-65504.0, dtype=mx.float16))
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=fp16_mask)
"""
    rep_clean = check_mlx_source(clean_src, flag_bool_sdpa_as_error=True)
    assert rep_clean.findings == []


def test_detects_mixed_bf16_fp16_and_implicit_fp32_scale_upcasts() -> None:
    bad_src = """
import mlx.core as mx

def bad_mixed(n, k, w, rhs_idx):
    x = mx.ones((n, k), dtype=mx.bfloat16)
    s_fp16 = mx.ones((n, k // 64), dtype=mx.float16)
    y1 = x * s_fp16

    s_fp32 = mx.ones((n, 1), dtype=mx.float32)
    y2 = x * s_fp32

    y3 = mx.gather_qmm(x, w, scales=s_fp16, rhs_indices=rhs_idx)
    return y1, y2, y3
"""
    rep = check_mlx_source(bad_src)
    rule_ids = [f.rule_id for f in rep.errors]
    assert "MLX_MIXED_BF16_FP16" in rule_ids
    assert "MLX_IMPLICIT_FP32_SCALE_UPCAST" in rule_ids
    assert "MLX_GATHER_QMM_MISSING_LHS" in rule_ids

    matmul_bad = """
import mlx.core as mx

def bad_mm(m, k, n):
    x16 = mx.ones((m, k), dtype=mx.float16)
    w_bf16 = mx.ones((k, n), dtype=mx.bfloat16)
    w_f32 = mx.ones((k, n), dtype=mx.float32)
    c_f32 = mx.ones((m, n), dtype=mx.float32)
    y1 = x16 @ w_bf16
    y2 = mx.matmul(x16, w_f32)
    y3 = mx.addmm(c_f32, x16, x16.astype(mx.float16))
    return y1, y2, y3
"""
    rep_mm = check_mlx_source(matmul_bad)
    mm_rules = [f.rule_id for f in rep_mm.errors]
    assert mm_rules == [
        "MLX_MIXED_BF16_FP16",
        "MLX_IMPLICIT_FP32_SCALE_UPCAST",
        "MLX_IMPLICIT_FP32_SCALE_UPCAST",
    ]


def test_detects_metal_fp32_outputs_and_0d_scalar() -> None:
    bad_src = """
import mlx.core as mx

_BAD_KERN = mx.fast.metal_kernel(
    name="bad_kern",
    input_names=["X", "Scalar"],
    output_names=["Y"],
    source='''
        uint tid = thread_position_in_grid.x;
        device float* out_ptr = (device float*)Y;
        out_ptr[tid] = float(X[tid]) * float(Scalar[0]);
    ''',
)

def launch(x, s):
    sc0 = s.reshape(())
    return _BAD_KERN(
        inputs=[x, sc0],
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
        grid=(x.size, 1, 1),
        threadgroup=(32, 1, 1),
    )[0]
"""
    rep = check_mlx_source(bad_src)
    rule_ids = {f.rule_id for f in rep.errors}
    assert "METAL_FP32_OUTPUT_PTR" in rule_ids
    assert "MLX_METAL_FP32_OUTPUT" in rule_ids
    assert "MLX_METAL_0D_SCALAR" in rule_ids


def test_detects_const_cast_float4_and_bfloat_pointer_mismatches() -> None:
    const_cast_src = """
import mlx.core as mx

_BAD_CC = mx.fast.metal_kernel(
    name="bad_cc",
    input_names=["X"],
    output_names=["Y"],
    source='''
        device float* out_ptr = const_cast<device float*>((const device float*)Y);
        out_ptr[0] = float(X[0]);
    ''',
)
_BAD_F4 = mx.fast.metal_kernel(
    name="bad_f4",
    input_names=["X"],
    output_names=["Y"],
    source='''
        const device float4* ro = (const device float4*)X;
        device float4* rw = (device float4*)Y;
        rw[0] = ro[0];
    ''',
)
_GOOD_RO = mx.fast.metal_kernel(
    name="good_ro_f32",
    input_names=["X"],
    output_names=["Y"],
    source='''
        const device float* x_ro = (const device float*)X;
        device const float4* x4_ro = (device const float4*)X;
        device half* out = (device half*)Y;
        out[0] = half(x_ro[0] + x4_ro[0].x);
    ''',
)
"""
    rep = check_mlx_source(const_cast_src)
    assert [f.rule_id for f in rep.errors] == ["METAL_FP32_OUTPUT_PTR", "METAL_FP32_OUTPUT_PTR"]
    assert {f.symbol for f in rep.errors} == {"<module>"}
    assert len(rep.findings) == 2

    bf16_in_fp16 = """
import mlx.core as mx

_MISMATCH = mx.fast.metal_kernel(
    name="bf16_in_fp16",
    input_names=["X"],
    output_names=["Y"],
    source='''
        device bfloat4* yp = (device bfloat4*)Y;
        yp[0] = *(const device bfloat4*)X;
    ''',
)

def call_fp16(x):
    return _MISMATCH(inputs=[x], output_shapes=[x.shape], output_dtypes=[mx.float16], grid=(1, 1, 1), threadgroup=(1, 1, 1))[0]
"""
    rep_fp16_cfg = check_mlx_source(bf16_in_fp16, expected_act_dtype="float16")
    assert any(f.rule_id == "METAL_BF16_PTR_IN_FP16_KERNEL" for f in rep_fp16_cfg.errors)
    rep_call_mismatch = check_mlx_source(bf16_in_fp16)
    assert any(f.rule_id == "METAL_BF16_PTR_IN_FP16_KERNEL" for f in rep_call_mismatch.errors)


def test_infers_self_act_dtype_without_hardcoded_bfloat16_and_no_cross_class_leak() -> None:
    src = """
import mlx.core as mx

class Fp16Model:
    def __init__(self):
        self.act_dtype = mx.float16

    def forward(self, x):
        s_fp16 = mx.ones((4,), dtype=self.act_dtype)
        s_bf16 = mx.ones((4,), dtype=mx.bfloat16)
        ok = x.astype(self.act_dtype) * s_fp16
        bad = s_fp16 * s_bf16
        return ok, bad

class UnannotatedModel:
    def forward(self, z):
        s = mx.ones((4,), dtype=mx.float16)
        # self.act_dtype here must NOT leak as bfloat16 from global hardcoding or Fp16Model
        w = mx.ones((4,), dtype=self.act_dtype)
        return s * w
"""
    rep = check_mlx_source(src)
    assert [f.rule_id for f in rep.errors] == ["MLX_MIXED_BF16_FP16"]
    assert rep.errors[0].symbol == "<module>.Fp16Model.forward"


def test_detects_compile_and_eval_loop_hazards() -> None:
    bad_src = """
import mlx.core as mx

def forward_bad(self, x, off_cur):
    for layer in self.layers:
        fn = mx.compile(lambda z: z + 1)
        x = fn(x)
        mx.eval(x)
    while True:
        mx.eval(off_cur)
        break
    return x
"""
    rep = check_mlx_source(bad_src)
    rule_ids = {f.rule_id for f in rep.errors}
    assert "MLX_COMPILE_IN_LOOP" in rule_ids
    assert "MLX_EVAL_IN_LAYER_LOOP" in rule_ids
    assert "MLX_SYNC_EVAL_OFFSET_IN_LOOP" in rule_ids


def test_clean_pass_on_fast_repos() -> None:
    fast_shared_src = REPO_ROOT / "src" / "fast_shared"
    fast_shared_tests = REPO_ROOT / "tests"
    assert fast_shared_src.exists(), f"Missing {fast_shared_src}"
    rep_shared = check_mlx_paths([fast_shared_src, fast_shared_tests], flag_bool_sdpa_as_error=True)
    assert rep_shared.findings == [], "Unexpected guard findings in fast-shared:\n" + "\n".join(
        f.format() for f in rep_shared.findings
    )

    sibling_targets = (
        ("fast-gemma4", WORK_ROOT / "fast-gemma4" / "src" / "fast_gemma4", "bfloat16"),
        (
            "fast-embeddinggemma-2",
            WORK_ROOT / "fast-embeddinggemma-2" / "src" / "fast_embeddinggemma_2",
            "float16",
        ),
        ("fast-pplx", WORK_ROOT / "fast-pplx" / "src" / "fast_pplx", None),
    )
    for label, src_dir, act_dtype in sibling_targets:
        if not src_dir.exists():
            continue
        rep = check_mlx_paths([src_dir], expected_act_dtype=act_dtype)
        assert rep.ok, f"Unexpected guard errors in {label}:\n" + "\n".join(
            f.format() for f in rep.errors
        )


def test_detects_new_metal4_quantize_transpose_and_loop_sync_rules() -> None:
    src = """
import mlx.core as mx
import numpy as np

_TENSOR_BAD_HDR = mx.fast.metal_kernel(
    name="bad_hdr",
    input_names=["A"],
    output_names=["C"],
    header="#include <metal_stdlib>",
    source="auto tC = tensor<device float, dextents<int32_t, 2>>(C, dextents<int32_t, 2>(64, 64));",
)

_CONST_BF16 = mx.fast.metal_kernel(
    name="const_bf16_in_fp16",
    input_names=["A"],
    output_names=["C"],
    source="const device bfloat* ap = (const device bfloat*)A; device half* cp = (device half*)C; cp[0] = half(ap[0]);",
)

def bad_all(w_in, lhs_idx, rhs_idx):
    w_bf = mx.ones((64, 64), dtype=mx.bfloat16)
    x_fp = mx.ones((4, 64), dtype=mx.float16)
    wq, sc, bi = mx.quantize(w_bf, group_size=64, bits=4)
    y_qmm = mx.quantized_matmul(x_fp, wq, sc, bi, transpose=True, group_size=64, bits=4)

    x_bf = mx.ones((4, 64), dtype=mx.bfloat16)
    bi_fp = mx.ones((64, 1), dtype=mx.float16)
    y_bi = mx.quantized_matmul(x_bf, wq, sc, bi_fp, transpose=True, group_size=64, bits=4)

    y_unsorted = mx.gather_qmm(x_bf, wq, sc, bi, lhs_indices=lhs_idx, rhs_indices=rhs_idx)

    wt = x_fp.T
    z0 = mx.zeros(())
    out_nc = _CONST_BF16(
        inputs=[wt, z0],
        output_shapes=[(4, 64)],
        output_dtypes=[mx.float16],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]

    q = mx.ones((1, 4, 8, 64), dtype=mx.float16)
    k = mx.ones((1, 4, 8, 64), dtype=mx.float16)
    v = mx.ones((1, 4, 8, 64), dtype=mx.float16)
    m_bf = mx.zeros((1, 1, 8, 8), dtype=mx.bfloat16)
    sdpa_bad = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=m_bf)

    while True:
        mx.eval(x_bf)
        _ = x_bf.item()
        break

    return y_qmm, y_bi, y_unsorted, out_nc, sdpa_bad
"""
    rep = check_mlx_source(src)
    err_ids = {f.rule_id for f in rep.errors}
    warn_ids = {f.rule_id for f in rep.warnings}
    assert "METAL_TENSOR_MISSING_HEADER" in err_ids
    assert "METAL_FP32_OUTPUT_PTR" in err_ids
    assert "METAL_BF16_PTR_IN_FP16_KERNEL" in err_ids
    assert "MLX_IMPLICIT_FP32_SCALE_UPCAST" in err_ids
    assert "MLX_METAL_NONCONTIGUOUS_INPUT" in err_ids
    assert "MLX_METAL_0D_SCALAR" in err_ids
    assert "MLX_SDPA_DTYPE_MISMATCH" in err_ids
    assert "MLX_GATHER_QMM_UNSORTED_INDICES" in warn_ids
    assert "MLX_SYNC_EVAL_IN_WHILE_LOOP" in warn_ids
    assert "MLX_IMPLICIT_HOST_SYNC_IN_LOOP" in warn_ids

    # CoreML AST check flags mb.matmul on non-test files
    cm_src = "def build(mb, x, w):\n    return mb.matmul(x=x, y=w)\n"
    rep_cm = check_mlx_source(cm_src, filename="builder.py", check_coreml_ast=True)
    assert any(f.rule_id == "COREML_PY_MATMUL_ANE" for f in rep_cm.warnings)

    # CLI --coreml flag cleanly executes on a clean file
    assert ir_dtype_guard.main(["--coreml", str(REPO_ROOT / "src" / "fast_shared" / "paths.py")]) == 0


def test_coreml_mil_guard_detects_fp32_casts_and_passes_clean_fp16() -> None:
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    # 1. Clean native FP16 MIL program (with FP16 additive -65504.0 SDPA mask)
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 16, 1, 8), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, 8, 8), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def clean_prog(x, attn_bias):
        w = np.ones((16, 16, 1, 1), dtype=np.float16)
        y = mb.conv(x=x, weight=w, name="c0")
        q = mb.transpose(x=mb.reshape(x=y, shape=[1, 2, 8, 8], name="qr"), perm=[0, 1, 3, 2], name="qt")
        att = mb.scaled_dot_product_attention(query=q, key=q, value=q, attn_mask=attn_bias, name="sdpa")
        out = mb.reshape(x=mb.transpose(x=att, perm=[0, 1, 3, 2], name="at"), shape=[1, 16, 1, 8], name="out")
        return out

    rep_clean_prog = check_coreml_mil_program(clean_prog, model_name="clean_prog")
    assert rep_clean_prog.ok, [f.format() for f in rep_clean_prog.findings]

    ml_clean = ct.convert(
        clean_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    rep_clean_model = check_coreml_model(ml_clean, model_name="ml_clean")
    assert rep_clean_model.ok, [f.format() for f in rep_clean_model.findings]

    # 2. Broken program #1: built in FP32 -> triggers add_fp16_cast pass & fp32 I/O + boundary casts
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 16, 1, 8), dtype=types.fp32)],
        opset_version=ct.target.macOS15,
    )
    def fp32_cast_prog(x):
        w = np.ones((16, 16, 1, 1), dtype=np.float32)
        y = mb.conv(x=x, weight=w, name="c0")
        msq = mb.reduce_mean(x=mb.mul(x=y, y=y, name="sq"), axes=[1], keep_dims=True, name="msq")
        return mb.mul(x=y, y=mb.rsqrt(x=mb.add(x=msq, y=np.float32(1e-6), name="ae"), name="rsq"), name="out")

    ml_add_cast = ct.convert(
        fp32_cast_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    rep_cast = check_coreml_model(ml_add_cast, model_name="ml_add_cast")
    assert not rep_cast.ok
    cast_rules = {f.rule_id for f in rep_cast.errors}
    assert "COREML_NON_FP16_INPUT" in cast_rules
    assert "COREML_NON_FP16_OUTPUT" in cast_rules
    assert "COREML_ADD_FP16_CAST_ARTIFACT" in cast_rules
    assert "COREML_UNINTENDED_FP32_CAST" in cast_rules

    # 3. Broken program #2: explicit cast(..., dtype="fp32") mid-graph + fp32 SDPA mask
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 2, 8, 8), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, 8, 8), dtype=types.fp32),
        ],
        opset_version=ct.target.macOS15,
    )
    def mid_fp32_prog(q, bad_mask):
        q32 = mb.cast(x=q, dtype="fp32", name="bad_upcast")
        q16 = mb.cast(x=mb.add(x=q32, y=np.float32(1.0), name="add32"), dtype="fp16", name="downcast")
        return mb.scaled_dot_product_attention(query=q16, key=q16, value=q16, attn_mask=bad_mask, name="bad_sdpa")

    rep_mid = check_coreml_mil_program(mid_fp32_prog, model_name="mid_fp32_prog")
    assert not rep_mid.ok
    mid_rules = {f.rule_id for f in rep_mid.errors}
    assert "COREML_UNINTENDED_FP32_CAST" in mid_rules
    assert "COREML_FP32_COMPUTE_OP" in mid_rules
    assert "COREML_SDPA_MASK_NOT_FP16" in mid_rules

    # Verify explicit CPU_FP32_SKIP_OPS allowlist cleanly accepts intentional FP32 elementwise ops
    rep_allowed = check_coreml_mil_program(
        mid_fp32_prog,
        model_name="mid_fp32_allowed",
        require_fp16_io=False,
        allow_fp32_ops={"add", "reduce_mean", "rsqrt"},
        flag_add_fp16_cast=False,
    )
    allowed_rules = {f.rule_id for f in rep_allowed.errors}
    assert "COREML_UNINTENDED_FP32_CAST" not in allowed_rules
    assert "COREML_FP32_COMPUTE_OP" not in allowed_rules
    # Bad fp32 SDPA mask is still caught!
    assert "COREML_SDPA_MASK_NOT_FP16" in allowed_rules
