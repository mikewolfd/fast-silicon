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

    script_targets = (
        ("fast-bge-small/scripts", WORK_ROOT / "fast-bge-small" / "scripts"),
        ("fast-embeddinggemma-2/scripts", WORK_ROOT / "fast-embeddinggemma-2" / "scripts"),
        ("fast-gemma4/scripts", WORK_ROOT / "fast-gemma4" / "scripts"),
        ("fast-pplx/scripts", WORK_ROOT / "fast-pplx" / "scripts"),
        ("fast-decision-models/scripts", WORK_ROOT / "fast-decision-models" / "scripts"),
    )
    for label, scripts_dir in script_targets:
        if not scripts_dir.exists():
            continue
        rep_scripts = check_mlx_paths([scripts_dir], flag_bool_sdpa_as_error=True, check_coreml_ast=True)
        assert rep_scripts.findings == [], f"Unexpected guard findings in {label}:\n" + "\n".join(
            f.format() for f in rep_scripts.findings
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


def test_coreml_mil_guard_detects_fp16_subnormal_mask_and_divisibility() -> None:
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 16, 1, 8), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def bad_fp16_numerics_prog(x):
        w_odd = np.ones((20, 16, 1, 1), dtype=np.float16)
        c0 = mb.conv(x=x, weight=w_odd, name="conv_odd")
        msq = mb.reduce_mean(x=mb.mul(x=c0, y=c0, name="sq"), axes=[1], keep_dims=True, name="msq")
        rn = mb.mul(
            x=c0,
            y=mb.rsqrt(x=mb.add(x=msq, y=np.float16(1e-6), name="sub_ae"), name="sub_rsq"),  # guard: allow-coreml-eps
            name="rn",
        )
        cl = mb.clip(x=rn, alpha=np.float16(2e-5), beta=np.float16(65504.0), name="sub_clip")  # guard: allow-coreml-eps
        with np.errstate(over="ignore"):
            inf_arr = np.full((1, 20, 1, 8), -1e9, dtype=np.float16)  # guard: allow-coreml-mask
        bad_bias = mb.const(val=inf_arr, name="inf_bias")  # guard: allow-coreml-mask
        return mb.add(x=cl, y=bad_bias, name="out")  # guard: allow-coreml-mask

    rep_num = check_coreml_mil_program(
        bad_fp16_numerics_prog,
        model_name="bad_fp16_numerics_prog",
        check_block_size=32,
        block_axis=0,
    )
    num_rules = [f.rule_id for f in rep_num.errors]
    assert num_rules.count("COREML_ANE_SUBNORMAL_EPS") == 2
    assert num_rules.count("COREML_FP16_MASK_OVERFLOW") == 1
    assert num_rules.count("COREML_QUANT_BLOCK_NOT_DIVISIBLE") == 1


def test_check_divisibility_pattern7_and_ast_quant_block_guard() -> None:
    # 1. Pure-Python `check_divisibility` helper (dict, list, and `.named_modules()`)
    shapes = {
        "layers.0.q_proj": (1536, 1536),
        "layers.0.ple_gate": (256, 1536),
        "layers.0.odd_head": (50, 1536),
        "norm.weight": (1536,),  # 1D norm skipped by default min_ndim=2
    }
    incompat = ir_dtype_guard.check_divisibility(shapes, axis=0, block_size=32)
    assert incompat == {"layers.0.odd_head": (50, 32)}
    assert ir_dtype_guard.check_divisibility(shapes, axis=-1, block_size=64) == {}

    class _FakeMod:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.weight = np.zeros(shape, dtype=np.float16)

    class _FakeModel:
        def named_modules(self):
            return [("ok", _FakeMod((64, 128))), ("bad", _FakeMod((64, 100)))]

    assert ir_dtype_guard.check_divisibility(_FakeModel(), axis=1, block_size=32) == {"bad": (100, 32)}
    with pytest.raises(ValueError, match="block_size must be positive"):
        ir_dtype_guard.check_divisibility(shapes, axis=0, block_size=0)

    # 2. Static AST `COREML_QUANT_BLOCK_NOT_DIVISIBLE` check (true positive + clean exemption)
    bad_quant_src = """
import mlx.core as mx
import coremltools.optimize.coreml as cto

def quant_bad():
    w_odd = mx.ones((64, 50), dtype=mx.bfloat16)
    wq, sc, bi = mx.quantize(w_odd, group_size=64, bits=4)
    w_ok = mx.ones((64, 128), dtype=mx.bfloat16)
    wq2, sc2, bi2 = mx.quantize(w_ok, group_size=64, bits=4)
    cfg_bad = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=0)
    return wq, wq2, cfg_bad
"""
    rep = check_mlx_source(bad_quant_src, filename="quantize_weights.py", check_coreml_ast=True)
    div_errs = [f for f in rep.errors if f.rule_id == "COREML_QUANT_BLOCK_NOT_DIVISIBLE"]
    assert len(div_errs) == 2
    assert [f.line for f in div_errs] == [7, 10]


def test_coreml_ast_subnormal_eps_mask_overflow_and_float_literals() -> None:
    bad_src = """
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

def make_bad_ane_mask(vis):
    m1 = np.where(vis, 0.0, -1e9).astype(np.float16)
    m2 = np.ascontiguousarray(np.where(vis, 0.0, float("-inf")), dtype=np.float16)
    m3 = np.full((1, 1, 8, 8), -np.inf, dtype=np.float16)
    return m1, m2, m3

def build_ane_bad(w_np):
    H = 512

    def _bad_rms_default(x, eps: float = 1e-6):
        msq = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[1], keep_dims=True)
        ae = mb.add(x=msq, y=np.float16(eps))
        return mb.rsqrt(x=ae)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 8), dtype=types.fp16)])
    def prog(x):
        r1 = mb.rsqrt(x=mb.add(x=x, y=np.float16(6.1e-5)))
        c1 = mb.clip(x=r1, alpha=np.float16(1e-6), beta=np.float16(65504.0))
        m1 = mb.maximum(x=c1, y=np.float16(2e-5))
        ln1 = mb.layer_norm(x=m1, axes=[1], epsilon=np.float16(1e-12))
        s1 = mb.select(cond=x, a=np.float16(0.0), b=np.float16(-65505.0))
        f1 = mb.add(x=ln1, y=1.0)
        scale = 0.5
        f2 = mb.mul(x=f1, y=scale)
        f3 = mb.mul(x=f2, y=H ** -0.5)
        f4 = mb.const(val=0.25)
        f5 = mb.fill(shape=[1, 16, 1, 8], value=0.0)
        return mb.add(x=mb.add(x=s1, y=f3), y=mb.add(x=f4, y=f5))

    return prog

def torch_ane_bad(v, scores, mask):
    v = v.clamp(min=1e-6)
    return scores.masked_fill(mask, float("-inf"))
"""
    rep = check_mlx_source(bad_src, filename="build_ane_models.py", check_coreml_ast=True)
    counts: dict[str, int] = {}
    for f in rep.errors:
        counts[f.rule_id] = counts.get(f.rule_id, 0) + 1
    assert counts.get("COREML_ANE_SUBNORMAL_EPS", 0) == 6
    assert counts.get("COREML_FP16_MASK_OVERFLOW", 0) == 5
    assert counts.get("COREML_PY_UNCAST_FLOAT_LITERAL", 0) == 5


def test_coreml_ast_clean_exemptions_for_fp32_helpers_timers_and_valid_ane_ops() -> None:
    clean_src = """
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)
FP16_LDEXP_NORMAL = float(np.ldexp(1.0, -14))
RMS_EPS = 1e-6

def clamp_fp16_eps(eps: float) -> float:
    return float(max(float(eps), FP16_MIN_NORMAL))

def bench_timer(toks: int, dt: float, vec: np.ndarray) -> float:
    norm = np.maximum(np.linalg.norm(vec), 1e-12)
    return (toks / max(dt, 1e-9)) + float(norm)

def build_cpu_mix_w8(w4d):
    def _rmsnorm_mm32(x, w_arr, eps=1e-6, name="rn"):
        sq = mb.mul(x=x, y=x, name=f"{name}_sq")
        ms = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{name}_ms")
        rs = mb.rsqrt(x=mb.add(x=ms, y=np.float32(eps), name=f"{name}_ae"), name=f"{name}_rs")
        return mb.mul(x=x, y=rs, name=f"{name}_n")

    def _lin32(x, w_mat, name="lin"):
        w2d = np.ascontiguousarray(w_mat[:, :, 0, 0].T.astype(np.float32))
        return mb.matmul(x=x, y=mb.const(val=w2d, name=f"{name}_w"), name=name)

    return _rmsnorm_mm32, _lin32

def make_clean_masks(vis):
    ane_mask = np.ascontiguousarray(np.where(vis, 0.0, -40000.0), dtype=np.float16)
    ane_mask2 = np.where(vis, 0.0, -65504.0).astype(np.float16)
    fp32_ref = np.full((1, 1, 8, 8), -1e9, dtype=np.float32)
    best_err = np.full((8, 1), np.inf, dtype=np.float32)
    return ane_mask, ane_mask2, fp32_ref, best_err

def build_ane_clean(H: int = 512, chunk_size: int = 32):
    def _f16(val: float):
        return mb.const(val=np.float16(val))

    def _const_f16(arr: np.ndarray, name: str):
        return mb.const(val=np.ascontiguousarray(arr, dtype=np.float16), name=name)

    def _cumsum_tril(g_4d, prefix: str):
        triu_ones = np.triu(np.ones((chunk_size, chunk_size), dtype=np.float16), k=0)[None, None, :, :]
        return mb.matmul(x=g_4d, y=_const_f16(triu_ones, name=f"{prefix}_triu"), name=f"{prefix}_mm")

    def _rmsnorm_ok(x, eps: float = RMS_EPS):
        eps_val = clamp_fp16_eps(eps)
        msq = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[1], keep_dims=True)
        return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=msq, y=_f16(eps_val))))

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 1, 8), dtype=types.fp16)])
    def prog(x):
        n1 = _rmsnorm_ok(x)
        r_floor = mb.rsqrt(x=mb.add(x=n1, y=np.float16(FP16_MIN_NORMAL)))
        r_ldexp = mb.rsqrt(x=mb.add(x=r_floor, y=np.float16(FP16_LDEXP_NORMAL)))
        relu = mb.maximum(x=r_ldexp, y=np.float16(0.0))
        relu6 = mb.clip(x=relu, alpha=np.float16(0.0), beta=np.float16(6.0))
        safe_max = mb.maximum(x=relu6, y=np.float16(1e-4))
        scaled = mb.mul(x=safe_max, y=np.float16(H ** -0.5))
        q = mb.reshape(x=scaled, shape=[1, 2, 8, 8])
        k = mb.transpose(x=q, perm=[0, 1, 3, 2])
        qk = mb.matmul(x=q, y=k)
        cs = _cumsum_tril(qk, "cs")
        return mb.layer_norm(x=cs, axes=[1], epsilon=np.float16(FP16_MIN_NORMAL))

    return prog
"""
    rep = check_mlx_source(clean_src, filename="build_ane_models.py", check_coreml_ast=True)
    assert rep.findings == [], [f.format() for f in rep.findings]


def test_coreml_matmul_ane_refined_true_positives_and_exemptions() -> None:
    src = """
import numpy as np

def build_causal_transformer_ane(mb, x, w_proj):
    # Static-weight matmuls & mb.linear MUST still warn even if function name contains 'causal'
    y1 = mb.matmul(x=x, y=mb.const(val=w_proj, name="w_proj"))
    y2 = mb.linear(x=y1, weight=mb.const(val=w_proj, name="w_lin"))
    # Dynamic activation x activation matmul (Q@K^T) MUST be cleanly exempt
    q = mb.reshape(x=y2, shape=[1, 4, 8, 16])
    k = mb.transpose(x=q, perm=[0, 1, 3, 2])
    scores = mb.matmul(x=q, y=k)
    return scores
"""
    rep = check_mlx_source(src, filename="builder.py", check_coreml_ast=True)
    assert rep.errors == []
    assert [f.rule_id for f in rep.warnings] == ["COREML_PY_MATMUL_ANE", "COREML_PY_MATMUL_ANE"]
    assert [f.line for f in rep.warnings] == [6, 7]


def test_torch_only_quant_dtypes_rejected_and_cli_block_size(tmp_path: Path) -> None:
    from fast_shared.ir_dtype_guard import validate_export_quant_dtypes

    assert validate_export_quant_dtypes(["int8", "mxint8", "INT6", "DataType.UINT6"], export_backend="coreml") == [
        "int6",
        "mxint8",
        "uint6",
    ]
    assert validate_export_quant_dtypes(["mxint8", "int4"], export_backend="coreai") == ["mxint8"]
    assert validate_export_quant_dtypes(["mxint8", "int6", "uint6"], export_backend="ExportBackend.Torch") == []

    src = """
import coremltools.optimize.coreml as cto
import coreai_opt

def cfg():
    a = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="mxint8")
    b = coreai_opt.OpPalettizerConfig(mode="kmeans", quant_dtype="int6")
    c = coreai_opt.ModuleLinearQuantizerConfig(weight_dtype="uint6")
    d = coreai_opt.OpLinearQuantizerConfig(dtype="mxint8", export_backend="ExportBackend.Torch")
    e = cto.OpLinearQuantizerConfig(dtype="int8")
    return a, b, c, d, e
"""
    rep = check_mlx_source(src, filename="calibrate.py", check_coreml_ast=True)
    errs = [f for f in rep.errors if f.rule_id == "COREML_TORCH_ONLY_QUANT_DTYPE"]
    assert [f.line for f in errs] == [6, 7, 8]

    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import numpy as np

    w_odd = np.ones((50, 32, 1, 1), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 32, 1, 8), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def odd_prog(x):
        return mb.conv(x=x, weight=mb.const(val=w_odd, name="w_odd"), name="c_odd")

    ml_odd = ct.convert(
        odd_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    pkg = tmp_path / "odd.mlpackage"
    ml_odd.save(str(pkg))
    assert ir_dtype_guard.main(["--coreml", "--check-block-size", "32", "--block-axis", "0", str(pkg)]) == 1
    assert ir_dtype_guard.main(["--coreml", "--check-block-size", "25", "--block-axis", "0", str(pkg)]) == 0


def test_sync_eval_in_decode_for_loop_flagged_and_exempt_cpu() -> None:
    src = """
import mlx.core as mx

def _decode_wave(h, max_tokens: int):
    for _step in range(max_tokens):
        mx.eval(h)
    return h

def _decode_wave_annotated(h, max_tokens: int):
    for _step in range(max_tokens):
        mx.eval(h)  # guard: allow-eval
    return h

def train_step(x):
    for _step in range(4):
        mx.eval(x)
    return x
"""
    rep = check_mlx_source(src)
    for_warns = [f for f in rep.warnings if f.rule_id == "MLX_SYNC_EVAL_IN_FOR_LOOP"]
    assert len(for_warns) == 1
    assert for_warns[0].line == 6
