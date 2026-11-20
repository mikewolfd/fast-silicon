"""CPU, stdlib-isolation, IR-guard, and native-build tests for `fast-decision-models`."""

from __future__ import annotations

import io
import os
from pathlib import Path
import pickle
import re
import struct
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pytest

from fast_decision_models._common import softmax_2, softmax_vec
from fast_decision_models.builder import (
    FP16_MIN_NORMAL,
    _build_laya_quant_config,
    _laya_layer_res_scale,
    _laya_layernorm_ch_np,
    _laya_ln_gain_and_eps,
    _load_kev_head_tensors_stdlib_zip,
)
from fast_decision_models.kev_mlx import (
    _GDN_PACKED_BTREE_SRC,
    _KEV_SPECIAL_RE,
    _build_branch_attn_mask,
    _build_state_attn_mask,
    _kev_render,
)
from fast_decision_models.laya_ane import (
    COMPUTE_MODE_ALL,
    COMPUTE_MODE_CPU_AND_ANE,
    COMPUTE_MODE_CPU_ONLY,
    LayaAneBackend,
)
from fast_decision_models.paths import (
    _is_dylib_fresh,
    ensure_laya_dylib,
    resolve_artifacts_dir,
    resolve_home,
    resolve_kev_dir,
    resolve_laya_dir,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_stdlib_only_top_level_import() -> None:
    """Importing `fast_decision_models` stays 100% Python stdlib even under `-B -I -S`."""
    src_dir = _repo_root() / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src_dir)!r}); "
        "import fast_decision_models, fast_decision_models.paths, fast_decision_models.scorers, "
        "fast_decision_models.laya_ane, fast_decision_models.kev_mlx; "
        "banned = {'numpy', 'mlx', 'tokenizers', 'fast_shared', 'huggingface_hub', 'coremltools', 'torch'}; "
        "leaked = sorted(banned.intersection(sys.modules)); "
        "assert not leaked, f'Unexpected non-stdlib imports: {leaked}'; "
        "print('OK')"
    )
    proc = subprocess.run(
        ["/usr/bin/python3", "-B", "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_path_precedence_and_env_overrides(monkeypatch, tmp_path: Path) -> None:
    custom_home = tmp_path / "custom_home"
    custom_art = tmp_path / "custom_art"
    custom_laya = tmp_path / "custom_laya"
    custom_kev = tmp_path / "custom_kev"

    monkeypatch.setenv("FAST_DECISION_MODELS_HOME", str(custom_home))
    monkeypatch.setenv("FAST_DECISION_MODELS_ARTIFACTS_DIR", str(custom_art))
    monkeypatch.setenv("FAST_DECISION_MODELS_LAYA_DIR", str(custom_laya))
    monkeypatch.setenv("FAST_DECISION_MODELS_KEV_DIR", str(custom_kev))

    assert resolve_home() == custom_home.resolve()
    assert resolve_artifacts_dir() == custom_art.resolve()
    assert resolve_laya_dir() == custom_laya.resolve()
    assert resolve_kev_dir() == custom_kev.resolve()

    explicit = tmp_path / "explicit"
    assert resolve_home(explicit) == explicit.resolve()
    assert resolve_artifacts_dir(explicit) == explicit.resolve()
    assert resolve_laya_dir(explicit) == explicit.resolve()
    assert resolve_kev_dir(explicit) == explicit.resolve()


def test_softmax_numerical_stability() -> None:
    p0, p1 = softmax_2(1000.0, 1002.0, 1.0)
    assert abs((p0 + p1) - 1.0) < 1e-12
    assert p1 > 0.85

    vec = softmax_vec([500.0, 501.0, 499.0], 2.0)
    assert len(vec) == 3
    assert abs(sum(vec) - 1.0) < 1e-12
    assert vec[1] > vec[0] > vec[2]


def test_kev_render_and_special_token_escaping() -> None:
    rendered = _kev_render({"status": "ok", "items": ["tls", {"retries": 2}], "none": None})
    assert "status: ok" in rendered
    assert "- tls" in rendered
    assert "retries: 2" in rendered
    escaped = _KEV_SPECIAL_RE.sub(r"<¦\1¦>", "Check <|fim_prefix|> literal")
    assert escaped == "Check <¦fim_prefix¦> literal"


def _pt_float_storage() -> None:
    pass


def _pt_bf16_storage() -> None:
    pass


def _pt_rebuild_tensor_v2(
    storage: object,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool = False,
    backward_hooks: object = None,
) -> None:
    return None


def test_stdlib_zip_head_reader_synthetic_and_cached(tmp_path: Path) -> None:
    """Verify pure-stdlib PyTorch ZIP reader without requiring `torch`."""
    from collections import OrderedDict

    pt_path = tmp_path / "head.pt"
    arr_qw = np.arange(8, dtype=np.float32).reshape(2, 4)
    arr_qb = np.array([0.25, -0.5], dtype=np.float32)
    arr_kw = (np.arange(8, dtype=np.float32) * 0.5).reshape(2, 4)
    arr_kb_f32 = np.array([1.0, -2.0], dtype=np.float32)
    arr_kb_bf16_u16 = (arr_kb_f32.view(np.uint32) >> 16).astype(np.uint16)

    orig_torch_globals = (
        _pt_float_storage.__module__,
        _pt_float_storage.__name__,
        _pt_bf16_storage.__module__,
        _pt_bf16_storage.__name__,
        _pt_rebuild_tensor_v2.__module__,
        _pt_rebuild_tensor_v2.__name__,
    )
    orig_torch_mod = sys.modules.get("torch")
    orig_utils_mod = sys.modules.get("torch._utils")
    try:
        _pt_float_storage.__module__ = "torch"
        _pt_float_storage.__name__ = "FloatStorage"
        _pt_float_storage.__qualname__ = "FloatStorage"
        _pt_bf16_storage.__module__ = "torch"
        _pt_bf16_storage.__name__ = "BFloat16Storage"
        _pt_bf16_storage.__qualname__ = "BFloat16Storage"
        _pt_rebuild_tensor_v2.__module__ = "torch._utils"
        _pt_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
        _pt_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"

        mock_torch = type(sys)("torch")
        mock_torch.FloatStorage = _pt_float_storage
        mock_torch.BFloat16Storage = _pt_bf16_storage
        mock_utils = type(sys)("torch._utils")
        mock_utils._rebuild_tensor_v2 = _pt_rebuild_tensor_v2
        sys.modules["torch"] = mock_torch
        sys.modules["torch._utils"] = mock_utils

        class _PtPickler(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == "__STORAGE__":
                    return ("storage", obj[1], obj[2], "cpu", obj[3])
                return None

        class _TensorProxy:
            def __init__(self, st_cls, key: str, count: int, shape: tuple[int, ...]) -> None:
                self.st_cls = st_cls
                self.key = key
                self.count = count
                self.shape = shape

            def __reduce_ex__(self, _proto: int):
                return (
                    _pt_rebuild_tensor_v2,
                    (("__STORAGE__", self.st_cls, self.key, self.count), 0, self.shape, (1,), False, OrderedDict()),
                )

        payload = {
            "temperature": 2.25,
            "head": OrderedDict(
                [
                    ("q.weight", _TensorProxy(_pt_float_storage, "0", 8, (2, 4))),
                    ("q.bias", _TensorProxy(_pt_float_storage, "1", 2, (2,))),
                    ("k.weight", _TensorProxy(_pt_float_storage, "2", 8, (2, 4))),
                    ("k.bias", _TensorProxy(_pt_bf16_storage, "3", 2, (2,))),
                ]
            ),
        }
        buf = io.BytesIO()
        _PtPickler(buf, protocol=2).dump(payload)

        with zipfile.ZipFile(pt_path, "w") as zf:
            zf.writestr("head/data.pkl", buf.getvalue())
            zf.writestr("head/data/0", arr_qw.astype("<f4").tobytes())
            zf.writestr("head/data/1", arr_qb.astype("<f4").tobytes())
            zf.writestr("head/data/2", arr_kw.astype("<f4").tobytes())
            zf.writestr("head/data/3", arr_kb_bf16_u16.astype("<u2").tobytes())
    finally:
        (
            _pt_float_storage.__module__,
            _pt_float_storage.__name__,
            _pt_bf16_storage.__module__,
            _pt_bf16_storage.__name__,
            _pt_rebuild_tensor_v2.__module__,
            _pt_rebuild_tensor_v2.__name__,
        ) = orig_torch_globals
        _pt_float_storage.__qualname__ = "_pt_float_storage"
        _pt_bf16_storage.__qualname__ = "_pt_bf16_storage"
        _pt_rebuild_tensor_v2.__qualname__ = "_pt_rebuild_tensor_v2"
        if orig_torch_mod is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = orig_torch_mod
        if orig_utils_mod is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = orig_utils_mod

    tensors, temp = _load_kev_head_tensors_stdlib_zip(pt_path)
    assert temp == 2.25
    np.testing.assert_allclose(tensors["q_weight"], arr_qw)
    np.testing.assert_allclose(tensors["q_bias"], arr_qb)
    np.testing.assert_allclose(tensors["k_weight"], arr_kw)
    np.testing.assert_allclose(tensors["k_bias"], arr_kb_f32)


def test_cached_kev_head_pt_if_present() -> None:
    """Smoke check against cached HuggingFace `jaredpalmer/kev-0.8b` `head.pt` when available."""
    import glob

    hits = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--jaredpalmer--kev-0.8b/snapshots/*/head.pt"))
    )
    if not hits:
        return
    tensors, temp = _load_kev_head_tensors_stdlib_zip(Path(hits[-1]))
    assert temp > 0.0
    assert tensors["q_weight"].shape == (256, 1024)
    assert tensors["k_bias"].shape == (256,)


def test_ir_dtype_guard_clean() -> None:
    from fast_shared.ir_dtype_guard import check_mlx_paths

    report = check_mlx_paths([_repo_root() / "src", _repo_root() / "tests"], flag_bool_sdpa_as_error=True)
    assert report.ok, f"fast-ir-guard errors: {[f.format() for f in report.errors]}"
    assert not report.warnings, f"fast-ir-guard warnings: {[f.format() for f in report.warnings]}"


def test_native_dylib_builds_and_exports_symbols() -> None:
    import ctypes

    dylib = ensure_laya_dylib(_repo_root())
    assert dylib.is_file()
    lib = ctypes.CDLL(str(dylib))
    for fn in (
        "laya_ane_inspect_plan",
        "laya_ane_load",
        "laya_ane_exec_batch",
        "laya_ane_last_error",
        "laya_ane_free",
    ):
        assert hasattr(lib, fn), f"Missing exported symbol {fn}"
    lib.laya_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    counts = (ctypes.c_int * 3)(-1, -1, -1)
    lib.laya_ane_inspect_plan(b"/nonexistent/missing.mlmodelc", COMPUTE_MODE_CPU_AND_ANE, counts)
    assert list(counts) == [0, 0, 0]


def test_compute_mode_abi_aligned_with_fast_shared() -> None:
    """Verify ComputeMode enum values match `fast_shared_apple.hpp` (`0=CPU+ANE, 1=CPUOnly, 2=All`)."""
    assert COMPUTE_MODE_CPU_AND_ANE == 0
    assert COMPUTE_MODE_CPU_ONLY == 1
    assert COMPUTE_MODE_ALL == 2

    hdr_laya = (_repo_root() / "native" / "laya_ane_runner.h").read_text(encoding="utf-8")
    assert re.search(r"LAYA_COMPUTE_CPU_AND_ANE\s*=\s*0", hdr_laya)
    assert re.search(r"LAYA_COMPUTE_CPU_ONLY\s*=\s*1", hdr_laya)
    assert re.search(r"LAYA_COMPUTE_ALL\s*=\s*2", hdr_laya)

    shared_h = _repo_root().parent / "fast-shared" / "src" / "fast_shared" / "include" / "fast_shared_apple.hpp"
    if shared_h.is_file():
        txt = shared_h.read_text(encoding="utf-8")
        assert re.search(r"kCpuAndNeuralEngine\s*=\s*0", txt)
        assert re.search(r"kCpuOnly\s*=\s*1", txt)
        assert re.search(r"kAll\s*=\s*2", txt)


def test_laya_residual_layernorm_fp16_precision_and_ftz_cpu() -> None:
    """PyTorch-free NumPy verification of piecewise residual LayerNorm gain/eps & ModernBERT 33K outliers."""
    # 1. Check all 28 layers + head/scorer have normal FP16 eps_u >= 2^-14 and no double attenuation.
    for li in range(28):
        cum_r = _laya_layer_res_scale(li)
        ln_gain, eps_u = _laya_ln_gain_and_eps(cum_r)
        assert float(eps_u) >= FP16_MIN_NORMAL
        assert np.isfinite(float(ln_gain)) and float(ln_gain) > 0.0
        if cum_r < 1.0:
            assert float(ln_gain) >= 1.0, f"Layer {li} (cum_r={cum_r}) double-attenuated with ln_gain={ln_gain}"
    ln_gain_late, eps_late = _laya_ln_gain_and_eps(1.0 / 1024.0)
    assert float(ln_gain_late) == 4.0
    assert float(eps_late) >= FP16_MIN_NORMAL

    # 2. Verify FP16 LayerNorm (with subnormal FTZ enabled!) vs FP64 reference across all 4 residual regimes
    #    including ModernBERT 33,024 outlier tokens and zero-variance rows.
    rng = np.random.default_rng(1234)
    H, S = 1024, 16
    gamma = (1.0 + rng.normal(0.0, 0.1, (H,))).astype(np.float32)
    beta = rng.normal(0.0, 0.05, (H,)).astype(np.float32)

    def _ref_ln_f64(x_orig: np.ndarray, g: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        xf = x_orig.astype(np.float64)
        mn = xf.mean(axis=1, keepdims=True)
        xc = xf - mn
        var = (xc * xc).mean(axis=1, keepdims=True)
        out = (xc / np.sqrt(var + 1e-5)) * g.reshape(1, -1, 1, 1)
        if b is not None:
            out = out + b.reshape(1, -1, 1, 1)
        return out

    # Typical late-layer variance (~16..100) + extreme 33,024 outlier column + zero-variance column
    x_orig = (rng.standard_normal((1, H, 1, S)) * 8.0).astype(np.float32)
    x_orig[0, 17, 0, 0] = 33024.0   # ModernBERT-large outlier magnitude
    x_orig[0, 511, 0, 1] = -33024.0
    x_orig[0, :, 0, 2] = 5.0        # Zero-variance constant token (tests FTZ rsqrt safety!)

    for cum_r in (1.0, 1.0 / 32.0, 1.0 / 256.0, 1.0 / 1024.0):
        # Skip 33K outlier at cum_r=1.0 (recenter happens before layers reach >1K)
        x_in = x_orig.copy()
        if cum_r == 1.0:
            x_in[0, 17, 0, 0] = 400.0
            x_in[0, 511, 0, 1] = -400.0
        elif cum_r == 1.0 / 32.0:
            x_in[0, 17, 0, 0] = 4000.0
            x_in[0, 511, 0, 1] = -4000.0

        x_scaled_f16 = (x_in * cum_r).astype(np.float16)
        got = _laya_layernorm_ch_np(x_scaled_f16, gamma, beta, cum_r=cum_r, flush_subnormals=True)
        ref = _ref_ln_f64(x_in, gamma, beta)
        assert np.all(np.isfinite(got)), f"Non-finite output at cum_r={cum_r}"

        # Zero-variance token must stay finite and equal beta
        np.testing.assert_allclose(got[0, :, 0, 2].astype(np.float32), beta, atol=2e-2)

        # Check cosine similarity on outlier & normal tokens
        for s_idx in (0, 1, 3, 7):
            v_got = got[0, :, 0, s_idx].astype(np.float64)
            v_ref = ref[0, :, 0, s_idx]
            cos = float(np.dot(v_got, v_ref) / (np.linalg.norm(v_got) * np.linalg.norm(v_ref)))
            assert cos > 0.995, f"Low cosine={cos:.6f} at cum_r={cum_r}, s={s_idx}"


def test_laya_selective_quantization_preserves_rope_and_swa_masks() -> None:
    """Verify `linear_quantize_weights` config targets ONLY `conv` ops, preserving RoPE & SWA masks."""

    class _FakeOpCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeOptCfg:
        def __init__(self, *, global_config=None, op_type_configs=None):
            self.global_config = global_config
            self.op_type_configs = op_type_configs

    fake_cto = type("cto", (), {"OptimizationConfig": _FakeOptCfg, "OpLinearQuantizerConfig": _FakeOpCfg})
    cfg = _build_laya_quant_config(fake_cto)
    assert cfg.global_config is None
    assert set(cfg.op_type_configs.keys()) == {"conv"}
    conv_cfg = cfg.op_type_configs["conv"].kwargs
    assert conv_cfg["dtype"] == "int8"
    assert conv_cfg["mode"] == "linear_symmetric"
    assert conv_cfg["granularity"] == "per_channel"
    assert conv_cfg["weight_threshold"] >= 2048


def test_laya_selective_quantization_coreml_optional() -> None:
    """Full CoreML MIL quantization check when `coremltools` is installed."""
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    import coremltools.optimize.coreml as cto

    H, S = 64, 32
    rng = np.random.default_rng(7)
    w_conv = (rng.standard_normal((H, H, 1, 1)) * 0.1).astype(np.float16)
    cos_t = np.cos(np.linspace(0, 3.14, S * H, dtype=np.float32)).reshape(1, H, 1, S).astype(np.float16)
    swa_t = np.where(np.abs(np.arange(S)[:, None] - np.arange(S)[None, :]) <= 8, 0.0, -10000.0).astype(np.float16)[None, None, :, :]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, H, 1, S), dtype=ct.converters.mil.mil.types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def mini_prog(x):
        y = mb.conv(x=x, weight=mb.const(val=w_conv, name="proj_w"), name="proj")
        y = mb.mul(x=y, y=mb.const(val=cos_t, name="rope_cos"), name="rope_mul")
        m = mb.add(x=mb.const(val=swa_t, name="swa_const"), y=np.float16(0.0), name="swa_add")
        return mb.add(x=y, y=mb.reduce_mean(x=m, axes=[2], keep_dims=True), name="out")

    ml = ct.convert(
        mini_prog,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    q_ml = cto.linear_quantize_weights(ml, config=_build_laya_quant_config(cto))
    q_ops = [op for op in q_ml._mil_program["main"].operations if op.op_type == "constexpr_blockwise_shift_scale"]
    assert len(q_ops) == 1
    assert q_ops[0].name == "proj_w_quantized"
    assert np.dtype(q_ops[0].inputs["data"].val.dtype) == np.int8
    mil_str = str(q_ml._mil_program)
    assert "proj_w_quantized" in mil_str
    assert "rope_cos_quantized" not in mil_str
    assert "swa_const_quantized" not in mil_str


def test_paths_artifacts_marker_and_dylib_mtime_freshness(tmp_path: Path, monkeypatch) -> None:
    """Verify `resolve_home` requires `artifacts/` (not `pyproject.toml`/`native`) and mtime freshness works."""
    import fast_decision_models.paths as paths_mod

    fake_pkg = tmp_path / "monorepo" / "packages" / "fast-decision-models"
    fake_file = fake_pkg / "src" / "fast_decision_models" / "paths.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("", encoding="utf-8")
    (fake_pkg / "pyproject.toml").write_text("", encoding="utf-8")
    (fake_pkg / "native").mkdir()

    monkeypatch.delenv("FAST_DECISION_MODELS_HOME", raising=False)
    monkeypatch.delenv("FAST_ARTIFACTS_ROOT", raising=False)
    monkeypatch.setattr(paths_mod, "__file__", str(fake_file))
    # Without `artifacts/`, must fall back to `~/.cache/fast-decision-models`, NOT stop on pyproject.toml/native!
    assert paths_mod.resolve_home() == (Path("~/.cache") / "fast-decision-models").expanduser().resolve()

    # Once `artifacts/` exists, resolve_home picks the checkout!
    (fake_pkg / "artifacts").mkdir()
    assert paths_mod.resolve_home() == fake_pkg.resolve()

    # Verify `_is_dylib_fresh` detects stale dylibs when source mtime is newer
    dylib = tmp_path / "lib.dylib"
    src_mm = tmp_path / "runner.mm"
    src_h = tmp_path / "runner.h"
    src_mm.write_text("v1", encoding="utf-8")
    src_h.write_text("h1", encoding="utf-8")
    dylib.write_bytes(b"bin")
    os.utime(src_mm, (1000, 1000))
    os.utime(src_h, (1000, 1000))
    os.utime(dylib, (2000, 2000))
    assert _is_dylib_fresh(dylib, [src_mm, src_h]) is True
    os.utime(src_mm, (3000, 3000))
    assert _is_dylib_fresh(dylib, [src_mm, src_h]) is False


def test_laya_encode_item_truncation_modes() -> None:
    """Verify `LayaAneBackend.encode_item` warns or raises on >512 tokens instead of silent truncation."""
    mock_laya = LayaAneBackend.__new__(LayaAneBackend)
    mock_laya.max_len = 512
    mock_laya.head_max_len = 192
    mock_laya.cls_id = 1
    mock_laya.sep_id = 2
    mock_laya.pad_id = 3
    mock_laya.mask_id = 4
    mock_laya.mask_tok = "[MASK]"
    mock_laya._encode_ids = lambda text: [10 + (i % 50) for i in range(len(text.split()))]  # type: ignore[method-assign]

    qdef = {"type": "noul", "instructions": "Does it match?", "criteria": {"false": "no", "true": "yes"}}
    short_state = " ".join(["word"] * 20)
    enc_ok = mock_laya.encode_item(short_state, qdef, on_truncate="error")
    assert enc_ok["truncated"] is False

    long_state = " ".join(["word"] * 600)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        enc_warn = mock_laya.encode_item(long_state, qdef, on_truncate="warn")
    assert enc_warn["truncated"] is True
    assert len(enc_warn["ids"]) == 512
    assert any(issubclass(w.category, UserWarning) and "truncated" in str(w.message) for w in rec)

    with pytest.raises(ValueError, match="truncated"):
        mock_laya.encode_item(long_state, qdef, on_truncate="error")

    with warnings.catch_warnings(record=True) as rec_ign:
        warnings.simplefilter("always")
        enc_ign = mock_laya.encode_item(long_state, qdef, on_truncate="ignore")
    assert enc_ign["truncated"] is True and len(rec_ign) == 0


def test_kev_vectorized_masks_cache_limits_and_zero_clone_gdn_gpu() -> None:
    """Verify vectorized BF16 masks, cache/wired restoration, and zero-clone GDN `owner_idx` parity."""
    import mlx.core as mx
    from fast_decision_models.kev_mlx import Kev08bMlxBackend

    # 1. Equal-length waves use native "causal" string mask; ragged waves produce exact BF16 additive mask.
    assert _build_state_attn_mask(mx, [0, 0, 0], 16, mx.bfloat16) == "causal"
    assert _build_branch_attn_mask(mx, [0, 0], 16, 8, mx.bfloat16) == "causal"

    pad_lens = [0, 3, 5]
    s_max = 10
    m_st = _build_state_attn_mask(mx, pad_lens, s_max, mx.bfloat16)
    assert m_st.dtype == mx.bfloat16 and m_st.shape == (3, 1, s_max, s_max)
    ref_st = np.full((3, 1, s_max, s_max), -1e9, dtype=np.float32)
    for bi, pl in enumerate(pad_lens):
        for qt in range(pl, s_max):
            ref_st[bi, 0, qt, pl : qt + 1] = 0.0
    np.testing.assert_array_equal(
        np.asarray(m_st.astype(mx.float32)),
        np.asarray(mx.array(ref_st).astype(mx.bfloat16).astype(mx.float32)),
    )

    owner_pads = [0, 3, 3, 5]
    max_br = 6
    m_br = _build_branch_attn_mask(mx, owner_pads, s_max, max_br, mx.bfloat16)
    assert m_br.dtype == mx.bfloat16 and m_br.shape == (4, 1, max_br, s_max + max_br)
    ref_br = np.full((4, 1, max_br, s_max + max_br), -1e9, dtype=np.float32)
    for ri, pl in enumerate(owner_pads):
        for qt in range(max_br):
            ref_br[ri, 0, qt, pl : s_max + qt + 1] = 0.0
    np.testing.assert_array_equal(
        np.asarray(m_br.astype(mx.float32)),
        np.asarray(mx.array(ref_br).astype(mx.bfloat16).astype(mx.float32)),
    )

    # 2. Verify cache/wired limit restoration and inter-wave cache clearing on mock mx.
    events: list[tuple[str, int]] = []

    class _MockMx:
        def set_cache_limit(self, lim: int) -> int:
            events.append(("cache_limit", lim))
            return 999

        def set_wired_limit(self, lim: int) -> int:
            events.append(("wired_limit", lim))
            return 888

        def get_cache_memory(self) -> int:
            return 300 << 20

        def clear_cache(self) -> None:
            events.append(("clear_cache", 1))

    mock_kev = Kev08bMlxBackend.__new__(Kev08bMlxBackend)
    mock_kev._mx = _MockMx()
    mock_kev.backbone = object()
    mock_kev._prev_cache_limit = 12345
    mock_kev._prev_wired_limit = 67890
    mock_kev._limits_restored = False

    mock_kev._maybe_clear_wave_cache()
    assert ("clear_cache", 1) in events
    mock_kev.close()
    assert ("wired_limit", 67890) in events
    assert ("cache_limit", 12345) in events
    assert mock_kev.backbone is None
    # Idempotent second close does not re-restore
    n_events = len(events)
    mock_kev.close()
    assert sum(1 for k, _ in events[n_events:] if k in ("wired_limit", "cache_limit")) == 0

    # 3. Verify GDN Metal kernel `owner_idx` + `WriteState=False` zero-clone branch parity vs `mx.take`!
    gdn_k = mx.fast.metal_kernel(
        name="test_kev_gdn_owner_idx",
        input_names=["q", "k", "v", "g", "beta", "state_in", "owner_idx", "T"],
        output_names=["y", "acc_f32"],
        source=_GDN_PACKED_BTREE_SRC,
    )
    mx.random.seed(19)
    B_pre, N_br, T, Hk, Hv, Dk, Dv = 2, 5, 4, 4, 16, 128, 128
    owner = [0, 1, 0, 1, 0]
    owner_mx = mx.array(owner, dtype=mx.int32)
    st_pre = (mx.random.normal((B_pre, Hv, Dv, Dk)) * 0.2).astype(mx.float32)
    q = (mx.random.normal((N_br, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((N_br, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((N_br, T, Hv, Dv)) * 0.1).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((N_br, T, Hv))).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((N_br, T, Hv))).astype(mx.bfloat16)

    y_zero_clone, dummy_st = gdn_k(
        inputs=[q, k, v, g, beta, st_pre, owner_mx, T],
        template=[
            ("InT", mx.bfloat16), ("StT", mx.float32), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
            ("WriteState", False),
        ],
        grid=(32, Dv // 8, N_br * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(N_br, T, Hv, Dv), (1,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    st_cloned = mx.take(st_pre, owner_mx, axis=0)
    y_cloned, _ = gdn_k(
        inputs=[q, k, v, g, beta, st_cloned, mx.arange(N_br, dtype=mx.int32), T],
        template=[
            ("InT", mx.bfloat16), ("StT", mx.float32), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
            ("WriteState", True),
        ],
        grid=(32, Dv // 8, N_br * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(N_br, T, Hv, Dv), st_cloned.shape],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    mx.eval(y_zero_clone, dummy_st, y_cloned)
    assert dummy_st.shape == (1,)
    assert float(mx.max(mx.abs(y_zero_clone.astype(mx.float32) - y_cloned.astype(mx.float32)))) == 0.0
