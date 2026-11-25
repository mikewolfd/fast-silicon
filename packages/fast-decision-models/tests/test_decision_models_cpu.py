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
        [sys.executable, "-B", "-I", "-S", "-c", code],
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
        elif cum_r == 1.0 / 256.0:
            x_in[0, 17, 0, 0] = 16000.0
            x_in[0, 511, 0, 1] = -16000.0

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
    # Without `artifacts/`, must fall back to `$HOME/.cache/fast-decision-models`, NOT stop on pyproject.toml/native!
    assert paths_mod.resolve_home() == (Path.home() / ".cache" / "fast-decision-models").resolve()

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


def test_laya_choice6_honors_head_max_len_256_without_option_clipping() -> None:
    """Verify `head_max_len=256` from `rl_agent_config.json` prevents `Choice(6)` option clipping at `S=512`."""
    assert LayaAneBackend._resolve_head_max_len({}, 512) == 192
    assert LayaAneBackend._resolve_head_max_len({"head_max_len": 256}, 512) == 256
    assert LayaAneBackend._resolve_head_max_len({"head_max_len": 512}, 512) == 256

    mock_laya = LayaAneBackend.__new__(LayaAneBackend)
    mock_laya.max_len = 512
    mock_laya.head_max_len = LayaAneBackend._resolve_head_max_len({"head_max_len": 256}, 512)
    mock_laya.cls_id = 1
    mock_laya.sep_id = 2
    mock_laya.pad_id = 3
    mock_laya.mask_id = 4
    mock_laya.mask_tok = "[MASK]"
    mock_laya._encode_ids = lambda text: [10 + (i % 50) for i in range(len(text.split()))]  # type: ignore[method-assign]

    # 6 router options of 30 tokens each (+1 [MASK] = 31 tokens -> 186 total option tokens) + 25-token instructions.
    # At legacy `head_max_len=192`, `opt_budget = 6 < 16` clipped each option to 29 tokens; `head_max_len=256` preserves all 31!
    crit = {f"kind_{j}": " ".join(["desc"] * 29) for j in range(6)}
    qdef = {
        "type": "choice",
        "instructions": " ".join(["select"] * 22),
        "criteria": crit,
    }
    enc_512 = mock_laya.encode_item(" ".join(["body"] * 40), qdef, on_truncate="error")
    assert enc_512["truncated"] is False
    # Each consecutive marker is separated by exactly 31 tokens (`[MASK]` + 30 option tokens)
    gaps = [enc_512["markers"][i + 1] - enc_512["markers"][i] for i in range(5)]
    assert gaps == [31, 31, 31, 31, 31]

    # At `max_len=256`, `eff_head` safely drops to `128` leaving room for state tokens.
    enc_256 = mock_laya.encode_item(" ".join(["body"] * 20), qdef, max_len=256, on_truncate="ignore")
    assert len(enc_256["ids"]) <= 256
    assert enc_256["markers"][-1] < 128


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
        for qt in range(pl):
            ref_st[bi, 0, qt, qt] = 0.0
        for qt in range(pl, s_max):
            ref_st[bi, 0, qt, pl : qt + 1] = 0.0
    np.testing.assert_array_equal(
        np.asarray(m_st.astype(mx.float32)),
        np.asarray(mx.array(ref_st).astype(mx.bfloat16).astype(mx.float32)),
    )
    m_st_f16 = _build_state_attn_mask(mx, pad_lens, s_max, mx.float16)
    m_st_f16_np = np.asarray(m_st_f16.astype(mx.float32))
    assert np.isfinite(m_st_f16_np).all()
    assert float(m_st_f16_np.min()) == -65504.0
    assert float(m_st_f16_np[1, 0, 0, 0]) == 0.0
    assert float(m_st_f16_np[2, 0, 4, 4]) == 0.0

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


def test_laya_coreai_opt_export_sensitivity_and_mil_guards_cpu(tmp_path: Path) -> None:
    import argparse

    pytest.importorskip("coreai_opt")
    pytest.importorskip("coreai_torch")
    pytest.importorskip("coremltools")

    from fast_decision_models.builder import (
        _compile_laya_seq_bundle,
        add_setup_weights_arguments,
    )
    from fast_decision_models.coreai_export import (
        build_coreai_opt_laya_config,
        run_laya_coreai_export,
        run_laya_coreai_sensitivity_sweep,
    )

    # 1. Torch-only dtype rejection & Pattern 7 divisibility check
    with pytest.raises(ValueError, match="ExportBackend.Torch"):
        build_coreai_opt_laya_config(preset="w8", export_backend="coreml", requested_dtypes=["mxint8"])
    with pytest.raises(ValueError, match="ExportBackend.Torch"):
        build_coreai_opt_laya_config(preset="w4", export_backend="coreai", requested_dtypes=["int6"])

    _, qmeta = build_coreai_opt_laya_config(
        preset="w4_per_block",
        block_size=32,
        export_backend="coreai",
        weight_shapes={"tiny_proj": (64, 24, 1, 1), "ok_proj": (64, 64, 1, 1)},
    )
    assert "tiny_proj" in qmeta["excluded_modules"]
    assert qmeta["skipped_non_divisible"]["tiny_proj"] == [24, 32]

    # 2. Tiny ModernBERT + typed scoring head synthetic weights matching builder.py keys
    h, inter, heads, n_layers, seq_len = 64, 128, 4, 4, 32
    enc_cfg = {
        "hidden_size": h,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "local_attention": 16,
        "norm_eps": 1e-5,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    }
    rng = np.random.default_rng(13)
    w_np: dict[str, np.ndarray] = {
        "encoder.embeddings.norm.weight": np.ones((h,), dtype=np.float32),
        "encoder.final_norm.weight": np.ones((h,), dtype=np.float32),
        "scorer.0.weight": np.ones((h,), dtype=np.float32),
        "scorer.0.bias": np.zeros((h,), dtype=np.float32),
        "scorer.1.weight": (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32),
        "scorer.1.bias": np.zeros((h,), dtype=np.float32),
        "scorer.3.weight": (rng.normal(0.0, 0.02, size=(1, h))).astype(np.float32),
        "scorer.3.bias": np.zeros((1,), dtype=np.float32),
    }
    for i in range(n_layers):
        px = f"encoder.layers.{i}."
        if i > 0:
            w_np[px + "attn_norm.weight"] = np.ones((h,), dtype=np.float32)
        w_np[px + "mlp_norm.weight"] = np.ones((h,), dtype=np.float32)
        w_np[px + "attn.Wqkv.weight"] = (rng.normal(0.0, 0.02, size=(3 * h, h))).astype(np.float32)
        w_np[px + "attn.Wo.weight"] = (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32)
        w_np[px + "mlp.Wi.weight"] = (rng.normal(0.0, 0.02, size=(2 * inter, h))).astype(np.float32)
        w_np[px + "mlp.Wo.weight"] = (rng.normal(0.0, 0.02, size=(h, inter))).astype(np.float32)
    for hi in range(2):
        hp = f"head.layers.{hi}."
        w_np[hp + "norm1.weight"] = np.ones((h,), dtype=np.float32)
        w_np[hp + "norm1.bias"] = np.zeros((h,), dtype=np.float32)
        w_np[hp + "norm2.weight"] = np.ones((h,), dtype=np.float32)
        w_np[hp + "norm2.bias"] = np.zeros((h,), dtype=np.float32)
        w_np[hp + "self_attn.in_proj_weight"] = (rng.normal(0.0, 0.02, size=(3 * h, h))).astype(np.float32)
        w_np[hp + "self_attn.in_proj_bias"] = np.zeros((3 * h,), dtype=np.float32)
        w_np[hp + "self_attn.out_proj.weight"] = (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32)
        w_np[hp + "self_attn.out_proj.bias"] = np.zeros((h,), dtype=np.float32)
        head_ff = inter * 2  # Explicitly decouple head FFN width (4096 vs 2624 in prod Laya-Typed-421M).
        w_np[hp + "linear1.weight"] = (rng.normal(0.0, 0.02, size=(head_ff, h))).astype(np.float32)
        w_np[hp + "linear1.bias"] = np.zeros((head_ff,), dtype=np.float32)
        w_np[hp + "linear2.weight"] = (rng.normal(0.0, 0.02, size=(h, head_ff))).astype(np.float32)
        w_np[hp + "linear2.bias"] = np.zeros((h,), dtype=np.float32)

    # 3. Core AI (.aimodel) + CoreML (.mlpackage) export & sensitivity sweep
    res_ai = run_laya_coreai_export(
        out_dir=tmp_path / "ai",
        weights=w_np,
        enc_cfg=enc_cfg,
        seq_len=seq_len,
        export_backend="coreai",
        preset="w4_per_block",
        calib_batches=2,
        stem="tiny_laya_ai",
    )
    assert res_ai.asset_path.exists()
    assert res_ai.asset_summary is not None and res_ai.asset_summary["asset_valid"] is True
    assert "Float16" in res_ai.asset_summary["compute_types"]
    assert res_ai.sensitivity_report is not None
    assert "scorer_l2" in res_ai.sensitivity_report["excluded_modules"]
    assert res_ai.sensitivity_report["compute_dtype"] == "fp16"
    assert res_ai.sensitivity_report["cosine_vs_teacher"] > 0.99

    # Bug 4: w4_per_block is blocked on CoreML by default, while w8 CoreML export passes cleanly.
    with pytest.raises(ValueError, match="allow_experimental_coreml_w4_block"):
        run_laya_coreai_export(
            out_dir=tmp_path / "ml_bad",
            weights=w_np,
            enc_cfg=enc_cfg,
            seq_len=seq_len,
            export_backend="coreml",
            preset="w4_per_block",
            calib_batches=2,
            stem="tiny_laya_ml_bad",
        )

    res_ml = run_laya_coreai_export(
        out_dir=tmp_path / "ml",
        weights=w_np,
        enc_cfg=enc_cfg,
        seq_len=seq_len,
        export_backend="coreml",
        preset="w8",
        calib_batches=2,
        stem="tiny_laya_ml",
    )
    assert res_ml.asset_path.exists()
    assert res_ml.guard_report is not None and res_ml.guard_report["ok"] is True

    sweep = run_laya_coreai_sensitivity_sweep(
        out_dir=tmp_path / "sweep",
        weights=w_np,
        enc_cfg=enc_cfg,
        seq_len=seq_len,
        export_backend="coreai",
        presets=("fp16", "w8", "w4_per_block"),
        calib_batches=2,
        stem_prefix="tiny_laya_sweep",
    )
    assert [r["preset"] for r in sweep] == ["fp16", "w8", "w4_per_block"]
    assert sweep[0]["analytical_bpw"] == 16.0
    assert sweep[2]["analytical_bpw"] < sweep[1]["analytical_bpw"] < sweep[0]["analytical_bpw"]

    # 4. Existing CoreML MIL _compile_laya_seq_bundle passes ir_dtype_guard with flag_add_fp16_cast=True
    mc = _compile_laya_seq_bundle(seq_len, w_np, enc_cfg, tmp_path / "mil", guard_ir=True)
    assert mc.is_dir()

    # 5. CLI parser exposes Core AI + guard flags cleanly
    parser = argparse.ArgumentParser()
    add_setup_weights_arguments(parser)
    ns = parser.parse_args(
        ["--build-laya-ane", "--backend", "coreai", "--coreai-preset", "w4_per_block", "--guard-ir", "--inspect-placement"]
    )
    assert ns.backend == "coreai"
    assert ns.coreai_preset == "w4_per_block"
    assert ns.guard_ir is True
    assert ns.inspect_placement is True


def test_build_laya_ane_artifacts_coreai_uses_export_ready_key_cpu(monkeypatch: Any, tmp_path: Path) -> None:
    import json
    import types
    import fast_shared.hw_check as fs_hw
    import fast_decision_models.builder as fdm_builder
    import fast_decision_models.coreai_export as fdm_ce

    src_dir = tmp_path / "laya_src"
    (src_dir / "encoder").mkdir(parents=True)
    (src_dir / "tokenizer").mkdir(parents=True)
    (src_dir / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (src_dir / "encoder" / "config.json").write_text(
        json.dumps({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 4, "layer_types": ["full_attention"]}),
        encoding="utf-8",
    )
    (src_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    # Mock weights loading without MLX safetensors file by intercepting mx.load and _resolve_dir_or_hf.
    import mlx.core as mx

    fake_weights = {
        "encoder.embeddings.tok_embeddings.weight": mx.zeros((8, 64), dtype=mx.float32),
        "type_emb.weight": mx.zeros((4, 64), dtype=mx.float32),
    }
    monkeypatch.setattr(fdm_builder, "_resolve_dir_or_hf", lambda p, repo: src_dir)
    monkeypatch.setattr(mx, "load", lambda p: fake_weights)
    # Bug 1 regression guard: probe_core_ai_support exposes `export_ready` (NOT `coreai_export_ready`).
    monkeypatch.setattr(fs_hw, "probe_core_ai_support", lambda hw=None: {"export_ready": True})

    seen_stems: list[str] = []

    def _fake_export(**kwargs: Any) -> Any:
        seen_stems.append(kwargs["stem"])
        p = Path(kwargs["out_dir"]) / f"{kwargs['stem']}.aimodel"
        p.mkdir(parents=True, exist_ok=True)
        return types.SimpleNamespace(
            asset_path=p,
            cache_key_sha256="abc123",
            sensitivity_report={"cosine_vs_teacher": 1.0},
            asset_summary={"asset_valid": True},
        )

    monkeypatch.setattr(fdm_ce, "run_laya_coreai_export", _fake_export)
    out = fdm_builder.build_laya_ane_artifacts(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        laya_dir=str(src_dir),
        backend="coreai",
        coreai_preset="w8",
    )
    assert out.is_dir()
    assert seen_stems == ["laya_typed_s256_w8", "laya_typed_s512_w8"]


def test_laya_tokenizer_helper_zero_coreml_load_cpu(tmp_path: Path) -> None:
    """Verify `LayaTokenizerHelper` and `LayaAneBackend.from_tokenizer_only` initialize cleanly without `.mlmodelc` or dylibs."""
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers
    import fast_decision_models as fdm
    from fast_decision_models import LayaTokenizerHelper

    assert fdm.LayaTokenizerHelper is LayaTokenizerHelper
    assert issubclass(LayaAneBackend, LayaTokenizerHelper)

    laya_dir = tmp_path / "laya_snap"
    (laya_dir / "tokenizer").mkdir(parents=True)
    vocab = {
        "[UNK]": 0,
        "[CLS]": 10,
        "[SEP]": 11,
        "[PAD]": 12,
        "[MASK]": 13,
        "choice": 14,
        "noul": 15,
        "score": 16,
        "question:": 17,
        "select": 18,
        "option": 19,
        "a": 20,
        "b": 21,
        "false:": 22,
        "true:": 23,
        "level": 24,
        "0:": 25,
        "1:": 26,
        "low": 27,
        "high": 28,
        "state": 29,
        "body": 30,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = laya_dir / "tokenizer" / "tokenizer.json"
    tok.save(str(tok_path))

    rl_cfg = {
        "head_max_len": 256,
        "temperature": [1.2, 1.5, 0.8],
        "temperature_by_options": {"choice:2": 1.1, "choice:3-5": 1.3, "noul:2": 0.9},
    }
    (laya_dir / "rl_agent_config.json").write_text(json.dumps(rl_cfg), encoding="utf-8")

    # 1. Construct via from_laya_dir with head_max_len_cap (zero .mlmodelc files exist!)
    h_dir = LayaTokenizerHelper.from_laya_dir(laya_dir, max_len=384, head_max_len_cap=200)
    assert h_dir.laya_dir == laya_dir.resolve()
    assert h_dir.cls_id == 10 and h_dir.sep_id == 11 and h_dir.pad_id == 12 and h_dir.mask_id == 13
    assert h_dir.mask_tok == "[MASK]"
    assert h_dir.max_len == 384 and h_dir.head_max_len == 200
    assert h_dir.resolve_temperature(2, 2) == pytest.approx(0.9)
    assert h_dir._resolve_temperature(0, 2) == pytest.approx(1.1)
    assert h_dir.resolve_temperature(0, 4) == pytest.approx(1.3)
    assert h_dir.resolve_temperature(1, 3) == pytest.approx(1.5)
    assert h_dir.resolve_temperature(0, 2, temperature_override=0.5) == pytest.approx(0.5)

    # 2. Construct via LayaAneBackend.from_tokenizer_only directly from tokenizer.json
    h_tok = LayaAneBackend.from_tokenizer_only(tok_path, rl_agent_config=rl_cfg)
    assert isinstance(h_tok, LayaTokenizerHelper)
    assert h_tok.max_len == 512 and h_tok.head_max_len == 256

    enc_choice = h_tok.encode_item(
        "state [MASK] body",
        {"type": "choice", "instructions": "select option", "criteria": ["a", "b"]},
        on_truncate="error",
    )
    assert enc_choice["qtype"] == 0 and enc_choice["k"] == 2 and enc_choice["truncated"] is False
    assert enc_choice["ids"][0] == 10 and enc_choice["ids"][-1] == 11
    assert [enc_choice["ids"][m] for m in enc_choice["markers"]] == [13, 13]
    # User [MASK] inside state must be neutralized so only the 2 option markers have mask_id
    assert enc_choice["ids"].count(13) == 2

    enc_noul = h_tok.encode_item(
        {"k": "body"},
        {"type": "noul", "instructions": "select", "criteria": {"false": "a", "true": "b"}},
        on_truncate="error",
    )
    assert enc_noul["qtype"] == 2 and enc_noul["k"] == 2

    enc_score = h_tok.encode_item(
        "body",
        {"type": "score", "instructions": "select", "criteria": ["low", "high"]},
        on_truncate="error",
    )
    assert enc_score["qtype"] == 1 and enc_score["k"] == 2


def test_encode_kev_questions_pure_cpu_and_backend_parity(tmp_path: Path) -> None:
    """Verify pure `encode_kev_questions` (0 MLX load) matches `Kev08bMlxBackend.encode_questions` exactly."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    import fast_decision_models as fdm
    from fast_decision_models import (
        KEV_SPECIAL_TOKENS,
        Kev08bMlxBackend,
        encode_kev_questions,
        resolve_kev_special_ids,
    )

    assert fdm.encode_kev_questions is encode_kev_questions
    assert fdm.resolve_kev_special_ids is resolve_kev_special_ids
    assert len(KEV_SPECIAL_TOKENS) == 5

    vocab = {
        "<unk>": 0,
        "status:": 1,
        "ok": 2,
        "check": 3,
        "literal": 4,
        "no:": 5,
        "yes:": 6,
        "bad": 7,
        "good": 8,
        "a": 9,
        "b:": 10,
        "beta": 11,
        "low": 12,
        "high": 13,
        "word": 14,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(list(KEV_SPECIAL_TOKENS))
    tok.no_truncation()
    tok.no_padding()

    sp_ids = resolve_kev_special_ids(tok)
    assert len(sp_ids) == 5
    state_id, q_id, o_id, c_id, d_id = sp_ids

    state = {"status": "ok", "note": "check <|fim_prefix|> literal"}
    questions = [
        {"type": "noul", "instructions": "check", "criteria": {"false": "bad", "true": "good"}},
        {"type": "choice", "instructions": "check", "criteria": ["a", "b"]},
        {"type": "choice", "instructions": "check", "criteria": {"a": "", "b": "beta"}},
        {"type": "score", "instructions": "check", "criteria": ["low", "high"]},
    ]

    st_ids, rows = encode_kev_questions(tok, state, questions)
    assert st_ids[0] == state_id
    # Escaped <|fim_prefix|> in user text must NOT produce raw state_id delimiter inside payload
    assert state_id not in st_ids[1:]
    assert len(rows) == 4
    for row in rows:
        assert row["ids"][0] == q_id
        assert row["ids"][-1] == d_id
        assert row["decide"] == len(row["ids"]) - 1
        assert row["k"] == 2
        assert all(row["ids"][pos] == c_id for pos in row["opts"])
        assert row["ids"].count(o_id) == 2

    # Exact parity with Kev08bMlxBackend.encode_questions (single source of truth)
    stub_kev = Kev08bMlxBackend.__new__(Kev08bMlxBackend)
    stub_kev.tok = tok
    stub_kev.special_ids = sp_ids
    stub_kev.max_state = 1024
    stub_kev.max_branch = 2048
    assert stub_kev.encode_questions(state, questions) == (st_ids, rows)

    # Truncation bounds on max_state_tokens and max_branch_tokens
    long_state = " ".join(["word"] * 40)
    st_trunc, rows_trunc = encode_kev_questions(
        tok,
        long_state,
        [{"type": "choice", "instructions": "word word word", "criteria": ["word word", "word word"]}],
        max_state_tokens=10,
        max_branch_tokens=16,
    )
    assert len(st_trunc) == 10
    assert len(st_trunc) + len(rows_trunc[0]["ids"]) <= 16
    assert rows_trunc[0]["ids"][-1] == d_id
    assert all(0 <= pos <= rows_trunc[0]["decide"] - 1 for pos in rows_trunc[0]["opts"])

    # Invalid inputs raise ValueError cleanly
    with pytest.raises(ValueError, match="special_ids"):
        encode_kev_questions(tok, state, questions, special_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="max_state_tokens"):
        encode_kev_questions(tok, state, questions, max_state_tokens=0)
    with pytest.raises(ValueError, match="max_branch_tokens"):
        encode_kev_questions(tok, state, questions, max_branch_tokens=2)


def test_v2_exports_paths_and_stdlib_isolation_cpu() -> None:
    """Verify v2 exports (`Kev4bMlxBackend`, `LayaMlxGpuBackend`, `LayaHeteroEngine`) & zero stdlib leak."""
    import fast_decision_models as fdm
    from fast_decision_models import paths as paths_mod

    src_dir = _repo_root() / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(src_dir)!r}); "
        "import fast_decision_models, fast_decision_models.paths, fast_decision_models.scorers, "
        "fast_decision_models.laya_ane, fast_decision_models.laya_hetero, fast_decision_models.kev_mlx; "
        "banned = {'numpy', 'mlx', 'tokenizers', 'fast_shared', 'huggingface_hub', 'coremltools', 'torch'}; "
        "leaked = sorted(banned.intersection(sys.modules)); "
        "assert not leaked, f'Unexpected non-stdlib imports: {leaked}'; "
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "OK"

    assert paths_mod.DEFAULT_KEV_08B_MLX_REPO == "mikewolfd/kev-0.8b-mlx-w8"
    assert paths_mod.DEFAULT_KEV_4B_MLX_REPO == "AbstractFramework/kev-4b-mlx-4bit"
    assert paths_mod.DEFAULT_KEV_4B_ADAPTER_REPO == "jaredpalmer/kev-4b"
    assert paths_mod.DEFAULT_LAYA_COREML_REPO == "mikewolfd/laya-typed-decisions-421m-coreml-ane"
    assert paths_mod.DEFAULT_LAYA_NMTZ_COREML_REPO == "NMTZ/laya-typed-decisions-421m-coreml-ane"
    assert hasattr(paths_mod, "resolve_kev_4b_dir")

    for sym in ("Kev4bMlxBackend", "KevMlxBackend", "LayaMlxGpuBackend", "LayaHeteroEngine"):
        assert sym in fdm.__all__, f"Missing {sym} in fast_decision_models.__all__"
        assert getattr(fdm, sym) is not None


def test_laya_default_buckets_and_multifunction_builder_cpu(tmp_path: Path) -> None:
    """Verify 6-bucket `S=(64,128,192,256,384,512)` defaults + two-pass LN, split-head `einsum`, `marker_map` & multifunction."""
    from fast_decision_models import builder as fdm_builder

    assert fdm_builder.DEFAULT_LAYA_SEQ_LENS == (64, 128, 192, 256, 384, 512)

    ct = pytest.importorskip("coremltools")
    from fast_shared.ir_dtype_guard import check_coreml_model

    h, inter, heads, n_layers = 64, 128, 4, 2
    enc_cfg = {
        "hidden_size": h,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "local_attention": 16,
        "norm_eps": 1e-5,
        "layer_types": ["full_attention", "sliding_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    }
    rng = np.random.default_rng(29)
    w_np: dict[str, np.ndarray] = {
        "encoder.embeddings.norm.weight": np.ones((h,), dtype=np.float32),
        "encoder.final_norm.weight": np.ones((h,), dtype=np.float32),
        "scorer.0.weight": np.ones((h,), dtype=np.float32),
        "scorer.0.bias": np.zeros((h,), dtype=np.float32),
        "scorer.1.weight": (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32),
        "scorer.1.bias": np.zeros((h,), dtype=np.float32),
        "scorer.3.weight": (rng.normal(0.0, 0.02, size=(1, h))).astype(np.float32),
        "scorer.3.bias": np.zeros((1,), dtype=np.float32),
    }
    for i in range(n_layers):
        px = f"encoder.layers.{i}."
        if i > 0:
            w_np[px + "attn_norm.weight"] = np.ones((h,), dtype=np.float32)
        w_np[px + "mlp_norm.weight"] = np.ones((h,), dtype=np.float32)
        w_np[px + "attn.Wqkv.weight"] = (rng.normal(0.0, 0.02, size=(3 * h, h))).astype(np.float32)
        w_np[px + "attn.Wo.weight"] = (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32)
        w_np[px + "mlp.Wi.weight"] = (rng.normal(0.0, 0.02, size=(2 * inter, h))).astype(np.float32)
        w_np[px + "mlp.Wo.weight"] = (rng.normal(0.0, 0.02, size=(h, inter))).astype(np.float32)
    for hi in range(2):
        hp = f"head.layers.{hi}."
        w_np[hp + "norm1.weight"] = np.ones((h,), dtype=np.float32)
        w_np[hp + "norm1.bias"] = np.zeros((h,), dtype=np.float32)
        w_np[hp + "norm2.weight"] = np.ones((h,), dtype=np.float32)
        w_np[hp + "norm2.bias"] = np.zeros((h,), dtype=np.float32)
        w_np[hp + "self_attn.in_proj_weight"] = (rng.normal(0.0, 0.02, size=(3 * h, h))).astype(np.float32)
        w_np[hp + "self_attn.in_proj_bias"] = np.zeros((3 * h,), dtype=np.float32)
        w_np[hp + "self_attn.out_proj.weight"] = (rng.normal(0.0, 0.02, size=(h, h))).astype(np.float32)
        w_np[hp + "self_attn.out_proj.bias"] = np.zeros((h,), dtype=np.float32)
        w_np[hp + "linear1.weight"] = (rng.normal(0.0, 0.02, size=(inter, h))).astype(np.float32)
        w_np[hp + "linear1.bias"] = np.zeros((inter,), dtype=np.float32)
        w_np[hp + "linear2.weight"] = (rng.normal(0.0, 0.02, size=(h, inter))).astype(np.float32)
        w_np[hp + "linear2.bias"] = np.zeros((h,), dtype=np.float32)

    pkg_64 = fdm_builder._build_laya_seq_mlpackage(64, w_np, enc_cfg, tmp_path / "pkgs", guard_ir=True)
    spec = ct.utils.load_spec(str(pkg_64))
    fn = spec.mlProgram.functions["main"]
    in_shapes = {i.name: [d.constant.size for d in i.type.tensorType.dimensions] for i in fn.inputs}
    assert in_shapes["marker_map"] == [1, 64, 1, 32]
    blk = fn.block_specializations[fn.opset]
    op_types = [op.type for op in blk.operations]
    assert "einsum" in op_types
    assert "reduce_mean" in op_types
    assert "rsqrt" in op_types
    assert "layer_norm" not in op_types  # Strips CoreML fuse_layernorm_or_instancenorm so two-pass FP16 LN stays intact!
    assert "matmul" not in op_types  # Split-head einsum eliminates all 4D reshape+matmul ops!
    for op in blk.operations:
        if op.type == "gelu":
            mode_arg = op.inputs["mode"].arguments[0]
            # Mode is either immediate "EXACT" or a const bound to "EXACT"
            if mode_arg.value.WhichOneof("value") == "immediateValue":
                assert list(mode_arg.value.immediateValue.tensor.strings.values) == ["EXACT"]

    pkg_128 = fdm_builder._build_laya_seq_mlpackage(128, w_np, enc_cfg, tmp_path / "pkgs", guard_ir=True)
    mf_pkg = fdm_builder._save_laya_multifunction_mlpackage(
        {64: pkg_64, 128: pkg_128},
        tmp_path / "laya_typed_w8.mlpackage",
        default_seq_len=128,
    )
    mf_spec = ct.utils.load_spec(str(mf_pkg))
    fn_names = sorted(mf_spec.mlProgram.functions.keys())
    assert fn_names == ["s128", "s64"]


def test_laya_hetero_work_stealing_and_ane_sidecar_cpu() -> None:
    """Verify `LayaHeteroEngine` keeps `0` GPU ops in `mode='ane'` and steals long->GPU / short->ANE in `mode='hetero'`."""
    from fast_decision_models.laya_hetero import LayaHeteroEngine

    gpu_seen_lens: list[list[int]] = []
    ane_seen_lens: list[list[int]] = []

    class _FakeAne:
        max_len = 512
        head_max_len = 192
        pad_id = 0

        def encode_item(self, state, qdef, *, max_len=None, on_truncate="warn"):
            n = int(state)
            return {"ids": [1] * n, "markers": [1, 2], "qtype": 2, "k": 2, "truncated": False, "original_state_tokens": n}

        def score_choice_batch(self, items, *, temperature_override=None):
            ane_seen_lens.append([len(it["ids"]) for it in items])
            return [[0.25, 0.75] for _ in items]

        def score_noul_batch(self, items):
            ane_seen_lens.append([len(it["ids"]) for it in items])
            return [0.75 for _ in items]

        def close(self):
            pass

    class _FakeGpu:
        def score_choice_batch(self, items, *, temperature_override=None):
            gpu_seen_lens.append([len(it["ids"]) for it in items])
            return [[0.25, 0.75] for _ in items]

        def score_noul_batch(self, items):
            gpu_seen_lens.append([len(it["ids"]) for it in items])
            return [0.75 for _ in items]

        def close(self):
            pass

    # 1. mode="ane" (LLM sidecar triage) must NOT instantiate GPU backend (0 GPU bytes/ops)
    ane_only = LayaHeteroEngine(mode="ane", ane_backend=_FakeAne(), gpu_backend=None)
    assert ane_only.gpu_backend is None
    items_short = [{"ids": [1] * 80, "markers": [1, 2], "qtype": 2, "k": 2}]
    assert ane_only.score_noul_batch(items_short) == [0.75]
    assert len(gpu_seen_lens) == 0 and len(ane_seen_lens) == 1
    ane_only.close()

    # 2. mode="hetero" (standalone backfill) steals longest >=256 from head onto GPU and shortest <=256 from tail onto ANE
    ane_seen_lens.clear()
    gpu_seen_lens.clear()
    hetero = LayaHeteroEngine(
        mode="hetero",
        profile="backfill",
        ane_backend=_FakeAne(),
        gpu_backend=_FakeGpu(),
        gpu_batch_size=4,
        ane_batch_size=4,
    )
    mixed_items = [
        {"ids": [1] * ln, "markers": [1, 2], "qtype": 0, "k": 3}
        for ln in (64, 480, 96, 420, 128, 384, 160, 320, 192, 500, 72, 460)
    ]
    probs = hetero.score_choice_batch(mixed_items)
    assert len(probs) == len(mixed_items)
    assert all(p == [0.25, 0.75] for p in probs)
    assert len(gpu_seen_lens) >= 1 and len(ane_seen_lens) >= 1
    # GPU starts at head (max length 500), ANE starts at tail (min length 64)
    assert max(max(b) for b in gpu_seen_lens) == 500
    assert min(min(b) for b in ane_seen_lens) == 64
    hetero.close()

    # 3. Verify _build_laya_packed_batch_tensors builds 2D (B, S) position_ids (never 1D B*S!) & BF16 -1e4 masks
    from fast_decision_models.laya_hetero import _build_laya_packed_batch_tensors, _pack_laya_segments

    class _CpuMaskMx:
        int32 = np.int32
        bfloat16 = np.dtype("uint16")  # distinct sentinel dtype verifying mask uses mx.bfloat16, never float32!

        @staticmethod
        def array(x, dtype=None):
            if dtype is _CpuMaskMx.bfloat16:
                return np.asarray(x, dtype=np.float16)
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def abs(x):
            return np.abs(x)

        @staticmethod
        def where(cond, a, b):
            return np.where(cond, a, b)

    sample_encs = [
        {"ids": [10, 11, 12, 13, 14], "markers": [1, 3], "qtype": 0, "k": 2},
        {"ids": [20, 21, 22, 23], "markers": [1, 2], "qtype": 2, "k": 2},
        {"ids": [30, 31, 32], "markers": [1], "qtype": 1, "k": 1},
    ]
    lanes = _pack_laya_segments(sample_encs, [0, 1, 2], lane_cap=16)
    assert len(lanes) == 1 and len(lanes[0]) == 3
    ids_2d, pos_2d, qt_2d, full_m, swa_m, m_meta = _build_laya_packed_batch_tensors(
        _CpuMaskMx, np, sample_encs, lanes, pad_id=0, local_half_win=2
    )
    assert pos_2d.ndim == 2 and pos_2d.shape == (1, 32)
    # Segment 0 (len 5), Segment 1 (len 4), Segment 2 (len 3) each reset position_ids to 0!
    assert pos_2d[0, :12].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2]
    assert full_m.shape == (1, 1, 32, 32) and full_m.dtype == np.float16
    assert swa_m.shape == (1, 1, 32, 32) and swa_m.dtype == np.float16
    # Cross-segment attention (pos 0 in seg0 -> pos 5 in seg1) is masked (-10000.0), zero cross-doc leakage!
    assert float(full_m[0, 0, 0, 5]) == -10000.0
    assert float(full_m[0, 0, 0, 4]) == 0.0
    # SWA window (half_win=2): within seg0, pos 0 -> pos 4 (|0-4|=4 > 2) is masked in swa_m!
    assert float(swa_m[0, 0, 0, 4]) == -10000.0
    assert float(swa_m[0, 0, 0, 2]) == 0.0


def test_kev_horizontal_fusion_mixed_4b_quant_and_one_pass_gpu(tmp_path: Path) -> None:
    """Verify horizontal linear fusion (`9 -> 3`), mixed `4b/8b g32` Kev-4B loader, `gated_delta_update`, & 1-pass GPU head."""
    from fast_decision_models._common import acquire_hw_lock
    import json
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import Tokenizer, models, pre_tokenizers
    from fast_decision_models.kev_mlx import KEV_SPECIAL_TOKENS, Kev4bMlxBackend, _build_qwen35_backbone

    with acquire_hw_lock("test_kev_horizontal_fusion_4b_gpu"):
        # Build a miniature Kev-4B-topology checkpoint (`Hk != Hv`: Hk=2, Hv=4, Dk=128, Dv=128, g32 mixed 4-bit + 8-bit GDN!)
        hidden_size = 256
        inter_size = 512
        vocab_size = 64
        text_cfg = {
            "hidden_size": hidden_size,
            "intermediate_size": inter_size,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": vocab_size,
            "eos_token_id": 1,
            "full_attention_interval": 4,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        }
        mx.random.seed(7)
        bf16_weights: dict[str, mx.array] = {
            "model.embed_tokens.weight": (mx.random.normal((vocab_size, hidden_size)) * 0.02).astype(mx.bfloat16),
            "model.norm.weight": mx.ones((hidden_size,), dtype=mx.bfloat16),
        }
        key_dim = 2 * 128
        val_dim = 4 * 128
        conv_dim = 2 * key_dim + val_dim
        for li in range(4):
            px = f"model.layers.{li}."
            bf16_weights[px + "input_layernorm.weight"] = mx.ones((hidden_size,), dtype=mx.bfloat16)
            bf16_weights[px + "post_attention_layernorm.weight"] = mx.ones((hidden_size,), dtype=mx.bfloat16)
            bf16_weights[px + "mlp.gate_proj.weight"] = (mx.random.normal((inter_size, hidden_size)) * 0.02).astype(mx.bfloat16)
            bf16_weights[px + "mlp.up_proj.weight"] = (mx.random.normal((inter_size, hidden_size)) * 0.02).astype(mx.bfloat16)
            bf16_weights[px + "mlp.down_proj.weight"] = (mx.random.normal((hidden_size, inter_size)) * 0.02).astype(mx.bfloat16)
            if (li + 1) % 4 != 0:
                bf16_weights[px + "linear_attn.conv1d.weight"] = (mx.random.normal((conv_dim, 4, 1)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "linear_attn.in_proj_qkv.weight"] = (mx.random.normal((conv_dim, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "linear_attn.in_proj_z.weight"] = (mx.random.normal((val_dim, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "linear_attn.in_proj_b.weight"] = (mx.random.normal((4, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "linear_attn.in_proj_a.weight"] = (mx.random.normal((4, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "linear_attn.dt_bias"] = mx.ones((4,), dtype=mx.bfloat16)
                bf16_weights[px + "linear_attn.A_log"] = mx.zeros((4,), dtype=mx.bfloat16)
                bf16_weights[px + "linear_attn.norm.weight"] = mx.ones((128,), dtype=mx.float32)  # F32 like HF release!
                bf16_weights[px + "linear_attn.out_proj.weight"] = (mx.random.normal((hidden_size, val_dim)) * 0.02).astype(mx.bfloat16)
            else:
                bf16_weights[px + "self_attn.q_proj.weight"] = (mx.random.normal((4 * 64 * 2, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "self_attn.k_proj.weight"] = (mx.random.normal((2 * 64, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "self_attn.v_proj.weight"] = (mx.random.normal((2 * 64, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "self_attn.o_proj.weight"] = (mx.random.normal((4 * 64, hidden_size)) * 0.02).astype(mx.bfloat16)
                bf16_weights[px + "self_attn.q_norm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
                bf16_weights[px + "self_attn.k_norm.weight"] = mx.ones((64,), dtype=mx.bfloat16)

        # Save in AbstractFramework Kev-4B mixed 4-bit g32 + 8-bit g32 GDN format (`language_model.model.*`)
        af_dir = tmp_path / "kev_4b_af"
        af_dir.mkdir(parents=True)
        af_tensors: dict[str, mx.array] = {}
        quant_dict: dict[str, Any] = {"bits": 4, "group_size": 32, "mode": "affine"}
        for k, v in bf16_weights.items():
            af_k = "language_model." + k
            if k.endswith(".weight") and v.ndim == 2:
                stem = af_k[: -len(".weight")]
                bits = 8 if ".linear_attn." in k else 4
                wq, sc, bi = mx.quantize(v, group_size=32, bits=bits)
                af_tensors[stem + ".weight"] = wq
                af_tensors[stem + ".scales"] = sc
                af_tensors[stem + ".biases"] = bi
                quant_dict[stem] = {"bits": bits, "group_size": 32}
            else:
                af_tensors[af_k] = v
        mx.eval(af_tensors)
        mx.save_safetensors(str(af_dir / "model.safetensors"), af_tensors)

        head_tensors = {
            "q.weight": (mx.random.normal((32, hidden_size)) * 0.05).astype(mx.float32),
            "q.bias": mx.zeros((32,), dtype=mx.float32),
            "k.weight": (mx.random.normal((32, hidden_size)) * 0.05).astype(mx.float32),
            "k.bias": mx.zeros((32,), dtype=mx.float32),
        }
        mx.eval(head_tensors)
        mx.save_safetensors(str(af_dir / "head.safetensors"), head_tensors)
        (af_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "text_config": text_cfg, "quantization": quant_dict}),
            encoding="utf-8",
        )
        (af_dir / "decision.json").write_text(
            json.dumps({"temperature": 2.1, "head_file": "head.safetensors", "delimiters": list(KEV_SPECIAL_TOKENS)}),
            encoding="utf-8",
        )
        vocab = {f"w{i}": i for i in range(50)}
        vocab["<unk>"] = 0
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.add_special_tokens(list(KEV_SPECIAL_TOKENS))
        tok.save(str(af_dir / "tokenizer.json"))

        with Kev4bMlxBackend(af_dir, batch_docs=4, prefix_cache_mb=None) as kev4:
            # Verify horizontal fusion (`in_proj_all`, `qkv_proj`, `gate_up_proj`) & preserved 8-bit GDN / 4-bit MLP!
            gdn_blk = kev4.backbone.layers[0].linear_attn
            fa_blk = kev4.backbone.layers[3].self_attn
            mlp_blk = kev4.backbone.layers[0].mlp
            assert hasattr(gdn_blk, "in_proj_all") and isinstance(gdn_blk.in_proj_all, nn.QuantizedLinear)
            assert gdn_blk.in_proj_all.bits == 8 and gdn_blk.in_proj_all.group_size == 32
            assert hasattr(fa_blk, "qkv_proj") and isinstance(fa_blk.qkv_proj, nn.QuantizedLinear)
            assert fa_blk.qkv_proj.bits == 4 and fa_blk.qkv_proj.group_size == 32
            assert hasattr(mlp_blk, "gate_up_proj") and isinstance(mlp_blk.gate_up_proj, nn.QuantizedLinear)
            assert mlp_blk.gate_up_proj.bits == 4 and mlp_blk.gate_up_proj.group_size == 32

            # Verify 1-pass single-question B=1 vs B=3 (exact & padded) parity!
            qdef = {"type": "choice", "instructions": "w2 w3", "criteria": ["w4", "w5 w6", "w7"]}
            items = [("w10 w11 w12", [qdef]), ("w13 w14 w15 w16 w17", [qdef]), ("w18 w19", [qdef])]
            b1_outs = [kev4.score_multi_states([it], batch_docs=1)[0][0] for it in items]
            b3_outs = [r[0] for r in kev4.score_multi_states(items, batch_docs=4)]
            for (p1, z1), (p3, z3) in zip(b1_outs, b3_outs):
                np.testing.assert_allclose(p1, p3, atol=5e-3)
                np.testing.assert_allclose(z1, z3, atol=2e-2)

            # Also verify `cache=None` 1-pass matches 2-pass multi-branch on identical question!
            two_q = kev4.score_multi_states([("w10 w11 w12", [qdef, qdef])], batch_docs=1)[0]
            np.testing.assert_allclose(b1_outs[0][0], two_q[0][0], atol=5e-3)
            np.testing.assert_allclose(two_q[0][0], two_q[1][0], atol=1e-5)


def test_laya_mlx_gpu_packed512_and_marker_scorer_parity_gpu(tmp_path: Path) -> None:
    """Verify `LayaMlxGpuBackend` `Packed-512` segment-isolated packing matches unpacked single-item scoring."""
    from fast_decision_models._common import acquire_hw_lock
    import json
    import mlx.core as mx
    from tokenizers import Tokenizer, models, pre_tokenizers
    from fast_decision_models.laya_hetero import LayaMlxGpuBackend

    with acquire_hw_lock("test_laya_mlx_gpu_packed512"):
        laya_src = tmp_path / "tiny_laya"
        (laya_src / "encoder").mkdir(parents=True)
        (laya_src / "tokenizer").mkdir(parents=True)
        h, inter, heads, n_layers, vocab_sz = 128, 256, 4, 3, 64
        enc_cfg = {
            "hidden_size": h,
            "intermediate_size": inter,
            "num_hidden_layers": n_layers,
            "num_attention_heads": heads,
            "local_attention": 16,
            "norm_eps": 1e-5,
            "vocab_size": vocab_sz,
            "layer_types": ["full_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0},
                "sliding_attention": {"rope_theta": 10000.0},
            },
        }
        (laya_src / "encoder" / "config.json").write_text(json.dumps(enc_cfg), encoding="utf-8")
        (laya_src / "rl_agent_config.json").write_text(
            json.dumps({"head_max_len": 128, "temperature": [1.0, 1.0, 1.0]}),
            encoding="utf-8",
        )
        vocab = {f"tok{i}": i for i in range(1, vocab_sz - 4)}
        vocab.update({"[UNK]": 0, "[CLS]": 60, "[SEP]": 61, "[PAD]": 62, "[MASK]": 63})
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(str(laya_src / "tokenizer" / "tokenizer.json"))

        mx.random.seed(31)
        tensors: dict[str, mx.array] = {
            "encoder.embeddings.tok_embeddings.weight": (mx.random.normal((vocab_sz, h)) * 0.05).astype(mx.float16),
            "encoder.embeddings.norm.weight": mx.ones((h,), dtype=mx.float16),
            "encoder.final_norm.weight": mx.ones((h,), dtype=mx.float16),
            "type_emb.weight": (mx.random.normal((3, h)) * 0.05).astype(mx.float16),
            "scorer.0.weight": mx.ones((h,), dtype=mx.float16),
            "scorer.0.bias": mx.zeros((h,), dtype=mx.float16),
            "scorer.1.weight": (mx.random.normal((h, h)) * 0.05).astype(mx.float16),
            "scorer.1.bias": mx.zeros((h,), dtype=mx.float16),
            "scorer.3.weight": (mx.random.normal((1, h)) * 0.05).astype(mx.float16),
            "scorer.3.bias": mx.zeros((1,), dtype=mx.float16),
        }
        for i in range(n_layers):
            px = f"encoder.layers.{i}."
            if i > 0:
                tensors[px + "attn_norm.weight"] = mx.ones((h,), dtype=mx.float16)
            tensors[px + "mlp_norm.weight"] = mx.ones((h,), dtype=mx.float16)
            tensors[px + "attn.Wqkv.weight"] = (mx.random.normal((3 * h, h)) * 0.04).astype(mx.float16)
            tensors[px + "attn.Wo.weight"] = (mx.random.normal((h, h)) * 0.04).astype(mx.float16)
            tensors[px + "mlp.Wi.weight"] = (mx.random.normal((2 * inter, h)) * 0.04).astype(mx.float16)
            tensors[px + "mlp.Wo.weight"] = (mx.random.normal((h, inter)) * 0.04).astype(mx.float16)
        for hi in range(2):
            hp = f"head.layers.{hi}."
            tensors[hp + "norm1.weight"] = mx.ones((h,), dtype=mx.float16)
            tensors[hp + "norm1.bias"] = mx.zeros((h,), dtype=mx.float16)
            tensors[hp + "norm2.weight"] = mx.ones((h,), dtype=mx.float16)
            tensors[hp + "norm2.bias"] = mx.zeros((h,), dtype=mx.float16)
            tensors[hp + "self_attn.in_proj_weight"] = (mx.random.normal((3 * h, h)) * 0.04).astype(mx.float16)
            tensors[hp + "self_attn.in_proj_bias"] = mx.zeros((3 * h,), dtype=mx.float16)
            tensors[hp + "self_attn.out_proj.weight"] = (mx.random.normal((h, h)) * 0.04).astype(mx.float16)
            tensors[hp + "self_attn.out_proj.bias"] = mx.zeros((h,), dtype=mx.float16)
            tensors[hp + "linear1.weight"] = (mx.random.normal((inter * 2, h)) * 0.04).astype(mx.float16)
            tensors[hp + "linear1.bias"] = mx.zeros((inter * 2,), dtype=mx.float16)
            tensors[hp + "linear2.weight"] = (mx.random.normal((h, inter * 2)) * 0.04).astype(mx.float16)
            tensors[hp + "linear2.bias"] = mx.zeros((h,), dtype=mx.float16)
        mx.eval(tensors)
        mx.save_safetensors(str(laya_src / "model.safetensors"), tensors)

        with LayaMlxGpuBackend(laya_src, batch_size=4, pack_seq_len=512) as gpu_eng:
            encs = [
                gpu_eng.encode_item(
                    " ".join(f"tok{(i + j) % 40 + 1}" for j in range(12 + i * 7)),
                    {"type": "choice" if i % 2 == 0 else "noul", "instructions": "tok2 tok3", "criteria": ["tok4", "tok5", "tok6"] if i % 2 == 0 else {"false": "tok7", "true": "tok8"}},
                    on_truncate="ignore",
                )
                for i in range(6)
            ]
            single_probs = [gpu_eng.score_choice_batch([e])[0] for e in encs]
            packed_probs = gpu_eng.score_choice_batch(encs)
            for s_p, pk_p in zip(single_probs, packed_probs):
                np.testing.assert_allclose(s_p, pk_p, atol=8e-3)


def test_offline_local_files_only_bad_laya_src_and_noul_qtype_parity_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `local_files_only` / `HF_HUB_OFFLINE=1`, explicit bad `laya_src_dir` error, and `LayaAneBackend` `qtype=2` parity."""
    import ctypes
    import fast_shared.hf as shared_hf
    from fast_decision_models.kev_mlx import Kev08bMlxBackend
    from fast_decision_models.laya_ane import LayaAneBackend
    from fast_decision_models.laya_hetero import LayaHeteroEngine, LayaMlxGpuBackend

    hf_calls: list[tuple[str, bool]] = []

    def _fake_resolve_hf(_explicit: object, repo_id: str, *, local_files_only: bool = False, **_kw: object) -> Path:
        hf_calls.append((repo_id, bool(local_files_only)))
        raise FileNotFoundError(f"uncached repo {repo_id} (local_files_only={local_files_only})")

    monkeypatch.setattr(shared_hf, "resolve_hf_model_dir", _fake_resolve_hf)

    # 1. Explicit non-existent or non-Laya directory passed to LayaMlxGpuBackend MUST raise FileNotFoundError without HF fallback
    with pytest.raises(FileNotFoundError):
        LayaMlxGpuBackend(tmp_path / "does_not_exist")
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        LayaMlxGpuBackend(empty_dir)
    assert hf_calls == []

    # 2. ANE bundle dir (`laya_typed_emb_f16.npz` present) falls back to DEFAULT_LAYA_REPO respecting `local_files_only=True`
    ane_dir = tmp_path / "ane_bundle_dir"
    ane_dir.mkdir()
    (ane_dir / "laya_typed_emb_f16.npz").write_bytes(b"stub")
    with pytest.raises(FileNotFoundError):
        LayaMlxGpuBackend(ane_dir, local_files_only=True)
    assert hf_calls == [("convaiinnovations/laya-typed-decisions", True)]

    # 3. `HF_HUB_OFFLINE=1` honored across LayaAneBackend, LayaHeteroEngine, and Kev08bMlxBackend (never calls local_files_only=False)
    hf_calls.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("FAST_DECISION_MODELS_KEV_DIR", str(tmp_path / "artifacts" / "kev_08b"))
    fake_dylib = tmp_path / "fake.dylib"
    fake_dylib.write_bytes(b"stub")
    monkeypatch.setattr(LayaAneBackend, "_bind_native_lib", lambda self: None)
    with pytest.raises(FileNotFoundError):
        LayaAneBackend(tmp_path / "no_local_ane", fake_dylib, auto_prebuilt=True)
    with pytest.raises(FileNotFoundError):
        LayaHeteroEngine(tmp_path / "no_local_ane", fake_dylib, mode="ane")
    with pytest.raises(FileNotFoundError):
        Kev08bMlxBackend(None, auto_prebuilt=True, local_files_only=False)
    assert len(hf_calls) >= 3
    assert all(is_local is True for _, is_local in hf_calls)

    # 4. `LayaAneBackend.score_noul_batch` enforces `qtype=2` even when item was encoded with `qtype != 2`
    seen_qtypes: list[int] = []

    class _FakeLib:
        def laya_ane_max_markers(self, _h: object) -> int:
            return 80

        def laya_ane_exec_markers_batch(
            self,
            _h: object,
            n_items: int,
            _ids_ptr: int,
            _vl_ptr: int,
            qt_ptr: int,
            _mp_ptr: int,
            _mc_ptr: int,
            stride: int,
            out_ptr: int,
            _async_flag: int,
        ) -> float:
            qts = (ctypes.c_int32 * n_items).from_address(qt_ptr)
            seen_qtypes.extend(int(qts[i]) for i in range(n_items))
            out_arr = (ctypes.c_float * (n_items * stride)).from_address(out_ptr)
            for i in range(n_items):
                out_arr[i * stride + 0] = -0.5
                out_arr[i * stride + 1] = 1.5
            return 0.001

    import threading

    ane_mock = LayaAneBackend.__new__(LayaAneBackend)
    ane_mock._np = np
    ane_mock._exec_lock = threading.Lock()
    ane_mock._lib = _FakeLib()
    ane_mock._has_last_error = False
    ane_mock.pad_id = 0
    ane_mock.seq_buckets = (64,)
    ane_mock._handles = {64: 1}
    ane_mock.temperature = [1.0, 1.0, 1.0]
    ane_mock.temperature_by_options = {}
    res_noul = ane_mock.score_noul_batch([{"ids": [1, 2, 3, 4], "markers": [1, 2], "qtype": 0}])
    assert len(res_noul) == 1 and res_noul[0] > 0.8
    assert seen_qtypes == [2]


def test_kev_mixed_prefix_length_cold_warm_zero_diff_gpu(tmp_path: Path) -> None:
    """Verify mixed-prefix-length `b8` (`64`-tok and `128`-tok prefixes) achieves strict `0.0` cold vs warm diff."""
    from fast_decision_models._common import acquire_hw_lock
    import json
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, models, pre_tokenizers
    from fast_decision_models.kev_mlx import KEV_SPECIAL_TOKENS, KevMlxBackend
    from fast_decision_models.prefix_tree import PrefixTree

    with acquire_hw_lock("test_kev_mixed_prefix_zero_diff"):
        snap = tmp_path / "tiny_kev_mixed_p"
        snap.mkdir(parents=True)
        vocab = {f"w{i}": i + 1 for i in range(48)}
        vocab["<pad>"] = 0
        for idx_sp, sp in enumerate(KEV_SPECIAL_TOKENS, start=50):
            vocab[sp] = idx_sp
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(str(snap / "tokenizer.json"))

        cfg = {
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "full_attention_interval": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "rms_norm_eps": 1e-6,
                "vocab_size": 64,
                "eos_token_id": 0,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "rope_parameters": {"partial_rotary_factor": 0.5, "rope_theta": 10000.0},
            },
            "quantization": {"bits": 8, "group_size": 32},
        }
        (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (snap / "decision.json").write_text(
            json.dumps({"temperature": 0.25, "special_tokens": list(KEV_SPECIAL_TOKENS)}),
            encoding="utf-8",
        )

        mx.random.seed(19)
        raw_t: dict[str, mx.array] = {"model.norm.weight": mx.ones((64,), dtype=mx.bfloat16)}
        emb = nn.Embedding(64, 64)
        nn.quantize(emb, bits=8, group_size=32)
        for k, v in tree_flatten(emb.parameters()):
            raw_t[f"model.embed_tokens.{k}"] = v

        def _add_qlin(prefix: str, in_d: int, out_d: int) -> None:
            lin = nn.Linear(in_d, out_d, bias=False)
            nn.quantize(lin, bits=8, group_size=32)
            for k, v in tree_flatten(lin.parameters()):
                raw_t[f"{prefix}.{k}"] = v

        raw_t["model.layers.0.input_layernorm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
        raw_t["model.layers.0.post_attention_layernorm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
        _add_qlin("model.layers.0.linear_attn.in_proj_qkv", 64, 192)
        _add_qlin("model.layers.0.linear_attn.in_proj_z", 64, 64)
        raw_t["model.layers.0.linear_attn.in_proj_b.weight"] = (mx.random.normal((2, 64)) * 0.05).astype(mx.bfloat16)
        raw_t["model.layers.0.linear_attn.in_proj_a.weight"] = (mx.random.normal((2, 64)) * 0.05).astype(mx.bfloat16)
        raw_t["model.layers.0.linear_attn.conv1d.weight"] = (mx.random.normal((192, 1, 4)) * 0.05).astype(mx.bfloat16)
        raw_t["model.layers.0.linear_attn.dt_bias"] = mx.zeros((2,), dtype=mx.bfloat16)
        raw_t["model.layers.0.linear_attn.A_log"] = mx.zeros((2,), dtype=mx.float32)
        raw_t["model.layers.0.linear_attn.norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        _add_qlin("model.layers.0.linear_attn.out_proj", 64, 64)
        _add_qlin("model.layers.0.mlp.gate_proj", 64, 128)
        _add_qlin("model.layers.0.mlp.up_proj", 64, 128)
        _add_qlin("model.layers.0.mlp.down_proj", 128, 64)

        raw_t["model.layers.1.input_layernorm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
        raw_t["model.layers.1.post_attention_layernorm.weight"] = mx.ones((64,), dtype=mx.bfloat16)
        _add_qlin("model.layers.1.self_attn.q_proj", 64, 128)
        _add_qlin("model.layers.1.self_attn.k_proj", 64, 32)
        _add_qlin("model.layers.1.self_attn.v_proj", 64, 32)
        _add_qlin("model.layers.1.self_attn.o_proj", 64, 64)
        raw_t["model.layers.1.self_attn.q_norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        raw_t["model.layers.1.self_attn.k_norm.weight"] = mx.ones((32,), dtype=mx.bfloat16)
        _add_qlin("model.layers.1.mlp.gate_proj", 64, 128)
        _add_qlin("model.layers.1.mlp.up_proj", 64, 128)
        _add_qlin("model.layers.1.mlp.down_proj", 128, 64)

        mx.eval(raw_t)
        mx.save_safetensors(str(snap / "model.safetensors"), raw_t)
        head_t = {
            "q.weight": (mx.random.normal((64, 64)) * 0.1).astype(mx.float32),
            "q.bias": mx.zeros((64,), dtype=mx.float32),
            "k.weight": (mx.random.normal((64, 64)) * 0.1).astype(mx.float32),
            "k.bias": mx.zeros((64,), dtype=mx.float32),
        }
        mx.eval(head_t)
        mx.save_safetensors(str(snap / "head.safetensors"), head_t)

        tree = PrefixTree(chunk_tokens=64, byte_budget=64 << 20)
        with KevMlxBackend(snap, batch_docs=8, prefix_tree=tree) as eng:
            qdef = {"type": "noul", "instructions": "w1 w2", "criteria": {"false": "w3", "true": "w4"}}
            pref_short = " ".join(f"w{(j % 30) + 1}" for j in range(72))  # aligned_p = 64
            pref_long = " ".join(f"w{(j % 30) + 10}" for j in range(138))  # aligned_p = 128
            pairs = [
                *((f"{pref_short} w{5 + k} w{6 + k}", [qdef]) for k in range(5)),
                *((f"{pref_long} w{15 + k} w{16 + k} w{17 + k}", [qdef]) for k in range(3)),
            ]
            cold = eng.score_multi_states(pairs, batch_docs=8)
            warm = eng.score_multi_states(pairs, batch_docs=8)
            max_prob_diff = max(
                abs(cp - wp)
                for c_doc, w_doc in zip(cold, warm)
                for (c_pr, _), (w_pr, _) in zip(c_doc, w_doc)
                for cp, wp in zip(c_pr, w_pr)
            )
            max_logit_diff = max(
                abs(cl - wl)
                for c_doc, w_doc in zip(cold, warm)
                for (_, c_lg), (_, w_lg) in zip(c_doc, w_doc)
                for cl, wl in zip(c_lg, w_lg)
            )
            assert max_prob_diff == 0.0
            assert max_logit_diff == 0.0


def test_kev_d256_nax_force_fused_bfloat16_guard_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `kev_mlx` M5 NAX `D=256` `force_fused=True` fires for both `mx.bfloat16` and `mx.float16` on CPU."""
    import types
    from fast_decision_models.kev_mlx import _build_qwen35_backbone

    sdpa_calls: list[dict[str, Any]] = []

    class _FakeArr:
        def __init__(self, shape: tuple[int, ...], dtype: str = "bfloat16") -> None:
            self.shape = tuple(int(d) for d in shape)
            self.dtype = dtype

        def astype(self, dtype: str) -> "_FakeArr":
            return _FakeArr(self.shape, dtype)

        def reshape(self, *shape: Any) -> "_FakeArr":
            dims = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
            known = 1
            neg_idx = -1
            out: list[int] = []
            for idx, d in enumerate(dims):
                di = int(d)
                if di == -1:
                    neg_idx = idx
                    out.append(1)
                else:
                    known *= di
                    out.append(di)
            if neg_idx >= 0:
                total = 1
                for s in self.shape:
                    total *= s
                out[neg_idx] = total // known
            return _FakeArr(tuple(out), self.dtype)

        def transpose(self, *axes: int) -> "_FakeArr":
            return _FakeArr(tuple(self.shape[a] for a in axes), self.dtype)

        def __mul__(self, other: Any) -> "_FakeArr":
            return _FakeArr(self.shape, self.dtype)

        def __add__(self, other: Any) -> "_FakeArr":
            return _FakeArr(self.shape, self.dtype)

        def __sub__(self, other: Any) -> "_FakeArr":
            return _FakeArr(self.shape, self.dtype)

    class _FakeFast:
        @staticmethod
        def metal_kernel(**_kw: Any) -> object:
            return object()

        @staticmethod
        def scaled_dot_product_attention(
            q: _FakeArr, k: _FakeArr, v: _FakeArr, *, scale: float, mask: Any = None, **kwargs: Any
        ) -> _FakeArr:
            sdpa_calls.append({"dtype": q.dtype, "shape": q.shape, "force_fused": kwargs.get("force_fused")})
            return _FakeArr(q.shape, q.dtype)

    fake_mx = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        int32="int32",
        fast=_FakeFast(),
        compile=lambda fn, **_kw: fn,
        array=lambda data, dtype="float32": _FakeArr((len(data),), dtype) if isinstance(data, list) else _FakeArr((1,), dtype),
        cos=lambda x: _FakeArr(x.shape, x.dtype),
        sin=lambda x: _FakeArr(x.shape, x.dtype),
        split=lambda x, stops_or_n, axis=-1: (
            [
                _FakeArr(
                    (*x.shape[:axis], x.shape[axis] // int(stops_or_n), *x.shape[axis + 1 :])
                    if axis != -1
                    else (*x.shape[:-1], x.shape[-1] // int(stops_or_n)),
                    x.dtype,
                )
                for _ in range(int(stops_or_n))
            ]
            if isinstance(stops_or_n, int)
            else [
                _FakeArr(
                    (
                        *x.shape[: (axis if axis >= 0 else len(x.shape) + axis)],
                        hi - lo,
                        *x.shape[(axis if axis >= 0 else len(x.shape) + axis) + 1 :],
                    ),
                    x.dtype,
                )
                for lo, hi in zip([0, *stops_or_n], [*stops_or_n, x.shape[axis]])
            ]
        ),
        concatenate=lambda arrs, axis=0: _FakeArr(
            tuple(
                sum(a.shape[i] for a in arrs) if i == (axis if axis >= 0 else len(arrs[0].shape) + axis) else arrs[0].shape[i]
                for i in range(len(arrs[0].shape))
            ),
            arrs[0].dtype,
        ),
        sigmoid=lambda x: _FakeArr(x.shape, x.dtype),
        eval=lambda *_args: None,
    )

    class _FakeModule:
        def eval(self) -> None:
            pass

        def parameters(self) -> dict[str, Any]:
            return {}

    class _FakeLinear(_FakeModule):
        def __init__(self, in_d: int, out_d: int, bias: bool = False) -> None:
            self.out_d = out_d
            self.weight = _FakeArr((out_d, in_d), "bfloat16")

        def __call__(self, x: _FakeArr) -> _FakeArr:
            return _FakeArr((*x.shape[:-1], self.out_d), x.dtype)

    class _FakeNorm(_FakeModule):
        def __init__(self, dims: int, eps: float = 1e-6) -> None:
            self.weight = _FakeArr((dims,), "bfloat16")

        def __call__(self, x: _FakeArr) -> _FakeArr:
            return _FakeArr(x.shape, x.dtype)

    class _FakeRoPE(_FakeModule):
        def __init__(self, dims: int, **_kw: Any) -> None:
            self.dims = dims

        def __call__(self, x: _FakeArr, offset: int = 0) -> _FakeArr:
            return _FakeArr(x.shape, x.dtype)

    class _FakeEmbedding(_FakeModule):
        def __init__(self, vocab: int, dims: int) -> None:
            self.dims = dims
            self.weight = _FakeArr((vocab, dims), "bfloat16")

        def __call__(self, ids: _FakeArr) -> _FakeArr:
            return _FakeArr((*ids.shape, self.dims), "bfloat16")

    fake_nn = types.SimpleNamespace(
        Module=_FakeModule,
        Linear=_FakeLinear,
        RMSNorm=_FakeNorm,
        RoPE=_FakeRoPE,
        Embedding=_FakeEmbedding,
        silu=lambda x: _FakeArr(x.shape, x.dtype),
    )
    fake_mlx_pkg = types.SimpleNamespace(core=fake_mx, nn=fake_nn)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)

    hidden_size, inter_size, head_dim = 256, 256, 256
    text_cfg = {
        "hidden_size": hidden_size,
        "intermediate_size": inter_size,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "full_attention_interval": 1,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
    }
    weights = {
        "model.embed_tokens.weight": _FakeArr((16, hidden_size), "bfloat16"),
        "model.norm.weight": _FakeArr((hidden_size,), "bfloat16"),
        "model.layers.0.input_layernorm.weight": _FakeArr((hidden_size,), "bfloat16"),
        "model.layers.0.post_attention_layernorm.weight": _FakeArr((hidden_size,), "bfloat16"),
        "model.layers.0.self_attn.q_proj.weight": _FakeArr((2 * head_dim, hidden_size), "bfloat16"),
        "model.layers.0.self_attn.k_proj.weight": _FakeArr((head_dim, hidden_size), "bfloat16"),
        "model.layers.0.self_attn.v_proj.weight": _FakeArr((head_dim, hidden_size), "bfloat16"),
        "model.layers.0.self_attn.o_proj.weight": _FakeArr((hidden_size, head_dim), "bfloat16"),
        "model.layers.0.self_attn.q_norm.weight": _FakeArr((head_dim,), "bfloat16"),
        "model.layers.0.self_attn.k_norm.weight": _FakeArr((head_dim,), "bfloat16"),
        "model.layers.0.mlp.gate_proj.weight": _FakeArr((inter_size, hidden_size), "bfloat16"),
        "model.layers.0.mlp.up_proj.weight": _FakeArr((inter_size, hidden_size), "bfloat16"),
        "model.layers.0.mlp.down_proj.weight": _FakeArr((hidden_size, inter_size), "bfloat16"),
    }

    backbone = _build_qwen35_backbone(text_cfg, weights, quant_cfg=None, use_d256_force_fused=True)
    attn = backbone.layers[0].self_attn

    attn(_FakeArr((1, 256, hidden_size), "bfloat16"), None)
    attn(_FakeArr((1, 512, hidden_size), "float16"), None)
    attn(_FakeArr((1, 256, hidden_size), "float32"), None)
    attn(_FakeArr((1, 128, hidden_size), "bfloat16"), None)
    attn(_FakeArr((1, 1024, hidden_size), "bfloat16"), None)

    assert [c["force_fused"] for c in sdpa_calls] == [True, True, None, None, None]


def test_laya_shared_state_pretokenization_cpu() -> None:
    """Verify `encode_state`, `state_ids` reuse, and `encode_items_for_state` produce bit-identical encodings."""
    helper = LayaAneBackend.__new__(LayaAneBackend)
    helper.max_len = 512
    helper.head_max_len = 192
    helper.cls_id = 1
    helper.sep_id = 2
    helper.pad_id = 3
    helper.mask_id = 4
    helper.mask_tok = "[MASK]"

    encode_calls: list[str] = []

    def _tok_ids(text: str) -> list[int]:
        encode_calls.append(text)
        return [100 + (sum(ord(ch) for ch in tok) % 900) for tok in text.split()]

    helper._encode_ids = _tok_ids  # type: ignore[method-assign]

    state_dict = {"status": "alert [MASK] critical", "node": "edge-01", "retries": 3}
    qdefs = [
        {"type": "noul", "instructions": "Check if critical", "criteria": {"false": "no", "true": "yes"}},
        {"type": "choice", "instructions": "Select subsystem", "criteria": ["auth", "net", "disk"]},
        {"type": "score", "instructions": "Severity level", "criteria": ["low", "med", "high"]},
    ]

    # Empty qdefs returns [] with zero tokenization calls
    assert helper.encode_items_for_state(state_dict, []) == []
    assert encode_calls == []

    # Direct encode_state neutralizes [MASK] and matches reference encode_item state tokenization
    st_ids = helper.encode_state(state_dict)
    assert len(encode_calls) == 1
    assert "[MASK]" not in encode_calls[0]
    st_ids_snapshot = list(st_ids)

    ref_items = [helper.encode_item(state_dict, qd) for qd in qdefs]
    encode_calls.clear()

    # Shared-state encode_items_for_state tokenizes state ONLY ONCE across all 3 qdefs
    shared_items = helper.encode_items_for_state(state_dict, qdefs)
    # 1 state encode + per-question head + options encodes (3*(1+2) + 3*(1+3) + 3*(1+3) = 1 + 11 = 12)
    state_encode_hits = sum(1 for s in encode_calls if "edge-01" in s)
    assert state_encode_hits == 1
    assert shared_items == ref_items
    assert st_ids == st_ids_snapshot  # caller list is never mutated

    # Truncation parity at max_len=256 with state_ids passed explicitly
    long_state = " ".join(f"tok{i % 20}" for i in range(400))
    long_st_ids = helper.encode_state(long_state)
    ref_trunc = helper.encode_item(long_state, qdefs[1], max_len=256, on_truncate="ignore")
    reuse_trunc = helper.encode_item(long_state, qdefs[1], max_len=256, on_truncate="ignore", state_ids=long_st_ids)
    assert reuse_trunc == ref_trunc


def test_laya_lazy_multifunction_discovery_and_loading_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify symlink -> `metadata.json` -> `inspect_plan` discovery, lazy bucket loading, and `_h256`/`_h512` properties."""
    import ctypes
    import json
    import threading
    from tokenizers import Tokenizer, models, pre_tokenizers

    def _make_local_laya_dir(root: Path, *, with_mf: bool = True) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer").mkdir(exist_ok=True)
        vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4, "w": 5}
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(str(root / "tokenizer" / "tokenizer.json"))
        (root / "rl_agent_config.json").write_text(
            json.dumps({"head_max_len": 192, "temperature": [1.0, 1.0, 1.0]}), encoding="utf-8"
        )
        (root / "encoder_config.json").write_text(json.dumps({"local_attention": 128}), encoding="utf-8")
        np.savez(
            root / "laya_typed_emb_f16.npz",
            word_f16=np.zeros((32, 16), dtype=np.float16),
            type_f16=np.zeros((3, 16), dtype=np.float16),
        )
        if with_mf:
            (root / "laya_typed_w8.mlmodelc").mkdir(exist_ok=True)
        return root

    fake_dylib = tmp_path / "libfake.dylib"
    fake_dylib.write_bytes(b"stub")

    class _FakeLib:
        def __init__(self, *, load_fail_for: set[int] | None = None, plan_buckets: set[int] | None = None) -> None:
            self.inspect_calls: list[tuple[str, str]] = []
            self.load_calls: list[tuple[str, str, int]] = []
            self.freed: list[int] = []
            self.load_fail_for = load_fail_for or set()
            self.plan_buckets = plan_buckets if plan_buckets is not None else {64, 256, 512}
            self._next_ptr = 100

        def laya_ane_inspect_plan_fn(self, mc_b: bytes, fn_b: bytes, _mode: int, counts_ptr: Any) -> None:
            fn_s = fn_b.decode("utf-8")
            self.inspect_calls.append((mc_b.decode("utf-8"), fn_s))
            s_num = int(fn_s[1:]) if fn_s.startswith("s") and fn_s[1:].isdigit() else -1
            if s_num in self.plan_buckets or fn_s == "main":
                counts_ptr[0] = 1585
                counts_ptr[1] = 0
                counts_ptr[2] = 0
            else:
                counts_ptr[0] = 0
                counts_ptr[1] = 0
                counts_ptr[2] = 0

        def laya_ane_load_fn(
            self, mc_b: bytes, fn_b: bytes, _h: int, s_len: int, *_rest: Any
        ) -> int:
            self.load_calls.append((mc_b.decode("utf-8"), fn_b.decode("utf-8"), int(s_len)))
            if int(s_len) in self.load_fail_for:
                return 0
            self._next_ptr += 1
            return self._next_ptr

        def laya_ane_max_markers(self, _h: int) -> int:
            return 32

        def laya_ane_exec_markers_batch(
            self, _h: int, n_items: int, _ids: int, _vl: int, _qt: int, _mp: int, _mc: int, stride: int, out_ptr: int, _async: int
        ) -> float:
            out_arr = (ctypes.c_float * (n_items * stride)).from_address(out_ptr)
            for i in range(n_items):
                out_arr[i * stride + 0] = -0.2
                out_arr[i * stride + 1] = 0.8
            return 0.001

        def laya_ane_free(self, h_ptr: int) -> None:
            self.freed.append(h_ptr)

    active_lib: list[_FakeLib] = []

    def _bind_stub(self_obj: LayaAneBackend) -> None:
        self_obj._has_last_error = False
        self_obj._lib = active_lib[0]

    monkeypatch.setattr(LayaAneBackend, "_bind_native_lib", _bind_stub)

    # 1. Tier 1: symlinks discover buckets with ZERO inspect_plan_fn and ZERO load_fn on lazy_load=True
    dir_sym = _make_local_laya_dir(tmp_path / "mf_symlink", with_mf=True)
    for s_len in (64, 256, 512):
        os.symlink("laya_typed_w8.mlmodelc", dir_sym / f"laya_typed_s{s_len}_w8.mlmodelc")
    os.symlink("missing_target.mlmodelc", dir_sym / "laya_typed_s128_w8.mlmodelc")  # broken symlink skipped

    lib1 = _FakeLib()
    active_lib[:] = [lib1]
    be = LayaAneBackend(dir_sym, fake_dylib, lazy_load=True)
    assert lib1.inspect_calls == []
    assert lib1.load_calls == []
    assert be.seq_buckets == (64, 256, 512)
    assert be._handles == {}
    assert be._h256 is None and be._h512 is None
    assert be._load_dims == (16, 64, 32)
    assert be._bucket_sources[256] == ((dir_sym / "laya_typed_w8.mlmodelc").resolve(), "s256")

    # inspect_compute_plans still inspects all discovered buckets via _bucket_sources without loading handles
    plans = be.inspect_compute_plans()
    assert set(plans.keys()) == {"s64", "s256", "s512"}
    assert len(lib1.inspect_calls) == 3
    assert be._handles == {}

    # Scoring a 100-token item loads ONLY bucket 256
    out_256 = be.score_choice_batch([{"ids": [5] * 100, "markers": [1, 2], "qtype": 0}])
    assert len(out_256) == 1 and len(out_256[0]) == 2
    assert len(lib1.load_calls) == 1 and lib1.load_calls[0][1:] == ("s256", 256)
    assert be._h256 is not None and be._h512 is None

    # Repeat 100-token call reuses cached handle; 400-token call lazily loads 512
    be.score_choice_batch([{"ids": [5] * 120, "markers": [1, 2], "qtype": 0}])
    assert len(lib1.load_calls) == 1
    out_512 = be.score_noul_batch([{"ids": [5] * 400, "markers": [1, 2], "qtype": 2}])
    assert len(out_512) == 1 and out_512[0] > 0.5
    assert len(lib1.load_calls) == 2 and lib1.load_calls[1][1:] == ("s512", 512)
    assert be._h512 is not None

    # Concurrent _ensure_bucket_handle on unloaded bucket 64 invokes laya_ane_load_fn exactly once
    results_64: list[int] = []
    barrier = threading.Barrier(4)

    def _worker_64() -> None:
        barrier.wait()
        results_64.append(be._ensure_bucket_handle(64))

    threads = [threading.Thread(target=_worker_64) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results_64)) == 1
    assert sum(1 for _mc, fn_s, _s in lib1.load_calls if fn_s == "s64") == 1

    # close() frees all 3 handles without AttributeError on read-only properties
    be.close()
    assert len(lib1.freed) == 3
    assert be._h256 is None and be._h512 is None

    # 2. Tier 2: metadata.json discovery (no symlinks, zero inspect_plan_fn calls)
    dir_meta = _make_local_laya_dir(tmp_path / "mf_meta", with_mf=True)
    (dir_meta / "laya_typed_w8.mlmodelc" / "metadata.json").write_text(
        json.dumps([{"name": "s128"}, {"functionName": "s384"}]),
        encoding="utf-8",
    )
    lib2 = _FakeLib()
    active_lib[:] = [lib2]
    be_meta = LayaAneBackend(dir_meta, fake_dylib, lazy_load=True)
    assert lib2.inspect_calls == [] and lib2.load_calls == []
    assert be_meta.seq_buckets == (128, 384)
    be_meta.close()

    (dir_meta / "laya_typed_w8.mlmodelc" / "metadata.json").write_text(
        json.dumps({"functions": ["main", "s64", "s512"]}),
        encoding="utf-8",
    )
    lib2b = _FakeLib()
    active_lib[:] = [lib2b]
    be_meta2 = LayaAneBackend(dir_meta, fake_dylib, lazy_load=True)
    assert lib2b.inspect_calls == [] and lib2b.load_calls == []
    assert be_meta2.seq_buckets == (64, 512)
    be_meta2.close()

    # 3. Tier 3: fallback to inspect_plan_fn when neither symlinks nor metadata.json exist
    dir_fb = _make_local_laya_dir(tmp_path / "mf_fallback", with_mf=True)
    lib3 = _FakeLib(plan_buckets={192, 512})
    active_lib[:] = [lib3]
    be_fb = LayaAneBackend(dir_fb, fake_dylib, lazy_load=True)
    assert len(lib3.inspect_calls) == 8 and lib3.load_calls == []
    assert be_fb.seq_buckets == (192, 512)
    be_fb.close()

    # 4. Eager load (lazy_load=False) + error paths
    lib4 = _FakeLib()
    active_lib[:] = [lib4]
    be_eager = LayaAneBackend(dir_sym, fake_dylib, lazy_load=False)
    assert len(lib4.load_calls) == 3 and set(be_eager._handles.keys()) == {64, 256, 512}
    be_eager.close()

    lib_partial = _FakeLib(load_fail_for={64})
    active_lib[:] = [lib_partial]
    be_partial = LayaAneBackend(dir_sym, fake_dylib, lazy_load=False)
    assert be_partial.seq_buckets == (256, 512)
    assert set(be_partial._bucket_sources.keys()) == {256, 512}
    be_partial.close()

    lib_fail_all = _FakeLib(load_fail_for={64, 256, 512})
    active_lib[:] = [lib_fail_all]
    with pytest.raises(RuntimeError, match="Failed to load any"):
        LayaAneBackend(dir_sym, fake_dylib, lazy_load=False)

    lib_no_buckets = _FakeLib(plan_buckets=set())
    active_lib[:] = [lib_no_buckets]
    with pytest.raises(RuntimeError, match="Failed to load any"):
        LayaAneBackend(dir_fb, fake_dylib, lazy_load=True)

    # Per-bucket standalone dir with lazy_load=True
    dir_pb = _make_local_laya_dir(tmp_path / "per_bucket", with_mf=False)
    (dir_pb / "laya_typed_s512_w8.mlmodelc").mkdir()
    lib_pb = _FakeLib()
    active_lib[:] = [lib_pb]
    be_pb = LayaAneBackend(dir_pb, fake_dylib, lazy_load=True)
    assert be_pb.seq_buckets == (512,) and be_pb._bucket_sources[512][1] == "main"
    assert lib_pb.load_calls == []
    be_pb.score_noul_batch([{"ids": [5] * 300, "markers": [1, 2], "qtype": 2}])
    assert len(lib_pb.load_calls) == 1 and lib_pb.load_calls[0][1:] == ("main", 512)
    be_pb.close()


def test_laya_hetero_routing_shared_state_and_makespan_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `LayaHeteroEngine` shared-state forwarding, `n=1` fast-paths, and makespan-balanced dual split."""
    import threading
    import time
    from fast_decision_models.laya_hetero import LayaHeteroEngine, _choose_hetero_initial_split

    # 1. _choose_hetero_initial_split invariants across 8x512, 8xshort, 8xmixed, and 64xmixed
    g_512, a_512 = _choose_hetero_initial_split([512] * 8, 16, 8, (64, 128, 192, 256, 384, 512))
    assert (g_512, a_512) == (4, 4)

    g_short, a_short = _choose_hetero_initial_split([96] * 8, 16, 8, (64, 128, 192, 256, 384, 512))
    assert g_short >= 1 and a_short >= 1 and (g_short + a_short) == 8

    g_64, a_64 = _choose_hetero_initial_split([512, 256, 128, 64] * 16, 16, 8, (64, 128, 192, 256, 384, 512))
    assert (g_64, a_64) == (16, 8)

    # 2. Dispatch fast-paths, dual-engine split (8x512, 8xshort, 64x512 non-starving), and shared-state forwarding
    tok_helper = LayaAneBackend.__new__(LayaAneBackend)
    tok_helper.max_len = 512
    tok_helper.head_max_len = 192
    tok_helper.cls_id = 1
    tok_helper.sep_id = 2
    tok_helper.pad_id = 3
    tok_helper.mask_id = 4
    tok_helper.mask_tok = "[MASK]"
    state_tok_calls: list[str] = []

    def _enc_ids(text: str) -> list[int]:
        state_tok_calls.append(text)
        return [10 + (i % 40) for i in range(len(text.split()))]

    tok_helper._encode_ids = _enc_ids  # type: ignore[method-assign]

    dual_calls: list[tuple[str, list[int]]] = []
    gpu_delay_s = [0.0]

    class _MockNeural:
        max_len = 512
        head_max_len = 192
        pad_id = 3
        seq_buckets = (64, 128, 192, 256, 384, 512)

        def encode_state(self, state: Any) -> list[int]:
            return tok_helper.encode_state(state)

        def encode_item(self, state: Any, qdef: Any, **kwargs: Any) -> dict[str, Any]:
            return tok_helper.encode_item(state, qdef, **kwargs)

        def encode_items_for_state(self, state: Any, qdefs: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return tok_helper.encode_items_for_state(state, qdefs, **kwargs)

        def score_noul_batch(self, items: list[dict[str, Any]]) -> list[float]:
            dual_calls.append(("neural", [len(it["ids"]) for it in items]))
            return [0.1 + 0.001 * it["ids"][0] for it in items]

        def score_choice_batch(self, items: list[dict[str, Any]], *, temperature_override: float | None = None) -> list[list[float]]:
            dual_calls.append(("neural", [len(it["ids"]) for it in items]))
            return [[float(it["ids"][0]), float(len(it["ids"]))] for it in items]

        def close(self) -> None:
            pass

    class _MockMetal:
        def score_noul_batch(self, items: list[dict[str, Any]]) -> list[float]:
            if gpu_delay_s[0] > 0:
                time.sleep(gpu_delay_s[0])
            dual_calls.append(("metal", [len(it["ids"]) for it in items]))
            return [0.1 + 0.001 * it["ids"][0] for it in items]

        def score_choice_batch(self, items: list[dict[str, Any]], *, temperature_override: float | None = None) -> list[list[float]]:
            if gpu_delay_s[0] > 0:
                time.sleep(gpu_delay_s[0])
            dual_calls.append(("metal", [len(it["ids"]) for it in items]))
            return [[float(it["ids"][0]), float(len(it["ids"]))] for it in items]

        def close(self) -> None:
            pass

    het = LayaHeteroEngine(
        mode="hetero",
        ane_backend=_MockNeural(),
        gpu_backend=_MockMetal(),
        gpu_batch_size=16,
        ane_batch_size=8,
    )

    # Shared-state forwarding
    st_ids = het.encode_state("alpha beta gamma")
    assert len(state_tok_calls) == 1
    qdefs = [
        {"type": "noul", "instructions": "q1", "criteria": {"false": "no", "true": "yes"}},
        {"type": "noul", "instructions": "q2", "criteria": {"false": "no", "true": "yes"}},
    ]
    assert het.encode_item("alpha beta gamma", qdefs[0], state_ids=st_ids) == tok_helper.encode_item(
        "alpha beta gamma", qdefs[0]
    )
    assert len(het.encode_items_for_state("alpha beta gamma", qdefs)) == 2

    # n=1 short (<=256), n=1 long (>256), and n=2 <=192 use zero threads
    orig_thread = threading.Thread

    def _forbid_thread(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("Tiny batch fast-path must not spawn threads")

    monkeypatch.setattr(threading, "Thread", _forbid_thread)

    dual_calls.clear()
    assert het.score_noul_batch([{"ids": [7] * 256, "markers": [1, 2], "qtype": 2}]) == pytest.approx([0.107])
    assert dual_calls == [("neural", [256])]

    dual_calls.clear()
    assert het.score_noul_batch([{"ids": [8] * 257, "markers": [1, 2], "qtype": 2}]) == pytest.approx([0.108])
    assert dual_calls == [("metal", [257])]

    dual_calls.clear()
    assert het.score_choice_batch(
        [{"ids": [3] * 128, "markers": [1, 2], "qtype": 0}, {"ids": [4] * 192, "markers": [1, 2], "qtype": 0}]
    ) == [[3.0, 128.0], [4.0, 192.0]]
    assert dual_calls == [("neural", [128, 192])]

    monkeypatch.setattr(threading, "Thread", orig_thread)

    # 8x512 splits 4/4 across both engines
    dual_calls.clear()
    items_8x512 = [{"ids": [i + 1] * 512, "markers": [1, 2], "qtype": 0} for i in range(8)]
    res_8x512 = het.score_choice_batch(items_8x512)
    assert res_8x512 == [[float(i + 1), 512.0] for i in range(8)]
    by_engine = {eng: sum(len(lens) for e, lens in dual_calls if e == eng) for eng in ("metal", "neural")}
    assert by_engine == {"metal": 4, "neural": 4}

    # 8xshort exercises both engines cleanly
    dual_calls.clear()
    items_8xshort = [{"ids": [i + 1] * 96, "markers": [1, 2], "qtype": 2} for i in range(8)]
    res_8xshort = het.score_noul_batch(items_8xshort)
    assert res_8xshort == pytest.approx([0.1 + 0.001 * (i + 1) for i in range(8)])
    assert any(e == "metal" for e, _ in dual_calls) and any(e == "neural" for e, _ in dual_calls)

    # 64x512 with slow GPU verifies neural worker steals >256 items down to empty queue (zero starvation!)
    dual_calls.clear()
    gpu_delay_s[0] = 0.015
    items_64x512 = [{"ids": [i + 1] * 512, "markers": [1, 2], "qtype": 0} for i in range(64)]
    res_64 = het.score_choice_batch(items_64x512)
    assert res_64 == [[float(i + 1), 512.0] for i in range(64)]
    neural_batches = [lens for e, lens in dual_calls if e == "neural"]
    assert len(neural_batches) >= 2 and sum(len(b) for b in neural_batches) > 8

    # Legacy delegate without state_ids parameter stays compatible when state_ids is omitted
    class _LegacyTok:
        max_len = 512
        head_max_len = 192
        pad_id = 0

        def encode_item(self, state: Any, qdef: Any, *, max_len: int | None = None, on_truncate: str = "warn"):
            return {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 2}

    legacy_het = LayaHeteroEngine(mode="ane", ane_backend=_LegacyTok(), gpu_backend=None)
    assert legacy_het.encode_item("s", {"type": "noul"})["ids"] == [1, 2, 3]
    het.close()


def test_builder_pass_pipeline_ctx_and_laya_gpu_compile_encode_cpu() -> None:
    from types import SimpleNamespace
    import fast_decision_models.builder as bmod
    from fast_decision_models.laya_hetero import _LAYA_GPU_CACHE_LIMIT_BYTES, LayaMlxGpuBackend

    fake_common = [
        "common::dead_code_elimination",
        "common::fuse_elementwise_to_batchnorm",
        "common::add_fp16_cast",
        "common::const_elimination",
    ]
    fake_default = list(fake_common) + ["common::noop"]
    fake_pp_mod = SimpleNamespace(
        _COMMON_PASSES=fake_common,
        PassPipeline=SimpleNamespace(_PIPELINE_NAME_TO_PASSES={"default": list(fake_default)}),
    )

    with bmod._without_coreml_bn_fp16_passes(pass_pipeline_module=fake_pp_mod):
        assert "common::fuse_elementwise_to_batchnorm" not in fake_pp_mod._COMMON_PASSES
        assert "common::add_fp16_cast" not in fake_pp_mod._COMMON_PASSES
        assert "common::fuse_elementwise_to_batchnorm" not in fake_pp_mod.PassPipeline._PIPELINE_NAME_TO_PASSES["default"]
        assert "common::add_fp16_cast" not in fake_pp_mod.PassPipeline._PIPELINE_NAME_TO_PASSES["default"]
        assert "common::const_elimination" in fake_pp_mod._COMMON_PASSES

    assert fake_pp_mod._COMMON_PASSES == fake_common
    assert fake_pp_mod.PassPipeline._PIPELINE_NAME_TO_PASSES["default"] == fake_default

    with pytest.raises(RuntimeError, match="boom"):
        with bmod._without_coreml_bn_fp16_passes(pass_pipeline_module=fake_pp_mod):
            assert "common::fuse_elementwise_to_batchnorm" not in fake_pp_mod._COMMON_PASSES
            raise RuntimeError("boom")
    assert fake_pp_mod._COMMON_PASSES == fake_common
    assert fake_pp_mod.PassPipeline._PIPELINE_NAME_TO_PASSES["default"] == fake_default

    assert _LAYA_GPU_CACHE_LIMIT_BYTES == (512 << 20)

    # Verify compile_encode routing on LayaMlxGpuBackend._score_generic
    import numpy as np

    class _FakeArray:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        @property
        def shape(self):
            return self._arr.shape

    class _FakeMx:
        int32 = np.int32
        bfloat16 = np.float32

        @staticmethod
        def array(x, dtype=None):
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def abs(x):
            return np.abs(x)

        @staticmethod
        def where(cond, a, b):
            return np.where(cond, a, b)

        @staticmethod
        def eval(*_args):
            pass

        @staticmethod
        def clear_cache():
            pass

    calls: list[str] = []
    fake_backend = object.__new__(LayaMlxGpuBackend)
    fake_backend._mx = _FakeMx
    fake_backend._np = np
    fake_backend.batch_size = 4
    fake_backend.pack_seq_len = 512
    fake_backend.pad_id = 0
    fake_backend.local_half_win = 64
    fake_backend._prev_cache_limit = None
    fake_backend._limits_restored = True
    fake_backend.resolve_temperature = lambda _qt, _k, ov: ov or 1.0  # type: ignore[method-assign]
    fake_backend.model = SimpleNamespace(
        encode_hidden=lambda *args: calls.append("eager") or np.zeros((1, 32, 4), dtype=np.float32),
        _score_laya_markers_fp32=lambda h, lane, pos: np.array([[0.2, 0.8]], dtype=np.float32),
    )
    fake_backend._encode_hidden = lambda *args: calls.append("compiled") or np.zeros((1, 32, 4), dtype=np.float32)

    probs = fake_backend.score_noul_batch([{"ids": [1, 2, 3, 4], "markers": [1, 2], "qtype": 2}])
    assert len(probs) == 1
    assert calls == ["compiled"]

    calls.clear()
    fake_backend._encode_hidden = None
    probs2 = fake_backend.score_noul_batch([{"ids": [1, 2, 3, 4], "markers": [1, 2], "qtype": 2}])
    assert len(probs2) == 1 and probs2 == pytest.approx(probs)
    assert calls == ["eager"]
    fake_backend.close()


def test_build_decision_models_unwraps_bucket_compile_result_cpu(monkeypatch: Any, tmp_path: Path) -> None:
    """Verify `build_laya_ane_artifacts_parallel` unwraps `BucketCompileResult.value` when multifunction=True."""
    import importlib.util
    import sys
    from types import SimpleNamespace

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_decision_models.py"
    spec = importlib.util.spec_from_file_location("test_build_decision_models_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        src_dir = tmp_path / "src_laya"
        (src_dir / "encoder").mkdir(parents=True)
        (src_dir / "tokenizer").mkdir(parents=True)
        (src_dir / "rl_agent_config.json").write_text("{}", encoding="utf-8")
        (src_dir / "encoder" / "config.json").write_text('{"hidden_size": 16}', encoding="utf-8")
        (src_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        (src_dir / "model.safetensors").write_bytes(b"stub")

        art_dir = tmp_path / "artifacts"
        dylib_p = tmp_path / "liblaya_ane_runner.dylib"
        dylib_p.write_bytes(b"")

        monkeypatch.setattr(mod, "_resolve_dir_or_hf", lambda *_a, **_kw: src_dir)
        monkeypatch.setattr(mod, "ensure_laya_dylib", lambda *_a, **_kw: dylib_p)
        monkeypatch.setattr(
            mod,
            "_load_safetensors_as_float32_np",
            lambda _p: {
                "encoder.embeddings.tok_embeddings.weight": np.zeros((4, 16), dtype=np.float32),
                "type_emb.weight": np.zeros((3, 16), dtype=np.float32),
            },
        )
        fake_res_cls = type("BucketCompileResult", (), {})

        def _fake_compile(tasks, _fn, **_kw):
            out = []
            for idx, (s_len, _src, stage_root) in enumerate(tasks):
                pkg_p = Path(stage_root) / f"_laya_typed_s{s_len}_w8.mlpackage"
                pkg_p.mkdir(parents=True, exist_ok=True)
                rec = fake_res_cls()
                rec.index = idx
                rec.label = f"laya_typed_s{s_len}_w8"
                rec.value = str(pkg_p)
                out.append(rec)
            return out

        captured: dict[int, Path] = {}
        monkeypatch.setattr(mod, "compile_buckets_parallel", _fake_compile)
        monkeypatch.setattr(
            mod,
            "_save_laya_multifunction_mlpackage",
            lambda seq_pkgs, dst, **_kw: captured.update(seq_pkgs) or dst,
        )
        monkeypatch.setattr(mod, "_publish_multifunction_and_symlinks", lambda mf, out_d, _lens: out_d / "laya_typed_w8.mlmodelc")
        monkeypatch.setattr(mod.ctypes, "CDLL", lambda _p: SimpleNamespace(laya_ane_inspect_plan_fn=lambda _p, _fn, _m, c: (c.__setitem__(0, 1500), c.__setitem__(1, 0), c.__setitem__(2, 0))))

        out = mod.build_laya_ane_artifacts_parallel(tmp_path, artifacts_dir=art_dir, seq_lens=(64, 128), multifunction=True)
        assert out == art_dir / "laya_typed"
        assert set(captured.keys()) == {64, 128}
        assert all(isinstance(p, Path) and p.name.endswith(".mlpackage") for p in captured.values())
    finally:
        sys.modules.pop(spec.name, None)


def test_kev_min_lcp_group_threshold_and_tail_sorting_cpu() -> None:
    """Verify `_MIN_LCP_GROUP_TOKENS == 128` skips short shared prompts and long tails sort + chunk by `eff_b`."""
    from fast_decision_models.kev_mlx import (
        _MIN_LCP_GROUP_TOKENS,
        KevMlxBackend,
        _partition_exact_lcp_groups,
    )

    assert _MIN_LCP_GROUP_TOKENS == 128

    # Short 49-token shared prefixes must stay in singletons under default `min_lcp`;
    # >= 128-token shared prefixes must group together.
    short_prefix = list(range(10, 59))
    long_prefix = list(range(100, 232))
    encoded = [
        (short_prefix + [901, 902, 903], [{"ids": [1, 2], "decide": 1, "opts": [0]}]),
        (short_prefix + [904, 905, 906, 907], [{"ids": [1, 2], "decide": 1, "opts": [0]}]),
        (long_prefix + [801, 802], [{"ids": [1, 2], "decide": 1, "opts": [0]}]),
        (long_prefix + [803, 804, 805], [{"ids": [1, 2], "decide": 1, "opts": [0]}]),
    ]
    groups, singles = _partition_exact_lcp_groups([0, 1, 2, 3], encoded)
    assert singles == [0, 1]
    assert groups == [([2, 3], 132)]

    # Verify `_score_candidates_against_prefix_cache` sorts variable-length candidate tails (> 64 tokens)
    # and caps batch size by `eff_b` (4 instead of `br_chunk_lim=16`) while scattering to original slots.
    class _FakeMx:
        int32 = np.int32
        bfloat16 = np.float32

        @staticmethod
        def array(x, dtype=None):
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def arange(n, dtype=None):
            return np.arange(n, dtype=dtype)

    seen_calls: list[tuple[int, int]] = []

    class _FakeBackbone:
        _supports_stateless_fused_pass = True

        def __call__(self, ids_mx, *_args, **_kwargs):
            seen_calls.append(tuple(ids_mx.shape))
            return ids_mx

    fake = object.__new__(KevMlxBackend)
    fake._mx = _FakeMx
    fake._np = np
    fake.pad_id = 0
    fake.backbone = _FakeBackbone()
    fake._score_pointer_fp32 = lambda h_mx, decs, _opts, **_kw: [([float(row[d]), 0.0], [float(d), 0.0]) for row, d in zip(h_mx, decs)]

    # 5 candidates in the same 132-tok prefix group with descending tail lengths (between 70 and 200 tokens)
    p_len = 132
    prefix = list(range(p_len))
    tail_lens = [190, 72, 150, 80, 110]
    encoded_all = [
        (prefix + [1000 + i] * tl, [{"ids": [5000 + i, 6000 + i, 7000 + i], "decide": 2, "opts": [0, 1]}])
        for i, tl in enumerate(tail_lens)
    ]
    out_by_orig = [[] for _ in range(len(tail_lens))]
    fake._score_candidates_against_prefix_cache(
        [],
        [p_len],
        [0],
        p_len,
        [[0, 1, 2, 3, 4]],
        encoded_all,
        out_by_orig,
        br_chunk_lim=16,
        eff_b=3,
    )

    # Sorted tail+branch lengths: 75, 83, 113 | 153, 193 -> max batch width 113 and 193 (instead of single 5x193!)
    assert seen_calls == [(3, 113), (2, 193)]
    for idx in range(len(tail_lens)):
        probs, _logits = out_by_orig[idx][0]
        assert probs == [float(7000 + idx), 0.0]


def test_calibration_profiles_and_decision_cascade_cpu() -> None:
    """Verify Track D H3 Platt/Temperature calibration, 0 rank flips, and lazy `Laya -> Kev` cascade."""
    import fast_decision_models as fdm
    from fast_decision_models.scorers import (
        CalibrationProfile,
        DecisionCascadeScorer,
        apply_multiclass_temperature,
        fit_multiclass_temperature,
        fit_platt_scaling,
    )

    assert fdm.DecisionCascadeScorer is DecisionCascadeScorer

    rng = np.random.default_rng(2026)
    # 1. Binary Platt scaling
    raw_logits = np.linspace(-3.0, 3.0, 40)
    y_bin = (raw_logits > 0.2).astype(np.float64)
    a, b = fit_platt_scaling(raw_logits, y_bin, from_probs=False)
    prof_bin = CalibrationProfile(model_id="laya", task_key="ragtruth", method="platt", platt_a=a, platt_b=b)
    prof_bin_rt = CalibrationProfile.from_dict(prof_bin.to_dict())
    assert prof_bin_rt == prof_bin
    p_cal = prof_bin_rt.apply([0.2, 0.8], from_probs=True)
    assert p_cal.shape == (2,) and 0.0 < p_cal[0] < p_cal[1] < 1.0

    # 2. Overconfident multiclass temperature scaling preserves top-1 rankings with lower NLL
    y_mc = rng.integers(0, 4, size=60)
    base_logits = rng.normal(0.0, 0.5, size=(60, 4))
    base_logits[np.arange(60), y_mc] += 0.35
    sharp = np.exp((base_logits / 0.1) - np.max(base_logits / 0.1, axis=1, keepdims=True))
    probs_sharp = sharp / np.sum(sharp, axis=1, keepdims=True)
    t_fit = fit_multiclass_temperature(probs_sharp, y_mc, from_probs=True)
    assert t_fit > 1.0
    probs_cal = apply_multiclass_temperature(probs_sharp, t_fit, from_probs=True)
    assert np.array_equal(np.argmax(probs_cal, axis=1), np.argmax(probs_sharp, axis=1))
    nll_raw = -float(np.mean(np.log(np.clip(probs_sharp[np.arange(60), y_mc], 1e-30, 1.0))))
    nll_cal = -float(np.mean(np.log(np.clip(probs_cal[np.arange(60), y_mc], 1e-30, 1.0))))
    assert nll_cal < nll_raw

    # 3. DecisionCascadeScorer: confident items stay on primary (factory not called when 0 escalations)
    class _FakePrimary:
        def encode_item(self, st, qdef, on_truncate="ignore"):
            return (st, qdef)

        def score_noul_batch(self, encs):
            return [0.25 if "bad" in str(st) else 0.75 for st, _ in encs]

        def score_choice_batch(self, encs):
            out = []
            for st, qdef in encs:
                if qdef["id"] == "s1":
                    out.append([0.85, 0.10, 0.05] if st == "easy" else [0.40, 0.35, 0.25])
                else:
                    out.append([0.90, 0.10])
            return out

    factory_calls = 0

    class _FakeKev:
        def score_multi_states(self, pairs, batch_docs=8):
            out = []
            for _st, qdefs in pairs:
                if qdefs[0]["id"] == "s1":
                    out.append([([0.10, 0.80, 0.10], [0.0, 2.0, 0.0])])
                else:
                    out.append([([0.20, 0.80], [0.0, 1.0])])
            return out

    def _kev_factory():
        nonlocal factory_calls
        factory_calls += 1
        return _FakeKev()

    cascade = DecisionCascadeScorer(
        _FakePrimary(),
        escalation_factory=_kev_factory,
        binary_profile=prof_bin,
        confidence_threshold=0.60,
    )
    bin_out = cascade.score_binary_batch(["good doc", "bad doc"], {"id": "rt", "type": "noul"})
    assert len(bin_out) == 2 and bin_out[0] > bin_out[1]
    assert factory_calls == 0

    # All confident -> zero escalations -> factory still not called!
    res_easy = cascade.score_multiclass_cascade(
        ["easy", "easy"],
        {"id": "s1", "options": ["d0", "d1", "out_of_scope"]},
        stage1_labels=["d0", "d1", "out_of_scope"],
        stage2_qdefs={"d0": {"id": "s2_d0"}, "d1": {"id": "s2_d1"}},
    )
    assert res_easy["esc_count"] == 0
    assert factory_calls == 0

    # Mixed confident + uncertain -> lazy load once and escalate only uncertain item!
    res_mix = cascade.score_multiclass_cascade(
        ["easy", "hard"],
        {"id": "s1", "options": ["d0", "d1", "out_of_scope"]},
        stage1_labels=["d0", "d1", "out_of_scope"],
        stage2_qdefs={"d0": {"id": "s2_d0"}, "d1": {"id": "s2_d1"}},
    )
    assert factory_calls == 1
    assert res_mix["escalated_indices"] == [1]
    assert res_mix["stage2_preds"][0][0] == "d0"
    assert res_mix["stage2_preds"][1][0] == "d1"




