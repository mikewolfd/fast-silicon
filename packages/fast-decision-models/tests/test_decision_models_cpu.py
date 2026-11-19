"""CPU, stdlib-isolation, IR-guard, and native-build tests for `fast-decision-models`."""

from __future__ import annotations

import io
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import zipfile

import numpy as np

from fast_decision_models._common import softmax_2, softmax_vec
from fast_decision_models.builder import _load_kev_head_tensors_stdlib_zip
from fast_decision_models.kev_mlx import _KEV_SPECIAL_RE, _kev_render
from fast_decision_models.paths import (
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
    for fn in ("laya_ane_inspect_plan", "laya_ane_load", "laya_ane_exec_batch", "laya_ane_free"):
        assert hasattr(lib, fn), f"Missing exported symbol {fn}"
    lib.laya_ane_inspect_plan.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    counts = (ctypes.c_int * 3)(-1, -1, -1)
    lib.laya_ane_inspect_plan(b"/nonexistent/missing.mlmodelc", 1, counts)
    assert list(counts) == [0, 0, 0]
