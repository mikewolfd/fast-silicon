"""Hardware & CPU verification for 8-bit Linear ANE Reader, Zero-Copy IOSurface->Metal KV_i8 Bridge,
AsyncHybridPrefillRing (`use_ane_ring`), and Lossless GPU Writer Optimizations (`ff_q` causal jump, NaN/-1, full_logits_1d).
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from fast_gemma4 import FastGemma4Engine, resolve_paths
from fast_gemma4.ane_pipeline import (
    AnePrefixRunner,
    AnePrefixWorker,
    AsyncHybridPrefillRing,
    SlotCacheMetalBinding,
    inspect_ane_bundle_plan,
    is_ane_resident_bundle,
)
from fast_gemma4.grammar import GrammarHelper
from fast_gemma4.kernels import (
    fused_norm_add_norm,
    fused_ple_geglu_indexed,
    fused_res_rms_norm,
    fused_softcap_argmax_bf16,
)
from fast_gemma4.paths import Gemma4Paths
from fast_gemma4.speculative import run_decode_rolling

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_ane_script() -> Any:
    script_path = REPO_ROOT / "scripts" / "build_ane_models.py"
    spec = importlib.util.spec_from_file_location("build_ane_models_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_ane_models_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


@pytest.fixture(scope="module")
def gpu_lock():
    with open("/tmp/m4_gpu_validation.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield


@pytest.fixture(scope="module")
def paths() -> Gemma4Paths:
    return resolve_paths()


@pytest.fixture(scope="module")
def engine(paths: Gemma4Paths, gpu_lock):
    eng = FastGemma4Engine(home=paths.home, default_mode="auto", warmup=True, warmup_ane=False)
    try:
        yield eng
    finally:
        eng.close()


def test_build_ane_models_linear_w8_awq_and_e4b_arch_cpu(tmp_path: Path) -> None:
    """Verify Pillar 1 builder: linear per-channel int8 (no k-means/per_block), AWQ-MSE clipping, and E2B/E4B arch inference."""
    import ast

    script_src = (REPO_ROOT / "scripts" / "build_ane_models.py").read_text(encoding="utf-8")
    tree = ast.parse(script_src)
    attr_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "palettize_weights" not in attr_calls
    assert "linear_quantize_weights" in attr_calls
    kw_pairs = {
        (kw.arg, kw.value.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg is not None and isinstance(kw.value, ast.Constant)
    }
    assert ("mode", "linear_symmetric") in kw_pairs
    assert ("dtype", "int8") in kw_pairs
    assert ("granularity", "per_channel") in kw_pairs
    assert ("granularity", "per_block") not in kw_pairs

    mod = _load_build_ane_script()

    # 1. Synthetic E2B EarlyExit14 (k=15, H=1536, n_kv=1, 12 SWA + 3 Full)
    e2b_npz: dict[str, np.ndarray] = {
        "per_layer_model_proj_w": np.zeros((14 * 256, 1536, 1, 1), dtype=np.float16),
        "per_layer_proj_norm_w": np.ones((256,), dtype=np.float16),
    }
    for i in range(15):
        hd = 512 if i in (4, 9, 14) else 256
        e2b_npz[f"l{i}_w_in"] = np.ones((1, 1536, 1, 1), dtype=np.float16)
        e2b_npz[f"l{i}_w_kn"] = np.ones((1, 1, 1, hd), dtype=np.float16)
        if i == 14:
            e2b_npz[f"l{i}_kv"] = np.zeros((2 * hd, 1536, 1, 1), dtype=np.float16)
        else:
            e2b_npz[f"l{i}_qkv"] = np.zeros((10 * hd, 1536, 1, 1), dtype=np.float16)
    arch_e2b = mod.infer_ane_arch_from_npz(e2b_npz)
    assert arch_e2b["first_kv_shared"] == 15 and arch_e2b["exit_layer"] == 14
    assert arch_e2b["hidden_size"] == 1536 and arch_e2b["n_kv"] == 1
    assert arch_e2b["n_swa"] == 12 and arch_e2b["n_full"] == 3

    # 2. Synthetic E4B EarlyExit23 (k=24, H=2560, n_kv=2, 20 SWA + 4 Full {5, 11, 17, 23})
    e4b_npz: dict[str, np.ndarray] = {
        "per_layer_model_proj_w": np.zeros((23 * 256, 2560, 1, 1), dtype=np.float16),
        "per_layer_proj_norm_w": np.ones((256,), dtype=np.float16),
    }
    for i in range(24):
        hd = 512 if i in (5, 11, 17, 23) else 256
        e4b_npz[f"l{i}_w_in"] = np.ones((1, 2560, 1, 1), dtype=np.float16)
        e4b_npz[f"l{i}_w_kn"] = np.ones((1, 1, 1, hd), dtype=np.float16)
        if i == 23:
            e4b_npz[f"l{i}_kv"] = np.zeros((4 * hd, 2560, 1, 1), dtype=np.float16)
        else:
            e4b_npz[f"l{i}_qkv"] = np.zeros((12 * hd, 2560, 1, 1), dtype=np.float16)
    arch_e4b = mod.infer_ane_arch_from_npz(e4b_npz)
    assert arch_e4b["first_kv_shared"] == 24 and arch_e4b["exit_layer"] == 23
    assert arch_e4b["hidden_size"] == 2560 and arch_e4b["n_kv"] == 2
    assert arch_e4b["n_swa"] == 20 and arch_e4b["n_full"] == 4

    # 3. Closed-form AWQ-MSE channel clipping strictly lowers activation-weighted quantization error on outliers
    rng = np.random.default_rng(123)
    w = rng.standard_normal((16, 64, 1, 1)).astype(np.float32) * 0.2
    w[:, 0, 0, 0] = 6.0  # Outlier column with tiny activation energy
    col_sq = np.ones((64,), dtype=np.float32)
    col_sq[0] = 1e-4
    clipped, mean_alpha = mod.apply_awq_mse_clipping({"l0_wo": w.astype(np.float16)}, {"l0_wo": col_sq})
    assert 0.80 <= mean_alpha < 1.0

    w_orig = w.astype(np.float16)[:, :, 0, 0].astype(np.float32)

    def _w_err(w_mat: np.ndarray) -> float:
        wf = w_mat[:, :, 0, 0].astype(np.float32)
        sc = np.maximum(np.max(np.abs(wf), axis=1, keepdims=True) / 127.0, 1e-8)
        wq = np.clip(np.round(wf / sc), -127.0, 127.0) * sc
        diff = wq - w_orig
        return float(np.sum((diff * diff) * col_sq[None, :]))

    assert _w_err(clipped["l0_wo"]) < _w_err(w.astype(np.float16))

    # 4. Legacy LUT8 MIL is immediately rejected by is_ane_resident_bundle
    fake_lut_mc = tmp_path / "legacy_lut.mlmodelc"
    fake_lut_mc.mkdir()
    (fake_lut_mc / "model.mil").write_text("program(1.0) { constexpr_lut_to_dense(); }", encoding="utf-8")
    assert is_ane_resident_bundle(REPO_ROOT / "build" / "libgemma4_ane_prefix.dylib", fake_lut_mc) is False

    # 5. API contract signatures for run_decode_rolling, FastGemma4Engine.run_rolling, and generate_batch
    import inspect
    from fast_gemma4.speculative import _parse_use_ane_ring

    for sig, expected_use_ane_ring in (
        (inspect.signature(run_decode_rolling).parameters, False),
        (inspect.signature(FastGemma4Engine.run_rolling).parameters, False),
        (inspect.signature(FastGemma4Engine.generate_batch).parameters, "auto"),
    ):
        assert sig["use_ane_ring"].default == expected_use_ane_ring
        assert sig["ane_cohort_size"].default == 16
        assert sig["enable_ff_jump"].default is True
    assert _parse_use_ane_ring(False) == (False, False)
    assert _parse_use_ane_ring(True) == (True, True)
    assert _parse_use_ane_ring("require") == (True, True)
    assert _parse_use_ane_ring("auto") == (True, False)
    with pytest.raises(ValueError):
        _parse_use_ane_ring("bogus")


def test_build_ane_compile_mlpackage_in_process_without_xcrun_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CoreML .mlpackage compilation works in-process on CLT-only Macs without `xcrun coremlcompiler`."""
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import ast
    import subprocess as sp

    tree = ast.parse((REPO_ROOT / "scripts" / "build_ane_models.py").read_text(encoding="utf-8"))
    call_consts = {
        sub.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }
    assert not (call_consts & {"xcrun", "coremlc", "coremlcompiler"})

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
    def _tiny_prog(x):
        return mb.add(x=x, y=np.float32(1.0), name="y")

    ml = ct.convert(
        _tiny_prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = tmp_path / "tiny_gemma4.mlpackage"
    ml.save(str(pkg))
    dst = tmp_path / "compiled" / "tiny_gemma4.mlmodelc"
    dst.mkdir(parents=True)
    (dst / "stale.txt").write_text("stale", encoding="utf-8")

    orig_run = sp.run
    orig_check_call = sp.check_call

    def _forbid_xcrun(orig_fn):
        def _wrapper(cmd, *args, **kwargs):
            head = Path(str(cmd[0])).name if isinstance(cmd, (list, tuple)) and cmd else str(cmd)
            assert head not in {"xcrun", "coremlc", "coremlcompiler"}, f"Forbidden Xcode CLI call: {cmd}"
            return orig_fn(cmd, *args, **kwargs)
        return _wrapper

    monkeypatch.setattr(sp, "run", _forbid_xcrun(orig_run))
    monkeypatch.setattr(sp, "check_call", _forbid_xcrun(orig_check_call))

    mod = _load_build_ane_script()
    got = mod.compile_mlpackage_in_process(pkg, dst)
    assert got == dst
    assert dst.is_dir()
    assert not (dst / "stale.txt").exists()
    assert (dst / "model.mil").is_file() or (dst / "coremldata.bin").is_file()


def test_linear_w8_bundle_compute_plan_and_kv_fidelity(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify Pillar 1: 1,326/1,326 ANE ops (`0` CPU runtime ops) and `cos >= 0.9983` on Linear-W8 ANE KV."""
    w8_bundle = paths.resolve_ane_bundle("w8", model_dir=None)
    assert w8_bundle is not None and w8_bundle.exists(), f"Missing {paths.ane_w8_bundle}"
    plan = inspect_ane_bundle_plan(paths.lib_ane_prefix, w8_bundle, use_cache=False)
    assert plan["ok"] is True
    assert plan["ane_ops"] >= 1326
    assert plan["cpu_runtime_ops"] == 0
    assert plan["gpu_ops"] == 0
    assert plan["has_legacy_kmeans_lut"] is False
    assert plan["has_linear_w8"] is True
    assert is_ane_resident_bundle(paths.lib_ane_prefix, w8_bundle) is True

    fix_dir = paths.home / "tests" / "fixtures"
    if not fix_dir.is_dir():
        fix_dir = Path(__file__).resolve().parent / "fixtures"
    schema_str = (fix_dir / "extract_v1.schema.json").read_text(encoding="utf-8")
    docs = json.loads((fix_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    sys_prompt = (fix_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    prompts = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]
    enc = engine.encode_prompts(prompts, system_prompt=sys_prompt, use_chat_template=True)
    gh = engine.get_grammar(schema_str, whitespace_flexible=False)
    lcp = min(len(x) for x in enc)
    for k in range(lcp):
        if any(x[k] != enc[0][k] for x in enc[1:]):
            lcp = k
            break
    sys_ids = enc[0][:lcp]
    sufs_8 = [(x[lcp:] + list(gh.upfront_ff))[:257] for x in enc]
    pkv = engine.get_prefix_kv(sys_ids)
    model = engine.model

    c_gpu = model.make_slot_cache(pkv, batch_size=8, doc_cap_full=512, swa_ring_cap=512, kv_i8=False)
    model._prefill_early14_slot_group(sufs_8, list(range(8)), c_gpu, chunk_sz=256)

    w_emb, w_pli14 = model.build_cpu_ane_embed_tables()
    runner = AnePrefixRunner(paths.lib_ane_prefix, w8_bundle, pipe_depth=4, out_ring_slots=16)
    try:
        worker = AnePrefixWorker(model, runner, pkv, w_emb, w_pli14)
        ane_out = worker.run_wave_cpu_ane(sufs_8)
        act0, vlen0, ok_s, ov_s, ok_f, ov_f = ane_out["chunks"][0]
    finally:
        runner.close()

    gpu_ks = np.stack([np.array(c_gpu["layers"][l]["k"][:, :256, 0, :].astype(mx.float32)) for l in model.swa_layers], axis=1)
    gpu_vs = np.stack([np.array(c_gpu["layers"][l]["v"][:, :256, 0, :].astype(mx.float32)) for l in model.swa_layers], axis=1)
    gpu_kf = np.stack([np.array(c_gpu["layers"][l]["k"][:, :256, 0, :].astype(mx.float32)) for l in model.full_layers], axis=1)
    gpu_vf = np.stack([np.array(c_gpu["layers"][l]["v"][:, :256, 0, :].astype(mx.float32)) for l in model.full_layers], axis=1)

    ok_s_4d = ok_s[:, :, 0, :, :] if ok_s.ndim == 5 else ok_s
    ov_s_4d = ov_s[:, :, 0, :, :] if ov_s.ndim == 5 else ov_s
    ok_f_4d = ok_f[:, :, 0, :, :] if ok_f.ndim == 5 else ok_f
    ov_f_4d = ov_f[:, :, 0, :, :] if ov_f.ndim == 5 else ov_f

    cos_ks = float(np.mean([_cosine(gpu_ks[b, :, :vl], ok_s_4d[ri, :, :vl]) for ri, (b, vl) in enumerate(zip(act0, vlen0))]))
    cos_vs = float(np.mean([_cosine(gpu_vs[b, :, :vl], ov_s_4d[ri, :, :vl]) for ri, (b, vl) in enumerate(zip(act0, vlen0))]))
    cos_kf = float(np.mean([_cosine(gpu_kf[b, :, :vl], ok_f_4d[ri, :, :vl]) for ri, (b, vl) in enumerate(zip(act0, vlen0))]))
    cos_vf = float(np.mean([_cosine(gpu_vf[b, :, :vl], ov_f_4d[ri, :, :vl]) for ri, (b, vl) in enumerate(zip(act0, vlen0))]))
    cos_l14k = float(np.mean([_cosine(gpu_kf[b, 2, :vl], ok_f_4d[ri, 2, :vl]) for ri, (b, vl) in enumerate(zip(act0, vlen0))]))
    assert cos_ks >= 0.9983
    assert cos_vs >= 0.9983
    assert cos_kf >= 0.9983
    assert cos_vf >= 0.9983
    assert cos_l14k >= 0.9983


def test_zero_copy_iosurface_to_metal_kv_i8_scatter_bridge(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify Pillar 2: zero-copy IOSurface -> Metal KV_i8 scatter (`max_int8_diff == 0`, small B=2 sub-alloc + 512 SWA wrap)."""
    w8_bundle = paths.resolve_ane_bundle("w8", model_dir=None)
    assert w8_bundle is not None
    model = engine.model
    toks = engine.tokenizer.encode(
        "Pacific Transmission Consortium seasonal inverter and synchronous reserve technical reliability report.",
        add_special_tokens=False,
    )
    pkv = engine.get_prefix_kv(toks[:6])
    # 580 / 552 suffix tokens -> 3 chunks (256 + 256 + 67 / 39, both >= 32) crossing SWA ring cap 512!
    sufs_2 = [(toks[6:] * 120)[:580], (toks[6:] * 120)[:552]]

    def _ref_quant_i8_f16(x_f16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xf = x_f16.astype(np.float32)
        amax = np.max(np.abs(xf), axis=-1)
        sc = np.maximum(amax * np.float32(1.0 / 127.0), np.float32(1e-6))
        q = np.clip(np.rint(xf / sc[..., None]), -127.0, 127.0).astype(np.int8)
        return q, sc.astype(np.float16)

    ref_swa_k = np.zeros((2, 12, 512, 256), dtype=np.int8)
    ref_swa_ks = np.ones((2, 12, 512), dtype=np.float16)
    ref_swa_v = np.zeros((2, 12, 512, 256), dtype=np.int8)
    ref_swa_vs = np.ones((2, 12, 512), dtype=np.float16)
    ref_full_k = np.zeros((2, 3, 784, 512), dtype=np.int8)
    ref_full_ks = np.ones((2, 3, 784), dtype=np.float16)
    ref_full_v = np.zeros((2, 3, 784, 512), dtype=np.int8)
    ref_full_vs = np.ones((2, 3, 784), dtype=np.float16)

    c_zc = model.make_slot_cache(pkv, batch_size=2, doc_cap_full=768, swa_ring_cap=512, kv_i8=True)

    w_emb, w_pli14 = model.build_cpu_ane_embed_tables()
    runner = AnePrefixRunner(paths.lib_ane_prefix, w8_bundle, pipe_depth=4, out_ring_slots=16)
    try:
        assert runner.zero_copy is True
        worker = AnePrefixWorker(model, runner, pkv, w_emb, w_pli14)
        ane_ref = worker.run_wave_cpu_ane(sufs_2)
        for ch_k, (act, vlen, ks_c, vs_c, kf_c, vf_c) in ane_ref["chunks"].items():
            c0 = ch_k * 256
            ks_4d = ks_c[:, :, 0, :, :] if ks_c.ndim == 5 else ks_c
            vs_4d = vs_c[:, :, 0, :, :] if vs_c.ndim == 5 else vs_c
            kf_4d = kf_c[:, :, 0, :, :] if kf_c.ndim == 5 else kf_c
            vf_4d = vf_c[:, :, 0, :, :] if vf_c.ndim == 5 else vf_c
            for ri, b in enumerate(act):
                vl = vlen[ri]
                kq_s, ksc_s = _ref_quant_i8_f16(ks_4d[ri, :, :vl, :])
                vq_s, vsc_s = _ref_quant_i8_f16(vs_4d[ri, :, :vl, :])
                kq_f, ksc_f = _ref_quant_i8_f16(kf_4d[ri, :, :vl, :])
                vq_f, vsc_f = _ref_quant_i8_f16(vf_4d[ri, :, :vl, :])
                slots = np.arange(c0, c0 + vl) & 511
                for j, sl in enumerate(slots):
                    ref_swa_k[b, :, int(sl), :] = kq_s[:, j, :]
                    ref_swa_ks[b, :, int(sl)] = ksc_s[:, j]
                    ref_swa_v[b, :, int(sl), :] = vq_s[:, j, :]
                    ref_swa_vs[b, :, int(sl)] = vsc_s[:, j]
                ref_full_k[b, :, c0 : c0 + vl, :] = kq_f
                ref_full_ks[b, :, c0 : c0 + vl] = ksc_f
                ref_full_v[b, :, c0 : c0 + vl, :] = vq_f
                ref_full_vs[b, :, c0 : c0 + vl] = vsc_f

        ring = AsyncHybridPrefillRing(model, runner, pkv, w_emb, w_pli14, min_chunk_tokens=32)
        with SlotCacheMetalBinding(runner, model, c_zc) as binding:
            item = ring.prefill_cohort_into_ring([0, 1], sufs_2, pipe_depth=4)
            dt = ring.scatter_cohort_to_slot_cache(binding, item, [0, 1])
            assert 0.0 < dt < 0.05
    finally:
        runner.close()

    for si, li in enumerate(model.swa_layers):
        assert np.array_equal(ref_swa_k[:, si], np.array(c_zc["layers"][li]["k"])[:, :, 0, :])
        assert np.array_equal(ref_swa_v[:, si], np.array(c_zc["layers"][li]["v"])[:, :, 0, :])
        assert np.array_equal(ref_swa_ks[:, si], np.array(c_zc["layers"][li]["k_s"])[:, :, 0])
        assert np.array_equal(ref_swa_vs[:, si], np.array(c_zc["layers"][li]["v_s"])[:, :, 0])
    for fi, li in enumerate(model.full_layers):
        assert np.array_equal(ref_full_k[:, fi], np.array(c_zc["layers"][li]["k"])[:, :, 0, :])
        assert np.array_equal(ref_full_v[:, fi], np.array(c_zc["layers"][li]["v"])[:, :, 0, :])
        assert np.array_equal(ref_full_ks[:, fi], np.array(c_zc["layers"][li]["k_s"])[:, :, 0])
        assert np.array_equal(ref_full_vs[:, fi], np.array(c_zc["layers"][li]["v_s"])[:, :, 0])


def test_async_hybrid_prefill_ring_rolling_and_long_tail(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify Pillar 3: Wave-0 50/50 GPU+ANE bootstrap + background ANE refill + >768-token GPU tail finish."""
    fix_dir = paths.home / "tests" / "fixtures"
    if not fix_dir.is_dir():
        fix_dir = Path(__file__).resolve().parent / "fixtures"
    schema_str = (fix_dir / "extract_v1.schema.json").read_text(encoding="utf-8")
    docs = json.loads((fix_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    sys_prompt = (fix_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    prompts_8 = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]
    prompts_12 = prompts_8 + prompts_8[:4]

    ref_gpu = engine.generate_batch(
        prompts_12,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
        batch_size=6,
        use_ane_ring=False,
    )
    res_hyb = engine.generate_batch(
        prompts_12,
        system_prompt=sys_prompt,
        json_schema=schema_str,
        max_tokens=512,
        mode="gpu",
        speculative=False,
        batch_size=6,
        use_ane_ring=True,
        ane_cohort_size=3,
    )
    assert res_hyb.stats["used_ane_ring"] is True
    assert res_hyb.stats["gpu_boot_cohorts"] == 1
    assert res_hyb.stats["ane_cohorts"] >= 3
    assert res_hyb.stats["valid_json_count"] == 12
    assert res_hyb.stats["valid_schema_count"] == 12
    for a, b in zip(ref_gpu.items, res_hyb.items):
        assert a.parsed is not None and b.parsed is not None
        assert a.parsed["status"] == b.parsed["status"]
        assert a.parsed["text_quality"] == b.parsed["text_quality"]

    # Also test `engine.run_rolling` with a suffix longer than the ANE covers after this 5-token prefix
    # (8 chunks = 2,048 tokens), so the GPU tail finish still runs.
    gh_simple = engine.get_grammar(
        {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
        whitespace_flexible=False,
    )
    base_ids = engine.tokenizer.encode("Technical reliability engineering record summary.\n", add_special_tokens=False)
    sys_ids = [engine.tokenizer.bos_token_id] + base_ids[:4]
    long_suf = (base_ids * 400)[:2100] + list(gh_simple.upfront_ff)
    assert len(long_suf) - 1 > 2048
    roll_long = engine.run_rolling(
        sys_ids=sys_ids,
        suf_ids_list=[long_suf, long_suf[:300] + list(gh_simple.upfront_ff)],
        max_tokens=16,
        B_slots=2,
        gh=gh_simple,
        use_ane_ring=True,
        ane_cohort_size=1,
    )
    assert roll_long["used_ane_ring"] is True
    assert roll_long["valid_schemas"] == [True, True]


def test_writer_nan_argmax_empty_batch_and_full_vocab_fallback(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify Pillar 4a/4b: all-NaN `-1` argmax, FloatingPointError in rolling, `m_rows==0` guards, and `full_logits_1d` fallback."""
    # 1. fused_softcap_argmax_bf16 NaN -> -1 & empty batch
    empty_am = fused_softcap_argmax_bf16(mx.zeros((0, 1, 256), dtype=mx.bfloat16), 30.0)
    mx.eval(empty_am)
    assert empty_am.shape == (0,)

    nan_arr = np.zeros((3, 1, 512), dtype=np.float32)
    nan_arr[0, 0, :] = np.nan  # all-NaN row!
    nan_arr[1, 0, 77] = 10.0
    nan_arr[2, 0, 15] = 20.0
    nan_arr[2, 0, 499] = np.nan  # partial-NaN row!
    am_nan = fused_softcap_argmax_bf16(mx.array(nan_arr, dtype=mx.bfloat16), 30.0)
    mx.eval(am_nan)
    assert am_nan.tolist() == [-1, 77, -1]

    # Rolling decode must raise FloatingPointError when am_mx contains -1 (never wrap new_to_old[-1])
    class _NanSlotModel:
        window_size = 512
        kv_i8 = True
        new_to_old = np.arange(32, dtype=np.int32)

        def make_slot_cache(self, prefix_kv, batch_size, doc_cap_full=128, swa_ring_cap=512, kv_i8=None):
            return {"batch_size": batch_size, "l_sys": 0, "k_sys": [], "v_sys": [], "swa_cap": 512, "full_cap": 128, "kv_i8": True, "layers": []}

        def prefill_slots_lazy_lg(self, sub_sufs, target_slots, cache, **kw):
            x = mx.zeros((len(sub_sufs), 1, 16), dtype=mx.bfloat16)
            am = mx.array([-1] * len(sub_sufs), dtype=mx.int32)
            return x, am, [len(s) for s in sub_sufs]

    with pytest.raises(FloatingPointError, match="NaN logits"):
        run_decode_rolling(_NanSlotModel(), frozenset({1}), None, {"l_sys": 0, "k_sys": [], "v_sys": []}, [[10, 11]])

    # 2. Empty-batch (m_rows == 0) + 3D parity on norm/PLE kernels
    z_1536 = mx.zeros((0, 1536), dtype=mx.bfloat16)
    w_1536 = mx.ones((1536,), dtype=mx.bfloat16)
    h0, p0 = fused_norm_add_norm(z_1536, z_1536, w_1536, w_1536, 1e-6, None)
    r0 = fused_res_rms_norm(z_1536, z_1536, w_1536, 1e-6)
    g0 = fused_ple_geglu_indexed(
        mx.zeros((0, 1, 256), dtype=mx.bfloat16),
        mx.zeros((0, 1, 14, 256), dtype=mx.bfloat16),
        mx.array([0], dtype=mx.int32),
    )
    mx.eval(h0, p0, r0, g0)
    assert h0.shape == (0, 1536) and r0.shape == (0, 1536) and g0.shape == (0, 1, 256)

    # Verify E4B D=2560 support in fused_norm_add_norm and fused_res_rms_norm (including 3D B=1, L=6)
    rng = np.random.default_rng(99)
    h3 = mx.array(rng.standard_normal((1, 6, 2560)), dtype=mx.bfloat16)
    x3 = mx.array(rng.standard_normal((1, 6, 2560)), dtype=mx.bfloat16)
    wp = mx.array(rng.uniform(0.5, 1.5, size=(2560,)), dtype=mx.bfloat16)
    wn = mx.array(rng.uniform(0.5, 1.5, size=(2560,)), dtype=mx.bfloat16)
    sc = mx.array([0.85], dtype=mx.bfloat16)
    ra, na = fused_norm_add_norm(h3, x3, wp, wn, 1e-6, sc)
    res_ref = (h3 + mx.fast.rms_norm(x3, wp, 1e-6)) * sc
    na_ref = mx.fast.rms_norm(res_ref, wn, 1e-6)
    rr = fused_res_rms_norm(h3, x3, wp, 1e-6)
    rr_ref = h3 + mx.fast.rms_norm(x3, wp, 1e-6)
    mx.eval(ra, na, res_ref, na_ref, rr, rr_ref)
    assert _cosine(np.array(ra.astype(mx.float32)), np.array(res_ref.astype(mx.float32))) >= 0.9999
    assert _cosine(np.array(na.astype(mx.float32)), np.array(na_ref.astype(mx.float32))) >= 0.9999
    assert _cosine(np.array(rr.astype(mx.float32)), np.array(rr_ref.astype(mx.float32))) >= 0.9999

    # 3. Full-vocab (`262,144`) fallback hygiene when schema requires tokens outside pruned `150,786` head
    # Default stock engine uses full `262,144` vocab (`remapper is None`); construct explicit pruned engine to verify fallback.
    assert engine.model.remapper is None
    assert engine.model.vocab_size == 262144
    pruned_engine = FastGemma4Engine(
        home=paths.home,
        default_mode="gpu",
        vocab_out=150786,
        mtp_drafter=None,
        warmup=False,
        warmup_ane=False,
    )
    try:
        model = pruned_engine.model
        assert model.remapper is not None
        assert model.vocab_size == 150786
        x_probe = mx.array(rng.standard_normal((model.hidden_size,)), dtype=mx.bfloat16)
        full_lg = model.compute_full_vocab_logits_1d(x_probe, apply_softcap=True)
        pruned_lg = np.array(
            model._apply_lm_head(x_probe.reshape(1, 1, -1), apply_softcap=True)[0, 0].astype(mx.float32)
        )
        assert full_lg.shape == (262144,)
        assert np.allclose(full_lg[model.new_to_old], pruned_lg, atol=5e-2)

        # Pick two clean printable multi-char non-ASCII tokens that were pruned (`old_to_new == -1`)
        unkept = np.where(model.old_to_new < 0)[0]
        chosen_toks: list[str] = []
        for tid in unkept[:5000]:
            s = pruned_engine.tokenizer.decode([int(tid)], skip_special_tokens=False)
            if len(s) >= 2 and s.strip() and all(ord(ch) > 127 for ch in s):
                enc_back = pruned_engine.tokenizer.encode(s, add_special_tokens=False)
                if enc_back == [int(tid)]:
                    chosen_toks.append(s)
                    if len(chosen_toks) == 2:
                        break
        assert len(chosen_toks) == 2, f"Could not find 2 single-token unkept literals, got {chosen_toks}"
        unkept_schema = {
            "type": "object",
            "properties": {"tag": {"type": "string", "enum": chosen_toks}},
            "required": ["tag"],
            "additionalProperties": False,
        }
        res_unkept = pruned_engine.generate_batch(
            [f"Select tag {chosen_toks[1]} in JSON."],
            json_schema=unkept_schema,
            max_tokens=24,
            mode="gpu",
            speculative=False,
        )
        assert res_unkept.items[0].valid_json is True
        assert res_unkept.items[0].valid_schema is True
        assert res_unkept.items[0].parsed["tag"] in chosen_toks
    finally:
        pruned_engine.close()


def test_ff_q_causal_slice_jumping_and_prompt_lookup_parity(paths: Gemma4Paths, engine: FastGemma4Engine) -> None:
    """Verify Pillar 4c: single-launch `keep_from_col` / `forward_ff_jump_slot` bit/greedy parity on `docs_8` (B=1..8)."""
    model = engine.model
    span_ids = engine.tokenizer.encode('{"status":"usable","correspondence_flow":"', add_special_tokens=False)
    k = len(span_ids)
    assert k >= 5

    pkv = engine.model.prefill_prefix([engine.tokenizer.bos_token_id, 105, 2364])
    # 1. Slot KV_i8 sequential L_q=1 vs single-launch forward_ff_jump_slot
    c_seq = model.make_slot_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=512, kv_i8=True)
    sl0 = mx.array([0], dtype=mx.int32)
    for idx_t, tid in enumerate(span_ids):
        x_seq, am_seq = model.forward_decode_step_slot(
            mx.array([[tid]], dtype=mx.int32),
            c_seq,
            mx.array([pkv["l_sys"] + idx_t], dtype=mx.int32),
            sl0,
            n_keep=1,
            return_logits=True,
        )
    mx.eval(x_seq, am_seq)

    c_jmp = model.make_slot_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=512, kv_i8=True)
    x_jmp, am_jmp = model.forward_ff_jump_slot(
        mx.array([span_ids], dtype=mx.int32),
        c_jmp,
        mx.array([pkv["l_sys"]], dtype=mx.int32),
        sl0,
        return_logits=True,
    )
    mx.eval(x_jmp, am_jmp)

    lg_seq_np = np.array(x_seq[0, 0].astype(mx.float32))
    lg_jmp_np = np.array(x_jmp[0, 0].astype(mx.float32))
    assert _cosine(lg_seq_np, lg_jmp_np) >= 0.999
    assert am_seq.tolist() == am_jmp.tolist()

    # 2. sample_docs_8 greedy parity across B=1 (Rolling & PromptLookup), B=4, and B=8
    fix_dir = paths.home / "tests" / "fixtures"
    if not fix_dir.is_dir():
        fix_dir = Path(__file__).resolve().parent / "fixtures"
    schema_str = (fix_dir / "extract_v1.schema.json").read_text(encoding="utf-8")
    docs = json.loads((fix_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    sys_prompt = (fix_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    prompts_8 = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in docs]

    r1_no = engine.generate_batch(
        prompts_8[1:2], system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512,
        mode="gpu", speculative=False, enable_ff_jump=False,
    )
    r1_jmp = engine.generate_batch(
        prompts_8[1:2], system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512,
        mode="gpu", speculative=False, enable_ff_jump=True,
    )
    assert r1_jmp.stats["ff_jump_steps"] > 0 and r1_jmp.stats["ff_jump_tokens"] > 0
    assert (r1_jmp.stats["llm_steps"] + r1_jmp.stats["ff_early14_steps"]) < (
        r1_no.stats["llm_steps"] + r1_no.stats["ff_early14_steps"]
    )
    assert r1_no.items[0].token_ids == r1_jmp.items[0].token_ids

    r8_roll = engine.generate_batch(
        prompts_8, system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512,
        mode="gpu", speculative=False, enable_ff_jump=True,
    )
    r8_spec = engine.generate_batch(
        prompts_8, system_prompt=sys_prompt, json_schema=schema_str, max_tokens=512,
        mode="gpu", speculative=True, batch_size=8, enable_ff_jump=True,
    )
    assert r8_roll.stats["valid_schema_count"] == 8
    assert r8_spec.stats["valid_schema_count"] == 8
    for a, b in zip(r8_roll.items, r8_spec.items):
        assert a.parsed is not None and b.parsed is not None
        assert a.parsed["status"] == b.parsed["status"]
        assert a.parsed.get("primary_kind") == b.parsed.get("primary_kind")
