"""Compile CoreML ANE `gemma4_early{14,23}_prefix_S256_{fp16,w8}.mlmodelc` bundles.

Uses per-output-channel symmetric `int8` (`cto.coreml.linear_quantize_weights` with
`mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=2048`)
and optional closed-form `AWQ-MSE` activation-weighted channel clipping (`--awq-mse`, zero training).

Hardware compiler invariants (`H16ANE` / `macOS 15+`):
  - NEVER use `palettize_weights(mode="kmeans", nbits=8)` (global per-tensor 256-entry LUT crushes K/V SNR by ~5 dB).
  - NEVER use `granularity="per_block"` (`E5RT: ANECCompile() FAILED` + `MLComputePlan` XPC hang).
  - NEVER mix `S=1` spatial slices inside the static `S=256` MIL attention graph (`ANECCompile() FAILED`).
  - Cut strictly at `k = first_kv_shared` (`EarlyExit14` `k=15` on `E2B`, `EarlyExit23` `k=24` on `E4B`).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterator, Sequence

import numpy as np

DEFAULT_ANE_CHUNK_S = 256
DEFAULT_ANE_P_SWA = 512
DEFAULT_ANE_P_FULL = 1920
AWQ_ALPHA_GRID: tuple[float, ...] = (0.86, 0.89, 0.92, 0.94, 0.96, 0.975, 0.99, 1.00)
GPU_LOCK_PATH = "/tmp/m4_gpu_validation.lock"


@contextmanager
def _gpu_validation_lock(lock_path: str = GPU_LOCK_PATH) -> Iterator[None]:
    with open(lock_path, "w", encoding="utf-8") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def compute_safe_fp16_residual_scales(first_kv_shared: int) -> np.ndarray:
    """Exact power-of-2 residual scaling per layer keeping Gemma-4 L10..L14+ residuals strictly <65,504 in FP16.

    Gemma-4 E2B/E4B residual norms peak around ~215K at L13 (> FP16 max 65,504). Scaling `cur_i' = alpha_i * cur_i`
    with `alpha_10=0.5, alpha_11=0.25, alpha_{>=12}=0.0625` brings 215K down to ~13.4K while preserving exact
    layer outputs (RMSNorm is scale-invariant up to `eps_eff = eps * alpha_i^2`, residual additions scale by
    `alpha_i`, `ple_gate` divides by `alpha_i`, and `layer_scalar` multiplies by `alpha_{i+1}/alpha_i`).
    """
    scales = np.ones((int(first_kv_shared),), dtype=np.float32)
    for i in range(int(first_kv_shared)):
        if i == 10:
            scales[i] = 0.5
        elif i == 11:
            scales[i] = 0.25
        elif i >= 12:
            scales[i] = 0.0625
    return scales


def fold_fp16_residual_scales(
    npz: dict[str, np.ndarray],
    residual_scales: Sequence[float] | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fold exact power-of-2 FP16 residual scales into an exported Early14/Early23 `.npz` (idempotent, pure NumPy)."""
    k = 0
    while f"l{k}_w_in" in npz:
        k += 1
    if k < 2:
        raise ValueError(f"Expected >= 2 EarlyExit layers in weights npz, found k={k}")

    if "fp16_residual_protected" in npz and bool(np.asarray(npz["fp16_residual_protected"]).item()):
        existing = (
            np.asarray(npz["residual_scales"], dtype=np.float32).reshape(-1)
            if "residual_scales" in npz
            else np.ones((k,), dtype=np.float32)
        )
        return dict(npz), existing

    scales = (
        compute_safe_fp16_residual_scales(k)
        if residual_scales is None
        else np.asarray(residual_scales, dtype=np.float32).reshape(-1)
    )
    if len(scales) != k:
        raise ValueError(f"residual_scales length {len(scales)} != first_kv_shared {k}")
    if float(scales[0]) != 1.0:
        raise ValueError(f"residual_scales[0] must be 1.0 (got {scales[0]}) so input h_in is unscaled")

    out = dict(npz)
    for i in range(k - 1):
        alpha_i = float(scales[i])
        alpha_next = float(scales[i + 1])
        p = f"l{i}_"
        if alpha_i != 1.0:
            for wk in ("w_pa", "w_poff", "w_pple"):
                if (p + wk) in out:
                    out[p + wk] = (out[p + wk].astype(np.float32) * alpha_i).astype(np.float16)
            pg_key = (p + "w_pg") if (p + "w_pg") in out else (p + "pg")
            if pg_key in out:
                out[pg_key] = (out[pg_key].astype(np.float32) / alpha_i).astype(np.float16)
        if (p + "scalar") in out and (alpha_next != alpha_i):
            out[p + "scalar"] = (out[p + "scalar"].astype(np.float32) * (alpha_next / alpha_i)).astype(np.float16)

    out["residual_scales"] = np.ascontiguousarray(scales, dtype=np.float32)
    out["fp16_residual_protected"] = np.asarray(1, dtype=np.int32)
    return out, scales


def infer_ane_arch_from_npz(npz: dict[str, np.ndarray]) -> dict[str, Any]:
    """Infer `(first_kv_shared, hidden_size, ple_dim, n_q_heads, n_kv, layer_specs)` directly from `.npz` shapes."""
    k = 0
    while f"l{k}_w_in" in npz:
        k += 1
    if k < 2:
        raise ValueError(f"Expected >= 2 EarlyExit layers in weights npz, found k={k}")

    hidden_size = int(npz["l0_w_in"].shape[1])
    ple_dim = int(npz["per_layer_proj_norm_w"].shape[0])
    n_pli_layers = k - 1
    if int(npz["per_layer_model_proj_w"].shape[0]) != n_pli_layers * ple_dim:
        raise ValueError(
            f"Mismatched per_layer_model_proj_w shape {npz['per_layer_model_proj_w'].shape} "
            f"for n_pli_layers={n_pli_layers}, ple_dim={ple_dim}"
        )

    n_q_heads = int(np.asarray(npz["n_q_heads"]).item()) if "n_q_heads" in npz else 8
    res_scales = (
        np.asarray(npz["residual_scales"], dtype=np.float32).reshape(-1)
        if "residual_scales" in npz
        else np.ones((k,), dtype=np.float32)
    )

    layers: list[dict[str, Any]] = []
    n_swa = 0
    n_full = 0
    n_kv_global: int | None = None
    for i in range(k):
        p = f"l{i}_"
        hd = int(npz[p + "w_kn"].shape[-1])
        is_full = hd == 512
        if hd not in (256, 512):
            raise ValueError(f"Unsupported head_dim={hd} at layer {i} (expected 256 SWA or 512 Full)")
        qdim = n_q_heads * hd
        if i == k - 1:
            kv_w = npz[p + "kv"] if (p + "kv") in npz else npz[p + "qkv"][qdim:]
            kv_out = int(kv_w.shape[0])
        else:
            qkv_out = int(npz[p + "qkv"].shape[0])
            kv_out = qkv_out - qdim
        if kv_out <= 0 or kv_out % (2 * hd) != 0:
            raise ValueError(f"Invalid KV output channels={kv_out} for head_dim={hd} at layer {i}")
        n_kv = kv_out // (2 * hd)
        if n_kv_global is None:
            n_kv_global = n_kv
        elif n_kv != n_kv_global:
            raise ValueError(f"Heterogeneous n_kv ({n_kv} vs {n_kv_global}) unsupported in packed ANE KV tensor")
        if is_full:
            n_full += 1
        else:
            n_swa += 1
        layers.append({"idx": i, "is_full": is_full, "hd": hd, "qdim": qdim, "n_kv": n_kv})

    assert n_kv_global is not None
    if n_full < 1:
        raise ValueError("Expected at least 1 full_attention layer in EarlyExit trunk")
    return {
        "first_kv_shared": k,
        "exit_layer": k - 1,
        "hidden_size": hidden_size,
        "ple_dim": ple_dim,
        "n_pli_layers": n_pli_layers,
        "n_swa": n_swa,
        "n_full": n_full,
        "n_full_past": n_full - 1,
        "n_q_heads": n_q_heads,
        "n_kv": n_kv_global,
        "residual_scales": res_scales.tolist(),
        "fp16_residual_protected": bool(
            "fp16_residual_protected" in npz and int(np.asarray(npz["fp16_residual_protected"]).item()) != 0
        ),
        "layers": layers,
    }


def load_calibration_texts(
    *,
    calib_texts: Sequence[str] | None = None,
    calib_file: str | Path | None = None,
    max_docs: int = 32,
    repo_root: Path | None = None,
) -> tuple[list[str], str, str | None]:
    """Load real domain calibration texts from list, `.txt`/`.json`/`.jsonl`/`.parquet`, or repo fixtures."""
    if calib_texts:
        docs = [str(t) for t in calib_texts if str(t).strip()][: max(1, int(max_docs))]
        if docs:
            return docs, "corpus", None

    if calib_file is not None:
        p = Path(calib_file)
        if not p.is_file():
            raise FileNotFoundError(f"Calibration file not found: {p}")
        suf = p.suffix.lower()
        docs = []
        if suf == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    docs.append(obj)
                elif isinstance(obj, dict):
                    for key in ("text", "prompt", "content", "document"):
                        if key in obj and obj[key]:
                            docs.append(str(obj[key]))
                            break
                if len(docs) >= max_docs:
                    break
        elif suf == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                for item in obj[:max_docs]:
                    if isinstance(item, str):
                        docs.append(item)
                    elif isinstance(item, list) and len(item) >= 4:
                        docs.append(f"Known metadata: {item[2]}\nFilename: {item[1]}\nText:\n{item[3]}")
                    elif isinstance(item, dict):
                        for key in ("text", "prompt", "content", "document"):
                            if key in item and item[key]:
                                docs.append(str(item[key]))
                                break
        elif suf == ".parquet":
            import pyarrow.parquet as pq

            tbl = pq.read_table(p)
            col_name = next((c for c in ("text", "content", "prompt", "document") if c in tbl.column_names), tbl.column_names[0])
            docs = [str(v) for v in tbl[col_name].to_pylist()[:max_docs] if v]
        else:
            raw = p.read_text(encoding="utf-8")
            chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
            docs = chunks[:max_docs] if chunks else [raw]
        if not docs:
            raise ValueError(f"Calibration file {p} yielded 0 non-empty documents")
        return docs, "file", None

    root = repo_root or Path(__file__).resolve().parents[1]
    fix_docs = root / "tests" / "fixtures" / "sample_docs_8.json"
    fix_sys = root / "tests" / "fixtures" / "sys_extract_v1.txt"
    if fix_docs.is_file():
        arr = json.loads(fix_docs.read_text(encoding="utf-8"))
        docs = [f"Known metadata: {known}\nFilename: {fn}\nText:\n{txt}" for _, fn, known, txt in arr[:max_docs]]
        sys_txt = fix_sys.read_text(encoding="utf-8").strip() if fix_sys.is_file() else None
        if docs:
            return docs, "fixtures", sys_txt

    synth = [
        f"Pacific Transmission Consortium Technical Report #{c_i}: Engineering assessment regarding "
        "substation thermal ratings, battery inverter stability, statutory compliance deadlines, and grid reliability. " * 8
        for c_i in range(min(8, max(1, int(max_docs))))
    ]
    return synth, "synthetic", None


def write_ane_calibration_manifest(
    out_ane_dir: Path,
    *,
    bundle_name: str,
    variant: str,
    arch: dict[str, Any],
    compute_precision_mode: str,
    awq_mse: bool = False,
    mean_alpha: float = 1.0,
    calibration_source: str = "none",
    calibration_docs: int = 0,
    qat_validated: bool = False,
    checkpoint_key: str | None = None,
) -> Path:
    """Write/update `ane_calibration_manifest.json` alongside compiled `.mlmodelc` bundles."""
    manifest_p = out_ane_dir / "ane_calibration_manifest.json"
    data: dict[str, Any] = {"schema_version": 1, "bundles": {}}
    if manifest_p.is_file():
        try:
            loaded = json.loads(manifest_p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("bundles"), dict):
                data = loaded
        except Exception:
            pass
    data["bundles"][bundle_name] = {
        "bundle_name": bundle_name,
        "variant": variant,
        "exit_layer": int(arch["exit_layer"]),
        "n_kv": int(arch["n_kv"]),
        "compute_precision_mode": compute_precision_mode,
        "fp16_residual_protected": bool(arch.get("fp16_residual_protected", True)),
        "residual_scales": arch.get("residual_scales", []),
        "awq_mse": bool(awq_mse),
        "mean_alpha": round(float(mean_alpha), 5),
        "calibration_source": calibration_source,
        "calibration_docs": int(calibration_docs),
        "qat_validated": bool(qat_validated),
        "checkpoint_ane_key": checkpoint_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_p


def collect_awq_calibration_col_sq(
    model: Any,
    *,
    chunk_s: int = DEFAULT_ANE_CHUNK_S,
    calib_texts: Sequence[str] | None = None,
    calib_file: str | Path | None = None,
    system_prompt: str | None = None,
    max_docs: int = 32,
    return_meta: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect per-input-channel `E[x^2]` across real domain calibration chunks on GPU (zero training)."""
    import mlx.core as mx

    from fast_gemma4.kernels import (
        fused_geglu_split,
        fused_norm_add_norm,
        fused_ple_geglu_indexed,
        fused_qkv_packed_norm_rope,
        fused_res_rms_norm,
    )
    from fast_gemma4.model import _select_sdpa_mask
    from fast_gemma4.tokenizer import FastGemma4Tokenizer

    tok = FastGemma4Tokenizer(model.tokenizer_json)
    docs, calib_src, default_sys = load_calibration_texts(
        calib_texts=calib_texts, calib_file=calib_file, max_docs=max_docs
    )
    eff_sys = system_prompt if system_prompt is not None else default_sys
    if eff_sys:
        sys_ids = [tok.bos_token_id] + tok.encode(eff_sys, add_special_tokens=False)
    else:
        sys_ids = [tok.bos_token_id] + (
            tok.encode("Extract JSON metadata for this technical document:\n\n", add_special_tokens=False) * 20
        )
    sys_ids = sys_ids[: min(600, max(16, len(sys_ids)))]
    pkv = model.prefill_prefix(sys_ids)

    sufs = np.zeros((len(docs), chunk_s), dtype=np.int32)
    for idx, txt in enumerate(docs):
        enc = tok.encode(txt, add_special_tokens=False)[:chunk_s]
        sufs[idx, : len(enc)] = enc

    tok_mx = mx.array(sufs, dtype=mx.int32)
    B, S = int(tok_mx.shape[0]), int(tok_mx.shape[1])
    l_sys = int(pkv["l_sys"])
    k_sys_l, v_sys_l = pkv["k_sys"], pkv["v_sys"]
    off = l_sys
    off_meta = mx.array([off, S], dtype=mx.int32)
    col_sq: dict[str, np.ndarray] = {}

    def _rec(name: str, x: mx.array) -> None:
        xf = x.astype(mx.float32).reshape(-1, x.shape[-1])
        msq = mx.mean(xf * xf, axis=0)
        mx.eval(msq)
        col_sq[name] = np.array(msq, copy=True)

    h, pli = model._compute_embed_and_pli_early14(tok_mx)
    _rec("per_layer_model_proj_w", h)
    x_in = mx.fast.rms_norm(h, model.layers[0]["w_in"], model.eps)
    masks: dict[tuple[str, int, int], Any] = {}
    n_q = int(getattr(model, "n_q_heads", 8))
    for i in range(model.first_kv_shared):
        l = model.layers[i]
        p = f"l{i}_"
        D = int(l["head_dim"])
        n_kv_l = int(l.get("n_kv", getattr(model, "n_kv_heads", 1)))
        if i == model.first_kv_shared - 1:
            _rec(p + "kv", x_in)
            break
        _rec(p + "qkv", x_in)
        qkv = l["qkv_proj"](x_in)
        q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
            qkv,
            l["w_qn"],
            l["w_kn"],
            l["inv_freq"],
            off_meta,
            model.eps,
            n_q=n_q,
            n_kv=n_kv_l,
        )
        # k_rot/v_norm: (B, S, n_kv, D) -> transpose to (B, n_kv, S, D); k_sys_l[i]: (l_sys, n_kv, D)
        ks_i = k_sys_l[i]
        vs_i = v_sys_l[i]
        if ks_i.ndim == 2:
            ks_i = ks_i[:, None, :]
            vs_i = vs_i[:, None, :]
        k_rot_t = k_rot.transpose(0, 2, 1, 3) if k_rot.ndim == 4 else k_rot[:, None]
        v_norm_t = v_norm.transpose(0, 2, 1, 3) if v_norm.ndim == 4 else v_norm[:, None]

        if l["is_full"]:
            ks_b = mx.broadcast_to(ks_i.transpose(1, 0, 2)[None], (B, n_kv_l, l_sys, D))
            vs_b = mx.broadcast_to(vs_i.transpose(1, 0, 2)[None], (B, n_kv_l, l_sys, D))
            kf = mx.concatenate([ks_b, k_rot_t], axis=2)
            vf = mx.concatenate([vs_b, v_norm_t], axis=2)
            mk = ("full", off, S)
            if mk not in masks:
                masks[mk] = _select_sdpa_mask(S, offset=off, window_size=None, dtype=model.act_dtype)
        else:
            t_start = max(0, off - model.window_size + 1)
            ks_b = mx.broadcast_to(ks_i[t_start:l_sys].transpose(1, 0, 2)[None], (B, n_kv_l, l_sys - t_start, D))
            vs_b = mx.broadcast_to(vs_i[t_start:l_sys].transpose(1, 0, 2)[None], (B, n_kv_l, l_sys - t_start, D))
            kf = mx.concatenate([ks_b, k_rot_t], axis=2)
            vf = mx.concatenate([vs_b, v_norm_t], axis=2)
            mk = ("swa", off - t_start, S)
            if mk not in masks:
                masks[mk] = _select_sdpa_mask(
                    S, offset=(off - t_start), window_size=model.window_size, dtype=model.act_dtype
                )
        att = (
            mx.fast.scaled_dot_product_attention(q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk])
            .transpose(0, 2, 1, 3)
            .reshape(B, S, n_q * D)
        )
        _rec(p + "wo", att)
        ao = l["o_proj"](att)
        h_mid, mlp_in = fused_norm_add_norm(h, ao, l["w_post_attn"], l["w_pre_ff"], model.eps, None)
        _rec(p + "gu", mlp_in)
        mlp_act = fused_geglu_split(l["gate_up_proj"](mlp_in))
        _rec(p + "dn", mlp_act)
        mlp_out = model._run_mlp(mlp_in, l)
        h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], model.eps)
        _rec(p + "w_pg", h_ff)
        ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, model._layer_idx_mx[i])
        _rec(p + "w_pp", ple_g)
        ple_out = l["ple_proj"](ple_g)
        h, x_in = fused_norm_add_norm(
            h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], model.eps, l["layer_scalar"]
        )
    if return_meta:
        return col_sq, {"calibration_source": calib_src, "calibration_docs": len(docs)}
    return col_sq


def apply_awq_mse_clipping(
    npz: dict[str, np.ndarray],
    act_col_sq: dict[str, np.ndarray],
    *,
    alphas: tuple[float, ...] = AWQ_ALPHA_GRID,
) -> tuple[dict[str, np.ndarray], float]:
    """Closed-form per-output-channel symmetric int8 AWQ-MSE clipping (`0` gradient/training steps)."""
    clipped = dict(npz)
    alpha_means: list[float] = []
    for key, w_raw in npz.items():
        if key not in act_col_sq or w_raw.ndim != 4:
            continue
        w = w_raw[:, :, 0, 0].astype(np.float32)
        col_w = act_col_sq[key].astype(np.float32)[None, :]
        if col_w.shape[1] != w.shape[1]:
            continue
        amax = np.maximum(np.max(np.abs(w), axis=1, keepdims=True), 1e-8)
        best_alpha = np.ones_like(amax)
        best_err = np.full((w.shape[0], 1), np.inf, dtype=np.float32)
        for alpha in alphas:
            sc = (float(alpha) * amax) / 127.0
            wq = np.clip(np.round(w / sc), -127.0, 127.0) * sc
            diff = w - wq
            err = np.sum((diff * diff) * col_w, axis=1, keepdims=True)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_alpha = np.where(better, float(alpha), best_alpha)
        lim = best_alpha * amax
        w_clipped = np.clip(w, -lim, lim).astype(np.float16)
        clipped[key] = np.ascontiguousarray(w_clipped)[:, :, None, None]
        alpha_means.append(float(np.mean(best_alpha)))
    mean_alpha = float(np.mean(alpha_means)) if alpha_means else 1.0
    return clipped, mean_alpha


def build_early_prefix_mil_program(
    npz: dict[str, np.ndarray],
    *,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
    protect_fp16_residuals: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Build CoreML MIL `EarlyExit{k-1}` program (`k = first_kv_shared`: `15` on E2B, `24` on E4B).

    Protects FP16 execution around Gemma-4's L10..L14+ residual peaks (~215K > FP16 max 65,504) via:
      1. Exact power-of-2 residual-scale folding (`fold_fp16_residual_scales`) keeping residual magnitudes < 16K.
      2. Scale-normalized RMSNorm (`x / max(|x|, 1e-4)` before squaring) eliminating `x^2` FP16 overflow for |x|>256.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if S <= 1:
        raise ValueError(f"H16ANE requires static chunk S >= 16 (got S={S}); S=1 slices fail ANECCompile().")

    if protect_fp16_residuals:
        npz, res_scales = fold_fp16_residual_scales(npz)
    else:
        res_scales = (
            np.asarray(npz["residual_scales"], dtype=np.float32).reshape(-1)
            if "residual_scales" in npz
            else np.ones((sum(1 for i in range(64) if f"l{i}_w_in" in npz),), dtype=np.float32)
        )

    arch = infer_ane_arch_from_npz(npz)
    k = arch["first_kv_shared"]
    H = arch["hidden_size"]
    ple_dim = arch["ple_dim"]
    n_pli = arch["n_pli_layers"]
    n_swa = arch["n_swa"]
    n_full_past = arch["n_full_past"]
    n_q = arch["n_q_heads"]
    n_kv = arch["n_kv"]
    layers = arch["layers"]

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, 1, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n_pli * ple_dim, 1, S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(n_swa, n_kv, P_SWA, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(n_swa, n_kv, P_SWA, 256), dtype=types.fp16),
            mb.TensorSpec(shape=(n_full_past, n_kv, P_FULL, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(n_full_past, n_kv, P_FULL, 512), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, P_SWA + S), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1, S, P_FULL + S), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(
        h_in,
        pli_raw,
        cos_swa,
        sin_swa,
        cos_full,
        sin_full,
        k_past_swa,
        v_past_swa,
        k_past_full,
        v_past_full,
        mask_swa,
        mask_full,
    ):
        def _rmsnorm_safe(x, w, axis: int, prefix: str, sqrt_eps: float = 1e-3):
            # Scale by max(|x|, 1e-4) before squaring so `xs^2 <= 1.0` never overflows FP16 even when |x| > 256.
            ax = [axis]
            s_max = mb.maximum(
                x=mb.reduce_max(x=mb.abs(x=x, name=prefix + "_abs"), axes=ax, keep_dims=True, name=prefix + "_rmax"),
                y=np.float16(1e-4),
                name=prefix + "_s",
            )
            xs = mb.real_div(x=x, y=s_max, name=prefix + "_xs")
            msq = mb.reduce_mean(x=mb.mul(x=xs, y=xs, name=prefix + "_sq"), axes=ax, keep_dims=True, name=prefix + "_msq")
            eps_s = mb.real_div(x=np.float16(sqrt_eps), y=s_max, name=prefix + "_es")
            eps_sq = mb.mul(x=eps_s, y=eps_s, name=prefix + "_esq")
            rsq = mb.rsqrt(x=mb.add(x=msq, y=eps_sq, name=prefix + "_ae"), name=prefix + "_rsq")
            xn = mb.mul(x=xs, y=rsq, name=prefix + "_n")
            return mb.mul(x=xn, y=w, name=prefix + "_w") if w is not None else xn

        def _rmsnorm_ch(x, w1xc, prefix: str, sqrt_eps: float = 1e-3):
            return _rmsnorm_safe(x, w1xc, 1, prefix, sqrt_eps=sqrt_eps)

        def _rmsnorm_last(x, w, prefix: str, sqrt_eps: float = 1e-3):
            return _rmsnorm_safe(x, w, -1, prefix, sqrt_eps=sqrt_eps)

        def _rope(x, c, s, half_d: int, prefix: str):
            x1, x2 = mb.split(x=x, split_sizes=[half_d, half_d], axis=-1, name=prefix + "_sp")
            rot = mb.concat(
                values=[mb.mul(x=x2, y=np.float16(-1.0), name=prefix + "_neg"), x1], axis=-1, name=prefix + "_cat"
            )
            return mb.add(
                x=mb.mul(x=x, y=c, name=prefix + "_xc"),
                y=mb.mul(x=rot, y=s, name=prefix + "_rs"),
                name=prefix + "_out",
            )

        plm = mb.mul(
            x=mb.conv(x=h_in, weight=npz["per_layer_model_proj_w"], name="plm"),
            y=npz["per_layer_proj_scale"][0],
            name="plms",
        )
        plm_r = mb.reshape(x=plm, shape=[1, n_pli, ple_dim, S], name="plmr")
        pl_nw = _rmsnorm_safe(plm_r, npz["per_layer_proj_norm_w"].reshape(1, 1, ple_dim, 1), 2, "pl")
        pl_add = mb.add(x=mb.reshape(x=pl_nw, shape=[1, n_pli * ple_dim, 1, S], name="plnw_f"), y=pli_raw, name="pladd")
        pli_all = mb.mul(x=pl_add, y=npz["per_layer_input_scale"][0], name="plall")
        plis = mb.split(x=pli_all, split_sizes=[ple_dim] * n_pli, axis=1, name="plisp")

        k_past_swa_list = (
            [k_past_swa]
            if n_swa == 1
            else list(mb.split(x=k_past_swa, split_sizes=[1] * n_swa, axis=0, name="kps_sp"))
        )
        v_past_swa_list = (
            [v_past_swa]
            if n_swa == 1
            else list(mb.split(x=v_past_swa, split_sizes=[1] * n_swa, axis=0, name="vps_sp"))
        )
        k_past_full_list = (
            [k_past_full]
            if n_full_past == 1
            else list(mb.split(x=k_past_full, split_sizes=[1] * n_full_past, axis=0, name="kpf_sp"))
        )
        v_past_full_list = (
            [v_past_full]
            if n_full_past == 1
            else list(mb.split(x=v_past_full, split_sizes=[1] * n_full_past, axis=0, name="vpf_sp"))
        )

        cur = h_in
        k_swas, v_swas, k_fulls, v_fulls = [], [], [], []
        swa_idx = 0
        full_idx = 0

        for lspec in layers:
            i = lspec["idx"]
            p = f"l{i}_"
            is_full = lspec["is_full"]
            hd = lspec["hd"]
            qdim = lspec["qdim"]
            kv_dim = n_kv * hd
            res_sqrt_eps = max(1e-4, 1e-3 * float(res_scales[i]))
            in_n = _rmsnorm_ch(cur, npz[p + "w_in"], p + "in", sqrt_eps=res_sqrt_eps)

            if i == k - 1:
                kv_w = npz[p + "kv"] if (p + "kv") in npz else npz[p + "qkv"][qdim:]
                kv = mb.conv(x=in_n, weight=kv_w, name=p + "kv")
                k_t, v_t = mb.split(x=kv, split_sizes=[kv_dim, kv_dim], axis=1, name=p + "sp")
            else:
                qkv = mb.conv(x=in_n, weight=npz[p + "qkv"], name=p + "qkv")
                q, k_t, v_t = mb.split(x=qkv, split_sizes=[qdim, kv_dim, kv_dim], axis=1, name=p + "sp")

            kr = mb.transpose(
                x=mb.reshape(x=k_t, shape=[1, n_kv, hd, S], name=p + "kr"), perm=[0, 1, 3, 2], name=p + "kh"
            )
            vr = mb.transpose(
                x=mb.reshape(x=v_t, shape=[1, n_kv, hd, S], name=p + "vr"), perm=[0, 1, 3, 2], name=p + "vh"
            )
            kn = _rope(
                _rmsnorm_last(kr, npz[p + "w_kn"], p + "kn"),
                cos_full if is_full else cos_swa,
                sin_full if is_full else sin_swa,
                hd // 2,
                p + "rk",
            )
            vn = _rmsnorm_last(vr, None, p + "vn")
            if is_full:
                k_fulls.append(kn)
                v_fulls.append(vn)
            else:
                k_swas.append(kn)
                v_swas.append(vn)
            if i == k - 1:
                break

            if is_full:
                k_tot = mb.concat(values=[k_past_full_list[full_idx], kn], axis=2, name=p + "kcat")
                v_tot = mb.concat(values=[v_past_full_list[full_idx], vn], axis=2, name=p + "vcat")
                m_use = mask_full
                full_idx += 1
            else:
                k_tot = mb.concat(values=[k_past_swa_list[swa_idx], kn], axis=2, name=p + "kcat")
                v_tot = mb.concat(values=[v_past_swa_list[swa_idx], vn], axis=2, name=p + "vcat")
                m_use = mask_swa
                swa_idx += 1

            qr = mb.transpose(
                x=mb.reshape(
                    x=mb.transpose(x=q, perm=[0, 2, 3, 1], name=p + "qt"),
                    shape=[1, S, n_q, hd],
                    name=p + "qr",
                ),
                perm=[0, 2, 1, 3],
                name=p + "qh",
            )
            qn = _rope(
                _rmsnorm_last(qr, npz[p + "w_qn"], p + "qn"),
                cos_full if is_full else cos_swa,
                sin_full if is_full else sin_swa,
                hd // 2,
                p + "rq",
            )
            if n_kv == 1:
                sc = mb.add(
                    x=mb.matmul(x=qn, y=mb.transpose(x=k_tot, perm=[0, 1, 3, 2], name=p + "kt"), name=p + "qk"),
                    y=m_use,
                    name=p + "m",
                )
                att = mb.matmul(x=mb.softmax(x=sc, axis=-1, name=p + "sm"), y=v_tot, name=p + "av")
            else:
                q_per_kv = n_q // n_kv
                q_groups = mb.split(x=qn, split_sizes=[q_per_kv] * n_kv, axis=1, name=p + "qg_sp")
                k_groups = mb.split(x=k_tot, split_sizes=[1] * n_kv, axis=1, name=p + "kg_sp")
                v_groups = mb.split(x=v_tot, split_sizes=[1] * n_kv, axis=1, name=p + "vg_sp")
                att_groups = []
                for g_i in range(n_kv):
                    gp = f"{p}g{g_i}_"
                    sc_g = mb.add(
                        x=mb.matmul(
                            x=q_groups[g_i],
                            y=mb.transpose(x=k_groups[g_i], perm=[0, 1, 3, 2], name=gp + "kt"),
                            name=gp + "qk",
                        ),
                        y=m_use,
                        name=gp + "m",
                    )
                    att_groups.append(
                        mb.matmul(x=mb.softmax(x=sc_g, axis=-1, name=gp + "sm"), y=v_groups[g_i], name=gp + "av")
                    )
                att = mb.concat(values=att_groups, axis=1, name=p + "av_cat")

            att_ch = mb.transpose(
                x=mb.reshape(
                    x=mb.transpose(x=att, perm=[0, 2, 1, 3], name=p + "at"),
                    shape=[1, 1, S, qdim],
                    name=p + "ar",
                ),
                perm=[0, 3, 1, 2],
                name=p + "ac",
            )
            ao_n = _rmsnorm_ch(mb.conv(x=att_ch, weight=npz[p + "wo"], name=p + "wo"), npz[p + "w_pa"], p + "pa")
            c1 = mb.add(x=cur, y=ao_n, name=p + "res1")
            pff = _rmsnorm_ch(c1, npz[p + "w_pff"], p + "pff", sqrt_eps=res_sqrt_eps)
            g, u = mb.split(x=mb.conv(x=pff, weight=npz[p + "gu"], name=p + "gu"), num_splits=2, axis=1, name=p + "spgu")
            md = mb.conv(
                x=mb.mul(x=mb.gelu(x=g, mode="TANH_APPROXIMATION", name=p + "gelu"), y=u, name=p + "geglu"),
                weight=npz[p + "dn"],
                name=p + "dn",
            )
            c2 = mb.add(x=c1, y=_rmsnorm_ch(md, npz[p + "w_poff"], p + "poff"), name=p + "res2")
            w_pg = npz[p + "w_pg"] if (p + "w_pg") in npz else npz[p + "pg"]
            w_pp = npz[p + "w_pp"] if (p + "w_pp") in npz else npz[p + "pp"]
            pg = mb.conv(x=c2, weight=w_pg, name=p + "pg")
            pge = mb.mul(x=mb.gelu(x=pg, mode="TANH_APPROXIMATION", name=p + "pgelu"), y=plis[i], name=p + "pmul")
            cur = mb.mul(
                x=mb.add(
                    x=c2,
                    y=_rmsnorm_ch(mb.conv(x=pge, weight=w_pp, name=p + "pp"), npz[p + "w_pple"], p + "pple"),
                    name=p + "res3",
                ),
                y=npz[p + "scalar"][0],
                name=p + "sc",
            )

        return (
            mb.identity(x=mb.concat(values=k_swas, axis=0, name="ks_c"), name="k_swa"),
            mb.identity(x=mb.concat(values=v_swas, axis=0, name="vs_c"), name="v_swa"),
            mb.identity(x=mb.concat(values=k_fulls, axis=0, name="kf_c"), name="k_full"),
            mb.identity(x=mb.concat(values=v_fulls, axis=0, name="vf_c"), name="v_full"),
        )

    return prog, arch


def quantize_mlmodel_linear_w8_per_channel(ml_fp16: Any, *, weight_threshold: int = 2048) -> Any:
    """Quantize CoreML MLProgram conv1x1 weights to per-output-channel symmetric `int8` (`100%` H16ANE compatible)."""
    import coremltools.optimize as cto

    return cto.coreml.linear_quantize_weights(
        ml_fp16,
        config=cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",
                weight_threshold=weight_threshold,
            )
        ),
    )


def _inspect_compiled_bundle(
    lib_path: Path,
    mc_path: Path,
    *,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
) -> dict[str, int] | None:
    if not lib_path.exists():
        return None
    lib = ctypes.CDLL(str(lib_path))
    if not hasattr(lib, "gemma4_px_inspect_plan_ex"):
        return None
    mc_bytes = str(mc_path).encode("utf-8")
    # Warm-load once so ANECompilerService populates the E5RT cache before the 8s MLComputePlan guard.
    if hasattr(lib, "gemma4_px_load") and hasattr(lib, "gemma4_px_free"):
        lib.gemma4_px_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.gemma4_px_load.restype = ctypes.c_void_p
        lib.gemma4_px_free.argtypes = [ctypes.c_void_p]
        lib.gemma4_px_free.restype = None
        h = lib.gemma4_px_load(mc_bytes, int(S), int(P_SWA), int(P_FULL), 1)
        if not h:
            raise RuntimeError(f"Failed to load compiled CoreML bundle on ANE: {mc_path}")
        lib.gemma4_px_free(h)

    i32p = ctypes.POINTER(ctypes.c_int)
    lib.gemma4_px_inspect_plan_ex.argtypes = [ctypes.c_char_p, ctypes.c_int, i32p, i32p, i32p, i32p]
    lib.gemma4_px_inspect_plan_ex.restype = ctypes.c_int
    ane = ctypes.c_int(0)
    cpu_rt = ctypes.c_int(0)
    cpu_ce = ctypes.c_int(0)
    gpu = ctypes.c_int(0)
    rc = -1
    for attempt in range(2):
        rc = lib.gemma4_px_inspect_plan_ex(
            mc_bytes,
            0,
            ctypes.byref(ane),
            ctypes.byref(cpu_rt),
            ctypes.byref(cpu_ce),
            ctypes.byref(gpu),
        )
        if rc >= 0:
            break
        if attempt == 0:
            time.sleep(2.0)
    if rc < 0:
        raise RuntimeError(f"MLComputePlan inspection failed or timed out for {mc_path}")
    return {
        "ane_ops": ane.value,
        "cpu_runtime_ops": cpu_rt.value,
        "cpu_constexpr_ops": cpu_ce.value,
        "gpu_ops": gpu.value,
    }


def compile_mlpackage_in_process(pkg_path: Path, dst_mc: Path) -> Path:
    """Compile a saved `.mlpackage` to `dst_mc` via `coremltools.utils.compile_model` (no Xcode `coremlcompiler` CLI)."""
    import coremltools as ct

    dst_mc.parent.mkdir(parents=True, exist_ok=True)
    compiled_tmp = Path(ct.utils.compile_model(str(pkg_path)))
    if compiled_tmp.resolve() == dst_mc.resolve():
        return dst_mc
    if dst_mc.exists():
        shutil.rmtree(dst_mc)
    try:
        shutil.move(str(compiled_tmp), str(dst_mc))
    except Exception:
        shutil.rmtree(compiled_tmp, ignore_errors=True)
        raise
    return dst_mc


def compile_ane_bundles_from_npz(
    *,
    fp16_npz_path: Path,
    w8_npz_path: Path,
    out_ane_dir: Path,
    variant: str,
    keep_mlpackage: bool = False,
    lib_ane_prefix: Path | None = None,
    S: int = DEFAULT_ANE_CHUNK_S,
    P_SWA: int = DEFAULT_ANE_P_SWA,
    P_FULL: int = DEFAULT_ANE_P_FULL,
    compute_precision_mode: str = "fp16_safe",
    manifest_meta: dict[str, Any] | None = None,
) -> list[Path]:
    import coremltools as ct

    if compute_precision_mode not in ("fp16_safe", "fp32_norms", "fp32"):
        raise ValueError(f"Unsupported compute_precision_mode={compute_precision_mode!r}")

    out_ane_dir.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []
    meta = dict(manifest_meta or {})

    def _resolve_precision() -> Any:
        if compute_precision_mode == "fp32":
            return ct.precision.FLOAT32
        if compute_precision_mode == "fp32_norms":
            fp16_cls = getattr(getattr(ct, "transform", None), "FP16ComputePrecision", None)
            if fp16_cls is not None:
                return fp16_cls(
                    op_selector=lambda op: getattr(op, "op_type", "")
                    not in ("reduce_mean", "rsqrt", "softmax", "gelu", "real_div")
                )
        return ct.precision.FLOAT16

    def _compile(model_obj: Any, name_stem: str) -> Path:
        pkg = out_ane_dir / f"{name_stem}.mlpackage"
        mc = out_ane_dir / f"{name_stem}.mlmodelc"
        if pkg.exists():
            shutil.rmtree(pkg)
        if mc.exists():
            shutil.rmtree(mc)
        model_obj.save(str(pkg))
        compile_mlpackage_in_process(pkg, mc)
        if not keep_mlpackage and pkg.exists():
            shutil.rmtree(pkg)
        if lib_ane_prefix is not None and compute_precision_mode == "fp16_safe":
            plan_info = _inspect_compiled_bundle(lib_ane_prefix, mc, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
            if plan_info is not None:
                print(f"[MLComputePlan] {mc.name}: {plan_info}")
                if plan_info["cpu_runtime_ops"] > 0:
                    raise RuntimeError(f"Unexpected CPU runtime fallback ops in {mc}: {plan_info}")
        return mc

    ct_precision = _resolve_precision()

    if variant in ("both", "fp16"):
        npz_f16 = dict(np.load(fp16_npz_path))
        prog_fp16, arch = build_early_prefix_mil_program(npz_f16, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
        stem_prefix = f"gemma4_early{arch['exit_layer']}_prefix_S{S}"
        ml_fp16 = ct.convert(
            prog_fp16,
            convert_to="mlprogram",
            compute_precision=ct_precision,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        mc = _compile(ml_fp16, f"{stem_prefix}_fp16")
        write_ane_calibration_manifest(
            out_ane_dir,
            bundle_name=mc.name,
            variant="fp16",
            arch=arch,
            compute_precision_mode=compute_precision_mode,
            checkpoint_key=meta.get("checkpoint_ane_key"),
        )
        print(f"Compiled {mc}")
        built.append(mc)

    if variant in ("both", "w8"):
        npz_w8 = dict(np.load(w8_npz_path))
        prog_w8, arch = build_early_prefix_mil_program(npz_w8, S=S, P_SWA=P_SWA, P_FULL=P_FULL)
        stem_prefix = f"gemma4_early{arch['exit_layer']}_prefix_S{S}"
        ml_for_w8 = ct.convert(
            prog_w8,
            convert_to="mlprogram",
            compute_precision=ct_precision,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        ml_w8 = quantize_mlmodel_linear_w8_per_channel(ml_for_w8, weight_threshold=2048)
        mc8 = _compile(ml_w8, f"{stem_prefix}_w8")
        write_ane_calibration_manifest(
            out_ane_dir,
            bundle_name=mc8.name,
            variant="w8",
            arch=arch,
            compute_precision_mode=compute_precision_mode,
            awq_mse=bool(meta.get("awq_mse", False)),
            mean_alpha=float(meta.get("mean_alpha", 1.0)),
            calibration_source=str(meta.get("calibration_source", "none")),
            calibration_docs=int(meta.get("calibration_docs", 0)),
            qat_validated=bool(meta.get("qat_validated", False)),
            checkpoint_key=meta.get("checkpoint_ane_key"),
        )
        print(f"Compiled {mc8}")
        built.append(mc8)

    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CoreML ANE Early14/Early23 prefix chunk bundles (.mlmodelc)")
    parser.add_argument("--home", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--variant", choices=("both", "fp16", "w8"), default="both")
    parser.add_argument(
        "--compute-precision",
        choices=("fp16_safe", "fp32_norms", "fp32"),
        default="fp16_safe",
        help="CoreML precision policy (`fp16_safe` folds L10..L14 residual scales + safe RMSNorm for 100% ANE placement)",
    )
    parser.add_argument(
        "--awq-mse",
        action="store_true",
        help="Apply closed-form activation-weighted MSE per-channel weight clipping before linear int8 quantization",
    )
    parser.add_argument("--calib-file", default=None, help="Domain calibration corpus (.txt/.json/.jsonl/.parquet)")
    parser.add_argument("--calib-system-file", default=None, help="Optional system prompt file for AWQ calibration")
    parser.add_argument("--calib-max-docs", type=int, default=32, help="Max calibration documents for AWQ-MSE")
    parser.add_argument(
        "--mark-qat-validated",
        action="store_true",
        help="Certify W8 bundle in ane_calibration_manifest.json as QAT-validated",
    )
    parser.add_argument("--reexport", action="store_true", help="Re-export FP16 EarlyExit weights npz from model")
    parser.add_argument("--keep-mlpackage", action="store_true")
    parser.add_argument("--compile-only-fp16-npz", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-w8-npz", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-out-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-lib", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only-meta-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.compile_only_out_dir is not None:
        fp16_p = Path(args.compile_only_fp16_npz)
        w8_p = Path(args.compile_only_w8_npz)
        lib_p = Path(args.compile_only_lib) if args.compile_only_lib else None
        meta_obj = json.loads(args.compile_only_meta_json) if args.compile_only_meta_json else None
        compile_ane_bundles_from_npz(
            fp16_npz_path=fp16_p,
            w8_npz_path=w8_p,
            out_ane_dir=Path(args.compile_only_out_dir),
            variant=args.variant,
            keep_mlpackage=args.keep_mlpackage,
            lib_ane_prefix=lib_p,
            compute_precision_mode=args.compute_precision,
            manifest_meta=meta_obj,
        )
        return 0

    from fast_gemma4.ane_pipeline import ANE_CHUNK_S, ANE_P_FULL, ANE_P_SWA
    from fast_gemma4.paths import resolve_paths

    paths = resolve_paths(args.home)
    weights_npz = paths.resolve_early14_weights_npz(args.model_dir)
    act_col_sq: dict[str, np.ndarray] | None = None
    calib_meta: dict[str, Any] = {
        "awq_mse": bool(args.awq_mse),
        "mean_alpha": 1.0,
        "calibration_source": "none",
        "calibration_docs": 0,
        "qat_validated": bool(args.mark_qat_validated),
        "checkpoint_ane_key": (
            None if paths._is_default_model_dir(args.model_dir) else paths.checkpoint_ane_key(Path(args.model_dir))
        ),
    }

    need_model = args.reexport or (not weights_npz.exists()) or (args.awq_mse and args.variant in ("both", "w8"))
    if need_model:
        import mlx.core as mx
        from fast_shared.hf import copy_tokenizer_files

        from fast_gemma4.model import FastGemma4Model

        sys_prompt_calib = (
            Path(args.calib_system_file).read_text(encoding="utf-8").strip() if args.calib_system_file else None
        )
        with _gpu_validation_lock():
            t0 = time.perf_counter()
            model = FastGemma4Model(paths=paths, model_dir=args.model_dir, mlp_dtype="bfloat16", use_mpp=False)
            weights_npz = paths.resolve_early14_weights_npz(args.model_dir, exit_layer=model.first_kv_shared - 1)
            if not (paths.tokenizer_dir / "tokenizer.json").exists():
                copy_tokenizer_files(model.model_dir, paths.tokenizer_dir)
            if args.reexport or not weights_npz.exists():
                model.export_early14_f16(weights_npz)
                print(f"Exported {weights_npz} in {time.perf_counter() - t0:.2f}s")
            if args.awq_mse and args.variant in ("both", "w8"):
                tc0 = time.perf_counter()
                res_awq = collect_awq_calibration_col_sq(
                    model,
                    chunk_s=ANE_CHUNK_S,
                    calib_file=args.calib_file,
                    system_prompt=sys_prompt_calib,
                    max_docs=args.calib_max_docs,
                    return_meta=True,
                )
                assert isinstance(res_awq, tuple)
                act_col_sq, c_info = res_awq
                calib_meta.update(c_info)
                print(
                    f"Collected AWQ-MSE calibration E[x^2] ({c_info['calibration_source']}, "
                    f"{c_info['calibration_docs']} docs) in {time.perf_counter() - tc0:.2f}s"
                )
            del model
            mx.clear_cache()

    w8_npz_path = weights_npz
    tmp_awq_npz: Path | None = None
    if act_col_sq is not None:
        npz_raw = dict(np.load(weights_npz))
        clipped_npz, mean_alpha = apply_awq_mse_clipping(npz_raw, act_col_sq)
        calib_meta["mean_alpha"] = mean_alpha
        print(f"Applied AWQ-MSE symmetric channel clipping (mean_alpha={mean_alpha:.4f})")
        tmp_awq_npz = weights_npz.with_name("gemma4_early_awq_tmp.npz")
        np.savez(tmp_awq_npz, **clipped_npz)
        w8_npz_path = tmp_awq_npz
        del npz_raw, clipped_npz

    out_ane_dir = (
        paths.ane_models_dir
        if paths._is_default_model_dir(args.model_dir)
        else paths.ane_models_dir / paths.checkpoint_ane_key(Path(args.model_dir))
    )

    try:
        try:
            import coremltools as _ct  # noqa: F401

            has_ct = True
        except ModuleNotFoundError:
            has_ct = False

        if has_ct:
            compile_ane_bundles_from_npz(
                fp16_npz_path=weights_npz,
                w8_npz_path=w8_npz_path,
                out_ane_dir=out_ane_dir,
                variant=args.variant,
                keep_mlpackage=args.keep_mlpackage,
                lib_ane_prefix=paths.lib_ane_prefix,
                S=ANE_CHUNK_S,
                P_SWA=ANE_P_SWA,
                P_FULL=ANE_P_FULL,
                compute_precision_mode=args.compute_precision,
                manifest_meta=calib_meta,
            )
        else:
            uv_bin = shutil.which("uv")
            if not uv_bin:
                raise ModuleNotFoundError(
                    "coremltools is required to compile ANE .mlmodelc bundles (install `fast-gemma4[ane-build]` or `uv`)."
                )
            cmd = [
                uv_bin,
                "run",
                "--no-project",
                "--with",
                "coremltools",
                "python3",
                str(Path(__file__).resolve()),
                "--variant",
                args.variant,
                "--compute-precision",
                args.compute_precision,
                "--compile-only-fp16-npz",
                str(weights_npz),
                "--compile-only-w8-npz",
                str(w8_npz_path),
                "--compile-only-out-dir",
                str(out_ane_dir),
                "--compile-only-lib",
                str(paths.lib_ane_prefix),
                "--compile-only-meta-json",
                json.dumps(calib_meta),
            ]
            if args.keep_mlpackage:
                cmd.append("--keep-mlpackage")
            subprocess.check_call(cmd)
    finally:
        if tmp_awq_npz is not None and tmp_awq_npz.exists():
            tmp_awq_npz.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
