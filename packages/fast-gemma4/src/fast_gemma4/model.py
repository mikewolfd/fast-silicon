"""Standalone Gemma-4 E2B MLX + Metal 4 MPP model loader and ragged prefix-shared runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from .kernels import (
    fused_geglu_split,
    fused_gemma4_res_rms_bf16,
    fused_kv_only_norm_rope_ragged,
    fused_norm_add_norm,
    fused_ple_geglu_indexed,
    fused_q_norm_rope_ragged,
    fused_qkv_packed_norm_rope,
    fused_qkv_packed_norm_rope_ragged,
    fused_res_rms_norm,
    fused_softcap_argmax_bf16,
    gather_dequant_kv_pair_i8_bf16,
    mpp_fused_geglu_bf16,
    mpp_linear_bf16,
    mqa8_sdpa_prefix_ragged,
    mqa8_sdpa_prefix_ragged_slot,
    quantize_kv_pair_i8,
    scatter_kv_i8_slot_precomputed,
    scatter_kv_ragged,
    scatter_kv_slot_precomputed,
)
from .paths import Gemma4Paths, resolve_paths
from .tokenizer import DEFAULT_KEPT_VOCAB_SIZE, FULL_VOCAB_SIZE, VocabRemapper


def create_causal_mask(N: int, offset: int = 0, window_size: int | None = None) -> mx.array:
    """Boolean lower-triangular causal (+ optional sliding window) attention mask."""
    rinds = mx.arange(offset + N, dtype=mx.int32)[None, :]
    linds = mx.arange(offset, offset + N, dtype=mx.int32)[:, None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < (rinds + int(window_size)))
    return mask


@mx.compile
def logit_softcap(softcap: float, x: mx.array) -> mx.array:
    return mx.tanh(x / softcap) * softcap


def _qmeta_for(qcfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    meta = qcfg.get(prefix)
    return meta if isinstance(meta, dict) else {}


def _dequantize_linear(raw: dict[str, mx.array], prefix: str, qcfg: dict[str, Any], dtype: mx.Dtype) -> mx.array:
    w_key = f"{prefix}.weight"
    if f"{prefix}.scales" not in raw:
        return raw[w_key].astype(dtype)
    meta = _qmeta_for(qcfg, prefix)
    bits = int(meta.get("bits", qcfg.get("bits", 4)))
    gs = int(meta.get("group_size", qcfg.get("group_size", 64)))
    mode = str(meta.get("mode", qcfg.get("mode", "affine")))
    return mx.dequantize(
        raw[w_key],
        raw[f"{prefix}.scales"],
        raw.get(f"{prefix}.biases"),
        group_size=gs,
        bits=bits,
        mode=mode,
    ).astype(dtype)


def _load_linear(
    raw: dict[str, mx.array],
    prefix: str,
    qcfg: dict[str, Any],
    dtype: mx.Dtype,
    *,
    promote_bf16: bool = False,
) -> nn.Module:
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.scales"
    if s_key not in raw or promote_bf16:
        w = _dequantize_linear(raw, prefix, qcfg, dtype)
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight = w
        return lin

    meta = _qmeta_for(qcfg, prefix)
    bits = int(meta.get("bits", qcfg.get("bits", 4)))
    gs = int(meta.get("group_size", qcfg.get("group_size", 64)))
    mode = str(meta.get("mode", qcfg.get("mode", "affine")))
    scales = raw[s_key].astype(dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    out_f = int(scales.shape[0])
    in_f = int(scales.shape[1]) * gs
    qlin = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=gs, bits=bits, mode=mode)
    qlin.weight = raw[w_key]
    qlin.scales = scales
    qlin.biases = biases
    return qlin


def _load_qembedding(raw: dict[str, mx.array], prefix: str, qcfg: dict[str, Any], dtype: mx.Dtype) -> nn.Module:
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.scales"
    if s_key not in raw:
        w = raw[w_key].astype(dtype)
        emb = nn.Embedding(int(w.shape[0]), int(w.shape[1]))
        emb.weight = w
        return emb
    meta = _qmeta_for(qcfg, prefix)
    bits = int(meta.get("bits", qcfg.get("bits", 4)))
    gs = int(meta.get("group_size", qcfg.get("group_size", 64)))
    mode = str(meta.get("mode", qcfg.get("mode", "affine")))
    scales = raw[s_key].astype(dtype)
    biases = raw[f"{prefix}.biases"].astype(dtype) if f"{prefix}.biases" in raw else None
    num_emb = int(scales.shape[0])
    dims = int(scales.shape[1]) * gs
    qemb = nn.QuantizedEmbedding(num_emb, dims, group_size=gs, bits=bits, mode=mode)
    qemb.weight = raw[w_key]
    qemb.scales = scales
    qemb.biases = biases
    return qemb


def _fuse_linears(mods: Sequence[nn.Module], dtype: mx.Dtype) -> nn.Module:
    first = mods[0]
    if all(
        isinstance(m, nn.QuantizedLinear)
        and m.bits == first.bits
        and m.group_size == first.group_size
        and m.mode == first.mode
        and (m.biases is None) == (first.biases is None)
        for m in mods
    ):
        in_f = int(first.scales.shape[1]) * int(first.group_size)
        out_f = sum(int(m.weight.shape[0]) for m in mods)
        fused = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=first.group_size, bits=first.bits, mode=first.mode)
        fused.weight = mx.concatenate([m.weight for m in mods], axis=0)
        fused.scales = mx.concatenate([m.scales for m in mods], axis=0)
        fused.biases = mx.concatenate([m.biases for m in mods], axis=0) if first.biases is not None else None
        return fused
    weights = []
    for m in mods:
        if isinstance(m, nn.QuantizedLinear):
            weights.append(
                mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits, mode=m.mode).astype(dtype)
            )
        else:
            weights.append(m.weight.astype(dtype))
    fused = nn.Linear(int(weights[0].shape[1]), sum(int(w.shape[0]) for w in weights), bias=False)
    fused.weight = mx.concatenate(weights, axis=0)
    return fused


def _slice_q_from_qkv(qkv_proj: nn.Module, q_dim: int) -> nn.Module:
    """Extract [q_proj] rows [:q_dim] from fused qkv_proj without losing quantization."""
    if isinstance(qkv_proj, nn.QuantizedLinear):
        in_f = int(qkv_proj.scales.shape[1]) * int(qkv_proj.group_size)
        q_mod = nn.QuantizedLinear(
            in_f, q_dim, bias=False, group_size=qkv_proj.group_size, bits=qkv_proj.bits, mode=qkv_proj.mode
        )
        q_mod.weight = qkv_proj.weight[:q_dim]
        q_mod.scales = qkv_proj.scales[:q_dim]
        q_mod.biases = qkv_proj.biases[:q_dim] if qkv_proj.biases is not None else None
        return q_mod
    w = qkv_proj.weight[:q_dim]
    q_mod = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    q_mod.weight = w
    return q_mod


def _slice_kv_from_qkv(qkv_proj: nn.Module, q_dim: int) -> nn.Module:
    """Extract [k_proj; v_proj] rows [q_dim:] from fused qkv_proj without losing quantization."""
    if isinstance(qkv_proj, nn.QuantizedLinear):
        in_f = int(qkv_proj.scales.shape[1]) * int(qkv_proj.group_size)
        out_f = int(qkv_proj.weight.shape[0]) - q_dim
        kv_q = nn.QuantizedLinear(
            in_f, out_f, bias=False, group_size=qkv_proj.group_size, bits=qkv_proj.bits, mode=qkv_proj.mode
        )
        kv_q.weight = qkv_proj.weight[q_dim:]
        kv_q.scales = qkv_proj.scales[q_dim:]
        kv_q.biases = qkv_proj.biases[q_dim:] if qkv_proj.biases is not None else None
        return kv_q
    w = qkv_proj.weight[q_dim:]
    kv = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    kv.weight = w
    return kv


def _slice_to_bf16_padded(mod: nn.Module, pad_mult: int = 64) -> mx.array:
    if isinstance(mod, nn.QuantizedLinear):
        w = mx.dequantize(
            mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode
        ).astype(mx.bfloat16)
    else:
        w = mod.weight.astype(mx.bfloat16)
    N = int(w.shape[0])
    pad_n = (pad_mult - (N % pad_mult)) % pad_mult
    if pad_n:
        w = mx.pad(w, [(0, pad_n), (0, 0)])
    return w


def _build_rope_inv_freq(tcfg: dict[str, Any], is_full: bool) -> mx.array:
    rope_map = tcfg.get("rope_parameters", {})
    if is_full:
        D = int(tcfg["global_head_dim"])
        p = rope_map.get("full_attention", {})
        theta = float(p.get("rope_theta", 1_000_000.0))
        factor = float(p.get("partial_rotary_factor", 0.25))
        rotary_dims = int(D * factor)
        inv_rot = 1.0 / (theta ** (mx.arange(0, rotary_dims, 2, dtype=mx.float32) / float(D)))
        inv_zero = mx.zeros(((D - rotary_dims) // 2,), dtype=mx.float32)
        inv = mx.concatenate([inv_rot, inv_zero], axis=0).astype(mx.float32)
    else:
        D = int(tcfg["head_dim"])
        p = rope_map.get("sliding_attention", {})
        theta = float(p.get("rope_theta", 10_000.0))
        inv = (1.0 / (theta ** (mx.arange(0, D, 2, dtype=mx.float32) / float(D)))).astype(mx.float32)
    mx.eval(inv)
    return inv


def _compute_previous_kvs(layer_types: Sequence[str], num_kv_shared: int) -> list[int]:
    n = len(layer_types)
    first_shared = n - num_kv_shared
    prev = list(range(n))
    if num_kv_shared > 0 and first_shared > 0:
        by_type: dict[str, int] = {}
        for i in range(first_shared):
            by_type[layer_types[i]] = i
        for j in range(first_shared, n):
            prev[j] = by_type[layer_types[j]]
    return prev


class FastGemma4Model:
    """
    Standalone hardware-limit Gemma-4 E2B runner for Apple Silicon (M4 Pro):
      - Zero `transformers`/`torch`/`mlx_vlm` runtime dependencies
      - Shared `KV_sys` prefix broadcast across layers `0..14` + ragged per-doc `KV_doc`
      - Exact power-of-2 bitmask SWA ring buffer (`512`) + Full KV linear buffer
      - `EarlyExit14` prefill with L14 `kv_proj_only` (skips Q, SDPA, O, MLP, PLE, L15..34)
      - Apple Metal 4 `MPP::matmul2d` BF16 GeGLU (`32x32` / `64x32`) & `down_proj` (`32x32` / `64x64`)
      - Lossless English+JSON `150,786` LM head pruning (with full `262,144` input embeddings + PLE)
    """

    use_vec4_norm: bool = False

    def __init__(
        self,
        *,
        paths: Gemma4Paths | None = None,
        model_dir: str | Path | None = None,
        mlp_dtype: str = "bfloat16",
        vocab_out: int | None = DEFAULT_KEPT_VOCAB_SIZE,
        use_mpp: bool = True,
        lm_head_mode: str = "q4",
        local_files_only: bool = True,
        kv_i8: bool = True,
        vec4_reduce: bool = True,
        mpp_m32_cutoff: int = 96,
        use_fused_softcap_argmax: bool = True,
        use_vec4_norm: bool = False,
    ) -> None:
        if mlp_dtype not in ("bfloat16", "quantized"):
            raise ValueError(f"Unsupported mlp_dtype={mlp_dtype!r}")
        if lm_head_mode not in ("q4", "bf16_mpp"):
            raise ValueError(f"Unsupported lm_head_mode={lm_head_mode!r}")

        self.paths = paths or resolve_paths()
        self.model_dir = self.paths.resolve_model_dir(model_dir, local_files_only=local_files_only)
        self.tokenizer_json = self.paths.resolve_tokenizer_json(self.model_dir, local_files_only=local_files_only)
        self.mlp_dtype_name = mlp_dtype
        self.act_dtype = mx.bfloat16
        self.use_mpp = use_mpp and (mlp_dtype == "bfloat16")
        self.lm_head_mode = lm_head_mode
        self.kv_i8 = bool(kv_i8)
        self.vec4_reduce = bool(vec4_reduce)
        self.mpp_m32_cutoff = int(mpp_m32_cutoff)
        self.use_fused_softcap_argmax = bool(use_fused_softcap_argmax)
        self.use_vec4_norm = bool(use_vec4_norm)

        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        tcfg = cfg.get("text_config", cfg)
        qcfg = cfg.get("quantization") or cfg.get("quantization_config") or {}
        if tcfg.get("attention_k_eq_v", False):
            raise NotImplementedError("FastGemma4Model expects attention_k_eq_v=false (Gemma-4 E2B).")

        self.num_layers = int(tcfg["num_hidden_layers"])
        self.hidden_size = int(tcfg["hidden_size"])
        self.num_kv_shared = int(tcfg.get("num_kv_shared_layers", 20))
        self.first_kv_shared = self.num_layers - self.num_kv_shared
        self.layer_types = list(tcfg["layer_types"])
        self.previous_kvs = _compute_previous_kvs(self.layer_types, self.num_kv_shared)
        self.window_size = int(tcfg.get("sliding_window", 512))
        self.softcap = float(tcfg.get("final_logit_softcapping", 30.0))
        self.eps = float(tcfg.get("rms_norm_eps", 1e-6))
        self.ple_dim = int(tcfg.get("hidden_size_per_layer_input", 256))

        sf_files = sorted(self.model_dir.glob("model*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"Missing model*.safetensors in {self.model_dir}")
        raw: dict[str, mx.array] = {}
        for sf_path in sf_files:
            raw_all: dict[str, mx.array] = mx.load(str(sf_path))
            for k, v in raw_all.items():
                if k.startswith("language_model.model.") or k.startswith("model."):
                    raw[k] = v
            del raw_all

        base_p = "language_model.model" if any(k.startswith("language_model.model.") for k in raw) else "model"
        self.embed_tokens = _load_qembedding(raw, f"{base_p}.embed_tokens", qcfg, self.act_dtype)
        self.embed_scale = mx.array(self.hidden_size**0.5, dtype=self.act_dtype)
        self.embed_tokens_per_layer = _load_qembedding(raw, f"{base_p}.embed_tokens_per_layer", qcfg, self.act_dtype)
        self.embed_tokens_per_layer_scale = mx.array(self.ple_dim**0.5, dtype=self.act_dtype)
        self.per_layer_model_projection = _load_linear(raw, f"{base_p}.per_layer_model_projection", qcfg, self.act_dtype)
        self.per_layer_projection_scale = mx.array(self.hidden_size**-0.5, dtype=self.act_dtype)
        self.per_layer_projection_norm_w = raw[f"{base_p}.per_layer_projection_norm.weight"].astype(self.act_dtype)
        self.per_layer_input_scale = mx.array(2.0**-0.5, dtype=self.act_dtype)
        self.norm_w = raw[f"{base_p}.norm.weight"].astype(self.act_dtype)

        # Sliced EarlyExit14 PLI projection (cuts L0..13 prefill & FF-drain PLI projection by 60%)
        self._n_early_pli_layers = self.first_kv_shared - 1
        n_early_pli_dim = self._n_early_pli_layers * self.ple_dim
        pmp = self.per_layer_model_projection
        if isinstance(pmp, nn.QuantizedLinear):
            pli_proj_14 = nn.QuantizedLinear(
                int(pmp.scales.shape[1]) * int(pmp.group_size),
                n_early_pli_dim,
                bias=False,
                group_size=pmp.group_size,
                bits=pmp.bits,
                mode=pmp.mode,
            )
            pli_proj_14.weight = pmp.weight[:n_early_pli_dim]
            pli_proj_14.scales = pmp.scales[:n_early_pli_dim]
            pli_proj_14.biases = pmp.biases[:n_early_pli_dim] if pmp.biases is not None else None
        else:
            pli_proj_14 = nn.Linear(int(pmp.weight.shape[1]), n_early_pli_dim, bias=False)
            pli_proj_14.weight = pmp.weight[:n_early_pli_dim]
        self.pli_proj_14 = pli_proj_14

        etpl = self.embed_tokens_per_layer
        if isinstance(etpl, nn.QuantizedEmbedding) and n_early_pli_dim % int(etpl.group_size) == 0:
            self._etpl_14_groups = n_early_pli_dim // int(etpl.group_size)
            self._etpl_14_wcols = (int(etpl.weight.shape[1]) * self._etpl_14_groups) // int(etpl.scales.shape[1])
        else:
            self._etpl_14_groups = 0
            self._etpl_14_wcols = 0

        self._layer_idx_mx = [mx.array([i], dtype=mx.int32) for i in range(self.num_layers)]

        # Build full or lossless-pruned LM head sharing rows from tied embed_tokens
        if vocab_out is not None and int(vocab_out) < FULL_VOCAB_SIZE:
            ckpt_remap = self.model_dir / "vocab_remap.npz"
            if ckpt_remap.is_file() and int(vocab_out) == DEFAULT_KEPT_VOCAB_SIZE:
                self.remapper: VocabRemapper | None = VocabRemapper.load_or_build(
                    self.tokenizer_json,
                    ckpt_remap,
                    target_size=None,
                )
            else:
                self.remapper = VocabRemapper.load_or_build(
                    self.tokenizer_json,
                    self.paths.vocab_remap_npz,
                    target_size=int(vocab_out),
                )
            kept = mx.array(self.remapper.kept_old_ids, dtype=mx.int32)
            emb = self.embed_tokens
            if isinstance(emb, nn.QuantizedEmbedding):
                in_f = int(emb.scales.shape[1]) * int(emb.group_size)
                head = nn.QuantizedLinear(
                    in_f, len(self.remapper.kept_old_ids), bias=False, group_size=emb.group_size, bits=emb.bits, mode=emb.mode
                )
                head.weight = emb.weight[kept]
                head.scales = emb.scales[kept]
                head.biases = emb.biases[kept] if emb.biases is not None else None
            else:
                head = nn.Linear(self.hidden_size, len(self.remapper.kept_old_ids), bias=False)
                head.weight = emb.weight[kept]
            self.lm_head = head
            self.new_to_old = self.remapper.kept_old_ids
            self.old_to_new = self.remapper.old_to_new
        else:
            self.remapper = None
            emb = self.embed_tokens
            if isinstance(emb, nn.QuantizedEmbedding):
                in_f = int(emb.scales.shape[1]) * int(emb.group_size)
                head = nn.QuantizedLinear(
                    in_f, int(emb.weight.shape[0]), bias=False, group_size=emb.group_size, bits=emb.bits, mode=emb.mode
                )
                head.weight = emb.weight
                head.scales = emb.scales
                head.biases = emb.biases
            else:
                head = nn.Linear(self.hidden_size, int(emb.weight.shape[0]), bias=False)
                head.weight = emb.weight
            self.lm_head = head
            self.new_to_old = np.arange(int(head.weight.shape[0]), dtype=np.int32)
            self.old_to_new = self.new_to_old

        self.vocab_size = int(len(self.new_to_old))
        if self.lm_head_mode == "bf16_mpp":
            self.w_lm_bf16_pad: mx.array | None = _slice_to_bf16_padded(self.lm_head, pad_mult=64)
        else:
            self.w_lm_bf16_pad = None

        promote_mlp = self.mlp_dtype_name == "bfloat16"
        inv_swa = _build_rope_inv_freq(tcfg, is_full=False)
        inv_full = _build_rope_inv_freq(tcfg, is_full=True)

        self.layers: list[dict[str, Any]] = []
        for i in range(self.num_layers):
            lp = f"{base_p}.layers.{i}"
            is_full = self.layer_types[i] == "full_attention"
            head_dim = int(tcfg["global_head_dim"] if is_full else tcfg["head_dim"])
            is_shared = i >= self.first_kv_shared

            q_proj_raw = _load_linear(raw, f"{lp}.self_attn.q_proj", qcfg, self.act_dtype)
            if not is_shared:
                k_proj_raw = _load_linear(raw, f"{lp}.self_attn.k_proj", qcfg, self.act_dtype)
                v_proj_raw = _load_linear(raw, f"{lp}.self_attn.v_proj", qcfg, self.act_dtype)
                qkv_proj = _fuse_linears([q_proj_raw, k_proj_raw, v_proj_raw], self.act_dtype)
                q_proj = None
                w_kn = raw[f"{lp}.self_attn.k_norm.weight"].astype(self.act_dtype)
            else:
                qkv_proj = None
                q_proj = q_proj_raw
                w_kn = None

            o_proj = _load_linear(raw, f"{lp}.self_attn.o_proj", qcfg, self.act_dtype)
            gate_p = _load_linear(raw, f"{lp}.mlp.gate_proj", qcfg, self.act_dtype, promote_bf16=promote_mlp)
            up_p = _load_linear(raw, f"{lp}.mlp.up_proj", qcfg, self.act_dtype, promote_bf16=promote_mlp)
            down_p = _load_linear(raw, f"{lp}.mlp.down_proj", qcfg, self.act_dtype, promote_bf16=promote_mlp)
            gate_up_proj = _fuse_linears([gate_p, up_p], self.act_dtype)

            ple_gate = _load_linear(raw, f"{lp}.per_layer_input_gate", qcfg, self.act_dtype)
            ple_proj = _load_linear(raw, f"{lp}.per_layer_projection", qcfg, self.act_dtype)

            if i + 1 < self.num_layers:
                next_norm_w = raw[f"{base_p}.layers.{i + 1}.input_layernorm.weight"].astype(self.act_dtype)
            else:
                next_norm_w = self.norm_w

            ld: dict[str, Any] = {
                "idx": i,
                "is_shared": is_shared,
                "prev_kv": self.previous_kvs[i],
                "is_full": is_full,
                "head_dim": head_dim,
                "inv_freq": inv_full if is_full else inv_swa,
                "w_in": raw[f"{lp}.input_layernorm.weight"].astype(self.act_dtype),
                "w_post_attn": raw[f"{lp}.post_attention_layernorm.weight"].astype(self.act_dtype),
                "w_pre_ff": raw[f"{lp}.pre_feedforward_layernorm.weight"].astype(self.act_dtype),
                "w_post_ff": raw[f"{lp}.post_feedforward_layernorm.weight"].astype(self.act_dtype),
                "w_post_ple": raw[f"{lp}.post_per_layer_input_norm.weight"].astype(self.act_dtype),
                "w_next_norm": next_norm_w,
                "layer_scalar": raw[f"{lp}.layer_scalar"].astype(self.act_dtype).reshape((1,)),
                "w_qn": raw[f"{lp}.self_attn.q_norm.weight"].astype(self.act_dtype),
                "w_kn": w_kn,
                "qkv_proj": qkv_proj,
                "q_proj": q_proj,
                "o_proj": o_proj,
                "gate_up_proj": gate_up_proj,
                "down_proj": down_p,
                "ple_gate": ple_gate,
                "ple_proj": ple_proj,
            }
            if self.use_mpp and hasattr(gate_up_proj, "weight"):
                w_gu = gate_up_proj.weight
                h_mid = int(w_gu.shape[0]) // 2
                ld["w_gu"] = w_gu
                ld["w_g"] = w_gu[:h_mid]
                ld["w_u"] = w_gu[h_mid:]
                ld["w_dn"] = down_p.weight
            if i == self.first_kv_shared - 1:
                ld["kv_proj_only"] = _slice_kv_from_qkv(qkv_proj, 8 * head_dim)
                ld["q_proj_split"] = _slice_q_from_qkv(qkv_proj, 8 * head_dim)
            self.layers.append(ld)

        del raw
        self.swa_layers = [i for i in range(self.first_kv_shared) if not self.layers[i]["is_full"]]
        self.full_layers = [i for i in range(self.first_kv_shared) if self.layers[i]["is_full"]]
        self.swa_slot = {l: k for k, l in enumerate(self.swa_layers)}
        self.full_slot = {l: k for k, l in enumerate(self.full_layers)}

        mx.eval(self._eval_targets())
        self.param_gb = sum(int(v.nbytes) for _, v in tree_flatten(self._eval_targets())) / (1024**3)

    def _eval_targets(self) -> list[Any]:
        arrs: list[Any] = [
            self.embed_tokens.parameters(),
            self.embed_scale,
            self.embed_tokens_per_layer.parameters(),
            self.embed_tokens_per_layer_scale,
            self.per_layer_model_projection.parameters(),
            self.pli_proj_14.parameters(),
            self.per_layer_projection_scale,
            self.per_layer_projection_norm_w,
            self.per_layer_input_scale,
            self.norm_w,
            self.lm_head.parameters(),
            self._layer_idx_mx,
        ]
        if self.w_lm_bf16_pad is not None:
            arrs.append(self.w_lm_bf16_pad)
        for ld in self.layers:
            for k in (
                "inv_freq",
                "w_in",
                "w_post_attn",
                "w_pre_ff",
                "w_post_ff",
                "w_post_ple",
                "w_next_norm",
                "layer_scalar",
                "w_qn",
            ):
                arrs.append(ld[k])
            if ld["w_kn"] is not None:
                arrs.append(ld["w_kn"])
            for k in (
                "qkv_proj",
                "q_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
                "ple_gate",
                "ple_proj",
                "kv_proj_only",
                "q_proj_split",
            ):
                m = ld.get(k)
                if m is not None:
                    arrs.append(m.parameters())
            for k in ("w_g", "w_u"):
                if k in ld:
                    arrs.append(ld[k])
        return arrs

    def _compute_embed_and_pli(self, toks: mx.array) -> tuple[mx.array, mx.array]:
        h = self.embed_tokens(toks) * self.embed_scale
        pli_raw = (self.embed_tokens_per_layer(toks) * self.embed_tokens_per_layer_scale).reshape(
            *toks.shape, self.num_layers, self.ple_dim
        )
        proj = (self.per_layer_model_projection(h) * self.per_layer_projection_scale).reshape(
            *toks.shape, self.num_layers, self.ple_dim
        )
        proj = mx.fast.rms_norm(proj, self.per_layer_projection_norm_w, self.eps)
        pli = (proj + pli_raw) * self.per_layer_input_scale
        return h, pli

    def _compute_embed_and_pli_early14(self, toks: mx.array) -> tuple[mx.array, mx.array]:
        """Compute token embeddings + PLI strictly for Layers `0..first_kv_shared-2` (`EarlyExit14`)."""
        B, L = toks.shape
        n_l = self._n_early_pli_layers
        h = self.embed_tokens(toks) * self.embed_scale
        etpl = self.embed_tokens_per_layer
        if self._etpl_14_groups > 0:
            w_sub = etpl.weight[toks][..., : self._etpl_14_wcols]
            s_sub = etpl.scales[toks][..., : self._etpl_14_groups]
            b_sub = etpl.biases[toks][..., : self._etpl_14_groups] if etpl.biases is not None else None
            pli_tok = mx.dequantize(
                w_sub,
                scales=s_sub,
                biases=b_sub,
                group_size=etpl.group_size,
                bits=etpl.bits,
                mode=etpl.mode,
            ) * self.embed_tokens_per_layer_scale
        else:
            pli_tok = (
                etpl.weight[toks][..., : n_l * self.ple_dim].astype(self.act_dtype)
                * self.embed_tokens_per_layer_scale
            )
        pli_tok = pli_tok.reshape(B, L, n_l, self.ple_dim)
        pli_mod = (self.pli_proj_14(h) * self.per_layer_projection_scale).reshape(B, L, n_l, self.ple_dim)
        pli_mod = mx.fast.rms_norm(pli_mod, self.per_layer_projection_norm_w, self.eps)
        pli = (pli_mod + pli_tok) * self.per_layer_input_scale
        return h, pli

    def _apply_lm_head(self, x_in: mx.array, apply_softcap: bool = True) -> mx.array:
        if self.w_lm_bf16_pad is not None:
            lg = mpp_linear_bf16(x_in, self.w_lm_bf16_pad, tm=64, tn=64)[:, :, : self.vocab_size]
        else:
            lg = self.lm_head(x_in)
        if apply_softcap:
            lg = logit_softcap(self.softcap, lg)
        return lg

    def compute_full_vocab_logits_1d(self, x_row: mx.array, *, apply_softcap: bool = True) -> np.ndarray:
        """Compute lazy full `262,144`-vocab softcapped `float32 [V_full]` logits from a single hidden row `[D]`."""
        x_2d = mx.contiguous(x_row).reshape(1, self.hidden_size)
        emb = self.embed_tokens
        if isinstance(emb, nn.QuantizedEmbedding):
            lg = mx.quantized_matmul(
                x_2d,
                emb.weight,
                scales=emb.scales,
                biases=emb.biases,
                transpose=True,
                group_size=emb.group_size,
                bits=emb.bits,
                mode=emb.mode,
            )
        else:
            lg = x_2d @ emb.weight.T
        if apply_softcap:
            lg = logit_softcap(self.softcap, lg)
        lg_f32 = lg.reshape(-1).astype(mx.float32)
        mx.eval(lg_f32)
        return np.asarray(lg_f32)

    def _apply_lm_head_argmax(self, x_in: mx.array) -> mx.array:
        """Compute greedy argmax over logits without allocating `[B, 1, V]` float32 softcapped buffer."""
        if self.use_fused_softcap_argmax and x_in.dtype == mx.bfloat16:
            if self.w_lm_bf16_pad is not None:
                raw_lg = mpp_linear_bf16(x_in, self.w_lm_bf16_pad, tm=64, tn=64)
                return fused_softcap_argmax_bf16(raw_lg, self.softcap, vocab_limit=self.vocab_size)
            raw_lg = self.lm_head(x_in)
            if raw_lg.dtype == mx.bfloat16:
                return fused_softcap_argmax_bf16(raw_lg, self.softcap)
        lg = self._apply_lm_head(x_in, apply_softcap=True)
        return mx.argmax(lg[:, 0, :], axis=-1)

    def attach_dual_mlp_q4(self, *, q4_max_m: int = 32, skip_layers: frozenset[int] = frozenset()) -> int:
        """Lazily quantize missing MLPs to Q4 (~1.07 GB max) for auto B>=48 M<=q4_max_m decode tails."""
        self.dual_q4_max_m = int(q4_max_m)
        attached = 0
        for i, l in enumerate(self.layers):
            if i in skip_layers:
                continue
            gu, dn = l["gate_up_proj"], l["down_proj"]
            if isinstance(gu, nn.QuantizedLinear) or "q4_gate_up" in l:
                continue
            q_gu = nn.QuantizedLinear.from_linear(gu, group_size=64, bits=4, mode="affine")
            q_dn = nn.QuantizedLinear.from_linear(dn, group_size=64, bits=4, mode="affine")
            mx.eval(q_gu.parameters(), q_dn.parameters())
            l["q4_gate_up"] = q_gu
            l["q4_down"] = q_dn
            attached += 1
        return attached

    def _run_mlp(self, mlp_in: mx.array, ld: dict[str, Any], allow_mpp: bool = True) -> mx.array:
        M = mlp_in.size // mlp_in.shape[-1] if mlp_in.size > 0 else 0
        if getattr(self, "_dual_q4_active", False) and "q4_gate_up" in ld:
            q4_cap = getattr(self, "dual_q4_max_m", 32)
            if M <= q4_cap:
                mlp_act = fused_geglu_split(ld["q4_gate_up"](mlp_in))
                return ld["q4_down"](mlp_act)
        mpp_ok = allow_mpp and (M < 512 or M >= 2048)
        if self.use_mpp and mpp_ok and M >= 16 and "w_g" in ld:
            mlp_act = mpp_fused_geglu_bf16(
                mlp_in, ld["w_g"], ld["w_u"], ld["w_gu"], m32_cutoff=self.mpp_m32_cutoff
            )
            if M <= 64:
                return ld["down_proj"](mlp_act)
            if M >= 512:
                return mpp_linear_bf16(mlp_act, ld["w_dn"], 64, 64)
            return mpp_linear_bf16(mlp_act, ld["w_dn"], 32, 32)
        mlp_act = fused_geglu_split(ld["gate_up_proj"](mlp_in))
        return ld["down_proj"](mlp_act)

    # ------------------------------------------------------------------
    # Shared Prefix KV_sys Prefill (runs ONCE per system prompt)
    # ------------------------------------------------------------------
    def prefill_prefix(self, sys_ids: Sequence[int], chunk_sz: int = 256) -> dict[str, Any]:
        """Compute shared `kv_sys [15][L_sys, D]` for system prompt using EarlyExit14."""
        if not sys_ids:
            # Allocate non-empty device buffers so Metal kernels bind `k_sys`/`v_sys` as `device` (`l_sys=0` keeps them unread).
            empty_k = [mx.zeros((16, self.layers[i]["head_dim"]), dtype=self.act_dtype) for i in range(self.first_kv_shared)]
            empty_v = [mx.zeros((16, self.layers[i]["head_dim"]), dtype=self.act_dtype) for i in range(self.first_kv_shared)]
            mx.eval(*empty_k, *empty_v)
            return {"l_sys": 0, "k_sys": empty_k, "v_sys": empty_v}

        ids = mx.array(sys_ids, dtype=mx.int32)[None, :]
        l_sys = int(ids.shape[1])
        k_sys_list = [mx.zeros((l_sys, self.layers[i]["head_dim"]), dtype=self.act_dtype) for i in range(self.first_kv_shared)]
        v_sys_list = [mx.zeros((l_sys, self.layers[i]["head_dim"]), dtype=self.act_dtype) for i in range(self.first_kv_shared)]
        mx.eval(*k_sys_list, *v_sys_list)

        for s0 in range(0, l_sys, chunk_sz):
            s1 = min(l_sys, s0 + chunk_sz)
            lc = s1 - s0
            tok_c = ids[:, s0:s1]
            off_meta = mx.array([s0, lc], dtype=mx.int32)
            off_1 = mx.array([s0], dtype=mx.int32)
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                if i == self.first_kv_shared - 1:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_1, self.eps
                    )
                    k_sys_list[i][s0:s1, :] = k_rot[0]
                    v_sys_list[i][s0:s1, :] = v_norm[0]
                    break

                qkv = l["qkv_proj"](x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps
                )
                k_sys_list[i][s0:s1, :] = k_rot[0]
                v_sys_list[i][s0:s1, :] = v_norm[0]

                if l["is_full"]:
                    kf = k_sys_list[i][:s1][None, None, :, :]
                    vf = v_sys_list[i][:s1][None, None, :, :]
                    mk = ("full", s0, lc)
                    if mk not in masks:
                        masks[mk] = "causal" if (s0 == 0 and lc > 1) else create_causal_mask(lc, offset=s0, window_size=None)
                else:
                    st = max(0, s0 - self.window_size + 1)
                    kf = k_sys_list[i][st:s1][None, None, :, :]
                    vf = v_sys_list[i][st:s1][None, None, :, :]
                    mk = ("swa", s0 - st, lc)
                    if mk not in masks:
                        masks[mk] = create_causal_mask(lc, offset=(s0 - st), window_size=self.window_size)

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(1, lc, 8 * D)
                )
                attn_out = l["o_proj"](att)
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"])
            mx.eval(*k_sys_list, *v_sys_list)

        return {"l_sys": l_sys, "k_sys": k_sys_list, "v_sys": v_sys_list}

    # ------------------------------------------------------------------
    # Ragged Per-Doc Cache Allocation & Prefix-Shared Decode/Verify Step
    # ------------------------------------------------------------------
    def make_batch_cache(
        self,
        prefix_kv: dict[str, Any],
        batch_size: int,
        doc_cap_full: int = 1152,
        swa_ring_cap: int = 512,
    ) -> dict[str, Any]:
        if swa_ring_cap < self.window_size or (swa_ring_cap & (swa_ring_cap - 1)) != 0:
            raise ValueError(f"swa_ring_cap={swa_ring_cap} must be a power of 2 >= {self.window_size}")
        layers = []
        for i in range(self.first_kv_shared):
            l = self.layers[i]
            D = l["head_dim"]
            ring = not l["is_full"]
            cap = swa_ring_cap if ring else doc_cap_full
            layers.append(
                {
                    "k": mx.zeros((batch_size, cap, D), dtype=self.act_dtype),
                    "v": mx.zeros((batch_size, cap, D), dtype=self.act_dtype),
                    "cap": cap,
                    "ring": ring,
                }
            )
        mx.eval(*[(c["k"], c["v"]) for c in layers])
        l_sys = int(prefix_kv["l_sys"])
        off_b = mx.full((batch_size,), l_sys, dtype=mx.int32)
        ns_swa = 2 if batch_size >= 96 else 4
        ns_full = 4 if batch_size >= 96 else 8
        return {
            "batch_size": batch_size,
            "l_sys": l_sys,
            "k_sys": prefix_kv["k_sys"],
            "v_sys": prefix_kv["v_sys"],
            "layers": layers,
            "off_b": off_b,
            "ns_swa": ns_swa,
            "ns_full": ns_full,
            "meta_swa": mx.array([l_sys, swa_ring_cap, self.window_size, 1, ns_swa, swa_ring_cap - 1], dtype=mx.int32),
            "meta_full": mx.array([l_sys, doc_cap_full, 0, 0, ns_full, -1], dtype=mx.int32),
        }

    def forward_step_ragged(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        *,
        apply_softcap: bool = True,
        keep_from_col: int = 0,
    ) -> mx.array:
        """Full 35-layer ragged batched step for `L_q >= 1` (`1` decode, `K` verify, or `keep_from_col` causal slice jump)."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if keep_from_col < 0 or keep_from_col >= L_q:
            raise ValueError(f"Expected 0 <= keep_from_col < L_q ({L_q}), got {keep_from_col}")
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        vec4_norm = self.use_vec4_norm and (self.act_dtype == mx.bfloat16)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]
            cap = c["cap"]
            ring_cap = cap if c["ring"] else 0

            if not l["is_shared"]:
                if i == split_idx and keep_from_col > 0:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = l["q_proj_split"](x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)
                else:
                    qkv = l["qkv_proj"](x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            else:
                q_raw = l["q_proj"](x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)

            if l["is_full"]:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0,
                    doc_ring_cap=0,
                    n_splits=cache["ns_full"],
                    meta=cache["meta_full"],
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=self.window_size,
                    doc_ring_cap=cap,
                    n_splits=cache["ns_swa"],
                    meta=cache["meta_swa"],
                    vec4_reduce=self.vec4_reduce,
                )

            attn_out = l["o_proj"](att)
            if vec4_norm:
                h_mid, mlp_in = fused_gemma4_res_rms_bf16(h, attn_out, l["w_post_attn"], l["w_pre_ff"], eps=self.eps)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_gemma4_res_rms_bf16(h_mid, mlp_out, l["w_post_ff"], eps=self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                if i + 1 < self.num_layers:
                    h, x_in = fused_gemma4_res_rms_bf16(
                        h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], eps=self.eps, scalar=l["layer_scalar"]
                    )
                else:
                    x_in = fused_gemma4_res_rms_bf16(
                        h_ff,
                        ple_out,
                        l["w_post_ple"],
                        l["w_next_norm"],
                        eps=self.eps,
                        scalar=l["layer_scalar"],
                        emit_res=False,
                    )
            else:
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(
                    h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"]
                )

        cache["_last_x_in"] = x_in
        return self._apply_lm_head(x_in, apply_softcap=apply_softcap)

    # ------------------------------------------------------------------
    # GPU Suffix Prefill (Shared KV_sys + EarlyExit14 + Batched LastTok)
    # ------------------------------------------------------------------
    def _prefill_early14_group(
        self,
        sub_toks: list[list[int]],
        doc_indices: list[int],
        cache: dict[str, Any],
        chunk_sz: int = 256,
    ) -> None:
        Bg = len(doc_indices)
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        pre_lens = [len(s) - 1 for s in sub_toks]
        max_pre = max(pre_lens)
        if max_pre <= 0:
            return

        pad_arr = np.zeros((Bg, max_pre), dtype=np.int32)
        for j, s in enumerate(sub_toks):
            if pre_lens[j] > 0:
                pad_arr[j, : pre_lens[j]] = s[: pre_lens[j]]
        toks_mx = mx.array(pad_arr, dtype=mx.int32)
        b_idx = mx.array(doc_indices, dtype=mx.int32)

        for s0 in range(0, max_pre, chunk_sz):
            s1 = min(max_pre, s0 + chunk_sz)
            lc = s1 - s0
            off = l_sys + s0
            off_meta = mx.array([off, lc], dtype=mx.int32)
            off_bg = mx.full((Bg,), off, dtype=mx.int32)
            tok_c = toks_mx[:, s0:s1]
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                c = doc_layers[i]
                cap = c["cap"]

                if i == self.first_kv_shared - 1:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_bg, self.eps
                    )
                    if c["ring"]:
                        slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                        c["k"][b_idx[:, None], slots[None, :]] = k_rot
                        c["v"][b_idx[:, None], slots[None, :]] = v_norm
                    else:
                        c["k"][b_idx, s0:s1, :] = k_rot
                        c["v"][b_idx, s0:s1, :] = v_norm
                    break

                qkv = l["qkv_proj"](x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps
                )
                if not c["ring"]:
                    c["k"][b_idx, s0:s1, :] = k_rot
                    c["v"][b_idx, s0:s1, :] = v_norm
                    parts_k = []
                    parts_v = []
                    if l_sys > 0:
                        parts_k.append(mx.broadcast_to(k_sys_l[i][None, None, :, :], (Bg, 1, l_sys, D)))
                        parts_v.append(mx.broadcast_to(v_sys_l[i][None, None, :, :], (Bg, 1, l_sys, D)))
                    if s0 == 0:
                        parts_k.append(k_rot[:, None, :, :])
                        parts_v.append(v_norm[:, None, :, :])
                    else:
                        parts_k.append(c["k"][b_idx, :s1, :][:, None, :, :])
                        parts_v.append(c["v"][b_idx, :s1, :][:, None, :, :])
                    kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
                    vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
                    mk = ("full", off, lc)
                    if mk not in masks:
                        masks[mk] = "causal" if (off == 0 and lc > 1) else create_causal_mask(lc, offset=off, window_size=None)
                else:
                    slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                    t_start = max(0, off - self.window_size + 1)
                    k_parts = []
                    v_parts = []
                    if t_start < l_sys:
                        sys_lo = t_start
                        k_parts.append(
                            mx.broadcast_to(k_sys_l[i][sys_lo:l_sys][None, None, :, :], (Bg, 1, l_sys - sys_lo, D))
                        )
                        v_parts.append(
                            mx.broadcast_to(v_sys_l[i][sys_lo:l_sys][None, None, :, :], (Bg, 1, l_sys - sys_lo, D))
                        )
                    priv_lo = max(0, t_start - l_sys)
                    if priv_lo < s0:
                        p_slots = mx.arange(priv_lo, s0, dtype=mx.int32) & (cap - 1)
                        k_parts.append(c["k"][b_idx[:, None], p_slots[None, :]][:, None, :, :])
                        v_parts.append(c["v"][b_idx[:, None], p_slots[None, :]][:, None, :, :])
                    if priv_lo < s1:
                        chunk_lo = max(0, priv_lo - s0)
                        k_parts.append(k_rot[:, None, :, :] if chunk_lo == 0 else k_rot[:, None, chunk_lo:, :])
                        v_parts.append(v_norm[:, None, :, :] if chunk_lo == 0 else v_norm[:, None, chunk_lo:, :])
                    c["k"][b_idx[:, None], slots[None, :]] = k_rot
                    c["v"][b_idx[:, None], slots[None, :]] = v_norm
                    kf = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                    vf = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                    mk = ("swa", off - t_start, lc)
                    if mk not in masks:
                        masks[mk] = create_causal_mask(lc, offset=(off - t_start), window_size=self.window_size)

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(Bg, lc, 8 * D)
                )
                attn_out = l["o_proj"](att)
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"])
            mx.eval(*[(c["k"], c["v"]) for c in doc_layers])

    def prefill_suffix_batch(
        self,
        suf_ids_list: list[list[int]],
        cache: dict[str, Any],
        *,
        group_size: int = 16,
        apply_softcap: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Prefill unique doc suffixes via Shared KV_sys + EarlyExit14 + batched LastTok."""
        B = len(suf_ids_list)
        if B != cache["batch_size"]:
            raise ValueError(f"Batch size mismatch: {B} vs cache {cache['batch_size']}")
        l_sys = cache["l_sys"]

        order = sorted(range(B), key=lambda b: len(suf_ids_list[b]))
        g0 = 0
        while g0 < B:
            l0 = len(suf_ids_list[order[g0]])
            g1 = g0 + 1
            while g1 < B and (g1 - g0) < group_size and (len(suf_ids_list[order[g1]]) - l0) <= 48:
                g1 += 1
            idxs = order[g0:g1]
            sub_toks = [suf_ids_list[b] for b in idxs]
            self._prefill_early14_group(sub_toks, idxs, cache, chunk_sz=256)
            g0 = g1

        last_toks = mx.array([suf_ids_list[b][-1] for b in range(B)], dtype=mx.int32)[:, None]
        last_off = mx.array([l_sys + len(suf_ids_list[b]) - 1 for b in range(B)], dtype=mx.int32)
        lg = self.forward_step_ragged(last_toks, cache, last_off, apply_softcap=apply_softcap)
        next_off = last_off + 1
        mx.eval(lg, next_off)
        return lg, next_off

    # ------------------------------------------------------------------
    # ANE KV Prep, Ingestion, Cache Compaction & EarlyExit14 Tail
    # ------------------------------------------------------------------
    def ingest_ane_chunk_kv(
        self,
        cache: dict[str, Any],
        doc_indices: Sequence[int],
        priv_off: int,
        k_swa_f16: Any,
        v_swa_f16: Any,
        k_full_f16: Any,
        v_full_f16: Any,
        *,
        valid_len: int = 256,
        eval_cache: bool = True,
    ) -> None:
        b_idx = mx.array(doc_indices, dtype=mx.int32)
        ks_mx = mx.array(k_swa_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        vs_mx = mx.array(v_swa_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        kf_mx = mx.array(k_full_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        vf_mx = mx.array(v_full_f16[:, :, :valid_len, :]).astype(self.act_dtype)
        use_kv_i8 = bool(cache.get("kv_i8", False))

        doc_layers = cache["layers"]
        for slot_i, l_idx in enumerate(self.swa_layers):
            c = doc_layers[l_idx]
            cap = c["cap"]
            slots = mx.arange(priv_off, priv_off + valid_len, dtype=mx.int32) & (cap - 1)
            k_sl = ks_mx[:, slot_i, :, :]
            v_sl = vs_mx[:, slot_i, :, :]
            if use_kv_i8 and "k_s" in c:
                kq, ks, vq, vs = quantize_kv_pair_i8(k_sl, v_sl)
                c["k"][b_idx[:, None], slots[None, :]] = kq
                c["k_s"][b_idx[:, None], slots[None, :]] = ks
                c["v"][b_idx[:, None], slots[None, :]] = vq
                c["v_s"][b_idx[:, None], slots[None, :]] = vs
            else:
                c["k"][b_idx[:, None], slots[None, :]] = k_sl
                c["v"][b_idx[:, None], slots[None, :]] = v_sl
        for slot_i, l_idx in enumerate(self.full_layers):
            c = doc_layers[l_idx]
            k_sl = kf_mx[:, slot_i, :, :]
            v_sl = vf_mx[:, slot_i, :, :]
            if use_kv_i8 and "k_s" in c:
                kq, ks, vq, vs = quantize_kv_pair_i8(k_sl, v_sl)
                c["k"][b_idx, priv_off : priv_off + valid_len, :] = kq
                c["k_s"][b_idx, priv_off : priv_off + valid_len] = ks
                c["v"][b_idx, priv_off : priv_off + valid_len, :] = vq
                c["v_s"][b_idx, priv_off : priv_off + valid_len] = vs
            else:
                c["k"][b_idx, priv_off : priv_off + valid_len, :] = k_sl
                c["v"][b_idx, priv_off : priv_off + valid_len, :] = v_sl
        if eval_cache:
            self._eval_doc_layers(doc_layers, use_kv_i8 and bool(doc_layers and "k_s" in doc_layers[0]))

    def compact_cache(self, cache: dict[str, Any], keep_indices: Sequence[int]) -> dict[str, Any]:
        idx = mx.array(keep_indices, dtype=mx.int32)
        B_new = len(keep_indices)
        l_sys = cache["l_sys"]
        swa_cap = cache["layers"][self.swa_layers[0]]["cap"]
        full_cap = cache["layers"][self.full_layers[0]]["cap"]
        new_layers = [{"k": c["k"][idx], "v": c["v"][idx], "cap": c["cap"], "ring": c["ring"]} for c in cache["layers"]]
        ns_swa = 2 if B_new >= 96 else 4
        ns_full = 4 if B_new >= 96 else 8
        mx.eval(*[(c["k"], c["v"]) for c in new_layers])
        return {
            "batch_size": B_new,
            "l_sys": l_sys,
            "k_sys": cache["k_sys"],
            "v_sys": cache["v_sys"],
            "layers": new_layers,
            "off_b": cache["off_b"][idx],
            "ns_swa": ns_swa,
            "ns_full": ns_full,
            "meta_swa": mx.array([l_sys, swa_cap, self.window_size, 1, ns_swa, swa_cap - 1], dtype=mx.int32),
            "meta_full": mx.array([l_sys, full_cap, 0, 0, ns_full, -1], dtype=mx.int32),
        }

    def forward_early14_ragged(self, toks: mx.array, cache: dict[str, Any], off_b: mx.array) -> None:
        """Populate KV layers `0..14` only (`L_q >= 1`), exiting at L14 `kv_proj_only`."""
        B, L_q = toks.shape
        l_sys = cache["l_sys"]
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]

        h, pli = self._compute_embed_and_pli_early14(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)

        for i in range(self.first_kv_shared):
            l = self.layers[i]
            D = l["head_dim"]
            c = doc_layers[i]
            cap = c["cap"]
            ring_cap = cap if c["ring"] else 0

            if i == self.first_kv_shared - 1:
                kv = l["kv_proj_only"](x_in)
                k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                    kv, l["w_kn"], l["inv_freq"], off_b, self.eps
                )
                scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
                break

            qkv = l["qkv_proj"](x_in)
            q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
            )
            scatter_kv_ragged(c["k"], c["v"], k_rot, v_norm, off_b, l_sys, ring_cap)
            if l["is_full"]:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[i],
                    v_sys_l[i],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=0,
                    doc_ring_cap=0,
                    n_splits=cache["ns_full"],
                    meta=cache["meta_full"],
                    vec4_reduce=self.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[i],
                    v_sys_l[i],
                    c["k"],
                    c["v"],
                    off_b,
                    window_size=self.window_size,
                    doc_ring_cap=cap,
                    n_splits=cache["ns_swa"],
                    meta=cache["meta_swa"],
                    vec4_reduce=self.vec4_reduce,
                )
            attn_out = l["o_proj"](att)
            h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
            mlp_out = self._run_mlp(mlp_in, l)
            h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
            ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
            ple_out = l["ple_proj"](ple_g)
            h, x_in = fused_norm_add_norm(h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"])

    # ------------------------------------------------------------------
    # Slot-Indexed Continuous Cache (BF16 or INT8) + EarlyExit14 & Prefix Split
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_doc_layers(doc_layers: list[dict[str, Any]], kv_i8: bool) -> None:
        arrs: list[mx.array] = []
        if kv_i8:
            for c in doc_layers:
                arrs.extend((c["k"], c["k_s"], c["v"], c["v_s"]))
        else:
            for c in doc_layers:
                arrs.extend((c["k"], c["v"]))
        mx.eval(*arrs)

    def make_slot_cache(
        self,
        prefix_kv: dict[str, Any],
        batch_size: int,
        doc_cap_full: int = 1440,
        swa_ring_cap: int = 512,
        kv_i8: bool | None = None,
    ) -> dict[str, Any]:
        """Allocate fixed `B_slots` persistent KV cache for continuous rolling scheduling."""
        if swa_ring_cap < self.window_size or (swa_ring_cap & (swa_ring_cap - 1)) != 0:
            raise ValueError(f"swa_ring_cap={swa_ring_cap} must be a power of 2 >= {self.window_size}")
        use_kv_i8 = self.kv_i8 if kv_i8 is None else bool(kv_i8)
        # +16 slack allows padded EarlyExit14 FF drains up to doc_cap_full without bounds issues.
        full_cap_alloc = doc_cap_full + 16
        layers: list[dict[str, Any]] = []
        eval_arrs: list[mx.array] = []
        for i in range(self.first_kv_shared):
            l = self.layers[i]
            D = l["head_dim"]
            ring = not l["is_full"]
            cap = swa_ring_cap if ring else full_cap_alloc
            if use_kv_i8:
                c: dict[str, Any] = {
                    "k": mx.zeros((batch_size, cap, D), dtype=mx.int8),
                    "k_s": mx.ones((batch_size, cap), dtype=mx.float16),
                    "v": mx.zeros((batch_size, cap, D), dtype=mx.int8),
                    "v_s": mx.ones((batch_size, cap), dtype=mx.float16),
                    "cap": cap,
                    "ring": ring,
                }
                eval_arrs.extend((c["k"], c["k_s"], c["v"], c["v_s"]))
            else:
                c = {
                    "k": mx.zeros((batch_size, cap, D), dtype=self.act_dtype),
                    "v": mx.zeros((batch_size, cap, D), dtype=self.act_dtype),
                    "cap": cap,
                    "ring": ring,
                }
                eval_arrs.extend((c["k"], c["v"]))
            layers.append(c)
        mx.eval(*eval_arrs)

        return {
            "batch_size": batch_size,
            "l_sys": int(prefix_kv["l_sys"]),
            "k_sys": prefix_kv["k_sys"],
            "v_sys": prefix_kv["v_sys"],
            "layers": layers,
            "swa_cap": swa_ring_cap,
            "full_cap": full_cap_alloc,
            "kv_i8": use_kv_i8,
            "meta_cache": {},
        }

    def _get_sdpa_meta(self, cache: dict[str, Any], rows_cur: int) -> tuple[int, int, mx.array, mx.array]:
        ns_swa = 2 if rows_cur >= 96 else 4
        ns_full = 4 if rows_cur >= 96 else 8
        key = (ns_swa, ns_full)
        m = cache["meta_cache"].get(key)
        if m is None:
            l_sys = cache["l_sys"]
            swa_cap = cache["swa_cap"]
            full_cap = cache["full_cap"]
            m = (
                ns_swa,
                ns_full,
                mx.array([l_sys, swa_cap, self.window_size, 1, ns_swa, swa_cap - 1], dtype=mx.int32),
                mx.array([l_sys, full_cap, 0, 0, ns_full, -1], dtype=mx.int32),
            )
            mx.eval(m[2], m[3])
            cache["meta_cache"][key] = m
        return m

    @staticmethod
    def _precompute_scatter_pos(off_b: mx.array, l_sys: int, swa_cap: int, L_q: int) -> tuple[mx.array, mx.array]:
        if L_q == 1:
            priv = off_b - l_sys
        else:
            priv = off_b[:, None] + mx.arange(L_q, dtype=mx.int32)[None, :] - l_sys
        return priv & (swa_cap - 1), priv

    def forward_early14_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
    ) -> None:
        """Advance active slots through Layers `0..14` KV cache only (`L_q >= 1`)."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        rows = B * L_q
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx if L_q == 1 else slot_idx[:, None]

        h, pli = self._compute_embed_and_pli_early14(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)

        for i in range(self.first_kv_shared):
            l = self.layers[i]
            c = doc_layers[i]
            pos_slot = swa_slot if c["ring"] else full_slot

            if i == self.first_kv_shared - 1:
                kv = l["kv_proj_only"](x_in)
                k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                    kv, l["w_kn"], l["inv_freq"], off_b, self.eps
                )
                if use_kv_i8:
                    scatter_kv_i8_slot_precomputed(
                        c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                    )
                else:
                    scatter_kv_slot_precomputed(
                        c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                    )
                break

            qkv = l["qkv_proj"](x_in)
            q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
            )
            if use_kv_i8:
                scatter_kv_i8_slot_precomputed(
                    c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                )
            else:
                scatter_kv_slot_precomputed(
                    c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                )

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[i],
                v_sys_l[i],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )
            attn_out = l["o_proj"](att)
            h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
            mlp_out = self._run_mlp(mlp_in, l)
            h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
            ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
            ple_out = l["ple_proj"](ple_g)
            h, x_in = fused_norm_add_norm(h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"])

        self._eval_doc_layers(doc_layers, use_kv_i8)

    def forward_step_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        apply_softcap: bool = True,
        keep_from_col: int = 0,
    ) -> mx.array:
        """Full 35-layer slot-indexed step for `L_q >= 1` -> `[B, L_q - keep_from_col, V]`."""
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if keep_from_col < 0 or keep_from_col >= L_q:
            raise ValueError(f"Expected 0 <= keep_from_col < L_q ({L_q}), got {keep_from_col}")
        rows = B * L_q
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx if L_q == 1 else slot_idx[:, None]

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        vec4_norm = self.use_vec4_norm and (self.act_dtype == mx.bfloat16)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                if i == split_idx and keep_from_col > 0:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = l["q_proj_split"](x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B * (L_q - keep_from_col))
                else:
                    qkv = l["qkv_proj"](x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
            else:
                q_raw = l["q_proj"](x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = l["o_proj"](att)
            if vec4_norm:
                h_mid, mlp_in = fused_gemma4_res_rms_bf16(h, attn_out, l["w_post_attn"], l["w_pre_ff"], eps=self.eps)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_gemma4_res_rms_bf16(h_mid, mlp_out, l["w_post_ff"], eps=self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                if i + 1 < self.num_layers:
                    h, x_in = fused_gemma4_res_rms_bf16(
                        h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], eps=self.eps, scalar=l["layer_scalar"]
                    )
                else:
                    x_in = fused_gemma4_res_rms_bf16(
                        h_ff,
                        ple_out,
                        l["w_post_ple"],
                        l["w_next_norm"],
                        eps=self.eps,
                        scalar=l["layer_scalar"],
                        emit_res=False,
                    )
            else:
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(
                    h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"]
                )

        cache["_last_x_in"] = x_in
        return self._apply_lm_head(x_in, apply_softcap=apply_softcap)

    def forward_ff_jump_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        return_logits: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Single-launch causal slice jump over `k >= 1` forced `llguidance` tokens (`[B, k]`).

        Advances KV cache across all `k` tokens on layers `0..first_kv_shared-1`, slices to the final
        token `[:, -1:, :]` at `first_kv_shared - 1`, and evaluates `q_proj_split` + `L15..34` + `lm_head`
        ONLY on the final token (`[B, 1]`).
        """
        B, L_q = int(toks.shape[0]), int(toks.shape[1])
        if L_q == 1:
            return self.forward_decode_step_slot(
                toks, cache, off_b, slot_idx, n_keep=B, return_logits=return_logits
            )
        rows = B * L_q
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, rows)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, L_q)
        s_idx_sc = slot_idx[:, None]

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        keep_from_col = L_q - 1
        vec4_norm = self.use_vec4_norm and (self.act_dtype == mx.bfloat16)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                if i == split_idx:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    x_in = mx.contiguous(x_in[:, keep_from_col:, :])
                    h = mx.contiguous(h[:, keep_from_col:, :])
                    pli = mx.contiguous(pli[:, keep_from_col:, :, :])
                    off_b = off_b + int(keep_from_col)
                    q_raw = l["q_proj_split"](x_in)
                    q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
                else:
                    qkv = l["qkv_proj"](x_in)
                    q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                        qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
                    )
                    if use_kv_i8:
                        scatter_kv_i8_slot_precomputed(
                            c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
                    else:
                        scatter_kv_slot_precomputed(
                            c["k"], c["v"], k_rot, v_norm, s_idx_sc, pos_slot
                        )
            else:
                q_raw = l["q_proj"](x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = l["o_proj"](att)
            if vec4_norm:
                h_mid, mlp_in = fused_gemma4_res_rms_bf16(h, attn_out, l["w_post_attn"], l["w_pre_ff"], eps=self.eps)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_gemma4_res_rms_bf16(h_mid, mlp_out, l["w_post_ff"], eps=self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                if i + 1 < self.num_layers:
                    h, x_in = fused_gemma4_res_rms_bf16(
                        h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], eps=self.eps, scalar=l["layer_scalar"]
                    )
                else:
                    x_in = fused_gemma4_res_rms_bf16(
                        h_ff,
                        ple_out,
                        l["w_post_ple"],
                        l["w_next_norm"],
                        eps=self.eps,
                        scalar=l["layer_scalar"],
                        emit_res=False,
                    )
            else:
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(
                    h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"]
                )

        cache["_last_x_in"] = x_in
        if return_logits:
            lg = self._apply_lm_head(x_in, apply_softcap=True)
            am_mx = mx.argmax(lg[:, 0, :], axis=-1)
            return lg, am_mx
        am_mx = self._apply_lm_head_argmax(x_in)
        return x_in, am_mx

    def forward_decode_step_slot(
        self,
        toks: mx.array,
        cache: dict[str, Any],
        off_b: mx.array,
        slot_idx: mx.array,
        *,
        n_keep: int,
        return_logits: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """
        Run `L_q=1` slot-indexed decode step with Layer-15 Prefix Split (`n_keep <= B`) & lazy logits.
        When `return_logits=False`, returns `(x_samp [n_keep, 1, D], am_mx [n_keep])`.
        When `return_logits=True`, returns `(logits [n_keep, 1, V], am_mx [n_keep])`.
        """
        B = int(toks.shape[0])
        if not (0 < n_keep <= B):
            raise ValueError(f"Expected 0 < n_keep <= B, got n_keep={n_keep}, B={B}")
        l_sys = cache["l_sys"]
        swa_cap = cache["swa_cap"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
        swa_slot, full_slot = self._precompute_scatter_pos(off_b, l_sys, swa_cap, 1)

        h, pli = self._compute_embed_and_pli(toks)
        x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
        split_idx = self.first_kv_shared - 1
        vec4_norm = self.use_vec4_norm and (self.act_dtype == mx.bfloat16)

        for i, l in enumerate(self.layers):
            kv_idx = l["prev_kv"]
            c = doc_layers[kv_idx]

            if not l["is_shared"]:
                pos_slot = swa_slot if c["ring"] else full_slot
                qkv = l["qkv_proj"](x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope_ragged(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_b, self.eps
                )
                if use_kv_i8:
                    scatter_kv_i8_slot_precomputed(
                        c["k"], c["k_s"], c["v"], c["v_s"], k_rot, v_norm, slot_idx, pos_slot
                    )
                else:
                    scatter_kv_slot_precomputed(
                        c["k"], c["v"], k_rot, v_norm, slot_idx, pos_slot
                    )
                if i == split_idx and n_keep < B:
                    q_rot = q_rot[:n_keep]
                    h = h[:n_keep]
                    pli = pli[:n_keep]
                    off_b = off_b[:n_keep]
                    slot_idx = slot_idx[:n_keep]
                    B = n_keep
                    ns_swa, ns_full, meta_swa, meta_full = self._get_sdpa_meta(cache, B)
            else:
                q_raw = l["q_proj"](x_in)
                q_rot = fused_q_norm_rope_ragged(q_raw, l["w_qn"], l["inv_freq"], off_b, self.eps)

            att = mqa8_sdpa_prefix_ragged_slot(
                q_rot,
                k_sys_l[kv_idx],
                v_sys_l[kv_idx],
                c["k"],
                c["v"],
                off_b,
                slot_idx,
                ns_full if l["is_full"] else ns_swa,
                meta_full if l["is_full"] else meta_swa,
                k_s_doc=c.get("k_s"),
                v_s_doc=c.get("v_s"),
                vec4_reduce=self.vec4_reduce,
            )

            attn_out = l["o_proj"](att)
            if vec4_norm:
                h_mid, mlp_in = fused_gemma4_res_rms_bf16(h, attn_out, l["w_post_attn"], l["w_pre_ff"], eps=self.eps)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_gemma4_res_rms_bf16(h_mid, mlp_out, l["w_post_ff"], eps=self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                if i + 1 < self.num_layers:
                    h, x_in = fused_gemma4_res_rms_bf16(
                        h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], eps=self.eps, scalar=l["layer_scalar"]
                    )
                else:
                    x_in = fused_gemma4_res_rms_bf16(
                        h_ff,
                        ple_out,
                        l["w_post_ple"],
                        l["w_next_norm"],
                        eps=self.eps,
                        scalar=l["layer_scalar"],
                        emit_res=False,
                    )
            else:
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(
                    h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"]
                )

        cache["_last_x_in"] = x_in
        if return_logits:
            lg = self._apply_lm_head(x_in, apply_softcap=True)
            am_mx = mx.argmax(lg[:, 0, :], axis=-1)
            return lg, am_mx
        am_mx = self._apply_lm_head_argmax(x_in)
        return x_in, am_mx

    @staticmethod
    def _real_token_pairs(pre_lens: Sequence[int], s0: int, s1: int) -> tuple[np.ndarray, np.ndarray] | None:
        """`(doc row, chunk column)` of the real tokens in chunk `[s0, s1)`, or None when every doc fills it."""
        if min(pre_lens) >= s1:
            return None
        rows, cols = np.nonzero(np.arange(s0, s1)[None, :] < np.asarray(pre_lens)[:, None])
        return rows.astype(np.int32), cols.astype(np.int32)

    @staticmethod
    def _scatter_ring_chunk(c: dict[str, Any], b_idx: Any, slots: Any, values: dict[str, Any], real: Any) -> None:
        """Write one prefill chunk into a sliding-window ring layer, real tokens only.

        A group is right-padded to its longest doc. A pad position `g` of a shorter doc wraps to the slot of
        position `g - cap`, which that doc's next queries still attend when `cap` equals the window, so pads are
        never written (`real` from `_real_token_pairs`; None when the chunk has no pads).
        """
        if real is None:
            for key, val in values.items():
                c[key][b_idx[:, None], slots[None, :]] = val
            return
        rows, cols = real
        bb, ss = b_idx[rows], slots[cols]
        for key, val in values.items():
            c[key][bb, ss] = val[rows, cols]

    @staticmethod
    def _kv_cache_values(k_rot: mx.array, v_norm: mx.array, use_kv_i8: bool) -> dict[str, mx.array]:
        if not use_kv_i8:
            return {"k": k_rot, "v": v_norm}
        kq, ks, vq, vs = quantize_kv_pair_i8(k_rot, v_norm)
        return {"k": kq, "k_s": ks, "v": vq, "v_s": vs}

    def _prefill_early14_slot_group(
        self,
        sub_toks: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        chunk_sz: int = 256,
        start_s0: int = 0,
    ) -> None:
        Bg = len(slot_indices)
        l_sys = cache["l_sys"]
        use_kv_i8 = bool(cache.get("kv_i8", self.kv_i8))
        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        pre_lens = [len(s) - 1 for s in sub_toks]
        max_pre = max(pre_lens)
        if max_pre <= start_s0:
            return

        pad_arr = np.zeros((Bg, max_pre), dtype=np.int32)
        for j, s in enumerate(sub_toks):
            if pre_lens[j] > 0:
                pad_arr[j, : pre_lens[j]] = s[: pre_lens[j]]
        toks_mx = mx.array(pad_arr, dtype=mx.int32)
        b_idx = mx.array(slot_indices, dtype=mx.int32)

        for s0 in range(start_s0, max_pre, chunk_sz):
            s1 = min(max_pre, s0 + chunk_sz)
            lc = s1 - s0
            off = l_sys + s0
            off_meta = mx.array([off, lc], dtype=mx.int32)
            off_bg = mx.full((Bg,), off, dtype=mx.int32)
            tok_c = toks_mx[:, s0:s1]
            real_np = self._real_token_pairs(pre_lens, s0, s1)
            real = None if real_np is None else (mx.array(real_np[0]), mx.array(real_np[1]))
            h, pli = self._compute_embed_and_pli_early14(tok_c)
            x_in = mx.fast.rms_norm(h, self.layers[0]["w_in"], self.eps)
            masks: dict[tuple[str, int, int], Any] = {}

            for i in range(self.first_kv_shared):
                l = self.layers[i]
                D = l["head_dim"]
                c = doc_layers[i]
                cap = c["cap"]

                if i == self.first_kv_shared - 1:
                    kv = l["kv_proj_only"](x_in)
                    k_rot, v_norm = fused_kv_only_norm_rope_ragged(
                        kv, l["w_kn"], l["inv_freq"], off_bg, self.eps
                    )
                    if c["ring"]:
                        slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                        self._scatter_ring_chunk(c, b_idx, slots, self._kv_cache_values(k_rot, v_norm, use_kv_i8), real)
                    else:
                        if use_kv_i8:
                            kq, ks, vq, vs = quantize_kv_pair_i8(k_rot, v_norm)
                            c["k"][b_idx, s0:s1, :] = kq
                            c["k_s"][b_idx, s0:s1] = ks
                            c["v"][b_idx, s0:s1, :] = vq
                            c["v_s"][b_idx, s0:s1] = vs
                        else:
                            c["k"][b_idx, s0:s1, :] = k_rot
                            c["v"][b_idx, s0:s1, :] = v_norm
                    break

                qkv = l["qkv_proj"](x_in)
                q_rot, k_rot, v_norm = fused_qkv_packed_norm_rope(
                    qkv, l["w_qn"], l["w_kn"], l["inv_freq"], off_meta, self.eps
                )
                if not c["ring"]:
                    if use_kv_i8:
                        kq, ks, vq, vs = quantize_kv_pair_i8(k_rot, v_norm)
                        c["k"][b_idx, s0:s1, :] = kq
                        c["k_s"][b_idx, s0:s1] = ks
                        c["v"][b_idx, s0:s1, :] = vq
                        c["v_s"][b_idx, s0:s1] = vs
                        if s0 == 0:
                            kd = k_rot[:, None, :, :]
                            vd = v_norm[:, None, :, :]
                        else:
                            full_pos = mx.arange(0, s1, dtype=mx.int32)
                            kd, vd = gather_dequant_kv_pair_i8_bf16(
                                c["k"], c["k_s"], c["v"], c["v_s"], b_idx, full_pos
                            )
                    else:
                        c["k"][b_idx, s0:s1, :] = k_rot
                        c["v"][b_idx, s0:s1, :] = v_norm
                        if s0 == 0:
                            kd = k_rot[:, None, :, :]
                            vd = v_norm[:, None, :, :]
                        else:
                            kd = c["k"][b_idx, :s1, :][:, None, :, :]
                            vd = c["v"][b_idx, :s1, :][:, None, :, :]
                    parts_k = []
                    parts_v = []
                    if l_sys > 0:
                        parts_k.append(mx.broadcast_to(k_sys_l[i][None, None, :, :], (Bg, 1, l_sys, D)))
                        parts_v.append(mx.broadcast_to(v_sys_l[i][None, None, :, :], (Bg, 1, l_sys, D)))
                    parts_k.append(kd)
                    parts_v.append(vd)
                    kf = parts_k[0] if len(parts_k) == 1 else mx.concatenate(parts_k, axis=2)
                    vf = parts_v[0] if len(parts_v) == 1 else mx.concatenate(parts_v, axis=2)
                    mk = ("full", off, lc)
                    if mk not in masks:
                        masks[mk] = "causal" if (off == 0 and lc > 1) else create_causal_mask(lc, offset=off, window_size=None)
                else:
                    slots = mx.arange(s0, s1, dtype=mx.int32) & (cap - 1)
                    t_start = max(0, off - self.window_size + 1)
                    k_parts = []
                    v_parts = []
                    if t_start < l_sys:
                        sys_lo = t_start
                        k_parts.append(
                            mx.broadcast_to(k_sys_l[i][sys_lo:l_sys][None, None, :, :], (Bg, 1, l_sys - sys_lo, D))
                        )
                        v_parts.append(
                            mx.broadcast_to(v_sys_l[i][sys_lo:l_sys][None, None, :, :], (Bg, 1, l_sys - sys_lo, D))
                        )
                    priv_lo = max(0, t_start - l_sys)
                    if priv_lo < s0:
                        p_slots = mx.arange(priv_lo, s0, dtype=mx.int32) & (cap - 1)
                        if use_kv_i8:
                            kd_old, vd_old = gather_dequant_kv_pair_i8_bf16(
                                c["k"], c["k_s"], c["v"], c["v_s"], b_idx, p_slots
                            )
                        else:
                            kd_old = c["k"][b_idx[:, None], p_slots[None, :]][:, None, :, :]
                            vd_old = c["v"][b_idx[:, None], p_slots[None, :]][:, None, :, :]
                        k_parts.append(kd_old)
                        v_parts.append(vd_old)
                    if priv_lo < s1:
                        chunk_lo = max(0, priv_lo - s0)
                        k_parts.append(k_rot[:, None, :, :] if chunk_lo == 0 else k_rot[:, None, chunk_lo:, :])
                        v_parts.append(v_norm[:, None, :, :] if chunk_lo == 0 else v_norm[:, None, chunk_lo:, :])
                    self._scatter_ring_chunk(c, b_idx, slots, self._kv_cache_values(k_rot, v_norm, use_kv_i8), real)
                    kf = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                    vf = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                    mk = ("swa", off - t_start, lc)
                    if mk not in masks:
                        masks[mk] = create_causal_mask(lc, offset=(off - t_start), window_size=self.window_size)

                att = (
                    mx.fast.scaled_dot_product_attention(
                        q_rot.transpose(0, 2, 1, 3), kf, vf, scale=1.0, mask=masks[mk]
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(Bg, lc, 8 * D)
                )
                attn_out = l["o_proj"](att)
                h_mid, mlp_in = fused_norm_add_norm(h, attn_out, l["w_post_attn"], l["w_pre_ff"], self.eps, None)
                mlp_out = self._run_mlp(mlp_in, l)
                h_ff = fused_res_rms_norm(h_mid, mlp_out, l["w_post_ff"], self.eps)
                ple_g = fused_ple_geglu_indexed(l["ple_gate"](h_ff), pli, self._layer_idx_mx[i])
                ple_out = l["ple_proj"](ple_g)
                h, x_in = fused_norm_add_norm(h_ff, ple_out, l["w_post_ple"], l["w_next_norm"], self.eps, l["layer_scalar"])
            self._eval_doc_layers(doc_layers, use_kv_i8)

    def _prefill_early14_slot_groups(
        self,
        suf_ids_list: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        group_size: int = 16,
        start_s0: int = 0,
    ) -> None:
        """Prefill in length-sorted groups whose suffixes differ by at most 48 tokens.

        A group is right-padded to its longest member, so a small spread keeps wasted pad compute low. Pads are
        not written to the sliding-window ring layers (`_scatter_ring_chunk`), where they would wrap onto a
        shorter doc's real recent context.
        """
        K = len(suf_ids_list)
        if K != len(slot_indices):
            raise ValueError(f"Mismatched K={K} vs slot_indices={len(slot_indices)}")
        order = sorted(range(K), key=lambda b: len(suf_ids_list[b]))
        g0 = 0
        while g0 < K:
            l0 = len(suf_ids_list[order[g0]])
            g1 = g0 + 1
            while g1 < K and (g1 - g0) < group_size and (len(suf_ids_list[order[g1]]) - l0) <= 48:
                g1 += 1
            sub_idxs = order[g0:g1]
            self._prefill_early14_slot_group(
                [suf_ids_list[b] for b in sub_idxs],
                [slot_indices[b] for b in sub_idxs],
                cache,
                chunk_sz=256,
                start_s0=start_s0,
            )
            g0 = g1

    def prefill_slots_lazy_lg(
        self,
        suf_ids_list: list[list[int]],
        slot_indices: list[int],
        cache: dict[str, Any],
        *,
        group_size: int = 16,
        return_logits: bool = False,
        async_eval: bool = False,
    ) -> tuple[mx.array, mx.array, list[int]]:
        """Prefill `K` docs into `slot_indices` and run batched `LastTok` (`return_logits=False` skips `[K, 1, V]`)."""
        K = len(suf_ids_list)
        l_sys = cache["l_sys"]
        self._prefill_early14_slot_groups(suf_ids_list, slot_indices, cache, group_size=group_size)
        last_toks = mx.array([suf_ids_list[b][-1] for b in range(K)], dtype=mx.int32)[:, None]
        last_off = mx.array([l_sys + len(suf_ids_list[b]) - 1 for b in range(K)], dtype=mx.int32)
        slot_mx = mx.array(slot_indices, dtype=mx.int32)
        x_or_lg, am = self.forward_decode_step_slot(
            last_toks, cache, last_off, slot_mx, n_keep=K, return_logits=return_logits
        )
        next_off = [l_sys + len(suf_ids_list[b]) for b in range(K)]
        if async_eval:
            mx.async_eval(x_or_lg, am)
        else:
            mx.eval(x_or_lg, am)
        return x_or_lg, am, next_off

    def build_cpu_ane_embed_tables(self):
        """One-time dequantization of `embed_tokens` + PLI layers `0..13` to CPU FP16 tables (~2.68 GB)."""
        import numpy as np

        emb = self.embed_tokens
        pli = self.embed_tokens_per_layer
        if isinstance(emb, nn.QuantizedEmbedding):
            w_emb = (
                mx.dequantize(emb.weight, emb.scales, emb.biases, group_size=emb.group_size, bits=emb.bits, mode=emb.mode)
                * self.embed_scale
            ).astype(mx.float16)
        else:
            w_emb = (emb.weight * self.embed_scale).astype(mx.float16)

        n_pli_cols = (self.first_kv_shared - 1) * self.ple_dim  # 14 * 256 = 3584
        if isinstance(pli, nn.QuantizedEmbedding):
            pack_div = 32 // pli.bits
            w_words = n_pli_cols // pack_div
            g_cnt = n_pli_cols // pli.group_size
            w_pli14 = (
                mx.dequantize(
                    pli.weight[:, :w_words],
                    pli.scales[:, :g_cnt],
                    pli.biases[:, :g_cnt] if pli.biases is not None else None,
                    group_size=pli.group_size,
                    bits=pli.bits,
                    mode=pli.mode,
                )
                * self.embed_tokens_per_layer_scale
            ).astype(mx.float16)
        else:
            w_pli14 = (pli.weight[:, :n_pli_cols] * self.embed_tokens_per_layer_scale).astype(mx.float16)

        mx.eval(w_emb, w_pli14)
        return np.array(w_emb, copy=True), np.array(w_pli14, copy=True)

    # ------------------------------------------------------------------
    # Standalone Exact Export (Zero mlx_vlm) for CoreML ANE Compilation
    # ------------------------------------------------------------------
    def export_early14_f16(self, out_npz: str | Path) -> Path:
        """Export EarlyExit14 layers `0..14` FP16 weights in CoreML `[O, I, 1, 1]` layout."""
        import numpy as np

        def _lin_w_f16(mod: nn.Module) -> np.ndarray:
            if isinstance(mod, nn.QuantizedLinear):
                w = mx.dequantize(
                    mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits, mode=mod.mode
                ).astype(mx.float16)
            else:
                w = mod.weight.astype(mx.float16)
            mx.eval(w)
            return np.ascontiguousarray(np.asarray(w))[:, :, None, None]

        arrays: dict[str, np.ndarray] = {
            "per_layer_model_proj_w": _lin_w_f16(self.per_layer_model_projection)[: (self.first_kv_shared - 1) * self.ple_dim],
            "per_layer_proj_scale": np.array([float(self.per_layer_projection_scale.item())], dtype=np.float16),
            "per_layer_proj_norm_w": np.array(self.per_layer_projection_norm_w.astype(mx.float16), copy=True),
            "per_layer_input_scale": np.array([float(self.per_layer_input_scale.item())], dtype=np.float16),
        }
        for i in range(self.first_kv_shared):
            l = self.layers[i]
            p = f"l{i}_"
            arrays[p + "w_in"] = np.array(l["w_in"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
            if i == self.first_kv_shared - 1:
                arrays[p + "kv"] = _lin_w_f16(l["kv_proj_only"])
            else:
                arrays[p + "qkv"] = _lin_w_f16(l["qkv_proj"])
            arrays[p + "w_kn"] = np.array(l["w_kn"].astype(mx.float16), copy=True).reshape(1, 1, 1, l["head_dim"])
            if i < self.first_kv_shared - 1:
                arrays[p + "w_qn"] = np.array(l["w_qn"].astype(mx.float16), copy=True).reshape(1, 1, 1, l["head_dim"])
                arrays[p + "wo"] = _lin_w_f16(l["o_proj"])
                arrays[p + "w_pa"] = np.array(l["w_post_attn"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "w_pff"] = np.array(l["w_pre_ff"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "gu"] = _lin_w_f16(l["gate_up_proj"])
                arrays[p + "dn"] = _lin_w_f16(l["down_proj"])
                arrays[p + "w_poff"] = np.array(l["w_post_ff"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "w_pg"] = _lin_w_f16(l["ple_gate"])
                arrays[p + "w_pp"] = _lin_w_f16(l["ple_proj"])
                arrays[p + "w_pple"] = np.array(l["w_post_ple"].astype(mx.float16), copy=True).reshape(1, self.hidden_size, 1, 1)
                arrays[p + "scalar"] = np.array([float(l["layer_scalar"].astype(mx.float32).item())], dtype=np.float16)

        out_p = Path(out_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_p, **arrays)
        return out_p
