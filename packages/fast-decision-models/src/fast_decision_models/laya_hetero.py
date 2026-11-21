"""Metal MLX GPU (`LayaMlxGpuBackend`) & Heterogeneous GPU+ANE (`LayaHeteroEngine`) runners for `Laya-Typed-421M`."""

from __future__ import annotations

from collections import deque
from functools import partial
import json
import math
from pathlib import Path
import threading
from typing import Any, Literal, Mapping, Sequence

from ._common import softmax_2, softmax_vec
from .laya_ane import LayaAneBackend, LayaTokenizerHelper
from .paths import DEFAULT_LAYA_REPO, ensure_laya_dylib, resolve_laya_dir

_LAYA_GPU_CACHE_LIMIT_BYTES = 512 << 20
_LAYA_CLEAR_CACHE_THRESHOLD_BYTES = 256 << 20


def _pack_laya_segments(
    items: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    lane_cap: int = 512,
) -> list[list[int]]:
    """Pack length-sorted item indices into `lane_cap`-wide rows (`Packed-512` FFD / greedy)."""
    desc = sorted(indices, key=lambda i: len(items[i]["ids"]), reverse=True)
    lanes: list[list[int]] = []
    lane_rem: list[int] = []
    for idx in desc:
        ln = len(items[idx]["ids"])
        if ln > lane_cap:
            lanes.append([idx])
            lane_rem.append(0)
            continue
        best_l = -1
        best_rem = lane_cap + 1
        for li, rem in enumerate(lane_rem):
            if ln <= rem and rem < best_rem:
                best_rem = rem
                best_l = li
        if best_l >= 0:
            lanes[best_l].append(idx)
            lane_rem[best_l] -= ln
        else:
            lanes.append([idx])
            lane_rem.append(lane_cap - ln)
    return lanes


def _build_laya_packed_batch_tensors(
    mx: Any,
    np: Any,
    items: Sequence[Mapping[str, Any]],
    lanes: Sequence[Sequence[int]],
    *,
    pad_id: int,
    local_half_win: int,
    qtype_override: int | None = None,
) -> tuple[Any, Any, Any, Any, Any, list[tuple[int, int, int, int]]]:
    """Build 2D `(B, S)` `ids`, `pos_2d`, `qtypes_2d`, and `BF16` `(B, 1, S, S)` block-diagonal masks.

    Avoids `mlx_embeddings` 1D `(B*S,)` `position_ids` bug and `FP32` `-1e9` dense mask upcasts:
    every packed segment resets `pos_2d` to `0..len(seg)-1` and uses exact `mx.bfloat16` `-1e4` masks.
    """
    B = len(lanes)
    raw_max = max(sum(len(items[idx]["ids"]) for idx in ln) for ln in lanes)
    S = max(32, ((raw_max + 31) // 32) * 32)

    ids_np = np.full((B, S), int(pad_id), dtype=np.int32)
    pos_np = np.zeros((B, S), dtype=np.int32)
    seg_np = np.full((B, S), -1, dtype=np.int32)
    qt_np = np.zeros((B, S), dtype=np.int32)
    # List of (orig_item_idx, lane_b, marker_start_in_gather, marker_count, eff_qtype)
    marker_meta: list[tuple[int, int, list[int], int]] = []

    for b_i, lane in enumerate(lanes):
        cursor = 0
        for seg_k, item_idx in enumerate(lane):
            it = items[item_idx]
            toks = it["ids"]
            L = len(toks)
            eff_qt = int(qtype_override) if qtype_override is not None else int(it["qtype"])
            ids_np[b_i, cursor : cursor + L] = toks
            pos_np[b_i, cursor : cursor + L] = np.arange(L, dtype=np.int32)
            seg_np[b_i, cursor : cursor + L] = seg_k
            qt_np[b_i, cursor : cursor + L] = eff_qt
            shifted_m = [cursor + int(m) for m in it["markers"] if (cursor + int(m)) < S]
            marker_meta.append((int(item_idx), b_i, shifted_m, eff_qt))
            cursor += L

    ids_mx = mx.array(ids_np, dtype=mx.int32)
    pos_mx = mx.array(pos_np, dtype=mx.int32)
    seg_mx = mx.array(seg_np, dtype=mx.int32)
    qt_mx = mx.array(qt_np, dtype=mx.int32)

    same_seg = (seg_mx[:, None, :, None] == seg_mx[:, None, None, :]) & (seg_mx[:, None, None, :] >= 0)
    z_bf16 = mx.array(0.0, dtype=mx.bfloat16)
    neg_bf16 = mx.array(-1e4, dtype=mx.bfloat16)
    full_mask_bf16 = mx.where(same_seg, z_bf16, neg_bf16)

    dist = mx.abs(pos_mx[:, None, :, None] - pos_mx[:, None, None, :])
    in_win = same_seg & (dist <= int(local_half_win))
    swa_mask_bf16 = mx.where(in_win, z_bf16, neg_bf16)

    return ids_mx, pos_mx, qt_mx, full_mask_bf16, swa_mask_bf16, marker_meta


def _build_laya_mlx_model(enc_cfg: Mapping[str, Any], raw_w: Mapping[str, Any], *, quant_bits: int = 8, group_size: int = 64) -> Any:
    """Build native MLX `ModernBERT-large` (`8-bit g64`) + quantized 2L `head` + marker-gathered `scorer`."""
    import mlx.core as mx
    import mlx.nn as nn

    H = int(enc_cfg["hidden_size"])
    inter = int(enc_cfg.get("intermediate_size", H * 2))
    n_layers = int(enc_cfg["num_hidden_layers"])
    n_heads = int(enc_cfg["num_attention_heads"])
    hd = H // n_heads
    half_hd = hd // 2
    scale = hd**-0.5
    ln_eps = float(enc_cfg.get("norm_eps", enc_cfg.get("layer_norm_eps", 1e-5)))
    vocab_sz = int(enc_cfg.get("vocab_size", raw_w["encoder.embeddings.tok_embeddings.weight"].shape[0]))
    layer_types = list(enc_cfg["layer_types"])
    rope_p = enc_cfg.get("rope_parameters", {})
    theta_full = float(rope_p.get("full_attention", {}).get("rope_theta", 160000.0))
    theta_swa = float(rope_p.get("sliding_attention", {}).get("rope_theta", 10000.0))

    inv_full = mx.array([1.0 / (theta_full ** (float(i) / float(hd))) for i in range(0, hd, 2)], dtype=mx.float32)
    inv_swa = mx.array([1.0 / (theta_swa ** (float(i) / float(hd))) for i in range(0, hd, 2)], dtype=mx.float32)

    @partial(mx.compile, shapeless=True)
    def _rope_trig_2d(pos_2d: Any, inv_f: Any) -> tuple[Any, Any]:
        ang = pos_2d.astype(mx.float32)[:, None, :, None] * inv_f[None, None, None, :]
        return mx.cos(ang).astype(mx.bfloat16), mx.sin(ang).astype(mx.bfloat16)

    @partial(mx.compile, shapeless=True)
    def _fused_rot_pair(x0: Any, x1: Any, cos_a: Any, sin_a: Any) -> tuple[Any, Any]:
        return x0 * cos_a - x1 * sin_a, x1 * cos_a + x0 * sin_a

    def _apply_rope(x: Any, cos_a: Any, sin_a: Any) -> Any:
        x0, x1 = mx.split(x, [half_hd], axis=-1)
        y0, y1 = _fused_rot_pair(x0, x1, cos_a, sin_a)
        return mx.concatenate([y0, y1], axis=-1)

    @partial(mx.compile, shapeless=True)
    def _exact_geglu(val_h: Any, gate_h: Any) -> Any:
        return nn.gelu(val_h) * gate_h

    class _MbEncLayer(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            px = f"encoder.layers.{idx}."
            self.has_attn_norm = idx > 0
            self.is_full = layer_types[idx] == "full_attention"
            if self.has_attn_norm:
                self.attn_norm_w = raw_w[px + "attn_norm.weight"].astype(mx.bfloat16)
            self.mlp_norm_w = raw_w[px + "mlp_norm.weight"].astype(mx.bfloat16)

            self.wqkv = nn.Linear(H, 3 * H, bias=False)
            self.wqkv.weight = raw_w[px + "attn.Wqkv.weight"].astype(mx.bfloat16)
            self.wo = nn.Linear(H, H, bias=False)
            self.wo.weight = raw_w[px + "attn.Wo.weight"].astype(mx.bfloat16)
            self.wi = nn.Linear(H, 2 * inter, bias=False)
            self.wi.weight = raw_w[px + "mlp.Wi.weight"].astype(mx.bfloat16)
            self.wom = nn.Linear(inter, H, bias=False)
            self.wom.weight = raw_w[px + "mlp.Wo.weight"].astype(mx.bfloat16)

        def __call__(self, x: Any, mask_bf16: Any, cos_a: Any, sin_a: Any) -> Any:
            B, S, _ = x.shape
            xa = mx.fast.layer_norm(x, self.attn_norm_w, None, eps=ln_eps) if self.has_attn_norm else x
            qkv = self.wqkv(xa).reshape(B, S, 3, n_heads, hd)
            q = _apply_rope(qkv[:, :, 0].transpose(0, 2, 1, 3), cos_a, sin_a)
            k = _apply_rope(qkv[:, :, 1].transpose(0, 2, 1, 3), cos_a, sin_a)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
            att = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_bf16)
            x = x + self.wo(att.transpose(0, 2, 1, 3).reshape(B, S, H))
            xm = mx.fast.layer_norm(x, self.mlp_norm_w, None, eps=ln_eps)
            val_h, gate_h = mx.split(self.wi(xm), [inter], axis=-1)
            return x + self.wom(_exact_geglu(val_h, gate_h))

    class _LayaHeadLayer(nn.Module):
        def __init__(self, hi: int) -> None:
            super().__init__()
            hp = f"head.layers.{hi}."
            head_ff = int(raw_w[hp + "linear1.weight"].shape[0])
            self.n1_w = raw_w[hp + "norm1.weight"].astype(mx.bfloat16)
            self.n1_b = raw_w[hp + "norm1.bias"].astype(mx.bfloat16)
            self.n2_w = raw_w[hp + "norm2.weight"].astype(mx.bfloat16)
            self.n2_b = raw_w[hp + "norm2.bias"].astype(mx.bfloat16)

            self.in_proj = nn.Linear(H, 3 * H, bias=True)
            self.in_proj.weight = raw_w[hp + "self_attn.in_proj_weight"].astype(mx.bfloat16)
            self.in_proj.bias = raw_w[hp + "self_attn.in_proj_bias"].astype(mx.bfloat16)
            self.out_proj = nn.Linear(H, H, bias=True)
            self.out_proj.weight = raw_w[hp + "self_attn.out_proj.weight"].astype(mx.bfloat16)
            self.out_proj.bias = raw_w[hp + "self_attn.out_proj.bias"].astype(mx.bfloat16)
            self.l1 = nn.Linear(H, head_ff, bias=True)
            self.l1.weight = raw_w[hp + "linear1.weight"].astype(mx.bfloat16)
            self.l1.bias = raw_w[hp + "linear1.bias"].astype(mx.bfloat16)
            self.l2 = nn.Linear(head_ff, H, bias=True)
            self.l2.weight = raw_w[hp + "linear2.weight"].astype(mx.bfloat16)
            self.l2.bias = raw_w[hp + "linear2.bias"].astype(mx.bfloat16)

        def __call__(self, h: Any, full_mask_bf16: Any) -> Any:
            B, S, _ = h.shape
            hn1 = mx.fast.layer_norm(h, self.n1_w, self.n1_b, eps=1e-5)
            qkv = self.in_proj(hn1).reshape(B, S, 3, n_heads, hd)
            q = qkv[:, :, 0].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1].transpose(0, 2, 1, 3)
            v = qkv[:, :, 2].transpose(0, 2, 1, 3)
            att = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=full_mask_bf16)
            h = h + self.out_proj(att.transpose(0, 2, 1, 3).reshape(B, S, H))
            hn2 = mx.fast.layer_norm(h, self.n2_w, self.n2_b, eps=1e-5)
            return h + self.l2(nn.relu(self.l1(hn2)))

    class _LayaMlxModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_sz, H)
            self.tok_emb.weight = raw_w["encoder.embeddings.tok_embeddings.weight"].astype(mx.bfloat16)
            self.emb_norm_w = raw_w["encoder.embeddings.norm.weight"].astype(mx.bfloat16)
            self.final_norm_w = raw_w["encoder.final_norm.weight"].astype(mx.bfloat16)
            self.type_emb = raw_w["type_emb.weight"].astype(mx.bfloat16)
            self.layers = [_MbEncLayer(i) for i in range(n_layers)]
            self.head_layers = [_LayaHeadLayer(0), _LayaHeadLayer(1)]

            self.sc_n_w = raw_w["scorer.0.weight"].astype(mx.float32)
            self.sc_n_b = raw_w["scorer.0.bias"].astype(mx.float32)
            self.sc_1_w = raw_w["scorer.1.weight"].astype(mx.float32)
            self.sc_1_b = raw_w["scorer.1.bias"].astype(mx.float32)
            self.sc_3_w = raw_w["scorer.3.weight"].astype(mx.float32)
            self.sc_3_b = raw_w["scorer.3.bias"].astype(mx.float32)

        def encode_hidden(
            self,
            ids_2d: Any,
            pos_2d: Any,
            qtypes_2d: Any,
            full_mask_bf16: Any,
            swa_mask_bf16: Any,
        ) -> Any:
            x = mx.fast.layer_norm(self.tok_emb(ids_2d), self.emb_norm_w, None, eps=ln_eps)
            cos_f, sin_f = _rope_trig_2d(pos_2d, inv_full)
            cos_s, sin_s = _rope_trig_2d(pos_2d, inv_swa)
            for blk in self.layers:
                if blk.is_full:
                    x = blk(x, full_mask_bf16, cos_f, sin_f)
                else:
                    x = blk(x, swa_mask_bf16, cos_s, sin_s)
            h = mx.fast.layer_norm(x, self.final_norm_w, None, eps=ln_eps) + self.type_emb[qtypes_2d]
            for hblk in self.head_layers:
                h = hblk(h, full_mask_bf16)
            return h

        def _score_laya_markers_fp32(self, h: Any, lane_idx: Any, pos_idx: Any) -> Any:
            """Gather `[M_items, K_max, H]` option markers BEFORE the 2-layer scorer (`34x` fewer scorer FLOPs)."""
            h_m = h[lane_idx, pos_idx].astype(mx.float32)
            sn = mx.fast.layer_norm(h_m, self.sc_n_w, self.sc_n_b, eps=1e-5)
            s1 = nn.gelu(sn @ self.sc_1_w.T + self.sc_1_b)
            return (s1 @ self.sc_3_w.T + self.sc_3_b).squeeze(-1)

    mod = _LayaMlxModule()
    mod.eval()
    if quant_bits in (4, 8) and H % group_size == 0:
        nn.quantize(
            mod,
            group_size=group_size,
            bits=quant_bits,
            class_predicate=lambda _p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and int(m.weight.shape[-1]) % group_size == 0,
        )
    mx.eval(mod.parameters())
    return mod


class LayaMlxGpuBackend(LayaTokenizerHelper):
    """Native MLX `8-bit g64` `Laya-Typed-421M` GPU backend with `Packed-512` batching & pre-scorer marker gather."""

    def __init__(
        self,
        laya_src_dir: Path | str | None = None,
        *,
        batch_size: int = 16,
        pack_seq_len: int = 512,
        max_len: int = 512,
        quant_bits: int = 8,
        group_size: int = 64,
        local_files_only: bool = False,
    ) -> None:
        import os
        import mlx.core as mx
        import numpy as np

        self._mx = mx
        self._np = np
        self.batch_size = max(1, int(batch_size))
        self.pack_seq_len = max(64, int(pack_seq_len))
        self._prev_cache_limit: int | None = None
        self._limits_restored = False
        eff_offline = bool(local_files_only) or (
            os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        )

        def _resolve_default_laya_hf() -> Path:
            from fast_shared.hf import resolve_hf_model_dir

            if eff_offline:
                return resolve_hf_model_dir(None, DEFAULT_LAYA_REPO, local_files_only=True)
            try:
                return resolve_hf_model_dir(None, DEFAULT_LAYA_REPO, local_files_only=True)
            except Exception:
                return resolve_hf_model_dir(None, DEFAULT_LAYA_REPO, local_files_only=False)

        if laya_src_dir is not None:
            cand = Path(laya_src_dir).expanduser().resolve()
            if not cand.is_dir():
                raise FileNotFoundError(f"Laya source directory does not exist: {cand}")
            if (cand / "model.safetensors").is_file():
                snap = cand
            elif (cand / "laya_typed_emb_f16.npz").is_file():
                snap = _resolve_default_laya_hf()
            else:
                raise FileNotFoundError(
                    f"Missing model.safetensors (or ANE laya_typed_emb_f16.npz) in laya_src_dir={cand}"
                )
        else:
            snap = _resolve_default_laya_hf()

        enc_cfg = json.loads((snap / "encoder" / "config.json").read_text(encoding="utf-8"))
        rl_cfg = json.loads((snap / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.local_half_win = int(enc_cfg.get("local_attention", 128)) // 2
        super().__init__(
            snap / "tokenizer" / "tokenizer.json",
            rl_agent_config=rl_cfg,
            laya_dir=snap,
            max_len=max_len,
        )

        set_cache = getattr(mx, "set_cache_limit", None)
        if callable(set_cache):
            try:
                self._prev_cache_limit = int(set_cache(_LAYA_GPU_CACHE_LIMIT_BYTES))
            except Exception:
                self._prev_cache_limit = None

        try:
            raw_w = mx.load(str(snap / "model.safetensors"))
            self.model = _build_laya_mlx_model(enc_cfg, raw_w, quant_bits=quant_bits, group_size=group_size)
            del raw_w
            mx.clear_cache()
        except Exception:
            self._restore_limits()
            raise

    def _restore_limits(self) -> None:
        if getattr(self, "_limits_restored", True):
            return
        self._limits_restored = True
        if self._prev_cache_limit is not None:
            set_cache = getattr(self._mx, "set_cache_limit", None)
            if callable(set_cache):
                try:
                    set_cache(self._prev_cache_limit)
                except Exception:
                    pass
            self._prev_cache_limit = None

    def __enter__(self) -> "LayaMlxGpuBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _score_generic(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        qtype_override: int | None = None,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        if not encoded_items:
            return []
        mx = self._mx
        np = self._np

        n_items = len(encoded_items)
        out_probs: list[list[float]] = [[] for _ in range(n_items)]
        lanes = _pack_laya_segments(encoded_items, range(n_items), lane_cap=self.pack_seq_len)
        lanes.sort(key=lambda ln: sum(len(encoded_items[i]["ids"]) for i in ln))

        for w0 in range(0, len(lanes), self.batch_size):
            sub_lanes = lanes[w0 : w0 + self.batch_size]
            ids_mx, pos_mx, qt_mx, full_m, swa_m, m_meta = _build_laya_packed_batch_tensors(
                mx,
                np,
                encoded_items,
                sub_lanes,
                pad_id=self.pad_id,
                local_half_win=self.local_half_win,
                qtype_override=qtype_override,
            )
            h = self.model.encode_hidden(ids_mx, pos_mx, qt_mx, full_m, swa_m)

            m_count = len(m_meta)
            k_max = max((len(x[2]) for x in m_meta), default=0)
            if k_max == 0:
                del h
                continue

            lane_np = np.zeros((m_count, 1), dtype=np.int32)
            pos_g_np = np.zeros((m_count, k_max), dtype=np.int32)
            for mi, (_orig_i, b_i, m_list, _qt) in enumerate(m_meta):
                lane_np[mi, 0] = b_i
                if m_list:
                    pos_g_np[mi, : len(m_list)] = m_list

            logits_mx = self.model._score_laya_markers_fp32(
                h,
                mx.array(lane_np, dtype=mx.int32),
                mx.array(pos_g_np, dtype=mx.int32),
            )
            mx.eval(logits_mx)
            logits_np = np.asarray(logits_mx)

            for mi, (orig_i, _b_i, m_list, eff_qt) in enumerate(m_meta):
                k_i = len(m_list)
                if k_i == 0:
                    out_probs[orig_i] = []
                    continue
                t_val = self.resolve_temperature(eff_qt, k_i, temperature_override)
                raw_z = [float(x) for x in logits_np[mi, :k_i].tolist()]
                out_probs[orig_i] = softmax_vec(raw_z, t_val, min_temp=0.05)

            del h, logits_mx
            get_cm = getattr(mx, "get_cache_memory", None)
            if callable(get_cm) and int(get_cm()) > _LAYA_CLEAR_CACHE_THRESHOLD_BYTES:
                mx.clear_cache()

        return out_probs

    def score_noul_batch(self, encoded_items: Sequence[Mapping[str, Any]]) -> list[float]:
        """Score binary (`noul`) items on GPU and return calibrated `P(true)`."""
        probs = self._score_generic(encoded_items, qtype_override=2)
        return [float(p[1]) if len(p) >= 2 else (float(p[0]) if p else 0.5) for p in probs]

    def score_choice_batch(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        """Score `Choice(K)` / `Score(K)` items on GPU and return calibrated option probabilities."""
        return self._score_generic(encoded_items, qtype_override=None, temperature_override=temperature_override)

    def score_options_batch(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        return self.score_choice_batch(encoded_items, temperature_override=temperature_override)

    def close(self) -> None:
        self.model = None
        self._restore_limits()
        self._mx.clear_cache()


class LayaHeteroEngine:
    """Heterogeneous `MLX W8 GPU + Multi-Bucket CoreML ANE` work-stealing engine for `Laya-Typed-421M`.

    Routing policy:
      - `mode='ane'` (or `mode='auto'` with `profile='interactive'`): `0` GPU ops, `0` GPU bytes;
        runs 100% on CoreML ANE (`pipe_depth=4`), ideal for non-blocking LLM sidecar triage and `B=1` queries.
      - `mode='gpu'`: runs 100% on `LayaMlxGpuBackend` (`Packed-512` + marker-gathered scorer).
      - `mode='hetero'` (or `mode='auto'` with `profile='backfill'`): double-ended work-stealing queue sorted
        descending by token length — GPU steals longest `>= 256` batches from the head while ANE `pipe_depth=4`
        steals shortest `<= 256` items from the tail concurrently.
    """

    def __init__(
        self,
        laya_dir: Path | str | None = None,
        dylib_path: Path | str | None = None,
        *,
        laya_src_dir: Path | str | None = None,
        mode: Literal["auto", "hetero", "ane", "gpu"] = "auto",
        profile: Literal["backfill", "interactive"] = "backfill",
        pipe_depth: int = 4,
        gpu_batch_size: int = 16,
        ane_batch_size: int = 8,
        auto_prebuilt: bool = True,
        local_files_only: bool = False,
        ane_backend: Any | None = None,
        gpu_backend: Any | None = None,
    ) -> None:
        norm_mode = str(mode).strip().lower()
        norm_prof = str(profile).strip().lower()
        if norm_mode not in ("auto", "hetero", "ane", "gpu"):
            raise ValueError(f"Unsupported mode={mode!r}; expected 'auto', 'hetero', 'ane', or 'gpu'")
        if norm_mode == "auto":
            eff_mode = "ane" if norm_prof == "interactive" else "hetero"
        else:
            eff_mode = norm_mode

        self.mode = eff_mode
        self.profile = norm_prof
        self.gpu_batch_size = max(1, int(gpu_batch_size))
        self.ane_batch_size = max(1, int(ane_batch_size))
        self._owns_ane = False
        self._owns_gpu = False

        if eff_mode in ("ane", "hetero"):
            if ane_backend is not None:
                self.ane_backend = ane_backend
            else:
                eff_laya = resolve_laya_dir(laya_dir) if laya_dir is not None else resolve_laya_dir()
                eff_dylib = Path(dylib_path).expanduser().resolve() if dylib_path is not None else ensure_laya_dylib()
                self.ane_backend = LayaAneBackend(
                    eff_laya,
                    eff_dylib,
                    pipe_depth=pipe_depth,
                    auto_prebuilt=auto_prebuilt,
                    local_files_only=local_files_only,
                )
                self._owns_ane = True
        else:
            self.ane_backend = None

        if eff_mode in ("gpu", "hetero"):
            if gpu_backend is not None:
                self.gpu_backend = gpu_backend
            else:
                gpu_src: Path | str | None
                if laya_src_dir is not None:
                    gpu_src = laya_src_dir
                elif laya_dir is not None:
                    cand_laya = Path(laya_dir).expanduser().resolve()
                    if (
                        not auto_prebuilt
                        or (cand_laya / "model.safetensors").is_file()
                        or (cand_laya / "laya_typed_emb_f16.npz").is_file()
                    ):
                        gpu_src = cand_laya
                    else:
                        gpu_src = None
                else:
                    gpu_src = None
                self.gpu_backend = LayaMlxGpuBackend(
                    gpu_src,
                    batch_size=self.gpu_batch_size,
                    local_files_only=local_files_only,
                )
                self._owns_gpu = True
        else:
            self.gpu_backend = None

        tok_src = self.ane_backend if self.ane_backend is not None else self.gpu_backend
        assert tok_src is not None
        self._tok_delegate = tok_src
        self.max_len = getattr(tok_src, "max_len", 512)
        self.head_max_len = getattr(tok_src, "head_max_len", 192)
        self.pad_id = getattr(tok_src, "pad_id", 50283)

    def __enter__(self) -> "LayaHeteroEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def encode_item(
        self,
        state: Any,
        qdef: Mapping[str, Any],
        *,
        max_len: int | None = None,
        on_truncate: Literal["warn", "error", "ignore"] = "warn",
    ) -> dict[str, Any]:
        return self._tok_delegate.encode_item(state, qdef, max_len=max_len, on_truncate=on_truncate)

    def _dispatch(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        is_noul: bool,
        temperature_override: float | None = None,
    ) -> list[Any]:
        n_total = len(encoded_items)
        if n_total == 0:
            return []

        if self.mode == "ane" or self.gpu_backend is None:
            assert self.ane_backend is not None
            items_l = list(encoded_items)
            if is_noul:
                return self.ane_backend.score_noul_batch(items_l)
            return self.ane_backend.score_choice_batch(items_l, temperature_override=temperature_override)

        if self.mode == "gpu" or self.ane_backend is None:
            if is_noul:
                return self.gpu_backend.score_noul_batch(encoded_items)
            return self.gpu_backend.score_choice_batch(encoded_items, temperature_override=temperature_override)

        # Tiny interactive batch shortcut (<= 2 short items -> ANE directly without thread spawn)
        if n_total <= 2 and all(len(it["ids"]) <= 256 for it in encoded_items):
            items_l = list(encoded_items)
            if is_noul:
                return self.ane_backend.score_noul_batch(items_l)
            return self.ane_backend.score_choice_batch(items_l, temperature_override=temperature_override)

        # Double-ended work-stealing queue sorted descending by sequence length:
        #   - HEAD (`popleft`): longest items (`>= 256` first) -> GPU NAX tensor-core waves
        #   - TAIL (`pop`): shortest items (`<= 256` first) -> ANE multi-bucket `pipe_depth=4`
        # Pre-assign the initial head (GPU) and tail (ANE) waves before thread start so both engines
        # launch immediately with zero OS thread-spawn race.
        desc_indices = sorted(range(n_total), key=lambda i: len(encoded_items[i]["ids"]), reverse=True)
        work_dq: deque[int] = deque(desc_indices)
        init_gpu: list[int] = []
        init_ane: list[int] = []
        if len(work_dq) >= 2 and len(encoded_items[work_dq[-1]]["ids"]) <= 256:
            ane_seed = min(self.ane_batch_size, max(1, len(work_dq) // 3))
            for _ in range(ane_seed):
                if work_dq and len(encoded_items[work_dq[-1]]["ids"]) <= 256:
                    init_ane.append(work_dq.pop())
                else:
                    break
        if work_dq:
            gpu_seed = min(self.gpu_batch_size, len(work_dq))
            init_gpu = [work_dq.popleft() for _ in range(gpu_seed)]

        qlock = threading.Lock()
        results: list[Any] = [0.5 if is_noul else [] for _ in range(n_total)]
        errors: list[BaseException] = []

        def _run_gpu_batch(batch_idx: list[int]) -> None:
            assert self.gpu_backend is not None
            sub_items = [dict(encoded_items[i]) for i in batch_idx]
            if is_noul:
                sub_out = self.gpu_backend.score_noul_batch(sub_items)
            else:
                sub_out = self.gpu_backend.score_choice_batch(
                    sub_items, temperature_override=temperature_override
                )
            for orig_i, val in zip(batch_idx, sub_out):
                results[orig_i] = val

        def _run_ane_batch(batch_idx: list[int]) -> None:
            assert self.ane_backend is not None
            sub_items = [dict(encoded_items[i]) for i in batch_idx]
            if is_noul:
                sub_out = self.ane_backend.score_noul_batch(sub_items)
            else:
                sub_out = self.ane_backend.score_choice_batch(
                    sub_items, temperature_override=temperature_override
                )
            for orig_i, val in zip(batch_idx, sub_out):
                results[orig_i] = val

        def _gpu_worker() -> None:
            try:
                if init_gpu:
                    _run_gpu_batch(init_gpu)
                while not errors:
                    with qlock:
                        if not work_dq:
                            break
                        rem = len(work_dq)
                        if rem <= self.ane_batch_size and len(encoded_items[work_dq[0]]["ids"]) <= 192:
                            n_take = max(1, rem // 2)
                        else:
                            n_take = min(self.gpu_batch_size, rem)
                        batch_idx = [work_dq.popleft() for _ in range(n_take)]
                    _run_gpu_batch(batch_idx)
            except BaseException as exc:
                errors.append(exc)

        def _ane_worker() -> None:
            try:
                if init_ane:
                    _run_ane_batch(init_ane)
                while not errors:
                    with qlock:
                        if not work_dq:
                            break
                        rem = len(work_dq)
                        tail_len = len(encoded_items[work_dq[-1]]["ids"])
                        if tail_len > 256 and rem <= self.gpu_batch_size:
                            n_take = max(1, min(4, rem // 3)) if rem >= 3 else 0
                        else:
                            n_take = min(self.ane_batch_size, rem)
                        if n_take <= 0:
                            break
                        batch_idx = [work_dq.pop() for _ in range(n_take)]
                    _run_ane_batch(batch_idx)
            except BaseException as exc:
                errors.append(exc)

        t_gpu = threading.Thread(target=_gpu_worker, name="laya-hetero-gpu")
        t_ane = threading.Thread(target=_ane_worker, name="laya-hetero-ane")
        t_gpu.start()
        t_ane.start()
        t_gpu.join()
        t_ane.join()
        if errors:
            raise errors[0]
        return results

    def score_noul_batch(self, encoded_items: Sequence[Mapping[str, Any]]) -> list[float]:
        return self._dispatch(encoded_items, is_noul=True)

    def score_choice_batch(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        return self._dispatch(encoded_items, is_noul=False, temperature_override=temperature_override)

    def score_options_batch(
        self,
        encoded_items: Sequence[Mapping[str, Any]],
        *,
        temperature_override: float | None = None,
    ) -> list[list[float]]:
        return self.score_choice_batch(encoded_items, temperature_override=temperature_override)

    def close(self) -> None:
        if self._owns_gpu and self.gpu_backend is not None:
            self.gpu_backend.close()
            self.gpu_backend = None
        if self._owns_ane and self.ane_backend is not None:
            self.ane_backend.close()
            self.ane_backend = None


__all__ = [
    "LayaHeteroEngine",
    "LayaMlxGpuBackend",
    "_build_laya_packed_batch_tensors",
    "_pack_laya_segments",
]
