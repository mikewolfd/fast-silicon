"""Shared Apple Silicon Qwen3.5 Vision Transformer, 3D MRoPE, and image preprocessing utilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import struct
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .vision_preprocess import (
    PplxImageItem,
    QwenVisionInputs,
    build_qwen_vision_inputs,
    crop_to_ink,
    detect_ink_bbox,
    load_image,
    prepare_pplx_image_item,
    rotate_image_cw,
    smart_resize_qwen,
)


@dataclass(frozen=True)
class Qwen35VisionConfig:
    """Configuration for the Qwen3.5 dynamic-resolution Vision Transformer."""

    depth: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 1024

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Qwen35VisionConfig:
        vcfg = dict(raw.get("vision_config") or raw)
        return cls(
            depth=int(vcfg.get("depth", 12)),
            hidden_size=int(vcfg.get("hidden_size", 768)),
            intermediate_size=int(vcfg.get("intermediate_size", 3072)),
            num_heads=int(vcfg.get("num_heads", 12)),
            in_channels=int(vcfg.get("in_channels", 3)),
            patch_size=int(vcfg.get("patch_size", 16)),
            temporal_patch_size=int(vcfg.get("temporal_patch_size", 2)),
            spatial_merge_size=int(vcfg.get("spatial_merge_size", 2)),
            num_position_embeddings=int(vcfg.get("num_position_embeddings", 2304)),
            out_hidden_size=int(vcfg.get("out_hidden_size", 1024)),
        )


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@mx.compile
def _apply_vision_rope_qk(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compiled FP32 vision 2D rotary embedding for `(1, seq_len, num_heads, head_dim)` Q/K."""
    orig_dtype = q.dtype
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    cf = cos.astype(mx.float32)
    sf = sin.astype(mx.float32)
    q_out = (qf * cf) + (_rotate_half(qf) * sf)
    k_out = (kf * cf) + (_rotate_half(kf) * sf)
    return q_out.astype(orig_dtype), k_out.astype(orig_dtype)


class VisionPatchEmbed(nn.Module):
    """3D spatio-temporal Conv3d patch embedding."""

    def __init__(self, cfg: Qwen35VisionConfig) -> None:
        super().__init__()
        self.patch_size = int(cfg.patch_size)
        self.temporal_patch_size = int(cfg.temporal_patch_size)
        self.in_channels = int(cfg.in_channels)
        self.hidden_size = int(cfg.hidden_size)
        k = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=k,
            stride=k,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        x = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        return self.proj(x).reshape(-1, self.hidden_size)


class VisionPatchMerger(nn.Module):
    """`2x2` spatial token merger (`4 * hidden_size -> out_hidden_size`)."""

    def __init__(self, cfg: Qwen35VisionConfig) -> None:
        super().__init__()
        merged_dim = int(cfg.hidden_size) * (int(cfg.spatial_merge_size) ** 2)
        self.merged_dim = merged_dim
        self.norm = nn.LayerNorm(int(cfg.hidden_size), eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_dim, int(cfg.out_hidden_size), bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x).reshape(-1, self.merged_dim)
        return self.linear_fc2(self.act_fn(self.linear_fc1(h)))


class VisionAttention(nn.Module):
    """Per-image ViT attention with uniform-batch single-launch SDPA and variable-length fallback."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: list[int],
        rope_cos: mx.array,
        rope_sin: mx.array,
        *,
        uniform_frame_len: int = 0,
    ) -> mx.array:
        seq_len = int(x.shape[0])
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim).transpose(1, 0, 2, 3)
        q, k = _apply_vision_rope_qk(qkv[0][None, ...], qkv[1][None, ...], rope_cos, rope_sin)
        v = qkv[2]

        n_frames = len(cu_seqlens) - 1
        if n_frames > 1 and uniform_frame_len > 0 and n_frames * uniform_frame_len == seq_len:
            qb = q[0].reshape(n_frames, uniform_frame_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            kb = k[0].reshape(n_frames, uniform_frame_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            vb = v.reshape(n_frames, uniform_frame_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            attn_b = mx.fast.scaled_dot_product_attention(qb, kb, vb, scale=self.scale)
            out = attn_b.transpose(0, 2, 1, 3).reshape(seq_len, -1)
            return self.proj(out)

        qh = q[0].transpose(1, 0, 2)[None, ...]  # (1, H, L, hd)
        kh = k[0].transpose(1, 0, 2)[None, ...]
        vh = v.transpose(1, 0, 2)[None, ...]
        if n_frames <= 1:
            attn_out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=self.scale)
        else:
            split_pts = cu_seqlens[1:-1]
            qs = mx.split(qh, split_pts, axis=2)
            ks = mx.split(kh, split_pts, axis=2)
            vs = mx.split(vh, split_pts, axis=2)
            attn_out = mx.concatenate(
                [mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.scale) for qi, ki, vi in zip(qs, ks, vs)],
                axis=2,
            )

        out = attn_out[0].transpose(1, 0, 2).reshape(seq_len, -1)
        return self.proj(out)


class VisionMLP(nn.Module):
    """ViT MLP with `gelu_pytorch_tanh` activation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act_fn = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    """Single pre-LayerNorm Qwen3.5 ViT block."""

    def __init__(self, cfg: Qwen35VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg.hidden_size, cfg.num_heads)
        self.mlp = VisionMLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: list[int],
        rope_cos: mx.array,
        rope_sin: mx.array,
        *,
        uniform_frame_len: int = 0,
    ) -> mx.array:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            uniform_frame_len=uniform_frame_len,
        )
        return x + self.mlp(self.norm2(x))


def _normalize_vision_key(raw_key: str) -> str | None:
    for pfx in ("vision_tower.", "model.visual.", "visual."):
        if raw_key.startswith(pfx):
            return raw_key[len(pfx) :]
    return None


def _inspect_hdr_keys(shard_path: str | Path) -> set[str]:
    with open(shard_path, "rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            return set()
        hdr_len = struct.unpack("<Q", raw_len)[0]
        if hdr_len <= 0 or hdr_len > (64 << 20):
            return set()
        hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
    return {str(k) for k in hdr.keys() if k != "__metadata__"} if isinstance(hdr, dict) else set()


@dataclass(frozen=True)
class _GridGeometryCacheEntry:
    idx_mx: mx.array
    wgt_mx: mx.array
    split_offsets: tuple[int, ...]
    rope_cos: mx.array
    rope_sin: mx.array
    cu_seqlens: list[int]
    uniform_frame_len: int


class FastQwen35VisionModel(nn.Module):
    """Pure-MLX Qwen3.5 dynamic-resolution Vision Tower matching HF/MLX `OvisOCR2`, `NuExtract3`, & `pplx-embed-v2`."""

    def __init__(self, cfg: Qwen35VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = int(cfg.spatial_merge_size)
        self.num_grid_per_side = int(round(int(cfg.num_position_embeddings) ** 0.5))
        head_dim = int(cfg.hidden_size) // int(cfg.num_heads)
        self._rot_dim = head_dim // 2
        self._inv_freq_np = (
            1.0 / (10000.0 ** (np.arange(0, self._rot_dim, 2, dtype=np.float32) / float(self._rot_dim)))
        ).astype(np.float32)

        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(int(cfg.num_position_embeddings), int(cfg.hidden_size))
        self.blocks = [VisionBlock(cfg) for _ in range(int(cfg.depth))]
        self.merger = VisionPatchMerger(cfg)
        self._grid_cache: OrderedDict[tuple[tuple[tuple[int, int, int], ...], str], _GridGeometryCacheEntry] = OrderedDict()

    def _get_grid_geometry(
        self,
        grid_thw: Sequence[tuple[int, int, int]],
        dtype: mx.Dtype,
    ) -> _GridGeometryCacheEntry:
        norm_grid = tuple((int(t), int(h), int(w)) for t, h, w in grid_thw)
        cache_key = (norm_grid, str(dtype))
        cached = self._grid_cache.get(cache_key)
        if cached is not None:
            self._grid_cache.move_to_end(cache_key)
            return cached

        g = self.num_grid_per_side
        ms = self.spatial_merge_size

        # 1. Bilinear pos_embed indices + weights computed purely in NumPy (zero GPU->host `.tolist()` syncs)
        idx_parts: list[list[np.ndarray]] = [[], [], [], []]
        wgt_parts: list[list[np.ndarray]] = [[], [], [], []]
        split_offsets: list[int] = []
        pe_acc = 0

        for idx_i, (_t, h_i, w_i) in enumerate(norm_grid):
            if idx_i > 0:
                split_offsets.append(pe_acc)
            pe_acc += h_i * w_i
            h_idxs = np.linspace(0.0, g - 1.0, h_i, dtype=np.float32)
            w_idxs = np.linspace(0.0, g - 1.0, w_i, dtype=np.float32)
            hf = np.floor(h_idxs).astype(np.int32)
            wf = np.floor(w_idxs).astype(np.int32)
            hc = np.minimum(hf + 1, g - 1)
            wc = np.minimum(wf + 1, g - 1)
            dh = h_idxs - hf.astype(np.float32)
            dw = w_idxs - wf.astype(np.float32)
            bh = hf * g
            bhc = hc * g

            idx_parts[0].append((bh[:, None] + wf[None, :]).reshape(-1))
            idx_parts[1].append((bh[:, None] + wc[None, :]).reshape(-1))
            idx_parts[2].append((bhc[:, None] + wf[None, :]).reshape(-1))
            idx_parts[3].append((bhc[:, None] + wc[None, :]).reshape(-1))

            wgt_parts[0].append(((1.0 - dh)[:, None] * (1.0 - dw)[None, :]).reshape(-1))
            wgt_parts[1].append(((1.0 - dh)[:, None] * dw[None, :]).reshape(-1))
            wgt_parts[2].append((dh[:, None] * (1.0 - dw)[None, :]).reshape(-1))
            wgt_parts[3].append((dh[:, None] * dw[None, :]).reshape(-1))

        idx_np = np.stack([np.concatenate(p, axis=0) for p in idx_parts], axis=0).astype(np.int32)
        wgt_np = np.stack([np.concatenate(p, axis=0) for p in wgt_parts], axis=0).astype(np.float32)
        idx_mx = mx.array(idx_np, dtype=mx.int32)
        wgt_mx = mx.array(wgt_np, dtype=dtype)

        # 2. 2D ViT rotary frequencies & cu_seqlens precomputed in NumPy -> single host->GPU upload
        max_hw = max(max(h_i, w_i) for _, h_i, w_i in norm_grid)
        freq_table = np.outer(np.arange(max_hw, dtype=np.float32), self._inv_freq_np)
        coords_list: list[np.ndarray] = []
        cu_seqlens = [0]
        cu_acc = 0
        frame_lens: list[int] = []
        for t_i, h_i, w_i in norm_grid:
            mh, mw = h_i // ms, w_i // ms
            br = np.arange(mh, dtype=np.int32)[:, None, None, None] * ms + np.arange(ms, dtype=np.int32)[None, None, :, None]
            bc = np.arange(mw, dtype=np.int32)[None, :, None, None] * ms + np.arange(ms, dtype=np.int32)[None, None, None, :]
            row_idx = np.broadcast_to(br, (mh, mw, ms, ms)).reshape(-1)
            col_idx = np.broadcast_to(bc, (mh, mw, ms, ms)).reshape(-1)
            coords = np.stack([row_idx, col_idx], axis=-1)
            if t_i > 1:
                coords = np.tile(coords, (t_i, 1))
            coords_list.append(coords)
            flen = h_i * w_i
            for _ in range(t_i):
                cu_acc += flen
                cu_seqlens.append(cu_acc)
                frame_lens.append(flen)

        pos_ids = np.concatenate(coords_list, axis=0)
        freqs = np.concatenate([freq_table[pos_ids[:, 0]], freq_table[pos_ids[:, 1]]], axis=-1)
        freqs2 = np.concatenate([freqs, freqs], axis=-1)
        rope_cos = mx.array(np.cos(freqs2)[None, :, None, :], dtype=mx.float32)
        rope_sin = mx.array(np.sin(freqs2)[None, :, None, :], dtype=mx.float32)
        uniform_frame_len = (
            frame_lens[0]
            if (len(frame_lens) > 1 and all(fl == frame_lens[0] for fl in frame_lens))
            else 0
        )

        entry = _GridGeometryCacheEntry(
            idx_mx=idx_mx,
            wgt_mx=wgt_mx,
            split_offsets=tuple(split_offsets),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            cu_seqlens=cu_seqlens,
            uniform_frame_len=uniform_frame_len,
        )
        self._grid_cache[cache_key] = entry
        if len(self._grid_cache) > 16:
            self._grid_cache.popitem(last=False)
        return entry

    def _rot_cos_sin(self, grid_thw: Sequence[tuple[int, int, int]]) -> tuple[mx.array, mx.array]:
        geom = self._get_grid_geometry(grid_thw, self.patch_embed.proj.weight.dtype)
        return geom.rope_cos, geom.rope_sin

    def _pos_embed_interpolate(
        self,
        grid_thw: Sequence[tuple[int, int, int]],
        geom: _GridGeometryCacheEntry | None = None,
    ) -> mx.array:
        g_entry = geom or self._get_grid_geometry(grid_thw, self.pos_embed.weight.dtype)
        ms = self.spatial_merge_size
        pe = self.pos_embed(g_entry.idx_mx) * g_entry.wgt_mx[:, :, None]
        patch_pe = pe[0] + pe[1] + pe[2] + pe[3]
        parts = mx.split(patch_pe, list(g_entry.split_offsets), axis=0) if g_entry.split_offsets else [patch_pe]

        permuted: list[mx.array] = []
        for part, (t, h, w) in zip(parts, grid_thw):
            t_i, h_i, w_i = int(t), int(h), int(w)
            dim = int(part.shape[-1])
            tiled = mx.tile(part, (t_i, 1)).reshape(t_i, h_i // ms, ms, w_i // ms, ms, dim)
            permuted.append(tiled.transpose(0, 1, 3, 2, 4, 5).reshape(-1, dim))
        return mx.concatenate(permuted, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: Sequence[tuple[int, int, int]]) -> mx.array:
        dtype = self.patch_embed.proj.weight.dtype
        geom = self._get_grid_geometry(grid_thw, dtype)
        x = self.patch_embed(pixel_values.astype(dtype))
        x = x + self._pos_embed_interpolate(grid_thw, geom=geom)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=geom.cu_seqlens,
                rope_cos=geom.rope_cos,
                rope_sin=geom.rope_sin,
                uniform_frame_len=geom.uniform_frame_len,
            )
        return self.merger(x)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype | None = None,
        quantize_bits: int | None = None,
        quantize_group_size: int = 64,
        config_override: Qwen35VisionConfig | None = None,
    ) -> FastQwen35VisionModel:
        """Stream-load vision weights from HF/MLX `.safetensors` shards, optionally casting and quantizing."""
        mdir = Path(model_dir).expanduser().resolve()
        raw_cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
        cfg = config_override or Qwen35VisionConfig.from_dict(raw_cfg)
        vis = cls(cfg)

        shards = sorted(glob.glob(str(mdir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors shards found in {mdir}")

        vis_weights: list[tuple[str, mx.array]] = []
        for sp in shards:
            hdr_keys = _inspect_hdr_keys(sp)
            if not any(_normalize_vision_key(k) is not None for k in hdr_keys):
                continue
            loaded: dict[str, mx.array] = mx.load(sp)  # type: ignore[assignment]
            for rk, tensor in loaded.items():
                nk = _normalize_vision_key(rk)
                if nk is None or "position_ids" in nk:
                    continue
                if nk == "patch_embed.proj.weight" and tensor.ndim == 5:
                    # Convert PyTorch [out, in_c, t, h, w] -> MLX [out, t, h, w, in_c]
                    if int(tensor.shape[1]) == cfg.in_channels and int(tensor.shape[4]) != cfg.in_channels:
                        tensor = tensor.transpose(0, 2, 3, 4, 1)
                if dtype is not None and tensor.dtype in (mx.float32, mx.float16, mx.bfloat16):
                    tensor = tensor.astype(dtype)
                vis_weights.append((nk, tensor))
            del loaded
            mx.clear_cache()

        if not vis_weights:
            raise FileNotFoundError(f"No `visual.*` / `vision_tower.*` weights found in {mdir}")

        vis.load_weights(vis_weights, strict=False)
        if dtype is not None:
            vis.set_dtype(dtype)
        if quantize_bits is not None:
            q_bits = int(quantize_bits)
            q_gs = int(quantize_group_size)
            nn.quantize(
                vis,
                group_size=q_gs,
                bits=q_bits,
                class_predicate=lambda _, mod: isinstance(mod, nn.Linear),
            )
        mx.eval(vis.parameters())
        return vis


def compute_mrope_position_selector(
    mrope_section: Sequence[int] = (11, 11, 10),
    freq_dim: int = 32,
) -> mx.array:
    """Return length-`freq_dim` axis selector `[0|1|2]` matching interleaved `t, h, w` MRoPE."""
    if len(mrope_section) != 3:
        raise ValueError(f"Expected 3 MRoPE sections, got {len(mrope_section)}")
    selector = [0] * int(freq_dim)
    for dim, offset in ((1, 1), (2, 2)):
        limit = min(int(mrope_section[dim]) * 3, int(freq_dim))
        for idx in range(offset, limit, 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


def compute_mrope_positions(
    prompt_ids: Sequence[int],
    *,
    image_grid_thw: Sequence[tuple[int, int, int]],
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> tuple[mx.array, int]:
    """Compute `(3, 1, L)` interleaved MRoPE position IDs and integer `rope_offset_delta`.

    Matches `Qwen3_5Model.get_rope_index`:
      - Every contiguous text run gets 1D ascending positions `(pos, pos, pos)`.
      - Every image run of `t * (h // merge) * (w // merge)` tokens gets 3D coordinates `(t, h, w) + st_idx`.
      - Returns `rope_offset_delta = max_pos_id + 1 - len(prompt_ids)` so cached decode steps at
        `cache.offset = len(prompt_ids) + k` evaluate RoPE at `max_pos_id + 1 + k`.
    """
    seq = [int(t) for t in prompt_ids]
    seq_len = len(seq)
    if not image_grid_thw:
        pos_1d = mx.arange(seq_len, dtype=mx.int32)
        pos_3d = mx.broadcast_to(pos_1d[None, None, :], (3, 1, seq_len))
        return pos_3d, 0

    merge = int(spatial_merge_size)
    pos_parts: list[np.ndarray] = []
    st = 0
    img_idx = 0
    max_pos = -1

    while img_idx < len(image_grid_thw):
        try:
            ed = seq.index(int(image_token_id), st)
        except ValueError:
            break
        t_raw, h_raw, w_raw = image_grid_thw[img_idx]
        img_idx += 1
        gt = int(t_raw)
        gh = int(h_raw) // merge
        gw = int(w_raw) // merge
        n_vis = gt * gh * gw

        st_idx = max_pos + 1
        text_len = ed - st
        if text_len > 0:
            t_run = np.arange(text_len, dtype=np.int32) + st_idx
            pos_parts.append(np.broadcast_to(t_run[None, :], (3, text_len)))
            st_idx += text_len

        t_grid = np.repeat(np.arange(gt, dtype=np.int32), gh * gw)
        h_grid = np.tile(np.repeat(np.arange(gh, dtype=np.int32), gw), gt)
        w_grid = np.tile(np.arange(gw, dtype=np.int32), gt * gh)
        vis_pos = np.stack([t_grid, h_grid, w_grid], axis=0) + st_idx
        pos_parts.append(vis_pos)
        max_pos = int(vis_pos.max()) if n_vis > 0 else (st_idx - 1)
        st = ed + n_vis

    if st < seq_len:
        st_idx = max_pos + 1
        rem = seq_len - st
        t_run = np.arange(rem, dtype=np.int32) + st_idx
        pos_parts.append(np.broadcast_to(t_run[None, :], (3, rem)))
        max_pos = int(t_run[-1])

    compact = np.concatenate(pos_parts, axis=1) if pos_parts else np.zeros((3, 0), dtype=np.int32)
    if compact.shape[1] != seq_len:
        raise ValueError(
            f"Prompt length {seq_len} did not match MRoPE expanded position length {compact.shape[1]}"
        )
    rope_delta = (max_pos + 1 - seq_len) if seq_len > 0 else 0
    return mx.array(compact[:, None, :], dtype=mx.int32), int(rope_delta)


def precompute_mrope_trig(
    position_ids_3d: mx.array,
    *,
    rot_dims: int = 64,
    rope_theta: float = 10000000.0,
    mrope_section: Sequence[int] = (11, 11, 10),
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[mx.array, mx.array]:
    """Compute `(cos, sin)` of shape `(B, 1, L, rot_dims // 2)` supporting int32 and fractional float32 MRoPE."""
    freq_dim = int(rot_dims) // 2
    inv_freq = 1.0 / (
        float(rope_theta) ** (mx.arange(0, int(rot_dims), 2, dtype=mx.float32) / float(rot_dims))
    )
    selector = compute_mrope_position_selector(mrope_section=mrope_section, freq_dim=freq_dim)
    # position_ids_3d: (3, B, L) -> take(axis=0) -> (freq_dim, B, L) -> (B, L, freq_dim)
    selected_pos = mx.take(position_ids_3d, selector, axis=0).transpose(1, 2, 0).astype(mx.float32)
    angles = selected_pos * inv_freq[None, None, :]
    cos_a = mx.cos(angles)[:, None, :, :].astype(dtype)
    sin_a = mx.sin(angles)[:, None, :, :].astype(dtype)
    return cos_a, sin_a


__all__ = [
    "FastQwen35VisionModel",
    "PplxImageItem",
    "Qwen35VisionConfig",
    "QwenVisionInputs",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
    "VisionPatchEmbed",
    "VisionPatchMerger",
    "_apply_vision_rope_qk",
    "_normalize_vision_key",
    "_rotate_half",
    "build_qwen_vision_inputs",
    "compute_mrope_position_selector",
    "compute_mrope_positions",
    "crop_to_ink",
    "detect_ink_bbox",
    "load_image",
    "precompute_mrope_trig",
    "prepare_pplx_image_item",
    "rotate_image_cw",
    "smart_resize_qwen",
]
