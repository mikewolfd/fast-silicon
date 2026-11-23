"""Lazy pure-MLX Gemma-4 & EmbeddingGemma-2 shared vision encoder (`vision_tower` + `embed_vision`).

Zero top-level `PIL` import and zero text-only weight loading: callers import this module only when
images are passed, keeping text-only startup RAM and latency at zero overhead.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class Gemma4VisionConfig:
    """Configuration for Gemma-4 / EmbeddingGemma-2 SigLIP-style 2D-RoPE ViT + spatial pooler."""

    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3072
    patch_size: int = 16
    pooling_kernel_size: int = 3
    default_output_length: int = 280
    position_embedding_size: int = 10240
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    use_clipped_linears: bool = False
    standardize: bool = False
    text_hidden_size: int = 1536
    boi_token_id: int = 255999
    image_token_id: int = 258880
    eoi_token_id: int = 258882

    @classmethod
    def from_dict(cls, raw_cfg: Mapping[str, Any]) -> Gemma4VisionConfig:
        vc_raw = raw_cfg.get("vision_config")
        if vc_raw is None:
            raise ValueError("Missing `vision_config` in checkpoint config.json")
        vc = dict(vc_raw)
        tc = dict(raw_cfg.get("text_config") or {})
        rope_params = dict(vc.get("rope_parameters") or {})
        return cls(
            hidden_size=int(vc.get("hidden_size", 768)),
            num_hidden_layers=int(vc.get("num_hidden_layers", 16)),
            num_attention_heads=int(vc.get("num_attention_heads", 12)),
            num_key_value_heads=int(vc.get("num_key_value_heads", vc.get("num_attention_heads", 12))),
            head_dim=int(vc.get("head_dim", 64)),
            intermediate_size=int(vc.get("intermediate_size", 3072)),
            patch_size=int(vc.get("patch_size", 16)),
            pooling_kernel_size=int(vc.get("pooling_kernel_size", 3)),
            default_output_length=int(vc.get("default_output_length", raw_cfg.get("vision_soft_tokens_per_image", 280))),
            position_embedding_size=int(vc.get("position_embedding_size", 10240)),
            rms_norm_eps=float(vc.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_params.get("rope_theta", vc.get("rope_theta", 100.0))),
            use_clipped_linears=bool(vc.get("use_clipped_linears", False)),
            standardize=bool(vc.get("standardize", False)),
            text_hidden_size=int(tc.get("hidden_size", raw_cfg.get("text_hidden_size", 1536))),
            boi_token_id=int(raw_cfg.get("boi_token_id", 255999)),
            image_token_id=int(raw_cfg.get("image_token_id", raw_cfg.get("image_token_index", 258880))),
            eoi_token_id=int(raw_cfg.get("eoi_token_id", 258882)),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> Gemma4VisionConfig:
        mdir = Path(model_dir).expanduser().resolve()
        cfg_path = mdir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {mdir}")
        raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not raw_cfg.get("vision_config"):
            raise ValueError(f"Checkpoint {mdir} is missing `vision_config` in config.json")
        return cls.from_dict(raw_cfg)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    max_patches: int = 2520,
    pooling_kernel_size: int = 3,
) -> tuple[int, int]:
    """Compute Gemma-4 aspect-preserving `(height, width)` multiples of `pooling_kernel_size * patch_size`."""
    h = int(height)
    w = int(width)
    p = int(patch_size)
    k = int(pooling_kernel_size)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {(h, w)}")
    if int(max_patches) < k * k:
        raise ValueError(f"max_patches={max_patches} must be >= pooling_kernel_size^2 ({k * k})")

    target_px = int(max_patches) * (p * p)
    factor = math.sqrt(target_px / float(h * w))
    side_mult = k * p

    target_height = int(math.floor(factor * h / side_mult)) * side_mult
    target_width = int(math.floor(factor * w / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(f"Image dimensions {(h, w)} resize to 0x0 under max_patches={max_patches}")

    max_side_length = (int(max_patches) // (k * k)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(max(int(math.floor(w / h)), 1) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(max(int(math.floor(h / w)), 1) * side_mult, max_side_length)

    return int(target_height), int(target_width)


@dataclass(frozen=True)
class Gemma4ImageInputs:
    """Preprocessed Gemma-4 CHW image tensor and spatial patch grid metadata."""

    pixel_values: np.ndarray  # float32 (3, H, W) in [0.0, 1.0]
    grid_hw: tuple[int, int]  # (pH, pW)
    resized_height: int
    resized_width: int
    num_soft_tokens: int


def preprocess_gemma4_image(
    source: str | Path | Image.Image | np.ndarray,
    *,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
    max_soft_tokens: int = 280,
) -> Gemma4ImageInputs:
    """Resize RGB image to Gemma-4 `pooling_kernel_size * patch_size` grid and rescale to `[0, 1]` CHW float32."""
    from PIL import Image as _PILImage

    from .vision_preprocess import load_image

    img = load_image(source)
    orig_w, orig_h = img.size
    p = int(patch_size)
    k = int(pooling_kernel_size)
    max_patches = int(max_soft_tokens) * (k * k)
    target_h, target_w = get_aspect_ratio_preserving_size(
        orig_h,
        orig_w,
        patch_size=p,
        max_patches=max_patches,
        pooling_kernel_size=k,
    )
    if (orig_w, orig_h) != (target_w, target_h):
        img = img.resize((target_w, target_h), resample=_PILImage.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
    chw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)), dtype=np.float32)
    ph = target_h // p
    pw = target_w // p
    n_soft = (ph // k) * (pw // k)
    return Gemma4ImageInputs(
        pixel_values=chw,
        grid_hw=(ph, pw),
        resized_height=target_h,
        resized_width=target_w,
        num_soft_tokens=n_soft,
    )


def _rotate_half(x: mx.array) -> mx.array:
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_2d_rope_precomputed(
    x: mx.array,
    cos_x: mx.array,
    sin_x: mx.array,
    cos_y: mx.array,
    sin_y: mx.array,
) -> mx.array:
    head_dim = int(x.shape[-1])
    c_dim = 2 * (head_dim // 4)
    x0 = x[..., :c_dim]
    x1 = x[..., c_dim : 2 * c_dim]
    y0 = x0 * cos_x + _rotate_half(x0) * sin_x
    y1 = x1 * cos_y + _rotate_half(x1) * sin_y
    if 2 * c_dim < head_dim:
        return mx.concatenate([y0, y1, x[..., 2 * c_dim :]], axis=-1)
    return mx.concatenate([y0, y1], axis=-1)


def _pad_head_dim_for_sdpa(t: mx.array) -> int:
    head_dim = int(t.shape[-1])
    if head_dim in (64, 80, 128):
        return 0
    for target in (64, 80, 128):
        if target > head_dim:
            return target - head_dim
    return ((head_dim + 31) // 32) * 32 - head_dim


def _rms_norm_no_scale(x: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, None, float(eps))


class ClippableLinear(nn.Module):
    """Linear projection with optional static input/output clamping bounds."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_clipping: bool = False,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.linear: nn.Module = nn.Linear(in_features, out_features, bias=bias)
        self.use_clipping = bool(use_clipping)
        if self.use_clipping:
            self.input_min = mx.array(-float("inf"), dtype=dtype)
            self.input_max = mx.array(float("inf"), dtype=dtype)
            self.output_min = mx.array(-float("inf"), dtype=dtype)
            self.output_max = mx.array(float("inf"), dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        y = self.linear(x)
        if self.use_clipping:
            y = mx.clip(y, self.output_min, self.output_max)
        return y


class VisionAttention(nn.Module):
    """Gemma-4 ViT multi-head self-attention with Q/K/V RMSNorm and precomputed 2D RoPE."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.hidden_size = int(cfg.hidden_size)
        self.eps = float(cfg.rms_norm_eps)
        clip = bool(cfg.use_clipped_linears)

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = ClippableLinear(self.hidden_size, q_out, bias=False, use_clipping=clip, dtype=dtype)
        self.k_proj = ClippableLinear(self.hidden_size, kv_out, bias=False, use_clipping=clip, dtype=dtype)
        self.v_proj = ClippableLinear(self.hidden_size, kv_out, bias=False, use_clipping=clip, dtype=dtype)
        self.o_proj = ClippableLinear(q_out, self.hidden_size, bias=False, use_clipping=clip, dtype=dtype)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self._fused_qkv: nn.Linear | None = None
        self._q_dim = q_out
        self._kv_dim = kv_out

    def try_fuse_qkv(self) -> bool:
        if (
            self.q_proj is None
            or self.k_proj is None
            or self.v_proj is None
            or self.q_proj.use_clipping
            or self.k_proj.use_clipping
            or self.v_proj.use_clipping
        ):
            return False
        if not (
            isinstance(self.q_proj.linear, nn.Linear)
            and isinstance(self.k_proj.linear, nn.Linear)
            and isinstance(self.v_proj.linear, nn.Linear)
        ):
            return False
        w_q = self.q_proj.linear.weight
        w_k = self.k_proj.linear.weight
        w_v = self.v_proj.linear.weight
        w_cat = mx.contiguous(mx.concatenate([w_q, w_k, w_v], axis=0))
        fused = nn.Linear(self.hidden_size, self._q_dim + 2 * self._kv_dim, bias=False)
        fused.weight = w_cat
        self._fused_qkv = fused
        return True

    def __call__(
        self,
        x: mx.array,
        cos_x: mx.array,
        sin_x: mx.array,
        cos_y: mx.array,
        sin_y: mx.array,
    ) -> mx.array:
        b_sz, seq_len, _ = x.shape
        if self._fused_qkv is not None:
            qkv = self._fused_qkv(x)
            q_end = self._q_dim
            k_end = q_end + self._kv_dim
            q = qkv[..., :q_end]
            k = qkv[..., q_end:k_end]
            v = qkv[..., k_end:]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = self.q_norm(q.reshape(b_sz, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(b_sz, seq_len, self.num_kv_heads, self.head_dim))
        v = _rms_norm_no_scale(v.reshape(b_sz, seq_len, self.num_kv_heads, self.head_dim), self.eps)

        q = _apply_2d_rope_precomputed(q, cos_x, sin_x, cos_y, sin_y).transpose(0, 2, 1, 3)
        k = _apply_2d_rope_precomputed(k, cos_x, sin_x, cos_y, sin_y).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        pad = _pad_head_dim_for_sdpa(q)
        if pad > 0:
            q = mx.pad(q, [(0, 0), (0, 0), (0, 0), (0, pad)])
            k = mx.pad(k, [(0, 0), (0, 0), (0, 0), (0, pad)])
            v = mx.pad(v, [(0, 0), (0, 0), (0, 0), (0, pad)])
        att = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)
        if pad > 0:
            att = att[..., : self.head_dim]

        out = att.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, self._q_dim)
        return self.o_proj(out)


class VisionMLP(nn.Module):
    """Gemma-4 ViT GeGLU MLP with optional weight clamping and unfused/fused gate+up."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        clip = bool(cfg.use_clipped_linears)
        self.hidden_size = int(cfg.hidden_size)
        self.intermediate_size = int(cfg.intermediate_size)
        self.gate_proj = ClippableLinear(
            self.hidden_size, self.intermediate_size, bias=False, use_clipping=clip, dtype=dtype
        )
        self.up_proj = ClippableLinear(
            self.hidden_size, self.intermediate_size, bias=False, use_clipping=clip, dtype=dtype
        )
        self.down_proj = ClippableLinear(
            self.intermediate_size, self.hidden_size, bias=False, use_clipping=clip, dtype=dtype
        )
        self._fused_gate_up: nn.Linear | None = None

    def try_fuse_gate_up(self) -> bool:
        if self.gate_proj.use_clipping or self.up_proj.use_clipping:
            return False
        if not (isinstance(self.gate_proj.linear, nn.Linear) and isinstance(self.up_proj.linear, nn.Linear)):
            return False
        w_cat = mx.contiguous(mx.concatenate([self.gate_proj.linear.weight, self.up_proj.linear.weight], axis=0))
        fused = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        fused.weight = w_cat
        self._fused_gate_up = fused
        return True

    def __call__(self, x: mx.array) -> mx.array:
        if self._fused_gate_up is not None:
            gu = self._fused_gate_up(x)
            gate, up = mx.split(gu, 2, axis=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        return self.down_proj(nn.gelu_approx(gate) * up)


class VisionTransformerBlock(nn.Module):
    """Single Gemma-4 ViT transformer block with pre/post attention and FFN RMSNorms."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        eps = float(cfg.rms_norm_eps)
        self.self_attn = VisionAttention(cfg, dtype=dtype)
        self.mlp = VisionMLP(cfg, dtype=dtype)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        cos_x: mx.array,
        sin_x: mx.array,
        cos_y: mx.array,
        sin_y: mx.array,
    ) -> mx.array:
        attn_out = self.self_attn(self.input_layernorm(x), cos_x, sin_x, cos_y, sin_y)
        h = x + self.post_attention_layernorm(attn_out)
        ff_out = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(ff_out)


class VisionPatchEmbedder(nn.Module):
    """Patchify `[B, 3, H, W]` images and add direct 2D position embeddings (no one-hot matmul)."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.patch_size = int(cfg.patch_size)
        self.position_embedding_size = int(cfg.position_embedding_size)
        self.input_proj: nn.Module = nn.Linear(3 * self.patch_size * self.patch_size, self.hidden_size, bias=False)
        self.position_embedding_table = mx.zeros((2, self.position_embedding_size, self.hidden_size), dtype=dtype)

    def __call__(
        self,
        pixel_values_bchw: mx.array,
        xs: mx.array,
        ys: mx.array,
    ) -> mx.array:
        b_sz, c_sz, h_px, w_px = pixel_values_bchw.shape
        p = self.patch_size
        ph = h_px // p
        pw = w_px // p
        patches = pixel_values_bchw.reshape(b_sz, c_sz, ph, p, pw, p).transpose(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(b_sz, ph * pw, c_sz * p * p)
        proj_dtype = self.position_embedding_table.dtype
        patches = (mx.array(2.0, dtype=mx.float32) * (patches.astype(mx.float32) - mx.array(0.5, dtype=mx.float32)))
        hidden = self.input_proj(patches.astype(proj_dtype))
        pos_emb = self.position_embedding_table[0][xs] + self.position_embedding_table[1][ys]
        return hidden + pos_emb[None, :, :].astype(hidden.dtype)


class VisionTransformerEncoder(nn.Module):
    """Container holding `vision_tower.encoder.layers`."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.layers = [VisionTransformerBlock(cfg, dtype=dtype) for _ in range(int(cfg.num_hidden_layers))]

    def __call__(
        self,
        x: mx.array,
        cos_x: mx.array,
        sin_x: mx.array,
        cos_y: mx.array,
        sin_y: mx.array,
    ) -> mx.array:
        for layer in self.layers:
            x = layer(x, cos_x, sin_x, cos_y, sin_y)
        return x


class VisionPooler(nn.Module):
    """Spatial `k x k` average pooler over `(pH, pW)` patch grid with `sqrt(hidden_size)` scaling."""

    def __init__(self, cfg: Gemma4VisionConfig) -> None:
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.pooling_kernel_size = int(cfg.pooling_kernel_size)
        self.root_hidden_size = float(self.hidden_size**0.5)

    def __call__(self, hidden: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        b_sz = int(hidden.shape[0])
        ph, pw = int(grid_hw[0]), int(grid_hw[1])
        k = self.pooling_kernel_size
        if k > 1:
            if ph % k != 0 or pw % k != 0:
                raise ValueError(f"Grid {(ph, pw)} is not divisible by pooling_kernel_size={k}")
            pooled = (
                hidden.astype(mx.float32)
                .reshape(b_sz, ph // k, k, pw // k, k, self.hidden_size)
                .mean(axis=(2, 4))
                .reshape(b_sz, (ph // k) * (pw // k), self.hidden_size)
                .astype(hidden.dtype)
            )
        else:
            pooled = hidden
        return pooled * mx.array(self.root_hidden_size, dtype=pooled.dtype)


class Gemma4VisionTower(nn.Module):
    """Top-level `vision_tower` module matching checkpoint weight paths."""

    def __init__(self, cfg: Gemma4VisionConfig, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embedder = VisionPatchEmbedder(cfg, dtype=dtype)
        self.encoder = VisionTransformerEncoder(cfg, dtype=dtype)
        self.pooler = VisionPooler(cfg)
        if cfg.standardize:
            self.std_bias = mx.zeros((cfg.hidden_size,), dtype=dtype)
            self.std_scale = mx.ones((cfg.hidden_size,), dtype=dtype)

    def __call__(
        self,
        pixel_values_bchw: mx.array,
        grid_hw: tuple[int, int],
        xs: mx.array,
        ys: mx.array,
        cos_x: mx.array,
        sin_x: mx.array,
        cos_y: mx.array,
        sin_y: mx.array,
        encoder_fn: Any = None,
    ) -> mx.array:
        x = self.patch_embedder(pixel_values_bchw, xs, ys)
        enc = encoder_fn if encoder_fn is not None else self.encoder
        x = enc(x, cos_x, sin_x, cos_y, sin_y)
        pooled = self.pooler(x, grid_hw)
        if self.cfg.standardize:
            pooled = (pooled - self.std_bias) * self.std_scale
        return pooled


class MultimodalEmbedder(nn.Module):
    """Project `vision_tower` features via parameter-free RMSNorm + linear projection."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.embedding_projection: nn.Module = nn.Linear(embedding_dim, text_hidden_size, bias=False)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        normed = _rms_norm_no_scale(inputs_embeds, self.eps)
        return self.embedding_projection(normed)


@dataclass(frozen=True)
class _GridGeomEntry:
    xs: mx.array
    ys: mx.array
    cos_x: mx.array
    sin_x: mx.array
    cos_y: mx.array
    sin_y: mx.array


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


_LEAF_PROJ_NAMES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})


def _normalize_gemma4_vision_key(raw_key: str) -> str | None:
    key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
    if not key.startswith(("vision_tower.", "embed_vision.")):
        return None
    if key.startswith("vision_tower.encoder.layers."):
        parts = key.split(".")
        # Normalize `..._proj.{weight,scales,biases}` -> `..._proj.linear.{weight,scales,biases}` when unwrapped
        if len(parts) >= 2 and parts[-2] in _LEAF_PROJ_NAMES and parts[-1] in ("weight", "scales", "biases"):
            parts.insert(-1, "linear")
            key = ".".join(parts)
    return key


def _infer_quant_params(
    mod_key: str,
    w_arr: mx.array,
    scales_arr: mx.array,
    qcfg: Mapping[str, Any],
) -> tuple[int, int]:
    ov = qcfg.get(mod_key) or qcfg.get(f"model.{mod_key}")
    if isinstance(ov, Mapping) and "bits" in ov and "group_size" in ov:
        return int(ov["bits"]), int(ov["group_size"])
    default_bits = qcfg.get("bits")
    default_gs = qcfg.get("group_size")
    out_f, w_cols = int(w_arr.shape[0]), int(w_arr.shape[1])
    s_groups = int(scales_arr.shape[1])
    if s_groups > 0:
        for cand_bits in ([int(default_bits)] if default_bits is not None else []) + [4, 8, 2, 3, 6]:
            if (w_cols * 32) % cand_bits == 0:
                in_f = (w_cols * 32) // cand_bits
                if in_f % s_groups == 0:
                    gs = in_f // s_groups
                    if gs in (32, 64, 128):
                        return int(cand_bits), int(gs)
    if default_bits is not None and default_gs is not None:
        return int(default_bits), int(default_gs)
    raise ValueError(f"Could not infer quantization bits/group_size for {mod_key} ({w_arr.shape}, {scales_arr.shape})")


class FastGemma4VisionModel(nn.Module):
    """Fast pure-MLX Gemma-4 / EmbeddingGemma-2 vision encoder with header-only weight loading."""

    def __init__(self, cfg: Gemma4VisionConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.cfg = cfg
        self.act_dtype = dtype
        self.vision_tower = Gemma4VisionTower(cfg, dtype=dtype)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=cfg.hidden_size,
            text_hidden_size=cfg.text_hidden_size,
            eps=cfg.rms_norm_eps,
        )
        self._geom_cache: OrderedDict[tuple[int, int, str], _GridGeomEntry] = OrderedDict()
        self._compiled_encoder: Any = None
        self._finalized = False

    def finalize_weights(self) -> None:
        """Fuse QKV/GateUp when unclipped and unquantized, and compile the transformer trunk."""
        for layer in self.vision_tower.encoder.layers:
            layer.self_attn.try_fuse_qkv()
            layer.mlp.try_fuse_gate_up()
        encoder = self.vision_tower.encoder

        @mx.compile
        def _compiled_enc(
            x: mx.array,
            cos_x: mx.array,
            sin_x: mx.array,
            cos_y: mx.array,
            sin_y: mx.array,
        ) -> mx.array:
            return encoder(x, cos_x, sin_x, cos_y, sin_y)

        self._compiled_encoder = _compiled_enc
        self._finalized = True

    def _get_grid_geometry(self, grid_hw: tuple[int, int], dtype: mx.Dtype) -> _GridGeomEntry:
        ph, pw = int(grid_hw[0]), int(grid_hw[1])
        key = (ph, pw, str(dtype))
        cached = self._geom_cache.get(key)
        if cached is not None:
            self._geom_cache.move_to_end(key)
            return cached

        gx = np.arange(pw, dtype=np.int32)
        gy = np.arange(ph, dtype=np.int32)
        xs_np = np.tile(gx, ph)
        ys_np = np.repeat(gy, pw)

        c_dim = 2 * (int(self.cfg.head_dim) // 4)
        half = c_dim // 2
        exponents = (2.0 / float(c_dim)) * np.arange(half, dtype=np.float32)
        timescale = (float(self.cfg.rope_theta) ** exponents).astype(np.float32)

        ang_x = xs_np[:, None].astype(np.float32) / timescale[None, :]
        ang_y = ys_np[:, None].astype(np.float32) / timescale[None, :]
        cos_x_np = np.concatenate([np.cos(ang_x), np.cos(ang_x)], axis=-1)
        sin_x_np = np.concatenate([np.sin(ang_x), np.sin(ang_x)], axis=-1)
        cos_y_np = np.concatenate([np.cos(ang_y), np.cos(ang_y)], axis=-1)
        sin_y_np = np.concatenate([np.sin(ang_y), np.sin(ang_y)], axis=-1)

        entry = _GridGeomEntry(
            xs=mx.array(xs_np, dtype=mx.int32),
            ys=mx.array(ys_np, dtype=mx.int32),
            cos_x=mx.array(cos_x_np)[None, :, None, :].astype(dtype),
            sin_x=mx.array(sin_x_np)[None, :, None, :].astype(dtype),
            cos_y=mx.array(cos_y_np)[None, :, None, :].astype(dtype),
            sin_y=mx.array(sin_y_np)[None, :, None, :].astype(dtype),
        )
        mx.eval(entry.xs, entry.ys, entry.cos_x, entry.sin_x, entry.cos_y, entry.sin_y)
        self._geom_cache[key] = entry
        if len(self._geom_cache) > 32:
            self._geom_cache.popitem(last=False)
        return entry

    def encode_batch(self, pixel_values_bchw: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        """Encode uniform `[B, 3, H, W]` tensor on `grid_hw=(pH, pW)` into `[B, n_soft, text_hidden_size]`."""
        if not self._finalized:
            self.finalize_weights()
        dtype = self.vision_tower.patch_embedder.position_embedding_table.dtype
        geom = self._get_grid_geometry(grid_hw, dtype)
        pooled = self.vision_tower(
            pixel_values_bchw,
            grid_hw,
            geom.xs,
            geom.ys,
            geom.cos_x,
            geom.sin_x,
            geom.cos_y,
            geom.sin_y,
            encoder_fn=self._compiled_encoder,
        )
        return self.embed_vision(pooled)

    def __call__(self, pixel_values_bchw: mx.array) -> mx.array:
        p = int(self.cfg.patch_size)
        grid_hw = (int(pixel_values_bchw.shape[2]) // p, int(pixel_values_bchw.shape[3]) // p)
        return self.encode_batch(pixel_values_bchw, grid_hw)

    def encode_images(
        self,
        inputs: Sequence[Gemma4ImageInputs],
        *,
        batch_size: int = 8,
    ) -> list[mx.array]:
        """Batch same-resolution images cleanly up to `batch_size`, preserving original input order."""
        if not inputs:
            return []
        bs_cap = max(1, int(batch_size))
        groups: dict[tuple[int, int], list[tuple[int, Gemma4ImageInputs]]] = {}
        for idx, item in enumerate(inputs):
            groups.setdefault(item.grid_hw, []).append((idx, item))

        out: list[mx.array | None] = [None] * len(inputs)
        for grid_hw, group in groups.items():
            for i0 in range(0, len(group), bs_cap):
                chunk = group[i0 : i0 + bs_cap]
                px = mx.array(np.stack([item.pixel_values for _, item in chunk], axis=0))
                feats = self.encode_batch(px, grid_hw)
                mx.eval(feats)
                for row_i, (orig_idx, _) in enumerate(chunk):
                    out[orig_idx] = feats[row_i]
        return [feat for feat in out if feat is not None]

    def _replace_quantized_linears(self, weight_map: Mapping[str, mx.array], qcfg: Mapping[str, Any]) -> None:
        scale_keys = [k for k in weight_map if k.endswith(".scales")]
        for sk in scale_keys:
            mod_key = sk[: -len(".scales")]
            w_arr = weight_map.get(f"{mod_key}.weight")
            if w_arr is None or w_arr.dtype != mx.uint32:
                continue
            bits, gs = _infer_quant_params(mod_key, w_arr, weight_map[sk], qcfg)
            has_bias = f"{mod_key}.bias" in weight_map

            # Traverse module tree to swap Linear -> QuantizedLinear
            parts = mod_key.split(".")
            parent: Any = self
            for part in parts[:-1]:
                if isinstance(parent, list):
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            leaf_name = parts[-1]
            orig = getattr(parent, leaf_name)
            if not isinstance(orig, nn.Linear):
                continue
            out_f, in_f = int(orig.weight.shape[0]), int(orig.weight.shape[1])
            ql = nn.QuantizedLinear(in_f, out_f, bias=has_bias, group_size=gs, bits=bits)
            if f"{mod_key}.biases" not in weight_map:
                ql.biases = None
            setattr(parent, leaf_name, ql)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype | None = mx.bfloat16,
    ) -> FastGemma4VisionModel:
        """Load strictly `vision_tower.*` & `embed_vision.*` tensors using header-only shard filtering."""
        mdir = Path(model_dir).expanduser().resolve()
        raw_cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
        cfg = Gemma4VisionConfig.from_dict(raw_cfg)
        eff_dtype = dtype or mx.bfloat16
        vis = cls(cfg, dtype=eff_dtype)

        shards = sorted(glob.glob(str(mdir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors shards found in {mdir}")

        weight_map: dict[str, mx.array] = {}
        for sp in shards:
            hdr_keys = _inspect_hdr_keys(sp)
            if not any(_normalize_gemma4_vision_key(k) is not None for k in hdr_keys):
                continue
            loaded: dict[str, mx.array] = mx.load(sp)  # type: ignore[assignment]
            for rk, tensor in loaded.items():
                nk = _normalize_gemma4_vision_key(rk)
                if nk is None:
                    continue
                if (
                    eff_dtype is not None
                    and tensor.dtype in (mx.float32, mx.float16, mx.bfloat16)
                    and not nk.endswith((".scales", ".biases"))
                ):
                    tensor = tensor.astype(eff_dtype)
                weight_map[nk] = tensor
            del loaded
            mx.clear_cache()

        has_vt = any(k.startswith("vision_tower.") for k in weight_map)
        if not has_vt:
            raise ValueError(f"No `vision_tower.*` weights found in {mdir} (stripped text-only checkpoint?)")

        qcfg = dict(raw_cfg.get("quantization") or raw_cfg.get("quantization_config") or {})
        vis._replace_quantized_linears(weight_map, qcfg)
        vis.load_weights(list(weight_map.items()), strict=False)
        vis.finalize_weights()
        mx.eval(vis.parameters())
        return vis


__all__ = [
    "ClippableLinear",
    "FastGemma4VisionModel",
    "Gemma4ImageInputs",
    "Gemma4VisionConfig",
    "Gemma4VisionTower",
    "MultimodalEmbedder",
    "VisionAttention",
    "VisionMLP",
    "VisionPatchEmbedder",
    "VisionPooler",
    "VisionTransformerBlock",
    "VisionTransformerEncoder",
    "get_aspect_ratio_preserving_size",
    "preprocess_gemma4_image",
]
