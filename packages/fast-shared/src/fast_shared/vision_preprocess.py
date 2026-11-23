"""Zero-MLX Qwen3.5 vision & `pplx-embed-v2` image preprocessing (pure NumPy + lazy `PIL`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class QwenVisionInputs:
    """Preprocessed pixel patch matrix and `grid_thw` matching `Qwen3VLProcessor`."""

    pixel_values: np.ndarray  # float32 (N_patches, C * tps * ps * ps)
    grid_thw: tuple[int, int, int]  # (t, grid_h, grid_w)
    resized_height: int
    resized_width: int
    num_visual_tokens: int


def load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load an image source into an EXIF-transposed RGB `PIL.Image` (lazy-imports `PIL`)."""
    from PIL import Image as _PILImage
    from PIL import ImageOps as _PILImageOps

    if isinstance(source, _PILImage.Image):
        img = _PILImageOps.exif_transpose(source)
        return img.convert("RGB") if img.mode != "RGB" else img
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return _PILImage.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), mode="RGB")
    with _PILImage.open(Path(source).expanduser()) as opened:
        transposed = _PILImageOps.exif_transpose(opened)
        return transposed.convert("RGB")


def rotate_image_cw(image: Image.Image, degrees_cw: int) -> Image.Image:
    """Rotate `image` clockwise by `0`, `90`, `180`, or `270` degrees."""
    from PIL import Image as _PILImage

    deg = int(degrees_cw) % 360
    if deg == 0:
        return image
    if deg == 90:
        return image.transpose(_PILImage.Transpose.ROTATE_270)
    if deg == 180:
        return image.transpose(_PILImage.Transpose.ROTATE_180)
    if deg == 270:
        return image.transpose(_PILImage.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported clockwise rotation angle: {degrees_cw}")


def detect_ink_bbox(
    image: Image.Image,
    *,
    white_threshold: int = 246,
    pad_px: int = 24,
    min_save_ratio: float = 0.08,
    downsample_stride: int = 2,
) -> tuple[int, int, int, int] | None:
    """Find a tight bounding box around non-white foreground pixels, returning `None` if savings are minor."""
    w, h = image.size
    if w < 64 or h < 64:
        return None
    stride = max(1, int(downsample_stride))
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    sub = gray[::stride, ::stride]
    fg_mask = sub < int(white_threshold)
    if not np.any(fg_mask):
        return None
    ys, xs = np.where(fg_mask)
    if ys.size < 16:
        return None

    top = max(0, int(ys.min()) * stride - int(pad_px))
    bottom = min(h, (int(ys.max()) + 1) * stride + int(pad_px))
    left = max(0, int(xs.min()) * stride - int(pad_px))
    right = min(w, (int(xs.max()) + 1) * stride + int(pad_px))

    kept_area = (right - left) * (bottom - top)
    full_area = w * h
    if full_area <= 0 or (1.0 - kept_area / float(full_area)) < float(min_save_ratio):
        return None
    if (right - left) < 64 or (bottom - top) < 64:
        return None
    return (left, top, right, bottom)


def crop_to_ink(
    image: Image.Image,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    pad_px: int = 24,
    min_save_ratio: float = 0.08,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Crop `image` tightly around ink content when it eliminates at least `min_save_ratio` whitespace."""
    box = bbox if bbox is not None else detect_ink_bbox(
        image,
        pad_px=pad_px,
        min_save_ratio=min_save_ratio,
    )
    if box is None:
        return image, None
    left, top, right, bottom = box
    w, h = image.size
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(w, int(right))
    bottom = min(h, int(bottom))
    if right - left < 64 or bottom - top < 64 or (left == 0 and top == 0 and right == w and bottom == h):
        return image, None
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def smart_resize_qwen(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> tuple[int, int]:
    """Exact port of Qwen3-VL `smart_resize` keeping dimensions multiples of `factor` (`patch_size * merge_size`)."""
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError(f"Extreme image aspect ratio unsupported: {width}x{height}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(float(min_pixels) / max(1, height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return int(h_bar), int(w_bar)


def build_qwen_vision_inputs(
    image: Image.Image,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    image_mean: Sequence[float] = (0.5, 0.5, 0.5),
    image_std: Sequence[float] = (0.5, 0.5, 0.5),
) -> QwenVisionInputs:
    """Resize and patchify an RGB image into exact `(N_patches, C * tps * ps * ps)` layout."""
    from PIL import Image as _PILImage

    rgb = image.convert("RGB") if image.mode != "RGB" else image
    w, h = rgb.size
    factor = int(patch_size) * int(merge_size)
    target_h, target_w = smart_resize_qwen(
        h,
        w,
        factor=factor,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    if (target_w, target_h) != (w, h):
        rgb = rgb.resize((target_w, target_h), resample=_PILImage.Resampling.BICUBIC)

    arr_hwc = np.asarray(rgb, dtype=np.float32) * (1.0 / 255.0)
    mean = np.asarray(image_mean, dtype=np.float32)[None, None, :]
    std = np.asarray(image_std, dtype=np.float32)[None, None, :]
    arr_hwc = (arr_hwc - mean) / std
    arr_chw = np.transpose(arr_hwc, (2, 0, 1))

    tps = int(temporal_patch_size)
    ps = int(patch_size)
    ms = int(merge_size)
    grid_t = 1
    grid_h = target_h // ps
    grid_w = target_w // ps
    c = arr_chw.shape[0]

    patches = np.repeat(arr_chw[None, None, ...], tps, axis=1)
    patches = patches.reshape(
        grid_t,
        tps,
        c,
        grid_h // ms,
        ms,
        ps,
        grid_w // ms,
        ms,
        ps,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = np.ascontiguousarray(patches.reshape(grid_t * grid_h * grid_w, c * tps * ps * ps), dtype=np.float32)
    return QwenVisionInputs(
        pixel_values=flat,
        grid_thw=(grid_t, grid_h, grid_w),
        resized_height=target_h,
        resized_width=target_w,
        num_visual_tokens=(grid_t * grid_h * grid_w) // (ms * ms),
    )


@dataclass(frozen=True)
class PplxImageItem:
    """Single preprocessed `pplx-embed-v2` visual document matching `image_processing.py`."""

    vision_inputs: QwenVisionInputs
    input_ids: np.ndarray  # int32 (L,)
    position_ids: np.ndarray  # float32 (3, L)
    visual_mask: np.ndarray  # bool (L,)
    chunk_token_spans: tuple[tuple[int, int], ...]
    rotation_applied_cw: int = 0
    cropped_box_px: tuple[int, int, int, int] | None = None


def white_pad_to_grid_bucket(
    image: Image.Image,
    *,
    bucket_step_patches: int = 4,
    patch_size: int = 16,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize to native Qwen grid, then white-pad bottom-right to a coarser grid multiple for ViT batching."""
    from PIL import Image as _PILImage

    rgb = image.convert("RGB") if image.mode != "RGB" else image
    factor = int(patch_size) * int(merge_size)
    target_h, target_w = smart_resize_qwen(
        rgb.height,
        rgb.width,
        factor=factor,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    if (target_w, target_h) != (rgb.width, rgb.height):
        rgb = rgb.resize((target_w, target_h), resample=_PILImage.Resampling.BICUBIC)

    step = max(1, int(bucket_step_patches))
    if step <= 1:
        return rgb
    step_px = factor * step
    h_buck = ((target_h + step_px - 1) // step_px) * step_px
    w_buck = ((target_w + step_px - 1) // step_px) * step_px
    if h_buck * w_buck > int(max_pixels):
        if h_buck * target_w <= int(max_pixels):
            w_buck = target_w
        elif target_h * w_buck <= int(max_pixels):
            h_buck = target_h
        else:
            return rgb
    if (w_buck, h_buck) == (target_w, target_h):
        return rgb
    canvas = _PILImage.new("RGB", (w_buck, h_buck), fill_color)
    canvas.paste(rgb, (0, 0))
    return canvas


def prepare_pplx_image_item(
    source: str | Path | Image.Image | np.ndarray,
    *,
    document_prefix_id: int = 248078,
    vision_start_id: int = 248053,
    vision_end_id: int = 248054,
    image_token_id: int = 248056,
    chunk_sep_id: int = 248079,
    chunks: int = 8,
    min_pixels: int = 16384,
    max_pixels: int = 1800000,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    rotation_cw: int = 0,
    crop_margins: bool = False,
    crop_pad_px: int = 24,
    grid_bucket_step: int | None = None,
) -> PplxImageItem:
    """Build exact `pplx-embed-v2` token IDs, fractional 3D MRoPE positions, and row-band pooling spans."""
    n_chunks = max(1, int(chunks))
    img = load_image(source)
    rot_cw = int(rotation_cw) % 360
    if rot_cw != 0:
        img = rotate_image_cw(img, rot_cw)
    crop_box: tuple[int, int, int, int] | None = None
    if crop_margins:
        img, crop_box = crop_to_ink(img, pad_px=crop_pad_px)
    if grid_bucket_step is not None and int(grid_bucket_step) > 1:
        img = white_pad_to_grid_bucket(
            img,
            bucket_step_patches=int(grid_bucket_step),
            patch_size=patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    vi = build_qwen_vision_inputs(
        img,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    _t, h_p, w_p = vi.grid_thw
    ms = int(merge_size)
    h_m = h_p // ms
    w_m = w_p // ms
    n_vis = h_m * w_m
    start = 2  # [document_prefix_id, vision_start_id]

    row_groups = [grp for grp in np.array_split(np.arange(h_m, dtype=np.int32), min(n_chunks, h_m)) if grp.size > 0]
    band_lengths = [int(grp.size) * w_m for grp in row_groups]

    sep_count = max(0, len(row_groups) - 1)
    total_len = 3 + n_vis + sep_count
    ids = np.empty((total_len,), dtype=np.int32)
    vis_mask = np.zeros((total_len,), dtype=np.bool_)
    pos = np.empty((3, total_len), dtype=np.float32)

    ids[0] = int(document_prefix_id)
    pos[:, 0] = 0.0
    ids[1] = int(vision_start_id)
    pos[:, 1] = 1.0

    cursor = start
    spans: list[tuple[int, int]] = []
    w_center = float(start) + float(w_m - 1) * 0.5
    col_coords = np.arange(w_m, dtype=np.float32) + float(start)

    for b_idx, grp in enumerate(row_groups):
        if b_idx > 0:
            prev_last_row = float(row_groups[b_idx - 1][-1]) + float(start)
            cur_first_row = float(grp[0]) + float(start)
            ids[cursor] = int(chunk_sep_id)
            pos[0, cursor] = float(start)
            pos[1, cursor] = (prev_last_row + cur_first_row) * 0.5
            pos[2, cursor] = w_center
            cursor += 1
        blen = band_lengths[b_idx]
        b_end = cursor + blen
        ids[cursor:b_end] = int(image_token_id)
        vis_mask[cursor:b_end] = True
        r_coords = np.repeat(grp.astype(np.float32) + float(start), w_m)
        c_coords = np.tile(col_coords, int(grp.size))
        pos[0, cursor:b_end] = float(start)
        pos[1, cursor:b_end] = r_coords
        pos[2, cursor:b_end] = c_coords
        spans.append((cursor, b_end))
        cursor = b_end

    ids[cursor] = int(vision_end_id)
    end_pos = float(start + max(h_m, w_m))
    pos[:, cursor] = end_pos

    return PplxImageItem(
        vision_inputs=vi,
        input_ids=ids,
        position_ids=pos,
        visual_mask=vis_mask,
        chunk_token_spans=tuple(spans),
        rotation_applied_cw=rot_cw,
        cropped_box_px=crop_box,
    )


__all__ = [
    "PplxImageItem",
    "QwenVisionInputs",
    "build_qwen_vision_inputs",
    "crop_to_ink",
    "detect_ink_bbox",
    "load_image",
    "prepare_pplx_image_item",
    "rotate_image_cw",
    "smart_resize_qwen",
    "white_pad_to_grid_bucket",
]
