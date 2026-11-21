"""Post-norm Online Hadamard W8A8 ANE FFN, calibration, breaker, and runtime bridge.

Implements:
1. Block-diagonal post-norm Walsh-Hadamard transforms (128-wide input blocks,
   512-wide intermediate blocks) and symmetric INT8 W8A8 FFN emulation without
   modifying pre-FFN LayerNorm weights.
2. Multi-chunk ANE BLOBFILE weight packing and MIL 1.3 program builders for
   Gemma-4 GeGLU FFN segments (with FP16-safe tanh clamping).
3. Startup dynamic calibration, analytic split-cost fitting, refusal tracking,
   and rolling-window NoGain circuit breaker (`AneFfnCalibration`,
   `AneFfnBreaker`, `AneFfnStartup`).
4. Graceful ctypes bridge to `libfast_ane_handoff.dylib` (`_ANEClient` +
   `MTLSharedEvent` zero-CPU handoff) with automatic `Unavailable` fallback.
"""

from __future__ import annotations

import ctypes
import dataclasses
import enum
import functools
import json
import math
import struct
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Sequence

import numpy as np

# ABI / kernel geometry constants aligned with native implementation
ANE_FFN_ROTATION_UNIT = 128
ANE_FFN_INPUT_BLOCK = 128
ANE_FFN_INTERMEDIATE_BLOCK = 512
ANE_FFN_CHANNEL_UNIT = 512
ANE_FFN_DEFAULT_SEGMENT = 2560
ANE_FFN_TILE = 32
ANE_FFN_INT8_PEAK = 127.0
ANE_FFN_INT8_UNIT = 128.0
ANE_FFN_PEAK_FLOOR = 2.0**-14
ANE_FFN_INTERMEDIATE_FLOOR = 2.0**-9
ANE_FFN_HALF_MAX = 65504.0

DEFAULT_HADAMARD_SEED = 20260930
DEFAULT_MIN_GAIN = 0.05
DEFAULT_CHUNK_NOISE = 0.02
DEFAULT_BREAKER_WINDOW = 8
DEFAULT_MAX_REFUSALS = 2
DEFAULT_MIN_ROWS = 128

PIXEL_FORMAT_L00H = 0x4C303068  # 'L00h' fp16 IOSurface
PIXEL_FORMAT_L008 = 0x4C303038  # 'L008' 8-bit IOSurface

BLOB_HEADER_SIZE = 64
BLOB_CHUNK_HEADER_SIZE = 64
BLOB_MAGIC = 0xDEADBEEF
BLOB_DTYPE_FP16 = 1
BLOB_DTYPE_FP32 = 2
BLOB_DTYPE_UINT8 = 3
BLOB_DTYPE_INT8 = 4


class AneFfnState(str, enum.Enum):
    OFF = "Off"
    UNSUPPORTED = "Unsupported"
    REFUSED = "Refused"
    NO_GAIN = "NoGain"
    UNAVAILABLE = "Unavailable"
    SPLIT = "Split"


@functools.lru_cache(maxsize=8)
def hadamard_matrix(n: int) -> np.ndarray:
    """Return unnormalized Sylvester Walsh-Hadamard matrix H_n (float32, n x n)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard block size must be a positive power of 2, got {n}")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]]).astype(np.float32, copy=False)
    return h


def make_hadamard_signs(dim: int, seed: int = DEFAULT_HADAMARD_SEED) -> np.ndarray:
    """Deterministic Rademacher {+1, -1} vector (float32) of length `dim`."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    rng = np.random.Generator(np.random.MT19937(int(seed)))
    bits = rng.integers(0, 2, size=dim, dtype=np.int32)
    return (bits * 2 - 1).astype(np.float32)


def hadamard_rotate_blocks(
    x: np.ndarray,
    block_size: int,
    signs: np.ndarray | None = None,
) -> np.ndarray:
    """Apply block-diagonal post-norm orthogonal Hadamard Q_b = H_b * diag(s) / sqrt(b)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError(f"Expected >= 1D array, got shape {arr.shape}")
    dim = arr.shape[-1]
    if dim % block_size != 0:
        raise ValueError(f"Last dimension {dim} is not divisible by block_size {block_size}")
    if signs is not None:
        s = np.asarray(signs, dtype=np.float32)
        if s.shape != (dim,):
            raise ValueError(f"Expected signs shape {(dim,)}, got {s.shape}")
        arr = arr * s
    h = hadamard_matrix(block_size)
    inv_sqrt_b = np.float32(1.0 / math.sqrt(float(block_size)))
    blocks = arr.shape[:-1] + (dim // block_size, block_size)
    rotated = (arr.reshape(blocks) @ h) * inv_sqrt_b
    return rotated.reshape(arr.shape)


def quantize_symmetric_int8(
    x: np.ndarray,
    floor: float = ANE_FFN_PEAK_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization matching `absmax / 127.0` with peak floor."""
    arr = np.asarray(x, dtype=np.float32)
    peak = np.maximum(np.max(np.abs(arr), axis=-1), np.float32(floor))
    scale = (peak / np.float32(ANE_FFN_INT8_PEAK)).astype(np.float32)
    q = np.clip(np.rint(arr / scale[..., None]), -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK).astype(np.int8)
    return q, scale


def quantize_symmetric_int8_grouped(
    x: np.ndarray,
    group_size: int,
    floor: float = ANE_FFN_PEAK_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group symmetric INT8 quantization along last axis."""
    arr = np.asarray(x, dtype=np.float32)
    dim = arr.shape[-1]
    if dim % group_size != 0:
        raise ValueError(f"Dim {dim} not divisible by group_size {group_size}")
    grouped = arr.reshape(arr.shape[:-1] + (dim // group_size, group_size))
    q, s = quantize_symmetric_int8(grouped, floor=floor)
    return q.reshape(arr.shape), s


def gelu_pytorch_tanh(x: np.ndarray, fp16_safe_clamp: bool = False) -> np.ndarray:
    """Gemma-4 `gelu_pytorch_tanh` with optional FP16 cubic overflow protection inside tanh."""
    arr = np.asarray(x, dtype=np.float32)
    xc = np.clip(arr, -10.0, 10.0) if fp16_safe_clamp else arr
    inner = np.float32(0.7978845608028654) * (xc + np.float32(0.044715) * (xc * xc * xc))
    return np.float32(0.5) * arr * (np.float32(1.0) + np.tanh(inner))


def silu(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr / (np.float32(1.0) + np.exp(-np.clip(arr, -80.0, 80.0)))


@dataclasses.dataclass
class AneFfnW8A8Weights:
    hidden_size: int
    intermediate_size: int
    hadamard_mode: str
    signs_in: np.ndarray | None
    signs_mid: np.ndarray | None
    q_gate: np.ndarray
    scale_gate: np.ndarray
    q_up: np.ndarray
    scale_up: np.ndarray
    q_down: np.ndarray
    scale_down: np.ndarray


def prepare_ffn_w8a8_weights(
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    hadamard_mode: str = "full",
    seed: int = DEFAULT_HADAMARD_SEED,
) -> AneFfnW8A8Weights:
    """Prepare post-norm Hadamard-rotated INT8 FFN weights without touching layernorm."""
    wg = np.asarray(w_gate, dtype=np.float32)
    wu = np.asarray(w_up, dtype=np.float32)
    wd = np.asarray(w_down, dtype=np.float32)

    if wg.ndim != 2 or wu.ndim != 2 or wd.ndim != 2:
        raise ValueError("Weights must be 2D matrices")
    d_ff, h_dim = wg.shape
    if wu.shape != (d_ff, h_dim) or wd.shape != (h_dim, d_ff):
        raise ValueError(f"Inconsistent FFN shapes: gate={wg.shape}, up={wu.shape}, down={wd.shape}")
    if hadamard_mode not in ("full", "input_only", "none"):
        raise ValueError(f"Unknown hadamard_mode={hadamard_mode!r}")

    signs_in: np.ndarray | None = None
    signs_mid: np.ndarray | None = None

    if hadamard_mode in ("full", "input_only"):
        if h_dim % ANE_FFN_INPUT_BLOCK != 0:
            raise ValueError(f"hidden_size {h_dim} must be divisible by {ANE_FFN_INPUT_BLOCK}")
        signs_in = make_hadamard_signs(h_dim, seed=seed)
        wg_rot = hadamard_rotate_blocks(wg, ANE_FFN_INPUT_BLOCK, signs_in)
        wu_rot = hadamard_rotate_blocks(wu, ANE_FFN_INPUT_BLOCK, signs_in)
    else:
        wg_rot = wg
        wu_rot = wu

    if hadamard_mode == "full":
        if d_ff % ANE_FFN_INTERMEDIATE_BLOCK != 0:
            raise ValueError(f"intermediate_size {d_ff} must be divisible by {ANE_FFN_INTERMEDIATE_BLOCK}")
        signs_mid = make_hadamard_signs(d_ff, seed=seed + 1)
        wd_rot = hadamard_rotate_blocks(wd, ANE_FFN_INTERMEDIATE_BLOCK, signs_mid)
    else:
        wd_rot = wd

    q_gate, s_gate = quantize_symmetric_int8(wg_rot)
    q_up, s_up = quantize_symmetric_int8(wu_rot)
    q_down, s_down = quantize_symmetric_int8(wd_rot)

    return AneFfnW8A8Weights(
        hidden_size=h_dim,
        intermediate_size=d_ff,
        hadamard_mode=hadamard_mode,
        signs_in=signs_in,
        signs_mid=signs_mid,
        q_gate=q_gate,
        scale_gate=s_gate,
        q_up=q_up,
        scale_up=s_up,
        q_down=q_down,
        scale_down=s_down,
    )


def eval_ffn_fp32_ref(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    activation: str = "gelu_tanh",
) -> np.ndarray:
    """Exact FP32 FFN reference output."""
    xf = np.asarray(x, dtype=np.float32)
    gate = xf @ np.asarray(w_gate, dtype=np.float32).T
    up = xf @ np.asarray(w_up, dtype=np.float32).T
    if activation == "gelu_tanh":
        act = gelu_pytorch_tanh(gate, fp16_safe_clamp=False)
    elif activation == "silu":
        act = silu(gate)
    else:
        raise ValueError(f"Unsupported activation={activation!r}")
    return (act * up) @ np.asarray(w_down, dtype=np.float32).T


def _w8a8_linear(q_x: np.ndarray, s_x: np.ndarray, q_w: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    # Exact integer products (max |acc| <= K * 127^2 << 2^53) accelerated via float64 BLAS dgemm.
    acc = q_x.astype(np.float64) @ q_w.astype(np.float64).T
    return (acc * (s_x.astype(np.float64)[:, None] * s_w.astype(np.float64)[None, :])).astype(np.float32)


def eval_ffn_w8a8(
    x: np.ndarray,
    weights: AneFfnW8A8Weights,
    *,
    activation: str = "gelu_tanh",
    fp16_intermediate: bool = False,
    fp16_safe_clamp: bool = True,
) -> np.ndarray:
    """Evaluate FFN under post-norm Hadamard + symmetric INT8 W8A8 emulation."""
    xf = np.asarray(x, dtype=np.float32)
    if weights.signs_in is not None:
        x_in = hadamard_rotate_blocks(xf, ANE_FFN_INPUT_BLOCK, weights.signs_in)
    else:
        x_in = xf

    q_x, s_x = quantize_symmetric_int8(x_in, floor=ANE_FFN_PEAK_FLOOR)
    gate = _w8a8_linear(q_x, s_x, weights.q_gate, weights.scale_gate)
    up = _w8a8_linear(q_x, s_x, weights.q_up, weights.scale_up)

    if fp16_intermediate:
        gate = gate.astype(np.float16).astype(np.float32)
        up = up.astype(np.float16).astype(np.float32)

    if activation == "gelu_tanh":
        act = gelu_pytorch_tanh(gate, fp16_safe_clamp=fp16_safe_clamp)
    elif activation == "silu":
        act = silu(gate)
    else:
        raise ValueError(f"Unsupported activation={activation!r}")

    if fp16_intermediate:
        act = act.astype(np.float16).astype(np.float32)
    mid = act * up
    if fp16_intermediate:
        mid = np.clip(mid, -ANE_FFN_HALF_MAX, ANE_FFN_HALF_MAX).astype(np.float16).astype(np.float32)

    if weights.signs_mid is not None:
        mid_rot = hadamard_rotate_blocks(mid, ANE_FFN_INTERMEDIATE_BLOCK, weights.signs_mid)
    else:
        mid_rot = mid

    q_mid, s_mid = quantize_symmetric_int8(mid_rot, floor=ANE_FFN_INTERMEDIATE_FLOOR)
    out = _w8a8_linear(q_mid, s_mid, weights.q_down, weights.scale_down)
    return out


def compute_error_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    """Compute cosine similarity, relative RMS error, and max absolute error."""
    r = np.asarray(ref, dtype=np.float64).ravel()
    t = np.asarray(test, dtype=np.float64).ravel()
    r_norm = float(np.linalg.norm(r))
    t_norm = float(np.linalg.norm(t))
    diff_norm = float(np.linalg.norm(r - t))
    if r_norm == 0.0 and t_norm == 0.0:
        cos = 1.0
        rel_rms = 0.0
    elif r_norm == 0.0 or t_norm == 0.0:
        cos = 0.0
        rel_rms = float("inf")
    else:
        cos = float(np.dot(r, t) / (r_norm * t_norm))
        rel_rms = diff_norm / r_norm
    return {
        "cosine": cos,
        "rel_rms_err": rel_rms,
        "max_abs_err": float(np.max(np.abs(r - t))),
    }


def align_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def segment_intermediate_sizes(
    total_interm: int,
    segment_limit: int = ANE_FFN_DEFAULT_SEGMENT,
) -> list[int]:
    """Partition intermediate width `total_interm` into ANE segments <= `segment_limit`."""
    if total_interm <= 0:
        return []
    out: list[int] = []
    rem = total_interm
    while rem > 0:
        seg = min(rem, segment_limit)
        out.append(seg)
        rem -= seg
    return out


def pack_weight_blobs(
    arrays: Sequence[tuple[str, np.ndarray, int]],
) -> tuple[bytes, dict[str, int]]:
    """Pack multiple weight arrays into one 64-byte-aligned ANE weight.bin buffer.

    Returns `(blob_bytes, offset_by_name)` where `offset_by_name[name]` is the
    `BLOBFILE(..., offset = uint64(...))` byte offset (pointing at the 0xDEADBEEF chunk header).
    """
    buf = bytearray(BLOB_HEADER_SIZE)
    struct.pack_into("<II", buf, 0, len(arrays), 2)
    offsets: dict[str, int] = {}

    for name, arr, dtype_code in arrays:
        raw = np.ascontiguousarray(arr).tobytes()
        while len(buf) % 64 != 0:
            buf.append(0)
        chunk_offset = len(buf)
        data_offset = chunk_offset + BLOB_CHUNK_HEADER_SIZE
        hdr = bytearray(BLOB_CHUNK_HEADER_SIZE)
        struct.pack_into("<IIQQ", hdr, 0, BLOB_MAGIC, int(dtype_code), len(raw), data_offset)
        buf.extend(hdr)
        buf.extend(raw)
        offsets[name] = chunk_offset

    while len(buf) % 64 != 0:
        buf.append(0)
    return bytes(buf), offsets


def build_linear_conv_mil(
    in_channels: int,
    out_channels: int,
    tokens: int,
    *,
    weight_offset: int = BLOB_HEADER_SIZE,
    weight_dtype: str = "fp16",
    scale_offset: int | None = None,
    has_token_scale: bool = False,
    input_dtype: str = "fp16",
) -> str:
    """Build MIL 1.3 text for `[1, C_in, 1, S] -> [1, C_out, 1, S]` 1x1 conv."""
    if weight_dtype not in ("fp16", "int8_dequant"):
        raise ValueError(f"Unsupported weight_dtype={weight_dtype!r}")
    if input_dtype not in ("fp16", "int8"):
        raise ValueError(f"Unsupported input_dtype={input_dtype!r}")

    lines = [
        "program(1.3)",
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3605.1.1"}, {"coremlc-version", "3600.25.1"}})]',
        "{",
    ]
    if has_token_scale:
        sig = (
            f"    func main<ios18>(tensor<{input_dtype}, [1, {in_channels}, 1, {tokens}]> x, "
            f"tensor<fp16, [1, 1, 1, {tokens}]> tok_scale) {{"
        )
    else:
        sig = f"    func main<ios18>(tensor<{input_dtype}, [1, {in_channels}, 1, {tokens}]> x) {{"
    lines.append(sig)
    if input_dtype == "int8":
        lines.extend(
            [
                '        string cast_dt = const()[name = string("cast_dt"), val = string("fp16")];',
                f'        tensor<fp16, [1, {in_channels}, 1, {tokens}]> x_act = cast(dtype = cast_dt, x = x)[name = string("x_act")];',
            ]
        )
        x_var = "x_act"
    else:
        x_var = "x"
    lines.extend(
        [
            '        string pt = const()[name = string("pt"), val = string("valid")];',
            '        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];',
            '        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
            '        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];',
            '        int32 gr = const()[name = string("gr"), val = int32(1)];',
        ]
    )
    if weight_dtype == "fp16":
        lines.append(
            f'        tensor<fp16, [{out_channels}, {in_channels}, 1, 1]> w = const()[name = string("w"), '
            f'val = tensor<fp16, [{out_channels}, {in_channels}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({weight_offset})))];'
        )
    else:
        if scale_offset is None:
            raise ValueError("scale_offset required for int8_dequant")
        lines.append(
            f'        tensor<fp16, [{out_channels}, {in_channels}, 1, 1]> w = constexpr_blockwise_shift_scale('
            f'data = tensor<int8, [{out_channels}, {in_channels}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({weight_offset}))), '
            f'scale = tensor<fp16, [{out_channels}, 1, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({scale_offset}))))[name = string("w")];'
        )

    out_var = "y"
    if has_token_scale:
        lines.append(
            f'        tensor<fp16, [1, {out_channels}, 1, {tokens}]> y_raw = conv('
            f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = {x_var})[name = string("y_raw")];'
        )
        lines.append(
            f'        tensor<fp16, [1, {out_channels}, 1, {tokens}]> y = mul(x = y_raw, y = tok_scale)[name = string("y")];'
        )
    else:
        lines.append(
            f'        tensor<fp16, [1, {out_channels}, 1, {tokens}]> y = conv('
            f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = {x_var})[name = string("y")];'
        )

    lines.append(f"    }} -> ({out_var});")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_ffn_geglu_segment_mil(
    hidden_size: int,
    seg_interm: int,
    tokens: int,
    *,
    w_gate_off: int,
    w_up_off: int,
    w_down_off: int,
    w_gate_scale_off: int | None = None,
    w_up_scale_off: int | None = None,
    w_down_scale_off: int | None = None,
    weight_dtype: str = "fp16",
    input_dtype: str = "fp16",
    activation: str = "gelu_tanh",
    apply_token_scale: bool = False,
    packed_token_scale: bool = False,
    fp16_safe_clamp: bool = True,
    normalize_mid: bool = True,
) -> str:
    """Build MIL 1.3 GeGLU/SwiGLU FFN segment program `[1, H(+1), 1, S] -> [1, H, 1, S]`."""
    if input_dtype not in ("fp16", "int8"):
        raise ValueError(f"Unsupported input_dtype={input_dtype!r}")
    if packed_token_scale and apply_token_scale:
        raise ValueError("packed_token_scale and apply_token_scale are mutually exclusive")
    if packed_token_scale and input_dtype != "fp16":
        raise ValueError("packed_token_scale requires fp16 packed input surface")

    lines = [
        "program(1.3)",
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3605.1.1"}, {"coremlc-version", "3600.25.1"}})]',
        "{",
    ]
    if packed_token_scale:
        lines.append(f"    func main<ios18>(tensor<fp16, [1, {hidden_size + 1}, 1, {tokens}]> x_cat) {{")
    elif apply_token_scale:
        lines.append(
            f"    func main<ios18>(tensor<{input_dtype}, [1, {hidden_size}, 1, {tokens}]> x, "
            f"tensor<fp16, [1, 1, 1, {tokens}]> tok_scale) {{"
        )
    else:
        lines.append(f"    func main<ios18>(tensor<{input_dtype}, [1, {hidden_size}, 1, {tokens}]> x) {{")

    if packed_token_scale:
        lines.extend(
            [
                f'        tensor<int32, [4]> b_x = const()[name = string("b_x"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
                f'        tensor<int32, [4]> s_x = const()[name = string("s_x"), val = tensor<int32, [4]>([1, {hidden_size}, 1, {tokens}])];',
                f'        tensor<fp16, [1, {hidden_size}, 1, {tokens}]> x_act = slice_by_size(x = x_cat, begin = b_x, size = s_x)[name = string("x_act")];',
                f'        tensor<int32, [4]> b_t = const()[name = string("b_t"), val = tensor<int32, [4]>([0, {hidden_size}, 0, 0])];',
                f'        tensor<int32, [4]> s_t = const()[name = string("s_t"), val = tensor<int32, [4]>([1, 1, 1, {tokens}])];',
                f'        tensor<fp16, [1, 1, 1, {tokens}]> tok_scale = slice_by_size(x = x_cat, begin = b_t, size = s_t)[name = string("tok_scale")];',
            ]
        )
        x_var = "x_act"
    elif input_dtype == "int8":
        lines.extend(
            [
                '        string cast_dt = const()[name = string("cast_dt"), val = string("fp16")];',
                f'        tensor<fp16, [1, {hidden_size}, 1, {tokens}]> x_act = cast(dtype = cast_dt, x = x)[name = string("x_act")];',
            ]
        )
        x_var = "x_act"
    else:
        x_var = "x"

    lines.extend(
        [
            '        string pt = const()[name = string("pt"), val = string("valid")];',
            '        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];',
            '        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
            '        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];',
            '        int32 gr = const()[name = string("gr"), val = int32(1)];',
        ]
    )

    def _emit_weight(name: str, oc: int, ic: int, w_off: int, s_off: int | None) -> None:
        if weight_dtype == "fp16":
            lines.append(
                f'        tensor<fp16, [{oc}, {ic}, 1, 1]> {name} = const()[name = string("{name}"), '
                f'val = tensor<fp16, [{oc}, {ic}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({w_off})))];'
            )
        elif weight_dtype == "int8_dequant":
            if s_off is None:
                raise ValueError(f"Missing scale offset for {name}")
            lines.append(
                f'        tensor<fp16, [{oc}, {ic}, 1, 1]> {name} = constexpr_blockwise_shift_scale('
                f'data = tensor<int8, [{oc}, {ic}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({w_off}))), '
                f'scale = tensor<fp16, [{oc}, 1, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({s_off}))))[name = string("{name}")];'
            )
        else:
            raise ValueError(f"Unsupported weight_dtype={weight_dtype!r}")

    _emit_weight("wg", seg_interm, hidden_size, w_gate_off, w_gate_scale_off)
    _emit_weight("wu", seg_interm, hidden_size, w_up_off, w_up_scale_off)
    _emit_weight("wd", hidden_size, seg_interm, w_down_off, w_down_scale_off)

    lines.append(
        f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g_raw = conv('
        f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = wg, x = {x_var})[name = string("g_raw")];'
    )
    lines.append(
        f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> u_raw = conv('
        f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = wu, x = {x_var})[name = string("u_raw")];'
    )
    if packed_token_scale:
        # Only scale gate by tok_scale; keep u_raw unit-scaled until post-join so mid/down avoid FP16 overflow.
        lines.append(
            f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g = mul(x = g_raw, y = tok_scale)[name = string("g")];'
        )
        g_name = "g"
        u_name = "u_raw"
    elif apply_token_scale:
        lines.append(
            f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g = mul(x = g_raw, y = tok_scale)[name = string("g")];'
        )
        lines.append(
            f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> u = mul(x = u_raw, y = tok_scale)[name = string("u")];'
        )
        g_name = "g"
        u_name = "u"
    else:
        g_name = "g_raw"
        u_name = "u_raw"

    if activation == "gelu_tanh":
        lines.append('        string gelu_mode = const()[name = string("gelu_mode"), val = string("TANH_APPROXIMATION")];')
        if fp16_safe_clamp:
            lines.extend(
                [
                    '        fp16 c_lo = const()[name = string("c_lo"), val = fp16(-10.0)];',
                    '        fp16 c_hi = const()[name = string("c_hi"), val = fp16(10.0)];',
                    f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g_clip = clip(alpha = c_lo, beta = c_hi, x = {g_name})[name = string("g_clip")];',
                    f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> act_core = gelu(mode = gelu_mode, x = g_clip)[name = string("act_core")];',
                    f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g_diff = sub(x = {g_name}, y = g_clip)[name = string("g_diff")];',
                    f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> g_tail = relu(x = g_diff)[name = string("g_tail")];',
                    f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> act = add(x = act_core, y = g_tail)[name = string("act")];',
                ]
            )
        else:
            lines.append(
                f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> act = gelu(mode = gelu_mode, x = {g_name})[name = string("act")];'
            )
    elif activation == "silu":
        lines.extend(
            [
                f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> sig = sigmoid(x = {g_name})[name = string("sig")];',
                f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> act = mul(x = {g_name}, y = sig)[name = string("act")];',
            ]
        )
    else:
        raise ValueError(f"Unsupported activation={activation!r}")

    lines.append(
        f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> mid = mul(x = act, y = {u_name})[name = string("mid")];'
    )
    if normalize_mid and weight_dtype == "int8_dequant":
        lines.extend(
            [
                f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> mabs = abs(x = mid)[name = string("mabs")];',
                '        tensor<int32, [1]> ax1 = const()[name = string("ax1"), val = tensor<int32, [1]>([1])];',
                '        bool kd1 = const()[name = string("kd1"), val = bool(true)];',
                f'        tensor<fp16, [1, 1, 1, {tokens}]> mpeak = reduce_max(axes = ax1, keep_dims = kd1, x = mabs)[name = string("mpeak")];',
                '        fp16 mfl = const()[name = string("mfl"), val = fp16(1.0)];',
                f'        tensor<fp16, [1, 1, 1, {tokens}]> mscale = maximum(x = mpeak, y = mfl)[name = string("mscale")];',
                f'        tensor<fp16, [1, {seg_interm}, 1, {tokens}]> mid_u = real_div(x = mid, y = mscale)[name = string("mid_u")];',
                f'        tensor<fp16, [1, {hidden_size}, 1, {tokens}]> y_mid = conv('
                f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = wd, x = mid_u)[name = string("y_mid")];',
                f'        tensor<fp16, [1, {hidden_size}, 1, {tokens}]> y = mul(x = y_mid, y = mscale)[name = string("y")];',
            ]
        )
    else:
        lines.append(
            f'        tensor<fp16, [1, {hidden_size}, 1, {tokens}]> y = conv('
            f'dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = wd, x = mid)[name = string("y")];'
        )
    lines.extend(["    } -> (y);", "}"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dynamic Calibration (`AneFfnCalibration`), Breaker, and Startup Policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CalibrationSample:
    """One calibration measurement for ANE share `s in (0, 1]`."""

    share: float
    ane_ms: float
    tail_gpu_ms: float
    wall_ms: float


@dataclasses.dataclass(frozen=True)
class SplitDecision:
    state: AneFfnState
    share: float
    ane_channels: int
    gpu_channels: int
    predicted_split_ms: float
    gpu_full_ms: float
    predicted_gain: float
    reason: str = ""


class AneFfnCalibration:
    """Analytic overlap cost model: T(s) = max(A(s), G(s) + u * A(s))."""

    def __init__(
        self,
        *,
        ane_fixed_ms: float = 0.0,
        ane_slope_ms: float = 0.0,
        gpu_full_ms: float = 0.0,
        overlap_u: float = 0.0,
    ) -> None:
        self.ane_fixed_ms = max(0.0, float(ane_fixed_ms))
        self.ane_slope_ms = max(0.0, float(ane_slope_ms))
        self.gpu_full_ms = max(0.0, float(gpu_full_ms))
        self.overlap_u = max(0.0, float(overlap_u))

    def predict_ane_ms(self, share: float) -> float:
        if share <= 0.0:
            return 0.0
        return self.ane_fixed_ms + self.ane_slope_ms * float(share)

    def predict_gpu_ms(self, share: float) -> float:
        s = min(1.0, max(0.0, float(share)))
        tail = self.gpu_full_ms * (1.0 - s)
        return tail + self.overlap_u * self.predict_ane_ms(s)

    def predict_split_ms(self, share: float) -> float:
        if share <= 0.0:
            return self.gpu_full_ms
        return max(self.predict_ane_ms(share), self.predict_gpu_ms(share))

    @classmethod
    def fit(cls, samples: Sequence[CalibrationSample], gpu_full_ms: float) -> "AneFfnCalibration":
        """Fit `(ane_fixed_ms, ane_slope_ms, overlap_u)` from measured calibration samples."""
        if gpu_full_ms <= 0.0:
            raise ValueError(f"gpu_full_ms must be positive, got {gpu_full_ms}")
        valid = [p for p in samples if 0.0 < p.share <= 1.0 and p.ane_ms >= 0.0 and p.tail_gpu_ms >= 0.0]
        if not valid:
            return cls(ane_fixed_ms=0.0, ane_slope_ms=gpu_full_ms * 2.0, gpu_full_ms=gpu_full_ms, overlap_u=0.0)

        shares = np.array([p.share for p in valid], dtype=np.float64)
        ane_y = np.array([p.ane_ms for p in valid], dtype=np.float64)
        if len(valid) >= 2 and np.ptp(shares) > 1e-6:
            X = np.column_stack([np.ones_like(shares), shares])
            coef, *_ = np.linalg.lstsq(X, ane_y, rcond=None)
            fixed = max(0.0, float(coef[0]))
            slope = max(0.0, float(coef[1]))
            # If negative slope or fixed pushed intercept negative, clamp refit cleanly.
            if coef[0] < 0.0:
                fixed = 0.0
                slope = max(0.0, float(np.dot(shares, ane_y) / np.dot(shares, shares)))
            elif coef[1] < 0.0:
                slope = 0.0
                fixed = max(0.0, float(np.mean(ane_y)))
        else:
            fixed = 0.0
            slope = max(0.0, float(ane_y[0] / shares[0]))

        # Overlap u = excess GPU tail latency per ms of concurrent ANE work.
        denom = float(np.sum(ane_y * ane_y))
        if denom > 1e-12:
            residuals = np.array([p.tail_gpu_ms - gpu_full_ms * (1.0 - p.share) for p in valid], dtype=np.float64)
            overlap_u = max(0.0, float(np.dot(ane_y, residuals) / denom))
        else:
            overlap_u = 0.0

        return cls(
            ane_fixed_ms=fixed,
            ane_slope_ms=slope,
            gpu_full_ms=float(gpu_full_ms),
            overlap_u=overlap_u,
        )

    def refit(self, samples: Sequence[CalibrationSample], gpu_full_ms: float | None = None) -> "AneFfnCalibration":
        return AneFfnCalibration.fit(samples, self.gpu_full_ms if gpu_full_ms is None else gpu_full_ms)

    def slower_nowhere(
        self,
        share: float,
        *,
        min_gain: float = DEFAULT_MIN_GAIN,
        chunk_noise: float = DEFAULT_CHUNK_NOISE,
    ) -> bool:
        """Check whether predicted split latency clears `min_gain` and avoids boundary slowdown."""
        if self.gpu_full_ms <= 0.0 or share <= 0.0:
            return False
        pred = self.predict_split_ms(share)
        gain = (self.gpu_full_ms - pred) / self.gpu_full_ms
        if gain < min_gain:
            return False
        return pred <= self.gpu_full_ms * (1.0 - min_gain + chunk_noise)

    def choose(
        self,
        total_interm: int,
        *,
        channel_unit: int = ANE_FFN_CHANNEL_UNIT,
        min_gain: float = DEFAULT_MIN_GAIN,
        chunk_noise: float = DEFAULT_CHUNK_NOISE,
    ) -> SplitDecision:
        """Pick optimal `channel_unit`-aligned ANE split share or return `NO_GAIN`."""
        if total_interm <= 0 or channel_unit <= 0 or total_interm % channel_unit != 0:
            return SplitDecision(
                state=AneFfnState.UNSUPPORTED,
                share=0.0,
                ane_channels=0,
                gpu_channels=max(0, total_interm),
                predicted_split_ms=self.gpu_full_ms,
                gpu_full_ms=self.gpu_full_ms,
                predicted_gain=0.0,
                reason=f"total_interm={total_interm} incompatible with channel_unit={channel_unit}",
            )
        if self.gpu_full_ms <= 0.0:
            return SplitDecision(
                state=AneFfnState.NO_GAIN,
                share=0.0,
                ane_channels=0,
                gpu_channels=total_interm,
                predicted_split_ms=0.0,
                gpu_full_ms=self.gpu_full_ms,
                predicted_gain=0.0,
                reason="non-positive gpu_full_ms",
            )

        best_c = 0
        best_ms = self.gpu_full_ms
        n_units = total_interm // channel_unit
        for k in range(1, n_units + 1):
            c = k * channel_unit
            s = c / float(total_interm)
            t = self.predict_split_ms(s)
            if t < best_ms:
                best_ms = t
                best_c = c

        best_s = best_c / float(total_interm) if best_c > 0 else 0.0
        gain = (self.gpu_full_ms - best_ms) / self.gpu_full_ms if best_c > 0 else 0.0
        if best_c == 0 or not self.slower_nowhere(best_s, min_gain=min_gain, chunk_noise=chunk_noise):
            return SplitDecision(
                state=AneFfnState.NO_GAIN,
                share=0.0,
                ane_channels=0,
                gpu_channels=total_interm,
                predicted_split_ms=self.gpu_full_ms,
                gpu_full_ms=self.gpu_full_ms,
                predicted_gain=gain,
                reason=f"predicted gain {gain:.4f} < min_gain {min_gain:.4f}",
            )

        return SplitDecision(
            state=AneFfnState.SPLIT,
            share=best_s,
            ane_channels=best_c,
            gpu_channels=total_interm - best_c,
            predicted_split_ms=best_ms,
            gpu_full_ms=self.gpu_full_ms,
            predicted_gain=gain,
            reason="ok",
        )


class AneFfnBreaker:
    """Rolling-window runtime safety circuit breaker for ANE/GPU split execution."""

    def __init__(
        self,
        *,
        window: int = DEFAULT_BREAKER_WINDOW,
        min_gain: float = DEFAULT_MIN_GAIN,
        chunk_noise: float = DEFAULT_CHUNK_NOISE,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = int(window)
        self.min_gain = float(min_gain)
        self.chunk_noise = float(chunk_noise)
        self._samples: Deque[tuple[float, float]] = deque(maxlen=self.window)
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str:
        return self._reason

    def observe(self, gpu_full_ms: float, split_ms: float) -> bool:
        """Record one realized `(gpu_full_ms, split_ms)` pair; returns True if split remains allowed."""
        if self._tripped:
            return False
        if gpu_full_ms <= 0.0:
            return True
        self._samples.append((float(gpu_full_ms), float(split_ms)))
        if len(self._samples) < self.window:
            return True

        sum_gpu = sum(g for g, _ in self._samples)
        sum_split = sum(s for _, s in self._samples)
        realized_gain = (sum_gpu - sum_split) / sum_gpu if sum_gpu > 0.0 else -1.0
        allowed_min_gain = self.min_gain - self.chunk_noise
        if realized_gain < allowed_min_gain:
            self._tripped = True
            self._reason = (
                f"window={self.window} realized_gain={realized_gain:.4f} "
                f"< threshold={allowed_min_gain:.4f} (gpu={sum_gpu:.2f}ms split={sum_split:.2f}ms)"
            )
            return False
        return True


class AneFfnStartup:
    """Startup feasibility, refusal limit, and dynamic calibration evaluator."""

    @staticmethod
    def decide(
        *,
        enabled: bool,
        ane_available: bool,
        tokens: int,
        hidden_size: int,
        intermediate_size: int,
        samples: Sequence[CalibrationSample],
        gpu_full_ms: float,
        refusal_count: int = 0,
        max_refusals: int = DEFAULT_MAX_REFUSALS,
        min_rows: int = DEFAULT_MIN_ROWS,
        min_gain: float = DEFAULT_MIN_GAIN,
        chunk_noise: float = DEFAULT_CHUNK_NOISE,
        channel_unit: int = ANE_FFN_CHANNEL_UNIT,
    ) -> tuple[SplitDecision, AneFfnCalibration | None]:
        if not enabled:
            return (
                SplitDecision(
                    state=AneFfnState.OFF,
                    share=0.0,
                    ane_channels=0,
                    gpu_channels=intermediate_size,
                    predicted_split_ms=gpu_full_ms,
                    gpu_full_ms=gpu_full_ms,
                    predicted_gain=0.0,
                    reason="disabled by config",
                ),
                None,
            )

        if refusal_count >= max_refusals:
            return (
                SplitDecision(
                    state=AneFfnState.REFUSED,
                    share=0.0,
                    ane_channels=0,
                    gpu_channels=intermediate_size,
                    predicted_split_ms=gpu_full_ms,
                    gpu_full_ms=gpu_full_ms,
                    predicted_gain=0.0,
                    reason=f"refusal_count={refusal_count} >= max_refusals={max_refusals}",
                ),
                None,
            )

        if (
            tokens < min_rows
            or hidden_size <= 0
            or intermediate_size <= 0
            or hidden_size % ANE_FFN_INPUT_BLOCK != 0
            or intermediate_size % channel_unit != 0
        ):
            return (
                SplitDecision(
                    state=AneFfnState.UNSUPPORTED,
                    share=0.0,
                    ane_channels=0,
                    gpu_channels=max(0, intermediate_size),
                    predicted_split_ms=gpu_full_ms,
                    gpu_full_ms=gpu_full_ms,
                    predicted_gain=0.0,
                    reason=(
                        f"unsupported shape M={tokens}, H={hidden_size}, D={intermediate_size} "
                        f"(min_rows={min_rows}, H%{ANE_FFN_INPUT_BLOCK}, D%{channel_unit})"
                    ),
                ),
                None,
            )

        if not ane_available:
            return (
                SplitDecision(
                    state=AneFfnState.UNAVAILABLE,
                    share=0.0,
                    ane_channels=0,
                    gpu_channels=intermediate_size,
                    predicted_split_ms=gpu_full_ms,
                    gpu_full_ms=gpu_full_ms,
                    predicted_gain=0.0,
                    reason="ANE private runtime or MTLSharedEvent unavailable",
                ),
                None,
            )

        cal = AneFfnCalibration.fit(samples, gpu_full_ms=gpu_full_ms)
        decision = cal.choose(
            intermediate_size,
            channel_unit=channel_unit,
            min_gain=min_gain,
            chunk_noise=chunk_noise,
        )
        return decision, cal


# ---------------------------------------------------------------------------
# Native Shared Library (`libfast_ane_handoff.dylib`) Bridge
# ---------------------------------------------------------------------------


def default_native_ane_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "native_ane"


def default_native_ane_dylib() -> Path:
    return default_native_ane_dir() / "build" / "libfast_ane_handoff.dylib"


def ensure_native_ane_built(*, clean: bool = False) -> Path:
    native_dir = default_native_ane_dir()
    dylib = default_native_ane_dylib()
    if dylib.exists() and not clean:
        return dylib
    targets = ["clean", "all"] if clean else ["all"]
    subprocess.run(["make", "-C", str(native_dir), *targets], check=True)
    return dylib


class _FastAneHandoffMetrics(ctypes.Structure):
    _fields_ = [
        ("gpu_pre_ms", ctypes.c_double),
        ("ane_submit_ms", ctypes.c_double),
        ("gpu_post_and_total_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("used_shared_event", ctypes.c_int32),
        ("used_iosurface_metal_buffer", ctypes.c_int32),
        ("final_shared_event_value", ctypes.c_uint64),
    ]


class _FastAneWaveMetrics(ctypes.Structure):
    _fields_ = [
        ("gpu_submit_ms", ctypes.c_double),
        ("ane_submit_ms", ctypes.c_double),
        ("finish_wait_ms", ctypes.c_double),
        ("total_wall_ms", ctypes.c_double),
        ("used_shared_event", ctypes.c_int32),
    ]


class AneNativeError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"FastAne status={status}: {message}")
        self.status = status
        self.message = message


class AneHandoffLibrary:
    """Thin ctypes wrapper over `libfast_ane_handoff.dylib`."""

    def __init__(self, dylib_path: str | Path | None = None, *, auto_build: bool = False) -> None:
        path = Path(dylib_path) if dylib_path is not None else default_native_ane_dylib()
        if not path.exists() and auto_build and dylib_path is None:
            path = ensure_native_ane_built()
        if not path.exists():
            raise FileNotFoundError(f"Missing native ANE library: {path}")
        self.dylib_path = path
        self._lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.fast_ane_probe_compat_json.argtypes = []
        lib.fast_ane_probe_compat_json.restype = ctypes.c_void_p
        lib.fast_ane_free_string.argtypes = [ctypes.c_void_p]
        lib.fast_ane_free_string.restype = None

        lib.fast_ane_context_create.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_context_create.restype = ctypes.c_int32
        lib.fast_ane_context_destroy.argtypes = [ctypes.c_void_p]
        lib.fast_ane_context_ane_available.argtypes = [ctypes.c_void_p]
        lib.fast_ane_context_ane_available.restype = ctypes.c_int32
        lib.fast_ane_context_shared_event_available.argtypes = [ctypes.c_void_p]
        lib.fast_ane_context_shared_event_available.restype = ctypes.c_int32
        lib.fast_ane_context_iosurface_metal_available.argtypes = [ctypes.c_void_p]
        lib.fast_ane_context_iosurface_metal_available.restype = ctypes.c_int32

        lib.fast_ane_surface_create.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_surface_create.restype = ctypes.c_int32
        lib.fast_ane_surface_destroy.argtypes = [ctypes.c_void_p]
        lib.fast_ane_surface_width.argtypes = [ctypes.c_void_p]
        lib.fast_ane_surface_width.restype = ctypes.c_uint32
        lib.fast_ane_surface_height.argtypes = [ctypes.c_void_p]
        lib.fast_ane_surface_height.restype = ctypes.c_uint32
        lib.fast_ane_surface_row_bytes.argtypes = [ctypes.c_void_p]
        lib.fast_ane_surface_row_bytes.restype = ctypes.c_uint32
        lib.fast_ane_surface_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        lib.fast_ane_surface_write.restype = ctypes.c_int32
        lib.fast_ane_surface_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        lib.fast_ane_surface_read.restype = ctypes.c_int32

        lib.fast_ane_program_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_program_create.restype = ctypes.c_int32
        lib.fast_ane_program_create_from_dir.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_program_create_from_dir.restype = ctypes.c_int32
        lib.fast_ane_program_destroy.argtypes = [ctypes.c_void_p]

        lib.fast_ane_program_eval.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_program_eval.restype = ctypes.c_int32

        lib.fast_ane_handoff_roundtrip.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(_FastAneHandoffMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_handoff_roundtrip.restype = ctypes.c_int32

        lib.fast_ane_ffn_wave_begin.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_ffn_wave_begin.restype = ctypes.c_int32
        lib.fast_ane_ffn_wave_finish.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_FastAneWaveMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.fast_ane_ffn_wave_finish.restype = ctypes.c_int32
        lib.fast_ane_ffn_wave_destroy.argtypes = [ctypes.c_void_p]
        lib.fast_ane_ffn_wave_destroy.restype = None

    def probe_compat(self) -> dict[str, Any]:
        ptr = self._lib.fast_ane_probe_compat_json()
        try:
            raw = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
            return json.loads(raw)
        finally:
            self._lib.fast_ane_free_string(ptr)

    def create_context(self, *, allow_unavailable: bool = True) -> "AneContext":
        return AneContext(self, allow_unavailable=allow_unavailable)


class AneSurface:
    def __init__(self, lib: AneHandoffLibrary, width: int, height: int, *, dtype: np.dtype = np.dtype(np.float16)) -> None:
        self._lib = lib
        self._ptr = ctypes.c_void_p()
        dt = np.dtype(dtype)
        if dt == np.float16:
            bpe = 2
            pix = PIXEL_FORMAT_L00H
        elif dt in (np.int8, np.uint8):
            bpe = 1
            pix = PIXEL_FORMAT_L008
        else:
            raise ValueError(f"Unsupported ANE surface dtype {dt}")
        self.width = int(width)
        self.height = int(height)
        self.dtype = dt
        self.bytes_per_elem = bpe
        err = ctypes.create_string_buffer(2048)
        st = lib._lib.fast_ane_surface_create(
            self.width, self.height, bpe, pix, ctypes.byref(self._ptr), err, len(err)
        )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))

    def write(self, arr: np.ndarray) -> None:
        c_arr = np.ascontiguousarray(arr, dtype=self.dtype)
        if c_arr.shape != (self.height, self.width):
            raise ValueError(f"Expected shape {(self.height, self.width)}, got {c_arr.shape}")
        st = self._lib._lib.fast_ane_surface_write(
            self._ptr, c_arr.ctypes.data, self.width * self.bytes_per_elem, self.height
        )
        if st != 0:
            raise AneNativeError(st, "fast_ane_surface_write failed")

    def read(self) -> np.ndarray:
        out = np.empty((self.height, self.width), dtype=self.dtype)
        st = self._lib._lib.fast_ane_surface_read(
            self._ptr, out.ctypes.data, self.width * self.bytes_per_elem, self.height
        )
        if st != 0:
            raise AneNativeError(st, "fast_ane_surface_read failed")
        return out

    def close(self) -> None:
        if self._ptr:
            self._lib._lib.fast_ane_surface_destroy(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> "AneSurface":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class AneProgram:
    def __init__(
        self,
        ctx: "AneContext",
        mil_text: str | None = None,
        *,
        bundle_dir: str | Path | None = None,
        weight_blob: bytes | None = None,
        weight_dtype_code: int = BLOB_DTYPE_FP16,
        preformatted_blob: bool = False,
        qos: int = 21,
    ) -> None:
        self._ctx = ctx
        self._lib = ctx._lib
        self._ptr = ctypes.c_void_p()
        err = ctypes.create_string_buffer(4096)
        if bundle_dir is not None:
            dir_bytes = str(bundle_dir).encode("utf-8")
            st = self._lib._lib.fast_ane_program_create_from_dir(
                ctx._ptr, dir_bytes, qos, ctypes.byref(self._ptr), err, len(err)
            )
        else:
            if mil_text is None:
                raise ValueError("Either mil_text or bundle_dir must be provided")
            mil_bytes = mil_text.encode("utf-8")
            buf = (ctypes.c_ubyte * len(weight_blob)).from_buffer_copy(weight_blob) if weight_blob else None
            st = self._lib._lib.fast_ane_program_create(
                ctx._ptr,
                mil_bytes,
                len(mil_bytes),
                ctypes.cast(buf, ctypes.c_void_p) if buf is not None else None,
                len(weight_blob) if weight_blob else 0,
                weight_dtype_code,
                1 if preformatted_blob else 0,
                qos,
                ctypes.byref(self._ptr),
                err,
                len(err),
            )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))

    def eval(self, inputs: Sequence[AneSurface], outputs: Sequence[AneSurface]) -> float:
        in_arr = (ctypes.c_void_p * len(inputs))(*[s._ptr for s in inputs])
        out_arr = (ctypes.c_void_p * len(outputs))(*[s._ptr for s in outputs])
        ms = ctypes.c_double(0.0)
        err = ctypes.create_string_buffer(2048)
        st = self._lib._lib.fast_ane_program_eval(
            self._ctx._ptr,
            self._ptr,
            in_arr,
            len(inputs),
            out_arr,
            len(outputs),
            ctypes.byref(ms),
            err,
            len(err),
        )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))
        return float(ms.value)

    def close(self) -> None:
        if self._ptr:
            self._lib._lib.fast_ane_program_destroy(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> "AneProgram":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class AneContext:
    def __init__(self, lib: AneHandoffLibrary, *, allow_unavailable: bool = True) -> None:
        self._lib = lib
        self._ptr = ctypes.c_void_p()
        err = ctypes.create_string_buffer(2048)
        st = lib._lib.fast_ane_context_create(1 if allow_unavailable else 0, ctypes.byref(self._ptr), err, len(err))
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))

    @property
    def ane_available(self) -> bool:
        return bool(self._ptr and self._lib._lib.fast_ane_context_ane_available(self._ptr))

    @property
    def shared_event_available(self) -> bool:
        return bool(self._ptr and self._lib._lib.fast_ane_context_shared_event_available(self._ptr))

    @property
    def iosurface_metal_available(self) -> bool:
        return bool(self._ptr and self._lib._lib.fast_ane_context_iosurface_metal_available(self._ptr))

    def surface(self, width: int, height: int, *, dtype: np.dtype = np.dtype(np.float16)) -> AneSurface:
        return AneSurface(self._lib, width, height, dtype=dtype)

    def compile_program(
        self,
        mil_text: str,
        *,
        weight_blob: bytes | None = None,
        weight_dtype_code: int = BLOB_DTYPE_FP16,
        preformatted_blob: bool = False,
        qos: int = 21,
    ) -> AneProgram:
        return AneProgram(
            self,
            mil_text,
            weight_blob=weight_blob,
            weight_dtype_code=weight_dtype_code,
            preformatted_blob=preformatted_blob,
            qos=qos,
        )

    def load_program_dir(self, bundle_dir: str | Path, *, qos: int = 21) -> AneProgram:
        return AneProgram(self, bundle_dir=bundle_dir, qos=qos)

    def begin_ffn_wave(
        self,
        seg_progs: Sequence[AneProgram],
        seg_out_surfs: Sequence[AneSurface],
        surf_in: AneSurface,
        host_in_fp16: np.ndarray,
        signs_fp16: np.ndarray,
        *,
        quantize_a8: bool = True,
        use_shared_event: bool = True,
        timeout_ms: int = 5000,
    ) -> "AneWave":
        return AneWave(
            self,
            seg_progs=seg_progs,
            seg_out_surfs=seg_out_surfs,
            surf_in=surf_in,
            host_in_fp16=host_in_fp16,
            signs_fp16=signs_fp16,
            quantize_a8=quantize_a8,
            use_shared_event=use_shared_event,
            timeout_ms=timeout_ms,
        )

    def handoff_roundtrip(
        self,
        prog: AneProgram,
        surf_in: AneSurface,
        surf_out: AneSurface,
        host_in_fp16: np.ndarray,
        *,
        use_shared_event: bool = True,
        timeout_ms: int = 5000,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        in_arr = np.ascontiguousarray(host_in_fp16, dtype=np.float16)
        out_arr = np.empty_like(in_arr)
        metrics = _FastAneHandoffMetrics()
        err = ctypes.create_string_buffer(2048)
        st = self._lib._lib.fast_ane_handoff_roundtrip(
            self._ptr,
            prog._ptr,
            surf_in._ptr,
            surf_out._ptr,
            in_arr.ctypes.data,
            out_arr.ctypes.data,
            in_arr.size,
            0 if use_shared_event else 1,
            timeout_ms,
            ctypes.byref(metrics),
            err,
            len(err),
        )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))
        return out_arr, {
            "gpu_pre_ms": metrics.gpu_pre_ms,
            "ane_submit_ms": metrics.ane_submit_ms,
            "gpu_post_ms": metrics.gpu_post_and_total_ms,
            "total_ms": metrics.total_ms,
            "used_shared_event": bool(metrics.used_shared_event),
            "used_iosurface_metal_buffer": bool(metrics.used_iosurface_metal_buffer),
            "final_shared_event_value": int(metrics.final_shared_event_value),
        }

    def close(self) -> None:
        if self._ptr:
            self._lib._lib.fast_ane_context_destroy(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> "AneContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class AneWave:
    """In-flight Metal rotate_pack -> ANE segment(s) -> Metal join pipeline."""

    def __init__(
        self,
        ctx: AneContext,
        *,
        seg_progs: Sequence[AneProgram],
        seg_out_surfs: Sequence[AneSurface],
        surf_in: AneSurface,
        host_in_fp16: np.ndarray,
        signs_fp16: np.ndarray,
        quantize_a8: bool = True,
        use_shared_event: bool = True,
        timeout_ms: int = 5000,
    ) -> None:
        self._ctx = ctx
        self._lib = ctx._lib
        self._ptr = ctypes.c_void_p()
        if len(seg_progs) == 0 or len(seg_progs) != len(seg_out_surfs):
            raise ValueError("seg_progs and seg_out_surfs must be non-empty and equal length")
        in_arr = np.ascontiguousarray(host_in_fp16, dtype=np.float16)
        if in_arr.ndim != 2:
            raise ValueError(f"Expected 2D [M, H] input, got shape {in_arr.shape}")
        tokens, hidden = in_arr.shape
        signs_arr = np.ascontiguousarray(signs_fp16, dtype=np.float16)
        if signs_arr.shape != (hidden,):
            raise ValueError(f"Expected signs shape {(hidden,)}, got {signs_arr.shape}")

        self.tokens = tokens
        self.hidden_size = hidden
        # Keep numpy buffers alive until native wave begin returns (buffers copied to MTLBuffer inside begin)
        progs_c = (ctypes.c_void_p * len(seg_progs))(*[p._ptr for p in seg_progs])
        outs_c = (ctypes.c_void_p * len(seg_out_surfs))(*[s._ptr for s in seg_out_surfs])
        err = ctypes.create_string_buffer(2048)
        st = self._lib._lib.fast_ane_ffn_wave_begin(
            ctx._ptr,
            progs_c,
            outs_c,
            len(seg_progs),
            surf_in._ptr,
            in_arr.ctypes.data,
            signs_arr.ctypes.data,
            tokens,
            hidden,
            1 if quantize_a8 else 0,
            1 if use_shared_event else 0,
            timeout_ms,
            ctypes.byref(self._ptr),
            err,
            len(err),
        )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))

    def finish(self, opt_gpu_tail_fp16: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        out = np.empty((self.tokens, self.hidden_size), dtype=np.float16)
        tail_ptr = None
        if opt_gpu_tail_fp16 is not None:
            tail_arr = np.ascontiguousarray(opt_gpu_tail_fp16, dtype=np.float16)
            if tail_arr.shape != (self.tokens, self.hidden_size):
                raise ValueError(f"Expected tail shape {(self.tokens, self.hidden_size)}, got {tail_arr.shape}")
            tail_ptr = tail_arr.ctypes.data
        metrics = _FastAneWaveMetrics()
        err = ctypes.create_string_buffer(2048)
        st = self._lib._lib.fast_ane_ffn_wave_finish(
            self._ptr, tail_ptr, out.ctypes.data, ctypes.byref(metrics), err, len(err)
        )
        if st != 0:
            raise AneNativeError(st, err.value.decode("utf-8", errors="replace"))
        return out, {
            "gpu_submit_ms": metrics.gpu_submit_ms,
            "ane_submit_ms": metrics.ane_submit_ms,
            "finish_wait_ms": metrics.finish_wait_ms,
            "total_wall_ms": metrics.total_wall_ms,
            "used_shared_event": bool(metrics.used_shared_event),
        }

    def close(self) -> None:
        if self._ptr:
            self._lib._lib.fast_ane_ffn_wave_destroy(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> "AneWave":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
