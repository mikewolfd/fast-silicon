"""Shared embedding helpers: Matryoshka truncation, cache limits, mapped tables, and tile slicing."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import struct
from typing import Any, Callable

import numpy as np

_SAFETENSORS_MMAP_DTYPES: dict[str, np.dtype[Any]] = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}

_GATHER_OUT_DTYPES: dict[str, np.dtype[Any]] = {
    "float16": np.dtype(np.float16),
    "fp16": np.dtype(np.float16),
    "f16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "fp32": np.dtype(np.float32),
    "f32": np.dtype(np.float32),
}


def set_mlx_cache_limit(limit_mb: int = 512) -> int | None:
    """Set MLX's free buffer cache limit in MiB when `mlx.core` is importable.

    Returns the applied byte limit (`0` when `limit_mb <= 0` resets via
    `mx.set_cache_limit(0)`), or `None` if `mlx.core` is unavailable.
    """
    if isinstance(limit_mb, bool) or not isinstance(limit_mb, (int, np.integer)):
        raise ValueError(f"`limit_mb` must be an integer, got {limit_mb!r}")
    mb = int(limit_mb)
    try:
        import mlx.core as mx
    except ImportError:
        return None
    set_limit = getattr(mx, "set_cache_limit", None)
    if not callable(set_limit):
        return None
    if mb <= 0:
        set_limit(0)
        return 0
    limit_bytes = mb * 1024 * 1024
    set_limit(limit_bytes)
    return limit_bytes


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"Truncated safetensors file: {path}")
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        if header_len == 0 or 8 + header_len > file_size:
            raise ValueError(f"Invalid safetensors header length ({header_len}) in {path}")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"Truncated safetensors JSON header in {path}")
    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return header_len, header


class MappedEmbedding:
    """Pure-NumPy read-only `np.memmap` gather table over `.safetensors` (`BF16`/`F16`/`F32`)."""

    def __init__(
        self,
        safetensors_path: str | Path,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> None:
        self.path = Path(safetensors_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Safetensors file not found: {self.path}")
        self.tensor_name = tensor_name

        header_len, header = _read_safetensors_header(self.path)
        entry = header.get(tensor_name)
        if entry is None or tensor_name == "__metadata__" or not isinstance(entry, dict):
            raise KeyError(f"Tensor {tensor_name!r} not found in {self.path}")

        dtype_name = entry.get("dtype")
        if dtype_name not in _SAFETENSORS_MMAP_DTYPES:
            raise ValueError(
                f"Tensor {tensor_name!r} has unsupported dtype {dtype_name!r}; "
                f"expected one of {sorted(_SAFETENSORS_MMAP_DTYPES)}"
            )
        mm_dtype = _SAFETENSORS_MMAP_DTYPES[dtype_name]

        shape_raw = entry.get("shape")
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise ValueError(f"Expected 2D embedding tensor for {tensor_name!r}, got shape={shape_raw!r}")
        vocab_size, hidden_size = int(shape_raw[0]), int(shape_raw[1])
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError(f"Invalid embedding shape {(vocab_size, hidden_size)} for {tensor_name!r}")

        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid data_offsets {offsets!r} for {tensor_name!r}")
        begin, end = int(offsets[0]), int(offsets[1])
        expected_bytes = vocab_size * hidden_size * mm_dtype.itemsize
        file_size = self.path.stat().st_size
        abs_begin = 8 + header_len + begin
        abs_end = 8 + header_len + end
        if begin < 0 or (end - begin) != expected_bytes or abs_end > file_size:
            raise ValueError(
                f"Corrupt data_offsets {offsets!r} for {tensor_name!r} "
                f"(expected {expected_bytes} bytes within file size {file_size})"
            )

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dtype = str(dtype_name)
        self._mm: np.memmap | None = np.memmap(
            self.path,
            dtype=mm_dtype,
            mode="r",
            offset=abs_begin,
            shape=(vocab_size, hidden_size),
        )

    @classmethod
    def from_checkpoint(
        cls,
        path_or_dir: str | Path,
        tensor_name: str = "model.embed_tokens.weight",
    ) -> "MappedEmbedding":
        """Open a mapped embedding table from a `.safetensors` file or checkpoint directory."""
        target = Path(path_or_dir)
        if not target.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {target}")
        if target.is_file():
            return cls(target, tensor_name=tensor_name)
        if not target.is_dir():
            raise FileNotFoundError(f"Checkpoint path is neither file nor directory: {target}")

        index_path = target / "model.safetensors.index.json"
        if index_path.is_file():
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            if tensor_name not in weight_map:
                raise KeyError(f"Tensor {tensor_name!r} not found in {index_path}")
            return cls(target / weight_map[tensor_name], tensor_name=tensor_name)

        single_path = target / "model.safetensors"
        if single_path.is_file():
            return cls(single_path, tensor_name=tensor_name)

        shards = sorted(target.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No *.safetensors or model.safetensors.index.json found in {target}")
        for shard in shards:
            _, hdr = _read_safetensors_header(shard)
            if tensor_name in hdr and tensor_name != "__metadata__":
                return cls(shard, tensor_name=tensor_name)
        raise KeyError(f"Tensor {tensor_name!r} not found in any *.safetensors shard under {target}")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self) -> str:
        return self._dtype

    def gather(self, token_ids: np.ndarray, *, out_dtype: str = "float16") -> np.ndarray:
        """Gather embedding rows on CPU without allocating the full table on GPU."""
        if self._mm is None:
            raise ValueError("I/O operation on closed MappedEmbedding.")
        key = out_dtype.lower() if isinstance(out_dtype, str) else out_dtype
        if isinstance(key, np.dtype):
            if key == np.dtype(np.float16):
                target_dtype = np.dtype(np.float16)
            elif key == np.dtype(np.float32):
                target_dtype = np.dtype(np.float32)
            else:
                raise ValueError(f"Unsupported out_dtype {out_dtype!r}; expected 'float16' or 'float32'")
        elif key in _GATHER_OUT_DTYPES:
            target_dtype = _GATHER_OUT_DTYPES[key]
        else:
            raise ValueError(f"Unsupported out_dtype {out_dtype!r}; expected 'float16' or 'float32'")

        ids = np.asarray(token_ids)
        if ids.size == 0:
            return np.empty((*ids.shape, self._hidden_size), dtype=target_dtype)
        if ids.dtype == np.bool_ or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"`token_ids` must have an integer dtype, got {ids.dtype}")
        min_id = int(ids.min())
        max_id = int(ids.max())
        if min_id < 0 or max_id >= self._vocab_size:
            raise IndexError(
                f"`token_ids` out of range [0, {self._vocab_size}): min={min_id}, max={max_id}"
            )

        flat_ids = np.asarray(ids.reshape(-1), dtype=np.intp)
        raw = np.asarray(self._mm[flat_ids])
        if self._dtype == "BF16":
            # Shift BF16 upper 16 bits into IEEE-754 binary32 before converting to out_dtype.
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            vals = f32.astype(target_dtype, copy=False)
        else:
            vals = raw.astype(target_dtype, copy=False)
        return np.ascontiguousarray(vals.reshape(*ids.shape, self._hidden_size), dtype=target_dtype)

    def close(self) -> None:
        """Close the underlying `np.memmap` handle."""
        mm = self._mm
        self._mm = None
        if mm is not None:
            backing = getattr(mm, "_mmap", None)
            if backing is not None and not getattr(backing, "closed", False):
                backing.close()

    def __enter__(self) -> "MappedEmbedding":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._mm is None else "open"
        return (
            f"MappedEmbedding(path={str(self.path)!r}, tensor_name={self.tensor_name!r}, "
            f"shape=({self._vocab_size}, {self._hidden_size}), dtype={self._dtype!r}, state={state!r})"
        )


def tile_aligned_slices(seq_len: int, slice_len: int = 512, min_tail: int = 64) -> list[tuple[int, int]]:
    """Partition `[0, seq_len)` into half-open `(start, end)` chunks, merging short final tails.

    When there are at least two slices and the trailing chunk is shorter than
    `min_tail`, it is merged into the penultimate slice so `mx.fast.sdpa` stays on
    the identical Metal matmul tile path.
    """
    for name, val in (("seq_len", seq_len), ("slice_len", slice_len), ("min_tail", min_tail)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"`{name}` must be an integer, got {val!r}")
    n = int(seq_len)
    step = int(slice_len)
    tail_min = int(min_tail)
    if n < 0:
        raise ValueError(f"`seq_len` must be >= 0, got {n}")
    if step <= 0:
        raise ValueError(f"`slice_len` must be > 0, got {step}")
    if tail_min < 0:
        raise ValueError(f"`min_tail` must be >= 0, got {tail_min}")
    if n == 0:
        return []
    slices = [(start, min(start + step, n)) for start in range(0, n, step)]
    if len(slices) >= 2 and (slices[-1][1] - slices[-1][0]) < tail_min:
        prev_start, _ = slices[-2]
        _, last_end = slices.pop()
        slices[-1] = (prev_start, last_end)
    return slices


def resolve_embedding_dim(
    requested: Any,
    *,
    max_dim: int,
    default_dim: int | None = None,
    supports_dimensions: bool = True,
) -> int:
    """Validate and resolve a requested Matryoshka output dimension."""
    eff_max = int(max_dim)
    eff_default = eff_max if default_dim is None else int(default_dim)
    if requested is None:
        return eff_default
    if isinstance(requested, bool) or not isinstance(requested, (int, np.integer)):
        raise ValueError(f"`dimensions` must be a positive integer <= {eff_max}, got {requested!r}")
    val = int(requested)
    if val <= 0 or val > eff_max:
        raise ValueError(f"`dimensions` must be between 1 and {eff_max}, got {val}")
    if not supports_dimensions and val != eff_default:
        raise ValueError(
            f"This embedding engine does not support Matryoshka dimension selection; only {eff_default} is supported"
        )
    return val


def truncate_and_normalize(vecs: np.ndarray, dim: int | None = None) -> np.ndarray:
    """Slice embeddings to `[:dim]` and re-normalize L2 across the trailing axis."""
    arr = np.asarray(vecs, dtype=np.float32)
    if dim is None:
        return np.ascontiguousarray(arr, dtype=np.float32)
    eff_dim = int(dim)
    if eff_dim <= 0 or eff_dim > arr.shape[-1]:
        raise ValueError(f"Cannot truncate vectors of width {arr.shape[-1]} to dimensions={eff_dim}")
    if eff_dim == arr.shape[-1]:
        return np.ascontiguousarray(arr, dtype=np.float32)
    sub = np.ascontiguousarray(arr[..., :eff_dim], dtype=np.float32)
    if sub.shape[0] == 0:
        return sub
    norms = np.linalg.norm(sub, axis=-1, keepdims=True)
    sub /= np.maximum(norms, 1e-12)
    return sub


def accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """Return True if `fn` accepts keyword argument `name` (explicitly or via `**kwargs`)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    for p in params:
        if p.name == name and p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return True
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


__all__ = [
    "MappedEmbedding",
    "accepts_kwarg",
    "resolve_embedding_dim",
    "set_mlx_cache_limit",
    "tile_aligned_slices",
    "truncate_and_normalize",
]
