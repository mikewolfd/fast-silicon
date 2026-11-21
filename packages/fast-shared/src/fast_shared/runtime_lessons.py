"""Cross-silicon Apple M4 (Apple9) & M5+ (Apple10/NAX) runtime lessons & dynamic policy engine.

Codifies empirically validated production invariants shared across Apple Silicon inference,
training, and RLVR workflows (100% pure Python + NumPy, zero GPU/ANE execution on import):

1. **Auto-RAM & Working-Set Sizing (Lesson 1 — ``recommend_silicon_budget``):**
   Enforces the strict ``<= 80%`` unified-memory wired ceiling and adapts MLX memory/cache limits,
   batch size, sequence pack length, and gradient accumulation across ``16 GB``, ``24 GB`` Base M5,
   and ``>= 36 GB`` M4 Pro / Max hosts so training and extraction avoid the macOS UMA swap/OOM cliff.
2. **Deterministic FFD Sequence Packing & Shared-System Deduplication (Lesson 6 — ``pack_sequences_ffd`` & ``pack_encoder_batches``):**
   Packs variable-length causal or bidirectional token sequences into fixed-capacity bins via
   First-Fit-Decreasing (FFD) with per-segment RoPE ``position_ids`` resets, ``segment_ids``,
   NaN-free block-diagonal masks, and optional shared-system-prompt prefix deduplication.
3. **Batch-Regime Speculative Policy Guard (Lesson 7 — ``select_speculative_policy``):**
   Encodes chipset-aware Apple9/Apple10 NAX thresholds: enables MTP at ``B == 1`` (and ``2 <= B <= 4``
   with ``draft_k=2`` on ``Apple9`` only), disables MTP on ``Apple10+ NAX`` for ``B >= 2`` and across
   all silicon for ``B >= 5`` (where ``B=8`` regresses to ``0.56x``), and enables prefix-tree KV +
   tree-mask verification when ``shared_prefix_tokens >= 32`` and ``num_candidates >= 2``.
4. **Disaggregated GPU Actor + ANE Critic Coordinator (``DisaggregatedCriticBudget`` / ``validate_grpo_ane_offload``):**
   Verifies ``0.0 GB`` critic Metal allocation, enforces ``actor_gpu_gb + ane_mapped_gb <= 0.80 * host_mem_gb``,
   computes remaining GPU actor headroom, and builds the pipelined ANE critic microbatch schedule.
5. **Unified Runtime Policy (``DynamicSiliconPolicy`` / ``resolve_dynamic_execution_plan``):**
   Executes all lessons dynamically both per-host on init (via ``probe_host_hardware()`` by default)
   and per-call on ``(batch_size, seq_len, shared_prefix_tokens, num_candidates, temperature, mtp_mode)``,
   combining chipset (``Apple9`` vs ``Apple10+ NAX``), RAM tier (``<= 24 GB`` vs ``>= 36 GB``),
   OS gates (``< 26.2``, ``>= 26.4``, ``>= 27.2.1``), lazy ``mtp_mode="auto"`` drafter gating,
   and promotion-gated ANE backend selection (``"coreml_mil"`` vs ``"coreai"``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import math
import platform
import re
import subprocess
import threading
from typing import Any, Literal

import numpy as np

from .hw_check import parse_version_tuple, probe_host_hardware, supports_apple10_nax

ProfileMode = Literal["backfill", "interactive"]
ThermalLevel = Literal["nominal", "fair", "serious", "critical"]

WIRED_MEMORY_CEILING_FRACTION: float = 0.80
MIN_SHARED_PREFIX_TOKENS: int = 32
MASK_NEG_VAL_FP16: float = -65504.0
DEFAULT_MLX_CACHE_LIMIT_MB: int = 256
VALID_WORKLOADS: tuple[str, ...] = ("extraction", "qlora", "encoder_train", "grpo")
VALID_MTP_MODES: tuple[str, ...] = ("auto", "on", "off")
VALID_PROFILES: tuple[str, ...] = ("backfill", "interactive")
APPLE10_FUSED_SWA_SDPA_SEQ_LENS: tuple[int, ...] = (384, 512)

_TIGHT_CAP_THRESHOLD_GB: float = 15.0
_HIGH_CAP_THRESHOLD_GB: float = 26.0
_GRPO_ANE_RESERVE_GB: float = 1.2
_MTP_DRAFTER_ESTIMATED_MB: float = 150.0


class _ImplicitBackfillProfile(str):
    """Sentinel default `'backfill'` string distinguishing omitted vs explicit `profile` kwargs."""


_DEFAULT_PROFILE: Any = _ImplicitBackfillProfile("backfill")


def _normalize_profile(profile: str | None, *, default: str = "backfill") -> str:
    raw = default if profile is None else str(profile).strip().lower()
    if raw not in VALID_PROFILES:
        raise ValueError(f"Unsupported profile={profile!r}; choose from {VALID_PROFILES}")
    return raw


def _validate_positive_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return val


def _validate_non_negative_finite(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return val


def _floor_4dp(value: float) -> float:
    """Round downward to 4 decimal places so ceilings never exceed exact mathematical bounds."""
    return math.floor((float(value) + 1e-12) * 10000.0) / 10000.0


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _version_ge(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    if not lhs or not rhs:
        return False
    n = max(len(lhs), len(rhs))
    a = lhs + (0,) * (n - len(lhs))
    b = rhs + (0,) * (n - len(rhs))
    return a >= b


# ---------------------------------------------------------------------------
# 1. Lesson 1: Hardware-Aware Working-Set & Batch Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiliconBudget:
    """Hardware-aware unified-memory and batching budget for an Apple Silicon workload."""

    workload: str
    host_mem_gb: float
    wired_ceiling_gb: float
    effective_cap_gb: float
    memory_limit_gb: float
    cache_limit_gb: float
    batch_size: int
    seq_pack_len: int
    grad_accum: int
    ckpt_tokens: int
    tier: str
    notes: str

    @property
    def pack_len(self) -> int:
        """Alias for :attr:`seq_pack_len`."""
        return self.seq_pack_len

    def to_dict(self) -> dict[str, Any]:
        """Serialize budget parameters to a JSON-friendly dictionary."""
        return {
            "workload": self.workload,
            "host_mem_gb": self.host_mem_gb,
            "wired_ceiling_gb": self.wired_ceiling_gb,
            "effective_cap_gb": self.effective_cap_gb,
            "memory_limit_gb": self.memory_limit_gb,
            "cache_limit_gb": self.cache_limit_gb,
            "batch_size": self.batch_size,
            "seq_pack_len": self.seq_pack_len,
            "grad_accum": self.grad_accum,
            "ckpt_tokens": self.ckpt_tokens,
            "tier": self.tier,
            "notes": self.notes,
        }


def recommend_silicon_budget(
    host_mem_gb: float,
    metal_working_set_gb: float | None = None,
    workload: str = "extraction",
    *,
    wired_ceiling_fraction: float = WIRED_MEMORY_CEILING_FRACTION,
) -> SiliconBudget:
    """Recommend wired-safe MLX memory/cache limits, batch size, and pack length.

    Strictly enforces the ``<= 80%`` unified-memory wired ceiling:
    ``effective_cap_gb = min(0.80 * host_mem_gb, metal_working_set_gb or 0.80 * host_mem_gb)``.

    Tier targets:
      - ``<= 16 GB`` (``effective_cap_gb < 15.0 GB``, tight): conservative limits (``extraction B=64``,
        ``qlora pack_len=512, grad_accum=4``).
      - ``24 GB`` Base M5 (``effective_cap_gb = 19.2 GB``, standard): ``memory_limit_gb=19.2``,
        ``cache_limit_gb=1.5``, ``extraction B=128``, ``qlora pack_len=1024, grad_accum=2``.
      - ``>= 36 GB`` M4 Pro / Max (``effective_cap_gb >= 26.0 GB``, high): ``memory_limit_gb=24.0``,
        ``cache_limit_gb=2.0``, ``extraction B=256``, ``qlora pack_len=2048, grad_accum=1``.

    Args:
        host_mem_gb: Total unified host RAM in GB (e.g. ``16.0``, ``24.0``, ``36.0``, ``48.0``).
        metal_working_set_gb: Optional Metal ``maxRecommendedWorkingSetSize`` in GB.
            ``None`` or ``0.0`` falls back to ``wired_ceiling_fraction * host_mem_gb``.
        workload: One of ``"extraction"``, ``"qlora"``, ``"encoder_train"``, or ``"grpo"``.
        wired_ceiling_fraction: Maximum fraction of unified RAM allowed to be wired
            (must satisfy ``0 < wired_ceiling_fraction <= 0.80``).

    Returns:
        Frozen :class:`SiliconBudget` with wired-safe limits and workload-tuned dimensions.
    """
    host_gb = _validate_positive_finite("host_mem_gb", host_mem_gb)
    frac = _validate_positive_finite("wired_ceiling_fraction", wired_ceiling_fraction)
    if frac > WIRED_MEMORY_CEILING_FRACTION + 1e-12:
        raise ValueError(
            f"wired_ceiling_fraction must be <= {WIRED_MEMORY_CEILING_FRACTION}, got {wired_ceiling_fraction!r}"
        )

    wl = str(workload).strip().lower()
    if wl not in VALID_WORKLOADS:
        raise ValueError(f"Unsupported workload={workload!r}; choose from {VALID_WORKLOADS}")

    wired_ceiling_gb = _floor_4dp(frac * host_gb)

    if metal_working_set_gb is None:
        effective_cap_gb = wired_ceiling_gb
    else:
        metal_gb = _validate_non_negative_finite("metal_working_set_gb", metal_working_set_gb)
        if metal_gb == 0.0:
            effective_cap_gb = wired_ceiling_gb
        else:
            effective_cap_gb = _floor_4dp(min(wired_ceiling_gb, metal_gb))

    if effective_cap_gb < _TIGHT_CAP_THRESHOLD_GB:
        tier = "tight"
    elif effective_cap_gb < _HIGH_CAP_THRESHOLD_GB:
        tier = "standard"
    else:
        tier = "high"

    if tier == "tight":
        base_mem_limit_gb = effective_cap_gb
        raw_cache_gb = 1.5
        if wl == "extraction":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 64, 1024, 1, 0
        elif wl == "qlora":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 1, 512, 4, 256
        elif wl == "encoder_train":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 8, 512, 2, 512
        else:  # grpo
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 4, 512, 4, 256
    elif tier == "standard":
        base_mem_limit_gb = _floor_4dp(min(22.0 if host_gb > 24.0 else 19.2, effective_cap_gb))
        raw_cache_gb = 1.5
        if wl == "extraction":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 128, 2048, 1, 0
        elif wl == "qlora":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 1, 1024, 2, 512
        elif wl == "encoder_train":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 16, 512, 1, 0
        else:  # grpo
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 8, 1024, 2, 512
    else:  # high (>= 36 GB / effective_cap_gb >= 26.0 GB)
        base_mem_limit_gb = _floor_4dp(min(24.0, effective_cap_gb))
        raw_cache_gb = 2.0
        if wl == "extraction":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 256, 2048, 1, 0
        elif wl == "qlora":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 1, 2048, 1, 4096
        elif wl == "encoder_train":
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 32, 512, 1, 0
        else:  # grpo
            batch_size, seq_pack_len, grad_accum, ckpt_tokens = 16, 2048, 1, 2048

    # For disaggregated GRPO (GPU actor + ANE critic), ensure >= 1.2 GB remains below effective_cap_gb.
    if wl == "grpo":
        ane_reserve = min(_GRPO_ANE_RESERVE_GB, _floor_4dp(0.10 * effective_cap_gb))
        memory_limit_gb = _floor_4dp(max(0.25, min(base_mem_limit_gb, effective_cap_gb - ane_reserve)))
    else:
        memory_limit_gb = base_mem_limit_gb

    cache_limit_gb = _floor_4dp(min(raw_cache_gb, max(0.0625, 0.10 * memory_limit_gb)))

    notes = (
        f"tier={tier} workload={wl}: effective_cap={effective_cap_gb:.2f} GB "
        f"(<= {int(round(frac * 100))}% of {host_gb:.1f} GB UMA = {wired_ceiling_gb:.2f} GB), "
        f"mem_limit={memory_limit_gb:.2f} GB, cache={cache_limit_gb:.3f} GB, "
        f"batch={batch_size}, pack_len={seq_pack_len}, grad_accum={grad_accum}."
    )
    return SiliconBudget(
        workload=wl,
        host_mem_gb=host_gb,
        wired_ceiling_gb=wired_ceiling_gb,
        effective_cap_gb=effective_cap_gb,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
        batch_size=batch_size,
        seq_pack_len=seq_pack_len,
        grad_accum=grad_accum,
        ckpt_tokens=ckpt_tokens,
        tier=tier,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 2. Lesson 6: Deterministic FFD Sequence Packing & Shared-System Deduplication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedSegmentSpan:
    """Location and original provenance of one packed sequence segment inside a bin."""

    orig_index: int
    start: int
    end: int
    position_offset: int
    length: int


@dataclass(frozen=True)
class PackedSequenceBin:
    """One fixed-capacity FFD packed token bin with segment/position and mask metadata."""

    input_ids: np.ndarray
    position_ids: np.ndarray
    segment_ids: np.ndarray
    item_indices: tuple[int, ...]
    segment_spans: tuple[PackedSegmentSpan, ...]
    real_tokens: int
    bin_len: int
    shared_prefix_len: int = 0
    attention_mask: np.ndarray | None = None

    @property
    def num_examples(self) -> int:
        return len(self.item_indices)

    @property
    def utilization(self) -> float:
        return float(self.real_tokens) / float(max(1, self.bin_len))


@dataclass(frozen=True)
class PackedSequenceResult:
    """Complete deterministic FFD packing plan with optional shared-system prefix deduplication."""

    bins: tuple[PackedSequenceBin, ...]
    max_seq_len: int
    shared_prefix_ids: np.ndarray
    shared_prefix_len: int
    num_sequences: int
    num_bins: int
    logical_tokens: int
    packed_real_tokens: int
    packed_total_tokens: int
    effective_compute_tokens: int
    deduplicated_tokens: int
    packing_efficiency: float
    pad_savings_ratio: float


@dataclass(frozen=True)
class PackedEncoderBatch:
    """Stacked 2D bidirectional FFD packed batch ready for encoder fine-tuning or scoring."""

    input_ids: np.ndarray
    position_ids: np.ndarray
    segment_ids: np.ndarray
    cls_positions: np.ndarray
    item_indices: tuple[int, ...]
    token_spans: tuple[tuple[int, int, int], ...]
    attention_mask: np.ndarray | None
    num_items: int
    num_bins: int
    bin_width: int
    real_tokens: int
    padded_tokens: int
    packing_efficiency: float
    unpacked_padded_tokens: int


def _to_int32_1d_list(sequences: Sequence[Sequence[int] | np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for idx, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"Sequence at index {idx} must be 1D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"Sequence at index {idx} must be non-empty")
        out.append(arr)
    return out


def find_common_prefix_len(
    sequences: Sequence[Sequence[int] | np.ndarray],
    *,
    min_prefix_len: int = MIN_SHARED_PREFIX_TOKENS,
    prompt_lens: Sequence[int] | None = None,
    require_multiple: bool = True,
) -> int:
    """Return longest common leading token prefix length ``>= min_prefix_len``, else ``0``.

    Guarantees at least one suffix token remains in every sequence (and when ``prompt_lens``
    is provided, stops strictly before every example's first answer predictor ``prompt_len - 1``).
    """
    if int(min_prefix_len) <= 0:
        raise ValueError(f"min_prefix_len must be positive, got {min_prefix_len!r}")
    if not sequences:
        return 0
    if require_multiple and len(sequences) < 2:
        return 0

    arrs = _to_int32_1d_list(sequences)
    if prompt_lens is not None:
        if len(prompt_lens) != len(arrs):
            raise ValueError(
                f"prompt_lens length ({len(prompt_lens)}) must match sequences length ({len(arrs)})"
            )
        caps: list[int] = []
        for idx, (arr, plen) in enumerate(zip(arrs, prompt_lens)):
            p = int(plen)
            if p < 1 or p > int(arr.size):
                raise ValueError(
                    f"prompt_lens[{idx}]={p} out of valid range [1, {int(arr.size)}]"
                )
            caps.append(min(p - 1, int(arr.size) - 1))
        max_cap = min(caps)
    else:
        max_cap = min(int(arr.size) - 1 for arr in arrs)

    if max_cap < int(min_prefix_len):
        return 0

    ref = arrs[0][:max_cap]
    common = max_cap
    for arr in arrs[1:]:
        diff_idx = np.flatnonzero(arr[:common] != ref[:common])
        if diff_idx.size > 0:
            common = int(diff_idx[0])
        if common < int(min_prefix_len):
            return 0
    return common if common >= int(min_prefix_len) else 0


def build_block_diagonal_mask(
    segment_ids: np.ndarray | Sequence[int],
    *,
    causal: bool = True,
    shared_prefix_len: int = 0,
    position_ids: np.ndarray | Sequence[int] | None = None,
    window_size: int | None = None,
    additive: bool = False,
    neg_value: float = MASK_NEG_VAL_FP16,
    dtype: Any = np.float16,
) -> np.ndarray:
    """Construct a block-diagonal segment attention mask (boolean or FP16-safe additive).

    Enforces three invariants:
      - Real tokens (``segment_id >= 0``) attend only within their own segment (and causally
        ``k <= q`` when ``causal=True``, optionally clipped to ``window_size``).
      - When ``shared_prefix_len > 0``, real tokens also attend to the leading prefix columns
        ``[0, shared_prefix_len)`` (clipped by ``window_size`` via ``position_ids`` if set).
      - Pad tokens (``segment_id < 0``) never attend to real or prefix tokens and retain strict
        diagonal self-attention so downstream softmax never encounters an all-masked row.
    """
    pfx_len = int(shared_prefix_len)
    if pfx_len < 0:
        raise ValueError(f"shared_prefix_len must be non-negative, got {shared_prefix_len!r}")
    if window_size is not None and int(window_size) <= 0:
        raise ValueError(f"window_size must be positive when set, got {window_size!r}")

    seg_arr = np.asarray(segment_ids, dtype=np.int32)
    if seg_arr.ndim == 1:
        seg_2d = seg_arr[None, :]
        squeeze_batch = True
    elif seg_arr.ndim == 2:
        seg_2d = seg_arr
        squeeze_batch = False
    else:
        raise ValueError(f"segment_ids must be 1D or 2D, got shape {seg_arr.shape}")

    bsz, seq_len = int(seg_2d.shape[0]), int(seg_2d.shape[1])
    q_idx = np.arange(seq_len, dtype=np.int32)[None, :, None]
    k_idx = np.arange(seq_len, dtype=np.int32)[None, None, :]

    is_real_q = seg_2d[:, :, None] >= 0
    same_real_seg = (seg_2d[:, :, None] == seg_2d[:, None, :]) & is_real_q

    if causal:
        body_ok = same_real_seg & (q_idx >= k_idx)
        if window_size is not None:
            body_ok = body_ok & (q_idx < (k_idx + int(window_size)))
    else:
        body_ok = same_real_seg
        if window_size is not None:
            body_ok = body_ok & (np.abs(q_idx - k_idx) < int(window_size))

    # Diagonal self-attention for pad rows prevents NaN under -inf softmax while isolating pad tokens.
    pad_self = (~is_real_q) & (q_idx == k_idx)
    body_ok = body_ok | pad_self

    if pfx_len > 0:
        if window_size is not None:
            if position_ids is not None:
                pos_arr = np.asarray(position_ids, dtype=np.int32)
                pos_2d = pos_arr[None, :] if pos_arr.ndim == 1 else pos_arr
                if pos_2d.shape != (bsz, seq_len):
                    raise ValueError(
                        f"position_ids shape {pos_arr.shape} does not match segment_ids shape {seg_arr.shape}"
                    )
            else:
                pos_2d = np.broadcast_to(
                    np.arange(pfx_len, pfx_len + seq_len, dtype=np.int32)[None, :], (bsz, seq_len)
                )
            k_pfx = np.arange(pfx_len, dtype=np.int32)[None, None, :]
            q_pos = pos_2d[:, :, None]
            w = int(window_size)
            pfx_ok = is_real_q & (q_pos >= k_pfx) & (q_pos < (k_pfx + w))
        else:
            pfx_ok = np.broadcast_to(is_real_q, (bsz, seq_len, pfx_len)).copy()
        allowed = np.concatenate([pfx_ok, body_ok], axis=-1)
    else:
        allowed = body_ok

    if additive:
        out_dt = np.dtype(dtype)
        zero = np.asarray(0.0, dtype=out_dt)
        neg = np.asarray(float(neg_value), dtype=out_dt)
        result = np.where(allowed, zero, neg)
    else:
        result = allowed

    if squeeze_batch:
        return result[0]
    return result[:, None, :, :]


def _ffd_assign_bins(lengths: Sequence[int], max_seq_len: int) -> list[list[int]]:
    """Deterministic First-Fit-Decreasing (FFD) bin assignment over item indices.

    Items are sorted by ``(-length, orig_idx)`` for stability. Oversized items
    (``length > max_seq_len``) are placed intact in singleton bins without truncation.
    """
    order = sorted(range(len(lengths)), key=lambda idx: (-int(lengths[idx]), idx))
    bins: list[list[int]] = []
    used: list[int] = []
    cap = int(max_seq_len)

    for orig_idx in order:
        item_len = int(lengths[orig_idx])
        if item_len > cap:
            bins.append([orig_idx])
            used.append(item_len)
            continue
        placed = False
        for bi in range(len(bins)):
            if used[bi] + item_len <= cap:
                bins[bi].append(orig_idx)
                used[bi] += item_len
                placed = True
                break
        if not placed:
            bins.append([orig_idx])
            used.append(item_len)
    return bins


def pack_sequences_ffd(
    sequences: Sequence[Sequence[int] | np.ndarray],
    max_seq_len: int,
    *,
    pad_token_id: int = 0,
    pad_to_max: bool = True,
    pad_multiple: int = 1,
    dedup_shared_prefix: bool = False,
    min_shared_prefix_len: int = MIN_SHARED_PREFIX_TOKENS,
    prompt_lens: Sequence[int] | None = None,
    causal: bool = True,
    window_size: int | None = None,
    include_mask: bool = True,
) -> PackedSequenceResult:
    """Pack variable-length token sequences into capacity-``max_seq_len`` bins via deterministic FFD.

    Supports per-segment RoPE ``position_ids`` resets, block-diagonal segment causal/bidirectional
    masks, tile-multiple padding, and optional shared-system-prompt prefix deduplication.
    """
    cap = int(max_seq_len)
    if cap <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len!r}")
    pmult = int(pad_multiple)
    if pmult <= 0:
        raise ValueError(f"pad_multiple must be positive, got {pad_multiple!r}")
    if int(min_shared_prefix_len) <= 0:
        raise ValueError(f"min_shared_prefix_len must be positive, got {min_shared_prefix_len!r}")
    if window_size is not None and int(window_size) <= 0:
        raise ValueError(f"window_size must be positive when set, got {window_size!r}")

    if not sequences:
        return PackedSequenceResult(
            bins=(),
            max_seq_len=cap,
            shared_prefix_ids=np.empty((0,), dtype=np.int32),
            shared_prefix_len=0,
            num_sequences=0,
            num_bins=0,
            logical_tokens=0,
            packed_real_tokens=0,
            packed_total_tokens=0,
            effective_compute_tokens=0,
            deduplicated_tokens=0,
            packing_efficiency=0.0,
            pad_savings_ratio=1.0,
        )

    arrs = _to_int32_1d_list(sequences)
    logical_tokens = sum(int(a.size) for a in arrs)
    max_input_len = max(int(a.size) for a in arrs)

    if dedup_shared_prefix:
        shared_prefix_len = find_common_prefix_len(
            arrs,
            min_prefix_len=min_shared_prefix_len,
            prompt_lens=prompt_lens,
            require_multiple=True,
        )
    else:
        if prompt_lens is not None:
            if len(prompt_lens) != len(arrs):
                raise ValueError(
                    f"prompt_lens length ({len(prompt_lens)}) must match sequences length ({len(arrs)})"
                )
            for idx, (arr, plen) in enumerate(zip(arrs, prompt_lens)):
                if int(plen) < 1 or int(plen) > int(arr.size):
                    raise ValueError(
                        f"prompt_lens[{idx}]={plen} out of valid range [1, {int(arr.size)}]"
                    )
        shared_prefix_len = 0

    if shared_prefix_len > 0:
        shared_prefix_ids = arrs[0][:shared_prefix_len].copy()
        pack_bodies = [a[shared_prefix_len:] for a in arrs]
    else:
        shared_prefix_ids = np.empty((0,), dtype=np.int32)
        pack_bodies = arrs

    bin_assignments = _ffd_assign_bins([int(b.size) for b in pack_bodies], cap)
    packed_bins: list[PackedSequenceBin] = []

    for orig_indices in bin_assignments:
        real_toks = sum(int(pack_bodies[i].size) for i in orig_indices)
        raw_bin_len = max(cap, real_toks) if pad_to_max else real_toks
        bin_len = _align_up(raw_bin_len, pmult)

        in_ids = np.full((bin_len,), int(pad_token_id), dtype=np.int32)
        pos_ids = np.full((bin_len,), shared_prefix_len, dtype=np.int32)
        seg_ids = np.full((bin_len,), -1, dtype=np.int32)
        spans: list[PackedSegmentSpan] = []

        cursor = 0
        for orig_idx in orig_indices:
            body = pack_bodies[orig_idx]
            seg_len = int(body.size)
            in_ids[cursor : cursor + seg_len] = body
            pos_ids[cursor : cursor + seg_len] = np.arange(
                shared_prefix_len, shared_prefix_len + seg_len, dtype=np.int32
            )
            seg_ids[cursor : cursor + seg_len] = orig_idx
            spans.append(
                PackedSegmentSpan(
                    orig_index=orig_idx,
                    start=cursor,
                    end=cursor + seg_len,
                    position_offset=shared_prefix_len,
                    length=seg_len,
                )
            )
            cursor += seg_len

        mask = (
            build_block_diagonal_mask(
                seg_ids,
                causal=causal,
                shared_prefix_len=shared_prefix_len,
                position_ids=pos_ids,
                window_size=window_size,
                additive=False,
            )
            if include_mask
            else None
        )
        packed_bins.append(
            PackedSequenceBin(
                input_ids=in_ids,
                position_ids=pos_ids,
                segment_ids=seg_ids,
                item_indices=tuple(orig_indices),
                segment_spans=tuple(spans),
                real_tokens=real_toks,
                bin_len=bin_len,
                shared_prefix_len=shared_prefix_len,
                attention_mask=mask,
            )
        )

    packed_real_tokens = sum(b.real_tokens for b in packed_bins)
    packed_total_tokens = sum(b.bin_len for b in packed_bins)
    effective_compute_tokens = shared_prefix_len + packed_real_tokens
    deduplicated_tokens = max(0, (len(arrs) - 1) * shared_prefix_len)
    packing_efficiency = float(packed_real_tokens) / float(max(1, packed_total_tokens))
    padded_baseline_tokens = len(arrs) * max_input_len
    pad_savings_ratio = float(padded_baseline_tokens) / float(
        max(1, shared_prefix_len + packed_total_tokens)
    )

    return PackedSequenceResult(
        bins=tuple(packed_bins),
        max_seq_len=cap,
        shared_prefix_ids=shared_prefix_ids,
        shared_prefix_len=shared_prefix_len,
        num_sequences=len(arrs),
        num_bins=len(packed_bins),
        logical_tokens=logical_tokens,
        packed_real_tokens=packed_real_tokens,
        packed_total_tokens=packed_total_tokens,
        effective_compute_tokens=effective_compute_tokens,
        deduplicated_tokens=deduplicated_tokens,
        packing_efficiency=packing_efficiency,
        pad_savings_ratio=pad_savings_ratio,
    )


def pack_encoder_batches(
    sequences: Sequence[Sequence[int] | np.ndarray],
    max_seq_len: int = 512,
    *,
    batch_size: int | None = None,
    pad_token_id: int = 0,
    pad_to_max: bool = True,
    pad_multiple: int = 1,
    include_mask: bool = True,
) -> list[PackedEncoderBatch]:
    """Pack variable-length encoder sequences into stacked bidirectional FFD batches.

    Mirrors the proven ``ModernBERT`` / dense-encoder packing algorithm: each step's items are
    FFD-packed into ``(num_bins, bin_width)`` arrays with exact flat ``cls_positions`` restoring
    original batch order and NaN-free block-diagonal bidirectional masks ``(num_bins, 1, S, S)``.
    """
    cap = int(max_seq_len)
    if cap <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len!r}")
    pmult = int(pad_multiple)
    if pmult <= 0:
        raise ValueError(f"pad_multiple must be positive, got {pad_multiple!r}")
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive when set, got {batch_size!r}")

    if not sequences:
        return []

    arrs = _to_int32_1d_list(sequences)
    step = len(arrs) if batch_size is None else int(batch_size)
    batches: list[PackedEncoderBatch] = []

    for start_idx in range(0, len(arrs), step):
        chunk = arrs[start_idx : start_idx + step]
        n_items = len(chunk)
        bin_assignments = _ffd_assign_bins([int(s.size) for s in chunk], cap)
        num_bins = len(bin_assignments)

        max_bin_real = max(sum(int(chunk[i].size) for i in grp) for grp in bin_assignments)
        raw_width = max(cap, max_bin_real) if pad_to_max else max_bin_real
        bin_width = _align_up(raw_width, pmult)

        input_ids = np.full((num_bins, bin_width), int(pad_token_id), dtype=np.int32)
        position_ids = np.zeros((num_bins, bin_width), dtype=np.int32)
        segment_ids = np.full((num_bins, bin_width), -1, dtype=np.int32)
        cls_positions = np.zeros((n_items,), dtype=np.int32)
        token_spans_list: list[tuple[int, int, int]] = [(0, 0, 0)] * n_items

        real_tokens = 0
        for bi, local_indices in enumerate(bin_assignments):
            cursor = 0
            for local_i in local_indices:
                seq = chunk[local_i]
                seg_len = int(seq.size)
                input_ids[bi, cursor : cursor + seg_len] = seq
                position_ids[bi, cursor : cursor + seg_len] = np.arange(seg_len, dtype=np.int32)
                segment_ids[bi, cursor : cursor + seg_len] = local_i
                cls_positions[local_i] = bi * bin_width + cursor
                token_spans_list[local_i] = (bi, cursor, cursor + seg_len)
                cursor += seg_len
                real_tokens += seg_len

        mask = (
            build_block_diagonal_mask(segment_ids, causal=False, additive=False)
            if include_mask
            else None
        )
        padded_tokens = num_bins * bin_width
        unpacked_width = _align_up(max(int(s.size) for s in chunk), 32)
        unpacked_padded_tokens = n_items * unpacked_width

        batches.append(
            PackedEncoderBatch(
                input_ids=input_ids,
                position_ids=position_ids,
                segment_ids=segment_ids,
                cls_positions=cls_positions,
                item_indices=tuple(range(start_idx, start_idx + n_items)),
                token_spans=tuple(token_spans_list),
                attention_mask=mask,
                num_items=n_items,
                num_bins=num_bins,
                bin_width=bin_width,
                real_tokens=real_tokens,
                padded_tokens=padded_tokens,
                packing_efficiency=float(real_tokens) / float(max(1, padded_tokens)),
                unpacked_padded_tokens=unpacked_padded_tokens,
            )
        )

    return batches


# ---------------------------------------------------------------------------
# 3. Lesson 7: Batch-Regime Speculative Policy Guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeculativePolicy:
    """Empirical Apple9/Apple10 NAX batch-regime decision for MTP and prefix-tree speculation."""

    mtp_enabled: bool
    prefix_tree_enabled: bool
    mtp_draft_k: int
    decoder_mode: str
    batch_size: int
    shared_prefix_tokens: int
    num_candidates: int
    apple10_nax: bool | None
    regime: str
    reason: str
    mtp_reason: str
    prefix_tree_reason: str

    @property
    def draft_k(self) -> int:
        """Alias for :attr:`mtp_draft_k`."""
        return self.mtp_draft_k

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy decision to a JSON-friendly dictionary."""
        return {
            "mtp_enabled": self.mtp_enabled,
            "prefix_tree_enabled": self.prefix_tree_enabled,
            "mtp_draft_k": self.mtp_draft_k,
            "decoder_mode": self.decoder_mode,
            "batch_size": self.batch_size,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "num_candidates": self.num_candidates,
            "apple10_nax": self.apple10_nax,
            "regime": self.regime,
            "reason": self.reason,
            "mtp_reason": self.mtp_reason,
            "prefix_tree_reason": self.prefix_tree_reason,
        }


def select_speculative_policy(
    batch_size: int = 1,
    *,
    enable_mtp: bool = True,
    shared_prefix_tokens: int = 0,
    num_candidates: int = 1,
    min_shared_prefix_tokens: int = MIN_SHARED_PREFIX_TOKENS,
    has_grammar: bool = False,
    temperature: float = 0.0,
    use_ane: bool = False,
    apple10_nax: bool | None = None,
    profile: ProfileMode | None = None,
) -> SpeculativePolicy:
    """Select safe MTP speculative drafting and prefix-tree KV verification policy for Apple Silicon.

    Encodes our empirical Apple9/Apple10 NAX thresholds:
      - ``profile == "backfill"`` (when explicitly selected): disables ``B=1`` MTP speculative
        drafting so backfill corpora never pay drafter weight residency or verify overhead.
      - ``B == 1`` (greedy GPU decode in interactive regime): ``mtp_enabled = bool(enable_mtp)``
        with ``draft_k=3`` (with grammar) or ``draft_k=4`` (free-text).
      - ``2 <= B <= 4``: chipset-aware — when ``apple10_nax=False`` (``Apple9`` M4), allows
        ``mtp_enabled=True, draft_k=2`` (keeping ``M = B*(K+1) <= 12 < 16``); when
        ``apple10_nax=True`` (``Apple10+`` M5 NAX) or unspecified (``None``, NAX-safe default),
        switches to ``mtp_enabled=False`` (``decoder_mode="ragged_b_slots"``) because NAX
        TensorOps accelerate ``M >= 8`` directly and saturate during batched verification.
      - ``B >= 5``: always ``mtp_enabled=False`` (``B=8`` regresses to ``0.56x``).
      - ``prefix_tree_enabled = bool(shared_prefix_tokens >= 32 and num_candidates >= 2)``:
        enables shared-prefix KV + tree-mask verification where ``N=4..8`` yields up to ``2.59x``.
    """
    bsz = int(batch_size)
    if bsz < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    pfx_toks = int(shared_prefix_tokens)
    if pfx_toks < 0:
        raise ValueError(f"shared_prefix_tokens must be >= 0, got {shared_prefix_tokens!r}")
    n_cands = int(num_candidates)
    if n_cands < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates!r}")
    min_pfx = int(min_shared_prefix_tokens)
    if min_pfx < 1:
        raise ValueError(f"min_shared_prefix_tokens must be >= 1, got {min_shared_prefix_tokens!r}")
    temp = _validate_non_negative_finite("temperature", temperature)
    prof_norm = None if profile is None else _normalize_profile(profile)

    if prof_norm == "backfill":
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = "MTP speculative/drafter gated off under profile='backfill'."
    elif not enable_mtp:
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = "MTP disabled by caller (enable_mtp=False)."
    elif use_ane:
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = "MTP disabled on ANE execution path (GPU decode only)."
    elif temp > 0.0:
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = f"MTP greedy verification disabled for stochastic sampling (temperature={temp:g} > 0)."
    elif bsz == 1:
        mtp_enabled = True
        mtp_draft_k = 3 if has_grammar else 4
        mtp_reason = (
            f"MTP enabled at batch_size=1 (draft_k={mtp_draft_k}; memory-bandwidth-bound regime "
            "below NAX verify tile saturation)."
        )
    elif 2 <= bsz <= 4 and apple10_nax is False:
        mtp_enabled = True
        mtp_draft_k = 2
        m_verify = bsz * (mtp_draft_k + 1)
        mtp_reason = (
            f"MTP enabled on Apple9 (has_nax=False) at batch_size={bsz} with draft_k=2 "
            f"(verify M={m_verify} <= 12 < 16 MPP tile threshold)."
        )
    elif 2 <= bsz <= 4 and apple10_nax is True:
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = (
            f"MTP auto-disabled on Apple10+ NAX at batch_size={bsz} (2 <= B <= 4): "
            "NAX TensorOps accelerate M >= 8 directly by 1.5x-1.9x; use ragged_b_slots."
        )
    else:
        mtp_enabled = False
        mtp_draft_k = 0
        mtp_reason = (
            f"MTP auto-disabled at batch_size={bsz}: Apple9/Apple10 NAX matmul tiles saturate "
            "during batched verification (B=8 regresses to 0.56x); use ragged_b_slots."
        )

    decoder_mode = "mtp_speculative" if mtp_enabled else ("ragged_b_slots" if bsz >= 2 else "autoregressive")

    prefix_tree_enabled = bool(pfx_toks >= min_pfx and n_cands >= 2)
    if prefix_tree_enabled:
        prefix_tree_reason = (
            f"Prefix-tree KV + tree-mask verification enabled (shared_prefix_tokens={pfx_toks} >= {min_pfx}, "
            f"num_candidates={n_cands} >= 2; multi-candidate N=4..8 yields up to 2.59x)."
        )
    elif pfx_toks < min_pfx and n_cands < 2:
        prefix_tree_reason = (
            f"Prefix-tree disabled: shared_prefix_tokens={pfx_toks} < {min_pfx} "
            f"and num_candidates={n_cands} < 2."
        )
    elif pfx_toks < min_pfx:
        prefix_tree_reason = (
            f"Prefix-tree disabled: shared_prefix_tokens={pfx_toks} < {min_pfx} minimum threshold."
        )
    else:
        prefix_tree_reason = (
            f"Prefix-tree disabled: num_candidates={n_cands} < 2 (no multi-candidate prefix reuse)."
        )

    if mtp_enabled and prefix_tree_enabled:
        regime = "mtp_and_prefix_tree"
    elif mtp_enabled:
        regime = "mtp_single_stream" if bsz == 1 else "mtp_apple9_microbatch"
    elif prefix_tree_enabled:
        regime = "prefix_tree_multi_candidate"
    elif bsz >= 2:
        regime = "batched_autoregressive"
    else:
        regime = "autoregressive"

    return SpeculativePolicy(
        mtp_enabled=mtp_enabled,
        prefix_tree_enabled=prefix_tree_enabled,
        mtp_draft_k=mtp_draft_k,
        decoder_mode=decoder_mode,
        batch_size=bsz,
        shared_prefix_tokens=pfx_toks,
        num_candidates=n_cands,
        apple10_nax=apple10_nax,
        regime=regime,
        reason=f"{mtp_reason} {prefix_tree_reason}",
        mtp_reason=mtp_reason,
        prefix_tree_reason=prefix_tree_reason,
    )


# ---------------------------------------------------------------------------
# 4. Disaggregated GPU Actor + ANE Critic Coordinator Helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisaggregatedCriticBudget:
    """Memory headroom and ANE critic microbatch schedule for disaggregated GRPO/Best-of-N."""

    host_mem_gb: float
    wired_ceiling_gb: float
    effective_gpu_cap_gb: float
    actor_gpu_gb: float
    ane_mapped_gb: float
    critic_metal_gb: float
    combined_gb: float
    actor_gpu_headroom_gb: float
    unified_headroom_gb: float
    total_rollouts: int
    group_size: int
    ane_pipe_depth: int
    ane_microbatches: tuple[int, ...]
    overlapped_critic: bool
    valid: bool
    violations: tuple[str, ...]
    reason: str

    @property
    def num_ane_microbatches(self) -> int:
        return len(self.ane_microbatches)

    def to_dict(self) -> dict[str, Any]:
        """Serialize disaggregated budget and schedule to a JSON-friendly dictionary."""
        return {
            "host_mem_gb": self.host_mem_gb,
            "wired_ceiling_gb": self.wired_ceiling_gb,
            "effective_gpu_cap_gb": self.effective_gpu_cap_gb,
            "actor_gpu_gb": self.actor_gpu_gb,
            "ane_mapped_gb": self.ane_mapped_gb,
            "critic_metal_gb": self.critic_metal_gb,
            "combined_gb": self.combined_gb,
            "actor_gpu_headroom_gb": self.actor_gpu_headroom_gb,
            "unified_headroom_gb": self.unified_headroom_gb,
            "total_rollouts": self.total_rollouts,
            "group_size": self.group_size,
            "ane_pipe_depth": self.ane_pipe_depth,
            "ane_microbatches": list(self.ane_microbatches),
            "num_ane_microbatches": self.num_ane_microbatches,
            "overlapped_critic": self.overlapped_critic,
            "valid": self.valid,
            "violations": list(self.violations),
            "reason": self.reason,
        }


def validate_grpo_ane_offload(
    host_mem_gb: float,
    actor_gpu_gb: float,
    ane_mapped_gb: float,
    *,
    critic_metal_gb: float = 0.0,
    metal_working_set_gb: float | None = None,
    num_prompts: int = 4,
    group_size: int = 4,
    ane_pipe_depth: int = 4,
    wired_ceiling_fraction: float = WIRED_MEMORY_CEILING_FRACTION,
    raise_on_error: bool = False,
) -> DisaggregatedCriticBudget:
    """Validate disaggregated ``GPU Actor + ANE Critic`` headroom and build the ANE critic schedule.

    Enforces:
      1. ``critic_metal_gb == 0.0`` (``0.0 GB`` critic Metal allocation — zero GPU contention).
      2. ``actor_gpu_gb + ane_mapped_gb + critic_metal_gb <= 0.80 * host_mem_gb``.
      3. ``actor_gpu_gb + critic_metal_gb <= effective_gpu_cap_gb``.
    """
    host_gb = _validate_positive_finite("host_mem_gb", host_mem_gb)
    actor_gb = _validate_non_negative_finite("actor_gpu_gb", actor_gpu_gb)
    ane_gb = _validate_non_negative_finite("ane_mapped_gb", ane_mapped_gb)
    crit_metal_gb = _validate_non_negative_finite("critic_metal_gb", critic_metal_gb)
    frac = _validate_positive_finite("wired_ceiling_fraction", wired_ceiling_fraction)
    if frac > WIRED_MEMORY_CEILING_FRACTION + 1e-12:
        raise ValueError(
            f"wired_ceiling_fraction must be <= {WIRED_MEMORY_CEILING_FRACTION}, got {wired_ceiling_fraction!r}"
        )

    n_prompts = int(num_prompts)
    g_size = int(group_size)
    pipe_depth = int(ane_pipe_depth)
    if n_prompts < 1:
        raise ValueError(f"num_prompts must be >= 1, got {num_prompts!r}")
    if g_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size!r}")
    if pipe_depth < 1:
        raise ValueError(f"ane_pipe_depth must be >= 1, got {ane_pipe_depth!r}")

    wired_ceiling_gb = _floor_4dp(frac * host_gb)
    if metal_working_set_gb is None:
        effective_gpu_cap_gb = wired_ceiling_gb
    else:
        metal_gb = _validate_non_negative_finite("metal_working_set_gb", metal_working_set_gb)
        effective_gpu_cap_gb = (
            wired_ceiling_gb if metal_gb == 0.0 else _floor_4dp(min(wired_ceiling_gb, metal_gb))
        )

    combined_gb = round(actor_gb + ane_gb + crit_metal_gb, 4)
    gpu_total_gb = round(actor_gb + crit_metal_gb, 4)
    unified_headroom_gb = round(wired_ceiling_gb - combined_gb, 4)
    actor_gpu_headroom_gb = round(min(effective_gpu_cap_gb - gpu_total_gb, unified_headroom_gb), 4)

    violations: list[str] = []
    if crit_metal_gb > 0.0:
        violations.append(
            f"ANE critic must have 0.0 GB Metal allocation, got critic_metal_gb={crit_metal_gb:.4f} GB."
        )
    if combined_gb > wired_ceiling_gb + 1e-9:
        violations.append(
            f"Combined GPU actor + ANE critic footprint ({combined_gb:.3f} GB) exceeds "
            f"{int(round(frac * 100))}% UMA ceiling ({wired_ceiling_gb:.3f} GB)."
        )
    if gpu_total_gb > effective_gpu_cap_gb + 1e-9:
        violations.append(
            f"GPU actor footprint ({gpu_total_gb:.3f} GB) exceeds effective Metal cap "
            f"({effective_gpu_cap_gb:.3f} GB)."
        )

    total_rollouts = n_prompts * g_size
    full_waves, rem = divmod(total_rollouts, pipe_depth)
    microbatches = tuple([pipe_depth] * full_waves + ([rem] if rem > 0 else []))

    valid = len(violations) == 0
    overlapped_critic = valid and crit_metal_gb == 0.0
    if valid:
        reason = (
            f"Disaggregated GPU Actor ({actor_gb:.2f} GB) + ANE Critic ({ane_gb:.2f} GB mapped, "
            f"0.00 GB Metal) fits within {wired_ceiling_gb:.2f} GB UMA ceiling "
            f"(actor headroom={actor_gpu_headroom_gb:.2f} GB, ANE schedule={list(microbatches)})."
        )
    else:
        reason = " ".join(violations)

    budget = DisaggregatedCriticBudget(
        host_mem_gb=host_gb,
        wired_ceiling_gb=wired_ceiling_gb,
        effective_gpu_cap_gb=effective_gpu_cap_gb,
        actor_gpu_gb=actor_gb,
        ane_mapped_gb=ane_gb,
        critic_metal_gb=crit_metal_gb,
        combined_gb=combined_gb,
        actor_gpu_headroom_gb=actor_gpu_headroom_gb,
        unified_headroom_gb=unified_headroom_gb,
        total_rollouts=total_rollouts,
        group_size=g_size,
        ane_pipe_depth=pipe_depth,
        ane_microbatches=microbatches,
        overlapped_critic=overlapped_critic,
        valid=valid,
        violations=tuple(violations),
        reason=reason,
    )
    if raise_on_error and not valid:
        raise ValueError(budget.reason)
    return budget


# ---------------------------------------------------------------------------
# 5. Unified Runtime Policy: Per-Host Init + Per-Call Dynamic Dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostSiliconProfile:
    """Resolved host chipset, RAM tier, and OS capability gates cached at policy init."""

    gpu_family: str
    chip_brand: str
    host_mem_gb: float
    metal_working_set_gb: float | None
    macos_version: str | None
    macos_version_tuple: tuple[int, ...]
    apple10_nax: bool
    os_nax_ready: bool
    metal4_int_tensor_ops_ready: bool
    ane_core_fix_ready: bool
    os_coreai_runtime_ready: bool
    fused_swa_sdpa_enabled: bool
    fused_swa_sdpa_seq_lens: tuple[int, ...]
    tf32_nax_gemm_enabled: bool
    gemm_schedule: str
    hetero2_gpu_share: float
    ram_tier: str
    memory_limit_gb: float
    cache_limit_gb: float
    wired_ceiling_gb: float
    effective_cap_gb: float
    extraction_batch_size: int
    ane_ring_variant: str
    qlora_pack_len: int
    qlora_grad_accum: int
    has_calibrated_ane_w8: bool
    mlx_cache_limit_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB

    @property
    def has_nax(self) -> bool:
        """Alias for :attr:`apple10_nax`."""
        return self.apple10_nax

    def to_dict(self) -> dict[str, Any]:
        """Serialize host profile to a JSON-friendly dictionary."""
        return {
            "gpu_family": self.gpu_family,
            "chip_brand": self.chip_brand,
            "host_mem_gb": self.host_mem_gb,
            "metal_working_set_gb": self.metal_working_set_gb,
            "macos_version": self.macos_version,
            "macos_version_tuple": list(self.macos_version_tuple),
            "apple10_nax": self.apple10_nax,
            "os_nax_ready": self.os_nax_ready,
            "metal4_int_tensor_ops_ready": self.metal4_int_tensor_ops_ready,
            "ane_core_fix_ready": self.ane_core_fix_ready,
            "os_coreai_runtime_ready": self.os_coreai_runtime_ready,
            "fused_swa_sdpa_enabled": self.fused_swa_sdpa_enabled,
            "fused_swa_sdpa_seq_lens": list(self.fused_swa_sdpa_seq_lens),
            "tf32_nax_gemm_enabled": self.tf32_nax_gemm_enabled,
            "gemm_schedule": self.gemm_schedule,
            "hetero2_gpu_share": self.hetero2_gpu_share,
            "ram_tier": self.ram_tier,
            "memory_limit_gb": self.memory_limit_gb,
            "cache_limit_gb": self.cache_limit_gb,
            "mlx_cache_limit_mb": self.mlx_cache_limit_mb,
            "wired_ceiling_gb": self.wired_ceiling_gb,
            "effective_cap_gb": self.effective_cap_gb,
            "extraction_batch_size": self.extraction_batch_size,
            "ane_ring_variant": self.ane_ring_variant,
            "qlora_pack_len": self.qlora_pack_len,
            "qlora_grad_accum": self.qlora_grad_accum,
            "has_calibrated_ane_w8": self.has_calibrated_ane_w8,
        }


@dataclass(frozen=True)
class DynamicExecutionPlan:
    """Per-call runtime execution plan synthesized from host silicon + live batch/prefix shape."""

    # Per-call request descriptors
    workload: str
    batch_size: int
    seq_len: int
    shared_prefix_tokens: int
    num_candidates: int
    temperature: float
    has_grammar: bool
    mtp_mode: str
    model_name: str | None

    # Speculative & decoder dispatch
    mtp_enabled: bool
    mtp_draft_k: int
    should_load_mtp_drafter: bool
    estimated_drafter_mem_mb: float
    decoder_mode: str
    prefix_tree_enabled: bool

    # Encoder / hetero scheduler & chipset compute paths
    scheduler: str
    hetero2_gpu_share: float
    apple10_nax: bool
    fused_swa_sdpa_enabled: bool
    use_fused_swa_sdpa: bool
    tf32_nax_gemm_enabled: bool
    gemm_schedule: str

    # Host RAM tier & workload sizing
    ram_tier: str
    memory_limit_gb: float
    cache_limit_gb: float
    wired_ceiling_gb: float
    effective_cap_gb: float
    extraction_batch_size: int
    ane_ring_variant: str
    qlora_pack_len: int
    qlora_grad_accum: int

    # OS gates & ANE backend selection
    macos_version: str | None
    os_nax_ready: bool
    metal4_int_tensor_ops_ready: bool
    ane_core_fix_ready: bool
    os_coreai_runtime_ready: bool
    preferred_ane_backend: str
    coreai_promotion_passed: bool

    # Detailed sub-objects & explanation
    speculative: SpeculativePolicy
    budget: SiliconBudget
    reason: str
    profile: str = "backfill"
    mlx_cache_limit_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB

    @property
    def draft_k(self) -> int:
        """Alias for :attr:`mtp_draft_k`."""
        return self.mtp_draft_k

    @property
    def has_nax(self) -> bool:
        """Alias for :attr:`apple10_nax`."""
        return self.apple10_nax

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution plan to a JSON-friendly dictionary."""
        return {
            "profile": self.profile,
            "workload": self.workload,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "num_candidates": self.num_candidates,
            "temperature": self.temperature,
            "has_grammar": self.has_grammar,
            "mtp_mode": self.mtp_mode,
            "model_name": self.model_name,
            "mtp_enabled": self.mtp_enabled,
            "mtp_draft_k": self.mtp_draft_k,
            "should_load_mtp_drafter": self.should_load_mtp_drafter,
            "estimated_drafter_mem_mb": self.estimated_drafter_mem_mb,
            "decoder_mode": self.decoder_mode,
            "prefix_tree_enabled": self.prefix_tree_enabled,
            "scheduler": self.scheduler,
            "hetero2_gpu_share": self.hetero2_gpu_share,
            "apple10_nax": self.apple10_nax,
            "fused_swa_sdpa_enabled": self.fused_swa_sdpa_enabled,
            "use_fused_swa_sdpa": self.use_fused_swa_sdpa,
            "tf32_nax_gemm_enabled": self.tf32_nax_gemm_enabled,
            "gemm_schedule": self.gemm_schedule,
            "ram_tier": self.ram_tier,
            "memory_limit_gb": self.memory_limit_gb,
            "cache_limit_gb": self.cache_limit_gb,
            "mlx_cache_limit_mb": self.mlx_cache_limit_mb,
            "wired_ceiling_gb": self.wired_ceiling_gb,
            "effective_cap_gb": self.effective_cap_gb,
            "extraction_batch_size": self.extraction_batch_size,
            "ane_ring_variant": self.ane_ring_variant,
            "qlora_pack_len": self.qlora_pack_len,
            "qlora_grad_accum": self.qlora_grad_accum,
            "macos_version": self.macos_version,
            "os_nax_ready": self.os_nax_ready,
            "metal4_int_tensor_ops_ready": self.metal4_int_tensor_ops_ready,
            "ane_core_fix_ready": self.ane_core_fix_ready,
            "os_coreai_runtime_ready": self.os_coreai_runtime_ready,
            "preferred_ane_backend": self.preferred_ane_backend,
            "coreai_promotion_passed": self.coreai_promotion_passed,
            "speculative": self.speculative.to_dict(),
            "budget": self.budget.to_dict(),
            "reason": self.reason,
        }


def _normalize_promotion_targets(
    targets: Iterable[tuple[str, int]] | None,
) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    if targets is None:
        return out
    for model_name, batch_size in targets:
        key = str(model_name).strip().lower()
        bsz = int(batch_size)
        if not key:
            raise ValueError("Promoted CoreAI model_name must be non-empty.")
        if bsz < 1:
            raise ValueError(f"Promoted CoreAI batch_size must be >= 1, got {batch_size!r}")
        out.add((key, bsz))
    return out


def _resolve_host_profile(
    *,
    hw: Mapping[str, Any] | None = None,
    host_mem_gb: float | None = None,
    metal_working_set_gb: float | None = None,
    macos_version: str | None = None,
    apple10_nax: bool | None = None,
    gpu_family: str | None = None,
    chip_brand: str | None = None,
    mlx_enable_tf32: bool | None = None,
    has_calibrated_ane_w8: bool = False,
) -> HostSiliconProfile:
    """Resolve immutable host profile, probing ``probe_host_hardware()`` only when needed."""
    needs_probe = hw is None and (
        host_mem_gb is None or macos_version is None or (apple10_nax is None and gpu_family is None)
    )
    hw_info: dict[str, Any] = dict(probe_host_hardware() if needs_probe else (hw or {}))

    # Resolve macOS version and OS capability gates first (OS < 26.2 overrides NAX assumptions).
    eff_macos = macos_version if macos_version is not None else hw_info.get("macos_version")
    eff_macos_str = str(eff_macos).strip() if eff_macos else None
    os_tuple = parse_version_tuple(eff_macos_str)

    # Known OS version gates:
    #   < 26.2   : disables NAX assumptions
    #   >= 26.4  : unlocks Metal 4 int tensor ops
    #   >= 27.2.1: unlocks ANE core fix & Core AI OS runtime (still promotion-gated per (model, batch))
    os_nax_ready = _version_ge(os_tuple, (26, 2)) if os_tuple else False
    metal4_int_tensor_ops_ready = _version_ge(os_tuple, (26, 4)) if os_tuple else False
    ane_core_fix_ready = _version_ge(os_tuple, (27, 2, 1)) if os_tuple else False
    os_coreai_runtime_ready = ane_core_fix_ready

    # Resolve Apple10 NAX capability while respecting hw_check semantics + < 26.2 gate.
    if apple10_nax is not None:
        raw_nax = bool(apple10_nax)
        if hw_info.get("mlx_metal_no_nax"):
            raw_nax = False
        if eff_macos_str is not None and not os_nax_ready:
            raw_nax = False
        eff_nax = raw_nax
    else:
        probe_dict = dict(hw_info)
        if gpu_family is not None:
            probe_dict["gpu_family"] = gpu_family
            if gpu_family == "Apple10":
                probe_dict.setdefault("supports_apple10", True)
            elif gpu_family == "Apple9":
                probe_dict.setdefault("supports_apple10", False)
        if chip_brand is not None:
            probe_dict["brand"] = chip_brand
        if eff_macos_str is not None:
            probe_dict["macos_version"] = eff_macos_str
        # For synthetic Apple Silicon test dicts that omit system/machine, default to Darwin/arm64.
        if "system" not in probe_dict and (
            probe_dict.get("gpu_family") in ("Apple9", "Apple10")
            or probe_dict.get("supports_apple10") is not None
            or "apple" in str(probe_dict.get("brand") or "").lower()
        ):
            probe_dict["system"] = "Darwin"
            probe_dict.setdefault("machine", "arm64")
        eff_nax = bool(supports_apple10_nax(probe_dict))
        if eff_macos_str is not None and not os_nax_ready:
            eff_nax = False

    if gpu_family is not None:
        eff_gpu_family = str(gpu_family)
    elif hw_info.get("gpu_family"):
        eff_gpu_family = str(hw_info["gpu_family"])
    else:
        eff_gpu_family = "Apple10" if eff_nax else "Apple9"

    if chip_brand is not None:
        eff_brand = str(chip_brand)
    elif hw_info.get("brand"):
        eff_brand = str(hw_info["brand"])
    else:
        eff_brand = "Apple M5" if eff_gpu_family == "Apple10" else "Apple M4 Pro"

    raw_mem = host_mem_gb if host_mem_gb is not None else hw_info.get("mem_gb")
    if raw_mem is None or float(raw_mem) <= 0.0:
        eff_mem_gb = 24.0
    else:
        eff_mem_gb = _validate_positive_finite("host_mem_gb", float(raw_mem))

    eff_metal_ws: float | None
    if metal_working_set_gb is not None:
        mval = _validate_non_negative_finite("metal_working_set_gb", metal_working_set_gb)
        eff_metal_ws = None if mval == 0.0 else mval
    else:
        eff_metal_ws = None

    ext_budget = recommend_silicon_budget(eff_mem_gb, eff_metal_ws, workload="extraction")
    qlora_budget = recommend_silicon_budget(eff_mem_gb, eff_metal_ws, workload="qlora")

    # On <= 24 GB hosts, opt into W8 ANE ring only when an AWQ-MSE/QAT calibrated bundle is present.
    ane_ring_variant = "w8" if (eff_mem_gb <= 24.0 and has_calibrated_ane_w8) else "fp16"

    tf32_flag = bool(hw_info.get("mlx_enable_tf32", True)) if mlx_enable_tf32 is None else bool(mlx_enable_tf32)
    tf32_nax_gemm_enabled = bool(eff_nax and tf32_flag)
    if eff_nax:
        gemm_schedule = "nax_tf32" if tf32_nax_gemm_enabled else "nax_strict_fp"
        hetero2_gpu_share = 0.70
    else:
        gemm_schedule = "apple9_simdgroup"
        hetero2_gpu_share = 0.50

    return HostSiliconProfile(
        gpu_family=eff_gpu_family,
        chip_brand=eff_brand,
        host_mem_gb=eff_mem_gb,
        metal_working_set_gb=eff_metal_ws,
        macos_version=eff_macos_str,
        macos_version_tuple=os_tuple,
        apple10_nax=eff_nax,
        os_nax_ready=os_nax_ready,
        metal4_int_tensor_ops_ready=metal4_int_tensor_ops_ready,
        ane_core_fix_ready=ane_core_fix_ready,
        os_coreai_runtime_ready=os_coreai_runtime_ready,
        fused_swa_sdpa_enabled=eff_nax,
        fused_swa_sdpa_seq_lens=APPLE10_FUSED_SWA_SDPA_SEQ_LENS if eff_nax else (),
        tf32_nax_gemm_enabled=tf32_nax_gemm_enabled,
        gemm_schedule=gemm_schedule,
        hetero2_gpu_share=hetero2_gpu_share,
        ram_tier=ext_budget.tier,
        memory_limit_gb=ext_budget.memory_limit_gb,
        cache_limit_gb=ext_budget.cache_limit_gb,
        wired_ceiling_gb=ext_budget.wired_ceiling_gb,
        effective_cap_gb=ext_budget.effective_cap_gb,
        extraction_batch_size=ext_budget.batch_size,
        ane_ring_variant=ane_ring_variant,
        qlora_pack_len=qlora_budget.seq_pack_len,
        qlora_grad_accum=qlora_budget.grad_accum,
        has_calibrated_ane_w8=bool(has_calibrated_ane_w8),
    )


class DynamicSiliconPolicy:
    """Unified per-host + per-call Apple Silicon runtime execution policy.

    Probes host hardware once at init (via ``probe_host_hardware()`` when host arguments are
    ``None``), caches chipset/RAM/OS gates, and resolves per-call execution plans in ``O(1)``
    pure Python based on ``batch_size``, ``seq_len``, ``shared_prefix_tokens``, ``num_candidates``,
    ``temperature``, ``mtp_mode``, and Core AI promotion gates.
    """

    def __init__(
        self,
        *,
        profile: ProfileMode = _DEFAULT_PROFILE,
        mlx_cache_limit_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB,
        hw: Mapping[str, Any] | None = None,
        host_mem_gb: float | None = None,
        metal_working_set_gb: float | None = None,
        macos_version: str | None = None,
        apple10_nax: bool | None = None,
        gpu_family: str | None = None,
        chip_brand: str | None = None,
        mlx_enable_tf32: bool | None = None,
        has_calibrated_ane_w8: bool = False,
        promoted_coreai_targets: Iterable[tuple[str, int]] | None = None,
    ) -> None:
        self._explicit_profile: bool = not isinstance(profile, _ImplicitBackfillProfile)
        self.profile: str = _normalize_profile(profile)
        if isinstance(mlx_cache_limit_mb, bool) or int(mlx_cache_limit_mb) < 0:
            raise ValueError(f"mlx_cache_limit_mb must be a non-negative integer, got {mlx_cache_limit_mb!r}")
        self.mlx_cache_limit_mb: int = int(mlx_cache_limit_mb)
        self.host: HostSiliconProfile = _resolve_host_profile(
            hw=hw,
            host_mem_gb=host_mem_gb,
            metal_working_set_gb=metal_working_set_gb,
            macos_version=macos_version,
            apple10_nax=apple10_nax,
            gpu_family=gpu_family,
            chip_brand=chip_brand,
            mlx_enable_tf32=mlx_enable_tf32,
            has_calibrated_ane_w8=has_calibrated_ane_w8,
        )
        self._promoted_coreai_targets: set[tuple[str, int]] = _normalize_promotion_targets(
            promoted_coreai_targets
        )
        self._budgets: dict[str, SiliconBudget] = {
            wl: recommend_silicon_budget(
                self.host.host_mem_gb, self.host.metal_working_set_gb, workload=wl
            )
            for wl in VALID_WORKLOADS
        }

    @property
    def b1_speculative_budget_enabled(self) -> bool:
        """Return True when `B=1` speculative/drafter budget is enabled (`profile == 'interactive'`)."""
        return self.profile == "interactive"

    @property
    def drafter_budget_mb(self) -> float:
        """Estimated MTP drafter memory budget (`0.0` in `'backfill'`, `150.0` in `'interactive'`)."""
        return _MTP_DRAFTER_ESTIMATED_MB if self.profile == "interactive" else 0.0

    def apply_mlx_cache_limit(self) -> int | None:
        """Apply `self.mlx_cache_limit_mb` (`256 MiB` default) via `set_mlx_cache_limit`."""
        from .embeddings import set_mlx_cache_limit

        return set_mlx_cache_limit(self.mlx_cache_limit_mb)

    def poll_thermal_state(self) -> "ThermalStateSnapshot":
        """Poll host macOS thermal state for sustained multi-hour backfill pacing."""
        return poll_macos_thermal_state()

    @property
    def promoted_coreai_targets(self) -> frozenset[tuple[str, int]]:
        """Return immutable view of ``(model_name, batch_size)`` pairs promoted to Core AI."""
        return frozenset(self._promoted_coreai_targets)

    def register_coreai_promotion(self, model_name: str, batch_size: int) -> None:
        """Mark a ``(model_name, batch_size)`` pair as having passed Core AI promotion gates."""
        key = str(model_name).strip().lower()
        bsz = int(batch_size)
        if not key:
            raise ValueError("model_name must be non-empty.")
        if bsz < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self._promoted_coreai_targets.add((key, bsz))

    def budget_for(self, workload: str) -> SiliconBudget:
        """Return cached :class:`SiliconBudget` for ``workload``."""
        wl = str(workload).strip().lower()
        if wl in ("embedding", "encoder"):
            return self._budgets["extraction"]
        if wl not in self._budgets:
            raise ValueError(
                f"Unsupported workload={workload!r}; choose from {(*VALID_WORKLOADS, 'embedding')}"
            )
        return self._budgets[wl]

    def resolve(
        self,
        *,
        batch_size: int | None = None,
        seq_len: int = 512,
        workload: str = "extraction",
        shared_prefix_tokens: int = 0,
        num_candidates: int = 1,
        temperature: float = 0.0,
        has_grammar: bool = False,
        mtp_mode: str = "auto",
        enable_mtp: bool | None = None,
        use_ane: bool = False,
        model_name: str | None = None,
        coreai_promotion_passed: bool = False,
        profile: ProfileMode | None = None,
    ) -> DynamicExecutionPlan:
        """Resolve a per-call :class:`DynamicExecutionPlan` for the given batch and sequence shape."""
        eff_profile = self.profile if profile is None else _normalize_profile(profile)
        profile_gated: ProfileMode | None = (
            eff_profile  # type: ignore[assignment]
            if (profile is not None or self._explicit_profile)
            else None
        )
        if batch_size is None:
            bsz = self.host.extraction_batch_size if eff_profile == "backfill" else 1
        else:
            bsz = int(batch_size)
        if bsz < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        slen = int(seq_len)
        if slen < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len!r}")
        mode_norm = str(mtp_mode).strip().lower()
        if mode_norm not in VALID_MTP_MODES:
            raise ValueError(f"Unsupported mtp_mode={mtp_mode!r}; choose from {VALID_MTP_MODES}")

        wl_norm = str(workload).strip().lower()
        budget = self.budget_for(wl_norm)

        req_mtp = (mode_norm != "off") if enable_mtp is None else bool(enable_mtp)
        spec = select_speculative_policy(
            batch_size=bsz,
            enable_mtp=req_mtp,
            shared_prefix_tokens=shared_prefix_tokens,
            num_candidates=num_candidates,
            has_grammar=has_grammar,
            temperature=temperature,
            use_ane=use_ane,
            apple10_nax=self.host.apple10_nax,
            profile=profile_gated,
        )

        # Lazy drafter loading under mtp_mode="auto": backfill & B>=8 batch jobs pay 0 MB drafter RAM.
        if profile_gated == "backfill" or not req_mtp:
            should_load_drafter = False
        elif mode_norm == "on":
            should_load_drafter = True
        else:
            should_load_drafter = spec.mtp_enabled
        est_drafter_mb = _MTP_DRAFTER_ESTIMATED_MB if should_load_drafter else 0.0

        # Encoder / hetero work-stealing dispatch by batch regime & sequence length.
        if bsz == 1:
            scheduler = "direct_gpu"
        elif bsz <= 4:
            scheduler = "gpu_batched"
        else:
            scheduler = "hetero2_ffd"

        # Apple10 NAX shifts hetero2 GPU share higher vs Apple9 (and further for long contexts).
        if slen >= 2048:
            hetero2_gpu_share = 0.85 if self.host.apple10_nax else 0.65
        else:
            hetero2_gpu_share = self.host.hetero2_gpu_share

        use_fused_swa = bool(
            self.host.fused_swa_sdpa_enabled and slen in self.host.fused_swa_sdpa_seq_lens
        )

        # Keep preferred_ane_backend="coreml_mil" (100% pure ANE) unless OS >= 27.2.1 AND
        # the (model, batch) pair explicitly passed promotion gates.
        model_key = str(model_name).strip().lower() if model_name else None
        promoted = bool(
            coreai_promotion_passed
            or (model_key is not None and (model_key, bsz) in self._promoted_coreai_targets)
        )
        preferred_ane_backend = (
            "coreai" if (self.host.os_coreai_runtime_ready and promoted) else "coreml_mil"
        )

        reason = (
            f"profile={eff_profile}; host={self.host.gpu_family} (nax={self.host.apple10_nax}, "
            f"ram={self.host.host_mem_gb:.0f}GB tier={self.host.ram_tier}, os={self.host.macos_version or 'unknown'}); "
            f"B={bsz}, S={slen}, scheduler={scheduler}, decoder={spec.decoder_mode}, "
            f"ane_backend={preferred_ane_backend} (promoted={promoted}); {spec.reason}"
        )

        return DynamicExecutionPlan(
            workload=wl_norm,
            batch_size=bsz,
            seq_len=slen,
            shared_prefix_tokens=int(shared_prefix_tokens),
            num_candidates=int(num_candidates),
            temperature=float(temperature),
            has_grammar=bool(has_grammar),
            mtp_mode=mode_norm,
            model_name=model_name,
            mtp_enabled=spec.mtp_enabled,
            mtp_draft_k=spec.mtp_draft_k,
            should_load_mtp_drafter=should_load_drafter,
            estimated_drafter_mem_mb=est_drafter_mb,
            decoder_mode=spec.decoder_mode,
            prefix_tree_enabled=spec.prefix_tree_enabled,
            scheduler=scheduler,
            hetero2_gpu_share=hetero2_gpu_share,
            apple10_nax=self.host.apple10_nax,
            fused_swa_sdpa_enabled=self.host.fused_swa_sdpa_enabled,
            use_fused_swa_sdpa=use_fused_swa,
            tf32_nax_gemm_enabled=self.host.tf32_nax_gemm_enabled,
            gemm_schedule=self.host.gemm_schedule,
            ram_tier=self.host.ram_tier,
            memory_limit_gb=budget.memory_limit_gb,
            cache_limit_gb=budget.cache_limit_gb,
            wired_ceiling_gb=budget.wired_ceiling_gb,
            effective_cap_gb=budget.effective_cap_gb,
            extraction_batch_size=self.host.extraction_batch_size,
            ane_ring_variant=self.host.ane_ring_variant,
            qlora_pack_len=self.host.qlora_pack_len,
            qlora_grad_accum=self.host.qlora_grad_accum,
            macos_version=self.host.macos_version,
            os_nax_ready=self.host.os_nax_ready,
            metal4_int_tensor_ops_ready=self.host.metal4_int_tensor_ops_ready,
            ane_core_fix_ready=self.host.ane_core_fix_ready,
            os_coreai_runtime_ready=self.host.os_coreai_runtime_ready,
            preferred_ane_backend=preferred_ane_backend,
            coreai_promotion_passed=promoted,
            speculative=spec,
            budget=budget,
            reason=reason,
            profile=eff_profile,
            mlx_cache_limit_mb=self.mlx_cache_limit_mb,
        )

    def __call__(self, **kwargs: Any) -> DynamicExecutionPlan:
        return self.resolve(**kwargs)


def resolve_dynamic_execution_plan(
    *,
    batch_size: int = 1,
    seq_len: int = 512,
    workload: str = "extraction",
    shared_prefix_tokens: int = 0,
    num_candidates: int = 1,
    temperature: float = 0.0,
    has_grammar: bool = False,
    mtp_mode: str = "auto",
    enable_mtp: bool | None = None,
    use_ane: bool = False,
    model_name: str | None = None,
    coreai_promotion_passed: bool = False,
    profile: ProfileMode | None = None,
    mlx_cache_limit_mb: int = DEFAULT_MLX_CACHE_LIMIT_MB,
    hw: Mapping[str, Any] | None = None,
    host_mem_gb: float | None = None,
    metal_working_set_gb: float | None = None,
    macos_version: str | None = None,
    apple10_nax: bool | None = None,
    gpu_family: str | None = None,
    chip_brand: str | None = None,
    mlx_enable_tf32: bool | None = None,
    has_calibrated_ane_w8: bool = False,
    promoted_coreai_targets: Iterable[tuple[str, int]] | None = None,
) -> DynamicExecutionPlan:
    """One-shot dynamic execution planner (probes ``probe_host_hardware()`` when host args are ``None``)."""
    policy_kw: dict[str, Any] = {
        "mlx_cache_limit_mb": mlx_cache_limit_mb,
        "hw": hw,
        "host_mem_gb": host_mem_gb,
        "metal_working_set_gb": metal_working_set_gb,
        "macos_version": macos_version,
        "apple10_nax": apple10_nax,
        "gpu_family": gpu_family,
        "chip_brand": chip_brand,
        "mlx_enable_tf32": mlx_enable_tf32,
        "has_calibrated_ane_w8": has_calibrated_ane_w8,
        "promoted_coreai_targets": promoted_coreai_targets,
    }
    if profile is not None:
        policy_kw["profile"] = profile
    policy = DynamicSiliconPolicy(**policy_kw)
    return policy.resolve(
        batch_size=batch_size,
        seq_len=seq_len,
        workload=workload,
        shared_prefix_tokens=shared_prefix_tokens,
        num_candidates=num_candidates,
        temperature=temperature,
        has_grammar=has_grammar,
        mtp_mode=mtp_mode,
        enable_mtp=enable_mtp,
        use_ane=use_ane,
        model_name=model_name,
        coreai_promotion_passed=coreai_promotion_passed,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# 6. Persistent Cross-Batch Hetero2 (GPU Head + ANE Tail) Work-Stealing Queue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeteroWorkItem:
    """One length-tagged work item inside a :class:`Hetero2WorkStealingQueue`."""

    seq_id: int
    orig_index: int
    seq_len: int
    payload: Any


class Hetero2WorkStealingQueue:
    """Thread-safe persistent double-ended queue for 2-way `GPU + ANE` (`hetero2`) work-stealing.

    Longest sequences sit at the head (left) so GPU batches maximize tile saturation, while ANE
    workers steal shortest sequences (`<= max_seq_len`) from the tail (right) concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deque: deque[HeteroWorkItem] = deque()
        self._next_seq_id: int = 0
        self.gpu_items_completed: int = 0
        self.ane_items_completed: int = 0
        self.gpu_tokens_completed: int = 0
        self.ane_tokens_completed: int = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def gpu_share_actual(self) -> float:
        with self._lock:
            tot = self.gpu_tokens_completed + self.ane_tokens_completed
            return float(self.gpu_tokens_completed) / float(tot) if tot > 0 else 0.0

    def enqueue_batch(
        self,
        items: Sequence[Any],
        *,
        lengths: Sequence[int] | None = None,
    ) -> list[int]:
        """Enqueue a batch of items, keeping the persistent deque sorted longest-first (head) to shortest (tail)."""
        if lengths is not None and len(lengths) != len(items):
            raise ValueError(f"lengths ({len(lengths)}) must match items ({len(items)})")
        if not items:
            return []

        with self._lock:
            seq_ids: list[int] = []
            combined: list[HeteroWorkItem] = list(self._deque)
            for idx, payload in enumerate(items):
                if lengths is not None:
                    slen = max(1, int(lengths[idx]))
                elif hasattr(payload, "__len__"):
                    slen = max(1, len(payload))
                else:
                    slen = 1
                sid = self._next_seq_id
                self._next_seq_id += 1
                seq_ids.append(sid)
                combined.append(HeteroWorkItem(seq_id=sid, orig_index=idx, seq_len=slen, payload=payload))
            combined.sort(key=lambda w: (-w.seq_len, w.seq_id))
            self._deque = deque(combined)
            return seq_ids

    def pop_gpu_batch(
        self,
        max_items: int = 32,
        *,
        max_tokens: int | None = None,
    ) -> list[HeteroWorkItem]:
        """Steal up to `max_items` (and optional `max_tokens` budget) longest items from the head."""
        cap = max(1, int(max_items))
        tok_cap = int(max_tokens) if max_tokens is not None and int(max_tokens) > 0 else None
        out: list[HeteroWorkItem] = []
        tok_acc = 0
        with self._lock:
            while self._deque and len(out) < cap:
                nxt = self._deque[0]
                if tok_cap is not None and out and (tok_acc + nxt.seq_len) > tok_cap:
                    break
                item = self._deque.popleft()
                out.append(item)
                tok_acc += item.seq_len
            self.gpu_items_completed += len(out)
            self.gpu_tokens_completed += tok_acc
        return out

    def pop_ane_item(self, *, max_seq_len: int | None = 512) -> HeteroWorkItem | None:
        """Steal one shortest item from the tail if `seq_len <= max_seq_len`, else `None`."""
        with self._lock:
            if not self._deque:
                return None
            tail = self._deque[-1]
            if max_seq_len is not None and tail.seq_len > int(max_seq_len):
                return None
            item = self._deque.pop()
            self.ane_items_completed += 1
            self.ane_tokens_completed += item.seq_len
            return item

    def pop_ane_batch(
        self,
        max_items: int = 1,
        *,
        max_seq_len: int | None = 512,
    ) -> list[HeteroWorkItem]:
        """Steal up to `max_items` shortest items (`<= max_seq_len`) from the tail."""
        cap = max(1, int(max_items))
        out: list[HeteroWorkItem] = []
        for _ in range(cap):
            it = self.pop_ane_item(max_seq_len=max_seq_len)
            if it is None:
                break
            out.append(it)
        return out

    def run_concurrent(
        self,
        items: Sequence[Any],
        *,
        gpu_fn: Callable[[list[HeteroWorkItem]], Sequence[Any]],
        ane_fn: Callable[[list[HeteroWorkItem]], Sequence[Any]] | None = None,
        lengths: Sequence[int] | None = None,
        gpu_batch_size: int = 32,
        gpu_max_tokens: int | None = None,
        ane_batch_size: int = 1,
        ane_max_seq_len: int | None = 512,
        ane_workers: int = 1,
    ) -> list[Any]:
        """Drain `items` concurrently across GPU (head) and ANE (tail), restoring exact input order."""
        n = len(items)
        if n == 0:
            return []
        seq_ids = self.enqueue_batch(items, lengths=lengths)
        slot_by_sid = {sid: idx for idx, sid in enumerate(seq_ids)}
        results: list[Any] = [None] * n
        res_lock = threading.Lock()
        errors: list[BaseException] = []

        def _store(batch: list[HeteroWorkItem], outs: Sequence[Any]) -> None:
            if len(outs) != len(batch):
                raise RuntimeError(f"Hetero worker returned {len(outs)} outputs for batch of {len(batch)}.")
            with res_lock:
                for wi, val in zip(batch, outs):
                    pos = slot_by_sid.get(wi.seq_id)
                    if pos is not None:
                        results[pos] = val

        def _ane_loop() -> None:
            assert ane_fn is not None
            try:
                while not errors:
                    batch = self.pop_ane_batch(ane_batch_size, max_seq_len=ane_max_seq_len)
                    if not batch:
                        return
                    _store(batch, ane_fn(batch))
            except BaseException as exc:
                with res_lock:
                    errors.append(exc)

        threads: list[threading.Thread] = []
        if ane_fn is not None and int(ane_workers) > 0:
            for _ in range(int(ane_workers)):
                t = threading.Thread(target=_ane_loop, daemon=True)
                t.start()
                threads.append(t)

        try:
            while not errors:
                batch = self.pop_gpu_batch(gpu_batch_size, max_tokens=gpu_max_tokens)
                if not batch:
                    break
                _store(batch, gpu_fn(batch))
        except BaseException as exc:
            with res_lock:
                errors.append(exc)
        finally:
            for t in threads:
                t.join(timeout=10.0)

        if errors:
            raise errors[0]
        return results


# ---------------------------------------------------------------------------
# 7. macOS Thermal State Polling Helper (Foundation ctypes + pmset Fallback)
# ---------------------------------------------------------------------------

_THERMAL_LEVEL_BY_RAW: dict[int, ThermalLevel] = {
    0: "nominal",
    1: "fair",
    2: "serious",
    3: "critical",
}
_PMSET_SPEED_LIMIT_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ThermalStateSnapshot:
    """macOS `NSProcessInfo.thermalState` snapshot and backfill pacing recommendation."""

    raw_state: int
    level: ThermalLevel
    source: str
    throttle_recommended: bool
    recommended_batch_scale: float
    cooldown_sleep_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_state": self.raw_state,
            "level": self.level,
            "source": self.source,
            "throttle_recommended": self.throttle_recommended,
            "recommended_batch_scale": self.recommended_batch_scale,
            "cooldown_sleep_sec": self.cooldown_sleep_sec,
        }


def _read_foundation_thermal_state() -> int | None:
    """Query `[[NSProcessInfo processInfo] thermalState]` via arm64-safe `ctypes` prototypes."""
    if platform.system() != "Darwin":
        return None
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")

        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p

        cls_ptr = objc.objc_getClass(b"NSProcessInfo")
        if not cls_ptr:
            return None
        sel_proc = objc.sel_registerName(b"processInfo")
        sel_therm = objc.sel_registerName(b"thermalState")
        if not sel_proc or not sel_therm:
            return None

        msg_ptr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        if not msg_ptr:
            return None
        send_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(msg_ptr)
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(msg_ptr)

        proc_info = send_id(cls_ptr, sel_proc)
        if not proc_info:
            return None
        raw = int(send_long(proc_info, sel_therm))
        return max(0, min(3, raw))
    except Exception:
        return None


def _read_pmset_thermal_output() -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        proc = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except Exception:
        pass
    return None


def _snapshot_from_raw_thermal(raw: int, source: str) -> ThermalStateSnapshot:
    clamped = max(0, min(3, int(raw)))
    level = _THERMAL_LEVEL_BY_RAW[clamped]
    if clamped <= 1:
        return ThermalStateSnapshot(
            raw_state=clamped,
            level=level,
            source=source,
            throttle_recommended=False,
            recommended_batch_scale=1.0,
            cooldown_sleep_sec=0.0,
        )
    if clamped == 2:
        return ThermalStateSnapshot(
            raw_state=2,
            level="serious",
            source=source,
            throttle_recommended=True,
            recommended_batch_scale=0.5,
            cooldown_sleep_sec=0.05,
        )
    return ThermalStateSnapshot(
        raw_state=3,
        level="critical",
        source=source,
        throttle_recommended=True,
        recommended_batch_scale=0.25,
        cooldown_sleep_sec=0.25,
    )


def poll_macos_thermal_state() -> ThermalStateSnapshot:
    """Poll macOS thermal state via `Foundation` (`NSProcessInfo.thermalState`) or `pmset -g therm` fallback."""
    raw = _read_foundation_thermal_state()
    if raw is not None:
        return _snapshot_from_raw_thermal(raw, "foundation")

    pmset_out = _read_pmset_thermal_output()
    if pmset_out:
        m = _PMSET_SPEED_LIMIT_RE.search(pmset_out)
        if m is not None:
            speed = int(m.group(1))
            if speed <= 50:
                return _snapshot_from_raw_thermal(3, "pmset")
            if speed < 85:
                return _snapshot_from_raw_thermal(2, "pmset")
            if speed < 100:
                return _snapshot_from_raw_thermal(1, "pmset")
            return _snapshot_from_raw_thermal(0, "pmset")
        low = pmset_out.lower()
        if "sleeping" in low or "critical" in low:
            return _snapshot_from_raw_thermal(3, "pmset")
        if "thermal warning" in low and "no thermal warning" not in low:
            return _snapshot_from_raw_thermal(2, "pmset")
        return _snapshot_from_raw_thermal(0, "pmset")

    return _snapshot_from_raw_thermal(0, "fallback")


__all__ = [
    "APPLE10_FUSED_SWA_SDPA_SEQ_LENS",
    "DEFAULT_MLX_CACHE_LIMIT_MB",
    "DisaggregatedCriticBudget",
    "DynamicExecutionPlan",
    "DynamicSiliconPolicy",
    "Hetero2WorkStealingQueue",
    "HeteroWorkItem",
    "HostSiliconProfile",
    "MASK_NEG_VAL_FP16",
    "MIN_SHARED_PREFIX_TOKENS",
    "PackedEncoderBatch",
    "PackedSegmentSpan",
    "PackedSequenceBin",
    "PackedSequenceResult",
    "ProfileMode",
    "SiliconBudget",
    "SpeculativePolicy",
    "ThermalLevel",
    "ThermalStateSnapshot",
    "VALID_MTP_MODES",
    "VALID_PROFILES",
    "VALID_WORKLOADS",
    "WIRED_MEMORY_CEILING_FRACTION",
    "build_block_diagonal_mask",
    "find_common_prefix_len",
    "pack_encoder_batches",
    "pack_sequences_ffd",
    "poll_macos_thermal_state",
    "recommend_silicon_budget",
    "resolve_dynamic_execution_plan",
    "select_speculative_policy",
    "validate_grpo_ane_offload",
]
