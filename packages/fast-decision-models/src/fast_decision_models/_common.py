"""Stdlib-only math and hardware-lock utilities shared across decision-model runners."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import time
from typing import Iterator, Sequence

HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")


@contextmanager
def acquire_hw_lock(label: str) -> Iterator[None]:
    """Serialize Apple Silicon GPU/ANE/CoreML work on `/tmp/m4_gpu_validation.lock`."""
    HW_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(HW_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {label} {time.time():.3f}\n".encode("utf-8"))
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def softmax_vec(logits: Sequence[float], temp: float, *, min_temp: float = 1e-5) -> list[float]:
    """Numerically stable temperature-scaled softmax returning calibrated option probabilities."""
    if not logits:
        return []
    inv_t = 1.0 / max(float(temp), min_temp)
    scaled = [float(z) * inv_t for z in logits]
    m_z = max(scaled)
    exps = [math.exp(z - m_z) for z in scaled]
    denom = sum(exps)
    return [e / denom for e in exps]


def softmax_2(z0: float, z1: float, temp: float, *, min_temp: float = 1e-5) -> tuple[float, float]:
    """Fast two-option temperature-scaled softmax returning `(P(false), P(true))`."""
    inv_t = 1.0 / max(float(temp), min_temp)
    a = float(z0) * inv_t
    b = float(z1) * inv_t
    m = a if a > b else b
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    denom = ea + eb
    return ea / denom, eb / denom
