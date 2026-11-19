"""Reusable CI fault-injection harness for verifying numerical and behavioral regressions."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .bench import cos_sim

_VALID_METRICS = frozenset({"max_abs_diff", "cos_drop", "raises"})


@dataclass(frozen=True)
class FaultInjectionCase:
    """Single fault-injection test case verified against a baseline output."""

    name: str
    apply_fault: Callable[[], AbstractContextManager[None]]
    min_delta: float = 1e-2
    metric: str = "max_abs_diff"

    def __post_init__(self) -> None:
        if self.metric not in _VALID_METRICS:
            raise ValueError(
                f"Unsupported fault injection metric {self.metric!r}; "
                f"expected one of {sorted(_VALID_METRICS)}"
            )
        if not (self.min_delta >= 0.0):
            raise ValueError(f"`min_delta` must be >= 0, got {self.min_delta}")


@dataclass(frozen=True)
class FaultInjectionResult:
    """Outcome of executing one `FaultInjectionCase`."""

    name: str
    passed: bool
    measured_delta: float
    error: str | None = None


def run_fault_injection_suite(
    baseline_fn: Callable[[], np.ndarray],
    cases: Sequence[FaultInjectionCase],
) -> list[FaultInjectionResult]:
    """Run fault-injection cases in isolation against a single baseline evaluation.

    Computes `baseline_fn()` once, then enters each case's context manager in
    isolation (guaranteeing context exit restores state) and verifies that the
    injected fault either raises (`metric="raises"`) or shifts the output by at
    least `case.min_delta` (`metric="max_abs_diff"` or `"cos_drop"`).
    """
    baseline = np.array(baseline_fn(), copy=True)
    results: list[FaultInjectionResult] = []

    for case in cases:
        if case.metric == "raises":
            try:
                with case.apply_fault():
                    _ = baseline_fn()
            except Exception:
                results.append(
                    FaultInjectionResult(
                        name=case.name,
                        passed=True,
                        measured_delta=0.0,
                        error=None,
                    )
                )
            else:
                results.append(
                    FaultInjectionResult(
                        name=case.name,
                        passed=False,
                        measured_delta=0.0,
                        error="Expected injected fault to raise an exception, but baseline_fn() succeeded",
                    )
                )
            continue

        try:
            with case.apply_fault():
                corrupted = np.array(baseline_fn(), copy=True)
        except Exception as exc:
            results.append(
                FaultInjectionResult(
                    name=case.name,
                    passed=False,
                    measured_delta=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if corrupted.shape != baseline.shape:
            results.append(
                FaultInjectionResult(
                    name=case.name,
                    passed=False,
                    measured_delta=0.0,
                    error=f"Output shape changed under fault: baseline={baseline.shape}, got={corrupted.shape}",
                )
            )
            continue

        if baseline.size == 0:
            delta = 0.0
        elif case.metric == "max_abs_diff":
            delta = float(np.max(np.abs(corrupted.astype(np.float64) - baseline.astype(np.float64))))
        else:
            delta = float(1.0 - cos_sim(baseline, corrupted))

        if np.isnan(delta):
            results.append(
                FaultInjectionResult(
                    name=case.name,
                    passed=False,
                    measured_delta=delta,
                    error="Measured delta is NaN",
                )
            )
            continue

        results.append(
            FaultInjectionResult(
                name=case.name,
                passed=bool(delta >= case.min_delta),
                measured_delta=delta,
                error=None,
            )
        )

    return results


__all__ = [
    "FaultInjectionCase",
    "FaultInjectionResult",
    "run_fault_injection_suite",
]
