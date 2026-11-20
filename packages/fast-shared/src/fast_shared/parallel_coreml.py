"""Crash-safe, deterministic multi-process CoreML bucket compilation (`spawn`-only on macOS).

CoreML, Objective-C, and Metal runtimes are NOT fork-safe on macOS (`fork()` after Objective-C /
Metal initialization deadlocks or aborts in `libdispatch` / `ANECompilerService`). This module
strictly uses ``multiprocessing.get_context("spawn")`` for multi-worker CoreML `.mlpackage` ->
`.mlmodelc` builds, caps per-worker CPU BLAS threads to avoid oversubscription, isolates per-task
Python temp dirs, and publishes compiled bundles atomically so a crashed worker never leaves a
half-written `.mlmodelc` directory behind.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar
import uuid

T = TypeVar("T")

SPAWN_START_METHOD = "spawn"
DEFAULT_MAX_COMPILE_WORKERS = 4

DEFAULT_WORKER_ENV: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class BucketCompileResult(Generic[T]):
    """Result metadata for one compiled CoreML bucket task."""

    index: int
    label: str
    value: T
    elapsed_sec: float
    pid: int


@dataclass(frozen=True)
class BucketCompileFailure:
    """Structured failure record for a bucket task that raised or crashed."""

    index: int
    label: str
    error_type: str
    message: str
    traceback_text: str


@dataclass(frozen=True)
class _WorkerOutcome(Generic[T]):
    """Picklable transport envelope returned across `spawn` IPC."""

    ok: bool
    result: BucketCompileResult[T] | None = None
    failure: BucketCompileFailure | None = None


class BucketCompileError(RuntimeError):
    """Raised when one or more CoreML bucket compilation tasks fail or crash."""

    def __init__(
        self,
        failures: Sequence[BucketCompileFailure],
        completed: Sequence[BucketCompileResult[Any]] = (),
    ) -> None:
        self.failures: tuple[BucketCompileFailure, ...] = tuple(failures)
        self.completed: tuple[BucketCompileResult[Any], ...] = tuple(completed)
        lines = [
            f"CoreML bucket compilation failed ({len(self.failures)} failure(s), {len(self.completed)} completed):"
        ]
        for fail in self.failures:
            lines.append(f"  - [{fail.index}] {fail.label}: {fail.error_type}: {fail.message}")
            if fail.traceback_text:
                tb_tail = fail.traceback_text.strip().splitlines()[-6:]
                for tb_line in tb_tail:
                    lines.append(f"      {tb_line}")
        super().__init__("\n".join(lines))


def default_compile_jobs(
    jobs: int | None = None,
    *,
    cap: int = DEFAULT_MAX_COMPILE_WORKERS,
) -> int:
    """Resolve worker count: default bounded `min(cap, cpu_count)`, validating explicit `--jobs`."""
    if isinstance(cap, bool) or int(cap) < 1:
        raise ValueError(f"cap must be a positive integer, got {cap!r}")
    if jobs is None:
        cpus = os.cpu_count() or 1
        return max(1, min(int(cap), int(cpus)))
    if isinstance(jobs, bool) or int(jobs) < 1:
        raise ValueError(f"jobs must be a positive integer (>= 1), got {jobs!r}")
    return int(jobs)


def resolve_compile_workers(
    n_tasks: int,
    max_workers: int | None = None,
    *,
    cap: int = DEFAULT_MAX_COMPILE_WORKERS,
) -> int:
    """Bound effective worker processes to `min(default_compile_jobs(max_workers), n_tasks)`."""
    jobs = default_compile_jobs(max_workers, cap=cap)
    if n_tasks <= 0:
        return 1
    return max(1, min(jobs, int(n_tasks)))


def get_spawn_context() -> multiprocessing.context.BaseContext:
    """Return macOS Objective-C/CoreML-safe `spawn` multiprocessing context."""
    return multiprocessing.get_context(SPAWN_START_METHOD)


def atomic_publish_dir(
    src_dir: Path | str,
    dst_dir: Path | str,
    *,
    move: bool = False,
) -> Path:
    """Atomically stage and swap directory `src_dir` into `dst_dir` without leaving partial bundles.

    Works safely whether `src_dir` and `dst_dir` live on the same or different filesystems.
    If staging or swapping fails, any pre-existing `dst_dir` is preserved intact.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Source bundle directory does not exist: {src}")
    if dst.exists() and src.resolve() == dst.resolve():
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    stage = dst.parent / f".{dst.name}.stage.{token}"
    backup = dst.parent / f".{dst.name}.bak.{token}"

    try:
        if move:
            shutil.move(str(src), str(stage))
        else:
            shutil.copytree(src, stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    moved_old = False
    try:
        if dst.exists() or dst.is_symlink():
            os.rename(dst, backup)
            moved_old = True
        os.rename(stage, dst)
    except BaseException:
        if moved_old and backup.exists() and not dst.exists():
            try:
                os.rename(backup, dst)
            except OSError:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise
    else:
        if moved_old:
            if backup.is_symlink() or backup.is_file():
                backup.unlink(missing_ok=True)
            elif backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

    return dst


def _worker_init(env_overrides: dict[str, str], extra_sys_paths: tuple[str, ...]) -> None:
    """Initializer executed inside each spawned worker process."""
    for key, val in env_overrides.items():
        os.environ[key] = val
    for entry in reversed(extra_sys_paths):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)


def _execute_task_isolated(
    index: int,
    label: str,
    worker_fn: Callable[[Any], T],
    task: Any,
    isolate_tmpdir: bool,
) -> _WorkerOutcome[T]:
    """Run one bucket task with optional per-task `TMPDIR` isolation and structured error capture."""
    prev_tmpdir_env = os.environ.get("TMPDIR")
    prev_tf_tempdir = tempfile.tempdir
    tmp_ctx = tempfile.TemporaryDirectory(prefix="coreml_bucket_") if isolate_tmpdir else None
    t0 = time.perf_counter()
    try:
        if tmp_ctx is not None:
            os.environ["TMPDIR"] = tmp_ctx.name
            tempfile.tempdir = tmp_ctx.name
        value = worker_fn(task)
        elapsed = time.perf_counter() - t0
        return _WorkerOutcome(
            ok=True,
            result=BucketCompileResult(
                index=index,
                label=label,
                value=value,
                elapsed_sec=elapsed,
                pid=os.getpid(),
            ),
        )
    except Exception as exc:
        return _WorkerOutcome(
            ok=False,
            failure=BucketCompileFailure(
                index=index,
                label=label,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            ),
        )
    finally:
        tempfile.tempdir = prev_tf_tempdir
        if prev_tmpdir_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = prev_tmpdir_env
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def compile_buckets_parallel(
    tasks: Sequence[Any],
    worker_fn: Callable[[Any], T],
    *,
    max_workers: int | None = None,
    cap: int = DEFAULT_MAX_COMPILE_WORKERS,
    label_fn: Callable[[Any], str] | None = None,
    on_complete: Callable[[BucketCompileResult[T]], None] | None = None,
    env_overrides: Mapping[str, str] | None = None,
    isolate_tmpdir: bool = True,
    force_spawn: bool = False,
    extra_sys_paths: Sequence[str | Path] = (),
) -> list[BucketCompileResult[T]]:
    """Compile CoreML bucket tasks deterministically using a macOS-safe `spawn` process pool.

    Args:
        tasks: Sequence of lightweight picklable task payloads (prefer weight file paths over large arrays).
        worker_fn: Top-level picklable callable taking one task payload and returning its result.
        max_workers: Explicit worker count (`>= 1`) or `None` for bounded `min(cap, cpu_count)`.
        cap: Default upper bound when `max_workers` is `None` (default `4`).
        label_fn: Optional formatter producing a human-readable bucket label for logs/errors.
        on_complete: Optional parent-process callback invoked as each task completes.
        env_overrides: Optional environment overrides applied in spawned workers on top of `DEFAULT_WORKER_ENV`.
        isolate_tmpdir: Create & clean up a per-task `TMPDIR` so concurrent `coremltools` builds never collide.
        force_spawn: If True, still use a `spawn` pool even when `eff_workers == 1`.
        extra_sys_paths: Additional directories prepended to `sys.path` inside spawned workers.

    Returns:
        `list[BucketCompileResult[T]]` strictly in the original `tasks` order.
    """
    eff_workers = resolve_compile_workers(len(tasks), max_workers, cap=cap)
    if not tasks:
        return []

    indexed_tasks: list[tuple[int, str, Any]] = [
        (idx, label_fn(task) if label_fn is not None else f"bucket[{idx}]", task)
        for idx, task in enumerate(tasks)
    ]

    if eff_workers == 1 and not force_spawn:
        ordered_single: list[BucketCompileResult[T]] = []
        for idx, label, task in indexed_tasks:
            outcome = _execute_task_isolated(idx, label, worker_fn, task, isolate_tmpdir)
            if not outcome.ok or outcome.result is None:
                failure = outcome.failure or BucketCompileFailure(
                    index=idx,
                    label=label,
                    error_type="RuntimeError",
                    message="Bucket compilation failed without error details.",
                    traceback_text="",
                )
                raise BucketCompileError([failure], ordered_single)
            ordered_single.append(outcome.result)
            if on_complete is not None:
                on_complete(outcome.result)
        return ordered_single

    merged_env = dict(DEFAULT_WORKER_ENV)
    if env_overrides:
        for k, v in env_overrides.items():
            merged_env[str(k)] = str(v)

    sys_paths = tuple(str(Path(p).resolve()) for p in extra_sys_paths if p)
    ctx = get_spawn_context()

    results_by_idx: dict[int, BucketCompileResult[T]] = {}
    failures: list[BucketCompileFailure] = []

    executor = ProcessPoolExecutor(
        max_workers=eff_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(merged_env, sys_paths),
    )
    future_to_meta: dict[Any, tuple[int, str]] = {}
    try:
        for idx, label, task in indexed_tasks:
            fut = executor.submit(_execute_task_isolated, idx, label, worker_fn, task, isolate_tmpdir)
            future_to_meta[fut] = (idx, label)

        for fut in as_completed(future_to_meta):
            idx, label = future_to_meta[fut]
            try:
                outcome = fut.result()
            except BrokenProcessPool as exc:
                failures.append(
                    BucketCompileFailure(
                        index=idx,
                        label=label,
                        error_type="BrokenProcessPool",
                        message=f"Worker process crashed unexpectedly while compiling {label}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                )
                for other_fut in future_to_meta:
                    other_fut.cancel()
                break
            except Exception as exc:
                failures.append(
                    BucketCompileFailure(
                        index=idx,
                        label=label,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                )
                for other_fut in future_to_meta:
                    other_fut.cancel()
                break

            if not outcome.ok or outcome.result is None:
                failures.append(
                    outcome.failure
                    or BucketCompileFailure(
                        index=idx,
                        label=label,
                        error_type="RuntimeError",
                        message=f"Worker returned empty failure outcome for {label}",
                        traceback_text="",
                    )
                )
                for other_fut in future_to_meta:
                    other_fut.cancel()
                break

            results_by_idx[idx] = outcome.result
            if on_complete is not None:
                on_complete(outcome.result)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if failures:
        completed_ordered = [results_by_idx[i] for i in sorted(results_by_idx)]
        raise BucketCompileError(failures, completed_ordered)

    return [results_by_idx[i] for i in range(len(indexed_tasks))]


__all__ = [
    "BucketCompileError",
    "BucketCompileFailure",
    "BucketCompileResult",
    "DEFAULT_MAX_COMPILE_WORKERS",
    "DEFAULT_WORKER_ENV",
    "SPAWN_START_METHOD",
    "atomic_publish_dir",
    "compile_buckets_parallel",
    "default_compile_jobs",
    "get_spawn_context",
    "resolve_compile_workers",
]
