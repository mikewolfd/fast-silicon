"""Apple Silicon benchmark hygiene, contention checks, wired limits, ABBA harness, and MLX dtype guards."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .hw_check import probe_host_hardware, supports_apple10_nax, supports_metal4_bf16

_RUSAGE_INFO_V4 = 4
_RUSAGE_INFO_V4_SIZE = 296
_RUSAGE_INFO_V6 = 6
_RUSAGE_INFO_V6_SIZE = 464
_RESIDENT_SIZE_OFFSET = 64
_PHYS_FOOTPRINT_OFFSET = 72
_LIFETIME_MAX_FOOTPRINT_OFFSET = 240
_NEURAL_FOOTPRINT_V6_OFFSET = 368
_LIFETIME_MAX_NEURAL_FOOTPRINT_V6_OFFSET = 376

# Public macOS Mach / POSIX VM constants (verified against arm64 Darwin SDK headers; zero private SPIs).
_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = 93
_TVMI_INTERNAL_OFFSET = 48
_TVMI_EXTERNAL_OFFSET = 64
_TVMI_COMPRESSED_OFFSET = 120
_TVMI_PHYS_FOOTPRINT_OFFSET = 144
_TVMI_NEURAL_FOOTPRINT_OFFSET = 304
_TVMI_NEURAL_NOFOOTPRINT_OFFSET = 320
_TVMI_NEURAL_NOFOOTPRINT_TOTAL_OFFSET = 356

_VM_REGION_SUBMAP_INFO_COUNT_64 = 21
_VRS_PAGES_RESIDENT_OFFSET = 24
_VRS_EXTERNAL_PAGER_OFFSET = 46
_VRS_SHARE_MODE_OFFSET = 47
_VRS_IS_SUBMAP_OFFSET = 48
_VRS_OBJECT_ID_FULL_OFFSET = 68
_SM_SHARED = 4
_SM_TRUESHARED = 5
_SM_SHARED_ALIASED = 7
_SHARED_REGION_BASE_ARM64 = 0x180000000
_SHARED_REGION_SIZE_ARM64 = 0xE40000000
_SHARED_REGION_BASE_X86_64 = 0x7FF800000000
_SHARED_REGION_SIZE_X86_64 = 0x80000000
_MS_SYNC = 0x0010
_MS_INVALIDATE = 0x0002
_PROC_PIDPATHINFO_MAXSIZE = 4096

ANE_DAEMON_PROCESS_NAMES: frozenset[str] = frozenset(
    {"aned", "ANECompilerService", "ANELargeModelCompilerService"}
)
DEFAULT_METAL_TRIP_FRACTION: float = 0.98
DEFAULT_PAST_ARM_MARGIN: float = 0.02

_LIBPROC_CACHE: Any = None
_LIBSYSTEM_CACHE: Any = None
_MACH_TASK_SELF_CACHE: int | None = None


def _load_libproc() -> Any:
    global _LIBPROC_CACHE
    if _LIBPROC_CACHE is None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.proc_listallpids.restype = ctypes.c_int
        lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        lib.proc_pidpath.restype = ctypes.c_int
        lib.proc_regionfilename.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_uint32]
        lib.proc_regionfilename.restype = ctypes.c_int
        _LIBPROC_CACHE = lib
    return _LIBPROC_CACHE


def _load_libsystem() -> tuple[Any, int]:
    global _LIBSYSTEM_CACHE, _MACH_TASK_SELF_CACHE
    if _LIBSYSTEM_CACHE is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.task_info.restype = ctypes.c_int
        lib.mach_vm_region_recurse.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.mach_vm_region_recurse.restype = ctypes.c_int
        lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.msync.restype = ctypes.c_int
        _MACH_TASK_SELF_CACHE = int(ctypes.c_uint32.in_dll(lib, "mach_task_self_").value)
        _LIBSYSTEM_CACHE = lib
    assert _MACH_TASK_SELF_CACHE is not None
    return _LIBSYSTEM_CACHE, _MACH_TASK_SELF_CACHE


def get_unified_memory_footprint(pid: int | None = None) -> dict[str, float | int]:
    """Read current & lifetime-peak OS physical footprint (dirty/anonymous CPU + Metal/IOSurface) on macOS.

    Uses ``proc_pid_rusage(..., RUSAGE_INFO_V4)`` from ``/usr/lib/libproc.dylib``.
    Note: ``ri_phys_footprint`` reflects dirty/anonymous CPU, Metal, and IOSurface allocations
    but **excludes** clean file-backed ``mmap`` pages (e.g. ``MappedEmbedding`` weights),
    ``task_vm_info.ledger_tag_neural_nofootprint``, and out-of-process ANE daemons
    (``aned`` / ``ANECompilerService`` / ``ANELargeModelCompilerService``). Use
    ``probe_four_bucket_memory()`` or ``FourBucketMemoryProbe`` when full 4-bucket accounting
    is needed. Falls back cleanly to ``ru_maxrss`` on non-Darwin platforms.
    """
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    is_darwin = platform.system() == "Darwin"
    rss_bytes = int(ru_self.ru_maxrss) if is_darwin else int(ru_self.ru_maxrss) * 1024
    if not is_darwin:
        return {
            "phys_footprint_bytes": rss_bytes,
            "lifetime_max_phys_footprint_bytes": rss_bytes,
            "rss_bytes": rss_bytes,
            "phys_footprint_gb": rss_bytes / 1e9,
            "peak_footprint_gb": rss_bytes / 1e9,
        }

    buf = ctypes.create_string_buffer(512)
    target_pid = os.getpid() if pid is None else int(pid)
    rc = _load_libproc().proc_pid_rusage(target_pid, _RUSAGE_INFO_V4, ctypes.byref(buf))
    if rc != 0:
        raise OSError(f"proc_pid_rusage failed for pid={target_pid} (rc={rc})")

    phys_now = int.from_bytes(buf.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little")
    phys_peak = int.from_bytes(
        buf.raw[_LIFETIME_MAX_FOOTPRINT_OFFSET : _LIFETIME_MAX_FOOTPRINT_OFFSET + 8], "little"
    )
    return {
        "phys_footprint_bytes": phys_now,
        "lifetime_max_phys_footprint_bytes": phys_peak,
        "rss_bytes": rss_bytes,
        "phys_footprint_gb": phys_now / 1e9,
        "peak_footprint_gb": phys_peak / 1e9,
    }


def peak_footprint_gb(pid: int | None = None) -> float:
    """Return process lifetime-max unified-memory physical footprint in decimal GB."""
    return float(get_unified_memory_footprint(pid=pid)["peak_footprint_gb"])


def _normalize_model_prefixes(model_directories: Sequence[str | Path] | None) -> tuple[str, ...]:
    if not model_directories:
        return ()
    out: list[str] = []
    for raw in model_directories:
        rp = os.path.realpath(os.fspath(raw))
        if os.path.isfile(rp):
            out.append(rp)
        else:
            out.append(rp.rstrip("/") + "/")
    return tuple(dict.fromkeys(out))


def _path_matches_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in prefixes)


def probe_task_vm_info() -> dict[str, int | bool]:
    """Read current task VM ledger stats via public Mach ``task_info(..., TASK_VM_INFO)``.

    Exposes ``phys_footprint_bytes``, clean file-backed ``external_bytes``, and
    ``ledger_tag_neural_nofootprint_bytes`` without using any private SPIs.
    """
    if platform.system() != "Darwin":
        return {
            "available": False,
            "phys_footprint_bytes": 0,
            "internal_bytes": 0,
            "external_bytes": 0,
            "compressed_bytes": 0,
            "ledger_tag_neural_footprint_bytes": 0,
            "ledger_tag_neural_nofootprint_bytes": 0,
            "ledger_tag_neural_nofootprint_total_bytes": 0,
        }

    lib, task_self = _load_libsystem()
    buf = ctypes.create_string_buffer(512)
    cnt = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
    kr = int(lib.task_info(task_self, _TASK_VM_INFO, ctypes.byref(buf), ctypes.byref(cnt)))
    if kr != 0:
        raise OSError(f"task_info(TASK_VM_INFO) failed (kr={kr})")

    filled_bytes = int(cnt.value) * 4

    def _u64(off: int) -> int:
        if filled_bytes < off + 8:
            return 0
        return int.from_bytes(buf.raw[off : off + 8], "little", signed=False)

    def _i64(off: int) -> int:
        if filled_bytes < off + 8:
            return 0
        return int.from_bytes(buf.raw[off : off + 8], "little", signed=True)

    return {
        "available": True,
        "phys_footprint_bytes": _u64(_TVMI_PHYS_FOOTPRINT_OFFSET),
        "internal_bytes": _u64(_TVMI_INTERNAL_OFFSET),
        "external_bytes": _u64(_TVMI_EXTERNAL_OFFSET),
        "compressed_bytes": _u64(_TVMI_COMPRESSED_OFFSET),
        "ledger_tag_neural_footprint_bytes": _i64(_TVMI_NEURAL_FOOTPRINT_OFFSET),
        "ledger_tag_neural_nofootprint_bytes": _i64(_TVMI_NEURAL_NOFOOTPRINT_OFFSET),
        "ledger_tag_neural_nofootprint_total_bytes": _i64(_TVMI_NEURAL_NOFOOTPRINT_TOTAL_OFFSET),
    }


def _iter_darwin_vm_regions() -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield `(address, size, pages_resident, external_pager, share_mode, object_id_full)` for non-submap VM regions."""
    lib, task_self = _load_libsystem()
    is_arm64 = platform.machine().lower() in ("arm64", "aarch64")
    shared_base = _SHARED_REGION_BASE_ARM64 if is_arm64 else _SHARED_REGION_BASE_X86_64
    shared_end = shared_base + (_SHARED_REGION_SIZE_ARM64 if is_arm64 else _SHARED_REGION_SIZE_X86_64)

    addr = ctypes.c_uint64(0)
    size = ctypes.c_uint64(0)
    depth = ctypes.c_uint32(0)
    rbuf = ctypes.create_string_buffer(128)
    while True:
        rcnt = ctypes.c_uint32(_VM_REGION_SUBMAP_INFO_COUNT_64)
        kr = int(
            lib.mach_vm_region_recurse(
                task_self,
                ctypes.byref(addr),
                ctypes.byref(size),
                ctypes.byref(depth),
                ctypes.byref(rbuf),
                ctypes.byref(rcnt),
            )
        )
        if kr != 0:
            break
        cur_addr = int(addr.value)
        cur_size = int(size.value)
        if cur_size <= 0 or (cur_addr + cur_size) <= cur_addr:
            break
        addr.value = cur_addr + cur_size

        is_submap = int.from_bytes(rbuf.raw[_VRS_IS_SUBMAP_OFFSET : _VRS_IS_SUBMAP_OFFSET + 4], "little")
        if is_submap != 0 or (shared_base <= cur_addr < shared_end):
            continue

        pages_resident = int.from_bytes(
            rbuf.raw[_VRS_PAGES_RESIDENT_OFFSET : _VRS_PAGES_RESIDENT_OFFSET + 4], "little"
        )
        external_pager = int(rbuf.raw[_VRS_EXTERNAL_PAGER_OFFSET])
        share_mode = int(rbuf.raw[_VRS_SHARE_MODE_OFFSET])
        object_id_full = int.from_bytes(
            rbuf.raw[_VRS_OBJECT_ID_FULL_OFFSET : _VRS_OBJECT_ID_FULL_OFFSET + 8], "little"
        )
        yield (cur_addr, cur_size, pages_resident, external_pager, share_mode, object_id_full)


def probe_mapped_files(model_directories: Sequence[str | Path] | None = None) -> dict[str, Any]:
    """Walk this process's VM regions to measure resident clean file-backed ``mmap`` pages.

    Skips submaps and the dyld shared cache region, deduplicates shared mappings by
    ``object_id_full``, and optionally partitions resident bytes into ``model_bytes``
    (under ``model_directories``) vs ``shared_bytes`` (framework/code).
    """
    prefixes = _normalize_model_prefixes(model_directories)
    if platform.system() != "Darwin":
        return {
            "available": False,
            "total_bytes": 0,
            "model_bytes": 0,
            "shared_bytes": 0,
            "resident_by_path": {},
            "model_directories": list(prefixes),
        }

    libproc = _load_libproc()
    pid = os.getpid()
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    path_buf = ctypes.create_string_buffer(_PROC_PIDPATHINFO_MAXSIZE)
    seen_shared_objects: set[int] = set()
    resident_by_path: dict[str, int] = {}

    for cur_addr, _cur_size, pages_resident, external_pager, share_mode, object_id_full in _iter_darwin_vm_regions():
        if share_mode in (_SM_SHARED, _SM_TRUESHARED, _SM_SHARED_ALIASED):
            if object_id_full in seen_shared_objects:
                continue
            seen_shared_objects.add(object_id_full)
        if external_pager == 0 or pages_resident <= 0:
            continue
        resident_bytes = pages_resident * page_size
        nlen = int(libproc.proc_regionfilename(pid, cur_addr, path_buf, _PROC_PIDPATHINFO_MAXSIZE))
        path_str = path_buf.value.decode("utf-8", errors="ignore") if nlen > 0 else "(unnamed file)"
        resident_by_path[path_str] = resident_by_path.get(path_str, 0) + resident_bytes

    total_bytes = sum(resident_by_path.values())
    if prefixes:
        model_bytes = sum(v for k, v in resident_by_path.items() if _path_matches_prefixes(k, prefixes))
        shared_bytes = total_bytes - model_bytes
    else:
        model_bytes = 0
        shared_bytes = total_bytes

    return {
        "available": True,
        "total_bytes": total_bytes,
        "model_bytes": model_bytes,
        "shared_bytes": shared_bytes,
        "resident_by_path": resident_by_path,
        "model_directories": list(prefixes),
    }


def evict_mapped_files(model_directories: Sequence[str | Path]) -> dict[str, int | bool]:
    """Drop clean resident page-cache pages for mapped files under ``model_directories`` via ``msync``.

    Uses POSIX ``msync(addr, size, MS_SYNC | MS_INVALIDATE)`` so dirty pages are written back
    before clean pages are invalidated (pages wired by Metal/GPU drivers stay resident).
    """
    prefixes = _normalize_model_prefixes(model_directories)
    if platform.system() != "Darwin" or not prefixes:
        return {
            "available": platform.system() == "Darwin",
            "regions_evicted": 0,
            "bytes_requested": 0,
            "failed_regions": 0,
        }

    libsys, _ = _load_libsystem()
    libproc = _load_libproc()
    pid = os.getpid()
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    path_buf = ctypes.create_string_buffer(_PROC_PIDPATHINFO_MAXSIZE)
    regions_evicted = 0
    bytes_requested = 0
    failed_regions = 0

    for cur_addr, cur_size, pages_resident, external_pager, _share_mode, _obj_id in _iter_darwin_vm_regions():
        if external_pager == 0 or pages_resident <= 0:
            continue
        nlen = int(libproc.proc_regionfilename(pid, cur_addr, path_buf, _PROC_PIDPATHINFO_MAXSIZE))
        if nlen <= 0:
            continue
        path_str = path_buf.value.decode("utf-8", errors="ignore")
        if not _path_matches_prefixes(path_str, prefixes):
            continue
        rc = int(libsys.msync(ctypes.c_void_p(cur_addr), ctypes.c_size_t(cur_size), _MS_SYNC | _MS_INVALIDATE))
        if rc == 0:
            regions_evicted += 1
            bytes_requested += pages_resident * page_size
        else:
            failed_regions += 1

    return {
        "available": True,
        "regions_evicted": regions_evicted,
        "bytes_requested": bytes_requested,
        "failed_regions": failed_regions,
    }


def read_process_rusage_v6(pid: int) -> dict[str, int] | None:
    """Read ``RUSAGE_INFO_V6`` RSS, physical footprint, and neural footprint for ``pid``, or ``None`` if unreadable."""
    if platform.system() != "Darwin":
        return None
    buf = ctypes.create_string_buffer(512)
    rc = int(_load_libproc().proc_pid_rusage(int(pid), _RUSAGE_INFO_V6, ctypes.byref(buf)))
    if rc != 0:
        return None
    return {
        "pid": int(pid),
        "rss_bytes": int.from_bytes(buf.raw[_RESIDENT_SIZE_OFFSET : _RESIDENT_SIZE_OFFSET + 8], "little"),
        "phys_footprint_bytes": int.from_bytes(buf.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little"),
        "lifetime_max_phys_footprint_bytes": int.from_bytes(
            buf.raw[_LIFETIME_MAX_FOOTPRINT_OFFSET : _LIFETIME_MAX_FOOTPRINT_OFFSET + 8], "little"
        ),
        "neural_footprint_bytes": int.from_bytes(
            buf.raw[_NEURAL_FOOTPRINT_V6_OFFSET : _NEURAL_FOOTPRINT_V6_OFFSET + 8], "little"
        ),
        "lifetime_max_neural_footprint_bytes": int.from_bytes(
            buf.raw[_LIFETIME_MAX_NEURAL_FOOTPRINT_V6_OFFSET : _LIFETIME_MAX_NEURAL_FOOTPRINT_V6_OFFSET + 8],
            "little",
        ),
    }


def _scan_processes_rusage_v6(
    names: frozenset[str],
    *,
    excluded_pids: frozenset[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan matching executable basenames via ``libproc``; return ``(readable_readings, inaccessible_pids)``."""
    if platform.system() != "Darwin":
        return [], []
    libproc = _load_libproc()
    excluded = excluded_pids if excluded_pids is not None else frozenset({os.getpid()})
    cap = max(64, int(libproc.proc_listallpids(None, 0)) + 64)
    pid_arr = (ctypes.c_int * cap)()
    cnt = int(libproc.proc_listallpids(pid_arr, ctypes.sizeof(pid_arr)))
    path_buf = ctypes.create_string_buffer(_PROC_PIDPATHINFO_MAXSIZE)

    readable: list[dict[str, Any]] = []
    inaccessible: list[dict[str, Any]] = []
    for idx in range(max(0, min(cnt, cap))):
        pid = int(pid_arr[idx])
        if pid <= 0 or pid in excluded:
            continue
        if int(libproc.proc_pidpath(pid, path_buf, _PROC_PIDPATHINFO_MAXSIZE)) <= 0:
            continue
        proc_name = os.path.basename(path_buf.value.decode("utf-8", errors="ignore"))
        if proc_name not in names:
            continue
        ru = read_process_rusage_v6(pid)
        if ru is None:
            inaccessible.append({"pid": pid, "name": proc_name})
        else:
            readable.append(
                {
                    "pid": pid,
                    "name": proc_name,
                    "footprint_bytes": int(ru["phys_footprint_bytes"]),
                    "rss_bytes": int(ru["rss_bytes"]),
                    "neural_footprint_bytes": int(ru["neural_footprint_bytes"]),
                }
            )
    return readable, inaccessible


class AneDaemonMemoryProbe:
    """Track ``RUSAGE_INFO_V6`` RSS & footprint deltas for out-of-process ANE daemons.

    Watches ``aned``, ``ANECompilerService``, and ``ANELargeModelCompilerService`` by default.
    Without root privileges, macOS returns ``EPERM`` for root-owned services; those PIDs are
    reported explicitly under ``inaccessible_pids`` with ``permission_limited=True``.
    """

    def __init__(
        self,
        names: Sequence[str] = tuple(sorted(ANE_DAEMON_PROCESS_NAMES)),
        *,
        reader: Callable[[frozenset[str]], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.names: frozenset[str] = frozenset(str(n) for n in names)
        self._reader = reader or (lambda ns: _scan_processes_rusage_v6(ns))
        init_readings, init_inacc = self._reader(self.names)
        self._baseline_footprint: dict[int, int] = {int(r["pid"]): int(r["footprint_bytes"]) for r in init_readings}
        self._baseline_rss: dict[int, int] = {int(r["pid"]): int(r["rss_bytes"]) for r in init_readings}
        self._initial_inaccessible = list(init_inacc)

    def poll(self) -> dict[str, Any]:
        """Sample watched daemons and return current totals plus positive growth above baseline."""
        readings, inaccessible = self._reader(self.names)
        procs: list[dict[str, Any]] = []
        tot_fp = 0
        tot_rss = 0
        tot_fp_delta = 0
        tot_rss_delta = 0
        for r in readings:
            pid = int(r["pid"])
            fp = int(r["footprint_bytes"])
            rss = int(r["rss_bytes"])
            fp_base = self._baseline_footprint.get(pid, 0)
            rss_base = self._baseline_rss.get(pid, 0)
            fp_delta = max(0, fp - fp_base)
            rss_delta = max(0, rss - rss_base)
            tot_fp += fp
            tot_rss += rss
            tot_fp_delta += fp_delta
            tot_rss_delta += rss_delta
            procs.append(
                {
                    "pid": pid,
                    "name": str(r["name"]),
                    "footprint_bytes": fp,
                    "rss_bytes": rss,
                    "footprint_delta_bytes": fp_delta,
                    "rss_delta_bytes": rss_delta,
                }
            )
        return {
            "processes": procs,
            "total_footprint_bytes": tot_fp,
            "total_rss_bytes": tot_rss,
            "total_footprint_delta_bytes": tot_fp_delta,
            "total_rss_delta_bytes": tot_rss_delta,
            "inaccessible_pids": inaccessible,
            "permission_limited": bool(inaccessible),
        }


class FourBucketMemoryProbe:
    """Opt-in, zero-private-SPI 4-bucket macOS unified-memory probe.

    Captures:
      1. ``phys_footprint``: process dirty/anonymous CPU + Metal/IOSurface (``RUSAGE_INFO_V4``).
      2. ``neural_nofootprint``: ``task_vm_info.ledger_tag_neural_nofootprint`` (plus ``external``).
      3. ``mapped_files``: clean file-backed ``mmap`` resident pages (optional ``MS_SYNC|MS_INVALIDATE`` eviction).
      4. ``ane_daemons``: ``RUSAGE_INFO_V6`` RSS & footprint deltas for ``aned`` / ``ANECompilerService`` / ``ANELargeModelCompilerService``.
    """

    def __init__(
        self,
        *,
        model_directories: Sequence[str | Path] | None = None,
        daemon_names: Sequence[str] = tuple(sorted(ANE_DAEMON_PROCESS_NAMES)),
        daemon_reader: Callable[[frozenset[str]], tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
        evict_baseline: bool = False,
    ) -> None:
        self.model_directories = _normalize_model_prefixes(model_directories)
        self._daemon_probe = AneDaemonMemoryProbe(names=daemon_names, reader=daemon_reader)
        if evict_baseline and self.model_directories:
            evict_mapped_files(self.model_directories)
        self.baseline = self.snapshot(evict_mapped=False)

    def snapshot(self, *, evict_mapped: bool = False) -> dict[str, Any]:
        """Capture all 4 memory buckets (optionally evicting clean model page-cache pages first)."""
        evict_stats: dict[str, int | bool] | None = None
        if evict_mapped and self.model_directories:
            evict_stats = evict_mapped_files(self.model_directories)

        fp = get_unified_memory_footprint()
        vm = probe_task_vm_info()
        mapped = probe_mapped_files(self.model_directories or None)
        daemons = self._daemon_probe.poll()
        rusage_v6 = read_process_rusage_v6(os.getpid())

        phys_b = int(fp["phys_footprint_bytes"])
        neural_nf_b = max(0, int(vm["ledger_tag_neural_nofootprint_bytes"]))
        model_map_b = int(mapped["model_bytes"])
        all_map_b = int(mapped["total_bytes"])
        daemon_delta_b = int(daemons["total_footprint_delta_bytes"])

        peak_phys_b = max(
            int(fp.get("lifetime_max_phys_footprint_bytes", phys_b)),
            int(rusage_v6["lifetime_max_phys_footprint_bytes"]) if rusage_v6 is not None else 0,
        )
        peak_neural_b = max(
            neural_nf_b,
            int(rusage_v6["lifetime_max_neural_footprint_bytes"]) if rusage_v6 is not None else 0,
        )

        combined_model_b = phys_b + neural_nf_b + model_map_b + daemon_delta_b
        combined_all_b = phys_b + neural_nf_b + all_map_b + daemon_delta_b
        # Lifetime-max phys_footprint already captures mapped pages when they were resident at peak,
        # so do not double-count current clean mapped_files onto lifetime_max_phys_footprint_bytes.
        peak_combined_model_b = max(combined_model_b, peak_phys_b + peak_neural_b)

        return {
            "phys_footprint": fp,
            "task_vm_info": vm,
            "mapped_files": {
                "available": mapped["available"],
                "total_bytes": all_map_b,
                "model_bytes": model_map_b,
                "shared_bytes": int(mapped["shared_bytes"]),
                "file_count": len(mapped["resident_by_path"]),
                "model_directories": mapped["model_directories"],
                "eviction": evict_stats,
            },
            "ane_daemons": daemons,
            "peak_footprint_gb": round(peak_phys_b / 1e9, 4),
            "combined_with_model_mapped_bytes": combined_model_b,
            "combined_with_all_mapped_bytes": combined_all_b,
            "combined_with_model_mapped_gb": round(combined_model_b / 1e9, 4),
            "combined_with_all_mapped_gb": round(combined_all_b / 1e9, 4),
            "peak_combined_with_model_mapped_bytes": peak_combined_model_b,
            "peak_combined_with_model_mapped_gb": round(peak_combined_model_b / 1e9, 4),
        }

    def delta(self, *, evict_mapped: bool = False) -> dict[str, Any]:
        """Take a current 4-bucket snapshot and report deltas against baseline."""
        cur = self.snapshot(evict_mapped=evict_mapped)
        base = self.baseline
        phys_delta = int(cur["phys_footprint"]["phys_footprint_bytes"]) - int(
            base["phys_footprint"]["phys_footprint_bytes"]
        )
        neural_delta = int(cur["task_vm_info"]["ledger_tag_neural_nofootprint_bytes"]) - int(
            base["task_vm_info"]["ledger_tag_neural_nofootprint_bytes"]
        )
        model_map_delta = int(cur["mapped_files"]["model_bytes"]) - int(base["mapped_files"]["model_bytes"])
        all_map_delta = int(cur["mapped_files"]["total_bytes"]) - int(base["mapped_files"]["total_bytes"])
        daemon_fp_delta = int(cur["ane_daemons"]["total_footprint_delta_bytes"])
        daemon_rss_delta = int(cur["ane_daemons"]["total_rss_delta_bytes"])
        return {
            "current": cur,
            "deltas": {
                "phys_footprint_delta_bytes": phys_delta,
                "neural_nofootprint_delta_bytes": neural_delta,
                "model_mapped_delta_bytes": model_map_delta,
                "all_mapped_delta_bytes": all_map_delta,
                "ane_daemon_footprint_delta_bytes": daemon_fp_delta,
                "ane_daemon_rss_delta_bytes": daemon_rss_delta,
            },
        }


def probe_four_bucket_memory(
    *,
    model_directories: Sequence[str | Path] | None = None,
    evict_mapped: bool = False,
) -> dict[str, Any]:
    """One-shot opt-in 4-bucket memory snapshot (``phys_footprint``, ``task_vm_info``, ``mapped_files``, ``ane_daemons``)."""
    probe = FourBucketMemoryProbe(model_directories=model_directories, evict_baseline=False)
    return probe.snapshot(evict_mapped=evict_mapped)


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Point-in-time Metal working-set and process resident memory snapshot."""

    allocated_bytes: int
    recommended_bytes: int
    footprint_bytes: int
    mapped_bytes: int = 0

    @property
    def resident_bytes(self) -> int:
        return int(self.footprint_bytes) + int(self.mapped_bytes)

    @property
    def allocated_fraction(self) -> float:
        return (float(self.allocated_bytes) / float(self.recommended_bytes)) if self.recommended_bytes > 0 else 0.0

    @property
    def resident_fraction(self) -> float:
        return (float(self.resident_bytes) / float(self.recommended_bytes)) if self.recommended_bytes > 0 else 0.0

    def format_summary(self) -> str:
        gib = 1024.0**3
        return (
            f"Metal {self.allocated_bytes / gib:.2f} GiB allocated of "
            f"{self.recommended_bytes / gib:.2f} GiB recommended "
            f"({self.allocated_fraction * 100.0:.1f}%), "
            f"footprint {self.footprint_bytes / gib:.2f} GiB + "
            f"mapped files {self.mapped_bytes / gib:.2f} GiB "
            f"(resident {self.resident_bytes / gib:.2f} GiB)"
        )


def measure_device_memory_snapshot() -> DeviceMemorySnapshot:
    """Measure Metal allocated/recommended working-set bytes plus process footprint & mapped external bytes."""
    allocated = 0
    recommended = 0
    try:
        import mlx.core as mx

        get_active = getattr(mx, "get_active_memory", None)
        get_cache = getattr(mx, "get_cache_memory", None)
        if callable(get_active):
            allocated = int(get_active()) + (int(get_cache()) if callable(get_cache) else 0)
        dev_info_fn = getattr(mx, "device_info", None)
        if callable(dev_info_fn):
            dev = dev_info_fn()
            if isinstance(dev, dict):
                recommended = int(dev.get("max_recommended_working_set_size", 0))
    except Exception:
        pass

    vm = probe_task_vm_info()
    if vm.get("available"):
        footprint = int(vm["phys_footprint_bytes"])
        mapped = int(vm["external_bytes"])
    else:
        fp = get_unified_memory_footprint()
        footprint = int(fp["phys_footprint_bytes"])
        mapped = 0
    return DeviceMemorySnapshot(
        allocated_bytes=allocated,
        recommended_bytes=recommended,
        footprint_bytes=footprint,
        mapped_bytes=mapped,
    )


class DeviceMemoryGuard:
    """``DeviceMemoryGuard``-style Metal working-set guard (``0.98 * recommended``, ``past_arm_margin=0.02``).

    Watches two thresholds once ``arm()`` is called after model load:
      - Metal allocated bytes against ``trip_fraction * recommendedMaxWorkingSetSize`` (default ``0.98``).
      - Process resident bytes (``phys_footprint + mapped_external``) against ``recommendedMaxWorkingSetSize``.

    If a figure is already above its baseline threshold at ``arm()``, its refusal line is placed
    ``past_arm_margin * recommended`` (default ``0.02``) above its armed level so already-loaded
    servers can still run while tripping on further growth. Refusal clears automatically once
    memory falls back under both lines.
    """

    def __init__(
        self,
        *,
        trip_fraction: float = DEFAULT_METAL_TRIP_FRACTION,
        past_arm_margin: float = DEFAULT_PAST_ARM_MARGIN,
        measure: Callable[[], DeviceMemorySnapshot] | None = None,
    ) -> None:
        if not (0.0 < float(trip_fraction) <= 1.0):
            raise ValueError(f"trip_fraction must be in (0, 1], got {trip_fraction}")
        if float(past_arm_margin) < 0.0:
            raise ValueError(f"past_arm_margin must be >= 0, got {past_arm_margin}")
        self.trip_fraction = float(trip_fraction)
        self.past_arm_margin = float(past_arm_margin)
        self._measure = measure or measure_device_memory_snapshot
        self._lock = threading.Lock()
        self._allocated_line: int | None = None
        self._resident_line: int | None = None
        self._refusing: bool = False

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._allocated_line is not None and self._resident_line is not None

    @property
    def lines(self) -> tuple[int, int] | None:
        with self._lock:
            if self._allocated_line is None or self._resident_line is None:
                return None
            return (self._allocated_line, self._resident_line)

    def snapshot(self) -> DeviceMemorySnapshot:
        return self._measure()

    def arm(self) -> DeviceMemorySnapshot:
        """Latch refusal thresholds against current memory; no-op if device reports no recommendation."""
        start = self._measure()
        if start.recommended_bytes <= 0:
            return start
        margin = int(float(start.recommended_bytes) * self.past_arm_margin)
        metal_line = int(float(start.recommended_bytes) * self.trip_fraction)
        alloc_line = metal_line if start.allocated_bytes < metal_line else (start.allocated_bytes + margin)
        res_line = (
            start.recommended_bytes if start.resident_bytes < start.recommended_bytes else (start.resident_bytes + margin)
        )
        with self._lock:
            self._allocated_line = alloc_line
            self._resident_line = res_line
            self._refusing = False
        return start

    def check(self) -> tuple[DeviceMemorySnapshot, str | None]:
        """Measure once and return ``(snapshot, refusal_reason_or_None)``."""
        now = self._measure()
        with self._lock:
            if self._allocated_line is None or self._resident_line is None:
                return now, None
            refusal: str | None = None
            if now.allocated_bytes >= self._allocated_line:
                refusal = (
                    f"Metal allocated memory exceeded guard line "
                    f"({now.allocated_bytes / (1024**3):.2f} >= {self._allocated_line / (1024**3):.2f} GiB; "
                    f"{now.format_summary()})"
                )
            elif now.resident_bytes >= self._resident_line:
                refusal = (
                    f"Process resident memory (footprint + mapped) exceeded guard line "
                    f"({now.resident_bytes / (1024**3):.2f} >= {self._resident_line / (1024**3):.2f} GiB; "
                    f"{now.format_summary()})"
                )
            self._refusing = refusal is not None
            return now, refusal


def check_metal_working_set_headroom(
    *,
    trip_fraction: float = DEFAULT_METAL_TRIP_FRACTION,
    measure: Callable[[], DeviceMemorySnapshot] | None = None,
) -> dict[str, Any]:
    """One-shot preflight check against ``trip_fraction * recommendedMaxWorkingSetSize`` (default ``0.98``)."""
    snap = (measure or measure_device_memory_snapshot)()
    warning: str | None = None
    if snap.recommended_bytes > 0:
        metal_line = int(float(snap.recommended_bytes) * float(trip_fraction))
        if snap.allocated_bytes >= metal_line:
            warning = (
                f"Metal allocated {snap.allocated_bytes / (1024**3):.2f} GiB >= "
                f"{trip_fraction * 100.0:.0f}% of recommended {snap.recommended_bytes / (1024**3):.2f} GiB"
            )
        elif snap.resident_bytes >= snap.recommended_bytes:
            warning = (
                f"Process resident {snap.resident_bytes / (1024**3):.2f} GiB >= "
                f"recommended {snap.recommended_bytes / (1024**3):.2f} GiB ({snap.format_summary()})"
            )
    return {
        "ok": warning is None,
        "warning": warning,
        "allocated_fraction": round(snap.allocated_fraction, 4),
        "resident_fraction": round(snap.resident_fraction, 4),
        "snapshot": {
            "allocated_bytes": snap.allocated_bytes,
            "recommended_bytes": snap.recommended_bytes,
            "footprint_bytes": snap.footprint_bytes,
            "mapped_bytes": snap.mapped_bytes,
            "resident_bytes": snap.resident_bytes,
        },
    }


@contextmanager
def wired_limit(limit_bytes: int | None = None) -> Iterator[int | None]:
    """Context manager applying `mx.set_wired_limit(...)` and restoring the previous limit on exit.

    When `limit_bytes` is `None`, uses `mx.device_info()["max_recommended_working_set_size"]`.
    Yields the applied limit in bytes, or `None` if MLX / `set_wired_limit` is unavailable.
    """
    try:
        import mlx.core as mx
    except ImportError:
        yield None
        return

    set_wired = getattr(mx, "set_wired_limit", None)
    if not callable(set_wired):
        yield None
        return

    if limit_bytes is None:
        try:
            dev = mx.device_info()
            target = int(dev.get("max_recommended_working_set_size", 0)) if isinstance(dev, dict) else 0
        except Exception:
            target = 0
    else:
        target = int(limit_bytes)

    if target <= 0:
        yield None
        return

    prev = set_wired(target)
    try:
        yield target
    finally:
        if prev is not None:
            try:
                set_wired(int(prev))
            except Exception:
                pass


def collect_env_telemetry() -> dict[str, Any]:
    """Collect host silicon, OS/build, Xcode SDK, MLX version, NAX, and TF32 benchmark telemetry."""
    hw = probe_host_hardware()
    return {
        "brand": hw.get("brand"),
        "chip_family": hw.get("chip_family"),
        "hw_model": hw.get("hw_model"),
        "macos_version": hw.get("macos_version"),
        "macos_build": hw.get("macos_build"),
        "xcode_sdk_version": hw.get("xcode_sdk_version"),
        "python_version": platform.python_version(),
        "mlx_version": hw.get("mlx_version"),
        "mlx_architecture": hw.get("mlx_architecture"),
        "gpu_family": hw.get("gpu_family"),
        "gpu_cores": hw.get("gpu_cores"),
        "ane_cores": hw.get("ane_cores"),
        "cpu_cores": hw.get("cpu_cores"),
        "cpu_topology": hw.get("cpu_topology"),
        "mem_gb": hw.get("mem_gb"),
        "thunderbolt": hw.get("thunderbolt"),
        "supports_apple10_nax": supports_apple10_nax(hw),
        "supports_metal4_bf16": supports_metal4_bf16(hw),
        "supports_placement_sparse": hw.get("supports_placement_sparse"),
        "mlx_enable_tf32": hw.get("mlx_enable_tf32"),
        "mlx_enable_tf32_env": hw.get("mlx_enable_tf32_env"),
        "mlx_metal_no_nax": hw.get("mlx_metal_no_nax"),
    }


def _classify_quiescence_line(
    line: str,
    *,
    ignore_pids: frozenset[int] | set[int] = frozenset(),
) -> str | None:
    """Return a formatted warning if a `ps aux` line indicates active GPU/CPU contention."""
    parts = line.split(None, 10)
    if len(parts) < 11:
        return None
    if ignore_pids:
        try:
            if int(parts[1]) in ignore_pids:
                return None
        except ValueError:
            pass
    try:
        cpu = float(parts[2])
    except ValueError:
        cpu = 0.0
    cmd = parts[10]
    if "ab_bench_harness.py" in cmd or "fast-bench-check" in cmd:
        return None
    low = line.lower()
    cmd_base = os.path.basename(cmd.split()[0]).lower() if cmd.split() else ""

    is_gpu_browser = (
        ("chrome" in low or "microsoft edge" in low or "msedge" in low or "/arc.app/" in low or cmd_base == "arc")
        and ("gpu" in low or "helper" in low or cpu >= 10.0)
        and cpu >= 0.5
    )
    is_other_browser = ("firefox" in low or "safari" in low) and cpu >= 15.0
    is_ml_runner = ("mlx" in low or "pytest" in low or "llama" in low or "coreml" in low) and cpu >= 10.0
    is_spotlight = cmd_base in ("mds", "mds_stores", "mdworker", "mdworker_shared") and cpu >= 5.0
    is_window_server = cmd_base == "windowserver" and cpu >= 15.0
    is_backup = cmd_base == "backupd" and cpu >= 5.0
    is_build_tool = cmd_base in ("xcodebuild", "clang", "clang++", "swift-frontend", "metal") and cpu >= 10.0

    if is_gpu_browser or is_other_browser or is_ml_runner or is_spotlight or is_window_server or is_backup or is_build_tool or cpu >= 40.0:
        return f"PID={parts[1]} CPU={cpu:.1f}% RSS_KB={parts[5]} :: {cmd[:120]}"
    return None


def check_system_quiescence() -> list[str]:
    """Return active GPU/CPU contention warning lines from `ps aux` (empty if clear)."""
    out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    self_pids = frozenset({os.getpid(), os.getppid()})
    warnings: list[str] = []
    for line in out.splitlines()[1:]:
        warn = _classify_quiescence_line(line, ignore_pids=self_pids)
        if warn is not None:
            warnings.append(warn)
    return warnings


def clear_mlx_arena(threshold_mib: int = 256) -> None:
    """Release cached MLX wave allocations once `mx.get_cache_memory()` exceeds `threshold_mib`."""
    try:
        import mlx.core as mx
    except ImportError:
        if threshold_mib <= 0:
            gc.collect()
        return

    get_cache = getattr(mx, "get_cache_memory", None)
    clear_cache = getattr(mx, "clear_cache", None)
    if threshold_mib <= 0:
        gc.collect()
        if callable(clear_cache):
            clear_cache()
        return
    if callable(get_cache) and get_cache() > (threshold_mib << 20):
        gc.collect()
        if callable(clear_cache):
            clear_cache()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute float64 cosine similarity between two tensors."""
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    denom = max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1e-12)
    return float(np.dot(af, bf) / denom)


def check_tensor_dtypes(
    named_arrays: Sequence[tuple[str, Any]],
    *,
    allow_fp32_substrings: Sequence[str] = ("sink", "rope", "inv_freq", "norm"),
    expected_dtype: str | None = None,
    check_masks: bool = True,
) -> list[str]:
    """Flag unexpected FP32 tensors, int/bool attention masks, and fp16/bf16 mismatches."""
    flags: list[str] = []
    low_float_seen: dict[str, list[str]] = {"float16": [], "bfloat16": []}

    for name, arr in named_arrays:
        dt_str = str(getattr(arr, "dtype", ""))
        dt_low = dt_str.lower()
        name_low = name.lower()
        shape = getattr(arr, "shape", None)

        if "float32" in dt_low and not any(tok in name_low for tok in allow_fp32_substrings):
            flags.append(f"{name}: {dt_str} shape={shape}")
            continue

        if check_masks and "mask" in name_low:
            if "bool" in dt_low or any(tok in dt_low for tok in ("int8", "int16", "int32", "int64", "uint")):
                flags.append(f"{name}: mask uses {dt_str} (prefer native fp16/bf16 additive mask) shape={shape}")
                continue

        if "bfloat16" in dt_low:
            low_float_seen["bfloat16"].append(name)
            if expected_dtype == "float16":
                flags.append(f"{name}: {dt_str} mismatches expected float16 shape={shape}")
        elif "float16" in dt_low:
            low_float_seen["float16"].append(name)
            if expected_dtype == "bfloat16":
                flags.append(f"{name}: {dt_str} mismatches expected bfloat16 shape={shape}")

    if expected_dtype is None and low_float_seen["float16"] and low_float_seen["bfloat16"]:
        flags.append(
            f"mixed fp16/bf16 tensors: float16={low_float_seen['float16']} vs bfloat16={low_float_seen['bfloat16']}"
        )
    return flags


def interleaved_abba(
    fn_a: Callable[[], dict[str, Any]],
    fn_b: Callable[[], dict[str, Any]],
    *,
    warmup: bool = True,
    rounds: int = 2,
    cooldown_sec: float = 0.0,
    use_wired_limit: bool = False,
    wired_limit_bytes: int | None = None,
    include_telemetry: bool = True,
) -> dict[str, Any]:
    """Run `fn_a` and `fn_b` in interleaved `A-B-B-A` order to cancel linear thermal drift."""
    import mlx.core as mx

    ctx = wired_limit(wired_limit_bytes) if use_wired_limit else _nullcontext()
    with ctx:
        if warmup:
            _ = fn_a()
            mx.synchronize()
            _ = fn_b()
            mx.synchronize()
            clear_mlx_arena(0)

        a_runs: list[dict[str, Any]] = []
        b_runs: list[dict[str, Any]] = []
        order = ["A", "B", "B", "A"] * max(1, rounds // 2)
        if rounds % 2 == 1:
            order.extend(["A", "B"])

        for idx, tag in enumerate(order):
            if idx > 0 and cooldown_sec > 0.0:
                time.sleep(cooldown_sec)
            mx.reset_peak_memory()
            mx.synchronize()
            t0 = time.perf_counter()
            res = dict(fn_a() if tag == "A" else fn_b())
            mx.synchronize()
            wall = round(time.perf_counter() - t0, 4)
            res.setdefault("wall_sec", wall)
            res.setdefault("peak_gb", round(mx.get_peak_memory() / (1024**3), 2))
            clear_mlx_arena(256)
            (a_runs if tag == "A" else b_runs).append(res)

    def _agg(runs: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return {k: round(float(np.median([r[k] for r in runs])), 4) for k in keys}

    def _drift(runs: list[dict[str, Any]]) -> dict[str, float | None]:
        if len(runs) < 2:
            return {"ratio": None, "delta_pct": None}
        first = float(runs[0].get("wall_sec", 0.0))
        last = float(runs[-1].get("wall_sec", 0.0))
        if first <= 0.0:
            return {"ratio": None, "delta_pct": None}
        ratio = last / first
        return {"ratio": round(ratio, 4), "delta_pct": round((ratio - 1.0) * 100.0, 2)}

    a_drift = _drift(a_runs)
    b_drift = _drift(b_runs)

    out: dict[str, Any] = {
        "order": order,
        "A_runs": a_runs,
        "B_runs": b_runs,
        "A_median": _agg(a_runs),
        "B_median": _agg(b_runs),
        "drift": {
            "A_last_over_first": a_drift["ratio"],
            "A_drift_pct": a_drift["delta_pct"],
            "B_last_over_first": b_drift["ratio"],
            "B_drift_pct": b_drift["delta_pct"],
        },
        "peak_footprint_gb": round(peak_footprint_gb(), 3),
    }
    if include_telemetry:
        out["env_telemetry"] = collect_env_telemetry()
    return out


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield None


def collect_daemon_four_bucket_metrics(
    probe: FourBucketMemoryProbe | None = None,
    *,
    model_directories: Sequence[str | Path] | None = None,
    pid: int | None = None,
    evict_mapped: bool = False,
) -> dict[str, Any]:
    """Collect subprocess/daemon 4-bucket unified-memory and ``RUSAGE_INFO_V6`` metrics.

    Intended for isolated Core ML / Core AI worker subprocesses (``pid is None`` or ``os.getpid()``)
    or parent observers inspecting a running child ``pid`` before shutdown.
    """
    target_pid = os.getpid() if pid is None else int(pid)
    in_process = target_pid == os.getpid()
    rusage = read_process_rusage_v6(target_pid)

    four_bucket_cur: dict[str, Any] | None = None
    four_bucket_deltas: dict[str, Any] | None = None
    if probe is not None:
        d = probe.delta(evict_mapped=evict_mapped)
        four_bucket_cur = d["current"]
        four_bucket_deltas = d["deltas"]
    elif in_process:
        four_bucket_cur = probe_four_bucket_memory(
            model_directories=model_directories,
            evict_mapped=evict_mapped,
        )

    mib = 1024.0 * 1024.0
    rss_bytes = int(rusage["rss_bytes"]) if rusage is not None else 0
    phys_bytes = int(rusage["phys_footprint_bytes"]) if rusage is not None else 0
    max_phys_bytes = int(rusage["lifetime_max_phys_footprint_bytes"]) if rusage is not None else 0
    neural_rusage_bytes = int(rusage["neural_footprint_bytes"]) if rusage is not None else 0
    max_neural_rusage_bytes = int(rusage["lifetime_max_neural_footprint_bytes"]) if rusage is not None else 0

    vm_info = four_bucket_cur.get("task_vm_info", {}) if (four_bucket_cur and in_process) else {}
    mapped_info = four_bucket_cur.get("mapped_files", {}) if (four_bucket_cur and in_process) else {}
    daemon_info = four_bucket_cur.get("ane_daemons", {}) if four_bucket_cur else {}

    if not in_process:
        neural_nf_bytes = 0
        mapped_model_bytes = 0
        mapped_all_bytes = 0
    else:
        neural_nf_bytes = max(0, int(vm_info.get("ledger_tag_neural_nofootprint_bytes", 0)))
        mapped_model_bytes = int(mapped_info.get("model_bytes", 0))
        mapped_all_bytes = int(mapped_info.get("total_bytes", 0))

    daemon_fp_delta_bytes = int(daemon_info.get("total_footprint_delta_bytes", 0))
    daemon_rss_delta_bytes = int(daemon_info.get("total_rss_delta_bytes", 0))
    combined_model_bytes = phys_bytes + neural_nf_bytes + mapped_model_bytes + daemon_fp_delta_bytes
    peak_neural_bytes = max(neural_nf_bytes, max_neural_rusage_bytes)
    peak_combined_model_bytes = max(combined_model_bytes, max_phys_bytes + peak_neural_bytes)

    return {
        "pid": target_pid,
        "in_process": in_process,
        "rusage_version": 6 if rusage is not None else None,
        "phys_footprint_mb": round(phys_bytes / mib, 2),
        "lifetime_max_phys_footprint_mb": round(max_phys_bytes / mib, 2),
        "resident_size_mb": round(rss_bytes / mib, 2),
        "neural_footprint_mb": round(neural_rusage_bytes / mib, 2),
        "lifetime_max_neural_footprint_mb": round(max_neural_rusage_bytes / mib, 2),
        "neural_nofootprint_mb": round(neural_nf_bytes / mib, 2),
        "mapped_model_mb": round(mapped_model_bytes / mib, 2),
        "mapped_all_mb": round(mapped_all_bytes / mib, 2),
        "ane_daemon_footprint_delta_mb": round(daemon_fp_delta_bytes / mib, 2),
        "ane_daemon_rss_delta_mb": round(daemon_rss_delta_bytes / mib, 2),
        "ane_daemon_permission_limited": bool(daemon_info.get("permission_limited", False)),
        "combined_with_model_mapped_mb": round(combined_model_bytes / mib, 2),
        "combined_with_model_mapped_gb": round(combined_model_bytes / 1e9, 4),
        "peak_combined_with_model_mapped_bytes": peak_combined_model_bytes,
        "peak_combined_with_model_mapped_mb": round(peak_combined_model_bytes / mib, 2),
        "peak_combined_with_model_mapped_gb": round(peak_combined_model_bytes / 1e9, 4),
        "four_bucket_deltas": four_bucket_deltas,
        "ane_daemons": daemon_info if daemon_info else None,
    }


EXIT_SKIP: int = 3


def _coerce_subprocess_text(val: bytes | str | None) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


def _tail_lines(text: str | None, max_lines: int = 20) -> list[str]:
    if not text or max_lines <= 0:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-max_lines:]


def write_worker_json(
    payload: Any,
    *,
    worker_out_file: str | Path | None = None,
    marker: str | None = None,
) -> None:
    """Atomically write worker JSON to ``worker_out_file`` when given, or emit ``marker`` to stdout."""
    serialized = json.dumps(payload, ensure_ascii=False)
    if worker_out_file is not None and str(worker_out_file).strip():
        dest = Path(worker_out_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.tmp.", dir=str(dest.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
                fp.write(serialized)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_path, dest)
            return
        finally:
            tmp_path.unlink(missing_ok=True)
    line = f"{marker}{serialized}" if marker else serialized
    print(line, flush=True)


def _extract_marked_json(text: str, markers: Sequence[str]) -> Any | None:
    if not text:
        return None
    for line in reversed(text.splitlines()):
        for marker in markers:
            if not marker:
                continue
            idx = line.find(marker)
            if idx < 0:
                continue
            candidate = line[idx + len(marker) :].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                for offset, ch in enumerate(candidate):
                    if ch in "{[":
                        try:
                            obj, _ = decoder.raw_decode(candidate, offset)
                            return obj
                        except json.JSONDecodeError:
                            continue
    return None


def _kill_process_group(proc: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        rem_out, rem_err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        rem_out, rem_err = "", ""
    return _coerce_subprocess_text(rem_out), _coerce_subprocess_text(rem_err)


def run_isolated_json_worker(
    cmd: Sequence[str],
    *,
    worker_out_arg: str | None = "--worker-out-file",
    worker_out_file: str | Path | None = None,
    marker: str | None = None,
    markers: Sequence[str] | None = None,
    timeout_sec: float = 300.0,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    skip_exit_codes: Sequence[int] = (EXIT_SKIP,),
    stderr_tail_lines: int = 20,
) -> dict[str, Any]:
    """Run an isolated worker subprocess and extract its structured JSON result.

    Launches in a fresh session (``start_new_session=True``), kills the full
    process group with ``os.killpg(..., SIGKILL)`` on timeout, supports atomic
    ``--worker-out-file`` output with substring marker fallback, handles
    ``EXIT_SKIP == 3`` cleanly, and returns a 20-line stderr tail on exit.
    """
    eff_markers: tuple[str, ...] = tuple(
        m for m in ([marker] if marker else []) + list(markers or ()) if m
    )
    eff_env = dict(os.environ)
    if env:
        eff_env.update({str(k): str(v) for k, v in env.items()})

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    out_path: Path | None = Path(worker_out_file) if worker_out_file is not None else None
    worker_cmd = list(cmd)
    if worker_out_arg:
        if worker_out_arg in worker_cmd:
            idx = worker_cmd.index(worker_out_arg)
            if out_path is None and idx + 1 < len(worker_cmd):
                out_path = Path(worker_cmd[idx + 1])
        else:
            if out_path is None:
                tmp_ctx = tempfile.TemporaryDirectory(prefix="fs_worker_")
                out_path = Path(tmp_ctx.name) / "worker_out.json"
            worker_cmd.extend([worker_out_arg, str(out_path)])

    try:
        proc = subprocess.Popen(
            worker_cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=eff_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out_s, err_s = proc.communicate(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired as exc:
            rem_out, rem_err = _kill_process_group(proc)
            full_out = _coerce_subprocess_text(exc.stdout) + rem_out
            full_err = _coerce_subprocess_text(exc.stderr) + rem_err
            return {
                "status": "timeout",
                "returncode": proc.returncode,
                "payload": None,
                "source": None,
                "timed_out": True,
                "stderr_tail": _tail_lines(full_err, stderr_tail_lines),
                "stdout": full_out,
                "stderr": full_err,
            }

        out_s = _coerce_subprocess_text(out_s)
        err_s = _coerce_subprocess_text(err_s)
        rc = int(proc.returncode or 0)

        payload: Any | None = None
        source: str | None = None
        parse_error: str | None = None
        if out_path is not None and out_path.is_file():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                source = "out_file"
            except Exception as exc:
                parse_error = f"invalid worker_out_file JSON: {exc}"
        if payload is None and eff_markers:
            for stream_text in (out_s, err_s):
                payload = _extract_marked_json(stream_text, eff_markers)
                if payload is not None:
                    source = "marker"
                    break

        stderr_tail = _tail_lines(err_s, stderr_tail_lines)
        if rc in set(skip_exit_codes):
            status = "skipped"
        elif rc == 0 and payload is not None:
            status = "ok"
        else:
            status = "failed"
        res: dict[str, Any] = {
            "status": status,
            "returncode": rc,
            "payload": payload,
            "source": source,
            "timed_out": False,
            "stderr_tail": stderr_tail,
            "stdout": out_s,
            "stderr": err_s,
        }
        if parse_error is not None and status == "failed":
            res["parse_error"] = parse_error
        return res
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def summarize_cold_warm_latencies(
    *,
    cold_load_ms: float | None = None,
    cold_first_infer_ms: float | None = None,
    warm_latencies_ms: Sequence[float] = (),
) -> dict[str, Any]:
    """Summarize separate cold model-load/specialization cost vs. warm steady-state latencies."""
    warm = [float(x) for x in warm_latencies_ms if float(x) >= 0.0]
    cold_load = round(float(cold_load_ms), 3) if cold_load_ms is not None else None
    cold_first = round(float(cold_first_infer_ms), 3) if cold_first_infer_ms is not None else None
    cold_total: float | None = None
    if cold_load is not None or cold_first is not None:
        cold_total = round((cold_load or 0.0) + (cold_first or 0.0), 3)

    if warm:
        arr = np.asarray(warm, dtype=np.float64)
        warm_med = round(float(np.median(arr)), 3)
        warm_p95 = round(float(np.percentile(arr, 95)), 3)
        warm_min = round(float(np.min(arr)), 3)
        warm_max = round(float(np.max(arr)), 3)
    else:
        warm_med = None
        warm_p95 = None
        warm_min = None
        warm_max = None

    cold_to_warm_ratio: float | None = None
    if warm_med is not None and warm_med > 0.0:
        cold_ref = cold_first if cold_first is not None else cold_total
        if cold_ref is not None:
            cold_to_warm_ratio = round(cold_ref / warm_med, 3)

    return {
        "cold_load_ms": cold_load,
        "cold_first_infer_ms": cold_first,
        "cold_total_ms": cold_total,
        "warm_count": len(warm),
        "warm_median_ms": warm_med,
        "warm_p95_ms": warm_p95,
        "warm_min_ms": warm_min,
        "warm_max_ms": warm_max,
        "cold_to_warm_ratio": cold_to_warm_ratio,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `fast-bench-check` / `ab_bench_harness.py`."""
    p = argparse.ArgumentParser(description="Benchmark quiescence check & ABBA harness utilities.")
    p.add_argument("--check-ps", action="store_true", help="Inspect running processes for GPU/CPU contention")
    p.add_argument("--telemetry", action="store_true", help="Print host/OS/SDK/MLX benchmark telemetry JSON")
    args = p.parse_args(argv)

    if args.telemetry:
        import json

        print(json.dumps(collect_env_telemetry(), indent=2))

    if args.check_ps or not args.telemetry:
        warns = check_system_quiescence()
        if not warns:
            print("QUIESCENT: no heavy GPU/CPU contenders detected.")
        else:
            print(f"CONTENTION WARNING ({len(warns)} process(es)):")
            for w in warns:
                print(f"  {w}")


__all__ = [
    "ANE_DAEMON_PROCESS_NAMES",
    "AneDaemonMemoryProbe",
    "DEFAULT_METAL_TRIP_FRACTION",
    "DEFAULT_PAST_ARM_MARGIN",
    "DeviceMemoryGuard",
    "DeviceMemorySnapshot",
    "EXIT_SKIP",
    "FourBucketMemoryProbe",
    "check_metal_working_set_headroom",
    "check_system_quiescence",
    "check_tensor_dtypes",
    "clear_mlx_arena",
    "collect_daemon_four_bucket_metrics",
    "collect_env_telemetry",
    "cos_sim",
    "evict_mapped_files",
    "get_unified_memory_footprint",
    "interleaved_abba",
    "main",
    "measure_device_memory_snapshot",
    "peak_footprint_gb",
    "probe_four_bucket_memory",
    "probe_mapped_files",
    "probe_task_vm_info",
    "read_process_rusage_v6",
    "run_isolated_json_worker",
    "summarize_cold_warm_latencies",
    "wired_limit",
    "write_worker_json",
]


if __name__ == "__main__":
    main()
