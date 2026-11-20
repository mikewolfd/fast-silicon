"""CPU unit tests for 4-bucket unified-memory telemetry and Metal working-set guard in `fast_shared.bench`."""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import platform

import pytest

from fast_shared.bench import (
    ANE_DAEMON_PROCESS_NAMES,
    DEFAULT_METAL_TRIP_FRACTION,
    DEFAULT_PAST_ARM_MARGIN,
    AneDaemonMemoryProbe,
    DeviceMemoryGuard,
    DeviceMemorySnapshot,
    FourBucketMemoryProbe,
    check_metal_working_set_headroom,
    collect_daemon_four_bucket_metrics,
    evict_mapped_files,
    get_unified_memory_footprint,
    measure_device_memory_snapshot,
    probe_four_bucket_memory,
    probe_mapped_files,
    probe_task_vm_info,
    read_process_rusage_v6,
    summarize_cold_warm_latencies,
)


def test_unified_memory_footprint_docstring_accurately_notes_mmap_exclusion_cpu() -> None:
    doc = (get_unified_memory_footprint.__doc__ or "").lower()
    assert "excludes" in doc
    assert "mmap" in doc
    assert "ledger_tag_neural_nofootprint" in doc
    assert "aned" in doc


def test_probe_task_vm_info_and_rusage_v6_self_cpu() -> None:
    vm = probe_task_vm_info()
    v6 = read_process_rusage_v6(os.getpid())
    if platform.system() == "Darwin":
        assert vm["available"] is True
        assert int(vm["phys_footprint_bytes"]) > 0
        assert int(vm["external_bytes"]) >= 0
        assert isinstance(vm["ledger_tag_neural_nofootprint_bytes"], int)
        assert v6 is not None
        assert v6["pid"] == os.getpid()
        assert v6["rss_bytes"] > 0
        assert v6["phys_footprint_bytes"] > 0
        assert v6["lifetime_max_phys_footprint_bytes"] >= v6["phys_footprint_bytes"]
    else:
        assert vm["available"] is False
        assert v6 is None


def test_probe_mapped_files_and_msync_eviction_roundtrip_cpu(tmp_path: Path) -> None:
    if platform.system() != "Darwin":
        res = probe_mapped_files([tmp_path])
        assert res["available"] is False
        assert res["total_bytes"] == 0
        return

    blob_bytes = 4 * 1024 * 1024  # 4 MiB clean file-backed mmap
    weight_file = tmp_path / "mapped_embed.bin"
    weight_file.write_bytes(b"\x5a" * blob_bytes)
    resolved_weight = os.path.realpath(str(weight_file))

    with open(weight_file, "rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # Fault every page into the process's resident page tables.
            _ = mm[::4096]
            before = probe_mapped_files([tmp_path])
            assert before["available"] is True
            assert before["model_bytes"] >= blob_bytes
            assert resolved_weight in before["resident_by_path"]
            assert before["resident_by_path"][resolved_weight] >= blob_bytes
            assert before["total_bytes"] == before["model_bytes"] + before["shared_bytes"]

            ev = evict_mapped_files([tmp_path])
            assert ev["available"] is True
            assert int(ev["regions_evicted"]) >= 1
            assert int(ev["bytes_requested"]) >= blob_bytes
            assert int(ev["failed_regions"]) == 0

            after = probe_mapped_files([tmp_path])
            assert after["model_bytes"] < before["model_bytes"]
            assert after["model_bytes"] == 0
        finally:
            mm.close()


def test_ane_daemon_memory_probe_delta_and_permission_limited_cpu() -> None:
    assert ANE_DAEMON_PROCESS_NAMES == frozenset({"aned", "ANECompilerService", "ANELargeModelCompilerService"})

    state: dict[str, int] = {"step": 0}

    def _fake_reader(names: frozenset[str]):
        assert "aned" in names
        if state["step"] == 0:
            return (
                [
                    {"pid": 100, "name": "aned", "footprint_bytes": 50_000_000, "rss_bytes": 40_000_000},
                    {"pid": 101, "name": "ANECompilerService", "footprint_bytes": 20_000_000, "rss_bytes": 18_000_000},
                ],
                [{"pid": 999, "name": "ANELargeModelCompilerService"}],
            )
        return (
            [
                # PID 100 grew by +120 MB footprint / +60 MB RSS
                {"pid": 100, "name": "aned", "footprint_bytes": 170_000_000, "rss_bytes": 100_000_000},
                # PID 101 shrank -> clamped to 0 delta
                {"pid": 101, "name": "ANECompilerService", "footprint_bytes": 10_000_000, "rss_bytes": 12_000_000},
                # PID 202 newly spawned -> counts in full!
                {"pid": 202, "name": "ANECompilerService", "footprint_bytes": 80_000_000, "rss_bytes": 75_000_000},
            ],
            [{"pid": 999, "name": "ANELargeModelCompilerService"}],
        )

    probe = AneDaemonMemoryProbe(reader=_fake_reader)
    state["step"] = 1
    polled = probe.poll()

    assert polled["total_footprint_delta_bytes"] == 120_000_000 + 80_000_000
    assert polled["total_rss_delta_bytes"] == 60_000_000 + 75_000_000
    assert polled["total_footprint_bytes"] == 170_000_000 + 10_000_000 + 80_000_000
    assert polled["permission_limited"] is True
    assert polled["inaccessible_pids"] == [{"pid": 999, "name": "ANELargeModelCompilerService"}]


def test_four_bucket_memory_probe_accounting_and_deltas_cpu(tmp_path: Path) -> None:
    daemon_step = {"n": 0}

    def _daemon_reader(_names: frozenset[str]):
        if daemon_step["n"] == 0:
            return ([{"pid": 500, "name": "aned", "footprint_bytes": 10_000_000, "rss_bytes": 9_000_000}], [])
        return ([{"pid": 500, "name": "aned", "footprint_bytes": 35_000_000, "rss_bytes": 24_000_000}], [])

    probe = FourBucketMemoryProbe(model_directories=[tmp_path], daemon_reader=_daemon_reader)
    weight_file = tmp_path / "weights.bin"
    blob_bytes = 2 * 1024 * 1024
    weight_file.write_bytes(b"\x3c" * blob_bytes)

    with open(weight_file, "rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            _ = mm[::4096]
            daemon_step["n"] = 1
            rep = probe.delta(evict_mapped=False)
            cur = rep["current"]
            deltas = rep["deltas"]

            assert deltas["ane_daemon_footprint_delta_bytes"] == 25_000_000
            assert deltas["ane_daemon_rss_delta_bytes"] == 15_000_000
            if platform.system() == "Darwin":
                assert cur["mapped_files"]["model_bytes"] >= blob_bytes
                assert deltas["model_mapped_delta_bytes"] >= blob_bytes
                expected_model_total = (
                    int(cur["phys_footprint"]["phys_footprint_bytes"])
                    + max(0, int(cur["task_vm_info"]["ledger_tag_neural_nofootprint_bytes"]))
                    + int(cur["mapped_files"]["model_bytes"])
                    + 25_000_000
                )
                assert cur["combined_with_model_mapped_bytes"] == expected_model_total
                assert cur["combined_with_all_mapped_bytes"] >= cur["combined_with_model_mapped_bytes"]

                evicted_snap = probe.snapshot(evict_mapped=True)
                assert evicted_snap["mapped_files"]["eviction"] is not None
                assert evicted_snap["mapped_files"]["model_bytes"] == 0
        finally:
            mm.close()

    one_shot = probe_four_bucket_memory(model_directories=[tmp_path])
    assert set(one_shot.keys()) >= {
        "phys_footprint",
        "task_vm_info",
        "mapped_files",
        "ane_daemons",
        "combined_with_model_mapped_bytes",
        "combined_with_all_mapped_bytes",
    }


def test_device_memory_guard_trip_past_arm_margin_and_recovery_cpu() -> None:
    assert DEFAULT_METAL_TRIP_FRACTION == pytest.approx(0.98)
    assert DEFAULT_PAST_ARM_MARGIN == pytest.approx(0.02)

    gib = 1024**3
    rec = 16 * gib  # 16 GiB recommended working set (e.g. 24 GiB M5)
    cur_snap = [DeviceMemorySnapshot(allocated_bytes=4 * gib, recommended_bytes=rec, footprint_bytes=2 * gib, mapped_bytes=4 * gib)]
    guard = DeviceMemoryGuard(measure=lambda: cur_snap[0])

    # Unarmed guard never refuses even if over limit.
    cur_snap[0] = DeviceMemorySnapshot(allocated_bytes=rec, recommended_bytes=rec, footprint_bytes=rec, mapped_bytes=0)
    _, refusal = guard.check()
    assert refusal is None
    assert not guard.armed

    # Arm under normal load -> lines at 0.98 * rec (Metal) and 1.00 * rec (resident = footprint + mapped).
    cur_snap[0] = DeviceMemorySnapshot(allocated_bytes=4 * gib, recommended_bytes=rec, footprint_bytes=2 * gib, mapped_bytes=4 * gib)
    guard.arm()
    assert guard.armed
    assert guard.lines == (int(rec * 0.98), rec)
    _, refusal = guard.check()
    assert refusal is None

    # Trip on Metal allocated >= 0.98 * rec.
    cur_snap[0] = DeviceMemorySnapshot(allocated_bytes=int(rec * 0.98), recommended_bytes=rec, footprint_bytes=2 * gib, mapped_bytes=4 * gib)
    snap, refusal = guard.check()
    assert refusal is not None and "Metal allocated" in refusal
    assert snap.allocated_fraction == pytest.approx(0.98)

    # Recover automatically once Metal drops back below line.
    cur_snap[0] = DeviceMemorySnapshot(allocated_bytes=int(rec * 0.90), recommended_bytes=rec, footprint_bytes=2 * gib, mapped_bytes=4 * gib)
    _, refusal = guard.check()
    assert refusal is None

    # Trip on resident = footprint + mapped >= rec even when Metal allocated is small (mmap blind spot!).
    cur_snap[0] = DeviceMemorySnapshot(allocated_bytes=2 * gib, recommended_bytes=rec, footprint_bytes=5 * gib, mapped_bytes=11 * gib)
    _, refusal = guard.check()
    assert refusal is not None and "resident memory" in refusal

    # Already past line at arm() -> adds past_arm_margin (0.02 * rec) above armed level.
    start_over = DeviceMemorySnapshot(
        allocated_bytes=int(rec * 0.99),
        recommended_bytes=rec,
        footprint_bytes=int(rec * 0.60),
        mapped_bytes=int(rec * 0.45),
    )
    cur_snap[0] = start_over
    guard_over = DeviceMemoryGuard(measure=lambda: cur_snap[0])
    guard_over.arm()
    margin = int(rec * 0.02)
    assert guard_over.lines == (start_over.allocated_bytes + margin, start_over.resident_bytes + margin)

    # +0.01 * rec growth is still admitted; +0.02 * rec trips refusal.
    cur_snap[0] = DeviceMemorySnapshot(
        allocated_bytes=start_over.allocated_bytes + int(rec * 0.01),
        recommended_bytes=rec,
        footprint_bytes=start_over.footprint_bytes,
        mapped_bytes=start_over.mapped_bytes,
    )
    assert guard_over.check()[1] is None

    cur_snap[0] = DeviceMemorySnapshot(
        allocated_bytes=start_over.allocated_bytes + margin,
        recommended_bytes=rec,
        footprint_bytes=start_over.footprint_bytes,
        mapped_bytes=start_over.mapped_bytes,
    )
    assert guard_over.check()[1] is not None

    # Recommended == 0 -> arm() is a no-op and never refuses.
    zero_guard = DeviceMemoryGuard(
        measure=lambda: DeviceMemorySnapshot(allocated_bytes=10 * gib, recommended_bytes=0, footprint_bytes=10 * gib)
    )
    zero_guard.arm()
    assert not zero_guard.armed
    assert zero_guard.check()[1] is None

    # One-shot headroom check + live smoke check.
    ok_res = check_metal_working_set_headroom(
        measure=lambda: DeviceMemorySnapshot(allocated_bytes=8 * gib, recommended_bytes=rec, footprint_bytes=4 * gib, mapped_bytes=2 * gib)
    )
    assert ok_res["ok"] is True and ok_res["warning"] is None

    warn_res = check_metal_working_set_headroom(
        measure=lambda: DeviceMemorySnapshot(allocated_bytes=int(rec * 0.985), recommended_bytes=rec, footprint_bytes=4 * gib, mapped_bytes=2 * gib)
    )
    assert warn_res["ok"] is False and warn_res["warning"] is not None

    live_snap = measure_device_memory_snapshot()
    assert live_snap.footprint_bytes >= 0
    assert live_snap.mapped_bytes >= 0


def test_collect_daemon_four_bucket_metrics_and_cold_warm_split_cpu() -> None:
    import subprocess
    import sys

    step = [0]

    def _fake_reader(names: frozenset[str]):
        step[0] += 1
        base = 20 * 1024 * 1024
        growth = 0 if step[0] == 1 else 64 * 1024 * 1024
        return ([{"pid": 4242, "name": "aned", "footprint_bytes": base + growth, "rss_bytes": base + growth}], [])

    probe = FourBucketMemoryProbe(daemon_reader=_fake_reader)
    self_metrics = collect_daemon_four_bucket_metrics(probe)
    assert self_metrics["in_process"] is True
    assert self_metrics["ane_daemon_footprint_delta_mb"] == pytest.approx(64.0, abs=0.01)
    assert self_metrics["ane_daemon_rss_delta_mb"] == pytest.approx(64.0, abs=0.01)
    if platform.system() == "Darwin":
        assert self_metrics["rusage_version"] == 6
        assert self_metrics["phys_footprint_mb"] > 0.0
        assert self_metrics["lifetime_max_phys_footprint_mb"] >= self_metrics["phys_footprint_mb"]

    # Remote PID mode must never attribute parent self task_vm_info/mapped_files to the child.
    remote_metrics = collect_daemon_four_bucket_metrics(probe, pid=-1)
    assert remote_metrics["in_process"] is False
    assert remote_metrics["neural_nofootprint_mb"] == 0.0
    assert remote_metrics["mapped_model_mb"] == 0.0
    assert remote_metrics["ane_daemon_footprint_delta_mb"] == pytest.approx(64.0, abs=0.01)

    split = summarize_cold_warm_latencies(
        cold_load_ms=420.0,
        cold_first_infer_ms=45.0,
        warm_latencies_ms=[10.0, 12.0, 20.0],
    )
    assert split["cold_load_ms"] == 420.0
    assert split["cold_first_infer_ms"] == 45.0
    assert split["cold_total_ms"] == 465.0
    assert split["warm_count"] == 3
    assert split["warm_median_ms"] == 12.0
    assert split["cold_to_warm_ratio"] == pytest.approx(3.75, abs=0.001)

    empty_split = summarize_cold_warm_latencies(cold_load_ms=100.0, warm_latencies_ms=[])
    assert empty_split["warm_count"] == 0
    assert empty_split["warm_median_ms"] is None
    assert empty_split["cold_to_warm_ratio"] is None

    # Verify fast_shared.bench / hw_check / ir_dtype_guard import cleanly without mlx or pyarrow.
    env = dict(os.environ)
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = f"{src_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_root
    code = (
        "import sys\n"
        "for m in ('mlx', 'mlx.core', 'pyarrow', 'pyarrow.compute', 'pyarrow.parquet'):\n"
        "    sys.modules[m] = None\n"
        "import fast_shared\n"
        "from fast_shared.bench import collect_daemon_four_bucket_metrics, summarize_cold_warm_latencies\n"
        "from fast_shared.hw_check import summarize_accelerator_placement\n"
        "try:\n"
        "    _ = fast_shared.make_mpp_matmul_kernel\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('Expected ImportError when mlx missing')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"stderr={res.stderr}"
