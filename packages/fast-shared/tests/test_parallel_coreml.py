"""CPU unit tests for crash-safe, `spawn`-only multi-process CoreML bucket compilation."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pytest

from fast_shared.parallel_coreml import (
    DEFAULT_MAX_COMPILE_WORKERS,
    SPAWN_START_METHOD,
    BucketCompileError,
    BucketCompileResult,
    atomic_publish_dir,
    compile_buckets_parallel,
    default_compile_jobs,
    get_spawn_context,
    resolve_compile_workers,
)


# ---------------------------------------------------------------------------
# Top-level picklable worker callables for `spawn` multiprocessing tests
# ---------------------------------------------------------------------------


def _worker_staggered_square(task: tuple[int, float]) -> dict[str, Any]:
    """Sleep `delay` seconds (so later tasks finish first) and return env/tmp metadata."""
    val, delay = task
    if delay > 0.0:
        time.sleep(delay)
    tmpdir_env = os.environ.get("TMPDIR", "")
    marker = Path(tmpdir_env) / f"marker_{val}.txt"
    marker.write_text("ok", encoding="utf-8")
    return {
        "val": val,
        "sq": val * val,
        "pid": os.getpid(),
        "tmpdir": tmpdir_env,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "tok_par": os.environ.get("TOKENIZERS_PARALLELISM"),
        "custom_env": os.environ.get("FAST_COREML_TEST_TAG"),
    }


def _worker_raise_on_target(task: tuple[int, int]) -> int:
    idx, fail_idx = task
    if idx == fail_idx:
        raise ValueError(f"synthetic compile failure at bucket {idx}")
    time.sleep(0.02)
    return idx * 10


def _worker_hard_exit_on_target(task: tuple[int, int]) -> int:
    idx, crash_idx = task
    if idx == crash_idx:
        os._exit(99)
    time.sleep(0.05)
    return idx + 1


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_default_compile_jobs_and_worker_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_MAX_COMPILE_WORKERS == 4

    monkeypatch.setattr(os, "cpu_count", lambda: 14)
    assert default_compile_jobs() == 4
    assert default_compile_jobs(None, cap=6) == 6
    assert default_compile_jobs(2) == 2
    assert default_compile_jobs(8) == 8

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert default_compile_jobs() == 2

    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_compile_jobs() == 1

    assert resolve_compile_workers(0, 4) == 1
    assert resolve_compile_workers(2, 4) == 2
    assert resolve_compile_workers(9, 4) == 4
    assert resolve_compile_workers(9, 1) == 1

    for bad in (0, -1, True, False):
        with pytest.raises(ValueError):
            default_compile_jobs(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        default_compile_jobs(None, cap=0)


def test_spawn_context_invariant_no_fork() -> None:
    ctx = get_spawn_context()
    assert ctx.get_start_method() == SPAWN_START_METHOD == "spawn"

    src_path = Path(__file__).resolve().parents[1] / "src" / "fast_shared" / "parallel_coreml.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_context":
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    assert arg.value == "spawn", f"Forbidden multiprocessing start method: {arg.value!r}"


def test_compile_buckets_parallel_empty_and_single_worker() -> None:
    assert compile_buckets_parallel([], _worker_staggered_square, max_workers=4) == []

    completed: list[str] = []
    res = compile_buckets_parallel(
        [(3, 0.0), (5, 0.0)],
        _worker_staggered_square,
        max_workers=1,
        label_fn=lambda t: f"b_{t[0]}",
        on_complete=lambda r: completed.append(r.label),
    )
    assert [r.label for r in res] == ["b_3", "b_5"]
    assert completed == ["b_3", "b_5"]
    assert [r.value["sq"] for r in res] == [9, 25]
    assert all(r.pid == os.getpid() for r in res)
    # Isolated per-task TMPDIR must be cleaned up after completion
    for r in res:
        assert r.value["tmpdir"]
        assert not Path(r.value["tmpdir"]).exists()


def test_compile_buckets_parallel_spawn_deterministic_order_and_isolation() -> None:
    # Earlier tasks sleep longer so later tasks finish first; results must still preserve input order.
    tasks = [
        (0, 0.16),
        (1, 0.12),
        (2, 0.08),
        (3, 0.04),
        (4, 0.01),
        (5, 0.00),
    ]
    seen_labels: list[str] = []
    results = compile_buckets_parallel(
        tasks,
        _worker_staggered_square,
        max_workers=3,
        label_fn=lambda t: f"s{(t[0] + 1) * 64}",
        on_complete=lambda r: seen_labels.append(r.label),
        env_overrides={"FAST_COREML_TEST_TAG": "spawn_ok"},
        extra_sys_paths=(Path(__file__).resolve().parent,),
    )

    assert len(results) == len(tasks)
    assert [r.index for r in results] == list(range(len(tasks)))
    assert [r.label for r in results] == [f"s{(i + 1) * 64}" for i in range(len(tasks))]
    assert [r.value["sq"] for r in results] == [i * i for i in range(len(tasks))]
    assert set(seen_labels) == {r.label for r in results}

    parent_pid = os.getpid()
    worker_pids = {r.pid for r in results}
    assert parent_pid not in worker_pids
    assert len(worker_pids) >= 2

    tmpdirs = [r.value["tmpdir"] for r in results]
    assert len(set(tmpdirs)) == len(tasks)
    for r in results:
        assert r.value["omp_threads"] == "1"
        assert r.value["tok_par"] == "false"
        assert r.value["custom_env"] == "spawn_ok"
        assert not Path(r.value["tmpdir"]).exists()


def test_compile_buckets_parallel_force_spawn_single_worker() -> None:
    results = compile_buckets_parallel(
        [(7, 0.0)],
        _worker_staggered_square,
        max_workers=1,
        force_spawn=True,
        extra_sys_paths=(Path(__file__).resolve().parent,),
    )
    assert len(results) == 1
    assert isinstance(results[0], BucketCompileResult)
    assert results[0].value["sq"] == 49
    assert results[0].pid != os.getpid()


def test_compile_buckets_parallel_exception_handling_single_and_spawn() -> None:
    for workers in (1, 2):
        with pytest.raises(BucketCompileError) as exc_info:
            compile_buckets_parallel(
                [(0, 1), (1, 1), (2, 1), (3, 1)],
                _worker_raise_on_target,
                max_workers=workers,
                label_fn=lambda t: f"bucket_{t[0]}",
                extra_sys_paths=(Path(__file__).resolve().parent,),
            )
        err = exc_info.value
        assert len(err.failures) >= 1
        fail = err.failures[0]
        assert fail.index == 1
        assert fail.label == "bucket_1"
        assert fail.error_type == "ValueError"
        assert "synthetic compile failure at bucket 1" in fail.message
        assert "ValueError" in fail.traceback_text


def test_compile_buckets_parallel_worker_crash_recovery() -> None:
    with pytest.raises(BucketCompileError) as exc_info:
        compile_buckets_parallel(
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            _worker_hard_exit_on_target,
            max_workers=2,
            label_fn=lambda t: f"crash_b{t[0]}",
            extra_sys_paths=(Path(__file__).resolve().parent,),
        )
    err = exc_info.value
    assert len(err.failures) >= 1
    assert err.failures[0].error_type == "BrokenProcessPool"


def test_atomic_publish_dir_replaces_and_preserves_on_failure(tmp_path: Path) -> None:
    dst = tmp_path / "model.mlmodelc"
    dst.mkdir()
    (dst / "old.bin").write_text("old-data", encoding="utf-8")

    # Missing source must preserve existing destination intact
    with pytest.raises(FileNotFoundError):
        atomic_publish_dir(tmp_path / "missing_src", dst, move=True)
    assert (dst / "old.bin").read_text(encoding="utf-8") == "old-data"

    # Successful copy publish replaces destination and leaves zero staging/backup dirs
    src_copy = tmp_path / "compiled_copy"
    src_copy.mkdir()
    (src_copy / "model.mil").write_text("v2", encoding="utf-8")
    out = atomic_publish_dir(src_copy, dst, move=False)
    assert out == dst
    assert src_copy.is_dir()
    assert not (dst / "old.bin").exists()
    assert (dst / "model.mil").read_text(encoding="utf-8") == "v2"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".model.mlmodelc")]
    assert leftovers == []

    # Move publish consumes source cleanly
    src_move = tmp_path / "compiled_move"
    src_move.mkdir()
    (src_move / "model.mil").write_text("v3", encoding="utf-8")
    atomic_publish_dir(src_move, dst, move=True)
    assert not src_move.exists()
    assert (dst / "model.mil").read_text(encoding="utf-8") == "v3"

    # Symlink destination replaced cleanly without leaving backup symlink behind
    sym_target = tmp_path / "sym_target"
    sym_target.mkdir()
    sym_dst = tmp_path / "sym.mlmodelc"
    sym_dst.symlink_to(sym_target)
    src_sym = tmp_path / "compiled_sym"
    src_sym.mkdir()
    (src_sym / "ok.txt").write_text("yes", encoding="utf-8")
    atomic_publish_dir(src_sym, sym_dst, move=True)
    assert not sym_dst.is_symlink()
    assert (sym_dst / "ok.txt").read_text(encoding="utf-8") == "yes"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".sym.mlmodelc")] == []


def test_monorepo_builder_scripts_wire_jobs_and_safetensors_loader(tmp_path: Path) -> None:
    """Verify P0.4 `--jobs` CLI wiring across sibling CoreML builder scripts when present."""
    pkgs_root = Path(__file__).resolve().parents[2]
    bge_script = pkgs_root / "fast-bge-small" / "scripts" / "build_ane_models.py"
    dec_script = pkgs_root / "fast-decision-models" / "scripts" / "build_decision_models.py"
    g4_script = pkgs_root / "fast-gemma4" / "scripts" / "build_ane_models.py"

    if not (bge_script.is_file() and dec_script.is_file() and g4_script.is_file()):
        pytest.skip("Sibling monorepo builder scripts not colocated in standalone checkout")

    for script_p in (bge_script, dec_script, g4_script):
        src = script_p.read_text(encoding="utf-8")
        assert "compile_buckets_parallel" in src, f"Missing parallel_coreml wiring in {script_p}"
        assert '"--jobs"' in src or "'--jobs'" in src, f"Missing --jobs flag in {script_p}"

    # Verify BGE bucket parser
    bge_spec = importlib.util.spec_from_file_location("bge_build_test_mod", bge_script)
    assert bge_spec and bge_spec.loader
    bge_mod = importlib.util.module_from_spec(bge_spec)
    bge_spec.loader.exec_module(bge_mod)
    assert bge_mod._parse_bucket_specs("1x24, 8x256") == [(1, 24), (8, 256)]
    with pytest.raises(ValueError):
        bge_mod._parse_bucket_specs("0x64")

    # Verify Gemma-4 chunk parser & signature
    g4_spec = importlib.util.spec_from_file_location("g4_build_test_mod", g4_script)
    assert g4_spec and g4_spec.loader
    g4_mod = importlib.util.module_from_spec(g4_spec)
    g4_spec.loader.exec_module(g4_mod)
    assert g4_mod._parse_chunk_sizes("128,256", 256) == (128, 256)
    with pytest.raises(ValueError):
        g4_mod._parse_chunk_sizes("1", 256)

    # Verify pure-NumPy safetensors loader in build_decision_models.py (F32, F16, BF16)
    dec_spec = importlib.util.spec_from_file_location("dec_build_test_mod", dec_script)
    assert dec_spec and dec_spec.loader
    dec_mod = importlib.util.module_from_spec(dec_spec)
    dec_spec.loader.exec_module(dec_mod)

    f32_arr = np.array([[1.25, -2.5], [3.75, 0.5]], dtype=np.float32)
    f16_arr = np.array([0.5, -1.5, 4.0], dtype=np.float16)
    bf16_src = np.array([1.0, -3.5, 2.25, 0.125], dtype=np.float32)
    bf16_u16 = (bf16_src.view(np.uint32) >> 16).astype("<u2")

    raw_f32 = f32_arr.astype("<f4").tobytes()
    raw_f16 = f16_arr.astype("<f2").tobytes()
    raw_bf16 = bf16_u16.tobytes()
    o0 = 0
    o1 = o0 + len(raw_f32)
    o2 = o1 + len(raw_f16)
    o3 = o2 + len(raw_bf16)
    hdr = {
        "__metadata__": {"format": "pt"},
        "w_f32": {"dtype": "F32", "shape": [2, 2], "data_offsets": [o0, o1]},
        "w_f16": {"dtype": "F16", "shape": [3], "data_offsets": [o1, o2]},
        "w_bf16": {"dtype": "BF16", "shape": [4], "data_offsets": [o2, o3]},
    }
    hdr_bytes = json.dumps(hdr).encode("utf-8")
    sf_path = tmp_path / "tiny.safetensors"
    sf_path.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + raw_f32 + raw_f16 + raw_bf16)

    loaded = dec_mod._load_safetensors_as_float32_np(sf_path)
    assert set(loaded) == {"w_f32", "w_f16", "w_bf16"}
    assert np.allclose(loaded["w_f32"], f32_arr)
    assert np.allclose(loaded["w_f16"], f16_arr.astype(np.float32))
    assert np.allclose(loaded["w_bf16"], bf16_src)
