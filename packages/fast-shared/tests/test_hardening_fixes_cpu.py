"""CPU unit tests for Gate 1/2 hardening, worker IPC, ETL manifest/index safety, server SSE/batching, and PrefixTree."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_gate1_repeat_4gram_distinguishes_boilerplate_from_loops() -> None:
    from fast_shared.outcome_eval.chat_reasoning import check_chat_validity, repeat_4gram_rate
    from fast_shared.outcome_eval.metrics import evaluate_promotion_gates

    # 1. Valid CORD-v2 style JSON array with repeated field names across distinct items must not false-flag.
    cord_items = [
        {"nm": f"Dish {idx} Special", "cnt": "1", "price": f"{idx + 3}.{10 + idx}"}
        for idx in range(18)
    ]
    cord_json = json.dumps({"menu": cord_items, "total": {"total_price": "199.00"}})
    assert repeat_4gram_rate(cord_json) < 0.15
    ok_schema, rep_schema = check_chat_validity(cord_json, finish_reason="stop", valid_schema=True)
    assert ok_schema is True and rep_schema < 0.15

    # 2. Valid long MATH-500 solution with scattered repeating notation must not false-flag.
    math_steps = [
        f"In step {i} we substitute u = {i * 3 + 1} into \\frac{{1}}{{2}} and obtain discriminant {i * i + 7} after simplifying."
        for i in range(16)
    ]
    math_text = " ".join(math_steps) + " Therefore the final answer is \\boxed{42}."
    assert repeat_4gram_rate(math_text) < 0.15
    ok_math, rep_math = check_chat_validity(math_text, finish_reason="stop")
    assert ok_math is True and rep_math < 0.15

    # 3. Genuine loops (stride 1..4, multi-token cycle 5..64, tail collapse, and finish_reason="length") are caught.
    assert repeat_4gram_rate("repeat " * 12) >= 0.50
    assert repeat_4gram_rate("alpha beta gamma " * 8) >= 0.50
    assert repeat_4gram_rate("one two three four five six " * 5) >= 0.50

    tail_loop_text = " ".join(f"preamble_token_{i}_unique" for i in range(250)) + " " + ("stuck loop phrase here " * 25)
    assert repeat_4gram_rate(tail_loop_text) >= 0.15

    # Length truncation + elevated non-contiguous repetition triggers Gate 1 check cleanly.
    ok_len, rep_len = check_chat_validity(math_text + " \\frac{1}{2} " * 4, finish_reason="length")
    assert ok_len is False and rep_len >= 0.15

    # 4. evaluate_promotion_gates(..., profile="lossless") gates against baseline.repeat_loop_count.
    base_rec = {
        "category": 3,
        "objective_score_e2e": 0.75,
        "n_items": 4,
        "per_item_effective_scores": [1.0, 1.0, 0.0, 1.0],
        "effective_outcome_per_sec": 100.0,
        "diagnostics": {"mask_fallback_rate": 0.0, "repeat_loop_count": 3},
    }
    cand_same = {
        "category": 3,
        "objective_score_e2e": 0.75,
        "n_items": 4,
        "per_item_effective_scores": [1.0, 1.0, 0.0, 1.0],
        "effective_outcome_per_sec": 105.0,
        "diagnostics": {"mask_fallback_rate": 0.0, "repeat_loop_count": 3},
    }
    ev_same = evaluate_promotion_gates(base_rec, cand_same, profile="lossless", min_eot_ratio=1.0)
    assert ev_same.gate1_quality_pass is True and ev_same.verdict == "PASS"

    cand_worse = dict(cand_same, diagnostics={"mask_fallback_rate": 0.0, "repeat_loop_count": 4})
    ev_worse = evaluate_promotion_gates(base_rec, cand_worse, profile="lossless", min_eot_ratio=1.0)
    assert ev_worse.gate1_quality_pass is False and ev_worse.verdict == "FAIL"
    assert any("repeat_loop_count=4 > baseline=3" in r for r in ev_worse.reasons)


def test_gate2_peak_memory_probe_and_isolated_worker_ipc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_shared.bench as bench
    from fast_shared.bench import (
        EXIT_SKIP,
        FourBucketMemoryProbe,
        run_isolated_json_worker,
        write_worker_json,
    )
    from fast_shared.outcome_eval.metrics import evaluate_memory_gate

    # 1. FourBucketMemoryProbe.snapshot computes peak_combined_with_model_mapped_gb without double-counting model_bytes.
    monkeypatch.setattr(
        bench,
        "get_unified_memory_footprint",
        lambda pid=None: {
            "phys_footprint_bytes": 4_000_000_000,
            "lifetime_max_phys_footprint_bytes": 24_000_000_000,
            "rss_bytes": 4_000_000_000,
            "phys_footprint_gb": 4.0,
            "peak_footprint_gb": 24.0,
        },
    )
    monkeypatch.setattr(
        bench,
        "read_process_rusage_v6",
        lambda pid: {
            "pid": pid,
            "rss_bytes": 4_000_000_000,
            "phys_footprint_bytes": 4_000_000_000,
            "lifetime_max_phys_footprint_bytes": 24_500_000_000,
            "neural_footprint_bytes": 500_000_000,
            "lifetime_max_neural_footprint_bytes": 2_500_000_000,
        },
    )
    monkeypatch.setattr(
        bench,
        "probe_task_vm_info",
        lambda: {"available": True, "ledger_tag_neural_nofootprint_bytes": 500_000_000},
    )
    monkeypatch.setattr(
        bench,
        "probe_mapped_files",
        lambda dirs=None: {
            "available": True,
            "total_bytes": 6_000_000_000,
            "model_bytes": 6_000_000_000,
            "shared_bytes": 0,
            "resident_by_path": {},
            "model_directories": (),
        },
    )
    probe = FourBucketMemoryProbe(daemon_reader=lambda names: ([], []), evict_baseline=False)
    snap = probe.snapshot()
    # Peak = max(4 + 0.5 + 6.0 = 10.5GB, 24.5 + 2.5 = 27.0GB) -> 27.0GB (model_bytes 6.0GB NOT double-counted on 24.5GB).
    assert snap["combined_with_model_mapped_gb"] == pytest.approx(10.5, abs=1e-4)
    assert snap["peak_combined_with_model_mapped_gb"] == pytest.approx(27.0, abs=1e-4)

    # evaluate_memory_gate uses max(combined_with_model_mapped_gb, peak_footprint_gb, peak_combined_with_model_mapped_gb)
    ok_peak, ceil_gb, eff_gb, reasons = evaluate_memory_gate(snap, host_mem_gb=32.0, max_host_fraction=0.80)
    assert ok_peak is False
    assert ceil_gb == pytest.approx(25.6)
    assert eff_gb == pytest.approx(27.0)
    assert reasons

    ok_low, _, eff_low, _ = evaluate_memory_gate(
        {"combined_with_model_mapped_gb": 10.5, "peak_footprint_gb": 27.0},
        host_mem_gb=32.0,
    )
    assert ok_low is False and eff_low == pytest.approx(27.0)

    # 2. run_isolated_json_worker: out-file mode, marker fallback, EXIT_SKIP, timeout killpg, and 20-line stderr tail.
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    worker_py = tmp_path / "ipc_worker.py"
    worker_py.write_text(
        "\n".join(
            [
                "import argparse, json, subprocess, sys, time",
                f"sys.path.insert(0, {repo_src!r})",
                "from fast_shared.bench import EXIT_SKIP, write_worker_json",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--mode', required=True)",
                "p.add_argument('--worker-out-file', default=None)",
                "p.add_argument('--pid-file', default=None)",
                "a = p.parse_args()",
                "if a.mode == 'outfile':",
                "    write_worker_json({'ok': True, 'val': 7}, worker_out_file=a.worker_out_file)",
                "elif a.mode == 'marker':",
                "    sys.stdout.write('C++_NOISE_NO_NEWLINE')",
                "    sys.stdout.write('__MK__' + json.dumps({'via': 'marker'}) + '\\n')",
                "elif a.mode == 'skip':",
                "    sys.stderr.write('model not staged\\n')",
                "    sys.exit(EXIT_SKIP)",
                "elif a.mode == 'fail':",
                "    for i in range(30): sys.stderr.write(f'err-line-{i}\\n')",
                "    sys.exit(2)",
                "elif a.mode == 'timeout':",
                "    gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "    with open(a.pid_file, 'w') as fp: fp.write(str(gc.pid))",
                "    sys.stderr.write('before-timeout\\n')",
                "    sys.stderr.flush()",
                "    time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )

    out_json_path = tmp_path / "explicit_out.json"
    res_file = run_isolated_json_worker(
        [sys.executable, str(worker_py), "--mode", "outfile"],
        worker_out_file=out_json_path,
    )
    assert res_file["status"] == "ok" and res_file["source"] == "out_file"
    assert res_file["payload"] == {"ok": True, "val": 7}

    res_marker = run_isolated_json_worker(
        [sys.executable, str(worker_py), "--mode", "marker"],
        marker="__MK__",
    )
    assert res_marker["status"] == "ok" and res_marker["source"] == "marker"
    assert res_marker["payload"] == {"via": "marker"}

    res_skip = run_isolated_json_worker([sys.executable, str(worker_py), "--mode", "skip"])
    assert res_skip["status"] == "skipped" and res_skip["returncode"] == EXIT_SKIP

    res_fail = run_isolated_json_worker([sys.executable, str(worker_py), "--mode", "fail"])
    assert res_fail["status"] == "failed" and res_fail["returncode"] == 2
    assert len(res_fail["stderr_tail"]) == 20
    assert res_fail["stderr_tail"][0] == "err-line-10" and res_fail["stderr_tail"][-1] == "err-line-29"

    pid_file = tmp_path / "gc.pid"
    res_to = run_isolated_json_worker(
        [sys.executable, str(worker_py), "--mode", "timeout", "--pid-file", str(pid_file)],
        timeout_sec=0.4,
    )
    assert res_to["status"] == "timeout" and res_to["timed_out"] is True
    assert "before-timeout" in res_to["stderr_tail"]
    gc_pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 2.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(gc_pid, 0)
            time.sleep(0.05)
        except OSError:
            alive = False
            break
    assert not alive

    # Direct write_worker_json smoke test cleanly replaces target.
    direct_p = tmp_path / "direct.json"
    write_worker_json({"x": 1}, worker_out_file=direct_p)
    assert json.loads(direct_p.read_text(encoding="utf-8")) == {"x": 1}


def test_etl_pipeline_llm_orig_row_idx_manifest_and_dedup(tmp_path: Path) -> None:
    from fast_shared.pipeline_llm import extract_parquet

    in_pq = tmp_path / "llm_in.parquet"
    # Row 0: empty & skipped; Row 1: non-OCR filtered; Row 2..3: kept OCR docs -> orig indices 2 and 3; Row 4: skipped empty
    pq.write_table(
        pa.table(
            {
                "id": [100, 101, 102, 103, 104],
                "is_ocr": [True, False, True, True, True],
                "body_text": ["   ", "ignored-non-ocr", "doc-two", "doc-three", ""],
            }
        ),
        in_pq,
    )

    seen_callable_rows: list[tuple[Any, int]] = []
    schema_a = {"type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"]}
    schema_b = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    seq_grammars = [schema_a, schema_a, schema_b, schema_a, schema_a]

    def _per_doc_cb(row: dict[str, Any]) -> dict[str, Any]:
        seen_callable_rows.append((row["id"], row["_row_index"]))
        return seq_grammars[row["_row_index"]]

    class _IdxTestEngine:
        model_name = "test-llm-v1"

        def generate_batch(self, prompts, **kwargs):
            items = []
            for p in prompts:
                if p == "doc-two":
                    # Succeeds only after retry if invalid on pass 1
                    items.append(
                        SimpleNamespace(
                            text='{"bad": 1}',
                            parsed={"bad": 1},
                            valid_json=True,
                            valid_schema=False,
                            finish_reason="stop",
                            prompt_tokens=3,
                            completion_tokens=2,
                        )
                    )
                else:
                    items.append(
                        SimpleNamespace(
                            text='{"v":"ok"}',
                            parsed={"v": "ok"},
                            valid_json=True,
                            valid_schema=True,
                            finish_reason="stop",
                            prompt_tokens=3,
                            completion_tokens=2,
                        )
                    )
            return SimpleNamespace(items=items, stats={"valid_json_count": len(items), "valid_schema_count": 1})

    deferred: list[Any] = []
    out_dir = tmp_path / "llm_out"
    extract_parquet(
        _IdxTestEngine(),
        in_pq,
        out_dir,
        system_prompt="sys-v1",
        per_doc_grammars=_per_doc_cb,
        filter_ocr=True,
        skip_empty=True,
        defer_retries=True,
        deferred_retries=deferred,
        shard_rows=10,
    )
    assert seen_callable_rows == [(102, 2), (103, 3)]
    assert [req.row_index for req in deferred] == [2]

    # Sequence per_doc_grammars also indexes orig_row_idx without skewing onto row 0/1!
    seen_seq_schemas: list[Any] = []

    class _SeqCheckEngine:
        model_name = "test-llm-v1"

        def generate_batch(self, prompts, **kwargs):
            seen_seq_schemas.append(kwargs.get("json_schema"))
            items = [
                SimpleNamespace(
                    text='{"n":1}',
                    parsed={"n": 1},
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    prompt_tokens=2,
                    completion_tokens=2,
                )
                for _ in prompts
            ]
            return SimpleNamespace(items=items, stats={"valid_json_count": len(items), "valid_schema_count": len(items)})

    extract_parquet(
        _SeqCheckEngine(),
        in_pq,
        tmp_path / "seq_out",
        system_prompt="sys-v1",
        per_doc_grammars=seq_grammars,
        filter_ocr=True,
        skip_empty=True,
        shard_rows=10,
    )
    # Row 2 got schema_b and row 3 got schema_a -> two distinct schemas in sequence mode.
    assert seen_seq_schemas != [schema_a, schema_a]

    # Resume retry must NOT leave duplicate primary key id=102 across shards!
    class _Pass2FixEngine:
        model_name = "test-llm-v1"

        def __init__(self) -> None:
            self.seen_prompts: list[str] = []

        def generate_batch(self, prompts, **kwargs):
            self.seen_prompts.extend(prompts)
            items = [
                SimpleNamespace(
                    text='{"n":2}',
                    parsed={"n": 2},
                    valid_json=True,
                    valid_schema=True,
                    finish_reason="stop",
                    prompt_tokens=2,
                    completion_tokens=2,
                )
                for _ in prompts
            ]
            return SimpleNamespace(items=items, stats={"valid_json_count": len(items), "valid_schema_count": len(items)})

    eng2 = _Pass2FixEngine()
    extract_parquet(
        eng2,
        in_pq,
        out_dir,
        system_prompt="sys-v1",
        per_doc_grammars=_per_doc_cb,
        filter_ocr=True,
        skip_empty=True,
        resume=True,
        shard_rows=10,
    )
    assert eng2.seen_prompts == ["doc-two"]
    all_rows = pa.concat_tables([pq.read_table(p) for p in sorted(out_dir.glob("part-*.parquet"))])
    ids = all_rows.column("id").to_pylist()
    assert sorted(ids) == [102, 103] and len(ids) == 2

    # Fingerprint mismatch fails closed on resume (changed prompt or model_name).
    with pytest.raises(ValueError, match="manifest mismatch"):
        extract_parquet(
            eng2,
            in_pq,
            out_dir,
            system_prompt="sys-v2-changed",
            resume=True,
        )

    class _OtherModelEngine(_Pass2FixEngine):
        model_name = "test-llm-v2"

    with pytest.raises(ValueError, match="manifest mismatch"):
        extract_parquet(
            _OtherModelEngine(),
            in_pq,
            out_dir,
            system_prompt="sys-v1",
            resume=True,
        )

    # Corrupt .extract_manifest.json fails closed instead of silently ignoring.
    (out_dir / ".extract_manifest.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt extract resume manifest"):
        extract_parquet(eng2, in_pq, out_dir, system_prompt="sys-v1", resume=True)


def test_etl_parquet_pipeline_manifest_fingerprint_and_early_skip(tmp_path: Path) -> None:
    import numpy as np
    from fast_shared.parquet_pipeline import _load_manifest, _MANIFEST_FILENAME, embed_parquet
    from fast_shared.tokenizer import BaseFastTokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "alpha": 4, "beta": 5, "gamma": 6}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = WhitespaceSplit()
    tok_obj.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    tok_json = tmp_path / "tok.json"
    tok_obj.save(str(tok_json))
    tok_wrap = BaseFastTokenizer(tok_json, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    tok_wrap.name_or_path = "dummy-tok-v1"  # type: ignore[attr-defined]

    class _EmbedSpyEngine:
        dim = 4

        def __init__(self, model_name: str = "embed-v1") -> None:
            self.model_name = model_name
            self.tokenizer = tok_wrap
            self.max_length = 32
            self.encode_calls: list[int] = []

        def encode_tokens(self, tok_lists, **kwargs):
            n = len(tok_lists)
            self.encode_calls.append(n)
            return np.ones((n, 4), dtype=np.float32)

    in_pq = tmp_path / "embed_in.parquet"
    pq.write_table(
        pa.table({"id": list(range(6)), "body": ["alpha beta" for _ in range(6)]}),
        in_pq,
    )
    out_dir = tmp_path / "embed_out"
    eng1 = _EmbedSpyEngine("embed-v1")
    embed_parquet(
        eng1,
        in_pq,
        out_dir,
        batch_rows=2,
        shard_rows=2,
        dimensions=4,
        dtype="float32",
        sort_macro_window=False,
        progress_every=0,
    )

    # _load_manifest fails closed on corrupt or non-dict JSON.
    manifest_p = out_dir / _MANIFEST_FILENAME
    assert manifest_p.is_file()
    bad_p = tmp_path / "bad_manifest.json"
    bad_p.write_text("{broken json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        _load_manifest(bad_p)
    bad_p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        _load_manifest(bad_p)

    # Corrupt output manifest fails closed during resume=True.
    saved_manifest = manifest_p.read_text(encoding="utf-8")
    manifest_p.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        embed_parquet(eng1, in_pq, out_dir, batch_rows=2, shard_rows=2, dimensions=4, dtype="float32", resume=True, progress_every=0)
    manifest_p.write_text(saved_manifest, encoding="utf-8")

    # Deleting shard 2 with resume=True skips shards 0 & 1 BEFORE tokenizing their rows!
    (out_dir / "part-00002.parquet").unlink()
    tok_encoded_counts: list[int] = []
    orig_encode_batch = tok_wrap.encode_batch

    def _spy_encode_batch(texts, **kwargs):
        tok_encoded_counts.append(len(texts))
        return orig_encode_batch(texts, **kwargs)

    tok_wrap.encode_batch = _spy_encode_batch  # type: ignore[method-assign]
    eng2 = _EmbedSpyEngine("embed-v1")
    s_res = embed_parquet(
        eng2,
        in_pq,
        out_dir,
        batch_rows=2,
        shard_rows=2,
        dimensions=4,
        dtype="float32",
        sort_macro_window=False,
        resume=True,
        progress_every=0,
    )
    assert s_res.rows_emitted == 2
    assert sum(tok_encoded_counts) == 2
    assert eng2.encode_calls == [2]

    # Changed model_name invalidates fingerprint so all 6 rows re-encode cleanly.
    tok_encoded_counts.clear()
    eng3 = _EmbedSpyEngine("embed-v2")
    s_model_chg = embed_parquet(
        eng3,
        in_pq,
        out_dir,
        batch_rows=2,
        shard_rows=2,
        dimensions=4,
        dtype="float32",
        sort_macro_window=False,
        resume=True,
        progress_every=0,
    )
    assert s_model_chg.rows_emitted == 6 and sum(tok_encoded_counts) == 6

    # Changed input file size/mtime invalidates fingerprint cleanly.
    tok_encoded_counts.clear()
    pq.write_table(
        pa.table({"id": list(range(6)), "body": [f"alpha gamma" for _ in range(6)]}),
        in_pq,
    )
    eng4 = _EmbedSpyEngine("embed-v2")
    s_input_chg = embed_parquet(
        eng4,
        in_pq,
        out_dir,
        batch_rows=2,
        shard_rows=2,
        dimensions=4,
        dtype="float32",
        sort_macro_window=False,
        resume=True,
        progress_every=0,
    )
    assert s_input_chg.rows_emitted == 6 and sum(tok_encoded_counts) == 6


def test_hub_loader_pre_replace_sha_and_gov_corpora_manifest_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fast_shared.outcome_eval.gov_corpora as gov
    import fast_shared.outcome_eval.hub_loader as hl

    # 1. hub_loader: SHA-256 verified BEFORE replace(); corrupt download unlinks temp & leaves no poisoned file.
    good_bytes = b"valid-pinned-content"
    good_sha = hashlib.sha256(good_bytes).hexdigest()
    custom_lock = {
        "datasets": {
            "dummy_github_ds": {
                "source": "github",
                "repo_id": "example/repo",
                "revision": "deadbeef",
                "pinned_files": {"data.jsonl": {"sha256": good_sha}},
            }
        }
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(hl.urllib.request, "urlopen", lambda req, timeout=60: _FakeResp(b"CORRUPT-BYTES"))
    cache_root = tmp_path / "hub_cache"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        hl.download_pinned_file(
            "dummy_github_ds",
            "data.jsonl",
            lock=custom_lock,
            cache_dir=cache_root,
            verify_checksum=True,
        )
    target = cache_root / "dummy_github_ds" / "data.jsonl"
    assert not target.exists()
    assert list(target.parent.glob("*.tmp.*")) == []

    monkeypatch.setattr(hl.urllib.request, "urlopen", lambda req, timeout=60: _FakeResp(good_bytes))
    ok_path = hl.download_pinned_file(
        "dummy_github_ds",
        "data.jsonl",
        lock=custom_lock,
        cache_dir=cache_root,
        verify_checksum=True,
    )
    assert ok_path == target and ok_path.read_bytes() == good_bytes

    # 2. gov_corpora: stage_us_gov_ocr merges into existing manifest.json instead of clobbering other sources.
    monkeypatch.setattr(
        gov,
        "_stage_govinfo_sitemap_items",
        lambda root, *, collection, branch, year, per_source: [
            {"id": f"{collection}-1", "branch": branch, "source": f"govinfo_{collection.lower()}"}
        ],
    )
    monkeypatch.setattr(
        gov,
        "_stage_fr_items",
        lambda root, *, per_source: [{"id": "FR-1", "branch": "executive", "source": "govinfo_fr"}],
    )

    gov_cache = tmp_path / "gov_cache"
    res1 = gov.stage_us_gov_ocr(branches=("legislative",), sources=("govinfo_plaw",), cache_dir=gov_cache, per_source=1)
    assert res1["n_items"] == 1
    res2 = gov.stage_us_gov_ocr(branches=("executive",), sources=("govinfo_fr",), cache_dir=gov_cache, per_source=1)
    assert res2["n_items"] == 2
    assert res2["by_source"] == {"govinfo_fr": 1, "govinfo_plaw": 1}
    manifest = json.loads((gov_cache / "us_gov_ocr" / "manifest.json").read_text(encoding="utf-8"))
    assert [it["source"] for it in manifest["items"]] == ["govinfo_plaw", "govinfo_fr"]


def test_server_schema_key_sse_error_and_prefix_tree_admission() -> None:
    from fast_shared.prefix_tree import (
        DEFAULT_ADMISSION_MIN_HITS,
        PrefixMatch,
        PrefixTree,
        dequantize_gdn_state_block32,
        quantize_gdn_state_block32,
    )
    from fast_shared.server import (
        ManagedHTTPServer,
        OpenAIChatAdapter,
        make_chat_handler,
        schema_cache_key,
    )

    # 1. schema_cache_key preserves llguidance property insertion order while canonicalizing schema metadata keys & whitespace.
    s_ab = {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}}
    s_ab_reordered = {"properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "additionalProperties": False, "type": "object"}
    s_ba = {"type": "object", "additionalProperties": False, "properties": {"b": {"type": "string"}, "a": {"type": "integer"}}}
    assert schema_cache_key(s_ab) == schema_cache_key(s_ab_reordered)
    assert schema_cache_key(s_ab) != schema_cache_key(s_ba)
    assert schema_cache_key('{"b":1,"a":2}') == '{"b":1,"a":2}'

    # 2. _send_sse + iter_chat_sse_events returns HTTP 500 JSON when first-token generation raises.
    class _StreamErrEngine:
        model_name = "dummy-sse"
        default_mode = "gpu"
        ane_available = False

        def generate_chat_stream(self, messages, **kwargs):
            last = messages[-1]["content"]
            if last == "fail-first":
                raise RuntimeError("prefill failed before token 0")
            yield {"delta": "hello ", "prompt_tokens": 2, "completion_tokens": 1}
            yield {"delta": "world", "finish_reason": "stop", "prompt_tokens": 2, "completion_tokens": 2}

    adapter = OpenAIChatAdapter(_StreamErrEngine(), coalesce_ms=0.0)
    with ManagedHTTPServer(make_chat_handler(adapter), host="127.0.0.1", port=0) as srv:
        url = f"http://{srv.host}:{srv.port}/v1/chat/completions"

        bad_req = urllib.request.Request(
            url,
            data=json.dumps({"messages": [{"role": "user", "content": "fail-first"}], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_req, timeout=5)
        assert exc_info.value.code == 500
        err_body = json.loads(exc_info.value.read().decode())
        assert "prefill failed before token 0" in err_body["error"]["message"]

        ok_req = urllib.request.Request(
            url,
            data=json.dumps({"messages": [{"role": "user", "content": "ok"}], "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ok_req, timeout=5) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            sse_text = resp.read().decode("utf-8")
            assert '"role": "assistant"' in sse_text
            assert '"hello "' in sse_text
            assert "data: [DONE]" in sse_text

    # 3. PrefixTree admit_prefix(..., admission_min_hits=2) & record(..., count_for_circuit_breaker=...)
    assert DEFAULT_ADMISSION_MIN_HITS == 1
    tree = PrefixTree(chunk_tokens=4, byte_budget=4096, circuit_breaker_min_lookups=4, min_hit_rate=0.5)
    toks = [1, 2, 3, 4, 5, 6]
    assert tree.admit_prefix(toks, admission_min_hits=2) is False
    assert tree.admit_prefix(toks, admission_min_hits=2) is True
    assert tree.admit_prefix(toks, profile="off", admission_min_hits=2) is False

    empty_match = PrefixMatch(keys=(), chunks=(), snapshot=None, reused_tokens=0, replayed_tokens=4)
    for _ in range(10):
        tree.record(empty_match, restored=False, count_for_circuit_breaker=False)
    assert tree.circuit_breaker_open is False

    for _ in range(4):
        tree.record(empty_match, restored=False, count_for_circuit_breaker=True)
    assert tree.circuit_breaker_open is True
    assert callable(tree.rollback_keys)
    assert callable(quantize_gdn_state_block32) and callable(dequantize_gdn_state_block32)
