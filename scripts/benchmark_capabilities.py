#!/usr/bin/env python3
"""Post-Training, Co-Execution, Speculative Decoding & Stateful Caching ABBA Suite for `fast-silicon`.

Consolidates all advanced engine capability benchmarks into one isolated-subprocess CLI
serialized under `/tmp/m4_gpu_validation.lock`, pairing every `fast-*` capability against
its fastest verified stock `MLX` / `coremltools` baseline:

  - `qlora-packing`   : Packed 8-bit and unpacked BF16 training on the same 12 examples,
                        plus a separate eight-example, four-copy system prompt workload
  - `encoder-finetune`: ModernBERT-421M (`28L, H=1024, freeze<18`) `BF16+compile` + `Packed 512`
                        + `Cached L0..17` vs stock unpacked MLX `nn.Module` (gradient parity + ABBA)
  - `grpo-ane-critic` : `G=4` GRPO / Best-of-N rolling `FastGemma4Engine` + concurrent `Laya` ANE
                        critic + clipped-surrogate GRPO step vs stock `mlx-vlm` + GPU `Kev-0.8B`
  - `mtp-speculative` : Production `FastGemma4Engine(mtp_drafter=True)` (`B=1` & `B=4` free-text
                        + constrained JSON) vs `FastGemma4Engine` AR & stock `mlx-vlm` (`B=1` & `B=4`)
  - `prefix-tree`     : Production `Kev08bMlxBackend(prefix_tree=PrefixTree(256))` (`GDN+KV` state
                        snapshot `B=1` & `B=8`, `max |dp| == 0.0`) vs uncached & stock `mlx-lm`
  - `parallel-coreml` : Production `fast_shared.parallel_coreml` (`jobs=4` spawn pool across 9
                        BGE-small ANE shape buckets) vs stock sequential `coremltools` (`jobs=1`)
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = "/tmp/m4_gpu_validation.lock"
_WORKER_MARKER = "__CAPABILITY_WORKER_JSON__="

for _pkg in (
    "fast-shared",
    "fast-bge-small",
    "fast-embeddinggemma-2",
    "fast-pplx",
    "fast-decision-models",
    "fast-gemma4",
    "fast-gpt-oss",
):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from fast_shared.bench import (  # noqa: E402
    EXIT_SKIP,
    check_system_quiescence,
    collect_env_telemetry,
    get_unified_memory_footprint,
    run_isolated_json_worker,
    wired_limit,
    write_worker_json,
)
from public_baselines import resolve_hf_snapshot, run_subprocess_with_killpg  # noqa: E402


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str
    needs_lora_python: bool = False


SUITE_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec("qlora-packing", "Gemma-4 E2B QLoRA 8-bit g64 + pack=2048 + shared-sys-kv vs stock mlx-lm", True),
    CapabilitySpec("encoder-finetune", "ModernBERT-421M BF16+compile + Packed-512 + Cached L0..17 vs stock MLX", False),
    CapabilitySpec("grpo-ane-critic", "G=4 GRPO rolling FastGemma4 + concurrent Laya ANE critic vs stock mlx-vlm", False),
    CapabilitySpec("mtp-speculative", "Gemma-4 E2B MTP DraftScan (B=1 & B=4) vs AR & stock mlx-vlm", False),
    CapabilitySpec("prefix-tree", "Kev-0.8B 256-tok GDN+KV PrefixTree vs uncached & stock mlx-lm", False),
    CapabilitySpec("parallel-coreml", "9-bucket CoreML ANE spawn pool (jobs=4) vs sequential coremltools (jobs=1)", False),
)
_SUITE_BY_NAME: dict[str, CapabilitySpec] = {s.name: s for s in SUITE_SPECS}


def select_suites(raw: str) -> list[CapabilitySpec]:
    if raw.strip() == "all":
        return list(SUITE_SPECS)
    out: list[CapabilitySpec] = []
    for part in raw.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in _SUITE_BY_NAME:
            raise ValueError(f"Unknown suite {key!r}; choose from all, {', '.join(_SUITE_BY_NAME)}")
        out.append(_SUITE_BY_NAME[key])
    return out


def resolve_pkg_home(pkg_name: str, env_var: str) -> Path:
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    mono = _REPO_ROOT / "packages" / pkg_name
    sib = _REPO_ROOT.parent / pkg_name
    strong = ("weights", "ane_models", "kev_08b", "laya_typed", "models")
    mono_ok = any((mono / "artifacts" / s).is_dir() and any((mono / "artifacts" / s).iterdir()) for s in strong)
    sib_ok = any((sib / "artifacts" / s).is_dir() and any((sib / "artifacts" / s).iterdir()) for s in strong)
    if not mono_ok and sib_ok:
        return sib
    return mono


@contextmanager
def _hw_lock() -> Iterator[None]:
    if os.environ.get("M4_GPU_LOCK_HELD") == "1":
        yield
        return
    import fcntl

    fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        prev = os.environ.get("M4_GPU_LOCK_HELD")
        os.environ["M4_GPU_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("M4_GPU_LOCK_HELD", None)
            else:
                os.environ["M4_GPU_LOCK_HELD"] = prev
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# 1. QLoRA Packing + Shared-Sys-KV (`fast_gemma4.lora_accel` vs stock `mlx-lm`)
# ---------------------------------------------------------------------------


def _qlora_pairs(
    raw_docs: Sequence[Any], sys_txt: str, tokenizer: Any, *, n_docs: int, sys_repeat: int,
) -> list[tuple[list[int], int]]:
    stxt = "\n\n".join([sys_txt] * sys_repeat) if sys_repeat > 1 else sys_txt
    pairs: list[tuple[list[int], int]] = []
    for i in range(n_docs):
        doc_id, fname, meta_str, body = raw_docs[i % len(raw_docs)]
        expanded = "\n".join(f"[Section {r + 1} - Record {i:02d}] {body}" for r in range(2 + (i % 2)))
        prompt = (
            f"<bos><start_of_turn>user\n{stxt}\n\n"
            f"FILE: {fname} (batch_item={i})\nTEXT:\n{expanded}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        meta_obj = json.loads(meta_str)
        ans_obj = {
            "status": "usable",
            "text_quality": "native",
            "primary_kind": "technical_report",
            "domain": meta_obj["domain"],
            "title": f"{doc_id} Report {i}",
            "organization": meta_obj["org"],
            "doc_date": meta_obj["date"],
            "year": int(meta_obj["date"][:4]),
            "people": [{"name": "Lead Analyst", "role": "author", "org": meta_obj["org"]}],
            "refs": [f"REF-{i:03d}-A", f"STD-{200 + i}"],
            "summary": f"Technical assessment summarizing operational metrics for {fname}.",
            "topics": ["reliability", "compliance", "operations", f"topic_{i % 4}"],
        }
        ans = json.dumps(ans_obj, separators=(",", ":")) + "<end_of_turn>\n"
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        a_ids = tokenizer.encode(ans, add_special_tokens=False)
        pairs.append((p_ids + a_ids, len(p_ids)))
    return pairs


def _qlora_regime_report(
    pairs: Sequence[tuple[list[int], int]], runs: Sequence[dict[str, Any]], *, prompt_copies: int, pack_tokens: int,
) -> dict[str, Any]:
    examples = len(pairs)
    logical_inputs = sum(len(full) - 1 for full, _ in pairs)
    supervised = sum(len(full) - prompt for full, prompt in pairs)
    for run in runs:
        actual = (run["examples"], run["logical_input_tokens"], run["supervised_tokens"])
        if actual != (examples, logical_inputs, supervised):
            raise ValueError(f"QLoRA pass trained on {actual}, expected {(examples, logical_inputs, supervised)}")
    seconds = statistics.median(run["elapsed_seconds"] for run in runs)
    report: dict[str, Any] = {
        "input_ids_and_prompt_lengths_sha256": hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest(),
        "system_instruction_copies": prompt_copies, "pack_tokens": pack_tokens,
        "examples": examples, "logical_input_tokens": logical_inputs, "supervised_tokens": supervised,
        "elapsed_seconds": seconds,
        "supervised_tokens_per_sec": statistics.median(supervised / run["elapsed_seconds"] for run in runs),
        "examples_per_sec": statistics.median(examples / run["elapsed_seconds"] for run in runs),
        # Historical tok_s counts logical input positions, including any reused prefixes.
        "tok_s": round(statistics.median(logical_inputs / run["elapsed_seconds"] for run in runs), 1),
        "peak_gb": statistics.median(run["peak_gb"] for run in runs),
        "optimizer_steps_per_pass": [run["optimizer_steps"] for run in runs],
        "runs": list(runs),
    }
    if all("peak_gib" in run for run in runs):
        report["peak_gib"] = statistics.median(run["peak_gib"] for run in runs)
    return report


_PREQUANT_GEMMA4_REPOS: tuple[str, ...] = (
    "mlx-community/gemma-4-e2b-it-8bit",
    "lmstudio-community/gemma-4-E2B-it-MLX-4bit",
    "mlx-community/gemma-4-e2b-it-4bit",
    "mlx-community/gemma-4-E2B-it-qat-4bit",
)


def select_qlora_policy(mem_gb: float) -> dict[str, Any]:
    """Select wired-safe QLoRA packing, checkpointing (`ckpt_tokens=512`), and `<36 GB` BF16 gating."""
    from fast_shared.runtime_lessons import recommend_silicon_budget

    eff_mem = max(16.0, float(mem_gb or 24.0))
    budget = recommend_silicon_budget(eff_mem, workload="qlora")
    raw_ckpt = int(budget.ckpt_tokens)
    ckpt_tokens = 512 if raw_ckpt <= 0 else min(512, max(256, raw_ckpt))
    if eff_mem >= 20.0:
        ckpt_tokens = 512
    return {
        "pack_tokens": int(budget.seq_pack_len),
        "grad_accum": int(budget.grad_accum),
        "ckpt_tokens": ckpt_tokens,
        "memory_limit_gb": float(budget.memory_limit_gb),
        "cache_limit_gb": float(budget.cache_limit_gb),
        "run_bf16_baseline": float(mem_gb or 0.0) >= 36.0,
    }


def _resolve_prequant_gemma4_snapshot() -> Path | None:
    for repo in _PREQUANT_GEMMA4_REPOS:
        snap = resolve_hf_snapshot(repo)
        if snap is not None:
            return snap
    return None


def _run_qlora_packing(args: argparse.Namespace, host_mem_gb: float) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_model, load_tokenizer

    from fast_gemma4.lora_accel import (
        CompiledOptimizerStep,
        build_packed_groups,
        build_shared_prefix_packed_groups,
        cast_lora_adapters_dtype,
        chunked_softcap_ce_loss,
        compute_packed_tail_ce_loss,
        find_shared_prefix_len,
        grad_tree_cosine,
        pack_examples_to_bin,
        pack_suffixes_for_shared_prefix,
        quantize_frozen_trunk_8bit,
        run_frozen_prefix_packed,
        run_shared_prefix_grad_accum_step,
        unwrap_kv_shared_lora_modules,
    )

    pol = select_qlora_policy(host_mem_gb)
    run_bf16_baseline = bool(pol["run_bf16_baseline"])
    prequant_snap = _resolve_prequant_gemma4_snapshot()
    bf16_snap = resolve_hf_snapshot("mlx-community/gemma-4-e2b-it-bf16") if run_bf16_baseline else None

    if not run_bf16_baseline:
        # Gate the 10.24 GB BF16 checkpoint strictly off on < 36 GB hosts to prevent Metal OOM.
        if prequant_snap is None:
            return {
                "suite": "qlora-packing",
                "status": "skipped",
                "reason": "missing pre-quantized Gemma-4 E2B 4-bit/8-bit snapshot (<36 GB host skips 10.24 GB BF16)",
            }
        tok_snap = prequant_snap
    else:
        tok_snap = prequant_snap or bf16_snap
        if tok_snap is None:
            return {"suite": "qlora-packing", "status": "skipped", "reason": "missing Gemma-4 E2B snapshot"}

    low_ram = not run_bf16_baseline
    max_rec = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
    mem_cap = (max_rec if max_rec > 0 else int(18.5 * (1024**3))) if low_ram else int(36.0 * (1024**3))
    mx.set_memory_limit(mem_cap)
    mx.set_cache_limit(256 * 1024 * 1024)

    fg4_home = resolve_pkg_home("fast-gemma4", "FAST_GEMMA4_HOME")
    fix_dir = fg4_home / "tests" / "fixtures"
    sys_txt = (fix_dir / "sys_extract_v1.txt").read_text(encoding="utf-8").strip()
    raw_docs = json.loads((fix_dir / "sample_docs_8.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tok_snap)
    num_lora_layers = 23
    ckpt_toks = int(pol["ckpt_tokens"])
    pack_tok_bench = int(pol["pack_tokens"])

    def _load_lora(*, snap_path: Path, qlora_quant: bool, seed: int = 42) -> tuple[Any, int]:
        mx.random.seed(seed)
        mdl, _ = load_model(snap_path, strict=False)
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(mdl, attr):
                setattr(mdl, attr, None)
        gc.collect()
        mx.clear_cache()
        if qlora_quant:
            quantize_frozen_trunk_8bit(mdl, group_size=64, bits=8, quantize_ple_embedding=True, quantize_embed_tokens=True)
            mx.eval(mdl.parameters())
            gc.collect()
            mx.clear_cache()
        mdl.freeze()
        mx.random.seed(seed)
        linear_to_lora_layers(mdl, num_layers=num_lora_layers, config={"rank": 24, "scale": 2.0, "dropout": 0.0})
        cast_lora_adapters_dtype(mdl, mx.bfloat16)
        unwrap_kv_shared_lora_modules(mdl)
        mx.eval(mdl.parameters())
        gc.collect()
        mx.clear_cache()
        return mdl, len(mdl.layers) - num_lora_layers

    def _frozen_legacy(mdl: Any, inp: Any, first_l: int) -> tuple[Any, ...]:
        tm = mdl.language_model.model
        emb = tm.embed_tokens(inp)
        h = emb * tm.embed_scale
        pli = tm._project_per_layer_inputs(h, tm._get_per_layer_inputs(inp, emb))
        pli_l = [pli[:, :, i, :] for i in range(len(tm.layers))]
        masks = tm._make_masks(h, [None] * len(tm.layers))
        inter: list[Any] = [(None, None)] * len(tm.layers)
        for idx in range(first_l):
            kvs, off = inter[tm.previous_kvs[idx]]
            h, kvs, off = tm.layers[idx](h, masks[idx], None, per_layer_input=pli_l[idx], shared_kv=kvs, offset=off)
            inter[idx] = (kvs, off)
        return h, masks, inter, pli_l

    def _tail_legacy(
        mdl: Any,
        h: Any,
        masks: list[Any],
        inter_in: list[Any],
        pli_l: list[Any],
        spans: list[tuple[int, int]],
        y_flat: Any,
        first_l: int,
        use_ckpt: bool,
        compiled_ce: bool,
    ) -> Any:
        lm = mdl.language_model
        tm = lm.model
        inter = list(inter_in)
        for idx in range(first_l, len(tm.layers)):
            kvs, off = inter[tm.previous_kvs[idx]]
            lyr = tm.layers[idx]
            if use_ckpt:

                def _cf(hi: Any, mi: Any, pi: Any, ki: Any, oi: Any, _l: Any = lyr) -> Any:
                    return _l(hi, mi, None, per_layer_input=pi, shared_kv=ki, offset=oi)

                h, kvs, off = nn.utils.checkpoint(lyr, _cf)(h, masks[idx], pli_l[idx], kvs, off)
            else:
                h, kvs, off = lyr(h, masks[idx], None, per_layer_input=pli_l[idx], shared_kv=kvs, offset=off)
            inter[idx] = (kvs, off)
        hn = tm.norm(h)
        hans = mx.concatenate([hn[bi, s0:s1, :] for bi, (s0, s1) in enumerate(spans)], axis=0)
        head_fn = tm.embed_tokens.as_linear if lm.tie_word_embeddings else lm.lm_head
        return chunked_softcap_ce_loss(
            head_fn, hans, y_flat, softcap=lm.final_logit_softcapping, chunk_size=256, compiled=compiled_ce
        )

    def _to_host_f32_tree(gtree: Any) -> Any:
        return tree_map(lambda x: np.asarray(x.astype(mx.float32)).copy(), gtree)

    def _np_tree_to_mx(ntree: Any) -> Any:
        return tree_map(lambda x: mx.array(x), ntree)

    def _run_regime(
        mdl: Any,
        first_l: int,
        pair_list: list[tuple[list[int], int]],
        *,
        pack_tokens: int,
        pad_mult: int,
        shared_sys: bool,
        compiled: bool,
        ckpt_toks: int,
    ) -> dict[str, float]:
        opt = optim.AdamW(learning_rate=1.8e-4, weight_decay=0.01)
        stepper = CompiledOptimizerStep(opt, mdl, max_grad_norm=1.0, compiled=compiled)
        vgl = nn.value_and_grad(mdl, _tail_legacy)
        vgp = nn.value_and_grad(mdl, compute_packed_tail_ce_loss)
        mdl.train()
        if shared_sys:
            batches = build_shared_prefix_packed_groups(pair_list, pack_tokens=pack_tokens, min_shared_prefix=32, grad_accum=4, seed=1)
        elif pack_tokens > 0:
            batches = build_packed_groups(pair_list, pack_tokens=pack_tokens, seed=1)
        else:
            batches = [[p] for p in pair_list]

        def _pass(b_list: list[list[tuple[list[int], int]]]) -> dict[str, int]:
            counts = {"logical_input_tokens": 0, "supervised_tokens": 0, "examples": 0, "optimizer_steps": 0}
            if shared_sys:
                for cur in range(0, len(b_list), 4):
                    win = b_list[cur : cur + 4]
                    all_ex = [ex for b in win for ex in b]
                    slen = find_shared_prefix_len(all_ex, min_len=32)
                    sbins = [
                        pack_suffixes_for_shared_prefix(
                            b, sys_len=slen, pack_tokens=pack_tokens, pad_to_fixed=(pack_tokens > 0), pad_multiple=pad_mult, loss_norm="token"
                        )
                        for b in win
                    ]
                    _, grads, _ = run_shared_prefix_grad_accum_step(
                        mdl, all_ex[0][0][:slen], sbins, first_l, ckpt_tokens=ckpt_toks, compiled_ce=compiled
                    )
                    stepper.step(mdl, grads, lr=1.8e-4, accum_divisor=1)
                    mx.eval(mdl.parameters(), opt.state)
                    counts["logical_input_tokens"] += sum(sb.logical_total_tokens for sb in sbins)
                    counts["supervised_tokens"] += sum(sb.y_flat.size for sb in sbins)
                    counts["examples"] += sum(sb.num_examples for sb in sbins)
                    counts["optimizer_steps"] += 1
                return counts

            acc_g: Any = None
            acc_n = 0
            for idx_b, b in enumerate(b_list):
                if pack_tokens > 0:
                    pb = pack_examples_to_bin(b, pack_tokens=pack_tokens, pad_to_fixed=True, pad_multiple=pad_mult, loss_norm="token")
                    h1, m1, i1, p1, r1 = run_frozen_prefix_packed(mdl, pb, first_l)
                    loss, grads = vgp(
                        mdl, h1, m1, i1, p1, r1, pb.ans_indices, pb.y_flat, None, first_l,
                        use_ckpt=(pb.bin_tokens > ckpt_toks), chunk_size=256, compiled_ce=compiled,
                    )
                    counts["logical_input_tokens"] += pb.real_tokens
                    counts["supervised_tokens"] += pb.y_flat.size
                    counts["examples"] += pb.num_examples
                else:
                    mx_i = max(len(f) - 1 for f, _ in b)
                    inp = mx.array([f[:-1] + [0] * (mx_i - (len(f) - 1)) for f, _ in b], dtype=mx.int32)
                    sp = [(pl - 1, len(f) - 1) for f, pl in b]
                    yf = mx.array([tid for f, pl in b for tid in f[pl:]], dtype=mx.int32)
                    bt = int(inp.shape[0] * inp.shape[1])
                    h1, m1, i1, p1 = _frozen_legacy(mdl, inp, first_l)
                    loss, grads = vgl(mdl, h1, m1, i1, p1, sp, yf, first_l, bt > ckpt_toks, compiled)
                    counts["logical_input_tokens"] += sum(len(full) - 1 for full, _ in b)
                    counts["supervised_tokens"] += yf.size
                    counts["examples"] += len(b)
                acc_g = stepper.accumulate(acc_g, grads)
                acc_n += 1
                mx.eval(loss, acc_g)
                if acc_n >= 4 or idx_b == len(b_list) - 1:
                    stepper.step(mdl, acc_g, lr=1.8e-4, accum_divisor=acc_n)
                    mx.eval(mdl.parameters(), opt.state)
                    counts["optimizer_steps"] += 1
                    acc_g = None
                    acc_n = 0
            return counts

        _ = _pass(batches[: max(1, min(len(batches), 4 if shared_sys else 1))])
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        counts = _pass(batches)
        mx.synchronize()
        dt = time.perf_counter() - t0
        peak_bytes = float(mx.get_peak_memory())
        pk_gb = peak_bytes / 1e9
        pk_gib = peak_bytes / (1024**3)
        del opt, stepper
        gc.collect()
        mx.clear_cache()
        return {**counts, "elapsed_seconds": dt, "peak_gb": round(pk_gb, 3), "peak_gib": round(pk_gib, 3)}

    pairs = _qlora_pairs(raw_docs, sys_txt, tokenizer, n_docs=12, sys_repeat=1)
    pairs_long_sys = _qlora_pairs(raw_docs, sys_txt, tokenizer, n_docs=8, sys_repeat=4)
    sub_p = pairs[: (1 if low_ram else 2)]
    max_in = max(len(f) - 1 for f, _ in sub_p)
    inp_u = mx.array([f[:-1] + [0] * (max_in - (len(f) - 1)) for f, _ in sub_p], dtype=mx.int32)
    spans_u = [(pl - 1, len(f) - 1) for f, pl in sub_p]
    y_u = mx.array([tok_id for f, pl in sub_p for tok_id in f[pl:]], dtype=mx.int32)
    pbin = pack_examples_to_bin(sub_p, pack_tokens=pack_tok_bench, pad_to_fixed=True, loss_norm="token")

    cos_q8: float | None = None
    if run_bf16_baseline and bf16_snap is not None:
        mdl_bf16, first_l_bf16 = _load_lora(snap_path=bf16_snap, qlora_quant=False, seed=42)
        vg_leg = nn.value_and_grad(mdl_bf16, _tail_legacy)
        vg_pck = nn.value_and_grad(mdl_bf16, compute_packed_tail_ce_loss)

        hu, mu, iu, pu = _frozen_legacy(mdl_bf16, inp_u, first_l_bf16)
        lu, gu = vg_leg(mdl_bf16, hu, mu, iu, pu, spans_u, y_u, first_l_bf16, True, False)
        mx.eval(lu, gu)
        loss_u = float(lu.item())
        gu_np = _to_host_f32_tree(gu)
        del lu, gu, hu, mu, iu, pu
        gc.collect()
        mx.clear_cache()

        hp, mp, ip, pp, rtp = run_frozen_prefix_packed(mdl_bf16, pbin, first_l_bf16)
        lp, gp = vg_pck(
            mdl_bf16, hp, mp, ip, pp, rtp, pbin.ans_indices, pbin.y_flat, None, first_l_bf16,
            use_ckpt=True, chunk_size=256, compiled_ce=True,
        )
        mx.eval(lp, gp)
        loss_p = float(lp.item())
        gp_np = _to_host_f32_tree(gp)
        del lp, gp, hp, mp, ip, pp, rtp
        gc.collect()
        mx.clear_cache()

        cos_pack, min_cos_pack, _ = grad_tree_cosine(_np_tree_to_mx(gu_np), _np_tree_to_mx(gp_np))
        del gu_np

        a_runs = [
            _run_regime(
                mdl_bf16, first_l_bf16, pairs, pack_tokens=0, pad_mult=1, shared_sys=False, compiled=False, ckpt_toks=ckpt_toks
            )
            for _ in range(2)
        ]
        stock_regime_key = "stock_bf16_unpacked"
        stock_desc = "Unpacked BF16"
        del mdl_bf16
        gc.collect()
        mx.clear_cache()

        q_snap = prequant_snap if prequant_snap is not None else bf16_snap
        mdl_q, first_l_q = _load_lora(snap_path=q_snap, qlora_quant=(prequant_snap is None), seed=42)
        hp8, mp8, ip8, pp8, rtp8 = run_frozen_prefix_packed(mdl_q, pbin, first_l_q)
        vg_q8 = nn.value_and_grad(mdl_q, compute_packed_tail_ce_loss)
        lq8, gq8 = vg_q8(
            mdl_q, hp8, mp8, ip8, pp8, rtp8, pbin.ans_indices, pbin.y_flat, None, first_l_q,
            use_ckpt=True, chunk_size=256, compiled_ce=True,
        )
        mx.eval(lq8, gq8)
        gq8_np = _to_host_f32_tree(gq8)
        cos_q8_val, _, _ = grad_tree_cosine(_np_tree_to_mx(gp_np), _np_tree_to_mx(gq8_np))
        cos_q8 = float(cos_q8_val)
        del lq8, gq8, hp8, mp8, ip8, pp8, rtp8, gp_np, gq8_np
        gc.collect()
        mx.clear_cache()
    else:
        assert prequant_snap is not None
        mdl_q, first_l_q = _load_lora(snap_path=prequant_snap, qlora_quant=False, seed=42)
        vg_leg_q = nn.value_and_grad(mdl_q, _tail_legacy)
        vg_pck_q = nn.value_and_grad(mdl_q, compute_packed_tail_ce_loss)

        hu, mu, iu, pu = _frozen_legacy(mdl_q, inp_u, first_l_q)
        lu, gu = vg_leg_q(mdl_q, hu, mu, iu, pu, spans_u, y_u, first_l_q, True, False)
        mx.eval(lu, gu)
        loss_u = float(lu.item())
        gu_np = _to_host_f32_tree(gu)
        del lu, gu, hu, mu, iu, pu
        gc.collect()
        mx.clear_cache()

        hp, mp, ip, pp, rtp = run_frozen_prefix_packed(mdl_q, pbin, first_l_q)
        lp, gp = vg_pck_q(
            mdl_q, hp, mp, ip, pp, rtp, pbin.ans_indices, pbin.y_flat, None, first_l_q,
            use_ckpt=True, chunk_size=256, compiled_ce=True,
        )
        mx.eval(lp, gp)
        loss_p = float(lp.item())
        gp_np = _to_host_f32_tree(gp)
        del lp, gp, hp, mp, ip, pp, rtp
        gc.collect()
        mx.clear_cache()

        cos_pack, min_cos_pack, _ = grad_tree_cosine(_np_tree_to_mx(gu_np), _np_tree_to_mx(gp_np))
        del gu_np, gp_np
        gc.collect()
        mx.clear_cache()

        a_runs = [
            _run_regime(
                mdl_q, first_l_q, pairs, pack_tokens=0, pad_mult=1, shared_sys=False, compiled=False, ckpt_toks=ckpt_toks
            )
            for _ in range(2)
        ]
        stock_regime_key = "stock_qlora_unpacked"
        stock_desc = "Unpacked pre-quantized QLoRA"
        gc.collect()
        mx.clear_cache()

    b_pack_runs = [
        _run_regime(
            mdl_q, first_l_q, pairs, pack_tokens=pack_tok_bench, pad_mult=64, shared_sys=False, compiled=True, ckpt_toks=ckpt_toks
        )
        for _ in range(2)
    ]
    gc.collect()
    mx.clear_cache()
    b_sys_runs = [
        _run_regime(
            mdl_q, first_l_q, pairs_long_sys, pack_tokens=pack_tok_bench, pad_mult=64, shared_sys=True, compiled=True, ckpt_toks=ckpt_toks
        )
        for _ in range(2)
    ]
    del mdl_q
    gc.collect()
    mx.clear_cache()

    stock = _qlora_regime_report(pairs, a_runs, prompt_copies=1, pack_tokens=0)
    packed = _qlora_regime_report(pairs, b_pack_runs, prompt_copies=1, pack_tokens=pack_tok_bench)
    shared = _qlora_regime_report(pairs_long_sys, b_sys_runs, prompt_copies=4, pack_tokens=pack_tok_bench)

    parity_payload: dict[str, Any] = {
        "packed_vs_unpacked_loss_diff": round(abs(loss_p - loss_u), 6),
        "packed_vs_unpacked_grad_cos": round(float(cos_pack), 6),
        "packed_min_tensor_cos": round(float(min_cos_pack), 6),
        "qlora_8bit_vs_bf16_grad_cos": round(float(cos_q8), 6) if cos_q8 is not None else None,
    }
    parity_ok = (
        float(cos_pack) >= 0.995
        and float(min_cos_pack) >= 0.99
        and (cos_q8 is None or float(cos_q8) >= 0.98)
    )
    return {
        "suite": "qlora-packing",
        "status": "ok" if parity_ok else "failed",
        "policy": pol,
        "fast_metric": (
            f"Packed QLoRA ({pack_tok_bench}) {packed['supervised_tokens_per_sec']:,.1f} supervised tokens/s on the same {packed['examples']} examples. "
            f"The separate shared-prefix run uses {shared['examples']} examples with four copies of the instructions."
        ),
        "stock_baseline": f"{stock_desc} {stock['supervised_tokens_per_sec']:,.1f} supervised tokens/s",
        "speedup_vs_stock": round(packed["supervised_tokens_per_sec"] / max(stock["supervised_tokens_per_sec"], 1e-6), 2),
        "comparison": (
            f"Packed QLoRA (ckpt_tokens={ckpt_toks}) vs {stock_desc.lower()} on identical input and answer IDs. "
            "The separate shared-prefix pass uses eight examples with four copies of the instructions."
        ),
        "measurement": "Two consecutive timed passes per regime after optimizer warmup. Each elapsed time counts training on those examples. Optimizer steps can differ by packing. tok_s counts logical input positions, including reused prefixes. peak_gb records decimal GB (1e9 bytes) of MLX allocations; peak_gib records GiB.",
        "peak_gb": packed["peak_gb"],
        "stock_peak_gb": stock["peak_gb"],
        "parity": parity_payload,
        "regimes": {
            stock_regime_key: stock,
            f"qlora_packed_{pack_tok_bench}": packed,
            "qlora_shared_sys_kv": {**shared, "matched_baseline": None},
        },
    }


# ---------------------------------------------------------------------------
# 2. Authentic ModernBERT-large (`421M`, `28L`, `freeze<18`) Validator Fine-Tuning
# ---------------------------------------------------------------------------

try:
    import mlx.core as _mb_mx
    import mlx.nn as _mb_nn

    _MbModuleBase: Any = _mb_nn.Module
except Exception:  # pragma: no cover
    _mb_mx = None  # type: ignore[assignment]
    _mb_nn = None  # type: ignore[assignment]
    _MbModuleBase = object


class _ModernBertBlock(_MbModuleBase):  # type: ignore[misc]
    """Authentic ModernBERT encoder block (`attn_norm=None` at L0, biasless LayerNorm, exact GeGLU, 2D RoPE)."""

    def __init__(
        self,
        layer_idx: int,
        *,
        hidden: int = 1024,
        n_heads: int = 16,
        d_ff: int = 2624,
        local_attention: int = 128,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.is_global = (self.layer_idx % 3) == 0
        self.rope_theta = 160_000.0 if self.is_global else 10_000.0
        self.local_attention = int(local_attention)
        self.n_heads = int(n_heads)
        self.head_dim = int(hidden) // int(n_heads)
        self.scale = self.head_dim**-0.5
        self.attn_norm = None if self.layer_idx == 0 else _mb_nn.LayerNorm(hidden, eps=1e-5, bias=False)
        self.wqkv = _mb_nn.Linear(hidden, 3 * hidden, bias=False)
        self.wo = _mb_nn.Linear(hidden, hidden, bias=False)
        self.mlp_norm = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=False)
        self.wi = _mb_nn.Linear(hidden, 2 * d_ff, bias=False)
        self.wd = _mb_nn.Linear(d_ff, hidden, bias=False)

    def _apply_2d_rope(self, q: Any, k: Any, position_ids: Any) -> tuple[Any, Any]:
        # 2D (B, S) position_ids supporting per-segment RoPE resets on Packed-512 (avoids upstream 1D B*S bug).
        inv_freq = 1.0 / (
            self.rope_theta ** (_mb_mx.arange(0, self.head_dim, 2, dtype=_mb_mx.float32) / float(self.head_dim))
        )
        angles = position_ids[:, None, :, None].astype(_mb_mx.float32) * inv_freq[None, None, None, :]
        cos = _mb_mx.cos(angles).astype(q.dtype)
        sin = _mb_mx.sin(angles).astype(q.dtype)
        q0, q1 = _mb_mx.split(q, 2, axis=-1)
        k0, k1 = _mb_mx.split(k, 2, axis=-1)
        qr = _mb_mx.concatenate([q0 * cos - q1 * sin, q1 * cos + q0 * sin], axis=-1)
        kr = _mb_mx.concatenate([k0 * cos - k1 * sin, k1 * cos + k0 * sin], axis=-1)
        return qr, kr

    def __call__(self, x: Any, position_ids: Any, mask_4d: Any) -> Any:
        B, S, H = x.shape
        xa = x if self.attn_norm is None else self.attn_norm(x)
        qkv = self.wqkv(xa).reshape(B, S, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        q, k = self._apply_2d_rope(q, k, position_ids)
        att = _mb_mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask_4d.astype(q.dtype))
        x = x + self.wo(att.transpose(0, 2, 1, 3).reshape(B, S, H))
        val, gate = _mb_mx.split(self.wi(self.mlp_norm(x)), 2, axis=-1)
        return x + self.wd(_mb_nn.gelu(val) * gate)


class _LayaDecisionHeadLayer(_MbModuleBase):  # type: ignore[misc]
    """2-layer post-norm/pre-norm Transformer decision head matching `convaiinnovations/laya-typed-decisions`."""

    def __init__(self, hidden: int = 1024, n_heads: int = 16, head_ff: int = 2048) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(hidden) // int(n_heads)
        self.scale = self.head_dim**-0.5
        self.norm1 = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=True)
        self.in_proj = _mb_nn.Linear(hidden, 3 * hidden, bias=True)
        self.out_proj = _mb_nn.Linear(hidden, hidden, bias=True)
        self.norm2 = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=True)
        self.linear1 = _mb_nn.Linear(hidden, head_ff, bias=True)
        self.linear2 = _mb_nn.Linear(head_ff, hidden, bias=True)

    def __call__(self, x: Any, mask_full: Any) -> Any:
        B, S, H = x.shape
        qkv = self.in_proj(self.norm1(x)).reshape(B, S, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        att = _mb_mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask_full.astype(q.dtype))
        x = x + self.out_proj(att.transpose(0, 2, 1, 3).reshape(B, S, H))
        return x + self.linear2(_mb_nn.relu(self.linear1(self.norm2(x))))


class _ModernBertLargeValidator(_MbModuleBase):  # type: ignore[misc]
    """Authentic ModernBERT-large (`28L`, global/local RoPE `160K`/`10K`, `local_attention=128`, 2L head + scorer)."""

    def __init__(
        self,
        vocab: int = 50368,
        hidden: int = 1024,
        n_heads: int = 16,
        d_ff: int = 2624,
        n_layers: int = 28,
        freeze_below: int = 18,
        head_ff: int = 2048,
        scorer_hidden: int | None = None,
        local_attention: int = 128,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.freeze_below = int(freeze_below)
        self.local_attention = int(local_attention)
        sc_dim = int(scorer_hidden) if scorer_hidden is not None else int(hidden)
        self.tok_emb = _mb_nn.Embedding(vocab, hidden)
        self.emb_norm = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=False)
        self.layers = [
            _ModernBertBlock(
                i,
                hidden=hidden,
                n_heads=n_heads,
                d_ff=d_ff,
                local_attention=local_attention,
            )
            for i in range(n_layers)
        ]
        self.final_norm = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=False)
        self.type_emb = _mb_nn.Embedding(3, hidden)
        self.head_layers = [_LayaDecisionHeadLayer(hidden=hidden, n_heads=n_heads, head_ff=head_ff) for _ in range(2)]
        self.sc_norm = _mb_nn.LayerNorm(hidden, eps=1e-5, bias=True)
        self.sc_l1 = _mb_nn.Linear(hidden, sc_dim, bias=True)
        self.sc_l2 = _mb_nn.Linear(sc_dim, 1, bias=True)

    def forward_frozen(self, input_ids: Any, position_ids: Any, mask_full: Any, mask_local: Any) -> Any:
        h = self.emb_norm(self.tok_emb(input_ids))
        for i in range(self.freeze_below):
            lyr = self.layers[i]
            m = mask_full if lyr.is_global else mask_local
            h = lyr(h, position_ids, m)
        return _mb_mx.stop_gradient(h)

    def forward_upper(
        self,
        h: Any,
        position_ids: Any,
        mask_full: Any,
        mask_local: Any,
        qtypes: Any,
        marker_positions: Any,
        *,
        fp32_head: bool = False,
    ) -> Any:
        for i in range(self.freeze_below, len(self.layers)):
            lyr = self.layers[i]
            m = mask_full if lyr.is_global else mask_local
            h = lyr(h, position_ids, m)
        hn = self.final_norm(h) + self.type_emb(qtypes)
        for hl in self.head_layers:
            hn = hl(hn, mask_full)
        B, S, H = hn.shape
        n_items, k_opts = marker_positions.shape
        gathered = _mb_mx.take(hn.reshape(B * S, H), marker_positions.reshape(-1), axis=0).reshape(n_items, k_opts, H)
        if fp32_head:
            gathered = gathered.astype(_mb_mx.float32)
        sc = self.sc_l2(_mb_nn.gelu(self.sc_l1(self.sc_norm(gathered)))).squeeze(-1)
        return sc


def _build_modernbert_2d_masks_np(
    seg: np.ndarray,
    pos: np.ndarray,
    *,
    local_attention: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build NaN-safe 4D `[B, 1, S, S]` global and local SWA boolean masks (`pad` attends only to self diagonal)."""
    B, S = seg.shape
    same_seg = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] >= 0)
    diag = np.eye(S, dtype=np.bool_)[None, :, :]
    pad_self = (seg[:, :, None] < 0) & diag
    full_ok = same_seg | pad_self
    half_win = max(1, int(local_attention) // 2)
    within_win = np.abs(pos[:, :, None] - pos[:, None, :]) <= half_win
    local_ok = (same_seg & within_win) | pad_self
    return full_ok[:, None, :, :], local_ok[:, None, :, :]


def _prepare_modernbert_step_pair(
    seqs: Sequence[np.ndarray],
    markers: Sequence[Sequence[int]],
    qtypes: np.ndarray,
    labels: np.ndarray,
    *,
    pack_cap: int = 512,
    local_attention: int = 128,
    dtype: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build matched unpacked vs FFD-packed (`Packed-512`) batches with 2D per-segment RoPE reset and BF16 masks."""
    import mlx.core as mx

    eff_dtype = mx.bfloat16 if dtype is None else dtype
    zero_val = mx.array(0.0, dtype=eff_dtype)
    neg_val = mx.array(-3e4, dtype=eff_dtype)

    n_items = len(seqs)
    k_opts = len(markers[0]) if markers else 2
    max_raw = max(len(s) for s in seqs)
    max_l = max(8, int(math.ceil(max_raw / 32.0) * 32)) if max_raw > 16 else max_raw

    # 1. Unpacked batch
    u_ids = np.zeros((n_items, max_l), dtype=np.int32)
    u_pos = np.zeros((n_items, max_l), dtype=np.int32)
    u_seg = np.full((n_items, max_l), -1, dtype=np.int32)
    u_qt = np.zeros((n_items, max_l), dtype=np.int32)
    u_mark = np.zeros((n_items, k_opts), dtype=np.int32)
    for idx, s in enumerate(seqs):
        L = len(s)
        u_ids[idx, :L] = s
        u_pos[idx, :L] = np.arange(L, dtype=np.int32)
        u_seg[idx, :L] = idx
        u_qt[idx, :L] = int(qtypes[idx])
        for opt_j, m_off in enumerate(markers[idx]):
            u_mark[idx, opt_j] = idx * max_l + int(m_off)
    u_full_ok, u_loc_ok = _build_modernbert_2d_masks_np(u_seg, u_pos, local_attention=local_attention)
    unpacked = {
        "input_ids": mx.array(u_ids),
        "position_ids": mx.array(u_pos),
        "mask_full": mx.where(mx.array(u_full_ok), zero_val, neg_val),
        "mask_local": mx.where(mx.array(u_loc_ok), zero_val, neg_val),
        "qtypes": mx.array(u_qt),
        "marker_positions": mx.array(u_mark),
        "labels": mx.array(np.asarray(labels, dtype=np.int32)),
    }

    # 2. FFD Packed-512 batch with per-segment position_ids reset (0..L-1)
    bins: list[list[tuple[int, np.ndarray]]] = []
    used: list[int] = []
    for idx in sorted(range(n_items), key=lambda k: -len(seqs[k])):
        s = seqs[idx]
        for bi in range(len(bins)):
            if used[bi] + len(s) <= pack_cap:
                bins[bi].append((idx, s))
                used[bi] += len(s)
                break
        else:
            bins.append([(idx, s)])
            used.append(len(s))

    nb = len(bins)
    p_ids = np.zeros((nb, pack_cap), dtype=np.int32)
    p_pos = np.zeros((nb, pack_cap), dtype=np.int32)
    p_seg = np.full((nb, pack_cap), -1, dtype=np.int32)
    p_qt = np.zeros((nb, pack_cap), dtype=np.int32)
    p_mark = np.zeros((n_items, k_opts), dtype=np.int32)
    for bi, bitems in enumerate(bins):
        cur = 0
        for orig_i, s in bitems:
            L = len(s)
            p_ids[bi, cur : cur + L] = s
            p_pos[bi, cur : cur + L] = np.arange(L, dtype=np.int32)
            p_seg[bi, cur : cur + L] = orig_i
            p_qt[bi, cur : cur + L] = int(qtypes[orig_i])
            for opt_j, m_off in enumerate(markers[orig_i]):
                p_mark[orig_i, opt_j] = bi * pack_cap + cur + int(m_off)
            cur += L
    p_full_ok, p_loc_ok = _build_modernbert_2d_masks_np(p_seg, p_pos, local_attention=local_attention)
    packed = {
        "input_ids": mx.array(p_ids),
        "position_ids": mx.array(p_pos),
        "mask_full": mx.where(mx.array(p_full_ok), zero_val, neg_val),
        "mask_local": mx.where(mx.array(p_loc_ok), zero_val, neg_val),
        "qtypes": mx.array(p_qt),
        "marker_positions": mx.array(p_mark),
        "labels": mx.array(np.asarray(labels, dtype=np.int32)),
    }
    return unpacked, packed


def _maybe_load_laya_weights_into_validator(mdl: _ModernBertLargeValidator, raw_w: Mapping[str, Any]) -> None:
    """Populate `_ModernBertLargeValidator` with authentic `convaiinnovations/laya-typed-decisions` weights."""
    pairs: list[tuple[str, Any]] = [
        ("tok_emb.weight", raw_w["encoder.embeddings.tok_embeddings.weight"]),
        ("emb_norm.weight", raw_w["encoder.embeddings.norm.weight"]),
        ("final_norm.weight", raw_w["encoder.final_norm.weight"]),
        ("type_emb.weight", raw_w["type_emb.weight"]),
        ("sc_norm.weight", raw_w["scorer.0.weight"]),
        ("sc_norm.bias", raw_w["scorer.0.bias"]),
        ("sc_l1.weight", raw_w["scorer.1.weight"]),
        ("sc_l1.bias", raw_w["scorer.1.bias"]),
        ("sc_l2.weight", raw_w["scorer.3.weight"]),
        ("sc_l2.bias", raw_w["scorer.3.bias"]),
    ]
    for i in range(len(mdl.layers)):
        px = f"encoder.layers.{i}."
        if i > 0:
            pairs.append((f"layers.{i}.attn_norm.weight", raw_w[px + "attn_norm.weight"]))
        pairs.extend(
            [
                (f"layers.{i}.wqkv.weight", raw_w[px + "attn.Wqkv.weight"]),
                (f"layers.{i}.wo.weight", raw_w[px + "attn.Wo.weight"]),
                (f"layers.{i}.mlp_norm.weight", raw_w[px + "mlp_norm.weight"]),
                (f"layers.{i}.wi.weight", raw_w[px + "mlp.Wi.weight"]),
                (f"layers.{i}.wd.weight", raw_w[px + "mlp.Wo.weight"]),
            ]
        )
    for hi in range(len(mdl.head_layers)):
        hp = f"head.layers.{hi}."
        pairs.extend(
            [
                (f"head_layers.{hi}.norm1.weight", raw_w[hp + "norm1.weight"]),
                (f"head_layers.{hi}.norm1.bias", raw_w[hp + "norm1.bias"]),
                (f"head_layers.{hi}.in_proj.weight", raw_w[hp + "self_attn.in_proj_weight"]),
                (f"head_layers.{hi}.in_proj.bias", raw_w[hp + "self_attn.in_proj_bias"]),
                (f"head_layers.{hi}.out_proj.weight", raw_w[hp + "self_attn.out_proj.weight"]),
                (f"head_layers.{hi}.out_proj.bias", raw_w[hp + "self_attn.out_proj.bias"]),
                (f"head_layers.{hi}.norm2.weight", raw_w[hp + "norm2.weight"]),
                (f"head_layers.{hi}.norm2.bias", raw_w[hp + "norm2.bias"]),
                (f"head_layers.{hi}.linear1.weight", raw_w[hp + "linear1.weight"]),
                (f"head_layers.{hi}.linear1.bias", raw_w[hp + "linear1.bias"]),
                (f"head_layers.{hi}.linear2.weight", raw_w[hp + "linear2.weight"]),
                (f"head_layers.{hi}.linear2.bias", raw_w[hp + "linear2.bias"]),
            ]
        )
    mdl.load_weights(pairs, strict=True)


def _run_encoder_finetune(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.set_cache_limit(256 * 1024 * 1024)
    laya_snap = resolve_hf_snapshot("convaiinnovations/laya-typed-decisions")
    raw_laya_w: dict[str, Any] | None = None
    head_ff = 2048
    scorer_hidden = 1024
    local_attn = 128
    weights_source = "random_init"
    if laya_snap is not None and (laya_snap / "model.safetensors").is_file():
        raw_laya_w = mx.load(str(laya_snap / "model.safetensors"))
        head_ff = int(raw_laya_w["head.layers.0.linear1.weight"].shape[0])
        scorer_hidden = int(raw_laya_w["scorer.1.weight"].shape[0])
        enc_cfg_path = laya_snap / "encoder" / "config.json"
        if enc_cfg_path.is_file():
            enc_cfg = json.loads(enc_cfg_path.read_text(encoding="utf-8"))
            local_attn = int(enc_cfg.get("local_attention", 128))
        weights_source = "convaiinnovations/laya-typed-decisions"

    rng = np.random.default_rng(42)
    n_steps, items_per_step, pack_cap, k_opts = 8, 16, 512, 4
    total_items = n_steps * items_per_step
    u_mlx: list[dict[str, Any]] = []
    p_mlx: list[dict[str, Any]] = []

    for _ in range(n_steps):
        lens = rng.integers(88, 212, size=items_per_step).tolist()
        seqs = [rng.integers(4, 50000, size=int(L), dtype=np.int32) for L in lens]
        markers: list[list[int]] = []
        for s in seqs:
            s[0], s[-1] = 50281, 50282
            step_m = max(2, (len(s) - 4) // k_opts)
            markers.append([1 + j * step_m for j in range(k_opts)])
        qtypes = rng.integers(0, 3, size=items_per_step, dtype=np.int32)
        labels = rng.integers(0, k_opts, size=items_per_step, dtype=np.int32)
        u_step, p_step = _prepare_modernbert_step_pair(
            seqs, markers, qtypes, labels, pack_cap=pack_cap, local_attention=local_attn, dtype=mx.bfloat16
        )
        u_mlx.append(u_step)
        p_mlx.append(p_step)

    def _init_mdl(*, fp32_head: bool) -> _ModernBertLargeValidator:
        mx.random.seed(7)
        m = _ModernBertLargeValidator(
            freeze_below=18,
            head_ff=head_ff,
            scorer_hidden=scorer_hidden,
            local_attention=local_attn,
        )
        if raw_laya_w is not None:
            _maybe_load_laya_weights_into_validator(m, raw_laya_w)
        m.set_dtype(mx.bfloat16)
        if fp32_head:
            m.sc_norm.set_dtype(mx.float32)
            m.sc_l1.set_dtype(mx.float32)
            m.sc_l2.set_dtype(mx.float32)
        m.freeze(recurse=False)
        m.tok_emb.freeze()
        m.emb_norm.freeze()
        for i in range(18):
            m.layers[i].freeze()
        mx.eval(m.parameters())
        return m

    mdl_bf16 = _init_mdl(fp32_head=False)
    mdl_a = _init_mdl(fp32_head=True)
    del raw_laya_w
    gc.collect()
    mx.clear_cache()

    def _loss_bf16(
        m: _ModernBertLargeValidator,
        hi: Any,
        pos_i: Any,
        mf_i: Any,
        ml_i: Any,
        qt_i: Any,
        mk_i: Any,
        yi: Any,
    ) -> Any:
        lg = m.forward_upper(hi, pos_i, mf_i, ml_i, qt_i, mk_i, fp32_head=False).astype(mx.float32)
        return mx.mean(nn.losses.cross_entropy(lg, yi))

    def _loss_a(
        m: _ModernBertLargeValidator,
        hi: Any,
        pos_i: Any,
        mf_i: Any,
        ml_i: Any,
        qt_i: Any,
        mk_i: Any,
        yi: Any,
    ) -> Any:
        lg = m.forward_upper(hi, pos_i, mf_i, ml_i, qt_i, mk_i, fp32_head=True)
        return mx.mean(nn.losses.cross_entropy(lg, yi))

    vg_bf16 = nn.value_and_grad(mdl_bf16, _loss_bf16)
    vg_a = nn.value_and_grad(mdl_a, _loss_a)

    # Step-0 parity: unpacked vs Packed-512 with per-segment RoPE reset + local_attention=128
    b_u0, b_p0 = u_mlx[0], p_mlx[0]
    hu0 = mdl_bf16.forward_frozen(b_u0["input_ids"], b_u0["position_ids"], b_u0["mask_full"], b_u0["mask_local"])
    hp0 = mdl_bf16.forward_frozen(b_p0["input_ids"], b_p0["position_ids"], b_p0["mask_full"], b_p0["mask_local"])
    lu0, gu0 = vg_bf16(
        mdl_bf16, hu0, b_u0["position_ids"], b_u0["mask_full"], b_u0["mask_local"], b_u0["qtypes"], b_u0["marker_positions"], b_u0["labels"]
    )
    lp0, gp0 = vg_bf16(
        mdl_bf16, hp0, b_p0["position_ids"], b_p0["mask_full"], b_p0["mask_local"], b_p0["qtypes"], b_p0["marker_positions"], b_p0["labels"]
    )
    mx.eval(lu0, gu0, lp0, gp0)
    f1, f2 = dict(tree_flatten(gu0)), dict(tree_flatten(gp0))
    dot = sum(float(mx.sum(f1[k].astype(mx.float32) * f2[k].astype(mx.float32)).item()) for k in f1 if k in f2)
    n1 = sum(float(mx.sum(f1[k].astype(mx.float32) ** 2).item()) for k in f1 if k in f2)
    n2 = sum(float(mx.sum(f2[k].astype(mx.float32) ** 2).item()) for k in f1 if k in f2)
    grad_cos = dot / max(math.sqrt(n1 * n2), 1e-12)
    loss_diff = abs(float(lu0.item()) - float(lp0.item()))
    del gu0, gp0, hu0, hp0
    gc.collect()
    mx.clear_cache()

    # Explicitly time cold L0..17 frozen-feature cache build across all packed steps
    _ = mdl_bf16.forward_frozen(b_p0["input_ids"], b_p0["position_ids"], b_p0["mask_full"], b_p0["mask_local"])
    mx.synchronize()
    t_cache0 = time.perf_counter()
    cached_p: list[dict[str, Any]] = []
    for bp in p_mlx:
        hf = mdl_bf16.forward_frozen(bp["input_ids"], bp["position_ids"], bp["mask_full"], bp["mask_local"])
        mx.eval(hf)
        cached_p.append({**bp, "cached_h": hf})
    mx.synchronize()
    cold_cache_build_sec = time.perf_counter() - t_cache0

    def _make_compiled_step(m: _ModernBertLargeValidator) -> tuple[Any, list[Any]]:
        opt = optim.AdamW(learning_rate=2e-4, weight_decay=0.01)
        st = [m.state, opt.state]

        def _raw_step(hi: Any, pos_i: Any, mf_i: Any, ml_i: Any, qt_i: Any, mk_i: Any, yi: Any) -> Any:
            l_val, g_val = vg_bf16(m, hi, pos_i, mf_i, ml_i, qt_i, mk_i, yi)
            opt.update(m, g_val)
            return l_val

        return mx.compile(_raw_step, inputs=st, outputs=st), st

    def _run_stock_a() -> dict[str, float]:
        opt = optim.AdamW(learning_rate=2e-4, weight_decay=0.01)
        hf0 = mdl_a.forward_frozen(b_u0["input_ids"], b_u0["position_ids"], b_u0["mask_full"], b_u0["mask_local"])
        lw, gw = vg_a(
            mdl_a, hf0, b_u0["position_ids"], b_u0["mask_full"], b_u0["mask_local"], b_u0["qtypes"], b_u0["marker_positions"], b_u0["labels"]
        )
        opt.update(mdl_a, gw)
        mx.eval(lw, mdl_a.parameters(), opt.state)
        mx.synchronize()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for bu in u_mlx:
            hf = mdl_a.forward_frozen(bu["input_ids"], bu["position_ids"], bu["mask_full"], bu["mask_local"])
            lv, gv = vg_a(
                mdl_a, hf, bu["position_ids"], bu["mask_full"], bu["mask_local"], bu["qtypes"], bu["marker_positions"], bu["labels"]
            )
            opt.update(mdl_a, gv)
            mx.eval(lv, mdl_a.parameters(), opt.state)
        mx.synchronize()
        dt = time.perf_counter() - t0
        peak_bytes = float(mx.get_peak_memory())
        return {
            "items_s": total_items / max(dt, 1e-9),
            "elapsed_sec": dt,
            "peak_gb": peak_bytes / 1e9,
            "peak_gib": peak_bytes / (1024**3),
        }

    def _run_compiled(batches: list[dict[str, Any]], *, pre_cached: bool) -> dict[str, float]:
        step_fn, st = _make_compiled_step(mdl_bf16)
        b0 = batches[0]
        hf0 = (
            b0["cached_h"]
            if pre_cached
            else mdl_bf16.forward_frozen(b0["input_ids"], b0["position_ids"], b0["mask_full"], b0["mask_local"])
        )
        lw = step_fn(hf0, b0["position_ids"], b0["mask_full"], b0["mask_local"], b0["qtypes"], b0["marker_positions"], b0["labels"])
        mx.eval(lw, st)
        mx.synchronize()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for item in batches:
            hf = (
                item["cached_h"]
                if pre_cached
                else mdl_bf16.forward_frozen(item["input_ids"], item["position_ids"], item["mask_full"], item["mask_local"])
            )
            lv = step_fn(
                hf, item["position_ids"], item["mask_full"], item["mask_local"], item["qtypes"], item["marker_positions"], item["labels"]
            )
            mx.eval(lv, st)
        mx.synchronize()
        dt = time.perf_counter() - t0
        peak_bytes = float(mx.get_peak_memory())
        return {
            "items_s": total_items / max(dt, 1e-9),
            "elapsed_sec": dt,
            "peak_gb": peak_bytes / 1e9,
            "peak_gib": peak_bytes / (1024**3),
        }

    runners = {
        "stock_unpacked_fp32_head": _run_stock_a,
        "bf16_compiled": lambda: _run_compiled(u_mlx, pre_cached=False),
        "packed_512": lambda: _run_compiled(p_mlx, pre_cached=False),
        "packed_512_cached_L0_17": lambda: _run_compiled(cached_p, pre_cached=True),
    }
    names = list(runners.keys())
    samples: dict[str, list[dict[str, float]]] = {k: [] for k in names}
    for r in range(max(1, int(args.abba_rounds))):
        seq = names if r % 2 == 0 else list(reversed(names))
        for k in seq:
            samples[k].append(runners[k]())
            mx.clear_cache()

    med = {
        k: {
            "items_s": round(float(statistics.median(x["items_s"] for x in v)), 2),
            "elapsed_sec": round(float(statistics.median(x["elapsed_sec"] for x in v)), 4),
            "peak_gb": round(float(statistics.median(x["peak_gb"] for x in v)), 3),
            "peak_gib": round(float(statistics.median(x["peak_gib"] for x in v)), 3),
        }
        for k, v in samples.items()
    }
    base_ips = med["stock_unpacked_fp32_head"]["items_s"]
    p512_ips = med["packed_512"]["items_s"]
    cached_ips = med["packed_512_cached_L0_17"]["items_s"]
    cached_ep_sec = max(1e-9, float(med["packed_512_cached_L0_17"]["elapsed_sec"]))
    ep1_incl_ips = round(total_items / max(cold_cache_build_sec + cached_ep_sec, 1e-9), 2)
    amort_3ep_ips = round((3 * total_items) / max(cold_cache_build_sec + 3.0 * cached_ep_sec, 1e-9), 2)
    amort_5ep_ips = round((5 * total_items) / max(cold_cache_build_sec + 5.0 * cached_ep_sec, 1e-9), 2)
    parity_ok = float(loss_diff) <= 1e-2 and float(grad_cos) >= 0.995
    return {
        "suite": "encoder-finetune",
        "status": "ok" if parity_ok else "failed",
        "weights_source": weights_source,
        "fast_metric": (
            f"Packed-512 {p512_ips:.1f} items/s (cached L0..17 warm {cached_ips:.1f} items/s; "
            f"cold build {cold_cache_build_sec:.2f}s, 3-ep amortized {amort_3ep_ips:.1f} items/s)"
        ),
        "stock_baseline": f"Stock MLX unpacked: {base_ips:.1f} items/s",
        "speedup_vs_stock": round(p512_ips / max(base_ips, 1e-6), 2),
        "cached_speedup_vs_stock": round(cached_ips / max(base_ips, 1e-6), 2),
        "amortized_3ep_speedup_vs_stock": round(amort_3ep_ips / max(base_ips, 1e-6), 2),
        "cold_cache_build_sec": round(cold_cache_build_sec, 4),
        "epoch1_with_cache_build_items_s": ep1_incl_ips,
        "amortized_3ep_items_s": amort_3ep_ips,
        "amortized_5ep_items_s": amort_5ep_ips,
        "peak_gb": med["packed_512"]["peak_gb"],
        "stock_peak_gb": med["stock_unpacked_fp32_head"]["peak_gb"],
        "parity": {"abs_loss_diff": round(loss_diff, 6), "grad_cosine": round(grad_cos, 6)},
        "regimes": med,
    }


# ---------------------------------------------------------------------------
# 3. GRPO / Best-of-N Rolling Rollouts + Concurrent ANE Critic
# ---------------------------------------------------------------------------


def _run_grpo_ane_critic(args: argparse.Namespace) -> dict[str, Any]:
    from fast_decision_models.laya_ane import LayaAneBackend
    from fast_decision_models.paths import ensure_laya_dylib, resolve_laya_dir
    from fast_gemma4.cli import _load_fixture_dataset
    from fast_gemma4.engine import FastGemma4Engine
    from fast_gemma4.paths import resolve_paths as resolve_g4_paths

    fg4_home = resolve_pkg_home("fast-gemma4", "FAST_GEMMA4_HOME")
    fdm_home = resolve_pkg_home("fast-decision-models", "FAST_DECISION_MODELS_HOME")
    laya_dir = resolve_laya_dir(None, artifacts_dir=fdm_home / "artifacts")
    if not (laya_dir / "laya_typed_s512_w8.mlmodelc").exists():
        return {"suite": "grpo-ane-critic", "status": "skipped", "reason": f"missing Laya ANE bundles in {laya_dir}"}

    sys_prompt, schema_str, base_prompts = _load_fixture_dataset(resolve_g4_paths(fg4_home))
    n_docs, group_sz, max_toks = 4, 4, 96
    cand_prompts = [base_prompts[i] for i in range(n_docs) for _ in range(group_sz)]
    qdef = {
        "id": "schema_grounding",
        "type": "noul",
        "instructions": "Is the extracted JSON response complete, non-empty, and grounded in the source text?",
        "criteria": {
            "false": "The output is malformed, truncated, or contradicts the source text.",
            "true": "The output is valid JSON and grounded in the source text.",
        },
    }

    # Optimized Leg B: FastGemma4Engine rolling B=16 + overlapped Laya ANE critic
    eng = FastGemma4Engine(home=fg4_home, default_mode="gpu", kv_i8=True, local_files_only=True, warmup=True, warmup_ane=False)
    laya = LayaAneBackend(laya_dir, ensure_laya_dylib(), pipe_depth=4)
    saved_rollouts: list[dict[str, Any]] = []
    try:
        _ = eng.generate_batch(cand_prompts[:4], system_prompt=sys_prompt, json_schema=schema_str, max_tokens=32, mode="gpu")
        _ = laya.score_noul_batch([laya.encode_item("warmup", qdef, on_truncate="ignore")])

        b_times_stoch: list[float] = []
        b_times_bon: list[float] = []
        exposed_ane_ms: list[float] = []
        for _ in range(max(1, int(args.abba_rounds))):
            t0 = time.perf_counter()
            res_s = eng.generate_batch(
                cand_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=max_toks,
                temperature=0.7,
                mode="gpu",
                batch_size=len(cand_prompts),
            )
            encs = [
                laya.encode_item({"source": p[:600], "candidate": it.text}, qdef, on_truncate="ignore")
                for p, it in zip(cand_prompts, res_s.items)
            ]
            t_c0 = time.perf_counter()
            scores = laya.score_noul_batch(encs)
            exposed_ane_ms.append((time.perf_counter() - t_c0) * 1000.0)
            b_times_stoch.append(time.perf_counter() - t0)
            if not saved_rollouts:
                mean_s = float(np.mean(scores))
                std_s = max(float(np.std(scores)), 1e-3)
                saved_rollouts = [
                    {"text": it.text, "token_ids": list(it.token_ids), "advantage": float((sc - mean_s) / std_s)}
                    for it, sc in zip(res_s.items, scores)
                ]

            t1 = time.perf_counter()
            res_b = eng.generate_batch(
                cand_prompts,
                system_prompt=sys_prompt,
                json_schema=schema_str,
                max_tokens=max_toks,
                temperature=0.0,
                mode="gpu",
                batch_size=len(cand_prompts),
            )
            encs_b = [
                laya.encode_item({"source": p[:600], "candidate": it.text}, qdef, on_truncate="ignore")
                for p, it in zip(cand_prompts, res_b.items)
            ]
            _ = laya.score_noul_batch(encs_b)
            b_times_bon.append(time.perf_counter() - t1)
    finally:
        eng.close()
        laya.close()

    # Stock Baseline A in `--public-python`: sequential `mlx-vlm` 4-bit + stock GPU `kev-0.8b-mlx-8bit` critic
    pub_py = args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    stock_res: dict[str, Any] | None = None
    if Path(pub_py).is_file():
        stock_code = r"""
import json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_vlm

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sys.path.insert(0, cfg["scripts_dir"])
from public_baselines import resolve_hf_snapshot
from _outcome_public_backends import PublicKev08bMlxBackend

s_g4 = resolve_hf_snapshot("lmstudio-community/gemma-4-E2B-it-MLX-4bit")
s_k8 = resolve_hf_snapshot("AbstractFramework/kev-0.8b-mlx-8bit")
if s_g4 is None or s_k8 is None:
    print(json.dumps({"status": "skipped"}))
    raise SystemExit(0)

model, proc = mlx_vlm.load(str(s_g4))
tok = getattr(proc, "tokenizer", proc)
fmts = []
for p in cfg["cand_prompts"]:
    msgs = [{"role": "system", "content": cfg["sys_prompt"]}, {"role": "user", "content": p}]
    try:
        fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    except TypeError:
        fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

_ = mlx_vlm.generate(model, proc, fmts[0], max_tokens=16, temp=0.0, verbose=False)
mx.synchronize()
t0 = time.perf_counter()
outs = [getattr(mlx_vlm.generate(model, proc, f, max_tokens=cfg["max_toks"], temp=0.7, verbose=False), "text", "") for f in fmts]
mx.synchronize()
roll_s = time.perf_counter() - t0
del model, proc
mx.clear_cache()

critic = PublicKev08bMlxBackend(Path(cfg["kev_dir"]), batch_docs=1)
try:
    _ = critic.score_multi_states([({"source": "warmup", "candidate": "warmup"}, [cfg["qdef"]])])
    mx.synchronize()
    t1 = time.perf_counter()
    for p, tx in zip(cfg["cand_prompts"], outs, strict=True):
        critic.score_multi_states([({"source": p[:600], "candidate": tx}, [cfg["qdef"]])])
    mx.synchronize()
    crit_s = time.perf_counter() - t1
finally:
    critic.close()
tot_s = roll_s + crit_s
print(json.dumps({"status": "ok", "wall_sec": round(tot_s, 3), "rollout_sec": round(roll_s, 3), "critic_sec": round(crit_s, 3), "rollouts_s": round(len(fmts) / max(tot_s, 1e-6), 3)}))
"""
        with tempfile.TemporaryDirectory(prefix="grpo_stock_") as td:
            cf = Path(td) / "cfg.json"
            cf.write_text(
                json.dumps(
                    {
                        "scripts_dir": str(_REPO_ROOT / "scripts"),
                        "kev_dir": str(fdm_home / "artifacts" / "kev_08b"),
                        "sys_prompt": sys_prompt,
                        "cand_prompts": cand_prompts,
                        "max_toks": max_toks,
                        "qdef": qdef,
                    }
                ),
                encoding="utf-8",
            )
            try:
                pr = run_subprocess_with_killpg([str(pub_py), "-c", stock_code, str(cf)], timeout_sec=600)
                if pr.returncode == 0 and pr.stdout.strip():
                    stock_res = json.loads(pr.stdout.strip().splitlines()[-1])
                else:
                    stock_res = {
                        "status": "failed",
                        "returncode": pr.returncode,
                        "reason": (pr.stderr or pr.stdout or "").strip()[-1000:] or f"exit={pr.returncode}",
                    }
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                stock_res = {"status": "failed", "reason": str(exc)}
            if stock_res.get("status") == "failed":
                low_rsn = str(stock_res.get("reason") or "").lower()
                if "no module named 'mlx_vlm'" in low_rsn or "no module named 'mlx_lm'" in low_rsn or "modulenotfounderror" in low_rsn:
                    stock_res["status"] = "skipped"

    # Real GRPO policy step + concurrent Laya ANE critic overlap in `--lora-python`
    lora_py = args.lora_python or pub_py
    grpo_step_res: dict[str, Any] | None = None
    if Path(lora_py).is_file() and saved_rollouts:
        grpo_code = r"""
import gc, json, sys, threading, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.gemma4_text import logit_softcap
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load_model

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for p in cfg["py_paths"]:
    if p not in sys.path:
        sys.path.insert(0, p)
from public_baselines import resolve_hf_snapshot
from fast_gemma4.lora_accel import quantize_frozen_trunk_8bit, unwrap_kv_shared_lora_modules
from fast_gemma4.tokenizer import FastGemma4Tokenizer
from fast_decision_models.laya_ane import LayaAneBackend
from fast_decision_models.paths import ensure_laya_dylib

snap = resolve_hf_snapshot("mlx-community/gemma-4-e2b-it-bf16")
if snap is None:
    print(json.dumps({"status": "skipped"}))
    raise SystemExit(0)

low_ram = bool(cfg["low_ram"])
mx.set_memory_limit(int((18.0 if low_ram else 36.0) * (1024**3)))
mx.set_cache_limit(256 * 1024 * 1024)
mdl, _ = load_model(snap, strict=False)
for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
    if hasattr(mdl, attr):
        setattr(mdl, attr, None)
if low_ram:
    quantize_frozen_trunk_8bit(mdl, group_size=64, bits=8, quantize_ple_embedding=True, quantize_embed_tokens=False)
mdl.freeze()
first_l = 12
linear_to_lora_layers(mdl, num_layers=len(mdl.layers) - first_l, config={"rank": 24, "scale": 2.0, "dropout": 0.0})
unwrap_kv_shared_lora_modules(mdl)
mx.eval(mdl.parameters())

tok = FastGemma4Tokenizer(snap / "tokenizer.json")
g_seqs = []
for p_txt, it in zip(cfg["cand_prompts"][:4], cfg["saved_rollouts"][:4]):
    p_ids = [tok.bos_token_id] + tok.encode(cfg["sys_prompt"] + "\n\n" + p_txt, add_special_tokens=False)[:320]
    c_ids = list(it["token_ids"][:96]) or [tok.bos_token_id]
    g_seqs.append((p_ids + c_ids, len(p_ids), float(it["advantage"])))

max_in = max(len(s) - 1 for s, _, _ in g_seqs)
in_mx = mx.array([s[:-1] + [0] * (max_in - (len(s) - 1)) for s, _, _ in g_seqs], dtype=mx.int32)
spans = [(pl - 1, len(s) - 1) for s, pl, _ in g_seqs]
y_mx = mx.array([tid for s, pl, _ in g_seqs for tid in s[pl:]], dtype=mx.int32)
adv_mx = mx.array([av for s, pl, av in g_seqs for _ in s[pl:]], dtype=mx.float32)

tm = mdl.language_model.model
emb = tm.embed_tokens(in_mx)
h_p = emb * tm.embed_scale
pli = tm._project_per_layer_inputs(h_p, tm._get_per_layer_inputs(in_mx, emb))
pli_l = [pli[:, :, i, :] for i in range(len(tm.layers))]
masks = tm._make_masks(h_p, [None] * len(tm.layers))
inter_p = [(None, None)] * len(tm.layers)
for idx in range(first_l):
    kvs, off = inter_p[tm.previous_kvs[idx]]
    h_p, kvs, off = tm.layers[idx](h_p, masks[idx], None, per_layer_input=pli_l[idx], shared_kv=kvs, offset=off)
    inter_p[idx] = (kvs, off)

def _logprobs(m):
    lm = m.language_model
    tmod = lm.model
    inter = list(inter_p)
    h = h_p
    for idx in range(first_l, len(tmod.layers)):
        kvs, off = inter[tmod.previous_kvs[idx]]
        h, kvs, off = tmod.layers[idx](h, masks[idx], None, per_layer_input=pli_l[idx], shared_kv=kvs, offset=off)
        inter[idx] = (kvs, off)
    hn = tmod.norm(h)
    hans = mx.concatenate([hn[bi, s0:s1, :] for bi, (s0, s1) in enumerate(spans)], axis=0)
    lps = []
    for c0 in range(0, int(hans.shape[0]), 256):
        c1 = min(int(hans.shape[0]), c0 + 256)
        lg = tmod.embed_tokens.as_linear(hans[c0:c1]) if lm.tie_word_embeddings else lm.lm_head(hans[c0:c1])
        if lm.final_logit_softcapping is not None:
            lg = logit_softcap(lm.final_logit_softcapping, lg)
        lg32 = lg.astype(mx.float32)
        lps.append(mx.take_along_axis(lg32, y_mx[c0:c1, None], axis=-1).squeeze(-1) - mx.logsumexp(lg32, axis=-1))
    return mx.concatenate(lps, axis=0)

old_lp = mx.stop_gradient(_logprobs(mdl))
mx.eval(h_p, old_lp)

def _loss(m):
    nlp = _logprobs(m)
    rat = mx.exp(mx.clip(nlp - old_lp, -10.0, 10.0))
    surr = mx.minimum(rat * adv_mx, mx.clip(rat, 0.8, 1.2) * adv_mx)
    kl = mx.exp(mx.clip(old_lp - nlp, -10.0, 10.0)) - (old_lp - nlp) - 1.0
    return -mx.mean(surr - 0.04 * kl)

vag = nn.value_and_grad(mdl, _loss)
opt = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)
laya = LayaAneBackend(Path(cfg["laya_dir"]), ensure_laya_dylib(), pipe_depth=4)
c_encs = [laya.encode_item({"source": p[:600], "candidate": it["text"]}, cfg["qdef"], on_truncate="ignore") for p, it in zip(cfg["cand_prompts"], cfg["saved_rollouts"])]
lw, gw = vag(mdl)
mx.eval(lw, gw)
_ = laya.score_noul_batch(c_encs[:2])

ane_box = {}
def _ane_job():
    t_a = time.perf_counter()
    _ = laya.score_noul_batch(c_encs)
    ane_box["ms"] = (time.perf_counter() - t_a) * 1000.0

t0 = time.perf_counter()
th = threading.Thread(target=_ane_job, daemon=True)
th.start()
lv, gv = vag(mdl)
opt.update(mdl, gv)
mx.eval(lv, mdl.parameters(), opt.state)
gpu_ms = (time.perf_counter() - t0) * 1000.0
th.join()
wall_ms = (time.perf_counter() - t0) * 1000.0
laya.close()
print(json.dumps({"status": "ok", "grpo_gpu_ms": round(gpu_ms, 2), "ane_critic_16_ms": round(ane_box["ms"], 2), "exposed_ane_ms": round(max(0.0, wall_ms - gpu_ms), 2)}))
"""
        with tempfile.TemporaryDirectory(prefix="grpo_step_") as td2:
            cf2 = Path(td2) / "gcfg.json"
            cf2.write_text(
                json.dumps(
                    {
                        "py_paths": [
                            str(_REPO_ROOT / "scripts"),
                            str(_REPO_ROOT / "packages" / "fast-shared" / "src"),
                            str(_REPO_ROOT / "packages" / "fast-gemma4" / "src"),
                            str(_REPO_ROOT / "packages" / "fast-decision-models" / "src"),
                        ],
                        "low_ram": 0 < _get_host_ram_gb() <= 28.5,
                        "laya_dir": str(laya_dir),
                        "sys_prompt": sys_prompt,
                        "cand_prompts": cand_prompts,
                        "saved_rollouts": saved_rollouts,
                        "qdef": qdef,
                    }
                ),
                encoding="utf-8",
            )
            try:
                pr2 = run_subprocess_with_killpg([str(lora_py), "-c", grpo_code, str(cf2)], timeout_sec=600)
                if pr2.returncode == 0 and pr2.stdout.strip():
                    grpo_step_res = json.loads(pr2.stdout.strip().splitlines()[-1])
                else:
                    grpo_step_res = {
                        "status": "failed",
                        "returncode": pr2.returncode,
                        "reason": (pr2.stderr or pr2.stdout or "").strip()[-1000:] or f"exit={pr2.returncode}",
                    }
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                grpo_step_res = {"status": "failed", "reason": str(exc)}

    med_stoch_s = float(statistics.median(b_times_stoch))
    med_bon_s = float(statistics.median(b_times_bon))
    b_rps = len(cand_prompts) / max(med_stoch_s, 1e-6)
    bon_rps = len(cand_prompts) / max(med_bon_s, 1e-6)
    a_rps = float((stock_res or {}).get("rollouts_s") or 0.0)
    fp = get_unified_memory_footprint()

    return {
        "suite": "grpo-ane-critic",
        "status": "ok",
        "fast_metric": f"{b_rps:.2f} rollouts/s (T=0.7 GRPO) | {bon_rps:.2f} rollouts/s (T=0.0 BoN)",
        "stock_baseline": f"mlx-vlm + Kev-8bit GPU: {a_rps:.2f} rollouts/s" if a_rps > 0 else "stock unavailable",
        "speedup_vs_stock": round(b_rps / max(a_rps, 1e-6), 2) if a_rps > 0 else None,
        "bon_speedup_vs_stock": round(bon_rps / max(a_rps, 1e-6), 2) if a_rps > 0 else None,
        "peak_gb": round(float(fp["peak_footprint_gb"]), 3),
        "grpo_step_overlap": grpo_step_res,
        "stock_detail": stock_res,
    }


# ---------------------------------------------------------------------------
# 4. MTP Speculative Decoding + DraftScan (`FastGemma4Engine(mtp_drafter=True)`)
# ---------------------------------------------------------------------------


def _measure_mtp_pair(generate_ar: Callable[[], Any], generate_mtp: Callable[[], Any], *, batch_size: int, rounds: int) -> dict[str, Any]:
    generators = {"ar": generate_ar, "mtp": generate_mtp}
    for generate in generators.values():
        generate()
    samples: dict[str, list[dict[str, Any]]] = {"ar": [], "mtp": []}
    reference = None
    exact = [True] * batch_size
    for round_idx in range(max(1, rounds)):
        order = ("ar", "mtp", "mtp", "ar") if round_idx % 2 == 0 else ("mtp", "ar", "ar", "mtp")
        for arm in order:
            started = time.perf_counter()
            batch = generators[arm]()
            seconds = time.perf_counter() - started
            if len(batch.items) != batch_size:
                raise ValueError(f"{arm} returned {len(batch.items)} completions for {batch_size} prompts")
            tokens = [list(item.token_ids) for item in batch.items]
            if reference is None:
                reference = tokens  # First timed AR output for each prompt.
            matches = [ids == expected for ids, expected in zip(tokens, reference, strict=True)]
            exact = [all(pair) for pair in zip(exact, matches, strict=True)]
            counts = [len(ids) for ids in tokens]
            samples[arm].append({
                "elapsed_seconds": seconds, "output_tokens_by_prompt": counts, "output_tokens": sum(counts),
                "exact_vs_first_ar_by_prompt": matches,
            })

    med_seconds = {arm: statistics.median(r["elapsed_seconds"] for r in runs) for arm, runs in samples.items()}
    rates = {arm: statistics.median(r["output_tokens"] / r["elapsed_seconds"] for r in runs) for arm, runs in samples.items()}
    stable_totals = {}
    for arm, runs in samples.items():
        totals = {r["output_tokens"] for r in runs}
        stable_totals[arm] = totals.pop() if len(totals) == 1 else None
    elapsed_ratio = med_seconds["ar"] / med_seconds["mtp"]
    return {
        "ar_tok_s": rates["ar"], "mtp_tok_s": rates["mtp"],
        "ar_output_tokens": stable_totals["ar"], "mtp_output_tokens": stable_totals["mtp"],
        "ar_median_seconds": med_seconds["ar"], "mtp_median_seconds": med_seconds["mtp"],
        "elapsed_speedup_vs_ar": elapsed_ratio, "token_rate_ratio_vs_ar": rates["mtp"] / rates["ar"] if rates["ar"] else None,
        "exact_output_by_prompt": exact, "samples": samples,
        # Historical fields: speedup meant elapsed time and total_out_tokens meant MTP output.
        "speedup_vs_ar": round(elapsed_ratio, 3), "total_out_tokens": stable_totals["mtp"],
    }


def _stock_generated_token_count(result: Any, batch_size: int) -> int:
    if batch_size > 1 and len(result.texts) != batch_size:
        raise ValueError("Stock mlx-vlm returned the wrong number of completions")
    value = getattr(result, "generation_tokens", None)
    if value is None:
        value = getattr(getattr(result, "stats", None), "generation_tokens", None)
    if isinstance(value, list) and len(value) == batch_size and all(isinstance(n, int) and n >= 0 for n in value):
        return sum(value)
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError("Stock mlx-vlm did not expose a generated-token count")


def _run_mtp_speculative(args: argparse.Namespace) -> dict[str, Any]:
    from fast_gemma4.cli import _load_fixture_dataset
    from fast_gemma4.engine import FastGemma4Engine
    from fast_gemma4.mtp_drafter import resolve_mtp_drafter_dir
    from fast_gemma4.paths import resolve_paths as resolve_g4_paths

    try:
        drafter_dir = resolve_mtp_drafter_dir(local_files_only=True)
    except Exception as exc:
        return {"suite": "mtp-speculative", "status": "skipped", "reason": f"missing MTP drafter ({exc})"}

    fg4_home = resolve_pkg_home("fast-gemma4", "FAST_GEMMA4_HOME")
    sys_prompt, schema_str, json_prompts = _load_fixture_dataset(resolve_g4_paths(fg4_home))
    free_prompts = [
        "Explain step by step how a two-phase commit protocol works in distributed databases and how Raft avoids blocking.",
        "Compare copy-on-write B-trees versus log-structured merge-trees for write-heavy NVMe time-series storage.",
        "Analyze three architectural reasons why early RISC pipelines outperformed CISC designs.",
        "Write an engineering root-cause analysis template for a cascading retry storm triggered by stale DNS TTLs.",
    ]

    eng = FastGemma4Engine(home=fg4_home, default_mode="gpu", kv_i8=True, mtp_drafter=drafter_dir, local_files_only=True, warmup=True, warmup_ane=False)
    workloads: dict[str, Any] = {}
    try:
        for wname, plist, sys_p, sch, mtok in (
            ("free_b1", free_prompts[:1], None, None, 192),
            ("free_b4", free_prompts[:4], None, None, 192),
            ("json_b1", json_prompts[:1], sys_prompt, schema_str, 256),
            ("json_b4", json_prompts[:4], sys_prompt, schema_str, 256),
        ):
            settings = dict(system_prompt=sys_p, json_schema=sch, max_tokens=mtok, temperature=0.0, mode="gpu", batch_size=len(plist))
            workloads[wname] = _measure_mtp_pair(
                partial(eng.generate_batch, plist, speculative=False, **settings),
                partial(eng.generate_batch, plist, speculative=True, **settings),
                batch_size=len(plist), rounds=args.abba_rounds,
            )
    finally:
        eng.close()

    # Stock `mlx-vlm` B=1 & B=4 on the exact same JSON prompts
    pub_py = args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    stock_vlm: dict[str, Any] = {"status": "unavailable", "reason": f"Missing stock interpreter {pub_py}"}
    if Path(pub_py).is_file():
        vlm_code = r"""
import json, statistics, sys, time
from pathlib import Path
import mlx.core as mx, mlx_vlm
sys.path.insert(0, sys.argv[2])
from public_baselines import resolve_hf_snapshot
from benchmark_capabilities import _stock_generated_token_count

snap = resolve_hf_snapshot("lmstudio-community/gemma-4-E2B-it-MLX-4bit")
if snap is None:
    print(json.dumps({"status": "skipped"}))
    raise SystemExit(0)
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model, proc = mlx_vlm.load(str(snap))
tok = getattr(proc, "tokenizer", proc)
fmts = []
for p in cfg["json_prompts"]:
    msgs = [{"role": "system", "content": cfg["sys_prompt"]}, {"role": "user", "content": p}]
    try:
        fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    except TypeError:
        fmts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

_ = mlx_vlm.generate(model, proc, fmts[0], max_tokens=32, temp=0.0, verbose=False)
mx.synchronize()
result = {"status": "ok", "generation_mode": "Same JSON prompts with unrestricted decoding"}
for batch_size in (1, 4):
    key = f"json_b{batch_size}"
    samples = []
    for _ in range(2):
        t0 = time.perf_counter()
        if batch_size == 1:
            out = mlx_vlm.generate(model, proc, fmts[0], max_tokens=256, temp=0.0, verbose=False)
        else:
            out = mlx_vlm.batch_generate(model, proc, prompts=fmts[:batch_size], max_tokens=256, verbose=False)
        mx.synchronize()
        seconds = time.perf_counter() - t0
        samples.append({"elapsed_seconds": seconds, "output_tokens": _stock_generated_token_count(out, batch_size)})
    result[key] = {"samples": samples, "median_seconds": statistics.median(r["elapsed_seconds"] for r in samples)}
    result[key + "_tok_s"] = statistics.median(r["output_tokens"] / r["elapsed_seconds"] for r in samples)
print(json.dumps(result))
"""
        with tempfile.TemporaryDirectory(prefix="mtp_stock_") as td:
            cf = Path(td) / "vcfg.json"
            cf.write_text(json.dumps({"sys_prompt": sys_prompt, "json_prompts": json_prompts[:4]}), encoding="utf-8")
            try:
                pr = subprocess.run([str(pub_py), "-c", vlm_code, str(cf), str(_REPO_ROOT / "scripts")], capture_output=True, text=True, timeout=300, check=False)
                if pr.returncode == 0 and pr.stdout.strip():
                    stock_vlm = json.loads(pr.stdout.strip().splitlines()[-1])
                else:
                    stock_vlm = {"status": "unavailable", "reason": pr.stderr.strip()[-1000:] or "No stock result"}
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                stock_vlm = {"status": "unavailable", "reason": str(exc)}

    fp = get_unified_memory_footprint()
    jb1 = workloads["json_b1"]["mtp_tok_s"]
    jb4 = workloads["json_b4"]["mtp_tok_s"]
    for name in ("json_b1", "json_b4"):
        row = workloads[name]
        stock_rate = stock_vlm.get(name + "_tok_s")
        stock_seconds = (stock_vlm.get(name) or {}).get("median_seconds")
        row["token_rate_ratio_vs_stock"] = row["mtp_tok_s"] / stock_rate if stock_rate else None
        row["elapsed_speedup_vs_stock"] = stock_seconds / row["mtp_median_seconds"] if stock_seconds else None
    sb1 = stock_vlm.get("json_b1_tok_s")
    sb4 = stock_vlm.get("json_b4_tok_s")
    exact = [flag for row in workloads.values() for flag in row["exact_output_by_prompt"]]
    exact_ok = bool(exact) and all(exact)
    return {
        "suite": "mtp-speculative",
        "status": "ok" if exact_ok else "failed",
        "fast_metric": (
            f"Constrained JSON B=1 {jb1:.1f} output tokens/s ({workloads['json_b1']['elapsed_speedup_vs_ar']:.2f}x elapsed vs AR), "
            f"B=4 {jb4:.1f} tokens/s. Exact AR/MTP outputs on {sum(exact)}/{len(exact)} prompts across workloads."
        ),
        "stock_baseline": f"Unrestricted mlx-vlm B=1 {sb1:.1f}, B=4 {sb4:.1f} output tokens/s" if sb1 is not None and sb4 is not None else "stock unavailable",
        "speedup_vs_stock": workloads["json_b1"]["token_rate_ratio_vs_stock"],
        "b4_speedup_vs_stock": workloads["json_b4"]["token_rate_ratio_vs_stock"],
        "comparison": "Project rates count the token IDs returned in each timed pass. Stock rates use the library's generated-token counter. Elapsed speedup divides median wall times. Each exact-output flag requires identical token IDs on that prompt across every timed AR and MTP pass. The stock ratio compares output-token rates under different decoding constraints.",
        "peak_gb": round(float(fp["peak_footprint_gb"]), 3),
        "greedy_exact_token_matches": f"{sum(exact)}/{len(exact)}",
        "workloads": workloads,
        "stock_mlx_vlm": stock_vlm,
    }


# ---------------------------------------------------------------------------
# 5. Kev-0.8B `PrefixTree` (`256`-tok `GDN+KV` State Cache)
# ---------------------------------------------------------------------------


def _run_prefix_tree(args: argparse.Namespace) -> dict[str, Any]:
    from _kev_benchmark import cache_probe_inputs, score_differences
    from fast_decision_models.paths import resolve_kev_dir

    fdm_home = resolve_pkg_home("fast-decision-models", "FAST_DECISION_MODELS_HOME")
    kev_dir = resolve_kev_dir(None, artifacts_dir=fdm_home / "artifacts")
    try:
        records = cache_probe_inputs()
    except FileNotFoundError as exc:
        return {"suite": "prefix-tree", "status": "skipped", "reason": str(exc)}

    # Each arm has one scorer and its own allocator. Warm model weights before measuring
    # the first pass through an empty prefix tree and the repeat of the exact same inputs.
    public_python = args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON") or sys.executable
    schedule = []
    for i in range(max(1, args.abba_rounds)):
        order = ("plain", "cached", "stock") if i % 2 == 0 else ("stock", "plain", "cached")
        schedule.extend((*order, *reversed(order)))
    runs = []
    for arm in schedule:
        request = {"arm": arm, "records": records, "kev_dir": str(kev_dir), "use_wired_limit": not args.no_wired_limit}
        try:
            proc = run_subprocess_with_killpg(
                [public_python if arm == "stock" else sys.executable, str(_REPO_ROOT / "scripts" / "_kev_benchmark.py")],
                input_text=json.dumps(request),
                timeout_sec=180,
            )
        except subprocess.TimeoutExpired:
            return {"suite": "prefix-tree", "status": "failed", "reason": "timeout (180s)", "arm": arm}
        if proc.returncode:
            err_txt = proc.stderr[-1200:]
            low_err = err_txt.lower()
            is_opt_skip = any(tok in low_err for tok in ("no module named 'mlx_lm'", "no module named 'mlx_vlm'", "filenotfounderror"))
            return {"suite": "prefix-tree", "status": "skipped" if is_opt_skip else "failed", "reason": err_txt, "arm": arm}
        run = json.loads(proc.stdout.splitlines()[-1])
        if run.get("status") == "skipped":
            return {"suite": "prefix-tree", "status": "skipped", "reason": run["reason"], "arm": arm}
        if runs and runs[0]["token_layouts"] != run["token_layouts"]:
            raise ValueError(f"Kev token IDs or readout positions differed in {arm}")
        runs.append(run)

    summary = {}
    for arm in ("plain", "cached", "stock"):
        selected = [r for r in runs if r["arm"] == arm]
        summary[arm] = {
            batch: {phase: statistics.median(r["workloads"][batch][phase]["seconds"] for r in selected)
                    for phase in ("cold", "warm")}
            for batch in ("b1", "b8")
        }
    cached, plain, stock = (summary[a] for a in ("cached", "plain", "stock"))
    repeated = [
        score_differences(work["cold"]["scores"], work["warm"]["scores"])
        for run in runs if run["arm"] == "cached"
        for work in run["workloads"].values()
    ]
    max_diff = max(check["max_abs_probability"] for check in repeated)
    max_logit_diff = max(check["max_abs_logit"] for check in repeated)
    return {
        "suite": "prefix-tree", "status": "ok" if max_diff == max_logit_diff == 0.0 else "failed",
        "record_ids": [r["id"] for r in records], "dataset": "wandb/RAGTruth-processed",
        "fast_metric": f"warm B=1 {1/cached['b1']['warm']:.2f} q/s | B=8 {8/cached['b8']['warm']:.2f} q/s",
        "stock_baseline": f"stock mlx-lm, same eight questions {8/stock['b8']['warm']:.2f} q/s",
        "speedup_vs_stock": stock["b8"]["warm"] / cached["b8"]["warm"],
        "speedup_vs_plain": plain["b8"]["warm"] / cached["b8"]["warm"],
        "peak_gb": max(r["memory_gb"]["process_peak_footprint"] for r in runs if r["arm"] == "cached"),
        "cold_vs_warm_max_prob_diff": max_diff, "cold_vs_warm_max_logit_diff": max_logit_diff,
        "median_seconds": summary, "runs": runs,
    }


# ---------------------------------------------------------------------------
# 6. Parallel Multi-Bucket CoreML ANE Compilation (`jobs=4` vs `jobs=1`)
# ---------------------------------------------------------------------------


def _predict_bge_buckets(bge_home: Path, compiled_home: Path) -> dict[str, np.ndarray]:
    from fast_bge_small.engine import FastBgeSmall, _ANE_BUCKETS

    assets = compiled_home / "artifacts"
    for name in ("weights", "tokenizer"):
        (assets / name).symlink_to(bge_home / "artifacts" / name, target_is_directory=True)
    (compiled_home / "build").symlink_to(bge_home / "build", target_is_directory=True)
    outputs = {}
    with FastBgeSmall(home=compiled_home, default_mode="ane", warmup=False, verify_ane_plan=False) as engine:
        try:
            for length, filename, batch_size, _ in _ANE_BUCKETS:
                if length not in engine.buckets or (batch_size == 1 and length not in engine.query_buckets):
                    raise RuntimeError(f"Could not load {filename} from {assets / 'ane_models'}")
                # Give each row a different input and force dispatch to this exact compiled shape.
                sequences = [
                    engine.tokenizer.encode(
                        f"Record {row + 1}. " + "The team reviewed the annual report. " * 80,
                        max_length=length,
                    )
                    for row in range(batch_size)
                ]
                if any(len(ids) != length for ids in sequences):
                    raise ValueError(f"Input did not reach the {length}-token bucket")
                if batch_size == 1:
                    vectors = engine.encode_query(sequences[0], prefer_ane=True)[None, :]
                else:
                    vectors = engine.encode_corpus(sequences, mode="ane")
                if vectors.shape != (batch_size, engine.dim) or not np.isfinite(vectors).all():
                    raise ValueError(f"Invalid embeddings for {filename}: shape {vectors.shape}, expected finite {(batch_size, engine.dim)}")
                norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
                if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
                    raise ValueError(f"Expected unit-length embeddings for {filename}")
                outputs[filename] = vectors
        finally:
            # The native bucket registry is process-wide. Release these builds before loading the other arm.
            if engine.ane is not None:
                engine.ane.bge_ane_free_all()
    return outputs


def _compare_bge_builds(bge_home: Path, builds: Mapping[int, Path]) -> dict[str, Any]:
    serial = _predict_bge_buckets(bge_home, builds[1])
    parallel = _predict_bge_buckets(bge_home, builds[4])
    buckets = []
    for name, left in serial.items():
        right = parallel[name]
        a, b = left.astype(np.float64), right.astype(np.float64)
        cosines = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        buckets.append({
            "bundle": name,
            "vectors": len(left),
            "min_cosine": float(np.clip(cosines.min(), -1.0, 1.0)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "bit_identical": left.tobytes() == right.tobytes(),
        })
    min_cosine = min(row["min_cosine"] for row in buckets)
    return {
        "status": "ok" if min_cosine >= 0.9998 else "failed",
        "required_min_cosine": 0.9998,
        "min_cosine": min_cosine,
        "all_bit_identical": all(row["bit_identical"] for row in buckets),
        "buckets": buckets,
    }


def _run_parallel_coreml(args: argparse.Namespace) -> dict[str, Any]:
    bge_home = resolve_pkg_home("fast-bge-small", "FAST_BGE_SMALL_HOME")
    npz_p = bge_home / "artifacts" / "weights" / "bge_small_f16.npz"
    build_script = _REPO_ROOT / "packages" / "fast-bge-small" / "scripts" / "build_ane_models.py"
    if not npz_p.exists():
        return {"suite": "parallel-coreml", "status": "skipped", "reason": f"missing {npz_p}"}

    with tempfile.TemporaryDirectory(prefix="parallel_coreml_bench_") as td:
        tmp_r = Path(td)
        seq_times: list[float] = []
        par_times: list[float] = []
        builds = {}
        for r in range(max(1, int(args.abba_rounds))):
            order = ((1, seq_times), (4, par_times)) if r % 2 == 0 else ((4, par_times), (1, seq_times))
            for jobs, sink in order:
                builds[jobs] = tmp_r / f"r{r}_j{jobs}"
                out_d = builds[jobs] / "artifacts" / "ane_models"
                t0 = time.perf_counter()
                subprocess.check_call(
                    [sys.executable, str(build_script), "--home", str(bge_home), "--out-dir", str(out_d), "--jobs", str(jobs)],
                    stdout=subprocess.DEVNULL,
                )
                sink.append(time.perf_counter() - t0)
        try:
            embeddings = _compare_bge_builds(bge_home, builds)
        except Exception as exc:
            embeddings = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    med_1 = round(float(statistics.median(seq_times)), 2)
    med_4 = round(float(statistics.median(par_times)), 2)
    fp = get_unified_memory_footprint()
    return {
        "suite": "parallel-coreml",
        "status": embeddings["status"],
        "fast_metric": f"9 BGE-small buckets in {med_4:.2f}s (jobs=4 spawn pool)",
        "stock_baseline": f"Sequential coremltools (jobs=1): {med_1:.2f}s",
        "speedup_vs_stock": round(med_1 / max(med_4, 1e-6), 2),
        "peak_gb": round(float(fp["peak_footprint_gb"]), 3),
        "jobs_1_median_sec": med_1,
        "jobs_4_median_sec": med_4,
        "embedding_comparison": embeddings,
    }


def _get_host_ram_gb() -> float:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / (1024**3)
    except Exception:
        return 24.0


# ---------------------------------------------------------------------------
# Subprocess Isolation, Formatting & CLI
# ---------------------------------------------------------------------------


def _run_worker_main(args: argparse.Namespace) -> int:
    host_mem = _get_host_ram_gb()
    suite = args.worker_suite
    ctx = wired_limit() if not args.no_wired_limit else _null_ctx()
    with ctx:
        if suite == "qlora-packing":
            rec = _run_qlora_packing(args, host_mem)
        elif suite == "encoder-finetune":
            rec = _run_encoder_finetune(args)
        elif suite == "grpo-ane-critic":
            rec = _run_grpo_ane_critic(args)
        elif suite == "mtp-speculative":
            rec = _run_mtp_speculative(args)
        elif suite == "prefix-tree":
            rec = _run_prefix_tree(args)
        elif suite == "parallel-coreml":
            rec = _run_parallel_coreml(args)
        else:
            raise ValueError(f"Unknown worker suite: {suite}")
    write_worker_json(rec, worker_out_file=getattr(args, "worker_out_file", None), marker=_WORKER_MARKER)
    return EXIT_SKIP if rec.get("status") == "skipped" else 0


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield None


def format_capabilities_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Capability Suite | `fast-*` Throughput / Result | Stock Baseline | Speedup | Peak Mem (GB) | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in results:
        sp = r.get("speedup_vs_stock")
        sp_txt = f"`{float(sp):.2f}x`" if isinstance(sp, (int, float)) else "-"
        pk = r.get("peak_gb")
        pk_txt = f"{float(pk):.3f}" if isinstance(pk, (int, float)) else "-"
        lines.append(
            f"| `{r.get('suite', '-')}` | {r.get('fast_metric', '-')} | {r.get('stock_baseline', '-')} | "
            f"{sp_txt} | {pk_txt} | `{r.get('status', 'unknown')}` |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark-capabilities",
        description="Run isolated ABBA capability benchmarks (QLoRA, ModernBERT, GRPO+ANE, MTP, PrefixTree, Parallel CoreML).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    wk = sub.add_parser("_worker", help=argparse.SUPPRESS)
    wk.add_argument("--worker-suite", required=True)
    wk.add_argument("--public-python", default=None)
    wk.add_argument("--lora-python", default=None)
    wk.add_argument("--abba-rounds", type=int, default=2)
    wk.add_argument("--no-wired-limit", action="store_true")
    wk.add_argument("--worker-out-file", default=None)

    p.add_argument("--suites", default="all", help=f"Comma-separated list or 'all' ({', '.join(_SUITE_BY_NAME)})")
    p.add_argument("--python", default=sys.executable, help="Primary fast-silicon Python executable")
    p.add_argument("--public-python", default=None, help="Python executable with stock MLX runtimes (mlx-lm, mlx-vlm)")
    p.add_argument("--lora-python", default=None, help="Python executable with mlx-lm LoRA tuner (defaults to --public-python)")
    p.add_argument("--abba-rounds", type=int, default=2)
    p.add_argument("--cooldown-sec", type=float, default=1.0)
    p.add_argument("--timeout-sec", type=int, default=900)
    p.add_argument("--no-wired-limit", action="store_true")
    p.add_argument("--allow-contended", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json-out", default=None)
    p.add_argument("--markdown-out", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "_worker":
        return _run_worker_main(args)

    specs = select_suites(args.suites)
    env_meta = collect_env_telemetry()
    quiescence = check_system_quiescence()

    if args.dry_run:
        plan = {
            "dry_run": True,
            "suites": [s.name for s in specs],
            "python": args.python,
            "public_python": args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON"),
            "lora_python": args.lora_python or args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON"),
            "abba_rounds": int(args.abba_rounds),
            "environment": env_meta,
            "quiescence_warnings": quiescence,
        }
        print(json.dumps(plan, indent=2))
        return 0

    if quiescence and not args.allow_contended:
        print(f"QUIESCENCE WARNING ({len(quiescence)} active contenders):", file=sys.stderr)
        for line in quiescence:
            print(f"  {line}", file=sys.stderr)
        print("Aborting: re-run when quiet or pass --allow-contended.", file=sys.stderr)
        return 2

    pub_py = args.public_python or os.environ.get("FAST_SILICON_PUBLIC_PYTHON")
    lora_py = args.lora_python or pub_py or args.python
    results: list[dict[str, Any]] = []

    with _hw_lock():
        for idx, spec in enumerate(specs, 1):
            if idx > 1 and args.cooldown_sec > 0:
                time.sleep(float(args.cooldown_sec))
            print(f"[{idx}/{len(specs)}] Running capability suite {spec.name}...", file=sys.stderr, flush=True)
            worker_py = str(lora_py) if spec.needs_lora_python else str(args.python)
            cmd = [
                worker_py,
                str(Path(__file__).resolve()),
                "_worker",
                "--worker-suite",
                spec.name,
                "--abba-rounds",
                str(args.abba_rounds),
            ]
            if pub_py:
                cmd.extend(["--public-python", str(pub_py)])
            if lora_py:
                cmd.extend(["--lora-python", str(lora_py)])
            if args.no_wired_limit:
                cmd.append("--no-wired-limit")

            env = dict(os.environ)
            env["M4_GPU_LOCK_HELD"] = "1"
            wres = run_isolated_json_worker(
                cmd,
                worker_out_arg="--worker-out-file",
                marker=_WORKER_MARKER,
                timeout_sec=int(args.timeout_sec),
                cwd=_REPO_ROOT,
                env=env,
                skip_exit_codes=(EXIT_SKIP,),
                stderr_tail_lines=20,
            )
            if wres["timed_out"]:
                results.append(
                    {
                        "suite": spec.name,
                        "status": "failed",
                        "fast_metric": f"timeout ({args.timeout_sec}s)",
                        "stderr_tail": wres["stderr_tail"],
                    }
                )
                continue

            parsed = wres.get("payload")
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                tail = str(wres.get("stderr_tail") or wres.get("stdout") or f"exit={wres.get('returncode')}").strip().splitlines()
                results.append(
                    {
                        "suite": spec.name,
                        "status": "skipped" if wres["status"] == "skipped" else "failed",
                        "fast_metric": "-",
                        "stock_baseline": "-",
                        "error": tail[-1][:200] if tail else f"exit={wres.get('returncode')}",
                        "stderr_tail": wres["stderr_tail"],
                    }
                )

    md = format_capabilities_markdown(results)
    payload = {
        "schema_version": 1,
        "timestamp_epoch": int(time.time()),
        "environment": {
            "hw_model": env_meta.get("hw_model"),
            "chip": env_meta.get("brand"),
            "mem_gb": env_meta.get("mem_gb"),
            "macos_version": env_meta.get("macos_version"),
            "mlx_version": env_meta.get("mlx_version"),
            "python_version": platform.python_version(),
        },
        "results": results,
    }
    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        out_m = Path(args.markdown_out)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md + "\n", encoding="utf-8")

    print(md)
    return 1 if any(r.get("status") == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
