#!/usr/bin/env python3
"""Train a Gemma-4 E2B LoRA on ready-made token sequences, fuse it, 8-bit pack it, and compile its ANE bundle.

  1. BF16 LoRA SFT (rank 24, layers 12..34) on the `(token ids, prompt length)` pairs in `--pairs`. The loss
     covers the tokens after the prompt only. `B=1` with gradient accumulation is faster per example than `B=2`
     on a 48 GB M4 Pro (about 440 tok/s). On the first task this trained (3,077 documents, 3.8 h per epoch) one
     epoch was the measured optimum: a second pass raised held-out loss and lowered end-to-end scores.
  2. Exact `float32 -> BF16` LoRA merge, then `pack_checkpoint.py` (8-bit g64 non-MLP, BF16 MLP, 162,304-token
     Latin-Extended `lm_head` remap). Each `--checkpoint-config` file is copied into the checkpoint.
  3. Checkpoint-scoped Early14 Linear-W8 + AWQ-MSE CoreML prefill bundle via `build_ane_models.py`.

The caller owns the task: what the prompts say, what the answers are, which examples are held out. `--pairs` is a
JSON file `{"train": [[ids, prompt_len], ...], "valid": [[ids, prompt_len], ...], "stats": {...}}`, tokenized
with the base snapshot's tokenizer and chat template by your downstream task pipeline.

Run with a Python that has `mlx_lm` with Gemma-4 support (`~/.venv-vllm-metal/bin/python`).
"""

from __future__ import annotations

import argparse
import contextlib
from contextlib import contextmanager
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

DEFAULT_BF16_BASE_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/fb0b166bbb9a0eb4b37915bfc515a197c9122f39"
)
_VENV_PY = REPO_ROOT / ".venv/bin/python"
DEFAULT_FG4_PYTHON = _VENV_PY if _VENV_PY.exists() else Path(shutil.which("python3") or "python3")
HW_LOCK_PATH = Path("/tmp/m4_gpu_validation.lock")

@contextmanager
def acquire_hw_lock(label: str) -> Iterator[None]:
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


def build_length_bucketed_batches(
    items: list[tuple[list[int], int]],
    *,
    batch_size: int = 4,
    max_batch_tokens: int = 7600,
    pool_factor: int = 24,
    seed: int = 20261117,
) -> list[list[tuple[list[int], int]]]:
    """Group sequences of similar length into causal right-padded mini-batches (`B<=batch_size`, `B*L<=max_batch_tokens`)."""
    rng = random.Random(seed)
    perm = list(items)
    rng.shuffle(perm)
    pool_sz = max(batch_size * pool_factor, batch_size)
    batches: list[list[tuple[list[int], int]]] = []

    for p0 in range(0, len(perm), pool_sz):
        pool = sorted(perm[p0 : p0 + pool_sz], key=lambda x: len(x[0]))
        cur: list[tuple[list[int], int]] = []
        cur_max_l = 0
        for it in pool:
            l_in = len(it[0]) - 1
            next_max = max(cur_max_l, l_in)
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * next_max > max_batch_tokens):
                batches.append(cur)
                cur = [it]
                cur_max_l = l_in
            else:
                cur.append(it)
                cur_max_l = next_max
        if cur:
            batches.append(cur)

    rng.shuffle(batches)
    return batches


def _host_ram_gib() -> float:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0**3)
    except Exception:
        return 0.0


def compute_safe_memory_limit_gb(
    requested_gb: float | None,
    host_ram_gib: float,
    max_rec_ws_bytes: int | None = None,
    *,
    force: bool = False,
) -> float:
    """Compute a safe MLX memory limit in GiB so <=24 GiB hosts (Base M5) never wedge macOS wired memory."""
    if max_rec_ws_bytes is not None and int(max_rec_ws_bytes) > 0:
        rec_gb = (int(max_rec_ws_bytes) / float(1024**3)) * 0.92
    elif host_ram_gib > 0.0:
        rec_gb = host_ram_gib * 0.75
    else:
        rec_gb = 18.0
    if host_ram_gib > 0.0:
        rec_gb = min(rec_gb, max(4.0, host_ram_gib - 4.0))
    safe_cap = round(max(4.0, rec_gb), 2)
    if requested_gb is None:
        return round(min(38.0, safe_cap), 2)
    if force:
        return float(requested_gb)
    return round(min(float(requested_gb), safe_cap), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Gemma-4 E2B/E4B LoRA on token pairs, then fuse and pack BF16->8bit")
    ap.add_argument("--base-snap", type=Path, default=DEFAULT_BF16_BASE_DIR)
    ap.add_argument("--pairs", type=Path, default=None, help="JSON of tokenized train/valid pairs; required unless --fuse-only")
    ap.add_argument("--checkpoint-config", type=Path, action="append", default=[],
                    help="a JSON file to copy into the fused checkpoint under its own name (repeatable)")
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-model-dir", type=Path, required=True)
    ap.add_argument("--num-layers", type=int, default=23)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2, help="batches per optimizer step")
    ap.add_argument("--max-batch-tokens", type=int, default=3800)
    ap.add_argument(
        "--ckpt-tokens",
        type=int,
        default=None,
        help="activation-checkpoint batches above this size (default: 2200 for BF16, 4096 for --qlora-8bit)",
    )
    ap.add_argument(
        "--pack-tokens",
        type=int,
        default=0,
        help="fixed-bin varlen sequence packing capacity (e.g. 2048 or 4096); 0 disables packing",
    )
    ap.add_argument(
        "--loss-norm",
        choices=("token", "example"),
        default="token",
        help="answer loss weighting in packed bins (token-mean or example-mean)",
    )
    ap.add_argument(
        "--no-fixed-pad",
        action="store_true",
        help="disable trailing bin padding to --pack-tokens (allows variable bin shapes)",
    )
    ap.add_argument(
        "--qlora-8bit",
        action="store_true",
        help="quantize frozen trunk to 8-bit g64 with BF16 LoRA adapters and auto-raise --ckpt-tokens to 4096",
    )
    ap.add_argument(
        "--qlora-quantize-lm-head",
        action="store_true",
        help="also quantize tied embed_tokens/lm_head to 8-bit g64 under --qlora-8bit",
    )
    ap.add_argument(
        "--bf16-lora",
        action="store_true",
        help="cast LoRA (A, B) adapter weights to BF16 (automatically enabled under --qlora-8bit)",
    )
    ap.add_argument(
        "--shared-sys-kv",
        action="store_true",
        help="reuse differentiable system-prefix KV across grad_accum micro-batches",
    )
    ap.add_argument(
        "--min-shared-prefix",
        type=int,
        default=32,
        help="minimum common prefix tokens required to activate --shared-sys-kv",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="disable @mx.compile on chunked softcap+CE and AdamW updates (for A/B baselines)",
    )
    ap.add_argument("--val-loss-docs", type=int, default=96, help="held-out docs scored at each eval")
    ap.add_argument("--resume", action="store_true", help="continue from adapters_last + train_state.json")
    ap.add_argument("--init-adapter", type=Path, default=None, help="warm-start weights for a fresh schedule")
    ap.add_argument("--data-seed", type=int, default=20261117, help="batch-order seed; change it for a warm restart")
    ap.add_argument("--seed", type=int, default=20261117, help="LoRA init and dropout seed")
    ap.add_argument("--train-only", action="store_true", help="stop after training; skip fuse, pack and ANE")
    ap.add_argument(
        "--memory-limit-gb",
        type=float,
        default=None,
        help="hard MLX memory limit in GiB (auto-clamped to host RAM / recommended working set unless --force-memory-limit)",
    )
    ap.add_argument("--force-memory-limit", action="store_true", help="Bypass host RAM safety clamp on --memory-limit-gb")
    ap.add_argument("--vocab-out", type=int, default=162304)
    ap.add_argument("--lr", type=float, default=1.8e-4)
    ap.add_argument("--min-lr", type=float, default=1.8e-5)
    ap.add_argument("--warmup-steps", type=int, default=30, help="optimizer steps")
    ap.add_argument("--eval-every", type=int, default=400, help="batches")
    ap.add_argument("--smoke-steps", type=int, default=0, help="Run N steps only to measure step timing")
    ap.add_argument("--fuse-only", action="store_true", help="Skip training and fuse+pack existing adapters_best")
    ap.add_argument("--skip-ane", action="store_true", help="Skip Step 3 CoreML ANE compilation")
    ap.add_argument("--ane-variant", choices=("fp16", "w8", "both"), default="fp16", help="ANE bundle variant for Step 3")
    ap.add_argument("--calib-file", type=Path, default=None, help="Optional domain corpus for AWQ-MSE W8 calibration")
    ap.add_argument("--no-hw-lock", action="store_true", help="do not wait on the shared GPU lock (fuse/pack beside a paused run)")
    args = ap.parse_args()
    if args.pairs is None and not args.fuse_only:
        ap.error("--pairs is required unless --fuse-only is given")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import load_model
    from fast_gemma4.lora_accel import (
        CompiledOptimizerStep,
        build_packed_groups,
        build_shared_prefix_packed_groups,
        cast_lora_adapters_dtype,
        chunked_softcap_ce_loss,
        compute_effective_ckpt_tokens,
        compute_packed_tail_ce_loss,
        find_shared_prefix_len,
        pack_examples_to_bin,
        pack_suffixes_for_shared_prefix,
        quantize_frozen_trunk_8bit,
        run_frozen_prefix_packed,
        run_shared_prefix_grad_accum_step,
        unwrap_kv_shared_lora_modules,
    )

    use_compile = not args.no_compile
    eff_ckpt_tokens = compute_effective_ckpt_tokens(args.ckpt_tokens, qlora_8bit=args.qlora_8bit)

    # GPU memory is wired and cannot be paged out, so clamp cleanly to host RAM & Metal recommended working set.
    host_ram = _host_ram_gib()
    dev_info_fn = getattr(mx, "device_info", None)
    max_rec_ws = int(dev_info_fn().get("max_recommended_working_set_size", 0)) if callable(dev_info_fn) else 0
    eff_mem_limit_gb = compute_safe_memory_limit_gb(
        args.memory_limit_gb, host_ram, max_rec_ws or None, force=args.force_memory_limit
    )
    if args.memory_limit_gb is not None and eff_mem_limit_gb < float(args.memory_limit_gb):
        print(
            f"[memory-clamp] Clamped --memory-limit-gb from {args.memory_limit_gb:.2f} GiB -> "
            f"{eff_mem_limit_gb:.2f} GiB for {host_ram:.1f} GiB host.",
            flush=True,
        )
    mx.set_memory_limit(int(eff_mem_limit_gb * 1024**3))
    mx.set_cache_limit(512 * 1024 * 1024)
    prev_wired_limit: int | None = None
    set_wl = getattr(mx, "set_wired_limit", None)
    if callable(set_wl) and max_rec_ws > 0:
        try:
            prev_wired_limit = int(set_wl(max_rec_ws))
        except Exception:
            prev_wired_limit = None
    mx.random.seed(args.seed)
    random.seed(args.seed)

    def unwrap_kv_shared_lora(model: Any) -> int:
        return unwrap_kv_shared_lora_modules(model)

    def run_frozen_prefix(model: Any, inputs: Any, first_lora_layer: int) -> tuple[Any, list[Any], list[Any], list[Any]]:
        tm = model.language_model.model
        input_embeddings = tm.embed_tokens(inputs)
        h = input_embeddings * tm.embed_scale
        if tm.hidden_size_per_layer_input:
            pli = tm._get_per_layer_inputs(inputs, input_embeddings)
            pli = tm._project_per_layer_inputs(h, pli)
            pli_list = [pli[:, :, i, :] for i in range(len(tm.layers))]
        else:
            pli_list = [None] * len(tm.layers)
        caches = [None] * len(tm.layers)
        masks = tm._make_masks(h, caches)
        intermediates: list[Any] = [(None, None)] * len(tm.layers)
        for idx in range(first_lora_layer):
            kvs, offset = intermediates[tm.previous_kvs[idx]]
            h, kvs, offset = tm.layers[idx](
                h, masks[idx], None, per_layer_input=pli_list[idx], shared_kv=kvs, offset=offset
            )
            intermediates[idx] = (kvs, offset)
        return h, masks, intermediates, pli_list

    def make_padded_batch(batch: list[tuple[list[int], int]]) -> tuple[Any, list[tuple[int, int]], Any]:
        """Right-pad causal sequences (`cos=1.0` exact on valid prefix!) and pack flat answer targets."""
        max_in = max(len(f_ids) - 1 for f_ids, _ in batch)
        in_rows: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        y_flat: list[int] = []
        for f_ids, p_len in batch:
            inp = f_ids[:-1]
            pad_n = max_in - len(inp)
            in_rows.append(inp + [0] * pad_n)
            spans.append((p_len - 1, len(f_ids) - 1))
            y_flat.extend(f_ids[p_len:])
        return mx.array(in_rows, dtype=mx.int32), spans, mx.array(y_flat, dtype=mx.int32)

    def compute_batched_tail_ce_loss(
        model: Any,
        h: Any,
        masks: list[Any],
        prefix_intermediates: list[Any],
        pli_list: list[Any],
        spans: list[tuple[int, int]],
        y_flat: Any,
        first_lora_layer: int,
        use_ckpt: bool = False,
    ) -> Any:
        lm = model.language_model
        tm = lm.model
        intermediates = list(prefix_intermediates)
        for idx in range(first_lora_layer, len(tm.layers)):
            kvs, offset = intermediates[tm.previous_kvs[idx]]
            layer = tm.layers[idx]
            if use_ckpt:

                def _ckpt_fn(h_in: Any, m_in: Any, pli_in: Any, kvs_in: Any, off_in: Any, _l: Any = layer) -> Any:
                    return _l(h_in, m_in, None, per_layer_input=pli_in, shared_kv=kvs_in, offset=off_in)

                h, kvs, offset = nn.utils.checkpoint(layer, _ckpt_fn)(
                    h, masks[idx], pli_list[idx], kvs, offset
                )
            else:
                h, kvs, offset = layer(
                    h, masks[idx], None, per_layer_input=pli_list[idx], shared_kv=kvs, offset=offset
                )
            intermediates[idx] = (kvs, offset)

        h_norm = tm.norm(h)
        # Gather ONLY true answer positions across the batch -> zero 262K lm_head FLOPs on prompt or pad tokens!
        h_slices = [h_norm[b_i, s0:s1, :] for b_i, (s0, s1) in enumerate(spans)]
        h_ans_2d = mx.concatenate(h_slices, axis=0)
        lm_head_fn = tm.embed_tokens.as_linear if lm.tie_word_embeddings else lm.lm_head
        return chunked_softcap_ce_loss(
            lm_head_fn,
            h_ans_2d,
            y_flat,
            softcap=lm.final_logit_softcapping,
            chunk_size=256,
            compiled=use_compile,
        )

    def eval_dataset_loss(model: Any, dataset: list[tuple[list[int], int]], first_lora_layer: int) -> float:
        model.eval()
        tot_loss = 0.0
        tot_toks = 0
        if args.pack_tokens > 0:
            eval_bins = build_packed_groups(dataset, pack_tokens=args.pack_tokens, seed=1)
            for grp in eval_bins:
                pbin = pack_examples_to_bin(
                    grp,
                    pack_tokens=args.pack_tokens,
                    pad_to_fixed=not args.no_fixed_pad,
                    loss_norm="token",
                )
                n_ans = int(pbin.y_flat.shape[0])
                h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(model, pbin, first_lora_layer)
                loss = compute_packed_tail_ce_loss(
                    model,
                    h,
                    masks,
                    inter,
                    pli_list,
                    rope_tables,
                    pbin.ans_indices,
                    pbin.y_flat,
                    None,
                    first_lora_layer,
                    use_ckpt=False,
                    compiled_ce=use_compile,
                )
                mx.eval(loss)
                tot_loss += float(loss.item()) * n_ans
                tot_toks += n_ans
            return tot_loss / max(tot_toks, 1)

        eval_batches = build_length_bucketed_batches(
            dataset, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens, pool_factor=64, seed=1
        )
        for batch in eval_batches:
            inputs, spans, y_flat = make_padded_batch(batch)
            n_ans = int(y_flat.shape[0])
            h, masks, inter, pli_list = run_frozen_prefix(model, inputs, first_lora_layer)
            loss = compute_batched_tail_ce_loss(
                model, h, masks, inter, pli_list, spans, y_flat, first_lora_layer, use_ckpt=False
            )
            mx.eval(loss)
            tot_loss += float(loss.item()) * n_ans
            tot_toks += n_ans
        return tot_loss / max(tot_toks, 1)

    @contextlib.contextmanager
    def _restore_wired_limit() -> Iterator[None]:
        try:
            yield
        finally:
            if prev_wired_limit is not None and callable(set_wl):
                try:
                    set_wl(prev_wired_limit)
                except Exception:
                    pass

    with _restore_wired_limit(), (contextlib.nullcontext() if args.no_hw_lock else acquire_hw_lock("train_lora")):
        t_start = time.perf_counter()
        print(f"1. Loading native BF16 Gemma-4 base from {args.base_snap}...", flush=True)
        model, _ = load_model(args.base_snap, strict=False)
        for mm_attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(model, mm_attr):
                setattr(model, mm_attr, None)
        gc.collect()
        mx.clear_cache()

        q_stats: dict[str, int] | None = None
        if args.qlora_8bit and not args.fuse_only:
            q_stats = quantize_frozen_trunk_8bit(
                model,
                group_size=64,
                bits=8,
                quantize_ple_embedding=True,
                quantize_embed_tokens=args.qlora_quantize_lm_head,
            )
            mx.eval(model.parameters())
            gc.collect()
            mx.clear_cache()

        model.freeze()
        lora_cfg = {"rank": args.rank, "scale": args.scale, "dropout": args.dropout}
        linear_to_lora_layers(model, num_layers=args.num_layers, config=lora_cfg)
        n_bf16_cast = 0
        if args.qlora_8bit or args.bf16_lora:
            n_bf16_cast = cast_lora_adapters_dtype(model, mx.bfloat16)
        n_unwrapped = unwrap_kv_shared_lora(model)
        expected_lora_modules = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        first_lora_layer = len(model.layers) - args.num_layers
        mode_tag = "QLoRA-8bit-g64" if args.qlora_8bit else "BF16"
        q_desc = (
            f" | q8_linears={q_stats['linear_quantized']} q8_embs={q_stats['embedding_quantized']}"
            if q_stats is not None
            else ""
        )
        print(
            f"   Loaded {mode_tag} base in {time.perf_counter() - t_start:.2f}s | "
            f"active LoRA modules={expected_lora_modules} (unwrapped={n_unwrapped}, bf16_lora={n_bf16_cast}) | "
            f"layers=[{first_lora_layer}..{len(model.layers) - 1}] | ckpt_tokens={eff_ckpt_tokens} "
            f"pack_tokens={args.pack_tokens} shared_sys_kv={args.shared_sys_kv} compiled={use_compile}{q_desc}",
            flush=True,
        )
        print_trainable_parameters(model)

        adapter_dir = args.adapter_dir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        best_adapter_path = adapter_dir / "adapters_best.safetensors"

        if not args.fuse_only:
            pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
            train_data = [(list(ids), int(p_len)) for ids, p_len in pairs["train"]]
            val_data = [(list(ids), int(p_len)) for ids, p_len in pairs["valid"]]
            d_stats = pairs.get("stats", {})
            del pairs
            accum = max(1, args.grad_accum)
            if args.shared_sys_kv:
                epoch_batches = [
                    build_shared_prefix_packed_groups(
                        train_data,
                        pack_tokens=args.pack_tokens,
                        min_shared_prefix=args.min_shared_prefix,
                        grad_accum=accum,
                        seed=args.data_seed + ep,
                    )
                    for ep in range(1, args.epochs + 1)
                ]
            elif args.pack_tokens > 0:
                epoch_batches = [
                    build_packed_groups(
                        train_data,
                        pack_tokens=args.pack_tokens,
                        seed=args.data_seed + ep,
                    )
                    for ep in range(1, args.epochs + 1)
                ]
            else:
                epoch_batches = [
                    build_length_bucketed_batches(
                        train_data,
                        batch_size=args.batch_size,
                        max_batch_tokens=args.max_batch_tokens,
                        seed=args.data_seed + ep,
                    )
                    for ep in range(1, args.epochs + 1)
                ]
            total_steps = sum(len(b) for b in epoch_batches)
            total_opt_steps = math.ceil(total_steps / accum)
            print(f"2. Dataset prepared: {json.dumps(d_stats)} | total_batches={total_steps}", flush=True)

            last_adapter_path = adapter_dir / "adapters_last.safetensors"
            state_path = adapter_dir / "train_state.json"
            val_loss_set = val_data[:: max(1, math.ceil(len(val_data) / max(1, args.val_loss_docs)))]
            if args.smoke_steps > 0:
                val_loss_set = val_loss_set[:16]

            def save_adapter(path: Path) -> None:
                mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))

            state: dict[str, Any] = {"batches_done": 0, "best_val": None, "best_step": 0, "val_history": []}
            if args.resume and state_path.is_file() and last_adapter_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                model.load_weights(list(mx.load(str(last_adapter_path)).items()), strict=False)
                print(f"   Resumed at batch {state['batches_done']}/{total_steps}", flush=True)
            elif args.init_adapter is not None:
                model.load_weights(list(mx.load(str(args.init_adapter)).items()), strict=False)
                print(f"   Warm-started from {args.init_adapter}", flush=True)

            if state["best_val"] is None:
                t_ev0 = time.perf_counter()
                v0 = eval_dataset_loss(model, val_loss_set, first_lora_layer)
                print(
                    f"   [Start] valid ({len(val_loss_set)} docs) loss = {v0:.4f} "
                    f"(ppl = {math.exp(v0):.3f}, {time.perf_counter() - t_ev0:.1f}s)",
                    flush=True,
                )
                state["best_val"] = v0
                state["val_history"].append({"step": 0, "val_loss": round(v0, 5)})
            v0 = float(state["val_history"][0]["val_loss"])
            best_val = float(state["best_val"])
            best_step = int(state["best_step"])
            val_history: list[dict[str, Any]] = list(state["val_history"])

            def get_lr(opt_i: int) -> float:
                if opt_i < args.warmup_steps:
                    return args.lr * (opt_i + 1) / max(1, args.warmup_steps)
                prog = (opt_i - args.warmup_steps) / max(1, total_opt_steps - args.warmup_steps)
                return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))

            optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
            opt_stepper = CompiledOptimizerStep(
                optimizer, model, max_grad_norm=1.0, compiled=use_compile
            )
            loss_and_grad_fn = nn.value_and_grad(model, compute_batched_tail_ce_loss)
            packed_loss_and_grad_fn = nn.value_and_grad(model, compute_packed_tail_ce_loss)

            step = 0
            skip_batches = int(state["batches_done"])
            t_train0 = time.perf_counter()
            running_loss = 0.0
            running_cnt = 0
            toks_done = 0
            acc_grads: Any = None
            acc_n = 0
            cur_lr = get_lr(skip_batches // accum)

            def _maybe_log_eval_or_stop(
                ep_idx_: int,
                prev_step_: int,
                cur_step_: int,
                shape_desc_: str,
            ) -> bool:
                nonlocal running_loss, running_cnt, best_val, best_step
                should_log = (
                    (prev_step_ // 50) < (cur_step_ // 50)
                    or prev_step_ == skip_batches
                    or (args.smoke_steps > 0 and cur_step_ >= args.smoke_steps)
                )
                if should_log:
                    elapsed = time.perf_counter() - t_train0
                    tok_s = toks_done / max(elapsed, 1e-6)
                    s_per_step = elapsed / max(1, cur_step_ - skip_batches)
                    eta_m = s_per_step * max(0, total_steps - cur_step_) / 60.0
                    peak_gb = mx.get_peak_memory() / (1024**3)
                    avg_l = running_loss / max(running_cnt, 1)
                    running_loss = 0.0
                    running_cnt = 0
                    print(
                        f"   [Ep {ep_idx_}/{args.epochs} | Batch {cur_step_}/{total_steps}] "
                        f"train_loss={avg_l:.4f} lr={cur_lr:.2e} {shape_desc_} "
                        f"({s_per_step:.2f}s/batch, {tok_s:.0f} tok/s, ETA={eta_m:.1f}m, peak_mem={peak_gb:.2f}GB)",
                        flush=True,
                    )

                if args.smoke_steps and cur_step_ >= args.smoke_steps:
                    v_smoke = eval_dataset_loss(model, val_loss_set, first_lora_layer)
                    print(
                        f"   [Smoke] post-{cur_step_}-batches val_loss ({len(val_loss_set)} docs) = {v_smoke:.4f}",
                        flush=True,
                    )
                    return True

                if (prev_step_ // args.eval_every) < (cur_step_ // args.eval_every) or cur_step_ == total_steps:
                    t_ev = time.perf_counter()
                    v_cur = eval_dataset_loss(model, val_loss_set, first_lora_layer)
                    val_history.append({"step": cur_step_, "epoch": ep_idx_, "val_loss": round(v_cur, 5)})
                    marker = ""
                    if v_cur < best_val:
                        best_val = v_cur
                        best_step = cur_step_
                        save_adapter(best_adapter_path)
                        marker = " *BEST*"
                    # Only a batch count on an optimizer-step boundary is resumable without dropping gradients.
                    if acc_n == 0:
                        save_adapter(last_adapter_path)
                        state_path.write_text(
                            json.dumps(
                                {
                                    "batches_done": cur_step_,
                                    "best_val": best_val,
                                    "best_step": best_step,
                                    "val_history": val_history,
                                },
                                indent=1,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    print(
                        f"   >>> [Eval @ batch {cur_step_}] val_loss={v_cur:.4f} (ppl={math.exp(v_cur):.3f}, {time.perf_counter()-t_ev:.1f}s){marker}",
                        flush=True,
                    )
                    gc.collect()
                    mx.clear_cache()
                    model.train()
                return False

            def _run_single_batch(batch: list[tuple[list[int], int]]) -> tuple[float, int, str]:
                nonlocal acc_grads, acc_n, cur_lr
                if args.pack_tokens > 0:
                    pbin = pack_examples_to_bin(
                        batch,
                        pack_tokens=args.pack_tokens,
                        pad_to_fixed=not args.no_fixed_pad,
                        pad_multiple=64 if args.no_fixed_pad else 1,
                        loss_norm=args.loss_norm,
                    )
                    use_ckpt = pbin.bin_tokens > eff_ckpt_tokens
                    h, masks, inter, pli_list, rope_tables = run_frozen_prefix_packed(
                        model, pbin, first_lora_layer
                    )
                    loss, grads = packed_loss_and_grad_fn(
                        model,
                        h,
                        masks,
                        inter,
                        pli_list,
                        rope_tables,
                        pbin.ans_indices,
                        pbin.y_flat,
                        pbin.token_weights,
                        first_lora_layer,
                        use_ckpt=use_ckpt,
                        compiled_ce=use_compile,
                    )
                    b_toks = pbin.real_tokens
                    shape_desc = f"P={pbin.num_examples}docs/{pbin.real_tokens}->{pbin.bin_tokens}"
                else:
                    inputs, spans, y_flat = make_padded_batch(batch)
                    b_toks = int(inputs.shape[0] * inputs.shape[1])
                    use_ckpt = b_toks > eff_ckpt_tokens
                    h, masks, inter, pli_list = run_frozen_prefix(model, inputs, first_lora_layer)
                    loss, grads = loss_and_grad_fn(
                        model, h, masks, inter, pli_list, spans, y_flat, first_lora_layer, use_ckpt
                    )
                    shape_desc = f"B={inputs.shape[0]}x{inputs.shape[1]}"

                acc_grads = opt_stepper.accumulate(acc_grads, grads)
                acc_n += 1
                mx.eval(loss, acc_grads)
                if acc_n >= accum or step == total_steps:
                    cur_lr = get_lr((step - 1) // accum)
                    opt_stepper.step(model, acc_grads, lr=cur_lr, accum_divisor=acc_n)
                    mx.eval(model.parameters(), optimizer.state)
                    acc_grads = None
                    acc_n = 0
                return float(loss.item()), b_toks, shape_desc

            for ep_idx, batches in enumerate(epoch_batches, start=1):
                model.train()
                b_cursor = 0
                while b_cursor < len(batches):
                    if args.shared_sys_kv and acc_n == 0 and step >= skip_batches:
                        win_len = min(accum, len(batches) - b_cursor)
                        if args.smoke_steps > 0:
                            win_len = min(win_len, max(1, args.smoke_steps - step))
                        window = batches[b_cursor : b_cursor + win_len]
                        all_ex = [ex for b in window for ex in b]
                        sys_len = find_shared_prefix_len(all_ex, min_len=args.min_shared_prefix)
                        if sys_len >= args.min_shared_prefix:
                            prev_step = step
                            step += win_len
                            b_cursor += win_len
                            suf_cap = args.pack_tokens if args.pack_tokens > 0 else 0
                            suffix_bins = [
                                pack_suffixes_for_shared_prefix(
                                    b,
                                    sys_len=sys_len,
                                    pack_tokens=suf_cap,
                                    pad_to_fixed=(args.pack_tokens > 0 and not args.no_fixed_pad),
                                    pad_multiple=64 if (args.pack_tokens > 0 and args.no_fixed_pad) else 1,
                                    loss_norm=args.loss_norm,
                                )
                                for b in window
                            ]
                            avg_l, total_grads, per_bin_losses = run_shared_prefix_grad_accum_step(
                                model,
                                all_ex[0][0][:sys_len],
                                suffix_bins,
                                first_lora_layer,
                                ckpt_tokens=eff_ckpt_tokens,
                                compiled_ce=use_compile,
                                sys_dropout_seed=(args.seed + step) if args.dropout > 0.0 else None,
                            )
                            cur_lr = get_lr((step - 1) // accum)
                            opt_stepper.step(model, total_grads, lr=cur_lr, accum_divisor=1)
                            mx.eval(model.parameters(), optimizer.state)
                            running_loss += sum(per_bin_losses)
                            running_cnt += len(per_bin_losses)
                            toks_done += sum(sb.logical_total_tokens for sb in suffix_bins)
                            shape_desc = (
                                f"SYS={sys_len}+K={win_len}x"
                                f"{sum(sb.real_suffix_tokens for sb in suffix_bins) // max(1, win_len)}"
                            )
                            if _maybe_log_eval_or_stop(ep_idx, prev_step, step, shape_desc):
                                return 0
                            continue

                    batch = batches[b_cursor]
                    b_cursor += 1
                    prev_step = step
                    step += 1
                    if step <= skip_batches:
                        continue
                    l_val, b_toks, shape_desc = _run_single_batch(batch)
                    running_loss += l_val
                    running_cnt += 1
                    toks_done += b_toks
                    if _maybe_log_eval_or_stop(ep_idx, prev_step, step, shape_desc):
                        return 0

            adapter_meta = {
                "fine_tune_type": "qlora_8bit" if args.qlora_8bit else "lora",
                "version": "v6_4_c2_c4_bf16",
                "base_snap": str(args.base_snap),
                "num_layers": args.num_layers,
                "expected_lora_modules": expected_lora_modules,
                "lora_parameters": lora_cfg,
                "seed": args.seed,
                "train_minutes": round((time.perf_counter() - t_train0) / 60.0, 1),
                "peak_mem_gb": round(mx.get_peak_memory() / (1024**3), 2),
                "dataset_stats": d_stats,
                "pre_lora_val_loss": v0,
                "best_val_loss": best_val,
                "best_step": best_step,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": accum,
                "lr": args.lr,
                "qlora_8bit": bool(args.qlora_8bit),
                "pack_tokens": int(args.pack_tokens),
                "loss_norm": str(args.loss_norm),
                "shared_sys_kv": bool(args.shared_sys_kv),
                "compiled_ce_adamw": bool(use_compile),
                "effective_ckpt_tokens": int(eff_ckpt_tokens),
                "val_history": val_history,
            }
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(adapter_meta, indent=2) + "\n", encoding="utf-8"
            )
            if args.train_only:
                print(f"TRAIN ONLY COMPLETE: best_val={best_val:.4f} @ batch {best_step}", flush=True)
                return 0

        # ---------------------------------------------------------------------
        # Step 2: Exact float32 -> BF16 LoRA Fuse + Zero-QAT 8-Bit Packager
        # ---------------------------------------------------------------------
        print(f"3. Reloading best adapter {best_adapter_path} and merging in float32 -> BF16...", flush=True)
        best_weights = dict(mx.load(str(best_adapter_path)))
        model.load_weights(list(best_weights.items()), strict=False)
        mx.eval(model.parameters())

        lora_by_key: dict[str, Any] = {}
        for mod_path, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                lora_by_key[f"{mod_path}.weight"] = mod
                lora_by_key[f"language_model.{mod_path}.weight"] = mod

        tmp_bf16_dir = args.out_model_dir.parent / f".tmp_fused_bf16_{os.getpid()}"
        if tmp_bf16_dir.exists():
            shutil.rmtree(tmp_bf16_dir)
        tmp_bf16_dir.mkdir(parents=True, exist_ok=True)

        try:
            fused_count = 0
            idx_json_p = args.base_snap / "model.safetensors.index.json"
            shard_files = sorted(args.base_snap.glob("model*.safetensors"))
            new_weight_map: dict[str, str] = {}

            for sf in shard_files:
                shard_tensors = dict(mx.load(str(sf)))
                out_shard: dict[str, Any] = {}
                for k, t_val in shard_tensors.items():
                    if any(
                        k.startswith(pfx) or f".{pfx}" in k
                        for pfx in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector")
                    ):
                        continue
                    if k in lora_by_key:
                        mod = lora_by_key[k]
                        delta_f32 = float(mod.scale) * (
                            mod.lora_b.astype(mx.float32).T @ mod.lora_a.astype(mx.float32).T
                        )
                        merged_bf16 = (t_val.astype(mx.float32) + delta_f32).astype(mx.bfloat16)
                        mx.eval(merged_bf16)
                        out_shard[k] = merged_bf16
                        fused_count += 1
                    else:
                        out_shard[k] = t_val
                    new_weight_map[k] = sf.name
                if out_shard:
                    mx.save_safetensors(str(tmp_bf16_dir / sf.name), out_shard)
                del shard_tensors, out_shard
                gc.collect()
                mx.clear_cache()

            if fused_count != expected_lora_modules:
                raise RuntimeError(f"Expected {expected_lora_modules} fused modules, got {fused_count}")

            if idx_json_p.is_file():
                orig_idx = json.loads(idx_json_p.read_text(encoding="utf-8"))
                orig_idx["weight_map"] = new_weight_map
                (tmp_bf16_dir / "model.safetensors.index.json").write_text(
                    json.dumps(orig_idx, indent=2) + "\n", encoding="utf-8"
                )

            for extra_name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
                src_p = args.base_snap / extra_name
                if src_p.is_file():
                    shutil.copy2(src_p, tmp_bf16_dir / extra_name)

            for cfg_p in args.checkpoint_config:
                shutil.copy2(cfg_p, tmp_bf16_dir / cfg_p.name)
            print(f"   Merged {fused_count}/{expected_lora_modules} modules into BF16 staging dir.", flush=True)

            del model, best_weights, lora_by_key
            gc.collect()
            mx.clear_cache()

            print(f"4. Running Zero-QAT 8-bit non-MLP + BF16 MLP packager -> {args.out_model_dir}...", flush=True)
            env_sub = dict(os.environ)
            env_sub["PYTHONPATH"] = str(REPO_ROOT / "src")
            pack_cmd = [
                str(DEFAULT_FG4_PYTHON),
                str(REPO_ROOT / "scripts" / "pack_checkpoint.py"),
                str(tmp_bf16_dir),
                str(args.out_model_dir),
                "--vocab-out",
                str(args.vocab_out),
            ]
            for cfg_p in args.checkpoint_config:
                pack_cmd.extend(["--extra-file", cfg_p.name])
            subprocess.check_call(pack_cmd, env=env_sub)
        finally:
            if tmp_bf16_dir.exists():
                shutil.rmtree(tmp_bf16_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Step 3: Compile Checkpoint-Scoped Early14/Early23 ANE Bundle
    # -------------------------------------------------------------------------
    if args.skip_ane:
        stale_local_ane = args.out_model_dir / "ane_models"
        if stale_local_ane.exists():
            shutil.rmtree(stale_local_ane, ignore_errors=True)
            print(f"5. Removed stale local ANE directory {stale_local_ane} (--skip-ane).", flush=True)
    else:
        print(f"5. Compiling checkpoint-scoped {args.ane_variant} EarlyExit ANE prefill bundle...", flush=True)
        env_ane = dict(os.environ)
        env_ane["PYTHONPATH"] = str(REPO_ROOT / "src")
        ane_cmd = [
            str(DEFAULT_FG4_PYTHON),
            str(REPO_ROOT / "scripts" / "build_ane_models.py"),
            "--home",
            str(REPO_ROOT),
            "--model-dir",
            str(args.out_model_dir),
            "--variant",
            args.ane_variant,
            "--reexport",
        ]
        if args.ane_variant in ("w8", "both"):
            ane_cmd.append("--awq-mse")
            if args.calib_file is not None:
                ane_cmd.extend(["--calib-file", str(args.calib_file)])
        subprocess.check_call(ane_cmd, env=env_ane)

    print(f"ALL 3 STEPS COMPLETE in {(time.perf_counter() - t_start) / 60.0:.2f} min -> {args.out_model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
