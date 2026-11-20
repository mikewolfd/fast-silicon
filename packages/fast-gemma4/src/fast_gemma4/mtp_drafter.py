"""Opt-in Gemma-4 E2B MTP Drafter (`google/gemma-4-E2B-it-assistant`) + GPU `DraftScan`, strictly gated to `B=1..4`."""

from __future__ import annotations

import collections
from pathlib import Path
import time
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from .grammar import GrammarHelper
from .kernels import (
    fused_geglu_split,
    fused_norm_add_norm,
    fused_q_norm_rope_ragged,
    mqa8_sdpa_prefix_ragged,
    mqa8_sdpa_prefix_ragged_slot,
)
from .model import FastGemma4Model, safe_argmax_with_nan_guard
from .speculative import _pick_valid_token_safe

DEFAULT_MTP_DRAFTER_REPO: str = "google/gemma-4-E2B-it-assistant"
DEFAULT_MTP_TOP_K: int = 32
MAX_MTP_BATCH_SIZE: int = 4


def should_use_mtp_drafter(
    drafter: Any | None,
    use_ane: bool,
    batch_w: int,
    temperature: float,
    *,
    speculative: bool | None = None,
) -> bool:
    """Enable MTP drafter strictly for `1 <= B <= 4`, `temperature == 0.0`, non-ANE GPU waves."""
    if drafter is None or speculative is False or use_ane or temperature > 0.0:
        return False
    return 1 <= int(batch_w) <= MAX_MTP_BATCH_SIZE


def recommended_mtp_draft_k(batch_size: int, *, has_grammar: bool = False) -> int:
    """Select roof-line optimal draft depth `K` for `1 <= B <= 4` on Apple Silicon.

    - `B == 1`: `K = 3` (`L_q = 4`, optimal `2.63x` JSON / `2.13x` free-text; `K = 4` also strong on free-text).
    - `2 <= B <= 4`: `K = 2` keeps `M = B * (K + 1) <= 12 < 16`, staying below the `16 <= M <= 64`
      MPP tile threshold where verify latency jumps `~2.4x`.
    """
    b = int(batch_size)
    if not (1 <= b <= MAX_MTP_BATCH_SIZE):
        raise ValueError(
            f"MTP speculative drafting is strictly gated to 1 <= B <= {MAX_MTP_BATCH_SIZE} "
            f"(never enable for B >= 8); got B={b}"
        )
    if b == 1:
        return 3 if has_grammar else 4
    return 2


def resolve_mtp_drafter_dir(
    drafter_repo_or_dir: str | Path = DEFAULT_MTP_DRAFTER_REPO,
    *,
    local_files_only: bool = True,
) -> Path:
    """Resolve local checkpoint directory or HuggingFace cache snapshot for the Gemma-4 E2B MTP assistant."""
    p = Path(drafter_repo_or_dir).expanduser()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(drafter_repo_or_dir),
            allow_patterns=["*.safetensors", "config.json"],
            local_files_only=local_files_only,
        )
    )


def draft_scan_top_k(
    x_norm_2d: mx.array,
    centroids_w: mx.array,
    embed_perm: mx.array,
    order_2d: mx.array,
    *,
    top_k: int = DEFAULT_MTP_TOP_K,
) -> mx.array:
    """Score `2048` centroids, gather `top_k` candidate clusters (`32 x 128 = 4,096` rows), and return argmax token IDs."""
    B = int(x_norm_2d.shape[0])
    num_centroids = int(centroids_w.shape[0])
    per_centroid = int(embed_perm.shape[1])
    eff_k = max(1, min(int(top_k), num_centroids))

    c_logits = x_norm_2d @ centroids_w.T  # (B, num_centroids)
    if eff_k < num_centroids:
        top_c = mx.argpartition(-c_logits, kth=eff_k, axis=-1)[:, :eff_k]
    else:
        top_c = mx.broadcast_to(mx.arange(num_centroids, dtype=mx.int32)[None, :], (B, num_centroids))
    cand_w = embed_perm[top_c]  # (B, eff_k, per_centroid, 256)
    scores = mx.matmul(cand_w, x_norm_2d[:, None, :, None]).squeeze(-1)  # (B, eff_k, per_centroid)
    best_flat = mx.argmax(scores.reshape(B, eff_k * per_centroid), axis=-1).astype(mx.int32)
    best_c_pos = best_flat // per_centroid
    best_local = best_flat % per_centroid
    best_c = mx.take_along_axis(top_c, best_c_pos[:, None], axis=1).squeeze(1)
    return order_2d[best_c, best_local]


class Gemma4MtpDrafter:
    """4-layer BF16 MTP draft head (`google/gemma-4-E2B-it-assistant`) + GPU `DraftScan` over target E2B KV caches.

    Attends directly in-place to target Layer 13 (last sliding storing layer) and Layer 14 (last full
    storing layer) across both BF16 `make_batch_cache` and `KV_i8` `make_slot_cache`, strictly for `B=1..4`.
    """

    MAX_BATCH_SIZE: int = MAX_MTP_BATCH_SIZE

    def __init__(
        self,
        target_model: FastGemma4Model,
        drafter_repo_or_dir: str | Path = DEFAULT_MTP_DRAFTER_REPO,
        *,
        top_k: int = DEFAULT_MTP_TOP_K,
        local_files_only: bool = True,
    ) -> None:
        if int(getattr(target_model, "hidden_size", 0)) != 1536 or int(getattr(target_model, "n_kv_heads", 0)) != 1:
            raise ValueError(
                "Gemma4MtpDrafter requires a Gemma-4 E2B target (hidden_size=1536, n_kv_heads=1); "
                f"got hidden_size={getattr(target_model, 'hidden_size', None)}, "
                f"n_kv_heads={getattr(target_model, 'n_kv_heads', None)}"
            )
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.drafter_dir = resolve_mtp_drafter_dir(drafter_repo_or_dir, local_files_only=local_files_only)
        sf_path = self.drafter_dir / "model.safetensors"
        if not sf_path.is_file():
            raise FileNotFoundError(f"Missing model.safetensors in MTP drafter directory {self.drafter_dir}")

        self.target = target_model
        self.top_k = int(top_k)
        self.eps = 1e-6

        raw = mx.load(str(sf_path))
        self.pre_proj_w = raw["pre_projection.weight"].astype(mx.bfloat16)  # (256, 3072)
        if int(self.pre_proj_w.shape[1]) != 2 * target_model.hidden_size:
            raise ValueError(
                f"Drafter pre_projection expects width {int(self.pre_proj_w.shape[1])}, "
                f"but 2 * target.hidden_size = {2 * target_model.hidden_size}"
            )
        self.post_proj_w = raw["post_projection.weight"].astype(mx.bfloat16)  # (1536, 256)
        self.norm_w = raw["model.norm.weight"].astype(mx.bfloat16)  # (256,)
        self.centroids_w = raw["masked_embedding.centroids.weight"].astype(mx.bfloat16)  # (2048, 256)

        self.num_centroids = int(self.centroids_w.shape[0])
        self.vocab_size = int(raw["model.embed_tokens.weight"].shape[0])
        if self.num_centroids <= 0 or self.vocab_size % self.num_centroids != 0:
            raise ValueError(
                f"Drafter vocab {self.vocab_size} is not divisible by num_centroids {self.num_centroids}"
            )
        self.per_centroid = self.vocab_size // self.num_centroids
        self.draft_dim = int(self.norm_w.shape[0])

        order = raw["masked_embedding.token_ordering"].astype(mx.int32)
        self.embed_perm = mx.contiguous(
            raw["model.embed_tokens.weight"][order]
            .astype(mx.bfloat16)
            .reshape(self.num_centroids, self.per_centroid, self.draft_dim)
        )
        self.order_2d = mx.contiguous(order.reshape(self.num_centroids, self.per_centroid))

        self.last_swa_layer = max(target_model.swa_layers)
        self.last_full_layer = max(target_model.full_layers)

        self.layers: list[dict[str, Any]] = []
        eval_targets: list[Any] = [
            self.pre_proj_w,
            self.post_proj_w,
            self.norm_w,
            self.centroids_w,
            self.embed_perm,
            self.order_2d,
        ]
        for i in range(4):
            lp = f"model.layers.{i}"
            is_full = i == 3
            kv_idx = self.last_full_layer if is_full else self.last_swa_layer
            hd = int(target_model.layers[kv_idx]["head_dim"])
            w_gate = raw[f"{lp}.mlp.gate_proj.weight"].astype(mx.bfloat16)
            w_up = raw[f"{lp}.mlp.up_proj.weight"].astype(mx.bfloat16)
            w_gu = mx.contiguous(mx.concatenate([w_gate, w_up], axis=0))
            w_next = raw[f"model.layers.{i + 1}.input_layernorm.weight"].astype(mx.bfloat16) if i < 3 else self.norm_w
            ld = {
                "idx": i,
                "is_full": is_full,
                "kv_idx": kv_idx,
                "head_dim": hd,
                "n_q": 4,
                "inv_freq": target_model.layers[kv_idx]["inv_freq"],
                "w_in": raw[f"{lp}.input_layernorm.weight"].astype(mx.bfloat16),
                "w_qn": raw[f"{lp}.self_attn.q_norm.weight"].astype(mx.bfloat16),
                "q_proj_w": raw[f"{lp}.self_attn.q_proj.weight"].astype(mx.bfloat16),
                "o_proj_w": raw[f"{lp}.self_attn.o_proj.weight"].astype(mx.bfloat16),
                "w_post_attn": raw[f"{lp}.post_attention_layernorm.weight"].astype(mx.bfloat16),
                "w_pre_ff": raw[f"{lp}.pre_feedforward_layernorm.weight"].astype(mx.bfloat16),
                "gate_up_w": w_gu,
                "down_w": raw[f"{lp}.mlp.down_proj.weight"].astype(mx.bfloat16),
                "w_post_ff": raw[f"{lp}.post_feedforward_layernorm.weight"].astype(mx.bfloat16),
                "w_next_norm": w_next,
                "layer_scalar": raw[f"{lp}.layer_scalar"].astype(mx.bfloat16).reshape((1,)),
            }
            self.layers.append(ld)
            eval_targets.extend(
                (
                    ld["w_in"],
                    ld["w_qn"],
                    ld["q_proj_w"],
                    ld["o_proj_w"],
                    ld["w_post_attn"],
                    ld["w_pre_ff"],
                    ld["gate_up_w"],
                    ld["down_w"],
                    ld["w_post_ff"],
                    ld["layer_scalar"],
                )
            )
        del raw
        mx.eval(*eval_targets)
        self.param_mb = sum(int(x.nbytes) for x in eval_targets) / (1024.0 * 1024.0)

    def draft_step(
        self,
        last_tokens: mx.array,
        last_hidden: mx.array,
        cache: dict[str, Any],
        off_kv: mx.array,
        off_rope: mx.array,
        *,
        slot_idx: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run one MTP step for `B <= 4` -> `(next_token_ids [B], next_hidden [B, 1, 1536])`."""
        B = int(last_tokens.shape[0])
        if not (1 <= B <= self.MAX_BATCH_SIZE):
            raise ValueError(f"Gemma4MtpDrafter is strictly gated to 1 <= B <= {self.MAX_BATCH_SIZE}, got B={B}")

        # Gemma-4 backbone embeddings are scaled by sqrt(hidden_size) before concatenating with post-norm hidden.
        tok_emb = self.target.embed_tokens(last_tokens) * self.target.embed_scale
        inp = mx.concatenate([tok_emb, last_hidden], axis=-1)
        x = inp @ self.pre_proj_w.T
        x_in = mx.fast.rms_norm(x, self.layers[0]["w_in"], self.eps)

        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        is_slot_cache = slot_idx is not None
        if is_slot_cache:
            ns_swa, ns_full, meta_swa, meta_full = self.target._get_sdpa_meta(cache, B)

        for ld in self.layers:
            kv_idx = ld["kv_idx"]
            c = doc_layers[kv_idx]
            q_raw = x_in @ ld["q_proj_w"].T
            q_rot = fused_q_norm_rope_ragged(q_raw, ld["w_qn"], ld["inv_freq"], off_rope, self.eps, n_q=4)

            if is_slot_cache:
                att = mqa8_sdpa_prefix_ragged_slot(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_kv,
                    slot_idx,
                    ns_full if ld["is_full"] else ns_swa,
                    meta_full if ld["is_full"] else meta_swa,
                    k_s_doc=c.get("k_s"),
                    v_s_doc=c.get("v_s"),
                    vec4_reduce=self.target.vec4_reduce,
                )
            else:
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_kv,
                    window_size=0 if ld["is_full"] else self.target.window_size,
                    doc_ring_cap=0 if ld["is_full"] else c["cap"],
                    n_splits=cache["ns_full"] if ld["is_full"] else cache["ns_swa"],
                    meta=cache["meta_full"] if ld["is_full"] else cache["meta_swa"],
                    vec4_reduce=self.target.vec4_reduce,
                )

            attn_out = att @ ld["o_proj_w"].T
            h_mid, mlp_in = fused_norm_add_norm(x, attn_out, ld["w_post_attn"], ld["w_pre_ff"], self.eps, None)
            mlp_act = fused_geglu_split(mlp_in @ ld["gate_up_w"].T)
            mlp_out = mlp_act @ ld["down_w"].T
            x, x_in = fused_norm_add_norm(
                h_mid, mlp_out, ld["w_post_ff"], ld["w_next_norm"], self.eps, ld["layer_scalar"]
            )

        next_hidden = x_in @ self.post_proj_w.T
        next_tok = draft_scan_top_k(x_in[:, 0, :], self.centroids_w, self.embed_perm, self.order_2d, top_k=self.top_k)
        return next_tok, next_hidden

    def draft_k(
        self,
        anchor_tokens: mx.array,
        seed_hidden: mx.array,
        cache: dict[str, Any],
        off_cur: mx.array,
        k: int,
        *,
        slot_idx: mx.array | None = None,
    ) -> mx.array:
        """Autoregressively draft `k` tokens in one fused GPU graph -> `(B, k)` int32."""
        B = int(anchor_tokens.shape[0])
        if not (1 <= B <= self.MAX_BATCH_SIZE):
            raise ValueError(f"Gemma4MtpDrafter is strictly gated to 1 <= B <= {self.MAX_BATCH_SIZE}, got B={B}")
        if int(k) <= 0:
            raise ValueError(f"draft_k must be positive, got {k}")
        off_kv = mx.maximum(off_cur - 1, 0)
        cur_toks = anchor_tokens.reshape(B, 1)
        cur_h = seed_hidden.reshape(B, 1, self.target.hidden_size)
        cols: list[mx.array] = []
        for i in range(int(k)):
            nxt_tok, cur_h = self.draft_step(cur_toks, cur_h, cache, off_kv, off_cur + i, slot_idx=slot_idx)
            cols.append(nxt_tok[:, None])
            cur_toks = nxt_tok[:, None]
        return mx.concatenate(cols, axis=1)


def run_decode_mtp(
    model: FastGemma4Model,
    drafter: Gemma4MtpDrafter,
    stop_token_ids: frozenset[int],
    grammar: GrammarHelper | None,
    init_lg: mx.array,
    init_off: mx.array,
    cache: dict[str, Any],
    B: int,
    *,
    max_tokens: int = 512,
    draft_k: int | None = None,
    enable_ff_jump: bool = True,
) -> dict[str, Any]:
    """Speculative MTP decode (`B=1..4`, `T=0`) supporting both free-text and `llguidance` constrained JSON."""
    if not (1 <= int(B) <= MAX_MTP_BATCH_SIZE):
        raise ValueError(f"run_decode_mtp is strictly gated to 1 <= B <= {MAX_MTP_BATCH_SIZE}, got B={B}")
    init_x = cache.get("_last_x_in")
    if init_x is None:
        raise RuntimeError("MTP decode requires `cache['_last_x_in']` from prefill_suffix_batch.")

    t_dec0 = time.perf_counter()
    new_to_old = model.new_to_old
    fallback_pad = grammar.upfront_ff[0] if (grammar is not None and grammar.upfront_ff) else int(new_to_old[0])

    if grammar is None:
        return _run_decode_mtp_unconstrained(
            model,
            drafter,
            stop_token_ids,
            init_lg,
            init_x,
            init_off,
            cache,
            B,
            max_tokens=max_tokens,
            draft_k=draft_k,
            t_dec0=t_dec0,
        )

    matchers = [grammar.new_matcher() for _ in range(B)]
    queues: list[collections.deque[int]] = [collections.deque() for _ in range(B)]
    gen_toks: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    active = list(range(B))

    ff_skipped = 0
    mask_fallbacks = 0
    verify_steps = 1
    mtp_rounds = 0
    mtp_proposed = 0
    mtp_accepted = 0
    ff_jump_steps = 0
    ff_jump_tokens = 0

    def sample_and_queue(
        orig_b: int,
        row_logits: np.ndarray | None,
        argmax_j: int,
        x_row_mx: mx.array | None = None,
    ) -> bool:
        nonlocal ff_skipped, mask_fallbacks
        m = matchers[orig_b]
        if row_logits is None:
            t_orig = int(new_to_old[argmax_j])
            if not m.validate_tokens([t_orig]):
                return False
        else:
            t_orig, fell = _pick_valid_token_safe(grammar, model, m, row_logits, argmax_j, x_row_mx=x_row_mx)
            if fell:
                mask_fallbacks += 1
        m.consume_token(t_orig)
        gen_toks[orig_b].append(t_orig)
        if m.is_stopped() and m.is_accepting():
            done[orig_b] = True
            queues[orig_b].clear()
            return True
        queues[orig_b].append(t_orig)
        ff = m.compute_ff_tokens()
        if ff:
            m.consume_tokens(ff)
            gen_toks[orig_b].extend(ff)
            ff_skipped += len(ff)
            if m.is_stopped() and m.is_accepting():
                done[orig_b] = True
                queues[orig_b].clear()
                return True
            queues[orig_b].extend(ff)
        if len(gen_toks[orig_b]) >= max_tokens:
            done[orig_b] = True
            queues[orig_b].clear()
        return True

    am0 = safe_argmax_with_nan_guard(init_lg[:, 0, :], axis=-1)
    mx.eval(am0, init_x)
    am0_list = am0.tolist()
    if any(int(x) < 0 for x in am0_list):
        raise FloatingPointError("NaN logits detected in run_decode_mtp init_lg")
    need_mask0: list[int] = []
    for r, b in enumerate(active):
        if not sample_and_queue(b, None, am0_list[r]):
            need_mask0.append(r)
    if need_mask0:
        lg_sub0 = np.array(init_lg[mx.array(need_mask0, dtype=mx.int32), 0, :].astype(mx.float32))
        for k_i, r in enumerate(need_mask0):
            sample_and_queue(active[r], lg_sub0[k_i], am0_list[r], x_row_mx=init_x[r, 0])

    seed_h = init_x[:, 0, :]
    seed_valid = [len(queues[b]) == 1 and not done[b] for b in active]
    off_cur = init_off

    while True:
        keep_rows = [r for r, b in enumerate(active) if not done[b]]
        if not keep_rows:
            break
        if len(keep_rows) < len(active):
            idx_mx = mx.array(keep_rows, dtype=mx.int32)
            cache = model.compact_cache(cache, keep_rows)
            off_cur = off_cur[idx_mx]
            seed_h = seed_h[idx_mx]
            seed_valid = [seed_valid[r] for r in keep_rows]
            active = [active[r] for r in keep_rows]

        B_cur = len(active)
        cur_k = int(draft_k) if draft_k is not None else recommended_mtp_draft_k(B_cur, has_grammar=True)
        L_step = cur_k + 1

        q_lens = [len(queues[b]) for b in active]
        if enable_ff_jump and min(q_lens) >= 2 and min(q_lens) == max(q_lens):
            k_cand = min(64, min(q_lens))
            cur_swa_cap = int(cache["layers"][model.swa_layers[0]]["cap"]) if cache.get("layers") else 512
            full_cap = int(cache["layers"][model.full_layers[0]]["cap"]) if cache.get("layers") else 1024
            swa_safe = (cur_swa_cap >= model.window_size + k_cand - 1) or (full_cap <= cur_swa_cap)
            if not swa_safe:
                priv_max = int(mx.max(off_cur).item()) - int(cache["l_sys"])  # guard: allow-sync (ring wrap check)
                swa_safe = (priv_max + k_cand - 1) < cur_swa_cap
            if swa_safe:
                seq_np = np.empty((B_cur, k_cand), dtype=np.int32)
                for r, b in enumerate(active):
                    q = queues[b]
                    for cj in range(k_cand):
                        seq_np[r, cj] = q.popleft()
                lg_j = model.forward_step_ragged(
                    mx.array(seq_np, dtype=mx.int32), cache, off_cur, apply_softcap=True, keep_from_col=k_cand - 1
                )
                am_j = safe_argmax_with_nan_guard(lg_j[:, 0, :], axis=-1)
                x_j = cache["_last_x_in"]
                mx.eval(am_j, x_j)  # guard: allow-eval (causal FF jump step requires synchronous token IDs)
                am_jl = am_j.tolist()
                if any(int(x) < 0 for x in am_jl):
                    raise FloatingPointError("NaN logits detected in run_decode_mtp FF jump step")
                need_m: list[int] = []
                for r, b in enumerate(active):
                    if len(queues[b]) == 0 and not sample_and_queue(b, None, am_jl[r]):
                        need_m.append(r)
                if need_m:
                    lg_sub = np.array(lg_j[mx.array(need_m, dtype=mx.int32), 0, :].astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
                    for k_i, r in enumerate(need_m):
                        sample_and_queue(active[r], lg_sub[k_i], am_jl[r], x_row_mx=x_j[r, 0])
                off_cur = off_cur + k_cand
                seed_h = x_j[:, 0, :]
                seed_valid = [len(queues[b]) == 1 and not done[b] for b in active]
                verify_steps += 1
                ff_jump_steps += 1
                ff_jump_tokens += k_cand * B_cur
                del lg_j, am_j, x_j
                continue

        batch_seq = np.full((B_cur, L_step), fallback_pad, dtype=np.int32)
        n_ff_feed = [0] * B_cur
        first_toks = [fallback_pad] * B_cur
        drafts: list[list[int]] = [[] for _ in range(B_cur)]
        need_mtp_rows: list[int] = []
        mtp_anchors: list[int] = []

        for r, b in enumerate(active):
            q = queues[b]
            fed: list[int] = []
            while q and len(fed) < L_step:
                fed.append(q.popleft())
            n_ff_feed[r] = len(fed)
            if fed:
                first_toks[r] = int(fed[0])
            if not q and len(fed) == 1 and seed_valid[r]:
                need_mtp_rows.append(r)
                mtp_anchors.append(int(fed[0]))
            else:
                pad_t = fed[-1] if fed else fallback_pad
                while len(fed) < L_step:
                    fed.append(pad_t)
                batch_seq[r] = fed

        if need_mtp_rows:
            mtp_rounds += 1
            anc_by_row = {r: mtp_anchors[idx] for idx, r in enumerate(need_mtp_rows)}
            all_anc = [anc_by_row.get(r, first_toks[r]) for r in range(B_cur)]
            mtp_dr = drafter.draft_k(mx.array(all_anc, dtype=mx.int32)[:, None], seed_h, cache, off_cur, cur_k)
            mx.eval(mtp_dr)  # guard: allow-eval (draft tokens merged into grammar verify batch)
            mtp_dr_list = mtp_dr.tolist()
            for r in need_mtp_rows:
                row_dr = mtp_dr_list[r]
                batch_seq[r, 0] = anc_by_row[r]
                batch_seq[r, 1 : 1 + cur_k] = row_dr[:cur_k]
                drafts[r] = [int(t) for t in row_dr[:cur_k]]

        keep_from_col = 0
        if enable_ff_jump and L_step >= 2:
            min_nf = min(n_ff_feed)
            if min_nf >= 2:
                keep_from_col = min(L_step - 1, min_nf - 1)
                ff_jump_steps += 1
                ff_jump_tokens += (keep_from_col + 1) * B_cur

        batch_mx = mx.array(batch_seq, dtype=mx.int32)
        if keep_from_col > 0:
            lg_step = model.forward_step_ragged(batch_mx, cache, off_cur, apply_softcap=True, keep_from_col=keep_from_col)
        else:
            lg_step = model.forward_step_ragged(batch_mx, cache, off_cur, apply_softcap=True)
        am_mx = safe_argmax_with_nan_guard(lg_step, axis=-1)
        x_step = cache["_last_x_in"]
        mx.eval(am_mx, x_step)  # guard: allow-eval (speculative draft verification step requires synchronous token IDs)
        am_list = am_mx.tolist()
        if any(int(x) < 0 for row in am_list for x in row):
            raise FloatingPointError("NaN logits detected in run_decode_mtp verification step")
        verify_steps += 1

        lg_full_np: np.ndarray | None = None
        acc_lens = [0] * B_cur
        seed_col_idx = [0] * B_cur
        next_seed_valid = [False] * B_cur

        for r, b in enumerate(active):
            nf = n_ff_feed[r]
            dr = drafts[r]
            mtp_proposed += len(dr)

            if queues[b]:
                acc_lens[r] = nf
                seed_col_idx[r] = max(0, nf - 1 - keep_from_col)
                next_seed_valid[r] = False
                continue

            matched = 0
            stopped = False
            for j, d_tok in enumerate(dr):
                pos = (nf - 1 - keep_from_col) + j
                pred_orig = int(new_to_old[am_list[r][pos]])
                if not matchers[b].validate_tokens([pred_orig]):
                    if lg_full_np is None:
                        lg_full_np = np.array(lg_step.astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
                    sample_and_queue(b, lg_full_np[r, pos], am_list[r][pos], x_row_mx=x_step[r, pos])
                    acc_lens[r] = nf + matched
                    seed_col_idx[r] = pos
                    next_seed_valid[r] = len(queues[b]) == 1 and not done[b]
                    stopped = True
                    break
                if pred_orig == d_tok:
                    matchers[b].consume_token(pred_orig)
                    gen_toks[b].append(pred_orig)
                    matched += 1
                    mtp_accepted += 1
                    if len(gen_toks[b]) >= max_tokens or (matchers[b].is_stopped() and matchers[b].is_accepting()):
                        done[b] = True
                        queues[b].clear()
                        acc_lens[r] = nf + matched
                        seed_col_idx[r] = pos
                        stopped = True
                        break
                    ff = matchers[b].compute_ff_tokens()
                    if ff:
                        matchers[b].consume_tokens(ff)
                        gen_toks[b].extend(ff)
                        ff_skipped += len(ff)
                        if (matchers[b].is_stopped() and matchers[b].is_accepting()) or len(gen_toks[b]) >= max_tokens:
                            done[b] = True
                            queues[b].clear()
                        else:
                            queues[b].extend(ff)
                        acc_lens[r] = nf + matched
                        seed_col_idx[r] = pos
                        next_seed_valid[r] = False
                        stopped = True
                        break
                else:
                    sample_and_queue(b, None, am_list[r][pos])
                    acc_lens[r] = nf + matched
                    seed_col_idx[r] = pos
                    next_seed_valid[r] = len(queues[b]) == 1 and not done[b]
                    stopped = True
                    break

            if not stopped:
                pos = (nf - 1 - keep_from_col) + len(dr)
                if pos >= 0:
                    ok = sample_and_queue(b, None, am_list[r][pos])
                    if not ok:
                        if lg_full_np is None:
                            lg_full_np = np.array(lg_step.astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
                        sample_and_queue(b, lg_full_np[r, pos], am_list[r][pos], x_row_mx=x_step[r, pos])
                acc_lens[r] = nf + len(dr)
                seed_col_idx[r] = max(0, pos)
                next_seed_valid[r] = len(queues[b]) == 1 and not done[b]

        row_ix = mx.arange(B_cur, dtype=mx.int32)
        col_ix = mx.array(seed_col_idx, dtype=mx.int32)
        seed_h = x_step[row_ix, col_ix, :]
        seed_valid = next_seed_valid
        off_cur = off_cur + mx.array(acc_lens, dtype=mx.int32)
        del lg_step, am_mx, x_step

    dec_sec = time.perf_counter() - t_dec0
    upfront_len = len(grammar.upfront_ff)
    tot_gen = sum(len(g) for g in gen_toks) + upfront_len * B
    return {
        "decode_sec": dec_sec,
        "gen_toks": gen_toks,
        "tot_out_tok": tot_gen,
        "verify_steps": verify_steps,
        "acc_tok_per_step": tot_gen / max(B * verify_steps, 1),
        "mtp_rounds": mtp_rounds,
        "mtp_accept_rate": mtp_accepted / max(mtp_proposed, 1),
        "mtp_accepted": mtp_accepted,
        "mtp_proposed": mtp_proposed,
        "ff_tokens": ff_skipped + upfront_len * B,
        "ff_jump_steps": ff_jump_steps,
        "ff_jump_tokens": ff_jump_tokens,
        "mask_fallbacks": mask_fallbacks,
        "decode_tok_s": tot_gen / max(dec_sec, 1e-9),
        "used_speculative": True,
        "speculative_backend": "mtp",
    }


def _run_decode_mtp_unconstrained(
    model: FastGemma4Model,
    drafter: Gemma4MtpDrafter,
    stop_token_ids: frozenset[int],
    init_lg: mx.array,
    init_x: mx.array,
    init_off: mx.array,
    cache: dict[str, Any],
    B: int,
    *,
    max_tokens: int,
    draft_k: int | None,
    t_dec0: float,
) -> dict[str, Any]:
    """Zero-sync-between-draft-and-verify unconstrained greedy MTP loop for `1 <= B <= 4`."""
    new_to_old = model.new_to_old
    gen_toks: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    active = list(range(B))

    am0 = safe_argmax_with_nan_guard(init_lg[:, 0, :], axis=-1)
    mx.eval(am0, init_x)
    am0_list = am0.tolist()
    if any(int(x) < 0 for x in am0_list):
        raise FloatingPointError("NaN logits detected in run_decode_mtp init_lg")

    anchors = [0] * B
    for r, b in enumerate(active):
        t_orig = int(new_to_old[am0_list[r]])
        if t_orig in stop_token_ids:
            done[b] = True
        else:
            gen_toks[b].append(t_orig)
            anchors[r] = t_orig
            if len(gen_toks[b]) >= max_tokens:
                done[b] = True

    seed_h = init_x[:, 0, :]
    off_cur = init_off
    verify_steps = 1
    mtp_rounds = 0
    mtp_proposed = 0
    mtp_accepted = 0
    fallback_tok = int(new_to_old[0])

    while True:
        keep_rows = [r for r, b in enumerate(active) if not done[b]]
        if not keep_rows:
            break
        if len(keep_rows) < len(active):
            idx_mx = mx.array(keep_rows, dtype=mx.int32)
            cache = model.compact_cache(cache, keep_rows)
            off_cur = off_cur[idx_mx]
            seed_h = seed_h[idx_mx]
            anchors = [anchors[r] for r in keep_rows]
            active = [active[r] for r in keep_rows]

        B_cur = len(active)
        cur_k = int(draft_k) if draft_k is not None else recommended_mtp_draft_k(B_cur, has_grammar=False)
        anchor_mx = mx.array(anchors, dtype=mx.int32)[:, None]

        # Fuse all K draft steps + target verify into ONE GPU submission (zero CPU sync between draft & verify).
        drafts_mx = drafter.draft_k(anchor_mx, seed_h, cache, off_cur, cur_k)
        verify_toks = mx.concatenate([anchor_mx, drafts_mx], axis=1)
        lg_step = model.forward_step_ragged(verify_toks, cache, off_cur, apply_softcap=True)
        am_mx = safe_argmax_with_nan_guard(lg_step, axis=-1)
        x_step = cache["_last_x_in"]
        mx.eval(drafts_mx, am_mx, x_step)  # guard: allow-eval (fused speculative draft+verify sync)
        verify_steps += 1
        mtp_rounds += 1

        drafts_list = drafts_mx.tolist()
        am_list = am_mx.tolist()
        if any(int(x) < 0 for row in am_list for x in row):
            raise FloatingPointError("NaN logits detected in run_decode_mtp verification step")

        acc_lens = [0] * B_cur
        seed_col_idx = [0] * B_cur
        new_anchors = [fallback_tok] * B_cur
        for r, b in enumerate(active):
            dr = drafts_list[r]
            am_r = am_list[r]
            mtp_proposed += len(dr)
            matched = 0
            stopped = False
            for j, d_tok in enumerate(dr):
                pred = int(new_to_old[am_r[j]])
                if pred == d_tok:
                    if pred in stop_token_ids:
                        done[b] = True
                        acc_lens[r] = 1 + matched
                        seed_col_idx[r] = matched
                        stopped = True
                        break
                    gen_toks[b].append(pred)
                    matched += 1
                    mtp_accepted += 1
                    if len(gen_toks[b]) >= max_tokens:
                        done[b] = True
                        acc_lens[r] = 1 + matched
                        seed_col_idx[r] = matched
                        stopped = True
                        break
                else:
                    if pred in stop_token_ids:
                        done[b] = True
                    else:
                        gen_toks[b].append(pred)
                        new_anchors[r] = pred
                        if len(gen_toks[b]) >= max_tokens:
                            done[b] = True
                    acc_lens[r] = 1 + matched
                    seed_col_idx[r] = matched
                    stopped = True
                    break
            if not stopped:
                bonus = int(new_to_old[am_r[len(dr)]])
                if bonus in stop_token_ids:
                    done[b] = True
                else:
                    gen_toks[b].append(bonus)
                    new_anchors[r] = bonus
                    if len(gen_toks[b]) >= max_tokens:
                        done[b] = True
                acc_lens[r] = 1 + len(dr)
                seed_col_idx[r] = len(dr)

        row_ix = mx.arange(B_cur, dtype=mx.int32)
        col_ix = mx.array(seed_col_idx, dtype=mx.int32)
        seed_h = x_step[row_ix, col_ix, :]
        anchors = new_anchors
        off_cur = off_cur + mx.array(acc_lens, dtype=mx.int32)
        del lg_step, am_mx, x_step

    dec_sec = time.perf_counter() - t_dec0
    tot_gen = sum(len(g) for g in gen_toks)
    return {
        "decode_sec": dec_sec,
        "gen_toks": gen_toks,
        "tot_out_tok": tot_gen,
        "verify_steps": verify_steps,
        "acc_tok_per_step": tot_gen / max(B * verify_steps, 1),
        "mtp_rounds": mtp_rounds,
        "mtp_accept_rate": mtp_accepted / max(mtp_proposed, 1),
        "mtp_accepted": mtp_accepted,
        "mtp_proposed": mtp_proposed,
        "ff_tokens": 0,
        "ff_jump_steps": 0,
        "ff_jump_tokens": 0,
        "mask_fallbacks": 0,
        "decode_tok_s": tot_gen / max(dec_sec, 1e-9),
        "used_speculative": True,
        "speculative_backend": "mtp",
    }


__all__ = [
    "DEFAULT_MTP_DRAFTER_REPO",
    "DEFAULT_MTP_TOP_K",
    "MAX_MTP_BATCH_SIZE",
    "Gemma4MtpDrafter",
    "draft_scan_top_k",
    "recommended_mtp_draft_k",
    "resolve_mtp_drafter_dir",
    "run_decode_mtp",
    "should_use_mtp_drafter",
]
