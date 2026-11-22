"""Opt-in Gemma-4 MTP Drafter (`gemma-4-*-assistant`) + GPU `DraftScan`, strictly gated to `B=1..4`."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import time
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .grammar import GrammarHelper
from .kernels import (
    fused_geglu_split,
    fused_norm_add_norm,
    fused_q_norm_rope_ragged,
    mqa8_sdpa_prefix_ragged,
    mqa8_sdpa_prefix_ragged_slot,
)
from .model import (
    FastGemma4Model,
    _fuse_linears,
    _load_linear,
    _load_qembedding,
    _singleton_verify_linear,
    safe_argmax_with_nan_guard,
)
from .speculative import (
    _REPEAT_4GRAM_BREAKER_DEFAULT,
    _REPEAT_4GRAM_MIN_HITS,
    _REPEAT_4GRAM_MIN_HITS_CONSTRAINED,
    _REPEAT_4GRAM_PENALTY,
    _REPEAT_4GRAM_WINDOW,
    _apply_repeat_penalties_row,
    _pick_valid_token_safe,
    _repeated_4gram_penalties,
    _seed_tables,
    _update_ngrams_from_tail,
)

DEFAULT_MTP_DRAFTER_REPO: str = "google/gemma-4-E2B-it-assistant"
DEFAULT_QAT_MTP_DRAFTER_REPO: str = "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant"
DEFAULT_MTP_TOP_K: int = 32
MAX_MTP_BATCH_SIZE: int = 4
MAX_CONSTRAINED_MTP_BATCH_SIZE_APPLE9: int = 2

_APPLE10_NAX_CACHE: bool | None = None


def select_mtp_drafter_repo(target_model: Any) -> str:
    """Auto-pair QAT target checkpoints with the matching QAT MTP assistant repo."""
    if bool(getattr(target_model, "is_qat_checkpoint", False)):
        return DEFAULT_QAT_MTP_DRAFTER_REPO
    mdir = str(getattr(target_model, "model_dir", "") or "").lower()
    if "qat" in mdir:
        return DEFAULT_QAT_MTP_DRAFTER_REPO
    return DEFAULT_MTP_DRAFTER_REPO


def _host_supports_apple10_nax() -> bool:
    """Probe Apple10+ NAX support via `fast_shared.hw_check.probe_host_hardware()` once per process."""
    global _APPLE10_NAX_CACHE
    if _APPLE10_NAX_CACHE is not None:
        return _APPLE10_NAX_CACHE
    detected = False
    try:
        from fast_shared.hw_check import probe_host_hardware, supports_apple10_nax

        hw = probe_host_hardware()
        if isinstance(hw, dict):
            probe_dict = dict(hw)
            if "system" not in probe_dict and (
                probe_dict.get("gpu_family") in ("Apple9", "Apple10")
                or probe_dict.get("supports_apple10") is not None
                or "apple" in str(probe_dict.get("brand") or "").lower()
            ):
                probe_dict["system"] = "Darwin"
                probe_dict.setdefault("machine", "arm64")
            detected = bool(supports_apple10_nax(probe_dict))
    except Exception:
        detected = False
    _APPLE10_NAX_CACHE = detected
    return detected


def _resolve_apple10_nax(apple10_nax: bool | None) -> bool:
    if apple10_nax is not None:
        return bool(apple10_nax)
    return _host_supports_apple10_nax()


def should_use_mtp_drafter(
    drafter: Any | None,
    use_ane: bool,
    batch_w: int,
    temperature: float,
    *,
    speculative: bool | None = None,
    apple10_nax: bool | None = None,
    has_grammar: bool = False,
) -> bool:
    """Enable MTP drafter strictly for winning batch regimes on each GPU generation."""
    if drafter is None or speculative is False or use_ane or temperature > 0.0:
        return False
    if _resolve_apple10_nax(apple10_nax):
        max_b = 1
    elif has_grammar:
        max_b = MAX_CONSTRAINED_MTP_BATCH_SIZE_APPLE9
    else:
        max_b = MAX_MTP_BATCH_SIZE
    return 1 <= int(batch_w) <= max_b


def recommended_mtp_draft_k(
    batch_size: int,
    *,
    has_grammar: bool = False,
    apple10_nax: bool | None = None,
) -> int:
    """Select roof-line optimal draft depth `K` for Apple9 (`1 <= B <= 4`) or Apple10+ NAX (`B == 1`).

    - `Apple10+ NAX` (M5+): strictly `B == 1` (`K = 3` with grammar, `K = 4` free-text); for `B >= 2`,
      NAX TensorOps (`M >= 8`) already accelerate batched decode by `1.5x–1.9x`, whereas MTP adds overhead.
    - `Apple9` (M4):
      - `B == 1`: `K = 3` (`L_q = 4`, optimal `2.63x` JSON / `2.13x` free-text; `K = 4` also strong on free-text).
      - `2 <= B <= 4`: `K = 2` keeps `M = B * (K + 1) <= 12 < 16`, staying below the `16 <= M <= 64`
        MPP tile threshold where verify latency jumps `~2.4x`.
    """
    b = int(batch_size)
    is_nax = _resolve_apple10_nax(apple10_nax)
    max_b = 1 if is_nax else MAX_MTP_BATCH_SIZE
    if not (1 <= b <= max_b):
        if is_nax:
            raise ValueError(
                f"MTP speculative drafting on Apple10+ NAX is strictly gated to B == 1 "
                f"(NAX TensorOps accelerate B >= 2 directly); got B={b}"
            )
        raise ValueError(
            f"MTP speculative drafting is strictly gated to 1 <= B <= {MAX_MTP_BATCH_SIZE} "
            f"(never enable for B >= 8); got B={b}"
        )
    if b == 1:
        return 3 if has_grammar else 4
    return 2


def _has_drafter_weights(cand: Path) -> bool:
    return cand.is_dir() and any(cand.glob("model*.safetensors"))


def resolve_mtp_drafter_dir(
    drafter_repo_or_dir: str | Path = DEFAULT_MTP_DRAFTER_REPO,
    *,
    local_files_only: bool = True,
) -> Path:
    """Resolve local checkpoint directory or HuggingFace cache snapshot for the Gemma-4 E2B MTP assistant."""
    p = Path(drafter_repo_or_dir).expanduser()
    if _has_drafter_weights(p):
        return p.resolve()
    from huggingface_hub import snapshot_download

    repo_str = str(drafter_repo_or_dir)
    candidates = (
        [repo_str, DEFAULT_MTP_DRAFTER_REPO]
        if (local_files_only and repo_str == DEFAULT_QAT_MTP_DRAFTER_REPO)
        else [repo_str]
    )
    last_err: Exception | None = None
    for cand_repo in candidates:
        try:
            snap = Path(
                snapshot_download(
                    cand_repo,
                    allow_patterns=["*.safetensors", "config.json"],
                    local_files_only=local_files_only,
                )
            )
            if _has_drafter_weights(snap):
                return snap
            last_err = FileNotFoundError(f"Missing model*.safetensors in {snap}")
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"Unable to resolve MTP drafter: {drafter_repo_or_dir}")


def draft_scan_top_k(
    x_norm_2d: mx.array,
    centroids_w: mx.array,
    embed_perm: mx.array,
    order_2d: mx.array,
    *,
    top_k: int = DEFAULT_MTP_TOP_K,
    embed_perm_t: mx.array | None = None,
) -> mx.array:
    """Score centroids, scan contiguous pre-permuted `top_k` centroid tiles (`gather_mm`), and return argmax IDs."""
    B = int(x_norm_2d.shape[0])
    num_centroids = int(centroids_w.shape[0])
    per_centroid = int(embed_perm.shape[1])
    eff_k = max(1, min(int(top_k), num_centroids))

    c_logits = x_norm_2d @ centroids_w.T  # (B, num_centroids)
    if eff_k < num_centroids:
        top_c = mx.argpartition(-c_logits, kth=eff_k, axis=-1)[:, :eff_k]
    else:
        top_c = mx.broadcast_to(mx.arange(num_centroids, dtype=mx.int32)[None, :], (B, num_centroids))

    if mx.default_device() == mx.gpu:
        ep_t = embed_perm_t if embed_perm_t is not None else mx.contiguous(embed_perm.swapaxes(-1, -2))
        scores = mx.gather_mm(
            x_norm_2d[:, None, None, :].astype(ep_t.dtype),
            ep_t,
            rhs_indices=top_c,
        ).squeeze(-2)
    else:
        cand_w = embed_perm[top_c]
        scores = mx.matmul(cand_w, x_norm_2d[:, None, :, None].astype(cand_w.dtype)).squeeze(-1)

    best_flat = mx.argmax(scores.reshape(B, eff_k * per_centroid), axis=-1).astype(mx.int32)
    best_c_pos = best_flat // per_centroid
    best_local = best_flat % per_centroid
    best_c = mx.take_along_axis(top_c, best_c_pos[:, None], axis=1).squeeze(1)
    return order_2d[best_c, best_local]


def draft_scan_top_candidates(
    x_norm_2d: mx.array,
    centroids_w: mx.array,
    embed_perm: mx.array,
    order_2d: mx.array,
    *,
    top_k: int = DEFAULT_MTP_TOP_K,
    top_m: int = 16,
    embed_perm_t: mx.array | None = None,
) -> mx.array:
    """Return top `top_m` candidate token IDs `[B, M]` descending by score across `top_k` centroid tiles."""
    B = int(x_norm_2d.shape[0])
    num_centroids = int(centroids_w.shape[0])
    per_centroid = int(embed_perm.shape[1])
    eff_k = max(1, min(int(top_k), num_centroids))

    c_logits = x_norm_2d @ centroids_w.T
    if eff_k < num_centroids:
        top_c = mx.argpartition(-c_logits, kth=eff_k, axis=-1)[:, :eff_k]
    else:
        top_c = mx.broadcast_to(mx.arange(num_centroids, dtype=mx.int32)[None, :], (B, num_centroids))

    if mx.default_device() == mx.gpu:
        ep_t = embed_perm_t if embed_perm_t is not None else mx.contiguous(embed_perm.swapaxes(-1, -2))
        scores = mx.gather_mm(
            x_norm_2d[:, None, None, :].astype(ep_t.dtype),
            ep_t,
            rhs_indices=top_c,
        ).squeeze(-2)
    else:
        cand_w = embed_perm[top_c]
        scores = mx.matmul(cand_w, x_norm_2d[:, None, :, None].astype(cand_w.dtype)).squeeze(-1)

    flat_scores = scores.reshape(B, eff_k * per_centroid)
    total_cands = int(flat_scores.shape[1])
    m_eff = max(1, min(int(top_m), total_cands))
    if m_eff == 1:
        best_flat = mx.argmax(flat_scores, axis=-1).astype(mx.int32)[:, None]
    else:
        if m_eff < total_cands:
            top_idx = mx.argpartition(-flat_scores, kth=m_eff, axis=-1)[:, :m_eff]
        else:
            top_idx = mx.broadcast_to(mx.arange(total_cands, dtype=mx.int32)[None, :], (B, total_cands))
        top_vals = mx.take_along_axis(flat_scores, top_idx, axis=-1)
        ord_m = mx.argsort(-top_vals, axis=-1)
        best_flat = mx.take_along_axis(top_idx, ord_m, axis=-1).astype(mx.int32)
    best_c_pos = best_flat // per_centroid
    best_local = best_flat % per_centroid
    best_c = mx.take_along_axis(top_c, best_c_pos, axis=1)
    return order_2d[best_c, best_local]


class _ConstrainedMtpWalker:
    """Splash-style single-`LLMatcher` DFS walker (`_reach` / `_walk` + prompt lookup + rollback)."""

    def __init__(
        self,
        matcher: Any,
        vocab_size: int,
        stop_token_ids: frozenset[int],
        t3: dict[tuple[int, int, int], list[int]],
        t2: dict[tuple[int, int], list[int]],
    ) -> None:
        self.matcher = matcher
        self.vocab_size = int(vocab_size)
        self.stop_token_ids = stop_token_ids
        self.t3 = t3
        self.t2 = t2

    def _reach(self, rows_rem: int) -> tuple[int, ...]:
        if rows_rem <= 0:
            return ()
        return tuple(int(t) for t in self.matcher.compute_ff_tokens()[:rows_rem])

    def _prompt_lookup_candidates(self, ctx: Sequence[int]) -> list[int]:
        if len(ctx) >= 3:
            hit = self.t3.get((int(ctx[-3]), int(ctx[-2]), int(ctx[-1])))
            if hit:
                return [int(t) for t in hit]
        if len(ctx) >= 2:
            hit2 = self.t2.get((int(ctx[-2]), int(ctx[-1])))
            if hit2:
                return [int(t) for t in hit2]
        return []

    def _walk(
        self,
        cand_step: int,
        slots_rem: int,
        cands_by_step: Sequence[Sequence[int]],
        ctx: list[int],
        shifted: bool,
        *,
        max_branch: int = 4,
    ) -> tuple[list[int], bool]:
        """DFS up to `slots_rem` tokens using `try_consume_tokens` + `rollback`; restores matcher state."""
        if slots_rem <= 0:
            return [], False

        ff = self._reach(slots_rem)
        if ff:
            n_ff = int(self.matcher.try_consume_tokens(list(ff)))
            if n_ff > 0:
                taken = list(ff[:n_ff])
                accepting = bool(self.matcher.is_stopped() and self.matcher.is_accepting())
                if accepting or n_ff >= slots_rem:
                    self.matcher.rollback(n_ff)
                    return taken, accepting
                ctx.extend(taken)
                tail, tail_acc = self._walk(
                    cand_step,
                    slots_rem - n_ff,
                    cands_by_step,
                    ctx,
                    True,
                    max_branch=max_branch,
                )
                del ctx[-n_ff:]
                self.matcher.rollback(n_ff)
                return taken + tail, tail_acc

        pl = self._prompt_lookup_candidates(ctx)
        step_i = min(cand_step, len(cands_by_step) - 1) if cands_by_step else 0
        dr = [int(t) for t in cands_by_step[step_i]] if cands_by_step else []

        ordered: list[tuple[int, bool, int]] = []
        seen: set[int] = set()
        if not shifted and dr:
            t0 = dr[0]
            if 0 <= t0 < self.vocab_size and t0 not in self.stop_token_ids:
                seen.add(t0)
                ordered.append((t0, False, cand_step + 1))
        for pt in pl:
            if pt not in seen and 0 <= pt < self.vocab_size and pt not in self.stop_token_ids:
                seen.add(pt)
                ordered.append((pt, True, cand_step))
        for dt in dr:
            if dt not in seen and 0 <= dt < self.vocab_size and dt not in self.stop_token_ids:
                seen.add(dt)
                ordered.append((dt, shifted or not dr or dt != dr[0], cand_step + 1))

        best_path: list[int] = []
        best_acc = False
        tried = 0
        branch_limit = max_branch if cand_step == 0 and not shifted else min(2, max_branch)
        for tok, child_shifted, next_cand_step in ordered:
            if self.matcher.try_consume_tokens([tok]) <= 0:
                continue
            tried += 1
            accepting = bool(self.matcher.is_stopped() and self.matcher.is_accepting())
            if accepting or slots_rem == 1:
                cand_path, cand_acc = [tok], accepting
            else:
                ctx.append(tok)
                tail, cand_acc = self._walk(
                    next_cand_step,
                    slots_rem - 1,
                    cands_by_step,
                    ctx,
                    child_shifted,
                    max_branch=branch_limit,
                )
                ctx.pop()
                cand_path = [tok] + tail
            self.matcher.rollback(1)
            if (len(cand_path) > len(best_path)) or (cand_acc and not best_acc and len(cand_path) == len(best_path)):
                best_path = cand_path
                best_acc = cand_acc
            if len(best_path) >= slots_rem or (tried >= branch_limit) or (not child_shifted and best_path):
                break
        return best_path, best_acc

    def build_draft(
        self,
        cands_by_step: Sequence[Sequence[int]],
        ctx: Sequence[int],
        max_draft: int,
    ) -> list[int]:
        if max_draft <= 0 or (self.matcher.is_stopped() and self.matcher.is_accepting()):
            return []
        path, _ = self._walk(0, int(max_draft), cands_by_step, [int(t) for t in ctx], False)
        return path[: int(max_draft)]


class Gemma4MtpDrafter:
    """Multi-layer MTP draft head (`gemma-4-*-assistant`) + pre-permuted GPU `DraftScan` / projected `lm_head`.

    Attends directly in-place to target last sliding and last full storing layers across both BF16
    `make_batch_cache` and `KV_i8`/BF16 `make_slot_cache`, strictly for `B=1..4`.
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
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        ref_str = str(drafter_repo_or_dir)
        is_e2b_repo_ref = ("e2b" in ref_str.lower()) and not Path(ref_str).is_dir()
        if is_e2b_repo_ref and (
            int(getattr(target_model, "hidden_size", 0)) != 1536
            or int(getattr(target_model, "n_kv_heads", 0)) != 1
        ):
            raise ValueError(
                "Gemma4MtpDrafter requires a Gemma-4 E2B target (hidden_size=1536, n_kv_heads=1); "
                f"got hidden_size={getattr(target_model, 'hidden_size', None)}, "
                f"n_kv_heads={getattr(target_model, 'n_kv_heads', None)}"
            )

        self.drafter_dir = resolve_mtp_drafter_dir(drafter_repo_or_dir, local_files_only=local_files_only)
        sf_files = sorted(self.drafter_dir.glob("model*.safetensors"))
        if not sf_files:
            raise FileNotFoundError(f"Missing model*.safetensors in MTP drafter directory {self.drafter_dir}")

        self.target = target_model
        self.top_k = int(top_k)

        cfg_path = self.drafter_dir / "config.json"
        dcfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        dtcfg: dict[str, Any] = dcfg.get("text_config") if isinstance(dcfg.get("text_config"), dict) else {}
        qcfg_d: dict[str, Any] = {}
        if isinstance(dcfg.get("quantization_config"), dict):
            qcfg_d.update(dcfg["quantization_config"])
        if isinstance(dcfg.get("quantization"), dict):
            qcfg_d.update(dcfg["quantization"])

        self.eps = float(dtcfg.get("rms_norm_eps", getattr(target_model, "eps", 1e-6)))

        raw: dict[str, mx.array] = {}
        for sf_path in sf_files:
            raw.update(mx.load(str(sf_path)))

        self.pre_projection = _load_linear(
            raw, "pre_projection", qcfg_d, mx.bfloat16, expected_in_f=2 * int(target_model.hidden_size)
        )
        pre_in_f = (
            int(self.pre_projection.scales.shape[1]) * int(self.pre_projection.group_size)
            if isinstance(self.pre_projection, nn.QuantizedLinear)
            else int(self.pre_projection.weight.shape[1])
        )
        if pre_in_f != 2 * int(target_model.hidden_size):
            raise ValueError(
                f"Drafter pre_projection expects width {pre_in_f}, "
                f"but 2 * target.hidden_size = {2 * int(target_model.hidden_size)}"
            )
        self.draft_dim = int(raw["model.norm.weight"].shape[0])
        self.post_projection = _load_linear(
            raw, "post_projection", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim
        )
        self.norm_w = raw["model.norm.weight"].astype(mx.bfloat16)

        has_ordered_keys = (
            "masked_embedding.centroids.weight" in raw and "masked_embedding.token_ordering" in raw
        )
        self.use_ordered_embeddings = bool(dcfg.get("use_ordered_embeddings", has_ordered_keys)) and has_ordered_keys

        eval_targets: list[Any] = [
            self.pre_projection.parameters(),
            self.post_projection.parameters(),
            self.norm_w,
        ]

        if self.use_ordered_embeddings:
            self.centroids_w: mx.array | None = raw["masked_embedding.centroids.weight"].astype(mx.bfloat16)
            self.num_centroids = int(self.centroids_w.shape[0])
            emb_mod = _load_qembedding(
                raw, "model.embed_tokens", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim
            )
            if isinstance(emb_mod, nn.QuantizedEmbedding):
                emb_bf16 = mx.dequantize(
                    emb_mod.weight,
                    emb_mod.scales,
                    emb_mod.biases,
                    group_size=emb_mod.group_size,
                    bits=emb_mod.bits,
                    mode=emb_mod.mode,
                ).astype(mx.bfloat16)
            else:
                emb_bf16 = emb_mod.weight.astype(mx.bfloat16)
            self.vocab_size = int(emb_bf16.shape[0])
            if self.num_centroids <= 0 or self.vocab_size % self.num_centroids != 0:
                raise ValueError(
                    f"Drafter vocab {self.vocab_size} is not divisible by num_centroids {self.num_centroids}"
                )
            self.per_centroid = self.vocab_size // self.num_centroids
            order = raw["masked_embedding.token_ordering"].astype(mx.int32)
            # Pre-permute embedding table ONCE at load time into a single contiguous `float16`
            # `[num_centroids, draft_dim, per_centroid]` buffer (`1x` RAM instead of `3x`!) and alias
            # `embed_perm_f16` / `embed_perm` as zero-copy transposed views.
            perm_f16 = emb_bf16[order].astype(mx.float16).reshape(
                self.num_centroids, self.per_centroid, self.draft_dim
            )
            self.embed_perm_t: mx.array | None = mx.contiguous(perm_f16.swapaxes(-1, -2))
            self.embed_perm_f16: mx.array | None = self.embed_perm_t.swapaxes(-1, -2)
            self.embed_perm: mx.array | None = self.embed_perm_f16
            self.order_2d: mx.array | None = mx.contiguous(order.reshape(self.num_centroids, self.per_centroid))
            self.lm_head: nn.Module | None = None
            eval_targets.extend((self.centroids_w, self.embed_perm_t, self.order_2d))
        else:
            # `use_ordered_embeddings=False` (`26B-A4B` / `31B` assistants): full projected `lm_head` (untied or tied).
            self.centroids_w = None
            self.num_centroids = 0
            self.per_centroid = 0
            self.embed_perm = None
            self.embed_perm_f16 = None
            self.embed_perm_t = None
            self.order_2d = None
            tie_word = bool(dcfg.get("tie_word_embeddings", "lm_head.weight" not in raw))
            if not tie_word and "lm_head.weight" in raw:
                self.lm_head = _load_linear(raw, "lm_head", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim)
            else:
                emb_mod = _load_qembedding(
                    raw, "model.embed_tokens", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim
                )
                if isinstance(emb_mod, nn.QuantizedEmbedding):
                    ql = nn.QuantizedLinear(
                        self.draft_dim,
                        int(emb_mod.weight.shape[0]),
                        bias=False,
                        group_size=emb_mod.group_size,
                        bits=emb_mod.bits,
                        mode=emb_mod.mode,
                    )
                    ql.weight = emb_mod.weight
                    ql.scales = emb_mod.scales
                    ql.biases = emb_mod.biases
                    self.lm_head = ql
                else:
                    lin = nn.Linear(self.draft_dim, int(emb_mod.weight.shape[0]), bias=False)
                    lin.weight = emb_mod.weight
                    self.lm_head = lin
            self.vocab_size = int(self.lm_head.weight.shape[0])
            eval_targets.append(self.lm_head.parameters())

        self.last_swa_layer = max(target_model.swa_layers)
        self.last_full_layer = max(target_model.full_layers)

        n_d_layers = int(dtcfg.get("num_hidden_layers", 0))
        if n_d_layers <= 0:
            while f"model.layers.{n_d_layers}.input_layernorm.weight" in raw:
                n_d_layers += 1
        d_layer_types = list(dtcfg.get("layer_types") or [])

        self.layers: list[dict[str, Any]] = []
        for i in range(n_d_layers):
            lp = f"model.layers.{i}"
            if i < len(d_layer_types):
                is_full = d_layer_types[i] == "full_attention"
            else:
                is_full = i == (n_d_layers - 1)
            kv_idx = self.last_full_layer if is_full else self.last_swa_layer
            hd = int(target_model.layers[kv_idx]["head_dim"])

            q_mod = _load_linear(raw, f"{lp}.self_attn.q_proj", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim)
            q_out_dim = int(q_mod.weight.shape[0])
            n_q_d = q_out_dim // hd
            o_mod = _load_linear(raw, f"{lp}.self_attn.o_proj", qcfg_d, mx.bfloat16, expected_in_f=q_out_dim)
            g_mod = _load_linear(raw, f"{lp}.mlp.gate_proj", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim)
            u_mod = _load_linear(raw, f"{lp}.mlp.up_proj", qcfg_d, mx.bfloat16, expected_in_f=self.draft_dim)
            gu_mod = _fuse_linears([g_mod, u_mod], mx.bfloat16)
            interm_d = int(g_mod.weight.shape[0])
            dn_mod = _load_linear(raw, f"{lp}.mlp.down_proj", qcfg_d, mx.bfloat16, expected_in_f=interm_d)

            w_next = (
                raw[f"model.layers.{i + 1}.input_layernorm.weight"].astype(mx.bfloat16)
                if i + 1 < n_d_layers
                else self.norm_w
            )
            l_sc = raw.get(f"{lp}.layer_scalar")
            layer_scalar = l_sc.astype(mx.bfloat16).reshape((1,)) if l_sc is not None else mx.ones((1,), dtype=mx.bfloat16)
            ld = {
                "idx": i,
                "is_full": is_full,
                "kv_idx": kv_idx,
                "head_dim": hd,
                "n_q": n_q_d,
                "inv_freq": target_model.layers[kv_idx]["inv_freq"],
                "w_in": raw[f"{lp}.input_layernorm.weight"].astype(mx.bfloat16),
                "w_qn": raw[f"{lp}.self_attn.q_norm.weight"].astype(mx.bfloat16),
                "q_proj": q_mod,
                "o_proj": o_mod,
                "w_post_attn": raw[f"{lp}.post_attention_layernorm.weight"].astype(mx.bfloat16),
                "w_pre_ff": raw[f"{lp}.pre_feedforward_layernorm.weight"].astype(mx.bfloat16),
                "gate_up_proj": gu_mod,
                "down_proj": dn_mod,
                "w_post_ff": raw[f"{lp}.post_feedforward_layernorm.weight"].astype(mx.bfloat16),
                "w_next_norm": w_next,
                "layer_scalar": layer_scalar,
            }
            self.layers.append(ld)
            eval_targets.extend(
                (
                    ld["w_in"],
                    ld["w_qn"],
                    ld["q_proj"].parameters(),
                    ld["o_proj"].parameters(),
                    ld["w_post_attn"],
                    ld["w_pre_ff"],
                    ld["gate_up_proj"].parameters(),
                    ld["down_proj"].parameters(),
                    ld["w_post_ff"],
                    ld["layer_scalar"],
                )
            )
        del raw
        from mlx.utils import tree_flatten

        mx.eval(eval_targets)
        self.param_mb = sum(int(x.nbytes) for _, x in tree_flatten(eval_targets)) / (1024.0 * 1024.0)
        self._drafter_swa_meta_cache: dict[tuple[int, int, int, int], mx.array] = {}

    def _get_drafter_swa_meta(self, l_sys: int, swa_cap: int, eff_win: int, ns_swa: int) -> mx.array:
        key = (int(l_sys), int(swa_cap), int(eff_win), int(ns_swa))
        meta = self._drafter_swa_meta_cache.get(key)
        if meta is None:
            meta = mx.array([l_sys, swa_cap, eff_win, 1, ns_swa, swa_cap - 1], dtype=mx.int32)
            mx.eval(meta)
            self._drafter_swa_meta_cache[key] = meta
        return meta

    def draft_scan_top_k(self, h_2d: mx.array, *, k: int | None = None) -> mx.array:
        """Select greedy draft token IDs `[B]` via pre-permuted `gather_mm` centroid tiles or projected `lm_head`."""
        eff_k = self.top_k if k is None else int(k)
        if self.use_ordered_embeddings:
            assert self.centroids_w is not None and self.embed_perm is not None and self.order_2d is not None
            return draft_scan_top_k(
                h_2d,
                self.centroids_w,
                self.embed_perm,
                self.order_2d,
                top_k=eff_k,
                embed_perm_t=self.embed_perm_t,
            )
        assert self.lm_head is not None
        dr_lg = _singleton_verify_linear(self.lm_head, h_2d[:, None, :])
        return mx.argmax(dr_lg[:, 0, :], axis=-1).astype(mx.int32)

    def draft_scan_top_m(self, h_2d: mx.array, *, k: int | None = None, top_m: int = 16) -> mx.array:
        """Select top `top_m` draft candidate IDs `[B, M]` via centroid tiles or projected `lm_head`."""
        eff_k = self.top_k if k is None else int(k)
        if self.use_ordered_embeddings:
            assert self.centroids_w is not None and self.embed_perm is not None and self.order_2d is not None
            return draft_scan_top_candidates(
                h_2d,
                self.centroids_w,
                self.embed_perm,
                self.order_2d,
                top_k=eff_k,
                top_m=top_m,
                embed_perm_t=self.embed_perm_t,
            )
        assert self.lm_head is not None
        dr_lg = _singleton_verify_linear(self.lm_head, h_2d[:, None, :])[:, 0, :]
        m_eff = max(1, min(int(top_m), int(dr_lg.shape[-1])))
        if m_eff == 1:
            return mx.argmax(dr_lg, axis=-1).astype(mx.int32)[:, None]
        top_idx = mx.argpartition(-dr_lg, kth=m_eff, axis=-1)[:, :m_eff] if m_eff < int(dr_lg.shape[-1]) else mx.broadcast_to(mx.arange(m_eff, dtype=mx.int32)[None, :], (int(dr_lg.shape[0]), m_eff))
        top_vals = mx.take_along_axis(dr_lg, top_idx, axis=-1)
        ord_m = mx.argsort(-top_vals, axis=-1)
        return mx.take_along_axis(top_idx, ord_m, axis=-1).astype(mx.int32)

    def draft_step(
        self,
        last_tokens: mx.array,
        last_hidden: mx.array,
        cache: dict[str, Any],
        off_kv: mx.array,
        off_rope: mx.array,
        *,
        slot_idx: mx.array | None = None,
        top_m: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Run one MTP step for `B <= 4` -> `(next_token_ids [B] or [B, M], next_hidden [B, 1, H_target])`."""
        B = int(last_tokens.shape[0])
        if not (1 <= B <= self.MAX_BATCH_SIZE):
            raise ValueError(f"Gemma4MtpDrafter is strictly gated to 1 <= B <= {self.MAX_BATCH_SIZE}, got B={B}")

        tok_emb = self.target.embed_tokens(last_tokens) * self.target.embed_scale
        inp = mx.concatenate([tok_emb, last_hidden], axis=-1)
        x = self.pre_projection(inp)
        x_in = mx.fast.rms_norm(x, self.layers[0]["w_in"], self.eps)

        k_sys_l = cache["k_sys"]
        v_sys_l = cache["v_sys"]
        doc_layers = cache["layers"]
        l_sys = int(cache["l_sys"])
        # Because `off_kv = off_cur - 1` sets `t_hi = off_cur`, `eff_win = window_size - 1` sets
        # `t_lo = max(0, off_cur - window_size + 1)`, matching the exact anchor SWA window!
        eff_win = max(1, int(self.target.window_size) - 1)
        is_slot_cache = "swa_cap" in cache
        eff_slot_idx = (mx.arange(B, dtype=mx.int32) if slot_idx is None else slot_idx) if is_slot_cache else None
        if is_slot_cache:
            ns_swa, ns_full, _, meta_full = self.target._get_sdpa_meta(cache, B)
            meta_swa_dr = self._get_drafter_swa_meta(l_sys, int(cache["swa_cap"]), eff_win, ns_swa)

        for ld in self.layers:
            kv_idx = ld["kv_idx"]
            c = doc_layers[kv_idx]
            q_raw = ld["q_proj"](x_in)
            q_rot = fused_q_norm_rope_ragged(q_raw, ld["w_qn"], ld["inv_freq"], off_rope, self.eps, n_q=ld["n_q"])

            if is_slot_cache:
                assert eff_slot_idx is not None
                att = mqa8_sdpa_prefix_ragged_slot(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_kv,
                    eff_slot_idx,
                    ns_full if ld["is_full"] else ns_swa,
                    meta_full if ld["is_full"] else meta_swa_dr,
                    k_s_doc=c.get("k_s"),
                    v_s_doc=c.get("v_s"),
                    vec4_reduce=self.target.vec4_reduce,
                )
            else:
                meta_swa_b = (
                    cache["meta_full"]
                    if ld["is_full"]
                    else self._get_drafter_swa_meta(l_sys, int(c["cap"]), eff_win, int(cache["ns_swa"]))
                )
                att = mqa8_sdpa_prefix_ragged(
                    q_rot,
                    k_sys_l[kv_idx],
                    v_sys_l[kv_idx],
                    c["k"],
                    c["v"],
                    off_kv,
                    window_size=0 if ld["is_full"] else eff_win,
                    doc_ring_cap=0 if ld["is_full"] else c["cap"],
                    n_splits=cache["ns_full"] if ld["is_full"] else cache["ns_swa"],
                    meta=meta_swa_b,
                    vec4_reduce=self.target.vec4_reduce,
                )

            attn_out = ld["o_proj"](att)
            h_mid, mlp_in = fused_norm_add_norm(x, attn_out, ld["w_post_attn"], ld["w_pre_ff"], self.eps, None)
            mlp_act = fused_geglu_split(ld["gate_up_proj"](mlp_in))
            mlp_out = ld["down_proj"](mlp_act)
            x, x_in = fused_norm_add_norm(
                h_mid, mlp_out, ld["w_post_ff"], ld["w_next_norm"], self.eps, ld["layer_scalar"]
            )

        next_hidden = self.post_projection(x_in)
        if int(top_m) > 1:
            next_tok = self.draft_scan_top_m(x_in[:, 0, :], k=self.top_k, top_m=int(top_m))
        else:
            next_tok = self.draft_scan_top_k(x_in[:, 0, :], k=self.top_k)
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
        """Autoregressively draft `k` tokens in one fused GPU graph -> `(B, k)` int32 (holding `off_rope == off_cur`)."""
        B = int(anchor_tokens.shape[0])
        if not (1 <= B <= self.MAX_BATCH_SIZE):
            raise ValueError(f"Gemma4MtpDrafter is strictly gated to 1 <= B <= {self.MAX_BATCH_SIZE}, got B={B}")
        if int(k) <= 0:
            raise ValueError(f"draft_k must be positive, got {k}")
        off_kv = mx.maximum(off_cur - 1, 0)
        cur_toks = anchor_tokens.reshape(B, 1)
        cur_h = seed_hidden.reshape(B, 1, self.target.hidden_size)
        cols: list[mx.array] = []
        for _ in range(int(k)):
            # Hold RoPE anchor `off_cur` constant across all K steps (`gemma4_assistant.py` invariant).
            nxt_tok, cur_h = self.draft_step(cur_toks, cur_h, cache, off_kv, off_cur, slot_idx=slot_idx)
            cols.append(nxt_tok[:, None])
            cur_toks = nxt_tok[:, None]
        return mx.concatenate(cols, axis=1)

    def draft_k_candidates(
        self,
        anchor_tokens: mx.array,
        seed_hidden: mx.array,
        cache: dict[str, Any],
        off_cur: mx.array,
        k: int,
        *,
        top_m: int = 16,
        slot_idx: mx.array | None = None,
    ) -> mx.array:
        """Autoregressively draft `k` steps with top `top_m` candidates `[B, k, M]` in one fused GPU graph."""
        B = int(anchor_tokens.shape[0])
        if not (1 <= B <= self.MAX_BATCH_SIZE):
            raise ValueError(f"Gemma4MtpDrafter is strictly gated to 1 <= B <= {self.MAX_BATCH_SIZE}, got B={B}")
        if int(k) <= 0:
            raise ValueError(f"draft_k must be positive, got {k}")
        m_eff = max(1, int(top_m))
        off_kv = mx.maximum(off_cur - 1, 0)
        cur_toks = anchor_tokens.reshape(B, 1)
        cur_h = seed_hidden.reshape(B, 1, self.target.hidden_size)
        cols: list[mx.array] = []
        for _ in range(int(k)):
            nxt_cands, cur_h = self.draft_step(
                cur_toks, cur_h, cache, off_kv, off_cur, slot_idx=slot_idx, top_m=m_eff
            )
            nxt_top1 = nxt_cands[:, 0:1] if m_eff > 1 else nxt_cands[:, None]
            cols.append((nxt_cands if m_eff > 1 else nxt_top1)[:, None, :])
            cur_toks = nxt_top1
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
    init_am: mx.array | None = None,
    per_doc_grammars: Sequence[Any] | None = None,
    doc_prompts: Sequence[Sequence[int]] | None = None,
    doc_indices: Sequence[int] | None = None,
    constrained_top_m: int = 16,
    apple10_nax: bool | None = None,
    repeat_4gram_breaker: bool | None = None,
) -> dict[str, Any]:
    """Speculative MTP decode (`B=1..4` Apple9 / `B=1` Apple10, `T=0`) supporting constrained JSON + DFS."""
    if not (1 <= int(B) <= MAX_MTP_BATCH_SIZE):
        raise ValueError(f"run_decode_mtp is strictly gated to 1 <= B <= {MAX_MTP_BATCH_SIZE}, got B={B}")
    if apple10_nax is True and int(B) > 1:
        raise ValueError(f"run_decode_mtp on Apple10+ NAX is strictly gated to B == 1, got B={B}")
    if per_doc_grammars is not None and len(per_doc_grammars) != int(B):
        raise ValueError(f"Mismatched per_doc_grammars={len(per_doc_grammars)} vs B={B}")
    if doc_prompts is not None and len(doc_prompts) != int(B):
        raise ValueError(f"Mismatched doc_prompts={len(doc_prompts)} vs B={B}")
    if doc_indices is not None and len(doc_indices) != int(B):
        raise ValueError(f"Mismatched doc_indices={len(doc_indices)} vs B={B}")

    init_x = cache.get("_last_x_in")
    if init_x is None:
        raise RuntimeError("MTP decode requires `cache['_last_x_in']` from prefill.")

    t_dec0 = time.perf_counter()
    new_to_old = model.new_to_old
    eff_repeat_breaker = _REPEAT_4GRAM_BREAKER_DEFAULT if repeat_4gram_breaker is None else bool(repeat_4gram_breaker)
    doc_helpers = [per_doc_grammars[b] if per_doc_grammars is not None else grammar for b in range(B)]
    if all(h is None for h in doc_helpers):
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
            init_am=init_am,
            repeat_4gram_breaker=eff_repeat_breaker,
        )

    orig_doc_ids = [int(doc_indices[b]) if doc_indices is not None else b for b in range(B)]

    def _helper_upfront_tokens(b: int) -> list[int]:
        h = doc_helpers[b]
        if h is None:
            return []
        up_fn = getattr(h, "upfront_ff_for_doc", None)
        if callable(up_fn):
            return list(up_fn(orig_doc_ids[b]))
        return list(getattr(h, "upfront_ff", ()))

    def _new_doc_matcher(b: int) -> Any:
        h = doc_helpers[b]
        if h is None:
            return None
        by_doc = getattr(h, "new_matcher_for_doc", None)
        if callable(by_doc):
            return by_doc(orig_doc_ids[b])
        return h.new_matcher()

    upfront_lists = [_helper_upfront_tokens(b) for b in range(B)]
    upfront_lens = [len(u) for u in upfront_lists]
    fallback_pad = next((int(u[0]) for u in upfront_lists if u), int(new_to_old[0]))

    is_slot_cache = "swa_cap" in cache
    slot_mx = mx.arange(B, dtype=mx.int32)
    matchers = [_new_doc_matcher(b) for b in range(B)]
    doc_ctx: list[list[int]] = [
        [int(t) for t in doc_prompts[b]] if doc_prompts is not None else [] for b in range(B)
    ]
    prompt_tables = _seed_tables(doc_ctx)
    n_indexed = [len(ctx) for ctx in doc_ctx]
    vocab_sz = max(int(getattr(drafter, "vocab_size", 0)), int(new_to_old.size))
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

    def _append_tok(orig_b: int, tok: int) -> None:
        gen_toks[orig_b].append(tok)
        doc_ctx[orig_b].append(tok)
        t3, t2 = prompt_tables[orig_b]
        n_indexed[orig_b] = _update_ngrams_from_tail(t3, t2, doc_ctx[orig_b], n_indexed[orig_b])

    def _append_toks(orig_b: int, toks: Sequence[int]) -> None:
        if not toks:
            return
        gen_toks[orig_b].extend(toks)
        doc_ctx[orig_b].extend(toks)
        t3, t2 = prompt_tables[orig_b]
        n_indexed[orig_b] = _update_ngrams_from_tail(t3, t2, doc_ctx[orig_b], n_indexed[orig_b])

    def _row_repeat_penalties(orig_b: int, *, constrained: bool) -> dict[int, float]:
        if not eff_repeat_breaker:
            return {}
        return _repeated_4gram_penalties(
            gen_toks[orig_b],
            window=_REPEAT_4GRAM_WINDOW,
            min_prior_hits=_REPEAT_4GRAM_MIN_HITS_CONSTRAINED if constrained else _REPEAT_4GRAM_MIN_HITS,
            base_penalty=_REPEAT_4GRAM_PENALTY,
            stop_token_ids=stop_token_ids,
        )

    def sample_and_queue(
        orig_b: int,
        row_logits: np.ndarray | None,
        argmax_j: int,
        x_row_mx: mx.array | None = None,
        *,
        penalized: bool = False,
    ) -> bool:
        nonlocal ff_skipped, mask_fallbacks
        m = matchers[orig_b]
        h = doc_helpers[orig_b]
        if m is None or h is None:
            t_orig = int(new_to_old[argmax_j])
            if not penalized:
                pen = _row_repeat_penalties(orig_b, constrained=False)
                if pen and t_orig in pen:
                    if row_logits is None:
                        return False
                    _, adj_am = _apply_repeat_penalties_row(row_logits, pen, model.old_to_new)
                    t_orig = int(new_to_old[adj_am])
            _append_tok(orig_b, t_orig)
            if t_orig in stop_token_ids or len(gen_toks[orig_b]) >= max_tokens:
                done[orig_b] = True
                queues[orig_b].clear()
                return True
            queues[orig_b].append(t_orig)
            return True

        pen = {} if penalized else _row_repeat_penalties(orig_b, constrained=True)
        if row_logits is None:
            t_orig = int(new_to_old[argmax_j])
            if (pen and t_orig in pen) or not m.validate_tokens([t_orig]):
                return False
        else:
            if pen:
                row_logits, argmax_j = _apply_repeat_penalties_row(row_logits, pen, model.old_to_new)
            t_orig, fell = _pick_valid_token_safe(h, model, m, row_logits, argmax_j, x_row_mx=x_row_mx)
            if fell:
                mask_fallbacks += 1
        m.consume_token(t_orig)
        _append_tok(orig_b, t_orig)
        if m.is_stopped() and m.is_accepting():
            done[orig_b] = True
            queues[orig_b].clear()
            return True
        queues[orig_b].append(t_orig)
        ff = m.compute_ff_tokens()
        if ff:
            m.consume_tokens(ff)
            _append_toks(orig_b, ff)
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

    am0 = init_am if init_am is not None else safe_argmax_with_nan_guard(init_lg[:, 0, :], axis=-1)
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
            if is_slot_cache:
                slot_mx = slot_mx[idx_mx]
            else:
                cache = model.compact_cache(cache, keep_rows)
            off_cur = off_cur[idx_mx]
            seed_h = seed_h[idx_mx]
            seed_valid = [seed_valid[r] for r in keep_rows]
            active = [active[r] for r in keep_rows]

        B_cur = len(active)
        if draft_k is not None:
            cur_k = int(draft_k)
        else:
            eff_nax_k = False if (apple10_nax is None and B_cur > 1) else apple10_nax
            cur_k = recommended_mtp_draft_k(B_cur, has_grammar=True, apple10_nax=eff_nax_k)
        L_step = cur_k + 1

        q_lens = [len(queues[b]) for b in active]
        if enable_ff_jump and min(q_lens) >= 2 and min(q_lens) == max(q_lens):
            max_jmp = int(cache.get("drain_slack", 64)) if is_slot_cache else 64
            k_cand = min(max(2, max_jmp), min(q_lens))
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
                seq_mx = mx.array(seq_np, dtype=mx.int32)
                if is_slot_cache and hasattr(model, "forward_ff_jump_slot"):
                    lg_j, am_j = model.forward_ff_jump_slot(seq_mx, cache, off_cur, slot_mx, return_logits=True)
                elif hasattr(model, "forward_verify_mtp_slot"):
                    lg_j, am_j_2d = model.forward_verify_mtp_slot(
                        seq_mx,
                        cache,
                        off_cur,
                        slot_idx=(slot_mx if is_slot_cache else None),
                        return_logits=True,
                        keep_from_col=k_cand - 1,
                    )
                    am_j = am_j_2d[:, 0]
                elif is_slot_cache:
                    lg_j = model.forward_step_slot(
                        seq_mx, cache, off_cur, slot_mx, apply_softcap=True, keep_from_col=k_cand - 1
                    )
                    am_j = safe_argmax_with_nan_guard(lg_j[:, 0, :], axis=-1)
                else:
                    lg_j = model.forward_step_ragged(
                        seq_mx, cache, off_cur, apply_softcap=True, keep_from_col=k_cand - 1
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
            any_constrained_row = any(matchers[active[r]] is not None for r in need_mtp_rows)
            top_m_req = max(1, int(constrained_top_m)) if any_constrained_row else 1
            mtp_cands = drafter.draft_k_candidates(
                mx.array(all_anc, dtype=mx.int32)[:, None],
                seed_h,
                cache,
                off_cur,
                cur_k,
                top_m=top_m_req,
                slot_idx=(slot_mx if is_slot_cache else None),
            )
            mx.eval(mtp_cands)  # guard: allow-eval (draft candidate tree merged into grammar verify batch)
            mtp_cands_list = mtp_cands.tolist()
            for r in need_mtp_rows:
                b = active[r]
                row_cands = mtp_cands_list[r]
                m_row = matchers[b]
                if m_row is None:
                    row_dr = [int(step_cands[0]) for step_cands in row_cands[:cur_k]]
                else:
                    t3, t2 = prompt_tables[b]
                    walker = _ConstrainedMtpWalker(
                        m_row,
                        vocab_sz,
                        stop_token_ids,
                        t3,
                        t2,
                    )
                    row_dr = walker.build_draft(row_cands[:cur_k], doc_ctx[b], cur_k)
                batch_seq[r, 0] = anc_by_row[r]
                if row_dr:
                    n_dr = min(len(row_dr), cur_k)
                    batch_seq[r, 1 : 1 + n_dr] = row_dr[:n_dr]
                drafts[r] = [int(t) for t in row_dr[:cur_k]]

        keep_from_col = 0
        if enable_ff_jump and L_step >= 2:
            min_nf = min(n_ff_feed)
            if min_nf >= 2:
                keep_from_col = min(L_step - 1, min_nf - 1)
                ff_jump_steps += 1
                ff_jump_tokens += (keep_from_col + 1) * B_cur

        batch_mx = mx.array(batch_seq, dtype=mx.int32)
        if hasattr(model, "forward_verify_mtp_slot"):
            lg_step, am_mx = model.forward_verify_mtp_slot(
                batch_mx,
                cache,
                off_cur,
                slot_idx=(slot_mx if is_slot_cache else None),
                return_logits=True,
                keep_from_col=keep_from_col,
            )
        elif is_slot_cache:
            lg_step = model.forward_step_slot(
                batch_mx, cache, off_cur, slot_mx, apply_softcap=True, keep_from_col=keep_from_col
            )
            am_mx = safe_argmax_with_nan_guard(lg_step, axis=-1)
        elif keep_from_col > 0:
            lg_step = model.forward_step_ragged(batch_mx, cache, off_cur, apply_softcap=True, keep_from_col=keep_from_col)
            am_mx = safe_argmax_with_nan_guard(lg_step, axis=-1)
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

            m_b = matchers[b]
            matched = 0
            stopped = False
            j = 0
            while j < len(dr):
                d_tok = dr[j]
                pos = (nf - 1 - keep_from_col) + j
                eff_am = am_list[r][pos]
                pred_orig = int(new_to_old[eff_am])
                adj_lg: np.ndarray | None = None
                pen = _row_repeat_penalties(b, constrained=(m_b is not None))
                if pen and pred_orig in pen:
                    if lg_full_np is None:
                        lg_full_np = np.array(lg_step.astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
                    adj_lg, eff_am = _apply_repeat_penalties_row(lg_full_np[r, pos], pen, model.old_to_new)
                    pred_orig = int(new_to_old[eff_am])
                if m_b is None:
                    if pred_orig == d_tok:
                        _append_tok(b, pred_orig)
                        matched += 1
                        mtp_accepted += 1
                        if pred_orig in stop_token_ids or len(gen_toks[b]) >= max_tokens:
                            done[b] = True
                            queues[b].clear()
                            acc_lens[r] = nf + matched
                            seed_col_idx[r] = pos
                            stopped = True
                            break
                        j += 1
                        continue
                    sample_and_queue(b, adj_lg, eff_am, penalized=(adj_lg is not None))
                    acc_lens[r] = nf + matched
                    seed_col_idx[r] = pos
                    next_seed_valid[r] = len(queues[b]) == 1 and not done[b]
                    stopped = True
                    break

                if not m_b.validate_tokens([pred_orig]):
                    if adj_lg is None:
                        if lg_full_np is None:
                            lg_full_np = np.array(lg_step.astype(mx.float32))  # guard: allow-sync (lazy D2H on evaluated logits)
                        sample_and_queue(b, lg_full_np[r, pos], eff_am, x_row_mx=x_step[r, pos], penalized=False)
                    else:
                        sample_and_queue(b, adj_lg, eff_am, x_row_mx=x_step[r, pos], penalized=True)
                    acc_lens[r] = nf + matched
                    seed_col_idx[r] = pos
                    next_seed_valid[r] = len(queues[b]) == 1 and not done[b]
                    stopped = True
                    break
                if pred_orig == d_tok:
                    m_b.consume_token(pred_orig)
                    _append_tok(b, pred_orig)
                    matched += 1
                    mtp_accepted += 1
                    if len(gen_toks[b]) >= max_tokens or (m_b.is_stopped() and m_b.is_accepting()):
                        done[b] = True
                        queues[b].clear()
                        acc_lens[r] = nf + matched
                        seed_col_idx[r] = pos
                        stopped = True
                        break
                    ff = m_b.compute_ff_tokens()
                    if ff:
                        n_ff_in_dr = 0
                        rem_dr = len(dr) - (j + 1)
                        if rem_dr > 0:
                            while (
                                n_ff_in_dr < len(ff)
                                and n_ff_in_dr < rem_dr
                                and dr[j + 1 + n_ff_in_dr] == ff[n_ff_in_dr]
                            ):
                                n_ff_in_dr += 1
                        if n_ff_in_dr > 0:
                            ff_in = ff[:n_ff_in_dr]
                            m_b.consume_tokens(ff_in)
                            _append_toks(b, ff_in)
                            ff_skipped += n_ff_in_dr
                            matched += n_ff_in_dr
                            mtp_accepted += n_ff_in_dr
                        ff_rem = ff[n_ff_in_dr:]
                        if ff_rem:
                            m_b.consume_tokens(ff_rem)
                            _append_toks(b, ff_rem)
                            ff_skipped += len(ff_rem)
                        if (m_b.is_stopped() and m_b.is_accepting()) or len(gen_toks[b]) >= max_tokens:
                            done[b] = True
                            queues[b].clear()
                            acc_lens[r] = nf + matched
                            seed_col_idx[r] = pos + n_ff_in_dr
                            stopped = True
                            break
                        if ff_rem:
                            queues[b].extend(ff_rem)
                            acc_lens[r] = nf + matched
                            seed_col_idx[r] = pos + n_ff_in_dr
                            next_seed_valid[r] = False
                            stopped = True
                            break
                        j += 1 + n_ff_in_dr
                        continue
                    j += 1
                else:
                    sample_and_queue(b, adj_lg, eff_am, penalized=(adj_lg is not None))
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
    tot_upfront = sum(upfront_lens)
    tot_gen = sum(len(g) for g in gen_toks) + tot_upfront
    valid_schemas = [
        True if matchers[b] is None else bool(matchers[b].is_stopped() and matchers[b].is_accepting())
        for b in range(B)
    ]
    return {
        "decode_sec": dec_sec,
        "gen_toks": gen_toks,
        "valid_schemas": valid_schemas,
        "tot_out_tok": tot_gen,
        "verify_steps": verify_steps,
        "acc_tok_per_step": tot_gen / max(B * verify_steps, 1),
        "mtp_rounds": mtp_rounds,
        "mtp_accept_rate": mtp_accepted / max(mtp_proposed, 1),
        "mtp_accepted": mtp_accepted,
        "mtp_proposed": mtp_proposed,
        "ff_tokens": ff_skipped + tot_upfront,
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
    init_am: mx.array | None = None,
    repeat_4gram_breaker: bool | None = None,
) -> dict[str, Any]:
    """Zero-sync-between-draft-and-verify unconstrained greedy MTP loop (`0.0` ULP exact to `L_q=1`)."""
    is_slot_cache = "swa_cap" in cache
    slot_mx = mx.arange(B, dtype=mx.int32)
    new_to_old = model.new_to_old
    eff_repeat_breaker = _REPEAT_4GRAM_BREAKER_DEFAULT if repeat_4gram_breaker is None else bool(repeat_4gram_breaker)
    gen_toks: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    active = list(range(B))

    am0 = init_am if init_am is not None else safe_argmax_with_nan_guard(init_lg[:, 0, :], axis=-1)
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
            if is_slot_cache:
                slot_mx = slot_mx[idx_mx]
            else:
                cache = model.compact_cache(cache, keep_rows)
            off_cur = off_cur[idx_mx]
            seed_h = seed_h[idx_mx]
            anchors = [anchors[r] for r in keep_rows]
            active = [active[r] for r in keep_rows]

        B_cur = len(active)
        cur_k = int(draft_k) if draft_k is not None else recommended_mtp_draft_k(B_cur, has_grammar=False)
        anchor_mx = mx.array(anchors, dtype=mx.int32)[:, None]

        # Fuse all K draft steps + singleton target verify into ONE GPU submission!
        drafts_mx = drafter.draft_k(
            anchor_mx,
            seed_h,
            cache,
            off_cur,
            cur_k,
            slot_idx=(slot_mx if is_slot_cache else None),
        )
        verify_toks = mx.concatenate([anchor_mx, drafts_mx], axis=1)
        if hasattr(model, "forward_verify_mtp_slot"):
            x_step, am_mx = model.forward_verify_mtp_slot(
                verify_toks,
                cache,
                off_cur,
                slot_idx=(slot_mx if is_slot_cache else None),
                return_logits=False,
            )
        else:
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
        def _pick_unconstrained(orig_b: int, raw_am: int, r: int, pos: int) -> int:
            pred = int(new_to_old[raw_am])
            if eff_repeat_breaker:
                pen = _repeated_4gram_penalties(
                    gen_toks[orig_b],
                    window=_REPEAT_4GRAM_WINDOW,
                    min_prior_hits=_REPEAT_4GRAM_MIN_HITS,
                    base_penalty=_REPEAT_4GRAM_PENALTY,
                    stop_token_ids=stop_token_ids,
                )
                if pen and pred in pen:
                    x_cell = x_step[r : r + 1, pos : pos + 1, :]
                    lg_cell = model._apply_lm_head(x_cell, apply_softcap=True)[0, 0]
                    row_lg = np.array(lg_cell.astype(mx.float32))  # guard: allow-sync (rare loop-breaker fallback)
                    _, adj_am = _apply_repeat_penalties_row(row_lg, pen, model.old_to_new)
                    return int(new_to_old[adj_am])
            return pred

        for r, b in enumerate(active):
            dr = drafts_list[r]
            am_r = am_list[r]
            mtp_proposed += len(dr)
            matched = 0
            stopped = False
            for j, d_tok in enumerate(dr):
                pred = _pick_unconstrained(b, am_r[j], r, j)
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
                bonus = _pick_unconstrained(b, am_r[len(dr)], r, len(dr))
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
        del am_mx, x_step

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
    "DEFAULT_QAT_MTP_DRAFTER_REPO",
    "MAX_MTP_BATCH_SIZE",
    "Gemma4MtpDrafter",
    "draft_scan_top_k",
    "recommended_mtp_draft_k",
    "resolve_mtp_drafter_dir",
    "run_decode_mtp",
    "select_mtp_drafter_repo",
    "should_use_mtp_drafter",
]
