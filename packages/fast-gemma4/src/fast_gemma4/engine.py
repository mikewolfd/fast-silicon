"""High-throughput Gemma-4 E2B/E4B inference engine (Metal 4 MPP GPU + CoreML ANE + llguidance + PromptLookup)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import gc
import json
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import warnings

import mlx.core as mx
import numpy as np

from .ane_pipeline import (
    ANE_CHUNK_S,
    ANE_P_FULL,
    ANE_P_SWA,
    AnePrefixRunner,
    AnePrefixWorker,
    AsyncHybridPrefillRing,
    can_use_ane_prefix,
    ingest_ane_finish_tail_and_last,
    is_ane_resident_bundle,
    is_w8_bundle_calibrated,
)
from .grammar import DEFAULT_MAX_WHITESPACE_RUN, GrammarHelper
from .model import FastGemma4Model
from .mtp_drafter import (
    DEFAULT_MTP_DRAFTER_REPO,
    DEFAULT_MTP_TOP_K,
    Gemma4MtpDrafter,
    _resolve_apple10_nax,
    resolve_mtp_drafter_dir,
    run_decode_mtp,
    should_use_mtp_drafter,
)
from .paths import Gemma4Paths, resolve_paths
from .speculative import (
    _host_ram_gib,
    _parse_use_ane_ring,
    run_decode_batch,
    run_decode_prompt_lookup,
    run_decode_rolling,
)
from .tokenizer import DEFAULT_KEPT_VOCAB_SIZE, FastGemma4Tokenizer


@dataclass
class GenerationItem:
    text: str
    parsed: Any | None
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    valid_json: bool
    valid_schema: bool
    finish_reason: str


@dataclass
class GenerationBatchResult:
    items: list[GenerationItem]
    stats: dict[str, Any] = field(default_factory=dict)


PromptInput = str | Sequence[dict[str, Any]]


def _normalize_schema_key(
    json_schema: str | dict[str, Any],
    whitespace_flexible: bool,
    max_whitespace_run: int | None = None,
) -> tuple[str, bool, int]:
    eff_ws = (
        0
        if not whitespace_flexible
        else (DEFAULT_MAX_WHITESPACE_RUN if max_whitespace_run is None else max(0, int(max_whitespace_run)))
    )
    if isinstance(json_schema, str):
        return json_schema, whitespace_flexible, eff_ws
    return json.dumps(json_schema, sort_keys=True, ensure_ascii=False), whitespace_flexible, eff_ws


def select_execution_plan(
    mode: str,
    ane_available: bool,
    lcp: int,
    max_suf: int,
    n_docs: int,
    batch_size: int | None,
    *,
    max_b_slots: int = 256,
) -> tuple[bool, int]:
    """Choose `(use_ane, wave_bs)`: auto prefers V4 rolling GPU (`<=max_b_slots`); explicit `hetero` enables ANE+GPU."""
    eff_mode = mode.lower()
    ane_ok = ane_available and can_use_ane_prefix(lcp, max_suf)
    use_ane = bool(eff_mode == "hetero" and ane_ok)
    cap_slots = max(1, int(max_b_slots))
    if batch_size is not None:
        wave_bs = int(batch_size)
    elif use_ane:
        wave_bs = min(64, cap_slots) if n_docs >= 128 else min(n_docs, cap_slots)
    else:
        wave_bs = cap_slots if n_docs > cap_slots else n_docs
    return use_ane, wave_bs


def should_use_speculative(
    speculative: bool | None,
    use_ane: bool,
    batch_w: int,
    temperature: float,
) -> bool:
    """Enable PromptLookup automatically only in low-B bandwidth regime (<=4) where L_q=4 beats L_q=1."""
    if temperature > 0.0:
        return False
    if speculative is not None:
        return bool(speculative)
    return (not use_ane) and batch_w <= 4


def _probe_host_hardware_safe() -> dict[str, Any] | None:
    """Probe host hardware via `fast_shared.hw_check.probe_host_hardware()`, returning None on failure."""
    try:
        import fast_shared.hw_check as _hw_check

        hw = _hw_check.probe_host_hardware()
        return dict(hw) if isinstance(hw, dict) else None
    except Exception:
        return None


def _extract_host_mem_gb(hw: dict[str, Any] | None, *, fallback_ram_gib: float = 0.0) -> float:
    """Extract host RAM in GiB/GB from probed hardware metadata with fallback to `_host_ram_gib()`."""
    if isinstance(hw, dict):
        if "mem_gb" in hw and hw["mem_gb"] is not None:
            try:
                return float(hw["mem_gb"])
            except (TypeError, ValueError):
                pass
        if "mem_bytes" in hw and hw["mem_bytes"] is not None:
            try:
                mem_b = float(hw["mem_bytes"])
                if mem_b > 0.0:
                    return mem_b / (1024.0**3)
            except (TypeError, ValueError):
                pass
    if fallback_ram_gib > 0.0:
        return float(fallback_ram_gib)
    return float(_host_ram_gib())


def _resolve_ane_bundle_safe(
    paths: Any,
    variant: str,
    *,
    model_dir: Any,
    local_files_only: bool,
    exit_layer: int,
) -> Path | None:
    if paths is None:
        return None
    try:
        return paths.resolve_ane_bundle(
            variant,
            model_dir=model_dir,
            local_files_only=local_files_only,
            exit_layer=exit_layer,
        )
    except TypeError:
        return paths.resolve_ane_bundle(
            variant,
            model_dir=model_dir,
            local_files_only=local_files_only,
        )


def _select_auto_ane_ring_variant(
    paths: Any,
    *,
    model_dir: Any,
    local_files_only: bool = True,
    exit_layer: int = 14,
    hw: dict[str, Any] | None = None,
    fallback_ram_gib: float = 0.0,
) -> tuple[str, Any | None]:
    """Resolve `ane_ring_variant='auto'` via `probe_host_hardware()` + `DynamicSiliconPolicy`.

    On `0 < host_mem_gb <= 24`, selects `"w8"` only when a staged `w8` ring bundle exists and
    `is_w8_bundle_calibrated(...)` is `True`; otherwise falls back cleanly to `"fp16"`.
    """
    hw_info = _probe_host_hardware_safe() if hw is None else dict(hw)
    host_mem_gb = _extract_host_mem_gb(hw_info, fallback_ram_gib=fallback_ram_gib)

    has_calibrated_w8 = False
    if 0.0 < host_mem_gb <= 24.0:
        w8_cand = _resolve_ane_bundle_safe(
            paths,
            "w8",
            model_dir=model_dir,
            local_files_only=local_files_only,
            exit_layer=exit_layer,
        )
        has_calibrated_w8 = bool(w8_cand is not None and is_w8_bundle_calibrated(w8_cand))

    variant = "w8" if (0.0 < host_mem_gb <= 24.0 and has_calibrated_w8) else "fp16"
    policy: Any | None = None
    if host_mem_gb > 0.0:
        try:
            from fast_shared.runtime_lessons import DynamicSiliconPolicy

            policy = DynamicSiliconPolicy(
                hw=hw_info if hw_info is not None else {},
                host_mem_gb=host_mem_gb,
                has_calibrated_ane_w8=has_calibrated_w8,
            )
            variant = str(policy.host.ane_ring_variant)
        except Exception:
            pass
    return variant, policy


class FastGemma4Engine:
    """Thread-safe hardware-limit Gemma-4 E2B/E4B engine for Apple Silicon (M4 / M5+)."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        model_dir: str | Path | None = None,
        default_mode: str = "auto",
        ane_variant: str = "fp16",
        ane_ring_variant: str = "auto",
        vocab_out: int | None = DEFAULT_KEPT_VOCAB_SIZE,
        lm_head_mode: str = "q4",
        mlp_dtype: str = "bfloat16",
        use_mpp: bool = True,
        kv_i8: bool = True,
        vec4_reduce: bool = True,
        mpp_m32_cutoff: int = 96,
        use_fused_softcap_argmax: bool = True,
        max_prefix_cache: int | None = None,
        max_b_slots: int | None = None,
        allow_ane_on_low_ram: bool = False,
        mtp_drafter: str | Path | bool | None = "auto",
        mtp_top_k: int = DEFAULT_MTP_TOP_K,
        mtp_draft_k: int | None = None,
        local_files_only: bool = True,
        warmup: bool = True,
        warmup_ane: bool = False,
    ) -> None:
        if default_mode not in ("auto", "gpu", "hetero"):
            raise ValueError(f"Unsupported default_mode={default_mode!r}")
        if ane_variant not in ("fp16", "w8", "auto"):
            raise ValueError(f"Unsupported ane_variant={ane_variant!r}")
        if ane_ring_variant not in ("fp16", "w8", "auto"):
            raise ValueError(f"Unsupported ane_ring_variant={ane_ring_variant!r}")

        self._host_ram_gib = _host_ram_gib()
        self._is_low_ram = 0.0 < self._host_ram_gib < 32.0
        self._allow_ane_on_low_ram = bool(allow_ane_on_low_ram)
        self.default_max_b_slots = int(max_b_slots) if max_b_slots is not None else (128 if self._is_low_ram else 256)
        eff_prefix_cache = (2 if self._is_low_ram else 8) if max_prefix_cache is None else int(max_prefix_cache)

        # Pin Metal wired working set cleanly (restored in close()) to eliminate macOS swap/page-fault jitter.
        self._prev_wired_limit: int | None = None
        set_wl = getattr(mx, "set_wired_limit", None)
        dev_info_fn = getattr(mx, "device_info", None)
        if callable(set_wl) and callable(dev_info_fn):
            try:
                rec_ws = int(dev_info_fn().get("max_recommended_working_set_size", 0))
                if rec_ws > 0:
                    self._prev_wired_limit = int(set_wl(rec_ws))
            except Exception:
                self._prev_wired_limit = None

        self.paths: Gemma4Paths = resolve_paths(home)
        self.model = FastGemma4Model(
            paths=self.paths,
            model_dir=model_dir,
            mlp_dtype=mlp_dtype,
            vocab_out=vocab_out,
            use_mpp=use_mpp,
            lm_head_mode=lm_head_mode,
            local_files_only=local_files_only,
            kv_i8=kv_i8,
            vec4_reduce=vec4_reduce,
            mpp_m32_cutoff=mpp_m32_cutoff,
            use_fused_softcap_argmax=use_fused_softcap_argmax,
        )
        self.tokenizer = FastGemma4Tokenizer(self.model.tokenizer_json)
        is_e4b = (
            getattr(self.model, "hidden_size", 1536) >= 2560
            or getattr(self.model, "num_layers", 35) >= 42
            or getattr(self.model, "n_kv_heads", 1) >= 2
        )
        self.model_name = "google/gemma-4-E4B-it" if is_e4b else "google/gemma-4-E2B-it"
        self.default_mode = default_mode
        self.ane_variant = ane_variant
        self.max_prefix_cache = max(1, eff_prefix_cache)

        self._gpu_lock = threading.Lock()
        self._prefix_cache: OrderedDict[tuple[int, ...], dict[str, Any]] = OrderedDict()
        self._grammar_cache: dict[tuple[str, bool, int], GrammarHelper] = {}
        self._ane_runner: AnePrefixRunner | None = None
        self._ane_ring_runner: AnePrefixRunner | None = None
        self._ane_ring_checked: bool = False
        self._ane_ring_bundle: Path | None = None
        self._w_emb: np.ndarray | None = None
        self._w_pli14: np.ndarray | None = None
        self._ane_workers: dict[tuple[int, ...], AnePrefixWorker] = {}
        self._ane_rings: dict[tuple[int, ...], AsyncHybridPrefillRing] = {}
        self._local_files_only = bool(local_files_only)

        exit_layer = getattr(self.model, "first_kv_shared", 15) - 1
        # Default hybrid prefill ring to dynamic "auto" (calibrated W8 on <=24 GB, FP16 on >24 GB or when uncalibrated/missing).
        self._ane_ring_variant_requested = ane_ring_variant
        self._host_hw: dict[str, Any] | None = None
        self.silicon_policy: Any | None = None
        if ane_ring_variant == "auto":
            self._host_hw = _probe_host_hardware_safe()
            self.ane_ring_variant, self.silicon_policy = _select_auto_ane_ring_variant(
                self.paths,
                model_dir=self.model.model_dir,
                local_files_only=local_files_only,
                exit_layer=exit_layer,
                hw=self._host_hw,
                fallback_ram_gib=self._host_ram_gib,
            )
        else:
            self.ane_ring_variant = ane_ring_variant

        eff_var = "fp16" if ane_variant == "auto" else ane_variant
        bundle = self.paths.resolve_ane_bundle(
            eff_var, model_dir=self.model.model_dir, local_files_only=local_files_only, exit_layer=exit_layer
        )
        if bundle is None and eff_var != "fp16":
            bundle = self.paths.resolve_ane_bundle(
                "fp16", model_dir=self.model.model_dir, local_files_only=local_files_only, exit_layer=exit_layer
            )
        if bundle is None and eff_var != "w8":
            bundle = self.paths.resolve_ane_bundle(
                "w8", model_dir=self.model.model_dir, local_files_only=local_files_only, exit_layer=exit_layer
            )
        self._ane_bundle = bundle
        low_ram_ane_disabled = self._is_low_ram and not self._allow_ane_on_low_ram and default_mode != "hetero"
        self.ane_available = bool(
            (not low_ram_ane_disabled) and bundle is not None and self.paths.lib_ane_prefix.exists()
        )

        self.mtp_drafter: Gemma4MtpDrafter | None = None
        self.mtp_draft_k: int | None = None if mtp_draft_k is None else int(mtp_draft_k)
        self._mtp_top_k: int = int(mtp_top_k)
        self._mtp_drafter_auto: bool = False
        self._mtp_drafter_auto_attempted: bool = False
        if isinstance(mtp_drafter, str) and mtp_drafter.strip().lower() == "auto":
            self._mtp_drafter_auto = True
        elif mtp_drafter:
            drafter_ref: str | Path = DEFAULT_MTP_DRAFTER_REPO if mtp_drafter is True else mtp_drafter
            self.mtp_drafter = Gemma4MtpDrafter(
                self.model,
                drafter_ref,
                top_k=self._mtp_top_k,
                local_files_only=local_files_only,
            )

        if warmup:
            self.warmup(enable_ane=warmup_ane)

    def release_ane_resources(self) -> None:
        """Close native ANE runners and free the ~2.68 GiB host FP16 ANE embedding/PLE tables."""
        if getattr(self, "_ane_runner", None) is not None:
            self._ane_runner.close()  # type: ignore[union-attr]
            self._ane_runner = None
        if getattr(self, "_ane_ring_runner", None) is not None:
            self._ane_ring_runner.close()  # type: ignore[union-attr]
            self._ane_ring_runner = None
        model = getattr(self, "model", None)
        if model is not None:
            for attr in ("_auto_ane_w8_runner", "_auto_ane_runner"):
                auto_r = getattr(model, attr, None)
                if auto_r is not None:
                    auto_r.close()
                    setattr(model, attr, None)
            if getattr(model, "_auto_cpu_ane_tables", None) is not None:
                model._auto_cpu_ane_tables = None  # type: ignore[attr-defined]
        if hasattr(self, "_ane_workers"):
            self._ane_workers.clear()
        if hasattr(self, "_ane_rings"):
            self._ane_rings.clear()
        self._w_emb = None
        self._w_pli14 = None
        gc.collect()

    def close(self) -> None:
        self.mtp_drafter = None
        self._mtp_drafter_auto_attempted = False
        self.release_ane_resources()
        prev_wl = getattr(self, "_prev_wired_limit", None)
        if prev_wl is not None:
            set_wl = getattr(mx, "set_wired_limit", None)
            if callable(set_wl):
                try:
                    set_wl(prev_wl)
                except Exception:
                    pass
            self._prev_wired_limit = None

    def _ensure_ane_ready(self) -> tuple[AnePrefixRunner, np.ndarray, np.ndarray]:
        if not self.ane_available or self._ane_bundle is None:
            raise RuntimeError("ANE Early14 bundle or libgemma4_ane_prefix.dylib is unavailable.")
        if self._ane_runner is None:
            self._ane_runner = AnePrefixRunner(self.paths.lib_ane_prefix, self._ane_bundle)
        if self._w_emb is None or self._w_pli14 is None:
            self._w_emb, self._w_pli14 = self.model.build_cpu_ane_embed_tables()
        return self._ane_runner, self._w_emb, self._w_pli14

    def warmup(self, *, enable_ane: bool = False) -> None:
        with self._gpu_lock:
            toks = [self.tokenizer.bos_token_id, 14937, 4818, 12375]
            pkv = self.model.prefill_prefix(toks[:2])
            cache = self.model.make_batch_cache(pkv, batch_size=1, doc_cap_full=64, swa_ring_cap=512)
            lg, ob = self.model.prefill_suffix_batch([toks[2:]], cache, group_size=1)
            mx.eval(lg, ob)
            del cache
            sc = self.model.make_slot_cache(pkv, batch_size=2, doc_cap_full=64, swa_ring_cap=512)
            xs, am, _ = self.model.prefill_slots_lazy_lg([toks[2:], toks[2:]], [0, 1], sc, group_size=2)
            mx.eval(xs, am)
            if self.model.kv_i8:
                self.model.forward_early14_slot(
                    mx.array([[toks[2]]], dtype=mx.int32),
                    sc,
                    mx.array([2], dtype=mx.int32),
                    mx.array([0], dtype=mx.int32),
                )
            del sc
            if enable_ane and self.ane_available:
                self._ensure_ane_ready()

    def get_grammar(
        self,
        json_schema: str | dict[str, Any],
        *,
        whitespace_flexible: bool = False,
        max_whitespace_run: int | None = None,
    ) -> GrammarHelper:
        key = _normalize_schema_key(json_schema, whitespace_flexible, max_whitespace_run)
        gh = self._grammar_cache.get(key)
        if gh is None:
            gh = GrammarHelper(
                self.tokenizer,
                json_schema,
                self.model.new_to_old,
                whitespace_flexible=whitespace_flexible,
                max_whitespace_run=max_whitespace_run,
            )
            self._grammar_cache[key] = gh
        return gh

    def get_prefix_kv(self, sys_ids: Sequence[int]) -> dict[str, Any]:
        key = tuple(int(x) for x in sys_ids)
        hit = self._prefix_cache.get(key)
        if hit is not None:
            self._prefix_cache.move_to_end(key)
            return hit
        pkv = self.model.prefill_prefix(key)
        self._prefix_cache[key] = pkv
        while len(self._prefix_cache) > self.max_prefix_cache:
            old_key, _ = self._prefix_cache.popitem(last=False)
            self._ane_workers.pop(old_key, None)
            if hasattr(self, "_ane_rings"):
                self._ane_rings.pop(old_key, None)
        return pkv

    def _get_ane_worker(self, sys_ids: Sequence[int], pkv: dict[str, Any]) -> AnePrefixWorker:
        runner, w_emb, w_pli14 = self._ensure_ane_ready()
        key = tuple(int(x) for x in sys_ids)
        worker = self._ane_workers.get(key)
        if worker is None:
            worker = AnePrefixWorker(self.model, runner, pkv, w_emb, w_pli14)
            self._ane_workers[key] = worker
        return worker

    @property
    def ane_ring_bundle(self) -> Path | None:
        """Bundle the hybrid prefill ring runs on; None until a ring engaged, and when it fell back to GPU only."""
        return getattr(self, "_ane_ring_bundle", None)

    def _effective_ane_ring_variant(self) -> str:
        """Return concrete `'fp16'` or `'w8'` ring variant, resolving `'auto'` dynamically if needed."""
        variant = getattr(self, "ane_ring_variant", "fp16")
        if variant != "auto":
            return variant
        paths = getattr(self, "paths", None)
        local_only = bool(getattr(self, "_local_files_only", True))
        exit_layer = getattr(self.model, "first_kv_shared", 15) - 1
        resolved, policy = _select_auto_ane_ring_variant(
            paths,
            model_dir=getattr(self.model, "model_dir", None),
            local_files_only=local_only,
            exit_layer=exit_layer,
            hw=getattr(self, "_host_hw", None),
            fallback_ram_gib=float(getattr(self, "_host_ram_gib", 0.0)),
        )
        self._ane_ring_variant_requested = "auto"
        self.ane_ring_variant = resolved
        if policy is not None:
            self.silicon_policy = policy
        return resolved

    def _engine_apple10_nax(self) -> bool | None:
        pol = getattr(self, "silicon_policy", None)
        hw = getattr(self, "_host_hw", None)
        has_chip_hw = isinstance(hw, dict) and any(
            k in hw for k in ("supports_apple10", "gpu_family", "brand", "mlx_architecture")
        )
        if pol is not None and has_chip_hw:
            return bool(pol.host.apple10_nax)
        if has_chip_hw:
            try:
                from fast_shared.hw_check import supports_apple10_nax

                probe_dict = dict(hw)  # type: ignore[arg-type]
                probe_dict.setdefault("system", "Darwin")
                probe_dict.setdefault("machine", "arm64")
                return bool(supports_apple10_nax(probe_dict))
            except Exception:
                pass
        return None

    def _has_mtp_drafter_candidate(self) -> bool:
        return getattr(self, "mtp_drafter", None) is not None or (
            bool(getattr(self, "_mtp_drafter_auto", False))
            and not bool(getattr(self, "_mtp_drafter_auto_attempted", False))
        )

    def _maybe_ensure_mtp_drafter(
        self,
        *,
        use_ane: bool,
        batch_w: int,
        temperature: float,
        speculative: bool | None = None,
    ) -> Gemma4MtpDrafter | None:
        """Return active MTP drafter for this wave, lazy-loading once under `mtp_drafter='auto'`."""
        apple10_nax = self._engine_apple10_nax()
        existing = getattr(self, "mtp_drafter", None)
        if existing is not None:
            return (
                existing
                if should_use_mtp_drafter(
                    existing,
                    use_ane,
                    batch_w,
                    temperature,
                    speculative=speculative,
                    apple10_nax=apple10_nax,
                )
                else None
            )
        if not getattr(self, "_mtp_drafter_auto", False) or getattr(self, "_mtp_drafter_auto_attempted", False):
            return None
        if not should_use_mtp_drafter(
            True,
            use_ane,
            batch_w,
            temperature,
            speculative=speculative,
            apple10_nax=apple10_nax,
        ):
            return None

        self._mtp_drafter_auto_attempted = True
        if (
            int(getattr(self.model, "hidden_size", 1536)) != 1536
            or int(getattr(self.model, "n_kv_heads", 1)) != 1
        ):
            return None
        try:
            drafter_dir = resolve_mtp_drafter_dir(DEFAULT_MTP_DRAFTER_REPO, local_files_only=True)
        except Exception:
            return None
        if drafter_dir is None:
            return None
        try:
            self.mtp_drafter = Gemma4MtpDrafter(
                self.model,
                drafter_dir,
                top_k=int(getattr(self, "_mtp_top_k", DEFAULT_MTP_TOP_K)),
                local_files_only=True,
            )
        except Exception:
            self.mtp_drafter = None
            return None
        return self.mtp_drafter

    def _resolve_ane_ring_runner(self, *, require: bool) -> AnePrefixRunner | None:
        """Runner for the `ane_ring_variant` bundle, or None when that bundle is missing or not fully on the ANE."""
        auto_mode = (
            getattr(self, "_ane_ring_variant_requested", None) == "auto"
            or getattr(self, "ane_ring_variant", "fp16") == "auto"
        )
        variant = self._effective_ane_ring_variant()
        if not getattr(self, "_ane_ring_checked", False):
            paths = getattr(self, "paths", None)
            local_only = bool(getattr(self, "_local_files_only", True))
            exit_layer = getattr(self.model, "first_kv_shared", 15) - 1
            cand = _resolve_ane_bundle_safe(
                paths,
                variant,
                model_dir=self.model.model_dir,
                local_files_only=local_only,
                exit_layer=exit_layer,
            )
            resident_cand = (
                cand if (paths is not None and is_ane_resident_bundle(paths.lib_ane_prefix, cand)) else None
            )
            if resident_cand is None and auto_mode and variant == "w8":
                fp16_cand = _resolve_ane_bundle_safe(
                    paths,
                    "fp16",
                    model_dir=self.model.model_dir,
                    local_files_only=local_only,
                    exit_layer=exit_layer,
                )
                resident_cand = (
                    fp16_cand
                    if (paths is not None and is_ane_resident_bundle(paths.lib_ane_prefix, fp16_cand))
                    else None
                )
                if resident_cand is not None:
                    variant = "fp16"
                    self.ane_ring_variant = "fp16"
            self._ane_ring_bundle = resident_cand
            self._ane_ring_checked = True
        if self._ane_ring_bundle is None:
            if require:
                raise RuntimeError(
                    f"No valid {variant} ANE ring bundle (`0` CPU runtime ops) available for {self.model.model_dir}."
                )
            return None
        if variant == "w8" and not is_w8_bundle_calibrated(self._ane_ring_bundle):
            warnings.warn(
                f"ANE ring bundle {self._ane_ring_bundle.name} is Linear-W8 without a domain-corpus/QAT "
                "calibration manifest; use ane_ring_variant='fp16' or rebuild with `--awq-mse --calib-file ...` "
                "to avoid repeat-loop regressions.",
                UserWarning,
                stacklevel=2,
            )
        if getattr(self, "_ane_ring_runner", None) is None:
            self._ane_ring_runner = AnePrefixRunner(
                self.paths.lib_ane_prefix, self._ane_ring_bundle, pipe_depth=4, out_ring_slots=96
            )
        return self._ane_ring_runner

    def _maybe_get_ane_ring(
        self,
        sys_ids: Sequence[int],
        pkv: dict[str, Any],
        suf_ids_list: Sequence[Sequence[int]],
        *,
        use_ane_ring: bool | str,
        kv_i8: bool,
        ane_cohort_size: int = 16,
        mode: str = "auto",
    ) -> AsyncHybridPrefillRing | None:
        want_ane, require_ane = _parse_use_ane_ring(use_ane_ring)
        if not want_ane:
            return None
        if not require_ane:
            if str(mode).strip().lower() == "gpu":
                return None
            if _resolve_apple10_nax(self._engine_apple10_nax()):
                return None
            if getattr(self, "_is_low_ram", False) and not getattr(self, "_allow_ane_on_low_ram", False):
                return None
        if not kv_i8:
            if require_ane:
                raise ValueError("use_ane_ring=True requires kv_i8=True.")
            return None
        l_sys = int(pkv["l_sys"])
        if l_sys > ANE_P_FULL - ANE_CHUNK_S:
            if require_ane:
                raise ValueError(
                    f"Prefix length l_sys={l_sys} exceeds ANE Full geometry (max {ANE_P_FULL - ANE_CHUNK_S} for P_FULL={ANE_P_FULL})."
                )
            return None
        min_auto_batch = max(16, int(ane_cohort_size))
        max_suf = max((len(s) for s in suf_ids_list), default=0)
        if not require_ane and (len(suf_ids_list) < min_auto_batch or max_suf < 64):
            return None

        runner = self._resolve_ane_ring_runner(require=require_ane)
        if runner is None:
            return None
        if getattr(self, "_w_emb", None) is None or getattr(self, "_w_pli14", None) is None:
            self._w_emb, self._w_pli14 = self.model.build_cpu_ane_embed_tables()
        if not hasattr(self, "_ane_rings"):
            self._ane_rings = {}
        key = tuple(int(x) for x in sys_ids)
        ring = self._ane_rings.get(key)
        if ring is None:
            ring = AsyncHybridPrefillRing(self.model, runner, pkv, self._w_emb, self._w_pli14)
            self._ane_rings[key] = ring
        return ring

    def encode_prompts(
        self,
        prompts: Sequence[PromptInput],
        *,
        system_prompt: str | None = None,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
    ) -> list[list[int]]:
        raw_texts: list[str] = []
        for item in prompts:
            if isinstance(item, str):
                if use_chat_template:
                    content = f"{system_prompt}\n\n{item}" if system_prompt else item
                    raw_texts.append(
                        self.tokenizer.format_chat(
                            [{"role": "user", "content": content}],
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    )
                else:
                    raw_texts.append(f"{system_prompt}\n\n{item}" if system_prompt else item)
            else:
                msgs = list(item)
                if system_prompt and msgs and msgs[0].get("role") not in ("system", "developer"):
                    msgs = [{"role": "system", "content": system_prompt}] + msgs
                raw_texts.append(
                    self.tokenizer.format_chat(
                        msgs,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                )
        return self.tokenizer.encode_batch(raw_texts, add_special_tokens=False)

    def generate(
        self,
        prompt: PromptInput,
        *,
        system_prompt: str | None = None,
        json_schema: str | dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        mode: str | None = None,
        speculative: bool | None = None,
        use_chat_template: bool = True,
        whitespace_flexible: bool = True,
        max_whitespace_run: int | None = None,
    ) -> GenerationItem:
        res = self.generate_batch(
            [prompt],
            system_prompt=system_prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            mode=mode,
            speculative=speculative,
            use_chat_template=use_chat_template,
            whitespace_flexible=whitespace_flexible,
            max_whitespace_run=max_whitespace_run,
        )
        return res.items[0]

    def run_rolling(
        self,
        *,
        sys_ids: Sequence[int],
        suf_ids_list: Sequence[Sequence[int]],
        max_tokens: int = 512,
        B_slots: int = 256,
        refill_min: int | None = None,
        doc_cap_full: int = 0,
        swa_ring_cap: int = 512,
        group_size: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        gh: Any = None,
        per_doc_grammars: Sequence[Any] | None = None,
        sort_by_length: bool = True,
        kv_i8: bool | None = None,
        use_ane_ring: bool | str = False,
        ane_cohort_size: int = 16,
        enable_ff_jump: bool = True,
        override_memory_cap: bool = False,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        """Thread-safe GPU (+ optional Hybrid ANE) rolling decode entrypoint."""
        with self._gpu_lock:
            pkv = self.get_prefix_kv(sys_ids)
            eff_kv_i8 = self.model.kv_i8 if kv_i8 is None else bool(kv_i8)
            eff_b_slots = (
                int(B_slots)
                if override_memory_cap
                else min(int(B_slots), int(getattr(self, "default_max_b_slots", 256)))
            )
            ring = self._maybe_get_ane_ring(
                sys_ids,
                pkv,
                suf_ids_list,
                use_ane_ring=use_ane_ring,
                kv_i8=eff_kv_i8,
                ane_cohort_size=ane_cohort_size,
            )
            return run_decode_rolling(
                self.model,
                self.tokenizer.stop_token_ids,
                gh,
                pkv,
                suf_ids_list,
                max_tokens=max_tokens,
                b_slots=eff_b_slots,
                refill_min=refill_min,
                swa_ring_cap=swa_ring_cap,
                group_size=group_size,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                sort_by_length=sort_by_length,
                kv_i8=eff_kv_i8,
                doc_cap_full=doc_cap_full,
                per_doc_grammars=per_doc_grammars,
                use_ane_ring=use_ane_ring if ring is not None else False,
                ane_ring=ring,
                ane_cohort_size=ane_cohort_size,
                enable_ff_jump=enable_ff_jump,
                ane_ring_variant=self._effective_ane_ring_variant(),
            )

    def generate_batch(
        self,
        prompts: Sequence[PromptInput],
        *,
        system_prompt: str | None = None,
        json_schema: str | dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        mode: str | None = None,
        speculative: bool | None = None,
        batch_size: int | None = None,
        compact: bool = True,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        whitespace_flexible: bool = False,
        max_whitespace_run: int | None = None,
        per_doc_grammars: Sequence[str | dict[str, Any] | Any | None] | None = None,
        use_ane_ring: bool | str = "auto",
        ane_cohort_size: int = 16,
        enable_ff_jump: bool = True,
    ) -> GenerationBatchResult:
        if not prompts:
            return GenerationBatchResult(items=[], stats={"docs_per_sec": 0.0, "decode_tok_s": 0.0})

        if per_doc_grammars is not None and len(per_doc_grammars) != len(prompts):
            raise ValueError(f"Mismatched per_doc_grammars={len(per_doc_grammars)} vs prompts={len(prompts)}")

        eff_mode = (mode or self.default_mode).lower()
        if eff_mode not in ("auto", "gpu", "hetero"):
            raise ValueError(f"Unsupported mode={eff_mode!r}")

        with self._gpu_lock:
            mx.reset_peak_memory()
            t_all0 = time.perf_counter()

            prompt_ids_list = self.encode_prompts(
                prompts,
                system_prompt=system_prompt,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            grammar = (
                self.get_grammar(
                    json_schema,
                    whitespace_flexible=whitespace_flexible,
                    max_whitespace_run=max_whitespace_run,
                )
                if (json_schema is not None and per_doc_grammars is None)
                else None
            )
            resolved_grammars: list[Any] | None = None
            if per_doc_grammars is not None:
                resolved_grammars = []
                for g_spec in per_doc_grammars:
                    if g_spec is None:
                        resolved_grammars.append(None)
                    elif isinstance(g_spec, (str, dict)):
                        resolved_grammars.append(
                            self.get_grammar(
                                g_spec,
                                whitespace_flexible=whitespace_flexible,
                                max_whitespace_run=max_whitespace_run,
                            )
                        )
                    else:
                        resolved_grammars.append(g_spec)

            def _upfront_for(i_doc: int) -> list[int]:
                if resolved_grammars is not None:
                    gd = resolved_grammars[i_doc]
                    if gd is None:
                        return []
                    up_fn = getattr(gd, "upfront_ff_for_doc", None)
                    return list(up_fn(i_doc)) if callable(up_fn) else list(getattr(gd, "upfront_ff", ()))
                return list(grammar.upfront_ff) if grammar is not None else []

            upfront_by_doc = [_upfront_for(i) for i in range(len(prompts))]
            bos_id = int(self.tokenizer.bos_token_id)
            prompt_ids_list = [x if (x or upfront_by_doc[i]) else [bos_id] for i, x in enumerate(prompt_ids_list)]

            # Longest shared prompt prefix across batch (`KV_sys` reuse + ANE chunking)
            lcp = min(len(x) for x in prompt_ids_list)
            for k in range(lcp):
                t0_k = prompt_ids_list[0][k]
                if any(x[k] != t0_k for x in prompt_ids_list[1:]):
                    lcp = k
                    break
            # Always keep >= 1 token per suffix so `LastTok` has an input token
            min_raw_len = min(len(x) for x in prompt_ids_list)
            has_empty_upfront = any(not u for u in upfront_by_doc)
            if has_empty_upfront and lcp >= min_raw_len:
                lcp = max(0, min_raw_len - 1)

            sys_ids = prompt_ids_list[0][:lcp]
            pkv = self.get_prefix_kv(sys_ids)
            full_prompts = [x + upfront_by_doc[i] for i, x in enumerate(prompt_ids_list)]
            suf_all = [x[lcp:] + upfront_by_doc[i] for i, x in enumerate(prompt_ids_list)]

            n_docs = len(prompts)
            max_suf = max(len(s) for s in suf_all)
            ring_for_batch = self._maybe_get_ane_ring(
                sys_ids,
                pkv,
                suf_all,
                use_ane_ring=use_ane_ring,
                kv_i8=bool(self.model.kv_i8),
                ane_cohort_size=ane_cohort_size,
                mode=eff_mode,
            )
            use_ane, wave_bs = select_execution_plan(
                eff_mode,
                self.ane_available,
                lcp,
                max_suf,
                n_docs,
                batch_size,
                max_b_slots=int(getattr(self, "default_max_b_slots", 256)),
            )
            if ring_for_batch is not None:
                use_ane = False
            waves = [list(range(i0, min(n_docs, i0 + wave_bs))) for i0 in range(0, n_docs, wave_bs)]

            all_gen_toks: list[list[int]] = [[] for _ in range(n_docs)]
            tot_pre_sec = 0.0
            tot_dec_sec = 0.0
            tot_out_toks = 0
            tot_ff_toks = 0
            any_spec_used = False
            rolling_used = False
            used_ane_ring_flag = False
            roll_valid_schemas: list[bool] | None = None
            extra_roll_stats: dict[str, Any] = {}

            def _decode_wave(w_indices: list[int], lg0: mx.array, ob0: mx.array, cache: dict[str, Any]) -> dict[str, Any]:
                nonlocal any_spec_used
                Bw = len(w_indices)
                mtp_drafter = self._maybe_ensure_mtp_drafter(
                    use_ane=use_ane,
                    batch_w=Bw,
                    temperature=temperature,
                    speculative=speculative,
                )
                if mtp_drafter is not None:
                    any_spec_used = True
                    return run_decode_mtp(
                        self.model,
                        mtp_drafter,
                        self.tokenizer.stop_token_ids,
                        grammar,
                        lg0,
                        ob0,
                        cache,
                        Bw,
                        max_tokens=max_tokens,
                        draft_k=getattr(self, "mtp_draft_k", None),
                        enable_ff_jump=enable_ff_jump,
                    )
                use_spec = should_use_speculative(speculative, use_ane, Bw, temperature)
                if use_spec:
                    any_spec_used = True
                    w_prompts = [full_prompts[idx] for idx in w_indices]
                    return run_decode_prompt_lookup(
                        self.model,
                        self.tokenizer.stop_token_ids,
                        grammar,
                        lg0,
                        ob0,
                        cache,
                        w_prompts,
                        Bw,
                        max_tokens=max_tokens,
                        enable_ff_jump=enable_ff_jump,
                    )
                dec_res = run_decode_batch(
                    self.model,
                    self.tokenizer.stop_token_ids,
                    grammar,
                    lg0,
                    ob0,
                    cache,
                    Bw,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    compact=compact,
                    tail_handoff_b=0 if speculative is False else 4,
                    doc_prompts=[suf_all[idx] for idx in w_indices],
                    enable_ff_jump=enable_ff_jump,
                )
                if dec_res.get("used_speculative"):
                    any_spec_used = True
                return dec_res

            if use_ane and resolved_grammars is None and ring_for_batch is None:
                ane_worker = self._get_ane_worker(sys_ids, pkv)
                ane_cur = ane_worker.run_wave_cpu_ane([suf_all[i] for i in waves[0]])
                for w_idx, w_docs in enumerate(waves):
                    Bw = len(w_docs)
                    sufs_w = [suf_all[i] for i in w_docs]
                    cap_full = max(max(len(s) for s in sufs_w) + max_tokens + 64, 128)
                    wave_swa_cap = 512 if (speculative is False or cap_full <= self.model.window_size) else 1024
                    cache = self.model.make_batch_cache(
                        pkv, batch_size=Bw, doc_cap_full=cap_full, swa_ring_cap=wave_swa_cap
                    )

                    tp0 = time.perf_counter()
                    lg0, ob0, _, _, _ = ingest_ane_finish_tail_and_last(self.model, cache, sufs_w, ane_cur, lcp)
                    tot_pre_sec += time.perf_counter() - tp0

                    # Overlap next wave's CPU+ANE prefill concurrently while GPU decodes current wave!
                    next_box: dict[str, Any] = {}
                    next_err: list[BaseException] = []
                    t_bg: threading.Thread | None = None
                    if w_idx + 1 < len(waves):
                        nxt_docs = waves[w_idx + 1]
                        nxt_sufs = [suf_all[i] for i in nxt_docs]

                        def _run_next() -> None:
                            try:
                                next_box.update(ane_worker.run_wave_cpu_ane(nxt_sufs))
                            except BaseException as exc:
                                next_err.append(exc)

                        t_bg = threading.Thread(target=_run_next, daemon=True)
                        t_bg.start()

                    dec_res = _decode_wave(w_docs, lg0, ob0, cache)
                    tot_dec_sec += dec_res["decode_sec"]
                    tot_out_toks += dec_res["tot_out_tok"]
                    tot_ff_toks += dec_res["ff_tokens"]
                    for r, doc_i in enumerate(w_docs):
                        all_gen_toks[doc_i] = dec_res["gen_toks"][r]

                    del cache
                    gc.collect()
                    mx.clear_cache()

                    if t_bg is not None:
                        t_wait0 = time.perf_counter()
                        t_bg.join()
                        tot_pre_sec += time.perf_counter() - t_wait0
                        if next_err:
                            raise next_err[0]
                        ane_cur = next_box
            elif (
                ring_for_batch is None
                and resolved_grammars is None
                and (
                    should_use_mtp_drafter(
                        True if self._has_mtp_drafter_candidate() else None,
                        False,
                        min(n_docs, wave_bs),
                        temperature,
                        speculative=speculative,
                        apple10_nax=self._engine_apple10_nax(),
                    )
                    or should_use_speculative(speculative, False, min(n_docs, wave_bs), temperature)
                )
            ):
                for w_docs in waves:
                    Bw = len(w_docs)
                    sufs_w = [suf_all[i] for i in w_docs]
                    cap_full = max(max(len(s) for s in sufs_w) + max_tokens + 64, 128)
                    wave_swa_cap = 512 if cap_full <= self.model.window_size else 1024
                    cache = self.model.make_batch_cache(
                        pkv, batch_size=Bw, doc_cap_full=cap_full, swa_ring_cap=wave_swa_cap
                    )
                    tp0 = time.perf_counter()
                    lg0, ob0 = self.model.prefill_suffix_batch(
                        sufs_w,
                        cache,
                        group_size=8 if Bw <= 32 else 16,
                        apply_softcap=True,
                    )
                    tot_pre_sec += time.perf_counter() - tp0
                    dec_res = _decode_wave(w_docs, lg0, ob0, cache)
                    tot_dec_sec += dec_res["decode_sec"]
                    tot_out_toks += dec_res["tot_out_tok"]
                    tot_ff_toks += dec_res["ff_tokens"]
                    if "ff_jump_steps" in dec_res:
                        extra_roll_stats["ff_jump_steps"] = extra_roll_stats.get("ff_jump_steps", 0) + int(
                            dec_res.get("ff_jump_steps", 0)
                        )
                        extra_roll_stats["ff_jump_tokens"] = extra_roll_stats.get("ff_jump_tokens", 0) + int(
                            dec_res.get("ff_jump_tokens", 0)
                        )
                    if dec_res.get("speculative_backend") == "mtp":
                        extra_roll_stats["speculative_backend"] = "mtp"
                        extra_roll_stats["mtp_rounds"] = extra_roll_stats.get("mtp_rounds", 0) + int(
                            dec_res.get("mtp_rounds", 0)
                        )
                        extra_roll_stats["mtp_accepted"] = extra_roll_stats.get("mtp_accepted", 0) + int(
                            dec_res.get("mtp_accepted", 0)
                        )
                        extra_roll_stats["mtp_proposed"] = extra_roll_stats.get("mtp_proposed", 0) + int(
                            dec_res.get("mtp_proposed", 0)
                        )
                        extra_roll_stats["mtp_accept_rate"] = extra_roll_stats["mtp_accepted"] / max(
                            extra_roll_stats["mtp_proposed"], 1
                        )
                    for r, doc_i in enumerate(w_docs):
                        all_gen_toks[doc_i] = dec_res["gen_toks"][r]
                    del cache
                    gc.collect()
                    mx.clear_cache()
            else:
                rolling_used = True
                roll_slots = max(1, wave_bs)
                roll_swa_cap = (
                    1024
                    if (
                        enable_ff_jump
                        and n_docs <= 4
                        and (max_suf + max_tokens + 64) > self.model.window_size
                    )
                    else 512
                )
                roll_res = run_decode_rolling(
                    self.model,
                    self.tokenizer.stop_token_ids,
                    grammar,
                    pkv,
                    suf_all,
                    max_tokens=max_tokens,
                    b_slots=roll_slots,
                    swa_ring_cap=roll_swa_cap,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    kv_i8=self.model.kv_i8,
                    per_doc_grammars=resolved_grammars,
                    use_ane_ring=use_ane_ring if ring_for_batch is not None else False,
                    ane_ring=ring_for_batch,
                    ane_cohort_size=ane_cohort_size,
                    enable_ff_jump=enable_ff_jump,
                    ane_ring_variant=self._effective_ane_ring_variant(),
                )
                all_gen_toks = roll_res["gen_toks"]
                roll_valid_schemas = roll_res.get("valid_schemas")
                tot_pre_sec = roll_res["prefill_sec"]
                tot_dec_sec = roll_res["decode_sec"]
                tot_out_toks = roll_res["tot_out_tok"]
                tot_ff_toks = roll_res["ff_tokens"]
                used_ane_ring_flag = bool(roll_res.get("used_ane_ring", False))
                extra_roll_stats = {
                    "llm_steps": roll_res["llm_steps"],
                    "ff_early14_steps": roll_res["ff_early14_steps"],
                    "ff_jump_steps": int(roll_res.get("ff_jump_steps", 0)),
                    "ff_jump_tokens": int(roll_res.get("ff_jump_tokens", 0)),
                    "mask_fallbacks": roll_res["mask_fallbacks"],
                    "used_ane_ring": used_ane_ring_flag,
                    "ane_cohorts": int(roll_res.get("ane_cohorts", 0)),
                    "ane_busy_sec": float(roll_res.get("ane_busy_sec", 0.0)),
                    "ane_nan_fallbacks": int(roll_res.get("ane_nan_fallbacks", 0)),
                    "gpu_boot_cohorts": int(roll_res.get("gpu_boot_cohorts", 0)),
                }
                if roll_res.get("used_speculative"):
                    any_spec_used = True
                gc.collect()
                if n_docs >= 8:
                    mx.clear_cache()

            items: list[GenerationItem] = []
            valid_json_cnt = 0
            valid_schema_cnt = 0
            for idx in range(n_docs):
                g_toks = all_gen_toks[idx]
                doc_g = resolved_grammars[idx] if resolved_grammars is not None else grammar
                full_out_ids = upfront_by_doc[idx] + g_toks
                txt = self.tokenizer.decode(full_out_ids, skip_special_tokens=True)
                parsed = None
                is_json = False
                is_schema = doc_g is None
                try:
                    parsed = json.loads(txt)
                    is_json = True
                    valid_json_cnt += 1
                except Exception:
                    pass
                if doc_g is not None:
                    if roll_valid_schemas is not None:
                        is_schema = bool(roll_valid_schemas[idx])
                    else:
                        is_schema = bool(doc_g.validate_token_sequence(full_out_ids))
                if is_schema:
                    valid_schema_cnt += 1
                finish_reason = "length" if len(g_toks) >= max_tokens and not (doc_g is not None and is_schema) else "stop"
                items.append(
                    GenerationItem(
                        text=txt,
                        parsed=parsed,
                        token_ids=full_out_ids,
                        prompt_tokens=len(prompt_ids_list[idx]),
                        completion_tokens=len(full_out_ids),
                        valid_json=is_json,
                        valid_schema=is_schema,
                        finish_reason=finish_reason,
                    )
                )

            elapsed = time.perf_counter() - t_all0
            prompt_tok_tot = sum(len(x) for x in prompt_ids_list)
            peak_gb = mx.get_peak_memory() / (1024**3)
            stats = {
                "mode_requested": eff_mode,
                "mode_used": "hetero_ane_gpu" if (use_ane or used_ane_ring_flag) else "gpu",
                "rolling": rolling_used,
                "kv_i8": self.model.kv_i8 if rolling_used else False,
                "speculative": any_spec_used,
                "docs": n_docs,
                "waves": 1 if rolling_used else len(waves),
                "lcp": lcp,
                "prompt_tokens": prompt_tok_tot,
                "completion_tokens": tot_out_toks,
                "ff_tokens": tot_ff_toks,
                "prefill_sec": tot_pre_sec,
                "decode_sec": tot_dec_sec,
                "elapsed_sec": elapsed,
                "decode_tok_s": tot_out_toks / max(tot_dec_sec, 1e-9),
                "total_tok_s": (prompt_tok_tot + tot_out_toks) / max(elapsed, 1e-9),
                "docs_per_sec": n_docs / max(elapsed, 1e-9),
                "valid_json_count": valid_json_cnt,
                "valid_schema_count": valid_schema_cnt,
                "peak_gb": peak_gb,
                **extra_roll_stats,
            }
            if n_docs >= 8 and mx.get_cache_memory() > (256 << 20):
                mx.clear_cache()
            return GenerationBatchResult(items=items, stats=stats)
