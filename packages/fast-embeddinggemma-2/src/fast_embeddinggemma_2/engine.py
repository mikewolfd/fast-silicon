"""Unified M4 Pro EmbeddingGemma-2 engine (Metal MPP GPU + 16-core ANE + CPU SME)."""

from __future__ import annotations

from collections import deque
import ctypes
from pathlib import Path
import threading
import time
from typing import Sequence

import mlx.core as mx
import numpy as np
from fast_shared.embeddings import resolve_embedding_dim, truncate_and_normalize

from .gpu import (
    ANE_MS_COST,
    EMBED_DIM,
    GPU_B_OPTS,
    MAX_SEQ_LEN,
    OVER_B_OPTS,
    FastEmbeddingGemma2GPU,
    build_over_chunks,
    fast_pack_cost_aware,
)
from .paths import require_runtime_paths
from .tokenizer import FastEmbeddingGemma2Tokenizer

_ANE_BUNDLES: tuple[tuple[int, str, int], ...] = (
    (32, "emb2_pk_s32_w8.mlmodelc", 1),
    (64, "emb2_pk_s64_w8.mlmodelc", 4),
    (128, "emb2_pk_s128_w8.mlmodelc", 4),
    (192, "emb2_pk_s192_w8.mlmodelc", 4),
    (256, "emb2_pk_s256_w8.mlmodelc", 4),
    (320, "emb2_pk_s320_w8.mlmodelc", 4),
    (384, "emb2_pk_s384_w8.mlmodelc", 4),
    (448, "emb2_pk_s448_w8.mlmodelc", 4),
    (512, "emb2_pk_s512_w8.mlmodelc", 4),
)

_CPU_BUNDLES: tuple[tuple[int, str, int], ...] = (
    (256, "emb2_cpu_s256_mix_w8.mlmodelc", 4),
    (384, "emb2_cpu_s384_mix_w8.mlmodelc", 4),
    (512, "emb2_cpu_s512_mix_w8.mlmodelc", 4),
)

_QUERY_CAPS: tuple[int, ...] = (32, 64, 128, 192, 256, 320, 384, 448, 512)

HETERO_CPU_MAX_SEC: float = 0.75
HETERO_CPU_MAX_DOCS: int = 192
HETERO_CPU_MAX_TOKS: int = 32768


def _should_stop_hetero_cpu(elapsed_s: float, cpu_docs: int, cpu_toks: int, *, eff_mode: str) -> bool:
    """Keep CPU-SME on short bursts only in `hetero` mode so sustained runs do not throttle ANE/GPU."""
    if eff_mode != "hetero":
        return False
    return (
        (elapsed_s >= HETERO_CPU_MAX_SEC)
        or (cpu_docs >= HETERO_CPU_MAX_DOCS)
        or (cpu_toks >= HETERO_CPU_MAX_TOKS)
    )


def _load_coreml_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.emb2_pk_load_units.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.emb2_pk_load_units.restype = ctypes.c_void_p
    lib.emb2_pk_submit_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pk_submit_queue.restype = ctypes.c_void_p
    lib.emb2_pk_wait_queue.argtypes = [ctypes.c_void_p]
    lib.emb2_pk_wait_queue.restype = ctypes.c_double
    lib.emb2_pk_exec_queue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pk_exec_queue.restype = ctypes.c_double
    lib.emb2_pk_free.argtypes = [ctypes.c_void_p]
    lib.emb2_pk_free.restype = None
    return lib


def _load_io_lib(dylib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(dylib_path))
    lib.emb2_prep_batch.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_prep_batch.restype = None
    lib.emb2_pool_project_batch.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.emb2_pool_project_batch.restype = None
    lib.emb2_pool_project_batch_dim.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.emb2_pool_project_batch_dim.restype = None
    return lib


class CoreMLScratch:
    """Reusable aligned buffers for C++/NEON prep -> CoreML -> Accelerate projection."""

    def __init__(self, max_N: int, max_S: int) -> None:
        self.h_all = np.empty((max_N, 512, max_S), dtype=np.float16)
        self.b_all = np.empty((max_N, max_S, max_S), dtype=np.float16)
        self.cs_all = np.empty((max_N, max_S, 128), dtype=np.float16)
        self.ss_all = np.empty((max_N, max_S, 128), dtype=np.float16)
        self.cf_all = np.empty((max_N, max_S, 256), dtype=np.float16)
        self.sf_all = np.empty((max_N, max_S, 256), dtype=np.float16)
        self.out_all = np.empty((max_N, max_S, 512), dtype=np.float16)
        self.ids_flat = np.empty(max_N * max_S, dtype=np.int32)
        self.pos_flat = np.empty(max_N * max_S, dtype=np.int32)
        self.arange_s = np.arange(max_S, dtype=np.int32)
        self.span_offsets = np.empty(max_N + 1, dtype=np.int32)
        self.spans_se = np.empty((max_N * 16, 2), dtype=np.int32)
        self.doc_lane_se = np.empty((max_N * 16, 3), dtype=np.int32)
        self.scratch_pooled = np.empty((max_N * 16, 512), dtype=np.float32)
        self.out_embs = np.empty((max_N * 16, EMBED_DIM), dtype=np.float32)

    def ensure_docs(self, tot_docs: int) -> None:
        if tot_docs > len(self.spans_se):
            cap = tot_docs + 64
            self.spans_se = np.empty((cap, 2), dtype=np.int32)
            self.doc_lane_se = np.empty((cap, 3), dtype=np.int32)
            self.scratch_pooled = np.empty((cap, 512), dtype=np.float32)
            self.out_embs = np.empty((cap, EMBED_DIM), dtype=np.float32)


MAX_EMBED_DIM: int = EMBED_DIM
MRL_DIMENSIONS: tuple[int, ...] = (128, 256, 512, 768)


class FastEmbeddingGemma2:
    """Heterogeneous M4 Pro EmbeddingGemma-2 engine (GPU MPP + ANE W8 + CPU-SME mix-W8)."""

    dim: int = EMBED_DIM
    max_dim: int = EMBED_DIM
    supports_dimensions: bool = True
    mrl_dimensions: tuple[int, ...] = MRL_DIMENSIONS
    model_name: str = "google/embeddinggemma-2"
    max_seq_len: int = MAX_SEQ_LEN
    ane_query_max_len: int = 512

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        default_mode: str = "hetero2",
        warmup: bool = True,
        enable_ane: bool = True,
        enable_cpu_sme: bool = True,
        dimensions: int | None = None,
    ) -> None:
        self.dim = resolve_embedding_dim(dimensions, max_dim=EMBED_DIM)
        self.paths = require_runtime_paths(home, require_ane=False)
        self.tokenizer = FastEmbeddingGemma2Tokenizer(self.paths.tokenizer_json)
        self.gpu = FastEmbeddingGemma2GPU(self.paths.weights_npz)

        self._coreml: ctypes.CDLL | None = None
        self._io: ctypes.CDLL | None = None
        self.ane_handles: dict[int, int] = {}
        self.cpu_handles: dict[int, int] = {}
        self._handle_locks: dict[tuple[str, int], threading.Lock] = {}
        self.ane_available = False
        self.cpu_sme_available = False

        self._emb_scaled_np: np.ndarray | None = None
        self._emb_proj_T32_np: np.ndarray | None = None
        self._c256_np: np.ndarray | None = None
        self._s256_np: np.ndarray | None = None
        self._c512_np: np.ndarray | None = None
        self._s512_np: np.ndarray | None = None
        self._neg_inf_u16 = int(np.float16(-10000.0).view(np.uint16))

        self._query_scratch: CoreMLScratch | None = None
        self._ane_scratch: CoreMLScratch | None = None
        self._ane_scratch_b: CoreMLScratch | None = None
        self._cpu_scratch: CoreMLScratch | None = None
        self._gpu_lock = threading.Lock()
        self._corpus_lock = threading.Lock()
        self.last_stats: dict[str, int] | None = None

        if (
            (enable_ane or enable_cpu_sme)
            and self.paths.coreml_dylib.exists()
            and self.paths.io_dylib.exists()
            and self.paths.ane_models_dir.exists()
        ):
            self._init_coreml(enable_ane=enable_ane, enable_cpu_sme=enable_cpu_sme)

        self.default_mode = self._resolve_mode(default_mode)
        if warmup:
            self.warmup()

    def _resolve_dim(self, dimensions: int | None) -> int:
        return resolve_embedding_dim(dimensions, max_dim=EMBED_DIM, default_dim=self.dim)

    def _resolve_mode(self, mode: str) -> str:
        if mode == "hetero":
            if self.ane_available and self.cpu_sme_available:
                return "hetero"
            if self.ane_available:
                return "hetero2"
            if self.cpu_sme_available:
                return "hetero"
            return "gpu"
        if mode == "hetero2":
            return "hetero2" if self.ane_available else "gpu"
        if mode == "ane" and not self.ane_available:
            raise RuntimeError(
                f"ANE requested but CoreML runner dylibs or .mlmodelc bundles are missing under {self.paths.home}"
            )
        if mode == "cpu" and not self.cpu_sme_available:
            raise RuntimeError(
                f"CPU-SME requested but CoreML runner dylibs or cpu .mlmodelc bundles are missing under {self.paths.home}"
            )
        if mode not in ("gpu", "ane", "cpu", "hetero", "hetero2"):
            raise ValueError(f"Unsupported mode: {mode!r}")
        return mode

    def _init_coreml(self, *, enable_ane: bool, enable_cpu_sme: bool) -> None:
        coreml = _load_coreml_lib(self.paths.coreml_dylib)
        io_lib = _load_io_lib(self.paths.io_dylib)

        if enable_ane:
            for seq_len, fname, pd in _ANE_BUNDLES:
                p = self.paths.ane_models_dir / fname
                if not p.exists():
                    continue
                h = coreml.emb2_pk_load_units(str(p).encode("utf-8"), seq_len, pd, 0)
                if h:
                    self.ane_handles[seq_len] = h
                    self._handle_locks[("ane", seq_len)] = threading.Lock()

        if enable_cpu_sme:
            for seq_len, fname, pd in _CPU_BUNDLES:
                p = self.paths.ane_models_dir / fname
                if not p.exists():
                    continue
                h = coreml.emb2_pk_load_units(str(p).encode("utf-8"), seq_len, pd, 1)
                if h:
                    self.cpu_handles[seq_len] = h
                    self._handle_locks[("cpu", seq_len)] = threading.Lock()

        if self.ane_handles or self.cpu_handles:
            self._coreml = coreml
            self._io = io_lib
            self._emb_scaled_np = np.ascontiguousarray(np.array(self.gpu.embed_scaled))
            self._emb_proj_T32_np = np.ascontiguousarray(np.array(self.gpu.emb_proj_wT32))
            self._c256_np = np.ascontiguousarray(np.array(self.gpu.c256))
            self._s256_np = np.ascontiguousarray(np.array(self.gpu.s256))
            self._c512_np = np.ascontiguousarray(np.array(self.gpu.c512))
            self._s512_np = np.ascontiguousarray(np.array(self.gpu.s512))
            self._query_scratch = CoreMLScratch(1, 512)
            self._ane_scratch = CoreMLScratch(32, 512)
            self._ane_scratch_b = CoreMLScratch(32, 512)
            self._cpu_scratch = CoreMLScratch(24, 512)
            self.ane_available = bool(self.ane_handles)
            self.cpu_sme_available = bool(self.cpu_handles)

    def close(self) -> None:
        if self._coreml is not None:
            for h in self.ane_handles.values():
                self._coreml.emb2_pk_free(h)
            for h in self.cpu_handles.values():
                self._coreml.emb2_pk_free(h)
            self.ane_handles.clear()
            self.cpu_handles.clear()
        self.ane_available = False
        self.cpu_sme_available = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _prep_coreml_lanes(
        self,
        S: int,
        lanes: list,
        scratch: CoreMLScratch,
    ) -> tuple[int, int, int, list[int]]:
        assert self._io is not None
        assert self._emb_scaled_np is not None
        assert self._c256_np is not None and self._s256_np is not None
        assert self._c512_np is not None and self._s512_np is not None

        N = len(lanes)
        ids = scratch.ids_flat[: N * S].reshape(N, S)
        pos = scratch.pos_flat[: N * S].reshape(N, S)
        ids.fill(-1)
        pos.fill(0)
        span_offsets = scratch.span_offsets
        span_offsets[0] = 0
        sp_i = 0
        doc_idxs: list[int] = []
        tot_docs = sum(len(ln) for ln in lanes)
        scratch.ensure_docs(tot_docs)

        spans_se = scratch.spans_se
        doc_lane_se = scratch.doc_lane_se
        ar = scratch.arange_s
        real_toks = 0
        for n, ln in enumerate(lanes):
            cur = 0
            for idx, tok in ln:
                ll = len(tok)
                real_toks += ll
                ids[n, cur : cur + ll] = tok
                pos[n, cur : cur + ll] = ar[:ll]
                spans_se[sp_i, 0] = cur
                spans_se[sp_i, 1] = cur + ll
                doc_lane_se[sp_i, 0] = n
                doc_lane_se[sp_i, 1] = cur
                doc_lane_se[sp_i, 2] = cur + ll
                doc_idxs.append(idx)
                sp_i += 1
                cur += ll
            span_offsets[n + 1] = sp_i

        self._io.emb2_prep_batch(
            N,
            S,
            self._emb_scaled_np.ctypes.data,
            self._c256_np.ctypes.data,
            self._s256_np.ctypes.data,
            self._c512_np.ctypes.data,
            self._s512_np.ctypes.data,
            ids.ctypes.data,
            pos.ctypes.data,
            span_offsets.ctypes.data,
            spans_se.ctypes.data,
            self._neg_inf_u16,
            scratch.h_all.ctypes.data,
            scratch.b_all.ctypes.data,
            scratch.cs_all.ctypes.data,
            scratch.ss_all.ctypes.data,
            scratch.cf_all.ctypes.data,
            scratch.sf_all.ctypes.data,
        )
        return N, tot_docs, real_toks, doc_idxs

    def _finish_coreml_lanes(
        self,
        S: int,
        tot_docs: int,
        scratch: CoreMLScratch,
        out_dest: np.ndarray,
        doc_idxs: list[int],
        eff_dim: int,
    ) -> None:
        assert self._io is not None and self._emb_proj_T32_np is not None
        self._io.emb2_pool_project_batch_dim(
            S,
            scratch.out_all.ctypes.data,
            tot_docs,
            scratch.doc_lane_se.ctypes.data,
            self._emb_proj_T32_np.ctypes.data,
            scratch.scratch_pooled.ctypes.data,
            scratch.out_embs.ctypes.data,
            eff_dim,
        )
        out_dest[doc_idxs] = scratch.out_embs.ravel()[: tot_docs * eff_dim].reshape(tot_docs, eff_dim)

    def _exec_coreml_lanes(
        self,
        kind: str,
        S: int,
        lanes: list,
        scratch: CoreMLScratch,
        out_dest: np.ndarray,
        *,
        eff_dim: int = EMBED_DIM,
    ) -> tuple[int, int]:
        assert self._coreml is not None
        handle = self.ane_handles[S] if kind == "ane" else self.cpu_handles[S]
        lock = self._handle_locks[(kind, S)]
        N, tot_docs, real_toks, doc_idxs = self._prep_coreml_lanes(S, lanes, scratch)
        with lock:
            self._coreml.emb2_pk_exec_queue(
                handle,
                N,
                scratch.h_all.ctypes.data,
                scratch.b_all.ctypes.data,
                scratch.cs_all.ctypes.data,
                scratch.ss_all.ctypes.data,
                scratch.cf_all.ctypes.data,
                scratch.sf_all.ctypes.data,
                scratch.out_all.ctypes.data,
            )
        self._finish_coreml_lanes(S, tot_docs, scratch, out_dest, doc_idxs, eff_dim)
        return real_toks, N * S

    def warmup(
        self,
        *,
        prewarm_gpu_shapes: bool = True,
        prewarm_over_shapes: bool = False,
    ) -> None:
        if self.ane_available:
            self.encode_query([2, 3250, 236787, 7293, 1109, 1816, 236787, 1], prefer_ane=True)
        if self.default_mode == "ane" and self.ane_available:
            return
        with self._gpu_lock:
            self.gpu.encode_single([2, 3250, 236787, 7293, 1109, 1816, 236787, 1])
            if prewarm_gpu_shapes:
                warm_arrs: list[mx.array] = []
                for L_l, b_tup in GPU_B_OPTS.items():
                    for B in b_tup:
                        _, _, _, o = self.gpu.exec_packed(L_l, [[(0, [2, 1])]] * B)
                        warm_arrs.append(o)
                if prewarm_over_shapes:
                    for ml, b_tup in OVER_B_OPTS.items():
                        for B in b_tup:
                            _, _, _, o = self.gpu.exec_over(ml, [(0, [2, 1])] * B)
                            warm_arrs.append(o)
                mx.eval(*warm_arrs)
                mx.synchronize()

    def encode_query(
        self,
        token_ids: Sequence[int] | np.ndarray,
        prefer_ane: bool = True,
        *,
        dimensions: int | None = None,
    ) -> np.ndarray:
        """Low-latency single-query encoder returning a `(dim,)` L2-normalized float32 vector."""
        eff_dim = self._resolve_dim(dimensions)
        arr = np.ascontiguousarray(token_ids[:MAX_SEQ_LEN], dtype=np.int32)
        if arr.shape[0] == 0:
            arr = np.array([self.tokenizer.bos_id, self.tokenizer.eos_id], dtype=np.int32)
        sl = int(arr.shape[0])

        if prefer_ane and self.ane_available and sl <= self.ane_query_max_len:
            assert self._query_scratch is not None
            for cap in _QUERY_CAPS:
                if sl <= cap and cap in self.ane_handles:
                    out = np.empty((1, eff_dim), dtype=np.float32)
                    self._exec_coreml_lanes("ane", cap, [[(0, arr)]], self._query_scratch, out, eff_dim=eff_dim)
                    return out[0]

        with self._gpu_lock:
            if isinstance(self.gpu, FastEmbeddingGemma2GPU):
                return self.gpu.encode_single(arr, dimensions=eff_dim)
            vec = self.gpu.encode_single(arr)
        return truncate_and_normalize(vec, eff_dim) if vec.shape[0] != eff_dim else vec

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool = False,
        add_query_prefix: bool = False,
        add_document_prefix: bool = False,
        prefix: str | None = None,
        prompt_name: str | None = None,
        mode: str | None = None,
        max_length: int | None = None,
        dimensions: int | None = None,
    ) -> np.ndarray:
        """Embed a single string `(dim,)` or sequence of strings `[N, dim]` into L2-normalized float32 vectors."""
        eff_dim = self._resolve_dim(dimensions)
        eff_mode = self._resolve_mode(mode or self.default_mode)
        prefer_ane = eff_mode != "gpu"
        ml = MAX_SEQ_LEN if max_length is None else min(MAX_SEQ_LEN, int(max_length))

        if isinstance(texts, str):
            ids = self.tokenizer.encode(
                texts,
                prefix=prefix,
                prompt_name=prompt_name,
                add_query_prefix=add_query_prefix,
                add_document_prefix=add_document_prefix,
                max_length=ml,
            )
            return self.encode_query(ids, prefer_ane=prefer_ane, dimensions=eff_dim)

        items = list(texts)
        if not items:
            return np.empty((0, eff_dim), dtype=np.float32)

        tok_lists = self.tokenizer.encode_batch(
            items,
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_query_prefix,
            add_document_prefix=add_document_prefix,
            max_length=ml,
        )
        if is_query or len(tok_lists) == 1:
            if len(tok_lists) == 1:
                return self.encode_query(tok_lists[0], prefer_ane=prefer_ane, dimensions=eff_dim)[None, :]
            return np.stack([self.encode_query(ids, prefer_ane=prefer_ane, dimensions=eff_dim) for ids in tok_lists], axis=0)

        return self.encode_corpus(tok_lists, mode=eff_mode, dimensions=eff_dim)

    def encode_corpus(
        self,
        token_lists: Sequence[Sequence[int]],
        *,
        mode: str | None = None,
        dimensions: int | None = None,
    ) -> np.ndarray:
        """Batch-embed pre-tokenized sequences `[N]` into `[N, dim]` float32 L2-normalized vectors."""
        eff_dim = self._resolve_dim(dimensions)
        n = len(token_lists)
        if n == 0:
            self.last_stats = {"gpu_toks": 0, "ane_toks": 0, "cpu_toks": 0, "gpu_docs": 0, "ane_docs": 0, "cpu_docs": 0}
            return np.empty((0, eff_dim), dtype=np.float32)

        eff_mode = self._resolve_mode(mode or self.default_mode)
        empty_seq = (self.tokenizer.bos_id, self.tokenizer.eos_id)
        seqs = [
            (i, t[:MAX_SEQ_LEN] if len(t) > 0 else empty_seq)
            for i, t in enumerate(token_lists)
        ]

        if len(seqs) == 1 and eff_mode in ("gpu", "hetero", "hetero2"):
            v = self.encode_query(seqs[0][1], prefer_ane=(eff_mode != "gpu"), dimensions=eff_dim)
            self.last_stats = {
                "gpu_toks": 0 if (eff_mode != "gpu" and self.ane_available and len(seqs[0][1]) <= self.ane_query_max_len) else len(seqs[0][1]),
                "ane_toks": len(seqs[0][1]) if (eff_mode != "gpu" and self.ane_available and len(seqs[0][1]) <= self.ane_query_max_len) else 0,
                "cpu_toks": 0,
                "gpu_docs": 0 if (eff_mode != "gpu" and self.ane_available and len(seqs[0][1]) <= self.ane_query_max_len) else 1,
                "ane_docs": 1 if (eff_mode != "gpu" and self.ane_available and len(seqs[0][1]) <= self.ane_query_max_len) else 0,
                "cpu_docs": 0,
            }
            return v[None, :]

        outs = np.empty((n, eff_dim), dtype=np.float32)

        with self._corpus_lock:
            by_b, over = fast_pack_cost_aware(seqs, ANE_MS_COST)
            over_q = build_over_chunks(over)
            qlock = threading.Lock()
            use_ane = eff_mode in ("hetero", "hetero2", "ane") and self.ane_available
            use_cpu = eff_mode in ("hetero", "cpu") and self.cpu_sme_available
            use_gpu = eff_mode in ("hetero", "hetero2", "gpu") or (len(over_q) > 0)

            stats = {"gpu": 0, "ane": 0, "cpu": 0, "gpu_docs": 0, "ane_docs": 0, "cpu_docs": 0}

            def pop_lanes(L_l: int, count: int) -> list:
                q = by_b[L_l]
                take = min(count, len(q))
                return [q.popleft() for _ in range(take)]

            def total_shared_rem() -> int:
                return sum(len(q) for q in by_b.values())

            def gpu_worker() -> None:
                prev_o: mx.array | None = None
                prev_idxs: list[int] | None = None
                gpu_only_mode = (eff_mode == "gpu") or (not use_ane and not use_cpu)
                gpu_pref = sorted(by_b.keys()) if gpu_only_mode else (448, 320, 192, 512, 384, 256, 128)
                with self._gpu_lock:
                    while True:
                        job = None
                        with qlock:
                            if over_q:
                                ml, chk = over_q.popleft()
                                job = ("over", ml, chk)
                            elif eff_mode in ("hetero", "hetero2", "gpu"):
                                for L_l in gpu_pref:
                                    q_b = by_b.get(L_l)
                                    n_av = len(q_b) if q_b is not None else 0
                                    if n_av == 0:
                                        continue
                                    opts = GPU_B_OPTS[L_l]
                                    cand_opts = opts if (gpu_only_mode or not use_ane) else opts[:-1]
                                    chosen = 0
                                    for opt in cand_opts:
                                        if opt <= n_av:
                                            chosen = opt
                                            break
                                    if chosen > 0:
                                        job = ("packed", L_l, pop_lanes(L_l, chosen))
                                        break
                        if job is None:
                            break
                        kind, param, chk = job
                        if kind == "over":
                            rt, _, di, cur_o = self.gpu.exec_over(param, chk, dimensions=eff_dim)
                        else:
                            rt, _, di, cur_o = self.gpu.exec_packed(param, chk, dimensions=eff_dim)
                        mx.async_eval(cur_o)
                        if prev_o is not None and prev_idxs is not None:
                            outs[prev_idxs] = np.asarray(prev_o, dtype=np.float32)
                        prev_o, prev_idxs = cur_o, di
                        stats["gpu"] += rt
                        stats["gpu_docs"] += len(di)
                    if prev_o is not None and prev_idxs is not None:
                        outs[prev_idxs] = np.asarray(prev_o, dtype=np.float32)
                    mx.synchronize()

            def ane_worker() -> None:
                assert self._ane_scratch is not None and self._ane_scratch_b is not None
                assert self._coreml is not None
                ane_pref = [(128, 24), (192, 20), (256, 20), (320, 16), (384, 16), (448, 12), (512, 12)]
                fallback = [(192, 256), (320, 384), (448, 512), (128, 128), (256, 256), (384, 384), (512, 512)]
                scratches = (self._ane_scratch, self._ane_scratch_b)
                buf_idx = 0
                inflight = None

                def pop_ane_job():
                    with qlock:
                        rem_sh = total_shared_rem()
                        for S, batch_n in ane_pref:
                            if S not in self.ane_handles:
                                continue
                            n_av = len(by_b.get(S, ()))
                            if n_av > 0:
                                if eff_mode == "ane":
                                    n_take = min(batch_n, n_av)
                                elif rem_sh <= 20:
                                    n_take = max(4, min(batch_n, (n_av + 1) // 2)) if n_av > 4 else n_av
                                elif rem_sh <= 48:
                                    n_take = min(batch_n, max(8, (n_av + 1) // 2))
                                else:
                                    n_take = min(batch_n, n_av)
                                chk = pop_lanes(S, n_take)
                                S_exec = 64 if (S == 128 and 64 in self.ane_handles and all(sum(len(t) for _, t in ln) <= 64 for ln in chk)) else S
                                return (S_exec, chk)
                        for src_b, s_tgt in fallback:
                            if s_tgt not in self.ane_handles:
                                continue
                            n_av = len(by_b.get(src_b, ()))
                            if n_av > 0:
                                n_take = min(8, n_av) if eff_mode == "ane" else min(8, max(1, (n_av + 1) // 2))
                                return (s_tgt, pop_lanes(src_b, n_take))
                    return None

                try:
                    while True:
                        job = pop_ane_job()
                        if job is None:
                            break
                        S_exec, chk = job
                        cur_sc = scratches[buf_idx]
                        buf_idx ^= 1
                        N_c, tot_docs_c, real_toks_c, doc_idxs_c = self._prep_coreml_lanes(S_exec, chk, cur_sc)

                        prev = inflight
                        inflight = None
                        if prev is not None:
                            p_lock, p_flight, p_S, p_tot_docs, p_doc_idxs, p_rt, p_sc = prev
                            try:
                                self._coreml.emb2_pk_wait_queue(p_flight)
                            finally:
                                p_lock.release()

                        c_lock = self._handle_locks[("ane", S_exec)]
                        c_handle = self.ane_handles[S_exec]
                        c_lock.acquire()
                        try:
                            c_flight = self._coreml.emb2_pk_submit_queue(
                                c_handle,
                                N_c,
                                cur_sc.h_all.ctypes.data,
                                cur_sc.b_all.ctypes.data,
                                cur_sc.cs_all.ctypes.data,
                                cur_sc.ss_all.ctypes.data,
                                cur_sc.cf_all.ctypes.data,
                                cur_sc.sf_all.ctypes.data,
                                cur_sc.out_all.ctypes.data,
                            )
                        except BaseException:
                            c_lock.release()
                            raise
                        inflight = (c_lock, c_flight, S_exec, tot_docs_c, doc_idxs_c, real_toks_c, cur_sc)

                        if prev is not None:
                            self._finish_coreml_lanes(p_S, p_tot_docs, p_sc, outs, p_doc_idxs, eff_dim)
                            stats["ane"] += p_rt
                            stats["ane_docs"] += p_tot_docs

                    if inflight is not None:
                        p_lock, p_flight, p_S, p_tot_docs, p_doc_idxs, p_rt, p_sc = inflight
                        inflight = None
                        try:
                            self._coreml.emb2_pk_wait_queue(p_flight)
                        finally:
                            p_lock.release()
                        self._finish_coreml_lanes(p_S, p_tot_docs, p_sc, outs, p_doc_idxs, eff_dim)
                        stats["ane"] += p_rt
                        stats["ane_docs"] += p_tot_docs
                finally:
                    if inflight is not None:
                        p_lock, p_flight, *_ = inflight
                        try:
                            self._coreml.emb2_pk_wait_queue(p_flight)
                        finally:
                            p_lock.release()

            def cpu_worker() -> None:
                assert self._cpu_scratch is not None
                cpu_pref_hi = [(256, 16), (384, 12), (512, 8)]
                cpu_pref_lo = [(256, 8), (384, 6), (512, 4)]
                cpu_fallback = [(128, 256), (192, 256), (320, 384), (448, 512), (256, 256), (384, 384), (512, 512)]
                cpu_t0 = time.perf_counter()
                while True:
                    if use_gpu and use_ane and _should_stop_hetero_cpu(
                        time.perf_counter() - cpu_t0, stats["cpu_docs"], stats["cpu"], eff_mode=eff_mode
                    ):
                        break
                    job = None
                    with qlock:
                        rem_sh = total_shared_rem()
                        cpu_pref = cpu_pref_hi if rem_sh > 64 else cpu_pref_lo
                        for S, batch_n in cpu_pref:
                            if S not in self.cpu_handles:
                                continue
                            n_av = len(by_b.get(S, ()))
                            if n_av > 0:
                                if eff_mode == "cpu":
                                    n_take = min(batch_n, n_av)
                                elif rem_sh <= 20:
                                    n_take = max(4, min(batch_n, (n_av + 2) // 3)) if n_av > 4 else n_av
                                else:
                                    n_take = min(batch_n, n_av)
                                job = (S, pop_lanes(S, n_take))
                                break
                        if job is None and (eff_mode == "cpu" or not use_ane):
                            for src_b, s_tgt in cpu_fallback:
                                if s_tgt not in self.cpu_handles:
                                    continue
                                n_av = len(by_b.get(src_b, ()))
                                if n_av > 0:
                                    n_take = min(8, n_av) if eff_mode == "cpu" else min(4, max(1, (n_av + 1) // 2))
                                    job = (s_tgt, pop_lanes(src_b, n_take))
                                    break
                        elif job is None:
                            for src_b, s_tgt in ((192, 256), (320, 384), (448, 512)):
                                if s_tgt not in self.cpu_handles:
                                    continue
                                n_av = len(by_b.get(src_b, ()))
                                if n_av > 0:
                                    n_take = min(4, max(1, (n_av + 1) // 2))
                                    job = (s_tgt, pop_lanes(src_b, n_take))
                                    break
                    if job is None:
                        break
                    S, chk = job
                    rt, _ = self._exec_coreml_lanes("cpu", S, chk, self._cpu_scratch, outs, eff_dim=eff_dim)
                    stats["cpu"] += rt
                    stats["cpu_docs"] += sum(len(ln) for ln in chk)

            threads: list[threading.Thread] = []
            if use_gpu:
                threads.append(threading.Thread(target=gpu_worker, name="emb2-gpu"))
            if use_ane:
                threads.append(threading.Thread(target=ane_worker, name="emb2-ane"))
            if use_cpu:
                threads.append(threading.Thread(target=cpu_worker, name="emb2-cpu"))
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            self.last_stats = {
                "gpu_toks": stats["gpu"],
                "ane_toks": stats["ane"],
                "cpu_toks": stats["cpu"],
                "gpu_docs": stats["gpu_docs"],
                "ane_docs": stats["ane_docs"],
                "cpu_docs": stats["cpu_docs"],
            }
        return outs
