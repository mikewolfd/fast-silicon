"""Stdlib-only hard-RAM-bounded `PrefixTree` for `256`-token KV + hybrid GDN state caching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import threading
from typing import Any, Callable, Sequence

ROOT_KEY: str = "root"
DEFAULT_CHUNK_TOKENS: int = 256
_MIB = 1 << 20
_GIB = 1 << 30
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def recommended_prefix_budget_bytes(host_ram_bytes: int) -> int:
    """Return conservative hard RAM budget for hybrid KV+GDN prefix caching (`128 MiB` on `<=24 GiB`, `512 MiB` on `>=48 GiB`)."""
    ram = int(host_ram_bytes)
    if ram <= 0:
        return 128 * _MIB
    if ram <= 24 * _GIB:
        return 128 * _MIB
    if ram <= 36 * _GIB:
        return 256 * _MIB
    return 512 * _MIB


def payload_nbytes(obj: Any) -> int:
    """Recursively measure exact payload bytes for bytes-like objects, array-likes (`.nbytes`), or nested containers."""
    if obj is None:
        return 0
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, memoryview):
        return int(obj.nbytes)
    nb = getattr(obj, "nbytes", None)
    if isinstance(nb, int):
        return nb
    if isinstance(obj, (tuple, list)):
        return sum(payload_nbytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(payload_nbytes(val) for val in obj.values())
    raise TypeError(
        f"Cannot determine exact byte size for prefix cache payload of type {type(obj).__name__}; "
        "pass explicit `byte_size` or store bytes/array-like containers."
    )


def chain_key(parent_key: str, tokens: Sequence[int]) -> str:
    """Compute chained SHA-256 hex digest extending `parent_key` with one chunk of `int32` token IDs."""
    h = hashlib.sha256()
    h.update(parent_key.encode("utf-8"))
    ints = [int(t) for t in tokens]
    if any(t < _INT32_MIN or t > _INT32_MAX for t in ints):
        raise ValueError("Token IDs must fit in signed int32 range for PrefixTree hashing.")
    h.update(struct.pack(f"<{len(ints)}i", *ints))
    return h.hexdigest()


def snapshot_boundaries(
    start: int,
    total_tokens: int,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    snapshot_stride: int = 1,
    include_deepest: bool = True,
) -> list[int]:
    """Return sorted absolute token positions in `(start, total_tokens]` where hybrid GDN snapshots should be taken."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if snapshot_stride <= 0:
        raise ValueError(f"snapshot_stride must be positive, got {snapshot_stride}")
    if total_tokens <= start or total_tokens < chunk_tokens:
        return []
    first_depth = (max(0, start) // chunk_tokens) + 1
    last_depth = total_tokens // chunk_tokens
    if first_depth > last_depth:
        return []
    out = [d * chunk_tokens for d in range(first_depth, last_depth + 1) if d % snapshot_stride == 0]
    deepest = last_depth * chunk_tokens
    if include_deepest and deepest > start and (not out or out[-1] != deepest):
        out.append(deepest)
    return out


@dataclass(frozen=True)
class PrefixMatch:
    """Result of walking a token sequence down a `PrefixTree` without mutating LRU counters."""

    keys: tuple[str, ...] = ()
    chunks: tuple[Any, ...] = ()
    snapshot: Any = None
    reused_tokens: int = 0
    replayed_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return self.reused_tokens == 0


@dataclass
class _PrefixNode:
    depth: int
    parent: str | None
    tokens: tuple[int, ...]
    chunk: Any
    chunk_bytes: int
    snapshot: Any = None
    snapshot_bytes: int = 0
    children: int = 0
    last_used: int = 0

    @property
    def total_bytes(self) -> int:
        return self.chunk_bytes + self.snapshot_bytes


class PrefixTree:
    """Chained SHA-256 `256`-token prefix tree holding KV chunks + hybrid recurrent/conv state snapshots.

    Invariants:
      - Every node covers exactly `chunk_tokens` tokens at `[ (depth-1)*chunk_tokens, depth*chunk_tokens )`.
      - Lookups and duplicate inserts verify exact token tuple equality in addition to chained SHA-256 keys.
      - For hybrid models (`needs_snapshot=True`), reuse caps at the deepest matched node carrying a snapshot.
      - `0 <= bytes_held <= byte_budget` is strictly enforced; eviction drops LRU leaves (`children == 0`) only,
        never stranding a child without its ancestors and never evicting the active protected chain.
    """

    def __init__(
        self,
        *,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        byte_budget: int = 256 * _MIB,
    ) -> None:
        if int(chunk_tokens) <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
        if int(byte_budget) < 0:
            raise ValueError(f"byte_budget must be non-negative, got {byte_budget}")
        self.chunk_tokens = int(chunk_tokens)
        self.byte_budget = int(byte_budget)

        self._lock = threading.RLock()
        self._nodes: dict[str, _PrefixNode] = {}
        self._clock: int = 0
        self._bytes_held: int = 0

        self.hits: int = 0
        self.misses: int = 0
        self.reused_tokens: int = 0
        self.matched_but_replayed_tokens: int = 0
        self.evicted_nodes: int = 0

    @property
    def bytes_held(self) -> int:
        with self._lock:
            return self._bytes_held

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return sum(1 for n in self._nodes.values() if n.snapshot is not None)

    def has_snapshot(self, key: str) -> bool:
        with self._lock:
            node = self._nodes.get(key)
            return node is not None and node.snapshot is not None

    def match(
        self,
        tokens: Sequence[int],
        *,
        limit: int | None = None,
        needs_snapshot: bool = True,
    ) -> PrefixMatch:
        """Walk `tokens` down the tree and return the longest usable prefix without mutating LRU state."""
        with self._lock:
            eff_limit = len(tokens) if limit is None else min(int(limit), len(tokens))
            if not self._nodes or eff_limit < self.chunk_tokens:
                return PrefixMatch()

            key = ROOT_KEY
            path: list[str] = []
            snapshot_depth = 0
            depth = 0
            ct = self.chunk_tokens
            while (depth + 1) * ct <= eff_limit:
                chunk_tup = tuple(int(t) for t in tokens[depth * ct : (depth + 1) * ct])
                key = chain_key(key, chunk_tup)
                node = self._nodes.get(key)
                if node is None or node.tokens != chunk_tup:
                    break
                path.append(key)
                if node.snapshot is not None:
                    snapshot_depth = len(path)
                depth += 1

            usable_depth = snapshot_depth if needs_snapshot else len(path)
            if usable_depth == 0:
                return PrefixMatch(replayed_tokens=len(path) * ct)

            usable_keys = tuple(path[:usable_depth])
            usable_chunks = tuple(self._nodes[k].chunk for k in usable_keys)
            snap = self._nodes[usable_keys[-1]].snapshot if needs_snapshot else None
            return PrefixMatch(
                keys=usable_keys,
                chunks=usable_chunks,
                snapshot=snap,
                reused_tokens=usable_depth * ct,
                replayed_tokens=(len(path) - usable_depth) * ct,
            )

    def record(self, match: PrefixMatch, *, restored: bool) -> None:
        """Record a match outcome and bump LRU timestamps along matched nodes."""
        with self._lock:
            if match.keys:
                self._clock += 1
                for k in match.keys:
                    node = self._nodes.get(k)
                    if node is not None:
                        node.last_used = self._clock
            if restored:
                self.hits += 1
                self.reused_tokens += match.reused_tokens
                self.matched_but_replayed_tokens += match.replayed_tokens
            elif match.is_empty:
                self.misses += 1
                self.matched_but_replayed_tokens += match.replayed_tokens

    def insert(
        self,
        *,
        parent: str | None,
        tokens: Sequence[int],
        depth: int,
        chunk: Any | Callable[[], Any],
        byte_size: int | None = None,
    ) -> str | None:
        """Insert one full `chunk_tokens`-wide KV chunk under `parent`, enforcing hard `byte_budget`.

        `chunk` may be a zero-arg factory so a chunk that cannot fit in budget is never materialized.
        Returns the node key if present/inserted, or `None` if invalid, colliding, or over budget.
        """
        with self._lock:
            if depth <= 0 or len(tokens) != self.chunk_tokens:
                return None
            if parent is None:
                if depth != 1:
                    return None
            else:
                pnode = self._nodes.get(parent)
                if pnode is None or pnode.depth != depth - 1:
                    return None

            tok_tup = tuple(int(t) for t in tokens)
            key = chain_key(parent if parent is not None else ROOT_KEY, tok_tup)
            existing = self._nodes.get(key)
            if existing is not None:
                if existing.tokens != tok_tup:
                    return None
                self._clock += 1
                existing.last_used = self._clock
                return key

            if callable(chunk):
                if byte_size is None:
                    raise ValueError("byte_size is required when `chunk` is a lazy factory callable.")
                size = int(byte_size)
                if size < 0:
                    raise ValueError(f"byte_size must be non-negative, got {size}")
                if not self._make_room(size, self._ancestors(parent)):
                    return None
                payload = chunk()
                actual = payload_nbytes(payload)
                if actual != size:
                    raise ValueError(f"Chunk byte_size mismatch: declared {size}, actual {actual}")
            else:
                payload = chunk
                actual = payload_nbytes(payload)
                size = actual if byte_size is None else int(byte_size)
                if size < 0:
                    raise ValueError(f"byte_size must be non-negative, got {size}")
                if size != actual:
                    raise ValueError(f"Chunk byte_size mismatch: declared {size}, actual {actual}")
                if not self._make_room(size, self._ancestors(parent)):
                    return None

            self._clock += 1
            self._nodes[key] = _PrefixNode(
                depth=depth,
                parent=parent,
                tokens=tok_tup,
                chunk=payload,
                chunk_bytes=size,
                last_used=self._clock,
            )
            self._bytes_held += size
            if parent is not None:
                self._nodes[parent].children += 1
            return key

    def attach_snapshot(
        self,
        key: str,
        snapshot: Any,
        *,
        keeping_chain_to: str | None = None,
        byte_size: int | None = None,
    ) -> bool:
        """Attach a hybrid recurrent/conv state snapshot to an existing node without stranding its chain."""
        with self._lock:
            node = self._nodes.get(key)
            if node is None or node.snapshot is not None:
                return False
            actual = payload_nbytes(snapshot)
            size = actual if byte_size is None else int(byte_size)
            if size < 0:
                raise ValueError(f"byte_size must be non-negative, got {size}")
            if size != actual:
                raise ValueError(f"Snapshot byte_size mismatch: declared {size}, actual {actual}")

            protected = self._ancestors(key) | self._ancestors(keeping_chain_to)
            if not self._make_room(size, protected):
                return False
            self._clock += 1
            node.snapshot = snapshot
            node.snapshot_bytes = size
            node.last_used = self._clock
            self._bytes_held += size
            return True

    def clear(self) -> int:
        """Drop all cached nodes and snapshots, returning freed bytes."""
        with self._lock:
            freed = self._bytes_held
            self.evicted_nodes += len(self._nodes)
            self._nodes.clear()
            self._bytes_held = 0
            return freed

    shed = clear

    def stats(self) -> dict[str, int | float]:
        """Return structured telemetry snapshot."""
        with self._lock:
            reqs = self.hits + self.misses
            hit_rate = (self.hits / reqs) if reqs > 0 else 0.0
            return {
                "chunk_tokens": self.chunk_tokens,
                "node_count": len(self._nodes),
                "snapshot_count": sum(1 for n in self._nodes.values() if n.snapshot is not None),
                "bytes_held": self._bytes_held,
                "byte_budget": self.byte_budget,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "reused_tokens": self.reused_tokens,
                "matched_but_replayed_tokens": self.matched_but_replayed_tokens,
                "evicted_nodes": self.evicted_nodes,
            }

    def summary(self) -> str:
        st = self.stats()
        reqs = int(st["hits"]) + int(st["misses"])
        return (
            f"{st['node_count']} node(s), {st['snapshot_count']} snapshot(s), "
            f"{st['bytes_held'] / _MIB:.1f} of {st['byte_budget'] / _MIB:.1f} MiB, "
            f"{st['hits']}/{reqs} hit ({float(st['hit_rate']) * 100.0:.0f}%), "
            f"{st['reused_tokens']} token(s) reused, "
            f"{st['matched_but_replayed_tokens']} replayed, {st['evicted_nodes']} evicted"
        )

    def _ancestors(self, key: str | None) -> set[str]:
        out: set[str] = set()
        cur = key
        while cur is not None:
            node = self._nodes.get(cur)
            if node is None:
                break
            out.add(cur)
            cur = node.parent
        return out

    def _make_room(self, needed_bytes: int, protected: set[str]) -> bool:
        protected_bytes = sum(self._nodes[k].total_bytes for k in protected if k in self._nodes)
        if protected_bytes + needed_bytes > self.byte_budget:
            return False
        while self._bytes_held + needed_bytes > self.byte_budget:
            victim_key: str | None = None
            victim_used = 0
            for k, n in self._nodes.items():
                if n.children == 0 and k not in protected:
                    if victim_key is None or n.last_used < victim_used or (n.last_used == victim_used and k < victim_key):
                        victim_key = k
                        victim_used = n.last_used
            if victim_key is None:
                return False
            victim = self._nodes.pop(victim_key)
            if victim.parent is not None and victim.parent in self._nodes:
                self._nodes[victim.parent].children -= 1
            self._bytes_held -= victim.total_bytes
            self.evicted_nodes += 1
        return True


__all__ = [
    "DEFAULT_CHUNK_TOKENS",
    "ROOT_KEY",
    "PrefixMatch",
    "PrefixTree",
    "chain_key",
    "payload_nbytes",
    "recommended_prefix_budget_bytes",
    "snapshot_boundaries",
]
