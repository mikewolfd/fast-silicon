"""Rust-backed Gemma-4 tokenizer, chat formatter, and optional domain/ASCII-focused LM-head vocabulary pruner (zero transformers/torch)."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from tokenizers import Tokenizer

DEFAULT_KEPT_VOCAB_SIZE = 150786
LATIN_EXTENDED_KEPT_VOCAB_SIZE = 162304
FULL_VOCAB_SIZE = 262144
GEMMA4_EOS_TOKEN_IDS: frozenset[int] = frozenset({1, 106, 50})

_CHANNEL_THOUGHT_RE = re.compile(r"<\|channel>.*?<channel\|>", flags=re.DOTALL)
_THOUGHT_RE = re.compile(r"<\|thought>.*?<thought\|>", flags=re.DOTALL)
_LEADING_THOUGHT_HEADER_RE = re.compile(r"^\s*thought(?:\s+|$)", flags=re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    stripped = _CHANNEL_THOUGHT_RE.sub("", text or "")
    stripped = _THOUGHT_RE.sub("", stripped)
    if "<channel|>" in stripped:
        stripped = stripped.rsplit("<channel|>", 1)[-1]
    if "<thought|>" in stripped:
        stripped = stripped.rsplit("<thought|>", 1)[-1]
    if "<|channel>" in stripped:
        stripped = stripped.split("<|channel>", 1)[0]
    if "<|thought>" in stripped:
        stripped = stripped.split("<|thought>", 1)[0]
    return stripped.strip()


def format_gemma4_chat(
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Format Gemma-4 IT messages matching HuggingFace `chat_template.jinja` for text chats."""
    if not messages:
        raise ValueError("messages must be non-empty")

    parts: list[str] = ["<bos>"]
    loop_messages = list(messages)

    first_role = loop_messages[0].get("role", "")
    if enable_thinking or first_role in ("system", "developer"):
        parts.append("<|turn>system\n")
        if enable_thinking:
            parts.append("<|think|>\n")
        if first_role in ("system", "developer"):
            parts.append(str(loop_messages[0].get("content", "")).strip())
            loop_messages = loop_messages[1:]
        parts.append("<turn|>\n")

    prev_non_tool_role: str | None = None
    last_role: str | None = None
    for msg in loop_messages:
        raw_role = str(msg.get("role", "user"))
        if raw_role == "tool":
            continue
        role = "model" if raw_role == "assistant" else raw_role
        last_role = role

        is_cont = prev_non_tool_role == "assistant" and raw_role == "assistant"
        if not is_cont:
            parts.append(f"<|turn>{role}\n")

        content = msg.get("content", "")
        if isinstance(content, str):
            text = _strip_thinking(content) if role == "model" else content.strip()
            parts.append(text)
        elif isinstance(content, Sequence):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type")
                    if t == "text":
                        txt = str(item.get("text", ""))
                        parts.append(_strip_thinking(txt) if role == "model" else txt.strip())
                    elif t == "image":
                        parts.append("<|image|>")
                    elif t == "audio":
                        parts.append("<|audio|>")
                    elif t == "video":
                        parts.append("<|video|>")
        parts.append("<turn|>\n")
        prev_non_tool_role = raw_role

    if add_generation_prompt and last_role != "tool":
        parts.append("<|turn>model\n")
    return "".join(parts)


def build_kept_vocab(tokenizer_json_path: str | Path, target_size: int = DEFAULT_KEPT_VOCAB_SIZE):
    """Select an English and JSON output vocabulary with byte fallbacks for other text."""
    with Path(tokenizer_json_path).open("r", encoding="utf-8") as fp:
        tj = json.load(fp)

    id_to_piece: dict[int, str] = {}
    piece_to_id: dict[str, int] = {}
    for at in tj.get("added_tokens", []):
        idx = int(at["id"])
        p = at["content"]
        id_to_piece[idx] = p
        piece_to_id[p] = idx

    for p, idx in tj["model"]["vocab"].items():
        idx_i = int(idx)
        id_to_piece[idx_i] = p
        piece_to_id[p] = idx_i

    kept: set[int] = set()
    for at in tj.get("added_tokens", []):
        kept.add(int(at["id"]))
    byte_id: dict[int, int] = {}
    for idx, piece in id_to_piece.items():
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                byte_id[int(piece[3:5], 16)] = idx
                kept.add(idx)
            except ValueError:
                pass
        if idx < 512:
            kept.add(idx)

    legal_extra = set("§¶—–‘’“”•°©®™†‡½¼¾×÷±≤≥≠≈…")
    allowed_chars = set(chr(c) for c in range(32, 127)) | {"\n", "\t", "\r", "\u2581"} | legal_extra

    short_and_json: list[int] = []
    ascii_candidates: list[int] = []
    for idx in sorted(id_to_piece.keys()):
        if idx in kept:
            continue
        p = id_to_piece[idx]
        if not all(ch in allowed_chars for ch in p):
            continue
        stripped = p.lstrip("\u2581")
        is_no_alpha = all(not ch.isalpha() for ch in p)
        if len(p) <= 2 or len(stripped) <= 2 or is_no_alpha:
            short_and_json.append(idx)
        else:
            ascii_candidates.append(idx)

    for idx in short_and_json:
        if len(kept) >= target_size:
            break
        kept.add(idx)

    for idx in ascii_candidates:
        if len(kept) >= target_size:
            break
        kept.add(idx)
        p = id_to_piece[idx]
        if p.startswith("\u2581"):
            bare = p[1:]
            for alt in (bare, bare.lower(), bare.capitalize()):
                alt_id = piece_to_id.get(alt)
                if alt_id is not None and alt_id not in kept and len(kept) < target_size:
                    kept.add(alt_id)

    # Tier 2 (active when target_size > ~150,849, e.g. LATIN_EXTENDED_KEPT_VOCAB_SIZE = 162,304):
    # Include all Latin-1 / Latin-Extended-A/B accented letters (U+00C0..U+024F: Raúl, Luján,
    # Velázquez, García, etc.) plus checkmarks, currency, and Greek/technical symbols without
    # displacing high-rank ASCII tokens at target_size == 150,786.
    if len(kept) < target_size:
        latin_and_sym = (
            allowed_chars
            | set(chr(c) for c in range(0x00C0, 0x0250))
            | set("✓✔✗✘«»¡¿ªº¢£€¥¤¦¨¯´¸αβγδεζηθικλμνξοπρστυφχψωΔΣΩ")
        )
        for idx in sorted(id_to_piece.keys()):
            if len(kept) >= target_size:
                break
            if idx in kept:
                continue
            p = id_to_piece[idx]
            if all(ch in latin_and_sym for ch in p):
                kept.add(idx)

    kept_sorted = sorted(kept)
    new_id_map = {old_id: new_id for new_id, old_id in enumerate(kept_sorted)}
    kept_piece_to_new = {id_to_piece[old_id]: new_id for old_id, new_id in new_id_map.items()}
    return kept_sorted, new_id_map, kept_piece_to_new, id_to_piece, byte_id


class VocabRemapper:
    """Reversible mapping of retained vocabulary IDs (`kept_old_ids` <-> contiguous `0..V_new-1`)."""

    def __init__(
        self,
        tokenizer_json_path: str | Path,
        target_size: int = DEFAULT_KEPT_VOCAB_SIZE,
        *,
        kept_old_ids: np.ndarray | None = None,
    ) -> None:
        self.tokenizer_json_path = Path(tokenizer_json_path)
        self.target_size = int(target_size)

        with self.tokenizer_json_path.open("r", encoding="utf-8") as fp:
            tj = json.load(fp)
        id_to_piece: dict[int, str] = {}
        for at in tj.get("added_tokens", []):
            id_to_piece[int(at["id"])] = at["content"]
        for p, idx in tj["model"]["vocab"].items():
            id_to_piece[int(idx)] = p

        byte_id: dict[int, int] = {}
        for idx, piece in id_to_piece.items():
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    byte_id[int(piece[3:5], 16)] = idx
                except ValueError:
                    pass

        if kept_old_ids is None:
            kept_sorted, new_id_map, kept_piece_to_new, _, byte_id = build_kept_vocab(
                self.tokenizer_json_path, self.target_size
            )
            self.kept_old_ids = np.asarray(kept_sorted, dtype=np.int32)
        else:
            self.kept_old_ids = np.asarray(kept_old_ids, dtype=np.int32)
            new_id_map = {int(old_id): new_id for new_id, old_id in enumerate(self.kept_old_ids.tolist())}
            kept_piece_to_new = {id_to_piece[int(old_id)]: new_id for old_id, new_id in new_id_map.items()}

        self.vocab_size = int(len(self.kept_old_ids))
        self.new_id_map = new_id_map
        self.kept_piece_to_new = kept_piece_to_new
        self.id_to_piece = id_to_piece
        self.byte_id = byte_id
        self.max_piece_len = max(len(p) for p in kept_piece_to_new)
        self._fallback_cache: dict[int, tuple[int, ...]] = {}

        o2n = np.full(FULL_VOCAB_SIZE, -1, dtype=np.int32)
        o2n[self.kept_old_ids] = np.arange(self.vocab_size, dtype=np.int32)
        self.old_to_new = o2n

    @classmethod
    def load_or_build(
        cls,
        tokenizer_json_path: str | Path,
        cache_npz_path: str | Path | None = None,
        target_size: int = DEFAULT_KEPT_VOCAB_SIZE,
    ) -> "VocabRemapper":
        if cache_npz_path is not None and Path(cache_npz_path).exists():
            with np.load(cache_npz_path) as data:
                kept = np.asarray(data["kept_old_ids"] if "kept_old_ids" in data else data["old_ids"], dtype=np.int32)
            if target_size is None or len(kept) == int(target_size):
                return cls(tokenizer_json_path, len(kept), kept_old_ids=kept)
        inst = cls(tokenizer_json_path, target_size=target_size)
        if cache_npz_path is not None:
            p = Path(cache_npz_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, kept_old_ids=inst.kept_old_ids)
        return inst

    def map_old_id(self, old_id: int) -> tuple[int, ...]:
        new_id = self.new_id_map.get(old_id)
        if new_id is not None:
            return (new_id,)
        cached = self._fallback_cache.get(old_id)
        if cached is not None:
            return cached
        piece = self.id_to_piece[old_id]
        res: list[int] = []
        pos = 0
        n = len(piece)
        while pos < n:
            matched = None
            upper = min(n, pos + self.max_piece_len)
            for end in range(upper, pos, -1):
                sub = piece[pos:end]
                nid = self.kept_piece_to_new.get(sub)
                if nid is not None:
                    matched = nid
                    pos = end
                    break
            if matched is not None:
                res.append(matched)
            else:
                ch = " " if piece[pos] == "\u2581" else piece[pos]
                for b in ch.encode("utf-8"):
                    res.append(self.new_id_map[self.byte_id[b]])
                pos += 1
        tup = tuple(res)
        self._fallback_cache[old_id] = tup
        return tup

    def encode_remap(self, old_ids: Sequence[int]) -> tuple[list[int], int]:
        out: list[int] = []
        misses = 0
        for oid in old_ids:
            if oid not in self.new_id_map:
                misses += 1
            out.extend(self.map_old_id(int(oid)))
        return out, misses

    def decode_to_old(self, new_ids: Sequence[int]) -> list[int]:
        return [int(self.kept_old_ids[nid]) for nid in new_ids]


class FastGemma4Tokenizer:
    """Rust `tokenizers` wrapper for Gemma-4 with chat templating and `llguidance` bridge."""

    def __init__(self, tokenizer_json_path: str | Path) -> None:
        self.tokenizer_json_path = Path(tokenizer_json_path)
        self._tok = Tokenizer.from_file(str(self.tokenizer_json_path))
        self._tok.no_padding()
        self._tok.no_truncation()
        self.vocab_size = int(self._tok.get_vocab_size(with_added_tokens=True))
        self.pad_token_id = int(self._tok.token_to_id("<pad>") or 0)
        self.eos_token_id = int(self._tok.token_to_id("<eos>") or 1)
        self.bos_token_id = int(self._tok.token_to_id("<bos>") or 2)
        self.eot_token_id = int(self._tok.token_to_id("<turn|>") or 106)
        self.tool_response_token_id = int(self._tok.token_to_id("<|tool_response>") or 50)
        self.channel_open_id: int | None = self._tok.token_to_id("<|channel>")
        self.channel_close_id: int | None = self._tok.token_to_id("<channel|>")
        self.think_token_id: int | None = self._tok.token_to_id("<|think|>")
        self.stop_token_ids = frozenset(
            {
                *GEMMA4_EOS_TOKEN_IDS,
                self.eos_token_id,
                self.eot_token_id,
                self.tool_response_token_id,
            }
        )
        self._lltok = None

    @property
    def backend(self) -> Tokenizer:
        return self._tok

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: Sequence[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        if not texts:
            return []
        return [enc.ids for enc in self._tok.encode_batch(list(texts), add_special_tokens=add_special_tokens)]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_channel_close_suffix(self) -> list[int]:
        """Encode Gemma-4's channel closing delimiter `\\n<channel|>` for bounded thinking."""
        return [int(t) for t in self.encode("\n<channel|>", add_special_tokens=False)]

    def decode_answer_and_reasoning(self, token_ids: Sequence[int]) -> tuple[str, str | None, list[int]]:
        """Strip `<|channel>...<channel|>` thinking blocks before `skip_special_tokens=True`."""
        ids = [int(t) for t in token_ids]
        open_id = getattr(self, "channel_open_id", None)
        close_id = getattr(self, "channel_close_id", None)

        if (open_id is not None and int(open_id) in ids) or (close_id is not None and int(close_id) in ids):
            reasoning_blocks: list[list[int]] = []
            cur_reasoning: list[int] = []
            answer_ids: list[int] = []
            in_channel = False
            for tid in ids:
                if open_id is not None and tid == int(open_id):
                    if in_channel and cur_reasoning:
                        reasoning_blocks.append(cur_reasoning)
                        cur_reasoning = []
                    in_channel = True
                    continue
                if close_id is not None and tid == int(close_id):
                    if in_channel:
                        if cur_reasoning:
                            reasoning_blocks.append(cur_reasoning)
                            cur_reasoning = []
                    elif answer_ids:
                        reasoning_blocks.append(answer_ids)
                        answer_ids = []
                    in_channel = False
                    continue
                if in_channel:
                    cur_reasoning.append(tid)
                else:
                    answer_ids.append(tid)
            if cur_reasoning:
                reasoning_blocks.append(cur_reasoning)
            reasoning_parts = [
                _LEADING_THOUGHT_HEADER_RE.sub("", self.decode(blk, skip_special_tokens=True), count=1).strip()
                for blk in reasoning_blocks
            ]
            cleaned_reasoning = "\n\n".join(p for p in reasoning_parts if p)
            answer_text = _strip_thinking(self.decode(answer_ids, skip_special_tokens=True)) if answer_ids else ""
            return answer_text, (cleaned_reasoning or None), answer_ids

        decoded = self.decode(ids, skip_special_tokens=True)
        if "<|channel>" in decoded or "<channel|>" in decoded or "<|thought>" in decoded or "<thought|>" in decoded:
            reasoning_parts: list[str] = []
            for m in _CHANNEL_THOUGHT_RE.finditer(decoded):
                seg = _LEADING_THOUGHT_HEADER_RE.sub("", m.group(0)[len("<|channel>") : -len("<channel|>")], count=1).strip()
                if seg:
                    reasoning_parts.append(seg)
            for m in _THOUGHT_RE.finditer(decoded):
                seg = m.group(0)[len("<|thought>") : -len("<thought|>")].strip()
                if seg:
                    reasoning_parts.append(seg)
            rem = _THOUGHT_RE.sub("", _CHANNEL_THOUGHT_RE.sub("", decoded))
            if "<channel|>" in rem:
                prefix, rem = rem.rsplit("<channel|>", 1)
                seg = _LEADING_THOUGHT_HEADER_RE.sub("", prefix, count=1).strip()
                if seg:
                    reasoning_parts.append(seg)
            if "<thought|>" in rem:
                prefix, rem = rem.rsplit("<thought|>", 1)
                seg = prefix.strip()
                if seg:
                    reasoning_parts.append(seg)
            if "<|channel>" in rem:
                _, tail = rem.split("<|channel>", 1)
                seg = _LEADING_THOUGHT_HEADER_RE.sub("", tail, count=1).strip()
                if seg:
                    reasoning_parts.append(seg)
            elif "<|thought>" in rem:
                _, tail = rem.split("<|thought>", 1)
                seg = tail.strip()
                if seg:
                    reasoning_parts.append(seg)
            reasoning_text = "\n\n".join(reasoning_parts) or None
            return _strip_thinking(decoded), reasoning_text, ids
        return decoded, None, ids

    def format_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return format_gemma4_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **_kwargs: Any,
    ) -> str | list[int]:
        text = self.format_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[int]:
        return self.encode(
            self.format_chat(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking),
            add_special_tokens=False,
        )

    def encode_chat_batch(
        self,
        conversations: Sequence[Sequence[dict[str, Any]]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[list[int]]:
        formatted = [
            self.format_chat(conv, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
            for conv in conversations
        ]
        return self.encode_batch(formatted, add_special_tokens=False)

    def llguidance_tokenizer(self):
        if self._lltok is None:
            import llguidance

            self._lltok = llguidance.LLTokenizer(self._tok.to_str(), eos_token=self.eos_token_id)
        return self._lltok
