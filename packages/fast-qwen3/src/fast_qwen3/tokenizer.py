"""Rust `tokenizers` ChatML formatter, thinking-budget policy, and active-vocab remapper for `fast-qwen3`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from tokenizers import Tokenizer

DEFAULT_QWEN_VOCAB_SIZE: int = 248320
MIN_RECOMMENDED_THINKING_BUDGET: int = 384
MAX_SHORT_THINKING_VALLEY_TOKENS: int = 128

_SPECIAL_TOKEN_INJECTION_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def sanitize_user_text(text: str) -> str:
    """Neutralize accidental `<|...|>` ChatML control tokens inside untrusted document text."""
    if "<|" not in text:
        return text
    return _SPECIAL_TOKEN_INJECTION_RE.sub(r"<¦\1¦>", text)


def validate_thinking_budget(
    thinking_budget_tokens: int | None,
    *,
    enable_thinking: bool = False,
    allow_short_thinking_budget: bool = False,
) -> int:
    """Resolve and validate two-phase thinking token budget.

    Empirically on Qwen3.5 (`MATH-500-JSON` ABBA), short budgets `1 <= B <= 128` enter a
    Truncated-Thinking Valley (`28.1% -> 15.6%` at `8x` latency). Either disable thinking
    (`0` tokens, default for structured extraction) or allocate `>= 384` tokens unless
    `allow_short_thinking_budget=True` is explicitly requested.
    """
    if not enable_thinking and (thinking_budget_tokens is None or int(thinking_budget_tokens) <= 0):
        return 0
    if thinking_budget_tokens is None:
        return MIN_RECOMMENDED_THINKING_BUDGET if enable_thinking else 0
    budget = int(thinking_budget_tokens)
    if budget <= 0:
        return 0
    if budget < MIN_RECOMMENDED_THINKING_BUDGET and not allow_short_thinking_budget:
        raise ValueError(
            f"thinking_budget_tokens={budget} falls inside Qwen3.5's truncated-thinking valley "
            f"(< {MIN_RECOMMENDED_THINKING_BUDGET} tokens degrades accuracy below no-think). "
            f"Use enable_thinking=False (budget=0) or thinking_budget_tokens >= {MIN_RECOMMENDED_THINKING_BUDGET}, "
            f"or pass allow_short_thinking_budget=True to override."
        )
    return budget


def inject_reasoning_prefix_property(
    schema: str | Mapping[str, Any],
    *,
    field_name: str = "_reasoning",
    max_length: int = 1024,
) -> dict[str, Any]:
    """Prepend a bounded `field_name` string property as key 0 of an object JSON Schema."""
    obj = json.loads(schema) if isinstance(schema, str) else dict(schema)
    if obj.get("type") != "object":
        return obj
    old_props = dict(obj.get("properties") or {})
    if field_name in old_props:
        return obj
    new_props: dict[str, Any] = {
        field_name: {
            "type": "string",
            "maxLength": int(max_length),
        }
    }
    new_props.update(old_props)
    out = dict(obj)
    out["properties"] = new_props
    old_req = list(obj.get("required") or [])
    out["required"] = [field_name] + [k for k in old_req if k != field_name]
    return out


class QwenVocabRemapper:
    """Optional active-vocabulary subsetter for untied `lm_head` checkpoints (`9B`, `35B-A3B`)."""

    def __init__(self, new_to_old: np.ndarray, full_vocab_size: int) -> None:
        arr = np.asarray(new_to_old, dtype=np.int32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("new_to_old must be a non-empty 1D int32 array")
        self.new_to_old = np.unique(arr)
        self.full_vocab_size = int(full_vocab_size)
        self.old_to_new = np.full((self.full_vocab_size,), -1, dtype=np.int32)
        self.old_to_new[self.new_to_old] = np.arange(self.new_to_old.size, dtype=np.int32)

    @property
    def vocab_size(self) -> int:
        return int(self.new_to_old.size)

    @classmethod
    def build_ascii_latin_subset(
        cls,
        tokenizer: Tokenizer,
        *,
        full_vocab_size: int = DEFAULT_QWEN_VOCAB_SIZE,
        target_size: int | None = None,
    ) -> QwenVocabRemapper:
        """Keep all ASCII/Latin/JSON/punctuation/digit/special tokens plus lowest-ID BPE merges."""
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        real_tok_limit = min(int(full_vocab_size), int(tokenizer.get_vocab_size(with_added_tokens=True)))
        keep: set[int] = set()
        for tok_str, tid in vocab.items():
            if tid < 0 or tid >= real_tok_limit:
                continue
            if tid < 1024 or tok_str.startswith("<|") or tok_str in (THINK_OPEN, THINK_CLOSE):
                keep.add(int(tid))
                continue
            # Keep printable ASCII + Latin-1/Extended tokens used in Western structured text
            if all(ord(ch) < 0x0250 or ch in "ĠĊĉ—–“”‘’•§¶°±×÷€£¥¢" for ch in tok_str):
                keep.add(int(tid))
        if target_size is not None and int(target_size) > len(keep):
            cap = min(int(target_size), real_tok_limit)
            for tid in range(real_tok_limit):
                if len(keep) >= cap:
                    break
                keep.add(tid)
        ordered = np.array(sorted(keep), dtype=np.int32)
        return cls(ordered, full_vocab_size=full_vocab_size)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            new_to_old=self.new_to_old,
            full_vocab_size=np.array([self.full_vocab_size], dtype=np.int32),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> QwenVocabRemapper:
        data = np.load(Path(path))
        return cls(data["new_to_old"], full_vocab_size=int(data["full_vocab_size"][0]))


class FastQwen3Tokenizer:
    """Fast Rust `Tokenizer` wrapper with Qwen3.5/3.6 ChatML formatting and `llguidance` bridge."""

    def __init__(
        self,
        tokenizer_or_path: Tokenizer | str | Path,
        *,
        full_vocab_size: int | None = None,
    ) -> None:
        if isinstance(tokenizer_or_path, Tokenizer):
            self._tok = tokenizer_or_path
            self.tokenizer_json_path: Path | None = None
        else:
            p = Path(tokenizer_or_path).expanduser()
            tok_file = p / "tokenizer.json" if p.is_dir() else p
            self.tokenizer_json_path = tok_file.resolve()
            self._tok = Tokenizer.from_file(str(self.tokenizer_json_path))

        im_end_id = self._tok.token_to_id(IM_END)
        eot_id = self._tok.token_to_id(ENDOFTEXT)
        self.im_start_id: int | None = self._tok.token_to_id(IM_START)
        self.im_end_id: int | None = im_end_id
        self.eos_token_id: int = int(im_end_id if im_end_id is not None else (eot_id or 0))
        stop_ids = {self.eos_token_id}
        if eot_id is not None:
            stop_ids.add(int(eot_id))
        self.stop_token_ids: frozenset[int] = frozenset(stop_ids)
        self.think_open_id: int | None = self._tok.token_to_id(THINK_OPEN)
        self.think_close_id: int | None = self._tok.token_to_id(THINK_CLOSE)
        self.vocab_size: int = int(full_vocab_size or self._tok.get_vocab_size(with_added_tokens=True))
        self._llg_tok: Any | None = None

    @property
    def backend_tokenizer(self) -> Tokenizer:
        return self._tok

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [int(t) for t in self._tok.encode(text, add_special_tokens=add_special_tokens).ids]

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tok.decode([int(t) for t in token_ids], skip_special_tokens=skip_special_tokens)

    def format_chat(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
         sanitize: bool = True,
    ) -> str:
        """Render Qwen3.5/3.6 ChatML prompt with deterministic `<think>` prefix control."""
        msgs: list[tuple[str, str]] = []
        if isinstance(prompt_or_messages, str):
            if system_prompt:
                msgs.append(("system", system_prompt))
            user_txt = sanitize_user_text(prompt_or_messages) if sanitize else prompt_or_messages
            msgs.append(("user", user_txt))
        else:
            has_sys = any(str(m.get("role", "")) == "system" for m in prompt_or_messages)
            if system_prompt and not has_sys:
                msgs.append(("system", system_prompt))
            for m in prompt_or_messages:
                role = str(m.get("role", "user"))
                raw_c = m.get("content", "")
                if isinstance(raw_c, list):
                    txt = "\n".join(
                        str(part.get("text", ""))
                        for part in raw_c
                        if isinstance(part, Mapping) and part.get("type") in ("text", None)
                    )
                else:
                    txt = str(raw_c)
                if sanitize and role == "user":
                    txt = sanitize_user_text(txt)
                msgs.append((role, txt))

        chunks = [f"{IM_START}{role}\n{content}{IM_END}\n" for role, content in msgs]
        if enable_thinking:
            chunks.append(f"{IM_START}assistant\n{THINK_OPEN}\n")
        else:
            chunks.append(f"{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
        return "".join(chunks)

    def encode_chat(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        sanitize: bool = True,
    ) -> list[int]:
        rendered = self.format_chat(
            prompt_or_messages,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            sanitize=sanitize,
        )
        return self.encode(rendered, add_special_tokens=False)

    def encode_think_close_suffix(self, *, trailing_newline_before_close: bool = True) -> list[int]:
        prefix = "\n" if trailing_newline_before_close else ""
        return self.encode(f"{prefix}{THINK_CLOSE}\n\n", add_special_tokens=False)

    def llguidance_tokenizer(self) -> Any:
        if self._llg_tok is None:
            import llguidance

            if self.tokenizer_json_path is not None and self.tokenizer_json_path.is_file():
                raw_str = self.tokenizer_json_path.read_text(encoding="utf-8")
            else:
                raw_str = self._tok.to_str()
            self._llg_tok = llguidance.LLTokenizer(raw_str, n_vocab=self.vocab_size, eos_token=self.eos_token_id)
        return self._llg_tok
