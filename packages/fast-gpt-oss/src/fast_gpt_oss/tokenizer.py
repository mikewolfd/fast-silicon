"""Rust `tokenizers` + Harmony chat templating + lossless `o200k` output-vocab pruner (zero transformers/torch)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tokenizers import Tokenizer

DEFAULT_KEPT_VOCAB_SIZE = 131072
FULL_VOCAB_SIZE = 201088
DEFAULT_MODEL_IDENTITY = "You are ChatGPT, a large language model trained by OpenAI."


def build_o200k_kept_vocab(tokenizer_json_path: str | Path, target_size: int = DEFAULT_KEPT_VOCAB_SIZE) -> np.ndarray:
    """Build deterministic lossless-ASCII + JSON + legal-punctuation + byte-fallback + Harmony special subset."""
    with Path(tokenizer_json_path).open("r", encoding="utf-8") as fp:
        tj = json.load(fp)
    vocab = tj["model"]["vocab"]
    added = {int(at["id"]): at["content"] for at in tj.get("added_tokens", [])}
    inv_vocab = {int(v): k for k, v in vocab.items()}

    kept: set[int] = set(added.keys())
    for idx in range(min(512, len(vocab))):
        if len(kept) >= target_size:
            break
        kept.add(idx)

    # Standard GPT-2 / o200k bytes_to_unicode inverse map
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    for b in range(256):
        if len(kept) >= target_size:
            break
        ch = chr(cs[bs.index(b)])
        if ch in vocab:
            kept.add(int(vocab[ch]))

    allowed_bytes = set(range(32, 127)) | {9, 10, 13}
    for idx in range(len(vocab)):
        if len(kept) >= target_size:
            break
        if idx in kept:
            continue
        piece = inv_vocab.get(idx)
        if not piece:
            continue
        raw = bytes(u2b[c] for c in piece if c in u2b)
        if len(raw) == len(piece) and all(b in allowed_bytes for b in raw):
            kept.add(idx)
        else:
            try:
                s = raw.decode("utf-8")
                if all(32 <= ord(ch) <= 126 or ch in "\n\t\r\u00a0—–‐‑…‘’“”§°•" for ch in s):
                    kept.add(idx)
            except UnicodeDecodeError:
                pass

    if len(kept) < target_size:
        for idx in range(len(vocab)):
            if idx not in kept:
                kept.add(idx)
                if len(kept) >= target_size:
                    break

    return np.asarray(sorted(kept)[:target_size], dtype=np.int32)


class O200kVocabRemapper:
    """Asymmetric output-only `o200k_harmony` remapping (`kept_old_ids` <-> `0..V_out-1`)."""

    def __init__(
        self,
        tokenizer_json_path: str | Path,
        target_size: int = DEFAULT_KEPT_VOCAB_SIZE,
        *,
        kept_old_ids: np.ndarray | None = None,
        full_vocab_size: int | None = None,
    ) -> None:
        self.tokenizer_json_path = Path(tokenizer_json_path)
        self.target_size = int(target_size)
        if kept_old_ids is None:
            self.kept_old_ids = build_o200k_kept_vocab(self.tokenizer_json_path, target_size=self.target_size)
        else:
            self.kept_old_ids = np.asarray(kept_old_ids, dtype=np.int32)
        self.vocab_size = int(len(self.kept_old_ids))
        max_id = int(self.kept_old_ids.max()) + 1 if self.vocab_size > 0 else 0
        eff_full = max(int(full_vocab_size or FULL_VOCAB_SIZE), max_id)
        o2n = np.full(eff_full, -1, dtype=np.int32)
        if self.vocab_size > 0:
            o2n[self.kept_old_ids] = np.arange(self.vocab_size, dtype=np.int32)
        self.old_to_new = o2n

    @property
    def new_to_old(self) -> np.ndarray:
        return self.kept_old_ids

    @classmethod
    def load_or_build(
        cls,
        tokenizer_json_path: str | Path,
        cache_npz_path: str | Path | None = None,
        target_size: int = DEFAULT_KEPT_VOCAB_SIZE,
        full_vocab_size: int | None = None,
    ) -> "O200kVocabRemapper":
        if cache_npz_path is not None and Path(cache_npz_path).exists():
            data = np.load(cache_npz_path)
            kept = np.asarray(data["kept_old_ids"], dtype=np.int32)
            if target_size is None or len(kept) == int(target_size):
                return cls(tokenizer_json_path, len(kept), kept_old_ids=kept, full_vocab_size=full_vocab_size)
        inst = cls(tokenizer_json_path, target_size=target_size, full_vocab_size=full_vocab_size)
        if cache_npz_path is not None:
            p = Path(cache_npz_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, kept_old_ids=inst.kept_old_ids)
        return inst


def _extract_msg_text(content: Any) -> str:
    if isinstance(content, Sequence) and not isinstance(content, str):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def format_harmony_chat(
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    reasoning_effort: str = "low",
    model_identity: str = DEFAULT_MODEL_IDENTITY,
    current_date: str | None = None,
) -> str:
    """Format chat messages into OpenAI's official Harmony framing.

    `<|start|>system<|message|>` carries model identity, cutoff/date, `Reasoning: {effort}`,
    and `# Valid channels: analysis, commentary, final.`, while caller `system` / `developer`
    instructions go inside `<|start|>developer<|message|># Instructions\\n\\n...<|end|>`.
    """
    if not messages:
        raise ValueError("messages must be non-empty")

    eff = str(reasoning_effort or "low").lower().strip()
    if eff not in ("skip", "none", "low", "medium", "high"):
        raise ValueError(f"Unsupported reasoning_effort={reasoning_effort!r}; expected skip|none|low|medium|high")

    sys_effort = "low" if eff in ("skip", "none") else eff
    eff_date = current_date if current_date is not None else datetime.date.today().isoformat()
    sys_header = (
        f"{model_identity}\n"
        f"Knowledge cutoff: 2024-06\n"
        f"Current date: {eff_date}\n\n"
        f"Reasoning: {sys_effort}\n\n"
        f"# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    parts: list[str] = [f"<|start|>system<|message|>{sys_header}<|end|>"]

    loop_msgs = list(messages)
    dev_sections: list[str] = []
    while loop_msgs and str(loop_msgs[0].get("role", "")) in ("system", "developer"):
        txt = _extract_msg_text(loop_msgs[0].get("content", "")).strip()
        if txt:
            dev_sections.append(txt)
        loop_msgs = loop_msgs[1:]

    if dev_sections:
        dev_body = "\n\n".join(dev_sections)
        if not dev_body.startswith("# Instructions"):
            dev_body = f"# Instructions\n\n{dev_body}"
        parts.append(f"<|start|>developer<|message|>{dev_body}<|end|>")

    for msg in loop_msgs:
        role = str(msg.get("role", "user"))
        text = _extract_msg_text(msg.get("content", ""))
        if role == "assistant":
            text, _ = parse_harmony_output(text, started_in_analysis=False)
            parts.append(f"<|start|>assistant<|channel|>final<|message|>{text}<|end|>")
        else:
            parts.append(f"<|start|>{role}<|message|>{text}<|end|>")

    if add_generation_prompt:
        if eff in ("skip", "none"):
            parts.append("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>")
        else:
            parts.append("<|start|>assistant<|channel|>analysis<|message|>")
    return "".join(parts)


_STOP_MARKERS = (
    "<|return|>",
    "<|call|>",
    "<|end|>",
    "<|endoftext|>",
)


def parse_harmony_output(
    raw_text: str,
    *,
    started_in_analysis: bool = False,
    allow_analysis_fallback: bool = False,
) -> tuple[str, str | None]:
    """Split Harmony output into `(final_text, reasoning_text)` without leaking truncated analysis."""
    final_marker = "<|channel|>final<|message|>"
    ana_marker = "<|channel|>analysis<|message|>"
    reasoning: str | None = None
    body = raw_text

    if final_marker in raw_text:
        pre, body = raw_text.split(final_marker, 1)
        ana = pre
        for tok in ("<|start|>assistant", ana_marker, "<|end|>"):
            ana = ana.replace(tok, "")
        ana = ana.strip()
        reasoning = ana if ana else None
        for stop_tok in (*_STOP_MARKERS, "<|start|>"):
            if stop_tok in body:
                body = body.split(stop_tok, 1)[0]
        return body, reasoning

    if ana_marker in raw_text or started_in_analysis:
        ana = raw_text.split(ana_marker, 1)[-1] if ana_marker in raw_text else raw_text
        for tok in ("<|start|>assistant", *_STOP_MARKERS):
            ana = ana.replace(tok, "")
        ana = ana.strip()
        if allow_analysis_fallback:
            return ana, (ana or None)
        return "", (ana or None)

    for tok in _STOP_MARKERS:
        body = body.replace(tok, "")
    return body, reasoning


class FastGptOssTokenizer:
    """Rust `tokenizers` wrapper for `gpt-oss-20b` with Harmony chat & `llguidance` bridge."""

    def __init__(self, tokenizer_json_path: str | Path) -> None:
        self.tokenizer_json_path = Path(tokenizer_json_path)
        self._tok = Tokenizer.from_file(str(self.tokenizer_json_path))
        self._tok.no_padding()
        self._tok.no_truncation()
        self.vocab_size = int(self._tok.get_vocab_size(with_added_tokens=True))

        def _tid(tok: str, fallback: int) -> int:
            v = self._tok.token_to_id(tok)
            return int(v) if v is not None else int(fallback)

        self.endoftext_id = _tid("<|endoftext|>", 199999)
        self.return_token_id = _tid("<|return|>", self.endoftext_id)
        self.end_token_id = _tid("<|end|>", self.return_token_id)
        self.call_token_id = _tid("<|call|>", self.return_token_id)
        self.start_token_id = _tid("<|start|>", 0)
        self.channel_token_id = _tid("<|channel|>", 0)
        self.message_token_id = _tid("<|message|>", 0)
        self.eos_token_id = self.return_token_id
        self.final_stop_token_ids = frozenset(
            {self.return_token_id, self.end_token_id, self.call_token_id, self.endoftext_id}
        )
        self.analysis_stop_token_ids = frozenset(
            {self.return_token_id, self.call_token_id, self.endoftext_id}
        )
        self.final_channel_seq: tuple[int, ...] = tuple(
            self.encode("<|channel|>final<|message|>", add_special_tokens=False)
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

    def format_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        reasoning_effort: str = "low",
    ) -> str:
        return format_harmony_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )

    def encode_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        reasoning_effort: str = "low",
    ) -> list[int]:
        return self.encode(
            self.format_chat(messages, add_generation_prompt=add_generation_prompt, reasoning_effort=reasoning_effort),
            add_special_tokens=False,
        )

    def llguidance_tokenizer(self):
        if self._lltok is None:
            import llguidance

            self._lltok = llguidance.LLTokenizer(self._tok.to_str(), eos_token=self.eos_token_id)
        return self._lltok


__all__ = [
    "DEFAULT_KEPT_VOCAB_SIZE",
    "DEFAULT_MODEL_IDENTITY",
    "FULL_VOCAB_SIZE",
    "FastGptOssTokenizer",
    "O200kVocabRemapper",
    "build_o200k_kept_vocab",
    "format_harmony_chat",
    "parse_harmony_output",
]
