"""OpenAI-compatible HTTP inference server for `FastQwen3Engine` built DRY on `fast_shared.server`."""

from __future__ import annotations

from typing import Any

from fast_shared.server import (
    ManagedHTTPServer,
    OpenAIChatAdapter,
    make_chat_handler,
)


class FastQwen3OpenAIAdapter(OpenAIChatAdapter):
    """OpenAI `/v1/chat/completions` adapter for `FastQwen3Engine`."""

    def __init__(
        self,
        engine: Any,
        *,
        model_name: str | None = None,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 64,
    ) -> None:
        super().__init__(
            engine,
            model_name=model_name or getattr(engine, "model_name", "mlx-community/Qwen3.5-4B-OptiQ-4bit"),
            owned_by="qwen",
            extra_key="fast_qwen3",
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
            default_whitespace_flexible=None,
            extra_health=lambda eng: {
                "enable_thinking": getattr(eng, "default_enable_thinking", False),
                "has_mtp_drafter": getattr(eng, "mtp_drafter", None) is not None,
                "vocab_out": int(getattr(getattr(eng, "model", None), "vocab_n", 0)),
            },
        )


def make_qwen3_handler(
    engine: Any,
    *,
    model_name: str | None = None,
    coalesce_ms: float = 0.0,
    max_batch_size: int = 64,
    adapter: FastQwen3OpenAIAdapter | None = None,
):
    eff_adapter = adapter or FastQwen3OpenAIAdapter(
        engine,
        model_name=model_name,
        coalesce_ms=coalesce_ms,
        max_batch_size=max_batch_size,
    )
    return make_chat_handler(eff_adapter)


class FastQwen3Server(ManagedHTTPServer):
    """Context-managed OpenAI-compatible HTTP inference server for Qwen3.5 & Qwen3.6."""

    def __init__(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str | None = None,
        *,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 64,
    ) -> None:
        self.engine = engine
        self.adapter = FastQwen3OpenAIAdapter(
            engine,
            model_name=model_name,
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
        )
        super().__init__(make_chat_handler(self.adapter), host=host, port=port)


__all__ = ["FastQwen3OpenAIAdapter", "FastQwen3Server", "make_qwen3_handler"]
