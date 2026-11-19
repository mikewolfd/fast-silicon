"""OpenAI-compatible HTTP inference server built DRY on `fast_shared.server`."""

from __future__ import annotations

from typing import Any

from fast_shared.server import (
    ChatMicroBatcher as _ChatMicroBatcher,
    ManagedHTTPServer,
    OpenAIChatAdapter,
    make_chat_handler,
)

from .engine import FastGemma4Engine


class FastGemma4OpenAIAdapter(OpenAIChatAdapter):
    """Reusable OpenAI chat/completion request handler pluggable into `FastGemma4Server` or `UnifiedOpenAIServer`."""

    def __init__(
        self,
        engine: FastGemma4Engine | Any,
        *,
        model_name: str | None = None,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
    ) -> None:
        super().__init__(
            engine,
            model_name=model_name or getattr(engine, "model_name", "google/gemma-4-E2B-it"),
            owned_by="google",
            extra_key="fast_gemma4",
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
            default_whitespace_flexible=True,
        )


def make_gemma4_handler(
    engine: FastGemma4Engine | Any,
    *,
    model_name: str | None = None,
    coalesce_ms: float = 0.0,
    max_batch_size: int = 128,
    adapter: FastGemma4OpenAIAdapter | None = None,
):
    eff_adapter = adapter or FastGemma4OpenAIAdapter(
        engine,
        model_name=model_name,
        coalesce_ms=coalesce_ms,
        max_batch_size=max_batch_size,
    )
    return make_chat_handler(eff_adapter)


class FastGemma4Server(ManagedHTTPServer):
    """Context-managed OpenAI-compatible HTTP inference server for Gemma-4 E2B."""

    def __init__(
        self,
        engine: FastGemma4Engine | Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str | None = None,
        *,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
    ) -> None:
        self.engine = engine
        self.adapter = FastGemma4OpenAIAdapter(
            engine,
            model_name=model_name,
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
        )
        super().__init__(make_chat_handler(self.adapter), host=host, port=port)


__all__ = [
    "FastGemma4OpenAIAdapter",
    "FastGemma4Server",
    "_ChatMicroBatcher",
    "make_gemma4_handler",
]
