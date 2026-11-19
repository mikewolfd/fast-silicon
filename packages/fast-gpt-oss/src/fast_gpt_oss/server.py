"""OpenAI-compatible HTTP inference server built DRY on `fast_shared.server`."""

from __future__ import annotations

from typing import Any

from fast_shared.server import (
    ManagedHTTPServer,
    OpenAIChatAdapter,
    make_chat_handler,
)


class FastGptOssOpenAIAdapter(OpenAIChatAdapter):
    """OpenAI chat/completion adapter for `FastGptOssEngine` (Standalone or `UnifiedOpenAIServer`)."""

    def __init__(
        self,
        engine: Any,
        *,
        model_name: str | None = None,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
    ) -> None:
        super().__init__(
            engine,
            model_name=model_name or getattr(engine, "model_name", "openai/gpt-oss-20b"),
            owned_by="openai",
            extra_key="fast_gpt_oss",
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
            default_whitespace_flexible=True,
            extra_health=lambda eng: {
                "reasoning_effort": getattr(eng, "default_reasoning_effort", "low"),
                "vocab_out": int(getattr(getattr(eng, "model", None), "vocab_n", 0)),
            },
        )


def make_gpt_oss_handler(
    engine: Any,
    *,
    model_name: str | None = None,
    coalesce_ms: float = 0.0,
    max_batch_size: int = 128,
    adapter: FastGptOssOpenAIAdapter | None = None,
):
    eff_adapter = adapter or FastGptOssOpenAIAdapter(
        engine,
        model_name=model_name,
        coalesce_ms=coalesce_ms,
        max_batch_size=max_batch_size,
    )
    return make_chat_handler(eff_adapter)


class FastGptOssServer(ManagedHTTPServer):
    """Context-managed OpenAI-compatible HTTP inference server for `gpt-oss-20b`."""

    def __init__(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str | None = None,
        *,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
    ) -> None:
        self.engine = engine
        self.adapter = FastGptOssOpenAIAdapter(
            engine,
            model_name=model_name,
            coalesce_ms=coalesce_ms,
            max_batch_size=max_batch_size,
        )
        super().__init__(make_chat_handler(self.adapter), host=host, port=port)


__all__ = ["FastGptOssOpenAIAdapter", "FastGptOssServer", "make_gpt_oss_handler"]
