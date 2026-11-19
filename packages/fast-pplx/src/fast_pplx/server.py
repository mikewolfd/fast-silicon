"""OpenAI-compatible HTTP embedding server backed by `fast_shared.server`."""

from __future__ import annotations

from typing import Any

from fast_shared.server import EmbeddingHTTPServer, make_embedding_handler

from .config import DEFAULT_MODEL_NAME


def make_handler(engine: Any, model_name: str = DEFAULT_MODEL_NAME):
    return make_embedding_handler(engine, model_name=model_name)


class FastPplxServer(EmbeddingHTTPServer):
    """Context-managed HTTP server defaulting to pplx-embed-v2-context-0.6b."""

    def __init__(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        super().__init__(engine, host=host, port=port, model_name=model_name)


__all__ = ["FastPplxServer", "make_handler"]
