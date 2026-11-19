"""OpenAI-compatible HTTP embedding server backed by `fast_shared.server`."""

from __future__ import annotations

from typing import Any

from fast_shared.server import EmbeddingHTTPServer, make_embedding_handler


def make_handler(engine: Any, model_name: str = "BAAI/bge-small-en-v1.5"):
    return make_embedding_handler(engine, model_name=model_name)


class FastBgeServer(EmbeddingHTTPServer):
    """Context-managed HTTP server defaulting to BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        super().__init__(engine, host=host, port=port, model_name=model_name)


__all__ = ["FastBgeServer", "make_handler"]
