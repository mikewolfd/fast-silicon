"""OpenAI-compatible HTTP server primitives, embedding daemon, and unified multi-model gateway."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Callable
from urllib.parse import unquote
import uuid
import weakref

import numpy as np

from .bench import clear_mlx_arena, wired_limit
from .embeddings import accepts_kwarg, resolve_embedding_dim, set_mlx_cache_limit, truncate_and_normalize

DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB

_ENGINE_LOCKS_GUARD = threading.Lock()
_ENGINE_LOCKS_WEAK: weakref.WeakKeyDictionary[Any, threading.Lock] = weakref.WeakKeyDictionary()
_ENGINE_LOCKS_FALLBACK: dict[int, threading.Lock] = {}


def get_engine_lock(engine: Any) -> threading.Lock:
    """Return a stable per-engine `threading.Lock` guarding MLX/ANE inference across HTTP worker threads."""
    existing = getattr(engine, "_fast_shared_inference_lock", None)
    if isinstance(existing, type(threading.Lock())):
        return existing
    with _ENGINE_LOCKS_GUARD:
        try:
            lock = _ENGINE_LOCKS_WEAK.get(engine)
            if lock is None:
                lock = threading.Lock()
                _ENGINE_LOCKS_WEAK[engine] = lock
            return lock
        except TypeError:
            eid = id(engine)
            lock = _ENGINE_LOCKS_FALLBACK.get(eid)
            if lock is None:
                lock = threading.Lock()
                _ENGINE_LOCKS_FALLBACK[eid] = lock
            return lock


class _PayloadTooLargeError(ValueError):
    """Raised when an HTTP request body exceeds `max_body_bytes`."""


def extract_request_param(req: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read `key` from top-level request or OpenAI SDK `extra_body`."""
    if key in req and req[key] is not None:
        return req[key]
    extra = req.get("extra_body")
    if isinstance(extra, dict) and key in extra and extra[key] is not None:
        return extra[key]
    return default


def extract_json_schema(req: dict[str, Any]) -> Any | None:
    """Extract JSON Schema from OpenAI `response_format` or `json_schema` fields."""
    rf = extract_request_param(req, "response_format")
    if isinstance(rf, dict):
        rtype = rf.get("type")
        if rtype == "json_schema" and isinstance(rf.get("json_schema"), dict):
            return rf["json_schema"].get("schema")
        if rtype == "json_object":
            return {"type": "object"}
    return extract_request_param(req, "json_schema")


class JsonHTTPHandler(BaseHTTPRequestHandler):
    """Base HTTP handler providing JSON parsing/responses (with size limit & chunked body support) and SSE streaming."""

    protocol_version = "HTTP/1.1"
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep stdout clean in library/CLI runs

    def _read_exact(self, n: int) -> bytes:
        chunks: list[bytes] = []
        rem = n
        while rem > 0:
            data = self.rfile.read(rem)
            if not data:
                raise ValueError("Truncated request body")
            chunks.append(data)
            rem -= len(data)
        return b"".join(chunks)

    def _read_chunked_body(self) -> bytes:
        chunks: list[bytes] = []
        total = 0
        limit = int(self.max_body_bytes)
        while True:
            size_line = self.rfile.readline(4096)
            if not size_line:
                raise ValueError("Truncated chunked request body")
            size_hex = size_line.split(b";", 1)[0].strip()
            try:
                chunk_size = int(size_hex, 16)
            except ValueError as exc:
                raise ValueError(f"Invalid chunk size {size_hex!r}") from exc
            if chunk_size < 0:
                raise ValueError("Negative chunk size")
            if chunk_size == 0:
                # Consume optional trailer headers up to blank line
                while True:
                    trailer = self.rfile.readline(4096)
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                break
            total += chunk_size
            if total > limit:
                raise _PayloadTooLargeError(f"Request body exceeds limit ({limit} bytes)")
            chunks.append(self._read_exact(chunk_size))
            crlf = self.rfile.readline(4)
            if crlf not in (b"\r\n", b"\n"):
                raise ValueError("Malformed chunk terminator")
        return b"".join(chunks)

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            te = str(self.headers.get("Transfer-Encoding", "")).lower()
            if "chunked" in te:
                raw_bytes = self._read_chunked_body()
                if not raw_bytes:
                    raise ValueError("Missing request body")
            else:
                raw_len = self.headers.get("Content-Length")
                if raw_len is None:
                    raise ValueError("Missing Content-Length header")
                length = int(raw_len)
                if length <= 0:
                    raise ValueError("Missing request body")
                if length > int(self.max_body_bytes):
                    self.close_connection = True
                    raise _PayloadTooLargeError(f"Request body ({length} bytes) exceeds limit ({self.max_body_bytes} bytes)")
                raw_bytes = self._read_exact(length)
            req = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("JSON body must be an object")
            return req
        except _PayloadTooLargeError as exc:
            self.close_connection = True
            self._send_error(413, str(exc))
            return None
        except Exception as exc:
            self._send_error(400, f"Invalid JSON request: {exc}")
            return None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: int, message: str, *, err_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type}})

    def _send_sse(self, events: Iterable[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for ev in events:
            chunk = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class _ReuseAddrThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class ManagedHTTPServer:
    """Context-managed `ThreadingHTTPServer` supporting both `start`/`stop` and `start_background`/`close`."""

    def __init__(
        self,
        handler_cls: type[BaseHTTPRequestHandler],
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        mlx_cache_limit_mb: int | None = 512,
        use_wired_limit: bool = False,
    ) -> None:
        self.httpd = _ReuseAddrThreadingHTTPServer((host, port), handler_cls)
        addr = self.httpd.server_address
        self.host = str(addr[0])
        self.port = int(addr[1])
        self._thread: threading.Thread | None = None
        self._closed = False
        self._mlx_cache_limit_mb = mlx_cache_limit_mb
        self._use_wired_limit = bool(use_wired_limit)
        self._runtime_stack: ExitStack | None = None
        self._prev_cache_limit_bytes: int | None = None

    def _enter_runtime_limits(self) -> None:
        if self._runtime_stack is not None:
            return
        stack = ExitStack()
        if self._mlx_cache_limit_mb is not None:
            try:
                import mlx.core as mx

                set_limit = getattr(mx, "set_cache_limit", None)
                if callable(set_limit):
                    prev = set_limit(max(0, int(self._mlx_cache_limit_mb)) * 1024 * 1024)
                    if isinstance(prev, int):
                        self._prev_cache_limit_bytes = prev
            except Exception:
                set_mlx_cache_limit(int(self._mlx_cache_limit_mb))
        if self._use_wired_limit:
            stack.enter_context(wired_limit())
        self._runtime_stack = stack

    def _exit_runtime_limits(self) -> None:
        stack = self._runtime_stack
        self._runtime_stack = None
        if stack is not None:
            stack.close()
        if self._prev_cache_limit_bytes is not None:
            prev = self._prev_cache_limit_bytes
            self._prev_cache_limit_bytes = None
            try:
                import mlx.core as mx

                set_limit = getattr(mx, "set_cache_limit", None)
                if callable(set_limit):
                    set_limit(prev)
            except Exception:
                pass
        clear_mlx_arena(256)

    def start(self) -> ManagedHTTPServer:
        if self._thread is None:
            self._enter_runtime_limits()
            self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def start_background(self) -> None:
        self.start()

    def serve_forever(self) -> None:
        self._enter_runtime_limits()
        try:
            self.httpd.serve_forever()
        finally:
            self._exit_runtime_limits()

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._exit_runtime_limits()

    def __enter__(self) -> ManagedHTTPServer:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def embedding_health_payload(engine: Any, model_name: str | None = None, *, default_mode: str | None = None) -> dict[str, Any]:
    eff_model = model_name or getattr(engine, "model_name", "fast-embedding")
    payload: dict[str, Any] = {
        "status": "ok",
        "model": eff_model,
        "dim": int(getattr(engine, "dim")),
        "ane": bool(getattr(engine, "ane_available", False)),
        "cpu_sme": bool(getattr(engine, "cpu_sme_available", False)),
        "default_mode": default_mode or getattr(engine, "default_mode", "gpu"),
    }
    if getattr(engine, "max_dim", None) is not None:
        payload["max_dim"] = int(getattr(engine, "max_dim"))
    if getattr(engine, "mrl_dimensions", None):
        payload["mrl_dimensions"] = list(getattr(engine, "mrl_dimensions"))
    return payload


def run_embedding_request(
    engine: Any,
    req: dict[str, Any],
    *,
    model_name: str | None = None,
    default_mode: str | None = None,
    lock: threading.Lock | None = None,
) -> tuple[int, dict[str, Any]]:
    eff_model = model_name or getattr(engine, "model_name", "fast-embedding")
    max_seq_len = int(getattr(engine, "max_seq_len", 512))
    ane_query_max_len = int(getattr(engine, "ane_query_max_len", 512))

    inp = req.get("input")
    if isinstance(inp, str):
        texts = [inp]
        is_single_query = bool(req.get("is_query", False))
    elif isinstance(inp, list) and all(isinstance(x, str) for x in inp):
        texts = inp
        is_single_query = bool(req.get("is_query", False))
    else:
        return 400, {"error": {"message": "`input` must be a string or list of strings", "type": "invalid_request_error"}}

    enc_fmt = str(req.get("encoding_format") or "float").lower()
    if enc_fmt not in ("float", "base64"):
        return 400, {"error": {"message": f"Unsupported encoding_format={enc_fmt!r}", "type": "invalid_request_error"}}

    default_dim = int(getattr(engine, "dim"))
    max_dim = int(getattr(engine, "max_dim", default_dim))
    supports_dims = bool(getattr(engine, "supports_dimensions", False))
    req_dims = extract_request_param(req, "dimensions")
    try:
        target_dim = resolve_embedding_dim(
            req_dims,
            max_dim=max_dim,
            default_dim=default_dim,
            supports_dimensions=supports_dims,
        )
    except ValueError as exc:
        return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}

    req_model = str(req.get("model") or eff_model)
    if not texts:
        return 200, {
            "object": "list",
            "data": [],
            "model": req_model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    add_prefix = bool(req.get("add_query_prefix", False))
    add_doc_prefix = bool(req.get("add_document_prefix", False))
    prefix = req.get("prefix", None)
    prompt_name = req.get("prompt_name", None)
    req_mode = req.get("mode", None)
    mode = req_mode if req_mode is not None else default_mode

    if hasattr(engine.tokenizer, "resolve_prefix"):
        eff_prefix = engine.tokenizer.resolve_prefix(
            prefix=prefix,
            prompt_name=prompt_name,
            add_query_prefix=add_prefix,
            add_document_prefix=add_doc_prefix,
        )
        tok_lists = engine.tokenizer.encode_batch(
            texts,
            prefix=eff_prefix,
            max_length=max_seq_len,
        )
    else:
        tok_lists = engine.tokenizer.encode_batch(
            texts,
            add_query_prefix=add_prefix,
            max_length=max_seq_len,
        )

    prompt_tokens = sum(len(x) for x in tok_lists)
    eng_lock = lock if lock is not None else get_engine_lock(engine)
    with eng_lock:
        if len(tok_lists) == 1 and (is_single_query or len(tok_lists[0]) <= ane_query_max_len):
            q_kw: dict[str, Any] = {}
            if req_dims is not None and accepts_kwarg(engine.encode_query, "dimensions"):
                q_kw["dimensions"] = target_dim
            if mode is not None:
                if accepts_kwarg(engine.encode_query, "mode"):
                    q_kw["mode"] = mode
                elif accepts_kwarg(engine.encode_query, "prefer_ane"):
                    if mode == "gpu":
                        q_kw["prefer_ane"] = False
                    elif mode == "ane":
                        q_kw["prefer_ane"] = True
            vecs = engine.encode_query(tok_lists[0], **q_kw)[None, :]
        else:
            c_kw: dict[str, Any] = {"mode": mode}
            if req_dims is not None and accepts_kwarg(engine.encode_corpus, "dimensions"):
                c_kw["dimensions"] = target_dim
            vecs = engine.encode_corpus(tok_lists, **c_kw)
        clear_mlx_arena(256)

    vecs = truncate_and_normalize(vecs, target_dim)
    arr = np.asarray(vecs, dtype="<f4")
    if enc_fmt == "base64":
        data = [
            {"object": "embedding", "index": i, "embedding": base64.b64encode(arr[i].tobytes()).decode("ascii")}
            for i in range(arr.shape[0])
        ]
    else:
        vecs_list = arr.astype(np.float32).tolist()
        data = [{"object": "embedding", "index": i, "embedding": vecs_list[i]} for i in range(len(vecs_list))]
    return 200, {
        "object": "list",
        "data": data,
        "model": req_model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def make_embedding_handler(
    engine: Any,
    *,
    model_name: str | None = None,
    mode: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
):
    eff_model = model_name or getattr(engine, "model_name", "fast-embedding")
    eng_lock = get_engine_lock(engine)

    class EmbeddingHandler(JsonHTTPHandler):
        pass

    EmbeddingHandler.max_body_bytes = int(max_body_bytes)

    def do_GET(self: JsonHTTPHandler) -> None:
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, embedding_health_payload(engine, eff_model, default_mode=mode))
        elif self.path == "/v1/models":
            self._send_json(
                200,
                {"object": "list", "data": [{"id": eff_model, "object": "model", "owned_by": "local"}]},
            )
        elif self.path.startswith("/v1/models/"):
            req_id = unquote(self.path[len("/v1/models/") :])
            if req_id == eff_model:
                self._send_json(200, {"id": eff_model, "object": "model", "owned_by": "local"})
            else:
                self._send_error(404, f"Model `{req_id}` not found")
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self: JsonHTTPHandler) -> None:
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self._send_error(404, f"Not found: {self.path}")
            return
        req = self._read_json_body()
        if req is None:
            return
        try:
            status, payload = run_embedding_request(
                engine,
                req,
                model_name=eff_model,
                default_mode=mode,
                lock=eng_lock,
            )
            self._send_json(status, payload)
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            self._send_error(500, str(exc), err_type="server_error")

    EmbeddingHandler.do_GET = do_GET  # type: ignore[method-assign]
    EmbeddingHandler.do_POST = do_POST  # type: ignore[method-assign]
    return EmbeddingHandler


class EmbeddingHTTPServer(ManagedHTTPServer):
    """Context-managed wrapper around `ThreadingHTTPServer` for clean startup and shutdown."""

    def __init__(
        self,
        engine: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_name: str | None = None,
        mode: str | None = None,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        mlx_cache_limit_mb: int | None = 512,
        use_wired_limit: bool = False,
    ) -> None:
        self.engine = engine
        self.mode = mode
        super().__init__(
            make_embedding_handler(
                engine,
                model_name=model_name,
                mode=mode,
                max_body_bytes=max_body_bytes,
            ),
            host=host,
            port=port,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )


def build_chat_completion(
    req_model: str,
    item: Any,
    *,
    stats: dict[str, Any] | None = None,
    extra_key: str | None = None,
) -> dict[str, Any]:
    """Format an OpenAI `/v1/chat/completions` JSON response."""
    msg: dict[str, Any] = {"role": "assistant", "content": item.text}
    reasoning = getattr(item, "reasoning", None)
    if reasoning:
        msg["reasoning"] = reasoning
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req_model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": item.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(item.prompt_tokens),
            "completion_tokens": int(item.completion_tokens),
            "total_tokens": int(item.prompt_tokens + item.completion_tokens),
        },
    }
    if extra_key and stats is not None:
        payload[extra_key] = stats
    return payload


def build_text_completion(
    req_model: str,
    items: list[Any],
    *,
    stats: dict[str, Any] | None = None,
    extra_key: str | None = None,
) -> dict[str, Any]:
    """Format an OpenAI `/v1/completions` JSON response (single or batched prompts)."""
    prompt_toks = sum(int(it.prompt_tokens) for it in items)
    comp_toks = sum(int(it.completion_tokens) for it in items)
    payload: dict[str, Any] = {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req_model,
        "choices": [
            {"index": i, "text": it.text, "finish_reason": it.finish_reason}
            for i, it in enumerate(items)
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
        },
    }
    if extra_key and stats is not None:
        payload[extra_key] = stats
    return payload


def iter_chat_sse_events(
    req_model: str,
    item: Any,
    *,
    include_usage: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI `chat.completion.chunk` SSE events from a generated item."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    events: list[dict[str, Any]] = [
        {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req_model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req_model,
            "choices": [{"index": 0, "delta": {"content": item.text}, "finish_reason": None}],
        },
        {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": item.finish_reason}],
        },
    ]
    if include_usage:
        events[-1]["usage"] = {
            "prompt_tokens": int(item.prompt_tokens),
            "completion_tokens": int(item.completion_tokens),
            "total_tokens": int(item.prompt_tokens + item.completion_tokens),
        }
    return events


def iter_completion_sse_events(
    req_model: str,
    items: list[Any],
    *,
    include_usage: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI `text_completion` SSE events from generated item(s)."""
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    events: list[dict[str, Any]] = [
        {
            "id": cmpl_id,
            "object": "text_completion",
            "created": created,
            "model": req_model,
            "choices": [
                {"index": i, "text": it.text, "finish_reason": it.finish_reason}
                for i, it in enumerate(items)
            ],
        }
    ]
    if include_usage:
        prompt_toks = sum(int(it.prompt_tokens) for it in items)
        comp_toks = sum(int(it.completion_tokens) for it in items)
        events[0]["usage"] = {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
        }
    return events


def schema_cache_key(schema: Any) -> str | None:
    if schema is None:
        return None
    if isinstance(schema, str):
        return schema
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)


def _gen_kwargs(req: dict[str, Any], *, default_whitespace_flexible: bool = True) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "json_schema": extract_json_schema(req),
        "max_tokens": int(extract_request_param(req, "max_tokens", 512)),
        "temperature": float(extract_request_param(req, "temperature", 0.0)),
        "top_p": float(extract_request_param(req, "top_p", 1.0)),
        "seed": int(extract_request_param(req, "seed", 42)),
        "mode": extract_request_param(req, "mode"),
        "speculative": extract_request_param(req, "speculative"),
        "whitespace_flexible": bool(extract_request_param(req, "whitespace_flexible", default_whitespace_flexible)),
        "enable_thinking": bool(extract_request_param(req, "enable_thinking", False)),
    }
    sys_prompt = extract_request_param(req, "system_prompt")
    if sys_prompt is not None:
        kw["system_prompt"] = str(sys_prompt)
    batch_size = extract_request_param(req, "batch_size")
    if batch_size is not None:
        kw["batch_size"] = int(batch_size)
    if extract_request_param(req, "compact") is not None:
        kw["compact"] = bool(extract_request_param(req, "compact"))
    reasoning_effort = extract_request_param(req, "reasoning_effort")
    if reasoning_effort is not None:
        kw["reasoning_effort"] = str(reasoning_effort)
    reasoning_budget = extract_request_param(req, "reasoning_budget")
    if reasoning_budget is not None:
        kw["reasoning_budget"] = int(reasoning_budget)
    return kw


@dataclass
class _PendingChatRequest:
    key: tuple[Any, ...]
    prompts: list[Any]
    max_tokens: int
    kwargs: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    items: list[Any] | None = None
    stats: dict[str, Any] | None = None
    error: BaseException | None = None


class ChatMicroBatcher:
    """Leader-follower micro-batcher coalescing concurrent chat/completion requests without background threads.

    Coalesces compatible requests using `max(max_tokens)` across the batch rather than requiring
    exact `max_tokens` equality, and serializes `engine.generate_batch` calls under a per-engine lock.
    """

    def __init__(
        self,
        engine: Any,
        *,
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
        lock: threading.Lock | None = None,
    ) -> None:
        self.engine = engine
        self.coalesce_sec = max(0.0, float(coalesce_ms)) / 1000.0
        self.max_batch_size = max(1, int(max_batch_size))
        self._engine_lock = lock if lock is not None else get_engine_lock(engine)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: list[_PendingChatRequest] = []
        self._active_token: int | None = None
        self._seq = 0

    def _build_entry(
        self,
        prompts: list[Any],
        *,
        kind: str,
        json_schema: Any | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool = True,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        compact: bool | None = None,
        **extra_kwargs: Any,
    ) -> _PendingChatRequest:
        eff_seed = 0 if float(temperature) == 0.0 else int(seed)
        clean_extra = {k: v for k, v in extra_kwargs.items() if v is not None}
        # Note: `max_tokens` is deliberately excluded from the coalescing key and merged via `max(max_tokens)`.
        key = (
            kind,
            schema_cache_key(json_schema),
            float(temperature),
            float(top_p),
            eff_seed,
            mode,
            speculative,
            bool(whitespace_flexible),
            bool(use_chat_template),
            bool(enable_thinking),
            system_prompt,
            compact,
            tuple(sorted(clean_extra.items())),
        )
        kwargs: dict[str, Any] = {
            "json_schema": json_schema,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "mode": mode,
            "speculative": speculative,
            "whitespace_flexible": bool(whitespace_flexible),
            "use_chat_template": bool(use_chat_template),
            "enable_thinking": bool(enable_thinking),
            **clean_extra,
        }
        if compact is not None:
            kwargs["compact"] = bool(compact)
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        return _PendingChatRequest(key=key, prompts=prompts, max_tokens=int(max_tokens), kwargs=kwargs)

    def _enqueue_and_wait(self, entry: _PendingChatRequest) -> tuple[list[Any], dict[str, Any]]:
        leader_token: int | None = None
        with self._cv:
            self._pending.append(entry)
            if self._active_token is None:
                self._seq += 1
                leader_token = self._seq
                self._active_token = leader_token
            else:
                self._cv.notify_all()

        if leader_token is not None:
            try:
                self._drain_loop(wait_initial=(self.coalesce_sec > 0.0))
            finally:
                with self._cv:
                    if self._active_token == leader_token:
                        self._active_token = None
                        for leftover in self._pending:
                            leftover.error = RuntimeError("Batch leader exited unexpectedly")
                            leftover.done.set()
                        self._pending.clear()
                    self._cv.notify_all()

        entry.done.wait()
        if entry.error is not None:
            raise entry.error
        return (entry.items or []), (entry.stats or {})

    def submit(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: Any | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool = True,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        compact: bool | None = None,
        **extra_kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        entry = self._build_entry(
            [messages],
            kind="chat",
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            mode=mode,
            speculative=speculative,
            whitespace_flexible=whitespace_flexible,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            compact=compact,
            **extra_kwargs,
        )
        items, stats = self._enqueue_and_wait(entry)
        return items[0], stats

    def submit_prompts(
        self,
        prompts: list[str],
        *,
        json_schema: Any | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        mode: str | None = None,
        speculative: bool | None = None,
        whitespace_flexible: bool = True,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        compact: bool | None = None,
        **extra_kwargs: Any,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Submit one or more text-completion prompts through the serialized micro-batcher."""
        entry = self._build_entry(
            list(prompts),
            kind="completion",
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            mode=mode,
            speculative=speculative,
            whitespace_flexible=whitespace_flexible,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            compact=compact,
            **extra_kwargs,
        )
        return self._enqueue_and_wait(entry)

    def _drain_loop(self, *, wait_initial: bool) -> None:
        first_round = True
        while True:
            with self._cv:
                if first_round and wait_initial and sum(len(r.prompts) for r in self._pending) < self.max_batch_size:
                    deadline = time.monotonic() + self.coalesce_sec
                    while sum(len(r.prompts) for r in self._pending) < self.max_batch_size:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            break
                        self._cv.wait(timeout=rem)
                first_round = False

                if not self._pending:
                    self._active_token = None
                    return

                head = self._pending.pop(0)
                batch = [head]
                total_prompts = len(head.prompts)
                keep: list[_PendingChatRequest] = []
                for req in self._pending:
                    if total_prompts + len(req.prompts) <= self.max_batch_size and req.key == head.key:
                        batch.append(req)
                        total_prompts += len(req.prompts)
                    else:
                        keep.append(req)
                self._pending = keep

            flat_prompts: list[Any] = []
            for req in batch:
                flat_prompts.extend(req.prompts)
            batch_max_tokens = max(req.max_tokens for req in batch)
            run_kwargs = dict(head.kwargs)
            run_kwargs["max_tokens"] = batch_max_tokens

            try:
                with self._engine_lock:
                    res = self.engine.generate_batch(flat_prompts, **run_kwargs)
                    clear_mlx_arena(256)
                if len(res.items) != len(flat_prompts):
                    raise RuntimeError(f"Expected {len(flat_prompts)} items from generate_batch, got {len(res.items)}")
                base_stats = dict(getattr(res, "stats", {}) or {})
                base_stats["coalesced_batch_size"] = len(flat_prompts)
                base_stats["batch_max_tokens"] = batch_max_tokens
                cursor = 0
                for req in batch:
                    n = len(req.prompts)
                    req.items = list(res.items[cursor : cursor + n])
                    cursor += n
                    req.stats = dict(base_stats)
                    req.done.set()
            except BaseException as exc:
                for req in batch:
                    req.error = exc
                    req.done.set()


class OpenAIChatAdapter:
    """Reusable OpenAI chat/completion request handler pluggable into standalone or `UnifiedOpenAIServer`."""

    def __init__(
        self,
        engine: Any,
        *,
        model_name: str | None = None,
        owned_by: str = "local",
        extra_key: str = "fast_chat",
        coalesce_ms: float = 0.0,
        max_batch_size: int = 128,
        default_whitespace_flexible: bool = True,
        extra_health: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.model_name = model_name or getattr(engine, "model_name", "fast-chat")
        self.owned_by = owned_by
        self.extra_key = extra_key
        self.default_whitespace_flexible = default_whitespace_flexible
        self._extra_health = extra_health
        self.batcher = ChatMicroBatcher(engine, coalesce_ms=coalesce_ms, max_batch_size=max_batch_size)

    def health_info(self) -> dict[str, Any]:
        model_obj = getattr(self.engine, "model", None)
        payload: dict[str, Any] = {
            "status": "ok",
            "model": self.model_name,
            "vocab_size": int(getattr(model_obj, "vocab_size", getattr(model_obj, "vocab_n", 0))) if model_obj is not None else 0,
            "ane": bool(getattr(self.engine, "ane_available", False)),
            "default_mode": getattr(self.engine, "default_mode", "auto"),
            "coalesce_ms": round(self.batcher.coalesce_sec * 1000.0, 3),
            "max_batch_size": self.batcher.max_batch_size,
        }
        if model_obj is not None and hasattr(model_obj, "lm_head_mode"):
            payload["lm_head_mode"] = getattr(model_obj, "lm_head_mode")
        if self._extra_health is not None:
            payload.update(self._extra_health(self.engine))
        return payload

    def handle_chat_completion(
        self,
        req: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]], bool]:
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return 400, {"error": {"message": "`messages` must be a non-empty list", "type": "invalid_request_error"}}, False

        req_model = str(req.get("model") or model_name or self.model_name)
        kw = _gen_kwargs(req, default_whitespace_flexible=self.default_whitespace_flexible)
        kw["use_chat_template"] = bool(extract_request_param(req, "use_chat_template", True))
        if "compact" not in kw:
            kw["compact"] = True
        kw.pop("batch_size", None)
        item, stats = self.batcher.submit(msgs, **kw)
        if bool(req.get("stream", False)):
            sopts = req.get("stream_options") or {}
            return 200, iter_chat_sse_events(req_model, item, include_usage=bool(sopts.get("include_usage", False))), True
        return 200, build_chat_completion(req_model, item, stats=stats, extra_key=self.extra_key), False

    def handle_completion(
        self,
        req: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]], bool]:
        prompt = req.get("prompt")
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt):
            prompts = prompt
        else:
            return 400, {"error": {"message": "`prompt` must be a non-empty string or list of strings", "type": "invalid_request_error"}}, False

        req_model = str(req.get("model") or model_name or self.model_name)
        kw = _gen_kwargs(req, default_whitespace_flexible=self.default_whitespace_flexible)
        kw["use_chat_template"] = bool(extract_request_param(req, "use_chat_template", True))
        if "compact" not in kw:
            kw["compact"] = True
        items, stats = self.batcher.submit_prompts(prompts, **kw)
        if bool(req.get("stream", False)):
            sopts = req.get("stream_options") or {}
            return 200, iter_completion_sse_events(req_model, items, include_usage=bool(sopts.get("include_usage", False))), True
        return 200, build_text_completion(req_model, items, stats=stats, extra_key=self.extra_key), False


def make_chat_handler(adapter: OpenAIChatAdapter, *, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES):
    """Build standalone OpenAI chat/completion HTTP handler from an `OpenAIChatAdapter`."""

    class ChatHandler(JsonHTTPHandler):
        pass

    ChatHandler.max_body_bytes = int(max_body_bytes)

    def do_GET(self: JsonHTTPHandler) -> None:
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, adapter.health_info())
        elif self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": adapter.model_name, "object": "model", "owned_by": adapter.owned_by}],
                },
            )
        elif self.path.startswith("/v1/models/"):
            req_id = unquote(self.path[len("/v1/models/") :])
            if req_id == adapter.model_name:
                self._send_json(200, {"id": adapter.model_name, "object": "model", "owned_by": adapter.owned_by})
            else:
                self._send_error(404, f"Model `{req_id}` not found")
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self: JsonHTTPHandler) -> None:
        if self.path not in ("/v1/chat/completions", "/v1/completions", "/chat/completions", "/completions"):
            self._send_error(404, f"Not found: {self.path}")
            return
        req = self._read_json_body()
        if req is None:
            return
        try:
            if self.path.endswith("/chat/completions"):
                status, payload, is_sse = adapter.handle_chat_completion(req)
            else:
                status, payload, is_sse = adapter.handle_completion(req)

            if is_sse and status == 200:
                self._send_sse(payload)  # type: ignore[arg-type]
            else:
                self._send_json(status, payload)  # type: ignore[arg-type]
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            self._send_error(500, str(exc), err_type="server_error")

    ChatHandler.do_GET = do_GET  # type: ignore[method-assign]
    ChatHandler.do_POST = do_POST  # type: ignore[method-assign]
    return ChatHandler


def dispatch_chat_completion(
    engine_or_adapter: Any,
    req: dict[str, Any],
    *,
    model_name: str | None = None,
    extra_key: str = "fast_gemma4",
    lock: threading.Lock | None = None,
) -> tuple[int, dict[str, Any] | list[dict[str, Any]], bool]:
    """Run `/v1/chat/completions` via adapter method or locked `engine.generate_batch`."""
    if hasattr(engine_or_adapter, "handle_chat_completion"):
        return engine_or_adapter.handle_chat_completion(req, model_name=model_name)

    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return 400, {"error": {"message": "`messages` must be a non-empty list", "type": "invalid_request_error"}}, False

    eff_model = str(req.get("model") or model_name or getattr(engine_or_adapter, "model_name", "fast-chat"))
    eng_lock = lock if lock is not None else get_engine_lock(engine_or_adapter)
    with eng_lock:
        res = engine_or_adapter.generate_batch([msgs], **_gen_kwargs(req))
        clear_mlx_arena(256)
    item = res.items[0]
    stream = bool(req.get("stream", False))
    if stream:
        sopts = req.get("stream_options") or {}
        return 200, iter_chat_sse_events(eff_model, item, include_usage=bool(sopts.get("include_usage", False))), True
    return 200, build_chat_completion(eff_model, item, stats=res.stats, extra_key=extra_key), False


def dispatch_completion(
    engine_or_adapter: Any,
    req: dict[str, Any],
    *,
    model_name: str | None = None,
    extra_key: str = "fast_gemma4",
    lock: threading.Lock | None = None,
) -> tuple[int, dict[str, Any] | list[dict[str, Any]], bool]:
    """Run `/v1/completions` via adapter method or locked `engine.generate_batch`."""
    if hasattr(engine_or_adapter, "handle_completion"):
        return engine_or_adapter.handle_completion(req, model_name=model_name)

    prompt = req.get("prompt")
    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt):
        prompts = prompt
    else:
        return 400, {"error": {"message": "`prompt` must be a non-empty string or list of strings", "type": "invalid_request_error"}}, False

    eff_model = str(req.get("model") or model_name or getattr(engine_or_adapter, "model_name", "fast-chat"))
    kw = _gen_kwargs(req)
    kw["use_chat_template"] = bool(extract_request_param(req, "use_chat_template", True))
    eng_lock = lock if lock is not None else get_engine_lock(engine_or_adapter)
    with eng_lock:
        res = engine_or_adapter.generate_batch(prompts, **kw)
        clear_mlx_arena(256)
    stream = bool(req.get("stream", False))
    if stream:
        sopts = req.get("stream_options") or {}
        return 200, iter_completion_sse_events(eff_model, res.items, include_usage=bool(sopts.get("include_usage", False))), True
    return 200, build_text_completion(eff_model, res.items, stats=res.stats, extra_key=extra_key), False


def _resolve_route_target(
    registry: dict[str, Any],
    default_id: str | None,
    requested_id: str | None,
    kind: str,
) -> tuple[str | None, Any | None, tuple[int, dict[str, Any]] | None]:
    if not registry:
        return None, None, (404, {"error": {"message": f"No {kind} models registered", "type": "invalid_request_error"}})
    if requested_id:
        if requested_id in registry:
            return requested_id, registry[requested_id], None
        return None, None, (
            404,
            {"error": {"message": f"Model `{requested_id}` not found in {kind} registry", "type": "invalid_request_error"}},
        )
    chosen = default_id or next(iter(registry))
    return chosen, registry[chosen], None


def make_unified_handler(
    *,
    embedding_engines: dict[str, Any] | None = None,
    chat_engines: dict[str, Any] | None = None,
    default_embedding_model: str | None = None,
    default_chat_model: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
):
    """Build an OpenAI-compatible HTTP handler routing embeddings and chat/completions on one port."""
    emb_map = dict(embedding_engines or {})
    chat_map = dict(chat_engines or {})
    emb_locks = {mid: get_engine_lock(eng) for mid, eng in emb_map.items()}
    chat_locks = {
        mid: get_engine_lock(getattr(obj, "engine", obj))
        for mid, obj in chat_map.items()
    }

    class UnifiedHandler(JsonHTTPHandler):
        pass

    UnifiedHandler.max_body_bytes = int(max_body_bytes)

    def do_GET(self: JsonHTTPHandler) -> None:
        if self.path in ("/health", "/v1/health"):
            emb_info = {mid: embedding_health_payload(eng, mid) for mid, eng in emb_map.items()}
            chat_info: dict[str, Any] = {}
            for mid, obj in chat_map.items():
                if hasattr(obj, "health_info"):
                    chat_info[mid] = obj.health_info()
                else:
                    chat_info[mid] = {
                        "status": "ok",
                        "model": mid,
                        "ane": bool(getattr(obj, "ane_available", False)),
                        "default_mode": getattr(obj, "default_mode", "auto"),
                    }
            self._send_json(
                200,
                {
                    "status": "ok",
                    "embedding_models": emb_info,
                    "chat_models": chat_info,
                },
            )
        elif self.path == "/v1/models":
            data = [
                {"id": mid, "object": "model", "owned_by": "local", "capabilities": ["embeddings"]}
                for mid in emb_map
            ] + [
                {"id": mid, "object": "model", "owned_by": "local", "capabilities": ["chat", "completions"]}
                for mid in chat_map
            ]
            self._send_json(200, {"object": "list", "data": data})
        elif self.path.startswith("/v1/models/"):
            req_id = unquote(self.path[len("/v1/models/") :])
            if req_id in emb_map:
                self._send_json(200, {"id": req_id, "object": "model", "owned_by": "local", "capabilities": ["embeddings"]})
            elif req_id in chat_map:
                self._send_json(200, {"id": req_id, "object": "model", "owned_by": "local", "capabilities": ["chat", "completions"]})
            else:
                self._send_error(404, f"Model `{req_id}` not found")
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self: JsonHTTPHandler) -> None:
        req = self._read_json_body()
        if req is None:
            return
        req_model = req.get("model")
        try:
            if self.path in ("/v1/embeddings", "/embeddings"):
                mid, eng, err = _resolve_route_target(emb_map, default_embedding_model, req_model, "embedding")
                if err is not None:
                    self._send_json(err[0], err[1])
                    return
                assert mid is not None
                status, payload = run_embedding_request(eng, req, model_name=mid, lock=emb_locks[mid])
                self._send_json(status, payload)
                return

            if self.path in ("/v1/chat/completions", "/chat/completions"):
                mid, obj, err = _resolve_route_target(chat_map, default_chat_model, req_model, "chat")
                if err is not None:
                    self._send_json(err[0], err[1])
                    return
                assert mid is not None
                status, payload, is_sse = dispatch_chat_completion(obj, req, model_name=mid, lock=chat_locks[mid])
                if is_sse and status == 200:
                    self._send_sse(payload)  # type: ignore[arg-type]
                else:
                    self._send_json(status, payload)  # type: ignore[arg-type]
                return

            if self.path in ("/v1/completions", "/completions"):
                mid, obj, err = _resolve_route_target(chat_map, default_chat_model, req_model, "chat")
                if err is not None:
                    self._send_json(err[0], err[1])
                    return
                assert mid is not None
                status, payload, is_sse = dispatch_completion(obj, req, model_name=mid, lock=chat_locks[mid])
                if is_sse and status == 200:
                    self._send_sse(payload)  # type: ignore[arg-type]
                else:
                    self._send_json(status, payload)  # type: ignore[arg-type]
                return

            self._send_error(404, f"Not found: {self.path}")
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            self._send_error(500, str(exc), err_type="server_error")

    UnifiedHandler.do_GET = do_GET  # type: ignore[method-assign]
    UnifiedHandler.do_POST = do_POST  # type: ignore[method-assign]
    return UnifiedHandler


class UnifiedOpenAIServer(ManagedHTTPServer):
    """Single-port OpenAI HTTP server hosting multiple embedding and generative engines by model ID."""

    def __init__(
        self,
        *,
        embedding_engines: dict[str, Any] | None = None,
        chat_engines: dict[str, Any] | None = None,
        default_embedding_model: str | None = None,
        default_chat_model: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        mlx_cache_limit_mb: int | None = 512,
        use_wired_limit: bool = False,
    ) -> None:
        self.embedding_engines = dict(embedding_engines or {})
        self.chat_engines = dict(chat_engines or {})
        handler_cls = make_unified_handler(
            embedding_engines=self.embedding_engines,
            chat_engines=self.chat_engines,
            default_embedding_model=default_embedding_model,
            default_chat_model=default_chat_model,
            max_body_bytes=max_body_bytes,
        )
        super().__init__(
            handler_cls,
            host=host,
            port=port,
            mlx_cache_limit_mb=mlx_cache_limit_mb,
            use_wired_limit=use_wired_limit,
        )


__all__ = [
    "ChatMicroBatcher",
    "DEFAULT_MAX_BODY_BYTES",
    "EmbeddingHTTPServer",
    "JsonHTTPHandler",
    "ManagedHTTPServer",
    "OpenAIChatAdapter",
    "UnifiedOpenAIServer",
    "build_chat_completion",
    "build_text_completion",
    "dispatch_chat_completion",
    "dispatch_completion",
    "embedding_health_payload",
    "extract_json_schema",
    "extract_request_param",
    "get_engine_lock",
    "iter_chat_sse_events",
    "iter_completion_sse_events",
    "make_chat_handler",
    "make_embedding_handler",
    "make_unified_handler",
    "run_embedding_request",
    "schema_cache_key",
]
