"""MCP server exposing the RAG backend as tools.

Tools:
  - list_knowledgebases()                       -> the caller's RAGs
  - search_rag(knowledgebase_id, query, top_k?) -> raw matching chunks (no LLM)
  - ask_rag(knowledgebase_id, query)            -> grounded answer + citations

Auth: every tool resolves the calling user from the `x-api-key` header. The
HTTP layer (see main.py) sets `current_user_id` from that header before the
tool runs; tools read it via `_require_user()` and enforce `assert_kb_access`
so a key can only reach its own (or legacy NULL-owner) knowledgebases.
"""

import os
import contextvars
import logging
import time
from functools import wraps
from collections import defaultdict
from threading import Lock
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Simple in-memory sliding-window rate limiter per API key: 60 requests per minute
LIMIT_WINDOW = 60  # seconds
LIMIT_MAX_REQUESTS = 60
_rate_limit_lock = Lock()
_rate_limit_records = defaultdict(list)  # api_key -> list of float timestamps

# Populated by the ASGI wrapper in main.py from the `x-api-key` header.
current_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_user_id", default=None
)

mcp = FastMCP("RAGA")


def log_tool_invocation(tool_name: str):
    """Decorator to log tool performance, caller, and success/error metrics.

    Preserves signature and docstrings so FastMCP still discovers tools correctly.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = current_user_id.get() or "unknown"
            logger.info(
                "MCP_TOOL_START tool=%s user_id=%s args=%s kwargs=%s",
                tool_name,
                user_id,
                args,
                {k: v for k, v in kwargs.items() if k != "query"},
            )
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                meta = {}
                if isinstance(result, list):
                    meta["count"] = len(result)
                elif isinstance(result, dict):
                    if "chunks" in result:
                        meta["chunks"] = len(result["chunks"])
                    if "citations" in result:
                        meta["citations"] = len(result["citations"])
                    if "answer" in result:
                        meta["answer_length"] = len(result["answer"])
                logger.info(
                    "MCP_TOOL_SUCCESS tool=%s user_id=%s elapsed=%.2fs meta=%s",
                    tool_name,
                    user_id,
                    elapsed,
                    meta,
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "MCP_TOOL_FAILURE tool=%s user_id=%s elapsed=%.2fs error_type=%s error=%s",
                    tool_name,
                    user_id,
                    elapsed,
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator


def _require_user() -> str:
    user_id = current_user_id.get()
    if not user_id:
        raise ValueError(
            "Unauthenticated: provide a valid `x-api-key` header for the MCP server."
        )
    return user_id


@mcp.tool()
@log_tool_invocation("list_knowledgebases")
def list_knowledgebases() -> list[dict[str, Any]]:
    """List the knowledgebases (RAGs) the authenticated key can query.

    Returns each RAG's `rag_id`, `name`, and `description`. Use a `rag_id` as
    the `knowledgebase_id` argument to `search_rag` or `ask_rag`.
    """
    from db_utils import list_rags_for_user

    user_id = _require_user()
    rags = list_rags_for_user(user_id)
    return [
        {
            "knowledgebase_id": rag.get("rag_id"),
            "name": rag.get("name"),
            "description": rag.get("description"),
            "document_count": rag.get("document_count"),
        }
        for rag in rags
    ]


@mcp.tool()
@log_tool_invocation("search_rag")
def search_rag(
    knowledgebase_id: str,
    query: str,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    """Semantic search over a knowledgebase. Returns matching chunks (no LLM).

    Args:
        knowledgebase_id: The RAG to search (from `list_knowledgebases`).
        query: Natural-language search query.
        top_k: Max chunks to return (1-30). Defaults to the RAG's configured top_k.
    """
    from db_utils import assert_kb_access
    from chroma_utils import get_chunks_from_chroma
    from main import _build_source_chunks

    user_id = _require_user()
    assert_kb_access(user_id, knowledgebase_id)

    resolved_top_k = max(1, min(int(top_k), 30)) if top_k is not None else 8
    chunks = get_chunks_from_chroma(query, knowledgebase_id, top_k=resolved_top_k)
    source_chunks = _build_source_chunks(chunks or [])
    return {
        "knowledgebase_id": knowledgebase_id,
        "query": query,
        "count": len(source_chunks),
        "chunks": source_chunks,
    }


@mcp.tool()
@log_tool_invocation("ask_rag")
def ask_rag(knowledgebase_id: str, query: str) -> dict[str, Any]:
    """Answer a question grounded in a knowledgebase, with citations.

    Retrieves relevant chunks, asks the LLM using only that context, and returns
    the answer plus inline citations and the source chunks used.

    Args:
        knowledgebase_id: The RAG to ground the answer in (from `list_knowledgebases`).
        query: The question to answer.
    """
    from langchain_openai import ChatOpenAI
    from db_utils import assert_kb_access
    from chroma_utils import get_chunks_from_chroma
    from main import _build_source_chunks, _build_citations

    user_id = _require_user()
    assert_kb_access(user_id, knowledgebase_id)

    chunks = get_chunks_from_chroma(query, knowledgebase_id)
    source_chunks = _build_source_chunks(chunks or [])
    context = "\n\n".join(c["content"] for c in (chunks or []) if c)

    prompt = (
        "Use the following context to answer the question. "
        "If the context does not contain relevant information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=15.0,
    )
    answer = llm.invoke(prompt).content
    citations = _build_citations(answer, source_chunks)
    return {
        "knowledgebase_id": knowledgebase_id,
        "query": query,
        "answer": answer,
        "citations": citations,
        "chunks": source_chunks,
    }


# ---------------------------------------------------------------------------
# HTTP transport: authenticate the `x-api-key` header, then run the MCP app.
# ---------------------------------------------------------------------------

# Serve the streamable-HTTP endpoint at the mount root so `app.mount("/mcp", ...)`
# exposes it exactly at `/mcp` (not `/mcp/mcp`).
mcp.settings.streamable_http_path = "/"


class _ApiKeyAuthMiddleware:
    """Pure-ASGI middleware that gates the MCP app on the `x-api-key` header.

    On a valid key it sets the `current_user_id` contextvar for the duration of
    the request so tools can resolve the caller. Invalid/missing keys get 401.
    Includes rate limiting to ensure reliability (max 60 req/min).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Lazy import avoids a circular import at module load.
        from db_utils import get_api_key_record, touch_api_key_last_used

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        api_key = headers.get("x-api-key")
        record = get_api_key_record(api_key) if api_key else None

        if not record:
            await self._unauthorized(send)
            return

        # Rate Limit check
        now = time.time()
        with _rate_limit_lock:
            timestamps = [t for t in _rate_limit_records[api_key] if now - t < LIMIT_WINDOW]
            if len(timestamps) >= LIMIT_MAX_REQUESTS:
                logger.warning(
                    "MCP_RATE_LIMIT_TRIPPED user_id=%s count=%d window=%ds",
                    record["user_id"],
                    len(timestamps),
                    LIMIT_WINDOW,
                )
                await self._rate_limited(send)
                return
            timestamps.append(now)
            _rate_limit_records[api_key] = timestamps

        touch_api_key_last_used(api_key)
        token = current_user_id.set(record["user_id"])
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token)

    @staticmethod
    async def _unauthorized(send):
        body = b'{"error":"Invalid or missing x-api-key header."}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _rate_limited(send):
        body = b'{"error":"Rate limit exceeded. Max 60 requests per minute."}'
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def build_mcp_asgi_app():
    """Return the auth-wrapped streamable-HTTP ASGI app to mount at `/mcp`."""
    return _ApiKeyAuthMiddleware(mcp.streamable_http_app())


if __name__ == "__main__":
    # Allow running standalone over stdio for local testing with an MCP client.
    mcp.run()