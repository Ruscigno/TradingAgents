"""Thin sync wrapper around the async MCP streamable-HTTP client.

The cascade runs as a CLI command (Typer is sync), but the MCP SDK is async.
This module exposes ``call_tool_sync(name, args, url)`` that does the full
connect → initialize → call → close round-trip in one shot per call.

For now the cascade only calls ``list_tickers`` once at the start (etapa 0)
and ``get_ohlc`` per ticker in etapas 2/3. If we ever need long-lived sessions
(e.g. streaming), refactor this to keep a session open with a lifespan.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


log = logging.getLogger(__name__)


class MCPError(RuntimeError):
    """Raised when the MCP server returns an error or unexpected payload."""


async def _call_tool(url: str, name: str, args: dict[str, Any] | None) -> Any:
    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, args or {})


def call_tool_sync(
    url: str,
    name: str,
    args: dict[str, Any] | None = None,
) -> dict:
    """Run an MCP tool call and return the parsed JSON body.

    Server tools (in our case fastapi-mcp) wrap a JSON response in
    ``CallToolResult.content[0].text``. We parse it here so callers
    can work with plain dicts.
    """
    log.debug("MCP call: %s %s args=%s", url, name, args)
    result = asyncio.run(_call_tool(url, name, args))
    if getattr(result, "isError", False):
        raise MCPError(f"MCP tool {name!r} returned isError")
    content = getattr(result, "content", None) or []
    if not content:
        raise MCPError(f"MCP tool {name!r} returned empty content")
    block = content[0]
    text = getattr(block, "text", None)
    if text is None:
        raise MCPError(f"MCP tool {name!r}: first content block has no text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise MCPError(f"MCP tool {name!r}: response is not valid JSON: {e}") from e
