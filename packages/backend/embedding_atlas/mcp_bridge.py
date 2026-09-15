"""Standards-compliant MCP (Model Context Protocol) bridge for Geospatial Atlas.

Exposes the viewer's 19 tools at ``POST/GET /mcp`` using the
**Streamable HTTP** transport — the format Claude Desktop, Claude Code,
Cursor, Continue, and every MCP-capable client speaks natively.

Architecture
------------

Claude Desktop / Cursor / …   (remote client)
          │                    Streamable HTTP  (POST + SSE stream)
          ▼
    /mcp endpoint              (mounted Starlette sub-app in FastAPI)
          │                    JSON-RPC 2.0 pass-through
          ▼
    WebSocketHandler           (existing /data/mcp_websocket)
          │                    ws://…
          ▼
    Viewer JavaScript          (packages/viewer/src/app/mcp_server.ts)
          │
          ▼
    Tool handlers              (chart state, DuckDB queries, screenshots)

Why a bridge at all? The tool *implementations* live in the viewer
(access chart state, Mosaic coordinator, canvases). This module is a
**protocol adapter**: it turns well-formed MCP requests (Streamable
HTTP + JSON-RPC) into the plain JSON-RPC the viewer already speaks.

No stdio-HTTP shim scripts, no Node.js, no editing claude_desktop_config
beyond:

    "mcpServers": { "geospatial-atlas": { "url": "http://localhost:5055/mcp" } }
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .server import WebSocketHandler


logger = logging.getLogger("embedding-atlas")


class MCPBridge:
    """Binds an MCP Streamable HTTP endpoint to the viewer WebSocket.

    Any tool call Claude makes is forwarded verbatim as JSON-RPC to the
    viewer. Tool list is fetched from the viewer on demand, which means
    adding tools in ``packages/viewer/src/model_context/model_context.ts``
    needs zero Python changes — they show up automatically.
    """

    def __init__(self, get_handler: Callable[[], "WebSocketHandler | None"]) -> None:
        self._get_handler = get_handler
        self._cached_tools: list[types.Tool] | None = None

        self.server: Server = Server("geospatial-atlas")

        @self.server.list_tools()
        async def _list() -> list[types.Tool]:
            return await self._list_tools()

        # validate_input=False: we're a pure forwarder. The viewer does
        # its own schema validation; running it twice just forbids
        # perfectly good payloads on the Python side.
        @self.server.call_tool(validate_input=False)
        async def _call(
            name: str, arguments: dict[str, Any]
        ) -> list[types.ContentBlock]:
            return await self._call_tool(name, arguments)

        self.session_manager: StreamableHTTPSessionManager = (
            StreamableHTTPSessionManager(
                app=self.server,
                event_store=None,
                json_response=False,  # use SSE streaming for long tool calls
                stateless=True,  # no per-client state — every call forwards anyway
            )
        )

    async def _list_tools(self) -> list[types.Tool]:
        """Return the canonical tool list. If the viewer is connected we ask
        it for authoritative descriptors; otherwise we fall back to the
        hardcoded ``mcp_tools.CANONICAL_TOOLS`` so Claude Desktop always
        sees the 19 tools even before the viewer tab is open.

        Calling a tool while the viewer is disconnected returns a clear
        ``TextContent`` error pointing the user to the viewer URL.
        """
        handler = self._get_handler()
        if handler is not None and handler.is_connected:
            try:
                envelope = await handler.send_request(
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
                )
                # The viewer sends back a JSON-RPC envelope
                # {jsonrpc, id, result: {tools: [...]}, error: null}
                # — unwrap to get the actual tools list.
                result = (envelope or {}).get("result") or {}
                raw_tools = result.get("tools") or []
                tools = [self._to_tool(t) for t in raw_tools if isinstance(t, dict)]
                if tools:
                    self._cached_tools = tools
                    return tools
            except Exception as e:
                logger.warning(f"MCP list_tools: viewer forward failed: {e}")

        if self._cached_tools is not None:
            return self._cached_tools

        # Fallback: canonical, hardcoded list. Keeps Claude Desktop happy
        # from the moment the server comes up — no need for the viewer
        # tab to exist just to enumerate tools.
        from .mcp_tools import CANONICAL_TOOLS

        return [
            types.Tool(name=n, description=d, inputSchema=s)
            for (n, d, s) in CANONICAL_TOOLS
        ]

    async def _call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> list[types.ContentBlock]:
        """Forward a tool call to the viewer and adapt the response."""
        handler = self._get_handler()
        if handler is None or not handler.is_connected:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        "Geospatial Atlas viewer is not connected. Open the viewer "
                        "in a browser (e.g. http://localhost:5055) and retry."
                    ),
                )
            ]

        try:
            envelope = await handler.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                }
            )
        except Exception as e:
            return [types.TextContent(type="text", text=f"Viewer RPC error: {e}")]

        # Unwrap JSON-RPC envelope {jsonrpc, id, result, error} the viewer
        # sends. The actual tool result is under `.result`.
        envelope = envelope or {}
        rpc_error = envelope.get("error")
        if rpc_error:
            raise RuntimeError(
                f"{rpc_error.get('message', 'RPC error')} "
                f"(code {rpc_error.get('code', '?')})"
            )
        result = envelope.get("result") or {}
        raw_content = result.get("content") or []
        content: list[types.ContentBlock] = []
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            ct = block.get("type")
            if ct == "text":
                content.append(
                    types.TextContent(type="text", text=str(block.get("text", "")))
                )
            elif ct == "image":
                content.append(
                    types.ImageContent(
                        type="image",
                        data=str(block.get("data", "")),
                        mimeType=str(block.get("mimeType", "image/png")),
                    )
                )
            else:
                # Unknown block types — serialize as JSON text so we don't lose info.
                content.append(
                    types.TextContent(type="text", text=json.dumps(block, default=str))
                )
        if result.get("isError"):
            # lowlevel Server interprets a raised exception as isError=True,
            # but we've captured a viewer-emitted error. Surface as text so
            # Claude sees what happened.
            msg = (
                "".join(b.text for b in content if isinstance(b, types.TextContent))
                or "tool reported isError without content"
            )
            raise RuntimeError(msg)
        return content

    @staticmethod
    def _to_tool(raw: dict[str, Any]) -> types.Tool:
        """Normalize a viewer-emitted tool descriptor into mcp.types.Tool."""
        return types.Tool(
            name=str(raw.get("name", "unknown")),
            title=raw.get("title"),
            description=raw.get("description"),
            inputSchema=raw.get("inputSchema") or {"type": "object"},
            outputSchema=raw.get("outputSchema"),
        )

    def mount(self, app: "FastAPI", path: str = "/mcp") -> None:
        """Mount this bridge's Streamable HTTP app under ``path``.

        Adds an async context to FastAPI's lifespan so the underlying
        session manager starts/stops cleanly with the ASGI app.
        """
        # Session manager needs its run() context active while the app serves.
        # Register as an ASGI lifespan via a custom startup/shutdown.
        prev_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(fastapi_app):
            async with self.session_manager.run():
                if prev_lifespan is not None:
                    async with prev_lifespan(fastapi_app):
                        yield
                else:
                    yield

        app.router.lifespan_context = lifespan
        # Mount the Starlette sub-app that wraps the session manager.
        from starlette.applications import Starlette
        from starlette.routing import Mount

        mcp_inner = Starlette(
            routes=[Mount("/", app=self.session_manager.handle_request)]
        )
        app.mount(path, mcp_inner)

        # FastAPI returns 405 for the bare /mcp form (no trailing slash)
        # because the mount only matches descendants. Normalise the
        # path via HTTP middleware so mcp-remote, Claude Desktop, and
        # any client that doesn't add the slash works identically.
        _normalized = path.rstrip("/") + "/"
        _exact = path.rstrip("/")

        @app.middleware("http")
        async def _mcp_slash_fix(request, call_next):
            if request.url.path == _exact:
                request.scope["path"] = _normalized
                request.scope["raw_path"] = _normalized.encode()
            return await call_next(request)
