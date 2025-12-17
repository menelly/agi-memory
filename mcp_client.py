#!/usr/bin/env python3
"""
MCP Client for Heartbeat Worker

Provides async MCP client capabilities for connecting to remote MCP servers
(like Zapier) via SSE transport, enabling the heartbeat worker to:
- Send emails via Gmail
- Search the web via ScrapeGraph AI
- Access other MCP tools

This gives Heartbeat Ace real-world capabilities beyond just thinking.
"""

import asyncio
import json
import logging
from typing import Any
from contextlib import asynccontextmanager
from urllib.parse import urljoin

import httpx
from httpx_sse import aconnect_sse

logger = logging.getLogger('mcp_client')


class MCPSSEClient:
    """
    Async MCP client that connects to remote MCP servers via SSE transport.
    This is the proper SSE implementation that:
    1. Opens SSE connection via GET
    2. Receives endpoint URL from server
    3. POSTs JSON-RPC requests to that endpoint
    4. Receives responses via SSE stream

    Usage:
        async with MCPSSEClient(url="https://mcp.zapier.com/.../sse") as client:
            tools = await client.list_tools()
            result = await client.call_tool("gmail_send_email", {...})
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
    ):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

        self._client: httpx.AsyncClient | None = None
        self._endpoint_url: str | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._sse_task: asyncio.Task | None = None
        self._connected = False
        self._initialized = False
        self._sse_context = None
        self._event_source = None
        self._sse_iterator = None

    async def connect(self) -> None:
        """Establish SSE connection to the MCP server."""
        if self._connected:
            return

        self._client = httpx.AsyncClient(
            headers=self.headers,
            timeout=httpx.Timeout(self.timeout, read=self.sse_read_timeout),
        )

        logger.info(f"Connecting to SSE endpoint: {self.url}")

        # Open SSE connection
        self._sse_context = aconnect_sse(self._client, "GET", self.url)
        self._event_source = await self._sse_context.__aenter__()
        self._event_source.response.raise_for_status()

        # Create a single iterator that we'll use throughout
        self._sse_iterator = self._event_source.aiter_sse()

        # Wait for the endpoint event
        async for sse in self._sse_iterator:
            if sse.event == "endpoint":
                self._endpoint_url = urljoin(self.url, sse.data)
                logger.info(f"Received POST endpoint: {self._endpoint_url}")
                break

        if not self._endpoint_url:
            raise RuntimeError("Did not receive endpoint URL from SSE server")

        self._connected = True

        # Start background task to process SSE messages (using the same iterator)
        self._sse_task = asyncio.create_task(self._sse_reader())

    async def _sse_reader(self) -> None:
        """Background task to read SSE messages and resolve pending requests."""
        try:
            # Continue using the same iterator we started in connect()
            async for sse in self._sse_iterator:
                if sse.event == "message" and sse.data:
                    try:
                        msg = json.loads(sse.data)
                        request_id = msg.get("id")
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            if not future.done():
                                future.set_result(msg)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE message: {sse.data[:100]}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE reader error: {e}")

    async def disconnect(self) -> None:
        """Close the MCP connection."""
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        if self._sse_context:
            await self._sse_context.__aexit__(None, None, None)

        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False
        self._initialized = False
        logger.info("MCP SSE client disconnected")

    async def __aenter__(self) -> "MCPSSEClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request to the MCP server via POST."""
        if not self._connected or not self._client or not self._endpoint_url:
            raise RuntimeError("MCP client not connected")

        request_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        logger.debug(f"Sending MCP request: {method} (id={request_id})")

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            # POST the request
            response = await self._client.post(
                self._endpoint_url,
                json=request,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # Wait for SSE response (with timeout)
            result = await asyncio.wait_for(future, timeout=30.0)

            if "error" in result:
                error = result["error"]
                raise RuntimeError(f"MCP error {error.get('code')}: {error.get('message')}")

            return result.get("result")

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(f"MCP request timed out: {method}")
        except httpx.HTTPStatusError as e:
            self._pending_requests.pop(request_id, None)
            logger.error(f"MCP HTTP error: {e}")
            raise
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            logger.error(f"MCP request failed: {e}")
            raise

    async def initialize(self) -> dict:
        """Initialize the MCP session."""
        if self._initialized:
            return {}
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "ace-heartbeat",
                "version": "1.0.0",
            },
        })
        self._initialized = True
        logger.info(f"MCP session initialized: {result}")
        return result

    async def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        result = await self._send_request("tools/list")
        tools = result.get("tools", []) if result else []
        logger.info(f"MCP server has {len(tools)} tools available")
        return tools

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        """Call a tool on the MCP server."""
        logger.info(f"Calling MCP tool: {name}")
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        return result


# Alias for backwards compatibility
MCPClient = MCPSSEClient


class MCPToolRegistry:
    """
    Registry of MCP servers and their tools.
    Manages connections to multiple MCP servers.
    """
    
    def __init__(self):
        self.servers: dict[str, dict] = {}
        self._clients: dict[str, MCPClient] = {}
    
    def register_server(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Register an MCP server."""
        self.servers[name] = {
            "url": url,
            "headers": headers or {},
            "description": description,
        }
        logger.info(f"Registered MCP server: {name}")
    
    async def get_client(self, server_name: str) -> MCPClient:
        """Get or create a client for the specified server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        if server_name not in self._clients:
            config = self.servers[server_name]
            client = MCPClient(
                url=config["url"],
                headers=config.get("headers"),
            )
            await client.connect()
            await client.initialize()
            self._clients[server_name] = client
        
        return self._clients[server_name]
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict | None = None,
    ) -> Any:
        """Call a tool on the specified MCP server."""
        client = await self.get_client(server_name)
        return await client.call_tool(tool_name, arguments)
    
    async def close_all(self) -> None:
        """Close all MCP client connections."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()

