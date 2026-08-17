"""MCP client: offset speaking the standard tool protocol.

`manager.Manager` is the only thing a caller needs — it reads `mcp.json`,
connects, and hands back plain `offset.tools.base.Tool` objects, so the rest of
offset never learns that a tool lives on the other end of a pipe.
"""

from __future__ import annotations

from offset.tools.mcp.client import (
    ACCEPTED,
    PROTOCOL_VERSION,
    CallOutcome,
    MCPCancelled,
    MCPClient,
    MCPError,
    MCPTimeout,
    Prompt,
    RemoteTool,
    Resource,
    ServerGone,
)
from offset.tools.mcp.manager import (
    DOWN,
    IDLE,
    LIVE,
    OFF,
    Config,
    Manager,
    MCPTool,
    ServerConfig,
    Status,
    config_paths,
    load_config,
    parse_config,
    tool_name,
)
from offset.tools.mcp.transport import (
    HTTPTransport,
    StdioTransport,
    Transport,
    TransportClosed,
    TransportError,
)

__all__ = [
    "ACCEPTED",
    "PROTOCOL_VERSION",
    "CallOutcome",
    "Config",
    "DOWN",
    "HTTPTransport",
    "IDLE",
    "LIVE",
    "MCPCancelled",
    "MCPClient",
    "MCPError",
    "MCPTimeout",
    "MCPTool",
    "Manager",
    "OFF",
    "Prompt",
    "RemoteTool",
    "Resource",
    "ServerConfig",
    "ServerGone",
    "Status",
    "StdioTransport",
    "Transport",
    "TransportClosed",
    "TransportError",
    "config_paths",
    "load_config",
    "parse_config",
    "tool_name",
]
