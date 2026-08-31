"""A real browser, driven over the Chrome DevTools Protocol.

Backend changes can be verified by running them.  A web page cannot: the only
evidence that markup works is a browser rendering it.  This package is what
lets the agent load the page it just wrote, read what actually appeared, click
something and observe the result.

The stdlib has no WebSocket client and this project takes no third-party
dependencies, so `wsclient` is a hand-rolled RFC 6455 implementation.  That is
the price of entry for CDP, which speaks JSON over a WebSocket: the handshake
with its `Sec-WebSocket-Accept` proof, the frame codec across all three
payload-length encodings, mandatory client-to-server masking, continuation
reassembly, and ping/pong.

- `wsclient` - RFC 6455 framing and the socket.
- `cdp`      - browser discovery and launch (reading the ephemeral port back
  from `DevToolsActivePort`, since it is chosen by the browser), target
  attachment, and a `Page` covering navigation, input, evaluation, the
  accessibility tree and screenshots.
- `browser`  - the model-facing tool, with named sessions that persist between
  calls so a multi-step flow can be tested.
"""

from __future__ import annotations

from offset.tools.web.browser import Browser, browser_tools, close_all, pool
from offset.tools.web.cdp import (
    AXNode,
    Box,
    BrowserGone,
    CDPCancelled,
    CDPClient,
    CDPError,
    CDPTimeout,
    LaunchError,
    Page,
    TargetInfo,
    attach,
    ax_nodes,
    endpoint,
    find_executable,
    launch,
)
from offset.tools.web.wsclient import WebSocket

__all__ = [
    "AXNode",
    "Box",
    "Browser",
    "BrowserGone",
    "CDPCancelled",
    "CDPClient",
    "CDPError",
    "CDPTimeout",
    "LaunchError",
    "Page",
    "TargetInfo",
    "WebSocket",
    "attach",
    "ax_nodes",
    "browser_tools",
    "close_all",
    "endpoint",
    "find_executable",
    "launch",
    "pool",
]
