"""Language Server Protocol: symbol-aware code intelligence.

Text search finds strings; a language server understands scope.  It follows
imports and re-exports, distinguishes a shadowed local from the module-level
name it hides, and knows which of four same-named methods a call resolves to.
That is the difference between renaming a symbol and corrupting a file.

The layering mirrors `offset/tools/mcp`, for the same reason: framing, protocol
types, one client per server process, a registry that owns process lifetime,
and a thin tool surface on top.

- `protocol` - the `Content-Length` codec, URI and UTF-16 conversion, and the
  wire types.  No I/O.
- `client`   - one server process: handshake, capability gating, document
  version tracking, buffered `publishDiagnostics`, request correlation.
- `servers`  - which server to run for a language, discovered on PATH and
  overridable through `lsp.json`; one process per `(language, root)`.
- `tool`     - the two model-facing tools, split by danger.
"""

from __future__ import annotations

from offset.tools.lsp.client import (
    LSPCancelled,
    LSPClient,
    LSPError,
    LSPTimeout,
    ServerGone,
    Unsupported,
)
from offset.tools.lsp.protocol import (
    CodeAction,
    Diagnostic,
    Location,
    Position,
    Range,
    Symbol,
    TextEdit,
    WorkspaceEdit,
    from_uri,
    to_uri,
)
from offset.tools.lsp.servers import (
    Candidate,
    Config,
    ServerConfig,
    Servers,
    language_for,
    languages,
    load_config,
    missing_message,
)
from offset.tools.lsp.tool import LspEdit, LspQuery, lsp_tools

__all__ = [
    "Candidate",
    "CodeAction",
    "Config",
    "Diagnostic",
    "LSPCancelled",
    "LSPClient",
    "LSPError",
    "LSPTimeout",
    "Location",
    "LspEdit",
    "LspQuery",
    "Position",
    "Range",
    "ServerConfig",
    "ServerGone",
    "Servers",
    "Symbol",
    "TextEdit",
    "Unsupported",
    "WorkspaceEdit",
    "from_uri",
    "language_for",
    "languages",
    "load_config",
    "lsp_tools",
    "missing_message",
    "to_uri",
]
