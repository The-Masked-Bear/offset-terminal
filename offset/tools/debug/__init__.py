"""Debug Adapter Protocol: stop the program and read it.

The value is narrow and large: a debugger replaces inference about state with
observation of it.  A model asked why a value is wrong can add prints, re-run
and reason, or it can stop on the line and read the locals.  The second is
shorter and does not require the guess to have been right.

DAP is close enough to LSP to share the framing and far enough to need its own
client.  Both use `Content-Length` headers over stdio; DAP then uses its own
envelope (`{seq, type, command}`, replies carrying `request_seq` and `success`)
and, unlike LSP, has a startup sequence whose ORDER is load-bearing:

    initialize -> wait for the `initialized` EVENT -> breakpoints
    -> configurationDone -> launch/attach

Breakpoints may only be sent inside that window, and `configurationDone` closes
it.  Sending it early, or setting breakpoints after it, is the classic DAP bug -
the adapter answers everything correctly and simply never stops anywhere.
`client.open_session` implements the sequence, including the adapters that
withhold `initialized` until they have been told what they are debugging.

- `protocol` - framing, the envelope, the wire types, and the two channels.
- `client`   - one adapter process: the sequence above, event routing, the
  `stopped` queue that makes state inspectable, and session ownership.
- `adapters` - which adapter to run for a language, found on PATH and
  overridable through `dap.json`.
- `tool`     - the two model-facing tools, split by danger.
"""

from __future__ import annotations

from offset.tools.debug.adapters import (
    AdapterConfig,
    Candidate,
    Config,
    Launch,
    available,
    choose,
    language_for,
    languages,
    load_config,
    missing_message,
)
from offset.tools.debug.client import (
    Configured,
    DebugCancelled,
    DebugClient,
    DebugError,
    Evaluation,
    OutputChunk,
    RequestFailed,
    RequestTimeout,
    Session,
    SessionBook,
    Stop,
    open_session,
)
from offset.tools.debug.protocol import (
    AdapterClosed,
    Breakpoint,
    Capabilities,
    Channel,
    Event,
    FramingError,
    ProtocolError,
    Response,
    Scope,
    SocketChannel,
    SourceBreakpoint,
    StackFrame,
    StdioChannel,
    Thread,
    Variable,
    frames_report,
)
from offset.tools.debug.tool import Debug, DebugInspect, book, debug_tools

__all__ = [
    "AdapterClosed",
    "AdapterConfig",
    "Breakpoint",
    "Candidate",
    "Capabilities",
    "Channel",
    "Config",
    "Configured",
    "Debug",
    "DebugCancelled",
    "DebugClient",
    "DebugError",
    "DebugInspect",
    "Evaluation",
    "Event",
    "FramingError",
    "Launch",
    "OutputChunk",
    "ProtocolError",
    "RequestFailed",
    "RequestTimeout",
    "Response",
    "Scope",
    "Session",
    "SessionBook",
    "SocketChannel",
    "SourceBreakpoint",
    "StackFrame",
    "StdioChannel",
    "Stop",
    "Thread",
    "Variable",
    "available",
    "book",
    "choose",
    "debug_tools",
    "frames_report",
    "language_for",
    "languages",
    "load_config",
    "missing_message",
    "open_session",
]
