"""Model providers: one wire format each, one event stream out.

Every provider converts its own protocol into the same `Event` sequence, so
nothing above this layer knows whether it is talking to Anthropic, OpenAI,
Google or a local llama.cpp.  That uniformity is what lets several different
models run side by side in one session.
"""

from offset.providers.base import (
    Event,
    Message,
    Request,
    Stop,
    StreamError,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    Turn,
    TurnBuilder,
    Usage,
)
from offset.providers.registry import MODELS, ModelInfo, resolve

__all__ = [
    "MODELS",
    "Event",
    "Message",
    "ModelInfo",
    "Request",
    "Stop",
    "StreamError",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolSpec",
    "Turn",
    "TurnBuilder",
    "Usage",
    "resolve",
]
