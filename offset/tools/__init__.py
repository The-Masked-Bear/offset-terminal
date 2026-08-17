"""Tools: the agent's hands.

Everything here is on by default.  A capability that ships disabled is a
capability nobody uses, so the switch that matters is not "is this tool
loaded" but "may this particular call proceed" — which is the approval policy
in `offset.tools.runtime`, decided per call against the damage it can do.
"""

from offset.tools.base import (
    Cancelled,
    Danger,
    Tool,
    ToolContext,
    ToolResult,
    Toolbox,
    validate,
)
from offset.tools.runtime import Approval, Runtime

__all__ = [
    "Approval", "Cancelled", "Danger", "Runtime", "Tool", "ToolContext",
    "ToolResult", "Toolbox", "validate",
]
