"""Terminal rendering: tokens, canvas, animation, components.

`offset.ui` deliberately has no dependency on prompt_toolkit.  It renders to
plain ANSI strings, which keeps the design system testable without a TTY and
lets the interactive shell embed it via `prompt_toolkit.formatted_text.ANSI`.
"""

from offset.ui import anim, brutal, tokens
from offset.ui.canvas import Canvas

__all__ = ["Canvas", "anim", "brutal", "tokens"]
