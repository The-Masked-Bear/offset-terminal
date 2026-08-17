"""Frame-based animation.

The reference stylesheet ships nine keyframe animations — glitch, marquee,
star-spin, shake, svg-pop, dot-bounce, wiggle, hack-glitch, radar-spin — and a
single spring easing curve.  All nine are reproduced here as pure functions of
absolute time, so any frame can be reconstructed exactly (which is what makes
recorded sessions replayable).

Every function takes `t` in seconds and is deterministic: same `t`, same
output.  Randomised effects seed from the frame index, never from global RNG.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Final, Sequence

from offset.ui.tokens import G, text_width

# --------------------------------------------------------------------------
# easing
# --------------------------------------------------------------------------


def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    """CSS `cubic-bezier()`, solved with Newton-Raphson then bisection."""
    cx = 3.0 * x1
    bx = 3.0 * (x2 - x1) - cx
    ax = 1.0 - cx - bx
    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by

    def sample_x(t: float) -> float:
        return ((ax * t + bx) * t + cx) * t

    def sample_y(t: float) -> float:
        return ((ay * t + by) * t + cy) * t

    def dx(t: float) -> float:
        return (3.0 * ax * t + 2.0 * bx) * t + cx

    def solve(x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        t = x
        for _ in range(8):
            err = sample_x(t) - x
            if abs(err) < 1e-6:
                return sample_y(t)
            d = dx(t)
            if abs(d) < 1e-9:
                break
            t -= err / d
        lo, hi, t = 0.0, 1.0, x
        for _ in range(24):
            err = sample_x(t) - x
            if abs(err) < 1e-6:
                break
            if err > 0:
                hi = t
            else:
                lo = t
            t = (lo + hi) * 0.5
        return sample_y(t)

    return solve


#: `cubic-bezier(0.175, 0.885, 0.32, 1.275)` — overshoots past 1.0 and settles.
SPRING: Final = cubic_bezier(0.175, 0.885, 0.32, 1.275)
EASE: Final = cubic_bezier(0.25, 1.0, 0.5, 1.0)


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# --------------------------------------------------------------------------
# @keyframes marquee
# --------------------------------------------------------------------------


def marquee(items: Sequence[str], width: int, t: float, *, speed: float = 6.0, sep: str | None = None) -> str:
    """Endless star-separated ticker, exactly the site's masthead."""
    if width <= 0:
        return ""
    joiner = f" {G.STAR} " if sep is None else sep
    strip = joiner.join(items) + joiner
    if not strip:
        return " " * width
    reps = -(-(width + len(strip)) // len(strip)) + 1
    doubled = strip * reps
    off = int(t * speed) % len(strip)
    return doubled[off : off + width]


# --------------------------------------------------------------------------
# @keyframes glitch / hack-glitch
# --------------------------------------------------------------------------


def glitch(text: str, t: float, *, rate: float = 12.0, intensity: float = 0.18, pool: str | None = None) -> str:
    """Corrupt a fraction of the characters, stable within a frame."""
    if intensity <= 0.0 or not text:
        return text
    chars = pool or G.GLITCH_POOL
    rng = random.Random(int(t * rate))
    return "".join(rng.choice(chars) if ch != " " and rng.random() < intensity else ch for ch in text)


def hack_glitch(text: str, t: float, *, rate: float = 20.0) -> str:
    """The heavier variant used when something is actively going wrong."""
    return glitch(text, t, rate=rate, intensity=0.45)


def glitch_burst(text: str, t: float, *, period: float = 5.0, dur: float = 0.32) -> str:
    """Clean most of the time; corrupts for `dur` once per `period`."""
    phase = t % period
    if phase > dur:
        return text
    return glitch(text, t, intensity=0.35 * (1.0 - phase / dur))


# --------------------------------------------------------------------------
# @keyframes shake / wiggle
# --------------------------------------------------------------------------


def shake(t: float, *, amp: int = 1, freq: float = 22.0, dur: float | None = None) -> int:
    """Horizontal cell offset.  `dur` decays the amplitude to zero."""
    if dur is not None:
        if t >= dur:
            return 0
        amp = max(0, round(amp * (1.0 - t / dur)))
    return round(math.sin(t * freq) * amp)


_WIGGLE: Final = ("/", "|", "\\", "|") if G.BLOCK == "#" else ("\u2571", "\u2502", "\u2572", "\u2502")


def wiggle(t: float, *, freq: float = 3.0) -> str:
    """A rocking stroke — the cell-grid stand-in for a few degrees of tilt."""
    return _WIGGLE[int(t * freq) % 4]


# --------------------------------------------------------------------------
# @keyframes star-spin / radar-spin / dot-bounce
# --------------------------------------------------------------------------


def star_spin(t: float, *, freq: float = 8.0) -> str:
    return G.STARS[int(t * freq) % len(G.STARS)]


def radar_spin(t: float, *, freq: float = 10.0) -> str:
    return G.SPINNER[int(t * freq) % len(G.SPINNER)]


def dot_bounce(t: float, *, freq: float = 6.0, n: int = 3) -> str:
    active = int(t * freq) % (2 * n - 2) if n > 1 else 0
    if active >= n:
        active = 2 * n - 2 - active
    return "".join(G.DOT if i == active else G.DOT_EMPTY for i in range(n))


# --------------------------------------------------------------------------
# @keyframes svg-pop + the count-up statistics
# --------------------------------------------------------------------------


def svg_pop(t: float, *, dur: float = 0.45, delay: float = 0.0) -> float:
    """Scale factor 0..~1.1..1.0 — springs in, overshoots, settles."""
    return SPRING(clamp01((t - delay) / dur)) if dur > 0 else 1.0


def count_up(target: float, t: float, *, dur: float = 1.4, delay: float = 0.0, ease: Callable[[float], float] = EASE) -> float:
    """The site's hero counters: every number arrives by climbing to itself."""
    return target * ease(clamp01((t - delay) / dur)) if dur > 0 else target


def type_on(text: str, t: float, *, cps: float = 45.0, delay: float = 0.0) -> str:
    """Reveal text one character at a time."""
    n = int(max(0.0, t - delay) * cps)
    return text[:n]


def pulse(t: float, *, freq: float = 1.6) -> float:
    """0..1 triangle wave for breathing highlights."""
    return abs(((t * freq) % 2.0) - 1.0)


def center(text: str, width: int) -> int:
    """Left column that centres `text` in `width` (never negative)."""
    return max(0, (width - text_width(text)) // 2)
