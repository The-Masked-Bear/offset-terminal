"""The easter egg engine: triggers, cooldowns, discovery, achievements.

Design constraint that shapes everything here: an egg is a *rendering*, never
an action.  `fire()` returns a `Reveal` describing something to draw and the
caller decides whether there is room for it.  Nothing in this module can
modify a file, a session, or a tool result, which is what makes it safe to
leave permanently switched on.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

Tone = str  # a palette role name from offset.ui.tokens.TONES


@dataclass(slots=True)
class Reveal:
    """Something to draw.  Purely cosmetic, by construction."""

    egg_id: str
    title: str
    lines: list[str] = field(default_factory=list)
    tone: Tone = "branch"
    frames: list[list[str]] = field(default_factory=list)
    duration: float = 4.0
    achievement: bool = False

    @property
    def animated(self) -> bool:
        return bool(self.frames)


# -- triggers ---------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Trigger:
    kind: str  # command | keys | count | moment | chance | milestone | idle
    commands: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    event: str = ""
    threshold: int = 0
    probability: float = 0.0
    seconds: float = 0.0
    when: Callable[[time.struct_time], bool] | None = None

    @staticmethod
    def command(*names: str) -> "Trigger":
        return Trigger("command", commands=tuple(n.lower() for n in names))

    @staticmethod
    def sequence(keys: Sequence[str]) -> "Trigger":
        # Not named `keys`: the slots descriptor for the field would shadow it.
        return Trigger("keys", keys=tuple(keys))

    @staticmethod
    def count(event: str, threshold: int) -> "Trigger":
        return Trigger("count", event=event, threshold=threshold)

    @staticmethod
    def moment(when: Callable[[time.struct_time], bool]) -> "Trigger":
        return Trigger("moment", when=when)

    @staticmethod
    def chance(event: str, probability: float) -> "Trigger":
        return Trigger("chance", event=event, probability=probability)

    @staticmethod
    def milestone(event: str) -> "Trigger":
        return Trigger("milestone", event=event)

    @staticmethod
    def idle(seconds: float) -> "Trigger":
        return Trigger("idle", seconds=seconds)


#: An egg's payload: static text, or a function of the triggering context.
Payload = Callable[[dict[str, Any]], Reveal]


@dataclass(slots=True)
class Egg:
    id: str
    name: str
    trigger: Trigger
    payload: Payload
    hint: str = ""  # shown in the trophy room once discovered
    achievement: bool = False
    repeatable: bool = True

    def reveal(self, context: dict[str, Any]) -> Reveal:
        got = self.payload(context)
        got.egg_id = self.id
        got.achievement = self.achievement
        return got


# -- engine -----------------------------------------------------------------


class EggEngine:
    """Holds the catalogue, the counters, and the discovery record."""

    __slots__ = ("eggs", "_by_command", "_counters", "_found", "_last_fire", "_cooldown", "_rng", "_clock", "_path", "_keys", "_last_input")

    def __init__(
        self,
        eggs: Iterable[Egg] = (),
        *,
        store: Path | str | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        cooldown: float = 20.0,
    ) -> None:
        self.eggs: dict[str, Egg] = {}
        self._by_command: dict[str, Egg] = {}
        for egg in eggs:
            self.register(egg)
        self._counters: dict[str, int] = {}
        self._found: dict[str, float] = {}
        self._last_fire = -1e9
        self._cooldown = cooldown
        self._rng = rng or random.Random()
        self._clock = clock
        self._keys: list[str] = []
        self._last_input = clock()
        self._path = Path(store) if store else None
        self._load()

    def register(self, egg: Egg) -> Egg:
        self.eggs[egg.id] = egg
        if egg.trigger.kind == "command":
            for name in egg.trigger.commands:
                self._by_command[name] = egg
        return egg

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            found = raw.get("found")
            counters = raw.get("counters")
            if isinstance(found, dict):
                self._found = {str(k): float(v) for k, v in found.items()}
            if isinstance(counters, dict):
                self._counters = {str(k): int(v) for k, v in counters.items()}

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"found": self._found, "counters": self._counters}, indent=1)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self._path)

    # -- state ------------------------------------------------------------

    @property
    def discovered(self) -> set[str]:
        return set(self._found)

    def progress(self) -> tuple[int, int]:
        return len(self._found), len(self.eggs)

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def trophies(self) -> list[tuple[Egg, bool]]:
        """Every egg with whether it has been found — the achievements screen."""
        return [(egg, egg.id in self._found) for egg in self.eggs.values()]

    # -- firing -----------------------------------------------------------

    def _allowed(self, egg: Egg) -> bool:
        if not egg.repeatable and egg.id in self._found:
            return False
        return self._clock() - self._last_fire >= self._cooldown

    def _fire(self, egg: Egg, context: dict[str, Any] | None = None) -> Reveal:
        self._last_fire = self._clock()
        self._found.setdefault(egg.id, time.time())
        self.save()
        return egg.reveal(context or {})

    def command(self, text: str) -> Reveal | None:
        """A typed command.  Explicit, so it ignores the cooldown entirely."""
        name = text.strip().lstrip("/").split()[0].lower() if text.strip() else ""
        egg = self._by_command.get(name)
        if egg is None:
            return None
        # Counted so an egg can escalate across repeat invocations.
        key = f"cmd:{name}"
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._fire(egg, {"input": text, "engine": self})

    def key(self, key: str) -> Reveal | None:
        """Feed keystrokes; matches any registered sequence at its tail."""
        self._last_input = self._clock()
        self._keys.append(key)
        if len(self._keys) > 24:
            del self._keys[:-24]
        for egg in self.eggs.values():
            seq = egg.trigger.keys
            if seq and tuple(self._keys[-len(seq):]) == seq and self._allowed(egg):
                self._keys.clear()
                return self._fire(egg, {"keys": seq})
        return None

    def event(self, name: str, **data: Any) -> Reveal | None:
        """Report that something happened; may or may not produce a reveal.

        `suppress` is honoured absolutely: when the caller says the user is
        reading an error, nothing fires.
        """
        self._counters[name] = self._counters.get(name, 0) + 1
        count = self._counters[name]
        if data.get("suppress"):
            return None

        for egg in self.eggs.values():
            t = egg.trigger
            if t.event != name:
                continue
            if t.kind == "count" and count == t.threshold and self._allowed(egg):
                return self._fire(egg, {"count": count, **data})
            if t.kind == "milestone" and self._allowed(egg):
                return self._fire(egg, {"count": count, **data})
            if t.kind == "chance" and self._rng.random() < t.probability and self._allowed(egg):
                return self._fire(egg, {"count": count, **data})
        return None

    def tick(self, now: time.struct_time | None = None) -> Reveal | None:
        """Time-based eggs: the clock, the calendar, and going quiet."""
        stamp = now or time.localtime()
        for egg in self.eggs.values():
            t = egg.trigger
            if t.kind == "moment" and t.when and t.when(stamp) and self._allowed(egg):
                return self._fire(egg, {"time": stamp})
            if t.kind == "idle" and (self._clock() - self._last_input) >= t.seconds and self._allowed(egg):
                self._last_input = self._clock()
                return self._fire(egg, {"idle": t.seconds})
        return None

    def touch(self) -> None:
        """Note user activity, resetting the idle timer."""
        self._last_input = self._clock()


def text_egg(title: str, *lines: str, tone: Tone = "branch", duration: float = 4.0) -> Payload:
    """The common case: a fixed block of text."""

    def payload(_context: dict[str, Any]) -> Reveal:
        return Reveal("", title, list(lines), tone, duration=duration)

    return payload
