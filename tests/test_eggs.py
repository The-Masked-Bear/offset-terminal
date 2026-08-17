"""The egg engine.

The catalogue's contents are a matter of taste; these tests defend the
properties that keep eggs from becoming a liability — they never change state,
never bury an error the user must read, never spam, and remember what has been
found.
"""

from __future__ import annotations

import json
import random
import time

import pytest

from offset.eggs.catalogue import CATALOGUE, build_engine
from offset.eggs.engine import Egg, EggEngine, Reveal, Trigger, text_egg


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock():
    return Clock()


def engine(*eggs: Egg, store=None, clock=None, rng=None, cooldown=20.0) -> EggEngine:
    return EggEngine(eggs or CATALOGUE, store=store, clock=clock or time.monotonic,
                     rng=rng or random.Random(0), cooldown=cooldown)


# -- catalogue sanity -------------------------------------------------------


def test_the_catalogue_is_substantial_and_unique():
    assert len(CATALOGUE) >= 40, "the brief asked for a ton of them"
    ids = [e.id for e in CATALOGUE]
    assert len(ids) == len(set(ids)), "duplicate egg ids"


def test_every_egg_produces_something_drawable():
    room = engine()
    for egg in CATALOGUE:
        got = egg.reveal({"engine": room})
        assert isinstance(got, Reveal)
        assert got.egg_id == egg.id
        assert got.title or got.lines or got.frames, f"{egg.id} draws nothing"


def test_commands_are_not_claimed_twice():
    claimed: dict[str, str] = {}
    for egg in CATALOGUE:
        for name in egg.trigger.commands:
            assert name not in claimed, f"{name} claimed by {claimed.get(name)} and {egg.id}"
            claimed[name] = egg.id


def test_achievements_do_not_repeat():
    for egg in CATALOGUE:
        if egg.achievement:
            assert not egg.repeatable, f"{egg.id} would fire its achievement more than once"


# -- triggers ---------------------------------------------------------------


def test_a_typed_command_fires():
    room = build_engine()
    got = room.command("bear")
    assert got is not None and got.title == "ARE YOU A BEAR?" and got.lines == ["No."]


def test_an_unknown_command_is_not_an_egg():
    assert build_engine().command("deploy to production") is None


def test_slash_and_arguments_are_tolerated():
    assert build_engine().command("/neofetch --verbose") is not None


def test_escalating_eggs_change_with_repetition():
    room = build_engine()
    answers = [room.command("bear").lines[0] for _ in range(4)]
    assert answers[0] == "No."
    assert answers[1] == "Still no."
    assert len(set(answers)) == 4, "each repeat must say something new"


def test_sudo_keeps_its_promise():
    room = build_engine()
    first = room.command("sudo")
    assert "sudoers" in first.lines[0]
    assert "reported to The-Masked-Bear" in first.lines[1]


def test_a_key_sequence_fires_at_its_tail(clock):
    room = engine(clock=clock)
    for key in ("x", "y", "up", "up", "down", "down", "left", "right", "left", "right", "b"):
        assert room.key(key) is None
    assert room.key("a") is not None, "the konami sequence should complete"


def test_a_wrong_sequence_does_nothing(clock):
    room = engine(clock=clock)
    for key in ("up", "down", "up", "down", "b", "a"):
        assert room.key(key) is None


def test_counted_eggs_fire_exactly_on_the_threshold(clock):
    egg = Egg("hundred", "Hundred", Trigger.count("tool_call", 3), text_egg("THREE", "third call"))
    room = engine(egg, clock=clock, cooldown=0.0)
    assert room.event("tool_call") is None
    assert room.event("tool_call") is None
    assert room.event("tool_call") is not None
    assert room.event("tool_call") is None, "it must not keep firing afterwards"


def test_milestones_fire_once_when_not_repeatable(clock):
    egg = Egg("first", "First", Trigger.milestone("branch_passed"),
              text_egg("FIRST BLOOD", "nice"), achievement=True, repeatable=False)
    room = engine(egg, clock=clock, cooldown=0.0)
    assert room.event("branch_passed") is not None
    assert room.event("branch_passed") is None


def test_chance_eggs_respect_their_probability(clock):
    never = Egg("never", "Never", Trigger.chance("tick", 0.0), text_egg("NO", "no"))
    always = Egg("always", "Always", Trigger.chance("tock", 1.0), text_egg("YES", "yes"))
    room = engine(never, always, clock=clock, cooldown=0.0)
    assert all(room.event("tick") is None for _ in range(50))
    assert room.event("tock") is not None


def test_time_eggs_read_the_clock():
    egg = Egg("threeam", "3am", Trigger.moment(lambda t: t.tm_hour == 3), text_egg("03:00", "go to bed"))
    room = engine(egg, cooldown=0.0)
    assert room.tick(time.struct_time((2026, 8, 17, 14, 0, 0, 0, 229, 0))) is None
    assert room.tick(time.struct_time((2026, 8, 17, 3, 0, 0, 0, 229, 0))) is not None


def test_idle_eggs_wait_for_silence(clock):
    egg = Egg("quiet", "Quiet", Trigger.idle(60.0), text_egg("STILL HERE", "hello?"))
    room = engine(egg, clock=clock, cooldown=0.0)
    clock.advance(30)
    assert room.tick() is None
    clock.advance(31)
    assert room.tick() is not None


def test_typing_resets_the_idle_timer(clock):
    egg = Egg("quiet", "Quiet", Trigger.idle(60.0), text_egg("STILL HERE", "hello?"))
    room = engine(egg, clock=clock, cooldown=0.0)
    clock.advance(59)
    room.touch()
    clock.advance(30)
    assert room.tick() is None


# -- the rules that keep eggs safe ------------------------------------------


def test_nothing_fires_while_the_user_is_reading_an_error(clock):
    egg = Egg("always", "Always", Trigger.chance("tool_call", 1.0), text_egg("HI", "hello"))
    room = engine(egg, clock=clock, cooldown=0.0)
    assert room.event("tool_call", suppress=True) is None, "an egg must never sit on top of an error"
    assert room.event("tool_call") is not None


def test_a_cooldown_stops_egg_spam(clock):
    egg = Egg("always", "Always", Trigger.chance("tool_call", 1.0), text_egg("HI", "hello"))
    room = engine(egg, clock=clock, cooldown=20.0)
    assert room.event("tool_call") is not None
    assert room.event("tool_call") is None
    clock.advance(21)
    assert room.event("tool_call") is not None


def test_typed_commands_ignore_the_cooldown(clock):
    """An egg the user asked for by name should always answer."""
    room = build_engine(clock=clock, cooldown=999.0)
    assert room.command("ping") is not None
    assert room.command("uptime") is not None


def test_suppressed_events_still_count(clock):
    egg = Egg("hundred", "Hundred", Trigger.count("tool_call", 2), text_egg("TWO", "second"))
    room = engine(egg, clock=clock, cooldown=0.0)
    room.event("tool_call", suppress=True)
    assert room.counter("tool_call") == 1


# -- discovery and persistence ----------------------------------------------


def test_discovery_is_recorded_and_reloaded(tmp_path):
    store = tmp_path / "eggs.json"
    room = build_engine(store)
    room.command("bear")
    room.command("ping")
    assert room.discovered == {"bear", "ping"}

    again = build_engine(store)
    assert again.discovered == {"bear", "ping"}
    assert again.progress() == (2, len(CATALOGUE))


def test_counters_survive_a_restart(tmp_path):
    store = tmp_path / "eggs.json"
    room = build_engine(store)
    for _ in range(3):
        room.command("bear")
    assert build_engine(store).counter("cmd:bear") == 3


def test_a_corrupt_store_is_ignored_not_fatal(tmp_path):
    store = tmp_path / "eggs.json"
    store.write_text("{ not json", encoding="utf-8")
    room = build_engine(store)
    assert room.discovered == set()
    room.command("bear")
    assert json.loads(store.read_text())["found"]


def test_the_trophy_room_lists_everything(tmp_path):
    room = build_engine(tmp_path / "eggs.json")
    room.command("bear")
    trophies = room.trophies()
    assert len(trophies) == len(CATALOGUE)
    assert sum(1 for _, found in trophies if found) == 1


def test_progress_starts_empty():
    found, total = build_engine().progress()
    assert found == 0 and total >= 40


# -- the deep one -----------------------------------------------------------


def test_the_three_step_egg_needs_all_three_steps():
    room = build_engine()
    assert "There is no rabbit." in room.command("rabbit").lines
    room.command("matrix")
    room.command("follow")
    assert "DEEP EGG 1 OF 3 FOUND." in room.command("rabbit").lines


def test_an_egg_cannot_touch_anything_real():
    """Reveals are data. There is no hook here that could mutate state."""
    room = build_engine()
    got = room.command("gravity")
    assert isinstance(got, Reveal) and got.animated
    assert not hasattr(got, "apply") and not callable(getattr(got, "run", None))
