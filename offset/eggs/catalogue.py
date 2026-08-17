"""The catalogue.

Voice rules, taken from the owner's own work: deadpan, short, never winking too
hard.  The joke is that the machine answers seriously.  Nothing here is
placeholder copy — a line that is not funny should be deleted, not softened.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from offset.eggs.engine import Egg, EggEngine, Reveal, Trigger, text_egg

BEAR = [
    "   \u2584\u2584\u2584     \u2584\u2584\u2584   ",
    "  \u2588\u2588\u2588\u2588\u2584\u2584\u2584\u2584\u2584\u2588\u2588\u2588\u2588  ",
    " \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588 ",
    " \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588 ",
    " \u2588\u2588\u2588\u2588\u2588 \u2588\u2588 \u2588\u2588\u2588\u2588\u2588 ",
    " \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588 ",
    "  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588  ",
]

GLYPHS = "01\u2588\u2593\u2592\u2591<>/\\{}[]=+*#@$%&"


def _matrix_frames(rng: random.Random, width: int = 46, rows: int = 8, count: int = 12) -> list[list[str]]:
    return [
        ["".join(rng.choice(GLYPHS) if rng.random() < 0.35 else " " for _ in range(width)) for _ in range(rows)]
        for _ in range(count)
    ]


def _gravity_frames(text: str = "EVERYTHING FALLS DOWN EVENTUALLY", rows: int = 7) -> list[list[str]]:
    frames: list[list[str]] = []
    for step in range(rows):
        frame = [""] * rows
        frame[min(step, rows - 1)] = text
        frames.append(frame)
    frames.append([""] * (rows - 1) + [text.replace(" ", "_")])
    return frames


def _times(context: dict[str, Any], name: str) -> int:
    engine = context.get("engine")
    return engine.counter(name) if isinstance(engine, EggEngine) else 1


# -- the escalating ones ----------------------------------------------------


def _bear(context: dict[str, Any]) -> Reveal:
    """Four stages.  The FAQ answer on his site was already the joke."""
    n = _times(context, "cmd:bear")
    if n <= 1:
        return Reveal("", "ARE YOU A BEAR?", ["No."], "plain", duration=2.5)
    if n == 2:
        return Reveal("", "ARE YOU A BEAR?", ["Still no."], "plain", duration=2.5)
    if n == 3:
        return Reveal("", "ARE YOU A BEAR?", ["...", "Define bear."], "muted", duration=3.0)
    return Reveal("", "FINE. YES.", [*BEAR, "", "This changes nothing about the code review."], "branch", duration=6.0)


def _sudo(context: dict[str, Any]) -> Reveal:
    n = _times(context, "cmd:sudo")
    if n <= 1:
        return Reveal("", "PERMISSION DENIED", [
            "guest is not in the sudoers file.",
            "This incident will be reported to The-Masked-Bear.",
        ], "err", duration=4.5)
    if n == 2:
        return Reveal("", "STILL DENIED", ["The incident was reported.", "He read it. He said no."], "err")
    if n == 3:
        return Reveal("", "DENIED, WITH FEELING", [
            "Escalation noted.",
            "sudo attempts: 3.  sudo successes: 0.",
        ], "err")
    return Reveal("", "ROOT GRANTED", [
        "Just kidding.",
        "You have root over this dialog box and nothing else.",
        "Use it wisely.",
    ], "accent", duration=5.0)


def _matrix(_context: dict[str, Any]) -> Reveal:
    return Reveal("", "WAKE UP, OPERATOR", ["follow the white rabbit"], "ok",
                  frames=_matrix_frames(random.Random(int(time.time()))), duration=3.0)


def _rabbit(context: dict[str, Any]) -> Reveal:
    if _times(context, "cmd:follow") == 0:
        return Reveal("", "WHAT RABBIT", ["There is no rabbit.", "Type `follow` first."], "muted", duration=3.0)
    return Reveal("", "YOU FOLLOWED IT", [
        "Down here the code compiles on the first try.",
        "Nobody gets to stay.",
        "",
        "DEEP EGG 1 OF 3 FOUND.",
    ], "branch", duration=6.0)


def _neofetch(context: dict[str, Any]) -> Reveal:
    engine = context.get("engine")
    found, total = engine.progress() if isinstance(engine, EggEngine) else (0, 0)
    facts = [
        "operator@offset",
        "---------------",
        "OS      : offset 0.1.0",
        "Style   : neubrutalist",
        "WM      : hard borders, zero blur",
        "Theme   : #FFDE59 on #111111",
        "Corners : 0px, non-negotiable",
        f"Eggs    : {found}/{total} found",
        "Uptime  : long enough",
    ]
    art = BEAR + [""] * max(0, len(facts) - len(BEAR))
    return Reveal("", "NEOFETCH", [f"{a:<18}{f}" for a, f in zip(art, facts)], "info", duration=8.0)


def _konami(context: dict[str, Any]) -> Reveal:
    return Reveal("", "30 LIVES GRANTED", [
        "You now have thirty attempts at this refactor.",
        "You will need four.",
    ], "branch", duration=5.0)


CATALOGUE: list[Egg] = [
    # -- the fake shell, ported from the reference site ---------------------
    Egg("bear", "Are you a bear?", Trigger.command("bear"), _bear, hint="ask four times"),
    Egg("sudo", "Sudo", Trigger.command("sudo"), _sudo, hint="insist"),
    Egg("matrix", "Wake up", Trigger.command("matrix"), _matrix, hint="then follow"),
    Egg("follow", "Follow it", Trigger.command("follow"),
        text_egg("FOLLOWING", "The rabbit went that way.", "Type `rabbit` when you catch up.", tone="ok")),
    Egg("rabbit", "Down the hole", Trigger.command("rabbit"), _rabbit, hint="matrix, then follow, then rabbit"),
    Egg("gravity", "Gravity", Trigger.command("gravity"),
        lambda _c: Reveal("", "GRAVITY ENABLED", [], "err", frames=_gravity_frames(), duration=3.0)),
    Egg("neofetch", "Neofetch", Trigger.command("neofetch", "fetch"), _neofetch),
    Egg("hunter2", "hunter2", Trigger.command("hunter2", "password"),
        text_egg("ACCESS DENIED", "KEY: HUNTER2", "Everyone else in this room just sees *******.", tone="err")),
    Egg("whoami", "Who am I", Trigger.command("whoami"),
        text_egg("WHOAMI", "An operator with four branches open and no plan.", tone="info")),
    Egg("uptime", "Uptime", Trigger.command("uptime"),
        text_egg("UPTIME", "up 3 days, 4 users, load average: yes, yes, yes", tone="info")),
    Egg("uname", "Uname", Trigger.command("uname"),
        text_egg("UNAME -A", "offset 0.1.0 #1 SMP PREEMPT NEUBRUTALIST aarch64 GNU/Probably", tone="info")),
    Egg("ping", "Ping", Trigger.command("ping"),
        text_egg("PING", "pong. 0.00 ms.", "It is a local variable.", tone="ok")),
    Egg("rm", "rm -rf /", Trigger.command("rm"),
        text_egg("NICE TRY", "Deleting everything...", "...just kidding.",
                 "Your files are exactly where you left them.", tone="err")),
    Egg("ls", "ls", Trigger.command("ls", "dir"),
        text_egg("LS", "branches/  regrets/  tests_that_pass/  tests_that_pass_locally/", tone="info")),
    Egg("cd", "cd", Trigger.command("cd"),
        text_egg("CD", "You cannot leave. There is only this repository now.", tone="muted")),
    Egg("exit", "There is no exit", Trigger.command("exit", "quit"),
        text_egg("THERE IS NO EXIT", "There is only ctrl-c, and disappointment.", tone="muted")),
    Egg("vim", "Editor wars", Trigger.command("vim", "emacs", "nano"),
        text_egg("EDITOR WARS", "You are inside an editor with strong opinions about corners.",
                 "Pick your battles.", tone="accent")),
    Egg("make", "Make me a sandwich", Trigger.command("make"),
        text_egg("MAKE", "make: *** No rule to make target 'sandwich'.  Stop.", tone="err")),
    Egg("42", "The answer", Trigger.command("42", "answer", "meaning"),
        text_egg("42", "The answer is 42.", "The question is still in code review.", tone="accent")),
    Egg("hello", "Hello", Trigger.command("hello", "hi", "hey"),
        text_egg("HELLO", "Hello.", "I was hoping for something more specific.", tone="plain")),
    Egg("please", "Manners", Trigger.command("please"),
        text_egg("MANNERS NOTED", "Politeness does not increase the token budget.",
                 "It is still appreciated.", tone="ok")),
    Egg("why", "Why", Trigger.command("why"),
        text_egg("WHY", "Because the first approach failed and nobody wrote down why.", tone="muted")),
    Egg("blame", "Blame", Trigger.command("blame"),
        text_egg("GIT BLAME", "It was you.", "Four minutes ago. On branch C.", tone="err")),
    Egg("coffee", "Coffee", Trigger.command("coffee", "brew"),
        text_egg("HTCPCP 418", "I'm a teapot.",
                 "I am also a Raspberry Pi. Do not pour anything into me.", tone="err")),
    Egg("hack", "Hack the planet", Trigger.command("hack"),
        text_egg("HACKING THE PLANET", "Progress: 100%.", "Planet: unchanged.", tone="ok")),
    Egg("konami", "The old sequence",
        Trigger.sequence(("up", "up", "down", "down", "left", "right", "left", "right", "b", "a")),
        _konami, hint="you know the one"),

    # -- self-aware ---------------------------------------------------------
    Egg("sentient", "Sentience check", Trigger.command("sentient", "conscious"),
        text_egg("NO", "I am a very confident autocomplete with filesystem access.",
                 "That should worry you slightly.", tone="muted")),
    Egg("trust", "Trust", Trigger.command("trust"),
        text_egg("TRUST, BUT VERIFY", "Four models agreed with each other.",
                 "That is not the same as being right.", tone="accent")),
    Egg("sorry", "Apology", Trigger.command("sorry"),
        text_egg("APOLOGY ACCEPTED", "The branch you abandoned is still in the session tree.",
                 "Nothing is ever really gone here.", tone="ok")),
    Egg("honest", "Honesty", Trigger.command("honest", "truth"),
        text_egg("HONESTLY", "Two of the four branches were the same idea with different variable names.",
                 tone="muted")),

    # -- achievements, earned by real milestones ----------------------------
    Egg("first-blood", "First blood", Trigger.milestone("branch_passed"),
        text_egg("FIRST BLOOD", "A speculative branch passed its tests.",
                 "You may now argue with the other three.", tone="ok"),
        achievement=True, repeatable=False),
    Egg("hung-jury", "Hung jury", Trigger.milestone("models_disagreed"),
        text_egg("HUNG JURY", "Every model gave a different answer.",
                 "That is information, not failure.", tone="err"),
        achievement=True, repeatable=False),
    Egg("unanimous", "Suspiciously unanimous", Trigger.milestone("models_agreed"),
        text_egg("SUSPICIOUSLY UNANIMOUS", "Every model returned the same answer.",
                 "Either it is obvious, or they share a training set.", tone="info"),
        achievement=True, repeatable=False),
    Egg("byo", "Bring your own", Trigger.milestone("custom_tool_loaded"),
        text_egg("BRING YOUR OWN", "You loaded a tool you wrote yourself.",
                 "That is the entire point.", tone="branch"),
        achievement=True, repeatable=False),
    Egg("second-thoughts", "Second thoughts", Trigger.milestone("turn_cancelled"),
        text_egg("SECOND THOUGHTS", "You cancelled mid-turn.",
                 "The results so far were kept anyway.", tone="muted"),
        achievement=True, repeatable=False),
    Egg("clean-sweep", "Clean sweep", Trigger.milestone("all_branches_passed"),
        text_egg("CLEAN SWEEP", "Every branch passed.", "Now the hard part: choosing.", tone="ok"),
        achievement=True, repeatable=False),
    Egg("manual-labour", "Manual labour", Trigger.count("tool_call", 100),
        text_egg("MANUAL LABOUR", "One hundred tool calls.",
                 "Consider writing one tool that does all of it.", tone="accent"),
        achievement=True, repeatable=False),
    Egg("centurion", "Centurion", Trigger.count("tool_call", 1000),
        text_egg("CENTURION", "One thousand tool calls in this workspace.",
                 "The Pi is holding up better than expected.", tone="accent"),
        achievement=True, repeatable=False),
    Egg("parallel-universe", "Parallel universe", Trigger.count("branch_created", 10),
        text_egg("PARALLEL UNIVERSE", "Ten speculative branches.",
                 "Nine were wrong. That was the plan.", tone="branch"),
        achievement=True, repeatable=False),
    Egg("archivist", "Archivist", Trigger.count("session_forked", 5),
        text_egg("ARCHIVIST", "Five forked sessions.", "History is cheap. Use more of it.", tone="info"),
        achievement=True, repeatable=False),
    Egg("polyglot", "Polyglot", Trigger.count("model_changed", 6),
        text_egg("POLYGLOT", "Six different models this session.", "Somebody is shopping around.", tone="info"),
        achievement=True, repeatable=False),
    Egg("time-traveller", "Time traveller", Trigger.count("tree_navigated", 20),
        text_egg("TIME TRAVELLER", "Twenty jumps around the session tree.",
                 "The leaf moved; nothing was lost. That is the design.", tone="branch"),
        achievement=True, repeatable=False),

    # -- the clock and the calendar -----------------------------------------
    Egg("nocturnal", "Nocturnal", Trigger.moment(lambda t: t.tm_hour == 3),
        text_egg("03:00", "Nothing good is merged at three in the morning.",
                 "The branch will still be there tomorrow.", tone="muted"),
        achievement=True, repeatable=False),
    Egg("friday13", "Friday the thirteenth", Trigger.moment(lambda t: t.tm_mday == 13 and t.tm_wday == 4),
        text_egg("FRIDAY THE 13TH", "Deploy freeze recommended.",
                 "Speculative branches: encouraged.", tone="err")),
    Egg("newyear", "New year", Trigger.moment(lambda t: t.tm_mon == 1 and t.tm_mday == 1),
        text_egg("NEW YEAR", "Same repository. New optimism.", tone="accent")),
    Egg("halloween", "Halloween", Trigger.moment(lambda t: t.tm_mon == 10 and t.tm_mday == 31),
        text_egg("SPOOKY", "The scariest thing here is the untested branch.", tone="branch")),
    Egg("leapday", "Leap day", Trigger.moment(lambda t: t.tm_mon == 2 and t.tm_mday == 29),
        text_egg("LEAP DAY", "An extra day of tokens.", "Spend it badly.", tone="accent")),
    Egg("saturday-night", "Saturday night", Trigger.moment(lambda t: t.tm_wday == 5 and t.tm_hour >= 22),
        text_egg("SATURDAY NIGHT", "The tests do not know what day it is.",
                 "They still have to pass.", tone="muted")),

    # -- rare and random ----------------------------------------------------
    Egg("wandering-bear", "A bear walks past", Trigger.chance("tool_call", 0.002),
        lambda _c: Reveal("", "", BEAR, "muted", duration=2.0), hint="keep working"),
    Egg("cosmic-ray", "Cosmic ray", Trigger.chance("tool_call", 0.001),
        text_egg("BIT FLIP DETECTED", "A cosmic ray flipped a bit somewhere.",
                 "Probably not in your code. Probably.", tone="err")),
    Egg("compliment", "Unsolicited compliment", Trigger.chance("turn_finished", 0.01),
        text_egg("FOR WHAT IT IS WORTH", "That was a reasonable prompt.",
                 "Most of them are not.", tone="ok")),

    # -- going quiet --------------------------------------------------------
    Egg("still-here", "Still here", Trigger.idle(600.0),
        text_egg("STILL HERE", "Ten minutes of nothing.",
                 "The branches are waiting. So am I.", tone="muted")),
    Egg("an-hour", "An hour", Trigger.idle(3600.0),
        text_egg("AN HOUR", "I have been looking at this prompt for an hour.",
                 "One of us should do something.", tone="muted")),
]


def build_engine(store: Path | str | None = None, *, rng: random.Random | None = None, **kwargs: Any) -> EggEngine:
    """The catalogue, wired up and on by default, like everything else."""
    return EggEngine(CATALOGUE, store=store, rng=rng, **kwargs)
