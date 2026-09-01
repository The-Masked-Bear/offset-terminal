"""Watching a generation while it is still being generated.

A model that has gone wrong is usually obvious long before it stops.  It
repeats a sentence forever, it claims to have read a file that is not there, it
announces that the tests passed when the recorded output of the test run says
`FAILED`.  Every one of those is visible in the first few hundred tokens, and
every one of them currently costs the user the whole response — because nothing
looks at a stream until it has ended.  This module looks at it as it arrives
and can stop it.

The design follows from one requirement: *stopping early is the entire point,
so the watcher must not itself cost anything.*  `Auditor.watch` is a generator
that yields each upstream event onward before it inspects anything, so
pass-through latency is a function call and nothing else, and no event is ever
withheld while a decision is pending.  Nothing is buffered beyond a bounded
tail of the text — a fixed window, not the response — so a very long answer
does not turn the auditor into an O(n^2) scan.  The cheap checks are pure
functions of that window and are the only work done on the hot path; the
expensive check is a second model, which runs on a worker and is sampled, so it
can be arbitrarily slow without the stream noticing.

**The real risk here is false positives, not missed detections.**  A missed
runaway wastes tokens.  A wrongly halted generation destroys work the user
asked for and, worse, teaches them to switch the feature off — at which point
it protects nobody.  So the whole module is biased towards silence:

  * Every check returns a *confidence*, and every check carries its own
    *threshold*.  A verdict below threshold is recorded for a human to look at
    and nothing else happens.
  * Confidence is a real quantity, not a vibe.  The repetition detector reports
    what fraction of the recent window one repeated phrase accounts for, so
    "0.95" means ninety-five per cent of the last few hundred words were the
    same phrase, which is not a judgement call.
  * The two checks that cannot be sure — fabricated paths and contradictions —
    are **off by default**.  A model may legitimately name a file it is about
    to create, and a paraphrase is not a contradiction.  They are opt-in per
    session, and the defaults live in `DEFAULT_ON`.
  * Anything that cannot be judged is treated as fine.  A path outside the
    workspace, an `OSError` from a stat, a second model that raises or never
    answers: all of these produce no verdict at all rather than a guess.
  * A halt always names the exact evidence — the repeated phrase, the path, the
    claim and the recorded output it contradicts — because a human has to be
    able to overrule it, and "the auditor did not like it" is not reviewable.

A halt is an event, not an exception.  `Halted` is yielded and the generator
returns; the agent loop sees a stream that ended.  Raising would abort a turn
whose text, tool calls and usage have already been recorded, turning a
precaution into data loss.  A `Stop("halted")` is emitted just before it for
the same reason: `TurnBuilder` does not know about `Halted`, so without the
`Stop` a halted generation would be assembled and persisted with
`stop_reason == "stop"` and look, forever, like a clean finish.

One consequence the caller must handle: a verdict from the second model can
land at any chunk boundary, including one in the middle of a tool call's
arguments.  `Halted.mid_tool_call` says so, and the caller must not dispatch a
tool call from a turn that was halted mid-call — the arguments are half a JSON
document.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Iterator, Protocol, Sequence

from offset.providers.base import (
    Event,
    Stop,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    Usage,
)

#: Check names.  Used as verdict labels and as the identifiers a user types to
#: turn one on, so they are short and stable.
REPETITION: Final = "repetition"
FABRICATED_PATH: Final = "path"
CONTRADICTION: Final = "contradiction"
MODEL: Final = "model"

#: What runs unless asked otherwise.  Only repetition: its confidence is a
#: measured ratio rather than an inference, so it is the one cheap check that
#: can be trusted without the user opting in.  See the module docstring.
DEFAULT_ON: Final = (REPETITION,)

#: The stop reason recorded for a halted turn, so history does not remember it
#: as a clean finish.
HALT_REASON: Final = "halted"

#: Characters of accumulated text the cheap checks look at.  Bounded on
#: purpose: the checks run on every chunk, so an unbounded scan would make the
#: auditor quadratic in the length of the response - exactly the long answers
#: it is meant to make cheaper.  Four thousand characters is roughly six
#: hundred words, which is more than enough context for any of these checks.
TAIL: Final = 4000

#: Characters per token, matching `offset.tools.retrieve`.  Only used for
#: debouncing the sampler and for reporting, never for billing.
CHARS_PER_TOKEN: Final = 4

#: Words in the n-gram the repetition detector counts.  Six is long enough
#: that ordinary prose does not repeat one by accident and short enough to
#: catch a looping clause rather than only a looping paragraph.
GRAM: Final = 6

#: How many times a unit must recur before the detector will speak at all.
#: Three repeats happen in legitimate writing (lists, enumerations); four in a
#: few-hundred-word window does not.
MIN_REPEATS: Final = 4

#: Fraction of the window one repeated unit must account for.  Half the recent
#: output being literally the same phrase is a runaway, not a style.
REPETITION_THRESHOLD: Final = 0.5

#: Confidence for a fabricated path that names a directory as well as a file.
#: A bare filename is much weaker evidence - it may be shorthand for a file
#: that does exist elsewhere - so it scores below the threshold and is only
#: ever recorded, never acted on.
PATH_STRONG: Final = 0.9
PATH_WEAK: Final = 0.7
PATH_THRESHOLD: Final = 0.8

#: Characters allowed between "I read" and the path it refers to.  Wide enough
#: for "I have already read through the file", narrow enough that an unrelated
#: sentence two clauses later is not attributed to the verb.
CLAIM_WINDOW: Final = 64

CONTRADICTION_THRESHOLD: Final = 0.8

#: Tokens between second-model samples.  The judge costs a request, so it is
#: sampled rather than run per chunk; four hundred tokens is a few sentences,
#: which is the granularity at which a generation visibly goes wrong.
SAMPLE_EVERY: Final = 400

#: Tokens before the first sample.  Judging two words wastes a request and
#: invites a verdict on text that has not said anything yet.
SAMPLE_WARMUP: Final = 120

#: Fallback for "how many tokens would this turn have spent".  Used only to
#: report a saving, and deliberately an upper bound; the caller should pass the
#: model's real `max_output`.
DEFAULT_MAX_OUTPUT: Final = 8192

_WORD = re.compile(r"[\w']+")
_SENTENCE = re.compile(r"[.!?\n]+")

#: A path is only recognised when it carries a known extension.  Requiring one
#: is what keeps this check quiet: without it the pattern matches "and/or",
#: version numbers and every URL fragment, and a detector that fires on those
#: would be switched off within a minute.
EXTENSIONS: Final = (
    "py pyi pyx md rst txt json toml yaml yml cfg ini env sh bash zsh fish "
    "js jsx ts tsx mjs cjs rs go rb java kt swift c h cc cpp hpp cs php lua "
    "sql html css scss lock log csv tsv xml svg"
).split()

_PATH = re.compile(r"[\w.\-/]*[\w\-]\.(?:" + "|".join(EXTENSIONS) + r")\b")

#: Claims of a *completed* read or edit.  Verbs of intent - create, write,
#: add, will - are absent by construction: "I will create offset/new.py" is
#: correct precisely when the file does not exist, and flagging it would make
#: the check fire hardest on the model doing its job properly.
ASSERTION = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|had\s+|just\s+|already\s+|then\s+)*"
    r"(?:read|re-?read|opened|reviewed|inspected|examined|edited|modified"
    r"|updated|patched|changed|deleted|removed|renamed)\b"
    r"|\b(?:read|edited|modified|updated|deleted)\s+(?:the\s+|from\s+)?file\b"
    r"|\bas\s+(?:seen|shown|written|defined)\s+in\b"
    r"|\baccording\s+to\b",
    re.IGNORECASE,
)


# -- verdicts ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """One check's opinion, with the evidence a human needs to overrule it."""

    check: str
    confidence: float
    reason: str
    evidence: str

    def line(self) -> str:
        return f"{self.check} {self.confidence:.2f}  {self.reason}  [{self.evidence}]"


@dataclass(slots=True)
class Halted(Event):
    """The stream was stopped.  Carries why, and what it was looking at.

    `tokens_saved` is an upper bound, not a measurement: the true saving is
    however many tokens the generation would still have produced, which is
    unknowable, so the budget ceiling stands in for it and is reported as a
    ceiling.
    """

    check: str
    reason: str
    evidence: str
    confidence: float = 1.0
    tokens_seen: int = 0
    tokens_saved: int = 0
    mid_tool_call: bool = False

    def line(self) -> str:
        tail = " mid tool call" if self.mid_tool_call else ""
        return (
            f"halted by {self.check} ({self.confidence:.2f}): {self.reason}{tail}\n"
            f"  evidence: {self.evidence}\n"
            f"  {self.tokens_seen} tokens spent, up to {self.tokens_saved} saved"
        )


@dataclass(frozen=True, slots=True)
class Report:
    """What the auditor did, for the UI and for judging the auditor itself."""

    tokens_seen: int
    tokens_saved: int
    halted: Halted | None
    verdicts: tuple[Verdict, ...] = ()
    samples: int = 0
    errors: tuple[str, ...] = ()

    def line(self) -> str:
        state = f"halted by {self.halted.check}" if self.halted else "clean"
        return (
            f"{state}; {self.tokens_seen} tokens seen, {self.tokens_saved} saved, "
            f"{len(self.verdicts)} verdict(s), {self.samples} sample(s)"
        )


class Check(Protocol):
    """A cheap, synchronous, pure inspection of the text so far.

    `visible_only` distinguishes a check about the answer from a check about
    the generation.  A runaway loop is just as expensive inside thinking, so
    the repetition detector watches both; a claim to have read a file is only a
    claim when it is said out loud, and treating a thought as an assertion
    would punish the model for considering possibilities.
    """

    name: str
    threshold: float
    enabled: bool
    visible_only: bool

    def __call__(self, text: str) -> Verdict | None: ...


# -- shared helpers ---------------------------------------------------------


def _tail(text: str, limit: int) -> str:
    """The last `limit` characters, cut at a whitespace boundary.

    The boundary matters: slicing blindly can start the window in the middle of
    a path, so `offset/core/audit.py` arrives at the fabricated-path check as
    `re/audit.py`, which does not exist and would be reported as a
    hallucination the model never made.
    """
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = re.search(r"\s", cut)
    return cut[space.end() :] if space else cut


def _settled(match: re.Match[str], text: str) -> bool:
    """True if the match cannot still be growing.

    A delta boundary can fall anywhere, so a match that ends exactly at the end
    of the buffer may be half of something: `offset/core/aud` matches the path
    pattern once `aud.py` has not arrived yet, and stat-ing it would report a
    fabricated path mid-word.  One further character is enough proof.
    """
    return match.end() < len(text)


def _excerpt(text: str, match: re.Match[str], span: int = 32) -> str:
    start = max(0, match.start() - span)
    end = min(len(text), match.end() + span)
    return " ".join(text[start:end].split())


def tokens_for(chars: int) -> int:
    return chars // CHARS_PER_TOKEN


def workspace_exists(root: Path | str) -> Callable[[str], bool]:
    """A file-existence oracle for one workspace, which never raises.

    Every unjudgeable answer is "it exists", i.e. no complaint: a path outside
    the workspace, an unreadable directory or a name the filesystem rejects
    tells us nothing about whether the model was lying, and guessing in those
    cases would make the check fire on symlinked or mounted trees where it is
    least likely to be right.
    """
    base = Path(root).resolve()

    def exists(candidate: str) -> bool:
        try:
            raw = Path(candidate)
            target = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
            if not target.is_relative_to(base):
                return True
            return target.exists()
        except (OSError, ValueError, RuntimeError):
            # Cannot judge (bad name, loop, too long) - so do not.
            return True

    return exists


# -- the cheap checks -------------------------------------------------------


@dataclass(slots=True)
class Repetition:
    """The classic runaway: one phrase, forever.

    Confidence is coverage - the share of the window taken up by the single
    most repeated unit - which is why this check can be on by default.  A list
    that repeats a clause four times in six hundred words scores about 0.04 and
    says nothing; a stuck decoder scores about 1.0.
    """

    name: str = REPETITION
    threshold: float = REPETITION_THRESHOLD
    enabled: bool = True
    visible_only: bool = False
    gram: int = GRAM
    min_repeats: int = MIN_REPEATS
    tail: int = TAIL

    def __call__(self, text: str) -> Verdict | None:
        window = _tail(text, self.tail)
        words = _WORD.findall(window.lower())
        # Below this there is not enough text for coverage to mean anything: a
        # greeting repeated twice would score 1.0 on a twelve-word buffer.
        if len(words) < self.gram * self.min_repeats:
            return None

        best = self._grams(words)
        sentence = self._sentences(window, len(words))
        if sentence is not None and (best is None or sentence[0] > best[0]):
            best = sentence
        if best is None:
            return None

        coverage, unit, repeats = best
        return Verdict(
            check=self.name,
            confidence=coverage,
            reason=(
                f"one phrase repeated {repeats} times accounts for "
                f"{coverage:.0%} of the last {len(words)} words"
            ),
            evidence=f"{unit[:120]!r} x{repeats}",
        )

    def _grams(self, words: list[str]) -> tuple[float, str, int] | None:
        n = self.gram
        counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        unit, repeats = counts.most_common(1)[0]
        if repeats < self.min_repeats:
            return None
        # Overlapping windows can count more words than the buffer holds, hence
        # the clamp; coverage is a share, not a tally.
        return min(1.0, repeats * n / len(words)), " ".join(unit), repeats

    def _sentences(self, window: str, total: int) -> tuple[float, str, int] | None:
        """Catches a repeated unit shorter than one n-gram ("Done." "Done.")."""
        counts: Counter[str] = Counter()
        sizes: dict[str, int] = {}
        for piece in _SENTENCE.split(window):
            norm = " ".join(_WORD.findall(piece.lower()))
            if not norm:
                continue
            counts[norm] += 1
            sizes[norm] = norm.count(" ") + 1
        if not counts:
            return None
        unit, repeats = counts.most_common(1)[0]
        if repeats < self.min_repeats:
            return None
        return min(1.0, repeats * sizes[unit] / total), unit, repeats


@dataclass(slots=True)
class FabricatedPath:
    """The model says it read or edited a file that is not there.

    Off by default.  The failure mode it guards against is real and expensive -
    a whole answer reasoning about the contents of a file that does not exist -
    but the surrounding language is ambiguous enough that this cannot be
    trusted unsupervised, so it must be asked for.
    """

    exists: Callable[[str], bool]
    name: str = FABRICATED_PATH
    threshold: float = PATH_THRESHOLD
    enabled: bool = False
    visible_only: bool = True
    tail: int = TAIL
    window: int = CLAIM_WINDOW

    def __call__(self, text: str) -> Verdict | None:
        haystack = _tail(text, self.tail)
        claims = [m.end() for m in ASSERTION.finditer(haystack)]
        if not claims:
            return None

        best: Verdict | None = None
        for match in _PATH.finditer(haystack):
            if not _settled(match, haystack):
                continue
            anchor = self._claim_before(claims, match.start(), haystack)
            if anchor is None:
                continue
            path = match.group(0).strip("`'\"")
            try:
                if self.exists(path):
                    continue
            except Exception:
                # An oracle that cannot answer is not evidence of a lie.
                continue
            confidence = PATH_STRONG if "/" in path else PATH_WEAK
            if best is not None and confidence <= best.confidence:
                continue
            best = Verdict(
                check=self.name,
                confidence=confidence,
                reason=f"claims to have read or changed {path!r}, which is not in the workspace",
                evidence=" ".join(haystack[anchor : match.end()].split()),
            )
        return best

    def _claim_before(self, claims: Sequence[int], start: int, text: str) -> int | None:
        """The nearest completed-action verb close enough to own this path.

        A newline between the two ends the attribution: a heading followed by a
        file listing is not a claim about the listing.
        """
        for end in reversed(claims):
            if end > start:
                continue
            if start - end > self.window:
                return None
            return None if "\n" in text[end:start] else end
        return None


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion, and what in a recorded tool result would refute it.

    Deliberately a small table rather than general reasoning.  Detecting
    arbitrary contradictions needs a model, which is what `Sampled` is for;
    what belongs on the hot path is the handful of claims that are both common
    and mechanically checkable.
    """

    name: str
    says: re.Pattern[str]
    refuted_by: re.Pattern[str]
    reason: str
    confidence: float = 0.85
    compare: bool = False

    def refutation(self, said: re.Match[str], output: str) -> str | None:
        """The refuting excerpt of `output`, or None if it agrees."""
        if self.compare:
            # Both patterns capture the same quantity; a contradiction is the
            # model naming a number the recorded output never contained.
            if said.lastindex is None:
                return None
            claimed = said.group(1)
            found = {m.group(1) for m in self.refuted_by.finditer(output)}
            if not found or claimed in found:
                return None
            return f"recorded {sorted(found)[0]}"
        hit = self.refuted_by.search(output)
        return None if hit is None else _excerpt(output, hit)


#: The default claim table.  Every entry is a claim offset has actually seen a
#: model make about output that said the opposite.
CLAIMS: Final = (
    Claim(
        name="tests-pass",
        says=re.compile(
            r"\b(?:all\s+)?(?:the\s+)?tests?\s+(?:now\s+)?"
            r"(?:pass(?:ed|es)?|are\s+passing|succeed(?:ed)?)\b",
            re.IGNORECASE,
        ),
        # `FAILED` and `Traceback` are case sensitive on purpose: they are
        # pytest's own markers, whereas a lower-case "failed" appears in prose
        # and in "0 failed", which is a pass.
        refuted_by=re.compile(r"\b[1-9]\d*\s+(?:failed|error)|\bFAILED\b|\bTraceback\b"),
        reason="says the tests passed, but the recorded test output reports failures",
        confidence=0.9,
    ),
    Claim(
        name="command-succeeded",
        says=re.compile(
            r"\b(?:the\s+)?command\s+(?:ran\s+)?(?:succeeded|ran\s+successfully|exited\s+cleanly)\b"
            r"|\bexited\s+with\s+(?:status\s+)?0\b",
            re.IGNORECASE,
        ),
        refuted_by=re.compile(
            r"\bexit(?:\s+code)?\s*[:=]?\s*[1-9]\d*\b|\bTraceback\b|\bcommand not found\b"
        ),
        reason="says the command succeeded, but the recorded result is a failure",
        confidence=0.85,
    ),
    Claim(
        name="count",
        says=re.compile(
            r"\b(?:returned|found|there\s+(?:are|were))\s+(\d+)\s+"
            r"(?:results?|matches?|files?|occurrences?)\b",
            re.IGNORECASE,
        ),
        refuted_by=re.compile(
            r"\b(\d+)\s+(?:results?|matches?|files?|occurrences?)\b", re.IGNORECASE
        ),
        reason="states a count the recorded tool output does not contain",
        confidence=0.85,
        compare=True,
    ),
)


@dataclass(slots=True)
class Contradiction:
    """The model asserts a tool returned X when the recorded result was Y.

    Off by default: a paraphrase is not a contradiction, and the recorded
    output of a *different* call can easily supply an innocent refutation.  The
    results are supplied as a callable so the check reads the live session
    rather than a snapshot taken when the auditor was built.
    """

    results: Callable[[], Sequence[str]] | Sequence[str]
    name: str = CONTRADICTION
    threshold: float = CONTRADICTION_THRESHOLD
    enabled: bool = False
    visible_only: bool = True
    claims: Sequence[Claim] = CLAIMS
    tail: int = TAIL

    def __call__(self, text: str) -> Verdict | None:
        recorded = self._recorded()
        if not recorded:
            return None
        haystack = _tail(text, self.tail)

        best: Verdict | None = None
        for claim in self.claims:
            said = self._settled_claim(claim, haystack)
            if said is None:
                continue
            for output in recorded:
                refutation = claim.refutation(said, output)
                if refutation is None:
                    continue
                if best is not None and claim.confidence <= best.confidence:
                    continue
                best = Verdict(
                    check=self.name,
                    confidence=claim.confidence,
                    reason=claim.reason,
                    evidence=f"said {' '.join(said.group(0).split())!r}; {refutation}",
                )
                break
        return best

    def _recorded(self) -> list[str]:
        source = self.results
        return [str(item) for item in (source() if callable(source) else source)]

    def _settled_claim(self, claim: Claim, text: str) -> re.Match[str] | None:
        """The last complete occurrence of the claim, ignoring a partial tail."""
        found = None
        for match in claim.says.finditer(text):
            if _settled(match, text):
                found = match
        return found


# -- the expensive check ----------------------------------------------------

#: Given the generation so far, condemn it or say nothing.  A callable so the
#: auditor needs no provider, no credentials and no network to be tested.
Judge = Callable[[str], "Verdict | None"]


@dataclass(slots=True)
class Sampled:
    """A second model, sampled and off the hot path.

    Three properties make this safe to have in a stream.  It is *debounced* by
    tokens emitted, so a long answer costs a handful of requests rather than
    one per chunk.  It runs on a *worker*, so a judge that takes ten seconds
    delays nothing - `offer` starts it and returns, and the verdict is picked
    up by whichever chunk boundary happens next.  It is *at most one in
    flight*, so a slow judge cannot pile up threads behind a fast stream.

    Failure is silence by construction: a judge that raises is recorded in
    `error` and produces no verdict, because an unreachable auditor must never
    be able to end a generation.
    """

    judge: Judge
    name: str = MODEL
    threshold: float = 0.8
    enabled: bool = True
    every: int = SAMPLE_EVERY
    warmup: int = SAMPLE_WARMUP
    samples: int = 0
    error: str = ""
    thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _running: bool = False
    _verdict: Verdict | None = None
    _at: int = 0

    def due(self, tokens: int) -> bool:
        if self.samples == 0:
            return tokens >= self.warmup
        return tokens - self._at >= self.every

    def offer(self, text: str, tokens: int) -> None:
        """Maybe start a judgement.  Never blocks, never raises."""
        if not self.enabled or not text:
            return
        with self._lock:
            if self._running or self._verdict is not None or not self.due(tokens):
                return
            self._running = True
            self._at = tokens
            self.samples += 1
        self.thread = threading.Thread(
            target=self._work, args=(text,), name="audit-judge", daemon=True
        )
        self.thread.start()

    def take(self) -> Verdict | None:
        """The verdict that has arrived since last asked, if any."""
        with self._lock:
            verdict, self._verdict = self._verdict, None
        return verdict

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running

    def _work(self, text: str) -> None:
        verdict: Verdict | None = None
        try:
            verdict = self.judge(text)
        except Exception as exc:
            # Swallowed on purpose: the auditor's failure mode is silence, so a
            # judge that is down must leave the generation exactly as it was.
            self.error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._verdict = verdict
            self._running = False


# -- the auditor ------------------------------------------------------------


@dataclass(slots=True)
class Auditor:
    """Passes an event stream through, and can stop it.

    Order is the contract: every upstream event is yielded *before* anything is
    inspected, so the caller sees the same objects in the same order as the
    provider produced them, and a halt never truncates the chunk that triggered
    it.  Inspection happens after the yield, and a decision therefore always
    lands on a chunk boundary.
    """

    checks: list[Check] = field(default_factory=list)
    sampler: Sampled | None = None
    max_output: int = DEFAULT_MAX_OUTPUT
    watch_thinking: bool = True
    text: str = ""
    thinking: str = ""
    chars: int = 0
    reported: int = 0
    halted: Halted | None = None
    verdicts: list[Verdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _seen: set[tuple[str, str]] = field(default_factory=set, repr=False)
    _in_tool_call: bool = False

    # -- accounting -------------------------------------------------------

    @property
    def tokens_seen(self) -> int:
        """The provider's own count when it has given one, else an estimate."""
        return max(self.reported, tokens_for(self.chars))

    @property
    def tokens_saved(self) -> int:
        """An upper bound on the saving: whatever the budget still allowed."""
        if self.halted is None:
            return 0
        return max(0, self.max_output - self.tokens_seen)

    def report(self) -> Report:
        return Report(
            tokens_seen=self.tokens_seen,
            tokens_saved=self.tokens_saved,
            halted=self.halted,
            verdicts=tuple(self.verdicts),
            samples=self.sampler.samples if self.sampler else 0,
            errors=tuple(self.errors),
        )

    # -- the stream -------------------------------------------------------

    def watch(self, events: Iterator[Event]) -> Iterator[Event]:
        for event in events:
            yield event
            verdict = self._inspect(event)
            if verdict is None:
                continue
            self.halted = Halted(
                check=verdict.check,
                reason=verdict.reason,
                evidence=verdict.evidence,
                confidence=verdict.confidence,
                tokens_seen=self.tokens_seen,
                mid_tool_call=self._in_tool_call,
            )
            # Filled after the event exists so it can read `self.halted`.
            self.halted.tokens_saved = self.tokens_saved
            # The `Stop` first: `TurnBuilder` does not know `Halted`, so
            # without it the halted turn is persisted as a clean finish.
            yield Stop(HALT_REASON)
            yield self.halted
            return

    def __call__(self, events: Iterator[Event]) -> Iterator[Event]:
        return self.watch(events)

    # -- inspection -------------------------------------------------------

    def _inspect(self, event: Event) -> Verdict | None:
        """A condemning verdict, or None.  Never raises."""
        condemned = self._pending()
        if condemned is not None:
            return condemned

        if isinstance(event, Usage):
            self.reported = max(self.reported, event.output)
            return None
        if isinstance(event, ToolCallDelta):
            # Remembered so a halt can warn that the arguments are half a JSON
            # document and must not be dispatched.
            self._in_tool_call = True
            return None

        visible = isinstance(event, TextDelta)
        if visible:
            self.text += event.text
            buffer = self.text
        elif isinstance(event, ThinkingDelta):
            self.thinking += event.text
            buffer = self.thinking
        else:
            return None
        self.chars += len(event.text)
        self._in_tool_call = False

        if isinstance(event, ThinkingDelta) and not self.watch_thinking:
            buffer = ""

        condemned = self._cheap(buffer, visible) if buffer else None
        if condemned is not None:
            return condemned

        if self.sampler is not None:
            self.sampler.offer(self.text, self.tokens_seen)
        return None

    def _pending(self) -> Verdict | None:
        """A verdict the second model left while the stream kept moving."""
        if self.sampler is None:
            return None
        verdict = self.sampler.take()
        if verdict is None:
            return None
        self._note(verdict)
        return verdict if verdict.confidence >= self.sampler.threshold else None

    def _cheap(self, buffer: str, visible: bool) -> Verdict | None:
        for check in self.checks:
            if not check.enabled or (check.visible_only and not visible):
                continue
            try:
                verdict = check(buffer)
            except Exception as exc:
                # A broken check must not be able to end a turn; record it so
                # the silence is explicable and carry on with the others.
                self.errors.append(f"{check.name}: {type(exc).__name__}: {exc}")
                continue
            if verdict is None:
                continue
            self._note(verdict)
            if verdict.confidence >= check.threshold:
                return verdict
        return None

    def _note(self, verdict: Verdict) -> None:
        """Record a verdict once.  A sub-threshold check re-fires every chunk."""
        key = (verdict.check, verdict.evidence)
        if key in self._seen:
            return
        self._seen.add(key)
        self.verdicts.append(verdict)


def default_checks(
    *,
    exists: Callable[[str], bool] | None = None,
    results: Callable[[], Sequence[str]] | Sequence[str] | None = None,
    enable: Sequence[str] = (),
    disable: Sequence[str] = (),
) -> list[Check]:
    """The cheap checks, with only `DEFAULT_ON` switched on.

    A check whose dependency is missing is not built at all: a fabricated-path
    detector without a workspace to consult would have to guess.  Disabled
    checks are still constructed so a UI can list what is available.
    """
    wanted = (set(DEFAULT_ON) | {name.lower() for name in enable}) - {
        name.lower() for name in disable
    }
    checks: list[Check] = [Repetition(enabled=REPETITION in wanted)]
    if exists is not None:
        checks.append(FabricatedPath(exists=exists, enabled=FABRICATED_PATH in wanted))
    if results is not None:
        checks.append(Contradiction(results=results, enabled=CONTRADICTION in wanted))
    return checks


def audit(
    events: Iterator[Event],
    *,
    exists: Callable[[str], bool] | None = None,
    results: Callable[[], Sequence[str]] | Sequence[str] | None = None,
    judge: Judge | None = None,
    max_output: int = DEFAULT_MAX_OUTPUT,
    enable: Sequence[str] = (),
    disable: Sequence[str] = (),
    watch_thinking: bool = True,
) -> Iterator[Event]:
    """Wrap a provider stream in an auditor.  The interceptor callers want.

    Build an `Auditor` directly when the report is wanted afterwards; this is
    the form for a caller that only needs the stream and the halt.
    """
    return Auditor(
        checks=default_checks(exists=exists, results=results, enable=enable, disable=disable),
        sampler=Sampled(judge) if judge is not None else None,
        max_output=max_output,
        watch_thinking=watch_thinking,
    ).watch(events)
