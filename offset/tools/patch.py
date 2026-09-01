"""Hash-anchored patching: an edit that lands exactly where it was aimed, or not at all.

The failure this module exists to prevent has two shapes, and both of them are
silent, which is what makes them expensive.

The first is the line-number patch.  An agent reads a file, decides to rewrite
lines 40 to 50, and by the time the call arrives the file has moved: a formatter
ran, a sibling agent landed an edit, or the agent's own earlier hunk shifted
everything down by three.  Lines 40 to 50 are now somebody else's code, and the
patch overwrites it without complaint.  The diff looks plausible.  The bug
surfaces much later, somewhere else.

The second is the context-match patch.  Instead of a line number it carries a
few lines of surrounding text and applies wherever they match.  In real code
that text is very often not unique - two identical `except OSError: return None`
blocks, two identical getters, a repeated table row - and the patch lands on
whichever one the search happened to reach first.

So a patch here identifies its target by the SHA-256 of the region's exact bytes
*and* by the digests of the text on either side of it.  At apply time both are
recomputed.  A mismatch is a refusal carrying the reason, never a best-effort
application: the region is either byte-identical to what the agent read, in
which case the edit means what the agent thought it meant, or it is not, in
which case the only honest answer is to re-read.  Ambiguity is a refusal too -
if two places in the file match the region *and* its context, no amount of
cleverness can tell which one was intended.

Two deliberate compromises make that strictness usable.

The whole-file digest is advisory.  A patch carries `file_hash`, and a mismatch
is reported as a note rather than a refusal, because the overwhelmingly common
case is a file that changed three hundred lines away from the region being
edited.  Refusing there would train the caller to re-read constantly and would
buy nothing: the anchored region is what the edit depends on, and that is
verified exactly.

`normalise` is opt-in.  Anchors are byte-exact by default because indentation is
meaning: in Python a re-indented line is a different program, and a "whitespace
only" difference is exactly what a half-applied earlier edit looks like.  A
caller who knows a formatter has been through the file can ask for whitespace
tolerance explicitly and accepts, by asking, that leading and trailing spaces
stop being part of the region's identity.

`apply` is pure - text in, text out, no filesystem - so every one of these rules
is testable without a temporary directory.  The `Tool` on top is a thin shell:
read, apply, and either write through a temporary file and `os.replace` or hand
back the refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: Hex characters kept from each SHA-256.  All 64 would be correct and useless:
#: every digest crosses a model's context at least twice, and 64 bits over the
#: handful of windows a single file offers makes an accidental collision far less
#: likely than the model mistyping the digest it was given.
DIGEST: Final = 16

#: Characters of the region quoted verbatim inside the anchor.  This is only a
#: prefilter - `str.find` runs in C, so it turns "hash every offset in the file"
#: into "hash the two or three places the region could possibly be".  The digest
#: still decides; the probe only proposes.
PROBE: Final = 64

#: Characters of context hashed on each side when minting a strict anchor.  Kept
#: small on purpose.  Context exists to tell two identical blocks apart, and a
#: wide window would also refuse a patch because something harmless moved a few
#: functions away - which is the case `file_hash` is deliberately lenient about.
CONTEXT: Final = 160

#: Ceiling for the widened context.  `anchor_for` widens only when the default
#: window leaves the anchor ambiguous, and stops here: past this point widening
#: trades away the "changed elsewhere still applies" property in pursuit of a
#: uniqueness it may never reach, so it is better to let `apply` refuse and say
#: the anchor is ambiguous.
MAX_CONTEXT: Final = 4_000

#: Context *lines* each side when normalising.  Whitespace tolerance is a
#: line-level idea - reindentation moves whole lines - so a normalised anchor
#: counts context in lines rather than characters, and four is roughly the same
#: amount of evidence as `CONTEXT` characters of code.
CONTEXT_LINES: Final = 4

#: Ceiling for the widened normalised context, for the same reason as
#: `MAX_CONTEXT`.
MAX_CONTEXT_LINES: Final = 60

#: Largest file the tool will read.  A patch aimed at something bigger than this
#: is almost always aimed at generated output or a data blob, where anchoring by
#: hash buys nothing and the linear scans stop being free.
MAX_BYTES: Final = 2_000_000

REPLACE: Final = "replace"
INSERT_BEFORE: Final = "insert_before"
INSERT_AFTER: Final = "insert_after"
DELETE: Final = "delete"

#: The complete set.  Anything else is a caller error reported as such, rather
#: than an action quietly treated as a replace.
ACTIONS: Final = (REPLACE, INSERT_BEFORE, INSERT_AFTER, DELETE)


def digest(text: str) -> str:
    """The anchor's notion of identity: SHA-256 over the exact UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DIGEST]


def _deindent(text: str) -> str:
    """`text` with each line's leading and trailing whitespace removed.

    This is the whole of what `normalise` tolerates, and the narrowness is the
    point.  A formatter that reindents a block, or an editor that trims trailing
    spaces, leaves every line's content untouched; anything that changes the
    content of a line still changes the digest and is still refused.
    """
    return "\n".join(line.strip() for line in text.split("\n"))


def _widths(start: int, ceiling: int) -> list[int]:
    """Doubling context widths, `start` first, `ceiling` last."""
    out = [start]
    while out[-1] < ceiling:
        out.append(min(out[-1] * 2, ceiling))
    return out


# -- the anchor -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anchor:
    """A region of a file, identified by content rather than by position.

    `size`, `before_size` and `after_size` are counted in characters for a
    strict anchor and in lines for a normalised one.  The two units are not
    interchangeable and the mode decides which applies: a character window would
    slice through the very indentation a normalised anchor is meant to ignore,
    so normalising forces whole lines.

    `line` is the 1-based line the region started on when the anchor was minted.
    It is advisory - used in messages and never to choose between candidates,
    because "the nearest match to where it used to be" is precisely the guess
    that corrupts files.
    """

    #: Digest of the region itself; `_deindent`ed first when `normalise`.
    region: str
    size: int
    #: Digest of the context immediately before the region, and its extent.
    before: str = ""
    before_size: int = 0
    #: Digest of the context immediately after the region, and its extent.
    after: str = ""
    after_size: int = 0
    line: int = 0
    #: Verbatim head of the region.  A prefilter and a message, not evidence.
    probe: str = ""
    #: Lines the region spans.  The match window when normalising.
    lines: int = 1
    normalise: bool = False

    def describe(self) -> str:
        unit = "line" if self.normalise else "char"
        mode = ", whitespace-tolerant" if self.normalise else ""
        return f"{self.region} ({self.lines} line(s), {self.size} {unit}(s) from line {self.line}{mode})"

    def to_json(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "size": self.size,
            "before": self.before,
            "before_size": self.before_size,
            "after": self.after,
            "after_size": self.after_size,
            "line": self.line,
            "probe": self.probe,
            "lines": self.lines,
            "normalise": self.normalise,
        }

    @classmethod
    def from_json(cls, data: Any) -> "Anchor":
        if not isinstance(data, dict):
            raise ValueError("an anchor must be an object")
        region = data.get("region")
        if not isinstance(region, str) or not region:
            raise ValueError("an anchor needs a 'region' digest")
        size = data.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("an anchor needs a non-negative integer 'size'")
        return cls(
            region=region,
            size=size,
            before=str(data.get("before", "")),
            before_size=int(data.get("before_size", 0) or 0),
            after=str(data.get("after", "")),
            after_size=int(data.get("after_size", 0) or 0),
            line=int(data.get("line", 0) or 0),
            probe=str(data.get("probe", "")),
            lines=int(data.get("lines", 1) or 1),
            normalise=bool(data.get("normalise", False)),
        )


@dataclass(frozen=True, slots=True)
class Site:
    """Where an anchor was found in the text as it stands now."""

    start: int
    end: int
    line: int


def _line_offsets(lines: list[str]) -> list[int]:
    """Character offset of the start of each line, given `text.split("\\n")`."""
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)  # the separator counts
    return offsets


def anchor_for(text: str, start: int, end: int, *, normalise: bool = False) -> Anchor:
    """Mint an anchor for `text[start:end]`, from the text as it was just read.

    The context window starts at `CONTEXT` (or `CONTEXT_LINES`) and doubles only
    while the anchor still matches more than one place in `text`, so a normal
    region carries the smallest amount of context that identifies it and a
    duplicated one carries as much as the ceiling allows.  If even that leaves it
    ambiguous the anchor is returned anyway: `apply` will refuse it and say so,
    which is more useful than an exception at mint time when the caller may not
    be patching this region at all.

    A normalised anchor is snapped outwards to whole lines, because whitespace
    tolerance is defined per line.
    """
    if not isinstance(text, str):
        raise TypeError("anchor_for works on text")
    if not 0 <= start < end <= len(text):
        # An empty region has no identity: every offset in the file matches it
        # equally well, so an insert has to anchor on the line it goes next to.
        raise ValueError(f"anchor range {start}:{end} is not a non-empty region of {len(text)} characters")

    if normalise:
        return _line_anchor(text, start, end)

    region = text[start:end]
    line = text.count("\n", 0, start) + 1
    minted = None
    for width in _widths(CONTEXT, MAX_CONTEXT):
        before = text[max(0, start - width) : start]
        after = text[end : end + width]
        minted = Anchor(
            region=digest(region),
            size=end - start,
            before=digest(before),
            before_size=len(before),
            after=digest(after),
            after_size=len(after),
            line=line,
            probe=region[:PROBE],
            lines=region.count("\n") + 1,
            normalise=False,
        )
        if len(locate(text, minted)) == 1:
            return minted
        if len(before) < width and len(after) < width:
            break  # both sides are exhausted; widening cannot add evidence
    assert minted is not None
    return minted


def _line_anchor(text: str, start: int, end: int) -> Anchor:
    """`anchor_for` for the normalising case, working in whole lines."""
    lines = text.split("\n")
    offsets = _line_offsets(lines)
    first = text.count("\n", 0, start)
    last = text.count("\n", 0, max(start, end - 1))
    span = lines[first : last + 1]
    normalised = [line.strip() for line in lines]
    minted = None
    for width in _widths(CONTEXT_LINES, MAX_CONTEXT_LINES):
        before = normalised[max(0, first - width) : first]
        after = normalised[last + 1 : last + 1 + width]
        minted = Anchor(
            region=digest(_deindent("\n".join(span))),
            size=offsets[last] + len(lines[last]) - offsets[first],
            before=digest("\n".join(before)),
            before_size=len(before),
            after=digest("\n".join(after)),
            after_size=len(after),
            line=first + 1,
            probe=span[0].strip()[:PROBE],
            lines=len(span),
            normalise=True,
        )
        if len(locate(text, minted)) == 1:
            return minted
        if len(before) < width and len(after) < width:
            break
    assert minted is not None
    return minted


def anchor_from_text(
    region: str,
    *,
    before: str = "",
    after: str = "",
    normalise: bool = False,
    line: int = 0,
) -> Anchor:
    """Mint an anchor from text a caller quotes, rather than from offsets.

    This is the shape a model can actually produce: it echoes the region it read
    and, when the region is not unique, a little of what surrounds it.  The
    digests are computed here from that quoted text, so the guarantee is the
    same one - the file must still contain exactly this - without the caller
    having to compute SHA-256 in its head.

    When normalising, blank lines at the outer edges of `before` and `after` are
    dropped: a quoted context block almost always ends with the newline that
    ended the quote, and treating that as "the file must have a blank line here"
    would refuse every patch for a reason the caller never intended.
    """
    if not region:
        raise ValueError("a hunk needs a non-empty region to anchor on")
    if normalise:
        span = region.split("\n")
        lead = _trim_blank(before.split("\n")) if before else []
        trail = _trim_blank(after.split("\n")) if after else []
        return Anchor(
            region=digest(_deindent(region)),
            size=len(region),
            before=digest(_deindent("\n".join(lead))),
            before_size=len(lead),
            after=digest(_deindent("\n".join(trail))),
            after_size=len(trail),
            line=line,
            probe=span[0].strip()[:PROBE],
            lines=len(span),
            normalise=True,
        )
    return Anchor(
        region=digest(region),
        size=len(region),
        before=digest(before),
        before_size=len(before),
        after=digest(after),
        after_size=len(after),
        line=line,
        probe=region[:PROBE],
        lines=region.count("\n") + 1,
        normalise=False,
    )


def _trim_blank(lines: list[str]) -> list[str]:
    """`lines` without whitespace-only entries at either end."""
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


# -- locating ---------------------------------------------------------------


def locate(text: str, anchor: Anchor, *, context: bool = True) -> list[Site]:
    """Every place in `text` the anchor matches, in file order.

    With `context=False` only the region's own digest has to match, which is how
    a refusal works out whether the region moved or the region changed.
    """
    if anchor.normalise:
        return _locate_lines(text, anchor, context)
    return _locate_exact(text, anchor, context)


def _locate_exact(text: str, anchor: Anchor, context: bool) -> list[Site]:
    found: list[Site] = []
    size = anchor.size
    if size <= 0 or size > len(text):
        return found
    probe = anchor.probe
    at = text.find(probe)
    while at != -1:
        end = at + size
        if end <= len(text) and digest(text[at:end]) == anchor.region:
            if not context or _context_matches(text, at, end, anchor):
                found.append(Site(at, end, text.count("\n", 0, at) + 1))
        at = text.find(probe, at + 1)
    return found


def _context_matches(text: str, start: int, end: int, anchor: Anchor) -> bool:
    before = text[max(0, start - anchor.before_size) : start]
    after = text[end : end + anchor.after_size]
    return digest(before) == anchor.before and digest(after) == anchor.after


def _locate_lines(text: str, anchor: Anchor, context: bool) -> list[Site]:
    lines = text.split("\n")
    normalised = [line.strip() for line in lines]
    offsets = _line_offsets(lines)
    window = max(1, anchor.lines)
    found: list[Site] = []
    for i in range(0, len(lines) - window + 1):
        if not normalised[i].startswith(anchor.probe):
            continue
        if digest("\n".join(normalised[i : i + window])) != anchor.region:
            continue
        if context:
            before = normalised[max(0, i - anchor.before_size) : i]
            after = normalised[i + window : i + window + anchor.after_size]
            if digest("\n".join(before)) != anchor.before:
                continue
            if digest("\n".join(after)) != anchor.after:
                continue
        last = i + window - 1
        found.append(Site(offsets[i], offsets[last] + len(lines[last]), i + 1))
    return found


# -- the patch --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hunk:
    """One anchored edit.  `text` is the replacement, or what to insert."""

    action: str
    anchor: Anchor
    text: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action, "anchor": self.anchor.to_json(), "text": self.text}

    @classmethod
    def from_json(cls, data: Any) -> "Hunk":
        if not isinstance(data, dict):
            raise ValueError("a hunk must be an object")
        action = str(data.get("action") or REPLACE)
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
        raw = data.get("anchor")
        if isinstance(raw, dict):
            anchor = Anchor.from_json(raw)
        else:
            # The quoted-text form: the caller echoes what it read and we hash
            # it here.  Same guarantee, no digest arithmetic at the caller.
            region = data.get("region")
            if not isinstance(region, str) or not region:
                raise ValueError("a hunk needs either an 'anchor' object or a 'region' of quoted text")
            anchor = anchor_from_text(
                region,
                before=str(data.get("before", "") or ""),
                after=str(data.get("after", "") or ""),
                normalise=bool(data.get("normalise", False)),
                line=int(data.get("line", 0) or 0),
            )
        body = data.get("text", "")
        if action == DELETE and body:
            raise ValueError("a delete hunk must not carry replacement text")
        return cls(action=action, anchor=anchor, text=str(body or ""))


@dataclass(frozen=True, slots=True)
class Patch:
    """A set of hunks that apply together or not at all."""

    hunks: tuple[Hunk, ...]
    #: Digest of the whole file when it was read.  Advisory: see the module
    #: docstring for why a mismatch here is a note rather than a refusal.
    file_hash: str = ""
    path: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "file_hash": self.file_hash, "hunks": [h.to_json() for h in self.hunks]}

    @classmethod
    def from_json(cls, data: Any) -> "Patch":
        if not isinstance(data, dict):
            raise ValueError("a patch must be an object")
        raw = data.get("hunks")
        if not isinstance(raw, list) or not raw:
            raise ValueError("a patch needs a non-empty 'hunks' array")
        return cls(
            hunks=tuple(Hunk.from_json(item) for item in raw),
            file_hash=str(data.get("file_hash", "") or ""),
            path=str(data.get("path", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class Result:
    """The outcome of `apply`: new text, or the reason there is none."""

    ok: bool
    text: str = ""
    reason: str = ""
    #: Things worth telling the caller about an edit that did apply.
    notes: tuple[str, ...] = ()
    applied: int = 0

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def refused(cls, reason: str, text: str = "", notes: tuple[str, ...] = ()) -> "Result":
        """A refusal, carrying the file *unchanged*.

        `text` is always "what this file should contain", refusal or not, so a
        caller that writes it back without checking `ok` writes the original
        rather than truncating the file to nothing.  Leaving it empty made the
        careless path catastrophic instead of merely wrong.
        """
        return cls(ok=False, text=text, reason=reason, notes=notes)

    @classmethod
    def done(cls, text: str, applied: int, notes: tuple[str, ...] = ()) -> "Result":
        return cls(ok=True, text=text, applied=applied, notes=notes)


@dataclass(frozen=True, slots=True)
class _Edit:
    """A resolved hunk: the span to overwrite and what to put there."""

    start: int
    end: int
    text: str
    index: int
    action: str


def apply(text: str, patch: Patch) -> Result:
    """Apply every hunk, or none of them.  Pure: no filesystem, no clock.

    Each anchor is resolved against `text` as given, so hunks never see each
    other's changes and their order in the patch does not affect where they
    land.  The splices happen back to front afterwards, which keeps every
    resolved offset valid without recomputing anything.
    """
    notes: list[str] = []
    if patch.file_hash:
        current = digest(text)
        if current != patch.file_hash:
            notes.append(
                f"the file changed since it was read ({patch.file_hash} -> {current}); "
                "the anchored regions are byte-identical, so this patch still means what it said"
            )
    if not patch.hunks:
        return Result.refused("the patch has no hunks", text)

    edits: list[_Edit] = []
    problems: list[str] = []
    for index, hunk in enumerate(patch.hunks, 1):
        if hunk.action not in ACTIONS:
            problems.append(f"hunk {index}: unknown action {hunk.action!r}; use one of {', '.join(ACTIONS)}")
            continue
        sites = locate(text, hunk.anchor)
        if len(sites) != 1:
            problems.append(_refusal(text, index, hunk, sites))
            continue
        start, end = _span(text, sites[0], hunk.action)
        edits.append(_Edit(start, end, "" if hunk.action == DELETE else hunk.text, index, hunk.action))

    if problems:
        # All or nothing: one unresolvable anchor means the caller's picture of
        # the file is wrong, and the hunks that did resolve were written against
        # that same wrong picture.
        head = f"refused, {len(problems)} of {len(patch.hunks)} hunk(s) could not be anchored:"
        return Result.refused(head + "\n" + "\n".join(problems), text, tuple(notes))

    ordered = sorted(edits, key=lambda e: (e.start, e.end, e.index))
    for left, right in zip(ordered, ordered[1:]):
        if left.end > right.start:
            return Result.refused(
                f"refused: hunks {left.index} and {right.index} touch the same region "
                f"(characters {left.start}-{left.end} and {right.start}-{right.end}); "
                "split them into separate patches or anchor one region covering both",
                text,
                tuple(notes),
            )

    out = text
    for edit in reversed(ordered):  # back to front, so earlier offsets stay true
        out = out[: edit.start] + edit.text + out[edit.end :]
    if out == text:
        notes.append("the patch applied cleanly but changed nothing")
    return Result.done(out, len(edits), tuple(notes))


def _span(text: str, site: Site, action: str) -> tuple[int, int]:
    """The characters an action overwrites at `site`.

    Whole-line regions get their trailing newline handled here rather than by
    the caller: deleting lines 3 and 4 should leave no blank line behind, and
    text inserted after a whole-line region belongs on the line after it, not
    glued to the end of the last one.
    """
    whole_line = (site.start == 0 or text[site.start - 1] == "\n") and site.end < len(text) and text[site.end] == "\n"
    if action == REPLACE:
        return site.start, site.end
    if action == DELETE:
        return site.start, site.end + 1 if whole_line else site.end
    if action == INSERT_BEFORE:
        return site.start, site.start
    at = site.end + 1 if site.end < len(text) and text[site.end] == "\n" else site.end
    return at, at


def _refusal(text: str, index: int, hunk: Hunk, sites: list[Site]) -> str:
    """Why a hunk could not be anchored, in terms of what actually changed."""
    anchor = hunk.anchor
    head = f"  hunk {index} ({hunk.action}, anchored at line {anchor.line}): "
    if len(sites) > 1:
        where = ", ".join(str(s.line) for s in sites[:6])
        return (
            head + f"ambiguous - {len(sites)} regions match the anchor and its context (lines {where}). "
            "Anchor a larger region, or one whose surroundings differ."
        )

    loose = locate(text, anchor, context=False)
    if loose:
        where = ", ".join(str(s.line) for s in loose[:6])
        return (
            head + f"the region is still in the file (line {where}) but the text around it changed, "
            f"so which one you meant is no longer decidable. Re-read the file and anchor again."
        )

    detail = _what_changed(text, anchor)
    return head + f"the anchored region changed. {detail} Re-read the file and anchor again."


def _what_changed(text: str, anchor: Anchor) -> str:
    """The most specific thing we can say about a region that no longer matches."""
    probe = anchor.probe
    lines = text.split("\n")
    at_line = lines[anchor.line - 1] if 0 < anchor.line <= len(lines) else None

    if anchor.normalise:
        seen = [i + 1 for i, line in enumerate(lines) if probe and line.strip().startswith(probe)]
    else:
        seen = []
        found = text.find(probe) if probe else -1
        while found != -1 and len(seen) < 6:
            seen.append(text.count("\n", 0, found) + 1)
            found = text.find(probe, found + 1)

    if not seen:
        return f"Its first line ({probe.strip()!r}) is not in the file any more."
    if anchor.size <= len(text):
        # The region starts where it should but the bytes after that differ:
        # naming the digest we wanted against the digest that is there makes the
        # mismatch checkable rather than a matter of opinion.
        offsets = [o for o in _probe_offsets(text, probe)]
        actual = digest(text[offsets[0] : offsets[0] + anchor.size]) if offsets else "?"
        return (
            f"Its first line is still at line {seen[0]}, but the {anchor.size} characters from there "
            f"hash to {actual}, not {anchor.region}."
            + (f" Line {anchor.line} now reads {at_line.strip()!r}." if at_line is not None else "")
        )
    return f"The file is shorter than the region ({len(text)} characters against {anchor.size})."


def _probe_offsets(text: str, probe: str) -> list[int]:
    if not probe:
        return []
    out: list[int] = []
    at = text.find(probe)
    while at != -1 and len(out) < 6:
        out.append(at)
        at = text.find(probe, at + 1)
    return out


# -- the tool ---------------------------------------------------------------


def _atomic_write(target: Path, data: bytes) -> None:
    """Write through a sibling temporary file, preserving the existing mode.

    A patch that dies mid-write must leave the previous file, not half of the
    new one; and a rewritten script that lost its executable bit is a bug
    report, while `mkstemp` creates `0o600`.
    """
    mode = target.stat().st_mode & 0o7777 if target.exists() else None
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".patch-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())  # the rename is only atomic if the data is down
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class PatchFile(Tool):
    """Anchor a region by hash, then patch it or refuse."""

    name = "patch"
    description = (
        "Edit a file by content-hash anchor instead of by line number, so an edit can never "
        "land on the wrong region. Call it with op='anchor' and a line range to get an anchor "
        "for what you read, then op='apply' with hunks to change it. A hunk is "
        "{action, anchor|region, text}: action is replace, insert_before, insert_after or delete; "
        "pass the anchor object from op='anchor', or quote the exact region text as 'region' "
        "(plus 'before'/'after' context if the region is not unique). All hunks apply together "
        "or none do. If the file moved under you the patch is refused and tells you what changed - "
        "re-read and anchor again rather than retrying. Set normalise=true only when a formatter "
        "may have reindented the region; anchors are byte-exact by default."
    )
    danger = Danger.WRITE
    #: Two concurrent patches to the same file would each write a file the other
    #: never saw, and the last `os.replace` would win silently.
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["anchor", "apply"], "description": "anchor a region, or apply hunks"},
            "path": {"type": "string", "description": "file to anchor in or patch"},
            "start_line": {"type": "integer", "minimum": 1, "description": "op=anchor: first line of the region"},
            "end_line": {"type": "integer", "minimum": 1, "description": "op=anchor: last line, inclusive"},
            "normalise": {"type": "boolean", "description": "tolerate whitespace-only reindentation"},
            "file_hash": {"type": "string", "description": "op=apply: the file_hash you were given"},
            "hunks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "op=apply: [{action, anchor|region, text}]",
            },
        },
        "required": ["path"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        if str(args.get("op", "")) == "anchor" or "hunks" not in args:
            return f"anchor {path}:{args.get('start_line', 1)}"
        hunks = args.get("hunks")
        count = len(hunks) if isinstance(hunks, list) else 0
        return f"patch {path} ({count} hunk(s))"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = ctx.resolve(str(args.get("path", "")))
        except PermissionError as exc:
            return ToolResult.fail(str(exc))
        if not target.exists():
            return ToolResult.fail(f"no such file: {args.get('path')}")
        if target.is_dir():
            return ToolResult.fail(f"{args.get('path')} is a directory")
        if target.stat().st_size > MAX_BYTES:
            return ToolResult.fail(f"{args.get('path')} is larger than {MAX_BYTES} bytes; patch is for source files")
        try:
            raw = target.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult.fail(f"{args.get('path')} is not UTF-8 text")

        op = str(args.get("op") or ("apply" if args.get("hunks") else "anchor"))
        if op == "anchor":
            return self._anchor(args, text)
        if op != "apply":
            return ToolResult.fail(f"unknown op {op!r}; use 'anchor' or 'apply'")
        return self._apply(args, target, raw, text)

    def _anchor(self, args: dict[str, Any], text: str) -> ToolResult:
        lines = text.split("\n")
        first = int(args.get("start_line", 1) or 1)
        last = int(args.get("end_line", first) or first)
        if first < 1 or first > len(lines):
            return ToolResult.fail(f"start_line {first} is outside the file's {len(lines)} line(s)")
        last = min(max(last, first), len(lines))
        offsets = _line_offsets(lines)
        start = offsets[first - 1]
        end = offsets[last - 1] + len(lines[last - 1])
        if start == end:
            # A blank line has no bytes to hash; widening to the next line keeps
            # the anchor meaningful instead of minting one that matches nothing.
            end = min(end + 1, len(text))
        if start >= end:
            return ToolResult.fail(f"lines {first}-{last} are empty; anchor a region with content")
        anchor = anchor_for(text, start, end, normalise=bool(args.get("normalise")))
        ambiguous = len(locate(text, anchor)) != 1
        payload = {"path": str(args.get("path", "")), "file_hash": digest(text), "anchor": anchor.to_json()}
        body = json.dumps(payload, ensure_ascii=False)
        warning = (
            "\nWARNING: this region and its surroundings are not unique in the file; "
            "a patch using this anchor will be refused. Anchor a wider range."
            if ambiguous
            else ""
        )
        return ToolResult.text(
            f"anchored lines {first}-{last} of {args.get('path')}\n{body}\nregion:\n{text[start:end]}{warning}",
            display=f"anchor {args.get('path')}:{first}-{last}",
            anchor=anchor.to_json(),
            file_hash=payload["file_hash"],
            ambiguous=ambiguous,
        )

    def _apply(self, args: dict[str, Any], target: Path, raw: bytes, text: str) -> ToolResult:
        try:
            patch = Patch.from_json(
                {
                    "path": str(args.get("path", "")),
                    "file_hash": str(args.get("file_hash", "") or ""),
                    "hunks": args.get("hunks"),
                }
            )
        except ValueError as exc:
            return ToolResult.fail(f"bad patch: {exc}")

        result = apply(text, patch)
        if not result.ok:
            return ToolResult.fail(result.reason + ("\n" + "\n".join(result.notes) if result.notes else ""))

        # Between the read above and the replace below the file could change
        # again.  `os.replace` cannot be made conditional portably, so the best
        # available guarantee is to check immediately before it and refuse: that
        # turns a lost concurrent edit into a retry rather than into silence.
        try:
            if target.read_bytes() != raw:
                return ToolResult.fail(
                    f"{args.get('path')} changed while the patch was being prepared; nothing written, re-read it"
                )
        except OSError as exc:
            return ToolResult.fail(f"could not re-check {args.get('path')}: {exc}")

        try:
            _atomic_write(target, result.text.encode("utf-8"))
        except OSError as exc:
            return ToolResult.fail(f"could not write {args.get('path')}: {exc}")

        delta = len(result.text.split("\n")) - len(text.split("\n"))
        summary = f"patched {args.get('path')}: {result.applied} hunk(s), {delta:+d} line(s)"
        return ToolResult.text(
            "\n".join([summary, *result.notes]),
            display=summary,
            hunks=result.applied,
            file_hash=digest(result.text),
        )


def patch_tools() -> list[Tool]:
    """The one tool.  Anchoring and applying share a file read, so they share a tool."""
    return [PatchFile()]
