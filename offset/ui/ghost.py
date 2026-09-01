"""Ghost text: the dim completion that runs ahead of the cursor.

The input box already had a completion menu, and a menu is the wrong shape for
the thing people actually want most of the time - to keep typing and press one
key when the machine guessed right. Ghost text is that: the remainder of a
likely line, drawn dim after the cursor, accepted with a keystroke and ignored
by continuing to type.

The engine lives here rather than in `shell/app.py` for two reasons. The first
is that prompt_toolkit's `AutoSuggest` protocol is synchronous and is called on
the keypress path, so anything slow in it is felt as a stutter in the terminal;
keeping the logic out of the binding makes the latency budget explicit and
testable. The second is that a suggester wired into a `Buffer` cannot be tested
without a TTY, and the interesting behaviour here is all in the edge cases.

Three decisions shape the file:

**A missed deadline is an absent suggestion, never a stalled prompt.** The
command and history sources are pure memory and answer immediately. Path
completion has to read a directory, which on a cold cache, a network mount or a
loaded machine can take arbitrarily long, so it runs on a worker thread and the
caller waits at most `DEADLINE`. Past the deadline the keystroke is served with
no suggestion and the scan is left running: it lands in the cache and shows up
on the next keystroke instead of being thrown away. A directory that has
already missed a deadline is remembered, so a slow filesystem costs one late
suggestion rather than a deadline on every subsequent character.

**Typing coalesces.** The worker has a one-slot mailbox, so eight characters
typed quickly queue at most one scan behind the one in flight, and that one is
for the text the user has now rather than for a prefix they have moved past.
The alternative - a queue - spends the whole burst servicing dead requests and
delivers the answer to the last one last.

**The suggestion is the remainder, not the line.** `Suggestion.text` is only
what would be appended, which is exactly what a renderer needs in order to draw
dim characters after the cursor without repainting the line the user typed, and
is what makes `accept` a concatenation rather than a diff.

A suggestion is offered only when the cursor sits at the very end of the
buffer. Text after the cursor - the rest of a word, or a following line in a
multi-line prompt - makes an appended completion something the user cannot
accept coherently, so the honest answer is silence.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

#: How long a keystroke will wait for a filesystem scan. 40ms is under the
#: threshold at which a person attributes a delay to their own typing, and well
#: under the ~100ms at which a terminal starts to feel unresponsive. It is a
#: ceiling, not a target: a warm cache answers in microseconds.
DEADLINE: Final = 0.04

#: How long a directory listing is trusted without asking the filesystem again.
#: Two seconds is long enough that typing a path costs one scan rather than one
#: per character, and short enough that a file created in another pane appears
#: while the user is still looking at the same prompt.
TTL: Final = 2.0

#: Previous prompts kept for the history source. Prefix-matching a few hundred
#: strings is nothing, and beyond this the oldest lines are so unlikely to be
#: re-typed that they are only a way to hold a session's text in memory.
HISTORY_LIMIT: Final = 500

#: Names read from one directory. A scan of a generated tree with a hundred
#: thousand entries would blow the deadline every time and suggest nothing
#: useful anyway; truncating keeps the cost bounded per directory.
MAX_ENTRIES: Final = 2000

#: Where `accept_word` stops. `/` is included so that accepting a word inside a
#: path lands on a directory boundary, which is how a shell behaves and is the
#: only useful place to stop inside `offset/ui/ghost.py`.
WORD_BREAKS: Final = frozenset(" \t\n/")

SOURCE_COMMAND: Final = "command"
SOURCE_PATH: Final = "path"
SOURCE_HISTORY: Final = "history"


@dataclass(frozen=True, slots=True)
class Suggestion:
    """What to append after the cursor, and which source proposed it.

    `text` is never empty: a source that has nothing to add returns no
    suggestion at all, so a renderer can treat a `Suggestion` as something
    worth drawing without checking it first.
    """

    text: str
    source: str


def accept(buffer: str, suggestion: Suggestion | None) -> str:
    """The buffer with the whole suggestion taken.

    Tolerates `None` so a key binding can call it unconditionally: the key that
    accepts a suggestion also has to do something sensible when there is none,
    and that something is to leave the line alone.
    """
    if suggestion is None or not suggestion.text:
        return buffer
    return buffer + suggestion.text


def accept_word(buffer: str, suggestion: Suggestion | None) -> str:
    """The buffer with one word of the suggestion taken.

    The point of a partial accept is to keep a long guess useful when only its
    first part is right - accepting `offset/` out of `offset/ui/ghost.py` and
    then typing something else. A trailing `/` is taken with the word because a
    directory without its separator is not a position anything can continue
    from; a trailing space is not, because the next word is exactly what the
    user is choosing to type themselves.
    """
    if suggestion is None or not suggestion.text:
        return buffer
    text = suggestion.text
    cut = 0
    while cut < len(text) and text[cut] in WORD_BREAKS:
        cut += 1
    while cut < len(text) and text[cut] not in WORD_BREAKS:
        cut += 1
    if cut < len(text) and text[cut] == "/":
        cut += 1
    return buffer + text[:cut]


# -- the command source -----------------------------------------------------


def _literal(word: str) -> bool:
    """Whether a usage fragment is a word to type rather than a placeholder."""
    return bool(word) and not any(char in word for char in "<>.")


def _usage_literals(usage: str, name: str) -> tuple[str, ...]:
    """The fixed words a command accepts, mined from its `usage` string.

    `Command` has no field for subcommands, and adding one would mean editing
    every declaration in `shell/commands.py` and every subsystem that appends
    to it. The usage strings already carry the information - `/approve
    safe|auto-edit|yolo` - so this reads them.

    A bracketed group is only treated as literal when it alternates:
    `[judge|vote|race|relay]` enumerates words, whereas `[query]` and `[name]`
    are placeholders written without angle brackets. The rule costs the one
    genuine literal in that shape (`/compact [now]`) and in exchange never
    suggests a placeholder as though it were a word, which is the failure that
    would make the whole source untrustworthy.
    """
    out: list[str] = []
    for token in usage.split():
        if token.startswith("/") or token == "|":
            continue  # the command echoing its own name, or an alternation bar
        bare = token.strip("[]")
        if bare.startswith("-"):
            out.append(bare)  # a flag is always literal, brackets or not
            continue
        if "|" in bare:
            out.extend(part for part in bare.split("|") if _literal(part))
            continue
        if token.startswith("[") or not _literal(bare):
            continue
        out.append(bare)
    # `dict.fromkeys` rather than a set: the declared order is the order a
    # person reading the usage string expects the suggestions to arrive in.
    return tuple(dict.fromkeys(out))


def command_table() -> dict[str, tuple[str, ...]]:
    """Every slash command name, aliases included, and its literal arguments.

    Imported inside the function deliberately. `offset.shell.commands` imports
    most of the application, so a module-level import of it from `offset.ui`
    becomes a cycle the moment the shell wires this engine into its input box -
    the same reason `core/tasks.py` builds its `COMMANDS` lazily.
    """
    try:
        from offset.shell.commands import COMMANDS
    except Exception:
        # A shell that will not import costs command completion and nothing
        # else; the path and history sources are independent of it. Raising
        # here would take the input box down with it.
        return {}
    table: dict[str, tuple[str, ...]] = {}
    for command in COMMANDS:
        subs = _usage_literals(command.usage, command.name)
        for name in (command.name, *command.aliases):
            table[name] = subs
    return table


# -- the path source --------------------------------------------------------


def read_dir(path: str) -> tuple[str, ...]:
    """The names in a directory, sorted, directories marked with a slash.

    The slash is part of the name on purpose: it makes a directory suggestion
    immediately continuable, and it is what `accept_word` stops on.
    """
    names: list[str] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = entry.name
                try:
                    if entry.is_dir():
                        name += "/"
                except OSError:
                    # A broken symlink or a race with a delete: the entry is
                    # still worth offering, just not as a directory.
                    pass
                names.append(name)
                if len(names) >= MAX_ENTRIES:
                    break
    except OSError:
        # A directory that does not exist, or that we may not read, simply has
        # no completions. Mid-keystroke is the wrong moment to report it.
        return ()
    return tuple(sorted(names))


@dataclass(slots=True)
class _Listing:
    """One directory's names, when they were read, and its mtime then."""

    names: tuple[str, ...]
    mtime: int
    stamp: float


@dataclass(slots=True)
class _Job:
    """A requested scan and the event that says it has settled.

    `names` stays `None` when the job was coalesced away, which is why the
    waiter checks the value rather than trusting the event.
    """

    key: str
    done: threading.Event = field(default_factory=threading.Event)
    names: tuple[str, ...] | None = None


class _Scanner:
    """One worker thread, a one-slot mailbox and a per-directory cache.

    The mailbox is what makes rapid typing cheap: a new request replaces the
    waiting one instead of joining a queue, so at most one scan is in flight
    and at most one is pending, and the pending one is always for the newest
    input.
    """

    def __init__(
        self,
        read: Callable[[str], tuple[str, ...]] = read_dir,
        *,
        ttl: float = TTL,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self._read = read
        self._ttl = ttl
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: _Job | None = None
        self._running: _Job | None = None
        self._cache: dict[str, _Listing] = {}
        self._thread: threading.Thread | None = None
        #: Directories that have already missed a deadline. Waiting again on a
        #: filesystem that has proved slow turns one late suggestion into a
        #: stutter on every character, which is the bug this set prevents.
        self._slow: set[str] = set()
        #: Real directory reads. Only the mtime-invalidation test looks at it,
        #: and only the worker thread writes it.
        self.reads = 0

    # -- the keypress side ---------------------------------------------------

    def cached(self, key: str) -> tuple[str, ...] | None:
        """A listing fresh enough to use, without touching the filesystem."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or time.monotonic() - entry.stamp > self._ttl:
                return None
            return entry.names

    def fetch(self, key: str, deadline: float) -> tuple[str, ...] | None:
        """Ask for a listing and wait no longer than `deadline` for it."""
        with self._lock:
            slow = key in self._slow
        job = self._submit(key)
        if slow:
            return None
        if job.done.wait(deadline) and job.names is not None:
            return job.names
        with self._lock:
            self._slow.add(key)
        return None

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            # A short join only. A scan blocked on an unresponsive filesystem
            # cannot be interrupted, and the shell must not hang on exit
            # waiting for it; the thread is a daemon for exactly this case.
            thread.join(0.2)

    # -- internals -----------------------------------------------------------

    def _submit(self, key: str) -> _Job:
        with self._lock:
            for existing in (self._running, self._pending):
                if existing is not None and existing.key == key:
                    return existing  # already being answered; do not duplicate it
            if self._pending is not None:
                # Coalesce. The waiter for the superseded request is released
                # now with no names rather than left to time out on work that
                # will never run.
                self._pending.done.set()
            job = _Job(key)
            self._pending = job
            if self._thread is None and not self._stop.is_set():
                # Started on first use, so a session that never types a path
                # never pays for a thread.
                self._thread = threading.Thread(
                    target=self._loop, name="offset-ghost", daemon=True
                )
                self._thread.start()
            self._wake.set()
            return job

    def _loop(self) -> None:
        while not self._stop.is_set():
            # A bounded wait rather than an indefinite one: `close` sets both
            # events, but a spurious wake must not spin either.
            if not self._wake.wait(0.2):
                continue
            with self._lock:
                self._wake.clear()
                job = self._pending
                self._pending = None
                self._running = job
            if job is None:
                continue
            try:
                names = self._scan(job.key)
            except Exception:
                # The reader already swallows OSError; anything else is a bug,
                # and losing one suggestion is a better outcome than losing the
                # thread and with it every later suggestion.
                names = ()
            with self._lock:
                self._running = None
                self._slow.discard(job.key)
            job.names = names
            job.done.set()
            if self._on_ready is not None:
                try:
                    self._on_ready()
                except Exception:
                    # A repaint hook that raises must not kill the worker.
                    pass

    def _scan(self, key: str) -> tuple[str, ...]:
        """Refresh one directory, re-reading it only if its mtime moved."""
        with self._lock:
            entry = self._cache.get(key)
        try:
            mtime = os.stat(key).st_mtime_ns
        except OSError:
            with self._lock:
                self._cache.pop(key, None)
            return ()
        if entry is not None and entry.mtime == mtime:
            # Nothing has been created, renamed or deleted here, so the names
            # cannot have changed: extend the trust instead of reading again.
            entry.stamp = time.monotonic()
            return entry.names
        names = self._read(key)
        self.reads += 1
        with self._lock:
            self._cache[key] = _Listing(names, mtime, time.monotonic())
        return names


# -- the engine -------------------------------------------------------------


def _last_token(text: str) -> str:
    """The whitespace-delimited chunk the cursor is at the end of.

    Empty when the text ends in whitespace, which is the correct answer: there
    is no token there yet.
    """
    tail = text
    for space in (" ", "\t", "\n"):
        tail = tail.rsplit(space, 1)[-1]
    return tail


def _looks_like_path(token: str) -> bool:
    """Whether a token is worth spending a directory read on.

    A bare word is not: almost every word of English prose would trigger a scan
    and most would match some file, so the prompt would fill with nonsense
    completions of ordinary sentences. A separator or a leading dot is the
    user saying they mean a path.
    """
    return bool(token) and ("/" in token or token.startswith(("~", ".")))


def _pick(candidates: Sequence[str], prefix: str) -> str:
    """The remainder of the best candidate extending `prefix`, or "".

    The shortest match wins, ties alphabetically. The obvious alternative - the
    longest common prefix of all matches - is wrong for ghost text: it draws
    text that is not any real command or filename, so accepting it leaves the
    user with a value that does not exist. A complete candidate can always be
    accepted, and typing one more character moves to the next one.
    """
    best = ""
    for candidate in candidates:
        if len(candidate) <= len(prefix) or not candidate.startswith(prefix):
            continue
        if not best or (len(candidate), candidate) < (len(best), best):
            best = candidate
    return best[len(prefix):] if best else ""


class Suggester:
    """Produces the ghost completion for a buffer.

    Not thread-safe: `suggest` is meant to be called from the one thread that
    owns the input box. The worker it drives is internally locked, so a slow
    scan cannot corrupt anything, but two concurrent callers would race on the
    history list.
    """

    def __init__(
        self,
        *,
        workspace: Path | str | None = None,
        history: Sequence[str] = (),
        commands: Callable[[], Mapping[str, Sequence[str]]] = command_table,
        scan: Callable[[str], tuple[str, ...]] = read_dir,
        deadline: float = DEADLINE,
        ttl: float = TTL,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        #: Resolved once, off the keypress path, so that a workspace reached
        #: through a symlink still matches the paths derived from it below.
        self._workspace = Path(workspace or Path.cwd()).resolve()
        self._root = str(self._workspace)
        #: Newest first, which is the order the history source wants and the
        #: order a caller seeding from a session should supply.
        self._history: list[str] = [line for line in history if line]
        self._commands = commands
        self._table: dict[str, tuple[str, ...]] | None = None
        self._deadline = deadline
        self._scanner = _Scanner(scan, ttl=ttl, on_ready=on_ready)

    # -- history -------------------------------------------------------------

    def remember(self, line: str) -> None:
        """Record a prompt the user actually sent.

        Newest first, and de-duplicated: re-typing yesterday's command should
        move it to the front rather than leaving two of it in the list, because
        a prefix match would otherwise offer the same completion twice and the
        second one could never be reached.
        """
        line = line.strip()
        if not line:
            return
        if line in self._history:
            self._history.remove(line)
        self._history.insert(0, line)
        del self._history[HISTORY_LIMIT:]

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    # -- suggestion ----------------------------------------------------------

    def suggest(self, buffer: str, cursor: int | None = None) -> Suggestion | None:
        """The completion for this buffer, or None. Never blocks past the deadline."""
        if cursor is None:
            cursor = len(buffer)
        if not buffer.strip():
            return None
        if cursor != len(buffer):
            # Anything after the cursor - the tail of a word, or a following
            # line - means an appended completion would land in the middle of
            # the user's own text, which they cannot accept coherently.
            return None
        token = _last_token(buffer)
        for source in (self._command, self._path, self._recall):
            found = source(buffer, token)
            if found is not None:
                return found
        return None

    def close(self) -> None:
        """Stop the worker. The suggester is not reusable afterwards."""
        self._scanner.close()

    # -- sources -------------------------------------------------------------

    def _command(self, buffer: str, token: str) -> Suggestion | None:
        """Command names, then the literal arguments of a named command."""
        if not buffer.startswith("/") or "\n" in buffer:
            return None  # a slash command is always one line
        if self._table is None:
            # Read once: the command set is fixed after `shell.commands` has
            # imported its extensions, and rebuilding it per keystroke would
            # re-parse every usage string for nothing.
            self._table = {name: tuple(subs) for name, subs in self._commands().items()}
        parts = buffer.split()
        trailing = buffer[-1].isspace()
        if len(parts) == 1 and not trailing:
            prefix = parts[0][1:].lower()
            if not prefix:
                return None  # a lone slash says nothing about which command
            remainder = _pick(sorted(self._table), prefix)
            return Suggestion(remainder, SOURCE_COMMAND) if remainder else None
        if len(parts) == 1:
            prefix = ""
        elif len(parts) == 2 and not trailing:
            prefix = parts[1]
        else:
            return None  # past the first argument there is nothing declared
        subs = self._table.get(parts[0].lstrip("/").lower())
        if not subs:
            return None
        remainder = _pick(subs, prefix)
        return Suggestion(remainder, SOURCE_COMMAND) if remainder else None

    def _path(self, buffer: str, token: str) -> Suggestion | None:
        """A filename under the workspace, if the token looks like a path."""
        if buffer.startswith("/") and len(buffer.split()) == 1:
            return None  # that leading slash is the command sigil, not a root
        if not _looks_like_path(token):
            return None
        resolved = self._resolve(token)
        if resolved is None:
            return None
        key, prefix = resolved
        names = self._scanner.cached(key)
        if names is None:
            names = self._scanner.fetch(key, self._deadline)
        if names is None:
            return None
        if not prefix.startswith("."):
            # Hidden files are noise unless the user has said `.`, and asking
            # for them is exactly what a leading dot means.
            names = tuple(name for name in names if not name.startswith("."))
        remainder = _pick(names, prefix)
        return Suggestion(remainder, SOURCE_PATH) if remainder else None

    def _resolve(self, token: str) -> tuple[str, str] | None:
        """The directory to scan and the prefix to match, or None if out of scope.

        Normalised lexically rather than with `Path.resolve`, because resolve
        issues readlink calls and this runs on the keypress path. The workspace
        root was already resolved in `__init__`, so a token that stays inside
        the tree normalises to a path under it.

        Confinement is not a security boundary - the user can type any path
        they like - it is a cost bound. Completing `/usr/lib` would read a
        directory that has nothing to do with the project.
        """
        raw = os.path.expanduser(token)
        head, sep, prefix = raw.rpartition("/")
        if not sep:
            base, prefix = ".", raw
        else:
            base = head or "/"
        combined = base if os.path.isabs(base) else os.path.join(self._root, base)
        key = os.path.normpath(combined)
        if key != self._root and not key.startswith(self._root + os.sep):
            return None
        return key, prefix

    def _recall(self, buffer: str, token: str) -> Suggestion | None:
        """A previous prompt that starts with what has been typed, newest first."""
        for past in self._history:
            if len(past) > len(buffer) and past.startswith(buffer):
                return Suggestion(past[len(buffer):], SOURCE_HISTORY)
        return None
