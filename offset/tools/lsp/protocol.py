"""Wire format for the Language Server Protocol: framing, URIs and types.

LSP is JSON-RPC 2.0, exactly as MCP is, but its framing is HTTP-style headers
rather than one object per line — and that single difference is where most
hand-written clients break.  `Content-Length` counts *bytes* of UTF-8, the
header block may or may not carry `Content-Type`, and a pipe hands over
whatever bytes happened to arrive, so one frame routinely spans several reads
and two frames routinely arrive in one.  `Framer` therefore owns a byte buffer
and yields a message only once the declared number of bytes is present.  It is
the only thing in this package permitted to touch the stream; a partial read
must never be able to desynchronise it.

Positions are the other trap.  LSP counts lines from zero and measures columns
in UTF-16 code units; a person, and every compiler message a person has ever
read, counts lines from one and columns in characters.  The conversion happens
at exactly two boundaries: `Position` always holds the protocol's numbering,
and the tool layer converts on the way in and on the way out.  Storing a mix
would mean guessing later which convention a number arrived in, which is the
same class of bug as the framing one and just as tedious to find.

Everything here is pure: no process, no socket, no file.  That is what makes
the framing testable one byte at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit
from urllib.request import pathname2url, url2pathname

#: Longest single frame accepted.  A server stuck in a print loop must not be
#: able to grow this process without bound.
MAX_FRAME: Final = 16 << 20

#: Longest header block accepted before the buffer is assumed to be noise
#: rather than a frame in progress.  Real header blocks are under 100 bytes.
MAX_HEADER: Final = 8 << 10

#: Both separators are in the wild.  The spec says CRLF, and a server sending
#: LF only would otherwise hang us forever waiting for a header that arrived.
_CRLF: Final = b"\r\n\r\n"
_LF: Final = b"\n\n"


# -- framing ----------------------------------------------------------------


def encode(message: dict[str, Any]) -> bytes:
    """Frame one JSON-RPC message.

    `ensure_ascii=False` keeps the payload compact for non-ASCII content, which
    is precisely why the length is taken from the *encoded bytes*: a header
    counting characters would leave the reader one frame behind for the rest of
    the session.
    """
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return b"Content-Length: " + str(len(body)).encode("ascii") + _CRLF + body


def _header_end(buffer: bytearray) -> tuple[int, int]:
    """Index of the header terminator and its width, or `(-1, 0)`."""
    crlf = buffer.find(_CRLF)
    lf = buffer.find(_LF)
    if crlf >= 0 and (lf < 0 or crlf <= lf):
        return crlf, len(_CRLF)
    if lf >= 0:
        return lf, len(_LF)
    return -1, 0


def content_length(header: str) -> int | None:
    """The declared body length, ignoring `Content-Type` and header case."""
    for line in header.replace("\r\n", "\n").split("\n"):
        name, _, value = line.partition(":")
        if name.strip().lower() != "content-length":
            continue
        try:
            length = int(value.strip())
        except ValueError:
            return None
        return length if length >= 0 else None
    return None


class Framer:
    """Reassembles `Content-Length` frames from arbitrary byte chunks.

    Nothing is consumed until a whole body is present, so feeding one byte at a
    time gives the same messages as feeding the entire session at once.
    """

    __slots__ = ("_buffer", "_skip", "dropped", "max_frame")

    def __init__(self, *, max_frame: int = MAX_FRAME) -> None:
        self.max_frame = max_frame
        #: Frames thrown away: no length, oversized, or not a JSON object.
        self.dropped = 0
        self._buffer = bytearray()
        self._skip = 0

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        """Every complete message the buffer now holds, in arrival order."""
        self._buffer += chunk
        out: list[dict[str, Any]] = []
        while True:
            if self._skip:
                drop = min(self._skip, len(self._buffer))
                del self._buffer[:drop]
                self._skip -= drop
                if self._skip:
                    return out
            end, width = _header_end(self._buffer)
            if end < 0:
                if len(self._buffer) > MAX_HEADER:
                    # A server printing a banner and never framing anything is
                    # the observed failure here; keep only enough tail for a
                    # separator that is itself split across reads.
                    self.dropped += 1
                    del self._buffer[: len(self._buffer) - 3]
                return out
            header = bytes(self._buffer[:end]).decode("ascii", "replace")
            length = content_length(header)
            if length is None:
                self.dropped += 1
                del self._buffer[: end + width]
                continue
            if length > self.max_frame:
                self.dropped += 1
                del self._buffer[: end + width]
                self._skip = length
                continue
            start = end + width
            if len(self._buffer) - start < length:
                return out  # incomplete body: consume nothing, wait for more
            body = bytes(self._buffer[start : start + length])
            del self._buffer[: start + length]
            try:
                frame = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self.dropped += 1
                continue
            if isinstance(frame, dict):
                out.append(frame)
            else:
                self.dropped += 1


# -- URIs -------------------------------------------------------------------


def to_uri(path: Path | str) -> str:
    """A `file://` URI for an absolute path, percent-encoding what must be.

    Spaces and non-ASCII characters are ordinary in real trees, and a server
    handed a raw one either errors or silently answers about a different file.
    """
    absolute = Path(path).expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    return "file://" + pathname2url(str(absolute))


def from_uri(uri: str) -> Path:
    """The path a `file://` URI names.  A bare path is passed through."""
    if not uri.startswith("file:"):
        return Path(uri)
    parts = urlsplit(uri)
    body = parts.path
    if parts.netloc and parts.netloc not in ("", "localhost"):
        body = f"//{parts.netloc}{parts.path}"  # a UNC share, spelled as one
    return Path(url2pathname(body))


def normalise_uri(uri: str) -> str:
    """Re-spell a server's URI the way `to_uri` would.

    Servers vary in how they encode the same path, and diagnostics keyed by one
    spelling would never be found under the other.
    """
    if not uri.startswith("file:"):
        return uri
    return to_uri(from_uri(uri))


# -- columns ----------------------------------------------------------------


def to_utf16(line: str, index: int) -> int:
    """Character index -> UTF-16 code units, which is what LSP columns are."""
    return len(line[:index].encode("utf-16-le")) // 2


def from_utf16(line: str, units: int) -> int:
    """UTF-16 code units -> character index, clamped to the line."""
    if units <= 0:
        return 0
    seen = 0
    for index, char in enumerate(line):
        if seen >= units:
            return index
        seen += 2 if ord(char) > 0xFFFF else 1
    return len(line)


# -- types ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Position:
    """A point in a document, in the protocol's own zero-based numbering."""

    line: int = 0
    character: int = 0

    @classmethod
    def parse(cls, raw: Any) -> "Position":
        if not isinstance(raw, dict):
            return cls()
        return cls(_int(raw.get("line")), _int(raw.get("character")))

    def wire(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}

    @property
    def human(self) -> str:
        """`line:column`, both counted from one, for a person to read."""
        return f"{self.line + 1}:{self.character + 1}"


@dataclass(frozen=True, slots=True)
class Range:
    start: Position = field(default_factory=Position)
    end: Position = field(default_factory=Position)

    @classmethod
    def parse(cls, raw: Any) -> "Range":
        if not isinstance(raw, dict):
            return cls()
        start = Position.parse(raw.get("start"))
        return cls(start, Position.parse(raw.get("end")) if "end" in raw else start)

    @classmethod
    def at(cls, position: Position) -> "Range":
        return cls(position, position)

    def wire(self) -> dict[str, Any]:
        return {"start": self.start.wire(), "end": self.end.wire()}

    @property
    def one_line(self) -> bool:
        return self.start.line == self.end.line


@dataclass(frozen=True, slots=True)
class Location:
    uri: str = ""
    range: Range = field(default_factory=Range)

    @classmethod
    def parse(cls, raw: Any) -> "Location | None":
        """A `Location` or a `LocationLink`; servers send whichever they like.

        `linkSupport` is declared, so both shapes must be understood — a client
        that declares it and then only reads `uri` finds no definitions at all.
        """
        if not isinstance(raw, dict):
            return None
        if "targetUri" in raw:
            uri = raw.get("targetUri")
            span = raw.get("targetSelectionRange") or raw.get("targetRange")
        else:
            uri = raw.get("uri")
            span = raw.get("range")
        if not isinstance(uri, str) or not uri:
            return None
        return cls(normalise_uri(uri), Range.parse(span))

    @property
    def path(self) -> Path:
        return from_uri(self.uri)

    def label(self, root: Path | None = None) -> str:
        """`path:line:column`, counted from one."""
        path = self.path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return f"{path}:{self.range.start.human}"


def locations(result: Any) -> list[Location]:
    """Flatten the three shapes a location request may answer with."""
    if result is None:
        return []
    raw = result if isinstance(result, list) else [result]
    out = [Location.parse(item) for item in raw]
    return [item for item in out if item is not None]


#: Names for `DiagnosticSeverity`, whose numbers nobody remembers.
SEVERITY: Final = {1: "error", 2: "warning", 3: "info", 4: "hint"}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    range: Range = field(default_factory=Range)
    message: str = ""
    severity: int = 1
    source: str = ""
    code: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, item: Any) -> "Diagnostic | None":
        if not isinstance(item, dict):
            return None
        code = item.get("code")
        severity = item.get("severity")
        return cls(
            range=Range.parse(item.get("range")),
            message=str(item.get("message") or "").strip(),
            severity=severity if isinstance(severity, int) and severity in SEVERITY else 1,
            source=str(item.get("source") or ""),
            code="" if code is None else str(code),
            raw=item,
        )

    @property
    def level(self) -> str:
        return SEVERITY.get(self.severity, "error")

    def label(self) -> str:
        origin = " ".join(part for part in (self.source, self.code) if part)
        head = f"{self.range.start.human} {self.level}"
        return f"{head}: {self.message}" + (f" [{origin}]" if origin else "")


def diagnostics(items: Any) -> list[Diagnostic]:
    parsed = [Diagnostic.parse(item) for item in items if isinstance(items, list)]
    return [item for item in parsed if item is not None]


#: `SymbolKind`, so a listing says "function" rather than "12".
SYMBOL_KINDS: Final = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object",
    20: "key", 21: "null", 22: "enum member", 23: "struct", 24: "event",
    25: "operator", 26: "type parameter",
}


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str = ""
    kind: int = 0
    location: Location = field(default_factory=Location)
    container: str = ""
    detail: str = ""
    depth: int = 0

    @property
    def kind_name(self) -> str:
        return SYMBOL_KINDS.get(self.kind, "symbol")

    def label(self, root: Path | None = None) -> str:
        indent = "  " * self.depth
        scope = f" in {self.container}" if self.container else ""
        detail = f" {self.detail}" if self.detail else ""
        return f"{indent}{self.kind_name} {self.name}{detail}{scope} — {self.location.label(root)}"


def symbols(result: Any, *, uri: str = "") -> list[Symbol]:
    """Flatten either symbol shape into one depth-tagged list.

    `documentSymbol` may answer with a nested `DocumentSymbol[]` or a flat
    `SymbolInformation[]`, and a client that reads only one of them shows an
    empty outline against half the servers in existence.
    """
    out: list[Symbol] = []
    _collect(result, uri=uri, container="", depth=0, out=out)
    return out


def _collect(result: Any, *, uri: str, container: str, depth: int, out: list[Symbol]) -> None:
    if not isinstance(result, list):
        return
    for item in result:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = _int(item.get("kind"))
        if isinstance(item.get("location"), dict):  # SymbolInformation
            where = Location.parse(item["location"]) or Location(normalise_uri(uri))
            parent = str(item.get("containerName") or "")
            out.append(Symbol(name, kind, where, parent, "", depth))
            continue
        span = item.get("selectionRange") or item.get("range")
        where = Location(normalise_uri(uri), Range.parse(span))
        out.append(Symbol(name, kind, where, container, str(item.get("detail") or ""), depth))
        _collect(item.get("children"), uri=uri, container=name, depth=depth + 1, out=out)


@dataclass(frozen=True, slots=True)
class TextEdit:
    range: Range = field(default_factory=Range)
    new_text: str = ""

    @classmethod
    def parse(cls, raw: Any) -> "TextEdit | None":
        if not isinstance(raw, dict) or "range" not in raw:
            return None
        return cls(Range.parse(raw.get("range")), str(raw.get("newText") or ""))

    def wire(self) -> dict[str, Any]:
        return {"range": self.range.wire(), "newText": self.new_text}


def text_edits(raw: Any) -> list[TextEdit]:
    if not isinstance(raw, list):
        return []
    parsed = [TextEdit.parse(item) for item in raw]
    return [edit for edit in parsed if edit is not None]


@dataclass(slots=True)
class WorkspaceEdit:
    """Edits keyed by URI, plus any file operation offset will not perform.

    `documentChanges` can also create, rename and delete files.  Those are
    recorded rather than executed: a rename that moves a file is not what the
    caller asked for, and doing it silently would be worse than saying so.
    """

    changes: dict[str, list[TextEdit]] = field(default_factory=dict)
    operations: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: Any) -> "WorkspaceEdit":
        out = cls()
        if not isinstance(raw, dict):
            return out
        plain = raw.get("changes")
        if isinstance(plain, dict):
            for uri, edits in plain.items():
                parsed = text_edits(edits)
                if parsed:
                    out.changes.setdefault(normalise_uri(str(uri)), []).extend(parsed)
        for item in raw.get("documentChanges") or ():
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind in ("create", "rename", "delete"):
                target = item.get("uri") or item.get("oldUri") or "?"
                out.operations.append(f"{kind} {from_uri(str(target))}")
                continue
            doc = item.get("textDocument")
            uri = doc.get("uri") if isinstance(doc, dict) else None
            parsed = text_edits(item.get("edits"))
            if isinstance(uri, str) and parsed:
                out.changes.setdefault(normalise_uri(uri), []).extend(parsed)
        return out

    @property
    def empty(self) -> bool:
        return not self.changes and not self.operations

    def report(self, root: Path | None = None) -> list[str]:
        lines: list[str] = []
        for uri in sorted(self.changes):
            path = from_uri(uri)
            if root is not None:
                try:
                    path = path.relative_to(root)
                except ValueError:
                    pass
            edits = self.changes[uri]
            lines.append(f"{path}: {len(edits)} edit{'s' if len(edits) != 1 else ''}")
            for edit in sorted(edits, key=lambda e: (e.range.start.line, e.range.start.character)):
                shown = edit.new_text.replace("\n", "\\n")
                lines.append(f"  {edit.range.start.human} -> {shown!r}")
        lines.extend(f"not applied: {op}" for op in self.operations)
        return lines


@dataclass(slots=True)
class CodeAction:
    title: str = ""
    kind: str = ""
    edit: WorkspaceEdit | None = None
    command: dict[str, Any] | None = None
    diagnostics: int = 0
    preferred: bool = False
    data: Any = None
    disabled: str = ""

    @classmethod
    def parse(cls, raw: Any) -> "CodeAction | None":
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("title") or "").strip()
        if not title:
            return None
        command = raw.get("command")
        if isinstance(command, str):
            # A bare `Command`, which older servers return in place of an
            # action; normalise it so callers only handle one shape.
            command = {"command": command, "arguments": raw.get("arguments") or []}
        elif not isinstance(command, dict):
            command = None
        reason = raw.get("disabled")
        edit = raw.get("edit")
        found = raw.get("diagnostics")
        return cls(
            title=title,
            kind=str(raw.get("kind") or ""),
            edit=WorkspaceEdit.parse(edit) if isinstance(edit, dict) else None,
            command=command,
            diagnostics=len(found) if isinstance(found, list) else 0,
            preferred=bool(raw.get("isPreferred")),
            data=raw.get("data"),
            disabled=str(reason.get("reason") or "") if isinstance(reason, dict) else "",
        )

    def label(self, index: int) -> str:
        marks = [part for part in (self.kind, "preferred" if self.preferred else "") if part]
        if self.disabled:
            marks.append(f"disabled: {self.disabled}")
        elif self.edit is None and self.command is None:
            marks.append("needs resolving")
        return f"{index}. {self.title}" + (f" ({', '.join(marks)})" if marks else "")


def code_actions(result: Any) -> list[CodeAction]:
    if not isinstance(result, list):
        return []
    parsed = [CodeAction.parse(item) for item in result]
    return [action for action in parsed if action is not None]


# -- applying edits ---------------------------------------------------------


def apply_edits(text: str, edits: list[TextEdit]) -> str:
    """Apply one document's edits to its text.

    LSP guarantees the edits in a single array are non-overlapping and are all
    expressed against the *original* document, so they are applied from the end
    backwards; going forwards would shift every position after the first.
    """
    if not edits:
        return text
    lines = text.splitlines(keepends=True) or [""]
    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    total = cursor

    def offset(position: Position) -> int:
        if position.line >= len(lines):
            return total
        line = lines[position.line]
        bare = line.rstrip("\r\n")
        return starts[position.line] + min(from_utf16(bare, position.character), len(line))

    ordered = sorted(edits, key=lambda e: (e.range.start.line, e.range.start.character), reverse=True)
    out = text
    for edit in ordered:
        start, end = offset(edit.range.start), offset(edit.range.end)
        if end < start:
            start, end = end, start
        out = out[:start] + edit.new_text + out[end:]
    return out


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
