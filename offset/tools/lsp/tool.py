"""The two model-facing LSP tools.

Split in two because a tool carries one `Danger` and these are not the same
risk: asking where a symbol is defined cannot change the tree, while applying a
rename rewrites every file that mentions it.  One tool with an `apply` flag
would have to be classified at the dangerous end, which would put a hover
behind an approval prompt in `safe` mode and train the user to wave edits
through.  So `lsp` is SAFE and read-only, `lsp_edit` is WRITE.

Positions are the other reason this layer exists.  The protocol addresses code
by zero-based line and UTF-16 code unit; a model reasons about it as "the
`connect` on line 40".  Every entry point here takes a 1-based `line` and an
optional `symbol` substring, finds the column by reading the line, and converts
at the boundary - `protocol.to_utf16` handles the astral-plane case that makes
a naive character index wrong.  Asking the model for a column would be asking
it to count characters, which it cannot reliably do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from offset.tools.base import Danger, Tool, ToolContext, ToolResult
from offset.tools.lsp.client import LSPClient, LSPError, LSPTimeout, ServerGone, Unsupported
from offset.tools.lsp.protocol import Diagnostic, Location, Position, Symbol, to_utf16
from offset.tools.lsp.servers import Servers

#: Read-only actions, and the capability each one needs.  Kept beside the
#: dispatch table so a new action cannot be added without declaring its gate.
QUERY_ACTIONS: Final[dict[str, str]] = {
    "diagnostics": "",  # notification-driven; no request capability to check
    "definition": "definitionProvider",
    "type_definition": "typeDefinitionProvider",
    "implementation": "implementationProvider",
    "references": "referencesProvider",
    "hover": "hoverProvider",
    "symbols": "documentSymbolProvider",
    "status": "",
}

EDIT_ACTIONS: Final[dict[str, str]] = {
    "rename": "renameProvider",
    "code_actions": "codeActionProvider",
    "format": "documentFormattingProvider",
}

#: How many locations or diagnostics to render before summarising.  A reference
#: search on a popular helper can return hundreds; the model needs the shape of
#: the answer, not every row.
LIMIT: Final = 40


def _resolve_line(text: str, line: int) -> tuple[str, str]:
    """The 1-based `line` of `text`, or an empty string and a reason."""
    lines = text.splitlines()
    if line < 1:
        return "", f"line {line} is before the start of the file"
    if line > len(lines):
        return "", f"line {line} is past the end of the file ({len(lines)} lines)"
    return lines[line - 1], ""


def _position(text: str, line: int, symbol: str | None) -> tuple[Position | None, str]:
    """Where in the file to ask about.

    `symbol` is a substring of the line, which is how a model naturally refers
    to code.  Without it the request lands on the first non-blank character,
    which is right for line-scoped questions and wrong for anything else - so a
    missing symbol on a crowded line is reported rather than guessed at.
    """
    row, why = _resolve_line(text, line)
    if why:
        return None, why
    if not symbol:
        column = len(row) - len(row.lstrip())
        return Position(line - 1, to_utf16(row, column)), ""
    index = row.find(symbol)
    if index < 0:
        return None, f"{symbol!r} is not on line {line}: {row.strip()[:60]!r}"
    return Position(line - 1, to_utf16(row, index)), ""


def _read(ctx: ToolContext, path: str) -> tuple[Path | None, str, str]:
    """Resolve and read a file inside the workspace."""
    try:
        resolved = ctx.resolve(path)
    except PermissionError as exc:
        return None, "", str(exc)
    if not resolved.exists():
        return None, "", f"no such file: {path}"
    if resolved.is_dir():
        return None, "", f"{path} is a directory"
    try:
        return resolved, resolved.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return None, "", f"could not read {path}: {exc}"


def _locations(found: list[Location], root: Path) -> list[str]:
    lines = []
    for loc in found[:LIMIT]:
        try:
            shown = loc.path.relative_to(root)
        except (ValueError, AttributeError):
            shown = loc.path
        lines.append(f"{shown}:{loc.range.start.line + 1}:{loc.range.start.character + 1}")
    if len(found) > LIMIT:
        lines.append(f"... {len(found) - LIMIT} more")
    return lines


def _diagnostics(items: list[Diagnostic], path: Path, root: Path) -> list[str]:
    try:
        shown = path.relative_to(root)
    except ValueError:
        shown = path
    lines = []
    for d in items[:LIMIT]:
        where = f"{shown}:{d.range.start.line + 1}:{d.range.start.character + 1}"
        source = f" [{d.source}]" if d.source else ""
        lines.append(f"{d.severity}: {where}{source} {d.message}")
    if len(items) > LIMIT:
        lines.append(f"... {len(items) - LIMIT} more")
    return lines


def _symbols(found: list[Symbol]) -> list[str]:
    lines = []
    for s in found[:LIMIT]:
        container = f" in {s.container}" if getattr(s, "container", "") else ""
        line = getattr(s.range.start, "line", 0) + 1 if getattr(s, "range", None) else 0
        lines.append(f"{s.kind} {s.name}{container} :{line}")
    if len(found) > LIMIT:
        lines.append(f"... {len(found) - LIMIT} more")
    return lines


class _Base(Tool):
    """Shared plumbing: one `Servers` per tool instance, lazily started."""

    __slots__ = ("servers",)

    def __init__(self, servers: Servers | None = None) -> None:
        self.servers = servers if servers is not None else Servers()

    def _client(self, ctx: ToolContext, path: Path) -> tuple[LSPClient | None, str]:
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        return self.servers.for_file(path, root)

    def _fail(self, exc: Exception) -> ToolResult:
        """Protocol faults are answers, not crashes."""
        if isinstance(exc, Unsupported):
            return ToolResult.fail(str(exc))
        if isinstance(exc, LSPTimeout):
            return ToolResult.fail(f"the language server did not answer in time: {exc}")
        if isinstance(exc, ServerGone):
            return ToolResult.fail(f"the language server exited: {exc}")
        if isinstance(exc, LSPError):
            return ToolResult.fail(str(exc))
        return ToolResult.fail(f"{type(exc).__name__}: {exc}")


class LspQuery(_Base):
    """Read-only code intelligence."""

    name = "lsp"
    description = (
        "Ask a language server about code: diagnostics, definition, references, hover, "
        "symbols. Follows imports, re-exports and shadowing, so it finds callsites that "
        "text search misses. Give a 1-based line and the symbol text on that line."
    )
    danger = Danger.SAFE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(QUERY_ACTIONS),
                "description": "what to ask",
            },
            "file": {"type": "string", "description": "path, relative to the workspace"},
            "line": {"type": "integer", "minimum": 1, "description": "1-based line number"},
            "symbol": {
                "type": "string",
                "maxLength": 200,
                "description": "the identifier on that line to ask about",
            },
            "query": {
                "type": "string",
                "maxLength": 200,
                "description": "for action=symbols, search the whole workspace instead of one file",
            },
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        action = args.get("action", "?")
        target = args.get("file") or args.get("query") or ""
        return f"lsp {action} {target}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        if action not in QUERY_ACTIONS:
            return ToolResult.fail(
                f"no lsp action {action!r}. available: {', '.join(sorted(QUERY_ACTIONS))}"
            )

        if action == "status":
            lines = self.servers.report()
            return ToolResult.text("\n".join(lines) or "no language servers started")

        if action == "symbols" and args.get("query") and not args.get("file"):
            return self._workspace_symbols(str(args["query"]), ctx)

        path_arg = args.get("file")
        if not path_arg:
            return ToolResult.fail(f"action {action!r} needs a file")
        path, text, why = _read(ctx, str(path_arg))
        if path is None:
            return ToolResult.fail(why)

        client, why = self._client(ctx, path)
        if client is None:
            return ToolResult.fail(why)

        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        try:
            client.sync(path, text)
            ctx.check()
            if action == "diagnostics":
                found = client.diagnostics(path)
                if not found:
                    return ToolResult.text(f"no diagnostics for {path_arg}")
                return ToolResult.text("\n".join(_diagnostics(found, path, root)))

            if action == "symbols":
                found = client.document_symbols(path)
                if not found:
                    return ToolResult.text(f"no symbols in {path_arg}")
                return ToolResult.text("\n".join(_symbols(found)))

            line = args.get("line")
            if not isinstance(line, int):
                return ToolResult.fail(f"action {action!r} needs a line")
            position, why = _position(text, line, args.get("symbol"))
            if position is None:
                return ToolResult.fail(why)

            if action == "hover":
                answer = client.hover(path, position)
                return ToolResult.text(answer or "no hover information here")

            getter = {
                "definition": client.definition,
                "type_definition": client.type_definition,
                "implementation": client.implementation,
                "references": client.references,
            }[action]
            found = getter(path, position)
            if not found:
                return ToolResult.text(f"no {action.replace('_', ' ')} found")
            return ToolResult.text("\n".join(_locations(found, root)))
        except Exception as exc:  # protocol faults are values here
            return self._fail(exc)

    def _workspace_symbols(self, query: str, ctx: ToolContext) -> ToolResult:
        """Search every started server; a cold workspace has none, and says so."""
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        collected: list[Symbol] = []
        problems: list[str] = []
        for status in self.servers.status():
            client = self.servers.live(status.language, root) if hasattr(self.servers, "live") else None
            if client is None:
                client, why = self.servers.client(status.language, root)
                if client is None:
                    problems.append(why)
                    continue
            try:
                collected.extend(client.workspace_symbols(query))
            except Exception as exc:
                problems.append(f"{status.language}: {exc}")
        if not collected:
            note = "; ".join(problems) if problems else "no matching symbols"
            return ToolResult.text(note)
        return ToolResult.text("\n".join(_symbols(collected)))


class LspEdit(_Base):
    """Symbol-aware edits: rename, code actions, formatting."""

    name = "lsp_edit"
    description = (
        "Apply a symbol-aware edit through a language server: rename across every "
        "referencing file, apply a code action, or format a document. A rename here "
        "follows shadowing and re-exports, which a text replace does not."
    )
    danger = Danger.WRITE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(EDIT_ACTIONS), "description": "what to do"},
            "file": {"type": "string", "description": "path, relative to the workspace"},
            "line": {"type": "integer", "minimum": 1, "description": "1-based line number"},
            "symbol": {
                "type": "string",
                "maxLength": 200,
                "description": "the identifier on that line to act on",
            },
            "new_name": {"type": "string", "maxLength": 200, "description": "for action=rename"},
            "apply": {
                "type": "boolean",
                "description": "write the edit; false previews the files it would touch",
            },
        },
        "required": ["action", "file"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        action = args.get("action", "?")
        return f"lsp_edit {action} {args.get('file', '')}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        if action not in EDIT_ACTIONS:
            return ToolResult.fail(
                f"no lsp_edit action {action!r}. available: {', '.join(sorted(EDIT_ACTIONS))}"
            )
        path, text, why = _read(ctx, str(args.get("file", "")))
        if path is None:
            return ToolResult.fail(why)
        client, why = self._client(ctx, path)
        if client is None:
            return ToolResult.fail(why)

        apply = bool(args.get("apply", True))
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        try:
            client.sync(path, text)
            ctx.check()

            if action == "format":
                edits = client.formatting(path)
                if not edits:
                    return ToolResult.text("already formatted")
                return self._write_one(path, text, edits, apply=apply)

            if action == "code_actions":
                line = args.get("line")
                if not isinstance(line, int):
                    return ToolResult.fail("code_actions needs a line")
                position, why = _position(text, line, args.get("symbol"))
                if position is None:
                    return ToolResult.fail(why)
                actions = client.code_actions(path, position)
                if not actions:
                    return ToolResult.text("no code actions here")
                titles = [f"{i}. {a.title}" for i, a in enumerate(actions, 1)]
                return ToolResult.text("\n".join(titles))

            # rename
            new_name = str(args.get("new_name", "")).strip()
            if not new_name:
                return ToolResult.fail("rename needs new_name")
            line = args.get("line")
            if not isinstance(line, int):
                return ToolResult.fail("rename needs a line")
            position, why = _position(text, line, args.get("symbol"))
            if position is None:
                return ToolResult.fail(why)
            edit = client.rename(path, position, new_name)
            return self._apply_workspace(edit, root, apply=apply)
        except Exception as exc:
            return self._fail(exc)

    def _write_one(self, path: Path, text: str, edits: list[Any], *, apply: bool) -> ToolResult:
        from offset.tools.lsp.protocol import apply_edits

        updated = apply_edits(text, edits)
        if not apply:
            return ToolResult.text(f"would rewrite {path.name} ({len(edits)} edit(s))")
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult.fail(f"could not write {path.name}: {exc}")
        return ToolResult.text(f"rewrote {path.name} ({len(edits)} edit(s))")

    def _apply_workspace(self, edit: Any, root: Path, *, apply: bool) -> ToolResult:
        """Write a `WorkspaceEdit` across every file it touches.

        Read-modify-write per file, because the server's ranges are against the
        text it was last told about; applying them to anything else corrupts the
        file quietly.
        """
        from offset.tools.lsp.protocol import apply_edits

        changes = getattr(edit, "changes", None) or {}
        if not changes:
            return ToolResult.text("the server proposed no changes")

        touched: list[str] = []
        failures: list[str] = []
        for uri, edits in changes.items():
            from offset.tools.lsp.protocol import from_uri

            try:
                target = from_uri(uri)
            except (ValueError, OSError) as exc:
                failures.append(f"{uri}: {exc}")
                continue
            try:
                shown = str(target.relative_to(root))
            except ValueError:
                shown = str(target)
            if not apply:
                touched.append(f"{shown} ({len(edits)} edit(s))")
                continue
            try:
                before = target.read_text(encoding="utf-8")
                target.write_text(apply_edits(before, list(edits)), encoding="utf-8")
            except OSError as exc:
                failures.append(f"{shown}: {exc}")
                continue
            touched.append(f"{shown} ({len(edits)} edit(s))")

        verb = "would change" if not apply else "changed"
        lines = [f"{verb} {len(touched)} file(s)"] + touched
        if failures:
            lines.append("failed:")
            lines.extend(f"  {f}" for f in failures)
        return ToolResult(ok=not failures, content="\n".join(lines))


def lsp_tools(servers: Servers | None = None) -> list[Tool]:
    """The pair, sharing one server pool so a query and an edit reuse a process."""
    shared = servers if servers is not None else Servers()
    return [LspQuery(shared), LspEdit(shared)]
