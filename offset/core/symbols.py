"""Symbol extraction and the dependency graph.

Two honest halves, and the honesty is the point.  For Python the extractor is
the standard library's `ast` module: real parsing, so a method on a class
nested inside another class is attributed to `Outer.Inner`, a decorated
`async def` is still a function at the line its `def` sits on, and
`from x import y as z` records all three of module, member and local binding.
Nothing about that half is guesswork.

For JavaScript, TypeScript, Go, Rust, Java and C the extractor is a table of
line-anchored regular expressions.  **It is a heuristic and it is described as
one everywhere it surfaces.**  It finds top-level declarations written in the
ordinary way; it will miss a function returned from a factory, it does not know
that a `//` inside a string is not a comment, and it cannot see through a
macro.  Shipping a wrong answer confidently is worse than shipping a modest one
plainly labelled, so `Extraction.exact` says which half produced the result and
every renderer prints it.

The graph exists for one question the flat symbol table cannot answer: *who
depends on this file?*  Forward edges are cheap — the imports are right there
in the source.  The reverse edge is not, because it requires resolving every
import in the workspace to a path before you can invert the relation.  That
resolution is per-language and deliberately conservative: an import that cannot
be pointed at a file in this workspace produces no edge rather than a guess, so
`importers_of` under-reports on exotic build layouts and never invents a
dependency that is not there.

References are textual, and named as such.  `defines` comes from the parser;
`references` is a word-boundary scan of the files the caller says are worth
looking at, with any line that is itself a definition of that name excluded.
Resolving a reference properly needs type inference, which needs a type
checker, which is not in the standard library.
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Final, Iterable, Mapping

#: Extension to language.  The only place the mapping exists, so the index and
#: the extractors cannot disagree about what a `.mjs` file is.
LANGUAGES: Final[dict[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".css": "css",
    ".html": "html",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".rst": "markdown",
    ".txt": "text",
    ".cfg": "config",
    ".ini": "config",
}

#: Languages with a structural extractor.  Everything else is still indexed as
#: text — searchable, just without a symbol table.
STRUCTURED: Final[frozenset[str]] = frozenset(
    {"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp"}
)

#: Suffixes a relative JavaScript or TypeScript import may have left off.
_JS_SUFFIXES: Final = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def language_of(path: str | Path) -> str:
    """The language of `path` by extension, or `"unknown"`."""
    name = PurePosixPath(str(path).replace("\\", "/")).name
    dot = name.rfind(".")
    if dot <= 0:
        return "unknown"
    return LANGUAGES.get(name[dot:].lower(), "unknown")


# -- records ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Symbol:
    """One declaration.  `parent` is the dotted enclosing scope, `""` at top."""

    name: str
    kind: str  # function | method | class | variable | constant | type | impl | macro
    line: int
    parent: str = ""
    signature: str = ""
    decorators: tuple[str, ...] = ()

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def __str__(self) -> str:
        return f"{self.kind} {self.qualname}"


@dataclass(frozen=True, slots=True)
class Import:
    """One import edge as written.

    `module` is the thing imported from, `name` the member for a `from` import
    (`""` otherwise), `alias` the local binding when renamed, and `level` the
    number of leading dots on a Python relative import.
    """

    module: str
    name: str = ""
    alias: str = ""
    line: int = 0
    level: int = 0

    @property
    def dotted(self) -> str:
        """What the source says, normalised: `.pkg.mod.member`."""
        head = "." * self.level + self.module
        return f"{head}.{self.name}" if self.name else head


@dataclass(frozen=True, slots=True)
class Location:
    """Where a name occurs.  `kind` is the symbol kind, or `"reference"`."""

    path: str
    line: int
    kind: str
    name: str
    text: str = ""

    def __str__(self) -> str:
        tail = f"  {self.text}" if self.text else ""
        return f"{self.path}:{self.line}  {self.kind}{tail}"


@dataclass(frozen=True, slots=True)
class Extraction:
    """What one file declares and what it depends on.

    `exact` is False when a regular-expression table produced this rather than
    a parser.  Callers that report symbols to a human must say so.
    """

    language: str
    symbols: tuple[Symbol, ...] = ()
    imports: tuple[Import, ...] = ()
    exact: bool = False
    error: str = ""


# -- Python: the real parser ------------------------------------------------


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    except Exception:  # pragma: no cover - unparse is total in practice
        args, returns = "...", ""
    return f"{prefix} {node.name}({args}){returns}"


def _bases(node: ast.ClassDef) -> str:
    try:
        parts = [ast.unparse(b) for b in node.bases]
        parts += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords if kw.arg]
    except Exception:  # pragma: no cover
        return f"class {node.name}"
    return f"class {node.name}({', '.join(parts)})" if parts else f"class {node.name}"


def _decorators(node: ast.AST) -> tuple[str, ...]:
    raw = getattr(node, "decorator_list", ())
    out: list[str] = []
    for dec in raw:
        try:
            out.append(ast.unparse(dec))
        except Exception:  # pragma: no cover
            continue
    return tuple(out)


def _assigned(target: ast.expr) -> list[tuple[str, int]]:
    """The plain names bound by one assignment target, tuples unpacked."""
    if isinstance(target, ast.Name):
        return [(target.id, target.lineno)]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[tuple[str, int]] = []
        for element in target.elts:
            out += _assigned(element)
        return out
    return []


def _join(parent: str, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _python_body(
    body: Iterable[ast.stmt],
    parent: str,
    symbols: list[Symbol],
    imports: list[Import],
    *,
    top: bool,
) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # `node.lineno` is the `def` line, not the first decorator: a
            # decorated function must report where its body starts or every
            # snippet built from it is offset by the decorator count.
            symbols.append(
                Symbol(
                    node.name,
                    "method" if parent else "function",
                    node.lineno,
                    parent,
                    _signature(node),
                    _decorators(node),
                )
            )
            # Descend: a class defined inside a function is still a definition
            # somebody will search for.  `top` goes False so its locals are not
            # mistaken for module-level names.
            _python_body(node.body, _join(parent, node.name), symbols, imports, top=False)
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                Symbol(node.name, "class", node.lineno, parent, _bases(node), _decorators(node))
            )
            _python_body(node.body, _join(parent, node.name), symbols, imports, top=False)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(Import(alias.name, "", alias.asname or "", node.lineno, 0))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    Import(node.module or "", alias.name, alias.asname or "", node.lineno, node.level)
                )
        elif top and isinstance(node, ast.Assign):
            for target in node.targets:
                for name, line in _assigned(target):
                    symbols.append(Symbol(name, _assign_kind(name), line, parent))
        elif top and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(Symbol(node.target.id, _assign_kind(node.target.id), node.lineno, parent))
        elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            # `if TYPE_CHECKING:` guards real imports and `if sys.platform`
            # guards real module-level constants.  Both are still declarations.
            _python_body(node.body, parent, symbols, imports, top=top)
            _python_body(getattr(node, "orelse", ()), parent, symbols, imports, top=top)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody):
                _python_body(block, parent, symbols, imports, top=top)
            for handler in node.handlers:
                _python_body(handler.body, parent, symbols, imports, top=top)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                _python_body(case.body, parent, symbols, imports, top=top)


def _assign_kind(name: str) -> str:
    return "constant" if name.isupper() and any(c.isalpha() for c in name) else "variable"


def python_extraction(text: str) -> Extraction:
    """Parse Python with `ast`.  A syntax error becomes `Extraction.error`."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        # A half-written file is the normal state of a file being edited, so
        # this is a recorded fact, not a failure of the index.
        return Extraction("python", exact=True, error=f"{type(exc).__name__}: {exc}")
    symbols: list[Symbol] = []
    imports: list[Import] = []
    _python_body(tree.body, "", symbols, imports, top=True)
    return Extraction("python", tuple(symbols), tuple(imports), exact=True)


# -- everything else: a labelled heuristic ----------------------------------

_JS_DECLS: Final = (
    ("function", r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)"),
    ("class", r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)"),
    ("type", r"(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)"),
    (
        "function",
        r"(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
        r"(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)",
    ),
    ("variable", r"(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*[:=]"),
)

_GO_DECLS: Final = (
    ("method", r"func\s+\([^)]*\)\s*(?P<name>\w+)"),
    ("function", r"func\s+(?P<name>\w+)"),
    ("type", r"type\s+(?P<name>\w+)"),
    ("variable", r"(?:var|const)\s+(?P<name>\w+)"),
)

_RUST_DECLS: Final = (
    ("function", r"(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(?P<name>\w+)"),
    ("type", r"(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union|type)\s+(?P<name>\w+)"),
    ("constant", r"(?:pub(?:\([^)]*\))?\s+)?(?:static|const)\s+(?:mut\s+)?(?P<name>\w+)\s*:"),
    ("impl", r"impl(?:<[^>]*>)?\s+(?:(?:\w|::|<|>|,|\s)+\s+for\s+)?(?P<name>\w+)"),
    ("macro", r"macro_rules!\s*(?P<name>\w+)"),
)

_JAVA_DECLS: Final = (
    ("class", r"(?:(?:public|private|protected|abstract|final|static|sealed|strictfp)\s+)*(?:class|interface|enum|record)\s+(?P<name>\w+)"),
    (
        "method",
        r"(?:(?:public|private|protected|abstract|final|static|synchronized|native|default)\s+)+"
        r"(?:<[^>]+>\s*)?[\w.$<>\[\],\s]+?\s+(?P<name>\w+)\s*\(",
    ),
)

_C_DECLS: Final = (
    ("macro", r"#\s*define\s+(?P<name>\w+)"),
    ("type", r"(?:typedef\s+)?(?:struct|union|enum|class)\s+(?P<name>\w+)\s*[{:;]"),
    (
        "function",
        r"(?:[A-Za-z_]\w*(?:\s*(?:\*|&|::)\s*|\s+))+(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:const\s*)?\{",
    ),
)

#: Line-anchored declaration rules per language.  Anchored with `^\s*` so an
#: indented Java method is found but a `foo(` deep inside an expression is not.
_DECLARATIONS: Final[dict[str, tuple[tuple[str, re.Pattern[str]], ...]]] = {
    lang: tuple((kind, re.compile(r"^\s*" + body)) for kind, body in rules)
    for lang, rules in (
        ("javascript", _JS_DECLS),
        ("typescript", _JS_DECLS),
        ("go", _GO_DECLS),
        ("rust", _RUST_DECLS),
        ("java", _JAVA_DECLS),
        ("c", _C_DECLS),
        ("cpp", _C_DECLS),
    )
}

#: Import rules.  Searched rather than matched, because `export {a} from "b"`
#: and `const x = require("y")` both put the module in the middle of the line.
_IMPORT_RULES: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    lang: tuple(re.compile(p) for p in pats)
    for lang, pats in (
        (
            "javascript",
            (
                r"""(?:^|\s)from\s+['"](?P<module>[^'"]+)['"]""",
                r"""(?:^|[^\w.])import\s*\(\s*['"](?P<module>[^'"]+)['"]""",
                r"""(?:^|[^\w.])import\s+['"](?P<module>[^'"]+)['"]""",
                r"""(?:^|[^\w.])require\s*\(\s*['"](?P<module>[^'"]+)['"]""",
            ),
        ),
        ("go", (r'^\s*import\s+(?:[\w.]+\s+)?"(?P<module>[^"]+)"',)),
        ("rust", (r"^\s*(?:pub\s+)?use\s+(?P<module>[^;{]+)",)),
        ("java", (r"^\s*import\s+(?:static\s+)?(?P<module>[\w.$*]+)\s*;",)),
        ("c", (r'^\s*#\s*include\s*[<"](?P<module>[^>"]+)[>"]',)),
    )
}
_IMPORT_RULES["typescript"] = _IMPORT_RULES["javascript"]
_IMPORT_RULES["cpp"] = _IMPORT_RULES["c"]

#: Words that the Java and C function rules would otherwise report as
#: declarations, because `if (x) {` has exactly the shape of a call.
_NOT_A_NAME: Final[frozenset[str]] = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "else", "do", "try", "sizeof", "new", "case"}
)


def heuristic_extraction(language: str, text: str) -> Extraction:
    """Regular-expression declaration scan.  Approximate, and says so.

    This is not a parser.  It reads one line at a time, so a declaration split
    across lines is found only if its name is on the first of them, and a
    `//` inside a string literal will still be treated as a comment.
    """
    rules = _DECLARATIONS.get(language, ())
    import_rules = _IMPORT_RULES.get(language, ())
    symbols: list[Symbol] = []
    imports: list[Import] = []
    seen: set[tuple[str, int]] = set()
    in_go_block = False
    block_string = re.compile(r'"(?P<module>[^"]+)"')

    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "*/")):
            continue
        if language == "go":
            if stripped.startswith("import ("):
                in_go_block = True
                continue
            if in_go_block:
                if stripped.startswith(")"):
                    in_go_block = False
                    continue
                found = block_string.search(stripped)
                if found:
                    imports.append(Import(found.group("module"), line=number))
                continue
        for pattern in import_rules:
            hit = pattern.search(line)
            if hit:
                module = hit.group("module").strip()
                if language == "rust":
                    module = module.split(" as ")[0].strip()
                if module:
                    imports.append(Import(module, line=number))
                break
        for kind, pattern in rules:
            hit = pattern.match(line)
            if not hit:
                continue
            name = hit.group("name")
            if name in _NOT_A_NAME or (name, number) in seen:
                break
            seen.add((name, number))
            symbols.append(Symbol(name, kind, number, "", stripped[:160]))
            break

    return Extraction(language, tuple(symbols), tuple(imports), exact=False)


def extract(path: str | Path, text: str) -> Extraction:
    """Symbols and imports for one file, dispatched on its extension."""
    language = language_of(path)
    if language == "python":
        return python_extraction(text)
    if language in _DECLARATIONS:
        return heuristic_extraction(language, text)
    return Extraction(language)


# -- module resolution ------------------------------------------------------


def _module_table(files: Mapping[str, Extraction]) -> tuple[dict[str, str], dict[str, str]]:
    """Two lookup tables: dotted/path keys, and unambiguous basenames.

    The basename table is the fallback for `#include "queue.h"`, which names a
    file and not a path.  It only holds names owned by exactly one file — two
    `utils.py` in one tree mean neither is resolvable that way, which is the
    correct answer rather than a coin toss.
    """
    keys: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    for path in sorted(files):
        extraction = files[path]
        pure = PurePosixPath(path)
        if extraction.language == "python":
            parts = list(pure.parent.parts)
            if pure.name not in ("__init__.py", "__init__.pyi"):
                parts.append(pure.stem)
            dotted = ".".join(parts)
            if dotted:
                keys.setdefault(dotted, path)
        keys.setdefault(path, path)
        keys.setdefault(str(pure.with_suffix("")), path)
        names.setdefault(pure.name, []).append(path)
    unique = {name: paths[0] for name, paths in names.items() if len(paths) == 1}
    return keys, unique


def _python_target(path: str, imp: Import) -> str:
    """The dotted module an import names, relative imports made absolute."""
    if not imp.level:
        return imp.module
    base = PurePosixPath(path).parent
    # level 1 is the package containing this file, so only the levels above
    # that walk upwards.
    for _ in range(imp.level - 1):
        base = base.parent
    parts = [p for p in base.parts if p not in (".", "")]
    if imp.module:
        parts += imp.module.split(".")
    return ".".join(parts)


def _candidates(path: str, imp: Import, language: str) -> list[str]:
    """Keys to try in the module table, most specific first."""
    if language == "python":
        dotted = _python_target(path, imp)
        if not dotted:
            return []
        # `from pkg import mod` names a module; `from pkg import thing` names a
        # member of one.  Try the longer reading first.
        return [f"{dotted}.{imp.name}", dotted] if imp.name else [dotted]

    module = imp.module
    if language in ("javascript", "typescript"):
        if not module.startswith("."):
            return []  # a bare specifier is a package, not a file in this tree
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(path), module))
        out = [joined, *(joined + suffix for suffix in _JS_SUFFIXES)]
        out += [posixpath.join(joined, "index" + suffix) for suffix in _JS_SUFFIXES]
        return out
    if language == "rust":
        parts = [p for p in module.replace("::", "/").split("/") if p]
        while parts and parts[0] in ("crate", "self", "super"):
            parts.pop(0)
        if not parts:
            return []
        stem = "/".join(parts)
        # `use a::b::Thing` is more often module `a::b` than module `a::b::Thing`.
        return [stem, f"src/{stem}", "/".join(parts[:-1]), f"src/{'/'.join(parts[:-1])}"]
    if language == "java":
        parts = [p for p in module.split(".") if p and p != "*"]
        if not parts:
            return []
        stem = "/".join(parts)
        return [stem, f"src/main/java/{stem}", f"src/{stem}"]
    if language in ("c", "cpp", "go"):
        return [module, posixpath.normpath(posixpath.join(posixpath.dirname(path), module))]
    return [module]


# -- the graph --------------------------------------------------------------

#: How many files `references` will read before giving up.  A word scan of the
#: whole tree is affordable; a word scan of the whole tree per keystroke is not.
REFERENCE_FILES: Final = 400


class SymbolGraph:
    """Definitions, references and the two directions of the import edge.

    Built from a mapping of workspace-relative path to `Extraction`.  `root`
    lets it read files back off disk for reference scanning; without one,
    `references` returns nothing rather than pretending.
    """

    __slots__ = ("_files", "_root", "_shortlist", "_defs", "_keys", "_names", "_out", "_in")

    def __init__(
        self,
        files: Mapping[str, Extraction],
        *,
        root: str | Path | None = None,
        shortlist: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self._files: dict[str, Extraction] = dict(files)
        self._root = Path(root) if root is not None else None
        #: Narrows a reference scan to the files that can contain the word.
        #: The index supplies one from its postings; without it, every file.
        self._shortlist = shortlist
        self._defs: dict[str, list[Location]] = {}
        for path in sorted(self._files):
            for symbol in self._files[path].symbols:
                self._defs.setdefault(symbol.name, []).append(
                    Location(path, symbol.line, symbol.kind, symbol.name, symbol.signature)
                )
        self._keys, self._names = _module_table(self._files)
        self._out: dict[str, tuple[str, ...]] = {}
        self._in: dict[str, list[str]] = {}
        for path in sorted(self._files):
            targets: list[str] = []
            for imp in self._files[path].imports:
                hit = self.resolve(path, imp)
                if hit is not None and hit != path and hit not in targets:
                    targets.append(hit)
            self._out[path] = tuple(targets)
            for target in targets:
                self._in.setdefault(target, []).append(path)

    # -- resolution ---------------------------------------------------------

    def resolve(self, path: str, imp: Import) -> str | None:
        """The workspace file an import points at, or None if it leaves the tree."""
        language = self._files[path].language if path in self._files else language_of(path)
        for key in _candidates(path, imp, language):
            if key and key in self._keys:
                return self._keys[key]
        if language in ("c", "cpp"):
            # `#include "queue.h"` names a file, not a path.
            return self._names.get(posixpath.basename(imp.module))
        return None

    # -- queries ------------------------------------------------------------

    def defines(self, symbol: str) -> list[Location]:
        """Every declaration of `symbol`, ordered by path then line."""
        return sorted(self._defs.get(symbol, ()), key=lambda loc: (loc.path, loc.line))

    def references(self, symbol: str, *, limit: int = 200) -> list[Location]:
        """Word-boundary occurrences of `symbol` that are not its definition.

        Textual, not resolved: a local variable of the same name in an
        unrelated file will appear here.  Callers presenting this to a human
        should describe it as "mentions", not "callers".
        """
        if self._root is None or not symbol or not symbol.isidentifier():
            return []
        word = re.compile(rf"\b{re.escape(symbol)}\b")
        declared = {(loc.path, loc.line) for loc in self._defs.get(symbol, ())}
        paths = list(self._shortlist(symbol)) if self._shortlist is not None else sorted(self._files)
        out: list[Location] = []
        for path in paths[:REFERENCE_FILES]:
            try:
                text = (self._root / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if (path, number) in declared or not word.search(line):
                    continue
                out.append(Location(path, number, "reference", symbol, line.strip()[:200]))
                if len(out) >= limit:
                    return out
        return out

    def imports_of(self, path: str) -> list[str]:
        """The modules `path` imports, as the source writes them.

        Deliberately *not* the inverse of `importers_of`: this is what the file
        says, including third-party modules that have no file in this
        workspace.  For the resolved in-workspace edge use `dependencies_of`.
        """
        extraction = self._files.get(path)
        if extraction is None:
            return []
        seen: dict[str, None] = {}
        for imp in extraction.imports:
            seen.setdefault(imp.dotted, None)
        return list(seen)

    def dependencies_of(self, path: str) -> list[str]:
        """The workspace files `path` imports.  Forward edge, resolved."""
        return list(self._out.get(path, ()))

    def importers_of(self, path: str) -> list[str]:
        """The workspace files that import `path`.  The reverse edge."""
        return sorted(self._in.get(path, ()))

    def neighbourhood(self, path: str, depth: int = 1) -> list[str]:
        """Files within `depth` import hops of `path`, in both directions.

        Ordered nearest first, then by path, so a caller taking the first N
        gets the closest N and a replay gets the same list.
        """
        if depth < 1 or path not in self._files:
            return []
        seen = {path}
        frontier = [path]
        out: list[str] = []
        for _ in range(depth):
            nxt: list[str] = []
            for current in frontier:
                for other in sorted({*self._out.get(current, ()), *self._in.get(current, ())}):
                    if other in seen:
                        continue
                    seen.add(other)
                    out.append(other)
                    nxt.append(other)
            if not nxt:
                break
            frontier = nxt
        return out

    def distance(self, path: str, depth: int = 2) -> dict[str, int]:
        """Hop count from `path` to every file within `depth`.  `path` is 0."""
        out = {path: 0}
        frontier = [path]
        for hop in range(1, depth + 1):
            nxt: list[str] = []
            for current in frontier:
                for other in sorted({*self._out.get(current, ()), *self._in.get(current, ())}):
                    if other in out:
                        continue
                    out[other] = hop
                    nxt.append(other)
            if not nxt:
                break
            frontier = nxt
        return out

    # -- rendering ----------------------------------------------------------

    def report(self, path: str) -> list[str]:
        """Human-readable dependency summary for one file."""
        extraction = self._files.get(path)
        if extraction is None:
            return [f"{path}: not indexed"]
        kind = "parsed" if extraction.exact else "heuristic"
        lines = [f"{path}  ({extraction.language}, {kind}, {len(extraction.symbols)} symbols)"]
        if extraction.error:
            lines.append(f"  parse error: {extraction.error}")
        deps = self.dependencies_of(path)
        users = self.importers_of(path)
        lines.append(f"  imports {len(extraction.imports)} modules, {len(deps)} of them in this tree")
        lines += [f"    -> {d}" for d in deps[:12]]
        lines.append(f"  imported by {len(users)}")
        lines += [f"    <- {u}" for u in users[:12]]
        return lines
