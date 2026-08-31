"""An incremental index of the workspace, in SQLite.

Three decisions.

**SQLite, not JSON.** It is stdlib, it gives transactions so a crash mid-refresh
cannot leave a half-written index, and it can answer a postings query without
loading every record into memory. A JSON blob would have to be read and written
whole on every change, which for a large repository is the difference between a
refresh you can run on every query and one you cannot.

**Content hash, not mtime.** mtime alone re-parses a file that a checkout
touched without changing, and misses a change made inside the same second. Size
and mtime are the cheap pre-filter; the hash is what decides. That is what makes
`refresh()` cheap enough to call before answering, so the index is never stale
in a way the user has to think about.

**Postings, not just documents.** The table of which word appears in which file
is what lets `SymbolGraph` narrow a reference scan from every file to the few
that can contain the word, and what makes BM25 affordable. It costs disk and
buys the two features that matter.

The index is deliberately unaware of ranking; `tools/retrieve` owns that.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterator, Sequence

from offset.core.symbols import Extraction, extract, language_of

#: Directories never worth indexing.  Mirrors `speculate.SKIP` and adds the
#: build output of the ecosystems we can parse.
SKIP: Final[frozenset[str]] = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".offset", "dist", "build", "target", ".next", ".nuxt", ".tox",
    "site-packages", ".gradle", ".idea", ".vscode", "coverage",
    ".terraform", "vendor", ".cargo", ".eggs",
})

#: Files larger than this are recorded but not parsed.  A megabyte of source is
#: generated, minified or vendored; indexing it costs more than it returns.
MAX_BYTES: Final = 1_000_000

#: Extensions worth reading at all.  Anything else is recorded as a path so the
#: tree is complete, without being opened.
TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".rb", ".php", ".cs", ".swift", ".scala", ".sh", ".bash", ".zsh",
    ".sql", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".ini", ".cfg", ".html", ".css", ".scss", ".vue", ".svelte", ".lua",
})

SCHEMA_VERSION: Final = 1

#: Words shorter than this are not worth a posting: they match everything.
MIN_WORD: Final = 2

#: A word appearing in more than this share of files carries no signal and is
#: dropped from postings at query time rather than at write time, so the
#: threshold can change without a reindex.
COMMON_SHARE: Final = 0.6

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def split_identifier(word: str) -> list[str]:
    """`getUserAuth` -> `get user auth`; `HTTP_PORT` -> `http port`.

    This is the whole of the "semantic" story and it is worth being plain about
    it: splitting identifiers and matching the pieces is lexical expansion, not
    embeddings. It makes `getUserAuth` find `get_user_authentication`, which is
    the case that actually comes up, and it costs nothing at query time.
    """
    parts: list[str] = []
    for chunk in word.split("_"):
        if not chunk:
            continue
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(chunk))
    return [p for p in parts if len(p) >= MIN_WORD]


def words_of(text: str) -> Iterator[str]:
    """Every identifier and its pieces, lowercased."""
    for match in _WORD.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if len(lowered) >= MIN_WORD:
            yield lowered
        pieces = split_identifier(word)
        if len(pieces) > 1:
            yield from pieces


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    language: str
    size: int
    mtime: float
    digest: str
    lines: int
    parsed: bool
    error: str = ""


@dataclass(slots=True)
class IndexStats:
    """What a refresh did.  `parsed` is the number that matters: on a warm
    index it should be zero, which is the property the tests assert."""

    scanned: int = 0
    parsed: int = 0
    skipped: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def report(self) -> list[str]:
        lines = [
            (
                f"scanned {self.scanned}, parsed {self.parsed}, "
                f"unchanged {max(0, self.scanned - self.parsed)}, "
                f"removed {self.removed} in {self.seconds:.2f}s"
            )
        ]
        lines.extend(f"  {e}" for e in self.errors[:10])
        return lines


@dataclass(frozen=True, slots=True)
class Hit:
    path: str
    score: float
    reason: str = ""


def _gitignore_matcher(root: Path) -> Callable[[str], bool]:
    """A matcher for the common `.gitignore` forms.

    Deliberately partial and documented as such: negation, nested ignore files
    and full pathspec semantics are not implemented. It handles directory
    names, `*.ext` globs and rooted paths, which is what excludes the noise.
    """
    patterns: list[tuple[str, bool]] = []
    path = root / ".gitignore"
    if path.exists():
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith(("#", "!")):
                    continue
                rooted = line.startswith("/")
                patterns.append((line.strip("/"), rooted))
        except OSError:
            pass

    if not patterns:
        return lambda _relative: False

    from fnmatch import fnmatch

    def ignored(relative: str) -> bool:
        parts = relative.split("/")
        for pattern, rooted in patterns:
            if rooted:
                if fnmatch(relative, pattern) or relative.startswith(pattern + "/"):
                    return True
                continue
            if fnmatch(relative, pattern):
                return True
            if any(fnmatch(part, pattern) for part in parts):
                return True
        return False

    return ignored


class Index:
    """The workspace index.  One SQLite file under `.offset/index`."""

    __slots__ = ("_conn", "_ignored", "root")

    def __init__(self, root: Path | str, *, path: Path | str | None = None) -> None:
        self.root = Path(root).resolve()
        target = Path(path) if path is not None else self.root / ".offset" / "index" / "index.db"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(target), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ignored: Callable[[str], bool] | None = None
        self._prepare()

    # -- schema -------------------------------------------------------------

    def _prepare(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        row = cur.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        version = int(row["value"]) if row else 0
        if version and version != SCHEMA_VERSION:
            # A schema change is not worth a migration for a derived artefact:
            # the cheapest correct answer is to drop it and re-index.
            for table in ("postings", "symbols", "imports", "files"):
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            version = 0
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY, language TEXT NOT NULL DEFAULT '',
                size INTEGER NOT NULL DEFAULT 0, mtime REAL NOT NULL DEFAULT 0,
                digest TEXT NOT NULL DEFAULT '', lines INTEGER NOT NULL DEFAULT 0,
                parsed INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
                terms INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS symbols (
                path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL DEFAULT '',
                line INTEGER NOT NULL DEFAULT 0, parent TEXT NOT NULL DEFAULT '',
                signature TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
            CREATE TABLE IF NOT EXISTS imports (
                path TEXT NOT NULL, module TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '', alias TEXT NOT NULL DEFAULT '',
                line INTEGER NOT NULL DEFAULT 0, level INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS imports_path ON imports(path);
            CREATE TABLE IF NOT EXISTS postings (
                word TEXT NOT NULL, path TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS postings_word ON postings(word);
            CREATE INDEX IF NOT EXISTS postings_path ON postings(path);
            """
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('version', ?)", (str(SCHEMA_VERSION),)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- walking ------------------------------------------------------------

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def walk(self) -> Iterator[Path]:
        """Every candidate file, skipping the usual noise and `.gitignore`."""
        if self._ignored is None:
            self._ignored = _gitignore_matcher(self.root)
        stack = [self.root]
        while stack:
            here = stack.pop()
            try:
                entries = sorted(here.iterdir())
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                if name in SKIP or name.startswith(".") and name not in (".github",):
                    if entry.is_dir():
                        continue
                    if name.startswith("."):
                        continue
                relative = self._relative(entry)
                if self._ignored(relative):
                    continue
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if name not in SKIP:
                        stack.append(entry)
                    continue
                if entry.suffix.lower() in TEXT_SUFFIXES:
                    yield entry

    # -- refresh ------------------------------------------------------------

    def refresh(self, *, cancel: Callable[[], bool] | None = None) -> IndexStats:
        """Bring the index up to date, parsing only what changed."""
        started = time.monotonic()
        stats = IndexStats()
        cur = self._conn.cursor()
        known = {
            row["path"]: (row["size"], row["mtime"], row["digest"])
            for row in cur.execute("SELECT path, size, mtime, digest FROM files")
        }
        seen: set[str] = set()

        for path in self.walk():
            if cancel is not None and cancel():
                break
            relative = self._relative(path)
            seen.add(relative)
            stats.scanned += 1
            try:
                info = path.stat()
            except OSError as exc:
                stats.errors.append(f"{relative}: {exc}")
                continue
            if info.st_size > MAX_BYTES:
                stats.skipped += 1
                self._record_shell(relative, info, "too large to parse")
                continue

            previous = known.get(relative)
            if previous is not None and previous[0] == info.st_size and abs(previous[1] - info.st_mtime) < 1e-6:
                continue  # cheap pre-filter: nothing to do

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                stats.errors.append(f"{relative}: {exc}")
                continue
            digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if previous is not None and previous[2] == digest:
                # Touched but unchanged: refresh the stat so the pre-filter
                # catches it next time, and do not re-parse.
                cur.execute(
                    "UPDATE files SET size=?, mtime=? WHERE path=?",
                    (info.st_size, info.st_mtime, relative),
                )
                continue

            self._store(relative, text, info, digest)
            stats.parsed += 1

        gone = set(known) - seen
        for relative in gone:
            self._forget(relative)
            stats.removed += 1

        self._conn.commit()
        stats.seconds = time.monotonic() - started
        return stats

    def _record_shell(self, relative: str, info: Any, error: str) -> None:
        """Record a file we deliberately did not parse."""
        self._conn.execute(
            "INSERT OR REPLACE INTO files(path, language, size, mtime, digest, lines, parsed, error, terms)"
            " VALUES(?,?,?,?,?,?,0,?,0)",
            (relative, language_of(relative), info.st_size, info.st_mtime, "", 0, error),
        )

    def _forget(self, relative: str) -> None:
        for table in ("postings", "symbols", "imports", "files"):
            self._conn.execute(f"DELETE FROM {table} WHERE path=?", (relative,))

    def _store(self, relative: str, text: str, info: Any, digest: str) -> None:
        self._forget(relative)
        found = extract(relative, text)
        counts: dict[str, int] = {}
        for word in words_of(text):
            counts[word] = counts.get(word, 0) + 1
        lines = text.count("\n") + 1

        self._conn.execute(
            "INSERT INTO files(path, language, size, mtime, digest, lines, parsed, error, terms)"
            " VALUES(?,?,?,?,?,?,1,?,?)",
            (relative, found.language, info.st_size, info.st_mtime, digest, lines,
             found.error, sum(counts.values())),
        )
        if found.symbols:
            self._conn.executemany(
                "INSERT INTO symbols(path, name, kind, line, parent, signature) VALUES(?,?,?,?,?,?)",
                [(relative, s.name, s.kind, s.line, s.parent, s.signature) for s in found.symbols],
            )
        if found.imports:
            self._conn.executemany(
                "INSERT INTO imports(path, module, name, alias, line, level) VALUES(?,?,?,?,?,?)",
                [(relative, i.module, i.name, i.alias, i.line, i.level)
                 for i in found.imports],
            )
        if counts:
            self._conn.executemany(
                "INSERT INTO postings(word, path, count) VALUES(?,?,?)",
                [(word, relative, n) for word, n in counts.items()],
            )

    # -- reading ------------------------------------------------------------

    def file(self, path: str) -> FileRecord | None:
        row = self._conn.execute(
            "SELECT path, language, size, mtime, digest, lines, parsed, error FROM files WHERE path=?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        return FileRecord(
            path=row["path"], language=row["language"], size=row["size"], mtime=row["mtime"],
            digest=row["digest"], lines=row["lines"], parsed=bool(row["parsed"]), error=row["error"],
        )

    def files(self) -> list[str]:
        return [r["path"] for r in self._conn.execute("SELECT path FROM files ORDER BY path")]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()
        return int(row["n"]) if row else 0

    def text_of(self, path: str) -> str:
        """Read a file's current contents from disk.

        The index stores no bodies on purpose: they would double the disk cost
        and go stale the moment anything writes, and the file is right there.
        """
        try:
            return (self.root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def postings(self, word: str) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT path, count FROM postings WHERE word=?", (word.lower(),)
        ).fetchall()
        return [(r["path"], r["count"]) for r in rows]

    def shortlist(self, word: str) -> list[str]:
        """Files that contain a word.  This is `SymbolGraph`'s narrowing hook."""
        return [path for path, _ in self.postings(word)]

    def lengths(self) -> dict[str, int]:
        return {
            r["path"]: int(r["terms"])
            for r in self._conn.execute("SELECT path, terms FROM files WHERE parsed=1")
        }

    def document_frequency(self, words: Sequence[str]) -> dict[str, int]:
        """How many files contain each word, in one query."""
        if not words:
            return {}
        marks = ",".join("?" * len(words))
        rows = self._conn.execute(
            f"SELECT word, COUNT(DISTINCT path) AS n FROM postings WHERE word IN ({marks}) GROUP BY word",
            [w.lower() for w in words],
        ).fetchall()
        return {r["word"]: int(r["n"]) for r in rows}

    def symbols_named(self, name: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT path, name, kind, line, parent, signature FROM symbols WHERE name=? ORDER BY path",
            (name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def symbols_like(self, fragment: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT path, name, kind, line, parent, signature FROM symbols"
            " WHERE name LIKE ? ORDER BY LENGTH(name), name LIMIT ?",
            (f"%{fragment}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def extractions(self) -> dict[str, Extraction]:
        """Rebuild the per-file extractions the symbol graph needs.

        Read back from the index rather than re-parsing: on a warm index this
        is the difference between a graph that is free and one that costs a
        full re-parse of the tree.
        """
        from offset.core.symbols import Import, Symbol

        by_path: dict[str, dict[str, Any]] = {}
        for row in self._conn.execute("SELECT path, language FROM files WHERE parsed=1"):
            by_path[row["path"]] = {"language": row["language"], "symbols": [], "imports": []}
        for row in self._conn.execute(
            "SELECT path, name, kind, line, parent, signature FROM symbols"
        ):
            entry = by_path.get(row["path"])
            if entry is not None:
                entry["symbols"].append(
                    Symbol(row["name"], row["kind"], row["line"], row["parent"], row["signature"])
                )
        for row in self._conn.execute(
            "SELECT path, module, name, alias, line, level FROM imports"
        ):
            entry = by_path.get(row["path"])
            if entry is not None:
                entry["imports"].append(
                    Import(row["module"], row["name"], row["alias"], row["line"], row["level"])
                )
        return {
            path: Extraction(
                language=data["language"],
                symbols=tuple(data["symbols"]),
                imports=tuple(data["imports"]),
                exact=data["language"] == "python",
            )
            for path, data in by_path.items()
        }

    def graph(self) -> Any:
        """A `SymbolGraph` over the index, narrowed by our postings."""
        from offset.core.symbols import SymbolGraph

        return SymbolGraph(self.extractions(), root=self.root, shortlist=self.shortlist)

    def report(self) -> list[str]:
        total = self.count()
        parsed = self._conn.execute("SELECT COUNT(*) AS n FROM files WHERE parsed=1").fetchone()["n"]
        symbols = self._conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]
        words = self._conn.execute("SELECT COUNT(DISTINCT word) AS n FROM postings").fetchone()["n"]
        return [
            f"{total} file(s) indexed, {parsed} parsed",
            f"{symbols} symbol(s), {words} distinct word(s)",
            f"root: {self.root}",
        ]


#: One index per root, so a tool call does not reopen the database every time.
_OPEN: dict[str, Index] = {}


def open_index(root: Path | str) -> Index:
    key = str(Path(root).resolve())
    existing = _OPEN.get(key)
    if existing is not None:
        return existing
    made = Index(key)
    _OPEN[key] = made
    return made


def close_all() -> None:
    for index in list(_OPEN.values()):
        try:
            index.close()
        except sqlite3.Error:
            continue
    _OPEN.clear()
