"""Whether the code is safe, not whether it works.

`offset` already has a `critic` seat that judges correctness. Nothing asked the
other question, and the two are genuinely different: a change can pass every
test and still hand an attacker a shell. `ROLES` gained a `security` seat for
this module to fill.

Three decisions shape everything below, and all three come from the same
observation: **a scanner that cries wolf gets switched off**, and a switched-off
scanner is worth less than none at all because it also buys false comfort.

**Every finding carries checkable evidence.** A file, a line number, the
offending source, and a sentence saying what an attacker *gets*. "S105
hardcoded password" tells you a rule fired. "this AWS key is live in the repo
and grants whatever the account grants" tells you why to care. Rules that cannot
say the second thing are not worth having.

**False positives are treated as bugs, not noise.** Each rule has a test that
it fires on real vulnerable code *and* a test that it stays silent on a safe
near-miss - `password` as a dict key, a fixture in a test file, an example in a
docstring. Those three are where naive scanners lose their users.

**There is always an escape hatch.** A trailing `# noqa: security` and a
`.offset-security-ignore` glob file. Without one, the first false positive on a
Friday afternoon gets the whole thing disabled permanently.

The model pass is additive and never authoritative: a reply that is prose,
empty, or malformed degrades to the static findings, never to a crash and never
to an invented finding with a plausible-looking line number.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from offset.core.scoring import Criterion

#: Severities, worst first.  Three levels, not five: nobody agrees on the
#: difference between "moderate" and "medium", and a scale people argue about
#: gets ignored.
HIGH: Final = "high"
MEDIUM: Final = "medium"
LOW: Final = "low"
SEVERITIES: Final = (HIGH, MEDIUM, LOW)

Severity = str

#: Suppression comment.  Deliberately explicit - a bare `# noqa` should not
#: silence a security finding as a side effect of silencing a lint.
SUPPRESS: Final = re.compile(r"#\s*noqa:\s*security\b", re.IGNORECASE)

#: Glob patterns, one per line, `#` comments allowed.
IGNORE_NAME: Final = ".offset-security-ignore"

#: Files we never scan: caches, vendored trees and version control.  Scanning
#: a vendored dependency produces findings nobody in this repo can fix.
SKIP_DIRS: Final = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build", ".tox",
    "site-packages", ".eggs",
})

#: Extensions worth reading.  A binary blob has no line numbers to report.
SOURCE_SUFFIXES: Final = frozenset({
    ".py", ".pyi", ".sh", ".bash", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".rb", ".go", ".rs", ".java", ".php", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".env", ".conf", ".json",
})

#: Refuse to read anything larger.  A minified bundle is not source.
MAX_FILE_BYTES: Final = 512 * 1024

#: Longest evidence line kept.  A minified line would otherwise fill the pane.
CLIP: Final = 160


@dataclass(slots=True, frozen=True)
class Finding:
    """One thing worth fixing, with the evidence to check it."""

    rule: str
    severity: Severity
    path: str
    line: int
    evidence: str
    impact: str
    source: str = "static"      # "static" or "model"

    def key(self) -> tuple[str, int, str]:
        """Identity for de-duplication.

        Path, line and rule - *not* the impact sentence, which the model
        phrases differently every time and would defeat the merge.
        """
        return (self.path, self.line, self.rule)

    def line_text(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.severity:<6}] {where}  {self.rule}"


# -- rules ------------------------------------------------------------------------


@dataclass(slots=True)
class Rule:
    """A pattern, and what it means if it matches.

    `refute` is what keeps the rule usable: a second pattern that, when it also
    matches the line, means this is the safe form.  Encoding the near-miss
    beside the rule is what stops the false positives being spread across
    special cases elsewhere.
    """

    name: str
    severity: Severity
    pattern: re.Pattern[str]
    impact: str
    refute: re.Pattern[str] | None = None
    suffixes: frozenset[str] | None = None
    #: Whether this rule matches a *value* rather than *code*.
    #:
    #: The distinction decides where a match counts.  A call is only dangerous
    #: where it runs, so a code rule must ignore string literals and comments -
    #: a sample in a docstring is documentation and a rule's own pattern is not
    #: a call to the thing it matches.  A value is dangerous wherever it
    #: appears: a committed AWS key lives inside a string by definition, and
    #: commenting out the line that holds it does not un-leak it.
    #:
    #: Both halves of that were regressions found by running the scanner over
    #: this repository: exempting strings silenced the key rules entirely, and
    #: not exempting comments made this file report itself.
    literal_value: bool = False

    def applies_to(self, path: str) -> bool:
        if self.suffixes is None:
            return True
        return any(path.endswith(s) for s in self.suffixes)

    def hit(self, line: str) -> bool:
        if not self.pattern.search(line):
            return False
        return not (self.refute is not None and self.refute.search(line))


#: A quoted string that is not obviously a placeholder.  Used by the credential
#: rules so `password = ""`, `password = None` and `password = os.environ[...]`
#: do not fire - the last of those is the *fix*, and flagging it would teach
#: people the scanner is stupid.
_REAL_SECRET: Final = r"""=\s*["'][^"'\n]{8,}["']"""

#: Placeholder values people legitimately commit.
_PLACEHOLDER: Final = re.compile(
    r"""(?ix) ["'] (?:
        x{3,} | \.{3} | changeme | your[-_ ]? (?:key|token|secret|password)
        | replace[-_ ]?me | example | placeholder | dummy | fake | test
        | \$\{ [^}]* \} | \{\{ [^}]* \}\} | % [sd] | <[^>]+>
    ) """)

RULES: Final[tuple[Rule, ...]] = (
    Rule(
        "aws-access-key", HIGH,
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "this is a live AWS key id; with its secret it grants whatever the "
        "account grants, and rotating it is the only fix once committed",
        literal_value=True,
    ),
    Rule(
        "private-key", HIGH,
        re.compile(r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE KEY"),
        "a private key in the repository authenticates anyone who clones it",
        literal_value=True,
    ),
    Rule(
        "hardcoded-password", HIGH,
        re.compile(r"""(?ix) \b (?:password|passwd|pwd) \s* """ + _REAL_SECRET),
        "a credential in source is readable by everyone with repository access "
        "and survives in history after it is deleted",
        # `"password": value` is a dict *key*, and `password == "..."` is a
        # comparison.  Neither is a stored credential.  This one refutation
        # removes most of the noise this rule is famous for.
        refute=re.compile(r"""(?ix) (?: ["'] (?:password|passwd|pwd) ["'] \s* : )
                                    | (?: [=!]= ) | (?: getenv | environ | prompt | input )"""),
    ),
    Rule(
        "hardcoded-api-key", HIGH,
        re.compile(r"""(?ix) \b (?:api[-_]?key|secret[-_]?key|access[-_]?token
                                 |auth[-_]?token|bearer[-_]?token) \s* """ + _REAL_SECRET),
        "an API key in source is a credential anyone with the repository can "
        "spend, and it will be scraped if the repository ever goes public",
        refute=re.compile(r"""(?ix) (?: ["'] [a-z_]* (?:key|token) ["'] \s* : )
                                    | (?: [=!]= ) | (?: getenv | environ | keyring )"""),
    ),
    Rule(
        "shell-injection", HIGH,
        re.compile(r"""(?x) (?: subprocess\.(?:run|call|check_output|check_call|Popen)
                              | os\.(?:system|popen) ) [^\n]*?
                            (?: shell \s* = \s* True | \b os\.system \s* \( )"""),
        "an interpolated shell command lets any attacker-controlled substring "
        "run arbitrary commands; pass a list and drop shell=True",
        # A fully literal command with no interpolation cannot be injected.
        refute=re.compile(r"""(?x) (?: os\.system | \( ) \s* ["'][^"'{%]*["'] \s* \)"""),
    ),
    Rule(
        "eval-exec", HIGH,
        re.compile(r"""(?x) \b (?:eval|exec) \s* \( \s* (?! ["'] )"""),
        "evaluating a non-literal string executes whatever produced it, which "
        "is remote code execution if any of it came from outside",
        refute=re.compile(r"""(?x) \b (?:ast\.literal_)eval | \. (?:eval|exec) \s* \("""),
    ),
    Rule(
        "insecure-deserialise", HIGH,
        re.compile(r"""(?x) \b pickle \s* \. \s* loads? \s* \(
                          | \b (?:cPickle|dill|shelve) \s* \. \s* loads? \s* \("""),
        "unpickling attacker-controlled bytes runs arbitrary code by design; "
        "pickle is not a data format for untrusted input",
    ),
    Rule(
        "yaml-load", HIGH,
        re.compile(r"\byaml\s*\.\s*load\s*\("),
        "yaml.load without a safe loader constructs arbitrary Python objects "
        "from the document, which is remote code execution",
        refute=re.compile(r"""(?ix) safe_load | Loader \s* = \s* (?:yaml\.)? (?:Safe|C?Safe)"""),
    ),
    Rule(
        "sql-string-building", HIGH,
        re.compile(r"""(?ix) (?:execute|executemany|cursor\.execute|query) \s* \(
                            [^)\n]*?
                            (?: f["'] [^"'\n]* (?:select|insert|update|delete)
                              | ["'][^"'\n]*(?:select|insert|update|delete)[^"'\n]*["'] \s* (?:\+|%|\.format)
                              ) """),
        "a query assembled by concatenation is SQL injection; the parameter "
        "form exists precisely to make this impossible",
        # `?`/`%s` placeholders with a separate parameter tuple are the fix.
        refute=re.compile(r"""(?x) \? | %s | : [a-z_]+ \b .* , \s* [\(\[]"""),
    ),
    Rule(
        "tls-verification-off", HIGH,
        re.compile(r"""(?ix) verify \s* = \s* False
                            | CERT_NONE
                            | check_hostname \s* = \s* False
                            | ssl\._create_unverified_context"""),
        "with certificate verification off, anyone on the path can read and "
        "rewrite the traffic; TLS is doing nothing",
    ),
    Rule(
        "world-writable", MEDIUM,
        re.compile(r"""(?x) chmod \s* \( [^)\n]* 0o?[0-7]?[0-7][2367] \s* \)
                          | chmod \s+ [0-7]?[0-7][0-7][2367] \b"""),
        "a world-writable file lets any local user replace its contents, which "
        "for anything executable or imported is privilege escalation",
    ),
    Rule(
        "insecure-temp", MEDIUM,
        re.compile(r"\btempfile\s*\.\s*mktemp\s*\("),
        "mktemp returns a name, not a file, so another process can win the race "
        "and place a symlink there; use mkstemp or NamedTemporaryFile",
    ),
    Rule(
        "plaintext-http", LOW,
        re.compile(r"""(?ix) ["'] http:// (?! localhost | 127\.0\.0\.1 | \[::1\] | 0\.0\.0\.0 )
                             [a-z0-9.-]* [a-z] """),
        "credentials and payloads over plain HTTP are readable and modifiable "
        "in transit",
        # Three legitimate shapes that are not a plaintext request: a scheme
        # *check* naming both schemes, an XML namespace (an identifier that is
        # never fetched), and a DOCTYPE.  All three fired on this repository
        # the first time the rule ran over it.
        refute=re.compile(r"""(?ix) https:// | xmlns | schemas\. | w3\.org
                                    | purl\.org | DOCTYPE | namespace"""),
    ),
    Rule(
        "binding-all-interfaces", MEDIUM,
        re.compile(r"""(?ix) (?:bind|host) \s* [=\(] \s* ["'] (?: 0\.0\.0\.0 | :: ) ["']"""),
        "binding every interface exposes the service to the whole network, "
        "which for a development server usually was not intended",
    ),
)


# -- scanning ------------------------------------------------------------------------


def _ignore_patterns(root: Path) -> tuple[str, ...]:
    try:
        text = (root / IGNORE_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return tuple(out)


def _ignored(rel: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(Path(rel).name, p)
               for p in patterns)


def _is_fixture(path: str) -> bool:
    """Whether a credential-shaped literal here is expected rather than a leak.

    A test suite that exercises a credential parser *must* contain
    credential-shaped strings.  Reporting them trains people to ignore the
    scanner, so a test path downgrades the credential rules rather than firing
    them - it does not silence the injection or deserialisation rules, which
    are just as dangerous in a test as anywhere else.
    """
    parts = Path(path).parts
    name = Path(path).name
    return ("tests" in parts or "test" in parts or "fixtures" in parts
            or name.startswith("test_") or name.endswith("_test.py")
            or "conftest" in name)


CREDENTIAL_RULES: Final = frozenset({
    "hardcoded-password", "hardcoded-api-key", "aws-access-key", "private-key",
})


def _docstring_spans(text: str) -> set[int]:
    """Line numbers inside triple-quoted blocks.

    An example in a docstring is documentation, not a deployed credential.
    A real parser would be better; a scanner that needs a full parse per
    language would not run on a Pi, and this is accurate for the shape that
    actually causes false positives.
    """
    inside: set[int] = set()
    delim = ""
    for number, line in enumerate(text.splitlines(), 1):
        rest = line
        while rest:
            if delim:
                index = rest.find(delim)
                inside.add(number)
                if index < 0:
                    break
                rest = rest[index + 3:]
                delim = ""
                continue
            match = re.search(r'"""|\'\'\'', rest)
            if match is None:
                break
            after = rest[match.end():]
            closing = after.find(match.group(0))
            if closing >= 0:                       # opened and closed on one line
                rest = after[closing + 3:]
                continue
            delim = match.group(0)
            inside.add(number)
            break
    return inside


def _quoted_spans(line: str) -> list[tuple[int, int]]:
    """Character ranges inside single-line string literals.

    A rule match that *begins* inside a quoted region is not executed code:
    `"def parse(s): return eval(s)"` contains no eval, and a rule's own regex
    is not a call to the thing it matches.  Both of those fired on this
    repository - one of them on this very file - the first time the scanner
    ran over real code.

    Matching on where the match *begins* is what keeps the credential rules
    working: `password = "hunter2"` begins at `password`, outside the quotes.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char not in ("'", '"'):
            index += 1
            continue
        quote = char
        start = index
        index += 1
        while index < length:
            if line[index] == "\\":
                index += 2
                continue
            if line[index] == quote:
                index += 1
                break
            index += 1
        spans.append((start, index))
    return spans


def _inside(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def _comment_at(line: str, quoted: Sequence[tuple[int, int]]) -> int:
    """Where a trailing comment begins, or -1.

    A `#` inside a string literal is not a comment, which is why this needs
    the quoted spans rather than a `str.find`.
    """
    for index, char in enumerate(line):
        if char == "#" and not _inside(index, quoted):
            return index
    return -1


def scan_text(text: str, path: str = "<text>") -> list[Finding]:
    """Every finding in one file's contents."""
    findings: list[Finding] = []
    docstrings = _docstring_spans(text) if path.endswith((".py", ".pyi")) else set()
    fixture = _is_fixture(path)
    for number, line in enumerate(text.splitlines(), 1):
        if SUPPRESS.search(line):
            continue
        if _PLACEHOLDER.search(line):
            continue
        quoted = _quoted_spans(line)
        comment = _comment_at(line, quoted)
        for rule in RULES:
            if not rule.applies_to(path):
                continue
            match = rule.pattern.search(line)
            if match is None or (rule.refute is not None and rule.refute.search(line)):
                continue
            if not rule.literal_value:
                if _inside(match.start(), quoted):
                    # A string literal: documentation, a fixture, or a rule's
                    # own pattern.  Not code that runs.  See `_quoted_spans`.
                    continue
                if 0 <= comment <= match.start():
                    # Commented out, so it does not run either.
                    continue
            if number in docstrings:
                # Inside a triple-quoted block, which `_quoted_spans` cannot
                # see because it reads one line at a time.  Same reasoning: a
                # code sample in a docstring, or a rule's own multi-line
                # pattern, is not a call.  This file's own `CERT_NONE` pattern
                # was the finding that proved it.
                continue
            if rule.name in CREDENTIAL_RULES and fixture:
                # Expected here.  See `_is_fixture`.
                continue
            findings.append(Finding(
                rule=rule.name, severity=rule.severity, path=path, line=number,
                evidence=line.strip()[:CLIP], impact=rule.impact,
            ))
    return findings


def source_files(root: Path, *, limit: int = 5000) -> list[Path]:
    out: list[Path] = []
    patterns = _ignore_patterns(root)
    for path in sorted(root.rglob("*")):
        if len(out) >= limit:
            break
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if _ignored(rel, patterns):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(path)
    return out


def scan(root: Path | str) -> Report:
    """Audit a working tree."""
    base = Path(root)
    if base.is_file():
        files = [base]
        base = base.parent
    else:
        files = source_files(base)
    findings: list[Finding] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = str(path.relative_to(base))
        except ValueError:
            rel = path.name
        findings.extend(scan_text(text, rel))
    return Report(findings=findings, scanned=len(files))


#: `+++ b/path` then `@@ -a,b +c,d @@`, so an added line can be given the line
#: number it will have in the new file - a finding whose line number is wrong is
#: not evidence, it is a wild goose chase.
_DIFF_FILE: Final = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def scan_diff(diff: str) -> Report:
    """Audit only the added lines of a unified diff.

    Reviewing a change should not report the whole repository's existing debt;
    the finding that matters is the one this change introduces.
    """
    findings: list[Finding] = []
    path = ""
    number = 0
    files: set[str] = set()
    for raw in diff.splitlines():
        header = _DIFF_FILE.match(raw)
        if header:
            path = header.group(1)
            files.add(path)
            continue
        hunk = _DIFF_HUNK.match(raw)
        if hunk:
            number = int(hunk.group(1))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            findings.extend(
                Finding(rule=f.rule, severity=f.severity, path=path, line=number,
                        evidence=f.evidence, impact=f.impact)
                for f in scan_text(raw[1:], path)
            )
            number += 1
        elif not raw.startswith("-"):
            number += 1
    return Report(findings=findings, scanned=len(files))


# -- the model pass ---------------------------------------------------------------------

#: What the model is asked to emit.  A fixed line format rather than JSON: a
#: model that trails prose after valid JSON breaks a strict parser, whereas an
#: unparseable line here is simply skipped.
PROMPT: Final = """\
Audit the code below for security vulnerabilities.

Report ONLY findings you can point at a specific line for. For each one emit
exactly one line, no prose, no markdown:

SEVERITY|FILE|LINE|RULE|WHAT AN ATTACKER GAINS

SEVERITY is high, medium or low. RULE is a short slug like sql-injection.
If you find nothing, emit the single word NONE.

{code}
"""

_REPLY_LINE: Final = re.compile(
    r"^\s*(high|medium|low)\s*\|\s*([^|]{1,200}?)\s*\|\s*(\d{1,7})\s*\|\s*([^|]{1,60}?)\s*\|\s*(.{3,400})$",
    re.IGNORECASE,
)


def parse_reply(reply: str) -> list[Finding]:
    """Findings from a model reply.  Anything unparseable is dropped.

    Never invents a line number, and never promotes a prose paragraph into a
    finding: a fabricated location wastes more of a reviewer's time than a
    missed one, because it costs trust in every other finding too.
    """
    out: list[Finding] = []
    for raw in (reply or "").splitlines():
        match = _REPLY_LINE.match(raw)
        if match is None:
            continue
        severity = match.group(1).lower()
        if severity not in SEVERITIES:
            continue
        out.append(Finding(
            rule=match.group(4).strip().lower().replace(" ", "-")[:60],
            severity=severity, path=match.group(2).strip(),
            line=int(match.group(3)), evidence="",
            impact=match.group(5).strip()[:400], source="model",
        ))
    return out


def audit(text: str, *, path: str = "<text>",
          ask: Callable[[str], str] | None = None) -> Report:
    """Static findings, plus a model's if one is reachable.

    The static pass is authoritative; the model pass is additive.  If `ask`
    raises, times out or returns rubbish, the report is exactly the static one -
    a degraded audit is useful, a crashed audit is not.
    """
    report = Report(findings=scan_text(text, path), scanned=1)
    if ask is None:
        return report
    try:
        reply = ask(PROMPT.format(code=text[:20_000]))
    except Exception:
        # An unreachable or misbehaving model must not fail the audit.
        return report
    return report.merged(parse_reply(reply))


# -- the report -----------------------------------------------------------------------


@dataclass(slots=True)
class Report:
    findings: list[Finding] = field(default_factory=list)
    scanned: int = 0

    @property
    def worst(self) -> Severity | None:
        for severity in SEVERITIES:
            if any(f.severity == severity for f in self.findings):
                return severity
        return None

    @property
    def ok(self) -> bool:
        """Nothing high or medium.  Low findings are worth knowing, not blocking."""
        return self.worst in (None, LOW)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def merged(self, extra: Iterable[Finding]) -> Report:
        """Fold in more findings, dropping ones already reported.

        De-duplicated on path, line and rule.  The model routinely restates a
        static finding in its own words; reporting it twice makes the report
        look padded and hides the ones only it found.
        """
        seen = {f.key() for f in self.findings}
        combined = list(self.findings)
        for finding in extra:
            if finding.key() in seen:
                continue
            seen.add(finding.key())
            combined.append(finding)
        return Report(findings=combined, scanned=self.scanned)

    def lines(self) -> list[str]:
        if not self.findings:
            return [f"no findings in {self.scanned} file(s)"]
        order = {s: i for i, s in enumerate(SEVERITIES)}
        ranked = sorted(self.findings,
                        key=lambda f: (order.get(f.severity, 9), f.path, f.line))
        out = [f"{len(self.findings)} finding(s) in {self.scanned} file(s): "
               + ", ".join(f"{self.count(s)} {s}" for s in SEVERITIES if self.count(s))]
        for finding in ranked:
            out.append("")
            out.append(finding.line_text())
            if finding.evidence:
                out.append(f"         {finding.evidence}")
            out.append(f"         {finding.impact}")
            if finding.source == "model":
                out.append("         (reported by the model, not the rule set)")
        return out


# -- scoring ---------------------------------------------------------------------------

SECURITY: Final = "security"


def security_criterion(report: Report | None, weight: float) -> Criterion:
    """How a branch's safety folds into `/spec`'s ranking.

    `applies=False` when no audit ran, which the scoring module excludes from
    the denominator rather than scoring zero.  Penalising an unaudited branch
    would make the winner depend on whether the audit happened to be reachable,
    which is not a property of the code.
    """
    if report is None:
        return Criterion(SECURITY, 0.0, weight, "no audit ran", applies=False)
    if not report.findings:
        return Criterion(SECURITY, 1.0, weight,
                         f"no findings in {report.scanned} file(s)")
    high, medium, low = (report.count(s) for s in SEVERITIES)
    # A single high finding is disqualifying rather than merely costly: shipping
    # one remote-code-execution path is worse than a dozen plaintext-HTTP notes,
    # and an average would let volume drown severity.
    if high:
        return Criterion(SECURITY, 0.0, weight,
                         f"{high} high-severity finding(s): "
                         + ", ".join(sorted({f.rule for f in report.findings
                                             if f.severity == HIGH})))
    value = max(0.0, 1.0 - 0.25 * medium - 0.05 * low)
    return Criterion(SECURITY, value, weight,
                     f"{medium} medium and {low} low finding(s)")


# -- the command --------------------------------------------------------------------------


def _staged_diff(cwd: Path) -> str:
    for argv in (["git", "diff", "--cached", "-U0"], ["git", "diff", "-U0"]):
        try:
            done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                  timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout
    return ""


def _audit_command(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Outcome

    workspace = Path(getattr(state, "workspace", None) or Path.cwd())
    if args and args[0] in ("--diff", "-d", "diff"):
        diff = _staged_diff(workspace)
        if not diff:
            return Outcome(["nothing staged or modified to audit"], TONE_INFO)
        report = scan_diff(diff)
        title = "audited the pending diff"
    else:
        target = workspace / args[0] if args else workspace
        if not target.exists():
            return Outcome.error(f"no such path: {target}")
        report = scan(target)
        title = f"audited {target}"

    tone = TONE_OK if report.ok else TONE_ERR
    return Outcome([title, *report.lines()], tone)


def security_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("audit", "scan for security findings", _audit_command,
                usage="/audit [path | --diff]", aliases=("security",)),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """Built on first access.  The re-check is the same guard `tasks.py` carries:
    building imports the shell registry, which re-enters here before the outer
    call has stored anything, so one check registers every command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = security_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
