"""Watching a long run from a phone.

`offset` already runs headless (`offset/core/daemon.py`) and keeps background
jobs that survive a restart (`offset/core/jobs.py`). The only way to see either
was to be sitting at the terminal. This is a small read-mostly HTTP server that
answers "what is it doing" from anywhere on the network the user chooses.

**The security is the substance of this module, not a wrapper around it.** This
binds a socket on a machine running an agent that can execute tools, so every
default is the closed one:

- **Loopback by default.** A wider interface requires an explicit host
  argument. Never inferred, never widened because a phone could not reach it -
  the user who wants that has to say so.
- **A token on every route, including the page.** Generated with
  `secrets.token_urlsafe`, stored 0600, compared with `secrets.compare_digest`.
  Not `==`: string comparison short-circuits on the first wrong byte, which
  leaks the token's prefix to anyone who can time the responses.
- **The one mutating route needs the token in a header.** A query string rides
  along in a link, a bookmark and a browser history, so a page on another
  origin can navigate to it. A header cannot be set by a plain navigation.
- **No path from the request ever reaches the filesystem.** There is no static
  file route at all; the page is a constant in this module. A monitor with a
  traversal bug would hand over the machine.
- **Credential-shaped values are redacted from every response.** The whole
  point is to be readable from a phone on a network the user does not control.

`stop()` must leave nothing behind. A monitor that outlives its test wedges the
next run of the suite on a bound port, so shutdown is synchronous and
idempotent.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

from offset.core import settings

#: Loopback.  The one setting that turns a convenience into an exposure.
DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8787

#: 0600, beside the other secrets offset keeps.
TOKEN_NAME: Final = "monitor-token"

#: Header the mutating route requires.  A navigation cannot set it.
TOKEN_HEADER: Final = "X-Offset-Token"

#: How long a client waits before giving up, and how long shutdown waits for
#: the serving thread.  Both short: a monitor is a convenience.
SHUTDOWN_TIMEOUT: Final = 5.0

#: Values that must never leave the process, matched on the *key* rather than
#: the value - a key called `api_key` is a credential whatever it contains.
SECRET_KEYS: Final = re.compile(
    r"(?i)(token|secret|password|passwd|api[-_]?key|credential|authorization|bearer)")

#: Credential-shaped values, matched on the value for the cases where the key
#: gives nothing away.
SECRET_VALUES: Final = re.compile(
    r"(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|glpat-[A-Za-z0-9_-]{16,}"
    r"|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")

REDACTED: Final = "[redacted]"


def token_file(home: Path | None = None) -> Path:
    return (home if home is not None else settings.home()) / TOKEN_NAME


def read_or_make_token(home: Path | None = None) -> str:
    """The monitor's token, created on first use.

    Written 0600 before anything is served, so the token never exists on disk
    in a world-readable state even briefly.
    """
    path = token_file(home)
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    token = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create with the right mode rather than chmod afterwards: between the
        # write and the chmod there is a window where it is readable.
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError:
        pass  # an unwritable home still gets a working, in-memory token
    return token


def redact(value: Any) -> Any:
    """Strip anything credential-shaped from a structure about to be served."""
    if isinstance(value, dict):
        return {k: (REDACTED if SECRET_KEYS.search(str(k)) else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUES.sub(REDACTED, value)
    return value


# -- what it reports ---------------------------------------------------------------


@dataclass(slots=True)
class Snapshot:
    """Everything the monitor knows, gathered on the calling thread.

    A callable rather than a live reference to the shell: the serving threads
    must not touch `ShellState`, which is not designed for concurrent readers.
    """

    model: str = ""
    session: str = ""
    started: float = field(default_factory=time.time)
    busy: bool = False
    workspace: str = ""

    def payload(self) -> dict[str, Any]:
        return redact({
            "model": self.model,
            "session": self.session,
            "workspace": self.workspace,
            "uptime_seconds": round(max(0.0, time.time() - self.started), 1),
            "busy": self.busy,
            "at": round(time.time(), 3),
        })


def _jobs_payload(home: Path) -> dict[str, Any]:
    try:
        from offset.core.jobs import JobStore
    except ImportError:
        return {"jobs": [], "note": "job store unavailable"}
    try:
        store = JobStore(home)
        jobs = store.list()
    except Exception:
        # A corrupt job file must not take the monitor down with it.
        return {"jobs": [], "note": "could not read the job store"}
    return redact({"jobs": [
        {"id": getattr(j, "id", ""), "state": getattr(j, "state", ""),
         "label": str(getattr(j, "label", ""))[:120],
         "started": getattr(j, "started", 0.0)}
        for j in jobs
    ]})


def _usage_payload(home: Path) -> dict[str, Any]:
    try:
        from offset.core.telemetry import Ledger, Total, rollup
    except ImportError:
        return {"note": "telemetry unavailable"}
    entries = Ledger(home).read()
    if not entries:
        # Reporting zeros here would read as "this cost nothing", which is a
        # different claim from "nothing was recorded".
        return {"note": "nothing recorded yet", "turns": 0}
    grand = Total()
    for entry in entries:
        grand.add(entry)
    return {
        "turns": grand.turns,
        "tokens_in": grand.tokens_in,
        "tokens_out": grand.tokens_out,
        "cost": round(grand.cost, 4),
        "cost_is_partial": grand.partial,
        "failures": grand.failures,
        "by_model": {
            name: {"turns": t.turns, "cost": round(t.cost, 4),
                   "partial": t.partial, "failures": t.failures}
            for name, t in rollup(entries, "model").items()
        },
    }


# -- the page -----------------------------------------------------------------------

#: One self-contained document.  No CDN, no external font, no framework: this
#: is served to a phone over a network the user may not control, and every
#: external asset is another party in that conversation.
PAGE: Final = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>offset</title>
<style>
 :root { --ink:#111111; --bg:#F4F4F0; --surface:#FFFFFF; --muted:#555555;
         --yellow:#FFDE59; --mint:#B2FF9E; --red:#FF5A5F; }
 * { box-sizing:border-box; margin:0; }
 body { background:var(--bg); color:var(--ink); font-family:ui-monospace,
        SFMono-Regular,Menlo,monospace; padding:14px; font-size:17px;
        line-height:1.45; }
 h1 { font-size:1.5rem; letter-spacing:3px; text-transform:uppercase;
      margin-bottom:14px; }
 .card { background:var(--surface); border:3px solid var(--ink);
         box-shadow:6px 6px 0 0 var(--ink); padding:14px; margin-bottom:18px; }
 .card h2 { font-size:.8rem; letter-spacing:2px; text-transform:uppercase;
            color:var(--muted); margin-bottom:10px; }
 .row { display:flex; justify-content:space-between; gap:12px;
        padding:4px 0; border-bottom:1px solid #eee; }
 .row:last-child { border-bottom:none; }
 /* The break belongs on the value, never the row: a session id is one long
    unbreakable token, and breaking the whole row snapped the *label* mid-word
    ("sessio / n") on a 390px screen. */
 .k { color:var(--muted); white-space:nowrap; }
 .v { font-weight:700; text-align:right; word-break:break-all; min-width:0; }
 .pill { display:inline-block; border:2px solid var(--ink); padding:1px 8px;
         font-size:.75rem; text-transform:uppercase; letter-spacing:1px; }
 .busy { background:var(--yellow); } .idle { background:var(--mint); }
 .bad { background:var(--red); }
 button { font:inherit; font-weight:700; border:3px solid var(--ink);
          background:var(--yellow); padding:6px 12px; cursor:pointer; }
 button:active { transform:translate(3px,3px); }
 .err { color:#a00; font-size:.85rem; }
</style></head><body>
<h1>offset</h1>
<div class="card"><h2>status</h2><div id="status">loading...</div></div>
<div class="card"><h2>jobs</h2><div id="jobs">loading...</div></div>
<div class="card"><h2>usage</h2><div id="usage">loading...</div></div>
<p class="err" id="err"></p>
<script>
 const token = new URLSearchParams(location.search).get("token") || "";
 const esc = s => String(s).replace(/[&<>"']/g,
   c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
 const row = (k, v) => `<div class="row"><span class="k">${esc(k)}</span>`
   + `<span class="v">${v}</span></div>`;
 async function get(path) {
   const r = await fetch(path + "?token=" + encodeURIComponent(token));
   if (!r.ok) throw new Error(path + " -> " + r.status);
   return r.json();
 }
 async function cancel(id) {
   await fetch("/api/cancel?token=" + encodeURIComponent(token), {
     method: "POST", headers: { "X-Offset-Token": token,
       "Content-Type": "application/json" },
     body: JSON.stringify({ id }) });
   refresh();
 }
 function renderStatus(d) {
   const cls = d.busy ? "busy" : "idle";
   document.getElementById("status").innerHTML =
     row("state", `<span class="pill ${cls}">${d.busy?"working":"idle"}</span>`)
     + row("model", esc(d.model || "-"))
     + row("session", esc(d.session || "-"))
     + row("workspace", esc(d.workspace || "-"))
     + row("uptime", Math.round(d.uptime_seconds) + "s");
 }
 function renderJobs(d) {
   const js = d.jobs || [];
   if (!js.length) { document.getElementById("jobs").textContent =
     d.note || "no background jobs"; return; }
   document.getElementById("jobs").innerHTML = js.map(j => {
     const live = j.state === "running" || j.state === "queued";
     const btn = live ? ` <button onclick="cancel('${esc(j.id)}')">stop</button>` : "";
     return row(j.label || j.id,
       `<span class="pill">${esc(j.state)}</span>${btn}`);
   }).join("");
 }
 function renderUsage(d) {
   if (d.note && !d.turns) { document.getElementById("usage").textContent = d.note;
     return; }
   const money = d.cost_is_partial ? "$" + d.cost + "+" : "$" + d.cost;
   document.getElementById("usage").innerHTML =
     row("turns", d.turns) + row("tokens in", (d.tokens_in||0).toLocaleString())
     + row("tokens out", (d.tokens_out||0).toLocaleString())
     + row("cost", money)
     + row("failures", `<span class="pill ${d.failures?"bad":"idle"}">`
        + d.failures + "</span>");
 }
 async function refresh() {
   try {
     renderStatus(await get("/api/status"));
     renderJobs(await get("/api/jobs"));
     renderUsage(await get("/api/usage"));
     document.getElementById("err").textContent = "";
   } catch (e) { document.getElementById("err").textContent = e.message; }
 }
 refresh(); setInterval(refresh, 3000);
</script></body></html>
"""


def page() -> str:
    return PAGE


# -- the server ---------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "offset-monitor"
    #: Silence the default stderr access log: it would interleave with the
    #: shell's own output and there is nothing useful in it.
    def log_message(self, *_args: Any) -> None:
        return

    @property
    def monitor(self) -> Monitor:
        return self.server.monitor            # type: ignore[attr-defined]

    def _authorised(self, *, require_header: bool = False) -> bool:
        expected = self.monitor.token
        supplied = ""
        header = self.headers.get(TOKEN_HEADER, "")
        if require_header:
            # A query string rides along in links, bookmarks and browser
            # history, so a page on another origin can navigate to it.  A
            # header cannot be set by a plain navigation.
            supplied = header
        else:
            supplied = header or self._query_token()
        # compare_digest, never `==`: string comparison short-circuits on the
        # first wrong byte, which leaks the prefix to anyone timing responses.
        return bool(supplied) and secrets.compare_digest(supplied, expected)

    def _query_token(self) -> str:
        from urllib.parse import parse_qs, urlsplit

        return (parse_qs(urlsplit(self.path).query).get("token") or [""])[0]

    def _route(self) -> str:
        from urllib.parse import urlsplit

        return urlsplit(self.path).path

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        # No caching: a stale status page is worse than a slow one.  And no
        # referrer, so the token cannot leak into another site's logs.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # a phone that closed the tab is not an error

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _deny(self) -> None:
        self._json(401, {"error": "a valid token is required"})

    def do_GET(self) -> None:            # the stdlib dictates this name
        route = self._route()
        if not self._authorised():
            self._deny()
            return
        if route == "/":
            self._send(200, page().encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/status":
            self._json(200, self.monitor.snapshot().payload())
        elif route == "/api/jobs":
            self._json(200, _jobs_payload(self.monitor.home))
        elif route == "/api/usage":
            self._json(200, _usage_payload(self.monitor.home))
        else:
            # Deliberately not a file lookup.  There is no static route: a
            # traversal bug here would hand over the machine.
            self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:           # the stdlib dictates this name
        if self._route() != "/api/cancel":
            self._json(404, {"error": "no such route"})
            return
        if not self._authorised(require_header=True):
            self._deny()
            return
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            self._json(400, {"error": "expected a JSON body"})
            return
        job_id = str((body or {}).get("id") or "").strip()
        if not job_id:
            self._json(400, {"error": "no job id"})
            return
        self._json(200, {"cancelled": self.monitor.cancel(job_id), "id": job_id})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass(slots=True)
class Monitor:
    """A read-mostly HTTP view of what the agent is doing."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str = ""
    home: Path = field(default_factory=settings.home)
    #: How the monitor learns the current state.  Injected so the serving
    #: threads never touch `ShellState`, which has no concurrent readers.
    source: Any = None
    _server: Any = field(default=None, repr=False)
    _thread: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.token:
            self.token = read_or_make_token(self.home)

    def snapshot(self) -> Snapshot:
        if self.source is None:
            return Snapshot()
        try:
            got = self.source()
        except Exception:
            return Snapshot()
        return got if isinstance(got, Snapshot) else Snapshot()

    def cancel(self, job_id: str) -> bool:
        try:
            from offset.core.jobs import JobStore

            return bool(JobStore(self.home).cancel(job_id))
        except Exception:
            return False

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/?token={self.token}"

    def start(self) -> Monitor:
        if self.running:
            return self
        server = _Server((self.host, self.port), _Handler)
        server.monitor = self                        # type: ignore[attr-defined]
        # Port 0 asks the kernel for a free one; read back what it gave us so
        # `url` is correct and a test can connect.
        self.port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever,
                                  name="offset-monitor", daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def stop(self) -> None:
        """Tear it down completely.  Idempotent.

        A monitor that outlives its caller holds the port, and the next start
        fails with an error nobody connects to the leak.  So this is
        synchronous: it waits for the serving thread to actually finish.
        """
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=SHUTDOWN_TIMEOUT)


def free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


# -- the command ---------------------------------------------------------------------

#: One per process.  A second monitor on a second port is not a feature, it is
#: two sockets to forget about.
_active: Monitor | None = None


def _shell_snapshot(state: Any) -> Any:
    started = time.time()

    def read() -> Snapshot:
        return Snapshot(
            model=str(getattr(state, "model", "") or ""),
            session=str(getattr(getattr(state, "session", None), "id", "") or ""),
            workspace=str(getattr(state, "workspace", "") or ""),
            busy=bool(getattr(state, "live", None)),
            started=started,
        )

    return read


def _monitor_command(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    global _active
    action = (args[0].lower() if args else "status")
    #: Resolved on the shell's thread.  A serving thread that asks
    #: `settings.home()` for itself answers with whatever the environment says
    #: by then, which for an exited shell is the wrong directory.
    home = settings.home()

    if action in ("stop", "off"):
        if _active is None:
            return Outcome(["the monitor is not running"], TONE_INFO)
        _active.stop()
        _active = None
        return Outcome(["monitor stopped"], TONE_OK)

    if action in ("status", "url"):
        if _active is None:
            return Outcome(["the monitor is not running - /monitor start"], TONE_INFO)
        return Outcome([f"monitor: {_active.url}",
                        f"bound to {_active.host}:{_active.port}"], TONE_INFO)

    if action not in ("start", "on"):
        return Outcome.error("usage: /monitor [start|stop|status] [host]")

    if _active is not None:
        return Outcome([f"already running: {_active.url}"], TONE_INFO)

    host = args[1] if len(args) > 1 else DEFAULT_HOST
    try:
        monitor = Monitor(host=host, port=free_port(host), home=home,
                          source=_shell_snapshot(state)).start()
    except OSError as exc:
        return Outcome.error(f"could not bind {host}: {exc}")
    _active = monitor

    lines = [f"monitor: {monitor.url}", "the token is in that URL - treat it as a password"]
    if host not in ("127.0.0.1", "localhost", "::1"):
        lines.append(f"warning: bound to {host}, which is reachable from the "
                     f"network, not just this machine")
    return Outcome(lines, TONE_OK)


def monitor_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("monitor", "watch this session from a phone", _monitor_command,
                usage="/monitor [start|stop|status] [host]"),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """Built on first access.  The re-check is the guard `tasks.py` carries:
    building imports the shell registry, which re-enters this module before the
    outer call has stored anything, so one check registers every command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = monitor_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
