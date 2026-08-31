"""Work that outlives the terminal it was started from.

A subagent call blocks the thread that made it, which is fine for a thirty
second job and useless for a twenty minute one: closing the terminal threw the
result away.  The fix is not a bigger thread pool, it is to stop treating the
running process as the owner of the work.

So a job is a *file*.  The record under `$OFFSET_HOME/jobs/<id>.json` is the
only authority on what a job is and how it ended; the process doing the work is
just something that updates that file.  Everything follows from that:

  * a detached job is spawned as `python3 -m offset.core.jobs run <id>` with
    `start_new_session=True`, so the terminal's SIGHUP never reaches it and it
    keeps its own process group.  Its stdout and stderr go to `<id>.log`;
  * `status`, `logs` and `wait` work from a *cold* process, because they only
    ever read files.  Collecting a `/spec` you started yesterday is a read;
  * a worker touches `<id>.beat` while it lives, and records its pid.  A worker
    that is killed leaves the record saying `running` for ever, so `reap()`
    settles any record whose pid is gone or whose heartbeat stopped.  Nothing
    can hang for ever waiting on a process that no longer exists;
  * cancellation is written down before it is signalled, so a worker that dies
    to the signal cannot overwrite the user's last word with its own guess.

Every rewrite goes through a sibling temporary file and `os.replace`, so a
reader never sees half a record.  A failure is a value: a job that could not be
spawned comes back as a `Job` in state `failed` carrying the reason, and a
handler that raises becomes `error` text rather than an exception in the caller.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Sequence

from offset.core import settings
from offset.core.entries import new_id

# -- states -----------------------------------------------------------------

QUEUED: Final = "queued"
RUNNING: Final = "running"
DONE: Final = "done"
FAILED: Final = "failed"
CANCELLED: Final = "cancelled"

#: Ordered for display, and the accepted values of the `/jobs` filter.
STATES: Final[tuple[str, ...]] = (QUEUED, RUNNING, DONE, FAILED, CANCELLED)

LIVE: Final = frozenset({QUEUED, RUNNING})
TERMINAL: Final = frozenset({DONE, FAILED, CANCELLED})

#: Seconds between heartbeat touches.  Small enough that staleness is obvious
#: long before `STALE_AFTER`, large enough to be free.
HEARTBEAT: Final = 2.0

#: Silence from a live pid for this long means the pid was recycled and the
#: worker is gone.  Fifteen heartbeats of margin: a loaded machine will not
#: trip it, a rebooted one will.
STALE_AFTER: Final = 30.0

#: Where a kind's worker lives, as `module:function`, resolved only when a job
#: of that kind actually runs.  A dotted string rather than the function itself
#: so a cold `python3 -m offset.core.jobs` does not import every subsystem that
#: can submit work — importing the agent stack to run a task, or vice versa, is
#: two seconds of startup for nothing.
HANDLERS: Final[dict[str, str]] = {
    "subagent": "offset.tools.agents:run_job",
    "task": "offset.core.tasks:run_job",
}


# -- what a worker produced -------------------------------------------------


@dataclass(slots=True)
class JobOutcome:
    """A handler's report.  `error` set means the job failed."""

    text: str = ""
    error: str | None = None
    #: Path to the session log the work was recorded in, so `/job` can point at
    #: a replayable transcript from a process that never saw the run.
    session: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    #: Kind-specific facts worth keeping: a task id, a branch name.
    data: dict[str, Any] = field(default_factory=dict)


#: A job kind's worker.  It gets the record, a private directory it may write
#: into, and the cancellation flag its process installed on SIGTERM.
Handler = Callable[["Job", Path, threading.Event], JobOutcome]


# -- the record -------------------------------------------------------------


@dataclass(slots=True)
class Job:
    """One unit of background work, exactly as it is stored."""

    id: str
    kind: str = "subagent"
    prompt: str = ""
    agent: str = ""
    state: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    #: The pid of whatever is doing the work: the spawned process for a
    #: detached job, this process for an in-thread one.  It is how `reap` tells
    #: an abandoned record from a working one.
    pid: int = 0
    detached: bool = True
    model: str = ""
    approval: str = "safe"
    cwd: str = ""
    timeout: float = 0.0
    result: str = ""
    error: str | None = None
    session: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def live(self) -> bool:
        return self.state in LIVE

    @property
    def ok(self) -> bool:
        return self.state == DONE

    @property
    def age(self) -> float:
        """Seconds since it was submitted, or how long it took if it is over."""
        return max(0.0, (self.finished or time.time()) - self.created)

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.agent}" if self.agent else self.kind

    def summary(self, width: int = 52) -> str:
        body = " ".join((self.prompt or self.result or "").split())
        return body if len(body) <= width else body[: width - 1] + "\u2026"

    def line(self, width: int = 96) -> str:
        """One row of the `/jobs` table."""
        room = max(16, width - 44)
        return f"{self.id[-8:]:<9} {self.state:<10} {ago(self.age):>7}  {self.label:<16} {self.summary(room)}"

    def report(self) -> list[str]:
        """The whole record, for `/job <id>`."""
        lines = [
            f"job {self.id}",
            f"  kind       {self.label}",
            f"  state      {self.state}",
            f"  submitted  {_stamp(self.created)} ({ago(self.age)} ago)" if not self.finished
            else f"  submitted  {_stamp(self.created)}",
        ]
        if self.started:
            lines.append(f"  started    {_stamp(self.started)}")
        if self.finished:
            lines.append(f"  finished   {_stamp(self.finished)} after {ago(self.finished - self.created)}")
        lines.append(f"  process    {'detached pid ' + str(self.pid) if self.detached else 'in-process pid ' + str(self.pid)}")
        if self.model:
            lines.append(f"  model      {self.model}")
        if self.cwd:
            lines.append(f"  workspace  {self.cwd}")
        if self.session:
            lines.append(f"  session    {self.session}")
        if self.usage:
            spent = ", ".join(f"{k} {v}" for k, v in sorted(self.usage.items()) if v)
            if spent:
                lines.append(f"  usage      {spent}")
        if self.prompt:
            lines += ["", "prompt:", *_indent(self.prompt, 12)]
        if self.error:
            lines += ["", f"error: {self.error}"]
        if self.result:
            lines += ["", "result:", *_indent(self.result, 4000)]
        return lines

    # -- storage ----------------------------------------------------------

    def to_data(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "agent": self.agent,
            "state": self.state,
            "created": round(self.created, 6),
            "started": round(self.started, 6),
            "finished": round(self.finished, 6),
            "pid": self.pid,
            "detached": self.detached,
            "model": self.model,
            "approval": self.approval,
            "cwd": self.cwd,
            "timeout": self.timeout,
            "result": self.result,
            "error": self.error,
            "session": self.session,
            "usage": self.usage,
            "data": self.data,
        }

    @classmethod
    def from_data(cls, obj: Any) -> "Job":
        """Read a record.  A shape that is not a job at all is an error; a
        record missing a field it gained later is not."""
        if not isinstance(obj, dict):
            raise ValueError("a job record must be an object")
        jid, state = obj.get("id"), obj.get("state")
        if not isinstance(jid, str) or not jid:
            raise ValueError("a job record needs a string id")
        if not isinstance(state, str) or state not in STATES:
            raise ValueError(f"a job record needs one of these states: {', '.join(STATES)}")
        usage = obj.get("usage")
        data = obj.get("data")
        error = obj.get("error")
        return cls(
            id=jid,
            kind=str(obj.get("kind") or "subagent"),
            prompt=str(obj.get("prompt") or ""),
            agent=str(obj.get("agent") or ""),
            state=state,
            created=float(obj.get("created") or 0.0),
            started=float(obj.get("started") or 0.0),
            finished=float(obj.get("finished") or 0.0),
            pid=int(obj.get("pid") or 0),
            detached=bool(obj.get("detached", True)),
            model=str(obj.get("model") or ""),
            approval=str(obj.get("approval") or "safe"),
            cwd=str(obj.get("cwd") or ""),
            timeout=float(obj.get("timeout") or 0.0),
            result=str(obj.get("result") or ""),
            error=error if isinstance(error, str) else None,
            session=str(obj.get("session") or ""),
            usage={str(k): int(v) for k, v in usage.items()} if isinstance(usage, dict) else {},
            data=data if isinstance(data, dict) else {},
        )


# -- the store --------------------------------------------------------------


class JobStore:
    """The jobs directory, and the two ways of putting work into it."""

    __slots__ = ("_cancels", "_lock", "_root", "_threads")

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        #: Left unresolved when not given: `settings.home()` moves under the
        #: tests and under `--home`, and a store built at import time would
        #: have baked in the wrong directory.
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else settings.home() / "jobs"

    # -- paths ------------------------------------------------------------

    def record(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.log"

    def beat_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.beat"

    def workdir(self, job_id: str) -> Path:
        """A directory the job owns: child sessions, scratch files."""
        return self.root / job_id

    # -- reading ----------------------------------------------------------

    def status(self, job_id: str) -> Job | None:
        return self._read(self.record(job_id))

    def list(self, *, state: str | None = None) -> list[Job]:
        """Newest first.  A record that will not parse is left out, not fatal."""
        root = self.root
        if not root.is_dir():
            return []
        out: list[Job] = []
        for path in sorted(root.glob("*.json")):
            job = self._read(path)
            if job is None or (state is not None and job.state != state):
                continue
            out.append(job)
        out.sort(key=lambda j: (j.created, j.id), reverse=True)
        return out

    def logs(self, job_id: str, *, tail: int = 0) -> list[str]:
        """What the worker printed.  Missing log, empty list: a job that has
        not started yet has nothing to say, and that is not an error."""
        try:
            text = self.log_path(job_id).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        lines = text.splitlines()
        return lines[-tail:] if tail > 0 else lines

    def _read(self, path: Path) -> Job | None:
        try:
            return Job.from_data(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    # -- writing ----------------------------------------------------------

    def save(self, job: Job) -> Job:
        """Replace the record atomically.  A reader never sees half of one."""
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        target = self.record(job.id)
        fd, tmp = tempfile.mkstemp(dir=str(root), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(job.to_data(), fh, ensure_ascii=False)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return job

    # -- starting ---------------------------------------------------------

    def start(
        self,
        prompt: str,
        *,
        kind: str = "subagent",
        agent: str = "",
        model: str = "",
        approval: str = "safe",
        cwd: str | os.PathLike[str] | None = None,
        timeout: float = 0.0,
        detached: bool = True,
        data: dict[str, Any] | None = None,
    ) -> Job:
        """Submit work and return its handle immediately.

        `detached` spawns a process that survives this one.  Without it the job
        runs on a daemon thread here, which is the right choice when the caller
        is going to wait anyway and wants no process overhead — but it dies with
        this process, and the record says so.
        """
        if kind not in HANDLERS:
            return Job(id=new_id(), kind=kind, state=FAILED, prompt=prompt, detached=detached,
                       error=f"no job kind named {kind!r}; available: {', '.join(sorted(HANDLERS))}")
        if not prompt.strip() and kind in ("subagent", "task"):
            return Job(id=new_id(), kind=kind, state=FAILED, detached=detached,
                       error="a background job needs a prompt describing the whole piece of work")
        with self._lock:  # `new_id` keeps its counter in module state
            job = Job(
                id=new_id(),
                kind=kind,
                prompt=prompt,
                agent=agent,
                model=model,
                approval=approval,
                cwd=str(Path(cwd) if cwd is not None else Path.cwd()),
                timeout=timeout,
                detached=detached,
                data=dict(data or {}),
            )
        self.workdir(job.id).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return self._spawn(job) if detached else self._thread(job)

    def _spawn(self, job: Job) -> Job:
        """Start the job in a process of its own.

        `start_new_session` is the whole point: the child leads a new session
        and process group, so the SIGHUP that arrives when the terminal closes
        and the SIGINT from Ctrl-C both go to the shell's group and not to this.
        """
        try:
            handle = self.log_path(job.id).open("ab")
        except OSError as exc:
            return self._refuse(job, f"cannot open the job log: {exc}")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "offset.core.jobs", "run", job.id],
                cwd=job.cwd or None,
                env=self._child_env(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            return self._refuse(job, f"cannot start the job process: {exc}")
        finally:
            handle.close()
        job.pid = proc.pid
        return self.save(job)

    def _child_env(self) -> dict[str, str]:
        """The environment a cold worker needs to find us.

        Two things are not inheritable by luck.  `OFFSET_JOBS` names the exact
        registry directory, because a store built on an explicit root is not
        `$OFFSET_HOME/jobs` and the child must not guess.  `PYTHONPATH` carries
        the directory `offset` lives in, because the job runs with the
        *workspace* as its cwd and the package is not necessarily installed.
        """
        env = dict(os.environ)
        env["OFFSET_JOBS"] = str(self.root)
        env["OFFSET_HOME"] = str(settings.home())
        package_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
        return env

    def _thread(self, job: Job) -> Job:
        cancel = threading.Event()
        with self._lock:
            self._cancels[job.id] = cancel
        # Claim the pid before the thread starts: a `reap()` between submission
        # and the worker's first write would otherwise see pid 0 and no
        # heartbeat, and settle a job that is about to run perfectly well.
        job.pid = os.getpid()
        self.save(job)
        worker = threading.Thread(target=self.run, args=(job.id,), kwargs={"cancel": cancel},
                                  name=f"job-{job.id[-8:]}", daemon=True)
        with self._lock:
            self._threads[job.id] = worker
        worker.start()
        return job

    def _refuse(self, job: Job, reason: str) -> Job:
        job.state, job.error, job.finished = FAILED, reason, time.time()
        return self.save(job)

    # -- doing the work ---------------------------------------------------

    def run(self, job_id: str, *, cancel: threading.Event | None = None) -> Job:
        """Execute one job, in whatever process calls this.

        Both entry points come through here: the daemon thread of an
        in-process job, and `python3 -m offset.core.jobs run <id>` from cold.
        """
        job = self.status(job_id)
        if job is None:
            return Job(id=job_id, state=FAILED, error=f"no job with id {job_id!r} in {self.root}")
        if job.state in TERMINAL:
            return job  # cancelled between submission and the worker starting
        job.state, job.started, job.pid = RUNNING, time.time(), os.getpid()
        self.save(job)

        flag = cancel if cancel is not None else threading.Event()
        stop = threading.Event()
        beat = threading.Thread(target=self._pulse, args=(job.id, stop), name=f"beat-{job.id[-8:]}", daemon=True)
        beat.start()
        outcome = JobOutcome()
        try:
            outcome = handler_for(job.kind)(job, self.workdir(job.id), flag)
        except Exception as exc:  # a broken handler fails its job, nothing else
            outcome = JobOutcome(error=f"the job crashed: {type(exc).__name__}: {exc}")
        finally:
            # Deliberately no `return` in here: a BaseException must keep
            # travelling, and the record it leaves saying `running` is exactly
            # what `reap()` exists to settle.
            stop.set()
        return self._finish(job, outcome)

    def _pulse(self, job_id: str, stop: threading.Event) -> None:
        path = self.beat_path(job_id)
        while not stop.is_set():
            try:
                path.touch()
            except OSError:
                return  # registry gone; `reap` will fall back to the pid check
            stop.wait(HEARTBEAT)

    def _finish(self, job: Job, outcome: JobOutcome) -> Job:
        current = self.status(job.id)
        # A cancel that landed while the handler was working is the user's last
        # word.  Keeping it matters: the handler's own view of a cancelled run
        # is "it stopped early", which reads as a bug rather than an intention.
        cancelled = current is not None and current.state == CANCELLED
        job.state = CANCELLED if cancelled else (FAILED if outcome.error else DONE)
        job.finished = time.time()
        job.result = outcome.text
        job.error = outcome.error or (current.error if cancelled and current else None)
        job.session = outcome.session or job.session
        job.usage = dict(outcome.usage)
        job.data = {**job.data, **outcome.data}
        with self._lock:
            self._cancels.pop(job.id, None)
            self._threads.pop(job.id, None)
        return self.save(job)

    # -- stopping ---------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop.  False means there was nothing to stop.

        The record is written *before* the signal, so a worker that dies to it
        cannot report its own version of events on the way out.
        """
        job = self.status(job_id)
        if job is None or job.state in TERMINAL:
            return False
        job.state, job.finished = CANCELLED, time.time()
        job.error = "cancelled"
        self.save(job)
        with self._lock:
            event = self._cancels.get(job_id)
        if event is not None:
            event.set()
        if job.pid and job.pid != os.getpid():
            _terminate(job.pid)
        return True

    def wait(self, job_id: str, timeout: float = 60.0) -> Job | None:
        """Block until the job settles, the timeout runs out, or its worker
        turns out to be gone.  None means there is no such job."""
        limit = time.monotonic() + max(0.0, timeout)
        next_reap = time.monotonic() + 1.0
        interval = 0.02
        while True:
            job = self.status(job_id)
            if job is None:
                return None
            if job.state in TERMINAL:
                return job
            now = time.monotonic()
            if now >= next_reap:
                # Waiting on a process that died is the one way this could hang
                # for ever, so the reaper runs inside the wait, not after it.
                self.reap()
                next_reap = now + 1.0
                continue
            if now >= limit:
                return job
            time.sleep(min(interval, max(0.005, limit - now)))
            interval = min(0.2, interval * 1.6)

    # -- reconciliation ---------------------------------------------------

    def reap(self) -> list[Job]:
        """Settle every record whose worker is gone, and return those records.

        This is what makes the file the authority rather than the process.  A
        machine that lost power mid-job, a `kill -9`, a parent that exited
        while an in-process job was on a daemon thread: all of them leave a
        record claiming to be live, and all of them are the same bug from the
        user's side — a job that never finishes.
        """
        settled: list[Job] = []
        for job in self.list():
            if job.state not in LIVE:
                continue
            reason = self._orphaned(job)
            if not reason:
                continue
            job.state, job.finished = FAILED, time.time()
            job.error = f"the job stopped without reporting: {reason}"
            settled.append(self.save(job))
        return settled

    def _orphaned(self, job: Job) -> str:
        """Why this live record has no worker any more, or "" if it has one."""
        if not _alive(job.pid):
            return f"its process (pid {job.pid or 'unknown'}) is gone"
        if job.state != RUNNING:
            return ""
        # The pid is alive, but pids are recycled: after a reboot this number
        # belongs to something else, and the record would say `running` for
        # ever. A worker touches its heartbeat every couple of seconds, so
        # silence this long means the number is not ours any more.
        try:
            last = self.beat_path(job.id).stat().st_mtime
        except OSError:
            last = job.started or job.created
        idle = time.time() - last
        return f"its heartbeat stopped {int(idle)}s ago" if idle > STALE_AFTER else ""

    def prune(self, *, keep: int = 200) -> int:
        """Drop the oldest settled jobs and their logs.  Live ones are safe."""
        settled = [j for j in self.list() if j.state in TERMINAL]
        doomed = settled[keep:]
        for job in doomed:
            for path in (self.record(job.id), self.log_path(job.id), self.beat_path(job.id)):
                try:
                    path.unlink()
                except OSError:
                    pass
            _remove_tree(self.workdir(job.id))
        return len(doomed)


# -- process helpers --------------------------------------------------------


def _alive(pid: int) -> bool:
    """Whether a process with this pid exists at all."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process, but it does exist
    except OSError:
        return True  # unknown failure: assume alive rather than kill a live job
    return True


def _terminate(pid: int) -> bool:
    """SIGTERM the job's whole process group.

    The group, not the process: `start_new_session=True` gave the worker a
    group of its own, and a detached job that left a `pytest` running is not
    cancelled while that pytest still holds the worktree.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def handler_for(kind: str) -> Handler:
    """Import and return a kind's worker.  Raises for an unknown kind, which
    `JobStore.run` turns into the job's error."""
    target = HANDLERS.get(kind)
    if target is None:
        raise ValueError(f"no job kind named {kind!r}; available: {', '.join(sorted(HANDLERS))}")
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    worker = getattr(module, attribute, None)
    if worker is None:
        raise ValueError(f"{target} does not exist; the {kind!r} job kind is misconfigured")
    return worker


def _remove_tree(path: Path) -> None:
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            child.rmdir() if child.is_dir() else child.unlink()
        except OSError:
            return
    try:
        path.rmdir()
    except OSError:
        pass


# -- formatting -------------------------------------------------------------


def ago(seconds: float) -> str:
    """A duration in the shortest form that is still honest."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86_400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86_400:.1f}d"


def _stamp(when: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when)) if when else "-"


def _indent(text: str, cap: int) -> list[str]:
    body = text if len(text) <= cap else text[:cap] + "\u2026"
    return [f"  {line}" for line in body.splitlines()] or ["  (empty)"]


# -- the process-wide store -------------------------------------------------
#
# One store, because the jobs directory is one directory.  The free function
# keeps it lazy: building it at import would resolve `settings.home()` before
# `--home` or a test had a chance to move it.

_active: JobStore | None = None
_active_lock = threading.Lock()


def store() -> JobStore:
    global _active
    with _active_lock:
        if _active is None:
            _active = JobStore()
        return _active


def install(state: Any) -> None:
    """Startup wiring: settle anything the last run left claiming to be alive.

    A session that ended with in-process jobs on daemon threads, or a machine
    that went down mid-job, leaves records saying `running`.  Doing this once at
    startup means `/jobs` is honest the first time it is asked, rather than
    after the user has waited on something that died yesterday.
    """
    reaped = store().reap()
    if reaped and getattr(state, "session", None) is not None:
        state.session.append("job_reaped", {"jobs": [j.id for j in reaped]}, parent=None)


# -- slash commands ---------------------------------------------------------


def _jobs(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Outcome

    box = store()
    box.reap()
    wanted: str | None = None
    if args:
        wanted = args[0].strip().lower()
        if wanted not in STATES:
            return Outcome.error(f"no job state named {wanted!r}", f"one of: {', '.join(STATES)}")
    jobs = box.list(state=wanted)
    if not jobs:
        return Outcome(
            [f"no {wanted} jobs" if wanted else "no background jobs",
             "background work is started by the task tool or /task, and lands in " + str(box.root)],
            TONE_INFO,
        )
    header = f"{'id':<9} {'state':<10} {'age':>7}  {'kind':<16} what"
    rows = [job.line(state.width if getattr(state, "width", 0) else 96) for job in jobs]
    live = sum(1 for j in jobs if j.live)
    tone = TONE_INFO if live else (TONE_ERR if any(j.state == FAILED for j in jobs) else TONE_OK)
    return Outcome([header, *rows, "", f"{len(jobs)} jobs, {live} still going; /job <id> for one of them"], tone)


def _job(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_ERR, TONE_INFO, Outcome

    box = store()
    box.reap()
    if not args:
        return Outcome.error("usage: /job <id>", "/jobs lists the ids")
    job = _find(box, args[0])
    if job is None:
        return Outcome.error(f"no job matching {args[0]!r}", "/jobs lists the ids")
    tail = box.logs(job.id, tail=20)
    lines = job.report()
    if tail:
        lines += ["", f"last {len(tail)} log lines:", *(f"  {line}" for line in tail)]
    return Outcome(lines, TONE_ERR if job.state == FAILED else TONE_INFO)


def _cancel(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_OK, Outcome

    box = store()
    if not args:
        return Outcome.error("usage: /cancel <id>", "/jobs lists the ids")
    job = _find(box, args[0])
    if job is None:
        return Outcome.error(f"no job matching {args[0]!r}", "/jobs lists the ids")
    if not box.cancel(job.id):
        return Outcome.error(f"job {job.id[-8:]} already finished ({job.state})",
                             "there is nothing left to stop; /job for its result")
    return Outcome([f"cancelled {job.id[-8:]}",
                    "a detached job was signalled; whatever it had written is kept"], TONE_OK)


def _find(box: JobStore, needle: str) -> Job | None:
    """Accept a whole id or any unambiguous tail of one: the `/jobs` table
    shows the last eight characters, and typing 26 is not a user interface."""
    exact = box.status(needle)
    if exact is not None:
        return exact
    matches = [j for j in box.list() if j.id.endswith(needle) or j.id.startswith(needle)]
    return matches[0] if len(matches) == 1 else None


def _commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("jobs", "background jobs and how they ended", _jobs, usage="/jobs [state]"),
        Command("job", "one job in full, with its log tail", _job, usage="/job <id>"),
        Command("cancel", "stop a background job", _cancel, usage="/cancel <id>"),
    ]


COMMANDS: list[Any] = _commands()


# -- the cold entry point ---------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python3 -m offset.core.jobs run <job_id>` — one job, from nothing.

    This is the whole reason a job is a file.  Nothing is inherited from the
    process that submitted the work except two environment variables, so the
    terminal can be closed the moment after `Popen` returns.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "run":
        print("usage: python3 -m offset.core.jobs run <job_id>", file=sys.stderr)
        return 2
    box = JobStore(os.environ.get("OFFSET_JOBS") or None)
    cancel = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        # Cooperative: the record already says cancelled, so all this has to do
        # is let the handler unwind and write down what it had got.
        cancel.set()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, stop)
        except (OSError, ValueError):
            pass  # not the main thread, or the platform has no such signal
    job = box.run(args[1], cancel=cancel)
    print(f"job {job.id} -> {job.state}" + (f": {job.error}" if job.error else ""))
    return 0 if job.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
