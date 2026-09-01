"""Real-time multiplayer: one session, several humans, one wheel.

Pairing with an agent falls apart the moment a second person wants to watch.
Screen sharing shows one person's terminal at one person's latency and gives
the guest no way to act; giving everybody their own shell gives everybody
their own session, which is emphatically not the same conversation.  So the
session stays exactly where it is and a room publishes it, in the same way the
editor bridge publishes it to editors.

This module is deliberately the bridge's sibling.  It imports the bridge's
framing, its error codes, its token discipline and its socket helpers rather
than restating them: two hand-rolled JSON-RPC dialects in one process drift
apart, and the day they do, one of them is subtly wrong and nobody knows which.
Newline-delimited JSON-RPC 2.0, requests carrying an `id`, events pushed as
notifications without one — the same wire an extension already speaks.

Four decisions carry the design.

*Authority is not a lock.*  The host may always drive, so the host does not
hold the wheel; the wheel is a token at most one *peer* may hold, and the host
prompts regardless of who has it.  Modelling the host as the default holder
would mean a guest could only drive after the host released, and a host that
forgot to release — or crashed mid-turn — would leave a room nobody could use.
Reclaiming is therefore unconditional and instant, and the wheel comes back on
its own when the driver disconnects, because otherwise one closed laptop lid
would strand the room forever.

*A second claim is answered, never queued.*  Telling somebody "you are number
two" invites them to wait for a promotion that may never arrive, and a queued
claim that fires ten minutes later hands the wheel to somebody who has walked
away.  A refused claim names who is driving, which is the only thing the asker
actually needs in order to go and ask that person.

*Chat is not a prompt.*  `say` reaches humans, `prompt` reaches the model.
Folding them together is the failure this separation exists to prevent: an
observer typing "no, not that file" as an aside would become an instruction to
the agent, and the room would have no way to tell the two apart afterwards.
Observers may always talk and may never prompt.

*A slow peer must not stall the room.*  Every peer gets its own reader thread,
its own writer thread and its own bounded queue; broadcasting is `put_nowait`
into each queue and nothing else.  A peer that stops reading fills its queue
and is dropped with a counted reason.  Backpressure is not propagated — the
alternative is an agent that pauses mid-turn because somebody's laptop went to
sleep — and the count is what keeps a room that drops everybody visible rather
than mysteriously empty.

Domain failures are results, not JSON-RPC errors, as in the bridge: prompting
without the wheel answers `{"ok": false, "error": ...}`.  The `error` member is
reserved for protocol faults — bad JSON, unknown method, missing token — so a
client can tell "you asked wrongly" from "that could not be done".
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from offset.core import settings
from offset.core.agent import Finished, StepStarted, ToolFinished, ToolStarted
from offset.core.bridge import (
    AUTH_GRACE,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MAX_FRAME,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL,
    SEND_TIMEOUT,
    UNAUTHENTICATED,
    _TICK,
    _encode,
    _failure,
    _hangup,
    _reply,
    _write_secret,
    _alive,
)
from offset.core.entries import new_id
from offset.providers.base import StreamError

#: Bumped when the wire contract changes in a way an older client would
#: misread.  Separate from the bridge's version: the two publish different
#: method sets and a client that speaks one does not speak the other.
COLLAB_VERSION: Final = "1"

SOCKET_NAME: Final = "collab.sock"
TOKEN_NAME: Final = "collab.token"
DESCRIPTOR_NAME: Final = "collab.json"

#: Events a peer may fall behind by before it is dropped.  A turn with a lot of
#: tool calls plus a chatty room is well under this, so a peer merely repainting
#: survives; a peer that has stopped reading cannot pin megabytes of history.
QUEUE_LIMIT: Final = 256

#: Humans in one room.  Past this it is a reconnect loop, not a mob programming
#: session, and every extra peer costs two threads.
MAX_PEERS: Final = 16

#: Longest display name kept.  Names are attacker-supplied strings that get
#: broadcast to everybody and printed in a roster; without a ceiling one peer
#: could push a megabyte of name through every other peer's terminal.
MAX_NAME: Final = 24

#: Longest chat line relayed.  Same reasoning as the name: chat fans out to
#: every peer, so its size is everyone's problem, not just the sender's.
MAX_SAY: Final = 2000

#: Seconds a `call` waits for its reply before giving up.  Generous because a
#: `prompt` reply arrives only when the whole turn has finished.
CALL_TIMEOUT: Final = 600.0

#: Seconds to spend connecting and authenticating.  Short: a room that is not
#: answering now will not start answering, and the shell is blocked meanwhile.
JOIN_TIMEOUT: Final = 10.0

#: Roles.  `host` is not a peer role — the host is the process, not a
#: connection — but the roster reports it so a guest can see who they joined.
HOST: Final = "host"
DRIVER: Final = "driver"
OBSERVER: Final = "observer"

JOINED: Final = "peer.joined"
LEFT: Final = "peer.left"
TURN_STARTED: Final = "turn.started"
TURN_FINISHED: Final = "turn.finished"
TOOL_STARTED: Final = "tool.started"
TOOL_FINISHED: Final = "tool.finished"
DRIVER_CHANGED: Final = "driver.changed"
SAID: Final = "chat.said"
CLOSING: Final = "room.closing"

#: Every notification a room will ever push, declared so a client can be
#: written against a closed set rather than against whatever it happens to see.
EVENTS: Final = (
    JOINED,
    LEFT,
    TURN_STARTED,
    TURN_FINISHED,
    TOOL_STARTED,
    TOOL_FINISHED,
    DRIVER_CHANGED,
    SAID,
    CLOSING,
)

#: Methods a peer may call once authenticated.  `hello` is answered before
#: authentication and is therefore deliberately absent from the dispatch table.
METHODS: Final = ("roster", "status", "say", "drive", "prompt", "leave")


class CollabError(RuntimeError):
    """Anything that stopped a room operation from happening."""


class JoinError(CollabError):
    """The room could not be reached, or refused this peer."""


class RpcError(CollabError):
    """A protocol fault came back on the wire."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# -- what a room needs from a shell -----------------------------------------
#
# Injected rather than reached for, so the tests drive real sockets without a
# model, a provider or a session behind them.


def _no_agent(text: str, who: str) -> tuple[bool, str]:
    return False, "no agent is attached to this room"


def _no_model() -> str:
    return ""


@dataclass(slots=True)
class RoomHooks:
    """Everything a room needs from the process hosting it."""

    workspace: Path = field(default_factory=Path.cwd)
    #: How the host appears in the roster.  Not the login name by default: a
    #: room is a place, and "host" reads better than "pi" in a chat log.
    name: str = HOST
    #: Drives one real turn.  Takes the text and who asked for it, because the
    #: shell may want to attribute a guest's prompt in the transcript.
    prompt: Callable[[str, str], tuple[bool, str]] = _no_agent
    model: Callable[[], str] = _no_model


# -- addresses ---------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Address:
    """Where a room is, and the token needed to enter it."""

    kind: str
    path: str = ""
    host: str = ""
    port: int = 0
    token: str = ""

    def __str__(self) -> str:
        return self.path if self.kind == "unix" else f"{self.host}:{self.port}"


def parse_address(spec: str) -> Address:
    """Read `path`, `host:port`, or either with `#token` appended.

    The token may travel with the address because that is how an address gets
    shared in practice — pasted into a chat window as one string — and a user
    who has to paste two things will paste one of them wrongly.
    """
    body, _, token = spec.strip().partition("#")
    body, token = body.strip(), token.strip()
    if not body:
        raise JoinError("a room address is needed: a socket path or host:port")
    if "/" in body or body.endswith(".sock"):
        return Address("unix", path=body, token=token)
    host, sep, port = body.rpartition(":")
    if not sep:
        raise JoinError(f"{body!r} is not a socket path and has no port; use host:port")
    try:
        number = int(port)
    except ValueError:
        raise JoinError(f"{port!r} is not a port number") from None
    return Address("tcp", host=host or "127.0.0.1", port=number, token=token)


def read_descriptor(home: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """What a local peer does to find the room on this machine.

    Returns `(descriptor, problem)`.  The token is read from the descriptor's
    own `token_path` rather than embedded in it, so the discovery file can stay
    readable for diagnosis while the secret stays `0o600`.
    """
    base = Path(home) if home is not None else settings.home()
    target = base / DESCRIPTOR_NAME
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"no room is published here ({target} is absent)"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{target}: {exc}"
    if not isinstance(raw, dict):
        return None, f"{target} does not contain a JSON object"
    return raw, None


# -- one connected human ----------------------------------------------------


class Peer:
    """A socket, a name, a bounded queue and the two threads either side of it.

    The queue is the whole point.  `Room.broadcast` runs on whichever thread
    happened to cause the event — often the agent's — and must never block, so
    it only ever puts; the writer thread is the only code that touches the
    socket for output and is allowed to be as slow as the peer is.
    """

    __slots__ = (
        "authenticated",
        "drained",
        "dropped",
        "id",
        "name",
        "outbox",
        "reader",
        "retired",
        "since",
        "sock",
        "writer",
    )

    def __init__(self, sock: socket.socket, limit: int) -> None:
        self.id = new_id()
        self.sock = sock
        self.name = ""
        self.outbox: queue.Queue[bytes | None] = queue.Queue(maxsize=max(1, limit))
        self.authenticated = False
        self.dropped: str | None = None
        self.retired = False
        self.since = time.time()
        self.reader: threading.Thread | None = None
        self.writer: threading.Thread | None = None
        #: Set whenever the writer empties the queue.  A teardown that wants a
        #: peer to actually receive `room.closing` waits on this instead of
        #: sleeping and hoping: closing the socket first loses the frame and
        #: the peer reports a bare EOF, which tells the human nothing.
        self.drained = threading.Event()
        self.drained.set()

    def send(self, frame: bytes) -> bool:
        """Queue a frame.  False means too far behind to keep."""
        if self.dropped is not None:
            return False
        self.drained.clear()
        try:
            self.outbox.put_nowait(frame)
        except queue.Full:
            return False
        return True

    def payload(self, driver_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": DRIVER if driver_id == self.id else OBSERVER,
            "since": self.since,
            "queued": self.outbox.qsize(),
        }


# -- the room ---------------------------------------------------------------


class Room:
    """One published session, and every human currently in it."""

    __slots__ = (
        "_accept",
        "_driver",
        "_lock",
        "_methods",
        "_owns_files",
        "_peers",
        "_server",
        "_stop",
        "_turn",
        "bind_host",
        "descriptor_path",
        "drops",
        "dropped",
        "home",
        "hooks",
        "listen",
        "max_peers",
        "port",
        "problems",
        "queue_limit",
        "send_timeout",
        "socket_path",
        "started",
        "token",
        "token_path",
    )

    def __init__(
        self,
        hooks: RoomHooks | None = None,
        *,
        home: str | os.PathLike[str] | None = None,
        listen: str = "",
        queue_limit: int = QUEUE_LIMIT,
        send_timeout: float = SEND_TIMEOUT,
        max_peers: int = MAX_PEERS,
    ) -> None:
        self.hooks = hooks or RoomHooks()
        #: Resolved now rather than at import: `--home` and the tests both move
        #: it after this module has loaded.
        self.home = Path(home) if home is not None else settings.home()
        self.socket_path = self.home / SOCKET_NAME
        self.token_path = self.home / TOKEN_NAME
        self.descriptor_path = self.home / DESCRIPTOR_NAME
        self.queue_limit = max(1, queue_limit)
        self.send_timeout = max(0.05, send_timeout)
        self.max_peers = max(1, max_peers)
        #: Empty means a unix socket, which is right for two people sharing one
        #: machine over ssh.  `tcp` or `host:port` forces a socket somebody
        #: else's laptop can reach — the whole point of a room — and the token
        #: is what makes that safe, so it is never optional.  Binding beyond
        #: loopback stays a deliberate act.
        self.listen = listen.strip()
        self.token = ""
        self.bind_host = ""
        self.port = 0
        self.started = 0.0
        #: Peers hung up on, counted rather than raised: one guest falling
        #: behind is a fact about that guest, but a room that drops everybody
        #: needs to be visible in `/collab`.
        self.dropped = 0
        #: The reasons, newest last and bounded.  Kept because the count alone
        #: cannot distinguish "the network died" from "one laptop slept".
        self.drops: list[str] = []
        self.problems: list[str] = []
        self._server: socket.socket | None = None
        self._accept: threading.Thread | None = None
        self._peers: dict[str, Peer] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        #: Empty string means nobody holds the wheel, which is the state in
        #: which the host is the only one who can prompt.  See the module
        #: docstring: the host is never the holder.
        self._driver = ""
        #: Held while a turn this room started is running.  Non-blocking
        #: acquisition is how a second prompt is told "already busy" rather
        #: than being queued behind a turn it cannot see.
        self._turn = threading.Lock()
        self._owns_files = False
        self._methods: dict[str, Callable[[Peer, dict[str, Any]], Any]] = {
            "roster": self._m_roster,
            "status": self._m_status,
            "say": self._m_say,
            "drive": self._m_drive,
            "prompt": self._m_prompt,
            "leave": self._m_leave,
        }

    # -- lifecycle ----------------------------------------------------------

    @property
    def unix(self) -> bool:
        """Whether this room is on a unix domain socket.

        False on Windows, which has none with a filesystem mode, and false
        whenever a listen address was asked for: a peer on another machine
        cannot reach a socket file on this one.
        """
        return hasattr(socket, "AF_UNIX") and not self.listen

    @property
    def listening(self) -> bool:
        return self._server is not None and not self._stop.is_set()

    @property
    def host_name(self) -> str:
        return _clean_name(self.hooks.name) or HOST

    def address(self) -> str:
        """What to paste to somebody who wants to join, token included."""
        if not self.listening:
            return ""
            
        where = str(self.socket_path) if self.unix else f"{self.bind_host or '127.0.0.1'}:{self.port}"
        return f"{where}#{self.token}"

    def serve(self) -> list[str]:
        """Bind, publish the descriptor, and accept in a daemon thread.

        Returns the reasons it could not start, empty on success.  A room is a
        convenience: a shell whose room will not bind must still run, so this
        never raises.
        """
        if self._server is not None:
            return []
        self.problems = []
        try:
            self.home.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.problems = [f"{self.home}: {type(exc).__name__}: {exc}"]
            return list(self.problems)

        server, problem = self._bind()
        if server is None:
            self.problems = [problem or "the room socket could not be bound"]
            return list(self.problems)

        self.token = secrets.token_urlsafe(32)
        problem = self._publish_descriptor()
        if problem:
            server.close()
            self.problems = [problem]
            return list(self.problems)

        self._server = server
        self._stop.clear()
        self.started = time.time()
        self._owns_files = True
        self._accept = threading.Thread(target=self._accept_loop, name="offset-collab", daemon=True)
        self._accept.start()
        return []

    def _address(self) -> tuple[str, int]:
        """The host and port a TCP room should bind.

        `""` and `tcp` both mean loopback on an ephemeral port; anything else is
        `host`, `host:port` or `:port`.  A host that is not loopback is the user
        deliberately inviting their network in.
        """
        spec = self.listen
        if not spec or spec == "tcp":
            return "127.0.0.1", 0
        host, sep, port = spec.rpartition(":")
        if not sep:
            return spec, 0
        try:
            return (host or "127.0.0.1"), int(port)
        except ValueError:
            # `[::1]` and other bracketless colons: treat the whole thing as a
            # host rather than silently binding a port nobody asked for.
            return spec, 0

    def _bind(self) -> tuple[socket.socket | None, str | None]:
        if not self.unix:
            host, port = self._address()
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            server = socket.socket(family, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((host, port))
                server.listen(self.max_peers)
            except OSError as exc:
                server.close()
                return None, f"{host}:{port}: {type(exc).__name__}: {exc}"
            bound = server.getsockname()
            self.bind_host, self.port = host, int(bound[1])
            server.settimeout(_TICK)
            return server, None

        path = str(self.socket_path)
        # 108 on Linux, 104 on BSD; both are the size of sockaddr_un.sun_path.
        if len(path.encode("utf-8")) >= 104:
            return None, (
                f"{path} is too long for a unix socket path; "
                "set OFFSET_HOME to a shorter directory"
            )
        if self.socket_path.exists():
            if _alive(self.socket_path):
                return None, f"a room is already published on {path}"
            # Stale: whatever made it is gone, so the file is rubbish.  Probed
            # before unlinking, because deleting first would let a second shell
            # silently steal the first shell's guests.
            try:
                self.socket_path.unlink()
            except OSError as exc:
                return None, f"{path} is stale but could not be removed: {exc}"

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Created under a restrictive umask rather than chmod'ed afterwards:
        # between bind and chmod the socket is connectable by anyone.
        previous = os.umask(0o177)
        try:
            server.bind(path)
            server.listen(self.max_peers)
        except OSError as exc:
            server.close()
            return None, f"{path}: {type(exc).__name__}: {exc}"
        finally:
            os.umask(previous)
        try:
            os.chmod(path, 0o600)  # belt and braces: some filesystems ignore umask
        except OSError:
            pass
        server.settimeout(_TICK)
        return server, None

    def _publish_descriptor(self) -> str | None:
        """Write the token and the discovery file, both `0o600`."""
        descriptor = {
            "version": COLLAB_VERSION,
            "protocol": PROTOCOL,
            "transport": "unix" if self.unix else "tcp",
            "path": str(self.socket_path) if self.unix else "",
            "host": "" if self.unix else (self.bind_host or "127.0.0.1"),
            "port": self.port,
            "token_path": str(self.token_path),
            "host_name": self.host_name,
            "pid": os.getpid(),
            "started": time.time(),
            "events": list(EVENTS),
            "methods": list(METHODS),
        }
        for target, body in (
            (self.token_path, self.token),
            (self.descriptor_path, json.dumps(descriptor, indent=2)),
        ):
            problem = _write_secret(target, body)
            if problem:
                return problem
        return None

    def close(self) -> None:
        """Announce, let it flush, hang up on everyone, remove the files.

        The announcement goes out before the socket closes and the peers are
        given until `send_timeout` to receive it, because the difference between
        "the host closed the room" and an unexplained EOF is the difference
        between a guest reconnecting and a guest filing a bug.  Nothing here is
        counted as a drop: shutting down is not a peer's fault.
        """
        if self._server is None and not self._peers:
            return
        peers = self._snapshot()
        if peers:
            self.broadcast(CLOSING, {"reason": "the host closed the room"})
            deadline = time.monotonic() + self.send_timeout
            for peer in peers:
                peer.drained.wait(max(0.0, deadline - time.monotonic()))

        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        for peer in peers:
            self._retire(peer, "the room closed", count=False)
        accept, self._accept = self._accept, None
        if accept is not None and accept is not threading.current_thread():
            accept.join(timeout=2.0)
        if self._owns_files:
            self._owns_files = False
            for path in (self.socket_path, self.token_path, self.descriptor_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.token = ""
        self._driver = ""

    # -- presence -----------------------------------------------------------

    def _snapshot(self) -> list[Peer]:
        with self._lock:
            return list(self._peers.values())

    @property
    def driver_id(self) -> str:
        return self._driver

    def driver_name(self) -> str:
        """Who may prompt besides the host.  The host's own name when nobody
        holds the wheel, because "nobody" is not a useful answer to "who is
        driving?" — the host is."""
        with self._lock:
            holder = self._peers.get(self._driver) if self._driver else None
        return holder.name if holder is not None else self.host_name

    def roster(self) -> list[dict[str, Any]]:
        """Everybody in the room, the host first."""
        driver = self._driver
        rows: list[dict[str, Any]] = [
            {
                "id": "",
                "name": self.host_name,
                "role": HOST,
                "since": self.started,
                "queued": 0,
            }
        ]
        rows.extend(peer.payload(driver) for peer in self._snapshot() if peer.authenticated)
        return rows

    def _unique_name(self, wanted: str) -> str:
        """A name nobody else in the room is already using.

        Two guests called `bob` make a roster and a chat log that cannot be
        read, so the second one becomes `bob (2)`.  Renaming the newcomer
        rather than refusing the join is the kinder failure: nobody is locked
        out of a session because a colleague picked the same shell prompt.
        """
        base = _clean_name(wanted) or "guest"
        taken = {peer.name for peer in self._snapshot()}
        if base not in taken:
            return base
        for suffix in range(2, self.max_peers + 2):
            candidate = f"{base} ({suffix})"
            if candidate not in taken:
                return candidate
        return f"{base} ({new_id()[-4:]})"

    # -- the wheel ----------------------------------------------------------

    def claim(self, peer_id: str) -> tuple[bool, str]:
        """Promote one peer to driver.  Never queues a refusal."""
        with self._lock:
            peer = self._peers.get(peer_id)
            if peer is None or not peer.authenticated:
                return False, "you are not in this room"
            if self._driver == peer_id:
                return True, "you already have the wheel"
            holder = self._peers.get(self._driver) if self._driver else None
            if holder is not None:
                held = holder.name
            else:
                self._driver = peer_id
                held = ""
        if held:
            return False, f"{held} is driving; ask them to hand over"
        self.broadcast(DRIVER_CHANGED, {"driver": peer_id, "name": peer.name, "reason": "claimed the wheel"})
        return True, "you have the wheel"

    def reclaim(self) -> tuple[bool, str]:
        """Take the wheel back for the host.  Always succeeds — see the module
        docstring: the host's authority is not a lock, so it cannot be lost."""
        with self._lock:
            holder = self._peers.get(self._driver) if self._driver else None
            self._driver = ""
        if holder is None:
            return True, "you already had the wheel"
        self.broadcast(
            DRIVER_CHANGED,
            {"driver": "", "name": self.host_name, "reason": f"the host took the wheel from {holder.name}"},
        )
        return True, f"the wheel is back with {self.host_name}"

    def may_drive(self, peer_id: str) -> bool:
        return bool(peer_id) and self._driver == peer_id

    # -- accepting ----------------------------------------------------------

    def _accept_loop(self) -> None:
        server = self._server
        while not self._stop.is_set() and server is not None:
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                # Not a sleep for pacing's sake: a listening socket that errors
                # without being closed would otherwise spin a core flat.
                self._stop.wait(_TICK)
                continue
            self._adopt(conn)

    def _adopt(self, conn: socket.socket) -> None:
        peer = Peer(conn, self.queue_limit)
        conn.settimeout(self.send_timeout)
        with self._lock:
            over = len(self._peers) >= self.max_peers
            if not over:
                self._peers[peer.id] = peer
        if over:
            # Answered rather than dropped silently, so a reconnect loop shows
            # its reason in the guest's terminal instead of looking like a
            # crash on the host.
            try:
                conn.sendall(_failure(None, INTERNAL_ERROR, f"the room already has {self.max_peers} peers"))
            except OSError:
                pass
            _hangup(conn)
            return
        tag = peer.id[-6:]
        peer.writer = threading.Thread(target=self._write_loop, args=(peer,), name=f"collab-tx-{tag}", daemon=True)
        peer.reader = threading.Thread(target=self._read_loop, args=(peer,), name=f"collab-rx-{tag}", daemon=True)
        peer.writer.start()
        peer.reader.start()

    def _retire(self, peer: Peer, reason: str, *, count: bool) -> tuple[bool, bool]:
        """Remove a peer and unblock both of its threads.

        Returns `(first, surrendered)`: whether this call was the one that
        retired it, and whether it was holding the wheel.  Idempotent and the
        only place `dropped` is counted, because two threads routinely notice
        the same dead socket at once — the writer's `sendall` fails while the
        reader sees EOF — and a room that counted both would report twice the
        departures that happened.
        """
        with self._lock:
            first = not peer.retired
            peer.retired = True
            if first:
                peer.dropped = peer.dropped or reason
                if count:
                    self.dropped += 1
                    self.drops.append(f"{peer.name or peer.id[-6:]}: {reason}")
                    del self.drops[: max(0, len(self.drops) - 2 * self.max_peers)]
            self._peers.pop(peer.id, None)
            surrendered = first and bool(self._driver) and self._driver == peer.id
            if surrendered:
                # The wheel comes home by itself.  Without this a driver whose
                # laptop slept would leave a room in which nobody could prompt.
                self._driver = ""
        try:
            peer.outbox.put_nowait(None)  # wake the writer even if it is idle
        except queue.Full:
            pass
        _hangup(peer.sock)
        return first, surrendered

    def _evict(self, peer: Peer, reason: str, *, count: bool = True) -> None:
        """Retire a peer and tell the room, in that order.

        The order matters: `_retire` removes the peer before the announcement
        goes out, so the `peer.left` broadcast cannot try to deliver itself to
        the peer that just left and drop it a second time.
        """
        first, surrendered = self._retire(peer, reason, count=count)
        if not first:
            return
        if peer.authenticated:
            self.broadcast(LEFT, {"peer": peer.id, "name": peer.name, "reason": reason})
        if surrendered:
            self.broadcast(
                DRIVER_CHANGED,
                {"driver": "", "name": self.host_name, "reason": f"{peer.name} left with the wheel"},
            )

    # -- reading ------------------------------------------------------------

    def _read_loop(self, peer: Peer) -> None:
        buf = bytearray()
        deadline = time.monotonic() + AUTH_GRACE
        try:
            while not self._stop.is_set() and peer.dropped is None:
                try:
                    chunk = peer.sock.recv(65536)
                except TimeoutError:
                    if not peer.authenticated and time.monotonic() > deadline:
                        self._reject(peer, None, "no hello frame arrived; this room requires a token")
                        return
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                if len(buf) > MAX_FRAME:
                    self._reject(peer, None, f"a single frame exceeded {MAX_FRAME} bytes")
                    return
                while (nl := buf.find(b"\n")) >= 0:
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if line.strip():
                        self._handle(peer, line)
                    if peer.dropped is not None:
                        return
        finally:
            self._evict(peer, peer.dropped or "disconnected", count=peer.dropped is not None)

    def _handle(self, peer: Peer, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            peer.send(_failure(None, PARSE_ERROR, f"frame is not valid JSON: {exc}"))
            return
        if not isinstance(message, dict):
            peer.send(_failure(None, INVALID_REQUEST, "every frame must be a JSON object"))
            return

        ident = message.get("id")
        method = message.get("method")
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(method, str) or not method:
            peer.send(_failure(ident, INVALID_REQUEST, "every request needs a string 'method'"))
            return
        if not isinstance(params, dict):
            peer.send(_failure(ident, INVALID_PARAMS, f"params for {method} must be a JSON object"))
            return

        if not peer.authenticated:
            if method != "hello":
                self._reject(peer, ident, f"{method} was sent before hello; authenticate first")
                return
            offered = params.get("token")
            # `compare_digest` against a fresh random string when the room has
            # no token: a room that is shutting down must not accept everybody.
            if not isinstance(offered, str) or not secrets.compare_digest(offered, self.token or new_id()):
                self._reject(peer, ident, "the token does not match this room")
                return
            peer.name = self._unique_name(str(params.get("name") or ""))
            peer.authenticated = True
            peer.send(_reply(ident, self._greeting(peer)))
            self.broadcast(JOINED, {"peer": peer.id, "name": peer.name, "role": OBSERVER})
            return

        if method == "hello":
            peer.send(_reply(ident, self._greeting(peer)))
            return

        handler = self._methods.get(method)
        if handler is None:
            peer.send(_failure(
                ident,
                METHOD_NOT_FOUND,
                f"no method named {method!r}. available: " + ", ".join(sorted(self._methods)),
            ))
            return
        try:
            result = handler(peer, params)
        except Exception as exc:  # a broken handler must not kill the connection
            peer.send(_failure(ident, INTERNAL_ERROR, f"{method} failed: {type(exc).__name__}: {exc}"))
            return
        if ident is not None:
            peer.send(_reply(ident, result))

    def _reject(self, peer: Peer, ident: Any, why: str) -> None:
        """One error frame, then the connection goes.

        A socket that can drive an agent does not get to retry its way in.  The
        frame is flushed first — waiting on the writer's own signal rather than
        guessing at a delay — because a guest who sees only EOF has no idea
        whether they typed the token wrongly or the host is down.
        """
        peer.send(_failure(ident, UNAUTHENTICATED, why))
        peer.drained.wait(min(1.0, self.send_timeout))
        self._evict(peer, why)

    def _greeting(self, peer: Peer) -> dict[str, Any]:
        return {
            "ok": True,
            "version": COLLAB_VERSION,
            "protocol": PROTOCOL,
            "peer": peer.id,
            "name": peer.name,
            "role": DRIVER if self.may_drive(peer.id) else OBSERVER,
            "driver": self.driver_name(),
            "host": self.host_name,
            "workspace": str(self.hooks.workspace),
            "events": list(EVENTS),
            "methods": list(METHODS),
        }

    # -- writing ------------------------------------------------------------

    def _write_loop(self, peer: Peer) -> None:
        while True:
            try:
                frame = peer.outbox.get(timeout=_TICK)
            except queue.Empty:
                peer.drained.set()
                if self._stop.is_set() or peer.dropped is not None:
                    return
                continue
            if frame is None:
                peer.drained.set()
                return
            try:
                peer.sock.sendall(frame)
            except (TimeoutError, OSError) as exc:
                # A peer that is not reading blocks here until the deadline;
                # that deadline is the second half of the drop policy, the
                # first being a queue that fills.
                self._evict(peer, f"send failed after {self.send_timeout:g}s: {type(exc).__name__}")
                return
            if peer.outbox.empty():
                peer.drained.set()

    # -- events -------------------------------------------------------------

    def broadcast(self, event: str, payload: dict[str, Any] | None = None) -> int:
        """Push a notification to every authenticated peer.

        Never blocks and never raises: this runs on the agent's own thread.
        Returns how many peers it reached, which is what makes "nobody is
        watching" cheap to detect.
        """
        peers = self._snapshot()
        if not peers:
            return 0
        frame = _encode({
            "jsonrpc": PROTOCOL,
            "method": event,
            "params": {"event": event, "at": time.time(), **(payload or {})},
        })
        delivered = 0
        behind: list[Peer] = []
        for peer in peers:
            if not peer.authenticated or peer.dropped is not None:
                continue
            if peer.send(frame):
                delivered += 1
            else:
                behind.append(peer)
        # Evicted after the loop, not during it, so one slow peer's departure
        # notice does not interleave with the delivery it interrupted.
        for peer in behind:
            self._evict(peer, f"fell more than {self.queue_limit} events behind")
        return delivered

    def chat(self, text: str, *, name: str = "") -> int:
        """The host says something to the humans.  Reaches nobody's model."""
        body = _clip(text, MAX_SAY)
        if not body:
            return 0
        return self.broadcast(SAID, {"from": "", "name": name or self.host_name, "text": body})

    def observe(self, event: Any) -> None:
        """Mirror one agent-loop event into the room.

        Tool traffic is what makes an observer's screen worth looking at, so it
        is relayed verbatim; the turn boundaries come from `run_turn` instead,
        because a room needs to name whose prompt it was and an agent event
        does not know.
        """
        if isinstance(event, ToolStarted):
            self.broadcast(TOOL_STARTED, {"id": event.call.id, "tool": event.call.name, "args": event.call.args})
        elif isinstance(event, ToolFinished):
            inv = event.invocation
            self.broadcast(TOOL_FINISHED, {
                "id": inv.call.id,
                "tool": inv.call.name,
                "ok": inv.result.ok,
                "error": inv.result.error,
                "summary": inv.result.display or inv.result.content[:200],
                "duration": round(inv.result.duration, 4),
            })
        elif isinstance(event, StepStarted):
            self.broadcast(TURN_STARTED, {"step": event.index, "model": event.model, "by": ""})
        elif isinstance(event, Finished):
            self.broadcast(TURN_FINISHED, {"ok": True, "reason": event.reason, "steps": event.steps, "text": event.text})
        elif isinstance(event, StreamError):
            self.broadcast(TURN_FINISHED, {"ok": False, "reason": "error", "steps": 0, "text": "", "error": event.message})

    def run_turn(self, text: str, who: str) -> dict[str, Any]:
        """Drive one turn and bracket it with events everybody can see.

        The brackets are emitted here rather than derived from agent events so
        that a room still shows a turn starting and finishing when the hook
        behind it is a queue, a stub or a model that failed before emitting
        anything at all.
        """
        body = text.strip()
        if not body:
            return {"ok": False, "error": "a prompt needs some text"}
        if not self._turn.acquire(blocking=False):
            return {"ok": False, "error": "a turn is already running in this room"}
        self.broadcast(TURN_STARTED, {"by": who, "text": _clip(body, 200)})
        try:
            ok, reply = self.hooks.prompt(body, who)
        except Exception as exc:  # a provider blowing up is a result, not a fault
            ok, reply = False, f"{type(exc).__name__}: {exc}"
        finally:
            self._turn.release()
        self.broadcast(TURN_FINISHED, {
            "by": who,
            "ok": ok,
            "text": reply if ok else "",
            "error": "" if ok else reply,
        })
        return {"ok": True, "text": reply} if ok else {"ok": False, "error": reply}

    def prompt(self, text: str) -> dict[str, Any]:
        """The host prompts.  Never refused: the host is not a wheel holder."""
        return self.run_turn(text, self.host_name)

    # -- methods ------------------------------------------------------------

    def _m_roster(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        return {"peers": self.roster(), "driver": self.driver_name(), "host": self.host_name}

    def _m_status(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": COLLAB_VERSION,
            "workspace": str(self.hooks.workspace),
            "model": self.hooks.model(),
            "host": self.host_name,
            "driver": self.driver_name(),
            "you": peer.name,
            "role": DRIVER if self.may_drive(peer.id) else OBSERVER,
            "peers": len(self._peers),
            "dropped": self.dropped,
            "busy": self._turn.locked(),
            "started": self.started,
        }

    def _m_say(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "say needs a non-empty 'text'"}
        body = _clip(text, MAX_SAY)
        # Echoed to the sender as well.  A chat client that has to fake its own
        # messages locally shows them in a different order from everybody else.
        heard = self.broadcast(SAID, {"from": peer.id, "name": peer.name, "text": body})
        return {"ok": True, "heard": heard}

    def _m_drive(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        ok, message = self.claim(peer.id)
        if ok:
            return {"ok": True, "role": DRIVER, "message": message}
        return {"ok": False, "error": message, "driver": self.driver_name()}

    def _m_prompt(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "prompt needs a non-empty 'text'"}
        if not self.may_drive(peer.id):
            # The refusal names the holder, because "ask them" is the only
            # useful next step and the asker cannot work it out otherwise.
            return {
                "ok": False,
                "error": f"you are observing; {self.driver_name()} has the wheel. "
                         "use say to talk to the humans, or ask for the wheel",
                "driver": self.driver_name(),
            }
        wait = params.get("wait")
        wait = True if wait is None else bool(wait)
        if not wait:
            threading.Thread(
                target=self.run_turn, args=(text, peer.name), name="collab-turn", daemon=True
            ).start()
            return {"ok": True, "accepted": True, "text": ""}
        return self.run_turn(text, peer.name)

    def _m_leave(self, peer: Peer, params: dict[str, Any]) -> dict[str, Any]:
        # Answered before the eviction so the reply is in the queue ahead of the
        # hangup; a clean goodbye is how a guest knows they left rather than
        # dropped.  The eviction itself happens on the reader's way out, which
        # `dropped` being set forces.
        peer.dropped = "left the room"
        return {"ok": True, "message": "you left the room"}

    # -- reporting ----------------------------------------------------------

    def report(self) -> list[str]:
        """Human-facing state, for `/collab`."""
        if self.problems:
            return ["collab room: not running"] + [f"  {p}" for p in self.problems]
        if not self.listening:
            return ["collab room: not started", "  /collab host publishes this session"]
        where = str(self.socket_path) if self.unix else f"{self.bind_host or '127.0.0.1'}:{self.port}"
        lines = [
            f"collab room: listening on {where}",
            f"  invite: {self.address()}",
            f"  driving: {self.driver_name()}",
        ]
        lines.extend(roster_lines(self.roster()))
        if self.dropped:
            lines.append(f"  dropped: {self.dropped}")
            lines.extend(f"    {reason}" for reason in self.drops[-3:])
        return lines


# -- one joined peer, from the guest's side ---------------------------------


class Client:
    """The guest half: a socket, a reader thread, replies and an event feed.

    Replies and events arrive interleaved on one socket, so the reader thread
    demultiplexes them: a frame with an `id` belongs to a waiting caller, a
    frame with a `method` goes on the feed.  Callers block on their own queue
    rather than on the socket, which is what lets a guest read chat while a
    prompt they sent is still running.
    """

    __slots__ = ("_lock", "_pending", "_reader", "_stop", "closed", "events", "greeting", "id", "name", "role", "seen", "sock")

    def __init__(self, sock: socket.socket, greeting: dict[str, Any]) -> None:
        self.sock = sock
        self.greeting = greeting
        self.id = str(greeting.get("peer") or "")
        self.name = str(greeting.get("name") or "")
        self.role = str(greeting.get("role") or OBSERVER)
        #: Unbounded on purpose: this is the guest's own memory, and dropping
        #: their own chat history to save a few kilobytes would be absurd.
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        #: Events `wait_for` skipped past, kept so a later assertion or a
        #: scrollback view can still find them.
        self.seen: list[dict[str, Any]] = []
        self.closed = threading.Event()
        self._pending: dict[str, queue.Queue[Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, name="collab-client", daemon=True)
        self._reader.start()

    # -- wire ---------------------------------------------------------------

    def _read_loop(self) -> None:
        buf = bytearray()
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.sock.recv(65536)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                if len(buf) > MAX_FRAME:
                    return
                while (nl := buf.find(b"\n")) >= 0:
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if line.strip():
                        self._deliver(line)
        finally:
            self._fail_pending("the room hung up")
            self.closed.set()

    def _deliver(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(message, dict):
            return
        ident = message.get("id")
        if ident is not None:
            with self._lock:
                waiter = self._pending.pop(str(ident), None)
            if waiter is not None:
                waiter.put(message)
            return
        method = message.get("method")
        if isinstance(method, str):
            params = message.get("params")
            payload = dict(params) if isinstance(params, dict) else {}
            payload.setdefault("event", method)
            self.events.put(payload)

    def _fail_pending(self, why: str) -> None:
        with self._lock:
            waiting, self._pending = self._pending, {}
        for waiter in waiting.values():
            waiter.put(RpcError(INTERNAL_ERROR, why))

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout: float = CALL_TIMEOUT) -> Any:
        """One request, one reply.  Raises on a protocol fault, not on a
        domain refusal — a refused prompt is a result the caller must read."""
        ident = new_id()
        waiter: queue.Queue[Any] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[ident] = waiter
        frame = _encode({"jsonrpc": PROTOCOL, "id": ident, "method": method, "params": params or {}})
        try:
            self.sock.sendall(frame)
        except OSError as exc:
            with self._lock:
                self._pending.pop(ident, None)
            raise CollabError(f"{method} could not be sent: {exc}") from exc
        try:
            message = waiter.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(ident, None)
            raise CollabError(f"{method} got no reply within {timeout:g}s") from None
        if isinstance(message, RpcError):
            raise message
        error = message.get("error")
        if isinstance(error, dict):
            raise RpcError(int(error.get("code", INTERNAL_ERROR)), str(error.get("message", "")))
        return message.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        try:
            self.sock.sendall(_encode({"jsonrpc": PROTOCOL, "method": method, "params": params or {}}))
        except OSError:
            # Nothing to salvage: a notification has no reply to fail, and the
            # reader thread will report the dead socket a moment from now.
            pass

    # -- the room, from here ------------------------------------------------

    def roster(self) -> list[dict[str, Any]]:
        result = self.call("roster")
        rows = result.get("peers") if isinstance(result, dict) else None
        return list(rows) if isinstance(rows, list) else []

    def status(self) -> dict[str, Any]:
        result = self.call("status")
        return dict(result) if isinstance(result, dict) else {}

    def say(self, text: str) -> dict[str, Any]:
        result = self.call("say", {"text": text})
        return dict(result) if isinstance(result, dict) else {}

    def drive(self) -> dict[str, Any]:
        result = self.call("drive")
        out = dict(result) if isinstance(result, dict) else {}
        if out.get("ok"):
            self.role = DRIVER
        return out

    def prompt(self, text: str, *, wait: bool = True, timeout: float = CALL_TIMEOUT) -> dict[str, Any]:
        result = self.call("prompt", {"text": text, "wait": wait}, timeout=timeout)
        return dict(result) if isinstance(result, dict) else {}

    def wait_for(
        self,
        event: str,
        *,
        where: Callable[[dict[str, Any]], bool] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Block until a matching event arrives.

        Waits on the queue rather than polling a flag, so a test or a UI never
        has to guess how long the room will take.  Non-matching events go to
        `seen` instead of being thrown away, because the event that proves a
        later assertion is often the one skipped past here.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CollabError(f"no {event} arrived within {timeout:g}s")
            try:
                payload = self.events.get(timeout=remaining)
            except queue.Empty:
                raise CollabError(f"no {event} arrived within {timeout:g}s") from None
            if payload.get("event") == event and (where is None or where(payload)):
                return payload
            self.seen.append(payload)

    def close(self) -> None:
        """Say goodbye, then hang up.  The goodbye is a notification because a
        reply cannot be waited for on a socket that is about to close."""
        if not self._stop.is_set():
            self.notify("leave")
        self._stop.set()
        _hangup(self.sock)
        self.closed.wait(2.0)


# -- process state ----------------------------------------------------------
#
# One process, one room and at most one membership, held at module level for
# the same reason `settings` is: they describe this run, and every caller wants
# the same one.

_room: Room | None = None
_client: Client | None = None


def active_room() -> Room | None:
    """The room this process is hosting, or None."""
    return _room


def active_client() -> Client | None:
    """The room this process has joined, or None."""
    return _client


def host(
    hooks: RoomHooks | None = None,
    *,
    home: str | os.PathLike[str] | None = None,
    listen: str = "",
    queue_limit: int = QUEUE_LIMIT,
    send_timeout: float = SEND_TIMEOUT,
) -> Room:
    """Publish this session and remember it as the process's room.

    Returns the room whether or not it bound: `problems` and `listening` say
    which, because a shell that cannot publish must still be a shell.
    """
    global _room
    existing = _room
    if existing is not None and existing.listening:
        return existing
    room = Room(hooks, home=home, listen=listen, queue_limit=queue_limit, send_timeout=send_timeout)
    room.serve()
    _room = room
    return room


def join(
    addr: str = "",
    *,
    token: str = "",
    name: str = "",
    home: str | os.PathLike[str] | None = None,
    timeout: float = JOIN_TIMEOUT,
    install: bool = True,
) -> Client:
    """Connect to a room, authenticate, and return the membership.

    Raises `JoinError` rather than returning a half-dead client: every caller
    would otherwise have to check, and the one that forgot would report an
    empty roster instead of a refused token.
    """
    global _client
    where, discovered = _resolve(addr, home)
    secret = token or where.token or discovered
    if not secret:
        raise JoinError("a room token is needed; append it as address#token")

    sock = _connect(where, timeout)
    sock.settimeout(max(timeout, 1.0))
    try:
        sock.sendall(_encode({
            "jsonrpc": PROTOCOL,
            "id": "hello",
            "method": "hello",
            "params": {"token": secret, "name": name or _default_name(), "version": COLLAB_VERSION},
        }))
        greeting = _read_reply(sock, timeout)
    except OSError as exc:
        _hangup(sock)
        raise JoinError(f"{where}: {type(exc).__name__}: {exc}") from exc
    except JoinError:
        _hangup(sock)
        raise

    # The read timeout is dropped to a poll slice now that the handshake is
    # done: the reader thread must notice `close()` promptly, and a room that
    # is merely quiet is not a room that is broken.
    sock.settimeout(_TICK)
    client = Client(sock, greeting)
    if install:
        _client = client
    return client


def leave() -> list[str]:
    """Leave whichever room this process is in, hosting or visiting."""
    global _client, _room
    lines: list[str] = []
    client, _client = _client, None
    if client is not None:
        client.close()
        lines.append("left the room")
    room, _room = _room, None
    if room is not None:
        room.close()
        lines.append("closed the room")
    return lines or ["you are not in a room"]


def _resolve(addr: str, home: str | os.PathLike[str] | None) -> tuple[Address, str]:
    """Where to connect, and the token found on disk if there was one."""
    if addr.strip():
        where = parse_address(addr)
        local = _local_token(home) if where.kind == "unix" else ""
        return where, local

    descriptor, problem = read_descriptor(home)
    if descriptor is None:
        raise JoinError(problem or "no room was found on this machine")
    if str(descriptor.get("transport")) == "unix":
        where = Address("unix", path=str(descriptor.get("path") or ""))
    else:
        where = Address("tcp", host=str(descriptor.get("host") or "127.0.0.1"), port=int(descriptor.get("port") or 0))
    return where, _local_token(home, descriptor)


def _local_token(home: str | os.PathLike[str] | None, descriptor: dict[str, Any] | None = None) -> str:
    """The token file beside a room on this machine.

    Only ever consulted for a local room: reading it for a remote address
    would silently send this machine's secret to somebody else's socket.
    """
    if descriptor is not None and descriptor.get("token_path"):
        target = Path(str(descriptor["token_path"]))
    else:
        base = Path(home) if home is not None else settings.home()
        target = base / TOKEN_NAME
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _connect(where: Address, timeout: float) -> socket.socket:
    if where.kind == "unix":
        if not hasattr(socket, "AF_UNIX"):
            raise JoinError("this platform has no unix sockets; join over host:port instead")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        target: Any = where.path
    else:
        family = socket.AF_INET6 if ":" in where.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        target = (where.host, where.port)
    sock.settimeout(max(timeout, 1.0))
    try:
        sock.connect(target)
    except OSError as exc:
        sock.close()
        raise JoinError(f"{where}: {type(exc).__name__}: {exc}") from exc
    return sock


def _read_reply(sock: socket.socket, timeout: float) -> dict[str, Any]:
    """The handshake reply, read on this thread before the reader starts."""
    buf = bytearray()
    deadline = time.monotonic() + max(timeout, 1.0)
    while (nl := buf.find(b"\n")) < 0:
        if time.monotonic() > deadline:
            raise JoinError("the room did not answer hello in time")
        chunk = sock.recv(65536)
        if not chunk:
            raise JoinError("the room closed the connection during the handshake")
        buf += chunk
        if len(buf) > MAX_FRAME:
            raise JoinError("the room's greeting was implausibly large")
    try:
        message = json.loads(bytes(buf[:nl]))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JoinError(f"the room's greeting was not JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise JoinError("the room's greeting was not a JSON object")
    error = message.get("error")
    if isinstance(error, dict):
        raise JoinError(str(error.get("message") or "the room refused this peer"))
    result = message.get("result")
    if not isinstance(result, dict) or not result.get("ok"):
        raise JoinError("the room's greeting did not confirm the join")
    return result


def _default_name() -> str:
    """A name that means something in a chat log without asking for one."""
    for key in ("OFFSET_COLLAB_NAME", "USER", "LOGNAME"):
        value = os.environ.get(key, "").strip()
        if value:
            return _clean_name(value)
    return "guest"


# -- text -------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    body = text.strip()
    return body if len(body) <= limit else body[: limit - 1] + "\u2026"


def _clean_name(raw: str) -> str:
    """A display name safe to print in somebody else's terminal.

    Control characters are stripped rather than escaped: a name containing an
    escape sequence would repaint every other guest's screen, which is a real
    attack on a room whose whole purpose is showing you what somebody else is
    doing.
    """
    kept = "".join(ch for ch in str(raw) if ch.isprintable())
    return _clip(kept, MAX_NAME)


def roster_lines(rows: list[dict[str, Any]]) -> list[str]:
    """The roster as a human reads it, host first, driver marked."""
    lines: list[str] = []
    for row in rows:
        role = str(row.get("role", OBSERVER))
        mark = "*" if role in (HOST, DRIVER) else " "
        lines.append(f"  {mark} {str(row.get('name', '?')):<24} {role}")
    return lines or ["  (nobody)"]


# -- the shell surface ------------------------------------------------------


def shell_hooks(state: Any, room: Room) -> RoomHooks:
    """Wire a room to a running shell.

    The room drives the agent loop itself rather than posting into the shell's
    input queue: a guest's prompt has to work whether or not the TUI is sitting
    at a prompt, and driving the loop here is what makes tool events reach the
    other humans.
    """

    def prompt(text: str, who: str) -> tuple[bool, str]:
        agent = getattr(state, "agent", None)
        if agent is None:
            return False, "no agent is attached to this room"
        reply, error = "", None
        for event in agent.run(text):
            room.observe(event)
            if isinstance(event, Finished):
                reply = event.text
            elif isinstance(event, StreamError):
                error = event.message
        if error is not None:
            return False, error
        return True, reply

    def model() -> str:
        return str(getattr(state, "model", "") or "")

    return RoomHooks(
        workspace=Path(getattr(state, "workspace", Path.cwd())),
        name=_default_name(),
        prompt=prompt,
        model=model,
    )


def _host(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_ERR, TONE_OK, Outcome

    if _client is not None:
        return Outcome.error("you are already in somebody else's room", "/collab leave first")
    room = active_room()
    if room is not None and room.listening:
        return Outcome(["this session is already published"] + room.report(), TONE_OK)
    listen = rest[0] if rest else ""
    room = host(listen=listen)
    room.hooks = shell_hooks(state, room)
    if not room.listening:
        return Outcome(["the room could not be published"] + [f"  {p}" for p in room.problems], TONE_ERR)
    return Outcome(room.report() + ["", "share the invite line; the token is the password"], TONE_OK)


def _join(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    if not rest:
        return Outcome.error("usage: /collab join <address>#<token>", "or /collab join for a room on this machine")
    if _client is not None:
        return Outcome.error("you are already in a room", "/collab leave first")
    addr = rest[0] if rest[0] != "-" else ""

    def connect() -> Any:
        try:
            client = join(addr)
        except CollabError as exc:
            return Outcome.error(f"could not join: {exc}")
        rows = client.roster()
        return Outcome(
            [f"joined as {client.name} ({client.role})", f"driving: {client.greeting.get('driver', '?')}"]
            + roster_lines(rows),
            TONE_OK,
        )

    return Outcome([f"joining {addr or 'the room on this machine'}..."], TONE_INFO, job=connect)


def _who(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, Outcome

    client = active_client()
    if client is not None:
        result = client.call("roster")
        rows = result.get("peers", []) if isinstance(result, dict) else []
        return Outcome([f"driving: {result.get('driver', '?')}"] + roster_lines(rows), TONE_INFO)
    room = active_room()
    if room is None or not room.listening:
        return Outcome.error("you are not in a room", "/collab host or /collab join <address>")
    return Outcome([f"driving: {room.driver_name()}"] + roster_lines(room.roster()), TONE_INFO)


def _drive(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_OK, Outcome

    client = active_client()
    if client is not None:
        result = client.drive()
        if result.get("ok"):
            return Outcome([str(result.get("message") or "you have the wheel")], TONE_OK)
        return Outcome.error(str(result.get("error") or "the wheel is taken"))
    room = active_room()
    if room is None or not room.listening:
        return Outcome.error("you are not in a room", "/collab host or /collab join <address>")
    ok, message = room.reclaim()
    return Outcome([message], TONE_OK if ok else "err")


def _say(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, Outcome

    text = " ".join(rest).strip()
    if not text:
        return Outcome.error("usage: /collab say <text>", "chat reaches the humans, never the model")
    client = active_client()
    if client is not None:
        result = client.say(text)
        if not result.get("ok"):
            return Outcome.error(str(result.get("error") or "nobody heard that"))
        return Outcome([f"said to {result.get('heard', 0)} peer(s)"], TONE_INFO)
    room = active_room()
    if room is None or not room.listening:
        return Outcome.error("you are not in a room", "/collab host or /collab join <address>")
    return Outcome([f"said to {room.chat(text)} peer(s)"], TONE_INFO)


def _status(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, Outcome

    client = active_client()
    if client is not None:
        status = client.status()
        return Outcome(
            [
                f"in {status.get('host', '?')}'s room as {status.get('you', '?')} ({status.get('role', '?')})",
                f"driving: {status.get('driver', '?')}",
                f"workspace: {status.get('workspace', '?')}",
            ],
            TONE_INFO,
        )
    room = active_room()
    if room is None:
        return Outcome(
            [
                "collab room: not started",
                "  /collab host          publish this session",
                "  /collab join <addr>   join somebody else's",
            ],
            TONE_INFO,
        )
    return Outcome(room.report(), TONE_INFO)


def _leave(state: Any, rest: list[str]) -> Any:
    from offset.shell.commands import TONE_OK, Outcome

    return Outcome(leave(), TONE_OK)


#: Subcommand table.  A dict rather than a chain of `if`s so `/collab` can list
#: what it accepts when given something it does not.
SUBCOMMANDS: Final = {
    "host": _host,
    "join": _join,
    "leave": _leave,
    "who": _who,
    "drive": _drive,
    "say": _say,
    "status": _status,
}


def _collab(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import Outcome

    if not args:
        return _status(state, [])
    name = args[0].lower()
    handler = SUBCOMMANDS.get(name)
    if handler is None:
        return Outcome.error(
            f"no /collab {name}",
            "try: " + ", ".join(sorted(SUBCOMMANDS)),
        )
    return handler(state, args[1:])


def collab_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command(
            "collab",
            "share this session with other humans",
            _collab,
            usage="/collab host | join <addr#token> | leave | who | drive | say <text>",
        ),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handlers import from `offset.shell.commands`,
    which imports this module: resolving at import time would be a cycle.

    The re-check after building is the same guard `offset.core.tasks` needs and
    for the same reason — importing the shell registry re-enters here before
    the outer call has stored anything, so a single access would otherwise
    produce two lists and register `/collab` twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = collab_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
