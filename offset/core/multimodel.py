"""Several models working at once.

The unit is a `Seat`: one model, playing one role, with a weight.  A roster of
seats can answer the same request concurrently and then be reconciled four
ways — race for latency, vote for agreement, council when a judge should
choose, relay when the roles are a pipeline.

Three properties this module is built to guarantee:

  * **Failure isolation.**  A seat that dies takes only its own answer with it.
    One 429 does not end the turn; the other models keep going.
  * **Order stability.**  Opinions come back in seat order no matter who
    finished first, so a session replays identically.
  * **Legible concurrency.**  `stream()` interleaves every seat's events on one
    queue tagged with its seat, which is what lets the UI show N lanes filling
    in at once instead of a scrambled single column.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from offset.providers.base import (
    Event,
    Message,
    Provider,
    Request,
    Stop,
    Turn,
    TurnBuilder,
    Usage,
)
from offset.providers.registry import ModelInfo, credential, info

#: Conventional roles.  Nothing enforces them; the scheduler just routes by name.
#:
#: `security` and `researcher` are seats a pipeline can fill rather than
#: behaviours the scheduler knows: an auditor is only useful if it is a
#: *different* model from the one that wrote the code, and naming the role is
#: what lets `/flow` and `/decompose` guarantee that.
ROLES = ("planner", "researcher", "implementer", "critic", "security",
         "referee", "cheap", "bulk")


@dataclass(slots=True)
class Seat:
    model: str
    role: str = "implementer"
    weight: float = 1.0
    provider: Provider | None = None
    api_key: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.role}:{self.model}"

    def endpoint(self, resolver: Callable[[str], tuple[Provider, ModelInfo]] | None = None) -> tuple[Provider, ModelInfo]:
        provider, meta = (resolver or _resolver())(self.model)
        return (self.provider or provider), meta

    def cred(self, provider: Provider) -> Any:
        """The credential to use, if any."""
        if self.api_key is not None:
            from offset.providers.auth import Credential
            return Credential(provider=provider.name, value=self.api_key, kind="api_key")
        from offset.providers.auth import load
        return load(provider.name)
        
    def key(self, provider: Provider) -> str | None:
        cred = self.cred(provider)
        return cred.value if cred and cred.kind == "api_key" else None


@dataclass(slots=True)
class Opinion:
    seat: Seat
    turn: Turn = field(default_factory=Turn)
    latency: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.turn.error is None

    @property
    def text(self) -> str:
        return self.turn.text


@dataclass(slots=True)
class Verdict:
    winner: Opinion | None
    reason: str
    opinions: list[Opinion] = field(default_factory=list)
    tally: dict[str, float] = field(default_factory=dict)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for o in self.opinions:
            total = total + o.turn.usage
        return total

    @property
    def dissent(self) -> list[Opinion]:
        """Everyone who did not win but did answer — worth showing the user."""
        return [o for o in self.opinions if o.ok and o is not self.winner]


def normalise(text: str) -> str:
    """Collapse whitespace and case so equivalent answers compare equal."""
    return " ".join(text.split()).strip().lower()


class Ensemble:
    """A roster of seats that can answer the same request together."""

    __slots__ = ("_resolve", "_workers", "seats")

    def __init__(
        self,
        seats: Sequence[Seat],
        *,
        resolver: Callable[[str], tuple[Provider, ModelInfo]] | None = None,
        max_workers: int = 8,
    ) -> None:
        if not seats:
            raise ValueError("an ensemble needs at least one seat")
        self.seats = list(seats)
        #: Resolved on every use rather than captured here: a credential added
        #: or a config reloaded after this object was built must still be seen.
        self._resolve = resolver
        self._workers = max(1, min(max_workers, len(seats)))

    # -- one seat ---------------------------------------------------------

    def ask(self, seat: Seat, request: Request) -> Opinion:
        """Run one seat.  Never raises: a failure becomes an Opinion."""
        started = time.monotonic()
        try:
            provider, meta = seat.endpoint(self._resolve)
            scoped = request.with_model(seat.model)
            scoped.max_tokens = min(request.max_tokens, meta.max_output)
            if not meta.thinking:
                scoped.thinking_budget = None
            if not meta.tools:
                scoped.tools = ()
            turn = TurnBuilder().consume(provider.stream(scoped, api_key=seat.key(provider), credential=seat.cred(provider))).finish()
            return Opinion(seat, turn, time.monotonic() - started, turn.error)
        except Exception as exc:  # isolation: this seat only
            return Opinion(seat, Turn(stop_reason="error"), time.monotonic() - started, f"{type(exc).__name__}: {exc}")

    # -- all seats --------------------------------------------------------

    def gather(self, request: Request, seats: Sequence[Seat] | None = None) -> list[Opinion]:
        """Every seat answers concurrently; results come back in seat order."""
        chosen = list(seats or self.seats)
        if len(chosen) == 1:
            return [self.ask(chosen[0], request)]
        with ThreadPoolExecutor(max_workers=min(self._workers, len(chosen))) as pool:
            return list(pool.map(lambda s: self.ask(s, request), chosen))

    def stream(self, request: Request, seats: Sequence[Seat] | None = None) -> Iterator[tuple[Seat, Event]]:
        """Interleave every seat's events on one queue, tagged with its seat.

        This is the multi-lane view: the UI reads one stream and paints N
        panels.  A seat that fails emits its `Stop("error")` like any other, so
        a lane always terminates.
        """
        chosen = list(seats or self.seats)
        outbox: queue.Queue[tuple[Seat, Event] | None] = queue.Queue()

        def pump(seat: Seat) -> None:
            try:
                provider, meta = seat.endpoint(self._resolve)
                scoped = request.with_model(seat.model)
                scoped.max_tokens = min(request.max_tokens, meta.max_output)
                for event in provider.stream(scoped, api_key=seat.key(provider), credential=seat.cred(provider)):
                    outbox.put((seat, event))
            except Exception as exc:
                outbox.put((seat, Stop("error")))
                outbox.put((seat, _error_event(f"{type(exc).__name__}: {exc}")))
            finally:
                outbox.put(None)

        threads = [threading.Thread(target=pump, args=(s,), name=f"seat-{s.label}", daemon=True) for s in chosen]
        for t in threads:
            t.start()
        live = len(threads)
        while live:
            item = outbox.get()
            if item is None:
                live -= 1
                continue
            yield item

    # -- reconciliation ---------------------------------------------------

    def race(self, request: Request, seats: Sequence[Seat] | None = None) -> Verdict:
        """First usable answer wins; the rest are abandoned, not awaited."""
        chosen = list(seats or self.seats)
        results: queue.Queue[Opinion] = queue.Queue()
        for seat in chosen:
            threading.Thread(target=lambda s=seat: results.put(self.ask(s, request)), daemon=True).start()
        collected: list[Opinion] = []
        for _ in chosen:
            opinion = results.get()
            collected.append(opinion)
            if opinion.ok:
                return Verdict(opinion, f"{opinion.seat.label} answered first", collected)
        return Verdict(None, "every seat failed", collected)

    def vote(self, request: Request, seats: Sequence[Seat] | None = None) -> Verdict:
        """Weighted agreement on the normalised answer text."""
        opinions = self.gather(request, seats)
        tally: dict[str, float] = {}
        for o in opinions:
            if o.ok and o.text.strip():
                tally[normalise(o.text)] = tally.get(normalise(o.text), 0.0) + o.seat.weight
        if not tally:
            return Verdict(None, "no seat produced an answer", opinions)
        best = max(tally.items(), key=lambda kv: kv[1])[0]
        winner = next(o for o in opinions if o.ok and normalise(o.text) == best)
        agreed = sum(1 for o in opinions if o.ok and normalise(o.text) == best)
        return Verdict(winner, f"{agreed}/{len(opinions)} agreed (weight {tally[best]:g})", opinions, tally)

    def council(
        self,
        request: Request,
        judge: Seat,
        seats: Sequence[Seat] | None = None,
        *,
        criterion: str = "correctness, then concision",
    ) -> Verdict:
        """Everyone answers, then a judge model picks the best one.

        The judge sees anonymised answers so it cannot favour a model by name.
        If it fails or answers unparseably, this degrades to a weighted vote
        rather than throwing the whole turn away.
        """
        opinions = self.gather(request, seats)
        usable = [o for o in opinions if o.ok and o.text.strip()]
        if not usable:
            return Verdict(None, "no seat produced an answer", opinions)
        if len(usable) == 1:
            return Verdict(usable[0], "only one seat answered", opinions)

        listing = "\n\n".join(f"[{i}]\n{o.text}" for i, o in enumerate(usable))
        ask = Request(
            model=judge.model,
            system=(
                "You are judging candidate answers. Reply with the index of the best one "
                "as a bare number, then a short reason on the same line."
            ),
            messages=[Message("user", f"Criterion: {criterion}\n\nCandidates:\n\n{listing}\n\nBest index:")],
            max_tokens=200,
        )
        ruling = self.ask(judge, ask)
        pick = _first_index(ruling.text, len(usable))
        if pick is None:
            # Fall back to a weighted vote over the answers we ALREADY have.
            # This used to call self.vote(), which re-ran every seat - a second
            # full round of paid model calls whose result was then discarded.
            tally: dict[str, float] = {}
            for o in usable:
                tally[normalise(o.text)] = tally.get(normalise(o.text), 0.0) + o.seat.weight
            best = max(tally.items(), key=lambda kv: kv[1])[0]
            winner = next(o for o in usable if normalise(o.text) == best)
            return Verdict(winner, "judge unavailable; fell back to weighted vote", opinions, tally)
        return Verdict(usable[pick], f"{judge.label} chose [{pick}]: {ruling.text.strip()[:120]}", opinions)

    def staff(self, order: Sequence[str]) -> list[tuple[str, Seat]]:
        """Assign a seat to every role in `order`, filling gaps.

        A roster is built from whatever models the user can actually reach, and
        the catalogue's own hints (`cheap`, `bulk`, `test`) do not have to cover
        the roles a caller asks for. Skipping unfilled roles meant `relay` over a
        roster of local models produced nothing at all, so a role nobody claims
        goes to the strongest seat not already working - and only repeats a seat
        once every seat is busy.
        """
        ranked = sorted(enumerate(self.seats), key=lambda pair: (-pair[1].weight, pair[0]))
        rank = {index: position for position, (index, _) in enumerate(ranked)}
        used: dict[int, int] = dict.fromkeys(rank, 0)
        previous = -1
        out: list[tuple[str, Seat]] = []
        for role in order:
            exact = [pair for pair in ranked if pair[1].role == role]
            idle = [pair for pair in ranked if not used[pair[0]]]
            # A seat that has not spoken yet outranks the nominal role holder: a
            # critic that already answered as the implementer would be reviewing
            # its own work, which is the one thing a relay exists to avoid.
            pool = [p for p in exact if not used[p[0]]] or idle or exact or ranked
            # Least-used seat wins; never the same seat twice running when there
            # is an alternative; strongest breaks the remaining tie. With more
            # roles than seats this walks the roster instead of handing every
            # leftover role to whichever seat happens to be strongest.
            index, seat = min(pool, key=lambda pair: (used[pair[0]], pair[0] == previous, rank[pair[0]]))
            used[index] += 1
            previous = index
            out.append((role, seat))
        return out

    def relay(self, request: Request, order: Sequence[str] = ("planner", "implementer", "critic")) -> list[Opinion]:
        """Run roles in sequence, each seeing what the previous one produced."""
        conversation = list(request.messages)
        out: list[Opinion] = []
        for role, seat in self.staff(order):
            step = Request(
                model=seat.model,
                messages=list(conversation),
                system=request.system,
                tools=request.tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            opinion = self.ask(seat, step)
            out.append(opinion)
            if opinion.ok and opinion.text:
                conversation.append(Message("assistant", opinion.text))
                conversation.append(Message("user", f"Continue as the {role} handoff. Improve on the above."))
        return out

    # -- routing ----------------------------------------------------------

    def pick(self, role: str) -> Seat | None:
        """Heaviest seat for a role, or None."""
        candidates = [s for s in self.seats if s.role == role]
        return max(candidates, key=lambda s: s.weight) if candidates else None

    def cheapest(self) -> Seat:
        """Prefer a local model, then the lightest declared weight."""
        return min(self.seats, key=lambda s: (not info(s.model).local, s.weight))

    def by_role(self) -> dict[str, list[Seat]]:
        out: dict[str, list[Seat]] = {}
        for seat in self.seats:
            out.setdefault(seat.role, []).append(seat)
        return out

    def __len__(self) -> int:
        return len(self.seats)

    def __iter__(self) -> Iterator[Seat]:
        return iter(self.seats)


def _first_index(text: str, count: int) -> int | None:
    """Pull the first in-range integer out of a judge's reply."""
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return None
    value = int(digits)
    return value if 0 <= value < count else None


def _resolver() -> Callable[[str], tuple[Provider, ModelInfo]]:
    """The current resolver, looked up now rather than at import time."""
    from offset.providers import registry

    return registry.resolve


def _error_event(message: str) -> Event:
    from offset.providers.base import StreamError

    return StreamError(message)


def default_roster(models: Sequence[str] | None = None) -> Ensemble:
    """Build a roster from the catalogue's own role hints."""
    from offset.providers.registry import available

    chosen = list(models) if models else [m.id for m in available()][:4]
    if not chosen:
        chosen = ["mock"]
    seats = [Seat(model=m, role=info(m).role_hint or "implementer") for m in chosen]
    return Ensemble(seats)


def seat_roster(active: str) -> Ensemble:
    """The models a multi-model run may use, active model first.

    `ShellState.ensemble` used to be left at `None`, so every ensemble feature -
    /council, and the per-branch models in /spec - silently collapsed onto one
    model. The active model leads the roster because it is the one deliberately
    chosen; the rest fill the other roles, and a single-seat roster simply means
    every branch uses the same model, which is what happened before.
    """
    from offset.providers.registry import available, reachable

    # `available` only asks whether a key exists; a seat that cannot answer is
    # worse than no seat, so a council is not filled with connection refusals.
    usable = [m.id for m in available() if reachable(m.id)]
    ordered = [active, *(m for m in usable if m != active)] if active in usable or active == "mock" \
        else [active, *usable]
    return default_roster(ordered[:4])
