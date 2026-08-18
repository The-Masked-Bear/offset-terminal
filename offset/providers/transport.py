"""HTTP transport: streaming POST with a retry policy, stdlib only.

Two rules that matter more than they look:

  * Retry only before the first byte of the body reaches the caller.  Once a
    token has been handed upstream the stream is not resumable, and retrying
    would duplicate output.
  * Honour `Retry-After` when the server sends it.  Guessing a backoff when
    the server already told you the answer is how rate limits turn into
    outages.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

#: Statuses worth trying again.  408 request timeout, 409 conflict (some
#: gateways use it for cold starts), 429 rate limit, and the 5xx family.
RETRYABLE: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class HTTPFailure(Exception):
    def __init__(self, status: int, body: str, *, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE

    def detail(self) -> str:
        """The provider's own error message, when it sent a JSON one."""
        try:
            obj = json.loads(self.body)
        except (json.JSONDecodeError, TypeError):
            return self.body[:400]
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err or obj)[:400]


@dataclass(slots=True)
class Retry:
    attempts: int = 5
    base: float = 0.6
    cap: float = 30.0
    jitter: float = 0.25

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Server instruction wins; otherwise exponential backoff with jitter."""
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.cap)
        raw = min(self.base * (2**attempt), self.cap)
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    for key in ("retry-after-ms", "Retry-After-Ms"):
        raw = headers.get(key)
        if raw:
            try:
                return float(raw) / 1000.0
            except ValueError:
                pass
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def post_lines(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float = 300.0,
    retry: Retry | None = None,
    sleep=time.sleep,
) -> Iterator[bytes]:
    """POST JSON and yield response lines as they arrive."""
    policy = retry or Retry()
    body = json.dumps(payload).encode("utf-8")
    sent = {"Content-Type": "application/json", "Accept": "text/event-stream", **headers}
    last: HTTPFailure | None = None

    for attempt in range(policy.attempts):
        request = urllib.request.Request(url, data=body, headers=sent, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for line in response:
                    yield line.rstrip(b"\r\n")
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
            last = HTTPFailure(exc.code, detail, retry_after=_retry_after_seconds(exc.headers))
        except urllib.error.URLError as exc:
            last = HTTPFailure(0, f"{exc.reason}", retry_after=None)
        except TimeoutError:
            last = HTTPFailure(0, "request timed out")

        transient = last.status == 0 or last.retryable
        if not transient or attempt == policy.attempts - 1:
            raise last
        sleep(policy.delay(attempt, last.retry_after))

    raise last if last else RuntimeError("unreachable")
