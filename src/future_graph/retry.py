"""Retrying a call the provider failed to answer, and nothing else.

Across thirty-one sequential calls, three came back with zero characters. Nothing was generated, so
there was no candidate to be wrong about, and yet two of three attempted chains ended there. A
method meant to run for the length of an episode cannot be stopped by that, and counting it as a
protocol failure would put provider hiccups into the one number this system exists to measure.

So this retries **operational** failures: an empty completion, a timeout, a connection error, a 429,
a provider 5xx. At most two retries, so at most three identical attempts, and the first completion
that arrives ends it.

It retries nothing else. A response whose shape is wrong is a broken adapter, not a hiccup. A
revision that failed to parse and a graph that failed to validate are the model's answers, and
asking again until one passes would turn one sample into a best-of-three and quietly inflate every
acceptance rate. Retry therefore sits strictly around the call and cannot see what happens after it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .adapter import EmptyModelCompletion
from .artifacts import ModelCall

MAX_ATTEMPTS = 3
BACKOFF_S = (2.0, 4.0)

# Provider failures, named rather than caught by type so this does not depend on the SDK being
# importable. The status codes carry the same meaning as the names and are checked first, because a
# generic status error is the shape the SDK uses for most of them.
OPERATIONAL_NAMES = frozenset({
    "APITimeoutError", "APIConnectionError", "APIConnectionTimeoutError", "RateLimitError",
    "InternalServerError", "ServiceUnavailableError", "Timeout", "TimeoutError",
    "ConnectionError", "ConnectionResetError",
})


@dataclass(frozen=True)
class Attempt:
    """One call on the wire, whatever became of it."""
    ordinal: int
    outcome: str            # "completion", or the class name of the failure
    detail: str = ""

    def as_list(self) -> list:
        return [self.ordinal, self.outcome, self.detail]


class ExhaustedAttempts(RuntimeError):
    """Every attempt failed operationally. Carries them so the boundary can be recorded as what it
    was: a provider that did not answer, not a compressor that answered badly."""

    def __init__(self, attempts: tuple[Attempt, ...], last: BaseException) -> None:
        super().__init__(f"the provider did not answer in {len(attempts)} attempts: {last}")
        self.attempts = attempts
        self.last = last


def is_operational(err: BaseException) -> bool:
    """Is this the provider failing to answer, rather than an answer this system dislikes?"""
    if isinstance(err, EmptyModelCompletion):
        return True
    status = getattr(err, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return type(err).__name__ in OPERATIONAL_NAMES


def call_with_retry(model: Callable[[ModelCall], str], call: ModelCall,
                    max_attempts: int = MAX_ATTEMPTS,
                    sleep: Callable[[float], None] | None = None
                    ) -> tuple[str, tuple[Attempt, ...]]:
    """Call until something comes back, up to `max_attempts` identical attempts.

    Every attempt sends exactly the same `ModelCall`. Nothing is rephrased, nothing is added about
    the previous failure, and no configuration is changed between them, so the attempts are
    independent draws from the same distribution rather than a conversation.
    """
    if max_attempts < 1:
        raise ValueError(f"a call takes at least one attempt, got {max_attempts}")
    # Resolved here rather than as a default, so that the waiting is real in a run and can be
    # taken out of the suite without the production signature growing a test parameter.
    wait = sleep if sleep is not None else time.sleep
    attempts: list[Attempt] = []
    for ordinal in range(1, max_attempts + 1):
        try:
            text = model(call)
        except Exception as err:              # noqa: BLE001 - classified immediately below
            if not is_operational(err):
                raise
            attempts.append(Attempt(ordinal, type(err).__name__, str(err)))
            if ordinal == max_attempts:
                raise ExhaustedAttempts(tuple(attempts), err) from err
            wait(BACKOFF_S[min(ordinal, len(BACKOFF_S)) - 1])
            continue
        # Anything that is not text is a broken adapter rather than a provider hiccup, and it is
        # left for the caller to report as one. Measuring it here would be the wrong layer.
        size = f"{len(text)} characters" if isinstance(text, str) else type(text).__name__
        attempts.append(Attempt(ordinal, "completion", size))
        return text, tuple(attempts)
    raise AssertionError("unreachable")
