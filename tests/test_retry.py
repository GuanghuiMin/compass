"""What is retried and what is not.

The line these tests defend is the one that makes the measurements mean anything: a provider that
did not answer may be asked again, and an answer this system read and rejected may not. Blurring it
would turn one sample into a best-of-three and inflate every acceptance rate in the results. Here
that line is tested on the retry loop itself; the callers that use it test their own side of it.
"""

import pytest

from future_graph.adapter import AdapterError, EmptyModelCompletion
from future_graph.retry import (
    Attempt, ExhaustedAttempts, MAX_ATTEMPTS, call_with_retry, is_operational,
)


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    """The backoff is real seconds in a run and must not be real seconds in the suite. Tests that
    assert on the waiting pass their own `sleep` and are unaffected by this."""
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _seconds: None)


def never_sleeps(_seconds):
    return None


def sequence(*outcomes):
    """Produces each outcome in turn: raises it if it is an exception, otherwise returns it."""
    remaining = list(outcomes)
    calls = []

    def model(call):
        calls.append(call)
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    model.calls = calls
    return model


class RateLimited(Exception):
    status_code = 429


class ProviderBroke(Exception):
    status_code = 503


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


# --------------------------------------------------------------------------- classification

@pytest.mark.parametrize("err", [
    EmptyModelCompletion("nothing came back"),
    RateLimited(),
    ProviderBroke(),
    APITimeoutError(),
    APIConnectionError(),
    TimeoutError(),
    ConnectionError(),
])
def test_a_provider_that_did_not_answer_is_operational(err):
    assert is_operational(err)


@pytest.mark.parametrize("err", [
    AdapterError("the response carries no choices"),
    AdapterError("the content is int, not text"),
    ValueError("something else entirely"),
    TypeError("a model returns text"),
])
def test_an_answer_in_the_wrong_shape_is_not_operational(err):
    """A malformed response is a broken adapter, and asking again would not mend it."""
    assert not is_operational(err)


def test_a_four_hundred_is_not_operational():
    class BadRequest(Exception):
        status_code = 400

    assert not is_operational(BadRequest())


# --------------------------------------------------------------------------- the loop

def test_a_call_that_works_is_one_attempt():
    text, attempts = call_with_retry(sequence("ok"), object(), sleep=never_sleeps)
    assert text == "ok"
    assert attempts == (Attempt(1, "completion", "2 characters"),)


def test_the_first_completion_ends_it():
    model = sequence(EmptyModelCompletion("nothing"), "ok", "never reached")
    text, attempts = call_with_retry(model, object(), sleep=never_sleeps)
    assert text == "ok"
    assert [a.outcome for a in attempts] == ["EmptyModelCompletion", "completion"]
    assert len(model.calls) == 2


def test_at_most_three_attempts():
    model = sequence(*[EmptyModelCompletion("nothing")] * 5)
    with pytest.raises(ExhaustedAttempts) as raised:
        call_with_retry(model, object(), sleep=never_sleeps)
    assert len(model.calls) == MAX_ATTEMPTS == 3
    assert len(raised.value.attempts) == 3
    assert all(a.outcome == "EmptyModelCompletion" for a in raised.value.attempts)


def test_the_third_attempt_can_still_succeed():
    model = sequence(RateLimited(), ProviderBroke(), "ok")
    text, attempts = call_with_retry(model, object(), sleep=never_sleeps)
    assert text == "ok"
    assert [a.outcome for a in attempts] == ["RateLimited", "ProviderBroke", "completion"]


def test_a_failure_that_is_not_operational_is_raised_at_once():
    model = sequence(AdapterError("the message carries no content"), "ok")
    with pytest.raises(AdapterError):
        call_with_retry(model, object(), sleep=never_sleeps)
    assert len(model.calls) == 1


def test_every_attempt_sends_exactly_the_same_call():
    """Independent draws from the same distribution, not a conversation. Nothing is rephrased and
    nothing is added about the failure that came before."""
    call = object()
    model = sequence(EmptyModelCompletion("nothing"), EmptyModelCompletion("nothing"), "ok")
    call_with_retry(model, call, sleep=never_sleeps)
    assert model.calls == [call, call, call]


def test_it_waits_between_attempts():
    waits = []
    model = sequence(RateLimited(), ProviderBroke(), "ok")
    call_with_retry(model, object(), sleep=waits.append)
    assert waits == [2.0, 4.0]
    assert len(waits) == 2                    # never after the last attempt


def test_retrying_can_be_switched_off():
    model = sequence(EmptyModelCompletion("nothing"), "ok")
    with pytest.raises(ExhaustedAttempts):
        call_with_retry(model, object(), max_attempts=1, sleep=never_sleeps)
    assert len(model.calls) == 1
