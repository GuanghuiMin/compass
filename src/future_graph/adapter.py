"""The one real model adapter, and everything it refuses.

The endpoint and the model are matched exactly, with no defaults. The client this replaces defaulted
to an internal endpoint serving a weaker model, and a run that fell back to it produced numbers that
looked fine and were not comparable with anything.

`max_retries=0` is not decoration. The SDK retries connection errors, timeouts, 408, 409, 429 and 5xx
twice by default, so one call from here would be up to three requests on the wire — which spends
tokens nobody counted and quietly turns one sample into the first of several.

A provider that does not answer is not a model that answered badly. A missing choice, a missing
message or a content that is not a string is an AdapterError; an empty string is what the model said
and goes to the parser, where it becomes a parse rejection like any other unusable answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from .artifacts import ModelCall

BASE_URL = "https://ollama.com/v1"
MODEL = "minimax-m3"
TIMEOUT_S = 240

BASE_URL_VAR = "TRACE_MINIMAX_BASE_URL"
MODEL_VAR = "TRACE_MINIMAX_MODEL"
API_KEY_VAR = "TRACE_MINIMAX_API_KEY"


class AdapterError(RuntimeError):
    """The provider did not answer in the shape this contract requires."""


@dataclass(frozen=True)
class Adapter:
    """Callable as a Model. Builds its request from the ModelCall and nothing else."""
    client: Any
    model: str = MODEL

    def __call__(self, call: ModelCall) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": call.system},
                      {"role": "user", "content": call.user}],
            **dict(call.config),
        )
        return _content(response)


def _content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise AdapterError("the response carries no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise AdapterError("the first choice carries no message")
    content = getattr(message, "content", None)
    if content is None:
        raise AdapterError("the message carries no content")
    if not isinstance(content, str):
        raise AdapterError(f"the content is {type(content).__name__}, not text")
    return content


def from_environment(client_factory: Callable[..., Any] | None = None) -> Adapter:
    """Build the adapter, or refuse before anything is constructed.

    The key is read here and passed to the client. It is never returned, recorded, logged, or put in
    an exception message.
    """
    base_url = os.environ.get(BASE_URL_VAR)
    model = os.environ.get(MODEL_VAR)
    api_key = os.environ.get(API_KEY_VAR)

    if not base_url:
        raise AdapterError(f"{BASE_URL_VAR} is not set, and there is no default")
    if base_url != BASE_URL:
        raise AdapterError(f"{BASE_URL_VAR} is {base_url!r}, and this run is only for {BASE_URL!r}")
    if not model:
        raise AdapterError(f"{MODEL_VAR} is not set, and there is no default")
    if model != MODEL:
        raise AdapterError(f"{MODEL_VAR} is {model!r}, and this run is only for {MODEL!r}")
    if not api_key:
        raise AdapterError(f"{API_KEY_VAR} is not set")

    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    try:
        client = client_factory(base_url=base_url, api_key=api_key, timeout=TIMEOUT_S,
                                max_retries=0)
    except Exception as err:
        # One kind of failure for "the adapter could not be built", so a caller can settle the
        # adapter before claiming anything and know what a failure there means.
        raise AdapterError(f"the client could not be constructed: {err}") from err
    return Adapter(client=client, model=model)
