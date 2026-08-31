"""Thin OpenAI-compatible chat client with retries and usage accounting.

Providers: ``openai`` (api.openai.com) and ``ollama`` (ollama.com cloud). Keys
are read from ``W:/context_compression/.env`` (``OPENAI_API_KEY``, ``SECONDARY_API_KEY``).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT.parent / ".env")
load_dotenv(_ROOT / ".env")

PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "ollama": {"base_url": "https://ollama.com/v1", "key_env": "SECONDARY_API_KEY"},
}


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    log: list = field(default_factory=list)

    def add(self, resp, wall: float, tag: str) -> None:
        self.calls += 1
        u = getattr(resp, "usage", None)
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.wall_seconds += wall
        self.log.append({"tag": tag, "prompt_tokens": pt, "completion_tokens": ct, "wall": round(wall, 2)})

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "wall_seconds": round(self.wall_seconds, 1)}


class LLM:
    def __init__(self, model: str, provider: str = "openai", temperature: float = 0.0,
                 max_tokens: int = 2048, retries: int = 4):
        p = PROVIDERS[provider]
        self.client = OpenAI(api_key=os.environ[p["key_env"]], base_url=p["base_url"])
        self.model, self.provider = model, provider
        self.temperature, self.max_tokens, self.retries = temperature, max_tokens, retries
        self.usage = Usage()

    def chat(self, messages: list[dict], *, tag: str = "agent", max_tokens: int | None = None,
             temperature: float | None = None, json_mode: bool = False) -> str:
        kw = dict(model=self.model, messages=messages,
                  max_tokens=max_tokens or self.max_tokens,
                  temperature=self.temperature if temperature is None else temperature)
        # JSON mode only on OpenAI: on Ollama Cloud, response_format=json_object makes
        # reasoning models (deepseek-v4) spend the token budget on hidden reasoning and
        # return a truncated object; plain prompting yields complete JSON there.
        if json_mode and self.provider == "openai":
            kw["response_format"] = {"type": "json_object"}
        last = None
        for i in range(self.retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**kw)
                self.usage.add(resp, time.time() - t0, tag)
                content = resp.choices[0].message.content or ""
                if not content.strip() and i < self.retries - 1:
                    last = RuntimeError("empty completion")
                    continue
                return content
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2 ** i, 20))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")
