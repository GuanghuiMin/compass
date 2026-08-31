"""Agent prompt (ACON default ICL template, sent as one user message) and
conversation-to-text rendering shared by every compressor."""
from __future__ import annotations

from pathlib import Path

import tiktoken
from jinja2 import Template

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
_ENC = tiktoken.get_encoding("cl100k_base")

AGENT_SYSTEM = ("You are an AI assistant that writes Python code to complete tasks. You should respond "
                "with clear, executable Python code to interact with APIs and solve the given task "
                "completely autonomously.")


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def render_agent_prompt(world) -> str:
    t = Template(load_prompt("appworld_agent.jinja").lstrip())
    return t.render(supervisor=world.task.supervisor, instruction=world.task.instruction)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text or "", disallowed_special=()))


def turns_to_text(turns: list[dict]) -> str:
    """ACON/Trace history text: USER:/ASSISTANT: blocks, code then observation."""
    out = []
    for t in turns:
        out.append(f"ASSISTANT:\n{t['code']}\n\n")
        out.append(f"USER:\n{t['observation']}\n\n")
    return "".join(out)
