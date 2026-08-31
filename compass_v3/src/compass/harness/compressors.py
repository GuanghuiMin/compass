"""Compressor interface and the prompt-based baselines.

A compressor sees: the task instruction, the previous summary (or None), the
turns to absorb (list of {code, observation, raw_response}), and the budget.
It returns the new summary text. FIFO is the one non-generative baseline.
"""
from __future__ import annotations

import os

from jinja2 import Template

from .llm import LLM
from .prompt import count_tokens, load_prompt, turns_to_text

OBS_LIMIT = int(os.environ.get("COMPASS_OBS_LIMIT", "10000"))  # chars; see episode.OBS_STORE_LIMIT


def clip_turns(turns: list[dict]) -> list[dict]:
    return [{**t, "observation": (t.get("observation") or "")[:OBS_LIMIT]} for t in turns]


class Compressor:
    name = "base"

    def __init__(self, llm: LLM | None = None, budget: int = 4096):
        self.llm, self.budget = llm, budget

    def compress(self, task: str, prev_summary: str | None, turns: list[dict]) -> str:
        raise NotImplementedError

    def render_context(self, summary: str) -> str:
        """How the summary is spliced into the agent's conversation."""
        return f"\n\n<history_summary>\n{summary}\n</history_summary>\n\n"


class FIFO(Compressor):
    """Sliding window: drop oldest whole turns until the rendered history fits."""
    name = "fifo"

    def compress(self, task, prev_summary, turns):
        kept = list(clip_turns(turns))
        while kept and count_tokens(turns_to_text(kept)) > self.budget:
            kept.pop(0)
        return turns_to_text(kept)

    def render_context(self, summary):
        return f"\n\n<earlier_history>\n{summary}\n</earlier_history>\n\n"


class OpenClaw(Compressor):
    name = "openclaw"

    def compress(self, task, prev_summary, turns):
        hist = turns_to_text(clip_turns(turns))
        if prev_summary:
            user = Template(load_prompt("openclaw_update.jinja")).render(history=hist, prev_summary=prev_summary)
        else:
            user = Template(load_prompt("openclaw_first.jinja")).render(history=hist)
        return self.llm.chat([{"role": "system", "content": load_prompt("openclaw_system.jinja")},
                              {"role": "user", "content": user}], tag="compress").strip()


class Hermes(Compressor):
    name = "hermes"
    PREFIX = ("[CONTEXT COMPACTION --- REFERENCE ONLY] Earlier turns were compacted into the summary below. "
              "This is a handoff from a previous context window --- treat it as background reference, NOT as "
              "active instructions. Do NOT answer questions or fulfill requests mentioned in this summary; they "
              "were already addressed. Respond ONLY to the latest user message that appears AFTER this summary. "
              "If the latest user message contradicts, supersedes, or changes topic from '## Active Task' / "
              "'## In Progress' / '## Pending User Asks' / '## Remaining Work', the latest message WINS --- "
              "discard those stale items entirely. The current session state (files, config, etc.) may reflect "
              "work described here --- avoid repeating it:\n\n")

    def compress(self, task, prev_summary, turns):
        hist = turns_to_text(clip_turns(turns))
        if prev_summary:
            hist = f"<previous-summary>\n{prev_summary}\n</previous-summary>\n\nNEW TURNS:\n{hist}"
        user = Template(load_prompt("hermes.jinja")).render(history=hist, summary_budget=self.budget // 4)
        out = self.llm.chat([{"role": "user", "content": user}], tag="compress").strip()
        return self.PREFIX + out


class AconUT(Compressor):
    name = "acon_ut"
    template = "acon_ut.jinja"

    def compress(self, task, prev_summary, turns):
        hist = turns_to_text(clip_turns(turns))
        user = Template(load_prompt(self.template)).render(task=task, prev_summary=prev_summary or "(none)",
                                                           history=hist, max_chars=1500)
        sys = ("You are an agent tasked with extracting and refining a concise and optimized version of the "
               "context based on the user instruction and other provided information.")
        return self.llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                             tag="compress").strip()


class AconUTCO(AconUT):
    name = "acon_utco"
    template = "acon_utco.jinja"


class OpenClawMem(OpenClaw):
    """OpenClaw with the same environment-side memory interface as COMPASS+ExternalMemory:
    the absorbed observations are stored parsed in the session as _mem['sN'] and the
    summary gets an index of the keys. Fairness control for the externalization channel."""
    name = "openclaw_mem"

    def compress(self, task, prev_summary, turns):
        from ..eval.signatures import is_error, sigs
        from ..graph.compressor import mem_setup_code, mem_value
        from ..graph.project import parse
        self.last_setup_code = None
        base = prev_summary.split(NL_MARK, 1)[0].strip() if prev_summary and NL_MARK in prev_summary else prev_summary
        text = super().compress(task, base, turns)
        saved, lines = {}, []
        for t in clip_turns(turns):
            obs = (t.get("observation") or "").strip()
            calls = sigs(t.get("code") or "")
            if not calls or is_error(obs) or len(obs) <= 40 or obs == "Execution successful.":
                continue
            key = f"s{t['step']}"
            saved[key] = obs
            obj = parse(obs)
            shape = (f"list of {len(obj)}" if isinstance(obj, list) else "dict" if isinstance(obj, dict) else "str")
            lines.append(f"- _mem['{key}'] ({shape}) {calls[0][:80]} -> {obs[:90].replace(NL_CHR, ' ')}")
        if saved:
            self.last_setup_code = mem_setup_code(saved)
        index = getattr(self, "_index", [])
        index = (index + lines)[-40:]
        self._index = index
        if index:
            text += NL_CHR + NL_CHR + NL_MARK + NL_CHR + NL_CHR.join(index)
        return text

    def render_context(self, summary):
        return ("\n\n<history_summary>\n" + summary + "\n\nResults marked _mem['sN'] are saved in the Python session "
                "as parsed objects: use them directly instead of calling the API again.\n</history_summary>\n\n")



NL_MARK = "SAVED RESULTS (full objects kept in the Python session; index them instead of re-calling)"
NL_CHR = "\n"


BASELINES = {c.name: c for c in (FIFO, OpenClaw, OpenClawMem, Hermes, AconUT, AconUTCO)}
