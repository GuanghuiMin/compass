"""One closed-loop AppWorld episode for a frozen CodeAct agent with a pluggable
history compressor.

Protocol (matches ACON / Trace / the paper):
  * agent prompt = ACON ICL template as one user message (+ system line);
  * conversation = prompt, then alternating assistant(code) / user(observation);
  * before every agent call, if ``compressor`` is set and
    tokens(prev_summary + absorbable turns) > budget, the absorbable turns
    (everything not yet summarized except the last ``preserve_last_k`` turns)
    are folded into a new summary, which is spliced into the task prompt as a
    ``<history_summary>`` block; the preserved turns stay raw;
  * stop on ``task_completed()`` or ``max_steps``; score with ``evaluate()``.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .. import compat
from ..eval.signatures import is_error, sigs
from .compressors import Compressor
from .llm import LLM
from .prompt import AGENT_SYSTEM, count_tokens, render_agent_prompt, turns_to_text

compat.install()

# Characters of each observation the agent (and compressor) may see. Trace's
# cohort used 4000, which truncates AppWorld's larger API listings (Spotify:
# 9.1k chars / 91 APIs) and makes several tasks unsolvable for every method;
# the end-to-end protocol therefore uses 10000. Override with COMPASS_OBS_LIMIT.
OBS_STORE_LIMIT = int(os.environ.get("COMPASS_OBS_LIMIT", "10000"))


def extract_code(response: str) -> str:
    """ACON's extraction: strip <think>, strip fences, else first fenced block."""
    text = re.sub(r"<\s*think[^>]*>", "", response, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
    # DeepSeek occasionally wraps code in its own markup, <｜DSML｜python> ... </｜DSML｜python>
    # (U+FF5C bars); treat it like a fence and strip any other <｜...｜> special tokens
    m = re.search(r"<｜DSML｜python>\s*(.*?)\s*</｜DSML｜python>", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    text = re.sub(r"</?｜[^｜>]*｜[^>]*>", "", text)
    for pat in (r"```python\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
    text = re.sub(r"^```python\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)
    return text.strip()


def build_messages(base_prompt: str, summary_block: str | None, raw_turns: list[dict]) -> list[dict]:
    first = base_prompt + (summary_block or "")
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": first}]
    for t in raw_turns:
        msgs.append({"role": "assistant", "content": t["code"]})
        msgs.append({"role": "user", "content": t["observation"][:OBS_STORE_LIMIT]})
    return msgs


def run_episode(task_id: str, agent: LLM, compressor: Compressor | None, *, budget: int = 4096,
                max_steps: int = 50, preserve_last_k: int = 1, experiment_name: str = "compass",
                out_path: str | Path | None = None, split: str = "test_normal") -> dict:
    from appworld import AppWorld

    t0 = time.time()
    rec = {"task_id": task_id, "split": split, "method": compressor.name if compressor else "full",
           "agent_model": agent.model, "budget": budget, "max_steps": max_steps, "obs_limit": OBS_STORE_LIMIT,
           "steps": [], "boundaries": [], "termination_reason": None, "error": None}
    turns: list[dict] = []
    summary: str | None = None
    absorbed_upto = 0
    peak_prompt_tokens = 0
    seen_exact: set[str] = set()

    world = AppWorld(task_id=task_id, experiment_name=experiment_name)
    try:
        rec["instruction"] = world.task.instruction
        base_prompt = render_agent_prompt(world)
        reason = "max_steps"
        for step in range(1, max_steps + 1):
            # ---- compaction check (before the agent call) ----
            if compressor is not None:
                absorbable = turns[absorbed_upto: max(absorbed_upto, len(turns) - preserve_last_k)]
                if absorbable:
                    load = count_tokens((summary or "") + turns_to_text(
                        [{**t, "observation": t["observation"][:OBS_STORE_LIMIT]} for t in absorbable]))
                    if load > budget:
                        prev = summary
                        tb = time.time()
                        try:
                            summary = compressor.compress(world.task.instruction, prev, absorbable)
                            status = "ok"
                            setup = getattr(compressor, "last_setup_code", None)
                            if setup:
                                world.execute(setup)   # externalized observations enter the session, not the prompt
                        except Exception as e:  # noqa: BLE001
                            status, summary = f"compressor_error: {e}", prev
                        rec["boundaries"].append({
                            "before_step": step, "absorbed_steps": [absorbable[0]["step"], absorbable[-1]["step"]],
                            "retained_steps": [t["step"] for t in turns[len(turns) - preserve_last_k:]],
                            "load_tokens": load, "prev_summary": prev, "summary": summary,
                            "summary_tokens": count_tokens(summary or ""), "status": status,
                            "wall": round(time.time() - tb, 2),
                            "extra": getattr(compressor, "last_extra", None)})
                        if status == "ok":
                            absorbed_upto = len(turns) - preserve_last_k
            block = compressor.render_context(summary) if (compressor is not None and summary) else None
            msgs = build_messages(base_prompt, block, turns[absorbed_upto:])
            ptok = sum(count_tokens(m["content"]) for m in msgs)
            peak_prompt_tokens = max(peak_prompt_tokens, ptok)

            # ---- agent call ----
            try:
                raw = agent.chat(msgs, tag="agent")
            except Exception as e:  # noqa: BLE001
                reason, rec["error"] = "agent_error", str(e)
                break
            code = extract_code(raw)
            if not code:
                # an empty/prose-only completion is usually a transient provider hiccup:
                # ask once more before ending the episode (same for every method)
                try:
                    raw = agent.chat(msgs + [{"role": "user", "content":
                                              "Your previous reply contained no code. Reply with the next Python code block only."}],
                                     tag="agent")
                    code = extract_code(raw)
                except Exception as e:  # noqa: BLE001
                    reason, rec["error"] = "agent_error", str(e)
                    break
            if not code:
                reason = "no_code"
                break
            obs = world.execute(code) or ""
            ex = sigs(code)
            refetch = any(s in seen_exact for s in ex[:1])
            seen_exact.update(ex)
            turns.append({"step": step, "raw_response": raw, "code": code, "observation": obs})
            rec["steps"].append({"step": step, "code": code, "raw_response": raw[:3000],
                                 "observation": obs[:OBS_STORE_LIMIT],
                                 "is_error": is_error(obs), "sigs": ex, "refetch": refetch,
                                 "prompt_tokens": ptok, "compressed": summary is not None})
            if world.task_completed():
                reason = "task_completed"
                break
        rec["termination_reason"] = reason
        tr = world.evaluate()
        rec["success"] = bool(tr.success)
        rec["score"] = float(tr.pass_percentage) / 100.0
        rec["n_tests"] = tr.num_tests
    except Exception as e:  # noqa: BLE001
        rec["termination_reason"], rec["error"] = "harness_error", f"{type(e).__name__}: {e}"
        rec.setdefault("success", False)
        rec.setdefault("score", 0.0)
    finally:
        try:
            world.close()
        except Exception:  # noqa: BLE001
            pass
    rec["n_steps"] = len(rec["steps"])
    rec["n_blocked"] = sum(s["is_error"] for s in rec["steps"])
    rec["n_refetch"] = sum(s["refetch"] for s in rec["steps"])
    rec["peak_prompt_tokens"] = peak_prompt_tokens
    rec["agent_usage"] = agent.usage.to_dict()
    rec["compressor_usage"] = compressor.llm.usage.to_dict() if (compressor and compressor.llm) else None
    rec["wall_seconds"] = round(time.time() - t0, 1)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec
