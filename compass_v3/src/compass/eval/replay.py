"""Trace-style boundary replay verifier.

For a recorded episode and one of its compaction boundaries, both arms restore
the identical environment state by re-executing the recorded code prefix into a
fresh AppWorld (no model in the loop), then show the frozen agent exactly one
context and free-run at most ``k_max`` actions:

  PRE   first boundary: the full raw prefix; later boundary: the previous
        summary block plus the raw suffix since the previously retained turn.
  POST  a summary block plus the single retained raw turn.

POST's summary defaults to the one the episode recorded, but ``summary`` may be
overridden (a different compressor's output for the same absorbable turns) --
that is how Phase 2 scores COMPASS and its ablations on a shared cohort.
"""
from __future__ import annotations

import hashlib
import json
import time

from .. import compat
from ..harness.episode import OBS_STORE_LIMIT, build_messages, extract_code
from ..harness.llm import LLM
from ..harness.prompt import render_agent_prompt
from .signatures import is_error, sigs

compat.install()
K_MAX = 5


def boundary_specs(ep: dict) -> list[dict]:
    """One spec per recorded (status ok) boundary of an episode."""
    steps = {s["step"]: s for s in ep["steps"]}
    out = []
    prev_retained_lo = 1
    for i, b in enumerate(ep["boundaries"]):
        if b["status"] != "ok":
            continue
        n = b["before_step"]
        prefix = [steps[j]["code"] for j in range(1, n) if j in steps]
        retained = b["retained_steps"]
        first = b["prev_summary"] is None
        suffix_lo = prev_retained_lo if not first else 1
        pre_turns = [steps[j] for j in range(suffix_lo, n) if j in steps]
        post_turns = [steps[j] for j in retained if j in steps]
        absorbable = [steps[j] for j in range(b["absorbed_steps"][0], b["absorbed_steps"][1] + 1) if j in steps]
        hist = set()
        for j in range(1, n):
            if j in steps:
                hist.update(steps[j]["sigs"])
        out.append({
            "boundary_id": f"{ep['method']}::{ep['task_id']}#t{i + 1}",
            "task_id": ep["task_id"], "method": ep["method"], "index": i + 1,
            "before_step": n, "first": first, "prefix_codes": prefix,
            "replay_identity": hashlib.sha256(json.dumps([ep["task_id"], n, prefix]).encode()).hexdigest()[:16],
            "pre": {"summary": b["prev_summary"], "turns": pre_turns},
            "post": {"summary": b["summary"], "turns": post_turns},
            "absorbable_turns": absorbable, "prev_summary": b["prev_summary"],
            # every recorded step before the boundary except the retained turn(s): what a
            # compressor that had been running all along would have absorbed by now
            "prefix_turns": [steps[j] for j in range(1, n) if j in steps and j not in retained],
            "historical_exact": sorted(hist),
        })
        prev_retained_lo = retained[0] if retained else n
    return out


def rollout(spec: dict, arm: str, agent: LLM, *, summary: str | None = None,
            render_context=None, k_max: int = K_MAX, sample: int = 0,
            experiment_name: str = "replay", setup_code: str | None = None) -> dict:
    from appworld import AppWorld

    ctx = spec[arm.lower()]
    summ = ctx["summary"] if summary is None else summary
    render = render_context or (lambda s: f"\n\n<history_summary>\n{s}\n</history_summary>\n\n")
    rec = {"boundary_id": spec["boundary_id"], "task_id": spec["task_id"], "arm": arm, "sample": sample,
           "replay_identity": spec["replay_identity"], "summary_used": summ, "steps": [],
           "termination": None, "error": None}
    t0 = time.time()
    world = AppWorld(task_id=spec["task_id"], experiment_name=f"{experiment_name}_{arm}_{sample}")
    try:
        for code in spec["prefix_codes"]:
            world.execute(code)
        if setup_code:
            world.execute(setup_code)
        base = render_agent_prompt(world)
        turns = [{"code": t["code"], "observation": t["observation"]} for t in ctx["turns"]]
        block = render(summ) if summ else None
        msgs = build_messages(base, block, turns)
        hist = set(spec["historical_exact"])
        seen: set[str] = set()
        rec["termination"] = "reached_max_actions"
        for i in range(k_max):
            raw = agent.chat(msgs, tag="replay")
            code = extract_code(raw)
            if not code:
                rec["termination"] = "no_code"
                break
            obs = world.execute(code) or ""
            ss = sigs(code)
            ex = ss[0] if ss else None
            refetch = ex is not None and (ex in hist or ex in seen)
            seen.update(ss)
            rec["steps"].append({"index": i + 1, "code": code, "observation": obs[:OBS_STORE_LIMIT],
                                 "is_error": is_error(obs), "exact": ex, "refetch": refetch,
                                 "union": is_error(obs) or refetch})
            msgs.append({"role": "assistant", "content": code})
            msgs.append({"role": "user", "content": obs[:OBS_STORE_LIMIT]})
            if world.task_completed():
                rec["termination"] = "task_completed"
                break
    except Exception as e:  # noqa: BLE001
        rec["termination"], rec["error"] = "harness_error", f"{type(e).__name__}: {e}"
    finally:
        try:
            world.close()
        except Exception:  # noqa: BLE001
            pass
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec
