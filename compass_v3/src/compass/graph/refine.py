"""Local LLM refinement: the only model call in a COMPASS boundary.

The model sees the goal, the current plan (intents with grounded statuses), the
live variables, facts, and the new steps in compact evidence form. It returns
JSON *proposals*; every item is validated by ``Graph`` and dropped
individually when illegal. There is no whole-response rejection.
"""
from __future__ import annotations

import json
import re

from ..harness.llm import LLM
from .build import Graph, compact_step_view

SYSTEM = """You maintain the PLAN GRAPH of a coding agent that solves a task by executing Python cells against app APIs.
You never execute anything. You read the evidence and update a small plan: which sub-goals exist, which are done
(only if a step's observation proves it), which are blocked, what information each remaining sub-goal needs, and
which durable facts/constraints matter for the remaining work. Be concrete: name APIs, variables, ids, amounts.
Return ONLY a JSON object."""

USER = """GOAL:
{goal}
{legacy}
CURRENT PLAN (id | parent | status | description | evidence steps | needs):
{plan}

LIVE VARIABLES (id: name = source/api -> value hint):
{infos}

FACTS SO FAR:
{facts}

NEW STEPS SINCE THE LAST UPDATE (id [status] apis defs | agent comment -> observation excerpt):
{steps}

Return JSON:
{{
 "plan_updates": [
   {{"id": "c3" or null (null = create), "parent": "c1" or null, "description": "...",
     "status": "pending|active|done|blocked|invalidated", "evidence": ["s12"], "needs": ["i15"],
     "note": "one short line, e.g. why blocked or what remains" }}
 ],
 "realizes": {{"s12": "c2"}},
 "facts": [ {{"kind": "constraint|fact|failure|data", "text": "...", "steps": ["s3"]}} ],
 "remove_facts": ["f2"]
}}
Rules:
- First update: create a top-level decomposition of the GOAL (3-8 intents), then refine only the sub-goals that are
  active now; leave distant work coarse.
- 'done' requires evidence steps whose observation proves completion (e.g. API returned success for every item).
  A printed claim like "all done" without the corresponding successful API calls is NOT done.
- 'blocked' requires a failing step; put the exact error and the fix idea in note.
- If new evidence invalidates a sub-goal or a chosen route (API does not exist, wrong assumption), mark it invalidated
  and add the corrected sub-goal.
- needs = variable ids the sub-goal will read (tokens, id lists, computed sets). Only ids from LIVE VARIABLES.
- facts: constraints from the goal or observations that future steps must respect (definitions like which contacts
  count as 'friends', pagination limits, note strings, amounts), and failures worth not repeating. Keep each <= 200 chars.
  EVERY fact must cite the step ids it comes from in "steps" (a fact about the goal cites the first step); facts
  without a citation are discarded.
- kind "data": CONCRETE VALUES read from observations that the remaining sub-goals will need and that are not
  simply a variable still bound in the session -- e.g. "Brenda (9312015677) asked: add 'Song A' by X, remove
  'Song B'", "playlist 'Roadtrip' id=654 has songs [12, 40, 88]", "invoice total 128.00, split 4 ways = 32.00".
  Write each as one dense line (<= 200 chars) with the exact ids/names/amounts; cite the steps. These survive
  compaction verbatim, so include everything the plan's open items depend on.
- Do not restate steps as sub-goals; sub-goals are outcomes.
- NEVER conclude that an API or capability does not exist unless a COMPLETE, untruncated API listing for that app
  was observed. A failed guess of an API name (404 / "No API named ...") only proves that name is wrong.
  Record such attempts as failures ("tried spotify.play -> no such API"), not as absence of capability."""


def _plan_lines(g: Graph) -> str:
    if not g.intents:
        return "(empty)"
    out = []
    for it in sorted(g.intents.values(), key=lambda x: x.order):
        needs = ",".join(it.needs) or "-"
        ev = ",".join(it.evidence[-4:]) or "-"
        out.append(f"{it.id} | {it.parent or '-'} | {it.status} | {it.description} | {ev} | {needs}"
                   + (f" | note: {it.note}" if it.note else ""))
    return "\n".join(out)


def _info_lines(g: Graph, limit: int = 60) -> str:
    live = [i for i in g.live_infos(recent_steps=6) if i.kind == "runtime_reference"]
    live = live[-limit:]
    if not live:
        return "(none)"
    return "\n".join(f"{i.id}: {i.name} = {i.source_api or 'computed'} -> {i.value_hint or '?'}" for i in live)


def _fact_lines(g: Graph) -> str:
    return "\n".join(f"{f['id']} [{f['kind']}] {f['text']}" for f in g.facts) or "(none)"


def _json_load(text: str) -> dict | None:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    # truncated output: keep the complete plan_updates items that were emitted
    try:
        dec = json.JSONDecoder()
        i = text.find('"plan_updates"')
        j = text.find("[", i)
        items, k = [], j + 1
        while j >= 0:
            while k < len(text) and text[k] in " \n\r\t,":
                k += 1
            if k >= len(text) or text[k] != "{":
                break
            try:
                obj, end = dec.raw_decode(text, k)
            except json.JSONDecodeError:
                break                      # the item that was cut off
            items.append(obj)
            k = end
        return {"plan_updates": items, "_truncated": True} if items else None
    except Exception:  # noqa: BLE001
        return None


def refine_graph(g: Graph, new_step_ids: list[str], llm: LLM, *, obs_chars: int = 220) -> dict:
    steps = "\n".join(compact_step_view(g.steps[s], obs_chars) for s in new_step_ids) or "(none)"
    legacy = f"\nPREVIOUS (non-graph) SUMMARY, treat as evidence:\n{g.legacy_summary}\n" if g.legacy_summary else ""
    user = USER.format(goal=g.goal, legacy=legacy, plan=_plan_lines(g), infos=_info_lines(g),
                       facts=_fact_lines(g), steps=steps)
    raw = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                   tag="refine", json_mode=True, max_tokens=4096)
    prop = _json_load(raw)
    stats = {"parsed": prop is not None, "applied": 0, "dropped_before": len(g.log)}
    if prop is None:
        g.log.append({"drop": "unparseable_json"})
        return stats
    pending_status = []
    for u in prop.get("plan_updates") or []:
        if not isinstance(u, dict) or not u.get("description"):
            g.log.append({"drop": "update_missing_description", "item": str(u)[:120]})
            continue
        iid = u.get("id")
        if iid in (None, "", "null") or iid not in g.intents:
            it = g.add_intent(u["description"], parent=u.get("parent") or None, iid=iid if isinstance(iid, str) else None)
            if it is None:
                continue
            iid = it.id
        else:
            g.intents[iid].description = u["description"].strip()[:300]
            if u.get("parent") in g.intents and u["parent"] != iid and u["parent"] != g.intents[iid].parent:
                old = g.intents[iid].parent
                if old and iid in g.intents[old].children:
                    g.intents[old].children.remove(iid)
                g.intents[iid].parent = u["parent"]
                g.intents[u["parent"]].children.append(iid)
        pending_status.append((iid, u))
        stats["applied"] += 1
    for iid, u in pending_status:
        if u.get("status"):
            g.set_status(iid, u["status"], list(u.get("evidence") or []), u.get("note"))
        g.set_needs(iid, [n for n in (u.get("needs") or []) if isinstance(n, str)])
    for iid in list(g.intents):
        seen, cur = set(), iid
        while cur:
            if cur in seen:
                g.intents[iid].parent = None
                g.log.append({"drop": "cycle_broken", "intent": iid})
                break
            seen.add(cur)
            cur = g.intents[cur].parent
    for sid, cid in (prop.get("realizes") or {}).items():
        g.set_realizes(str(sid), str(cid))
    rm = set(prop.get("remove_facts") or [])
    if rm:
        g.facts = [f for f in g.facts if f["id"] not in rm]
    for f in prop.get("facts") or []:
        if isinstance(f, dict) and f.get("text"):
            g.add_fact(f["text"], f.get("kind", "fact"), f.get("steps") or [])
    stats["dropped"] = len(g.log) - stats["dropped_before"]
    return stats
