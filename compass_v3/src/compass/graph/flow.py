"""Flow-graph view of COMPASS: information is attached to the future action nodes
that will use it, absorbed evidence is pruned, and the checkpoint is rendered
plan-first.

Three operations, run at every compaction after the proposal call:

  attach_to_frontier   for each open intent, collect what it will need -- the
                       signatures and proven argument shapes of the APIs it names
                       (or its own evidence used), the variables it needs, the
                       data facts about the same entities, and the credentials --
                       and store them on the node (``Intent.carry``).
  prune_evidence       steps that are neither recent, nor blocked, nor the
                       producer of something still carried lose their raw
                       observation (a one-line excerpt remains): the graph keeps
                       provenance, not transcripts.
  render_flow          the plan tree is the checkpoint: done subtrees collapse to
                       one line with what they produced, open nodes list what they
                       carry, and only a small residual (constraints, failed
                       shapes, unattached data) is global.
"""
from __future__ import annotations

import re

from ..harness.prompt import count_tokens
from .build import Graph, Intent, _excerpt

API_MENTION = re.compile(r"\b([a-z_]+\.[a-z_]+)\b")
CRED = re.compile(r"token|password|login|session|auth|cred", re.I)
STOP = {"the", "and", "for", "with", "from", "that", "this", "each", "all", "into", "then", "their", "them",
        "have", "has", "are", "was", "were", "use", "using", "get", "list", "find", "make", "check", "which"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]{4,}", (text or "").lower()) if w not in STOP}


def attach_to_frontier(g: Graph, recent_ids: list[str], *, per_node: int = 6) -> int:
    """Fill ``Intent.carry`` for every open intent. Returns the number of attachments."""
    open_intents = [it for it in sorted(g.intents.values(), key=lambda x: x.order)
                    if it.status not in ("done", "invalidated")]
    if not open_intents:
        return 0
    specs = {i.name: i for i in g.infos.values() if i.kind == "api_spec"}
    forms_by_api: dict[str, str] = {}
    for form in g.calls_ok:
        forms_by_api.setdefault(form.split("(", 1)[0], form)
    live = [i for i in g.infos.values() if i.kind == "runtime_reference" and not i.superseded]
    creds = [i for i in live if CRED.search(i.name)]
    data_facts = [f for f in g.facts if f["kind"] == "data"]
    results = [i for i in g.infos.values() if i.kind == "api_result" and i.value_hint]
    total = 0
    placed: dict[str, str] = {}      # payload key -> intent id that already carries it
    for pos, it in enumerate(open_intents):
        desc = it.description.lower()
        toks = _tokens(desc)
        carry: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(kind: str, payload: str):
            key = kind + payload[:60]
            if not payload or key in seen:
                return
            seen.add(key)
            if key in placed:
                carry.append(("ref", f"{payload.split('(', 1)[0].split(' =', 1)[0][:40]} (see {placed[key]})"))
            else:
                placed[key] = it.id
                carry.append((kind, payload))

        # (a) interface knowledge for the APIs this node names or already touched
        used = {a for e in it.evidence if e in g.steps for a in g.steps[e]["api_names"]}
        named = set(API_MENTION.findall(desc)) | used
        for name, spec in specs.items():
            verb = name.split(".")[-1].replace("_", " ")
            if name in named or (len(verb) > 5 and verb in desc):
                add("api", spec.value_hint or name)
                if name in forms_by_api:
                    add("call", forms_by_api[name])
        for name, form in forms_by_api.items():
            if name in named:
                add("call", form)
        # (b) values this node needs (proposal + deterministic needs)
        for n in it.needs:
            i = g.infos.get(n)
            if i and i.kind == "runtime_reference" and not i.superseded and not CRED.search(i.name):
                add("var", i.name + (f" = {i.value_hint}" if i.value_hint else
                                     (f" = {i.source_api}(...)" if i.source_api else "")))
        # (c) values produced by this node's own or its parent's evidence
        parent = g.intents.get(it.parent) if it.parent else None
        for src in ([it] + ([parent] if parent else [])):
            for e in src.evidence:
                st = g.steps.get(e)
                if not st:
                    continue
                for iid in st["produces"]:
                    i = g.infos.get(iid)
                    if i and i.kind == "runtime_reference" and not i.superseded and i.value_hint \
                            and not CRED.search(i.name):
                        add("var", f"{i.name} = {i.value_hint}")
        # (d) data facts and observed results about the same entities
        for f in data_facts:
            if len(toks & _tokens(f["text"])) >= 2:
                add("data", f["text"])
        for r in results[-20:]:
            if len(toks & _tokens(r.name + " " + (r.value_hint or ""))) >= 2:
                add("result", f"{_excerpt(r.name, 70)} -> {r.value_hint}")
        firsts = [c for c in carry if c[0] != "ref"]
        refs = [c for c in carry if c[0] == "ref"]
        it.carry = firsts[:per_node] + refs[:3]
        # anything that fell off this node's cap must not be referenced by later nodes
        kept = {k + p[:60] for k, p in it.carry}
        for key, owner in list(placed.items()):
            if owner == it.id and key not in kept:
                del placed[key]
        total += len(firsts)
    # credentials are rendered once, globally (every call needs them; they must never be cap-cut)
    g.credentials = [c.name + (f" = {_excerpt(c.value_hint, 80)}" if c.value_hint else
                              (f" = {c.source_api}(...)" if c.source_api else "")) for c in creds[-6:]]
    return total


NL_SYSTEM = ("You rewrite a structured checkpoint of an agent's progress into a concise natural-language handover "
             "for the same agent. Rules: state only what the checkpoint contains -- do not add, guess, or drop "
             "any id, value, signature, argument name or status; keep every value verbatim; keep the plan order; "
             "make explicit what is done, what is next, and what each next step needs. No preamble.")


def narrate(text: str, llm, budget_tokens: int) -> str:
    """Optional final surface: the LLM narrates the structured checkpoint (content is fixed by the graph)."""
    out = llm.chat([{"role": "system", "content": NL_SYSTEM},
                    {"role": "user", "content": f"Target length: at most {budget_tokens} tokens.\n\nCHECKPOINT:\n{text}"}],
                   tag="narrate", max_tokens=min(4096, budget_tokens + 400)).strip()
    return out or text


def prune_evidence(g: Graph, recent_ids: list[str], *, keep_last: int = 6) -> int:
    """Drop raw observations of absorbed steps. A step keeps its observation if it is
    recent, blocked (its error is a warning), or produced a live variable that some
    open node carries. Returns the number of steps pruned."""
    order = sorted(g.steps, key=lambda s: int(s[1:]))
    keep = set(order[-keep_last:]) | set(recent_ids)
    carried_vars = set()
    for it in g.intents.values():
        for kind, payload in getattr(it, "carry", []):
            if kind == "var":
                carried_vars.add(payload.split(" =", 1)[0].strip())
    pruned = 0
    for sid in order:
        st = g.steps[sid]
        if sid in keep or st.get("pruned") or st["status"] == "blocked":
            continue
        if any(g.infos[i].kind == "runtime_reference" and not g.infos[i].superseded and g.infos[i].name in carried_vars
               for i in st["produces"] if i in g.infos):
            continue
        st["observation"] = _excerpt(st["observation"], 160)
        st["pruned"] = True
        pruned += 1
    return pruned


def _subtree_done(g: Graph, it: Intent) -> bool:
    if it.status not in ("done", "invalidated"):
        return False
    return all(_subtree_done(g, g.intents[c]) for c in it.children if c in g.intents)


def _tree(g: Graph):
    roots = [it for it in g.intents.values() if it.parent is None or it.parent not in g.intents]
    out = []

    def walk(it: Intent, depth: int):
        out.append((it, depth))
        for c in it.children:
            if c in g.intents:
                walk(g.intents[c], depth + 1)
    for r in sorted(roots, key=lambda x: x.order):
        walk(r, 0)
    return out


MARK = {"done": "[x]", "blocked": "[!]", "invalidated": "[-]", "active": "[>]", "pending": "[ ]"}
TAG = {"api": "api ", "call": "call", "var": "var ", "data": "data", "result": "seen", "ref": "ref "}


def render_flow(g: Graph, level: int, recent_ids: list[str]) -> str:
    """Plan-first checkpoint. ``level`` 0..3 shrinks what each node carries."""
    cap, chars = {0: (10, 140), 1: (6, 100), 2: (3, 60), 3: (2, 40)}[level]
    # carried values are what the agent will act on: never cut them below the v2 floor while budget allows
    vchars = {0: 900, 1: 400, 2: 160, 3: 60}[level]
    L: list[str] = ["GOAL", g.goal.strip()]
    cons = [f for f in g.facts if f["kind"] == "constraint"]
    if cons:
        L += ["", "CONSTRAINTS"] + [f"- {_excerpt(f['text'], 160)}" for f in cons[-6:]]
    executed_all: list[str] = []
    for s in sorted(g.steps, key=lambda x: int(x[1:])):
        if g.steps[s]["status"] == "ok":
            executed_all += [a for a in g.steps[s]["api_names"] if not a.startswith("api_docs")]
    tree = _tree(g)
    creds = getattr(g, "credentials", [])
    if creds:
        L += ["", "CREDENTIALS (still bound; reuse, do not re-login): " + "; ".join(creds)]
    shown_keys: set[str] = set()
    rendered_nodes: set[str] = set()
    if tree:
        L += ["", "PLAN  ([x] done  [>] active  [ ] pending  [!] blocked  [-] invalidated; open nodes list what they carry)"]
        skip: set[str] = set()
        for it, depth in tree:
            if it.parent in skip:
                skip.add(it.id)
                continue
            ind = "  " * min(depth, 2)
            if it.status in ("done", "invalidated") and _subtree_done(g, it):
                skip.add(it.id)
                line = f"{ind}{MARK[it.status]} {it.id} {_excerpt(it.description, 140)}"
                apis: list[str] = []
                for e in it.evidence:
                    st = g.steps.get(e)
                    if st and st["status"] == "ok":
                        apis += [a for a in st["api_names"] if not a.startswith("api_docs")]
                apis = list(dict.fromkeys(apis))
                if apis and level <= 1:
                    line += f"  (did: {', '.join(apis[:5])})"
                prod = g.produced_by_intent(it, limit=3)
                if prod:
                    line += "  -> " + "; ".join(f"{i.name}={_excerpt(i.value_hint, 50 if level <= 1 else 25)}" for i in prod)
                if it.status == "invalidated" and it.note:
                    line += f"  [{_excerpt(it.note, 60)}; agent's own conclusion, re-verify]"
                L.append(line)
                continue
            line = f"{ind}{MARK[it.status]} {it.id} {_excerpt(it.description, 160)}"
            if it.note and it.status in ("blocked", "active"):
                line += f"  -- {_excerpt(it.note, 100)}"
            L.append(line)
            shown = getattr(it, "carry", [])[:cap]
            shown_keys.update(k + p[:60] for k, p in shown)
            for kind, payload in shown:
                if kind == "ref":
                    target = payload.split("(see ", 1)[-1].rstrip(")")
                    if target not in rendered_nodes:
                        continue
                L.append(f"{ind}    {TAG[kind]}: {_excerpt(payload, vchars if kind in ('var', 'result', 'data') else chars + (60 if kind == 'api' else 0))}")
            rendered_nodes.add(it.id)
    # ---- global evidence layer (kept: it is where the measured gains come from); items a node
    # already carries are not repeated here
    # only items actually rendered on a node count as carried; cap-cut ones fall back to the global sections
    carried_short = {k[4:] if k.startswith(("api", "call", "var", "data")) else k for k in shown_keys}
    carried_short = {p[:60] for k, p in ((k2, p2) for it in g.intents.values() for k2, p2 in getattr(it, "carry", []))
                     if k + p[:60] in shown_keys}
    forms = [f for f in list(g.calls_ok) if f[:60] not in carried_short]
    if forms and level <= 2:
        L += ["", "CALL SHAPES THAT WORKED (argument names; reuse, do not re-login)"] + [f"- {f}" for f in forms[-(24 if level <= 1 else 12):]]
    if g.calls_failed and level <= 2:
        L += ["", "CALL SHAPES THAT FAILED"] + [f"- {f} -> {_excerpt(e, 80)}" for f, e in list(g.calls_failed.items())[-4:]]
    specs = [i for i in g.infos.values() if i.kind == "api_spec" and (i.value_hint or "")[:60] not in carried_short]
    if specs and level <= 2:
        L += ["", "API SIGNATURES READ (exact; do not re-read docs)"] + [
            f"- {_excerpt(i.value_hint, 200 if level <= 1 else 120)}" for i in specs[-(20 if level <= 1 else 8):]]
    lists = [i for i in g.infos.values() if i.kind == "api_list" and i.name != "apps"]
    if lists and level <= 2:
        # the per-app api-name listing is what stops api_docs re-reads (v2 kept it through level 2)
        L += ["", "APPS EXPLORED (api names already listed; do not call api_docs.show_api_descriptions again)"]
        for i in lists[-6:]:
            hint = i.value_hint or ""
            if level == 2 and len(hint) > 300:
                hint = hint[:300] + " ... (more; listing shortened here)"
            L.append(f"- {i.name} apis: {hint}")
    carried_vars = {p.split(" =", 1)[0].strip() for it in g.intents.values() for k, p in getattr(it, "carry", [])
                    if k == "var" and k + p[:60] in shown_keys}
    cred_names = {c.split(" =", 1)[0].strip() for c in creds}
    live = [i for i in g.infos.values() if i.kind == "runtime_reference" and not i.superseded
            and i.name not in carried_vars and i.name not in cred_names]
    if live:
        # needed-by-frontier values keep their hint at every level (v2's floor); the rest thin out
        needed = set()
        for it in g.frontier():
            needed.update(g.infos[n].name for n in it.needs if n in g.infos)
        recent = set(recent_ids)
        live.sort(key=lambda i: (i.name not in needed, not any(c in recent for c in i.consumers)))
        L += ["", "OTHER LIVE VARIABLES (still bound in the Python session)"]
        for i in live[:(30 if level <= 1 else 14)]:
            src = f" = {i.source_api}(...)" if i.source_api else ""
            keep_hint = i.value_hint and (level <= 1 or i.name in needed)
            hint = f" -> {_excerpt(i.value_hint, 100 if level <= 1 else 60)}" if keep_hint else ""
            L.append(f"- {i.name}{src}{hint}")
    results = [i for i in g.infos.values() if i.kind == "api_result" and i.value_hint]
    if results and level <= 2:
        keep, rchars = (20, 160) if level <= 1 else (12, 90)
        L += ["", "RESULTS ALREADY OBSERVED (printed, not stored; do not call again)"] + [
            f"- {_excerpt(i.name, 70)} -> {_excerpt(i.value_hint, rchars)}" for i in results[-keep:]]
    if executed_all and level <= 2:
        L += ["", "ALREADY EXECUTED: " + ", ".join(list(dict.fromkeys(executed_all))[-25:])]
    carried = {p for it in g.intents.values() for k, p in getattr(it, "carry", []) if k + p[:60] in shown_keys}
    loose = [f for f in g.facts if f["kind"] == "data" and f["text"] not in carried]
    if loose and level <= 1:
        L += ["", "OTHER DATA EXTRACTED"] + [f"- {_excerpt(f['text'], 160)}" for f in loose[-8:]]
    other = [f for f in g.facts if f["kind"] in ("fact", "failure")]
    if other and level <= 1:
        L += ["", "NOTES"] + [f"- {_excerpt(f['text'], 140)}" for f in other[-5:]]
    front = g.frontier()
    if front:
        L += ["", "NEXT: " + "; ".join(f"{it.id} {_excerpt(it.description, 80)}" for it in front[:4])]
    return "\n".join(L)


def render_flow_to_budget(g: Graph, budget: int, recent_ids: list[str]) -> tuple[str, int]:
    text = ""
    for level in (0, 1, 2, 3):
        text = render_flow(g, level, recent_ids)
        if count_tokens(text) <= budget:
            return text, level
    return text, 4
