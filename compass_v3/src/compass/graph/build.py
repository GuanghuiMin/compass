"""The three-layer text-attributed action graph.

Layers
  Step    s<t>   one action-observation, fully deterministic (parse.py)
  Info    i<n>   a value the run holds: a runtime variable, an API result, a
                 constraint or a fact. PRODUCES / CONSUMES edges come from the
                 AST def/use analysis, so they are exact for CodeAct agents.
  Intent  c<n>   a plan item (goal decomposition). Intents are proposed by a
                 small LLM call (refine.py) and validated here; their status is
                 grounded in step evidence, never merely declared.

The graph is a plain data structure: every mutation goes through a method that
keeps it legal (no dangling references, parents exist, no cycles), so an LLM
proposal can only ever add legal structure or be dropped item by item.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field

from .parse import parse_step

STATUSES = ("pending", "active", "done", "blocked", "invalidated")


@dataclass
class Info:
    id: str
    kind: str                      # runtime_reference | api_result | fact | constraint | failure
    name: str                      # variable name or short label
    producer: str | None = None    # step id
    source_api: str | None = None  # app.api that produced it (if known)
    value_hint: str | None = None  # short canonical excerpt of the value
    value_full: str | None = None  # the observation the hint was cut from (for field-wise projection)
    description: str | None = None
    consumers: list[str] = field(default_factory=list)   # step ids that read it
    needed_by: list[str] = field(default_factory=list)   # intent ids predicted to need it
    superseded: bool = False       # a later def rebound the same name


@dataclass
class Intent:
    id: str
    description: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)    # step ids
    needs: list[str] = field(default_factory=list)       # info ids
    produces: list[str] = field(default_factory=list)    # info ids
    note: str | None = None                              # short LLM note (e.g. why blocked)
    order: int = 0
    carry: list = field(default_factory=list)            # (kind, payload) attached by flow.attach_to_frontier


class Graph:
    def __init__(self, goal: str):
        self.goal = goal
        self.steps: dict[str, dict] = {}
        self.infos: dict[str, Info] = {}
        self.intents: dict[str, Intent] = {}
        self.facts: list[dict] = []             # {id, kind, text, steps}
        self.step_intent: dict[str, str] = {}   # REALIZES
        self.legacy_summary: str | None = None  # a non-graph previous summary, kept verbatim-ish
        self.calls_ok: dict[str, int] = {}       # call form -> times it executed without error
        self.calls_failed: dict[str, str] = {}   # call form -> last error line (cleared on later success)
        self.mem_keys: list[str] = []            # step ids whose full observation was saved in the session (_mem)
        self.narrative: str | None = None        # LLM progress note (done / in progress / next), if enabled
        self.call_prefix: str = ""              # how the agent must spell a tool call (CodeAct: 'apis.')
        # bounded private state: raw observations are cut to obs_keep chars once extraction is done
        # (0 = keep whole, only for the externalization / projection variants), and the whole graph
        # payload is held under graph_budget chars (0 = unbounded) by evicting the oldest evidence first
        self.obs_keep: int = 600
        self.graph_budget: int = 0
        self.requirements: list[dict] = []       # V^requirement: {id, text, status, supports: [step ids]}
        self.req_graph: dict | None = None       # serialized ReqGraph (frontier-conditioned variants)
        self.protect_steps: set = set()          # evidence of active requirements: never evicted
        self.protect_infos: set = set()          # NEEDED_BY the frontier: never evicted, rendered first
        self.bytes_log: list[dict] = []          # graph payload size per boundary
        self._n_info = 0
        self._n_intent = 0
        self._n_fact = 0
        self.log: list[dict] = []               # dropped proposals, for diagnostics

    # ------------------------------------------------------------------ steps / infos
    def _latest_info_for(self, name: str) -> Info | None:
        cands = [i for i in self.infos.values() if i.kind == "runtime_reference" and i.name == name and not i.superseded]
        return cands[-1] if cands else None

    def add_turn(self, turn: dict, adapter: str = "codeact") -> dict:
        s = parse_step(turn, adapter)
        sid = f"s{s['step']}"
        s["id"] = sid
        s["produces"] = []
        s["consumes"] = []
        self.steps[sid] = s
        # call forms: which argument shapes worked, which failed
        forms = [f for f in s.get("call_forms", []) if not f.startswith("api_docs.")]
        if s["status"] == "ok":
            for f in forms:
                self.calls_ok[f] = self.calls_ok.get(f, 0) + 1
                self.calls_failed.pop(f, None)
        elif forms and s.get("error_line"):
            # attribute the error to the call named in the traceback when possible, else the last call
            culprit = next((f for f in forms if f.split("(")[0].split(".")[-1] in s["observation"]), forms[-1])
            if culprit not in self.calls_ok:
                self.calls_failed[culprit] = s["error_line"]
        # CONSUMES: free loads bound by an earlier cell
        for name in s["uses"]:
            info = self._latest_info_for(name)
            if info is not None:
                info.consumers.append(sid)
                s["consumes"].append(info.id)
        # PRODUCES: module-level defs
        api_by_var = _assigned_api_calls(s["code"])
        literals = _literal_assignments(s["code"])
        derived = _derived_from(s["code"])
        for name in s["defs"]:
            prev = self._latest_info_for(name)
            if prev is not None:
                prev.superseded = True
            src = api_by_var.get(name)
            if src is None and name in derived:
                base = self._latest_info_for(derived[name])
                if base is not None and base.source_api:
                    src = base.source_api + f"[{derived[name]}]"
            self._n_info += 1
            info = Info(id=f"i{self._n_info}", kind="runtime_reference", name=name, producer=sid,
                        source_api=src,
                        value_hint=literals.get(name) or _value_hint(s, name),
                        value_full=(None if literals.get(name) or not _value_hint(s, name)
                                    else s["observation"][:VALUE_FULL_CHARS]))
            self.infos[info.id] = info
            s["produces"].append(info.id)
        # API documentation read by the agent -> api_spec / api_list infos (deduped by name)
        for spec in s.get("api_specs", []):
            name = spec.split("(", 1)[0]
            old = next((i for i in self.infos.values() if i.kind == "api_spec" and i.name == name), None)
            if old is not None:
                old.value_hint, old.producer = spec, sid
                continue
            self._n_info += 1
            self.infos[f"i{self._n_info}"] = Info(id=f"i{self._n_info}", kind="api_spec", name=name, producer=sid,
                                                  source_api="api_docs.show_api_doc", value_hint=spec)
            s["produces"].append(f"i{self._n_info}")
        for app, names in (s.get("api_lists") or {}).items():
            old = next((i for i in self.infos.values() if i.kind == "api_list" and i.name == app), None)
            hint = ", ".join(names[:60])
            if s.get("api_list_truncated"):
                hint += " ... (listing was TRUNCATED by the output limit: more APIs exist; " \
                        "query api_docs.show_api_doc for specific names or use page_limit/keywords)"
            if old is not None:
                old.value_hint, old.producer = hint, sid
                continue
            self._n_info += 1
            self.infos[f"i{self._n_info}"] = Info(id=f"i{self._n_info}", kind="api_list", name=app, producer=sid,
                                                  source_api="api_docs.show_api_descriptions", value_hint=hint)
            s["produces"].append(f"i{self._n_info}")
        # api results not bound to a variable but printed (non-doc calls) -> api_result info;
        # under the generic adapter (no outcome, no program state) every tool observation is one
        if s["status"] in ("ok", "unknown") and s["api_names"] and not s["defs"] and not s.get("api_specs") \
                and not s.get("api_lists"):
            obs = s["observation"]
            # one result node per distinct call (signature with arguments), so a loop of
            # lookups keeps every answer instead of collapsing into one line per API
            keys = s["api_sigs"][:4] or s["api_names"]
            for key in keys:
                api = key.split("(", 1)[0]
                old = next((i for i in self.infos.values() if i.kind == "api_result" and i.name == key), None)
                hint = _excerpt(obs, SMALL_VALUE_CHARS if len(obs) <= SMALL_VALUE_CHARS else 240)
                if old is not None:
                    old.value_hint, old.producer, old.value_full = hint, sid, obs[:VALUE_FULL_CHARS]
                    continue
                self._n_info += 1
                info = Info(id=f"i{self._n_info}", kind="api_result", name=key, producer=sid, source_api=api,
                            value_hint=hint, value_full=obs[:VALUE_FULL_CHARS])
                self.infos[info.id] = info
                s["produces"].append(info.id)
        specs = getattr(self, "extract_specs", None)
        if specs and len(s["observation"] or "") > self.obs_keep:
            from .project import extract_fields
            for path, val in extract_fields(s["observation"], specs):
                self._n_info += 1
                info = Info(id=f"i{self._n_info}", kind="api_result", name=f"{path} @ {sid}",
                            producer=sid, source_api=(s["api_names"] or [None])[0],
                            value_hint=f"{path} = {val}")
                self.infos[info.id] = info
                s["produces"].append(info.id)
        if self.obs_keep:
            # Q_k = Extract(delta tau): everything above was read from the raw observation; from here on
            # only the bounded excerpt survives in the graph, the raw text is dropped
            s["observation"] = _excerpt(s["observation"], self.obs_keep)
            s["printed"] = {}          # parsed observation, only needed while extracting hints above
            for iid in s["produces"]:
                self.infos[iid].value_full = None
        return s

    # ------------------------------------------------------------------ bounded private state
    def payload_bytes(self) -> int:
        import json as _json
        return len(_json.dumps(self.to_dict(), ensure_ascii=False))

    def approx_bytes(self) -> int:
        """Cheap size estimate used while evicting: serializing the whole graph on every candidate
        turned eviction into an O(n^2) json.dumps loop (5.7s per boundary at 300 info nodes)."""
        n = len(self.goal) + 400 + len(self.narrative or "")
        for s in self.steps.values():
            n += len(s.get("code", "")) + len(s.get("observation", "")) + 420
        for i in self.infos.values():
            n += len(i.value_hint or "") + len(i.value_full or "") + len(i.name or "") + 200
        n += sum(len(k) + 12 for k in self.calls_ok) + sum(len(k) + len(v) + 8 for k, v in self.calls_failed.items())
        n += sum(len(str(f)) for f in self.facts) + 120 * len(self.requirements)
        if self.req_graph:
            n += 300 * len(self.req_graph.get("nodes", {}))
        return n

    def enforce_budget(self) -> dict:
        """Hold the serialized graph under graph_budget chars: oldest steps lose their observation
        excerpt and code first, then the oldest result hints shrink, then superseded infos go, and
        finally the oldest unprotected results are dropped outright (needs-conditioned extraction
        can create hundreds of small result nodes that none of the earlier levers can shrink)."""
        before = self.payload_bytes()
        if not self.graph_budget or before <= self.graph_budget:
            self.bytes_log.append({"n_steps": len(self.steps), "bytes": before, "evicted": 0})
            return {"before": before, "after": before, "evicted": 0}
        # evict against the estimate, scaled so that it agrees with the real size right now
        scale = before / max(1, self.approx_bytes())
        budget = self.graph_budget / max(scale, 0.2)
        evicted = 0
        for s in self.steps.values():
            if self.approx_bytes() <= budget:
                break
            if s["id"] in self.protect_steps:
                continue
            if s.get("observation") or len(s.get("code", "")) > 200:
                s["observation"] = ""
                s["code"] = s.get("code", "")[:200]
                evicted += 1
        for i in self.infos.values():
            if self.approx_bytes() <= budget:
                break
            if i.id in self.protect_infos:
                continue
            if i.kind == "api_result" and i.value_hint and len(i.value_hint) > 90:
                i.value_hint = i.value_hint[:90] + "..."
                evicted += 1
        for k, i in list(self.infos.items()):
            if self.approx_bytes() <= budget:
                break
            if i.superseded and k not in self.protect_infos:
                del self.infos[k]
                evicted += 1
        for k, i in list(self.infos.items()):
            if self.approx_bytes() <= budget:
                break
            if i.kind == "api_result" and k not in self.protect_infos:
                del self.infos[k]
                evicted += 1
        after = self.payload_bytes()
        self.bytes_log.append({"n_steps": len(self.steps), "bytes": after, "evicted": evicted})
        return {"before": before, "after": after, "evicted": evicted}

    # ------------------------------------------------------------------ intents
    def add_intent(self, description: str, parent: str | None = None, iid: str | None = None) -> Intent | None:
        if parent is not None and parent not in self.intents:
            self.log.append({"drop": "intent_parent_missing", "parent": parent, "desc": description})
            return None
        if iid is None or iid in self.intents or not re.fullmatch(r"c\d+", iid):
            self._n_intent += 1
            iid = f"c{self._n_intent}"
        else:
            self._n_intent = max(self._n_intent, int(iid[1:]))
        it = Intent(id=iid, description=description.strip()[:300], parent=parent, order=len(self.intents))
        self.intents[iid] = it
        if parent:
            self.intents[parent].children.append(iid)
        return it

    def set_status(self, iid: str, status: str, evidence: list[str], note: str | None = None) -> bool:
        """Grounded status update. ``done`` and ``blocked`` need step evidence."""
        it = self.intents.get(iid)
        if it is None or status not in STATUSES:
            self.log.append({"drop": "bad_status", "intent": iid, "status": status})
            return False
        ev = [e for e in evidence if e in self.steps]
        if status in ("done", "blocked", "invalidated") and not ev:
            # provenance is deterministic, semantics are proposed: a status change must cite evidence
            self.log.append({"drop": "status_without_citation", "intent": iid, "status": status})
            status = "active" if it.status in ("pending", "active") else it.status
        if status == "done":
            # with a generic adapter (outcome unknown) citation is the only grounding available
            ok = [e for e in ev if self.steps[e]["status"] in ("ok", "unknown")]
            if not ok:
                self.log.append({"drop": "done_without_ok_evidence", "intent": iid})
                status = "active" if it.status in ("pending", "active") else it.status
                ev = ev or it.evidence
        if status == "blocked":
            bad = [e for e in ev if self.steps[e]["status"] == "blocked"]
            if not bad:
                self.log.append({"drop": "blocked_without_error_evidence", "intent": iid})
                status = "active"
        if self.ungrounded_negative(note):
            # "no API exists" style conclusions need a complete API listing as evidence
            self.log.append({"drop": "ungrounded_negative_status", "intent": iid, "note": (note or "")[:80]})
            if status in ("blocked", "invalidated"):
                status = "active"
            note = "guessed API names failed; the API listing seen was incomplete -- look for other APIs"
        it.status = status
        it.evidence = list(dict.fromkeys(it.evidence + ev))
        if note:
            it.note = note[:200]
        return True

    def set_realizes(self, sid: str, iid: str) -> bool:
        if sid in self.steps and iid in self.intents:
            self.step_intent[sid] = iid
            if sid not in self.intents[iid].evidence:
                self.intents[iid].evidence.append(sid)
            return True
        self.log.append({"drop": "bad_realizes", "step": sid, "intent": iid})
        return False

    def set_needs(self, iid: str, info_ids: list[str]) -> None:
        it = self.intents.get(iid)
        if it is None:
            return
        for i in info_ids:
            if i in self.infos and i not in it.needs:
                it.needs.append(i)
                self.infos[i].needed_by.append(iid)

    def augment_needs(self, recent_ids: list[str]) -> int:
        """Deterministic NEEDS: every open intent is assumed to need the current
        credentials/tokens, everything consumed by the recent steps, and every
        variable consumed by two or more distinct steps so far (reused state)."""
        open_intents = [it for it in self.frontier()]
        if not open_intents:
            return 0
        cand = []
        recent = set(recent_ids)
        for i in self.infos.values():
            if i.kind != "runtime_reference" or i.superseded:
                continue
            n = i.name.lower()
            if re.search(r"token|password|login|session|auth|cred", n):
                cand.append(i.id)
            elif any(c in recent for c in i.consumers):
                cand.append(i.id)
        added = 0
        for it in open_intents:
            before = len(it.needs)
            self.set_needs(it.id, cand)
            added += len(it.needs) - before
        return added

    NEGATIVE = re.compile(r"\b(no|not|cannot|can't|missing|unavailable|does not exist|doesn't exist|impossible|"
                          r"unsupported|lack|absent|none)\b", re.I)

    def complete_listing_seen(self, text: str = "") -> bool:
        """True if the app(s) the text talks about had a complete (untruncated)
        API listing observed. With no app named, any complete per-app listing counts."""
        complete, apps = set(), set()
        for st in self.steps.values():
            for app in (st.get("api_lists") or {}):
                if app == "apps":
                    apps.update(st["api_lists"][app])
                    continue
                apps.add(app)
                if not st.get("api_list_truncated"):
                    complete.add(app)
        named = {a for a in apps if re.search(rf"\b{re.escape(a)}\b", text, re.I)}
        if named:
            return named <= complete
        return bool(complete)

    def ungrounded_negative(self, text: str | None) -> bool:
        return bool(text) and bool(self.NEGATIVE.search(text)) \
            and bool(re.search(r"\bapi|endpoint|function|feature|way to|capabilit", text, re.I)) \
            and not self.complete_listing_seen(text)

    def produced_by_intent(self, it: Intent, limit: int = 4) -> list[Info]:
        """Current (non-superseded) variables defined by the intent's evidence steps, with hints."""
        out = []
        for e in it.evidence:
            st = self.steps.get(e)
            if not st:
                continue
            for iid in st["produces"]:
                i = self.infos.get(iid)
                if i and i.kind == "runtime_reference" and not i.superseded and i.value_hint:
                    out.append(i)
        return out[-limit:]

    def add_fact(self, text: str, kind: str = "fact", steps: list[str] | None = None) -> dict | None:
        cited = [s for s in (steps or []) if s in self.steps]
        if not cited:
            self.log.append({"drop": "fact_without_citation", "text": text[:80]})
            return None
        if self.ungrounded_negative(text):
            self.log.append({"drop": "ungrounded_negative_fact", "text": text[:100]})
            return None
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        for f in self.facts:
            if re.sub(r"\W+", " ", f["text"].lower()).strip() == norm:
                return None
        self._n_fact += 1
        f = {"id": f"f{self._n_fact}", "kind": kind if kind in ("fact", "constraint", "failure", "data") else "fact",
             "text": text.strip()[:300], "steps": [s for s in (steps or []) if s in self.steps]}
        self.facts.append(f)
        return f

    # ------------------------------------------------------------------ derived
    def frontier(self) -> list[Intent]:
        """Unresolved intents not represented by unresolved children."""
        out = []
        for it in sorted(self.intents.values(), key=lambda x: x.order):
            if it.status in ("done", "invalidated"):
                continue
            if it.children and any(self.intents[c].status not in ("done", "invalidated") for c in it.children):
                continue
            out.append(it)
        return out

    def live_infos(self, recent_steps: int = 3) -> list[Info]:
        """Infos something ahead can still use: needed by an open intent, consumed
        recently, or a current (non-superseded) runtime binding."""
        open_intents = {it.id for it in self.intents.values() if it.status not in ("done", "invalidated")}
        last = sorted(self.steps, key=lambda s: int(s[1:]))[-recent_steps:]
        out = []
        for i in self.infos.values():
            if i.kind == "runtime_reference" and i.superseded:
                continue
            if any(n in open_intents for n in i.needed_by) or any(c in last for c in i.consumers):
                out.append(i)
            elif i.kind == "runtime_reference":
                out.append(i)
        return out

    def to_dict(self) -> dict:
        return {"goal": self.goal, "steps": self.steps, "infos": {k: asdict(v) for k, v in self.infos.items()},
                "intents": {k: asdict(v) for k, v in self.intents.items()}, "facts": self.facts,
                "step_intent": self.step_intent, "legacy_summary": self.legacy_summary,
                "calls_ok": self.calls_ok, "calls_failed": self.calls_failed, "mem_keys": self.mem_keys,
                "narrative": self.narrative, "call_prefix": self.call_prefix,
                "requirements": self.requirements, "req_graph": self.req_graph, "obs_keep": self.obs_keep,
                "graph_budget": self.graph_budget, "bytes_log": self.bytes_log[-50:],
                "counters": [self._n_info, self._n_intent, self._n_fact], "log": self.log}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls(d["goal"])
        g.steps = d["steps"]
        g.infos = {k: Info(**v) for k, v in d["infos"].items()}
        g.intents = {k: Intent(**v) for k, v in d["intents"].items()}
        g.facts = d["facts"]
        g.step_intent = d["step_intent"]
        g.legacy_summary = d.get("legacy_summary")
        g.calls_ok = d.get("calls_ok", {})
        g.calls_failed = d.get("calls_failed", {})
        g.mem_keys = d.get("mem_keys", [])
        g.narrative = d.get("narrative")
        g.call_prefix = d.get("call_prefix", "")
        g.requirements = d.get("requirements", [])
        g.req_graph = d.get("req_graph")
        g.obs_keep = d.get("obs_keep", 600)
        g.graph_budget = d.get("graph_budget", 0)
        g.bytes_log = d.get("bytes_log", [])
        g._n_info, g._n_intent, g._n_fact = d["counters"]
        g.log = d.get("log", [])
        return g


# ---------------------------------------------------------------------- helpers
def _assigned_api_calls(code: str) -> dict[str, str]:
    """{var: 'app.api'} for ``var = apis.app.api(...)`` (possibly nested in an expression)."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            src = ast.unparse(node.value)
            m = re.search(r"apis\.(\w+)\.(\w+)\(", src)
            if m:
                out[node.targets[0].id] = f"{m.group(1)}.{m.group(2)}"
    return out


def _derived_from(code: str) -> dict[str, str]:
    """{var: base_var} for ``var = base[...]`` / ``base.get(...)`` / ``base['k']['j']``."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            while isinstance(v, (ast.Subscript, ast.Attribute, ast.Call)):
                v = v.value if not isinstance(v, ast.Call) else (v.func.value if isinstance(v.func, ast.Attribute) else None)
                if v is None:
                    break
            if isinstance(v, ast.Name) and v.id != node.targets[0].id:
                out[node.targets[0].id] = v.id
    return out


def _literal_assignments(code: str) -> dict[str, str]:
    """{var: repr(literal)} for ``var = <constant or short literal container>``."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, ast.Constant) or (isinstance(v, (ast.List, ast.Tuple, ast.Dict, ast.Set))
                                               and len(ast.unparse(v)) <= 120):
                out[node.targets[0].id] = ast.unparse(v)[:120]
    return out


def _value_hint(step: dict, name: str) -> str | None:
    """A short value excerpt when the cell printed this variable (or only this one)."""
    code, obs = step["code"], step["observation"]
    if step["status"] != "ok" or not obs.strip() or obs.strip() == "Execution successful.":
        return None
    p = step["printed"]
    if (name.endswith("token") or name.endswith("login")) and p.get("tokens"):
        j = p.get("json")
        if isinstance(j, dict) and "access_token" in j:
            # a login *response*, not the token itself: say so, or the agent passes the dict as a token
            return "{access_token: " + p["tokens"][0][:10] + "..., token_type} (dict; use " + name + "['access_token'])"
        return "access_token " + p["tokens"][0][:10] + "..."
    printed = re.findall(r"print\(([^)]*)\)", code)
    hits = [pp for pp in printed if re.search(rf"\b{re.escape(name)}\b", pp)]
    if len(printed) > 1 and len(step["defs"]) > 1:
        # several values printed by one cell: the observation cannot be attributed to
        # one variable, and a wrong value is worse than none
        return None
    if hits or (len(step["defs"]) == 1 and printed):
        # small structured results (credential lists, profiles, short id lists) are kept whole:
        # an excerpt of them is exactly what makes the agent re-fetch
        return _excerpt(obs, SMALL_VALUE_CHARS if len(obs) <= SMALL_VALUE_CHARS else 120)
    return None


SMALL_VALUE_CHARS = 900
VALUE_FULL_CHARS = 6000


def _excerpt(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[:n] + "..."


def compact_step_view(s: dict, obs_chars: int = 160) -> str:
    """One-line evidence view of a step for LLM prompts and DONE sections."""
    tag = "OK " if s["status"] == "ok" else f"ERR:{s['error_class']}"
    apis = ",".join(s["api_names"][:4]) or "-"
    defs = ",".join(s["defs"][:6]) or "-"
    com = (" | " + s["comments"][0][:80]) if s["comments"] else ""
    return f"{s['id']} [{tag}] apis={apis} defs={defs}{com} -> {_excerpt(s['observation'], obs_chars)}"


def dumps(g: Graph) -> str:
    return json.dumps(g.to_dict(), ensure_ascii=False)
