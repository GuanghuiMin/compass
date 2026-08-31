"""Requirement graph: stable nodes decomposed once from the instruction, updated only through
local operators proposed by the model and validated here.

The model judges semantics; this engine owns structure and constraints:

- ``decompose`` runs once per episode. Every node quotes a verbatim span of the instruction
  (checked), gets a stable id, an optional expected count (multi-object coverage) and an
  optional ordered flag for its sibling group.
- Later boundaries apply only local ops: ``REFINE`` (split an unfinished node into children),
  ``UPDATE_STATUS`` (with evidence ids that must exist and have succeeded; DONE on a node with
  ``expect`` further requires covered count >= expect), ``PROPOSE_NEXT`` (the node's next
  concrete action), ``DECLARE_NEED`` (an information spec the node still requires).
- The executable frontier F_k = unfinished leaves whose ordered predecessors are DONE.
  ``needed_infos`` maps the frontier's declared needs onto information nodes (NEEDED_BY),
  which drives rendering priority and protects those nodes from budget eviction.
"""
from __future__ import annotations

import re

STATUSES = ("NOT_STARTED", "IN_PROGRESS", "PARTIAL", "DONE", "BLOCKED")
_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", (t or "").strip().lower())


class ReqGraph:
    """The requirement layer of the compressor's private graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.order: list[str] = []          # document order of top-level nodes
        self.log: list[dict] = []

    # ------------------------------------------------------------------ construction
    def decompose(self, instruction: str, clauses: list[dict]) -> int:
        """One-time decomposition. Each clause: {text, expect?, ordered?}. A clause whose text is
        not a (whitespace-normalized) substring of the instruction is dropped; if nothing
        survives, the whole instruction becomes r1."""
        if self.nodes:
            self.log.append({"drop": "decompose_repeated"})
            return 0
        hay = _norm(instruction)
        kept = 0
        for c in clauses:
            text = (c.get("text") or "").strip().strip('"')
            if not text or _norm(text) not in hay:
                self.log.append({"drop": "span_not_in_instruction", "text": text[:80]})
                continue
            rid = f"r{kept + 1}"
            self.nodes[rid] = {"id": rid, "text": text, "status": "NOT_STARTED",
                               "expect": int(c["expect"]) if c.get("expect") else None,
                               "count": 0, "ordered": bool(c.get("ordered")), "parent": None,
                               "children": [], "evidence": [], "needs": [], "next_action": None}
            self.order.append(rid)
            kept += 1
        if not kept:
            self.nodes["r1"] = {"id": "r1", "text": instruction.strip()[:300], "status": "NOT_STARTED",
                                "expect": None, "count": 0, "ordered": False, "parent": None,
                                "children": [], "evidence": [], "needs": [], "next_action": None}
            self.order.append("r1")
            kept = 1
        return kept

    # ------------------------------------------------------------------ local operators
    def apply_ops(self, ops: list[dict], steps: dict) -> dict:
        """Validate and apply model-proposed local updates. ``steps`` is the evidence layer
        (g.steps). Returns per-op acceptance stats; every rejection is logged with a reason."""
        stats = {"applied": 0, "rejected": 0}
        for op in ops or []:
            kind = (op.get("op") or "").upper()
            ok, why = False, "unknown_op"
            if kind == "REFINE":
                ok, why = self._refine(op)
            elif kind == "UPDATE_STATUS":
                ok, why = self._update_status(op, steps)
            elif kind == "PROPOSE_NEXT":
                ok, why = self._propose_next(op)
            elif kind == "DECLARE_NEED":
                ok, why = self._declare_need(op)
            if ok:
                stats["applied"] += 1
            else:
                stats["rejected"] += 1
                self.log.append({"drop": why, "op": {k: str(v)[:80] for k, v in op.items()}})
        self._roll_up()
        return stats

    def _get(self, rid) -> dict | None:
        return self.nodes.get(str(rid or "").strip())

    def _refine(self, op) -> tuple[bool, str]:
        parent = self._get(op.get("parent"))
        if parent is None:
            return False, "refine_unknown_parent"
        if parent["status"] == "DONE":
            return False, "refine_done_parent"
        if parent["children"]:
            return False, "refine_already_refined"
        if parent["parent"]:
            return False, "refine_too_deep"            # at most two levels: rX and rX.Y
        children = op.get("children") or []
        if not (1 < len(children) <= 4):
            return False, "refine_bad_children"       # coarse sub-goals; per-entity lists go into expect counts
        for i, c in enumerate(children, 1):
            text = ((c.get("text") if isinstance(c, dict) else str(c)) or "").strip()
            if not text:
                return False, "refine_empty_child"
            rid = f"{parent['id']}.{i}"
            self.nodes[rid] = {"id": rid, "text": text[:200], "status": "NOT_STARTED",
                               "expect": int(c["expect"]) if isinstance(c, dict) and c.get("expect") else None,
                               "count": 0, "ordered": bool(op.get("ordered")), "parent": parent["id"],
                               "children": [], "evidence": [], "needs": [], "next_action": None}
            parent["children"].append(rid)
        return True, ""

    def _update_status(self, op, steps) -> tuple[bool, str]:
        node = self._get(op.get("id"))
        if node is None:
            return False, "status_unknown_id"
        if node["children"]:
            return False, "status_on_refined_parent"      # parents roll up from children
        status = str(op.get("status") or "").upper().replace(" ", "_")
        if status not in STATUSES:
            return False, "status_invalid"
        ev = [str(e) for e in (op.get("evidence") or [])]
        if status in ("DONE", "PARTIAL", "IN_PROGRESS"):
            good = [e for e in ev if e in steps and steps[e].get("status") == "ok"]
            if status in ("DONE", "PARTIAL") and not good:
                if node["status"] == "NOT_STARTED":
                    node["status"] = "IN_PROGRESS"
                return False, "status_no_valid_evidence"
            node["evidence"] = sorted(set(node["evidence"]) | set(good))
            count = op.get("count")
            node["count"] = max(node["count"], int(count) if count else len(node["evidence"]))
            if status == "DONE" and node["expect"] and node["count"] < node["expect"]:
                node["status"] = "PARTIAL"                 # coverage: DONE needs count >= expect
                return False, "done_coverage_short"
        node["status"] = status
        return True, ""

    def _propose_next(self, op) -> tuple[bool, str]:
        node = self._get(op.get("id"))
        if node is None:
            return False, "next_unknown_id"
        if node["status"] == "DONE":
            return False, "next_on_done"
        action = (op.get("action") or "").strip()
        if not action:
            return False, "next_empty"
        node["next_action"] = action[:200]
        return True, ""

    def _declare_need(self, op) -> tuple[bool, str]:
        node = self._get(op.get("id"))
        if node is None:
            return False, "need_unknown_id"
        if node["status"] == "DONE":
            return False, "need_on_done"
        need = op.get("need") or {}
        if not isinstance(need, dict) or not (need.get("api") or need.get("fields") or need.get("desc")):
            return False, "need_empty"
        spec = {"api": str(need.get("api") or "")[:80],
                "fields": [str(f)[:40] for f in (need.get("fields") or [])][:8],
                "desc": str(need.get("desc") or "")[:120]}
        if spec not in node["needs"]:
            node["needs"] = (node["needs"] + [spec])[-6:]
        return True, ""

    def _roll_up(self) -> None:
        """Parent status is derived from children, never set directly."""
        for rid in sorted(self.nodes, key=len, reverse=True):
            node = self.nodes[rid]
            if not node["children"]:
                continue
            ch = [self.nodes[c] for c in node["children"]]
            if all(c["status"] == "DONE" for c in ch):
                node["status"] = "DONE"
            elif any(c["status"] == "BLOCKED" for c in ch):
                node["status"] = "BLOCKED"
            elif any(c["status"] != "NOT_STARTED" for c in ch):
                node["status"] = "PARTIAL" if any(c["status"] == "DONE" for c in ch) else "IN_PROGRESS"

    # ------------------------------------------------------------------ frontier and needs
    def _sort_key(self, rid: str):
        return [int(x) for x in rid[1:].split(".")]

    def _siblings(self, node) -> list[dict]:
        if node["parent"]:
            return [self.nodes[c] for c in self.nodes[node["parent"]]["children"]]
        return [self.nodes[r] for r in self.order]

    def frontier(self) -> list[dict]:
        """Unfinished leaves whose ordered predecessors are DONE."""
        out = []
        for rid in sorted(self.nodes, key=self._sort_key):
            node = self.nodes[rid]
            if node["children"] or node["status"] == "DONE":
                continue
            if node["ordered"]:
                sibs = self._siblings(node)
                idx = sibs.index(node)
                if any(s["status"] != "DONE" for s in sibs[:idx]):
                    continue
            out.append(node)
        return out

    def ancestors(self, node) -> list[dict]:
        out = []
        while node.get("parent"):
            node = self.nodes[node["parent"]]
            out.append(node)
        return out

    def frontier_needs(self) -> list[dict]:
        seen, out = set(), []
        for node in self.frontier():
            for spec in node["needs"]:
                key = (spec["api"], tuple(spec["fields"]))
                if key not in seen:
                    seen.add(key)
                    out.append({**spec, "req": node["id"]})
        return out

    def needed_infos(self, infos: dict) -> dict[str, list[str]]:
        """NEEDED_BY: info id -> requirement ids, by api-name and field-token match against the
        frontier's declared needs."""
        needs = self.frontier_needs()
        out: dict[str, list[str]] = {}
        if not needs:
            return out
        for iid, info in infos.items():
            blob = _norm(f"{getattr(info, 'name', '')} {getattr(info, 'source_api', '') or ''} "
                         f"{getattr(info, 'value_hint', '') or ''}")
            for spec in needs:
                hit = (spec["api"] and _norm(spec["api"]).split("(")[0] in blob) or \
                      any(_norm(f) in blob for f in spec["fields"] if f)
                if hit:
                    out.setdefault(iid, []).append(spec["req"])
        return out

    # ------------------------------------------------------------------ rendering / io
    def render(self, max_nodes: int = 24) -> str:
        if not self.nodes:
            return ""
        front = {n["id"] for n in self.frontier()}
        # Lower-bound semantics: a status only ever states what evidence already confirmed at the
        # last compaction. Everything else is rendered as open rather than as "not started", so a
        # stale line can never contradict work the agent did after that boundary.
        lines = ["TASK REQUIREMENTS (confirmed progress as of the last compaction -- a LOWER BOUND; "
                 "anything you did since is simply not recorded here yet; [>] = still open, do next)"]
        shown = 0

        def emit(rid: str, depth: int) -> None:
            nonlocal shown
            if shown >= max_nodes:
                return
            n = self.nodes[rid]
            done = n["status"] == "DONE"
            mark = "[x]" if done else ("[!]" if n["status"] == "BLOCKED" else ("[>]" if rid in front else "[ ]"))
            state = "confirmed done" if done else ("blocked" if n["status"] == "BLOCKED" else "open")
            cov = f" (at least {n['count']} of {n['expect']} done)" if n["expect"] and n["count"] else (
                f" (0 of {n['expect']} confirmed)" if n["expect"] else "")
            ev = f" [{', '.join(n['evidence'][-4:])}]" if n["evidence"] else ""
            lines.append(f"{'  ' * depth}{mark} {rid}: \"{n['text']}\" -- {state}{cov}{ev}")
            if rid in front and n["next_action"]:
                lines.append(f"{'  ' * depth}    next: {n['next_action']}")
            if rid in front and n["needs"]:
                for spec in n["needs"][-2:]:
                    what = spec["api"] or ", ".join(spec["fields"]) or spec["desc"]
                    extra = f" ({spec['desc']})" if spec["desc"] and what != spec["desc"] else ""
                    lines.append(f"{'  ' * depth}    needs: {what}{extra}")
            shown += 1
            for c in self.nodes[rid]["children"]:
                emit(c, depth + 1)

        for rid in self.order:
            emit(rid, 0)
        short = [n for n in self.nodes.values()
                 if n["expect"] and n["count"] < n["expect"] and not n["children"]]
        if short:
            gaps = "; ".join(f"{n['id']} at {n['count']} of {n['expect']}" for n in short[:4])
            lines.append(f"COMPLETION CHECK: {gaps}. The counts come from executed calls, so before "
                         f"completing the task either finish the remaining items or re-read the "
                         f"current state to confirm they are already done.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "order": self.order, "log": self.log[-30:]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReqGraph":
        rg = cls()
        if d:
            rg.nodes = d.get("nodes", {})
            rg.order = d.get("order", [])
            rg.log = d.get("log", [])
        return rg
