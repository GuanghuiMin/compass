"""Requirement engine: decomposition spans, operator validation, coverage, frontier, NEEDED_BY."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass.graph.build import Graph  # noqa: E402
from compass.graph.requirements import ReqGraph  # noqa: E402

INSTR = "Send $20 to each of my 5 coworkers via venmo with a note, 'For Lunch'. Refill venmo balance if you need to."


def make_steps():
    g = Graph(INSTR)
    for k in (1, 2, 3):
        g.add_turn({"step": k, "code": f"print(apis.venmo.create_transaction(amount=20, receiver_email='u{k}@x.com'))",
                    "observation": '{"message": "Sent money.", "transaction_id": %d}' % (8000 + k)})
    g.add_turn({"step": 4, "code": "print(apis.venmo.show_balance())",
                "observation": "Execution failed. Traceback:\nNameError: x"})
    return g


def test_decompose_span_check_and_ops():
    g = make_steps()
    rg = ReqGraph()
    n = rg.decompose(INSTR, [
        {"text": "Send $20 to each of my 5 coworkers via venmo", "expect": 5},
        {"text": "with a note, 'For Lunch'"},
        {"text": "paraphrased requirement that is not a span"},
        {"text": "Refill venmo balance if you need to"},
    ])
    assert n == 3 and rg.nodes["r1"]["expect"] == 5
    assert rg.decompose(INSTR, [{"text": "again"}]) == 0          # one-time only

    st = rg.apply_ops([
        {"op": "UPDATE_STATUS", "id": "r1", "status": "DONE", "evidence": ["s1", "s2", "s3"]},
        {"op": "UPDATE_STATUS", "id": "r2", "status": "DONE", "evidence": ["s99"]},
        {"op": "UPDATE_STATUS", "id": "r3", "status": "DONE", "evidence": ["s4"]},
        {"op": "PROPOSE_NEXT", "id": "r1", "action": "create_transaction for the 2 remaining coworkers"},
        {"op": "DECLARE_NEED", "id": "r1", "need": {"api": "phone.search_contacts", "fields": ["email"]}},
        {"op": "UPDATE_STATUS", "id": "nope", "status": "DONE"},
    ], g.steps)
    assert rg.nodes["r1"]["status"] == "PARTIAL" and rg.nodes["r1"]["count"] == 3
    assert rg.nodes["r2"]["status"] in ("NOT_STARTED", "IN_PROGRESS")
    assert st["applied"] == 2 and st["rejected"] == 4

    st2 = rg.apply_ops([{"op": "UPDATE_STATUS", "id": "r1", "status": "DONE",
                         "evidence": ["s1", "s2", "s3"], "count": 5}], g.steps)
    assert st2["applied"] == 1 and rg.nodes["r1"]["status"] == "DONE"


def test_refine_frontier_and_needed_infos():
    g = make_steps()
    rg = ReqGraph()
    rg.decompose(INSTR, [{"text": "Send $20 to each of my 5 coworkers via venmo"},
                         {"text": "Refill venmo balance if you need to"}])
    st = rg.apply_ops([{"op": "REFINE", "parent": "r1", "ordered": True,
                        "children": [{"text": "find coworker emails"}, {"text": "send the 5 transactions", "expect": 5}]},
                       {"op": "DECLARE_NEED", "id": "r1.1", "need": {"api": "phone.search_contacts", "fields": ["email"]}}],
                      g.steps)
    assert st["applied"] == 2
    front = [n["id"] for n in rg.frontier()]
    assert "r1.1" in front and "r1.2" not in front and "r2" in front       # ordered gating
    rg.apply_ops([{"op": "UPDATE_STATUS", "id": "r1.1", "status": "DONE", "evidence": ["s1"]}], g.steps)
    assert "r1.2" in [n["id"] for n in rg.frontier()]
    assert rg.nodes["r1"]["status"] in ("PARTIAL", "IN_PROGRESS")          # roll-up

    needed = rg.needed_infos(g.infos)
    assert isinstance(needed, dict)
    txt = rg.render()
    assert "[>]" in txt and "r1.2" in txt

    d = rg.to_dict()
    rg2 = ReqGraph.from_dict(d)
    assert [n["id"] for n in rg2.frontier()] == [n["id"] for n in rg.frontier()]


def test_needs_extraction_and_protected_eviction():
    from compass.graph.project import extract_fields
    obs = ('[{"contact_id": 1, "name": "Ann", "email": "ann@x.com", "phone": "123", "bio": "' + "b" * 300 + '"},'
           ' {"contact_id": 2, "name": "Bo", "email": "bo@x.com", "phone": "456", "bio": "c"}]')
    got = extract_fields(obs, ["email"])
    assert any(p == "[0].email" for p, _ in got) and any(p == "[1].email" for p, _ in got)

    g = Graph("t")
    g.obs_keep, g.graph_budget = 100, 7000
    g.extract_specs = ["email"]
    big = "[" + ", ".join('{"contact_id": %d, "email": "u%d@x.com", "pad": "%s"}' % (i, i, "p" * 40) for i in range(60)) + "]"
    s1 = g.add_turn({"step": 1, "code": "print(apis.phone.search_contacts())", "observation": big})
    extracted = [g.infos[i] for i in s1["produces"] if "email" in (g.infos[i].value_hint or "")]
    assert extracted and len(s1["observation"]) <= 104
    g.protect_infos = {extracted[0].id}
    g.protect_steps = {"s1"}
    for k in range(2, 9):
        g.add_turn({"step": k, "code": f"r{k} = apis.a.b()\nprint(r{k})", "observation": big})
    g.enforce_budget()
    assert g.steps["s1"]["observation"] != "" and extracted[0].id in g.infos


def test_strip_requirements_keeps_the_rest_of_the_note():
    from compass.graph.compressor import strip_requirements
    note = ('## Task requirements (quoted from the instruction) and verified status\n'
            '- r1: "send $20" -- PARTIAL (3 of 5) [s1]\n\n'
            '## Handled so far\n- 3 of 5 transactions sent (ids 8216, 8217, 8218)\n\n'
            '## Not yet done / unverified\n- 2 coworkers remain\n\n## Next Steps\n1. continue')
    out = strip_requirements(note)
    assert "Task requirements" not in out and "r1:" not in out
    assert "Handled so far" in out and "8216" in out and "Next Steps" in out
