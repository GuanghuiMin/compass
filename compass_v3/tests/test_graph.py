"""Deterministic-layer tests: parsing, dataflow, grounding, rendering budget."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass.eval.signatures import canon, is_error, sigs  # noqa: E402
from compass.graph.build import Graph  # noqa: E402
from compass.graph.parse import defs_uses, error_class  # noqa: E402
from compass.graph.render import render_to_budget  # noqa: E402
from compass.harness.prompt import count_tokens  # noqa: E402

ERR = "Execution failed. Traceback:\n  File \"<python-input>\", line 1\nNameError: name 'tok' is not defined"


def test_sigs_canonical():
    assert sigs("apis.venmo.add_friend(user_email=e, access_token=t)") == ["venmo.add_friend(access_token=t,user_email=e)"]
    assert canon("b=2, a=1") == "a=1,b=2"
    assert is_error(ERR) and not is_error("[]")


def test_defs_uses_and_loop_vars():
    code = ("tok = apis.venmo.login(username=u, password=p)['access_token']\n"
            "for f in friends:\n    print(f, tok)\n"
            "def helper(x):\n    return x + tok\n")
    defs, uses, ok = defs_uses(code)
    assert ok
    assert defs == ["tok", "helper"]
    assert "f" not in defs and "f" not in uses
    assert set(uses) == {"u", "p", "friends"}


def test_extract_code_strips_model_markup():
    from compass.harness.episode import extract_code
    assert extract_code("\n\n<｜DSML｜python>\nprint(1)\n</｜DSML｜python>") == "print(1)"
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    # ACON semantics: think *tags* are removed, their content is kept
    assert extract_code("<think>hmm</think>\nprint(2)") == "hmm\nprint(2)"


def test_error_class():
    assert error_class(ERR) == "name_error"
    assert error_class("No code available to execute.") == "no_code"
    assert error_class("[]") is None


def test_graph_dataflow_and_supersede():
    g = Graph("goal")
    g.add_turn({"step": 1, "code": "a = apis.x.y()\nprint(a)", "observation": "[1, 2]"})
    g.add_turn({"step": 2, "code": "b = len(a)", "observation": "Execution successful."})
    g.add_turn({"step": 3, "code": "a = 5", "observation": "Execution successful."})
    a1 = next(i for i in g.infos.values() if i.name == "a" and i.producer == "s1")
    assert a1.source_api == "x.y" and a1.consumers == ["s2"] and a1.superseded
    assert g.steps["s2"]["consumes"] == [a1.id]
    live = {i.name for i in g.live_infos()}
    assert "a" in live and "b" in live


def test_status_grounding():
    g = Graph("goal")
    g.add_turn({"step": 1, "code": "x = apis.a.b()", "observation": ERR})
    g.add_turn({"step": 2, "code": "x = apis.a.b()", "observation": "{\"ok\": true}"})
    it = g.add_intent("do the thing")
    assert g.set_status(it.id, "done", ["s1"]) and it.status == "active"      # no ok evidence -> downgraded
    assert g.set_status(it.id, "done", ["s2"]) and it.status == "done"
    assert g.set_status(it.id, "blocked", ["s2"]) and it.status == "active"   # no error evidence
    assert g.set_status(it.id, "blocked", ["s1"]) and it.status == "blocked"
    assert g.add_intent("child of missing", parent="c99") is None
    assert g.log[-1]["drop"] == "intent_parent_missing"


def test_fact_dedup_and_citation():
    g = Graph("goal")
    g.add_turn({"step": 1, "code": "print(1)", "observation": "1"})
    assert g.add_fact("Friends are contacts with relationship 'friend'.", "constraint") is None   # no citation
    assert g.add_fact("Friends are contacts with relationship 'friend'.", "constraint", ["s1"]) is not None
    assert g.add_fact("friends are contacts with relationship friend", "constraint", ["s1"]) is None  # duplicate
    assert len(g.facts) == 1


def test_flow_render_all_levels():
    from compass.graph.flow import attach_to_frontier, prune_evidence, render_flow
    g = Graph("Update the playlist with the siblings' suggestions")
    for k in range(1, 12):
        g.add_turn({"step": k, "code": f"r{k} = apis.spotify.search_songs(query='q{k}', page_limit=3)\nprint(r{k})",
                    "observation": "[{\"song_id\": %d, \"title\": \"Song %d\"}]" % (k, k)})
    g.add_turn({"step": 12, "code": "tok = apis.spotify.login(username=u, password=p)['access_token']",
                "observation": "Execution successful."})
    a = g.add_intent("Search release dates on Spotify")
    b = g.add_intent("Update the playlist")
    g.add_fact("Brenda asked: add 'Song 1', remove 'Song 2'", "data", ["s1"])
    g.add_fact("Playlist id=654 'Roadtrip'", "data", ["s2"])
    g.augment_needs(["s12"])
    attach_to_frontier(g, ["s12"])
    assert prune_evidence(g, ["s12"]) > 0
    for level in range(4):
        text = render_flow(g, level, ["s12"])
        assert "GOAL" in text and "PLAN" in text
        for line in text.splitlines():
            if "(see " in line:
                assert line.split("(see ")[-1].rstrip(")") in (a.id, b.id)


def test_adapter_tiers():
    turn = {"step": 1, "code": "tok = apis.venmo.login(username=u, password=p)\nprint(tok)",
            "observation": "Execution failed. Traceback:\n  x\nNameError: name 'u' is not defined"}
    g = Graph("goal")
    s = g.add_turn(turn, "generic")
    assert s["status"] == "unknown" and s["call_forms"] == [] and s["defs"] == []
    g2 = Graph("goal")
    s2 = g2.add_turn(turn, "schema")
    assert s2["status"] == "blocked" and s2["call_forms"] == ["venmo.login(username, password)"] and s2["defs"] == []
    g3 = Graph("goal")
    s3 = g3.add_turn(turn, "codeact")
    assert s3["defs"] == ["tok"] and set(s3["uses"]) == {"u", "p"}


def test_render_within_budget():
    g = Graph("Reset friends on venmo")
    big = "[" + ", ".join(f'{{"contact_id": {i}, "email": "u{i}@x.com"}}' for i in range(400)) + "]"
    for k in range(1, 30):
        g.add_turn({"step": k, "code": f"v{k} = apis.phone.search_contacts(page_index={k})\nprint(v{k})",
                    "observation": big})
    root = g.add_intent("sync friends")
    for k in range(1, 10):
        c = g.add_intent(f"sub {k}", parent=root.id)
        g.set_status(c.id, "done", [f"s{k}"])
    text, level = render_to_budget(g, 500, ["s29"])
    assert level in (1, 2, 3)
    if level < 3:
        assert count_tokens(text) <= 500
    assert "GOAL" in text and "PLAN" in text


def test_projection_keeps_fields_not_prefixes():
    from compass.graph.project import focus_tokens, project
    obs = ('[{"song_id": 55, "title": "Tangled Lies", "album_id": 11, "duration": 186, '
           '"artists": [{"id": 3, "name": "Ava Morgan"}], "release_date": "2022-03-12", "lyrics": "' + "la " * 200 + '"}]')
    out = project(obs, 200, focus_tokens("add the release month for each song"))
    assert out.startswith("1 items: [{") and "lyrics" not in out
    assert out.index("release_date=") < out.index("duration=")   # goal-named field before the rest
    assert project('{"a": 1, "b": [1, 2]}', 100) == "{a=1, b=[1, 2]}"
    assert project("plain text " * 30, 40).endswith("...")


def test_proj_render_and_mem_setup():
    from compass.graph.compressor import mem_setup_code
    from compass.graph.render import render
    g = Graph("Add release month for every liked song")
    g.obs_keep = 0     # externalization / projection variants keep the raw observation
    obs = '[{"song_id": 55, "title": "Tangled Lies", "release_date": "2022-03-12", "duration": 186}]'
    g.add_turn({"step": 1, "code": "print(apis.spotify.search_songs(query='Tangled Lies'))", "observation": obs})
    g.add_turn({"step": 2, "code": "r = apis.spotify.search_songs(query='Other')\nprint(r)", "observation": obs})
    proj = render(g, 0, ["s2"], proj=True)
    assert proj.count("release_date=") >= 2 and "_mem[" not in proj
    g.mem_keys.append("s1")
    assert "_mem['s1'] (list of 1 dicts with keys song_id, title, release_date, duration) " in render(g, 0, ["s2"], proj=True)
    g.call_prefix = "apis."
    assert "- apis.spotify.search_songs(" in render(g, 0, ["s2"])
    code = mem_setup_code({"s1": obs})
    ns: dict = {}
    exec(code, ns)
    assert ns["_mem"]["s1"][0]["song_id"] == 55      # parsed, indexable like the API result
    exec(mem_setup_code({"s2": "x"}), ns)
    assert set(ns["_mem"]) == {"s1", "s2"}


def test_bounded_private_graph_and_requirements():
    from compass.graph.compressor import parse_requirements, splice_requirements
    g = Graph("Send $20 to each coworker")
    g.obs_keep, g.graph_budget = 100, 7000
    big = "[" + ", ".join('{"id": %d, "name": "user%d"}' % (i, i) for i in range(200)) + "]"
    for k in range(1, 8):
        g.add_turn({"step": k, "code": f"r{k} = apis.phone.search_contacts(page_index={k})\nprint(r{k})", "observation": big})
    assert all(len(s["observation"]) <= 104 for s in g.steps.values())   # raw observation gone after Apply
    assert all(i.value_full is None for i in g.infos.values())
    st = g.enforce_budget()
    assert st["after"] <= 7000 and st["evicted"] > 0 and g.bytes_log[-1]["bytes"] == st["after"]
    note = ('## Task requirements\n- "send $20 to each coworker" -- PARTIAL (2 of 5) [s1, s2]\n'
            '- "add a note" -- DONE\n- "refill balance" -- DONE [s99]\n## Next Steps\n1. continue')
    reqs = parse_requirements(note, g)
    assert [r["status"] for r in reqs] == ["PARTIAL", "NOT DONE", "NOT DONE"]
    assert reqs[0]["supports"] == ["s1", "s2"] and reqs[2]["supports"] == []
    out = splice_requirements(note, reqs)
    assert "verified status" in out and "- r1:" in out and "## Next Steps" in out
