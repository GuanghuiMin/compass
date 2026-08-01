"""The repairability probe, on a fixture that mimics a finished run. No real call anywhere.

The fixture reproduces the shape of the source run rather than reading it, so these pass on a machine
that never ran the preflight, and a change to the real artifacts cannot quietly change what is tested.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from future_graph.artifacts import ModelCall, prompt_sha
from future_graph.regeneration import load_prompt
from future_graph.run import FROZEN_CONFIG


def load_probe():
    path = Path(__file__).resolve().parents[1] / "scripts" / "repairability_probe.py"
    spec = importlib.util.spec_from_file_location("repairability_probe", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because a dataclass in the module resolves its annotations
    # through sys.modules while the decorator runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = load_probe()

VALID = "BEGIN_GRAPH\n\nCOMPUTATION c1\ndescription: Do the work\nEND_COMPUTATION\n\nEND_GRAPH\n"
CYCLE = ("BEGIN_GRAPH\nCOMPUTATION c1\ndescription: One\nEND_COMPUTATION\n"
         "COMPUTATION c2\ndescription: Two\nEND_COMPUTATION\n"
         "EDGE c1 PRECEDES c2\nEDGE c2 PRECEDES c1\nEND_GRAPH\n")
GARBAGE = "here is the graph you asked for\n"


class Stub:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[ModelCall] = []

    def __call__(self, call):
        self.calls.append(call)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def record(system, accepted, raw, parse_errors=(), violations=()):
    return {
        "goal": "g", "rules": "r", "delta_h": "d",
        "previous_snapshot": {"computations": [], "information": [], "edges": []},
        "model_call": {"system": system, "user": "USER TEXT",
                       "config": [[k, v] for k, v in sorted(FROZEN_CONFIG.items())]},
        "prompt_sha": prompt_sha(system), "raw_output": raw,
        "normalizations": [], "parse_errors": [list(e) for e in parse_errors],
        "parsed_candidate_snapshot": None, "violations": [list(v) for v in violations],
        "accepted": accepted,
        "resulting_snapshot": {"computations": [], "information": [], "edges": []},
        "collected": [], "handover": "NOTHING REMAINS",
    }


def build_source(tmp_path, monkeypatch, parse_keys=None, validation_keys=None):
    """A run directory shaped like the real one, with the same rejection set by default."""
    parse_keys = P.EXPECTED_PARSE if parse_keys is None else parse_keys
    validation_keys = P.EXPECTED_VALIDATION if validation_keys is None else validation_keys
    system = load_prompt()
    monkeypatch.setattr(P, "SOURCE_PROMPT_SHA", prompt_sha(system))

    episodes = sorted({e for e, _ in set(parse_keys) | set(validation_keys)})
    run = tmp_path / "source"
    run.mkdir()
    for episode in episodes:
        (run / f"episode_{episode}").mkdir()
    for key in sorted(set(parse_keys) | set(validation_keys)):
        episode, index = key
        empty = key in {("6b6ca61_2", 4), ("83a7951_2", 1), ("83a7951_2", 9)}
        if key in set(parse_keys):
            body = record(system, False, "" if empty else GARBAGE,
                          parse_errors=[[1, "no BEGIN_GRAPH"]])
        else:
            body = record(system, False, CYCLE, violations=[["cycle", "a cycle", []]])
        (run / f"episode_{episode}" / f"boundary_{index:03d}.json").write_text(
            json.dumps(body), encoding="utf-8")
    # one accepted boundary, which discovery must ignore
    (run / f"episode_{episodes[0]}" / "boundary_099.json").write_text(
        json.dumps(record(system, True, VALID)), encoding="utf-8")

    manifest = {"episode_order": episodes, "status": "completed", "commit": P.SOURCE_COMMIT,
                "input_manifest_sha256": P.SOURCE_INPUT_MANIFEST,
                "prompt_sha": prompt_sha(system), "model": "minimax-m3",
                "config": dict(FROZEN_CONFIG)}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


# --------------------------------------------------------------------------- discovery

def test_discovery_finds_the_refused_boundaries_and_ignores_the_accepted_one(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    found = P.discover(run)
    assert len(found) == 13
    assert {r.key for r in found if r.cohort == "parse"} == set(P.EXPECTED_PARSE)
    assert {r.key for r in found if r.cohort == "validation"} == set(P.EXPECTED_VALIDATION)


def test_the_three_empty_outputs_are_marked(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    empty = {r.key for r in P.discover(run) if r.empty_output}
    assert empty == {("6b6ca61_2", 4), ("83a7951_2", 1), ("83a7951_2", 9)}


# --------------------------------------------------------------------------- the retry call

def test_the_stored_system_message_is_reused_byte_for_byte(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    for r in P.discover(run):
        assert P.retry_call(r).system == r.record["model_call"]["system"]


def test_each_cohort_gets_its_own_fixed_suffix_and_nothing_else(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    for r in P.discover(run):
        call = P.retry_call(r)
        expected = P.PARSE_SUFFIX if r.cohort == "parse" else P.VALIDATION_SUFFIX
        assert call.user == r.record["model_call"]["user"] + expected


def test_a_validation_rejection_is_never_told_it_could_not_be_parsed():
    """It parsed. Telling it otherwise would be false, and a probe that lies measures nothing."""
    assert "could not be parsed" in P.PARSE_SUFFIX
    assert "could not be parsed" not in P.VALIDATION_SUFFIX
    assert "not accepted as a valid graph" in P.VALIDATION_SUFFIX
    assert P.PARSE_SUFFIX != P.VALIDATION_SUFFIX


def test_the_configuration_is_carried_over_unchanged(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    call = P.retry_call(P.discover(run)[0])
    assert dict(call.config) == FROZEN_CONFIG


# --------------------------------------------------------------------------- judging

def test_acceptance_is_decided_by_the_unchanged_parser_and_validator():
    assert P.judge(VALID)["accepted_on_retry"] is True
    bad_parse = P.judge(GARBAGE)
    assert bad_parse["accepted_on_retry"] is False and bad_parse["layer"] == "parse"
    bad_semantics = P.judge(CYCLE)
    assert bad_semantics["accepted_on_retry"] is False and bad_semantics["layer"] == "validation"
    assert [v[0] for v in bad_semantics["violations"]] == ["cycle"]


# --------------------------------------------------------------------------- running

def test_one_call_per_boundary_and_a_record_for_each(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    stub = Stub(VALID)
    out = tmp_path / "probe"
    manifest = P.probe(run, out, stub, "deadbeef")
    assert len(stub.calls) == 13
    assert len(manifest["completed"]) == 13 and manifest["status"] == "completed"
    assert len(list(out.glob("boundary_*.json"))) == 13


def test_the_source_run_is_never_written_to(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in run.rglob("*.json")}
    P.probe(run, tmp_path / "probe", Stub(VALID), "deadbeef")
    assert {p: p.read_bytes() for p in run.rglob("*.json")} == before


def test_the_manifest_separates_the_two_commits_and_both_suffixes(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    manifest = P.probe(run, tmp_path / "probe", Stub(VALID), "probe-commit-sha")
    assert manifest["source_run_commit"] == P.SOURCE_COMMIT
    assert manifest["probe_commit"] == "probe-commit-sha"
    assert manifest["parse_suffix_sha256"] == prompt_sha(P.PARSE_SUFFIX)
    assert manifest["validation_suffix_sha256"] == prompt_sha(P.VALIDATION_SUFFIX)
    assert manifest["source_input_manifest_hash"] == P.SOURCE_INPUT_MANIFEST


def test_a_boundary_record_reads_back(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    out = tmp_path / "probe"
    P.probe(run, out, Stub(VALID), "deadbeef")
    body = json.loads((out / "boundary_6b6ca61_2_004.json").read_text())
    assert body["cohort"] == "parse" and body["empty_original_output"] is True
    assert body["suffix"] == P.PARSE_SUFFIX
    assert body["accepted_on_retry"] is True
    assert body["retry_raw_output"] == VALID


# --------------------------------------------------------------------------- refusing

@pytest.mark.parametrize("field,value", [
    ("status", "operational_failure"), ("commit", "0" * 40),
    ("input_manifest_sha256", "0" * 64), ("prompt_sha", "0" * 64), ("model", "gpt-4"),
])
def test_a_source_run_that_is_not_the_expected_one_stops_before_any_call(tmp_path, monkeypatch,
                                                                        field, value):
    run = build_source(tmp_path, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest[field] = value
    (run / "manifest.json").write_text(json.dumps(manifest))
    stub = Stub(VALID)
    with pytest.raises(P.ProbeError):
        P.probe(run, tmp_path / "probe", stub, "deadbeef")
    assert stub.calls == []


def test_a_different_rejection_set_stops_before_any_call(tmp_path, monkeypatch):
    """A probe that silently drops a boundary reports a rate over a denominator nobody chose."""
    fewer = set(P.EXPECTED_PARSE) - {("042a9fc_3", 0)}
    run = build_source(tmp_path, monkeypatch, parse_keys=fewer)
    stub = Stub(VALID)
    with pytest.raises(P.ProbeError, match="parse rejections are"):
        P.probe(run, tmp_path / "probe", stub, "deadbeef")
    assert stub.calls == []


def test_a_stored_system_that_does_not_hash_to_the_run_prompt_stops_before_any_call(
        tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    path = run / "episode_042a9fc_3" / "boundary_000.json"
    body = json.loads(path.read_text())
    body["model_call"]["system"] += " altered"
    path.write_text(json.dumps(body))
    stub = Stub(VALID)
    with pytest.raises(P.ProbeError, match="does not hash"):
        P.probe(run, tmp_path / "probe", stub, "deadbeef")
    assert stub.calls == []


def test_a_stored_configuration_that_differs_stops_before_any_call(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    path = run / "episode_042a9fc_3" / "boundary_001.json"
    body = json.loads(path.read_text())
    body["model_call"]["config"] = [["temperature", 1.0]]
    path.write_text(json.dumps(body))
    stub = Stub(VALID)
    with pytest.raises(P.ProbeError, match="stored config"):
        P.probe(run, tmp_path / "probe", stub, "deadbeef")
    assert stub.calls == []


def test_an_existing_output_directory_is_refused(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    out = tmp_path / "probe"
    out.mkdir()
    with pytest.raises(FileExistsError):
        P.probe(run, out, Stub(VALID), "deadbeef")


def test_a_failure_stops_the_probe_and_keeps_what_was_done(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    out = tmp_path / "probe"
    stub = Stub(VALID, VALID, RuntimeError("the service is down"))
    with pytest.raises(RuntimeError, match="the service is down"):
        P.probe(run, out, stub, "deadbeef")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "operational_failure"
    assert len(manifest["completed"]) == 2
    assert manifest["operational_failure"]["boundary"]
    assert len(list(out.glob("boundary_*.json"))) == 2


def test_a_model_returning_something_other_than_text_stops_the_probe(tmp_path, monkeypatch):
    run = build_source(tmp_path, monkeypatch)
    with pytest.raises(P.ProbeError, match="a model returns text"):
        P.probe(run, tmp_path / "probe", Stub(7), "deadbeef")


def test_the_command_requires_one_path_component(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        P.main(["--run-id", "a/b"])
