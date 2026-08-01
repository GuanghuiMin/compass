"""The whole boundary: what is sent, what comes back, and what survives either outcome.

Every model here is a stub. No test in this file makes a real call, and the graphs in the fixtures are
written to exercise the pipeline, not to resemble any episode.
"""

import json
import os

import pytest

from future_graph import (
    ArtifactError, ComputationNode, GRAMMAR, InformationKind, InformationNode, ModelCall,
    PromptError, Relation, RegenerationRecord, StateGraph, build, load_prompt, parse,
    regenerate_graph, to_protocol,
)
from future_graph.rendering import render
from future_graph.regeneration import PROMPT_PATH, build_user_message


class Stub:
    """A model that answers with fixed text and counts how often it was asked."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[ModelCall] = []

    def __call__(self, call: ModelCall):
        self.calls.append(call)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


VALID = """\
BEGIN_GRAPH

INFO i1
kind: fact
available: true
description: The identifier the work needs
END_INFO

COMPUTATION c1
description: Carry out the remaining work
END_COMPUTATION

EDGE i1 REQUIRES c1

END_GRAPH
"""

UNPARSEABLE = "here is your graph:\nINFO without a beginning\n"

INVALID = """\
BEGIN_GRAPH

COMPUTATION c1
description: One
END_COMPUTATION

COMPUTATION c2
description: Two
END_COMPUTATION

EDGE c1 PRECEDES c2
EDGE c2 PRECEDES c1

END_GRAPH
"""

WITH_DEAD_INFORMATION = """\
BEGIN_GRAPH

INFO i1
kind: fact
available: true
description: Needed by the work
END_INFO

INFO i2
kind: fact
available: true
description: Needed by nobody
END_INFO

COMPUTATION c1
description: Carry out the remaining work
END_COMPUTATION

EDGE i1 REQUIRES c1

END_GRAPH
"""


def previous_graph():
    return build(
        nodes=[ComputationNode(id="c1", description="The work as it stood before"),
               InformationNode(id="i1", kind=InformationKind.FACT, description="Something known",
                               available=True)],
        edges=[("i1", Relation.REQUIRES, "c1")])


def run(answer, previous=None, **kw):
    stub = Stub(answer)
    previous = previous_graph() if previous is None else previous
    result = regenerate_graph("the goal", "the rules", previous, "the slice", stub, **kw)
    return stub, result


# --------------------------------------------------------------------------- outcomes

def test_a_valid_graph_becomes_the_state():
    stub, result = run(VALID)
    assert result.record.accepted
    assert [c.id for c in result.graph.computations] == ["c1"]
    assert result.record.violations == () and result.record.parse_errors == ()


def test_the_returned_graph_is_the_one_the_record_describes():
    for answer in (VALID, UNPARSEABLE, INVALID, WITH_DEAD_INFORMATION):
        _, result = run(answer)
        assert result.graph == StateGraph.from_snapshot(result.record.resulting_snapshot)
        assert result.record.handover == render(result.graph)


def test_an_unparseable_answer_leaves_the_previous_graph_untouched():
    previous = previous_graph()
    before = previous.to_snapshot()
    _, result = run(UNPARSEABLE, previous)
    assert not result.record.accepted
    assert result.record.resulting_snapshot == before
    assert previous.to_snapshot() == before
    assert result.record.parse_errors and result.record.violations == ()
    assert result.record.parsed_candidate_snapshot is None
    assert result.record.collected == ()


def test_an_invalid_graph_leaves_the_previous_graph_untouched():
    previous = previous_graph()
    before = previous.to_snapshot()
    _, result = run(INVALID, previous)
    assert not result.record.accepted
    assert result.record.resulting_snapshot == before
    assert [code for code, _, _ in result.record.violations] == ["cycle"]
    assert result.record.parse_errors == ()
    assert result.record.parsed_candidate_snapshot is not None
    assert result.record.collected == ()


def test_what_the_model_wrote_survives_even_though_collection_removed_part_of_it():
    """The candidate is snapshotted before replacement, which collects in place."""
    _, result = run(WITH_DEAD_INFORMATION)
    candidate = result.record.parsed_candidate_snapshot
    assert [i["id"] for i in candidate["information"]] == ["i1", "i2"]
    assert result.record.collected == ("i2",)
    assert [i["id"] for i in result.record.resulting_snapshot["information"]] == ["i1"]
    assert result.record.accepted


def test_the_first_boundary_is_the_same_call_with_an_empty_graph():
    stub, result = run(VALID, StateGraph())
    assert result.record.previous_snapshot == {"computations": [], "information": [], "edges": []}
    assert "BEGIN_GRAPH\n\nEND_GRAPH" in stub.calls[0].user
    assert result.record.accepted and len(result.graph) == 2


# --------------------------------------------------------------------------- the call itself

def test_the_call_is_exactly_the_template_and_the_four_inputs():
    previous = previous_graph()
    stub = Stub(VALID)
    regenerate_graph("the goal", "the rules", previous, "the slice", stub, {"temperature": 0})
    assert stub.calls == [ModelCall(
        system=load_prompt(),
        user=("BEGIN_ORIGINAL_GOAL\nthe goal\nEND_ORIGINAL_GOAL\n"
              "\nBEGIN_FIXED_RULES\nthe rules\nEND_FIXED_RULES\n"
              "\nBEGIN_PREVIOUS_GRAPH\n" + to_protocol(previous) + "END_PREVIOUS_GRAPH\n"
              "\nBEGIN_DELTA_H\nthe slice\nEND_DELTA_H\n"),
        config=(("temperature", 0),),
    )]


def test_each_input_appears_exactly_as_given():
    stub, _ = run(VALID)
    user = stub.calls[0].user
    for value in ("the goal", "the rules", "the slice"):
        assert value in user
    assert to_protocol(previous_graph()) in user


def test_nothing_the_model_should_not_see_appears():
    stub, _ = run(VALID)
    whole = stub.calls[0].system + stub.calls[0].user
    assert render(previous_graph()) not in whole
    for forbidden in ("CURRENT COMPUTATIONS", "LATER COMPUTATIONS", "registry", "history"):
        assert forbidden not in stub.calls[0].user


def test_the_system_message_carries_the_grammar_the_parser_enforces():
    assert GRAMMAR in load_prompt()


def test_the_input_is_not_tidied_on_the_way_in():
    stub = Stub(VALID)
    regenerate_graph("  goal with space  ", "\nrules\n", StateGraph(), "  slice  ", stub)
    assert "BEGIN_ORIGINAL_GOAL\n  goal with space  \nEND" in stub.calls[0].user
    assert "BEGIN_DELTA_H\n  slice  \nEND" in stub.calls[0].user


def test_one_call_per_boundary():
    for answer in (VALID, UNPARSEABLE, INVALID):
        stub, _ = run(answer)
        assert len(stub.calls) == 1


# --------------------------------------------------------------------------- configuration

def test_the_configuration_reaches_the_model_sorted_and_is_recorded_as_sent():
    stub = Stub(VALID)
    result = regenerate_graph("g", "r", StateGraph(), "d", stub,
                              {"top_p": 1.0, "model": "m", "temperature": 0, "stream": False})
    assert stub.calls[0].config == (("model", "m"), ("stream", False), ("temperature", 0),
                                    ("top_p", 1.0))
    assert result.record.model_call == stub.calls[0]


def test_changing_the_caller_s_mapping_afterwards_changes_nothing():
    config = {"temperature": 0}
    stub = Stub(VALID)
    result = regenerate_graph("g", "r", StateGraph(), "d", stub, config)
    config["temperature"] = 1
    config["sneaked"] = "in"
    assert result.record.model_call.config == (("temperature", 0),)


@pytest.mark.parametrize("config", [{7: "x"}, {None: "x"}])
def test_a_config_key_that_is_not_text_is_refused_before_any_call(config):
    stub = Stub(VALID)
    with pytest.raises(TypeError, match="config key is text"):
        regenerate_graph("g", "r", StateGraph(), "d", stub, config)
    assert stub.calls == []


@pytest.mark.parametrize("value", [[1], {"a": 1}, (1,), {1, 2}, object(), print])
def test_a_config_value_that_cannot_be_recorded_is_refused_before_any_call(value):
    stub = Stub(VALID)
    with pytest.raises(TypeError, match="not a value this can record"):
        regenerate_graph("g", "r", StateGraph(), "d", stub, {"a": value})
    assert stub.calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_config_value_is_refused_before_any_call(value):
    stub = Stub(VALID)
    with pytest.raises(ValueError, match="cannot be recorded faithfully"):
        regenerate_graph("g", "r", StateGraph(), "d", stub, {"temperature": value})
    assert stub.calls == []


def test_nothing_in_the_configuration_is_stringified():
    stub = Stub(VALID)
    result = regenerate_graph("g", "r", StateGraph(), "d", stub,
                              {"a": 1, "b": 1.5, "c": True, "d": None, "e": "text"})
    assert dict(result.record.model_call.config) == {"a": 1, "b": 1.5, "c": True, "d": None,
                                                     "e": "text"}


# --------------------------------------------------------------------------- model failures

def test_a_model_that_raises_is_not_a_rejected_graph():
    previous = previous_graph()
    before = previous.to_snapshot()
    stub = Stub(RuntimeError("the service is down"))
    with pytest.raises(RuntimeError, match="the service is down"):
        regenerate_graph("g", "r", previous, "d", stub)
    assert len(stub.calls) == 1
    assert previous.to_snapshot() == before


@pytest.mark.parametrize("answer", [None, 7, b"BEGIN_GRAPH", ["BEGIN_GRAPH"]])
def test_a_model_that_returns_something_other_than_text_is_a_broken_adapter(answer):
    previous = previous_graph()
    before = previous.to_snapshot()
    with pytest.raises(TypeError, match="a model returns text"):
        regenerate_graph("g", "r", previous, "d", Stub(answer))
    assert previous.to_snapshot() == before


# --------------------------------------------------------------------------- the prompt

def test_the_prompt_loads_from_anywhere(tmp_path):
    here = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert GRAMMAR in load_prompt()
    finally:
        os.chdir(here)


def test_a_template_without_the_placeholder_fails_before_a_call(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("no placeholder here", encoding="utf-8")
    with pytest.raises(PromptError, match="0 times"):
        load_prompt(path)


def test_a_template_with_two_placeholders_fails_before_a_call(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("{{PROTOCOL_GRAMMAR}} and {{PROTOCOL_GRAMMAR}}", encoding="utf-8")
    with pytest.raises(PromptError, match="2 times"):
        load_prompt(path)


def test_a_missing_template_fails_before_a_call(tmp_path):
    with pytest.raises(PromptError, match="could not be read"):
        load_prompt(tmp_path / "absent.md")


def test_the_example_in_the_prompt_parses_and_is_a_graph_that_would_be_accepted():
    example = PROMPT_PATH.read_text(encoding="utf-8").rsplit("BEGIN_GRAPH", 1)[1]
    text = "BEGIN_GRAPH" + example.split("END_GRAPH")[0] + "END_GRAPH\n"
    outcome = parse(text)
    assert outcome.errors == (), outcome.errors
    from future_graph.validation import validate
    assert validate(outcome.graph) == ()


def test_the_example_never_states_an_ordering_the_information_flow_already_states():
    """A redundant PRECEDES in the example is a redundant PRECEDES in every graph after it."""
    example = PROMPT_PATH.read_text(encoding="utf-8").rsplit("BEGIN_GRAPH", 1)[1]
    graph = parse("BEGIN_GRAPH" + example.split("END_GRAPH")[0] + "END_GRAPH\n").graph
    implied = {(producer, consumer)
               for information in graph.information
               for producer in graph.producers_of(information.id)
               for consumer in graph.consumers_of(information.id)}
    stated = {(e.source, e.target) for e in graph.edges_of(Relation.PRECEDES)}
    assert implied & stated == set()
    assert stated, "the example should still show what PRECEDES is for"


# --------------------------------------------------------------------------- the record

def test_a_record_round_trips():
    _, result = run(WITH_DEAD_INFORMATION)
    again = RegenerationRecord.from_dict(json.loads(json.dumps(result.record.to_dict())))
    assert again == result.record


def test_a_rejected_record_round_trips():
    _, result = run(INVALID)
    again = RegenerationRecord.from_dict(json.loads(json.dumps(result.record.to_dict())))
    assert again == result.record


def test_the_hash_is_of_the_system_message_as_sent():
    from future_graph import prompt_sha
    _, result = run(VALID)
    assert result.record.prompt_sha == prompt_sha(result.record.model_call.system)
    assert result.record.prompt_sha != prompt_sha(PROMPT_PATH.read_text(encoding="utf-8"))


def _record_dict():
    _, result = run(VALID)
    return json.loads(json.dumps(result.record.to_dict()))


def test_a_record_missing_a_field_is_rejected():
    raw = _record_dict()
    del raw["handover"]
    with pytest.raises(ArtifactError, match="missing handover"):
        RegenerationRecord.from_dict(raw)


def test_a_record_with_an_unknown_field_is_rejected():
    raw = _record_dict()
    raw["notes"] = "anything"
    with pytest.raises(ArtifactError, match="unknown notes"):
        RegenerationRecord.from_dict(raw)


def test_accepted_as_a_string_is_rejected_rather_than_coerced():
    raw = _record_dict()
    raw["accepted"] = "false"
    with pytest.raises(ArtifactError, match="accepted"):
        RegenerationRecord.from_dict(raw)


def test_a_hash_that_does_not_match_its_system_message_is_rejected():
    raw = _record_dict()
    raw["model_call"]["system"] = raw["model_call"]["system"] + " altered"
    with pytest.raises(ArtifactError, match="prompt_sha"):
        RegenerationRecord.from_dict(raw)


@pytest.mark.parametrize("config", [[["a", [1]]], [["a"]], [[7, "b"]], "not a list"])
def test_a_malformed_config_entry_is_rejected(config):
    raw = _record_dict()
    raw["model_call"]["config"] = config
    with pytest.raises(ArtifactError):
        RegenerationRecord.from_dict(raw)


def test_a_malformed_snapshot_is_caught_when_the_graph_is_rebuilt():
    from future_graph import SchemaError
    raw = _record_dict()
    raw["resulting_snapshot"] = {"computations": [], "information": []}
    record = RegenerationRecord.from_dict(raw)
    with pytest.raises(SchemaError, match="missing edges"):
        StateGraph.from_snapshot(record.resulting_snapshot)


def test_a_malformed_parse_error_entry_is_rejected():
    raw = _record_dict()
    raw["parse_errors"] = [["line one", "message"]]
    with pytest.raises(ArtifactError, match="line and a message"):
        RegenerationRecord.from_dict(raw)


def test_a_malformed_violation_entry_is_rejected():
    raw = _record_dict()
    raw["violations"] = [["cycle", "a message"]]
    with pytest.raises(ArtifactError, match="code, a message"):
        RegenerationRecord.from_dict(raw)


def test_the_user_message_builder_is_the_only_layout():
    assert build_user_message("g", "r", StateGraph(), "d") == (
        "BEGIN_ORIGINAL_GOAL\ng\nEND_ORIGINAL_GOAL\n"
        "\nBEGIN_FIXED_RULES\nr\nEND_FIXED_RULES\n"
        "\nBEGIN_PREVIOUS_GRAPH\nBEGIN_GRAPH\n\nEND_GRAPH\nEND_PREVIOUS_GRAPH\n"
        "\nBEGIN_DELTA_H\nd\nEND_DELTA_H\n")
