"""The whole boundary: what is sent, what comes back, and what survives either outcome.

Every model here is a stub. No test in this file makes a real call, and the graphs in the fixtures are
written to exercise the pipeline, not to resemble any episode.
"""

import json
import os

import pytest

from future_graph import (
    ArtifactError, ComputationNode, GRAMMAR, InformationKind, InformationNode, ModelCall,
    PromptError, Relation, RegenerationRecord, StateGraph, build, parse, regenerate_graph,
    to_protocol,
)
from future_graph.artifacts import prompt_sha
from future_graph.rendering import render
from future_graph.regeneration import PROMPT_PATH, build_user_message, load_prompt


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


def run(answer, previous=None, config=None):
    stub = Stub(answer)
    previous = previous_graph() if previous is None else previous
    result = regenerate_graph("the goal", "the rules", previous, "the slice", stub,
                              {} if config is None else config)
    return stub, result


EXAMPLE_HEADING = "# An example of the form"


def prompt_example() -> str:
    """The one graph under the example heading, located by the heading rather than by position.

    Taking the last block in the file would silently start checking a different one the day the
    prompt gains a section after the example.
    """
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert text.count(EXAMPLE_HEADING) == 1, "the example heading is not unique"
    after = text.split(EXAMPLE_HEADING, 1)[1]
    assert after.count("```text") == 1, "the example section holds more than one fenced block"
    block = after.split("```text", 1)[1].split("```", 1)[0]
    assert block.count("BEGIN_GRAPH") == 1 and block.count("END_GRAPH") == 1
    return block.strip() + "\n"


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
    regenerate_graph("  goal with space  ", "\nrules\n", StateGraph(), "  slice  ",
                     stub, {})
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


@pytest.mark.parametrize("config", [[("a", 1)], [["a", 1]], (("a", 1),), "a=1", None])
def test_a_config_that_is_not_a_mapping_is_refused_before_any_call(config):
    """Converting it first would accept shapes the interface does not, and fold duplicate keys."""
    stub = Stub(VALID)
    with pytest.raises(TypeError, match="config is a mapping"):
        regenerate_graph("g", "r", StateGraph(), "d", stub, config)
    assert stub.calls == []


def test_there_is_no_way_to_send_a_prompt_other_than_the_repository_one():
    import inspect
    from future_graph.regeneration import build_call
    for function in (regenerate_graph, build_call):
        assert "prompt" not in inspect.signature(function).parameters


def test_config_is_required():
    import inspect
    parameter = inspect.signature(regenerate_graph).parameters["config"]
    assert parameter.default is inspect.Parameter.empty


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
        regenerate_graph("g", "r", previous, "d", stub, {})
    assert len(stub.calls) == 1
    assert previous.to_snapshot() == before


@pytest.mark.parametrize("answer", [None, 7, b"BEGIN_GRAPH", ["BEGIN_GRAPH"]])
def test_a_model_that_returns_something_other_than_text_is_a_broken_adapter(answer):
    previous = previous_graph()
    before = previous.to_snapshot()
    with pytest.raises(TypeError, match="a model returns text"):
        regenerate_graph("g", "r", previous, "d", Stub(answer), {})
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
    """Through the pipeline, because the example is a semantic core by design.

    It writes no interface edges, so raw validation refuses it -- and that is the example doing its
    job: it shows the model the shape of an answer it is expected to give, which the system then
    completes.
    """
    from future_graph.lifecycle import replace
    outcome = parse(prompt_example())
    assert outcome.errors == (), outcome.errors
    result = replace(StateGraph(), outcome.graph)
    assert result.accepted, [str(v) for v in result.violations]
    assert [c.action for c in result.interface_changes] == ["added", "added"]


def test_the_example_never_states_an_ordering_the_information_flow_already_states():
    """A redundant PRECEDES in the example is a redundant PRECEDES in every graph after it."""
    graph = parse(prompt_example()).graph
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


@pytest.mark.parametrize("field", ["previous_snapshot", "resulting_snapshot",
                                   "parsed_candidate_snapshot"])
def test_a_malformed_snapshot_is_caught_while_the_record_is_read(field):
    """Not later, in the middle of an analysis, when someone happens to rebuild the graph."""
    _, result = run(WITH_DEAD_INFORMATION)
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw[field] = {"computations": [], "information": []}
    with pytest.raises(ArtifactError, match=f"record {field}.*missing edges"):
        RegenerationRecord.from_dict(raw)


def test_a_handover_that_is_not_the_rendering_of_its_graph_is_rejected():
    raw = _record_dict()
    raw["handover"] = raw["handover"] + "and one more thing\n"
    with pytest.raises(ArtifactError, match="handover is not the rendering"):
        RegenerationRecord.from_dict(raw)


def test_a_record_that_accepts_and_lists_violations_is_rejected():
    raw = _record_dict()
    raw["violations"] = [["cycle", "a message", []]]
    with pytest.raises(ArtifactError, match="accepted and also lists violations"):
        RegenerationRecord.from_dict(raw)


def test_a_record_that_collected_from_a_graph_it_refused_is_rejected():
    _, result = run(INVALID)
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["collected"] = ["i2"]
    with pytest.raises(ArtifactError, match="collected information from a graph it did not accept"):
        RegenerationRecord.from_dict(raw)


def test_a_refusal_with_nothing_parsed_and_no_parse_error_is_rejected():
    _, result = run(UNPARSEABLE)
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["parse_errors"] = []
    with pytest.raises(ArtifactError, match="refused with nothing parsed and no parse error"):
        RegenerationRecord.from_dict(raw)


def test_a_refusal_of_a_candidate_with_no_violation_is_rejected():
    _, result = run(INVALID)
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["violations"] = []
    with pytest.raises(ArtifactError, match="refused a candidate that had no violation"):
        RegenerationRecord.from_dict(raw)


@pytest.mark.parametrize("answer", [UNPARSEABLE, INVALID])
def test_a_refusal_that_carries_a_different_graph_forward_is_rejected(answer):
    """Otherwise a record could claim to have changed nothing while changing everything."""
    _, result = run(answer)
    other = build(nodes=[ComputationNode(id="c1", description="something else entirely")])
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["resulting_snapshot"] = other.to_snapshot()
    raw["handover"] = render(other)          # keep the handover honest so the outcome check fires
    with pytest.raises(ArtifactError, match="refused a graph and did not keep the previous one"):
        RegenerationRecord.from_dict(raw)


def test_a_record_with_parse_errors_and_a_candidate_is_rejected():
    _, result = run(UNPARSEABLE)
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["parsed_candidate_snapshot"] = {"computations": [], "information": [], "edges": []}
    with pytest.raises(ArtifactError, match="parse errors and a parsed candidate"):
        RegenerationRecord.from_dict(raw)


def test_a_recorded_config_out_of_canonical_order_is_rejected():
    _, result = run(VALID, config={"b": 1, "a": 2})
    raw = json.loads(json.dumps(result.record.to_dict()))
    raw["model_call"]["config"] = [["b", 1], ["a", 2]]
    raw["prompt_sha"] = result.record.prompt_sha
    with pytest.raises(ArtifactError, match="not in the order"):
        RegenerationRecord.from_dict(raw)


def test_a_recorded_config_with_a_repeated_key_is_rejected():
    raw = _record_dict()
    raw["model_call"]["config"] = [["a", 1], ["a", 2]]
    with pytest.raises(ArtifactError, match="appears twice"):
        RegenerationRecord.from_dict(raw)


def test_a_recorded_non_finite_config_value_is_rejected():
    raw = _record_dict()
    raw["model_call"]["config"] = [["temperature", float("inf")]]
    with pytest.raises(ArtifactError, match="never recordable"):
        RegenerationRecord.from_dict(raw)


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


# --------------------------------------------------------------------------- absorbing a slice
#
# Two different questions live below, and conflating them would be the easiest way to overclaim.
#
# The first is fidelity: does the exact slice this system was handed reach the model unrewritten.
# That is a property of `build_user_message` and it is fully testable here.
#
# The second is what the pipeline does with a candidate that absorbed the slice well or badly. Those
# tests hand the pipeline a fixed answer and check that acceptance, collection and refusal behave.
# **They demonstrate nothing about whether a model would write that answer**; only a model-backed run
# can speak to that.

ERROR_SLICE = (
    'ASSISTANT:\nprint(apis.example.open_entry(record_id=91001, catalogue="spring"))\n\n'
    'USER:\n'
    'PermissionError: open_entry requires a curator token.\n'
    '  Obtain one with apis.example.sign_in(username=..., password=...)\n'
    '  and pass it as curator_token="...".\n'
    '  Field \'catalogue\' accepts one of: "spring" | "summer" | \'autumn\'.\n'
    '\tTraceback (most recent call last): 100% [==>] {"code": 403}\n'
)


def test_the_exact_slice_reaches_the_model_unrewritten():
    """Punctuation, quotes of both kinds, tabs, newlines, an operation and a parameter name."""
    stub = Stub(VALID)
    regenerate_graph("the goal", "the rules", previous_graph(), ERROR_SLICE, stub, {})
    user = stub.calls[0].user
    assert f"BEGIN_DELTA_H\n{ERROR_SLICE}\nEND_DELTA_H\n" in user


def test_the_slice_appears_exactly_once_in_the_user_message():
    """Duplicating it would double every observation and change what the model is reading."""
    stub = Stub(VALID)
    regenerate_graph("the goal", "the rules", previous_graph(), ERROR_SLICE, stub, {})
    assert stub.calls[0].user.count(ERROR_SLICE) == 1
    assert stub.calls[0].user.count("BEGIN_DELTA_H") == 1


@pytest.mark.parametrize("fragment", [
    'curator_token="..."', "apis.example.sign_in(username=..., password=...)",
    '"spring" | "summer" | \'autumn\'', '{"code": 403}', "100% [==>]",
    "\tTraceback", "PermissionError: open_entry requires a curator token.",
])
def test_no_fragment_of_the_slice_is_escaped_or_normalized(fragment):
    """Each of these has been mangled by some layer in some system. Not by this one."""
    stub = Stub(VALID)
    regenerate_graph("g", "r", previous_graph(), ERROR_SLICE, stub, {})
    assert fragment in stub.calls[0].user


def test_the_slice_is_recorded_as_it_was_received():
    """A review of what the model saw depends on the record holding the slice, not a copy of it."""
    result = regenerate_graph("g", "r", previous_graph(), ERROR_SLICE, Stub(VALID), {})
    assert result.record.delta_h == ERROR_SLICE
    assert result.record.to_dict()["delta_h"] == ERROR_SLICE


def test_a_slice_without_an_error_takes_the_same_path():
    """No branch anywhere looks at whether the slice contains a failure."""
    plain = "ASSISTANT:\nprint(apis.example.list_entries())\n\nUSER:\n[{'id': 91001}]\n"
    stub = Stub(VALID)
    regenerate_graph("the goal", "the rules", previous_graph(), plain, stub, {})
    assert f"BEGIN_DELTA_H\n{plain}\nEND_DELTA_H\n" in stub.calls[0].user
    assert build_user_message("the goal", "the rules", previous_graph(), plain) \
        == stub.calls[0].user


# --------------------------------------------------------------------------- candidate semantics

REPLACED_BRANCH = """\
BEGIN_GRAPH

INFO i1
kind: failure_consequence
available: true
description: The batch route does not exist, so entries are opened one at a time
END_INFO

INFO i2
kind: contract
available: true
description: The confirmed single-entry interface
contract-operation: example.open_entry
contract-parameter: record_id
contract-parameter: curator_token
END_INFO

INFO i3
kind: runtime_reference
available: true
description: The curator token the sign-in established
runtime-name: curator_token
END_INFO

COMPUTATION c1
description: Open a catalogue entry for each record in turn
operation: example.open_entry
argument curator_token = @i3
END_COMPUTATION

EDGE i1 REQUIRES c1
EDGE i2 REQUIRES c1
EDGE i3 REQUIRES c1

END_GRAPH
"""


def batch_route_graph():
    """The previous plan: one call registering everything at once, plus what only it needed."""
    from future_graph import ContractPayload
    return build(
        nodes=[ComputationNode(id="c1", description="Register every record in one batch call",
                               operation="example.batch_register"),
               InformationNode(id="i1", kind=InformationKind.CONTRACT,
                               description="The batch interface", available=True,
                               payload=ContractPayload("example.batch_register", ("records",))),
               InformationNode(id="i2", kind=InformationKind.FACT,
                               description="The records to register", available=True)],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1")])


def test_a_candidate_that_replaces_an_invalid_branch_is_accepted_and_rendered():
    stub, result = run(REPLACED_BRANCH, previous=batch_route_graph())
    assert result.record.accepted and result.record.violations == ()
    assert [c.id for c in result.graph.computations] == ["c1"]
    assert result.graph.node("c1").operation == "example.open_entry"
    assert "example.batch_register" not in result.record.handover


def test_a_candidate_keeping_exact_recovery_detail_is_accepted_with_it_intact():
    """The bound name, the replacement operation and the parameter names survive verbatim."""
    _, result = run(REPLACED_BRANCH, previous=batch_route_graph())
    handover = result.record.handover
    assert "bound as curator_token" in handover
    assert "example.open_entry, takes record_id, curator_token" in handover
    assert "curator_token = @i3" in handover


def test_information_only_the_removed_branch_used_is_collected():
    """It is not in the candidate at all, so it simply does not survive the replacement."""
    previous = batch_route_graph()
    before = previous.to_snapshot()
    _, result = run(REPLACED_BRANCH, previous=previous)
    surviving = {i.description for i in result.graph.information}
    assert "The batch interface" not in surviving
    assert previous.to_snapshot() == before        # the previous graph itself is untouched


SHARED_AND_STALE = """\
BEGIN_GRAPH

INFO i1
kind: fact
available: true
description: The records to register
END_INFO

INFO i2
kind: fact
available: true
description: Wanted by nothing that remains
END_INFO

COMPUTATION c1
description: Open a catalogue entry for each record in turn
END_COMPUTATION

COMPUTATION c2
description: Confirm every entry was opened
END_COMPUTATION

EDGE i1 REQUIRES c1
EDGE i1 REQUIRES c2
EDGE c1 PRECEDES c2

END_GRAPH
"""


def test_information_a_surviving_branch_still_shares_remains():
    _, result = run(SHARED_AND_STALE, previous=batch_route_graph())
    assert result.record.accepted
    kept = {i.description for i in result.graph.information}
    assert "The records to register" in kept
    assert result.graph.consumers_of("i1") == ("c1", "c2")


def test_information_the_revised_plan_does_not_consume_is_collected():
    _, result = run(SHARED_AND_STALE, previous=batch_route_graph())
    assert result.record.collected == ("i2",)
    assert "Wanted by nothing that remains" not in result.record.handover
    # what the model wrote and what was committed are two different things, and both are recorded
    written = {i["description"] for i in result.record.parsed_candidate_snapshot["information"]}
    assert "Wanted by nothing that remains" in written


PARTIAL_PROGRESS = """\
BEGIN_GRAPH

INFO i1
kind: result
available: true
description: The eleven record ids still to be registered
payload-type: list
item: 91002
item: 91003
END_INFO

INFO i2
kind: runtime_reference
available: true
description: The accumulator holding the entries opened so far
runtime-name: opened_entries
END_INFO

INFO i3
kind: result
available: false
description: The complete set of opened entries
END_INFO

COMPUTATION c1
description: Continue opening entries for the records that remain
END_COMPUTATION

COMPUTATION c2
description: Confirm the complete set is present
END_COMPUTATION

EDGE i1 REQUIRES c1
EDGE i2 REQUIRES c1
EDGE c1 PRODUCES i3
EDGE i3 REQUIRES c2

END_GRAPH
"""


def test_partial_progress_becomes_continuation_information_plus_remaining_work():
    """What was achieved is available; what remains continues; the whole is a separate node."""
    _, result = run(PARTIAL_PROGRESS, previous=batch_route_graph())
    assert result.record.accepted
    graph = result.graph
    assert graph.node("i1").available and graph.node("i2").available
    assert not graph.node("i3").available          # the complete set does not exist yet
    assert graph.produces_of("c1") == ("i3",)
    assert "Continue opening entries" in graph.node("c1").description
    # the partial accumulator and the finished whole are two nodes, so a consumer needing all of
    # it cannot read the part that exists
    assert graph.consumers_of("i2") == ("c1",)
    assert graph.consumers_of("i3") == ("c2",)


def test_a_rejected_candidate_leaves_the_previous_graph_byte_identical():
    previous = batch_route_graph()
    before = previous.to_snapshot()
    _, result = run(INVALID, previous=previous)
    assert result.record.accepted is False
    assert result.graph is previous
    assert previous.to_snapshot() == before
    assert result.record.resulting_snapshot == before
    assert result.record.collected == ()
    assert result.record.delta_h == "the slice"


# --------------------------------------------------------------------------- interface ownership

SEMANTIC_CORE = """\
BEGIN_GRAPH

INFO listing
kind: fact
available: true
description: The confirmed listing interface
END_INFO

INFO gathered
kind: result
available: false
description: The gathered records
END_INFO

COMPUTATION gather
description: Gather the records
END_COMPUTATION

COMPUTATION first_page
description: Retrieve the first page
END_COMPUTATION

COMPUTATION rest
description: Continue to the end
END_COMPUTATION

COMPUTATION use
description: Apply the change to each record
END_COMPUTATION

EDGE gather REFINES first_page
EDGE gather REFINES rest
EDGE listing REQUIRES first_page
EDGE rest PRODUCES gathered
EDGE gathered REQUIRES use

END_GRAPH
"""


def test_a_candidate_with_labels_and_no_interfaces_is_accepted():
    """Both repairs at once: hierarchical-free labels canonicalized, interfaces derived."""
    _, result = run(SEMANTIC_CORE, previous=previous_graph())
    assert result.record.accepted, result.record.violations
    assert [c.id for c in result.graph.computations] == ["c1", "c2", "c3", "c4"]


def test_the_record_carries_the_interface_edits():
    _, result = run(SEMANTIC_CORE, previous=previous_graph())
    assert result.record.interface_changes == (
        ("added", "i1", "interface_input", "c1"),
        ("added", "c1", "interface_output", "i2"))


def test_the_parsed_candidate_is_recorded_before_completion():
    """The snapshot shows what the model wrote; the result shows what was committed."""
    _, result = run(SEMANTIC_CORE, previous=previous_graph())
    written = result.record.parsed_candidate_snapshot["edges"]
    committed = result.record.resulting_snapshot["edges"]
    assert not any(e["relation"].startswith("interface") for e in written)
    assert sum(e["relation"].startswith("interface") for e in committed) == 2


def test_an_unchanged_boundary_records_no_interface_changes():
    _, result = run(VALID, previous=previous_graph())
    assert result.record.accepted and result.record.interface_changes == ()


def test_the_interface_changes_round_trip_through_the_record():
    _, result = run(SEMANTIC_CORE, previous=previous_graph())
    payload = json.loads(json.dumps(result.record.to_dict()))
    assert RegenerationRecord.from_dict(payload).interface_changes \
        == result.record.interface_changes


@pytest.mark.parametrize("entry,fragment", [
    ([["moved", "i1", "interface_input", "c1"]], "removed or added"),
    ([["added", "i1", "interface_input"]], "action, a source"),
    ([["added", "i1", "interface_input", 7]], "action, a source"),
])
def test_a_malformed_interface_change_entry_is_rejected(entry, fragment):
    raw = _record_dict()
    raw["interface_changes"] = entry
    with pytest.raises(ArtifactError, match=fragment):
        RegenerationRecord.from_dict(raw)
