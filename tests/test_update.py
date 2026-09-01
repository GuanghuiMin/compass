"""The updater as a boundary actually runs it, and the record it has to leave behind.

The model is a stub returning fixed text, because what is under test is everything around the model:
that the call carries the four inputs, that a refusal keeps the previous graph byte-identically and
discards the slice, that the record reads back as exactly what it claims, and that each change is
filed under whoever made it.
"""

import json

import pytest

from future_graph import ComputationNode as C, InformationNode as I
from future_graph import InformationKind as K, Relation as R, build
from future_graph.adapter import EmptyModelCompletion
from future_graph.artifacts import ArtifactError, RevisionRecord
from future_graph.retry import ExhaustedAttempts
from future_graph.state_graph import StateGraph
from future_graph.update import build_user_message, load_prompt, update_graph

CONFIG = {"model": "a-model", "temperature": 0.0}


def stub(text):
    def model(call):
        return text
    return model


@pytest.fixture
def no_real_waiting(monkeypatch):
    """Retry backoff is real seconds in a run and must not be real seconds in the suite."""
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _seconds: None)


@pytest.fixture
def nursery():
    return build(
        nodes=[C(id="c1", description="Register every seedling"),
               C(id="c2", description="Open an entry", operation="nursery.create_entry"),
               C(id="c3", description="Attach the photo", operation="nursery.attach_photo"),
               I(id="i1", kind=K.FACT, description="The twelve seedlings", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"),
               ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.REQUIRES, "c2"),
               ("c2", R.PRECEDES, "c3")])


GOOD = """BEGIN_REVISION
REPLACE c2
reason-for-replacement: create_entry was retired in favour of create_entry_v2
COMPUTATION +open
description: Open a catalogue entry
operation: nursery.create_entry_v2
requires: i1
END_COMPUTATION
END_REPLACE
REVISE c3
add-after: +open
END_REVISE
END_REVISION
"""

UNPARSABLE = "I have considered the trajectory and nothing needs to change.\n"

REFUSED = """BEGIN_REVISION
REPLACE c2
reason-for-replacement: it changed
COMPUTATION +open
description: Open a catalogue entry
operation: nursery.create_entry_v2
END_COMPUTATION
END_REPLACE
REVISE c3
add-after: +open
END_REVISE
END_REVISION
"""

EMPTY = "BEGIN_REVISION\nEND_REVISION\n"


# --------------------------------------------------------------------------- the call

def test_the_prompt_carries_the_revision_grammar_once():
    prompt = load_prompt()
    assert "{{REVISION_GRAMMAR}}" not in prompt
    assert prompt.count("BEGIN_REVISION") >= 1
    assert "END_REVISION" in prompt


def test_the_message_carries_the_four_inputs_in_order(nursery):
    message = build_user_message("the goal", "the rules", nursery, "the slice")
    assert message.index("ORIGINAL_GOAL") < message.index("FIXED_RULES") \
        < message.index("PREVIOUS_GRAPH") < message.index("DELTA_H")
    assert "the goal" in message and "the rules" in message and "the slice" in message
    assert "Register every seedling" in message


def test_an_empty_previous_graph_is_shown_as_one():
    message = build_user_message("g", "r", StateGraph(), "d")
    assert "BEGIN_GRAPH" in message and "END_GRAPH" in message


def test_the_configuration_travels_with_the_call(nursery):
    result = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG)
    assert result.record.model_call.config == (("model", "a-model"), ("temperature", 0.0))


# --------------------------------------------------------------------------- accepted

def test_an_accepted_revision_changes_the_graph_and_says_what_it_changed(nursery):
    result = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG)
    record = result.record
    assert record.accepted
    assert record.faults == () and record.violations == () and record.parse_errors == ()
    assert [str(r) for r in record.affected_roots] == ["previous:c2"]
    assert [str(r) for r in record.touched_nodes] == ["previous:c3"]
    assert {n.description for n in result.graph.computations} == {
        "Register every seedling", "Open a catalogue entry", "Attach the photo"}


def test_the_handover_is_the_rendering_of_what_was_committed(nursery):
    result = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG)
    from future_graph.rendering import render
    assert result.record.handover == render(result.graph)


def test_the_graph_before_completion_is_kept_beside_the_one_after(nursery):
    result = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG)
    record = result.record
    assert record.assembled_snapshot is not None
    # The assembled graph is what the revision said; the committed one also holds the interface
    # edges the code derived from it, so they are not the same object and must not be recorded as
    # if they were.
    assert record.assembled_snapshot != record.resulting_snapshot


# --------------------------------------------------------------------------- the empty revision

def test_an_empty_revision_is_accepted_and_changes_nothing(nursery):
    before = nursery.to_snapshot()
    result = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG)
    record = result.record
    assert record.accepted and record.empty_revision
    assert result.graph is nursery
    assert record.resulting_snapshot == before
    assert record.affected_roots == () and record.touched_nodes == ()
    assert record.removed_nodes == () and record.removed_edges == ()
    assert record.completion_changes == () and record.interface_changes == ()
    assert record.argument_dependency_changes == () and record.ordering_repairs == ()
    assert record.collected == ()


def test_an_empty_revision_leaves_the_handover_as_it_was(nursery):
    from future_graph.rendering import render
    before = render(nursery)
    result = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG)
    assert result.record.handover == before


def test_an_empty_revision_does_not_run_the_pipeline_over_the_previous_graph():
    """Collection edits a graph in place. Handing the previous graph to the completion pipeline
    would let a boundary that said nothing quietly edit the state it was preserving."""
    previous = build(
        nodes=[C(id="c1", description="Register every seedling"),
               I(id="i1", kind=K.FACT, description="Something nothing needs", available=True)],
        edges=[])
    before = previous.to_snapshot()
    result = update_graph("g", "r", previous, "d", stub(EMPTY), CONFIG)
    assert result.record.accepted and result.record.collected == ()
    assert previous.to_snapshot() == before
    assert result.record.resulting_snapshot == before


def test_an_empty_revision_against_an_empty_graph_is_still_a_no_op():
    result = update_graph("g", "r", StateGraph(), "d", stub(EMPTY), CONFIG)
    assert result.record.accepted and result.record.empty_revision
    assert result.graph.computations == ()


# --------------------------------------------------------------------------- refusal

@pytest.mark.parametrize("output", [UNPARSABLE, REFUSED])
def test_a_refused_boundary_keeps_the_previous_graph_byte_identically(nursery, output):
    before = json.dumps(nursery.to_snapshot(), sort_keys=True)
    result = update_graph("g", "r", nursery, "d", stub(output), CONFIG)
    assert not result.record.accepted
    assert json.dumps(result.record.resulting_snapshot, sort_keys=True) == before
    assert json.dumps(result.graph.to_snapshot(), sort_keys=True) == before
    assert result.record.collected == ()


def test_a_refused_boundary_keeps_the_slice_and_does_not_carry_it_forward(nursery):
    """The slice belongs to the boundary that failed. Replaying it later would hand the method a
    second attempt the schedule never gave it, and a boundary that was not absorbed is a real loss
    that belongs in the results as one."""
    first = update_graph("g", "r", nursery, "slice one", stub(REFUSED), CONFIG)
    assert not first.record.accepted
    assert first.record.delta_h == "slice one"

    second = update_graph("g", "r", first.graph, "slice two", stub(EMPTY), CONFIG)
    assert second.record.delta_h == "slice two"
    assert "slice one" not in second.record.model_call.user


def test_text_instead_of_a_revision_is_a_parse_failure(nursery):
    record = update_graph("g", "r", nursery, "d", stub(UNPARSABLE), CONFIG).record
    assert record.parse_errors and not record.faults and not record.violations
    assert record.assembled_snapshot is None


def test_an_unaccounted_crossing_is_a_fault_and_not_a_parse_error(nursery):
    record = update_graph("g", "r", nursery, "d", stub(REFUSED), CONFIG).record
    assert record.faults and not record.parse_errors
    assert record.faults[0].code == "unaccounted_crossing_relation"


def test_a_model_that_returns_something_other_than_text_is_a_caller_mistake(nursery):
    with pytest.raises(TypeError, match="a model returns text"):
        update_graph("g", "r", nursery, "d", lambda call: None, CONFIG)


# --------------------------------------------------------------------------- who did what

def test_the_record_separates_the_model_s_changes_from_every_derived_one():
    previous = build(
        nodes=[C(id="c1", description="Register every seedling"),
               C(id="c2", description="Open an entry", operation="nursery.create_entry"),
               I(id="i1", kind=K.FACT, description="The twelve seedlings", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("i1", R.INTERFACE_INPUT, "c1"),
               ("i1", R.REQUIRES, "c2")])
    output = """BEGIN_REVISION
ADD
COMPUTATION +login
description: Obtain a curator token
operation: nursery.login
produces: +token
END_COMPUTATION
INFORMATION +token
kind: result
available: false
description: A curator token
END_INFORMATION
END_ADD
REPLACE c2
reason-for-replacement: create_entry needs a curator token
COMPUTATION +open
description: Open a catalogue entry
operation: nursery.create_entry
argument token = @+token
requires: i1
after: +login
END_COMPUTATION
END_REPLACE
END_REVISION
"""
    record = update_graph("g", "r", previous, "d", stub(output), CONFIG).record
    assert record.accepted
    assert [str(r) for r in record.affected_roots] == ["previous:c2"]
    assert [str(r.node) for r in record.removed_nodes if r.reason == "affected_region"] \
        == ["previous:c2"]
    assert record.replacement_boundary_changes            # the position the replacement inherited
    assert record.argument_dependency_changes             # the edge the argument already implied
    assert record.interface_changes                       # the boundary the dataflow crosses
    assert record.completion_changes == ()                # nothing completed


def test_completion_changes_are_their_own_field():
    previous = build(
        nodes=[C(id="c1", description="Confirm the interface"),
               C(id="c2", description="Read it", operation="nursery.describe"),
               C(id="c3", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="How registration works", available=False)],
        edges=[("c1", R.REFINES, "c2"), ("c2", R.PRODUCES, "i1"),
               ("c1", R.INTERFACE_OUTPUT, "i1"), ("i1", R.REQUIRES, "c3")])
    output = """BEGIN_REVISION
COMPLETE c1
NOW_AVAILABLE i1
kind: contract
description: The confirmed interface for registering one seedling
contract-operation: nursery.register_one
END_NOW_AVAILABLE
END_COMPLETE
END_REVISION
"""
    record = update_graph("g", "r", previous, "d", stub(output), CONFIG).record
    assert record.accepted
    actions = {change.action for change in record.completion_changes}
    assert actions == {"became_available", "producer_removed", "content_replaced"}
    # The availability transition is the model's declaration carried out, so it is filed there and
    # nowhere else. Nothing was derived on this boundary.
    assert record.interface_changes == ()
    assert record.argument_dependency_changes == ()
    assert [(n["id"], n["kind"], n["available"])
            for n in record.resulting_snapshot["information"]] == [("i1", "contract", True)]


def test_information_created_and_thrown_away_in_one_boundary_is_surfaced(nursery):
    """Two readings the code cannot tell apart: a call worth making for its effect whose return
    value nobody needs, or a result the model established and forgot to connect. Collection stays
    as it is; the audit decides which happened."""
    output = """BEGIN_REVISION
REPLACE c2
reason-for-replacement: create_entry returns an entry id now
COMPUTATION +open
description: Open a catalogue entry
operation: nursery.create_entry_v2
requires: i1
produces: +entry_ids
END_COMPUTATION
INFORMATION +entry_ids
kind: result
available: false
description: The entry ids the calls returned
END_INFORMATION
END_REPLACE
REVISE c3
remove-after: c2
add-after: +open
END_REVISE
END_REVISION
"""
    record = update_graph("g", "r", nursery, "d", stub(output), CONFIG).record
    assert record.accepted
    created = next(m.target for m in record.id_map if m.source.id == "+entry_ids")
    assert record.newly_created_then_collected == (created,)
    assert created in record.collected


def test_nothing_is_surfaced_when_what_was_created_is_consumed(nursery):
    output = """BEGIN_REVISION
REPLACE c2
reason-for-replacement: create_entry returns an entry id now
COMPUTATION +open
description: Open a catalogue entry
operation: nursery.create_entry_v2
requires: i1
produces: +entry_ids
END_COMPUTATION
INFORMATION +entry_ids
kind: result
available: false
description: The entry ids the calls returned
END_INFORMATION
END_REPLACE
REVISE c3
remove-after: c2
add-requires: +entry_ids
END_REVISE
END_REVISION
"""
    record = update_graph("g", "r", nursery, "d", stub(output), CONFIG).record
    assert record.accepted
    assert record.newly_created_then_collected == ()
    assert record.collected == ()


def test_normalizations_are_recorded_when_the_form_needed_repair(nursery):
    record = update_graph("g", "r", nursery, "d", stub(GOOD.lower()), CONFIG).record
    assert any("structural keyword case" in n for n in record.normalizations)


# --------------------------------------------------------------------------- the record itself

@pytest.mark.parametrize("output", [GOOD, EMPTY, UNPARSABLE, REFUSED])
def test_every_record_reads_back_as_what_it_says_it_is(nursery, output):
    record = update_graph("g", "r", nursery, "d", stub(output), CONFIG).record
    again = RevisionRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert again == record


def test_a_record_claiming_an_empty_revision_that_did_work_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["empty_revision"] = True
    with pytest.raises(ArtifactError, match="empty"):
        RevisionRecord.from_dict(raw)


def test_a_record_that_refused_and_changed_the_graph_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(REFUSED), CONFIG).record.to_dict()
    raw["resulting_snapshot"] = {"computations": [], "information": [], "edges": []}
    with pytest.raises(ArtifactError):
        RevisionRecord.from_dict(raw)


def test_a_record_that_accepted_and_also_faulted_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["faults"] = [["something", "went wrong", []]]
    with pytest.raises(ArtifactError, match="faults"):
        RevisionRecord.from_dict(raw)


def test_a_record_with_an_invented_removal_reason_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["removed_nodes"] = [["c2", "felt_like_it"]]
    with pytest.raises(ArtifactError, match="removed_nodes"):
        RevisionRecord.from_dict(raw)


def test_a_record_whose_handover_does_not_render_its_graph_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["handover"] = "something else"
    with pytest.raises(ArtifactError, match="handover"):
        RevisionRecord.from_dict(raw)


def test_a_record_missing_a_field_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    del raw["id_map"]
    with pytest.raises(ArtifactError, match="missing id_map"):
        RevisionRecord.from_dict(raw)


def test_a_record_with_a_field_it_never_had_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["retries"] = 2
    with pytest.raises(ArtifactError, match="unknown retries"):
        RevisionRecord.from_dict(raw)


def test_a_record_whose_hash_does_not_match_its_prompt_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["model_call"]["system"] = "a different prompt"
    with pytest.raises(ArtifactError, match="prompt_sha"):
        RevisionRecord.from_dict(raw)


# --------------------------------------------------------------------------- attempts

def failing_then(*answers):
    """A model that raises or returns each answer in turn, counting what it received."""
    remaining = list(answers)
    calls = []

    def model(call):
        calls.append(call)
        answer = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    model.calls = calls
    return model


def test_a_boundary_survives_an_empty_completion(nursery, no_real_waiting):
    model = failing_then(EmptyModelCompletion("nothing came back"), EMPTY)
    result = update_graph("g", "r", nursery, "d", model, CONFIG)
    assert result.record.accepted
    assert [a[1] for a in result.record.attempts] == ["EmptyModelCompletion", "completion"]


def test_a_boundary_records_the_one_attempt_it_needed(nursery):
    result = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG)
    assert result.record.attempts == ((1, "completion", f"{len(EMPTY)} characters"),)


def test_a_boundary_whose_provider_never_answered_has_no_result(nursery, no_real_waiting):
    """A provider that did not answer is not a compressor that answered badly, so there is no
    record claiming a refusal, and the attempts travel on the exception instead."""
    model = failing_then(*[EmptyModelCompletion("nothing")] * 3)
    with pytest.raises(ExhaustedAttempts) as raised:
        update_graph("g", "r", nursery, "d", model, CONFIG)
    assert len(raised.value.attempts) == 3


@pytest.mark.parametrize("output", [UNPARSABLE, REFUSED])
def test_a_revision_this_system_rejected_is_never_retried(nursery, output):
    """The one thing that would make every acceptance rate a best-of-three."""
    model = failing_then(output, EMPTY, EMPTY)
    result = update_graph("g", "r", nursery, "d", model, CONFIG)
    assert not result.record.accepted
    assert len(model.calls) == 1
    assert result.record.attempts == ((1, "completion", f"{len(output)} characters"),)


def test_a_record_with_no_attempts_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG).record.to_dict()
    raw["attempts"] = []
    with pytest.raises(ArtifactError, match="a call was made"):
        RevisionRecord.from_dict(raw)


def test_a_record_whose_last_attempt_failed_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG).record.to_dict()
    raw["attempts"] = [[1, "EmptyModelCompletion", "nothing"]]
    with pytest.raises(ArtifactError, match="did not complete"):
        RevisionRecord.from_dict(raw)


def test_a_record_that_retried_a_completion_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG).record.to_dict()
    raw["attempts"] = [[1, "completion", "x"], [2, "completion", "y"]]
    with pytest.raises(ArtifactError, match="not retried"):
        RevisionRecord.from_dict(raw)


def test_a_record_with_gaps_in_its_attempts_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(EMPTY), CONFIG).record.to_dict()
    raw["attempts"] = [[1, "RateLimited", "429"], [3, "completion", "x"]]
    with pytest.raises(ArtifactError, match="ordinals"):
        RevisionRecord.from_dict(raw)


# --------------------------------------------------------------------------- identity spaces

def test_a_removed_previous_id_and_a_different_assembled_id_cannot_be_confused():
    """Boundary 4 of the recurrent run, exactly. Completing the whole plan removed the previous
    `i2`, and the node that had been `i3` was renumbered into `i2` and then collected. Written as
    bare strings the two are indistinguishable, and any join across the fields is wrong."""
    previous = build(
        nodes=[C(id="c1", description="Update the playlist"),
               C(id="c2", description="Find the playlist", operation="spotify.playlists"),
               C(id="c3", description="Add the songs", operation="spotify.add"),
               I(id="i1", kind=K.FACT, description="The access token", available=True),
               I(id="i2", kind=K.RESULT, description="The playlist id", available=False),
               I(id="i3", kind=K.FACT, description="The parsed suggestions", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"),
               ("i1", R.REQUIRES, "c2"), ("i1", R.REQUIRES, "c3"),
               ("c2", R.PRODUCES, "i2"), ("i2", R.REQUIRES, "c3"),
               ("i3", R.REQUIRES, "c3")])
    record = update_graph("g", "r", previous, "d",
                          stub("BEGIN_REVISION\nCOMPLETE c1\nEND_COMPLETE\nEND_REVISION\n"),
                          CONFIG).record
    assert record.accepted

    removed = {str(r.node) for r in record.removed_nodes}
    collected = {str(r) for r in record.collected}
    assert "previous:i2" in removed
    assert "assembled:i2" in collected
    # The strings do not collide, so nothing joins them; and they really are different nodes.
    assert not removed & collected
    was = {n["id"]: n["description"] for n in record.previous_snapshot["information"]}
    became = {n["id"]: n["description"] for n in record.assembled_snapshot["information"]}
    assert was["i2"] == "The playlist id"
    assert became["i2"] == "The parsed suggestions"


def test_every_reference_in_a_record_says_which_graph_it_belongs_to(nursery):
    record = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record
    raw = record.to_dict()
    assert set(raw["id_spaces"]) == {"previous", "revision", "assembled"}
    for field in ("affected_roots", "touched_nodes", "newly_created_then_collected", "collected"):
        assert all(set(item) == {"id", "space"} for item in raw[field])
    for field in ("removed_edges", "replacement_boundary_changes", "interface_changes",
                  "argument_dependency_changes", "ordering_repairs"):
        for item in raw[field]:
            assert set(item["source"]) == {"id", "space"}
            assert set(item["target"]) == {"id", "space"}


def test_each_field_is_held_to_the_one_graph_it_can_talk_about(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["affected_roots"] = [{"id": "c2", "space": "assembled"}]
    with pytest.raises(ArtifactError, match="affected_roots"):
        RevisionRecord.from_dict(raw)


def test_a_removed_node_cannot_claim_an_assembled_identity(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["removed_nodes"] = [{"node": {"id": "c2", "space": "assembled"},
                             "reason": "affected_region"}]
    with pytest.raises(ArtifactError, match="only in the previous graph"):
        RevisionRecord.from_dict(raw)


def test_a_reference_in_a_space_that_does_not_exist_is_refused(nursery):
    raw = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record.to_dict()
    raw["collected"] = [{"id": "i9", "space": "imagined"}]
    with pytest.raises(ArtifactError, match="imagined"):
        RevisionRecord.from_dict(raw)


def test_the_mapping_runs_from_what_the_model_wrote_to_what_it_became(nursery):
    record = update_graph("g", "r", nursery, "d", stub(GOOD), CONFIG).record
    spaces = {(m.source.space, m.target.space) for m in record.id_map}
    assert spaces <= {("previous", "assembled"), ("revision", "assembled")}
    assert ("revision", "assembled") in spaces
    assert any(m.source.id == "+open" for m in record.id_map)


def test_a_completion_keeps_its_removed_producer_in_the_graph_it_came_from():
    previous = build(
        nodes=[C(id="c1", description="Read it", operation="nursery.describe"),
               C(id="c2", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="How registration works", available=False)],
        edges=[("c1", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c2")])
    output = ("BEGIN_REVISION\nCOMPLETE c1\nNOW_AVAILABLE i1\nkind: contract\n"
              "contract-operation: nursery.register_one\nEND_NOW_AVAILABLE\n"
              "END_COMPLETE\nEND_REVISION\n")
    record = update_graph("g", "r", previous, "d", stub(output), CONFIG).record
    removed = next(c for c in record.completion_changes if c.action == "producer_removed")
    assert str(removed.node) == "assembled:i1"
    assert str(removed.producer) == "previous:c1"
    assert all(c.producer is None for c in record.completion_changes
               if c.action != "producer_removed")


# --------------------------------------------------------------------------- routing

def test_the_updater_does_not_go_through_complete_graph_regeneration(nursery):
    """The baseline stays reachable and stays a baseline. If the normal path quietly used it, the
    protocol failure surface would grow with the state again, which is the thing this replaced."""
    seen = {}

    def model(call):
        seen["system"] = call.system
        return EMPTY

    update_graph("g", "r", nursery, "d", model, CONFIG)
    assert "BEGIN_REVISION" in seen["system"]
    assert "Return the whole graph every time" not in seen["system"]


def test_the_baseline_still_works_and_is_untouched(nursery):
    from future_graph import regenerate_graph
    whole = """BEGIN_GRAPH
COMPUTATION c1
description: Register every seedling
END_COMPUTATION
END_GRAPH
"""
    result = regenerate_graph("g", "r", nursery, "d", stub(whole), CONFIG)
    assert result.record.accepted
    assert "Return the whole graph every time" in result.record.model_call.system
