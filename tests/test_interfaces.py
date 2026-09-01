"""Who writes which edge, and what the code does with the ones it owns.

The split under test: every `INTERFACE_INPUT` and every `INTERFACE_OUTPUT` naming information that
does not exist yet are derived from the dataflow, so whatever the model wrote for them is removed and
the real set is put in its place. An `INTERFACE_OUTPUT` naming information that already exists is a
provenance claim the graph cannot check for itself, so it is kept and validated.

The point of removing even a correctly written declaration is that correctness stops being the
model's problem: a candidate must not fail because residue of a relation it no longer owns was
missing, doubled or reversed.
"""

import pytest

from future_graph import (
    ComputationNode, InformationKind, InformationNode, Relation, build,
)
from future_graph.interfaces import (
    complete_interfaces, crossing_inputs, crossing_unavailable_outputs,
)
from future_graph.lifecycle import replace
from future_graph.state_graph import StateGraph
from future_graph.validation import validate


def comp(cid, description="Do the work", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description="the value", kind=InformationKind.FACT, available=True):
    return InformationNode(id=iid, kind=kind, description=description, available=available)


def semantic_core():
    """What the model is now asked for: computations, information, REFINES, leaf dataflow.

    c1 is refined into c2 and c3. i1 comes from outside and c2 needs it; i2 is established by c3
    and consumed by c4, which is outside. No interface edge is written by hand.
    """
    return build(
        nodes=[comp("c1", "Gather the records"), comp("c2", "Retrieve the first page"),
               comp("c3", "Continue to the end"), comp("c4", "Use the records"),
               info("i1", "the listing interface"),
               info("i2", "the gathered records", kind=InformationKind.RESULT, available=False)],
        edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
               ("i1", Relation.REQUIRES, "c2"), ("c3", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c4")])


def relations(graph, relation):
    return sorted((e.source, e.target) for e in graph.edges_of(relation))


# --------------------------------------------------------------------------- the sets

def test_the_crossing_sets_are_what_completion_and_validation_both_read():
    g = semantic_core()
    assert crossing_inputs(g, "c1") == {"i1"}
    assert crossing_unavailable_outputs(g, "c1") == {"i2"}


def test_information_made_and_used_inside_crosses_nothing():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"),
                     info("i1", kind=InformationKind.RESULT, available=False)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
                     ("c2", Relation.PRODUCES, "i1"), ("i1", Relation.REQUIRES, "c3")])
    assert crossing_inputs(g, "c1") == set()
    assert crossing_unavailable_outputs(g, "c1") == set()


# --------------------------------------------------------------------------- completion

def test_omitted_interfaces_are_generated():
    completed, changes = complete_interfaces(semantic_core())
    assert relations(completed, Relation.INTERFACE_INPUT) == [("i1", "c1")]
    assert relations(completed, Relation.INTERFACE_OUTPUT) == [("c1", "i2")]
    assert [(c.action, c.source, c.target) for c in changes] == [
        ("added", "i1", "c1"), ("added", "c1", "i2")]
    assert validate(completed) == ()


def test_a_correctly_written_declaration_is_removed_and_regenerated_silently():
    """It ends up in the graph either way, and the record does not pretend an edit happened."""
    g = semantic_core()
    g.add_edge("i1", Relation.INTERFACE_INPUT, "c1")
    completed, changes = complete_interfaces(g)
    assert relations(completed, Relation.INTERFACE_INPUT) == [("i1", "c1")]
    assert [(c.action, c.source, c.target) for c in changes] == [("added", "c1", "i2")]


def test_a_backwards_interface_input_is_removed_and_the_real_one_generated():
    g = semantic_core()
    g.add_edge("c1", Relation.INTERFACE_INPUT, "i1")        # written the wrong way round
    completed, changes = complete_interfaces(g)
    assert relations(completed, Relation.INTERFACE_INPUT) == [("i1", "c1")]
    assert ("removed", "c1", "i1") in [(c.action, c.source, c.target) for c in changes]
    assert validate(completed) == ()


def test_an_extra_deterministic_declaration_is_removed_and_not_regenerated():
    """Nothing crosses, so nothing is put back, and it is not a refusal either."""
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"),
                     info("i1", kind=InformationKind.RESULT, available=False)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
                     ("c2", Relation.PRODUCES, "i1"), ("i1", Relation.REQUIRES, "c3"),
                     ("c1", Relation.INTERFACE_OUTPUT, "i1")])
    completed, changes = complete_interfaces(g)
    assert relations(completed, Relation.INTERFACE_OUTPUT) == []
    assert [(c.action, c.source, c.target) for c in changes] == [("removed", "c1", "i1")]
    assert validate(completed) == ()


def test_nested_boundaries_are_completed_independently():
    """i1 crosses into c2 only; c1 and c2 get the interfaces each of them actually has."""
    g = build(nodes=[comp("c1", "The whole job"), comp("c2", "The part that reads"),
                     comp("c3", "Read a page"), comp("c5", "Work out where to start"),
                     info("i1", "the starting point", kind=InformationKind.RESULT,
                          available=False)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c5"),
                     ("c2", Relation.REFINES, "c3"),
                     ("c5", Relation.PRODUCES, "i1"), ("i1", Relation.REQUIRES, "c3")])
    completed, _ = complete_interfaces(g)
    assert relations(completed, Relation.INTERFACE_INPUT) == [("i1", "c2")]
    assert relations(completed, Relation.INTERFACE_OUTPUT) == []
    assert validate(completed) == ()


def test_an_available_interface_output_is_preserved():
    """The producing child has left the graph; only the model can say the subtree delivered it."""
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3", "Use it"), info("i1", "an earlier result")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i1"),
                     ("i1", Relation.REQUIRES, "c3")])
    completed, changes = complete_interfaces(g)
    assert relations(completed, Relation.INTERFACE_OUTPUT) == [("c1", "i1")]
    assert changes == ()
    assert validate(completed) == ()


def test_an_available_output_still_produced_inside_is_still_refused():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3", "Use it"), info("i1", "an earlier result")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i1"),
                     ("c2", Relation.PRODUCES, "i1"), ("i1", Relation.REQUIRES, "c3")])
    completed, _ = complete_interfaces(g)
    assert "available_interface_output_is_produced_inside" in [v.code for v in validate(completed)]


def test_an_available_output_nothing_outside_consumes_is_still_refused():
    g = build(nodes=[comp("c1"), comp("c2"), info("i1", "an earlier result")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i1")])
    completed, _ = complete_interfaces(g)
    assert "internal_information_declared_as_output" in [v.code for v in validate(completed)]


def test_an_available_output_on_a_leaf_is_still_refused():
    """The source of an available interface output has to be a refined computation."""
    g = build(nodes=[comp("c1"), comp("c2", "Use it"), info("i1", "an earlier result")],
              edges=[("c1", Relation.INTERFACE_OUTPUT, "i1"), ("i1", Relation.REQUIRES, "c2")])
    completed, _ = complete_interfaces(g)
    assert "leaf_interface_edge" in [v.code for v in validate(completed)]


def test_changes_are_ordered_deterministically_and_empty_when_nothing_moves():
    g = semantic_core()
    assert complete_interfaces(g)[1] == complete_interfaces(g)[1]
    completed, _ = complete_interfaces(g)
    assert complete_interfaces(completed)[1] == ()      # already complete: nothing to report


def test_completion_does_not_touch_the_candidate():
    g = semantic_core()
    before = g.to_snapshot()
    complete_interfaces(g)
    assert g.to_snapshot() == before


# --------------------------------------------------------------------------- in the pipeline

def test_completion_happens_before_validation():
    """The semantic core alone would fail the interface invariants; completed, it passes."""
    candidate = semantic_core()
    assert [v.code for v in validate(candidate)] == [
        "undeclared_interface_input", "undeclared_interface_output"]
    result = replace(StateGraph(), candidate)
    assert result.accepted
    assert relations(result.graph, Relation.INTERFACE_INPUT) == [("i1", "c1")]


def test_a_refusal_after_completion_leaves_the_previous_graph_untouched():
    previous = build(nodes=[comp("c1", "The work as it stood"), info("i1")],
                     edges=[("i1", Relation.REQUIRES, "c1")])
    before = previous.to_snapshot()
    candidate = build(nodes=[comp("c1"), comp("c2"), info("i1", "an earlier result")],
                      edges=[("c1", Relation.REFINES, "c2"),
                             ("c1", Relation.INTERFACE_OUTPUT, "i1")])
    result = replace(previous, candidate)
    assert result.rejected and result.graph is previous
    assert previous.to_snapshot() == before
    assert result.interface_changes == ()      # nothing was code-owned here
