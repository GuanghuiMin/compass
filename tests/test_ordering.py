"""A `PRECEDES` edge from a computation to itself, and the one that looks like it but is not."""

import pytest

from future_graph import ComputationNode, InformationKind, InformationNode, Relation, build
from future_graph.lifecycle import replace
from future_graph.ordering import remove_reflexive_precedes
from future_graph.state_graph import StateGraph
from future_graph.validation import validate


def comp(cid, description="Do the work", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def test_a_precedes_self_loop_is_removed():
    """It says c1 runs before c1: no execution satisfies it and none is forbidden by it."""
    g = build(nodes=[comp("c1")], edges=[("c1", Relation.PRECEDES, "c1")])
    assert "cycle" in [v.code for v in validate(g)]
    repaired, repairs = remove_reflexive_precedes(g)
    assert repaired.edges == ()
    assert [(r.action, r.source, r.relation.value, r.target) for r in repairs] == [
        ("removed", "c1", "precedes", "c1")]
    assert validate(repaired) == ()


def test_ordinary_ordering_is_untouched():
    g = build(nodes=[comp("c1"), comp("c2")], edges=[("c1", Relation.PRECEDES, "c2")])
    repaired, repairs = remove_reflexive_precedes(g)
    assert repairs == ()
    assert repaired is g          # nothing to do, so nothing is rebuilt


def test_a_refines_self_loop_is_left_alone_and_refused():
    """It may be standing where a missing child should be, and dropping it would turn a refined
    obligation into a leaf. There is no unique repair, so the cycle check has it."""
    g = build(nodes=[comp("c1")], edges=[("c1", Relation.REFINES, "c1")])
    repaired, repairs = remove_reflexive_precedes(g)
    assert repairs == ()
    assert "cycle" in [v.code for v in validate(repaired)]


def test_a_two_step_ordering_cycle_is_still_refused():
    """Only the reflexive case is impossible to mean anything by; a real cycle is a real fault."""
    g = build(nodes=[comp("c1"), comp("c2")],
              edges=[("c1", Relation.PRECEDES, "c2"), ("c2", Relation.PRECEDES, "c1")])
    repaired, repairs = remove_reflexive_precedes(g)
    assert repairs == ()
    assert "cycle" in [v.code for v in validate(repaired)]


def test_the_candidate_is_not_mutated():
    g = build(nodes=[comp("c1")], edges=[("c1", Relation.PRECEDES, "c1")])
    before = g.to_snapshot()
    remove_reflexive_precedes(g)
    assert g.to_snapshot() == before


def test_the_repair_reaches_the_pipeline_and_is_recorded_apart():
    """It is a graph edit, so it travels with the graph edits and not with the normalizations."""
    candidate = build(
        nodes=[comp("c1", "Close the task out", operation="apis.supervisor.complete_task"),
               InformationNode(id="i1", kind=InformationKind.FACT, description="what was done",
                               available=True)],
        edges=[("c1", Relation.PRECEDES, "c1"), ("i1", Relation.REQUIRES, "c1")])
    result = replace(StateGraph(), candidate)
    assert result.accepted
    assert [(r.action, r.source, r.target) for r in result.ordering_repairs] == [
        ("removed", "c1", "c1")]
    assert result.graph.edges_of(Relation.PRECEDES) == ()
