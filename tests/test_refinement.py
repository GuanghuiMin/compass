"""Coarse computations, their children, and the interface between the two.

The graphs here are fixtures. The point of every one of them is that the interface a coarse
computation declares and the work its leaves do name the *same* information node -- so the fixtures
build that identity explicitly, and the failing variants break it in exactly one way each.
"""

import pytest

from future_graph import (
    ComputationNode, InformationKind, InformationNode, Relation, StateGraph, build,
)
from future_graph.validation import validate


def comp(cid, description="Do the work", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description="the token", available=True, kind=InformationKind.FACT):
    return InformationNode(id=iid, kind=kind, description=description, available=available)


def refined():
    """c1 is refined into c2 and c3. It needs i1 from outside and will establish i2.

    i1 is required by c2 and i2 is produced by c3, so both halves of the interface are realized by
    the same nodes the coarse computation named.
    """
    return build(
        nodes=[comp("c1", "Gather the records that satisfy the request"),
               comp("c2", "Retrieve the first page"),
               comp("c3", "Continue until no page remains"),
               info("i1", "the confirmed listing interface"),
               info("i2", "the gathered records", available=False,
                    kind=InformationKind.RESULT)],
        edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
               ("i1", Relation.INTERFACE_INPUT, "c1"),
               ("c1", Relation.INTERFACE_OUTPUT, "i2"),
               ("i1", Relation.REQUIRES, "c2"), ("c3", Relation.PRODUCES, "i2"),
               ("c2", Relation.PRECEDES, "c3")])


def codes(graph):
    return sorted(v.code for v in validate(graph))


def test_a_refined_graph_with_a_realized_interface_is_sound():
    assert validate(refined()) == ()


# --------------------------------------------------------------------------- accessors

def test_the_hierarchy_is_read_off_the_edges():
    g = refined()
    assert g.refinement_children_of("c1") == ("c2", "c3")
    assert g.refinement_parents_of("c2") == ("c1",)
    assert g.refinement_parents_of("c1") == ()
    assert g.is_coarse("c1") and not g.is_leaf("c1")
    assert g.is_leaf("c2") and g.is_leaf("c3")
    assert g.descendant_leaves_of("c1") == ("c2", "c3")
    assert g.interface_inputs_of("c1") == ("i1",)
    assert g.interface_outputs_of("c1") == ("i2",)


def test_ancestors_are_found_through_more_than_one_level():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3")],
              edges=[("c1", Relation.REFINES, "c2"), ("c2", Relation.REFINES, "c3")])
    assert g.refinement_ancestors_of("c3") == ("c1", "c2")
    assert g.descendant_leaves_of("c1") == ("c3",)


def test_traversal_terminates_on_a_cyclic_candidate():
    """Validation runs on graphs that are wrong, including this one; it must not hang on them."""
    g = build(nodes=[comp("c1"), comp("c2")],
              edges=[("c1", Relation.REFINES, "c2"), ("c2", Relation.REFINES, "c1")])
    assert g.refinement_ancestors_of("c1") == ("c2",)
    assert g.descendant_leaves_of("c1") == ()
    assert "cycle" in codes(g)


def test_traversal_terminates_when_a_child_has_two_parents():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3")],
              edges=[("c1", Relation.REFINES, "c3"), ("c2", Relation.REFINES, "c3")])
    assert g.refinement_parents_of("c3") == ("c1", "c2")
    assert "multiple_refinement_parents" in codes(g)


def test_a_parent_may_have_many_children():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"), comp("c4")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
                     ("c1", Relation.REFINES, "c4")])
    assert g.refinement_children_of("c1") == ("c2", "c3", "c4")
    assert validate(g) == ()


# --------------------------------------------------------------------------- roles

def test_a_coarse_computation_may_not_carry_an_operation():
    g = refined()
    broken = build(nodes=[comp("c1", "Gather the records", operation="example.list_records"),
                          comp("c2"), info("i1")],
                   edges=[("c1", Relation.REFINES, "c2"), ("i1", Relation.INTERFACE_INPUT, "c1"),
                          ("i1", Relation.REQUIRES, "c2")])
    assert validate(g) == ()
    assert "coarse_is_executable" in codes(broken)


def test_a_coarse_computation_may_not_carry_arguments():
    g = build(nodes=[comp("c1", arguments={"page": 1}), comp("c2")],
              edges=[("c1", Relation.REFINES, "c2")])
    assert "coarse_is_executable" in codes(g)


@pytest.mark.parametrize("relation,source,target", [
    (Relation.REQUIRES, "i1", "c1"),
    (Relation.PRODUCES, "c1", "i1"),
])
def test_a_coarse_computation_may_not_use_operational_edges(relation, source, target):
    g = build(nodes=[comp("c1"), comp("c2"), info("i1")],
              edges=[("c1", Relation.REFINES, "c2"), (source, relation, target),
                     ("i1", Relation.REQUIRES, "c2")])
    assert "coarse_operational_edge" in codes(g)


@pytest.mark.parametrize("relation,source,target", [
    (Relation.INTERFACE_INPUT, "i1", "c1"),
    (Relation.INTERFACE_OUTPUT, "c1", "i1"),
])
def test_a_leaf_may_not_declare_an_interface(relation, source, target):
    g = build(nodes=[comp("c1"), info("i1")],
              edges=[(source, relation, target), ("i1", Relation.REQUIRES, "c1")])
    assert "leaf_interface_edge" in codes(g)


# --------------------------------------------------------------------------- realization

def test_a_coarse_input_no_descendant_requires_is_refused():
    g = build(nodes=[comp("c1"), comp("c2"), info("i1")],
              edges=[("c1", Relation.REFINES, "c2"), ("i1", Relation.INTERFACE_INPUT, "c1")])
    assert "unrealized_interface_input" in codes(g)


def test_a_coarse_input_realized_by_a_grandchild_is_accepted():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"), info("i1")],
              edges=[("c1", Relation.REFINES, "c2"), ("c2", Relation.REFINES, "c3"),
                     ("i1", Relation.INTERFACE_INPUT, "c1"), ("i1", Relation.REQUIRES, "c3")])
    assert validate(g) == ()


def test_a_coarse_input_required_only_outside_the_subtree_is_refused():
    """The consumer has to be underneath the computation that declared it."""
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"), info("i1")],
              edges=[("c1", Relation.REFINES, "c2"), ("i1", Relation.INTERFACE_INPUT, "c1"),
                     ("i1", Relation.REQUIRES, "c3")])
    assert "unrealized_interface_input" in codes(g)


def test_an_unavailable_coarse_output_needs_exactly_one_descendant_producer():
    assert validate(refined()) == ()


def test_an_unavailable_coarse_output_with_no_producer_is_refused():
    g = build(nodes=[comp("c1"), comp("c2"),
                     info("i2", "the gathered records", available=False,
                          kind=InformationKind.RESULT)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i2")])
    assert "unrealized_interface_output" in codes(g)


def test_an_unavailable_coarse_output_with_two_producers_is_refused():
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"),
                     info("i2", "the gathered records", available=False,
                          kind=InformationKind.RESULT)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
                     ("c1", Relation.INTERFACE_OUTPUT, "i2"),
                     ("c2", Relation.PRODUCES, "i2"), ("c3", Relation.PRODUCES, "i2")])
    assert "unrealized_interface_output" in codes(g)


def test_an_available_coarse_output_needs_no_producer():
    """Whatever established it has already left the graph; the output stays for what comes next."""
    g = build(nodes=[comp("c1"), comp("c2"), comp("c3"), info("i2", "the gathered records")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i2"),
                     ("i2", Relation.REQUIRES, "c3")])
    assert validate(g) == ()


def test_an_interface_output_is_not_counted_as_a_producer():
    """Availability still asks for exactly one PRODUCES edge, and the interface edge is not one."""
    g = build(nodes=[comp("c1"), comp("c2"),
                     info("i2", "the gathered records", available=False,
                          kind=InformationKind.RESULT)],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.INTERFACE_OUTPUT, "i2"),
                     ("c2", Relation.PRODUCES, "i2")])
    assert g.producers_of("i2") == ("c2",)
    assert validate(g) == ()


# --------------------------------------------------------------------------- shared information

def shared_across_branches():
    """One token, declared once by two coarse computations and consumed by a leaf under each."""
    return build(
        nodes=[comp("c1", "Correct the affected records"), comp("c2", "Update each record"),
               comp("c3", "Verify every correction"), comp("c4", "Re-read each record"),
               info("i1", "the session token")],
        edges=[("c1", Relation.REFINES, "c2"), ("c3", Relation.REFINES, "c4"),
               ("i1", Relation.INTERFACE_INPUT, "c1"), ("i1", Relation.INTERFACE_INPUT, "c3"),
               ("i1", Relation.REQUIRES, "c2"), ("i1", Relation.REQUIRES, "c4"),
               ("c1", Relation.PRECEDES, "c3")])


def test_shared_information_is_one_node_consumed_by_leaves_in_two_branches():
    g = shared_across_branches()
    assert validate(g) == ()
    assert [i.id for i in g.information] == ["i1"]
    assert g.consumers_of("i1") == ("c2", "c4")
    assert g.interface_inputs_of("c1") == ("i1",) and g.interface_inputs_of("c3") == ("i1",)


def test_every_refinement_fault_is_reported_not_just_the_first():
    g = build(nodes=[comp("c1", operation="example.run"), comp("c2"), comp("c3"),
                     info("i1"), info("i2", available=False, kind=InformationKind.RESULT)],
              edges=[("c1", Relation.REFINES, "c3"), ("c2", Relation.REFINES, "c3"),
                     ("i1", Relation.INTERFACE_INPUT, "c1"),
                     ("c1", Relation.INTERFACE_OUTPUT, "i2"),
                     ("i1", Relation.REQUIRES, "c1")])
    assert set(codes(g)) >= {"multiple_refinement_parents", "coarse_is_executable",
                             "coarse_operational_edge", "unrealized_interface_input",
                             "unrealized_interface_output"}


# --------------------------------------------------------------------------- atomicity

def test_a_candidate_refused_for_a_refinement_fault_changes_nothing():
    from future_graph.lifecycle import replace
    previous = refined()
    before = previous.to_snapshot()
    candidate = build(nodes=[comp("c1"), comp("c2"), info("i1")],
                      edges=[("c1", Relation.REFINES, "c2"),
                             ("i1", Relation.INTERFACE_INPUT, "c1")])
    result = replace(previous, candidate)
    assert result.rejected and result.graph is previous
    assert previous.to_snapshot() == before
    assert result.collected == ()


def test_an_accepted_refined_candidate_becomes_the_state():
    from future_graph.lifecycle import replace
    result = replace(StateGraph(), refined())
    assert result.accepted
    assert [c.id for c in result.graph.computations] == ["c1", "c2", "c3"]
