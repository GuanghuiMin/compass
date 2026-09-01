"""The `REQUIRES` edge an argument reference already implies.

`argument curator_token = @i2` names the information, names the computation consuming it, and fixes
the relation. There is nothing left to decide, so the code writes the edge and the model does not
have to write it twice.

What this must not become is a repair. It adds the one edge the reference determines, removes
nothing, and touches no other requirement -- including one the model put on the refined computation
above, which may be a misplacement or may be obligation-level knowledge, and telling those apart is
a judgement about meaning.
"""

from future_graph import (
    ComputationNode, InformationKind, InformationNode, InformationReference, Relation, build,
)
from future_graph.arguments import complete_argument_dependencies
from future_graph.lifecycle import replace
from future_graph.state_graph import StateGraph
from future_graph.validation import validate


def comp(cid, description="Do the work", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description="the token", kind=InformationKind.FACT, available=True):
    return InformationNode(id=iid, kind=kind, description=description, available=available)


def requires(graph):
    return sorted((e.source, e.target) for e in graph.edges_of(Relation.REQUIRES))


def test_the_edge_an_argument_reference_implies_is_added():
    g = build(nodes=[comp("c1", "Open an entry", operation="example.open_entry",
                          arguments={"token": InformationReference("i1")}),
                     info("i1")])
    assert "unlinked_argument_reference" in [v.code for v in validate(g)]
    completed, changes = complete_argument_dependencies(g)
    assert requires(completed) == [("i1", "c1")]
    assert [(c.action, c.source, c.relation.value, c.target) for c in changes] == [
        ("added", "i1", "requires", "c1")]
    assert validate(completed) == ()


def test_an_edge_the_model_already_wrote_is_not_reported_as_a_change():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i1")}), info("i1")],
              edges=[("i1", Relation.REQUIRES, "c1")])
    completed, changes = complete_argument_dependencies(g)
    assert requires(completed) == [("i1", "c1")]
    assert changes == ()


def test_nothing_is_removed_including_a_requirement_one_level_up():
    """synthetic_11: the token was required at the obligation and referenced in the leaf's call.

    Adding the leaf edge is determined. Deleting the parent's would mean deciding the obligation is
    not really governed by the token, which is exactly the judgement the code may not make.
    """
    g = build(nodes=[comp("c1", "Register every seedling"),
                     comp("c2", "Open one entry", operation="example.open_entry",
                          arguments={"curator_token": InformationReference("i2")}),
                     info("i2", "the curator token")],
              edges=[("c1", Relation.REFINES, "c2"), ("i2", Relation.REQUIRES, "c1")])
    completed, changes = complete_argument_dependencies(g)
    assert requires(completed) == [("i2", "c1"), ("i2", "c2")]
    assert [(c.source, c.target) for c in changes] == [("i2", "c2")]


def test_an_unrelated_requirement_the_model_wrote_is_left_alone():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i1")}),
                     info("i1"), info("i2", "something else it needs")],
              edges=[("i2", Relation.REQUIRES, "c1")])
    completed, _ = complete_argument_dependencies(g)
    assert requires(completed) == [("i1", "c1"), ("i2", "c1")]


def test_a_reference_naming_nothing_is_left_for_validation():
    """Adding an edge would mean inventing the node the reference got wrong."""
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i9")})])
    completed, changes = complete_argument_dependencies(g)
    assert changes == () and requires(completed) == []
    assert "unknown_argument_reference" in [v.code for v in validate(completed)]


def test_completion_does_not_touch_the_candidate():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i1")}), info("i1")])
    before = g.to_snapshot()
    complete_argument_dependencies(g)
    assert g.to_snapshot() == before


def test_two_arguments_naming_one_node_add_one_edge():
    g = build(nodes=[comp("c1", arguments={"a": InformationReference("i1"),
                                           "b": InformationReference("i1")}), info("i1")])
    _, changes = complete_argument_dependencies(g)
    assert len(changes) == 1


# --------------------------------------------------------------------------- ordering

def test_the_derived_edge_can_be_what_makes_information_cross_a_boundary():
    """Why arguments are completed before interfaces.

    Nothing here writes `i1 REQUIRES c2` or any interface edge. The argument reference implies the
    first; the first is what makes i1 cross c1's boundary. Deriving interfaces earlier would leave
    the boundary incomplete, and the graph would be refused for a gap it does not have.
    """
    candidate = build(
        nodes=[comp("c1", "The obligation"),
               comp("c2", "The call", operation="example.call",
                    arguments={"token": InformationReference("i1")}),
               info("i1", "the token")],
        edges=[("c1", Relation.REFINES, "c2")])
    result = replace(StateGraph(), candidate)
    assert result.accepted, [str(v) for v in result.violations]
    assert requires(result.graph) == [("i1", "c2")]
    assert result.graph.interface_inputs_of("c1") == ("i1",)
    assert [(c.action, c.source, c.target) for c in result.argument_dependency_changes] == [
        ("added", "i1", "c2")]
    assert [(c.action, c.source, c.target) for c in result.interface_changes] == [
        ("added", "i1", "c1")]


def test_the_two_completions_are_recorded_apart():
    """An argument dependency is not an interface, and the record does not blur them."""
    candidate = build(
        nodes=[comp("c1", "The obligation"),
               comp("c2", "The call", arguments={"token": InformationReference("i1")}),
               info("i1", "the token")],
        edges=[("c1", Relation.REFINES, "c2")])
    result = replace(StateGraph(), candidate)
    assert all(c.relation is Relation.REQUIRES for c in result.argument_dependency_changes)
    assert all(c.relation is Relation.INTERFACE_INPUT for c in result.interface_changes)


def test_this_step_never_removes_anything():
    """Asserted on the shape of what it reports, so a later edit cannot quietly make it destructive."""
    candidate = build(
        nodes=[comp("c1", arguments={"a": InformationReference("i1")}), info("i1"),
               info("i2", "wanted by nobody")],
        edges=[("i2", Relation.REQUIRES, "c1")])
    _, changes = complete_argument_dependencies(candidate)
    assert {c.action for c in changes} == {"added"}
