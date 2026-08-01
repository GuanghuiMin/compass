"""Executability is read off the graph, never stored."""

from future_graph import (
    ComputationNode, InformationKind, InformationNode, Relation, build,
)
from future_graph.frontier import blockers, frontier, is_executable, ordered_computations


def comp(cid, description, **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description, available=True, kind=InformationKind.FACT):
    return InformationNode(id=iid, kind=kind, description=description, available=available)


def chain():
    """login -> token -> pay, with the token not yet in hand."""
    return build(
        nodes=[comp("c1", "Obtain a usable Venmo access token"),
               comp("c2", "Execute the remaining payments"),
               info("i1", "Confirmed Venmo login interface", available=True),
               info("i2", "Venmo access token", available=False)],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2")])


def test_a_computation_with_nothing_before_it_and_everything_available_is_on_the_frontier():
    assert [c.id for c in frontier(chain())] == ["c1"]


def test_a_computation_waiting_on_another_computation_is_not():
    g = build(nodes=[comp("c1", "Find the payments"), comp("c2", "Execute them")],
              edges=[("c1", Relation.PRECEDES, "c2")])
    assert [c.id for c in frontier(g)] == ["c1"]
    assert not is_executable(g, "c2")


def test_a_computation_waiting_on_unavailable_information_is_not():
    assert not is_executable(chain(), "c2")


def test_the_producer_of_what_a_computation_waits_for_is_upstream():
    g = chain()
    waiting = {b.computation_id: b for b in blockers(g)}["c2"]
    assert waiting.waiting_for_information == ("i2",)
    assert g.producers_of("i2") == ("c1",)


def test_information_becoming_available_moves_the_frontier():
    g = build(nodes=[comp("c1", "Obtain a token"), comp("c2", "Execute the payments"),
                     info("i2", "Venmo access token", available=True)],
              edges=[("i2", Relation.REQUIRES, "c2")])
    assert [c.id for c in frontier(g)] == ["c1", "c2"]


def test_several_independent_computations_are_all_on_the_frontier():
    g = build(nodes=[comp("c1", "Log in to Venmo"), comp("c2", "Log in to Splitwise")])
    assert [c.id for c in frontier(g)] == ["c1", "c2"]


def test_executability_is_derived_and_not_a_field():
    assert not any(f in ComputationNode.__dataclass_fields__
                   for f in ("status", "ready", "executable", "blocked"))


def test_ordering_follows_dependencies_and_breaks_ties_by_id():
    g = build(nodes=[comp("c1", "Third"), comp("c2", "First"), comp("c3", "Second")],
              edges=[("c2", Relation.PRECEDES, "c3"), ("c3", Relation.PRECEDES, "c1")])
    assert [c.id for c in ordered_computations(g)] == ["c2", "c3", "c1"]


def test_produced_information_orders_a_producer_before_its_consumer():
    assert [c.id for c in ordered_computations(chain())] == ["c1", "c2"]


def test_an_empty_graph_has_an_empty_frontier():
    assert frontier(build()) == ()
