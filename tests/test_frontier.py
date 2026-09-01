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
    """Obtain a token, then pay: the token is a result nothing has established yet.

    It is not a runtime reference. A reference names something the agent bound, and nothing has been
    bound until the login runs; it becomes one in the snapshot after that.
    """
    return build(
        nodes=[comp("c1", "Obtain a usable Venmo access token"),
               comp("c2", "Execute the remaining payments"),
               info("i1", "Confirmed Venmo login interface", available=True),
               info("i2", "A usable access token", available=False,
                    kind=InformationKind.RESULT)],
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


# --------------------------------------------------------------------------- refinement

def refined():
    """c1 is refined into c2 then c3; c4 comes after the whole of c1."""
    return build(
        nodes=[comp("c1", "Gather the records"), comp("c2", "Retrieve the first page"),
               comp("c3", "Continue until no page remains"), comp("c4", "Apply the change"),
               info("i1", "the confirmed listing interface")],
        edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
               ("i1", Relation.INTERFACE_INPUT, "c1"), ("i1", Relation.REQUIRES, "c2"),
               ("c2", Relation.PRECEDES, "c3"), ("c1", Relation.PRECEDES, "c4")])


def test_a_refined_computation_is_never_on_the_frontier():
    g = refined()
    assert not is_executable(g, "c1")
    assert "c1" not in [c.id for c in frontier(g)]


def test_a_satisfied_leaf_is_on_the_frontier():
    assert [c.id for c in frontier(refined())] == ["c2"]


def test_a_coarse_predecessor_blocks_every_descendant_leaf():
    """c4 waits for the whole of c1, so refining c1 must not make c4 runnable."""
    g = build(nodes=[comp("c1", "Gather"), comp("c2", "Retrieve"), comp("c4", "Apply")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.PRECEDES, "c4")])
    assert [c.id for c in frontier(g)] == ["c2"]
    assert not is_executable(g, "c4")


def test_ordering_reaches_a_leaf_through_its_ancestors():
    """The blocker is on the parent, and the child inherits it rather than escaping it."""
    g = build(nodes=[comp("c1", "First"), comp("c2", "Second"), comp("c3", "A child of c2")],
              edges=[("c1", Relation.PRECEDES, "c2"), ("c2", Relation.REFINES, "c3")])
    assert not is_executable(g, "c3")
    assert {b.computation_id: b for b in blockers(g)}["c3"].waiting_for_computations == ("c1",)


def test_information_requirements_do_not_inherit_from_an_ancestor():
    """c2 does not need what c1 declared, so c2 runs while c3 waits for it."""
    g = build(nodes=[comp("c1", "Gather"), comp("c2", "Retrieve the first page"),
                     comp("c3", "Summarize"),
                     info("i1", "the summary template", available=False,
                          kind=InformationKind.RESULT),
                     comp("c4", "Draft the template")],
              edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
                     ("i1", Relation.INTERFACE_INPUT, "c1"), ("i1", Relation.REQUIRES, "c3"),
                     ("c4", Relation.PRODUCES, "i1")])
    assert [c.id for c in frontier(g)] == ["c2", "c4"]
    assert not is_executable(g, "c3")


def test_an_unrefined_computation_with_no_operation_is_still_executable():
    """Working out how to do something is work, so a coarse-sounding leaf reaches the frontier."""
    g = build(nodes=[comp("c1", "Report the outcome")])
    assert g.is_leaf("c1") and not g.is_coarse("c1")
    assert ComputationNode(id="c1", description="Report the outcome").operation is None
    assert [c.id for c in frontier(g)] == ["c1"]


def test_a_coarse_computation_is_not_reported_as_blocked():
    g = refined()
    assert "c1" not in [b.computation_id for b in blockers(g)]
    assert sorted(b.computation_id for b in blockers(g)) == ["c3", "c4"]


def test_a_refined_computation_is_ordered_before_its_children():
    g = refined()
    order = [c.id for c in ordered_computations(g)]
    assert order.index("c1") < order.index("c2") < order.index("c3")
