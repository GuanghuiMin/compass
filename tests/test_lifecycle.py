"""Dead information goes; a bad candidate changes nothing at all."""

from future_graph import (
    ComputationNode, InformationKind, InformationNode, InformationReference, Relation, StateGraph,
    build,
)
from future_graph.lifecycle import collect_dead_information, replace


def comp(cid="c1", description="Obtain a usable Venmo access token", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid="i1", kind=InformationKind.FACT, description="Confirmed recipients",
         available=True, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


def sound():
    return build(nodes=[comp("c1"), info("i1")], edges=[("i1", Relation.REQUIRES, "c1")])


# --------------------------------------------------------------------------- collection

def test_information_with_no_consumer_is_collected():
    g = build(nodes=[comp("c1"), info("i1"), info("i2", description="an old note")],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert collect_dead_information(g) == ("i2",)
    assert [i.id for i in g.information] == ["i1"]


def test_information_shared_by_two_computations_survives():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments"), info("i1")],
              edges=[("i1", Relation.REQUIRES, "c1"), ("i1", Relation.REQUIRES, "c2")])
    assert collect_dead_information(g) == ()
    assert [i.id for i in g.information] == ["i1"]


def test_information_goes_once_its_last_consumer_goes():
    g = build(nodes=[comp("c1"), info("i1")], edges=[("i1", Relation.REQUIRES, "c1")])
    g.remove("c1")
    assert collect_dead_information(g) == ("i1",)
    assert g.information == ()


def test_produced_information_survives_without_its_producer():
    """A result outlives the computation that made it, which is how the past collapses."""
    g = build(nodes=[comp("c2", description="Execute the payments"), info("i1", available=True)],
              edges=[("i1", Relation.REQUIRES, "c2")])
    assert collect_dead_information(g) == ()
    assert [i.id for i in g.information] == ["i1"]


def test_collection_is_a_fixpoint():
    g = build(nodes=[comp("c1"), info("i1")], edges=[("i1", Relation.REQUIRES, "c1")])
    assert collect_dead_information(g) == ()
    assert collect_dead_information(g) == ()


def test_nothing_is_collected_on_the_grounds_that_it_used_to_matter():
    g = build(nodes=[comp("c1"), info("i1", kind=InformationKind.CONTRACT,
                                      description="Venmo login")],
              edges=[("i1", Relation.REQUIRES, "c1")])
    collect_dead_information(g)
    assert [i.id for i in g.information] == ["i1"]


# --------------------------------------------------------------------------- replacement

def test_a_sound_candidate_becomes_the_state():
    previous, candidate = sound(), build(
        nodes=[comp("c1", description="Execute the remaining payments"), info("i1")],
        edges=[("i1", Relation.REQUIRES, "c1")])
    result = replace(previous, candidate)
    assert result.accepted and result.graph is candidate


def test_a_candidate_with_a_fault_leaves_the_previous_graph_identical():
    previous = sound()
    before = previous.to_snapshot()
    candidate = build(nodes=[comp("c1"), comp("c2", description="Execute the payments")],
                      edges=[("c1", Relation.PRECEDES, "c2"), ("c2", Relation.PRECEDES, "c1")])
    result = replace(previous, candidate)
    assert result.rejected and result.graph is previous
    assert previous.to_snapshot() == before


def test_a_rejected_candidate_collects_nothing():
    """Deleting on account of a graph that turned out to be invalid is how a contract gets lost."""
    previous = sound()
    candidate = build(nodes=[comp("c1"), info("i1"), info("i2", description="unconsumed")],
                      edges=[("i1", Relation.REQUIRES, "c1"), ("i9", Relation.REQUIRES, "c1")])
    result = replace(previous, candidate)
    assert result.rejected and result.collected == ()
    assert [i.id for i in candidate.information] == ["i1", "i2"]


def test_a_rejection_reports_every_reason():
    previous = sound()
    candidate = build(nodes=[comp("c1", arguments={"t": InformationReference("i7")}),
                             info("i1", available=False)],
                      edges=[("i1", Relation.REQUIRES, "c1")])
    result = replace(previous, candidate)
    assert {v.code for v in result.violations} == {"availability", "unknown_argument_reference"}


def test_collection_happens_on_the_accepted_candidate():
    previous = sound()
    candidate = build(nodes=[comp("c1"), info("i1"), info("i2", description="nobody wants this")],
                      edges=[("i1", Relation.REQUIRES, "c1")])
    result = replace(previous, candidate)
    assert result.accepted and result.collected == ("i2",)
    assert [i.id for i in result.graph.information] == ["i1"]


def test_an_empty_candidate_is_a_valid_end_state():
    result = replace(sound(), StateGraph())
    assert result.accepted and len(result.graph) == 0
