"""Invariants, and the fact that a candidate is told about all of its faults at once."""

from future_graph import (
    ComputationNode, InformationKind, InformationNode, InformationReference, Relation, StateGraph,
    build,
)
from future_graph.validation import unconsumed_information, validate


def comp(cid="c1", description="Obtain a usable Venmo access token", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid="i1", kind=InformationKind.FACT, description="Confirmed recipients",
         available=True, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


def codes(graph):
    return sorted(v.code for v in validate(graph))


def test_a_sound_graph_has_nothing_to_say():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments"),
                     info("i1"), info("i2", description="token", available=False)],
              edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
                     ("i2", Relation.REQUIRES, "c2"), ("c1", Relation.PRECEDES, "c2")])
    assert validate(g) == ()


# --------------------------------------------------------------------------- endpoints

def test_requires_between_two_computations_is_rejected():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments")],
              edges=[("c1", Relation.REQUIRES, "c2")])
    assert "endpoint_type" in codes(g)


def test_precedes_from_information_is_rejected():
    g = build(nodes=[comp("c1"), info("i1")],
              edges=[("i1", Relation.PRECEDES, "c1"), ("i1", Relation.REQUIRES, "c1")])
    assert "endpoint_type" in codes(g)


def test_produces_into_a_computation_is_rejected():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments")],
              edges=[("c1", Relation.PRODUCES, "c2")])
    assert "endpoint_type" in codes(g)


# --------------------------------------------------------------------------- structure

def test_a_cycle_is_rejected():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments")],
              edges=[("c1", Relation.PRECEDES, "c2"), ("c2", Relation.PRECEDES, "c1")])
    assert "cycle" in codes(g)


def test_a_cycle_through_information_is_rejected():
    g = build(nodes=[comp("c1"), info("i1", available=False)],
              edges=[("c1", Relation.PRODUCES, "i1"), ("i1", Relation.REQUIRES, "c1")])
    assert "cycle" in codes(g)


def test_an_edge_to_an_undeclared_id_is_dangling_not_a_phantom_node():
    g = build(nodes=[comp("c1")], edges=[("i9", Relation.REQUIRES, "c1")])
    assert "dangling_edge" in codes(g)
    assert [c.id for c in g.computations] == ["c1"] and g.information == ()


# --------------------------------------------------------------------------- availability

def test_unavailable_information_needs_exactly_one_producer():
    g = build(nodes=[comp("c1"), info("i1", available=False)],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert "availability" in codes(g)


def test_unavailable_information_with_two_producers_is_rejected():
    g = build(nodes=[comp("c1"), comp("c2", description="Execute the payments"),
                     info("i1", available=False)],
              edges=[("c1", Relation.PRODUCES, "i1"), ("c2", Relation.PRODUCES, "i1"),
                     ("i1", Relation.REQUIRES, "c1")])
    assert "availability" in codes(g)


def test_available_information_may_have_no_producer():
    g = build(nodes=[comp("c1"), info("i1", available=True)],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert validate(g) == ()


# --------------------------------------------------------------------------- argument references

def test_an_argument_naming_a_missing_information_node_is_rejected():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i7")})])
    assert "unknown_argument_reference" in codes(g)


def test_an_argument_reference_without_a_requires_edge_is_rejected_not_repaired():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i1")}),
                     info("i1", description="token"), comp("c2", description="Use the token")],
              edges=[("i1", Relation.REQUIRES, "c2")])
    assert "unlinked_argument_reference" in codes(g)
    assert g.requires_of("c1") == ()          # nothing was added on the way past


def test_an_argument_reference_with_its_edge_passes():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i1")}),
                     info("i1", description="token")],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert validate(g) == ()


def test_a_literal_argument_needs_no_edge():
    g = build(nodes=[comp("c1", arguments={"limit": 20})])
    assert validate(g) == ()


# --------------------------------------------------------------------------- all at once

def test_every_fault_is_reported_not_just_the_first():
    g = build(nodes=[comp("c1", arguments={"token": InformationReference("i7")}),
                     comp("c2", description="Execute the payments"),
                     info("i1", available=False)],
              edges=[("c1", Relation.PRECEDES, "c2"), ("c2", Relation.PRECEDES, "c1"),
                     ("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c2")])
    assert set(codes(g)) == {"cycle", "availability", "unknown_argument_reference",
                             "dangling_edge"}


def test_a_dangling_id_of_any_shape_is_a_violation_and_not_an_exception():
    """Sorting must never be the thing that raises: the id in an edge can be anything at all."""
    g = build(nodes=[comp("c1")], edges=[("not-an-id", Relation.REQUIRES, "c1"),
                                         ("c1", Relation.PRECEDES, "")])
    assert codes(g).count("dangling_edge") == 2


def test_violations_read_as_sentences():
    g = build(nodes=[comp("c1")], edges=[("i9", Relation.REQUIRES, "c1")])
    assert "i9" in str(validate(g)[0])


# --------------------------------------------------------------------------- liveness is not a fault

def test_information_without_a_consumer_is_not_a_validation_failure():
    """It is garbage, and lifecycle collects it. Refusing the whole graph over it would be wrong."""
    g = build(nodes=[comp("c1"), info("i1")])
    assert validate(g) == ()
    assert unconsumed_information(g) == ("i1",)


def test_there_is_no_way_to_express_a_completed_computation():
    """The old graph needed a rule against historical nodes; here there is no field to put one in."""
    assert not hasattr(comp(), "status")
    assert not any(f in ComputationNode.__dataclass_fields__
                   for f in ("status", "done", "parent_id", "acquired_information"))
