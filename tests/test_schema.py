"""Nodes, payloads, and the graph that holds them."""

import pytest

from future_graph import (
    ComputationNode, ContractPayload, EntityType, InformationKind, InformationNode,
    InformationReference, ListPayload, MappingPayload, Relation, RuntimeReferencePayload,
    ScalarPayload, SchemaError, StateGraph, build, is_serialized_container,
)


def comp(cid="c1", description="Obtain a usable Venmo access token", **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid="i1", kind=InformationKind.FACT, description="The confirmed recipient list",
         available=True, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


# --------------------------------------------------------------------------- nodes load

def test_a_computation_loads_with_only_a_description():
    c = comp()
    assert c.operation is None and dict(c.arguments) == {}


def test_a_computation_loads_with_an_operation_and_arguments():
    c = comp(operation="apis.venmo.login", arguments={"username": "a@b.c", "password": "x"})
    assert c.operation == "apis.venmo.login"
    assert c.arguments["username"] == "a@b.c"


def test_an_information_node_loads():
    i = info(kind=InformationKind.CONTRACT, payload=ContractPayload("apis.venmo.login",
                                                                   ("username", "password")))
    assert i.kind is InformationKind.CONTRACT
    assert i.payload.parameters == ("username", "password")


def test_ids_are_snapshot_local_and_shaped():
    with pytest.raises(SchemaError, match="c<number>"):
        comp(cid="n1")
    with pytest.raises(SchemaError, match="i<number>"):
        info(iid="c1")


def test_an_unknown_information_kind_is_rejected():
    with pytest.raises(SchemaError, match="not a known information kind"):
        InformationNode(id="i1", kind="contract", description="d", available=True)


def test_an_empty_description_is_rejected():
    with pytest.raises(SchemaError, match="non-empty"):
        comp(description="   ")


# --------------------------------------------------------------------------- payload flatness

def test_a_nested_mapping_payload_value_is_rejected():
    with pytest.raises(SchemaError, match="nested dict"):
        MappingPayload((("expenses", {"name": "Jeffrey"}),))


def test_a_nested_list_payload_value_is_rejected():
    with pytest.raises(SchemaError, match="nested list"):
        ListPayload(([1, 2],))


def test_a_stringified_dict_is_not_a_payload_value():
    blob = "{'key': 'access_token', 'value': 'eyJhbGciOiJIUzI1NiJ9'}"
    assert is_serialized_container(blob)
    with pytest.raises(SchemaError, match="stringified container"):
        ScalarPayload(blob)


def test_a_stringified_list_is_not_a_description():
    with pytest.raises(SchemaError, match="stringified container"):
        info(description="[{'name': 'Jeffrey Smith', 'amount': 50.0}]")


def test_ordinary_prose_with_a_brace_is_left_alone():
    assert not is_serialized_container("the amount {as shown on the receipt}")
    assert info(description="the amount {as shown on the receipt}")


def test_a_runtime_reference_stores_a_name_not_a_value():
    node = info(iid="i2", kind=InformationKind.RUNTIME_REFERENCE,
                description="Venmo access token bound by the agent",
                payload=RuntimeReferencePayload("venmo_access_token"))
    assert node.payload.name == "venmo_access_token"
    with pytest.raises(SchemaError, match="carries a name"):
        info(iid="i3", kind=InformationKind.RUNTIME_REFERENCE, description="d",
             payload=ScalarPayload("eyJhbGciOiJIUzI1NiJ9"))


def test_a_contract_keeps_the_exact_operation_and_parameter_names():
    payload = ContractPayload("apis.splitwise.record_payment", ("group_id", "amount"),
                              ("amount must be positive",))
    node = info(iid="i4", kind=InformationKind.CONTRACT, description="d", payload=payload)
    assert node.payload.operation == "apis.splitwise.record_payment"
    assert node.payload.parameters == ("group_id", "amount")
    assert node.payload.constraints == ("amount must be positive",)


def test_a_contract_node_will_not_take_some_other_payload():
    with pytest.raises(SchemaError, match="carries an interface"):
        info(iid="i5", kind=InformationKind.CONTRACT, description="d",
             payload=ScalarPayload("apis.venmo.login"))


def test_a_duplicate_mapping_key_is_rejected():
    with pytest.raises(SchemaError, match="duplicate key"):
        MappingPayload((("a", 1), ("a", 2)))


# --------------------------------------------------------------------------- arguments

def test_an_argument_may_reference_an_information_node():
    c = comp(arguments={"access_token": InformationReference("i2")})
    assert c.referenced_information == ("i2",)


def test_a_literal_argument_stays_literal():
    c = comp(arguments={"limit": 20})
    assert c.arguments["limit"] == 20 and c.referenced_information == ()


def test_an_argument_cannot_be_a_nested_structure():
    with pytest.raises(SchemaError, match="nested dict"):
        comp(arguments={"body": {"a": 1}})


def test_arguments_are_read_only_after_construction():
    c = comp(arguments={"a": 1})
    with pytest.raises(TypeError):
        c.arguments["a"] = 2


# --------------------------------------------------------------------------- the graph

def test_duplicate_ids_within_a_snapshot_are_rejected():
    g = StateGraph()
    g.add(comp())
    with pytest.raises(SchemaError, match="duplicate id"):
        g.add(comp(description="something else"))


def test_an_unknown_relation_is_rejected():
    g = StateGraph()
    g.add(comp())
    g.add(info())
    with pytest.raises(SchemaError, match="not a known relation"):
        g.add_edge("i1", "requires", "c1")


def test_accessors_read_the_relations_apart():
    g = build(
        nodes=[comp("c1"), comp("c2", description="Execute the remaining payments"),
               info("i1", description="Confirmed Venmo login interface"),
               info("i2", description="Venmo access token", available=False)],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2"), ("c1", Relation.PRECEDES, "c2")],
    )
    assert g.requires_of("c1") == ("i1",)
    assert g.produces_of("c1") == ("i2",)
    assert g.consumers_of("i2") == ("c2",)
    assert g.producers_of("i2") == ("c1",)
    assert g.predecessors_of("c2") == ("c1",)
    assert g.successors_of("c1") == ("c2",)
    assert g.kind_of("c1") is EntityType.COMPUTATION
    assert g.kind_of("i1") is EntityType.INFORMATION


def test_nodes_sort_numerically_not_lexically():
    g = build(nodes=[comp("c10"), comp("c2")])
    assert [c.id for c in g.computations] == ["c2", "c10"]


# --------------------------------------------------------------------------- artifacts

def test_a_snapshot_round_trips_exactly():
    g = build(
        nodes=[comp("c1", operation="apis.venmo.login",
                    arguments={"username": "a@b.c", "token": InformationReference("i2")}),
               info("i1", kind=InformationKind.CONTRACT, description="Venmo login",
                    payload=ContractPayload("apis.venmo.login", ("username",))),
               info("i2", kind=InformationKind.RUNTIME_REFERENCE, description="token",
                    available=False, payload=RuntimeReferencePayload("venmo_access_token"))],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2")],
    )
    snapshot = g.to_snapshot()
    again = StateGraph.from_snapshot(snapshot)
    assert again.to_snapshot() == snapshot
    assert again == g


def test_a_snapshot_is_stable_regardless_of_insertion_order():
    a = build(nodes=[comp("c1"), comp("c2")], edges=[("c1", Relation.PRECEDES, "c2")])
    b = StateGraph()
    b.add(comp("c2"))
    b.add(comp("c1"))
    b.add_edge("c1", Relation.PRECEDES, "c2")
    assert a.to_snapshot() == b.to_snapshot()


def test_every_payload_kind_survives_the_round_trip():
    g = build(nodes=[
        info("i1", payload=ScalarPayload(3084)),
        info("i2", payload=ListPayload((285, 291, 306))),
        info("i3", payload=MappingPayload((("group_id", 7), ("amount", 50.0)))),
        info("i4", kind=InformationKind.RUNTIME_REFERENCE, payload=RuntimeReferencePayload("tok")),
        info("i5", kind=InformationKind.CONTRACT, payload=ContractPayload("apis.x.y", ("a",))),
    ])
    assert StateGraph.from_snapshot(g.to_snapshot()) == g


def test_a_snapshot_with_an_unknown_relation_is_rejected():
    with pytest.raises(SchemaError, match="unknown relation"):
        StateGraph.from_snapshot({"computations": [{"id": "c1", "description": "d"}],
                                  "information": [],
                                  "edges": [{"source": "c1", "relation": "refines",
                                             "target": "c1"}]})


def test_the_graph_carries_no_state_beside_itself():
    """Contracts and references are nodes, so a rebuilt graph knows everything the first one did."""
    g = build(nodes=[comp("c1"),
                     info("i1", kind=InformationKind.CONTRACT, description="Venmo login",
                          payload=ContractPayload("apis.venmo.login", ("username",))),
                     info("i2", kind=InformationKind.RUNTIME_REFERENCE, description="token",
                          payload=RuntimeReferencePayload("venmo_access_token"))],
              edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1")])
    rebuilt = StateGraph.from_snapshot(g.to_snapshot())
    contracts = [i.payload.operation for i in rebuilt.information
                 if i.kind is InformationKind.CONTRACT]
    references = [i.payload.name for i in rebuilt.information
                  if i.kind is InformationKind.RUNTIME_REFERENCE]
    assert contracts == ["apis.venmo.login"] and references == ["venmo_access_token"]
