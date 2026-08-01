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


@pytest.mark.parametrize("bad", ["c²", "c١", "c", "c1a", "cⅣ", "c1.0", "c-1"])
def test_an_id_whose_number_is_not_ascii_decimal_is_rejected(bad):
    """str.isdigit() accepts '²', which int() will not read, and the sort would then raise."""
    with pytest.raises(SchemaError, match="c<number>"):
        comp(cid=bad)


def test_ids_that_read_as_the_same_number_still_sort_deterministically():
    """c1 and c01 are both legal and both read as one; order must not come from insertion."""
    a = StateGraph()
    a.add(comp("c1", description="first written"))
    a.add(comp("c01", description="second written"))
    b = StateGraph()
    b.add(comp("c01", description="second written"))
    b.add(comp("c1", description="first written"))
    assert a.to_snapshot() == b.to_snapshot()
    assert [c.id for c in a.computations] == [c.id for c in b.computations]


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
    with pytest.raises(SchemaError, match="carries a RuntimeReferencePayload"):
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
    with pytest.raises(SchemaError, match="carries a ContractPayload"):
        info(iid="i5", kind=InformationKind.CONTRACT, description="d",
             payload=ScalarPayload("apis.venmo.login"))


# --------------------------------------------------------------------------- kind and payload

@pytest.mark.parametrize("kind,payload", [
    (InformationKind.CONTRACT, ContractPayload("apis.a.b", ("x",))),
    (InformationKind.RUNTIME_REFERENCE, RuntimeReferencePayload("some_name")),
    (InformationKind.FACT, ScalarPayload(7)),
    (InformationKind.RESULT, ListPayload((1, 2))),
    (InformationKind.CONSTRAINT, MappingPayload((("limit", 5),))),
    (InformationKind.FAILURE_CONSEQUENCE, None),
])
def test_legal_available_pairings_load(kind, payload):
    assert info(kind=kind, available=True, payload=payload)


@pytest.mark.parametrize("kind", [InformationKind.CONTRACT, InformationKind.RUNTIME_REFERENCE])
def test_an_available_typed_kind_must_carry_its_payload(kind):
    with pytest.raises(SchemaError, match="carries a"):
        info(kind=kind, available=True, payload=None)


@pytest.mark.parametrize("kind", [InformationKind.CONTRACT, InformationKind.RUNTIME_REFERENCE])
def test_contract_and_runtime_reference_cannot_be_unavailable(kind):
    """What a computation will establish is a result until it is established."""
    with pytest.raises(SchemaError, match="cannot be unavailable"):
        info(kind=kind, available=False, payload=None)


@pytest.mark.parametrize("kind", [InformationKind.FACT, InformationKind.RESULT,
                                  InformationKind.CONSTRAINT,
                                  InformationKind.FAILURE_CONSEQUENCE])
def test_a_typed_payload_may_not_sit_on_an_untyped_kind(kind):
    with pytest.raises(SchemaError, match="belongs to contract"):
        info(kind=kind, available=True, payload=ContractPayload("apis.a.b"))
    with pytest.raises(SchemaError, match="belongs to runtime_reference"):
        info(kind=kind, available=True, payload=RuntimeReferencePayload("n"))


@pytest.mark.parametrize("payload", [ScalarPayload(1), ListPayload((1,)),
                                     MappingPayload((("a", 1),))])
def test_unavailable_information_carries_no_payload(payload):
    with pytest.raises(SchemaError, match="has no payload"):
        info(kind=InformationKind.RESULT, available=False, payload=payload)


def test_unavailable_information_without_a_payload_is_the_way_to_say_it_is_coming():
    node = info(kind=InformationKind.RESULT, available=False, payload=None,
                description="a usable access token")
    assert node.available is False and node.payload is None


# --------------------------------------------------------------------------- single-line text

@pytest.mark.parametrize("text", ["two\nlines", "carriage\rreturn"])
def test_multi_line_text_is_rejected_everywhere_the_protocol_must_carry_it(text):
    with pytest.raises(SchemaError, match="more than one line"):
        info(description=text)
    with pytest.raises(SchemaError, match="more than one line"):
        comp(description=text)
    with pytest.raises(SchemaError, match="more than one line"):
        comp(operation=text)
    with pytest.raises(SchemaError, match="more than one line"):
        ScalarPayload(text)
    with pytest.raises(SchemaError, match="more than one line"):
        RuntimeReferencePayload(text)
    with pytest.raises(SchemaError, match="more than one line"):
        ContractPayload(text)
    with pytest.raises(SchemaError, match="more than one line"):
        ContractPayload("apis.a.b", (text,))


def test_an_argument_name_may_not_contain_an_equals_sign_or_a_line_break():
    with pytest.raises(SchemaError, match="separates a name from its value"):
        comp(arguments={"a=b": 1})
    with pytest.raises(SchemaError, match="more than one line"):
        comp(arguments={"a\nb": 1})


@pytest.mark.parametrize("char", ["\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85",
                                  "\u2028", "\u2029"])
def test_every_character_that_splits_a_line_is_rejected(char):
    """str.splitlines() breaks on all of these, so the protocol would too."""
    assert len(f"a{char}b".splitlines()) == 2
    with pytest.raises(SchemaError, match="more than one line"):
        info(description=f"a{char}b")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_rejected(value):
    """nan is not equal to itself, so a graph holding one cannot round-trip through anything."""
    with pytest.raises(SchemaError, match="not a finite number"):
        ScalarPayload(value)
    with pytest.raises(SchemaError, match="not a finite number"):
        comp(arguments={"a": value})


# --------------------------------------------------------------------------- normalization sticks

def test_a_reference_keeps_the_id_the_check_normalized():
    assert InformationReference(" i1 ").information_id == "i1"


def test_a_mapping_keeps_the_key_the_check_normalized():
    assert MappingPayload(((" a ", 1),)).values == (("a", 1),)


def test_two_keys_that_normalize_alike_are_a_duplicate():
    with pytest.raises(SchemaError, match="duplicate key"):
        MappingPayload((("a", 1), (" a ", 2)))


def test_two_argument_names_that_normalize_alike_are_a_duplicate():
    with pytest.raises(SchemaError, match="is given twice"):
        comp(arguments={"a": 1, " a ": 2})


def test_contract_text_is_stored_stripped():
    payload = ContractPayload("  apis.a.b  ", ("  x  ",), ("  y  ",))
    assert payload.operation == "apis.a.b"
    assert payload.parameters == ("x",) and payload.constraints == ("y",)


def test_a_mapping_key_may_not_contain_an_equals_sign_or_a_line_break():
    with pytest.raises(SchemaError, match="separates a name from its value"):
        MappingPayload((("a=b", 1),))
    with pytest.raises(SchemaError, match="more than one line"):
        MappingPayload((("a\nb", 1),))


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
               info("i2", kind=InformationKind.RESULT, description="a usable access token",
                    available=False)],
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


def snapshot(**overrides):
    base = {
        "computations": [{"id": "c1", "description": "d", "operation": None, "arguments": {}}],
        "information": [{"id": "i1", "kind": "fact", "description": "d", "available": True,
                         "payload": None}],
        "edges": [{"source": "i1", "relation": "requires", "target": "c1"}],
    }
    base.update(overrides)
    return base


def test_a_snapshot_with_an_unknown_relation_is_rejected():
    with pytest.raises(SchemaError, match="unknown relation"):
        StateGraph.from_snapshot(snapshot(edges=[{"source": "i1", "relation": "refines",
                                                  "target": "c1"}]))


def test_a_well_formed_snapshot_loads():
    assert len(StateGraph.from_snapshot(snapshot())) == 2


@pytest.mark.parametrize("section", ["computations", "information", "edges"])
def test_a_missing_section_raises_rather_than_reading_as_empty(section):
    payload = snapshot()
    del payload[section]
    with pytest.raises(SchemaError, match=f"missing {section}"):
        StateGraph.from_snapshot(payload)


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(SchemaError, match="unknown notes"):
        StateGraph.from_snapshot(snapshot(notes="anything"))


@pytest.mark.parametrize("section", ["computations", "information", "edges"])
def test_a_section_that_is_not_a_list_is_rejected(section):
    with pytest.raises(SchemaError, match="expected a list"):
        StateGraph.from_snapshot(snapshot(**{section: {"id": "c1"}}))


def test_available_as_the_string_false_is_rejected_rather_than_read_as_true():
    """bool("false") is True, and that is how an artifact starts lying."""
    with pytest.raises(SchemaError, match="expected true or false"):
        StateGraph.from_snapshot(snapshot(information=[
            {"id": "i1", "kind": "fact", "description": "d", "available": "false",
             "payload": None}]))


def test_a_node_missing_a_key_is_rejected():
    with pytest.raises(SchemaError, match="missing operation"):
        StateGraph.from_snapshot(snapshot(computations=[{"id": "c1", "description": "d",
                                                         "arguments": {}}]))


def test_a_node_with_an_unknown_key_is_rejected():
    with pytest.raises(SchemaError, match="unknown status"):
        StateGraph.from_snapshot(snapshot(computations=[
            {"id": "c1", "description": "d", "operation": None, "arguments": {},
             "status": "pending"}]))


@pytest.mark.parametrize("argument", [
    {"literal": 1, "reference": "i1"}, {}, {"value": 1}, "i1", None,
])
def test_a_malformed_argument_encoding_is_rejected(argument):
    with pytest.raises(SchemaError, match="exactly one of"):
        StateGraph.from_snapshot(snapshot(computations=[
            {"id": "c1", "description": "d", "operation": None, "arguments": {"a": argument}}]))


def test_a_payload_with_an_unknown_type_is_rejected():
    with pytest.raises(SchemaError, match="unknown payload type"):
        StateGraph.from_snapshot(snapshot(information=[
            {"id": "i1", "kind": "fact", "description": "d", "available": True,
             "payload": {"type": "blob", "value": 1}}]))


def test_a_payload_missing_a_key_is_rejected():
    with pytest.raises(SchemaError, match="missing constraints"):
        StateGraph.from_snapshot(snapshot(information=[
            {"id": "i1", "kind": "contract", "description": "d", "available": True,
             "payload": {"type": "contract", "operation": "apis.a.b", "parameters": []}}]))


@pytest.mark.parametrize("tag", [[], {}, 7, None])
def test_an_unhashable_or_non_text_payload_type_raises_schema_error_not_type_error(tag):
    with pytest.raises(SchemaError, match="a payload type is text"):
        StateGraph.from_snapshot(snapshot(information=[
            {"id": "i1", "kind": "fact", "description": "d", "available": True,
             "payload": {"type": tag, "value": 1}}]))


def test_a_mapping_entry_that_is_not_a_pair_is_rejected():
    with pytest.raises(SchemaError, match="is a pair"):
        StateGraph.from_snapshot(snapshot(information=[
            {"id": "i1", "kind": "fact", "description": "d", "available": True,
             "payload": {"type": "mapping", "values": [["a", 1, 2]]}}]))


def test_the_loader_leaves_graph_semantics_to_validation():
    """A snapshot with a cycle loads; saying it does not hold together is validate()'s job."""
    loaded = StateGraph.from_snapshot(snapshot(
        computations=[{"id": "c1", "description": "d", "operation": None, "arguments": {}},
                      {"id": "c2", "description": "d", "operation": None, "arguments": {}}],
        information=[],
        edges=[{"source": "c1", "relation": "precedes", "target": "c2"},
               {"source": "c2", "relation": "precedes", "target": "c1"}]))
    assert len(loaded) == 2


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
