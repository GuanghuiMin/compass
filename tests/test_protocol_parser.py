"""The block form, and a parser that tolerates surface and repairs nothing.

The graphs written out here are fixtures. Any resemblance to a replay episode is illustration, not a
rule the parser knows about.
"""

import pytest

from future_graph import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload, StateGraph,
    build, parse, to_protocol,
)

WHOLE = """\
BEGIN_GRAPH

INFO i1
kind: contract
available: true
description: Confirmed login interface
contract-operation: apis.example.login
contract-parameter: username
contract-parameter: password
END_INFO

INFO i2
kind: runtime_reference
available: false
description: Access token produced by logging in
runtime-name: example_access_token
END_INFO

COMPUTATION c1
description: Log in and obtain a usable access token
operation: apis.example.login
argument username = "someone@example.com"
END_COMPUTATION

COMPUTATION c2
description: Execute the remaining transfers
argument access_token = @i2
END_COMPUTATION

EDGE i1 REQUIRES c1
EDGE c1 PRODUCES i2
EDGE i2 REQUIRES c2
EDGE c1 PRECEDES c2

END_GRAPH
"""


def parsed(text=WHOLE):
    outcome = parse(text)
    assert outcome.errors == (), outcome.errors
    return outcome.graph


def messages(text):
    return " | ".join(e.message for e in parse(text).errors)


# --------------------------------------------------------------------------- a whole graph

def test_a_whole_graph_parses():
    g = parsed()
    assert [c.id for c in g.computations] == ["c1", "c2"]
    assert [i.id for i in g.information] == ["i1", "i2"]
    assert g.requires_of("c1") == ("i1",) and g.produces_of("c1") == ("i2",)
    assert g.predecessors_of("c2") == ("c1",)


def test_payload_content_survives_exactly():
    g = parsed()
    assert g.node("i1").payload == ContractPayload("apis.example.login", ("username", "password"))
    assert g.node("i2").payload == RuntimeReferencePayload("example_access_token")
    assert g.node("i2").available is False


def test_an_empty_graph_parses():
    g = parsed("BEGIN_GRAPH\nEND_GRAPH\n")
    assert len(g) == 0


# --------------------------------------------------------------------------- round trip

def test_the_whole_graph_round_trips():
    g = parsed()
    assert parse(to_protocol(g)).graph == g


@pytest.mark.parametrize("payload", [
    ScalarPayload(3084), ScalarPayload("a plain phrase"), ScalarPayload(True), ScalarPayload(None),
    ListPayload((285, 291, 306)), MappingPayload((("group_id", 7), ("amount", 50.5))),
    RuntimeReferencePayload("some_name"), ContractPayload("apis.a.b", ("x",), ("y must hold",)),
])
def test_every_payload_kind_round_trips(payload):
    kind = {RuntimeReferencePayload: InformationKind.RUNTIME_REFERENCE,
            ContractPayload: InformationKind.CONTRACT}.get(type(payload), InformationKind.FACT)
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work"),
                     InformationNode(id="i1", kind=kind, description="A thing that is known",
                                     available=True, payload=payload)],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert parse(to_protocol(g)).graph == g


def test_serialization_is_canonical_regardless_of_insertion_order():
    a = build(nodes=[ComputationNode(id="c1", description="First"),
                     ComputationNode(id="c2", description="Second")],
              edges=[("c1", Relation.PRECEDES, "c2")])
    b = StateGraph()
    b.add(ComputationNode(id="c2", description="Second"))
    b.add(ComputationNode(id="c1", description="First"))
    b.add_edge("c1", Relation.PRECEDES, "c2")
    assert to_protocol(a) == to_protocol(b)


# --------------------------------------------------------------------------- tolerated surface

def test_markdown_fences_are_tolerated_and_logged():
    outcome = parse("```text\n" + WHOLE + "```\n")
    assert outcome.ok
    assert any("markdown fence" in n for n in outcome.normalizations)


def test_indentation_and_blank_lines_are_tolerated_and_logged():
    text = WHOLE.replace("kind: contract", "    kind: contract").replace("\n\n", "\n\n\n")
    outcome = parse(text)
    assert outcome.ok
    assert any("indentation" in n for n in outcome.normalizations)
    assert any("blank line" in n for n in outcome.normalizations)


def test_structural_keyword_case_is_tolerated_and_logged():
    text = WHOLE.replace("BEGIN_GRAPH", "begin_graph").replace("EDGE i1 REQUIRES c1",
                                                               "edge i1 requires c1")
    outcome = parse(text)
    assert outcome.ok
    assert any("structural keyword case" in n for n in outcome.normalizations)


def test_field_name_case_is_tolerated_and_logged():
    outcome = parse(WHOLE.replace("description: Log in", "Description: Log in"))
    assert outcome.ok
    assert any("field name case" in n for n in outcome.normalizations)


def test_field_order_does_not_matter():
    text = WHOLE.replace("kind: runtime_reference\navailable: false\n",
                         "available: false\nkind: runtime_reference\n")
    assert parse(text).ok


def test_optional_quotes_are_tolerated_and_logged():
    outcome = parse(WHOLE.replace('argument username = "someone@example.com"',
                                  "argument username = someone@example.com"))
    assert outcome.ok
    assert outcome.graph.node("c1").arguments["username"] == "someone@example.com"
    assert any("scalar quotes" in n for n in parse(WHOLE).normalizations)


def test_trailing_whitespace_is_tolerated():
    assert parse(WHOLE.replace("kind: contract", "kind: contract   ")).ok


def test_ids_and_content_keep_their_own_case():
    text = WHOLE.replace("apis.example.login", "APIs.Example.Login")
    g = parse(text).graph
    assert g.node("i1").payload.operation == "APIs.Example.Login"
    assert g.node("c1").operation == "APIs.Example.Login"


# --------------------------------------------------------------------------- scalars

@pytest.mark.parametrize("written,expected", [
    ("true", True), ("false", False), ("null", None), ("7", 7), ("50.5", 50.5),
    ("plain words", "plain words"), ('"quoted words"', "quoted words"),
])
def test_scalars_read_as_written(written, expected):
    text = ("BEGIN_GRAPH\nCOMPUTATION c1\ndescription: d\nargument a = " + written +
            "\nEND_COMPUTATION\nEND_GRAPH\n")
    assert parse(text).graph.node("c1").arguments["a"] == expected


def test_an_at_reference_is_a_reference_and_a_quoted_one_is_text():
    text = ("BEGIN_GRAPH\nINFO i3\nkind: fact\navailable: true\ndescription: d\nEND_INFO\n"
            "COMPUTATION c1\ndescription: d\nargument a = @i3\nargument b = \"@i3\"\n"
            "END_COMPUTATION\nEDGE i3 REQUIRES c1\nEND_GRAPH\n")
    arguments = parse(text).graph.node("c1").arguments
    assert arguments["a"] == InformationReference("i3")
    assert arguments["b"] == "@i3"


def test_an_at_reference_to_an_absent_node_is_still_a_reference():
    """Whether i9 exists is validation's question, not a hint about what was meant."""
    text = ("BEGIN_GRAPH\nCOMPUTATION c1\ndescription: d\nargument a = @i9\n"
            "END_COMPUTATION\nEND_GRAPH\n")
    assert parse(text).graph.node("c1").arguments["a"] == InformationReference("i9")


# --------------------------------------------------------------------------- refusals

def test_an_edge_naming_an_undeclared_id_yields_no_graph_and_no_placeholder():
    text = ("BEGIN_GRAPH\nCOMPUTATION c1\ndescription: d\nEND_COMPUTATION\n"
            "EDGE i9 REQUIRES c1\nEDGE c1 PRECEDES c7\nEND_GRAPH\n")
    outcome = parse(text)
    assert outcome.graph is None
    assert [e.message for e in outcome.errors] == [
        "edge names 'i9', which no block declares",
        "edge names 'c7', which no block declares",
    ]


def test_an_unknown_field_is_rejected():
    assert "'priority' is not a field" in messages(
        WHOLE.replace("operation: apis.example.login", "priority: high"))


def test_an_unknown_information_kind_is_rejected():
    assert "is not an information kind" in messages(WHOLE.replace("kind: contract", "kind: memory"))


def test_an_unknown_relation_is_rejected():
    assert "is not a relation" in messages(WHOLE.replace("EDGE i1 REQUIRES c1",
                                                         "EDGE i1 REFINES c1"))


def test_a_block_without_its_end_is_rejected():
    assert "never closed" in messages(WHOLE.replace("END_INFO\n\nINFO i2", "\nINFO i2"))


def test_text_outside_the_graph_is_rejected():
    assert "text before BEGIN_GRAPH" in messages("here is the graph you asked for\n" + WHOLE)
    assert "text after END_GRAPH" in messages(WHOLE + "let me know if that works\n")


def test_a_missing_begin_or_end_is_rejected():
    assert "no BEGIN_GRAPH" in messages(WHOLE.replace("BEGIN_GRAPH\n", ""))
    assert "no END_GRAPH" in messages(WHOLE.replace("END_GRAPH\n", ""))


def test_two_payload_kinds_in_one_block_are_rejected():
    text = WHOLE.replace("runtime-name: example_access_token",
                         "runtime-name: example_access_token\nvalue: 7")
    assert "a payload is one kind" in messages(text)


def test_a_repeated_singleton_field_is_rejected():
    assert "'description' is given twice" in messages(
        WHOLE.replace("description: Confirmed login interface",
                      "description: Confirmed login interface\ndescription: something else"))


def test_a_repeated_argument_name_is_rejected_rather_than_overwritten():
    text = WHOLE.replace('argument username = "someone@example.com"',
                         'argument username = "a@b.c"\nargument username = "d@e.f"')
    assert "argument 'username' is given twice" in messages(text)


def test_a_repeated_entry_key_is_rejected():
    text = ("BEGIN_GRAPH\nINFO i1\nkind: fact\navailable: true\ndescription: d\n"
            "entry a = 1\nentry a = 2\nEND_INFO\nEND_GRAPH\n")
    assert "entry 'a' is given twice" in messages(text)


def test_a_duplicate_id_is_rejected():
    text = WHOLE.replace("COMPUTATION c2", "COMPUTATION c1").replace("END_COMPUTATION\n\nEDGE",
                                                                     "END_COMPUTATION\nEDGE")
    assert "declared twice" in messages(text)


def test_a_block_missing_a_required_field_is_rejected():
    assert "has no kind" in messages(WHOLE.replace("kind: contract\n", ""))
    assert "has no description" in messages(WHOLE.replace(
        "description: Log in and obtain a usable access token\n", ""))


def test_a_stringified_container_is_rejected_by_the_schema_through_the_parser():
    text = ("BEGIN_GRAPH\nINFO i1\nkind: result\navailable: true\n"
            "description: the recorded expenses\nvalue: {'name': 'someone', 'amount': 50.0}\n"
            "END_INFO\nEND_GRAPH\n")
    assert "stringified container" in messages(text)


def test_an_id_of_the_wrong_shape_is_rejected():
    assert "c<number>" in messages(WHOLE.replace("COMPUTATION c1", "COMPUTATION node_one")
                                        .replace("EDGE i1 REQUIRES c1", "EDGE i1 REQUIRES node_one")
                                        .replace("EDGE c1 PRODUCES i2", "EDGE node_one PRODUCES i2")
                                        .replace("EDGE c1 PRECEDES c2", "EDGE node_one PRECEDES c2"))


# --------------------------------------------------------------------------- error policy

def test_every_error_is_reported_not_just_the_first():
    text = ("BEGIN_GRAPH\n"
            "INFO i1\nkind: memory\navailable: perhaps\ndescription: d\nEND_INFO\n"
            "COMPUTATION c1\npriority: high\nEND_COMPUTATION\n"
            "EDGE i1 REFINES c1\n"
            "END_GRAPH\n")
    reported = messages(text)
    assert "is not a relation" in reported
    assert "is not a field" in reported
    assert "is not an information kind" in reported


def test_any_error_means_no_graph_at_all():
    outcome = parse(WHOLE.replace("EDGE i1 REQUIRES c1", "EDGE i1 REFINES c1"))
    assert outcome.errors and outcome.graph is None


def test_errors_carry_a_line_number():
    outcome = parse(WHOLE.replace("kind: contract", "kind: memory"))
    assert outcome.errors[0].line > 0
    assert "line" in str(outcome.errors[0])


def test_the_grammar_describes_exactly_what_the_parser_accepts():
    """A grammar the model reads and a parser that disagrees with it is a rejection waiting to happen."""
    from future_graph import GRAMMAR
    from future_graph.protocol import COMPUTATION_FIELDS, INFORMATION_FIELDS

    for name in INFORMATION_FIELDS | COMPUTATION_FIELDS:
        assert name in GRAMMAR, f"{name} is accepted but not documented"
    for kind in InformationKind:
        assert kind.value in GRAMMAR
    for relation in Relation:
        assert relation.name in GRAMMAR


def test_the_parser_adds_no_edges_of_its_own():
    """An argument reference without its requires edge stays missing, for validation to catch."""
    text = ("BEGIN_GRAPH\nINFO i1\nkind: fact\navailable: true\ndescription: d\nEND_INFO\n"
            "COMPUTATION c1\ndescription: d\nargument a = @i1\nEND_COMPUTATION\nEND_GRAPH\n")
    g = parse(text).graph
    assert g.requires_of("c1") == ()
