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
kind: result
available: false
description: A usable access token
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
    assert g.node("i2").kind is InformationKind.RESULT
    assert g.node("i2").available is False and g.node("i2").payload is None


def test_an_established_runtime_reference_parses():
    text = ("BEGIN_GRAPH\nINFO i1\nkind: runtime_reference\navailable: true\n"
            "description: the token the agent bound\nruntime-name: example_access_token\n"
            "END_INFO\nCOMPUTATION c1\ndescription: Use it\nEND_COMPUTATION\n"
            "EDGE i1 REQUIRES c1\nEND_GRAPH\n")
    g = parse(text).graph
    assert g.node("i1").payload == RuntimeReferencePayload("example_access_token")


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


RESERVED_LOOKING = ["true", "false", "null", "7", "50.5", "-3", "@i3", "@", "@iX",
                    "", "  padded  ", "1e5", "0", "yes"]


@pytest.mark.parametrize("literal", RESERVED_LOOKING)
def test_a_string_that_looks_like_something_else_round_trips_as_a_string(literal):
    """Quoting only the awkward-looking values is how "true" comes back as a boolean."""
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work"),
                     InformationNode(id="i1", kind=InformationKind.FACT, description="A known thing",
                                     available=True, payload=ScalarPayload(literal))],
              edges=[("i1", Relation.REQUIRES, "c1")])
    back = parse(to_protocol(g)).graph
    assert back == g
    assert back.node("i1").payload.value == literal


@pytest.mark.parametrize("literal", RESERVED_LOOKING)
def test_an_argument_string_that_looks_like_something_else_round_trips(literal):
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work",
                                     arguments={"a": literal})])
    back = parse(to_protocol(g)).graph
    assert back == g
    assert back.node("c1").arguments["a"] == literal


def test_a_literal_at_string_and_a_reference_stay_apart_through_a_round_trip():
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work",
                                     arguments={"text": "@i1", "ref": InformationReference("i1")}),
                     InformationNode(id="i1", kind=InformationKind.FACT, description="A known thing",
                                     available=True)],
              edges=[("i1", Relation.REQUIRES, "c1")])
    back = parse(to_protocol(g)).graph
    assert back.node("c1").arguments["text"] == "@i1"
    assert back.node("c1").arguments["ref"] == InformationReference("i1")


@pytest.mark.parametrize("literal", RESERVED_LOOKING)
def test_list_and_mapping_payloads_keep_string_types_too(literal):
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work"),
                     InformationNode(id="i1", kind=InformationKind.FACT, description="A list",
                                     available=True, payload=ListPayload((literal,))),
                     InformationNode(id="i2", kind=InformationKind.FACT, description="A mapping",
                                     available=True, payload=MappingPayload((("k", literal),)))],
              edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1")])
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


def test_field_order_does_not_matter_and_is_not_logged():
    """Order is part of the grammar, so writing it differently is not a deviation to count."""
    text = WHOLE.replace("kind: result\navailable: false\n",
                         "available: false\nkind: result\n")
    outcome = parse(text)
    assert outcome.ok
    assert not any("order" in n for n in outcome.normalizations)


def test_ids_that_read_as_the_same_number_serialize_in_one_order():
    a = build(nodes=[ComputationNode(id="c1", description="first"),
                     ComputationNode(id="c01", description="second")])
    b = StateGraph()
    b.add(ComputationNode(id="c01", description="second"))
    b.add(ComputationNode(id="c1", description="first"))
    assert to_protocol(a) == to_protocol(b)
    assert parse(to_protocol(a)).graph == a


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
    text = _payload_fixture("runtime-name: a_name\nvalue: 7\n")
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


@pytest.mark.parametrize("written", ["@", "@i", "@iX", "@c1", "@ i3", "@i3x", "@1"])
def test_a_malformed_reference_is_an_error_and_never_an_exception(written):
    text = ("BEGIN_GRAPH\nCOMPUTATION c1\ndescription: d\nargument a = " + written +
            "\nEND_COMPUTATION\nEND_GRAPH\n")
    outcome = parse(text)          # must not raise
    assert outcome.graph is None and outcome.errors
    assert "is not a reference" in " ".join(e.message for e in outcome.errors)


def test_both_a_bad_kind_and_a_bad_available_in_one_block_are_reported():
    text = ("BEGIN_GRAPH\nINFO i1\nkind: memory\navailable: perhaps\ndescription: d\n"
            "END_INFO\nEND_GRAPH\n")
    reported = messages(text)
    assert "is not an information kind" in reported
    assert "available reads true or false" in reported


@pytest.mark.parametrize("line", ["BEGIN_GRAPH extra", "END_GRAPH extra"])
def test_graph_delimiters_reject_extra_words(line):
    keyword = line.split()[0]
    assert "takes 1 word" in messages(WHOLE.replace(keyword + "\n", line + "\n"))


@pytest.mark.parametrize("line", ["END_INFO now", "END_COMPUTATION please"])
def test_block_terminators_reject_extra_words(line):
    keyword = line.split()[0]
    assert "takes 1 word" in messages(WHOLE.replace(keyword + "\n", line + "\n", 1))


def test_a_block_header_with_extra_words_is_rejected_and_builds_no_node():
    outcome = parse(WHOLE.replace("INFO i1", "INFO i1 contract"))
    assert outcome.graph is None
    assert "takes 2 words" in " ".join(e.message for e in outcome.errors)


def test_a_second_end_graph_is_rejected():
    assert "a second END_GRAPH" in messages(WHOLE + "END_GRAPH\n")


def test_a_second_begin_graph_is_rejected():
    assert "a second BEGIN_GRAPH" in messages(WHOLE.replace("BEGIN_GRAPH\n",
                                                            "BEGIN_GRAPH\nBEGIN_GRAPH\n"))


def test_only_one_surrounding_fence_pair_is_absorbed():
    fenced = "```text\n" + WHOLE + "```\n"
    assert parse(fenced).ok
    inside = WHOLE.replace("EDGE i1 REQUIRES c1", "```\nEDGE i1 REQUIRES c1")
    assert "is not a known statement" in messages(inside)
    trailing = "```text\n" + WHOLE + "```\n```\n"
    assert not parse(trailing).ok


def _payload_fixture(payload_lines):
    return ("BEGIN_GRAPH\nINFO i1\nkind: result\navailable: true\ndescription: d\n"
            + payload_lines + "END_INFO\nCOMPUTATION c1\ndescription: d\nEND_COMPUTATION\n"
            "EDGE i1 REQUIRES c1\nEND_GRAPH\n")


def test_an_empty_list_round_trips_as_an_empty_list():
    """The query succeeded and matched nothing. That is a state, not the absence of one."""
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work"),
                     InformationNode(id="i1", kind=InformationKind.RESULT, description="no matches",
                                     available=True, payload=ListPayload(()))],
              edges=[("i1", Relation.REQUIRES, "c1")])
    back = parse(to_protocol(g)).graph
    assert back == g and back.node("i1").payload == ListPayload(())


def test_an_empty_mapping_round_trips_as_an_empty_mapping():
    g = build(nodes=[ComputationNode(id="c1", description="Do the remaining work"),
                     InformationNode(id="i1", kind=InformationKind.RESULT, description="nothing set",
                                     available=True, payload=MappingPayload(()))],
              edges=[("i1", Relation.REQUIRES, "c1")])
    back = parse(to_protocol(g)).graph
    assert back == g and back.node("i1").payload == MappingPayload(())


def test_items_without_a_declared_list_are_refused_rather_than_guessed():
    assert "without 'payload-type: list'" in messages(_payload_fixture("item: 1\nitem: 2\n"))


def test_entries_without_a_declared_mapping_are_refused():
    assert "without 'payload-type: mapping'" in messages(_payload_fixture("entry a = 1\n"))


@pytest.mark.parametrize("declared,intruder", [
    ("list", "value: 7"), ("list", "entry a = 1"), ("list", "runtime-name: n"),
    ("list", "contract-operation: apis.a.b"),
    ("mapping", "value: 7"), ("mapping", "item: 1"), ("mapping", "runtime-name: n"),
])
def test_a_declared_payload_refuses_fields_of_another_kind(declared, intruder):
    text = _payload_fixture(f"payload-type: {declared}\n{intruder}\n")
    assert "and also gives" in messages(text)


def test_an_unknown_payload_type_is_rejected():
    assert "payload-type reads list or mapping" in messages(
        _payload_fixture("payload-type: bag\nitem: 1\n"))


def test_payload_type_is_a_singleton():
    assert "'payload-type' is given twice" in messages(
        _payload_fixture("payload-type: list\npayload-type: mapping\n"))


@pytest.mark.parametrize("written", ["entry a = 1", "entry a=1", "entry a =1", "entry a= 1",
                                     "entry  a  =  1"])
def test_pair_spacing_variants_parse_and_are_logged(written):
    outcome = parse(_payload_fixture(f"payload-type: mapping\n{written}\n"))
    assert outcome.ok, outcome.errors
    assert outcome.graph.node("i1").payload == MappingPayload((("a", 1),))
    if written != "entry a = 1":
        assert any("pair separator spacing" in n for n in outcome.normalizations)
    else:
        assert not any("pair separator spacing" in n for n in outcome.normalizations)


@pytest.mark.parametrize("written", ["value: 7", "value:7", "value : 7", "value  :  7"])
def test_the_approved_colon_spacings_all_parse(written):
    text = ("BEGIN_GRAPH\nINFO i1\nkind: fact\navailable: true\ndescription: d\n" + written +
            "\nEND_INFO\nCOMPUTATION c1\ndescription: d\nEND_COMPUTATION\n"
            "EDGE i1 REQUIRES c1\nEND_GRAPH\n")
    outcome = parse(text)
    assert outcome.ok, outcome.errors
    assert outcome.graph.node("i1").payload == ScalarPayload(7)
    if written != "value: 7":
        assert any("field separator spacing" in n for n in outcome.normalizations)


def test_canonical_spacing_is_not_logged_as_a_normalization():
    assert not any("field separator spacing" in n for n in parse(WHOLE).normalizations)


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
