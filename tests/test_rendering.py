"""The handover shows the future and only the future."""

import pytest

from future_graph import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload, build,
)
from future_graph.rendering import render


def comp(cid, description, **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description, available=True, kind=InformationKind.FACT, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


def episode():
    """A snapshot before the login has run: the token is a result, not a bound name."""
    return build(
        nodes=[comp("c1", "Log in to Venmo and obtain a usable access token",
                    operation="apis.venmo.login",
                    arguments={"username": "paul@example.com"}),
               comp("c2", "Execute the remaining Venmo payments",
                    arguments={"access_token": InformationReference("i2")}),
               comp("c3", "Verify every requested payment outcome"),
               info("i1", "Confirmed Venmo login interface", kind=InformationKind.CONTRACT,
                    payload=ContractPayload("apis.venmo.login", ("username", "password"))),
               info("i2", "A usable access token", available=False,
                    kind=InformationKind.RESULT)],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2"), ("c2", Relation.PRECEDES, "c3")])


def after_login():
    """The snapshot after it ran: the same token is now a name the agent bound, with no producer."""
    return build(
        nodes=[comp("c1", "Execute the remaining Venmo payments",
                    arguments={"access_token": InformationReference("i1")}),
               comp("c2", "Verify every requested payment outcome"),
               info("i1", "The Venmo access token", kind=InformationKind.RUNTIME_REFERENCE,
                    payload=RuntimeReferencePayload("venmo_access_token"))],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i1", Relation.REQUIRES, "c2"),
               ("c1", Relation.PRECEDES, "c2")])


def test_the_frontier_comes_first_and_later_work_is_separated():
    text = render(episode())
    assert text.index("CURRENT COMPUTATIONS") < text.index("[c1]")
    assert text.index("[c1]") < text.index("LATER COMPUTATIONS") < text.index("[c2]")


def test_information_is_shown_under_the_computation_that_needs_it():
    text = render(episode())
    block = text.split("[c1]")[1].split("[c2]")[0]
    assert "Confirmed Venmo login interface" in block
    assert "apis.venmo.login, takes username, password" in block


def test_an_expected_result_is_marked_as_not_yet_available():
    assert "[not yet available]" in render(episode())


def test_a_runtime_reference_shows_its_name_and_no_value():
    text = render(after_login())
    assert "bound as venmo_access_token" in text
    assert "eyJ" not in text


def test_an_argument_referencing_information_renders_as_a_reference():
    assert "access_token = @i2" in render(episode())


def test_a_dependency_on_another_computation_is_stated():
    assert "Depends on: c2" in render(episode())


def test_nothing_historical_appears():
    text = render(episode())
    for forbidden in ("COMPLETED", "HISTORY", "CORRECTIONS", "status", "done", "pending"):
        assert forbidden not in text


def test_rendering_is_deterministic():
    assert render(episode()) == render(episode())


def test_an_empty_graph_says_so():
    assert render(build()) == "NOTHING REMAINS"


# --------------------------------------------------------------------------- how information appears

def _mentions(text):
    """Every information id the text mentions, in order, split into definitions and references.

    A definition reads `- [i1|fact] ...` and a reference reads `- [i1]`.
    """
    definitions, references = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- [i"):
            continue
        inside = stripped[3:stripped.index("]")]
        if "|" in inside:
            definitions.append(inside.split("|")[0])
        else:
            references.append(inside)
    return definitions, references


def shared():
    """One contract needed by two computations, and one result needed by only the second."""
    return build(
        nodes=[comp("c1", "Find the payments that satisfy the request"),
               comp("c2", "Record each payment on Splitwise"),
               info("i1", "Confirmed Splitwise interface", kind=InformationKind.CONTRACT,
                    payload=ContractPayload("apis.splitwise.record_payment", ("group_id",))),
               info("i2", "The group id for the shared house", kind=InformationKind.FACT)],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i1", Relation.REQUIRES, "c2"),
               ("i2", Relation.REQUIRES, "c2"), ("c1", Relation.PRECEDES, "c2")])


def test_unshared_information_is_defined_under_its_only_consumer():
    definitions, references = _mentions(render(shared()))
    assert definitions.count("i2") == 1 and "i2" not in references


def test_shared_information_is_defined_once_and_referenced_afterwards():
    definitions, references = _mentions(render(shared()))
    assert definitions.count("i1") == 1
    assert references.count("i1") == 1


def test_no_information_is_referenced_before_it_is_defined():
    seen = set()
    for line in render(shared()).splitlines():
        stripped = line.strip()
        if not stripped.startswith("- [i"):
            continue
        inside = stripped[3:stripped.index("]")]
        if "|" in inside:
            seen.add(inside.split("|")[0])
        else:
            assert inside in seen, f"{inside} is referred to before it is defined"


def test_a_produced_result_is_defined_under_its_producer():
    """SPEC: first structural mention, which for produced information is the producer."""
    text = render(episode())
    produced = text.split("Produces:")[1].splitlines()[1]
    assert produced.strip().startswith("- [i2|result]")
    consumer_block = text.split("[c2]")[1]
    assert "- [i2]" in consumer_block and "[i2|result]" not in consumer_block


def test_a_definition_carries_its_kind_and_a_reference_does_not():
    text = render(shared())
    assert "[i1|contract]" in text
    assert "- [i1]" in text


@pytest.mark.parametrize("graph_of", [episode, after_login, shared])
def test_every_information_node_is_defined_exactly_once(graph_of):
    """True of a committed graph, where collection has already removed anything with no consumer."""
    graph = graph_of()
    definitions, _ = _mentions(render(graph))
    assert sorted(definitions) == sorted(i.id for i in graph.information)
    assert len(definitions) == len(set(definitions))


def test_there_is_no_standalone_information_inventory():
    text = render(shared())
    assert "INFORMATION" not in text
    for line in text.splitlines():
        if line.strip().startswith("- [i"):
            assert text.index(line) > text.index("[c1]")


# --------------------------------------------------------------------------- types survive

def test_a_string_payload_and_a_number_payload_do_not_read_the_same():
    def one(payload):
        g = build(nodes=[comp("c1", "Do the work"),
                         info("i1", "the value", payload=payload)],
                  edges=[("i1", Relation.REQUIRES, "c1")])
        return render(g)
    assert one(ScalarPayload("7")) != one(ScalarPayload(7))
    assert '("7")' in one(ScalarPayload("7"))
    assert "(7)" in one(ScalarPayload(7))


def test_a_string_argument_and_a_number_argument_do_not_read_the_same():
    def one(value):
        return render(build(nodes=[comp("c1", "Do the work", arguments={"a": value})]))
    assert 'a = "7"' in one("7")
    assert "a = 7" in one(7)


def _rendered_payload(payload):
    g = build(nodes=[comp("c1", "Do the work"), info("i1", "the value", payload=payload)],
              edges=[("i1", Relation.REQUIRES, "c1")])
    return render(g)


@pytest.mark.parametrize("payload,shown", [
    (ScalarPayload("x"), '("x")'),
    (ScalarPayload(7), "(7)"),
    (ListPayload(("x",)), '(["x"])'),
    (ListPayload(()), "([])"),
    (MappingPayload((("a", 1),)), "({a=1})"),
    (MappingPayload(()), "({})"),
])
def test_a_value_shows_its_container_as_well_as_its_type(payload, shown):
    assert shown in _rendered_payload(payload)


def test_no_two_payload_shapes_render_alike():
    shapes = [ScalarPayload("x"), ScalarPayload(7), ListPayload(("x",)), ListPayload(()),
              MappingPayload((("a", 1),)), MappingPayload(())]
    rendered = [_rendered_payload(p) for p in shapes]
    assert len(set(rendered)) == len(shapes)


# --------------------------------------------------------------------------- refinement

def refined():
    """c1 is being worked on and is expanded; c4 is refined but distant, so it stays shut.

    c2 can run, c3 waits for it, and c4's child c5 is not shown at all. i2 crosses out of c1 and
    into c4, so both boundaries declare it and it is one node throughout.
    """
    return build(
        nodes=[comp("c1", "Gather the records that satisfy the request"),
               comp("c2", "Retrieve the first page", operation="example.list_records",
                    arguments={"page": 1}),
               comp("c3", "Continue until no page remains"),
               comp("c4", "Report the outcome"),
               comp("c5", "Draft the summary line"),
               info("i1", "The confirmed listing interface", kind=InformationKind.CONTRACT,
                    payload=ContractPayload("example.list_records", ("page",))),
               info("i2", "The gathered records", available=False,
                    kind=InformationKind.RESULT),
               info("i3", "The house style for summaries")],
        edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
               ("c4", Relation.REFINES, "c5"),
               ("i1", Relation.INTERFACE_INPUT, "c1"),
               ("c1", Relation.INTERFACE_OUTPUT, "i2"),
               ("i1", Relation.REQUIRES, "c2"), ("c3", Relation.PRODUCES, "i2"),
               ("i2", Relation.INTERFACE_INPUT, "c4"), ("i2", Relation.REQUIRES, "c5"),
               ("i3", Relation.INTERFACE_INPUT, "c4"), ("i3", Relation.REQUIRES, "c5"),
               ("c2", Relation.PRECEDES, "c3"), ("c1", Relation.PRECEDES, "c4")])


def test_the_refined_fixture_is_a_graph_the_validator_accepts():
    """A rendering fixture that could never be committed would prove nothing about the handover."""
    from future_graph.validation import validate
    assert validate(refined()) == ()


def test_a_graph_without_refinement_renders_the_way_it_always_has():
    text = render(episode())
    assert "CURRENT COMPUTATIONS" in text
    assert "REFINED PLAN OVERVIEW" not in text
    assert "ACTIVE WORK" not in text


def test_the_refined_plan_comes_first_and_shows_only_the_interface():
    text = render(refined())
    plan = text.split("ACTIVE WORK")[0]
    assert text.index("REFINED PLAN OVERVIEW") == 0
    assert "[c1] Gather the records that satisfy the request" in plan
    assert "[c4] Report the outcome" in plan
    assert "Interface in:" in plan and "Interface out:" in plan
    assert "example.list_records" in plan          # the contract payload, named once
    assert "Operation:" not in plan                # no execution detail at the refined level


def test_active_work_shows_the_executable_leaf_in_full():
    text = render(refined())
    active = text.split("ACTIVE WORK")[1].split("LATER COMPUTATIONS")[0]
    assert "[c2] Retrieve the first page" in active
    assert "Operation: example.list_records" in active
    assert "page = 1" in active


def test_active_work_holds_an_executable_abstract_leaf_that_was_never_refined():
    """Which is why the heading is ACTIVE WORK: what lands here need not be a refinement path."""
    g = build(nodes=[comp("c1", "Gather the records"), comp("c2", "Retrieve the first page"),
                     comp("c3", "Send the acknowledgement")],
              edges=[("c1", Relation.REFINES, "c2")])
    text = render(g)
    active = text.split("ACTIVE WORK")[1]
    assert "[c3] Send the acknowledgement" in active
    assert g.is_leaf("c3") and not g.refinement_parents_of("c3")


def test_a_distant_refined_subtree_is_not_expanded():
    """c4 is refined, but nothing under it can run, so its child stays out of the handover."""
    text = render(refined())
    assert "[c5]" not in text
    assert "Draft the summary line" not in text


def test_a_blocked_sibling_of_the_frontier_is_shown_as_later_work():
    text = render(refined())
    later = text.split("LATER COMPUTATIONS")[1]
    assert "[c3] Continue until no page remains" in later


def test_information_only_a_hidden_subtree_touches_is_not_rendered():
    text = render(refined())
    assert "house style" in text          # c4 declares it as an interface input
    assert text.count("house style") == 1


def test_every_visible_information_node_is_defined_exactly_once():
    text = render(refined())
    definitions, _ = _mentions(text)
    assert len(definitions) == len(set(definitions))
    assert sorted(definitions) == ["i1", "i2", "i3"]


def test_no_information_is_referenced_before_it_is_defined_under_refinement():
    seen = set()
    for line in render(refined()).splitlines():
        stripped = line.strip()
        if not stripped.startswith("- [i"):
            continue
        inside = stripped[3:stripped.index("]")]
        if "|" in inside:
            seen.add(inside.split("|")[0])
        else:
            assert inside in seen, f"{inside} is referred to before it is defined"


def test_a_coarse_computation_names_its_children():
    assert "Refined into: c2, c3" in render(refined())


def test_refined_rendering_is_deterministic():
    assert render(refined()) == render(refined())


def test_a_refined_computation_shows_what_governs_it():
    """Obligation-level knowledge reaches the handover, or the graph holds it and nobody sees it."""
    g = build(nodes=[comp("c1", "Register every seedling"),
                     comp("c2", "Open one entry", operation="example.open_entry"),
                     info("i1", "The batch route was retired and no batch interface exists",
                          kind=InformationKind.FAILURE_CONSEQUENCE),
                     info("i2", "The delivery to register")],
              edges=[("c1", Relation.REFINES, "c2"),
                     ("i1", Relation.REQUIRES, "c1"),
                     ("i2", Relation.INTERFACE_INPUT, "c1"), ("i2", Relation.REQUIRES, "c2")])
    text = render(g)
    plan = text.split("ACTIVE WORK")[0]
    assert "Needs:" in plan
    assert "The batch route was retired" in plan
    assert plan.index("Needs:") < plan.index("Interface in:")
    assert "[i1|failure_consequence]" in plan
