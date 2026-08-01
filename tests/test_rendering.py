"""The handover shows the future and only the future."""

import pytest

from future_graph import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, Relation, RuntimeReferencePayload, ScalarPayload, build,
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
    """Every information id the text mentions, in order, split into definitions and references."""
    definitions, references = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- [i"):
            continue
        ident = stripped[3:stripped.index("]")]
        (definitions if stripped[stripped.index("]") + 1:].strip() else references).append(ident)
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
    text = render(shared())
    definitions, _ = _mentions(text)
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- [i"):
            continue
        ident = stripped[3:stripped.index("]")]
        if stripped[stripped.index("]") + 1:].strip():
            seen.add(ident)
        else:
            assert ident in seen, f"{ident} is referred to before it is defined"


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


def test_an_empty_list_payload_says_it_holds_nothing():
    g = build(nodes=[comp("c1", "Do the work"),
                     info("i1", "matches for the query", payload=ListPayload(()))],
              edges=[("i1", Relation.REQUIRES, "c1")])
    assert "(nothing)" in render(g)
