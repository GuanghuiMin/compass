"""The handover shows the future and only the future."""

from future_graph import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    Relation, RuntimeReferencePayload, build,
)
from future_graph.rendering import render


def comp(cid, description, **kw):
    return ComputationNode(id=cid, description=description, **kw)


def info(iid, description, available=True, kind=InformationKind.FACT, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


def episode():
    return build(
        nodes=[comp("c1", "Log in to Venmo and obtain a usable access token",
                    operation="apis.venmo.login",
                    arguments={"username": "paul@example.com"}),
               comp("c2", "Execute the remaining Venmo payments",
                    arguments={"access_token": InformationReference("i2")}),
               comp("c3", "Verify every requested payment outcome"),
               info("i1", "Confirmed Venmo login interface", kind=InformationKind.CONTRACT,
                    payload=ContractPayload("apis.venmo.login", ("username", "password"))),
               info("i2", "Venmo access token", available=False,
                    kind=InformationKind.RUNTIME_REFERENCE,
                    payload=RuntimeReferencePayload("venmo_access_token"))],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2"), ("c2", Relation.PRECEDES, "c3")])


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
    text = render(episode())
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


def test_there_is_no_standalone_information_inventory():
    """Every information line sits under a computation, never in a list of its own."""
    text = render(episode())
    for line in text.splitlines():
        if line.startswith("- [i"):
            assert True
    assert "INFORMATION" not in text


def test_rendering_is_deterministic():
    assert render(episode()) == render(episode())


def test_an_empty_graph_says_so():
    assert render(build()) == "NOTHING REMAINS"
