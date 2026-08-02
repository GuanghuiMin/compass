"""What the revision form accepts, what it repairs, and what it will not read at all.

Tolerance here is deliberate and bounded: the same surface slips that cost whole regenerations are
absorbed, and nothing beyond them is guessed at. Every test that expects a repair also checks that
the repair was recorded, because a normalization nobody can see is indistinguishable from a model
that never made the mistake.
"""

import pytest

from future_graph.revision import (
    Add, Complete, Invalidate, InvalidateInformation, Replace, ReviseComputation,
    ReviseInformation,
)
from future_graph.revision_parser import parse_revision
from future_graph.schema import InformationKind, InformationReference


def only(text):
    outcome = parse_revision(text)
    assert not outcome.errors, [str(e) for e in outcome.errors]
    return outcome


def wrapped(body: str) -> str:
    return f"BEGIN_REVISION\n{body}\nEND_REVISION\n"


# --------------------------------------------------------------------------- the empty revision

def test_a_revision_with_nothing_in_it_is_a_revision():
    outcome = only("BEGIN_REVISION\nEND_REVISION\n")
    assert outcome.revision.operations == ()
    assert outcome.revision.is_empty


def test_an_empty_revision_survives_blank_lines_and_fences():
    outcome = only("```text\n\nBEGIN_REVISION\n\n\nEND_REVISION\n\n```\n")
    assert outcome.revision.is_empty


def test_nothing_at_all_is_not_an_empty_revision():
    outcome = parse_revision("")
    assert outcome.revision is None
    assert any("BEGIN_REVISION" in e.message for e in outcome.errors)


# --------------------------------------------------------------------------- operations

def test_add_carries_new_computations_and_information():
    outcome = only(wrapped(
        "ADD\n"
        "COMPUTATION +login\n"
        "description: Obtain a token\n"
        "operation: nursery.login\n"
        "produces: +token\n"
        "END_COMPUTATION\n"
        "INFORMATION +token\n"
        "kind: result\n"
        "available: false\n"
        "description: A curator token\n"
        "END_INFORMATION\n"
        "END_ADD"))
    add = outcome.revision.operations[0]
    assert isinstance(add, Add)
    assert add.computations[0].label == "+login"
    assert add.computations[0].produces == ("+token",)
    assert add.information[0].kind is InformationKind.RESULT
    assert add.information[0].available is False


def test_replace_needs_a_reason():
    outcome = parse_revision(wrapped("REPLACE c3\nEND_REPLACE"))
    assert outcome.revision is None
    assert any("reason-for-replacement" in e.message for e in outcome.errors)


def test_replace_carries_its_droppings():
    outcome = only(wrapped(
        "REPLACE c3\n"
        "reason-for-replacement: the route is closed\n"
        "no-longer-requires: i4, i5\n"
        "no-longer-after: c1\n"
        "END_REPLACE"))
    op = outcome.revision.operations[0]
    assert isinstance(op, Replace) and op.anchor == "c3"
    assert op.no_longer_requires == ("i4", "i5")
    assert op.no_longer_after == ("c1",)


def test_complete_carries_what_now_exists():
    outcome = only(wrapped(
        "COMPLETE c7\n"
        "NOW_AVAILABLE i6\n"
        "kind: contract\n"
        "description: The confirmed interface\n"
        "contract-operation: nursery.register_one\n"
        "contract-parameter: seedling_id\n"
        "END_NOW_AVAILABLE\n"
        "END_COMPLETE"))
    op = outcome.revision.operations[0]
    assert isinstance(op, Complete)
    entry = op.now_available[0]
    assert entry.anchor == "i6" and entry.kind is InformationKind.CONTRACT
    assert entry.payload.operation == "nursery.register_one"


def test_complete_may_establish_nothing():
    outcome = only(wrapped("COMPLETE c7\nEND_COMPLETE"))
    assert outcome.revision.operations[0].now_available == ()


def test_the_other_three_operations_read_as_themselves():
    outcome = only(wrapped(
        "INVALIDATE c9\nEND_INVALIDATE\n"
        "REVISE c4\nadd-after: c2\nremove-requires: i3\nEND_REVISE\n"
        "REVISE_INFO i2\ndescription: what it really is\nEND_REVISE_INFO\n"
        "INVALIDATE_INFO i5\nEND_INVALIDATE_INFO"))
    first, second, third, fourth = outcome.revision.operations
    assert isinstance(first, Invalidate) and first.anchor == "c9"
    assert isinstance(second, ReviseComputation)
    assert second.add_after == ("c2",) and second.remove_requires == ("i3",)
    assert isinstance(third, ReviseInformation) and third.description == "what it really is"
    assert isinstance(fourth, InvalidateInformation) and fourth.anchor == "i5"


def test_revise_info_that_says_nothing_about_the_payload_leaves_it_alone():
    outcome = only(wrapped("REVISE_INFO i2\ndescription: clearer\nEND_REVISE_INFO"))
    op = outcome.revision.operations[0]
    assert op.payload_given is False and op.payload is None


def test_revise_info_that_gives_a_payload_says_so():
    outcome = only(wrapped("REVISE_INFO i2\nvalue: 12\nEND_REVISE_INFO"))
    op = outcome.revision.operations[0]
    assert op.payload_given is True and op.payload.value == 12


# --------------------------------------------------------------------------- names

def test_a_new_name_carries_its_plus_and_an_anchor_does_not():
    outcome = only(wrapped(
        "REPLACE c3\nreason-for-replacement: r\n"
        "COMPUTATION +open\ndescription: d\nrequires: i1\nEND_COMPUTATION\nEND_REPLACE"))
    computation = outcome.revision.operations[0].computations[0]
    assert computation.label == "+open" and computation.requires == ("i1",)


def test_an_operation_cannot_name_something_new():
    outcome = parse_revision(wrapped("INVALIDATE +ghost\nEND_INVALIDATE"))
    assert outcome.revision is None
    assert any("declares a new one" in e.message for e in outcome.errors)


def test_a_declared_computation_must_be_new():
    outcome = parse_revision(wrapped(
        "ADD\nCOMPUTATION c3\ndescription: d\nEND_COMPUTATION\nEND_ADD"))
    assert outcome.revision is None
    assert any("leading +" in e.message for e in outcome.errors)


def test_now_available_names_something_that_already_exists():
    outcome = parse_revision(wrapped(
        "COMPLETE c7\nNOW_AVAILABLE +fresh\nEND_NOW_AVAILABLE\nEND_COMPLETE"))
    assert outcome.revision is None
    assert any("already in the graph" in e.message for e in outcome.errors)


def test_an_argument_may_reference_a_new_or_an_existing_information_node():
    outcome = only(wrapped(
        "ADD\nCOMPUTATION +call\ndescription: d\n"
        "argument token = @+token\nargument page = 2\nEND_COMPUTATION\nEND_ADD"))
    arguments = outcome.revision.operations[0].computations[0].arguments
    assert isinstance(arguments["token"], InformationReference)
    assert arguments["token"].information_id == "+token"
    assert arguments["page"] == 2


# --------------------------------------------------------------------------- tolerance

def test_keyword_case_is_repaired_and_recorded():
    outcome = only("begin_revision\ninvalidate c9\nend_invalidate\nend_revision\n")
    assert isinstance(outcome.revision.operations[0], Invalidate)
    assert any("structural keyword case" in n for n in outcome.normalizations)


def test_indentation_and_blank_lines_are_repaired():
    outcome = only("BEGIN_REVISION\n\n  REVISE c4\n\n    add-after: c2\n"
                   "  END_REVISE\n\nEND_REVISION\n")
    assert outcome.revision.operations[0].add_after == ("c2",)
    assert any("indentation" in n for n in outcome.normalizations)


def test_a_missing_block_terminator_is_closed_by_the_next_statement():
    outcome = only(wrapped(
        "REVISE c4\nadd-after: c2\n"
        "INVALIDATE c9\nEND_INVALIDATE"))
    assert len(outcome.revision.operations) == 2
    assert outcome.revision.operations[0].add_after == ("c2",)
    assert isinstance(outcome.revision.operations[1], Invalidate)
    assert any("block terminator" in n for n in outcome.normalizations)


def test_a_missing_terminator_does_not_swallow_the_statement_that_closed_it():
    outcome = only(wrapped(
        "ADD\nCOMPUTATION +a\ndescription: first\n"
        "COMPUTATION +b\ndescription: second\nEND_COMPUTATION\nEND_ADD"))
    labels = [c.label for c in outcome.revision.operations[0].computations]
    assert labels == ["+a", "+b"]


def test_the_last_block_may_be_closed_by_the_end_of_the_revision():
    outcome = only(wrapped("REVISE c4\nadd-after: c2"))
    assert outcome.revision.operations[0].add_after == ("c2",)


def test_a_list_field_may_be_written_twice_and_accumulates():
    outcome = only(wrapped("REVISE c4\nadd-after: c2\nadd-after: c3\nEND_REVISE"))
    assert outcome.revision.operations[0].add_after == ("c2", "c3")


def test_field_name_case_and_spacing_are_repaired():
    outcome = only(wrapped("REVISE c4\nAdd-After:c2\nEND_REVISE"))
    assert outcome.revision.operations[0].add_after == ("c2",)


# --------------------------------------------------------------------------- refusals

def test_an_unknown_statement_is_refused():
    outcome = parse_revision(wrapped("MERGE c1 c2"))
    assert outcome.revision is None
    assert any("not a known statement" in e.message for e in outcome.errors)


def test_a_field_that_belongs_to_another_block_is_refused():
    outcome = parse_revision(wrapped("REVISE c4\nreason-for-replacement: no\nEND_REVISE"))
    assert outcome.revision is None
    assert any("not a field here" in e.message for e in outcome.errors)


def test_text_outside_the_revision_is_refused():
    outcome = parse_revision("Here is my revision.\nBEGIN_REVISION\nEND_REVISION\n")
    assert outcome.revision is None
    assert any("before BEGIN_REVISION" in e.message for e in outcome.errors)


def test_an_information_block_without_its_fields_is_refused():
    outcome = parse_revision(wrapped(
        "ADD\nINFORMATION +x\nkind: fact\nEND_INFORMATION\nEND_ADD"))
    assert outcome.revision is None
    assert any("available" in e.message for e in outcome.errors)


def test_an_unavailable_node_with_a_payload_is_refused_where_the_schema_says_so():
    outcome = parse_revision(wrapped(
        "ADD\nINFORMATION +x\nkind: result\navailable: false\ndescription: d\n"
        "value: 3\nEND_INFORMATION\nEND_ADD"))
    assert outcome.revision is None


def test_now_available_is_not_a_field_of_an_add():
    outcome = parse_revision(wrapped(
        "ADD\nNOW_AVAILABLE i2\nEND_NOW_AVAILABLE\nEND_ADD"))
    assert outcome.revision is None
    assert any("not part of a ADD" in e.message for e in outcome.errors)


@pytest.mark.parametrize("bad", ["c 3", "3c", "c-3"])
def test_a_name_that_is_not_a_name_is_refused(bad):
    outcome = parse_revision(wrapped(f"INVALIDATE {bad}\nEND_INVALIDATE"))
    assert outcome.revision is None
