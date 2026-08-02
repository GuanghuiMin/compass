"""Applying a local revision: what is preserved, what is removed, and what is refused.

The tests are written against the whole pipeline a boundary actually runs -- parse, apply, then the
same completion, validation and collection the baseline uses -- because the interesting failures are
not in any one step. Information left behind by a removed subtree only becomes a problem at
validation; a crossing relation nobody accounted for only becomes a wrong graph after completion
fills in an interface that should never have existed.

Each refusal test also asserts the previous graph is untouched. A refusal that half-applied would be
worse than no updater at all.
"""

import pytest

from future_graph import ComputationNode as C, InformationNode as I
from future_graph import InformationKind as K, Relation as R, build
from future_graph.lifecycle import replace
from future_graph.revision import apply_revision
from future_graph.revision_parser import parse_revision
from future_graph.schema import ContractPayload, InformationReference, ScalarPayload
from future_graph.state_graph import StateGraph


def revise(previous, text):
    """Parse and apply, with no completion yet: for looking at what the revision itself did."""
    outcome = parse_revision(text)
    assert not outcome.errors, [str(e) for e in outcome.errors]
    return apply_revision(previous, outcome.revision)


def commit(previous, text):
    """The whole boundary. Returns the applied revision and the replacement it produced."""
    applied = revise(previous, text)
    if applied.graph is None:
        return applied, None
    return applied, replace(previous, applied.graph)


def codes(applied):
    return sorted(f.code for f in applied.faults)


def described(graph):
    return {n.description for n in graph.computations}


def wrapped(body):
    return f"BEGIN_REVISION\n{body}\nEND_REVISION\n"


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def nursery():
    """A refined obligation with three ordered leaves, one of which needs the seedling list."""
    return build(
        nodes=[C(id="c1", description="Register every seedling"),
               C(id="c2", description="Open an entry", operation="nursery.create_entry"),
               C(id="c3", description="Attach the photo", operation="nursery.attach_photo"),
               C(id="c4", description="Set the status", operation="nursery.set_status"),
               I(id="i1", kind=K.FACT, description="The twelve seedlings", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"), ("c1", R.REFINES, "c4"),
               ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.REQUIRES, "c2"),
               ("c2", R.PRECEDES, "c3"), ("c3", R.PRECEDES, "c4")])


@pytest.fixture
def two_branches():
    """Two independent refined obligations, so a revision can touch one and leave the other."""
    return build(
        nodes=[C(id="c1", description="Register every seedling"),
               C(id="c2", description="Open an entry", operation="nursery.create_entry"),
               C(id="c3", description="Attach the photo", operation="nursery.attach_photo"),
               C(id="c4", description="Publish the catalogue"),
               C(id="c5", description="Build the index", operation="nursery.build_index"),
               C(id="c6", description="Announce it", operation="nursery.announce"),
               I(id="i1", kind=K.FACT, description="The twelve seedlings", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"),
               ("c4", R.REFINES, "c5"), ("c4", R.REFINES, "c6"),
               ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.REQUIRES, "c2"),
               ("c5", R.PRECEDES, "c6"), ("c1", R.PRECEDES, "c4")])


@pytest.fixture
def promised():
    """Work that will establish something a later computation needs."""
    return build(
        nodes=[C(id="c1", description="Confirm how one seedling is registered"),
               C(id="c2", description="Read the interface", operation="nursery.describe"),
               C(id="c3", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="How one seedling is registered",
                 available=False)],
        edges=[("c1", R.REFINES, "c2"), ("c2", R.PRODUCES, "i1"),
               ("c1", R.INTERFACE_OUTPUT, "i1"), ("i1", R.REQUIRES, "c3")])


# --------------------------------------------------------------------------- the empty cases

def test_an_empty_graph_is_built_entirely_through_add():
    applied, replacement = commit(StateGraph(), wrapped(
        "ADD\n"
        "COMPUTATION +login\ndescription: Obtain a curator token\n"
        "operation: nursery.login\nproduces: +token\nEND_COMPUTATION\n"
        "COMPUTATION +register\ndescription: Register every seedling\n"
        "argument token = @+token\nEND_COMPUTATION\n"
        "INFORMATION +token\nkind: result\navailable: false\n"
        "description: A curator token\nEND_INFORMATION\n"
        "END_ADD"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {"Obtain a curator token", "Register every seedling"}
    assert applied.changes.affected_roots == ()
    assert applied.changes.removed_nodes == ()


def test_an_empty_revision_changes_nothing(nursery):
    applied = revise(nursery, "BEGIN_REVISION\nEND_REVISION\n")
    assert applied.faults == ()
    assert applied.graph.to_snapshot() == nursery.to_snapshot()
    assert applied.changes.affected_roots == ()
    assert applied.changes.touched_nodes == ()
    assert applied.changes.removed_nodes == ()
    assert applied.changes.removed_edges == ()
    assert applied.changes.completion_changes == ()


# --------------------------------------------------------------------------- replacement

def test_replacing_a_leaf_keeps_its_siblings(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: create_entry needs a curator token\n"
        "COMPUTATION +open\ndescription: Open an entry with a token\n"
        "operation: nursery.create_entry\nrequires: i1\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {
        "Register every seedling", "Open an entry with a token", "Attach the photo",
        "Set the status"}
    # The siblings kept their own ordering, which nothing in the revision mentioned.
    graph = replacement.graph
    attach = next(n.id for n in graph.computations if n.description == "Attach the photo")
    status = next(n.id for n in graph.computations if n.description == "Set the status")
    assert status in graph.successors_of(attach)


def test_replacing_a_coarse_computation_removes_its_whole_subtree(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c1\nreason-for-replacement: the bulk route was retired\n"
        "no-longer-requires: i1\n"
        "COMPUTATION +one_by_one\ndescription: Register each seedling on its own\n"
        "operation: nursery.register_one\nEND_COMPUTATION\n"
        "END_REPLACE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {"Register each seedling on its own"}
    removed = {node for node, reason in
               [(r.node_id, r.reason) for r in applied.changes.removed_nodes]
               if reason == "affected_region"}
    assert removed == {"c1", "c2", "c3", "c4"}


def test_two_disjoint_regions_are_replaced_in_one_revision(two_branches):
    applied, replacement = commit(two_branches, wrapped(
        "REPLACE c2\nreason-for-replacement: the entry route changed\n"
        "COMPUTATION +open\ndescription: Open an entry properly\n"
        "operation: nursery.create_entry_v2\nrequires: i1\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REPLACE c5\nreason-for-replacement: the index is built differently now\n"
        "COMPUTATION +index\ndescription: Build the index incrementally\n"
        "operation: nursery.build_index_incremental\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE\n"
        "REVISE c6\nadd-after: +index\nEND_REVISE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {
        "Register every seedling", "Open an entry properly", "Attach the photo",
        "Publish the catalogue", "Build the index incrementally", "Announce it"}
    assert applied.changes.affected_roots == ("c2", "c5")


def test_a_region_inside_another_region_is_refused(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c1\nreason-for-replacement: everything changes\n"
        "COMPUTATION +a\ndescription: something\nEND_COMPUTATION\nEND_REPLACE\n"
        "REPLACE c2\nreason-for-replacement: this too\n"
        "COMPUTATION +b\ndescription: something else\nEND_COMPUTATION\nEND_REPLACE"))
    assert "overlapping_affected_regions" in codes(applied)
    assert applied.graph is None


def test_one_computation_cannot_be_named_by_two_operations(nursery):
    applied = revise(nursery, wrapped(
        "COMPLETE c2\nEND_COMPLETE\n"
        "INVALIDATE c2\nEND_INVALIDATE"))
    assert "conflicting_operations" in codes(applied)


def test_a_replacement_takes_the_position_of_what_it_replaced(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: it splits in two\n"
        "COMPUTATION +check\ndescription: Check the entry does not exist\n"
        "operation: nursery.lookup\nrequires: i1\nEND_COMPUTATION\n"
        "COMPUTATION +open\ndescription: Open the entry\n"
        "operation: nursery.create_entry\nafter: +check\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    graph = replacement.graph
    parent = next(n.id for n in graph.computations if n.description == "Register every seedling")
    children = {graph.node(c).description for c in graph.refinement_children_of(parent)}
    assert children == {"Check the entry does not exist", "Open the entry", "Attach the photo",
                        "Set the status"}


def test_a_replacement_of_top_level_work_stays_top_level(two_branches):
    _, replacement = commit(two_branches, wrapped(
        "REPLACE c4\nreason-for-replacement: publishing is one action now\n"
        "no-longer-after: c1\n"
        "COMPUTATION +publish\ndescription: Publish in one call\n"
        "operation: nursery.publish\nEND_COMPUTATION\n"
        "END_REPLACE"))
    graph = replacement.graph
    publish = next(n.id for n in graph.computations if n.description == "Publish in one call")
    assert graph.refinement_parents_of(publish) == ()


def test_a_nested_replacement_keeps_the_grandparent_intact():
    previous = build(
        nodes=[C(id="c1", description="Publish the catalogue"),
               C(id="c2", description="Prepare the pages"),
               C(id="c3", description="Render one page", operation="nursery.render"),
               C(id="c4", description="Announce it", operation="nursery.announce"),
               I(id="i1", kind=K.FACT, description="The catalogue template", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c4"), ("c2", R.REFINES, "c3"),
               ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.INTERFACE_INPUT, "c2"),
               ("i1", R.REQUIRES, "c3")])
    _, replacement = commit(previous, wrapped(
        "REPLACE c2\nreason-for-replacement: pages render in batches now\n"
        "COMPUTATION +batch\ndescription: Render the pages in one batch\n"
        "operation: nursery.render_all\nrequires: i1\nEND_COMPUTATION\n"
        "END_REPLACE"))
    graph = replacement.graph
    top = next(n.id for n in graph.computations if n.description == "Publish the catalogue")
    children = {graph.node(c).description for c in graph.refinement_children_of(top)}
    assert children == {"Render the pages in one batch", "Announce it"}
    # The template still crosses into the grandparent, because a leaf inside it still needs it.
    template = next(n.id for n in graph.information if n.description == "The catalogue template")
    assert top in [e.target for e in graph.edges
                   if e.source == template and e.relation is R.INTERFACE_INPUT]


# --------------------------------------------------------------------------- crossings

def test_information_the_replaced_work_needed_must_be_required_again_or_dropped(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: it changed\n"
        "COMPUTATION +open\ndescription: Open an entry\n"
        "operation: nursery.create_entry\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert any("i1" in f.message for f in applied.faults)


def test_saying_it_is_no_longer_required_accounts_for_it(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: the list is not needed to open an entry\n"
        "no-longer-requires: i1\n"
        "COMPUTATION +open\ndescription: Open an empty entry\n"
        "operation: nursery.create_entry\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted
    # Nothing requires the seedling list any more, so collection takes it.
    assert "The twelve seedlings" not in {n.description for n in replacement.graph.information}


def test_an_order_the_replaced_work_was_under_must_be_kept_or_dropped(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c3\nreason-for-replacement: photos attach differently\n"
        "COMPUTATION +attach\ndescription: Attach every photo at once\n"
        "operation: nursery.attach_all\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c4\nadd-after: +attach\nEND_REVISE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert any("c2" in f.nodes for f in applied.faults)


def test_the_replacement_may_keep_the_order_the_replaced_work_was_under(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c3\nreason-for-replacement: photos attach differently\n"
        "COMPUTATION +attach\ndescription: Attach every photo at once\n"
        "operation: nursery.attach_all\nafter: c2\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c4\nadd-after: +attach\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    opened = next(n.id for n in graph.computations if n.description == "Open an entry")
    attach = next(n.id for n in graph.computations
                  if n.description == "Attach every photo at once")
    assert attach in graph.successors_of(opened)


def test_saying_the_order_no_longer_holds_accounts_for_it(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c3\nreason-for-replacement: photos are attached before entries exist\n"
        "no-longer-after: c2\n"
        "COMPUTATION +attach\ndescription: Upload every photo first\n"
        "operation: nursery.upload\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c4\nadd-after: +attach\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    opened = next(n.id for n in graph.computations if n.description == "Open an entry")
    attach = next(n.id for n in graph.computations
                  if n.description == "Upload every photo first")
    assert attach not in graph.successors_of(opened)


def test_completed_work_carries_away_what_it_needed_and_what_it_waited_on(nursery):
    """`COMPLETE` and `INVALIDATE` account for their own incoming crossings: the work is gone, so
    what it required and the order it was under go with it."""
    applied, replacement = commit(nursery, wrapped("COMPLETE c2\nEND_COMPLETE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {
        "Register every seedling", "Attach the photo", "Set the status"}


def test_work_waiting_on_removed_work_must_be_told_what_it_waits_on_now(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: it changed\n"
        "COMPUTATION +open\ndescription: Open an entry\n"
        "operation: nursery.create_entry\nrequires: i1\nEND_COMPUTATION\n"
        "END_REPLACE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert any("waits on" in f.message for f in applied.faults)


def fed_by_its_predecessor():
    """A survivor ordered after work that also produces what it needs, so the dataflow looks like
    it says the same thing the ordering says."""
    return build(
        nodes=[C(id="c1", description="Fetch the roster", operation="nursery.roster"),
               C(id="c2", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="The roster", available=False)],
        edges=[("c1", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c2"), ("c1", R.PRECEDES, "c2")])


def test_dataflow_into_a_successor_does_not_account_for_the_ordering_it_lost():
    """Needing what one part of a replacement produces orders the successor after that part. The
    removed edge may have meant it waits for the whole obligation, and only the model can say which
    -- so a successor the revision never mentions is refused, however well the dataflow lines up."""
    applied = revise(fed_by_its_predecessor(), wrapped(
        "REPLACE c1\nreason-for-replacement: the roster is paginated\n"
        "COMPUTATION +page\ndescription: Fetch the roster page by page\n"
        "operation: nursery.roster_page\nproduces: i1\nEND_COMPUTATION\n"
        "END_REPLACE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert applied.graph is None


def test_a_successor_the_replacement_feeds_is_accounted_for_by_remove_after_alone():
    """The whole point of allowing `remove-after` on its own: the model says the old explicit
    ordering no longer holds and leaves the rest to the dataflow, without writing the duplicate
    `PRECEDES` that an `add-after` would be here."""
    previous = fed_by_its_predecessor()
    applied, replacement = commit(previous, wrapped(
        "REPLACE c1\nreason-for-replacement: the roster is paginated\n"
        "COMPUTATION +page\ndescription: Fetch the roster page by page\n"
        "operation: nursery.roster_page\nproduces: i1\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c2\nremove-after: c1\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    page = next(n.id for n in graph.computations
                if n.description == "Fetch the roster page by page")
    register = next(n.id for n in graph.computations
                    if n.description == "Register each seedling")
    assert graph.successors_of(page) == ()          # no ordering the dataflow already gives
    assert register in graph.consumers_of(graph.information[0].id)


def test_revising_a_successor_about_something_else_does_not_account_for_the_ordering():
    """Otherwise any `REVISE` on the right computation would launder the crossing, and the
    accounting would be satisfied by a revision that never mentioned an ordering at all."""
    previous = fed_by_its_predecessor()
    applied = revise(previous, wrapped(
        "ADD\nINFORMATION +quota\nkind: constraint\navailable: true\n"
        "description: At most four a minute\nEND_INFORMATION\nEND_ADD\n"
        "REPLACE c1\nreason-for-replacement: the roster is paginated\n"
        "COMPUTATION +page\ndescription: Fetch the roster page by page\n"
        "operation: nursery.roster_page\nproduces: i1\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c2\nadd-requires: +quota\nEND_REVISE"))
    assert "unaccounted_crossing_relation" in codes(applied)


def test_a_successor_may_also_be_given_a_new_ordering_over_the_replacement():
    """When the replacement splits into parts and the successor really must wait for all of them,
    the model says so, and that is a different graph from the one above."""
    previous = fed_by_its_predecessor()
    applied, replacement = commit(previous, wrapped(
        "REPLACE c1\nreason-for-replacement: the roster is fetched and then checked\n"
        "COMPUTATION +page\ndescription: Fetch the roster page by page\n"
        "operation: nursery.roster_page\nproduces: i1\nEND_COMPUTATION\n"
        "COMPUTATION +check\ndescription: Check the roster is complete\n"
        "operation: nursery.verify_roster\nrequires: i1\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c2\nremove-after: c1\nadd-after: +check\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    check = next(n.id for n in graph.computations
                 if n.description == "Check the roster is complete")
    register = next(n.id for n in graph.computations
                    if n.description == "Register each seedling")
    assert register in graph.successors_of(check)


def test_something_the_removed_work_would_have_produced_must_be_accounted_for(promised):
    applied = revise(promised, wrapped(
        "INVALIDATE c1\nEND_INVALIDATE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert any("i1" in f.message for f in applied.faults)


def test_invalidating_the_result_too_still_leaves_the_consumer_to_be_answered_for(promised):
    applied = revise(promised, wrapped(
        "INVALIDATE c1\nEND_INVALIDATE\n"
        "INVALIDATE_INFO i1\nEND_INVALIDATE_INFO"))
    assert "unhandled_information_reference" in codes(applied)


def test_an_invalidated_prerequisite_cannot_silently_unlock_its_successor(promised):
    """The whole point of refusing: `c3` needed something that will now never exist, and a graph
    where it simply requires nothing would read as ready to run."""
    applied, replacement = commit(promised, wrapped(
        "INVALIDATE c1\nEND_INVALIDATE\n"
        "INVALIDATE_INFO i1\nEND_INVALIDATE_INFO\n"
        "REVISE c3\nremove-requires: i1\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert described(replacement.graph) == {"Register each seedling"}


def test_information_cannot_be_invalidated_while_something_still_produces_it(promised):
    applied = revise(promised, wrapped("INVALIDATE_INFO i1\nEND_INVALIDATE_INFO"))
    assert "surviving_producer_of_invalidated_information" in codes(applied)


def test_a_reference_into_a_removed_region_is_refused(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c1\nreason-for-replacement: everything changes\nno-longer-requires: i1\n"
        "COMPUTATION +a\ndescription: something\nafter: c2\nEND_COMPUTATION\nEND_REPLACE"))
    assert "reference_into_removed_region" in codes(applied)


def test_an_anchor_that_names_nothing_is_refused(nursery):
    applied = revise(nursery, wrapped("INVALIDATE c99\nEND_INVALIDATE"))
    assert "unknown_anchor" in codes(applied)


def test_a_label_used_but_never_declared_is_reported_once_with_all_its_uses(nursery):
    """Boundary 2 of the recurrent run. The code cannot invent the node -- it knows neither the
    kind, the availability, the description nor the payload -- so this stays a refusal. What it can
    do is say the whole thing once instead of once per mention."""
    applied = revise(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: the messages have to be fetched first\n"
        "COMPUTATION +search\ndescription: Retrieve the messages\n"
        "operation: phone.search\nrequires: i1\nproduces: +messages\nEND_COMPUTATION\n"
        "COMPUTATION +parse\ndescription: Parse the messages\n"
        "requires: +messages\nafter: +search\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c3\nremove-after: c2\nadd-after: +parse\nEND_REVISE"))
    undeclared = [f for f in applied.faults if f.code == "undeclared_new_label"]
    assert len(undeclared) == 1
    fault = undeclared[0]
    assert fault.nodes == ("+messages",)
    assert fault.sites == ("+search.produces", "+parse.requires")
    assert "+messages" in fault.message


def test_a_reference_into_a_removed_region_is_also_reported_once(nursery):
    applied = revise(nursery, wrapped(
        "REPLACE c1\nreason-for-replacement: everything changes\nno-longer-requires: i1\n"
        "COMPUTATION +a\ndescription: something\nafter: c2\nEND_COMPUTATION\n"
        "COMPUTATION +b\ndescription: something else\nafter: c2\nEND_COMPUTATION\n"
        "END_REPLACE"))
    removed = [f for f in applied.faults if f.code == "reference_into_removed_region"]
    assert len(removed) == 1
    assert removed[0].sites == ("+a.after", "+b.after")


def test_a_bare_name_that_names_nothing_says_how_a_new_node_is_written(nursery):
    applied = revise(nursery, wrapped(
        "ADD\nCOMPUTATION +a\ndescription: d\nrequires: i9\nEND_COMPUTATION\nEND_ADD"))
    fault = next(f for f in applied.faults if f.code == "unknown_anchor")
    assert "+i9" in fault.message


def test_a_declaration_that_reuses_the_replaced_name_commits():
    """The same shape the recurrent run refused: refine an abstract leaf in place, giving the
    replacement the name the leaf had."""
    previous = build(
        nodes=[C(id="c1", description="Do the whole job"),
               C(id="c2", description="First part", operation="nursery.first"),
               C(id="c3", description="Work out the rest"),
               I(id="i1", kind=K.FACT, description="The starting facts", available=True)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"),
               ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.REQUIRES, "c2"),
               ("c2", R.PRECEDES, "c3")])
    applied, replacement = commit(previous, wrapped(
        "REPLACE c3\nreason-for-replacement: it is understood well enough to break down now\n"
        "COMPUTATION c3\ndescription: Work out the rest\n"
        "refined-into: +look, +decide\nafter: c2\nEND_COMPUTATION\n"
        "COMPUTATION +look\ndescription: Look at what the first part produced\n"
        "operation: nursery.look\nrequires: i1\nEND_COMPUTATION\n"
        "COMPUTATION +decide\ndescription: Decide what to do\n"
        "operation: nursery.decide\nafter: +look\nEND_COMPUTATION\n"
        "END_REPLACE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    rest = next(n.id for n in graph.computations if n.description == "Work out the rest")
    children = {graph.node(c).description for c in graph.refinement_children_of(rest)}
    assert children == {"Look at what the first part produced", "Decide what to do"}
    top = next(n.id for n in graph.computations if n.description == "Do the whole job")
    assert rest in graph.refinement_children_of(top)


# --------------------------------------------------------------------------- completion

def test_completing_work_establishes_what_it_promised(promised):
    applied, replacement = commit(promised, wrapped(
        "COMPLETE c1\n"
        "NOW_AVAILABLE i1\n"
        "kind: contract\n"
        "description: The confirmed interface for registering one seedling\n"
        "contract-operation: nursery.register_one\n"
        "contract-parameter: seedling_id\n"
        "END_NOW_AVAILABLE\n"
        "END_COMPLETE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    assert described(graph) == {"Register each seedling"}
    node = graph.information[0]
    assert node.available is True
    assert node.kind is K.CONTRACT
    assert isinstance(node.payload, ContractPayload)
    assert node.payload.operation == "nursery.register_one"


def test_completion_is_recorded_as_completion_and_not_as_derivation(promised):
    applied, _ = commit(promised, wrapped(
        "COMPLETE c1\nNOW_AVAILABLE i1\nkind: contract\ncontract-operation: nursery.register_one\n"
        "END_NOW_AVAILABLE\nEND_COMPLETE"))
    actions = {c.action for c in applied.changes.completion_changes}
    assert "became_available" in actions
    assert "producer_removed" in actions
    assert "content_replaced" in actions


def test_completion_materializes_the_provenance_the_structure_lost():
    """Once the producing child is gone, no edge in the graph says the refinement established the
    result -- and an available node has no producer either way, so nothing could reconstruct it."""
    previous = build(
        nodes=[C(id="c1", description="Confirm how registration works"),
               C(id="c2", description="Read the interface", operation="nursery.describe"),
               C(id="c3", description="Check the quota", operation="nursery.quota"),
               C(id="c4", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="How registration works", available=False)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"), ("c2", R.PRODUCES, "i1"),
               ("c1", R.INTERFACE_OUTPUT, "i1"), ("i1", R.REQUIRES, "c4")])
    applied, replacement = commit(previous, wrapped(
        "COMPLETE c2\nNOW_AVAILABLE i1\nkind: contract\n"
        "contract-operation: nursery.register_one\nEND_NOW_AVAILABLE\nEND_COMPLETE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    parent = next(n.id for n in graph.computations
                  if n.description == "Confirm how registration works")
    result = next(n.id for n in graph.information if n.available)
    assert (parent, R.INTERFACE_OUTPUT, result) in [
        (e.source, e.relation, e.target) for e in graph.edges]
    assert "provenance_materialized" in {c.action for c in applied.changes.completion_changes}


def test_now_available_must_name_something_the_completed_work_was_producing(promised):
    previous = build(
        nodes=[C(id="c1", description="Read the interface", operation="nursery.describe"),
               C(id="c2", description="Fetch the quota", operation="nursery.quota"),
               C(id="c3", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="The quota", available=False)],
        edges=[("c2", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c3")])
    applied = revise(previous, wrapped(
        "COMPLETE c1\nNOW_AVAILABLE i1\nEND_NOW_AVAILABLE\nEND_COMPLETE"))
    assert "now_available_producer_outside_region" in codes(applied)
    assert applied.graph is None


def test_now_available_cannot_name_something_that_already_exists(nursery):
    applied = revise(nursery, wrapped(
        "COMPLETE c2\nNOW_AVAILABLE i1\nEND_NOW_AVAILABLE\nEND_COMPLETE"))
    assert "now_available_is_already_available" in codes(applied)


def test_completing_work_that_promised_something_nobody_needs_drops_it():
    previous = build(
        nodes=[C(id="c1", description="Read the interface", operation="nursery.describe"),
               C(id="c2", description="Register each seedling"),
               I(id="i1", kind=K.RESULT, description="How registration works", available=False)],
        edges=[("c1", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c2")])
    applied, replacement = commit(previous, wrapped(
        "COMPLETE c1\nEND_COMPLETE\nREVISE c2\nremove-requires: i1\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert replacement.graph.information == ()
    # It went with the region rather than through collection, because the only thing that was
    # going to produce it is gone and its one consumer said it no longer needs it.
    assert [r.reason for r in applied.changes.removed_nodes if r.node_id == "i1"] \
        == ["region_internal"]


# --------------------------------------------------------------------------- internal information

def test_information_used_only_inside_a_removed_region_goes_with_it():
    """Correction A. Nothing outside ever needed the draft, so leaving it behind would fail the
    one-producer rule at validation and refuse a revision the model wrote correctly."""
    previous = build(
        nodes=[C(id="c1", description="Publish the catalogue"),
               C(id="c2", description="Draft the pages", operation="nursery.draft"),
               C(id="c3", description="Typeset the draft", operation="nursery.typeset"),
               C(id="c4", description="Announce it", operation="nursery.announce"),
               I(id="i1", kind=K.RESULT, description="The drafted pages", available=False)],
        edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"), ("c1", R.REFINES, "c4"),
               ("c2", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c3"), ("c3", R.PRECEDES, "c4")])
    applied, replacement = commit(previous, wrapped(
        "REPLACE c2\nreason-for-replacement: pages are drafted and typeset in one pass\n"
        "COMPUTATION +both\ndescription: Draft and typeset in one pass\n"
        "operation: nursery.compose\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REPLACE c3\nreason-for-replacement: it is part of composing now\n"
        "COMPUTATION +noop\ndescription: Verify the composed pages\n"
        "operation: nursery.verify\nafter: +both\nEND_COMPUTATION\n"
        "END_REPLACE\n"
        "REVISE c4\nadd-after: +noop\nEND_REVISE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert replacement.graph.information == ()
    internal = [r.node_id for r in applied.changes.removed_nodes if r.reason == "region_internal"]
    assert internal == ["i1"]


def test_region_internal_removal_is_recorded_apart_from_invalidation(promised):
    applied, _ = commit(promised, wrapped(
        "INVALIDATE c1\nEND_INVALIDATE\n"
        "INVALIDATE_INFO i1\nEND_INVALIDATE_INFO\n"
        "REVISE c3\nremove-requires: i1\nEND_REVISE"))
    reasons = {r.node_id: r.reason for r in applied.changes.removed_nodes}
    assert reasons["i1"] == "invalidated_information"


def test_information_with_a_surviving_consumer_is_not_internal():
    """It leaves the region, so it is the model's to account for and not the code's to delete."""
    previous = build(
        nodes=[C(id="c1", description="Draft the pages", operation="nursery.draft"),
               C(id="c2", description="Typeset the draft", operation="nursery.typeset"),
               I(id="i1", kind=K.RESULT, description="The drafted pages", available=False)],
        edges=[("c1", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c2")])
    applied = revise(previous, wrapped(
        "REPLACE c1\nreason-for-replacement: drafting changed\n"
        "COMPUTATION +draft\ndescription: Draft from the template\n"
        "operation: nursery.draft_v2\nEND_COMPUTATION\nEND_REPLACE"))
    assert "unaccounted_crossing_relation" in codes(applied)
    assert not [r for r in applied.changes.removed_nodes if r.reason == "region_internal"]


def test_information_the_revision_reuses_is_not_internal():
    previous = build(
        nodes=[C(id="c1", description="Draft the pages", operation="nursery.draft"),
               C(id="c2", description="Typeset the draft", operation="nursery.typeset"),
               I(id="i1", kind=K.RESULT, description="The drafted pages", available=False)],
        edges=[("c1", R.PRODUCES, "i1"), ("i1", R.REQUIRES, "c2"), ("c1", R.PRECEDES, "c2")])
    applied, replacement = commit(previous, wrapped(
        "REPLACE c1\nreason-for-replacement: drafting changed\n"
        "COMPUTATION +draft\ndescription: Draft from the template\n"
        "operation: nursery.draft_v2\nproduces: i1\nEND_COMPUTATION\nEND_REPLACE\n"
        "REPLACE c2\nreason-for-replacement: typesetting changed too\n"
        "COMPUTATION +set\ndescription: Typeset with the new engine\n"
        "operation: nursery.typeset_v2\nrequires: i1\nEND_COMPUTATION\nEND_REPLACE"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    assert [n.description for n in replacement.graph.information] == ["The drafted pages"]


# --------------------------------------------------------------------------- touching what stays

def test_revising_information_keeps_its_availability_and_its_relations(nursery):
    applied, replacement = commit(nursery, wrapped(
        "REVISE_INFO i1\n"
        "description: The twelve seedlings, by nursery identifier\n"
        "END_REVISE_INFO"))
    assert applied.faults == ()
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    node = graph.information[0]
    assert node.description == "The twelve seedlings, by nursery identifier"
    assert node.available is True and node.kind is K.FACT
    assert len(graph.consumers_of(node.id)) == 1
    assert applied.changes.touched_nodes == ("i1",)


def test_revising_information_may_sharpen_its_kind_and_payload():
    previous = build(
        nodes=[C(id="c1", description="Register", operation="nursery.register_one"),
               I(id="i1", kind=K.FACT, description="The registration route", available=True)],
        edges=[("i1", R.REQUIRES, "c1")])
    _, replacement = commit(previous, wrapped(
        "REVISE_INFO i1\nkind: contract\ncontract-operation: nursery.register_one\n"
        "contract-parameter: seedling_id\nEND_REVISE_INFO"))
    node = replacement.graph.information[0]
    assert node.kind is K.CONTRACT and node.payload.parameters == ("seedling_id",)


def test_a_computation_that_stays_can_be_given_a_new_requirement(two_branches):
    applied, replacement = commit(two_branches, wrapped(
        "ADD\nINFORMATION +quota\nkind: constraint\navailable: true\n"
        "description: At most four entries a minute\nEND_INFORMATION\nEND_ADD\n"
        "REVISE c2\nadd-requires: +quota\nEND_REVISE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    graph = replacement.graph
    quota = next(n.id for n in graph.information
                 if n.description == "At most four entries a minute")
    assert len(graph.consumers_of(quota)) == 1
    assert applied.changes.touched_nodes == ("c2",)


# --------------------------------------------------------------------------- identity

def test_identity_is_allocated_in_a_fixed_order(nursery):
    applied = revise(nursery, wrapped(
        "ADD\nCOMPUTATION +login\ndescription: Obtain a token\n"
        "operation: nursery.login\nEND_COMPUTATION\nEND_ADD\n"
        "REPLACE c2\nreason-for-replacement: it needs a token\nno-longer-requires: i1\n"
        "COMPUTATION +open\ndescription: Open an entry\n"
        "operation: nursery.create_entry\nafter: +login\nEND_COMPUTATION\nEND_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    mapping = dict(applied.changes.id_map)
    # Survivors first, in the order the previous graph held them; then what the revision added.
    assert mapping["c1"] == "c1"
    assert mapping["c3"] == "c2"
    assert mapping["c4"] == "c3"
    assert mapping["+login"] == "c4"
    assert mapping["+open"] == "c5"
    assert mapping["i1"] == "i1"


def test_the_mapping_covers_every_node_of_the_resulting_graph(nursery):
    applied = revise(nursery, wrapped(
        "ADD\nCOMPUTATION +extra\ndescription: Something new\nEND_COMPUTATION\nEND_ADD"))
    mapping = dict(applied.changes.id_map)
    assert set(mapping.values()) == {n.id for n in applied.graph.computations} \
        | {n.id for n in applied.graph.information}


def test_the_same_revision_twice_gives_the_same_identity(nursery):
    text = wrapped(
        "ADD\nCOMPUTATION +a\ndescription: first\nEND_COMPUTATION\n"
        "COMPUTATION +b\ndescription: second\nEND_COMPUTATION\nEND_ADD")
    first, second = revise(nursery, text), revise(nursery, text)
    assert first.changes.id_map == second.changes.id_map
    assert first.graph.to_snapshot() == second.graph.to_snapshot()


def test_a_label_declared_twice_is_refused(nursery):
    applied = revise(nursery, wrapped(
        "ADD\nCOMPUTATION +a\ndescription: first\nEND_COMPUTATION\n"
        "COMPUTATION +a\ndescription: second\nEND_COMPUTATION\nEND_ADD"))
    assert "redeclared_label" in codes(applied)


# --------------------------------------------------------------------------- who did what

def test_what_the_code_worked_out_is_kept_apart_from_what_the_model_wrote(nursery):
    applied, replacement = commit(nursery, wrapped(
        "ADD\nCOMPUTATION +login\ndescription: Obtain a token\n"
        "operation: nursery.login\nproduces: +token\nEND_COMPUTATION\n"
        "INFORMATION +token\nkind: result\navailable: false\n"
        "description: A curator token\nEND_INFORMATION\nEND_ADD\n"
        "REPLACE c2\nreason-for-replacement: create_entry needs a token\n"
        "COMPUTATION +open\ndescription: Open an entry with a token\n"
        "operation: nursery.create_entry\nargument token = @+token\nrequires: i1\n"
        "after: +login\nEND_COMPUTATION\nEND_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    assert replacement.accepted, [str(v) for v in replacement.violations]
    changes = applied.changes
    # The model named one region and touched one node; it wrote no interface edge and no
    # dependency the argument already implied.
    assert changes.affected_roots == ("c2",)
    assert changes.touched_nodes == ("c3",)
    assert changes.replacement_boundary_changes
    assert replacement.argument_dependency_changes
    assert replacement.interface_changes
    assert changes.completion_changes == ()


def test_collection_is_recorded_as_collection_and_not_as_removal(nursery):
    """Information nobody needs is not something the revision removed; it is something that stopped
    being needed. Pooling the two would hide which of them the model was responsible for."""
    applied, replacement = commit(nursery, wrapped(
        "REPLACE c2\nreason-for-replacement: the list is not needed to open an entry\n"
        "no-longer-requires: i1\n"
        "COMPUTATION +open\ndescription: Open an empty entry\n"
        "operation: nursery.create_entry\nEND_COMPUTATION\nEND_REPLACE\n"
        "REVISE c3\nadd-after: +open\nEND_REVISE"))
    assert replacement.collected
    assert "i1" not in [r.node_id for r in applied.changes.removed_nodes]


def test_an_argument_still_implies_the_dependency_it_always_did(nursery):
    _, replacement = commit(nursery, wrapped(
        "ADD\nCOMPUTATION +check\ndescription: Check the roster\n"
        "operation: nursery.check\nargument roster = @i1\nEND_COMPUTATION\nEND_ADD"))
    graph = replacement.graph
    check = next(n.id for n in graph.computations if n.description == "Check the roster")
    assert "i1" in [e.source for e in graph.edges
                    if e.target == check and e.relation is R.REQUIRES]


# --------------------------------------------------------------------------- refusal is total

@pytest.mark.parametrize("body", [
    "INVALIDATE c99\nEND_INVALIDATE",
    "REPLACE c1\nreason-for-replacement: a\nCOMPUTATION +x\ndescription: d\nEND_COMPUTATION\n"
    "END_REPLACE\nREPLACE c2\nreason-for-replacement: b\nCOMPUTATION +y\ndescription: d\n"
    "END_COMPUTATION\nEND_REPLACE",
    "REPLACE c2\nreason-for-replacement: a\nCOMPUTATION +x\ndescription: d\nEND_COMPUTATION\n"
    "END_REPLACE",
])
def test_a_refused_revision_leaves_the_previous_graph_exactly_as_it_was(nursery, body):
    before = nursery.to_snapshot()
    applied = revise(nursery, wrapped(body))
    assert applied.graph is None
    assert applied.faults
    assert nursery.to_snapshot() == before
