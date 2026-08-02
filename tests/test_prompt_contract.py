"""The clauses the updater prompt has to keep saying.

These pin meaning, not wording. Asserting the prompt byte-for-byte would make every rephrasing a test
failure and would teach nobody anything; asserting nothing would let the absorption contract be
edited away one sentence at a time and the suite stay green. So each test names one requirement and
looks for the phrases that carry it, with enough alternatives that an honest rewrite survives and a
deletion does not.

The requirement each test defends is in its docstring, because that is the thing under test -- the
strings below are only how it is currently said.
"""

import re

import pytest

from future_graph.regeneration import PROMPT_PATH, load_prompt


@pytest.fixture(scope="module")
def prompt() -> str:
    return load_prompt()


def says(text: str, *phrases: str) -> bool:
    """Does the prompt contain every one of these, ignoring case and line wrapping?

    Whitespace is collapsed on both sides so that reflowing a paragraph -- which changes where the
    line breaks fall and nothing else -- cannot fail a test about what the prompt requires.
    """
    flat = re.sub(r"\s+", " ", text).lower()
    return all(re.sub(r"\s+", " ", p).lower() in flat for p in phrases)


def test_the_prompt_is_the_committed_file(prompt):
    """A prompt found anywhere else would be a different experiment."""
    assert PROMPT_PATH.name == "regenerate_graph.md"
    assert prompt.startswith("You maintain the state of a long task as a graph")


# --------------------------------------------------------------------------- the order

def test_the_whole_slice_is_read_before_anything_is_dropped(prompt):
    """1. Read all of DELTA_H, including long observations, before deciding what to keep."""
    assert says(prompt, "read all of `delta_h`")
    assert says(prompt, "every observation in full")
    assert says(prompt, "a long error is not less important for being long")


def test_the_plan_is_revised_before_information_is_pruned(prompt):
    """2. Revision comes first; retention is decided against the revised plan, not the old one."""
    ordering = re.search(r"^A\..*?^D\.", prompt, re.S | re.M)
    assert ordering, "the prompt no longer states an A-to-D order"
    steps = ordering.group(0).lower()
    assert steps.index("what the remaining plan should now be") < steps.index("still consumes")
    assert says(prompt, "revise first, then keep what the revision consumes")
    assert says(prompt, "now, and only now, work out which information the revised plan consumes")


def test_pruning_first_is_named_as_the_mistake_it_is(prompt):
    """2. The reason for the order is stated, because a rule without a reason gets optimized away."""
    assert says(prompt, "deciding what to keep before deciding what the plan is")


# --------------------------------------------------------------------------- errors

def test_a_generic_failure_sentence_is_refused(prompt):
    """3. "The previous call failed" is not an absorbed error."""
    assert says(prompt, "the previous call failed", "is not an absorbed error")


def test_an_actionable_error_must_change_the_plan(prompt):
    """3. When an error changes what happens next, the returned graph carries what recovery needs."""
    assert says(prompt, "when an error changes what has to happen next")
    for detail in ("which operation failed", "which operation replaces it",
                   "the exact parameter", "authentication requirement",
                   "granularity", "must not be done again"):
        assert says(prompt, detail), detail


def test_exact_details_may_not_be_generalized(prompt):
    """4. Operation names, parameters, values, identifiers and cursors stay exact."""
    assert says(prompt, "keep exact things exact")
    assert says(prompt, "copied as they were established")
    assert says(prompt, "the detail thrown away at the moment it became necessary")


def test_a_failure_consequence_does_not_stand_in_for_the_details(prompt):
    """3. The consequence and the exact facts are different nodes with different jobs."""
    assert says(prompt, "does not stand in for the details")
    assert says(prompt, "a sentence about the failure is not something a computation can use")


def test_the_information_kinds_are_the_existing_six(prompt):
    """3. No error, history, memory or observation kind is introduced."""
    for kind in ("fact", "constraint", "result", "contract", "runtime_reference",
                 "failure_consequence"):
        assert kind in prompt, kind
    lowered = prompt.lower()
    for forbidden in ("kind: error", "kind: memory", "kind: history", "kind: observation",
                      "error_message"):
        assert forbidden not in lowered, forbidden


# --------------------------------------------------------------------------- routes

def test_an_invalid_route_must_disappear_rather_than_be_annotated(prompt):
    """6. A closed route is removed or changed, not kept beside a warning."""
    assert says(prompt, "an invalid route disappears")
    assert says(prompt, "must not still contain it with a warning")
    for change in ("the operation", "the arguments", "a new prerequisite",
                   "different refinement children", "a different branch", "a different order",
                   "an added recovery step", "an added verification step"):
        assert says(prompt, change), change


def test_a_transient_failure_does_not_force_revision(prompt):
    """5. A call that stayed correct and failed temporarily may remain in the plan."""
    assert says(prompt, "only temporary is not a reason to replan")
    assert says(prompt, "the call was correct and the failure was transient")
    assert says(prompt, "do not invent a retry policy")


# --------------------------------------------------------------------------- resuming

def test_partial_progress_resumes_rather_than_restarts(prompt):
    """7. Completed work is not redone after compaction."""
    assert says(prompt, "continues rather than restarts")
    assert says(prompt, "the partial thing and the complete thing are two nodes")
    assert says(prompt, "the agent must not redo any of it")


# --------------------------------------------------------------------------- retention

def test_only_what_the_revised_plan_consumes_survives(prompt):
    """8. Retention follows the revised plan's consumers, and nothing else."""
    assert says(prompt, "keep what at least one remaining computation requires")
    assert says(prompt, "drop what only completed work used")
    assert says(prompt, "what only an invalidated branch used")


def test_the_sufficiency_criterion_is_in_the_prompt(prompt):
    """The seven questions the graph and handover must answer on their own."""
    assert says(prompt, "whether you absorbed enough")
    for question in ("what remaining objective is active", "what is already done",
                     "which route is closed", "the exact next recovery or continuation step",
                     "what must not be repeated"):
        assert says(prompt, question), question


def test_no_second_call_or_analysis_section_is_requested(prompt):
    """The answer is still one graph and nothing else."""
    assert says(prompt, "write only the graph")
    assert says(prompt, "the complete replacement graph, and nothing else")
    lowered = prompt.lower()
    for forbidden in ("step by step", "think aloud", "explain your reasoning",
                      "before the graph, write", "output a summary of"):
        assert forbidden not in lowered, forbidden


def test_no_scoring_or_ranking_mechanism_is_introduced(prompt):
    """Retention is a graph relation, not a judgement about importance."""
    lowered = prompt.lower()
    for forbidden in ("relevance score", "importance score", "importance weight",
                      "rank the information", "score each"):
        assert forbidden not in lowered, forbidden
