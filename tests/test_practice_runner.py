"""What the practice runner is measuring, pinned so it cannot quietly go back to measuring less.

The runner exists to see a *transition*: a plan that was already committed to, plus one new
interaction, and what the updater does with the pair. Two mistakes would turn it back into a test of
one-shot construction while every record still looked fine -- starting a case from an empty graph,
and replaying the whole prefix as the new evidence. Both are cheap to make and invisible in the
output, so both are refused here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from future_graph.state_graph import StateGraph
from future_graph.validation import validate

REPO = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_practice_cases", REPO / "scripts" / "run_practice_cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()
CASE_IDS = sorted(RUNNER.PREVIOUS_GRAPHS)


def cases():
    return RUNNER.load_cases(RUNNER.CASES_PATH, CASE_IDS)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_no_case_starts_from_an_empty_graph(case_id):
    """An empty previous graph would measure construction and be recorded as a transition."""
    previous = RUNNER.PREVIOUS_GRAPHS[case_id]()
    assert isinstance(previous, StateGraph)
    assert len(previous) > 0
    assert previous.computations, f"{case_id}: a previous plan with no computation is not a plan"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_every_previous_graph_is_a_graph_the_validator_accepts(case_id):
    """A fixture the system would refuse could never have been the state before the step."""
    assert validate(RUNNER.PREVIOUS_GRAPHS[case_id]()) == ()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_the_slice_is_only_the_triggering_step(case_id):
    """Earlier steps are represented in the previous graph; replaying them would double the
    evidence and hand the model back the reconstruction problem."""
    selected, _ = RUNNER.load_cases(RUNNER.CASES_PATH, [case_id])
    case = selected[0]
    delta = RUNNER.build_slice(case)
    last = case["prefix"][-1]
    for key in ("reasoning", "code", "observation"):
        if last.get(key):
            assert last[key] in delta, f"{case_id}: the triggering {key} is missing"
    for earlier in case["prefix"][:-1]:
        for key in ("reasoning", "code", "observation"):
            if earlier.get(key):
                assert earlier[key] not in delta, f"{case_id}: an earlier {key} leaked in"


def test_a_multi_step_case_really_does_drop_its_earlier_steps():
    """synthetic_02 is the only case with a prefix longer than one step, so it is the only one
    where the parametrized check above can actually fail. Named separately so that is visible."""
    selected, _ = RUNNER.load_cases(RUNNER.CASES_PATH, ["synthetic_02"])
    case = selected[0]
    assert len(case["prefix"]) == 2
    delta = RUNNER.build_slice(case)
    assert "show_delivery_contents" not in delta
    assert "create_entry(seedling_id=91001" in delta


def test_a_case_without_a_fixture_is_refused_rather_than_started_from_nothing():
    with pytest.raises(SystemExit, match="no previous-graph fixture"):
        RUNNER.load_cases(RUNNER.CASES_PATH, ["synthetic_01"])


def test_the_runner_holds_no_expected_answers():
    """A fixture is the plan before the step, not the graph the model ought to return.

    Checked on the module's surface rather than its prose: each entry is a graph and nothing is
    paired with it, and no function here offers to judge an outcome.
    """
    for case_id, make in RUNNER.PREVIOUS_GRAPHS.items():
        assert isinstance(make(), StateGraph), case_id
    judging = [name for name in dir(RUNNER)
               if any(word in name.lower() for word in ("expect", "score", "grade", "correct"))]
    assert judging == []
