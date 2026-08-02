"""The recurrent runner: what it records, and the two guarantees it checks rather than trusts.

A chain is where the interesting failures live. A boundary that half-applied, a refused slice that
came back, a graph that drifted after a refusal -- none of these show up in a single boundary, and
all of them would quietly change what the results mean. So the runner asserts them at runtime and
these tests assert that it does.
"""

import json

import pytest

from future_graph.artifacts import ModelCall, RevisionRecord, prompt_sha
from future_graph.episodes import Episode, PreflightInputs
from future_graph.run import OperationalFailure, PreparedRun
from future_graph.revision_run import measure, replay_revision
from future_graph.update import load_prompt

EMPTY = "BEGIN_REVISION\nEND_REVISION\n"
UNPARSABLE = "Nothing needs to change here.\n"

FIRST = """BEGIN_REVISION
ADD
COMPUTATION +survey
description: Survey the seedlings
operation: nursery.survey
produces: +roster
END_COMPUTATION
COMPUTATION +register
description: Register every seedling
argument roster = @+roster
END_COMPUTATION
INFORMATION +roster
kind: result
available: false
description: The roster of seedlings
END_INFORMATION
END_ADD
END_REVISION
"""

SECOND = """BEGIN_REVISION
COMPLETE c1
NOW_AVAILABLE i1
kind: fact
description: The roster of twelve seedlings
END_NOW_AVAILABLE
END_COMPLETE
END_REVISION
"""


@pytest.fixture(autouse=True)
def no_real_waiting(monkeypatch):
    """Retry backoff is real seconds in a run and must not be real seconds in the suite."""
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _seconds: None)


class Stub:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[ModelCall] = []

    def __call__(self, call: ModelCall) -> str:
        self.calls.append(call)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def inputs_of(*slices, episode_id="ep") -> PreflightInputs:
    return PreflightInputs(
        episodes=(Episode(id=episode_id, goal="the goal", rules="the rules",
                          boundaries=tuple(enumerate(slices))),),
        input_manifest_sha256="0" * 64, source_kind="reconstructed",
        historical_byte_identity_verified=False, sampling_is_deterministic=False)


def prepared(run_dir) -> PreparedRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    return PreparedRun(run_dir=run_dir, commit_sha="0" * 40,
                       prompt_sha=prompt_sha(load_prompt()), openai_version="1.60.1")


def rows_of(run_dir, episode_id="ep"):
    summary = json.loads((run_dir / f"episode_{episode_id}" / "summary.json").read_text())
    return summary["boundaries"]


# --------------------------------------------------------------------------- a chain

def test_a_chain_runs_every_boundary_in_order(tmp_path):
    run_dir = tmp_path / "run"
    model = Stub(FIRST, SECOND, EMPTY)
    manifest = replay_revision(inputs_of("one", "two", "three"), model, prepared(run_dir))
    assert manifest["status"] == "completed"
    assert manifest["completed_boundaries"] == {"ep": 3}
    assert manifest["accepted_boundaries"] == {"ep": 3}
    assert len(model.calls) == 3
    assert manifest["operator"] == "local_revision"


def test_each_boundary_sees_the_graph_the_last_one_left(tmp_path):
    run_dir = tmp_path / "run"
    model = Stub(FIRST, SECOND, EMPTY)
    replay_revision(inputs_of("one", "two", "three"), model, prepared(run_dir))
    assert "BEGIN_GRAPH\n\nEND_GRAPH" in model.calls[0].user
    assert "Survey the seedlings" in model.calls[1].user
    # The second boundary completed the survey, so the third must not still be shown it.
    assert "Survey the seedlings" not in model.calls[2].user
    assert "Register every seedling" in model.calls[2].user


def test_a_record_is_written_for_every_boundary_and_reads_back(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, SECOND), prepared(run_dir))
    for index in (0, 1):
        path = run_dir / "episode_ep" / f"boundary_{index:03d}.json"
        RevisionRecord.from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------------- refusal in a chain

def test_a_refused_boundary_does_not_stop_the_chain(tmp_path):
    """The point of running a chain: the previous graph carries on and the next boundary is tried
    against the state as it stood."""
    run_dir = tmp_path / "run"
    manifest = replay_revision(inputs_of("one", "two", "three"),
                               Stub(FIRST, UNPARSABLE, EMPTY), prepared(run_dir))
    assert manifest["status"] == "completed"
    assert manifest["completed_boundaries"] == {"ep": 3}
    assert manifest["accepted_boundaries"] == {"ep": 2}


def test_a_refusal_records_that_the_graph_was_preserved_and_the_slice_dropped(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, UNPARSABLE), prepared(run_dir))
    first, second = rows_of(run_dir)
    assert first["accepted"] and first["previous_graph_preserved_byte_identically"] is None
    assert not second["accepted"]
    assert second["previous_graph_preserved_byte_identically"] is True
    assert second["delta_h_discarded"] is True
    assert second["refusal"]["parse_errors"]


def test_a_refused_slice_that_came_back_stops_the_run(tmp_path):
    """Replaying a refused slice would hand the method an attempt the schedule never gave it, and
    nothing in the records would show it."""
    run_dir = tmp_path / "run"
    with pytest.raises(OperationalFailure, match="refused earlier"):
        replay_revision(inputs_of("a unique slice", "a unique slice again"),
                        Stub(UNPARSABLE, EMPTY), prepared(run_dir))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "operational_failure"
    assert manifest["operational_failure"]["layer"] == "delta_h_identity"


# --------------------------------------------------------------------------- what is measured

def test_the_summary_carries_the_sizes_the_question_is_about(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, SECOND), prepared(run_dir))
    first, second = rows_of(run_dir)
    assert first["sizes"]["previous_graph_bytes"] > 0
    assert first["sizes"]["revision_bytes"] == len(FIRST.encode("utf-8"))
    assert second["sizes"]["revision_bytes"] == len(SECOND.encode("utf-8"))
    assert second["sizes"]["handover_bytes"] > 0


def test_the_summary_counts_what_was_preserved(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, SECOND), prepared(run_dir))
    second = rows_of(run_dir)[1]
    nodes = second["nodes"]
    assert nodes["previous_total"] == 3                 # two computations and the roster
    assert nodes["affected_roots"] == 1
    assert nodes["preserved"] == nodes["previous_total"] - nodes["replaced_or_removed"]
    assert nodes["resulting_computations"] == 1


def test_the_summary_separates_the_model_s_work_from_the_code_s(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, SECOND), prepared(run_dir))
    second = rows_of(run_dir)[1]
    assert second["model_authored"]["removed_regions"] == ["c1"]
    assert any(change[0] == "became_available"
               for change in second["code_owned"]["completion_changes"])


def test_an_empty_revision_is_recorded_as_the_no_op_it_is(tmp_path):
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one", "two"), Stub(FIRST, EMPTY), prepared(run_dir))
    second = rows_of(run_dir)[1]
    assert second["accepted"] and second["empty_revision"]
    assert second["nodes"]["replaced_or_removed"] == 0
    assert second["nodes"]["preserved"] == second["nodes"]["previous_total"]


def test_operational_attempts_are_counted_per_boundary(tmp_path):
    from future_graph.adapter import EmptyModelCompletion
    run_dir = tmp_path / "run"
    model = Stub(FIRST, EmptyModelCompletion("nothing"), EMPTY)
    replay_revision(inputs_of("one", "two"), model, prepared(run_dir))
    first, second = rows_of(run_dir)
    assert first["operational_attempts"] == 0
    assert second["operational_attempts"] == 1
    assert [a[1] for a in second["attempts"]] == ["EmptyModelCompletion", "completion"]


def test_a_provider_that_never_answered_stops_the_run_and_says_so(tmp_path):
    from future_graph.adapter import EmptyModelCompletion
    from future_graph.retry import ExhaustedAttempts
    run_dir = tmp_path / "run"
    model = Stub(FIRST, *[EmptyModelCompletion("nothing")] * 3)
    with pytest.raises(ExhaustedAttempts):
        replay_revision(inputs_of("one", "two"), model, prepared(run_dir))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "operational_failure"
    assert manifest["operational_failure"]["layer"] == "model"
    assert len(manifest["operational_failure"]["attempts"]) == 3
    # The boundary that did work is still recorded; the one that never answered is not.
    assert (run_dir / "episode_ep" / "boundary_000.json").exists()
    assert not (run_dir / "episode_ep" / "boundary_001.json").exists()


def test_a_prompt_that_is_not_the_declared_one_stops_the_run(tmp_path):
    run_dir = tmp_path / "run"
    settled = PreparedRun(run_dir=run_dir, commit_sha="0" * 40, prompt_sha="a" * 64,
                          openai_version="1.60.1")
    run_dir.mkdir(parents=True)
    with pytest.raises(OperationalFailure, match="does not hash"):
        replay_revision(inputs_of("one"), Stub(EMPTY), settled)


def test_a_run_never_writes_over_one_that_already_happened(tmp_path):
    """Records are immutable, and the cheapest way to keep them so is to refuse the directory
    before a single call is made."""
    run_dir = tmp_path / "run"
    replay_revision(inputs_of("one"), Stub(FIRST), prepared(run_dir))
    model = Stub(FIRST)
    with pytest.raises(FileExistsError):
        replay_revision(inputs_of("one"), model, prepared(run_dir))
    assert model.calls == []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "operational_failure"


def test_a_boundary_record_is_never_replaced(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "episode_ep").mkdir(parents=True)
    (run_dir / "episode_ep" / "boundary_000.json").write_text("{}")
    with pytest.raises(FileExistsError):
        replay_revision(inputs_of("one"), Stub(FIRST), prepared(run_dir))


# --------------------------------------------------------------------------- measure itself

def test_measure_reports_information_created_and_collected_in_one_boundary(tmp_path):
    run_dir = tmp_path / "run"
    stillborn = """BEGIN_REVISION
ADD
COMPUTATION +call
description: Do the thing
operation: nursery.do
produces: +receipt
END_COMPUTATION
INFORMATION +receipt
kind: result
available: false
description: The receipt the call returns
END_INFORMATION
END_ADD
END_REVISION
"""
    replay_revision(inputs_of("one"), Stub(stillborn), prepared(run_dir))
    row = rows_of(run_dir)[0]
    assert row["accepted"]
    assert row["newly_created_then_collected"] == ["i1"]
    assert row["code_owned"]["collected"] == ["i1"]


def test_measure_is_a_pure_reading_of_a_record(tmp_path):
    """It must not be able to change anything, so a summary can never disagree with its records."""
    from future_graph.state_graph import StateGraph
    from future_graph.update import update_graph
    previous = StateGraph()
    result = update_graph("g", "r", previous, "d", Stub(FIRST), {"model": "m"})
    before = json.dumps(result.graph.to_snapshot(), sort_keys=True)
    row = measure(previous, "d", result.record, result.graph)
    assert json.dumps(result.graph.to_snapshot(), sort_keys=True) == before
    assert row["accepted"] is True
