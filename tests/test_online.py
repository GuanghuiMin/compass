"""The online adapter: the two-phase transition, and what must not move between the phases.

The offline updater could treat a boundary as one event. Online it is two, because the host asks
for a handover and only afterwards splices it into the context the agent acts on. Everything here
defends the gap between them: the graph does not move when the handover is produced, no record is
committed until the splice returned, and an abort leaves both exactly as they were.

The model is a stub throughout. What is under test is the state machine around it.
"""

import json
from pathlib import Path

import pytest

from future_graph import ComputationNode as C, InformationNode as I
from future_graph import InformationKind as K, Relation as R, build
from future_graph.artifacts import ArtifactError, ModelCall, RevisionRecord
from future_graph.online import (
    ACCEPTED, EMPTY, REFUSED, LocalRevisionOptimizer, OnlineIntegrationError, continuation_of,
)
from future_graph.online_run import (
    PROVIDER_CALLS, Continuation, RepoIdentity, build_manifest, describe_repo,
    describe_response, environment_identity, finish_manifest, hashes_of, host_now, load_run,
    monotonic_seconds, openai_version, prepare_online_run, recording_adapter, write_json,
)
from future_graph.rendering import render
from future_graph.run import RunError
from future_graph.state_graph import StateGraph

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

EMPTY_REVISION = "BEGIN_REVISION\nEND_REVISION\n"
UNPARSABLE = "Nothing needs to change here.\n"


def stub(*answers):
    remaining = list(answers)
    calls = []

    def model(call: ModelCall) -> str:
        calls.append(call)
        answer = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    model.calls = calls
    return model


def optimizer(tmp_path, model, window=10, **kw):
    return LocalRevisionOptimizer(
        goal="the goal", rules="the rules", model=model,
        count_tokens=lambda text: len(text.split()), window=window,
        boundaries_dir=tmp_path / "boundaries", pending_dir=tmp_path / ".pending",
        config={"temperature": 0.0, "seed": 1}, **kw)


@pytest.fixture
def opt(tmp_path):
    return optimizer(tmp_path, stub(FIRST, SECOND, EMPTY_REVISION))


# --------------------------------------------------------------------------- the trigger

def test_the_trigger_is_the_hosts_and_measures_the_same_thing(opt):
    assert not opt.check_summarization_needed("one two three")
    assert opt.check_summarization_needed(" ".join(["word"] * 11))
    assert opt.check_summarization_needed(" ".join(["word"] * 6),
                                          prev_history_summary=" ".join(["word"] * 6))


def test_no_boundary_leaves_everything_alone(opt, tmp_path):
    before = opt.graph.to_snapshot()
    assert not opt.check_summarization_needed("short")
    assert opt.pending is None
    assert opt.graph.to_snapshot() == before
    assert list((tmp_path / "boundaries").glob("*")) == [] if (tmp_path / "boundaries").is_dir() \
        else True


# --------------------------------------------------------------------------- prepare

def test_the_first_boundary_starts_from_an_empty_graph_and_goes_through_the_updater(opt):
    assert opt.graph.computations == ()
    handover = opt.process(task="t", history="a slice", prev_history_summary=None)
    assert "Survey the seedlings" in handover
    assert "BEGIN_REVISION" in opt.model.calls[0].system
    assert "Return the whole graph every time" not in opt.model.calls[0].system


def test_preparing_does_not_move_the_graph(opt):
    before = opt.graph.to_snapshot()
    opt.process(task="t", history="a slice")
    assert opt.graph.to_snapshot() == before
    assert opt.graph.computations == ()
    assert opt.pending is not None
    assert opt.boundary_index == 0


def test_preparing_writes_only_a_pending_record(opt, tmp_path):
    opt.process(task="t", history="a slice")
    assert list((tmp_path / "boundaries").glob("*")) == []
    pending = list((tmp_path / ".pending").glob("*.pending"))
    assert len(pending) == 1
    RevisionRecord.from_dict(json.loads(pending[0].read_text()))


def test_a_second_prepare_before_a_commit_is_an_integration_error(opt):
    opt.process(task="t", history="a slice")
    with pytest.raises(OnlineIntegrationError, match="never committed"):
        opt.process(task="t", history="another slice")


# --------------------------------------------------------------------------- commit

def test_committing_moves_the_graph_and_the_record_together(opt, tmp_path):
    handover = opt.process(task="t", history="a slice")
    committed = opt.commit_pending()
    assert opt.graph.computations != ()
    assert render(opt.graph) == handover
    assert committed.status == ACCEPTED
    assert committed.path == tmp_path / "boundaries" / "boundary_000.json"
    assert committed.path.exists()
    assert list((tmp_path / ".pending").glob("*")) == []
    assert opt.pending is None
    assert opt.boundary_index == 1


def test_committing_without_a_prepared_transition_is_an_integration_error(opt):
    with pytest.raises(OnlineIntegrationError, match="no prepared transition"):
        opt.commit_pending()


def test_the_committed_graph_is_the_object_the_updater_produced(opt):
    opt.process(task="t", history="a slice")
    prepared = opt.pending.resulting_graph
    opt.commit_pending()
    assert opt.graph is prepared          # a rebind, not another transformation


def test_a_later_boundary_sees_the_graph_the_last_one_left(opt):
    opt.process(task="t", history="one")
    first = opt.commit_pending()
    opt.process(task="t", history="two", prev_history_summary=first.handover)
    second = opt.commit_pending()
    assert second.boundary_index == 1
    assert "Survey the seedlings" not in render(opt.graph)
    assert "Register every seedling" in render(opt.graph)


# --------------------------------------------------------------------------- abort

def test_aborting_leaves_the_graph_and_removes_the_pending_record(opt, tmp_path):
    before = opt.graph.to_snapshot()
    opt.process(task="t", history="a slice")
    aborted = opt.abort_pending()
    assert aborted is not None
    assert opt.graph.to_snapshot() == before
    assert opt.pending is None
    assert list((tmp_path / ".pending").glob("*")) == []
    assert list((tmp_path / "boundaries").glob("*")) == []


def test_aborting_nothing_is_allowed(opt):
    assert opt.abort_pending() is None


def test_an_abort_does_not_consume_the_boundary_index(tmp_path):
    """A run stops after an abort, so this is not a retry path. It is here because the index
    advancing on a transition that never happened would misnumber every record after it."""
    opt = optimizer(tmp_path, stub(FIRST, FIRST))
    opt.process(task="t", history="a slice")
    opt.abort_pending()
    assert opt.boundary_index == 0
    opt.process(task="t", history="a slice")
    assert opt.pending.boundary_index == 0
    assert opt.commit_pending().path.name == "boundary_000.json"


# --------------------------------------------------------------------------- the three outcomes

def test_an_accepted_boundary_returns_the_new_handover(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST))
    handover = opt.process(task="t", history="a slice")
    opt.commit_pending()
    assert handover == render(opt.graph)
    assert handover != "NOTHING REMAINS"


def test_an_empty_revision_keeps_the_graph_and_the_handover_but_is_a_boundary(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST, EMPTY_REVISION))
    opt.process(task="t", history="one")
    first = opt.commit_pending()
    handover = opt.process(task="t", history="two", prev_history_summary=first.handover)
    second = opt.commit_pending()
    assert second.status == EMPTY
    assert handover == first.handover
    assert render(opt.graph) == first.handover
    assert second.path.exists()          # still a committed boundary


def test_a_refused_revision_keeps_the_graph_and_returns_the_previous_handover(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST, UNPARSABLE))
    opt.process(task="t", history="one")
    first = opt.commit_pending()
    before = opt.graph.to_snapshot()
    handover = opt.process(task="t", history="two", prev_history_summary=first.handover)
    second = opt.commit_pending()
    assert second.status == REFUSED
    assert handover == first.handover
    assert opt.graph.to_snapshot() == before
    assert second.record.parse_errors


def test_a_refusal_consumes_its_slice(tmp_path):
    """The agent executed those steps; only their future textual representation is gone."""
    opt = optimizer(tmp_path, stub(FIRST, UNPARSABLE, EMPTY_REVISION))
    opt.process(task="t", history="one")
    first = opt.commit_pending()
    opt.process(task="t", history="the refused slice", prev_history_summary=first.handover)
    opt.commit_pending()
    opt.process(task="t", history="the next slice", prev_history_summary=first.handover)
    third = opt.commit_pending()
    assert third.delta_h == "the next slice"
    assert "the refused slice" not in opt.model.calls[2].user
    assert "the next slice" in opt.model.calls[2].user


def test_a_refusal_is_recorded_as_a_method_failure_and_never_retried(tmp_path):
    opt = optimizer(tmp_path, stub(UNPARSABLE, FIRST))
    opt.process(task="t", history="one")
    committed = opt.commit_pending()
    assert committed.status == REFUSED
    assert not committed.record.accepted
    assert len(opt.model.calls) == 1


# --------------------------------------------------------------------------- host agreement

def test_a_host_summary_that_is_not_the_active_handover_is_refused_before_the_call(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST, SECOND))
    opt.process(task="t", history="one")
    opt.commit_pending()
    with pytest.raises(OnlineIntegrationError, match="not the handover"):
        opt.process(task="t", history="two", prev_history_summary="something else entirely")
    assert len(opt.model.calls) == 1          # the second call never happened
    assert opt.pending is None


def test_a_summary_before_the_first_boundary_means_something_else_compacted(opt):
    with pytest.raises(OnlineIntegrationError, match="before any boundary committed"):
        opt.process(task="t", history="one", prev_history_summary="a summary from nowhere")
    assert opt.model.calls == []


@pytest.mark.parametrize("empty", [None, ""])
def test_the_first_boundary_accepts_the_hosts_empty_summary(tmp_path, empty):
    opt = optimizer(tmp_path, stub(FIRST))
    opt.process(task="t", history="one", prev_history_summary=empty)
    assert opt.pending is not None


# --------------------------------------------------------------------------- what it must not see

def test_the_retained_full_history_never_reaches_the_updater(opt):
    """The host still holds the uncompacted history. Reading it would give the updater evidence
    the method says it does not have."""
    secret = "THIS IS THE DISCARDED FULL HISTORY"
    opt.process(task="t", history="a slice",
                raw_history=[{"role": "user", "content": secret}],
                opt_args={"action_history": [secret], "preserved_turns": [{"content": secret}]})
    assert secret not in opt.model.calls[0].user
    assert secret not in opt.model.calls[0].system


def test_the_updater_sees_exactly_the_four_inputs(opt):
    opt.process(task="t", history="the slice itself")
    user = opt.model.calls[0].user
    assert "the goal" in user and "the rules" in user and "the slice itself" in user
    assert "BEGIN_PREVIOUS_GRAPH" in user
    assert user.index("ORIGINAL_GOAL") < user.index("FIXED_RULES") \
        < user.index("PREVIOUS_GRAPH") < user.index("DELTA_H")


def test_the_goal_the_host_passes_as_task_does_not_override_the_captured_goal(opt):
    """The goal and rules are captured once at task start; the host's per-call `task` is not a
    second channel into the updater."""
    opt.process(task="a different instruction", history="a slice")
    assert "a different instruction" not in opt.model.calls[0].user


# --------------------------------------------------------------------------- one graph per episode

def test_two_optimizers_do_not_share_a_graph(tmp_path):
    first = optimizer(tmp_path / "a", stub(FIRST))
    second = optimizer(tmp_path / "b", stub(FIRST))
    first.process(task="t", history="one")
    first.commit_pending()
    assert first.graph.computations != ()
    assert second.graph.computations == ()
    assert second.boundary_index == 0


def test_a_record_is_never_replaced(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST, FIRST))
    opt.process(task="t", history="one")
    opt.commit_pending()
    opt.boundary_index = 0          # a defect that would otherwise overwrite boundary 000
    with pytest.raises(OnlineIntegrationError, match="never replaced"):
        opt.process(task="t", history="two", prev_history_summary=opt.handover())


# --------------------------------------------------------------------------- retry, unchanged

def test_an_operational_failure_is_retried_and_does_not_prepare_twice(tmp_path, monkeypatch):
    from future_graph.adapter import EmptyModelCompletion
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _s: None)
    opt = optimizer(tmp_path, stub(EmptyModelCompletion("nothing"), FIRST))
    opt.process(task="t", history="one")
    assert [a[1] for a in opt.pending.record.attempts] == ["EmptyModelCompletion", "completion"]
    assert len(opt.model.calls) == 2
    assert opt.model.calls[0].user == opt.model.calls[1].user
    assert opt.graph.computations == ()


def test_a_provider_that_never_answered_leaves_nothing_prepared(tmp_path, monkeypatch):
    from future_graph.adapter import EmptyModelCompletion
    from future_graph.retry import ExhaustedAttempts
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _s: None)
    opt = optimizer(tmp_path, stub(*[EmptyModelCompletion("nothing")] * 3))
    with pytest.raises(ExhaustedAttempts):
        opt.process(task="t", history="one")
    assert opt.pending is None
    assert opt.graph.computations == ()
    assert list((tmp_path / ".pending").glob("*")) == []


def test_the_online_client_is_built_with_no_hidden_retries(monkeypatch):
    """The updater's own adapter, not the host's client: the host's builds an OpenAI client with
    the SDK's default two retries, which would compose with ours into invisible extra requests."""
    from future_graph import adapter as adapter_module

    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv(adapter_module.BASE_URL_VAR, adapter_module.BASE_URL)
    monkeypatch.setenv(adapter_module.MODEL_VAR, adapter_module.MODEL)
    monkeypatch.setenv(adapter_module.API_KEY_VAR, "a-key")
    built = adapter_module.from_environment(client_factory=factory)
    assert seen["max_retries"] == 0
    assert seen["base_url"] == adapter_module.BASE_URL
    assert seen["timeout"] == adapter_module.TIMEOUT_S
    assert built.model == adapter_module.MODEL


def test_the_frozen_call_reaches_the_model_unchanged(tmp_path):
    opt = optimizer(tmp_path, stub(FIRST))
    opt.process(task="t", history="one")
    assert opt.model.calls[0].config == (("seed", 1), ("temperature", 0.0))


# --------------------------------------------------------------------------- artifacts

def repos_of(tmp_path):
    return {"compass-v2": _here(), "trace-v2": _here()}


def _here():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def test_a_run_pins_both_repositories(tmp_path):
    identity = describe_repo("compass-v2", _here())
    assert identity.commit and len(identity.commit) == 40
    assert identity.branch
    assert isinstance(identity.clean, bool)


def test_a_dirty_repository_refuses_the_run_before_anything_is_sent(tmp_path, monkeypatch):
    def dirty(name, path):
        return RepoIdentity(name=name, path=str(path), remote="r", branch="b",
                            commit="0" * 40, clean=(name != "trace-v2"))

    monkeypatch.setattr("future_graph.online_run.describe_repo", dirty)
    with pytest.raises(RunError, match="trace-v2"):
        prepare_online_run("a-run", {"compass-v2": _here(), "trace-v2": _here()},
                           artifact_root=tmp_path)
    assert not (tmp_path / "a-run").exists()


def test_a_prepared_run_claims_its_directories(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    assert prepared.boundaries_dir.is_dir()
    assert prepared.continuations_dir.is_dir()
    assert prepared.pending_dir.is_dir()
    assert prepared.openai_version == openai_version()


def test_a_run_id_that_could_escape_the_artifact_root_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("future_graph.online_run.describe_repo",
                        lambda name, path: RepoIdentity(name, str(path), "r", "b", "0" * 40, True))
    with pytest.raises(RunError, match="single path component"):
        prepare_online_run("../escape", {"a": _here(), "b": _here()}, artifact_root=tmp_path)


def _clean_run(tmp_path, monkeypatch, run_id="a-run"):
    monkeypatch.setattr("future_graph.online_run.describe_repo",
                        lambda name, path: RepoIdentity(name, str(path), "r", "b", "0" * 40, True))
    return prepare_online_run(run_id, {"compass-v2": _here(), "trace-v2": _here()},
                              artifact_root=tmp_path)


def _finished(prepared, boundaries, continuations=0, status="completed"):
    manifest = build_manifest(
        prepared, task_id="042a9fc_1", split="test_normal", window=4096, preserved_turns=1,
        max_steps=30, tasklist={"path": "t.jsonl", "sha256": "0" * 64},
        downstream={"model": "m"}, updater={"model": "u"},
        hashes=hashes_of("system", "rules", "prompt"),
        environment=environment_identity(
            task_id="042a9fc_1", split="test_normal", instruction="do the thing",
            experiment_name="x", command=["run", "--task", "042a9fc_1"]),
        started_monotonic=monotonic_seconds())
    manifest.update({"boundaries": boundaries, "continuations": continuations})
    finish_manifest(manifest, status)
    write_json(prepared.manifest_path, manifest, overwrite=True)
    return manifest


def test_a_run_reads_back(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    opt = LocalRevisionOptimizer(
        goal="g", rules="r", model=stub(FIRST), count_tokens=lambda t: len(t.split()),
        window=1, boundaries_dir=prepared.boundaries_dir, pending_dir=prepared.pending_dir)
    opt.process(task="t", history="one")
    committed = opt.commit_pending()
    write_json(continuation_path_of(prepared, 0), Continuation(
        boundary_index=0, step_index=3, first_post_compaction_decision=True,
        handover_present=True, messages=[{"role": "user", "content": committed.handover}],
        system_message="sys", reasoning="because", tool_calls=[], executed_code="print(1)",
        observation="1", task_completed=False,
        boundary_artifact=committed.path.name).to_dict(), overwrite=False)
    _finished(prepared, boundaries=1, continuations=1)

    run = load_run(prepared.run_dir)
    assert len(run.boundaries) == 1 and len(run.continuations) == 1
    assert run.continuations[0].handover_present
    assert len(run.repos) == 2


def continuation_path_of(prepared, index):
    from future_graph.online_run import continuation_path
    return continuation_path(prepared, index)


def test_a_completed_run_with_a_prepared_boundary_left_over_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    _finished(prepared, boundaries=0)
    (prepared.pending_dir / "boundary_000.json.pending").write_text("{}")
    with pytest.raises(ArtifactError, match="uncommitted"):
        load_run(prepared.run_dir)


def test_a_manifest_that_miscounts_its_boundaries_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    _finished(prepared, boundaries=2)
    with pytest.raises(ArtifactError, match="counts 2 boundaries"):
        load_run(prepared.run_dir)


def test_a_manifest_naming_one_repository_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    manifest["repos"] = manifest["repos"][:1]
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="pins the updater and the host"):
        load_run(prepared.run_dir)


def test_a_manifest_cannot_be_read_as_another_method(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    for method, operator in (("openclaw", "local_revision"), ("future_graph_v1", "summary"),
                             ("compass_v4", "compass_v1")):
        manifest["method"], manifest["operator"] = method, operator
        write_json(prepared.manifest_path, manifest, overwrite=True)
        with pytest.raises(ArtifactError, match="is not this method"):
            load_run(prepared.run_dir)


def test_a_continuation_that_is_missing_a_field_is_refused():
    raw = Continuation(0, 1, True, True, [], "s", "r", [], "c", "o", False, "b").to_dict()
    del raw["handover_present"]
    with pytest.raises(ArtifactError, match="missing handover_present"):
        Continuation.from_dict(raw)


# --------------------------------------------------------------------------- clocks

def test_the_host_clock_and_the_duration_survive_a_frozen_episode(tmp_path, monkeypatch):
    """The first online run recorded start == finish == 2023-05-18 and an elapsed of -101287824.8,
    because a host epoch was reduced by a simulated one. AppWorld freezes `time.time`,
    `datetime.now`, `monotonic` and `perf_counter` alike; only `clock_gettime` still moves."""
    freezegun = pytest.importorskip("freezegun")
    prepared = _clean_run(tmp_path, monkeypatch)
    with freezegun.freeze_time("2023-05-18 05:00:00"):
        started = monotonic_seconds()
        manifest = build_manifest(
            prepared, task_id="t", split="s", window=4096, preserved_turns=1, max_steps=30,
            tasklist={}, downstream={}, updater={}, hashes=hashes_of("a", "b", "c"),
            environment=environment_identity(task_id="t", split="s", instruction="i",
                                             experiment_name="e", command=["c"]),
            started_monotonic=started)
        import time as _time
        _time.sleep(0.05)
        finish_manifest(manifest, "completed")

    assert manifest["elapsed_s"] >= 0
    assert manifest["elapsed_s"] < 60
    assert not manifest["host_started_at"].startswith("2023-05-18")
    assert not manifest["host_finished_at"].startswith("2023-05-18")
    assert "started_at" not in manifest and "finished_at" not in manifest


def test_a_duration_is_never_a_host_epoch_minus_a_simulated_one(tmp_path, monkeypatch):
    freezegun = pytest.importorskip("freezegun")
    prepared = _clean_run(tmp_path, monkeypatch)
    started = monotonic_seconds()                       # outside the freeze, as a real run does
    manifest = build_manifest(
        prepared, task_id="t", split="s", window=4096, preserved_turns=1, max_steps=30,
        tasklist={}, downstream={}, updater={}, hashes=hashes_of("a", "b", "c"),
        environment=environment_identity(task_id="t", split="s", instruction="i",
                                         experiment_name="e", command=["c"]),
        started_monotonic=started)
    with freezegun.freeze_time("2023-05-18 05:00:00"):  # and finished inside it, as it did
        finish_manifest(manifest, "completed")
    assert manifest["elapsed_s"] >= 0


def test_a_completed_run_with_a_negative_duration_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    manifest["elapsed_s"] = -101287824.8
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="negative time"):
        load_run(prepared.run_dir)


def test_a_completed_run_that_finished_before_it_started_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    manifest["host_started_at"], manifest["host_finished_at"] = \
        manifest["host_finished_at"], manifest["host_started_at"]
    manifest["host_started_at"] = "2030-01-01T00:00:00+00:00"
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="finished before it started"):
        load_run(prepared.run_dir)


@pytest.mark.parametrize("field", ["host_started_at", "host_finished_at", "elapsed_s"])
def test_a_completed_run_missing_a_timing_field_is_refused(tmp_path, monkeypatch, field):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    manifest[field] = None
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match=field):
        load_run(prepared.run_dir)


def test_the_first_online_run_still_reads_back_and_says_its_timing_cannot_be_trusted():
    """It is a real record of a real run and is not edited to suit a later schema. What it gets
    instead is a reader that knows which instrumentation produced it."""
    run_dir = Path(__file__).resolve().parents[1] / "artifacts" / "online"
    runs = sorted(p for p in run_dir.glob("*") if (p / "manifest.json").exists()) \
        if run_dir.is_dir() else []
    if not runs:
        pytest.skip("no online run has been committed yet")
    first = load_run(runs[0])
    assert first.manifest["status"] == "completed"
    if first.instrumentation == 1:
        assert not first.timing_is_trustworthy
        assert first.manifest["elapsed_s"] < 0          # the defect, preserved as it happened
    else:
        assert first.timing_is_trustworthy


def test_an_unfinished_run_is_not_held_to_the_completed_checks(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0, status="integration_failure")
    manifest["elapsed_s"] = None
    write_json(prepared.manifest_path, manifest, overwrite=True)
    assert load_run(prepared.run_dir).manifest["status"] == "integration_failure"


# --------------------------------------------------------------------------- provider metadata

class FakeResponse:
    def __init__(self, finish_reason="stop"):
        self.id = "resp-1"
        self.model = "minimax-m3"
        self.created = 1
        self.system_fingerprint = "fp"
        self.choices = [type("C", (), {"finish_reason": finish_reason,
                                       "message": type("M", (), {"content": "text"})()})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3,
                                    "total_tokens": 13, "completion_tokens_details": None})()


def test_the_providers_account_of_a_call_is_written_down(tmp_path):
    described = describe_response(FakeResponse(finish_reason="length"))
    assert described["finish_reason"] == "length"
    assert described["response_id"] == "resp-1"
    assert described["completion_tokens"] == 3
    assert described["total_tokens"] == 13


def test_a_response_that_carries_none_of_it_is_still_describable():
    described = describe_response(object())
    assert set(described) >= {"response_id", "finish_reason", "completion_tokens"}
    assert all(v is None for v in described.values())


def test_recording_wraps_the_client_and_not_the_request(monkeypatch):
    """The request is still built by the validated adapter, so there is no second copy of it to
    drift, and `max_retries=0` is the same client's."""
    from future_graph import adapter as adapter_module

    sent = {}

    class Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return FakeResponse()

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()
            self.max_retries = 0

    base = adapter_module.Adapter(client=Client(), model="minimax-m3")
    recording, calls = recording_adapter(base)
    text = recording(ModelCall(system="sys", user="usr", config=(("temperature", 0.0),)))

    assert text == "text"
    assert sent["model"] == "minimax-m3"
    assert sent["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "usr"}]
    assert sent["temperature"] == 0.0
    assert recording.client.max_retries == 0           # reached through the pass-through
    assert len(calls) == 1 and calls[0]["finish_reason"] == "stop"
    assert calls[0]["elapsed_s"] >= 0


def test_a_failed_call_is_recorded_as_a_failed_call():
    from future_graph import adapter as adapter_module

    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("the provider hung up")

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()

    recording, calls = recording_adapter(adapter_module.Adapter(client=Client(), model="m"))
    with pytest.raises(RuntimeError):
        recording(ModelCall(system="s", user="u"))
    assert len(calls) == 1 and "hung up" in calls[0]["error"]


def test_recording_does_not_change_what_counts_as_operational():
    """Metadata is written down and interpreted nowhere. A completion that ends mid-block is still
    the model's answer, and deciding otherwise from a finish_reason is a separate question."""
    from future_graph.retry import is_operational

    class Truncated(Exception):
        pass

    assert not is_operational(Truncated())
    source = (Path(__file__).resolve().parents[1]
              / "src" / "future_graph" / "online_run.py").read_text()
    assert "finish_reason" in source
    for word in ("is_operational", "ExhaustedAttempts", "EmptyModelCompletion"):
        assert word not in source


def test_provider_calls_must_match_the_attempts_that_were_made(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    opt = LocalRevisionOptimizer(
        goal="g", rules="r", model=stub(FIRST), count_tokens=lambda t: len(t.split()),
        window=1, boundaries_dir=prepared.boundaries_dir, pending_dir=prepared.pending_dir)
    opt.process(task="t", history="one")
    opt.commit_pending()
    write_json(prepared.run_dir / PROVIDER_CALLS, [{"finish_reason": "stop"},
                                                   {"finish_reason": "stop"}], overwrite=True)
    _finished(prepared, boundaries=1)
    with pytest.raises(ArtifactError, match="2 provider calls and 1 attempts"):
        load_run(prepared.run_dir)


# --------------------------------------------------------------------------- task identity

def test_the_environment_identity_names_the_installation_and_the_task_text():
    identity = environment_identity(task_id="042a9fc_1", split="test_normal",
                                    instruction="update the playlist",
                                    experiment_name="fg_run_042a9fc_1",
                                    command=["python", "runner.py", "--task", "042a9fc_1"])
    assert identity["task_id"] == "042a9fc_1"
    assert identity["instruction_sha256"] == \
        __import__("hashlib").sha256(b"update the playlist").hexdigest()
    assert identity["instruction_bytes"] == 19
    assert identity["command"][-1] == "042a9fc_1"
    assert "appworld_version" in identity          # None where the package is not installed


def test_a_different_task_text_is_a_different_identity():
    first = environment_identity(task_id="t", split="s", instruction="do A",
                                 experiment_name="e", command=[])
    second = environment_identity(task_id="t", split="s", instruction="do B",
                                  experiment_name="e", command=[])
    assert first["instruction_sha256"] != second["instruction_sha256"]


def test_a_completed_run_without_environment_identity_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    del manifest["environment"]
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="environment identity is missing"):
        load_run(prepared.run_dir)


def test_a_manifest_whose_environment_names_another_task_is_refused(tmp_path, monkeypatch):
    prepared = _clean_run(tmp_path, monkeypatch)
    manifest = _finished(prepared, boundaries=0)
    manifest["environment"]["task_id"] = "some_other_task"
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="different task"):
        load_run(prepared.run_dir)


def test_the_handover_check_is_a_reading_of_the_messages_that_were_sent():
    handover = "REFINED PLAN OVERVIEW\n[c1] do the thing"
    assert continuation_of([{"role": "user", "content": f"task\n\n{handover}\n\nend"}], handover)
    assert not continuation_of([{"role": "user", "content": "task"}], handover)
    assert not continuation_of([{"role": "user", "content": "task"}], "")
