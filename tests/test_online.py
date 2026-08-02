"""The online adapter: the two-phase transition, and what must not move between the phases.

The offline updater could treat a boundary as one event. Online it is two, because the host asks
for a handover and only afterwards splices it into the context the agent acts on. Everything here
defends the gap between them: the graph does not move when the handover is produced, no record is
committed until the splice returned, and an abort leaves both exactly as they were.

The model is a stub throughout. What is under test is the state machine around it.
"""

import json

import pytest

from future_graph import ComputationNode as C, InformationNode as I
from future_graph import InformationKind as K, Relation as R, build
from future_graph.artifacts import ArtifactError, ModelCall, RevisionRecord
from future_graph.online import (
    ACCEPTED, EMPTY, REFUSED, LocalRevisionOptimizer, OnlineIntegrationError, continuation_of,
)
from future_graph.online_run import (
    Continuation, RepoIdentity, build_manifest, describe_repo, hashes_of, load_run,
    openai_version, prepare_online_run, write_json,
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
        hashes=hashes_of("system", "rules", "prompt"))
    manifest.update({"status": status, "boundaries": boundaries,
                     "continuations": continuations, "finished_at": "now"})
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


def test_the_handover_check_is_a_reading_of_the_messages_that_were_sent():
    handover = "REFINED PLAN OVERVIEW\n[c1] do the thing"
    assert continuation_of([{"role": "user", "content": f"task\n\n{handover}\n\nend"}], handover)
    assert not continuation_of([{"role": "user", "content": "task"}], handover)
    assert not continuation_of([{"role": "user", "content": "task"}], "")
