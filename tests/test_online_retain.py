"""`future_graph_v1r`: a refusal keeps its source, and everything else is v1.

Two things are under test. First, that the one behavioural change is the only one -- accepted and
empty transitions must be byte-equivalent to `future_graph_v1` given the same inputs, or the
diagnostic would be measuring two variables. Second, that retention is real: the slice is neither
deleted nor duplicated, and the ledger can prove it on read rather than in prose.

The three indices are tested apart on purpose. A committed boundary, a revision attempt and a
provider attempt are different counts, and conflating them is how a refusal would end up read as a
boundary in a report.
"""

import json
from pathlib import Path

import pytest

from future_graph.artifacts import ArtifactError, ModelCall, RevisionRecord
from future_graph.online import ACCEPTED, EMPTY, REFUSED, LocalRevisionOptimizer
from future_graph.online import OnlineIntegrationError
from future_graph.online_retain import (
    METHOD, RetainingLocalRevisionOptimizer, RevisionRefused, check_interval, sha256_of,
)
from future_graph.online_run import (
    ATTEMPTS, INSTRUMENTATION_RETAIN, SLICE_LEDGER, RepoIdentity, build_manifest,
    environment_identity, finish_manifest, hashes_of, load_run, monotonic_seconds,
    prepare_online_run, write_json,
)
from future_graph.rendering import render

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

TURN = "ASSISTANT:\nreasoning {n}\nprint({n})\n\nUSER:\nobservation {n}\n\n"


def slice_of(n: int) -> str:
    """A host-rendered interval of n turns, in the shape convert_llm_history_to_text produces."""
    return "".join(TURN.format(n=i) for i in range(1, n + 1))


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


def retaining(tmp_path, model, **kw):
    return RetainingLocalRevisionOptimizer(
        goal="the goal", rules="the rules", model=model,
        count_tokens=lambda t: len(t.split()), window=1,
        boundaries_dir=tmp_path / "boundaries", pending_dir=tmp_path / ".pending",
        attempts_dir=tmp_path / "attempts", config={"temperature": 0.0, "seed": 1}, **kw)


def plain(tmp_path, model):
    return LocalRevisionOptimizer(
        goal="the goal", rules="the rules", model=model,
        count_tokens=lambda t: len(t.split()), window=1,
        boundaries_dir=tmp_path / "b", pending_dir=tmp_path / "p",
        config={"temperature": 0.0, "seed": 1})


# --------------------------------------------------------------------------- v1 equivalence

def test_an_accepted_transition_is_byte_equivalent_to_v1(tmp_path):
    """If accepted behaviour differed at all, the diagnostic would be moving two variables."""
    a = plain(tmp_path / "v1", stub(FIRST))
    b = retaining(tmp_path / "v1r", stub(FIRST))
    handover_a = a.process(task="t", history=slice_of(2))
    handover_b = b.process(task="t", history=slice_of(2))
    assert handover_a == handover_b
    first, second = a.commit_pending(), b.commit_pending()
    assert first.status == second.status == ACCEPTED
    assert a.graph.to_snapshot() == b.graph.to_snapshot()
    assert render(a.graph) == render(b.graph)
    assert a.model.calls[0].user == b.model.calls[0].user
    assert first.record.to_dict() == second.record.to_dict()


def test_an_empty_transition_without_a_prior_refusal_is_byte_equivalent_to_v1(tmp_path):
    a = plain(tmp_path / "v1", stub(FIRST, EMPTY_REVISION))
    b = retaining(tmp_path / "v1r", stub(FIRST, EMPTY_REVISION))
    for opt in (a, b):
        opt.process(task="t", history=slice_of(2))
        opt.commit_pending()
    ha = a.process(task="t", history=slice_of(4), prev_history_summary=render(a.graph))
    hb = b.process(task="t", history=slice_of(4), prev_history_summary=render(b.graph))
    assert ha == hb
    ca, cb = a.commit_pending(), b.commit_pending()
    assert ca.status == cb.status == EMPTY
    assert a.graph.to_snapshot() == b.graph.to_snapshot()
    assert b.absorbed_by_empty() == []          # nothing was retained, so nothing was absorbed


def test_v1_still_returns_the_previous_handover_on_refusal(tmp_path):
    """The frozen version is untouched; this is the behaviour v1r removes."""
    a = plain(tmp_path, stub(FIRST, UNPARSABLE))
    a.process(task="t", history=slice_of(2))
    first = a.commit_pending()
    handover = a.process(task="t", history=slice_of(4), prev_history_summary=first.handover)
    assert handover == first.handover
    assert a.commit_pending().status == REFUSED


# --------------------------------------------------------------------------- the one change

def test_a_refusal_raises_instead_of_returning_a_handover(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    with pytest.raises(RevisionRefused) as raised:
        opt.process(task="t", history=slice_of(2))
    refused = raised.value
    assert refused.revision_attempt_index == 0
    assert refused.candidate_boundary_index == 0
    assert refused.record_status == REFUSED
    assert refused.delta_h_sha256 == sha256_of(slice_of(2))
    assert refused.delta_h_bytes == len(slice_of(2).encode("utf-8"))
    assert not hasattr(refused, "handover")


def test_a_refusal_writes_an_attempt_and_never_a_boundary(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    assert list((tmp_path / "boundaries").glob("*.json")) == []
    written = sorted((tmp_path / "attempts").glob("attempt_*.json"))
    assert [p.name for p in written] == ["attempt_000.identity.json", "attempt_000.json"]
    record = RevisionRecord.from_dict(json.loads((tmp_path / "attempts" /
                                                  "attempt_000.json").read_text()))
    assert not record.accepted
    identity = json.loads((tmp_path / "attempts" / "attempt_000.identity.json").read_text())
    assert identity["committed"] is False
    assert identity["candidate_boundary_index"] == 0


def test_a_refusal_leaves_the_graph_and_the_boundary_index_alone(tmp_path):
    opt = retaining(tmp_path, stub(FIRST, UNPARSABLE))
    opt.process(task="t", history=slice_of(2))
    committed = opt.commit_pending()
    before, index = opt.graph.to_snapshot(), opt.boundary_index
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(4), prev_history_summary=committed.handover)
    assert opt.graph.to_snapshot() == before
    assert render(opt.graph) == committed.handover
    assert opt.boundary_index == index


def test_no_pending_transition_survives_a_refusal(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    assert opt.pending is None
    assert list((tmp_path / ".pending").glob("*")) == []


def test_the_record_is_read_back_strictly_before_the_refusal_is_raised(tmp_path, monkeypatch):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    from future_graph import online_retain

    def corrupt(raw):
        raise ArtifactError("this record is not readable")

    monkeypatch.setattr(online_retain.RevisionRecord, "from_dict", staticmethod(corrupt))
    with pytest.raises(OnlineIntegrationError, match="did not read back"):
        opt.process(task="t", history=slice_of(2))


# --------------------------------------------------------------------------- three indices

def test_revision_attempts_advance_where_boundaries_do_not(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE, UNPARSABLE, FIRST))
    for turns in (2, 4):
        with pytest.raises(RevisionRefused):
            opt.process(task="t", history=slice_of(turns))
    assert opt.revision_attempt_index == 2
    assert opt.boundary_index == 0
    opt.process(task="t", history=slice_of(6))
    assert opt.revision_attempt_index == 3
    assert opt.boundary_index == 0                       # not until the host splice commits
    committed = opt.commit_pending()
    assert committed.boundary_index == 0
    assert opt.boundary_index == 1


def test_consecutive_refusals_share_a_candidate_boundary_and_take_separate_paths(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    candidates = []
    for turns in (2, 4, 6):
        with pytest.raises(RevisionRefused) as raised:
            opt.process(task="t", history=slice_of(turns))
        candidates.append(raised.value.candidate_boundary_index)
    assert candidates == [0, 0, 0]
    assert sorted(p.name for p in (tmp_path / "attempts").glob("attempt_*.json")
                  if not p.name.endswith(".identity.json")) == \
        ["attempt_000.json", "attempt_001.json", "attempt_002.json"]


def test_provider_attempts_belong_to_their_revision_attempt(tmp_path, monkeypatch):
    from future_graph.adapter import EmptyModelCompletion
    monkeypatch.setattr("future_graph.retry.time.sleep", lambda _s: None)
    opt = retaining(tmp_path, stub(EmptyModelCompletion("nothing"), UNPARSABLE, FIRST))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    record = RevisionRecord.from_dict(json.loads(
        (tmp_path / "attempts" / "attempt_000.json").read_text()))
    assert [a[1] for a in record.attempts] == ["EmptyModelCompletion", "completion"]
    assert len(record.attempts) == 2          # two provider attempts inside one revision attempt


# --------------------------------------------------------------------------- the interval

def test_the_next_submission_must_extend_the_refused_one(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE, FIRST))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    opt.process(task="t", history=slice_of(4))            # a strict extension: accepted
    assert opt.pending is not None


@pytest.mark.parametrize("bad,match", [
    (slice_of(1), "not a prefix"),
    (slice_of(2), "not longer"),
    ("PREFIX" + slice_of(2), "not a prefix"),
])
def test_an_interval_that_is_not_the_retained_slice_plus_new_turns_stops_the_run(tmp_path, bad,
                                                                                 match):
    opt = retaining(tmp_path, stub(UNPARSABLE, FIRST))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    with pytest.raises(OnlineIntegrationError, match=match):
        opt.process(task="t", history=bad)
    assert len(opt.model.calls) == 1                      # refused before a second model call


def test_a_resubmission_with_no_new_action_is_refused_as_an_integration_failure(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE, FIRST))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    with pytest.raises(OnlineIntegrationError, match="no new action and observation"):
        opt.process(task="t", history=slice_of(2) + "trailing text with no turn\n")


def test_a_duplicated_retained_slice_stops_the_run():
    retained = slice_of(2)
    with pytest.raises(OnlineIntegrationError, match="exactly once"):
        check_interval(retained, retained + slice_of(1) + retained)


def test_the_interval_check_accepts_a_genuine_extension():
    check_interval(slice_of(2), slice_of(3))              # no exception


# --------------------------------------------------------------------------- absorption

def test_a_later_accepted_transition_consumes_the_retained_interval(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE, UNPARSABLE, FIRST))
    for turns in (2, 4):
        with pytest.raises(RevisionRefused):
            opt.process(task="t", history=slice_of(turns))
    assert len(opt.unabsorbed) == 2
    opt.process(task="t", history=slice_of(6))
    committed = opt.commit_pending()
    assert opt.unabsorbed == []
    ledger = opt.slice_ledger()
    for entry in ledger["slices"]:
        assert entry["absorbed_by_revision_attempt"] == 2
        assert entry["absorbed_by_boundary"] == committed.boundary_index
        assert entry["absorbed_by_empty_revision"] is False
    assert opt.last_refused_slice is None


def test_an_empty_revision_that_absorbs_a_retained_slice_is_marked_separately(tmp_path):
    """Not treated as evidence that the empty revision represented the retained information. An
    empty revision claims the slice changed nothing the state must carry, and that claim is exactly
    what a reader should weigh when the slice is one a revision was already refused for."""
    opt = retaining(tmp_path, stub(FIRST, UNPARSABLE, EMPTY_REVISION))
    opt.process(task="t", history=slice_of(2))
    first = opt.commit_pending()
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(4), prev_history_summary=first.handover)
    opt.process(task="t", history=slice_of(6), prev_history_summary=first.handover)
    committed = opt.commit_pending()
    assert committed.status == EMPTY
    absorbed = opt.absorbed_by_empty()
    assert len(absorbed) == 1
    assert absorbed[0].absorbed_by_empty_revision is True
    assert opt.slice_ledger()["absorbed_by_empty_revision"] == ["slice_001"]


def test_a_slice_still_retained_at_the_end_is_named(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    opt.close()
    ledger = opt.slice_ledger()
    assert ledger["unabsorbed_slices"] == ["slice_000"]
    assert ledger["slices"][0]["unabsorbed_at_task_end"] is True
    assert ledger["slices"][0]["absorbed_by_revision_attempt"] is None


def test_a_retained_slice_that_vanished_from_a_submission_stops_the_run(tmp_path):
    """It can only happen if something spliced behind our back, which is the defect to catch."""
    opt = retaining(tmp_path, stub(UNPARSABLE, FIRST))
    with pytest.raises(RevisionRefused):
        opt.process(task="t", history=slice_of(2))
    opt.last_refused_slice = None                # suppress the prefix check, as a splice would
    with pytest.raises(OnlineIntegrationError, match="not present in revision attempt"):
        opt.process(task="t", history="ASSISTANT:\nfresh\n\nUSER:\nnew\n\n")


# --------------------------------------------------------------------------- budget

def test_the_budget_is_measured_with_the_hosts_own_tokenizer(tmp_path):
    opt = retaining(tmp_path, stub(UNPARSABLE, UNPARSABLE))
    for turns in (2, 4):
        with pytest.raises(RevisionRefused):
            opt.process(task="t", history=slice_of(turns))
    budget = opt.retention_budget()
    assert budget["retained_history_bytes"] == \
        len(slice_of(2).encode()) + len(slice_of(4).encode())
    assert budget["retained_history_tokens"] == \
        len(slice_of(2).split()) + len(slice_of(4).split())
    assert budget["first_refused_revision_attempt"] == 0
    assert budget["unabsorbed_slice_count"] == 2
    assert budget["updater_calls_including_refused_history"] == 3


def test_the_budget_uses_no_provider_usage(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src" / "future_graph"
              / "online_retain.py").read_text()
    body = source.split("def retention_budget")[1].split('"""')[2]      # code, not the docstring
    for forbidden in ("prompt_tokens", "completion_tokens", "usage", "provider_calls"):
        assert forbidden not in body, "provider usage is missing too often to account with"


# --------------------------------------------------------------------------- the loader

def _here():
    return Path(__file__).resolve().parents[1]


def _clean_run(tmp_path, monkeypatch, run_id="v1r-run"):
    monkeypatch.setattr("future_graph.online_run.describe_repo",
                        lambda name, path: RepoIdentity(name, str(path), "r", "b", "0" * 40, True))
    return prepare_online_run(run_id, {"compass-v2": _here(), "trace-v2": _here()},
                              artifact_root=tmp_path)


def _run_with(tmp_path, monkeypatch, answers, turns, close=True):
    prepared = _clean_run(tmp_path, monkeypatch)
    (prepared.run_dir / ATTEMPTS).mkdir(exist_ok=True)
    opt = RetainingLocalRevisionOptimizer(
        goal="g", rules="r", model=stub(*answers), count_tokens=lambda t: len(t.split()),
        window=1, boundaries_dir=prepared.boundaries_dir, pending_dir=prepared.pending_dir,
        attempts_dir=prepared.run_dir / ATTEMPTS)
    for count in turns:
        try:
            opt.process(task="t", history=slice_of(count),
                        prev_history_summary=(opt.committed[-1].handover
                                              if opt.committed else None))
            opt.commit_pending()
        except RevisionRefused:
            pass
    if close:
        opt.close()
    manifest = build_manifest(
        prepared, task_id="t", split="test_normal", window=4096, preserved_turns=1, max_steps=50,
        tasklist={}, downstream={}, updater={}, hashes=hashes_of("a", "b", "c"),
        environment=environment_identity(task_id="t", split="test_normal", instruction="i",
                                         experiment_name="e", command=["c"]),
        started_monotonic=monotonic_seconds())
    manifest["instrumentation"] = INSTRUMENTATION_RETAIN
    manifest.update({
        "committed_boundaries": len(opt.committed),
        "revision_attempts": opt.revision_attempt_index,
        "refused_revision_attempts": len(opt.refused_attempts),
        "provider_attempts": sum(len(c.record.attempts) for c in opt.committed)
                             + sum(1 for _ in opt.refused_attempts),
        "unabsorbed_slices": len(opt.unabsorbed),
        "boundaries": len(opt.committed),
    })
    write_json(prepared.run_dir / SLICE_LEDGER, opt.slice_ledger(), overwrite=True)
    finish_manifest(manifest, "completed")
    write_json(prepared.manifest_path, manifest, overwrite=True)
    return prepared, opt, manifest


def test_a_retaining_run_reads_back_with_its_attempts_and_ledger(tmp_path, monkeypatch):
    prepared, opt, _ = _run_with(tmp_path, monkeypatch, [UNPARSABLE, UNPARSABLE, FIRST],
                                 [2, 4, 6])
    run = load_run(prepared.run_dir)
    assert run.instrumentation == INSTRUMENTATION_RETAIN and run.retains_refused_slices
    assert len(run.boundaries) == 1 and len(run.attempts) == 2
    assert all(not r.accepted for r in run.attempts)
    assert run.slice_ledger["method"] == METHOD


def test_a_manifest_that_counts_a_refusal_as_a_boundary_is_refused(tmp_path, monkeypatch):
    prepared, _opt, manifest = _run_with(tmp_path, monkeypatch, [UNPARSABLE, FIRST], [2, 4])
    manifest["committed_boundaries"] = 2
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="committed boundaries"):
        load_run(prepared.run_dir)


def test_a_manifest_with_fewer_revision_attempts_than_records_is_refused(tmp_path, monkeypatch):
    prepared, _opt, manifest = _run_with(tmp_path, monkeypatch, [UNPARSABLE, FIRST], [2, 4])
    manifest["revision_attempts"] = 1
    write_json(prepared.manifest_path, manifest, overwrite=True)
    with pytest.raises(ArtifactError, match="fewer than"):
        load_run(prepared.run_dir)


def test_a_ledger_that_lost_a_slice_is_refused(tmp_path, monkeypatch):
    prepared, opt, manifest = _run_with(tmp_path, monkeypatch, [UNPARSABLE, UNPARSABLE, FIRST],
                                        [2, 4, 6])
    ledger = opt.slice_ledger()
    ledger["slices"] = ledger["slices"][:1]
    write_json(prepared.run_dir / SLICE_LEDGER, ledger, overwrite=True)
    with pytest.raises(ArtifactError, match="1 slices and 2 refused"):
        load_run(prepared.run_dir)


def test_a_ledger_that_records_the_same_slice_twice_is_refused(tmp_path, monkeypatch):
    prepared, opt, _ = _run_with(tmp_path, monkeypatch, [UNPARSABLE, UNPARSABLE, FIRST], [2, 4, 6])
    ledger = opt.slice_ledger()
    ledger["slices"][1]["sha256"] = ledger["slices"][0]["sha256"]
    write_json(prepared.run_dir / SLICE_LEDGER, ledger, overwrite=True)
    with pytest.raises(ArtifactError, match="identical resubmission"):
        load_run(prepared.run_dir)


def test_a_ledger_that_duplicates_an_inclusion_is_refused(tmp_path, monkeypatch):
    prepared, opt, _ = _run_with(tmp_path, monkeypatch, [UNPARSABLE, FIRST], [2, 4])
    ledger = opt.slice_ledger()
    entry = ledger["slices"][0]
    entry["included_in_revision_attempts"] = [0, 0]
    entry["included_at_offset"] = [0, 0]
    write_json(prepared.run_dir / SLICE_LEDGER, ledger, overwrite=True)
    with pytest.raises(ArtifactError, match="duplicated history"):
        load_run(prepared.run_dir)


def test_a_slice_neither_absorbed_nor_marked_unabsorbed_is_refused(tmp_path, monkeypatch):
    prepared, opt, _ = _run_with(tmp_path, monkeypatch, [UNPARSABLE], [2], close=False)
    ledger = opt.slice_ledger()
    write_json(prepared.run_dir / SLICE_LEDGER, ledger, overwrite=True)
    with pytest.raises(ArtifactError, match="neither absorbed nor marked"):
        load_run(prepared.run_dir)


def test_a_retaining_run_without_a_ledger_is_refused(tmp_path, monkeypatch):
    prepared, _opt, _ = _run_with(tmp_path, monkeypatch, [UNPARSABLE, FIRST], [2, 4])
    (prepared.run_dir / SLICE_LEDGER).unlink()
    with pytest.raises(ArtifactError, match="requires slice_ledger"):
        load_run(prepared.run_dir)


def test_earlier_instrumentation_is_unaffected():
    root = _here() / "artifacts"
    runs = [p.parent for p in root.rglob("manifest.json")
            if "online" in p.parts or "diagnostic" in p.parts]
    if not runs:
        pytest.skip("no online run has been committed yet")
    for run_dir in runs:
        run = load_run(run_dir)
        assert run.instrumentation in (1, 2, 3)
        if run.instrumentation < INSTRUMENTATION_RETAIN:
            assert run.attempts == () and run.slice_ledger is None
            assert not run.retains_refused_slices
