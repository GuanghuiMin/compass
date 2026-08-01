"""The preflight harness, on a fixture of its own and never a real call.

Every episode, manifest and rules file here is built by the test. Nothing reads the frozen corpus, so
these pass on a machine that has never seen it, and no test in this file constructs a real client.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from future_graph.adapter import (
    API_KEY_VAR, BASE_URL, BASE_URL_VAR, MODEL, MODEL_VAR, Adapter, AdapterError, from_environment,
)
from future_graph.artifacts import ModelCall, RegenerationRecord, prompt_sha
from future_graph.episodes import InputError, canonical_inputs_sha256, load
from future_graph.regeneration import load_prompt
from future_graph.run import (
    FROZEN_CONFIG, OperationalFailure, RunError, prepare_run, replay, validate_run_id, write_record,
)

VALID = """\
BEGIN_GRAPH

COMPUTATION c1
description: Carry out the remaining work
END_COMPUTATION

END_GRAPH
"""

WITH_DEAD = """\
BEGIN_GRAPH

INFO i1
kind: fact
available: true
description: Needed by nobody
END_INFO

COMPUTATION c1
description: Carry out the remaining work
END_COMPUTATION

END_GRAPH
"""

CYCLE = """\
BEGIN_GRAPH

COMPUTATION c1
description: One
END_COMPUTATION

COMPUTATION c2
description: Two
END_COMPUTATION

EDGE c1 PRECEDES c2
EDGE c2 PRECEDES c1

END_GRAPH
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_fixture(tmp_path, episodes=(("ep_one", ["first slice", "second slice"]),
                                      ("ep_two", ["only slice, with a \u00e9 in it"]))):
    """A source directory, a rules directory and a manifest that agree with each other."""
    source = tmp_path / "source"
    source.mkdir()
    repo = tmp_path / "repo"
    (repo / "inputs" / "preflight" / "rules").mkdir(parents=True)

    entries = []
    for name, slices in episodes:
        goal = f"the goal of {name}"
        document = {"task_id": name, "instruction": goal,
                    "events": [{"compass": {"compaction_index": i, "delta_h": s}}
                               for i, s in enumerate(slices)]}
        raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
        (source / f"{name}.json").write_bytes(raw)
        rules = f"the fixed rules of {name}\n".encode("utf-8")
        (repo / "inputs" / "preflight" / "rules" / f"{name}.txt").write_bytes(rules)
        entries.append({
            "id": name, "episode_file": f"{name}.json", "episode_file_sha256": sha(raw),
            "goal_bytes": len(goal.encode()), "goal_sha256": sha(goal.encode()),
            "rules_file": f"inputs/preflight/rules/{name}.txt",
            "rules_bytes": len(rules), "rules_sha256": sha(rules),
            "boundary_count": len(slices),
            "boundaries": [{"compaction_index": i, "delta_h_bytes": len(s.encode("utf-8")),
                            "delta_h_sha256": sha(s.encode("utf-8"))}
                           for i, s in enumerate(slices)],
        })

    inputs = {"episodes": entries}
    manifest = {"inputs": inputs, "input_manifest_sha256": canonical_inputs_sha256(inputs),
                "source_path_at_freeze": str(source), "source_kind": "reconstructed",
                "historical_byte_identity_verified": False, "sampling_is_deterministic": False}
    manifest_path = repo / "inputs" / "preflight" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return source, manifest_path, repo


def rewrite_manifest(manifest_path, change):
    manifest = json.loads(manifest_path.read_text())
    change(manifest)
    manifest["input_manifest_sha256"] = canonical_inputs_sha256(manifest["inputs"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


class Stub:
    """A model returning fixed answers in order, counting the calls it received."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[ModelCall] = []

    def __call__(self, call: ModelCall) -> str:
        self.calls.append(call)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


# --------------------------------------------------------------------------- input verification

def test_a_sound_fixture_loads(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    assert [e.id for e in inputs.episodes] == ["ep_one", "ep_two"]
    assert inputs.boundary_count == 3
    assert inputs.episodes[0].boundaries[1] == (1, "second slice")
    assert inputs.episodes[0].rules == "the fixed rules of ep_one\n"
    assert inputs.source_kind == "reconstructed"
    assert inputs.historical_byte_identity_verified is False
    assert inputs.sampling_is_deterministic is False


def test_the_loaded_object_carries_the_verified_manifest_hash(tmp_path):
    """The run manifest takes its identity from here, never from a second read of the file."""
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    recorded = json.loads(manifest.read_text())["input_manifest_sha256"]
    assert inputs.input_manifest_sha256 == recorded


def test_a_manifest_whose_own_hash_is_wrong_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["input_manifest_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(InputError, match="inputs hash to"):
        load(source, manifest, repo)


def test_a_missing_episode_file_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    (source / "ep_two.json").unlink()
    with pytest.raises(InputError, match="is not there"):
        load(source, manifest, repo)


def test_an_altered_episode_file_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    (source / "ep_one.json").write_bytes(b'{"instruction": "different", "events": []}')
    with pytest.raises(InputError, match="hashes to"):
        load(source, manifest, repo)


def test_a_goal_hash_that_disagrees_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][0].__setitem__(
        "goal_sha256", "0" * 64))
    with pytest.raises(InputError, match="goal hashes to"):
        load(source, manifest, repo)


def test_a_goal_length_that_disagrees_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][0].__setitem__("goal_bytes", 3))
    with pytest.raises(InputError, match="goal is .* bytes"):
        load(source, manifest, repo)


def test_altered_rules_bytes_are_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    (repo / "inputs/preflight/rules/ep_one.txt").write_bytes(b"tampered\n")
    with pytest.raises(InputError, match="rules"):
        load(source, manifest, repo)


def test_a_missing_rules_file_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    (repo / "inputs/preflight/rules/ep_one.txt").unlink()
    with pytest.raises(InputError, match="is not there"):
        load(source, manifest, repo)


def test_a_boundary_count_that_disagrees_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][0].__setitem__(
        "boundary_count", 5))
    with pytest.raises(InputError, match="slices"):
        load(source, manifest, repo)


def test_a_slice_hash_that_disagrees_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][0]["boundaries"][1].__setitem__(
        "delta_h_sha256", "0" * 64))
    with pytest.raises(InputError, match="slice hashes to"):
        load(source, manifest, repo)


def test_slices_are_measured_in_bytes_not_characters(tmp_path):
    """The non-ASCII slice passes on bytes and would fail on a character count."""
    source, manifest, repo = build_fixture(tmp_path)
    entry = json.loads(manifest.read_text())["inputs"]["episodes"][1]
    text = "only slice, with a \u00e9 in it"
    assert entry["boundaries"][0]["delta_h_bytes"] == len(text.encode("utf-8"))
    assert len(text.encode("utf-8")) != len(text)
    assert load(source, manifest, repo)

    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][1]["boundaries"][0].__setitem__(
        "delta_h_bytes", len(text)))
    with pytest.raises(InputError, match="slice is .* bytes"):
        load(source, manifest, repo)


def test_a_gap_in_the_compaction_indices_is_refused(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    document = json.loads((source / "ep_one.json").read_text())
    document["events"][1]["compass"]["compaction_index"] = 5
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    (source / "ep_one.json").write_bytes(raw)
    rewrite_manifest(manifest, lambda m: m["inputs"]["episodes"][0].__setitem__(
        "episode_file_sha256", sha(raw)))
    with pytest.raises(InputError, match="0 upwards with no gap"):
        load(source, manifest, repo)


# --------------------------------------------------------------------------- sequencing

def test_each_boundary_is_asked_once_and_the_accepted_graph_moves_forward(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    stub = Stub(VALID)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = replay(inputs, stub, run_dir)
    assert len(stub.calls) == 3
    assert result["status"] == "completed"
    assert result["completed_boundaries"] == {"ep_one": 2, "ep_two": 1}
    second = json.loads((run_dir / "episode_ep_one" / "boundary_001.json").read_text())
    assert second["previous_snapshot"]["computations"][0]["id"] == "c1"


def test_each_episode_starts_from_an_empty_graph(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(VALID), run_dir)
    first_of_second = json.loads((run_dir / "episode_ep_two" / "boundary_000.json").read_text())
    assert first_of_second["previous_snapshot"] == {"computations": [], "information": [],
                                                    "edges": []}


def test_a_refused_graph_leaves_the_previous_one_in_front_of_the_next_boundary(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(VALID, CYCLE, VALID), run_dir)
    rejected = json.loads((run_dir / "episode_ep_one" / "boundary_001.json").read_text())
    assert rejected["accepted"] is False
    assert rejected["resulting_snapshot"] == rejected["previous_snapshot"]
    assert rejected["previous_snapshot"]["computations"][0]["id"] == "c1"


def test_collection_is_visible_in_the_record(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(WITH_DEAD), run_dir)
    first = json.loads((run_dir / "episode_ep_one" / "boundary_000.json").read_text())
    assert first["collected"] == ["i1"]
    assert [i["id"] for i in first["parsed_candidate_snapshot"]["information"]] == ["i1"]
    assert first["resulting_snapshot"]["information"] == []


def test_the_call_carries_the_episode_goal_rules_and_slice(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    stub = Stub(VALID)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, stub, run_dir)
    user = stub.calls[0].user
    assert "the goal of ep_one" in user
    assert "the fixed rules of ep_one" in user
    assert "first slice" in user
    assert stub.calls[0].config == (("max_tokens", 16384), ("seed", 1), ("temperature", 0.0))


def test_the_frozen_configuration_is_not_a_parameter():
    import inspect
    assert "config" not in inspect.signature(replay).parameters
    assert FROZEN_CONFIG == {"temperature": 0.0, "max_tokens": 16384, "seed": 1}


# --------------------------------------------------------------------------- stopping

def test_a_model_failure_stops_the_run_and_leaves_later_work_absent(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    stub = Stub(VALID, RuntimeError("the service is down"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="the service is down"):
        replay(inputs, stub, run_dir)

    assert (run_dir / "episode_ep_one" / "boundary_000.json").exists()
    assert not (run_dir / "episode_ep_one" / "boundary_001.json").exists()
    assert not (run_dir / "episode_ep_two").exists()
    recorded = json.loads((run_dir / "manifest.json").read_text())
    assert recorded["status"] == "operational_failure"
    assert recorded["operational_failure"]["layer"] == "RuntimeError"
    assert recorded["completed_boundaries"] == {"ep_one": 1, "ep_two": 0}
    assert len(stub.calls) == 2


def test_a_failure_is_never_written_as_a_refused_graph(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(AdapterError):
        replay(inputs, Stub(AdapterError("no choices")), run_dir)
    assert list((run_dir / "episode_ep_one").glob("boundary_*.json")) == []


# --------------------------------------------------------------------------- the manifest

def test_the_manifest_records_what_the_run_declared(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recorded = replay(inputs, Stub(VALID), run_dir)
    assert recorded["input_manifest_sha256"] == inputs.input_manifest_sha256
    assert recorded["prompt_sha"] == prompt_sha(load_prompt())
    assert recorded["model"] == MODEL
    assert recorded["config"] == FROZEN_CONFIG
    assert recorded["per_request_timeout_s"] == 240
    assert recorded["temperature_effective"] is False
    assert recorded["sampling_is_deterministic"] is False
    assert recorded["episode_order"] == ["ep_one", "ep_two"]
    assert recorded["expected_boundaries"] == {"ep_one": 2, "ep_two": 1}
    assert recorded["openai_version"] == "1.60.1"
    assert recorded["started_at"] and recorded["finished_at"]


def test_every_record_must_hash_to_the_prompt_the_run_declared(tmp_path, monkeypatch):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    import future_graph.run as run_module
    monkeypatch.setattr(run_module, "prompt_sha", lambda _text: "0" * 64)
    with pytest.raises(OperationalFailure, match="does not hash"):
        replay(inputs, Stub(VALID), run_dir)


# --------------------------------------------------------------------------- artifacts

def test_a_record_is_read_back_before_it_is_placed(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(VALID), run_dir)
    path = run_dir / "episode_ep_one" / "boundary_000.json"
    assert RegenerationRecord.from_dict(json.loads(path.read_text()))
    assert not list(path.parent.glob("*.tmp"))


def test_an_unreadable_record_is_never_placed(tmp_path, monkeypatch):
    """An exception before the rename leaves the final path absent and no temporary behind."""
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    import future_graph.run as run_module
    monkeypatch.setattr(run_module.RegenerationRecord, "from_dict",
                        staticmethod(lambda _raw: (_ for _ in ()).throw(ValueError("bad"))))
    with pytest.raises(OperationalFailure, match="did not read back"):
        replay(inputs, Stub(VALID), run_dir)
    episode_dir = run_dir / "episode_ep_one"
    assert not (episode_dir / "boundary_000.json").exists()
    assert not list(episode_dir.glob("*.tmp"))


def test_an_existing_boundary_record_is_never_replaced(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(VALID), run_dir)
    path = run_dir / "episode_ep_one" / "boundary_000.json"
    record = RegenerationRecord.from_dict(json.loads(path.read_text()))
    with pytest.raises(OperationalFailure, match="never replaced"):
        write_record(path, record)


def test_the_run_manifest_is_rewritten_as_the_status_moves(tmp_path):
    """Unlike a boundary record: a manifest that could not be rewritten could not record a failure."""
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(VALID), run_dir)
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"


# --------------------------------------------------------------------------- run preconditions

@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "/absolute", "./x"])
def test_a_run_id_that_is_not_one_component_is_refused(bad):
    with pytest.raises(RunError, match="single path component"):
        validate_run_id(bad)


def test_a_run_directory_that_exists_is_refused(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        prepare_run("taken", tmp_path, check_git=False)


def test_claiming_a_run_directory_creates_it(tmp_path):
    run_dir = prepare_run("fresh", tmp_path, check_git=False)
    assert run_dir.is_dir() and run_dir.name == "fresh"


def test_the_pinned_openai_version_is_checked_not_only_recorded():
    from future_graph.run import REQUIRED_OPENAI, check_openai_version
    assert REQUIRED_OPENAI == "1.60.1"
    assert check_openai_version() == REQUIRED_OPENAI


# --------------------------------------------------------------------------- the adapter

class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response=None, **kwargs):
        self.kwargs = kwargs
        self.chat = type("Chat", (), {"completions": FakeCompletions(response)})()


class Message:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, content):
        self.message = Message(content)


class Response:
    def __init__(self, content):
        self.choices = [Choice(content)]


def environment(monkeypatch, base=BASE_URL, model=MODEL, key="secret"):
    for name, value in ((BASE_URL_VAR, base), (MODEL_VAR, model), (API_KEY_VAR, key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_the_client_is_built_with_the_endpoint_timeout_and_no_sdk_retries(monkeypatch):
    """The SDK retries twice by default, which would make one sample up to three requests."""
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return FakeClient(Response("ok"))

    environment(monkeypatch)
    from_environment(factory)
    assert seen["base_url"] == "https://ollama.com/v1"
    assert seen["timeout"] == 240
    assert seen["max_retries"] == 0
    assert seen["api_key"] == "secret"


@pytest.mark.parametrize("missing", [BASE_URL_VAR, MODEL_VAR, API_KEY_VAR])
def test_a_missing_environment_variable_refuses_before_a_client_exists(monkeypatch, missing):
    environment(monkeypatch)
    monkeypatch.delenv(missing, raising=False)
    built = []
    with pytest.raises(AdapterError, match="is not set"):
        from_environment(lambda **kw: built.append(kw))
    assert built == []


def test_a_different_endpoint_is_refused(monkeypatch):
    environment(monkeypatch, base="http://10.183.22.68:8005/v1")
    with pytest.raises(AdapterError, match="only for"):
        from_environment(lambda **kw: None)


def test_a_different_model_is_refused(monkeypatch):
    environment(monkeypatch, model="MiniMaxAI/MiniMax-M2.5")
    with pytest.raises(AdapterError, match="only for"):
        from_environment(lambda **kw: None)


def test_the_key_never_appears_in_the_refusal(monkeypatch):
    environment(monkeypatch, model="wrong", key="super-secret-key")
    with pytest.raises(AdapterError) as caught:
        from_environment(lambda **kw: None)
    assert "super-secret-key" not in str(caught.value)


def test_the_request_is_built_from_the_model_call_alone():
    client = FakeClient(Response("the answer"))
    adapter = Adapter(client=client, model=MODEL)
    call = ModelCall(system="S", user="U", config=(("seed", 1), ("temperature", 0.0)))
    assert adapter(call) == "the answer"
    request = client.chat.completions.requests[0]
    assert request == {"model": "minimax-m3",
                       "messages": [{"role": "system", "content": "S"},
                                    {"role": "user", "content": "U"}],
                       "seed": 1, "temperature": 0.0}


@pytest.mark.parametrize("response,message", [
    (type("R", (), {"choices": []})(), "no choices"),
    (type("R", (), {"choices": [type("C", (), {})()]})(), "no message"),
    (Response(None), "no content"),
    (Response(7), "not text"),
])
def test_a_provider_that_answers_in_the_wrong_shape_is_an_adapter_failure(response, message):
    adapter = Adapter(client=FakeClient(response), model=MODEL)
    with pytest.raises(AdapterError, match=message):
        adapter(ModelCall(system="S", user="U"))


def test_an_empty_answer_is_a_model_answer_and_not_an_adapter_failure():
    adapter = Adapter(client=FakeClient(Response("")), model=MODEL)
    assert adapter(ModelCall(system="S", user="U")) == ""


def test_an_empty_answer_becomes_a_parse_rejection(tmp_path):
    source, manifest, repo = build_fixture(tmp_path)
    inputs = load(source, manifest, repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    replay(inputs, Stub(""), run_dir)
    first = json.loads((run_dir / "episode_ep_one" / "boundary_000.json").read_text())
    assert first["accepted"] is False and first["parse_errors"]
    assert first["parsed_candidate_snapshot"] is None


# --------------------------------------------------------------------------- verify-only

def test_verify_only_touches_no_environment_and_makes_no_call(tmp_path, monkeypatch, capsys):
    source, manifest, repo = build_fixture(tmp_path)
    for name in (BASE_URL_VAR, MODEL_VAR, API_KEY_VAR):
        monkeypatch.delenv(name, raising=False)
    inputs = load(source, manifest, repo)          # what --verify-only does, and nothing more
    assert inputs.boundary_count == 3
    assert not (tmp_path / "artifacts").exists()


def test_the_command_offers_no_way_to_aim_elsewhere():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "replay_preflight.py").read_text()
    assert "--manifest-path" not in text
    assert "--output-root" not in text
    for flag in ("--source-dir", "--run-id", "--verify-only"):
        assert flag in text
