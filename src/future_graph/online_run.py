"""What an online run leaves behind, and what has to be true before it starts.

An offline run pins one repository. An online run is two: the updater lives here and the agent loop
lives in the host, and a result that names only one of them cannot be rebuilt. So both are recorded
with remote, branch, commit and cleanliness, and both must be clean before a model is reached.

The other thing this exists for is the pending record. A boundary is prepared before the host
splices and becomes true only after, so a record can exist on disk for a transition that never
happened. Pending records therefore live in their own directory, out of the way of discovery, and a
run that claims to have completed while one is still lying there is refused when read back --
because that is exactly the shape of a run that stopped between preparing a boundary and committing
it, and it must not be mistaken for a clean one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, RevisionRecord, prompt_sha
from .episodes import REPO_ROOT
from .run import RunError, validate_run_id

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "online"
METHOD = "future_graph_v1"
OPERATOR = "local_revision"

BOUNDARIES = "boundaries"
CONTINUATIONS = "continuations"
PENDING = ".pending"
PROVIDER_CALLS = "provider_calls.json"

# 1 is the first online smoke, whose timing fields are wrong and whose provider metadata and
# environment identity were never captured. It is a real run and stays readable exactly as it was
# written; the checks that would refuse it apply from 2 onwards, and a reader is told which it has
# rather than being left to infer it from a missing key.
INSTRUMENTATION = 2


# --------------------------------------------------------------------------- clocks
#
# AppWorld runs an episode inside freezegun, and freezegun patches `time.time`, `datetime.now`,
# `time.monotonic` and `time.perf_counter` alike. Measured, not assumed: across a real 0.4s sleep
# inside the context all four report a delta of 0.000, while `clock_gettime` reports 0.400.
#
# The first online run recorded `started_at == finished_at == 2023-05-18` and an `elapsed_s` of
# -101287824.8, because a duration was taken as a host epoch minus a simulated one. A simulated
# datetime and a host epoch are different quantities and subtracting one from the other is not a
# duration; the two clocks below are kept apart so that cannot be written again.

def host_now() -> str:
    """Wall time on the machine, in UTC, unaffected by the episode's simulated clock."""
    return datetime.fromtimestamp(time.clock_gettime(time.CLOCK_REALTIME),
                                  timezone.utc).isoformat()


def monotonic_seconds() -> float:
    """The only stdlib clock that still advances inside a frozen episode. Durations use this."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def now() -> str:
    """Kept as the host clock, so no caller can reach the simulated one through this module."""
    return host_now()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if result.returncode != 0:
        raise RunError(f"git {' '.join(args)} in {repo} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def openai_version() -> str:
    """Recorded, not enforced.

    The offline runs pin a version because they own their interpreter. The online run borrows the
    host's, so the honest thing is to write down what actually ran rather than refuse it -- and the
    updater's client is built by `adapter.from_environment`, whose endpoint, model, timeout and
    `max_retries=0` do not depend on which of the two major versions is installed.
    """
    import openai
    return openai.__version__


# --------------------------------------------------------------------------- provider metadata
#
# The first online run's opening boundary was refused for a completion that ended inside an
# unclosed block. The artifact could not say whether the provider stopped early or the model simply
# wrote that, because nothing but the text was kept. It still cannot be decided after the fact, and
# nothing here decides it: a non-empty completion remains the model's answer and the retry policy is
# untouched. What changes is only that the provider's own account of the call is now written down
# beside the text, so the question is answerable next time instead of arguable.

def describe_response(response: Any) -> dict:
    """What the provider said about a call, read defensively and interpreted not at all."""
    choice = (getattr(response, "choices", None) or [None])[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "response_id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "incomplete_details": _plain(getattr(response, "incomplete_details", None)),
        "service_tier": getattr(response, "service_tier", None),
    }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return method()
            except Exception:                                       # noqa: BLE001
                break
    return str(value)


class _RecordingCompletions:
    def __init__(self, inner: Any, sink: list) -> None:
        self._inner, self._sink = inner, sink

    def create(self, *args: Any, **kwargs: Any) -> Any:
        started = monotonic_seconds()
        try:
            response = self._inner.create(*args, **kwargs)
        except Exception as err:                                    # noqa: BLE001
            self._sink.append({"error": f"{type(err).__name__}: {err}"[:500],
                               "elapsed_s": round(monotonic_seconds() - started, 3)})
            raise
        described = describe_response(response)
        described["elapsed_s"] = round(monotonic_seconds() - started, 3)
        self._sink.append(described)
        return response


class _RecordingChat:
    def __init__(self, inner: Any, sink: list) -> None:
        self.completions = _RecordingCompletions(inner.completions, sink)


class RecordingClient:
    """A pass-through that keeps the provider's account of every call it forwards.

    It wraps the client rather than the adapter, so the request is still built by the validated
    `Adapter.__call__` -- same endpoint, same model, same messages, same `max_retries=0` -- and
    there is no second copy of that construction to drift from the first.
    """

    def __init__(self, inner: Any, sink: list) -> None:
        self._inner = inner
        self.chat = _RecordingChat(inner.chat, sink)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def recording_adapter(base: Any) -> tuple[Any, list]:
    """The validated adapter with a recording client behind it, and the list it fills."""
    sink: list = []
    return replace(base, client=RecordingClient(base.client, sink)), sink


@dataclass(frozen=True)
class RepoIdentity:
    name: str
    path: str
    remote: str
    branch: str
    commit: str
    clean: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "remote": self.remote,
                "branch": self.branch, "commit": self.commit, "clean": self.clean}

    @classmethod
    def from_dict(cls, raw: object, where: str) -> "RepoIdentity":
        if not isinstance(raw, dict):
            raise ArtifactError(f"online manifest {where}: expected an object")
        expected = {"name", "path", "remote", "branch", "commit", "clean"}
        missing = sorted(expected - set(raw))
        if missing:
            raise ArtifactError(f"online manifest {where}: missing {', '.join(missing)}")
        if not isinstance(raw["clean"], bool):
            raise ArtifactError(f"online manifest {where}: clean is true or false")
        return cls(**{k: raw[k] for k in expected})


def describe_repo(name: str, path: Path) -> RepoIdentity:
    path = Path(path)
    try:
        remote = git(path, "remote", "get-url", "origin")
    except RunError:
        remote = ""
    return RepoIdentity(
        name=name, path=str(path), remote=remote,
        branch=git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        commit=git(path, "rev-parse", "HEAD"),
        clean=not git(path, "status", "--porcelain"),
    )


@dataclass(frozen=True)
class PreparedOnlineRun:
    run_dir: Path
    repos: tuple[RepoIdentity, ...]
    openai_version: str

    @property
    def boundaries_dir(self) -> Path:
        return self.run_dir / BOUNDARIES

    @property
    def pending_dir(self) -> Path:
        return self.run_dir / PENDING

    @property
    def continuations_dir(self) -> Path:
        return self.run_dir / CONTINUATIONS

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


def prepare_online_run(run_id: str, repos: dict[str, Path],
                       artifact_root: Path = ARTIFACT_ROOT) -> PreparedOnlineRun:
    """Settle both repositories, then claim the directory. Nothing is sent before this holds."""
    validate_run_id(run_id)
    identities = tuple(describe_repo(name, path) for name, path in repos.items())
    dirty = [r.name for r in identities if not r.clean]
    if dirty:
        raise RunError(f"the working tree of {', '.join(dirty)} has changes; a run must name the "
                       "commits it ran")
    version = openai_version()
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in (BOUNDARIES, CONTINUATIONS, PENDING):
        (run_dir / name).mkdir()
    return PreparedOnlineRun(run_dir=run_dir, repos=identities, openai_version=version)


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise ArtifactError(f"{path} already exists and records are never replaced")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


# --------------------------------------------------------------------------- the manifest

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_version(name: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(name)
    except Exception:                                               # noqa: BLE001
        return None


def environment_identity(*, task_id: str, split: str, instruction: str, experiment_name: str,
                         command: list[str]) -> dict:
    """Which installation, which task, which text.

    Two runs can pin the same two commits and still be different experiments, because the AppWorld
    package and its task data live outside both repositories. The instruction hash is the cheapest
    thing that would notice.
    """
    return {
        "appworld_version": package_version("appworld"),
        "task_id": task_id,
        "split": split,
        "experiment_name": experiment_name,
        "instruction_sha256": sha256_of(instruction),
        "instruction_bytes": len(instruction.encode("utf-8")),
        "command": list(command),
    }


def build_manifest(prepared: PreparedOnlineRun, *, task_id: str, split: str, window: int,
                   preserved_turns: int, max_steps: int, tasklist: dict,
                   downstream: dict, updater: dict, hashes: dict,
                   environment: dict, started_monotonic: float) -> dict:
    return {
        "run_id": prepared.run_dir.name,
        "operator": OPERATOR,
        "method": METHOD,
        "instrumentation": INSTRUMENTATION,
        "status": "planned",
        "repos": [r.to_dict() for r in prepared.repos],
        "openai_version": prepared.openai_version,
        "environment": environment,
        "downstream": downstream,
        "updater": updater,
        "hashes": hashes,
        "task_id": task_id,
        "split": split,
        "tasklist": tasklist,
        "window": window,
        "preserved_turns": preserved_turns,
        "max_steps": max_steps,
        # Host wall time for reading, monotonic seconds for measuring, and never one minus the
        # other. The episode's own simulated clock is recorded, if at all, under its own name.
        "host_started_at": host_now(),
        "host_finished_at": None,
        "started_monotonic": started_monotonic,
        "elapsed_s": None,
        "boundaries": 0,
        "accepted_boundaries": 0,
        "refused_boundaries": 0,
        "empty_boundaries": 0,
        "operational_attempts": 0,
        "continuations": 0,
        "steps": 0,
        "termination_reason": None,
        "evaluation": None,
        "operational_failure": None,
        "integration_failure": None,
    }


def finish_manifest(manifest: dict, status: str) -> dict:
    """Close the timing the only way that can produce a real duration."""
    manifest["host_finished_at"] = host_now()
    manifest["elapsed_s"] = round(monotonic_seconds() - manifest["started_monotonic"], 3)
    manifest["status"] = status
    return manifest


def hashes_of(system_prompt: str, rules: str, updater_prompt: str) -> dict:
    """Hashes of the three texts a run's meaning depends on and that are too large to inline."""
    return {"downstream_system_prompt_sha256": prompt_sha(system_prompt),
            "operating_rules_sha256": prompt_sha(rules),
            "updater_prompt_sha256": prompt_sha(updater_prompt)}


# --------------------------------------------------------------------------- continuations

@dataclass(frozen=True)
class Continuation:
    """The first downstream decision taken under a committed handover.

    `messages` is the exact list handed to the downstream model. It is what makes the claim
    checkable rather than asserted: the handover is either in there or it is not.
    """
    boundary_index: int
    step_index: int
    first_post_compaction_decision: bool
    handover_present: bool
    messages: list
    system_message: str
    reasoning: str
    tool_calls: list
    executed_code: str
    observation: str
    task_completed: bool
    boundary_artifact: str

    def to_dict(self) -> dict:
        return {
            "boundary_index": self.boundary_index,
            "step_index": self.step_index,
            "first_post_compaction_decision": self.first_post_compaction_decision,
            "handover_present": self.handover_present,
            "messages": self.messages,
            "system_message": self.system_message,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "executed_code": self.executed_code,
            "observation": self.observation,
            "task_completed": self.task_completed,
            "boundary_artifact": self.boundary_artifact,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "Continuation":
        if not isinstance(raw, dict):
            raise ArtifactError("a continuation: expected an object")
        expected = {f for f in cls.__dataclass_fields__}
        missing = sorted(expected - set(raw))
        unknown = sorted(set(raw) - expected)
        if missing:
            raise ArtifactError(f"a continuation: missing {', '.join(missing)}")
        if unknown:
            raise ArtifactError(f"a continuation: unknown {', '.join(unknown)}")
        for name in ("first_post_compaction_decision", "handover_present", "task_completed"):
            if not isinstance(raw[name], bool):
                raise ArtifactError(f"a continuation {name}: expected true or false")
        for name in ("messages", "tool_calls"):
            if not isinstance(raw[name], list):
                raise ArtifactError(f"a continuation {name}: expected a list")
        if not isinstance(raw["boundary_index"], int) or not isinstance(raw["step_index"], int):
            raise ArtifactError("a continuation: indices are whole numbers")
        return cls(**raw)


def continuation_path(prepared: PreparedOnlineRun, boundary_index: int) -> Path:
    return prepared.continuations_dir / f"continuation_{boundary_index:03d}.json"


# --------------------------------------------------------------------------- reading it back

@dataclass(frozen=True)
class OnlineRun:
    manifest: dict
    repos: tuple[RepoIdentity, ...]
    boundaries: tuple[RevisionRecord, ...]
    continuations: tuple[Continuation, ...]
    provider_calls: tuple[dict, ...] = ()

    @property
    def instrumentation(self) -> int:
        return int(self.manifest.get("instrumentation", 1))

    @property
    def timing_is_trustworthy(self) -> bool:
        """False for the first online run, whose duration was measured across two clocks."""
        return self.instrumentation >= 2


def _check_timing(manifest: dict) -> None:
    """A duration is a duration, or the run does not claim to have one.

    The first online run recorded a negative `elapsed_s` because a host epoch was reduced by a
    simulated one. Nothing downstream noticed, which is the actual defect: a timing field that can
    be nonsense and still pass is a timing field nobody can use.
    """
    for name in ("host_started_at", "host_finished_at"):
        value = manifest.get(name)
        if not isinstance(value, str) or not value:
            raise ArtifactError(f"online manifest: {name} is missing from a completed run")
        try:
            datetime.fromisoformat(value)
        except ValueError as err:
            raise ArtifactError(f"online manifest {name}: {value!r} is not a time") from err
    if datetime.fromisoformat(manifest["host_finished_at"]) \
            < datetime.fromisoformat(manifest["host_started_at"]):
        raise ArtifactError("online manifest: the run finished before it started")
    elapsed = manifest.get("elapsed_s")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        raise ArtifactError("online manifest: elapsed_s is missing from a completed run")
    if elapsed < 0:
        raise ArtifactError(f"online manifest: elapsed_s is {elapsed}, and no run takes "
                            "negative time; a duration was measured across two clocks")


def _check_environment(manifest: dict) -> None:
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ArtifactError("online manifest: environment identity is missing")
    for name in ("appworld_version", "task_id", "split", "instruction_sha256", "command"):
        if name not in environment:
            raise ArtifactError(f"online manifest environment: missing {name}")
    if environment["task_id"] != manifest["task_id"]:
        raise ArtifactError("online manifest: the environment names a different task than the run")


def _check_provider_calls(manifest: dict, calls: tuple[dict, ...],
                          boundaries: tuple[RevisionRecord, ...]) -> None:
    """One recorded call per attempt: the point of recording them is the ones that went wrong."""
    if not calls:
        return                       # a run from before this was recorded, or one with no boundary
    attempts = sum(len(record.attempts) for record in boundaries)
    if len(calls) != attempts:
        raise ArtifactError(f"online run recorded {len(calls)} provider calls and "
                            f"{attempts} attempts across its boundaries")


def load_run(run_dir: Path) -> OnlineRun:
    """Read a run back strictly, and refuse the shapes that would misreport what happened."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("run_id", "operator", "method", "status", "repos", "task_id"):
        if name not in manifest:
            raise ArtifactError(f"online manifest: missing {name}")
    if manifest["operator"] != OPERATOR or manifest["method"] != METHOD:
        raise ArtifactError(f"online manifest: {manifest['method']!r} by "
                            f"{manifest['operator']!r} is not this method")
    repos = tuple(RepoIdentity.from_dict(r, "repos") for r in manifest["repos"])
    if len(repos) < 2:
        raise ArtifactError("online manifest: an online run pins the updater and the host, "
                            f"and names {len(repos)}")

    pending = sorted((run_dir / PENDING).glob("*.pending")) if (run_dir / PENDING).is_dir() else []
    if pending and manifest["status"] == "completed":
        raise ArtifactError(
            f"online run says completed and left {len(pending)} prepared boundary(ies) "
            "uncommitted; a run that stopped between preparing and committing is not a clean one")

    boundaries = []
    for path in sorted((run_dir / BOUNDARIES).glob("boundary_*.json")):
        boundaries.append(RevisionRecord.from_dict(
            json.loads(path.read_text(encoding="utf-8"))))
    if len(boundaries) != manifest["boundaries"]:
        raise ArtifactError(f"online manifest counts {manifest['boundaries']} boundaries and "
                            f"{len(boundaries)} are on disk")

    continuations = []
    if (run_dir / CONTINUATIONS).is_dir():
        for path in sorted((run_dir / CONTINUATIONS).glob("continuation_*.json")):
            continuations.append(Continuation.from_dict(
                json.loads(path.read_text(encoding="utf-8"))))
    for continuation in continuations:
        if continuation.boundary_index >= len(boundaries):
            raise ArtifactError(f"a continuation names boundary {continuation.boundary_index} "
                                f"and there are {len(boundaries)}")

    calls: tuple[dict, ...] = ()
    call_path = run_dir / PROVIDER_CALLS
    if call_path.exists():
        raw = json.loads(call_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(c, dict) for c in raw):
            raise ArtifactError(f"{PROVIDER_CALLS}: expected a list of calls")
        calls = tuple(raw)

    if manifest["status"] == "completed" and int(manifest.get("instrumentation", 1)) >= 2:
        _check_timing(manifest)
        _check_environment(manifest)
        _check_provider_calls(manifest, calls, tuple(boundaries))

    return OnlineRun(manifest=manifest, repos=repos, boundaries=tuple(boundaries),
                     continuations=tuple(continuations), provider_calls=calls)
