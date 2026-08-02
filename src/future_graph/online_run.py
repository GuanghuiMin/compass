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

import json
import os
import subprocess
from dataclasses import dataclass
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

def build_manifest(prepared: PreparedOnlineRun, *, task_id: str, split: str, window: int,
                   preserved_turns: int, max_steps: int, tasklist: dict,
                   downstream: dict, updater: dict, hashes: dict) -> dict:
    return {
        "run_id": prepared.run_dir.name,
        "operator": OPERATOR,
        "method": METHOD,
        "status": "planned",
        "repos": [r.to_dict() for r in prepared.repos],
        "openai_version": prepared.openai_version,
        "downstream": downstream,
        "updater": updater,
        "hashes": hashes,
        "task_id": task_id,
        "split": split,
        "tasklist": tasklist,
        "window": window,
        "preserved_turns": preserved_turns,
        "max_steps": max_steps,
        "started_at": now(),
        "finished_at": None,
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
    return OnlineRun(manifest=manifest, repos=repos, boundaries=tuple(boundaries),
                     continuations=tuple(continuations))
