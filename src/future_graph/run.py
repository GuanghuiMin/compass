"""One pass over the frozen boundaries, and the evidence it leaves behind.

The configuration is frozen here rather than accepted from a caller, so the same command cannot become
a different experiment by being invoked differently.

Two kinds of file, with two kinds of promise. A boundary record is immutable: it is written to a
temporary file, read back through `from_dict`, and only then moved into place, and an existing one is
never replaced. The run manifest is mutable by design, because a manifest that could not be rewritten
could not record the failure that stopped it.

A model or service failure stops everything at once. It is recorded as an operational failure with the
layer it came from, never as a refused graph, and never retried.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapter import MODEL, TIMEOUT_S
from .artifacts import RegenerationRecord, prompt_sha
from .episodes import REPO_ROOT, PreflightInputs
from .regeneration import Model, load_prompt, regenerate_graph
from .state_graph import StateGraph

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "preflight"
REQUIRED_OPENAI = "1.60.1"

# The values the frozen corpus was produced with. Not a parameter: an experiment that can be
# reconfigured from the command line is several experiments sharing a name.
FROZEN_CONFIG: dict[str, Any] = {"temperature": 0.0, "max_tokens": 16384, "seed": 1}


class RunError(RuntimeError):
    """A precondition of the run that does not hold."""


class OperationalFailure(RuntimeError):
    """The run could not continue for a reason that is not a graph being refused."""


@dataclass(frozen=True)
class PreparedRun:
    """Everything settled before the directory was claimed, so the run never re-derives it."""
    run_dir: Path
    commit_sha: str
    prompt_sha: str
    openai_version: str


def validate_run_id(run_id: str) -> str:
    """One path component, so an explicit argument cannot leave the artifact root."""
    if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id \
            or Path(run_id).is_absolute() or run_id != Path(run_id).name:
        raise RunError(f"run id {run_id!r} must be a single path component")
    return run_id


def openai_version() -> str:
    import openai
    return openai.__version__


def check_openai_version() -> str:
    found = openai_version()
    if found != REQUIRED_OPENAI:
        raise RunError(f"openai is {found}, and this run is pinned to {REQUIRED_OPENAI}")
    return found


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RunError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def check_tree_clean() -> None:
    if git("status", "--porcelain"):
        raise RunError("the working tree has changes; a run must name the commit it ran")


def prepare_run(run_id: str, artifact_root: Path = ARTIFACT_ROOT) -> PreparedRun:
    """Settle everything that can fail without sending a request, then claim the directory.

    The claim comes last. A directory taken before the environment was checked would be an empty
    run nobody can use and a run id nobody can reuse.
    """
    version = check_openai_version()
    check_tree_clean()
    validate_run_id(run_id)
    commit = git("rev-parse", "HEAD")
    sha = prompt_sha(load_prompt())
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return PreparedRun(run_dir=run_dir, commit_sha=commit, prompt_sha=sha, openai_version=version)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise OperationalFailure(f"{path} already exists and records are never replaced")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_record(path: Path, record: RegenerationRecord) -> None:
    """Write, read back, then place. A record that cannot be read back is never placed."""
    if path.exists():
        raise OperationalFailure(f"{path} already exists and records are never replaced")
    payload = record.to_dict()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    try:
        RegenerationRecord.from_dict(json.loads(temporary.read_text(encoding="utf-8")))
    except Exception as err:
        temporary.unlink(missing_ok=True)
        raise OperationalFailure(f"{path.name} did not read back: {err}") from err
    os.replace(temporary, path)


def _manifest(inputs: PreflightInputs, prepared: PreparedRun, status: str,
              started_at: str) -> dict:
    return {
        "run_id": prepared.run_dir.name,
        "status": status,
        "commit": prepared.commit_sha,
        "prompt_sha": prepared.prompt_sha,
        "model": MODEL,
        "config": dict(FROZEN_CONFIG),
        "per_request_timeout_s": TIMEOUT_S,
        "openai_version": prepared.openai_version,
        "input_manifest_sha256": inputs.input_manifest_sha256,
        "source_kind": inputs.source_kind,
        "historical_byte_identity_verified": inputs.historical_byte_identity_verified,
        "sampling_is_deterministic": inputs.sampling_is_deterministic,
        "temperature_requested": FROZEN_CONFIG["temperature"],
        "temperature_effective": False,
        "seed_requested": FROZEN_CONFIG["seed"],
        "episode_order": [e.id for e in inputs.episodes],
        "expected_boundaries": {e.id: len(e.boundaries) for e in inputs.episodes},
        "completed_boundaries": {e.id: 0 for e in inputs.episodes},
        "started_at": started_at,
        "finished_at": None,
        "operational_failure": None,
    }


def replay(inputs: PreflightInputs, model: Model, prepared: PreparedRun) -> dict:
    """Regenerate every frozen boundary once, in order, recording each one before the next."""
    started = _now()
    manifest_path = prepared.run_dir / "manifest.json"
    manifest = _manifest(inputs, prepared, "planned", started)
    _write_json(manifest_path, manifest, overwrite=True)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest, overwrite=True)

    def stop(layer: str, err: BaseException, where: dict | None) -> None:
        """Record which layer failed and which frozen boundary it stopped at, then let it travel."""
        manifest["status"] = "operational_failure"
        manifest["operational_failure"] = {
            "layer": layer,
            "exception_type": type(err).__name__,
            "message": str(err)[:2000],
            "boundary": where,
        }
        manifest["finished_at"] = _now()
        _write_json(manifest_path, manifest, overwrite=True)

    for episode in inputs.episodes:
        episode_dir = prepared.run_dir / f"episode_{episode.id}"
        try:
            episode_dir.mkdir(exist_ok=False)
        except BaseException as err:
            stop("run", err, {"episode": episode.id, "compaction_index": None})
            raise
        previous = StateGraph()                      # each episode starts from nothing
        for index, delta_h in episode.boundaries:
            where = {"episode": episode.id, "compaction_index": index}
            try:
                result = regenerate_graph(episode.goal, episode.rules, previous, delta_h,
                                          model, FROZEN_CONFIG)
            except BaseException as err:
                stop("regeneration", err, where)
                raise

            if result.record.prompt_sha != prepared.prompt_sha:
                err = OperationalFailure(
                    f"{episode.id} #{index}: the prompt sent does not hash to the one this run "
                    "declared")
                stop("prompt_identity", err, where)
                raise err

            try:
                write_record(episode_dir / f"boundary_{index:03d}.json", result.record)
            except BaseException as err:
                stop("boundary_artifact", err, where)
                raise

            manifest["completed_boundaries"][episode.id] = index + 1
            _write_json(manifest_path, manifest, overwrite=True)
            previous = result.graph

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    _write_json(manifest_path, manifest, overwrite=True)
    return manifest
