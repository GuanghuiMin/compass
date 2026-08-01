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
class Boundary:
    episode: str
    index: int


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


def prepare_run(run_id: str, artifact_root: Path = ARTIFACT_ROOT,
                check_git: bool = True) -> Path:
    """Everything that must hold before the first call, in the order it must hold."""
    check_openai_version()
    if check_git:
        check_tree_clean()
    validate_run_id(run_id)
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)      # claims the directory, or refuses
    return run_dir


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


def _manifest(inputs: PreflightInputs, run_id: str, expected_prompt_sha: str,
              status: str, started_at: str) -> dict:
    return {
        "run_id": run_id,
        "status": status,
        "commit": git("rev-parse", "HEAD"),
        "prompt_sha": expected_prompt_sha,
        "model": MODEL,
        "config": dict(FROZEN_CONFIG),
        "per_request_timeout_s": TIMEOUT_S,
        "openai_version": openai_version(),
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


def replay(inputs: PreflightInputs, model: Model, run_dir: Path) -> dict:
    """Regenerate every frozen boundary once, in order, recording each one before the next."""
    started = _now()
    expected_prompt_sha = prompt_sha(load_prompt())
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = _manifest(inputs, Path(run_dir).name, expected_prompt_sha, "planned", started)
    _write_json(manifest_path, manifest, overwrite=True)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest, overwrite=True)

    try:
        for episode in inputs.episodes:
            episode_dir = Path(run_dir) / f"episode_{episode.id}"
            episode_dir.mkdir(exist_ok=False)
            previous = StateGraph()                  # each episode starts from nothing
            for index, delta_h in episode.boundaries:
                result = regenerate_graph(episode.goal, episode.rules, previous, delta_h,
                                          model, FROZEN_CONFIG)
                if result.record.prompt_sha != expected_prompt_sha:
                    raise OperationalFailure(
                        f"{episode.id} #{index}: the prompt sent does not hash to the one this "
                        "run declared")
                write_record(episode_dir / f"boundary_{index:03d}.json", result.record)
                manifest["completed_boundaries"][episode.id] = index + 1
                _write_json(manifest_path, manifest, overwrite=True)
                previous = result.graph
    except BaseException as err:
        manifest["status"] = "operational_failure"
        manifest["operational_failure"] = {"layer": type(err).__name__, "message": str(err)[:2000]}
        manifest["finished_at"] = _now()
        _write_json(manifest_path, manifest, overwrite=True)
        raise

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    _write_json(manifest_path, manifest, overwrite=True)
    return manifest
