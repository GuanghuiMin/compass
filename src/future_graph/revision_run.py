"""One recurrent pass over the frozen boundaries through the local-revision updater.

Separate from `run.replay`, which stays exactly as it is: that one measures the complete-graph
baseline, and a shared runner would make it impossible to say later which operator produced a
number.

What this adds beyond writing records is the per-boundary measurement the recurrent question needs
answering. Complete rewriting failed because the answer grew with the state, so the thing to watch
is whether a revision's size tracks the change while the graph goes on growing, and whether a chain
of boundaries survives at all. Those are read off the artifacts rather than asserted here.

Two guarantees are checked at runtime rather than trusted, because both are easy to get wrong and
invisible when wrong: after a refusal the previous graph must be byte-identical to what it was, and
the slice that was refused must not reappear in the next boundary's input.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .adapter import MODEL, TIMEOUT_S
from .artifacts import RevisionRecord, prompt_sha
from .episodes import REPO_ROOT, PreflightInputs
from .protocol import to_protocol
from .retry import MAX_ATTEMPTS, ExhaustedAttempts
from .run import (
    FROZEN_CONFIG, OperationalFailure, PreparedRun, RunError, check_openai_version,
    check_tree_clean, git, validate_run_id,
)
from .state_graph import StateGraph
from .update import Model, load_prompt, update_graph

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "recurrent"


def prepare_revision_run(run_id: str, artifact_root: Path = ARTIFACT_ROOT) -> PreparedRun:
    """As `run.prepare_run`, but pinning the revision prompt this operator actually sends."""
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


def _write_json(path: Path, payload: object, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise OperationalFailure(f"{path} already exists and records are never replaced")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_record(path: Path, record: RevisionRecord) -> None:
    """Write, read back, then place. A record that cannot be read back is never placed."""
    if path.exists():
        raise OperationalFailure(f"{path} already exists and records are never replaced")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    try:
        RevisionRecord.from_dict(json.loads(temporary.read_text(encoding="utf-8")))
    except Exception as err:
        temporary.unlink(missing_ok=True)
        raise OperationalFailure(f"{path.name} did not read back: {err}") from err
    os.replace(temporary, path)


@dataclass(frozen=True)
class Sizes:
    previous_graph_bytes: int
    delta_h_bytes: int
    revision_bytes: int
    handover_bytes: int


def _sizes(previous: StateGraph, delta_h: str, record: RevisionRecord) -> Sizes:
    """Byte lengths, in UTF-8, of the four things whose relative growth is the whole question."""
    return Sizes(
        previous_graph_bytes=len(to_protocol(previous).encode("utf-8")),
        delta_h_bytes=len(delta_h.encode("utf-8")),
        revision_bytes=len(record.raw_output.encode("utf-8")),
        handover_bytes=len(record.handover.encode("utf-8")),
    )


def measure(previous: StateGraph, delta_h: str, record: RevisionRecord,
            resulting: StateGraph) -> dict:
    """Everything about one boundary that the recurrent question is asked of.

    `preserved` is the point of comparison: work the model never mentioned and the code carried
    across unchanged. If that number stays large while `revision_bytes` stays small, local revision
    is doing what complete rewriting could not.
    """
    sizes = _sizes(previous, delta_h, record)
    previous_nodes = len(previous.computations) + len(previous.information)
    removed = {reason: [str(r.node) for r in record.removed_nodes if r.reason == reason]
               for reason in ("affected_region", "region_internal", "invalidated_information")}
    removed_total = sum(len(nodes) for nodes in removed.values())
    return {
        "accepted": record.accepted,
        "empty_revision": record.empty_revision,
        "attempts": [list(attempt) for attempt in record.attempts],
        "operational_attempts": len(record.attempts) - 1,
        "sizes": {
            "previous_graph_bytes": sizes.previous_graph_bytes,
            "delta_h_bytes": sizes.delta_h_bytes,
            "revision_bytes": sizes.revision_bytes,
            "handover_bytes": sizes.handover_bytes,
            "revision_over_previous_graph": (round(sizes.revision_bytes
                                                   / sizes.previous_graph_bytes, 3)
                                             if sizes.previous_graph_bytes else None),
        },
        "nodes": {
            "previous_total": previous_nodes,
            "affected_roots": len(record.affected_roots),
            "replaced_or_removed": removed_total,
            "removed_by_reason": {reason: len(nodes) for reason, nodes in removed.items()},
            "touched": len(record.touched_nodes),
            "preserved": previous_nodes - removed_total,
            "resulting_computations": len(resulting.computations),
            "resulting_information": len(resulting.information),
            "resulting_edges": len(resulting.edges),
        },
        # Every id here carries its space, as `space:id`, because the same string in two of these
        # lists routinely means two different nodes.
        "model_authored": {
            "affected_roots": [str(r) for r in record.affected_roots],
            "touched_nodes": [str(r) for r in record.touched_nodes],
            "removed_regions": removed["affected_region"],
        },
        "code_owned": {
            "region_internal": removed["region_internal"],
            "invalidated_information": removed["invalidated_information"],
            "replacement_boundary_changes": len(record.replacement_boundary_changes),
            "completion_changes": [
                [c.action, str(c.node), str(c.producer) if c.producer else "", c.detail]
                for c in record.completion_changes],
            "argument_dependency_changes": len(record.argument_dependency_changes),
            "interface_changes": len(record.interface_changes),
            "ordering_repairs": len(record.ordering_repairs),
            "collected": [str(r) for r in record.collected],
        },
        "newly_created_then_collected": [str(r) for r in record.newly_created_then_collected],
        "normalizations": list(record.normalizations),
        "refusal": None if record.accepted else {
            "parse_errors": [list(e) for e in record.parse_errors],
            "faults": [_report(f) for f in record.faults],
            "violations": [_report(v) for v in record.violations],
        },
    }


def _report(report) -> dict:
    return {"code": report.code, "message": report.message,
            "nodes": [str(n) for n in report.nodes], "sites": list(report.sites)}


def _manifest(inputs: PreflightInputs, prepared: PreparedRun, status: str,
              started_at: str) -> dict:
    return {
        "run_id": prepared.run_dir.name,
        "operator": "local_revision",
        "status": status,
        "commit": prepared.commit_sha,
        "prompt_sha": prepared.prompt_sha,
        "model": MODEL,
        "config": dict(FROZEN_CONFIG),
        "per_request_timeout_s": TIMEOUT_S,
        "max_attempts_per_boundary": MAX_ATTEMPTS,
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
        "accepted_boundaries": {e.id: 0 for e in inputs.episodes},
        "started_at": started_at,
        "finished_at": None,
        "operational_failure": None,
    }


def replay_revision(inputs: PreflightInputs, model: Model, prepared: PreparedRun) -> dict:
    """Update the graph at every frozen boundary in order, recording each before the next.

    A refused boundary does not stop the chain. That is the whole point of running it: the previous
    graph carries on, the slice is gone, and the next boundary is attempted against the state as it
    stood. Only a provider that never answered stops anything.
    """
    started = _now()
    manifest_path = prepared.run_dir / "manifest.json"
    manifest = _manifest(inputs, prepared, "planned", started)
    _write_json(manifest_path, manifest, overwrite=True)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest, overwrite=True)

    def stop(layer: str, err: BaseException, where: dict | None) -> None:
        manifest["status"] = "operational_failure"
        manifest["operational_failure"] = {
            "layer": layer,
            "exception_type": type(err).__name__,
            "message": str(err)[:2000],
            "attempts": [a.as_list() for a in getattr(err, "attempts", ())],
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
        previous = StateGraph()
        rows: list[dict] = []
        refused_slices: list[str] = []

        for index, delta_h in episode.boundaries:
            where = {"episode": episode.id, "compaction_index": index}
            before = json.dumps(previous.to_snapshot(), sort_keys=True, ensure_ascii=False)

            # A slice that was refused must not come back. Checked against the input rather than
            # assumed, because buffering a refused slice would hand the method an attempt the
            # schedule never gave it and would be invisible in the records.
            for earlier in refused_slices:
                if earlier and earlier in delta_h:
                    err = OperationalFailure(
                        f"{episode.id} #{index}: the slice refused earlier appears in this one")
                    stop("delta_h_identity", err, where)
                    raise err

            try:
                result = update_graph(episode.goal, episode.rules, previous, delta_h,
                                      model, FROZEN_CONFIG)
            except ExhaustedAttempts as err:
                stop("model", err, where)
                raise
            except BaseException as err:
                stop("update", err, where)
                raise

            if result.record.prompt_sha != prepared.prompt_sha:
                err = OperationalFailure(
                    f"{episode.id} #{index}: the prompt sent does not hash to the one this run "
                    "declared")
                stop("prompt_identity", err, where)
                raise err

            after = json.dumps(result.graph.to_snapshot(), sort_keys=True, ensure_ascii=False)
            if not result.record.accepted and after != before:
                err = OperationalFailure(
                    f"{episode.id} #{index}: a refused boundary changed the graph")
                stop("atomicity", err, where)
                raise err

            row = measure(previous, delta_h, result.record, result.graph)
            row["compaction_index"] = index
            row["previous_graph_preserved_byte_identically"] = (
                None if result.record.accepted else after == before)
            row["delta_h_discarded"] = not result.record.accepted
            rows.append(row)

            try:
                write_record(episode_dir / f"boundary_{index:03d}.json", result.record)
            except BaseException as err:
                stop("boundary_artifact", err, where)
                raise
            _write_json(episode_dir / "summary.json", {
                "episode": episode.id,
                "goal_bytes": len(episode.goal.encode("utf-8")),
                "rules_bytes": len(episode.rules.encode("utf-8")),
                "boundaries": rows,
            }, overwrite=True)

            if not result.record.accepted:
                refused_slices.append(delta_h)
            manifest["completed_boundaries"][episode.id] = index + 1
            manifest["accepted_boundaries"][episode.id] = sum(1 for r in rows if r["accepted"])
            _write_json(manifest_path, manifest, overwrite=True)
            previous = result.graph

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    _write_json(manifest_path, manifest, overwrite=True)
    return manifest
