"""Reading a pilot back: one compatibility row per future-graph cell, and the counting rules.

The canonical record of a future-graph cell is its online run directory. This derives a row in the
shape the existing baseline aggregation already speaks, so one reader can see all four conditions --
but the row is derived, never authored. It is produced only after the canonical artifact has
strict-loaded with no pending transition, and if the two ever disagree the aggregation refuses and
the artifact wins. A row that could repair an artifact would be a second source of truth, and then
there would be no source of truth.

The counting rules are here rather than in a notebook because they were fixed before the run and
have to stay that way. Every one of them can make a method look better or worse, so each is written
down once, applied to every condition alike, and reported with its denominator in view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactError
from .online_run import PENDING, load_run
from .pilot import CONDITIONS, REPLICATES, sha256_bytes

FUTURE_GRAPH = "future_graph_v1"

# A cell ends in exactly one of these. The first three are outcomes of a run; the last two are the
# run failing to be one, and they are never mixed into an effectiveness denominator unexamined.
COMPLETED = "completed"
OPERATIONAL_FAILURE = "operational_failure"
INTEGRATION_FAILURE = "integration_failure"
CONTEXT_LIMIT = "context_limit"
NOT_LAUNCHED = "not_launched"


class PilotAggregationError(ValueError):
    """The pilot's records disagree with each other, so no number may be taken from them."""


# --------------------------------------------------------------------------- the derived row

def compatibility_row(run_dir: Path, *, condition: str, task_id: str, replicate: int) -> dict:
    """Derive a baseline-shaped row from a canonical online run, or refuse to.

    Everything the row says is read out of the artifact here and now. Nothing is remembered from
    when the run happened, so a row can always be rebuilt and compared against the thing it claims
    to describe.
    """
    run_dir = Path(run_dir)
    run = load_run(run_dir)                                    # strict: every record, every check
    manifest = run.manifest

    leftover = sorted((run_dir / PENDING).glob("*.pending")) if (run_dir / PENDING).is_dir() else []
    if leftover:
        raise PilotAggregationError(
            f"{run_dir.name} left {len(leftover)} prepared boundary(ies) uncommitted")

    if manifest["task_id"] != task_id:
        raise PilotAggregationError(f"{run_dir.name} is task {manifest['task_id']}, "
                                    f"and the schedule says {task_id}")
    if manifest["method"] != FUTURE_GRAPH or condition != FUTURE_GRAPH:
        raise PilotAggregationError(f"{run_dir.name} is {manifest['method']}, not {FUTURE_GRAPH}")

    evaluation = manifest.get("evaluation") or {}
    status = manifest["status"]
    if status == "completed":
        outcome = COMPLETED
    elif manifest.get("integration_failure"):
        outcome = INTEGRATION_FAILURE
    elif manifest.get("operational_failure"):
        outcome = OPERATIONAL_FAILURE
    else:
        outcome = status

    return {
        "condition": condition,
        "task_id": task_id,
        "replicate": replicate,
        "cell_id": f"{condition}/{task_id}/replicate_{replicate}",
        "difficulty": manifest.get("difficulty"),
        "success": bool(evaluation.get("success", False)),
        "score": float(evaluation.get("score", 0.0) or 0.0),
        "num_steps": int(manifest.get("steps", 0)),
        "peak_prompt_tokens": int(manifest.get("peak_prompt_tokens", 0)),
        "cumulative_input_tokens": int(manifest.get("cumulative_input_tokens", 0)),
        "termination_reason": manifest.get("termination_reason"),
        "outcome": outcome,
        "canonical_artifact": str(run_dir),
        "canonical_manifest_sha256": sha256_bytes((run_dir / "manifest.json").read_bytes()),
        "repos": {r.name: r.commit for r in run.repos},
        "instrumentation": run.instrumentation,
        "boundaries": manifest.get("boundaries", 0),
        "accepted_boundaries": manifest.get("accepted_boundaries", 0),
        "refused_boundaries": manifest.get("refused_boundaries", 0),
        "empty_boundaries": manifest.get("empty_boundaries", 0),
        "operational_attempts": manifest.get("operational_attempts", 0),
        "continuations": manifest.get("continuations", 0),
    }


def check_row_against_artifact(row: dict) -> None:
    """Rebuild the row from its artifact and refuse the pilot if the two have drifted apart."""
    rebuilt = compatibility_row(Path(row["canonical_artifact"]), condition=row["condition"],
                                task_id=row["task_id"], replicate=row["replicate"])
    differing = sorted(k for k in rebuilt if k in row and rebuilt[k] != row[k])
    if differing:
        raise PilotAggregationError(
            f"{row['cell_id']}: the recorded row and its canonical artifact disagree about "
            f"{', '.join(differing)}; the artifact is the record and the row is never used to "
            "repair it")


# --------------------------------------------------------------------------- plumbing metrics

def plumbing(run_dir: Path) -> dict:
    """What the future-graph machinery did, in counts. Never compared across conditions.

    A baseline has no boundaries, no revisions and no graph, so a number from here has nothing on
    the other side to be compared with, and putting one in a comparison table would invent one.
    """
    run = load_run(Path(run_dir))
    boundaries = []
    for index, record in enumerate(run.boundaries):
        previous_bytes = len(json.dumps(record.previous_snapshot).encode("utf-8"))
        boundaries.append({
            "boundary": index,
            "accepted": record.accepted,
            "empty": record.empty_revision,
            "parse_errors": len(record.parse_errors),
            "faults": [f.code for f in record.faults],
            "violations": [v.code for v in record.violations],
            "attempts": len(record.attempts),
            "revision_bytes": len(record.raw_output.encode("utf-8")),
            "previous_snapshot_bytes": previous_bytes,
            "handover_bytes": len(record.handover.encode("utf-8")),
            "affected_roots": len(record.affected_roots),
            "touched_nodes": len(record.touched_nodes),
            "removed_nodes": len(record.removed_nodes),
            "newly_created_then_collected": [str(r) for r
                                             in record.newly_created_then_collected],
        })
    linked = {c.boundary_index for c in run.continuations}
    return {
        "run": Path(run_dir).name,
        "strict_load": True,
        "leftover_pending": 0,
        "boundaries": boundaries,
        "continuations": [{"boundary": c.boundary_index, "step": c.step_index,
                           "handover_present": c.handover_present,
                           "first_post_compaction_decision": c.first_post_compaction_decision}
                          for c in run.continuations],
        "boundaries_with_continuation": sorted(linked),
        "provider_calls": len(run.provider_calls),
        "finish_reasons": sorted({str(c.get("finish_reason")) for c in run.provider_calls}),
    }


# --------------------------------------------------------------------------- counting

@dataclass(frozen=True)
class Denominators:
    planned: int
    launched: int
    completed: int
    operational: int
    integration: int
    not_launched: int


def denominators(rows: list[dict], planned: int) -> Denominators:
    launched = [r for r in rows if r["outcome"] != NOT_LAUNCHED]
    return Denominators(
        planned=planned,
        launched=len(launched),
        completed=sum(1 for r in launched if r["outcome"] == COMPLETED),
        operational=sum(1 for r in launched if r["outcome"] == OPERATIONAL_FAILURE),
        integration=sum(1 for r in launched if r["outcome"] == INTEGRATION_FAILURE),
        not_launched=planned - len(launched),
    )


def success_both_ways(rows: list[dict]) -> dict:
    """Two rates, always reported together.

    Excluding operational failures rewards a method for being flaky; counting them as task failures
    blames a method for a provider. Neither is right on its own, so both are shown and the raw
    count is shown beside them.
    """
    launched = [r for r in rows if r["outcome"] != NOT_LAUNCHED]
    operational = [r for r in launched if r["outcome"] == OPERATIONAL_FAILURE]
    judged = [r for r in launched if r["outcome"] not in (OPERATIONAL_FAILURE,
                                                          INTEGRATION_FAILURE)]
    passed = sum(1 for r in judged if r["success"])
    return {
        "launched": len(launched),
        "operational_failures": len(operational),
        "excluding_operational": {"passed": passed, "of": len(judged),
                                  "rate": _rate(passed, len(judged))},
        "counting_operational_as_failure": {
            "passed": passed, "of": len(judged) + len(operational),
            "rate": _rate(passed, len(judged) + len(operational))},
    }


def _rate(passed: int, total: int) -> float | None:
    return None if total == 0 else round(passed / total, 4)


def matched_blocks(rows: list[dict]) -> list[tuple[str, int]]:
    """Blocks where all four conditions finished, so a cross-condition reading uses one task set."""
    finished: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        if row["outcome"] == COMPLETED:
            finished.setdefault((row["task_id"], row["replicate"]), set()).add(row["condition"])
    return sorted(key for key, conditions in finished.items()
                  if conditions >= set(CONDITIONS))


def aggregate(rows: list[dict], planned_per_condition: int) -> dict:
    """Every number the pilot may report, with its denominator attached to it."""
    for row in rows:
        if row["condition"] not in CONDITIONS:
            raise PilotAggregationError(f"{row['condition']!r} is not a pilot condition")
        if row["replicate"] not in REPLICATES:
            raise PilotAggregationError(f"replicate {row['replicate']} is not in the pilot")

    blocks = matched_blocks(rows)
    per_condition = {}
    for condition in CONDITIONS:
        mine = [r for r in rows if r["condition"] == condition]
        counts = denominators(mine, planned_per_condition)
        per_condition[condition] = {
            "planned": counts.planned, "launched": counts.launched,
            "completed": counts.completed, "operational_failures": counts.operational,
            "integration_failures": counts.integration, "not_launched": counts.not_launched,
            "success": success_both_ways(mine),
            "by_difficulty": _by_difficulty(mine),
            "matched_blocks_only": success_both_ways(
                [r for r in mine if (r["task_id"], r["replicate"]) in set(blocks)]),
        }
    return {
        "conditions": per_condition,
        "matched_complete_blocks": [{"task_id": t, "replicate": r} for t, r in blocks],
        "matched_complete_block_count": len(blocks),
        "note": ("In this 15-task, two-replicate exploratory pilot no significance test, "
                 "confidence interval, or superiority claim is computed. full_context is a "
                 "ceiling reference whose interface is not matched to the 4096-window "
                 "conditions."),
        "rows": rows,
    }


def _by_difficulty(rows: list[dict]) -> dict:
    out = {}
    for difficulty in sorted({r.get("difficulty") for r in rows if r.get("difficulty")}):
        out[str(difficulty)] = success_both_ways(
            [r for r in rows if r.get("difficulty") == difficulty])
    return out


def zero_boundary_cells(rows: list[dict]) -> list[str]:
    """Kept in the primary denominator, and named so a reader can see how many there were.

    They are not evidence that the conditions were equivalent on those tasks: whether a boundary
    happens depends on the trajectory, and two draws of the same condition may differ.
    """
    return [r["cell_id"] for r in rows
            if r["condition"] == FUTURE_GRAPH and r.get("boundaries", 0) == 0]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        if row["condition"] == FUTURE_GRAPH:
            check_row_against_artifact(row)
    return rows


def read_baseline_records(path: Path, *, condition: str, replicate: int,
                          wrapper: dict) -> list[dict]:
    """The baselines' own `run_records.jsonl`, read as rows and left where it is.

    Nothing is written back into a baseline's artifacts. The wrapper identity travels beside the
    rows so a row can always be traced to the command that produced it.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({
            "condition": condition,
            "task_id": record["task_id"],
            "replicate": replicate,
            "cell_id": f"{condition}/{record['task_id']}/replicate_{replicate}",
            "difficulty": None,
            "success": bool(record.get("success", False)),
            "score": float(record.get("score", 0.0) or 0.0),
            "num_steps": int(record.get("num_steps", 0)),
            "peak_prompt_tokens": int(record.get("peak_prompt_tokens", 0)),
            "cumulative_input_tokens": int(record.get("cumulative_input_tokens", 0)),
            "termination_reason": record.get("termination_reason"),
            "outcome": COMPLETED if not record.get("error") else OPERATIONAL_FAILURE,
            "error": record.get("error"),
            "canonical_artifact": str(path),
            "wrapper": wrapper,
        })
    return rows


def apply_difficulty(rows: list[dict], difficulty_of: dict) -> list[dict]:
    for row in rows:
        row["difficulty"] = difficulty_of.get(row["task_id"], row.get("difficulty"))
    return rows
