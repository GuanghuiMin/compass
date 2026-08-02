"""Which tasks the pilot runs, and in what order, decided from metadata alone.

This module can see three things: the official split file, the task ids in it, and each task's
ground-truth difficulty. It cannot see an outcome, because it never opens anything that holds one.
That is the whole design constraint and the reason selection lives in its own file with no artifact
imports -- a reader can settle the question by looking at what this module reads, rather than by
trusting a claim about it.

Prior outcomes do exist for these benchmark tasks. The pilot is exploratory rather than an untouched
holdout, and the honest guarantee is the narrower one: no outcome took part in choosing the tasks.

Ranking is by SHA-256 of a fixed key rather than by a seeded shuffle. A hash is auditable by hand
and cannot drift with a Python release, and there is no seed anyone could quietly try again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPLIT = "test_normal"
SELECTION_KEY = "test-normal-small-pilot-v1"
ALGORITHM = "scenario-dedup-then-difficulty-stratified-hash-rank-v1"
PER_DIFFICULTY = 5
DIFFICULTIES = (1, 2, 3)

CONDITIONS = ("future_graph_v1", "openclaw", "fifo", "full_context")
REPLICATES = (1, 2)

WINDOW = 4096
MAX_STEPS = 50
PRESERVED_TURNS = 1
WORKERS = 6


class PilotError(ValueError):
    """A precondition of the pilot that does not hold."""


def digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scenario_prefix(task_id: str) -> str:
    """`5238afc_1`, `_2` and `_3` are three variants of the scenario `5238afc`.

    Without this the sample would happily take three variants of one scenario and call them three
    tasks, which is three draws from one situation.
    """
    return task_id.rsplit("_", 1)[0]


# --------------------------------------------------------------------------- inputs

def load_split(split_path: Path) -> tuple[str, ...]:
    """The official list, read as ids. Order here is never trusted; everything below sorts."""
    raw = Path(split_path).read_bytes()
    ids = [token for token in raw.decode("utf-8").split() if token]
    if not ids:
        raise PilotError(f"{split_path} holds no task ids")
    if len(set(ids)) != len(ids):
        raise PilotError(f"{split_path} lists a task twice")
    return tuple(sorted(ids))


def load_difficulty(data_root: Path, task_id: str) -> int:
    """The benchmark's own difficulty, from its ground truth. The only per-task fact read."""
    path = Path(data_root) / "tasks" / task_id / "ground_truth" / "metadata.json"
    if not path.exists():
        raise PilotError(f"{task_id} has no ground-truth metadata at {path}")
    value = json.loads(path.read_text(encoding="utf-8")).get("difficulty")
    if value not in DIFFICULTIES:
        raise PilotError(f"{task_id} has difficulty {value!r}, expected one of {DIFFICULTIES}")
    return int(value)


# --------------------------------------------------------------------------- selection

@dataclass(frozen=True)
class SelectedTask:
    task_id: str
    scenario_prefix: str
    difficulty: int
    variant_hash: str          # why this variant of its scenario
    stratum_hash: str          # why this task, among its difficulty's scenarios

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "scenario_prefix": self.scenario_prefix,
                "difficulty": self.difficulty, "variant_hash": self.variant_hash,
                "stratum_hash": self.stratum_hash}


@dataclass(frozen=True)
class Selection:
    tasks: tuple[SelectedTask, ...]
    manifest: dict

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def difficulty_of(self, task_id: str) -> int:
        for task in self.tasks:
            if task.task_id == task_id:
                return task.difficulty
        raise PilotError(f"{task_id} is not in the pilot")


def select(split_path: Path, data_root: Path) -> Selection:
    """Fifteen tasks: one variant per scenario, five per difficulty, all of it hash-ordered."""
    ids = load_split(split_path)

    scenarios: dict[str, list[str]] = {}
    for task_id in ids:
        scenarios.setdefault(scenario_prefix(task_id), []).append(task_id)

    # One variant per scenario, chosen by hash rather than by which happened to sort first.
    representatives: list[tuple[str, str, str]] = []
    for prefix in sorted(scenarios):
        ranked = sorted(scenarios[prefix], key=lambda t: (digest(SELECTION_KEY, t), t))
        chosen = ranked[0]
        representatives.append((prefix, chosen, digest(SELECTION_KEY, chosen)))

    eligible: dict[int, list[SelectedTask]] = {d: [] for d in DIFFICULTIES}
    for prefix, task_id, variant_hash in representatives:
        difficulty = load_difficulty(data_root, task_id)
        eligible[difficulty].append(SelectedTask(
            task_id=task_id, scenario_prefix=prefix, difficulty=difficulty,
            variant_hash=variant_hash,
            stratum_hash=digest(SELECTION_KEY, f"difficulty-{difficulty}", task_id)))

    chosen: list[SelectedTask] = []
    for difficulty in DIFFICULTIES:
        stratum = sorted(eligible[difficulty], key=lambda t: (t.stratum_hash, t.task_id))
        if len(stratum) < PER_DIFFICULTY:
            raise PilotError(f"difficulty {difficulty} has {len(stratum)} eligible scenarios "
                             f"and the pilot needs {PER_DIFFICULTY}")
        chosen.extend(stratum[:PER_DIFFICULTY])

    tasks = tuple(chosen)                      # difficulty order, then hash order within it
    manifest = {
        "split": SPLIT,
        "split_path": str(split_path),
        "split_sha256": sha256_bytes(Path(split_path).read_bytes()),
        "split_task_count": len(ids),
        "algorithm": ALGORITHM,
        "selection_key": SELECTION_KEY,
        "per_difficulty": PER_DIFFICULTY,
        "eligible_scenarios_by_difficulty": {str(d): len(eligible[d]) for d in DIFFICULTIES},
        "scenario_count": len(scenarios),
        "tasks": [task.to_dict() for task in tasks],
        "selected_task_ids": [task.task_id for task in tasks],
        "selected_task_list_sha256": sha256_bytes(
            ("\n".join(task.task_id for task in tasks) + "\n").encode("utf-8")),
        "difficulty_counts": {str(d): sum(1 for t in tasks if t.difficulty == d)
                              for d in DIFFICULTIES},
    }
    return Selection(tasks=tasks, manifest=manifest)


# --------------------------------------------------------------------------- the schedule

@dataclass(frozen=True)
class Cell:
    """One task under one condition on one draw, named before anything runs."""
    condition: str
    task_id: str
    replicate: int
    difficulty: int
    block: int                 # which (task, replicate) block it belongs to
    position: int              # where in that block it launches
    schedule_hash: str

    @property
    def cell_id(self) -> str:
        return f"{self.condition}/{self.task_id}/replicate_{self.replicate}"

    @property
    def run_id(self) -> str:
        """The same identity as one path component, for artifact roots that take one."""
        return f"{self.condition}__{self.task_id}__replicate_{self.replicate}"

    def to_dict(self) -> dict:
        return {"cell_id": self.cell_id, "run_id": self.run_id, "condition": self.condition,
                "task_id": self.task_id, "replicate": self.replicate,
                "difficulty": self.difficulty, "block": self.block, "position": self.position,
                "schedule_hash": self.schedule_hash}


def build_schedule(selection: Selection) -> tuple[Cell, ...]:
    """Every planned cell, with its launch order fixed in advance.

    Conditions are ordered inside a block by hash, so no method is always first. Running one method
    first everywhere would confound it with whatever drifts over a long run -- provider load, rate
    limiting, a warming cache -- and nothing in the results could separate the two afterwards.
    """
    cells: list[Cell] = []
    block = 0
    for task in selection.tasks:
        for replicate in REPLICATES:
            ranked = sorted(
                CONDITIONS,
                key=lambda c: (digest(SELECTION_KEY, "schedule", task.task_id,
                                      str(replicate), c), c))
            for position, condition in enumerate(ranked):
                cells.append(Cell(
                    condition=condition, task_id=task.task_id, replicate=replicate,
                    difficulty=task.difficulty, block=block, position=position,
                    schedule_hash=digest(SELECTION_KEY, "schedule", task.task_id,
                                         str(replicate), condition)))
            block += 1
    return tuple(cells)


def schedule_manifest(selection: Selection, cells: tuple[Cell, ...]) -> dict:
    return {
        "selection_key": SELECTION_KEY,
        "algorithm": ALGORITHM,
        "selected_task_list_sha256": selection.manifest["selected_task_list_sha256"],
        "conditions": list(CONDITIONS),
        "replicates": list(REPLICATES),
        "configuration": {"window": WINDOW, "max_steps": MAX_STEPS,
                          "preserved_turns": PRESERVED_TURNS, "workers": WORKERS},
        "planned_cells": len(cells),
        "planned_blocks": len({(c.task_id, c.replicate) for c in cells}),
        "cells": [cell.to_dict() for cell in cells],
        "schedule_sha256": sha256_bytes(
            "\n".join(f"{c.block}:{c.position}:{c.cell_id}" for c in cells).encode("utf-8")),
    }


def check_schedule(cells: tuple[Cell, ...]) -> None:
    """The shapes a schedule has to have, checked once rather than assumed everywhere."""
    if len(cells) != len(CONDITIONS) * len(REPLICATES) * PER_DIFFICULTY * len(DIFFICULTIES):
        raise PilotError(f"the schedule holds {len(cells)} cells")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise PilotError("the schedule names a cell twice")
    for (task_id, replicate), group in _blocks(cells).items():
        if {cell.condition for cell in group} != set(CONDITIONS):
            raise PilotError(f"block {task_id}/{replicate} does not hold every condition")
        if sorted(cell.position for cell in group) != list(range(len(CONDITIONS))):
            raise PilotError(f"block {task_id}/{replicate} has no clear launch order")


def _blocks(cells: tuple[Cell, ...]) -> dict[tuple[str, int], list[Cell]]:
    grouped: dict[tuple[str, int], list[Cell]] = {}
    for cell in cells:
        grouped.setdefault((cell.task_id, cell.replicate), []).append(cell)
    return grouped


def write_manifests(selection: Selection, cells: tuple[Cell, ...], directory: Path) -> dict:
    """Write both manifests once. Neither is ever rewritten, so a later run cannot reselect."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, payload in (("tasks.json", selection.manifest),
                          ("schedule.json", schedule_manifest(selection, cells))):
        path = directory / name
        if path.exists():
            raise PilotError(f"{path} already exists; the pilot list is chosen once")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        written[name] = path
    (directory / "tasks.txt").write_text(
        "\n".join(selection.task_ids) + "\n", encoding="utf-8")
    return written


def load_manifests(directory: Path) -> tuple[dict, dict]:
    """Read the committed pilot back, and refuse a pair that does not belong together."""
    directory = Path(directory)
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    schedule = json.loads((directory / "schedule.json").read_text(encoding="utf-8"))
    if tasks["selected_task_list_sha256"] != schedule["selected_task_list_sha256"]:
        raise PilotError("the schedule was built for a different task list")
    if tasks["selection_key"] != SELECTION_KEY or schedule["selection_key"] != SELECTION_KEY:
        raise PilotError("the committed pilot was selected with another key")
    return tasks, schedule

