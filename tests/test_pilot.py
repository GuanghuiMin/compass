"""The pilot's selection, its schedule, and the counting rules, all fixed before anything runs.

The load-bearing test here is the first one. Prior outcomes exist for every task in `test_normal`,
so the pilot's honesty rests entirely on the selection never consulting them -- and that is a claim
about what a module reads, which is checkable, rather than about what someone intended.

Everything else defends a number's denominator. Each of these rules can flatter or damage a method,
so they are written down once, tested against synthetic cells whose outcomes are known by
construction, and applied to every condition alike.
"""

import json
from pathlib import Path

import pytest

from future_graph.pilot import (
    ALGORITHM, CONDITIONS, DIFFICULTIES, PER_DIFFICULTY, REPLICATES, SELECTION_KEY, SPLIT,
    PilotError, build_schedule, check_schedule, digest, load_manifests, load_split,
    schedule_manifest, scenario_prefix, select, write_manifests,
)
from future_graph.pilot_aggregate import (
    COMPLETED, FUTURE_GRAPH, INTEGRATION_FAILURE, NOT_LAUNCHED, OPERATIONAL_FAILURE,
    PilotAggregationError, aggregate, check_row_against_artifact, compatibility_row,
    denominators, matched_blocks, plumbing, success_both_ways, zero_boundary_cells,
)

SOURCE = Path(__file__).resolve().parents[1] / "src" / "future_graph"


# --------------------------------------------------------------------------- the contract

def test_selection_can_only_see_the_split_the_ids_and_the_difficulty():
    """Every task in test_normal already has outcomes recorded under many methods. The pilot is
    exploratory rather than an untouched holdout, and the guarantee is the narrow one: nothing that
    holds an outcome is reachable from this module."""
    source = (SOURCE / "pilot.py").read_text()
    for forbidden in ("run_records", "summary.md", "method_compare", "pertest_deltas",
                      "trajectories/", "artifacts/", "eval_success", "run_dir",
                      "boundaries", "handover"):
        assert forbidden not in source, f"selection must not be able to reach {forbidden!r}"
    # And it imports nothing that could reach one on its behalf: no artifact module, no run
    # module, nothing from this package at all.
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["from __future__ import annotations", "import hashlib", "import json",
                       "from dataclasses import dataclass", "from pathlib import Path"]


def test_selection_reads_only_two_kinds_of_file(tmp_path, monkeypatch):
    """The real guarantee, checked by watching what it opens rather than by reading its source."""
    root = _fixture(tmp_path, scenarios=18)
    opened = []
    real = Path.read_bytes
    real_text = Path.read_text

    def watch_bytes(self, *a, **kw):
        opened.append(str(self))
        return real(self, *a, **kw)

    def watch_text(self, *a, **kw):
        opened.append(str(self))
        return real_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", watch_bytes)
    monkeypatch.setattr(Path, "read_text", watch_text)
    select(root / "datasets" / "test_normal.txt", root)
    for path in opened:
        assert path.endswith("test_normal.txt") or path.endswith("ground_truth/metadata.json")


# --------------------------------------------------------------------------- determinism

def _fixture(tmp_path, scenarios=9, variants=3):
    """A synthetic split with no outcomes anywhere near it."""
    root = tmp_path / "data"
    (root / "datasets").mkdir(parents=True)
    ids = []
    for s in range(scenarios):
        for v in range(1, variants + 1):
            task_id = f"scen{s:03d}_{v}"
            ids.append(task_id)
            meta = root / "tasks" / task_id / "ground_truth"
            meta.mkdir(parents=True)
            (meta / "metadata.json").write_text(json.dumps({"difficulty": (s % 3) + 1}))
    (root / "datasets" / "test_normal.txt").write_text("\n".join(ids))
    return root


def test_the_same_inputs_give_the_same_fifteen_tasks(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    first = select(root / "datasets" / "test_normal.txt", root)
    second = select(root / "datasets" / "test_normal.txt", root)
    assert first.task_ids == second.task_ids
    assert first.manifest["selected_task_list_sha256"] == \
        second.manifest["selected_task_list_sha256"]


def test_the_order_of_the_split_file_changes_nothing(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    path = root / "datasets" / "test_normal.txt"
    ordered = select(path, root)
    ids = path.read_text().split()
    path.write_text("\n".join(reversed(ids)))
    assert select(path, root).task_ids == ordered.task_ids


def test_five_of_each_difficulty_and_one_variant_per_scenario(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    selection = select(root / "datasets" / "test_normal.txt", root)
    assert len(selection.tasks) == PER_DIFFICULTY * len(DIFFICULTIES) == 15
    for difficulty in DIFFICULTIES:
        assert sum(1 for t in selection.tasks if t.difficulty == difficulty) == PER_DIFFICULTY
    prefixes = [t.scenario_prefix for t in selection.tasks]
    assert len(set(prefixes)) == len(prefixes)


def test_a_split_without_enough_scenarios_is_refused_rather_than_padded(tmp_path):
    root = _fixture(tmp_path, scenarios=6)          # two scenarios per difficulty
    with pytest.raises(PilotError, match="eligible scenarios"):
        select(root / "datasets" / "test_normal.txt", root)


def test_the_variant_is_chosen_by_hash_and_not_by_being_first(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    selection = select(root / "datasets" / "test_normal.txt", root)
    chosen = {t.scenario_prefix: t.task_id for t in selection.tasks}
    for prefix, task_id in chosen.items():
        siblings = [f"{prefix}_{v}" for v in (1, 2, 3)]
        expected = min(siblings, key=lambda t: (digest(SELECTION_KEY, t), t))
        assert task_id == expected
    assert not all(t.endswith("_1") for t in chosen.values())


def test_the_scenario_prefix_is_the_id_without_its_variant():
    assert scenario_prefix("5238afc_1") == "5238afc"
    assert scenario_prefix("5238afc_3") == "5238afc"
    assert scenario_prefix("042a9fc_11") == "042a9fc"


def test_a_task_without_metadata_is_refused(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    for meta in (root / "tasks").glob("scen000_*/ground_truth/metadata.json"):
        meta.unlink()
    with pytest.raises(PilotError, match="no ground-truth metadata"):
        select(root / "datasets" / "test_normal.txt", root)


def test_a_duplicated_task_id_is_refused(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    path = root / "datasets" / "test_normal.txt"
    path.write_text(path.read_text() + "\nscen000_1")
    with pytest.raises(PilotError, match="lists a task twice"):
        load_split(path)


# --------------------------------------------------------------------------- the schedule

@pytest.fixture
def scheduled(tmp_path):
    root = _fixture(tmp_path, scenarios=18)
    selection = select(root / "datasets" / "test_normal.txt", root)
    return selection, build_schedule(selection)


def test_the_schedule_is_one_hundred_and_twenty_named_cells(scheduled):
    _, cells = scheduled
    check_schedule(cells)
    assert len(cells) == 15 * len(CONDITIONS) * len(REPLICATES) == 120
    assert len({c.cell_id for c in cells}) == 120
    assert len({c.run_id for c in cells}) == 120


def test_every_block_holds_all_four_conditions_and_both_replicates(scheduled):
    _, cells = scheduled
    blocks = {}
    for cell in cells:
        blocks.setdefault((cell.task_id, cell.replicate), []).append(cell.condition)
    assert len(blocks) == 30
    assert all(sorted(v) == sorted(CONDITIONS) for v in blocks.values())
    assert {r for _, r in blocks} == set(REPLICATES)


def test_no_condition_always_launches_first(scheduled):
    _, cells = scheduled
    firsts = [c.condition for c in cells if c.position == 0]
    assert len(set(firsts)) == len(CONDITIONS), \
        "one method always going first would confound it with anything that drifts over the run"
    assert max(firsts.count(c) for c in CONDITIONS) < len(firsts)


def test_the_launch_order_is_fixed_by_hash_and_reproducible(scheduled):
    selection, cells = scheduled
    again = build_schedule(selection)
    assert [c.to_dict() for c in cells] == [c.to_dict() for c in again]
    for cell in cells:
        assert cell.schedule_hash == digest(SELECTION_KEY, "schedule", cell.task_id,
                                            str(cell.replicate), cell.condition)


def test_a_cell_id_names_condition_task_and_replicate(scheduled):
    _, cells = scheduled
    cell = cells[0]
    assert cell.cell_id == f"{cell.condition}/{cell.task_id}/replicate_{cell.replicate}"
    assert "/" not in cell.run_id


def test_the_manifests_are_written_once_and_belong_together(scheduled, tmp_path):
    selection, cells = scheduled
    directory = tmp_path / "pilot"
    write_manifests(selection, cells, directory)
    tasks, schedule = load_manifests(directory)
    assert tasks["algorithm"] == ALGORITHM and tasks["split"] == SPLIT
    assert schedule["planned_cells"] == 120
    assert (directory / "tasks.txt").read_text().split() == list(selection.task_ids)
    with pytest.raises(PilotError, match="chosen once"):
        write_manifests(selection, cells, directory)


def test_a_schedule_built_for_another_task_list_is_refused(scheduled, tmp_path):
    selection, cells = scheduled
    directory = tmp_path / "pilot"
    write_manifests(selection, cells, directory)
    schedule = json.loads((directory / "schedule.json").read_text())
    schedule["selected_task_list_sha256"] = "0" * 64
    (directory / "schedule.json").write_text(json.dumps(schedule))
    with pytest.raises(PilotError, match="different task list"):
        load_manifests(directory)


# --------------------------------------------------------------------------- counting

def row(condition, task, replicate, outcome=COMPLETED, success=True, difficulty=1, **kw):
    base = {"condition": condition, "task_id": task, "replicate": replicate,
            "cell_id": f"{condition}/{task}/replicate_{replicate}", "difficulty": difficulty,
            "success": success, "score": 1.0 if success else 0.0, "num_steps": 10,
            "peak_prompt_tokens": 100, "cumulative_input_tokens": 500,
            "termination_reason": "task_completed", "outcome": outcome}
    base.update(kw)
    return base


def test_a_refused_boundary_never_removes_a_cell_from_the_denominator():
    """The task continued under the previous handover; its outcome is an outcome."""
    rows = [row(FUTURE_GRAPH, "t1", 1, success=False, refused_boundaries=3, boundaries=3)]
    counts = success_both_ways(rows)
    assert counts["excluding_operational"]["of"] == 1
    assert counts["excluding_operational"]["passed"] == 0


def test_operational_failures_are_reported_both_ways():
    rows = [row(FUTURE_GRAPH, "t1", 1, success=True),
            row(FUTURE_GRAPH, "t2", 1, success=False),
            row(FUTURE_GRAPH, "t3", 1, outcome=OPERATIONAL_FAILURE, success=False)]
    counts = success_both_ways(rows)
    assert counts["operational_failures"] == 1
    assert counts["excluding_operational"] == {"passed": 1, "of": 2, "rate": 0.5}
    assert counts["counting_operational_as_failure"] == {"passed": 1, "of": 3,
                                                         "rate": round(1 / 3, 4)}


def test_a_cell_that_never_launched_is_not_in_the_denominator():
    rows = [row(FUTURE_GRAPH, "t1", 1), row(FUTURE_GRAPH, "t2", 1, outcome=NOT_LAUNCHED)]
    counts = denominators(rows, planned=2)
    assert counts.planned == 2 and counts.launched == 1 and counts.not_launched == 1
    assert success_both_ways(rows)["excluding_operational"]["of"] == 1


def test_zero_boundary_cells_stay_in_the_denominator_and_are_named():
    rows = [row(FUTURE_GRAPH, "t1", 1, boundaries=0), row(FUTURE_GRAPH, "t2", 1, boundaries=2)]
    assert success_both_ways(rows)["excluding_operational"]["of"] == 2
    assert zero_boundary_cells(rows) == ["future_graph_v1/t1/replicate_1"]


def test_a_matched_block_needs_all_four_conditions_to_have_finished():
    rows = [row(c, "t1", 1) for c in CONDITIONS]
    rows += [row(c, "t2", 1) for c in CONDITIONS[:3]]
    rows += [row("full_context", "t2", 1, outcome=OPERATIONAL_FAILURE)]
    assert matched_blocks(rows) == [("t1", 1)]


def test_the_aggregate_carries_every_denominator_and_no_statistic():
    rows = [row(c, f"t{i}", r, success=(i % 2 == 0))
            for c in CONDITIONS for i in range(1, 4) for r in REPLICATES]
    summary = aggregate(rows, planned_per_condition=6)
    for condition in CONDITIONS:
        cell = summary["conditions"][condition]
        assert cell["planned"] == 6 and cell["launched"] == 6
        assert "excluding_operational" in cell["success"]
        assert "counting_operational_as_failure" in cell["success"]
    assert summary["matched_complete_block_count"] == 6
    assert "no significance test" in summary["note"]
    assert not _statistic_keys(summary), "the pilot reports counts and rates, and nothing inferred"


def _statistic_keys(value, found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("p_value", "confidence_interval", "ci_low", "ci_high", "significant",
                       "std_err", "effect_size", "t_stat"):
                found.append(key)
            _statistic_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _statistic_keys(item, found)
    return found


def test_an_unknown_condition_is_refused_rather_than_counted():
    with pytest.raises(PilotAggregationError, match="not a pilot condition"):
        aggregate([row("llmlingua", "t1", 1)], planned_per_condition=1)


def test_a_third_replicate_is_refused():
    with pytest.raises(PilotAggregationError, match="replicate 3"):
        aggregate([row(FUTURE_GRAPH, "t1", 3)], planned_per_condition=1)


# --------------------------------------------------------------------------- canonical artifacts

def committed_run():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "online"
    runs = sorted(p for p in root.glob("*") if (p / "manifest.json").exists()) \
        if root.is_dir() else []
    if not runs:
        pytest.skip("no online run has been committed yet")
    return runs[0]


def test_a_row_is_derived_from_the_artifact_and_rebuilds_from_it():
    run_dir = committed_run()
    task_id = json.loads((run_dir / "manifest.json").read_text())["task_id"]
    first = compatibility_row(run_dir, condition=FUTURE_GRAPH, task_id=task_id, replicate=1)
    check_row_against_artifact(first)                       # rebuilds and compares
    second = compatibility_row(run_dir, condition=FUTURE_GRAPH, task_id=task_id, replicate=1)
    assert first == second


def test_a_row_that_disagrees_with_its_artifact_refuses_the_aggregation():
    run_dir = committed_run()
    task_id = json.loads((run_dir / "manifest.json").read_text())["task_id"]
    tampered = compatibility_row(run_dir, condition=FUTURE_GRAPH, task_id=task_id, replicate=1)
    tampered["success"] = not tampered["success"]
    with pytest.raises(PilotAggregationError, match="the artifact is the record"):
        check_row_against_artifact(tampered)


def test_a_row_for_the_wrong_task_is_refused():
    run_dir = committed_run()
    with pytest.raises(PilotAggregationError, match="the schedule says"):
        compatibility_row(run_dir, condition=FUTURE_GRAPH, task_id="not_this_task", replicate=1)


def test_a_row_carries_both_repository_commits_and_the_manifest_hash():
    run_dir = committed_run()
    task_id = json.loads((run_dir / "manifest.json").read_text())["task_id"]
    derived = compatibility_row(run_dir, condition=FUTURE_GRAPH, task_id=task_id, replicate=1)
    assert set(derived["repos"]) == {"compass-v2", "trace-v2"}
    assert all(len(sha) == 40 for sha in derived["repos"].values())
    assert len(derived["canonical_manifest_sha256"]) == 64
    assert derived["canonical_artifact"].endswith(run_dir.name)
    assert derived["instrumentation"] >= 1


def test_plumbing_is_read_from_the_artifact_and_never_shaped_for_comparison():
    run_dir = committed_run()
    measured = plumbing(run_dir)
    assert measured["strict_load"] and measured["leftover_pending"] == 0
    assert len(measured["boundaries"]) == json.loads(
        (run_dir / "manifest.json").read_text())["boundaries"]
    for boundary in measured["boundaries"]:
        assert {"accepted", "revision_bytes", "handover_bytes", "attempts"} <= set(boundary)
    assert all(c["handover_present"] for c in measured["continuations"])
