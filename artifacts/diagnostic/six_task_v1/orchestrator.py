"""Drive the twelve diagnostic cells with the already-pushed runners, changing no code.

Not committed to either repository and not importable by anything: it only shells out. The two
runners it invokes are exactly the ones at the authorised commits, and every command it issues is
written into the cell log so the run can be read back from the artifacts alone.

Artifacts are staged outside both repositories. A cell refuses to start unless both trees are
clean, so writing the first cell's artifact into compass-v2 would make every later cell refuse.
They are moved in at publication time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TRACE = Path("/workspace/trace-v2-online/motivation")
PY = "/workspace/acon/.venv/bin/python"
STAGE = Path("/workspace/pilot_artifacts/diagnostic_six_task_v1")
METHOD = "/workspace/trace-v2-online/external_prompts/future_graph_v1"
SPLIT = "test_normal"
MAX_STEPS = 50
WINDOW = 4096

# Per task, the two conditions in the order the committed 120-cell schedule fixed for replicate 1.
PLAN = [
    ("f3f60f0_2", 1, ["future_graph_v1", "full_context"]),
    ("29a7b7e_2", 1, ["full_context", "future_graph_v1"]),
    ("3d9a636_2", 2, ["full_context", "future_graph_v1"]),
    ("d194965_1", 2, ["future_graph_v1", "full_context"]),
    ("83a7951_3", 3, ["full_context", "future_graph_v1"]),
    ("986aa4e_2", 3, ["future_graph_v1", "full_context"]),
]

LOG = STAGE / "cells.jsonl"


def command_for(condition: str, task_id: str) -> list[str]:
    if condition == "future_graph_v1":
        return [PY, str(TRACE / "scripts" / "55_run_future_graph_online.py"),
                "--task", task_id, "--split", SPLIT, "--method", METHOD,
                "--window", str(WINDOW), "--max-steps", str(MAX_STEPS), "--workers", "1",
                "--run-id", f"future_graph_v1__{task_id}__replicate_1",
                "--artifact-root", str(STAGE / "future_graph_v1")]
    return [PY, str(TRACE / "scripts" / "01_run_full_context_trajectories.py"),
            "--split", SPLIT, "--task_ids", task_id,
            "--cap_steps", str(MAX_STEPS), "--workers", "1",
            "--out", str(STAGE / "full_context" / f"{task_id}.jsonl")]


def append(payload: dict) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def lane(task_id: str, difficulty: int, order: list[str]) -> list[dict]:
    """One task's two cells, in the committed order, never reordered on what the first did."""
    results = []
    for position, condition in enumerate(order):
        command = command_for(condition, task_id)
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        started_at = time.clock_gettime(time.CLOCK_REALTIME)
        done = subprocess.run(command, capture_output=True, text=True, cwd=str(TRACE))
        record = {
            "cell_id": f"{condition}/{task_id}/replicate_1",
            "condition": condition, "task_id": task_id, "difficulty": difficulty,
            "replicate": 1, "order_in_task": position, "command": command,
            "worker_pid": os.getpid(),
            "started_at_epoch": started_at,
            "elapsed_s": round(time.clock_gettime(time.CLOCK_MONOTONIC) - started, 2),
            "returncode": done.returncode,
            "stdout_tail": (done.stdout or "")[-3000:],
            "stderr_tail": (done.stderr or "")[-3000:],
        }
        append(record)
        results.append(record)
        print(f"  {record['cell_id']}: rc={done.returncode} in {record['elapsed_s']}s",
              flush=True)
    return results


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "full_context").mkdir(exist_ok=True)
    print(f"12 cells across 6 task lanes, at most 6 concurrent")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(lane, task, difficulty, order)
                   for task, difficulty, order in PLAN]
        for future in as_completed(futures):
            future.result()
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    bad = [r["cell_id"] for r in rows if r["returncode"] != 0]
    print(f"\n{len(rows)} cells finished, {len(bad)} non-zero exits: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
