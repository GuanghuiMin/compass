"""Run the invented practice cases through the regeneration operator, once each.

This is a diagnostic, not an experiment and not a harness. It reads the committed practice cases,
turns each one into the four inputs the operator already takes, calls it once, and writes the
ordinary `RegenerationRecord` so the result can be read the same way any boundary is read.

What it deliberately does not do, because each would make its output something other than evidence:

  * it holds no state between cases -- every case starts from an empty graph, because a practice
    case is one decision point and not an episode;
  * it knows no expected answer, scores nothing, and computes no verdict. Whether a case absorbed
    its evidence is a judgement made by reading the record, and a number here would invite the
    judgement to be skipped;
  * it does not emulate the compaction loop, hold a per-episode graph, or bridge any host.

The slice is built from the case's own fields and nothing else. Each recorded step becomes the
reasoning, the code and the observation exactly as the case states them, in the CodeAct shape the
frozen corpus uses, and the result goes into `delta_h` unmodified from there on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph.adapter import from_environment                     # noqa: E402
from future_graph.artifacts import RegenerationRecord                 # noqa: E402
from future_graph.episodes import REPO_ROOT                           # noqa: E402
from future_graph.regeneration import regenerate_graph                # noqa: E402
from future_graph.run import FROZEN_CONFIG                            # noqa: E402
from future_graph.state_graph import StateGraph                       # noqa: E402

CASES_PATH = REPO_ROOT / "inputs" / "diagnostic" / "practice_cases.json"


def build_slice(case: dict) -> str:
    """The case's recorded steps, in the CodeAct shape, and nothing added.

    One blank line between steps and none at the end, so that two runs of the same case produce the
    same string and a diff between records is a difference in the model rather than in this file.
    """
    blocks = []
    for step in case["prefix"]:
        parts = []
        for label, key in (("REASONING", "reasoning"), ("CODE", "code"),
                           ("OBSERVATION", "observation")):
            value = step.get(key)
            if value:
                parts.append(f"{label}:\n{value}")
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def load_cases(path: Path, wanted: list[str]) -> tuple[list[dict], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    by_id = {case["practice_id"]: case for case in document["cases"]}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise SystemExit(f"no such practice case: {', '.join(missing)}")
    return [by_id[name] for name in wanted], document["shared_rules"]


def write_record(path: Path, record: RegenerationRecord) -> None:
    """Write, read back, then place -- the same promise `run.py` makes about a boundary record."""
    payload = record.to_dict()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    try:
        RegenerationRecord.from_dict(json.loads(temporary.read_text(encoding="utf-8")))
    except Exception as err:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"{path.name} did not read back: {err}") from err
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="a directory to create; an existing one is refused")
    parser.add_argument("--case", action="append", required=True, dest="cases",
                        help="a practice_id, repeatable")
    parser.add_argument("--cases-path", default=CASES_PATH, type=Path)
    args = parser.parse_args(argv)

    cases, shared_rules = load_cases(Path(args.cases_path), args.cases)

    # Everything that can fail without sending a request is settled first, and the directory is
    # claimed last -- the ordering `run.py` uses, for its reason: a directory taken before the
    # environment was checked is an empty output nobody can use and a path nobody can reuse.
    model = from_environment()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists, and a run never writes into an existing directory")
    args.out.mkdir(parents=True, exist_ok=False)
    for case in cases:
        name = case["practice_id"]
        delta_h = build_slice(case)
        print(f"{name}: {len(delta_h)} bytes of slice", flush=True)
        result = regenerate_graph(case["goal"], shared_rules, StateGraph(), delta_h,
                                  model, FROZEN_CONFIG)
        write_record(args.out / f"{name}.json", result.record)
        verdict = "accepted" if result.record.accepted else "refused"
        print(f"{name}: {verdict}, "
              f"{len(result.record.parse_errors)} parse errors, "
              f"{len(result.record.violations)} violations", flush=True)
    print(f"\nrecords in {args.out}")
    print("Read them. Parsing and validating is not absorbing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
