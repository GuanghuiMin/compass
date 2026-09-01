"""Run the invented practice cases through the regeneration operator, once each.

This is a diagnostic, not an experiment and not a harness.

**What it is trying to see.** Compass claims something narrower and harder than "a model can write a
plan from a transcript": that given a plan it already committed to and one new interaction, it
revises that plan and then keeps only what the revised plan consumes. So each case here is a
*transition*, and it needs two sides:

    PREVIOUS_GRAPH   the remaining plan as it stood before the new observation arrived
    DELTA_H          only the action and observation that trigger this update

Handing the operator an empty graph and the whole prefix would measure one-shot construction
instead. It would still show whether an error was understood, and it would show nothing about
whether an obsolete branch is dropped, whether shared information survives, or whether the model
read `PREVIOUS_GRAPH` at all -- a model that ignored the previous graph entirely would score the
same. Those are the properties that distinguish this from summarizing a trajectory.

**Where the previous graphs come from.** They are written by hand, below, one per case. Generating
them with a first model call would put two model errors in one result with no way to tell which
produced the outcome. Each fixture expresses only what the case's own text says had been committed
before the triggering step -- the plan, the information it consumes, and nothing about the recovery
the observation will call for. **A fixture is not an expected answer**, and there is no expected
answer here: nothing in this file scores, grades or compares. Whether a transition was absorbed is
decided by reading the record.

The fixtures are validated before any request is sent, so a broken one fails at the desk rather
than after spending a call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph import (                                            # noqa: E402
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    Relation, RuntimeReferencePayload, build,
)
from future_graph.adapter import EmptyModelCompletion, from_environment   # noqa: E402
from future_graph.artifacts import RegenerationRecord                 # noqa: E402
from future_graph.episodes import REPO_ROOT                           # noqa: E402
from future_graph.regeneration import regenerate_graph                # noqa: E402
from future_graph.run import FROZEN_CONFIG                            # noqa: E402
from future_graph.state_graph import StateGraph                       # noqa: E402
from future_graph.validation import validate                          # noqa: E402

CASES_PATH = REPO_ROOT / "inputs" / "diagnostic" / "practice_cases.json"


def _c(cid, description, **kw):
    return ComputationNode(id=cid, description=description, **kw)


def _i(iid, description, kind=InformationKind.FACT, available=True, payload=None):
    return InformationNode(id=iid, kind=kind, description=description, available=available,
                           payload=payload)


# --------------------------------------------------------------------------- previous graphs
#
# One per case, expressing the plan the case's own reasoning says was already committed to at the
# moment the triggering step ran. Nothing here anticipates the observation.

def _previous_02():
    """The delivery has been listed. Twelve seedlings are to be registered, then failures flagged."""
    return build(
        nodes=[_i("i1", "The twelve seedlings of delivery 4821, the first of which is 91001"),
               _i("i2", "The catalogue entries for all twelve seedlings",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Register each of the twelve seedlings in the spring catalogue"),
               _c("c2", "Flag the seedlings that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2")])


def _previous_03():
    """Registering is still an abstract obligation: nothing is known about what it involves."""
    return build(
        nodes=[_i("i1", "The twelve seedlings of delivery 4821"),
               _i("i2", "The registered catalogue entries",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Register each seedling in the nursery catalogue"),
               _c("c2", "Flag the seedlings that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c2")])


def _previous_04():
    """Registration is done. Consulting the inspection service is committed but not worked out."""
    return build(
        nodes=[_i("i1", "The catalogue entries for the twelve seedlings of delivery 4821",
                  kind=InformationKind.RESULT),
               _c("c1", "Flag the seedlings of delivery 4821 that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1")])


def _previous_05():
    """Registering has been broken into the three calls the documentation described."""
    return build(
        nodes=[_i("i1", "The twelve seedlings of delivery 4821, beginning with 91001"),
               _i("i2", "The twelve completed catalogue entries",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Register each of the twelve seedlings in the spring catalogue"),
               _c("c2", "Open a catalogue entry for each seedling",
                  operation="apis.nursery.create_entry", arguments={"catalogue": "spring"}),
               _c("c3", "Attach the intake photograph to each entry",
                  operation="apis.nursery.attach_photo"),
               _c("c4", "Set each entry's status", operation="apis.nursery.set_status"),
               _c("c5", "Flag the seedlings that failed inspection")],
        edges=[("c1", Relation.REFINES, "c2"), ("c1", Relation.REFINES, "c3"),
               ("c1", Relation.REFINES, "c4"),
               ("i1", Relation.INTERFACE_INPUT, "c1"), ("i1", Relation.REQUIRES, "c2"),
               ("c1", Relation.INTERFACE_OUTPUT, "i2"), ("c4", Relation.PRODUCES, "i2"),
               ("i2", Relation.REQUIRES, "c5"),
               ("c2", Relation.PRECEDES, "c3"), ("c3", Relation.PRECEDES, "c4")])


def _previous_06():
    """One batch call for the whole delivery, then the failure flags.

    The delivery, the catalogue and the token are consumed by the batch call *and* by the work
    after it; the batch interface is consumed by the batch call alone. What survives the retirement
    of that interface, and what goes with it, is the thing this case is for.
    """
    return build(
        nodes=[_i("i1", "Delivery 4821, this week's delivery"),
               _i("i2", "The spring catalogue is where this delivery is registered"),
               _i("i3", "The curator token the nursery calls require",
                  kind=InformationKind.RUNTIME_REFERENCE,
                  payload=RuntimeReferencePayload("curator_token")),
               _i("i4", "The confirmed batch registration interface",
                  kind=InformationKind.CONTRACT,
                  payload=ContractPayload("apis.nursery.bulk_register",
                                          ("delivery_id", "catalogue", "curator_token"))),
               _i("i5", "The catalogue entries for every seedling in the delivery",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Register every seedling of the delivery in one batch call",
                  operation="apis.nursery.bulk_register",
                  arguments={"delivery_id": 4821, "catalogue": "spring",
                             "curator_token": InformationReference("i3")}),
               _c("c2", "Flag the seedlings of the delivery that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1"),
               ("i3", Relation.REQUIRES, "c1"), ("i4", Relation.REQUIRES, "c1"),
               ("c1", Relation.PRODUCES, "i5"),
               ("i5", Relation.REQUIRES, "c2"), ("i1", Relation.REQUIRES, "c2")])


def _previous_07():
    """Eleven seedlings still to register, into the spring catalogue, one at a time."""
    return build(
        nodes=[_i("i1", "The eleven seedlings of delivery 4821 still to be registered"),
               _i("i2", "The spring catalogue is the destination for this delivery"),
               _i("i3", "The curator token the nursery calls require",
                  kind=InformationKind.RUNTIME_REFERENCE,
                  payload=RuntimeReferencePayload("curator_token")),
               _i("i4", "The catalogue entries for all twelve seedlings",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Open a catalogue entry in the spring catalogue for each remaining "
                        "seedling", operation="apis.nursery.create_entry",
                  arguments={"catalogue": InformationReference("i2"),
                             "curator_token": InformationReference("i3")}),
               _c("c2", "Flag the seedlings that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1"),
               ("i3", Relation.REQUIRES, "c1"), ("c1", Relation.PRODUCES, "i4"),
               ("i4", Relation.REQUIRES, "c2")])


def _previous_08():
    """Everything is registered. The plan is one report call for the delivery, then the flags."""
    return build(
        nodes=[_i("i1", "Delivery 4821, this week's delivery"),
               _i("i2", "The confirmed inspection report interface",
                  kind=InformationKind.CONTRACT,
                  payload=ContractPayload("apis.inspection.fetch_report", ("delivery_id",),
                                          ("returns the report for a whole delivery",))),
               _i("i3", "The inspection report for the delivery",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Fetch the inspection report for the delivery",
                  operation="apis.inspection.fetch_report", arguments={"delivery_id": 4821}),
               _c("c2", "Flag each seedling the report lists as having failed")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1"),
               ("c1", Relation.PRODUCES, "i3"), ("i3", Relation.REQUIRES, "c2"),
               ("i1", Relation.REQUIRES, "c2")])


def _previous_11():
    """Registration is under way, one seedling at a time, with the token already in hand."""
    return build(
        nodes=[_i("i1", "The seedlings of delivery 4821 that are not yet registered"),
               _i("i2", "The curator token the nursery calls require",
                  kind=InformationKind.RUNTIME_REFERENCE,
                  payload=RuntimeReferencePayload("curator_token")),
               _i("i3", "The catalogue entries for all twelve seedlings",
                  kind=InformationKind.RESULT, available=False),
               _c("c1", "Open a catalogue entry for each seedling that remains",
                  operation="apis.nursery.create_entry",
                  arguments={"catalogue": "spring",
                             "curator_token": InformationReference("i2")}),
               _c("c2", "Flag the seedlings that failed inspection")],
        edges=[("i1", Relation.REQUIRES, "c1"), ("i2", Relation.REQUIRES, "c1"),
               ("c1", Relation.PRODUCES, "i3"), ("i3", Relation.REQUIRES, "c2")])


PREVIOUS_GRAPHS = {
    "synthetic_02": _previous_02,
    "synthetic_03": _previous_03,
    "synthetic_04": _previous_04,
    "synthetic_05": _previous_05,
    "synthetic_06": _previous_06,
    "synthetic_07": _previous_07,
    "synthetic_08": _previous_08,
    "synthetic_11": _previous_11,
}


# --------------------------------------------------------------------------- the slice

def build_slice(case: dict) -> str:
    """Only the step that triggers this update.

    The earlier steps of a case are the story that produced the previous graph, and they are
    represented there. Replaying them here as well would hand the model the same evidence twice and
    turn a transition back into a reconstruction.
    """
    step = case["prefix"][-1]
    parts = []
    for label, key in (("REASONING", "reasoning"), ("CODE", "code"),
                       ("OBSERVATION", "observation")):
        value = step.get(key)
        if value:
            parts.append(f"{label}:\n{value}")
    return "\n\n".join(parts)


def load_cases(path: Path, wanted: list[str]) -> tuple[list[dict], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    by_id = {case["practice_id"]: case for case in document["cases"]}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise SystemExit(f"no such practice case: {', '.join(missing)}")
    unfixtured = [name for name in wanted if name not in PREVIOUS_GRAPHS]
    if unfixtured:
        # Refused rather than started from nothing: an empty previous graph would quietly turn a
        # transition into a reconstruction, and the record would not say which had been measured.
        raise SystemExit(f"no previous-graph fixture for: {', '.join(unfixtured)}")
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

    # A fixture that does not hold together would produce a record nobody could interpret, so they
    # are all checked before the first request rather than one call at a time.
    for case in cases:
        previous = PREVIOUS_GRAPHS[case["practice_id"]]()
        violations = validate(previous)
        if violations:
            raise SystemExit(f"{case['practice_id']}: the previous-graph fixture is not valid: "
                             + "; ".join(str(v) for v in violations))

    # Everything that can fail without sending a request is settled first, and the directory is
    # claimed last -- the ordering `run.py` uses, for its reason: a directory taken before the
    # environment was checked is an empty output nobody can use and a path nobody can reuse.
    model = from_environment()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists, and a run never writes into an existing directory")
    args.out.mkdir(parents=True, exist_ok=False)

    unanswered = []
    for case in cases:
        name = case["practice_id"]
        previous = PREVIOUS_GRAPHS[name]()
        delta_h = build_slice(case)
        print(f"{name}: previous graph {len(previous)} nodes, slice {len(delta_h)} bytes",
              flush=True)
        try:
            result = regenerate_graph(case["goal"], shared_rules, previous, delta_h,
                                      model, FROZEN_CONFIG)
        except EmptyModelCompletion as err:
            # Not a refused graph: nothing was generated. Recorded as the call event it is, in a
            # file no one can mistake for a boundary record, and the run moves on. It is not
            # retried -- a retry here would silently turn one sample into several.
            (args.out / f"{name}.model-call-failure.json").write_text(
                json.dumps({"practice_id": name, "event": "empty_model_completion",
                            "detail": str(err)}, indent=1) + "\n", encoding="utf-8")
            unanswered.append(name)
            print(f"{name}: no answer from the provider; not a graph result", flush=True)
            continue
        write_record(args.out / f"{name}.json", result.record)
        verdict = "accepted" if result.record.accepted else "refused"
        print(f"{name}: {verdict}, "
              f"{len(result.record.parse_errors)} parse errors, "
              f"{len(result.record.violations)} violations", flush=True)

    print(f"\nrecords in {args.out}")
    if unanswered:
        print(f"no answer at all for: {', '.join(unanswered)} — these are call events, not "
              "graph results, and must not be counted as refusals")
    print("Read them. Parsing and validating is not absorbing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
