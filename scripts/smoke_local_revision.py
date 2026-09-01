"""One synthetic boundary through the local-revision updater, printed in full.

Not a measurement and not part of any evaluation. It exists to show that a real model, given a real
previous graph and a real slice, writes a revision this code can read and apply -- and to show
exactly what it wrote, so the parts the model authored can be told apart from the parts the code
derived. One boundary, one call, no retry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from future_graph import ComputationNode as C, InformationNode as I     # noqa: E402
from future_graph import InformationKind as K, Relation as R, build     # noqa: E402
from future_graph.adapter import from_environment                       # noqa: E402
from future_graph.protocol import to_protocol                           # noqa: E402
from future_graph.run import FROZEN_CONFIG                              # noqa: E402
from future_graph.update import update_graph                            # noqa: E402

GOAL = ("Register all twelve of the nursery's new seedlings in the public catalogue, each with its "
        "photo, and set every one of them to visible.")

RULES = ("Use only the documented nursery APIs. Do not modify entries that already exist. "
         "Report what you did when the task is complete.")

PREVIOUS = build(
    nodes=[C(id="c1", description="Register every seedling in the catalogue"),
           C(id="c2", description="Open a catalogue entry for all twelve seedlings at once",
             operation="apis.nursery.create_entries", arguments={"count": 12}),
           C(id="c3", description="Attach each seedling's photo to its entry",
             operation="apis.nursery.attach_photo"),
           C(id="c4", description="Set every entry to visible",
             operation="apis.nursery.set_status", arguments={"status": "visible"}),
           I(id="i1", kind=K.FACT, description="The twelve seedlings awaiting registration",
             available=True)],
    edges=[("c1", R.REFINES, "c2"), ("c1", R.REFINES, "c3"), ("c1", R.REFINES, "c4"),
           ("i1", R.INTERFACE_INPUT, "c1"), ("i1", R.REQUIRES, "c2"),
           ("c2", R.PRECEDES, "c3"), ("c3", R.PRECEDES, "c4")])

DELTA_H = """\
ASSISTANT: I will open the catalogue entries for all twelve seedlings in one call.
print(apis.nursery.create_entries(count=12))

USER: Execution failed: 410 Gone.
{"error": "create_entries was retired on 2024-03-01. Use create_entry, which registers one
seedling at a time and requires a curator token. Obtain a token with apis.nursery.login(
username, password); it expires after 30 minutes. create_entry takes (curator_token,
seedling_id) and returns the new entry_id."}

ASSISTANT: The bulk route is gone. Let me check what credentials I have.
print(apis.supervisor.show_account_passwords())

USER: [{"account_name": "nursery", "password": "GreenHouse4!"}]

ASSISTANT: I have the nursery password. The account username is curator@nursery.example.
"""


def main() -> int:
    model = from_environment()
    result = update_graph(GOAL, RULES, PREVIOUS, DELTA_H, model, FROZEN_CONFIG)
    record = result.record

    print("=" * 78)
    print("PREVIOUS GRAPH, AS THE MODEL SAW IT")
    print("=" * 78)
    print(to_protocol(PREVIOUS))

    print("=" * 78)
    print("MODEL OUTPUT, EXACTLY AS RETURNED")
    print("=" * 78)
    print(record.raw_output)

    print("=" * 78)
    print("WHAT THE CODE DID")
    print("=" * 78)
    print(json.dumps({
        "accepted": record.accepted,
        "empty_revision": record.empty_revision,
        "normalizations": list(record.normalizations),
        "parse_errors": [list(e) for e in record.parse_errors],
        "faults": [[c, m, list(n)] for c, m, n in record.faults],
        "violations": [[c, m, list(n)] for c, m, n in record.violations],
        "affected_roots": list(record.affected_roots),
        "touched_nodes": list(record.touched_nodes),
        "removed_nodes": [list(p) for p in record.removed_nodes],
        "removed_edges": [list(c) for c in record.removed_edges],
        "replacement_boundary_changes": [list(c) for c in record.replacement_boundary_changes],
        "completion_changes": [list(c) for c in record.completion_changes],
        "argument_dependency_changes": [list(c) for c in record.argument_dependency_changes],
        "interface_changes": [list(c) for c in record.interface_changes],
        "ordering_repairs": [list(c) for c in record.ordering_repairs],
        "collected": list(record.collected),
        "id_map": [list(p) for p in record.id_map],
    }, indent=2))

    print("=" * 78)
    print("RESULTING GRAPH")
    print("=" * 78)
    print(to_protocol(result.graph))

    print("=" * 78)
    print("HANDOVER")
    print("=" * 78)
    print(record.handover)

    return 0 if record.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
