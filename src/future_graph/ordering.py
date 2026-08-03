"""An ordering edge from a computation to itself, which cannot mean anything.

`PRECEDES` is strict: it says one computation runs before another. `EDGE c1 PRECEDES c1` says c1
runs before c1, which constrains nothing and can be satisfied by no execution. There is no other
target the text could have meant, so removing it is not choosing a dependency -- it is deleting an
edge that carries none.

This is a change to the graph and is recorded with the other graph edits, not with the parser's
normalizations. Normalizations are tolerated surface; this removes a relation.

**Only `PRECEDES`.** A `REFINES` self-loop looks similar and is not: it may be standing where a
missing child should be, and dropping it would turn a refined obligation into a leaf and quietly
change the plan. That has no unique repair, so it stays and the acyclicity check refuses it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Relation
from .state_graph import StateGraph, _safe_order


@dataclass(frozen=True)
class OrderingRepair:
    """One reflexive ordering edge removed."""
    action: str            # always "removed": this step adds nothing
    source: str
    relation: Relation
    target: str


def remove_reflexive_precedes(
        graph: StateGraph) -> tuple[StateGraph, tuple[OrderingRepair, ...]]:
    """Return the graph without any `PRECEDES` edge whose two ends are the same computation."""
    reflexive = [edge for edge in graph.edges_of(Relation.PRECEDES)
                 if edge.source == edge.target]
    if not reflexive:
        return graph, ()

    dropped = {(edge.source, edge.target) for edge in reflexive}
    repaired = StateGraph()
    for node in (*graph.information, *graph.computations):
        repaired.add(node)
    for edge in graph.edges:
        if edge.relation is Relation.PRECEDES and (edge.source, edge.target) in dropped:
            continue
        repaired.add_edge(edge.source, edge.relation, edge.target)

    repairs = tuple(OrderingRepair("removed", source, Relation.PRECEDES, target)
                    for source, target in sorted(dropped, key=lambda p: _safe_order(p[0])))
    return repaired, repairs
