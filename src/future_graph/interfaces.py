"""What crosses a refinement boundary, and who is responsible for saying so.

There is one definition of "crosses", and it lives here. Validation and completion both call these
functions rather than each computing the answer, because two implementations of the same set would
eventually disagree and the disagreement would look like a model error.

**The ownership split.** Most of a refinement interface is a function of the dataflow underneath it,
so the code derives it:

    every INTERFACE_INPUT                      derived
    every INTERFACE_OUTPUT that is unavailable  derived
    an INTERFACE_OUTPUT that is available       the model's to declare

The last line is not an inconsistency, it is the one case the graph cannot answer. An available
information node may have no producer at all, so a result a partly-finished subtree already
delivered -- its producing child having left the graph when it completed -- has exactly the shape of
a value established somewhere else entirely: available, no `PRODUCES` edge, consumed outside the
subtree. Deriving it would mean either dropping the fact that the subtree delivered it, or claiming
that every available value a subtree touches came from it. Only whoever read the trajectory knows,
so that declaration stays with the model and is validated rather than generated.

Completion is not repair. It adds and removes edges of a relation the model does not own; it never
invents a computation, an information node, a consumer or a producer, and a graph that does not hold
together after completion is still refused.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Relation
from .state_graph import StateGraph, _safe_order


@dataclass(frozen=True)
class InterfaceChange:
    """One edge of a code-owned relation, taken out of the candidate or put into it."""
    action: str            # "removed" or "added"
    source: str
    relation: Relation
    target: str


# --------------------------------------------------------------------------- the sets

def refinement_region(graph: StateGraph, computation_id: str) -> tuple[set[str], tuple[str, ...]]:
    """Everything below this computation, and the leaves among it.

    Both come from `state_graph`, whose traversals carry a visited set: this runs on candidates
    that have not been validated yet, so a refinement cycle must terminate here rather than hang.
    """
    inside = set(graph.refinement_descendants_of(computation_id))
    return inside, graph.descendant_leaves_of(computation_id)


def _known_information(graph: StateGraph) -> set[str]:
    """Ids an information node actually declares. An edge may name anything at all."""
    return {node.id for node in graph.information}


def crossing_inputs(graph: StateGraph, computation_id: str) -> set[str]:
    """Information the refined work needs and does not establish for itself."""
    inside, leaves = refinement_region(graph, computation_id)
    known = _known_information(graph)
    produced_inside = {i for c in inside for i in graph.produces_of(c)}
    required_inside = {i for leaf in leaves for i in graph.requires_of(leaf)}
    return {i for i in required_inside if i not in produced_inside} & known


def crossing_unavailable_outputs(graph: StateGraph, computation_id: str) -> set[str]:
    """Information the refined work will establish for something outside itself.

    Restricted to information that does not exist yet, which is the part the structure settles:
    exactly one leaf below produces it, and at least one consumer sits outside.
    """
    inside, leaves = refinement_region(graph, computation_id)
    known = _known_information(graph)
    crossing = set()
    for information_id in {i for leaf in leaves for i in graph.produces_of(leaf)} & known:
        if graph.node(information_id).available:
            continue
        producers = [leaf for leaf in leaves if information_id in graph.produces_of(leaf)]
        outside = [c for c in graph.consumers_of(information_id) if c not in inside]
        if len(producers) == 1 and outside:
            crossing.add(information_id)
    return crossing


def code_owned(graph: StateGraph, edge) -> bool:
    """Is this an edge of a relation the code owns, and therefore not the model's to write?

    Every `INTERFACE_INPUT`, however it was written -- correctly, redundantly, or reversed -- and
    every `INTERFACE_OUTPUT` naming information that does not exist yet.
    """
    if edge.relation is Relation.INTERFACE_INPUT:
        return True
    if edge.relation is not Relation.INTERFACE_OUTPUT:
        return False
    if edge.target not in _known_information(graph):
        return False           # not a declaration this owns; dangling, and reported as that
    return not graph.node(edge.target).available


# --------------------------------------------------------------------------- completion

def complete_interfaces(graph: StateGraph) -> tuple[StateGraph, tuple[InterfaceChange, ...]]:
    """Return the candidate with the code-owned interface edges replaced by the derived ones.

    A new graph rather than a mutation, so a candidate that is later refused cannot have been
    altered on the way to being refused.
    """
    removed = [edge for edge in graph.edges if code_owned(graph, edge)]
    kept = [edge for edge in graph.edges if edge not in set(removed)]

    derived: list[tuple[str, Relation, str]] = []
    for computation in graph.computations:
        if not graph.is_coarse(computation.id):
            continue
        for information_id in sorted(crossing_inputs(graph, computation.id), key=_safe_order):
            derived.append((information_id, Relation.INTERFACE_INPUT, computation.id))
        for information_id in sorted(crossing_unavailable_outputs(graph, computation.id),
                                     key=_safe_order):
            derived.append((computation.id, Relation.INTERFACE_OUTPUT, information_id))

    completed = StateGraph()
    for node in (*graph.information, *graph.computations):
        completed.add(node)
    for edge in kept:
        completed.add_edge(edge.source, edge.relation, edge.target)
    for source, relation, target in derived:
        completed.add_edge(source, relation, target)

    # An edge the model wrote that the derivation also produces is not a change: it was removed and
    # put back identically, and reporting it would bury the real edits in noise.
    removed_set = {(e.source, e.relation, e.target) for e in removed}
    derived_set = set(derived)
    changes = [InterfaceChange("removed", s, r, t)
               for s, r, t in sorted(removed_set - derived_set, key=_change_order)]
    changes += [InterfaceChange("added", s, r, t)
                for s, r, t in sorted(derived_set - removed_set, key=_change_order)]
    return completed, tuple(changes)


def _change_order(edge: tuple[str, Relation, str]) -> tuple:
    source, relation, target = edge
    return relation.value, _safe_order(source), _safe_order(target)
