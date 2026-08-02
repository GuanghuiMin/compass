"""The dependency an argument reference already states.

`argument curator_token = @i2` on a computation says three things without ambiguity: which
information is consumed, which computation consumes it, and that the relation is `REQUIRES`. Asking
the model to write the matching edge as well adds a way for a sound graph to be refused and adds
nothing else, so the code writes it.

This is not the interface completion in `interfaces.py` and is deliberately not recorded with it. An
interface edge is derived from a whole subtree's dataflow; this is one edge restating one argument.

**Additive only.** Nothing is removed here. If the model also required the same information at the
refined computation above, that edge stays: deciding whether the obligation is genuinely governed by
it, or whether the model simply attached it a level too high, is a judgement about meaning, and the
code has no way to make it. The result may say the same thing twice for a while. Saying it twice is
honest; guessing which one to delete is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import InformationReference, Relation
from .state_graph import StateGraph, _safe_order


@dataclass(frozen=True)
class ArgumentDependency:
    """One `REQUIRES` edge added because an argument reference already implied it."""
    action: str            # always "added": this step removes nothing
    source: str
    relation: Relation
    target: str


def complete_argument_dependencies(
        graph: StateGraph) -> tuple[StateGraph, tuple[ArgumentDependency, ...]]:
    """Return the graph with every argument reference's `REQUIRES` edge present.

    A new graph rather than a mutation, so a candidate that is later refused was not altered on the
    way to being refused.

    It runs before interface completion, and the order matters: an edge added here can be the reason
    an information node crosses a refinement boundary, so deriving the interfaces first would leave
    the boundary incomplete.
    """
    known = {node.id for node in graph.information}
    existing = {(edge.source, edge.target) for edge in graph.edges_of(Relation.REQUIRES)}

    missing: list[tuple[str, str]] = []
    for computation in graph.computations:
        for _, value in sorted(computation.arguments.items()):
            if not isinstance(value, InformationReference):
                continue
            pair = (value.information_id, computation.id)
            if value.information_id not in known:
                continue          # names nothing; validation reports it, and inventing it here
                                  # would be creating the node the reference got wrong
            if pair in existing or pair in missing:
                continue
            missing.append(pair)

    completed = StateGraph()
    for node in (*graph.information, *graph.computations):
        completed.add(node)
    for edge in graph.edges:
        completed.add_edge(edge.source, edge.relation, edge.target)
    for source, target in missing:
        completed.add_edge(source, Relation.REQUIRES, target)

    changes = tuple(ArgumentDependency("added", source, Relation.REQUIRES, target)
                    for source, target in sorted(missing,
                                                 key=lambda p: (_safe_order(p[0]),
                                                                _safe_order(p[1]))))
    return completed, changes
