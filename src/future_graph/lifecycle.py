"""Collecting what nothing needs, and swapping one graph for another without a middle state.

Two rules, and both are about not being clever. Information with no consumer goes, and code never
guesses who the consumer should have been -- an edge the generator did not write is an edge that does
not exist. A candidate lands whole or not at all, and a rejected one leaves the previous graph
identical down to its serialization, so a refusal can never be the thing that lost a contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .state_graph import StateGraph
from .validation import Violation, unconsumed_information, validate


@dataclass(frozen=True)
class Replacement:
    """What one boundary did, and enough to re-examine it later."""
    accepted: bool
    graph: StateGraph
    violations: tuple[Violation, ...] = ()
    collected: tuple[str, ...] = field(default=())

    @property
    def rejected(self) -> bool:
        return not self.accepted


def collect_dead_information(graph: StateGraph) -> tuple[str, ...]:
    """Remove information no computation requires. Returns what went, in order.

    One pass is enough: information only ever points at computations, so removing information cannot
    make other information dead. The loop is here to make that claim testable rather than assumed.
    """
    removed: list[str] = []
    while True:
        dead = unconsumed_information(graph)
        if not dead:
            return tuple(removed)
        for node_id in dead:
            graph.remove(node_id)
            removed.append(node_id)


def replace(previous: StateGraph, candidate: StateGraph) -> Replacement:
    """Validate a whole candidate and, only if it holds together, let it become the state.

    Collection happens after validation and before the swap, so nothing is ever deleted on account of
    a graph that turned out to be invalid.
    """
    violations = validate(candidate)
    if violations:
        return Replacement(accepted=False, graph=previous, violations=violations)
    collected = collect_dead_information(candidate)
    return Replacement(accepted=True, graph=candidate, collected=collected)
