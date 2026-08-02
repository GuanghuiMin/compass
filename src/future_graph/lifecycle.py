"""Collecting what nothing needs, and swapping one graph for another without a middle state.

Two rules, and both are about not being clever. Information with no consumer goes, and code never
guesses who the consumer should have been -- an edge the generator did not write is an edge that does
not exist. A candidate lands whole or not at all, and a rejected one leaves the previous graph
identical down to its serialization, so a refusal can never be the thing that lost a contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import InterfaceChange, complete_interfaces
from .state_graph import StateGraph
from .validation import Violation, unconsumed_information, validate


@dataclass(frozen=True)
class Replacement:
    """What one boundary did, and enough to re-examine it later."""
    accepted: bool
    graph: StateGraph
    violations: tuple[Violation, ...] = ()
    collected: tuple[str, ...] = field(default=())
    interface_changes: tuple[InterfaceChange, ...] = field(default=())

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
    """Complete the code-owned interfaces, validate the whole thing, and only then swap.

    Completion comes first because the thing validated has to be the thing committed. It is not a
    repair: it replaces the edges of a relation the model does not own with the ones the dataflow
    implies, and a graph that still does not hold together afterwards is still refused.

    Collection happens after validation and before the swap, so nothing is ever deleted on account of
    a graph that turned out to be invalid. The candidate is never mutated -- completion returns a new
    graph -- so a refusal leaves both graphs exactly as they were.
    """
    completed, interface_changes = complete_interfaces(candidate)
    violations = validate(completed)
    if violations:
        return Replacement(accepted=False, graph=previous, violations=violations,
                           interface_changes=interface_changes)
    collected = collect_dead_information(completed)
    return Replacement(accepted=True, graph=completed, collected=collected,
                       interface_changes=interface_changes)
