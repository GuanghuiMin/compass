"""What can be done now, worked out from the graph.

Nothing is stored. A computation is not marked ready or blocked; it is ready when nothing precedes it
and everything it requires is available, and the graph says both. The previous implementation carried a
status word per node, which meant the graph and the status could disagree and the status won.

Whatever holds a computation up is visible in the same graph: an edge from unfinished work, or an edge
from information that is not available yet, whose producer is upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .schema import ComputationNode, Relation
from .state_graph import StateGraph, _order


@dataclass(frozen=True)
class Blocker:
    """Why a computation is not on the frontier, in the graph's own terms."""
    computation_id: str
    waiting_for_computations: tuple[str, ...]
    waiting_for_information: tuple[str, ...]


def is_executable(graph: StateGraph, computation_id: str) -> bool:
    if graph.predecessors_of(computation_id):
        return False
    return all(graph.node(i).available for i in graph.requires_of(computation_id))


def frontier(graph: StateGraph) -> tuple[ComputationNode, ...]:
    """Computations that could run now, in the graph's stable order."""
    return tuple(c for c in graph.computations if is_executable(graph, c.id))


def blockers(graph: StateGraph) -> tuple[Blocker, ...]:
    out = []
    for computation in graph.computations:
        if is_executable(graph, computation.id):
            continue
        pending = tuple(i for i in graph.requires_of(computation.id)
                        if not graph.node(i).available)
        out.append(Blocker(computation.id, graph.predecessors_of(computation.id), pending))
    return tuple(out)


def ordered_computations(graph: StateGraph) -> tuple[ComputationNode, ...]:
    """Every computation in dependency order, ties broken by id so the order never wobbles."""
    plain = nx.DiGraph()
    ids = [c.id for c in graph.computations]
    plain.add_nodes_from(ids)
    for edge in graph.edges_of(Relation.PRECEDES):
        if edge.source in plain and edge.target in plain:
            plain.add_edge(edge.source, edge.target)
    # information a computation produces also orders it before that information's consumers
    for edge in graph.edges_of(Relation.PRODUCES):
        for consumer in graph.consumers_of(edge.target):
            if edge.source in plain and consumer in plain:
                plain.add_edge(edge.source, consumer)
    order = list(nx.lexicographical_topological_sort(plain, key=_order))
    by_id = {c.id: c for c in graph.computations}
    return tuple(by_id[i] for i in order)
