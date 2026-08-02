"""Every way a candidate graph can fail to hold together, reported together.

Collecting all violations rather than raising on the first is not a convenience. In the previous
implementation one check fired wrongly -- refusing a name the agent had genuinely bound -- and because
validation stopped there, nothing is known about the rest of those candidates. A refusal that names one
fault teaches nothing about the others, and a refusal that names the wrong fault hides them.

What is *not* here matters as much. Whether a contract or a runtime reference was really established by
the environment is not checked, because the only inputs a check could use are the previous graph and
the current slice, and a check against "what the state happens to hold" is not a check against reality:
it made anything the graph dropped once permanently unmentionable. That question is an audit question.

Liveness is also not here. Information nobody consumes is not an error to refuse, it is garbage to
collect, and `lifecycle.collect_dead_information` does that deterministically.

What refinement adds is one idea checked three ways: a coarse computation states an interface, its
leaves do the work, and the two talk about the same information nodes. Nothing here matches a coarse
input to a leaf requirement by name, description or similarity -- they are the same node or the
graph is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .schema import ENDPOINTS, EntityType, InformationReference, Relation
from .state_graph import StateGraph


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    nodes: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{', '.join(self.nodes)}]" if self.nodes else ""
        return f"{self.code}: {self.message}{where}"


def validate(graph: StateGraph) -> tuple[Violation, ...]:
    """All structural and semantic faults of a candidate, in a stable order."""
    found: list[Violation] = []
    found += _dangling(graph)
    found += _endpoints(graph)
    found += _cycles(graph)
    found += _availability(graph)
    found += _argument_references(graph)
    found += _refinement_parents(graph)
    found += _coarse_and_leaf_roles(graph)
    found += _interface_realization(graph)
    return tuple(found)


def _dangling(graph: StateGraph) -> list[Violation]:
    return [Violation("dangling_edge",
                      f"an edge points at {nid!r}, which no node declares", (nid,))
            for nid in graph.dangling_ids]


def _endpoints(graph: StateGraph) -> list[Violation]:
    out: list[Violation] = []
    declared = set(graph.dangling_ids)
    for edge in graph.edges:
        if edge.source in declared or edge.target in declared:
            continue          # already reported as dangling; its type is unknowable
        want_source, want_target = ENDPOINTS[edge.relation]
        got_source, got_target = graph.kind_of(edge.source), graph.kind_of(edge.target)
        if got_source is not want_source or got_target is not want_target:
            out.append(Violation(
                "endpoint_type",
                f"{edge.relation.value} runs {want_source.value} -> {want_target.value}, "
                f"but this one runs {_name(got_source)} -> {_name(got_target)}",
                (edge.source, edge.target)))
    return out


def _name(kind: EntityType | None) -> str:
    return kind.value if kind is not None else "nothing"


def _cycles(graph: StateGraph) -> list[Violation]:
    plain = graph.as_digraph()
    if nx.is_directed_acyclic_graph(plain):
        return []
    cycle = next(iter(nx.simple_cycles(plain)), ())
    return [Violation("cycle", "the graph contains a cycle, so nothing in it can be reached",
                      tuple(cycle))]


def _availability(graph: StateGraph) -> list[Violation]:
    out: list[Violation] = []
    for node in graph.information:
        producers = graph.producers_of(node.id)
        if not node.available and len(producers) != 1:
            out.append(Violation(
                "availability",
                "information that is not available yet must be produced by exactly one future "
                f"computation, and this one has {len(producers)}",
                (node.id, *producers)))
    return out


def _argument_references(graph: StateGraph) -> list[Violation]:
    out: list[Violation] = []
    information_ids = {i.id for i in graph.information}
    for computation in graph.computations:
        required = set(graph.requires_of(computation.id))
        for name, value in sorted(computation.arguments.items()):
            if not isinstance(value, InformationReference):
                continue
            target = value.information_id
            if target not in information_ids:
                out.append(Violation(
                    "unknown_argument_reference",
                    f"argument {name!r} names {target!r}, which is not an information node",
                    (computation.id, target)))
            elif target not in required:
                out.append(Violation(
                    "unlinked_argument_reference",
                    f"argument {name!r} uses {target!r} without a requires edge saying so; "
                    "nothing here adds the edge for you",
                    (computation.id, target)))
    return out


def _refinement_parents(graph: StateGraph) -> list[Violation]:
    """A computation belongs to one unit of work, so it is refined from at most one place."""
    out: list[Violation] = []
    for computation in graph.computations:
        parents = graph.refinement_parents_of(computation.id)
        if len(parents) > 1:
            out.append(Violation(
                "multiple_refinement_parents",
                f"a computation is refined from {len(parents)} places at once, and it belongs to "
                "one unit of work",
                (computation.id, *parents)))
    return out


def _coarse_and_leaf_roles(graph: StateGraph) -> list[Violation]:
    """A refined computation declares an interface; a leaf does the work.

    Keeping the two kinds of edge apart is what stops a coarse node from quietly holding
    execution state, and what stops a leaf from declaring an interface nothing realizes.
    """
    out: list[Violation] = []
    for computation in graph.computations:
        interface = (graph.interface_inputs_of(computation.id)
                     + graph.interface_outputs_of(computation.id))
        if graph.is_coarse(computation.id):
            if computation.operation is not None or computation.arguments:
                out.append(Violation(
                    "coarse_is_executable",
                    "a refined computation carries an operation or arguments, but the work is "
                    "its children's",
                    (computation.id,)))
            operational = graph.requires_of(computation.id) + graph.produces_of(computation.id)
            if operational:
                out.append(Violation(
                    "coarse_operational_edge",
                    "a refined computation uses requires or produces; a coarse node states its "
                    "interface and its leaves do the consuming and producing",
                    (computation.id, *operational)))
        elif interface:
            out.append(Violation(
                "leaf_interface_edge",
                "a computation with no children carries an interface edge; an interface is "
                "realized by descendants, and this one has none",
                (computation.id, *interface)))
    return out


def _interface_realization(graph: StateGraph) -> list[Violation]:
    """The coarse interface and the execution below it name the same information nodes.

    Not a similar node, not a copy of the value, not an entry in a table beside the graph. The
    identity is the check: an input a coarse node declares is required by a leaf underneath it,
    and an output it promises is produced by one.
    """
    out: list[Violation] = []
    declared = {node.id for node in graph.information}
    for computation in graph.computations:
        if not graph.is_coarse(computation.id):
            continue
        leaves = graph.descendant_leaves_of(computation.id)

        for information_id in graph.interface_inputs_of(computation.id):
            if information_id not in declared:
                continue                  # already reported as dangling
            if not any(information_id in graph.requires_of(leaf) for leaf in leaves):
                out.append(Violation(
                    "unrealized_interface_input",
                    "a coarse computation declares an input that no descendant leaf requires",
                    (computation.id, information_id)))

        for information_id in graph.interface_outputs_of(computation.id):
            if information_id not in declared:
                continue
            if graph.node(information_id).available:
                # Already established. Whatever produced it has left the graph, which is how a
                # completed computation collapses while its output stays for what comes next.
                continue
            producers = [leaf for leaf in leaves if information_id in graph.produces_of(leaf)]
            if len(producers) != 1:
                out.append(Violation(
                    "unrealized_interface_output",
                    "a coarse computation promises information that is not available yet, so "
                    f"exactly one descendant leaf must produce it, and {len(producers)} do",
                    (computation.id, information_id, *producers)))
    return out


def unconsumed_information(graph: StateGraph) -> tuple[str, ...]:
    """Information no surviving computation requires. Collected, never refused.

    Deliberately unchanged by refinement. An interface edge does not keep information alive: a
    valid interface input is required by a descendant leaf anyway, so counting the interface edge
    would add nothing except a way for dead information to survive on a promise.
    """
    return tuple(node.id for node in graph.information if not graph.consumers_of(node.id))
