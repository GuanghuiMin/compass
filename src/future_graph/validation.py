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

from .interfaces import crossing_inputs, crossing_unavailable_outputs, refinement_region
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
    """A refinement boundary is complete: its interface is everything that crosses it.

    Not merely "whatever was declared is realized". That would leave declaring optional, and a
    coarse computation could reach outside itself without saying so -- which is the difference
    between a boundary and an annotation. The declared set and the crossing set are compared in
    both directions, so an undeclared crossing and a declaration that crosses nothing are each a
    refusal.

    The identity is the whole mechanism: not a similar node, not a copy of the value, not an entry
    in a table beside the graph, and never a match inferred from a name.

    Each boundary is judged on its own, which is what makes the rules compose. Information one
    child establishes and another child consumes crosses each of those two boundaries and not the
    boundary of the parent they share.
    """
    out: list[Violation] = []
    known = {node.id for node in graph.information}
    for computation in graph.computations:
        if not graph.is_coarse(computation.id):
            continue
        inside, leaves = refinement_region(graph, computation.id)
        required_inside = {i for leaf in leaves for i in graph.requires_of(leaf)}
        out += _crossing_inputs(graph, computation.id, known, required_inside)
        out += _crossing_outputs(graph, computation.id, known, inside, leaves)
    return out


def _crossing_inputs(graph: StateGraph, coarse_id: str, known: set[str],
                     required_inside: set[str]) -> list[Violation]:
    """Information the refined work needs and does not establish for itself.

    The set comes from `interfaces.crossing_inputs`, the same function completion uses. Under the
    ownership split these edges are code-generated, so on a completed candidate the two sides agree
    by construction and nothing here fires. It is kept because `validate` is also called directly,
    on hand-built graphs and on graphs loaded from a snapshot, and because a completion bug should
    be caught by the invariant rather than by whatever reads the graph next.
    """
    declared = {i for i in graph.interface_inputs_of(coarse_id) if i in known}
    crossing = crossing_inputs(graph, coarse_id)

    out: list[Violation] = []
    for information_id in sorted(crossing - declared):
        out.append(Violation(
            "undeclared_interface_input",
            "a leaf inside this refinement requires information from outside it, and the coarse "
            "computation does not declare it as an interface input",
            (coarse_id, information_id)))
    for information_id in sorted(declared - crossing):
        if information_id not in required_inside:
            out.append(Violation(
                "unrealized_interface_input",
                "a coarse computation declares an input that no descendant leaf requires",
                (coarse_id, information_id)))
        else:
            out.append(Violation(
                "internal_information_declared_as_input",
                "a coarse computation declares as an interface input something its own refinement "
                "produces, so it does not cross the boundary",
                (coarse_id, information_id)))
    return out


def _crossing_outputs(graph: StateGraph, coarse_id: str, known: set[str],
                      inside: set[str], leaves: tuple[str, ...]) -> list[Violation]:
    """Information the refined work establishes for something outside itself."""
    def producers_of(information_id: str) -> list[str]:
        return [leaf for leaf in leaves if information_id in graph.produces_of(leaf)]

    def consumers_outside(information_id: str) -> list[str]:
        return [c for c in graph.consumers_of(information_id) if c not in inside]

    declared = [i for i in graph.interface_outputs_of(coarse_id) if i in known]
    out: list[Violation] = []

    for information_id in declared:
        producers = producers_of(information_id)
        if graph.node(information_id).available:
            # Already established, so whatever produced it has left the graph. That is how a
            # completed computation collapses while its output stays for what comes next, and it
            # is why a descendant still producing it would be a contradiction.
            if producers:
                out.append(Violation(
                    "available_interface_output_is_produced_inside",
                    "a coarse computation declares an output that already exists and is also "
                    "produced by one of its own leaves",
                    (coarse_id, information_id, *producers)))
                continue
        elif len(producers) != 1:
            out.append(Violation(
                "unrealized_interface_output",
                "a coarse computation promises information that is not available yet, so "
                f"exactly one descendant leaf must produce it, and {len(producers)} do",
                (coarse_id, information_id, *producers)))
            continue
        if not consumers_outside(information_id):
            out.append(Violation(
                "internal_information_declared_as_output",
                "a coarse computation declares as an interface output something nothing outside "
                "the refinement consumes, so it does not cross the boundary",
                (coarse_id, information_id)))

    for information_id in sorted(crossing_unavailable_outputs(graph, coarse_id) - set(declared)):
        out.append(Violation(
            "undeclared_interface_output",
            "a leaf inside this refinement establishes information consumed outside it, and "
            "the coarse computation does not declare it as an interface output",
            (coarse_id, information_id)))
    return out


def unconsumed_information(graph: StateGraph) -> tuple[str, ...]:
    """Information no surviving computation requires. Collected, never refused.

    Deliberately unchanged by refinement. An interface edge does not keep information alive: a
    valid interface input is required by a descendant leaf anyway, so counting the interface edge
    would add nothing except a way for dead information to survive on a promise.
    """
    return tuple(node.id for node in graph.information if not graph.consumers_of(node.id))
