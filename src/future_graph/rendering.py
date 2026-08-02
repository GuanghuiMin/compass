"""The handover the downstream agent reads: the future, and nothing else.

No history section, no corrections, no raw trajectory, no status words, no inventory of information
sitting apart from the work that needs it. Information appears under the computations that consume it,
because that is the only reason it is here.

Information is written out once, at its first structural mention in rendered order: under its future
producer when it has one, otherwise under its first consumer. Every later mention is the id alone, so
the text reads top to bottom with nothing referred to before it has been defined.

A rendered scalar keeps the type the graph holds: `7` and `"7"` are different states and must not read
the same. The protocol's canonical form is reused rather than reimplemented, so the two cannot drift.

Where the graph is refined, active work is shown in full and distant work stays coarse. Every piece of
information the reader can see is still defined exactly once; information that only a hidden subtree
touches is not shown at all, because a definition with nothing visible to consume it is the free-
floating note this format exists to avoid.
"""

from __future__ import annotations

from .frontier import is_executable, ordered_computations
from .protocol import format_scalar
from .schema import (
    ComputationNode, ContractPayload, InformationNode, InformationReference, ListPayload,
    MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload,
)
from .state_graph import StateGraph


def render(graph: StateGraph) -> str:
    """A deterministic view of the remaining work.

    A graph with no refinement in it takes the path it has always taken and comes out byte for
    byte as before. That is a branch rather than a promise on purpose: the frozen replays and the
    artifacts they produced were rendered by `_render_flat`, and a heading renamed for the sake of
    tidiness would silently invalidate every one of them.
    """
    if graph.edges_of(Relation.REFINES):
        return _render_refined(graph)
    return _render_flat(graph)


def _render_flat(graph: StateGraph) -> str:
    ordered = ordered_computations(graph)
    now = [c for c in ordered if is_executable(graph, c.id)]
    later = [c for c in ordered if not is_executable(graph, c.id)]
    seen: set[str] = set()

    lines: list[str] = []
    if now:
        lines.append("CURRENT COMPUTATIONS")
        lines.append("")
        for computation in now:
            lines += _computation_block(graph, computation, seen)
    if later:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("LATER COMPUTATIONS")
        lines.append("")
        for computation in later:
            lines += _computation_block(graph, computation, seen)
    if not lines:
        return "NOTHING REMAINS"
    return "\n".join(lines).rstrip() + "\n"


def _render_refined(graph: StateGraph) -> str:
    """The refined plan, the work being done now in full, and what follows it.

    Three sections and each computation in exactly one of them. A distant subtree that has been
    refined is not expanded: its parent states what it needs and what it will establish, which is
    what a reader deciding what to do next actually uses, and its internals would be detail about
    work that is not happening yet.

    The first section covers the refined roots and not the whole remaining plan, and it is named
    for what it holds. An abstract leaf that is a root of nothing appears where it can be acted on.
    """
    ordered = ordered_computations(graph)
    executable = {c.id for c in ordered if is_executable(graph, c.id)}

    active: set[str] = set()
    for computation_id in executable:
        active.add(computation_id)
        active.update(graph.refinement_ancestors_of(computation_id))

    coarse_roots = [c for c in ordered
                    if not graph.refinement_parents_of(c.id) and graph.is_coarse(c.id)]
    seen: set[str] = set()
    written: set[str] = set()
    lines: list[str] = []

    def section(title: str, computations: list, block) -> None:
        if not computations:
            return
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(title)
        lines.append("")
        for computation in computations:
            written.add(computation.id)
            lines.extend(block(graph, computation, seen))

    section("REFINED PLAN OVERVIEW", coarse_roots, _coarse_block)

    # ACTIVE WORK rather than ACTIVE REFINEMENT: what lands here is whatever can run and whatever
    # stands above it, and an executable abstract leaf that was never refined is not a refinement
    # path. The selection is unchanged; only the name was wrong.
    on_the_path = [c for c in ordered if c.id in active and c.id not in written]
    section("ACTIVE WORK", on_the_path,
            lambda g, c, s: (_coarse_block(g, c, s) if g.is_coarse(c.id)
                             else _computation_block(g, c, s)))

    later = [c for c in ordered
             if c.id not in written and graph.is_leaf(c.id) and _is_visible(graph, c.id, active)]
    section("LATER COMPUTATIONS", later, _computation_block)

    if not lines:
        return "NOTHING REMAINS"
    return "\n".join(lines).rstrip() + "\n"


def _is_visible(graph: StateGraph, computation_id: str, active: set[str]) -> bool:
    """A leaf is shown when it is a root, or when the work it is part of is under way."""
    ancestors = graph.refinement_ancestors_of(computation_id)
    if not ancestors:
        return True
    return any(ancestor in active for ancestor in ancestors)


def _coarse_block(graph: StateGraph, computation: ComputationNode,
                  seen: set[str]) -> list[str]:
    """A refined computation: what it is for, what governs it, and its boundary.

    The requirements come first and are not the interface. They are what the obligation itself
    consumes -- a closed route, a restriction on how it may be done -- and leaving them out of the
    handover would drop the one thing standing between the agent and a repeat of a failure, while
    the graph went on holding it.
    """
    lines = [f"[{computation.id}] {computation.description}"]
    requires = graph.requires_of(computation.id)
    if requires:
        lines.append("Needs:")
        lines += [f"- {_information_line(graph.node(i), seen)}" for i in requires]
    inputs = graph.interface_inputs_of(computation.id)
    if inputs:
        lines.append("Interface in:")
        lines += [f"- {_information_line(graph.node(i), seen)}" for i in inputs]
    outputs = graph.interface_outputs_of(computation.id)
    if outputs:
        lines.append("Interface out:")
        lines += [f"- {_information_line(graph.node(i), seen)}" for i in outputs]
    children = graph.refinement_children_of(computation.id)
    if children:
        lines.append("Refined into: " + ", ".join(children))
    predecessors = graph.predecessors_of(computation.id)
    if predecessors:
        lines.append("Depends on: " + ", ".join(predecessors))
    lines.append("")
    return lines


def _computation_block(graph: StateGraph, computation: ComputationNode,
                       seen: set[str]) -> list[str]:
    lines = [f"[{computation.id}] {computation.description}"]

    requires = graph.requires_of(computation.id)
    if requires:
        lines.append("Needs:")
        for information_id in requires:
            lines.append(f"- {_information_line(graph.node(information_id), seen)}")

    if computation.operation:
        lines.append(f"Operation: {computation.operation}")
    if computation.arguments:
        lines.append("Arguments:")
        for name, value in sorted(computation.arguments.items()):
            lines.append(f"- {name} = {_argument(value)}")

    produces = graph.produces_of(computation.id)
    if produces:
        lines.append("Produces:")
        for information_id in produces:
            lines.append(f"- {_information_line(graph.node(information_id), seen)}")

    predecessors = graph.predecessors_of(computation.id)
    if predecessors:
        lines.append("Depends on: " + ", ".join(predecessors))

    lines.append("")
    return lines


def _information_line(node: InformationNode, seen: set[str]) -> str:
    """Defined once with its kind; every later mention is the id, which the reader has already met."""
    if node.id in seen:
        return f"[{node.id}]"
    seen.add(node.id)
    text = f"[{node.id}|{node.kind.value}] {node.description}"
    detail = _payload(node)
    if detail:
        text += f" ({detail})"
    if not node.available:
        text += " [not yet available]"
    return text


def _payload(node: InformationNode) -> str:
    payload = node.payload
    if payload is None:
        return ""
    if isinstance(payload, RuntimeReferencePayload):
        return f"bound as {payload.name}"
    if isinstance(payload, ContractPayload):
        parts = [payload.operation]
        if payload.parameters:
            parts.append("takes " + ", ".join(payload.parameters))
        if payload.constraints:
            parts.append("; ".join(payload.constraints))
        return ", ".join(parts)
    if isinstance(payload, ScalarPayload):
        return format_scalar(payload.value)
    if isinstance(payload, ListPayload):
        # Brackets, so a string and a list of one string do not read the same, and an empty list
        # does not read like an empty mapping.
        return "[" + ", ".join(format_scalar(v) for v in payload.values) + "]"
    if isinstance(payload, MappingPayload):
        return "{" + ", ".join(f"{k}={format_scalar(v)}" for k, v in payload.values) + "}"
    return ""


def _argument(value) -> str:
    if isinstance(value, InformationReference):
        return f"@{value.information_id}"
    return format_scalar(value)
