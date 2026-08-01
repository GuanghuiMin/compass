"""The handover the downstream agent reads: the future, and nothing else.

No history section, no corrections, no raw trajectory, no status words, no inventory of information
sitting apart from the work that needs it. Information appears under the computations that consume it,
because that is the only reason it is here.

Information is written out once, at its first structural mention in rendered order: under its future
producer when it has one, otherwise under its first consumer. Every later mention is the id alone, so
the text reads top to bottom with nothing referred to before it has been defined.

A rendered scalar keeps the type the graph holds: `7` and `"7"` are different states and must not read
the same. The protocol's canonical form is reused rather than reimplemented, so the two cannot drift.
"""

from __future__ import annotations

from .frontier import is_executable, ordered_computations
from .protocol import format_scalar
from .schema import (
    ComputationNode, ContractPayload, InformationNode, InformationReference, ListPayload,
    MappingPayload, RuntimeReferencePayload, ScalarPayload,
)
from .state_graph import StateGraph


def render(graph: StateGraph) -> str:
    """A deterministic view of the remaining work."""
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
