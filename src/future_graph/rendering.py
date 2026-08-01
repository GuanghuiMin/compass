"""The handover the downstream agent reads: the future, and nothing else.

No history section, no corrections, no raw trajectory, no status words, no inventory of information
sitting apart from the work that needs it. Information appears under the computations that consume it,
because that is the only reason it is here.

A piece of information consumed by several computations is written out under each of them. Defining it
once and referring to it by id would save a line and cost the reader a lookup; the line is cheaper.
"""

from __future__ import annotations

from .frontier import is_executable, ordered_computations
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

    lines: list[str] = []
    if now:
        lines.append("CURRENT COMPUTATIONS")
        lines.append("")
        for computation in now:
            lines += _computation_block(graph, computation)
    if later:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("LATER COMPUTATIONS")
        lines.append("")
        for computation in later:
            lines += _computation_block(graph, computation)
    if not lines:
        return "NOTHING REMAINS"
    return "\n".join(lines).rstrip() + "\n"


def _computation_block(graph: StateGraph, computation: ComputationNode) -> list[str]:
    lines = [f"[{computation.id}] {computation.description}"]

    requires = graph.requires_of(computation.id)
    if requires:
        lines.append("Needs:")
        for information_id in requires:
            lines.append(f"- {_information_line(graph.node(information_id))}")

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
            lines.append(f"- {_information_line(graph.node(information_id))}")

    predecessors = graph.predecessors_of(computation.id)
    if predecessors:
        lines.append("Depends on: " + ", ".join(predecessors))

    lines.append("")
    return lines


def _information_line(node: InformationNode) -> str:
    text = f"[{node.id}] {node.description}"
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
        return f"{payload.value}"
    if isinstance(payload, ListPayload):
        return ", ".join(str(v) for v in payload.values)
    if isinstance(payload, MappingPayload):
        return ", ".join(f"{k}={v}" for k, v in payload.values)
    return ""


def _argument(value) -> str:
    if isinstance(value, InformationReference):
        return f"@{value.information_id}"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)
