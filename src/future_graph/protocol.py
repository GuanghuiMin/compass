"""The block form the model writes, and the same form written back out.

One field per line, and no brackets to balance. Whole-graph output in JSON put every node behind one
punctuation error: two of twelve recorded refusals were a missing comma or quote three thousand
characters into a graph that never reached semantic validation. Here a damaged line costs that line.

Nothing about this loosens what a graph has to mean. The parser normalizes surface and refuses to
invent structure; every semantic question is still answered by validation.
"""

from __future__ import annotations

from .schema import (
    ComputationNode, ContractPayload, InformationKind, InformationNode, InformationReference,
    ListPayload, MappingPayload, Relation, RuntimeReferencePayload, ScalarPayload,
)
from .state_graph import StateGraph

BEGIN_GRAPH = "BEGIN_GRAPH"
END_GRAPH = "END_GRAPH"
BEGIN_INFO = "INFO"
END_INFO = "END_INFO"
BEGIN_COMPUTATION = "COMPUTATION"
END_COMPUTATION = "END_COMPUTATION"
EDGE = "EDGE"

INFORMATION_FIELDS = frozenset({
    "kind", "available", "description",
    "payload-type", "value", "item", "entry",
    "runtime-name",
    "contract-operation", "contract-parameter", "contract-constraint",
})

# A list and a mapping say which they are, because a list of nothing and no list at all are
# different states and "the query matched nothing" is worth being able to write down.
DECLARED_PAYLOADS = ("list", "mapping")
COMPUTATION_FIELDS = frozenset({"description", "operation", "argument"})


def to_protocol(graph: StateGraph) -> str:
    """Write a graph in the form the model is asked to produce. Round-trips through the parser.

    Canonical: information, then computations, then edges; nodes by numeric id, arguments by name,
    edges by relation then source then target. Insertion order cannot show through.
    """
    lines = [BEGIN_GRAPH, ""]
    for node in graph.information:
        lines += _information_block(node)
    for node in graph.computations:
        lines += _computation_block(node)
    for edge in graph.edges:
        lines.append(f"{EDGE} {edge.source} {edge.relation.name} {edge.target}")
    if graph.edges:
        lines.append("")
    lines.append(END_GRAPH)
    return "\n".join(lines) + "\n"


def _information_block(node: InformationNode) -> list[str]:
    lines = [f"{BEGIN_INFO} {node.id}",
             f"kind: {node.kind.value}",
             f"available: {'true' if node.available else 'false'}",
             f"description: {node.description}"]
    payload = node.payload
    if isinstance(payload, ScalarPayload):
        lines.append(f"value: {format_scalar(payload.value)}")
    elif isinstance(payload, ListPayload):
        lines.append("payload-type: list")
        lines += [f"item: {format_scalar(v)}" for v in payload.values]
    elif isinstance(payload, MappingPayload):
        lines.append("payload-type: mapping")
        lines += [f"entry {k} = {format_scalar(v)}" for k, v in payload.values]
    elif isinstance(payload, RuntimeReferencePayload):
        lines.append(f"runtime-name: {payload.name}")
    elif isinstance(payload, ContractPayload):
        lines.append(f"contract-operation: {payload.operation}")
        lines += [f"contract-parameter: {p}" for p in payload.parameters]
        lines += [f"contract-constraint: {c}" for c in payload.constraints]
    lines += [END_INFO, ""]
    return lines


def _computation_block(node: ComputationNode) -> list[str]:
    lines = [f"{BEGIN_COMPUTATION} {node.id}", f"description: {node.description}"]
    if node.operation:
        lines.append(f"operation: {node.operation}")
    for name, value in sorted(node.arguments.items()):
        lines.append(f"argument {name} = {_argument_out(value)}")
    lines += [END_COMPUTATION, ""]
    return lines


def format_scalar(value) -> str:
    """Write a scalar so that reading it back gives the same value, and the same type.

    Every string is quoted, without exception. Quoting only the awkward-looking ones means `"true"`
    comes back as a boolean, `"7"` as an integer and `"@i3"` as a reference to a node, and a serializer
    that changes types is worse than one that is verbose.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _argument_out(value) -> str:
    if isinstance(value, InformationReference):
        return f"@{value.information_id}"
    return format_scalar(value)


GRAMMAR = f"""\
Structural words and field names are case-insensitive. Ids, argument names, operations, runtime names,
descriptions and payload text keep the case they were written in.

Every field below is given at most once, except `item`, `entry`, `contract-parameter`,
`contract-constraint` and `argument`, which may repeat. A repeated `entry` key or `argument` name is an
error rather than the later value winning.

{BEGIN_GRAPH}

{BEGIN_INFO} <i-id>
kind: {' | '.join(k.value for k in InformationKind)}
available: true | false
description: <text>
  and at most one payload, written as one of:
    value: <scalar>
    payload-type: list
    item: <scalar>                 (repeat, zero or more)
    payload-type: mapping
    entry <key> = <scalar>         (repeat, zero or more)
    runtime-name: <name the agent bound>
    contract-operation: <operation>
    contract-parameter: <name>     (repeat)
    contract-constraint: <text>    (repeat)

  A list or a mapping says so with payload-type, so that one holding nothing is still one.
  `item` without `payload-type: list`, or `entry` without `payload-type: mapping`, is refused
  rather than guessed at.

  contract and runtime_reference are available-only, and nothing unavailable carries a payload.
{END_INFO}

{BEGIN_COMPUTATION} <c-id>
description: <text>
operation: <function>              (optional)
argument <name> = <scalar or @i-id>   (repeat, optional)
{END_COMPUTATION}

{EDGE} <source> {' | '.join(r.name for r in Relation)} <target>

{END_GRAPH}
"""
