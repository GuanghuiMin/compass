"""The graph, and the only place state lives.

There is no dictionary of contracts beside this, no inventory of runtime references, no record of what
has been done. Anything that looks like an index is derived on demand from the graph and a test rebuilds
it to prove the graph is sufficient.

Ids are local to the snapshot. Nothing here compares an id with one from a previous boundary, and the
serialization carries no boundary-crossing identity for anything to hang on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import networkx as nx

from .schema import (
    ComputationNode, EntityType, InformationKind, InformationNode, InformationReference,
    ContractPayload, ListPayload, MappingPayload, Node, Relation, RuntimeReferencePayload,
    ScalarPayload, SchemaError, entity_type,
)


@dataclass(frozen=True)
class Edge:
    source: str
    relation: Relation
    target: str


class StateGraph:
    """A typed directed graph of remaining computations and the information they need."""

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # ------------------------------------------------------------------ building
    def add(self, node: Node) -> None:
        if node.id in self._g:
            raise SchemaError(f"duplicate id {node.id!r} in one snapshot")
        self._g.add_node(node.id, entity_type=entity_type(node), payload=node)

    def add_edge(self, source: str, relation: Relation, target: str) -> None:
        """An edge to an id that does not exist is kept, not refused.

        networkx would silently invent the endpoint. Inventing it and reporting it as dangling is what
        lets validation collect every fault in one pass instead of dying on the first one.
        """
        if not isinstance(relation, Relation):
            raise SchemaError(f"{relation!r} is not a known relation")
        self._g.add_edge(source, target, key=relation.value, relation=relation)

    # ------------------------------------------------------------------ reading
    def __contains__(self, node_id: object) -> bool:
        return node_id in self._g

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def node(self, node_id: str) -> Node:
        if node_id not in self._g or "payload" not in self._g.nodes[node_id]:
            raise KeyError(node_id)
        return self._g.nodes[node_id]["payload"]

    def kind_of(self, node_id: str) -> EntityType | None:
        """The type of a declared node, or None for an id only an edge mentions."""
        return self._g.nodes[node_id].get("entity_type") if node_id in self._g else None

    @property
    def computations(self) -> tuple[ComputationNode, ...]:
        return tuple(sorted((n["payload"] for _, n in self._g.nodes(data=True)
                             if n.get("entity_type") is EntityType.COMPUTATION),
                            key=lambda c: _order(c.id)))

    @property
    def information(self) -> tuple[InformationNode, ...]:
        return tuple(sorted((n["payload"] for _, n in self._g.nodes(data=True)
                             if n.get("entity_type") is EntityType.INFORMATION),
                            key=lambda i: _order(i.id)))

    @property
    def dangling_ids(self) -> tuple[str, ...]:
        """Ids an edge points at that no node declares."""
        return tuple(sorted((nid for nid, data in self._g.nodes(data=True)
                             if "payload" not in data), key=_safe_order))

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(sorted((Edge(u, d["relation"], v) for u, v, d in self._g.edges(data=True)),
                            key=lambda e: (e.relation.value, _order(e.source), _order(e.target))))

    def edges_of(self, relation: Relation) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.relation is relation)

    def requires_of(self, computation_id: str) -> tuple[str, ...]:
        """Information nodes this computation requires."""
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.REQUIRES), key=_order))

    def produces_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRODUCES), key=_order))

    def consumers_of(self, information_id: str) -> tuple[str, ...]:
        """Computations that require this information. Emptiness here is what makes it dead."""
        return tuple(sorted((v for _, v, d in self._g.out_edges(information_id, data=True)
                             if d["relation"] is Relation.REQUIRES), key=_order))

    def producers_of(self, information_id: str) -> tuple[str, ...]:
        return tuple(sorted((u for u, _, d in self._g.in_edges(information_id, data=True)
                             if d["relation"] is Relation.PRODUCES), key=_order))

    def predecessors_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRECEDES), key=_order))

    def successors_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRECEDES), key=_order))

    def remove(self, node_id: str) -> None:
        self._g.remove_node(node_id)

    def as_digraph(self) -> nx.DiGraph:
        """A plain view for graph algorithms that do not care which relation an edge carries."""
        plain = nx.DiGraph()
        plain.add_nodes_from(self._g.nodes)
        plain.add_edges_from((u, v) for u, v, _ in self._g.edges(data=True))
        return plain

    # ------------------------------------------------------------------ artifacts
    def to_snapshot(self) -> dict:
        """A stable, ordered dict. Two equal graphs serialize identically, byte for byte."""
        return {
            "computations": [_computation_to_dict(c) for c in self.computations],
            "information": [_information_to_dict(i) for i in self.information],
            "edges": [{"source": e.source, "relation": e.relation.value, "target": e.target}
                      for e in self.edges],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "StateGraph":
        graph = cls()
        for raw in snapshot.get("computations") or []:
            graph.add(_computation_from_dict(raw))
        for raw in snapshot.get("information") or []:
            graph.add(_information_from_dict(raw))
        for raw in snapshot.get("edges") or []:
            try:
                relation = Relation(raw["relation"])
            except (KeyError, ValueError) as err:
                raise SchemaError(f"unknown relation in snapshot: {raw!r}") from err
            graph.add_edge(raw["source"], relation, raw["target"])
        return graph

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateGraph) and self.to_snapshot() == other.to_snapshot()

    def __iter__(self) -> Iterator[str]:
        return iter(self._g.nodes)


def _order(node_id: str) -> tuple[str, int]:
    """Sort c2 before c10, and keep computations and information apart."""
    return node_id[0], int(node_id[1:])


def _safe_order(node_id: str) -> tuple[str, int]:
    """Ordering for ids that may be malformed, since a dangling id can be anything."""
    try:
        return _order(node_id)
    except (ValueError, IndexError):
        return node_id, -1


def build(nodes: Iterable[Node] = (), edges: Iterable[tuple[str, Relation, str]] = ()) -> StateGraph:
    graph = StateGraph()
    for node in nodes:
        graph.add(node)
    for source, relation, target in edges:
        graph.add_edge(source, relation, target)
    return graph


# --------------------------------------------------------------------------- serialization

def _argument_to_dict(value) -> dict:
    if isinstance(value, InformationReference):
        return {"reference": value.information_id}
    return {"literal": value}


def _argument_from_dict(raw: dict):
    if "reference" in raw:
        return InformationReference(raw["reference"])
    return raw.get("literal")


def _computation_to_dict(node: ComputationNode) -> dict:
    return {
        "id": node.id,
        "description": node.description,
        "operation": node.operation,
        "arguments": {k: _argument_to_dict(v) for k, v in sorted(node.arguments.items())},
    }


def _computation_from_dict(raw: dict) -> ComputationNode:
    return ComputationNode(
        id=raw["id"], description=raw["description"], operation=raw.get("operation"),
        arguments={k: _argument_from_dict(v) for k, v in (raw.get("arguments") or {}).items()},
    )


_PAYLOAD_TAGS = {
    ScalarPayload: "scalar", ListPayload: "list", MappingPayload: "mapping",
    RuntimeReferencePayload: "runtime_reference", ContractPayload: "contract",
}


def _payload_to_dict(payload) -> dict | None:
    if payload is None:
        return None
    tag = _PAYLOAD_TAGS[type(payload)]
    if isinstance(payload, ScalarPayload):
        body = {"value": payload.value}
    elif isinstance(payload, ListPayload):
        body = {"values": list(payload.values)}
    elif isinstance(payload, MappingPayload):
        body = {"values": [list(pair) for pair in payload.values]}
    elif isinstance(payload, RuntimeReferencePayload):
        body = {"name": payload.name}
    else:
        body = {"operation": payload.operation, "parameters": list(payload.parameters),
                "constraints": list(payload.constraints)}
    return {"type": tag, **body}


def _payload_from_dict(raw: dict | None):
    if raw is None:
        return None
    tag = raw.get("type")
    if tag == "scalar":
        return ScalarPayload(raw["value"])
    if tag == "list":
        return ListPayload(tuple(raw["values"]))
    if tag == "mapping":
        return MappingPayload(tuple((k, v) for k, v in raw["values"]))
    if tag == "runtime_reference":
        return RuntimeReferencePayload(raw["name"])
    if tag == "contract":
        return ContractPayload(raw["operation"], tuple(raw.get("parameters") or ()),
                               tuple(raw.get("constraints") or ()))
    raise SchemaError(f"unknown payload type {tag!r}")


def _information_to_dict(node: InformationNode) -> dict:
    return {"id": node.id, "kind": node.kind.value, "description": node.description,
            "available": node.available, "payload": _payload_to_dict(node.payload)}


def _information_from_dict(raw: dict) -> InformationNode:
    try:
        kind = InformationKind(raw["kind"])
    except (KeyError, ValueError) as err:
        raise SchemaError(f"unknown information kind in snapshot: {raw.get('kind')!r}") from err
    return InformationNode(id=raw["id"], kind=kind, description=raw["description"],
                           available=bool(raw["available"]),
                           payload=_payload_from_dict(raw.get("payload")))
