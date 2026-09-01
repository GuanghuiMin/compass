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
        for endpoint in (source, target):
            if not isinstance(endpoint, str):
                raise SchemaError(f"an edge endpoint is text, got {type(endpoint).__name__}")
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
        # _safe_order, because an edge may name an id of any shape and sorting must not be the thing
        # that raises: a dangling endpoint has to reach validate() as a violation.
        return tuple(sorted((Edge(u, d["relation"], v) for u, v, d in self._g.edges(data=True)),
                            key=lambda e: (e.relation.value, _safe_order(e.source),
                                           _safe_order(e.target))))

    def edges_of(self, relation: Relation) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.relation is relation)

    def requires_of(self, computation_id: str) -> tuple[str, ...]:
        """Information nodes this computation requires."""
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.REQUIRES), key=_safe_order))

    def produces_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRODUCES), key=_safe_order))

    def consumers_of(self, information_id: str) -> tuple[str, ...]:
        """Computations that require this information. Emptiness here is what makes it dead."""
        return tuple(sorted((v for _, v, d in self._g.out_edges(information_id, data=True)
                             if d["relation"] is Relation.REQUIRES), key=_safe_order))

    def producers_of(self, information_id: str) -> tuple[str, ...]:
        return tuple(sorted((u for u, _, d in self._g.in_edges(information_id, data=True)
                             if d["relation"] is Relation.PRODUCES), key=_safe_order))

    def predecessors_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRECEDES), key=_safe_order))

    def successors_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.PRECEDES), key=_safe_order))

    # ------------------------------------------------------------------ refinement
    # Every accessor below is read off the edges on demand. None of them is cached, and none of
    # them assumes the candidate is sound: validation runs on graphs that may have a refinement
    # cycle or a child with two parents, so a traversal that trusted the hierarchy would hang on
    # exactly the input it exists to report.

    def refinement_children_of(self, computation_id: str) -> tuple[str, ...]:
        """The computations this one is refined into."""
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.REFINES), key=_safe_order))

    def refinement_parents_of(self, computation_id: str) -> tuple[str, ...]:
        """Plural, because an invalid candidate may give a computation two parents.

        A singular accessor would have to pick one of them, and picking is how a graph that should
        have been refused becomes a graph that was silently reinterpreted.
        """
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.REFINES), key=_safe_order))

    def is_coarse(self, computation_id: str) -> bool:
        """Refined into children, so the work is theirs and not its own."""
        return bool(self.refinement_children_of(computation_id))

    def is_leaf(self, computation_id: str) -> bool:
        return not self.refinement_children_of(computation_id)

    def refinement_ancestors_of(self, computation_id: str) -> tuple[str, ...]:
        """Every computation above this one, cycles included and visited once each."""
        return self._reach(computation_id, self.refinement_parents_of)

    def refinement_descendants_of(self, computation_id: str) -> tuple[str, ...]:
        return self._reach(computation_id, self.refinement_children_of)

    def descendant_leaves_of(self, computation_id: str) -> tuple[str, ...]:
        """The leaves below this one. Empty when a cycle leaves the subtree without any."""
        return tuple(sorted((n for n in self.refinement_descendants_of(computation_id)
                             if self.is_leaf(n)), key=_safe_order))

    def _reach(self, start: str, step) -> tuple[str, ...]:
        seen: set[str] = set()
        pending = list(step(start))
        while pending:
            node_id = pending.pop()
            if node_id in seen or node_id == start:
                continue
            seen.add(node_id)
            pending.extend(step(node_id))
        return tuple(sorted(seen, key=_safe_order))

    def interface_inputs_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((u for u, _, d in self._g.in_edges(computation_id, data=True)
                             if d["relation"] is Relation.INTERFACE_INPUT), key=_safe_order))

    def interface_outputs_of(self, computation_id: str) -> tuple[str, ...]:
        return tuple(sorted((v for _, v, d in self._g.out_edges(computation_id, data=True)
                             if d["relation"] is Relation.INTERFACE_OUTPUT), key=_safe_order))

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
        """Load a snapshot, refusing anything it does not recognise.

        A loader that reads a missing section as empty and `"false"` as true turns a damaged artifact
        into a plausible graph, and a replay built on plausible graphs measures nothing. Structure is
        checked here; whether the graph holds together is still validation's question.
        """
        _exact_keys(snapshot, {"computations", "information", "edges"}, "snapshot")
        graph = cls()
        for raw in _sequence(snapshot["computations"], "snapshot.computations"):
            graph.add(_computation_from_dict(raw))
        for raw in _sequence(snapshot["information"], "snapshot.information"):
            graph.add(_information_from_dict(raw))
        for raw in _sequence(snapshot["edges"], "snapshot.edges"):
            _exact_keys(raw, {"source", "relation", "target"}, "snapshot.edges entry")
            try:
                relation = Relation(raw["relation"])
            except ValueError as err:
                raise SchemaError(f"unknown relation in snapshot: {raw['relation']!r}") from err
            graph.add_edge(_text(raw["source"], "edge source"),
                           relation, _text(raw["target"], "edge target"))
        return graph

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateGraph) and self.to_snapshot() == other.to_snapshot()

    def __iter__(self) -> Iterator[str]:
        return iter(self._g.nodes)


def _order(node_id: str) -> tuple[str, int, str]:
    """Sort c2 before c10, keep computations and information apart, and never leave a tie.

    The raw id breaks ties because c1 and c01 are both legal and both read as one; without it their
    order would come from whichever was inserted first, and a serialization that depends on
    insertion order is not canonical.
    """
    return node_id[0], int(node_id[1:]), node_id


def _safe_order(node_id: str) -> tuple[str, int, str]:
    """Ordering for a string of any shape, since an edge may name one that no node declares and
    sorting must never be what raises instead of reporting it."""
    try:
        return _order(node_id)
    except (ValueError, IndexError):
        return node_id, -1, node_id


def build(nodes: Iterable[Node] = (), edges: Iterable[tuple[str, Relation, str]] = ()) -> StateGraph:
    graph = StateGraph()
    for node in nodes:
        graph.add(node)
    for source, relation, target in edges:
        graph.add_edge(source, relation, target)
    return graph


# --------------------------------------------------------------------------- reading artifacts

def _exact_keys(raw: object, expected: set[str], where: str) -> None:
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: expected an object, got {type(raw).__name__}")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise SchemaError(f"{where}: missing {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{where}: unknown {', '.join(unknown)}")


def _sequence(raw: object, where: str) -> list:
    if not isinstance(raw, list):
        raise SchemaError(f"{where}: expected a list, got {type(raw).__name__}")
    return raw


def _text(raw: object, where: str) -> str:
    if not isinstance(raw, str):
        raise SchemaError(f"{where}: expected text, got {type(raw).__name__}")
    return raw


def _boolean(raw: object, where: str) -> bool:
    """No coercion. `"false"` is a string, and reading it as True is how an artifact starts lying."""
    if not isinstance(raw, bool):
        raise SchemaError(f"{where}: expected true or false, got {raw!r}")
    return raw


# --------------------------------------------------------------------------- serialization

def _argument_to_dict(value) -> dict:
    if isinstance(value, InformationReference):
        return {"reference": value.information_id}
    return {"literal": value}


def _argument_from_dict(raw: dict, where: str):
    if not isinstance(raw, dict) or set(raw) not in ({"reference"}, {"literal"}):
        raise SchemaError(f"{where}: an argument is exactly one of "
                          "{'literal': ...} or {'reference': ...}")
    if "reference" in raw:
        return InformationReference(_text(raw["reference"], f"{where} reference"))
    return raw["literal"]


def _computation_to_dict(node: ComputationNode) -> dict:
    return {
        "id": node.id,
        "description": node.description,
        "operation": node.operation,
        "arguments": {k: _argument_to_dict(v) for k, v in sorted(node.arguments.items())},
    }


def _computation_from_dict(raw: dict) -> ComputationNode:
    _exact_keys(raw, {"id", "description", "operation", "arguments"}, "a computation")
    where = f"computation {raw['id']!r}"
    if not isinstance(raw["arguments"], dict):
        raise SchemaError(f"{where}: arguments must be an object")
    operation = raw["operation"]
    if operation is not None:
        operation = _text(operation, f"{where} operation")
    return ComputationNode(
        id=_text(raw["id"], "a computation id"),
        description=_text(raw["description"], f"{where} description"),
        operation=operation,
        arguments={k: _argument_from_dict(v, f"{where} argument {k!r}")
                   for k, v in raw["arguments"].items()},
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


_PAYLOAD_KEYS = {
    "scalar": {"type", "value"},
    "list": {"type", "values"},
    "mapping": {"type", "values"},
    "runtime_reference": {"type", "name"},
    "contract": {"type", "operation", "parameters", "constraints"},
}


def _payload_from_dict(raw: dict | None):
    if raw is None:
        return None
    if not isinstance(raw, dict) or "type" not in raw:
        raise SchemaError("a payload is an object with a type")
    tag = raw["type"]
    if not isinstance(tag, str):
        # `tag not in _PAYLOAD_KEYS` would raise TypeError on an unhashable one, and this module
        # promises SchemaError for everything a damaged artifact can hold.
        raise SchemaError(f"a payload type is text, got {type(tag).__name__}")
    if tag not in _PAYLOAD_KEYS:
        raise SchemaError(f"unknown payload type {tag!r}")
    _exact_keys(raw, _PAYLOAD_KEYS[tag], f"a {tag} payload")
    if tag == "scalar":
        return ScalarPayload(raw["value"])
    if tag == "list":
        return ListPayload(tuple(_sequence(raw["values"], "a list payload")))
    if tag == "mapping":
        pairs = []
        for pair in _sequence(raw["values"], "a mapping payload"):
            if not isinstance(pair, list) or len(pair) != 2:
                raise SchemaError(f"a mapping payload entry is a pair, got {pair!r}")
            pairs.append((pair[0], pair[1]))
        return MappingPayload(tuple(pairs))
    if tag == "runtime_reference":
        return RuntimeReferencePayload(_text(raw["name"], "a runtime reference name"))
    return ContractPayload(_text(raw["operation"], "a contract operation"),
                           tuple(_sequence(raw["parameters"], "contract parameters")),
                           tuple(_sequence(raw["constraints"], "contract constraints")))


def _information_to_dict(node: InformationNode) -> dict:
    return {"id": node.id, "kind": node.kind.value, "description": node.description,
            "available": node.available, "payload": _payload_to_dict(node.payload)}


def _information_from_dict(raw: dict) -> InformationNode:
    _exact_keys(raw, {"id", "kind", "description", "available", "payload"}, "an information node")
    where = f"information {raw['id']!r}"
    try:
        kind = InformationKind(raw["kind"])
    except ValueError as err:
        raise SchemaError(f"unknown information kind in snapshot: {raw['kind']!r}") from err
    return InformationNode(id=_text(raw["id"], "an information id"), kind=kind,
                           description=_text(raw["description"], f"{where} description"),
                           available=_boolean(raw["available"], f"{where} available"),
                           payload=_payload_from_dict(raw["payload"]))
