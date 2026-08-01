"""The two node types, their payloads, and the three relations.

Everything here is frozen and validated at construction, so a node that exists is a node that is
structurally sound. Meaning is not checked here: whether a contract was really established, or whether
a description honestly names one computation, belongs to audit, not to a constructor.

Payloads are deliberately flat. The previous implementation had one free-text list per node, and
results ended up serialized into it -- an access token as `"{'key': 'access_token', 'value': 'eyJ...'}"`
-- which is how a graph of future work turns into a store of past output. A payload that cannot nest
cannot absorb a result, so the pressure goes where it belongs: into a separate information node with
its own consumers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Union

Scalar = Union[str, int, float, bool, None]


class SchemaError(ValueError):
    """A node or payload that cannot be built, as opposed to a graph that does not hold together."""


class InformationKind(str, Enum):
    FACT = "fact"
    CONSTRAINT = "constraint"
    RESULT = "result"
    CONTRACT = "contract"
    RUNTIME_REFERENCE = "runtime_reference"
    FAILURE_CONSEQUENCE = "failure_consequence"


class Relation(str, Enum):
    PRECEDES = "precedes"     # computation -> computation
    REQUIRES = "requires"     # information -> computation
    PRODUCES = "produces"     # computation -> information


class EntityType(str, Enum):
    COMPUTATION = "computation"
    INFORMATION = "information"


# --------------------------------------------------------------------------- serialized containers

def is_serialized_container(value: object) -> bool:
    """Is this text a dict, list or tuple that has been stringified?

    Tight on purpose: the text has to both look like a container and parse as one, so ordinary prose
    that happens to contain a brace is left alone while `"{'key': 'access_token', 'value': ...}"` is
    caught. Nothing here guesses what the container meant.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not (text.startswith(("{", "[", "(")) and text.endswith(("}", "]", ")"))):
        return False
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False
    return isinstance(parsed, (dict, list, tuple, set))


def _check_scalar(value: object, where: str) -> Scalar:
    if isinstance(value, (dict, list, tuple, set)):
        raise SchemaError(f"{where}: nested {type(value).__name__} is not a payload value; "
                          "give the parts their own information nodes")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise SchemaError(f"{where}: {type(value).__name__} is not a scalar")
    if is_serialized_container(value):
        raise SchemaError(f"{where}: a stringified container is not a value; "
                          "an established result belongs in its own information node")
    return value


def _check_text(text: object, where: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise SchemaError(f"{where}: must be non-empty text")
    if is_serialized_container(text):
        raise SchemaError(f"{where}: a stringified container is not a description")
    return text.strip()


def _check_id(value: object, prefix: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{where}: id must be non-empty text")
    ident = value.strip()
    if not ident.startswith(prefix) or not ident[len(prefix):].isdigit():
        raise SchemaError(f"{where}: id {ident!r} must be {prefix}<number>, local to this snapshot")
    return ident


# --------------------------------------------------------------------------- payloads

@dataclass(frozen=True)
class ScalarPayload:
    value: Scalar

    def __post_init__(self) -> None:
        _check_scalar(self.value, "ScalarPayload.value")


@dataclass(frozen=True)
class ListPayload:
    values: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise SchemaError("ListPayload.values: must be a tuple")
        for i, v in enumerate(self.values):
            _check_scalar(v, f"ListPayload.values[{i}]")


@dataclass(frozen=True)
class MappingPayload:
    values: tuple[tuple[str, Scalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise SchemaError("MappingPayload.values: must be a tuple of pairs")
        seen = set()
        for i, pair in enumerate(self.values):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise SchemaError(f"MappingPayload.values[{i}]: must be a (key, value) pair")
            key, value = pair
            if not isinstance(key, str) or not key.strip():
                raise SchemaError(f"MappingPayload.values[{i}]: key must be non-empty text")
            if key in seen:
                raise SchemaError(f"MappingPayload: duplicate key {key!r}")
            seen.add(key)
            _check_scalar(value, f"MappingPayload[{key}]")


@dataclass(frozen=True)
class RuntimeReferencePayload:
    """The name the agent bound. Never the value behind it, which is never read."""
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("RuntimeReferencePayload.name: must be non-empty text")
        if self.name != self.name.strip():
            object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True)
class ContractPayload:
    operation: str
    parameters: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise SchemaError("ContractPayload.operation: must be non-empty text")
        for name, seq in (("parameters", self.parameters), ("constraints", self.constraints)):
            if not isinstance(seq, tuple):
                raise SchemaError(f"ContractPayload.{name}: must be a tuple")
            for i, item in enumerate(seq):
                if not isinstance(item, str) or not item.strip():
                    raise SchemaError(f"ContractPayload.{name}[{i}]: must be non-empty text")


InformationPayload = Union[ScalarPayload, ListPayload, MappingPayload,
                           RuntimeReferencePayload, ContractPayload]

_PAYLOAD_TYPES = (ScalarPayload, ListPayload, MappingPayload,
                  RuntimeReferencePayload, ContractPayload)


# --------------------------------------------------------------------------- nodes

@dataclass(frozen=True)
class InformationReference:
    """An argument that names an information node instead of repeating its value."""
    information_id: str

    def __post_init__(self) -> None:
        _check_id(self.information_id, "i", "InformationReference.information_id")


ArgumentValue = Union[Scalar, InformationReference]


@dataclass(frozen=True)
class ComputationNode:
    id: str
    description: str
    operation: str | None = None
    arguments: Mapping[str, ArgumentValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _check_id(self.id, "c", "ComputationNode.id"))
        object.__setattr__(self, "description",
                           _check_text(self.description, f"{self.id}.description"))
        if self.operation is not None:
            if not isinstance(self.operation, str) or not self.operation.strip():
                raise SchemaError(f"{self.id}.operation: must be non-empty text or absent")
            object.__setattr__(self, "operation", self.operation.strip())
        if not isinstance(self.arguments, Mapping):
            raise SchemaError(f"{self.id}.arguments: must be a mapping")
        checked: dict[str, ArgumentValue] = {}
        for key, value in self.arguments.items():
            if not isinstance(key, str) or not key.strip():
                raise SchemaError(f"{self.id}.arguments: argument name must be non-empty text")
            if isinstance(value, InformationReference):
                checked[key] = value
            else:
                checked[key] = _check_scalar(value, f"{self.id}.arguments[{key}]")
        object.__setattr__(self, "arguments", MappingProxyType(dict(checked)))

    @property
    def referenced_information(self) -> tuple[str, ...]:
        return tuple(sorted(v.information_id for v in self.arguments.values()
                            if isinstance(v, InformationReference)))


@dataclass(frozen=True)
class InformationNode:
    id: str
    kind: InformationKind
    description: str
    available: bool
    payload: InformationPayload | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _check_id(self.id, "i", "InformationNode.id"))
        if not isinstance(self.kind, InformationKind):
            raise SchemaError(f"{self.id}.kind: {self.kind!r} is not a known information kind")
        object.__setattr__(self, "description",
                           _check_text(self.description, f"{self.id}.description"))
        if not isinstance(self.available, bool):
            raise SchemaError(f"{self.id}.available: must be true or false")
        if self.payload is not None and not isinstance(self.payload, _PAYLOAD_TYPES):
            raise SchemaError(f"{self.id}.payload: {type(self.payload).__name__} is not a payload")
        if self.kind is InformationKind.RUNTIME_REFERENCE and self.payload is not None \
                and not isinstance(self.payload, RuntimeReferencePayload):
            raise SchemaError(f"{self.id}: a runtime reference carries a name, "
                              f"not a {type(self.payload).__name__}")
        if self.kind is InformationKind.CONTRACT and self.payload is not None \
                and not isinstance(self.payload, ContractPayload):
            raise SchemaError(f"{self.id}: a contract carries an interface, "
                              f"not a {type(self.payload).__name__}")


Node = Union[ComputationNode, InformationNode]

ENDPOINTS: dict[Relation, tuple[EntityType, EntityType]] = {
    Relation.PRECEDES: (EntityType.COMPUTATION, EntityType.COMPUTATION),
    Relation.REQUIRES: (EntityType.INFORMATION, EntityType.COMPUTATION),
    Relation.PRODUCES: (EntityType.COMPUTATION, EntityType.INFORMATION),
}


def entity_type(node: Node) -> EntityType:
    if isinstance(node, ComputationNode):
        return EntityType.COMPUTATION
    if isinstance(node, InformationNode):
        return EntityType.INFORMATION
    raise SchemaError(f"{type(node).__name__} is not a node")
