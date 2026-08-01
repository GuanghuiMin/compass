"""A compaction state holding the remaining computation and the information it needs."""

from .schema import (
    ArgumentValue, ComputationNode, ContractPayload, EntityType, InformationKind, InformationNode,
    InformationPayload, InformationReference, ListPayload, MappingPayload, Relation,
    RuntimeReferencePayload, ScalarPayload, SchemaError, is_serialized_container,
)
from .parser import ParseError, ParseOutcome, parse
from .protocol import GRAMMAR, to_protocol
from .state_graph import Edge, StateGraph, build

__all__ = [
    "ArgumentValue", "ComputationNode", "ContractPayload", "Edge", "EntityType", "GRAMMAR",
    "InformationKind", "InformationNode", "InformationPayload", "InformationReference",
    "ListPayload", "MappingPayload", "ParseError", "ParseOutcome", "Relation",
    "RuntimeReferencePayload", "ScalarPayload", "SchemaError", "StateGraph", "build",
    "is_serialized_container", "parse", "to_protocol",
]
