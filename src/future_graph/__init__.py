"""A compaction state holding the remaining computation and the information it needs."""

from .schema import (
    ArgumentValue, ComputationNode, ContractPayload, EntityType, InformationKind, InformationNode,
    InformationPayload, InformationReference, ListPayload, MappingPayload, Relation,
    RuntimeReferencePayload, ScalarPayload, SchemaError, is_serialized_container,
)
from .state_graph import Edge, StateGraph, build

__all__ = [
    "ArgumentValue", "ComputationNode", "ContractPayload", "Edge", "EntityType", "InformationKind",
    "InformationNode", "InformationPayload", "InformationReference", "ListPayload", "MappingPayload",
    "Relation", "RuntimeReferencePayload", "ScalarPayload", "SchemaError", "StateGraph", "build",
    "is_serialized_container",
]
