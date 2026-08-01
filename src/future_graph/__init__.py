"""A compaction state holding the remaining computation and the information it needs."""

from .schema import (
    ArgumentValue, ComputationNode, ContractPayload, EntityType, InformationKind, InformationNode,
    InformationPayload, InformationReference, ListPayload, MappingPayload, Relation,
    RuntimeReferencePayload, ScalarPayload, SchemaError, is_serialized_container,
)
from .artifacts import ArtifactError, ConfigScalar, ModelCall, RegenerationRecord
from .parser import ParseError, ParseOutcome, parse
from .protocol import GRAMMAR, format_scalar, to_protocol
from .regeneration import Model, PromptError, RegenerationResult, regenerate_graph
from .state_graph import Edge, StateGraph, build

# Assembly and serialization helpers stay in their own modules. What a caller needs is a way to
# regenerate a graph and a way to read the record of having done so.
__all__ = [
    "ArgumentValue", "ArtifactError", "ComputationNode", "ConfigScalar", "ContractPayload", "Edge",
    "EntityType", "GRAMMAR", "InformationKind", "InformationNode", "InformationPayload",
    "InformationReference", "ListPayload", "MappingPayload", "Model", "ModelCall", "ParseError",
    "ParseOutcome", "PromptError", "RegenerationRecord", "RegenerationResult", "Relation",
    "RuntimeReferencePayload", "ScalarPayload", "SchemaError", "StateGraph", "build",
    "format_scalar", "is_serialized_container", "parse", "regenerate_graph", "to_protocol",
]
