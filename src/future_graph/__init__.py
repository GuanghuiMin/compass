"""A compaction state holding the remaining computation and the information it needs."""

from .schema import (
    ArgumentValue, ComputationNode, ContractPayload, EntityType, InformationKind, InformationNode,
    InformationPayload, InformationReference, ListPayload, MappingPayload, Relation,
    RuntimeReferencePayload, ScalarPayload, SchemaError, is_serialized_container,
)
from .artifacts import (
    ArtifactError, ConfigScalar, ModelCall, RegenerationRecord, RevisionRecord,
)
from .parser import ParseError, ParseOutcome, parse
from .protocol import GRAMMAR, format_scalar, to_protocol
from .regeneration import Model, PromptError, RegenerationResult, regenerate_graph
from .revision import Revision, RevisionError, apply_revision
from .revision_parser import RevisionOutcome, parse_revision
from .state_graph import Edge, StateGraph, build
from .update import UpdateResult, update_graph

# Assembly and serialization helpers stay in their own modules. What a caller needs is a way to
# update a graph across a boundary and a way to read the record of having done so. Complete-graph
# regeneration stays exported as the baseline it is, not as the way an agent is meant to run.
__all__ = [
    "ArgumentValue", "ArtifactError", "ComputationNode", "ConfigScalar", "ContractPayload", "Edge",
    "EntityType", "GRAMMAR", "InformationKind", "InformationNode", "InformationPayload",
    "InformationReference", "ListPayload", "MappingPayload", "Model", "ModelCall", "ParseError",
    "ParseOutcome", "PromptError", "RegenerationRecord", "RegenerationResult", "Relation",
    "Revision", "RevisionError", "RevisionOutcome", "RevisionRecord", "RuntimeReferencePayload",
    "ScalarPayload", "SchemaError", "StateGraph", "UpdateResult", "apply_revision", "build",
    "format_scalar", "is_serialized_container", "parse", "parse_revision", "regenerate_graph",
    "to_protocol", "update_graph",
]
