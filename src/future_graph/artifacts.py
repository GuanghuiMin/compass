"""What was sent, what came back, and what the boundary did with it.

The record exists so a boundary can be re-examined without being re-run, which means it has to hold
the call as it was actually made rather than a description of it. Configuration a record describes but
an adapter never received would make every measurement unfalsifiable, so the configuration travels
inside `ModelCall` and the record keeps that same object.

Reading a record back is strict for the same reason the snapshot loader is: a field read as a default
turns a damaged artifact into a plausible one, and an analysis built on plausible artifacts measures
nothing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Union

ConfigScalar = Union[str, int, float, bool, None]


class ArtifactError(ValueError):
    """A record that cannot be read back as what it claims to be."""


@dataclass(frozen=True)
class ModelCall:
    """Exactly what the adapter is handed. Nothing here is interpreted."""
    system: str
    user: str
    config: tuple[tuple[str, ConfigScalar], ...] = ()


def freeze_config(config: Mapping[str, ConfigScalar]) -> tuple[tuple[str, ConfigScalar], ...]:
    """Order it, copy it, and refuse anything that cannot be recorded as what it is.

    Raising here rather than coercing keeps the failure at the layer that produced it: a bad decoding
    setting is a caller mistake about the model interface, not a graph that failed to validate, and it
    must not reach the model at all.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"config is a mapping, got {type(config).__name__}")
    frozen: list[tuple[str, ConfigScalar]] = []
    for key, value in config.items():
        if not isinstance(key, str):
            raise TypeError(f"config key is text, got {type(key).__name__}")
        if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
            frozen.append((key, value))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"config {key!r} is {value}, which cannot be recorded faithfully")
            frozen.append((key, value))
        else:
            raise TypeError(f"config {key!r} is {type(value).__name__}, "
                            "which is not a value this can record")
    return tuple(sorted(frozen, key=lambda pair: pair[0]))


def prompt_sha(system: str) -> str:
    """A hash of the system message as sent, not of the file it came from."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegenerationRecord:
    """One boundary, in enough detail to be argued with later."""
    goal: str
    rules: str
    delta_h: str
    previous_snapshot: dict
    model_call: ModelCall
    raw_output: str
    normalizations: tuple[str, ...]
    parse_errors: tuple[tuple[int, str], ...]
    parsed_candidate_snapshot: dict | None
    interface_changes: tuple[tuple[str, str, str, str], ...]
    argument_dependency_changes: tuple[tuple[str, str, str, str], ...]
    ordering_repairs: tuple[tuple[str, str, str, str], ...]
    violations: tuple[tuple[str, str, tuple[str, ...]], ...]
    accepted: bool
    resulting_snapshot: dict
    collected: tuple[str, ...]
    handover: str

    @property
    def prompt_sha(self) -> str:
        return prompt_sha(self.model_call.system)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "rules": self.rules,
            "delta_h": self.delta_h,
            "previous_snapshot": self.previous_snapshot,
            "model_call": {
                "system": self.model_call.system,
                "user": self.model_call.user,
                "config": [[k, v] for k, v in self.model_call.config],
            },
            "prompt_sha": self.prompt_sha,
            "raw_output": self.raw_output,
            "normalizations": list(self.normalizations),
            "parse_errors": [[line, message] for line, message in self.parse_errors],
            "parsed_candidate_snapshot": self.parsed_candidate_snapshot,
            "interface_changes": [list(change) for change in self.interface_changes],
            "argument_dependency_changes": [list(change) for change
                                            in self.argument_dependency_changes],
            "ordering_repairs": [list(change) for change in self.ordering_repairs],
            "violations": [[code, message, list(nodes)]
                           for code, message, nodes in self.violations],
            "accepted": self.accepted,
            "resulting_snapshot": self.resulting_snapshot,
            "collected": list(self.collected),
            "handover": self.handover,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RegenerationRecord":
        _exact_keys(raw, set(_FIELD_TYPES), "a record")
        for name, expected in _FIELD_TYPES.items():
            if expected is not None and not isinstance(raw[name], expected):
                raise ArtifactError(f"record {name}: expected {_name_of(expected)}, "
                                    f"got {type(raw[name]).__name__}")
        if isinstance(raw["accepted"], int) and not isinstance(raw["accepted"], bool):
            raise ArtifactError("record accepted: expected true or false")
        call = _call_from_dict(raw["model_call"])
        if raw["prompt_sha"] != prompt_sha(call.system):
            raise ArtifactError("record prompt_sha does not hash the system message it sits beside")
        candidate = raw["parsed_candidate_snapshot"]
        if candidate is not None and not isinstance(candidate, dict):
            raise ArtifactError("record parsed_candidate_snapshot: expected an object or null")

        # A snapshot that only looks like an object is a damaged record, and finding that out later,
        # when someone happens to rebuild the graph, is finding it out in the middle of an analysis.
        _rebuild(raw["previous_snapshot"], "previous_snapshot")
        resulting = _rebuild(raw["resulting_snapshot"], "resulting_snapshot")
        if candidate is not None:
            _rebuild(candidate, "parsed_candidate_snapshot")

        from .rendering import render
        if raw["handover"] != render(resulting):
            raise ArtifactError("record handover is not the rendering of its resulting graph")

        # Entries first, then the cross-field check: asking whether the outcome is consistent only
        # means something once each field is known to be the shape it claims.
        normalizations = tuple(_strings(raw["normalizations"], "normalizations"))
        parse_errors = tuple(_parse_error(item) for item in raw["parse_errors"])
        interface_changes = tuple(_interface_change(item) for item in raw["interface_changes"])
        argument_changes = tuple(_interface_change(item, "argument_dependency_changes")
                                 for item in raw["argument_dependency_changes"])
        ordering_repairs = tuple(_interface_change(item, "ordering_repairs")
                                 for item in raw["ordering_repairs"])
        violations = tuple(_violation(item) for item in raw["violations"])
        collected = tuple(_strings(raw["collected"], "collected"))
        _check_outcome(raw)

        return cls(
            goal=raw["goal"], rules=raw["rules"], delta_h=raw["delta_h"],
            previous_snapshot=raw["previous_snapshot"], model_call=call,
            raw_output=raw["raw_output"], normalizations=normalizations,
            parse_errors=parse_errors, parsed_candidate_snapshot=candidate,
            interface_changes=interface_changes,
            argument_dependency_changes=argument_changes,
            ordering_repairs=ordering_repairs,
            violations=violations, accepted=raw["accepted"],
            resulting_snapshot=raw["resulting_snapshot"], collected=collected,
            handover=raw["handover"],
        )


_FIELD_TYPES: dict[str, object] = {
    "goal": str, "rules": str, "delta_h": str, "previous_snapshot": dict,
    "model_call": dict, "prompt_sha": str, "raw_output": str, "normalizations": list,
    "parse_errors": list, "parsed_candidate_snapshot": None, "interface_changes": list,
    "argument_dependency_changes": list, "ordering_repairs": list,
    "violations": list,
    "accepted": bool, "resulting_snapshot": dict, "collected": list, "handover": str,
}


def _name_of(expected) -> str:
    return expected.__name__ if isinstance(expected, type) else str(expected)


def _exact_keys(raw: object, expected: set[str], where: str) -> None:
    if not isinstance(raw, dict):
        raise ArtifactError(f"{where}: expected an object, got {type(raw).__name__}")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise ArtifactError(f"{where}: missing {', '.join(missing)}")
    if unknown:
        raise ArtifactError(f"{where}: unknown {', '.join(unknown)}")


def _rebuild(snapshot: dict, where: str):
    from .state_graph import StateGraph
    from .schema import SchemaError
    try:
        return StateGraph.from_snapshot(snapshot)
    except SchemaError as err:
        raise ArtifactError(f"record {where}: {err}") from err


def _check_outcome(raw: dict) -> None:
    """A boundary ends in one of three shapes, and a record must be one of them.

    Accepted: parsed, no errors, no violations, and a result that may differ from the previous graph.
    Parse failure: nothing parsed, errors, no violations, nothing collected, previous graph intact.
    Validation rejection: parsed, no errors, violations, nothing collected, previous graph intact.

    Anything else is a record disagreeing with itself. The refusals matter most: a refusal that
    carries a different graph forward would be a boundary claiming to have changed nothing while
    changing everything.
    """
    accepted = raw["accepted"]
    candidate = raw["parsed_candidate_snapshot"]
    parse_errors, violations, collected = raw["parse_errors"], raw["violations"], raw["collected"]

    if accepted:
        if candidate is None:
            raise ArtifactError("record says accepted with nothing parsed")
        if parse_errors:
            raise ArtifactError("record says accepted and also lists parse errors")
        if violations:
            raise ArtifactError("record says accepted and also lists violations")
        return

    if collected:
        raise ArtifactError("record collected information from a graph it did not accept")
    if raw["resulting_snapshot"] != raw["previous_snapshot"]:
        raise ArtifactError("record refused a graph and did not keep the previous one")
    if candidate is None:
        if not parse_errors:
            raise ArtifactError("record refused with nothing parsed and no parse error")
        if violations:
            raise ArtifactError("record has violations for a candidate that never parsed")
    else:
        if parse_errors:
            raise ArtifactError("record has parse errors and a parsed candidate")
        if not violations:
            raise ArtifactError("record refused a candidate that had no violation")


def _call_from_dict(raw: dict) -> ModelCall:
    _exact_keys(raw, {"system", "user", "config"}, "a model call")
    for name in ("system", "user"):
        if not isinstance(raw[name], str):
            raise ArtifactError(f"model call {name}: expected text")
    if not isinstance(raw["config"], list):
        raise ArtifactError("model call config: expected a list of pairs")
    config: list[tuple[str, ConfigScalar]] = []
    for pair in raw["config"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ArtifactError(f"model call config: {pair!r} is not a pair")
        key, value = pair
        if not isinstance(key, str):
            raise ArtifactError("model call config: a key is text")
        if not (value is None or isinstance(value, (str, int, float, bool))):
            raise ArtifactError(f"model call config {key!r}: {type(value).__name__} "
                                "is not a recordable value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ArtifactError(f"model call config {key!r}: {value} was never recordable")
        config.append((key, value))
    # The same shape freeze_config produces, or this record did not come from a call this makes.
    keys = [key for key, _ in config]
    if len(set(keys)) != len(keys):
        raise ArtifactError("model call config: a key appears twice")
    if keys != sorted(keys):
        raise ArtifactError("model call config: keys are not in the order a call is built with")
    return ModelCall(system=raw["system"], user=raw["user"], config=tuple(config))


def _strings(raw: list, where: str) -> list[str]:
    for item in raw:
        if not isinstance(item, str):
            raise ArtifactError(f"record {where}: expected text, got {type(item).__name__}")
    return raw


def _parse_error(item: object) -> tuple[int, str]:
    if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], int) \
            or isinstance(item[0], bool) or not isinstance(item[1], str):
        raise ArtifactError(f"record parse_errors: {item!r} is not a line and a message")
    return item[0], item[1]


def _interface_change(item: object, where: str = "interface_changes") -> tuple[str, str, str, str]:
    """One edge the code took out of the candidate or put into it, in canonical ids."""
    if not isinstance(item, list) or len(item) != 4 or not all(isinstance(x, str) for x in item):
        raise ArtifactError(f"record {where}: {item!r} is not an action, a source, "
                            "a relation and a target")
    if item[0] not in ("removed", "added"):
        raise ArtifactError(f"record {where}: {item[0]!r} is not removed or added")
    if where == "argument_dependency_changes" and item[0] != "added":
        raise ArtifactError("record argument_dependency_changes: this step removes nothing, "
                            f"and {item[0]!r} says otherwise")
    if where == "ordering_repairs" and item[0] != "removed":
        raise ArtifactError("record ordering_repairs: this step adds nothing, "
                            f"and {item[0]!r} says otherwise")
    return item[0], item[1], item[2], item[3]


def _violation(item: object) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], str) \
            or not isinstance(item[1], str) or not isinstance(item[2], list):
        raise ArtifactError(f"record violations: {item!r} is not a code, a message and its nodes")
    return item[0], item[1], tuple(_strings(item[2], "violations"))


# --------------------------------------------------------------------------- identity spaces

# A local revision touches three graphs, and every one of them numbers its nodes from one. The
# previous `i2` and the assembled `i2` are routinely different information nodes -- in the recurrent
# run's last boundary, one was a playlist id that had been removed and the other was the parsed
# suggestions that had been renumbered into its place. Written as bare strings, those two join
# silently and wrongly, and no amount of care by a reader prevents it. So every reference in a
# record says which graph it belongs to.
PREVIOUS = "previous"          # ids in previous_snapshot, the graph the model was shown
REVISION = "revision"          # +labels the model wrote, local to its answer
ASSEMBLED = "assembled"        # canonical ids after application, and the ids the result keeps
ID_SPACES = (PREVIOUS, REVISION, ASSEMBLED)


@dataclass(frozen=True)
class Ref:
    """One node, and the graph its id belongs to."""
    id: str
    space: str

    def __post_init__(self) -> None:
        if self.space not in ID_SPACES:
            raise ArtifactError(f"{self.space!r} is not one of {', '.join(ID_SPACES)}")

    def __str__(self) -> str:
        return f"{self.space}:{self.id}"

    def to_dict(self) -> dict:
        return {"id": self.id, "space": self.space}

    @classmethod
    def from_dict(cls, raw: object, where: str) -> "Ref":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not a reference")
        _exact_keys(raw, {"id", "space"}, f"a reference in {where}")
        if not isinstance(raw["id"], str) or not isinstance(raw["space"], str):
            raise ArtifactError(f"record {where}: a reference is an id and a space")
        if raw["space"] not in ID_SPACES:
            raise ArtifactError(f"record {where}: {raw['space']!r} is not one of "
                                f"{', '.join(ID_SPACES)}")
        return cls(raw["id"], raw["space"])


def previous_ref(node_id: str) -> Ref:
    return Ref(node_id, PREVIOUS)


def revision_ref(label: str) -> Ref:
    return Ref(label, REVISION)


def assembled_ref(node_id: str) -> Ref:
    return Ref(node_id, ASSEMBLED)


def authored_ref(name: str) -> Ref:
    """A name as the model wrote it: its own label, or an anchor into the graph it was shown."""
    return revision_ref(name) if name.startswith("+") else previous_ref(name)


@dataclass(frozen=True)
class Removal:
    node: Ref
    reason: str

    def to_dict(self) -> dict:
        return {"node": self.node.to_dict(), "reason": self.reason}

    @classmethod
    def from_dict(cls, raw: object, where: str = "removed_nodes") -> "Removal":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not a removal")
        _exact_keys(raw, {"node", "reason"}, f"a removal in {where}")
        if raw["reason"] not in _REMOVAL_REASONS:
            raise ArtifactError(f"record {where}: {raw['reason']!r} is not one of "
                                f"{', '.join(_REMOVAL_REASONS)}")
        return cls(Ref.from_dict(raw["node"], where), raw["reason"])


@dataclass(frozen=True)
class EdgeChange:
    """One edge a step of the pipeline took out of a graph or put into it."""
    action: str                    # "added" or "removed"
    source: Ref
    relation: str
    target: Ref

    def to_dict(self) -> dict:
        return {"action": self.action, "source": self.source.to_dict(),
                "relation": self.relation, "target": self.target.to_dict()}

    @classmethod
    def from_dict(cls, raw: object, where: str) -> "EdgeChange":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not an edge change")
        _exact_keys(raw, {"action", "source", "relation", "target"}, f"an edge change in {where}")
        if raw["action"] not in ("added", "removed"):
            raise ArtifactError(f"record {where}: {raw['action']!r} is not removed or added")
        if not isinstance(raw["relation"], str):
            raise ArtifactError(f"record {where}: a relation is text")
        if where == "argument_dependency_changes" and raw["action"] != "added":
            raise ArtifactError("record argument_dependency_changes: this step removes nothing, "
                                f"and {raw['action']!r} says otherwise")
        if where == "ordering_repairs" and raw["action"] != "removed":
            raise ArtifactError("record ordering_repairs: this step adds nothing, "
                                f"and {raw['action']!r} says otherwise")
        return cls(raw["action"], Ref.from_dict(raw["source"], where), raw["relation"],
                   Ref.from_dict(raw["target"], where))


@dataclass(frozen=True)
class CompletionEvent:
    """One step of carrying out a `COMPLETE ... NOW_AVAILABLE` declaration.

    `node` is the information node in the assembled graph. `producer` is the computation that was
    going to produce it, which exists only in the previous graph and therefore has its own field.
    """
    action: str
    node: Ref
    detail: str = ""
    producer: Ref | None = None

    def to_dict(self) -> dict:
        return {"action": self.action, "node": self.node.to_dict(), "detail": self.detail,
                "producer": self.producer.to_dict() if self.producer is not None else None}

    @classmethod
    def from_dict(cls, raw: object, where: str = "completion_changes") -> "CompletionEvent":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not a completion change")
        _exact_keys(raw, {"action", "node", "detail", "producer"}, f"a change in {where}")
        if raw["action"] not in _COMPLETION_ACTIONS:
            raise ArtifactError(f"record {where}: {raw['action']!r} is not one of "
                                f"{', '.join(_COMPLETION_ACTIONS)}")
        if not isinstance(raw["detail"], str):
            raise ArtifactError(f"record {where}: a detail is text")
        producer = (None if raw["producer"] is None
                    else Ref.from_dict(raw["producer"], f"{where} producer"))
        if (raw["action"] == "producer_removed") != (producer is not None):
            raise ArtifactError("record completion_changes: a producer belongs to "
                                "producer_removed and to nothing else")
        return cls(raw["action"], Ref.from_dict(raw["node"], where), raw["detail"], producer)


@dataclass(frozen=True)
class Renaming:
    """What one entity of the answer became in the assembled graph."""
    source: Ref                    # previous anchor, or a label the model wrote
    target: Ref                    # assembled

    def to_dict(self) -> dict:
        return {"from": self.source.to_dict(), "to": self.target.to_dict()}

    @classmethod
    def from_dict(cls, raw: object, where: str = "id_map") -> "Renaming":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not a mapping entry")
        _exact_keys(raw, {"from", "to"}, f"an entry in {where}")
        source = Ref.from_dict(raw["from"], where)
        target = Ref.from_dict(raw["to"], where)
        if source.space not in (PREVIOUS, REVISION):
            raise ArtifactError(f"record {where}: a mapping starts in {PREVIOUS} or {REVISION}")
        if target.space != ASSEMBLED:
            raise ArtifactError(f"record {where}: a mapping ends in {ASSEMBLED}")
        return cls(source, target)


@dataclass(frozen=True)
class Report:
    """One reason something was refused: a fault from application or a validation violation."""
    code: str
    message: str
    nodes: tuple[Ref, ...] = ()
    sites: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "nodes": [n.to_dict() for n in self.nodes], "sites": list(self.sites)}

    @classmethod
    def from_dict(cls, raw: object, where: str) -> "Report":
        if not isinstance(raw, dict):
            raise ArtifactError(f"record {where}: {raw!r} is not a report")
        _exact_keys(raw, {"code", "message", "nodes", "sites"}, f"a report in {where}")
        for name in ("code", "message"):
            if not isinstance(raw[name], str):
                raise ArtifactError(f"record {where}: a {name} is text")
        if not isinstance(raw["nodes"], list) or not isinstance(raw["sites"], list):
            raise ArtifactError(f"record {where}: nodes and sites are lists")
        return cls(raw["code"], raw["message"],
                   tuple(Ref.from_dict(n, where) for n in raw["nodes"]),
                   tuple(_strings(raw["sites"], f"{where} sites")))


_REMOVAL_REASONS = ("affected_region", "region_internal", "invalidated_information")
_COMPLETION_ACTIONS = ("became_available", "producer_removed", "content_replaced",
                       "provenance_materialized")


# --------------------------------------------------------------------------- the record

@dataclass(frozen=True)
class RevisionRecord:
    """One boundary under the local-revision updater.

    Every change is filed under who made it. The model's own edits, the removal of regions it named,
    the completions it declared, and each deterministic step the code took afterwards are separate
    fields, because a record that pooled them could not answer the only question worth asking of it:
    what did the model actually get right.

    Every node reference carries its identity space, so two ids that read alike and mean different
    nodes cannot be joined by accident.
    """
    goal: str
    rules: str
    delta_h: str
    previous_snapshot: dict
    model_call: ModelCall
    raw_output: str
    attempts: tuple[tuple[int, str, str], ...]
    normalizations: tuple[str, ...]
    parse_errors: tuple[tuple[int, str], ...]
    empty_revision: bool
    affected_roots: tuple[Ref, ...]
    touched_nodes: tuple[Ref, ...]
    removed_nodes: tuple[Removal, ...]
    removed_edges: tuple[EdgeChange, ...]
    replacement_boundary_changes: tuple[EdgeChange, ...]
    completion_changes: tuple[CompletionEvent, ...]
    id_map: tuple[Renaming, ...]
    newly_created_then_collected: tuple[Ref, ...]
    argument_dependency_changes: tuple[EdgeChange, ...]
    interface_changes: tuple[EdgeChange, ...]
    ordering_repairs: tuple[EdgeChange, ...]
    faults: tuple[Report, ...]
    violations: tuple[Report, ...]
    accepted: bool
    assembled_snapshot: dict | None
    resulting_snapshot: dict
    collected: tuple[Ref, ...]
    handover: str

    @property
    def prompt_sha(self) -> str:
        return prompt_sha(self.model_call.system)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "rules": self.rules,
            "delta_h": self.delta_h,
            "previous_snapshot": self.previous_snapshot,
            "model_call": {
                "system": self.model_call.system,
                "user": self.model_call.user,
                "config": [[k, v] for k, v in self.model_call.config],
            },
            "prompt_sha": self.prompt_sha,
            "raw_output": self.raw_output,
            "attempts": [list(attempt) for attempt in self.attempts],
            "normalizations": list(self.normalizations),
            "parse_errors": [[line, message] for line, message in self.parse_errors],
            "empty_revision": self.empty_revision,
            "id_spaces": {
                PREVIOUS: "ids in previous_snapshot",
                REVISION: "labels the model wrote in its revision",
                ASSEMBLED: "canonical ids in assembled_snapshot, kept by resulting_snapshot",
            },
            "affected_roots": [r.to_dict() for r in self.affected_roots],
            "touched_nodes": [r.to_dict() for r in self.touched_nodes],
            "removed_nodes": [r.to_dict() for r in self.removed_nodes],
            "removed_edges": [c.to_dict() for c in self.removed_edges],
            "replacement_boundary_changes": [c.to_dict() for c
                                             in self.replacement_boundary_changes],
            "completion_changes": [c.to_dict() for c in self.completion_changes],
            "id_map": [m.to_dict() for m in self.id_map],
            "newly_created_then_collected": [r.to_dict() for r
                                             in self.newly_created_then_collected],
            "argument_dependency_changes": [c.to_dict() for c
                                            in self.argument_dependency_changes],
            "interface_changes": [c.to_dict() for c in self.interface_changes],
            "ordering_repairs": [c.to_dict() for c in self.ordering_repairs],
            "faults": [f.to_dict() for f in self.faults],
            "violations": [v.to_dict() for v in self.violations],
            "accepted": self.accepted,
            "assembled_snapshot": self.assembled_snapshot,
            "resulting_snapshot": self.resulting_snapshot,
            "collected": [r.to_dict() for r in self.collected],
            "handover": self.handover,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RevisionRecord":
        _exact_keys(raw, set(_REVISION_FIELD_TYPES), "a revision record")
        for name, expected in _REVISION_FIELD_TYPES.items():
            if expected is not None and not isinstance(raw[name], expected):
                raise ArtifactError(f"revision record {name}: expected {_name_of(expected)}, "
                                    f"got {type(raw[name]).__name__}")
        for name in ("accepted", "empty_revision"):
            if isinstance(raw[name], int) and not isinstance(raw[name], bool):
                raise ArtifactError(f"revision record {name}: expected true or false")
        if set(raw["id_spaces"]) != set(ID_SPACES):
            raise ArtifactError(f"revision record id_spaces: expected {', '.join(ID_SPACES)}")
        call = _call_from_dict(raw["model_call"])
        if raw["prompt_sha"] != prompt_sha(call.system):
            raise ArtifactError("revision record prompt_sha does not hash the system message "
                                "it sits beside")
        assembled = raw["assembled_snapshot"]
        if assembled is not None and not isinstance(assembled, dict):
            raise ArtifactError("revision record assembled_snapshot: expected an object or null")

        _rebuild(raw["previous_snapshot"], "previous_snapshot")
        resulting = _rebuild(raw["resulting_snapshot"], "resulting_snapshot")
        if assembled is not None:
            _rebuild(assembled, "assembled_snapshot")

        from .rendering import render
        if raw["handover"] != render(resulting):
            raise ArtifactError("revision record handover is not the rendering of its "
                                "resulting graph")

        attempts = tuple(_attempt(item) for item in raw["attempts"])
        _check_attempts(attempts)
        fields = dict(
            normalizations=tuple(_strings(raw["normalizations"], "normalizations")),
            parse_errors=tuple(_parse_error(item) for item in raw["parse_errors"]),
            affected_roots=_refs(raw["affected_roots"], "affected_roots", PREVIOUS),
            touched_nodes=_refs(raw["touched_nodes"], "touched_nodes", PREVIOUS),
            removed_nodes=tuple(Removal.from_dict(item) for item in raw["removed_nodes"]),
            removed_edges=_edges(raw["removed_edges"], "removed_edges", PREVIOUS),
            replacement_boundary_changes=_edges(raw["replacement_boundary_changes"],
                                                "replacement_boundary_changes", ASSEMBLED),
            completion_changes=tuple(CompletionEvent.from_dict(item)
                                     for item in raw["completion_changes"]),
            id_map=tuple(Renaming.from_dict(item) for item in raw["id_map"]),
            newly_created_then_collected=_refs(raw["newly_created_then_collected"],
                                               "newly_created_then_collected", ASSEMBLED),
            argument_dependency_changes=_edges(raw["argument_dependency_changes"],
                                               "argument_dependency_changes", ASSEMBLED),
            interface_changes=_edges(raw["interface_changes"], "interface_changes", ASSEMBLED),
            ordering_repairs=_edges(raw["ordering_repairs"], "ordering_repairs", ASSEMBLED),
            faults=tuple(Report.from_dict(item, "faults") for item in raw["faults"]),
            violations=tuple(Report.from_dict(item, "violations") for item in raw["violations"]),
            collected=_refs(raw["collected"], "collected", ASSEMBLED),
        )
        _check_removal_spaces(fields["removed_nodes"])
        _check_completion_spaces(fields["completion_changes"])
        _check_revision_outcome(raw)

        return cls(
            goal=raw["goal"], rules=raw["rules"], delta_h=raw["delta_h"],
            previous_snapshot=raw["previous_snapshot"], model_call=call,
            raw_output=raw["raw_output"], attempts=attempts,
            empty_revision=raw["empty_revision"], accepted=raw["accepted"],
            assembled_snapshot=assembled, resulting_snapshot=raw["resulting_snapshot"],
            handover=raw["handover"], **fields,
        )


_REVISION_FIELD_TYPES: dict[str, object] = {
    "goal": str, "rules": str, "delta_h": str, "previous_snapshot": dict,
    "model_call": dict, "prompt_sha": str, "raw_output": str, "attempts": list,
    "normalizations": list, "parse_errors": list, "empty_revision": bool, "id_spaces": dict,
    "affected_roots": list, "touched_nodes": list,
    "removed_nodes": list, "removed_edges": list, "replacement_boundary_changes": list,
    "completion_changes": list, "id_map": list, "newly_created_then_collected": list,
    "argument_dependency_changes": list, "interface_changes": list, "ordering_repairs": list,
    "faults": list, "violations": list,
    "accepted": bool, "assembled_snapshot": None, "resulting_snapshot": dict,
    "collected": list, "handover": str,
}


def _refs(raw: list, where: str, space: str) -> tuple[Ref, ...]:
    """Read references and hold each field to the one graph it can be talking about."""
    refs = tuple(Ref.from_dict(item, where) for item in raw)
    wrong = [r for r in refs if r.space != space]
    if wrong:
        raise ArtifactError(f"record {where}: holds {space} references, and found "
                            f"{', '.join(str(r) for r in wrong)}")
    return refs


def _edges(raw: list, where: str, space: str) -> tuple[EdgeChange, ...]:
    changes = tuple(EdgeChange.from_dict(item, where) for item in raw)
    wrong = [r for c in changes for r in (c.source, c.target) if r.space != space]
    if wrong:
        raise ArtifactError(f"record {where}: holds {space} references, and found "
                            f"{', '.join(str(r) for r in wrong)}")
    return changes


def _check_removal_spaces(removals: tuple[Removal, ...]) -> None:
    """A removed node is gone, so it has no assembled identity and can only be a previous one."""
    wrong = [r for r in removals if r.node.space != PREVIOUS]
    if wrong:
        raise ArtifactError("record removed_nodes: a removed node exists only in the previous "
                            f"graph, and found {', '.join(str(r.node) for r in wrong)}")


def _check_completion_spaces(events: tuple[CompletionEvent, ...]) -> None:
    for event in events:
        if event.node.space != ASSEMBLED:
            raise ArtifactError("record completion_changes: the information node is the one in "
                                f"the assembled graph, and found {event.node}")
        if event.producer is not None and event.producer.space != PREVIOUS:
            raise ArtifactError("record completion_changes: the producer was removed and exists "
                                f"only in the previous graph, and found {event.producer}")


def _attempt(item: object) -> tuple[int, str, str]:
    if not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], int) \
            or isinstance(item[0], bool) or not isinstance(item[1], str) \
            or not isinstance(item[2], str):
        raise ArtifactError(f"revision record attempts: {item!r} is not an ordinal, an outcome "
                            "and a detail")
    return item[0], item[1], item[2]


def _check_attempts(attempts: tuple[tuple[int, str, str], ...]) -> None:
    """A record only exists when a call returned, so the attempts end in exactly one completion.

    Numbering is checked too. Attempts that skipped an ordinal would be a record of a different
    sequence of calls than the one that happened, and the point of recording them is to know how
    often the provider had to be asked twice.
    """
    if not attempts:
        raise ArtifactError("revision record attempts: a record exists, so a call was made")
    if [ordinal for ordinal, _, _ in attempts] != list(range(1, len(attempts) + 1)):
        raise ArtifactError("revision record attempts: ordinals are not 1..n in order")
    outcomes = [outcome for _, outcome, _ in attempts]
    if outcomes[-1] != "completion":
        raise ArtifactError("revision record attempts: the last attempt did not complete, "
                            "and there would be no record if none had")
    if "completion" in outcomes[:-1]:
        raise ArtifactError("revision record attempts: a completion is not retried")


def _check_revision_outcome(raw: dict) -> None:
    """The same three shapes as a regeneration record, plus the empty revision.

    An empty revision is accepted and changes nothing, so it is the one accepted shape whose
    resulting graph must equal the previous one and whose change fields must all be empty. Letting
    a record claim an empty revision alongside recorded work would make the no-op unfalsifiable.
    """
    accepted, assembled = raw["accepted"], raw["assembled_snapshot"]
    parse_errors, faults, violations = raw["parse_errors"], raw["faults"], raw["violations"]

    if raw["empty_revision"]:
        if parse_errors:
            raise ArtifactError("revision record says the revision was empty and also lists "
                                "parse errors")
        if not accepted:
            raise ArtifactError("revision record refused an empty revision, which is a no-op")
        for name in ("affected_roots", "touched_nodes", "removed_nodes", "removed_edges",
                     "replacement_boundary_changes", "completion_changes",
                     "argument_dependency_changes", "interface_changes", "ordering_repairs",
                     "collected", "newly_created_then_collected"):
            if raw[name]:
                raise ArtifactError(f"revision record says the revision was empty and records "
                                    f"{name}")
        if raw["resulting_snapshot"] != raw["previous_snapshot"]:
            raise ArtifactError("revision record says the revision was empty and changed the graph")
        return

    if accepted:
        if assembled is None:
            raise ArtifactError("revision record says accepted with nothing assembled")
        if parse_errors:
            raise ArtifactError("revision record says accepted and also lists parse errors")
        if faults:
            raise ArtifactError("revision record says accepted and also lists faults")
        if violations:
            raise ArtifactError("revision record says accepted and also lists violations")
        return

    if raw["collected"]:
        raise ArtifactError("revision record collected information from a graph it did not accept")
    if raw["resulting_snapshot"] != raw["previous_snapshot"]:
        raise ArtifactError("revision record refused a revision and did not keep the "
                            "previous graph")
    if not (parse_errors or faults or violations):
        raise ArtifactError("revision record refused a revision and gave no reason")
    if parse_errors and (faults or violations):
        raise ArtifactError("revision record has parse errors and also reasons that require "
                            "a parsed revision")
    if assembled is None and violations:
        raise ArtifactError("revision record has violations for a graph it never assembled")
