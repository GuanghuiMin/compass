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


@dataclass(frozen=True)
class RevisionRecord:
    """One boundary under the local-revision updater.

    Every change is filed under who made it. The model's own edits, the removal of regions it named,
    the completions it declared, and each deterministic step the code took afterwards are separate
    fields, because a record that pooled them could not answer the only question worth asking of it:
    what did the model actually get right.
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
    affected_roots: tuple[str, ...]
    touched_nodes: tuple[str, ...]
    removed_nodes: tuple[tuple[str, str], ...]
    removed_edges: tuple[tuple[str, str, str, str], ...]
    replacement_boundary_changes: tuple[tuple[str, str, str, str], ...]
    completion_changes: tuple[tuple[str, str, str], ...]
    id_map: tuple[tuple[str, str], ...]
    newly_created_then_collected: tuple[str, ...]
    argument_dependency_changes: tuple[tuple[str, str, str, str], ...]
    interface_changes: tuple[tuple[str, str, str, str], ...]
    ordering_repairs: tuple[tuple[str, str, str, str], ...]
    faults: tuple[tuple[str, str, tuple[str, ...]], ...]
    violations: tuple[tuple[str, str, tuple[str, ...]], ...]
    accepted: bool
    assembled_snapshot: dict | None
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
            "attempts": [list(attempt) for attempt in self.attempts],
            "normalizations": list(self.normalizations),
            "parse_errors": [[line, message] for line, message in self.parse_errors],
            "empty_revision": self.empty_revision,
            "affected_roots": list(self.affected_roots),
            "touched_nodes": list(self.touched_nodes),
            "removed_nodes": [list(pair) for pair in self.removed_nodes],
            "removed_edges": [list(change) for change in self.removed_edges],
            "replacement_boundary_changes": [list(change) for change
                                             in self.replacement_boundary_changes],
            "completion_changes": [list(change) for change in self.completion_changes],
            "id_map": [list(pair) for pair in self.id_map],
            "newly_created_then_collected": list(self.newly_created_then_collected),
            "argument_dependency_changes": [list(change) for change
                                            in self.argument_dependency_changes],
            "interface_changes": [list(change) for change in self.interface_changes],
            "ordering_repairs": [list(change) for change in self.ordering_repairs],
            "faults": [[code, message, list(nodes)] for code, message, nodes in self.faults],
            "violations": [[code, message, list(nodes)]
                           for code, message, nodes in self.violations],
            "accepted": self.accepted,
            "assembled_snapshot": self.assembled_snapshot,
            "resulting_snapshot": self.resulting_snapshot,
            "collected": list(self.collected),
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
        normalizations = tuple(_strings(raw["normalizations"], "normalizations"))
        parse_errors = tuple(_parse_error(item) for item in raw["parse_errors"])
        removed_nodes = tuple(_removal(item) for item in raw["removed_nodes"])
        removed_edges = tuple(_interface_change(item, "removed_edges")
                              for item in raw["removed_edges"])
        boundary = tuple(_interface_change(item, "replacement_boundary_changes")
                         for item in raw["replacement_boundary_changes"])
        completion = tuple(_completion_change(item) for item in raw["completion_changes"])
        id_map = tuple(_pair(item, "id_map") for item in raw["id_map"])
        stillborn = tuple(_strings(raw["newly_created_then_collected"],
                                   "newly_created_then_collected"))
        interface_changes = tuple(_interface_change(item) for item in raw["interface_changes"])
        argument_changes = tuple(_interface_change(item, "argument_dependency_changes")
                                 for item in raw["argument_dependency_changes"])
        ordering_repairs = tuple(_interface_change(item, "ordering_repairs")
                                 for item in raw["ordering_repairs"])
        faults = tuple(_violation(item) for item in raw["faults"])
        violations = tuple(_violation(item) for item in raw["violations"])
        collected = tuple(_strings(raw["collected"], "collected"))
        _check_revision_outcome(raw)

        return cls(
            goal=raw["goal"], rules=raw["rules"], delta_h=raw["delta_h"],
            previous_snapshot=raw["previous_snapshot"], model_call=call,
            raw_output=raw["raw_output"], attempts=attempts, normalizations=normalizations,
            parse_errors=parse_errors, empty_revision=raw["empty_revision"],
            affected_roots=tuple(_strings(raw["affected_roots"], "affected_roots")),
            touched_nodes=tuple(_strings(raw["touched_nodes"], "touched_nodes")),
            removed_nodes=removed_nodes, removed_edges=removed_edges,
            replacement_boundary_changes=boundary, completion_changes=completion, id_map=id_map,
            newly_created_then_collected=stillborn,
            argument_dependency_changes=argument_changes, interface_changes=interface_changes,
            ordering_repairs=ordering_repairs, faults=faults, violations=violations,
            accepted=raw["accepted"], assembled_snapshot=assembled,
            resulting_snapshot=raw["resulting_snapshot"], collected=collected,
            handover=raw["handover"],
        )


_REVISION_FIELD_TYPES: dict[str, object] = {
    "goal": str, "rules": str, "delta_h": str, "previous_snapshot": dict,
    "model_call": dict, "prompt_sha": str, "raw_output": str, "attempts": list,
    "normalizations": list,
    "parse_errors": list, "empty_revision": bool, "affected_roots": list, "touched_nodes": list,
    "removed_nodes": list, "removed_edges": list, "replacement_boundary_changes": list,
    "completion_changes": list, "id_map": list, "newly_created_then_collected": list,
    "argument_dependency_changes": list,
    "interface_changes": list, "ordering_repairs": list, "faults": list, "violations": list,
    "accepted": bool, "assembled_snapshot": None, "resulting_snapshot": dict,
    "collected": list, "handover": str,
}

_REMOVAL_REASONS = ("affected_region", "region_internal", "invalidated_information")
_COMPLETION_ACTIONS = ("became_available", "producer_removed", "content_replaced",
                       "provenance_materialized")


def _pair(item: object, where: str) -> tuple[str, str]:
    if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, str) for x in item):
        raise ArtifactError(f"revision record {where}: {item!r} is not a pair of names")
    return item[0], item[1]


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


def _removal(item: object) -> tuple[str, str]:
    node, reason = _pair(item, "removed_nodes")
    if reason not in _REMOVAL_REASONS:
        raise ArtifactError(f"revision record removed_nodes: {reason!r} is not one of "
                            f"{', '.join(_REMOVAL_REASONS)}")
    return node, reason


def _completion_change(item: object) -> tuple[str, str, str]:
    if not isinstance(item, list) or len(item) != 3 or not all(isinstance(x, str) for x in item):
        raise ArtifactError(f"revision record completion_changes: {item!r} is not an action, "
                            "a node and a detail")
    if item[0] not in _COMPLETION_ACTIONS:
        raise ArtifactError(f"revision record completion_changes: {item[0]!r} is not one of "
                            f"{', '.join(_COMPLETION_ACTIONS)}")
    return item[0], item[1], item[2]


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
