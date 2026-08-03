"""The updater: one model call per boundary, producing a local revision.

This is the normal path for every boundary, empty graph or not. An empty previous graph has no
anchors to name, so its revision is all `ADD`; there is no second protocol and no routing on the
size of the state. Complete-graph regeneration stays in `regeneration.py`, unchanged, as a baseline
and a debugging tool.

The division of labour is the point. The model decides what the trajectory changed about the plan
and what consumes what; the code preserves everything unnamed, closes the removed regions, builds
the directed edges, completes what the declarations already determine, validates, collects and
renders. A refused revision leaves the previous graph untouched and `delta_h` is discarded -- not
buffered, not replayed. Failing to absorb a boundary is a real loss and belongs in the results as
one; carrying the slice forward would hand the method a second attempt the schedule never gave it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .artifacts import (
    CompletionEvent, ConfigScalar, EdgeChange, ModelCall, Ref, Removal, Renaming, Report,
    RevisionRecord, assembled_ref, authored_ref, freeze_config, previous_ref,
)
from .lifecycle import replace
from .protocol import to_protocol
from .regeneration import Model, PromptError
from .rendering import render
from .revision import RevisionChanges, apply_revision
from .revision_parser import parse_revision
from .retry import MAX_ATTEMPTS, call_with_retry
from .revision_protocol import GRAMMAR
from .state_graph import StateGraph

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "revise_graph.md"
GRAMMAR_PLACEHOLDER = "{{REVISION_GRAMMAR}}"


@dataclass(frozen=True)
class UpdateResult:
    graph: StateGraph
    record: RevisionRecord


def load_prompt(path: Path = PROMPT_PATH) -> str:
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as err:
        raise PromptError(f"the prompt could not be read at {path}: {err}") from err
    found = template.count(GRAMMAR_PLACEHOLDER)
    if found != 1:
        raise PromptError(f"the prompt holds {GRAMMAR_PLACEHOLDER} {found} times, expected once")
    return template.replace(GRAMMAR_PLACEHOLDER, GRAMMAR)


def build_user_message(goal: str, rules: str, previous: StateGraph, delta_h: str) -> str:
    """The four inputs, in this order, inserted exactly as given.

    The only transformation anywhere is `to_protocol` on the previous graph, which is how the model
    sees the ids it may anchor to.
    """
    return (
        "BEGIN_ORIGINAL_GOAL\n" + goal + "\nEND_ORIGINAL_GOAL\n"
        "\nBEGIN_FIXED_RULES\n" + rules + "\nEND_FIXED_RULES\n"
        "\nBEGIN_PREVIOUS_GRAPH\n" + to_protocol(previous) + "END_PREVIOUS_GRAPH\n"
        "\nBEGIN_DELTA_H\n" + delta_h + "\nEND_DELTA_H\n"
    )


def build_call(goal: str, rules: str, previous: StateGraph, delta_h: str,
               config: Mapping[str, ConfigScalar]) -> ModelCall:
    return ModelCall(system=load_prompt(),
                     user=build_user_message(goal, rules, previous, delta_h),
                     config=freeze_config(config))


def update_graph(goal: str, rules: str, previous: StateGraph, delta_h: str,
                 model: Model, config: Mapping[str, ConfigScalar],
                 max_attempts: int = MAX_ATTEMPTS) -> UpdateResult:
    """Ask for a local revision and take it only if the whole graph it implies holds together.

    `max_attempts` bounds provider failures only (`retry`), and never a revision this system read
    and rejected. If every attempt fails the call raises and there is no result: a provider that
    did not answer is not a compressor that answered badly.
    """
    call = build_call(goal, rules, previous, delta_h, config)
    previous_snapshot = previous.to_snapshot()

    raw, attempts = call_with_retry(model, call, max_attempts=max_attempts)
    if not isinstance(raw, str):
        raise TypeError(f"a model returns text, got {type(raw).__name__}")

    outcome = parse_revision(raw)
    graph = previous
    changes = RevisionChanges()
    faults: tuple[Report, ...] = ()
    violations: tuple[Report, ...] = ()
    collected: tuple[Ref, ...] = ()
    interface_changes: tuple[EdgeChange, ...] = ()
    argument_changes: tuple[EdgeChange, ...] = ()
    ordering_repairs: tuple[EdgeChange, ...] = ()
    assembled_snapshot = None
    empty = False

    if outcome.revision is not None and outcome.revision.is_empty:
        # Nothing to assemble and nothing to change. Handing the previous graph to `replace` would
        # run completion and collection over the object the caller still holds, so a boundary that
        # said nothing could quietly edit the state it was preserving.
        empty = True
    elif outcome.revision is not None:
        applied = apply_revision(previous, outcome.revision)
        changes = applied.changes
        # A fault names entities as the model wrote them, so each is tagged by what it is: its own
        # label, or an anchor into the graph it was shown.
        faults = tuple(Report(f.code, f.message, tuple(authored_ref(n) for n in f.nodes), f.sites)
                       for f in applied.faults)
        if applied.graph is not None:
            # Before any completion, so the record can tell what the revision asked for apart from
            # what the code then worked out. Everything from here on is in assembled ids, which
            # collection does not renumber, so the resulting graph keeps them.
            assembled_snapshot = applied.graph.to_snapshot()
            replacement = replace(previous, applied.graph)
            graph = replacement.graph
            violations = tuple(Report(v.code, v.message,
                                      tuple(assembled_ref(n) for n in v.nodes))
                               for v in replacement.violations)
            collected = tuple(assembled_ref(n) for n in replacement.collected)
            interface_changes = _assembled_edges(replacement.interface_changes)
            argument_changes = _assembled_edges(replacement.argument_dependency_changes)
            ordering_repairs = _assembled_edges(replacement.ordering_repairs)

    # An empty revision is a boundary that ended normally, and the state it leaves behind is the
    # state it was given, unrenumbered and with its derived edges intact.
    accepted = empty or (outcome.revision is not None and not faults and not violations
                         and assembled_snapshot is not None)

    record = RevisionRecord(
        goal=goal, rules=rules, delta_h=delta_h, previous_snapshot=previous_snapshot,
        model_call=call, raw_output=raw,
        attempts=tuple((a.ordinal, a.outcome, a.detail) for a in attempts),
        normalizations=outcome.normalizations,
        parse_errors=tuple((e.line, e.message) for e in outcome.errors),
        empty_revision=empty,
        # What the model named, and what left with it, are the graph it was shown.
        affected_roots=tuple(previous_ref(n) for n in changes.affected_roots),
        touched_nodes=tuple(previous_ref(n) for n in changes.touched_nodes),
        removed_nodes=tuple(Removal(previous_ref(r.node_id), r.reason)
                            for r in changes.removed_nodes),
        removed_edges=tuple(EdgeChange(c.action, previous_ref(c.source), c.relation.value,
                                       previous_ref(c.target)) for c in changes.removed_edges),
        replacement_boundary_changes=tuple(
            EdgeChange(c.action, assembled_ref(c.source), c.relation.value,
                       assembled_ref(c.target)) for c in changes.replacement_boundary_changes),
        completion_changes=tuple(
            CompletionEvent(c.action, assembled_ref(c.node_id), c.detail,
                            None if c.producer_id is None else previous_ref(c.producer_id))
            for c in changes.completion_changes),
        id_map=tuple(Renaming(authored_ref(source), assembled_ref(target))
                     for source, target in changes.id_map),
        newly_created_then_collected=_newly_created_then_collected(changes, collected),
        argument_dependency_changes=argument_changes,
        interface_changes=interface_changes,
        ordering_repairs=ordering_repairs,
        faults=faults, violations=violations, accepted=accepted,
        assembled_snapshot=assembled_snapshot,
        resulting_snapshot=graph.to_snapshot(), collected=collected, handover=render(graph),
    )
    return UpdateResult(graph=graph, record=record)


def _assembled_edges(changes) -> tuple[EdgeChange, ...]:
    return tuple(EdgeChange(c.action, assembled_ref(c.source), c.relation.value,
                            assembled_ref(c.target)) for c in changes)


def _newly_created_then_collected(changes: RevisionChanges,
                                  collected: tuple[Ref, ...]) -> tuple[Ref, ...]:
    """Information this revision introduced and this same boundary threw away.

    Not a violation, and deliberately not treated as one. It has two readings the code cannot tell
    apart: the call is worth making for its effect and its return value genuinely has no consumer,
    or the model established a result and forgot to connect it to the work that needs it. The
    second is a real absorption failure and the first is correct pruning, so collection stays as it
    is and this is surfaced for a human to judge in the recurrent audit.

    The join runs through `id_map`, which is the only thing that can relate a label the model wrote
    to the id its node was given.
    """
    dead = {ref.id for ref in collected}
    return tuple(assembled_ref(new_id) for label, new_id in sorted(changes.id_map)
                 if label.startswith("+") and new_id in dead)
