"""The one operator that produces every graph this system holds.

The first graph is this function with an empty previous graph. There is no separate construction path,
because two paths would eventually disagree about what a graph is.

Nothing here knows a model vendor. The caller passes a callable, so a test can hand it a fixed string
and the pipeline is exercised without a network. Nothing here interprets decoding configuration
either: it is frozen, carried inside the call the adapter receives, and recorded as that same call.

Failures keep their own names. A parse error or a validation violation leaves the previous graph
standing and is reported as what it is. A model that raises is not a graph that was rejected -- the
exception travels, nothing is retried, and no record is produced at all, because a service outage
recorded as a refused graph would read later as evidence that the compressor writes bad ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .artifacts import ConfigScalar, ModelCall, RegenerationRecord, freeze_config
from .lifecycle import replace
from .parser import parse
from .protocol import GRAMMAR, to_protocol
from .rendering import render
from .state_graph import StateGraph

Model = Callable[[ModelCall], str]

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "regenerate_graph.md"
GRAMMAR_PLACEHOLDER = "{{PROTOCOL_GRAMMAR}}"


class PromptError(ValueError):
    """The prompt template is not what this expects, found before anything is sent."""


@dataclass(frozen=True)
class RegenerationResult:
    graph: StateGraph
    record: RegenerationRecord


def load_prompt(path: Path = PROMPT_PATH) -> str:
    """Read the template and put the grammar in it, or fail before a call is made.

    The path is resolved from this module, not from the working directory, and there is no search and
    no fallback: a prompt found somewhere else is a different experiment.
    """
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

    The only transformation anywhere is `to_protocol` on the previous graph. Nothing is stripped,
    normalized or escaped, because the model has to see what the caller meant it to see.
    """
    return (
        "BEGIN_ORIGINAL_GOAL\n" + goal + "\nEND_ORIGINAL_GOAL\n"
        "\nBEGIN_FIXED_RULES\n" + rules + "\nEND_FIXED_RULES\n"
        "\nBEGIN_PREVIOUS_GRAPH\n" + to_protocol(previous) + "END_PREVIOUS_GRAPH\n"
        "\nBEGIN_DELTA_H\n" + delta_h + "\nEND_DELTA_H\n"
    )


def build_call(goal: str, rules: str, previous: StateGraph, delta_h: str,
               config: Mapping[str, ConfigScalar]) -> ModelCall:
    """The system message is the repository's prompt. There is no way to send another one.

    An experiment whose prompt cannot be recovered from the commit is not an experiment, so this takes
    no override, and the mapping goes to `freeze_config` exactly as the caller passed it -- converting
    it first would quietly accept shapes the interface does not, and could fold duplicate keys.
    """
    return ModelCall(system=load_prompt(),
                     user=build_user_message(goal, rules, previous, delta_h),
                     config=freeze_config(config))


def regenerate_graph(goal: str, rules: str, previous: StateGraph, delta_h: str,
                     model: Model, config: Mapping[str, ConfigScalar]) -> RegenerationResult:
    """Ask for the whole remaining graph, and take it only if it holds together."""
    call = build_call(goal, rules, previous, delta_h, config)
    previous_snapshot = previous.to_snapshot()

    raw = model(call)
    if not isinstance(raw, str):
        raise TypeError(f"a model returns text, got {type(raw).__name__}")

    outcome = parse(raw)
    # Before replacement, which collects dead information from the candidate in place. What the model
    # produced and what was committed are different things, and only one of them is recoverable after.
    candidate_snapshot = outcome.graph.to_snapshot() if outcome.graph is not None else None

    if outcome.graph is None:
        graph, violations, collected = previous, (), ()
        interface_changes, argument_changes = (), ()
    else:
        replacement = replace(previous, outcome.graph)
        graph = replacement.graph
        violations = tuple((v.code, v.message, v.nodes) for v in replacement.violations)
        collected = replacement.collected
        interface_changes = tuple((c.action, c.source, c.relation.value, c.target)
                                  for c in replacement.interface_changes)
        argument_changes = tuple((c.action, c.source, c.relation.value, c.target)
                                 for c in replacement.argument_dependency_changes)

    record = RegenerationRecord(
        goal=goal, rules=rules, delta_h=delta_h, previous_snapshot=previous_snapshot,
        model_call=call, raw_output=raw,
        normalizations=outcome.normalizations,
        parse_errors=tuple((e.line, e.message) for e in outcome.errors),
        parsed_candidate_snapshot=candidate_snapshot,
        interface_changes=interface_changes,
        argument_dependency_changes=argument_changes,
        violations=violations,
        accepted=outcome.graph is not None and not violations,
        resulting_snapshot=graph.to_snapshot(), collected=collected, handover=render(graph),
    )
    return RegenerationResult(graph=graph, record=record)
