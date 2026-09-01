"""The updater as the live agent's compactor.

Offline the boundary is one call and one record. Online it is two moments that must not come apart:
the host asks for a handover, and only afterwards does it splice that handover into the context the
agent will act on. Between the two, the graph must still be the old graph, because if the splice
fails and the graph has already moved, the state of record describes a context the agent never saw.

So this prepares and then commits. `process` computes the whole transition, writes its record to a
pending path, reads it back strictly, and returns the handover -- while leaving the active graph
alone. `commit_pending` runs after the host's splice returns, rebinds the graph and moves the record
into place. `abort_pending` throws the prepared transition away. Nothing else may touch the graph.

The adapter is deliberately narrow. It holds one graph for one episode, calls `update_graph` and
nothing else, and never sees the retained full history the host still has in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .artifacts import ConfigScalar, RevisionRecord
from .rendering import render
from .retry import MAX_ATTEMPTS, ExhaustedAttempts
from .run import FROZEN_CONFIG
from .state_graph import StateGraph
from .update import Model, update_graph

ACCEPTED, REFUSED, EMPTY = "accepted", "refused", "empty"


class OnlineIntegrationError(RuntimeError):
    """The live loop and this adapter disagree about the state of the boundary.

    Always a defect in the integration rather than in a revision, so it stops the task instead of
    being recorded as a method failure.
    """


@dataclass
class PendingTransition:
    """A computed boundary that the host has not yet spliced."""
    boundary_index: int
    previous_graph: StateGraph
    resulting_graph: StateGraph
    previous_handover: str
    handover: str
    status: str
    delta_h: str
    record: RevisionRecord
    pending_path: Path
    final_path: Path


@dataclass(frozen=True)
class CommittedBoundary:
    """A boundary whose graph and whose downstream context moved together."""
    boundary_index: int
    status: str
    handover: str
    delta_h: str
    path: Path
    record: RevisionRecord


@dataclass
class LocalRevisionOptimizer:
    """One episode's graph, driven by the host's existing compaction seam.

    Duck-typed to the history-optimizer interface the host expects, but the two methods it exposes
    beyond that -- `commit_pending` and `abort_pending` -- are the point: the host's seam cannot
    express a transition that is computed now and becomes true later, so the runner drives those.
    """
    goal: str
    rules: str
    model: Model
    count_tokens: Callable[[str], int]
    window: int
    boundaries_dir: Path
    pending_dir: Path
    config: Mapping[str, ConfigScalar] = field(default_factory=lambda: dict(FROZEN_CONFIG))
    max_attempts: int = MAX_ATTEMPTS

    graph: StateGraph = field(default_factory=StateGraph)
    boundary_index: int = 0
    pending: PendingTransition | None = None
    committed: list[CommittedBoundary] = field(default_factory=list)
    uncollected: CommittedBoundary | None = None

    # The host reads these off a history optimizer. They exist for the duck type and nothing here
    # depends on them.
    history: list = field(default_factory=list)
    history_summarization_threshold: int = 0

    def __post_init__(self) -> None:
        self.history_summarization_threshold = self.window
        # Both directories exist before the first boundary, so a commit cannot fail on a missing
        # parent after the host has already spliced.
        self.boundaries_dir = Path(self.boundaries_dir)
        self.pending_dir = Path(self.pending_dir)
        self.boundaries_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ the host's interface
    def check_summarization_needed(self, history_text: str, prev_history_summary=None) -> bool:
        """The host's existing trigger, unchanged: rendered pending history against the window."""
        text = (f"{prev_history_summary}\n{history_text}" if prev_history_summary else history_text)
        return self.count_tokens(text) > self.window

    def handover(self) -> str:
        """What the downstream agent would be shown for the state as it stands."""
        return render(self.graph)

    def process(self, task, history, prev_history_summary=None, raw_history=None,
                opt_args=None, target_tokens=None, **kw) -> str:
        """Prepare the transition and hand back the handover. The graph does not move here.

        `raw_history` and `opt_args` exist in the signature because the host passes them. They are
        the retained full history and the host's own bookkeeping, and reading either would give the
        updater evidence the method says it does not have.
        """
        if self.pending is not None:
            raise OnlineIntegrationError(
                f"boundary {self.pending.boundary_index} was prepared and never committed, "
                "so this boundary cannot start")

        previous_graph = self.graph
        previous_handover = render(previous_graph)
        self._check_host_agrees(prev_history_summary, previous_handover)

        delta_h = history if isinstance(history, str) else str(history)
        index = self.boundary_index
        result = update_graph(self.goal, self.rules, previous_graph, delta_h,
                              self.model, self.config, max_attempts=self.max_attempts)
        record = result.record

        if record.empty_revision:
            status = EMPTY
        elif record.accepted:
            status = ACCEPTED
        else:
            status = REFUSED
        if status != ACCEPTED and record.handover != previous_handover:
            # A refusal keeps the previous graph, so its rendering is the previous handover. If
            # those ever differed the agent would be handed a context for a state nothing holds.
            raise OnlineIntegrationError(
                f"boundary {index} was {status} and produced a handover that is not the "
                "previous one")

        pending_path = self.pending_dir / f"boundary_{index:03d}.json.pending"
        final_path = self.boundaries_dir / f"boundary_{index:03d}.json"
        if final_path.exists():
            raise OnlineIntegrationError(f"{final_path} already exists and records are never "
                                         "replaced")
        _write_and_read_back(pending_path, record)

        self.pending = PendingTransition(
            boundary_index=index, previous_graph=previous_graph,
            resulting_graph=result.graph, previous_handover=previous_handover,
            handover=record.handover, status=status, delta_h=delta_h, record=record,
            pending_path=pending_path, final_path=final_path)
        return record.handover

    # ------------------------------------------------------------------ the two phases
    def commit_pending(self) -> CommittedBoundary:
        """The host spliced successfully, so the boundary becomes true: graph, then record."""
        pending = self.pending
        if pending is None:
            raise OnlineIntegrationError("there is no prepared transition to commit")
        self.graph = pending.resulting_graph          # an in-memory rebind, nothing derived
        os.replace(pending.pending_path, pending.final_path)
        self.pending = None
        self.boundary_index = pending.boundary_index + 1
        committed = CommittedBoundary(
            boundary_index=pending.boundary_index, status=pending.status,
            handover=pending.handover, delta_h=pending.delta_h, path=pending.final_path,
            record=pending.record)
        self.committed.append(committed)
        self.uncollected = committed
        return committed

    def abort_pending(self) -> PendingTransition | None:
        """Drop the prepared transition. The graph never moved, so there is nothing to undo."""
        pending = self.pending
        if pending is None:
            return None
        self.pending = None
        try:
            pending.pending_path.unlink(missing_ok=True)
        except OSError:
            pass
        return pending

    def take_committed(self) -> CommittedBoundary | None:
        """The boundary whose first downstream decision has not been recorded yet."""
        committed, self.uncollected = self.uncollected, None
        return committed

    # ------------------------------------------------------------------ agreement with the host
    def _check_host_agrees(self, prev_history_summary, previous_handover: str) -> None:
        """The host's summary must be the handover this graph renders to.

        Checked before the call, so a desynchronised loop costs nothing and cannot produce a record
        of a boundary that was computed against the wrong state. Before the first commit the host
        has no summary at all, which is the one case where the two legitimately differ.
        """
        if not self.committed:
            if prev_history_summary in (None, ""):
                return
            raise OnlineIntegrationError(
                "the host carries a summary before any boundary committed, so something other "
                "than this adapter has been compacting")
        if prev_history_summary != previous_handover:
            raise OnlineIntegrationError(
                f"the host's summary is not the handover of the active graph at boundary "
                f"{self.boundary_index}: {len(prev_history_summary or '')} characters against "
                f"{len(previous_handover)}")


def _write_and_read_back(path: Path, record: RevisionRecord) -> None:
    """Write, then read back through the strict loader. A record that cannot be read is not kept."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=1) + "\n"
    path.write_text(payload, encoding="utf-8")
    try:
        RevisionRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as err:
        path.unlink(missing_ok=True)
        raise OnlineIntegrationError(f"{path.name} did not read back: {err}") from err


def continuation_of(messages: Sequence[dict], handover: str) -> bool:
    """Does this message list carry that handover?

    The one check that separates an action generated under the graph from an action that merely
    followed one in time.
    """
    return any(handover and handover in (m.get("content") or "") for m in messages)


__all__ = ["ACCEPTED", "REFUSED", "EMPTY", "CommittedBoundary", "ExhaustedAttempts",
           "LocalRevisionOptimizer", "OnlineIntegrationError", "PendingTransition",
           "continuation_of"]
