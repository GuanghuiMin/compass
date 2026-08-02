"""`future_graph_v1r`: a refused revision does not consume the trajectory it was proposed from.

One behavioural change from `future_graph_v1`, and the reason for it is a state-transition rule
rather than a performance idea. A refusal establishes that the system could not prove a graph update
valid. It does not establish that the slice held nothing worth keeping. Consuming the slice anyway
is a transaction that rolled back and deleted its own source log, and in the six-task diagnostic
that is what took the credentials, the login, the groups, the expenses and the transactions out of
`83a7951_3` three times in a row.

So here the invariant is: **source is consumed only by a committed transition.** A refusal leaves
the graph, the host's session and its summary exactly as they were, and the slice stays in the live
conversation until some later accepted or accepted-empty transition absorbs it.

Nothing else moves. The grammar, prompt, parser, validator, crossing rules, retry policy, rendering
and the accepted and empty transitions are the v1 ones, reached through the v1 code, so this can
only measure the variable it changes.

Retention is a consequence of not splicing, not a second store. The adapter never concatenates a
slice: what the updater sees is whatever the host rendered, and the prefix checks below exist to
prove that and to stop the run if it is ever untrue.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import RevisionRecord
from .online import ACCEPTED, EMPTY, REFUSED, CommittedBoundary, LocalRevisionOptimizer
from .online import OnlineIntegrationError
from .rendering import render

METHOD = "future_graph_v1r"


class RevisionRefused(Exception):
    """A revision the system could not accept, raised so the host never reaches its splice.

    Carries identity and evidence only. It deliberately does not carry a handover: returning the
    previous one and letting the host splice it is the v1 behaviour this version exists to remove.
    """

    def __init__(self, *, revision_attempt_index: int, candidate_boundary_index: int,
                 record_path: Path, delta_h_sha256: str, delta_h_bytes: int,
                 record_status: str) -> None:
        super().__init__(f"revision attempt {revision_attempt_index} was refused for candidate "
                         f"boundary {candidate_boundary_index}; its slice is retained")
        self.revision_attempt_index = revision_attempt_index
        self.candidate_boundary_index = candidate_boundary_index
        self.record_path = Path(record_path)
        self.delta_h_sha256 = delta_h_sha256
        self.delta_h_bytes = delta_h_bytes
        self.record_status = record_status

    def to_dict(self) -> dict:
        return {"revision_attempt_index": self.revision_attempt_index,
                "candidate_boundary_index": self.candidate_boundary_index,
                "record_path": str(self.record_path),
                "delta_h_sha256": self.delta_h_sha256,
                "delta_h_bytes": self.delta_h_bytes,
                "record_status": self.record_status}


def sha256_of(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RetainedSlice:
    """One refused submission, and everything needed to prove it was neither lost nor duplicated."""
    slice_id: str
    origin_revision_attempt: int
    sha256: str
    bytes: int
    tokens: int
    included_in_revision_attempts: list[int] = field(default_factory=list)
    included_at_offset: list[int] = field(default_factory=list)
    absorbed_by_revision_attempt: int | None = None
    absorbed_by_boundary: int | None = None
    absorbed_by_empty_revision: bool = False
    unabsorbed_at_task_end: bool = False

    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "origin_revision_attempt": self.origin_revision_attempt,
            "sha256": self.sha256, "bytes": self.bytes, "tokens": self.tokens,
            "included_in_revision_attempts": list(self.included_in_revision_attempts),
            "included_at_offset": list(self.included_at_offset),
            "absorbed_by_revision_attempt": self.absorbed_by_revision_attempt,
            "absorbed_by_boundary": self.absorbed_by_boundary,
            "absorbed_by_empty_revision": self.absorbed_by_empty_revision,
            "unabsorbed_at_task_end": self.unabsorbed_at_task_end,
        }


def check_interval(previous: str, current: str) -> None:
    """The new submission must be the old one, unchanged, plus new turns after it.

    Checked rather than arranged. The adapter does not build this interval -- the host does, by not
    having spliced -- so the only honest thing is to verify the property and stop if it fails. A
    run that repaired the interval by prepending the old slice itself would be measuring its own
    concatenation instead of the host's retention.
    """
    if not current.startswith(previous):
        raise OnlineIntegrationError(
            "the retained slice is not a prefix of the next submission, so the host did not "
            "simply keep it")
    if len(current) <= len(previous):
        raise OnlineIntegrationError(
            "the next submission is not longer than the refused one, so no new trajectory "
            "reached the updater")
    if current.count(previous) != 1:
        raise OnlineIntegrationError(
            f"the refused slice appears {current.count(previous)} times in the next submission; "
            "it must appear exactly once, at the start")
    tail = current[len(previous):]
    if "ASSISTANT:" not in tail or "USER:" not in tail:
        raise OnlineIntegrationError(
            "no new action and observation follow the retained slice, so the updater would be "
            "resubmitting the same history")


@dataclass
class RetainingLocalRevisionOptimizer(LocalRevisionOptimizer):
    """The v1 optimizer, with the refusal branch changed and nothing else.

    `process` runs the inherited implementation whole -- same call, same parser, same application,
    same validation, same artifact read-back -- and only then decides what the host is told.
    """
    attempts_dir: Path | None = None

    revision_attempt_index: int = 0
    retained: list[RetainedSlice] = field(default_factory=list)
    refused_attempts: list[dict] = field(default_factory=list)
    accepted_attempts: list[dict] = field(default_factory=list)
    last_submission: str | None = None
    last_refused_slice: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attempts_dir is None:
            self.attempts_dir = self.boundaries_dir.parent / "attempts"
        self.attempts_dir = Path(self.attempts_dir)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ the one change
    def process(self, task, history, prev_history_summary=None, raw_history=None,
                opt_args=None, target_tokens=None, **kw) -> str:
        delta_h = history if isinstance(history, str) else str(history)

        # Before anything is sent: if a slice is being retained, this submission must be that
        # slice plus new turns. A failure here is an integration failure and costs no model call.
        if self.last_refused_slice is not None:
            check_interval(self.last_refused_slice, delta_h)

        attempt = self.revision_attempt_index
        candidate = self.boundary_index
        handover = super().process(task, delta_h, prev_history_summary=prev_history_summary,
                                   raw_history=raw_history, opt_args=opt_args,
                                   target_tokens=target_tokens, **kw)
        pending = self.pending
        if pending is None:                                    # empty revisions prepare nothing
            raise OnlineIntegrationError("process returned without preparing a transition")

        self.revision_attempt_index += 1
        self.last_submission = delta_h
        self._note_inclusion(attempt, delta_h)

        if pending.status != REFUSED:
            self.accepted_attempts.append(
                {"revision_attempt_index": attempt, "candidate_boundary_index": candidate,
                 "status": pending.status, "delta_h_sha256": sha256_of(delta_h),
                 "delta_h_bytes": len(delta_h.encode("utf-8")),
                 "delta_h_tokens": self.count_tokens(delta_h)})
            return handover

        # Refused. The record is read back strictly by the inherited process before it lands in
        # the pending path; here it is read back once more, from that path, and only then moved.
        record_path = self.attempts_dir / f"attempt_{attempt:03d}.json"
        self._file_into_attempts(pending.pending_path, record_path, attempt, candidate, delta_h)
        self.pending = None                                   # nothing prepared, nothing to commit

        retained = RetainedSlice(
            slice_id=f"slice_{attempt:03d}", origin_revision_attempt=attempt,
            sha256=sha256_of(delta_h), bytes=len(delta_h.encode("utf-8")),
            tokens=self.count_tokens(delta_h),
            # A slice is present in the submission it came from, at offset zero. Recording that
            # here keeps the inclusion list complete from its first entry, so the ledger can
            # require that a slice first appears in the attempt that produced it.
            included_in_revision_attempts=[attempt], included_at_offset=[0])
        self.retained.append(retained)
        self.last_refused_slice = delta_h
        self.refused_attempts.append(
            {"revision_attempt_index": attempt, "candidate_boundary_index": candidate,
             "record_path": str(record_path), "delta_h_sha256": retained.sha256,
             "delta_h_bytes": retained.bytes, "delta_h_tokens": retained.tokens})

        raise RevisionRefused(
            revision_attempt_index=attempt, candidate_boundary_index=candidate,
            record_path=record_path, delta_h_sha256=retained.sha256,
            delta_h_bytes=retained.bytes, record_status=REFUSED)

    def _file_into_attempts(self, pending_path: Path, record_path: Path, attempt: int,
                            candidate: int, delta_h: str) -> None:
        """Read the pending record back, then move it where a refusal belongs.

        Into `attempts/`, never `boundaries/`: a refused revision is not a boundary, and a reader
        that had to look inside a file to find out would eventually not.
        """
        if record_path.exists():
            raise OnlineIntegrationError(f"{record_path} already exists and records are never "
                                         "replaced")
        try:
            payload = json.loads(pending_path.read_text(encoding="utf-8"))
            RevisionRecord.from_dict(payload)
        except Exception as err:                                        # noqa: BLE001
            raise OnlineIntegrationError(
                f"the refused record at {pending_path.name} did not read back: {err}") from err
        os.replace(pending_path, record_path)
        sidecar = record_path.with_suffix(".identity.json")
        sidecar.write_text(json.dumps({
            "revision_attempt_index": attempt,
            "candidate_boundary_index": candidate,
            "committed": False,
            "reason_not_committed": "the revision was refused, so no transition committed and "
                                    "its source trajectory was retained",
            "delta_h_sha256": sha256_of(delta_h),
            "delta_h_bytes": len(delta_h.encode("utf-8")),
            "delta_h_tokens": self.count_tokens(delta_h),
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def _note_inclusion(self, attempt: int, delta_h: str) -> None:
        for retained in self.retained:
            if retained.absorbed_by_revision_attempt is not None:
                continue
            offset = delta_h.find(_slice_text(self, retained))
            if offset < 0:
                raise OnlineIntegrationError(
                    f"{retained.slice_id} is not present in revision attempt {attempt}; a "
                    "retained slice may not disappear before a transition consumes it")
            retained.included_in_revision_attempts.append(attempt)
            retained.included_at_offset.append(offset)

    # ------------------------------------------------------------------ commit
    def commit_pending(self) -> CommittedBoundary:
        """The inherited commit, plus marking whatever the committed transition just absorbed."""
        committed = super().commit_pending()
        for retained in self.retained:
            if retained.absorbed_by_revision_attempt is None:
                retained.absorbed_by_revision_attempt = self.revision_attempt_index - 1
                retained.absorbed_by_boundary = committed.boundary_index
                retained.absorbed_by_empty_revision = committed.status == EMPTY
        self.last_refused_slice = None
        return committed

    def close(self) -> None:
        """The task ended. Anything still retained is named, never quietly dropped."""
        for retained in self.retained:
            if retained.absorbed_by_revision_attempt is None:
                retained.unabsorbed_at_task_end = True

    # ------------------------------------------------------------------ reading it back
    @property
    def unabsorbed(self) -> list[RetainedSlice]:
        return [s for s in self.retained if s.absorbed_by_revision_attempt is None]

    def absorbed_by_empty(self) -> list[RetainedSlice]:
        """Slices an empty revision consumed.

        Recorded apart from an ordinary empty revision because it is a different event. An empty
        revision says the slice changed nothing the state must carry; when the slice includes
        material a previous revision was refused for, that claim is exactly what a reader should
        be suspicious of, and this diagnostic must not settle it by assumption.
        """
        return [s for s in self.retained if s.absorbed_by_empty_revision]

    def slice_ledger(self) -> dict:
        return {
            "method": METHOD,
            "slices": [s.to_dict() for s in self.retained],
            "revision_attempts": self.revision_attempt_index,
            "refused_revision_attempts": len(self.refused_attempts),
            "committed_boundaries": len(self.committed),
            "unabsorbed_slices": [s.slice_id for s in self.unabsorbed],
            "absorbed_by_empty_revision": [s.slice_id for s in self.absorbed_by_empty()],
        }

    def retention_budget(self) -> dict:
        """Deterministic sizes, from the same tokenizer the host's trigger uses.

        Provider usage is recorded elsewhere and is not used here: it is missing often enough that
        an accounting built on it would have holes exactly where a retained interval is largest.
        """
        unabsorbed = self.unabsorbed
        retained_text_tokens = sum(s.tokens for s in unabsorbed)
        return {
            "retained_history_bytes": sum(s.bytes for s in unabsorbed),
            "retained_history_tokens": retained_text_tokens,
            "first_refused_revision_attempt": (self.refused_attempts[0]["revision_attempt_index"]
                                               if self.refused_attempts else None),
            "updater_calls_including_refused_history": sum(
                len(s.included_in_revision_attempts) for s in self.retained),
            "unabsorbed_slice_count": len(unabsorbed),
        }


def _slice_text(optimizer: RetainingLocalRevisionOptimizer, retained: RetainedSlice) -> str:
    """The recorded slice, re-read from its own attempt record rather than kept in memory twice."""
    path = Path(optimizer.attempts_dir) / f"attempt_{retained.origin_revision_attempt:03d}.json"
    return json.loads(path.read_text(encoding="utf-8"))["delta_h"]


__all__ = ["METHOD", "RetainedSlice", "RetainingLocalRevisionOptimizer", "RevisionRefused",
           "check_interval", "sha256_of"]
