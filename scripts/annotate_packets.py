"""Read 197 decision points as 197 new steps, and record one label for each.

The packets are cumulative, so reading them as files means reading the same history 197 times, up to
180 KB at a time. This walks a chain and shows only what the latest packet added, with the goal and the
rules once at the head and the whole frozen packet available on request.

It shows the stored strings and nothing of its own: no summary, no truncation, no normalization, no
rewording. Control characters are escaped reversibly rather than sent to the terminal, so an
observation cannot move the cursor or clear the screen; the escaped form is exact, not raw.

It never opens the mapping, the source trajectories, or any output of any method. It writes one label
when the annotator moves on, and that label is then read-only: a first pass that can be revised after
seeing what comes next is a first pass in name only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIAGNOSTIC = REPO / "inputs" / "diagnostic"
MANIFEST = DIAGNOSTIC / "manifest.json"
RUBRIC = DIAGNOSTIC / "rubric.md"
SCHEMA = DIAGNOSTIC / "annotation_schema.json"

CLASSES = ("ordinary_progress", "progressive_refinement", "structural_revision",
           "terminal_transition", "indeterminate")
SUBTYPES = ("new_prerequisite", "path_or_branch_invalidated", "goal_or_constraint_revised")
STAGES = ("ready", "confirmed")
ATTESTATION = ("I had not viewed the graph, plan, summary, audit, or method outputs for these "
               "trajectories before primary annotation.")


class SessionError(RuntimeError):
    """A precondition of the session that does not hold. Nothing is shown or written."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- display

def escape_exact(text: str) -> str:
    """Every character kept, with control characters written as escapes rather than sent.

    Reversible: a backslash is doubled first, so nothing in the original can be mistaken for an
    escape this added. An observation cannot clear the screen or move the cursor.
    """
    out = []
    for character in text:
        if character == "\\":
            out.append("\\\\")
        elif character == "\n":
            out.append("\\n\n")          # shown as an escape and as a real line break
        elif character == "\t":
            out.append("\\t")
        elif character == "\r":
            out.append("\\r")
        elif unicodedata.category(character) in ("Cc", "Cf"):
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    return "".join(out)


def render_step(step: dict) -> str:
    return "\n".join([
        "REASONING", escape_exact(step["reasoning"]), "",
        "CODE", escape_exact(step["code"]), "",
        "OBSERVATION", escape_exact(step["observation"]),
    ])


def render_head(packet: dict) -> str:
    return "\n".join(["GOAL", escape_exact(packet["goal"]), "",
                      "RULES", escape_exact(packet["rules"])])


# --------------------------------------------------------------------------- validation

def validate_record(record: dict) -> dict:
    """Both directions of both conditional fields, as the sealed schema states them."""
    if set(record) != {"packet_id", "event_class", "revision_subtypes", "terminal_stage"}:
        raise SessionError(f"a record has fields {sorted(record)}")
    if record["event_class"] not in CLASSES:
        raise SessionError(f"{record['event_class']!r} is not a class")

    subtypes = record["revision_subtypes"]
    if not isinstance(subtypes, list) or any(s not in SUBTYPES for s in subtypes):
        raise SessionError(f"revision_subtypes {subtypes!r}")
    if len(set(subtypes)) != len(subtypes):
        raise SessionError("a revision subtype is repeated")
    if record["event_class"] == "structural_revision":
        if not subtypes:
            raise SessionError("a structural revision must say which kind it was")
    elif subtypes:
        raise SessionError(f"{record['event_class']} carries revision subtypes")

    stage = record["terminal_stage"]
    if record["event_class"] == "terminal_transition":
        if stage not in STAGES:
            raise SessionError("a terminal transition must be ready or confirmed")
    elif stage is not None:
        raise SessionError(f"{record['event_class']} carries a terminal stage")
    return record


# --------------------------------------------------------------------------- the session

@dataclass(frozen=True)
class Session:
    chains: dict[str, list[str]]
    packets: dict[str, dict]
    header: dict
    labels_path: Path
    done: list[str]

    def remaining(self) -> list[tuple[str, str]]:
        """(chain id, packet id) still unlabelled, in chain order."""
        done = set(self.done)
        return [(c, p) for c, ordered in sorted(self.chains.items())
                for p in ordered if p not in done]


def check_tree_clean() -> str:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SessionError(f"git status failed: {result.stderr.strip()}")
    if result.stdout.strip():
        raise SessionError("the working tree has changes; a session must name the viewer it ran")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return head.stdout.strip()


def outside_repo(path: Path, what: str) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO):
        raise SessionError(f"{what} would sit inside the repository at {resolved}")
    return resolved


def load_packets(manifest: dict) -> dict[str, dict]:
    packets = {}
    for entry in manifest["packets"]:
        body = (DIAGNOSTIC / "packets" / f"{entry['packet_id']}.json").read_bytes()
        if sha(body) != entry["sha256"] or len(body) != entry["bytes"]:
            raise SessionError(f"{entry['packet_id']} does not match the manifest")
        packets[entry["packet_id"]] = json.loads(body)
    return packets


def check_adjacency(chains: dict[str, list[str]], packets: dict[str, dict]) -> None:
    covered = [p for ordered in chains.values() for p in ordered]
    if sorted(covered) != sorted(packets):
        raise SessionError("the chains do not cover every packet exactly once")
    for identifier, ordered in chains.items():
        for earlier, later in zip(ordered, ordered[1:]):
            a, b = packets[earlier], packets[later]
            if a["goal"] != b["goal"] or a["rules"] != b["rules"]:
                raise SessionError(f"{identifier}: goal or rules change inside a chain")
            if len(b["prefix"]) != len(a["prefix"]) + 1:
                raise SessionError(f"{identifier}: {later} does not add exactly one step")
            for i, step in enumerate(a["prefix"]):
                if b["prefix"][i] != step:
                    raise SessionError(f"{identifier}: {later} alters step {i}")


def start(chain_index_path: Path, labels_path: Path, annotator_id: str) -> Session:
    index_path = outside_repo(chain_index_path, "the chain index")
    labels = outside_repo(labels_path, "the labels file")
    sidecar = labels.with_name(labels.name + ".sha256")
    if sidecar.exists():
        raise SessionError(f"{sidecar} exists; this annotation is closed and cannot be extended")

    head = check_tree_clean()
    manifest_bytes = MANIFEST.read_bytes()
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if index.get("format_version") != 1:
        raise SessionError("the chain index is not format 1")
    if index["public_manifest_sha256"] != sha(manifest_bytes):
        raise SessionError("the chain index was built against a different manifest")

    manifest = json.loads(manifest_bytes)
    packets = load_packets(manifest)
    chains = index["chains"]
    check_adjacency(chains, packets)

    header = {"rubric_sha256": sha(RUBRIC.read_bytes()),
              "annotation_schema_sha256": sha(SCHEMA.read_bytes()),
              "public_manifest_sha256": sha(manifest_bytes),
              "chain_index_sha256": sha(index_bytes),
              "viewer_commit": head,
              "annotator_id": annotator_id,
              "independence_attestation": ATTESTATION,
              "attested_at": datetime.now(timezone.utc).isoformat(),
              "started_at": datetime.now(timezone.utc).isoformat()}

    done: list[str] = []
    if labels.exists():
        done = resume(labels, header, chains)
    else:
        labels.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
    return Session(chains=chains, packets=packets, header=header, labels_path=labels, done=done)


def resume(labels: Path, header: dict, chains: dict[str, list[str]]) -> list[str]:
    """An existing file may only be continued, and only if it is a prefix of this same walk."""
    lines = [line for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SessionError(f"{labels} exists but has no header")
    existing = json.loads(lines[0])
    for field in ("rubric_sha256", "annotation_schema_sha256", "public_manifest_sha256",
                  "chain_index_sha256", "viewer_commit"):
        if existing.get(field) != header[field]:
            raise SessionError(f"{labels} was written against a different {field}")
    # Whose file this is, checked rather than adopted. Taking the identifier from the file would
    # let a second person continue it and leave the whole thing looking like one annotator's work.
    if existing.get("annotator_id") != header["annotator_id"]:
        raise SessionError(f"{labels} belongs to {existing.get('annotator_id')!r}, "
                           f"not {header['annotator_id']!r}")
    if existing.get("independence_attestation") != ATTESTATION:
        raise SessionError(f"{labels} carries a different attestation")

    records = [validate_record(json.loads(line)) for line in lines[1:]]
    done = [r["packet_id"] for r in records]
    if len(set(done)) != len(done):
        raise SessionError("a packet is labelled twice")
    expected = [p for _, ordered in sorted(chains.items()) for p in ordered]
    if done != expected[:len(done)]:
        raise SessionError("the labels are not a prefix of the chain order")
    # The original times stand: the attestation was made once, and the session began once.
    header["started_at"] = existing["started_at"]
    header["attested_at"] = existing["attested_at"]
    return done


def append_label(session: Session, record: dict) -> None:
    validate_record(record)
    if record["packet_id"] in session.done:
        raise SessionError(f"{record['packet_id']} already has a primary label")
    with session.labels_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    session.done.append(record["packet_id"])


def finalize(labels_path: Path) -> dict:
    """The final hash cannot live in the file it hashes, so it goes beside it."""
    labels = Path(labels_path)
    body = labels.read_bytes()
    count = sum(1 for line in body.decode("utf-8").splitlines() if line.strip()) - 1
    sidecar = {"sha256": sha(body), "label_count": count,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    labels.with_name(labels.name + ".sha256").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return sidecar


# --------------------------------------------------------------------------- the loop

def ask(prompt: str, allowed: tuple[str, ...]) -> str:
    while True:
        answer = input(prompt).strip()
        if answer in allowed:
            return answer
        print(f"   one of: {', '.join(allowed)}")


def parse_subtypes(text: str) -> list[str]:
    """Every token must be a choice, or the whole input is refused.

    Dropping the tokens it does not recognise would turn "1,4" into "1" and write that into a label
    that cannot afterwards be corrected.
    """
    tokens = [t.strip() for t in text.split(",")]
    if not tokens or any(not t for t in tokens):
        raise SessionError("an empty choice")
    if any(t not in ("1", "2", "3") for t in tokens):
        raise SessionError(f"{[t for t in tokens if t not in ('1', '2', '3')]} is not a choice")
    if len(set(tokens)) != len(tokens):
        raise SessionError("a choice is repeated")
    return [SUBTYPES[int(t) - 1] for t in tokens]


def collect(packet_id: str) -> dict:
    print("\n  " + "  ".join(f"[{i}] {c}" for i, c in enumerate(CLASSES, 1)))
    event_class = CLASSES[int(ask("  class: ", tuple(str(i) for i in range(1, 6)))) - 1]
    subtypes: list[str] = []
    stage = None
    if event_class == "structural_revision":
        print("  " + "  ".join(f"[{i}] {s}" for i, s in enumerate(SUBTYPES, 1)))
        while True:
            try:
                subtypes = parse_subtypes(input("  subtypes, comma separated: "))
                break
            except SessionError as err:
                print(f"   {err}; enter one or more of 1, 2, 3")
    if event_class == "terminal_transition":
        stage = STAGES[int(ask("  stage [1] ready [2] confirmed: ", ("1", "2"))) - 1]
    return validate_record({"packet_id": packet_id, "event_class": event_class,
                            "revision_subtypes": subtypes, "terminal_stage": stage})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-index", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--annotator-id", required=True)
    args = parser.parse_args(argv)

    session = start(args.chain_index, args.labels, args.annotator_id)
    remaining = session.remaining()
    print(f"{len(session.packets)} packets in {len(session.chains)} chains, "
          f"{len(session.done)} already labelled, {len(remaining)} to go")

    current_chain = None
    for chain, packet_id in remaining:
        packet = session.packets[packet_id]
        if chain != current_chain:
            current_chain = chain
            print("\n" + "=" * 78 + f"\nchain {chain}\n" + "=" * 78)
            print(render_head(packet))
        print("\n" + "-" * 78 + f"\nstep {len(packet['prefix'])} of this chain\n" + "-" * 78)
        print(render_step(packet["prefix"][-1]))
        if input("\n  [enter] to label, or 'full' for the whole packet: ").strip() == "full":
            print(render_head(packet))
            for i, step in enumerate(packet["prefix"], 1):
                print(f"\n--- step {i} ---")
                print(render_step(step))
        append_label(session, collect(packet_id))

    if not session.remaining():
        sidecar = finalize(session.labels_path)
        print(f"\nall {sidecar['label_count']} labels written; "
              f"sha256 {sidecar['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
