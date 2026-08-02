"""Freeze one packet per decision point, with every method's output kept out of it.

A decision point is the moment after a step's observation and before the next action. Each packet
carries what was visible then and nothing that came later: the goal, the frozen rules, and the CodeAct
history through that observation, with reasoning, code and observation kept as three separate fields
because in this framework the step is all three and dropping the reasoning would remove what the agent
believed it was doing.

Packet ids are an HMAC over the canonical id with a secret kept outside the repository. A plain hash
of "episode:step" would be blinding in name only: five episodes and 197 positions can be enumerated in
a second. The mapping is written outside the repository too, and this refuses to write either the
secret or the mapping anywhere inside it.

These are method-output-blinded packets, not task-blinded ones. The goal, the rules and the trajectory
can still reveal what the task is; what they cannot reveal is what any method produced.

Nothing here assigns an event class or any other label. Whoever built these has read the graphs, the
plans and the audit, and could not annotate them independently.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT_MANIFEST = REPO / "inputs" / "preflight" / "manifest.json"
OUT_DIR = REPO / "inputs" / "diagnostic"
EPISODES = ["042a9fc_3", "6b6ca61_2", "6f4b9a5_3", "83a7951_2", "9dabbc9_3"]
EXPECTED_COUNTS = {"042a9fc_3": 33, "6b6ca61_2": 50, "6f4b9a5_3": 50, "83a7951_2": 50,
                   "9dabbc9_3": 14}
PACKET_KEYS = {"packet_id", "goal", "rules", "prefix"}
STEP_KEYS = {"reasoning", "code", "observation"}

# The commit whose tree holds the 197 packets. Recorded so a manifest can be traced to the corpus
# it describes without naming any local path.
PACKET_CORPUS_COMMIT = "d23271f02dc6060e8cd6195dc95f9f76e5e5f6d7"


class BuildError(RuntimeError):
    """A precondition of the build that does not hold. Nothing is written."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packet_id(secret: bytes, canonical_id: str) -> str:
    return hmac.new(secret, canonical_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


@dataclass(frozen=True)
class DecisionPoint:
    episode: str
    step: int                      # 1-based, the step whose observation has just arrived
    episode_length: int
    goal: str
    rules: str
    prefix: list[dict]

    @property
    def canonical_id(self) -> str:
        return f"{self.episode}:{self.step}"

    @property
    def phase(self) -> int:
        """early, middle or late third of this episode, frozen for the later control draw."""
        return min(2, 3 * (self.step - 1) // self.episode_length)


def load_rules() -> dict[str, str]:
    """The committed per-episode rules, checked against the hashes they were frozen with."""
    manifest = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))
    rules: dict[str, str] = {}
    for entry in manifest["inputs"]["episodes"]:
        data = (REPO / entry["rules_file"]).read_bytes()
        if sha(data) != entry["rules_sha256"]:
            raise BuildError(f"{entry['id']}: rules hash to {sha(data)}, "
                             f"frozen as {entry['rules_sha256']}")
        if len(data) != entry["rules_bytes"]:
            raise BuildError(f"{entry['id']}: rules are {len(data)} bytes, "
                             f"frozen as {entry['rules_bytes']}")
        rules[entry["id"]] = data.decode("utf-8")
    return rules


def enumerate_points(source_dir: Path, rules: dict[str, str]) -> list[DecisionPoint]:
    """One point after every recorded step's observation, in episode and step order."""
    points: list[DecisionPoint] = []
    for episode in EPISODES:
        document = json.loads((source_dir / f"{episode}.json").read_bytes())
        steps = [e for e in document["events"] if "code" in e]
        if len(steps) != EXPECTED_COUNTS[episode]:
            raise BuildError(f"{episode}: {len(steps)} steps, expected "
                             f"{EXPECTED_COUNTS[episode]}")
        for index, event in enumerate(steps):
            if set(event) != {"step", "reasoning", "code", "observation"}:
                raise BuildError(f"{episode} step {index + 1}: unexpected fields {sorted(event)}")
        for k in range(1, len(steps) + 1):
            prefix = [{"reasoning": s["reasoning"], "code": s["code"],
                       "observation": s["observation"]} for s in steps[:k]]
            points.append(DecisionPoint(episode=episode, step=k, episode_length=len(steps),
                                        goal=document["instruction"], rules=rules[episode],
                                        prefix=prefix))
    return points


def build_packet(point: DecisionPoint, secret: bytes) -> dict:
    return {"packet_id": packet_id(secret, point.canonical_id), "goal": point.goal,
            "rules": point.rules, "prefix": point.prefix}


def check_packet(packet: dict, point: DecisionPoint, source_steps: list[dict]) -> None:
    """Schema, prefix boundary and provenance. Not a word search: those words occur legitimately."""
    if set(packet) != PACKET_KEYS:
        raise BuildError(f"{point.canonical_id}: packet keys {sorted(packet)}")
    for i, step in enumerate(packet["prefix"]):
        if set(step) != STEP_KEYS:
            raise BuildError(f"{point.canonical_id}: step {i} keys {sorted(step)}")
    if len(packet["prefix"]) != point.step:
        raise BuildError(f"{point.canonical_id}: prefix holds {len(packet['prefix'])} steps")
    for i, (written, source) in enumerate(zip(packet["prefix"], source_steps[:point.step])):
        for field in STEP_KEYS:
            if written[field] != source[field]:
                raise BuildError(f"{point.canonical_id}: step {i} {field} was altered")
    # The boundary is guaranteed structurally, above: the prefix is exactly the first k source
    # steps, field by field, and holds k of them. Comparing values against the following step
    # would add nothing and would fire on legitimate repetition -- this corpus has steps whose
    # observation is identical to an earlier one, because the agent looked the same thing up twice.
    blob = json.dumps(packet, ensure_ascii=False)
    for forbidden in (point.canonical_id, f'"{point.episode}"', f'"step": {point.step}'):
        if forbidden in blob:
            raise BuildError(f"{point.canonical_id}: {forbidden!r} appears in the packet")


def outside_repo(path: Path, what: str) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        raise BuildError(f"{what} would be written inside the repository at {resolved}; "
                         "the secret and the mapping stay outside it until unblinding")
    return resolved


def build(source_dir: Path, secret_file: Path, key_out: Path, out_dir: Path = OUT_DIR) -> dict:
    secret_path = outside_repo(secret_file, "the blinding secret")
    key_path = outside_repo(key_out, "the mapping")
    secret = secret_path.read_bytes().strip()
    if len(secret) < 16:
        raise BuildError("the blinding secret is shorter than 16 bytes")

    rules = load_rules()
    points = enumerate_points(source_dir, rules)
    if len(points) != sum(EXPECTED_COUNTS.values()):
        raise BuildError(f"{len(points)} decision points, expected "
                         f"{sum(EXPECTED_COUNTS.values())}")

    source_steps = {e: [s for s in json.loads((source_dir / f"{e}.json").read_bytes())["events"]
                        if "code" in s] for e in EPISODES}

    packets_dir = out_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    entries, mapping, seen = [], {}, set()
    for point in points:
        packet = build_packet(point, secret)
        check_packet(packet, point, source_steps[point.episode])
        if packet["packet_id"] in seen:
            raise BuildError(f"packet id {packet['packet_id']} occurs twice")
        seen.add(packet["packet_id"])
        body = (json.dumps(packet, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (packets_dir / f"{packet['packet_id']}.json").write_bytes(body)
        entries.append({"packet_id": packet["packet_id"], "bytes": len(body),
                        "sha256": sha(body)})
        mapping[packet["packet_id"]] = {"canonical_id": point.canonical_id,
                                        "episode": point.episode, "step": point.step,
                                        "episode_length": point.episode_length,
                                        "phase": point.phase}

    key_body = (json.dumps(mapping, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    key_path.write_text(key_body, encoding="utf-8")

    preflight = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "packet_count": len(entries),
        "per_episode_counts": EXPECTED_COUNTS,
        "decision_point": ("after a recorded step's observation and before the next action; "
                           "these episodes contain no user-message events"),
        # Neither the secret nor the mapping is located here. This manifest travels with the packets
        # to whoever annotates them, and a manifest that says where the unblinding material lives is
        # not a blinded bundle.
        "blinding": {"scheme": "HMAC-SHA256(secret, canonical_id) truncated to 32 hex characters",
                     "canonical_id": "{episode}:{step}",
                     "key_sha256": sha(key_body.encode("utf-8")),
                     "note": ("method-output-blinded, not task-blinded: the goal, rules and "
                              "trajectory may still reveal the task")},
        "source": {"episode_file_sha256": {e["id"]: e["episode_file_sha256"]
                                           for e in preflight["inputs"]["episodes"]},
                   "note": ("identified by content hash; the trajectory source is not a git "
                            "repository and its local path is deliberately not recorded")},
        "packet_corpus_commit": PACKET_CORPUS_COMMIT,
        "builder_sha256": sha(Path(__file__).resolve().read_bytes()),
        "builder_note": ("the hash is of the builder in the commit carrying this manifest; the "
                         "packets were produced at packet_corpus_commit, by a builder differing "
                         "only in which fields this manifest holds"),
        "rules_sha256": {e["id"]: e["rules_sha256"] for e in preflight["inputs"]["episodes"]},
        "input_manifest_sha256": preflight["input_manifest_sha256"],
        "control_selection": {
            "positives": "every structural_revision decision point",
            "pool": "every ordinary_progress decision point",
            "phase": "min(2, 3 * (step - 1) // episode_length)",
            "order": "sha256 of the canonical id, within each episode-and-phase stratum",
            "draw": "round robin across strata until the count equals the positives",
            "state": "algorithm frozen here; nothing is drawn until labels exist and are unblinded"},
        "packets": entries,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--secret-file", required=True, type=Path,
                        help="a file outside the repository holding the blinding secret")
    parser.add_argument("--key-out", required=True, type=Path,
                        help="where to write the mapping, outside the repository")
    args = parser.parse_args(argv)
    manifest = build(args.source_dir, args.secret_file, args.key_out)
    sizes = sorted(p["bytes"] for p in manifest["packets"])
    print(f"wrote {manifest['packet_count']} packets, {sum(sizes)} bytes total")
    print(f"packet bytes: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    # Printed for the operator running the build, never written into the manifest.
    print(f"mapping at {args.key_out.resolve()}, sha256 {manifest['blinding']['key_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
