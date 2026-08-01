"""Loading the frozen inputs, and refusing to load anything else.

Every value the run depends on is checked against the committed manifest before a model is reached:
the episode files, the goals, the rules bytes, and all 32 slices by length and by hash. Lengths are
UTF-8 byte lengths, because six of the slices contain non-ASCII and a character count would pass while
the bytes differed.

What comes back carries its own identity. The run manifest takes the input hash, the episode order and
the boundary counts from this object rather than re-reading a file, so there is one verified path in
and no second reading to disagree with it.

Nothing here imports the discarded project or re-renders anything. The rules are bytes that were
frozen in a commit; this reads them and checks them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "inputs" / "preflight" / "manifest.json"


class InputError(ValueError):
    """A frozen input that is not what the manifest says it is."""


@dataclass(frozen=True)
class Episode:
    id: str
    goal: str
    rules: str
    boundaries: tuple[tuple[int, str], ...]        # (compaction_index, delta_h)


@dataclass(frozen=True)
class PreflightInputs:
    episodes: tuple[Episode, ...]
    input_manifest_sha256: str
    source_kind: str
    historical_byte_identity_verified: bool
    sampling_is_deterministic: bool

    @property
    def boundary_count(self) -> int:
        return sum(len(e.boundaries) for e in self.episodes)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_inputs_sha256(inputs: dict) -> str:
    """The hash the manifest records: canonical JSON of the inputs object, and nothing else."""
    canon = json.dumps(inputs, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
    return _sha(canon)


def _slices(episode: dict, where: str) -> list[tuple[int, str]]:
    found = [(e["compass"]["compaction_index"], e["compass"]["delta_h"])
             for e in episode.get("events") or []
             if e.get("compass") and e["compass"].get("delta_h")]
    found.sort(key=lambda pair: pair[0])
    for position, (index, _) in enumerate(found):
        if index != position:
            raise InputError(f"{where}: compaction indices are {[i for i, _ in found]}, "
                             "expected 0 upwards with no gap")
    return found


def load(source_dir: Path, manifest_path: Path = MANIFEST_PATH,
         repo_root: Path = REPO_ROOT) -> PreflightInputs:
    """Read and verify every frozen input. Any disagreement is an InputError naming it."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    recorded = manifest["input_manifest_sha256"]
    recomputed = canonical_inputs_sha256(manifest["inputs"])
    if recomputed != recorded:
        raise InputError(f"manifest: inputs hash to {recomputed}, recorded as {recorded}")

    episodes = []
    for entry in manifest["inputs"]["episodes"]:
        episodes.append(_episode(entry, Path(source_dir), Path(repo_root)))

    return PreflightInputs(
        episodes=tuple(episodes),
        input_manifest_sha256=recorded,
        source_kind=manifest["source_kind"],
        historical_byte_identity_verified=manifest["historical_byte_identity_verified"],
        sampling_is_deterministic=manifest["sampling_is_deterministic"],
    )


def _episode(entry: dict, source_dir: Path, repo_root: Path) -> Episode:
    name = entry["id"]
    path = source_dir / entry["episode_file"]
    if not path.exists():
        raise InputError(f"{name}: {path} is not there")
    raw = path.read_bytes()
    if _sha(raw) != entry["episode_file_sha256"]:
        raise InputError(f"{name}: {entry['episode_file']} hashes to {_sha(raw)}, "
                         f"recorded as {entry['episode_file_sha256']}")

    document = json.loads(raw)
    goal = document["instruction"]
    goal_bytes = goal.encode("utf-8")
    if len(goal_bytes) != entry["goal_bytes"]:
        raise InputError(f"{name}: goal is {len(goal_bytes)} bytes, recorded as "
                         f"{entry['goal_bytes']}")
    if _sha(goal_bytes) != entry["goal_sha256"]:
        raise InputError(f"{name}: goal hashes to {_sha(goal_bytes)}, "
                         f"recorded as {entry['goal_sha256']}")

    rules_path = repo_root / entry["rules_file"]
    if not rules_path.exists():
        raise InputError(f"{name}: {rules_path} is not there")
    rules_bytes = rules_path.read_bytes()
    if len(rules_bytes) != entry["rules_bytes"]:
        raise InputError(f"{name}: rules are {len(rules_bytes)} bytes, recorded as "
                         f"{entry['rules_bytes']}")
    if _sha(rules_bytes) != entry["rules_sha256"]:
        raise InputError(f"{name}: rules hash to {_sha(rules_bytes)}, "
                         f"recorded as {entry['rules_sha256']}")
    rules = rules_bytes.decode("utf-8")          # strict, and nothing is stripped or normalized

    found = _slices(document, name)
    rows = entry["boundaries"]
    if not (len(found) == len(rows) == entry["boundary_count"]):
        raise InputError(f"{name}: {len(found)} slices, {len(rows)} rows, "
                         f"boundary_count {entry['boundary_count']}")
    for (index, slice_text), row in zip(found, rows):
        if index != row["compaction_index"]:
            raise InputError(f"{name}: slice {index} where the manifest has "
                             f"{row['compaction_index']}")
        data = slice_text.encode("utf-8")
        if len(data) != row["delta_h_bytes"]:
            raise InputError(f"{name} #{index}: slice is {len(data)} bytes, recorded as "
                             f"{row['delta_h_bytes']}")
        if _sha(data) != row["delta_h_sha256"]:
            raise InputError(f"{name} #{index}: slice hashes to {_sha(data)}, "
                             f"recorded as {row['delta_h_sha256']}")

    return Episode(id=name, goal=goal, rules=rules, boundaries=tuple(found))
