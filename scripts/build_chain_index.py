"""Turn the mapping into ordered chains of packet ids, and nothing more.

This is the only tool that reads the KEY. It writes an index whose entries say which packets follow
which, under an id derived from those packets alone, so the viewer can present a trajectory in order
without ever opening the mapping.

The order is not a secret the index gives away. The packets are cumulative prefixes and their sequence
can be recovered from them directly; the index stays outside the repository because it is derived from
the KEY and because keeping it out is what stops an implementation from quietly writing the canonical
episode and step next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "inputs" / "diagnostic" / "manifest.json"
FORBIDDEN_IN_OUTPUT = ("episode", "step", "task", "canonical")


class ChainError(RuntimeError):
    """A precondition of the index that does not hold. Nothing is written."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chain_id(packet_ids: list[str]) -> str:
    return sha("\n".join(packet_ids).encode("utf-8"))[:16]


def outside_repo(path: Path, what: str) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        raise ChainError(f"{what} would be written inside the repository at {resolved}")
    return resolved


def build_index(key_path: Path, manifest_path: Path = MANIFEST) -> dict:
    key_bytes = Path(key_path).read_bytes()
    mapping = json.loads(key_bytes)
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)

    known = {entry["packet_id"] for entry in manifest["packets"]}
    if set(mapping) != known:
        raise ChainError(f"the mapping covers {len(mapping)} packets and the manifest {len(known)}")

    groups: dict[str, list[tuple[int, str]]] = {}
    for packet_id, entry in mapping.items():
        groups.setdefault(entry["episode"], []).append((entry["step"], packet_id))

    chains: dict[str, list[str]] = {}
    for _, pairs in sorted(groups.items()):
        steps = sorted(pairs)
        if [s for s, _ in steps] != list(range(1, len(steps) + 1)):
            raise ChainError("a trajectory's steps are not 1..n with no gap")
        ordered = [packet_id for _, packet_id in steps]
        identifier = chain_id(ordered)
        if identifier in chains:
            raise ChainError(f"two chains hash to {identifier}")
        chains[identifier] = ordered

    covered = [p for ordered in chains.values() for p in ordered]
    if sorted(covered) != sorted(known):
        raise ChainError("the chains do not cover every packet exactly once")

    index = {"format_version": 1,
             "public_manifest_sha256": sha(manifest_bytes),
             "source_key_sha256": sha(key_bytes),
             "chains": chains}

    blob = json.dumps(index, ensure_ascii=False)
    for word in FORBIDDEN_IN_OUTPUT:
        if word in blob:
            raise ChainError(f"the index would carry the word {word!r}")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, type=Path, help="the mapping, outside the repo")
    parser.add_argument("--out", required=True, type=Path, help="where to write, outside the repo")
    args = parser.parse_args(argv)

    outside_repo(args.key, "the mapping")
    out = outside_repo(args.out, "the chain index")
    index = build_index(args.key)
    body = json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    out.write_text(body, encoding="utf-8")

    print(f"wrote {len(index['chains'])} chains covering "
          f"{sum(len(v) for v in index['chains'].values())} packets")
    for identifier, packets in sorted(index["chains"].items()):
        print(f"   {identifier}  {len(packets)} packets")
    print(f"index at {out}, sha256 {sha(body.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
