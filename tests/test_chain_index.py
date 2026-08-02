"""The chain index: ordered packet ids, and no trace of what they are."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load(name, filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = load("build_chain_index", "build_chain_index.py")
REPO = Path(__file__).resolve().parents[1]


def fixture(tmp_path, shape=(("e_one", 3), ("e_two", 2))):
    """A mapping and a manifest that agree, in the shape the real ones have."""
    mapping, packets = {}, []
    for episode, n in shape:
        for step in range(1, n + 1):
            pid = hashlib.sha256(f"{episode}:{step}".encode()).hexdigest()[:32]
            mapping[pid] = {"canonical_id": f"{episode}:{step}", "episode": episode,
                            "step": step, "episode_length": n, "phase": 0}
            packets.append({"packet_id": pid, "bytes": 1, "sha256": "0" * 64})
    key = tmp_path / "KEY.json"
    key.write_text(json.dumps(mapping), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"packets": packets}), encoding="utf-8")
    return key, manifest, mapping


def test_one_chain_per_trajectory_in_step_order(tmp_path):
    key, manifest, mapping = fixture(tmp_path)
    index = C.build_index(key, manifest)
    assert len(index["chains"]) == 2
    for ordered in index["chains"].values():
        steps = [mapping[p]["step"] for p in ordered]
        assert steps == sorted(steps) == list(range(1, len(steps) + 1))


def test_the_chain_id_is_derived_from_the_packets_alone(tmp_path):
    key, manifest, _ = fixture(tmp_path)
    index = C.build_index(key, manifest)
    for identifier, ordered in index["chains"].items():
        assert identifier == hashlib.sha256("\n".join(ordered).encode()).hexdigest()[:16]
        assert len(identifier) == 16


def test_the_index_carries_no_canonical_metadata(tmp_path):
    key, manifest, mapping = fixture(tmp_path)
    blob = json.dumps(C.build_index(key, manifest), ensure_ascii=False)
    for word in ("episode", "step", "task", "canonical"):
        assert word not in blob
    for entry in mapping.values():
        assert entry["canonical_id"] not in blob
        assert f'"{entry["episode"]}"' not in blob


def test_the_index_records_both_source_hashes(tmp_path):
    key, manifest, _ = fixture(tmp_path)
    index = C.build_index(key, manifest)
    assert index["public_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert index["source_key_sha256"] == hashlib.sha256(key.read_bytes()).hexdigest()
    assert index["format_version"] == 1


def test_every_packet_is_covered_exactly_once(tmp_path):
    key, manifest, mapping = fixture(tmp_path)
    index = C.build_index(key, manifest)
    covered = [p for ordered in index["chains"].values() for p in ordered]
    assert sorted(covered) == sorted(mapping)


def test_a_mapping_that_does_not_match_the_manifest_is_refused(tmp_path):
    key, manifest, _ = fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["packets"].append({"packet_id": "f" * 32, "bytes": 1, "sha256": "0" * 64})
    manifest.write_text(json.dumps(payload))
    with pytest.raises(C.ChainError, match="covers"):
        C.build_index(key, manifest)


def test_a_gap_in_the_steps_is_refused(tmp_path):
    key, manifest, mapping = fixture(tmp_path)
    broken = {k: dict(v) for k, v in mapping.items()}
    first = next(iter(broken))
    broken[first]["step"] = 9
    key.write_text(json.dumps(broken))
    with pytest.raises(C.ChainError, match="no gap"):
        C.build_index(key, manifest)


@pytest.mark.parametrize("target", ["chain.json", "inputs/diagnostic/chain.json"])
def test_writing_the_index_inside_the_repository_is_refused(target):
    with pytest.raises(C.ChainError, match="inside the repository"):
        C.outside_repo(REPO / target, "the chain index")


def test_reading_the_key_from_inside_the_repository_is_refused():
    with pytest.raises(C.ChainError, match="inside the repository"):
        C.outside_repo(REPO / "KEY.json", "the mapping")
