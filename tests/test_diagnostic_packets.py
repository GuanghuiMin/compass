"""The packet builder, on a fixture of its own, with no method output anywhere near it."""

import hashlib
import hmac
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_diagnostic_packets.py"
    spec = importlib.util.spec_from_file_location("build_diagnostic_packets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # a dataclass resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


B = load_builder()
REPO = Path(__file__).resolve().parents[1]


def make_source(tmp_path, counts=None):
    """Episode files shaped like the frozen ones, with the real per-episode step counts."""
    counts = counts or B.EXPECTED_COUNTS
    source = tmp_path / "source"
    source.mkdir()
    for episode, n in counts.items():
        events = [{"step": i, "reasoning": f"thinking at {episode} {i}",
                   "code": f"print({i})", "observation": f"observed {i}"}
                  for i in range(1, n + 1)]
        events.append({"compass": {"compaction_index": 0, "delta_h": "x"},
                       "compression_before_step": 3, "kind": "compaction",
                       "summary": "s", "summary_tokens": 1})
        (source / f"{episode}.json").write_text(json.dumps(
            {"task_id": episode, "instruction": f"the goal of {episode}", "events": events}),
            encoding="utf-8")
    return source


@pytest.fixture
def secret(tmp_path):
    path = tmp_path / "secret"
    path.write_bytes(b"a" * 32)
    return path


@pytest.fixture
def built(tmp_path, secret, monkeypatch):
    source = make_source(tmp_path)
    monkeypatch.setattr(B, "load_rules",
                        lambda: {e: f"the rules of {e}\n" for e in B.EPISODES})
    out = tmp_path / "diagnostic"
    manifest = B.build(source, secret, tmp_path / "KEY.json", out)
    return source, out, manifest, tmp_path / "KEY.json"


# --------------------------------------------------------------------------- counts

def test_one_packet_per_decision_point(built):
    _, out, manifest, _ = built
    assert manifest["packet_count"] == 197
    assert len(list((out / "packets").glob("*.json"))) == 197


def test_the_per_episode_counts_are_the_recorded_ones(built):
    _, _, _, key = built
    mapping = json.loads(key.read_text())
    from collections import Counter
    counts = Counter(v["episode"] for v in mapping.values())
    assert dict(counts) == {"042a9fc_3": 33, "6b6ca61_2": 50, "6f4b9a5_3": 50,
                            "83a7951_2": 50, "9dabbc9_3": 14}


def test_the_last_step_still_produces_a_packet(built):
    _, _, _, key = built
    mapping = json.loads(key.read_text())
    steps = {v["step"] for v in mapping.values() if v["episode"] == "9dabbc9_3"}
    assert steps == set(range(1, 15))


def test_a_source_with_the_wrong_step_count_is_refused(tmp_path, secret, monkeypatch):
    source = make_source(tmp_path, counts={**B.EXPECTED_COUNTS, "9dabbc9_3": 13})
    monkeypatch.setattr(B, "load_rules", lambda: {e: "r" for e in B.EPISODES})
    with pytest.raises(B.BuildError, match="13 steps, expected 14"):
        B.build(source, secret, tmp_path / "KEY.json", tmp_path / "out")


# --------------------------------------------------------------------------- contents

def test_a_packet_holds_the_three_codeact_fields_for_every_step(built):
    source, out, _, key = built
    mapping = json.loads(key.read_text())
    pid = next(k for k, v in mapping.items() if v["episode"] == "9dabbc9_3" and v["step"] == 5)
    packet = json.loads((out / "packets" / f"{pid}.json").read_text())
    assert len(packet["prefix"]) == 5
    for i, step in enumerate(packet["prefix"], 1):
        assert set(step) == {"reasoning", "code", "observation"}
        assert step["reasoning"] == f"thinking at 9dabbc9_3 {i}"
        assert step["code"] == f"print({i})"
        assert step["observation"] == f"observed {i}"


def test_the_current_observation_is_present_and_the_next_step_is_not(built):
    source, out, _, key = built
    mapping = json.loads(key.read_text())
    pid = next(k for k, v in mapping.items() if v["episode"] == "9dabbc9_3" and v["step"] == 5)
    packet = json.loads((out / "packets" / f"{pid}.json").read_text())
    assert packet["prefix"][-1]["observation"] == "observed 5"
    blob = json.dumps(packet, ensure_ascii=False)
    assert "observed 6" not in blob and "print(6)" not in blob
    assert "thinking at 9dabbc9_3 6" not in blob


def test_the_packet_has_exactly_the_four_fields(built):
    _, out, _, _ = built
    for path in (out / "packets").glob("*.json"):
        assert set(json.loads(path.read_text())) == {"packet_id", "goal", "rules", "prefix"}


def test_no_canonical_metadata_appears(built):
    _, out, _, key = built
    mapping = json.loads(key.read_text())
    for path in (out / "packets").glob("*.json"):
        blob = path.read_text()
        entry = mapping[path.stem]
        assert entry["canonical_id"] not in blob
        assert f'"{entry["episode"]}"' not in blob
        assert '"step"' not in blob and '"compaction_index"' not in blob


def test_goal_and_rules_are_carried_verbatim(built):
    _, out, _, key = built
    mapping = json.loads(key.read_text())
    for path in (out / "packets").glob("*.json"):
        packet = json.loads(path.read_text())
        episode = mapping[path.stem]["episode"]
        assert packet["goal"] == f"the goal of {episode}"
        assert packet["rules"] == f"the rules of {episode}\n"


def test_the_real_rules_are_checked_against_the_frozen_hashes():
    """Not the fixture: the committed rules must still hash to what the preflight froze."""
    rules = B.load_rules()
    manifest = json.loads(B.PREFLIGHT_MANIFEST.read_text())
    for entry in manifest["inputs"]["episodes"]:
        assert hashlib.sha256(rules[entry["id"]].encode()).hexdigest() == entry["rules_sha256"]


def test_altered_rules_are_refused(tmp_path, monkeypatch):
    original = B.PREFLIGHT_MANIFEST.read_text()
    manifest = json.loads(original)
    manifest["inputs"]["episodes"][0]["rules_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(B, "PREFLIGHT_MANIFEST", path)
    with pytest.raises(B.BuildError, match="rules hash to"):
        B.load_rules()


# --------------------------------------------------------------------------- blinding

def test_the_id_is_an_hmac_of_the_canonical_id_under_the_external_secret(tmp_path, secret):
    expected = hmac.new(b"a" * 32, b"9dabbc9_3:5", hashlib.sha256).hexdigest()[:32]
    assert B.packet_id(b"a" * 32, "9dabbc9_3:5") == expected
    assert len(expected) == 32


def test_a_different_secret_gives_different_ids():
    assert B.packet_id(b"a" * 32, "x:1") != B.packet_id(b"b" * 32, "x:1")


def test_a_plain_hash_of_the_canonical_id_is_not_the_id():
    """Five episodes and 197 positions are enumerable, so an unkeyed digest blinds nothing."""
    plain = hashlib.sha256(b"9dabbc9_3:5").hexdigest()[:32]
    assert B.packet_id(b"a" * 32, "9dabbc9_3:5") != plain


def test_ids_are_unique_across_all_packets(built):
    _, out, manifest, _ = built
    ids = [p["packet_id"] for p in manifest["packets"]]
    assert len(set(ids)) == len(ids) == 197


@pytest.mark.parametrize("target", ["inputs/diagnostic/KEY.json", "KEY.json",
                                    "scripts/../KEY.json"])
def test_writing_the_mapping_inside_the_repository_is_refused(tmp_path, secret, monkeypatch,
                                                              target):
    source = make_source(tmp_path)
    monkeypatch.setattr(B, "load_rules", lambda: {e: "r" for e in B.EPISODES})
    with pytest.raises(B.BuildError, match="inside the repository"):
        B.build(source, secret, REPO / target, tmp_path / "out")


def test_a_secret_inside_the_repository_is_refused(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    monkeypatch.setattr(B, "load_rules", lambda: {e: "r" for e in B.EPISODES})
    with pytest.raises(B.BuildError, match="inside the repository"):
        B.build(source, REPO / "secret", tmp_path / "KEY.json", tmp_path / "out")


def test_a_short_secret_is_refused(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    weak = tmp_path / "weak"
    weak.write_bytes(b"short")
    monkeypatch.setattr(B, "load_rules", lambda: {e: "r" for e in B.EPISODES})
    with pytest.raises(B.BuildError, match="shorter than 16 bytes"):
        B.build(source, weak, tmp_path / "KEY.json", tmp_path / "out")


def test_the_manifest_carries_no_mapping(built):
    _, out, manifest, _ = built
    blob = json.dumps(manifest, ensure_ascii=False)
    for episode in B.EPISODES:
        assert f'"{episode}:' not in blob
    assert "canonical_id" not in {p for entry in manifest["packets"] for p in entry}
    assert manifest["blinding"]["key_sha256"]


def test_every_packet_hash_in_the_manifest_matches_the_file(built):
    _, out, manifest, _ = built
    for entry in manifest["packets"]:
        body = (out / "packets" / f"{entry['packet_id']}.json").read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert len(body) == entry["bytes"]


# --------------------------------------------------------------------------- what it does not do

def test_no_event_class_or_label_is_written(built):
    _, out, manifest, _ = built
    for path in (out / "packets").glob("*.json"):
        packet = json.loads(path.read_text())
        assert "event_class" not in packet and "label" not in packet
    assert manifest["control_selection"]["state"].startswith("algorithm frozen")


def test_the_builder_never_reads_a_method_artifact():
    source = (Path(__file__).resolve().parents[1] / "scripts"
              / "build_diagnostic_packets.py").read_text()
    for forbidden in ("artifacts/preflight", "artifacts/repairability", "ptc2d",
                      "resulting_snapshot", "handover"):
        assert forbidden not in source


def test_the_phase_rule_is_the_frozen_one():
    point = B.DecisionPoint(episode="e", step=1, episode_length=30, goal="g", rules="r", prefix=[])
    assert point.phase == 0
    assert B.DecisionPoint("e", 11, 30, "g", "r", []).phase == 1
    assert B.DecisionPoint("e", 21, 30, "g", "r", []).phase == 2
    assert B.DecisionPoint("e", 30, 30, "g", "r", []).phase == 2
