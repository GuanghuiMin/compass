"""The viewer: what it shows, what it refuses, and what it will not let an annotator take back."""

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


A = load("annotate_packets", "annotate_packets.py")
REPO = Path(__file__).resolve().parents[1]


def record(packet_id, event_class="ordinary_progress", subtypes=None, stage=None):
    return {"packet_id": packet_id, "event_class": event_class,
            "revision_subtypes": subtypes or [], "terminal_stage": stage}


# --------------------------------------------------------------------------- display

def test_control_characters_are_escaped_and_never_sent():
    shown = A.escape_exact("before\x1b[2Jafter\x07")
    assert "\x1b" not in shown and "\x07" not in shown
    assert "\\u001b" in shown and "\\u0007" in shown


def test_the_escape_is_reversible_because_a_backslash_is_escaped_first():
    text = "a literal \\u001b written by the agent"
    shown = A.escape_exact(text)
    assert shown == "a literal \\\\u001b written by the agent"
    assert shown.replace("\\\\", "\\") == text


def test_tabs_and_returns_are_escaped_and_newlines_stay_readable():
    shown = A.escape_exact("one\ttwo\rthree\nfour")
    assert "\\t" in shown and "\\r" in shown
    assert "\\n\n" in shown          # written as an escape and kept as a break


def test_nothing_is_summarized_or_truncated():
    long = "x" * 5000 + "\u00e9"
    assert A.escape_exact(long) == long
    step = {"reasoning": "r" * 100, "code": "c" * 100, "observation": "o" * 100}
    shown = A.render_step(step)
    assert "r" * 100 in shown and "c" * 100 in shown and "o" * 100 in shown


# --------------------------------------------------------------------------- validation

def test_a_well_formed_record_passes():
    assert A.validate_record(record("a" * 32))
    assert A.validate_record(record("a" * 32, "structural_revision", ["new_prerequisite"]))
    assert A.validate_record(record("a" * 32, "terminal_transition", stage="ready"))


def test_a_structural_revision_without_a_subtype_is_refused():
    with pytest.raises(A.SessionError, match="which kind"):
        A.validate_record(record("a" * 32, "structural_revision"))


def test_a_terminal_transition_without_a_stage_is_refused():
    with pytest.raises(A.SessionError, match="ready or confirmed"):
        A.validate_record(record("a" * 32, "terminal_transition"))


def test_subtypes_on_another_class_are_refused():
    with pytest.raises(A.SessionError, match="carries revision subtypes"):
        A.validate_record(record("a" * 32, "ordinary_progress", ["new_prerequisite"]))
    with pytest.raises(A.SessionError, match="carries revision subtypes"):
        A.validate_record(record("a" * 32, "progressive_refinement", ["new_prerequisite"]))


def test_a_stage_on_another_class_is_refused():
    with pytest.raises(A.SessionError, match="carries a terminal stage"):
        A.validate_record(record("a" * 32, "ordinary_progress", stage="ready"))


def test_an_unknown_class_or_subtype_or_stage_is_refused():
    with pytest.raises(A.SessionError, match="is not a class"):
        A.validate_record(record("a" * 32, "replan"))
    with pytest.raises(A.SessionError, match="revision_subtypes"):
        A.validate_record(record("a" * 32, "structural_revision", ["invented"]))
    with pytest.raises(A.SessionError, match="ready or confirmed"):
        A.validate_record(record("a" * 32, "terminal_transition", stage="done"))


def test_a_repeated_subtype_is_refused():
    with pytest.raises(A.SessionError, match="repeated"):
        A.validate_record(record("a" * 32, "structural_revision",
                                 ["new_prerequisite", "new_prerequisite"]))


def test_several_subtypes_are_allowed_together():
    assert A.validate_record(record("a" * 32, "structural_revision",
                                    ["new_prerequisite", "path_or_branch_invalidated"]))


def test_an_extra_field_is_refused():
    bad = record("a" * 32)
    bad["note"] = "anything"
    with pytest.raises(A.SessionError, match="has fields"):
        A.validate_record(bad)


# --------------------------------------------------------------------------- adjacency

def packet(pid, goal, rules, steps):
    return {"packet_id": pid, "goal": goal, "rules": rules,
            "prefix": [{"reasoning": f"r{i}", "code": f"c{i}", "observation": f"o{i}"}
                       for i in range(1, steps + 1)]}


def chain_of(n, goal="g", rules="r", prefix_id="p"):
    packets = {f"{prefix_id}{i}": packet(f"{prefix_id}{i}", goal, rules, i)
               for i in range(1, n + 1)}
    return {f"chain_{prefix_id}": [f"{prefix_id}{i}" for i in range(1, n + 1)]}, packets


def test_a_sound_chain_passes_adjacency():
    chains, packets = chain_of(4)
    A.check_adjacency(chains, packets)


def test_a_chain_that_skips_a_step_is_refused():
    chains, packets = chain_of(3)
    packets["p3"]["prefix"].append({"reasoning": "r4", "code": "c4", "observation": "o4"})
    with pytest.raises(A.SessionError, match="exactly one step"):
        A.check_adjacency(chains, packets)


def test_a_chain_whose_history_changes_is_refused():
    chains, packets = chain_of(3)
    packets["p3"]["prefix"][0]["code"] = "rewritten"
    with pytest.raises(A.SessionError, match="alters step 0"):
        A.check_adjacency(chains, packets)


def test_a_chain_whose_goal_changes_is_refused():
    chains, packets = chain_of(3)
    packets["p2"]["goal"] = "different"
    with pytest.raises(A.SessionError, match="goal or rules change"):
        A.check_adjacency(chains, packets)


def test_a_packet_left_out_of_every_chain_is_refused():
    chains, packets = chain_of(3)
    packets["orphan"] = packet("orphan", "g", "r", 1)
    with pytest.raises(A.SessionError, match="exactly once"):
        A.check_adjacency(chains, packets)


# --------------------------------------------------------------------------- append only

def session(tmp_path, done=None):
    chains, packets = chain_of(3)
    header = {"rubric_sha256": "a", "annotation_schema_sha256": "b",
              "public_manifest_sha256": "c", "chain_index_sha256": "d",
              "viewer_commit": "e", "annotator_id": "annotator_A",
              "independence_attestation": A.ATTESTATION, "attested_at": "t", "started_at": "t"}
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return A.Session(chains=chains, packets=packets, header=header, labels_path=labels,
                     done=list(done or []))


def test_a_label_is_written_once_and_then_read_only(tmp_path):
    s = session(tmp_path)
    A.append_label(s, record("p1"))
    with pytest.raises(A.SessionError, match="already has a primary label"):
        A.append_label(s, record("p1", "structural_revision", ["new_prerequisite"]))
    lines = s.labels_path.read_text().splitlines()
    assert len(lines) == 2                      # header and one record


def test_an_invalid_record_never_reaches_the_file(tmp_path):
    s = session(tmp_path)
    with pytest.raises(A.SessionError):
        A.append_label(s, record("p1", "structural_revision"))
    assert len(s.labels_path.read_text().splitlines()) == 1


def test_remaining_follows_chain_order(tmp_path):
    s = session(tmp_path, done=["p1"])
    assert [p for _, p in s.remaining()] == ["p2", "p3"]


# --------------------------------------------------------------------------- resume

def write_labels(tmp_path, header, records):
    labels = tmp_path / "labels.jsonl"
    body = [json.dumps(header)] + [json.dumps(r) for r in records]
    labels.write_text("\n".join(body) + "\n", encoding="utf-8")
    return labels


def header_for(**overrides):
    base = {"rubric_sha256": "a", "annotation_schema_sha256": "b",
            "public_manifest_sha256": "c", "chain_index_sha256": "d", "viewer_commit": "e",
            "annotator_id": "annotator_A", "independence_attestation": A.ATTESTATION,
            "attested_at": "t", "started_at": "t"}
    base.update(overrides)
    return base


def test_resuming_returns_what_was_already_labelled(tmp_path):
    chains, _ = chain_of(3)
    labels = write_labels(tmp_path, header_for(), [record("p1"), record("p2")])
    assert A.resume(labels, header_for(), chains) == ["p1", "p2"]


def test_resuming_refuses_labels_written_against_a_different_rubric(tmp_path):
    chains, _ = chain_of(3)
    labels = write_labels(tmp_path, header_for(rubric_sha256="old"), [record("p1")])
    with pytest.raises(A.SessionError, match="different rubric_sha256"):
        A.resume(labels, header_for(), chains)


def test_resuming_refuses_labels_that_are_not_a_prefix_of_the_walk(tmp_path):
    chains, _ = chain_of(3)
    labels = write_labels(tmp_path, header_for(), [record("p2")])
    with pytest.raises(A.SessionError, match="not a prefix"):
        A.resume(labels, header_for(), chains)


def test_resuming_refuses_a_duplicate(tmp_path):
    chains, _ = chain_of(3)
    labels = write_labels(tmp_path, header_for(), [record("p1"), record("p1")])
    with pytest.raises(A.SessionError, match="labelled twice"):
        A.resume(labels, header_for(), chains)


def test_resuming_refuses_a_record_that_breaks_the_schema(tmp_path):
    chains, _ = chain_of(3)
    labels = write_labels(tmp_path, header_for(),
                          [{"packet_id": "p1", "event_class": "structural_revision",
                            "revision_subtypes": [], "terminal_stage": None}])
    with pytest.raises(A.SessionError, match="which kind"):
        A.resume(labels, header_for(), chains)


# --------------------------------------------------------------------------- completion

def test_the_final_hash_is_written_beside_the_file_not_inside_it(tmp_path):
    s = session(tmp_path)
    A.append_label(s, record("p1"))
    A.append_label(s, record("p2"))
    sidecar = A.finalize(s.labels_path)
    assert sidecar["label_count"] == 2
    assert sidecar["sha256"] == hashlib.sha256(s.labels_path.read_bytes()).hexdigest()
    written = json.loads((tmp_path / "labels.jsonl.sha256").read_text())
    assert written == sidecar
    assert sidecar["sha256"] not in s.labels_path.read_text()


def test_a_later_edit_makes_the_sidecar_disagree(tmp_path):
    s = session(tmp_path)
    A.append_label(s, record("p1"))
    sidecar = A.finalize(s.labels_path)
    with s.labels_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record("p2")) + "\n")
    assert hashlib.sha256(s.labels_path.read_bytes()).hexdigest() != sidecar["sha256"]


def test_a_closed_annotation_cannot_be_reopened(tmp_path, monkeypatch):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    (tmp_path / "labels.jsonl.sha256").write_text("{}", encoding="utf-8")
    with pytest.raises(A.SessionError, match="closed"):
        A.start(tmp_path / "chain.json", labels, "annotator_A")


# --------------------------------------------------------------------------- what it may touch

@pytest.mark.parametrize("target", ["labels.jsonl", "inputs/diagnostic/labels.jsonl"])
def test_writing_labels_inside_the_repository_is_refused(target):
    with pytest.raises(A.SessionError, match="inside the repository"):
        A.outside_repo(REPO / target, "the labels file")


def test_the_viewer_never_opens_the_key_or_a_method_artifact():
    """Path-shaped, not word-shaped: these words occur legitimately in the prose that explains
    what the viewer does not read."""
    source = (REPO / "scripts" / "annotate_packets.py").read_text()
    for forbidden in ("KEY.json", "artifacts/", "trajectories/", "_ext", "ptc2d",
                      "resulting_snapshot", "canonical_id", "episode"):
        assert forbidden not in source, forbidden


def test_the_viewer_reads_the_sealed_rubric_and_schema():
    assert A.RUBRIC.name == "rubric.md" and A.SCHEMA.name == "annotation_schema.json"
    assert A.CLASSES == ("ordinary_progress", "progressive_refinement", "structural_revision",
                         "terminal_transition", "indeterminate")
    assert A.SUBTYPES == ("new_prerequisite", "path_or_branch_invalidated",
                          "goal_or_constraint_revised")
    assert A.STAGES == ("ready", "confirmed")
