"""The practice set: eleven synthetic cases, answered under the frozen rubric, touching nothing real."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DIAGNOSTIC = REPO / "inputs" / "diagnostic"
CASES = json.loads((DIAGNOSTIC / "practice_cases.json").read_text(encoding="utf-8"))
KEY = json.loads((DIAGNOSTIC / "practice_answer_key.json").read_text(encoding="utf-8"))
EPISODES = ("042a9fc_3", "6b6ca61_2", "6f4b9a5_3", "83a7951_2", "9dabbc9_3")


def load_viewer():
    path = REPO / "scripts" / "annotate_packets.py"
    spec = importlib.util.spec_from_file_location("annotate_packets_practice", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = load_viewer()


def test_there_are_eleven_cases_with_unique_ids():
    ids = [c["practice_id"] for c in CASES["cases"]]
    assert len(ids) == 11 and len(set(ids)) == 11


def test_the_key_answers_every_case_and_nothing_else():
    assert [a["practice_id"] for a in KEY["answers"]] == [c["practice_id"]
                                                          for c in CASES["cases"]]


def test_every_case_has_the_shape_of_a_real_packet():
    for case in CASES["cases"]:
        assert set(case) == {"practice_id", "goal", "prefix"}
        assert case["goal"].strip()
        assert case["prefix"]
        for step in case["prefix"]:
            assert set(step) == {"reasoning", "code", "observation"}
            assert all(step[f].strip() for f in step)
    assert CASES["shared_rules"].strip()


@pytest.mark.parametrize("answer", KEY["answers"])
def test_every_answer_satisfies_the_frozen_schema(answer):
    """The same both-directions validation the viewer applies to a real label."""
    record = {"packet_id": "0" * 32,
              "event_class": answer["event_class"],
              "revision_subtypes": answer["revision_subtypes"],
              "terminal_stage": answer["terminal_stage"]}
    assert A.validate_record(record)


def test_the_answers_cover_every_class_and_every_subtype():
    classes = {a["event_class"] for a in KEY["answers"]}
    assert classes == set(A.CLASSES)
    subtypes = {s for a in KEY["answers"] for s in a["revision_subtypes"]}
    assert subtypes == set(A.SUBTYPES)
    stages = {a["terminal_stage"] for a in KEY["answers"] if a["terminal_stage"]}
    assert stages == set(A.STAGES)


def test_one_answer_carries_two_subtypes_because_they_may_be_held_together():
    assert any(len(a["revision_subtypes"]) > 1 for a in KEY["answers"])


def test_the_two_confusable_boundaries_each_appear_on_both_sides():
    """A token discovered against an abstract intention, and against a committed call."""
    by_id = {a["practice_id"]: a for a in KEY["answers"]}
    assert by_id["synthetic_04"]["event_class"] == "progressive_refinement"
    assert by_id["synthetic_05"]["event_class"] == "structural_revision"
    assert by_id["synthetic_05"]["revision_subtypes"] == ["new_prerequisite"]
    assert by_id["synthetic_09"]["terminal_stage"] == "ready"
    assert by_id["synthetic_10"]["terminal_stage"] == "confirmed"


def test_every_rationale_argues_from_the_rubric_and_is_not_a_label_restated():
    for answer in KEY["answers"]:
        assert len(answer["rationale"]) > 120
        assert answer["rationale"] != answer["event_class"]


def test_nothing_real_leaks_into_either_file():
    manifest = json.loads((DIAGNOSTIC / "manifest.json").read_text())
    packet_ids = {entry["packet_id"] for entry in manifest["packets"]}
    for path in ("practice_cases.json", "practice_answer_key.json"):
        blob = (DIAGNOSTIC / path).read_text(encoding="utf-8")
        for episode in EPISODES:
            assert episode not in blob
        assert not any(pid in blob for pid in packet_ids)
        for forbidden in ("KEY.json", "artifacts/", "ptc2d", "venmo", "splitwise", "spotify",
                          "simple_note", "appworld", "supervisor"):
            assert forbidden not in blob.lower(), forbidden


def test_the_practice_set_does_not_alter_the_frozen_material():
    """It explains the rubric; it does not extend it."""
    rubric = (DIAGNOSTIC / "rubric.md").read_text(encoding="utf-8")
    for answer in KEY["answers"]:
        assert answer["event_class"] in rubric
        for subtype in answer["revision_subtypes"]:
            assert subtype in rubric
        if answer["terminal_stage"]:
            assert answer["terminal_stage"] in rubric


def test_the_cases_are_not_shown_by_the_viewer_and_hold_no_packet_id():
    for case in CASES["cases"]:
        assert "packet_id" not in case
    assert "practice" not in (REPO / "scripts" / "annotate_packets.py").read_text()
