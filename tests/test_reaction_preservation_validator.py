import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME
from t2pw.pipeline.reaction_preservation_validator import (
    validate_reaction_preservation,
    write_reaction_preservation_report_if_manifest,
)


def _manifest():
    return [
        {
            "locked_reaction_id": "rxn_lock_001",
            "source_reaction_id": "R1",
            "name": "A to B",
            "inputs": ["A"],
            "outputs": ["B"],
            "modifiers_or_enzymes": ["Enzyme 1"],
            "evidence_quote": "Enzyme 1 converts A to B.",
        },
        {
            "locked_reaction_id": "rxn_lock_002",
            "source_reaction_id": "R2",
            "name": "C to D",
            "inputs": ["C"],
            "outputs": ["D"],
            "modifiers_or_enzymes": [],
            "evidence_quote": "C becomes D.",
        },
    ]


def test_validate_reaction_preservation_marks_exact_matches():
    current = {
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "reaction_id": "reaction_1",
                    "source_reaction_id": "R1",
                    "name": "A to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": [{"entity": "Enzyme 1"}],
                    "evidence": "Enzyme 1 converts A to B.",
                }
            ]
        }
    }

    report = validate_reaction_preservation(current, [_manifest()[0]], stage="after_stage2")

    assert report["locked_reaction_count"] == 1
    assert report["preserved_count"] == 1
    assert report["missing_count"] == 0
    assert report["details"][0]["status"] == "preserved_exactly"
    assert report["details"][0]["matched_reaction_id"] == "reaction_1"


def test_validate_reaction_preservation_reports_missing_reaction():
    current = {"processes": {"reactions": []}}

    report = validate_reaction_preservation(current, _manifest(), stage="after_audit")

    assert report["locked_reaction_count"] == 2
    assert report["preserved_count"] == 0
    assert report["missing_count"] == 2
    assert {detail["status"] for detail in report["details"]} == {"missing"}


def test_validate_reaction_preservation_detects_name_and_io_changes():
    current = {
        "processes": {
            "reactions": [
                {
                    "reaction_id": "R1",
                    "name": "Renamed conversion",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": ["Enzyme 1"],
                    "evidence": "Enzyme 1 converts A to B.",
                },
                {
                    "reaction_id": "R2",
                    "name": "C to D",
                    "inputs": ["C", "ATP"],
                    "outputs": ["D"],
                    "evidence": "C becomes D.",
                },
            ]
        }
    }

    report = validate_reaction_preservation(current, _manifest(), stage="after_normalization")
    by_lock = {detail["locked_reaction_id"]: detail for detail in report["details"]}

    assert by_lock["rxn_lock_001"]["status"] == "preserved_with_name_changes"
    assert by_lock["rxn_lock_001"]["warnings"] == ["name_changed"]
    assert by_lock["rxn_lock_002"]["status"] == "preserved_with_input_output_changes"
    assert "inputs_outputs_changed" in by_lock["rxn_lock_002"]["warnings"]
    assert report["changed_count"] == 2


def test_validate_reaction_preservation_marks_merged_candidates():
    current = {
        "processes": {
            "reactions": [
                {
                    "reaction_id": "merged_reaction",
                    "name": "A to B and C to D",
                    "inputs": ["A", "C"],
                    "outputs": ["B", "D"],
                    "evidence": "Enzyme 1 converts A to B. C becomes D.",
                }
            ]
        }
    }

    report = validate_reaction_preservation(current, _manifest(), stage="after_audit")

    assert report["preserved_count"] == 2
    assert report["changed_count"] == 2
    assert {detail["status"] for detail in report["details"]} == {"merged_candidate"}


def test_validate_reaction_preservation_reads_locked_quarantine_records():
    current = {
        "processes": {"reactions": []},
        "quarantined_locked_reactions": [
            {
                "locked_reaction_id": "rxn_lock_001",
                "reaction_name": "A to B",
                "reason": "missing_outputs",
                "missing_entities": ["<missing outputs>"],
                "original_reaction": {
                    "locked_reaction_id": "rxn_lock_001",
                    "source_reaction_id": "R1",
                    "name": "A to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "evidence": "Enzyme 1 converts A to B.",
                },
            }
        ],
    }

    report = validate_reaction_preservation(current, [_manifest()[0]], stage="after_stage2")

    assert report["quarantined_count"] == 1
    assert report["details"][0]["status"] == "quarantined"
    assert report["details"][0]["matched_reaction_id"] == "R1"


def test_write_report_skips_absent_manifest_and_writes_expected_names(tmp_path):
    current = {
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "A to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                }
            ]
        }
    }

    assert write_reaction_preservation_report_if_manifest(current, tmp_path, stage="after_stage2") is None

    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps([_manifest()[0]]), encoding="utf-8")
    report = write_reaction_preservation_report_if_manifest(current, tmp_path, stage="after_stage2")

    assert report is not None
    expected = tmp_path / "reaction_preservation_report_after_stage2.json"
    alias = tmp_path / "reaction_preservation_after_stage2.json"
    assert expected.exists()
    assert alias.exists()
    assert json.loads(expected.read_text(encoding="utf-8"))["stage"] == "after_stage2"
