"""Tests for focused repair passes (src/t2pw/pipeline/focused_repair.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.focused_repair import (
    ALL_PASS_NAMES,
    run_focused_repair_passes,
    _run_missing_entity_repair,
    _collect_declared_entity_names,
    _normalize_name,
    _STANDARD_COFACTORS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_pathway() -> Dict[str, Any]:
    """A minimal pathway with one locked reaction that references undeclared entities."""
    return {
        "entities": {
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "proteins": [{"name": "Hexokinase"}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose", "ATP"],
                    "outputs": ["Glucose-6-phosphate", "ADP"],
                    "enzymes": [{"protein": "Hexokinase"}],
                    "evidence": "Hexokinase catalyses the phosphorylation of glucose to glucose-6-phosphate.",
                },
                {
                    "locked_reaction_id": "rxn_lock_002",
                    "name": "Fructose-6-phosphate isomerisation",
                    "inputs": ["Glucose-6-phosphate"],
                    "outputs": ["Fructose-6-phosphate"],
                    "evidence": "Phosphoglucose isomerase converts G6P to F6P.",
                },
            ],
        },
        "biological_states": [],
    }


def _sample_manifest() -> List[Dict[str, Any]]:
    return [
        {
            "locked_reaction_id": "rxn_lock_001",
            "source_reaction_id": "r1",
            "name": "Glucose phosphorylation",
            "inputs": ["Glucose", "ATP"],
            "outputs": ["Glucose-6-phosphate", "ADP"],
            "modifiers_or_enzymes": ["Hexokinase"],
            "evidence_quote": "Hexokinase catalyses the phosphorylation of glucose.",
            "locked": True,
            "status": "valid",
        },
        {
            "locked_reaction_id": "rxn_lock_002",
            "source_reaction_id": "r2",
            "name": "Fructose-6-phosphate isomerisation",
            "inputs": ["Glucose-6-phosphate"],
            "outputs": ["Fructose-6-phosphate"],
            "modifiers_or_enzymes": [],
            "evidence_quote": "Phosphoglucose isomerase converts G6P to F6P.",
            "locked": True,
            "status": "valid",
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Missing entity repair detects undeclared reaction inputs
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_entity_repair_detects_undeclared_inputs() -> None:
    """Pass 1 should detect Glucose, Glucose-6-phosphate, Fructose-6-phosphate
    as missing from entities since only ATP, ADP, and Hexokinase are declared."""
    pathway = _sample_pathway()
    manifest = _sample_manifest()

    # Mock the LLM call so we don't need a real API
    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        # Return a valid JSON response for ambiguous entities
        mock_llm.return_value = json.dumps({
            "patches": [
                {
                    "op": "add",
                    "path": "/entities/compounds/-",
                    "value": {"name": "Glucose"},
                    "reason": "Glucose is a standard metabolite compound",
                    "confidence": 0.95,
                }
            ],
            "warnings": [],
        })

        result = _run_missing_entity_repair(
            pathway, manifest,
            source_text="",
            temperature=0.0,
            max_tokens=4000,
        )

    assert result["pass"] == "missing_entity_repair"
    # Should find missing entities (Glucose, Glucose-6-phosphate, Fructose-6-phosphate)
    assert result["stats"]["missing_found"] >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: All passes skip when no locked manifest is provided
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_passes_skip_when_no_locked_manifest() -> None:
    """When locked_manifest is empty, all passes should be skipped."""
    pathway = _sample_pathway()

    repaired, report = run_focused_repair_passes(
        pathway,
        [],  # empty manifest
        source_text="some text",
    )

    assert report["passes_run"] == []
    assert report["total_patches_proposed"] == 0
    assert report["total_patches_applied"] == 0
    assert report["total_patches_rejected"] == 0
    assert report["skipped_reason"] == "no_locked_manifest"
    # Pathway should be unchanged (deep copy)
    assert repaired == pathway


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Generated patches have the correct format
# ═══════════════════════════════════════════════════════════════════════════════

def test_patch_format_correctness() -> None:
    """Patches generated by Pass 1 deterministic mode should have op, path,
    value, reason, and confidence fields."""
    pathway = {
        "entities": {
            "compounds": [{"name": "ATP"}],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "test reaction",
                    "inputs": ["NADPH", "ATP"],
                    "outputs": ["NADP+"],
                    "evidence": "NADPH is oxidized.",
                }
            ],
        },
    }
    manifest = [
        {
            "locked_reaction_id": "rxn_lock_001",
            "name": "test reaction",
            "inputs": ["NADPH", "ATP"],
            "outputs": ["NADP+"],
            "locked": True,
            "status": "valid",
        }
    ]

    # NADPH and NADP+ are standard cofactors, so they should be added
    # deterministically without LLM call
    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        mock_llm.return_value = json.dumps({"patches": [], "warnings": []})

        result = _run_missing_entity_repair(
            pathway, manifest,
            source_text="",
        )

    # Check that deterministic patches have the required fields
    for p in result["patches"]:
        assert "op" in p, f"Patch missing 'op': {p}"
        assert "path" in p, f"Patch missing 'path': {p}"
        assert "value" in p, f"Patch missing 'value': {p}"
        assert "reason" in p, f"Patch missing 'reason': {p}"
        assert "confidence" in p, f"Patch missing 'confidence': {p}"
        assert p["op"] == "add"
        assert p["path"].startswith("/entities/")
        assert isinstance(p["confidence"], (int, float))
        assert 0.0 <= p["confidence"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Pass 1 deterministic mode adds simple compounds without LLM call
# ═══════════════════════════════════════════════════════════════════════════════

def test_pass1_deterministic_adds_cofactors_without_llm() -> None:
    """Standard cofactors like NADPH should be added deterministically without
    calling the LLM at all."""
    pathway = {
        "entities": {
            "compounds": [],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "reduction",
                    "inputs": ["NADPH"],
                    "outputs": ["NADP+"],
                    "evidence": "NADPH-dependent reduction.",
                }
            ],
        },
    }
    manifest = [
        {
            "locked_reaction_id": "rxn_lock_001",
            "name": "reduction",
            "inputs": ["NADPH"],
            "outputs": ["NADP+"],
            "locked": True,
            "status": "valid",
        }
    ]

    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        mock_llm.return_value = json.dumps({"patches": [], "warnings": []})

        result = _run_missing_entity_repair(
            pathway, manifest,
            source_text="",
        )

    # NADPH is a known cofactor, so deterministic patches should be generated
    assert result["stats"]["deterministic"] >= 1
    # Check that at least one deterministic patch is for NADPH
    deterministic_names = [
        p["value"]["name"]
        for p in result["patches"]
        if p.get("value", {}).get("name", "").upper() in ("NADPH", "NADP+")
    ]
    assert len(deterministic_names) >= 1, "Expected deterministic patch for NADPH"

    # LLM may still be called for non-cofactor ambiguous names, but NADPH should
    # NOT be in the ambiguous set
    assert result["stats"]["ambiguous"] == 0, (
        "NADPH should be handled deterministically, not sent to LLM"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Overall report format is correct
# ═══════════════════════════════════════════════════════════════════════════════

def test_overall_report_format() -> None:
    """The repair report should have passes_run, total_patches_*, and pass_details."""
    pathway = _sample_pathway()
    manifest = _sample_manifest()

    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        mock_llm.return_value = json.dumps({"patches": [], "warnings": []})

        _, report = run_focused_repair_passes(
            pathway,
            manifest,
            source_text="",
            enabled_passes=["missing_entity_repair"],
        )

    # Top-level keys
    assert "passes_run" in report
    assert "total_patches_proposed" in report
    assert "total_patches_applied" in report
    assert "total_patches_rejected" in report
    assert "pass_details" in report

    assert isinstance(report["passes_run"], list)
    assert isinstance(report["total_patches_proposed"], int)
    assert isinstance(report["total_patches_applied"], int)
    assert isinstance(report["total_patches_rejected"], int)
    assert isinstance(report["pass_details"], list)

    # passes_run should contain the pass we requested
    assert "missing_entity_repair" in report["passes_run"]

    # pass_details entries should have required fields
    for detail in report["pass_details"]:
        assert "pass" in detail
        assert "patches_proposed" in detail
        assert "patches_applied" in detail
        assert "patches_rejected" in detail
        assert "warnings" in detail
        assert "duration_seconds" in detail
        assert isinstance(detail["patches_proposed"], int)
        assert isinstance(detail["patches_applied"], int)
        assert isinstance(detail["patches_rejected"], int)
        assert isinstance(detail["warnings"], list)
        assert isinstance(detail["duration_seconds"], (int, float))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: enabled_passes parameter works correctly
# ═══════════════════════════════════════════════════════════════════════════════

def test_enabled_passes_filters_correctly() -> None:
    """Only passes listed in enabled_passes should run."""
    pathway = _sample_pathway()
    manifest = _sample_manifest()

    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        mock_llm.return_value = json.dumps({"patches": [], "warnings": []})

        _, report = run_focused_repair_passes(
            pathway,
            manifest,
            source_text="",
            enabled_passes=["missing_entity_repair", "alias_repair"],
        )

    # Only the two requested passes should have run
    assert set(report["passes_run"]).issubset({"missing_entity_repair", "alias_repair"})
    # Make sure no other passes appear in details
    for detail in report["pass_details"]:
        assert detail["pass"] in ("missing_entity_repair", "alias_repair")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: LLM failure is logged and next pass continues
# ═══════════════════════════════════════════════════════════════════════════════

def test_llm_failure_does_not_block_subsequent_passes() -> None:
    """If the LLM call fails for one pass, subsequent passes should still run."""
    pathway = _sample_pathway()
    manifest = _sample_manifest()

    call_count = {"n": 0}

    def mock_chat_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            raise RuntimeError("Simulated LLM failure")
        return json.dumps({"patches": [], "warnings": []})

    with patch("t2pw.pipeline.focused_repair._call_repair_llm", side_effect=mock_chat_side_effect):
        _, report = run_focused_repair_passes(
            pathway,
            manifest,
            source_text="",
            enabled_passes=["missing_entity_repair", "alias_repair"],
        )

    # Both passes should appear in details even though one may have had errors
    pass_names_in_details = [d["pass"] for d in report["pass_details"]]
    assert "missing_entity_repair" in pass_names_in_details
    assert "alias_repair" in pass_names_in_details


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Normalize name helper
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalize_name() -> None:
    assert _normalize_name("ATP") == "atp"
    assert _normalize_name("  glucose-6-phosphate  ") == "glucose6phosphate"
    assert _normalize_name("NAD+") == "nad"
    assert _normalize_name("Coenzyme A") == "coenzyme a"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: collect_declared_entity_names
# ═══════════════════════════════════════════════════════════════════════════════

def test_collect_declared_entity_names() -> None:
    pathway = _sample_pathway()
    declared = _collect_declared_entity_names(pathway)
    assert "atp" in declared["compounds"]
    assert "adp" in declared["compounds"]
    assert "hexokinase" in declared["proteins"]
    assert len(declared["protein_complexes"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: Pathway is unchanged when all patches are empty
# ═══════════════════════════════════════════════════════════════════════════════

def test_pathway_unchanged_when_no_patches() -> None:
    """When LLM returns no patches and nothing is missing deterministically,
    the pathway should be unchanged."""
    pathway = {
        "entities": {
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "proteins": [{"name": "Kinase"}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "locked_reaction_id": "rxn_lock_001",
                    "name": "ATP to ADP",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"protein": "Kinase"}],
                    "evidence": "Kinase phosphorylates.",
                    "biological_state": "cytoplasm",
                }
            ],
        },
        "biological_states": [{"name": "cytoplasm"}],
    }
    manifest = [
        {
            "locked_reaction_id": "rxn_lock_001",
            "name": "ATP to ADP",
            "inputs": ["ATP"],
            "outputs": ["ADP"],
            "modifiers_or_enzymes": ["Kinase"],
            "locked": True,
            "status": "valid",
        }
    ]

    import copy
    original = copy.deepcopy(pathway)

    with patch("t2pw.pipeline.focused_repair._call_repair_llm") as mock_llm:
        mock_llm.return_value = json.dumps({"patches": [], "warnings": []})
        repaired, report = run_focused_repair_passes(
            pathway, manifest, source_text="",
        )

    # entities and processes should be structurally equivalent
    assert repaired["entities"] == original["entities"]
    assert repaired["processes"] == original["processes"]
