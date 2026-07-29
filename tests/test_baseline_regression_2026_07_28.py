"""Offline regression harness built from the 2026-07-28 batch runs.

Eight curated payload fragments in ``fixtures/baseline_2026_07_28/cases.json``
stand in for the failure classes that night produced, and
``baseline_metrics.json`` records the run's headline numbers. The fixtures are
hand-written rather than copied: one merged payload from that run is 306 KB in
the 21:22 leg and 4.7 MB in the 09:19 leg, and carries verbatim source passages.

Everything here is pure and offline -- no network, no database, no batch run.
See docs/regression_baseline_2026-07-28.md for what the numbers mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from t2pw.pipeline.entity_identity import (
    component_stoichiometry,
    has_protein_external_identity,
    is_pathbank_unknown_protein,
    protein_external_identity,
    protein_species_context,
)
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    run_strict_post_normalization_gates,
    validate_registry_references,
)
from t2pw.pipeline.reaction_lock_manifest import build_locked_reaction_manifest
from t2pw.rag.provenance import validate_provenance


FIXTURES = Path(__file__).parent / "fixtures" / "baseline_2026_07_28"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _cases() -> Dict[str, Dict[str, Any]]:
    return {case["id"]: case for case in _load("cases.json")["cases"]}


def _case(case_id: str) -> Dict[str, Any]:
    return _cases()[case_id]


def _gate_errors(payload: Dict[str, Any], **kwargs: Any) -> list:
    """Run the strict gate and return its errors, whether or not it raised."""

    try:
        result = run_strict_post_normalization_gates(payload, **kwargs)
    except GateValidationError as exc:
        return list(exc.details.get("errors") or [])
    return list((result or {}).get("errors") or [])


# ── fixture integrity ──────────────────────────────────────────────────────


def test_fixture_set_covers_every_baseline_failure_class() -> None:
    """The eight cases are the harness's contract; losing one loses a class."""

    assert set(_cases()) == {
        "valid_mapped_enzyme",
        "ambiguous_protein_mapping",
        "evidence_backed_unmapped_enzyme",
        "cofactor_misclassified_as_protein",
        "disconnected_mapped_protein",
        "unknown_sentinel_functional_complex",
        "rag_reaction_with_provenance_and_scope",
        "curation_removal_orphans_reference",
    }


def test_fixtures_stay_small_enough_to_read() -> None:
    """A curated fixture that grows into a run artifact has stopped being one."""

    for path in sorted(FIXTURES.glob("*.json")):
        assert path.stat().st_size < 64 * 1024, f"{path.name} is no longer compact"


# ── entity identity ────────────────────────────────────────────────────────


def test_valid_mapped_enzyme_passes_every_identity_check() -> None:
    case = _case("valid_mapped_enzyme")
    row = case["row"]

    assert has_protein_external_identity(row) is True
    assert protein_external_identity(row) == case["expect"]["external_identity"]
    assert bool(protein_species_context(row)) is True
    assert is_pathbank_unknown_protein(row) is False


def test_ambiguous_mapping_ships_an_identifier_and_records_the_choice() -> None:
    case = _case("ambiguous_protein_mapping")
    row = case["row"]
    resolution = row["mapping_meta"]["resolution"]

    assert has_protein_external_identity(row) is True
    assert protein_external_identity(row) == case["expect"]["external_identity"]
    # The gate sees a clean row; only mapping_meta records that a choice was
    # made and what it was made over.
    assert resolution["status"] == case["expect"]["resolution_status"]
    assert resolution["issue"] == case["expect"]["resolution_issue"]
    assert len(row["mapping_meta"]["rejected_candidates"]) == 2


def test_evidence_backed_enzyme_is_unmapped_despite_being_sourced() -> None:
    """Evidence is not identity: the row is well-sourced and still fails."""

    case = _case("evidence_backed_unmapped_enzyme")
    row = case["row"]

    assert has_protein_external_identity(row) is False
    assert bool(protein_species_context(row)) is True
    assert bool(row.get("evidence")) is True
    assert row["mapping_meta"]["resolution"]["status"] == "unmapped"


def test_unmapped_enzyme_gate_error_carries_a_bounded_detail() -> None:
    """The diagnostic channel end-to-end: present, complete, evidence-free."""

    case = _case("evidence_backed_unmapped_enzyme")
    expect = case["expect"]
    payload = {
        "metadata": {"organism": "Bacillus subtilis"},
        "entities": {
            "species": [{"name": "Bacillus subtilis"}],
            "compounds": [],
            "proteins": [case["row"]],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    errors = _gate_errors(payload)
    match = next(error for error in errors if error.get("reason") == expect["gate_reason"])

    # path/reason keep their pre-detail shape; detail is purely additive.
    assert match["path"] == "/entities/proteins/0"
    detail = match["detail"]
    for key in expect["detail_keys_required"]:
        assert key in detail
    # Bulky provenance is censused rather than copied, so the reviewer learns
    # that evidence exists and how much without shipping a word of the paper.
    for key in expect["detail_elided_keys"]:
        assert str(detail[key]).endswith("chars elided")
        assert case["row"][key] not in str(detail[key])
    # The resolution survives the scrub -- "unmapped, no_candidate" is the
    # difference between a lookup that failed and one that never ran.
    assert detail["mapping_meta"]["resolution"]["status"] == "unmapped"


def test_cofactors_misfiled_as_proteins_fail_both_identity_gates() -> None:
    """The night's highest-volume post-gate class: CoA/succinyl-CoA as proteins."""

    case = _case("cofactor_misclassified_as_protein")
    expected = case["expect"]["each_row"]

    for row in case["rows"]:
        assert has_protein_external_identity(row) is expected["has_external_identity"]
        assert bool(protein_species_context(row)) is expected["has_species_context"]
        assert is_pathbank_unknown_protein(row) is expected["is_pathbank_unknown"]


def test_cofactor_rows_produce_the_recorded_gate_reasons() -> None:
    case = _case("cofactor_misclassified_as_protein")
    payload = {
        "metadata": {"organism": "Escherichia coli"},
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "compounds": [],
            "proteins": list(case["rows"]),
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    reasons = {error.get("reason") for error in _gate_errors(payload)}

    for expected in case["expect"]["gate_reasons"]:
        assert expected in reasons


def test_unknown_sentinel_complex_stays_recognisable_as_a_placeholder() -> None:
    case = _case("unknown_sentinel_functional_complex")
    expect = case["expect"]
    component = case["component_row"]

    assert is_pathbank_unknown_protein(component) is expect["component_is_pathbank_unknown"]
    # It satisfies the identity gates -- that is the point of a sentinel -- so
    # only is_pathbank_unknown_protein separates it from a real mapping.
    assert has_protein_external_identity(component) is expect["component_has_external_identity"]
    assert bool(protein_species_context(component)) is expect["component_has_species_context"]
    assert is_pathbank_unknown_protein(case["complex_row"]) is expect["complex_is_pathbank_unknown"]
    assert (
        component_stoichiometry(case["complex_row"]["components"][0])
        == expect["component_stoichiometry"]
    )


# ── connectivity ───────────────────────────────────────────────────────────


def test_disconnected_mapped_protein_fails_only_the_connectivity_gate() -> None:
    """FabA carries a species and a UniProt ID; the reactions name FabZ."""

    case = _case("disconnected_mapped_protein")
    expect = case["expect"]

    errors = _gate_errors(case["payload"], **case["gate_kwargs"])
    match = next(error for error in errors if error.get("reason") == expect["gate_reason"])

    assert match["path"] == expect["gate_path"]
    detail = match["detail"]
    assert detail["protein"] == expect["detail"]["protein"]
    assert detail["is_complex_component"] is expect["detail"]["is_complex_component"]
    assert detail["proteins_degree0"] == expect["detail"]["proteins_degree0"]
    assert detail["proteins_total"] == expect["detail"]["proteins_total"]
    # The message names FabA and stops; did_you_mean is what makes it a fix.
    assert detail["did_you_mean"] == expect["detail"]["did_you_mean"]


def test_connectivity_gate_stays_silent_without_the_strict_flag() -> None:
    case = _case("disconnected_mapped_protein")

    reasons = {error.get("reason") for error in _gate_errors(case["payload"])}

    assert case["expect"]["gate_reason"] not in reasons


# ── RAG additive channels ──────────────────────────────────────────────────


def test_rag_reaction_keeps_provenance_and_scope_membership() -> None:
    case = _case("rag_reaction_with_provenance_and_scope")
    expect = case["expect"]
    payload = case["payload"]

    report = validate_provenance(payload)
    assert report.ok is expect["provenance_ok"]
    assert report.checked_reactions == expect["checked_reactions"]
    assert report.issues == []

    entries, summary = build_locked_reaction_manifest(payload)
    assert len(entries) == expect["manifest_entry_count"]
    assert summary["out_of_scope_excluded_count"] == expect["manifest_excluded_out_of_scope"]
    assert payload["processes"]["reactions"][0]["scope_membership"] == expect["scope_membership_kept"]


def test_rag_additive_keys_do_not_trip_the_strict_gate() -> None:
    """Stage separation: provenance is additive and must never block export."""

    case = _case("rag_reaction_with_provenance_and_scope")

    errors = _gate_errors(case["payload"], enforce_all_proteins_connected=True)

    assert errors == []


# ── curation referential integrity ─────────────────────────────────────────


def test_orphaning_removal_is_detectable_and_safe_removal_is_not_blocked() -> None:
    case = _case("curation_removal_orphans_reference")
    expect = case["expect"]

    validate_registry_references(case["payload"])  # clean before any removal

    orphaning = json.loads(json.dumps(case["payload"]))
    orphaning["entities"]["compounds"] = [
        row
        for row in orphaning["entities"]["compounds"]
        if row["name"] != case["orphaning_removal"]["name"]
    ]
    with pytest.raises(Exception):
        validate_registry_references(orphaning)
    assert expect["registry_valid_after_orphaning_removal"] is False

    safe = json.loads(json.dumps(case["payload"]))
    safe["entities"]["compounds"] = [
        row for row in safe["entities"]["compounds"] if row["name"] != case["safe_removal"]["name"]
    ]
    validate_registry_references(safe)  # the duplicate cleanup must still go through


# ── recorded baseline ──────────────────────────────────────────────────────


def test_baseline_metrics_record_the_2026_07_28_strict_run() -> None:
    """Pins the numbers the stabilization work was measured against."""

    metrics = _load("baseline_metrics.json")
    strict = metrics["strict_legs"]

    assert metrics["batch"]["papers_attempted"] == 16
    assert strict["executed"] == 16
    assert strict["passed"] == 0
    assert strict["failed_before_post_pipeline_gates"] == 9
    assert strict["failed_at_post_pipeline_gates"] == 7
    assert strict["executed"] == (
        strict["failed_before_post_pipeline_gates"] + strict["failed_at_post_pipeline_gates"]
    )
    assert {
        "missing_external_identifier",
        "missing_species_organism",
        "degree_zero_protein",
    } <= set(strict["post_gate_failure_classes"])


def test_pre_gate_breakdown_is_counted_per_strict_leg() -> None:
    """Per strict leg from manifest.jsonl, not per paper from failures_by_code.

    The two views disagree: that report ranks distinct papers across both modes,
    which moves a paper into `processes_required` whose strict leg died at the
    gates, and files `no_reactions` separately as an uncoded failure_kind. Both
    slips leave the total at nine, so a sum check alone does not catch them --
    hence the exact breakdown.
    """

    strict = _load("baseline_metrics.json")["strict_legs"]
    codes = strict["pre_gate_failure_codes"]

    assert codes == {
        "entities_required": 3,
        "processes_required": 3,
        "no_reactions": 1,
        "ambiguous_review_scope": 2,
    }
    assert sum(codes.values()) == strict["failed_before_post_pipeline_gates"] == 9


def test_baseline_records_the_pmc12444477_payload_size_collapse() -> None:
    """Same paper, twelve hours apart: 4.7 MB passing, 306 KB failing.

    Filesystem bytes, not the manifest's decoded-character counts -- these
    payloads carry Greek letters in entity names, so the two differ by ~1%.
    """

    block = _load("baseline_metrics.json")["payload_size_pmc12444477"]
    sizes = block["bytes"]
    early = sizes["2026-07-28_0919"]["strict"]
    late = sizes["2026-07-28_2122"]["strict"]

    assert "bytes" in block["units"]
    assert early["status"] == "pass" and early["merged_payload"] == 4707785
    assert late["status"] == "fail" and late["merged_payload"] == 306325
    assert round(early["merged_payload"] / late["merged_payload"], 2) == (
        block["merged_payload_shrink_strict"]
    )


def test_every_baseline_failure_class_has_a_fixture() -> None:
    """The metrics and the fixtures must not drift apart."""

    classes = set(_load("baseline_metrics.json")["strict_legs"]["post_gate_failure_classes"])
    covered = {
        "missing_external_identifier": "evidence_backed_unmapped_enzyme",
        "missing_species_organism": "cofactor_misclassified_as_protein",
        "degree_zero_protein": "disconnected_mapped_protein",
    }

    assert set(covered) <= classes
    assert set(covered.values()) <= set(_cases())
