"""ORCH-795 (D-114 AMENDMENT 2 § 1, ORCH-789 § 3 F-1): the headline leaves out EVERY omitted interaction PathWhiz
cannot carry, whatever its omission reason; the full count beside it keeps them.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR.** Only names that exist at base ``f3a80e09`` are imported. At base the
headline drops a KI-008 row only when the IR withheld it (``e["ki"] == "KI-008" and e["group"] == "ir"``), so an
unsigned interaction quarantined BEFORE the IR for its entity (labelled KI-001 here) stays in the denominator and
``test_g9_*`` fails on values. ``test_preservation_*``: a SIGNED interaction quarantined pre-freeze still counts.
"""

from __future__ import annotations

from t2pw.app.reaction_evidence import build_report, render_text


def _artifacts(word: str):
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}]},
               "processes": {"reactions": [{"name": "R1", "inputs": ["A"], "outputs": ["B"]}], "interactions": []}}
    quarantine = {"ok": True, "quarantined": [
        {"process_kind": "interaction", "name": f"A {word} P", "pointer": "/processes/interactions/0",
         "state": "quarantined_unmapped_entity", "reason": "protein_missing_external_identity",
         "original_process": {"name": f"A {word} P", "entity_1": "A", "entity_2": "P", "relationship": word}}]}
    ir = {"processes": {"reactions": [{"name": "R1"}], "transports": [], "interactions": []}}
    return {"final_mapped.json": payload, "quarantine_report.json": quarantine, "pwml_ir.json": ir, "pathway.pwml": True}


def test_g9_an_unsigned_interaction_omitted_before_the_ir_leaves_the_headline():
    report = build_report(_artifacts("binds"))
    cov = report["pwml_coverage"]
    assert (cov["drawn"], cov["total"], cov["by_design_withheld"]) == (1, 2, 1)
    lines = render_text(report).splitlines()
    assert lines[2].startswith("PWML coverage: 1 of 1 processes (100.0%) -- target 75%")
    assert lines[3] == "  Full count, the 1 KI-008 interactions included: 1 of 2 (50.0%)"
    # still listed, under its own omission reason and KI
    assert [(e["name"], e["ki"]) for e in report["omitted"]] == [("A binds P", "KI-001")]


def test_preservation_a_signed_interaction_omitted_before_the_ir_still_counts():
    cov = build_report(_artifacts("activates"))["pwml_coverage"]
    assert (cov["drawn"], cov["total"], cov["by_design_withheld"]) == (1, 2, 0)


def test_g9_stage6_protein_removals_reach_the_text_record():
    """ORCH-795 follow-up (ORCH-792): every ``quarantined_proteins`` record is listed once, with its KI; one that
    ``removed_entity_report`` already lists is not repeated; no coverage count moves. Base renders none of them."""

    arts = _artifacts("activates")
    before = build_report(arts)["pwml_coverage"]
    drop = {"name": "EntX", "reason": "reaction_enzyme_protein_dropped_no_external_id",
            "pass": "map_payload.enzyme_complex_rewrite", "process": "R1", "row_kept": True}
    arts["final_mapped.json"]["quarantined_proteins"] = [
        drop, dict(drop),  # the same record twice: listed once
        {"name": "CHLD", "reason": "not_referenced_by_any_surviving_process", "pass": "drop_process_orphan_proteins"},
        {"name": "GluX", "reason": "protein_missing_external_identity", "pass": "x"},  # already in removed_entity_report
    ]
    arts["removed_entity_report.json"] = {"removed_entities": [
        {"bucket": "proteins", "name": "GluX", "reason": "protein_missing_external_identity"}]}
    report = build_report(arts)
    text = render_text(report)
    assert ("Proteins left out at Stage 6 (2), from final_mapped.json quarantined_proteins:\n"
            "- proteins EntX: reaction_enzyme_protein_dropped_no_external_id (KI-007); "
            "pass map_payload.enzyme_complex_rewrite; removed from R1\n"
            "- proteins CHLD: not_referenced_by_any_surviving_process (KI-013); pass drop_process_orphan_proteins\n") in text
    assert "- proteins GluX:" not in text.split("Proteins left out at Stage 6", 1)[1]  # listed once, above it
    assert report["pwml_coverage"] == before
