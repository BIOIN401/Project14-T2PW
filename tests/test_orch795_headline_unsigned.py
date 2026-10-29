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
