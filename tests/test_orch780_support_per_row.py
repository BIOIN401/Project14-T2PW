"""ORCH-780 (D-116 s1): a reaction without reaction-level support is omitted before the freeze, per row.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR.** Every name imported here exists at base ``4a15ef42`` with the
same signature, so the module imports there and each ``test_g9_*`` fails there on VALUES:

* ``test_g9_unsupported_reaction_is_omitted_and_the_interactions_export`` (the fixture the lane is charged
  with: one unsupported reaction, two signed protein interactions). Base: no PWML, the required-field gate
  refuses the pathway with ``no_defensible_reaction_support`` (KI-022). Tip: a PWML with both interactions,
  the reaction recorded in ``quarantined[]`` under that code and listed in the text record's OMITTED section.
``test_preservation_*`` pass at base and at tip: supported reactions are never omitted; a payload the gate
passes is not touched, including an unsupported reaction beside a supported one (the gate is a leg-level
floor and D-116 s2 keeps it "exactly as strict"; extending the omission to that case lost gold recall on
the corpus, ``ORCH-780-RESULT.md`` s4.5); a payload with no lineage carrier is not judged (the gate's own
archival guard); research mode applies nothing; and an omission that would leave nothing to draw is
rolled back to the gate's own refusal.

The rows are the ones the gate's own row predicate names: ``reaction_support.reaction_support_class(row)
is None``, read only when the gate itself refuses. The gate module is not changed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from test_orch773_omit_per_row import _omitted, _quarantine, _refusals, _review_required, _text_record
from test_strict_quarantine_contract_alignment import _base

from t2pw.pipeline.reaction_support import evaluate_reaction_support, reaction_support_issue
from t2pw.pipeline.strict_quarantine import QUARANTINED_BROKEN_REFERENCE
from t2pw.pwml.ir import build_pwml_ir
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

CODE = "no_defensible_reaction_support"

#: A Stage-2A style row: inferred, no paper or RAG entry. Unsupported by the gate's own predicate.
UNSUPPORTED = [{"stage": "deterministic_inference", "origin": "inferred", "support": "derived",
                "paper_explicit": "not_evaluated", "reason": "", "review_required": False, "uncertainty": "",
                "sources": []}]
#: A Stage-1 row the paper states.
PAPER = [{"stage": "paper_extraction", "origin": "paper_stated", "support": "direct", "paper_explicit": "explicit",
          "reason": "", "review_required": False, "uncertainty": "", "sources": []}]

GCL, GS = "glutamate-cysteine ligase complex", "glutathione synthetase complex"
SIGNED = [
    {"name": f"{GS} activates {GCL}", "entity_1": GS, "relationship": "activates", "entity_2": GCL,
     "biological_state": "cytosol"},
    {"name": f"{GCL} inhibits {GS}", "entity_1": GCL, "relationship": "inhibits", "entity_2": GS,
     "biological_state": "cytosol"},
]


def _toy(*lineages: Any, interactions: bool = True) -> Dict[str, Any]:
    """The contract-alignment pathway (two reactions, each catalysed by a one-protein wrapper of a UniProt
    protein; passes every gate on its own), cut to one reaction per lineage given (``None`` = no lineage),
    plus two signed interactions between the two enzyme complexes."""

    payload = _base()
    payload["metadata"].pop("key_compounds")
    reactions = payload["processes"]["reactions"]
    del reactions[len(lineages):]
    for row, lineage in zip(reactions, lineages):
        if lineage is not None:
            row["provenance_lineage"] = deepcopy(lineage)
    if interactions:
        payload["processes"]["interactions"] = deepcopy(SIGNED)
    return payload


def _drawn_interactions(payload: Dict[str, Any]) -> List[str]:
    export = deepcopy(payload)
    run_prefreeze_resolution(export, strict_db=True)
    ir, _ = build_pwml_ir(export, pathway_name="Generated Pathway", pathway_subject="Metabolic", strict_db=True,
                          width=3200, height=1400)
    return [str(row.get("name")) for row in (ir.get("processes") or {}).get("interactions") or []]


def _reactions(result) -> List[str]:
    return [row["name"] for row in result.payload["processes"].get("reactions") or []]


# ── G9 ─────────────────────────────────────────────────────────────────────


def test_g9_unsupported_reaction_is_omitted_and_the_interactions_export() -> None:
    payload = _toy(UNSUPPORTED)
    assert reaction_support_issue(payload)["code"] == CODE  # the gate's verdict on the input, unchanged

    result = _quarantine(payload, None)

    assert _refusals(result) == []  # base: ["required:no_defensible_reaction_support"]
    assert _omitted(result) == [
        ("gamma-glutamylcysteine synthesis", QUARANTINED_BROKEN_REFERENCE, CODE, "process", "", ""),
    ]
    [row] = [r for r in result.quarantine_report["quarantined"] if r["reason"] == CODE]
    assert row["detail"]["omission"]["gate"] == "required_field_gate"
    assert row["original_process"]["provenance_lineage"] == UNSUPPORTED  # kept verbatim for a reviewer
    assert _reactions(result) == []
    assert [row["name"] for row in result.payload["processes"]["interactions"]] == [r["name"] for r in SIGNED]
    assert sorted(_drawn_interactions(result.payload)) == sorted(r["name"] for r in SIGNED)
    _review_required(result)

    # Listed in the text record, under the gate's code and its catalogue id.
    report, text = _text_record(result)
    [entry] = [e for e in report["omitted"] if e["name"] == "gamma-glutamylcysteine synthesis"]
    assert (entry["reason"], entry["ki"], entry["counted"]) == (CODE, "KI-022", True)
    section = text[text.index("\nOMITTED FROM THE PWML ("):]
    assert "Name: gamma-glutamylcysteine synthesis" in section and f"Reason: {CODE}" in section


# ── Preservation (pass at base and at tip) ────────────────────────────────


def test_preservation_a_payload_the_gate_passes_is_untouched() -> None:
    """One supported and one unsupported reaction: the gate passes it, so nothing is omitted, as at base."""

    payload = _toy(PAPER, UNSUPPORTED)
    assert evaluate_reaction_support(payload)["verdict"] == "supported"
    result = _quarantine(payload, None)
    assert _refusals(result) == [] and _omitted(result) == []
    assert _reactions(result) == ["gamma-glutamylcysteine synthesis", "glutathione synthesis"]


def test_preservation_supported_reactions_are_never_omitted() -> None:
    rag = _toy(PAPER, None)
    rag["processes"]["reactions"][1]["rag_provenance"] = {"admission": "admitted"}
    for payload in (_toy(PAPER, PAPER), rag):
        result = _quarantine(payload, None)
        assert _refusals(result) == []
        assert _omitted(result) == []
        assert _reactions(result) == ["gamma-glutamylcysteine synthesis", "glutathione synthesis"]


def test_preservation_no_lineage_carrier_nothing_is_judged() -> None:
    """No row carries lineage: support is unknown, not absent, and the gate abstains; so does the pass."""

    payload = _toy(None, None)
    assert evaluate_reaction_support(payload)["verdict"] == "indeterminate"
    result = _quarantine(payload, None)
    assert _refusals(result) == [] and _omitted(result) == []
    assert _reactions(result) == ["gamma-glutamylcysteine synthesis", "glutathione synthesis"]


def test_preservation_research_mode_omits_nothing() -> None:
    payload = _toy(UNSUPPORTED)
    result = _quarantine(payload, None, mode="research")
    assert result.payload == payload
    assert _omitted(result) == []


def test_preservation_nothing_left_to_draw_rolls_back_to_the_gates_refusal() -> None:
    """Every reaction unsupported and nothing else: omitting them buys no PWML, so nothing is omitted and the
    required-field gate refuses exactly as at base."""

    result = _quarantine(_toy(UNSUPPORTED, UNSUPPORTED, interactions=False), None)
    assert _omitted(result) == []
    assert _refusals(result) == [f"required:{CODE}"]
