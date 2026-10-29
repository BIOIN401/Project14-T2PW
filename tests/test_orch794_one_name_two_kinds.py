"""ORCH-794 (D-119 § 3, ORCH-789 C7): one name declared as two kinds keeps the kind the paper supports.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR, NOT NEW WORK.** Only names that exist at the base SHA
``f3a80e09`` are imported, so this module imports cleanly there, and every ``test_g9_*`` test fails there on
VALUES: base quarantines every process naming the name as ``ambiguous_entity_type:<a>+<b>``.
``test_preservation_*`` passes at base and tip: an undecidable name still omits only its own rows (KI-011).

The rule, in order: exactly one kind carries a database identity; the paper writes the name in a role only a
protein can fill (enzyme, transporter, complex component); the name's own wording (tRNA/RNA/DNA -> nucleic
acid; receptor/channel/transporter/pump/kinase -> protein). The other kind's row and its own-kind locations are
removed pre-freeze with ``same_name_declared_as_another_kind:<kept bucket>`` (KI-051).

Real shape: the ORCH-759 chlorophyll-strict leg's quarantine input (``tests/fixtures/orch759c/``), which
declares ``tRNAGlu`` as compound and nucleic acid (ORCH-785 chlorophyll also declares ``Glu-tRNAGlu`` so).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from t2pw.pipeline.strict_quarantine import quarantine_and_close

from test_strict_quarantine_contract_alignment import _base

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orch759c"
DROP = "same_name_declared_as_another_kind:"


def _payload():
    payload = _base()
    entities, locations = payload["entities"], payload["element_locations"]
    # (a) wording: tRNAGlu as compound and nucleic acid, a reaction input, no identity either side
    entities["compounds"].append({"name": "tRNAGlu", "class": "compound"})
    entities["nucleic_acids"] = [{"name": "tRNAGlu", "class": "nucleic_acid"}]
    locations["compound_locations"].append({"compound": "tRNAGlu", "biological_state": "cytosol"})
    locations["nucleic_acid_locations"] = [{"nucleic_acid": "tRNAGlu", "biological_state": "cytosol"}]
    payload["processes"]["reactions"][0]["inputs"].append("tRNAGlu")
    # (b) role: the enzyme wrapper of reaction 1 also declared as a compound
    entities["compounds"].append({"name": "glutathione synthetase complex", "class": "compound"})
    # (c) identity: a UniProt-backed protein also declared as a compound, an interaction target
    entities["compounds"].append({"name": "glutamate-cysteine ligase", "class": "compound"})
    # (d) undecidable: no identity, no protein-only role, no wording
    entities["compounds"].append({"name": "Xq", "class": "compound"})
    entities["nucleic_acids"].append({"name": "Xq", "class": "nucleic_acid"})
    payload["processes"]["interactions"] = [
        {"name": "glutathione inhibits glutamate-cysteine ligase", "entity_1": "glutathione",
         "entity_2": "glutamate-cysteine ligase", "relationship": "inhibition", "evidence": "feedback inhibition"},
        {"name": "glutathione binds Xq", "entity_1": "glutathione", "entity_2": "Xq", "relationship": "binding",
         "evidence": "binds"},
    ]
    return payload


def _reasons(result):
    return {(a["bucket"], a["index"]): a["reason"] for a in result.quarantine_report["admissions"]}


def _dropped(result):
    return {(e["bucket"], e["name"]): e["reason"] for e in result.removed_entity_report["removed_entities"]
            if e["reason"].startswith(DROP)}


def test_g9_trna_keeps_the_nucleic_acid_its_name_says():
    result = quarantine_and_close(_payload(), strict_db=True)
    assert _reasons(result)[("reactions", 0)] == "admitted"
    assert "tRNAGlu" in [r["name"] for r in result.payload["entities"]["nucleic_acids"]]
    assert "tRNAGlu" not in [r["name"] for r in result.payload["entities"]["compounds"]]
    assert _dropped(result)[("compounds", "tRNAGlu")] == DROP + "nucleic_acids"
    assert {"compound": "tRNAGlu", "biological_state": "cytosol"} not in result.payload["element_locations"]["compound_locations"]


def test_g9_an_enzyme_role_keeps_the_protein_kind():
    result = quarantine_and_close(_payload(), strict_db=True)
    assert _reasons(result)[("reactions", 1)] == "admitted"
    assert _dropped(result)[("compounds", "glutathione synthetase complex")] == DROP + "protein_complexes"


def test_g9_the_kind_with_a_database_identity_is_kept():
    result = quarantine_and_close(_payload(), strict_db=True)
    assert _reasons(result)[("interactions", 0)] == "admitted"
    assert _dropped(result)[("compounds", "glutamate-cysteine ligase")] == DROP + "proteins"


def test_g9_an_undecidable_name_omits_only_its_own_rows():
    result = quarantine_and_close(_payload(), strict_db=True)
    assert _reasons(result)[("interactions", 1)] == "ambiguous_entity_type:compounds+nucleic_acids"
    assert not any(name == "Xq" for _bucket, name in _dropped(result))
    assert result.ok and result.refusal_reasons == []  # base: every other row is ambiguous too, nothing survives


def test_preservation_an_undecidable_name_alone_is_unchanged():
    payload = _base()
    payload["entities"]["compounds"].append({"name": "Xq", "class": "compound"})
    payload["entities"]["nucleic_acids"] = [{"name": "Xq", "class": "nucleic_acid"}]
    payload["processes"]["interactions"] = [
        {"name": "glutathione binds Xq", "entity_1": "glutathione", "entity_2": "Xq", "relationship": "binding"}]
    result = quarantine_and_close(payload, strict_db=True)
    assert _reasons(result) == {("reactions", 0): "admitted", ("reactions", 1): "admitted",
                                ("interactions", 0): "ambiguous_entity_type:compounds+nucleic_acids"}
    assert _dropped(result) == {}
    assert result.ok and result.refusal_reasons == []


def test_g9_chlorophyll_leg_keeps_the_trna_reactions():
    payload = json.load(gzip.open(FIXTURES / "chlorophyll_PMC5130987_final_mapped.json.gz", "rt", encoding="utf-8"))
    context = json.loads((FIXTURES / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8"))
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    reasons = [a["reason"] for a in result.quarantine_report["admissions"]]
    assert "ambiguous_entity_type:compounds+nucleic_acids" not in reasons
    # On this (ORCH-759) leg only tRNAGlu is declared twice; ORCH-785 also declares Glu-tRNAGlu twice.
    assert _dropped(result) == {("compounds", "tRNAGlu"): DROP + "nucleic_acids"}
    by_name = {a["name"]: a["reason"] for a in result.quarantine_report["admissions"]}
    assert by_name["Glu-tRNAGlu reduction to GSA"] == "admitted"
    assert result.ok and result.refusal_reasons == []
