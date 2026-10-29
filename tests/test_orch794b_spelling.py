"""ORCH-794b: a two-kind decision must leave every reference resolvable by the exporter (ORCH-796 chlorophyll).

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR (a regression of ORCH-794 at 55fd55ed).** The live leg declared
``tRNA^Glu`` as compound and nucleic acid while a reaction input and a location row wrote ``tRNAGlu``. The quarantine's
name key makes them one name, the exporter's (``ir._norm``) does not. 55fd55ed kept the nucleic acid, left the
``tRNAGlu`` spellings in place, admitted the reaction, and IR validation then failed the whole pathway
(``unresolved_entity_reference``): ``test_g9_*`` fails there on that value. Only names present at 55fd55ed are used.
"""

from __future__ import annotations

from helpers_prefreeze import prefrozen
from test_orch773_omit_per_row import _quarantine, _refusals
from test_strict_quarantine_contract_alignment import _base


def _leg_shape():
    payload = _base()
    entities, locations = payload["entities"], payload["element_locations"]
    entities["compounds"].append({"name": "tRNA^Glu", "class": "compound"})
    entities["nucleic_acids"] = [{"name": "tRNA^Glu", "class": "nucleic_acid"}]
    locations["compound_locations"].append({"compound": "tRNAGlu", "biological_state": "cytosol"})
    locations["nucleic_acid_locations"] = [{"nucleic_acid": "tRNA^Glu", "biological_state": "cytosol"},
                                           {"nucleic_acid": "tRNAGlu", "biological_state": "cytosol"}]
    payload["processes"]["reactions"][0]["inputs"].append("tRNAGlu")
    payload["processes"]["reactions"][0]["evidence"] = "tRNAGlu"  # free text: never rewritten
    return prefrozen(payload)


def test_g9_the_kept_kind_is_named_as_the_references_spell_it_and_the_pwml_is_written():
    result = _quarantine(_leg_shape(), None)
    assert _refusals(result) == []  # 55fd55ed: ["ir:unresolved_entity_reference", ...]
    reaction = result.payload["processes"]["reactions"][0]
    assert "tRNA^Glu" in reaction["inputs"] and "tRNAGlu" not in reaction["inputs"]
    assert result.payload["element_locations"]["nucleic_acid_locations"] == [
        {"nucleic_acid": "tRNA^Glu", "biological_state": "cytosol"}]
    assert [r["name"] for r in result.payload["entities"]["nucleic_acids"]] == ["tRNA^Glu"]
    assert reaction["evidence"] == "tRNAGlu"


def test_preservation_the_base_pathway_still_exports():
    assert _refusals(_quarantine(prefrozen(_base()), None)) == []
