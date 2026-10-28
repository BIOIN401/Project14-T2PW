"""ORCH-788 (D-114 s2, D-116): a biological state with no subcellular location is omitted per row, pre-freeze.

**G9: A REGRESSION FIX OF OBSERVABLE BEHAVIOUR.** ORCH-784 sour_taste_v2-strict wrote no PWML: two states
("Type III taste cell state", "afferent nerve state") had no resolved subcellular location, the IR refused
(``biological_state_missing_subcellular_location``) and the whole pathway was lost. At base ``22538dee``
every ``test_g9_*`` fails on VALUES: ``_refusals`` names that IR code and nothing is recorded as removed.
At tip the state, and only what cannot be drawn without it, leaves before the freeze; the IR gate is unchanged.
``test_preservation_*`` pass at base and tip.
"""

from __future__ import annotations

from typing import Any, Dict

from helpers_prefreeze import prefrozen
from test_orch773_omit_per_row import _quarantine, _refusals
from test_strict_quarantine_contract_alignment import _base

from t2pw.pipeline.strict_quarantine import QUARANTINED_BROKEN_REFERENCE

CODE = "biological_state_missing_subcellular_location"
BAD = "Type III taste cell state"


def _toy(*, process_in_bad_state: bool = False) -> Dict[str, Any]:
    payload = _base()
    payload["biological_states"].append({"name": BAD, "species": "Homo sapiens", "cell_type": "Type III cells"})
    payload["element_locations"]["compound_locations"].append({"compound": "glutathione", "biological_state": BAD})
    if process_in_bad_state:
        payload["processes"]["reactions"].append(
            {"name": "glutathione synthesis in the bad state", "inputs": ["gamma-glutamylcysteine", "glycine"],
             "outputs": ["glutathione"], "enzymes": [], "biological_state": BAD}
        )
    return prefrozen(payload)  # production order: compound verdicts exist before the quarantine's IR check


def _removed(result):
    return [(r["bucket"], r["name"], r["reason"]) for r in result.removed_entity_report["removed_entities"]]


def test_g9_a_placement_only_state_is_omitted_and_the_pathway_exports() -> None:
    payload = _toy()
    reactions = [r["name"] for r in payload["processes"]["reactions"]]
    result = _quarantine(payload, None)
    assert _refusals(result) == []
    assert (("biological_states", BAD, CODE) in _removed(result)
            and ("element_locations/compound_locations", f"glutathione in {BAD}", CODE) in _removed(result))
    assert BAD not in [s["name"] for s in result.payload["biological_states"]]
    assert [r["name"] for r in result.payload["processes"]["reactions"]] == reactions  # no process lost
    # The good placement of the same compound stays.
    assert {"compound": "glutathione", "biological_state": "cytosol"} in result.payload["element_locations"]["compound_locations"]


def test_g9_a_process_placed_in_the_state_is_omitted_whole_and_the_rest_exports() -> None:
    result = _quarantine(_toy(process_in_bad_state=True), None)
    assert _refusals(result) == []
    omitted = [(r["name"], r["state"], r["reason"]) for r in result.quarantine_report["quarantined"]]
    assert omitted == [("glutathione synthesis in the bad state", QUARANTINED_BROKEN_REFERENCE, CODE)]
    assert result.quarantine_report["release"]["status"] == "review_required"
    assert len(result.payload["processes"]["reactions"]) == len(_base()["processes"]["reactions"])

def test_preservation_a_state_named_after_a_resolved_location_is_kept() -> None:
    """The IR already places a state named after a resolved location there: same identity, nothing omitted."""

    payload = _base()
    payload["biological_states"] = [{"name": "cytosol", "species": "Homo sapiens"}]
    result = _quarantine(prefrozen(payload), None)
    assert _refusals(result) == []
    assert [s["name"] for s in result.payload["biological_states"]] == ["cytosol"]
    assert not [r for r in _removed(result) if r[2] == CODE]
