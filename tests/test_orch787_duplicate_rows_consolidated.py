"""ORCH-787: two compound rows proven to be ONE PathBank entity are consolidated pre-freeze, not refused.

G9 labelling.
* ``test_correction_protoheme_and_heme_consolidate`` is a **CORRECTION**: at base 22538dee the
  pre-freeze seam raises ``PREFREEZE_DUPLICATE_CANONICAL_ROWS`` on this shape (ORCH-784 chlorophyll,
  whole-pathway refusal) and the test fails on that raise; at the tip it passes.
* ``test_new_acceptance_text_record_lists_the_merge`` is an explicitly labelled **NEW ACCEPTANCE**
  test (the text record has no consolidation to list before this change).
* ``test_control_conflicting_identifier_still_refuses`` is a **CONTROL** that passes on both: the
  gate is not weakened for rows whose identity is not proven (D-035 clauses 3b, 6, 7).

The DB is a stub answering the two queries PathBank answers for these names (compound 1799 "Heme"
lists "Protoheme" as a synonym, as the committed ``tests/fixtures/c119`` payload records).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import pytest

from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution

HEME = {"id": 1799, "name": "Heme", "short_name": "Heme", "hmdb_id": "HMDB0003178", "kegg_id": "C00032",
        "chebi_id": "CHEBI:17627", "pubchem_cid": "3334", "synonyms": "Haem; Protoheme; Heme B; Protoheme IX"}


class _PathBankStub:
    last_error = ""

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:
        first = str(params[0]).lower() if params else ""
        if "WHERE LOWER(name)=LOWER(%s)" in sql and first == "heme":
            return [dict(HEME)]
        if "LOWER(short_name)=LOWER(%s) OR LOWER(synonyms) LIKE" in sql and first in ("heme", "protoheme"):
            return [dict(HEME)]
        return []


#: The ``heme`` row as Stage 6 leaves it: PathBank 1799 stamped, name kept (the same paper's ORCH-781
#: ``final_mapped.json`` records exactly this, ``legacy_id_unverified``). ``protoheme`` arrives with no id
#: and the pre-freeze seam resolves it by PathBank synonym onto "Heme" -- the ORCH-784 collision.
STAGE6_HEME = {"pathbank_compound_id": 1799, "chebi_id": "17627", "kegg_id": "C00032"}


def _payload(**heme_extra: Any) -> Dict[str, Any]:
    heme_extra = {**STAGE6_HEME, **heme_extra}
    return {
        "entities": {"compounds": [
            {"name": "protoporphyrin IX", "class": "compound"},
            {"name": "protoheme", "class": "compound"},
            {"name": "heme", "class": "cofactor", **heme_extra},
            {"name": "biliverdin", "class": "compound"},
        ]},
        "processes": {"reactions": [
            {"name": "R1", "inputs": ["protoporphyrin IX"], "outputs": ["protoheme"]},
            {"name": "R2", "inputs": ["heme"], "outputs": ["biliverdin"]},
        ]},
    }


def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_prefreeze_resolution(payload, strict_db=True, db_resolver=_PathBankStub(), name_index=None)


def test_correction_protoheme_and_heme_consolidate() -> None:
    payload = _payload()
    report = _run(payload)  # base: raises PREFREEZE_DUPLICATE_CANONICAL_ROWS

    names = [row["name"] for row in payload["entities"]["compounds"]]
    assert names == ["protoporphyrin IX", "heme", "biliverdin"]
    survivor = payload["entities"]["compounds"][1]
    assert survivor["class"] == "cofactor" and survivor["pathbank_compound_id"] == 1799
    assert "protoheme" in survivor["synonyms"]
    assert survivor["consolidated_rows"] == [{
        "name": "protoheme", "merged_into": "heme", "db_id": 1799,
        "reason": "duplicate_canonical_row_consolidated_same_db_entity"}]
    assert report["compounds"]["rows_consolidated"] == survivor["consolidated_rows"]
    reactions = payload["processes"]["reactions"]
    assert reactions[0]["outputs"] == ["Heme"] and reactions[1]["inputs"] == ["heme"]


def test_control_conflicting_identifier_still_refuses() -> None:
    payload = _payload(kegg_id="C99999")  # heme row disagrees with the entity protoheme resolved to
    original = deepcopy(payload)
    with pytest.raises(PrefreezeResolutionError) as excinfo:
        _run(payload)
    assert excinfo.value.code == "PREFREEZE_DUPLICATE_CANONICAL_ROWS"
    assert payload == original


def test_new_acceptance_text_record_lists_the_merge() -> None:
    from t2pw.app import reaction_evidence as E

    payload = _payload()
    _run(payload)
    report = E.build_report({E.FINAL_MAPPED_FILENAME: payload}, no_pwml_cause="test")
    merged = [e for e in report["omitted"] if e["kind"] == "compound"]
    assert [(e["name"], e["ki"], e["counted"], e["merged_duplicate"]) for e in merged] == [
        ("protoheme", "KI-045", False, True)]
    text = E.render_text(report)
    assert "- merged duplicate, not counted: protoheme" in text and "(KI-045)" in text
