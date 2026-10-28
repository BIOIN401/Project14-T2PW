"""ORCH-787: two compound rows proven to be ONE PathBank entity are consolidated pre-freeze, not refused.

G9 labelling.
* ``test_correction_protoheme_and_heme_consolidate`` is a **CORRECTION**: at base 22538dee the
  pre-freeze seam raises ``PREFREEZE_DUPLICATE_CANONICAL_ROWS`` on this shape (ORCH-784 chlorophyll,
  whole-pathway refusal) and the test fails on that raise; at the tip it passes.
* ``test_new_acceptance_text_record_lists_the_merge`` is an explicitly labelled **NEW ACCEPTANCE**
  test (the text record has no consolidation to list before this change).
* ``test_correction_unproven_group_is_omitted_per_row`` is a **CORRECTION** (D-118 § 2): at base the
  same shape with a conflicting identifier refuses the whole pathway; at the tip nothing merges and only
  the conflicting row and the process naming it are omitted, listed with a KI id.
* ``test_preservation_seam3_an_unproven_row_never_reaches_a_pwml`` is a **PRESERVATION** test that
  passes on both: a path with no strict quarantine still cannot build a PWML from an unproven row.

The DB is a stub answering the two queries PathBank answers for these names (compound 1799 "Heme"
lists "Protoheme" as a synonym, as the committed ``tests/fixtures/c119`` payload records).
"""

from __future__ import annotations

from typing import Any, Dict, List


from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

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
    assert payload["prefreeze_compound_rows_consolidated"] == [{
        "name": "protoheme", "merged_into": "heme", "db_id": 1799,
        "reason": "duplicate_canonical_row_consolidated_same_db_entity"}]
    assert report["compounds"]["rows_consolidated"] == payload["prefreeze_compound_rows_consolidated"]
    reactions = payload["processes"]["reactions"]
    assert reactions[0]["outputs"] == ["Heme"] and reactions[1]["inputs"] == ["heme"]


def test_correction_unproven_group_is_omitted_per_row() -> None:
    """CORRECTION (D-118 § 2): base raises ``PREFREEZE_DUPLICATE_CANONICAL_ROWS`` for the whole pathway.

    The ``heme`` row's KEGG id disagrees with the entity ``protoheme`` resolved to, so identity is not
    proven: nothing merges. ``protoheme`` keeps its extraction name, gains no identity, is marked, and the
    strict quarantine omits it and the one process naming it; ``heme`` and R2 survive.
    """

    from t2pw.app import reaction_evidence as E
    from t2pw.pipeline.strict_quarantine import quarantine_and_close

    payload = _payload(kegg_id="C99999")
    report = _run(payload)  # base: raises
    rows = payload["entities"]["compounds"]
    assert [row["name"] for row in rows] == ["protoporphyrin IX", "protoheme", "heme", "biliverdin"]
    assert rows[1]["prefreeze_duplicate_omission"]["reason"] == "duplicate_canonical_row_identity_not_proven"
    assert "pathbank_compound_id" not in rows[1]
    assert [r["name"] for r in report["compounds"]["rows_omitted_duplicate"]] == ["protoheme"]
    assert payload["processes"]["reactions"][0]["outputs"] == ["protoheme"]

    result = quarantine_and_close(payload, strict_db=False)
    assert [r["name"] for r in result.payload["processes"]["reactions"]] == ["R2"]
    assert "protoheme" not in [r["name"] for r in result.payload["entities"]["compounds"]]
    quarantined = result.quarantine_report["quarantined"]
    assert [(q["name"], q["reason"]) for q in quarantined] == [("R1", "duplicate_canonical_row_identity_not_proven")]

    text_report = E.build_report({E.QUARANTINE_REPORT_FILENAME: result.quarantine_report,
                                  E.FINAL_MAPPED_FILENAME: result.payload}, no_pwml_cause="test")
    assert [(e["name"], e["ki"], e["counted"]) for e in text_report["omitted"] if e["group"] == "pre_freeze"] == [
        ("R1", "KI-015", True)]


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


def test_preservation_seam3_an_unproven_row_never_reaches_a_pwml() -> None:
    """PRESERVATION (G9): passes at base 22538dee and at the tip.

    ``run_pwml_export``'s seam and the writer CLI run the pre-freeze sequence and go straight to
    ``build_pwml_ir`` with no strict quarantine (research mode removes nothing either). Base refused
    the whole pathway at the pre-freeze raise; the tip marks the row instead, so the IR must refuse it
    with a blocking error. Either way no PWML can be written from this payload.
    """

    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError
    from t2pw.pwml.writer import blocking_pwml_ir_errors

    payload = _payload(kegg_id="C99999")
    try:
        _run(payload)
    except PrefreezeResolutionError as exc:  # base: the whole-pathway refusal
        assert exc.code == "PREFREEZE_DUPLICATE_CANONICAL_ROWS"
        return
    _ir, ir_report = build_pwml_ir(payload, strict_db=False)
    blocking = [e for e in blocking_pwml_ir_errors(ir_report)
                if e["code"] == "PWML_IR_UNPROVEN_DUPLICATE_COMPOUND_ROW"]
    assert [(e["name"], e["pointer"]) for e in blocking] == [("protoheme", "/entities/compounds/1")]

    from t2pw.pipeline.strict_quarantine import PREFREEZE_DUPLICATE_OMISSION_FIELD
    from t2pw.pwml.ir import UNPROVEN_DUPLICATE_MARKER_FIELD

    assert UNPROVEN_DUPLICATE_MARKER_FIELD == PREFREEZE_DUPLICATE_OMISSION_FIELD
