"""C-040a / D-028: which PathBank matches may rename a row and stamp identifiers.

**G9 REGRESSION PROOF, not a new capability.** Every ``test_g9_*`` asserts that
pre-existing *observable* behaviour is corrected, and each fails behaviourally at
base f2f7599 -- the row comes back renamed and carrying identifiers -- not on a
missing symbol. Run it against a clean export of the base by pointing
``PYTHONPATH`` there; that is why this module does NOT do the
``sys.path.insert(ROOT / "src")`` the older test modules do, which would pin the
import to this worktree and make a base run silently test the tip. Nothing here
touches a database -- a stub replaces the resolver entirely.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

import t2pw.pwml.compound_resolution as cr

#: Auditor-supplied. ``OPDA`` is NOT a PathBank synonym of row 104723 -- exact
#: name and synonym lookups both return empty -- and the fuzzy tie-break admits
#: this by 0.0006.
OPDA_MATCH: Dict[str, Any] = {
    "status": "matched", "confidence": 0.65, "chosen_rule": "fuzzy_name",
    "raw_name": "OPDA", "candidates": [],
    "chosen": {"id": 104723, "name": "Dinor-12-oxo-phytodienoate",
               "kegg_id": "C01226", "chebi_id": "57411", "pubchem_cid": "45266620"},
}

#: Must-still-work control: an exact synonym hit on a 7-character name.
OPC8_MATCH: Dict[str, Any] = {
    "status": "matched", "confidence": 0.70,
    "chosen_rule": "exact_short_name_or_synonym", "raw_name": "OPC-8:0",
    "candidates": [],
    "chosen": {"id": 104724,
               "name": "8-[(1R,2R)-3-oxo-2-(pent-2-enyl)cyclopentyl]octanoate",
               "kegg_id": "C04780", "chebi_id": "57433", "pubchem_cid": "5280411"},
}

#: Safety-valve control: a 3-character abbreviation matched by exact synonym.
COA_MATCH: Dict[str, Any] = {
    "status": "matched", "confidence": 0.70,
    "chosen_rule": "exact_short_name_or_synonym", "raw_name": "CoA",
    "candidates": [],
    "chosen": {"id": 1191, "name": "Coenzyme A", "kegg_id": "C00010",
               "chebi_id": "15346", "pubchem_cid": "87642",
               "hmdb_id": "HMDB0001423", "pwc_id": "PW_C001191", "short_name": "CoA"},
}

#: Identifier keys a refused match must not write onto the row.
STAMPED = ("pathwhiz_id", "pathbank_compound_id", "pw_compound_id", "db_id",
           "kegg_id", "chebi_id", "pubchem_cid", "hmdb_id", "pwc_id")


class _StubResolver:
    """Replaces ``PathWhizCompoundResolver``; returns one canned verdict."""

    def __init__(self, db: Any) -> None:
        self._match = db.match

    def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(self._match)


class _StubDb:
    def __init__(self, match: Dict[str, Any]) -> None:
        self.match = match

    def available(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _stub_resolver(monkeypatch: Any) -> None:
    print(f"t2pw.pwml.compound_resolution -> {cr.__file__}")
    monkeypatch.setattr(cr, "PathWhizCompoundResolver", _StubResolver)


def _resolve(row: Dict[str, Any], match: Dict[str, Any]) -> tuple:
    report: Dict[str, Any] = {}
    resolved = cr._resolve_compound_rows(
        [copy.deepcopy(row)], db_resolver=_StubDb(match), strict_db=True,
        report=report, pointer_prefix="/entities/compounds", name_index=None,
    )
    assert len(resolved) == 1, "a refused row must never be dropped from the row set"
    return resolved[0], report


def _refusals(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        issue for issue in (report.get("unresolved") or {}).get("db_identities", [])
        if isinstance(issue.get("admission"), dict) and not issue["admission"]["admitted"]
    ]


# --- G9: fails behaviourally at base f2f7599, passes at the tip ---------------


def test_g9_a_fuzzy_match_no_longer_renames_the_compound() -> None:
    """AT BASE: name becomes 'Dinor-12-oxo-phytodienoate'. AT TIP: stays 'OPDA'."""
    row, _ = _resolve({"name": "OPDA"}, OPDA_MATCH)
    assert row["name"] == "OPDA"


def test_g9_a_fuzzy_match_no_longer_stamps_identifiers() -> None:
    """AT BASE: pathwhiz_id/kegg_id/chebi_id/pubchem_cid are all written."""
    row, _ = _resolve({"name": "OPDA"}, OPDA_MATCH)
    written = sorted(key for key in STAMPED if row.get(key) not in (None, ""))
    assert written == [], f"D-028 rule 1 violated; row gained {written}"
    assert "db_row" not in row, "a refused match must not seed a trusted db_row"


def test_g9_the_fuzzy_refusal_is_recorded_and_does_not_raise() -> None:
    """D-028 rule 4. The refusal is filed for review, with its rule and score."""
    row, report = _resolve({"name": "OPDA"}, OPDA_MATCH)
    refusals = _refusals(report)
    assert len(refusals) == 1, "the sub-threshold refusal was not recorded"
    admission = refusals[0]["admission"]
    assert admission["reason"] == "fuzzy_name_match_never_admitted"
    assert admission["rule"] == "fuzzy_name" and admission["confidence"] == 0.65
    # Evidence preserved on the row, but not as a trusted identity.
    assert row["db_match"] is not None and row["db_status"] != "matched"


def test_g9_a_short_abbreviation_without_a_corroborating_identifier_is_refused() -> None:
    """AT BASE: 'CoA' is renamed to 'Coenzyme A' and stamped. AT TIP: refused.

    Same verdict as the safety-valve control below; the only difference is that
    this row carries no exact identifier of its own.
    """
    row, report = _resolve({"name": "CoA"}, COA_MATCH)
    assert row["name"] == "CoA"
    assert sorted(key for key in STAMPED if row.get(key) not in (None, "")) == []
    admission = _refusals(report)[0]["admission"]
    assert admission["reason"] == "short_abbreviation_without_corroborating_identifier"
    assert admission["normalized_length"] == 3
    assert admission["corroborating_ids"] == []


# --- Controls: what must still work ------------------------------------------


def test_control_an_exact_synonym_on_a_long_enough_name_still_renames_and_stamps() -> None:
    """OPC-8:0 normalizes to 7 characters, so rule 2's guard does not fire."""
    row, _ = _resolve({"name": "OPC-8:0"}, OPC8_MATCH)
    assert row["name"] == "8-[(1R,2R)-3-oxo-2-(pent-2-enyl)cyclopentyl]octanoate"
    assert row["pathwhiz_id"] == 104724
    assert row["kegg_id"] == "C04780" and row["chebi_id"] == "57433"
    assert row["raw_name"] == "OPC-8:0", "the extracted name must survive as raw_name"


def test_control_safety_valve_a_corroborated_short_abbreviation_still_resolves() -> None:
    """Rule 2's safety valve -- what protects ATP, NAD, FAD, CoA, NADP."""
    row, report = _resolve({"name": "CoA", "kegg_id": "C00010"}, COA_MATCH)
    assert row["name"] == "Coenzyme A"
    assert row["pathwhiz_id"] == 1191 and row["chebi_id"] == "15346"
    assert _refusals(report) == []


@pytest.mark.parametrize("id_field,value", [
    ("kegg_id", "C00010"), ("chebi_id", "15346"),
    ("pubchem_cid", "87642"), ("hmdb_id", "HMDB0001423"),
])
def test_control_each_of_the_four_exact_identifiers_corroborates(id_field: str, value: str) -> None:
    """D-028 rule 2 names KEGG, ChEBI, PubChem and HMDB; all four must count."""
    row, _ = _resolve({"name": "CoA", id_field: value}, COA_MATCH)
    assert row["name"] == "Coenzyme A"


#: The measured false admission that made corroboration mean AGREEMENT, not
#: presence: the committed `PE` row carries a KEGG id absent from PathBank's
#: compounds.kegg_id -- which is why no identifier rule matched it -- while the
#: synonym hit is on a different compound carrying a different KEGG id.
PE_MATCH: Dict[str, Any] = {
    "status": "matched", "confidence": 0.70,
    "chosen_rule": "exact_short_name_or_synonym", "raw_name": "PE",
    "candidates": [],
    "chosen": {"id": 149, "name": "O-Phosphoethanolamine", "kegg_id": "C00346",
               "chebi_id": "17553", "pubchem_cid": "1015"},
}


def test_a_disagreeing_identifier_does_not_corroborate_a_short_abbreviation() -> None:
    """`PE` (2 chars) must stay `PE`: its KEGG id is not the matched row's.

    PE means phosphatidylethanolamine, not O-phosphoethanolamine. Under a
    presence-only reading of rule 2 this row was admitted and renamed.
    """
    row, report = _resolve({"name": "PE", "mapped_ids": {"kegg": "C00012"}}, PE_MATCH)
    assert row["name"] == "PE"
    # It keeps its OWN kegg id -- hoisted out of mapped_ids by the pre-existing
    # _normalize_compound_external_ids, not stamped from the match -- and gains
    # none of the matched row's identity, including that row's different kegg id.
    assert row["kegg_id"] == "C00012" != PE_MATCH["chosen"]["kegg_id"]
    gained = sorted(k for k in STAMPED if k != "kegg_id" and row.get(k) not in (None, ""))
    assert gained == [], f"D-028 rule 3 violated; row gained {gained}"
    admission = _refusals(report)[0]["admission"]
    assert admission["reason"] == "short_abbreviation_identifier_disagrees_with_match"
    assert admission["corroborating_ids"] == [] and admission["disagreeing_ids"] == ["kegg"]


def test_an_identifier_with_no_counterpart_on_the_matched_row_corroborates_nothing() -> None:
    """Absent counterpart is not agreement, so it cannot open the valve."""
    match = copy.deepcopy(COA_MATCH)
    match["chosen"].pop("hmdb_id")
    row, report = _resolve({"name": "CoA", "hmdb_id": "HMDB0001423"}, match)
    assert row["name"] == "CoA"
    assert _refusals(report)[0]["admission"]["disagreeing_ids"] == ["hmdb"]


def test_corroboration_survives_the_shared_normalizers() -> None:
    """`CHEBI:` prefix and `cpd:` prefix must not read as disagreement."""
    row, _ = _resolve({"name": "CoA", "chebi_id": "CHEBI:15346"}, COA_MATCH)
    assert row["name"] == "Coenzyme A"
    row, _ = _resolve({"name": "CoA", "mapped_ids": {"kegg": "cpd:C00010"}}, COA_MATCH)
    assert row["name"] == "Coenzyme A"


def test_control_mapped_ids_corroborate_as_well_as_direct_columns() -> None:
    """``_compound_external_ids`` already reads both shapes; so must the guard."""
    row, _ = _resolve({"name": "CoA", "mapped_ids": {"kegg": "C00010"}}, COA_MATCH)
    assert row["name"] == "Coenzyme A"


def test_control_an_identifier_backed_match_is_untouched_by_the_new_gate() -> None:
    """The >= 0.85 arm is identifier-driven and must behave exactly as before."""
    match = {"status": "matched", "confidence": 0.95, "chosen_rule": "kegg_id_exact",
             "raw_name": "ATP", "candidates": [],
             "chosen": {"id": 1189, "name": "Adenosine triphosphate", "kegg_id": "C00002"}}
    row, report = _resolve({"name": "ATP", "kegg_id": "C00002"}, match)
    assert row["name"] == "Adenosine triphosphate" and row["pathwhiz_id"] == 1189
    assert row["db_status"] == "matched"
    assert (report.get("unresolved") or {}).get("db_identities") == []


# --- Rules 3 and 4 -----------------------------------------------------------


def test_a_refusal_is_never_a_partial_apply() -> None:
    """D-028 rule 3. Not one identity field may leak through a refusal."""
    row, _ = _resolve({"name": "OPDA"}, OPDA_MATCH)
    for key in ("short_name", *STAMPED, "db_row", "mapped_ids"):
        assert row.get(key) in (None, ""), f"refused row leaked {key}={row.get(key)!r}"


def test_an_unmatched_verdict_behaves_exactly_as_before() -> None:
    """The refusal path must not perturb the ordinary no-match record."""
    match = {"status": "unmatched", "confidence": 0.0, "chosen_rule": "",
             "raw_name": "novel thing", "chosen": None, "candidates": [],
             "reason": "No PathWhiz DB match by IDs or name"}
    row, report = _resolve({"name": "novel thing"}, match)
    assert row["name"] == "novel thing" and row["db_status"] == "unmatched"
    assert report["unresolved"]["db_identities"][0]["admission"]["reason"] == "no_match_to_admit"


def test_a_refused_row_does_not_abort_the_batch_and_keeps_its_neighbours(
    monkeypatch: Any,
) -> None:
    """D-028 rule 4 / merge rule 7: refuse, record, and carry on."""
    report: Dict[str, Any] = {}
    rows = [{"name": "OPDA"}, {"name": "OPC-8:0"}, {"name": "OPDA"}]

    class _PerName:
        def __init__(self, db: Any) -> None:
            self.db = db

        def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
            return copy.deepcopy(OPC8_MATCH if row["name"] == "OPC-8:0" else OPDA_MATCH)

    monkeypatch.setattr(cr, "PathWhizCompoundResolver", _PerName)
    resolved = cr._resolve_compound_rows(
        rows, db_resolver=_StubDb(OPDA_MATCH), strict_db=True, report=report,
        pointer_prefix="/entities/compounds", name_index=None,
    )
    assert [r["name"] for r in resolved] == [
        "OPDA", "8-[(1R,2R)-3-oxo-2-(pent-2-enyl)cyclopentyl]octanoate", "OPDA"]
    assert len(_refusals(report)) == 2


# --- Rule 5: the constants must be named -------------------------------------


def test_the_short_abbreviation_boundary_is_a_named_constant_at_four() -> None:
    assert cr.SHORT_ABBREVIATION_MAX_CHARS == 4
    assert cr.RECORD_ONLY_MATCH_RULES == frozenset({"fuzzy_name"})
    assert cr.EXACT_NAME_MATCH_RULES == frozenset({"exact_name", "exact_short_name_or_synonym"})
    assert cr.CORROBORATING_ID_KEYS == ("kegg", "chebi", "pubchem", "hmdb")
    assert cr.DB_MATCH_CONFIDENCE_FLOOR == 0.85


@pytest.mark.parametrize("name,expected_length", [
    ("CL", 2), ("PE", 2), ("THF", 3), ("G3P", 3), ("OPDA", 4),
    ("NAD+", 4), ("OPC-8:0", 7), ("glycerol-3-phosphate", 20),
])
def test_the_boundary_is_measured_after_the_modules_existing_normalization(
    name: str, expected_length: int,
) -> None:
    """``_norm`` is the normalization D-028 refers to; these are its lengths."""
    assert len(cr._norm(name)) == expected_length
    is_short = expected_length <= cr.SHORT_ABBREVIATION_MAX_CHARS
    row, _ = _resolve({"name": name}, dict(OPC8_MATCH, raw_name=name))
    assert (row["name"] == name) is is_short
