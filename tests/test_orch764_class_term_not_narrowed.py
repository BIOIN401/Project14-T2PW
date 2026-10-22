"""ORCH-764 / D-109 § 2: mapping never narrows a class name to a specific molecule.

MenA's co-substrate is named by PMC12421875 as a CLASS, "prenyl diphosphate" (gold R8 note: the
chain length is never stated). On the ORCH-759 3-draw leg it reached ``final_mapped.json`` as
PathBank compound 874 "Dimethylallylpyrophosphate" (DMAPP, the C5 member), renamed by pre-freeze
compound resolution: ``PathWhizCompoundResolver`` matched the PathBank SYNONYM "prenyl diphosphate"
of row 874 (rule ``exact_short_name_or_synonym``), D-028 admits every exact synonym longer than four
characters, and ``apply_compound_db_resolution`` renamed the row and stamped DMAPP's identifiers.

Every outside answer is a fixture (``tests/fixtures/orch764/``, cut by
``docs/pwml_recovery_sprint/evidence/orch764/make_fixture.py``); nothing touches a database or the
network. The PathBank database is a small SQL emulator over the fixture's recorded row.

* ``test_g9_*`` -- G9 REGRESSION PROOF: each fails behaviourally at base 79080db7 (the class term
  comes back as DMAPP, or carrying DMAPP's identifiers) and passes at the tip. No test imports a
  symbol the base lacks, so the base run is a behavioural failure, not an ImportError.
* ``test_fidelity_*`` -- the fixture reproduces the leg's recorded answer; passes at base and tip.
* ``test_preservation_*`` -- labelled PRESERVATION: today's behaviour that must not move; passes at
  base and tip.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from t2pw.mapping.map_ids import _apply_name_plausibility_gate, _enforce_shipped_identity_names
from t2pw.pwml.db_resolver import IDENTITY_REFUSED_STATUS, PathWhizCompoundResolver
from t2pw.pwml.prefreeze_resolution import resolve_compounds_prefreeze

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "orch764" / "menA_prenyl_diphosphate.json").read_text(encoding="utf-8")
)

#: The refusal reason the tip records. A string literal on purpose: the base has no such constant.
D109_REASON = "class_term_never_narrowed_to_specific_molecule"

#: Every identifier column ``apply_compound_db_resolution`` writes on an admitted match.
STAMPED = ("db_id", "db_row", "pathwhiz_id", "pathbank_compound_id", "pw_compound_id", "hmdb_id",
           "kegg_id", "chebi_id", "pubchem_cid", "pwc_id", "cas", "biocyc_id", "chemspider_id", "drugbank_id")

#: DMAPP's identifiers as ChEBI / KEGG / HMDB give them (the leg's recorded row 874).
DMAPP_IDS = {"chebi": "CHEBI:16057", "kegg": "C00235", "hmdb": "HMDB0001120"}

_COND = re.compile(r"(?:LOWER\((\w+)\)|(\w+))\s*(=|LIKE)\s*(?:LOWER\(%s\)|%s)")


class FixtureCompoundDb:
    """The PathBank ``compounds`` table, emulated over recorded rows for the SQL the resolver issues.

    ``PathWhizCompoundResolver`` issues ``col=%s`` / ``LOWER(col)=LOWER(%s)`` / ``LOWER(col) LIKE
    LOWER(%s)`` conditions joined by OR (``db_resolver.py``). Each is evaluated case-blind; a LIKE
    pattern is ``%text%`` there, so it is a substring test. Nothing else is modelled.
    """

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        where = sql.split(" WHERE ", 1)[1].split(" LIMIT ")[0]
        conds = _COND.findall(where)
        assert conds and len(conds) == len(params), (sql, params)
        out = []
        for row in self.rows:
            for (lowered, plain, op), value in zip(conds, params):
                cell = str(row.get(lowered or plain) or "").casefold()
                wanted = str(value).casefold()
                if (op == "=" and cell == wanted) or (op == "LIKE" and cell and wanted.strip("%") in cell):
                    out.append(dict(row))
                    break
        return out


def _rows() -> List[Dict[str, Any]]:
    return copy.deepcopy(FIXTURE["pathbank_compounds"])


def _payload(term: str = "prenyl diphosphate") -> Dict[str, Any]:
    """The leg's MenA slice as it stood before pre-freeze resolution, the class term spelled ``term``."""
    text = json.dumps(FIXTURE["prefreeze_input"], ensure_ascii=False)
    return json.loads(text.replace('"prenyl diphosphate"', json.dumps(term)))


def _prefreeze(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> tuple:
    report: Dict[str, Any] = {}
    summary = resolve_compounds_prefreeze(payload, db_resolver=FixtureCompoundDb(rows), report=report)
    return summary, report


def _compound(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    hits = [row for row in payload["entities"]["compounds"] if row.get("raw_name", row["name"]) == name]
    assert len(hits) == 1, [row["name"] for row in payload["entities"]["compounds"]]
    return hits[0]


# --------------------------------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------------------------------


def test_fidelity_fixture_db_reproduces_the_legs_recorded_lookup_answer() -> None:
    """The emulated database gives exactly the answer the leg recorded in ``db_match``."""
    row = _compound(_payload(), "prenyl diphosphate")
    answer = PathWhizCompoundResolver(FixtureCompoundDb(_rows())).resolve(row)
    assert answer == FIXTURE["recorded_db_match"]
    assert (answer["status"], answer["chosen_rule"], answer["confidence"], answer["chosen"]["id"]) == (
        "matched", "exact_short_name_or_synonym", 0.7, 874)


# --------------------------------------------------------------------------------------------------
# G9: the recorded narrowing, at its origin
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("term", ["prenyl diphosphate", "Prenyl Diphosphate"])
def test_g9_prefreeze_keeps_the_paper_class_term_and_flags_it(term: str) -> None:
    """G9. Base: the row and the MenA reaction come back as Dimethylallylpyrophosphate, DMAPP ids on."""
    payload = _payload(term)
    summary, report = _prefreeze(payload, _rows())

    row = _compound(payload, term)
    assert row["name"] == term
    assert not any(row.get(key) not in (None, "", {}) for key in STAMPED), {k: row.get(k) for k in STAMPED}
    assert not row.get("mapped_ids")
    assert row["db_status"] == IDENTITY_REFUSED_STATUS  # the existing record-only review flag
    assert row["db_match"]["chosen"]["id"] == 874  # the refused answer is kept for the reviewer

    menA = payload["processes"]["reactions"][0]
    assert menA["inputs"] == ["1,4-dihydroxy-2-naphthoic acid", term]
    assert summary["rename_map"] == {}
    assert "Dimethylallylpyrophosphate" not in json.dumps(payload["processes"])

    refusals = [issue for issue in report["unresolved"]["db_identities"] if issue["raw_name"] == term]
    assert len(refusals) == 1
    assert refusals[0]["admission"]["admitted"] is False
    assert refusals[0]["admission"]["reason"] == D109_REASON


# --------------------------------------------------------------------------------------------------
# G9: the same rule at Stage 6 (map_ids), on a provider record that names the class term itself
# --------------------------------------------------------------------------------------------------


def _same_named_record() -> Dict[str, Any]:
    """CONSTRUCTED, not recorded: a provider answer whose record carries the class term as its own
    name and DMAPP's identifiers -- the shape a ChEBI/KEGG/HMDB name search can return, since each
    lists "prenyl diphosphate" among DMAPP's names. The name gate keeps it on the shared tokens."""
    return {
        "status": "mapped", "provider": "ChEBI/KEGG/HMDB", "source": "api", "confidence": 0.9,
        "chosen_rule": "exact_name", "mapped_ids": dict(DMAPP_IDS),
        "candidates": [{"name": "prenyl diphosphate", "mapped_ids": dict(DMAPP_IDS)}],
        "resolution": {"status": "matched", "order_step": "api_external_id"},
    }


def test_g9_stage6_refuses_a_specific_record_for_a_class_term() -> None:
    """G9. Base: the result stays ``mapped`` with CHEBI:16057 / C00235 for "prenyl diphosphate"."""
    result = _apply_name_plausibility_gate(_same_named_record(), "prenyl diphosphate", kind="compound")
    assert result["status"] == "unmapped"
    assert result["chosen_rule"] == "novel_compound"
    assert not result.get("mapped_ids")
    assert result["rejected_mapped_ids"] == DMAPP_IDS
    assert result["name_gate"]["reason"] == D109_REASON


def test_g9_stage6_strips_a_specific_identity_already_on_a_class_row() -> None:
    """G9. An earlier pass (or a patch) left DMAPP's ids on the class row. Base: they stay on it."""
    row = {"name": "prenyl diphosphate", "mapped_ids": dict(DMAPP_IDS), "mapping_meta": {}}
    report = _enforce_shipped_identity_names(
        row, _same_named_record(), kind="compound", entity_name="prenyl diphosphate")
    assert report["rejected"] is True
    assert report["removed_ids"] == DMAPP_IDS
    assert row["mapped_ids"] == {}
    assert row["mapping_meta"]["name_gate"]["reason"] == D109_REASON


# --------------------------------------------------------------------------------------------------
# PRESERVATION
# --------------------------------------------------------------------------------------------------


def test_preservation_stage6_answer_on_the_leg_is_byte_identical_to_base() -> None:
    """PRESERVATION. The leg's own Stage-6 answer (committed cache row) was already refused by the name
    gate; its output, which feeds the LLM stages, must not move by a byte."""
    result = _apply_name_plausibility_gate(
        copy.deepcopy(FIXTURE["stage6_cache_answer"]), "prenyl diphosphate", kind="compound")
    assert json.dumps(result, sort_keys=True) == json.dumps(FIXTURE["stage6_gate_at_base"], sort_keys=True)


def test_preservation_a_paper_naming_dmapp_by_its_pathbank_name_still_maps() -> None:
    """PRESERVATION. A paper that names DMAPP itself keeps DMAPP (rule exact_name)."""
    payload = _payload("Dimethylallylpyrophosphate")
    _prefreeze(payload, _rows())
    row = _compound(payload, "Dimethylallylpyrophosphate")
    assert row["name"] == "Dimethylallylpyrophosphate"
    assert row["db_status"] == "matched"
    assert (row["pathbank_compound_id"], row["chebi_id"], row["kegg_id"]) == (874, "16057", "C00235")


def test_preservation_dmapp_named_otherwise_still_resolves_by_identifier() -> None:
    """PRESERVATION. "dimethylallyl diphosphate" carrying KEGG C00235 still resolves and is renamed."""
    payload = _payload("dimethylallyl diphosphate")
    _compound(payload, "dimethylallyl diphosphate")["mapped_ids"] = {"kegg": "C00235"}
    summary, _ = _prefreeze(payload, _rows())
    row = _compound(payload, "dimethylallyl diphosphate")
    assert row["name"] == "Dimethylallylpyrophosphate"
    assert row["pathbank_compound_id"] == 874
    assert summary["rename_map"] == {"dimethylallyl diphosphate": "Dimethylallylpyrophosphate"}
    assert payload["processes"]["reactions"][0]["inputs"][1] == "Dimethylallylpyrophosphate"


def test_preservation_stage6_keeps_dmapp_for_a_paper_that_names_it() -> None:
    """PRESERVATION. The same provider shape under a specific name is kept, as today."""
    record = _same_named_record()
    record["candidates"][0]["name"] = "dimethylallyl diphosphate"
    result = _apply_name_plausibility_gate(record, "dimethylallyl diphosphate", kind="compound")
    assert result["status"] == "mapped"
    assert result["mapped_ids"] == DMAPP_IDS


def test_preservation_a_specific_molecule_synonym_still_renames() -> None:
    """PRESERVATION. An exact PathBank synonym of a specific molecule is still admitted (D-028 rule 2)."""
    glucose = {"id": 77, "name": "D-Glucose", "short_name": "D-Glc", "hmdb_id": "HMDB0000122",
               "kegg_id": "C00031", "chebi_id": "4167", "pubchem_cid": "5793", "synonyms": "dextrose"}
    payload = _payload("dextrose")
    summary, _ = _prefreeze(payload, [glucose])
    row = _compound(payload, "dextrose")
    assert row["name"] == "D-Glucose"
    assert row["pathbank_compound_id"] == 77
    assert summary["rename_map"] == {"dextrose": "D-Glucose"}
