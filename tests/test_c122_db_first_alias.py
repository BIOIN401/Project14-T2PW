"""C-122 -- database-first alias resolution: duplicate collapse, alias memoization.

Three measured defects in ``src/t2pw/mapping/map_ids.py``, replayed through the
production functions. No network: the UniProt HTTP client, the EuropePMC
literature tier and the LLM ``chat`` entrypoint are all stubbed.

G9 CLASSIFICATION
-----------------
``G9 BASE-FAILURE PROOF`` -- corrects pre-existing observable behaviour and
therefore fails *on a value* at base SHA ``9cf64677``, never on a missing symbol.

``NEW-CAPABILITY ACCEPTANCE TEST`` -- describes capability that does not exist at
the base SHA at all (the alias memo). No base failure is claimed for these.

``REGRESSION GUARD`` -- pins a property that already holds at the base SHA. No
base failure is claimed for these either.

Change 1 -- same-gene record duplicates are not biological rivals.
    UniProt answers ``OPCL1`` in *Arabidopsis thaliana* with one reviewed record
    (Q84P21) and two unreviewed duplicates of the same gene (F4HST9,
    A0A1P8AUM2). Rung 6 counted the duplicates as independent rivals, the margin
    fell to 0.03, and a REVIEWED identity was thrown out as
    ``ambiguous_insufficient_margin``.

Change 2 -- stop asking the same alias question repeatedly.
    The tier-3 LLM alias lookup was called twice per resolution and memoized
    nowhere.

Change 3 -- request ``organism_id`` from UniProt and carry it onto the parsed
    candidate as ``taxonomy_id``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from t2pw.mapping import map_ids
from t2pw.mapping.map_ids import (
    MappingCache,
    _ai_protein_synonym_lookup,
    _extract_uniprot_candidates,
    _map_protein_with_strategy,
    map_payload,
    map_protein_uniprot,
    verify_real_protein_identity,
)


# ---------------------------------------------------------------------------
# Offline stubs. Nothing in this file touches the network.
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


class _StubUniProtClient:
    """Answers every UniProtKB search with the same fixture payload."""

    def __init__(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.payload = payload if payload is not None else {"results": []}
        self.queries: List[str] = []
        self.fields: List[str] = []
        self.urls: List[str] = []

    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> _StubResponse:
        query_params = params or {}
        self.urls.append(url)
        if "rest.uniprot.org" in url:
            self.queries.append(str(query_params.get("query") or ""))
            self.fields.append(str(query_params.get("fields") or ""))
            return _StubResponse(copy.deepcopy(self.payload))
        return _StubResponse({}, status_code=404)


def _no_literature_aliases(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    return {"status": "unmapped", "reason": "stubbed", "query": "", "aliases": []}


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

#: UniProtKB search answer for ``OPCL1`` restricted to *Arabidopsis thaliana*,
#: copied field-for-field from the live API on 2026-09-10 (query
#: ``(protein_name:"OPCL1" OR gene:"OPCL1") AND organism_name:"Arabidopsis thaliana"``,
#: fields ``accession,protein_name,gene_names,organism_name,reviewed,organism_id``).
#: One reviewed Swiss-Prot record and two unreviewed TrEMBL records of the SAME
#: gene in the SAME organism.
#:
#: Note where ``OPCL1`` actually sits on the reviewed record: it is a gene
#: *synonym* there, the primary symbol being ``4CLL5``. The collapse has to read
#: the whole gene-symbol set, not just the primary name.
OPCL1_UNIPROT_PAYLOAD: Dict[str, Any] = {
    "results": [
        {
            "primaryAccession": "Q84P21",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Peroxisomal OPC-8:0-CoA ligase 1"}
                },
                "alternativeNames": [
                    {"fullName": {"value": "4-coumarate--CoA ligase isoform 9"}},
                    {"fullName": {"value": "4-coumarate--CoA ligase-like 5"}},
                ],
            },
            "genes": [
                {
                    "geneName": {"value": "4CLL5"},
                    "synonyms": [{"value": "OPCL1"}],
                    "orfNames": [{"value": "F5M15.17"}],
                }
            ],
            "organism": {"scientificName": "Arabidopsis thaliana", "taxonId": 3702},
        },
        {
            "primaryAccession": "F4HST9",
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {
                "submissionNames": [{"fullName": {"value": "OPC-8:0 CoA ligase1"}}]
            },
            "genes": [
                {
                    "geneName": {"value": "OPCL1"},
                    "synonyms": [{"value": "OPC-8:0 CoA ligase1"}],
                    "orfNames": [{"value": "F5M15.17"}, {"value": "F5M15_17"}],
                }
            ],
            "organism": {"scientificName": "Arabidopsis thaliana", "taxonId": 3702},
        },
        {
            "primaryAccession": "A0A1P8AUM2",
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {
                "submissionNames": [{"fullName": {"value": "OPC-8:0 CoA ligase1"}}]
            },
            "genes": [
                {
                    "geneName": {"value": "OPCL1"},
                    "synonyms": [{"value": "OPC-8:0 CoA ligase1"}],
                    "orfNames": [{"value": "F5M15.17"}, {"value": "F5M15_17"}],
                }
            ],
            "organism": {"scientificName": "Arabidopsis thaliana", "taxonId": 3702},
        },
    ]
}

OPCL1_ENTITY_NAME = "OPCL1"
OPCL1_ORGANISM = "Arabidopsis thaliana"


def _opcl1_candidates() -> List[Dict[str, Any]]:
    """The parsed candidate rows, straight from the production parser.

    The accession under test is never written into the test as an input: it is
    read back off whichever row the parser ranks first.
    """

    return _extract_uniprot_candidates(
        copy.deepcopy(OPCL1_UNIPROT_PAYLOAD),
        query_name=OPCL1_ENTITY_NAME,
        organism=OPCL1_ORGANISM,
    )


def _opcl1_verdict() -> Dict[str, Any]:
    candidates = _opcl1_candidates()
    top = candidates[0]
    return verify_real_protein_identity(
        OPCL1_ENTITY_NAME,
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": str(top["accession"])},
        organism=OPCL1_ORGANISM,
        source="uniprot",
        resolved_name=str(top["protein_name"]),
        result={"confidence": top["score"]},
    )


#: ``PMC10031235`` -- the C-120 archived candidate, unchanged.
ARCHIVED_PSAT_CANDIDATE: Dict[str, Any] = {
    "pathbank_protein_id": 796,
    "name": "Phosphoserine aminotransferase",
    "gene_name": "PSAT1",
    "uniprot": "Q9Y617",
    "species_id": 1,
    "score": 0.65,
    "organism": "Homo sapiens",
    "taxonomy_id": "9606",
}


# ---------------------------------------------------------------------------
# A. PSAT / Homo sapiens -> Q9Y617.
# ---------------------------------------------------------------------------


def test_a_psat_family_stem_identity_still_verifies() -> None:
    """REGRESSION GUARD (no base failure claimed).

    This already passes at base SHA ``9cf64677`` through C-120's
    ``gene_symbol_family_identity`` rung. C-122 must not disturb it: a single
    candidate has no rival, so the duplicate collapse must be inert here.
    """

    verdict = verify_real_protein_identity(
        "PSAT",
        candidates=[copy.deepcopy(ARCHIVED_PSAT_CANDIDATE)],
        mapped_ids={"uniprot": "Q9Y617", "pathbank_protein_id": "796"},
        organism="Homo sapiens",
        source="db",
        resolved_name=str(ARCHIVED_PSAT_CANDIDATE["name"]),
        result={"confidence": ARCHIVED_PSAT_CANDIDATE["score"]},
    )

    assert verdict["verified"] is True
    assert verdict["reason"] == "verified_real_protein"
    assert verdict["checks"]["margin"] == "no_competing_candidate"
    assert "collapsed_duplicate_accessions" not in verdict


# ---------------------------------------------------------------------------
# B. OPCL1 / Arabidopsis thaliana -- the Change-1 base failure.
# ---------------------------------------------------------------------------


def test_b_one_gene_three_records_is_not_three_rivals() -> None:
    """G9 BASE-FAILURE PROOF (Change 1).

    At base SHA ``9cf64677`` this returns
    ``verified=False`` / ``reason='ambiguous_insufficient_margin'`` with
    ``margin`` 0.03 and three competing accessions -- a REVIEWED Swiss-Prot
    identity discarded because UniProt stores the same gene three times.

    At tip the two unreviewed records collapse into the reviewed one and the
    identity verifies. The failure is on VALUES (``verified``, ``reason``,
    ``margin``), not on a missing symbol.
    """

    candidates = _opcl1_candidates()
    top = candidates[0]

    # The fixture, not the test, decides which accession is shipped.
    assert top["reviewed"] is True, "the reviewed Swiss-Prot record must rank first"
    assert len(candidates) == 3

    verdict = _opcl1_verdict()

    assert verdict["verified"] is True, verdict
    assert verdict["reason"] == "verified_real_protein"
    assert verdict["identity"] == f"uniprot:{top['accession']}"
    assert verdict["checks"]["margin"] == "no_competing_candidate"

    collapsed = verdict.get("collapsed_duplicate_accessions")
    assert collapsed == sorted(
        str(row["accession"]).casefold() for row in candidates[1:]
    ), verdict
    # Evidence is preserved, never dropped: every pre-existing verdict field is
    # still populated.
    for field in ("checks", "judged_candidate", "judged_candidates", "score", "margin"):
        assert field in verdict


def test_b_collapse_records_the_reason_and_keeps_the_reviewed_record() -> None:
    """G9 BASE-FAILURE PROOF (Change 1, evidence-preservation half).

    At base there is no ``collapsed_duplicate_accessions`` key and the verdict
    reads ``ambiguous_insufficient_margin``; the assertion below on
    ``judged_candidate['reviewed']`` is what makes this a value failure rather
    than a symbol-absence failure -- at base the judged candidate never reaches
    a verified verdict at all.
    """

    verdict = _opcl1_verdict()

    assert verdict["verified"] is True
    judged = verdict["judged_candidate"]
    assert judged["reviewed"] is True
    assert judged["organism"] == OPCL1_ORGANISM
    # The collapsed rows are still listed in the verdict, so a reader can see
    # exactly which records were set aside and why.
    assert len(verdict["judged_candidates"]) == 3


# ---------------------------------------------------------------------------
# C. Paralog safety.
# ---------------------------------------------------------------------------


def _ormdl_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "accession": "Q9P0S3",
            "protein_name": "ORMDL sphingolipid biosynthesis regulator 1",
            "gene_names": ["ORMDL1"],
            "organism": "Homo sapiens",
            "taxonomy_id": "9606",
            "reviewed": True,
            "score": 0.68,
        },
        {
            "accession": "Q53FV1",
            "protein_name": "ORMDL sphingolipid biosynthesis regulator 2",
            "gene_names": ["ORMDL2"],
            "organism": "Homo sapiens",
            "taxonomy_id": "9606",
            "reviewed": True,
            "score": 0.64,
        },
    ]


def test_c_paralogs_with_different_gene_symbols_stay_rivals() -> None:
    """REGRESSION GUARD / SAFETY (no base failure claimed).

    ORMDL1 and ORMDL2 are different proteins in the same organism. They must
    still contest the margin: the duplicate collapse keys on a SHARED gene
    symbol, and an empty intersection is not a match.
    """

    candidates = _ormdl_candidates()
    verdict = verify_real_protein_identity(
        "ORMDL",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": candidates[0]["accession"]},
        organism="Homo sapiens",
        source="uniprot",
        resolved_name=str(candidates[0]["protein_name"]),
        result={"confidence": candidates[0]["score"]},
    )

    assert verdict["verified"] is False, verdict
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert verdict["competing_accessions"] == ["q53fv1"]
    assert not verdict.get("collapsed_duplicate_accessions")


def test_c_empty_gene_symbol_list_is_not_a_match() -> None:
    """REGRESSION GUARD / SAFETY (no base failure claimed).

    Two records that name no gene symbol at all share nothing, so they may not
    be collapsed into one protein on that silence.
    """

    candidates = [
        {
            "accession": "P11111",
            "protein_name": "Widget synthase",
            "gene_names": [],
            "organism": "Homo sapiens",
            "score": 0.70,
        },
        {
            "accession": "P22222",
            "protein_name": "Widget synthase",
            "gene_names": [],
            "organism": "Homo sapiens",
            "score": 0.66,
        },
    ]
    verdict = verify_real_protein_identity(
        "Widget synthase",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": "P11111"},
        organism="Homo sapiens",
        source="uniprot",
        resolved_name="Widget synthase",
        result={"confidence": 0.70},
    )

    assert verdict["verified"] is False, verdict
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert not verdict.get("collapsed_duplicate_accessions")


# ---------------------------------------------------------------------------
# C2. The two shapes REV-C122 round 1 found the first predicate could not tell
#     apart from OPCL1. Both pass at base SHA 9cf64677 and both failed at the
#     round-1 tip, which is what makes them guards rather than decoration.
# ---------------------------------------------------------------------------


def _enzx_style_candidates(reviewed: Any) -> List[Dict[str, Any]]:
    """Two INDEPENDENTLY CLAIMED accessions for one name, 0.02 apart.

    The shape of ``_AMBIGUOUS_CANDIDATES`` in
    ``tests/test_rag_typed_resolution_integrity.py``: both rows assert they ARE
    the entity, in the same organism, with the same gene symbol. ``reviewed`` is
    applied to BOTH rows -- the point is that neither row stands above the other.
    """

    rows: List[Dict[str, Any]] = [
        {
            "accession": "P12345",
            "protein_name": "EnzX",
            "gene_names": ["EnzX"],
            "primary_gene_names": ["EnzX"],
            "organism": "Pseudomonas putida",
            "taxonomy_id": "303",
            "score": 0.92,
        },
        {
            "accession": "Q99999",
            "protein_name": "EnzX",
            "gene_names": ["EnzX"],
            "primary_gene_names": ["EnzX"],
            "organism": "Pseudomonas putida",
            "taxonomy_id": "303",
            "score": 0.90,
        },
    ]
    if reviewed is not None:
        for row in rows:
            row["reviewed"] = reviewed
    return rows


@pytest.mark.parametrize(
    "reviewed, label",
    [
        (None, "neither row declares a review status"),
        (False, "both rows are unreviewed"),
        (True, "both rows are curated Swiss-Prot"),
    ],
)
def test_c2_two_independent_claims_on_one_name_stay_rivals(reviewed: Any, label: str) -> None:
    """REGRESSION GUARD / SAFETY (no base failure claimed).

    Same gene symbol, same organism, and NO Swiss-Prot superset relation between
    them. These are two independent claims to be the same protein, not one
    protein stored twice, and the ladder must confirm NEITHER. This is the
    property ``tests/test_rag_typed_resolution_integrity.py::
    test_an_insufficient_margin_leaves_both_accessions_unwritten`` pins, restated
    on this card's own predicate so a future edit to the predicate breaks here
    first.
    """

    candidates = _enzx_style_candidates(reviewed)
    verdict = verify_real_protein_identity(
        "EnzX",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": candidates[0]["accession"]},
        organism="Pseudomonas putida",
        source="uniprot",
        resolved_name="EnzX",
        result={"confidence": candidates[0]["score"]},
    )

    assert verdict["verified"] is False, (label, verdict)
    assert verdict["reason"] == "ambiguous_insufficient_margin", label
    assert verdict["checks"]["margin"].startswith("insufficient:"), label
    assert verdict["competing_accessions"] == ["q99999"], label
    assert not verdict.get("collapsed_duplicate_accessions"), label


#: Two paralogs that share only the FAMILY SYNONYM. This is the shape
#: ``_extract_uniprot_candidates`` really produces: ``gene_names`` is fed from
#: UniProt gene synonyms as well as gene names, so the two rows' symbol sets
#: intersect at "ORMDL" while their primary symbols differ.
ORMDL_UNIPROT_PAYLOAD: Dict[str, Any] = {
    "results": [
        {
            "primaryAccession": "Q9P0S3",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "ORMDL sphingolipid biosynthesis regulator 1"}
                }
            },
            "genes": [{"geneName": {"value": "ORMDL1"}, "synonyms": [{"value": "ORMDL"}]}],
            "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        },
        {
            "primaryAccession": "Q53FV1",
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {
                "submissionNames": [
                    {"fullName": {"value": "ORMDL sphingolipid biosynthesis regulator 2"}}
                ]
            },
            "genes": [{"geneName": {"value": "ORMDL2"}, "synonyms": [{"value": "ORMDL"}]}],
            "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        },
    ]
}


def test_c2_paralogs_sharing_only_a_family_synonym_stay_rivals() -> None:
    """REGRESSION GUARD / SAFETY (no base failure claimed).

    ORMDL1 (reviewed) and ORMDL2 (unreviewed) in one organism. The Swiss-Prot
    superset relation HOLDS here, so condition 1 does not save this case: it is
    condition 2, the shared symbol having to be a PRIMARY symbol somewhere, that
    keeps two different genes apart.

    Driven through the real parser so the symbol sets are the ones production
    builds, synonyms included, rather than a hand-written pair of singletons.
    """

    candidates = _extract_uniprot_candidates(
        copy.deepcopy(ORMDL_UNIPROT_PAYLOAD), query_name="ORMDL", organism="Homo sapiens"
    )

    # The premise of the test, stated only in terms production already had at
    # base: the two symbol sets really do intersect, and only on the synonym.
    symbol_sets = [{name.casefold() for name in row["gene_names"]} for row in candidates]
    assert symbol_sets[0] & symbol_sets[1] == {"ormdl"}
    assert candidates[0]["reviewed"] is True and candidates[1]["reviewed"] is False

    verdict = verify_real_protein_identity(
        "ORMDL",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": str(candidates[0]["accession"])},
        organism="Homo sapiens",
        source="uniprot",
        resolved_name=str(candidates[0]["protein_name"]),
        result={"confidence": candidates[0]["score"]},
    )

    assert verdict["verified"] is False, verdict
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert verdict["checks"]["margin"].startswith("insufficient:")
    assert not verdict.get("collapsed_duplicate_accessions")


def test_c2_parser_separates_primary_symbols_from_synonyms() -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST.

    ``gene_names`` is ``sorted(set(...))`` and cannot say which symbol UniProt
    listed as the gene's own name; ``primary_gene_names`` is what carries that,
    in declaration order. No base failure is claimed -- the field did not exist.
    """

    ormdl = _extract_uniprot_candidates(
        copy.deepcopy(ORMDL_UNIPROT_PAYLOAD), query_name="ORMDL", organism="Homo sapiens"
    )
    assert [row["primary_gene_names"] for row in ormdl] == [["ORMDL1"], ["ORMDL2"]]
    # The synonym is in gene_names and is NOT a primary symbol on either side.
    for row in ormdl:
        assert "ORMDL" in row["gene_names"]
        assert "ORMDL" not in row["primary_gene_names"]

    opcl1 = _opcl1_candidates()
    # On the live reviewed record OPCL1 is a SYNONYM; 4CLL5 is the primary
    # symbol. On the unreviewed records OPCL1 is the primary symbol. That
    # asymmetry is exactly why the rule asks for a primary symbol on EITHER side.
    by_accession = {row["accession"]: row for row in opcl1}
    assert by_accession["Q84P21"]["primary_gene_names"] == ["4CLL5"]
    assert by_accession["F4HST9"]["primary_gene_names"] == ["OPCL1"]


def test_c2_the_three_sorting_cases_are_decided_by_the_predicate_itself() -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST -- the predicate, stated as a truth table.

    The predicate is new, so no base failure is claimed for this one; it exists
    so a reader can check the rule sorts all three shapes in one place.

    One place a reader can check that the rule sorts all three shapes, without
    having to reconstruct the ladder around it.
    """

    opcl1 = _opcl1_candidates()
    ormdl = _extract_uniprot_candidates(
        copy.deepcopy(ORMDL_UNIPROT_PAYLOAD), query_name="ORMDL", organism="Homo sapiens"
    )
    enzx = _enzx_style_candidates(None)

    # One protein stored three times: reviewed absorbs unreviewed.
    assert map_ids._is_same_protein_record_duplicate(opcl1[0], opcl1[1]) is True
    assert map_ids._is_same_protein_record_duplicate(opcl1[2], opcl1[0]) is False  # wrong direction
    # Two genes of one family sharing a synonym.
    assert map_ids._is_same_protein_record_duplicate(ormdl[0], ormdl[1]) is False
    # Two independent claims on one name.
    assert map_ids._is_same_protein_record_duplicate(enzx[0], enzx[1]) is False


# ---------------------------------------------------------------------------
# D. Wrong-organism safety.
# ---------------------------------------------------------------------------


def test_d_wrong_organism_protein_is_still_refused_at_the_species_rung() -> None:
    """REGRESSION GUARD (preserves ``tests/test_c120_identity.py:582``).

    Candidate P10515 / DLTA / *Homo sapiens*, queried as the entity 'DltA' in
    *Staphylococcus aureus*, must stay refused with ``species_mismatch``. Rung 3
    runs before anything C-122 touches.
    """

    candidate = {
        "accession": "P10515",
        "name": "Dihydrolipoyllysine-residue acetyltransferase",
        "gene_name": "DLTA",
        "organism": "Homo sapiens",
        "taxonomy_id": "9606",
        "score": 0.81,
    }
    verdict = verify_real_protein_identity(
        "DltA",
        candidates=[copy.deepcopy(candidate)],
        mapped_ids={"uniprot": "P10515"},
        organism="Staphylococcus aureus",
        source="uniprot",
        resolved_name=str(candidate["name"]),
        result={"confidence": 0.81},
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "species_mismatch"
    assert verdict["checks"]["species"] == "mismatch"
    assert "name" not in verdict["checks"]


def test_d_same_gene_symbol_in_two_organisms_is_not_collapsed() -> None:
    """REGRESSION GUARD / SAFETY, Change-1 specific (no base failure claimed).

    The request is genus-level ('Escherichia'), so BOTH records survive the
    rival filter's own species screen -- 'same genus' is not 'mismatch'. The
    duplicate collapse must nonetheless refuse to merge them, because
    *Escherichia coli* and *Escherichia fergusonii* are different organisms and
    their proteins are not interchangeable.
    """

    candidates = [
        {
            "accession": "P0A6F3",
            "protein_name": "Glycerol kinase",
            "gene_names": ["GLPK"],
            "organism": "Escherichia coli",
            "taxonomy_id": "562",
            "reviewed": True,
            "score": 0.90,
        },
        {
            "accession": "B7LT03",
            "protein_name": "Glycerol kinase",
            "gene_names": ["GLPK"],
            "organism": "Escherichia fergusonii",
            "taxonomy_id": "564",
            "reviewed": False,
            "score": 0.85,
        },
    ]
    verdict = verify_real_protein_identity(
        "GLPK",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": candidates[0]["accession"]},
        organism="Escherichia",
        source="uniprot",
        resolved_name=str(candidates[0]["protein_name"]),
        result={"confidence": candidates[0]["score"]},
    )

    assert verdict["verified"] is False, verdict
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert verdict["competing_accessions"] == ["b7lt03"]
    assert not verdict.get("collapsed_duplicate_accessions")


def test_d_same_gene_symbol_disagreeing_taxon_ids_is_not_collapsed() -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 3 feeding Change 1).

    Two records naming the same organism string and the same gene symbol but
    DIFFERENT NCBI taxon ids are not one record. Supplying ``organism_id``
    strengthens the safeguard; it may never weaken it.
    """

    candidates = [
        {
            "accession": "P33333",
            "protein_name": "Widget synthase 1",
            "gene_names": ["WDG1"],
            "organism": "Escherichia coli",
            "taxonomy_id": "562",
            "reviewed": True,
            "score": 0.90,
        },
        {
            "accession": "P44444",
            "protein_name": "Widget synthase 1",
            "gene_names": ["WDG1"],
            "organism": "Escherichia coli",
            "taxonomy_id": "83333",
            "reviewed": False,
            "score": 0.85,
        },
    ]
    verdict = verify_real_protein_identity(
        "WDG1",
        candidates=copy.deepcopy(candidates),
        mapped_ids={"uniprot": "P33333"},
        organism="Escherichia coli",
        source="uniprot",
        resolved_name="Widget synthase 1",
        result={"confidence": 0.90},
    )

    assert verdict["verified"] is False, verdict
    assert verdict["reason"] == "ambiguous_insufficient_margin"
    assert not verdict.get("collapsed_duplicate_accessions")


# ---------------------------------------------------------------------------
# E. Genuinely unresolved stays unresolved.
# ---------------------------------------------------------------------------


def test_e_protein_with_no_candidate_stays_unresolved() -> None:
    """REGRESSION GUARD (no base failure claimed).

    Every tier answers nothing. The resolver must say so rather than guess: no
    accession, no ``mapped`` status. C-122 removes one duplicated LLM call from
    this path and must not turn silence into an answer.
    """

    client = _StubUniProtClient({"results": []})

    with patch(
        "t2pw.mapping.map_ids.lookup_literature_protein_aliases",
        side_effect=_no_literature_aliases,
    ), patch("t2pw.mapping.map_ids._ai_protein_synonym_lookup", return_value=[]):
        result = map_protein_uniprot(client, "Zzz hypothetical widget synthase", "Homo sapiens")

    assert result.get("status") != "mapped", result
    assert not str(_as_dict(result.get("mapped_ids")).get("uniprot") or "").strip()
    assert client.queries, "the ladder must still have issued the entity's own name"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# F. Alias cache is species-scoped.
# ---------------------------------------------------------------------------


def test_f_alias_memo_never_reuses_a_human_answer_for_a_plant(tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 2b).

    The memo did not exist at base SHA ``9cf64677`` -- ``_ai_protein_synonym_lookup``
    took no cache argument and stored nothing anywhere. No base failure is
    claimed. What is asserted here is the safety property of the NEW capability:
    the key carries the species, so a human alias decision can never be replayed
    for a plant query.
    """

    cache = MappingCache(tmp_path / "mapping_cache.json", enabled=True)
    prompts: List[str] = []

    def fake_chat(messages: Any, **_kwargs: Any) -> str:
        prompt = str(messages[0]["content"])
        prompts.append(prompt)
        if "Homo sapiens" in prompt:
            return json.dumps({"aliases": [{"alias": "HUMANGENE", "source": "gene_name"}]})
        return json.dumps({"aliases": [{"alias": "PLANTGENE", "source": "gene_name"}]})

    with patch("t2pw.llm.client.chat", side_effect=fake_chat):
        human = _ai_protein_synonym_lookup("CoA ligase 1", "Homo sapiens", cache=cache)
        plant = _ai_protein_synonym_lookup("CoA ligase 1", "Arabidopsis thaliana", cache=cache)
        human_again = _ai_protein_synonym_lookup("CoA ligase 1", "Homo sapiens", cache=cache)

    keys = sorted(k for k in cache.data["proteins"] if k.startswith("alias-v1::"))
    assert keys == [
        "alias-v1::coa ligase 1::arabidopsis thaliana",
        "alias-v1::coa ligase 1::homo sapiens",
    ], keys

    assert [entry["alias"] for entry in human] == ["HUMANGENE"]
    assert [entry["alias"] for entry in plant] == ["PLANTGENE"]
    assert human_again == human
    # Two distinct questions were asked; the repeat was served from the memo.
    assert len(prompts) == 2, prompts


def test_f_negative_alias_answer_is_memoized_too(tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 2b).

    'No aliases' is the common answer and the whole point of the memo. It must
    be stored, not re-asked.
    """

    cache = MappingCache(tmp_path / "mapping_cache.json", enabled=True)
    calls: List[str] = []

    def fake_chat(messages: Any, **_kwargs: Any) -> str:
        calls.append(str(messages[0]["content"]))
        return json.dumps({"aliases": []})

    with patch("t2pw.llm.client.chat", side_effect=fake_chat):
        first = _ai_protein_synonym_lookup("Zzz widget synthase", "Homo sapiens", cache=cache)
        second = _ai_protein_synonym_lookup("Zzz widget synthase", "Homo sapiens", cache=cache)

    assert first == []
    assert second == []
    assert len(calls) == 1, calls
    assert "alias-v1::zzz widget synthase::homo sapiens" in cache.data["proteins"]


# ---------------------------------------------------------------------------
# G. Call count (card section 14 evidence).
# ---------------------------------------------------------------------------


def _resolve_same_protein_twice(tmp_path: Path) -> int:
    """Resolve one (name, organism) twice in one run; return the LLM call count.

    The two rows differ only by ``pathbank_protein_id``, so the resolver's own
    result cache key differs and both resolutions really run the ladder. The
    body is IDENTICAL at base and at tip -- only the counted value moves.
    """

    cache = MappingCache(tmp_path / "mapping_cache.json", enabled=True)
    client = _StubUniProtClient({"results": []})
    calls: List[str] = []

    def counting_chat(messages: Any, **_kwargs: Any) -> str:
        calls.append(str(messages[0]["content"]))
        return json.dumps({"aliases": []})

    with patch(
        "t2pw.mapping.map_ids.lookup_literature_protein_aliases",
        side_effect=_no_literature_aliases,
    ), patch("t2pw.llm.client.chat", side_effect=counting_chat):
        for pathbank_id in (111, 222):
            _map_protein_with_strategy(
                id_source="api",
                db=None,
                client=client,
                cache=cache,
                name="Zzz hypothetical widget synthase",
                organism="Homo sapiens",
                protein_row={
                    "name": "Zzz hypothetical widget synthase",
                    "pathbank_protein_id": pathbank_id,
                },
            )

    return len(calls)


def test_g_repeated_alias_question_costs_one_llm_call(tmp_path: Path) -> None:
    """G9 BASE-FAILURE PROOF (Change 2a + 2b).

    Base SHA ``9cf64677``: 4 LLM calls -- two resolutions, and each one asks
    ``_ai_protein_synonym_lookup`` twice (once with the organism, once without).

    Tip: 1 LLM call -- the organism-less duplicate is gone and the negative
    answer is memoized under ``alias-v1::<name>::<organism>``.

    The failure at base is on the VALUE 4 != 1.
    """

    assert _resolve_same_protein_twice(tmp_path) == 1


# ---------------------------------------------------------------------------
# Change 3 -- the taxonomy id is requested and carried.
# ---------------------------------------------------------------------------


def test_change3_uniprot_query_requests_the_organism_id_field() -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 3).

    ``organism_id`` was never asked for at base, so the taxon id UniProt already
    holds never reached a candidate row.
    """

    client = _StubUniProtClient({"results": []})

    with patch(
        "t2pw.mapping.map_ids.lookup_literature_protein_aliases",
        side_effect=_no_literature_aliases,
    ), patch("t2pw.mapping.map_ids._ai_protein_synonym_lookup", return_value=[]):
        map_protein_uniprot(client, "OPCL1", OPCL1_ORGANISM)

    assert client.fields, "the ladder must have issued at least one query"
    for fields in client.fields:
        assert "organism_id" in fields, fields
        # The pre-existing fields are all still requested.
        for legacy in ("accession", "protein_name", "gene_names", "organism_name", "reviewed"):
            assert legacy in fields, fields


def test_change3_parsed_candidate_carries_the_taxonomy_id() -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 3)."""

    candidates = _opcl1_candidates()

    assert candidates, "fixture must parse"
    for row in candidates:
        assert row["taxonomy_id"] == "3702", row


# ---------------------------------------------------------------------------
# H. The Phase-2 fallback memo must not store a transport failure.
# ---------------------------------------------------------------------------


def _run_fallback_leg(tmp_path: Path, api_result: Dict[str, Any]) -> Dict[str, Any]:
    """Drive map_payload so the Phase-2 api_uniprot_fallback leg runs once.

    ``_map_protein_with_strategy`` is pinned to ``unmapped`` so the protein
    reaches the fallback, and ``map_protein_uniprot`` is pinned to whatever
    answer the caller wants to test. Returns the persisted cache file content.
    """

    cache_path = tmp_path / "id_mapping_cache.json"
    payload = {
        "entities": {
            "proteins": [{"name": "Zzz widget synthase", "organism": "Homo sapiens"}],
            "compounds": [],
        },
        "processes": {"reactions": []},
    }

    with patch(
        "t2pw.mapping.map_ids.hydrate_species_references",
        return_value={"hydrated": 0, "matched": 0, "novel": 0},
    ), patch(
        "t2pw.mapping.map_ids._rewrite_reaction_protein_enzymes_to_complexes",
        return_value={"summary": {}, "actions": []},
    ), patch(
        "t2pw.mapping.map_ids._map_protein_with_strategy",
        return_value={
            "status": "unmapped",
            "reason": "no_match",
            "source": "api",
            "provider": "UniProt",
            "candidates": [],
        },
    ), patch(
        "t2pw.mapping.map_ids.map_protein_uniprot",
        return_value=copy.deepcopy(api_result),
    ):
        map_payload(payload, cache_path=cache_path, id_source="api")

    return json.loads(cache_path.read_text(encoding="utf-8"))


def _fallback_keys(cache_data: Dict[str, Any]) -> List[str]:
    return [k for k in (cache_data.get("proteins") or {}) if k.startswith("api-fallback-v1::")]


def test_h_a_transport_failure_is_never_written_to_the_persistent_cache(tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 2c, safety half).

    The fallback memo is new on this branch, so no base failure is claimed.
    What is pinned is that it fails OPEN: ``data/id_mapping_cache.json`` outlives
    the run, and one unreachable UniProt must not become a permanent silent
    "no such protein" for that (name, organism). PRODUCT_CONTRACT S8 -- a lookup
    failure is not evidence that an accession is false.
    """

    cache_data = _run_fallback_leg(
        tmp_path,
        {
            "status": "unmapped",
            "reason": "network_error:HTTP request failed after retries",
            "provider": "UniProt",
            "candidates": [],
        },
    )

    assert _fallback_keys(cache_data) == [], cache_data.get("proteins")


def test_h_a_real_negative_answer_is_written(tmp_path: Path) -> None:
    """NEW-CAPABILITY ACCEPTANCE TEST (Change 2c).

    The counterpart: a lookup that COMPLETED and found nothing is a real answer
    and is memoized, which is the repetition this leg exists to stop.
    """

    cache_data = _run_fallback_leg(
        tmp_path,
        {
            "status": "unmapped",
            "reason": "no_match",
            "provider": "UniProt",
            "candidates": [],
        },
    )

    keys = _fallback_keys(cache_data)
    assert keys == ["api-fallback-v1::zzz widget synthase::homo sapiens"], keys
    assert cache_data["proteins"][keys[0]]["reason"] == "no_match"
