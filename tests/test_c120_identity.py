"""C-120 -- identity-resolution reliability.

Two measured failure mechanisms, replayed as tests through the production
functions. The fixtures are copied from the archived ORCH-732 payloads
(``runs_smoke/2026-09-07_2323``); no paper id, gene symbol, accession, organism
name or taxonomy id introduced here exists in executable production code.

G9 CLASSIFICATION
-----------------
Tests marked ``G9 base failure`` correct pre-existing observable behaviour and
therefore fail *behaviourally* (on a wrong value, never a missing symbol) at the
base SHA ``760c6d72``.

Tests marked ``NEW ACCEPTANCE TEST`` describe capability that did not exist at
the base SHA, or pin a safety property that already holds there. They are not
claimed as base failures.

Mechanism A -- ``PMC11487621``. Three species rows for one organism. Row 2's
name was mangled to an unbalanced fragment by ``_deterministic_species_name``
(A1); row 1 (``B. subtilis``) never resolved a taxonomy at all, although the
same organism was resolved twice elsewhere in the same payload (A2). The PWML
required-field gate then refused the export on
``species_missing_taxonomy`` / ``species_missing_classification``.

Mechanism B -- ``PMC10031235``. A protein whose single candidate was the right
one was refused at rung 4 (the name gate) because the entity name is the gene
symbol family stem and the candidate carries the numbered family member. The
Unknown fallback then fired legitimately as the terminal path. This file fixes
rung 4; it does not reorder or weaken the fallback.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import pytest

from t2pw.mapping.map_ids import (
    _name_gate_verdict,
    backfill_species_taxonomy,
    verify_real_protein_identity,
)
from t2pw.pwml.ir import _deterministic_species_name


# ---------------------------------------------------------------------------
# Fixtures copied verbatim from the archived ORCH-732 payloads.
# ---------------------------------------------------------------------------

#: ``PMC10031235`` -- ``final_mapped``/``gate_fail_report`` ->
#: ``mapping_meta.rejected_candidates[0]``. The single candidate, and the right one.
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
ARCHIVED_PSAT_ENTITY_NAME = "PSAT"
ARCHIVED_PSAT_ORGANISM = "Homo sapiens"
ARCHIVED_PSAT_MAPPED_IDS: Dict[str, Any] = {
    "uniprot": "Q9Y617",
    "pathbank_protein_id": "796",
}


def _psat_name_gate() -> Dict[str, Any]:
    return _name_gate_verdict(
        ARCHIVED_PSAT_ENTITY_NAME,
        candidates=[copy.deepcopy(ARCHIVED_PSAT_CANDIDATE)],
        mapped_ids=dict(ARCHIVED_PSAT_MAPPED_IDS),
        kind="protein",
        organism=ARCHIVED_PSAT_ORGANISM,
        fallback_name=str(ARCHIVED_PSAT_CANDIDATE["name"]),
        source="db",
    )


def _psat_ladder() -> Dict[str, Any]:
    return verify_real_protein_identity(
        ARCHIVED_PSAT_ENTITY_NAME,
        candidates=[copy.deepcopy(ARCHIVED_PSAT_CANDIDATE)],
        mapped_ids=dict(ARCHIVED_PSAT_MAPPED_IDS),
        organism=ARCHIVED_PSAT_ORGANISM,
        source="db",
        resolved_name=str(ARCHIVED_PSAT_CANDIDATE["name"]),
        result={"confidence": ARCHIVED_PSAT_CANDIDATE["score"]},
    )


def _species_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"entities": {"species": copy.deepcopy(rows)}}


def _row(name: str, taxonomy_id: str = "", classification: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name}
    if taxonomy_id:
        row["taxonomy_id"] = taxonomy_id
    if classification:
        row["classification"] = classification
    return row


#: ``PMC11487621`` -- the three species rows, reduced to the fields the backfill
#: reads. Row index 1 is the one that blocked the export.
RESOLVED_SPECIES_ROW = _row("Bacillus subtilis", "1423", "Prokaryote")
RESOLVED_STRAIN_ROW = _row("Bacillus subtilis (strain 168)", "224308", "Prokaryote")
UNRESOLVED_ABBREVIATION_ROW = _row("B. subtilis")


def _species_row(report_payload: Dict[str, Any], index: int) -> Dict[str, Any]:
    return report_payload["entities"]["species"][index]


# ---------------------------------------------------------------------------
# A stub standing in for HttpClient. Injected through the existing ``client``
# parameter of backfill_species_taxonomy -- no boundary is widened for it.
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status_code: int, payload: Optional[Dict[str, Any]] = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Dict[str, Any]:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _StubTaxonomyClient:
    """Answers ONLY for the *expanded* binomial.

    Anything else -- including the raw abbreviated name the existing NCBI pass
    tries first -- gets an empty id list. That is what proves the existing pass
    still runs first and still fails on the abbreviation before the new pass is
    reached. The taxid it hands back is deliberately synthetic so that a row
    carrying it can only have come from this lookup and not from a donor row.
    """

    EXPANDED_TERMS = frozenset({"Bacillus subtilis[Scientific Name]", "Bacillus subtilis"})
    STUB_TAXID = "424242"

    def __init__(self) -> None:
        self.terms: List[str] = []

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, **_: Any) -> _StubResponse:
        params = params or {}
        if url.endswith("esearch.fcgi"):
            term = str(params.get("term") or "")
            self.terms.append(term)
            idlist = [self.STUB_TAXID] if term in self.EXPANDED_TERMS else []
            return _StubResponse(200, {"esearchresult": {"idlist": idlist}})
        if url.endswith("efetch.fcgi"):
            taxid = str(params.get("id") or "")
            xml = (
                "<TaxaSet><Taxon>"
                f"<TaxId>{taxid}</TaxId>"
                "<Division>Bacteria</Division>"
                "<Lineage>cellular organisms; Bacteria; Bacillota</Lineage>"
                "</Taxon></TaxaSet>"
            )
            return _StubResponse(200, None, xml)
        return _StubResponse(404)


# ===========================================================================
# A1 -- _deterministic_species_name must never emit an unbalanced fragment.
# ===========================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        # G9 base failure. At 760c6d72 these return "Bacillus subtilis (strain"
        # and "Escherichia coli (strain" -- the rank-marker test compares
        # "(strain" against the markers and never matches, then the trailing
        # strain-code stripper eats "168)" and stops on "(strain".
        ("Bacillus subtilis (strain 168)", "Bacillus subtilis"),
        ("Escherichia coli (strain K-12)", "Escherichia coli"),
        ("Bacillus subtilis (strain", "Bacillus subtilis"),
    ],
)
def test_g9_a1_strain_bracket_never_leaves_an_unbalanced_fragment(raw: str, expected: str) -> None:
    """G9 base failure (A1). Wrong VALUE at base, correct value at tip."""
    assert _deterministic_species_name(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bacillus subtilis 168", "Bacillus subtilis"),
        ("Herbaspirillum huttiense subsp. huttiense IAM 15032", "Herbaspirillum huttiense"),
        ("Bacillus subtilis", "Bacillus subtilis"),
        ("Homo sapiens", "Homo sapiens"),
        ("Escherichia coli str. K-12 substr. MG1655", "Escherichia coli"),
        ("Bacillus subtilis (natto)", "Bacillus subtilis (natto)"),
        ("", ""),
    ],
)
def test_a1_currently_correct_inputs_are_unchanged(raw: str, expected: str) -> None:
    """NEW ACCEPTANCE TEST (A1 safety). Holds at base and must keep holding."""
    assert _deterministic_species_name(raw) == expected


# ===========================================================================
# A2 -- contextual alias reuse in backfill_species_taxonomy.
# ===========================================================================


def test_g9_a2_tier1_offline_alias_reuse_resolves_the_abbreviation() -> None:
    """G9 base failure (A2 tier 1).

    Fully offline. At 760c6d72 the abbreviated row is left with no taxonomy_id
    at all; at tip it takes the payload's own already-resolved binomial.
    """
    payload = _species_payload([RESOLVED_SPECIES_ROW, UNRESOLVED_ABBREVIATION_ROW])
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)

    row = _species_row(payload, 1)
    assert row.get("taxonomy_id") == "1423"
    assert row.get("classification") == "Prokaryote"
    # Display text is preserved -- normalization is for lookup/identity only.
    assert row["name"] == "B. subtilis"
    meta = row["mapping_meta"]["taxonomy_backfill"]
    assert meta["source"] == "pathway_alias_reuse"
    assert meta["expanded_from"] == "B. subtilis"
    assert meta["expanded_to"] == "Bacillus subtilis"
    assert meta["donors"] == ["Bacillus subtilis"]
    assert report["unresolved"] == []
    assert report["resolved"] == 1
    assert report["alias_reuse"]["resolved"] == 1


def test_g9_a2_tier2_lookup_repair_when_donors_disagree_on_taxonomy() -> None:
    """G9 base failure (A2 tier 2).

    The measured ``PMC11487621`` shape: the same binomial is resolved twice in
    the payload at two different ranks (a species-rank id and a strain-rank id).
    The abbreviation is a species-rank reference, so no strain id may be copied
    onto it; the EXPANDED binomial is resolved through the existing
    ``_ncbi_taxonomy_lookup`` instead, here against an injected stub client.

    At 760c6d72 the row is unresolved -- the existing pass asks only for the raw
    abbreviated name, which the stub refuses.
    """
    client = _StubTaxonomyClient()
    payload = _species_payload(
        [RESOLVED_SPECIES_ROW, UNRESOLVED_ABBREVIATION_ROW, RESOLVED_STRAIN_ROW]
    )
    report = backfill_species_taxonomy(payload, client=client, enable_ncbi=True)

    row = _species_row(payload, 1)
    assert row.get("taxonomy_id") == _StubTaxonomyClient.STUB_TAXID
    assert row.get("classification") == "Prokaryote"
    assert row["name"] == "B. subtilis"
    meta = row["mapping_meta"]["taxonomy_backfill"]
    assert meta["source"] == "pathway_alias_expansion_ncbi"
    assert meta["expanded_from"] == "B. subtilis"
    assert meta["expanded_to"] == "Bacillus subtilis"
    assert report["alias_reuse"]["resolved"] == 1
    # The existing raw-name attempt ran FIRST and failed on the abbreviation.
    assert client.terms[0] == "B. subtilis[Scientific Name]"
    assert "Bacillus subtilis[Scientific Name]" in client.terms
    # Neither already-resolved donor id was copied onto the unqualified name.
    assert row["taxonomy_id"] not in {"1423", "224308"}


def test_a2_tier2_leaves_the_row_unresolved_when_ncbi_is_disabled() -> None:
    """NEW ACCEPTANCE TEST (A2 safety). No network, no invention."""
    payload = _species_payload(
        [RESOLVED_SPECIES_ROW, UNRESOLVED_ABBREVIATION_ROW, RESOLVED_STRAIN_ROW]
    )
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)

    row = _species_row(payload, 1)
    assert "taxonomy_id" not in row
    assert "classification" not in row
    # The audit block is absent because the pass changed nothing.
    assert "alias_reuse" not in report
    assert report["unresolved"] == ["B. subtilis"]


def test_a2_no_compatible_donor_stays_unresolved() -> None:
    """NEW ACCEPTANCE TEST (safety: species over-expansion)."""
    payload = _species_payload([_row("Homo sapiens", "9606", "Eukaryote"), UNRESOLVED_ABBREVIATION_ROW])
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)

    row = _species_row(payload, 1)
    assert "taxonomy_id" not in row
    assert report["unresolved"] == ["B. subtilis"]


def test_a2_unrelated_epithet_never_expands() -> None:
    """NEW ACCEPTANCE TEST (safety: species over-expansion).

    A resolved donor sharing only the genus initial must not lend its taxonomy
    to a different epithet.
    """
    payload = _species_payload([RESOLVED_SPECIES_ROW, _row("B. something")])
    backfill_species_taxonomy(payload, client=None, enable_ncbi=False)
    assert "taxonomy_id" not in _species_row(payload, 1)


def test_a2_ambiguous_genus_expansion_is_fail_closed() -> None:
    """NEW ACCEPTANCE TEST (safety: species collision).

    Two genera share the initial and the epithet, so the abbreviation cannot be
    expanded unambiguously. It must stay unresolved rather than pick one.
    """
    payload = _species_payload(
        [
            RESOLVED_SPECIES_ROW,
            _row("Brevibacillus subtilis", "1425", "Prokaryote"),
            UNRESOLVED_ABBREVIATION_ROW,
        ]
    )
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)

    row = _species_row(payload, 2)
    assert "taxonomy_id" not in row
    assert report["unresolved"] == ["B. subtilis"]


@pytest.mark.parametrize(
    "name",
    [
        "Bs. subtilis",          # multi-letter prefix
        "B.",                    # no epithet
        "Bacillus",              # bare genus
        "B. Subtilis",           # capitalised epithet is not the abbreviation shape
        "B subtilis 168",        # strain-qualified, not a bare abbreviated binomial
    ],
)
def test_a2_only_single_letter_abbreviated_binomials_are_eligible(name: str) -> None:
    """NEW ACCEPTANCE TEST (safety). Never expand an arbitrary abbreviation."""
    payload = _species_payload([RESOLVED_SPECIES_ROW, _row(name)])
    backfill_species_taxonomy(payload, client=None, enable_ncbi=False)
    assert "taxonomy_id" not in _species_row(payload, 1)


def test_a2_is_inert_on_a_payload_whose_species_all_resolve() -> None:
    """NEW ACCEPTANCE TEST (safety: inertness).

    Byte-identical payload before and after. This is the property the reviewer
    checks first.
    """
    payload = _species_payload(
        [RESOLVED_SPECIES_ROW, RESOLVED_STRAIN_ROW, _row("Homo sapiens", "9606", "Eukaryote")]
    )
    before = json.dumps(payload, sort_keys=True)
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)
    after = json.dumps(payload, sort_keys=True)

    assert before == after
    assert report["checked"] == 0
    assert report["resolved"] == 0
    assert report["unresolved"] == []
    # Byte-identical report shape too: the audit block only appears when the
    # pass actually resolved something.
    assert "alias_reuse" not in report


def test_a2_does_not_disturb_an_unresolvable_row_it_cannot_help() -> None:
    """NEW ACCEPTANCE TEST (safety: inertness on the unresolved path)."""
    payload = _species_payload([_row("Novel organism sp. nov.")])
    before = json.dumps(payload, sort_keys=True)
    report = backfill_species_taxonomy(payload, client=None, enable_ncbi=False)

    assert json.dumps(payload, sort_keys=True) == before
    assert report["unresolved"] == ["Novel organism sp. nov."]


# ===========================================================================
# B -- gene-symbol-family rescue in _name_gate_verdict.
# ===========================================================================


def test_g9_b_name_gate_keeps_the_gene_symbol_family_stem() -> None:
    """G9 base failure (B).

    At 760c6d72 this returns verdict ``reject`` / ``no_shared_meaningful_token``:
    ``compared_names`` holds the display name only, and the existing
    ``exact_symbol_identity`` rescue needs the normalized strings to be equal.
    """
    gate = _psat_name_gate()
    assert gate["verdict"] == "keep"
    assert gate["reason"] == "gene_symbol_family_identity"
    assert gate["matched_symbol"] == "PSAT1"


def test_g9_b_full_identity_ladder_verifies_the_archived_row() -> None:
    """G9 base failure (B, full ladder).

    At 760c6d72: ``verified: false``, ``reason: implausible_name_match``,
    ``verification_status: rejected``, ``checks.name: reject``. Rungs 1-3 already
    passed at base; only rung 4 changes.
    """
    verdict = _psat_ladder()
    assert verdict["verified"] is True
    assert verdict["reason"] == "verified_real_protein"
    checks = verdict["checks"]
    assert checks["identifier_resolution"] == "ok"
    assert checks["candidate_evidence"] == "ok"
    assert checks["entity_type"] == "ok"
    assert checks["species"] == "ok"
    assert checks["name"] == "keep"


@pytest.mark.parametrize(
    "entity_name, symbol",
    [
        ("PSAT1", "PSAT2"),   # neither is a prefix of the other
        ("PSAT2", "PSAT1"),
        ("A", "A1"),          # stem too short
        ("AB", "AB1"),        # stem too short
        ("PSAT1", "PSAT12"),  # shorter stem does not end in a letter
        ("PSAT", "PSAT1A"),   # suffix is not a pure run of digits
        ("PSAT", "SAT1"),     # not a prefix
    ],
)
def test_b_symbol_family_precision(entity_name: str, symbol: str) -> None:
    """NEW ACCEPTANCE TEST (safety: symbol-family precision)."""
    candidate = {
        "pathbank_protein_id": 796,
        "name": "Phosphoserine aminotransferase",
        "gene_name": symbol,
        "uniprot": "Q9Y617",
        "organism": "Homo sapiens",
        "taxonomy_id": "9606",
        "score": 0.65,
    }
    gate = _name_gate_verdict(
        entity_name,
        candidates=[candidate],
        mapped_ids={"uniprot": "Q9Y617", "pathbank_protein_id": "796"},
        kind="protein",
        organism="Homo sapiens",
        fallback_name=str(candidate["name"]),
        source="db",
    )
    # ``A`` carries no meaningful token at all, so the gate returns ``skip``
    # rather than ``reject``; either way the family rescue must not have fired.
    assert gate["verdict"] != "keep"
    assert gate["reason"] != "gene_symbol_family_identity"


def test_b_rescue_does_not_apply_to_compounds() -> None:
    """NEW ACCEPTANCE TEST (safety). The rescue is protein-only."""
    candidate = {
        "pathbank_compound_id": 721,
        "name": "Nicotinamide adenine dinucleotide",
        "gene_name": "PSAT1",
        "mapped_ids": {"kegg": "C00003"},
    }
    gate = _name_gate_verdict(
        "PSAT",
        candidates=[candidate],
        mapped_ids={"kegg": "C00003"},
        kind="compound",
        fallback_name=str(candidate["name"]),
        source="db",
    )
    assert gate["verdict"] == "reject"


def test_b_mcr_genes_to_the_human_receptor_is_still_rejected() -> None:
    """NEW ACCEPTANCE TEST (safety, from the run-2026-07-28_0919 regression set).

    The alias 'mcr' does equal a human gene symbol, but the organism disagrees
    and the family rule needs a pure trailing digit run, not a trailing word.
    """
    candidate = {
        "accession": "P08235",
        "name": "Mineralocorticoid receptor",
        "gene_name": ["NR3C2", "MCR", "MLR"],
        "organism": "Homo sapiens",
        "matched_alias": "mcr",
        "alias_source": "literature",
        "score": 0.72,
    }
    gate = _name_gate_verdict(
        "mcr genes",
        candidates=[candidate],
        mapped_ids={"uniprot": "P08235"},
        kind="protein",
        organism="Escherichia coli",
        fallback_name=str(candidate["name"]),
        source="uniprot",
    )
    assert gate["verdict"] == "reject"
    assert gate["reason"] == "no_shared_meaningful_token"


def test_b_wrong_organism_protein_is_refused_at_the_species_rung() -> None:
    """NEW ACCEPTANCE TEST (safety: wrong-organism protein).

    The species rung is rung 3 and runs BEFORE the name gate, which is why the
    new rung-4 rescue can never re-admit a cross-organism symbol collision.
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
        candidates=[candidate],
        mapped_ids={"uniprot": "P10515"},
        organism="Staphylococcus aureus",
        source="uniprot",
        resolved_name=str(candidate["name"]),
        result={"confidence": 0.81},
    )
    assert verdict["verified"] is False
    assert verdict["reason"] == "species_mismatch"
    assert verdict["checks"]["species"] == "mismatch"
    # Rung 4 was never reached, so no name-gate rescue could have run.
    assert "name" not in verdict["checks"]


def test_b_symbol_family_requires_organism_agreement() -> None:
    """NEW ACCEPTANCE TEST (safety).

    Same guard the alias rescue uses: when both sides name an organism and they
    disagree, the family rescue does not fire. (Reached directly at the gate, so
    that the guard is proved on its own rather than by the species rung above.)
    """
    candidate = {
        "accession": "Q9Y617",
        "name": "Phosphoserine aminotransferase",
        "gene_name": "PSAT1",
        "organism": "Homo sapiens",
        "score": 0.65,
    }
    gate = _name_gate_verdict(
        "PSAT",
        candidates=[candidate],
        mapped_ids={"uniprot": "Q9Y617"},
        kind="protein",
        organism="Escherichia coli",
        fallback_name=str(candidate["name"]),
        source="uniprot",
    )
    assert gate["verdict"] == "reject"
    assert gate["reason"] == "no_shared_meaningful_token"


def test_b_protein_with_no_resolvable_identity_still_fails_closed() -> None:
    """NEW ACCEPTANCE TEST (safety: the Unknown sentinel stays reachable).

    Nothing in the pool describes the shipped accession and no evidence source
    is supplied, so the ladder still refuses to claim a verified identity and
    the caller still routes the actor to the Unknown-backed placeholder.
    """
    verdict = verify_real_protein_identity(
        "ZZZ hypothetical",
        candidates=[],
        mapped_ids={"uniprot": "Q9Y617"},
        organism="Homo sapiens",
        source="uniprot",
        resolved_name="",
        result={},
    )
    assert verdict["verified"] is False
    assert verdict["checks"]["candidate_evidence"] != "ok"


def test_b_existing_exact_symbol_rescue_is_unchanged() -> None:
    """NEW ACCEPTANCE TEST (safety). The prior rescue still wins on an exact hit."""
    candidate = {
        "accession": "P0AB58",
        "name": "Inner membrane protein PbgA",
        "gene_name": "yejM",
        "organism": "Escherichia coli",
        "score": 0.9,
    }
    gate = _name_gate_verdict(
        "YejM",
        candidates=[candidate],
        mapped_ids={"uniprot": "P0AB58"},
        kind="protein",
        organism="Escherichia coli",
        fallback_name=str(candidate["name"]),
        source="uniprot",
    )
    assert gate["verdict"] == "keep"
    assert gate["reason"] == "exact_symbol_identity"
