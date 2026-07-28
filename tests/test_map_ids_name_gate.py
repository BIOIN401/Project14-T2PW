"""Name-plausibility gate: stop shipping confidently wrong identifiers.

Every fixture in this file is copied from the real payloads of run
2026-07-28_0919, which passed every gate with 0 errors while shipping:

  * 'PhoP' (a DNA-binding response regulator) as KEGG C00003 / CHEBI:15846 /
    CAS 53-84-9 / PathBank compound 721, i.e. NAD+, resolution status "matched";
  * 'pmrHFIJKLM' (an LPS-modification operon) as UniProt P03023 "Lactose operon
    repressor", reached by degrading the query ladder to the literature alias
    'operon';
  * 'mcr genes' as UniProt P08235, the HUMAN mineralocorticoid receptor, on 10 of
    that paper's 14 reactions -- while its mapping_meta.resolved_name described
    P24200, a different protein entirely.

and, in the same files, correctly resolving 'MCR-1' to A0A0R6L508
"Phosphatidylethanolamine transferase Mcr-1", which must survive untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import (  # noqa: E402
    _apply_name_plausibility_gate,
    _is_generic_query_name,
    _map_compound_with_strategy,
    _map_protein_with_strategy,
    _name_gate_verdict,
    _name_match_is_plausible,
    map_payload,
    map_protein_uniprot,
)


class _MemoryCache:
    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    def get(self, namespace: str, key: str) -> Dict[str, Any] | None:
        return self.rows.get((namespace, key))

    def set(self, namespace: str, key: str, value: Dict[str, Any]) -> None:
        self.rows[(namespace, key)] = value


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._payload


class _RecordingUniProtClient:
    """Answers nothing, but records every query the ladder issues."""

    def __init__(self) -> None:
        self.queries: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None, **_: Any) -> _FakeResponse:
        params = params or {}
        query = str(params.get("query") or "")
        if "uniprot" in url:
            self.queries.append(query)
            return _FakeResponse({"results": []})
        if "europepmc" in url:
            return _FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "title": "Colistin resistance",
                                "abstractText": "The pmrHFIJKLM (or operon) locus modifies lipid A.",
                                "source": "PMC",
                                "id": "1",
                            }
                        ]
                    }
                }
            )
        return _FakeResponse({}, text="")


def _mcr_genes_api_result() -> Dict[str, Any]:
    """The real UniProt result run 2026-07-28_0919 produced for 'mcr genes'."""
    return {
        "status": "mapped",
        "reason": "best_effort_fallback",
        "provider": "UniProt",
        "source": "api",
        "mapped_ids": {"uniprot": "P24200"},
        "confidence": 0.85,
        "chosen_rule": "best_effort_fallback",
        "best_effort": True,
        "resolved_name": "Type IV methyl-directed restriction enzyme EcoKMcrA",
        "candidates": [
            {
                "accession": "P24200",
                "protein_name": "Type IV methyl-directed restriction enzyme EcoKMcrA",
                "gene_names": ["mcrA", "rglA"],
                "organism": "Escherichia coli (strain K12)",
                "reviewed": True,
                "score": 0.85,
                "matched_query": "mcrA",
                "alias_source": "synonym",
                "matched_alias": "mcrA",
            },
            {
                "accession": "P08235",
                "protein_name": "Mineralocorticoid receptor",
                "gene_names": ["MCR", "MLR", "NR3C2"],
                "organism": "Homo sapiens",
                "reviewed": True,
                "score": 0.56,
                "matched_query": "mcr",
                "alias_source": "gene_name",
                "matched_alias": "mcr",
            },
        ],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }


def _mcr1_api_result() -> Dict[str, Any]:
    """The real (correct) UniProt result for 'MCR-1' in the same run."""
    return {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "mapped_ids": {"uniprot": "A0A0R6L508"},
        "confidence": 1.0,
        "chosen_rule": "top_unique_candidate",
        "resolved_name": "Phosphatidylethanolamine transferase Mcr-1",
        "candidates": [
            {
                "accession": "A0A0R6L508",
                "protein_name": "Phosphatidylethanolamine transferase Mcr-1",
                "gene_names": ["mcr-1", "mcr1"],
                "organism": "Escherichia coli",
                "reviewed": True,
                "score": 1.0,
                "matched_query": "MCR-1",
                "alias_source": "primary_name",
            }
        ],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }


def _phop_compound_db_result() -> Dict[str, Any]:
    """The real PathBank result for the protein name 'PhoP' routed to compounds."""
    return {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "mapped_ids": {
            "hmdb": "HMDB0000902",
            "kegg": "C00003",
            "chebi": "CHEBI:15846",
            "pubchem": "5893",
            "cas": "53-84-9",
            "biocyc": "NAD",
            "chemspider": "5682",
            "pathbank_compound_id": "721",
        },
        "pathbank_compound_id": 721,
        "confidence": 1.0,
        "chosen_rule": "direct_id_match:kegg",
        "candidates": [
            {
                "pathbank_compound_id": 721,
                "name": "NAD",
                "short_name": "NAD",
                "matched_on": "kegg",
                "score": 1.0,
                "mapped_ids": {
                    "hmdb": "HMDB0000902",
                    "kegg": "C00003",
                    "chebi": "CHEBI:15846",
                    "pubchem": "5893",
                    "cas": "53-84-9",
                    "biocyc": "NAD",
                    "chemspider": "5682",
                },
            }
        ],
        "resolution": {"status": "matched", "order_step": "external_id:direct_id_match:kegg"},
    }


# ── The five name pairs named in the fix brief ───────────────────────────────


def test_mcr1_survives_the_name_gate() -> None:
    """'MCR-1' -> "Phosphatidylethanolamine transferase Mcr-1" is the correct
    match of run 2026-07-28_0919 and shares 'mcr'/'mcr1'."""
    assert _name_match_is_plausible("MCR-1", "Phosphatidylethanolamine transferase Mcr-1") is True


def test_mcr_genes_does_not_match_the_human_mineralocorticoid_receptor() -> None:
    """P08235 shipped as the enzyme of 10 of 14 reactions in PMC13278307."""
    assert _name_match_is_plausible("mcr genes", "Mineralocorticoid receptor") is False


def test_pmrhfijklm_does_not_match_the_lactose_operon_repressor() -> None:
    """'operon' is the only word the two names share and it is generic."""
    assert _name_match_is_plausible("pmrHFIJKLM", "Lactose operon repressor") is False
    assert _name_match_is_plausible("pmrHFIJKLM operon", "Lactose operon repressor") is False


def test_phop_does_not_match_nad() -> None:
    """A response regulator protein resolved to PathBank compound 721 (NAD+)."""
    assert _name_match_is_plausible("PhoP", "NAD") is False


def test_gate_never_rejects_when_there_is_nothing_to_compare() -> None:
    """Absence of evidence is not evidence of a wrong match."""
    assert _name_match_is_plausible("MCR-1", "") is True
    assert _name_match_is_plausible("", "Mineralocorticoid receptor") is True
    assert _name_match_is_plausible("operon", "Lactose operon repressor") is True


@pytest.mark.parametrize(
    "left,right",
    [
        ("pmrD", "Signal transduction protein PmrD"),
        ("LPS", "LPS with O-antigen"),
        ("Kdo", "KDO"),
        ("UDP-GlcNAc", "UDP-N-ACETYL-D-GLUCOSAMINE"),
        ("acetyl-CoA carboxylase (ACC)", "Biotin carboxyl carrier protein of acetyl-CoA carboxylase"),
        ("FtsH", "ATP-dependent zinc metalloprotease FtsH"),
    ],
)
def test_correct_matches_from_the_same_run_stay_plausible(left: str, right: str) -> None:
    assert _name_match_is_plausible(left, right) is True


# ── Generic-alias degradation ────────────────────────────────────────────────


def test_bare_generic_words_are_not_usable_query_names() -> None:
    for word in ["operon", "gene", "genes", "protein", "complex", "enzyme", "transferase", "the operon"]:
        assert _is_generic_query_name(word) is True
    for word in ["mcr", "pmrHFIJKLM", "MCR-1", "mcr genes"]:
        assert _is_generic_query_name(word) is False


def test_query_ladder_never_issues_a_bare_operon_query() -> None:
    """Run 2026-07-28_0919 issued '(protein_name:"operon" OR gene:"operon")' for
    pmrHFIJKLM and took the lactose operon repressor from the answer."""
    client = _RecordingUniProtClient()

    map_protein_uniprot(client, "pmrHFIJKLM", "Escherichia coli", aliases=[{"alias": "operon", "source": "literature_alias"}])

    assert client.queries, "the ladder must still issue the entity's own name"
    assert any("pmrHFIJKLM" in query for query in client.queries)
    assert not any("operon" in query for query in client.queries)


# ── Result-level gate ────────────────────────────────────────────────────────


def test_uniprot_result_naming_a_different_protein_becomes_novel() -> None:
    result = _apply_name_plausibility_gate(
        _mcr_genes_api_result(), "mcr genes", kind="protein", organism="Escherichia coli"
    )

    assert result["status"] == "unmapped"
    assert result["resolution"] == {
        "status": "novel",
        "issue": "implausible_name_match",
        "order_step": "name_plausibility_gate",
    }
    assert result["chosen_rule"] == "novel_protein"
    assert result.get("mapped_ids") is None
    # Nothing may promote the refused accession back to a match on a later pass.
    assert result["candidates"] == []
    assert result["rejected_mapped_ids"] == {"uniprot": "P24200"}


def test_correct_mcr1_result_is_untouched_by_the_gate() -> None:
    result = _apply_name_plausibility_gate(
        _mcr1_api_result(), "MCR-1", kind="protein", organism="Escherichia coli"
    )

    assert result["status"] == "mapped"
    assert result["mapped_ids"] == {"uniprot": "A0A0R6L508"}
    assert result["name_gate"]["verdict"] == "keep"


def test_protein_name_routed_to_the_compound_mapper_does_not_ship_nad() -> None:
    """The compound-vs-protein routing defect is out of scope here; shipping NAD+
    as PhoP is not."""
    cache = _MemoryCache()

    class _Db:
        last_error = ""

        def available(self) -> bool:
            return True

        def map_compound_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
            return _phop_compound_db_result()

    result = _map_compound_with_strategy(
        id_source="hybrid",
        db=_Db(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        name="PhoP",
        compound_row={"name": "PhoP", "mapped_ids": {"kegg": "C00003"}},
    )

    assert result["status"] == "unmapped"
    assert result["chosen_rule"] == "novel_compound"
    assert result["resolution"]["issue"] == "implausible_name_match"
    assert result["name_gate"]["resolved_name"] == "NAD"


def test_pathbank_protein_row_is_exempt_from_the_name_gate() -> None:
    """PathBank names a protein by function and stores the legacy gene symbol:
    'LpxL' -> P0ACV0 "Lipid A biosynthesis lauroyl acyltransferase" (gene htrB) is
    correct and must not be rejected for sharing no word."""
    db_result = {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "mapped_ids": {"uniprot": "P0ACV0", "pathbank_protein_id": "6214"},
        "pathbank_protein_id": 6214,
        "confidence": 1.08,
        "chosen_rule": "direct_id_match:uniprot_id",
        "candidates": [
            {
                "pathbank_protein_id": 6214,
                "name": "Lipid A biosynthesis lauroyl acyltransferase",
                "uniprot": "P0ACV0",
                "gene_name": "htrB",
                "species_id": 3,
                "matched_on": "uniprot_id",
                "score": 1.08,
            }
        ],
        "resolution": {"status": "matched", "order_step": "uniprot"},
    }

    result = _apply_name_plausibility_gate(
        db_result, "LpxL", kind="protein", organism="Escherichia coli"
    )

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "P0ACV0"


def test_renamed_protein_survives_through_exact_gene_symbol_identity() -> None:
    """'YejM' -> UniProt "Inner membrane protein PbgA": the display names share
    nothing, but the entity name is exactly one of the entry's gene symbols."""
    verdict = _name_gate_verdict(
        "YejM",
        candidates=[
            {
                "accession": "P0AD27",
                "protein_name": "Inner membrane protein PbgA",
                "gene_names": ["pbgA", "yejM"],
                "organism": "Escherichia coli",
                "score": 0.85,
            }
        ],
        mapped_ids={"uniprot": "P0AD27"},
        kind="protein",
        organism="Escherichia coli",
    )

    assert verdict["verdict"] == "keep"
    assert verdict["reason"] == "exact_symbol_identity"


def test_literature_alias_match_survives_when_the_alias_is_the_entrys_gene() -> None:
    """The ObiH/ObaG case of tests/test_map_ids.py: a non-generic alias that is
    exactly a gene symbol of the matched entry, in the requested organism."""
    verdict = _name_gate_verdict(
        "ObaG",
        candidates=[
            {
                "accession": "A0A1X9LWZ7",
                "protein_name": "Threonine aldolase",
                "gene_names": ["obiH"],
                "organism": "Pseudomonas fluorescens",
                "matched_alias": "ObiH",
                "alias_source": "literature_alias",
                "score": 0.8,
            }
        ],
        mapped_ids={"uniprot": "A0A1X9LWZ7"},
        kind="protein",
        organism="Pseudomonas fluorescens",
    )

    assert verdict["verdict"] == "keep"
    assert verdict["reason"] == "alias_corroborated_by_identity"


def test_alias_rescue_does_not_cross_organisms() -> None:
    """'mcr' really is the human gene symbol MCR -- which is exactly why the
    alias rescue requires the organism to agree."""
    verdict = _name_gate_verdict(
        "mcr genes",
        candidates=_mcr_genes_api_result()["candidates"],
        mapped_ids={"uniprot": "P08235"},
        kind="protein",
        organism="Escherichia coli",
    )

    assert verdict["verdict"] == "reject"
    assert verdict["resolved_name"] == "Mineralocorticoid receptor"


# ── Honest audit trail, end to end through map_payload ───────────────────────


def _mcr_genes_payload() -> Dict[str, Any]:
    """An 'mcr genes' row that already carries P08235 from an earlier mapping
    pass, exactly as run 2026-07-28_0919 handed it to the second pass."""
    return {
        "metadata": {"organism": "Escherichia coli"},
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562"}],
            "compounds": [],
            "proteins": [
                {
                    "name": "mcr genes",
                    "species": "Escherichia coli",
                    "organism": "Escherichia coli",
                    "mapped_ids": {"uniprot": "P08235"},
                }
            ],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _run_map_payload(payload: Dict[str, Any], tmp_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    with patch(
        "t2pw.mapping.map_ids.hydrate_species_references",
        return_value={"hydrated": 0, "matched": 0, "novel": 0},
    ), patch(
        "t2pw.mapping.map_ids._rewrite_reaction_protein_enzymes_to_complexes",
        return_value={"summary": {}, "actions": []},
    ), patch(
        "t2pw.mapping.map_ids._map_protein_with_strategy", return_value=result
    ), patch(
        "t2pw.mapping.map_ids.map_protein_uniprot",
        return_value={"status": "unmapped", "reason": "no_match", "candidates": []},
    ):
        return map_payload(
            payload,
            cache_path=tmp_path / "id_mapping_cache.json",
            id_source="api",
            use_cache=False,
            allow_complex_wrapper_creation=False,
        )


def test_shipped_accession_is_the_one_the_audit_trail_describes(tmp_path: Path) -> None:
    """Run 2026-07-28_0919 shipped mapped_ids {"uniprot": "P08235"} together with
    resolved_name "Type IV methyl-directed restriction enzyme EcoKMcrA", which is
    P24200. _merge_mapped_ids keeps the identifier an earlier pass already wrote,
    so the metadata described a match that never shipped."""
    result = _run_map_payload(_mcr_genes_payload(), tmp_path, _mcr_genes_api_result())
    protein = result["payload"]["entities"]["proteins"][0]
    meta = protein["mapping_meta"]

    assert meta["resolved_name"] == "Mineralocorticoid receptor"
    assert meta["mapped_id_conflict"]["shipped"] == "P08235"
    assert meta["mapped_id_conflict"]["resolver_chosen"] == "P24200"


def test_implausible_shipped_accession_is_pulled_back_off_the_row(tmp_path: Path) -> None:
    result = _run_map_payload(_mcr_genes_payload(), tmp_path, _mcr_genes_api_result())
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["mapped_ids"] == {}
    assert protein["mapping_meta"]["resolution"]["status"] == "novel"
    assert protein["mapping_meta"]["resolution"]["issue"] == "implausible_name_match"
    assert protein["mapping_meta"]["rejected_mapped_ids"] == {"uniprot": "P08235"}
    # The candidates are moved aside so the next pass cannot promote them back.
    assert protein["mapping_meta"]["candidates"] == []
    assert result["report"]["summary"]["name_gate_rejections"] == 1
    assert result["report"]["summary"]["proteins_mapped"] == 0

    log_row = result["report"]["entities"][0]
    assert log_row["status"] == "unmapped"
    assert log_row["reason"] == "implausible_name_match"
    assert log_row["resolution_status"] == "novel"


def test_correct_protein_is_not_disturbed_by_map_payload_enforcement(tmp_path: Path) -> None:
    payload = _mcr_genes_payload()
    payload["entities"]["proteins"] = [
        {"name": "MCR-1", "species": "Escherichia coli", "organism": "Escherichia coli"}
    ]

    result = _run_map_payload(payload, tmp_path, _mcr1_api_result())
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["mapped_ids"] == {"uniprot": "A0A0R6L508"}
    assert protein["mapping_meta"]["resolved_name"] == "Phosphatidylethanolamine transferase Mcr-1"
    assert protein["mapping_meta"]["resolution"]["status"] == "matched"
    assert result["report"]["summary"]["name_gate_rejections"] == 0
    assert result["report"]["summary"]["proteins_mapped"] == 1


def test_strategy_level_rejection_still_strips_a_stale_identifier(tmp_path: Path) -> None:
    """Second-pass shape: the resolver already refused, but the row carries an
    identifier the first pass wrote. It must not survive on the row."""
    from t2pw.mapping.map_ids import _novel_result_from_name_gate

    rejected = _novel_result_from_name_gate(
        _mcr_genes_api_result(),
        kind="protein",
        gate={"verdict": "reject", "reason": "no_shared_meaningful_token"},
    )

    result = _run_map_payload(_mcr_genes_payload(), tmp_path, rejected)
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["mapped_ids"] == {}
    assert protein["mapping_meta"]["resolution"]["status"] == "novel"


def test_db_protein_strategy_result_keeps_its_identifier(tmp_path: Path) -> None:
    """A PathBank protein row is exempt, so LpxL keeps P0ACV0 end to end."""
    payload = _mcr_genes_payload()
    payload["entities"]["proteins"] = [
        {"name": "LpxL", "species": "Escherichia coli", "organism": "Escherichia coli"}
    ]
    db_result = {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "mapped_ids": {"uniprot": "P0ACV0", "pathbank_protein_id": "6214"},
        "pathbank_protein_id": 6214,
        "confidence": 1.08,
        "chosen_rule": "direct_id_match:uniprot_id",
        "candidates": [
            {
                "pathbank_protein_id": 6214,
                "name": "Lipid A biosynthesis lauroyl acyltransferase",
                "uniprot": "P0ACV0",
                "gene_name": "htrB",
                "species_id": 3,
                "score": 1.08,
            }
        ],
        "resolution": {"status": "matched", "order_step": "uniprot"},
    }

    result = _run_map_payload(payload, tmp_path, db_result)
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["mapped_ids"]["uniprot"] == "P0ACV0"
    assert protein["mapping_meta"]["resolution"]["status"] == "matched"
    assert result["report"]["summary"]["name_gate_rejections"] == 0


def test_protein_strategy_gate_routes_to_novel_without_a_new_state() -> None:
    """The resolver funnel must emit the resolver's existing 'novel' resolution,
    not an invented status."""
    cache = _MemoryCache()

    class _NovelDb:
        last_error = ""

        def available(self) -> bool:
            return True

        def map_protein_row(self, row: Dict[str, Any], species: str) -> Dict[str, Any]:
            return {
                "status": "unmapped",
                "reason": "novel_protein",
                "provider": "PathBankDB",
                "source": "db",
                "candidates": [],
                "resolution": {"status": "novel", "issue": "no_db_candidates"},
            }

    with patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=_mcr_genes_api_result()):
        result = _map_protein_with_strategy(
            id_source="hybrid",
            db=_NovelDb(),  # type: ignore[arg-type]
            client=object(),  # type: ignore[arg-type]
            cache=cache,  # type: ignore[arg-type]
            name="mcr genes",
            organism="Escherichia coli",
            protein_row={"name": "mcr genes", "species": "Escherichia coli"},
        )

    assert result["status"] == "unmapped"
    assert result["resolution"]["status"] == "novel"
    assert result["chosen_rule"] == "novel_protein"
