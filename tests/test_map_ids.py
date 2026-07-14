from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import (  # noqa: E402
    PathBankDbResolver,
    _europepmc_full_text,
    _extract_aliases_from_literature_text,
    _extract_uniprot_candidates,
    _map_protein_with_strategy,
    _protein_alias_entries,
    _reconcile_components_against_local_proteins,
    _rewrite_reaction_protein_enzymes_to_complexes,
    map_payload,
    map_protein_uniprot,
    route_entity_for_mapping,
)


def _make_resolver() -> PathBankDbResolver:
    resolver = PathBankDbResolver.__new__(PathBankDbResolver)
    resolver.host = "localhost"
    resolver.port = 3306
    resolver.user = "test"
    resolver.password = ""
    resolver.schema = "pathbank"
    resolver.connect_timeout = 6
    resolver.read_timeout = 20
    resolver.write_timeout = 20
    resolver._driver = object()
    resolver._conn = None
    resolver.last_error = ""
    return resolver


def _unmapped(reason: str = "no_db_match") -> Dict[str, Any]:
    return {
        "status": "unmapped",
        "reason": reason,
        "provider": "PathBankDB",
        "source": "db",
        "confidence": 0.0,
        "chosen_rule": "",
        "candidates": [],
    }


class _MemoryCache:
    def __init__(self) -> None:
        self.rows: Dict[tuple[str, str], Dict[str, Any]] = {}

    def get(self, namespace: str, key: str) -> Dict[str, Any] | None:
        return self.rows.get((namespace, key))

    def set(self, namespace: str, key: str, value: Dict[str, Any]) -> None:
        self.rows[(namespace, key)] = value


class _AvailableDb:
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


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._payload


class _AliasUniProtClient:
    def __init__(self) -> None:
        self.queries: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None) -> _FakeResponse:
        params = params or {}
        query = str(params.get("query") or "")
        self.queries.append(query)
        if "fullTextXML" in url:
            return _FakeResponse(
                {},
                text="<article><body>The TTA known as ObiH (or ObaG) was discovered in obafluorin biosynthesis.</body></article>",
            )
        if "europepmc" in url:
            return _FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "title": "Discovery of L-threonine transaldolases",
                                "abstractText": "Beta-hydroxy amino acid biosynthesis.",
                                "source": "PMC",
                                "id": "10495429",
                                "pmcid": "PMC10495429",
                            }
                        ]
                    }
                }
            )
        if 'gene:"ObiH"' not in query and 'gene:"obiH"' not in query:
            return _FakeResponse({"results": []})
        return _FakeResponse(
            {
                "results": [
                    {
                        "primaryAccession": "A0A1X9LWZ7",
                        "entryType": "UniProtKB unreviewed (TrEMBL)",
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": "Threonine aldolase"}},
                        },
                        "genes": [
                            {
                                "geneName": {"value": "obiH"},
                                "synonyms": [{"value": "CIB54_12585"}],
                            }
                        ],
                        "organism": {"scientificName": "Pseudomonas fluorescens"},
                    }
                ]
            }
        )


class _NocBDomainUniProtClient:
    def __init__(self) -> None:
        self.queries: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None) -> _FakeResponse:
        params = params or {}
        query = str(params.get("query") or "")
        self.queries.append(query)
        if 'gene:"NocB"' not in query:
            return _FakeResponse({"results": []})
        return _FakeResponse(
            {
                "results": [
                    {
                        "primaryAccession": "Q5J1Q6",
                        "entryType": "UniProtKB unreviewed (TrEMBL)",
                        "proteinDescription": {
                            "recommendedName": {
                                "fullName": {"value": "Nonribosomal peptide synthetase NocB"}
                            },
                            "alternativeNames": [
                                {"fullName": {"value": "NocB thioesterase domain-containing protein"}}
                            ],
                        },
                        "genes": [{"geneName": {"value": "NocB"}}],
                        "organism": {"scientificName": "Nocardia uniformis subsp. tsuyamanensis"},
                    }
                ]
            }
        )


class _ParentheticalGeneUniProtClient:
    def __init__(self) -> None:
        self.queries: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None) -> _FakeResponse:
        params = params or {}
        query = str(params.get("query") or "")
        self.queries.append(query)
        if 'gene:"CTS"' not in query:
            return _FakeResponse({"results": []})
        return _FakeResponse(
            {
                "results": [
                    {
                        "primaryAccession": "Q9LQ16",
                        "entryType": "UniProtKB reviewed (Swiss-Prot)",
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": "Peroxisomal ABC transporter CTS"}},
                        },
                        "genes": [{"geneName": {"value": "CTS"}}],
                        "organism": {"scientificName": "Arabidopsis thaliana"},
                    }
                ]
            }
        )


def _uniprot_result(
    accession: str,
    *,
    protein_name: str,
    gene: str,
    organism: str = "Arabidopsis thaliana",
    reviewed: bool = True,
) -> Dict[str, Any]:
    return {
        "primaryAccession": accession,
        "entryType": "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": protein_name}},
        },
        "genes": [{"geneName": {"value": gene}}],
        "organism": {"scientificName": organism},
    }


class _ArabidopsisAmbiguousGeneUniProtClient:
    def __init__(self, gene: str, reviewed_accession: str, protein_name: str) -> None:
        self.gene = gene
        self.reviewed_accession = reviewed_accession
        self.protein_name = protein_name
        self.queries: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None) -> _FakeResponse:
        params = params or {}
        query = str(params.get("query") or "")
        self.queries.append(query)
        if f'gene:"{self.gene}"' not in query and f'"{self.gene}"' not in query:
            return _FakeResponse({"results": []})
        return _FakeResponse(
            {
                "results": [
                    _uniprot_result(
                        self.reviewed_accession,
                        protein_name=self.protein_name,
                        gene=self.gene,
                        reviewed=True,
                    ),
                    _uniprot_result(
                        f"A0A{self.reviewed_accession[-3:]}1",
                        protein_name=self.protein_name,
                        gene=self.gene,
                        reviewed=False,
                    ),
                    _uniprot_result(
                        f"A0A{self.reviewed_accession[-3:]}2",
                        protein_name=self.protein_name,
                        gene=self.gene,
                        reviewed=False,
                    ),
                ]
            }
        )


class _EuropePmcFullTextClient:
    def __init__(self) -> None:
        self.urls: List[str] = []

    def get(self, url: str, params: Dict[str, Any] | None = None) -> _FakeResponse:
        self.urls.append(url)
        if url.endswith("/PMC8072733/fullTextXML"):
            return _FakeResponse(
                {},
                text="<article><body>Biochemical assays demonstrated that ObiH (ObaG) is a new LTTA.</body></article>",
            )
        return _FakeResponse({}, status_code=404)


def _glycolysis_complex_result(protein_row: Dict[str, Any], species: str) -> Dict[str, Any]:
    assert species == "Homo sapiens"
    name = str(protein_row.get("name") or "")
    if name == "hexokinase":
        complex_name = "hexokinase complex"
        complex_id = 431773
        component = {"name": "Hexokinase-3", "uniprot": "P52790", "pathbank_protein_id": 161288, "stoichiometry": 1}
    elif name == "phosphoglucose isomerase":
        complex_name = "Glucose-6-phosphate isomerase"
        complex_id = 3607
        component = {
            "name": "Glucose-6-phosphate isomerase",
            "uniprot": "P06744",
            "pathbank_protein_id": 751,
            "stoichiometry": 1,
        }
    else:
        raise AssertionError(f"unexpected enzyme lookup for {name}")
    return {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "name": complex_name,
        "pathbank_complex_id": complex_id,
        "pathbank_protein_complex_id": complex_id,
        "species_id": 1,
        "components": [component],
        "confidence": 0.9,
        "chosen_rule": "enzyme_component_species",
        "candidates": [],
        "issues": [],
        "resolution": {"status": "matched"},
    }


def test_complex_component_reconciles_by_uniprot_without_replacing_db_ids() -> None:
    components = [{"name": "Hexokinase-3", "uniprot": "P52790", "pathbank_protein_id": 161288}]
    local_proteins = [
        {
            "key": "prot_hexokinase",
            "name": "hexokinase",
            "mapped_ids": {"uniprot": "P52790", "pathbank_protein_id": "206"},
        }
    ]

    reconciled = _reconcile_components_against_local_proteins(components, local_proteins)

    assert reconciled == [
        {
            "name": "hexokinase",
            "uniprot": "P52790",
            "pathbank_protein_id": 161288,
            "protein_key": "prot_hexokinase",
        }
    ]
    assert components == [{"name": "Hexokinase-3", "uniprot": "P52790", "pathbank_protein_id": 161288}]


def test_complex_component_reconciles_by_pathbank_protein_id() -> None:
    reconciled = _reconcile_components_against_local_proteins(
        [{"name": "DB display name", "mapped_ids": {"pathbank_protein_id": "751"}}],
        [{"name": "phosphoglucose isomerase", "mapping_meta": {"pathbank_protein_id": 751}}],
    )

    assert reconciled[0]["name"] == "phosphoglucose isomerase"
    assert reconciled[0]["mapped_ids"] == {"pathbank_protein_id": "751"}


def test_complex_component_without_local_protein_match_is_unchanged() -> None:
    components = [{"name": "Missing protein", "uniprot": "Q00000", "pathbank_protein_id": 99}]

    reconciled = _reconcile_components_against_local_proteins(
        components,
        [{"name": "different protein", "mapped_ids": {"uniprot": "P00001"}}],
    )

    assert reconciled == components
    assert reconciled[0] is not components[0]


def test_compound_resolves_by_hmdb_before_fuzzy_name() -> None:
    resolver = _make_resolver()
    db_row = {
        "id": 1420,
        "name": "Water",
        "hmdb_id": "HMDB0002111",
        "kegg_id": "C00001",
        "chebi_id": "15377",
        "pubchem_cid": "962",
        "cas": "7732-18-5",
    }

    with patch.object(resolver, "_query", return_value=[db_row]), \
            patch.object(resolver, "map_compound", side_effect=AssertionError("fuzzy name lookup should not run")):
        result = resolver.map_compound_row({"name": "ambiguous water synonym", "mapped_ids": {"hmdb": "HMDB0002111"}})

    assert result["status"] == "mapped"
    assert result["pathbank_compound_id"] == 1420
    assert result["resolution"]["status"] == "matched"
    assert result["resolution"]["order_step"].startswith("external_id")


def test_compound_synonym_resolves_before_novel_creation() -> None:
    resolver = _make_resolver()
    synonym_result = resolver._compound_result_from_row(
        {"id": 31, "name": "D-glucose", "hmdb_id": "HMDB0000122"},
        confidence=0.95,
        chosen_rule="synonym",
    )

    with patch.object(resolver, "_map_compound_exact_name", return_value=_unmapped()), \
            patch.object(resolver, "_map_compound_synonym", return_value=synonym_result), \
            patch.object(resolver, "map_compound", side_effect=AssertionError("fuzzy/novel fallback should not run")):
        result = resolver.map_compound_row({"name": "glucose synonym"})

    assert result["status"] == "mapped"
    assert result["pathbank_compound_id"] == 31
    assert result["chosen_rule"] == "synonym"
    assert result["resolution"]["order_step"] == "synonym"


def test_protein_without_species_requests_species_instead_of_name_matching() -> None:
    resolver = _make_resolver()

    with patch.object(resolver, "map_protein", side_effect=AssertionError("unsafe protein name lookup should not run")):
        result = resolver.map_protein_row({"name": "Albumin"}, "")

    assert result["status"] == "unmapped"
    assert result["reason"] == "needs_species"
    assert result["resolution"]["issue"] == "needs_species"


def test_protein_with_uniprot_resolves_directly_without_species() -> None:
    resolver = _make_resolver()
    db_row = {"id": 500, "name": "Albumin", "uniprot_id": "P02768", "gene_name": "ALB", "species_id": 1}

    with patch.object(resolver, "_query", return_value=[db_row]), \
            patch.object(resolver, "map_protein", side_effect=AssertionError("fuzzy protein lookup should not run")):
        result = resolver.map_protein_row({"name": "Albumin", "mapped_ids": {"uniprot": "P02768"}}, "")

    assert result["status"] == "mapped"
    assert result["pathbank_protein_id"] == 500
    assert result["mapped_ids"]["uniprot"] == "P02768"
    assert result["resolution"]["order_step"] == "uniprot"


def test_hybrid_protein_mapping_falls_back_to_uniprot_after_db_novel() -> None:
    api_result = {
        "status": "mapped",
        "mapped_ids": {"uniprot": "P19367"},
        "confidence": 0.91,
        "candidates": [{"uniprot": "P19367"}],
    }

    with patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=api_result) as api_lookup:
        result = _map_protein_with_strategy(
            id_source="hybrid",
            db=_AvailableDb(),  # type: ignore[arg-type]
            client=object(),  # type: ignore[arg-type]
            cache=_MemoryCache(),  # type: ignore[arg-type]
            name="Hexokinase",
            organism="Homo sapiens",
            protein_row={"name": "Hexokinase", "species": "Homo sapiens"},
        )

    assert result["status"] == "mapped"
    assert result["source"] == "api"
    assert result["mapped_ids"]["uniprot"] == "P19367"
    api_lookup.assert_called_once()


def test_cached_ambiguous_reviewed_exact_gene_candidate_is_promoted_without_network() -> None:
    cache = _MemoryCache()
    name = "COMATOSE (CTS)"
    organism = "Arabidopsis thaliana"
    aliases = _protein_alias_entries(name, {"name": name})
    base_key = "comatose cts::arabidopsis thaliana::::{}"
    legacy_key = f"api-v6::{base_key}::{json.dumps(aliases, sort_keys=True)}"
    cache.set(
        "proteins",
        legacy_key,
        {
            "status": "unmapped",
            "reason": "ambiguous",
            "provider": "UniProt",
            "source": "api",
            "confidence": 1.0,
            "candidates": [
                {
                    "accession": "Q94FB9",
                    "protein_name": "ABC transporter D family member 1",
                    "gene_names": ["ABCD1", "CTS"],
                    "organism": "Arabidopsis thaliana",
                    "reviewed": True,
                    "score": 1.0,
                    "matched_query": "CTS",
                    "matched_alias": "CTS",
                    "alias_source": "parenthetical_gene_symbol",
                },
                {
                    "accession": "F4JJ27",
                    "protein_name": "Peroxisomal ABC transporter 1",
                    "gene_names": ["COMATOSE", "CTS"],
                    "organism": "Arabidopsis thaliana",
                    "reviewed": False,
                    "score": 0.97,
                    "matched_query": "COMATOSE",
                    "alias_source": "primary_name",
                },
            ],
        },
    )

    result = _map_protein_with_strategy(
        id_source="hybrid",
        db=_AvailableDb(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        name=name,
        organism=organism,
        protein_row={"name": name, "species": organism},
    )

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q94FB9"
    assert result["resolution"]["order_step"] == "api_uniprot"


def test_mapping_meta_candidate_is_promoted_when_api_result_has_no_candidates() -> None:
    api_miss = {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "UniProt",
        "source": "api",
        "candidates": [],
    }
    protein_row = {
        "name": "acyl-CoA oxidase 1 (ACX1)",
        "species": "Arabidopsis thaliana",
        "mapping_meta": {
            "candidates": [
                {
                    "accession": "O65202",
                    "protein_name": "Peroxisomal acyl-coenzyme A oxidase 1",
                    "gene_names": ["ACX1"],
                    "organism": "Arabidopsis thaliana",
                    "reviewed": True,
                    "score": 1.0,
                    "matched_query": "ACX1",
                    "matched_alias": "ACX1",
                    "alias_source": "parenthetical_gene_symbol",
                },
                {
                    "accession": "F4JMK8",
                    "gene_names": ["ACX1"],
                    "organism": "Arabidopsis thaliana",
                    "reviewed": False,
                    "score": 0.97,
                    "matched_query": "ACX1",
                    "matched_alias": "ACX1",
                    "alias_source": "parenthetical_gene_symbol",
                },
            ]
        },
    }

    with patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=api_miss):
        result = _map_protein_with_strategy(
            id_source="hybrid",
            db=_AvailableDb(),  # type: ignore[arg-type]
            client=object(),  # type: ignore[arg-type]
            cache=_MemoryCache(),  # type: ignore[arg-type]
            name="acyl-CoA oxidase 1 (ACX1)",
            organism="Arabidopsis thaliana",
            protein_row=protein_row,
        )

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "O65202"
    assert result["mapping_meta_promoted"] is True


def test_uniprot_mapping_uses_literature_alias_for_obag() -> None:
    client = _AliasUniProtClient()

    result = map_protein_uniprot(client, "ObaG", "Pseudomonas fluorescens")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "A0A1X9LWZ7"
    assert result["matched_alias"] == "ObiH"
    assert result["alias_source"] == "literature_alias"
    assert result["resolved_name"] == "Threonine aldolase"
    assert result["chosen_rule"] == "top_unique_alias_candidate"
    assert result["literature_aliases"] == [{"alias": "ObiH", "source": "literature_alias"}]
    assert any('gene:"ObiH"' in query for query in client.queries)


def test_europepmc_full_text_uses_pmcid_endpoint() -> None:
    client = _EuropePmcFullTextClient()

    text = _europepmc_full_text(
        client,
        {"source": "MED", "id": "33900425", "pmcid": "PMC8072733"},
    )

    assert "ObiH (ObaG)" in text
    assert client.urls == ["https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8072733/fullTextXML"]


def test_literature_alias_extracts_parenthetical_gene_name_without_marker() -> None:
    aliases = _extract_aliases_from_literature_text(
        "ObaG",
        "Biochemical assays demonstrated that ObiH (ObaG) is a new LTTA.",
    )

    assert aliases == [{"alias": "ObiH", "source": "literature_alias"}]


def test_uniprot_candidate_parser_uses_submission_names_and_unreviewed_flag() -> None:
    result = _extract_uniprot_candidates(
        {
            "results": [
                {
                    "primaryAccession": "A0A1X9LWZ7",
                    "entryType": "UniProtKB unreviewed (TrEMBL)",
                    "proteinDescription": {
                        "submissionNames": [
                            {"fullName": {"value": "Threonine aldolase"}},
                        ],
                    },
                    "genes": [{"geneName": {"value": "obiH"}}],
                    "organism": {"scientificName": "Pseudomonas fluorescens"},
                }
            ]
        },
        query_name="Threonine aldolase",
        organism="Pseudomonas fluorescens",
    )

    assert result[0]["accession"] == "A0A1X9LWZ7"
    assert result[0]["protein_name"] == "Threonine aldolase"
    assert result[0]["reviewed"] is False
    assert result[0]["score"] == 0.8


def test_map_payload_returns_payload_and_report_without_mutating_input(tmp_path: Path) -> None:
    payload = {
        "entities": {
            "proteins": [{"name": "Hexokinase", "organism": "Homo sapiens"}],
            "compounds": [{"name": "glucose"}],
        },
        "processes": {"reactions": []},
    }

    with patch("t2pw.mapping.map_ids.hydrate_species_references", return_value={"hydrated": 0, "matched": 0, "novel": 0}), \
            patch("t2pw.mapping.map_ids._rewrite_reaction_protein_enzymes_to_complexes", return_value={"summary": {}, "actions": []}), \
            patch(
                "t2pw.mapping.map_ids._map_protein_with_strategy",
                return_value={
                    "status": "mapped",
                    "source": "api",
                    "provider": "UniProt",
                    "mapped_ids": {"uniprot": "P19367"},
                    "confidence": 0.99,
                    "candidates": [],
                },
            ), \
            patch(
                "t2pw.mapping.map_ids._map_compound_with_strategy",
                return_value={
                    "status": "mapped",
                    "source": "api",
                    "provider": "ChEBI",
                    "mapped_ids": {"chebi": "CHEBI:17234"},
                    "confidence": 0.99,
                    "candidates": [],
                },
            ):
        result = map_payload(payload, cache_path=tmp_path / "id_mapping_cache.json", id_source="api")

    mapped = result["payload"]
    report = result["report"]
    assert payload["entities"]["proteins"][0].get("mapped_ids") is None
    assert mapped["entities"]["proteins"][0]["mapped_ids"] == {"uniprot": "P19367"}
    assert mapped["entities"]["compounds"][0]["mapped_ids"] == {"chebi": "CHEBI:17234"}
    assert report["summary"]["proteins_mapped"] == 1
    assert report["summary"]["compounds_mapped"] == 1


def test_map_payload_can_invalidate_cache_keys(tmp_path: Path) -> None:
    cache_path = tmp_path / "id_mapping_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "proteins": {"stale-protein": {"status": "mapped"}},
                "compounds": {"stale-compound": {"status": "mapped"}},
                "complexes": {},
            }
        ),
        encoding="utf-8",
    )

    with patch("t2pw.mapping.map_ids.hydrate_species_references", return_value={"hydrated": 0, "matched": 0, "novel": 0}), \
            patch("t2pw.mapping.map_ids._rewrite_reaction_protein_enzymes_to_complexes", return_value={"summary": {}, "actions": []}):
        result = map_payload(
            {"entities": {"proteins": [], "compounds": []}, "processes": {"reactions": []}},
            cache_path=cache_path,
            id_source="api",
            invalidate_cache_keys={"proteins": ["stale-protein"]},
        )

    saved_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "stale-protein" not in saved_cache["proteins"]
    assert "stale-compound" in saved_cache["compounds"]
    assert result["report"]["summary"]["proteins_total"] == 0


def _pdh_complex_payload() -> Dict[str, Any]:
    """A protein complex whose plain-string components have no matching standalone
    entities.proteins rows (mirrors the pyruvate dehydrogenase complex gap-resolver
    bug: Stage 4a upgrades these to dicts with no stoichiometry, and nothing upstream
    can backfill it)."""
    return {
        "entities": {
            "proteins": [],
            "compounds": [],
            "protein_complexes": [
                {
                    "name": "pyruvate dehydrogenase complex",
                    "species": "Homo sapiens",
                    "components": [
                        "pyruvate dehydrogenase E1",
                        "dihydrolipoamide acetyltransferase E2",
                        "dihydrolipoamide dehydrogenase E3",
                    ],
                }
            ],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _pdh_db_hydrated_components() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Pyruvate dehydrogenase E1",
            "stoichiometry": 1,
            "pathbank_protein_id": 101,
            "mapped_ids": {"pathbank_protein_id": "101"},
        },
        {
            "name": "Dihydrolipoamide acetyltransferase E2",
            "stoichiometry": 1,
            "pathbank_protein_id": 102,
            "mapped_ids": {"pathbank_protein_id": "102"},
        },
        {
            "name": "Dihydrolipoamide dehydrogenase E3",
            "stoichiometry": 1,
            "pathbank_protein_id": 103,
            "mapped_ids": {"pathbank_protein_id": "103"},
        },
    ]


def _pdh_mapped_db_result() -> Dict[str, Any]:
    return {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "name": "Pyruvate dehydrogenase complex",
        "pathbank_complex_id": 555,
        "pathbank_protein_complex_id": 555,
        "species_id": 1,
        "components": _pdh_db_hydrated_components(),
        "confidence": 0.95,
        "chosen_rule": "pathbank_protein_complex_id",
        "candidates": [],
        "issues": [],
        "resolution": {"status": "matched", "order_step": "pathbank_protein_complex_id"},
    }


def test_confident_db_complex_match_overrides_stale_extraction_components(tmp_path: Path) -> None:
    """Regression test: when Stage 6 achieves a genuine, confident DB match for a
    protein complex, its DB-hydrated components (with real stoichiometry) must win
    over whatever incomplete extraction/Stage-4a-derived components the payload's
    own complex row already carries."""
    payload = _pdh_complex_payload()

    with patch(
        "t2pw.mapping.map_ids.hydrate_species_references",
        return_value={"hydrated": 0, "matched": 0, "novel": 0},
    ), patch(
        "t2pw.mapping.map_ids._map_complex_with_strategy",
        return_value=_pdh_mapped_db_result(),
    ):
        result = map_payload(
            payload,
            cache_path=tmp_path / "id_mapping_cache.json",
            id_source="api",
            use_cache=False,
        )

    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert complex_row["components"] == _pdh_db_hydrated_components()
    assert all("stoichiometry" in component for component in complex_row["components"])
    assert complex_row["pathbank_complex_id"] == 555
    assert result["report"]["summary"]["protein_complexes_mapped"] == 1


def test_unconfident_db_complex_match_keeps_extraction_components(tmp_path: Path) -> None:
    """Negative case: when Stage 6 does not achieve a confident DB match (status
    stays unmapped/ambiguous/novel), the original extraction-derived components on
    the payload's own complex row must be left untouched (current behavior)."""
    payload = _pdh_complex_payload()

    def _ambiguous_complex_result(**_: Any) -> Dict[str, Any]:
        return {
            "status": "unmapped",
            "reason": "ambiguous",
            "provider": "PathBankDB",
            "source": "db",
            "confidence": 0.4,
            "chosen_rule": "resolved_component_species",
            "candidates": [],
            "components": [],
            "issues": [],
            "resolution": {"status": "ambiguous", "issue": "ambiguous_component_complex_species"},
        }

    with patch(
        "t2pw.mapping.map_ids.hydrate_species_references",
        return_value={"hydrated": 0, "matched": 0, "novel": 0},
    ), patch(
        "t2pw.mapping.map_ids._map_complex_with_strategy",
        side_effect=_ambiguous_complex_result,
    ):
        result = map_payload(
            payload,
            cache_path=tmp_path / "id_mapping_cache.json",
            id_source="api",
            use_cache=False,
        )

    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert complex_row["components"] == [
        "pyruvate dehydrogenase E1",
        "dihydrolipoamide acetyltransferase E2",
        "dihydrolipoamide dehydrogenase E3",
    ]
    assert result["report"]["summary"]["protein_complexes_mapped"] == 0


def test_uniprot_mapping_uses_parent_alias_for_nocb_domain() -> None:
    client = _NocBDomainUniProtClient()

    result = map_protein_uniprot(client, "NocB thioesterase (TE) domain", "Nocardia uniformis")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q5J1Q6"
    assert result["matched_alias"] == "NocB"
    assert result["alias_source"] == "domain_parent"
    assert result["resolved_name"] == "Nonribosomal peptide synthetase NocB"
    assert result["chosen_rule"] == "top_unique_alias_candidate"
    assert any('gene:"NocB"' in query for query in client.queries)


def test_protein_alias_entries_extract_parenthetical_gene_symbols() -> None:
    comatose_aliases = {entry["alias"] for entry in _protein_alias_entries("COMATOSE (CTS)")}
    opcl_aliases = {entry["alias"] for entry in _protein_alias_entries("OPC-8:CoA ligase 1 (OPCL1)")}
    locus_aliases = {entry["alias"] for entry in _protein_alias_entries("4-coumarate-CoA ligase-like 5 (4CLL5)")}
    arabidopsis_aliases = {entry["alias"] for entry in _protein_alias_entries("example protein (At1g20510)")}

    assert "CTS" in comatose_aliases
    assert "OPCL1" in opcl_aliases
    assert "4CLL5" in locus_aliases
    assert "At1g20510" in arabidopsis_aliases
    assert "mitochondrial" not in {
        entry["alias"] for entry in _protein_alias_entries("example protein (mitochondrial)")
    }
    assert "16:3" not in {
        entry["alias"] for entry in _protein_alias_entries("hexadecatrienoic acid (16:3)")
    }
    assert "isoform 2" not in {
        entry["alias"] for entry in _protein_alias_entries("example protein (isoform 2)")
    }


def test_protein_alias_entries_add_conservative_arabidopsis_at_prefix_gene_symbols() -> None:
    assert {"ACH1"} <= {entry["alias"] for entry in _protein_alias_entries("AtACH1")}
    assert {"ACH2"} <= {entry["alias"] for entry in _protein_alias_entries("AtACH2")}
    assert "1g20510" not in {entry["alias"] for entry in _protein_alias_entries("At1g20510")}
    assert "4CL1" not in {entry["alias"] for entry in _protein_alias_entries("At4CL1")}


def test_uniprot_mapping_uses_parenthetical_gene_symbol_alias() -> None:
    client = _ParentheticalGeneUniProtClient()

    result = map_protein_uniprot(client, "COMATOSE (CTS)", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q9LQ16"
    assert result["matched_alias"] == "CTS"
    assert result["alias_source"] == "parenthetical_gene_symbol"
    assert any('gene:"CTS"' in query for query in client.queries)


def test_uniprot_mapping_accepts_comatose_reviewed_exact_gene_despite_close_duplicates() -> None:
    client = _ArabidopsisAmbiguousGeneUniProtClient("CTS", "Q94FB9", "ABC transporter D family member 1")

    result = map_protein_uniprot(client, "COMATOSE (CTS)", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q94FB9"
    assert result["matched_alias"] == "CTS"
    assert result["alias_source"] == "parenthetical_gene_symbol"
    assert any(not candidate["reviewed"] and candidate["score"] >= 0.97 for candidate in result["candidates"])


def test_uniprot_mapping_accepts_opcl1_reviewed_exact_gene_despite_close_duplicates() -> None:
    client = _ArabidopsisAmbiguousGeneUniProtClient("OPCL1", "Q84P21", "OPC-8:CoA ligase 1")

    result = map_protein_uniprot(client, "OPC-8:CoA ligase 1 (OPCL1)", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q84P21"
    assert result["matched_alias"] == "OPCL1"
    assert result["alias_source"] == "parenthetical_gene_symbol"


def test_uniprot_mapping_accepts_acx1_reviewed_exact_gene_despite_close_duplicates() -> None:
    client = _ArabidopsisAmbiguousGeneUniProtClient("ACX1", "O65202", "Peroxisomal acyl-coenzyme A oxidase 1")

    result = map_protein_uniprot(client, "acyl-CoA oxidase 1 (ACX1)", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "O65202"
    assert result["matched_alias"] == "ACX1"
    assert result["alias_source"] == "parenthetical_gene_symbol"


def test_uniprot_mapping_uses_atach1_gene_alias() -> None:
    client = _ArabidopsisAmbiguousGeneUniProtClient("ACH1", "Q5FYU1", "Acyl-CoA hydrolase 1")

    result = map_protein_uniprot(client, "AtACH1", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q5FYU1"
    assert result["matched_alias"] == "ACH1"
    assert result["alias_source"] == "arabidopsis_at_prefix_gene_symbol"
    assert any('gene:"ACH1"' in query for query in client.queries)


def test_uniprot_mapping_uses_atach2_gene_alias() -> None:
    client = _ArabidopsisAmbiguousGeneUniProtClient("ACH2", "F4HU51", "Acyl-CoA hydrolase 2")

    result = map_protein_uniprot(client, "AtACH2", "Arabidopsis thaliana")

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "F4HU51"
    assert result["matched_alias"] == "ACH2"
    assert result["alias_source"] == "arabidopsis_at_prefix_gene_symbol"
    assert any('gene:"ACH2"' in query for query in client.queries)


def test_db_protein_mapping_tries_parenthetical_gene_alias_before_fuzzy() -> None:
    resolver = _make_resolver()
    mapped = {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "mapped_ids": {"uniprot": "Q9ZVH4", "pathbank_protein_id": "42"},
        "pathbank_protein_id": 42,
        "confidence": 1.08,
        "chosen_rule": "direct_id_match:gene_name",
        "candidates": [{"gene_name": "OPCL1"}],
    }

    def by_ids(ids: Dict[str, str], *, species: str | None = None) -> Dict[str, Any]:
        if ids.get("gene_name") == "OPCL1" and species == "Arabidopsis thaliana":
            return dict(mapped)
        return _unmapped()

    with patch.object(resolver, "_find_species_ids", return_value=[1]), \
            patch.object(resolver, "_map_protein_exact_name_species", return_value=_unmapped()), \
            patch.object(resolver, "map_protein_by_ids", side_effect=by_ids), \
            patch.object(resolver, "map_protein", side_effect=AssertionError("fuzzy fallback should not run")):
        result = resolver.map_protein_row(
            {"name": "OPC-8:CoA ligase 1 (OPCL1)"},
            "Arabidopsis thaliana",
        )

    assert result["status"] == "mapped"
    assert result["mapped_ids"]["uniprot"] == "Q9ZVH4"
    assert result["matched_alias"] == "OPCL1"
    assert result["alias_source"] == "parenthetical_gene_symbol"
    assert result["resolution"]["order_step"] == "alias_gene_name_species"


def test_mapping_route_keeps_colon_protein_names_out_of_complex_route() -> None:
    protein_route = route_entity_for_mapping("OPC-8:CoA ligase 1 (OPCL1)", "protein")
    weak_route = route_entity_for_mapping("OPC-8:CoA ligase 1 (OPCL1)", "compound")
    metabolite_route = route_entity_for_mapping("OPC-6:0-CoA", "compound")

    assert protein_route["route"] == "protein"
    assert weak_route["route"] == "protein"
    assert metabolite_route["route"] == "compound"


def test_complex_maps_by_name_and_species() -> None:
    resolver = _make_resolver()
    complex_rows = [{"id": 301, "name": "MPC complex", "species_id": 1}]

    with patch.object(resolver, "_find_species_ids", return_value=[1]), \
            patch.object(resolver, "_query", return_value=complex_rows):
        result = resolver.map_protein_complex("MPC complex", "Homo sapiens")

    assert result["status"] == "mapped"
    assert result["pathbank_complex_id"] == 301
    assert result["species_id"] == 1


def test_complex_maps_by_component_and_species() -> None:
    resolver = _make_resolver()
    components: List[Dict[str, Any]] = [{"name": "MPC1", "pathbank_protein_id": 11, "stoichiometry": 1}]
    component_candidate = {"pathbank_complex_id": 301, "name": "MPC complex", "species_id": 1, "score": 0.9}

    with patch.object(resolver, "_resolve_complex_components", return_value=(components, [])), \
            patch.object(resolver, "_find_species_ids", return_value=[1]), \
            patch.object(resolver, "map_protein_complex", return_value=_unmapped()), \
            patch.object(resolver, "_find_complexes_by_component_protein_id", return_value=[component_candidate]), \
            patch.object(resolver, "_complex_component_rows", return_value=components):
        result = resolver.map_protein_complex_row(
            {"name": "MPC complex", "components": [{"name": "MPC1", "stoichiometry": 1}]},
            "Homo sapiens",
        )

    assert result["status"] == "mapped"
    assert result["pathbank_complex_id"] == 301
    assert result["components"] == components
    assert result["resolution"]["order_step"] == "resolved_component_species"


def test_complex_without_component_fails_with_gap_issue() -> None:
    resolver = _make_resolver()

    with patch.object(resolver, "_resolve_complex_components", return_value=([], [])), \
            patch.object(resolver, "_find_species_ids", return_value=[1]), \
            patch.object(resolver, "map_protein_complex", return_value=_unmapped()):
        result = resolver.map_protein_complex_row({"name": "Empty complex"}, "Homo sapiens")

    assert result["status"] == "unmapped"
    assert result["reason"] == "no_components"
    assert result["resolution"]["status"] == "unresolved"
    assert result["issues"][0]["issue"] == "protein_complex_missing_components"


def test_enzyme_protein_becomes_single_component_complex_when_no_db_complex_exists() -> None:
    resolver = _make_resolver()
    protein_result = {
        "status": "mapped",
        "pathbank_protein_id": 11,
        "mapped_ids": {"uniprot": "Q9Y5U8", "pathbank_protein_id": "11"},
        "confidence": 1.0,
        "chosen_rule": "pathbank_protein_id",
        "candidates": [],
    }

    with patch.object(resolver, "_map_protein_by_pathbank_id", return_value=protein_result), \
            patch.object(resolver, "_find_species_ids", return_value=[1]), \
            patch.object(resolver, "_find_complexes_by_component_protein_id", return_value=[]):
        result = resolver.map_enzyme_protein_to_complex(
            {"name": "MPC1", "pathbank_protein_id": 11},
            "Homo sapiens",
        )

    assert result["resolution"]["status"] == "novel"
    assert result["chosen_rule"] == "novel_enzyme_single_component_complex"
    assert result["name"] == "MPC1 complex"
    assert result["species_id"] == 1
    assert result["components"] == [{"name": "MPC1", "stoichiometry": 1, "pathbank_protein_id": 11, "mapped_ids": protein_result["mapped_ids"]}]


def test_glycolysis_reaction_enzyme_complex_components_reconcile_to_local_proteins() -> None:
    resolver = _make_resolver()
    mapped = {
        "entities": {
            "proteins": [
                {
                    "name": "hexokinase",
                    "pathbank_protein_id": 206,
                    "mapped_ids": {"uniprot": "P52790", "pathbank_protein_id": "206"},
                },
                {
                    "name": "phosphoglucose isomerase",
                    "pathbank_protein_id": 751,
                    "mapped_ids": {"uniprot": "P06744", "pathbank_protein_id": "751"},
                },
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {"name": "hexokinase reaction", "enzymes": [{"protein": "hexokinase"}]},
                {
                    "name": "phosphoglucose isomerase reaction",
                    "enzymes": [{"protein": "phosphoglucose isomerase"}],
                },
            ]
        },
    }

    with patch.object(resolver, "map_enzyme_protein_to_complex", side_effect=_glycolysis_complex_result):
        _rewrite_reaction_protein_enzymes_to_complexes(
            mapped,
            db=resolver,
            cache=_MemoryCache(),  # type: ignore[arg-type]
            global_organism="Homo sapiens",
        )

    complexes = {
        complex_row["pathbank_protein_complex_id"]: complex_row
        for complex_row in mapped["entities"]["protein_complexes"]
    }
    assert complexes[431773]["components"][0]["name"] == "hexokinase"
    assert complexes[431773]["components"][0]["pathbank_protein_id"] == 161288
    assert complexes[3607]["components"][0]["name"] == "phosphoglucose isomerase"
    assert complexes[3607]["components"][0]["pathbank_protein_id"] == 751
    assert all(
        reaction["enzymes"][0]["entity_type"] == "protein_complex"
        for reaction in mapped["processes"]["reactions"]
    )


def test_reaction_enzyme_rewrite_creates_generated_wrapper_from_valid_protein_component() -> None:
    mapped = {
        "entities": {
            "proteins": [
                {
                    "name": "NdmA",
                    "species": "Pseudomonas putida",
                    "mapped_ids": {"uniprot": "A0A000"},
                }
            ],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {"name": "caffeine demethylation", "enzymes": [{"protein": "NdmA"}]},
            ]
        },
    }

    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped,
        db=None,
        cache=_MemoryCache(),  # type: ignore[arg-type]
        global_organism="Pseudomonas putida",
    )

    complexes = mapped["entities"]["protein_complexes"]
    assert [row["name"] for row in complexes] == ["NdmA complex"]
    assert complexes[0]["generated"] is True
    assert complexes[0]["generation_reason"] == "single_protein_pathwhiz_wrapper"
    assert complexes[0]["components"][0]["name"] == "NdmA"
    assert complexes[0]["components"][0]["mapped_ids"]["uniprot"] == "A0A000"
    assert mapped["processes"]["reactions"][0]["enzymes"] == [
        {"entity": "NdmA complex", "entity_type": "protein_complex", "role": "catalyst"}
    ]
    assert report["summary"]["reaction_enzyme_complexes_novel"] == 1


def test_reaction_enzyme_rewrite_does_not_generate_wrapper_without_component_identity() -> None:
    mapped = {
        "entities": {
            "proteins": [{"name": "NdmA", "species": "Pseudomonas putida"}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {"name": "caffeine demethylation", "enzymes": [{"protein": "NdmA"}]},
            ]
        },
    }

    report = _rewrite_reaction_protein_enzymes_to_complexes(
        mapped,
        db=None,
        cache=_MemoryCache(),  # type: ignore[arg-type]
        global_organism="Pseudomonas putida",
    )

    assert mapped["entities"]["protein_complexes"] == []
    assert mapped["processes"]["reactions"][0]["enzymes"] == [{"protein": "NdmA"}]
    assert report["summary"]["reaction_enzyme_complexes_skipped_invalid_component"] == 1
    assert report["actions"][0]["type"] == "reaction_enzyme_complex_wrapper_skipped_invalid_component"
    assert report["actions"][0]["missing"] == ["uniprot_or_drugbank"]
