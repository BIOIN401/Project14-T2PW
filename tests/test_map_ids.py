from __future__ import annotations

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
    _map_protein_with_strategy,
    _reconcile_components_against_local_proteins,
    _rewrite_reaction_protein_enzymes_to_complexes,
    map_protein_uniprot,
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
