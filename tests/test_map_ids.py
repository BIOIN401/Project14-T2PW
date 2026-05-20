from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import PathBankDbResolver, _map_protein_with_strategy  # noqa: E402


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
