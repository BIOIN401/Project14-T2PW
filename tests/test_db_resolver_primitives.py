"""Unit tests for PathBankDbResolver DB lookup primitives.

All tests mock _query so no live DB connection is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import PathBankDbResolver, hydrate_species_references, resolve_mapping_gaps  # noqa: E402


def _make_resolver() -> PathBankDbResolver:
    """Return a resolver that never attempts a live DB connection."""
    r = PathBankDbResolver.__new__(PathBankDbResolver)
    r.host = "localhost"
    r.port = 3306
    r.user = "test"
    r.password = ""
    r.schema = "pathbank"
    r.connect_timeout = 6
    r.read_timeout = 20
    r.write_timeout = 20
    r._driver = object()  # non-None so available() returns True
    r._conn = None
    r.last_error = ""
    return r


def _patch_query(resolver: PathBankDbResolver, return_value: List[Dict[str, Any]]):
    """Context manager: patch resolver._query to return *return_value*."""
    return patch.object(resolver, "_query", return_value=return_value)


def test_complex_component_queries_use_protein_complex_id_join_column():
    r = _make_resolver()

    with patch.object(r, "_query", return_value=[]) as query:
        r._complex_component_rows(301)
        r._find_complexes_by_component_protein_id(500, [1])

    sql = "\n".join(call.args[0] for call in query.call_args_list)
    assert "pcp.protein_complex_id" in sql
    assert "pcp.complex_id" not in sql


def test_find_complex_by_component_uses_protein_complex_id_join_column():
    r = _make_resolver()
    protein_result = {
        "status": "mapped",
        "pathbank_protein_id": 500,
        "candidates": [],
        "confidence": 0.9,
        "chosen_rule": "db_top_candidate_relaxed",
    }

    with patch.object(r, "_find_species_ids", return_value=[1]), \
            patch.object(r, "map_protein_by_name_species", return_value=protein_result), \
            patch.object(r, "_query", return_value=[]) as query:
        r.find_complex_by_component("Albumin", "Homo sapiens")

    sql = "\n".join(call.args[0] for call in query.call_args_list)
    assert "pcp.protein_complex_id" in sql
    assert "pcp.complex_id" not in sql


# ---------------------------------------------------------------------------
# find_species
# ---------------------------------------------------------------------------

class TestFindSpecies:
    def test_exact_name_match(self):
        r = _make_resolver()
        rows = [{"id": 1, "name": "Homo sapiens", "common_name": "Human", "taxonomy_id": "9606"}]
        with _patch_query(r, rows):
            result = r.find_species("Homo sapiens")
        assert result["status"] == "mapped"
        assert result["confidence"] >= 0.9
        assert result["candidates"][0]["pathbank_species_id"] == 1
        assert result["candidates"][0]["taxonomy_id"] == "9606"

    def test_taxonomy_id_lookup(self):
        r = _make_resolver()
        rows = [{"id": 2, "name": "Mus musculus", "common_name": "Mouse", "taxonomy_id": "10090"}]
        with _patch_query(r, rows):
            result = r.find_species("Mouse", taxonomy_id="10090")
        assert result["status"] == "mapped"
        assert result["candidates"][0]["pathbank_species_id"] == 2

    def test_no_match_returns_unmapped(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.find_species("Unknown organism")
        assert result["status"] == "unmapped"
        assert result["candidates"] == []

    def test_empty_query_returns_unmapped(self):
        r = _make_resolver()
        result = r.find_species("")
        assert result["status"] == "unmapped"
        assert result["reason"] == "empty_query"

    def test_result_shape(self):
        r = _make_resolver()
        rows = [{"id": 3, "name": "Rattus norvegicus", "common_name": "Rat", "taxonomy_id": "10116"}]
        with _patch_query(r, rows):
            result = r.find_species("Rat")
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result
        cand = result["candidates"][0]
        for key in ("pathbank_species_id", "name", "taxonomy_id", "confidence"):
            assert key in cand


class TestHydrateSpeciesReferences:
    def test_explicit_entity_species_is_resolved_before_mapping(self):
        r = _make_resolver()
        payload = {"entities": {"proteins": [{"name": "Albumin", "species": "Homo sapiens"}], "protein_complexes": [], "species": []}}
        rows = [{"id": 1, "name": "Homo sapiens", "common_name": "Human", "taxonomy_id": "9606"}]
        with _patch_query(r, rows):
            report = hydrate_species_references(payload, db=r, use_llm=False)

        protein = payload["entities"]["proteins"][0]
        assert protein["organism"] == "Homo sapiens"
        assert protein["pathbank_species_id"] == 1
        assert protein["species_ref"]["taxonomy_id"] == "9606"
        assert report["matched"] == 1

    def test_single_pathway_species_hydrates_complexes(self):
        r = _make_resolver()
        payload = {
            "entities": {
                "proteins": [],
                "protein_complexes": [{"name": "Hemoglobin complex"}],
                "species": [{"name": "Homo sapiens", "pathbank_species_id": 1}],
            }
        }
        with _patch_query(r, [{"id": 1, "name": "Homo sapiens", "common_name": "Human", "taxonomy_id": "9606"}]):
            hydrate_species_references(payload, db=r, use_llm=False)

        complex_row = payload["entities"]["protein_complexes"][0]
        assert complex_row["species"] == "Homo sapiens"
        assert complex_row["species_id"] == 1

    def test_biological_state_species_is_used_when_no_pathway_species(self):
        r = _make_resolver()
        payload = {
            "entities": {"proteins": [{"name": "Albumin"}], "protein_complexes": [], "species": []},
            "biological_states": [{"name": "cytosol", "species": "Mus musculus", "subcellular_location": "cytosol"}],
            "element_locations": {"protein_locations": [{"protein": "Albumin", "biological_state": "cytosol"}]},
        }
        rows = [{"id": 2, "name": "Mus musculus", "common_name": "Mouse", "taxonomy_id": "10090"}]
        with _patch_query(r, rows):
            hydrate_species_references(payload, db=r, use_llm=False)

        assert payload["entities"]["proteins"][0]["species_ref"]["source"] == "biological_state_species"
        assert payload["entities"]["proteins"][0]["pathbank_species_id"] == 2

    def test_llm_inferred_species_uses_gap_resolver_hook(self):
        r = _make_resolver()
        payload = {"entities": {"proteins": [{"name": "Albumin"}], "protein_complexes": [], "species": []}}
        rows = [{"id": 3, "name": "Rattus norvegicus", "common_name": "Rat", "taxonomy_id": "10116"}]
        with patch(
            "t2pw.mapping.map_ids._infer_species_with_gap_resolver",
            return_value={"name": "Rattus norvegicus", "confidence": 0.8},
        ):
            with _patch_query(r, rows):
                hydrate_species_references(payload, db=r, use_llm=True)

        protein = payload["entities"]["proteins"][0]
        assert protein["species_ref"]["source"] == "gap_resolver_llm"
        assert protein["pathbank_species_id"] == 3

    def test_novel_species_record_is_created_when_db_has_no_match(self):
        r = _make_resolver()
        payload = {"entities": {"proteins": [{"name": "Novelase", "species": "Fictivus exampleus"}], "protein_complexes": [], "species": []}}
        with _patch_query(r, []):
            report = hydrate_species_references(payload, db=r, use_llm=False)

        protein = payload["entities"]["proteins"][0]
        assert protein["species_ref"]["status"] == "novel"
        assert protein["species"] == "Fictivus exampleus"
        assert payload["entities"]["species"][0]["name"] == "Fictivus exampleus"
        assert report["novel"] == 1


# ---------------------------------------------------------------------------
# find_subcellular_location
# ---------------------------------------------------------------------------

class TestFindSubcellularLocation:
    def test_exact_match(self):
        r = _make_resolver()
        rows = [{"id": 5, "name": "Cytoplasm"}]
        with _patch_query(r, rows):
            result = r.find_subcellular_location("Cytoplasm")
        assert result["status"] == "mapped"
        assert result["candidates"][0]["pathbank_subcellular_location_id"] == 5

    def test_no_match(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.find_subcellular_location("Imaginary space")
        assert result["status"] == "unmapped"

    def test_empty_input(self):
        r = _make_resolver()
        result = r.find_subcellular_location("")
        assert result["status"] == "unmapped"
        assert result["reason"] == "empty_query"

    def test_shape(self):
        r = _make_resolver()
        rows = [{"id": 7, "name": "Mitochondria"}]
        with _patch_query(r, rows):
            result = r.find_subcellular_location("Mitochondria")
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result


# ---------------------------------------------------------------------------
# find_cell_type
# ---------------------------------------------------------------------------

class TestFindCellType:
    def test_match(self):
        r = _make_resolver()
        rows = [{"id": 10, "name": "Hepatocyte"}]
        with _patch_query(r, rows):
            result = r.find_cell_type("Hepatocyte")
        assert result["status"] == "mapped"
        assert result["candidates"][0]["pathbank_cell_type_id"] == 10

    def test_no_match(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.find_cell_type("XYZ cell")
        assert result["status"] == "unmapped"

    def test_empty_input(self):
        r = _make_resolver()
        result = r.find_cell_type("")
        assert result["reason"] == "empty_query"


# ---------------------------------------------------------------------------
# find_tissue
# ---------------------------------------------------------------------------

class TestFindTissue:
    def test_match(self):
        r = _make_resolver()
        rows = [{"id": 20, "name": "Liver"}]
        with _patch_query(r, rows):
            result = r.find_tissue("Liver")
        assert result["status"] == "mapped"
        assert result["candidates"][0]["pathbank_tissue_id"] == 20

    def test_no_match(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.find_tissue("Nonexistent tissue")
        assert result["status"] == "unmapped"

    def test_empty_input(self):
        r = _make_resolver()
        result = r.find_tissue("")
        assert result["reason"] == "empty_query"


# ---------------------------------------------------------------------------
# find_biological_state
# ---------------------------------------------------------------------------

class TestFindBiologicalState:
    def _resolver_with_side_effects(self, species_rows, loc_rows, bs_rows):
        """Patch find_species, find_subcellular_location, and _query for BS query."""
        r = _make_resolver()
        # We patch the public helpers so they return structured results, then
        # patch _query for the final biological_states SQL.
        sp_result = {
            "status": "mapped",
            "candidates": [{"pathbank_species_id": 1, "confidence": 1.0}],
            "confidence": 1.0,
            "chosen_rule": "exact_match",
        }
        loc_result = {
            "status": "mapped",
            "candidates": [{"pathbank_subcellular_location_id": 5, "confidence": 1.0}],
            "confidence": 1.0,
            "chosen_rule": "exact_match",
        }
        return r, sp_result, loc_result

    def test_match_found(self):
        r = _make_resolver()
        sp_result = {
            "status": "mapped",
            "candidates": [{"pathbank_species_id": 1, "confidence": 1.0}],
            "confidence": 1.0, "chosen_rule": "exact_match",
        }
        loc_result = {
            "status": "mapped",
            "candidates": [{"pathbank_subcellular_location_id": 5, "confidence": 1.0}],
            "confidence": 1.0, "chosen_rule": "exact_match",
        }
        bs_rows = [{"id": 100, "species_id": 1, "subcellular_location_id": 5, "cell_type_id": None, "tissue_id": None}]
        with patch.object(r, "find_species", return_value=sp_result), \
             patch.object(r, "find_subcellular_location", return_value=loc_result), \
             patch.object(r, "_query", return_value=bs_rows):
            result = r.find_biological_state("Homo sapiens", "Cytoplasm")
        assert result["status"] == "mapped"
        assert result["candidates"][0]["pathbank_biological_state_id"] == 100

    def test_species_not_found(self):
        r = _make_resolver()
        sp_result = {"status": "unmapped", "candidates": [], "confidence": 0.0, "chosen_rule": ""}
        with patch.object(r, "find_species", return_value=sp_result):
            result = r.find_biological_state("Unknown", "Cytoplasm")
        assert result["status"] == "unmapped"
        assert "species_not_found" in result["reason"]

    def test_location_not_found(self):
        r = _make_resolver()
        sp_result = {
            "status": "mapped",
            "candidates": [{"pathbank_species_id": 1, "confidence": 1.0}],
            "confidence": 1.0, "chosen_rule": "exact_match",
        }
        loc_result = {"status": "unmapped", "candidates": [], "confidence": 0.0, "chosen_rule": ""}
        with patch.object(r, "find_species", return_value=sp_result), \
             patch.object(r, "find_subcellular_location", return_value=loc_result):
            result = r.find_biological_state("Homo sapiens", "Unknown place")
        assert result["status"] == "unmapped"
        assert "subcellular_location_not_found" in result["reason"]

    def test_shape(self):
        r = _make_resolver()
        sp_result = {
            "status": "mapped",
            "candidates": [{"pathbank_species_id": 1, "confidence": 0.9}],
            "confidence": 0.9, "chosen_rule": "exact_match",
        }
        loc_result = {
            "status": "mapped",
            "candidates": [{"pathbank_subcellular_location_id": 5, "confidence": 0.9}],
            "confidence": 0.9, "chosen_rule": "exact_match",
        }
        bs_rows = [{"id": 200, "species_id": 1, "subcellular_location_id": 5, "cell_type_id": None, "tissue_id": None}]
        with patch.object(r, "find_species", return_value=sp_result), \
             patch.object(r, "find_subcellular_location", return_value=loc_result), \
             patch.object(r, "_query", return_value=bs_rows):
            result = r.find_biological_state("Homo sapiens", "Cytoplasm")
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result


# ---------------------------------------------------------------------------
# map_compound_by_ids
# ---------------------------------------------------------------------------

class TestMapCompoundByIds:
    def _compound_row(self) -> Dict[str, Any]:
        return {
            "id": 1420, "name": "Water", "short_name": "H2O",
            "hmdb_id": "HMDB0002111", "kegg_id": "C00001",
            "chebi_id": "15377", "pubchem_cid": "962",
            "cas": "7732-18-5", "biocyc_id": "", "chemspider_id": "937",
            "drugbank_id": "",
        }

    def test_hmdb_hit(self):
        r = _make_resolver()
        with _patch_query(r, [self._compound_row()]):
            result = r.map_compound_by_ids({"hmdb": "HMDB0002111"})
        assert result["status"] == "mapped"
        assert result["pathbank_compound_id"] == 1420
        assert result["confidence"] == 1.0
        assert "hmdb" in result["chosen_rule"]

    def test_kegg_hit(self):
        r = _make_resolver()
        with _patch_query(r, [self._compound_row()]):
            result = r.map_compound_by_ids({"kegg": "C00001"})
        assert result["status"] == "mapped"

    def test_no_match(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.map_compound_by_ids({"hmdb": "HMDBXXXXXX"})
        assert result["status"] == "unmapped"

    def test_empty_ids(self):
        r = _make_resolver()
        result = r.map_compound_by_ids({})
        assert result["status"] == "unmapped"
        assert result["reason"] == "no_db_match"

    def test_shape(self):
        r = _make_resolver()
        with _patch_query(r, [self._compound_row()]):
            result = r.map_compound_by_ids({"chebi": "CHEBI:15377"})
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result

    def test_chebi_prefix_stripped(self):
        r = _make_resolver()
        with _patch_query(r, [self._compound_row()]):
            result = r.map_compound_by_ids({"chebi": "CHEBI:15377"})
        assert result["status"] == "mapped"


# ---------------------------------------------------------------------------
# map_compound_by_name
# ---------------------------------------------------------------------------

class TestMapCompoundByName:
    def test_delegates_to_map_compound(self):
        r = _make_resolver()
        expected = {"status": "mapped", "confidence": 0.95, "candidates": [], "chosen_rule": "test", "mapped_ids": {}}
        with patch.object(r, "map_compound", return_value=expected) as mock_mc:
            result = r.map_compound_by_name("Glucose")
        mock_mc.assert_called_once_with("Glucose")
        assert result == expected


# ---------------------------------------------------------------------------
# map_protein_by_ids
# ---------------------------------------------------------------------------

class TestMapProteinByIds:
    def _protein_row(self) -> Dict[str, Any]:
        return {"id": 500, "name": "Albumin", "uniprot_id": "P02768", "gene_name": "ALB", "species_id": 1}

    def test_uniprot_hit(self):
        r = _make_resolver()
        with _patch_query(r, [self._protein_row()]):
            result = r.map_protein_by_ids({"uniprot": "P02768"})
        assert result["status"] == "mapped"
        assert result["pathbank_protein_id"] == 500
        assert result["mapped_ids"]["uniprot"] == "P02768"

    def test_gene_name_hit(self):
        r = _make_resolver()
        with _patch_query(r, [self._protein_row()]):
            result = r.map_protein_by_ids({"gene_name": "ALB"})
        assert result["status"] == "mapped"

    def test_no_ids(self):
        r = _make_resolver()
        result = r.map_protein_by_ids({})
        assert result["status"] == "unmapped"
        assert result["reason"] == "no_ids_provided"

    def test_no_db_match(self):
        r = _make_resolver()
        with _patch_query(r, []):
            result = r.map_protein_by_ids({"uniprot": "ZZZZZZZZ"})
        assert result["status"] == "unmapped"

    def test_shape(self):
        r = _make_resolver()
        with _patch_query(r, [self._protein_row()]):
            result = r.map_protein_by_ids({"uniprot": "P02768"})
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result

    def test_species_filter_applied(self):
        r = _make_resolver()
        # Species lookup returns id=1; protein row has species_id=1 → gets bonus
        with patch.object(r, "_find_species_ids", return_value=[1]):
            with _patch_query(r, [self._protein_row()]):
                result = r.map_protein_by_ids({"uniprot": "P02768"}, species="Homo sapiens")
        assert result["status"] == "mapped"
        assert result["confidence"] > 1.0  # sp_bonus applied


# ---------------------------------------------------------------------------
# map_protein_by_name_species
# ---------------------------------------------------------------------------

class TestMapProteinByNameSpecies:
    def test_delegates_to_map_protein(self):
        r = _make_resolver()
        expected = {"status": "mapped", "confidence": 0.9, "candidates": [], "chosen_rule": "db_top_candidate_relaxed", "mapped_ids": {}}
        with patch.object(r, "map_protein", return_value=expected) as mock_mp:
            result = r.map_protein_by_name_species("Albumin", "Homo sapiens")
        mock_mp.assert_called_once_with("Albumin", "Homo sapiens")
        assert result == expected

    def test_requires_species(self):
        r = _make_resolver()
        result = r.map_protein_by_name_species("Albumin", "")
        assert result["status"] == "unmapped"
        assert result["reason"] == "species_required"


# ---------------------------------------------------------------------------
# row-aware ordered mapping
# ---------------------------------------------------------------------------

class TestRowAwareOrderedMapping:
    def test_compound_hmdb_resolves_before_fuzzy_name(self):
        r = _make_resolver()
        compound_row = {
            "id": 1420, "name": "Water", "short_name": "H2O",
            "hmdb_id": "HMDB0002111", "kegg_id": "C00001",
            "chebi_id": "15377", "pubchem_cid": "962",
            "cas": "7732-18-5", "biocyc_id": "", "chemspider_id": "937",
            "drugbank_id": "",
        }
        with _patch_query(r, [compound_row]), patch.object(r, "map_compound", side_effect=AssertionError("fuzzy should not run")):
            result = r.map_compound_row({"name": "not water by name", "mapped_ids": {"hmdb": "HMDB0002111"}})
        assert result["status"] == "mapped"
        assert result["pathbank_compound_id"] == 1420
        assert result["resolution"]["status"] == "matched"
        assert "hmdb" in result["chosen_rule"]

    def test_ambiguous_compound_fuzzy_is_not_marked_novel(self):
        r = _make_resolver()
        fuzzy = {
            "status": "unmapped",
            "reason": "ambiguous",
            "provider": "PathBankDB",
            "source": "db",
            "confidence": 0.78,
            "candidates": [{"pathbank_compound_id": 1, "name": "A"}, {"pathbank_compound_id": 2, "name": "B"}],
        }
        with _patch_query(r, []), patch.object(r, "map_compound", return_value=fuzzy):
            result = r.map_compound_row({"name": "near match"})
        assert result["status"] == "unmapped"
        assert result["reason"] == "ambiguous"
        assert result["resolution"]["status"] == "ambiguous"
        assert result["resolution"]["status"] != "novel"

    def test_protein_without_species_does_not_name_match(self):
        r = _make_resolver()
        with patch.object(r, "map_protein", side_effect=AssertionError("unsafe name match should not run")):
            result = r.map_protein_row({"name": "Albumin"}, "")
        assert result["status"] == "unmapped"
        assert result["reason"] == "needs_species"
        assert result["resolution"]["issue"] == "needs_species"

    def test_protein_uniprot_resolves_without_species(self):
        r = _make_resolver()
        protein_row = {"id": 500, "name": "Albumin", "uniprot_id": "P02768", "gene_name": "ALB", "species_id": 1}
        with _patch_query(r, [protein_row]), patch.object(r, "map_protein", side_effect=AssertionError("fuzzy should not run")):
            result = r.map_protein_row({"name": "Albumin", "mapped_ids": {"uniprot": "P02768"}}, "")
        assert result["status"] == "mapped"
        assert result["pathbank_protein_id"] == 500
        assert result["mapped_ids"]["uniprot"] == "P02768"
        assert result["resolution"]["status"] == "matched"


# ---------------------------------------------------------------------------
# map_protein_complex
# ---------------------------------------------------------------------------

class TestMapProteinComplex:
    def test_species_required(self):
        r = _make_resolver()
        result = r.map_protein_complex("SomeComplex", "")
        assert result["status"] == "unmapped"
        assert result["reason"] == "species_required"

    def test_species_not_found(self):
        r = _make_resolver()
        with patch.object(r, "_find_species_ids", return_value=[]):
            result = r.map_protein_complex("SomeComplex", "Unknown species")
        assert result["status"] == "unmapped"
        assert "species_not_found" in result["reason"]

    def test_complex_found(self):
        r = _make_resolver()
        rows = [{"id": 999, "name": "Hemoglobin complex", "species_id": 1}]
        with patch.object(r, "_find_species_ids", return_value=[1]):
            with _patch_query(r, rows):
                result = r.map_protein_complex("Hemoglobin complex", "Homo sapiens")
        assert result["status"] == "mapped"
        assert result["pathbank_complex_id"] == 999

    def test_no_match(self):
        r = _make_resolver()
        with patch.object(r, "_find_species_ids", return_value=[1]):
            with _patch_query(r, []):
                result = r.map_protein_complex("NonexistentComplex", "Homo sapiens")
        assert result["status"] == "unmapped"

    def test_shape(self):
        r = _make_resolver()
        rows = [{"id": 888, "name": "Test complex", "species_id": 1}]
        with patch.object(r, "_find_species_ids", return_value=[1]):
            with _patch_query(r, rows):
                result = r.map_protein_complex("Test complex", "Homo sapiens")
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result


# ---------------------------------------------------------------------------
# find_complex_by_component
# ---------------------------------------------------------------------------

class TestFindComplexByComponent:
    def test_species_required(self):
        r = _make_resolver()
        result = r.find_complex_by_component("Albumin", "")
        assert result["status"] == "unmapped"
        assert result["reason"] == "species_required"

    def test_species_not_found(self):
        r = _make_resolver()
        with patch.object(r, "_find_species_ids", return_value=[]):
            result = r.find_complex_by_component("Albumin", "Unknown species")
        assert result["status"] == "unmapped"
        assert "species_not_found" in result["reason"]

    def test_component_not_found(self):
        r = _make_resolver()
        protein_result = {"status": "unmapped", "candidates": [], "confidence": 0.0, "chosen_rule": ""}
        with patch.object(r, "_find_species_ids", return_value=[1]), \
             patch.object(r, "map_protein_by_name_species", return_value=protein_result), \
             _patch_query(r, []):
            result = r.find_complex_by_component("UnknownProtein", "Homo sapiens")
        assert result["status"] == "unmapped"

    def test_complexes_found(self):
        r = _make_resolver()
        protein_result = {
            "status": "mapped",
            "pathbank_protein_id": 500,
            "candidates": [], "confidence": 0.9, "chosen_rule": "db_top_candidate_relaxed",
        }
        complex_rows = [
            {"id": 301, "name": "Albumin-drug complex", "species_id": 1},
            {"id": 302, "name": "Albumin-fatty acid complex", "species_id": 1},
        ]
        with patch.object(r, "_find_species_ids", return_value=[1]), \
             patch.object(r, "map_protein_by_name_species", return_value=protein_result), \
             _patch_query(r, complex_rows):
            result = r.find_complex_by_component("Albumin", "Homo sapiens")
        assert result["status"] == "mapped"
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["pathbank_complex_id"] == 301

    def test_shape(self):
        r = _make_resolver()
        protein_result = {
            "status": "mapped",
            "pathbank_protein_id": 500,
            "candidates": [], "confidence": 0.9, "chosen_rule": "db_top_candidate_relaxed",
        }
        complex_rows = [{"id": 400, "name": "Test complex", "species_id": 1}]
        with patch.object(r, "_find_species_ids", return_value=[1]), \
             patch.object(r, "map_protein_by_name_species", return_value=protein_result), \
             _patch_query(r, complex_rows):
            result = r.find_complex_by_component("SomeProtein", "Homo sapiens")
        for key in ("status", "candidates", "chosen_rule", "confidence"):
            assert key in result


# ---------------------------------------------------------------------------
# resolve_mapping_gaps
# ---------------------------------------------------------------------------

def _make_payload_with_entities(
    proteins: List[Dict] = None,
    compounds: List[Dict] = None,
) -> Dict[str, Any]:
    return {
        "entities": {
            "proteins": proteins or [],
            "compounds": compounds or [],
            "protein_complexes": [],
            "species": [],
        },
        "biological_states": [],
    }


def _make_mapping_report(entities: List[Dict]) -> Dict[str, Any]:
    return {"summary": {}, "entities": entities}


class TestResolveMappingGaps:
    def test_resolves_unmapped_protein(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(proteins=[{"name": "Albumin"}])
        report = _make_mapping_report([
            {"name": "Albumin", "entity_type": "protein", "status": "unmapped", "reason": "no_db_match", "organism": "Homo sapiens"},
        ])
        mapped_result = {
            "status": "mapped", "confidence": 0.9, "chosen_rule": "db_top_candidate_relaxed",
            "candidates": [{"name": "Albumin", "uniprot": "P02768"}],
            "mapped_ids": {"uniprot": "P02768"},
            "pathbank_protein_id": 500,
        }
        with patch.object(r, "map_protein_by_name_species", return_value=mapped_result):
            out = resolve_mapping_gaps(payload, report, r, global_organism="Homo sapiens")

        assert out["resolved_count"] == 1
        assert out["total"] == 1
        assert out["rows"][0]["resolved"] is True
        assert out["rows"][0]["mapped_ids"] == {"uniprot": "P02768"}

    def test_resolves_unmapped_compound(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(compounds=[{"name": "Glucose"}])
        report = _make_mapping_report([
            {"name": "Glucose", "entity_type": "compound", "status": "unmapped", "reason": "no_db_match", "organism": ""},
        ])
        mapped_result = {
            "status": "mapped", "confidence": 0.95, "chosen_rule": "db_top_candidate_relaxed",
            "candidates": [{"name": "Glucose"}],
            "mapped_ids": {"hmdb": "HMDB0000122", "kegg": "C00031"},
            "pathbank_compound_id": 99,
        }
        with patch.object(r, "map_compound_by_name", return_value=mapped_result):
            out = resolve_mapping_gaps(payload, report, r)

        assert out["resolved_count"] == 1
        patched_compound = out["patched_payload"]["entities"]["compounds"][0]
        assert patched_compound["mapped_ids"]["hmdb"] == "HMDB0000122"
        assert patched_compound["mapping_meta"]["db_gap_resolved"] is True
        assert patched_compound["pathbank_compound_id"] == 99

    def test_unresolvable_entity_stays_in_rows(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(proteins=[{"name": "Mystery protein"}])
        report = _make_mapping_report([
            {"name": "Mystery protein", "entity_type": "protein", "status": "unmapped", "reason": "no_db_match", "organism": ""},
        ])
        unmapped_result = {
            "status": "unmapped", "reason": "no_db_match", "confidence": 0.0,
            "chosen_rule": "", "candidates": [],
        }
        with patch.object(r, "map_protein_by_name_species", return_value=unmapped_result):
            out = resolve_mapping_gaps(payload, report, r)

        assert out["resolved_count"] == 0
        assert out["total"] == 1
        assert out["rows"][0]["resolved"] is False

    def test_does_not_mutate_original_payload(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(compounds=[{"name": "ATP"}])
        report = _make_mapping_report([
            {"name": "ATP", "entity_type": "compound", "status": "unmapped", "reason": "no_db_match", "organism": ""},
        ])
        mapped_result = {
            "status": "mapped", "confidence": 0.9, "chosen_rule": "db_top_candidate_relaxed",
            "candidates": [], "mapped_ids": {"hmdb": "HMDB0001"}, "pathbank_compound_id": 1,
        }
        with patch.object(r, "map_compound_by_name", return_value=mapped_result):
            out = resolve_mapping_gaps(payload, report, r)

        # Original payload unchanged
        assert "mapped_ids" not in payload["entities"]["compounds"][0]
        # Patched payload has the IDs
        assert out["patched_payload"]["entities"]["compounds"][0].get("mapped_ids", {}).get("hmdb") == "HMDB0001"

    def test_ambiguous_entities_are_also_retried(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(proteins=[{"name": "Transferrin"}])
        report = _make_mapping_report([
            {"name": "Transferrin", "entity_type": "protein", "status": "unmapped", "reason": "ambiguous", "organism": "Homo sapiens"},
        ])
        mapped_result = {
            "status": "mapped", "confidence": 0.88, "chosen_rule": "db_top_candidate_relaxed",
            "candidates": [{"name": "Transferrin"}], "mapped_ids": {"uniprot": "P02787"},
            "pathbank_protein_id": 600,
        }
        with patch.object(r, "map_protein_by_name_species", return_value=mapped_result):
            out = resolve_mapping_gaps(payload, report, r, global_organism="Homo sapiens")

        assert out["resolved_count"] == 1
        assert out["rows"][0]["was_unmapped_reason"] == "ambiguous"

    def test_empty_report_returns_zero(self):
        r = _make_resolver()
        payload = _make_payload_with_entities()
        out = resolve_mapping_gaps(payload, _make_mapping_report([]), r)
        assert out["resolved_count"] == 0
        assert out["total"] == 0
        assert out["rows"] == []

    def test_result_shape(self):
        r = _make_resolver()
        payload = _make_payload_with_entities(compounds=[{"name": "Fructose"}])
        report = _make_mapping_report([
            {"name": "Fructose", "entity_type": "compound", "status": "unmapped", "reason": "no_db_match", "organism": ""},
        ])
        with patch.object(r, "map_compound_by_name", return_value={"status": "unmapped", "confidence": 0.0, "chosen_rule": "", "candidates": []}):
            out = resolve_mapping_gaps(payload, report, r)

        for key in ("resolved_count", "total", "rows", "patched_payload"):
            assert key in out
        row = out["rows"][0]
        for key in ("type", "name", "resolved", "confidence", "chosen_rule", "mapped_ids"):
            assert key in row
