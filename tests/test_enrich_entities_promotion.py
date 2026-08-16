from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# C-051 / D-015 (LOCKED): the exporter no longer resolves compound identity
# after the canonical freeze, so a raw extraction payload is taken through the
# pre-freeze stage that now does that work. helpers_prefreeze asserts the stage
# actually ruled on every compound row.
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.mapping.enrich_entities import enrich_payload  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402


def test_compound_enrichment_fields_are_promoted_without_fetch(tmp_path: Path) -> None:
    payload = {
        "entities": {
            "compounds": [
                {
                    "name": "ATP",
                    "mapped_ids": {"pubchem": "5957", "cas": "56-65-5", "biocyc": "ATP", "chemspider": "5742"},
                    "enrichment": {
                        "compound": {
                            "primary_name": "Adenosine triphosphate",
                            "synonyms": ["ATP", "Adenosine triphosphate"],
                            "inchi": "InChI=1S/C10H16N5O13P3",
                            "inchikey": "ZKHQWZAMYRWXGA-KQYNXXCUSA-N",
                            "taxonomy": {"class": "Nucleoside phosphates"},
                            "cross_references": {
                                "hmdb_id": "hmdb0000538",
                                "kegg_id": "cpd:C00002",
                                "chebi_id": "15422",
                                "pubchem_id": "5957",
                            },
                        }
                    },
                }
            ]
        },
        "element_locations": {
            "compound_locations": [
                {"compound": "ATP", "biological_state": "cytosol", "biological_state_id": 7}
            ]
        },
    }

    result = enrich_payload(payload, cache_path=tmp_path / "enrichment_cache.json")
    compound = result["payload"]["entities"]["compounds"][0]

    assert compound["hmdb_id"] == "HMDB0000538"
    assert compound["kegg_id"] == "C00002"
    assert compound["chebi_id"] == "CHEBI:15422"
    assert compound["pubchem_cid"] == "5957"
    assert compound["cas"] == "56-65-5"
    assert compound["biocyc_id"] == "ATP"
    assert compound["chemspider_id"] == "5742"
    assert compound["moldb_inchi"] == "InChI=1S/C10H16N5O13P3"
    assert compound["moldb_inchikey"] == "ZKHQWZAMYRWXGA-KQYNXXCUSA-N"
    assert compound["short_name"] == "Adenosine triphosphate"
    assert compound["synonyms"] == ["ATP", "Adenosine triphosphate"]
    assert compound["class"] == "Nucleoside phosphates"
    assert compound["element_states"] == [{"biological_state": "cytosol", "biological_state_id": 7}]


def test_protein_enrichment_fields_are_promoted_without_fetch(tmp_path: Path) -> None:
    payload = {
        "entities": {
            "proteins": [
                {
                    "name": "Hexokinase",
                    "uniprot_id": "p19367",
                    "species_ref": {"pathbank_species_id": 1},
                    "enrichment": {
                        "protein": {
                            "uniprot": {
                                "uniprot_id": "P19367",
                                "recommended_name": "Hexokinase-1",
                                "alternative_names": ["Hexokinase type I"],
                                "gene_names": ["HK1", "HXK1"],
                                "enzyme_commission": ["2.7.1.1"],
                            }
                        }
                    },
                }
            ]
        }
    }

    result = enrich_payload(payload, cache_path=tmp_path / "enrichment_cache.json")
    protein = result["payload"]["entities"]["proteins"][0]

    assert protein["uniprot"] == "P19367"
    assert protein["gene_name"] == "HK1"
    assert protein["ec_number"] == "2.7.1.1"
    assert protein["ec_numbers"] == ["2.7.1.1"]
    assert protein["species_id"] == 1
    assert protein["synonyms"] == ["Hexokinase type I", "HK1", "HXK1", "Hexokinase-1"]


def test_promoted_fields_survive_pwml_ir_build() -> None:
    payload = {
        "entities": {
            "compounds": [
                {
                    "name": "ATP",
                    "hmdb_id": "HMDB0000538",
                    "kegg_id": "C00002",
                    "chebi_id": "CHEBI:15422",
                    "pubchem_cid": "5957",
                    "cas": "56-65-5",
                    "biocyc_id": "ATP",
                    "chemspider_id": "5742",
                    "moldb_inchi": "InChI=1S/C10H16N5O13P3",
                    "moldb_inchikey": "ZKHQWZAMYRWXGA-KQYNXXCUSA-N",
                    "short_name": "ATP",
                    "synonyms": ["Adenosine triphosphate"],
                    "class": "Nucleoside phosphates",
                    "type": "metabolite",
                    "element_states": [{"biological_state": "cytosol"}],
                }
            ],
            "proteins": [
                {
                    "name": "Hexokinase",
                    "uniprot": "P19367",
                    "gene_name": "HK1",
                    "ec_number": "2.7.1.1",
                    "species_id": 1,
                    "synonyms": ["Hexokinase type I"],
                }
            ],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    ir, _report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=False)
    compound = ir["entities"]["compounds"][0]
    protein = ir["entities"]["proteins"][0]

    assert compound["hmdb_id"] == "HMDB0000538"
    assert compound["moldb_inchikey"] == "ZKHQWZAMYRWXGA-KQYNXXCUSA-N"
    assert compound["element_states"] == [{"biological_state": "cytosol"}]
    assert protein["uniprot"] == "P19367"
    assert protein["gene_name"] == "HK1"
    assert protein["ec_number"] == "2.7.1.1"
    assert protein["species_id"] == 1
