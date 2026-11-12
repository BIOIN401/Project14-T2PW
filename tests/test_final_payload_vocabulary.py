"""Final payload P3 (B2): vocabulary name map for locations, cell types and tissues (flag T2PW_FINAL_PAYLOAD)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from test_pwml_writer import build_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.vocabulary_map import PATHWHIZ_ROWS, SYNONYMS, map_vocabulary_name  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402


def test_every_synonym_target_is_a_pathwhiz_row_and_no_synonym_is_already_a_row() -> None:
    for kind, table in SYNONYMS.items():
        rows = {n.casefold() for n in PATHWHIZ_ROWS[kind]}
        for source, target in table.items():
            assert source == source.casefold()
            assert target in PATHWHIZ_ROWS[kind], (kind, source, target)
            assert source not in rows, (kind, source)


def test_map_outcomes() -> None:
    assert map_vocabulary_name("subcellular-location", "ER", None) == ("Endoplasmic Reticulum", None, "mapped")
    assert map_vocabulary_name("subcellular-location", "Extracellular", "") == ("Extracellular Space", None, "mapped")
    assert map_vocabulary_name("cell-type", "neurons", None)[::2] == ("Neuron", "mapped")
    assert map_vocabulary_name("tissue", "Taste Buds", None) == ("Taste Bud", None, "mapped")
    assert map_vocabulary_name("subcellular-location", "cytosol", None) == ("cytosol", None, "exact")  # case-insensitive hit
    assert map_vocabulary_name("tissue", "root", None) == ("root", None, "unmapped")
    assert map_vocabulary_name("subcellular-location", "cell", "GO:0005623") == ("cell", "GO:0005623", "new_row")
    # an id that agrees with the target row is dropped (name-only lookup); one that disagrees blocks the mapping
    assert map_vocabulary_name("subcellular-location", "ER", "GO:0005783")[2] == "mapped"
    assert map_vocabulary_name("subcellular-location", "ER", "GO:9999999") == ("ER", "GO:9999999", "new_row")
    # near-miss entities stay unmapped
    for kind, name in [("cell-type", "Type II cell"), ("cell-type", "Type III cell"), ("tissue", "central nervous system"),
                       ("tissue", "young leaf"), ("subcellular-location", "etioplast"), ("cell-type", "erythroid cells")]:
        assert map_vocabulary_name(kind, name, None)[2] == "unmapped", name


def _build():
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "ER", "pathwhiz_id": 2}, {"name": "cytosol", "pathwhiz_id": 3},
                                      {"name": "thylakoid membrane", "pathwhiz_id": 4},
                                      {"name": "cell", "ontology_id": "GO:0005623", "pathwhiz_id": 5}],
            "cell_types": [{"name": "neurons", "pathwhiz_id": 6}, {"name": "MG-63", "pathwhiz_id": 7}],
            "tissues": [{"name": "taste buds", "pathwhiz_id": 8}],
            "compounds": [{"name": "A", "pathbank_compound_id": 103}],
        },
        "biological_states": [{"name": "s", "species": "Homo sapiens", "subcellular_location": "ER", "cell_type": "neurons",
                               "tissue": "taste buds", "compartment_canonical": "ER"}],
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == []
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400,
                           width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    builder = DeterministicPwmlBuilder(extraction=ir, signature=discover_structure_signature(ROOT / "reference" / "PW000001.pwml"),
                                       args=args)
    return builder, builder.build().root


def _names(root):
    got = {}
    for tag, path in (("subcellular-location", ".//subcellular-locations/subcellular-location"),
                      ("cell-type", ".//cell-types/cell-type"), ("tissue", ".//tissues/tissue")):
        got[tag] = {n.findtext("name"): n.findtext("ontology-id") or None for n in root.findall(path)}
    return got


def test_flag_off_writes_names_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    builder, root = _build()
    got = _names(root)
    assert set(got["subcellular-location"]) == {"ER", "cytosol", "thylakoid membrane", "cell"}
    assert set(got["cell-type"]) == {"neurons", "MG-63"} and set(got["tissue"]) == {"taste buds"}
    assert not any(k.startswith("vocabulary_") for k in builder.layout_debug_counts) and builder.vocabulary_outcomes == []


def test_flag_on_maps_counts_and_keeps_ontology_id(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    builder, root = _build()
    got = _names(root)
    assert got["subcellular-location"] == {"Endoplasmic Reticulum": None, "cytosol": None, "thylakoid membrane": None,
                                           "cell": "GO:0005623"}
    assert set(got["cell-type"]) == {"Neuron", "MG-63"} and set(got["tissue"]) == {"Taste Bud"}
    counts = {k: v for k, v in builder.layout_debug_counts.items() if k.startswith("vocabulary_")}
    assert counts == {"vocabulary_mapped": 3, "vocabulary_exact": 1, "vocabulary_unmapped": 2, "vocabulary_new_row": 1}
    assert {(o["name"], o["outcome"]) for o in builder.vocabulary_outcomes} == {
        ("ER", "mapped"), ("neurons", "mapped"), ("taste buds", "mapped"), ("thylakoid membrane", "unmapped"),
        ("MG-63", "unmapped"), ("cell", "new_row")}
    # the biological state still points at the (renamed) location by id
    loc_ids = {n.findtext("id") for n in root.findall(".//subcellular-locations/subcellular-location")}
    assert {n.text for n in root.findall(".//biological-states/biological-state/subcellular-location-id")} <= loc_ids
