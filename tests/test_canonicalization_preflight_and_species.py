"""Tests for the two T2PW name-canonicalization fixes.

TASK A: a loud preflight warning fires when a generation run has neither a
reachable resolution DB nor an offline index covering the entities (so emitted
names may be non-canonical and will collide on PathWhiz import).

TASK B: the emitted species name is deterministic per taxonomy_id -- the same
taxonomy_id yields the same name regardless of which input name variant the
LLM/extraction produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import (  # noqa: E402
    _deterministic_species_name,
    build_pwml_ir,
)
from t2pw.pwml.name_index import PathwhizNameIndex  # noqa: E402


def _species_payload(species_name: str, *, taxonomy_id: str = "863372") -> dict:
    """A minimal but valid payload with a single taxonomy-identified species."""
    return {
        "entities": {
            "species": [{"name": species_name, "taxonomy_id": taxonomy_id}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "glycolate", "mapped_ids": {"kegg": "C00160"}},
            ],
            "proteins": [],
            "protein_complexes": [],
        },
        "biological_states": [
            {
                "name": "state",
                "species": species_name,
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


# ---------------------------------------------------------------------------
# TASK A -- preflight warning
# ---------------------------------------------------------------------------
def test_preflight_warns_when_no_db_and_no_covering_index() -> None:
    payload = _species_payload("Herbaspirillum huttiense IAM 15032")

    # name_index=None => no offline index; db_resolver=False sentinel below is
    # not passed, so from_env is attempted. Force the offline path by passing an
    # empty index that covers nothing and no db_resolver connection.
    empty_index = PathwhizNameIndex({})
    ir, report = build_pwml_ir(
        payload,
        strict_db=False,
        db_resolver=_UnavailableDb(),
        name_index=empty_index,
    )

    assert "preflight" in report, "preflight detail should be attached to the report"
    codes = {w.get("code") for w in report["warnings"]}
    assert "noncanonical_names_collision_risk" in codes
    detail = report["preflight"]
    assert detail["db_available"] is False
    # The glycolate compound carries a KEGG id but no DB/index hit -> at risk.
    assert "glycolate" in detail["compounds"]
    # The taxonomy-identified species is at risk too.
    assert any("Herbaspirillum huttiense" in name for name in detail["species"])


def test_preflight_silent_when_offline_index_covers_entities() -> None:
    payload = _species_payload("Herbaspirillum huttiense IAM 15032")

    # An offline index that covers both the compound (by KEGG) and the species
    # (by taxonomy) -> nothing is at risk, so no preflight warning.
    index = PathwhizNameIndex(
        {
            "compounds": {
                "kegg": {"C00160": 900},
                "by_id": {"900": {"name": "Glycolic acid", "kegg": "C00160"}},
            },
            "species": {
                "taxonomy": {"863372": 500},
                "by_id": {"500": {"name": "Herbaspirillum huttiense", "taxonomy_id": "863372"}},
            },
        }
    )
    ir, report = build_pwml_ir(
        payload,
        strict_db=False,
        db_resolver=_UnavailableDb(),
        name_index=index,
    )

    assert "preflight" not in report
    codes = {w.get("code") for w in report["warnings"]}
    assert "noncanonical_names_collision_risk" not in codes


class _UnavailableDb:
    """A db_resolver whose driver is missing, i.e. not usable for resolution."""

    last_error = "db_unavailable_for_test"

    def available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# TASK B -- deterministic species naming
# ---------------------------------------------------------------------------
def test_deterministic_species_name_strips_strain_qualifier() -> None:
    assert (
        _deterministic_species_name("Herbaspirillum huttiense IAM 15032")
        == "Herbaspirillum huttiense"
    )
    assert (
        _deterministic_species_name("Escherichia coli str. K-12 substr. MG1655")
        == "Escherichia coli"
    )
    # A plain binomial is untouched; the leading binomial is never stripped.
    assert _deterministic_species_name("Homo sapiens") == "Homo sapiens"
    assert _deterministic_species_name("Herbaspirillum huttiense") == "Herbaspirillum huttiense"


def test_same_taxonomy_id_yields_same_species_name_across_variants() -> None:
    variants = [
        "Herbaspirillum huttiense IAM 15032",
        "Herbaspirillum huttiense",
        "Herbaspirillum huttiense subsp. huttiense IAM 15032",
    ]
    emitted = []
    for variant in variants:
        ir, _report = build_pwml_ir(
            _species_payload(variant, taxonomy_id="863372"),
            strict_db=False,
            db_resolver=_UnavailableDb(),
            name_index=None,
        )
        names = [row["name"] for row in ir["species"]]
        assert len(names) == 1
        emitted.append(names[0])

    assert len(set(emitted)) == 1, f"species name drifted across variants: {emitted}"
    assert emitted[0] == "Herbaspirillum huttiense"


def test_novel_species_without_taxonomy_keeps_extraction_name() -> None:
    # No taxonomy_id => truly novel => extraction name preserved verbatim.
    payload = _species_payload("Wobblegut fictitious STRAIN-42", taxonomy_id="")
    ir, _report = build_pwml_ir(
        payload, strict_db=False, db_resolver=_UnavailableDb(), name_index=None
    )
    names = [row["name"] for row in ir["species"]]
    assert names == ["Wobblegut fictitious STRAIN-42"]
