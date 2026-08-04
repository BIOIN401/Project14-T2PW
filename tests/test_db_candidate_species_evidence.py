"""The identity ladder's species rung must be able to read PathBank DB rows.

A UniProt candidate row carries ``organism``; a PathBank DB row carries
``species_id`` -- an integer foreign key into PathBank's own ``species`` table --
and nothing else. Rung 3 (:func:`_candidate_species_verdict`) reads organism /
taxonomy names, so every DB-resolved protein came back ``species: unknown`` and
the ladder failed closed with ``identity_evidence_missing``, discarding
correctly resolved accessions (ALAS2 -> P22557 among them).

The fix carries the species row the resolver *already looked up* onto the
candidate. It supplies evidence to a rung; it does not relax one. The
companion test
``tests/test_protein_export_policy.py::test_a_candidate_with_no_organism_is_not_verified``
pins the other half of that statement: a candidate with no species evidence is
still rejected.
"""
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
    MappingCache,
    PathBankDbResolver,
    _map_protein_with_strategy,
    verify_real_protein_identity,
)

ORGANISM = "Homo sapiens"

# A real row, copied from a production artifact: PathBank protein 17, resolved
# from the entity name "ALAS2" in a heme-biosynthesis leg.
ALAS2_ROW: Dict[str, Any] = {
    "pathbank_protein_id": 17,
    "name": "5-aminolevulinate synthase, erythroid-specific, mitochondrial",
    "gene_name": "ALAS2",
    "uniprot": "P22557",
    "species_id": 1,
    "score": 1.08,
}

HUMAN_SPECIES_ROW: Dict[str, Any] = {
    "id": 1,
    "name": "Homo sapiens",
    "common_name": "Human",
    "taxonomy_id": "9606",
}


def _make_resolver() -> PathBankDbResolver:
    """A resolver that never attempts a live DB connection."""
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
    r._species_meta = {}
    r._species_meta_lookups = set()
    return r


def _query_dispatch(species_rows: List[Dict[str, Any]], protein_rows: List[Dict[str, Any]]):
    """Side effect for ``_query``: species rows for the species table, else proteins."""

    def _dispatch(sql: str, params: Any) -> List[Dict[str, Any]]:
        return list(species_rows) if "FROM species" in sql else list(protein_rows)

    return _dispatch


# ── the ladder itself ──────────────────────────────────────────────────────


def test_a_production_shaped_pathbank_candidate_verifies_with_species_evidence() -> None:
    """The positive regression the gap let through: this row is a correct answer.

    ALAS2 *is* P22557 in *Homo sapiens*, and every rung can say so once the
    species row is on the candidate -- rung 4 passes on the exact gene symbol,
    rung 5 on the DB score, rung 6 because there is no rival.
    """

    candidate = dict(ALAS2_ROW, organism=ORGANISM, taxonomy_id="9606")

    verdict = verify_real_protein_identity(
        "ALAS2",
        candidates=[candidate],
        mapped_ids={"uniprot": "P22557", "pathbank_protein_id": "17"},
        organism=ORGANISM,
        source="db",
        result={},
    )

    assert verdict["verified"] is True
    assert verdict["checks"]["species"] == "ok"


def test_the_same_row_without_species_evidence_is_still_rejected() -> None:
    """The rung is unchanged: what the fix supplies is evidence, not leniency."""

    verdict = verify_real_protein_identity(
        "ALAS2",
        candidates=[dict(ALAS2_ROW)],
        mapped_ids={"uniprot": "P22557", "pathbank_protein_id": "17"},
        organism=ORGANISM,
        source="db",
        result={},
    )

    assert verdict["verified"] is False
    assert verdict["checks"]["species"] == "unknown"


# ── the resolver's side map and the builders that read it ──────────────────


def test_find_species_ids_keeps_the_columns_it_used_to_discard() -> None:
    r = _make_resolver()

    with patch.object(r, "_query", return_value=[HUMAN_SPECIES_ROW]):
        ids = r._find_species_ids(ORGANISM)

    assert ids == [1]  # unchanged return type: eight call sites depend on it
    assert r._species_meta[1]["name"] == "Homo sapiens"
    assert r._species_meta[1]["taxonomy_id"] == "9606"


def test_map_protein_row_stamps_species_onto_its_candidates() -> None:
    r = _make_resolver()
    protein_rows = [
        {"id": 17, "name": ALAS2_ROW["name"], "uniprot_id": "P22557", "gene_name": "ALAS2", "species_id": 1},
    ]

    with patch.object(r, "_query", side_effect=_query_dispatch([HUMAN_SPECIES_ROW], protein_rows)):
        result = r.map_protein_row({"name": "ALAS2", "uniprot_id": "P22557"}, ORGANISM)

    assert result["status"] == "mapped"
    candidate = result["candidates"][0]
    assert candidate["species_id"] == 1
    assert candidate["organism"] == "Homo sapiens"
    assert candidate["taxonomy_id"] == "9606"


def test_map_protein_stamps_species_onto_its_candidates() -> None:
    r = _make_resolver()
    protein_rows = [
        {
            "id": 17,
            "name": ALAS2_ROW["name"],
            "uniprot_id": "P22557",
            "gene_name": "ALAS2",
            "species_id": 1,
            "synonyms": "ALAS2",
        },
    ]

    with patch.object(r, "_query", side_effect=_query_dispatch([HUMAN_SPECIES_ROW], protein_rows)):
        result = r.map_protein("ALAS2", ORGANISM)

    candidate = result["candidates"][0]
    assert candidate["organism"] == "Homo sapiens"
    assert candidate["taxonomy_id"] == "9606"


def test_a_species_id_the_resolver_never_saw_is_left_untouched() -> None:
    """Missing evidence stays missing -- ``unknown`` is the right verdict there."""

    r = _make_resolver()
    row = dict(ALAS2_ROW, species_id=4242)

    with patch.object(r, "_query", return_value=[HUMAN_SPECIES_ROW]):
        r.stamp_candidate_species([row], ORGANISM)

    assert "organism" not in row
    assert "taxonomy_id" not in row


def test_stamping_never_overwrites_species_a_candidate_already_declares() -> None:
    r = _make_resolver()
    row = dict(ALAS2_ROW, organism="Mus musculus", taxonomy_id="10090")

    with patch.object(r, "_query", return_value=[HUMAN_SPECIES_ROW]):
        r.stamp_candidate_species([row], ORGANISM)

    assert row["organism"] == "Mus musculus"
    assert row["taxonomy_id"] == "10090"


# ── the cache path: the fix is inert without read-time hydration ───────────


def test_a_cached_candidate_with_no_species_fields_verifies_after_hydration(tmp_path: Path) -> None:
    """``db_key`` carries no builder version, so stored rows are replayed as-is.

    The first call writes a pre-fix result to the cache (stamping suppressed);
    the second reads it back and must still verify, because hydration happens on
    the return path rather than at build time.
    """

    cache = MappingCache(tmp_path / "id_mapping_cache.json")
    protein_row = {"name": "ALAS2", "uniprot_id": "P22557"}
    legacy_result = {
        "status": "mapped",
        "provider": "PathBankDB",
        "source": "db",
        "mapped_ids": {"uniprot": "P22557", "pathbank_protein_id": "17"},
        "pathbank_protein_id": 17,
        "confidence": 1.08,
        "chosen_rule": "direct_id_match:uniprot_id",
        "candidates": [dict(ALAS2_ROW)],
    }

    writer = _make_resolver()
    with patch.object(writer, "map_protein_row", return_value=legacy_result), \
            patch.object(writer, "stamp_candidate_species"), \
            patch.object(writer, "_query", return_value=[HUMAN_SPECIES_ROW]):
        stale = _map_protein_with_strategy(
            id_source="db",
            db=writer,
            client=None,  # type: ignore[arg-type]
            cache=cache,
            name="ALAS2",
            organism=ORGANISM,
            protein_row=protein_row,
        )

    # Pre-fix behaviour, reproduced: the ladder had no species evidence to read.
    assert stale.get("identity_verdict", {}).get("verified") is not True

    reader = _make_resolver()
    with patch.object(reader, "_query", side_effect=_query_dispatch([HUMAN_SPECIES_ROW], [])) as query:
        hydrated = _map_protein_with_strategy(
            id_source="db",
            db=reader,
            client=None,  # type: ignore[arg-type]
            cache=cache,
            name="ALAS2",
            organism=ORGANISM,
            protein_row=protein_row,
        )

    assert hydrated["status"] == "mapped"
    assert hydrated["mapped_ids"]["uniprot"] == "P22557"
    assert hydrated["identity_verdict"]["verified"] is True
    assert hydrated["identity_verdict"]["checks"]["species"] == "ok"
    # One species lookup, not one per protein: the resolver memoises the organism.
    reader.stamp_candidate_species([dict(ALAS2_ROW)], ORGANISM)
    assert len(query.call_args_list) == 1
