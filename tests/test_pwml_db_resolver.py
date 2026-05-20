from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.db_resolver import PathWhizCompoundResolver  # noqa: E402


GLUCOSE_ROWS = [
    {
        "id": 77,
        "name": "D-Glucose",
        "short_name": "D-Glc",
        "hmdb_id": "HMDB0000122",
        "kegg_id": "C00031",
        "chebi_id": "4167",
        "pubchem_cid": "5793",
        "pwc_id": "PW_C000077",
    },
    {
        "id": 57903,
        "name": "D-glucopyranose",
        "short_name": "D-glucopyranose",
        "hmdb_id": "",
        "kegg_id": "",
        "chebi_id": "4167",
        "pubchem_cid": "",
        "pwc_id": "PW_C057903",
    },
]


class _CompoundDb:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        value = str(params[0]).strip().lower()
        column = ""
        for candidate in ["pwc_id", "hmdb_id", "kegg_id", "chebi_id", "pubchem_cid", "id"]:
            if f"{candidate})" in sql or f" {candidate}=" in sql:
                column = candidate
                break
        if not column:
            return []
        return [row for row in self.rows if str(row.get(column) or "").strip().lower() == value]


def test_identifier_ranking_disambiguates_glucose_chebi_collision() -> None:
    resolver = PathWhizCompoundResolver(_CompoundDb(GLUCOSE_ROWS))

    result = resolver.resolve(
        {
            "name": "Glucose",
            "pathbank_compound_id": 77,
            "pathwhiz_id": 77,
            "mapped_ids": {
                "hmdb": "HMDB0000122",
                "kegg": "C00031",
                "chebi": "CHEBI:4167",
                "pubchem": "5793",
                "pwc_id": "PW_C000077",
            },
        }
    )

    assert result["status"] == "matched"
    assert result["chosen"]["id"] == 77
    assert result["chosen_rule"] == "pathbank_compound_id"
    assert result["confidence"] == 1.0


def test_mapped_id_aliases_are_scored_before_ambiguous() -> None:
    resolver = PathWhizCompoundResolver(_CompoundDb(GLUCOSE_ROWS))

    result = resolver.resolve(
        {
            "name": "Glucose",
            "mapped_ids": {
                "pathbank id": "77",
                "hmdb": "HMDB0000122",
                "kegg": "cpd:C00031",
                "chebi": "CHEBI:4167",
                "pubchem": "5793",
                "pwc_id": "PW_C000077",
            },
        }
    )

    assert result["status"] == "matched"
    assert result["chosen"]["id"] == 77
    assert result["chosen"]["chebi_id"] == "4167"


def test_multiple_strong_ids_beat_single_chebi_collision() -> None:
    resolver = PathWhizCompoundResolver(_CompoundDb(GLUCOSE_ROWS))

    result = resolver.resolve(
        {
            "name": "Glucose",
            "mapped_ids": {
                "hmdb": "HMDB0000122",
                "kegg": "C00031",
                "chebi": "CHEBI:4167",
                "pubchem": "5793",
            },
        }
    )

    assert result["status"] == "matched"
    assert result["chosen"]["id"] == 77
    assert result["chosen_rule"] == "multiple_strong_ids_exact"


def test_chebi_only_collision_remains_ambiguous() -> None:
    resolver = PathWhizCompoundResolver(_CompoundDb(GLUCOSE_ROWS))

    result = resolver.resolve({"name": "Glucose", "mapped_ids": {"chebi": "CHEBI:4167"}})

    assert result["status"] == "ambiguous"
    assert {candidate["id"] for candidate in result["candidates"]} == {77, 57903}
    assert result["confidence"] == 0.9
