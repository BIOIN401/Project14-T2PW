"""NEW ACCEPTANCE TEST (W2, new capability).

Pins ``t2pw.service.organisms`` (reads the committed ``data/webserver/organisms.json``)
and the export logic of ``scripts/webserver/export_organisms.py`` against a fake
cursor. No database is used or opened by any test here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from t2pw.rag.eligibility import ObservedContext, RequestedScope, compare_organism
from t2pw.service import organisms
from t2pw.service.organisms import get_by_id, get_by_label, load_organisms, scope_for

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("export_organisms", ROOT / "scripts" / "webserver" / "export_organisms.py")
export_organisms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_organisms)


def test_committed_json_is_unique_sorted_and_scoped():
    orgs = load_organisms()
    doc = json.loads((ROOT / "data" / "webserver" / "organisms.json").read_text(encoding="utf-8"))
    assert len(orgs) == doc["count"] == len(doc["organisms"]) > 1000
    assert len({o.id for o in orgs}) == len(orgs)
    assert len({o.label.casefold() for o in orgs}) == len(orgs)
    assert [(o.label.casefold(), o.id) for o in orgs] == sorted((o.label.casefold(), o.id) for o in orgs)
    assert all(o.scope.strip() and o.scope == o.scope.strip() and o.label.strip() for o in orgs)


def test_known_species_and_lookups():
    human = get_by_label("Homo sapiens")
    assert human is not None and human.id == 1 and human.taxonomy_id == "9606"
    assert get_by_label("  homo SAPIENS ") == human and get_by_id(1) == human
    assert scope_for(1) == scope_for("Homo sapiens") == "Homo sapiens"
    assert scope_for("Escherichia coli") == "Escherichia coli"
    assert get_by_label("Escherichia coli (strain K12)").scope == "Escherichia coli (strain K12)"
    assert get_by_id(-1) is None and get_by_label("no such species") is None
    with pytest.raises(KeyError):
        scope_for("no such species")
    with pytest.raises(KeyError):
        scope_for(-1)


def test_scope_string_is_accepted_by_the_pipeline_matcher():
    # Strain-qualified PathBank names must still match the species a paper names.
    for key, observed in ((1, "human"), ("Escherichia coli (strain K12)", "Escherichia coli")):
        scope = RequestedScope(requested_organism=scope_for(key))
        assert compare_organism(scope, ObservedContext(observed_organisms=[observed])) == "match"


def test_import_opens_no_db_connection():
    code = "import sys, t2pw.service.organisms as o; o.load_organisms(); sys.exit(any(m.startswith('pymysql') for m in sys.modules))"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    assert subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT).returncode == 0


class FakeCursor:
    def __init__(self, species, pathway_ids):
        self.species, self.pathway_ids, self.executed, self._last = species, pathway_ids, [], None

    def execute(self, sql, params=None):
        assert sql.lstrip().upper().startswith("SELECT"), sql
        self.executed.append(sql)
        self._last = sql

    def fetchall(self):
        if "FROM species" in self._last:
            return list(self.species)
        return [{"species_id": i} for i in self.pathway_ids]


def test_export_logic_with_fake_cursor():
    rows = [
        {"id": 7, "name": " Serratia marcescens ", "taxonomy_id": "615", "common_name": "", "classification": "Prokaryote"},
        {"id": 3, "name": "Zea mays", "taxonomy_id": "0123", "common_name": "Maize", "classification": "Eukaryote"},
        {"id": 9, "name": "serratia marcescens", "taxonomy_id": None, "common_name": None, "classification": None},
        {"id": 1, "name": "Homo sapiens", "taxonomy_id": "9606", "common_name": "Human", "classification": "Eukaryote"},
        {"id": 5, "name": "   ", "taxonomy_id": "1", "common_name": None, "classification": None},
    ]
    cur = FakeCursor(rows, [1, 7])
    entries = export_organisms.fetch_entries(cur)
    assert all(s.lstrip().upper().startswith("SELECT") for s in cur.executed) and len(cur.executed) == 2
    assert [e["id"] for e in entries] == [1, 7, 9, 3]  # blank name dropped; sorted by label then id
    by_id = {e["id"]: e for e in entries}
    assert by_id[7]["label"] == "Serratia marcescens [PathBank id 7]" and by_id[7]["scope"] == "Serratia marcescens"
    assert by_id[9]["label"] == "serratia marcescens [PathBank id 9]"
    assert by_id[3]["taxonomy_id"] == "0123" and by_id[7]["common_name"] is None
    assert by_id[1]["has_pathways"] and by_id[7]["has_pathways"] and not by_id[3]["has_pathways"]
    assert export_organisms.render(entries) == export_organisms.render(export_organisms.fetch_entries(FakeCursor(rows[::-1], [7, 1])))
