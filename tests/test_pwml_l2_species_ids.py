"""L2 (correctness F2): distinct species never share a PWML species id."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

IR = json.loads((ROOT / "tests" / "fixtures" / "pwml_layout" / "ir_run4.json").read_text(encoding="utf-8"))


def _species_ids(monkeypatch, species):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    ir = copy.deepcopy(IR)
    ir["species"] = ir["species"] + species
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF",
                           ref=str(ROOT / "reference" / "PW000001.pwml"))
    sig = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    root = etree.fromstring(etree.tostring(DeterministicPwmlBuilder(extraction=ir, signature=sig, args=args).build().root))
    return {s.findtext("name"): s.findtext("id") for s in root.iter("species") if s.findtext("name")}


def test_species_without_a_database_id_never_takes_another_species_id(monkeypatch):
    # idx 134: Homo sapiens has PathBank id 1, the unresolved Mus musculus (listed first) drew the counter's 1 as well
    taken = [{"key": f"sp_x{i}", "name": f"Species {i}", "pathwhiz_id": i, "taxonomy_id": str(1000 + i)} for i in range(1, 9)]
    unresolved = [{"key": "sp_mus", "name": "Mus musculus", "taxonomy_id": "10090"}]
    ids = _species_ids(monkeypatch, unresolved + taken)
    assert len(ids) == len(set(ids.values())) == 10, ids
    assert [ids[f"Species {i}"] for i in range(1, 9)] == [str(i) for i in range(1, 9)]  # database ids untouched
