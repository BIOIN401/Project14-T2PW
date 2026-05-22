import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.compound_templates import select_compound_template_id  # noqa: E402


def test_pathwhiz_id_lookup():
    assert select_compound_template_id({"pathwhiz_id": 414}) == 42  # ATP
    assert select_compound_template_id({"pathwhiz_id": 1144}) == 60  # NADH


def test_name_lookup_exact():
    assert select_compound_template_id({"name": "ATP"}) == 42
    assert select_compound_template_id({"name": "NADH"}) == 60
    assert select_compound_template_id({"name": "NAD+"}) == 59
    assert select_compound_template_id({"name": "CoA-SH"}) == 85
    assert select_compound_template_id({"name": "FADH2"}) == 138
    assert select_compound_template_id({"name": "Water"}) == 49
    assert select_compound_template_id({"name": "Hydrogen Ion"}) == 55


def test_drugbank_fallback():
    rec = {"name": "Paracetamol", "mapped_ids": {"drugbank": "DB00316"}}
    assert select_compound_template_id(rec) == 157


def test_default_fallback():
    assert select_compound_template_id({"name": "Glycerol"}) == 81
    assert select_compound_template_id({"name": "Pyruvate"}) == 81
    assert select_compound_template_id({}) == 81


def test_pathwhiz_id_takes_priority_over_name():
    rec = {"pathwhiz_id": 414, "name": "Glycerol"}
    assert select_compound_template_id(rec) == 42
