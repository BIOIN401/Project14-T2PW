import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.gap_resolver import _ensure_biological_state  # noqa: E402


def test_ensure_biological_state_matches_by_species_location_and_optional_context() -> None:
    payload = {
        "biological_states": [
            {
                "name": "human_cytosol_hepatocyte_liver",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
                "cell_type": "hepatocyte",
                "tissue": "liver",
            },
            {
                "name": "mouse_cytosol_hepatocyte_liver",
                "species": "Mus musculus",
                "subcellular_location": "cytosol",
                "cell_type": "hepatocyte",
                "tissue": "liver",
            },
        ]
    }

    matched = _ensure_biological_state(
        payload,
        "cytosol",
        "Homo sapiens",
        cell_type="hepatocyte",
        tissue="liver",
    )
    created = _ensure_biological_state(payload, "cytosol", "Rattus norvegicus")

    assert matched == "human_cytosol_hepatocyte_liver"
    assert created not in {"human_cytosol_hepatocyte_liver", "mouse_cytosol_hepatocyte_liver"}
    assert len(payload["biological_states"]) == 3


def test_ensure_biological_state_does_not_create_without_species() -> None:
    payload = {"biological_states": []}

    state_name = _ensure_biological_state(payload, "cytosol", "")

    assert state_name == ""
    assert payload["biological_states"] == []
