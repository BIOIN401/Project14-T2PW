"""ORCH-758a (defect D2): ion charge notation is not a composite separator.

ORCH-755 sour taste (PMC7059786, strict): Stage 1 wrote the compound ``Ca++`` and
the protein ``voltage-gated Ca ++ channels``. Strict ``normalize_process_payload``
read both "+" runs as ``A + B`` separators and raised, the app carried on with the
un-normalized payload, and the pre-export Stage-3 revalidation stopped a correct
``review_required`` pathway on ``actor_schema_not_canonical`` (ORCH-758 § 4 D2).

G9: the ``test_g9_*`` tests are corrections of pre-existing behaviour and FAIL on
base 3582b0d7 (strict normalization raises). The ``test_preserved_*`` tests pin
behaviour that must be identical at base and tip: real composites still split
(or still abort strict mode) exactly as before.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    normalize_composites,
    normalize_process_payload,
    validate_no_composites,
)
from t2pw.pipeline.stage_contracts import validate_post_normalization  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "orch758a" / "sour_taste_PMC7059786_merged_excerpt.json"


def _transport_payload(ion: str, transporter: str) -> dict:
    """One ion moved across the apical membrane by one channel, Stage-1 actor shape."""
    return {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "subcellular_locations": [{"name": "apical membrane"}, {"name": "cytoplasm"}],
            "compounds": [{"name": ion, "class": "ion"}],
            "proteins": [{"name": transporter, "organism": "Homo sapiens", "uniprot": "Q7RTM1"}],
            "protein_complexes": [],
        },
        "biological_states": [
            {"name": "apical state", "species": "Homo sapiens", "subcellular_location": "apical membrane"},
            {"name": "cyto state", "species": "Homo sapiens", "subcellular_location": "cytoplasm"},
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": ion, "biological_state": "apical state"},
                {"compound": ion, "biological_state": "cyto state"},
            ],
            "protein_locations": [{"protein": transporter, "biological_state": "apical state"}],
        },
        "processes": {
            "reactions": [],
            "interactions": [],
            "transports": [
                {
                    "name": f"{ion} entry through {transporter}",
                    "cargo": ion,
                    "from_biological_state": "apical state",
                    "to_biological_state": "cyto state",
                    "transporters": [{"protein": transporter, "evidence": f"{ion} enters through {transporter}"}],
                    "evidence": f"{ion} enters through {transporter}",
                }
            ],
        },
    }


def _actor_schema_findings(normalized: dict, report: dict) -> list:
    contract = validate_post_normalization(normalized, report.get("gate") or {})
    return [e for e in contract.get("errors", []) if e.get("code") == "actor_schema_not_canonical"]


def _assert_canonical_transporters(normalized: dict) -> None:
    for transport in normalized["processes"]["transports"]:
        for actor in transport.get("transporters", []):
            assert isinstance(actor, dict), actor
            assert actor.get("entity"), actor
            assert actor.get("entity_type") == "protein", actor


# --- G9: fails at base (strict normalization raises), passes at tip ----------


@pytest.mark.parametrize(
    "ion, transporter",
    [
        ("Ca++", "OTOP1"),  # the ORCH-755 ion, as Stage 1 wrote it
        ("Fe+++", "OTOP1"),
        ("Ca ++", "OTOP1"),  # the paper's spaced form
        ("Ca++", "voltage-gated Ca ++ channels"),  # the ORCH-755 channel name
        ("Ca++", "voltage-gated Ca++ channels"),
        ("K+", "Na+/K+-ATPase"),
        ("Ca2+", "Ca2+-ATPase"),
    ],
)
def test_g9_strict_normalization_completes_with_charge_notation(ion: str, transporter: str) -> None:
    normalized, report = normalize_process_payload(_transport_payload(ion, transporter), mode="pathwhiz")

    # Nothing renamed: the ion and the channel keep the names Stage 1 gave them.
    assert [row["name"] for row in normalized["entities"]["compounds"]] == [ion]
    assert transporter in [row["name"] for row in normalized["entities"]["proteins"]]
    assert normalized["entities"]["protein_complexes"] == []
    (transport,) = normalized["processes"]["transports"]
    assert transport["cargo"] == ion
    assert [a["entity"] for a in transport["transporters"]] == [transporter]
    _assert_canonical_transporters(normalized)
    assert _actor_schema_findings(normalized, report) == []
    assert report["summary"]["n_plus_tokens_remaining"] == 0


@pytest.mark.parametrize("name", ["Ca++", "Fe+++", "Ca ++", "voltage-gated Ca ++ channels", "Mg2+/ATP"])
def test_g9_validate_no_composites_accepts_charge_notation(name: str) -> None:
    validate_no_composites({"entities": {"compounds": [{"name": name}]}, "processes": {}})


def test_g9_orch755_sour_taste_excerpt_normalizes_in_strict_mode() -> None:
    """Verbatim rows of the ORCH-755 leg's merged_payload.json.

    Source: runs_papers/ORCH-755/legs/sour_taste-strict/2026-10-21_0139/papers/
    PMC7059786/strict/merged_payload.json (sha256 bec156e6...c2f). Kept: transports
    0, 1, 3, 5 (three OTOP1 H+ rows, the Ca++ channel row), the H+ / Ca++ compounds,
    the OTOP1 / channel proteins, and the states and locations those rows name.
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    otop1_rows = [t for t in payload["processes"]["transports"] if t["transporters"][0].get("protein") == "OTOP1"]
    # Still the Stage-1 actor shape that reached the exporter un-normalized.
    assert len(otop1_rows) == 3 and all("entity" not in t["transporters"][0] for t in otop1_rows)

    normalized, report = normalize_process_payload(payload, mode="pathwhiz")

    assert "Ca++" in [row["name"] for row in normalized["entities"]["compounds"]]
    transporters = [a["entity"] for t in normalized["processes"]["transports"] for a in t["transporters"]]
    assert transporters.count("OTOP1") == 3
    assert "voltage-gated Ca ++ channels" in transporters
    _assert_canonical_transporters(normalized)
    assert _actor_schema_findings(normalized, report) == []
    assert normalized["entities"]["protein_complexes"] == []


# --- Preservation: identical at base and tip ---------------------------------


@pytest.mark.parametrize("name", ["NADH + H+", "glucose+fructose", "ATP + Ca2+", "Ca2+ + calmodulin", "Ca++ + calmodulin"])
def test_preserved_compound_composite_still_aborts_strict_mode(name: str) -> None:
    payload = {"entities": {"compounds": [{"name": name}], "proteins": [], "protein_complexes": []}, "processes": {}}
    with pytest.raises(ValueError, match="has no protein-like left component"):
        normalize_composites(payload)


@pytest.mark.parametrize("name", ["NADH + H+", "A + B", "EntB+EntE"])
def test_preserved_validate_no_composites_still_flags_real_composites(name: str) -> None:
    with pytest.raises(ValueError, match="Composite validation failed"):
        validate_no_composites({"entities": {"compounds": [{"name": name}]}, "processes": {}})


@pytest.mark.parametrize("ion", ["Ca2+", "Ca 2+", "NAD+", "H+", "Mg2+"])
def test_preserved_single_charge_ions_were_already_not_composites(ion: str) -> None:
    payload = {"entities": {"compounds": [{"name": ion}], "proteins": [], "protein_complexes": []}, "processes": {}}
    normalize_composites(payload)
    assert [row["name"] for row in payload["entities"]["compounds"]] == [ion]


@pytest.mark.parametrize("name", ["EntB + EntE", "EntB+EntE"])
def test_preserved_two_protein_composite_still_materializes(name: str) -> None:
    payload = {
        "entities": {
            "compounds": [],
            "proteins": [{"name": "EntB", "uniprot": "P0ADI4"}, {"name": "EntE", "uniprot": "P10378"}],
            "protein_complexes": [{"name": name}],
        },
        "processes": {"reactions": [], "transports": []},
    }
    normalize_composites(payload)
    assert [row["name"] for row in payload["entities"]["protein_complexes"]] == ["EntB:EntE"]
