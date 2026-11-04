"""ORCH-804 (ORCH-801 audit F7): an empty biological-state reference must not land
in an arbitrary paper compartment.

G9: ``test_empty_reference_does_not_take_the_first_paper_state`` and
``test_empty_reference_takes_the_state_its_located_members_share`` are CORRECTIONS;
both fail behaviourally on base ``563d39db`` (the reaction lands in the first state)
and pass at the tip. The other two pin unchanged behaviour (safety twin;
no-neutral-state fallback) and pass on both.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.compound_resolution import NO_DB_RESOLVER  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402

PEROX = {"name": "peroxisomal state", "species": "Oryza sativa", "subcellular_location": "peroxisome"}
AUTO = {"name": "__auto_state__", "species": "Oryza sativa", "subcellular_location": "cell"}


def _payload(states, reaction_state=None, member_state=None):
    reaction = {
        "name": "PAL",
        "inputs": [{"name": "phenylalanine"}],
        "outputs": [{"name": "trans-cinnamic acid"}],
    }
    if reaction_state is not None:
        reaction["biological_state"] = reaction_state
    payload = {
        "entities": {
            "species": [{"name": "Oryza sativa"}],
            "subcellular_locations": [{"name": "peroxisome"}, {"name": "cell"}, {"name": "cytosol"}],
            "compounds": [
                {"name": "phenylalanine", "db_status": "unmatched"},
                {"name": "trans-cinnamic acid", "db_status": "unmatched"},
            ],
        },
        "biological_states": states,
        "processes": {"reactions": [reaction]},
    }
    if member_state:
        payload["element_locations"] = {"compound_locations": [
            {"compound": "phenylalanine", "biological_state": member_state},
            {"compound": "trans-cinnamic acid", "biological_state": member_state},
        ]}
    return payload


def _build(payload):
    ir, _ = build_pwml_ir(payload, strict_db=False, db_resolver=NO_DB_RESOLVER)
    return ir, {s["key"]: s["name"] for s in ir["biological_states"]}


def _reaction_state_name(payload):
    ir, names = _build(payload)
    return names[ir["processes"]["reactions"][0]["biological_state_key"]]


def test_empty_reference_does_not_take_the_first_paper_state():
    assert _reaction_state_name(_payload([PEROX, AUTO])) == "__auto_state__"


def test_named_reference_still_lands_in_its_state():
    assert _reaction_state_name(_payload([PEROX, AUTO], "peroxisomal state")) == "peroxisomal state"


def test_no_neutral_state_keeps_first_state_so_the_pwml_still_exports():
    # No __auto_state__ (ensure_autostates did not run): unchanged fallback, no new error.
    assert _reaction_state_name(_payload([PEROX])) == "peroxisomal state"


CYTO = {"name": "cytosolic state", "species": "Oryza sativa", "subcellular_location": "cytosol"}


def test_empty_reference_takes_the_state_its_located_members_share():
    # Both members sit in the peroxisome per element_locations: the reaction goes
    # there (base put it in the first state, cytosol), and no member is drawn twice.
    ir, names = _build(_payload([CYTO, PEROX, AUTO], member_state="peroxisomal state"))
    assert names[ir["processes"]["reactions"][0]["biological_state_key"]] == "peroxisomal state"
    per_entity = {}
    for loc in ir["locations"]:
        per_entity.setdefault(loc["entity_key"], set()).add(loc["biological_state_key"])
    assert per_entity and all(len(v) == 1 for v in per_entity.values()), per_entity
