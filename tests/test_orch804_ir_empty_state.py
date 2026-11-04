"""ORCH-804 (ORCH-801 audit F7): an empty biological-state reference must not land
in an arbitrary paper compartment.

G9: ``test_empty_reference_does_not_take_the_first_paper_state`` is a CORRECTION;
it fails behaviourally on base ``563d39db`` (the PAL reaction lands in ``bs_1``,
the peroxisome) and passes at the tip. The other two pin behaviour that is
unchanged (safety twin; no-neutral-state fallback) and pass on both.
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


def _payload(states, reaction_state=None):
    reaction = {
        "name": "PAL",
        "inputs": [{"name": "phenylalanine"}],
        "outputs": [{"name": "trans-cinnamic acid"}],
    }
    if reaction_state is not None:
        reaction["biological_state"] = reaction_state
    return {
        "entities": {
            "species": [{"name": "Oryza sativa"}],
            "subcellular_locations": [{"name": "peroxisome"}, {"name": "cell"}],
            "compounds": [
                {"name": "phenylalanine", "db_status": "unmatched"},
                {"name": "trans-cinnamic acid", "db_status": "unmatched"},
            ],
        },
        "biological_states": states,
        "processes": {"reactions": [reaction]},
    }


def _reaction_state_name(payload):
    ir, _ = build_pwml_ir(payload, strict_db=False, db_resolver=NO_DB_RESOLVER)
    names = {s["key"]: s["name"] for s in ir["biological_states"]}
    return names[ir["processes"]["reactions"][0]["biological_state_key"]]


def test_empty_reference_does_not_take_the_first_paper_state():
    assert _reaction_state_name(_payload([PEROX, AUTO])) == "__auto_state__"


def test_named_reference_still_lands_in_its_state():
    assert _reaction_state_name(_payload([PEROX, AUTO], "peroxisomal state")) == "peroxisomal state"


def test_no_neutral_state_keeps_first_state_so_the_pwml_still_exports():
    # No __auto_state__ (ensure_autostates did not run): unchanged fallback, no new error.
    assert _reaction_state_name(_payload([PEROX])) == "peroxisomal state"
