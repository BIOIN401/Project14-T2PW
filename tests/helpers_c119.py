"""C-119 fixture loader. Shared by the G9 base-failure proof and the regressions.

IT IMPORTS ONLY THE STANDARD LIBRARY AND ``t2pw.pipeline.gate_reports``, both of
which exist unchanged at base SHA ``6746a8d31e993d8dc6968ad46534b4e48b9c7bee``. That
is deliberate and load-bearing: ``tests/test_c119_g9_base_failure_proof.py`` has to
IMPORT and RUN at the base and fail there on a VALUE, so nothing it depends on may
be a symbol this card introduced.

Each file in ``tests/fixtures/c119/`` is four objects copied verbatim out of one
archived strict leg by ``docs/pwml_recovery_sprint/evidence/c119_build_fixtures.py``.
That script asserts, before writing, that the archived ``final_mapped.json``
reproduces both canonical digests the archived ``final_stage3_gate_report.json``
recorded -- so the pairing this module reassembles is PROVED, not assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.pipeline.gate_reports import (  # noqa: E402
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
)

FIXTURES = ROOT / "tests" / "fixtures" / "c119"

#: The key every one of these archives carries its ``audit_round`` snapshot under.
SNAPSHOT = "post_normalization_contract_report"


def bundle(stem: str) -> Dict[str, Any]:
    """One archived leg's four objects, as the fixture builder wrote them."""

    return json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))


def artifacts(stem: str) -> Dict[str, Any]:
    """The app's ``post_pipeline_artifacts``, reassembled from one archived leg.

    Three keys are added to the report slice the archive persists, and none of them
    is an invention:

    * :data:`FINAL_GATE_REPORT_KEY` -- ``final_stage3_gate_report.json``, verbatim;
    * :data:`CANONICAL_PAYLOAD_KEY` -- ``final_mapped.json``, verbatim, which the
      fixture builder proved is the object that gate report validated;
    * ``artifact_set_version: 1`` -- ``streamlit_app.py:4392`` wraps every artifact
      set this build returns in ``stamp_artifact_set`` unconditionally, and every
      contract report in these archives carries ``report_schema_version: 1``, which
      only ``stamp_report`` from the same build writes. ``contract_reports.json`` is
      a report-only slice and simply does not persist top-level scalars.
    """

    data = bundle(stem)
    reassembled = dict(data["contract_reports"])
    reassembled[FINAL_GATE_REPORT_KEY] = data["final_stage3_gate_report"]
    reassembled[CANONICAL_PAYLOAD_KEY] = data["canonical_export_payload"]
    reassembled["artifact_set_version"] = 1
    return reassembled
