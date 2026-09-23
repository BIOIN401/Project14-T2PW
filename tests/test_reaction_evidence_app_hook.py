"""ORCH-742b -- the hook in the REAL ``streamlit_app.py`` renders the Reaction Evidence section.

**G9 LABEL: NEW ACCEPTANCE TEST FOR A NEW CAPABILITY** (the section does not exist at the
base SHA ``9f13188e``; no base failure is claimed). The whole app page is executed, but
no pipeline button is pressed, so no stage, model or network call runs. Run this file
ALONE in its own bounded process with ``T2PW_OFFLINE_CURATOR=1``.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from t2pw.app.reaction_evidence_view import DOWNLOAD_LABEL, NO_RUN_MESSAGE

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"
FIXTURE = ROOT / "tests" / "fixtures" / "reaction_evidence" / "enterobactin_strict"


def test_real_app_page_ends_with_the_reaction_evidence_section():
    at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    assert not at.exception
    assert "Reaction Evidence" in [s.value for s in at.subheader]
    assert NO_RUN_MESSAGE in [i.value for i in at.info]

    # What the app's quarantine boundary leaves in session state after a run.
    at.session_state["quarantine_artifacts"] = {"coverage_summary.json": str(FIXTURE / "coverage_summary.json")}
    at.run()
    assert not at.exception
    assert DOWNLOAD_LABEL in [b.label for b in at.download_button]
    reactions = [e.label for e in at.expander if e.label.startswith("Reaction ")]
    assert reactions[0] == "Reaction 1: chorismate -> isochorismate" and len(reactions) == 5
