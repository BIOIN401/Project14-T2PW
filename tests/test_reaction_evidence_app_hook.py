"""ORCH-742b -- the hook in the REAL ``streamlit_app.py`` renders the Reaction Evidence section.

**G9 LABEL: NEW ACCEPTANCE TEST FOR A NEW CAPABILITY** (the section does not exist at the
base SHA ``9f13188e``; no base failure is claimed). The whole app page is executed, but
no pipeline button is pressed, so no stage, model or network call runs. Run this file
ALONE in its own bounded process with ``T2PW_OFFLINE_CURATOR=1``.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from t2pw.app.reaction_evidence import DUPLICATE_WARNING
from t2pw.app.reaction_evidence_view import DOWNLOAD_LABEL, NO_RUN_MESSAGE

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"
BUNDLE = ROOT / "tests" / "fixtures" / "reaction_evidence" / "enterobactin_strict.bundle.json"


def test_real_app_page_ends_with_the_reaction_evidence_section(tmp_path):
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    reactions = bundle["final_mapped.json"]["processes"]["reactions"]
    reactions.append(dict(reactions[0], name="twin"))  # a duplicate, so the section has a flag to raise
    for name, content in bundle.items():
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        (tmp_path / name).write_text(text, encoding="utf-8")

    at = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    assert not at.exception
    assert "Reaction Evidence" in [s.value for s in at.subheader]
    assert NO_RUN_MESSAGE in [i.value for i in at.info]
    # batch.driver._collect_app_text reads exactly these three lists.
    said = [[e.value for e in at.error], [e.value for e in at.warning], [e.value for e in at.exception]]

    # What the app's quarantine boundary leaves in session state after a run.
    at.session_state["quarantine_artifacts"] = {"coverage_summary.json": str(tmp_path / "coverage_summary.json")}
    at.run()
    assert [[e.value for e in at.error], [e.value for e in at.warning], [e.value for e in at.exception]] == said
    assert DOWNLOAD_LABEL in [b.label for b in at.download_button]
    labels = [e.label for e in at.expander if e.label.startswith("Reaction ")]
    assert labels[0] == "Reaction 1: chorismate -> isochorismate" and len(labels) == 6
    assert sum(DUPLICATE_WARNING in m.value for m in at.markdown) == 2
