"""ORCH-742b -- the Streamlit "Reaction Evidence" section, driven through AppTest.

**G9 LABEL: NEW ACCEPTANCE TESTS FOR A NEW CAPABILITY.** The view does not exist at the
base SHA ``9f13188e``; no base failure is claimed. The script under test is the view
alone (``render_reaction_evidence()``), not the whole app, so no pipeline, model or
network call can run. Run this file ALONE in its own bounded process with
``T2PW_OFFLINE_CURATOR=1`` (AppTest files stall a shared pytest process).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

from t2pw.app.reaction_evidence import DUPLICATE_WARNING, FINAL_MAPPED_FILENAME
from t2pw.app.reaction_evidence_view import DOWNLOAD_LABEL, NO_RUN_MESSAGE

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "reaction_evidence" / "enterobactin_strict"


def _script():
    from t2pw.app.reaction_evidence_view import render_reaction_evidence

    render_reaction_evidence()


def _app() -> AppTest:
    return AppTest.from_function(_script, default_timeout=60)


def _texts(elements):
    return [e.value for e in elements]


def test_no_run_loaded_says_so_and_offers_no_download():
    at = _app().run()
    assert not at.exception
    assert _texts(at.subheader) == ["Reaction Evidence"]
    assert _texts(at.info) == [NO_RUN_MESSAGE]
    assert len(at.download_button) == 0 and len(at.expander) == 0


def test_missing_directory_warns():
    at = _app().run()
    at.text_input[0].input(str(FIXTURE / "does-not-exist")).run()
    assert not at.exception
    assert _texts(at.warning)[0].startswith("Not a directory:")


def test_missing_artifacts_read_not_recorded(tmp_path):
    at = _app().run()
    at.text_input[0].input(str(tmp_path)).run()
    assert not at.exception
    assert _texts(at.warning)[0].startswith("None of coverage_summary.json, quarantine_report.json")

    shutil.copy(FIXTURE / "coverage_summary.json", tmp_path / "coverage_summary.json")
    at.run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["0.75", "true", "9", "1"]
    assert _texts(at.info) == ["Surviving reactions: Not recorded (no final_mapped.json)"]
    assert [e.label for e in at.expander] == ["Quarantined / rejected candidate processes (Not recorded)"]
    assert "Release status: Not recorded" in "\n".join(_texts(at.code))


def test_leg_directory_renders_header_reactions_quarantine_and_download():
    at = _app().run()
    at.text_input[0].input(str(FIXTURE)).run()
    assert not at.exception
    assert [b.label for b in at.download_button] == [DOWNLOAD_LABEL]
    assert [(m.label, m.value) for m in at.metric] == [
        ("Coverage", "0.75"),
        ("Minimum core satisfied", "true"),
        ("Surviving processes", "9"),
        ("Quarantined processes", "1"),
    ]
    labels = [e.label for e in at.expander]
    assert labels[0] == "Reaction 1: chorismate -> isochorismate"
    assert len(labels) == 6 and labels[-1] == "Quarantined / rejected candidate processes (1)"
    codes = "\n".join(_texts(at.code))
    assert "Paper: PMC12096016" in codes and "Unmatched (4): NADH, ATP, EntD, Fur" in codes
    assert "QUARANTINED 1  [pre-quarantine /processes/reactions/5; reaction; lock rxn_lock_004]" in codes
    assert not at.warning


def test_session_run_directory_uses_session_payload_when_no_final_mapped_on_disk(tmp_path):
    for name in ("coverage_summary.json", "quarantine_report.json", "removed_entity_report.json"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    payload = json.loads((FIXTURE / FINAL_MAPPED_FILENAME).read_text(encoding="utf-8"))
    at = _app()
    at.session_state["quarantine_artifacts"] = {"coverage_summary.json": str(tmp_path / "coverage_summary.json")}
    at.session_state["post_pipeline_artifacts"] = {"canonical_export_payload": payload}
    at.run()
    assert not at.exception
    assert at.text_input[0].value == str(tmp_path)
    assert len([e for e in at.expander if e.label.startswith("Reaction ")]) == 5
    codes = "\n".join(_texts(at.code))
    assert "Surviving reactions read from: session post_pipeline_artifacts[canonical_export_payload]" in codes


def test_section_follows_a_run_that_completes_after_the_first_render():
    """The page is usually open BEFORE the run: the field must pick up the new directory."""

    at = _app().run()
    assert _texts(at.info) == [NO_RUN_MESSAGE]
    at.session_state["quarantine_artifacts"] = {"coverage_summary.json": str(FIXTURE / "coverage_summary.json")}
    at.run()
    assert not at.exception
    assert at.text_input[0].value == str(FIXTURE)
    assert len([e for e in at.expander if e.label.startswith("Reaction ")]) == 5


def test_duplicate_reaction_warning_is_shown(tmp_path):
    leg = tmp_path / "leg"
    shutil.copytree(FIXTURE, leg)
    path = leg / FINAL_MAPPED_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processes"]["reactions"].append(dict(payload["processes"]["reactions"][0], name="twin"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    at = _app().run()
    at.text_input[0].input(str(leg)).run()
    assert not at.exception
    assert _texts(at.warning) == [
        f"Reaction 1: {DUPLICATE_WARNING} (same substrates and products as reaction 6)",
        f"Reaction 6: {DUPLICATE_WARNING} (same substrates and products as reaction 1)",
    ]
    assert len([e for e in at.expander if e.label.startswith("Reaction ")]) == 6
