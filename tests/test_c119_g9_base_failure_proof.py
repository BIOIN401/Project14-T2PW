"""C-119 seams 1-2 -- **THE G9 BASE-FAILURE PROOF. A REGRESSION FIX, NOT NEW WORK.**

WHAT THIS FILE IS FOR, AND WHY IT IS SEPARATE. G9 requires that a claimed
correction of pre-existing observable behaviour carry a proof which **FAILS
BEHAVIOURALLY ON THE BASE SHA** ``6746a8d31e993d8dc6968ad46534b4e48b9c7bee`` and
passes at the tip, and that **symbol absence is not proof**. A test that dies with
``AttributeError: module 't2pw.batch.driver' has no attribute ...`` proves only that
a name is new; it says nothing about behaviour.

**SO EVERY NAME THIS MODULE TOUCHES EXISTS AT THE BASE SHA, WITH THE SAME
SIGNATURE.** ``_blocking_reports``, ``_collect_issue_codes``, ``_pwml_artifact_name``,
``run_one``, ``RunOutcome.to_dict``, ``gate_verdict``, ``PWML_REVIEW_REQUIRED_NAME``
and ``PWML_RELEASE_READY_NAME`` are all base symbols. The two strings C-119
introduces are written here as LITERALS rather than imported, for the same reason --
and ``tests/test_c119_superseded_intermediate_report.py`` pins each literal against
the real constant, so the duplication cannot silently drift.

This module therefore IMPORTS CLEANLY at the base and FAILS THERE ON VALUES:
``2 != 0``, ``'fail' != 'pass'``, and a ``pathway.review_required.pwml`` that the
base run never writes.

WHAT WAS BEING DESTROYED. ``batch/driver.py::_blocking_reports`` counted the errors
of a contract report the app itself stamps ``phase: audit_round`` and documents as
*"not a verdict about what shipped -- the remap below moves the payload again"*
(``streamlit_app.py:4032-4039``). A leg whose ``post_audit`` and ``post_remap``
boundaries were both clean, and whose ``final_pre_export`` Stage-3 gate passed on the
exact payload that shipped, was routed to ``_finalize_gate_failure`` and lost its
PWML entirely -- 4 of the 6 evaluable strict legs of the unseen pilot.

NO PIPELINE LEG RUNS HERE, no LLM draw is taken, no cache is touched and no network
is reached. Everything is a replay over archived payloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver  # noqa: E402
from t2pw.batch.driver import (  # noqa: E402
    PWML_RELEASE_READY_NAME,
    PWML_REVIEW_REQUIRED_NAME,
    STRICT,
    run_one,
)
from t2pw.pipeline.gate_reports import (  # noqa: E402
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
    PHASE_AUDIT_ROUND,
    gate_verdict,
)
from t2pw.pipeline.release_status import REVIEW_REQUIRED  # noqa: E402
from helpers_c119 import SNAPSHOT, artifacts, bundle  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: ``release_status.REASON_SUPERSEDED_INTERMEDIATE_REPORT``, written as a literal so
#: this module imports at the base SHA. Pinned against the constant in
#: ``tests/test_c119_superseded_intermediate_report.py``.
REASON_LITERAL = "superseded_intermediate_contract_report"

#: ``driver.WARN_SUPERSEDED_INTERMEDIATE_PREFIX``, likewise, and likewise pinned.
WARN_PREFIX_LITERAL = "serialized with a superseded intermediate contract report: "


def _contract_error_count(reassembled: dict) -> int:
    """The driver's own blocking arithmetic, through its own two base functions."""

    _codes, _lines, error_count = driver._collect_issue_codes(
        driver._blocking_reports(reassembled)
    )
    return error_count


def test_g9_the_superseded_audit_round_snapshot_no_longer_blocks_a_clean_leg() -> None:
    """PMC7232280/strict, 2026-09-06. **BASE: 2 blocking errors. TIP: 0.**

    The archived leg's live boundaries are all clean -- ``post_audit`` ok,
    ``post_remap`` ok, and the ``final_pre_export`` Stage-3 gate ok on the exact
    payload that shipped. The only thing that failed it was a snapshot the app had
    already superseded, whose two errors point at ``/entities/proteins/4`` and
    ``/5``.
    """

    reassembled = artifacts("PMC7232280_strict_2026-09-06")

    # The snapshot really is there, really is stamped audit_round, and really does
    # carry the two errors. Without this the proof below could pass vacuously.
    snapshot = reassembled[SNAPSHOT]
    assert snapshot["ok"] is False
    assert snapshot["phase"] == PHASE_AUDIT_ROUND
    assert len(snapshot["errors"]) == 2

    # BASE SHA: _blocking_reports returns the snapshot and this reads 2.
    assert _contract_error_count(reassembled) == 0
    assert SNAPSHOT not in driver._blocking_reports(reassembled)

    # The gate channel -- untouched by this card -- agrees the leg is clean, so
    # ``_drive``'s ``if blocking_gate or error_count`` no longer fires at all.
    assert gate_verdict(reassembled).failed is False


def test_g9_the_second_destroyed_leg_is_recovered_on_the_same_terms() -> None:
    """PMC8510960/strict, 2026-09-06. **BASE: 7 blocking errors. TIP: 0.**"""

    reassembled = artifacts("PMC8510960_strict_2026-09-06")
    assert reassembled[SNAPSHOT]["phase"] == PHASE_AUDIT_ROUND
    assert len(reassembled[SNAPSHOT]["errors"]) == 7
    assert _contract_error_count(reassembled) == 0
    assert gate_verdict(reassembled).failed is False


def test_g9_end_to_end_the_destroyed_leg_now_ships_a_review_required_pwml(
    tmp_path: Path,
) -> None:
    """The whole driver, over the archived artifact set. **BASE: ``fail``, no PWML.**

    This is the behavioural half of the obligation: not "an internal number moved"
    but "the product stopped destroying the pathway". It drives ``run_one`` against
    a fixture app that publishes PMC7232280's archived ``post_pipeline_artifacts``
    and its archived frozen release record, and asserts the leg PASSES with
    ``pathway.review_required.pwml`` written.

    AT THE BASE SHA the same fixture yields ``status="fail"``,
    ``failure_kind="contract"``, ``pwml_artifact=""`` and no PWML file at all --
    which is exactly the row ``runs_verify/2026-09-06_1425`` recorded for this leg.

    NOTHING IS PROMOTED. The status is ``review_required``, the filename is the
    review-required one, ``strict_acceptance_eligible`` stays ``False``, and the
    superseded finding is STATED on the row rather than dropped.
    """

    data = bundle("PMC7232280_strict_2026-09-06")
    artifacts_path = tmp_path / "artifacts.json"
    artifacts_path.write_text(
        json.dumps(
            {
                **data["contract_reports"],
                FINAL_GATE_REPORT_KEY: data["final_stage3_gate_report"],
                CANONICAL_PAYLOAD_KEY: data["canonical_export_payload"],
                "artifact_set_version": 1,
                "export_mode": "pathwhiz",
                "final_mapped": data["canonical_export_payload"],
                "final_mapped_db": data["canonical_export_payload"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps(data["quarantine_release"], ensure_ascii=False), encoding="utf-8"
    )

    body = f'''
import json
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {{"entities": {{}}, "processes": {{"reactions": []}}}}
    with open(r"{artifacts_path}", encoding="utf-8") as fh:
        st.session_state["final_payload"] = json.load(fh)["final_mapped"]
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        with open(r"{artifacts_path}", encoding="utf-8") as fh:
            st.session_state["post_pipeline_artifacts"] = json.load(fh)
    if st.session_state.get("post_pipeline_artifacts"):
        if st.button("Generate PWML", key="refinement_generate_pwml"):
            with open(r"{release_path}", encoding="utf-8") as fh:
                release = json.load(fh)
            st.session_state["pwml_export_result"] = {{
                "ok": True,
                "xml_bytes": b"<pathway><name>moco</name></pathway>",
                "quarantine_report": {{"release": release}},
                "counts": {{"reactions": 5}},
            }}
'''
    app = _write_app(tmp_path, "c119_pmc7232280", body)
    outcome = run_one(PAPER, STRICT, app_path=app, timeout=180.0, app_timeout=90.0)

    # BASE SHA: "fail" / "contract" / "" / the file is absent.
    assert outcome.status == "pass", outcome.message
    assert outcome.pwml_artifact == PWML_REVIEW_REQUIRED_NAME
    assert PWML_REVIEW_REQUIRED_NAME in outcome.artifacts
    assert PWML_RELEASE_READY_NAME not in outcome.artifacts

    row = outcome.to_dict()
    assert row["release_status"]["status"] == REVIEW_REQUIRED
    assert row["release_status"]["strict_acceptance_eligible"] is False
    # The superseded condition survived into the release record...
    assert any(
        str(reason).startswith(REASON_LITERAL)
        and f"{SNAPSHOT}@{PHASE_AUDIT_ROUND}=2" in str(reason)
        for reason in row["release_status"]["reasons"]
    ), row["release_status"]["reasons"]
    # ...and into the manifest row's own warnings channel, which is written
    # unconditionally and therefore survives even an unreadable release record.
    assert any(WARN_PREFIX_LITERAL in warning for warning in row["warnings"]), row["warnings"]
