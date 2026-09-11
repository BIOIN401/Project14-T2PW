"""C-126 -- **THE G9 BASE-FAILURE PROOF. A REGRESSION FIX, NOT NEW WORK.**

WHAT THIS FILE IS FOR, AND WHY IT IS SEPARATE. G9 requires that a claimed correction
of pre-existing observable behaviour carry a proof which **FAILS BEHAVIOURALLY ON THE
BASE SHA** ``c6159bc27989f5bc865d4b6946b0e8eb93fb5c12`` and passes at the tip, and
that **symbol absence is not proof**. A test that dies with ``AttributeError: module
't2pw.batch.driver' has no attribute _NON_AUTHORITATIVE_PHASES`` proves only that a
name is new; it says nothing about behaviour.

**SO EVERY NAME THIS MODULE TOUCHES EXISTS AT THE BASE SHA, WITH THE SAME
SIGNATURE.** ``_blocking_reports``, ``_collect_issue_codes``,
``_superseded_contract_reports``, ``_frozen_release_record``, ``_pwml_artifact_name``,
``run_one``, ``RunOutcome.to_dict``, ``gate_verdict``,
``PHASE_INITIAL_POST_NORMALIZATION``, ``PWML_REVIEW_REQUIRED_NAME`` and
``PWML_RELEASE_READY_NAME`` are all base symbols -- ``gate_reports.py`` has DEFINED
``PHASE_INITIAL_POST_NORMALIZATION`` since the phase stamp existed, which is the whole
point: the constant was there, documented as *"never a verdict about what shipped"*,
and ``batch/driver.py`` excluded the other one.

The two names C-126 introduces in the driver (``_NON_AUTHORITATIVE_PHASES`` and
``_EFFECT_FEED_AUDIT``) are **deliberately not referenced here**; where a value of
theirs is needed it is written as a LITERAL, and
``tests/test_c126_final_contract_authority.py`` pins each literal against the real
constant so the duplication cannot silently drift.

This module therefore IMPORTS CLEANLY at the base and FAILS THERE ON VALUES:
``2 != 0``, ``'fail' != 'pass'``, ``[] != [one descriptor]``, and a
``pathway.review_required.pwml`` that the base run never writes.

WHAT WAS BEING DESTROYED. ``batch/driver.py::_superseded_contract_reports`` excluded
exactly one of the two phases ``gate_reports.py`` DEFINES as non-authoritative::

    if _report_phase(report) != PHASE_AUDIT_ROUND:
        continue

A ``post_normalization_contract_report`` left stamped
``initial_post_normalization`` -- which is what the app's audit loop leaves it as
whenever ``settled_payload_changed`` is false, i.e. whenever the audit round correctly
declines to invent an identifier -- still failed the leg, even though its
``post_audit``, ``post_remap`` and ``final_pre_export`` boundaries were ALL clean and
the errors it carried addressed a PRE-REMAP protein list the shipped payload can no
longer host.

MEASURED REACH (REV-126). The defect was DIAGNOSED on four papers across three
independent cohorts, but a base-vs-tip census over all 188 archived legs on disk finds
it moves **nine** strict legs, three of which also carry a wider stale-finding class
than the card's four (``missing species/organism``, and ``/processes`` registry
validation naming an unknown entity). **Zero research legs move. None of the nine
becomes ``release_ready``** -- all nine were already ``review_required`` at base -- and
no PWML is claimed for any of them. See
``docs/pwml_recovery_sprint/F-147-RECURRENCE-DIAGNOSIS.md`` and
``_superseded_contract_reports``'s own docstring, which carries the full list.

NOTHING IS REPAIRED AND NOTHING IS PROMOTED. The stale report stays in
``contract_reports.json``, stays named in the review metadata, and the leg is bounded
to ``review_required``. The authoritative verdict on those same checks still lives in
``run_pwml_export``'s fail-closed pre-export Stage-3 revalidation, via
``_validate_stage8_export_payload``, and in ``pwml.ir``'s
``protein_missing_external_identity`` check; this seam only stops a PRE-Stage-3
snapshot from pre-empting it. Citations are symbolic because the ``streamlit_app.py``
line numbers this file used to carry were read from a working copy 22-23 lines ahead
of committed source.

NO PIPELINE LEG RUNS HERE, no LLM draw is taken, no cache is touched and no network is
reached. Everything is a replay over one archived artifact set.
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
    PHASE_INITIAL_POST_NORMALIZATION,
    gate_verdict,
)
from t2pw.pipeline.release_status import REVIEW_REQUIRED  # noqa: E402
from helpers_c119 import SNAPSHOT, artifacts, bundle  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: The F-147 recurrence leg this proof is built from. ``PMC13089919`` is the smallest
#: of the four: two errors, ``ChoC`` and ``Cho2``, both "missing a UniProt or DrugBank
#: identifier", both addressing a two-row protein list the shipped payload replaced
#: with a single ``Unknown`` row backing two functional complexes.
STEM = "PMC13089919_strict_2026-09-10_initial_post_normalization"

#: ``driver._EFFECT_FEED_AUDIT``, written as a literal so this module imports at the
#: base SHA. Pinned against the constant in
#: ``tests/test_c126_final_contract_authority.py``.
FEED_AUDIT_LITERAL = "feed_audit"

#: ``release_status.REASON_SUPERSEDED_INTERMEDIATE_REPORT``, likewise a base symbol but
#: spelled as a literal for symmetry with the C-119 proof module, and likewise pinned.
REASON_LITERAL = "superseded_intermediate_contract_report"

#: ``driver.WARN_SUPERSEDED_INTERMEDIATE_PREFIX``, likewise, and likewise pinned.
WARN_PREFIX_LITERAL = "serialized with a superseded intermediate contract report: "


def _contract_error_count(reassembled: dict) -> int:
    """The driver's own blocking arithmetic, through its own two base functions."""

    _codes, _lines, error_count = driver._collect_issue_codes(
        driver._blocking_reports(reassembled)
    )
    return error_count


def test_g9_the_stale_initial_post_normalization_report_no_longer_blocks() -> None:
    """PMC13089919/strict, 2026-09-10. **BASE: 2 blocking errors. TIP: 0.**

    THE PRECONDITIONS ARE ASSERTED FIRST so the proof below cannot pass vacuously,
    and they are the same three facts the card names:

    * ``gate_verdict(artifacts).failed`` is ``False`` -- the gate channel, which this
      card does not touch, says the leg is clean;
    * the contract ``error_count`` is ``2`` at the base;
    * **every one of those errors originates in
      ``post_normalization_contract_report``** -- it is the only failing report in the
      whole artifact set, so ``error_count`` alone is what refused the leg.
    """

    reassembled = artifacts(STEM)

    # --- preconditions, true at BOTH base and tip -------------------------------
    snapshot = reassembled[SNAPSHOT]
    assert snapshot["ok"] is False
    assert snapshot["phase"] == PHASE_INITIAL_POST_NORMALIZATION
    assert snapshot["effect_on_failure"] == FEED_AUDIT_LITERAL
    assert len(snapshot["errors"]) == 2
    # The gate channel agrees the leg is clean: the final pre-export Stage-3 gate
    # passed on the exact payload that shipped.
    assert gate_verdict(reassembled).failed is False
    # EVERY other contract report in the set is clean, so the 2 errors the base
    # counted can only have come from the snapshot.
    others = {
        key: len(report.get("errors") or [])
        for key, report in driver._blocking_reports(reassembled).items()
        if key != SNAPSHOT
    }
    assert others and all(count == 0 for count in others.values()), others
    # The three later boundaries that supersede it are all present and all clean.
    for key in ("post_audit_contract_report", "post_remap_contract_report"):
        assert reassembled[key]["ok"] is True
        assert not reassembled[key].get("errors")
    assert reassembled[FINAL_GATE_REPORT_KEY]["ok"] is True

    # --- the proof. BASE: 2 and True. TIP: 0 and False. ------------------------
    assert _contract_error_count(reassembled) == 0
    assert SNAPSHOT not in driver._blocking_reports(reassembled)

    # NOT DELETED -- still in the archive slice ``contract_reports.json`` is built
    # from, with its two errors intact. It stopped deciding; it did not vanish.
    archive = driver._collect_reports(reassembled)
    assert SNAPSHOT in archive
    assert len(archive[SNAPSHOT]["errors"]) == 2

    # ...and still NAMED in the review metadata, at its OWN phase. At the base this
    # list is ``[]``; C-119 would additionally have hard-coded ``audit_round`` here.
    assert driver._superseded_contract_reports(reassembled) == [
        {
            "report": SNAPSHOT,
            "phase": PHASE_INITIAL_POST_NORMALIZATION,
            "errors": 2,
            "superseded_by": [
                "post_audit_contract_report",
                "post_remap_contract_report",
                FINAL_GATE_REPORT_KEY,
            ],
        }
    ]


def test_g9_end_to_end_the_refused_leg_is_no_longer_refused_by_the_contract_channel(
    tmp_path: Path,
) -> None:
    """The whole driver, over the archived artifact set. **BASE: ``fail``, no PWML.**

    This is the behavioural half of the obligation: not "an internal number moved"
    but "the product stopped refusing the leg on stale payload history". It drives
    ``run_one`` against a fixture app that publishes PMC13089919's archived
    ``post_pipeline_artifacts`` and its archived frozen release record.

    AT THE BASE SHA the same fixture yields ``status="fail"``,
    ``failure_kind="contract"``, ``pwml_artifact=""`` and no PWML file at all.

    **THIS DOES NOT CLAIM THE REAL LEG PRODUCES A PWML.** The export result here is a
    STUB: the fixture app hands the driver an ``ok=True`` export, because what is
    under test is the BATCH DRIVER's contract channel, not ``run_pwml_export``. The
    real authoritative boundary (``run_pwml_export``'s fail-closed pre-export Stage-3
    revalidation) is not exercised by this test and is free to refuse the same
    payload. Eligibility is not export.

    NOTHING IS PROMOTED. The status is ``review_required`` -- the archived quarantine
    record already said so, for an independent reason
    (``semantic_evaluation_failed:actor_named_in_its_own_cited_span``) -- the filename
    is the review-required one, ``strict_acceptance_eligible`` stays ``False``, and
    the superseded finding is STATED on the row rather than dropped.
    """

    data = bundle(STEM)
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
                "xml_bytes": b"<pathway><name>choline</name></pathway>",
                "quarantine_report": {{"release": release}},
                "counts": {{"reactions": 5}},
            }}
'''
    app = _write_app(tmp_path, "c126_pmc13089919", body)
    outcome = run_one(PAPER, STRICT, app_path=app, timeout=180.0, app_timeout=90.0)

    # BASE SHA: "fail" / "contract" / "" / the file is absent.
    assert outcome.status == "pass", outcome.message
    assert outcome.pwml_artifact == PWML_REVIEW_REQUIRED_NAME
    assert PWML_REVIEW_REQUIRED_NAME in outcome.artifacts
    assert PWML_RELEASE_READY_NAME not in outcome.artifacts

    row = outcome.to_dict()
    assert row["release_status"]["status"] == REVIEW_REQUIRED
    assert row["release_status"]["strict_acceptance_eligible"] is False
    # The superseded condition survived into the release record, naming the report's
    # OWN phase -- C-119's hard-coded ``audit_round`` would have lied here.
    assert any(
        str(reason).startswith(REASON_LITERAL)
        and f"{SNAPSHOT}@{PHASE_INITIAL_POST_NORMALIZATION}=2" in str(reason)
        for reason in row["release_status"]["reasons"]
    ), row["release_status"]["reasons"]
    # The pre-existing, INDEPENDENT review reason is still there and still first.
    assert any(
        "semantic_evaluation_failed" in str(reason)
        for reason in row["release_status"]["reasons"]
    ), row["release_status"]["reasons"]
    # ...and into the manifest row's own warnings channel, which is written
    # unconditionally and therefore survives even an unreadable release record.
    assert any(WARN_PREFIX_LITERAL in warning for warning in row["warnings"]), row["warnings"]
    assert row["counts"]["superseded_contract_errors"] == 2
