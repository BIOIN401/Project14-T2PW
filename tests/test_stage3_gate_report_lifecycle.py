"""A strict PASS means the strict gate suite passed on the payload that shipped.

WHAT WAS BROKEN
---------------
The pipeline recorded gate *failure* and never recorded gate *success*:

* ``streamlit_app.py`` set ``gate_fail_report`` when the first post-normalization
  gate failed, wrote ``gate_fail_report.json``, and then handed those same errors
  to the audit as its **repair list**;
* each audit round re-ran the suite into a local ``fresh_gate_report`` and threw
  the contract it produced away;
* the pre-export revalidation wrote ``final_stage3_gate_report`` only on its two
  failing branches -- a pass wrote nothing at all;
* and ``batch/driver.py`` reduced the verdict to ``bool(... or gate_fail ...)``:
  a truthy *stale* dict that no artifact was able to countermand.

So a leg whose payload the audit had genuinely repaired was still filed FAIL. In
``runs/2026-08-02_2130`` every strict leg carrying a ``gate_fail_report.json``
was filed FAIL -- 6 of 6 -- and PMC12782028's ``final_mapped.json`` passes
``run_strict_post_normalization_gates(payload,
enforce_all_proteins_connected=True)`` outright: pure stale bookkeeping.

WHAT THIS FILE PINS
-------------------
1. A failure is superseded ONLY by a fresh run of the same suite, on the
   canonical payload, that passes. Never by "the audit said it fixed it", never
   by a round count, never because a later stage's report is clean.
2. Everything else fails closed: no final report, an unsupported version, the
   right report about the wrong phase, ``ok: true`` next to blocking errors, a
   report whose payload hash is not the payload in hand.
3. Historical reports are retained in full and take no part in the arithmetic.
4. Legacy artifact sets -- recorded before the lifecycle existed, and unable to
   produce a final report -- keep the old behaviour exactly.
5. Research mode may continue past a failing gate. It may not rewrite the fact.

The driver-level tests drive the REAL batch driver against a fixture app that
publishes an artifacts dict, the same technique as
``tests/test_batch_research_gate_fail_open.py``: the artifacts dict is the
driver's half of the app contract, so a fixture app that publishes one exercises
the leg under test without a live pipeline.
"""

from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.driver import RESEARCH, STRICT, run_one  # noqa: E402
from t2pw.pipeline.gate_reports import (  # noqa: E402
    ARTIFACT_SET_VERSION,
    ARTIFACT_SET_VERSION_KEY,
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
    INITIAL_GATE_REPORT_KEY,
    PAYLOAD_SHA256_KEY,
    PHASE_AUDIT_ROUND,
    PHASE_FINAL_PRE_EXPORT,
    PHASE_INITIAL_POST_NORMALIZATION,
    PHASE_KEY,
    REPORT_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION_KEY,
    SOURCE_FAIL_CLOSED,
    SOURCE_FINAL_REPORT,
    SOURCE_LEGACY,
    blocking_findings,
    dedupe_findings,
    finding_key,
    gate_verdict,
    payload_sha256,
    stamp_report,
)


# ---------------------------------------------------------------------------
# Real streamlit, whatever the rest of the suite did to sys.modules.
# ---------------------------------------------------------------------------
def _is_real_streamlit(module: object) -> bool:
    return isinstance(module, types.ModuleType) and hasattr(module, "__path__")


@pytest.fixture
def real_streamlit():
    """The genuine ``streamlit`` package for the duration of one test.

    Several modules in this suite install a ``MagicMock`` as
    ``sys.modules["streamlit"]`` at import time -- they must, since importing
    ``t2pw.app.streamlit_app`` executes a Streamlit script -- and pytest imports
    every module during collection. Without this fixture the ``AppTest`` below
    would be a MagicMock returning MagicMocks and every assertion here would pass
    for the wrong reason.
    """

    def _streamlit_keys():
        return [n for n in list(sys.modules) if n == "streamlit" or n.startswith("streamlit.")]

    saved = {name: sys.modules[name] for name in _streamlit_keys()}
    stubbed = not _is_real_streamlit(sys.modules.get("streamlit"))
    if stubbed:
        for name in saved:
            del sys.modules[name]
    import streamlit  # noqa: F401,PLC0415
    import streamlit.testing.v1  # noqa: F401,PLC0415

    assert _is_real_streamlit(sys.modules["streamlit"]), "the real streamlit did not load"
    try:
        yield
    finally:
        if stubbed:
            for name in _streamlit_keys():
                del sys.modules[name]
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Fixtures: payloads, reports, and the fixture app.
# ---------------------------------------------------------------------------
_STAGE1 = {
    "entities": {"compounds": [{"name": "glycine"}], "proteins": [{"name": "ALAS2"}]},
    "processes": {"reactions": [{"name": "ALA synthesis"}]},
}

#: Stands in for the canonical payload -- what the final gate validated, what is
#: serialized to ``final_mapped.json``, and what the exporter consumed.
_CANONICAL = {
    "entities": {
        "compounds": [{"name": "glycine"}, {"name": "5-aminolevulinate"}],
        "proteins": [{"name": "ALAS2"}],
    },
    "processes": {
        "reactions": [
            {
                "name": "ALA synthesis",
                "inputs": ["glycine"],
                "outputs": ["5-aminolevulinate"],
                "enzymes": ["ALAS2"],
            }
        ]
    },
}

#: One finding. It is deliberately the ALAS2 registry error from PMC12856317,
#: because that leg was reported as "2 blocking issue(s)" for exactly this one row
#: arriving once through the gate channel and once through the contract channel.
_ALAS2_GATE_ERROR = {
    "path": "/processes",
    "reason": (
        "Registry validation failed: /processes/interactions/0/entity_2 "
        "unknown entity: ALAS2"
    ),
}
_ALAS2_CONTRACT_ERROR = {
    "code": "",
    "pointer": "/processes",
    "message": "Registry validation failed: unknown entity: ALAS2",
}

#: The first gate run. Its errors are the audit's INPUT, which is precisely why
#: it may never be read as an outcome.
_INITIAL_FAILURE = stamp_report(
    {
        "status": "failed",
        "stage": "post_normalization_hard_gates",
        "error": "Hard-gate validation failed after normalization.",
        "errors": [_ALAS2_GATE_ERROR],
    },
    phase=PHASE_INITIAL_POST_NORMALIZATION,
)


def _final_report(*, ok: bool, errors=None, payload=_CANONICAL, **overrides) -> dict:
    report = stamp_report(
        {
            "stage": "final_pre_export_stage3_gates",
            "ok": ok,
            "errors": list(errors or []),
        },
        phase=PHASE_FINAL_PRE_EXPORT,
        payload=payload,
    )
    report.update(overrides)
    return report


def _artifacts(mode_value: str, **overrides) -> dict:
    """A CURRENT-version artifact set with every contract report green.

    Green contracts are the point: they isolate the gate leg, so a run that fails
    here can only have failed on the gate arithmetic under test.
    """

    base = {
        ARTIFACT_SET_VERSION_KEY: ARTIFACT_SET_VERSION,
        "gate_failed": False,
        "export_mode": mode_value,
        "research_review_flags": [],
        "research_skipped_format_rules": [],
        "research_normalization_actions": [],
        "final_mapped_db": _CANONICAL,
        "final_mapped": _CANONICAL,
        CANONICAL_PAYLOAD_KEY: _CANONICAL,
        "canonical_payload_sha256": payload_sha256(_CANONICAL),
        "quarantine_ok": True,
        "mapping_report": {"summary": {"mapped": 3}},
        "post_extraction_contract_report": {"ok": True, "errors": [], "warnings": []},
        "post_normalization_contract_report": {"ok": True, "errors": [], "warnings": []},
        # The default is the repaired case: the initial gate failed, the audit
        # fixed it, the final gate passed.
        "normalization_gate_failed": True,
        "gate_fail_report": _INITIAL_FAILURE,
        INITIAL_GATE_REPORT_KEY: _INITIAL_FAILURE,
        FINAL_GATE_REPORT_KEY: _final_report(ok=True),
    }
    base.update(overrides)
    return base


_PREAMBLE = '''\
import streamlit as st

st.radio("Export mode", ["Strict PWML", "Research (relaxed)"], key="export_mode_radio")
st.radio("Input mode", ["Paste text", "Upload PDF", "Text + PDF"], key="input_mode_radio")
with st.form("pwml_pipeline"):
    st.text_area("Paste pathway description:", key="pathway_text_0")
    st.text_area(
        "Optional extraction focus / task context",
        height=100,
        help="Use this to tell the model what pathway or scope you want extracted.",
    )
    submitted = st.form_submit_button("Run pipeline")
'''


def _write_app(tmp_path: Path, name: str, artifacts: dict) -> Path:
    body = f'''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {_STAGE1!r}
    st.session_state["final_payload"] = {_CANONICAL!r}
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["post_pipeline_artifacts"] = {artifacts!r}
    if st.button("Generate PWML", key="refinement_generate_pwml"):
        st.session_state["pwml_export_result"] = {{
            "ok": True,
            "xml_bytes": b"<Pathway/>",
            "pwml_ir": {{"pathway": "heme biosynthesis"}},
        }}
'''
    target = tmp_path / f"{name}.py"
    target.write_text(_PREAMBLE + body, encoding="utf-8")
    return target


class _Paper:
    paper_id = "PMC12856317"
    full_text = "Heme biosynthesis begins with the condensation of glycine and succinyl-CoA."
    topic = ""
    organism = ""


def _run(app: Path, mode: str):
    return run_one(_Paper(), mode, app_path=app, timeout=120.0, app_timeout=45.0)


def _gate_stage(outcome) -> bool:
    """Did this outcome fail at the post-pipeline gate leg?"""

    return outcome.status != "pass" and "post-pipeline validation failed" in (outcome.message or "")


# ===========================================================================
# 1. gate_verdict -- the one derivation, unit-tested at every branch.
# ===========================================================================


def test_a_repaired_payload_supersedes_the_initial_failure() -> None:
    """The headline defect. Initial fail + final pass == PASS."""

    verdict = gate_verdict(_artifacts("pathwhiz"))

    assert verdict.failed is False
    assert verdict.source == SOURCE_FINAL_REPORT
    assert verdict.errors == []


def test_the_initial_failure_is_retained_even_when_it_is_superseded() -> None:
    """Superseded is not deleted: the audit trail is the point of keeping it."""

    artifacts = _artifacts("pathwhiz")

    assert artifacts["gate_fail_report"]["errors"] == [_ALAS2_GATE_ERROR]
    assert artifacts[INITIAL_GATE_REPORT_KEY] is artifacts["gate_fail_report"]
    assert artifacts[INITIAL_GATE_REPORT_KEY][PHASE_KEY] == PHASE_INITIAL_POST_NORMALIZATION
    assert gate_verdict(artifacts).failed is False


def test_a_final_payload_that_still_fails_is_still_a_failure() -> None:
    """The repair is not assumed; it is measured."""

    verdict = gate_verdict(
        _artifacts(
            "pathwhiz",
            **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])},
        )
    )

    assert verdict.failed is True
    assert verdict.source == SOURCE_FINAL_REPORT
    assert verdict.errors == [_ALAS2_GATE_ERROR]


def test_an_initial_pass_and_a_final_pass_is_a_pass() -> None:
    verdict = gate_verdict(
        _artifacts(
            "pathwhiz",
            normalization_gate_failed=False,
            gate_fail_report={},
            **{INITIAL_GATE_REPORT_KEY: {}},
        )
    )

    assert verdict.failed is False


def test_an_audit_round_pass_cannot_speak_for_a_payload_the_remap_then_broke() -> None:
    """Step 4 -> 5 -> 6 of the pipeline: audit settles, remap moves it again.

    A clean audit-round report is evidence of progress and nothing more. Only the
    final report describes the object that shipped, so if the remap broke it the
    run fails no matter how green the round was.
    """

    audit_round_pass = stamp_report({"ok": True, "errors": []}, phase=PHASE_AUDIT_ROUND)
    verdict = gate_verdict(
        _artifacts(
            "pathwhiz",
            post_normalization_contract_report={"ok": True, "errors": [], **audit_round_pass},
            **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])},
        )
    )

    assert verdict.failed is True


# ── fail-closed shapes ─────────────────────────────────────────────────────


def test_a_pass_bound_to_a_different_payload_fails_closed() -> None:
    """``payload_sha256`` is what makes "on the payload that shipped" checkable."""

    moved = deepcopy(_CANONICAL)
    moved["entities"]["proteins"].append({"name": "ALAD"})
    verdict = gate_verdict(_artifacts("pathwhiz", **{CANONICAL_PAYLOAD_KEY: moved}))

    assert verdict.failed is True
    assert verdict.source == SOURCE_FAIL_CLOSED
    assert "payload_mismatch" in verdict.reason


def test_a_current_version_run_with_no_final_report_fails_closed() -> None:
    """"Unmeasured" reads as "failed". A producer bug is not a pass."""

    verdict = gate_verdict(_artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: {}}))

    assert verdict.failed is True
    assert verdict.source == SOURCE_FAIL_CLOSED
    assert verdict.reason == "final_gate_report_missing"


def test_an_unsupported_artifact_set_version_fails_closed() -> None:
    verdict = gate_verdict(_artifacts("pathwhiz", **{ARTIFACT_SET_VERSION_KEY: 99}))

    assert verdict.failed is True
    assert verdict.source == SOURCE_FAIL_CLOSED
    assert "unsupported_artifact_set_version" in verdict.reason


def test_an_unsupported_report_schema_version_fails_closed() -> None:
    report = _final_report(ok=True)
    report[REPORT_SCHEMA_VERSION_KEY] = REPORT_SCHEMA_VERSION + 1
    verdict = gate_verdict(_artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: report}))

    assert verdict.failed is True
    assert "unsupported_report_schema_version" in verdict.reason


def test_a_correct_report_about_the_wrong_phase_fails_closed() -> None:
    """An initial or audit-round report parked under the final key proves nothing.

    Without the phase check this is the whole defect running backwards: a green
    report about an intermediate payload would be accepted as a statement about
    the one that shipped.
    """

    misfiled = stamp_report({"ok": True, "errors": []}, phase=PHASE_AUDIT_ROUND, payload=_CANONICAL)
    verdict = gate_verdict(_artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: misfiled}))

    assert verdict.failed is True
    assert verdict.reason.startswith("final_gate_report_wrong_phase")


def test_ok_true_alongside_blocking_errors_is_malformed_and_fails_closed() -> None:
    verdict = gate_verdict(
        _artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: _final_report(ok=True, errors=[_ALAS2_GATE_ERROR])})
    )

    assert verdict.failed is True
    assert verdict.reason == "final_gate_report_malformed"


def test_a_final_report_with_no_payload_binding_fails_closed() -> None:
    report = _final_report(ok=True)
    report.pop(PAYLOAD_SHA256_KEY)
    verdict = gate_verdict(_artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: report}))

    assert verdict.failed is True
    assert verdict.reason == "final_gate_report_not_bound_to_a_payload"


def test_a_final_report_with_no_canonical_payload_to_compare_fails_closed() -> None:
    verdict = gate_verdict(_artifacts("pathwhiz", **{CANONICAL_PAYLOAD_KEY: {}}))

    assert verdict.failed is True
    assert verdict.reason == "canonical_payload_missing"


# ── legacy ─────────────────────────────────────────────────────────────────


def test_a_legacy_artifact_set_keeps_the_old_arithmetic() -> None:
    """Archived runs cannot produce a final report, so refusing them would
    rewrite the recorded history of every run in ``runs/``."""

    legacy = _artifacts("pathwhiz")
    legacy.pop(ARTIFACT_SET_VERSION_KEY)
    legacy.pop(FINAL_GATE_REPORT_KEY)

    verdict = gate_verdict(legacy)

    assert verdict.failed is True
    assert verdict.source == SOURCE_LEGACY
    assert verdict.errors == [_ALAS2_GATE_ERROR]


def test_a_legacy_artifact_set_with_no_gate_reports_passes() -> None:
    legacy = _artifacts("pathwhiz", normalization_gate_failed=False, gate_fail_report={})
    legacy.pop(ARTIFACT_SET_VERSION_KEY)
    legacy.pop(FINAL_GATE_REPORT_KEY)
    legacy.pop(INITIAL_GATE_REPORT_KEY)

    assert gate_verdict(legacy).failed is False


def test_the_legacy_branch_is_unreachable_for_a_stamped_run() -> None:
    """The version gate is what keeps the old bug from surviving in a corner.

    A current-version run missing its final report must NOT quietly fall back to
    "read the stale dict"; that is the behaviour being removed.
    """

    current = _artifacts("pathwhiz")
    current.pop(FINAL_GATE_REPORT_KEY)

    verdict = gate_verdict(current)

    assert verdict.source == SOURCE_FAIL_CLOSED
    assert verdict.source != SOURCE_LEGACY


# ── deduplication ──────────────────────────────────────────────────────────


def test_one_finding_reported_by_two_channels_is_counted_once() -> None:
    """PMC12856317's "2 blocking issue(s)" was one ALAS2 row, twice."""

    verdict = gate_verdict(
        _artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])})
    )

    merged = blocking_findings(verdict, [_ALAS2_CONTRACT_ERROR])

    assert len(merged) == 1


def test_dedup_keys_on_pointer_and_code_never_on_message_text() -> None:
    """Gate messages embed entity names, so message text is not an identity.

    This is the same trap that makes ``failures_by_code.txt`` fragment one defect
    into dozens of "distinct" codes -- a deduplicator with that flaw removes
    nothing at all.
    """

    same_row_different_wording = {
        "path": "/processes",
        "reason": "unknown entity: ALAS2 (registry)",
    }

    assert finding_key(_ALAS2_GATE_ERROR) == finding_key(same_row_different_wording)
    assert len(dedupe_findings([_ALAS2_GATE_ERROR], [same_row_different_wording])) == 1


def test_two_genuinely_different_rows_are_not_collapsed() -> None:
    other = {"path": "/entities/proteins/3/name", "reason": "unknown entity: SFXN1"}

    assert len(dedupe_findings([_ALAS2_GATE_ERROR, other])) == 2


def test_dedup_changes_the_count_and_not_the_verdict() -> None:
    """Both channels still block INDEPENDENTLY. Only the number moves."""

    verdict = gate_verdict(
        _artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])})
    )

    assert verdict.failed is True
    assert len(blocking_findings(verdict, [_ALAS2_CONTRACT_ERROR])) == 1


# ===========================================================================
# 2. The batch driver, end to end against a fixture app.
# ===========================================================================


def test_the_driver_files_a_repaired_strict_leg_as_pass(tmp_path: Path, real_streamlit) -> None:
    """The acceptance case: initial failure on disk, final gates green, PASS.

    Six of six strict legs in ``runs/2026-08-02_2130`` that carried a
    ``gate_fail_report.json`` were filed FAIL, and at least two of those payloads
    passed the strict suite. This is the assertion that had no way to be true.
    """

    app = _write_app(tmp_path, "repaired_strict", _artifacts("pathwhiz"))

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert not _gate_stage(outcome)
    assert outcome.counts.get("gate_errors", 0) == 0
    assert outcome.counts.get("blocking_issues", 0) == 0


def test_the_driver_still_fails_a_strict_leg_whose_final_payload_fails(
    tmp_path: Path, real_streamlit
) -> None:
    """Failing closed is correct: run 2026-07-28_0919 shipped PhoP as NAD+."""

    app = _write_app(
        tmp_path,
        "still_failing_strict",
        _artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])}),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert _gate_stage(outcome), outcome.message


def test_the_driver_fails_closed_when_a_current_run_has_no_final_report(
    tmp_path: Path, real_streamlit
) -> None:
    app = _write_app(
        tmp_path, "missing_final", _artifacts("pathwhiz", **{FINAL_GATE_REPORT_KEY: {}})
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert "final_gate_report_missing" in (outcome.message or "")


def test_the_driver_fails_closed_on_a_payload_hash_mismatch(
    tmp_path: Path, real_streamlit
) -> None:
    moved = deepcopy(_CANONICAL)
    moved["entities"]["proteins"].append({"name": "ALAD"})
    app = _write_app(
        tmp_path, "hash_mismatch", _artifacts("pathwhiz", **{CANONICAL_PAYLOAD_KEY: moved})
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert "payload_mismatch" in (outcome.message or "")


def test_the_driver_reports_one_blocking_issue_for_one_row_seen_twice(
    tmp_path: Path, real_streamlit
) -> None:
    """The literal PMC12856317 report line: "2 blocking issue(s)" for one row."""

    app = _write_app(
        tmp_path,
        "double_counted",
        _artifacts(
            "pathwhiz",
            **{
                FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR]),
                "post_normalization_contract_report": {
                    "ok": False,
                    "errors": [_ALAS2_CONTRACT_ERROR],
                    "warnings": [],
                },
            },
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.counts["blocking_issues"] == 1
    assert "1 blocking issue(s)" in (outcome.message or "")


def test_research_mode_continues_past_a_failing_gate_without_rewriting_it(
    tmp_path: Path, real_streamlit
) -> None:
    """Continuation is a POLICY decision; the gate result is a FACT.

    Research mode is allowed to deliver a citation report over a candidate the
    PathWhiz importer would refuse. It is not allowed to record that the strict
    gates passed when they did not.
    """

    artifacts = _artifacts(
        "research", **{FINAL_GATE_REPORT_KEY: _final_report(ok=False, errors=[_ALAS2_GATE_ERROR])}
    )
    app = _write_app(tmp_path, "research_failing_gate", artifacts)

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    # The fact survives, in the counts and in the archived report.
    assert outcome.counts["gate_errors"] == 1
    assert outcome.counts["gate_errors_not_blocking"] == 1
    archived = json.loads(outcome.artifacts["final_stage3_gate_report.json"])
    assert archived["ok"] is False
    assert archived[PHASE_KEY] == PHASE_FINAL_PRE_EXPORT


def test_the_initial_failure_reaches_the_run_directory_under_a_historical_name(
    tmp_path: Path, real_streamlit
) -> None:
    """Retained in full, under a key whose name says it is history."""

    app = _write_app(tmp_path, "history_artifact", _artifacts("pathwhiz"))

    outcome = _run(app, STRICT)

    assert json.loads(outcome.artifacts["gate_fail_report.json"])["errors"] == [_ALAS2_GATE_ERROR]
    historical = json.loads(outcome.artifacts["initial_stage3_gate_report.json"])
    assert historical[PHASE_KEY] == PHASE_INITIAL_POST_NORMALIZATION
    assert historical["errors"] == [_ALAS2_GATE_ERROR]


def test_final_mapped_json_is_the_canonical_payload(tmp_path: Path, real_streamlit) -> None:
    """The file an offline audit re-runs the gates on must BE the gated object.

    It used to be ``final_mapped_db`` -- post-remap and pre-enrichment -- so
    re-running the suite on it answered a question about a payload that was
    neither validated nor exported.
    """

    enriched_only = deepcopy(_CANONICAL)
    enriched_only["entities"]["proteins"][0]["mapping_meta"] = {"source": "pre_enrichment"}
    app = _write_app(
        tmp_path, "canonical_serialized", _artifacts("pathwhiz", final_mapped_db=enriched_only)
    )

    outcome = _run(app, STRICT)

    written = json.loads(outcome.artifacts["final_mapped.json"])
    assert written == _CANONICAL
    assert payload_sha256(written) == _artifacts("pathwhiz")["canonical_payload_sha256"]
