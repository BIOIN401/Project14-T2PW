"""D-004 / D-038: the batch artifact set names its PWML by the run's RELEASE STATE.

Two halves live here on purpose, and **each is labelled for what it is**, because
merge gate G9 turns on exactly that difference.

**BEHAVIOURAL CORRECTION -- fails on the base SHA.** ``driver.py:1380`` writes
``out["pathway.pwml"] = xml`` unconditionally, so a strict leg the quarantine
boundary FROZE as ``review_required`` ships under the filename PRODUCT_CONTRACT
SS13 reserves for ``release_ready``; ``RunOutcome.to_dict`` emits no
``release_status``, so the manifest cannot correct the impression either; and
``acceptance.py`` counts that leg as strict success from the filename alone, which
is TRAP-1 live rather than hypothetical (``LEDGER.md``, C-041a). The tests below
named ``test_correction_*`` assert the corrected behaviour. **Measured on a base
worktree at ``8920371``: 6 of the 8 fail there.** The two that PASS at the base
are ``test_correction_a_release_ready_leg_keeps_the_reserved_name`` and
``test_correction_a_release_ready_leg_still_counts``, and they pass **by design**:
they are the PRESERVATION arms of the correction, asserting that the reserved name
and the strict-success count are narrowed rather than retired, so a tree where
they failed would be a tree this card had broken. They are grouped with the
correction because they are the same behaviour seen from its other side, not
because a base failure is claimed for them.

**NEW ACCEPTANCE -- new capability, no base failure is claimed.** "The pipeline
passed and its classification is unavailable" is a state enumerated nowhere before
**D-038 SS3**: at the base the record never reached the driver at all, so there is
no prior behaviour to preserve or correct. Its four required properties -- the
bytes are kept under ``pathway.review_required.pwml``, no ``release_status`` key is
emitted, a warning names the missing record, and the leg cannot reach ``strict_ok``
-- are asserted by the ``test_new_*`` tests. Some of those assertions also happen
to fail at the base; that is a consequence of the shared naming rule, **not** a
correction claim, and it is stated here rather than left for a reviewer to infer.

**Nothing here weakens a gate and nothing re-derives biology.** The release record
is the one the boundary already froze, read out of
``pwml_result["quarantine_report"]["release"]`` and copied verbatim;
``classify_release_status`` is never called on the PASS path (merge rule 8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver, runner  # noqa: E402
from t2pw.batch.driver import MODE_STRICT, STRICT  # noqa: E402
from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.metrics import DENOM_STRICT  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _artifacts,
    _post_pipeline_body,
    _run,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: The two filenames PRODUCT_CONTRACT SS13 defines. Spelled as literals, never
#: imported: a proof that reads the name off the module under test would pass on
#: any tree that merely DEFINES the constant, and G9 is explicit that symbol
#: absence is not proof.
RELEASE_READY_NAME = "pathway.pwml"
REVIEW_REQUIRED_NAME = "pathway.review_required.pwml"


def _release(status: str, **overrides: Any) -> Dict[str, Any]:
    """A frozen release record shaped exactly as the boundary emits one.

    The key set is the MEASURED one -- ``ReleaseStatus.to_dict()`` plus the three
    fields ``strict_quarantine.quarantine_and_close`` adds at the seam -- captured
    from a live passing strict export in
    ``docs/pwml_recovery_sprint/evidence/c053_s0_release_record.json``.
    """

    record: Dict[str, Any] = {
        "status": status,
        "pipeline_executed": True,
        "strict_gates_passed": True,
        "semantic_evaluation": "not_evaluated",
        "semantic_not_evaluated_reason": "no semantic evaluation is wired in yet",
        "strict_acceptance_eligible": status == "release_ready",
        "completeness": None,
        "missing_anchors": [],
        "retrieval_attempts": 0,
        "retrieval_attempts_source": "surviving_rows_carrying_rag_provenance",
        "expansion_blocked_reason": "",
        "coverage_evaluated": True,
        "reasons": [],
        "applied": True,
        "review_reasons": [],
    }
    record.update(overrides)
    return record


def _pwml(release: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A successful PWML export result, optionally carrying a frozen record."""

    result: Dict[str, Any] = {
        "ok": True,
        "_xml": "<pathway><name>menaquinone</name></pathway>",
        "pwml_ir": {"pathway": {"name": "menaquinone"}},
        "counts": {"reactions": 1},
    }
    if release is not None:
        result["quarantine_report"] = {"release": release}
    return result


def _leg(tmp_path: Path, name: str, release: Optional[Dict[str, Any]]) -> Any:
    app = _write_app(
        tmp_path, name, _post_pipeline_body(_artifacts("pathwhiz"), pwml=_pwml(release))
    )
    return _run(app, STRICT)


# ---------------------------------------------------------------------------
# BEHAVIOURAL CORRECTION. Every test below fails on the base SHA.
# ---------------------------------------------------------------------------
def test_correction_a_review_required_leg_is_not_named_with_the_reserved_name(
    tmp_path: Path,
) -> None:
    """The base writes ``pathway.pwml`` for this leg; SS13 reserves that name.

    Merge rule 7: the bytes are kept, under the name that says what they are.
    """

    outcome = _leg(tmp_path, "review", _release("review_required"))

    assert outcome.status == "pass", outcome.detail
    assert RELEASE_READY_NAME not in outcome.artifacts
    assert outcome.artifacts[REVIEW_REQUIRED_NAME].startswith(b"<pathway>")


def test_correction_a_release_ready_leg_keeps_the_reserved_name(tmp_path: Path) -> None:
    """The reserved name is not retired -- it is narrowed to the one state that
    earns it. Preservation, proven on the same seam as the correction."""

    outcome = _leg(tmp_path, "ready", _release("release_ready"))

    assert outcome.status == "pass", outcome.detail
    assert outcome.artifacts[RELEASE_READY_NAME].startswith(b"<pathway>")
    assert REVIEW_REQUIRED_NAME not in outcome.artifacts


def test_correction_the_row_carries_the_frozen_record_and_the_filename(
    tmp_path: Path,
) -> None:
    """D-038 SS1's two keys, and only those two.

    The record is the boundary's, VERBATIM: a re-derived classification after the
    freeze is a merge rule 8 reject, so this compares against the exact dict the
    fixture froze rather than against a re-computation of it.
    """

    frozen = _release("review_required", completeness=0.25, reasons=["minimum_core:x"])
    outcome = _leg(tmp_path, "row", frozen)
    row = outcome.to_dict()

    assert row["release_status"] == frozen
    assert row["pwml_artifact"] == REVIEW_REQUIRED_NAME
    # D-038 struck these: two of them name nothing that exists, and the other two
    # already live INSIDE ``release_status``, where duplicating a benchmark-gating
    # flag would create a second source of truth for it.
    for struck in (
        "pipeline_status",
        "strict_acceptance_passed",
        "strict_acceptance_eligible",
        "completeness",
    ):
        assert struck not in row, struck


def test_correction_a_diagnostic_only_classification_writes_no_final_pwml(
    tmp_path: Path,
) -> None:
    """SS13: "No final PWML for ``diagnostic_only``." Loudly, not silently."""

    outcome = _leg(tmp_path, "diag", _release("diagnostic_only"))

    assert outcome.status == "pass", outcome.detail
    assert RELEASE_READY_NAME not in outcome.artifacts
    assert REVIEW_REQUIRED_NAME not in outcome.artifacts
    assert "pwml_artifact" not in outcome.to_dict()
    assert any("diagnostic_only" in w for w in outcome.warnings), outcome.warnings


def test_correction_the_writer_gate_covers_the_review_required_name(tmp_path: Path) -> None:
    """``runner.required_artifacts`` exists so a "pass" with no file on disk is
    caught. A renamed deliverable that it does not know about would walk straight
    past it, which is the failure mode the function was written for."""

    assert runner.required_artifacts(MODE_STRICT, {REVIEW_REQUIRED_NAME: b"x"}) == [
        REVIEW_REQUIRED_NAME
    ]
    assert runner.required_artifacts(MODE_STRICT, {RELEASE_READY_NAME: b"x"}) == [
        RELEASE_READY_NAME
    ]
    # Nothing produced, nothing required: an unattempted export is judged upstream.
    assert runner.required_artifacts(MODE_STRICT, {"notes.txt": "x"}) == []


# ---------------------------------------------------------------------------
# The strict denominator. Correction: TRAP-1 is live at the base.
# ---------------------------------------------------------------------------
def _score(tmp_path: Path, rows: List[Dict[str, Any]]) -> Any:
    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    run_dir.joinpath("manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return score_run(run_dir, load_gold_set())


def _manifest_row(paper_id: str, artifact: str, release: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "paper_id": paper_id,
        "slug": paper_id,
        "mode": "strict",
        "status": "pass",
        "stage": "pwml_export",
        "failure_kind": "",
        "issue_codes": [],
        "files": [{"name": f"papers/{paper_id}/strict/{artifact}", "bytes": 4210}],
        "pwml_artifact": artifact,
    }
    if release is not None:
        row["release_status"] = release
    return row


def test_correction_a_review_required_leg_never_counts_as_strict_success(
    tmp_path: Path,
) -> None:
    """TRAP-1, enforced where the number is actually produced.

    At the base this leg counts: ``deliverable`` is read off the filename and the
    filename is ``pathway.pwml`` for every export. ``strict_acceptance_eligible``
    is False for every ``review_required`` run (``release_status.py:317``), and the
    denominator must obey it -- PRODUCT_CONTRACT SS13, "Never strict success."
    """

    paper_id = load_gold_set().strict_expected_ids[0]
    report = _score(
        tmp_path,
        [_manifest_row(paper_id, RELEASE_READY_NAME, _release("review_required"))],
    )
    strict = report.denominators[DENOM_STRICT]

    assert strict.denominator == 1, "it is still IN the denominator; it just did not pass"
    assert paper_id not in strict.numerator_names
    assert strict.numerator == 0


def test_correction_a_release_ready_leg_still_counts(tmp_path: Path) -> None:
    """The gate is affirmative, not a blanket refusal: a run that measured
    ``release_ready`` counts exactly as it always did."""

    paper_id = load_gold_set().strict_expected_ids[0]
    report = _score(
        tmp_path, [_manifest_row(paper_id, RELEASE_READY_NAME, _release("release_ready"))]
    )
    strict = report.denominators[DENOM_STRICT]

    assert paper_id in strict.numerator_names
    assert strict.numerator == 1


def test_correction_the_review_required_file_is_still_a_deliverable(tmp_path: Path) -> None:
    """Renaming the file must not make the leg look like it produced nothing.

    ``deliverable`` answers "did an importable file land?" -- a different question
    from "may it count as strict success?", and collapsing the two is what this
    card exists to stop.
    """

    paper_id = load_gold_set().strict_expected_ids[0]
    report = _score(
        tmp_path,
        [_manifest_row(paper_id, REVIEW_REQUIRED_NAME, _release("review_required"))],
    )
    paper = next(p for p in report.papers if p.paper_id == paper_id)
    leg = paper.legs[MODE_STRICT]

    assert leg.deliverable is True
    assert leg.release_status == _release("review_required")
    assert leg.pwml_artifact == REVIEW_REQUIRED_NAME
    assert paper_id not in report.denominators[DENOM_STRICT].numerator_names


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- D-038 SS3. A state enumerated nowhere before it.
# ---------------------------------------------------------------------------
def test_new_an_unavailable_classification_keeps_the_bytes_and_names_them_honestly(
    tmp_path: Path,
) -> None:
    """**NEW ACCEPTANCE.** Pipeline passed, classification unavailable.

    Merge rule 7 forbids dropping the bytes and SS13 forbids the reserved name, so
    the one remaining honest name is ``pathway.review_required.pwml`` -- SS13
    defines it as "valid, needs review", which is exactly the state.
    """

    outcome = _leg(tmp_path, "absent", None)

    assert outcome.status == "pass", outcome.detail
    assert RELEASE_READY_NAME not in outcome.artifacts
    assert outcome.artifacts[REVIEW_REQUIRED_NAME].startswith(b"<pathway>")


def test_new_an_unavailable_classification_emits_no_release_status_and_says_so(
    tmp_path: Path,
) -> None:
    """**NEW ACCEPTANCE.** Fail loud, not silent: no invented key, one warning.

    Emitting a fabricated ``release_status`` would be the exporter answering a
    question the freeze never answered. ``report.py:860-861`` already renders an
    absent classification honestly, so the honest thing is to leave it absent and
    put the fact in ``warnings``, which travels all the way into the manifest row.
    """

    outcome = _leg(tmp_path, "absent_row", None)
    row = outcome.to_dict()

    assert "release_status" not in row
    assert row["pwml_artifact"] == REVIEW_REQUIRED_NAME
    assert any("release" in w and "unavailable" in w for w in row["warnings"]), row["warnings"]


def test_new_an_unrecognised_status_is_treated_as_unavailable(tmp_path: Path) -> None:
    """**NEW ACCEPTANCE.** A record naming a state the contract does not define is
    not a fourth state and must not be guessed at: it is exactly as unusable as a
    missing one, and is handled identically."""

    outcome = _leg(tmp_path, "unknown_state", _release("core_release_ready"))
    row = outcome.to_dict()

    assert RELEASE_READY_NAME not in outcome.artifacts
    assert REVIEW_REQUIRED_NAME in outcome.artifacts
    assert "release_status" not in row


def test_new_an_unavailable_classification_cannot_inflate_strict_ok(
    tmp_path: Path,
) -> None:
    """**NEW ACCEPTANCE.** D-038 SS3's last clause, stated as a test.

    The strict gate is an AFFIRMATIVE ``strict_acceptance_eligible is True``, so a
    row carrying no record is excluded by construction rather than by a rule that
    has to enumerate the ways a record can be missing. No new strict success
    without measured evidence.
    """

    paper_id = load_gold_set().strict_expected_ids[0]
    report = _score(tmp_path, [_manifest_row(paper_id, RELEASE_READY_NAME, None)])
    strict = report.denominators[DENOM_STRICT]

    assert strict.denominator == 1
    assert paper_id not in strict.numerator_names
    assert strict.numerator == 0


# ---------------------------------------------------------------------------
# The seam's own contract. NEW -- and NO base failure is offered for it.
# ---------------------------------------------------------------------------
def _direct_pwml(release: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """An export result exactly as the seam receives it in memory.

    The five JSON reports are here on purpose. The loop that writes them runs
    UNCONDITIONALLY, so a seam that reused that loop's target as its own return
    variable would answer with the last of THEM instead of the PWML filename --
    and including them is what makes the difference observable at all.
    """

    result: Dict[str, Any] = {
        "ok": True,
        "xml_bytes": b"<pathway><name>menaquinone</name></pathway>",
        "pwml_ir": {"pathway": {"name": "menaquinone"}},
        "pwml_ir_report": {"counts": {"reactions": 1}},
        "pwml_ir_validation": {"ok": True},
        "validation_report": {"issues": 0},
        "qa": {"ok": True},
        "required_gate_report": {"ok": True},
    }
    if release is not None:
        result["quarantine_report"] = {"release": release}
    return result


def test_new_the_seam_returns_the_name_it_actually_wrote() -> None:
    """**NEW, and deliberately NOT offered as a G9 base proof.**

    ``_add_strict_artifacts`` gained a return value in this card; at the base it
    returned ``None`` because there was no ``return`` statement. A test that
    "fails at the base" only because a value did not exist yet is symbol absence
    wearing a behavioural costume, and G9 says symbol absence is not proof. So
    this is labelled for what it is: a contract THIS card introduces, pinned so
    it cannot drift, not evidence that anything was corrected.

    It exists because the return was the one dimension of the new seam that
    nothing could observe -- both callers ignore or re-derive it -- and an
    unobservable documented contract is a trap for the next card on this seam.
    The name written and the name returned are asserted TOGETHER on every
    disposition, so they cannot diverge again.
    """

    for release, expected in (
        (_release("release_ready"), RELEASE_READY_NAME),
        (_release("review_required"), REVIEW_REQUIRED_NAME),
        (_release("diagnostic_only"), ""),
        (_release("core_release_ready"), REVIEW_REQUIRED_NAME),
        (None, REVIEW_REQUIRED_NAME),
    ):
        out: Dict[str, Any] = {}
        returned = driver._add_strict_artifacts({}, _direct_pwml(release), out)
        written = sorted(key for key in out if key.endswith(".pwml"))

        assert returned == expected, (release, returned)
        assert written == ([expected] if expected else []), (release, written)
        # PRESERVATION: the JSON reports still land under their own names. The
        # loop that writes them is what the return value collided with, so a fix
        # that freed the return by breaking them would be no fix at all.
        assert "pwml_ir.json" in out
        assert "pwml_required_field_gate_report.json" in out
        assert "pwml_validation_report.json" in out

    # No XML: nothing is named and nothing is claimed to have been.
    out = {}
    assert driver._add_strict_artifacts({}, {"ok": True, "pwml_ir": {"a": 1}}, out) == ""
    assert [key for key in out if key.endswith(".pwml")] == []
