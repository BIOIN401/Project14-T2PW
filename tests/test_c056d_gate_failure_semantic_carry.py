"""C-056d / F-055: the gate-failure path must CARRY the boundary's semantic verdict.

**G9 arm: CORRECTION.** This is not a new capability. The strict PASS path already
holds the property -- ``_finalize_pwml_export`` carries the frozen record instead of
re-deriving one -- and ``_release_status_row``'s own docstring already states the rule
the gate-failure path broke: *"nothing here re-classifies ... because a classification
produced after the freeze is a merge rule 8 defect, not a convenience."*

The pre-existing observable behaviour being corrected: a gate-failed leg's MANIFEST ROW
recorded ``semantic_evaluation: not_evaluated`` while the SAME leg's quarantine boundary
had computed ``failed`` and named the checks. Both records shipped, disagreeing, on both
gate-failed legs of ``runs_verify/2026-08-18_1328``.

Every test here drives symbols that exist unchanged on the base SHA
(``run_one``, ``driver._finalize_gate_failure``, ``RunOutcome``) and asserts on
ARTIFACT CONTENT, so the base failures are behavioural, never symbol absence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.batch import driver  # noqa: E402
from t2pw.batch.driver import MODE_STRICT, STRICT, RunOutcome, run_one  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    SEMANTIC_FAILED,
    SEMANTIC_INPUT_NOT_WIRED,
    SEMANTIC_NOT_EVALUATED,
    SEMANTIC_PASSED,
)
from t2pw.pipeline.strict_quarantine import QUARANTINE_REPORT_FILENAME  # noqa: E402
from test_batch_driver import (  # noqa: E402
    PAPER,
    _artifacts,
    _lit,
    _post_pipeline_body,
    _write_app,
    real_streamlit,  # noqa: F401 -- autouse fixture, re-exported on purpose
)

#: The committed evidence F-055 was registered from. Both legs are strict, both were
#: blocked by the gate channel, and both carry BOTH records on disk.
RUN_DIR = ROOT / "runs_verify" / "2026-08-18_1328"
GATE_FAILED_LEGS = ("PMC12452463", "PMC12096016")

_GATE_ERR = {"path": "/entities/protein_complexes/0/name", "reason": "Forbidden complex reference detected: x"}
_GATE_FAIL = {"status": "failed", "stage": "post_normalization_hard_gates", "errors": [_GATE_ERR]}


def _committed_release(paper_id: str) -> dict:
    """The boundary's frozen record for one committed gate-failed leg."""

    document = RUN_DIR / "papers" / paper_id / "strict" / QUARANTINE_REPORT_FILENAME
    report = json.loads(document.read_text(encoding="utf-8"))
    return report["release"]


def _shipped_row(paper_id: str) -> dict:
    """The ``release_status`` the same leg's manifest row actually shipped."""

    for line in (RUN_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("paper_id") == paper_id and row.get("mode") == MODE_STRICT:
            return row.get("release_status") or {}
    raise AssertionError(f"no strict manifest row for {paper_id}")


def _drive_gate_failure(tmp_path: Path, name: str, report: object) -> RunOutcome:
    """Run one gate-blocked strict leg whose boundary left ``report`` on disk.

    The quarantine report is written BY THE FIXTURE APP, to a real file named by a
    real ``quarantine_artifacts`` map, because that is the only hand-off
    ``_add_identity_artifacts`` accepts -- an allowlisted filename at a path whose
    basename matches. Nothing is injected into the driver.
    """

    document = tmp_path / name / "outputs" / QUARANTINE_REPORT_FILENAME
    extra = ""
    if report is not None:
        extra = (
            "    import json as _json, pathlib as _pl\n"
            f"    _p = _pl.Path({_lit(str(document))})\n"
            "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
            f"    _p.write_text({_lit(report if isinstance(report, str) else json.dumps(report))},"
            ' encoding="utf-8")\n'
        )
    artifacts = _artifacts(
        "pathwhiz",
        normalization_gate_failed=True,
        gate_fail_report=_GATE_FAIL,
        quarantine_artifacts={QUARANTINE_REPORT_FILENAME: str(document)},
    )
    app = _write_app(tmp_path, name, _post_pipeline_body(artifacts, extra=extra))
    return run_one(PAPER, STRICT, app_path=app, timeout=120.0, app_timeout=45.0)


# ---------------------------------------------------------------------------
# The evidence anchor. Fixture truth about committed files: green on both sides.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("paper_id", GATE_FAILED_LEGS)
def test_the_committed_run_shipped_two_release_records_that_disagree(paper_id: str) -> None:
    """F-055's evidence, read off disk. NOT the G9 proof -- history cannot change."""

    boundary = _committed_release(paper_id)
    shipped = _shipped_row(paper_id)
    assert boundary["semantic_evaluation"] == SEMANTIC_FAILED
    assert boundary["semantic_failed_checks"] == ["no_real_id_or_name_conflict"]
    assert shipped["semantic_evaluation"] == SEMANTIC_NOT_EVALUATED
    assert shipped["semantic_failed_checks"] == []
    # Same leg, same run, same status -- only the semantic verdict was discarded.
    assert boundary["status"] == shipped["status"] == DIAGNOSTIC_ONLY


# ---------------------------------------------------------------------------
# THE G9 PROOF. Behaviourally red on the base SHA.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("paper_id", GATE_FAILED_LEGS)
def test_the_manifest_row_now_carries_the_boundarys_semantic_verdict(
    tmp_path: Path, paper_id: str
) -> None:
    """RED AT BASE: the row records ``not_evaluated`` beside a boundary ``failed``.

    Driven end to end through ``run_one`` with the COMMITTED boundary record of a
    real gate-failed leg, so what is asserted is what an offline reader gets out of
    ``manifest.jsonl``.
    """

    boundary = _committed_release(paper_id)
    outcome = _drive_gate_failure(tmp_path, f"carry_{paper_id}", {"release": boundary})

    assert outcome.status == "fail"
    row = outcome.to_dict()["release_status"]
    assert row["semantic_evaluation"] == boundary["semantic_evaluation"] == SEMANTIC_FAILED
    assert row["semantic_failed_checks"] == boundary["semantic_failed_checks"]
    # The reason belongs to ``not_evaluated`` alone, so a real verdict clears it --
    # the second symptom F-055 registered: the unwired placeholder shipped beside a
    # runtime that HAD evaluated.
    assert row["semantic_not_evaluated_reason"] == ""
    # And the leg is still refused, on the driver's own technical observation.
    assert row["status"] == DIAGNOSTIC_ONLY
    assert row["strict_acceptance_eligible"] is False


def test_c056cs_evaluability_travels_with_the_verdict(tmp_path: Path) -> None:
    """RED AT BASE: the carrier arrived empty on every gate-failed leg."""

    evaluability = [
        {"check": "requested_pathway_anchors_present", "applicable": True, "inapplicable_reason": ""},
        {"check": "organism_compatible", "applicable": True, "inapplicable_reason": ""},
        {"check": "no_rejected_rag_reaction_reintroduced", "applicable": False,
         "inapplicable_reason": "no admission report exists at this seam"},
    ]
    release = {
        "status": DIAGNOSTIC_ONLY,
        "semantic_evaluation": SEMANTIC_FAILED,
        "semantic_failed_checks": ["no_real_id_or_name_conflict"],
        "semantic_check_evaluability": evaluability,
    }
    outcome = _drive_gate_failure(tmp_path, "evaluability", {"release": release})

    assert outcome.to_dict()["release_status"]["semantic_check_evaluability"] == evaluability


# ---------------------------------------------------------------------------
# Non-vacuity: attack the carry (shared execution block section 6).
# ---------------------------------------------------------------------------
def test_removing_the_boundary_report_puts_the_honest_not_evaluated_back(
    tmp_path: Path,
) -> None:
    """Delete the record the carry reads and the row must degrade, not invent.

    This is the arm that proves the assertions above are not vacuous: the same
    fixture, the same gate failure, with only the boundary's document withheld.
    """

    outcome = _drive_gate_failure(tmp_path, "no_report", None)

    assert QUARANTINE_REPORT_FILENAME not in outcome.artifacts
    row = outcome.to_dict()["release_status"]
    assert row["semantic_evaluation"] == SEMANTIC_NOT_EVALUATED
    assert row["semantic_not_evaluated_reason"] == SEMANTIC_INPUT_NOT_WIRED
    assert row["semantic_failed_checks"] == []


@pytest.mark.parametrize(
    "document",
    [
        "{not json at all",
        json.dumps({"release": {"semantic_evaluation": "probably_fine"}}),
        json.dumps({"release": "a string where a record should be"}),
        json.dumps({"no_release_key": True}),
        "",
    ],
    ids=["unparseable", "fourth_verdict_string", "release_not_a_record", "no_release", "empty"],
)
def test_an_uninterpretable_document_is_discarded_rather_than_trusted(
    tmp_path: Path, document: str
) -> None:
    """A record this code cannot interpret is the same predicament as having none."""

    outcome = _drive_gate_failure(tmp_path, "junk", document)

    row = outcome.to_dict()["release_status"]
    assert row["semantic_evaluation"] == SEMANTIC_NOT_EVALUATED
    assert row["semantic_not_evaluated_reason"] == SEMANTIC_INPUT_NOT_WIRED


# ---------------------------------------------------------------------------
# The carry can only ever REMOVE. It can never manufacture a strict success.
# ---------------------------------------------------------------------------
def test_the_carry_cannot_manufacture_a_strict_success(tmp_path: Path) -> None:
    """A boundary record claiming release_ready + passed must NOT lift this leg.

    The boundary's ``strict_gates_passed`` is its own closure/lock invariant set and
    is computed BEFORE the final pre-export Stage-3 suite runs, so a leg those later
    gates blocked can carry a boundary record that still reads ``release_ready``.
    Carrying the record WHOLESALE would put ``strict_acceptance_eligible: true`` on a
    leg that produced no PWML -- TRAP-1. Only the semantic fields travel.
    """

    release = {
        "status": "release_ready",
        "strict_gates_passed": True,
        "strict_acceptance_eligible": True,
        "semantic_evaluation": SEMANTIC_PASSED,
        "completeness": 1.0,
    }
    outcome = _drive_gate_failure(tmp_path, "no_inflation", {"release": release})

    row = outcome.to_dict()["release_status"]
    assert row["status"] == DIAGNOSTIC_ONLY
    assert row["strict_gates_passed"] is False
    assert row["strict_acceptance_eligible"] is False
    assert row["completeness"] is None
    # The verdict itself is still recorded honestly -- carried, never read.
    assert row["semantic_evaluation"] == SEMANTIC_PASSED
    assert "strict_technical_gates_blocked_export" in row["reasons"]


def test_the_seam_reads_only_the_artifact_set_it_is_handed() -> None:
    """``_gate_failure_semantic_carry`` is a pure read of ``outcome.artifacts``.

    Called directly, with no app and no disk, because the whole point of the source
    argument is that this seam has nothing else in scope.

    **NOT A G9 PROOF ARM.** This one errors on the BASE SHA with ``AttributeError``
    because the helper does not exist there, and symbol absence proves nothing. It
    is a unit test of the new private helper and is labelled so no reader counts it
    among the base-failing evidence. The behavioural proofs are the five above.
    """

    carried = driver._gate_failure_semantic_carry(
        {QUARANTINE_REPORT_FILENAME: json.dumps(
            {"release": {"semantic_evaluation": SEMANTIC_FAILED,
                         "semantic_failed_checks": ["organism_compatible"]}}
        )}
    )
    assert carried["semantic_evaluation"] == SEMANTIC_FAILED
    assert carried["semantic_failed_checks"] == ["organism_compatible"]
    assert driver._gate_failure_semantic_carry({}) == {}


# ---------------------------------------------------------------------------
# The second symptom: stale text naming a merged card as pending.
# ---------------------------------------------------------------------------
def test_the_unwired_reason_no_longer_names_a_merged_card_as_pending() -> None:
    """C-056a merged at ``93594aa``; the reason must state the CONDITION instead."""

    assert "C-056a" not in SEMANTIC_INPUT_NOT_WIRED
    assert "yet" not in SEMANTIC_INPUT_NOT_WIRED
    # What it must still say: a missing input, and neither of the two verdicts.
    assert "NOT a semantic failure" in SEMANTIC_INPUT_NOT_WIRED
    assert "NOT a pass" in SEMANTIC_INPUT_NOT_WIRED
